#!/usr/bin/env python3
"""
amd_init_helper.py — BEAGLE hybrid AMD backend init via tinygrad.

Mirrors nv_init_helper.py's role exactly, for the AMD vendor branch:

    python3 amd_init_helper.py <sock_fd> <dev_id> <output_json>

This script:
  1. Wraps the inherited FD in an APLRemotePCIDevice-compatible object
     (same InheritedFDPCIDevice pattern as nv_init_helper.py — the TinyGPU
     socket wire protocol is vendor-agnostic PCI BAR/config access; see
     GPUInterfaceTinyGPUHybrid.cpp's TGCmd enum).
  2. Boots the GPU using tinygrad's real AMDev (support/am/amdev.py) — a
     from-scratch AMDGPU driver (PSP/SMU/GFX/SDMA IP-block bring-up over
     raw PCI BAR/MMIO), the AMD analogue of NVDev.
  3. Creates ONE PM4 compute queue via AMDev.gfx.setup_ring() — this method
     already contains all the chip-specific MQD-building logic; we call it
     as-is rather than re-deriving it (same "reuse tinygrad's real driver
     code for bring-up, hand-roll only hot-path dispatch in C++" split
     already used for NV).
  4. Allocates VRAM for the ring, a small control page (rptr/wptr/completion
     signal), the EOP interrupt buffer setup_ring() itself requires, a code
     area (HSACO upload), and a data area (BEAGLE buffer pool).
  5. Writes a handoff JSON that C++ reads for hot-path dispatch: ring/
     doorbell/control-page VRAM offsets, plus the HDP-flush register's
     *resolved* dword index (see note below — it's chip/BIOS-programmed,
     not a fixed constant, so it must be read out here, not hardcoded in
     C++) and the concrete gfx1100 regCOMPUTE_* register addresses so C++
     doesn't need its own copy of tinygrad's register tables.

Register addresses embedded below were extracted directly from tinygrad's
own gc-11.0.0 (gfx1100) autogen table this session (AMDIP('gc', (11,0,0),
...) against tinygrad/runtime/autogen/am/navi_offsets.py) — not guessed, and
not reused from the older, unvalidated register constants in this repo's
dormant GPUInterfaceTinyGPU.cpp (those turned out to be for a different/older
GC generation and do not match gfx1100's real table).
"""

import sys, os, json, socket, time

_TINYGRAD_PATH = os.environ.get(
    "TINYGRAD_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "..", "..", "tinygrad"))
if not os.path.isdir(_TINYGRAD_PATH):
    print(f"amd_init_helper: cannot find tinygrad at {_TINYGRAD_PATH}\n"
          "Set TINYGRAD_PATH to the tinygrad checkout root.", file=sys.stderr)
    sys.exit(1)
sys.path.insert(0, os.path.abspath(_TINYGRAD_PATH))

from tinygrad.runtime.support.system import APLRemotePCIDevice, RemotePCIDevice
from tinygrad.runtime.support.am.amdev import AMDev
from tinygrad.runtime.support.memory import MemoryManager
from tinygrad.runtime.autogen.am import am

# ── macOS / TinyGPU socket safety patch ──────────────────────────────────────
# Same fix as nv_init_helper.py: TinyGPU.app's DriverKit extension crashes on
# BAR1/VRAM zero-writes larger than ~64 KB. AMMemoryManager.palloc(zero=True)
# is the same base MemoryManager.palloc() the NV patch targets (AMMemoryManager
# subclasses it without overriding palloc), so the identical patch protects
# the AMD boot path too -- confirmed by reading both classes this session,
# not assumed.
_PALLOC_ZERO_LIMIT = 64 << 10
_orig_palloc = MemoryManager.palloc
def _palloc_nozero_large(self, size, align=0x1000, zero=True, boot=False, ptable=False):
    if zero and size > _PALLOC_ZERO_LIMIT:
        zero = False
    return _orig_palloc(self, size, align, zero=zero, boot=boot, ptable=ptable)
MemoryManager.palloc = _palloc_nozero_large

# NOTE: unlike nv_init_helper.py, there is no AMD-specific timing/FLR patch
# here yet. NVDev needed patches for (a) PCIe-FLR-on-macOS-eGPU panics and
# (b) a SEC2-boot timing hang -- both discovered empirically on real NV
# hardware over this same socket. AMDev's own boot sequence uses an SMU
# "mode1_reset" (software reset) rather than a PCIe FLR when it detects a
# malformed prior state (amdev.py AMDev.__init__), which is a different,
# possibly-safer mechanism -- but this has never been exercised over the
# TinyGPU socket before, so treat this first run as diagnostic: if it panics
# or hangs, that tells us where an AMD-specific patch is needed, the same way
# each NV patch above was added reactively, not preemptively guessed.


class InheritedFDPCIDevice(APLRemotePCIDevice):
    """Identical to nv_init_helper.py's class of the same name -- the wire
    protocol/RPC framing is vendor-agnostic (TGC_MAP_BAR/CFG_READ/MMIO_*/
    etc.), so this is duplicated rather than imported to keep each helper
    script standalone, matching this project's existing style."""
    def __init__(self, sock_fd: int, dev_id: int = 0) -> None:
        inherited = socket.socket(fileno=os.dup(sock_fd))
        self.sock       = inherited
        self.pcibus     = "usb4"
        self.dev_id     = dev_id
        self.peer_group = "local"
        self.lock_fd    = None

    def reset(self) -> None:
        print("amd_init_helper: PCIe FLR suppressed (macOS eGPU safety)", file=sys.stderr)

    def bar_info(self, bar_idx: int):
        from tinygrad.runtime.support.system import RemoteCmd
        r0, r1 = RemotePCIDevice._rpc(
            self.sock, self.dev_id, RemoteCmd.MAP_BAR, bar=bar_idx)[:2]
        return (r0, r1)


class _TeeStream:
    """Same as nv_init_helper.py's _TeeStream."""
    def __init__(self, original, path):
        self._orig = original
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_SYNC, 0o644)
        self._log  = os.fdopen(fd, 'w', buffering=1)
    def write(self, s):
        self._orig.write(s); self._orig.flush()
        self._log.write(s);  self._log.flush()
        os.fsync(self._log.fileno())
    def flush(self):
        self._orig.flush(); self._log.flush()
    def fileno(self):
        return self._orig.fileno()


def main() -> None:
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <sock_fd> <dev_id> <output_json>", file=sys.stderr)
        sys.exit(1)

    sock_fd  = int(sys.argv[1])
    dev_id   = int(sys.argv[2])
    out_path = sys.argv[3]

    os.makedirs(os.path.expanduser("~/Library/Logs"), exist_ok=True)
    _log_path = os.path.expanduser("~/Library/Logs/amd_init_helper.log")
    sys.stderr = _TeeStream(sys.stderr, _log_path)
    print(f"amd_init_helper: logging to {_log_path}", file=sys.stderr)

    import traceback as _tb
    try:
        _main_impl(sock_fd, dev_id, out_path)
    except Exception:
        _tb.print_exc(file=sys.stderr)
        sys.exit(1)


def _main_impl(sock_fd: int, dev_id: int, out_path: str) -> None:
    def _step(msg: str) -> None:
        print(f"amd_init_helper: {msg}", file=sys.stderr, flush=True)

    _step("connecting via inherited socket FD …")
    pci_dev = InheritedFDPCIDevice(sock_fd, dev_id)

    # ── 1. Full GPU boot via AMDev ────────────────────────────────────────────
    _step("AMDev boot (PSP/SMU/GFX/SDMA bring-up) …")
    adev = AMDev(pci_dev)
    gfxver = adev.ip_ver[am.GC_HWIP]           # e.g. (11, 0, 0)
    gpu_arch = "gfx%d%x%x" % gfxver            # e.g. "gfx1100"
    _step(f"AMDev boot complete — arch={gpu_arch} vram={adev.vram_size>>20} MB")

    if gfxver[0] != 11:
        _step(f"WARNING: this backend's hardcoded regCOMPUTE_* addresses were "
              f"extracted for GC 11.0.0 (gfx1100); this chip reports GC "
              f"{gfxver} ({gpu_arch}) — dispatch will likely need re-checking.")

    # ── 2. Allocate VRAM: ring, control page, EOP buffer, code area, data area ──
    RING_SZ      = 1 << 20                                  # 1 MB PM4 ring
    EOP_SZ       = 0x1000                                   # HW-internal EOP interrupt buffer (setup_ring requirement, opaque to us)
    CTRL_SZ      = 0x1000                                   # rptr(8B) + wptr(8B) + completion-signal(8B), rest unused
    CODE_BUF_SZ  = int(os.environ.get("BEAGLE_AMD_CODE_MB", "64")) << 20
    _max_mb      = int(os.environ.get("BEAGLE_AMD_DATA_MB", "64"))
    DATA_POOL_SZ = ((_max_mb << 20) + (2 << 20) - 1) & ~((2 << 20) - 1)

    _step(f"allocating queue + pool resources (data_pool={DATA_POOL_SZ>>20} MB, "
          f"code={CODE_BUF_SZ>>20} MB) …")
    ring_area = adev.mm.valloc(RING_SZ, contiguous=True)
    eop_area  = adev.mm.valloc(EOP_SZ,  contiguous=True)
    ctrl_area = adev.mm.valloc(CTRL_SZ, contiguous=True)
    code_area = adev.mm.valloc(CODE_BUF_SZ,  contiguous=True)
    data_area = adev.mm.valloc(DATA_POOL_SZ, contiguous=True)
    _step("valloc complete")

    ring_vram = ring_area.paddrs[0][0]
    eop_vram  = eop_area.paddrs[0][0]
    ctrl_vram = ctrl_area.paddrs[0][0]
    code_vram = code_area.paddrs[0][0]
    data_vram = data_area.paddrs[0][0]

    rptr_va   = ctrl_area.va_addr + 0x0     # HW periodically writes its current read-pointer here
    wptr_va   = ctrl_area.va_addr + 0x8     # we write the new write-pointer here before ringing the doorbell
    signal_va = ctrl_area.va_addr + 0x10    # our own completion semaphore: RELEASE_MEM in the dispatch packet writes here
    signal_vram = ctrl_vram + 0x10

    # ── 3. Create the compute queue (all chip-specific MQD logic lives inside
    #      AM_GFX.setup_ring() -- called as-is, not re-derived). idx=0, aql=False
    #      picks the plain PM4 (non-AQL) queue type, matching tinygrad's own
    #      default for a single-XCC (consumer RDNA3) device (AMDDevice.__init__:
    #      is_aql = getenv("AMD_AQL", int(xccs > 1)), false when xccs==1).
    _step("gfx.setup_ring() — creating PM4 compute queue …")
    doorbell_index = adev.gfx.setup_ring(
        ring_area.va_addr, RING_SZ, rptr_va, wptr_va,
        eop_area.va_addr, EOP_SZ, 0, False)
    doorbell_byte_off = doorbell_index * 8
    _step(f"compute queue ready — doorbell_index={doorbell_index} "
          f"(BAR2 byte offset {doorbell_byte_off})")

    # ── 4. Resolve the HDP-flush trigger address. This is a BIOS/PCI-config-
    #      programmed *indirection* (regBIF_BX0_REMAP_HDP_MEM_FLUSH_CNTL holds
    #      the byte address of the real trigger register, not a fixed offset),
    #      so it must be read out here rather than hardcoded in C++ -- see
    #      AM_GMC.flush_hdp() in tinygrad's support/am/ip.py, which this
    #      mirrors. C++ must write 0 to this dword index (BAR5, MMIO/register
    #      aperture) before every doorbell ring so the GPU sees prior VRAM
    #      writes (ring entries, kernel args) -- same role as NV's implicit
    #      channel-setup ordering, made explicit here since AMD's queue
    #      submission is otherwise pure doorbell + polled wptr, no other
    #      ordering guarantee.
    hdp_flush_dword_idx = adev.reg("regBIF_BX0_REMAP_HDP_MEM_FLUSH_CNTL").read() // 4
    _step(f"HDP flush trigger resolved — dword index 0x{hdp_flush_dword_idx:x}")

    # ── 4b. Chip shader-array properties + a pre-allocated scratch (private
    #       segment) buffer. Every OpenCL kernel gets *some* private/scratch
    #       segment even when it looks trivial (measured empirically this
    #       session: 84 bytes/thread for a one-line kernel with no local
    #       arrays), so this can't be skipped. Formula mirrors tinygrad's own
    #       AMDDevice._ensure_has_local_memory() (ops_amd.py) exactly, with
    #       cu_cnt/se_cnt/max_slots_scratch_cu computed the same way
    #       PCIIface._compute_props() does -- both read here directly from
    #       adev.gc_info rather than re-deriving via the PCIIface class,
    #       which (per this backend's split, see GPUInterfaceTinyGPUHybrid.cpp)
    #       we deliberately don't construct.
    gi = adev.gc_info
    if gi.header.version_major == 2:
        cu_per_sa, max_sh_per_se = gi.gc_num_cu_per_sh, gi.gc_num_sh_per_se
    else:
        cu_per_sa, max_sh_per_se = 2 * (gi.gc_num_wgp0_per_sa + gi.gc_num_wgp1_per_sa), gi.gc_num_sa_per_se
    se_cnt = gi.gc_num_se
    cu_cnt = cu_per_sa * max_sh_per_se * se_cnt
    max_slots_scratch_cu = gi.gc_max_scratch_slots_per_cu

    # Pre-allocate scratch for up to PRIVATE_SEGMENT_CAP bytes/thread —
    # generous headroom over the ~84-256 byte range these kernels actually
    # need (measured this session); amd_compile_helper.py checks every real
    # kernel's private_segment_size against this cap and fails loudly rather
    # than silently overflow if it's ever wrong.
    PRIVATE_SEGMENT_CAP = 1024
    lanes_per_wave = 64
    mem_alignment_size = 256
    size_per_thread = (PRIVATE_SEGMENT_CAP + (mem_alignment_size // lanes_per_wave) - 1) \
                       // (mem_alignment_size // lanes_per_wave) * (mem_alignment_size // lanes_per_wave)
    scratch_sz = size_per_thread * lanes_per_wave * max_slots_scratch_cu * cu_cnt
    _step(f"allocating scratch buffer ({scratch_sz>>10} KB, cap={PRIVATE_SEGMENT_CAP} B/thread, "
          f"cu_cnt={cu_cnt} se_cnt={se_cnt} max_slots_scratch_cu={max_slots_scratch_cu}) …")
    scratch_area = adev.mm.valloc(scratch_sz, contiguous=True)

    # ── 5. Write handoff JSON ─────────────────────────────────────────────────
    state = {
        "gpu_arch":            gpu_arch,
        "gc_hwip_version":     list(gfxver),
        "vram_size":           adev.vram_size,
        "doorbell_byte_off":   doorbell_byte_off,
        "hdp_flush_dword_idx": hdp_flush_dword_idx,
        # Chip shader-array properties, for amd_compile_helper.py's TMPRING_SIZE calc.
        "cu_cnt":              cu_cnt,
        "se_cnt":              se_cnt,
        "max_slots_scratch_cu": max_slots_scratch_cu,
        # Scratch (private segment) buffer — see §4b above.
        "scratch_vram":        scratch_area.paddrs[0][0],
        "scratch_gpu_va":      scratch_area.va_addr,
        "scratch_sz":          scratch_sz,
        "scratch_cap_bytes_per_thread": PRIVATE_SEGMENT_CAP,
        # PM4 ring — C++ writes packets via tg_bulk_write(bar=0, ring_vram + wptr_bytes)
        "ring_vram":           ring_vram,
        "ring_gpu_va":         ring_area.va_addr,
        "ring_sz":             RING_SZ,
        # Control page (bar=0, VRAM)
        "rptr_vram":           ctrl_vram + 0x0,
        "wptr_vram":           ctrl_vram + 0x8,
        "signal_vram":         signal_vram,
        "signal_gpu_va":       signal_va,
        # Code area — compiled HSACO(s); gpu_va = code_gpu_va + offset
        "code_vram":           code_vram,
        "code_gpu_va":         code_area.va_addr,
        "code_sz":             CODE_BUF_SZ,
        # Data pool — AllocateMemory bump-allocates here
        "data_vram":           data_vram,
        "data_gpu_va":         data_area.va_addr,
        "data_sz":             DATA_POOL_SZ,
        "init_helper_pid":     os.getpid(),
    }
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(state, f, indent=2)
    os.rename(tmp_path, out_path)

    _step("handoff written — waiting for SIGTERM to finalize GPU")
    _step(f"doorbell_off=0x{doorbell_byte_off:x} ring_vram=0x{ring_vram:x} "
          f"data_pool={DATA_POOL_SZ>>20} MB")

    import signal as _signal
    _done = [False]
    def _sigterm(sig, frame): _done[0] = True
    _signal.signal(_signal.SIGTERM, _sigterm)
    _signal.signal(_signal.SIGINT, _sigterm)
    while not _done[0]:
        time.sleep(0.5)

    print("amd_init_helper: signal received — calling adev.fini()", file=sys.stderr, flush=True)
    adev.fini()
    print("amd_init_helper: fini complete — exiting cleanly", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
