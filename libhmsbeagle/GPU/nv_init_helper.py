#!/usr/bin/env python3
"""
nv_init_helper.py — BEAGLE hybrid NV backend init via tinygrad.

The C++ parent (GPUInterfaceTinyGPUHybrid) opens the TinyGPU socket,
clears O_CLOEXEC so the FD survives exec(), then spawns this script:

    python3 nv_init_helper.py <sock_fd> <dev_id> <output_json>

This script:
  1. Wraps the inherited FD in an APLRemotePCIDevice-compatible object.
  2. Boots the GPU using tinygrad's NVDev (GSP + golden image).
  3. Allocates GPFIFO ring, EOP, command queue, code buffer, and data
     pool in VRAM — and maps them in the GPU page tables.
  4. Creates a user RM client hierarchy and compute channel (non-priv
     path, exactly as tinygrad's PCIIface/NVDevice does in production).
  5. Writes a handoff JSON that C++ reads for hot-path dispatch.

The socket FD remains open in the C++ parent after this process exits,
so the TinyGPU server keeps the GPU state (page tables, RM objects) alive.

The data pool and code buffer pre-allocate VRAM that C++ will use for
AllocateMemory() and GetFunction().  GPU VAs for these regions are fixed
at init time so C++ can compute gpu_va = region_gpu_va + offset directly.
"""

import sys, os, json, ctypes, socket, struct, time

# Locate tinygrad — prefer env var, fall back to sibling checkout.
_TINYGRAD_PATH = os.environ.get(
    "TINYGRAD_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "..", "..", "tinygrad"))
if not os.path.isdir(_TINYGRAD_PATH):
    print(f"nv_init_helper: cannot find tinygrad at {_TINYGRAD_PATH}\n"
          "Set TINYGRAD_PATH to the tinygrad checkout root.", file=sys.stderr)
    sys.exit(1)
sys.path.insert(0, os.path.abspath(_TINYGRAD_PATH))

from tinygrad.runtime.support.system import APLRemotePCIDevice, RemotePCIDevice
from tinygrad.runtime.support.nv.nvdev import NVDev, NVMemoryManager
from tinygrad.runtime.support.nv.ip import NV_FLCN, NV_FLCN_COT, NV_GSP
from tinygrad.runtime.support.memory import MemoryManager
from tinygrad.runtime.autogen import nv_570 as nv_gpu

# ── macOS / TinyGPU socket safety patches ────────────────────────────────────

# 1. Suppress PCIe FLR and its post-reset polling.
# NVDev._early_ip_init() calls pci_dev.reset() when WPR2 is already up
# (GPU was previously initialized).  On macOS, a PCIe FLR via USB4 causes
# a kernel panic.  We override reset() on InheritedFDPCIDevice below and
# suppress the companion wait_for_reset() which would otherwise block forever.
NV_FLCN.wait_for_reset     = lambda self: None
NV_FLCN_COT.wait_for_reset = lambda self: None

# 2. Skip VRAM zeroing for large allocations only.
# MemoryManager.palloc(zero=True) issues a full-size BAR1 write via _bulk_write.
# TinyGPU.app's DriverKit extension crashes when the write payload exceeds its
# internal buffer limit (~64 KB).  Small zeroes (≤ 64 KB) are safe and required
# for GPU channel state (RAMFC, method buffer) to be clean.  Large post-init
# allocations (GPFIFO ring, code buffer, data pool) don't need zeroing.
_PALLOC_ZERO_LIMIT = 64 << 10   # 64 KB — below this, zeroing is safe via BAR1
_orig_palloc = MemoryManager.palloc
def _palloc_nozero_large(self, size, align=0x1000, zero=True, boot=False, ptable=False):
    if zero and size > _PALLOC_ZERO_LIMIT:
        zero = False
    return _orig_palloc(self, size, align, zero=zero, boot=boot, ptable=ptable)
MemoryManager.palloc = _palloc_nozero_large

# 3. Sleep after SEC2 start (only during gsp.init_hw) to let the GC6 BSI
# power domain stabilize before we poll NV_PGC6_BSI_SECURE_SCRATCH_14.
#
# Root cause of the silent hang: run_cpu_seq op 0x8 calls start_cpu(sec2)
# then immediately polls NV_PGC6_BSI_SECURE_SCRATCH_14 via BAR0 over the
# TinyGPU socket.  While SEC2 is booting, the GC6 BSI register domain is
# briefly power-gated; TinyGPU.app's PCIe read hangs, so sock.recv() never
# returns, and the wait_cond loop can never advance its timeout check.
#
# Fix: after start_cpu(sec2) (base == 0x00840000) inside gsp.init_hw(),
# sleep 20 s to allow SEC2 to complete and the register to become accessible.
# SEC2 typically boots in 2–5 s; 20 s is conservative but safe.
_SEC2_BASE = 0x00840000
_in_gsp_init = [False]

_orig_gsp_init_hw = NV_GSP.init_hw
def _patched_gsp_init_hw(self):
    _in_gsp_init[0] = True
    try:
        return _orig_gsp_init_hw(self)
    finally:
        _in_gsp_init[0] = False
NV_GSP.init_hw = _patched_gsp_init_hw

_orig_start_cpu = NV_FLCN.start_cpu
def _patched_start_cpu(self, base):
    _orig_start_cpu(self, base)
    if base == _SEC2_BASE and _in_gsp_init[0]:
        print("nv_init_helper: SEC2 started inside gsp.init_hw — sleeping 20 s "
              "for GC6 BSI domain to stabilise …", file=sys.stderr, flush=True)
        time.sleep(20)
NV_FLCN.start_cpu = _patched_start_cpu


# ─────────────────────────────────────────────────────────────────────────────
# Inherited-FD device: APLRemotePCIDevice without its own socket/lock setup.
# ─────────────────────────────────────────────────────────────────────────────

class InheritedFDPCIDevice(APLRemotePCIDevice):
    """
    APLRemotePCIDevice variant that wraps a pre-opened socket FD.

    The C++ parent holds the canonical socket reference; we dup() the FD so
    Python's GC closing this socket does not affect the C++ copy.

    APLRemotePCIDevice.alloc_sysmem uses MAP_SYSMEM_FD with SCM_RIGHTS to
    transfer a sysmem FD — this matches BEAGLE's tgpu_rpc_fd() exactly.
    """
    def __init__(self, sock_fd: int, dev_id: int = 0) -> None:
        # dup() so Python's GC does not close the C++ parent's socket.
        inherited = socket.socket(fileno=os.dup(sock_fd))
        # Bypass APLRemotePCIDevice.__init__ and RemotePCIDevice.__init__:
        # they open a new connection and acquire a file lock we don't need.
        self.sock       = inherited
        self.pcibus     = "usb4"
        self.dev_id     = dev_id
        self.peer_group = "local"
        self.lock_fd    = None

    def reset(self) -> None:
        # PCIe FLR is fatal on macOS eGPU (USB4): the kernel panics when the
        # device disappears.  NVDev will reset the Falcon MCUs via MMIO instead,
        # which is sufficient for a clean GSP reboot.
        print("nv_init_helper: PCIe FLR suppressed (macOS eGPU safety)", file=sys.stderr)

    # bar_info is @functools.cache on RemotePCIDevice; we re-implement without
    # cache since we're bypassing __init__ (which sets self.dev_id correctly).
    def bar_info(self, bar_idx: int):
        from tinygrad.runtime.support.system import RemoteCmd
        r0, r1 = RemotePCIDevice._rpc(
            self.sock, self.dev_id, RemoteCmd.MAP_BAR, bar=bar_idx)[:2]
        return (r0, r1)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sass_version(chip_name: str) -> int:
    """QMD SASS_VERSION byte = (SM_major << 4) | SM_minor.
    GA102 = SM 8.6 → 0x86, AD102 = SM 8.9 → 0x89, GB2xx = SM 10.0 → 0xa0.
    """
    major, minor = {'GA1': (8, 6), 'AD1': (8, 9), 'GB2': (10, 0)}.get(chip_name[:3], (8, 6))
    return (major << 4) | minor


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

class _TeeStream:
    """Write to both a file (O_SYNC, survives kernel panic) and the original stream."""
    def __init__(self, original, path):
        self._orig = original
        # O_SYNC: each write goes to disk before returning — survives kernel panic.
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
        print(f"Usage: {sys.argv[0]} <sock_fd> <dev_id> <output_json>",
              file=sys.stderr)
        sys.exit(1)

    sock_fd  = int(sys.argv[1])
    dev_id   = int(sys.argv[2])
    out_path = sys.argv[3]

    # ~/Library/Logs is APFS-journaled and persists across kernel panics.
    os.makedirs(os.path.expanduser("~/Library/Logs"), exist_ok=True)
    _log_path = os.path.expanduser("~/Library/Logs/nv_init_helper.log")
    sys.stderr = _TeeStream(sys.stderr, _log_path)
    print(f"nv_init_helper: logging to {_log_path}", file=sys.stderr)

    import traceback as _tb
    try:
        _main_impl(sock_fd, dev_id, out_path)
    except Exception:
        _tb.print_exc(file=sys.stderr)
        sys.exit(1)


def _main_impl(sock_fd: int, dev_id: int, out_path: str) -> None:
    print("nv_init_helper: connecting via inherited socket FD …", file=sys.stderr)
    pci_dev = InheritedFDPCIDevice(sock_fd, dev_id)

    def _step(msg: str) -> None:
        print(f"nv_init_helper: {msg}", file=sys.stderr, flush=True)

    # ── 0. WPR2 warm-start guard (direct BAR0 read, no NVDev required) ───────
    # NV_PFB_PRI_MMU_WPR2_ADDR_HI is at BAR0 byte-offset 0x1FA828 (index 518666).
    # If it's non-zero the GPU is still initialised from a prior session;
    # a full re-init without PCIe FLR (suppressed on macOS) will hang.
    # ACTION: unplug and replug the Thunderbolt cable to power-cycle the eGPU.
    _step("WPR2 pre-check (BAR0 direct read)")
    _quick_mmio = pci_dev.map_bar(0, fmt='I')
    _NV_PFB_PRI_MMU_WPR2_ADDR_HI_IDX = 2074664 // 4    # byte offset 0x1FA828; index 518666
    _wpr2_hi = _quick_mmio[_NV_PFB_PRI_MMU_WPR2_ADDR_HI_IDX]
    if _wpr2_hi != 0:
        print(f"\nnv_init_helper: WARM-START DETECTED (WPR2_HI=0x{_wpr2_hi:08x})\n"
              "The eGPU is still initialised from a prior session and cannot be\n"
              "re-initialised without a PCIe FLR (suppressed on macOS to prevent\n"
              "kernel panic).  ACTION REQUIRED: unplug and replug the Thunderbolt\n"
              "cable to power-cycle the eGPU, then retry.", file=sys.stderr)
        sys.exit(1)
    _step("WPR2 = 0 — cold start confirmed")

    # ── 1. Full GPU boot via NVDev ────────────────────────────────────────────
    # NVDev(pci_dev) runs the complete init sequence:
    #   _early_ip_init → _early_mmu_init → flcn.init_sw/hw → gsp.init_sw/hw
    # The patched start_cpu above sleeps 20 s after SEC2 starts (inside
    # gsp.init_hw) so the GC6 BSI register is accessible before we poll it.
    # NVDev._early_ip_init suppresses FLR via our overridden reset() method.
    _step("NVDev boot (this includes ~20 s SEC2 sleep — please wait) …")
    nvdev = NVDev(pci_dev)
    _step(f"NVDev boot complete — chip={nvdev.chip_name} vram={nvdev.vram_size>>20} MB")

    gsp          = nvdev.gsp
    bar1_paddr, _bar1_sz = pci_dev.bar_info(1)

    # ── 2. Allocate VRAM for channel + dispatch resources ────────────────────
    GPFIFO_ENTRIES = 0x10000          # 64K entries × 8 B = 512 KB
    RING_SZ        = GPFIFO_ENTRIES * 8
    AREA_SZ        = 3 << 20         # 3 MB: ring + USERD page + padding
    CMDQ_SZ        = 2 << 20         # 2 MB command queue (QMD + method pkts)
    EOP_SZ         = 0x1000          # 4 KB EOP semaphore page
    GPPUT_OFF      = 140             # GPPut byte-offset within USERD page

    # Code buffer: stores compiled cubin(s); defaults to 64 MB.
    CODE_BUF_SZ = int(os.environ.get("BEAGLE_NV_CODE_MB", "64")) << 20

    # Data pool: capped at 64 MB for initial testing; expand with BEAGLE_NV_DATA_MB.
    _max_mb      = int(os.environ.get("BEAGLE_NV_DATA_MB", "64"))
    DATA_POOL_SZ = (_max_mb << 20)
    DATA_POOL_SZ = (DATA_POOL_SZ + (2 << 20) - 1) & ~((2 << 20) - 1)  # 2MB align

    print(f"nv_init_helper: allocating channel resources "
          f"(data_pool={DATA_POOL_SZ >> 20} MB, code={CODE_BUF_SZ >> 20} MB) …",
          file=sys.stderr)

    _step("valloc gpfifo_area (3 MB)")
    gpfifo_area   = nvdev.mm.valloc(AREA_SZ,        contiguous=True)
    _step("valloc notifier_area (4 KB)")
    notifier_area = nvdev.mm.valloc(0x1000,          contiguous=True)
    _step("valloc cmdq_area (2 MB contiguous)")
    cmdq_area     = nvdev.mm.valloc(CMDQ_SZ,         contiguous=True)
    _step("valloc eop_area (4 KB)")
    eop_area      = nvdev.mm.valloc(EOP_SZ,          contiguous=True)
    _step(f"valloc code_area ({CODE_BUF_SZ >> 20} MB contiguous)")
    code_area     = nvdev.mm.valloc(CODE_BUF_SZ,     contiguous=True)
    _step(f"valloc data_area ({DATA_POOL_SZ >> 20} MB contiguous)")
    data_area     = nvdev.mm.valloc(DATA_POOL_SZ,    contiguous=True)
    _step("valloc complete")

    gpfifo_paddr   = gpfifo_area.paddrs[0][0]
    notifier_paddr = notifier_area.paddrs[0][0]

    # valloc.paddrs[0][0] is already a VRAM-relative byte offset from palloc's
    # TLSFAllocator.  That is the same byte offset C++ passes to TinyGPU as the
    # BAR1 offset in tg_bulk_write/tg_bulk_read.  bar_info(1)[0] is the HOST
    # physical address of BAR1 and must NOT be subtracted here.
    gpfifo_vram    = gpfifo_paddr
    userd_vram     = gpfifo_vram + RING_SZ
    cmdq_vram      = cmdq_area.paddrs[0][0]
    eop_vram       = eop_area.paddrs[0][0]
    code_vram      = code_area.paddrs[0][0]
    data_vram      = data_area.paddrs[0][0]

    # ── 3. Non-priv RM hierarchy (mirrors PCIIface / NVDevice in ops_nv.py) ──
    #
    # Use user root 0xc1000000 (not priv_root 0xc1e00004).  rpc_rm_alloc in
    # ip.py auto-handles the non-priv path:
    #   • FERMI_VASPACE_A → auto rpc_set_page_directory (no COPY_SERVER_RESERVED_PDES)
    #   • compute_class   → auto promote_ctx for grctx_bufs [0,1,2]
    USER_ROOT = 0xc1000000

    _step("RM 1 — NV01_ROOT (user root 0xc1000000)")
    gsp.rpc_rm_alloc(0, nv_gpu.NV01_ROOT, nv_gpu.NV0000_ALLOC_PARAMETERS(), USER_ROOT)

    _step("RM 2 — NV01_DEVICE_0")
    user_device = gsp.rpc_rm_alloc(
        USER_ROOT, nv_gpu.NV01_DEVICE_0,
        nv_gpu.NV0080_ALLOC_PARAMETERS(
            deviceId=0, hClientShare=USER_ROOT,
            vaMode=nv_gpu.NV_DEVICE_ALLOCATION_VAMODE_OPTIONAL_MULTIPLE_VASPACES),
        USER_ROOT)

    _step("RM 3 — NV20_SUBDEVICE_0")
    user_subdev = gsp.rpc_rm_alloc(
        user_device, nv_gpu.NV20_SUBDEVICE_0,
        nv_gpu.NV2080_ALLOC_PARAMETERS(), USER_ROOT)

    _step("RM 4 — FERMI_VASPACE_A (auto rpc_set_page_directory)")
    # IS_EXTERNALLY_OWNED: page tables managed by us (nvdev.mm).
    # ENABLE_PAGE_FAULTING: required for externally-owned vaspaces.
    # rpc_rm_alloc auto-calls rpc_set_page_directory(pdir=root_page_table.paddr).
    user_vaspace = gsp.rpc_rm_alloc(
        user_device, nv_gpu.FERMI_VASPACE_A,
        nv_gpu.NV_VASPACE_ALLOCATION_PARAMETERS(
            vaBase=0x1000, vaSize=0x1fffffb000000,
            flags=(nv_gpu.NV_VASPACE_ALLOCATION_FLAGS_ENABLE_PAGE_FAULTING |
                   nv_gpu.NV_VASPACE_ALLOCATION_FLAGS_IS_EXTERNALLY_OWNED)),
        USER_ROOT)

    _step("RM 5 — KEPLER_CHANNEL_GROUP_A (TSG)")
    user_cg = gsp.rpc_rm_alloc(
        user_device, nv_gpu.KEPLER_CHANNEL_GROUP_A,
        nv_gpu.NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS(
            engineType=nv_gpu.NV2080_ENGINE_TYPE_GRAPHICS),
        USER_ROOT)

    _step("RM 6 — FERMI_CONTEXT_SHARE_A")
    user_ctxshare = gsp.rpc_rm_alloc(
        user_cg, nv_gpu.FERMI_CONTEXT_SHARE_A,
        nv_gpu.NV_CTXSHARE_ALLOCATION_PARAMETERS(
            hVASpace=user_vaspace,
            flags=nv_gpu.NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_ASYNC),
        USER_ROOT)

    _step("RM 7 — gpfifo_class (auto ramfcMem + userdMem)")
    # hObjectError non-zero → rpc_rm_alloc auto-fills userdMem from
    # hUserdMemory[0] + userdOffset[0] (= gpfifo_paddr + RING_SZ).
    # hObjectBuffer = gpfifo_paddr (VRAM physical address of ring buffer).
    gpfifo_params = nv_gpu.NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS(
        gpFifoOffset  = gpfifo_area.va_addr,
        gpFifoEntries = GPFIFO_ENTRIES,
        hContextShare = user_ctxshare,
        hObjectError  = notifier_paddr,          # non-zero → triggers auto-fill
        hObjectBuffer = gpfifo_paddr,
        hUserdMemory  = (ctypes.c_uint32 * 8)(gpfifo_paddr),
        userdOffset   = (ctypes.c_uint64 * 8)(RING_SZ),
        engineType    = 0)
    user_gpfifo = gsp.rpc_rm_alloc(user_cg, gsp.gpfifo_class, gpfifo_params, USER_ROOT)
    _step(f"  gpfifo handle=0x{user_gpfifo:08x}")

    _step("RM 8 — compute_class (auto promote_ctx for grctx_bufs [0,1,2])")
    gsp.rpc_rm_alloc(user_gpfifo, gsp.compute_class, None, USER_ROOT)

    _step("RM 9 — dma_class")
    gsp.rpc_rm_alloc(user_gpfifo, gsp.dma_class, None, USER_ROOT)

    _step("RM 10 — schedule TSG")
    gsp.rpc_rm_control(
        user_cg, nv_gpu.NVA06C_CTRL_CMD_GPFIFO_SCHEDULE,
        nv_gpu.NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS(bEnable=1),
        USER_ROOT)

    _step("RM 11 — work submit token")
    ws = gsp.rpc_rm_control(
        user_gpfifo, nv_gpu.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN,
        nv_gpu.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS(
            workSubmitToken=-1),
        USER_ROOT)
    work_token = ws.workSubmitToken

    # ── 4. Write handoff JSON ─────────────────────────────────────────────────
    state = {
        # Channel dispatch
        "work_token":       work_token,
        # GPFIFO ring — C++ writes entries via nv_vram_wr(bar1, gpfifo_vram)
        "gpfifo_vram":      gpfifo_vram,
        "gpfifo_gpu_va":    gpfifo_area.va_addr,
        "gpfifo_entries":   GPFIFO_ENTRIES,
        # USERD page — C++ writes GPPut via nv_vram_wr(bar1, userd_vram + gpput_off)
        "userd_vram":       userd_vram,
        "gpput_off":        GPPUT_OFF,
        # EOP semaphore — GPU writes to eop_gpu_va; CPU polls via nv_vram_rd
        "eop_vram":         eop_vram,
        "eop_gpu_va":       eop_area.va_addr,
        # Command queue — QMD + method packets; gpu_va = cmdq_gpu_va + offset
        "cmdq_vram":        cmdq_vram,
        "cmdq_gpu_va":      cmdq_area.va_addr,
        "cmdq_sz":          CMDQ_SZ,
        # Code buffer — compiled cubin(s); gpu_va = code_gpu_va + offset
        "code_vram":        code_vram,
        "code_gpu_va":      code_area.va_addr,
        "code_sz":          CODE_BUF_SZ,
        # Data pool — AllocateMemory bump-allocates here; gpu_va = data_gpu_va + offset
        "data_vram":        data_vram,
        "data_gpu_va":      data_area.va_addr,
        "data_sz":          DATA_POOL_SZ,
        # Memory manager
        "mm_vram_pa_base":  bar1_paddr,
        "vram_size":        nvdev.vram_size,
        # Architecture
        "compute_class":    gsp.compute_class,
        "dma_class":        gsp.dma_class,
        "sass_version":     _sass_version(nvdev.chip_name),
        "chip_name":        nvdev.chip_name,
        # GSP RPC queue — needed by C++ gsp_unloading_guest_driver() at shutdown.
        # On macOS/TinyGPU, alloc_sysmem uses MAP_SYSMEM_FD (mmap path), so
        # cmd_q_view is a plain MMIOInterface without 'residx'.  In that case we
        # store 0 and the C++ RPC path is skipped (keeper fallback is used instead).
        "gsp_sysmem_handle": getattr(gsp.cmd_q_view, 'residx', 0),
        "init_helper_pid":  os.getpid(),
    }

    # Atomic write: .tmp then rename so C++ polling sees a complete file.
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(state, f, indent=2)
    os.rename(tmp_path, out_path)

    print(f"nv_init_helper: handoff written — waiting for SIGTERM to finalize GPU",
          file=sys.stderr, flush=True)
    print(f"  work_token=0x{work_token:08x} compute_class=0x{gsp.compute_class:x} "
          f"chip={nvdev.chip_name}", file=sys.stderr)
    print(f"  gpfifo_vram=0x{gpfifo_vram:x}  eop_vram=0x{eop_vram:x}  "
          f"data_pool={DATA_POOL_SZ>>20} MB", file=sys.stderr)

    # Stay alive until C++ destructor signals us via SIGTERM.
    import signal as _signal
    _done = [False]
    def _sigterm(sig, frame): _done[0] = True
    _signal.signal(_signal.SIGTERM, _sigterm)
    while not _done[0]:
        time.sleep(0.5)

    print("nv_init_helper: SIGTERM received — calling nvdev.fini()", file=sys.stderr, flush=True)
    nvdev.fini()
    print("nv_init_helper: fini complete — exiting cleanly", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
