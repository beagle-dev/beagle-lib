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
  4. Creates a user RM client hierarchy and compute channel.
  5. Writes a handoff JSON that C++ reads for hot-path dispatch.

The socket FD remains open in the C++ parent after this process exits,
so the TinyGPU server keeps the GPU state (page tables, RM objects) alive.

The data pool and code buffer pre-allocate VRAM that C++ will use for
AllocateMemory() and GetFunction().  GPU VAs for these regions are fixed
at init time so C++ can compute gpu_va = region_gpu_va + offset directly.
"""

import sys, os, json, ctypes, socket, struct

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
from tinygrad.runtime.support.nv.ip import NV_FLCN, NV_FLCN_COT
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
        for buf_type in (socket.SO_SNDBUF, socket.SO_RCVBUF):
            self.sock.setsockopt(socket.SOL_SOCKET, buf_type, 64 << 20)

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
    """Compute SASS version byte from chip name prefix (GA1/AD1/GB2)."""
    sm = {'GA1': 0x860, 'AD1': 0x890, 'GB2': 0xa04}.get(chip_name[:3], 0x860)
    return ((sm & 0xf00) >> 4) | (sm & 0xf)


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

    # ── 1. Boot GPU via tinygrad (stepped for crash diagnosis) ───────────────
    # Each step is flushed so the last visible line tells us where a panic hit.
    import types as _types

    def _step(msg: str) -> None:
        print(f"nv_init_helper: {msg}", file=sys.stderr, flush=True)

    _step("step 1/6 — NVDev.__new__ + map_bar(0)")
    nvdev = NVDev.__new__(NVDev)
    nvdev.pci_dev = pci_dev
    nvdev.devfmt  = pci_dev.pcibus
    nvdev.mmio    = pci_dev.map_bar(0, fmt='I')

    _step("step 2/6 — _early_ip_init (WPR2 check; PCIe FLR suppressed)")
    nvdev.smi_dev, nvdev.is_booting, nvdev.is_err_state = False, True, False
    nvdev._early_ip_init()

    _step("step 3/6 — _early_mmu_init (BAR1 map)")
    nvdev._early_mmu_init()
    nvdev.is_booting = False

    _step("step 4/6 — flcn.init_sw")
    nvdev.flcn.init_sw()
    _step("step 5/6 — gsp.init_sw")
    nvdev.gsp.init_sw()

    # Monkey-patch rpc_rm_alloc to capture golden device + vaspace handles
    # allocated inside init_golden_image() during gsp.init_hw() below.
    _golden_refs = {'device': None, 'vaspace': None}
    _orig_gsp_rpc_rm_alloc = type(nvdev.gsp).rpc_rm_alloc
    def _tracking_rpc_rm_alloc(self, hParent, hClass, params, client=None):
        result = _orig_gsp_rpc_rm_alloc(self, hParent, hClass, params, client)
        if hClass == nv_gpu.NV01_DEVICE_0 and _golden_refs['device'] is None:
            _golden_refs['device'] = result
        if hClass == nv_gpu.FERMI_VASPACE_A and _golden_refs['vaspace'] is None:
            _golden_refs['vaspace'] = result
        return result
    type(nvdev.gsp).rpc_rm_alloc = _tracking_rpc_rm_alloc

    _step("step 6a/6 — flcn.init_hw (Falcon MCU boot)")
    nvdev.flcn.init_hw()
    _step("step 6b/6 — gsp.init_hw (GSP firmware load)")
    nvdev.gsp.init_hw()
    type(nvdev.gsp).rpc_rm_alloc = _orig_gsp_rpc_rm_alloc  # restore

    _step("NVDev boot complete")
    gsp            = nvdev.gsp
    golden_client  = gsp.priv_root
    golden_device  = _golden_refs['device']
    golden_subdev  = gsp.subdevice
    golden_vaspace = _golden_refs['vaspace']
    if golden_device is None or golden_vaspace is None:
        raise RuntimeError(
            f"golden handle capture failed: device={golden_device!r} vaspace={golden_vaspace!r}")
    _step(f"golden client=0x{golden_client:08x} device=0x{golden_device:08x} "
          f"subdev=0x{golden_subdev:08x} vaspace=0x{golden_vaspace:08x}")

    bar1_paddr, _bar1_sz = pci_dev.bar_info(1)

    def to_vram_off(system_paddr: int) -> int:
        """BAR1 system physical address → VRAM byte offset."""
        return system_paddr - bar1_paddr

    # ── 2. Allocate VRAM for channel + dispatch resources ────────────────────
    GPFIFO_ENTRIES = 0x10000          # 64K entries × 8 B = 512 KB
    RING_SZ        = GPFIFO_ENTRIES * 8
    AREA_SZ        = 3 << 20         # 3 MB: ring + USERD page + padding
    CMDQ_SZ        = 2 << 20         # 2 MB command queue (QMD + method pkts)
    EOP_SZ         = 0x1000          # 4 KB EOP semaphore page
    GPPUT_OFF      = 140             # GPPut byte-offset within USERD page

    # Code buffer: stores compiled cubin(s); defaults to 64 MB.
    CODE_BUF_SZ = int(os.environ.get("BEAGLE_NV_CODE_MB", "64")) << 20

    # Data pool: capped at 64 MB for initial testing; expand with BEAGLE_NV_DATA_GB.
    _max_mb      = int(os.environ.get("BEAGLE_NV_DATA_MB", "64"))
    DATA_POOL_SZ = (_max_mb << 20)
    DATA_POOL_SZ = (DATA_POOL_SZ + (2 << 20) - 1) & ~((2 << 20) - 1)  # 2MB align

    print(f"nv_init_helper: allocating channel resources "
          f"(data_pool={DATA_POOL_SZ >> 20} MB, code={CODE_BUF_SZ >> 20} MB) …",
          file=sys.stderr)

    # All allocations go through NVMemoryManager.valloc() which maps them in
    # the GPU page tables.  C++ accesses them via (region_gpu_va + offset).
    _step("valloc gpfifo_area (3 MB)")
    gpfifo_area = nvdev.mm.valloc(AREA_SZ,        contiguous=True)
    _step("valloc cmdq_area (2 MB)")
    cmdq_area   = nvdev.mm.valloc(CMDQ_SZ,        contiguous=False)
    _step("valloc eop_area (4 KB)")
    eop_area    = nvdev.mm.valloc(EOP_SZ,          contiguous=True)
    _step(f"valloc code_area ({CODE_BUF_SZ >> 20} MB)")
    code_area   = nvdev.mm.valloc(CODE_BUF_SZ,    contiguous=False)
    _step(f"valloc data_area ({DATA_POOL_SZ >> 20} MB)")
    data_area   = nvdev.mm.valloc(DATA_POOL_SZ,   contiguous=False)
    _step("valloc complete")

    gpfifo_vram = to_vram_off(gpfifo_area.paddrs[0][0])
    cmdq_vram   = to_vram_off(cmdq_area.paddrs[0][0])
    eop_vram    = to_vram_off(eop_area.paddrs[0][0])
    code_vram   = to_vram_off(code_area.paddrs[0][0])
    data_vram   = to_vram_off(data_area.paddrs[0][0])

    userd_vram  = gpfifo_vram + RING_SZ  # USERD page offset within VRAM (for C++ GPPut writes)

    # ── 3. Share valloc'd VA ranges with golden_vaspace ─────────────────────
    # FERMI_VASPACE_A fails (result 59) for ALL additional vaspaces on this
    # macOS/TinyGPU.app setup — the GSP permits only one vaspace per device.
    # The golden image already owns golden_vaspace.  We reuse it and copy our
    # valloc'd page table pages into it with COPY_SERVER_RESERVED_PDES, exactly
    # the same mechanism init_golden_image() uses for the 512MB reserved region.

    vaspace = golden_vaspace  # reuse the golden vaspace for our channel

    _step("sharing valloc'd VA regions with golden_vaspace")
    for area, aname in [(gpfifo_area, 'gpfifo'), (cmdq_area, 'cmdq'),
                        (eop_area,    'eop'),    (code_area, 'code'),
                        (data_area,   'data')]:
        pts = nvdev.mm.page_tables(area.va_addr, area.size)
        n   = len(pts)
        _step(f"  COPY_SERVER_RESERVED_PDES {aname}: "
              f"va=0x{area.va_addr:x} sz=0x{area.size:x} levels={n}")
        bufs_p = nv_gpu.struct_NV90F1_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES_PARAMS(
            pageSize=area.size, numLevelsToCopy=n,
            virtAddrLo=area.va_addr, virtAddrHi=area.va_addr + area.size - 1)
        for i, pt in enumerate(pts):
            bufs_p.levels[i] = \
                nv_gpu.struct_NV90F1_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES_PARAMS_level(
                    physAddress=pt.paddr,
                    size=nvdev.mm.pte_cnt[0] * 8 if i == 0 else 0x1000,
                    pageShift=nvdev.mm.pte_covers[i].bit_length() - 1,
                    aperture=1)
        gsp.rpc_rm_control(golden_vaspace,
                            nv_gpu.NV90F1_CTRL_CMD_VASPACE_COPY_SERVER_RESERVED_PDES,
                            bufs_p, client=golden_client)

    _step("RM 5 — KEPLER_CHANNEL_GROUP_A")
    ch_grp = gsp.rpc_rm_alloc(golden_device, nv_gpu.KEPLER_CHANNEL_GROUP_A,
                                nv_gpu.NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS(
                                    engineType=nv_gpu.NV2080_ENGINE_TYPE_GRAPHICS),
                                client=golden_client)

    _step("RM 6 — FERMI_CONTEXT_SHARE_A")
    ctxshare = gsp.rpc_rm_alloc(ch_grp, nv_gpu.FERMI_CONTEXT_SHARE_A,
                                  nv_gpu.NV_CTXSHARE_ALLOCATION_PARAMETERS(
                                      hVASpace=vaspace,
                                      flags=nv_gpu.NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_ASYNC),
                                  client=golden_client)

    _step("RM 7 — NV_CHANNELGPFIFO (priv_root, explicit userdMem)")
    # vaspace comes from ctxshare — do NOT re-specify hVASpace here (causes
    # NV_ERR_INSUFFICIENT_RESOURCES as RM tries a second VA resource alloc).
    # For priv_root the auto-fill of userdMem (from hUserdMemory[]+userdOffset[])
    # is skipped by rpc_rm_alloc, so provide userdMem directly.
    userd_paddr   = gpfifo_area.paddrs[0][0] + RING_SZ
    userd_mem     = nv_gpu.NV_MEMORY_DESC_PARAMS(base=userd_paddr, size=0x200,
                                                   addressSpace=2, cacheAttrib=0)
    gpfifo_params = nv_gpu.NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS(
        gpFifoOffset  = gpfifo_area.va_addr,
        gpFifoEntries = GPFIFO_ENTRIES,
        hContextShare = ctxshare,
        userdMem      = userd_mem,
        engineType    = 0)
    gpfifo = gsp.rpc_rm_alloc(ch_grp, gsp.gpfifo_class, gpfifo_params,
                                client=golden_client)

    _step("RM 8 — compute_class + manual promote_ctx (priv_root skips auto-promote)")
    gsp.rpc_rm_alloc(gpfifo, gsp.compute_class, None, client=golden_client)
    # rpc_rm_alloc only calls promote_ctx for non-priv clients; call explicitly.
    phys_gr_ctx = gsp.promote_ctx(golden_client, golden_subdev, gpfifo,
                                   {k: v for k, v in gsp.grctx_bufs.items() if k in [0, 1, 2]},
                                   virt=False)
    gsp.promote_ctx(golden_client, golden_subdev, gpfifo,
                    {k: v for k, v in gsp.grctx_bufs.items() if k in [0, 1, 2]},
                    phys_gr_ctx, phys=False)

    _step("RM 9 — dma_class")
    gsp.rpc_rm_alloc(gpfifo, gsp.dma_class, None, client=golden_client)

    # Schedule channel group
    gsp.rpc_rm_control(ch_grp, nv_gpu.NVA06C_CTRL_CMD_GPFIFO_SCHEDULE,
                        nv_gpu.NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS(bEnable=1),
                        client=golden_client)

    # Work submission token
    ws = gsp.rpc_rm_control(
        gpfifo, nv_gpu.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN,
        nv_gpu.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS(
            workSubmitToken=-1),
        client=golden_client)
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
    }

    with open(out_path, "w") as f:
        json.dump(state, f, indent=2)

    print(f"nv_init_helper: done — work_token=0x{work_token:08x} "
          f"compute_class=0x{gsp.compute_class:x} chip={nvdev.chip_name}",
          file=sys.stderr)
    print(f"  gpfifo_vram=0x{gpfifo_vram:x}  eop_vram=0x{eop_vram:x}  "
          f"data_pool={DATA_POOL_SZ>>20} MB", file=sys.stderr)


if __name__ == "__main__":
    main()
