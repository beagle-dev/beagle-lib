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

# Locate tinygrad — prefer env var, fall back to the pinned worktree.
# Default: the tinygrad worktree pinned at a9830e2b4 -- tinygrad HEAD
# (after 2026-09-05) dropped the macOS TinyGPU transport and hcq1 (TODO.md
# Phase 140). TINYGRAD_PATH overrides.
_TINYGRAD_PATH = os.environ.get(
    "TINYGRAD_PATH", os.path.expanduser("~/Dropbox/Projects/tinygrad-hcq1"))
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


# ── 4. Warm-GPU refusal, unload diagnostics and boot baselines (TODO.md plan
# step P1). Everything below only adds reads to what tinygrad does; the one
# behaviour change is the refusal, which replaces tinygrad's WPR2-up branch
# (a bus-master CFG write, then a PCIe reset that is a no-op on macOS 14 and
# a doomed boot) with an error before any write. The originals are kept in
# _ORIG so the offline tests (tinygpu_tests/test_p1_diagnostics.py) can
# drive each wrapper on fakes. ────────────────────────────────────────────
import array as _array, hashlib as _hashlib, pathlib as _pathlib
from tinygrad.runtime.autogen import nv as _nv
from tinygrad.runtime.support.nv.ip import NVRpcQueue
from tinygrad.runtime import ops_nv as _ops_nv

class WarmGPUError(RuntimeError):
    """The GPU still carries a previous boot (WPR2 is up); only a power cycle clears it (STATUS.md R14)."""

_ORIG = {"early_ip_init": NVDev._early_ip_init, "gsp_fini_hw": NV_GSP.fini_hw, "read_resp": NVRpcQueue.read_resp,
         "run_cpu_seq": NV_GSP.run_cpu_seq, "execute_hs": NV_FLCN.execute_hs, "prep_ucode": NV_FLCN.prep_ucode,
         "new_gpu_fifo": _ops_nv.NVDevice._new_gpu_fifo}
_GSP_BASE, _WPR2_ADDR_HI = 0x00110000, 0x001FA828   # the GSP falcon; NV_PFB_PRI_MMU_WPR2_ADDR_HI (dev_fb.py)
_SUSPEND_TIMEOUT_S = 2.0
_in_unload = [False]
_BOOTING = [None]   # the NVDev being booted

def _p1log(msg: str) -> None:
    print(f"nv_init_helper: {msg}", file=sys.stderr, flush=True)

def _guarded_early_ip_init(self):
    _BOOTING[0] = self   # for unload_after_failed_boot (plan step P2)
    # tinygrad reads this register first too (nvdev.py:105); refuse before its bus-master CFG write
    wpr2_hi = self.mmio[_WPR2_ADDR_HI // 4]
    if wpr2_hi != 0:
        raise WarmGPUError(f"WARM GPU: WPR2 is up (NV_PFB_PRI_MMU_WPR2_ADDR_HI=0x{wpr2_hi:08x}), so the previous boot was "
                           "not torn down. Power-cycle the eGPU (unplug and replug it) and retry. Nothing was written to the GPU.")
    return _ORIG["early_ip_init"](self)
NVDev._early_ip_init = _guarded_early_ip_init

def _gsp_fini_hw_with_suspend_wait(self):
    # tinygrad's UNLOADING_GUEST_DRIVER RPC, then NVIDIA's wait for the GSP to report itself suspended
    # (MAILBOX0 == 0x80000000; 570.144 kernel_gsp_tu102.c:1116-1139, nouveau r535 gsp.c:1772-1779)
    nvdev = self.nvdev
    diag = nvdev.beagle_fini = {"unload_ok": False}
    _in_unload[0] = True
    # a LEVEL_0 unload (plan step P2) may post RUN_CPU_SEQUENCER, whose op 8 polls BSI right after starting SEC2:
    # the same hang BEAGLE's 20 s sleep avoids during gsp.init_hw (patch 3), so that sleep covers the unload too
    if _UNLOAD_LEVEL_0: _in_gsp_init[0] = True
    try: _ORIG["gsp_fini_hw"](self)
    finally: _in_unload[0] = _in_gsp_init[0] = False
    deadline = time.monotonic() + _SUSPEND_TIMEOUT_S
    while (mailbox0 := nvdev.NV_PGSP_FALCON_MAILBOX0.read()) != 0x80000000 and time.monotonic() < deadline: time.sleep(0.01)
    diag.update(mailbox0=mailbox0, riscv_cpuctl=nvdev.NV_PRISCV_RISCV_CPUCTL.with_base(_GSP_BASE).read(),
                wpr2_lo=nvdev.NV_PFB_PRI_MMU_WPR2_ADDR_LO.read(), wpr2_hi=nvdev.NV_PFB_PRI_MMU_WPR2_ADDR_HI.read(),
                unload_ok=mailbox0 == 0x80000000)
    _p1log(f"after the unload RPC: GSP MAILBOX0=0x{mailbox0:08x} ({'suspended' if diag['unload_ok'] else 'NOT SUSPENDED'}), "
           f"RISCV_CPUCTL=0x{diag['riscv_cpuctl']:08x}, WPR2_LO=0x{diag['wpr2_lo']:08x}, WPR2_HI=0x{diag['wpr2_hi']:08x}")
NV_GSP.fini_hw = _gsp_fini_hw_with_suspend_wait

def _rpc_name(func: int) -> str:
    return _nv.rpc_fns.get(func, _nv.rpc_events.get(func, f"0x{func:x}"))

def _logged_read_resp(self):
    for func, msg in _ORIG["read_resp"](self):   # same generator, same items; log the status-queue events of the unload
        if _in_unload[0]: _p1log(f"status-queue event during unload: {_rpc_name(func)} ({func:#x})")
        yield func, msg
NVRpcQueue.read_resp = _logged_read_resp

_SEQ_ARGS = {0x0: 2, 0x1: 3, 0x2: 5, 0x3: 1, 0x4: 2, 0x5: 0, 0x6: 0, 0x7: 0, 0x8: 0}   # run_cpu_seq's operand counts (ip.py:633-661)
def _seq_ops(seq_buf: bytes) -> list:
    hdr = _nv.rpc_run_cpu_sequencer_v17_00.from_buffer_copy(seq_buf[:(hdr_sz := ctypes.sizeof(_nv.rpc_run_cpu_sequencer_v17_00))])
    words, ops, i = memoryview(seq_buf[hdr_sz:]).cast('I')[:hdr.cmdIndex], [], 0
    while i < len(words):
        if (n := _SEQ_ARGS.get(op := words[i])) is None: ops.append(f"unknown {op}"); break
        ops.append(op); i += 1 + n
    return ops

def _logged_run_cpu_seq(self, seq_buf: bytes):
    _p1log(f"CPU sequencer ({'during unload' if _in_unload[0] else 'boot'}): ops {_seq_ops(seq_buf)}")
    return _ORIG["run_cpu_seq"](self, seq_buf)
NV_GSP.run_cpu_seq = _logged_run_cpu_seq

def _execute_hs_with_frts_checks(self, base, img_paddr, *args, **kwargs):
    frts, nvdev = img_paddr == getattr(self, "frts_image_paddr", None), self.nvdev
    if frts:   # the conditions tinygrad's (suppressed) wait_for_reset polls, ip.py:94-96
        plm = nvdev.NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_PRIV_LEVEL_MASK.read_bitfields()['read_protection_level0']
        gfw = nvdev.NV_PGC6_AON_SECURE_SCRATCH_GROUP_05[0].read() & 0xff
        _p1log(f"before FWSEC-FRTS: read_protection_level0={plm} (tinygrad waits for 1), SCRATCH_GROUP_05[0]&0xff=0x{gfw:02x} (waits for 0xff)")
    ret = _ORIG["execute_hs"](self, base, img_paddr, *args, **kwargs)
    if img_paddr == getattr(self, "booter_image_paddr", None) and ret is not None and ret[0] == 0:
        nvdev.beagle_gsp_started = True   # booter_load started GSP-RM, which runs from sysmem from here on (unload_after_failed_boot)
    if frts:   # NVIDIA's FRTS post-checks (570.144 kernel_gsp_frts_tu102.c:486-523), before tinygrad's WPR2_HI assert
        scratch = nvdev.NV_PBUS_VBIOS_SCRATCH[0x0e].read()
        wpr2_lo = nvdev.NV_PFB_PRI_MMU_WPR2_ADDR_LO.read_bitfields()['val']
        expected = self.frts_offset >> 12
        _p1log(f"after FWSEC-FRTS: VBIOS scratch 0x0E=0x{scratch:08x} (FRTS error code 0x{scratch >> 16:x}, 0 = none); "
               f"WPR2_LO.val=0x{wpr2_lo:x}, frts_offset>>12=0x{expected:x} ({'match' if wpr2_lo == expected else 'MISMATCH'})")
    return ret
NV_FLCN.execute_hs = _execute_hs_with_frts_checks

_VBIOS_SLICE = slice(0x00300000 // 4, (0x00300000 + 0x100000) // 4)   # the PROM window prep_ucode reads (ip.py:110)

class _VbiosCapture:
    """Pass-through for nvdev.mmio during prep_ucode that keeps a copy of the VBIOS slice it reads."""
    def __init__(self, inner): self._inner, self.vbios = inner, None
    def __getitem__(self, idx):
        val = self._inner[idx]
        if isinstance(idx, slice) and (idx.start, idx.stop) == (_VBIOS_SLICE.start, _VBIOS_SLICE.stop): self.vbios = _array.array('I', val).tobytes()
        return val
    def __setitem__(self, idx, val): self._inner[idx] = val
    def __getattr__(self, name): return getattr(self._inner, name)

def _prep_ucode_with_vbios_capture(self):
    nvdev = self.nvdev
    nvdev.mmio = capture = _VbiosCapture(nvdev.mmio)
    try: _ORIG["prep_ucode"](self)
    finally: nvdev.mmio = capture._inner
    if capture.vbios is None: return
    nvdev.beagle_vbios, digest = capture.vbios, _hashlib.sha256(capture.vbios).hexdigest()
    try:
        out = _pathlib.Path(os.environ.get("BEAGLE_TINYGPU_DATA", _pathlib.Path.home() / ".beagle/tinygpu")) / "vbios"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{nvdev.chip_name}_{digest[:16]}.rom"
        if not path.exists(): path.write_bytes(capture.vbios)
        _p1log(f"VBIOS captured: {len(capture.vbios)} bytes, sha256 {digest}, saved to {path}")
    except OSError as e: _p1log(f"VBIOS captured (sha256 {digest}) but not saved: {e}")
NV_FLCN.prep_ucode = _prep_ucode_with_vbios_capture

def _new_gpu_fifo_with_userd_baseline(self, gpfifo_area, ctxshare, channel_group, offset=0, entries=0x400, compute=False, video=False):
    fifo = _ORIG["new_gpu_fifo"](self, gpfifo_area, ctxshare, channel_group, offset=offset, entries=entries, compute=compute, video=video)
    ctl = _ops_nv.nv_gpu.AmpereAControlGPFifo   # USERD follows the ring, as tinygrad lays it out (ops_nv.py:644-666)
    userd = gpfifo_area.cpu_view().view(offset + entries * 8, fmt='I')
    get, put = userd[getattr(ctl, 'GPGet').offset // 4], userd[getattr(ctl, 'GPPut').offset // 4]
    kind = "compute" if compute else "video" if video else "copy"
    self.__dict__.setdefault("beagle_userd", {})[kind] = (get, put)
    _p1log(f"USERD baseline, {kind} GPFIFO (gpfifo_area+{offset:#x}, before any submission): GPGet=0x{get:x} GPPut=0x{put:x}")
    return fifo
_ops_nv.NVDevice._new_gpu_fifo = _new_gpu_fifo_with_userd_baseline


# ── 5. NVIDIA's driver-unload teardown (TODO.md plan step P2), only with
# BEAGLE_NV_TEARDOWN=1. tinygrad's only teardown is the FAST_UNLOAD RPC,
# NVIDIA's system-shutdown path, which leaves WPR2 up, so every boot needs a
# power cycle (STATUS.md R14). At driver unload NVIDIA (570.144: kgspUnloadRm
# -> kgspTeardown_TU102, kernel_gsp_tu102.c:579-623;
# kgspExecuteBooterUnloadIfNeeded_TU102, kernel_gsp_booter_tu102.c:129-190)
# and nouveau (tu102_gsp_fini) then reset the GSP falcon, run FWSEC-SB, reset
# SEC2 and run Booter Unload, after which WPR2 is down. tinygrad has no such
# code, so the sequence follows NVIDIA, on tinygrad's own falcon primitives
# (reset, execute_hs), with the two images prepared statement by statement
# the way tinygrad prepares FRTS and booter_load. With the variable unset
# nothing below changes a single GPU access. ──────────────────────────────
_TEARDOWN = os.environ.get("BEAGLE_NV_TEARDOWN", "0") not in ("", "0")
_UNLOAD_LEVEL_0 = os.environ.get("BEAGLE_NV_UNLOAD_LEVEL", "") == "0"
_BOOTER_UNLOAD_SHA = {"ad102": "975b85a14ded8e430d30f000c3c1afdd55c15dee04f35ff9dfd876acd7e67186",   # linux-firmware 0a6871b1
                      "ga102": "8e63db5b78d7d3e349f20a2d11099c3d7109081393cb09ffc0a28133324ae009"}
# 570.144 constants tinygrad's autogen lacks: FALCON_APPLICATION_INTERFACE_DMEM_MAPPER_V3_CMD_{FRTS,SB} and the
# NV_VBIOS_FWSECLIC_SCRATCH_INDEX_{0E,15} scratch registers: FRTS's error code is bits 31:16 of 0x0E, SB's is bits
# 15:0 of 0x15 (NV_VBIOS_FWSECLIC_{FRTS,SB}_ERR_CODE, kernel_gsp_frts_tu102.c:133-139)
_FWSEC_CMD_FRTS, _FWSEC_CMD_SB = 0x15, 0x19
_SCRATCH_FRTS_ERR, _SCRATCH_SB_ERR = 0x0e, 0x15
_ORIG.update(prep_booter=NV_FLCN.prep_booter, rpc_unloading_guest_driver=NV_GSP.rpc_unloading_guest_driver)
from tinygrad.helpers import round_up

def _fwsec_ucode(vbios: bytes):
    """prep_ucode's VBIOS walk (ip.py:110-145), statement by statement: the FWSEC descriptor, signature and image."""
    vbios_bytes, vbios_off = memoryview(vbios), 0
    while True:
        pci_blck = vbios_bytes[vbios_off + _nv.OFFSETOF_PCI_EXP_ROM_PCI_DATA_STRUCT_PTR:].cast('H')[0]
        imglen = vbios_bytes[vbios_off + pci_blck + _nv.OFFSETOF_PCI_DATA_STRUCT_IMAGE_LEN:].cast('H')[0] * _nv.PCI_ROM_IMAGE_BLOCK_SIZE
        match vbios_bytes[vbios_off + pci_blck + _nv.OFFSETOF_PCI_DATA_STRUCT_CODE_TYPE]:
            case _nv.NV_BCRT_HASH_INFO_BASE_CODE_TYPE_VBIOS_BASE: block_size = imglen
            case _nv.NV_BCRT_HASH_INFO_BASE_CODE_TYPE_VBIOS_EXT:
                expansion_rom_off = vbios_off - block_size
                break
        vbios_off += imglen

    bit_header = _nv.BIT_HEADER_V1_00.from_buffer_copy(vbios_bytes[(bit_addr := 0x1b0):bit_addr + ctypes.sizeof(_nv.BIT_HEADER_V1_00)])
    assert bit_header.Signature == 0x00544942, f"Invalid BIT header signature {hex(bit_header.Signature)}"

    for i in range(bit_header.TokenEntries):
        bit = _nv.BIT_TOKEN_V1_00.from_buffer_copy(vbios_bytes[bit_addr + bit_header.HeaderSize + i * bit_header.TokenSize:])
        if bit.TokenId != _nv.BIT_TOKEN_FALCON_DATA or bit.DataVersion != 2 or bit.DataSize < _nv.BIT_DATA_FALCON_DATA_V2_SIZE_4: continue

        falcon_data = _nv.BIT_DATA_FALCON_DATA_V2.from_buffer_copy(vbios_bytes[bit.DataPtr & 0xffff:])
        ucode_hdr = _nv.FALCON_UCODE_TABLE_HDR_V1.from_buffer_copy(vbios_bytes[(table_ptr := expansion_rom_off + falcon_data.FalconUcodeTablePtr):])
        for j in range(ucode_hdr.EntryCount):
            ucode_entry = _nv.FALCON_UCODE_TABLE_ENTRY_V1.from_buffer_copy(vbios_bytes[table_ptr + ucode_hdr.HeaderSize + j * ucode_hdr.EntrySize:])
            if ucode_entry.ApplicationID != _nv.FALCON_UCODE_ENTRY_APPID_FWSEC_PROD: continue

            ucode_desc_hdr = _nv.FALCON_UCODE_DESC_HEADER.from_buffer_copy(vbios_bytes[expansion_rom_off + ucode_entry.DescPtr:])
            ucode_desc_off = expansion_rom_off + ucode_entry.DescPtr
            ucode_desc_size = ucode_desc_hdr.vDesc >> 16

    desc_v3 = _nv.FALCON_UCODE_DESC_V3.from_buffer_copy(vbios_bytes[ucode_desc_off:ucode_desc_off + ucode_desc_size])

    sig_total_size = ucode_desc_size - _nv.FALCON_UCODE_DESC_V3_SIZE_44
    signature = vbios_bytes[ucode_desc_off + _nv.FALCON_UCODE_DESC_V3_SIZE_44:][:sig_total_size]
    image = vbios_bytes[ucode_desc_off + ucode_desc_size:][:round_up(desc_v3.StoredSize, 256)]
    return desc_v3, signature, image

def _fwsec_patch(desc_v3, image, signature, cmd_id: int, cmd: bytes) -> bytearray:
    """prep_ucode's __patch (ip.py:147-164), statement by statement, without its allocation."""
    patched_image = bytearray(image)

    dmem_offset = 0
    hdr = _nv.FALCON_APPLICATION_INTERFACE_HEADER_V1.from_buffer_copy(image[(app_hdr_off := desc_v3.IMEMLoadSize + desc_v3.InterfaceOffset):])
    ents = (_nv.FALCON_APPLICATION_INTERFACE_ENTRY_V1 * hdr.entryCount).from_buffer_copy(image[app_hdr_off + ctypes.sizeof(hdr):])
    for i in range(hdr.entryCount):
        if ents[i].id == _nv.FALCON_APPLICATION_INTERFACE_ENTRY_ID_DMEMMAPPER: dmem_offset = ents[i].dmemOffset

    # Patch image
    dmem = _nv.FALCON_APPLICATION_INTERFACE_DMEM_MAPPER_V3.from_buffer_copy(image[(dmem_mapper_offset := desc_v3.IMEMLoadSize + dmem_offset):])
    dmem.init_cmd = cmd_id
    patched_image[dmem_mapper_offset:dmem_mapper_offset + len(bytes(dmem))] = bytes(dmem)
    patched_image[(cmd_off := desc_v3.IMEMLoadSize + dmem.cmd_in_buffer_offset):cmd_off + len(cmd)] = cmd
    patched_image[(sig_off := desc_v3.IMEMLoadSize + desc_v3.PKCDataOffset):sig_off + 0x180] = signature[-0x180:]
    return patched_image

def _booter_ucode(fw_name: str, fw_file: str, sha: str):
    """prep_booter's body (ip.py:171-184), statement by statement: the signed booter image and its load parameters
    (data offset, data size, code offset, code size)."""
    from tinygrad.helpers import fetch_fw
    h = _nv.struct_nvfw_bin_hdr.from_buffer_copy(b := fetch_fw(f"nvidia/{fw_name}/gsp", fw_file, sha))
    lh = _nv.struct_nvfw_hs_load_header_v2.from_buffer_copy(b, (hs := _nv.struct_nvfw_hs_header_v2.from_buffer_copy(b, h.header_offset)).header_offset)
    app = _nv.struct_nvfw_hs_load_header_v2_app.from_buffer_copy(b, hs.header_offset + ctypes.sizeof(_nv.struct_nvfw_hs_load_header_v2))

    patch_loc, patch_sig = struct.unpack_from("<I", b, hs.patch_loc)[0], struct.unpack_from("<I", b, hs.patch_sig)[0]
    sig = b[(sig_off := hs.sig_prod_offset + patch_sig):sig_off + (sig_len := hs.sig_prod_size // struct.unpack_from("<I", b, hs.num_sig)[0])]

    (patched_image := bytearray(b[h.data_offset:h.data_offset + h.data_size]))[patch_loc:patch_loc + sig_len] = sig
    return patched_image, lh.os_data_offset, lh.os_data_size, app.offset, app.size

def _prep_booter_with_teardown_images(self):
    # right after tinygrad's own images, so FRTS and booter_load keep the addresses they have without the teardown
    _ORIG["prep_booter"](self)
    nvdev = self.nvdev
    if not _TEARDOWN or nvdev.fw_name not in _BOOTER_UNLOAD_SHA: return
    vbios = getattr(nvdev, "beagle_vbios", None)   # this boot's VBIOS, from prep_ucode's capture wrapper
    if vbios is None: raise RuntimeError("teardown: the VBIOS read by prep_ucode was not captured")
    desc_v3, signature, image = _fwsec_ucode(vbios)
    if bytes(desc_v3) != bytes(self.desc_v3): raise RuntimeError("teardown: the FWSEC descriptor differs from prep_ucode's")
    read_vbios_desc = _nv.FWSECLIC_READ_VBIOS_DESC(version=0x1, size=ctypes.sizeof(_nv.FWSECLIC_READ_VBIOS_DESC), flags=2)
    sb = _fwsec_patch(desc_v3, image, signature, _FWSEC_CMD_SB, bytes(read_vbios_desc))   # SB takes READ_VBIOS_DESC alone (frts_tu102.c:334-339)
    _, self.beagle_sb_image_paddr, _ = nvdev._alloc_boot_mem(len(sb), data=sb, sysmem=False)
    img, data_off, data_sz, code_off, code_sz = _booter_ucode(nvdev.fw_name, "booter_unload-570.144.bin", _BOOTER_UNLOAD_SHA[nvdev.fw_name])
    _, self.beagle_unload_image_paddr, _ = nvdev._alloc_boot_mem(len(img), data=img, sysmem=False)
    self.beagle_unload_params = (data_off, data_sz, code_off, code_sz)
    _p1log(f"teardown images: FWSEC-SB {len(sb)} bytes at VRAM 0x{self.beagle_sb_image_paddr:x}, Booter Unload {len(img)} bytes "
           f"at VRAM 0x{self.beagle_unload_image_paddr:x} (code 0x{code_off:x}+0x{code_sz:x}, data 0x{data_off:x}+0x{data_sz:x})")
NV_FLCN.prep_booter = _prep_booter_with_teardown_images

def _tolerant_reset(self, base: int, td: dict, name: str):
    # tinygrad's reset; as in NVIDIA's kflcnReset_TU102 (kernel_falcon_tu102.c:175-189), a core-select (BCR) timeout does not
    # stop the teardown and FALCON_RM is still written (tinygrad writes it only once the core select succeeds, ip.py:282-283)
    try: self.reset(base)
    except TimeoutError as e:
        if "RISCV core not booted" not in str(e): raise
        td[f"{name}_bcr_timeout"] = True
        self.nvdev.NV_PFALCON_FALCON_RM.with_base(base).write(self.nvdev.chip_id)
        _p1log(f"teardown: {name} reset: core select timed out ({e}); FALCON_RM written, continuing, as NVIDIA does")

_FALCON_ERRORS = (TimeoutError, RuntimeError, AssertionError)   # tinygrad's wait_cond timeouts, and asserts

def _flcn_fini_hw_teardown(self):
    """kgspTeardown_TU102 after the GSP unload (NVDev.fini runs gsp.fini_hw first, nvdev.py:88-89, NVIDIA's order). As in
    NVIDIA (kernel_gsp_tu102.c:597-620, kernel_gsp_booter_tu102.c:155) and nouveau (tu102_gsp_fini), a failed GSP reset,
    FWSEC-SB or SEC2 reset is recorded and Booter Unload still runs; only Booter Unload and WPR2 decide the outcome."""
    nvdev = self.nvdev
    diag = getattr(nvdev, "beagle_fini", None)
    if not _TEARDOWN or diag is None or not hasattr(self, "beagle_sb_image_paddr"): return
    td = diag["teardown"] = {}
    if not diag.get("unload_ok"):
        td["result"] = "skipped: the GSP did not confirm its unload, so no falcon is touched"
        _p1log(f"teardown {td['result']}")
        return
    falcon, sec2 = 0x00110000, 0x00840000   # as NV_FLCN.init_hw (ip.py:187)
    try:
        _tolerant_reset(self, falcon, td, "gsp")
        # FWSEC-SB, with the arguments tinygrad runs FWSEC-FRTS with (ip.py:190-193)
        self.execute_hs(falcon, self.beagle_sb_image_paddr, code_off=0x0, data_off=self.desc_v3.IMEMLoadSize,
                        imemPa=self.desc_v3.IMEMPhysBase, imemVa=self.desc_v3.IMEMVirtBase, imemSz=self.desc_v3.IMEMLoadSize,
                        dmemPa=self.desc_v3.DMEMPhysBase, dmemVa=0x0, dmemSz=self.desc_v3.DMEMLoadSize,
                        pkc_off=self.desc_v3.PKCDataOffset, engid=self.desc_v3.EngineIdMask, ucodeid=self.desc_v3.UcodeId)
        scratch = nvdev.NV_PBUS_VBIOS_SCRATCH[_SCRATCH_SB_ERR].read()
        td.update(sb_error=scratch & 0xffff,   # logged, not fatal (NVIDIA: NV_ASSERT_FAILED and continue; nouveau: WARN_ON)
                  plm=nvdev.NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_PRIV_LEVEL_MASK.read_bitfields()['read_protection_level0'],
                  gfw_progress=nvdev.NV_PGC6_AON_SECURE_SCRATCH_GROUP_05[0].read() & 0xff)
        _p1log(f"teardown: FWSEC-SB ran: VBIOS scratch 0x15=0x{scratch:08x} (SB error code 0x{scratch & 0xffff:x}, 0 = none), "
               f"read_protection_level0={td['plm']}, GFW progress 0x{td['gfw_progress']:02x}")
    except _FALCON_ERRORS as e:   # NVIDIA: NV_ASSERT_FAILED, then Booter Unload regardless (kernel_gsp_tu102.c:599-620)
        td["sb_failed"] = f"{type(e).__name__}: {e}"
        _p1log(f"teardown: GSP reset or FWSEC-SB failed ({td['sb_failed']}); continuing with Booter Unload, as NVIDIA does")
    try:
        if nvdev.NV_PFB_PRI_MMU_WPR2_ADDR_HI.read() == 0:   # NVIDIA skips Booter Unload when WPR2 is already down
            td["result"] = "done: WPR2 already down after FWSEC-SB"
            return
        try: _tolerant_reset(self, sec2, td, "sec2")
        except _FALCON_ERRORS as e:   # NVIDIA: a non-fatal NV_ASSERT_OK (kernel_gsp_booter_tu102.c:155)
            td["sec2_reset_failed"] = f"{type(e).__name__}: {e}"
            _p1log(f"teardown: SEC2 reset failed ({td['sec2_reset_failed']}); running Booter Unload anyway, as NVIDIA does")
        data_off, data_sz, code_off, code_sz = self.beagle_unload_params
        mbx = self.execute_hs(sec2, self.beagle_unload_image_paddr, code_off=code_off, data_off=data_off, imemPa=0x0, imemVa=code_off,
                              imemSz=code_sz, dmemPa=0x0, dmemVa=0x0, dmemSz=data_sz, pkc_off=0x10, engid=1, ucodeid=3,
                              mailbox=(0xff << 32) | 0xff)   # booter_load's parameters (ip.py:202-205); mailboxes 0xFF for a normal unload
        wpr2_hi = nvdev.NV_PFB_PRI_MMU_WPR2_ADDR_HI.read()
        td.update(booter_mailbox0=mbx[0], booter_mailbox1=mbx[1])
        td["result"] = ("done: Booter Unload lowered WPR2" if mbx[0] == 0 and wpr2_hi == 0 else
                        f"failed: Booter Unload returned mailbox0=0x{mbx[0]:x} and WPR2_HI=0x{wpr2_hi:x}")
    except _FALCON_ERRORS as e:   # Booter Unload's DMA or halt timeout
        td["result"] = f"failed: {type(e).__name__}: {e}"
    finally:
        diag.update(wpr2_lo=nvdev.NV_PFB_PRI_MMU_WPR2_ADDR_LO.read(), wpr2_hi=nvdev.NV_PFB_PRI_MMU_WPR2_ADDR_HI.read())
        diag["wpr2_down"] = diag["wpr2_hi"] == 0
        # the next boot needs no power cycle only if WPR2 is down and Booter Unload was not needed or returned 0 (plan (b) 8)
        diag["teardown_ok"] = diag["wpr2_down"] and td.get("result", "").startswith("done:")
        _p1log(f"teardown {td.get('result', 'interrupted')}; WPR2_HI=0x{diag['wpr2_hi']:08x}: "
               f"{'the next boot needs no power cycle' if diag['teardown_ok'] else 'power-cycle before the next boot'}")
NV_FLCN.fini_hw = _flcn_fini_hw_teardown

def unload_after_failed_boot():
    """A boot that failed after booter_load started GSP-RM, which then runs from sysmem: tinygrad adds a device to
    Device._opened_devices only once its constructor returns, so nothing would unload it, and closing the TinyGPU.app
    connection could unmap memory the GSP still uses. Sends the unload RPC and waits for the suspend (NV_GSP.fini_hw) and
    returns what it recorded; the caller holds the connection unless unload_ok. None if GSP-RM never started: closing is safe
    (a failed booter_load leaves it unstarted; the recorded 0x29 failures closed without DART events, STATUS.md R14)."""
    nvdev = _BOOTING[0]
    if nvdev is None or not getattr(nvdev, "beagle_gsp_started", False): return None
    try: nvdev.gsp.fini_hw()
    except Exception as e: _p1log(f"unload after the failed boot: {type(e).__name__}: {e}")
    return getattr(nvdev, "beagle_fini", {"unload_ok": False})

def _rpc_unloading_guest_driver_level0(self):
    # tinygrad's rpc_unloading_guest_driver with NVIDIA's driver-unload level (gpuStateDestroy -> kgspUnloadRm(NORMAL, LEVEL_0),
    # gpu.c:3269-3273) instead of FAST_UNLOAD; the fallback if Booter Unload fails after FAST_UNLOAD
    data = _nv.rpc_unloading_guest_driver_v(bInPMTransition=0, bGc6Entering=0, newLevel=0)
    self.cmd_q.send_rpc(_nv.NV_VGPU_MSG_FUNCTION_UNLOADING_GUEST_DRIVER, bytes(data))
    self.stat_q.wait_resp(_nv.NV_VGPU_MSG_FUNCTION_UNLOADING_GUEST_DRIVER)
if _UNLOAD_LEVEL_0: NV_GSP.rpc_unloading_guest_driver = _rpc_unloading_guest_driver_level0


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

def _sass_version_and_arch(sm_version: int) -> tuple:
    """Derive (QMD SASS_VERSION byte, ptxas --gpu-name) from the real
    NV2080_CTRL_GR_INFO_INDEX_SM_VERSION register value (queried at "RM 3b"
    below, the same RM_CONTROL call that already provides
    num_gpcs/num_tpc_per_gpc/num_sm_per_tpc/max_warps_per_sm) -- not a
    chip-name guess. Formula verbatim from tinygrad's own real-hardware
    reference (ops_nv.py NVDevice.__init__, its own comment: "FIXME: no
    idea how to convert this for blackwells" -- kept as-is rather than
    "improved", since it's the actual validated logic, not a guess):
        arch = "sm_120" if sm_version == 0xa04 else
               f"sm_{(sm_version>>8)&0xff}{...}"
        sass_version = ((sm_version & 0xf00) >> 4) | (sm_version & 0xf)
    See STATUS.md §62/TODO.md Phase 27 for why this driver's old
    chip-name-prefix table (mapping every "GB2"-prefixed chip, including
    this real one, to a single hardcoded compute capability 10.0) was
    wrong specifically for this consumer Blackwell chip.
    """
    arch = "sm_120" if sm_version == 0xa04 else \
        f"sm_{(sm_version >> 8) & 0xff}{(v >> 4) if (v := sm_version & 0xff) > 0xf else v}"
    sass_version = ((sm_version & 0xf00) >> 4) | (sm_version & 0xf)
    return sass_version, arch


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

    # ── 2b. Query real GR topology, allocate + program shader local memory ────
    #
    # Shader local memory (SLM) is per-thread stack/register-spill storage --
    # a completely separate physical region from shared memory. It's addressed
    # through a window (SET_SHADER_LOCAL_MEMORY_WINDOW_A, already programmed
    # above) that translates generic addresses into a backing VRAM region --
    # but that backing region must itself be allocated and pointed to via
    # SET_SHADER_LOCAL_MEMORY_A. This driver never did that (grep confirmed
    # zero hits for SET_SHADER_LOCAL_MEMORY_A anywhere in the C++ side) even
    # though kernelPartialsPartialsNoScale's own compiled SASS genuinely needs
    # 576 bytes/thread of it (register spills, confirmed via ELF metadata,
    # see STATUS.md "usb" branch). Reference: NVDevice._query_gpu_info() +
    # NVDevice._ensure_has_local_memory() in ops_nv.py.
    #
    # is_nvd() (isinstance(iface, PCIIface)) is true for this real-PCIe-GPU
    # setup, so tinygrad's own reference always takes the pointer-free
    # NV2080_CTRL_CMD_INTERNAL_STATIC_KGR_GET_INFO path here (a fixed-size,
    # fully self-contained struct -- no embedded pointers), not the sibling
    # NV2080_CTRL_CMD_GR_GET_INFO (which takes a raw host pointer as one of
    # its fields and is never exercised over GSP RPC for this device class).
    _step("RM 3b — query GR topology (NUM_GPCS/NUM_TPC_PER_GPC/NUM_SM_PER_TPC/MAX_WARPS_PER_SM)")
    gr_info = gsp.rpc_rm_control(
        user_subdev, nv_gpu.NV2080_CTRL_CMD_INTERNAL_STATIC_KGR_GET_INFO,
        nv_gpu.NV2080_CTRL_INTERNAL_STATIC_GR_GET_INFO_PARAMS(), USER_ROOT)
    def _gr_info(idx_name):
        # NOTE: the inner getattr's own `None` default is required -- it's
        # an argument expression, evaluated eagerly regardless of whether
        # the outer name resolves, so omitting it throws AttributeError on
        # any idx_name whose LITTER_-prefixed variant doesn't exist (e.g.
        # MAX_WARPS_PER_SM, which only has the non-LITTER name) even when
        # the outer lookup would have succeeded.
        idx = getattr(nv_gpu, 'NV2080_CTRL_GR_INFO_INDEX_' + idx_name,
               getattr(nv_gpu, 'NV2080_CTRL_GR_INFO_INDEX_LITTER_' + idx_name, None))
        assert idx is not None, f"unknown GR info index name: {idx_name}"
        return gr_info.engineInfo[0].infoList[idx].data
    topo_num_gpcs         = _gr_info('NUM_GPCS')
    topo_num_tpc_per_gpc  = _gr_info('NUM_TPC_PER_GPC')
    topo_num_sm_per_tpc   = _gr_info('NUM_SM_PER_TPC')
    topo_max_warps_per_sm = _gr_info('MAX_WARPS_PER_SM')
    topo_sm_version       = _gr_info('SM_VERSION')
    _step(f"  num_gpcs={topo_num_gpcs} num_tpc_per_gpc={topo_num_tpc_per_gpc} "
          f"num_sm_per_tpc={topo_num_sm_per_tpc} max_warps_per_sm={topo_max_warps_per_sm} "
          f"sm_version=0x{topo_sm_version:x}")

    # Real compute capability / ptxas target, derived from the sm_version
    # register just queried above -- not the chip-name-prefix guess
    # (_sass_version) this driver used before. See STATUS.md §62/TODO.md
    # Phase 27: that guess mapped every "GB2"-prefixed chip to compute
    # capability 10.0 (correct for datacenter GB100/GB200, wrong for this
    # real consumer chip, which public specs and this same real register
    # both indicate is compute capability 12.0 / sm_120).
    sass_version, gpu_arch = _sass_version_and_arch(topo_sm_version)
    _step(f"  derived sass_version=0x{sass_version:x} gpu_arch={gpu_arch}")

    # Per-thread SLM bound: generous static ceiling (BEAGLE's kernels are
    # compiled once at startup and their real requirement, e.g. 576 B/thread
    # for kernelPartialsPartialsNoScale, is known only later in the C++ side
    # -- see STATUS.md/TODO.md "usb" branch for why this can't easily be
    # queried lazily the way tinygrad does per-kernel). 4096 B/thread is a
    # wide margin over every kernel measured so far; override with
    # BEAGLE_NV_SLM_PER_THREAD if a future kernel ever needs more (a real,
    # observable failure -- not silent -- if this bound is ever too small,
    # since GetFunction() would then be building a QMD declaring more
    # per-thread SLM than this pool was sized for).
    slm_per_thread = int(os.environ.get("BEAGLE_NV_SLM_PER_THREAD", "4096"))

    def _round_up(x, n): return (x + n - 1) & ~(n - 1)
    LOCAL_MEM_TPC_BYTES = _round_up(_round_up(slm_per_thread * 32, 0x200) *
                                     topo_max_warps_per_sm * topo_num_sm_per_tpc, 0x8000)
    LOCAL_MEM_SZ = _round_up(LOCAL_MEM_TPC_BYTES * topo_num_tpc_per_gpc * topo_num_gpcs, 0x20000)
    _step(f"valloc local_mem_area ({LOCAL_MEM_SZ >> 20} MB contiguous, "
          f"tpc_bytes=0x{LOCAL_MEM_TPC_BYTES:x})")
    local_mem_area = nvdev.mm.valloc(LOCAL_MEM_SZ, contiguous=True)
    local_mem_vram = local_mem_area.paddrs[0][0]

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
        # Shader local memory (register-spill/stack backing store) — pointed
        # to via SET_SHADER_LOCAL_MEMORY_A in send_channel_setup(); never
        # host-accessed directly, so gpu_va is what C++ actually needs.
        "local_mem_vram":      local_mem_vram,
        "local_mem_gpu_va":    local_mem_area.va_addr,
        "local_mem_sz":        LOCAL_MEM_SZ,
        "local_mem_tpc_bytes": LOCAL_MEM_TPC_BYTES,
        # Memory manager
        "mm_vram_pa_base":  bar1_paddr,
        "vram_size":        nvdev.vram_size,
        # Architecture
        "compute_class":    gsp.compute_class,
        "dma_class":        gsp.dma_class,
        "sass_version":     sass_version,
        "gpu_arch":         gpu_arch,
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
    # Also treat SIGINT (Ctrl-C on the foreground process group, which
    # reaches this subprocess too) the same way: without this, SIGINT's
    # default KeyboardInterrupt is a BaseException that skips both the
    # `except Exception` wrapper in main() and nvdev.fini() below, dropping
    # the socket to a possibly-wedged GPU without a clean reset — fatal on
    # macOS eGPU (see InheritedFDPCIDevice.reset()).
    import signal as _signal
    _done = [False]
    def _sigterm(sig, frame): _done[0] = True
    _signal.signal(_signal.SIGTERM, _sigterm)
    _signal.signal(_signal.SIGINT, _sigterm)
    while not _done[0]:
        time.sleep(0.5)

    print("nv_init_helper: signal received — calling nvdev.fini()", file=sys.stderr, flush=True)
    nvdev.fini()
    print("nv_init_helper: fini complete — exiting cleanly", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
