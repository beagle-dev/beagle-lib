#!/usr/bin/env python3
"""
amd_diag_minimal_dispatch.py — isolated diagnostic, NOT part of the BEAGLE
build. Boots the AMD eGPU standalone (no C++, no BEAGLE), compiles ONE
trivial OpenCL kernel, and dispatches it via a PM4 packet stream built fresh
in Python (independent re-derivation from ops_amd.py's exec(), not copied
from GPUInterfaceTinyGPUHybridAMD.cpp) -- using tinygrad's own real
RemoteMMIOInterface for the actual bytes-over-the-socket mechanics (not the
hand-rolled C++ socket code), so this cross-checks two independent things at
once:
  1. Does the core dispatch mechanism (queue, PM4 packet content, doorbell,
     GPU-VM memory access from a running kernel) work AT ALL over this
     remote/USB4 setup, on hardware, isolated from any BEAGLE-specific
     buffer/pointer complexity?
  2. Does an independently-written Python transcription of the same PM4
     packet content agree with (or diverge from) the C++ version?

Safety: this is deliberately the smallest possible real dispatch -- one
kernel, one buffer, no BEAGLE machinery. Includes the same bounded
completion-poll timeout as the C++ path (no infinite wait).

Run: python3 amd_diag_minimal_dispatch.py
"""
import sys, os, ctypes, struct, time, pathlib

_TINYGRAD_PATH = os.environ.get("TINYGRAD_PATH", str(pathlib.Path.home() / "Dropbox/Projects/tinygrad"))
sys.path.insert(0, _TINYGRAD_PATH)
sys.path.insert(0, str(pathlib.Path.home() / "Dropbox/Projects/beagle-lib/libhmsbeagle/GPU"))

from tinygrad.runtime.support.system import APLRemotePCIDevice
from tinygrad.runtime.support.am.amdev import AMDev
from tinygrad.runtime.support.memory import MemoryManager
from tinygrad.runtime.autogen.am import am
import importlib

import amd_compile_helper as ach  # reuse the real compile_opencl()/parse_kernels(), not a rewrite

# Same TinyGPU.app VRAM-zero-write patch as amd_init_helper.py.
_PALLOC_ZERO_LIMIT = 64 << 10
_orig_palloc = MemoryManager.palloc
def _palloc_nozero_large(self, size, align=0x1000, zero=True, boot=False, ptable=False):
    if zero and size > _PALLOC_ZERO_LIMIT: zero = False
    return _orig_palloc(self, size, align, zero=zero, boot=boot, ptable=ptable)
MemoryManager.palloc = _palloc_nozero_large


def step(msg): print(f"[diag] {msg}", file=sys.stderr, flush=True)


def main():
    step("connecting to TinyGPU.app (fresh connection, no C++ parent) ...")
    pci_dev = APLRemotePCIDevice("AM", "usb4")

    step("AMDev boot ...")
    adev = AMDev(pci_dev)
    gfxver = adev.ip_ver[am.GC_HWIP]
    arch = "gfx%d%x%x" % gfxver
    step(f"boot complete — arch={arch} vram={adev.vram_size>>20} MB")

    # Printed as early as possible (right after boot, before anything else
    # touches hardware) so this diagnostic value survives even if something
    # later in the script has a problem. adev.regCOMPUTE_* are real
    # AMRegister objects built during AMDev's own _build_regs() at boot, from
    # the actual discovered per-board offsets used to program this hardware
    # -- NOT the same code path as the static navi_offsets.py extraction
    # GPUInterfaceTinyGPUHybridAMD.cpp's hardcoded constants came from. If
    # these disagree with the C++ values in the comments, that's a real,
    # actionable finding on its own.
    step(f"cross-check vs C++ hardcoded constants: "
         f"PGM_LO=0x{adev.regCOMPUTE_PGM_LO.addr[0]:x} (C++: 0x2e0c) "
         f"RSRC1=0x{adev.regCOMPUTE_PGM_RSRC1.addr[0]:x} (C++: 0x2e12) "
         f"DISPATCH_INITIATOR=0x{adev.regCOMPUTE_DISPATCH_INITIATOR.addr[0]:x} (C++: 0x2e00) "
         f"USER_DATA_0=0x{adev.regCOMPUTE_USER_DATA_0.addr[0]:x} (C++: 0x2e40) "
         f"TMPRING_SIZE=0x{adev.regCOMPUTE_TMPRING_SIZE.addr[0]:x} (C++: 0x2e18) "
         f"DISPATCH_SCRATCH_BASE_LO=0x{adev.regCOMPUTE_DISPATCH_SCRATCH_BASE_LO.addr[0]:x} (C++: 0x2e10)")

    # ---- tiny allocations: ring, eop, ctrl (rptr/wptr/signal), code, data, scratch ----
    RING_SZ, EOP_SZ, CTRL_SZ = 1 << 16, 0x1000, 0x1000  # 64KB ring is plenty for one packet
    ring_area = adev.mm.valloc(RING_SZ, contiguous=True)
    eop_area  = adev.mm.valloc(EOP_SZ, contiguous=True)
    ctrl_area = adev.mm.valloc(CTRL_SZ, contiguous=True)
    code_area = adev.mm.valloc(1 << 20, contiguous=True)   # 1MB, plenty for one kernel
    data_area = adev.mm.valloc(1 << 16, contiguous=True)   # output buffer + kernarg buffer

    rptr_va, wptr_va, signal_va = ctrl_area.va_addr, ctrl_area.va_addr + 8, ctrl_area.va_addr + 0x10
    signal_vram_off = ctrl_area.paddrs[0][0] + 0x10

    step("gfx.setup_ring() ...")
    doorbell_index = adev.gfx.setup_ring(ring_area.va_addr, RING_SZ, rptr_va, wptr_va,
                                          eop_area.va_addr, EOP_SZ, 0, False)
    doorbell_byte_off = doorbell_index * 8
    step(f"queue ready — doorbell_index={doorbell_index}")

    hdp_flush_dword_idx = adev.reg("regBIF_BX0_REMAP_HDP_MEM_FLUSH_CNTL").read() // 4

    # ---- chip props + scratch (same formula as amd_init_helper.py §4b) ----
    gi = adev.gc_info
    if gi.header.version_major == 2:
        cu_per_sa, max_sh_per_se = gi.gc_num_cu_per_sh, gi.gc_num_sh_per_se
    else:
        cu_per_sa, max_sh_per_se = 2 * (gi.gc_num_wgp0_per_sa + gi.gc_num_wgp1_per_sa), gi.gc_num_sa_per_se
    se_cnt = gi.gc_num_se
    cu_cnt = cu_per_sa * max_sh_per_se * se_cnt
    max_slots_scratch_cu = gi.gc_max_scratch_slots_per_cu
    PRIVATE_SEGMENT_CAP = 1024
    size_per_thread = ((PRIVATE_SEGMENT_CAP + 3) // 4) * 4
    scratch_sz = size_per_thread * 64 * max_slots_scratch_cu * cu_cnt
    scratch_area = adev.mm.valloc(scratch_sz, contiguous=True)
    step(f"scratch allocated ({scratch_sz>>10} KB)")

    # ---- compile ONE trivial kernel via the real amd_compile_helper.py code ----
    step("compiling trivial kernel via comgr ...")
    comgr, C = ach._load_comgr()
    src = "kernel void foo(global float* x) { x[get_global_id(0)] = 42.0f; }"
    hsaco = ach.compile_opencl(comgr, C, src, arch)
    image, kernels = ach.parse_kernels(hsaco)
    kd_addr, desc = kernels["foo"]
    entry_offset = kd_addr + desc.kernel_code_entry_byte_offset
    group_segment_size = desc.group_segment_fixed_size
    private_segment_size = desc.private_segment_fixed_size
    kernarg_size = desc.kernarg_size
    from tinygrad.runtime.autogen import hsa as _hsa
    wave32 = bool(desc.kernel_code_properties & _hsa.AMD_KERNEL_CODE_PROPERTIES_ENABLE_WAVEFRONT_SIZE32)
    enable_dispatch_ptr = bool(desc.kernel_code_properties & _hsa.AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR)
    enable_private_segment_sgpr = bool(desc.kernel_code_properties & _hsa.AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER)
    assert not enable_private_segment_sgpr, "trivial kernel unexpectedly needs the legacy private-segment-buffer SGPR path -- not implemented here"
    lds_size = ((group_segment_size + 511) // 512) & 0x1FF
    rsrc1 = desc.compute_pgm_rsrc1 | (1 << 20)  # gfx11 cwsr workaround
    rsrc2 = desc.compute_pgm_rsrc2 | (lds_size << 15)
    rsrc3 = desc.compute_pgm_rsrc3
    step(f"compiled — entry_offset={entry_offset} rsrc1=0x{rsrc1:x} rsrc2=0x{rsrc2:x} rsrc3=0x{rsrc3:x} "
         f"private_seg={private_segment_size} kernarg_size={kernarg_size} wave32={wave32} "
         f"enable_dispatch_ptr={enable_dispatch_ptr}")
    assert entry_offset % 256 == 0, "entry not 256-aligned -- would explain a DVA fault by itself"
    assert private_segment_size <= PRIVATE_SEGMENT_CAP

    # ---- upload code image, allocate output buffer + kernarg buffer ----
    adev.vram.view(code_area.paddrs[0][0], len(image), fmt='B')[:] = bytes(image)
    prog_addr = code_area.va_addr + entry_offset

    out_buf_va = data_area.va_addr           # global float* x  (1 float = 4 bytes; alloc plenty)
    out_buf_vram_off = data_area.paddrs[0][0]
    ka_va = data_area.va_addr + 0x100        # kernarg buffer, well clear of the output buffer
    ka_vram_off = data_area.paddrs[0][0] + 0x100
    ka_total = kernarg_size + (64 if enable_dispatch_ptr else 0)

    kbuf = bytearray(ka_total)
    struct.pack_into('<Q', kbuf, 0, out_buf_va)   # arg0: global float* x
    dispatch_ptr_va = 0
    if enable_dispatch_ptr:
        dp_off = kernarg_size
        struct.pack_into('<H', kbuf, dp_off + 4, 1)   # workgroup_size_x = 1
        struct.pack_into('<H', kbuf, dp_off + 6, 1)
        struct.pack_into('<H', kbuf, dp_off + 8, 1)
        struct.pack_into('<I', kbuf, dp_off + 12, 1)  # grid_size_x = 1 (total threads)
        struct.pack_into('<I', kbuf, dp_off + 16, 1)
        struct.pack_into('<I', kbuf, dp_off + 20, 1)
        struct.pack_into('<Q', kbuf, dp_off + 40, ka_va)  # kernarg_address
        dispatch_ptr_va = ka_va + kernarg_size

    adev.vram.view(out_buf_vram_off, 4, fmt='B')[:] = b'\x00\x00\x00\x00'
    adev.vram.view(ka_vram_off, len(kbuf), fmt='B')[:] = bytes(kbuf)
    adev.vram.view(signal_vram_off, 8, fmt='B')[:] = bytes(8)  # zero the completion signal

    # ---- build the PM4 packet stream (fresh transcription of ops_amd.py exec()) ----
    pm4 = importlib.import_module("tinygrad.runtime.autogen.am.pm4_nv")

    def pkt3(op, *vals):
        return [pm4.PACKET3(op, len(vals) - 1), *vals]

    def set_sh_reg(reg, *vals):
        return pkt3(pm4.PACKET3_SET_SH_REG, reg.addr[0] - pm4.PACKET3_SET_SH_REG_START, *vals)

    def lo_hi(v): return [v & 0xFFFFFFFF, v >> 32]

    q = []
    # ACQUIRE_MEM (gli=0, gl2=0), exactly as exec()'s first operation.
    gcr_cntl = (1<<5)|(1<<4)|(1<<7)|(1<<6)|(1<<8)|(1<<9)
    q += pkt3(pm4.PACKET3_ACQUIRE_MEM, 0, 0xFFFFFFFF, 0xFFFFFFFF, 0, 0, 0, gcr_cntl)
    q += set_sh_reg(adev.regCOMPUTE_PGM_LO, *lo_hi(prog_addr >> 8))
    q += set_sh_reg(adev.regCOMPUTE_PGM_RSRC1, rsrc1, rsrc2)
    q += set_sh_reg(adev.regCOMPUTE_PGM_RSRC3, rsrc3)
    tmpring = 0  # scratch used only if private_segment_size > 0; this kernel has none in practice, but
                 # write 0 defensively -- if private_segment_size WAS > 0 this would need the real formula.
    if private_segment_size > 0:
        lanes, align = 64, 256
        spt = ((private_segment_size + (align//lanes) - 1) // (align//lanes)) * (align//lanes)
        sxcc = spt * lanes * max_slots_scratch_cu * cu_cnt
        wscr = -(-(lanes*spt) // align)
        nwaves = min((sxcc // (wscr*align)) // se_cnt, cu_cnt*max_slots_scratch_cu)
        from tinygrad.runtime.autogen import hsa
        tmpring = int.from_bytes(bytes(hsa.union_COMPUTE_TMPRING_SIZE_GFX11_bitfields(WAVES=nwaves, WAVESIZE=wscr)), 'little')
    q += set_sh_reg(adev.regCOMPUTE_TMPRING_SIZE, tmpring)
    q += set_sh_reg(adev.regCOMPUTE_DISPATCH_SCRATCH_BASE_LO, *lo_hi(scratch_area.va_addr >> 8))
    q += set_sh_reg(adev.regCOMPUTE_RESTART_X, 0, 0, 0)
    ud = ([*lo_hi(dispatch_ptr_va)] if enable_dispatch_ptr else []) + [*lo_hi(ka_va)]
    q += set_sh_reg(adev.regCOMPUTE_USER_DATA_0, *ud)
    q += set_sh_reg(adev.regCOMPUTE_RESOURCE_LIMITS, 0)
    q += set_sh_reg(adev.regCOMPUTE_START_X, 0, 0, 0, 1, 1, 1)  # local_size = (1,1,1)
    q += pkt3(pm4.PACKET3_DISPATCH_DIRECT, 1, 1, 1,
              adev.regCOMPUTE_DISPATCH_INITIATOR.encode(cs_w32_en=int(wave32), force_start_at_000=1, compute_shader_en=1))
    q += pkt3(pm4.PACKET3_RELEASE_MEM,
              pm4.PACKET3_RELEASE_MEM_EVENT_TYPE(pm4.CACHE_FLUSH_AND_INV_TS_EVENT) | pm4.PACKET3_RELEASE_MEM_EVENT_INDEX(5),
              pm4.PACKET3_RELEASE_MEM_DATA_SEL(1),
              *lo_hi(signal_va), 1, 0)
    q += pkt3(pm4.PACKET3_EVENT_WRITE, (7 << 0) | (4 << 8))  # CS_PARTIAL_FLUSH

    step(f"packet stream: {len(q)} dwords")

    # ---- submit: write ring, wptr, HDP flush, doorbell ----
    ring_dwords = RING_SZ // 4
    ring_vram_off = ring_area.paddrs[0][0]
    adev.vram.view(ring_vram_off, len(q) * 4, fmt='I')[:len(q)] = q
    wptr = len(q)
    adev.vram.view(ctrl_area.paddrs[0][0] + 8, 8, fmt='B')[:] = struct.pack('<Q', wptr)
    adev.wreg(hdp_flush_dword_idx, 0)
    adev.doorbell64.view(doorbell_byte_off, 8, fmt='B')[:] = struct.pack('<Q', wptr)

    step("dispatched — polling completion signal (10s timeout) ...")
    t0 = time.monotonic()
    sig = 0
    while time.monotonic() - t0 < 10:
        sig = struct.unpack('<Q', bytes(adev.vram.view(signal_vram_off, 8, fmt='B')[:]))[0]
        if sig >= 1: break
        time.sleep(0.02)

    if sig >= 1:
        result = struct.unpack('<f', bytes(adev.vram.view(out_buf_vram_off, 4, fmt='B')[:]))[0]
        step(f"PASS — signal={sig}, output={result} (expect 42.0)")
    else:
        step(f"FAIL — signal never reached 1 (stuck at {sig}) within 10s timeout")

    step("adev.fini() ...")
    adev.fini()
    step("done")


if __name__ == "__main__":
    main()
