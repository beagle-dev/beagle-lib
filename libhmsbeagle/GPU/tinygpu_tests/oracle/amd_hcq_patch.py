#!/usr/bin/env python3
"""
amd_hcq_patch.py — targeted workaround for one real tinygrad bug hit on this
transport (STATUS.md AMD §15).

AMDComputeQueue.exec() (tinygrad/runtime/ops_amd.py) handles a kernel's
enable_dispatch_ptr=True case by writing workgroup/grid size through
disp_buf.cpu_view() (the transport-aware indirection every other buffer
access in this codebase goes through) -- but then writes
group_segment_size/private_segment_size/kernarg_address through a *raw*
ctypes struct overlay instead: `dp = hsa_kernel_dispatch_packet_t.from_address(
int(disp_buf.va_addr))`, `dp.group_segment_size = ...`. from_address()
dereferences its argument as a literal pointer valid in *this* process's
address space -- true on Linux with a real PCIe BAR mmap, false for our
remote-socket (TinyGPU.app/APLRemotePCIDevice) transport, where va_addr is a
GPU-side virtual address with no local meaning. Confirmed by hardware test
(amd_dispatch_ptr_test.py): segfaults every time, at exactly that line
(ops_amd.py:338, inside the `dp.group_segment_size = ...` assignment; a
Fatal Python error with a full C-level traceback via faulthandler pinned it
precisely).

Every real BEAGLE kernel calls get_global_id/get_local_id (enable_dispatch_
ptr=True), so this is not an edge case for us -- it is hit on every single
launch. Fix: monkeypatch AMDComputeQueue.exec with the same method body,
those three field writes redirected through bind_sints()/.cpu_view(),
exactly like the two field writes immediately above them in the original
(workgroup_size_x, grid_size_x) already do. Nothing else changes; every
other line here is copied verbatim from ops_amd.py's exec().

That fix alone got past the segfault (STATUS.md AMD §16) but real dispatch
then hit a GPU-side UTCL2_FAULT/SQ MEMVIOL -- the shader executed and faulted.
Note that .cpu_view() is *not* some transport-aware indirection either --
HCQBuffer.cpu_view() just returns HCQBuffer.view, an MMIOInterface built (by
whoever allocated the buffer) around to_mv(), i.e. the exact same
ctypes.from_address() mechanism as upstream's raw dp.field=value writes. The
difference -- and the reason those writes work while upstream's don't -- is
which *address* gets dereferenced: .view carries a real, separately-tracked
host-mapped pointer set once by the allocator, independent of va_addr (the
GPU-side address); upstream's from_address(int(disp_buf.va_addr)) ended up
using va_addr itself as if it were a host pointer, which it is not on this
transport. So the pointer-choice fix is correct and confirmed necessary,
just not sufficient on its own.

Neither upstream nor this patch's first version ever initializes the rest of
the dispatch packet (header/setup/kernel_object/reserved/completion_signal
are simply never written) or zeroes the region first -- on Linux, fresh
kfd/BO allocations are typically zero-filled by the kernel driver, so
`setup` (bits[0:1] = NDRange dimensionality per HSA_KERNEL_DISPATCH_PACKET_
SETUP_DIMENSIONS, autogen/hsa.py) reading as 0 may happen to not matter (or
happens to be a coincidentally-harmless value) there; whether it's
zero-filled the same way over TinyGPU.app's remote allocator is unverified.
Since get_global_id's compiled implementation reads through dispatch_ptr,
an unset/garbage `setup` is a plausible source of a bogus computed index ->
out-of-bounds write -> exactly the MEMVIOL observed. This version zeroes the
whole packet before writing the fields we know about, and sets setup=3
(declare full 3D; HSA allows over-declaring dimensions the kernel never
queries) as a defensive fix for that specific gap, plus logs the packet
contents so a future run can confirm or rule this out directly.

    import amd_hcq_patch; amd_hcq_patch.apply()   # call once, before any dispatch
"""
import sys, ctypes
from tinygrad.runtime.ops_amd import AMDComputeQueue, data64_le, EVENT_INDEX_PARTIAL_FLUSH, getenv
from tinygrad.runtime.autogen import hsa


def _patched_exec(self, prg, args_state, global_size, local_size):
    self.bind_args_state(args_state)

    self.acquire_mem(gli=0, gl2=0)

    # -- BEAGLE fix (STATUS.md AMD §22-24): __ockl_get_num_groups() (and,
    # defensively, __ockl_get_local_size()) read from the OpenCL/HIP "hidden
    # kernel arguments" block -- a fixed 256-byte struct the compiler
    # appends after the kernel's own explicit args (rounded up to 8 bytes),
    # completely separate from dispatch_ptr/enable_dispatch_ptr (confirmed:
    # this shows up even when kernel_code_properties has no dispatch_ptr bit
    # set at all). Neither upstream tinygrad's fill_kernargs/CLikeArgsState
    # nor this patch's own dispatch_ptr handling below ever populates it.
    # Confirmed as the cause of kernelMatrixMulADB's NaN output via a
    # hardware test (amd_2d_test.py) that reproduced it in isolation:
    # __ockl_get_num_groups(1) returned 0 instead of the correct value 1,
    # corrupting kernelMatrixMulADB's BLOCKS/EDGE computation into reading
    # 4 elements past the end of its 16-element shared-memory arrays. The
    # exact layout (hidden_block_count_x/y/z, u32 each, at offsets 0/4/8 of
    # the hidden block) is the well-established, stable part of this ABI;
    # group_size_x/y/z (u16 each, offsets 12/14/16) is written defensively
    # alongside it since it's part of the same struct and untested in
    # isolation, at negligible cost. The hidden block is only present when
    # the compiler actually needs it (kernargs_segment_size > where the
    # explicit args end) -- writing unconditionally would corrupt an
    # adjacent allocation (the dispatch packet, or just overrun the buffer)
    # for the many kernels that don't need it.
    explicit_bytes = len(args_state.bufs) * 8 + len(args_state.vals) * 4
    hidden_offset = -(-explicit_bytes // 8) * 8  # round up to a multiple of 8
    if prg.kernargs_segment_size > hidden_offset:
        hbuf = args_state.buf.offset(hidden_offset)
        self.bind_sints_to_mem(*global_size, mem=hbuf.cpu_view(), fmt='I', offset=0)
        self.bind_sints_to_mem(*local_size, mem=hbuf.cpu_view(), fmt='H', offset=12)
        if _LOG is not None:
            _LOG(f"[amd_hcq_patch] hidden args @ offset {hidden_offset:#x}: "
                 f"block_count={global_size} group_size={local_size}")

    user_regs = []
    if prg.enable_private_segment_sgpr:
        assert self.dev.xccs == 1, "Only architected flat scratch is supported on multi-xcc"
        scratch_hilo = data64_le(prg.dev.scratch.va_addr)
        user_regs = [scratch_hilo[0], scratch_hilo[1] | 1 << 31, 0xffffffff, 0x20c14000]

    disp_buf = None
    if prg.enable_dispatch_ptr:
        dp_t = hsa.hsa_kernel_dispatch_packet_t
        disp_buf = args_state.buf.offset(prg.kernargs_segment_size)
        dp_size = ctypes.sizeof(dp_t)

        # -- BEAGLE fix vs. upstream (see module docstring): zero the whole
        # packet first (upstream never initializes header/setup/kernel_object/
        # reserved/completion_signal at all), then write every field --
        # including setup (NDRange dimensionality) -- through the same
        # .cpu_view()-backed bind_sints() the two size fields already use,
        # instead of upstream's raw dp.field = value struct overlay onto
        # va_addr (a GPU-side address, not a valid host pointer here). --
        disp_buf.cpu_view().view(offset=0, size=dp_size, fmt='B')[:] = bytes(dp_size)
        self.bind_sints(3, mem=disp_buf.cpu_view(), struct_t=dp_t, start_field='setup', fmt='H')
        self.bind_sints(*local_size, mem=disp_buf.cpu_view(), struct_t=dp_t, start_field='workgroup_size_x', fmt='H')
        self.bind_sints(*[g * l for g, l in zip(global_size, local_size)], mem=disp_buf.cpu_view(), struct_t=dp_t,
                         start_field='grid_size_x', fmt='I')
        self.bind_sints(prg.group_segment_size, mem=disp_buf.cpu_view(), struct_t=dp_t, start_field='group_segment_size', fmt='I')
        self.bind_sints(prg.private_segment_size, mem=disp_buf.cpu_view(), struct_t=dp_t, start_field='private_segment_size', fmt='I')
        self.bind_sints(args_state.buf.va_addr, mem=disp_buf.cpu_view(), struct_t=dp_t, start_field='kernarg_address', fmt='Q')
        user_regs += [*data64_le(disp_buf.va_addr)]

        if _LOG is not None:
            raw = bytes(disp_buf.cpu_view().view(offset=0, size=dp_size, fmt='B')[:])
            _LOG(f"[amd_hcq_patch] dispatch packet @ {disp_buf.va_addr:#x} ({dp_size} bytes): {raw.hex()}")
            _LOG(f"[amd_hcq_patch] global_size={global_size} local_size={local_size} "
                 f"group_segment_size={prg.group_segment_size} private_segment_size={prg.private_segment_size} "
                 f"kernargs_segment_size={prg.kernargs_segment_size} kernarg_address={args_state.buf.va_addr:#x}")

    # -- BEAGLE addition vs. upstream (see module docstring): comgr's OpenCL
    # compile of BEAGLE's kernels also enables QUEUE_PTR and DISPATCH_ID
    # sgprs (kernel_code_properties=0x41e -- confirmed empirically, both
    # OpenCL 1.2 and 2.0 language modes), which upstream's exec() never
    # populates at all. Per the AMDGPU ABI, enabled sgpr-pairs are packed
    # into COMPUTE_USER_DATA_0.. in bit order: private_segment_buffer,
    # dispatch_ptr, queue_ptr, kernarg_segment_ptr, dispatch_id, ... .
    # Upstream only ever emitted dispatch_ptr then kernarg_segment_ptr
    # back-to-back, so with queue_ptr also enabled, kernarg_segment_ptr's
    # value actually landed in the queue_ptr slot, and nothing filled the
    # real kernarg_segment_ptr or dispatch_id slots -- confirmed as the
    # cause of the UTCL2_FAULT/SQ MEMVIOL (STATUS.md AMD §17). queue_ptr is
    # conventionally a real HSA queue pointer, but neither get_global_id nor
    # any BEAGLE kernel does device-side enqueue, so it's never expected to
    # be dereferenced; point it at any already-valid, mapped address rather
    # than 0, purely so a kernel that *did* read it wouldn't null-fault.
    # dispatch_id is a plain 64-bit counter value, not a pointer -- 0 is a
    # valid value, no dereference risk either way.
    if getattr(prg, 'enable_queue_ptr', False):
        placeholder = disp_buf.va_addr if disp_buf is not None else args_state.buf.va_addr
        user_regs += [*data64_le(placeholder)]

    user_regs += [*data64_le(args_state.buf.va_addr)]

    if getattr(prg, 'enable_dispatch_id', False):
        user_regs += [*data64_le(0)]

    if prg.dev.sqtt_enabled: self.sqtt_setup_exec(prg, global_size)

    self.wreg(self.gc.regCOMPUTE_PGM_LO, *data64_le(prg.prog_addr >> 8))
    self.wreg(self.gc.regCOMPUTE_PGM_RSRC1, prg.rsrc1, prg.rsrc2)
    self.wreg(self.gc.regCOMPUTE_PGM_RSRC3, prg.rsrc3)
    self.wreg(self.gc.regCOMPUTE_TMPRING_SIZE, prg.dev.tmpring_size)

    # this is what llvm refers to as "architected flat scratch"
    for xcc_id in range(self.dev.xccs):
        with self.pred_exec(xcc_mask=1 << xcc_id):
            scratch_base = prg.dev.scratch.va_addr + (prg.dev.scratch.size // self.dev.xccs * xcc_id)
            self.wreg(self.gc.regCOMPUTE_DISPATCH_SCRATCH_BASE_LO, *data64_le(scratch_base >> 8))

    self.wreg(self.gc.regCOMPUTE_RESTART_X, 0, 0, 0)
    self.wreg(self.gc.regCOMPUTE_USER_DATA_0, *user_regs)
    self.wreg(self.gc.regCOMPUTE_RESOURCE_LIMITS, waves_per_sh=getenv("WAVES_PER_SH"))
    self.wreg(self.gc.regCOMPUTE_START_X, 0, 0, 0, *local_size, 0, 0)

    self.pkt3(self.pm4.PACKET3_DISPATCH_DIRECT, *global_size,
              self.gc.regCOMPUTE_DISPATCH_INITIATOR.encode(**({'cs_w32_en': int(prg.wave32)} if prg.dev.target[0] != 9 else {}),
                                                             force_start_at_000=1, compute_shader_en=1))

    if prg.dev.sqtt_enabled:
        self.pkt3(self.pm4.PACKET3_EVENT_WRITE, self.pm4.EVENT_TYPE(self.soc.THREAD_TRACE_MARKER) | self.pm4.EVENT_INDEX(0))
    self.pkt3(self.pm4.PACKET3_EVENT_WRITE, self.pm4.EVENT_TYPE(self.soc.CS_PARTIAL_FLUSH) | self.pm4.EVENT_INDEX(EVENT_INDEX_PARTIAL_FLUSH))
    return self


_applied = False
_LOG = None


def set_logger(fn):
    """Optional: fn(str) called with a diagnostic line every dispatch-ptr launch. Call before apply()/dispatch."""
    global _LOG
    _LOG = fn


def apply():
    global _applied
    if _applied:
        return
    AMDComputeQueue.exec = _patched_exec
    _applied = True
