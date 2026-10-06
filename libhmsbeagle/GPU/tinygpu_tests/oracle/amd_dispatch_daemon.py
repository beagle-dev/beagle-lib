#!/usr/bin/env python3
"""
amd_dispatch_daemon.py — BEAGLE AMD hybrid backend, take 2.

Four hand-built PM4 dispatch attempts (two independent implementations,
across two rounds of real, mechanically-verified bug fixes) all crashed the
host identically (DART "read of DVA 0" panic — see STATUS.md AMD §3-§11).
The only thing that has ever worked on this hardware is stock, unmodified
tinygrad using the *full* AMDDevice/PCIIface/HCQCompiled stack (STATUS.md
§8) — never bare AMDev+setup_ring() in isolation, which is all the prior
attempts ever drove. This script stops hand-deriving the PM4 stream
entirely and drives dispatch through tinygrad's real AMDProgram/
HCQProgram.__call__ code instead.

Architecture change from amd_init_helper.py/GPUInterfaceTinyGPUAMD.cpp:
Python now stays resident and handles EVERY GPU operation (compile, alloc,
memcpy, launch, sync), not just bring-up — the C++ side
(GPUInterfaceTinyGPUAMD.cpp) becomes a thin RPC client. This trades
some per-call IPC overhead for using only code this session has verified
actually works on this hardware.

Protocol: JSON command messages, each preceded by its byte length as a
4-byte little-endian uint32 (the NV pair's framing), on a dedicated
socketpair. tinygrad runs over the plugin's own TinyGPU.app connection,
inherited as tgpu_fd (_install_inherited_tinygpu), not a second one of its
own. Commands that carry bulk data (h2d/d2h) are followed immediately by
that many raw bytes on the same stream, avoiding base64 overhead. One JSON
reply message per command (d2h's reply is followed by the reply's own raw
bytes).

Kernel launches are batched (cmd_launch_batch, STATUS.md AMD §26): profiling
(BEAGLE_AMD_PROFILE=1) found steady-state per-launch RPC overhead (~150-190us)
comparable to or larger than the actual GPU dispatch work (~100us).
GPUInterfaceTinyGPUAMD.cpp queues launches instead of sending each as
its own round-trip, and flushes the queue (one batched RPC call) before any
h2d/d2h/sync/fini. This is safe without any extra synchronization on either
side: launches here only enqueue PM4 packets into the ring and never block
(prg(..., wait=False), or by default one chained queue per batch), so
flushing at those points preserves submission order; and tinygrad's own HCQAllocator._copyin/_copyout/synchronize already
call self.dev.synchronize() internally before touching memory, so by the
time any h2d/d2h/sync actually reads or writes a buffer, every
already-flushed launch is guaranteed to have completed on the GPU.

    python3 amd_dispatch_daemon.py <cmd_sock_fd> [<tgpu_fd>]
"""
import sys, os, json, struct, pathlib, ctypes, weakref, time

# Default: the tinygrad worktree pinned at a9830e2b4 -- tinygrad HEAD
# (after 2026-09-05) dropped the macOS TinyGPU transport and hcq1 (TODO.md
# Phase 140). TINYGRAD_PATH overrides.
_TINYGRAD_PATH = os.environ.get("TINYGRAD_PATH", str(pathlib.Path.home() / "Dropbox/Projects/tinygrad-hcq1"))
sys.path.insert(0, _TINYGRAD_PATH)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tinygrad.helpers import DEV, round_up
from tinygrad.device import TinyELF, BufferSpec, Target
from tinygrad.dtype import dtypes
from tinygrad.runtime.support.hcq import HCQBuffer, HCQProgram, CLikeArgsState
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.runtime.autogen import amdgpu_kd, hsa

import amd_compile_helper as ach  # reuse the already-verified compile_opencl()/parse_kernels()
import amd_hcq_patch  # AMDComputeQueue.exec fixes: dispatch-ptr sgpr layout + hidden kernel args (STATUS.md AMD §15-17, §22-24)


def log(msg):
    print(f"[amd_dispatch_daemon] {msg}", file=sys.stderr, flush=True)


# Opt-in per-command timing (BEAGLE_AMD_PROFILE=1), matching the C++ side's
# round-trip profiling (GPUInterfaceTinyGPUAMD.cpp) -- breaks down how
# much of that round-trip is real GPU work (the dispatch/sync/copy call
# itself) vs. JSON parsing and Python/socket overhead. See the user's own
# question this was built to answer: is host-side overhead here actually
# worth optimizing (and if so, where) before considering anything riskier.
_PROFILE = bool(os.environ.get("BEAGLE_AMD_PROFILE"))

# One chained compute queue per launch_batch (default), at most _CHAIN_MAX
# launches per submit; see cmd_launch_batch.
_CHAIN_LAUNCHES = os.environ.get("BEAGLE_AMD_CHAIN_LAUNCHES", "1") != "0"
_CHAIN_MAX = 1024


class _Profiled:
    __slots__ = ("label", "t0")
    def __init__(self, label):
        self.label = label
    def __enter__(self):
        if _PROFILE:
            self.t0 = time.perf_counter()
        return self
    def __exit__(self, *exc):
        if _PROFILE:
            us = (time.perf_counter() - self.t0) * 1e6
            log(f"  [profile] {self.label:24s} {us:8.0f} us")


# ── BeagleAMDProgram: AMDProgram.__init__'s body, copied verbatim, with the
# ONE line that assumes a single kernel per ELF (rodata_entry = .rodata's own
# section address) replaced by the real per-kernel .kd symbol address
# amd_compile_helper.py's parse_kernels() already extracts correctly. Every
# other line is unmodified real tinygrad code -- this is a targeted patch to
# the one place AMDProgram's single-kernel-per-compile assumption doesn't
# hold for BEAGLE's one-big-multi-kernel-source compile, not a reimplementation. ──
class BeagleAMDProgram(HCQProgram):
    def __init__(self, dev, name, lib_bytes, kd_addr, n_int_args):
        self.dev, self.name, self.lib = dev, name, lib_bytes
        image, sections, relocs = elf_loader(self.lib)

        rodata_entry = kd_addr  # <-- the one substituted line; see class docstring above

        for apply_image_offset, rel_sym_offset, typ, addent in relocs:
            if typ == 5:
                image[apply_image_offset:apply_image_offset + 8] = struct.pack(
                    '<q', rel_sym_offset - apply_image_offset + addent)
            else:
                raise RuntimeError(f"unknown AMD reloc {typ}")

        self.lib_gpu = self.dev.allocator.alloc(round_up(image.nbytes, 0x1000), buf_spec := BufferSpec(nolru=True))
        self.dev.allocator._copyin(self.lib_gpu, image)
        self.dev.synchronize()

        desc_sz = ctypes.sizeof(amdgpu_kd.llvm_amdhsa_kernel_descriptor_t)
        desc = amdgpu_kd.llvm_amdhsa_kernel_descriptor_t.from_buffer_copy(bytes(image[rodata_entry:rodata_entry + desc_sz]))
        self.group_segment_size = desc.group_segment_fixed_size
        self.private_segment_size = desc.private_segment_fixed_size
        self.kernargs_segment_size = desc.kernarg_size
        lds_size = ((self.group_segment_size + 511) // 512) & 0x1FF
        if lds_size > (self.dev.iface.props['lds_size_in_kb'] * 1024) // 512:
            raise RuntimeError("Too many resources requested: group_segment_size")

        self.dev._ensure_has_local_memory(self.private_segment_size)

        self.wave32 = desc.kernel_code_properties & 0x400 == 0x400
        self.rsrc1 = desc.compute_pgm_rsrc1 | ((1 << 20) if self.dev.target[0] == 11 else 0)
        self.rsrc2 = desc.compute_pgm_rsrc2 | (lds_size << 15)
        self.rsrc3 = desc.compute_pgm_rsrc3
        self.aql_prog_addr = self.lib_gpu.va_addr + rodata_entry
        self.prog_addr = self.lib_gpu.va_addr + rodata_entry + desc.kernel_code_entry_byte_offset
        self.enable_dispatch_ptr = desc.kernel_code_properties & hsa.AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR
        self.enable_private_segment_sgpr = desc.kernel_code_properties & hsa.AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER
        if self.enable_private_segment_sgpr:
            raise RuntimeError(f"kernel {name} needs enable_private_segment_sgpr -- not implemented")
        # BEAGLE addition (not in upstream AMDProgram.__init__): comgr's OpenCL
        # compile of BEAGLE's kernels also enables these two SGPR features
        # (confirmed empirically -- kernel_code_properties=0x41e for even a
        # trivial get_global_id kernel, both OpenCL 1.2 and 2.0 language
        # modes), which upstream AMDComputeQueue.exec() never populates.
        # amd_hcq_patch.py reads these to fill the right USER_DATA slots.
        # See STATUS.md AMD §17 for the full SGPR-ordering bug this fixes.
        self.enable_queue_ptr = bool(desc.kernel_code_properties & hsa.AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_QUEUE_PTR)
        self.enable_dispatch_id = bool(desc.kernel_code_properties & hsa.AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_ID)
        additional_alloc_sz = ctypes.sizeof(hsa.hsa_kernel_dispatch_packet_t) if self.enable_dispatch_ptr else 0

        # obj.signature: n_int_args entries of uint32 -- matches BEAGLE's
        # KernelLauncher.cpp calling convention (all trailing scalar args are
        # unsigned int). Buffer args need no signature entry (fill_kernargs
        # only reads .va_addr off each buf, not the signature).
        signature = tuple((None, i, dtypes.uint32, ()) for i in range(n_int_args))
        obj = TinyELF(lib=lib_bytes, name=name, target=Target(), signature=signature)
        super().__init__(CLikeArgsState, self.dev, obj, kernargs_alloc_size=self.kernargs_segment_size + additional_alloc_sz,
                          base=self.lib_gpu.va_addr)
        weakref.finalize(self, self._fini, self.dev, self.lib_gpu, buf_spec)


def _install_inherited_tinygpu(tgpu_fd):
    """
    Run tinygrad over the plugin's own TinyGPU.app connection (inherited as
    tgpu_fd) instead of opening a second one: TinyGPU.app serves one client at
    a time, and a second connection's first RPC hung while the plugin's sat
    open (STATUS.md AMD §21). The plugin sends nothing on it while this
    daemon lives. nv_dispatch_daemon.py's (the NV oracle's) device: it also
    keeps a dup of each MAP_SYSMEM_FD fd, which hcq1's alloc_sysmem closes
    once mapped, so cmd_handoff can pass buffers to the C++ side.
    """
    import socket, mmap, itertools
    from tinygrad.helpers import ceildiv
    from tinygrad.runtime.support import system
    from tinygrad.runtime.support.hcq import FileIOInterface, MMIOInterface

    class BeagleTinyGPUDevice(system.APLRemotePCIDevice):
        def __init__(self, devpref, pcibus):
            # RemotePCIDevice.__init__ on the inherited connection, without APLRemotePCIDevice.__init__'s ensure_app and
            # connect (the plugin checked the app and connected) or the buffer sizes (the plugin set them when it connected,
            # and macOS refuses a second setting, ENOBUFS); the lock file as tinygrad takes it
            self.sock, self.pcibus, self.dev_id = socket.socket(fileno=os.dup(tgpu_fd)), "usb4", 0
            self.peer_group = self.sock.getpeername()[0]
            self.lock_fd = system.System.flock_acquire(f"{devpref.lower()}_usb4.lock")
            self.sysmem_fds = {}  # host address of a sysmem mapping -> (dup of its fd, mapped size)

        def alloc_sysmem(self, size, vaddr=0, contiguous=False):
            # APLRemotePCIDevice.alloc_sysmem, plus the fd dup
            mapped_size, _, _, fd = self._rpc(self.sock, self.dev_id, system.RemoteCmd.MAP_SYSMEM_FD, size, int(contiguous), has_fd=True)
            keep = os.dup(fd)
            memview = MMIOInterface(FileIOInterface(fd=fd).mmap(0, mapped_size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, 0),
                                    mapped_size, fmt='B')
            self.sysmem_fds[memview.addr] = (keep, mapped_size)
            paddrs_raw = list(itertools.takewhile(lambda p: p[1] != 0, zip(memview.view(fmt='Q')[0::2], memview.view(fmt='Q')[1::2])))
            return memview, [p + i for p, sz in paddrs_raw for i in range(0, sz, 0x1000)][:ceildiv(size, 0x1000)]

    system.APLRemotePCIDevice = BeagleTinyGPUDevice  # System.list_devices looks the name up at call time


class Daemon:
    def __init__(self, sock, tgpu_fd=None):
        self.sock = sock
        self.tgpu_fd = tgpu_fd
        self.handed_off = False
        self.dev = None
        self.programs = {}   # (name, n_int_args) -> BeagleAMDProgram
        self.image = None    # last-compiled multi-kernel ELF (image, kernels dict of name->(kd_addr,desc))
        self.kernels = None
        self.hsaco = None

    # ── wire I/O: each JSON message is preceded by its length as a 4-byte
    # little-endian uint32, so a message is two reads instead of one recv()
    # per byte ─────────────────────────────────────────────────────────────
    def recv_msg(self):
        hdr = self.sock.recv(4)
        if not hdr:
            return None
        if len(hdr) < 4:
            hdr += self.recv_exact(4 - len(hdr))
        return self.recv_exact(struct.unpack("<I", hdr)[0])

    def recv_exact(self, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise RuntimeError("socket closed mid-read")
            buf += chunk
        return bytes(buf)

    def send_json(self, obj):
        body = json.dumps(obj).encode()
        self.sock.sendall(struct.pack("<I", len(body)) + body)

    # ── commands ──────────────────────────────────────────────────────────
    def cmd_boot(self, req):
        amd_hcq_patch.set_logger(log)
        amd_hcq_patch.apply()
        log("amd_hcq_patch applied")
        if self.tgpu_fd is not None:
            _install_inherited_tinygpu(self.tgpu_fd)
            log(f"tinygrad uses the plugin's TinyGPU.app connection (fd {self.tgpu_fd})")
        DEV.value = "AMD"
        from tinygrad import Device
        self.dev = Device["AMD:0"]
        log(f"booted — {self.dev}")
        log("launch_batch: " + (f"one chained queue per batch, at most {_CHAIN_MAX} launches per submit (compute ring "
                                 f"{len(self.dev.compute_queue.ring) * 4} bytes)" if _CHAIN_LAUNCHES else
                                 "one queue per launch (BEAGLE_AMD_CHAIN_LAUNCHES=0)"))
        self.send_json({"ok": True, "arch": self.dev.arch})

    def cmd_compile_all(self, req):
        with open(req["cl_path"]) as f:
            src = f.read()
        # HIP language, not OpenCL -- see GPUImplDefs.h's FW_TINYGPU_AMD
        # branch and STATUS.md AMD §18: OpenCL's get_global_id() pulls in
        # dispatch_ptr/queue_ptr/dispatch_id sgprs and heavy scratch usage
        # this remote transport can't drive correctly; HIP's group/local-id
        # builtins need none of that. Confirmed offline against the full
        # real kernel set (all 9 state counts x SP/DP): 0/80 kernels ever
        # set dispatch_ptr/queue_ptr/dispatch_id.
        self.hsaco = ach.compile_hip(src, self.dev.arch)
        self.image, self.kernels = ach.parse_kernels(self.hsaco)
        log(f"compiled — {len(self.kernels)} kernels, image={len(self.image):#x} bytes")
        self.send_json({"ok": True, "kernels": list(self.kernels.keys())})

    def _get_program(self, name, n_int_args):
        key = (name, n_int_args)
        if key not in self.programs:
            kd_addr, desc = self.kernels[name]
            self.programs[key] = BeagleAMDProgram(self.dev, name, self.hsaco, kd_addr, n_int_args)
        return self.programs[key]

    def cmd_alloc(self, req):
        buf = self.dev.allocator.alloc(req["size"])
        self.send_json({"ok": True, "addr": buf.va_addr})
        # Keep a reference so it isn't garbage-collected/freed.
        self._allocs = getattr(self, "_allocs", {})
        self._allocs[buf.va_addr] = buf

    def cmd_h2d(self, req):
        n = req["size"]
        data = self.recv_exact(n)
        buf = HCQBuffer(req["addr"], n)
        with _Profiled(f"h2d._copyin({n}B)"):
            self.dev.allocator._copyin(buf, memoryview(bytearray(data)))
        self.send_json({"ok": True})

    def cmd_d2h(self, req):
        n = req["size"]
        buf = HCQBuffer(req["addr"], n)
        out = memoryview(bytearray(n))
        with _Profiled(f"d2h._copyout({n}B)"):
            self.dev.allocator._copyout(out, buf)
        self.send_json({"ok": True, "size": n})
        self.sock.sendall(bytes(out))

    def _debug_dump(self, kernel_name, ptrs, ints, grid, block):
        # Opt-in diagnostic (STATUS.md AMD §22-23): dumps the first 4 floats
        # of every pointer arg right before the named kernel launches. Set
        # BEAGLE_AMD_DEBUG_DUMP=<kernel name> to use it.
        self.dev.synchronize()  # make sure prior uploads/launches are visible
        for i, addr in enumerate(ptrs):
            raw = memoryview(bytearray(16))
            self.dev.allocator._copyout(raw, HCQBuffer(addr, 16))
            vals = struct.unpack('<4f', bytes(raw))
            log(f"  [debug_dump] {kernel_name} ptr[{i}] @ {addr:#x}: first 4 floats = {vals}")
        log(f"  [debug_dump] {kernel_name} ints = {ints} grid={grid} block={block}")

    def cmd_launch_batch(self, req):
        # Batches a sequence of kernel launches into one RPC round-trip
        # (STATUS.md AMD §26): the per-call socket+JSON overhead (~150-190us,
        # measured via BEAGLE_AMD_PROFILE) was comparable to or larger than
        # the actual GPU dispatch work (~100us) in steady state. Each item
        # here is dispatched exactly as cmd_launch (removed, superseded by
        # this) used to -- same _get_program/prg(...) calls, same wait=False
        # (ordering relative to h2d/d2h/sync is preserved on the C++ side:
        # GPUInterfaceTinyGPUAMD.cpp flushes any queued launches before
        # every h2d/d2h/sync/fini, and tinygrad's own _copyin/_copyout/
        # synchronize already wait for prior submitted work internally --
        # see the module-level comment for why that makes this safe).
        #
        # By default the whole batch is one compute queue: one timeline wait
        # and memory barrier, then each kernel's exec, which AMDComputeQueue.exec
        # (amd_hcq_patch's, with the hidden kernel arguments) opens with a cache
        # acquire and closes with a CS_PARTIAL_FLUSH, so each kernel starts after
        # the previous one finished (hcq1's HCQGraph relies on the same in-queue
        # ordering between dependent kernels), then one signal and one submit.
        # That is HCQProgram.__call__ with N execs instead of one, and one
        # doorbell per batch instead of per launch (TODO.md plan step A0, as the
        # NV oracle's launch_batch). hcq1's PM4 _submit never checks the ring's
        # read pointer, so a chain is submitted every _CHAIN_MAX launches (an
        # exec is under 64 dwords: under 256 KB per submit, of the 16 MB ring).
        # BEAGLE_AMD_CHAIN_LAUNCHES=0 submits each launch on its own queue, as
        # before.
        debug_dump_target = os.environ.get("BEAGLE_AMD_DEBUG_DUMP")
        launches = req["launches"]
        q = None
        with _Profiled(f"launch_batch({len(launches)})"):
            for i, item in enumerate(launches):
                kernel_name = item["kernel"]
                ptrs = item["ptrs"]
                ints = item["ints"]
                grid = item["grid"]
                block = item["block"]
                try:
                    if debug_dump_target == kernel_name:
                        q = self._submit_chain(q)  # the dump's synchronize must see every earlier launch
                        self._debug_dump(kernel_name, ptrs, ints, grid, block)
                    prg = self._get_program(kernel_name, len(ints))
                    bufs = tuple(HCQBuffer(addr, 0) for addr in ptrs)
                    if not _CHAIN_LAUNCHES:
                        prg(*bufs, global_size=tuple(grid), local_size=tuple(block), vals=tuple(ints), wait=False)
                        continue
                    if q is None:
                        q = self.dev.hw_compute_queue_t().wait(self.dev.timeline_signal, self.dev.timeline_value - 1).memory_barrier()
                    q.exec(prg, prg.fill_kernargs(bufs, tuple(ints)), tuple(grid), tuple(block))
                    if (i + 1) % _CHAIN_MAX == 0:
                        q = self._submit_chain(q)
                except Exception as e:
                    self.send_json({"ok": False, "error": f"launch_batch[{i}] {kernel_name}: {e}"})
                    return
            self._submit_chain(q)
        self.send_json({"ok": True, "count": len(launches)})

    def _submit_chain(self, q):
        # HCQProgram.__call__'s ending, once for the chained queue
        if q is not None:
            q.signal(self.dev.timeline_signal, self.dev.next_timeline()).submit(self.dev)
        return None

    def cmd_sync(self, req):
        with _Profiled("sync.synchronize"):
            self.dev.synchronize()
        self.send_json({"ok": True})

    def cmd_handoff(self, req):
        # TODO.md plan step A1e, the boot-only handoff: from here the C++ side (TinyGPUAMDRuntime.h) owns the GPU's
        # queues, and this daemon only waits for fini (run() refuses everything else). After a synchronize, it allocates
        # the C++ side's VRAM pool (pool_size, default half the VRAM) and a 16 MB staging buffer, then replies with flat
        # JSON: the sysmem mappings (sizes, in the order of the fds) and each object's (mapping, offset) in them, the
        # GPU addresses the packets name, the queues' doorbells and put_values, the BAR sizes, the MMIO register addresses
        # (discovered bases) the C++ side reads and writes, and the props. Then the HSACO compile_all compiled (blob_size
        # bytes; none when the C++ side has the build's own, plan step A1j), then the mappings' fds over SCM_RIGHTS.
        import socket as _socket
        from tinygrad.device import BufferSpec
        if self.handed_off: raise RuntimeError("handoff: already handed off")
        dev, adev = self.dev, self.dev.iface.dev_impl
        pci = dev.iface.pci_dev
        if not hasattr(pci, "sysmem_fds"): raise RuntimeError("handoff needs the plugin's TinyGPU.app connection (tgpu_fd)")
        dev.synchronize()
        pool = dev.allocator.alloc(int(req.get("pool_size") or adev.vram_size // 2), BufferSpec(nolru=True))
        staging = dev.allocator.alloc(16 << 20, BufferSpec(host=True, nolru=True))
        self._handoff_bufs = (pool, staging)   # never freed: the C++ side uses them until fini
        maps, info = [], {"ok": True}
        def place(key, addr):
            for base, (fd, size) in pci.sysmem_fds.items():
                if base <= addr < base + size:
                    if (fd, size) not in maps: maps.append((fd, size))
                    info[f"{key}_map"], info[f"{key}_off"] = maps.index((fd, size)), addr - base
                    return
            raise RuntimeError(f"handoff: {key} at host address {addr:#x} is in no sysmem mapping")
        for key, q in (("compute", dev.compute_queue), ("sdma", dev.sdma_queue(0))):
            if q.doorbell.residx != 2: raise RuntimeError(f"handoff: the {key} doorbell is not on BAR2")
            place(f"{key}_ring", q.ring.addr)
            place(f"{key}_rptr", q.read_ptr.addr)
            place(f"{key}_wptr", q.write_ptr.addr)
            info.update({f"{key}_ring_size": q.ring.nbytes, f"{key}_doorbell": q.doorbell.off, f"{key}_put": q.put_value})
        for key, sig in (("signal", dev.timeline_signal), ("shadow", dev._shadow_timeline_signal)):
            place(key, sig.base_buf.cpu_view().addr)
            info[f"{key}_va"] = sig.value_addr
        for key, buf in (("kargs", dev.kernargs_buf), ("staging", staging)):
            place(key, buf.cpu_view().addr)
            info.update({f"{key}_va": buf.va_addr, f"{key}_size": buf.size})
        info.update({"pool_va": pool.va_addr, "pool_size": pool.size, "timeline_value": dev.timeline_value, "vram_size": adev.vram_size,
                     "target_major": dev.target[0], "xccs": dev.xccs, "cu_cnt": dev.cu_cnt, "se_cnt": dev.se_cnt,
                     "max_slots_scratch_cu": dev.iface.props["max_slots_scratch_cu"], "lds_size_in_kb": dev.iface.props["lds_size_in_kb"],
                     "ih_ring_paddr": adev.ih.rings[0][0], "ih_ring_size": adev.ih.ring_size, "is_vf": int(adev.is_vf)})
        for bar in (0, 2, 5): info[f"bar{bar}_size"] = pci.bar_info(bar)[1]
        for key, reg in (("reg_hdp_remap", "regBIF_BX0_REMAP_HDP_MEM_FLUSH_CNTL"), ("reg_ih_wptr", "regIH_RB_WPTR"), ("reg_ih_rptr", "regIH_RB_RPTR"),
                         ("reg_ih_cntl", "regIH_RB_CNTL"), ("reg_fault_status", adev.gmc.pf_status_reg("GC")),
                         ("reg_fault_addr_lo", "regGCVM_L2_PROTECTION_FAULT_ADDR_LO32"), ("reg_fault_addr_hi", "regGCVM_L2_PROTECTION_FAULT_ADDR_HI32"),
                         ("reg_fault_cntl", "regGCVM_L2_PROTECTION_FAULT_CNTL")):
            info[key] = adev.reg(reg).addr[0]
        blob = self.hsaco or b""
        info.update({"nmaps": len(maps), "blob_size": len(blob)})
        for i, (_fd, size) in enumerate(maps): info[f"map{i}_size"] = size
        self.handed_off = True
        log(f"handoff: the C++ side owns the queues from here (VRAM pool {pool.size >> 20} MiB at {pool.va_addr:#x}, {len(maps)} mappings, "
            f"timeline {dev.timeline_value})")
        self.send_json(info)
        self.sock.sendall(blob)
        _socket.send_fds(self.sock, [b"F"], [fd for fd, _size in maps])

    def cmd_fini(self, req):
        self.send_json({"ok": True})

    def run(self):
        while True:
            msg = self.recv_msg()
            if msg is None:
                break
            req = json.loads(msg)
            cmd = req.get("cmd")
            try:
                if self.handed_off and cmd != "fini":   # two writers would corrupt the queues' put_values and the timeline
                    raise RuntimeError(f"{cmd}: the GPU queues belong to the C++ side after handoff")
                getattr(self, f"cmd_{cmd}")(req)
            except Exception as e:
                import traceback
                traceback.print_exc(file=sys.stderr)
                self.send_json({"ok": False, "error": str(e)})
            if cmd == "fini":
                break


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <cmd_sock_fd> [<tgpu_fd>]", file=sys.stderr)
        sys.exit(1)
    import socket
    cmd_fd = int(sys.argv[1])
    tgpu_fd = int(sys.argv[2]) if len(sys.argv) > 2 else None
    sock = socket.socket(fileno=cmd_fd)

    os.makedirs(os.path.expanduser("~/Library/Logs"), exist_ok=True)
    log_path = os.path.expanduser("~/Library/Logs/amd_dispatch_daemon.log")
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_SYNC, 0o644)
    sys.stderr = os.fdopen(fd, 'w', buffering=1)
    log(f"starting, cmd_fd={cmd_fd} tgpu_fd={tgpu_fd}")

    try:
        Daemon(sock, tgpu_fd).run()
    except Exception:
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    log("exiting cleanly")


if __name__ == "__main__":
    main()
