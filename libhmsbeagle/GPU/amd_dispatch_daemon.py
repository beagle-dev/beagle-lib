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

Architecture change from amd_init_helper.py/GPUInterfaceTinyGPUHybridAMD.cpp:
Python now stays resident and handles EVERY GPU operation (compile, alloc,
memcpy, launch, sync), not just bring-up — the C++ side
(GPUInterfaceTinyGPUHybridAMD.cpp) becomes a thin RPC client. This trades
some per-call IPC overhead for using only code this session has verified
actually works on this hardware.

Protocol: newline-terminated JSON command lines on a dedicated socketpair
(NOT the TinyGPU socket — AMDDevice("AMD:0") makes its own connection to
TinyGPU.app internally, matching how the STATUS.md §8 reference test
connected, with no inherited FD needed). Commands that carry bulk data
(h2d/d2h) are followed immediately by that many raw bytes on the same
stream, avoiding base64 overhead. One JSON reply line per command
(h2d/d2h's reply line is followed by the reply's own raw bytes for d2h).

Kernel launches are batched (cmd_launch_batch, STATUS.md AMD §26): profiling
(BEAGLE_AMD_PROFILE=1) found steady-state per-launch RPC overhead (~150-190us)
comparable to or larger than the actual GPU dispatch work (~100us).
GPUInterfaceTinyGPUHybridAMD.cpp queues launches instead of sending each as
its own round-trip, and flushes the queue (one batched RPC call) before any
h2d/d2h/sync/fini. This is safe without any extra synchronization on either
side: prg(...) here always uses wait=False (just enqueues PM4 packets into
the ring, doesn't block), so flushing at those points preserves submission
order; and tinygrad's own HCQAllocator._copyin/_copyout/synchronize already
call self.dev.synchronize() internally before touching memory, so by the
time any h2d/d2h/sync actually reads or writes a buffer, every
already-flushed launch is guaranteed to have completed on the GPU.

    python3 amd_dispatch_daemon.py <cmd_sock_fd>
"""
import sys, os, json, struct, pathlib, ctypes, weakref, time

_TINYGRAD_PATH = os.environ.get("TINYGRAD_PATH", str(pathlib.Path.home() / "Dropbox/Projects/tinygrad"))
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
# round-trip profiling (GPUInterfaceTinyGPUHybridAMD.cpp) -- breaks down how
# much of that round-trip is real GPU work (the dispatch/sync/copy call
# itself) vs. JSON parsing and Python/socket overhead. See the user's own
# question this was built to answer: is host-side overhead here actually
# worth optimizing (and if so, where) before considering anything riskier.
_PROFILE = bool(os.environ.get("BEAGLE_AMD_PROFILE"))


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


class Daemon:
    def __init__(self, sock):
        self.sock = sock
        self.dev = None
        self.programs = {}   # (name, n_int_args) -> BeagleAMDProgram
        self.image = None    # last-compiled multi-kernel ELF (image, kernels dict of name->(kd_addr,desc))
        self.kernels = None
        self.hsaco = None

    # ── wire I/O ──────────────────────────────────────────────────────────
    def recv_line(self):
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = self.sock.recv(1)
            if not chunk:
                return None
            buf += chunk
        return buf.decode()

    def recv_exact(self, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise RuntimeError("socket closed mid-read")
            buf += chunk
        return bytes(buf)

    def send_json(self, obj):
        self.sock.sendall((json.dumps(obj) + "\n").encode())

    # ── commands ──────────────────────────────────────────────────────────
    def cmd_boot(self, req):
        amd_hcq_patch.set_logger(log)
        amd_hcq_patch.apply()
        log("amd_hcq_patch applied")
        DEV.value = "AMD"
        from tinygrad import Device
        self.dev = Device["AMD:0"]
        log(f"booted — {self.dev}")
        self.send_json({"ok": True, "arch": self.dev.arch})

    def cmd_compile_all(self, req):
        with open(req["cl_path"]) as f:
            src = f.read()
        # HIP language, not OpenCL -- see GPUImplDefs.h's FW_TINYGPU_HYBRID_AMD
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
        # GPUInterfaceTinyGPUHybridAMD.cpp flushes any queued launches before
        # every h2d/d2h/sync/fini, and tinygrad's own _copyin/_copyout/
        # synchronize already wait for prior submitted work internally --
        # see the module-level comment for why that makes this safe).
        debug_dump_target = os.environ.get("BEAGLE_AMD_DEBUG_DUMP")
        launches = req["launches"]
        with _Profiled(f"launch_batch({len(launches)})"):
            for i, item in enumerate(launches):
                kernel_name = item["kernel"]
                ptrs = item["ptrs"]
                ints = item["ints"]
                grid = item["grid"]
                block = item["block"]
                try:
                    if debug_dump_target == kernel_name:
                        self._debug_dump(kernel_name, ptrs, ints, grid, block)
                    prg = self._get_program(kernel_name, len(ints))
                    bufs = tuple(HCQBuffer(addr, 0) for addr in ptrs)
                    prg(*bufs, global_size=tuple(grid), local_size=tuple(block), vals=tuple(ints), wait=False)
                except Exception as e:
                    self.send_json({"ok": False, "error": f"launch_batch[{i}] {kernel_name}: {e}"})
                    return
        self.send_json({"ok": True, "count": len(launches)})

    def cmd_sync(self, req):
        with _Profiled("sync.synchronize"):
            self.dev.synchronize()
        self.send_json({"ok": True})

    def cmd_fini(self, req):
        self.send_json({"ok": True})

    def run(self):
        while True:
            line = self.recv_line()
            if line is None:
                break
            req = json.loads(line)
            cmd = req.get("cmd")
            try:
                getattr(self, f"cmd_{cmd}")(req)
            except Exception as e:
                import traceback
                traceback.print_exc(file=sys.stderr)
                self.send_json({"ok": False, "error": str(e)})
            if cmd == "fini":
                break


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <cmd_sock_fd>", file=sys.stderr)
        sys.exit(1)
    import socket
    cmd_fd = int(sys.argv[1])
    sock = socket.socket(fileno=cmd_fd)

    os.makedirs(os.path.expanduser("~/Library/Logs"), exist_ok=True)
    log_path = os.path.expanduser("~/Library/Logs/amd_dispatch_daemon.log")
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_SYNC, 0o644)
    sys.stderr = os.fdopen(fd, 'w', buffering=1)
    log(f"starting, cmd_fd={cmd_fd}")

    try:
        Daemon(sock).run()
    except Exception:
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    log("exiting cleanly")


if __name__ == "__main__":
    main()
