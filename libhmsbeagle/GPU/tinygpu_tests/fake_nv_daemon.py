"""Stand-in for nv_dispatch_daemon.py: the real Daemon wire protocol and
launch_batch paths, over a fake device, to test GPUInterfaceTinyGPUHybridNV.cpp
end to end without booting the GPU. Kernels don't run, so logL is wrong.
With a TinyGPU fd argument (C++ dispatch) it also hands off, using the real
build_handoff over file-backed buffers in $FAKE_NV_MEM, which
fake_tinygpu_server.py maps to play the GPU; allocations then come from a
file-backed "VRAM" there too. For the C++ runtime (handoff with "programs": false) it
hands over no programs; instead it sends the real ptxas cubin of the plugin's PTX (looked up in the
harness cubin cache by the PTX's sha256, tgpaths.cubin_path; run_goldens.sh fills it), the runtime keys
of an RTX 4060-like device, and the file-backed VRAM as the pool."""
import os, sys, re, json, socket, types, mmap, ctypes
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tgpaths
tgpaths.setup()
import nv_dispatch_daemon as d
import nv_compile_helper as nch
from tinygrad.helpers import round_up
from tinygrad.runtime.support.hcq import MMIOInterface

MEM = os.environ.get("FAKE_NV_MEM", "")
BUFS = (("cmdq", 0x10_1000_0000, 2 << 20), ("kargs", 0x10_2000_0000, 16 << 20),
        ("staging", 0x10_4000_0000, 16 << 20), ("signal", 0x10_3000_0000, 0x4000))
VRAM_VA, VRAM_SIZE = 0x20_0000_0000, 1 << 30

class FakeQueue:
    def wait(self, sig, val): return self
    def memory_barrier(self): return self
    def exec(self, prg, kernargs, gs, ls): return self
    def signal(self, sig, val): return self
    def submit(self, dev): return self

class FakeAllocator:
    def __init__(self): self.allocs, self.next = [], 0x100000
    def alloc(self, size):
        b = type("B", (), {})(); b.va_addr = self.next
        self.allocs.append((self.next, bytearray(size))); self.next += (size + 0xffff) & ~0xffff
        return b
    def _find(self, addr, n):
        for base, mem in self.allocs:
            if base <= addr and addr + n <= base + len(mem): return mem, addr - base
        raise RuntimeError(f"fake: no allocation holds [{addr:#x}, +{n})")
    def _copyin(self, buf, mv): mem, off = self._find(buf.va_addr, len(mv)); mem[off:off + len(mv)] = mv
    def _copyout(self, mv, buf): mem, off = self._find(buf.va_addr, len(mv)); mv[:] = mem[off:off + len(mv)]

UNLOAD_DIAG = {"unload_ok": True, "mailbox0": 0x80000000, "riscv_cpuctl": 0, "wpr2_lo": 0x1ff00, "wpr2_hi": 0x1ff10}  # suspended, WPR2 still up

class FakeDev:
    def __init__(self):
        self.allocator, self.timeline_signal, self.timeline_value = FakeAllocator(), object(), 1
        self.iface = types.SimpleNamespace(dev_impl=types.SimpleNamespace())
        self.iface.dev_impl.gsp = types.SimpleNamespace(fini_hw=self._gsp_fini_hw)
    def _gsp_fini_hw(self):   # a hung fini's unload RPC only: a clean GSP unload, no teardown
        open(f"{MEM}/fini", "w").close()   # from here on the fake GPU flags any TinyGPU.app traffic (plan step P3)
        self.iface.dev_impl.beagle_fini = dict(UNLOAD_DIAG)
    def finalize(self):   # a clean GSP unload (nv_init_helper's report); unless BEAGLE_NV_TEARDOWN=0 also a successful teardown
        open(f"{MEM}/fini", "w").close()
        self.iface.dev_impl.beagle_fini = dict(UNLOAD_DIAG)
        if os.environ.get("BEAGLE_NV_TEARDOWN", "1") != "0":
            self.iface.dev_impl.beagle_fini.update(teardown={"result": "done: Booter Unload lowered WPR2", "booter_mailbox0": 0},
                                                   wpr2_lo=0, wpr2_hi=0, wpr2_down=True, teardown_ok=True)
    def hw_compute_queue_t(self): return FakeQueue()
    def next_timeline(self): self.timeline_value += 1; return self.timeline_value - 1
    def synchronize(self): pass

class FakePrg:
    def check_launch(self, gs, ls): pass
    def set_launch_dims(self, gs, ls): pass
    def fill_kernargs(self, bufs, vals): return None
    def __call__(self, *bufs, **kw): pass

def handoff_dev():  # an Ada device as build_handoff sees it; the fake server reads the same numbers back
    fifo = lambda off, token: types.SimpleNamespace(ring=types.SimpleNamespace(residx=1, off=off), entries_count=0x10000, token=token,
                                                    gpput=types.SimpleNamespace(residx=1, off=off + 0x8008c), put_value=5)
    return types.SimpleNamespace(iface=types.SimpleNamespace(compute_class=d.ops_nv.nv_gpu.ADA_COMPUTE_A), slm_per_thread=0x800,
                                 compute_gpfifo=fifo(0x100000, 0x11), dma_gpfifo=fifo(0x200000, 0x22),
                                 gpu_mmio=types.SimpleNamespace(residx=0, off=0xbb0000))

class FakeDaemon(d.Daemon):
    def _hold(self):   # the real daemon sleeps until it is killed; a fake GPU never needs that, so fail loudly instead
        print("fake_nv_daemon: HOLD requested (the fake GPU did not confirm its unload); exiting 3", file=sys.stderr, flush=True)
        os._exit(3)
    def cmd_boot(self, req):
        self.dev = FakeDev()
        self.vram_next = 0
        self.send_json({"ok": True, "arch": "sm_89"})
    def cmd_compile_all(self, req):
        ptx = open(req["ptx_path"], "rb").read()
        cubin = tgpaths.cubin_path(ptx, "sm_89")
        if cubin.exists():  # the real daemon's compile_all: the ptxas cubin, and the kernel names found in it
            self.elf_bytes = cubin.read_bytes()
            names = list(nch.extract_all_metadata(self.elf_bytes, is_blackwell=False)[1].keys())
        else:  # no cubin: kernel names only (the C++ runtime mode then fails to load programs)
            print(f"fake_nv_daemon: no cached cubin {cubin}; run run_goldens.sh first", file=sys.stderr)
            names = re.findall(r"\.entry\s+(\w+)", ptx.decode())
        self.kernel_names = set(names)
        self.send_json({"ok": True, "kernels": sorted(names)})
    def _get_program(self, name, n_int_args):
        if name not in self.kernel_names: raise RuntimeError(f"kernel {name!r} not in PTX")
        return FakePrg()
    def cmd_handoff(self, req):
        dev = handoff_dev()
        cb0 = 88 * 4 + 512
        programs = req.get("programs", True)
        progs = [types.SimpleNamespace(name=n, qmd=d.ops_nv.QMD(dev), lcmem_usage=0, max_threads=1024, fill_launch_dims=True,
                                       constbufs={0: (0, cb0)}, kernargs_alloc_size=round_up(cb0, 256) + (8 << 8), cbuf_0=[0] * 88,
                                       _dims_idx=(0, 3)) for n in sorted(self.kernel_names)] if programs else []
        fds = []
        for name, va, size in BUFS + (("vram", VRAM_VA, VRAM_SIZE),):
            fd = os.open(f"{MEM}/{name}.bin", os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
            os.ftruncate(fd, size)
            if name == "signal":   # the C++ timeline, which the daemon reads through the handoff buffer (plan step P3)
                self._signal_mm = mmap.mmap(fd, size)
                view = MMIOInterface(ctypes.addressof(ctypes.c_char.from_buffer(self._signal_mm)), size)
                self._handoff_bufs = {"signal": d.HCQBuffer(va, size, view=view)}
            if name != "vram": fds.append(fd)
            else: os.close(fd)
        json.dump({"va": VRAM_VA}, open(f"{MEM}/vram.json", "w"))
        info, blob = d.build_handoff(dev, progs, {n: types.SimpleNamespace(va_addr=va, size=s) for n, va, s in BUFS})
        if not programs:
            info.update(compute_class=dev.iface.compute_class, sass_version=0x89, shared_mem_window=0x729400000000,
                        local_mem_window=0x729300000000, num_gpcs=3, num_tpc_per_gpc=4, num_sm_per_tpc=2, max_warps_per_sm=48,
                        pool_va=VRAM_VA, pool_size=VRAM_SIZE, elf_size=len(self.elf_bytes))
        json.dump(info, open(f"{MEM}/handoff.json", "w"))
        info.update(ok=True, blob_size=len(blob), nfds=len(fds))
        self.send_json(info)
        self.sock.sendall(blob)
        if not programs: self.sock.sendall(self.elf_bytes)
        socket.send_fds(self.sock, [b"F"], fds)
        self.handed_off, self.runtime = True, not programs
    def cmd_alloc(self, req):
        if getattr(self, "runtime", False): raise RuntimeError("alloc after a C++ runtime handoff: the plugin allocates from the pool")
        if self.tgpu_fd is None: return super().cmd_alloc(req)
        va = VRAM_VA + self.vram_next
        self.vram_next += round_up(req["size"], 0x1000)
        if self.vram_next > VRAM_SIZE: raise RuntimeError("fake VRAM exhausted")
        self.send_json({"ok": True, "addr": va})

FakeDaemon(socket.socket(fileno=int(sys.argv[1])), int(sys.argv[2]) if len(sys.argv) > 2 else None).run()
