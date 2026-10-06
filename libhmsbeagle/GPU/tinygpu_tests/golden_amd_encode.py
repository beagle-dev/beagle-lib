"""Golden test for TinyGPUAMDDispatch.h (TODO.md plan step A1b). hcq1's own AMDComputeQueue, with amd_hcq_patch's
exec, and HCQProgram.fill_kernargs's CLikeArgsState encode random launch batches the way amd_dispatch_daemon.py's
launch_batch chains them (wait and memory_barrier, then execs, then signal and submit), then submit them into a small
ring through AMDQueueDesc.signal_doorbell. golden_amd_encode.cpp encodes the same batches with the C++ encoder alone.
Every queue dword, every kernargs byte (random-filled first, so untouched bytes compare too), the ring, put_value and
the order and values of the wptr, HDP flush and doorbell writes must be identical, across ring and kernargs wraps.
First, TinyGPUAMDTables.h must regenerate byte for byte (plan step A1a)."""
import os, sys, ctypes, random, subprocess, types, importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tgpaths
tgpaths.setup()
from tinygrad.runtime import ops_amd
from tinygrad.runtime.support.hcq import HCQBuffer, HCQProgram, MMIOInterface, CLikeArgsState
from tinygrad.runtime.support.memory import BumpAllocator
from tinygrad.runtime.support.amd import AMDIP, import_soc
from tinygrad.dtype import dtypes
import amd_hcq_patch
amd_hcq_patch.apply()

HERE, WORK, GPU = tgpaths.HERE, tgpaths.WORK, tgpaths.GPU_DIR
tgpaths.build_cpp(HERE / "golden_amd_encode.cpp", WORK / "golden_amd_encode")

# ---- A1a: the tables regenerate byte for byte
gen = subprocess.run([sys.executable, str(GPU / "make_tinygpu_amd_tables.py")], capture_output=True, text=True, check=True,
                     env={**os.environ, "TINYGRAD_PATH": tgpaths.TINYGRAD_PATH}).stdout
tables_ok = gen == (GPU / "TinyGPUAMDTables.h").read_text()
print(f"TinyGPUAMDTables.h regenerates from the pin: {'IDENTICAL' if tables_ok else 'DIFFERS (rerun make_tinygpu_amd_tables.py)'}")

SIG_VA, KARGS_VA = 0x7f_1000_0040, 0x7f_2000_0000

class Recorder:
    """An MMIOInterface stand-in for AMDQueueDesc's write_ptr and doorbell: records (what, value) in order."""
    def __init__(self, events, what): self.events, self.what = events, what
    def __setitem__(self, i, v): self.events.append((self.what, v))

def make_dev(rng, ring_dwords, put_value, events):
    off = importlib.import_module("tinygrad.runtime.autogen.am.navi_offsets")
    dev = types.SimpleNamespace(target=(11, 0, 0), xccs=1, sqtt_enabled=False, is_am=lambda: True, is_usb=lambda: False)
    dev.soc, dev.pm4 = import_soc(dev.target), importlib.import_module("tinygrad.runtime.autogen.am.pm4_nv")
    dev.gc = AMDIP("gc", (11, 0, 0), bases={i: tuple(getattr(off, f"GC_BASE__INST{i}_SEG{s}", 0) for s in range(6)) for i in range(6)})
    dev.nbio = AMDIP("nbio", (4, 3, 0), bases={i: tuple(getattr(off, f"NBIO_BASE__INST{i}_SEG{s}", 0) for s in range(9)) for i in range(6)})
    dev.scratch = types.SimpleNamespace(va_addr=0x7f_4000_0000 + (rng.getrandbits(20) << 8), size=rng.choice([1 << 20, 85 << 20]))
    dev.tmpring_size = rng.getrandbits(27)
    dev.ring_mem = (ctypes.c_uint32 * ring_dwords)(*[rng.getrandbits(32) for _ in range(ring_dwords)])
    dev.compute_queue = ops_amd.AMDQueueDesc(ring=MMIOInterface(ctypes.addressof(dev.ring_mem), ring_dwords * 4, fmt="I"), read_ptr=None,
                                             write_ptr=Recorder(events, "wptr"), doorbell=Recorder(events, "doorbell"), put_value=put_value)
    dev.iface = types.SimpleNamespace(dev_impl=types.SimpleNamespace(gmc=types.SimpleNamespace(flush_hdp=lambda: events.append(("hdp", 0)))))
    return dev

def make_prog(dev, rng, name, nptr, nint, hidden):
    """A kernel as BeagleAMDProgram sees it: a fixed signature (nptr pointers, then nint uint32), and the compiler's kernarg
    size: the explicit arguments (rounded to 8 or not), or a hidden block of 256 bytes at round_up(explicit, 8)."""
    explicit = 8 * nptr + 4 * nint
    p = types.SimpleNamespace(name=name, dev=dev, nptr=nptr, prog_addr=rng.getrandbits(40) << 8, rsrc1=rng.getrandbits(32),
                              rsrc2=rng.getrandbits(32), rsrc3=rng.getrandbits(32), wave32=rng.random() < 0.8, enable_dispatch_ptr=0,
                              enable_private_segment_sgpr=0, enable_queue_ptr=False, enable_dispatch_id=False, args_state_t=CLikeArgsState,
                              signature=tuple((None, i, dtypes.uint32, ()) for i in range(nint)))
    p.kernargs_segment_size = ((explicit + 7) // 8 * 8 + 256) if hidden else rng.choice([explicit, (explicit + 7) // 8 * 8])
    p.kernargs_alloc_size = p.kernargs_segment_size
    return p

def run(seed, ring_dwords, put_value, kargs_size, nbatches):
    rng = random.Random(seed)
    events = []
    dev = make_dev(rng, ring_dwords, put_value, events)
    ring_before = bytes(dev.ring_mem)
    kmem = (ctypes.c_uint8 * kargs_size)(*[rng.getrandbits(8) for _ in range(kargs_size)])
    kargs_before = bytes(kmem)
    dev.kernargs_buf = HCQBuffer(KARGS_VA, kargs_size, view=MMIOInterface(ctypes.addressof(kmem), kargs_size))
    dev.kernargs_offset_allocator = BumpAllocator(kargs_size, wrap=True)
    sig = types.SimpleNamespace(value_addr=SIG_VA, owner=None, is_timeline=True)
    progs = {f"k{i}": make_prog(dev, rng, f"k{i}", nptr, nint, hidden)
             for i, (nptr, nint, hidden) in enumerate([(3, 0, False), (5, 2, True), (10, 6, True), (4, 3, False), (7, 1, True), (3, 1, False)])}
    lines = [f"DEV {dev.scratch.va_addr} {dev.scratch.size} {dev.tmpring_size}", f"SIG {SIG_VA}", f"KARGS {KARGS_VA} {kargs_size}",
             f"RING {ring_dwords} {put_value}"]
    lines += [f"K {p.name} {p.prog_addr} {p.rsrc1} {p.rsrc2} {p.rsrc3} {p.kernargs_segment_size} {p.kernargs_alloc_size} {int(p.wave32)}"
              for p in progs.values()]
    timeline, queues = rng.getrandbits(20) + 2, []
    for _ in range(nbatches):
        q = ops_amd.AMDComputeQueue(dev).wait(sig, timeline - 1).memory_barrier()   # the daemon's chained launch_batch
        lines.append(f"Q {timeline - 1}")
        for _ in range(rng.randint(1, 5)):
            p = progs[rng.choice(list(progs))]
            nptr, nint = p.nptr, len(p.signature)
            ptrs = [rng.getrandbits(40) for _ in range(nptr)]
            ints = [rng.getrandbits(32) for _ in range(nint)]
            grid, block = [rng.randint(1, 4096), rng.randint(1, 64), rng.randint(1, 8)], [rng.choice([1, 16, 64, 256]), rng.randint(1, 4), 1]
            args = HCQProgram.fill_kernargs(p, tuple(HCQBuffer(a, 0) for a in ptrs), tuple(ints))
            q.exec(p, args, tuple(grid), tuple(block))
            lines.append(f"L {p.name} {' '.join(map(str, grid))} {' '.join(map(str, block))} {nptr} {' '.join(map(str, ptrs))} {nint} "
                         f"{' '.join(map(str, ints))}")
        q.signal(sig, timeline)
        lines.append(f"S {timeline}")
        timeline += 1
        queues.append(list(q._q))
        q.submit(dev)
    (WORK / "golden_amd_batch.txt").write_text("\n".join(lines) + "\n")
    (WORK / "golden_amd_kargs_in.bin").write_bytes(kargs_before)
    (WORK / "golden_amd_ring_in.bin").write_bytes(ring_before)
    subprocess.run([str(WORK / "golden_amd_encode"), str(WORK)], check=True)
    got_q = [[int(x) for x in l.split()] for l in (WORK / "golden_amd_out_queues.txt").read_text().splitlines()]
    got_ev = [tuple(l.split()[:1]) + (int(l.split()[1]),) for l in (WORK / "golden_amd_out_events.txt").read_text().splitlines()]
    got_kargs, got_ring = (WORK / "golden_amd_out_kargs.bin").read_bytes(), (WORK / "golden_amd_out_ring.bin").read_bytes()
    got_put = int((WORK / "golden_amd_out_put.txt").read_text())
    checks = {"queue dwords": got_q == queues, "kernargs bytes": got_kargs == bytes(kmem), "ring": got_ring == bytes(dev.ring_mem),
              "put_value": got_put == dev.compute_queue.put_value, "wptr/HDP/doorbell": got_ev == events}
    for what, ok in checks.items():
        if ok: continue
        if what == "queue dwords":
            for i, (a, b) in enumerate(zip(queues, got_q)):
                if a != b: print(f"  batch {i}: ref {[hex(x) for x in a]}\n           c++ {[hex(x) for x in b]}"); break
        if what == "kernargs bytes": print(f"  kernargs differ at {[i for i in range(kargs_size) if got_kargs[i] != kmem[i]][:16]}")
        if what == "wptr/HDP/doorbell": print(f"  events ref {events[:9]}\n         c++ {got_ev[:9]}")
    ok = all(checks.values())
    wraps = (put_value + sum(map(len, queues))) // ring_dwords
    print(f"seed {seed}: {nbatches} batches, {sum(map(len, queues))} dwords into a {ring_dwords}-dword ring from {put_value} ({wraps} wrap(s)), "
          f"{kargs_size} B kernargs: {'IDENTICAL' if ok else 'MISMATCH ' + str([w for w, v in checks.items() if not v])}")
    return ok

results = [tables_ok, run(1, 1024, 900, 1 << 16, 4), run(2, 512, 0, 1500, 12), run(3, 4096, 4000, 4096, 30), run(4, 1 << 16, 12345, 1 << 20, 3)]
print("A1a tables and A1b PM4 encoder vs hcq1:", "all identical" if all(results) else "MISMATCH")
sys.exit(0 if all(results) else 1)
