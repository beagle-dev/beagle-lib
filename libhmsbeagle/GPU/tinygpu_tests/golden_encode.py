"""Golden test for TinyGPUHybridNVDispatch.h: hcq1's own NVComputeQueue/NVCopyQueue encode a batch into real
memory; golden_encode.cpp encodes the same batch from nv_dispatch_daemon.build_handoff's output alone; both
results must be byte-identical. Templates are random bytes so every neighbouring bit is live."""
import os, sys, json, struct, ctypes, random, subprocess, types
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tgpaths
tgpaths.setup()
import nv_dispatch_daemon as d
ops_nv = d.ops_nv
from tinygrad.runtime.support.hcq import HCQBuffer, MMIOInterface
from tinygrad.helpers import round_up
from tinygrad.dtype import dtypes

HERE, WORK = str(tgpaths.HERE), str(tgpaths.WORK)
tgpaths.build_cpp(f"{HERE}/golden_encode.cpp", f"{WORK}/golden_encode")
KARGS_VA, KARGS_SIZE, SIG_VA, V = 0x10_2000_0000, 1 << 20, 0x10_3000_4000, 77

def make_dev(compute_class):
    dev = types.SimpleNamespace(pma_enabled=False, slm_per_thread=0x800, iface=types.SimpleNamespace(compute_class=compute_class))
    def fifo(off, token): return types.SimpleNamespace(ring=types.SimpleNamespace(residx=1, off=off), entries_count=0x10000, token=token,
                                                       gpput=types.SimpleNamespace(residx=1, off=off + 0x8008c), put_value=123)
    dev.compute_gpfifo, dev.dma_gpfifo = fifo(0x1000000, 0x10001), fifo(0x1100000, 0x10002)
    dev.gpu_mmio = types.SimpleNamespace(residx=0, off=0xbb0000)
    return dev

def make_prog(dev, name, rng, blackwell, fill):
    qmd = ops_nv.QMD(dev)
    qmd.mv[:] = bytes(rng.getrandbits(8) for _ in range(len(qmd.mv)))
    qmd.write(release0_enable=0)
    nwords = 224 if blackwell else 88
    cb0 = nwords * 4 + rng.choice([16, 64, 200])
    p = types.SimpleNamespace(name=name, qmd=qmd, dev=dev, lcmem_usage=0x400, max_threads=1024, fill_launch_dims=fill,
                              constbufs={0: (0, cb0)}, kernargs_alloc_size=round_up(cb0, 256) + (8 << 8),
                              cbuf_0=[rng.getrandbits(32) for _ in range(nwords)], _dims_idx=(216, 220) if blackwell else (0, 3))
    return p

def run(compute_class, tag, fill=True):
    rng = random.Random(1234)
    dev = make_dev(compute_class)
    blackwell = compute_class >= ops_nv.nv_gpu.BLACKWELL_COMPUTE_A
    progs = [make_prog(dev, n, rng, blackwell, fill) for n in ("kernelA", "kernelB")]
    bufs = {k: types.SimpleNamespace(va_addr=va, size=sz) for k, va, sz in
            (("cmdq", 0x10_1000_0000, 2 << 20), ("kargs", KARGS_VA, KARGS_SIZE), ("staging", 0x10_4000_0000, 16 << 20), ("signal", SIG_VA, 0x4000))}
    info, blob = d.build_handoff(dev, progs, bufs)

    # hcq1 reference: the daemon's chained launch_batch path, into real memory
    mem = (ctypes.c_uint8 * KARGS_SIZE)()
    kargs = HCQBuffer(KARGS_VA, KARGS_SIZE, view=MMIOInterface(ctypes.addressof(mem), KARGS_SIZE))
    sig = types.SimpleNamespace(value_addr=SIG_VA)
    launches = [(progs[0], (4, 3, 2), (16, 16, 1), [0x10_5000_0000, 0x10_5000_1000], [7]),
                (progs[1], (64, 1, 1), (128, 1, 1), [0x10_5000_2000], [1, 2, 3]),
                (progs[0], (1, 1, 1), (32, 4, 2), [0x10_5000_3000, 0x10_5000_4000, 0x10_5000_5000], [])]
    q = ops_nv.NVComputeQueue()
    q.wait(sig, V - 1).memory_barrier()
    pos, lines = 0, [f"SIG {SIG_VA} {V}", f"KARGS {KARGS_VA} {KARGS_SIZE}"]
    for p, grid, block, ptrs, ints in launches:
        p.signature = tuple((None, i, dtypes.uint32, ()) for i in range(len(ints)))
        d.BeagleNVProgram.set_launch_dims(p, grid, block)  # the daemon's own fill (zeros when it's off)
        off = round_up(pos, 256); pos = off + p.kernargs_alloc_size
        args = ops_nv.NVArgsState(kargs.offset(off, p.kernargs_alloc_size), p, tuple(HCQBuffer(x, 0) for x in ptrs), vals=tuple(ints))
        q.exec(p, args, grid, block)
        lines.append(f"L {p.name} {' '.join(map(str, grid))} {' '.join(map(str, block))} {len(ptrs)} {' '.join(map(str, ptrs))} "
                     f"{len(ints)} {' '.join(map(str, ints))}")
    q.signal(sig, V)
    ref_compute, ref_kargs = list(q._q), bytes(mem)

    copies = [(0x10_5000_0000, 0x10_4000_0000, 4096, V, V + 1), (0x10_4000_0100, 0x10_6000_0000, (1 << 31) + 4096, V + 1, V + 2)]
    ref_copy = []
    for dst, src, n, vw, vs in copies:
        cq = ops_nv.NVCopyQueue()
        cq.wait(sig, vw).copy(HCQBuffer(dst, n), HCQBuffer(src, n), n).signal(sig, vs)
        ref_copy += list(cq._q)
        lines.append(f"C {dst} {src} {n} {vw} {vs}")

    with open(f"{WORK}/golden_handoff.json", "w") as f: json.dump(info, f)
    with open(f"{WORK}/golden_blob.bin", "wb") as f: f.write(blob)
    with open(f"{WORK}/golden_batch.txt", "w") as f: f.write("\n".join(lines) + "\n")
    subprocess.run([f"{WORK}/golden_encode", WORK], check=True)
    out_compute = [int(x) for x in open(f"{WORK}/golden_out_compute.txt").read().split()]
    out_copy = [int(x) for x in open(f"{WORK}/golden_out_copy.txt").read().split()]
    out_kargs = open(f"{WORK}/golden_out_kargs.bin", "rb").read()
    ok = out_compute == ref_compute and out_copy == ref_copy and out_kargs == ref_kargs
    if not ok:
        if out_compute != ref_compute: print(" compute pushbuffer differs:\n  ref", [hex(x) for x in ref_compute], "\n  c++", [hex(x) for x in out_compute])
        if out_copy != ref_copy: print(" copy pushbuffer differs:\n  ref", [hex(x) for x in ref_copy], "\n  c++", [hex(x) for x in out_copy])
        diff = [i for i in range(KARGS_SIZE) if out_kargs[i] != ref_kargs[i]]
        if diff: print(f" kargs differ at {len(diff)} bytes, first {diff[:16]}")
    print(f"{tag}: QMD v{info['qmd_ver']}, {len(ref_compute)} compute words, {len(ref_copy)} copy words, "
          f"{pos} kargs bytes, launch dims {'filled' if fill else 'off'}: {'IDENTICAL' if ok else 'MISMATCH'}")
    return ok

nv_gpu = ops_nv.nv_gpu
results = [run(nv_gpu.ADA_COMPUTE_A, "Ada (sm_89)"), run(nv_gpu.BLACKWELL_COMPUTE_B, "Blackwell (sm_120)"),
           run(nv_gpu.ADA_COMPUTE_A, "Ada, fill off", fill=False)]
sys.exit(0 if all(results) else 1)
