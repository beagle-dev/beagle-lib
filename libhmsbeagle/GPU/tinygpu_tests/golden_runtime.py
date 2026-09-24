"""Golden test for the C++ runtime's device-side pieces (TinyGPUHybridNVProgram.h, TinyGPUHybridNVDispatch.h), each
against the tinygrad (hcq1) code it ports:
  1. the boot-only handoff: build_handoff with no programs, plus cmd_handoff's runtime keys, parses in C++;
  2. nvd_check_tables accepts tinygrad's QMD layout for Ada and Blackwell, and rejects a perturbed one;
  3. NVDevice._ensure_has_local_memory (on a fake device, real method) vs nvd_local_mem_size plus
     nvd_push_wait/nvd_push_setup_local_mem/nvd_push_signal: same local-memory size and same pushbuffer words;
  4. PCIIfaceBase.alloc + MemoryManager.alloc_vaddr (real methods, recording fakes) vs nvd_pool_alloc: same
     rounded sizes and address alignment."""
import os, sys, json, subprocess, types
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tgpaths
tgpaths.setup()
import nv_dispatch_daemon as d
from tinygrad.helpers import round_up
from tinygrad.runtime.support.system import PCIIfaceBase
from tinygrad.runtime.support.memory import MemoryManager
ops_nv = d.ops_nv
nv_gpu = ops_nv.nv_gpu

HERE, WORK = str(tgpaths.HERE), str(tgpaths.WORK)
SIG_VA, POOL_VA, POOL_SIZE, LM_VA = 0x10_3000_4000, 0x40_0000_0000, 16 << 30, 0x40_1234_0000
tgpaths.build_cpp(f"{HERE}/golden_runtime.cpp", f"{WORK}/golden_runtime")

def make_dev(compute_class):
    def fifo(off, token): return types.SimpleNamespace(ring=types.SimpleNamespace(residx=1, off=off), entries_count=0x10000, token=token,
                                                       gpput=types.SimpleNamespace(residx=1, off=off + 0x8008c), put_value=3)
    return types.SimpleNamespace(iface=types.SimpleNamespace(compute_class=compute_class), compute_gpfifo=fifo(0x1000000, 0x10001),
                                 dma_gpfifo=fifo(0x1100000, 0x10002), gpu_mmio=types.SimpleNamespace(residx=0, off=0xbb0000))

BUFS = {k: types.SimpleNamespace(va_addr=va, size=sz) for k, va, sz in
        (("cmdq", 0x10_1000_0000, 2 << 20), ("kargs", 0x10_2000_0000, 16 << 20), ("staging", 0x10_4000_0000, 16 << 20),
         ("signal", SIG_VA, 0x1000))}

def local_mem_reference(topo, lcmems, timeline_value):
    """tinygrad's _ensure_has_local_memory, once per program as NVProgram.__init__ calls it, on a fake device."""
    sig = types.SimpleNamespace(value_addr=SIG_VA)
    dev = types.SimpleNamespace(slm_per_thread=0, shader_local_mem=None, timeline_signal=sig, timeline_value=timeline_value, **topo)
    sizes, words = [], []
    def realloc(old, size): sizes.append(size); return types.SimpleNamespace(va_addr=LM_VA), True
    def next_timeline(): dev.timeline_value += 1; return dev.timeline_value - 1
    dev._realloc, dev.next_timeline = realloc, next_timeline
    orig_submit = ops_nv.NVComputeQueue.submit
    ops_nv.NVComputeQueue.submit = lambda q, _dev: words.append(list(q._q)) or q
    try:
        for r in lcmems: ops_nv.NVDevice._ensure_has_local_memory(dev, r)
    finally: ops_nv.NVComputeQueue.submit = orig_submit
    return dev.slm_per_thread, sizes[-1], words[-1], words

def pool_reference(sizes):
    """tinygrad's VRAM allocation of each size: what PCIIfaceBase.alloc hands valloc, and the alignment
    MemoryManager.alloc_vaddr asks the VA allocator for."""
    out = []
    for size in sizes:
        rec = {}
        mm = types.SimpleNamespace()
        def valloc(sz, uncached=False, contiguous=False, zero=False):
            rec["size"] = sz
            va_alloc = types.SimpleNamespace(alloc=lambda s, a: rec.update(va_size=s, align=a) or 0)
            MemoryManager.alloc_vaddr.__func__(types.SimpleNamespace(va_allocator=va_alloc), round_up(sz, 0x1000), 0x1000)  # as valloc does
            return types.SimpleNamespace(va_addr=0, paddrs=[(0, sz)])
        mm.valloc = valloc
        iface = types.SimpleNamespace(is_bar_small=lambda: False, dev_impl=types.SimpleNamespace(mm=mm), dev=None, vram_bar=1)
        PCIIfaceBase.alloc(iface, size)
        assert rec["va_size"] == round_up(rec["size"], 0x1000)
        out.append((rec["va_size"], rec["align"]))
    return out

TOPOS = [dict(num_gpcs=3, num_tpc_per_gpc=4, num_sm_per_tpc=2, max_warps_per_sm=48),   # an RTX 4060-like part
         dict(num_gpcs=5, num_tpc_per_gpc=6, num_sm_per_tpc=2, max_warps_per_sm=48),
         dict(num_gpcs=7, num_tpc_per_gpc=9, num_sm_per_tpc=2, max_warps_per_sm=64)]
LCMEMS = [[0x240], [0x260, 0x240, 0x400, 0x3e0], [0x241], [0x240 + 576, 0x240]]
POOL_SIZES = [1, 100, 0x1000, 0x1001, 0x3000, 1 << 20, (1 << 20) + 1, (8 << 20) - 1, 8 << 20, (8 << 20) + 1, 100 << 20, 5,
              (3 << 30) + 12345, 0x2345]

ok_all = True
for cc, tag in ((nv_gpu.ADA_COMPUTE_A, "Ada"), (nv_gpu.BLACKWELL_COMPUTE_B, "Blackwell")):
    info, blob = d.build_handoff(make_dev(cc), [], BUFS)
    assert blob == b"" and info["nkernels"] == 0
    cases = []
    for ti, topo in enumerate(TOPOS):
        for li, lc in enumerate(LCMEMS):
            tv = 41 + ti * 10 + li
            slm, size, last_words, all_words = local_mem_reference(topo, lc, tv)
            cases.append(dict(topo=topo, lcmems=lc, timeline=tv, slm=slm, size=size, words=last_words, submits=len(all_words)))
    ref_pool = pool_reference(POOL_SIZES)

    rt = dict(compute_class=cc, sass_version=0x89, shared_mem_window=0x729400000000, local_mem_window=0x729300000000,
              pool_va=POOL_VA, pool_size=POOL_SIZE, elf_size=0, **TOPOS[0])
    json.dump(dict(info, **rt), open(f"{WORK}/golden_rt_handoff.json", "w"))
    with open(f"{WORK}/golden_rt_cases.txt", "w") as f:
        for c in cases:
            t = c["topo"]
            # tinygrad grows local memory program by program; its last setup submit carries the final size, one
            # timeline value per earlier submit later than C++'s single submit
            f.write(f"LM {t['num_gpcs']} {t['num_tpc_per_gpc']} {t['num_sm_per_tpc']} {t['max_warps_per_sm']} "
                    f"{c['timeline'] + c['submits'] - 1} "
                    f"{len(c['lcmems'])} {' '.join(map(str, c['lcmems']))}\n")
        f.write(f"POOL {len(POOL_SIZES)} {' '.join(map(str, POOL_SIZES))}\n")
    r = subprocess.run([f"{WORK}/golden_runtime", WORK, str(SIG_VA), str(LM_VA)], capture_output=True, text=True)
    if r.returncode != 0: print(r.stdout, r.stderr); ok_all = False; continue
    lines = r.stdout.split("\n")
    assert lines[0] == "TABLES OK" and lines[1] == "TABLES PERTURBED REJECTED", lines[:2]
    ok, i = True, 2
    for c in cases:
        slm, size, words = lines[i].split(" ", 2); i += 1
        words = [int(x) for x in words.split()]
        if (int(slm), int(size), words) != (c["slm"], c["size"], c["words"]):
            ok = False
            print(f"  {tag} local memory case {c['topo']} {c['lcmems']}: ref slm {c['slm']:#x} size {c['size']:#x} words "
                  f"{[hex(w) for w in c['words']]}\n  c++ slm {int(slm):#x} size {int(size):#x} words {[hex(w) for w in words]}")
    prev_end = POOL_VA
    for (size, (ref_size, ref_align)) in zip(POOL_SIZES, ref_pool):
        va, pos = (int(x) for x in lines[i].split()); i += 1
        got_size = POOL_VA + pos - va
        if va % ref_align or got_size != ref_size or va < prev_end:
            ok = False
            print(f"  {tag} pool alloc {size:#x}: ref size {ref_size:#x} align {ref_align:#x}; c++ va {va:#x} size {got_size:#x}")
        prev_end = va + got_size
    lm_submits = sum(c["submits"] for c in cases)
    print(f"{tag} (QMD v{info['qmd_ver']}): boot-only handoff parsed, tables checked; {len(cases)} local-memory cases "
          f"({lm_submits} tinygrad setup submits); {len(POOL_SIZES)} pool allocations: {'IDENTICAL' if ok else 'MISMATCH'}")
    ok_all &= ok
sys.exit(0 if ok_all else 1)
