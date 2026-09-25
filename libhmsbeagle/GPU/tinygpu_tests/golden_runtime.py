"""Golden test for the C++ runtime's device-side pieces (TinyGPUHybridNVProgram.h, TinyGPUHybridNVDispatch.h), each
against the tinygrad (hcq1) code it ports:
  1. the boot-only handoff: build_handoff with no programs, plus cmd_handoff's runtime keys, parses in C++;
  2. nvd_check_tables accepts tinygrad's QMD layout for Ada and Blackwell, and rejects a perturbed one;
  3. NVDevice._ensure_has_local_memory (on a fake device, real method) vs nvd_local_mem_size plus
     nvd_push_wait/nvd_push_setup_local_mem/nvd_push_signal: same local-memory size and same pushbuffer words;
  4. PCIIfaceBase.alloc + MemoryManager.alloc_vaddr (real methods, recording fakes) vs nvd_pool_alloc: same
     rounded sizes and address alignment;
  5. (plan step P5) a second instance's programs after the first's on one device: _ensure_has_local_memory per
     program vs nvdLoadPrograms' rule (a setup only when the second needs more): same slm_per_thread, and the same
     last setup submit per instance, or none."""
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

def local_mem_two_reference(topo, lc_a, lc_b, timeline_value):
    """TODO.md plan step P5: a second instance's programs (lc_b) loaded after the first's (lc_a) on one device, tinygrad's
    _ensure_has_local_memory once per program. For each instance: the device's slm_per_thread, the local-memory size and
    words of its last setup submit (None if it submitted none), and its number of submits."""
    sig = types.SimpleNamespace(value_addr=SIG_VA)
    dev = types.SimpleNamespace(slm_per_thread=0, shader_local_mem=None, timeline_signal=sig, timeline_value=timeline_value, **topo)
    sizes, words = [], []
    def realloc(old, size): sizes.append(size); return types.SimpleNamespace(va_addr=LM_VA), True
    def next_timeline(): dev.timeline_value += 1; return dev.timeline_value - 1
    dev._realloc, dev.next_timeline = realloc, next_timeline
    orig_submit = ops_nv.NVComputeQueue.submit
    ops_nv.NVComputeQueue.submit = lambda q, _dev: words.append(list(q._q)) or q
    out = []
    try:
        for lcs in (lc_a, lc_b):
            n0 = len(words)
            for r in lcs: ops_nv.NVDevice._ensure_has_local_memory(dev, r)
            out.append((dev.slm_per_thread, sizes[-1] if len(words) > n0 else None, words[-1] if len(words) > n0 else None,
                        len(words) - n0))
    finally: ops_nv.NVComputeQueue.submit = orig_submit
    return out

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
# plan step P5: (first instance's programs, second instance's): the second needs no more, needs more after rounding to
# 32, or grows program by program
LCMEMS_TWO = [([0x240], [0x240]), ([0x260, 0x240], [0x250]), ([0x240], [0x241]), ([0x240], [0x400]), ([0x400], [0x240, 0x800, 0x600])]
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
    cases_two = []
    for ti, topo in enumerate(TOPOS):
        for li, (lc_a, lc_b) in enumerate(LCMEMS_TWO):
            tv = 71 + ti * 10 + li
            cases_two.append(dict(topo=topo, lc_a=lc_a, lc_b=lc_b, timeline=tv, ref=local_mem_two_reference(topo, lc_a, lc_b, tv)))
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
        for c in cases_two:
            # each instance's setup carries tinygrad's timeline value for its last submit
            t, (a, b) = c["topo"], c["ref"]
            f.write(f"LM2 {t['num_gpcs']} {t['num_tpc_per_gpc']} {t['num_sm_per_tpc']} {t['max_warps_per_sm']} "
                    f"{c['timeline'] + a[3] - 1} {len(c['lc_a'])} {' '.join(map(str, c['lc_a']))} "
                    f"{c['timeline'] + a[3] + max(b[3], 1) - 1} {len(c['lc_b'])} {' '.join(map(str, c['lc_b']))}\n")
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
    for c in cases_two:
        for which, (slm, size, words, _) in zip("AB", c["ref"]):
            got = lines[i].split(" ", 2); i += 1
            got_words = [int(x) for x in got[2].split()] if got[1] != "NONE" else None
            if (int(got[0]), None if got[1] == "NONE" else int(got[1]), got_words) != (slm, size, words):
                ok = False
                print(f"  {tag} two-instance case {c['topo']} {c['lc_a']} then {c['lc_b']}, instance {which}: ref slm {slm:#x} "
                      f"size {size} words {words}\n  c++ {' '.join(got[:2])} words {got_words}")
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
          f"({lm_submits} tinygrad setup submits), {len(cases_two)} two-instance cases; {len(POOL_SIZES)} pool allocations: "
          f"{'IDENTICAL' if ok else 'MISMATCH'}")
    ok_all &= ok
sys.exit(0 if ok_all else 1)
