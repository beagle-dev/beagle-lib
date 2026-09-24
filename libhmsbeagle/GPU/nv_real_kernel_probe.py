#!/usr/bin/env python3
"""
nv_real_kernel_probe.py -- compiles the real kernels4.cu (+kernelsAll.cu) with
nvcc -DCUDA -DFW_TINYGPU, exactly as make_tinygpu_kernels.sh does, and
dispatches kernelMatrixMulADB, alone or with the other four pipeline
kernels, through nv_dispatch_daemon.BeagleNVProgram on a tinygrad NVDevice
-- no C++ build, no daemon RPC. History: TODO.md Phases 68-140; root cause
of the NV bug (unpopulated cbuf0 launch dims) in STATUS.md §203.

Real inputs, verified against the actual source (not guessed):
  - A (dEvec), B (dIevc), D (dEigenValues): the *same* small buffers for
    every wMatrix -- one eigendecomposition, shared across all 16 blocks
    (kernelsAll.cu's own `a`/`b`/`d` offset arithmetic never includes
    wMatrix). JC69 constants, byte-for-byte from tinygpuhybridtest.cpp's
    `useDnaModel` branch (evec/ivec/eval).
  - distanceQueue[wMatrix] = edgeLengths[i] * categoryRates[j], wMatrix =
    i*kCategoryCount + j (i=edge 0..3, j=category 0..3) -- confirmed via
    BeagleGPUImpl.hpp's updateTransitionMatrices loop order (outer i,
    inner j). edgeLengths/categoryRates copied from tinygpuhybridtest.cpp.
  - listC[wMatrix] = wMatrix * stateCount^2 -- both Phase 65's proven
    closed form and BeagleGPUImpl.hpp's real formula
    (hPtrQueue[totalCount] = probabilityIndices[i]*kIndexOffsetMat +
    j*categoryOffset, with probabilityIndices=nodeIdx={0,1,2,3}
    sequential, kIndexOffsetMat==kMatrixSize==categoryOffset).
  - dMatrices: totalMatrix (16) real matrix slots + totalMatrix (16)
    ground-truth scratch slots, matching TINYGPU_DEBUG_DUMP_MATMUL_
    GROUND_TRUTH's own addressing exactly (dMatrices +
    totalMatrix*S^2 + blockIdx.x*S^2, S=stateCount=4) -- scratch region
    pre-seeded with the same -999.0 sentinel tinygpuhybridtest uses.
  - length=4, wB=4, totalMatrix=16 -- read directly off a real hardware
    launch log (`ints=[4,4,16]`), not guessed.
  - grid=(16,1,1) block=(16,16,1) -- the real launch shape.

Default macro: `TINYGPU_DEBUG_DUMP_MATMUL_GROUND_TRUTH` (a per-block dbg[]
scratch dump past the real matrices). A no-argument (solo) run reports
`RESULT: PASS -- no sentinels remain` when every block ran.

    python3 nv_real_kernel_probe.py [--dims-probe] [--batch] [--realloc] [--logl] [--logl-sweep [N]] [--sweep [N]] [--wide-grid [N]] [--chain-sweep N [--sync-each]] [--downstream-sweep [N]] [--maxrregcount N] [EXTRA_MACRO ...]

Examples:
    python3 nv_real_kernel_probe.py                                  # solo: kernelMatrixMulADB alone, dbg[] sentinel check (Phase 68)
    python3 nv_real_kernel_probe.py --batch                          # queued with the 4 other pipeline kernels, as cmd_launch_batch does (Phase 69)
    python3 nv_real_kernel_probe.py --realloc                        # solo, after BEAGLE's real ~15-buffer allocation set/order (Phase 70)
    python3 nv_real_kernel_probe.py --batch --realloc                # both (Phase 70)
    python3 nv_real_kernel_probe.py --logl                           # real 3-taxon logL vs the CPU reference (Phase 71)
    python3 nv_real_kernel_probe.py --logl-sweep 20                  # the 5-kernel logL chain 20x: PASS/FAIL/NaN rate (Phase 88)
    python3 nv_real_kernel_probe.py --sweep 20                       # 20 dispatches: per-wMatrix correctness vs the closed-form matrix, per-SM tables (Phases 71/90)
    python3 nv_real_kernel_probe.py --sweep --wide-grid 64           # --sweep with 64 blocks instead of 16 (default N=32; Phase 78)
    python3 nv_real_kernel_probe.py --sweep --maxrregcount 20        # --sweep with ptxas capped at 20 registers (Phase 85)
    python3 nv_real_kernel_probe.py --chain-sweep 2 --sync-each      # first N (1-5) pipeline stages 20x, drift check, sync after each stage (Phases 134/136)
    python3 nv_real_kernel_probe.py --downstream-sweep               # PPNS/IL/SS only, fed reference matrices; kernelMatrixMulADB never dispatched (Phase 99)
    python3 nv_real_kernel_probe.py --dims-probe                     # are the cbuf0 blockDim/gridDim words populated? fill OFF, then ON (Phase 140)
"""
import sys, os, pathlib, struct, subprocess, time, math
from collections import Counter, defaultdict

# Default: the tinygrad worktree pinned at a9830e2b4 -- tinygrad HEAD
# (after 2026-09-05) dropped the macOS TinyGPU transport and hcq1 (TODO.md
# Phase 140). TINYGRAD_PATH overrides.
_TINYGRAD_PATH = os.environ.get("TINYGRAD_PATH", str(pathlib.Path.home() / "Dropbox/Projects/tinygrad-hcq1"))
sys.path.insert(0, _TINYGRAD_PATH)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_KERNELS_DIR = _REPO_ROOT / "libhmsbeagle" / "GPU" / "kernels"

GRID = (16, 1, 1)   # totalMatrix -- the real kernelMatrixMulADB launch shape for this config
BLOCK = (16, 16, 1)
STATE_COUNT = 4
S2 = STATE_COUNT * STATE_COUNT   # 16 -- floats per matrix
TOTAL_MATRIX = 16                # 4 edges x 4 categories
SENTINEL = -999.0

# TODO.md Phase 97: set to the live NVDevice as soon as main() boots one,
# so the top-level exception handler can attempt a richer fault-diagnostic
# read (tinygrad's own on_device_hang()/NV83DE MMU-fault-info RM control,
# ops_nv.py -- real fault address/type/access-type, when available) after
# a "Device fault detected" exception, instead of just letting whatever
# (possibly empty) message that exception carried be the only record.
_dev_for_diagnostics = None

# Default macro: TINYGPU_DEBUG_DUMP_MATMUL_GROUND_TRUTH (kernelsAll.cu) writes
# a per-block dbg[] slot (csub0, As/Bs row/col 0, Ds[0..3], %smid) past the
# real matrices -- read by --sweep's SMID tables, --logl's failure dump and
# the solo/--batch/--realloc sentinel check. Extra macros from argv are appended.
BASE_MACROS = ["TINYGPU_DEBUG_DUMP_MATMUL_GROUND_TRUTH"]

# ---- Real JC69 + 4-category discrete Gamma model, byte-for-byte from
# ---- tinygpuhybridtest.cpp's useDnaModel branch. ----
EVEC = [1.0,  2.0,  0.0,  0.5,
        1.0, -2.0,  0.5,  0.0,
        1.0,  2.0,  0.0, -0.5,
        1.0, -2.0, -0.5,  0.0]
IVEC = [0.25,  0.25,  0.25,  0.25,
        0.125,-0.125, 0.125,-0.125,
        0.0,   1.0,   0.0,  -1.0,
        1.0,   0.0,  -1.0,   0.0]
EVAL = [0.0, -4.0/3.0, -4.0/3.0, -4.0/3.0]
EDGE_LENS = [0.1, 0.1, 0.2, 0.1]
CATEGORY_RATES = [0.03338775, 0.25191592, 0.82026848, 2.89442785]
CATEGORY_WEIGHTS = [0.25, 0.25, 0.25, 0.25]
STATE_FREQS = [0.25, 0.25, 0.25, 0.25]
K_REF = -1498.89812   # tinygpuhybridtest.cpp's own CPU reference logL

# TODO.md Phase 90: every "success"/"reliability" measurement this
# investigation has made through Phase 89 only ever checked whether
# kernelMatrixMulADB's real C[] output was *nonzero* -- never whether it
# was *correct*. A block could write plausible-looking garbage and every
# --sweep/--wide-grid/--maxrregcount run so far would have counted it as
# a success. This closes that gap: the real transition matrix
# kernelMatrixMulADB computes for a given branch*rate distance is
# P[ty][tx] = sum_k EVEC[ty][k] * exp(EVAL[k]*distance) * IVEC[k][tx] --
# derived directly from kernelsAll.cu's real Csub accumulation
# (As[ty][k]*Ds[k]*Bs[k][tx] with a=b=d=0 for this BLOCKS==1, by=bx=0
# config, not guessed), and independently cross-checked against the
# well-known closed-form JC69 transition matrix formula (P_ii(t) =
# 1/4 + 3/4*exp(-4t/3), P_ij(t) = 1/4 - 1/4*exp(-4t/3)) -- exact match,
# row sums exactly 1.0, verified numerically before this was trusted.
CORRECTNESS_TOL = 1e-3   # float32 JC69 entries are O(0.01-0.9); real FMA rounding is orders of magnitude tighter than this


def reference_transition_matrix(distance):
    """The real 16-float transition matrix kernelMatrixMulADB should
    produce for this branch*rate distance, row-major (matches the real
    C[STATE_COUNT*ty+tx] layout exactly)."""
    ds = [math.exp(EVAL[k] * distance) for k in range(STATE_COUNT)]
    P = [0.0] * S2
    for ty in range(STATE_COUNT):
        for tx in range(STATE_COUNT):
            P[STATE_COUNT * ty + tx] = sum(EVEC[STATE_COUNT * ty + k] * ds[k] * IVEC[STATE_COUNT * k + tx]
                                            for k in range(STATE_COUNT))
    return P


# ---- --logl mode: real 3-taxon likelihood, byte-for-byte from
# ---- tinygpuhybridtest.cpp's kHuman/kChimp/kGorilla + makePartials, and
# ---- the real (Human,Chimp)node3,(Gorilla,node3)root topology ops[2].
# ---- Verified this is 768 characters (== N_PATTERNS) by direct count,
# ---- not assumed.
K_HUMAN = (
    "AGAAATATGTCTGATAAAAGAGTTACTTTGATAGAGTAAATAATAGGAGCTTAAACCCCCTTATTTCTACTA"
    "GGACTATGAGAATCGAACCCATCCCTGAGAATCCAAAATTCTCCGTGCCACCTATCACACCCCATCCTAAGT"
    "AAGGTCAGCTAAATAAGCTATCGGGCCCATACCCCGAAAATGTTGGTTATACCCTTCCCGTACTAAGAAATT"
    "TAGGTTAAATACAGACCAAGAGCCTTCAAAGCCCTCAGTAAGTTG-CAATACTTAATTTCTGTAAGGACTGC"
    "AAAACCCCACTCTGCATCAACTGAACGCAAATCAGCCACTTTAATTAAGCTAAGCCCTTCTAGACCAATGGG"
    "ACTTAAACCCACAAACACTTAGTTAACAGCTAAGCACCCTAATCAAC-TGGCTTCAATCTAAAGCCCCGGCA"
    "GG-TTTGAAGCTGCTTCTTCGAATTTGCAATTCAATATGAAAA-TCACCTCGGAGCTTGGTAAAAAGAGGC"
    "CTAACCCCTGTCTTTAGATTTACAGTCCAATGCTTCA-CTCAGCCATTTTACCACAAAAAAGGAAGGAATCG"
    "AACCCCCCAAAGCTGGTTTCAAGCCAACCCCATGGCCTCCATGACTTTTTCAAAAGGTATTAGAAAAACCAT"
    "TTCATAACTTTGTCAAAGTTAAATTATAGGCT-AAATCCTATATATCTTA-CACTGTAAAGCTAACTTAGCA"
    "TTAACCTTTTAAGTTAAAGATTAAGAGAACCAACACCTCTTTACAGTGA")
K_CHIMP = (
    "AGAAATATGTCTGATAAAAGAATTACTTTGATAGAGTAAATAATAGGAGTTCAAATCCCCTTATTTCTACTA"
    "GGACTATAAGAATCGAACTCATCCCTGAGAATCCAAAATTCTCCGTGCCACCTATCACACCCCATCCTAAGT"
    "AAGGTCAGCTAAATAAGCTATCGGGCCCATACCCCGAAAATGTTGGTTACACCCTTCCCGTACTAAGAAATT"
    "TAGGTTAAGCACAGACCAAGAGCCTTCAAAGCCCTCAGCAAGTTA-CAATACTTAATTTCTGTAAGGACTGC"
    "AAAACCCCACTCTGCATCAACTGAACGCAAATCAGCCACTTTAATTAAGCTAAGCCCTTCTAGATTAATGGG"
    "ACTTAAACCCACAAACATTTAGTTAACAGCTAAACACCCTAATCAAC-TGGCTTCAATCTAAAGCCCCGGCA"
    "GG-TTTGAAGCTGCTTCTTCGAATTTGCAATTCAATATGAAAA-TCACCTCAGAGCTTGGTAAAAAGAGGC"
    "TTAACCCCTGTCTTTAGATTTACAGTCCAATGCTTCA-CTCAGCCATTTTACCACAAAAAAGGAAGGAATCG"
    "AACCCCCTAAAGCTGGTTTCAAGCCAACCCCATGACCTCCATGACTTTTTCAAAAGATATTAGAAAAACTAT"
    "TTCATAACTTTGTCAAAGTTAAATTACAGGTT-AACCCCCGTATATCTTA-CACTGTAAAGCTAACCTAGCA"
    "TTAACCTTTTAAGTTAAAGATTAAGAGGACCGACACCTCTTTACAGTGA")
K_GORILLA = (
    "AGAAATATGTCTGATAAAAGAGTTACTTTGATAGAGTAAATAATAGAGGTTTAAACCCCCTTATTTCTACTA"
    "GGACTATGAGAATTGAACCCATCCCTGAGAATCCAAAATTCTCCGTGCCACCTGTCACACCCCATCCTAAGT"
    "AAGGTCAGCTAAATAAGCTATCGGGCCCATACCCCGAAAATGTTGGTCACATCCTTCCCGTACTAAGAAATT"
    "TAGGTTAAACATAGACCAAGAGCCTTCAAAGCCCTTAGTAAGTTA-CAACACTTAATTTCTGTAAGGACTGC"
    "AAAACCCTACTCTGCATCAACTGAACGCAAATCAGCCACTTTAATTAAGCTAAGCCCTTCTAGATCAATGGG"
    "ACTCAAACCCACAAACATTTAGTTAACAGCTAAACACCCTAGTCAAC-TGGCTTCAATCTAAAGCCCCGGCA"
    "GG-TTTGAAGCTGCTTCTTCGAATTTGCAATTCAATATGAAAT-TCACCTCGGAGCTTGGTAAAAAGAGGC"
    "CCAGCCTCTGTCTTTAGATTTACAGTCCAATGCCTTA-CTCAGCCATTTTACCACAAAAAAGGAAGGAATCG"
    "AACCCCCCAAAGCTGGTTTCAAGCCAACCCCATGACCTTCATGACTTTTTCAAAAGATATTAGAAAAACTAT"
    "TTCATAACTTTGTCAAGGTTAAATTACGGGTT-AAACCCCGTATATCTTA-CACTGTAAAGCTAACCTAGCG"
    "TTAACCTTTTAAGTTAAAGATTAAGAGTATCGGCACCTCTTTGCAGTGA")
assert len(K_HUMAN) == len(K_CHIMP) == len(K_GORILLA) == 768


def make_tip_partials(seq, category_count):
    """Byte-for-byte port of tinygpuhybridtest.cpp's makePartials() +
    BeagleGPUImpl.hpp's setTipPartials() category-replication: one-hot
    A/C/G/T per pattern (anything else, e.g. '-', is all-ones/ambiguous),
    then the identical block repeated once per category (setTipPartials
    literally memcpy's the same kPaddedStateCount*kPaddedPatternCount
    block category_count times -- tip data doesn't vary by category, but
    every partials buffer, tip or internal, is allocated/addressed at the
    same category-scaled kPartialsSize regardless)."""
    one = []
    for ch in seq:
        if ch == 'A':
            one += [1.0, 0.0, 0.0, 0.0]
        elif ch == 'C':
            one += [0.0, 1.0, 0.0, 0.0]
        elif ch == 'G':
            one += [0.0, 0.0, 1.0, 0.0]
        elif ch == 'T':
            one += [0.0, 0.0, 0.0, 1.0]
        else:
            one += [1.0, 1.0, 1.0, 1.0]
    return one * category_count

# ---- --batch mode: the 4 other real kernels a real tinygpuhybridtest run
# ---- queues alongside kernelMatrixMulADB in the same cmd_launch_batch,
# ---- in the same order. Grid/block/int-arg shapes read directly off a
# ---- real hardware launch log (not guessed) -- nPatterns=768 (kHuman's
# ---- length), categoryCount=4. Buffer sizes computed from each kernel's
# ---- own real index arithmetic (DETERMINE_INDICES_4_GPU/
# ---- DETERMINE_INTEGRATE_INDICES_4_GPU in kernels4.cu) with headroom --
# ---- these 4 kernels' own *output correctness* is irrelevant to this
# ---- test (only kernelMatrixMulADB's ground-truth dump is read back);
# ---- they only need to be real kernels touching real, correctly-sized,
# ---- non-faulting memory so the queuing pattern is genuine.
N_PATTERNS = 768
CATEGORY_COUNT = 4
PPNS_GRID, PPNS_BLOCK, PPNS_END_PATTERN = (12, 4, 1), (16, 16, 1), N_PATTERNS         # kernelPartialsPartialsNoScale
IL_GRID, IL_BLOCK = (48, 1, 1), (4, 16, 1)                                            # kernelIntegrateLikelihoods
SS_GRID, SS_BLOCK = (6, 1, 1), (128, 1, 1)                                            # kernelSumSites1
# u = tx + 16*(groupId0*16+patIdx) + matrix*4*endPattern, max at
# groupId0=11,patIdx=15,tx=15,matrix=3: 15+16*191+3*4*768 = 12287
PARTIALS_FLOATS = 16384
MATRIX_FLOATS = 128          # x2=16*matrix (matrix<=3) + tx(<=15) -> max index 63
ROOT_PARTIALS_FLOATS = 16384  # u+delta*r, r<matrixCount=4 -> max index 12287 (same bound as above)
WEIGHTS_FREQ_FLOATS = 16
RESULT_FLOATS = 1024          # patternCount=768, rounded up
SUM_ARRAY_FLOATS = 1024
SUM_OUT_FLOATS = 64

# ---- --realloc mode: BEAGLE's real ~15-buffer allocation set, in the
# ---- real order, with real (AlignMemOffset-rounded) sizes -- derived by
# ---- reading BeagleGPUImpl.hpp's constructor directly (not guessed),
# ---- for this test's exact beagleCreateInstance(3, 5, 0, 4, 768, 1, 8, 4,
# ---- 0, ...) config: kTipCount=3, kPartialsBufferCount=5,
# ---- kCompactBufferCount=0, kPatternCount=768, kMatrixCount=8 (the
# ---- --diag-matmul-ground-truth-doubled nMatrixBuffers), kCategoryCount=4,
# ---- kEigenDecompCount=1, kPaddedStateCount=4 (STATE_COUNT==4 is never
# ---- padded). AlignMemOffset(x) = (x+255) & ~255 -- BEAGLE's own 256-byte
# ---- stride rounding (GPUInterfaceTinyGPUHybrid.cpp), applied to each
# ---- per-slot *stride* before multiplying by slot count.
K_MATRIX_COUNT = 8
K_MATRIX_SIZE = STATE_COUNT * STATE_COUNT          # 16
K_EIGEN_DECOMP_COUNT = 1
K_EIGEN_VALUES_SIZE = 2 * STATE_COUNT              # 8 (conservative: real/complex-capable sizing)
K_PARTIALS_BUFFER_COUNT = 5
K_TIP_PARTIALS_BUFFER_COUNT = 3
K_PADDED_PATTERN_COUNT = N_PATTERNS                # 768, already block-aligned for this config
K_RESULT_PADDED_PATTERNS = 0
K_PARTIALS_SIZE = K_PADDED_PATTERN_COUNT * STATE_COUNT * CATEGORY_COUNT   # 12288
K_BUFFER_COUNT = K_PARTIALS_BUFFER_COUNT           # + kCompactBufferCount(0)
K_SUM_SITES_BLOCK_COUNT = N_PATTERNS // 128        # 6
PARTIALS_BUFFER_COUNT_TOTAL = max(K_PARTIALS_BUFFER_COUNT, 2 * K_TIP_PARTIALS_BUFFER_COUNT)  # 6
PTR_QUEUE_LENGTH = K_MATRIX_COUNT * CATEGORY_COUNT * 3 * 3                # 288
DISTANCE_QUEUE_LENGTH = max(K_MATRIX_COUNT * CATEGORY_COUNT * 2, K_MATRIX_COUNT + CATEGORY_COUNT)  # 64


def align_mem_offset(x):
    return (x + 255) & ~255


def log(msg):
    print(f"[nv_real_kernel_probe] {msg}", file=sys.stderr, flush=True)


def compile_real_kernel(nch, dev, nvcc, macros, maxrregcount=None):
    """Compiles the real, unmodified kernels4.cu (which #includes the real
    kernelsAll.cu) via the real nvcc shim, exactly matching
    make_tinygpu_kernels.sh's own SP_4 recipe, with the given extra
    -D<macro> flags. Returns the compiled ELF bytes.

    maxrregcount: TODO.md Phase 85 -- if set, forces ptxas's real
    register-allocation step (not the nvcc -ptx frontend above, which
    always emits virtual/unbounded-register PTX regardless) to cap
    kernelMatrixMulADB at this many registers per thread, inducing local-
    memory spill/reload if the real, unmodified kernel naturally needs
    more (it does -- regs_usage=40 uncapped, established throughout this
    investigation). Pure ptxas-flag bisection: zero kernelsAll.cu source
    changes, tests the register-pressure hypothesis TODO.md Phase 84
    left open."""
    kernels4_cu = _KERNELS_DIR / "kernels4.cu"
    out_ptx = _KERNELS_DIR / "tmp_real_kernel_probe.ptx"
    cmd = [nvcc, "-o", str(out_ptx), "--default-stream", "per-thread", "-ptx",
           "-DCUDA", "-DFW_TINYGPU", "-DSTATE_COUNT=4"]
    cmd += [f"-D{m}" for m in macros]
    cmd += [str(kernels4_cu), "-O3", "-Wno-deprecated-gpu-targets", "-DHAVE_CONFIG_H",
            f"-I{_REPO_ROOT}"]
    log(f"compiling: {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"nvcc failed: {r.stderr.decode(errors='replace')}")
    extra_ptxas_args = [f"-maxrregcount={maxrregcount}"] if maxrregcount else None
    try:
        elf_bytes = nch.compile_ptx(str(out_ptx), dev.arch, kernel_name="kernelMatrixMulADB",
                                     extra_ptxas_args=extra_ptxas_args)
    finally:
        if out_ptx.exists():
            out_ptx.unlink()
    return elf_bytes


# TODO.md Phase 140 (--dims-probe): the block slot is computed from the
# gx/gy *arguments*, not gridDim, so every block lands in its own slot even
# if gridDim reads 0 -- all writes in bounds regardless of what the dims
# words contain.
DIMS_PROBE_SRC = r'''
extern "C" __global__ void beagleDimsProbe(unsigned int* out, int gx, int gy) {
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        unsigned int* o = out + 9 * (blockIdx.x + gx * (blockIdx.y + gy * blockIdx.z));
        o[0] = blockDim.x; o[1] = blockDim.y; o[2] = blockDim.z;
        o[3] = gridDim.x;  o[4] = gridDim.y;  o[5] = gridDim.z;
        o[6] = blockIdx.x; o[7] = blockIdx.y; o[8] = blockIdx.z;
    }
}
'''
DIMS_PROBE_GRID = (3, 5, 7)
DIMS_PROBE_BLOCK = (2, 4, 8)


def compile_dims_probe(nch, nvcc, arch):
    """Compiles DIMS_PROBE_SRC via the same nvcc shim + nch.compile_ptx path
    compile_real_kernel uses (temp files in _KERNELS_DIR, visible to the
    Docker-backed shims). Returns the ELF bytes."""
    src = _KERNELS_DIR / "tmp_dims_probe.cu"
    out_ptx = _KERNELS_DIR / "tmp_dims_probe.ptx"
    try:
        src.write_text(DIMS_PROBE_SRC)
        cmd = [nvcc, "-o", str(out_ptx), "-ptx", str(src), "-O3", "-Wno-deprecated-gpu-targets"]
        log(f"compiling: {' '.join(cmd)}")
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(f"nvcc failed: {r.stderr.decode(errors='replace')}")
        return nch.compile_ptx(str(out_ptx), arch, kernel_name="beagleDimsProbe")
    finally:
        for f in (src, out_ptx):
            if f.exists():
                f.unlink()


def run_dims_probe(dev, nch, nvcc, BeagleNVProgram, TinyELF, Target, dtypes, HCQBuffer):
    """TODO.md Phase 140: tests the hypothesis that the cbuf0 words ptxas
    reads %ntid/%nctaid (blockDim/gridDim) from are never populated on this
    stack -- which would make kernelMatrixMulADB's BLOCKS=gridDim.y read 0
    and EDGE=20. One boot, two dispatches of beagleDimsProbe with
    grid=DIMS_PROBE_GRID, block=DIMS_PROBE_BLOCK: first with BeagleNVProgram's
    launch-dims fill forced OFF (upstream behavior), then forced ON
    (BEAGLE_NV_FILL_LAUNCH_DIMS is ignored here -- both states are always
    run). Output pre-filled with 0xFFFFFFFF so unwritten slots show up."""
    elf_bytes = compile_dims_probe(nch, nvcc, dev.arch)
    log(f"compiled beagleDimsProbe -- {len(elf_bytes)} byte ELF")
    prg = make_program(dev, TinyELF, Target, dtypes, BeagleNVProgram, elf_bytes, "beagleDimsProbe", 2)
    gx, gy, gz = DIMS_PROBE_GRID
    n_blocks = gx * gy * gz
    n_bytes = 9 * n_blocks * 4
    expected = (*DIMS_PROBE_BLOCK, *DIMS_PROBE_GRID)
    zeros = (0,) * 6
    verdicts = {}
    for fill in (False, True):
        prg.fill_launch_dims = fill
        out = dev.allocator.alloc(n_bytes)
        dev.allocator._copyin(HCQBuffer(out.va_addr, n_bytes), memoryview(bytearray(b"\xff" * n_bytes)))
        prg(HCQBuffer(out.va_addr, n_bytes), global_size=DIMS_PROBE_GRID, local_size=DIMS_PROBE_BLOCK,
            vals=(gx, gy), wait=False)
        dev.synchronize()
        raw = memoryview(bytearray(n_bytes))
        dev.allocator._copyout(raw, HCQBuffer(out.va_addr, n_bytes))
        words = struct.unpack(f"<{9 * n_blocks}I", bytes(raw))
        seen = Counter()
        n_unwritten = n_bad_idx = 0
        for lin in range(n_blocks):
            o = words[9 * lin:9 * lin + 9]
            if all(w == 0xFFFFFFFF for w in o):
                n_unwritten += 1
                continue
            if tuple(o[6:9]) != (lin % gx, (lin // gx) % gy, lin // (gx * gy)):
                n_bad_idx += 1
            seen[tuple(o[0:6])] += 1
        state = "ON" if fill else "OFF"
        log(f"dims-probe fill={state}: grid={DIMS_PROBE_GRID} block={DIMS_PROBE_BLOCK} -- "
            f"{n_blocks - n_unwritten}/{n_blocks} blocks wrote, {n_bad_idx} with wrong blockIdx; "
            f"(blockDim.xyz, gridDim.xyz) seen: {dict(seen)}")
        if n_unwritten or n_bad_idx or len(seen) != 1:
            verdicts[state] = f"ANOMALOUS ({n_unwritten} unwritten, {n_bad_idx} bad blockIdx, dims seen {dict(seen)})"
        elif set(seen) == {zeros}:
            verdicts[state] = "ZEROS"
        elif set(seen) == {expected}:
            verdicts[state] = "CORRECT"
        else:
            verdicts[state] = f"WRONG {next(iter(seen))} (expected {expected})"
    meaning = {("ZEROS", "CORRECT"): "hypothesis CONFIRMED and fill offsets verified on this GPU",
               ("CORRECT", "CORRECT"): "hypothesis REFUTED -- dims populated without the fill"}.get(
               (verdicts["OFF"], verdicts["ON"]), "unexpected -- see log")
    print(f"RESULT: dims-probe fill-OFF={verdicts['OFF']} fill-ON={verdicts['ON']} -- {meaning}", file=sys.stdout, flush=True)


def make_program(dev, TinyELF, Target, dtypes, BeagleNVProgram, elf_bytes, name, n_int_args):
    """Builds a BeagleNVProgram for `name` out of the *same* compiled ELF
    kernelMatrixMulADB came from -- kernels4.cu #includes kernelsAll.cu
    unconditionally, so every kernel a real tinygpuhybridtest run uses
    (kernelPartialsPartialsNoScale, kernelIntegrateLikelihoods,
    kernelSumSites1) is already present in this one compile; no second
    nvcc invocation needed. Same signature convention as
    nv_dispatch_daemon.py's _get_program."""
    signature = tuple((None, i, dtypes.uint32, ()) for i in range(n_int_args))
    obj = TinyELF(lib=elf_bytes, name=name, target=Target(), signature=signature)
    return BeagleNVProgram(dev, obj)


def alloc_zeroed(dev, HCQBuffer, n_floats):
    buf = dev.allocator.alloc(n_floats * 4)
    dev.allocator._copyin(HCQBuffer(buf.va_addr, n_floats * 4), memoryview(bytearray(n_floats * 4)))
    return buf


def alloc_real_pipeline_buffers(dev, HCQBuffer):
    """--realloc: allocates BEAGLE's real ~15-buffer set, in BEAGLE's real
    order, with BEAGLE's real (AlignMemOffset-rounded) sizes -- exactly as
    BeagleGPUImpl.hpp's constructor does for this test's exact
    beagleCreateInstance config, before kernelMatrixMulADB (or anything
    else) ever launches. Every buffer is zero-filled at this point; the
    caller writes real content into the specific slots kernelMatrixMulADB
    itself needs (dMatricesOrigin, dEvecOrigin, dIevcOrigin,
    dEigenValuesOrigin, dDistanceQueue, dPtrQueue) afterward. Returns a
    dict of {name: buf} for every allocation, in allocation order."""
    order = [
        ("dMatricesOrigin",      K_MATRIX_COUNT * align_mem_offset(K_MATRIX_SIZE * CATEGORY_COUNT * 4)),
        ("dEvecOrigin",          K_EIGEN_DECOMP_COUNT * align_mem_offset(K_MATRIX_SIZE * 4)),
        ("dIevcOrigin",          K_EIGEN_DECOMP_COUNT * align_mem_offset(K_MATRIX_SIZE * 4)),
        ("dEigenValuesOrigin",   K_EIGEN_DECOMP_COUNT * align_mem_offset(K_EIGEN_VALUES_SIZE * 4)),
        ("dWeightsOrigin",       K_EIGEN_DECOMP_COUNT * align_mem_offset(CATEGORY_COUNT * 4)),
        ("dFrequenciesOrigin",   K_EIGEN_DECOMP_COUNT * align_mem_offset(STATE_COUNT * 4)),
        ("dIntegrationTmp",      (K_PADDED_PATTERN_COUNT + K_RESULT_PADDED_PATTERNS) * 4),
        ("dPatternWeights",      N_PATTERNS * 4),
        ("dSumLogLikelihood",    K_SUM_SITES_BLOCK_COUNT * 4),
        ("dPartialsTmp",         K_PARTIALS_SIZE * 4),
        ("dPartialsTmpOrigin",   PARTIALS_BUFFER_COUNT_TOTAL * align_mem_offset(K_PARTIALS_SIZE * 4)),
        ("dBranchLengths",       K_BUFFER_COUNT * 4),
        ("dDistanceQueue",       DISTANCE_QUEUE_LENGTH * 4),
        ("dPtrQueue",            PTR_QUEUE_LENGTH * 4),
        ("dDerivativeQueue",     K_BUFFER_COUNT * 3 * 4),
    ]
    bufs = {}
    total = 0
    for name, size in order:
        b = dev.allocator.alloc(size)
        dev.allocator._copyin(HCQBuffer(b.va_addr, size), memoryview(bytearray(size)))
        bufs[name] = b
        total += size
        log(f"  --realloc: {name:22s} {size:8d} bytes  addr={b.va_addr:#x}")
    log(f"--realloc: {len(order)} real-pipeline buffers allocated, {total} bytes total "
        f"(dPartialsTmpOrigin alone: {dict(order)['dPartialsTmpOrigin']} bytes)")
    return bufs


def sub(buf, HCQBuffer, byte_off, byte_size):
    """CreateSubPointer(base, off, size) on the real NV/TinyGPU backend is
    pure base+off arithmetic (GPUInterfaceTinyGPUHybrid.cpp) -- confirmed
    by reading the source, not assumed. Same here."""
    return HCQBuffer(buf.va_addr + byte_off, byte_size)


def main():
    batch = False
    realloc = False
    logl = False
    logl_sweep = None
    chain_sweep = None
    sync_each = False
    sweep = None
    wide_grid = None
    maxrregcount = None
    downstream_sweep = None
    dims_probe = False
    argv = sys.argv[1:]
    while argv and (argv[0] in ("--batch", "--realloc", "--logl", "--logl-sweep", "--chain-sweep", "--sync-each", "--sweep", "--wide-grid", "--maxrregcount", "--downstream-sweep", "--dims-probe")):
        if argv[0] == "--batch":
            batch = True
        elif argv[0] == "--realloc":
            realloc = True
        elif argv[0] == "--logl":
            logl = True
        elif argv[0] == "--logl-sweep":
            # TODO.md Phase 88: a single --logl draw has no statistical
            # power (Phase 86's --sweep showed even maxrregcount=24 only
            # reaches full success 60% of the time, not 100%) -- repeats
            # the real 5-kernel chain N times (default 20, matching
            # --sweep's convention) and reports a real PASS/FAIL/NaN rate
            # instead of one draw's verdict.
            logl_sweep = 20
            argv = argv[1:]
            if argv and argv[0].isdigit():
                logl_sweep = int(argv[0])
                argv = argv[1:]
            continue
        elif argv[0] == "--chain-sweep":
            # TODO.md Phase 134, user: "narrow the chain" -- Phase 133
            # ruled out buffer management entirely (leak, churn, address
            # reuse) as the cause of --logl-sweep's own iteration-to-
            # iteration degradation; this localizes WHICH SUBSET of the
            # real 5-kernel chain (kernelMatrixMulADB, PPNS(tip_h,tip_c->
            # node3), PPNS(tip_g,node3->root4), IL(root4->result),
            # SS(result->sum), in that real dispatch order) is enough to
            # trigger it, rather than only having data at the extremes
            # (1 kernel: Phase 127, clean; 5 kernels: Phase 129-133,
            # degrades). Mandatory integer argument N (1-5): dispatches
            # only the first N of these 5 real calls, 20 iterations,
            # Phase 132's own allocate-once-reuse buffer pattern (already
            # proven not to be the confound), checking whether the Nth
            # stage's own output buffer stays consistent with its own
            # iteration-0 result -- an iteration-to-iteration DRIFT
            # check, not a fresh correctness-vs-CPU-reference check
            # (already established for the full chain via iteration 0's
            # own exact match in every --logl-sweep run so far).
            assert len(argv) > 1 and argv[1].isdigit() and 1 <= int(argv[1]) <= 5, \
                "--chain-sweep requires an integer stage count 1-5"
            chain_sweep = int(argv[1])
            argv = argv[2:]
            continue
        elif argv[0] == "--sync-each":
            # TODO.md Phase 136, user: "do we need sync's after each
            # kernel?" -- --chain-sweep (Phase 134/135) queues every
            # dispatch with wait=False and syncs only once at the end,
            # relying on the real command queue's own in-order-execution
            # guarantee for cross-kernel memory visibility (the same
            # assumption --logl-sweep's own real 5-kernel chain has
            # always made). Phase 135's own drift pattern -- exact,
            # stable plateaus (0 -> 15 -> 41) rather than growing noise
            # -- is a real, specific hint that a downstream kernel might
            # occasionally be reading a stale (not-yet-visible) version
            # of an upstream kernel's write on this from-scratch driver
            # stack. This flag adds an explicit dev.synchronize() after
            # EVERY dispatched stage in --chain-sweep's own loop (only
            # -- --logl-sweep/--sweep untouched), isolating exactly this
            # one variable against Phase 135's own no-intermediate-sync
            # baseline.
            sync_each = True
        elif argv[0] == "--downstream-sweep":
            # TODO.md Phase 99: user asked to double-check PPNS/IL/SS in
            # isolation by substituting *known-correct* transition
            # matrices (the same reference_transition_matrix() formula
            # Phase 90 independently verified against the closed-form
            # JC69 result) instead of dispatching kernelMatrixMulADB at
            # all -- zero risk from that kernel's own fault-prone
            # address-computation code path (Phase 94-98), a completely
            # clean test of whether PPNS/IL/SS themselves are reliable
            # given guaranteed-correct inputs. Repeated N times (default
            # 20) for real statistical power, matching --logl-sweep's
            # own established discipline.
            downstream_sweep = 20
            argv = argv[1:]
            if argv and argv[0].isdigit():
                downstream_sweep = int(argv[0])
                argv = argv[1:]
            continue
        elif argv[0] == "--dims-probe":
            # TODO.md Phase 140: does anything populate the cbuf0 words
            # ptxas reads blockDim/gridDim from? No dispatch path writes
            # them. Tiny standalone kernel (no kernels4.cu compile, no BEAGLE
            # kernel), one boot, dispatched twice: launch-dims fill OFF,
            # then ON -- see run_dims_probe().
            dims_probe = True
        elif argv[0] == "--maxrregcount":
            # TODO.md Phase 85: forces ptxas to cap kernelMatrixMulADB's
            # real, unmodified register allocation (naturally 40,
            # uncapped) at N, inducing local-memory spill if N is below
            # that -- pure compiler-flag bisection, tests the register-
            # pressure hypothesis TODO.md Phase 84 left
            # open. Mandatory value, no sensible default cap.
            maxrregcount = int(argv[1])
            argv = argv[2:]
            continue
        elif argv[0] == "--sweep":
            sweep = 20  # default sample count
            argv = argv[1:]
            if argv and argv[0].isdigit():
                sweep = int(argv[0])
                argv = argv[1:]
            continue
        else:
            wide_grid = 32  # --wide-grid [N]: default block count
            argv = argv[1:]
            if argv and argv[0].isdigit():
                wide_grid = int(argv[0])
                argv = argv[1:]
            continue
        argv = argv[1:]
    macros = BASE_MACROS + argv
    # The loop above stops at the first token it does not know and passes that
    # token and everything after it to nvcc as -D macros. Reject flags here,
    # before the GPU boots.
    bad = [a for a in argv if a.startswith("-")]
    if bad:
        sys.exit(f"unknown or removed flag(s), or flags after the first macro: {bad}")

    os.makedirs(os.path.expanduser("~/Library/Logs"), exist_ok=True)
    fd = os.open(os.path.expanduser("~/Library/Logs/nv_real_kernel_probe.log"),
                 os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_SYNC, 0o644)
    sys.stderr = os.fdopen(fd, 'w', buffering=1)

    log(f"starting -- batch={batch} realloc={realloc} logl={logl} logl_sweep={logl_sweep} chain_sweep={chain_sweep} sync_each={sync_each} downstream_sweep={downstream_sweep} sweep={sweep} wide_grid={wide_grid} maxrregcount={maxrregcount} dims_probe={dims_probe} macros={macros}")
    import nv_init_helper  # noqa: F401 -- GSP/RM boot safety patches (module-level side effects)
    from tinygrad.runtime.support.system import APLRemotePCIDevice
    def _safe_reset(self):
        log("PCIe FLR suppressed (macOS eGPU safety)")
    APLRemotePCIDevice.reset = _safe_reset

    from tinygrad.helpers import DEV
    DEV.value = "NV"
    from tinygrad import Device
    dev = Device["NV:0"]
    log(f"booted -- {dev}, arch={dev.arch}, renderer={type(dev.renderer).__name__}")
    global _dev_for_diagnostics
    _dev_for_diagnostics = dev

    # TODO.md Phase 107: user asked to double-check whether tinygrad's
    # driver code hardcodes a "4" anywhere relevant to grid/block size --
    # traced through (Phase 107 finding: it doesn't; the real dispatch
    # path (ops_nv.py's NVComputeQueue.exec()) correctly writes the real
    # (16,16,1) BLOCK shape into the QMD's cta_thread_dimension0/1/2
    # fields, and local-memory backing-store sizing uses topology
    # constants from a genuine RM-control hardware query
    # (_query_gpu_info), not a hardcoded value). This prints those
    # queried values directly -- read-only, zero dispatch risk, no new
    # kernel/hardware exposure -- to check whether the *values themselves*
    # (as opposed to the Python code computing with them) are the
    # anomaly, e.g. if this from-scratch GSP-RM stack's RM-control
    # response decoding returns a wrong topology constant (a real "4"
    # showing up here, on real hardware, unexplained by any of this
    # session's source-level tracing, would be a direct, actionable
    # lead -- whereas expected-looking values would rule this out too).
    try:
        topo_line = (f"GPU topology (from real _query_gpu_info RM-control query): "
                     f"num_gpcs={dev.num_gpcs} num_tpc_per_gpc={dev.num_tpc_per_gpc} "
                     f"num_sm_per_tpc={dev.num_sm_per_tpc} max_warps_per_sm={dev.max_warps_per_sm} "
                     f"sm_version={dev.sm_version:#x} "
                     f"(total SMs = num_gpcs*num_tpc_per_gpc*num_sm_per_tpc = "
                     f"{dev.num_gpcs * dev.num_tpc_per_gpc * dev.num_sm_per_tpc})")
        log(topo_line)
        print(f"[nv_real_kernel_probe] {topo_line}", file=sys.stdout, flush=True)
    except Exception as topo_exc:
        topo_err = f"GPU topology query failed (non-fatal, diagnostic only): {topo_exc!r}"
        log(topo_err)
        print(f"[nv_real_kernel_probe] {topo_err}", file=sys.stdout, flush=True)

    import nv_compile_helper as nch
    from nv_dispatch_daemon import BeagleNVProgram
    from tinygrad.device import TinyELF, Target
    from tinygrad.dtype import dtypes
    from tinygrad.runtime.support.hcq import HCQBuffer

    nvcc = os.environ.get("TINYGPU_NVCC", os.path.expanduser("~/.local/bin/nvcc"))

    if dims_probe:
        run_dims_probe(dev, nch, nvcc, BeagleNVProgram, TinyELF, Target, dtypes, HCQBuffer)
        return

    elf_bytes = compile_real_kernel(nch, dev, nvcc, macros, maxrregcount=maxrregcount)
    log(f"compiled kernelMatrixMulADB -- {len(elf_bytes)} byte ELF")

    # 3 trailing scalar (length, wB, totalMatrix) uint32 args -- matches
    # BEAGLE's KernelLauncher.cpp calling convention, same signature
    # scheme nv_dispatch_daemon.py's _get_program uses.
    signature = tuple((None, i, dtypes.uint32, ()) for i in range(3))
    obj = TinyELF(lib=elf_bytes, name="kernelMatrixMulADB", target=Target(), signature=signature)
    prg = BeagleNVProgram(dev, obj)
    log(f"regs_usage={prg.regs_usage} shmem_usage={prg.shmem_usage} lcmem_usage={prg.lcmem_usage}")

    dispatch_local_size = BLOCK

    if sweep:
        # ---- Phase 71 (sweep): one boot, one compile, then `sweep` fresh
        # dispatches of the real kernelMatrixMulADB alone (no downstream
        # kernels), fresh buffers every iteration, each checked per wMatrix
        # against the closed-form reference matrix and tabulated per SM.
        distance_vals = [EDGE_LENS[i] * CATEGORY_RATES[j] for i in range(4) for j in range(4)]

        # TODO.md Phase 78: `--wide-grid [N]` (default N=32) launches N
        # blocks instead of 16. The kernel's `wMatrix = blockIdx.x %
        # totalMatrix`, so passing `totalMatrix=n_blocks` with an N-block
        # grid keeps every block's `bx = blockIdx.x / totalMatrix` at 0 and
        # needs no kernelsAll.cu change. distance_vals/listC are extended
        # cyclically (block 16 copies block 0's setup, 17 copies 1, etc.) so
        # every block gets a valid, unique output slot; A/B/D stay the same
        # shared buffers.
        n_blocks = wide_grid if wide_grid else TOTAL_MATRIX
        if wide_grid:
            distance_vals = [distance_vals[w % TOTAL_MATRIX] for w in range(n_blocks)]
            log(f"--wide-grid {n_blocks}: grid=({n_blocks},1,1), totalMatrix={n_blocks}, "
                f"distance_vals cyclically extended (block w uses block w%{TOTAL_MATRIX}'s value)")
        listc_vals = [w * S2 for w in range(n_blocks)]
        n_dmat_floats = 2 * n_blocks * S2
        dmat_init = [0.0] * (n_blocks * S2) + [SENTINEL] * (n_blocks * S2)

        # TODO.md Phase 90: every prior --sweep run only ever checked
        # "wrote anything nonzero" -- never whether the value written was
        # actually the *correct* transition matrix. Computed once here
        # (distance_vals is fixed for the whole run) so every iteration's
        # real C[] readback can be checked against the real answer, not
        # just against zero.
        reference_matrices = [reference_transition_matrix(distance_vals[w]) for w in range(n_blocks)]

        success_count = [0] * n_blocks     # any of the 16 entries nonzero (the old, weaker metric -- kept for comparability)
        correct_count = [0] * n_blocks     # all 16 entries match the real reference matrix within CORRECTNESS_TOL
        wrong_count = [0] * n_blocks       # wrote nonzero but does NOT match the reference -- a *real* failure, not a proxy
        full_row0_count = [0] * n_blocks   # row ty=0 (entries 0-3) all nonzero (the old, weaker per-row metric)
        all_populated = []  # per-iteration set of populated wMatrix, for exact-pattern comparison
        iter_times = []     # wall-clock seconds for dispatch+synchronize, per iteration -- a real,
                             # independent signal of GPU clock/DVFS warm-up state, to test directly
                             # whether success correlates with iteration index / dispatch speed
                             # rather than just asserting a "warm-up" story from the pattern alone.

        # STATUS.md #135: a claimed wMatrix->SMID->TPC pattern was assembled
        # from memory of earlier pasted output, not re-extracted from logs
        # mechanically -- flagged as unverified. dbg[13] (the ground-truth
        # scratch region's %smid capture in kernelsAll.cu's GROUND_TRUTH block) is
        # already inside dmat's second half, which this loop already reads
        # back every iteration (n_dmat_floats covers both halves) -- it was
        # just never extracted. Capturing it here, per wMatrix per
        # iteration, gets real per-run evidence for/against that pattern
        # instead of relying on memory.
        ran_count = [0] * n_blocks          # dbg[0] != SENTINEL: block's tx==0,ty==0 thread executed
        smid_by_w = [Counter() for _ in range(n_blocks)]   # wMatrix -> Counter of observed SMIDs
        smid_stats = defaultdict(lambda: [0, 0, 0])  # smid -> [ran_count, wrote_nonzero_count, correct_count]

        print(f"\n=== nv_real_kernel_probe --sweep {sweep}: per-iteration sequence ===", file=sys.stdout, flush=True)
        for it in range(sweep):
            t0 = time.time()
            a = dev.allocator.alloc(len(EVEC) * 4)
            dev.allocator._copyin(HCQBuffer(a.va_addr, len(EVEC) * 4), memoryview(struct.pack(f"<{len(EVEC)}f", *EVEC)))
            b = dev.allocator.alloc(len(IVEC) * 4)
            dev.allocator._copyin(HCQBuffer(b.va_addr, len(IVEC) * 4), memoryview(struct.pack(f"<{len(IVEC)}f", *IVEC)))
            d = dev.allocator.alloc(len(EVAL) * 4)
            dev.allocator._copyin(HCQBuffer(d.va_addr, len(EVAL) * 4), memoryview(struct.pack(f"<{len(EVAL)}f", *EVAL)))
            distq = dev.allocator.alloc(n_blocks * 4)
            dev.allocator._copyin(HCQBuffer(distq.va_addr, n_blocks * 4), memoryview(struct.pack(f"<{n_blocks}f", *distance_vals)))
            listc = dev.allocator.alloc(n_blocks * 4)
            dev.allocator._copyin(HCQBuffer(listc.va_addr, n_blocks * 4), memoryview(struct.pack(f"<{n_blocks}I", *listc_vals)))
            dmat = dev.allocator.alloc(n_dmat_floats * 4)
            dev.allocator._copyin(HCQBuffer(dmat.va_addr, n_dmat_floats * 4), memoryview(struct.pack(f"<{n_dmat_floats}f", *dmat_init)))

            prg(HCQBuffer(dmat.va_addr, n_dmat_floats * 4), HCQBuffer(listc.va_addr, n_blocks * 4),
                HCQBuffer(a.va_addr, len(EVEC) * 4), HCQBuffer(d.va_addr, len(EVAL) * 4),
                HCQBuffer(b.va_addr, len(IVEC) * 4), HCQBuffer(distq.va_addr, n_blocks * 4),
                global_size=(n_blocks, 1, 1), local_size=dispatch_local_size, vals=(STATE_COUNT, STATE_COUNT, n_blocks), wait=False)
            dev.synchronize()
            elapsed = time.time() - t0
            iter_times.append(elapsed)

            dmat_out = memoryview(bytearray(n_dmat_floats * 4))
            dev.allocator._copyout(dmat_out, HCQBuffer(dmat.va_addr, n_dmat_floats * 4))
            dmat_vals = struct.unpack(f"<{n_dmat_floats}f", bytes(dmat_out))
            real_matrices = dmat_vals[:n_blocks * S2]
            # dbg[] scratch region (kernelsAll.cu's TINYGPU_DEBUG_DUMP_
            # MATMUL_GROUND_TRUTH block): dMatrices + totalMatrix*S2 +
            # wMatrix*S2, dbg[0]=csub0 (written whenever tx==0,ty==0
            # actually executes), dbg[13]=%smid -- already inside the
            # dmat this loop already reads back, just never extracted.
            scratch = dmat_vals[n_blocks * S2:2 * n_blocks * S2]

            populated_this_iter = []
            for w in range(n_blocks):
                m = real_matrices[w * S2:(w + 1) * S2]
                wrote_nonzero = any(v != 0.0 for v in m)
                is_correct = False
                if wrote_nonzero:
                    success_count[w] += 1
                    populated_this_iter.append(w)
                    if all(v != 0.0 for v in m[0:4]):
                        full_row0_count[w] += 1
                    is_correct = all(abs(m[i] - reference_matrices[w][i]) < CORRECTNESS_TOL for i in range(S2))
                    if is_correct:
                        correct_count[w] += 1
                    else:
                        wrong_count[w] += 1
                dbg = scratch[w * S2:(w + 1) * S2]
                ran = dbg[0] != SENTINEL
                if ran:
                    ran_count[w] += 1
                    smid = int(dbg[13]) if dbg[13] != SENTINEL else None
                    if smid is not None:
                        smid_by_w[w][smid] += 1
                        stats = smid_stats[smid]
                        stats[0] += 1
                        if wrote_nonzero:
                            stats[1] += 1
                        if is_correct:
                            stats[2] += 1
            if it == 0 and populated_this_iter:
                # TODO.md Phase 91: one-time (first iteration, first
                # wrote-nonzero wMatrix only -- bounded, cheap), raw
                # side-by-side of what the real kernel actually wrote vs.
                # the reference, to see *how* it's wrong (garbage-level
                # vs. a subtle GPU-fast-math-exp()-precision difference)
                # before trusting a bare 0%-CORRECT verdict at face value.
                w0 = populated_this_iter[0]
                m0 = real_matrices[w0 * S2:(w0 + 1) * S2]
                ref0 = reference_matrices[w0]
                header = f"\n  --- one-time raw diagnostic (iter 0, wMatrix {w0}, distance={distance_vals[w0]:.6f}) ---"
                log(header)
                print(header, file=sys.stdout, flush=True)
                for row in range(STATE_COUNT):
                    real_row = [f"{v:.6f}" for v in m0[row * STATE_COUNT:(row + 1) * STATE_COUNT]]
                    ref_row = [f"{v:.6f}" for v in ref0[row * STATE_COUNT:(row + 1) * STATE_COUNT]]
                    delta_row = [f"{abs(a - b):.6f}" for a, b in zip(m0[row * STATE_COUNT:(row + 1) * STATE_COUNT],
                                                                      ref0[row * STATE_COUNT:(row + 1) * STATE_COUNT])]
                    line = f"    row {row}: real={real_row}  ref={ref_row}  |delta|={delta_row}"
                    log(line)
                    print(line, file=sys.stdout, flush=True)
            all_populated.append(tuple(populated_this_iter))
            line = (f"  iter {it:2d}: {len(populated_this_iter):2d}/{n_blocks} populated "
                    f"({'FULL' if len(populated_this_iter) == n_blocks else 'partial'})  "
                    f"dispatch+sync={elapsed*1000:7.2f}ms  populated={populated_this_iter}")
            log(line)
            print(line, file=sys.stdout, flush=True)

        # Direct, independent test of the warm-up/DVFS-ramp hypothesis
        # (every separate-process --logl run showed only the restricted
        # partial pattern; this --sweep's own later iterations reached
        # full success) -- split into first/second half and compare both
        # the success COUNT (does it trend up?) and the actual dispatch+
        # sync WALL-CLOCK TIME (does it trend down, i.e. does the GPU
        # genuinely get faster?), rather than just eyeballing the
        # per-iteration lines above.
        half = sweep // 2
        if half > 0:   # --sweep 1 (e.g. just for the one-time raw diagnostic) has no "two halves" to compare
            counts = [len(p) for p in all_populated]
            first_half_counts, second_half_counts = counts[:half], counts[half:]
            first_half_times, second_half_times = iter_times[:half], iter_times[half:]
            print("\n=== warm-up check: first half vs second half of the sweep ===", file=sys.stdout, flush=True)
            line = (f"  populated count: first half avg={sum(first_half_counts)/len(first_half_counts):.2f}/{n_blocks}  "
                    f"second half avg={sum(second_half_counts)/len(second_half_counts):.2f}/{n_blocks}")
            log(line)
            print(line, file=sys.stdout, flush=True)
            line = (f"  dispatch+sync time: first half avg={1000*sum(first_half_times)/len(first_half_times):.2f}ms  "
                    f"second half avg={1000*sum(second_half_times)/len(second_half_times):.2f}ms")
            log(line)
            print(line, file=sys.stdout, flush=True)

        print(f"\n=== nv_real_kernel_probe --sweep {sweep}: per-wMatrix success rate ===", file=sys.stdout, flush=True)
        for w in range(n_blocks):
            edge, cat = (w % TOTAL_MATRIX) // 4, (w % TOTAL_MATRIX) % 4
            copy_note = f"  (structural copy of wMatrix {w % TOTAL_MATRIX})" if w >= TOTAL_MATRIX else ""
            line = (f"  wMatrix {w:2d} (edge={edge} cat={cat}): "
                    f"{success_count[w]:2d}/{sweep} wrote anything ({100*success_count[w]/sweep:5.1f}%)  "
                    f"{correct_count[w]:2d}/{sweep} CORRECT ({100*correct_count[w]/sweep:5.1f}%)  "
                    f"{wrong_count[w]:2d}/{sweep} wrote-but-WRONG ({100*wrong_count[w]/sweep:5.1f}%)  "
                    f"{full_row0_count[w]:2d}/{sweep} row ty=0 fully populated{copy_note}")
            log(line)
            print(line, file=sys.stdout, flush=True)
        distinct_patterns = sorted(set(all_populated), key=lambda t: (-all_populated.count(t), t))
        line = f"  {len(set(all_populated))} distinct per-iteration populated-set(s) across {sweep} iterations"
        log(line)
        print(line, file=sys.stdout, flush=True)
        for pat in distinct_patterns:
            line = f"    {all_populated.count(pat):2d}x: {pat}"
            log(line)
            print(line, file=sys.stdout, flush=True)

        # STATUS.md #135's wMatrix->SMID->TPC claim, checked mechanically
        # against this run's own dbg[13] captures instead of memory of
        # earlier pasted output. Two independent questions: (1) is each
        # wMatrix pinned to one fixed physical SM across iterations, or
        # does it vary? (2) does success rate actually cluster by SMID/TPC
        # (TPC = SMID // 2, i.e. two SMs per TPC on this part), rather than
        # by wMatrix/blockIdx.x per se?
        print(f"\n=== nv_real_kernel_probe --sweep {sweep}: per-wMatrix observed SMID(s) (from dbg[13], mechanical) ===", file=sys.stdout, flush=True)
        for w in range(n_blocks):
            fixed = len(smid_by_w[w]) <= 1
            smid_desc = ", ".join(f"smid={s}x{n}" for s, n in smid_by_w[w].most_common())
            line = (f"  wMatrix {w:2d}: ran {ran_count[w]:2d}/{sweep}  "
                    f"{'FIXED' if fixed else 'VARIES'} SMID  {smid_desc if smid_desc else '(never ran)'}")
            log(line)
            print(line, file=sys.stdout, flush=True)

        print(f"\n=== nv_real_kernel_probe --sweep {sweep}: per-SMID / per-TPC success rate (mechanical) ===", file=sys.stdout, flush=True)
        tpc_stats = defaultdict(lambda: [0, 0, 0])
        for smid, (ran, wrote, correct) in smid_stats.items():
            tpc = smid // 2
            tpc_stats[tpc][0] += ran
            tpc_stats[tpc][1] += wrote
            tpc_stats[tpc][2] += correct
        for smid in sorted(smid_stats):
            ran, wrote, correct = smid_stats[smid]
            line = (f"  smid {smid:2d} (tpc {smid // 2}): {wrote:3d}/{ran:3d} wrote-nonzero ({100*wrote/ran:5.1f}%)  "
                    f"{correct:3d}/{ran:3d} CORRECT ({100*correct/ran:5.1f}%)")
            log(line)
            print(line, file=sys.stdout, flush=True)
        for tpc in sorted(tpc_stats):
            ran, wrote, correct = tpc_stats[tpc]
            line = (f"  tpc {tpc:2d} total: {wrote:3d}/{ran:3d} wrote-nonzero ({100*wrote/ran:5.1f}%)  "
                    f"{correct:3d}/{ran:3d} CORRECT ({100*correct/ran:5.1f}%)")
            log(line)
            print(line, file=sys.stdout, flush=True)

        log("exiting cleanly")
        return

    distance_vals = [EDGE_LENS[i] * CATEGORY_RATES[j] for i in range(4) for j in range(4)]
    assert len(distance_vals) == TOTAL_MATRIX
    listc_vals = [w * S2 for w in range(TOTAL_MATRIX)]
    dmat_init = [0.0] * (TOTAL_MATRIX * S2) + [SENTINEL] * (TOTAL_MATRIX * S2)
    n_dmat_floats = len(dmat_init)   # 512 -- identical either way (see below)

    real_bufs = None
    if not realloc:
        # ---- Phase 68/69 behavior, unchanged: ad-hoc buffers sized only
        # for what kernelMatrixMulADB itself needs. ----
        a = dev.allocator.alloc(len(EVEC) * 4)
        dev.allocator._copyin(HCQBuffer(a.va_addr, len(EVEC) * 4), memoryview(struct.pack(f"<{len(EVEC)}f", *EVEC)))
        b = dev.allocator.alloc(len(IVEC) * 4)
        dev.allocator._copyin(HCQBuffer(b.va_addr, len(IVEC) * 4), memoryview(struct.pack(f"<{len(IVEC)}f", *IVEC)))
        d = dev.allocator.alloc(len(EVAL) * 4)
        dev.allocator._copyin(HCQBuffer(d.va_addr, len(EVAL) * 4), memoryview(struct.pack(f"<{len(EVAL)}f", *EVAL)))
        distq = dev.allocator.alloc(TOTAL_MATRIX * 4)
        dev.allocator._copyin(HCQBuffer(distq.va_addr, TOTAL_MATRIX * 4), memoryview(struct.pack(f"<{TOTAL_MATRIX}f", *distance_vals)))
        listc = dev.allocator.alloc(TOTAL_MATRIX * 4)
        dev.allocator._copyin(HCQBuffer(listc.va_addr, TOTAL_MATRIX * 4), memoryview(struct.pack(f"<{TOTAL_MATRIX}I", *listc_vals)))
        dmat = dev.allocator.alloc(n_dmat_floats * 4)
        dev.allocator._copyin(HCQBuffer(dmat.va_addr, n_dmat_floats * 4), memoryview(struct.pack(f"<{n_dmat_floats}f", *dmat_init)))
    else:
        # ---- Phase 70: BEAGLE's real ~15-buffer allocation set, in
        # BEAGLE's real order, real sizes -- allocated *first*, exactly
        # like beagleCreateInstance does, before kernelMatrixMulADB's own
        # inputs are written. kernelMatrixMulADB's own arguments are then
        # sub-pointers *into* this real set (dMatricesOrigin, dEvecOrigin,
        # dIevcOrigin, dEigenValuesOrigin, dDistanceQueue, dPtrQueue) --
        # not separate allocations -- exactly matching how BEAGLE itself
        # sources kernelMatrixMulADB's real arguments.
        real_bufs = alloc_real_pipeline_buffers(dev, HCQBuffer)
        dmat = real_bufs["dMatricesOrigin"]
        a = real_bufs["dEvecOrigin"]
        b = real_bufs["dIevcOrigin"]
        d = real_bufs["dEigenValuesOrigin"]
        distq = real_bufs["dDistanceQueue"]
        listc = real_bufs["dPtrQueue"]
        assert n_dmat_floats * 4 == K_MATRIX_COUNT * align_mem_offset(K_MATRIX_SIZE * CATEGORY_COUNT * 4), \
            "dMatricesOrigin's real size must exactly match this probe's own real+scratch region size"
        dev.allocator._copyin(HCQBuffer(dmat.va_addr, n_dmat_floats * 4), memoryview(struct.pack(f"<{n_dmat_floats}f", *dmat_init)))
        dev.allocator._copyin(HCQBuffer(a.va_addr, len(EVEC) * 4), memoryview(struct.pack(f"<{len(EVEC)}f", *EVEC)))
        dev.allocator._copyin(HCQBuffer(b.va_addr, len(IVEC) * 4), memoryview(struct.pack(f"<{len(IVEC)}f", *IVEC)))
        dev.allocator._copyin(HCQBuffer(d.va_addr, len(EVAL) * 4), memoryview(struct.pack(f"<{len(EVAL)}f", *EVAL)))
        dev.allocator._copyin(HCQBuffer(distq.va_addr, TOTAL_MATRIX * 4), memoryview(struct.pack(f"<{TOTAL_MATRIX}f", *distance_vals)))
        dev.allocator._copyin(HCQBuffer(listc.va_addr, TOTAL_MATRIX * 4), memoryview(struct.pack(f"<{TOTAL_MATRIX}I", *listc_vals)))

    log(f"buffers allocated: dMatrices={dmat.va_addr:#x} listC={listc.va_addr:#x} A={a.va_addr:#x} "
        f"D={d.va_addr:#x} B={b.va_addr:#x} distanceQueue={distq.va_addr:#x}")
    log(f"distanceQueue values: {distance_vals}")

    if downstream_sweep:
        # ---- Phase 99: user asked to double-check PPNS/IL/SS in isolation
        # by substituting KNOWN-CORRECT transition matrices (the same
        # reference_transition_matrix() formula Phase 90 independently
        # verified against the closed-form JC69 result) instead of
        # dispatching kernelMatrixMulADB at all. That kernel is the one
        # that has faulted the hardware three times this session (Phase
        # 94-98); never calling it here makes this a zero-risk test of
        # whether PPNS/IL/SS themselves are reliable given guaranteed-
        # correct inputs.
        ppns = make_program(dev, TinyELF, Target, dtypes, BeagleNVProgram, elf_bytes, "kernelPartialsPartialsNoScale", 1)
        il = make_program(dev, TinyELF, Target, dtypes, BeagleNVProgram, elf_bytes, "kernelIntegrateLikelihoods", 2)
        ss = make_program(dev, TinyELF, Target, dtypes, BeagleNVProgram, elf_bytes, "kernelSumSites1", 1)

        def buf(a_, n): return HCQBuffer(a_.va_addr, n * 4)

        tip_h = make_tip_partials(K_HUMAN, CATEGORY_COUNT)
        tip_c = make_tip_partials(K_CHIMP, CATEGORY_COUNT)
        tip_g = make_tip_partials(K_GORILLA, CATEGORY_COUNT)

        def alloc_filled(vals, fmt):
            b = dev.allocator.alloc(len(vals) * struct.calcsize(fmt))
            dev.allocator._copyin(HCQBuffer(b.va_addr, len(vals) * struct.calcsize(fmt)), memoryview(struct.pack(f"<{len(vals)}{fmt}", *vals)))
            return b

        mstride = align_mem_offset(K_MATRIX_SIZE * CATEGORY_COUNT * 4)

        def matrix_of(dmat_buf, edge_index):
            return sub(dmat_buf, HCQBuffer, edge_index * mstride, K_MATRIX_SIZE * CATEGORY_COUNT * 4)

        # Build the flat, known-correct replacement for everything
        # kernelMatrixMulADB would normally have written: 16 wMatrix
        # slots, S2=16 floats each, laid out at float offset w*S2 --
        # exactly the layout matrix_of()/listC assume.
        correct_matrices_flat = []
        for w in range(TOTAL_MATRIX):
            correct_matrices_flat.extend(reference_transition_matrix(distance_vals[w]))
        assert len(correct_matrices_flat) == TOTAL_MATRIX * S2

        log(f"--downstream-sweep: kernelMatrixMulADB will NEVER be dispatched; "
            f"dMatrices is pre-filled with {TOTAL_MATRIX} known-correct reference matrices")

        n_pass = n_fail_finite = n_nan = 0
        for it in range(downstream_sweep):
            tip_h_buf = alloc_filled(tip_h, "f")
            tip_c_buf = alloc_filled(tip_c, "f")
            tip_g_buf = alloc_filled(tip_g, "f")
            node3_buf = alloc_zeroed(dev, HCQBuffer, K_PARTIALS_SIZE)
            root4_buf = alloc_zeroed(dev, HCQBuffer, K_PARTIALS_SIZE)
            weights_buf = alloc_filled(CATEGORY_WEIGHTS, "f")
            freqs_buf = alloc_filled(STATE_FREQS, "f")
            patw_buf = alloc_filled([1.0] * N_PATTERNS, "f")
            result_buf = alloc_zeroed(dev, HCQBuffer, N_PATTERNS)
            sum_buf = alloc_zeroed(dev, HCQBuffer, K_SUM_SITES_BLOCK_COUNT)

            # dmat_it holds ONLY the real-matrix region, pre-filled with
            # known-correct values -- no ground-truth scratch region is
            # needed since kernelMatrixMulADB is never dispatched.
            dmat_it = dev.allocator.alloc(TOTAL_MATRIX * S2 * 4)
            dev.allocator._copyin(HCQBuffer(dmat_it.va_addr, TOTAL_MATRIX * S2 * 4),
                                  memoryview(struct.pack(f"<{TOTAL_MATRIX * S2}f", *correct_matrices_flat)))

            ppns(buf(tip_h_buf, K_PARTIALS_SIZE), buf(tip_c_buf, K_PARTIALS_SIZE), buf(node3_buf, K_PARTIALS_SIZE),
                 matrix_of(dmat_it, 0), matrix_of(dmat_it, 1),
                 global_size=PPNS_GRID, local_size=PPNS_BLOCK, vals=(PPNS_END_PATTERN,), wait=False)
            ppns(buf(tip_g_buf, K_PARTIALS_SIZE), buf(node3_buf, K_PARTIALS_SIZE), buf(root4_buf, K_PARTIALS_SIZE),
                 matrix_of(dmat_it, 2), matrix_of(dmat_it, 3),
                 global_size=PPNS_GRID, local_size=PPNS_BLOCK, vals=(PPNS_END_PATTERN,), wait=False)
            il(buf(result_buf, N_PATTERNS), buf(root4_buf, K_PARTIALS_SIZE), buf(weights_buf, CATEGORY_COUNT), buf(freqs_buf, STATE_COUNT),
               global_size=IL_GRID, local_size=IL_BLOCK, vals=(CATEGORY_COUNT, N_PATTERNS), wait=False)
            ss(buf(result_buf, N_PATTERNS), buf(sum_buf, K_SUM_SITES_BLOCK_COUNT), buf(patw_buf, N_PATTERNS),
               global_size=SS_GRID, local_size=SS_BLOCK, vals=(N_PATTERNS,), wait=False)
            dev.synchronize()

            sum_out = memoryview(bytearray(K_SUM_SITES_BLOCK_COUNT * 4))
            dev.allocator._copyout(sum_out, HCQBuffer(sum_buf.va_addr, K_SUM_SITES_BLOCK_COUNT * 4))
            block_sums = struct.unpack(f"<{K_SUM_SITES_BLOCK_COUNT}f", bytes(sum_out))
            computed_logl = sum(block_sums)
            delta = abs(computed_logl - K_REF)

            if computed_logl != computed_logl:  # NaN self-inequality, no math.isnan import needed
                status = "NAN"
                n_nan += 1
            elif delta < 0.5:
                status = "PASS"
                n_pass += 1
            else:
                status = "FAIL"
                n_fail_finite += 1
            line = (f"  iter {it:2d}: logL={computed_logl:12.5f}  delta={delta:10.5f}  {status}  "
                    f"block_sums={[f'{v:g}' for v in block_sums]}")
            log(line)

        summary = (f"  {n_pass}/{downstream_sweep} PASS ({100*n_pass/downstream_sweep:.1f}%)  "
                   f"{n_fail_finite}/{downstream_sweep} FAIL ({100*n_fail_finite/downstream_sweep:.1f}%)  "
                   f"{n_nan}/{downstream_sweep} NAN ({100*n_nan/downstream_sweep:.1f}%)")
        log(summary)
        print(f"RESULT: downstream-sweep {'PASS' if n_pass == downstream_sweep else 'FAIL'} -- {summary.strip()}", file=sys.stdout, flush=True)
        return

    if logl_sweep:
        # ---- Phase 88: a single --logl draw has no statistical power --
        # Phase 86's --sweep showed even maxrregcount=24 only reaches
        # full ground-truth success 60% of the time, not 100%, so one
        # dispatch of the real 5-kernel chain is nowhere near enough to
        # tell "the mitigation doesn't help the real computation" from
        # "this one draw happened to land on a partial-failure
        # iteration." Repeats the exact same real chain --logl uses
        # (same programs, same real tip/weights/freqs data, same
        # dispatch order/sync), fresh buffers every iteration matching
        # --sweep's own established discipline, tallying a real PASS/
        # FAIL/NaN rate instead of one draw's verdict. No per-iteration
        # ground-truth dump (would be far too verbose across N
        # iterations) -- just PASS/FAIL/NaN and the computed logL itself.
        ppns = make_program(dev, TinyELF, Target, dtypes, BeagleNVProgram, elf_bytes, "kernelPartialsPartialsNoScale", 1)
        il = make_program(dev, TinyELF, Target, dtypes, BeagleNVProgram, elf_bytes, "kernelIntegrateLikelihoods", 2)
        ss = make_program(dev, TinyELF, Target, dtypes, BeagleNVProgram, elf_bytes, "kernelSumSites1", 1)
        log(f"logl-sweep programs: PPNS regs={ppns.regs_usage} IL regs={il.regs_usage} SS regs={ss.regs_usage}")

        def buf(a_, n):
            return HCQBuffer(a_.va_addr, n * 4)

        tip_h = make_tip_partials(K_HUMAN, CATEGORY_COUNT)
        tip_c = make_tip_partials(K_CHIMP, CATEGORY_COUNT)
        tip_g = make_tip_partials(K_GORILLA, CATEGORY_COUNT)
        assert len(tip_h) == len(tip_c) == len(tip_g) == K_PARTIALS_SIZE

        def alloc_filled(vals, fmt):
            b = dev.allocator.alloc(len(vals) * struct.calcsize(fmt))
            dev.allocator._copyin(HCQBuffer(b.va_addr, len(vals) * struct.calcsize(fmt)), memoryview(struct.pack(f"<{len(vals)}{fmt}", *vals)))
            return b

        def zero_existing(buf_, n_floats):
            # TODO.md Phase 132, user: "1" (allocate once, reuse every
            # iteration) -- re-fills an ALREADY-allocated buffer with
            # zeros, matching alloc_zeroed()'s own fill pattern exactly,
            # just without allocating a new one.
            dev.allocator._copyin(HCQBuffer(buf_.va_addr, n_floats * 4), memoryview(bytearray(n_floats * 4)))

        mstride = align_mem_offset(K_MATRIX_SIZE * CATEGORY_COUNT * 4)

        def matrix_of(dmat_buf, edge_index):
            return sub(dmat_buf, HCQBuffer, edge_index * mstride, K_MATRIX_SIZE * CATEGORY_COUNT * 4)

        # TODO.md Phase 132, user: "1" -- all 12 buffers allocated ONCE
        # here, outside the loop, instead of fresh every iteration
        # (Phase 130/131's own free()-every-iteration experiment changed
        # the degradation's trajectory but didn't fix it -- this tests
        # whether avoiding VRAM churn entirely, not just freeing
        # promptly, does). tip_h_buf/tip_c_buf/tip_g_buf/weights_buf/
        # freqs_buf/patw_buf/listc_it are pure kernel INPUTS -- confirmed
        # (by reading every dispatch call below) that no kernel in this
        # chain ever writes to them -- so their content is genuinely
        # constant across iterations and is filled once here, never
        # re-filled in the loop; not re-filling them is also a *more*
        # sensitive test for any silent cross-iteration corruption than
        # re-copying the same bytes in every time would be.
        # node3_buf/root4_buf/result_buf/sum_buf ARE kernel OUTPUTS
        # (written by PPNS/IL/SS each dispatch) -- re-zeroed at the start
        # of every iteration below, preserving the original always-
        # starts-from-zero semantics for exactly those four. dmat_it is
        # *also* a real kernel output (kernelMatrixMulADB writes the
        # actual transition matrices into it every dispatch, which PPNS
        # then reads) -- deliberately left un-reset between iterations
        # like the pure inputs, not re-zeroed like the other four
        # outputs, since kernelMatrixMulADB unconditionally overwrites
        # every element PPNS ever reads from it before PPNS's own
        # dispatch runs (same real dispatch-queue ordering guarantee
        # this whole chain already relies on) -- its pre-existing
        # content genuinely doesn't matter for correctness, and leaving
        # it alone is the more representative test of realistic buffer
        # reuse (a real, long-running BEAGLE instance would never re-
        # zero this buffer between likelihood evaluations either).
        tip_h_buf = alloc_filled(tip_h, "f")
        tip_c_buf = alloc_filled(tip_c, "f")
        tip_g_buf = alloc_filled(tip_g, "f")
        node3_buf = alloc_zeroed(dev, HCQBuffer, K_PARTIALS_SIZE)
        root4_buf = alloc_zeroed(dev, HCQBuffer, K_PARTIALS_SIZE)
        weights_buf = alloc_filled(CATEGORY_WEIGHTS, "f")
        freqs_buf = alloc_filled(STATE_FREQS, "f")
        patw_buf = alloc_filled([1.0] * N_PATTERNS, "f")
        result_buf = alloc_zeroed(dev, HCQBuffer, N_PATTERNS)
        sum_buf = alloc_zeroed(dev, HCQBuffer, K_SUM_SITES_BLOCK_COUNT)
        dmat_it = dev.allocator.alloc(n_dmat_floats * 4)
        dev.allocator._copyin(HCQBuffer(dmat_it.va_addr, n_dmat_floats * 4), memoryview(struct.pack(f"<{n_dmat_floats}f", *dmat_init)))
        listc_it = dev.allocator.alloc(TOTAL_MATRIX * 4)
        dev.allocator._copyin(HCQBuffer(listc_it.va_addr, TOTAL_MATRIX * 4), memoryview(struct.pack(f"<{TOTAL_MATRIX}I", *listc_vals)))

        n_pass = n_fail_finite = n_nan = 0
        print(f"\n=== nv_real_kernel_probe --logl-sweep {logl_sweep}: macros={macros} maxrregcount={maxrregcount} ===", file=sys.stdout, flush=True)
        for it in range(logl_sweep):
            zero_existing(node3_buf, K_PARTIALS_SIZE)
            zero_existing(root4_buf, K_PARTIALS_SIZE)
            zero_existing(result_buf, N_PATTERNS)
            zero_existing(sum_buf, K_SUM_SITES_BLOCK_COUNT)

            prg(buf(dmat_it, n_dmat_floats), buf(listc_it, TOTAL_MATRIX), buf(a, len(EVEC)), buf(d, len(EVAL)),
                buf(b, len(IVEC)), buf(distq, TOTAL_MATRIX),
                global_size=GRID, local_size=dispatch_local_size, vals=(STATE_COUNT, STATE_COUNT, TOTAL_MATRIX), wait=False)
            ppns(buf(tip_h_buf, K_PARTIALS_SIZE), buf(tip_c_buf, K_PARTIALS_SIZE), buf(node3_buf, K_PARTIALS_SIZE),
                 matrix_of(dmat_it, 0), matrix_of(dmat_it, 1),
                 global_size=PPNS_GRID, local_size=PPNS_BLOCK, vals=(PPNS_END_PATTERN,), wait=False)
            ppns(buf(tip_g_buf, K_PARTIALS_SIZE), buf(node3_buf, K_PARTIALS_SIZE), buf(root4_buf, K_PARTIALS_SIZE),
                 matrix_of(dmat_it, 2), matrix_of(dmat_it, 3),
                 global_size=PPNS_GRID, local_size=PPNS_BLOCK, vals=(PPNS_END_PATTERN,), wait=False)
            il(buf(result_buf, N_PATTERNS), buf(root4_buf, K_PARTIALS_SIZE), buf(weights_buf, CATEGORY_COUNT), buf(freqs_buf, STATE_COUNT),
               global_size=IL_GRID, local_size=IL_BLOCK, vals=(CATEGORY_COUNT, N_PATTERNS), wait=False)
            ss(buf(result_buf, N_PATTERNS), buf(sum_buf, K_SUM_SITES_BLOCK_COUNT), buf(patw_buf, N_PATTERNS),
               global_size=SS_GRID, local_size=SS_BLOCK, vals=(N_PATTERNS,), wait=False)
            dev.synchronize()

            sum_out = memoryview(bytearray(K_SUM_SITES_BLOCK_COUNT * 4))
            dev.allocator._copyout(sum_out, HCQBuffer(sum_buf.va_addr, K_SUM_SITES_BLOCK_COUNT * 4))
            block_sums = struct.unpack(f"<{K_SUM_SITES_BLOCK_COUNT}f", bytes(sum_out))
            computed_logl = sum(block_sums)
            delta = abs(computed_logl - K_REF)

            if computed_logl != computed_logl:  # NaN self-inequality, no math.isnan import needed
                status = "NAN"
                n_nan += 1
            elif delta < 0.5:
                status = "PASS"
                n_pass += 1
            else:
                status = "FAIL"
                n_fail_finite += 1
            line = (f"  iter {it:2d}: logL={computed_logl:12.5f}  delta={delta:10.5f}  {status}  "
                    f"block_sums={[f'{v:g}' for v in block_sums]}")
            log(line)
            print(line, file=sys.stdout, flush=True)

        print(f"\n=== nv_real_kernel_probe --logl-sweep {logl_sweep}: summary ===", file=sys.stdout, flush=True)
        summary = (f"  {n_pass}/{logl_sweep} PASS ({100*n_pass/logl_sweep:.1f}%)  "
                   f"{n_fail_finite}/{logl_sweep} FAIL-finite ({100*n_fail_finite/logl_sweep:.1f}%)  "
                   f"{n_nan}/{logl_sweep} NAN ({100*n_nan/logl_sweep:.1f}%)")
        log(summary)
        print(summary, file=sys.stdout, flush=True)
        log("exiting cleanly")
        return

    if chain_sweep:
        # TODO.md Phase 134: same program/buffer setup as --logl-sweep,
        # Phase 132's own allocate-once-reuse pattern (already proven not
        # to be the confound) -- see --chain-sweep's own flag-parsing
        # comment for full rationale. Dispatches only the first
        # `chain_sweep` of the 5 real calls, then checks the relevant
        # stage's own output for iteration-to-iteration drift (a final
        # logL is only meaningful once all 5 stages have run).
        ppns = make_program(dev, TinyELF, Target, dtypes, BeagleNVProgram, elf_bytes, "kernelPartialsPartialsNoScale", 1)
        il = make_program(dev, TinyELF, Target, dtypes, BeagleNVProgram, elf_bytes, "kernelIntegrateLikelihoods", 2)
        ss = make_program(dev, TinyELF, Target, dtypes, BeagleNVProgram, elf_bytes, "kernelSumSites1", 1)
        log(f"chain-sweep programs: PPNS regs={ppns.regs_usage} IL regs={il.regs_usage} SS regs={ss.regs_usage}")

        def buf(a_, n):
            return HCQBuffer(a_.va_addr, n * 4)

        tip_h = make_tip_partials(K_HUMAN, CATEGORY_COUNT)
        tip_c = make_tip_partials(K_CHIMP, CATEGORY_COUNT)
        tip_g = make_tip_partials(K_GORILLA, CATEGORY_COUNT)
        assert len(tip_h) == len(tip_c) == len(tip_g) == K_PARTIALS_SIZE

        def alloc_filled(vals, fmt):
            b = dev.allocator.alloc(len(vals) * struct.calcsize(fmt))
            dev.allocator._copyin(HCQBuffer(b.va_addr, len(vals) * struct.calcsize(fmt)), memoryview(struct.pack(f"<{len(vals)}{fmt}", *vals)))
            return b

        def zero_existing(buf_, n_floats):
            dev.allocator._copyin(HCQBuffer(buf_.va_addr, n_floats * 4), memoryview(bytearray(n_floats * 4)))

        mstride = align_mem_offset(K_MATRIX_SIZE * CATEGORY_COUNT * 4)

        def matrix_of(dmat_buf, edge_index):
            return sub(dmat_buf, HCQBuffer, edge_index * mstride, K_MATRIX_SIZE * CATEGORY_COUNT * 4)

        # Same allocate-once set as --logl-sweep (Phase 132) -- buffer
        # management is already ruled out (Phase 133), reused here
        # unchanged so this experiment varies only the chain length.
        tip_h_buf = alloc_filled(tip_h, "f")
        tip_c_buf = alloc_filled(tip_c, "f")
        tip_g_buf = alloc_filled(tip_g, "f")
        node3_buf = alloc_zeroed(dev, HCQBuffer, K_PARTIALS_SIZE)
        root4_buf = alloc_zeroed(dev, HCQBuffer, K_PARTIALS_SIZE)
        weights_buf = alloc_filled(CATEGORY_WEIGHTS, "f")
        freqs_buf = alloc_filled(STATE_FREQS, "f")
        patw_buf = alloc_filled([1.0] * N_PATTERNS, "f")
        result_buf = alloc_zeroed(dev, HCQBuffer, N_PATTERNS)
        sum_buf = alloc_zeroed(dev, HCQBuffer, K_SUM_SITES_BLOCK_COUNT)
        dmat_it = dev.allocator.alloc(n_dmat_floats * 4)
        dev.allocator._copyin(HCQBuffer(dmat_it.va_addr, n_dmat_floats * 4), memoryview(struct.pack(f"<{n_dmat_floats}f", *dmat_init)))
        listc_it = dev.allocator.alloc(TOTAL_MATRIX * 4)
        dev.allocator._copyin(HCQBuffer(listc_it.va_addr, TOTAL_MATRIX * 4), memoryview(struct.pack(f"<{TOTAL_MATRIX}I", *listc_vals)))

        stage_names = ["kernelMatrixMulADB", "PPNS(tip_h,tip_c->node3)", "PPNS(tip_g,node3->root4)",
                       "IL(root4->result)", "SS(result->sum)"]
        print(f"\n=== nv_real_kernel_probe --chain-sweep {chain_sweep}: dispatching stages {stage_names[:chain_sweep]} ===",
              file=sys.stdout, flush=True)

        iter0_ref = None
        n_match = n_drift = n_nan = 0
        for it in range(20):
            zero_existing(node3_buf, K_PARTIALS_SIZE)
            zero_existing(root4_buf, K_PARTIALS_SIZE)
            zero_existing(result_buf, N_PATTERNS)
            zero_existing(sum_buf, K_SUM_SITES_BLOCK_COUNT)

            prg(buf(dmat_it, n_dmat_floats), buf(listc_it, TOTAL_MATRIX), buf(a, len(EVEC)), buf(d, len(EVAL)),
                buf(b, len(IVEC)), buf(distq, TOTAL_MATRIX),
                global_size=GRID, local_size=dispatch_local_size, vals=(STATE_COUNT, STATE_COUNT, TOTAL_MATRIX), wait=False)
            if sync_each:
                dev.synchronize()
            if chain_sweep >= 2:
                ppns(buf(tip_h_buf, K_PARTIALS_SIZE), buf(tip_c_buf, K_PARTIALS_SIZE), buf(node3_buf, K_PARTIALS_SIZE),
                     matrix_of(dmat_it, 0), matrix_of(dmat_it, 1),
                     global_size=PPNS_GRID, local_size=PPNS_BLOCK, vals=(PPNS_END_PATTERN,), wait=False)
                if sync_each:
                    dev.synchronize()
            if chain_sweep >= 3:
                ppns(buf(tip_g_buf, K_PARTIALS_SIZE), buf(node3_buf, K_PARTIALS_SIZE), buf(root4_buf, K_PARTIALS_SIZE),
                     matrix_of(dmat_it, 2), matrix_of(dmat_it, 3),
                     global_size=PPNS_GRID, local_size=PPNS_BLOCK, vals=(PPNS_END_PATTERN,), wait=False)
                if sync_each:
                    dev.synchronize()
            if chain_sweep >= 4:
                il(buf(result_buf, N_PATTERNS), buf(root4_buf, K_PARTIALS_SIZE), buf(weights_buf, CATEGORY_COUNT), buf(freqs_buf, STATE_COUNT),
                   global_size=IL_GRID, local_size=IL_BLOCK, vals=(CATEGORY_COUNT, N_PATTERNS), wait=False)
                if sync_each:
                    dev.synchronize()
            if chain_sweep >= 5:
                ss(buf(result_buf, N_PATTERNS), buf(sum_buf, K_SUM_SITES_BLOCK_COUNT), buf(patw_buf, N_PATTERNS),
                   global_size=SS_GRID, local_size=SS_BLOCK, vals=(N_PATTERNS,), wait=False)
            dev.synchronize()  # always -- redundant but harmless if sync_each already synced after the last dispatched stage

            # Read back whichever buffer the Nth (last dispatched) stage
            # wrote. N=1 checks dmat_it's real-matrix region only
            # (redundant with the already-proven-clean --sweep, kept for
            # self-consistency of this new tool).
            if chain_sweep == 1:
                out_buf, out_n = dmat_it, TOTAL_MATRIX * S2
            elif chain_sweep == 2:
                out_buf, out_n = node3_buf, K_PARTIALS_SIZE
            elif chain_sweep == 3:
                out_buf, out_n = root4_buf, K_PARTIALS_SIZE
            elif chain_sweep == 4:
                out_buf, out_n = result_buf, N_PATTERNS
            else:
                out_buf, out_n = sum_buf, K_SUM_SITES_BLOCK_COUNT

            raw = memoryview(bytearray(out_n * 4))
            dev.allocator._copyout(raw, HCQBuffer(out_buf.va_addr, out_n * 4))
            vals = struct.unpack(f"<{out_n}f", bytes(raw))

            has_nan = any(v != v for v in vals)
            if it == 0:
                iter0_ref = vals
                status = "REF"
                max_diff = 0.0
            elif has_nan:
                status = "NAN"
                n_nan += 1
                max_diff = float("nan")
            else:
                max_diff = max(abs(v - r) for v, r in zip(vals, iter0_ref))
                if max_diff < 0.5:
                    status = "MATCH"
                    n_match += 1
                else:
                    status = "DRIFT"
                    n_drift += 1
            line = f"  iter {it:2d}: stage={stage_names[chain_sweep-1]}  max_abs_diff_vs_iter0={max_diff:10.5f}  {status}"
            log(line)
            print(line, file=sys.stdout, flush=True)

            # TODO.md Phase 137, user: "let's do some per-element
            # diagnostics" -- Phase 135's own max_abs_diff alone can't
            # distinguish a small, fixed set of stuck/aliased slots from
            # a rotating or growing set. On every DRIFT iteration, dump
            # every index whose value differs from iter0_ref by more
            # than the same 0.5 tolerance -- (index, iter0-correct
            # value, this-iteration's value, diff) -- capped at the
            # first 20 for readability, with the real total count always
            # reported even when capped.
            if status == "DRIFT":
                drift_idx = [i for i, (v, r) in enumerate(zip(vals, iter0_ref)) if abs(v - r) > 0.5]
                shown = drift_idx[:20]
                detail = ", ".join(f"[{i}] ref={iter0_ref[i]:.6f} now={vals[i]:.6f} diff={vals[i]-iter0_ref[i]:+.6f}" for i in shown)
                more = f" ... ({len(drift_idx) - 20} more)" if len(drift_idx) > 20 else ""
                detail_line = f"    drift detail ({len(drift_idx)} of {len(vals)} indices): {detail}{more}"
                log(detail_line)
                print(detail_line, file=sys.stdout, flush=True)

        print(f"\n=== nv_real_kernel_probe --chain-sweep {chain_sweep}: summary ===", file=sys.stdout, flush=True)
        summary = f"  {n_match}/19 MATCH  {n_drift}/19 DRIFT  {n_nan}/19 NAN  (iteration 0 is the reference, not counted)"
        log(summary)
        print(summary, file=sys.stdout, flush=True)
        log("exiting cleanly")
        return

    if logl:
        # ---- Phase 71: real 3-taxon log-likelihood -- kernelMatrixMulADB
        # (dispatched first, exactly as always) produces the real
        # transition matrices this chain consumes; two real
        # kernelPartialsPartialsNoScale calls, correctly chained, real
        # kernelIntegrateLikelihoods + kernelSumSites1, all wait=False,
        # one sync at the end (matching the real pipeline's own single-
        # queue ordering guarantee -- GPU command queues execute in
        # submission order, so kernelMatrixMulADB's writes are visible to
        # the PPNS calls queued after it with no intermediate sync
        # needed, the same guarantee the real cmd_launch_batch relies on).
        ppns = make_program(dev, TinyELF, Target, dtypes, BeagleNVProgram, elf_bytes, "kernelPartialsPartialsNoScale", 1)
        il = make_program(dev, TinyELF, Target, dtypes, BeagleNVProgram, elf_bytes, "kernelIntegrateLikelihoods", 2)
        ss = make_program(dev, TinyELF, Target, dtypes, BeagleNVProgram, elf_bytes, "kernelSumSites1", 1)
        log(f"logl programs: PPNS regs={ppns.regs_usage} IL regs={il.regs_usage} SS regs={ss.regs_usage}")

        def buf(a_, n):
            return HCQBuffer(a_.va_addr, n * 4)

        tip_h = make_tip_partials(K_HUMAN, CATEGORY_COUNT)
        tip_c = make_tip_partials(K_CHIMP, CATEGORY_COUNT)
        tip_g = make_tip_partials(K_GORILLA, CATEGORY_COUNT)
        assert len(tip_h) == len(tip_c) == len(tip_g) == K_PARTIALS_SIZE

        def alloc_filled(vals, fmt):
            b = dev.allocator.alloc(len(vals) * struct.calcsize(fmt))
            dev.allocator._copyin(HCQBuffer(b.va_addr, len(vals) * struct.calcsize(fmt)), memoryview(struct.pack(f"<{len(vals)}{fmt}", *vals)))
            return b

        tip_h_buf = alloc_filled(tip_h, "f")
        tip_c_buf = alloc_filled(tip_c, "f")
        tip_g_buf = alloc_filled(tip_g, "f")
        node3_buf = alloc_zeroed(dev, HCQBuffer, K_PARTIALS_SIZE)
        root4_buf = alloc_zeroed(dev, HCQBuffer, K_PARTIALS_SIZE)
        weights_buf = alloc_filled(CATEGORY_WEIGHTS, "f")
        freqs_buf = alloc_filled(STATE_FREQS, "f")
        patw_buf = alloc_filled([1.0] * N_PATTERNS, "f")
        result_buf = alloc_zeroed(dev, HCQBuffer, N_PATTERNS)
        sum_buf = alloc_zeroed(dev, HCQBuffer, K_SUM_SITES_BLOCK_COUNT)
        log(f"real tip/weights/freqs/pattern-weights buffers allocated -- "
            f"H={tip_h_buf.va_addr:#x} C={tip_c_buf.va_addr:#x} G={tip_g_buf.va_addr:#x}")

        mstride = align_mem_offset(K_MATRIX_SIZE * CATEGORY_COUNT * 4)

        def matrix_of(edge_index):
            return sub(dmat, HCQBuffer, edge_index * mstride, K_MATRIX_SIZE * CATEGORY_COUNT * 4)

        # kernelMatrixMulADB -- the real transition matrices this chain needs.
        prg(buf(dmat, n_dmat_floats), buf(listc, TOTAL_MATRIX), buf(a, len(EVEC)), buf(d, len(EVAL)),
            buf(b, len(IVEC)), buf(distq, TOTAL_MATRIX),
            global_size=GRID, local_size=dispatch_local_size, vals=(STATE_COUNT, STATE_COUNT, TOTAL_MATRIX), wait=False)
        # node3 = P(matrix0)*tipHuman ⊙ P(matrix1)*tipChimp -- real
        # updatePartials mapping: matrices1/2 = dMatrices[child1TransMatIndex]/
        # dMatrices[child2TransMatIndex], partials1/2 = dPartials[child1Index]/
        # dPartials[child2Index] (BeagleGPUImpl.hpp:2450/2452), for
        # ops[0] = {dest=3, child1=0(H), c1mat=0, child2=1(C), c2mat=1}.
        ppns(buf(tip_h_buf, K_PARTIALS_SIZE), buf(tip_c_buf, K_PARTIALS_SIZE), buf(node3_buf, K_PARTIALS_SIZE),
             matrix_of(0), matrix_of(1),
             global_size=PPNS_GRID, local_size=PPNS_BLOCK, vals=(PPNS_END_PATTERN,), wait=False)
        # root4 = P(matrix2)*tipGorilla ⊙ P(matrix3)*node3 -- ops[1] =
        # {dest=4, child1=2(G), c1mat=2, child2=3(node3), c2mat=3}.
        ppns(buf(tip_g_buf, K_PARTIALS_SIZE), buf(node3_buf, K_PARTIALS_SIZE), buf(root4_buf, K_PARTIALS_SIZE),
             matrix_of(2), matrix_of(3),
             global_size=PPNS_GRID, local_size=PPNS_BLOCK, vals=(PPNS_END_PATTERN,), wait=False)
        # kernelIntegrateLikelihoods: dResult, dRootPartials, dWeights, dFrequencies.
        il(buf(result_buf, N_PATTERNS), buf(root4_buf, K_PARTIALS_SIZE), buf(weights_buf, CATEGORY_COUNT), buf(freqs_buf, STATE_COUNT),
           global_size=IL_GRID, local_size=IL_BLOCK, vals=(CATEGORY_COUNT, N_PATTERNS), wait=False)
        # kernelSumSites1: dArray=dResult (real chaining), dSum, dPatternWeights.
        ss(buf(result_buf, N_PATTERNS), buf(sum_buf, K_SUM_SITES_BLOCK_COUNT), buf(patw_buf, N_PATTERNS),
           global_size=SS_GRID, local_size=SS_BLOCK, vals=(N_PATTERNS,), wait=False)

        dev.synchronize()
        log("full 5-kernel real chain + single synchronize completed without hang/fault")

        sum_out = memoryview(bytearray(K_SUM_SITES_BLOCK_COUNT * 4))
        dev.allocator._copyout(sum_out, HCQBuffer(sum_buf.va_addr, K_SUM_SITES_BLOCK_COUNT * 4))
        block_sums = struct.unpack(f"<{K_SUM_SITES_BLOCK_COUNT}f", bytes(sum_out))
        computed_logl = sum(block_sums)

        print(f"\n=== nv_real_kernel_probe --logl: macros={macros} ===", file=sys.stdout, flush=True)
        line1 = f"  per-block sums: {block_sums}"
        line2 = f"  computed logL = {computed_logl:.5f}"
        line3 = f"  reference logL = {K_REF:.5f}  (tinygpuhybridtest.cpp's own CPU reference)"
        line4 = f"  |delta| = {abs(computed_logl - K_REF):.5f}  (tolerance 0.5 nats, matching tinygpuhybridtest's own check)"
        for line in (line1, line2, line3, line4):
            log(line)
            print(line, file=sys.stdout, flush=True)
        summary = f"RESULT: {'PASS' if abs(computed_logl - K_REF) < 0.5 else 'FAIL'}"
        log(summary)
        print(summary, file=sys.stdout, flush=True)

        # If logL is wrong, read back kernelMatrixMulADB's ground-truth dump
        # and the real C[] matrix from this same dispatch.
        if abs(computed_logl - K_REF) >= 0.5:
            dmat_out = memoryview(bytearray(n_dmat_floats * 4))
            dev.allocator._copyout(dmat_out, HCQBuffer(dmat.va_addr, n_dmat_floats * 4))
            dmat_vals = struct.unpack(f"<{n_dmat_floats}f", bytes(dmat_out))
            scratch = dmat_vals[TOTAL_MATRIX * S2:]
            print("\n  (logL was wrong -- reading back kernelMatrixMulADB's own ground-truth dump from this same dispatch)", file=sys.stdout, flush=True)
            for w in range(TOTAL_MATRIX):
                slot = scratch[w * S2: w * S2 + 14]
                fails = [i for i, v in enumerate(slot) if v == SENTINEL]
                line = f"    wMatrix {w:2d}: Csub={slot[0]:g}  Ds[0..3]=({slot[9]:g},{slot[10]:g},{slot[11]:g},{slot[12]:g})  fails={fails}"
                log(line)
                print(line, file=sys.stdout, flush=True)

            # The ground-truth dump above only ever samples thread (0,0)'s
            # own view (its private csub0 re-computation, row/column 0 of
            # As/Bs/Ds) -- never the *real* C[] matrix, which is written
            # independently by all 256 threads, one per matrix entry. The
            # real C[] output (what kernelPartialsPartialsNoScale actually
            # reads as matrices1/matrices2) is dmat's own first
            # TOTAL_MATRIX*S2 floats -- read it back directly and check
            # every entry, not just the diagonal thread (0,0) samples.
            print("\n  (checking the REAL C[] matrix output -- all 16 entries per wMatrix, not just thread (0,0)'s)", file=sys.stdout, flush=True)
            real_matrices = dmat_vals[:TOTAL_MATRIX * S2]
            for w in range(TOTAL_MATRIX):
                m = real_matrices[w * S2:(w + 1) * S2]
                row_sums = [sum(m[r * STATE_COUNT:(r + 1) * STATE_COUNT]) for r in range(STATE_COUNT)]
                bad = [i for i, v in enumerate(m) if v != v]  # NaN check (self-inequality)
                line = (f"    wMatrix {w:2d}: row_sums={[f'{s:g}' for s in row_sums]}  "
                        f"nan_entries={bad}  raw={[f'{v:g}' for v in m]}")
                log(line)
                print(line, file=sys.stdout, flush=True)

        log("exiting cleanly")
        return

    if not batch:
        prg(HCQBuffer(dmat.va_addr, n_dmat_floats * 4),
            HCQBuffer(listc.va_addr, TOTAL_MATRIX * 4),
            HCQBuffer(a.va_addr, len(EVEC) * 4),
            HCQBuffer(d.va_addr, len(EVAL) * 4),
            HCQBuffer(b.va_addr, len(IVEC) * 4),
            HCQBuffer(distq.va_addr, TOTAL_MATRIX * 4),
            global_size=GRID, local_size=dispatch_local_size, vals=(STATE_COUNT, STATE_COUNT, TOTAL_MATRIX), wait=False)
        dev.synchronize()
        log("kernel launch + synchronize completed without hang/fault")
    else:
        # ---- --batch: queue kernelMatrixMulADB alongside the 4 other
        # real kernels a real tinygpuhybridtest run queues in the same
        # cmd_launch_batch, same order, same real grid/block shapes,
        # every launch wait=False, exactly one dev.synchronize() at the
        # very end -- matching nv_dispatch_daemon.py's cmd_launch_batch
        # loop precisely, not approximately.
        ppns = make_program(dev, TinyELF, Target, dtypes, BeagleNVProgram, elf_bytes, "kernelPartialsPartialsNoScale", 1)
        il = make_program(dev, TinyELF, Target, dtypes, BeagleNVProgram, elf_bytes, "kernelIntegrateLikelihoods", 2)
        ss = make_program(dev, TinyELF, Target, dtypes, BeagleNVProgram, elf_bytes, "kernelSumSites1", 1)
        log(f"filler programs: PPNS regs={ppns.regs_usage} IL regs={il.regs_usage} SS regs={ss.regs_usage}")

        def buf(a_, n):
            return HCQBuffer(a_.va_addr, n * 4)

        if not realloc:
            p1 = alloc_zeroed(dev, HCQBuffer, PARTIALS_FLOATS)
            p2 = alloc_zeroed(dev, HCQBuffer, PARTIALS_FLOATS)
            p3 = alloc_zeroed(dev, HCQBuffer, PARTIALS_FLOATS)
            m1 = alloc_zeroed(dev, HCQBuffer, MATRIX_FLOATS)
            m2 = alloc_zeroed(dev, HCQBuffer, MATRIX_FLOATS)
            p2b = alloc_zeroed(dev, HCQBuffer, PARTIALS_FLOATS)  # second PPNS launch's own partials2 (real log: 2 distinct launches)
            p3b = alloc_zeroed(dev, HCQBuffer, PARTIALS_FLOATS)
            root_partials = alloc_zeroed(dev, HCQBuffer, ROOT_PARTIALS_FLOATS)
            weights = alloc_zeroed(dev, HCQBuffer, WEIGHTS_FREQ_FLOATS)
            freqs = alloc_zeroed(dev, HCQBuffer, WEIGHTS_FREQ_FLOATS)
            result = alloc_zeroed(dev, HCQBuffer, RESULT_FLOATS)
            sum_array = alloc_zeroed(dev, HCQBuffer, SUM_ARRAY_FLOATS)
            sum_out = alloc_zeroed(dev, HCQBuffer, SUM_OUT_FLOATS)
            pattern_weights = alloc_zeroed(dev, HCQBuffer, SUM_ARRAY_FLOATS)
            p1b, p2buf, p3buf = buf(p1, PARTIALS_FLOATS), buf(p2, PARTIALS_FLOATS), buf(p3, PARTIALS_FLOATS)
            p3b_1, p2b_2, p3b_2 = buf(p3, PARTIALS_FLOATS), buf(p2b, PARTIALS_FLOATS), buf(p3b, PARTIALS_FLOATS)
            m1b, m2b = buf(m1, MATRIX_FLOATS), buf(m2, MATRIX_FLOATS)
            result_b, root_b, weights_b, freqs_b = buf(result, RESULT_FLOATS), buf(root_partials, ROOT_PARTIALS_FLOATS), buf(weights, WEIGHTS_FREQ_FLOATS), buf(freqs, WEIGHTS_FREQ_FLOATS)
            sumarr_b, sumout_b, patw_b = buf(sum_array, SUM_ARRAY_FLOATS), buf(sum_out, SUM_OUT_FLOATS), buf(pattern_weights, SUM_ARRAY_FLOATS)
            log("filler buffers allocated (ad-hoc, Phase 69)")
        else:
            # ---- Phase 70: source every filler kernel's own arguments
            # from the *same* real buffer set kernelMatrixMulADB's own
            # inputs come from -- dPartialsTmpOrigin's 6 real slots for
            # partials/root-partials (real stride: align_mem_offset(
            # K_PARTIALS_SIZE*4), matching BeagleGPUImpl.hpp exactly),
            # dMatricesOrigin's own sub-pointers for matrices1/2 (real
            # BEAGLE really does pass dMatrices[i] pointers into this
            # kernel), dWeightsOrigin/dFrequenciesOrigin/dIntegrationTmp/
            # dSumLogLikelihood/dPatternWeights for IntegrateLikelihoods/
            # SumSites1 -- and dIntegrationTmp doubles as SumSites1's
            # dArray, exactly the way real BEAGLE chains these two kernels
            # together (dResult produced by IntegrateLikelihoods really is
            # what SumSites1 reads).
            pstride = align_mem_offset(K_PARTIALS_SIZE * 4)
            mstride = align_mem_offset(K_MATRIX_SIZE * CATEGORY_COUNT * 4)
            po = real_bufs["dPartialsTmpOrigin"]
            mo = real_bufs["dMatricesOrigin"]
            p1b = sub(po, HCQBuffer, 0 * pstride, K_PARTIALS_SIZE * 4)
            p2buf = sub(po, HCQBuffer, 1 * pstride, K_PARTIALS_SIZE * 4)
            p3buf = sub(po, HCQBuffer, 2 * pstride, K_PARTIALS_SIZE * 4)
            p3b_1 = sub(po, HCQBuffer, 3 * pstride, K_PARTIALS_SIZE * 4)
            p2b_2 = sub(po, HCQBuffer, 4 * pstride, K_PARTIALS_SIZE * 4)
            p3b_2 = sub(po, HCQBuffer, 5 * pstride, K_PARTIALS_SIZE * 4)
            m1b = sub(mo, HCQBuffer, 0 * mstride, K_MATRIX_SIZE * CATEGORY_COUNT * 4)
            m2b = sub(mo, HCQBuffer, 1 * mstride, K_MATRIX_SIZE * CATEGORY_COUNT * 4)
            root_b = sub(po, HCQBuffer, 5 * pstride, K_PARTIALS_SIZE * 4)  # reuse a slot -- content irrelevant to this test
            weights_b = HCQBuffer(real_bufs["dWeightsOrigin"].va_addr, align_mem_offset(CATEGORY_COUNT * 4))
            freqs_b = HCQBuffer(real_bufs["dFrequenciesOrigin"].va_addr, align_mem_offset(STATE_COUNT * 4))
            result_b = HCQBuffer(real_bufs["dIntegrationTmp"].va_addr, (K_PADDED_PATTERN_COUNT + K_RESULT_PADDED_PATTERNS) * 4)
            sumarr_b = result_b  # real chaining: SumSites1's dArray = IntegrateLikelihoods' own dResult
            sumout_b = HCQBuffer(real_bufs["dSumLogLikelihood"].va_addr, K_SUM_SITES_BLOCK_COUNT * 4)
            patw_b = HCQBuffer(real_bufs["dPatternWeights"].va_addr, N_PATTERNS * 4)
            log("filler buffers sourced from the real allocation set (Phase 70)")

        # 1. kernelMatrixMulADB -- the real one under test.
        prg(buf(dmat, n_dmat_floats), buf(listc, TOTAL_MATRIX), buf(a, len(EVEC)), buf(d, len(EVAL)),
            buf(b, len(IVEC)), buf(distq, TOTAL_MATRIX),
            global_size=GRID, local_size=dispatch_local_size, vals=(STATE_COUNT, STATE_COUNT, TOTAL_MATRIX), wait=False)
        # 2+3. kernelPartialsPartialsNoScale x2 (real log: two separate launches).
        ppns(p1b, p2buf, p3buf, m1b, m2b,
             global_size=PPNS_GRID, local_size=PPNS_BLOCK, vals=(PPNS_END_PATTERN,), wait=False)
        ppns(p3b_1, p2b_2, p3b_2, m1b, m2b,
             global_size=PPNS_GRID, local_size=PPNS_BLOCK, vals=(PPNS_END_PATTERN,), wait=False)
        # 4. kernelIntegrateLikelihoods.
        il(result_b, root_b, weights_b, freqs_b,
           global_size=IL_GRID, local_size=IL_BLOCK, vals=(CATEGORY_COUNT, N_PATTERNS), wait=False)
        # 5. kernelSumSites1.
        ss(sumarr_b, sumout_b, patw_b,
           global_size=SS_GRID, local_size=SS_BLOCK, vals=(N_PATTERNS,), wait=False)

        dev.synchronize()
        log("batch of 5 launches + single synchronize completed without hang/fault")

    dmat_out = memoryview(bytearray(n_dmat_floats * 4))
    dev.allocator._copyout(dmat_out, HCQBuffer(dmat.va_addr, n_dmat_floats * 4))
    dmat_vals = struct.unpack(f"<{n_dmat_floats}f", bytes(dmat_out))

    print(f"\n=== nv_real_kernel_probe: batch={batch} realloc={realloc} macros={macros} ===", file=sys.stdout, flush=True)
    scratch = dmat_vals[TOTAL_MATRIX * S2:]
    any_fail = False
    for w in range(TOTAL_MATRIX):
        slot = scratch[w * S2: w * S2 + 14]
        csub, as0, as1, as2, as3, bs0, bs1, bs2, bs3, ds0, ds1, ds2, ds3, smid = slot
        fails = [i for i, v in enumerate(slot) if v == SENTINEL]
        if fails:
            any_fail = True
        line = (f"  wMatrix {w:2d}: Csub={csub:g}  As[0][0..3]=({as0:g},{as1:g},{as2:g},{as3:g})  "
                f"Bs[0..3][0]=({bs0:g},{bs1:g},{bs2:g},{bs3:g})  Ds[0..3]=({ds0:g},{ds1:g},{ds2:g},{ds3:g})  "
                f"SMID={smid:g}  fails(local dbg[] idx)={fails}")
        log(line)
        print(line, file=sys.stdout, flush=True)

    summary = f"RESULT: {'FAIL -- sentinel(s) remain' if any_fail else 'PASS -- no sentinels remain'}"
    log(summary)
    print(summary, file=sys.stdout, flush=True)

    log("exiting cleanly")


def _try_extra_fault_diagnostics():
    """TODO.md Phase 97: user asked to explore desc[UR] as a possible
    explanation for Phase 94/95's real GPU fault. Re-reading tinygrad's own
    ops_nv.py found the fault is a genuine NV_VGPU_MSG_EVENT_MMU_FAULT_
    QUEUED GSP event -- a real MMU/page-table-level fault, unrelated to
    desc[UR]'s SASS-level addressing mode (already established, STATUS.md
    Phase-5/6-era investigation: that's a fixed, always-zero, kernel-
    launch-level constant on real hardware too, confirmed via real H100
    cuda-gdb data -- not something that varies with buffer size, so not a
    plausible explanation for a buffer-size-dependent fault). tinygrad's
    own on_device_hang() (ops_nv.py) already knows how to read the *real*
    fault address/type/access-type via a real NV83DE_CTRL_CMD_DEBUG_READ_
    MMU_FAULT_INFO RM control call -- but Phase 94/95's actual fault
    produced an *empty* report (the exception's own message was blank),
    meaning either sm_errors.mmuFault.valid came back false or the SM
    error array was all-zero when that diagnostic ran. This best-effort,
    read-only, purely-after-the-fact helper re-issues the *same* RM
    control calls directly and prints their raw contents regardless of
    the `valid`/nonzero gating on_device_hang() applies -- more transparent
    on the next fault, whatever it turns out to be. Deliberately wrapped
    in its own try/except: a failure here must never mask or replace the
    real exception already being handled, and this never runs unless a
    fault has already happened -- adds no risk to any successful run."""
    dev = _dev_for_diagnostics
    if dev is None or not hasattr(dev, "iface") or not hasattr(dev, "debugger"):
        return
    try:
        # ops_nv.py's own `nv_gpu` is a *dynamically* reassigned module-level
        # global (nv_570/580/610, chosen at boot by detected driver version,
        # ops_nv.py:386-395) -- the generic `autogen.nv` module doesn't
        # define these RM control structs at all (checked directly: zero
        # hits vs. one hit in nv_570.py). Import the live, version-correct
        # reference straight from ops_nv itself, not a hardcoded guess.
        from tinygrad.runtime.ops_nv import nv_gpu
        sm_errors = dev.iface.rm_control(dev.debugger, nv_gpu.NV83DE_CTRL_CMD_DEBUG_READ_ALL_SM_ERROR_STATES,
            nv_gpu.NV83DE_CTRL_DEBUG_READ_ALL_SM_ERROR_STATES_PARAMS(hTargetChannel=dev.debug_channel, numSMsToRead=100))
        log(f"[fault diagnostics] sm_errors.mmuFault.valid={sm_errors.mmuFault.valid}")
        if sm_errors.mmuFault.valid:
            mmu = dev.iface.rm_control(dev.debugger, nv_gpu.NV83DE_CTRL_CMD_DEBUG_READ_MMU_FAULT_INFO,
                nv_gpu.NV83DE_CTRL_DEBUG_READ_MMU_FAULT_INFO_PARAMS())
            log(f"[fault diagnostics] mmu.count={mmu.count}")
            for i in range(mmu.count):
                pf = mmu.mmuFaultInfoList[i]
                log(f"[fault diagnostics]   MMU fault[{i}]: address=0x{pf.faultAddress:X} faultType={pf.faultType} accessType={pf.accessType}")
        nonzero_sm = [(i, e.hwwGlobalEsr, hex(e.hwwWarpEsr), hex(e.hwwWarpEsrPc64))
                      for i, e in enumerate(sm_errors.smErrorStateArray) if e.hwwGlobalEsr or e.hwwWarpEsr]
        log(f"[fault diagnostics] SMs with nonzero ESR state: {nonzero_sm if nonzero_sm else '(none)'}")
    except Exception as diag_exc:
        log(f"[fault diagnostics] the diagnostic read itself failed (not the original error): {diag_exc!r}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc(file=sys.stderr)
        _try_extra_fault_diagnostics()
        print("RESULT: FAIL (exception, see log)", file=sys.stdout, flush=True)
        sys.exit(1)
