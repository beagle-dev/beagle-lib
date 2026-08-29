#!/usr/bin/env python3
"""
nv_real_kernel_probe.py -- TODO.md "PICK UP HERE" -> NV Phase 68.

Every synthetic reconstruction attempted since Phase 47 (broadcast probe,
footprint sweep, combo probe, SFU probe, dist+exp probe -- Phases 49-67,
nv_broadcast_probe.py through nv_distexp_probe.py) has failed to reproduce
the real kernel's `Ds[2]` residual, even variants matching its resource
footprint, branch shape, broadcast+barrier+burst structure, SFU
instructions, and an all-256-threads data-dependent global load, alone and
combined. This replaces "reconstruct pieces of the real kernel
synthetically" with "compile and dispatch the actual real kernel source,
just faster": `kernels4.cu` (the real, unmodified production file,
`#include`-ing `kernelsAll.cu`) compiled directly via the real `nvcc`
shim, dispatched via the same `BeagleNVProgram` class the daemon uses,
with real input buffers matching `tinygpuhybridtest --diag-matmul-ground-
truth`'s exact setup -- bypassing BEAGLE's C++/daemon RPC pipeline
entirely. No C++ rebuild, no `sudo cmake --install`, no daemon spawn per
experiment: a new bisection is just a different macro list (and, for
source-level experiments, an edit to one of kernelsAll.cu's existing
opt-in `FW_TINYGPU`-gated blocks, exactly as before) and a re-run of this
one script.

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

Default macros (`TINYGPU_DEBUG_DUMP_MATMUL_GROUND_TRUTH` +
`TINYGPU_BISECT_NO_EXP`) reproduce the exact known baseline: wMatrix 0-2
all-zero, wMatrix>=3's Ds[2] stuck at -999, everything else correct.
Running this once with no arguments and confirming that exact pattern is
the acceptance test for the probe itself before trusting any new
macro combination's result.

Ran (Phase 68, solo mode): the real kernel, dispatched alone, came back
completely clean -- no reproduction. Traced the daemon's real dispatch
path (`nv_dispatch_daemon.py`'s `cmd_launch_batch`) and found the one
axis no probe has ever varied: the real pipeline queues *five* real
kernels (`kernelMatrixMulADB`, two `kernelPartialsPartialsNoScale`,
`kernelIntegrateLikelihoods`, `kernelSumSites1`) back-to-back with
`wait=False` on the same hardware queue, synchronizing only once at the
very end -- every probe including this one's solo mode launches exactly
one kernel and syncs immediately after. `--batch` (Phase 69) reproduces
that exact queuing pattern directly: real signatures, real grid/block
shapes (read off a real hardware launch log, not guessed), correctly-
sized-but-otherwise-arbitrary buffers for the four filler kernels (their
own numerical correctness is irrelevant to this test -- only
kernelMatrixMulADB's ground-truth dump is read back).

Ran `--batch` (Phase 69): still completely clean -- rules out multi-kernel
batching too. Compared (user's own direct instruction) exactly how real
BEAGLE allocates and accesses transition matrices/eigendecompositions vs.
this probe (`BeagleGPUImpl.hpp`'s constructor, line by line): `CreateSub-
Pointer(base,off,_)` is pure `base+off` arithmetic on the NV/TinyGPU
backend (`GPUInterfaceTinyGPUHybrid.cpp`) -- no separate allocation, no
distinct descriptor, ruled out as a mechanism. `dMatricesOrigin`'s real
size/scratch-region math turns out to match this probe's own `dmat`
buffer exactly, byte for byte -- not a bug. What's real and never
replicated: `beagleCreateInstance` allocates a *specific, ordered set* of
~15 separate buffers (eigenvectors, eigenvalues, weights, frequencies,
integration scratch, pattern weights, the *big* combined tip+internal+
root partials origin -- ~295KB, by far the largest single real allocation
this pipeline makes --, branch lengths, distance queue, pointer queue,
derivative queue) *before* `kernelMatrixMulADB` ever launches, each sized
via `AlignMemOffset` (256-byte-rounded stride). `--realloc` (Phase 70)
replicates this exact set, in this exact order, with real (`AlignMemOffset`-
matching) sizes -- and threads the filler kernels' own arguments through
these real buffers the same way real BEAGLE actually chains them (e.g.
`kernelIntegrateLikelihoods`' `dResult` really does feed `kernelSumSites1`'s
`dArray` in the real pipeline; this probe now does the same), rather than
disconnected ad-hoc buffers.

Ran `--realloc` alone and `--batch --realloc` (Phase 70): both clean --
memory allocation history, alone or combined with batching, is also ruled
out. Every mechanically-identifiable dispatch-context/allocation
difference this investigation could name has now been tested and found
insufficient.

Phase 71: user asked to double-check compilation/execution/argument
parity against the real, installed library before trusting any of the
above -- found and fixed one real, previously-unflagged discrepancy
(`BASE_MACROS` was missing `TINYGPU_BISECT_NO_LISTC`, which the currently
-installed library carries from Phase 65's build); with it added, this
probe's live `nvcc` compile is now byte-for-byte identical to the real,
installed `KERNELS_STRING_SP_4` -- diffed directly, all 20101 PTX lines,
not just kernelMatrixMulADB's. Execution (`BeagleNVProgram`'s
construction, `_use_nvjitlink()`'s tool selection) and argument order/
roles were independently re-verified against the real source and found
identical (the real `compile_ptx_split`/per-kernel-tool-selection path is
dead code for this test -- `cmd_launch_batch` never calls it).

Also added `--logl` (Phase 71): computes an actual, real 3-taxon log-
likelihood -- kernelMatrixMulADB's real transition matrices feed two
real, correctly-chained `kernelPartialsPartialsNoScale` calls
((Human,Chimp)->node3, (Gorilla,node3)->root, exactly BEAGLE's own real
topology/argument mapping, read from `updatePartials`'s actual
`dMatrices[child1TransMatIndex]`/`dPartials[child1Index]` source, not
guessed), then real `kernelIntegrateLikelihoods` + `kernelSumSites1`,
summed on the host -- comparable directly against `tinygpuhybridtest`'s
own CPU reference (`-1498.89812`). A real correctness check, not another
dispatch-context bisection.

Ran `--logl` (real kernel) 4 times: every run FAILs (`logL=-inf`), but
each run's own ground-truth dump (thread (0,0)'s view only) comes back
completely clean -- exposing a real blind spot in every ground-truth
probe since Phase 47 (it only ever samples one of 256 threads' stores).
Reading the *real* `C[]` matrix directly (all 16 entries, not just
thread (0,0)'s) across those 4 runs found two layered phenomena: which
`wMatrix` blocks write *anything* is non-deterministic run to run, but
`wMatrix` 4-11 (the middle half of `blockIdx.x` space) have never once
produced data in any run -- only the two *ends* (0-3, 12-15) ever
succeed. `--sweep [N]` (default `N=20`, Phase 71) automates this: one
boot, one compile, then `N` fresh dispatches of just `kernelMatrixMulADB`
in a tight in-process loop, tabulating per-`wMatrix` success rate and the
distinct populated-sets seen, to test whether that "middle never runs"
pattern is truly unconditional or just unlucky in a small sample.

    python3 nv_real_kernel_probe.py [--batch] [--realloc] [--logl] [--logl-sweep [N]] [--sweep [N]] [--swap-cat01] [--wide-grid [N]] [--maxrregcount N] [--per-thread-ds | --per-thread-ds-w0 | --dummy-third-block | --per-thread-dummy-w0 | --per-thread-ds-min-w0 | --local-mem-w0 | --shared-spill-w0 | --local-mem-flat-w0 | --shared-broadcast-flat-w0 | --flat-dispatch] [EXTRA_MACRO ...]

Examples:
    python3 nv_real_kernel_probe.py                                  # solo (Phase 68 baseline -- came back clean)
    python3 nv_real_kernel_probe.py --batch                          # queued with 4 other real kernels, matching cmd_launch_batch (Phase 69 -- also clean)
    python3 nv_real_kernel_probe.py --realloc                        # solo dispatch, but preceded by BEAGLE's real ~15-buffer allocation set/order (Phase 70 -- also clean)
    python3 nv_real_kernel_probe.py --batch --realloc                # the closest real-pipeline replication this investigation has built (Phase 70 -- also clean)
    python3 nv_real_kernel_probe.py --logl                           # real 3-taxon logL, compared against the CPU reference (Phase 71)
    python3 nv_real_kernel_probe.py --sweep                          # 20 fresh kernelMatrixMulADB dispatches, per-wMatrix success-rate table (Phase 71)
    python3 nv_real_kernel_probe.py --sweep 50                       # same, 50 iterations
    python3 nv_real_kernel_probe.py --sweep --swap-cat01             # Phase 75/76: swaps which *value* wMatrix 4/8/12 vs 5/9/13 (cat=0 vs cat=1) get, wMatrix/SMID unchanged -- does the rare-extra-success pattern follow the value or stay pinned to the slot? (Phase 77: it's the slot.)
    python3 nv_real_kernel_probe.py --sweep --wide-grid              # Phase 78: real kernel, 32 blocks instead of 16 (totalMatrix=32 too, stays in the same safe bx=0 path) -- does elevated reliability follow the SMID a wide-grid block lands on, or stay with its wMatrix-mod-16 identity? (Phase 79: it's the SMID -- 0/16 identical-data pairs matched.)
    python3 nv_real_kernel_probe.py --sweep --wide-grid 64           # Phase 79's follow-up: a grid=32 launch used exactly SMID 0-31, no repeats -- 64 is 2 clean waves of 32 (if that's really the whole SM pool) rather than 48's uneven half-wave, the cleanest way to test whether a *repeated* visit to the same physical SM in a later wave reproduces that SM's known reliability
    python3 nv_real_kernel_probe.py --sweep --maxrregcount 20        # Phase 85: real, unmodified kernel, ptxas capped to 20 registers (naturally 40 uncapped) -- forces local-memory spill, tests the register-pressure hypothesis Phase 84's trivial-kernel dial couldn't reach directly
    python3 nv_real_kernel_probe.py --logl-sweep --maxrregcount 24   # Phase 88: a single --logl draw has no statistical power -- repeats the real 5-kernel chain 20x, reports a real PASS/FAIL/NaN rate instead of one draw's verdict
    python3 nv_real_kernel_probe.py --sweep --maxrregcount 24        # Phase 90: --sweep now checks real CORRECTNESS (vs. the closed-form reference transition matrix), not just "wrote nonzero" -- every prior --sweep run only checked the latter, a real blind spot Phase 89's dead end exposed
    python3 nv_real_kernel_probe.py --sweep 1 --per-thread-ds        # Phase 93: Phase 91/92 found row ty=0 (Ds[]'s writers) exactly correct, rows ty>0 (Ds[]'s readers) wrong -- this has every one of the 16 real threads write its own observed Ds[0..3], settling directly whether ty>0 threads see the same barrier-published Ds[] ty=0 wrote (Phase 94/95: faulted the real GPU, 100% reproducibly; buffer size suspected)
    python3 nv_real_kernel_probe.py --sweep 1 --per-thread-ds-w0     # Phase 96: same diagnostic, wMatrix 0 only (dmat only 2304 bytes, close to the already-proven-safe baseline, vs. --per-thread-ds's 6144) -- tests whether buffer size was the operative variable behind Phase 94/95's fault before ever retrying the full version
    python3 nv_real_kernel_probe.py --downstream-sweep               # Phase 99: kernelMatrixMulADB has faulted the real GPU three times (Phase 94-98) -- this substitutes KNOWN-CORRECT reference transition matrices directly into dMatrices and dispatches only PPNS/IL/SS (kernelMatrixMulADB is NEVER dispatched), isolating whether those three downstream kernels are themselves reliable given guaranteed-correct inputs, with zero risk from kernelMatrixMulADB's own fault-prone code path (Phase 99 result: 20/20 PASS, fully deterministic -- PPNS/IL/SS confirmed reliable, the bug is isolated to kernelMatrixMulADB itself)
    python3 nv_real_kernel_probe.py --sweep 1 --dummy-third-block    # Phase 100: keeps TINYGPU_DEBUG_DUMP_MATMUL_GROUND_TRUTH active and adds a brand-new, content-unrelated third post-barrier write (one thread/block writes a hardcoded constant, 4 bytes/block -- far smaller than Phase 96's already-small w0-only version) -- tests whether ANY third write block faults this kernel, independent of size, content, or combining with the per-thread-Ds diagnostics specifically (Phase 100 result: no fault, 16/16 correct -- rules out that broad hypothesis, points back toward the specific G+D combination or per-thread-Ds's own write granularity)
    python3 nv_real_kernel_probe.py --sweep 1 --per-thread-dummy-w0  # Phase 101: isolates granularity from content -- all 16 real threads in wMatrix 0's block (matching --per-thread-ds's exact per-thread address pattern) each write a trivial, per-thread-identifiable value (their own (ty*EDGE+tx) index, not read from Ds[]/shared memory), same total footprint as Phase 100's dummy-third-block -- tests whether the many-threads-each-writing-their-own-slot pattern itself is what breaks this kernel, independent of shared-memory content (Phase 101 result: no fault, 16/16 correct -- rules out granularity too, narrows the live explanation to reading Ds[]/shared memory specifically)
    python3 nv_real_kernel_probe.py --sweep 1 --per-thread-ds-min-w0 # Phase 102: the last single-variable swap -- same footprint/granularity as Phase 101 (16 real threads, wMatrix 0 only), but each thread now writes a real Ds[tx] shared-memory readback instead of a trivial local value -- isolates whether reading Ds[]/shared memory in this third, post-barrier context is what breaks this kernel (Phase 102 result: no fault -- but only 7/16 correct, not 16/16: the first direct, non-inferred hardware confirmation of the Ds[] broadcast-visibility hypothesis)
    python3 nv_real_kernel_probe.py --sweep 20 --per-thread-ds-min-w0 # Phase 103: follows up on Phase 102's 7/16 finding with real statistical power (not just sweep=1) and a full per-(ty,tx)-slot breakdown, not just an aggregate rate -- directly tests whether ty=0 (Ds[]'s writer row) is consistently correct while ty>0 (readers) are consistently/inconsistently wrong, the same question Phase 91/92 could only answer indirectly from downstream C[] output (Phase 104 result under CUDA 13: byte-identical to v12.8, ruling out ptxas codegen)
    python3 nv_real_kernel_probe.py --sweep 20 --local-mem-w0          # Phase 106: tests register-spilled LOCAL memory correctness, independent of Ds[]/shared memory (never tested before) -- each of the 16 real threads in wMatrix 0's block writes a per-thread-identifiable value into a genuinely-spilled 4-element local array (forced via a runtime-varying index, verified necessary), then reads it back; unlike Ds[], local memory is per-thread-private, so "correct" means each thread sees its own write, not another thread's (Phase 106 result: every ty>0 thread reads back precisely thread (0, tx+4ty)'s own value -- a genuine cross-thread local-memory aliasing pattern, arithmetic proven correct via SASS decode)
    python3 nv_real_kernel_probe.py --sweep 20 --shared-spill-w0        # Phase 108: user's direct proposal -- use EXPLICIT shared memory instead of local memory for the same spill-and-reload pattern (source-controlled array indexing, not the driver's opaque per-thread stack pointer). Tests genuinely new territory: a ty>0 thread writing to its own uniquely-indexed shared slot and reading it back itself (same-thread round-trip) vs. Ds[]'s already-tested cross-thread (written-by-ty=0, read-by-ty>0) pattern (Phase 108 result: NO -- ty=0 20/20, every ty>0 slot 0/20, same shape as both Ds[] and local memory; the defect is triggered by ty>0 itself, not the storage class)
    python3 nv_real_kernel_probe.py --sweep 20 --local-mem-flat-w0     # Phase 109: does the tx+4ty local-memory aliasing depend on the real 2D (16,16,1) block dispatch, or does it persist under a flat (256,1,1) block? Dispatches with local_size=(256,1,1); the kernel derives its own logical tx=flat%16 (fast/warp-local, matching tx's native role in As[ty][tx]), ty=flat/16 (slow) from the single flat threadIdx.x, then early-returns before the kernel's own "Last block" guard (which would otherwise do a genuine out-of-bounds As[ty][tx] write under this dispatch shape) -- confirmed via SASS that the dangerous code is fully dead-code-eliminated, not just skipped at runtime (Phase 109 result: 320/320 (100%) correct -- the biggest finding this investigation has produced; the defect is triggered by the real 2D blockDim.y>1 dispatch, not by ty>0 in the abstract)
    python3 nv_real_kernel_probe.py --sweep 20 --shared-broadcast-flat-w0 # Phase 110: does Ds[]'s own broadcast-collapse bug (ty>0 readers see Ds[0]'s value regardless of index, Phase 90-103) ALSO disappear under a flat dispatch, the same way local memory's aliasing did? Same flat (256,1,1) dispatch and safety design as --local-mem-flat-w0, but tests the write-by-subset(logicalTy==0)/read-by-all shared-memory pattern via a dedicated sDs[16] array using the real Ds[]=exp(D[tx]*distance) formula (Phase 110 result: 320/320 (100%) correct, confirming both known defects share a common root cause tied to blockDim.y>1)
    python3 nv_real_kernel_probe.py --sweep 20 --flat-dispatch          # Phase 111: the real fix, not another isolated diagnostic -- kernelMatrixMulADB now has a real rewrite gated behind FW_TINYGPU_HYBRID_NV (tx,ty derived from a single flat KW_LOCAL_ID_0, everything else in the kernel untouched). This flag compiles with that macro and dispatches with the matching flat (256,1,1) block, verified via --sweep's own existing real_matrices-vs-reference_matrices correctness check -- does the REAL kernel's real output become correct for the first time in this whole investigation?
"""
import sys, os, pathlib, struct, subprocess, time, math
from collections import Counter, defaultdict

_TINYGRAD_PATH = os.environ.get("TINYGRAD_PATH", str(pathlib.Path.home() / "Dropbox/Projects/tinygrad"))
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

# Default macros: matches the *currently-installed* real library exactly
# (Phase 71 double-check: diffed this probe's own live nvcc compile
# against the real, installed BeagleTinyGPU_kernels.h's embedded
# KERNELS_STRING_SP_4 byte-for-byte -- kernelMatrixMulADB differed by 313
# lines with TINYGPU_BISECT_NO_LISTC omitted, confirming the installed
# library still carries it from Phase 65's build; with it included, the
# *entire* compiled PTX file -- all kernels, 20101 lines -- is identical).
# TINYGPU_BISECT_NO_LISTC was already proven functionally inert for the
# Ds[2] residual (Phase 65 hardware result), so this doesn't change any
# prior finding's validity -- it just makes future comparisons exact.
# Extra macros from argv are appended.
BASE_MACROS = ["TINYGPU_DEBUG_DUMP_MATMUL_GROUND_TRUTH", "TINYGPU_BISECT_NO_EXP", "TINYGPU_BISECT_NO_LISTC"]

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


def local_mem_expected(ty, tx, total_matrix):
    """TODO.md Phase 106: replicates kernelsAll.cu's TINYGPU_DEBUG_DUMP_
    LOCAL_MEM_W0 index recurrence exactly (idx = tx%4, then 16 iterations
    of idx = (idx+totalMatrix+i)%4) to determine which of spillTest[]'s 4
    slots is read back after the loop, and what value that thread itself
    wrote there (1000+ty*100+tx*10+idx -- the value formula only depends
    on the slot index, not which iteration wrote it, so knowing the final
    idx is sufficient)."""
    idx = tx % STATE_COUNT
    for i in range(16):
        idx = (idx + total_matrix + i) % STATE_COUNT
    return 1000 + ty * 100 + tx * 10 + idx


def ab_pattern_expected(a, b):
    """TODO.md Phase 113: the value thread (a,b) itself wrote to its own
    slot in --ab-pattern-2d-w0/--ab-pattern-flat-w0's private sAs/sBs
    arrays (kernelsAll.cu's TINYGPU_DEBUG_DUMP_AB_PATTERN_2D_W0/_FLAT_W0),
    for whichever (a,b) pair the *writer* of a given read actually is --
    e.g. reading As[ty][k] means the writer is thread (ty,k), so the
    expected value is ab_pattern_expected(ty, k). Injective over a,b in
    [0,16) since b<16 never carries into a's *16 term."""
    return 1000 + a * 16 + b


def combined_val(a, b):
    """TODO.md Phase 116: the value thread (a,b) wrote to its own slot in
    --combined-flat-w0's private sAs/sBs (kernelsAll.cu's TINYGPU_DEBUG_
    DUMP_COMBINED_FLAT_W0) -- deliberately small (a,b in [0,4) only, the
    only range ever actually read) so every intermediate product/sum in
    the Csub-analog reduction stays far under float32's 2^24 exact-
    integer bound."""
    return 10 + a * 4 + b


def combined_ds(k):
    """TODO.md Phase 116: the value thread (ty=0,k) wrote to sDs[k]."""
    return 1 + k


def combined_csub_expected(ty, tx):
    """TODO.md Phase 116: the exact (no float32 rounding, by construction)
    Csub-analog value --combined-flat-w0's own reduction should produce,
    mirroring the real kernel's `Csub += As[ty][k]*Ds[k]*Bs[k][tx]`."""
    return sum(combined_val(ty, k) * combined_ds(k) * combined_val(k, tx) for k in range(STATE_COUNT))


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
    changes, tests the register-pressure hypothesis Phase 84's trivial-
    kernel dial couldn't reach directly."""
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
    swap_cat01 = False
    wide_grid = None
    maxrregcount = None
    per_thread_ds = False
    per_thread_ds_w0 = False
    downstream_sweep = None
    dummy_third_block = False
    per_thread_dummy_w0 = False
    per_thread_ds_min_w0 = False
    local_mem_w0 = False
    shared_spill_w0 = False
    local_mem_flat_w0 = False
    shared_broadcast_flat_w0 = False
    flat_dispatch = False
    ab_pattern_2d_w0 = False
    ab_pattern_flat_w0 = False
    final_abcd_w0 = False
    combined_flat_w0 = False
    combined_ldg_flat_w0 = False
    combined_exp_flat_w0 = False
    combined_write_direct_w0 = False
    combined_write_listc_w0 = False
    single_tile_dispatch = False
    argv = sys.argv[1:]
    while argv and (argv[0] in ("--batch", "--realloc", "--logl", "--logl-sweep", "--chain-sweep", "--sync-each", "--sweep", "--swap-cat01", "--wide-grid", "--maxrregcount", "--per-thread-ds", "--per-thread-ds-w0", "--downstream-sweep", "--dummy-third-block", "--per-thread-dummy-w0", "--per-thread-ds-min-w0", "--local-mem-w0", "--shared-spill-w0", "--local-mem-flat-w0", "--shared-broadcast-flat-w0", "--flat-dispatch", "--ab-pattern-2d-w0", "--ab-pattern-flat-w0", "--final-abcd-w0", "--combined-flat-w0", "--combined-ldg-flat-w0", "--combined-exp-flat-w0", "--combined-write-direct-w0", "--combined-write-listc-w0", "--single-tile-dispatch")):
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
        elif argv[0] == "--swap-cat01":
            swap_cat01 = True
        elif argv[0] == "--per-thread-ds":
            # TODO.md Phase 93: directly tests Phase 91/92's shared-
            # memory Ds[] broadcast-visibility hypothesis -- --sweep-only,
            # adds TINYGPU_DEBUG_DUMP_PER_THREAD_DS to the compile and a
            # third dmat region so every one of the 16 real threads'
            # (not just thread (0,0)'s) own view of Ds[0..3] can be
            # compared against the writer row (ty=0)'s.
            per_thread_ds = True
        elif argv[0] == "--per-thread-ds-w0":
            # TODO.md Phase 96: the full --per-thread-ds (1024 extra
            # floats, dmat grown to 6144 bytes -- 3x the largest buffer
            # any probe here had used) triggered a real, 100%-reproducible
            # GPU fault (Phase 94/95) of unknown cause (the address math
            # itself was verified correct at the SASS level). This is the
            # same diagnostic restricted to wMatrix 0 only (64 extra
            # floats, dmat only 2304 bytes -- close to the already-proven-
            # safe baseline) -- tests whether buffer *size* was the
            # operative variable before ever re-attempting the full
            # version.
            per_thread_ds_w0 = True
        elif argv[0] == "--dummy-third-block":
            # TODO.md Phase 100: Phase 94/95/98 found that adding a THIRD
            # post-barrier write block (TINYGPU_DEBUG_DUMP_PER_THREAD_DS(_W0),
            # alongside the real matrix write and TINYGPU_DEBUG_DUMP_MATMUL_
            # GROUND_TRUTH) faults the real GPU 100% reproducibly, at both
            # full size and a much smaller w0-only size -- ruling out buffer
            # size. Two hypotheses remained: (1) it's specifically *this*
            # combination of diagnostics; (2) it's fundamentally about
            # adding *any* third post-barrier write block, independent of
            # size, content, or which diagnostics are combined. User's
            # direction: test (2). Keeps TINYGPU_DEBUG_DUMP_MATMUL_GROUND_
            # TRUTH active and adds a brand-new, content-unrelated third
            # write -- one thread per block writes a hardcoded constant
            # (kernelsAll.cu's TINYGPU_DEBUG_DUMP_DUMMY_THIRD_BLOCK) to a
            # dedicated scratch region far smaller (4 bytes/block) than
            # even Phase 96's already-small w0-only version.
            dummy_third_block = True
        elif argv[0] == "--per-thread-dummy-w0":
            # TODO.md Phase 101: Phase 100's --dummy-third-block (1
            # thread/block writes a hardcoded constant) ran clean --
            # ruling out hypothesis (2) in its broadest form. That probe
            # differed from --per-thread-ds(-w0) in two ways at once:
            # content (constant vs. real Ds[] shared-memory read) and
            # granularity (1 thread/block vs. 16 real threads each
            # writing their own slot). This isolates granularity alone:
            # all 16 threads in wMatrix 0's block write a trivial,
            # per-thread-identifiable value (their own (ty*EDGE+tx)
            # index, not read from Ds[]/shared memory at all) -- same
            # total footprint as --dummy-third-block (16 floats), same
            # G-active setup, only the number of writing threads differs.
            per_thread_dummy_w0 = True
        elif argv[0] == "--per-thread-ds-min-w0":
            # TODO.md Phase 102: Phase 100 (content swap) and Phase 101
            # (granularity swap) both ran clean -- the only variable left
            # distinguishing those two safe probes from the two real
            # faults is that the real --per-thread-ds(-w0) reads actual
            # Ds[] shared-memory values as its write content. This
            # isolates that last variable directly, at Phase 101's exact
            # same footprint/granularity: all 16 real threads in wMatrix
            # 0's block, each now writing Ds[tx] (a single real shared-
            # memory read) instead of a trivial local value.
            per_thread_ds_min_w0 = True
        elif argv[0] == "--local-mem-w0":
            # TODO.md Phase 106: register-spilled LOCAL memory has never
            # been tested for correctness, independent of Ds[]/shared
            # memory -- kernelMatrixMulADB genuinely spills registers
            # (lcmem_usage=576 bytes/thread). Local memory is per-thread-
            # private (no broadcast semantics like Ds[]), so this tests a
            # different question: does each thread's own spilled-and-
            # reloaded value survive a roundtrip correctly, or does
            # something (e.g. a local-memory-window/backing-store
            # addressing bug) corrupt or cross-alias it?
            local_mem_w0 = True
        elif argv[0] == "--shared-spill-w0":
            # TODO.md Phase 108: user's direct proposal -- since local
            # memory's per-thread addressing is corrupted (Phase 106),
            # what if EXPLICIT shared memory were used instead (a normal
            # source-level array index, not an opaque driver-provided
            # per-thread stack pointer)? Reuses --local-mem-w0's exact
            # write-then-read-back pattern, only the storage swapped to a
            # __shared__ array sized for the full 256-thread block,
            # indexed by each thread's own true linear position
            # (ty*16+tx). Tests genuinely new territory: a ty>0 thread
            # writing to its own uniquely-indexed shared slot and reading
            # it back itself (a same-thread round-trip), vs. Ds[]'s
            # tested cross-thread (written-by-ty=0, read-by-ty>0) pattern.
            shared_spill_w0 = True
        elif argv[0] == "--local-mem-flat-w0":
            # TODO.md Phase 109: user's direct question -- does the
            # tx+4ty local-memory aliasing (Phase 106) depend on the
            # kernel's real 2D (16,16,1) block dispatch, or does it
            # persist even when every thread's row/column identity is
            # linearized onto a single flat dimension? Dispatches
            # kernelMatrixMulADB with local_size=(256,1,1) instead of the
            # normal (16,16,1) -- the kernel computes its own logical
            # tx=flat/16 (slow), ty=flat%16 (fast, warp-contiguous) from
            # the single flat threadIdx.x, then runs the exact same
            # write-then-read-back local-memory test as --local-mem-w0.
            # Must return before the kernel's own "Last block" guard
            # (which writes As[ty][tx]/Bs[ty][tx] unconditionally, out of
            # bounds for tx>=16 under this dispatch) -- kernelsAll.cu
            # handles this via an early return, following the same
            # established-safe pattern TINYGPU_DEBUG_BROADCAST_PROBE uses.
            local_mem_flat_w0 = True
        elif argv[0] == "--shared-broadcast-flat-w0":
            # TODO.md Phase 110: user's direct follow-up to Phase 109's
            # 320/320 result -- does Ds[]'s own broadcast-collapse bug
            # (Phase 90-103: ty>0 readers see Ds[0]'s value regardless of
            # requested index) ALSO disappear under a flat (256,1,1)
            # dispatch, the same way local memory's tx+4ty aliasing did?
            # Mirrors --local-mem-flat-w0's exact dispatch/safety design,
            # but tests the write-by-subset/read-by-all shared-memory
            # broadcast pattern (a dedicated sDs[16] array, written only
            # by logicalTy==0 threads with the real Ds[]=exp(D[tx]*
            # distance) formula) instead of local memory.
            shared_broadcast_flat_w0 = True
        elif argv[0] == "--flat-dispatch":
            # TODO.md Phase 111: the real fix, not another isolated
            # diagnostic -- Phase 109/110 proved BOTH known defects
            # (local-memory tx+4ty aliasing, Ds[] broadcast collapse)
            # vanish completely under a flat block. kernelsAll.cu's
            # kernelMatrixMulADB now has a real (non-early-return)
            # rewrite gated behind FW_TINYGPU_HYBRID_NV: tx,ty derived
            # from a single flat KW_LOCAL_ID_0 instead of the native
            # KW_LOCAL_ID_0/KW_LOCAL_ID_1 pair, with every other line of
            # the kernel (As/Bs/Ds loading, the Csub reduction, the
            # boundary EDGE guard, the final write) completely untouched.
            # This flag adds that macro and dispatches with the matching
            # flat (256,1,1) block -- no diagnostic-region machinery
            # needed at all, since --sweep's own existing correctness
            # check (real_matrices vs. reference_matrices) already tells
            # us directly whether the REAL kernel output is now correct.
            flat_dispatch = True
        elif argv[0] == "--ab-pattern-2d-w0":
            # TODO.md Phase 113, user: "probe As[ty][k] etc." -- direct,
            # per-thread ground truth of the real kernel's own As[]/Bs[]
            # access pattern (own-slot write by EVERY thread including
            # ty>0, then cross-thread row/column read during the Csub
            # reduction), never directly isolated before -- Phase 112
            # inferred a likely defect here from downstream Csub output
            # alone. Uses the kernel's real REAL[16][16] shared-array shape
            # (private sAs/sBs copies, not the real As/Bs) under a flat
            # (256,1,1) dispatch, same early-return safety pattern as
            # --local-mem-flat-w0/--shared-broadcast-flat-w0.
            ab_pattern_2d_w0 = True
        elif argv[0] == "--ab-pattern-flat-w0":
            # TODO.md Phase 113, user: "also try linearizing the shared
            # memory into As[tid.x] and As[tx*16+k] etc." -- same ground
            # truth as --ab-pattern-2d-w0, but the shared array is a flat
            # REAL[256] with hand-computed addresses instead of a 2D
            # REAL[16][16] -- tests whether 2D-array codegen itself matters,
            # and additionally captures a "swapped" read (tx*16+k instead
            # of ty*16+k, the user's own suggested alternate indexing) in
            # the same pass.
            ab_pattern_flat_w0 = True
        elif argv[0] == "--final-abcd-w0":
            # TODO.md Phase 115, user: "please instrument the As / Bs / Ds
            # / Csub right before their final write" -- Phase 113/114:
            # every individual mechanism (Ds[] broadcast, local memory,
            # As[]/Bs[]'s own cross-thread pattern) is now proven correct
            # under flat dispatch in isolation, yet Phase 111's real,
            # combined kernel rewrite still failed identically to the
            # original bug. Unlike every probe since Phase 90, this one
            # uses no private storage and no early return -- it captures
            # the REAL As/Bs/Ds/Csub, for the REAL "Last block" case,
            # right after the real Csub reduction finishes and right
            # before the real barrier/C[] write, then lets the kernel
            # proceed completely unmodified (the real C[] output for
            # every wMatrix is still produced, so --sweep's own
            # correctness check still reports on it normally). Always
            # paired with FW_TINYGPU_HYBRID_NV + flat dispatch -- the
            # specific configuration under investigation is Phase 111's
            # real-kernel rewrite, not the native 2D path.
            final_abcd_w0 = True
        elif argv[0] == "--combined-flat-w0":
            # TODO.md Phase 116, user: "go back to the other candidate
            # from Phase 114 (a combined-mechanisms early-return probe,
            # which has consistently been the safe pattern all session)"
            # -- Phase 113 tested Ds[] and As[]/Bs[] SEPARATELY (each its
            # own private storage, its own barrier), both 100% clean
            # under flat dispatch. This tests them TOGETHER, under ONE
            # shared barrier, exactly mirroring the real "Last block"
            # guard's combined write/barrier/read structure -- private
            # sAs/sBs/sDs (not the real arrays), early return before the
            # real "Last block" guard, same established safety pattern
            # as every flat-dispatch probe since Phase 109.
            combined_flat_w0 = True
        elif argv[0] == "--combined-ldg-flat-w0":
            # TODO.md Phase 118, user: "a combined probe like this one,
            # but sourcing its written values from real LDGs of A/B/D
            # instead of register constants -- still safe, still early-
            # return" -- same combined write/single-barrier/reduce
            # structure as --combined-flat-w0, but As/Bs/Ds source their
            # values from the real A[]/B[]/D[] buffers via genuine global-
            # memory loads (the exact real a=b=d=0 offsets this test's
            # wMatrix-0 block uses), without exp()/distance yet (Ds[tx]=
            # D[tx] raw, matching TINYGPU_BISECT_NO_EXP's established
            # convention -- one variable at a time).
            combined_ldg_flat_w0 = True
        elif argv[0] == "--combined-exp-flat-w0":
            # TODO.md Phase 118, user: "reintroducing the real exp()/
            # distance computation specifically into a synthetic probe"
            # -- identical to --combined-ldg-flat-w0, except Ds[] uses the
            # real, full formula exp(D[tx]*distance). With real A/B/D and
            # real exp()/distance, a correct result here should exactly
            # equal reference_transition_matrix()'s real answer -- the
            # strongest synthetic test this investigation can build
            # without touching the real, non-early-return kernel path.
            combined_exp_flat_w0 = True
        elif argv[0] == "--combined-write-direct-w0":
            # TODO.md Phase 120, user: "let's try the final write now.
            # but write to dMatrices[figure out indices] first before
            # trying dMatrix + listC[wMatrix]" -- identical combined real-
            # A/B/D/exp() mechanism as --combined-exp-flat-w0, but now
            # performs the REAL final write for real -- into wMatrix 0's
            # actual matrix-output slot in dMatrices, at a direct, closed-
            # form index (matching TINYGPU_BISECT_NO_LISTC's own
            # established closed form), not through listC[]'s
            # indirection. No new diagnostic region -- the write lands
            # directly in wMatrix 0's real slot, so --sweep's own
            # existing correctness check already answers it.
            combined_write_direct_w0 = True
        elif argv[0] == "--combined-write-listc-w0":
            # TODO.md Phase 120, user's own "before trying dMatrix +
            # listC[wMatrix]" follow-up -- identical to
            # --combined-write-direct-w0, except the write address comes
            # from the real `dMatrices + listC[wMatrix]` indirection
            # (a genuine data-dependent load), matching the real kernel's
            # own mechanism exactly. For this test's real listC values,
            # resolves to the same address as the direct probe -- isolates
            # the extra indirect load itself as the only variable.
            combined_write_listc_w0 = True
        elif argv[0] == "--single-tile-dispatch":
            # TODO.md Phase 126, user's proposal B: every PADDED_STATE_
            # COUNT this kernel supports is a multiple of MULTIPLY_BLOCK_
            # SIZE (16) except 4 -- the one config this whole investigation
            # has ever tested -- so the general FW_TINYGPU_HYBRID_NV
            # dispatch (256 threads/block) wastes 240 of 256 threads on
            # the EDGE-boundary guard's padding branch, a pattern unique
            # to this config. This flag adds a SECOND macro,
            # FW_TINYGPU_HYBRID_NV_SINGLE_TILE, on top of
            # FW_TINYGPU_HYBRID_NV -- kernelsAll.cu then derives tx/ty
            # from a PADDED_STATE_COUNT-wide (not MULTIPLY_BLOCK_SIZE-
            # wide) flat layout, and dispatches exactly PADDED_STATE_
            # COUNT^2 (16) threads/block instead of 256 -- every
            # dispatched thread does real work, no padding threads at
            # all. Real kernel, no early return, no diagnostic region --
            # --sweep's own existing correctness check answers whether
            # this is correct, same minimal design as --flat-dispatch.
            single_tile_dispatch = True
        elif argv[0] == "--maxrregcount":
            # TODO.md Phase 85: forces ptxas to cap kernelMatrixMulADB's
            # real, unmodified register allocation (naturally 40,
            # uncapped) at N, inducing local-memory spill if N is below
            # that -- pure compiler-flag bisection, tests the register-
            # pressure hypothesis Phase 84's trivial-kernel dial left
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
            wide_grid = 32  # default block count (Phase 76's approach 1: real grid, cat=0-vs-cat=1 slot-vs-value question resolved to "slot" -- next, does a wMatrix landing on a *different* SMID in a bigger grid still show its old reliability, or does reliability follow the SMID?)
            argv = argv[1:]
            if argv and argv[0].isdigit():
                wide_grid = int(argv[0])
                argv = argv[1:]
            continue
        argv = argv[1:]
    # --logl/--sweep need the *real*, unbisected kernel: TINYGPU_BISECT_NO_EXP
    # replaces Ds[tx]=exp(D[d+tx]*distance) with Ds[tx]=D[d+tx] (raw
    # eigenvalues, no exponential) -- essential for the ground-truth-dump
    # tests (it's what lets wMatrix>=4 execute at all) but it means the
    # "transition matrix" computed under it is A*diag(eigenvalues)*B --
    # the JC69 rate matrix Q itself (row sums exactly 0, negative entries:
    # verified directly, see STATUS.md), not a valid probability matrix.
    # Feeding that into a real likelihood computation legitimately
    # produces sum(pattern)=0 -> log(0)=-inf for every pattern -- correct
    # arithmetic on deliberately-wrong inputs, not a new finding. Default
    # to the true production kernel (no bisect macros at all) here so the
    # result is actually meaningful; explicit macros in argv still apply
    # (e.g. `--logl TINYGPU_BISECT_NO_EXP` to deliberately test the
    # bisected variant's effect on logL, if ever wanted for comparison).
    # TODO.md Phase 134: chain_sweep added here -- without it, --chain-
    # sweep alone (no --logl/--logl-sweep/--sweep) would have silently
    # fallen through to BASE_MACROS, which includes TINYGPU_BISECT_NO_EXP
    # and TINYGPU_BISECT_NO_LISTC -- a *bisected*, not the real, kernel.
    # Caught before ever running this on hardware.
    macros = (["TINYGPU_DEBUG_DUMP_MATMUL_GROUND_TRUTH"] if (logl or logl_sweep or chain_sweep or sweep) else BASE_MACROS) + argv
    if per_thread_ds:
        macros = macros + ["TINYGPU_DEBUG_DUMP_PER_THREAD_DS"]
    if per_thread_ds_w0:
        macros = macros + ["TINYGPU_DEBUG_DUMP_PER_THREAD_DS_W0"]
    if dummy_third_block:
        macros = macros + ["TINYGPU_DEBUG_DUMP_DUMMY_THIRD_BLOCK"]
    if per_thread_dummy_w0:
        macros = macros + ["TINYGPU_DEBUG_DUMP_PER_THREAD_DUMMY_W0"]
    if per_thread_ds_min_w0:
        macros = macros + ["TINYGPU_DEBUG_DUMP_PER_THREAD_DS_MIN_W0"]
    if local_mem_w0:
        macros = macros + ["TINYGPU_DEBUG_DUMP_LOCAL_MEM_W0"]
    if shared_spill_w0:
        macros = macros + ["TINYGPU_DEBUG_DUMP_SHARED_SPILL_W0"]
    if local_mem_flat_w0:
        macros = macros + ["TINYGPU_DEBUG_DUMP_LOCAL_MEM_FLAT_W0"]
    if shared_broadcast_flat_w0:
        macros = macros + ["TINYGPU_DEBUG_DUMP_SHARED_BROADCAST_FLAT_W0"]
    if flat_dispatch:
        macros = macros + ["FW_TINYGPU_HYBRID_NV"]
    if ab_pattern_2d_w0:
        macros = macros + ["TINYGPU_DEBUG_DUMP_AB_PATTERN_2D_W0"]
    if ab_pattern_flat_w0:
        macros = macros + ["TINYGPU_DEBUG_DUMP_AB_PATTERN_FLAT_W0"]
    if final_abcd_w0:
        macros = macros + ["TINYGPU_DEBUG_DUMP_FINAL_ABCD_CSUB_W0", "FW_TINYGPU_HYBRID_NV"]
    if combined_flat_w0:
        macros = macros + ["TINYGPU_DEBUG_DUMP_COMBINED_FLAT_W0"]
    if combined_ldg_flat_w0:
        macros = macros + ["TINYGPU_DEBUG_DUMP_COMBINED_LDG_FLAT_W0"]
    if combined_exp_flat_w0:
        macros = macros + ["TINYGPU_DEBUG_DUMP_COMBINED_EXP_FLAT_W0"]
    if combined_write_direct_w0:
        macros = macros + ["TINYGPU_DEBUG_DUMP_COMBINED_WRITE_DIRECT_W0"]
    if combined_write_listc_w0:
        macros = macros + ["TINYGPU_DEBUG_DUMP_COMBINED_WRITE_LISTC_W0"]
    if single_tile_dispatch:
        macros = macros + ["FW_TINYGPU_HYBRID_NV", "FW_TINYGPU_HYBRID_NV_SINGLE_TILE"]

    os.makedirs(os.path.expanduser("~/Library/Logs"), exist_ok=True)
    fd = os.open(os.path.expanduser("~/Library/Logs/nv_real_kernel_probe.log"),
                 os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_SYNC, 0o644)
    sys.stderr = os.fdopen(fd, 'w', buffering=1)

    log(f"starting -- batch={batch} realloc={realloc} logl={logl} logl_sweep={logl_sweep} chain_sweep={chain_sweep} sync_each={sync_each} downstream_sweep={downstream_sweep} sweep={sweep} swap_cat01={swap_cat01} wide_grid={wide_grid} maxrregcount={maxrregcount} per_thread_ds={per_thread_ds} per_thread_ds_w0={per_thread_ds_w0} dummy_third_block={dummy_third_block} per_thread_dummy_w0={per_thread_dummy_w0} per_thread_ds_min_w0={per_thread_ds_min_w0} local_mem_w0={local_mem_w0} shared_spill_w0={shared_spill_w0} local_mem_flat_w0={local_mem_flat_w0} shared_broadcast_flat_w0={shared_broadcast_flat_w0} flat_dispatch={flat_dispatch} ab_pattern_2d_w0={ab_pattern_2d_w0} ab_pattern_flat_w0={ab_pattern_flat_w0} final_abcd_w0={final_abcd_w0} combined_flat_w0={combined_flat_w0} combined_ldg_flat_w0={combined_ldg_flat_w0} combined_exp_flat_w0={combined_exp_flat_w0} combined_write_direct_w0={combined_write_direct_w0} combined_write_listc_w0={combined_write_listc_w0} single_tile_dispatch={single_tile_dispatch} macros={macros}")
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

    elf_bytes = compile_real_kernel(nch, dev, nvcc, macros, maxrregcount=maxrregcount)
    log(f"compiled kernelMatrixMulADB -- {len(elf_bytes)} byte ELF")

    # 3 trailing scalar (length, wB, totalMatrix) uint32 args -- matches
    # BEAGLE's KernelLauncher.cpp calling convention, same signature
    # scheme nv_dispatch_daemon.py's _get_program uses.
    signature = tuple((None, i, dtypes.uint32, ()) for i in range(3))
    obj = TinyELF(lib=elf_bytes, name="kernelMatrixMulADB", target=Target(), signature=signature)
    prg = BeagleNVProgram(dev, obj)
    log(f"regs_usage={prg.regs_usage} shmem_usage={prg.shmem_usage} lcmem_usage={prg.lcmem_usage}")

    # TODO.md Phase 109-126: the real kernel's own dispatch shape --
    # (256,1,1) flat for every FW_TINYGPU_HYBRID_NV-family macro
    # (Phase 111's --flat-dispatch and every w0-only flat probe since),
    # (S2,1,1) for --single-tile-dispatch (Phase 126), else the native
    # (16,16,1) BLOCK. Computed here, unconditionally, right after the
    # one compile step -- not inside `if sweep:` -- so every dispatch
    # mode (--sweep, --logl-sweep, --logl, --batch, --realloc) that goes
    # on to actually launch kernelMatrixMulADB sees the same, correct
    # local_size for whichever macros were just compiled in. (Bug found
    # and fixed this phase: --logl-sweep's own kernelMatrixMulADB
    # dispatch previously hardcoded local_size=BLOCK regardless of any
    # of these flags, since this computation used to live only inside
    # the --sweep branch below -- --logl-sweep --single-tile-dispatch
    # would have silently dispatched the wrong shape.)
    if single_tile_dispatch:
        # TODO.md Phase 126: exactly PADDED_STATE_COUNT^2 (S2)
        # threads/block, flat -- matches kernelsAll.cu's own
        # FW_TINYGPU_HYBRID_NV_SINGLE_TILE tx/ty derivation exactly.
        dispatch_local_size = (S2, 1, 1)
    elif (local_mem_flat_w0 or shared_broadcast_flat_w0 or flat_dispatch or ab_pattern_2d_w0 or ab_pattern_flat_w0 or final_abcd_w0 or combined_flat_w0 or combined_ldg_flat_w0 or combined_exp_flat_w0 or combined_write_direct_w0 or combined_write_listc_w0):
        dispatch_local_size = (256, 1, 1)
    else:
        dispatch_local_size = BLOCK

    if sweep:
        # ---- Phase 71 (sweep): one boot, one compile, then `sweep` fresh
        # dispatches of the real, unbisected kernelMatrixMulADB -- fresh
        # buffers every iteration, matching the real content --logl uses
        # -- tabulating per-wMatrix success rate directly, to test
        # whether "the middle of blockIdx.x space never runs" (observed
        # in 4 separate process-level --logl runs) is unconditional or
        # just unlucky in a small sample. Only kernelMatrixMulADB itself
        # is dispatched -- the question is which blocks write anything at
        # all, not the downstream likelihood chain.
        distance_vals = [EDGE_LENS[i] * CATEGORY_RATES[j] for i in range(4) for j in range(4)]
        if swap_cat01:
            # TODO.md Phase 75/76: separates "it's the actual exp() argument
            # value" from "it's the wMatrix/SMID slot" for the cat=0-vs-
            # cat=1 asymmetry found within TPC1/2/3 (wMatrix 4/8/12 get rare
            # extra successes, wMatrix 5/9/13 never do). Swaps only which
            # *value* each of those two positions gets, per edge --
            # wMatrix stays fixed (so its SMID assignment, per the
            # mechanically-confirmed fixed mapping, is unchanged), only
            # distance_vals[edge*4+0] and distance_vals[edge*4+1] trade
            # places. If the success pattern follows the value (moves to
            # wMatrix 5/9/13), it's the number; if it stays with wMatrix
            # 4/8/12 regardless, it's the slot.
            for edge in range(4):
                i0, i1 = edge * 4 + 0, edge * 4 + 1
                distance_vals[i0], distance_vals[i1] = distance_vals[i1], distance_vals[i0]
            log(f"--swap-cat01: distance_vals[edge*4+0] <-> distance_vals[edge*4+1] for every edge -- "
                f"distance_vals={distance_vals}")

        # TODO.md Phase 76/77: Phase 77's swap-cat01 result showed the
        # cat=0-vs-cat=1 asymmetry tracks the wMatrix/SMID *slot*, not the
        # value. Phase 78's `--wide-grid [N]` (default N=32) follows up:
        # launches N blocks instead of 16 -- the real kernel's own
        # `wMatrix = blockIdx.x % totalMatrix` (kernelsAll.cu) means
        # passing `totalMatrix=n_blocks` alongside an N-block grid keeps
        # every block's `bx = blockIdx.x / totalMatrix` at 0, the same
        # safe, already-exercised BLOCKS==1 path every other --sweep run
        # uses -- no kernelsAll.cu changes needed. distance_vals/listC are
        # extended cyclically (block 16 is a structural copy of block 0's
        # setup, 17 of 1, etc.) so every block still gets a valid, unique
        # output slot; A/B/D stay the same shared buffers regardless of
        # block count. Whatever physical SM(s) the extra blocks land on
        # (unknown ahead of time -- that's what this run's own SMID table
        # answers), the question is whether elevated reliability follows
        # the SMID (e.g. a wide-grid block that lands on SMID 4 behaves
        # like wMatrix 8 always has) or stays with the original
        # wMatrix-mod-16 identity regardless of which SMID it lands on.
        n_blocks = wide_grid if wide_grid else TOTAL_MATRIX
        if wide_grid:
            distance_vals = [distance_vals[w % TOTAL_MATRIX] for w in range(n_blocks)]
            log(f"--wide-grid {n_blocks}: grid=({n_blocks},1,1), totalMatrix={n_blocks}, "
                f"distance_vals cyclically extended (block w uses block w%{TOTAL_MATRIX}'s value)")
        listc_vals = [w * S2 for w in range(n_blocks)]
        # TODO.md Phase 93: --per-thread-ds adds a third dmat region past
        # the real matrices and the existing ground-truth scratch --
        # STATE_COUNT**3 (64) floats per block, one 4-float Ds[0..3] view
        # per each of the 16 real threads (tx,ty in [0,EDGE)), matching
        # kernelsAll.cu's own TINYGPU_DEBUG_DUMP_PER_THREAD_DS layout
        # exactly (KW_GROUP_ID_0*STATE_COUNT**3 + (ty*EDGE+tx)*STATE_COUNT).
        per_thread_ds_size = STATE_COUNT ** 3
        n_dmat_floats = 2 * n_blocks * S2
        dmat_init = [0.0] * (n_blocks * S2) + [SENTINEL] * (n_blocks * S2)
        if per_thread_ds:
            n_dmat_floats += n_blocks * per_thread_ds_size
            dmat_init += [SENTINEL] * (n_blocks * per_thread_ds_size)
        elif per_thread_ds_w0:
            # TODO.md Phase 96: only wMatrix 0's block writes (kernelsAll.cu's
            # TINYGPU_DEBUG_DUMP_PER_THREAD_DS_W0 guards on KW_GROUP_ID_0==0),
            # so only one block's worth of the third region is ever touched --
            # sized/seeded for exactly that, not n_blocks*per_thread_ds_size.
            n_dmat_floats += per_thread_ds_size
            dmat_init += [SENTINEL] * per_thread_ds_size
        # TODO.md Phase 100: --dummy-third-block adds its own third dmat
        # region, at the *same* base offset D would use (2*n_blocks*S2) --
        # not tested combined with --per-thread-ds(-w0) in this experiment,
        # so no collision in practice -- but only n_blocks floats (one
        # hardcoded-constant write per block, matching kernelsAll.cu's
        # TINYGPU_DEBUG_DUMP_DUMMY_THIRD_BLOCK), far smaller than even
        # per_thread_ds_size (64 floats/block).
        if dummy_third_block:
            n_dmat_floats += n_blocks
            dmat_init += [SENTINEL] * n_blocks
        # TODO.md Phase 101: --per-thread-dummy-w0 adds its own third dmat
        # region, same base offset as the other third-region probes --
        # STATE_COUNT**2 (16) floats, one per real thread in wMatrix 0's
        # block only, matching kernelsAll.cu's TINYGPU_DEBUG_DUMP_PER_
        # THREAD_DUMMY_W0 layout exactly. Same total size as --dummy-
        # third-block's own region (16 floats) -- deliberate, for a
        # single-variable (granularity-only) comparison.
        per_thread_dummy_size = STATE_COUNT ** 2
        if per_thread_dummy_w0:
            n_dmat_floats += per_thread_dummy_size
            dmat_init += [SENTINEL] * per_thread_dummy_size
        # TODO.md Phase 102: --per-thread-ds-min-w0 adds its own third
        # dmat region, same base offset and same size (STATE_COUNT**2,
        # 16 floats) as --per-thread-dummy-w0's own region -- deliberate,
        # for a single-variable (content-only, real Ds[] read vs. a
        # trivial local value) comparison at identical footprint.
        per_thread_ds_min_size = STATE_COUNT ** 2
        if per_thread_ds_min_w0:
            n_dmat_floats += per_thread_ds_min_size
            dmat_init += [SENTINEL] * per_thread_ds_min_size
        # TODO.md Phase 106: --local-mem-w0 adds its own third dmat
        # region, same base offset and size (STATE_COUNT**2, 16 floats)
        # as the other w0-only per-thread probes -- tests LOCAL memory
        # (register spill) correctness instead of shared memory.
        local_mem_size = STATE_COUNT ** 2
        if local_mem_w0:
            n_dmat_floats += local_mem_size
            dmat_init += [SENTINEL] * local_mem_size
        # TODO.md Phase 108: --shared-spill-w0 adds its own third dmat
        # region, same base offset and size as --local-mem-w0 -- tests
        # explicit shared-memory storage instead of the driver's local-
        # memory-window mechanism, otherwise an identical diagnostic.
        shared_spill_size = STATE_COUNT ** 2
        if shared_spill_w0:
            n_dmat_floats += shared_spill_size
            dmat_init += [SENTINEL] * shared_spill_size
        # TODO.md Phase 109: --local-mem-flat-w0 adds its own third dmat
        # region, same base offset and size as --local-mem-w0 -- tests
        # the same local-memory mechanism under a flat (256,1,1) block
        # dispatch instead of the kernel's normal (16,16,1).
        local_mem_flat_size = STATE_COUNT ** 2
        if local_mem_flat_w0:
            n_dmat_floats += local_mem_flat_size
            dmat_init += [SENTINEL] * local_mem_flat_size
        # TODO.md Phase 110: --shared-broadcast-flat-w0 adds its own
        # third dmat region, same base offset and size -- tests Ds[]'s
        # broadcast mechanism under a flat (256,1,1) block dispatch.
        shared_broadcast_flat_size = STATE_COUNT ** 2
        if shared_broadcast_flat_w0:
            n_dmat_floats += shared_broadcast_flat_size
            dmat_init += [SENTINEL] * shared_broadcast_flat_size
        # TODO.md Phase 113: --ab-pattern-2d-w0 adds its own third dmat
        # region, same base offset as the other w0-only probes -- 16
        # threads * (4 As[ty][k] + 4 Bs[k][tx]) values = 128 floats,
        # matching kernelsAll.cu's TINYGPU_DEBUG_DUMP_AB_PATTERN_2D_W0
        # layout exactly.
        ab_pattern_2d_size = STATE_COUNT * STATE_COUNT * 2 * STATE_COUNT
        if ab_pattern_2d_w0:
            n_dmat_floats += ab_pattern_2d_size
            dmat_init += [SENTINEL] * ab_pattern_2d_size
        # TODO.md Phase 113: --ab-pattern-flat-w0 adds its own third dmat
        # region, same base offset -- 16 threads * (4 matching-As +
        # 4 matching-Bs + 4 swapped-As + 4 swapped-Bs) values = 256
        # floats, matching TINYGPU_DEBUG_DUMP_AB_PATTERN_FLAT_W0's layout.
        ab_pattern_flat_size = STATE_COUNT * STATE_COUNT * 4 * STATE_COUNT
        if ab_pattern_flat_w0:
            n_dmat_floats += ab_pattern_flat_size
            dmat_init += [SENTINEL] * ab_pattern_flat_size
        # TODO.md Phase 115: --final-abcd-w0 adds its own third dmat
        # region, same base offset -- 16 threads * (4 As[ty][k] + 4
        # Bs[k][tx] + 4 Ds[k] + 1 Csub) = 16*13 = 208 floats, matching
        # kernelsAll.cu's TINYGPU_DEBUG_DUMP_FINAL_ABCD_CSUB_W0 layout.
        final_abcd_size = STATE_COUNT * STATE_COUNT * (3 * STATE_COUNT + 1)
        if final_abcd_w0:
            n_dmat_floats += final_abcd_size
            dmat_init += [SENTINEL] * final_abcd_size
        # TODO.md Phase 116: --combined-flat-w0 adds its own third dmat
        # region, same base offset and layout size as --final-abcd-w0
        # (16 threads * (4 As + 4 Bs + 4 Ds + 1 Csub) = 208 floats).
        combined_flat_size = STATE_COUNT * STATE_COUNT * (3 * STATE_COUNT + 1)
        if combined_flat_w0:
            n_dmat_floats += combined_flat_size
            dmat_init += [SENTINEL] * combined_flat_size
        # TODO.md Phase 118: --combined-ldg-flat-w0/--combined-exp-flat-w0
        # each add their own third dmat region, same base offset and
        # layout size as --combined-flat-w0.
        combined_ldg_flat_size = STATE_COUNT * STATE_COUNT * (3 * STATE_COUNT + 1)
        if combined_ldg_flat_w0:
            n_dmat_floats += combined_ldg_flat_size
            dmat_init += [SENTINEL] * combined_ldg_flat_size
        combined_exp_flat_size = STATE_COUNT * STATE_COUNT * (3 * STATE_COUNT + 1)
        if combined_exp_flat_w0:
            n_dmat_floats += combined_exp_flat_size
            dmat_init += [SENTINEL] * combined_exp_flat_size

        # TODO.md Phase 90: every prior --sweep run only ever checked
        # "wrote anything nonzero" -- never whether the value written was
        # actually the *correct* transition matrix. Computed once here
        # (distance_vals is fixed for the whole run) so every iteration's
        # real C[] readback can be checked against the real answer, not
        # just against zero.
        reference_matrices = [reference_transition_matrix(distance_vals[w]) for w in range(n_blocks)]
        # TODO.md Phase 102: reference Ds[tx] = exp(EVAL[tx]*distance) for
        # wMatrix 0 -- the same per-element formula reference_transition_
        # matrix() uses internally, exposed here since --per-thread-ds-
        # min-w0 checks Ds[] readback directly, not the downstream matrix.
        ds_reference_w0 = [math.exp(EVAL[k] * distance_vals[0]) for k in range(STATE_COUNT)]

        success_count = [0] * n_blocks     # any of the 16 entries nonzero (the old, weaker metric -- kept for comparability)
        correct_count = [0] * n_blocks     # all 16 entries match the real reference matrix within CORRECTNESS_TOL
        wrong_count = [0] * n_blocks       # wrote nonzero but does NOT match the reference -- a *real* failure, not a proxy
        full_row0_count = [0] * n_blocks   # row ty=0 (entries 0-3) all nonzero (the old, weaker per-row metric)
        # TODO.md Phase 93: per-thread Ds[] broadcast-visibility check --
        # ds_broadcast_ok_count[w] counts iterations where every one of
        # the 12 non-writer threads (ty=1/2/3, tx=0..3) saw *exactly* the
        # same Ds[0..3] as the writer row (ty=0) did; ds_broadcast_
        # checked_count[w] counts iterations where the writer row itself
        # was captured (dbg[0]!=SENTINEL there) so a comparison was even
        # possible.
        ds_broadcast_ok_count = [0] * n_blocks
        ds_broadcast_checked_count = [0] * n_blocks
        # TODO.md Phase 100: dummy_ok_count[w] counts iterations where
        # wMatrix w's dummy-third-block slot read back exactly the
        # hardcoded constant; dummy_checked_count[w] counts iterations
        # where that slot was written at all (!= SENTINEL).
        dummy_ok_count = [0] * n_blocks
        dummy_checked_count = [0] * n_blocks
        # TODO.md Phase 101: per_thread_dummy_ok_count/checked_count track,
        # per wMatrix (only wMatrix 0 ever populated -- w0-only probe),
        # how many of the 16 real threads' own per-thread-identifiable
        # slots read back the correct (ty*EDGE+tx) value.
        per_thread_dummy_ok_count = [0] * n_blocks
        per_thread_dummy_checked_count = [0] * n_blocks
        # TODO.md Phase 103: per_thread_ds_min_ok_count/checked_count now
        # track PER-SLOT (one counter per (ty,tx), 16 total, wMatrix 0
        # only) -- Phase 102's 7/16 aggregate result directly confirmed
        # the long-standing Ds[] broadcast-visibility hypothesis for the
        # first time without inferring it from downstream C[] output, but
        # gave no indication of *which* threads are reliably right/wrong.
        # Per-slot tracking (plus running this with real sweep count, not
        # just sweep=1) gives that resolution with statistical power.
        per_thread_ds_min_ok_count = [0] * S2
        per_thread_ds_min_checked_count = [0] * S2
        # TODO.md Phase 106: local_mem_ok_count/checked_count track,
        # per-slot (16 total, wMatrix 0 only), how many of the 16 real
        # threads' own spilled-and-reloaded local-memory values matched
        # what that same thread itself wrote.
        local_mem_ok_count = [0] * S2
        local_mem_checked_count = [0] * S2
        # TODO.md Phase 108: shared_spill_ok_count/checked_count track
        # the same thing as local_mem_*, but for --shared-spill-w0's
        # explicit-shared-memory storage instead of local memory.
        shared_spill_ok_count = [0] * S2
        shared_spill_checked_count = [0] * S2
        # TODO.md Phase 109: local_mem_flat_ok_count/checked_count track
        # the same thing as local_mem_*, but under a flat (256,1,1)
        # dispatch instead of the kernel's normal (16,16,1).
        local_mem_flat_ok_count = [0] * S2
        local_mem_flat_checked_count = [0] * S2
        # TODO.md Phase 110: shared_broadcast_flat_ok_count/checked_count
        # track Ds[]'s own broadcast mechanism under a flat dispatch.
        shared_broadcast_flat_ok_count = [0] * S2
        shared_broadcast_flat_checked_count = [0] * S2
        # TODO.md Phase 113: ab_pattern_2d_*_ok_count/checked_count track,
        # per-slot (16 total, wMatrix 0 only), the real kernel's own
        # As[ty][k] (row-wise) / Bs[k][tx] (column-wise) access patterns
        # directly -- kept as separate as/bs tables since these are
        # mechanistically distinct reads (same-ty-different-writer-tx vs.
        # same-tx-different-writer-ty) that could fail independently.
        ab_pattern_2d_as_ok_count = [0] * S2
        ab_pattern_2d_as_checked_count = [0] * S2
        ab_pattern_2d_bs_ok_count = [0] * S2
        ab_pattern_2d_bs_checked_count = [0] * S2
        # TODO.md Phase 113: ab_pattern_flat_*_ok_count/checked_count track
        # the same two patterns under a flat REAL[256] array with
        # hand-computed addresses (match_* mirrors ab_pattern_2d_*'s own
        # read exactly; swap_* is the user-suggested tx*16+k alternate
        # indexing, a genuinely different set of physical slots).
        ab_pattern_flat_as_match_ok_count = [0] * S2
        ab_pattern_flat_as_match_checked_count = [0] * S2
        ab_pattern_flat_bs_match_ok_count = [0] * S2
        ab_pattern_flat_bs_match_checked_count = [0] * S2
        ab_pattern_flat_as_swap_ok_count = [0] * S2
        ab_pattern_flat_as_swap_checked_count = [0] * S2
        ab_pattern_flat_bs_swap_ok_count = [0] * S2
        ab_pattern_flat_bs_swap_checked_count = [0] * S2
        # TODO.md Phase 115: final_abcd_*_ok_count/checked_count track,
        # per-slot (16 total, wMatrix 0 only), the REAL As[ty][k]/
        # Bs[k][tx]/Ds[k] (this thread's own view)/Csub, captured right
        # before the real barrier/C[] write in Phase 111's real,
        # combined, non-early-return kernel path.
        final_abcd_as_ok_count = [0] * S2
        final_abcd_as_checked_count = [0] * S2
        final_abcd_bs_ok_count = [0] * S2
        final_abcd_bs_checked_count = [0] * S2
        final_abcd_ds_ok_count = [0] * S2
        final_abcd_ds_checked_count = [0] * S2
        final_abcd_csub_ok_count = [0] * S2
        final_abcd_csub_checked_count = [0] * S2
        # TODO.md Phase 116: combined_flat_*_ok_count/checked_count track,
        # per-slot (16 total, wMatrix 0 only), --combined-flat-w0's
        # private As/Bs/Ds/Csub-analog, written/synced/read together
        # exactly mirroring the real kernel's own combined structure.
        combined_flat_as_ok_count = [0] * S2
        combined_flat_as_checked_count = [0] * S2
        combined_flat_bs_ok_count = [0] * S2
        combined_flat_bs_checked_count = [0] * S2
        combined_flat_ds_ok_count = [0] * S2
        combined_flat_ds_checked_count = [0] * S2
        combined_flat_csub_ok_count = [0] * S2
        combined_flat_csub_checked_count = [0] * S2
        # TODO.md Phase 118: combined_ldg_flat_*/combined_exp_flat_*
        # ok_count/checked_count track the same As/Bs/Ds/Csub-analog
        # per-slot correctness, but for the real-LDG-sourced and real-
        # exp()/distance-sourced combined probes respectively.
        combined_ldg_flat_as_ok_count = [0] * S2
        combined_ldg_flat_as_checked_count = [0] * S2
        combined_ldg_flat_bs_ok_count = [0] * S2
        combined_ldg_flat_bs_checked_count = [0] * S2
        combined_ldg_flat_ds_ok_count = [0] * S2
        combined_ldg_flat_ds_checked_count = [0] * S2
        combined_ldg_flat_csub_ok_count = [0] * S2
        combined_ldg_flat_csub_checked_count = [0] * S2
        combined_exp_flat_as_ok_count = [0] * S2
        combined_exp_flat_as_checked_count = [0] * S2
        combined_exp_flat_bs_ok_count = [0] * S2
        combined_exp_flat_bs_checked_count = [0] * S2
        combined_exp_flat_ds_ok_count = [0] * S2
        combined_exp_flat_ds_checked_count = [0] * S2
        combined_exp_flat_csub_ok_count = [0] * S2
        combined_exp_flat_csub_checked_count = [0] * S2
        all_populated = []  # per-iteration set of populated wMatrix, for exact-pattern comparison
        iter_times = []     # wall-clock seconds for dispatch+synchronize, per iteration -- a real,
                             # independent signal of GPU clock/DVFS warm-up state, to test directly
                             # whether success correlates with iteration index / dispatch speed
                             # rather than just asserting a "warm-up" story from the pattern alone.

        # STATUS.md #135: a claimed wMatrix->SMID->TPC pattern was assembled
        # from memory of earlier pasted output, not re-extracted from logs
        # mechanically -- flagged as unverified. dbg[13] (the ground-truth
        # scratch region's %smid capture, kernelsAll.cu ~line 572) is
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
            # TODO.md Phase 93: third region, only present when
            # --per-thread-ds is set -- per wMatrix, 16 threads' own
            # 4-float Ds[0..3] views, laid out (ty*EDGE+tx)*STATE_COUNT
            # exactly matching kernelsAll.cu's TINYGPU_DEBUG_DUMP_PER_
            # THREAD_DS. EDGE==STATE_COUNT==4 for this BLOCKS==1 config.
            per_thread_region = None
            if per_thread_ds:
                per_thread_region = dmat_vals[2 * n_blocks * S2:2 * n_blocks * S2 + n_blocks * per_thread_ds_size]
            elif per_thread_ds_w0:
                # TODO.md Phase 96: only wMatrix 0's block ever writes here.
                per_thread_region = dmat_vals[2 * n_blocks * S2:2 * n_blocks * S2 + per_thread_ds_size]
            # TODO.md Phase 100: third region, only present when
            # --dummy-third-block is set -- one hardcoded-constant float
            # per block, at the same base offset as per_thread_region above
            # (not combined with --per-thread-ds(-w0) in this experiment).
            dummy_region = None
            if dummy_third_block:
                dummy_region = dmat_vals[2 * n_blocks * S2:2 * n_blocks * S2 + n_blocks]
            # TODO.md Phase 101: third region, only present when
            # --per-thread-dummy-w0 is set -- one per-thread-identifiable
            # float per real thread in wMatrix 0's block, same base
            # offset as the other third-region probes above.
            per_thread_dummy_region = None
            if per_thread_dummy_w0:
                per_thread_dummy_region = dmat_vals[2 * n_blocks * S2:2 * n_blocks * S2 + per_thread_dummy_size]
            # TODO.md Phase 102: third region, only present when
            # --per-thread-ds-min-w0 is set -- one real Ds[tx] readback
            # per real thread in wMatrix 0's block, same base offset and
            # size as per_thread_dummy_region above.
            per_thread_ds_min_region = None
            if per_thread_ds_min_w0:
                per_thread_ds_min_region = dmat_vals[2 * n_blocks * S2:2 * n_blocks * S2 + per_thread_ds_min_size]
            # TODO.md Phase 106: third region, only present when
            # --local-mem-w0 is set -- one real spilled-local-memory
            # readback per real thread in wMatrix 0's block, same base
            # offset and size as the other w0-only per-thread probes.
            local_mem_region = None
            if local_mem_w0:
                local_mem_region = dmat_vals[2 * n_blocks * S2:2 * n_blocks * S2 + local_mem_size]
            # TODO.md Phase 108: third region, only present when
            # --shared-spill-w0 is set -- same layout as local_mem_region.
            shared_spill_region = None
            if shared_spill_w0:
                shared_spill_region = dmat_vals[2 * n_blocks * S2:2 * n_blocks * S2 + shared_spill_size]
            # TODO.md Phase 109: third region, only present when
            # --local-mem-flat-w0 is set -- same layout as local_mem_region.
            local_mem_flat_region = None
            if local_mem_flat_w0:
                local_mem_flat_region = dmat_vals[2 * n_blocks * S2:2 * n_blocks * S2 + local_mem_flat_size]
            # TODO.md Phase 110: third region, only present when
            # --shared-broadcast-flat-w0 is set -- same layout.
            shared_broadcast_flat_region = None
            if shared_broadcast_flat_w0:
                shared_broadcast_flat_region = dmat_vals[2 * n_blocks * S2:2 * n_blocks * S2 + shared_broadcast_flat_size]
            # TODO.md Phase 113: third region, only present when
            # --ab-pattern-2d-w0 is set -- 128 floats, 16 threads x
            # (4 As[ty][k] + 4 Bs[k][tx]).
            ab_pattern_2d_region = None
            if ab_pattern_2d_w0:
                ab_pattern_2d_region = dmat_vals[2 * n_blocks * S2:2 * n_blocks * S2 + ab_pattern_2d_size]
            # TODO.md Phase 113: third region, only present when
            # --ab-pattern-flat-w0 is set -- 256 floats, 16 threads x
            # (4 matching-As + 4 matching-Bs + 4 swapped-As + 4 swapped-Bs).
            ab_pattern_flat_region = None
            if ab_pattern_flat_w0:
                ab_pattern_flat_region = dmat_vals[2 * n_blocks * S2:2 * n_blocks * S2 + ab_pattern_flat_size]
            # TODO.md Phase 115: third region, only present when
            # --final-abcd-w0 is set -- 208 floats, 16 threads x
            # (4 As + 4 Bs + 4 Ds + 1 Csub).
            final_abcd_region = None
            if final_abcd_w0:
                final_abcd_region = dmat_vals[2 * n_blocks * S2:2 * n_blocks * S2 + final_abcd_size]
            # TODO.md Phase 116: third region, only present when
            # --combined-flat-w0 is set -- same layout as final_abcd_region.
            combined_flat_region = None
            if combined_flat_w0:
                combined_flat_region = dmat_vals[2 * n_blocks * S2:2 * n_blocks * S2 + combined_flat_size]
            # TODO.md Phase 118: third region, only present when
            # --combined-ldg-flat-w0/--combined-exp-flat-w0 is set --
            # same layout as combined_flat_region.
            combined_ldg_flat_region = None
            if combined_ldg_flat_w0:
                combined_ldg_flat_region = dmat_vals[2 * n_blocks * S2:2 * n_blocks * S2 + combined_ldg_flat_size]
            combined_exp_flat_region = None
            if combined_exp_flat_w0:
                combined_exp_flat_region = dmat_vals[2 * n_blocks * S2:2 * n_blocks * S2 + combined_exp_flat_size]

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
                if per_thread_ds or (per_thread_ds_w0 and w == 0):
                    # Compare every one of the 16 threads' own captured
                    # Ds[0..3] against the writer row's (ty=0, tx=0's
                    # slot -- all 4 ty=0 threads should already agree
                    # with each other, they wrote the array together).
                    pt = (per_thread_region[w * per_thread_ds_size:(w + 1) * per_thread_ds_size] if per_thread_ds
                          else per_thread_region[0:per_thread_ds_size])
                    writer_ds = pt[0:STATE_COUNT]
                    if writer_ds[0] != SENTINEL:
                        ds_broadcast_checked_count[w] += 1
                        all_match = True
                        for ty in range(STATE_COUNT):
                            for tx in range(STATE_COUNT):
                                idx = (ty * STATE_COUNT + tx) * STATE_COUNT
                                reader_ds = pt[idx:idx + STATE_COUNT]
                                if reader_ds[0] == SENTINEL or any(abs(a - b) > 1e-9 for a, b in zip(reader_ds, writer_ds)):
                                    all_match = False
                        if all_match:
                            ds_broadcast_ok_count[w] += 1
                if dummy_third_block:
                    # TODO.md Phase 100: kernelsAll.cu's TINYGPU_DEBUG_DUMP_
                    # DUMMY_THIRD_BLOCK writes the literal constant 42 --
                    # any other value (or SENTINEL, meaning never written)
                    # is a real failure of this trivial, content-unrelated
                    # third write, not something ambiguous to interpret.
                    val = dummy_region[w]
                    if val != SENTINEL:
                        dummy_checked_count[w] += 1
                        if val == 42.0:
                            dummy_ok_count[w] += 1
                if per_thread_dummy_w0 and w == 0:
                    # TODO.md Phase 101: kernelsAll.cu's TINYGPU_DEBUG_
                    # DUMP_PER_THREAD_DUMMY_W0 writes each of the 16 real
                    # threads' own (ty*EDGE+tx) index to its own slot --
                    # checked per-slot (not per-iteration, unlike the
                    # other tables above) since there are 16 independent
                    # writers within this one block.
                    for ty in range(STATE_COUNT):
                        for tx in range(STATE_COUNT):
                            idx = ty * STATE_COUNT + tx
                            val = per_thread_dummy_region[idx]
                            if val != SENTINEL:
                                per_thread_dummy_checked_count[w] += 1
                                if val == float(idx):
                                    per_thread_dummy_ok_count[w] += 1
                if per_thread_ds_min_w0 and w == 0:
                    # TODO.md Phase 102/103: kernelsAll.cu's TINYGPU_DEBUG_
                    # DUMP_PER_THREAD_DS_MIN_W0 writes each of the 16 real
                    # threads' own Ds[tx] readback to its own slot --
                    # checked per-slot (not aggregated across slots, since
                    # Phase 102 showed the result is NOT uniform) against
                    # the reference exp(EVAL[tx]*distance).
                    for ty in range(STATE_COUNT):
                        for tx in range(STATE_COUNT):
                            idx = ty * STATE_COUNT + tx
                            val = per_thread_ds_min_region[idx]
                            if val != SENTINEL:
                                per_thread_ds_min_checked_count[idx] += 1
                                if abs(val - ds_reference_w0[tx]) < CORRECTNESS_TOL:
                                    per_thread_ds_min_ok_count[idx] += 1
                if local_mem_w0 and w == 0:
                    # TODO.md Phase 106: kernelsAll.cu's TINYGPU_DEBUG_DUMP_
                    # LOCAL_MEM_W0 writes each of the 16 real threads' own
                    # spilled-and-reloaded local-memory value (the slot its
                    # own 16-step index recurrence lands on, matching
                    # kernelsAll.cu's TINYGPU_DEBUG_DUMP_LOCAL_MEM_W0
                    # exactly -- see local_mem_expected()) to its own slot.
                    for ty in range(STATE_COUNT):
                        for tx in range(STATE_COUNT):
                            idx = ty * STATE_COUNT + tx
                            val = local_mem_region[idx]
                            if val != SENTINEL:
                                local_mem_checked_count[idx] += 1
                                expected = local_mem_expected(ty, tx, n_blocks)
                                if abs(val - expected) < CORRECTNESS_TOL:
                                    local_mem_ok_count[idx] += 1
                if shared_spill_w0 and w == 0:
                    # TODO.md Phase 108: kernelsAll.cu's TINYGPU_DEBUG_
                    # DUMP_SHARED_SPILL_W0 writes each of the 16 real
                    # threads' own explicit-shared-memory readback --
                    # same recurrence/formula as local_mem_expected(),
                    # just a different storage class.
                    for ty in range(STATE_COUNT):
                        for tx in range(STATE_COUNT):
                            idx = ty * STATE_COUNT + tx
                            val = shared_spill_region[idx]
                            if val != SENTINEL:
                                shared_spill_checked_count[idx] += 1
                                expected = local_mem_expected(ty, tx, n_blocks)
                                if abs(val - expected) < CORRECTNESS_TOL:
                                    shared_spill_ok_count[idx] += 1
                if local_mem_flat_w0 and w == 0:
                    # TODO.md Phase 109: kernelsAll.cu's TINYGPU_DEBUG_
                    # DUMP_LOCAL_MEM_FLAT_W0 writes each of the 16
                    # logical (ty,tx) threads' own local-memory readback,
                    # under a flat (256,1,1) dispatch -- same recurrence/
                    # formula as local_mem_expected().
                    for ty in range(STATE_COUNT):
                        for tx in range(STATE_COUNT):
                            idx = ty * STATE_COUNT + tx
                            val = local_mem_flat_region[idx]
                            if val != SENTINEL:
                                local_mem_flat_checked_count[idx] += 1
                                expected = local_mem_expected(ty, tx, n_blocks)
                                if abs(val - expected) < CORRECTNESS_TOL:
                                    local_mem_flat_ok_count[idx] += 1
                if shared_broadcast_flat_w0 and w == 0:
                    # TODO.md Phase 110: kernelsAll.cu's TINYGPU_DEBUG_
                    # DUMP_SHARED_BROADCAST_FLAT_W0 writes each of the 16
                    # logical (ty,tx) threads' own sDs[logicalTx]
                    # readback, under a flat (256,1,1) dispatch -- checked
                    # against ds_reference_w0[tx] (Ds[]'s own reference,
                    # already computed unconditionally above), same as
                    # --per-thread-ds-min-w0's own check.
                    for ty in range(STATE_COUNT):
                        for tx in range(STATE_COUNT):
                            idx = ty * STATE_COUNT + tx
                            val = shared_broadcast_flat_region[idx]
                            if val != SENTINEL:
                                shared_broadcast_flat_checked_count[idx] += 1
                                if abs(val - ds_reference_w0[tx]) < CORRECTNESS_TOL:
                                    shared_broadcast_flat_ok_count[idx] += 1
                if ab_pattern_2d_w0 and w == 0:
                    # TODO.md Phase 113: kernelsAll.cu's TINYGPU_DEBUG_DUMP_
                    # AB_PATTERN_2D_W0 writes each of the 16 logical (ty,tx)
                    # threads' own As[ty][0..3]/Bs[0..3][tx] readback --
                    # a slot's "ok" requires all 4 As k-values (and,
                    # separately, all 4 Bs k-values) to match what the
                    # actual writer thread wrote (ab_pattern_expected).
                    for ty in range(STATE_COUNT):
                        for tx in range(STATE_COUNT):
                            idx = ty * STATE_COUNT + tx
                            base = idx * 2 * STATE_COUNT
                            as_vals = ab_pattern_2d_region[base:base + STATE_COUNT]
                            bs_vals = ab_pattern_2d_region[base + STATE_COUNT:base + 2 * STATE_COUNT]
                            if as_vals[0] != SENTINEL:
                                ab_pattern_2d_as_checked_count[idx] += 1
                                if all(abs(as_vals[k] - ab_pattern_expected(ty, k)) < CORRECTNESS_TOL for k in range(STATE_COUNT)):
                                    ab_pattern_2d_as_ok_count[idx] += 1
                            if bs_vals[0] != SENTINEL:
                                ab_pattern_2d_bs_checked_count[idx] += 1
                                if all(abs(bs_vals[k] - ab_pattern_expected(k, tx)) < CORRECTNESS_TOL for k in range(STATE_COUNT)):
                                    ab_pattern_2d_bs_ok_count[idx] += 1
                if ab_pattern_flat_w0 and w == 0:
                    # TODO.md Phase 113: kernelsAll.cu's TINYGPU_DEBUG_DUMP_
                    # AB_PATTERN_FLAT_W0 writes each of the 16 logical
                    # (ty,tx) threads' own 4 readback groups (matching-As,
                    # matching-Bs, swapped-As, swapped-Bs) -- same
                    # all-4-k-match-per-slot convention as --ab-pattern-2d-
                    # w0 above, one table per group.
                    for ty in range(STATE_COUNT):
                        for tx in range(STATE_COUNT):
                            idx = ty * STATE_COUNT + tx
                            base = idx * 4 * STATE_COUNT
                            as_match = ab_pattern_flat_region[base:base + STATE_COUNT]
                            bs_match = ab_pattern_flat_region[base + STATE_COUNT:base + 2 * STATE_COUNT]
                            as_swap = ab_pattern_flat_region[base + 2 * STATE_COUNT:base + 3 * STATE_COUNT]
                            bs_swap = ab_pattern_flat_region[base + 3 * STATE_COUNT:base + 4 * STATE_COUNT]
                            if as_match[0] != SENTINEL:
                                ab_pattern_flat_as_match_checked_count[idx] += 1
                                if all(abs(as_match[k] - ab_pattern_expected(ty, k)) < CORRECTNESS_TOL for k in range(STATE_COUNT)):
                                    ab_pattern_flat_as_match_ok_count[idx] += 1
                            if bs_match[0] != SENTINEL:
                                ab_pattern_flat_bs_match_checked_count[idx] += 1
                                if all(abs(bs_match[k] - ab_pattern_expected(k, tx)) < CORRECTNESS_TOL for k in range(STATE_COUNT)):
                                    ab_pattern_flat_bs_match_ok_count[idx] += 1
                            if as_swap[0] != SENTINEL:
                                ab_pattern_flat_as_swap_checked_count[idx] += 1
                                # swapped read = sAsFlat[tx*16+k] -> writer thread's own linear id = tx*16+k -> writer is (ty'=tx, tx'=k)
                                if all(abs(as_swap[k] - ab_pattern_expected(tx, k)) < CORRECTNESS_TOL for k in range(STATE_COUNT)):
                                    ab_pattern_flat_as_swap_ok_count[idx] += 1
                            if bs_swap[0] != SENTINEL:
                                ab_pattern_flat_bs_swap_checked_count[idx] += 1
                                # swapped read = sBsFlat[k*16+ty] -> writer thread's own linear id = k*16+ty -> writer is (ty'=k, tx'=ty)
                                if all(abs(bs_swap[k] - ab_pattern_expected(k, ty)) < CORRECTNESS_TOL for k in range(STATE_COUNT)):
                                    ab_pattern_flat_bs_swap_ok_count[idx] += 1
                if final_abcd_w0 and w == 0:
                    # TODO.md Phase 115: kernelsAll.cu's TINYGPU_DEBUG_
                    # DUMP_FINAL_ABCD_CSUB_W0 writes each of the 16 real
                    # threads' own REAL As[ty][0..3]/Bs[0..3][tx]/
                    # Ds[0..3]/Csub, captured right before the real
                    # barrier/C[] write. As checked against EVEC (the
                    # real A buffer), Bs against IVEC (the real B
                    # buffer), Ds against ds_reference_w0 (this thread's
                    # own view -- broadcast visibility is exactly what
                    # Phase 90-103/110 found could differ per-reader),
                    # Csub against reference_matrices[0] directly (JC69
                    # entries are always in [0,1], so the pre-clamp
                    # captured value should equal the reference exactly
                    # whenever the real computation is correct).
                    for ty in range(STATE_COUNT):
                        for tx in range(STATE_COUNT):
                            idx = ty * STATE_COUNT + tx
                            base = idx * (3 * STATE_COUNT + 1)
                            as_vals = final_abcd_region[base:base + STATE_COUNT]
                            bs_vals = final_abcd_region[base + STATE_COUNT:base + 2 * STATE_COUNT]
                            ds_vals = final_abcd_region[base + 2 * STATE_COUNT:base + 3 * STATE_COUNT]
                            csub_val = final_abcd_region[base + 3 * STATE_COUNT]
                            if as_vals[0] != SENTINEL:
                                final_abcd_as_checked_count[idx] += 1
                                if all(abs(as_vals[k] - EVEC[STATE_COUNT * ty + k]) < CORRECTNESS_TOL for k in range(STATE_COUNT)):
                                    final_abcd_as_ok_count[idx] += 1
                            if bs_vals[0] != SENTINEL:
                                final_abcd_bs_checked_count[idx] += 1
                                if all(abs(bs_vals[k] - IVEC[STATE_COUNT * k + tx]) < CORRECTNESS_TOL for k in range(STATE_COUNT)):
                                    final_abcd_bs_ok_count[idx] += 1
                            if ds_vals[0] != SENTINEL:
                                final_abcd_ds_checked_count[idx] += 1
                                if all(abs(ds_vals[k] - ds_reference_w0[k]) < CORRECTNESS_TOL for k in range(STATE_COUNT)):
                                    final_abcd_ds_ok_count[idx] += 1
                            if csub_val != SENTINEL:
                                final_abcd_csub_checked_count[idx] += 1
                                if abs(csub_val - reference_matrices[0][idx]) < CORRECTNESS_TOL:
                                    final_abcd_csub_ok_count[idx] += 1
                if combined_flat_w0 and w == 0:
                    # TODO.md Phase 116: kernelsAll.cu's TINYGPU_DEBUG_
                    # DUMP_COMBINED_FLAT_W0 writes each of the 16 logical
                    # (ty,tx) threads' own private As[ty][0..3]/
                    # Bs[0..3][tx]/Ds[0..3]/Csub-analog, all written/
                    # synced/read together under ONE shared barrier,
                    # exactly mirroring the real kernel's own combined
                    # structure. Checked against combined_val()/
                    # combined_ds()/combined_csub_expected().
                    for ty in range(STATE_COUNT):
                        for tx in range(STATE_COUNT):
                            idx = ty * STATE_COUNT + tx
                            base = idx * (3 * STATE_COUNT + 1)
                            as_vals = combined_flat_region[base:base + STATE_COUNT]
                            bs_vals = combined_flat_region[base + STATE_COUNT:base + 2 * STATE_COUNT]
                            ds_vals = combined_flat_region[base + 2 * STATE_COUNT:base + 3 * STATE_COUNT]
                            csub_val = combined_flat_region[base + 3 * STATE_COUNT]
                            if as_vals[0] != SENTINEL:
                                combined_flat_as_checked_count[idx] += 1
                                if all(abs(as_vals[k] - combined_val(ty, k)) < CORRECTNESS_TOL for k in range(STATE_COUNT)):
                                    combined_flat_as_ok_count[idx] += 1
                            if bs_vals[0] != SENTINEL:
                                combined_flat_bs_checked_count[idx] += 1
                                if all(abs(bs_vals[k] - combined_val(k, tx)) < CORRECTNESS_TOL for k in range(STATE_COUNT)):
                                    combined_flat_bs_ok_count[idx] += 1
                            if ds_vals[0] != SENTINEL:
                                combined_flat_ds_checked_count[idx] += 1
                                if all(abs(ds_vals[k] - combined_ds(k)) < CORRECTNESS_TOL for k in range(STATE_COUNT)):
                                    combined_flat_ds_ok_count[idx] += 1
                            if csub_val != SENTINEL:
                                combined_flat_csub_checked_count[idx] += 1
                                if abs(csub_val - combined_csub_expected(ty, tx)) < CORRECTNESS_TOL:
                                    combined_flat_csub_ok_count[idx] += 1
                if combined_ldg_flat_w0 and w == 0:
                    # TODO.md Phase 118: real-LDG-sourced combined probe --
                    # As against EVEC[ty*4+k], Bs against IVEC[k*4+tx]
                    # (same real references as --final-abcd-w0), Ds
                    # against the RAW EVAL[k] (no exp() yet), Csub against
                    # the raw (un-exponentiated) reduction.
                    for ty in range(STATE_COUNT):
                        for tx in range(STATE_COUNT):
                            idx = ty * STATE_COUNT + tx
                            base = idx * (3 * STATE_COUNT + 1)
                            as_vals = combined_ldg_flat_region[base:base + STATE_COUNT]
                            bs_vals = combined_ldg_flat_region[base + STATE_COUNT:base + 2 * STATE_COUNT]
                            ds_vals = combined_ldg_flat_region[base + 2 * STATE_COUNT:base + 3 * STATE_COUNT]
                            csub_val = combined_ldg_flat_region[base + 3 * STATE_COUNT]
                            if as_vals[0] != SENTINEL:
                                combined_ldg_flat_as_checked_count[idx] += 1
                                if all(abs(as_vals[k] - EVEC[STATE_COUNT * ty + k]) < CORRECTNESS_TOL for k in range(STATE_COUNT)):
                                    combined_ldg_flat_as_ok_count[idx] += 1
                            if bs_vals[0] != SENTINEL:
                                combined_ldg_flat_bs_checked_count[idx] += 1
                                if all(abs(bs_vals[k] - IVEC[STATE_COUNT * k + tx]) < CORRECTNESS_TOL for k in range(STATE_COUNT)):
                                    combined_ldg_flat_bs_ok_count[idx] += 1
                            if ds_vals[0] != SENTINEL:
                                combined_ldg_flat_ds_checked_count[idx] += 1
                                if all(abs(ds_vals[k] - EVAL[k]) < CORRECTNESS_TOL for k in range(STATE_COUNT)):
                                    combined_ldg_flat_ds_ok_count[idx] += 1
                            if csub_val != SENTINEL:
                                combined_ldg_flat_csub_checked_count[idx] += 1
                                csub_expected = sum(EVEC[STATE_COUNT * ty + k] * EVAL[k] * IVEC[STATE_COUNT * k + tx] for k in range(STATE_COUNT))
                                if abs(csub_val - csub_expected) < CORRECTNESS_TOL:
                                    combined_ldg_flat_csub_ok_count[idx] += 1
                if combined_exp_flat_w0 and w == 0:
                    # TODO.md Phase 118: real-exp()/distance combined
                    # probe -- As/Bs same real references, Ds against
                    # ds_reference_w0[k] (the real exp(EVAL[k]*distance)),
                    # Csub against reference_matrices[0] directly (with
                    # real A/B/D and real exp()/distance, this reduction
                    # is mathematically identical to the real kernel's own
                    # intended Csub).
                    for ty in range(STATE_COUNT):
                        for tx in range(STATE_COUNT):
                            idx = ty * STATE_COUNT + tx
                            base = idx * (3 * STATE_COUNT + 1)
                            as_vals = combined_exp_flat_region[base:base + STATE_COUNT]
                            bs_vals = combined_exp_flat_region[base + STATE_COUNT:base + 2 * STATE_COUNT]
                            ds_vals = combined_exp_flat_region[base + 2 * STATE_COUNT:base + 3 * STATE_COUNT]
                            csub_val = combined_exp_flat_region[base + 3 * STATE_COUNT]
                            if as_vals[0] != SENTINEL:
                                combined_exp_flat_as_checked_count[idx] += 1
                                if all(abs(as_vals[k] - EVEC[STATE_COUNT * ty + k]) < CORRECTNESS_TOL for k in range(STATE_COUNT)):
                                    combined_exp_flat_as_ok_count[idx] += 1
                            if bs_vals[0] != SENTINEL:
                                combined_exp_flat_bs_checked_count[idx] += 1
                                if all(abs(bs_vals[k] - IVEC[STATE_COUNT * k + tx]) < CORRECTNESS_TOL for k in range(STATE_COUNT)):
                                    combined_exp_flat_bs_ok_count[idx] += 1
                            if ds_vals[0] != SENTINEL:
                                combined_exp_flat_ds_checked_count[idx] += 1
                                if all(abs(ds_vals[k] - ds_reference_w0[k]) < CORRECTNESS_TOL for k in range(STATE_COUNT)):
                                    combined_exp_flat_ds_ok_count[idx] += 1
                            if csub_val != SENTINEL:
                                combined_exp_flat_csub_checked_count[idx] += 1
                                if abs(csub_val - reference_matrices[0][idx]) < CORRECTNESS_TOL:
                                    combined_exp_flat_csub_ok_count[idx] += 1
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
                if per_thread_ds or (per_thread_ds_w0 and w0 == 0):
                    # TODO.md Phase 93/96: the direct test -- every one of
                    # the 16 real threads' own observed Ds[0..3], read
                    # straight out of shared memory by that thread
                    # itself, not inferred from the real C[] output.
                    pt0 = (per_thread_region[w0 * per_thread_ds_size:(w0 + 1) * per_thread_ds_size] if per_thread_ds
                           else per_thread_region[0:per_thread_ds_size])
                    sub_header = f"  --- per-thread Ds[0..3] view, wMatrix {w0} (writer row is ty=0) ---"
                    log(sub_header)
                    print(sub_header, file=sys.stdout, flush=True)
                    for ty in range(STATE_COUNT):
                        for tx in range(STATE_COUNT):
                            idx = (ty * STATE_COUNT + tx) * STATE_COUNT
                            view = pt0[idx:idx + STATE_COUNT]
                            desc = "(never wrote)" if view[0] == SENTINEL else f"{[f'{v:.6f}' for v in view]}"
                            line = f"    (ty={ty},tx={tx}): Ds={desc}"
                            log(line)
                            print(line, file=sys.stdout, flush=True)
                if per_thread_ds_min_w0 and w0 == 0:
                    # TODO.md Phase 103: raw real-vs-reference view for
                    # --per-thread-ds-min-w0, matching Phase 93/96's own
                    # raw diagnostic style above -- shows, per thread,
                    # whether it read the correct Ds[tx] straight out of
                    # shared memory, not just the aggregate rate.
                    sub_header = f"  --- per-thread Ds[tx] readback, wMatrix {w0} (writer row is ty=0) ---"
                    log(sub_header)
                    print(sub_header, file=sys.stdout, flush=True)
                    for ty in range(STATE_COUNT):
                        for tx in range(STATE_COUNT):
                            idx = ty * STATE_COUNT + tx
                            val = per_thread_ds_min_region[idx]
                            if val == SENTINEL:
                                desc = "(never wrote)"
                            else:
                                ref = ds_reference_w0[tx]
                                ok = abs(val - ref) < CORRECTNESS_TOL
                                desc = f"real={val:.6f} ref={ref:.6f} {'OK' if ok else 'WRONG'}"
                            line = f"    (ty={ty},tx={tx}): {desc}"
                            log(line)
                            print(line, file=sys.stdout, flush=True)
                if local_mem_w0 and w0 == 0:
                    # TODO.md Phase 106: raw real-vs-reference view for
                    # --local-mem-w0, matching the Ds[]-probes' own raw
                    # diagnostic style -- shows, per thread, whether its
                    # own spilled-and-reloaded local-memory value survived
                    # the roundtrip correctly.
                    sub_header = f"  --- per-thread local-memory (spill) readback, wMatrix {w0} ---"
                    log(sub_header)
                    print(sub_header, file=sys.stdout, flush=True)
                    for ty in range(STATE_COUNT):
                        for tx in range(STATE_COUNT):
                            idx = ty * STATE_COUNT + tx
                            val = local_mem_region[idx]
                            if val == SENTINEL:
                                desc = "(never wrote)"
                            else:
                                expected = local_mem_expected(ty, tx, n_blocks)
                                ok = abs(val - expected) < CORRECTNESS_TOL
                                desc = f"real={val:.6f} expected={expected} {'OK' if ok else 'WRONG'}"
                            line = f"    (ty={ty},tx={tx}): {desc}"
                            log(line)
                            print(line, file=sys.stdout, flush=True)
                if shared_spill_w0 and w0 == 0:
                    # TODO.md Phase 108: raw real-vs-reference view for
                    # --shared-spill-w0, matching --local-mem-w0's own
                    # raw diagnostic style.
                    sub_header = f"  --- per-thread explicit-shared-memory (spill) readback, wMatrix {w0} ---"
                    log(sub_header)
                    print(sub_header, file=sys.stdout, flush=True)
                    for ty in range(STATE_COUNT):
                        for tx in range(STATE_COUNT):
                            idx = ty * STATE_COUNT + tx
                            val = shared_spill_region[idx]
                            if val == SENTINEL:
                                desc = "(never wrote)"
                            else:
                                expected = local_mem_expected(ty, tx, n_blocks)
                                ok = abs(val - expected) < CORRECTNESS_TOL
                                desc = f"real={val:.6f} expected={expected} {'OK' if ok else 'WRONG'}"
                            line = f"    (ty={ty},tx={tx}): {desc}"
                            log(line)
                            print(line, file=sys.stdout, flush=True)
            if local_mem_flat_w0 and it == 0:
                # TODO.md Phase 109: raw real-vs-reference view for
                # --local-mem-flat-w0, matching --local-mem-w0's own raw
                # diagnostic style. Deliberately NOT nested inside "if
                # it==0 and populated_this_iter" above -- Phase 108's own
                # run showed wMatrix 0's *real matrix* output can fail to
                # populate on a given run (a first in this investigation,
                # but real), which would silently skip this dump too if
                # it depended on that same condition. This block checks
                # `it==0` alone, independent of the real matrix's status,
                # so the raw per-thread comparison is never silently lost.
                sub_header = "  --- per-thread local-memory (spill) readback, FLAT (256,1,1) dispatch, wMatrix 0 ---"
                log(sub_header)
                print(sub_header, file=sys.stdout, flush=True)
                for ty in range(STATE_COUNT):
                    for tx in range(STATE_COUNT):
                        idx = ty * STATE_COUNT + tx
                        val = local_mem_flat_region[idx]
                        if val == SENTINEL:
                            desc = "(never wrote)"
                        else:
                            expected = local_mem_expected(ty, tx, n_blocks)
                            ok = abs(val - expected) < CORRECTNESS_TOL
                            desc = f"real={val:.6f} expected={expected} {'OK' if ok else 'WRONG'}"
                        line = f"    (ty={ty},tx={tx}): {desc}"
                        log(line)
                        print(line, file=sys.stdout, flush=True)
            if shared_broadcast_flat_w0 and it == 0:
                # TODO.md Phase 110: raw real-vs-reference view for
                # --shared-broadcast-flat-w0, matching --local-mem-flat-
                # w0's own independent-of-populated_this_iter style.
                sub_header = "  --- per-thread Ds[]-broadcast (sDs) readback, FLAT (256,1,1) dispatch, wMatrix 0 ---"
                log(sub_header)
                print(sub_header, file=sys.stdout, flush=True)
                for ty in range(STATE_COUNT):
                    for tx in range(STATE_COUNT):
                        idx = ty * STATE_COUNT + tx
                        val = shared_broadcast_flat_region[idx]
                        if val == SENTINEL:
                            desc = "(never wrote)"
                        else:
                            ref = ds_reference_w0[tx]
                            ok = abs(val - ref) < CORRECTNESS_TOL
                            desc = f"real={val:.6f} ref={ref:.6f} {'OK' if ok else 'WRONG'}"
                        line = f"    (ty={ty},tx={tx}): {desc}"
                        log(line)
                        print(line, file=sys.stdout, flush=True)
            if ab_pattern_2d_w0 and it == 0:
                # TODO.md Phase 113: raw real-vs-reference view for
                # --ab-pattern-2d-w0, matching --local-mem-flat-w0's own
                # independent-of-populated_this_iter style.
                sub_header = "  --- per-thread As[ty][k]/Bs[k][tx] readback, 2D shared array, FLAT (256,1,1) dispatch, wMatrix 0 ---"
                log(sub_header)
                print(sub_header, file=sys.stdout, flush=True)
                for ty in range(STATE_COUNT):
                    for tx in range(STATE_COUNT):
                        idx = ty * STATE_COUNT + tx
                        base = idx * 2 * STATE_COUNT
                        as_vals = ab_pattern_2d_region[base:base + STATE_COUNT]
                        bs_vals = ab_pattern_2d_region[base + STATE_COUNT:base + 2 * STATE_COUNT]
                        if as_vals[0] == SENTINEL:
                            desc = "(never wrote)"
                        else:
                            as_desc = ", ".join(
                                f"k{k}={as_vals[k]:.0f}({'OK' if abs(as_vals[k]-ab_pattern_expected(ty,k))<CORRECTNESS_TOL else 'WRONG,exp='+str(ab_pattern_expected(ty,k))})"
                                for k in range(STATE_COUNT))
                            bs_desc = ", ".join(
                                f"k{k}={bs_vals[k]:.0f}({'OK' if abs(bs_vals[k]-ab_pattern_expected(k,tx))<CORRECTNESS_TOL else 'WRONG,exp='+str(ab_pattern_expected(k,tx))})"
                                for k in range(STATE_COUNT))
                            desc = f"As[{ty}][k]: {as_desc}  |  Bs[k][{tx}]: {bs_desc}"
                        line = f"    (ty={ty},tx={tx}): {desc}"
                        log(line)
                        print(line, file=sys.stdout, flush=True)
            if ab_pattern_flat_w0 and it == 0:
                # TODO.md Phase 113: raw real-vs-reference view for
                # --ab-pattern-flat-w0 -- flat REAL[256] array, hand-
                # computed addresses, both the matching and user-suggested
                # swapped (tx*16+k) read variants.
                sub_header = "  --- per-thread flat-array As/Bs readback (matching + swapped), FLAT (256,1,1) dispatch, wMatrix 0 ---"
                log(sub_header)
                print(sub_header, file=sys.stdout, flush=True)
                for ty in range(STATE_COUNT):
                    for tx in range(STATE_COUNT):
                        idx = ty * STATE_COUNT + tx
                        base = idx * 4 * STATE_COUNT
                        as_match = ab_pattern_flat_region[base:base + STATE_COUNT]
                        bs_match = ab_pattern_flat_region[base + STATE_COUNT:base + 2 * STATE_COUNT]
                        as_swap = ab_pattern_flat_region[base + 2 * STATE_COUNT:base + 3 * STATE_COUNT]
                        bs_swap = ab_pattern_flat_region[base + 3 * STATE_COUNT:base + 4 * STATE_COUNT]
                        if as_match[0] == SENTINEL:
                            desc = "(never wrote)"
                        else:
                            def _fmt(vals, exp_fn):
                                return ", ".join(
                                    f"k{k}={vals[k]:.0f}({'OK' if abs(vals[k]-exp_fn(k))<CORRECTNESS_TOL else 'WRONG,exp='+str(exp_fn(k))})"
                                    for k in range(STATE_COUNT))
                            as_m = _fmt(as_match, lambda k: ab_pattern_expected(ty, k))
                            bs_m = _fmt(bs_match, lambda k: ab_pattern_expected(k, tx))
                            as_s = _fmt(as_swap, lambda k: ab_pattern_expected(tx, k))
                            bs_s = _fmt(bs_swap, lambda k: ab_pattern_expected(k, ty))
                            desc = f"As-match: {as_m}  |  Bs-match: {bs_m}  |  As-swap(tx*16+k): {as_s}  |  Bs-swap(k*16+ty): {bs_s}"
                        line = f"    (ty={ty},tx={tx}): {desc}"
                        log(line)
                        print(line, file=sys.stdout, flush=True)
            if final_abcd_w0 and it == 0:
                # TODO.md Phase 115: raw real-vs-reference view for
                # --final-abcd-w0, matching --local-mem-flat-w0's own
                # independent-of-populated_this_iter style.
                sub_header = "  --- REAL As[ty][k]/Bs[k][tx]/Ds[k]/Csub, captured before the real barrier/C[] write, FLAT (256,1,1) dispatch, wMatrix 0 ---"
                log(sub_header)
                print(sub_header, file=sys.stdout, flush=True)
                for ty in range(STATE_COUNT):
                    for tx in range(STATE_COUNT):
                        idx = ty * STATE_COUNT + tx
                        base = idx * (3 * STATE_COUNT + 1)
                        as_vals = final_abcd_region[base:base + STATE_COUNT]
                        bs_vals = final_abcd_region[base + STATE_COUNT:base + 2 * STATE_COUNT]
                        ds_vals = final_abcd_region[base + 2 * STATE_COUNT:base + 3 * STATE_COUNT]
                        csub_val = final_abcd_region[base + 3 * STATE_COUNT]
                        if as_vals[0] == SENTINEL:
                            desc = "(never wrote)"
                        else:
                            def _fmt2(vals, exp_fn):
                                return ", ".join(
                                    f"k{k}={vals[k]:.6f}({'OK' if abs(vals[k]-exp_fn(k))<CORRECTNESS_TOL else 'WRONG,exp='+f'{exp_fn(k):.6f}'})"
                                    for k in range(STATE_COUNT))
                            as_d = _fmt2(as_vals, lambda k: EVEC[STATE_COUNT * ty + k])
                            bs_d = _fmt2(bs_vals, lambda k: IVEC[STATE_COUNT * k + tx])
                            ds_d = _fmt2(ds_vals, lambda k: ds_reference_w0[k])
                            csub_ref = reference_matrices[0][idx]
                            csub_ok = abs(csub_val - csub_ref) < CORRECTNESS_TOL
                            desc = (f"As[{ty}][k]: {as_d}  |  Bs[k][{tx}]: {bs_d}  |  Ds[k] (this thread's view): {ds_d}  |  "
                                    f"Csub={csub_val:.6f} ref={csub_ref:.6f} {'OK' if csub_ok else 'WRONG'}")
                        line = f"    (ty={ty},tx={tx}): {desc}"
                        log(line)
                        print(line, file=sys.stdout, flush=True)
            if combined_flat_w0 and it == 0:
                # TODO.md Phase 116: raw real-vs-reference view for
                # --combined-flat-w0, matching --final-abcd-w0's own
                # independent-of-populated_this_iter style.
                sub_header = "  --- private combined As[ty][k]/Bs[k][tx]/Ds[k]/Csub-analog, single shared barrier, FLAT (256,1,1) dispatch, wMatrix 0 ---"
                log(sub_header)
                print(sub_header, file=sys.stdout, flush=True)
                for ty in range(STATE_COUNT):
                    for tx in range(STATE_COUNT):
                        idx = ty * STATE_COUNT + tx
                        base = idx * (3 * STATE_COUNT + 1)
                        as_vals = combined_flat_region[base:base + STATE_COUNT]
                        bs_vals = combined_flat_region[base + STATE_COUNT:base + 2 * STATE_COUNT]
                        ds_vals = combined_flat_region[base + 2 * STATE_COUNT:base + 3 * STATE_COUNT]
                        csub_val = combined_flat_region[base + 3 * STATE_COUNT]
                        if as_vals[0] == SENTINEL:
                            desc = "(never wrote)"
                        else:
                            def _fmt3(vals, exp_fn):
                                return ", ".join(
                                    f"k{k}={vals[k]:.0f}({'OK' if abs(vals[k]-exp_fn(k))<CORRECTNESS_TOL else 'WRONG,exp='+str(exp_fn(k))})"
                                    for k in range(STATE_COUNT))
                            as_d = _fmt3(as_vals, lambda k: combined_val(ty, k))
                            bs_d = _fmt3(bs_vals, lambda k: combined_val(k, tx))
                            ds_d = _fmt3(ds_vals, lambda k: combined_ds(k))
                            csub_ref = combined_csub_expected(ty, tx)
                            csub_ok = abs(csub_val - csub_ref) < CORRECTNESS_TOL
                            desc = (f"As[{ty}][k]: {as_d}  |  Bs[k][{tx}]: {bs_d}  |  Ds[k]: {ds_d}  |  "
                                    f"Csub={csub_val:.0f} exp={csub_ref} {'OK' if csub_ok else 'WRONG'}")
                        line = f"    (ty={ty},tx={tx}): {desc}"
                        log(line)
                        print(line, file=sys.stdout, flush=True)
            if combined_ldg_flat_w0 and it == 0:
                # TODO.md Phase 118: raw real-vs-reference view for
                # --combined-ldg-flat-w0.
                sub_header = "  --- real-LDG-sourced combined As[ty][k]/Bs[k][tx]/Ds[k](raw)/Csub-analog, single shared barrier, FLAT (256,1,1) dispatch, wMatrix 0 ---"
                log(sub_header)
                print(sub_header, file=sys.stdout, flush=True)
                for ty in range(STATE_COUNT):
                    for tx in range(STATE_COUNT):
                        idx = ty * STATE_COUNT + tx
                        base = idx * (3 * STATE_COUNT + 1)
                        as_vals = combined_ldg_flat_region[base:base + STATE_COUNT]
                        bs_vals = combined_ldg_flat_region[base + STATE_COUNT:base + 2 * STATE_COUNT]
                        ds_vals = combined_ldg_flat_region[base + 2 * STATE_COUNT:base + 3 * STATE_COUNT]
                        csub_val = combined_ldg_flat_region[base + 3 * STATE_COUNT]
                        if as_vals[0] == SENTINEL:
                            desc = "(never wrote)"
                        else:
                            def _fmt4(vals, exp_fn):
                                return ", ".join(
                                    f"k{k}={vals[k]:.6f}({'OK' if abs(vals[k]-exp_fn(k))<CORRECTNESS_TOL else 'WRONG,exp='+f'{exp_fn(k):.6f}'})"
                                    for k in range(STATE_COUNT))
                            as_d = _fmt4(as_vals, lambda k: EVEC[STATE_COUNT * ty + k])
                            bs_d = _fmt4(bs_vals, lambda k: IVEC[STATE_COUNT * k + tx])
                            ds_d = _fmt4(ds_vals, lambda k: EVAL[k])
                            csub_ref = sum(EVEC[STATE_COUNT * ty + k] * EVAL[k] * IVEC[STATE_COUNT * k + tx] for k in range(STATE_COUNT))
                            csub_ok = abs(csub_val - csub_ref) < CORRECTNESS_TOL
                            desc = (f"As[{ty}][k]: {as_d}  |  Bs[k][{tx}]: {bs_d}  |  Ds[k] (raw): {ds_d}  |  "
                                    f"Csub={csub_val:.6f} ref={csub_ref:.6f} {'OK' if csub_ok else 'WRONG'}")
                        line = f"    (ty={ty},tx={tx}): {desc}"
                        log(line)
                        print(line, file=sys.stdout, flush=True)
            if combined_exp_flat_w0 and it == 0:
                # TODO.md Phase 118: raw real-vs-reference view for
                # --combined-exp-flat-w0.
                sub_header = "  --- real-exp()/distance combined As[ty][k]/Bs[k][tx]/Ds[k]/Csub-analog, single shared barrier, FLAT (256,1,1) dispatch, wMatrix 0 ---"
                log(sub_header)
                print(sub_header, file=sys.stdout, flush=True)
                for ty in range(STATE_COUNT):
                    for tx in range(STATE_COUNT):
                        idx = ty * STATE_COUNT + tx
                        base = idx * (3 * STATE_COUNT + 1)
                        as_vals = combined_exp_flat_region[base:base + STATE_COUNT]
                        bs_vals = combined_exp_flat_region[base + STATE_COUNT:base + 2 * STATE_COUNT]
                        ds_vals = combined_exp_flat_region[base + 2 * STATE_COUNT:base + 3 * STATE_COUNT]
                        csub_val = combined_exp_flat_region[base + 3 * STATE_COUNT]
                        if as_vals[0] == SENTINEL:
                            desc = "(never wrote)"
                        else:
                            def _fmt5(vals, exp_fn):
                                return ", ".join(
                                    f"k{k}={vals[k]:.6f}({'OK' if abs(vals[k]-exp_fn(k))<CORRECTNESS_TOL else 'WRONG,exp='+f'{exp_fn(k):.6f}'})"
                                    for k in range(STATE_COUNT))
                            as_d = _fmt5(as_vals, lambda k: EVEC[STATE_COUNT * ty + k])
                            bs_d = _fmt5(bs_vals, lambda k: IVEC[STATE_COUNT * k + tx])
                            ds_d = _fmt5(ds_vals, lambda k: ds_reference_w0[k])
                            csub_ref = reference_matrices[0][idx]
                            csub_ok = abs(csub_val - csub_ref) < CORRECTNESS_TOL
                            desc = (f"As[{ty}][k]: {as_d}  |  Bs[k][{tx}]: {bs_d}  |  Ds[k]: {ds_d}  |  "
                                    f"Csub={csub_val:.6f} ref={csub_ref:.6f} {'OK' if csub_ok else 'WRONG'}")
                        line = f"    (ty={ty},tx={tx}): {desc}"
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

        if per_thread_ds or per_thread_ds_w0:
            print(f"\n=== nv_real_kernel_probe --sweep {sweep}: per-wMatrix Ds[] broadcast-visibility rate ===", file=sys.stdout, flush=True)
            for w in range(n_blocks):
                checked, ok = ds_broadcast_checked_count[w], ds_broadcast_ok_count[w]
                rate = f"{100*ok/checked:5.1f}%" if checked else "  n/a"
                line = f"  wMatrix {w:2d}: {ok:2d}/{checked:2d} broadcast OK ({rate})"
                log(line)
                print(line, file=sys.stdout, flush=True)

        if dummy_third_block:
            print(f"\n=== nv_real_kernel_probe --sweep {sweep}: per-wMatrix dummy-third-block write reliability ===", file=sys.stdout, flush=True)
            for w in range(n_blocks):
                checked, ok = dummy_checked_count[w], dummy_ok_count[w]
                rate = f"{100*ok/checked:5.1f}%" if checked else "  n/a"
                line = f"  wMatrix {w:2d}: {ok:2d}/{checked:2d} correct constant ({rate})"
                log(line)
                print(line, file=sys.stdout, flush=True)

        if per_thread_dummy_w0:
            print(f"\n=== nv_real_kernel_probe --sweep {sweep}: wMatrix 0 per-thread-dummy write reliability (16 slots/iteration) ===", file=sys.stdout, flush=True)
            checked, ok = per_thread_dummy_checked_count[0], per_thread_dummy_ok_count[0]
            rate = f"{100*ok/checked:5.1f}%" if checked else "  n/a"
            line = f"  wMatrix  0: {ok:3d}/{checked:3d} correct per-thread slots ({rate})"
            log(line)
            print(line, file=sys.stdout, flush=True)

        if per_thread_ds_min_w0:
            # TODO.md Phase 103: Phase 102's aggregate 7/16 result directly
            # confirmed the Ds[] broadcast-visibility hypothesis for the
            # first time without inferring it from downstream C[] output,
            # but gave no indication of *which* threads are reliably
            # right/wrong. Per-slot breakdown, across the full --sweep,
            # tests directly whether ty=0 (the 4 threads that *write*
            # Ds[]) is consistently correct while ty>0 (readers) are
            # consistently/inconsistently wrong -- the same question
            # Phase 91/92 could only answer indirectly, now answerable at
            # the Ds[] source with real statistical power and zero fault
            # risk (Phase 100-102 already cleared this diagnostic on
            # hardware).
            print(f"\n=== nv_real_kernel_probe --sweep {sweep}: wMatrix 0 per-thread Ds[tx] readback reliability, per slot ===", file=sys.stdout, flush=True)
            total_checked = total_ok = 0
            for ty in range(STATE_COUNT):
                for tx in range(STATE_COUNT):
                    idx = ty * STATE_COUNT + tx
                    checked, ok = per_thread_ds_min_checked_count[idx], per_thread_ds_min_ok_count[idx]
                    total_checked += checked
                    total_ok += ok
                    rate = f"{100*ok/checked:5.1f}%" if checked else "  n/a"
                    writer = " (writer row)" if ty == 0 else ""
                    line = f"  (ty={ty},tx={tx}): {ok:3d}/{checked:3d} correct ({rate}){writer}"
                    log(line)
                    print(line, file=sys.stdout, flush=True)
            rate = f"{100*total_ok/total_checked:5.1f}%" if total_checked else "  n/a"
            line = f"  TOTAL: {total_ok:3d}/{total_checked:3d} correct ({rate})"
            log(line)
            print(line, file=sys.stdout, flush=True)

        if local_mem_w0:
            # TODO.md Phase 106: parallel to --per-thread-ds-min-w0's own
            # per-slot table above, but for LOCAL memory (register spill)
            # instead of shared memory. Local memory is per-thread-
            # private -- no broadcast semantics -- so "correct" here means
            # each thread's own spilled-and-reloaded value survived the
            # roundtrip, not that it matches some other thread's write.
            print(f"\n=== nv_real_kernel_probe --sweep {sweep}: wMatrix 0 per-thread local-memory (spill) readback reliability, per slot ===", file=sys.stdout, flush=True)
            total_checked = total_ok = 0
            for ty in range(STATE_COUNT):
                for tx in range(STATE_COUNT):
                    idx = ty * STATE_COUNT + tx
                    checked, ok = local_mem_checked_count[idx], local_mem_ok_count[idx]
                    total_checked += checked
                    total_ok += ok
                    rate = f"{100*ok/checked:5.1f}%" if checked else "  n/a"
                    line = f"  (ty={ty},tx={tx}): {ok:3d}/{checked:3d} correct ({rate})"
                    log(line)
                    print(line, file=sys.stdout, flush=True)
            rate = f"{100*total_ok/total_checked:5.1f}%" if total_checked else "  n/a"
            line = f"  TOTAL: {total_ok:3d}/{total_checked:3d} correct ({rate})"
            log(line)
            print(line, file=sys.stdout, flush=True)

        if shared_spill_w0:
            # TODO.md Phase 108: parallel to --local-mem-w0's own per-slot
            # table, but for explicit shared-memory storage. Same-thread
            # write-then-read-back, never tested before this probe.
            print(f"\n=== nv_real_kernel_probe --sweep {sweep}: wMatrix 0 per-thread explicit-shared-memory (spill) readback reliability, per slot ===", file=sys.stdout, flush=True)
            total_checked = total_ok = 0
            for ty in range(STATE_COUNT):
                for tx in range(STATE_COUNT):
                    idx = ty * STATE_COUNT + tx
                    checked, ok = shared_spill_checked_count[idx], shared_spill_ok_count[idx]
                    total_checked += checked
                    total_ok += ok
                    rate = f"{100*ok/checked:5.1f}%" if checked else "  n/a"
                    line = f"  (ty={ty},tx={tx}): {ok:3d}/{checked:3d} correct ({rate})"
                    log(line)
                    print(line, file=sys.stdout, flush=True)
            rate = f"{100*total_ok/total_checked:5.1f}%" if total_checked else "  n/a"
            line = f"  TOTAL: {total_ok:3d}/{total_checked:3d} correct ({rate})"
            log(line)
            print(line, file=sys.stdout, flush=True)

        if local_mem_flat_w0:
            # TODO.md Phase 109: parallel to --local-mem-w0's own per-slot
            # table, but under a flat (256,1,1) block dispatch instead of
            # the kernel's normal (16,16,1) -- tests whether the tx+4ty
            # aliasing depends on the real 2D block shape.
            print(f"\n=== nv_real_kernel_probe --sweep {sweep}: wMatrix 0 per-thread local-memory (spill) readback reliability, FLAT (256,1,1) dispatch, per slot ===", file=sys.stdout, flush=True)
            total_checked = total_ok = 0
            for ty in range(STATE_COUNT):
                for tx in range(STATE_COUNT):
                    idx = ty * STATE_COUNT + tx
                    checked, ok = local_mem_flat_checked_count[idx], local_mem_flat_ok_count[idx]
                    total_checked += checked
                    total_ok += ok
                    rate = f"{100*ok/checked:5.1f}%" if checked else "  n/a"
                    line = f"  (ty={ty},tx={tx}): {ok:3d}/{checked:3d} correct ({rate})"
                    log(line)
                    print(line, file=sys.stdout, flush=True)
            rate = f"{100*total_ok/total_checked:5.1f}%" if total_checked else "  n/a"
            line = f"  TOTAL: {total_ok:3d}/{total_checked:3d} correct ({rate})"
            log(line)
            print(line, file=sys.stdout, flush=True)

        if shared_broadcast_flat_w0:
            # TODO.md Phase 110: parallel to --local-mem-flat-w0's own
            # per-slot table, but for Ds[]'s own broadcast mechanism.
            print(f"\n=== nv_real_kernel_probe --sweep {sweep}: wMatrix 0 per-thread Ds[]-broadcast readback reliability, FLAT (256,1,1) dispatch, per slot ===", file=sys.stdout, flush=True)
            total_checked = total_ok = 0
            for ty in range(STATE_COUNT):
                for tx in range(STATE_COUNT):
                    idx = ty * STATE_COUNT + tx
                    checked, ok = shared_broadcast_flat_checked_count[idx], shared_broadcast_flat_ok_count[idx]
                    total_checked += checked
                    total_ok += ok
                    rate = f"{100*ok/checked:5.1f}%" if checked else "  n/a"
                    line = f"  (ty={ty},tx={tx}): {ok:3d}/{checked:3d} correct ({rate})"
                    log(line)
                    print(line, file=sys.stdout, flush=True)
            rate = f"{100*total_ok/total_checked:5.1f}%" if total_checked else "  n/a"
            line = f"  TOTAL: {total_ok:3d}/{total_checked:3d} correct ({rate})"
            log(line)
            print(line, file=sys.stdout, flush=True)

        def _print_per_slot_table(title, ok_count, checked_count):
            # TODO.md Phase 113: shared per-slot table printer, used by
            # --ab-pattern-2d-w0/--ab-pattern-flat-w0's four/six tables --
            # identical row/format to every other per-slot table above,
            # factored out only because this probe has more of them.
            print(f"\n=== nv_real_kernel_probe --sweep {sweep}: {title} ===", file=sys.stdout, flush=True)
            total_checked = total_ok = 0
            for ty in range(STATE_COUNT):
                for tx in range(STATE_COUNT):
                    idx = ty * STATE_COUNT + tx
                    checked, ok = checked_count[idx], ok_count[idx]
                    total_checked += checked
                    total_ok += ok
                    rate = f"{100*ok/checked:5.1f}%" if checked else "  n/a"
                    line = f"  (ty={ty},tx={tx}): {ok:3d}/{checked:3d} correct ({rate})"
                    log(line)
                    print(line, file=sys.stdout, flush=True)
            rate = f"{100*total_ok/total_checked:5.1f}%" if total_checked else "  n/a"
            line = f"  TOTAL: {total_ok:3d}/{total_checked:3d} correct ({rate})"
            log(line)
            print(line, file=sys.stdout, flush=True)

        if ab_pattern_2d_w0:
            # TODO.md Phase 113: direct ground truth of the real kernel's
            # own As[ty][k] (row-wise)/Bs[k][tx] (column-wise) access
            # pattern, under a flat (256,1,1) dispatch, using the same 2D
            # REAL[16][16] shared-array shape the real kernel uses.
            _print_per_slot_table("wMatrix 0 per-thread As[ty][k] readback reliability, 2D shared array, FLAT dispatch, per slot",
                                   ab_pattern_2d_as_ok_count, ab_pattern_2d_as_checked_count)
            _print_per_slot_table("wMatrix 0 per-thread Bs[k][tx] readback reliability, 2D shared array, FLAT dispatch, per slot",
                                   ab_pattern_2d_bs_ok_count, ab_pattern_2d_bs_checked_count)

        if ab_pattern_flat_w0:
            # TODO.md Phase 113: same ground truth, flat REAL[256] array
            # with hand-computed addresses -- match_* mirrors ab-pattern-
            # 2d-w0's own read exactly (tests whether 2D-array codegen
            # itself matters); swap_* is the user-suggested tx*16+k
            # alternate indexing (a genuinely different set of slots).
            _print_per_slot_table("wMatrix 0 per-thread As[ty][k]-matching readback reliability, flat array, FLAT dispatch, per slot",
                                   ab_pattern_flat_as_match_ok_count, ab_pattern_flat_as_match_checked_count)
            _print_per_slot_table("wMatrix 0 per-thread Bs[k][tx]-matching readback reliability, flat array, FLAT dispatch, per slot",
                                   ab_pattern_flat_bs_match_ok_count, ab_pattern_flat_bs_match_checked_count)
            _print_per_slot_table("wMatrix 0 per-thread As-swapped (tx*16+k) readback reliability, flat array, FLAT dispatch, per slot",
                                   ab_pattern_flat_as_swap_ok_count, ab_pattern_flat_as_swap_checked_count)
            _print_per_slot_table("wMatrix 0 per-thread Bs-swapped (k*16+ty) readback reliability, flat array, FLAT dispatch, per slot",
                                   ab_pattern_flat_bs_swap_ok_count, ab_pattern_flat_bs_swap_checked_count)

        if final_abcd_w0:
            # TODO.md Phase 115: direct ground truth of the REAL As/Bs/Ds/
            # Csub, captured right before the real barrier/C[] write in
            # Phase 111's real, combined, non-early-return kernel path.
            _print_per_slot_table("wMatrix 0 per-thread REAL As[ty][k] reliability (captured pre-final-write), per slot",
                                   final_abcd_as_ok_count, final_abcd_as_checked_count)
            _print_per_slot_table("wMatrix 0 per-thread REAL Bs[k][tx] reliability (captured pre-final-write), per slot",
                                   final_abcd_bs_ok_count, final_abcd_bs_checked_count)
            _print_per_slot_table("wMatrix 0 per-thread REAL Ds[k] (this thread's own view) reliability (captured pre-final-write), per slot",
                                   final_abcd_ds_ok_count, final_abcd_ds_checked_count)
            _print_per_slot_table("wMatrix 0 per-thread REAL Csub reliability (captured pre-final-write), per slot",
                                   final_abcd_csub_ok_count, final_abcd_csub_checked_count)

        if combined_flat_w0:
            # TODO.md Phase 116: direct ground truth of Ds[]/As[]/Bs[]
            # tested TOGETHER under ONE shared barrier, exactly mirroring
            # the real kernel's own combined write/barrier/read structure
            # -- private storage, early return, same safe pattern as
            # every flat-dispatch probe since Phase 109.
            _print_per_slot_table("wMatrix 0 per-thread combined-probe As[ty][k] reliability, per slot",
                                   combined_flat_as_ok_count, combined_flat_as_checked_count)
            _print_per_slot_table("wMatrix 0 per-thread combined-probe Bs[k][tx] reliability, per slot",
                                   combined_flat_bs_ok_count, combined_flat_bs_checked_count)
            _print_per_slot_table("wMatrix 0 per-thread combined-probe Ds[k] reliability, per slot",
                                   combined_flat_ds_ok_count, combined_flat_ds_checked_count)
            _print_per_slot_table("wMatrix 0 per-thread combined-probe Csub-analog reliability, per slot",
                                   combined_flat_csub_ok_count, combined_flat_csub_checked_count)

        if combined_ldg_flat_w0:
            # TODO.md Phase 118: same combined structure as
            # --combined-flat-w0, but sourced from real A[]/B[]/D[] LDGs
            # (no exp() yet).
            _print_per_slot_table("wMatrix 0 per-thread real-LDG-sourced As[ty][k] reliability, per slot",
                                   combined_ldg_flat_as_ok_count, combined_ldg_flat_as_checked_count)
            _print_per_slot_table("wMatrix 0 per-thread real-LDG-sourced Bs[k][tx] reliability, per slot",
                                   combined_ldg_flat_bs_ok_count, combined_ldg_flat_bs_checked_count)
            _print_per_slot_table("wMatrix 0 per-thread real-LDG-sourced Ds[k] (raw, no exp) reliability, per slot",
                                   combined_ldg_flat_ds_ok_count, combined_ldg_flat_ds_checked_count)
            _print_per_slot_table("wMatrix 0 per-thread real-LDG-sourced Csub-analog (raw) reliability, per slot",
                                   combined_ldg_flat_csub_ok_count, combined_ldg_flat_csub_checked_count)

        if combined_exp_flat_w0:
            # TODO.md Phase 118: same combined structure, real A[]/B[]/D[]
            # LDGs plus the real exp()/distance formula for Ds[] -- a
            # correct Csub-analog here should exactly equal
            # reference_transition_matrix()'s real answer.
            _print_per_slot_table("wMatrix 0 per-thread real-exp()-sourced As[ty][k] reliability, per slot",
                                   combined_exp_flat_as_ok_count, combined_exp_flat_as_checked_count)
            _print_per_slot_table("wMatrix 0 per-thread real-exp()-sourced Bs[k][tx] reliability, per slot",
                                   combined_exp_flat_bs_ok_count, combined_exp_flat_bs_checked_count)
            _print_per_slot_table("wMatrix 0 per-thread real-exp()-sourced Ds[k] reliability, per slot",
                                   combined_exp_flat_ds_ok_count, combined_exp_flat_ds_checked_count)
            _print_per_slot_table("wMatrix 0 per-thread real-exp()-sourced Csub-analog reliability (should equal the real reference matrix), per slot",
                                   combined_exp_flat_csub_ok_count, combined_exp_flat_csub_checked_count)

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

        # This is the *first* time this session's probe has ever compiled
        # the true, unmodified kernel (every prior run -- Phase 68/69/70
        # -- used TINYGPU_BISECT_NO_EXP by default). If computed_logl came
        # back wrong, read kernelMatrixMulADB's own ground-truth dump back
        # in this *same* run, from the *same* dispatch, to see concretely
        # whether it's the familiar wMatrix>=4-never-executes pattern
        # (Phase 57's original finding) or something new -- not guessed.
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
