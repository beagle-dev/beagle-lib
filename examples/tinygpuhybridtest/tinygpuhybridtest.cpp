/*
 * tinygpuhybridtest.cpp
 *
 * Tests the BEAGLE TinyGPUHybrid backend end-to-end:
 *   1. Enumerates all BEAGLE resources, identifies the TinyGPUHybrid device
 *   2. Initializes a BEAGLE instance on it (which calls nvHybridSetup internally)
 *   3. Dispatches GPU kernels: transition matrices + partial likelihood peeling
 *   4. Evaluates the root log-likelihood and checks against the reference
 *
 * Dataset : human / chimp / gorilla (same as tinytest / tinygputest) at
 *           stateCount=4 (the default); a synthetic one-hot dataset +
 *           generalized-JC model (buildEqualRateN, adapted from
 *           adjointtest4.cpp on the matrix_vector branch) for any other
 *           --state-count N -- exercises BEAGLE's other precompiled kernel
 *           variants (16/32/48/64/80/128/192/256).
 * Model   : JC69 + 4-category discrete Gamma (stateCount==4); generalized-JC
 *           + the same 4-category discrete Gamma otherwise
 * Tree    : ((H:0.1, C:0.1):0.1, G:0.2)  —  two peeling ops
 * Ref logL: -1498.89812 (PAUP, verified against BEAGLE CPU backend) --
 *           stateCount==4 only; no external reference exists for the
 *           synthetic model, so correctness there comes entirely from
 *           --diag-compare-cpu (auto-enabled for --state-count != 4)
 *
 * Usage:
 *   tinygpuhybridtest [--resource N] [--state-count N] [--reps N] [--poison] [--diag-reorder-partials-first] [--diag-compare-cpu] [--diag-inject-matrices] [--diag-matmul-ground-truth]
 *   --resource N          Force BEAGLE resource index N (skips auto-detect)
 *   --poison              Before the checked evaluation, fill the internal and root
 *                         partials buffers and the transition matrices with NaN
 *                         through the BEAGLE API, so results left in GPU memory
 *                         by an earlier run (VRAM survives a boot without a
 *                         power cycle) cannot pass for this run's.
 *   --reps N              After the checked evaluation, repeat the whole
 *                         evaluation (transition matrices, peeling, root
 *                         logL) N more times and report the steady-state
 *                         wall time per evaluation (min/median/mean/max).
 *                         Every repeat must reproduce the first logL
 *                         exactly. Not combinable with the --diag-inject-
 *                         matrices, --diag-reorder-partials-first or
 *                         --diag-matmul-ground-truth pipelines. SIGINT or
 *                         SIGTERM ends the repeats early; the instance is
 *                         still finalized (the GPU torn down) and the exit
 *                         status is 128 + signal. (For hosts and harnesses:
 *                         hardware runs still follow the never-Ctrl-C rule.)
 *   --state-count N       Use a synthetic generalized-JC model with N states
 *                         instead of the default 4-state DNA/JC69 dataset.
 *                         Any N >= 2 works; the values BEAGLE has a
 *                         dedicated precompiled kernel variant for are
 *                         16, 32, 48, 64, 80, 128, 192, 256. A list N,M,...
 *                         gives each of --instances its own, in turn.
 *   --instances K         K TinyGPU instances in one process, alive together
 *                         (TODO.md plan step P5; default: one per --state-count
 *                         entry). Each runs the default pipeline once against
 *                         its own CPU-resource instance (--diag-compare-cpu's
 *                         check), then --reps; afterwards every instance's tip
 *                         partials must read back exactly as set, which an
 *                         allocation overlapping another instance's would break
 *                         (each instance has its own tip data: the first at 4
 *                         states the DNA dataset, the others rotated one-hot
 *                         tips).
 *   --threads             With --instances: each instance's evaluations in its
 *                         own thread (instances are still created and finalized
 *                         one at a time: beagle.cpp's instance table is not
 *                         thread-safe).
 *   --cycles C            Create, run and finalize the instances C times in one
 *                         process.
 *   --fork-exit           After the instances are created (the GPU booted), a
 *                         forked child calls exit(); the parent then runs as
 *                         usual on a GPU the child must not have torn down.
 *   --exit-after MS       MS milliseconds into the --reps evaluations, another
 *                         thread calls exit(0) while this one goes on using
 *                         the GPU, as a host's shutdown would (TODO.md plan
 *                         step C12's exit matrix).
 *   --diag-compare-cpu    Mirror every step onto a second CPU-resource
 *                         instance and compare transition matrices,
 *                         post-peeling partials, and site log-likelihoods
 *                         to localize where a wrong logL first appears.
 *   --diag-inject-matrices  Compute the 4 transition matrices on a CPU
 *                         reference instance and inject them directly into
 *                         the GPU instance (beagleSetTransitionMatrix),
 *                         skipping beagleUpdateTransitionMatrices on the GPU
 *                         instance entirely -- kernelMatrixMulADB is never
 *                         launched. Isolates whether the peeling kernel
 *                         (kernelPartialsPartialsNoScale) computes correctly
 *                         given known-good input, and additionally compares
 *                         each of its two launches' output (buffers 3, 4)
 *                         against the CPU reference individually.
 *   --diag-matmul-ground-truth  Requires a build of the TinyGPU kernel
 *                         header compiled with -DTINYGPU_DEBUG_DUMP_MATMUL_
 *                         GROUND_TRUTH (see make_tinygpu_kernels.sh). Doubles
 *                         nMatrixBuffers so matrix indices 4-7 are unused,
 *                         real-hardware-backed scratch space; seeds them with
 *                         a sentinel, runs the real kernelMatrixMulADB
 *                         launch (unmodified logic/output), then reads back
 *                         each of the 16 wMatrix blocks' own
 *                         Csub/As/Bs/Ds ground truth (or "sentinel
 *                         unchanged" if that block never wrote at all) --
 *                         see TODO.md "PICK UP HERE" for why this
 *                         distinguishes "never ran" from "ran but wrong" in
 *                         a way the normal C[] output can't. Requires
 *                         --state-count 4 (the default).
 */

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <string>
#include <thread>
#include <vector>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include "libhmsbeagle/beagle.h"

// ── Reference sequences ───────────────────────────────────────────────────────

static const char* kHuman =
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
    "TTAACCTTTTAAGTTAAAGATTAAGAGAACCAACACCTCTTTACAGTGA";

static const char* kChimp =
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
    "TTAACCTTTTAAGTTAAAGATTAAGAGGACCGACACCTCTTTACAGTGA";

static const char* kGorilla =
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
    "TTAACCTTTTAAGTTAAAGATTAAGAGTATCGGCACCTCTTTGCAGTGA";

// ── Helpers ───────────────────────────────────────────────────────────────────

static std::vector<double> makePartials(const char* seq, int n) {
    std::vector<double> p(n * 4, 0.0);
    for (int i = 0; i < n; ++i) {
        switch (seq[i]) {
            case 'A': p[i*4+0] = 1.0; break;
            case 'C': p[i*4+1] = 1.0; break;
            case 'G': p[i*4+2] = 1.0; break;
            case 'T': p[i*4+3] = 1.0; break;
            default:  p[i*4+0] = p[i*4+1] = p[i*4+2] = p[i*4+3] = 1.0; break;
        }
    }
    return p;
}

// One-hot tip partials for an arbitrary state count -- used for
// --state-count N != 4, where there's no real DNA sequence to draw on.
// Deterministic (no RNG): taxon `taxonSeed` cycles through states offset by
// its own seed, so different taxa don't all encode the same pattern. `shift`
// rotates every site's state (--instances: each instance its own data).
static std::vector<double> makeSyntheticPartials(int n, int stateCount, int taxonSeed, int shift = 0) {
    std::vector<double> p((size_t) n * stateCount, 0.0);
    for (int i = 0; i < n; ++i) {
        int s = (i * 7 + taxonSeed * 13 + shift) % stateCount;
        p[(size_t) i * stateCount + s] = 1.0;
    }
    return p;
}

// Generalized-JC N-state model: Q[i,i]=-1, Q[i,j]=1/(N-1) for i != j -- a
// symmetric circulant matrix. Eigenvalues are real: 0 once (stationary
// direction, the all-ones vector) and -N/(N-1) with multiplicity N-1, so no
// BEAGLE_FLAG_EIGEN_COMPLEX is needed. Q = r*(J - N*I) where J is the
// all-ones matrix, so any orthogonal V whose first column is ones/sqrt(N)
// diagonalizes it; the symmetric Householder reflection H mapping e_0 to
// ones/sqrt(N) is used for both Evec and Ivec (H = H^T = H^-1). Adapted from
// adjointtest4.cpp's buildEqualRateN (examples/hmctest/, matrix_vector
// branch) -- works for any N, not just powers of two, unlike the Hadamard/
// Sylvester construction used elsewhere in this codebase (synthetictest.cpp).
static void buildEqualRateN(int N, std::vector<double>& evec, std::vector<double>& ivec,
                             std::vector<double>& eval) {
    const double r = 1.0 / (N - 1);
    const double offDiagEigenvalue = -N * r;

    evec.assign((size_t) N * N, 0.0);
    ivec.assign((size_t) N * N, 0.0);
    eval.assign(N, 0.0);

    eval[0] = 0.0;
    for (int k = 1; k < N; k++) eval[k] = offDiagEigenvalue;

    std::vector<double> u(N);
    const double invSqrtN = 1.0 / std::sqrt((double) N);
    u[0] = 1.0 - invSqrtN;
    for (int i = 1; i < N; i++) u[i] = -invSqrtN;
    double uNormSq = 0.0;
    for (int i = 0; i < N; i++) uNormSq += u[i] * u[i];

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double hij = (i == j ? 1.0 : 0.0) - 2.0 * u[i] * u[j] / uNormSq;
            evec[(size_t) i * N + j] = hij;
            ivec[(size_t) i * N + j] = hij;
        }
    }
}

static void printFlags(long f) {
    if (f & BEAGLE_FLAG_PROCESSOR_CPU)     printf(" CPU");
    if (f & BEAGLE_FLAG_PROCESSOR_GPU)     printf(" GPU");
    if (f & BEAGLE_FLAG_PRECISION_DOUBLE)  printf(" DOUBLE");
    if (f & BEAGLE_FLAG_PRECISION_SINGLE)  printf(" SINGLE");
    if (f & BEAGLE_FLAG_FRAMEWORK_CUDA)    printf(" CUDA");
    if (f & BEAGLE_FLAG_FRAMEWORK_OPENCL)  printf(" OPENCL");
    if (f & BEAGLE_FLAG_FRAMEWORK_TINYGPU) printf(" TINYGPU");
    if (f & BEAGLE_FLAG_FRAMEWORK_CPU)     printf(" CPU_FW");
}

// DIAGNOSTIC (see STATUS.md/TODO.md "PICK UP HERE" on the usb branch):
// element-wise compare a GPU-side array against a CPU-reference array of
// the same length, reporting the largest absolute difference (and where)
// plus any finite-vs-non-finite mismatches (the interesting case, since
// logL = -inf means a log(0) entered somewhere -- usually a whole buffer
// of zeros, not a subtly-off value). Returns true if everything matches
// within tolerance and finiteness agrees everywhere. Prints to out (--threads:
// the instance's own buffer).
static bool compareArrays(const char* label, const double* gpuVals, const double* cpuVals,
                           int n, double tol, FILE* out = stdout) {
    double maxAbsDiff = 0.0;
    int maxDiffIndex = -1;
    int nonFiniteMismatches = 0;
    int nMismatches = 0;  // count of elements exceeding tol, not just the largest
    for (int i = 0; i < n; ++i) {
        bool gpuFin = std::isfinite(gpuVals[i]);
        bool cpuFin = std::isfinite(cpuVals[i]);
        if (gpuFin != cpuFin) {
            if (nonFiniteMismatches < 5)
                fprintf(out, "    %s[%d]: GPU=%.10g CPU=%.10g  <- finiteness mismatch\n",
                       label, i, gpuVals[i], cpuVals[i]);
            ++nonFiniteMismatches;
            continue;
        }
        if (!gpuFin) continue;  // both non-finite the same way; nothing to diff
        double d = std::fabs(gpuVals[i] - cpuVals[i]);
        if (d > tol) {
            // Capped at 5 (not the original 300): at larger --state-count,
            // a systematic single-precision-accumulation pattern can have
            // thousands of over-tolerance entries, all telling the same
            // story -- the aggregate maxAbsDiff/count printed below is what
            // actually matters for those, a handful of examples is plenty.
            if (nMismatches < 5)
                fprintf(out, "    %s[%d]: GPU=%.10g CPU=%.10g  diff=%.6g\n",
                       label, i, gpuVals[i], cpuVals[i], d);
            ++nMismatches;
        }
        if (d > maxAbsDiff) { maxAbsDiff = d; maxDiffIndex = i; }
    }
    bool ok = (nonFiniteMismatches == 0) && (maxAbsDiff <= tol);
    fprintf(out, "  %-14s n=%-6d maxAbsDiff=%.6g", label, n, maxAbsDiff);
    if (maxDiffIndex >= 0)
        fprintf(out, " at [%d] (GPU=%.10g CPU=%.10g)", maxDiffIndex, gpuVals[maxDiffIndex], cpuVals[maxDiffIndex]);
    if (nMismatches > 0)
        fprintf(out, "  [%d/%d elements > tol]", nMismatches, n);
    if (nonFiniteMismatches > 0)
        fprintf(out, "  [%d finiteness mismatches]", nonFiniteMismatches);
    fprintf(out, "  %s\n", ok ? "OK" : "MISMATCH");
    return ok;
}

// SIGINT/SIGTERM only set this flag: the --reps loop stops and the instance is
// finalized normally, which tears the GPU down through the plugin (TODO.md plan
// step P3; inv:transport-teardown#8). The library installs no handlers.
static volatile sig_atomic_t gStopSignal = 0;
static void onStopSignal(int sig) { gStopSignal = sig; }

// --exit-after MS (TODO.md plan step C12's exit matrix): MS milliseconds after the --reps evaluations start, another thread
// calls exit(0) while they go on, as a host's shutdown (a JVM's System.exit) would
static std::atomic<bool> gEvaluating(false);
static void exitAfter(int ms) {
    std::thread([ms] {
        while (!gEvaluating) std::this_thread::sleep_for(std::chrono::milliseconds(5));
        std::this_thread::sleep_for(std::chrono::milliseconds(ms));
        printf("exit() from another thread, %d ms into the evaluations\n", ms);
        fflush(stdout);
        exit(0);
    }).detach();
}

// ── Several instances in one process (--instances, --threads, --cycles; TODO.md plan step P5) ──
// The TinyGPU instances share one boot. Each runs the default pipeline against its own CPU-resource instance; BEAGLE
// instances are created and finalized on the main thread only.

static const double kRefLogL = -1498.89812;   // the DNA/JC69 model's (stateCount 4), as main's check
static const double kRefTol  = 0.5;

struct InstanceRun {
    int index = 0, stateCount = 4, nPatterns = 0;
    int gpu = -1, cpu = -1;
    std::vector<double> tips[3], evec, ivec, eval, freqs;
    FILE* out = stdout;       // --threads: a memory stream, printed once the instance is done
    char* buf = nullptr;
    size_t len = 0;
    bool ok = false;
};

// main's instance creation and setupModel, for one instance and its CPU reference. Each instance has its own tip data,
// so an allocation shared with another instance of the same state count shows in checkTips and the CPU comparison: the
// first instance at 4 states keeps the DNA dataset (and main's reference logL), every later one one-hot tips rotated by
// its index (distinct while there are fewer instances than states).
static bool createRun(InstanceRun& r, int resourceIdx, bool forced, int cpuIdx) {
    r.nPatterns = (int)strlen(kHuman);
    if (r.stateCount == 4 && r.index == 0) {
        r.tips[0] = makePartials(kHuman, r.nPatterns);
        r.tips[1] = makePartials(kChimp, r.nPatterns);
        r.tips[2] = makePartials(kGorilla, r.nPatterns);
    } else {
        for (int t = 0; t < 3; ++t) r.tips[t] = makeSyntheticPartials(r.nPatterns, r.stateCount, t, r.index);
    }
    if (r.stateCount == 4) {
        r.evec = { 1.0, 2.0, 0.0, 0.5,   1.0, -2.0, 0.5, 0.0,   1.0, 2.0, 0.0, -0.5,   1.0, -2.0, -0.5, 0.0 };
        r.ivec = { 0.25, 0.25, 0.25, 0.25,   0.125, -0.125, 0.125, -0.125,   0.0, 1.0, 0.0, -1.0,   1.0, 0.0, -1.0, 0.0 };
        r.eval = { 0.0, -4.0/3.0, -4.0/3.0, -4.0/3.0 };
        r.freqs.assign(4, 0.25);
    } else {
        buildEqualRateN(r.stateCount, r.evec, r.ivec, r.eval);
        r.freqs.assign(r.stateCount, 1.0 / r.stateCount);
    }
    BeagleInstanceDetails det;
    r.gpu = beagleCreateInstance(3, 5, 0, r.stateCount, r.nPatterns, 1, 4, 4, 0, &resourceIdx, 1,
                                 BEAGLE_FLAG_PRECISION_DOUBLE | BEAGLE_FLAG_PROCESSOR_GPU,
                                 forced ? 0 : BEAGLE_FLAG_FRAMEWORK_TINYGPU, &det);
    if (r.gpu < 0) { fprintf(stderr, "beagleCreateInstance failed (error %d)\n", r.gpu); return false; }
    printf("instance %d: %d states on resource %d: %s (%s)\n", r.index, r.stateCount, det.resourceNumber, det.resourceName,
           det.implName);
    r.cpu = beagleCreateInstance(3, 5, 0, r.stateCount, r.nPatterns, 1, 4, 4, 0, &cpuIdx, 1,
                                 BEAGLE_FLAG_PRECISION_DOUBLE | BEAGLE_FLAG_PROCESSOR_CPU, 0, &det);
    if (r.cpu < 0) { fprintf(stderr, "beagleCreateInstance (CPU) failed (error %d)\n", r.cpu); return false; }
    double rates[4]   = { 0.03338775, 0.25191592, 0.82026848, 2.89442785 };
    double weights[4] = { 0.25, 0.25, 0.25, 0.25 };
    std::vector<double> patW(r.nPatterns, 1.0);
    for (int inst : { r.gpu, r.cpu }) {
        for (int t = 0; t < 3; ++t) beagleSetTipPartials(inst, t, r.tips[t].data());
        beagleSetEigenDecomposition(inst, 0, r.evec.data(), r.ivec.data(), r.eval.data());
        beagleSetStateFrequencies(inst, 0, r.freqs.data());
        beagleSetCategoryRates(inst, rates);
        beagleSetCategoryWeights(inst, 0, weights);
        beagleSetPatternWeights(inst, patW.data());
    }
    return true;
}

// main's default pipeline (--poison, the evaluation, --diag-compare-cpu's stages and tolerances, --reps) on one instance
static void evaluateRun(InstanceRun& r, bool poison, int reps) {
    FILE* out = r.out;
    int n = r.stateCount, np = r.nPatterns;
    int nodeIdx[4] = { 0, 1, 2, 3 };
    double edgeLens[4] = { 0.1, 0.1, 0.2, 0.1 };
    BeagleOperation ops[2] = { { 3, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 0, 0, 1, 1 }, { 4, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 2, 2, 3, 3 } };
    int rootBuf = 4, wBuf = 0, fBuf = 0, sBuf = BEAGLE_OP_NONE;
    auto evaluate = [&](int inst, double* logL) {
        int rc = beagleUpdateTransitionMatrices(inst, 0, nodeIdx, nullptr, nullptr, edgeLens, 4);
        if (rc >= 0) rc = beagleUpdatePartials(inst, ops, 2, BEAGLE_OP_NONE);
        if (rc >= 0) rc = beagleCalculateRootLogLikelihoods(inst, &rootBuf, &wBuf, &fBuf, &sBuf, 1, logL);
        return rc;
    };
    fprintf(out, "\n=== instance %d: %d states ===\n", r.index, n);
    if (poison) {
        std::vector<double> nanPartials((size_t) n * np * 4, std::nan("")), nanMatrix((size_t) n * n * 4, std::nan(""));
        for (int b = 3; b < 5; ++b) beagleSetPartials(r.gpu, b, nanPartials.data());
        for (int m = 0; m < 4; ++m) beagleSetTransitionMatrix(r.gpu, m, nanMatrix.data(), 0.0);
    }
    double logL = 0.0, cpuLogL = 0.0;
    int rc = evaluate(r.gpu, &logL);
    evaluate(r.cpu, &cpuLogL);
    fprintf(out, "logL = %.5f, CPU-reference logL = %.5f%s\n", logL, cpuLogL, rc < 0 ? " (the GPU evaluation failed)" : "");
    bool ok = rc >= 0 && (n != 4 || r.index != 0 || std::fabs(logL - kRefLogL) < kRefTol);   // the DNA dataset's reference
    std::vector<double> g((size_t) n * n * 4), c((size_t) n * n * 4);
    for (int m = 0; m < 4; ++m) {
        char label[32];
        snprintf(label, sizeof(label), "matrix[%d]", m);
        beagleGetTransitionMatrix(r.gpu, m, g.data());
        beagleGetTransitionMatrix(r.cpu, m, c.data());
        ok &= compareArrays(label, g.data(), c.data(), (int) g.size(), 1e-6 * std::max(1.0, n / 16.0), out);
    }
    g.resize((size_t) np * n * 4); c.resize(g.size());
    for (int b : { 3, 4 }) {
        char label[32];
        snprintf(label, sizeof(label), "partials[%d]", b);
        beagleGetPartials(r.gpu, b, BEAGLE_OP_NONE, g.data());
        beagleGetPartials(r.cpu, b, BEAGLE_OP_NONE, c.data());
        ok &= compareArrays(label, g.data(), c.data(), (int) g.size(), 1e-6, out);
    }
    g.resize(np); c.resize(np);
    beagleGetSiteLogLikelihoods(r.gpu, g.data());
    beagleGetSiteLogLikelihoods(r.cpu, c.data());
    ok &= compareArrays("siteLogL", g.data(), c.data(), np, 1e-3, out);
    std::vector<double> ms;   // as main, --reps runs whenever the evaluation itself succeeded
    int nDiffer = 0;
    gEvaluating = true;
    for (int i = 0; i < reps && rc >= 0 && !gStopSignal; ++i) {
        double repLogL = 0.0;
        auto s0 = std::chrono::steady_clock::now();
        if (evaluate(r.gpu, &repLogL) < 0) { fprintf(out, "--reps: evaluation %d failed\n", i); break; }
        ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s0).count());
        nDiffer += (repLogL != logL);
    }
    if (!ms.empty()) {
        std::vector<double> s(ms);
        std::sort(s.begin(), s.end());
        double sum = 0.0;
        for (double v : ms) sum += v;
        fprintf(out, "per evaluation: min %.3f ms  median %.3f ms  mean %.3f ms  max %.3f ms (%zu repeats)\n",
                s.front(), s[s.size() / 2], sum / s.size(), s.back(), s.size());
        fprintf(out, "logL: %d of %zu repeats differ from the first evaluation\n", nDiffer, ms.size());
    }
    r.ok = ok && nDiffer == 0 && (int) ms.size() == reps && !gStopSignal;
    fprintf(out, "instance %d (%d states): %s\n", r.index, n, r.ok ? "PASS" : "FAIL");
}

// Every tip partial reads back exactly as set: each is a 0/1 value that round-trips through single precision, so any
// difference is another instance's data in this instance's buffer.
static bool checkTips(const InstanceRun& r) {
    size_t per = (size_t) r.nPatterns * r.stateCount;
    std::vector<double> back(per * 4);   // 4 rate categories, each a copy of the tip (BeagleGPUImpl::setTipPartials)
    for (int t = 0; t < 3; ++t) {
        beagleGetPartials(r.gpu, t, BEAGLE_OP_NONE, back.data());
        for (int c = 0; c < 4; ++c)
            if (!std::equal(r.tips[t].begin(), r.tips[t].end(), back.begin() + c * per)) return false;
    }
    return true;
}

static int runInstances(const std::vector<int>& stateCounts, int instances, bool threads, int cycles, int resourceIdx,
                        bool forced, bool poison, int reps, bool forkExit, const BeagleResourceList* rList) {
    int cpuIdx = -1;
    for (int i = 0; i < rList->length && cpuIdx < 0; ++i)
        if (rList->list[i].supportFlags & BEAGLE_FLAG_PROCESSOR_CPU) cpuIdx = i;
    if (cpuIdx < 0) { fprintf(stderr, "no CPU resource for the reference instances\n"); return 1; }
    bool allOk = true;
    for (int cycle = 0; cycle < cycles && !gStopSignal; ++cycle) {
        printf("=== cycle %d of %d: %d instance(s)%s ===\n", cycle + 1, cycles, instances, threads ? ", one thread each" : "");
        std::vector<InstanceRun> runs(instances);
        bool created = true;
        for (int i = 0; i < instances && created; ++i) {
            runs[i].index = i;
            runs[i].stateCount = stateCounts[i % stateCounts.size()];
            created = createRun(runs[i], resourceIdx, forced, cpuIdx);
        }
        if (created && forkExit && cycle == 0) {   // plan step P5's review: a child forked after the boot exits normally,
            fflush(stdout);                         // running the plugin's atexit handler; the GPU is the parent's and must
            fflush(stderr);                         // stay up (the fake GPU flags any command after a teardown)
            pid_t child = fork();
            if (child == 0) exit(0);
            int st = 0;
            waitpid(child, &st, 0);
            printf("forked child %d exited with status %d\n", (int)child, WIFEXITED(st) ? WEXITSTATUS(st) : -1);
        }
        if (created) {
            fflush(stdout);
            if (threads) {
                std::vector<std::thread> pool;
                for (InstanceRun& r : runs) {
                    r.out = open_memstream(&r.buf, &r.len);
                    pool.emplace_back(evaluateRun, std::ref(r), poison, reps);
                }
                for (std::thread& t : pool) t.join();
                for (InstanceRun& r : runs) { fclose(r.out); fwrite(r.buf, 1, r.len, stdout); free(r.buf); r.out = stdout; }
            } else {
                for (InstanceRun& r : runs) evaluateRun(r, poison, reps);
            }
            bool tipsOk = true;
            for (const InstanceRun& r : runs) tipsOk &= checkTips(r);
            printf("tips: %s\n", tipsOk ? "every instance read back its own tip partials exactly"
                                        : "an instance read back tip partials it did not set -- FAIL");
            for (const InstanceRun& r : runs) allOk &= r.ok;
            allOk &= tipsOk;
        }
        allOk &= created;
        for (const InstanceRun& r : runs) {
            if (r.cpu >= 0) beagleFinalizeInstance(r.cpu);
            if (r.gpu >= 0) beagleFinalizeInstance(r.gpu);
        }
    }
    printf("\n%s\n", allOk && !gStopSignal ? "PASS" : "FAIL");
    return gStopSignal ? 128 + gStopSignal : (allOk ? 0 : 1);
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    int forceResource = -1;
    // --state-count N: exercise a different (padded) kernel variant. 4 (the
    // default) uses the real DNA dataset + JC69 model below, unchanged; any
    // other N uses a synthetic generalized-JC model (buildEqualRateN) and
    // synthetic one-hot tip data instead, since there's no real sequence
    // data or precomputed reference logL for an arbitrary state count.
    // Correctness for N != 4 is judged by GPU-vs-CPU-backend agreement
    // (--diag-compare-cpu's mechanism, auto-enabled in that case) rather
    // than a hardcoded logL, since no external reference exists.
    int stateCount = 4;
    // DIAGNOSTIC (see STATUS.md/TODO.md "Phase 2" on the usb branch): when
    // set, launches the partials-peeling kernels (plain global-pointer
    // params, no shared-memory pointer broadcast) *before*
    // updateTransitionMatrices (kernelMatrixMulADB, which does broadcast a
    // global pointer through shared memory). Matrix buffers will contain
    // garbage at that point — logL will be wrong — this only tests whether
    // kernelPartialsPartialsNoScale completes (advances eop, no SM
    // exceptions) when it isn't preceded by kernelMatrixMulADB. Remove once
    // Phase 2 is resolved.
    bool diagReorderPartialsFirst = false;
    // DIAGNOSTIC (see STATUS.md/TODO.md "PICK UP HERE" on the usb branch):
    // when set, mirrors every step run on the TinyGPUHybrid instance onto a
    // second instance forced onto the CPU resource, and compares the
    // transition matrices, post-peeling partials, and per-site log-
    // likelihoods against it to localize where a wrong `logL` first
    // enters the pipeline. Doesn't touch the TinyGPU/eGPU path at all
    // beyond what the non-diagnostic run already does.
    bool diagCompareCpu = false;
    // DIAGNOSTIC (see STATUS.md/TODO.md "PICK UP HERE" on the usb branch):
    // when set, the 4 transition matrices are computed on a CPU-reference
    // instance and injected directly into the GPU instance via
    // beagleSetTransitionMatrix -- beagleUpdateTransitionMatrices is never
    // called on the GPU instance, so kernelMatrixMulADB (the kernel already
    // known to be broken) never launches at all. Only
    // kernelPartialsPartialsNoScale (+ the root-logL reduction kernels) run
    // on the GPU. Isolates whether the peeling kernel itself is correct,
    // independent of kernelMatrixMulADB.
    bool diagInjectMatrices = false;
    // DIAGNOSTIC (see STATUS.md/TODO.md "PICK UP HERE" on the usb branch,
    // NV Phase 47): per-block (per-wMatrix) ground-truth dump of
    // kernelMatrixMulADB's own real Csub/As/Bs/Ds, written to a dedicated
    // scratch region (not the real, wMatrix-aliased C[] output) so a block
    // that never ran is distinguishable from one that ran and computed a
    // wrong value. Requires a kernel header built with
    // -DTINYGPU_DEBUG_DUMP_MATMUL_GROUND_TRUTH.
    bool diagMatmulGroundTruth = false;
    bool poison = false;
    int reps = 0;
    std::vector<int> stateCounts;   // --state-count's list: each of --instances takes the next (plan step P5)
    int instances = 0, cycles = 1;
    bool forkExit = false;   // --fork-exit (plan step P5's review)
    int exitAfterMs = -1;    // --exit-after (plan step C12)
    bool threads = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--resource" && i + 1 < argc)
            forceResource = atoi(argv[++i]);
        else if (a == "--state-count" && i + 1 < argc) {
            stateCounts.clear();
            for (const char* p = argv[++i]; *p; ++p) { stateCounts.push_back((int)strtol(p, (char**)&p, 10)); if (*p != ',') break; }
            stateCount = stateCounts[0];
        } else if (a == "--instances" && i + 1 < argc)
            instances = atoi(argv[++i]);
        else if (a == "--threads")
            threads = true;
        else if (a == "--cycles" && i + 1 < argc)
            cycles = atoi(argv[++i]);
        else if (a == "--fork-exit")
            forkExit = true;
        else if (a == "--exit-after" && i + 1 < argc)
            exitAfterMs = atoi(argv[++i]);
        else if (a == "--reps" && i + 1 < argc)
            reps = atoi(argv[++i]);
        else if (a == "--poison")
            poison = true;
        else if (a == "--diag-reorder-partials-first")
            diagReorderPartialsFirst = true;
        else if (a == "--diag-compare-cpu")
            diagCompareCpu = true;
        else if (a == "--diag-inject-matrices")
            diagInjectMatrices = true;
        else if (a == "--diag-matmul-ground-truth")
            diagMatmulGroundTruth = true;
        else {
            fprintf(stderr, "Usage: tinygpuhybridtest [--resource N] [--state-count N[,N...]] [--instances K] [--threads] [--cycles C] [--fork-exit] [--exit-after MS] [--reps N] [--poison] [--diag-reorder-partials-first] [--diag-compare-cpu] [--diag-inject-matrices] [--diag-matmul-ground-truth]\n");
            return 1;
        }
    }
    if (stateCounts.empty()) stateCounts.push_back(stateCount);
    for (int n : stateCounts) {
        if (n < 2) {
            fprintf(stderr, "--state-count must be >= 2 (got %d)\n", n);
            return 1;
        }
    }
    if (instances == 0) instances = (int)stateCounts.size();
    bool several = instances > 1 || threads || cycles > 1 || forkExit;
    if (instances < 1 || cycles < 1 || (several && (diagInjectMatrices || diagReorderPartialsFirst || diagMatmulGroundTruth))) {
        fprintf(stderr, "--instances and --cycles need a count >= 1, and several instances (or --threads, --cycles) run the "
                         "default pipeline only, not --diag-inject-matrices, --diag-reorder-partials-first or "
                         "--diag-matmul-ground-truth\n");
        return 1;
    }
    if (reps < 0 || (reps > 0 && (diagInjectMatrices || diagReorderPartialsFirst || diagMatmulGroundTruth))) {
        fprintf(stderr, "--reps needs N >= 0 and repeats the default pipeline, so it can't be combined with "
                         "--diag-inject-matrices, --diag-reorder-partials-first or --diag-matmul-ground-truth\n");
        return 1;
    }
    if (diagMatmulGroundTruth && (stateCount != 4 || diagInjectMatrices)) {
        fprintf(stderr, "--diag-matmul-ground-truth requires the default --state-count 4 "
                         "and is incompatible with --diag-inject-matrices "
                         "(which skips kernelMatrixMulADB entirely)\n");
        return 1;
    }
    bool useDnaModel = (stateCount == 4);
    if (!useDnaModel) diagCompareCpu = true;  // only available correctness check for a synthetic model

    // SA_RESTART: the plugin's socket loops (nv_recv_all, nv_send_all,
    // tg_send_all) take EINTR for a lost connection and would cut a message.
    struct sigaction sa = {};
    sa.sa_handler = onStopSignal;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);
    if (exitAfterMs >= 0) exitAfter(exitAfterMs);

    // ── Step 1: Enumerate all BEAGLE resources ────────────────────────────────
    printf("=== TinyGPUHybrid backend test ===\n\n");
    BeagleResourceList* rList = beagleGetResourceList();
    printf("BEAGLE %s  resources (%d):\n", beagleGetVersion(), rList->length);

    int hybridIdx = -1;
    for (int i = 0; i < rList->length; ++i) {
        const char* name = rList->list[i].name;
        const char* desc = rList->list[i].description ? rList->list[i].description : "";
        bool isHybrid = strstr(name, "Hybrid") != nullptr || strstr(desc, "hybrid") != nullptr;

        printf("  [%d] %s", i, name);
        if (isHybrid) {
            printf("  <-- TinyGPUHybrid");
            if (hybridIdx < 0) hybridIdx = i;
        }
        printf("\n");
        if (desc[0]) printf("       %s\n", desc);
        printf("       flags:");
        printFlags(rList->list[i].supportFlags);
        printf("\n");
    }
    printf("\n");

    // ── Step 2: Select resource ───────────────────────────────────────────────
    int resourceIdx = (forceResource >= 0) ? forceResource : hybridIdx;
    if (resourceIdx < 0) {
        fprintf(stderr,
            "No TinyGPUHybrid resource detected.\n"
            "Ensure hmsbeagle-tinygpu-hybrid.so is on the plugin path.\n"
            "Use --resource N to force a specific resource index.\n");
        return 1;
    }
    printf("Using resource index %d\n\n", resourceIdx);
    if (several)
        return runInstances(stateCounts, instances, threads, cycles, resourceIdx, forceResource >= 0, poison, reps, forkExit, rList);

    // ── Step 3: Create BEAGLE instance ───────────────────────────────────────
    // Dataset: 3 taxa, 5 partial buffers (tips 0-2, internal 3, root 4)
    int nPatterns = (int)strlen(kHuman);
    BeagleInstanceDetails det;
    // --diag-matmul-ground-truth doubles the matrix-buffer pool: indices
    // 0-3 are the real edges, exactly as before; indices 4-7 are never
    // touched by any real BEAGLE bookkeeping and become dedicated,
    // real-hardware-backed scratch space for the probe (dMatrices is one
    // contiguous allocation across all matrix-buffer indices, so this
    // just extends it -- see kernelsAll.cu's TINYGPU_DEBUG_DUMP_MATMUL_
    // GROUND_TRUTH comment).
    int instance = beagleCreateInstance(
        3,              // nTips
        5,              // nPartialBuffers
        0,              // nCompactBuffers
        stateCount,     // stateCount
        nPatterns,      // nPatterns
        1,              // nEigenBuffers
        diagMatmulGroundTruth ? 8 : 4,  // nMatrixBuffers
        4,              // nRateCats
        0,              // nScaleBuffers
        &resourceIdx, 1,
        BEAGLE_FLAG_PRECISION_DOUBLE  |
        BEAGLE_FLAG_PROCESSOR_GPU,
        // reqFlags: a forced --resource may be any backend (e.g. the CPU,
        // as a --reps baseline); auto-detect insists on TinyGPU
        forceResource >= 0 ? 0 : BEAGLE_FLAG_FRAMEWORK_TINYGPU,
        &det);

    if (instance < 0) {
        fprintf(stderr, "beagleCreateInstance failed (error %d)\n", instance);
        return 1;
    }

    printf("Resource %d: %s\n", det.resourceNumber, det.resourceName);
    printf("Implementation : %s\n", det.implName);
    if (det.implDescription && det.implDescription[0])
        printf("Description    : %s\n", det.implDescription);
    printf("Active flags   :");
    printFlags(det.flags);
    printf("\n\n");

    // ── Steps 4/5: tip partials + eigen decomposition ────────────────────────
    // Factored into a lambda so --diag-compare-cpu can apply the identical
    // model setup to a second (CPU) instance. stateCount==4 (default): real
    // DNA data + JC69, exactly as before. Otherwise: synthetic one-hot tips
    // + a generalized-JC model (buildEqualRateN) for the requested N.
    std::vector<double> hp, cp, gp;
    std::vector<double> evec, ivec, eval, freqs;
    if (useDnaModel) {
        hp = makePartials(kHuman,   nPatterns);
        cp = makePartials(kChimp,   nPatterns);
        gp = makePartials(kGorilla, nPatterns);
        evec = { 1.0,  2.0,  0.0,  0.5,
                 1.0, -2.0,  0.5,  0.0,
                 1.0,  2.0,  0.0, -0.5,
                 1.0, -2.0, -0.5,  0.0 };
        ivec = { 0.25,  0.25,  0.25,  0.25,
                 0.125,-0.125, 0.125,-0.125,
                 0.0,   1.0,   0.0,  -1.0,
                 1.0,   0.0,  -1.0,   0.0 };
        eval = { 0.0, -4.0/3.0, -4.0/3.0, -4.0/3.0 };
        freqs.assign(stateCount, 0.25);
    } else {
        hp = makeSyntheticPartials(nPatterns, stateCount, 0);
        cp = makeSyntheticPartials(nPatterns, stateCount, 1);
        gp = makeSyntheticPartials(nPatterns, stateCount, 2);
        buildEqualRateN(stateCount, evec, ivec, eval);
        freqs.assign(stateCount, 1.0 / stateCount);
    }

    // 4-category discrete Gamma (alpha ≈ 0.5) -- unrelated to stateCount
    double rates[4]   = { 0.03338775, 0.25191592, 0.82026848, 2.89442785 };
    double weights[4] = { 0.25, 0.25, 0.25, 0.25 };

    std::vector<double> patW(nPatterns, 1.0);

    auto setupModel = [&](int inst) {
        beagleSetTipPartials(inst, 0, hp.data());
        beagleSetTipPartials(inst, 1, cp.data());
        beagleSetTipPartials(inst, 2, gp.data());
        beagleSetEigenDecomposition(inst, 0, evec.data(), ivec.data(), eval.data());
        beagleSetStateFrequencies(inst, 0, freqs.data());
        beagleSetCategoryRates(inst, rates);
        beagleSetCategoryWeights(inst, 0, weights);
        beagleSetPatternWeights(inst, patW.data());
    };
    setupModel(instance);

    // --diag-matmul-ground-truth: seed the debug scratch region (matrix
    // indices 4-7, backing wMatrix 16-31 -- see kernelsAll.cu's
    // TINYGPU_DEBUG_DUMP_MATMUL_GROUND_TRUTH) with an obviously-invalid
    // sentinel before kernelMatrixMulADB ever runs, so the read-back below
    // can tell "this block never wrote here" (still the sentinel) apart
    // from "this block wrote its real ground truth" -- something the
    // kernel's normal, wMatrix-aliased C[] output can't distinguish.
    if (diagMatmulGroundTruth) {
        std::vector<double> sentinel((size_t) stateCount * stateCount * 4, -999.0);
        for (int m = 4; m < 8; ++m)
            beagleSetTransitionMatrix(instance, m, sentinel.data(), 0.0);
    }

    // ── Steps 6/7: transition matrices + partial likelihood peeling ─────────
    int    nodeIdx[4]  = { 0, 1, 2, 3 };
    double edgeLens[4] = { 0.1, 0.1, 0.2, 0.1 };
    // op: {destBuf, destScale, srcScale, child1buf, child1mat, child2buf, child2mat}
    BeagleOperation ops[2] = {
        { 3, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 0, 0, 1, 1 },  // node3 = H*C
        { 4, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 2, 2, 3, 3 }   // root4 = G*node3
    };

    // ── Step 7b (diagnostic): inject CPU-computed transition matrices ───────
    // cpuRefInstance is kept alive (not finalized here) so the per-launch
    // kernelPartialsPartialsNoScale check below can compare against it too.
    int cpuRefInstance = -1;
    if (diagInjectMatrices) {
        fprintf(stderr, "DIAGNOSTIC: injecting CPU-computed transition matrices into the "
                         "GPU instance; kernelMatrixMulADB will not be launched.\n");
        int cpuIdx = -1;
        for (int i = 0; i < rList->length; ++i) {
            if (rList->list[i].supportFlags & BEAGLE_FLAG_PROCESSOR_CPU) { cpuIdx = i; break; }
        }
        if (cpuIdx < 0) {
            fprintf(stderr, "DIAGNOSTIC: no CPU resource found, cannot inject matrices\n");
            beagleFinalizeInstance(instance);
            return 1;
        }
        BeagleInstanceDetails cpuDet;
        cpuRefInstance = beagleCreateInstance(
            3, 5, 0, stateCount, nPatterns, 1, 4, 4, 0,
            &cpuIdx, 1,
            BEAGLE_FLAG_PRECISION_DOUBLE | BEAGLE_FLAG_PROCESSOR_CPU,
            0,
            &cpuDet);
        if (cpuRefInstance < 0) {
            fprintf(stderr, "DIAGNOSTIC: beagleCreateInstance (CPU) failed: %d\n", cpuRefInstance);
            beagleFinalizeInstance(instance);
            return 1;
        }
        setupModel(cpuRefInstance);
        beagleUpdateTransitionMatrices(cpuRefInstance, 0, nodeIdx, nullptr, nullptr, edgeLens, 4);
        std::vector<double> refMat((size_t) stateCount * stateCount * 4);
        for (int m = 0; m < 4; ++m) {
            beagleGetTransitionMatrix(cpuRefInstance, m, refMat.data());
            beagleSetTransitionMatrix(instance, m, refMat.data(), 0.0);
        }
        // Peel on the CPU instance too, so buffers 3/4 hold a correct
        // reference to compare the GPU's kernelPartialsPartialsNoScale
        // output against, one launch at a time (see below).
        beagleUpdatePartials(cpuRefInstance, ops, 2, BEAGLE_OP_NONE);
    }

    auto doUpdateTransitionMatrices = [&]() -> int {
        int r = beagleUpdateTransitionMatrices(instance, 0, nodeIdx, nullptr, nullptr, edgeLens, 4);
        if (r < 0) fprintf(stderr, "updateTransitionMatrices failed: %d\n", r);
        return r;
    };
    auto doUpdatePartials = [&]() -> int {
        int r = beagleUpdatePartials(instance, ops, 2, BEAGLE_OP_NONE);
        if (r < 0) fprintf(stderr, "updatePartials failed: %d\n", r);
        return r;
    };
    // --diag-matmul-ground-truth: read back the 4 debug matrix indices
    // (4-7), decode each into its 4 real wMatrix slots (matrixIndex m
    // covers wMatrix {4(m-4)..4(m-4)+3}, same stride convention
    // --diag-compare-cpu's matrix[m] already uses), and report per-wMatrix
    // whether the probe ran at all (still the -999 sentinel = never wrote)
    // and, if so, its Csub/As/Bs/Ds ground truth.
    auto dumpMatmulGroundTruth = [&]() {
        printf("\n=== --diag-matmul-ground-truth: per-block (per-wMatrix) ground truth ===\n");
        std::vector<double> dbg((size_t) stateCount * stateCount * 4);
        for (int m = 4; m < 8; ++m) {
            beagleGetTransitionMatrix(instance, m, dbg.data());
            for (int c = 0; c < 4; ++c) {
                int wMatrix = (m - 4) * 4 + c;
                const double* d = dbg.data() + c * 16;
                bool neverWrote = true;
                for (int k = 0; k < 13; ++k)
                    if (d[k] != -999.0) { neverWrote = false; break; }
                if (neverWrote) {
                    printf("  wMatrix %2d: NEVER WROTE (sentinel unchanged)\n", wMatrix);
                } else {
                    // d[13] (Phase 50): %smid, captured in the same guarded
                    // write as d[0..12] -- checked separately from
                    // neverWrote above since wMatrix 3's own history
                    // (Phase 48) shows individual fields within an
                    // otherwise-running block can still miss.
                    if (d[13] == -999.0)
                        printf("  wMatrix %2d: Csub=%.10g  As[0][0..3]=(%.6g,%.6g,%.6g,%.6g)  "
                               "Bs[0..3][0]=(%.6g,%.6g,%.6g,%.6g)  Ds[0..3]=(%.6g,%.6g,%.6g,%.6g)  SMID=<not written>\n",
                               wMatrix, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8],
                               d[9], d[10], d[11], d[12]);
                    else
                        printf("  wMatrix %2d: Csub=%.10g  As[0][0..3]=(%.6g,%.6g,%.6g,%.6g)  "
                               "Bs[0..3][0]=(%.6g,%.6g,%.6g,%.6g)  Ds[0..3]=(%.6g,%.6g,%.6g,%.6g)  SMID=%.0f\n",
                               wMatrix, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8],
                               d[9], d[10], d[11], d[12], d[13]);
                }
            }
        }
    };

    int rc;
    if (diagInjectMatrices) {
        // kernelMatrixMulADB deliberately skipped -- matrices already injected above.
        rc = doUpdatePartials();
        if (rc < 0) { beagleFinalizeInstance(instance); return 1; }

        // Debug one kernelPartialsPartialsNoScale launch at a time: buffer 3
        // is the *first* launch's output (H*C, both children are tip
        // partials -- no dependency on any other GPU kernel's result).
        // buffer 4 is the *second* launch, which additionally reads back
        // buffer 3's GPU-computed output as an input. If buffer 3 already
        // mismatches, the bug is in the kernel/its inputs directly, not
        // something compounding across launches.
        printf("\n=== --diag-inject-matrices: per-launch kernelPartialsPartialsNoScale check ===\n");
        std::vector<double> gpuPart((size_t) nPatterns * stateCount * 4), cpuPart((size_t) nPatterns * stateCount * 4);
        for (int buf : {3, 4}) {
            beagleGetPartials(instance, buf, BEAGLE_OP_NONE, gpuPart.data());
            beagleGetPartials(cpuRefInstance, buf, BEAGLE_OP_NONE, cpuPart.data());
            char label[32];
            snprintf(label, sizeof(label), "partials[%d]", buf);
            compareArrays(label, gpuPart.data(), cpuPart.data(), (int) gpuPart.size(), 1e-6);
        }
    } else if (diagReorderPartialsFirst) {
        fprintf(stderr, "DIAGNOSTIC: launching kernelPartialsPartialsNoScale before "
                         "kernelMatrixMulADB (matrix buffers not yet valid; logL will be wrong)\n");
        rc = doUpdatePartials();
        if (rc < 0) { beagleFinalizeInstance(instance); return 1; }
        rc = doUpdateTransitionMatrices();
        if (rc < 0) { beagleFinalizeInstance(instance); return 1; }
        if (diagMatmulGroundTruth) dumpMatmulGroundTruth();
    } else {
        if (poison) {
            std::vector<double> nanPartials((size_t) stateCount * nPatterns * 4, std::nan(""));
            std::vector<double> nanMatrix((size_t) stateCount * stateCount * 4, std::nan(""));
            for (int b = 3; b < 5; ++b) beagleSetPartials(instance, b, nanPartials.data());
            for (int m = 0; m < 4; ++m) beagleSetTransitionMatrix(instance, m, nanMatrix.data(), 0.0);
            printf("--poison: partials buffers 3-4 and transition matrices 0-3 set to NaN before the evaluation\n");
        }
        rc = doUpdateTransitionMatrices();
        if (rc < 0) { beagleFinalizeInstance(instance); return 1; }
        if (diagMatmulGroundTruth) dumpMatmulGroundTruth();
        rc = doUpdatePartials();
        if (rc < 0) { beagleFinalizeInstance(instance); return 1; }
    }

    // ── Step 8: Root log-likelihood ───────────────────────────────────────────
    int rootBuf = 4, wBuf = 0, fBuf = 0, sBuf = BEAGLE_OP_NONE;
    double logL = 0.0;

    auto t0 = std::chrono::steady_clock::now();
    rc = beagleCalculateRootLogLikelihoods(instance, &rootBuf, &wBuf, &fBuf, &sBuf, 1, &logL);
    auto t1 = std::chrono::steady_clock::now();

    const double kRef = -1498.89812;   // only meaningful for the real DNA/JC69 model (stateCount==4)
    const double kTol = 0.5;   // nats; generous for single precision
    bool logLOk = (rc >= 0);
    double delta = 0.0;

    if (!logLOk) {
        // Kernels still ran and synced (sig reached) -- only the CPU-side
        // finiteness/error check failed. Buffers are still readable, so
        // don't bail out here if --diag-compare-cpu wants to inspect them.
        fprintf(stderr, "calculateRootLogLikelihoods failed: %d\n", rc);
        if (!diagCompareCpu) {
            beagleFinalizeInstance(instance);
            return 1;
        }
    } else {
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // ── Step 9: Check result ────────────────────────────────────────────
        // For a synthetic (non-DNA) model there's no precomputed reference
        // logL to compare against -- just report the value; PASS/FAIL for
        // that case comes entirely from --diag-compare-cpu below.
        printf("logL      = %12.5f\n", logL);
        if (useDnaModel) {
            delta = std::fabs(logL - kRef);
            printf("reference = %12.5f\n", kRef);
            printf("|delta|   = %12.5f  (tolerance %.1f nats)\n", delta, kTol);
        }
        printf("time      = %.3f ms\n", ms);
        if (useDnaModel) printf("\n%s\n", delta < kTol ? "PASS" : "FAIL");
    }

    // ── Step 10 (diagnostic): compare against a CPU-reference instance ───────
    // For a synthetic (non-DNA) model, diagCompareCpu is always forced on
    // (Step 9 skipped its own correctness check above) -- diagCompareCpuOk
    // is this test's *only* correctness verdict in that case, used for the
    // final PASS/FAIL below. Defaults false: if this block can't actually
    // run (no CPU resource, instance creation failure), there's no other
    // check to fall back on for a synthetic model, so don't claim PASS.
    bool diagCompareCpuOk = useDnaModel;
    if (diagCompareCpu) {
        printf("\n=== --diag-compare-cpu: comparing GPU buffers against a CPU-reference instance ===\n");

        int cpuIdx = -1;
        for (int i = 0; i < rList->length; ++i) {
            if (rList->list[i].supportFlags & BEAGLE_FLAG_PROCESSOR_CPU) { cpuIdx = i; break; }
        }
        if (cpuIdx < 0) {
            fprintf(stderr, "DIAGNOSTIC: no CPU resource found, skipping --diag-compare-cpu\n");
        } else {
            BeagleInstanceDetails cpuDet;
            int cpuInstance = beagleCreateInstance(
                3, 5, 0, stateCount, nPatterns, 1, 4, 4, 0,
                &cpuIdx, 1,
                BEAGLE_FLAG_PRECISION_DOUBLE | BEAGLE_FLAG_PROCESSOR_CPU,
                0,
                &cpuDet);
            if (cpuInstance < 0) {
                fprintf(stderr, "DIAGNOSTIC: beagleCreateInstance (CPU) failed: %d\n", cpuInstance);
            } else {
                printf("CPU reference resource %d: %s\n\n", cpuDet.resourceNumber, cpuDet.resourceName);
                setupModel(cpuInstance);
                beagleUpdateTransitionMatrices(cpuInstance, 0, nodeIdx, nullptr, nullptr, edgeLens, 4);
                beagleUpdatePartials(cpuInstance, ops, 2, BEAGLE_OP_NONE);

                // -- transition matrices: kernelMatrixMulADB's output --
                printf("-- transition matrices (4 matrices x stateCount=%d x categoryCount=4) --\n", stateCount);
                bool matricesOk = true;
                std::vector<double> gpuMat((size_t) stateCount * stateCount * 4), cpuMat((size_t) stateCount * stateCount * 4);
                // kernelMatrixMulADB's reduction sums stateCount terms per
                // output element (tiled across BLOCKS=ceil(stateCount/16)
                // passes) -- accumulated single-precision rounding error
                // grows with that, not with state count alone. Fixed at
                // 1e-6 through stateCount=16 (unaffected: that's where it
                // was already calibrated, including the default
                // stateCount=4 DNA/JC69 case); scaled linearly beyond that.
                // Calibrated against real hardware runs (STATUS.md AMD
                // §27ish): observed maxAbsDiff was 3.76e-7 (N=16), 7.85e-7
                // (32), 1.39e-6 (48), 1.45e-6 (64), 3.25e-6 (128) -- this
                // formula keeps ~2-2.5x margin above every one of those.
                double matrixTol = 1e-6 * std::max(1.0, stateCount / 16.0);
                for (int m = 0; m < 4; ++m) {
                    beagleGetTransitionMatrix(instance, m, gpuMat.data());
                    beagleGetTransitionMatrix(cpuInstance, m, cpuMat.data());
                    char label[32];
                    snprintf(label, sizeof(label), "matrix[%d]", m);
                    matricesOk &= compareArrays(label, gpuMat.data(), cpuMat.data(), (int) gpuMat.size(), matrixTol);
                }

                // -- post-peeling partials: kernelPartialsPartialsNoScale's output --
                printf("-- partials (buffer 3 = H*C, buffer 4 = root) --\n");
                bool partialsOk = true;
                std::vector<double> gpuPart((size_t) nPatterns * stateCount * 4), cpuPart((size_t) nPatterns * stateCount * 4);
                for (int buf : {3, 4}) {
                    beagleGetPartials(instance, buf, BEAGLE_OP_NONE, gpuPart.data());
                    beagleGetPartials(cpuInstance, buf, BEAGLE_OP_NONE, cpuPart.data());
                    char label[32];
                    snprintf(label, sizeof(label), "partials[%d]", buf);
                    partialsOk &= compareArrays(label, gpuPart.data(), cpuPart.data(), (int) gpuPart.size(), 1e-6);
                }

                // -- per-site log-likelihoods: kernelIntegrateLikelihoods/kernelSumSites1's output --
                printf("-- site log-likelihoods --\n");
                int cpuRootBuf = 4, cpuWBuf = 0, cpuFBuf = 0, cpuSBuf = BEAGLE_OP_NONE;
                double cpuLogL = 0.0;
                beagleCalculateRootLogLikelihoods(cpuInstance, &cpuRootBuf, &cpuWBuf, &cpuFBuf, &cpuSBuf, 1, &cpuLogL);
                std::vector<double> gpuSite(nPatterns), cpuSite(nPatterns);
                beagleGetSiteLogLikelihoods(instance, gpuSite.data());
                beagleGetSiteLogLikelihoods(cpuInstance, cpuSite.data());
                bool siteOk = compareArrays("siteLogL", gpuSite.data(), cpuSite.data(), nPatterns, 1e-3);

                printf("\nCPU-reference logL = %.5f  (GPU logL = %.5f)\n", cpuLogL, logL);
                printf("First mismatching stage: %s\n",
                       !matricesOk ? "transition matrices (kernelMatrixMulADB)" :
                       !partialsOk ? "partials (kernelPartialsPartialsNoScale)" :
                       !siteOk     ? "site log-likelihoods (kernelIntegrateLikelihoods/kernelSumSites1)" :
                                     "none -- everything matched the CPU reference");

                diagCompareCpuOk = matricesOk && partialsOk && siteOk;
                beagleFinalizeInstance(cpuInstance);
            }
        }
    }

    // ── Step 11 (optional): steady-state benchmark ───────────────────────────
    // The checked evaluation above paid the one-time costs (daemon boot,
    // kernel loads, first dispatch); these repeats don't, so their wall time
    // per evaluation is what an MCMC run sees. Same inputs and kernels every
    // time, so every repeat must reproduce the first logL exactly.
    bool repsOk = true;
    if (reps > 0 && logLOk) {
        std::vector<double> ms(reps);
        int nDiffer = 0;
        double maxDiff = 0.0;
        gEvaluating = true;
        for (int r = 0; r < reps; ++r) {
            if (gStopSignal) {
                fprintf(stderr, "--reps: interrupted by signal %d after %d of %d repeats; finalizing the instance\n",
                        (int)gStopSignal, r, reps);
                repsOk = false;
                break;
            }
            double repLogL = 0.0;
            auto s0 = std::chrono::steady_clock::now();
            int rr = doUpdateTransitionMatrices();
            if (rr >= 0) rr = doUpdatePartials();
            if (rr >= 0) rr = beagleCalculateRootLogLikelihoods(instance, &rootBuf, &wBuf, &fBuf, &sBuf, 1, &repLogL);
            auto s1 = std::chrono::steady_clock::now();
            if (rr < 0) {
                fprintf(stderr, "--reps: evaluation %d failed: %d\n", r, rr);
                repsOk = false;
                break;
            }
            ms[r] = std::chrono::duration<double, std::milli>(s1 - s0).count();
            if (repLogL != logL) {
                ++nDiffer;
                maxDiff = std::max(maxDiff, std::fabs(repLogL - logL));
            }
        }
        if (repsOk) {
            std::vector<double> sorted(ms);
            std::sort(sorted.begin(), sorted.end());
            double sum = 0.0;
            for (double v : ms) sum += v;
            printf("\n=== --reps %d: steady-state evaluations (updateTransitionMatrices + updatePartials + "
                   "calculateRootLogLikelihoods) ===\n", reps);
            printf("per evaluation: min %.3f ms  median %.3f ms  mean %.3f ms  max %.3f ms\n",
                   sorted.front(), sorted[reps / 2], sum / reps, sorted.back());
            if (nDiffer == 0) {
                printf("logL: all %d repeats equal the first evaluation exactly\n", reps);
            } else {
                printf("logL: %d of %d repeats differ from the first evaluation (max |diff| %.3g) -- FAIL\n",
                       nDiffer, reps, maxDiff);
                repsOk = false;
            }
        }
    }

    bool overallOk = (useDnaModel ? (logLOk && delta < kTol) : (logLOk && diagCompareCpuOk)) && repsOk;
    if (!useDnaModel) printf("\n%s\n", overallOk ? "PASS" : "FAIL");

    if (cpuRefInstance >= 0) beagleFinalizeInstance(cpuRefInstance);
    beagleFinalizeInstance(instance);
    return gStopSignal ? 128 + gStopSignal : (overallOk ? 0 : 1);
}
