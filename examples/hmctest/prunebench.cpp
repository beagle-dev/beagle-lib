/*
 *
 * Copyright 2026 Phylogenetic Likelihood Working Group
 *
 * This file is part of BEAGLE.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 *
 * Standalone timing benchmark for the spectral-representation post-order
 * (pruning) and pre-order ("Growing") partial-likelihood kernels, isolated
 * from the (already-optimized, separately-benchmarked in adjointbench.cpp)
 * adjoint cross-product step.
 *
 * The tree is built as `--nreplicas` independent copies of the exact 3-tip
 * cherry unit used by `adjointtest4.cpp`'s `runAdjoint4` (the FD/CPU-vs-GPU
 * *correctness*-verified test), each replica using disjoint buffers. This
 * deliberately avoids hand-deriving a general multi-level pre-order
 * BeagleOperation builder (see STATUS.md findings 6/7 for why that wasn't
 * fully resolved) while still exercising exactly the same per-op host-side
 * dispatch loop / kUsingMultiGrid-merge / launch-count bottleneck a real
 * many-branch tree would, since that machinery only cares about the
 * sequence of independent same-type operations submitted per call, not
 * about "real tree" shape. Same replication idea as adjointbench.cpp's
 * `--branch-multiplier`.
 *
 * Per replica r (0-indexed), buffers (R = --nreplicas):
 *   tips:            0 .. 3R-1            (tip(r,k) = 3r+k, k=0,1,2)
 *   post "node3"(r):  3R+r                (children tip(r,0),tip(r,1))
 *   post "root"(r):   4R+r                (children tip(r,2), node3(r))
 *   pre-order prior:  5R                  (shared frequencies buffer)
 *   pre node3(r):     5R+1+4r+0
 *   pre tip2(r):      5R+1+4r+1
 *   pre tip1(r):      5R+1+4r+2
 *   pre tip0(r):      5R+1+4r+3
 * partialsBufferCount = 9R+1, compactBufferCount = 3R, matrixBufferCount = 4R
 * (matrix 4r+{0,1,2,3} = tip0/tip1/tip2/node3's own edge for replica r).
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <random>

#include "libhmsbeagle/beagle.h"

/* Circulant eigendecomposition, copied from adjointbench.cpp (self-contained,
 * works for any state count including S=4). */
static void buildCirculantN(
    int N,
    std::vector<double>& evec,   /* N×N */
    std::vector<double>& ivec,   /* N×N */
    std::vector<double>& eval)   /* 2N  */
{
    const double r_fwd = 1.0, r_bkd = 0.5;
    const double two_pi_over_N = 2.0 * M_PI / N;
    const bool evenN = (N % 2 == 0);
    const int nPairs = evenN ? N/2 - 1 : (N-1)/2;

    evec.assign(N * N, 0.0);
    ivec.assign(N * N, 0.0);
    eval.assign(2 * N, 0.0);

    eval[0] = 0.0; eval[N] = 0.0;
    for (int j = 0; j < N; j++) {
        evec[j * N + 0] = 1.0 / N;
        ivec[0 * N + j] = 1.0;
    }

    for (int m = 1; m <= nPairs; m++) {
        int col_re = 2*m - 1, col_im = 2*m;
        double theta_m = two_pi_over_N * m;
        double a = (r_fwd + r_bkd) * (cos(theta_m) - 1.0);
        double b = (r_fwd - r_bkd) * sin(theta_m);
        eval[col_re] = a;  eval[N + col_re] =  b;
        eval[col_im] = a;  eval[N + col_im] = -b;
        for (int j = 0; j < N; j++) {
            double theta = theta_m * j;
            evec[j * N + col_re] = cos(theta) / N;
            evec[j * N + col_im] = sin(theta) / N;
            ivec[col_re * N + j] = 2.0 * cos(theta);
            ivec[col_im * N + j] = 2.0 * sin(theta);
        }
    }

    if (evenN) {
        int last = N - 1;
        eval[last] = -2.0 * (r_fwd + r_bkd);  eval[N + last] = 0.0;
        for (int j = 0; j < N; j++) {
            double v = (j % 2 == 0) ? 1.0 : -1.0;
            evec[j * N + last] = v / N;
            ivec[last * N + j] = v;
        }
    }
}

static int createInstance(bool useGpu, int whichDevice, int S, int R,
                           int nPatterns, int nCats, bool singlePrec) {
    long prefFlags = BEAGLE_FLAG_SCALERS_RAW;
    prefFlags |= useGpu ? BEAGLE_FLAG_PROCESSOR_GPU : BEAGLE_FLAG_PROCESSOR_CPU;
    prefFlags |= singlePrec ? BEAGLE_FLAG_PRECISION_SINGLE : BEAGLE_FLAG_PRECISION_DOUBLE;

    long reqFlags = BEAGLE_FLAG_EIGEN_COMPLEX | BEAGLE_FLAG_PREORDER_TRANSPOSE_AUTO |
                    BEAGLE_FLAG_SPECTRAL_REPRESENTATION;

    const int tipCount = 3 * R;
    const int partialsBufferCount = 9 * R + 1;
    const int matrixCount = 4 * R;

    BeagleInstanceDetails det;
    int inst = beagleCreateInstance(
        tipCount, partialsBufferCount, tipCount, S, nPatterns, 1, matrixCount,
        nCats, 0,
        whichDevice >= 0 ? &whichDevice : NULL,
        whichDevice >= 0 ? 1 : 0,
        prefFlags, reqFlags, &det);

    if (inst < 0) return inst;
    fprintf(stdout, "  [%s] resource %d: %s  impl: %s\n",
            useGpu ? "GPU" : "CPU", det.resourceNumber, det.resourceName, det.implName);
    return inst;
}

int main(int argc, const char *argv[]) {
    int S = 4, R = 4, nPat = 1000, nCats = 4, reps = 50, gpuDevice = -1;
    bool singlePrec = true;
    bool doCheck = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--nreplicas") == 0 && i+1 < argc) R = atoi(argv[++i]);
        else if (strcmp(argv[i], "--npat") == 0 && i+1 < argc) nPat = atoi(argv[++i]);
        else if (strcmp(argv[i], "--ncat") == 0 && i+1 < argc) nCats = atoi(argv[++i]);
        else if (strcmp(argv[i], "--reps") == 0 && i+1 < argc) reps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--gpu") == 0 && i+1 < argc) gpuDevice = atoi(argv[++i]);
        else if (strcmp(argv[i], "--double") == 0) singlePrec = false;
        else if (strcmp(argv[i], "--check") == 0) doCheck = true;
    }
    fprintf(stdout, "prunebench: S=%d nreplicas=%d nPat=%d nCats=%d reps=%d gpu=%d prec=%s check=%d\n",
            S, R, nPat, nCats, reps, gpuDevice, singlePrec ? "single" : "double", doCheck);

    const int tipCount = 3 * R;
    const int postNode3Base = tipCount;          // 3R
    const int postRootBase  = tipCount + R;       // 4R
    const int rootPriorBuf  = tipCount + 2 * R;   // 5R
    const int preBase       = rootPriorBuf + 1;   // 5R+1, 4 buffers per replica

    /* Same "mostly-conserved, bounded nearby mutation" tip generator as
     * adjointbench.cpp, to avoid exact-zero-probability underflow as nPat
     * grows (see that file's comment for the failure mode this avoids). */
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> stateDist(0, S - 1);
    std::uniform_real_distribution<double> unitDist(0.0, 1.0);
    std::uniform_int_distribution<int> stepDist(1, std::min(3, S - 1));
    auto nearbyState = [&](int anc) {
        int step = stepDist(rng);
        int dir = (unitDist(rng) < 0.5) ? 1 : -1;
        return ((anc + dir * step) % S + S) % S;
    };
    std::vector<int> hSt(nPat), cSt(nPat), gSt(nPat);
    for (int p = 0; p < nPat; p++) {
        int anc = stateDist(rng);
        hSt[p] = (unitDist(rng) < 0.85) ? anc : nearbyState(anc);
        cSt[p] = (unitDist(rng) < 0.85) ? anc : nearbyState(anc);
        gSt[p] = (unitDist(rng) < 0.85) ? anc : nearbyState(anc);
    }

    std::vector<double> evec, ivec, eval;
    buildCirculantN(S, evec, ivec, eval);
    std::vector<double> rates(nCats, 1.0), catWeights(nCats, 1.0 / nCats);
    std::vector<double> freqs(S, 1.0 / S);
    std::vector<double> pw(nPat, 1.0);

    double baseEdgeLengths[4] = { 0.1, 0.1, 0.2, 0.5 };

    auto setupInstance = [&](int inst) {
        for (int r = 0; r < R; r++) {
            beagleSetTipStates(inst, 3*r + 0, hSt.data());
            beagleSetTipStates(inst, 3*r + 1, cSt.data());
            beagleSetTipStates(inst, 3*r + 2, gSt.data());
        }
        beagleSetCategoryRates(inst, rates.data());
        beagleSetPatternWeights(inst, pw.data());
        beagleSetStateFrequencies(inst, 0, freqs.data());
        beagleSetCategoryWeights(inst, 0, catWeights.data());
        beagleSetEigenDecomposition(inst, 0, evec.data(), ivec.data(), eval.data());

        std::vector<int> nodeIdx(4 * R);
        std::vector<double> edgeLengths(4 * R);
        for (int r = 0; r < R; r++)
            for (int k = 0; k < 4; k++) {
                nodeIdx[4*r + k] = 4*r + k;
                edgeLengths[4*r + k] = baseEdgeLengths[k];
            }
        beagleUpdateTransitionMatrices(inst, 0, nodeIdx.data(), NULL, NULL,
                                        edgeLengths.data(), 4 * R);

        std::vector<double> rootPre((size_t)nCats * nPat * S);
        for (int c = 0; c < nCats; c++)
            for (int p = 0; p < nPat; p++)
                for (int s = 0; s < S; s++)
                    rootPre[(size_t)c * nPat * S + p * S + s] = freqs[s];
        beagleSetPartials(inst, rootPriorBuf, rootPre.data());
    };

    auto buildPostOps = [&](std::vector<BeagleOperation>& ops) {
        ops.resize(2 * R);
        for (int r = 0; r < R; r++) {
            int t0 = 3*r+0, t1 = 3*r+1, t2 = 3*r+2;
            int node3 = postNode3Base + r, root = postRootBase + r;
            int m0 = 4*r+0, m1 = 4*r+1, m2 = 4*r+2, m3 = 4*r+3;
            ops[2*r+0] = { node3, BEAGLE_OP_NONE, BEAGLE_OP_NONE, t0, m0, t1, m1 };
            ops[2*r+1] = { root,  BEAGLE_OP_NONE, BEAGLE_OP_NONE, t2, m2, node3, m3 };
        }
    };

    auto buildPreOps = [&](std::vector<BeagleOperation>& ops) {
        ops.resize(4 * R);
        for (int r = 0; r < R; r++) {
            int t0 = 3*r+0, t1 = 3*r+1, t2 = 3*r+2;
            int node3 = postNode3Base + r;
            int m0 = 4*r+0, m1 = 4*r+1, m2 = 4*r+2, m3 = 4*r+3;
            int preNode3 = preBase + 4*r + 0;
            int preTip2  = preBase + 4*r + 1;
            int preTip1  = preBase + 4*r + 2;
            int preTip0  = preBase + 4*r + 3;
            /* Mirrors adjointtest4.cpp's runAdjoint4 preOps exactly (per
             * replica): child1TransMatIndex = BEAGLE_OP_NONE throughout,
             * matching that FD-verified test's convention (see STATUS.md
             * findings 6/7 for why this, rather than a hand-derived general
             * rule, is what's used here). */
            ops[4*r+0] = { preNode3, BEAGLE_OP_NONE, BEAGLE_OP_NONE, rootPriorBuf, BEAGLE_OP_NONE, t2, m2 };
            ops[4*r+1] = { preTip2,  BEAGLE_OP_NONE, BEAGLE_OP_NONE, rootPriorBuf, BEAGLE_OP_NONE, node3, m3 };
            ops[4*r+2] = { preTip1,  BEAGLE_OP_NONE, BEAGLE_OP_NONE, preNode3, BEAGLE_OP_NONE, t0, m0 };
            ops[4*r+3] = { preTip0,  BEAGLE_OP_NONE, BEAGLE_OP_NONE, preNode3, BEAGLE_OP_NONE, t1, m1 };
        }
    };

    bool useGpu = (gpuDevice >= 0);
    int inst = createInstance(useGpu, gpuDevice, S, R, nPat, nCats, singlePrec);
    if (inst < 0) {
        fprintf(stderr, "createInstance failed: %d\n", inst);
        return 1;
    }
    setupInstance(inst);

    std::vector<BeagleOperation> postOps, preOps;
    buildPostOps(postOps);
    buildPreOps(preOps);

    /* BEAGLE computation defaults to synchronous (BEAGLE_FLAG_COMPUTATION_SYNCH
     * unless ASYNCH is requested, BeagleGPUImpl.hpp:477-481), but the
     * underlying OpenCL kernel enqueue can still return before the device
     * has actually finished (confirmed: a 2,000,000-pattern, 1-replica
     * post-order call timed at ~0.004ms total regardless of pattern count
     * up to 50,000,000 — physically impossible for the data volume moved —
     * see STATUS.md). `beagleWaitForPartials` does NOT fix this: it is an
     * unimplemented no-op in the GPU backend
     * (`BeagleGPUImpl.hpp:2820-2832`, parameters commented out, straight
     * `return BEAGLE_SUCCESS`). The only thing confirmed to force a real
     * device-completion wait is a blocking data readback
     * (`beagleGetPartials`, which is how `--check` gets correct values back)
     * — so that's what's used here as the sync point: read back a single
     * buffer (the last one enqueued this call) after each timed update,
     * relying on the OpenCL command queue being in-order (default; BEAGLE
     * only uses multiple/concurrent queues under
     * `BEAGLE_FLAG_PARALLELOPS_STREAMS`, not requested here) so that a
     * blocking read of the last-enqueued buffer also waits for everything
     * enqueued before it in the same call. This adds one buffer's readback
     * cost per rep (unavoidable with the public API) on top of real compute
     * time — a bounded, per-call (not per-replica) cost. */
    std::vector<double> syncBuf((size_t)nCats * nPat * S);
    int postSyncBuf = postOps.back().destinationPartials;
    int preSyncBuf  = preOps.back().destinationPartials;

    /* Warm-up (excludes one-time JIT/compile/allocation cost), then time
     * post-order and pre-order separately. */
    beagleUpdatePartials(inst, postOps.data(), (int)postOps.size(), BEAGLE_OP_NONE);
    beagleGetPartials(inst, postSyncBuf, BEAGLE_OP_NONE, syncBuf.data());
    beagleUpdatePrePartials_v5(inst, preOps.data(), (int)preOps.size(), BEAGLE_OP_NONE, BEAGLE_PARTIALS_TOP);
    beagleGetPartials(inst, preSyncBuf, BEAGLE_OP_NONE, syncBuf.data());

    auto t0 = std::chrono::steady_clock::now();
    for (int rep = 0; rep < reps; rep++) {
        beagleUpdatePartials(inst, postOps.data(), (int)postOps.size(), BEAGLE_OP_NONE);
        beagleGetPartials(inst, postSyncBuf, BEAGLE_OP_NONE, syncBuf.data());
    }
    auto t1 = std::chrono::steady_clock::now();
    double postMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    auto t2 = std::chrono::steady_clock::now();
    for (int rep = 0; rep < reps; rep++) {
        beagleUpdatePrePartials_v5(inst, preOps.data(), (int)preOps.size(), BEAGLE_OP_NONE, BEAGLE_PARTIALS_TOP);
        beagleGetPartials(inst, preSyncBuf, BEAGLE_OP_NONE, syncBuf.data());
    }
    auto t3 = std::chrono::steady_clock::now();
    double preMs = std::chrono::duration<double, std::milli>(t3 - t2).count();

    int nPostOps = (int)postOps.size(), nPreOps = (int)preOps.size();
    fprintf(stdout, "post-order: total %.3f ms over %d reps (%d ops/rep) => %.4f ms/rep, %.5f ms/op\n",
            postMs, reps, nPostOps, postMs / reps, postMs / (reps * nPostOps));
    fprintf(stdout, "pre-order:  total %.3f ms over %d reps (%d ops/rep) => %.4f ms/rep, %.5f ms/op\n",
            preMs, reps, nPreOps, preMs / reps, preMs / (reps * nPreOps));

    if (doCheck) {
        int cpuInst = createInstance(false, -1, S, R, nPat, nCats, false);
        if (cpuInst < 0) {
            fprintf(stderr, "CPU createInstance failed: %d\n", cpuInst);
        } else {
            setupInstance(cpuInst);
            beagleUpdatePartials(cpuInst, postOps.data(), (int)postOps.size(), BEAGLE_OP_NONE);
            beagleUpdatePrePartials_v5(cpuInst, preOps.data(), (int)preOps.size(), BEAGLE_OP_NONE, BEAGLE_PARTIALS_TOP);

            double maxAbsDiff = 0.0;
            bool anyNan = false;
            std::vector<double> gpuBuf((size_t)nCats * nPat * S), cpuBuf((size_t)nCats * nPat * S);
            for (int r = 0; r < R; r++) {
                int buffersToCheck[3] = { postRootBase + r, preBase + 4*r + 2, preBase + 4*r + 3 };
                for (int bi = 0; bi < 3; bi++) {
                    beagleGetPartials(inst, buffersToCheck[bi], BEAGLE_OP_NONE, gpuBuf.data());
                    beagleGetPartials(cpuInst, buffersToCheck[bi], BEAGLE_OP_NONE, cpuBuf.data());
                    for (size_t i = 0; i < gpuBuf.size(); i++) {
                        double diff = fabs(gpuBuf[i] - cpuBuf[i]);
                        if (std::isnan(diff)) anyNan = true;
                        else if (diff > maxAbsDiff) maxAbsDiff = diff;
                    }
                }
            }
            fprintf(stdout, "check: GPU vs CPU (spectral, double ref) max|diff| over %d replicas' "
                            "root/pre buffers = %.3e  nan=%d  %s\n",
                    R, maxAbsDiff, anyNan,
                    (!anyNan && maxAbsDiff < 1e-3) ? "OK" : "MISMATCH");
            beagleFinalizeInstance(cpuInst);
        }
    }

    beagleFinalizeInstance(inst);
    return 0;
}
