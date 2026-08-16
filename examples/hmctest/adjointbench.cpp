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
 * Standalone timing benchmark for calculateAdjointCrossProducts (via
 * beagleCalculateAdjointDerivative), configurable in state/pattern/category
 * count so the GPU adjoint-kernel optimization work can be measured rather
 * than reasoned about. Same 3-tip/1-internal-node cherry topology and
 * circulant eigen decomposition as adjointtest4.cpp's runTestN, generalized
 * over --nstates/--npat/--ncat/--reps.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <random>

#include "libhmsbeagle/beagle.h"

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

static int createInstance(bool useGpu, int whichDevice, int stateCount,
                           int nPatterns, int nCats, bool singlePrec) {
    long prefFlags = BEAGLE_FLAG_SCALERS_RAW;
    prefFlags |= useGpu ? BEAGLE_FLAG_PROCESSOR_GPU : BEAGLE_FLAG_PROCESSOR_CPU;
    prefFlags |= singlePrec ? BEAGLE_FLAG_PRECISION_SINGLE : BEAGLE_FLAG_PRECISION_DOUBLE;

    long reqFlags = BEAGLE_FLAG_EIGEN_COMPLEX | BEAGLE_FLAG_PREORDER_TRANSPOSE_AUTO |
                    BEAGLE_FLAG_SPECTRAL_REPRESENTATION;

    BeagleInstanceDetails det;
    int inst = beagleCreateInstance(
        3, 10, 3, stateCount, nPatterns, 1, 4, nCats, 0,
        whichDevice >= 0 ? &whichDevice : NULL,
        whichDevice >= 0 ? 1 : 0,
        prefFlags, reqFlags, &det);

    if (inst < 0) return inst;
    fprintf(stdout, "  [%s] resource %d: %s  impl: %s\n",
            useGpu ? "GPU" : "CPU", det.resourceNumber, det.resourceName, det.implName);
    return inst;
}

int main(int argc, const char *argv[]) {
    int S = 16, nPat = 1000, nCats = 4, reps = 50, gpuDevice = -1;
    int branchMultiplier = 1;
    bool singlePrec = true;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--nstates") == 0 && i+1 < argc) S = atoi(argv[++i]);
        else if (strcmp(argv[i], "--npat") == 0 && i+1 < argc) nPat = atoi(argv[++i]);
        else if (strcmp(argv[i], "--ncat") == 0 && i+1 < argc) nCats = atoi(argv[++i]);
        else if (strcmp(argv[i], "--reps") == 0 && i+1 < argc) reps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--gpu") == 0 && i+1 < argc) gpuDevice = atoi(argv[++i]);
        else if (strcmp(argv[i], "--double") == 0) singlePrec = false;
        else if (strcmp(argv[i], "--branch-multiplier") == 0 && i+1 < argc) branchMultiplier = atoi(argv[++i]);
    }
    fprintf(stdout, "adjointbench: S=%d nPat=%d nCats=%d reps=%d gpu=%d prec=%s branchMultiplier=%d\n",
            S, nPat, nCats, reps, gpuDevice, singlePrec ? "single" : "double", branchMultiplier);

    bool useGpu = (gpuDevice >= 0);
    int inst = createInstance(useGpu, gpuDevice, S, nPat, nCats, singlePrec);
    if (inst < 0) {
        fprintf(stderr, "createInstance failed: %d\n", inst);
        return 1;
    }

    /* Mostly-conserved synthetic tips (ancestral state + rare *nearby*
     * mutation), matching typical phylogenetic/phylogeographic data: nearby
     * states in the circulant chain are reachable with reasonable
     * probability over the branch lengths used below, whereas a uniformly
     * random "teleport" mutation requires transitioning across ~S/2 states,
     * whose probability genuinely underflows to exact 0.0 (even in double
     * precision) for these edge lengths once S and patternCount both grow —
     * that 0-probability pattern then poisons the whole gradient with NaN
     * via `lhsLs * (weight / exp(-inf)) = 0 * inf` in the adjoint kernels'
     * Phase 1 (same failure class as the earlier NaN-masking bug, see
     * STATUS.md). Bounding the mutation to a small neighborhood avoids this
     * without needing BEAGLE_FLAG_SCALERS_RAW rescaling. */
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
    beagleSetTipStates(inst, 0, hSt.data());
    beagleSetTipStates(inst, 1, cSt.data());
    beagleSetTipStates(inst, 2, gSt.data());

    std::vector<double> rates(nCats, 1.0), catWeights(nCats, 1.0 / nCats);
    std::vector<double> freqs(S, 1.0 / S);
    std::vector<double> pw(nPat, 1.0);
    beagleSetCategoryRates(inst, rates.data());
    beagleSetPatternWeights(inst, pw.data());
    beagleSetStateFrequencies(inst, 0, freqs.data());
    beagleSetCategoryWeights(inst, 0, catWeights.data());

    std::vector<double> evec, ivec, eval;
    buildCirculantN(S, evec, ivec, eval);
    beagleSetEigenDecomposition(inst, 0, evec.data(), ivec.data(), eval.data());

    double edgeLengths[4] = { 0.1, 0.1, 0.2, 0.5 };
    int nodeIdx[4] = { 0, 1, 2, 3 };
    beagleUpdateTransitionMatrices(inst, 0, nodeIdx, NULL, NULL, edgeLengths, 4);

    BeagleOperation postOps[2] = {
        { 3, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 0, 0, 1, 1 },
        { 4, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 2, 2, 3, 3 }
    };
    beagleUpdatePartials(inst, postOps, 2, BEAGLE_OP_NONE);

    if (getenv("ADJBENCH_DEBUG_PARTIALS")) {
        std::vector<double> p3((size_t)nCats*nPat*S), p4((size_t)nCats*nPat*S);
        beagleGetPartials(inst, 3, BEAGLE_OP_NONE, p3.data());
        beagleGetPartials(inst, 4, BEAGLE_OP_NONE, p4.data());
        bool nan3=false, nan4=false;
        double min3=1e300,max3=-1e300,min4=1e300,max4=-1e300;
        for (double v : p3) { if (std::isnan(v)) nan3=true; else { min3=std::min(min3,v); max3=std::max(max3,v);} }
        for (double v : p4) { if (std::isnan(v)) nan4=true; else { min4=std::min(min4,v); max4=std::max(max4,v);} }
        fprintf(stdout, "post3: nan=%d range=[%.3e,%.3e]\n", nan3, min3, max3);
        fprintf(stdout, "post4: nan=%d range=[%.3e,%.3e]\n", nan4, min4, max4);
        fprintf(stdout, "post4[cat=0,pat=0,:]:");
        for (int s = 0; s < S; s++) fprintf(stdout, " %.2e", p4[s]);
        fprintf(stdout, "\n");
        fprintf(stdout, "post3[cat=0,pat=0,:]:");
        for (int s = 0; s < S; s++) fprintf(stdout, " %.2e", p3[s]);
        fprintf(stdout, "\n");
    }

    int rootIdx = 4, cwIdx = 0, sfIdx = 0, csi = BEAGLE_OP_NONE;
    double logL = 0.0;
    beagleCalculateRootLogLikelihoods(inst, &rootIdx, &cwIdx, &sfIdx, &csi, 1, &logL);
    fprintf(stdout, "log-likelihood: %f\n", logL);

    std::vector<double> rootPre((size_t)nCats * nPat * S);
    for (int c = 0; c < nCats; c++)
        for (int p = 0; p < nPat; p++)
            for (int s = 0; s < S; s++)
                rootPre[(size_t)c * nPat * S + p * S + s] = freqs[s];
    beagleSetPartials(inst, 5, rootPre.data());

    /* buf6/buf7's parent (5) is the literal root prior -> parentTransMatIndex=NONE.
     * buf8/buf9's parent (6) is internal's own (non-root) pre-order buffer, so it
     * must be propagated through internal's own branch (matrix 3) before combining
     * with the sibling -> parentTransMatIndex=3, NOT NONE. */
    BeagleOperation preOps[4] = {
        { 6, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 5, BEAGLE_OP_NONE, 2, 2 },
        { 7, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 5, BEAGLE_OP_NONE, 3, 3 },
        { 8, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 6, 3, 0, 0 },
        { 9, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 6, 3, 1, 1 }
    };
    beagleUpdatePrePartials_v5(inst, preOps, 4, BEAGLE_OP_NONE, BEAGLE_PARTIALS_TOP);

    /* Base 4-branch operation list from the 3-tip cherry topology above.
     * --branch-multiplier replicates it N times into one larger
     * `calculateAdjointCrossProducts` call (count = 4*N) purely to measure
     * per-call kernel-launch-count overhead (Stage 2/3 of the single-launch
     * redesign — see STATUS.md/TODO.md): the *topology* stays a 3-tip tree,
     * so the summed gradient value is not biologically meaningful at
     * multiplier>1 (each branch's contribution is counted N times), but
     * every replicated (postIdx,preIdx,eigenIdx) triple is a genuinely
     * valid reference into the same buffers, so this exercises the exact
     * same host-side offset-queue-build + kernel-launch code path a real
     * many-branch tree would, without the risk of a hand-rolled generalized
     * pre-order tree builder for this timing-only tool. */
    BeagleBranchOperation baseBranchOps[4] = {
        { 1, 8, 1, 0 },
        { 0, 9, 0, 0 },
        { 2, 7, 2, 0 },
        { 3, 6, 3, 0 }
    };
    const int nBranches = 4 * branchMultiplier;
    std::vector<BeagleBranchOperation> branchOps(nBranches);
    for (int m = 0; m < branchMultiplier; m++)
        for (int k = 0; k < 4; k++)
            branchOps[m*4 + k] = baseBranchOps[k];

    std::vector<double> grad(S * S, 0.0);

    /* Warm-up call (excludes one-time JIT/compile/allocation cost). */
    beagleCalculateAdjointDerivative(inst, branchOps.data(), 0, 0, 4, 0, nBranches, grad.data(), NULL);

    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; r++) {
        beagleCalculateAdjointDerivative(inst, branchOps.data(), 0, 0, 4, 0, nBranches, grad.data(), NULL);
    }
    auto t1 = std::chrono::steady_clock::now();

    double totalMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    fprintf(stdout, "total: %.3f ms over %d reps (%d branches/rep) => %.4f ms/rep, %.4f ms/branch\n",
            totalMs, reps, nBranches, totalMs / reps, totalMs / (reps * nBranches));

    beagleFinalizeInstance(inst);
    return 0;
}
