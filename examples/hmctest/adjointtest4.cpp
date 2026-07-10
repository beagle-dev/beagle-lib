/*
 * adjointtest4.cpp
 *
 * Tests beagleCalculateAdjointCrossProductDerivative comparing CPU (double) vs
 * GPU/OpenCL (single) for:
 *   (a) 4-state JC69 model
 *   (b) 16-state asymmetric circulant model (complex eigenvalues, r_fwd=1.0, r_bkd=0.5)
 *   (c) 17-state asymmetric circulant model — GPU uses 32-state kernel
 *       (PADDED_STATE_COUNT=32, BLOCK_PEELING_SIZE=8 → 4 block-peel passes)
 *
 * For each case, post-order and pre-order-TOP partials are compared first, then
 * the full adjoint gradient.
 *
 * Usage:
 *   adjointtest4 [--gpu <device>] [--nstates 4|16|17]
 *
 * Tree: ((human:0.1, chimp:0.1):0.5, gorilla:0.2)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include "libhmsbeagle/beagle.h"

/* ── 4-state JC69 model ────────────────────────────────────────────────── */

static const char *human   = "GAGT";
static const char *chimp   = "GAGG";
static const char *gorilla = "AAAT";

static int* dnaToStates(const char *seq) {
    int n = strlen(seq);
    int *s = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        switch (seq[i]) {
            case 'A': s[i] = 0; break;
            case 'C': s[i] = 1; break;
            case 'G': s[i] = 2; break;
            case 'T': s[i] = 3; break;
            default:  s[i] = 4; break;
        }
    }
    return s;
}

/* JC69 eigenvectors, inverse eigenvectors, eigenvalues */
static double jcEvec[16] = {
     1.0,  2.0,  0.0,  0.5,
     1.0, -2.0,  0.5,  0.0,
     1.0,  2.0,  0.0, -0.5,
     1.0, -2.0, -0.5,  0.0
};
static double jcIvec[16] = {
     0.25,  0.25,  0.25,  0.25,
     0.125, -0.125,  0.125, -0.125,
     0.0,  1.0,  0.0, -1.0,
     1.0,  0.0, -1.0,  0.0
};
/* { real parts × 4, imaginary parts × 4 } */
static double jcEval[8] = {
    0.0, -1.3333333333333333, -1.3333333333333333, -1.3333333333333333,
    0.0, 0.0, 0.0, 0.0
};

static double freqs4[4]       = { 0.25, 0.25, 0.25, 0.25 };
static double rates2[2]       = { 0.5, 1.5 };
static double catWeights2[2]  = { 0.5, 0.5 };
static double edgeLengths[4]  = { 0.1, 0.1, 0.2, 0.5 };

/* ── Asymmetric circulant N-state model (complex eigenvalues) ──────────── */
/*
 * Rate matrix: Q[i,(i+1)%N]=r_fwd, Q[i,(i-1)%N]=r_bkd, Q[i,i]=-(r_fwd+r_bkd)
 * with r_fwd=1.0, r_bkd=0.5.  Non-reversible → most eigenvalues are complex.
 *
 * Eigenvalue k: λ_k = (r_fwd+r_bkd)*(cos(2πk/N)-1) + i*(r_fwd-r_bkd)*sin(2πk/N)
 * Real eigenvalues: k=0 always; k=N/2 when N is even.
 *
 * BEAGLE ordering (requires BEAGLE_FLAG_EIGEN_COMPLEX):
 *   index 0:           k=0, real
 *   indices 2m-1, 2m:  k=m conjugate pair  (m=1..(N/2-1) even N, or 1..(N-1)/2 odd N)
 *   index N-1:         k=N/2, real          (even N only)
 *
 * DFT eigenvectors: U[j,k] = (1/N)*exp(2πijk/N)
 * Inverse:          U^{-1}[k,j] = exp(-2πijk/N)
 *
 * BEAGLE real representation for complex pair m:
 *   Evec[j, 2m-1] = (1/N)*cos(2πjm/N)   Ievc[2m-1, j] = 2*cos(2πjm/N)
 *   Evec[j, 2m  ] = (1/N)*sin(2πjm/N)   Ievc[2m,   j] = 2*sin(2πjm/N)
 */
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

    /* index 0: k=0, λ=0 */
    eval[0] = 0.0; eval[N] = 0.0;
    for (int j = 0; j < N; j++) {
        evec[j * N + 0] = 1.0 / N;
        ivec[0 * N + j] = 1.0;
    }

    /* indices 2m-1, 2m: complex conjugate pairs k=1..nPairs */
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

    /* index N-1: k=N/2, real (even N only) */
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

/* ── Utility: download partials as double ─────────────────────────────── */

static std::vector<double> getPartials(int inst, int buf, int nCats, int nPat, int S) {
    std::vector<double> out((size_t)nCats * nPat * S, 0.0);
    beagleGetPartials(inst, buf, BEAGLE_OP_NONE, out.data());
    return out;
}

/* ── Compare and report partial buffer differences ────────────────────── */

static bool comparePartials(const char *label, const std::vector<double> &ref,
                             const std::vector<double> &tst, int nCats, int nPat, int S,
                             double tol, bool verbose) {
    bool ok = true;
    int mismatches = 0;
    double maxAbsDiff = 0.0;
    for (int c = 0; c < nCats; c++) {
        for (int p = 0; p < nPat; p++) {
            for (int s = 0; s < S; s++) {
                size_t idx = (size_t)c * nPat * S + p * S + s;
                double diff = fabs(ref[idx] - tst[idx]);
                if (diff > tol || std::isnan(diff)) { ok = false; mismatches++; }
                if (diff > maxAbsDiff || std::isnan(diff)) maxAbsDiff = diff;
            }
        }
    }
    fprintf(stdout, "  %-35s max|diff|=%.3e  %s\n",
            label, maxAbsDiff, ok ? "OK" : "MISMATCH");
    if (!ok && verbose) {
        fprintf(stdout, "    Ref vs Test (cat=0, first mismatches):\n");
        int shown = 0;
        for (int p = 0; p < nPat && shown < 4; p++) {
            for (int s = 0; s < S && shown < 4; s++) {
                size_t idx = (size_t)0 * nPat * S + p * S + s;
                double diff = fabs(ref[idx] - tst[idx]);
                if (diff > tol || std::isnan(diff)) {
                    fprintf(stdout, "    [cat=0,pat=%d,state=%d]: ref=%10.6f tst=%10.6f diff=%+.3e\n",
                            p, s, ref[idx], tst[idx], ref[idx]-tst[idx]);
                    shown++;
                }
            }
        }
    }
    return ok;
}

/* ── resource listing ────────────────────────────────────────────────── */

static void printResources() {
    BeagleResourceList *rList = beagleGetResourceList();
    fprintf(stdout, "Available resources:\n");
    for (int i = 0; i < rList->length; i++)
        fprintf(stdout, "  [%d] %s\n", i, rList->list[i].name);
    fprintf(stdout, "\n");
}

/* ── create one BEAGLE instance ──────────────────────────────────────── */

static int createInstance(bool useGpu, int whichDevice, bool singlePrec,
                          int stateCount, int nPatterns, int nCats,
                          bool complexEigen = false) {
    long prefFlags = BEAGLE_FLAG_SCALERS_RAW;
    prefFlags |= useGpu ? BEAGLE_FLAG_PROCESSOR_GPU : BEAGLE_FLAG_PROCESSOR_CPU;
    prefFlags |= singlePrec ? BEAGLE_FLAG_PRECISION_SINGLE : BEAGLE_FLAG_PRECISION_DOUBLE;

    long eigenFlag = complexEigen ? BEAGLE_FLAG_EIGEN_COMPLEX : BEAGLE_FLAG_EIGEN_REAL;
    long reqFlags = eigenFlag | BEAGLE_FLAG_PREORDER_TRANSPOSE_AUTO |
                    BEAGLE_FLAG_SPECTRAL_REPRESENTATION;

    BeagleInstanceDetails det;
    int inst = beagleCreateInstance(
        3,          /* nTips */
        10,         /* nPartials */
        3,          /* nCompact */
        stateCount,
        nPatterns,
        1,          /* nEigenDecomp */
        4,          /* nTransitionMatrices */
        nCats,
        0,          /* nScaleBuffers */
        whichDevice >= 0 ? &whichDevice : NULL,
        whichDevice >= 0 ? 1 : 0,
        prefFlags, reqFlags, &det);

    if (inst < 0) return inst;
    fprintf(stdout, "  [%s] resource %d: %s  impl: %s\n",
            useGpu ? "GPU" : "CPU",
            det.resourceNumber, det.resourceName, det.implName);
    return inst;
}

/* ═══════════════════════════════════════════════════════════════════════
 * 4-state test (JC69)
 * ═══════════════════════════════════════════════════════════════════════*/

struct AdjointResult4 {
    double logL;
    std::vector<double> grad;       /* stateCount*stateCount */
    std::vector<double> pre[5];     /* pre-order bufs 5..9 */
    std::vector<double> post[2];    /* post-order bufs 3, 4 */
};

static AdjointResult4 runAdjoint4(int instance) {
    const int S = 4, nPat = 4, nCats = 2;
    AdjointResult4 R;
    R.grad.resize(S * S, 0.0);

    int *hSt = dnaToStates(human);
    int *cSt = dnaToStates(chimp);
    int *gSt = dnaToStates(gorilla);
    beagleSetTipStates(instance, 0, hSt);
    beagleSetTipStates(instance, 1, cSt);
    beagleSetTipStates(instance, 2, gSt);
    free(hSt); free(cSt); free(gSt);

    beagleSetCategoryRates(instance, rates2);
    std::vector<double> pw(nPat, 1.0);
    beagleSetPatternWeights(instance, pw.data());
    beagleSetStateFrequencies(instance, 0, freqs4);
    beagleSetCategoryWeights(instance, 0, catWeights2);
    beagleSetEigenDecomposition(instance, 0, jcEvec, jcIvec, jcEval);

    int nodeIdx[4] = { 0, 1, 2, 3 };
    beagleUpdateTransitionMatrices(instance, 0, nodeIdx, NULL, NULL, edgeLengths, 4);

    BeagleOperation postOps[2] = {
        { 3, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 0, 0, 1, 1 },
        { 4, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 2, 2, 3, 3 }
    };
    beagleUpdatePartials(instance, postOps, 2, BEAGLE_OP_NONE);

    /* Download post-order partials for comparison */
    R.post[0] = getPartials(instance, 3, nCats, nPat, S);
    R.post[1] = getPartials(instance, 4, nCats, nPat, S);

    int rootIdx = 4, cwIdx = 0, sfIdx = 0, csi = BEAGLE_OP_NONE;
    beagleCalculateRootLogLikelihoods(instance, &rootIdx, &cwIdx, &sfIdx, &csi, 1, &R.logL);

    /* Root pre-order (stationary frequencies) */
    std::vector<double> rootPre((size_t)nCats * nPat * S);
    for (int c = 0; c < nCats; c++)
        for (int p = 0; p < nPat; p++)
            for (int s = 0; s < S; s++)
                rootPre[(size_t)c * nPat * S + p * S + s] = freqs4[s];
    beagleSetPartials(instance, 5, rootPre.data());

    BeagleOperation preOps[4] = {
        { 6, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 5, BEAGLE_OP_NONE, 2, 2 },
        { 7, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 5, BEAGLE_OP_NONE, 3, 3 },
        { 8, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 6, BEAGLE_OP_NONE, 0, 0 },
        { 9, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 6, BEAGLE_OP_NONE, 1, 1 }
    };
    beagleUpdatePrePartials_v5(instance, preOps, 4, BEAGLE_OP_NONE, BEAGLE_PARTIALS_TOP);

    /* Download pre-order partials for comparison (bufs 5..9 → indices 0..4) */
    for (int b = 0; b < 5; b++)
        R.pre[b] = getPartials(instance, 5 + b, nCats, nPat, S);

    BeagleBranchOperation branchOps[4] = {
        { 1, 8, 1, 0 },
        { 0, 9, 0, 0 },
        { 2, 7, 2, 0 },
        { 3, 6, 3, 0 }
    };

    double *grad = R.grad.data();
    beagleCalculateAdjointDerivative(instance, branchOps, 0, 0, 4, 0, 4, grad, NULL);

    /* Per-branch gradient printout */
    const char* branchNames[4] = { "chimp", "human", "gorilla", "internal" };
    for (int b = 0; b < 4; b++) {
        std::vector<double> gb(S * S, 0.0);
        beagleCalculateAdjointDerivative(instance, &branchOps[b], 0, 0, 4, 0, 1, gb.data(), NULL);
        fprintf(stdout, "  Branch %s gradient:\n", branchNames[b]);
        for (int i = 0; i < S; i++) {
            for (int j = 0; j < S; j++)
                fprintf(stdout, " %10.6f", gb[i*S+j]);
            fprintf(stdout, "\n");
        }
    }

    return R;
}

/* ── Host-side reference adjoint for one branch ─────────────────────── */

/* Computes adjoint cross-product for ONE branch in double, using downloaded
 * pre-order partials and either tip states (isStates) or post-order partials.
 * Adds result into grad[S*S].
 * evec[i*S+j] = U[i,j], ivec[i*S+j] = U^{-1}[i,j], evals[s] = λ_s
 * Lk[k] = total per-site likelihood (summed over categories). */
static void branchAdjointRef(
    const std::vector<double>& pre,      /* pre-order [nCats * nPat * S] */
    const std::vector<double>& post,     /* post-order partials [nCats*nPat*S] or empty */
    const int* tipStates,                /* tip states [nPat] or nullptr */
    int S, int nPat, int nCats,
    const double* evec, const double* ivec, const double* evals,
    const double* catWeights, double edgeLen, const double* catRates,
    const std::vector<double>& Lk,
    std::vector<double>& grad) {

    for (int c = 0; c < nCats; c++) {
        double t = edgeLen * catRates[c];
        double catW = catWeights[c];

        /* Precompute exp(λ_s * t) */
        std::vector<double> expat(S);
        for (int s = 0; s < S; s++) expat[s] = exp(evals[s] * t);

        for (int k = 0; k < nPat; k++) {
            const double* pre_ck = &pre[(size_t)c * nPat * S + k * S];

            /* lhs[ls] = sum_j U[j,ls] * pre_ck[j]  (= (U^T · pre)[ls]) */
            std::vector<double> lhs(S, 0.0);
            for (int ls = 0; ls < S; ls++)
                for (int j = 0; j < S; j++)
                    lhs[ls] += evec[j * S + ls] * pre_ck[j];

            /* rhs[rs]: either column of U^{-1} (tip states) or U^{-1}·post */
            std::vector<double> rhs(S, 0.0);
            if (tipStates) {
                int st = tipStates[k];
                for (int rs = 0; rs < S; rs++)
                    rhs[rs] = ivec[rs * S + st];
            } else {
                const double* post_ck = &post[(size_t)c * nPat * S + k * S];
                for (int rs = 0; rs < S; rs++)
                    for (int j = 0; j < S; j++)
                        rhs[rs] += ivec[rs * S + j] * post_ck[j];
            }

            double scale = catW / Lk[k]; /* patternWeight = 1.0 */

            for (int ls = 0; ls < S; ls++) {
                double la = evals[ls], ea = expat[ls];
                for (int rs = 0; rs < S; rs++) {
                    double lb = evals[rs], eb = expat[rs];
                    double coeff = (fabs(la - lb) * t < 1e-12)
                        ? t * ea : (ea - eb) / (la - lb);
                    grad[ls * S + rs] += lhs[ls] * rhs[rs] * scale * coeff;
                }
            }
        }
    }
}

/* Computes full reference adjoint gradient for 4-state test using
 * already-downloaded partials from any instance (CPU or GPU data).
 * Branch layout matches branchOps in runAdjoint4. */
static std::vector<double> computeRefGrad4(const AdjointResult4& R) {
    const int S = 4, nPat = 4, nCats = 2;

    /* Per-site likelihood from root post-order (buf4 = R.post[1]) */
    std::vector<double> Lk(nPat, 0.0);
    for (int c = 0; c < nCats; c++)
        for (int k = 0; k < nPat; k++)
            for (int s = 0; s < S; s++)
                Lk[k] += catWeights2[c] * freqs4[s]
                        * R.post[1][(size_t)c * nPat * S + k * S + s];

    /* Tip states (stored in branchOps post buffers 1,0,2 = chimp,human,gorilla) */
    int* cSt = dnaToStates(chimp);
    int* hSt = dnaToStates(human);
    int* gSt = dnaToStates(gorilla);

    std::vector<double> grad(S * S, 0.0);

    /* Branch order matches branchOps: chimp(1,8,t=0.1), human(0,9,t=0.1),
     * gorilla(2,7,t=0.2), internal(3,6,t=0.5). */
    double catRates[2] = { rates2[0], rates2[1] };
    /* chimp: post=states(cSt), pre=buf8=R.pre[3], t=0.1 */
    branchAdjointRef(R.pre[3], {}, cSt, S, nPat, nCats,
        jcEvec, jcIvec, jcEval, catWeights2, edgeLengths[1], catRates, Lk, grad);
    /* human: post=states(hSt), pre=buf9=R.pre[4], t=0.1 */
    branchAdjointRef(R.pre[4], {}, hSt, S, nPat, nCats,
        jcEvec, jcIvec, jcEval, catWeights2, edgeLengths[0], catRates, Lk, grad);
    /* gorilla: post=states(gSt), pre=buf7=R.pre[2], t=0.2 */
    branchAdjointRef(R.pre[2], {}, gSt, S, nPat, nCats,
        jcEvec, jcIvec, jcEval, catWeights2, edgeLengths[2], catRates, Lk, grad);
    /* internal: post=partials(buf3=R.post[0]), pre=buf6=R.pre[1], t=0.5 */
    branchAdjointRef(R.pre[1], R.post[0], nullptr, S, nPat, nCats,
        jcEvec, jcIvec, jcEval, catWeights2, edgeLengths[3], catRates, Lk, grad);

    free(cSt); free(hSt); free(gSt);
    return grad;
}

/* ── Run and compare 4-state ─────────────────────────────────────────── */

static int runTest4(int gpuDevice) {
    const int S = 4, nPat = 4, nCats = 2;
    const double tol = 1e-4;

    fprintf(stdout, "\n=== 4-state JC69 test ===\n");

    fprintf(stdout, "\nCreating CPU (double) instance:\n");
    int cpuInst = createInstance(false, -1, false, S, nPat, nCats);
    if (cpuInst < 0) { fprintf(stderr, "Failed CPU instance\n"); return 1; }

    AdjointResult4 cpuR = runAdjoint4(cpuInst);
    beagleFinalizeInstance(cpuInst);

    fprintf(stdout, "\nCPU log-likelihood: %.6f\n", cpuR.logL);
    fprintf(stdout, "CPU adjoint gradient (%dx%d):\n", S, S);
    for (int i = 0; i < S; i++) {
        for (int j = 0; j < S; j++) fprintf(stdout, " %10.6f", cpuR.grad[i*S+j]);
        fprintf(stdout, "\n");
    }

    if (gpuDevice < 0) return 0;

    fprintf(stdout, "\nCreating GPU (single) instance:\n");
    int gpuInst = createInstance(true, gpuDevice, true, S, nPat, nCats);
    if (gpuInst < 0) { fprintf(stderr, "Failed GPU instance\n"); return 1; }

    AdjointResult4 gpuR = runAdjoint4(gpuInst);
    beagleFinalizeInstance(gpuInst);

    fprintf(stdout, "\nGPU log-likelihood: %.6f\n", gpuR.logL);
    fprintf(stdout, "GPU adjoint gradient (%dx%d):\n", S, S);
    for (int i = 0; i < S; i++) {
        for (int j = 0; j < S; j++) fprintf(stdout, " %10.6f", gpuR.grad[i*S+j]);
        fprintf(stdout, "\n");
    }

    /* ── Partial buffer comparisons ── */
    fprintf(stdout, "\n--- Partial buffer comparisons (tol=%.0e) ---\n", tol);
    bool allOk = true;
    const char *postNames[2] = { "post-order buf3 (internal node)", "post-order buf4 (root)" };
    for (int b = 0; b < 2; b++)
        allOk &= comparePartials(postNames[b], cpuR.post[b], gpuR.post[b], nCats, nPat, S, tol, true);

    const char *preNames[5] = {
        "pre-order buf5 (root, set direct)",
        "pre-order buf6 (internal node)",
        "pre-order buf7 (gorilla)",
        "pre-order buf8 (chimp)",
        "pre-order buf9 (human)"
    };
    for (int b = 0; b < 5; b++)
        allOk &= comparePartials(preNames[b], cpuR.pre[b], gpuR.pre[b], nCats, nPat, S, tol, true);

    /* ── Host reference using CPU's downloaded partials (per-branch) ── */
    int* cSt2 = dnaToStates(chimp);
    int* hSt2 = dnaToStates(human);
    int* gSt2 = dnaToStates(gorilla);
    double catRates4[2] = { rates2[0], rates2[1] };
    std::vector<double> Lk4(nPat, 0.0);
    for (int c = 0; c < nCats; c++)
        for (int k = 0; k < nPat; k++)
            for (int s = 0; s < S; s++)
                Lk4[k] += catWeights2[c] * freqs4[s]
                    * cpuR.post[1][(size_t)c * nPat * S + k * S + s];

    const char* refBranch[4] = { "chimp", "human", "gorilla", "internal" };
    const std::vector<double>* refPre[4]  = { &cpuR.pre[3], &cpuR.pre[4], &cpuR.pre[2], &cpuR.pre[1] };
    const std::vector<double>* refPost[4] = { nullptr, nullptr, nullptr, &cpuR.post[0] };
    const int* refStates[4] = { cSt2, hSt2, gSt2, nullptr };
    double refEdge[4] = { edgeLengths[1], edgeLengths[0], edgeLengths[2], edgeLengths[3] };
    for (int b = 0; b < 4; b++) {
        std::vector<double> gb(S * S, 0.0);
        branchAdjointRef(*refPre[b],
            refPost[b] ? *refPost[b] : std::vector<double>{},
            refStates[b], S, nPat, nCats,
            jcEvec, jcIvec, jcEval, catWeights2, refEdge[b], catRates4, Lk4, gb);
        fprintf(stdout, "  Ref branch %s gradient [0,1]=%.6f [1,0]=%.6f [1,1]=%.6f\n",
                refBranch[b], gb[0*S+1], gb[1*S+0], gb[1*S+1]);
    }
    free(cSt2); free(hSt2); free(gSt2);

    std::vector<double> refGradCPU = computeRefGrad4(cpuR);
    std::vector<double> refGradGPU = computeRefGrad4(gpuR);

    fprintf(stdout, "\nRef gradient (double, from CPU partials):\n");
    for (int i = 0; i < S; i++) {
        for (int j = 0; j < S; j++) fprintf(stdout, " %10.6f", refGradCPU[i*S+j]);
        fprintf(stdout, "\n");
    }
    fprintf(stdout, "CPU kernel vs ref-CPU (should be ~0):\n");
    for (int i = 0; i < S; i++) {
        for (int j = 0; j < S; j++)
            fprintf(stdout, " %+10.2e", cpuR.grad[i*S+j] - refGradCPU[i*S+j]);
        fprintf(stdout, "\n");
    }
    fprintf(stdout, "GPU kernel vs ref-GPU (double ref from GPU partials):\n");
    for (int i = 0; i < S; i++) {
        for (int j = 0; j < S; j++)
            fprintf(stdout, " %+10.2e", gpuR.grad[i*S+j] - refGradGPU[i*S+j]);
        fprintf(stdout, "\n");
    }

    /* ── Gradient comparison ── */
    fprintf(stdout, "\nGradient comparison (tol=%.0e): %s\n", tol,
            [&]{ for (int k=0;k<S*S;k++) { double d = fabs(cpuR.grad[k]-gpuR.grad[k]); if (d > tol || std::isnan(d)) return "FAIL"; } return "PASS"; }());
    fprintf(stdout, "Differences (CPU - GPU):\n");
    for (int i = 0; i < S; i++) {
        for (int j = 0; j < S; j++)
            fprintf(stdout, " %+10.2e", cpuR.grad[i*S+j] - gpuR.grad[i*S+j]);
        fprintf(stdout, "\n");
    }

    bool gradOk = true;
    for (int k = 0; k < S*S; k++) {
        double d = fabs(cpuR.grad[k] - gpuR.grad[k]);
        if (d > tol || std::isnan(d)) { gradOk = false; break; }
    }
    return (allOk && gradOk) ? 0 : 1;
}

/* ═══════════════════════════════════════════════════════════════════════
 * 16-state test (asymmetric circulant, complex eigenvalues)
 * ═══════════════════════════════════════════════════════════════════════*/

static int runTest16(int gpuDevice) {
    const int S = 16, nPat = 4, nCats = 2;
    const double tol = 1e-3;   /* single vs double tolerance */

    fprintf(stdout, "\n=== 16-state asymmetric circulant test (complex eigenvalues) ===\n");
    fprintf(stdout, "    r_fwd=1.0, r_bkd=0.5 → 14 of 16 eigenvalues have non-zero imaginary parts\n");

    /* Build eigenvectors / inverse / eigenvalues for the asymmetric circulant:
     *   Q[i,(i+1)%16]=1.0,  Q[i,(i-1)%16]=0.5,  Q[i,i]=-1.5
     * Eigenvalues indexed as: 0 (real), 1,2 (pair k=1), ..., 13,14 (pair k=7), 15 (real) */
    std::vector<double> evec16, ivec16, eval16;
    buildCirculantN(16, evec16, ivec16, eval16);

    /* Print eigenvalues summary */
    fprintf(stdout, "  Eigenvalue real parts: ");
    for (int k = 0; k < S; k++) fprintf(stdout, "%.3f ", eval16[k]);
    fprintf(stdout, "\n  Eigenvalue imag parts: ");
    for (int k = 0; k < S; k++) fprintf(stdout, "%.3f ", eval16[S + k]);
    fprintf(stdout, "\n");

    /* Uniform frequencies (stationary distribution of the asymmetric circulant) */
    std::vector<double> freqs16(S, 1.0 / S);

    /* Rate categories (same as 4-state test) */
    double rates16[2]      = { 0.5, 1.5 };
    double catWts16[2]     = { 0.5, 0.5 };
    double edgeLens16[4]   = { 0.1, 0.1, 0.2, 0.5 };

    /* Sequences: use states 0-15 */
    int hSt16[4] = { 3, 0, 5, 7 };
    int cSt16[4] = { 3, 0, 5, 5 };
    int gSt16[4] = { 0, 0, 0, 7 };

    /* ── Lambda to create and run one instance ── */
    auto runOne = [&](bool useGpu, int dev, bool singlePrec) -> std::pair<double, std::vector<double>> {
        int inst = createInstance(useGpu, dev, singlePrec, S, nPat, nCats, /*complexEigen=*/true);
        if (inst < 0) return { 0.0, {} };

        beagleSetTipStates(inst, 0, hSt16);
        beagleSetTipStates(inst, 1, cSt16);
        beagleSetTipStates(inst, 2, gSt16);

        beagleSetCategoryRates(inst, rates16);
        std::vector<double> pw(nPat, 1.0);
        beagleSetPatternWeights(inst, pw.data());
        beagleSetStateFrequencies(inst, 0, freqs16.data());
        beagleSetCategoryWeights(inst, 0, catWts16);
        beagleSetEigenDecomposition(inst, 0, evec16.data(), ivec16.data(), eval16.data());

        int nodeIdx[4] = { 0, 1, 2, 3 };
        beagleUpdateTransitionMatrices(inst, 0, nodeIdx, NULL, NULL, edgeLens16, 4);

        BeagleOperation postOps[2] = {
            { 3, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 0, 0, 1, 1 },
            { 4, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 2, 2, 3, 3 }
        };
        beagleUpdatePartials(inst, postOps, 2, BEAGLE_OP_NONE);

        double logL = 0.0;
        int rootIdx = 4, cwIdx = 0, sfIdx = 0, csi = BEAGLE_OP_NONE;
        beagleCalculateRootLogLikelihoods(inst, &rootIdx, &cwIdx, &sfIdx, &csi, 1, &logL);

        std::vector<double> rootPre((size_t)nCats * nPat * S);
        for (int c = 0; c < nCats; c++)
            for (int p = 0; p < nPat; p++)
                for (int s = 0; s < S; s++)
                    rootPre[(size_t)c * nPat * S + p * S + s] = freqs16[s];
        beagleSetPartials(inst, 5, rootPre.data());

        BeagleOperation preOps[4] = {
            { 6, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 5, BEAGLE_OP_NONE, 2, 2 },
            { 7, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 5, BEAGLE_OP_NONE, 3, 3 },
            { 8, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 6, BEAGLE_OP_NONE, 0, 0 },
            { 9, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 6, BEAGLE_OP_NONE, 1, 1 }
        };
        beagleUpdatePrePartials_v5(inst, preOps, 4, BEAGLE_OP_NONE, BEAGLE_PARTIALS_TOP);

        BeagleBranchOperation branchOps[4] = {
            { 1, 8, 1, 0 },
            { 0, 9, 0, 0 },
            { 2, 7, 2, 0 },
            { 3, 6, 3, 0 }
        };
        std::vector<double> grad(S * S, 0.0);
        beagleCalculateAdjointDerivative(inst, branchOps, 0, 0, 4, 0, 4, grad.data(), NULL);

        /* Print partial buffer diagnostics for 16-state */
        std::vector<double> post3 = getPartials(inst, 3, nCats, nPat, S);
        std::vector<double> post4 = getPartials(inst, 4, nCats, nPat, S);
        std::vector<double> pre6  = getPartials(inst, 6, nCats, nPat, S);

        fprintf(stdout, "  post3[cat=0,pat=0]: ");
        for (int s = 0; s < S; s++) fprintf(stdout, " %.4f", post3[s]);
        fprintf(stdout, "\n  post4[cat=0,pat=0]: ");
        for (int s = 0; s < S; s++) fprintf(stdout, " %.4f", post4[s]);
        fprintf(stdout, "\n  pre6 [cat=0,pat=0]: ");
        for (int s = 0; s < S; s++) fprintf(stdout, " %.4f", pre6[s]);
        fprintf(stdout, "\n");

        beagleFinalizeInstance(inst);
        return { logL, grad };
    };

    fprintf(stdout, "\nCreating CPU (double) instance:\n");
    auto [cpuLogL16, cpuGrad16] = runOne(false, -1, false);
    if (cpuGrad16.empty()) return 1;
    fprintf(stdout, "CPU log-likelihood: %.6f\n", cpuLogL16);

    if (gpuDevice < 0) return 0;

    fprintf(stdout, "\nCreating GPU (single) instance:\n");
    auto [gpuLogL16, gpuGrad16] = runOne(true, gpuDevice, true);
    if (gpuGrad16.empty()) return 1;
    fprintf(stdout, "GPU log-likelihood: %.6f\n", gpuLogL16);

    /* Gradient comparison (print first 4x4 sub-block for brevity) */
    fprintf(stdout, "\nGradient comparison 16-state (tol=%.0e):\n", tol);
    fprintf(stdout, "Differences CPU-GPU (first 8x8 block):\n");
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++)
            fprintf(stdout, " %+8.2e", cpuGrad16[i*S+j] - gpuGrad16[i*S+j]);
        fprintf(stdout, "\n");
    }

    bool gradOk = true;
    double maxDiff = 0.0;
    for (int k = 0; k < S*S; k++) {
        double d = fabs(cpuGrad16[k] - gpuGrad16[k]);
        if (d > maxDiff || std::isnan(d)) maxDiff = d;
        if (d > tol || std::isnan(d)) gradOk = false;
    }
    fprintf(stdout, "16-state gradient max|diff|=%.3e  %s\n", maxDiff, gradOk ? "PASS" : "FAIL");
    return gradOk ? 0 : 1;
}

/* ═══════════════════════════════════════════════════════════════════════
 * 17-state test (asymmetric circulant, complex eigenvalues)
 * GPU uses 32-state kernel: PADDED_STATE_COUNT=32, BLOCK_PEELING_SIZE=8 → 4 peel passes
 * ═══════════════════════════════════════════════════════════════════════*/

static int runTest17(int gpuDevice) {
    const int S = 17, nPat = 4, nCats = 2;
    const double tol = 1e-3;       /* absolute tolerance for entries with |value|<=1 */
    const double relTol = 1e-2;    /* relative tolerance for entries with |value|>1 (single vs double precision) */

    fprintf(stdout, "\n=== 17-state asymmetric circulant test (complex eigenvalues) ===\n");
    fprintf(stdout, "    r_fwd=1.0, r_bkd=0.5; odd N → all non-zero modes are complex pairs\n");
    fprintf(stdout, "    GPU: kPaddedStateCount=32, BLOCK_PEELING_SIZE=8 → 4 peel passes\n");

    std::vector<double> evec17, ivec17, eval17;
    buildCirculantN(17, evec17, ivec17, eval17);

    fprintf(stdout, "  Eigenvalue real parts: ");
    for (int k = 0; k < S; k++) fprintf(stdout, "%.3f ", eval17[k]);
    fprintf(stdout, "\n  Eigenvalue imag parts: ");
    for (int k = 0; k < S; k++) fprintf(stdout, "%.3f ", eval17[S + k]);
    fprintf(stdout, "\n");

    std::vector<double> freqs17(S, 1.0 / S);
    double rates17[2]    = { 0.5, 1.5 };
    double catWts17[2]   = { 0.5, 0.5 };
    double edgeLens17[4] = { 0.1, 0.1, 0.2, 0.5 };

    int hSt17[4] = { 3,  0,  5,  7 };
    int cSt17[4] = { 3,  0,  5,  5 };
    int gSt17[4] = { 0,  0,  0, 12 };

    auto runOne = [&](bool useGpu, int dev, bool singlePrec) -> std::pair<double, std::vector<double>> {
        int inst = createInstance(useGpu, dev, singlePrec, S, nPat, nCats, /*complexEigen=*/true);
        if (inst < 0) return { 0.0, {} };

        beagleSetTipStates(inst, 0, hSt17);
        beagleSetTipStates(inst, 1, cSt17);
        beagleSetTipStates(inst, 2, gSt17);

        beagleSetCategoryRates(inst, rates17);
        std::vector<double> pw(nPat, 1.0);
        beagleSetPatternWeights(inst, pw.data());
        beagleSetStateFrequencies(inst, 0, freqs17.data());
        beagleSetCategoryWeights(inst, 0, catWts17);
        beagleSetEigenDecomposition(inst, 0, evec17.data(), ivec17.data(), eval17.data());

        int nodeIdx[4] = { 0, 1, 2, 3 };
        beagleUpdateTransitionMatrices(inst, 0, nodeIdx, NULL, NULL, edgeLens17, 4);

        BeagleOperation postOps[2] = {
            { 3, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 0, 0, 1, 1 },
            { 4, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 2, 2, 3, 3 }
        };
        beagleUpdatePartials(inst, postOps, 2, BEAGLE_OP_NONE);

        double logL = 0.0;
        int rootIdx = 4, cwIdx = 0, sfIdx = 0, csi = BEAGLE_OP_NONE;
        beagleCalculateRootLogLikelihoods(inst, &rootIdx, &cwIdx, &sfIdx, &csi, 1, &logL);

        std::vector<double> rootPre((size_t)nCats * nPat * S);
        for (int c = 0; c < nCats; c++)
            for (int p = 0; p < nPat; p++)
                for (int s = 0; s < S; s++)
                    rootPre[(size_t)c * nPat * S + p * S + s] = freqs17[s];
        beagleSetPartials(inst, 5, rootPre.data());

        BeagleOperation preOps[4] = {
            { 6, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 5, BEAGLE_OP_NONE, 2, 2 },
            { 7, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 5, BEAGLE_OP_NONE, 3, 3 },
            { 8, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 6, BEAGLE_OP_NONE, 0, 0 },
            { 9, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 6, BEAGLE_OP_NONE, 1, 1 }
        };
        beagleUpdatePrePartials_v5(inst, preOps, 4, BEAGLE_OP_NONE, BEAGLE_PARTIALS_TOP);

        BeagleBranchOperation branchOps[4] = {
            { 1, 8, 1, 0 },
            { 0, 9, 0, 0 },
            { 2, 7, 2, 0 },
            { 3, 6, 3, 0 }
        };
        /* Both CPU and GPU write kStateCount² values with stride kStateCount
         * (GPU downsamples from its internal kPaddedStateCount² buffer). */
        std::vector<double> grad(S * S, 0.0);
        beagleCalculateAdjointDerivative(inst, branchOps, 0, 0, 4, 0, 4, grad.data(), NULL);

        const char* preLabels[] = { "pre6", "pre7", "pre8", "pre9" };
        for (int b = 0; b < 4; b++) {
            std::vector<double> pbuf = getPartials(inst, 6+b, nCats, nPat, S);
            fprintf(stdout, "  %s[cat=0,pat=0]:", preLabels[b]);
            for (int s = 0; s < S; s++) fprintf(stdout, " %.4f", pbuf[s]);
            fprintf(stdout, "\n");
        }

        /* Per-branch gradient for the first 8x8 sub-block (diagnostic) */
        const char* bNames[4] = { "chimp", "human", "gorilla", "internal" };
        fprintf(stdout, "  Per-branch gradient G[ls,rs] (first 4x4 sub-block):\n");
        for (int b = 0; b < 4; b++) {
            std::vector<double> gbuf(S * S, 0.0);
            beagleCalculateAdjointDerivative(inst, &branchOps[b], 0, 0, 4, 0, 1, gbuf.data(), NULL);
            fprintf(stdout, "    Branch %s (transMatrix=%d):\n", bNames[b], branchOps[b].branchTransitionMatrix);
            for (int ls = 0; ls < 4; ls++) {
                for (int rs = 0; rs < 4; rs++)
                    fprintf(stdout, " %12.4f", gbuf[ls*S+rs]);
                fprintf(stdout, "\n");
            }
        }

        beagleFinalizeInstance(inst);
        return { logL, grad };
    };

    /* Independent CPU-double logL computation (no adjoint code involved at all),
     * used to validate G[ls,ls] via central finite differences on the eigenvalue
     * eval[ls]. Since G[ls,rs] = dlogL/dQ_eigenbasis[ls,rs] (shared across all
     * branches, because Q is shared), perturbing the diagonal entry eval[ls] by
     * +/-eps and recomputing logL directly tests whether G's diagonal matches a
     * true numerical derivative — independent of whether CPU or GPU adjoint
     * kernels are correct. */
    auto computeLogLOnly = [&](const std::vector<double>& evalIn) -> double {
        int inst = createInstance(false, -1, false, S, nPat, nCats, /*complexEigen=*/true);
        if (inst < 0) return NAN;
        beagleSetTipStates(inst, 0, hSt17);
        beagleSetTipStates(inst, 1, cSt17);
        beagleSetTipStates(inst, 2, gSt17);
        beagleSetCategoryRates(inst, rates17);
        std::vector<double> pw(nPat, 1.0);
        beagleSetPatternWeights(inst, pw.data());
        beagleSetStateFrequencies(inst, 0, freqs17.data());
        beagleSetCategoryWeights(inst, 0, catWts17);
        beagleSetEigenDecomposition(inst, 0, evec17.data(), ivec17.data(), evalIn.data());
        int nodeIdx[4] = { 0, 1, 2, 3 };
        beagleUpdateTransitionMatrices(inst, 0, nodeIdx, NULL, NULL, edgeLens17, 4);
        BeagleOperation postOps[2] = {
            { 3, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 0, 0, 1, 1 },
            { 4, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 2, 2, 3, 3 }
        };
        beagleUpdatePartials(inst, postOps, 2, BEAGLE_OP_NONE);
        double logL = 0.0;
        int rootIdx = 4, cwIdx = 0, sfIdx = 0, csi = BEAGLE_OP_NONE;
        beagleCalculateRootLogLikelihoods(inst, &rootIdx, &cwIdx, &sfIdx, &csi, 1, &logL);
        beagleFinalizeInstance(inst);
        return logL;
    };

    fprintf(stdout, "\nCreating CPU (double) instance:\n");
    auto [cpuLogL17, cpuGrad17] = runOne(false, -1, false);
    if (cpuGrad17.empty()) return 1;
    fprintf(stdout, "CPU log-likelihood: %.6f\n", cpuLogL17);

    fprintf(stdout, "\n--- Finite-difference check on diagonal G[ls,ls] (17-state) ---\n");
    const double fdEps = 1e-5;
    double logLSanity = computeLogLOnly(eval17);
    fprintf(stdout, "  logL sanity recompute = %.6f  (cpuLogL17=%.6f, diff=%+.3e)\n",
            logLSanity, cpuLogL17, logLSanity - cpuLogL17);
    std::vector<double> fdGrad(S, 0.0);
    for (int ls = 0; ls < S; ls++) {
        std::vector<double> evalPlus = eval17, evalMinus = eval17;
        evalPlus[ls]  += fdEps;
        evalMinus[ls] -= fdEps;
        double logLp = computeLogLOnly(evalPlus);
        double logLm = computeLogLOnly(evalMinus);
        fdGrad[ls] = (logLp - logLm) / (2.0 * fdEps);
    }
    /* For a complex-conjugate pair (li, li+1), eval[li]==eval[li+1] (shared
     * real part `a`); perturbing eval[li] moves `a` for the whole pair, so
     * the FD derivative corresponds to G[li,li]+G[li+1,li+1], not G[li,li]
     * alone. Real eigenvalues (imag==0) keep the direct one-to-one check. */
    fprintf(stdout, "  ls   FD_grad        G_diag(sum)    diff(FD-G)\n");
    double maxFdCpuDiff = 0.0;
    for (int ls = 0; ls < S; ls++) {
        if (eval17[S + ls] > 0.0) {
            double gSum = cpuGrad17[ls*S+ls] + cpuGrad17[(ls+1)*S+(ls+1)];
            double diff = fdGrad[ls] - gSum;
            if (fabs(diff) > maxFdCpuDiff) maxFdCpuDiff = fabs(diff);
            fprintf(stdout, "  %2d  %12.6f  %12.6f  %+.3e\n", ls, fdGrad[ls], gSum, diff);
            fprintf(stdout, "  %2d  N/A (2nd-of-pair; folded into row %d above)\n", ls+1, ls);
            ls++;
        } else {
            double diff = fdGrad[ls] - cpuGrad17[ls*S+ls];
            if (fabs(diff) > maxFdCpuDiff) maxFdCpuDiff = fabs(diff);
            fprintf(stdout, "  %2d  %12.6f  %12.6f  %+.3e\n", ls, fdGrad[ls], cpuGrad17[ls*S+ls], diff);
        }
    }
    fprintf(stdout, "  max|FD - CPU_G_diag| = %.3e\n", maxFdCpuDiff);

    if (gpuDevice < 0) return 0;

    fprintf(stdout, "\nCreating GPU (single) instance:\n");
    auto [gpuLogL17, gpuGrad17] = runOne(true, gpuDevice, true);
    if (gpuGrad17.empty()) return 1;
    fprintf(stdout, "GPU log-likelihood: %.6f\n", gpuLogL17);

    fprintf(stdout, "  ls   FD_grad        G_diag(sum)    diff(FD-G)\n");
    double maxFdGpuDiff = 0.0;
    for (int ls = 0; ls < S; ls++) {
        if (eval17[S + ls] > 0.0) {
            double gSum = gpuGrad17[ls*S+ls] + gpuGrad17[(ls+1)*S+(ls+1)];
            double diff = fdGrad[ls] - gSum;
            if (fabs(diff) > maxFdGpuDiff) maxFdGpuDiff = fabs(diff);
            fprintf(stdout, "  %2d  %12.6f  %12.6f  %+.3e\n", ls, fdGrad[ls], gSum, diff);
            fprintf(stdout, "  %2d  N/A (2nd-of-pair; folded into row %d above)\n", ls+1, ls);
            ls++;
        } else {
            double diff = fdGrad[ls] - gpuGrad17[ls*S+ls];
            if (fabs(diff) > maxFdGpuDiff) maxFdGpuDiff = fabs(diff);
            fprintf(stdout, "  %2d  %12.6f  %12.6f  %+.3e\n", ls, fdGrad[ls], gpuGrad17[ls*S+ls], diff);
        }
    }
    fprintf(stdout, "  max|FD - GPU_G_diag| = %.3e\n", maxFdGpuDiff);

    fprintf(stdout, "\nGradient comparison 17-state (tol=%.0e abs / %.0e rel):\n", tol, relTol);
    fprintf(stdout, "Differences CPU-GPU (first 8x8 block):\n");
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++)
            fprintf(stdout, " %+8.2e", cpuGrad17[i*S+j] - gpuGrad17[i*S+j]);
        fprintf(stdout, "\n");
    }

    /* Gradient magnitudes here range from ~0.01 to ~2000, and near-zero
     * entries are often the near-cancellation of large per-branch
     * contributions (e.g. ~+400 and ~-400), so single-precision absolute
     * error there scales with the *largest* magnitude in the computation, not
     * with that entry's own (possibly-cancelled) value. Threshold against the
     * matrix-wide max magnitude instead of a per-entry relative check. */
    double maxMagnitude = 0.0;
    for (int k = 0; k < S*S; k++) {
        if (!std::isnan(cpuGrad17[k])) maxMagnitude = std::max(maxMagnitude, fabs(cpuGrad17[k]));
        if (!std::isnan(gpuGrad17[k])) maxMagnitude = std::max(maxMagnitude, fabs(gpuGrad17[k]));
    }
    const double threshold = std::max(tol, relTol * maxMagnitude);

    bool gradOk = true;
    double maxDiff = 0.0;
    for (int k = 0; k < S*S; k++) {
        double d = fabs(cpuGrad17[k] - gpuGrad17[k]);
        if (d > maxDiff || std::isnan(d)) maxDiff = d;
        if (d > threshold || std::isnan(d)) gradOk = false;
    }
    fprintf(stdout, "17-state gradient max|diff|=%.3e  threshold=%.3e (rel to max|value|=%.3e)  %s\n",
            maxDiff, threshold, maxMagnitude, gradOk ? "PASS" : "FAIL");
    return gradOk ? 0 : 1;
}

/* ── main ────────────────────────────────────────────────────────────── */

int main(int argc, const char *argv[]) {
    printResources();

    bool useGpu    = false;
    int  whichDevice = -1;
    bool run4      = true;
    bool run16     = false;
    bool run17     = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--gpu") == 0 && i+1 < argc) {
            useGpu = true;
            whichDevice = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--nstates") == 0 && i+1 < argc) {
            int ns = atoi(argv[++i]);
            run4  = (ns == 4);
            run16 = (ns == 16);
            run17 = (ns == 17);
        } else if (strcmp(argv[i], "--all") == 0) {
            run4 = run16 = run17 = true;
        }
    }

    int gpuDev = useGpu ? whichDevice : -1;

    int result = 0;
    if (run4)  result |= runTest4(gpuDev);
    if (run16) result |= runTest16(gpuDev);
    if (run17) result |= runTest17(gpuDev);

    return result;
}
