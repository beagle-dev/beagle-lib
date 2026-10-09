/*
 * Copyright 2026 Phylogenetic Likelihood Working Group
 * This file is part of BEAGLE.
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 *
 * Index checks (CMake option BEAGLE_INDEX_CHECKS). With the option ON, the CPU entry points return
 * BEAGLE_ERROR_OUT_OF_RANGE for a partials, matrix or scale index outside its buffers instead of reading or writing
 * out of bounds. For CPU, CPU-SSE and both spectral implementations, and for every scaling mode on CPU:
 *   - a likelihood, pre-order, edge-likelihood, edge-derivative and scale-factor sequence of valid calls succeeds;
 *   - each call with one out-of-range partials, matrix or scale index returns BEAGLE_ERROR_OUT_OF_RANGE;
 *   - the root log likelihood after the rejected calls equals that of a twin instance that never saw them (DYNAMIC
 *     scaling depends on the history of valid calls), and the first root log likelihoods agree across implementations
 *     and scaling modes.
 * Built without the option, it reports itself skipped (exit code 77, ctest SKIP_RETURN_CODE).
 */

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "libhmsbeagle/beagle.h"

#ifndef BEAGLE_INDEX_CHECKS

int main() {
    printf("indexchecktest: SKIP, built without BEAGLE_INDEX_CHECKS\n");
    return 77;
}

#else

enum { kTips = 4, kNodes = 7, kRoot = 6, kPre = kNodes, kPartialsCount = 2 * kNodes, kStates = 4, kPatterns = 5,
       kCategories = 2, kQ = 6, kDenseP = 7, kMatrixCount = 8, kScaleCount = 8, kCumulative = 3 };

static int failures = 0;

static void expect(bool ok, const char* what, const char* impl, const char* mode) {
    if (!ok) {
        printf("FAIL %-24s %-8s %s\n", impl, mode, what);
        ++failures;
    }
}

struct Mode {
    const char* name;
    long flag;
};

struct LogLikelihoods {
    double first; // before any out-of-range call
    double last;  // after them
    bool created; // false when the implementation does not exist here (SSE on Linux ARM)
};

// one instance: valid calls, then (if reject) out-of-range calls, then the valid likelihood again
static LogLikelihoods run(long implementation, const Mode& mode, bool reject) {
    const bool manualOrDynamic = mode.flag == BEAGLE_FLAG_SCALING_MANUAL || mode.flag == BEAGLE_FLAG_SCALING_DYNAMIC;
    BeagleInstanceDetails details;
    const int instance = beagleCreateInstance(kTips, kPartialsCount, 0, kStates, kPatterns, 1, kMatrixCount,
                                              kCategories, kScaleCount, NULL, 0, 0,
                                              BEAGLE_FLAG_PROCESSOR_CPU | BEAGLE_FLAG_PRECISION_DOUBLE |
                                              implementation | mode.flag, &details);
    if (instance < 0) {
        if (implementation & BEAGLE_FLAG_VECTOR_SSE) { // the SSE plugins are platform-dependent
            if (reject) printf("skip %s: no SSE implementation of this kind\n", mode.name);
            return {0, 0, false};
        }
        printf("FAIL could not create an instance for %s\n", mode.name);
        ++failures;
        return {0, 0, true};
    }
    const char* impl = details.implName;
    // the scale buffers: MANUAL and DYNAMIC as requested, AUTO one, ALWAYS one per internal node plus one
    const int internal = kPartialsCount - kTips;
    const int scaleCount = manualOrDynamic ? kScaleCount : (mode.flag == BEAGLE_FLAG_SCALING_AUTO ? 1 : internal + 1);

    // the 4-state Jukes-Cantor model, Q = 1/3 off the diagonal: orthogonal eigenvectors (the columns), eigenvalues 0
    // and -4/3
    const double evec[16] = {1, 1, 1, 1, 1, -1, 1, 1, 1, 0, -2, 1, 1, 0, 0, -3};
    double ivec[16];
    for (int i = 0; i < 4; i++) {
        double norm = 0;
        for (int j = 0; j < 4; j++) norm += evec[j * 4 + i] * evec[j * 4 + i];
        for (int j = 0; j < 4; j++) ivec[i * 4 + j] = evec[j * 4 + i] / norm;
    }
    const double eval[4] = {0, -4.0 / 3, -4.0 / 3, -4.0 / 3};
    beagleSetEigenDecomposition(instance, 0, evec, ivec, eval);
    const double frequencies[4] = {0.25, 0.25, 0.25, 0.25};
    beagleSetStateFrequencies(instance, 0, frequencies);
    const double rates[kCategories] = {0.5, 1.5}, weights[kCategories] = {0.5, 0.5};
    beagleSetCategoryRates(instance, rates);
    beagleSetCategoryWeights(instance, 0, weights);
    const double patternWeights[kPatterns] = {1, 1, 1, 1, 1};
    beagleSetPatternWeights(instance, patternWeights);
    for (int tip = 0; tip < kTips; ++tip) {
        double partials[kStates * kPatterns];
        for (int k = 0; k < kStates * kPatterns; ++k) partials[k] = 0.1 + ((k * 7 + tip * 3) % 9) / 10.0;
        beagleSetTipPartials(instance, tip, partials);
    }
    double q[kStates * kStates * kCategories], p[kStates * kStates * kCategories];
    for (int c = 0; c < kCategories; c++)
        for (int i = 0; i < kStates; i++)
            for (int j = 0; j < kStates; j++) {
                q[(c * kStates + i) * kStates + j] = (i == j) ? -1.0 : 1.0 / 3;
                p[(c * kStates + i) * kStates + j] = (i == j) ? 0.7 : 0.1;
            }
    expect(beagleSetDifferentialMatrix(instance, kQ, q) == BEAGLE_SUCCESS, "setDifferentialMatrix", impl, mode.name);
    expect(beagleSetTransitionMatrix(instance, kDenseP, p, 1.0) == BEAGLE_SUCCESS, "setTransitionMatrix", impl,
           mode.name);

    const int branches[6] = {0, 1, 2, 3, 4, 5};
    const double lengths[6] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
    // MANUAL and DYNAMIC rescale into scale buffers 0-2 and accumulate into kCumulative; AUTO and ALWAYS ignore them
    const int w0 = manualOrDynamic ? 0 : BEAGLE_OP_NONE, w1 = manualOrDynamic ? 1 : BEAGLE_OP_NONE,
              w2 = manualOrDynamic ? 2 : BEAGLE_OP_NONE;
    // MANUAL accumulates the scale factors after the post-order pass; DYNAMIC keeps the cumulative buffer up to date
    const int cumulative = manualOrDynamic ? kCumulative : BEAGLE_OP_NONE;
    const int postCumulative = (mode.flag == BEAGLE_FLAG_SCALING_DYNAMIC) ? kCumulative : BEAGLE_OP_NONE;
    const BeagleOperation post[3] = {{4, w0, w0, 0, 0, 1, 1}, {5, w1, w1, 2, 2, 3, 3}, {6, w2, w2, 4, 4, 5, 5}};
    const int weightsIndex = 0, frequenciesIndex = 0, root = kRoot;

    auto logLikelihood = [&](double& logL) {
        bool ok = beagleUpdateTransitionMatrices(instance, 0, branches, NULL, NULL, lengths, 6) == BEAGLE_SUCCESS;
        if (manualOrDynamic) ok &= beagleResetScaleFactors(instance, kCumulative) == BEAGLE_SUCCESS;
        ok &= beagleUpdatePartials(instance, post, 3, postCumulative) == BEAGLE_SUCCESS;
        if (mode.flag == BEAGLE_FLAG_SCALING_MANUAL) {
            const int scaled[3] = {0, 1, 2};
            ok &= beagleAccumulateScaleFactors(instance, scaled, 3, kCumulative) == BEAGLE_SUCCESS;
        } else if (mode.flag == BEAGLE_FLAG_SCALING_AUTO) {
            // AUTO: the root reads the one cumulative buffer, filled from the internal nodes' partials indices
            const int internalNodes[3] = {4, 5, 6};
            ok &= beagleAccumulateScaleFactors(instance, internalNodes, 3, BEAGLE_OP_NONE) == BEAGLE_SUCCESS;
        }
        ok &= beagleCalculateRootLogLikelihoods(instance, &root, &weightsIndex, &frequenciesIndex, &cumulative, 1,
                                                &logL) == BEAGLE_SUCCESS;
        return ok;
    };
    double logL = 0;
    expect(logLikelihood(logL), "valid likelihood calls", impl, mode.name);

    // valid pre-order, edge derivatives and edge likelihood
    std::vector<double> prior(kStates * kPatterns * kCategories, 0.25);
    beagleSetPartials(instance, kPre + kRoot, prior.data());
    const int pw = (mode.flag == BEAGLE_FLAG_SCALING_DYNAMIC) ? 4 : BEAGLE_OP_NONE; // DYNAMIC rescales pre-order too
    const BeagleOperation pre[6] = {
            {kPre + 4, pw, pw, kPre + kRoot, 4, 5, 5}, {kPre + 5, pw, pw, kPre + kRoot, 5, 4, 4},
            {kPre + 0, pw, pw, kPre + 4, 0, 1, 1}, {kPre + 1, pw, pw, kPre + 4, 1, 0, 0},
            {kPre + 2, pw, pw, kPre + 5, 2, 3, 3}, {kPre + 3, pw, pw, kPre + 5, 3, 2, 2}};
    const int preCumulative = (mode.flag == BEAGLE_FLAG_SCALING_DYNAMIC) ? kCumulative : BEAGLE_OP_NONE;
    expect(beagleUpdatePrePartials(instance, pre, 6, preCumulative) == BEAGLE_SUCCESS, "valid pre-order", impl,
           mode.name);
    const int postIndices[6] = {0, 1, 2, 3, 4, 5};
    const int preIndices[6] = {kPre + 0, kPre + 1, kPre + 2, kPre + 3, kPre + 4, kPre + 5};
    const int qIndices[6] = {kQ, kQ, kQ, kQ, kQ, kQ};
    std::vector<double> derivatives(6 * kPatterns), sums(6);
    expect(beagleCalculateEdgeDerivatives(instance, postIndices, preIndices, qIndices, &weightsIndex, 6,
                                          derivatives.data(), sums.data(), NULL) == BEAGLE_SUCCESS,
           "valid calculateEdgeDerivatives", impl, mode.name);
    const int parent = 5, child = 4, denseP = kDenseP, none = BEAGLE_OP_NONE;
    double edgeLogL = 0;
    expect(beagleCalculateEdgeLogLikelihoods(instance, &parent, &child, &denseP, NULL, NULL, &weightsIndex,
                                             &frequenciesIndex, &none, 1, &edgeLogL, NULL, NULL) == BEAGLE_SUCCESS,
           "valid calculateEdgeLogLikelihoods", impl, mode.name);
    std::vector<double> matrix(kStates * kStates * kCategories);
    expect(beagleGetTransitionMatrix(instance, kDenseP, matrix.data()) == BEAGLE_SUCCESS, "valid getTransitionMatrix",
           impl, mode.name);

    // out-of-range indices
    const int badPartials[2] = {-1, kPartialsCount};
    const int badMatrix[2] = {-1, kMatrixCount};
    const int badScale[2] = {-2, scaleCount};
    std::vector<double> out(kStates * kStates * kCategories * 8);
    double sum[8], sumSquared[8];
    for (int b = 0; b < (reject ? 2 : 0); b++) {
        const int bp = badPartials[b], bm = badMatrix[b], bs = badScale[b];
        auto rejected = [&](int code, const char* what) {
            expect(code == BEAGLE_ERROR_OUT_OF_RANGE, what, impl, mode.name);
        };
        // partials
        {
            const BeagleOperation op[1] = {{bp, w0, w0, 0, 0, 1, 1}};
            rejected(beagleUpdatePartials(instance, op, 1, postCumulative), "updatePartials destination");
            const BeagleOperation op2[1] = {{4, w0, w0, bp, 0, 1, 1}};
            rejected(beagleUpdatePartials(instance, op2, 1, postCumulative), "updatePartials child");
            const BeagleOperation op3[1] = {{kPre + 0, pw, pw, bp, 0, 1, 1}};
            rejected(beagleUpdatePrePartials(instance, op3, 1, preCumulative), "updatePrePartials parent");
            rejected(beagleCalculateRootLogLikelihoods(instance, &bp, &weightsIndex, &frequenciesIndex, &cumulative,
                                                       1, sum), "calculateRootLogLikelihoods buffer");
            rejected(beagleCalculateEdgeLogLikelihoods(instance, &bp, &child, &denseP, NULL, NULL, &weightsIndex,
                                                       &frequenciesIndex, &none, 1, sum, NULL, NULL),
                     "calculateEdgeLogLikelihoods parent");
            const int posts[1] = {bp}, pres[1] = {kPre};
            rejected(beagleCalculateEdgeDerivatives(instance, posts, pres, qIndices, &weightsIndex, 1, out.data(),
                                                    sum, NULL), "calculateEdgeDerivatives post");
            const int rateIndex = 0;
            const double length = 0.1;
            rejected(beagleCalculateCrossProductDerivative(instance, postIndices, &bp, &rateIndex, &weightsIndex, &length, 1,
                                                  out.data(), out.data()), "calculateCrossProducts pre");
        }
        // matrices
        {
            rejected(beagleUpdateTransitionMatrices(instance, 0, &bm, NULL, NULL, lengths, 1),
                     "updateTransitionMatrices");
            const BeagleOperation op[1] = {{4, w0, w0, 0, bm, 1, 1}};
            rejected(beagleUpdatePartials(instance, op, 1, postCumulative), "updatePartials child matrix");
            const BeagleOperation op2[1] = {{kPre + 0, pw, pw, kPre + 4, 0, 1, bm}};
            rejected(beagleUpdatePrePartials(instance, op2, 1, preCumulative), "updatePrePartials sibling matrix");
            rejected(beagleGetTransitionMatrix(instance, bm, matrix.data()), "getTransitionMatrix");
            const int dense[1] = {kDenseP}, q1[1] = {kQ};
            rejected(beagleConvolveTransitionMatrices(instance, dense, q1, &bm, 1), "convolve result");
            rejected(beagleTransposeTransitionMatrices(instance, &bm, q1, 1), "transpose input");
            rejected(beagleCalculateEdgeLogLikelihoods(instance, &parent, &child, &bm, NULL, NULL, &weightsIndex,
                                                       &frequenciesIndex, &none, 1, sum, NULL, NULL),
                     "calculateEdgeLogLikelihoods matrix");
            const int posts[1] = {0}, pres[1] = {kPre};
            rejected(beagleCalculateEdgeDerivatives(instance, posts, pres, &bm, &weightsIndex, 1, out.data(), sum,
                                                    NULL), "calculateEdgeDerivatives matrix");
        }
        // scale buffers
        rejected(beagleResetScaleFactors(instance, bs), "resetScaleFactors");
        rejected(beagleCopyScaleFactors(instance, bs, 0), "copyScaleFactors destination");
        const int zero = 0;
        rejected(beagleRemoveScaleFactors(instance, &zero, 1, bs), "removeScaleFactors cumulative");
        std::vector<double> partials(kStates * kPatterns * kCategories);
        rejected(beagleGetPartials(instance, 4, bs, partials.data()), "getPartials scale");
        if (manualOrDynamic) {
            // under MANUAL a negative operation scale index means none, so only the large one is out of range
            const BeagleOperation op[1] = {{4, scaleCount, scaleCount, 0, 0, 1, 1}};
            rejected(beagleUpdatePartials(instance, op, 1, postCumulative), "updatePartials scale");
            rejected(beagleUpdatePartials(instance, post, 1, bs), "updatePartials cumulative scale");
            // a negative root scale index other than BEAGLE_OP_NONE is skipped by the generic root kernels and read
            // by the 4-state ones
            const bool fourState = strstr(impl, "4State") != NULL && strstr(impl, "Spectral") == NULL;
            const int code = beagleCalculateRootLogLikelihoods(instance, &root, &weightsIndex, &frequenciesIndex, &bs,
                                                               1, sum);
            if (bs >= 0 || fourState) {
                rejected(code, "calculateRootLogLikelihoods scale");
            } else {
                expect(code == BEAGLE_SUCCESS, "calculateRootLogLikelihoods with a skipped negative scale index", impl,
                       mode.name);
            }
            rejected(beagleAccumulateScaleFactors(instance, &bs, 1, kCumulative), "accumulateScaleFactors");
        } else {
            // AUTO and ALWAYS keep a node's scale state at its partials index minus the tip count
            const BeagleOperation op[1] = {{0, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 1, 1, 2, 2}};
            rejected(beagleUpdatePartials(instance, op, 1, BEAGLE_OP_NONE), "updatePartials into a tip");
        }
        // the cumulative scale buffer of a post-order call is read even with no operations
        rejected(beagleUpdatePartials(instance, post, 0, scaleCount), "updatePartials cumulative scale, no operations");
        if (mode.flag == BEAGLE_FLAG_SCALING_ALWAYS) {
            // a root at a tip has scale index tip - tipCount < 0: skipped by the generic root kernels, read by the
            // 4-state ones
            const bool fourState = strstr(impl, "4State") != NULL && strstr(impl, "Spectral") == NULL;
            const int tipRoot = 0;
            const int code = beagleCalculateRootLogLikelihoods(instance, &tipRoot, &weightsIndex, &frequenciesIndex,
                                                               &none, 1, sum);
            if (fourState) {
                rejected(code, "calculateRootLogLikelihoods at a tip (ALWAYS, 4-state)");
            } else {
                expect(code == BEAGLE_SUCCESS, "calculateRootLogLikelihoods at a tip (ALWAYS)", impl, mode.name);
            }
        }
        if (mode.flag == BEAGLE_FLAG_SCALING_AUTO) {
            const int tip = 0;
            rejected(beagleAccumulateScaleFactors(instance, &tip, 1, 0), "accumulateScaleFactors of a tip (AUTO)");
        }
        (void) sumSquared;
    }

    double again = 0;
    expect(logLikelihood(again), "valid likelihood calls after the rejected ones", impl, mode.name);
    if (reject) {
        printf("%-30s %-8s log likelihood %.10f, edge %.10f\n", impl, mode.name, logL, edgeLogL);
    }
    beagleFinalizeInstance(instance);
    return {logL, again, true};
}

int main() {
    const Mode manual = {"MANUAL", BEAGLE_FLAG_SCALING_MANUAL}, dynamic = {"DYNAMIC", BEAGLE_FLAG_SCALING_DYNAMIC},
               autoScaling = {"AUTO", BEAGLE_FLAG_SCALING_AUTO}, always = {"ALWAYS", BEAGLE_FLAG_SCALING_ALWAYS};
    const long cpu = BEAGLE_FLAG_VECTOR_NONE, sse = BEAGLE_FLAG_VECTOR_SSE,
               spectral = BEAGLE_FLAG_VECTOR_NONE | BEAGLE_FLAG_SPECTRAL_REPRESENTATION,
               spectralSSE = BEAGLE_FLAG_VECTOR_SSE | BEAGLE_FLAG_SPECTRAL_REPRESENTATION;
    struct Case { long implementation; Mode mode; };
    const Case cases[] = {{cpu, manual}, {cpu, dynamic}, {cpu, autoScaling}, {cpu, always}, {sse, manual},
                          {spectral, manual}, {spectralSSE, manual}, {spectral, always}};
    double first = 0;
    for (const Case& c : cases) {
        const LogLikelihoods checked = run(c.implementation, c.mode, true);
        if (!checked.created) continue;
        const LogLikelihoods twin = run(c.implementation, c.mode, false);
        expect(checked.last == twin.last, "root log likelihood differs from a twin instance without the rejected calls",
               "", c.mode.name);
        if (&c == cases) {
            first = checked.first;
        }
        expect(std::fabs(checked.first - first) < 1e-10 * std::fabs(first),
               "root log likelihood differs across implementations or scaling modes", "", c.mode.name);
    }
    printf("%s\n", failures == 0 ? "indexchecktest: all checks passed" : "indexchecktest: FAILED");
    return failures == 0 ? 0 : 1;
}

#endif
