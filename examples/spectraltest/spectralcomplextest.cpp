/*
 * Copyright 2026 Phylogenetic Likelihood Working Group
 * This file is part of BEAGLE.
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 *
 * Compares the spectral CPU implementation with the standard CPU implementation, which uses full transition
 * matrices, for models whose eigen decompositions have complex conjugate eigenvalue pairs:
 *   - one rate category has rate 0 (invariant sites), and
 *   - sibling branches use different eigen decompositions, one with complex pairs and one without.
 * Odd and even state counts are covered, and both spectral implementations (VECTOR_NONE and VECTOR_SSE). Exits
 * non-zero if the root log likelihoods or the pre-order partials disagree.
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "libhmsbeagle/beagle.h"

// circulant CTMC: complex conjugate eigenvalue pairs unless rFwd == rBkd
static void buildCirculant(int n, double rFwd, double rBkd,
                           std::vector<double>& evec, std::vector<double>& ivec, std::vector<double>& eval) {
    const double twoPiOverN = 2.0 * M_PI / n;
    const bool evenN = (n % 2 == 0);
    const int pairs = evenN ? n / 2 - 1 : (n - 1) / 2;
    evec.assign(n * n, 0.0);
    ivec.assign(n * n, 0.0);
    eval.assign(2 * n, 0.0);
    for (int j = 0; j < n; j++) {
        evec[j * n] = 1.0 / n;
        ivec[j] = 1.0;
    }
    for (int m = 1; m <= pairs; m++) {
        const int re = 2 * m - 1, im = 2 * m;
        const double theta = twoPiOverN * m;
        const double a = (rFwd + rBkd) * (cos(theta) - 1.0);
        const double b = (rFwd - rBkd) * sin(theta);
        eval[re] = a;
        eval[n + re] = b;
        eval[im] = a;
        eval[n + im] = -b;
        for (int j = 0; j < n; j++) {
            evec[j * n + re] = cos(theta * j) / n;
            evec[j * n + im] = sin(theta * j) / n;
            ivec[re * n + j] = 2.0 * cos(theta * j);
            ivec[im * n + j] = 2.0 * sin(theta * j);
        }
    }
    if (evenN) {
        const int last = n - 1;
        eval[last] = -2.0 * (rFwd + rBkd);
        for (int j = 0; j < n; j++) {
            const double v = (j % 2 == 0) ? 1.0 : -1.0;
            evec[j * n + last] = v / n;
            ivec[last * n + j] = v;
        }
    }
}

struct Result {
    double logL;
    std::vector<double> pre; // pre-order partials of the four tips
    const char* implName;
};

// tree ((0,1)4,(2,3)5)6; tips 0 and 2 use the complex model, 1 and 3 the real one, so siblings differ
static bool evaluate(bool spectral, long vector, int n, Result& result) {
    const int tips = 4, patterns = 5, categories = 3;
    const int nodes = 7, root = 6, preBase = nodes; // pre-order buffers follow the post-order buffers
    const long requirements = BEAGLE_FLAG_PROCESSOR_CPU | BEAGLE_FLAG_EIGEN_COMPLEX | BEAGLE_FLAG_PRECISION_DOUBLE |
                              (spectral ? BEAGLE_FLAG_SPECTRAL_REPRESENTATION | vector : 0);

    BeagleInstanceDetails details;
    const int instance = beagleCreateInstance(tips, 2 * nodes, 0, n, patterns, 2, nodes, categories, 0, NULL, 0,
                                              BEAGLE_FLAG_SCALING_MANUAL, requirements, &details);
    if (instance < 0) {
        fprintf(stderr, "Could not create a%s instance with %d states\n", spectral ? " spectral" : "", n);
        return false;
    }
    result.implName = details.implName;

    std::vector<double> evec, ivec, eval;
    buildCirculant(n, 1.0, 0.4, evec, ivec, eval);  // complex conjugate pairs
    beagleSetEigenDecomposition(instance, 0, evec.data(), ivec.data(), eval.data());
    buildCirculant(n, 0.7, 0.7, evec, ivec, eval);  // real eigenvalues
    beagleSetEigenDecomposition(instance, 1, evec.data(), ivec.data(), eval.data());

    std::vector<double> frequencies(n, 1.0 / n);
    beagleSetStateFrequencies(instance, 0, frequencies.data());
    const double rates[categories] = {0.0, 0.8, 2.2}; // an invariant-sites category first
    const double weights[categories] = {0.2, 0.4, 0.4};
    beagleSetCategoryRates(instance, rates);
    beagleSetCategoryWeights(instance, 0, weights);
    std::vector<double> patternWeights(patterns, 1.0);
    beagleSetPatternWeights(instance, patternWeights.data());

    for (int tip = 0; tip < tips; ++tip) {
        std::vector<double> partials(n * patterns);
        for (int k = 0; k < patterns; ++k) {
            for (int s = 0; s < n; ++s) {
                partials[k * n + s] = 0.1 + ((s * 7 + k * 3 + tip * 5) % (n + 2)) / double(n + 2);
            }
        }
        beagleSetTipPartials(instance, tip, partials.data());
    }

    // branches (the matrix index is the node number): complex model on 0, 2 and 4, real on 1, 3 and 5
    const int complexBranches[] = {0, 2, 4};
    const double complexLengths[] = {0.3, 0.7, 0.2};
    const int realBranches[] = {1, 3, 5};
    const double realLengths[] = {0.5, 0.4, 0.6};
    beagleUpdateTransitionMatrices(instance, 0, complexBranches, NULL, NULL, complexLengths, 3);
    beagleUpdateTransitionMatrices(instance, 1, realBranches, NULL, NULL, realLengths, 3);

    const BeagleOperation post[] = {
            {4, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 0, 0, 1, 1},
            {5, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 2, 2, 3, 3},
            {6, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 4, 4, 5, 5}};
    beagleUpdatePartials(instance, post, 3, BEAGLE_OP_NONE);

    const int rootIndex = root, weightsIndex = 0, frequenciesIndex = 0, scaleIndex = BEAGLE_OP_NONE;
    beagleCalculateRootLogLikelihoods(instance, &rootIndex, &weightsIndex, &frequenciesIndex, &scaleIndex, 1,
                                      &result.logL);

    // bottom pre-order partials: the root prior, then each node from its parent and sibling
    std::vector<double> prior(n * patterns * categories, 1.0 / n);
    beagleSetPartials(instance, preBase + root, prior.data());
    const BeagleOperation pre[] = {
            {preBase + 4, BEAGLE_OP_NONE, BEAGLE_OP_NONE, preBase + root, 4, 5, 5},
            {preBase + 5, BEAGLE_OP_NONE, BEAGLE_OP_NONE, preBase + root, 5, 4, 4},
            {preBase + 0, BEAGLE_OP_NONE, BEAGLE_OP_NONE, preBase + 4, 0, 1, 1},
            {preBase + 1, BEAGLE_OP_NONE, BEAGLE_OP_NONE, preBase + 4, 1, 0, 0},
            {preBase + 2, BEAGLE_OP_NONE, BEAGLE_OP_NONE, preBase + 5, 2, 3, 3},
            {preBase + 3, BEAGLE_OP_NONE, BEAGLE_OP_NONE, preBase + 5, 3, 2, 2}};
    beagleUpdatePrePartials(instance, pre, 6, BEAGLE_OP_NONE);

    result.pre.clear();
    std::vector<double> partials(n * patterns * categories);
    for (int tip = 0; tip < tips; ++tip) {
        beagleGetPartials(instance, preBase + tip, BEAGLE_OP_NONE, partials.data());
        result.pre.insert(result.pre.end(), partials.begin(), partials.end());
    }

    beagleFinalizeInstance(instance);
    return true;
}

int main() {
    int failures = 0;
    for (long vector : {BEAGLE_FLAG_VECTOR_NONE, BEAGLE_FLAG_VECTOR_SSE}) {
        for (int n : {3, 4, 5, 8, 9, 20, 21}) {
            Result standard, spectral;
            if (!evaluate(false, 0, n, standard) || !evaluate(true, vector, n, spectral)) {
                return 1;
            }

            double worst = std::fabs(standard.logL - spectral.logL) / std::fabs(standard.logL);
            for (size_t i = 0; i < standard.pre.size(); ++i) {
                const double scale = std::max(std::fabs(standard.pre[i]), 1e-300);
                worst = std::max(worst, std::fabs(standard.pre[i] - spectral.pre[i]) / scale);
            }
            const bool ok = worst < 1e-10;
            printf("%-24s %2d states: log likelihood %.12f (standard) %.12f, worst relative difference %.1e %s\n",
                   spectral.implName, n, standard.logL, spectral.logL, worst, ok ? "ok" : "FAIL");
            if (!ok) {
                ++failures;
            }
        }
    }
    return failures == 0 ? 0 : 1;
}
