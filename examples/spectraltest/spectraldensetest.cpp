/*
 * Copyright 2026 Phylogenetic Likelihood Working Group
 * This file is part of BEAGLE.
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 *
 * Dense matrix storage on the CPU spectral implementations. A spectral instance allocates a dense matrix for an index
 * only when the index is written densely (the setters, or the destination of convolve or transpose). For both
 * spectral implementations (VECTOR_NONE and VECTOR_SSE) at 4 and 17 states, against the standard SSE implementation
 * (the standard VECTOR_NONE one where there is no SSE implementation, as on Linux ARM):
 *   - setDifferentialMatrix followed by calculateEdgeDerivatives agrees to 1e-12 (relative);
 *   - setTransitionMatrices, convolve, transpose and the edge log likelihood of a densely written index agree;
 *   - getTransitionMatrix returns the dense copy of a densely written index exactly, and the matrix of the eigen
 *     decomposition for an index written only by updateTransitionMatrices;
 *   - a dense read of an index without dense contents returns BEAGLE_ERROR_OUT_OF_RANGE: calculateEdgeDerivatives,
 *     calculateEdgeLogLikelihoods, the inputs of convolve and transpose, and getTransitionMatrix of an index
 *     never written;
 *   - matrix indices outside [0, matrixBufferCount) return BEAGLE_ERROR_OUT_OF_RANGE from the setters and from
 *     getTransitionMatrix.
 * Exits non-zero on any failure.
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "libhmsbeagle/beagle.h"

// circulant CTMC: complex conjugate eigenvalue pairs unless rFwd == rBkd (as in spectralcomplextest)
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

// the rate matrix of buildCirculant, one copy per rate category
static std::vector<double> circulantQ(int n, double rFwd, double rBkd, int categories) {
    std::vector<double> q(n * n * categories, 0.0);
    for (int c = 0; c < categories; c++) {
        double* m = &q[c * n * n];
        for (int i = 0; i < n; i++) {
            m[i * n + (i + 1) % n] += rFwd;
            m[i * n + (i + n - 1) % n] += rBkd;
            m[i * n + i] -= rFwd + rBkd;
        }
    }
    return q;
}

// an arbitrary row-stochastic matrix per category, different for each seed
static std::vector<double> stochastic(int n, int categories, int seed) {
    std::vector<double> p(n * n * categories);
    for (int c = 0; c < categories; c++) {
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int j = 0; j < n; j++) {
                const double v = 0.05 + ((i * 13 + j * 7 + c * 5 + seed * 11) % (n + 3)) / double(n + 3);
                p[(c * n + i) * n + j] = v;
                sum += v;
            }
            for (int j = 0; j < n; j++) {
                p[(c * n + i) * n + j] /= sum;
            }
        }
    }
    return p;
}

static int failures = 0;

static void expect(bool ok, const char* what, const char* impl, int n) {
    if (!ok) {
        printf("FAIL %-26s %2d states: %s\n", impl, n, what);
        ++failures;
    }
}

static double relativeDifference(const std::vector<double>& x, const std::vector<double>& y) {
    double worst = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        worst = std::max(worst, std::fabs(x[i] - y[i]) / std::max(std::fabs(x[i]), 1e-300));
    }
    return worst;
}

// matrix buffer layout: branch matrices 0-5 (node numbers), Q of each model 6 and 7, dense matrices 8 and 9 (set
// together), the convolve and transpose destinations 10 and 11, and 12 never written
enum { kQComplex = 6, kQReal = 7, kDenseA = 8, kDenseB = 9, kConvolved = 10, kTransposed = 11, kUnwritten = 12,
       kMatrixCount = 13 };

struct Result {
    const char* implName;
    std::vector<double> derivatives;    // per node and pattern
    std::vector<double> sumDerivatives; // per node
    double edgeLogL = NAN;       // written only by a successful edge-likelihood call
    std::vector<double> dense;          // getTransitionMatrix of kDenseA, kDenseB, kConvolved, kTransposed, kQComplex
    std::vector<double> eigenMatrix;    // getTransitionMatrix of branch 0
};

// tree ((0,1)4,(2,3)5)6; tip 0 has compact states, the others partials; tips 0 and 2 use the complex model
static bool evaluate(bool spectral, long vector, int n, Result& result) {
    const int tips = 4, patterns = 7, categories = 2;
    const int nodes = 7, root = 6, preBase = nodes; // pre-order buffers follow the post-order buffers
    const long requirements = BEAGLE_FLAG_PROCESSOR_CPU | BEAGLE_FLAG_EIGEN_COMPLEX | BEAGLE_FLAG_PRECISION_DOUBLE |
                              BEAGLE_FLAG_SCALING_MANUAL | vector |
                              (spectral ? BEAGLE_FLAG_SPECTRAL_REPRESENTATION : 0);

    BeagleInstanceDetails details;
    const int instance = beagleCreateInstance(tips, 2 * nodes, 1, n, patterns, 2, kMatrixCount, categories, 0,
                                              NULL, 0, 0, requirements, &details);
    if (instance < 0) {
        fprintf(stderr, "Could not create a%s instance with %d states\n", spectral ? " spectral" : "", n);
        return false;
    }
    result.implName = details.implName;
    const bool isSpectral = strstr(details.implName, "Spectral") != NULL;
    if (isSpectral != spectral) {
        fprintf(stderr, "Expected a%s implementation, got %s\n", spectral ? " spectral" : " standard",
                details.implName);
        beagleFinalizeInstance(instance);
        return false;
    }

    const double rFwd = 1.0, rBkd = 0.4, rSym = 0.7;
    std::vector<double> evec, ivec, eval;
    buildCirculant(n, rFwd, rBkd, evec, ivec, eval);  // complex conjugate pairs
    beagleSetEigenDecomposition(instance, 0, evec.data(), ivec.data(), eval.data());
    buildCirculant(n, rSym, rSym, evec, ivec, eval);  // real eigenvalues
    beagleSetEigenDecomposition(instance, 1, evec.data(), ivec.data(), eval.data());

    std::vector<double> frequencies(n, 1.0 / n);
    beagleSetStateFrequencies(instance, 0, frequencies.data());
    const double rates[categories] = {0.6, 1.4};
    const double weights[categories] = {0.5, 0.5};
    beagleSetCategoryRates(instance, rates);
    beagleSetCategoryWeights(instance, 0, weights);
    std::vector<double> patternWeights(patterns, 1.0);
    beagleSetPatternWeights(instance, patternWeights.data());

    std::vector<int> states(patterns);
    for (int k = 0; k < patterns; ++k) {
        states[k] = (k * 3) % n;
    }
    beagleSetTipStates(instance, 0, states.data());
    for (int tip = 1; tip < tips; ++tip) {
        std::vector<double> partials(n * patterns);
        for (int k = 0; k < patterns; ++k) {
            for (int s = 0; s < n; ++s) {
                partials[k * n + s] = 0.1 + ((s * 7 + k * 3 + tip * 5) % (n + 2)) / double(n + 2);
            }
        }
        beagleSetTipPartials(instance, tip, partials.data());
    }

    // branch matrices (the matrix index is the node number): complex model on 0, 2 and 4, real on 1, 3 and 5
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

    const int weightsIndex = 0, frequenciesIndex = 0, scaleIndex = BEAGLE_OP_NONE;
    const char* impl = details.implName;

    // a dense read of indices without dense contents (on the standard implementation they hold uninitialised memory)
    const int postIndices[] = {0, 1, 2, 3, 4, 5};
    const int preIndices[] = {preBase + 0, preBase + 1, preBase + 2, preBase + 3, preBase + 4, preBase + 5};
    std::vector<double> derivatives(6 * patterns), sums(6);
    if (spectral) {
        const int unwritten[] = {kQComplex, kQReal, kQComplex, kQReal, kQComplex, 0};
        expect(beagleCalculateEdgeDerivatives(instance, postIndices, preIndices, unwritten, &weightsIndex, 6,
                                              derivatives.data(), sums.data(), NULL) == BEAGLE_ERROR_OUT_OF_RANGE,
               "calculateEdgeDerivatives with derivative indices not written densely", impl, n);
    }

    // setDifferentialMatrix and calculateEdgeDerivatives, as BEAST's branch-rate gradient uses them
    const std::vector<double> qComplex = circulantQ(n, rFwd, rBkd, categories);
    const std::vector<double> qReal = circulantQ(n, rSym, rSym, categories);
    expect(beagleSetDifferentialMatrix(instance, kQComplex, qComplex.data()) == BEAGLE_SUCCESS,
           "setDifferentialMatrix", impl, n);
    expect(beagleSetDifferentialMatrix(instance, kQReal, qReal.data()) == BEAGLE_SUCCESS,
           "setDifferentialMatrix", impl, n);
    const int derivativeIndices[] = {kQComplex, kQReal, kQComplex, kQReal, kQComplex, kQReal};
    expect(beagleCalculateEdgeDerivatives(instance, postIndices, preIndices, derivativeIndices, &weightsIndex, 6,
                                          derivatives.data(), sums.data(), NULL) == BEAGLE_SUCCESS,
           "calculateEdgeDerivatives", impl, n);
    result.derivatives = derivatives;
    result.sumDerivatives = sums;

    // setTransitionMatrices, then convolve and transpose; their inputs must be dense
    std::vector<double> denseAB = stochastic(n, categories, 1);
    const std::vector<double> denseB = stochastic(n, categories, 2);
    denseAB.insert(denseAB.end(), denseB.begin(), denseB.end());
    const int denseIndices[] = {kDenseA, kDenseB};
    const double paddedValues[] = {1.0, 1.0};
    expect(beagleSetTransitionMatrices(instance, denseIndices, denseAB.data(), paddedValues, 2) == BEAGLE_SUCCESS,
           "setTransitionMatrices", impl, n);
    {
        const int first[] = {0}, second[] = {kDenseB}, out[] = {kConvolved};
        const int code = beagleConvolveTransitionMatrices(instance, first, second, out, 1);
        expect(spectral ? code == BEAGLE_ERROR_OUT_OF_RANGE : code == BEAGLE_SUCCESS,
               "convolve with an eigen-only input", impl, n);
        const int input[] = {1}, transposed[] = {kTransposed};
        const int transposeCode = beagleTransposeTransitionMatrices(instance, input, transposed, 1);
        expect(spectral ? transposeCode == BEAGLE_ERROR_OUT_OF_RANGE : transposeCode == BEAGLE_SUCCESS,
               "transpose with an eigen-only input", impl, n);
    }
    {
        const int first[] = {kDenseA}, second[] = {kDenseB}, out[] = {kConvolved};
        expect(beagleConvolveTransitionMatrices(instance, first, second, out, 1) == BEAGLE_SUCCESS,
               "convolve", impl, n);
        const int input[] = {kDenseA}, transposed[] = {kTransposed};
        expect(beagleTransposeTransitionMatrices(instance, input, transposed, 1) == BEAGLE_SUCCESS,
               "transpose", impl, n);
    }

    // the edge log likelihood reads its matrix densely
    {
        const int parent[] = {5}, child[] = {4}, eigenOnly[] = {4}, dense[] = {kConvolved};
        double logL = 0.0;
        const int code = beagleCalculateEdgeLogLikelihoods(instance, parent, child, eigenOnly, NULL, NULL,
                                                           &weightsIndex, &frequenciesIndex, &scaleIndex, 1,
                                                           &logL, NULL, NULL);
        expect(spectral ? code == BEAGLE_ERROR_OUT_OF_RANGE : code == BEAGLE_SUCCESS,
               "calculateEdgeLogLikelihoods with an eigen-only index", impl, n);
        expect(beagleCalculateEdgeLogLikelihoods(instance, parent, child, dense, NULL, NULL, &weightsIndex,
                                                 &frequenciesIndex, &scaleIndex, 1, &result.edgeLogL,
                                                 NULL, NULL) == BEAGLE_SUCCESS,
               "calculateEdgeLogLikelihoods", impl, n);
    }

    // getTransitionMatrix: dense contents first, then the eigen decomposition, otherwise OUT_OF_RANGE
    std::vector<double> matrix(n * n * categories);
    result.dense.clear();
    for (int index : {kDenseA, kDenseB, kConvolved, kTransposed, kQComplex}) {
        expect(beagleGetTransitionMatrix(instance, index, matrix.data()) == BEAGLE_SUCCESS,
               "getTransitionMatrix of a dense index", impl, n);
        result.dense.insert(result.dense.end(), matrix.begin(), matrix.end());
    }
    expect(std::equal(result.dense.end() - matrix.size(), result.dense.end(), qComplex.begin()),
           "getTransitionMatrix returns the dense copy (with negative entries) exactly", impl, n);
    expect(std::equal(result.dense.begin(), result.dense.begin() + 2 * matrix.size(), denseAB.begin()),
           "getTransitionMatrix returns the matrices set by setTransitionMatrices exactly", impl, n);
    result.eigenMatrix.assign(n * n * categories, 0.0);
    expect(beagleGetTransitionMatrix(instance, 0, result.eigenMatrix.data()) == BEAGLE_SUCCESS,
           "getTransitionMatrix of an eigen-only index", impl, n);
    if (spectral) {
        expect(beagleGetTransitionMatrix(instance, kUnwritten, matrix.data()) == BEAGLE_ERROR_OUT_OF_RANGE,
               "getTransitionMatrix of an index never written", impl, n);
    }

    // indices outside [0, matrixBufferCount); only the setters and spectral getTransitionMatrix check them always
    for (int bad : {-1, int(kMatrixCount)}) {
        if (spectral) {
            expect(beagleGetTransitionMatrix(instance, bad, matrix.data()) == BEAGLE_ERROR_OUT_OF_RANGE,
                   "getTransitionMatrix of an out-of-range index", impl, n);
        }
        expect(beagleSetTransitionMatrix(instance, bad, denseAB.data(), 1.0) == BEAGLE_ERROR_OUT_OF_RANGE,
               "setTransitionMatrix of an out-of-range index", impl, n);
        expect(beagleSetDifferentialMatrix(instance, bad, qReal.data()) == BEAGLE_ERROR_OUT_OF_RANGE,
               "setDifferentialMatrix of an out-of-range index", impl, n);
        const int indices[] = {kDenseA, bad};
        expect(beagleSetTransitionMatrices(instance, indices, denseAB.data(), paddedValues, 2) ==
               BEAGLE_ERROR_OUT_OF_RANGE, "setTransitionMatrices with an out-of-range index", impl, n);
    }

    beagleFinalizeInstance(instance);
    return true;
}

int main() {
    for (long vector : {BEAGLE_FLAG_VECTOR_NONE, BEAGLE_FLAG_VECTOR_SSE}) {
        for (int n : {4, 17}) {
            Result standard, spectral;
            const bool reference = evaluate(false, BEAGLE_FLAG_VECTOR_SSE, n, standard) ||
                                   evaluate(false, BEAGLE_FLAG_VECTOR_NONE, n, standard); // no SSE on Linux ARM
            if (!reference || !evaluate(true, vector, n, spectral)) {
                return 1;
            }
            const double derivatives = std::max(relativeDifference(standard.derivatives, spectral.derivatives),
                                                relativeDifference(standard.sumDerivatives, spectral.sumDerivatives));
            const double edge = std::fabs(standard.edgeLogL - spectral.edgeLogL) / std::fabs(standard.edgeLogL);
            const bool dense = standard.dense == spectral.dense;
            double eigen = 0.0;
            for (size_t i = 0; i < standard.eigenMatrix.size(); ++i) {
                eigen = std::max(eigen, std::fabs(standard.eigenMatrix[i] - spectral.eigenMatrix[i]));
            }
            expect(derivatives < 1e-12, "edge derivatives differ from the standard implementation",
                   spectral.implName, n);
            expect(edge < 1e-12, "edge log likelihood differs from the standard implementation", spectral.implName, n);
            expect(dense, "dense matrices differ from the standard implementation", spectral.implName, n);
            expect(eigen < 1e-12, "eigen-derived matrix differs from the standard implementation", spectral.implName,
                   n);
            printf("%-26s vs %-22s %2d states: derivatives %.1e, edge log likelihood %.1e, dense matrices %s, "
                   "eigen matrix %.1e\n", spectral.implName, standard.implName, n, derivatives, edge,
                   dense ? "identical" : "DIFFER", eigen);
        }
    }
    printf("%s\n", failures == 0 ? "spectraldensetest: all checks passed" : "spectraldensetest: FAILED");
    return failures == 0 ? 0 : 1;
}
