/*
 * Copyright 2026 Phylogenetic Likelihood Working Group
 * This file is part of BEAGLE.
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 *
 * Dense matrix storage on the GPU spectral implementation, which allocates its matrix pool only at the first dense
 * write. Against the CPU spectral implementation (double precision), at 4 and 17 states, with pre-order transposes
 * MANUAL and AUTO:
 *   - setDifferentialMatrix followed by calculateEdgeDerivatives, and calculateCrossProducts, agree to single
 *     precision (relative 1e-4 of the largest value), and so does the standard GPU implementation against the
 *     standard CPU one. Both sum over the padded states, which must stay zero in every partial. With MANUAL
 *     and more than 4 states the GPU follows the caller's transposes (as in hmctest5): it takes Q^T, and the
 *     standard implementation takes transposed matrices in pre-order operations (spectral transposes them itself);
 *   - getTransitionMatrix returns matrices set by setTransitionMatrices (and, with MANUAL, setDifferentialMatrix);
 *   - a dense read of an index without dense contents returns BEAGLE_ERROR_OUT_OF_RANGE: calculateEdgeDerivatives,
 *     and getTransitionMatrix of an index written only by updateTransitionMatrices or never written;
 *   - matrix indices outside [0, matrixBufferCount) return BEAGLE_ERROR_OUT_OF_RANGE from the setters and from
 *     getTransitionMatrix;
 *   - convolve, transpose, the edge log likelihoods, updateTransitionMatricesWithModelCategories, pattern
 *     partitions and the ...ByPartition updates return BEAGLE_ERROR_NO_IMPLEMENTATION;
 *   - an instance that requires BEAGLE_FLAG_SCALING_DYNAMIC is not created; one that prefers it is created
 *     without it.
 * Usage: gpuspectraldensetest [resource]   (default: the first GPU resource). Exit 0 pass, 1 fail, 77 no GPU.
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include "libhmsbeagle/beagle.h"

// circulant CTMC: complex conjugate eigenvalue pairs unless rFwd == rBkd (as in spectraldensetest)
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

static void expect(bool ok, const char* what, const char* impl, int n, const char* transpose) {
    if (!ok) {
        printf("FAIL %-24s %2d states, %s: %s\n", impl, n, transpose, what);
        ++failures;
    }
}

// largest |x - y| relative to the largest |y|; infinite if any entry is NaN (so every later comparison fails)
static double scaledDifference(const std::vector<double>& x, const std::vector<double>& y) {
    double worst = 0.0, scale = 0.0;
    for (size_t i = 0; i < y.size(); ++i) scale = std::max(scale, std::fabs(y[i]));
    for (size_t i = 0; i < x.size(); ++i) {
        const double d = std::fabs(x[i] - y[i]) / scale;
        if (std::isnan(d)) return HUGE_VAL;
        if (d > worst) worst = d;
    }
    return worst;
}

enum { kQComplex = 6, kQReal = 7, kDenseA = 8, kDenseB = 9, kSpare = 10, kUnwritten = 11, kMatrixCount = 12 };

// With BEAGLE_FLAG_PREORDER_TRANSPOSE_MANUAL and more than 4 states, the GPU takes the caller's transposes (as in
// hmctest5): Q^T for setDifferentialMatrix and, in the standard implementation, transposed matrices in pre-order
// operations. The CPU and the 4-state GPU kernels take Q and the matrices as they are.
static bool callerTransposes(bool gpu, int n, long transposeFlag) {
    return gpu && n > 4 && transposeFlag == BEAGLE_FLAG_PREORDER_TRANSPOSE_MANUAL;
}

// the differential matrix of eigen-system 0, or its transpose (which swaps the circulant's two rates)
static std::vector<double> qComplexMatrix(int n, bool transposed) {
    return transposed ? circulantQ(n, 0.4, 1.0, 2) : circulantQ(n, 1.0, 0.4, 2);
}

// tree ((0,1)4,(2,3)5)6 as in spectraldensetest; returns the instance, or a negative code. A standard instance has
// 6 more matrices, kMatrixCount + i holding the transpose of branch i's matrix when callerTransposes.
static int setUp(bool gpu, bool spectral, int resource, long transposeFlag, int n, const char** implName) {
    const int tips = 4, patterns = 7, categories = 2, nodes = 7, root = 6, preBase = nodes;
    const long requirements = (gpu ? BEAGLE_FLAG_PROCESSOR_GPU | BEAGLE_FLAG_PRECISION_SINGLE
                                   : BEAGLE_FLAG_PROCESSOR_CPU | BEAGLE_FLAG_PRECISION_DOUBLE) |
                              BEAGLE_FLAG_EIGEN_COMPLEX | BEAGLE_FLAG_SCALING_MANUAL |
                              (spectral ? BEAGLE_FLAG_SPECTRAL_REPRESENTATION : 0) | transposeFlag;
    const int matrixCount = spectral ? kMatrixCount : kMatrixCount + 6;
    BeagleInstanceDetails details;
    const int instance = beagleCreateInstance(tips, 2 * nodes, 1, n, patterns, 2, matrixCount, categories, 0,
                                              gpu ? &resource : NULL, gpu ? 1 : 0, 0, requirements, &details);
    if (instance < 0) return instance;
    *implName = details.implName;

    std::vector<double> evec, ivec, eval;
    buildCirculant(n, 1.0, 0.4, evec, ivec, eval);
    beagleSetEigenDecomposition(instance, 0, evec.data(), ivec.data(), eval.data());
    buildCirculant(n, 0.7, 0.7, evec, ivec, eval);
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
    for (int k = 0; k < patterns; ++k) states[k] = (k * 3) % n;
    beagleSetTipStates(instance, 0, states.data());
    for (int tip = 1; tip < tips; ++tip) {
        std::vector<double> partials(n * patterns);
        for (int k = 0; k < patterns; ++k)
            for (int s = 0; s < n; ++s)
                partials[k * n + s] = 0.1 + ((s * 7 + k * 3 + tip * 5) % (n + 2)) / double(n + 2);
        beagleSetTipPartials(instance, tip, partials.data());
    }

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

    // the matrix of a node's own branch in its pre-order operation (spectral transposes it itself)
    int own = 0;
    if (!spectral && callerTransposes(gpu, n, transposeFlag)) {
        const int branches[] = {0, 1, 2, 3, 4, 5};
        const int transposed[] = {kMatrixCount, kMatrixCount + 1, kMatrixCount + 2, kMatrixCount + 3,
                                  kMatrixCount + 4, kMatrixCount + 5};
        beagleTransposeTransitionMatrices(instance, branches, transposed, 6);
        own = kMatrixCount;
    }
    std::vector<double> prior(n * patterns * categories, 1.0 / n);
    beagleSetPartials(instance, preBase + root, prior.data());
    const BeagleOperation pre[] = {
            {preBase + 4, BEAGLE_OP_NONE, BEAGLE_OP_NONE, preBase + root, own + 4, 5, 5},
            {preBase + 5, BEAGLE_OP_NONE, BEAGLE_OP_NONE, preBase + root, own + 5, 4, 4},
            {preBase + 0, BEAGLE_OP_NONE, BEAGLE_OP_NONE, preBase + 4, own + 0, 1, 1},
            {preBase + 1, BEAGLE_OP_NONE, BEAGLE_OP_NONE, preBase + 4, own + 1, 0, 0},
            {preBase + 2, BEAGLE_OP_NONE, BEAGLE_OP_NONE, preBase + 5, own + 2, 3, 3},
            {preBase + 3, BEAGLE_OP_NONE, BEAGLE_OP_NONE, preBase + 5, own + 3, 2, 2}};
    beagleUpdatePrePartials(instance, pre, 6, BEAGLE_OP_NONE);
    return instance;
}

static const int postIndices[] = {0, 1, 2, 3, 4, 5};
static const int preIndices[] = {7, 8, 9, 10, 11, 12};
static const int derivativeIndices[] = {kQComplex, kQReal, kQComplex, kQReal, kQComplex, kQReal};

// setDifferentialMatrix (Q, or Q^T when transposed) and calculateEdgeDerivatives; derivatives per node and pattern,
// then the per-node sums
static int derivatives(int instance, int n, bool transposed, std::vector<double>& out) {
    const int categories = 2, patterns = 7;
    const std::vector<double> qComplex = qComplexMatrix(n, transposed);
    const std::vector<double> qReal = circulantQ(n, 0.7, 0.7, categories); // symmetric
    int code = beagleSetDifferentialMatrix(instance, kQComplex, qComplex.data());
    if (code == BEAGLE_SUCCESS) code = beagleSetDifferentialMatrix(instance, kQReal, qReal.data());
    if (code != BEAGLE_SUCCESS) return code;
    const int weightsIndex = 0;
    std::vector<double> perPattern(6 * patterns), sums(6);
    code = beagleCalculateEdgeDerivatives(instance, postIndices, preIndices, derivativeIndices, &weightsIndex, 6,
                                          perPattern.data(), sums.data(), NULL);
    out = perPattern;
    out.insert(out.end(), sums.begin(), sums.end());
    return code;
}

// calculateCrossProducts over the same six edges: the stateCount x stateCount sum
static int crossProducts(int instance, int n, std::vector<double>& out) {
    const double edgeLengths[] = {0.3, 0.5, 0.7, 0.4, 0.2, 0.6}; // by node, as in setUp
    const int ratesIndex = 0, weightsIndex = 0;
    out.assign(n * n, 0.0);
    return beagleCalculateCrossProductDerivative(instance, postIndices, preIndices, &ratesIndex, &weightsIndex,
                                                 edgeLengths, 6, out.data(), NULL);
}

// edge derivatives (as BEAST's branch-rate gradient uses them) and cross products, GPU against CPU; returns the
// relative differences {derivatives, cross products}
static std::pair<double, double> compareDerivatives(int cpu, int gpu, const char* cpuName, const char* gpuName, int n,
                                                    long transposeFlag, const char* transpose) {
    std::vector<double> cpuDerivatives, gpuDerivatives, cpuCross, gpuCross;
    expect(derivatives(cpu, n, false, cpuDerivatives) == BEAGLE_SUCCESS, "CPU derivatives", cpuName, n, transpose);
    expect(derivatives(gpu, n, callerTransposes(true, n, transposeFlag), gpuDerivatives) == BEAGLE_SUCCESS,
           "setDifferentialMatrix, calculateEdgeDerivatives", gpuName, n, transpose);
    const double derivativeError = scaledDifference(gpuDerivatives, cpuDerivatives);
    expect(derivativeError < 1e-4, "edge derivatives differ from the CPU", gpuName, n, transpose);
    expect(crossProducts(cpu, n, cpuCross) == BEAGLE_SUCCESS, "CPU cross products", cpuName, n, transpose);
    expect(crossProducts(gpu, n, gpuCross) == BEAGLE_SUCCESS, "calculateCrossProducts", gpuName, n, transpose);
    const double crossError = scaledDifference(gpuCross, cpuCross);
    expect(crossError < 1e-4, "cross products differ from the CPU", gpuName, n, transpose);
    return {derivativeError, crossError};
}

static void runCase(int resource, int n, long transposeFlag) {
    const char* transpose = transposeFlag == BEAGLE_FLAG_PREORDER_TRANSPOSE_AUTO ? "AUTO" : "MANUAL";
    const int categories = 2, size = n * n * categories;
    const char* cpuName = "";
    const char* gpuName = "";
    const int cpu = setUp(false, true, 0, transposeFlag, n, &cpuName);
    const int gpu = setUp(true, true, resource, transposeFlag, n, &gpuName);
    if (cpu < 0 || gpu < 0) {
        printf("FAIL %d states, %s: instance creation returned %d (CPU) and %d (GPU)\n", n, transpose, cpu, gpu);
        ++failures;
        return;
    }

    // dense reads before any dense write: the matrices hold eigen-systems only
    std::vector<double> matrix(size);
    const int weightsIndex = 0;
    {
        std::vector<double> perPattern(6 * 7), sums(6);
        const int eigenOnly[] = {0, 1, 2, 3, 4, 5};
        expect(beagleCalculateEdgeDerivatives(gpu, postIndices, preIndices, eigenOnly, &weightsIndex, 6,
                                              perPattern.data(), sums.data(), NULL) == BEAGLE_ERROR_OUT_OF_RANGE,
               "calculateEdgeDerivatives with eigen-only derivative indices", gpuName, n, transpose);
    }
    expect(beagleGetTransitionMatrix(gpu, 0, matrix.data()) == BEAGLE_ERROR_OUT_OF_RANGE,
           "getTransitionMatrix of an eigen-only index", gpuName, n, transpose);

    // the dense derivatives
    const std::pair<double, double> errors = compareDerivatives(cpu, gpu, cpuName, gpuName, n, transposeFlag,
                                                                transpose);

    // dense round trips
    std::vector<double> denseAB = stochastic(n, categories, 1);
    const std::vector<double> denseB = stochastic(n, categories, 2);
    denseAB.insert(denseAB.end(), denseB.begin(), denseB.end());
    const int denseIndices[] = {kDenseA, kDenseB};
    const double paddedValues[] = {1.0, 1.0};
    expect(beagleSetTransitionMatrices(gpu, denseIndices, denseAB.data(), paddedValues, 2) == BEAGLE_SUCCESS,
           "setTransitionMatrices", gpuName, n, transpose);
    double roundTrip = 0.0;
    for (int m = 0; m < 2; m++) {
        if (beagleGetTransitionMatrix(gpu, denseIndices[m], matrix.data()) != BEAGLE_SUCCESS) {
            roundTrip = 1.0;
            continue;
        }
        std::vector<double> expected(denseAB.begin() + m * size, denseAB.begin() + (m + 1) * size);
        roundTrip = std::max(roundTrip, scaledDifference(matrix, expected));
    }
    if (transposeFlag == BEAGLE_FLAG_PREORDER_TRANSPOSE_MANUAL) { // with AUTO the differential matrix is stored as given
        // as derivatives() set it
        const std::vector<double> qComplex = qComplexMatrix(n, callerTransposes(true, n, transposeFlag));
        if (beagleGetTransitionMatrix(gpu, kQComplex, matrix.data()) == BEAGLE_SUCCESS) {
            roundTrip = std::max(roundTrip, scaledDifference(matrix, qComplex));
        } else {
            roundTrip = 1.0;
        }
    }
    expect(roundTrip < 1e-6, "getTransitionMatrix of a dense index", gpuName, n, transpose);
    expect(beagleGetTransitionMatrix(gpu, kUnwritten, matrix.data()) == BEAGLE_ERROR_OUT_OF_RANGE,
           "getTransitionMatrix of an index never written", gpuName, n, transpose);

    // indices outside [0, matrixBufferCount)
    for (int bad : {-1, int(kMatrixCount)}) {
        expect(beagleGetTransitionMatrix(gpu, bad, matrix.data()) == BEAGLE_ERROR_OUT_OF_RANGE,
               "getTransitionMatrix of an out-of-range index", gpuName, n, transpose);
        expect(beagleSetTransitionMatrix(gpu, bad, denseAB.data(), 1.0) == BEAGLE_ERROR_OUT_OF_RANGE,
               "setTransitionMatrix of an out-of-range index", gpuName, n, transpose);
        expect(beagleSetDifferentialMatrix(gpu, bad, denseAB.data()) == BEAGLE_ERROR_OUT_OF_RANGE,
               "setDifferentialMatrix of an out-of-range index", gpuName, n, transpose);
        const int indices[] = {kSpare, bad};
        expect(beagleSetTransitionMatrices(gpu, indices, denseAB.data(), paddedValues, 2) ==
               BEAGLE_ERROR_OUT_OF_RANGE, "setTransitionMatrices with an out-of-range index", gpuName, n, transpose);
    }
    expect(beagleGetTransitionMatrix(gpu, kSpare, matrix.data()) == BEAGLE_ERROR_OUT_OF_RANGE,
           "a rejected setTransitionMatrices writes nothing", gpuName, n, transpose);

    // paths that read P matrices spectral never computes, and partitions
    {
        const int first[] = {kDenseA}, second[] = {kDenseB}, out[] = {kSpare};
        expect(beagleConvolveTransitionMatrices(gpu, first, second, out, 1) == BEAGLE_ERROR_NO_IMPLEMENTATION,
               "convolveTransitionMatrices", gpuName, n, transpose);
        expect(beagleTransposeTransitionMatrices(gpu, first, out, 1) == BEAGLE_ERROR_NO_IMPLEMENTATION,
               "transposeTransitionMatrices", gpuName, n, transpose);
        const int parent[] = {5}, child[] = {4}, dense[] = {kDenseA}, frequencies = 0, scale = BEAGLE_OP_NONE;
        double logL = 0.0;
        expect(beagleCalculateEdgeLogLikelihoods(gpu, parent, child, dense, NULL, NULL, &weightsIndex, &frequencies,
                                                 &scale, 1, &logL, NULL, NULL) == BEAGLE_ERROR_NO_IMPLEMENTATION,
               "calculateEdgeLogLikelihoods", gpuName, n, transpose);
        int eigenIndices[] = {0, 1};
        const int probabilities[] = {0};
        const double lengths[] = {0.3};
        expect(beagleUpdateTransitionMatricesWithModelCategories(gpu, eigenIndices, probabilities, NULL, NULL,
                                                                 lengths, 1) == BEAGLE_ERROR_NO_IMPLEMENTATION,
               "updateTransitionMatricesWithModelCategories", gpuName, n, transpose);
        const int partitions[7] = {0, 0, 0, 1, 1, 1, 1};
        expect(beagleSetPatternPartitions(gpu, 2, partitions) == BEAGLE_ERROR_NO_IMPLEMENTATION,
               "setPatternPartitions", gpuName, n, transpose);
        const BeagleOperationByPartition operation = {4, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 0, 0, 1, 1, 0, BEAGLE_OP_NONE};
        expect(beagleUpdatePartialsByPartition(gpu, &operation, 1) == BEAGLE_ERROR_NO_IMPLEMENTATION,
               "updatePartialsByPartition", gpuName, n, transpose);
        expect(beagleUpdatePrePartialsByPartition(gpu, &operation, 1, BEAGLE_PARTIALS_BOTTOM) ==
               BEAGLE_ERROR_NO_IMPLEMENTATION,
               "updatePrePartialsByPartition", gpuName, n, transpose);
    }

    printf("%-24s vs %-28s %2d states, %-6s: derivatives %.1e, cross products %.1e, dense round trips %.1e\n",
           gpuName, cpuName, n, transpose, errors.first, errors.second, roundTrip);
    beagleFinalizeInstance(cpu);
    beagleFinalizeInstance(gpu);
}

// the standard implementations on the same tree: edge derivatives and cross products only
static void runStandardCase(int resource, int n, long transposeFlag) {
    const char* transpose = transposeFlag == BEAGLE_FLAG_PREORDER_TRANSPOSE_AUTO ? "AUTO" : "MANUAL";
    const char* cpuName = "";
    const char* gpuName = "";
    const int cpu = setUp(false, false, 0, transposeFlag, n, &cpuName);
    const int gpu = setUp(true, false, resource, transposeFlag, n, &gpuName);
    if (cpu < 0 || gpu < 0) {
        printf("FAIL %d states, %s: standard instance creation returned %d (CPU) and %d (GPU)\n", n, transpose, cpu,
               gpu);
        ++failures;
        return;
    }
    const std::pair<double, double> errors = compareDerivatives(cpu, gpu, cpuName, gpuName, n, transposeFlag,
                                                                transpose);
    printf("%-24s vs %-28s %2d states, %-6s: derivatives %.1e, cross products %.1e\n", gpuName, cpuName, n,
           transpose, errors.first, errors.second);
    beagleFinalizeInstance(cpu);
    beagleFinalizeInstance(gpu);
}

int main(int argc, const char* argv[]) {
    BeagleResourceList* resources = beagleGetResourceList();
    int resource = -1;
    if (argc > 1) {
        resource = atoi(argv[1]);
    } else {
        for (int i = 0; i < resources->length && resource < 0; i++) {
            if (resources->list[i].supportFlags & BEAGLE_FLAG_PROCESSOR_GPU) resource = i;
        }
    }
    if (resource < 0 || resource >= resources->length ||
        !(resources->list[resource].supportFlags & BEAGLE_FLAG_PROCESSOR_GPU) ||
        !(resources->list[resource].supportFlags & BEAGLE_FLAG_SPECTRAL_REPRESENTATION)) {
        printf("SKIP: no GPU resource with spectral support\n");
        return 77;
    }

    for (int n : {4, 17}) {
        for (long transposeFlag : {BEAGLE_FLAG_PREORDER_TRANSPOSE_MANUAL, BEAGLE_FLAG_PREORDER_TRANSPOSE_AUTO}) {
            runCase(resource, n, transposeFlag);
            runStandardCase(resource, n, transposeFlag);
        }
    }

    // BEAGLE's dynamic scaling passes transition matrices, which spectral never computes. Without scale buffers,
    // so that OpenCL's own refusal of dynamic scaling with scale buffers cannot be what refuses it.
    const long gpuSpectral = BEAGLE_FLAG_PROCESSOR_GPU | BEAGLE_FLAG_PRECISION_SINGLE |
                             BEAGLE_FLAG_SPECTRAL_REPRESENTATION;
    BeagleInstanceDetails details;
    const int plain = beagleCreateInstance(4, 14, 1, 4, 7, 1, kMatrixCount, 2, 0, &resource, 1, 0, gpuSpectral,
                                           &details);
    expect(plain >= 0, "an instance without SCALING_DYNAMIC is created", "GPU spectral", 4, "-");
    if (plain >= 0) beagleFinalizeInstance(plain);
    const int required = beagleCreateInstance(4, 14, 1, 4, 7, 1, kMatrixCount, 2, 0, &resource, 1, 0,
                                              gpuSpectral | BEAGLE_FLAG_SCALING_DYNAMIC, &details);
    expect(required < 0, "an instance that requires SCALING_DYNAMIC is refused", "GPU spectral", 4, "-");
    if (required >= 0) beagleFinalizeInstance(required);
    // a preferred SCALING_DYNAMIC is dropped (with scale buffers, which OpenCL dynamic scaling would refuse)
    const int preferred = beagleCreateInstance(4, 14, 1, 4, 7, 1, kMatrixCount, 2, 4, &resource, 1,
                                               BEAGLE_FLAG_SCALING_DYNAMIC, gpuSpectral, &details);
    expect(preferred >= 0 && !(details.flags & BEAGLE_FLAG_SCALING_DYNAMIC),
           "an instance that prefers SCALING_DYNAMIC is created without it", "GPU spectral", 4, "-");
    if (preferred >= 0) beagleFinalizeInstance(preferred);

    printf("%s\n", failures == 0 ? "gpuspectraldensetest: PASS" : "gpuspectraldensetest: FAIL");
    return failures == 0 ? 0 : 1;
}
