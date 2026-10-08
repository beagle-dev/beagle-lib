/*
 * gpumatrixtest: transition-matrix calls on a GPU against the CPU
 *
 * setTransitionMatrices of consecutive indices that start above 0 (one host write per run of up to three matrices
 * on CUDA, one per matrix on OpenCL), getTransitionMatrix of each, convolveTransitionMatrices and
 * transposeTransitionMatrices. No other example makes these calls on a GPU. Each GPU result must equal the same
 * call on a CPU instance to single precision. Four (state count, category count) shapes, so that matrix strides are
 * padded when sub-buffer offsets are aligned (BEAGLE_DEBUG_OPENCL_ALIGN in a BEAGLE_DEBUG_MEMORY build).
 * The GPU convolution multiplies in the other order from the CPU (first x second on the CPU, second x first on the
 * GPU, which keeps matrices transposed), so a GPU product may match either CPU product; a wrong offset matches neither.
 *
 * Usage: gpumatrixtest [resource]   (default: the first GPU resource). Exit 0 pass, 1 fail, 77 no GPU.
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "libhmsbeagle/beagle.h"

static const int MATRIX_COUNT = 12;

static int createInstance(int stateCount, int categoryCount, int resource, bool gpu) {
    BeagleInstanceDetails details;
    long preference = gpu ? (BEAGLE_FLAG_PROCESSOR_GPU | BEAGLE_FLAG_PRECISION_SINGLE)
                          : (BEAGLE_FLAG_PROCESSOR_CPU | BEAGLE_FLAG_PRECISION_DOUBLE);
    int instance = beagleCreateInstance(2, 4, 0, stateCount, 8, 1, MATRIX_COUNT, categoryCount, 0,
                                        &resource, 1, preference, BEAGLE_FLAG_PREORDER_TRANSPOSE_MANUAL, &details);
    if (instance >= 0 && gpu) {
        fprintf(stdout, "  %s on %s\n", details.implName, details.resourceName);
    }
    return instance;
}

static double difference(int cpu, int cpuIndex, const std::vector<double>& actual) {
    std::vector<double> expected(actual.size());
    if (beagleGetTransitionMatrix(cpu, cpuIndex, expected.data()) != BEAGLE_SUCCESS) return 1.0;
    double error = 0.0;
    for (size_t i = 0; i < actual.size(); i++) {
        double e = fabs(actual[i] - expected[i]) / (fabs(expected[i]) + 1e-3);
        if (e > error || e != e) error = e;
    }
    return error;
}

// the GPU matrix at index against the CPU matrix at cpuIndex, or at alternativeIndex if that is not negative
static double compare(int cpu, int gpu, int index, int size, const char* what,
                      int cpuIndex = -1, int alternativeIndex = -1) {
    std::vector<double> actual(size);
    int rc = beagleGetTransitionMatrix(gpu, index, actual.data());
    if (rc != BEAGLE_SUCCESS) {
        fprintf(stdout, "  %-34s getTransitionMatrix returned %d on the GPU\n", what, rc);
        return 1.0;
    }
    double error = difference(cpu, cpuIndex < 0 ? index : cpuIndex, actual);
    if (alternativeIndex >= 0) {
        double alternative = difference(cpu, alternativeIndex, actual);
        if (alternative < error || error != error) error = alternative;
    }
    fprintf(stdout, "  %-34s %s\n", what, error < 1e-5 ? "ok" : "MISMATCH");
    return error;
}

static bool runCase(int stateCount, int categoryCount, int resource) {
    fprintf(stdout, "%d states, %d categories\n", stateCount, categoryCount);
    int cpu = createInstance(stateCount, categoryCount, 0, false);
    int gpu = createInstance(stateCount, categoryCount, resource, true);
    if (cpu < 0 || gpu < 0) {
        fprintf(stdout, "  instance creation failed (%d, %d)\n", cpu, gpu);
        return false;
    }

    // four stochastic-looking matrices for indices 2, 3, 4 and 5, set in one call
    const int size = stateCount * stateCount * categoryCount;
    const int setIndices[4] = { 2, 3, 4, 5 };
    std::vector<double> matrices(4 * size);
    for (int m = 0; m < 4; m++) {
        for (int c = 0; c < categoryCount; c++) {
            for (int i = 0; i < stateCount; i++) {
                double rowSum = 0.0;
                for (int j = 0; j < stateCount; j++) {
                    double v = 1.0 + ((m * 7 + c * 5 + i * 3 + j * 11) % 13) + (i == j ? 20.0 : 0.0);
                    matrices[m * size + (c * stateCount + i) * stateCount + j] = v;
                    rowSum += v;
                }
                for (int j = 0; j < stateCount; j++) {
                    matrices[m * size + (c * stateCount + i) * stateCount + j] /= rowSum;
                }
            }
        }
    }
    double padded = 1.0;
    beagleSetTransitionMatrices(cpu, setIndices, matrices.data(), &padded, 4);
    beagleSetTransitionMatrices(gpu, setIndices, matrices.data(), &padded, 4);

    bool ok = true; // cleared by any error that is not below 1e-5, NaN included
    char label[64];
    for (int m = 0; m < 4; m++) {
        snprintf(label, sizeof(label), "set and get, index %d", setIndices[m]);
        ok &= compare(cpu, gpu, setIndices[m], size, label) < 1e-5;
    }

    const int first[2] = { 2, 4 }, second[2] = { 3, 5 }, product[2] = { 6, 7 }, swapped[2] = { 10, 11 };
    beagleConvolveTransitionMatrices(cpu, first, second, product, 2);
    beagleConvolveTransitionMatrices(cpu, second, first, swapped, 2);
    beagleConvolveTransitionMatrices(gpu, first, second, product, 2);
    ok &= compare(cpu, gpu, 6, size, "convolve 2 and 3 -> 6", 6, 10) < 1e-5;
    ok &= compare(cpu, gpu, 7, size, "convolve 4 and 5 -> 7", 7, 11) < 1e-5;

    const int input[2] = { 3, 5 }, transposed[2] = { 8, 9 };
    beagleTransposeTransitionMatrices(cpu, input, transposed, 2);
    beagleTransposeTransitionMatrices(gpu, input, transposed, 2);
    ok &= compare(cpu, gpu, 8, size, "transpose 3 -> 8") < 1e-5;
    ok &= compare(cpu, gpu, 9, size, "transpose 5 -> 9") < 1e-5;

    beagleFinalizeInstance(cpu);
    beagleFinalizeInstance(gpu);
    return ok;
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
        !(resources->list[resource].supportFlags & BEAGLE_FLAG_PROCESSOR_GPU)) {
        fprintf(stdout, "SKIP: no GPU resource\n");
        return 77;
    }

    bool pass = true;
    pass &= runCase(4, 1, resource);
    pass &= runCase(4, 2, resource);
    pass &= runCase(17, 1, resource);
    pass &= runCase(20, 3, resource);
    fprintf(stdout, "%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
