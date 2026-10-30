/*
 * mascot.cpp -- the MASCOT C API (mascot.h): instance table, kernel dispatch and argument checks.
 *
 * Kernel choice depends only on K and the instruction set, never on the FP mode, so a strict instance validates the
 * kernel an FMA instance runs. Measured on NEON (M1 Ultra, FMA build, maxStep 0.02): kernel A, which never pads, is
 * faster for K <= 7 except K = 6; kernel B (SIMD width 2, K padded to even) is faster for K = 6 and K >= 8. x86 uses
 * the same table until it has been timed on AVX2 hardware.
 */
#include "mascot.h"
#include "MascotProblem.hpp"

#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <vector>

namespace {

using mascot::Kernel;

bool cpuHasFma() {
#if defined(__aarch64__) || defined(_M_ARM64)
    return true;
#elif (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    return __builtin_cpu_supports("fma");
#else
    return false;
#endif
}

const char* isaName(bool fma) {
#if defined(__aarch64__) || defined(_M_ARM64)
    (void) fma;
    return "neon";
#elif defined(__x86_64__) || defined(__i386__) || defined(_M_X64)
    return fma ? "sse2+fma" : "sse2";
#else
    (void) fma;
    return "scalar";
#endif
}

bool useKernelA(int stateCount) {
    return stateCount <= 7 && stateCount != 6;
}

struct Instance {
    int stateCount;
    int lineageCount;
    int maxEpochCount;
    bool fma;
    std::unique_ptr<Kernel> kernel;
    std::vector<double> tipPartials;
    bool tipsSet = false;
    std::vector<double> thetaScratch;
};

std::mutex tableMutex;
std::vector<std::unique_ptr<Instance>> table;

Instance* lookup(int instance) {
    std::lock_guard<std::mutex> lock(tableMutex);
    if (instance < 0 || instance >= (int) table.size()) return nullptr;
    return table[instance].get();
}

std::unique_ptr<Kernel> makeKernel(bool kernelA, bool fma) {
    if (kernelA) return fma ? mascot::makeKernelAFma() : mascot::makeKernelAStrict();
    return fma ? mascot::makeKernelBFma() : mascot::makeKernelBStrict();
}

} // namespace

extern "C" {

int mascotGetApiVersion(int* major, int* minor) {
    if (!major || !minor) return MASCOT_ERROR_ARGUMENT;
    *major = MASCOT_API_VERSION_MAJOR;
    *minor = MASCOT_API_VERSION_MINOR;
    return MASCOT_OK;
}

int mascotCreateInstance(int stateCount, int lineageCount, int maxEpochCount, long flags) {
    if (stateCount < 2 || lineageCount < 1 || maxEpochCount < 1) return MASCOT_ERROR_ARGUMENT;
    if ((flags & MASCOT_FP_STRICT) && (flags & MASCOT_FP_FMA)) return MASCOT_ERROR_ARGUMENT;
    if ((flags & MASCOT_FORCE_KERNEL_A) && (flags & MASCOT_FORCE_KERNEL_B)) return MASCOT_ERROR_ARGUMENT;
    try {
        std::unique_ptr<Instance> instance(new Instance());
        instance->stateCount = stateCount;
        instance->lineageCount = lineageCount;
        instance->maxEpochCount = maxEpochCount;
        // FMA is the default; it falls back to strict where the CPU has no fused multiply-add
        instance->fma = !(flags & MASCOT_FP_STRICT) && cpuHasFma();
        bool kernelA = (flags & MASCOT_FORCE_KERNEL_A) ? true
                     : (flags & MASCOT_FORCE_KERNEL_B) ? false
                     : useKernelA(stateCount);
        instance->kernel = makeKernel(kernelA, instance->fma);
        instance->tipPartials.assign((size_t) lineageCount * stateCount, 0.0);
        instance->thetaScratch.assign((size_t) maxEpochCount * (stateCount * (stateCount - 1) + stateCount), 0.0);

        std::lock_guard<std::mutex> lock(tableMutex);
        for (size_t i = 0; i < table.size(); i++) {
            if (!table[i]) {
                table[i] = std::move(instance);
                return (int) i;
            }
        }
        table.push_back(std::move(instance));
        return (int) table.size() - 1;
    } catch (const std::bad_alloc&) {
        return MASCOT_ERROR_OUT_OF_MEMORY;
    }
}

int mascotFinalizeInstance(int instance) {
    std::lock_guard<std::mutex> lock(tableMutex);
    if (instance < 0 || instance >= (int) table.size() || !table[instance]) return MASCOT_ERROR_ARGUMENT;
    table[instance].reset();
    return MASCOT_OK;
}

int mascotGetInstanceDetails(int instance, MascotInstanceDetails* details) {
    Instance* in = lookup(instance);
    if (!in || !details) return MASCOT_ERROR_ARGUMENT;
    details->kernel = in->kernel->name();
    details->simdWidth = in->kernel->simdWidth();
    details->fpMode = in->fma ? MASCOT_FP_FMA : MASCOT_FP_STRICT;
    details->isa = isaName(in->fma);
    return MASCOT_OK;
}

int mascotSetTipPartials(int instance, const double* tipPartials) {
    Instance* in = lookup(instance);
    if (!in || !tipPartials) return MASCOT_ERROR_ARGUMENT;
    std::memcpy(in->tipPartials.data(), tipPartials, sizeof(double) * in->tipPartials.size());
    in->tipsSet = true;
    return MASCOT_OK;
}

int mascotEvaluate(int instance,
                   int intervalCount, double initialTime, int initialSampleNode,
                   const double* intervalLengths, const int* eventTypes, const int* sampleNodes,
                   const int* coalescentChild1, const int* coalescentChild2, const int* coalescentParent,
                   int epochCount, const double* boundaries,
                   const double* migrationMatrices,
                   const double* inversePopulations,
                   const double* branchRates,
                   const MascotStepRule* stepRule,
                   int checkProbabilities,
                   int checkpointReadSet, int checkpointWriteSet,
                   int outputs,
                   double* logLikelihood,
                   double* thetaGradient,
                   double* clockGradient,
                   double* ancestralStateScores) {
    Instance* in = lookup(instance);
    if (!in || !in->tipsSet) return MASCOT_ERROR_ARGUMENT;
    if (intervalCount < 0 || initialSampleNode < 0 || initialSampleNode >= in->lineageCount) {
        return MASCOT_ERROR_ARGUMENT;
    }
    if (intervalCount > 0 && (!intervalLengths || !eventTypes || !sampleNodes || !coalescentChild1 ||
                              !coalescentChild2 || !coalescentParent)) {
        return MASCOT_ERROR_ARGUMENT;
    }
    if (epochCount < 1 || epochCount > in->maxEpochCount || !boundaries || !migrationMatrices ||
        !inversePopulations || !stepRule || !logLikelihood) {
        return MASCOT_ERROR_ARGUMENT;
    }
    if (!(outputs & MASCOT_OUT_LOGL)) return MASCOT_ERROR_ARGUMENT;
    const bool wantTheta = (outputs & MASCOT_OUT_THETA_GRADIENT) != 0;
    const bool wantClock = (outputs & MASCOT_OUT_CLOCK_GRADIENT) != 0;
    const bool wantAncestral = (outputs & MASCOT_OUT_ANCESTRAL_SCORES) != 0;
    if ((wantTheta && !thetaGradient) || (wantClock && (!clockGradient || !branchRates)) ||
        (wantAncestral && !ancestralStateScores)) {
        return MASCOT_ERROR_ARGUMENT;
    }
    if (stepRule->kind == MASCOT_STEPS_ADAPTIVE) return MASCOT_ERROR_UNSUPPORTED;   // R3
    if (stepRule->kind != MASCOT_STEPS_FIXED || !(stepRule->maxStep > 0.0) || !std::isfinite(stepRule->maxStep)) {
        return MASCOT_ERROR_ARGUMENT;
    }
    if (checkpointReadSet >= 0 || checkpointWriteSet >= 0) return MASCOT_ERROR_UNSUPPORTED;     // R4

    mascot::Problem p;
    p.K = in->stateCount;
    p.lineageCount = in->lineageCount;
    p.intervalCount = intervalCount;
    p.initialSampleNode = initialSampleNode;
    p.epochCount = epochCount;
    p.maxStep = stepRule->maxStep;
    p.initialTime = initialTime;
    p.boundaries = boundaries;
    p.migration = migrationMatrices;
    p.inversePopulation = inversePopulations;
    p.branchRates = branchRates;
    p.intervalLengths = intervalLengths;
    p.eventTypes = eventTypes;
    p.sampleNodes = sampleNodes;
    p.child1 = coalescentChild1;
    p.child2 = coalescentChild2;
    p.parent = coalescentParent;
    p.tipPartials = in->tipPartials.data();
    p.checkProbabilities = checkProbabilities != 0;

    try {
        if (!wantTheta && !wantClock && !wantAncestral) {
            return in->kernel->forward(p, logLikelihood);
        }
        double* theta = wantTheta ? thetaGradient : in->thetaScratch.data();
        return in->kernel->forwardAndReverse(p, logLikelihood, theta, wantClock ? clockGradient : nullptr,
                                             wantAncestral ? ancestralStateScores : nullptr);
    } catch (const std::bad_alloc&) {
        return MASCOT_ERROR_OUT_OF_MEMORY;
    }
}

int mascotGetLastStepCounts(int instance, int* steps, int capacity, int* count) {
    (void) steps;
    (void) capacity;
    (void) count;
    if (!lookup(instance)) return MASCOT_ERROR_ARGUMENT;
    return MASCOT_ERROR_UNSUPPORTED;   // R3
}

} // extern "C"
