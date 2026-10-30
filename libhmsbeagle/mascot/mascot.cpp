/*
 * mascot.cpp -- the MASCOT C API (mascot.h): instance table, kernel dispatch and argument checks.
 *
 * Kernel choice depends only on K and the instruction set, never on the FP mode, so a strict instance validates the
 * kernel an FMA instance runs. Measured on NEON (M1 Ultra, FMA build, maxStep 0.02): kernel A, which never pads, is
 * faster for K <= 7 except K = 6; kernel B (SIMD width 2, K padded to even) is faster for K = 6 and K >= 8. x86 uses
 * the same table until it has been timed on AVX2 hardware. Where kernel B is not built (MSVC), kernel A serves every K.
 */
#include "mascot.h"
#include "MascotProblem.hpp"

#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <vector>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#define MASCOT_X86 1
#if defined(_MSC_VER) && !defined(__clang__)
#include <immintrin.h>
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

namespace {

using mascot::Kernel;

#ifdef MASCOT_X86
// CPUID leaf, subleaf 0: r = eax, ebx, ecx, edx.
void cpuid(unsigned leaf, unsigned r[4]) {
#if defined(_MSC_VER) && !defined(__clang__)
    int v[4];
    __cpuidex(v, (int) leaf, 0);
    for (int i = 0; i < 4; i++) r[i] = (unsigned) v[i];
#else
    __cpuid_count(leaf, 0, r[0], r[1], r[2], r[3]);
#endif
}

// XCR0, the register state the OS saves (valid only with CPUID.1:ECX.OSXSAVE).
unsigned long long xcr0() {
#if defined(_MSC_VER) && !defined(__clang__)
    return _xgetbv(0);
#else
    unsigned eax, edx;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return ((unsigned long long) edx << 32) | eax;
#endif
}
#endif

/*
 * x86: the FMA build is VEX-encoded (-mfma implies AVX; MSVC's FMA intrinsic is VEX), so it needs the FMA instructions,
 * AVX, and an OS that saves the AVX register state: CPUID.1:ECX bits FMA (12), OSXSAVE (27) and AVX (28), then XCR0
 * bits 1 and 2 (XMM and YMM state). Each is tested on its own; a hypervisor or emulator (Rosetta 2 reports none of
 * them) may hide any one. AVX2 is not needed while kernel B's x86 vectors are 128 bits wide.
 */
bool cpuHasFma() {
#if defined(__aarch64__) || defined(_M_ARM64)
    return true;
#elif defined(MASCOT_X86)
    unsigned r[4];
    cpuid(0, r);
    if (r[0] < 1) return false;
    cpuid(1, r);
    const bool fma = (r[2] >> 12) & 1, osxsave = (r[2] >> 27) & 1, avx = (r[2] >> 28) & 1;
    if (!fma || !osxsave || !avx) return false;
    return (xcr0() & 0x6) == 0x6;
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
    return !MASCOT_HAVE_KERNEL_B || (stateCount <= 7 && stateCount != 6);
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
    if ((flags & MASCOT_FORCE_KERNEL_B) && !MASCOT_HAVE_KERNEL_B) return MASCOT_ERROR_UNSUPPORTED;
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
    if (stepRule->kind == MASCOT_STEPS_FIXED) {
        if (!(stepRule->maxStep > 0.0) || !std::isfinite(stepRule->maxStep)) return MASCOT_ERROR_ARGUMENT;
    } else if (stepRule->kind == MASCOT_STEPS_ADAPTIVE) {
        if (!(stepRule->tau > 0.0) || !std::isfinite(stepRule->tau) || std::isnan(stepRule->maxStep) ||
            std::isnan(stepRule->hMin)) {
            return MASCOT_ERROR_ARGUMENT;
        }
    } else {
        return MASCOT_ERROR_ARGUMENT;
    }
    if (checkpointReadSet < -1 || checkpointReadSet > 1 || checkpointWriteSet < -1 || checkpointWriteSet > 1) {
        return MASCOT_ERROR_ARGUMENT;
    }

    mascot::Problem p;
    p.K = in->stateCount;
    p.lineageCount = in->lineageCount;
    p.intervalCount = intervalCount;
    p.initialSampleNode = initialSampleNode;
    p.epochCount = epochCount;
    p.stepRuleKind = stepRule->kind == MASCOT_STEPS_ADAPTIVE ? mascot::STEPS_ADAPTIVE : mascot::STEPS_FIXED;
    p.maxStep = stepRule->maxStep;
    p.tau = stepRule->tau;
    p.minStep = stepRule->hMin;
    p.maxStepsPerSegment = stepRule->maxStepsPerInterval;
    p.stepBudget = stepRule->stepBudget;
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
    // checkpoints serve forward-only calls; gradient and ancestral-score calls integrate fully
    const bool forwardOnly = !wantTheta && !wantClock && !wantAncestral;
    p.checkpointReadSet = forwardOnly ? checkpointReadSet : -1;
    p.checkpointWriteSet = forwardOnly ? checkpointWriteSet : -1;

    try {
        if (forwardOnly) {
            return in->kernel->forward(p, logLikelihood);
        }
        double* theta = wantTheta ? thetaGradient : in->thetaScratch.data();
        return in->kernel->forwardAndReverse(p, logLikelihood, theta, wantClock ? clockGradient : nullptr,
                                             wantAncestral ? ancestralStateScores : nullptr);
    } catch (const std::bad_alloc&) {
        return MASCOT_ERROR_OUT_OF_MEMORY;
    }
}

int mascotGetLastRestartInterval(int instance, int* interval) {
    Instance* in = lookup(instance);
    if (!in || !interval) return MASCOT_ERROR_ARGUMENT;
    *interval = in->kernel->lastRestartInterval();
    return MASCOT_OK;
}

int mascotGetLastStepCounts(int instance, int* steps, int capacity, int* count) {
    Instance* in = lookup(instance);
    if (!in || !count || (capacity > 0 && !steps)) return MASCOT_ERROR_ARGUMENT;
    return in->kernel->lastStepCounts(steps, capacity, count);
}

} // extern "C"
