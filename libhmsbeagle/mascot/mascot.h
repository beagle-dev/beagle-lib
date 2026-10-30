/*
 * mascot.h -- C API of the MASCOT component in beagle-lib (libhmsbeagle/mascot/mascot.h).
 *
 * API v1.0 (approved 2026-10-30).
 *
 * Contract:
 *  - An instance is not re-entrant; distinct instances are independent (no static or thread_local state).
 *  - All arrays are copied in during the call; no pointer is retained after return.
 *  - Inputs and outputs use the caller's K. Any SIMD padding is internal and invisible.
 *  - Java computes the migration matrices and inverse population sizes (no exp() in native code).
 *  - With MASCOT_FP_STRICT, gradients and ancestral scores equal the Java reference (GenericMascotLikelihoodDelegate)
 *    bit for bit; logL may differ by an ulp of log(lambda) per coalescence, because the kernel's log() comes from
 *    the platform libm (many ulp of logL only when its terms cancel).
 *  - With MASCOT_FP_FMA (the default), results differ from strict in the last bits (tolerance set by the golden tests).
 *  - The kernel chosen for K (A or B) depends only on K and the instruction set, never on the FP mode.
 */
#ifndef MASCOT_H
#define MASCOT_H

#ifdef __cplusplus
extern "C" {
#endif

#define MASCOT_API_VERSION_MAJOR 1
#define MASCOT_API_VERSION_MINOR 0

enum MascotFlags {
    MASCOT_FP_STRICT       = 1 << 0,  /* no fused multiply-add: bitwise with Java on gradients                        */
    MASCOT_FP_FMA          = 1 << 1,  /* explicit fused multiply-add where the CPU has it (default if neither is set)  */
    MASCOT_FORCE_KERNEL_A  = 1 << 2,  /* tests only: use kernel A at any K                                             */
    MASCOT_FORCE_KERNEL_B  = 1 << 3   /* tests only: use kernel B at any K                                             */
};

enum MascotOutputs {
    MASCOT_OUT_LOGL               = 1 << 0,
    MASCOT_OUT_THETA_GRADIENT     = 1 << 1,  /* epochCount x (K(K-1) + K): migration rates, then log Ne              */
    MASCOT_OUT_CLOCK_GRADIENT     = 1 << 2,  /* lineageCount, per-lineage clock rate (requires branchRates)          */
    MASCOT_OUT_ANCESTRAL_SCORES   = 1 << 3   /* lineageCount x K, from the same reverse pass as the gradient          */
};

enum MascotStatus {
    MASCOT_OK                    =  0,
    MASCOT_ERROR_ARGUMENT        = -1,  /* a bug in the caller: Java -> IllegalStateException                        */
    MASCOT_ERROR_COALESCENT_RATE = -2,  /* lambda <= 0 or non-finite: Java -> NumericalException -> -inf             */
    MASCOT_ERROR_PROBABILITIES   = -3,  /* checkProbabilities violation: same mapping                                */
    MASCOT_ERROR_UNSUPPORTED     = -4,  /* feature not available in this build (e.g. ADAPTIVE before R3): Java falls back */
    MASCOT_ERROR_OUT_OF_MEMORY   = -5
};

enum MascotStepRuleKind {
    MASCOT_STEPS_FIXED    = 0,  /* steps = max(1, ceil(len / maxStep)), equal sub-steps (today's Java rule)          */
    MASCOT_STEPS_ADAPTIVE = 1   /* R3; returns MASCOT_ERROR_UNSUPPORTED until implemented. Proposed rule, per segment
                                 * (an interval split at epoch boundaries, as MascotForwardEvaluator.integrateSegment):
                                 *   lambda = c * max_k |M_kk| + max_k q_k * sum_l p_lk   (state at segment start;
                                 *            c = largest branch rate among active lineages, 1 without branch rates)
                                 *   n      = max(1, ceil(len * lambda / tau))
                                 *   n      = max(n, ceil(len / maxStep))        if maxStep > 0   (h never above maxStep)
                                 *   n      = min(n, floor(len / hMin))          if hMin > 0      (h never below hMin, n >= 1)
                                 *   n      = min(n, maxStepsPerInterval)        if > 0           (guard)
                                 * equal sub-steps; evaluated in strict FP in every build, identically in Java.        */
};

typedef struct {
    int    kind;                  /* MascotStepRuleKind                                                              */
    double maxStep;               /* FIXED: the step bound from the XML; ADAPTIVE: upper bound on h (<= 0: none)      */
    double tau;                   /* ADAPTIVE only (default 0.25)                                                     */
    double hMin;                  /* ADAPTIVE only: lower bound on h (<= 0: none)                                     */
    int    maxStepsPerInterval;   /* ADAPTIVE only: guard against pathological lambda (<= 0: none)                    */
} MascotStepRule;

typedef struct {
    char   kernel;                /* 'A' or 'B'                                                                       */
    int    simdWidth;             /* doubles per vector in kernel B (1, 2, 4); 0 for kernel A                         */
    int    fpMode;                /* MASCOT_FP_STRICT or MASCOT_FP_FMA, as actually used (FMA needs CPU support)      */
    const char* isa;              /* e.g. "neon", "sse2", "avx2+fma"; static string                                    */
} MascotInstanceDetails;

int mascotGetApiVersion(int* major, int* minor);

/* Returns an instance id >= 0, or a negative MascotStatus. lineageCount = max lineage id + 1. */
int mascotCreateInstance(int stateCount, int lineageCount, int maxEpochCount, long flags);
int mascotFinalizeInstance(int instance);

/* For the startup log line: which kernel, SIMD width, FP mode and ISA this instance uses. */
int mascotGetInstanceDetails(int instance, MascotInstanceDetails* details);

/* Registered once per run (StructuredCoalescentTipData is constant); rows of internal nodes are ignored. */
int mascotSetTipPartials(int instance, const double* tipPartials /* lineageCount x K */);

/*
 * One evaluation. Forward only when outputs == MASCOT_OUT_LOGL; forward + tape + adjoint otherwise.
 * Schedule arrays are StructuredCoalescentSchedule's public final arrays, passed as they are.
 * Rates come from Java's validated epoch rates (no exp() in native code).
 * Checkpoint sets (R4): Java flips the indices in store/restore (no native call there); -1 disables restart.
 * Until R4 is implemented, any index >= 0 returns MASCOT_ERROR_UNSUPPORTED.
 */
int mascotEvaluate(int instance,
                   /* schedule */
                   int intervalCount, double initialTime, int initialSampleNode,
                   const double* intervalLengths, const int* eventTypes, const int* sampleNodes,
                   const int* coalescentChild1, const int* coalescentChild2, const int* coalescentParent,
                   /* model */
                   int epochCount, const double* boundaries,          /* epochCount + 1, last = +inf          */
                   const double* migrationMatrices,                   /* epochCount x K x K, row = source,
                                                                         diag = -row sum                      */
                   const double* inversePopulations,                  /* epochCount x K, q = exp(-log Ne)      */
                   const double* branchRates,                         /* lineageCount, or NULL                 */
                   const MascotStepRule* stepRule,
                   int checkProbabilities,
                   /* incremental restart (R4) */
                   int checkpointReadSet, int checkpointWriteSet,
                   /* outputs */
                   int outputs,
                   double* logLikelihood,
                   double* thetaGradient,
                   double* clockGradient,
                   double* ancestralStateScores);

/* Tests only: per-interval step counts of the last evaluation (Java/native step-rule parity, R3). */
int mascotGetLastStepCounts(int instance, int* steps, int capacity, int* count);

#ifdef __cplusplus
}
#endif
#endif /* MASCOT_H */
