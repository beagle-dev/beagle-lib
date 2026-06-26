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
 * @author Marc Suchard
 *
 * Spectral partial-likelihood GPU kernels — OpenCL-compatible implementation
 * using C-preprocessor #ifdef directives to emulate C++ templates.
 *
 * Correspondence with KernelsSpectral.cu (CUDA / C++17):
 *
 *   C++ construct                        │ Preprocessor equivalent (this file)
 *   ─────────────────────────────────────┼────────────────────────────────────
 *   template <typename Child1>           │ #define SPECTRAL_CHILD1_STATES
 *       where Child1 = States            │     (omit → Partials)
 *   template <typename Child2>           │ #define SPECTRAL_CHILD2_STATES
 *       where Child2 = States            │     (omit → Partials)
 *   template <bool useScaling = true>    │ #define SPECTRAL_USE_SCALING
 *                                        │     (omit → no scaling)
 *   if constexpr (is_same<C,States>)     │ #ifdef SPECTRAL_CHILD1_STATES
 *
 * Two usage models:
 *
 *   MODEL A — generic device function (mirrors the template):
 *     kernelSpectralBody() is a KW_DEVICE_FUNC whose body selects code paths
 *     at compile time via #ifdef.  Compile once per define-combination to
 *     obtain a single specialisation.  Six compilations yield six binaries.
 *
 *   MODEL B — single-compilation named kernels:
 *     The six KW_GLOBAL_KERNEL functions at the bottom of this file invoke
 *     the phase macros directly, so all six variants coexist in one OpenCL
 *     program object — the model required by BEAGLE's OpenCL backend.
 *
 * Phase-macro building blocks (used in both models):
 *
 *   SPECTRAL_INDICES_GPU()
 *   SPECTRAL_COMMON_SMEM_GPU()
 *   SPECTRAL_LOAD_PARTIALS1_GPU() / SPECTRAL_LOAD_PARTIALS2_GPU()
 *   SPECTRAL_LOAD_SCALE_GPU()
 *   SPECTRAL_EIGENVALS_GPU()           — computes sDs/sCs; ends with fence
 *   SPECTRAL_PHASE1_PARTIALS_GPU(BUF,SP,SQ,IEVC)  — block-peel ievc×p → q
 *   SPECTRAL_PHASE1_STATES_GPU(SQ,IEVC,STATES_ARR) — direct column lookup
 *   SPECTRAL_PHASE2_GPU()              — eigenvalue scaling; fenced both ends
 *   SPECTRAL_PHASE3_GPU()              — block-peel evec×tmp → sum
 *   SPECTRAL_WRITE_NO_SCALE_GPU()
 *   SPECTRAL_WRITE_FIXED_SCALE_GPU()
 */

#ifdef CUDA
    #include "libhmsbeagle/GPU/GPUImplDefs.h"
    extern "C" {
#elif defined(FW_OPENCL)
    #ifdef DOUBLE_PRECISION
        #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #endif
    #define __umul24(x, y) ((x) * (y))
#endif

/* ── FMA helper ─────────────────────────────────────────────────────────── */
#if (!defined DOUBLE_PRECISION && defined FP_FAST_FMAF) || \
    ( defined DOUBLE_PRECISION && defined FP_FAST_FMA)
    #define SPECTRAL_FMA(x, y, z)  (z = fma(x, y, z))
#else
    #define SPECTRAL_FMA(x, y, z)  (z += (x) * (y))
#endif

/* ── sincos helper: fused sin+cos in one transcendental instruction ──────── */
#if defined(CUDA)
    #ifdef DOUBLE_PRECISION
        #define SPECTRAL_SINCOS(angle, sv, cv)  sincos((angle), &(sv), &(cv))
    #else
        #define SPECTRAL_SINCOS(angle, sv, cv)  sincosf((angle), &(sv), &(cv))
    #endif
#else  /* OpenCL: sincos(x, *cosval) returns sinval */
    #define SPECTRAL_SINCOS(angle, sv, cv)  ((sv) = sincos((angle), &(cv)))
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 * Phase-macro definitions
 *
 * Each macro expands inline inside a kernel (or kernelSpectralBody).
 * Variable names declared inside each macro:
 *   SPECTRAL_INDICES_GPU      → state, patIdx, pattern, matrix,
 *                               deltaPartialsByState, deltaPartialsByMatrix,
 *                               u, y
 *   SPECTRAL_COMMON_SMEM_GPU  → sBuf1, sBuf2, sDs1, sCs1, sDs2, sCs2,
 *                               sQ1, sQ2
 *   SPECTRAL_LOAD_PARTIALS1   → sP1
 *   SPECTRAL_LOAD_PARTIALS2   → sP2
 *   SPECTRAL_LOAD_SCALE       → sScale
 *   SPECTRAL_PHASE3_GPU       → sum1, sum2
 * ═══════════════════════════════════════════════════════════════════════════*/

/* ── Thread / pattern / category indices ────────────────────────────────── */
#define SPECTRAL_INDICES_GPU() \
    int state   = KW_LOCAL_ID_0; \
    int patIdx  = KW_LOCAL_ID_1; \
    int pattern = __umul24(KW_GROUP_ID_0, PATTERN_BLOCK_SIZE) + patIdx; \
    int matrix  = KW_GROUP_ID_1; \
    int deltaPartialsByState  = pattern * PADDED_STATE_COUNT; \
    int deltaPartialsByMatrix = matrix  * PADDED_STATE_COUNT * totalPatterns; \
    int u = state + deltaPartialsByState + deltaPartialsByMatrix; \
    int y = deltaPartialsByState + deltaPartialsByMatrix;

/* ── Shared memory always present in all variants ───────────────────────── */
/* sBuf1/2: reused for ievc block-peel (Phase 1, Partials) and evec (Phase 3).
 * sDs/sCs: expat*cos(imag*t) and expat*sin(imag*t) per eigenstate.
 *   sCs == 0 → real eigenvalue
 *   sCs >  0 → first  of complex pair (imagEV > 0, neighbor = state+1)
 *   sCs <  0 → second of complex pair (imagEV < 0, neighbor = state-1)
 * sQ1/2: intermediate eigenspace vectors (phase 1 output, phase 2 input).
 * sBuf peeling stride is PADDED_STATE_COUNT (eigenstate space), NOT BLOCK_PEELING_SIZE
 * (which is tuned for the non-spectral thread layout and may exceed PADDED_STATE_COUNT). */
#define SPECTRAL_COMMON_SMEM_GPU() \
    KW_LOCAL_MEM REAL sBuf1[PADDED_STATE_COUNT][PADDED_STATE_COUNT]; \
    KW_LOCAL_MEM REAL sBuf2[PADDED_STATE_COUNT][PADDED_STATE_COUNT]; \
    KW_LOCAL_MEM REAL sDs1[PADDED_STATE_COUNT]; \
    KW_LOCAL_MEM REAL sCs1[PADDED_STATE_COUNT]; \
    KW_LOCAL_MEM REAL sDs2[PADDED_STATE_COUNT]; \
    KW_LOCAL_MEM REAL sCs2[PADDED_STATE_COUNT]; \
    KW_LOCAL_MEM REAL sQ1[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT]; \
    KW_LOCAL_MEM REAL sQ2[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];

/* ── Input partial loaders (Partials children only) ─────────────────────── */
#define SPECTRAL_LOAD_PARTIALS1_GPU() \
    KW_LOCAL_MEM REAL sP1[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT]; \
    if (pattern < totalPatterns) \
        sP1[patIdx][state] = partials1[y + state]; \
    else \
        sP1[patIdx][state] = (REAL)0;

#define SPECTRAL_LOAD_PARTIALS2_GPU() \
    KW_LOCAL_MEM REAL sP2[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT]; \
    if (pattern < totalPatterns) \
        sP2[patIdx][state] = partials2[y + state]; \
    else \
        sP2[patIdx][state] = (REAL)0;

/* ── Pre-computed scaling denominators (scaling variants only) ───────────── */
#define SPECTRAL_LOAD_SCALE_GPU() \
    KW_LOCAL_MEM REAL sScale[PATTERN_BLOCK_SIZE]; \
    if (patIdx == 0 && state < PATTERN_BLOCK_SIZE) \
        sScale[state] = scalingFactors[KW_GROUP_ID_0 * PATTERN_BLOCK_SIZE + state];

/* ── Per-category eigenvalue exponentials ───────────────────────────────── */
/* patIdx-0 threads: one thread per eigenstate (state = k).
 * eigenValues layout: [realEV_0..realEV_{S-1} | imagEV_0..imagEV_{S-1}]
 * distances[matrix] = branchLength * categoryRate[matrix].
 * Ends with KW_LOCAL_FENCE so sP*, sScale, sDs*, sCs* are all visible. */
#define SPECTRAL_EIGENVALS_GPU() \
    if (patIdx == 0) { \
        REAL t1  = distances1[matrix]; \
        REAL e1  = exp(eigenValues1[state] * t1); \
        REAL bt1 = eigenValues1[PADDED_STATE_COUNT + state] * t1; \
        REAL cv1, sv1; \
        SPECTRAL_SINCOS(bt1, sv1, cv1); \
        sDs1[state] = e1 * cv1; \
        sCs1[state] = e1 * sv1; \
        REAL t2  = distances2[matrix]; \
        REAL e2  = exp(eigenValues2[state] * t2); \
        REAL bt2 = eigenValues2[PADDED_STATE_COUNT + state] * t2; \
        REAL cv2, sv2; \
        SPECTRAL_SINCOS(bt2, sv2, cv2); \
        sDs2[state] = e2 * cv2; \
        sCs2[state] = e2 * sv2; \
    } \
    KW_LOCAL_FENCE;

/* ── Phase 1: Partials child — dot product q = U^{-1}·p ─────────────────── */
/* BUF  : sBuf1 or sBuf2 (scratch for ievc; reused for evec in Phase 3)
 * SP   : sP1 or sP2    (shared partial loaded above)
 * SQ   : sQ1 or sQ2    (output: eigenspace projection)
 * IEVC : ievc1 or ievc2
 *
 * For 4 states the 4×4 ievc matrix is 16 contiguous REALs.  The 16 threads
 * with patIdx<4 and state=0..3 load all 16 elements in one coalesced
 * transaction (flat index patIdx*4+state = 0..15), matching the
 * LOAD_MATRIX_4_GPU pattern used by the non-spectral kernels.
 * The 4-element dot product is fully unrolled; no trailing fence is emitted
 * because SPECTRAL_PHASE2_GPU opens with its own KW_LOCAL_FENCE.
 * Scoped in {} so q_loc does not collide across two consecutive invocations. */
#define SPECTRAL_PHASE1_PARTIALS_GPU(BUF, SP, SQ, IEVC) \
    { \
        if (patIdx < PADDED_STATE_COUNT) \
            BUF[patIdx][state] = (IEVC)[patIdx * PADDED_STATE_COUNT + state]; \
        KW_LOCAL_FENCE; \
        REAL q_loc = (REAL)0; \
        SPECTRAL_FMA(BUF[0][state], SP[patIdx][0], q_loc); \
        SPECTRAL_FMA(BUF[1][state], SP[patIdx][1], q_loc); \
        SPECTRAL_FMA(BUF[2][state], SP[patIdx][2], q_loc); \
        SPECTRAL_FMA(BUF[3][state], SP[patIdx][3], q_loc); \
        SQ[patIdx][state] = q_loc; \
    }

/* ── Phase 1: States child — direct column lookup q[k] = U^{-1}[k, s] ─── */
/* dIevc[s*S+k] = U^{-1}[k,s].  Threads k=0..S-1 with fixed s access
 * consecutive addresses → coalesced read.
 * Ambiguous tip (s >= PADDED_STATE_COUNT): treat child as all-ones partial;
 * q[k] = Σ_j U^{-1}[k,j] = row-k sum of U^{-1} → (P·1)[state] ≈ 1.
 * Scoped in {} to match the convention of SPECTRAL_PHASE1_PARTIALS_GPU. */
#define SPECTRAL_PHASE1_STATES_GPU(SQ, IEVC, STATES_ARR) \
    { \
        REAL q_loc = (REAL)0; \
        if (pattern < totalPatterns) { \
            int s = (STATES_ARR)[pattern]; \
            if (s < PADDED_STATE_COUNT) { \
                q_loc = (IEVC)[s * PADDED_STATE_COUNT + state]; \
            } else { \
                for (int j = 0; j < PADDED_STATE_COUNT; j++) \
                    q_loc += (IEVC)[j * PADDED_STATE_COUNT + state]; \
            } \
        } \
        SQ[patIdx][state] = q_loc; \
    }

/* ── Phase 2: eigenvalue scaling and complex-pair rotation ──────────────── */
/* Unified formula: tmp[k] = sDs[k]*q[k] + sCs[k]*q[neighbor(k)]
 *   real eigenvalue (sCs==0):  tmp[k] = sDs[k]*q[k]
 *   first  of pair  (sCs>0):   tmp[k] = sDs[k]*q[k]  + sCs[k]*q[k+1]
 *   second of pair  (sCs<0):   tmp[k] = sDs[k]*q[k]  + sCs[k]*q[k-1]
 * Opens with KW_LOCAL_FENCE (ensures sQ written by Phase 1 is visible).
 * Closes with KW_LOCAL_FENCE (ensures updated sQ is visible for Phase 3).
 * All reads of sQ precede all writes because each thread writes only its own
 * sQ[patIdx][state] after reading sQ[patIdx][state] and sQ[patIdx][neighbor]. */
#define SPECTRAL_PHASE2_GPU() \
    KW_LOCAL_FENCE; \
    { \
        REAL ec1 = sDs1[state], es1 = sCs1[state]; \
        int  nb1 = state + (es1 > (REAL)0 ? 1 : -1); \
        sQ1[patIdx][state] = (es1 == (REAL)0) \
            ? ec1 * sQ1[patIdx][state] \
            : ec1 * sQ1[patIdx][state] + es1 * sQ1[patIdx][nb1]; \
        REAL ec2 = sDs2[state], es2 = sCs2[state]; \
        int  nb2 = state + (es2 > (REAL)0 ? 1 : -1); \
        sQ2[patIdx][state] = (es2 == (REAL)0) \
            ? ec2 * sQ2[patIdx][state] \
            : ec2 * sQ2[patIdx][state] + es2 * sQ2[patIdx][nb2]; \
    } \
    KW_LOCAL_FENCE;

/* ── Phase 3: project back to state space — evec×tmp ────────────────────── */
/* Same coalesced 16-element load pattern as SPECTRAL_PHASE1_PARTIALS_GPU;
 * sBuf is reused for evec after Phase 2 has finished with sQ.
 * result[state] = Σ_k evec[k*S+state]*tmp[k] = (U·tmp)[state].
 * Both children are handled in one pass with fully unrolled 4-element FMAs.
 * No trailing fence: the global-memory write that follows needs none.
 * Declares sum1, sum2 at function scope so SPECTRAL_WRITE_*_GPU can use them. */
#define SPECTRAL_PHASE3_GPU() \
    REAL sum1 = (REAL)0, sum2 = (REAL)0; \
    if (patIdx < PADDED_STATE_COUNT) { \
        sBuf1[patIdx][state] = evec1[patIdx * PADDED_STATE_COUNT + state]; \
        sBuf2[patIdx][state] = evec2[patIdx * PADDED_STATE_COUNT + state]; \
    } \
    KW_LOCAL_FENCE; \
    SPECTRAL_FMA(sBuf1[0][state], sQ1[patIdx][0], sum1); \
    SPECTRAL_FMA(sBuf2[0][state], sQ2[patIdx][0], sum2); \
    SPECTRAL_FMA(sBuf1[1][state], sQ1[patIdx][1], sum1); \
    SPECTRAL_FMA(sBuf2[1][state], sQ2[patIdx][1], sum2); \
    SPECTRAL_FMA(sBuf1[2][state], sQ1[patIdx][2], sum1); \
    SPECTRAL_FMA(sBuf2[2][state], sQ2[patIdx][2], sum2); \
    SPECTRAL_FMA(sBuf1[3][state], sQ1[patIdx][3], sum1); \
    SPECTRAL_FMA(sBuf2[3][state], sQ2[patIdx][3], sum2);

/* ── Output writers ─────────────────────────────────────────────────────── */
#define SPECTRAL_WRITE_NO_SCALE_GPU() \
    if (pattern < totalPatterns) \
        partials3[u] = sum1 * sum2;

/* sScale loaded before SPECTRAL_EIGENVALS_GPU's fence → still valid here. */
#define SPECTRAL_WRITE_FIXED_SCALE_GPU() \
    if (pattern < totalPatterns) \
        partials3[u] = sum1 * sum2 * ((REAL)1 / sScale[patIdx]);

/* Auto-scaling: detect overflow/underflow per pattern, rescale if needed, and
 * write the per-pattern exponent to scalingFactors[matrix*totalPatterns+pattern]
 * as a signed char.  Reuses sQ1[patIdx][*] (free after Phase 3) as scratch for
 * the per-pattern max-exponent reduction.  Thread 0 of each pattern row does a
 * linear scan so correctness does not depend on PADDED_STATE_COUNT being a
 * power of two. */
#define SPECTRAL_WRITE_AUTO_SCALE_GPU() \
    { \
        REAL tmpPartial = sum1 * sum2; \
        int  expTmp; \
        REAL sigTmp = frexp(tmpPartial, &expTmp); \
        sQ1[patIdx][state] = (REAL)( \
            (pattern < totalPatterns && abs(expTmp) > SCALING_EXPONENT_THRESHOLD) \
            ? expTmp : 0); \
        KW_LOCAL_FENCE; \
        if (state == 0) { \
            REAL maxVal = sQ1[patIdx][0]; \
            for (int _i = 1; _i < PADDED_STATE_COUNT; _i++) \
                if (sQ1[patIdx][_i] > maxVal) maxVal = sQ1[patIdx][_i]; \
            sQ1[patIdx][0] = maxVal; \
        } \
        KW_LOCAL_FENCE; \
        int maxExp = (int)sQ1[patIdx][0]; \
        if (pattern < totalPatterns) \
            partials3[u] = (maxExp != 0) \
                ? ldexp(sigTmp, expTmp - maxExp) \
                : tmpPartial; \
        if (state == 0 && pattern < totalPatterns) \
            scalingFactors[matrix * totalPatterns + pattern] = (signed char)maxExp; \
    }

/* ═══════════════════════════════════════════════════════════════════════════
 * MODEL A — Generic device function using #ifdef to emulate templates.
 *
 * CUDA only: OpenCL forbids __local declarations inside non-kernel functions
 * (KW_DEVICE_FUNC expands to nothing in OpenCL, making this a plain C
 * function).  The named KW_GLOBAL_KERNEL functions in Model B cover all six
 * variants for OpenCL.
 *
 * Set defines before compilation to select a specialisation:
 *
 *   Combination                         defines to set
 *   ─────────────────────────────────────────────────────────────────
 *   PartialsPartials / no  scaling      (none)
 *   PartialsPartials / fixed scaling    SPECTRAL_USE_SCALING
 *   StatesPartials   / no  scaling      SPECTRAL_CHILD1_STATES
 *   StatesPartials   / fixed scaling    SPECTRAL_CHILD1_STATES  SPECTRAL_USE_SCALING
 *   StatesStates     / no  scaling      SPECTRAL_CHILD1_STATES  SPECTRAL_CHILD2_STATES
 *   StatesStates     / fixed scaling    SPECTRAL_CHILD1_STATES  SPECTRAL_CHILD2_STATES  SPECTRAL_USE_SCALING
 *
 * Both partials* and states* pointers are always present in the signature;
 * the unused pointer is passed as NULL by the caller and is never dereferenced
 * thanks to the #ifdef guards eliminating its use at compile time.
 * ═══════════════════════════════════════════════════════════════════════════*/
#ifdef CUDA
KW_DEVICE_FUNC void kernelSpectralBody(
        KW_GLOBAL_VAR REAL* KW_RESTRICT partials1,    /* NULL if CHILD1_STATES */
        KW_GLOBAL_VAR int*  KW_RESTRICT states1,      /* NULL if !CHILD1_STATES */
        KW_GLOBAL_VAR REAL* KW_RESTRICT partials2,    /* NULL if CHILD2_STATES */
        KW_GLOBAL_VAR int*  KW_RESTRICT states2,      /* NULL if !CHILD2_STATES */
        KW_GLOBAL_VAR REAL* KW_RESTRICT partials3,
        KW_GLOBAL_VAR REAL* KW_RESTRICT ievc1,
        KW_GLOBAL_VAR REAL* KW_RESTRICT evec1,
        KW_GLOBAL_VAR REAL* KW_RESTRICT eigenValues1,
        KW_GLOBAL_VAR REAL* KW_RESTRICT distances1,
        KW_GLOBAL_VAR REAL* KW_RESTRICT ievc2,
        KW_GLOBAL_VAR REAL* KW_RESTRICT evec2,
        KW_GLOBAL_VAR REAL* KW_RESTRICT eigenValues2,
        KW_GLOBAL_VAR REAL* KW_RESTRICT distances2,
        KW_GLOBAL_VAR REAL* KW_RESTRICT scalingFactors, /* NULL if !USE_SCALING */
        int totalPatterns) {

    SPECTRAL_INDICES_GPU()
    SPECTRAL_COMMON_SMEM_GPU()

    /* Load input partials — omitted by preprocessor for States children,
     * eliminating both the shared memory and the global-memory read. */
#ifndef SPECTRAL_CHILD1_STATES
    SPECTRAL_LOAD_PARTIALS1_GPU()
#endif
#ifndef SPECTRAL_CHILD2_STATES
    SPECTRAL_LOAD_PARTIALS2_GPU()
#endif
#ifdef SPECTRAL_USE_SCALING
    SPECTRAL_LOAD_SCALE_GPU()
#endif

    SPECTRAL_EIGENVALS_GPU()    /* ends with KW_LOCAL_FENCE */

    /* Phase 1: project to eigenspace.
     * #ifdef replaces if constexpr (is_same<Child1, States>) from the C++ version. */
#ifdef SPECTRAL_CHILD1_STATES
    SPECTRAL_PHASE1_STATES_GPU(sQ1, ievc1, states1)
#else
    SPECTRAL_PHASE1_PARTIALS_GPU(sBuf1, sP1, sQ1, ievc1)
#endif

#ifdef SPECTRAL_CHILD2_STATES
    SPECTRAL_PHASE1_STATES_GPU(sQ2, ievc2, states2)
#else
    SPECTRAL_PHASE1_PARTIALS_GPU(sBuf2, sP2, sQ2, ievc2)
#endif

    SPECTRAL_PHASE2_GPU()       /* fenced both ends */
    SPECTRAL_PHASE3_GPU()       /* declares sum1, sum2 */

#ifdef SPECTRAL_USE_SCALING
    SPECTRAL_WRITE_FIXED_SCALE_GPU()
#else
    SPECTRAL_WRITE_NO_SCALE_GPU()
#endif
}
#endif /* CUDA */

/* ═══════════════════════════════════════════════════════════════════════════
 * MODEL B — Named KW_GLOBAL_KERNEL functions, single-compilation model.
 *
 * Each kernel invokes the phase macros directly so that all six variants
 * coexist in one translation unit / OpenCL program object.  Each kernel has
 * a type-specific parameter list (no superfluous null pointers in the API).
 *
 * This mirrors the six extern "C" wrappers in KernelsSpectral.cu, but uses
 * macros in place of the C++ template device function.
 * ═══════════════════════════════════════════════════════════════════════════*/

/* ── PartialsPartials ──────────────────────────────────────────────────── */

KW_GLOBAL_KERNEL void kernelPartialsPartialsNoScaleSpectral(
        KW_GLOBAL_VAR REAL* KW_RESTRICT partials1,
        KW_GLOBAL_VAR REAL* KW_RESTRICT partials2,
        KW_GLOBAL_VAR REAL* KW_RESTRICT partials3,
        KW_GLOBAL_VAR REAL* KW_RESTRICT ievc1,
        KW_GLOBAL_VAR REAL* KW_RESTRICT evec1,
        KW_GLOBAL_VAR REAL* KW_RESTRICT eigenValues1,
        KW_GLOBAL_VAR REAL* KW_RESTRICT distances1,
        KW_GLOBAL_VAR REAL* KW_RESTRICT ievc2,
        KW_GLOBAL_VAR REAL* KW_RESTRICT evec2,
        KW_GLOBAL_VAR REAL* KW_RESTRICT eigenValues2,
        KW_GLOBAL_VAR REAL* KW_RESTRICT distances2,
        int totalPatterns) {
    SPECTRAL_INDICES_GPU()
    SPECTRAL_COMMON_SMEM_GPU()
    SPECTRAL_LOAD_PARTIALS1_GPU()
    SPECTRAL_LOAD_PARTIALS2_GPU()
    SPECTRAL_EIGENVALS_GPU()
    SPECTRAL_PHASE1_PARTIALS_GPU(sBuf1, sP1, sQ1, ievc1)
    SPECTRAL_PHASE1_PARTIALS_GPU(sBuf2, sP2, sQ2, ievc2)
    SPECTRAL_PHASE2_GPU()
    SPECTRAL_PHASE3_GPU()
    SPECTRAL_WRITE_NO_SCALE_GPU()
}

KW_GLOBAL_KERNEL void kernelPartialsPartialsFixedScaleSpectral(
        KW_GLOBAL_VAR REAL* KW_RESTRICT partials1,
        KW_GLOBAL_VAR REAL* KW_RESTRICT partials2,
        KW_GLOBAL_VAR REAL* KW_RESTRICT partials3,
        KW_GLOBAL_VAR REAL* KW_RESTRICT ievc1,
        KW_GLOBAL_VAR REAL* KW_RESTRICT evec1,
        KW_GLOBAL_VAR REAL* KW_RESTRICT eigenValues1,
        KW_GLOBAL_VAR REAL* KW_RESTRICT distances1,
        KW_GLOBAL_VAR REAL* KW_RESTRICT ievc2,
        KW_GLOBAL_VAR REAL* KW_RESTRICT evec2,
        KW_GLOBAL_VAR REAL* KW_RESTRICT eigenValues2,
        KW_GLOBAL_VAR REAL* KW_RESTRICT distances2,
        KW_GLOBAL_VAR REAL* KW_RESTRICT scalingFactors,
        int totalPatterns) {
    SPECTRAL_INDICES_GPU()
    SPECTRAL_COMMON_SMEM_GPU()
    SPECTRAL_LOAD_PARTIALS1_GPU()
    SPECTRAL_LOAD_PARTIALS2_GPU()
    SPECTRAL_LOAD_SCALE_GPU()
    SPECTRAL_EIGENVALS_GPU()
    SPECTRAL_PHASE1_PARTIALS_GPU(sBuf1, sP1, sQ1, ievc1)
    SPECTRAL_PHASE1_PARTIALS_GPU(sBuf2, sP2, sQ2, ievc2)
    SPECTRAL_PHASE2_GPU()
    SPECTRAL_PHASE3_GPU()
    SPECTRAL_WRITE_FIXED_SCALE_GPU()
}

/* ── StatesPartials ────────────────────────────────────────────────────── */
/* Convention: the States (tip) child is always child 1; caller swaps when
 * the partials child is child 1 in the tree traversal. */

KW_GLOBAL_KERNEL void kernelStatesPartialsNoScaleSpectral(
        KW_GLOBAL_VAR int*  KW_RESTRICT states1,
        KW_GLOBAL_VAR REAL* KW_RESTRICT partials2,
        KW_GLOBAL_VAR REAL* KW_RESTRICT partials3,
        KW_GLOBAL_VAR REAL* KW_RESTRICT ievc1,
        KW_GLOBAL_VAR REAL* KW_RESTRICT evec1,
        KW_GLOBAL_VAR REAL* KW_RESTRICT eigenValues1,
        KW_GLOBAL_VAR REAL* KW_RESTRICT distances1,
        KW_GLOBAL_VAR REAL* KW_RESTRICT ievc2,
        KW_GLOBAL_VAR REAL* KW_RESTRICT evec2,
        KW_GLOBAL_VAR REAL* KW_RESTRICT eigenValues2,
        KW_GLOBAL_VAR REAL* KW_RESTRICT distances2,
        int totalPatterns) {
    SPECTRAL_INDICES_GPU()
    SPECTRAL_COMMON_SMEM_GPU()
    SPECTRAL_LOAD_PARTIALS2_GPU()   /* no sP1: child 1 is States */
    SPECTRAL_EIGENVALS_GPU()
    SPECTRAL_PHASE1_STATES_GPU(sQ1, ievc1, states1)
    SPECTRAL_PHASE1_PARTIALS_GPU(sBuf2, sP2, sQ2, ievc2)
    SPECTRAL_PHASE2_GPU()
    SPECTRAL_PHASE3_GPU()
    SPECTRAL_WRITE_NO_SCALE_GPU()
}

KW_GLOBAL_KERNEL void kernelStatesPartialsFixedScaleSpectral(
        KW_GLOBAL_VAR int*  KW_RESTRICT states1,
        KW_GLOBAL_VAR REAL* KW_RESTRICT partials2,
        KW_GLOBAL_VAR REAL* KW_RESTRICT partials3,
        KW_GLOBAL_VAR REAL* KW_RESTRICT ievc1,
        KW_GLOBAL_VAR REAL* KW_RESTRICT evec1,
        KW_GLOBAL_VAR REAL* KW_RESTRICT eigenValues1,
        KW_GLOBAL_VAR REAL* KW_RESTRICT distances1,
        KW_GLOBAL_VAR REAL* KW_RESTRICT ievc2,
        KW_GLOBAL_VAR REAL* KW_RESTRICT evec2,
        KW_GLOBAL_VAR REAL* KW_RESTRICT eigenValues2,
        KW_GLOBAL_VAR REAL* KW_RESTRICT distances2,
        KW_GLOBAL_VAR REAL* KW_RESTRICT scalingFactors,
        int totalPatterns) {
    SPECTRAL_INDICES_GPU()
    SPECTRAL_COMMON_SMEM_GPU()
    SPECTRAL_LOAD_PARTIALS2_GPU()
    SPECTRAL_LOAD_SCALE_GPU()
    SPECTRAL_EIGENVALS_GPU()
    SPECTRAL_PHASE1_STATES_GPU(sQ1, ievc1, states1)
    SPECTRAL_PHASE1_PARTIALS_GPU(sBuf2, sP2, sQ2, ievc2)
    SPECTRAL_PHASE2_GPU()
    SPECTRAL_PHASE3_GPU()
    SPECTRAL_WRITE_FIXED_SCALE_GPU()
}

/* ── StatesStates ──────────────────────────────────────────────────────── */
/* No sP1 or sP2 needed; sBuf1/sBuf2 are used only in Phase 3 (evec peel). */

KW_GLOBAL_KERNEL void kernelStatesStatesNoScaleSpectral(
        KW_GLOBAL_VAR int*  KW_RESTRICT states1,
        KW_GLOBAL_VAR int*  KW_RESTRICT states2,
        KW_GLOBAL_VAR REAL* KW_RESTRICT partials3,
        KW_GLOBAL_VAR REAL* KW_RESTRICT ievc1,
        KW_GLOBAL_VAR REAL* KW_RESTRICT evec1,
        KW_GLOBAL_VAR REAL* KW_RESTRICT eigenValues1,
        KW_GLOBAL_VAR REAL* KW_RESTRICT distances1,
        KW_GLOBAL_VAR REAL* KW_RESTRICT ievc2,
        KW_GLOBAL_VAR REAL* KW_RESTRICT evec2,
        KW_GLOBAL_VAR REAL* KW_RESTRICT eigenValues2,
        KW_GLOBAL_VAR REAL* KW_RESTRICT distances2,
        int totalPatterns) {
    SPECTRAL_INDICES_GPU()
    SPECTRAL_COMMON_SMEM_GPU()      /* no LOAD_PARTIALS: both children are States */
    SPECTRAL_EIGENVALS_GPU()
    SPECTRAL_PHASE1_STATES_GPU(sQ1, ievc1, states1)
    SPECTRAL_PHASE1_STATES_GPU(sQ2, ievc2, states2)
    SPECTRAL_PHASE2_GPU()
    SPECTRAL_PHASE3_GPU()
    SPECTRAL_WRITE_NO_SCALE_GPU()
}

KW_GLOBAL_KERNEL void kernelStatesStatesFixedScaleSpectral(
        KW_GLOBAL_VAR int*  KW_RESTRICT states1,
        KW_GLOBAL_VAR int*  KW_RESTRICT states2,
        KW_GLOBAL_VAR REAL* KW_RESTRICT partials3,
        KW_GLOBAL_VAR REAL* KW_RESTRICT ievc1,
        KW_GLOBAL_VAR REAL* KW_RESTRICT evec1,
        KW_GLOBAL_VAR REAL* KW_RESTRICT eigenValues1,
        KW_GLOBAL_VAR REAL* KW_RESTRICT distances1,
        KW_GLOBAL_VAR REAL* KW_RESTRICT ievc2,
        KW_GLOBAL_VAR REAL* KW_RESTRICT evec2,
        KW_GLOBAL_VAR REAL* KW_RESTRICT eigenValues2,
        KW_GLOBAL_VAR REAL* KW_RESTRICT distances2,
        KW_GLOBAL_VAR REAL* KW_RESTRICT scalingFactors,
        int totalPatterns) {
    SPECTRAL_INDICES_GPU()
    SPECTRAL_COMMON_SMEM_GPU()
    SPECTRAL_LOAD_SCALE_GPU()
    SPECTRAL_EIGENVALS_GPU()
    SPECTRAL_PHASE1_STATES_GPU(sQ1, ievc1, states1)
    SPECTRAL_PHASE1_STATES_GPU(sQ2, ievc2, states2)
    SPECTRAL_PHASE2_GPU()
    SPECTRAL_PHASE3_GPU()
    SPECTRAL_WRITE_FIXED_SCALE_GPU()
}

/* ── PartialsPartials / auto-scaling ───────────────────────────────────── */

KW_GLOBAL_KERNEL void kernelPartialsPartialsAutoScaleSpectral(
        KW_GLOBAL_VAR REAL* KW_RESTRICT partials1,
        KW_GLOBAL_VAR REAL* KW_RESTRICT partials2,
        KW_GLOBAL_VAR REAL* KW_RESTRICT partials3,
        KW_GLOBAL_VAR REAL* KW_RESTRICT ievc1,
        KW_GLOBAL_VAR REAL* KW_RESTRICT evec1,
        KW_GLOBAL_VAR REAL* KW_RESTRICT eigenValues1,
        KW_GLOBAL_VAR REAL* KW_RESTRICT distances1,
        KW_GLOBAL_VAR REAL* KW_RESTRICT ievc2,
        KW_GLOBAL_VAR REAL* KW_RESTRICT evec2,
        KW_GLOBAL_VAR REAL* KW_RESTRICT eigenValues2,
        KW_GLOBAL_VAR REAL* KW_RESTRICT distances2,
        KW_GLOBAL_VAR signed char* KW_RESTRICT scalingFactors,
        int totalPatterns) {
    SPECTRAL_INDICES_GPU()
    SPECTRAL_COMMON_SMEM_GPU()
    SPECTRAL_LOAD_PARTIALS1_GPU()
    SPECTRAL_LOAD_PARTIALS2_GPU()
    SPECTRAL_EIGENVALS_GPU()
    SPECTRAL_PHASE1_PARTIALS_GPU(sBuf1, sP1, sQ1, ievc1)
    SPECTRAL_PHASE1_PARTIALS_GPU(sBuf2, sP2, sQ2, ievc2)
    SPECTRAL_PHASE2_GPU()
    SPECTRAL_PHASE3_GPU()
    SPECTRAL_WRITE_AUTO_SCALE_GPU()
}

#ifdef CUDA
} /* extern "C" */
#endif
