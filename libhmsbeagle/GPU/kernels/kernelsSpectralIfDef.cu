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
 * sQ1/2: intermediate eigenspace vectors (phase 1 output, phase 2 input). */
#define SPECTRAL_COMMON_SMEM_GPU() \
    KW_LOCAL_MEM REAL sBuf1[BLOCK_PEELING_SIZE][PADDED_STATE_COUNT]; \
    KW_LOCAL_MEM REAL sBuf2[BLOCK_PEELING_SIZE][PADDED_STATE_COUNT]; \
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

/* ── Phase 1: Partials child — block-peeled dot product q = U^{-1}·p ────── */
/* BUF  : sBuf1 or sBuf2 (scratch for ievc blocks; reused for evec in Phase 3)
 * SP   : sP1 or sP2    (shared partial loaded above)
 * SQ   : sQ1 or sQ2    (output: eigenspace projection)
 * IEVC : ievc1 or ievc2
 *
 * dIevc[row*S+col] = (U^{-1})^T[row,col] = U^{-1}[col,row], so
 * BUF[peel_row][k] = ievc[(i+peel_row)*S + k] = U^{-1}[k, i+peel_row].
 * The inner FMA accumulates q[k] = Σ_j U^{-1}[k,j] * p[j].
 * Scoped in {} so the local accumulator q_loc does not collide across two
 * consecutive macro invocations (child 1 then child 2). */
#define SPECTRAL_PHASE1_PARTIALS_GPU(BUF, SP, SQ, IEVC) \
    { \
        REAL q_loc = (REAL)0; \
        for (int i = 0; i < PADDED_STATE_COUNT; i += BLOCK_PEELING_SIZE) { \
            if (patIdx < BLOCK_PEELING_SIZE) \
                BUF[patIdx][state] = (IEVC)[(i + patIdx) * PADDED_STATE_COUNT + state]; \
            KW_LOCAL_FENCE; \
            for (int j = 0; j < BLOCK_PEELING_SIZE; j++) { \
                REAL sp = SP[patIdx][i + j]; \
                SPECTRAL_FMA(BUF[j][state], sp, q_loc); \
            } \
            KW_LOCAL_FENCE; \
        } \
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

/* ── Phase 1: Partials × 2 — fused dual-child block-peel ────────────────── */
/* Loads ievc1 and ievc2 in the same guarded block, halving the barrier count
 * for the PartialsPartials case relative to two sequential peel loops. */
#define SPECTRAL_PHASE1_PARTIALS_DUAL_GPU(BUF1, SP1, SQ1, IEVC1, BUF2, SP2, SQ2, IEVC2) \
    { \
        REAL q_loc1 = (REAL)0, q_loc2 = (REAL)0; \
        for (int i = 0; i < PADDED_STATE_COUNT; i += BLOCK_PEELING_SIZE) { \
            if (patIdx < BLOCK_PEELING_SIZE) { \
                BUF1[patIdx][state] = (IEVC1)[(i + patIdx) * PADDED_STATE_COUNT + state]; \
                BUF2[patIdx][state] = (IEVC2)[(i + patIdx) * PADDED_STATE_COUNT + state]; \
            } \
            KW_LOCAL_FENCE; \
            for (int j = 0; j < BLOCK_PEELING_SIZE; j++) { \
                REAL sp1 = SP1[patIdx][i + j]; \
                REAL sp2 = SP2[patIdx][i + j]; \
                SPECTRAL_FMA(BUF1[j][state], sp1, q_loc1); \
                SPECTRAL_FMA(BUF2[j][state], sp2, q_loc2); \
            } \
            KW_LOCAL_FENCE; \
        } \
        SQ1[patIdx][state] = q_loc1; \
        SQ2[patIdx][state] = q_loc2; \
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

/* ── Phase 3: project back to state space — block-peel evec×tmp ─────────── */
/* Same block-peel pattern as SPECTRAL_PHASE1_PARTIALS_GPU; sBuf is reused.
 * result[state] = Σ_k evec[k*S+state]*tmp[k] = (U·tmp)[state].
 * Declares sum1, sum2 at function scope so SPECTRAL_WRITE_*_GPU can use them. */
#define SPECTRAL_PHASE3_GPU() \
    REAL sum1 = (REAL)0, sum2 = (REAL)0; \
    for (int i = 0; i < PADDED_STATE_COUNT; i += BLOCK_PEELING_SIZE) { \
        if (patIdx < BLOCK_PEELING_SIZE) { \
            sBuf1[patIdx][state] = evec1[(i + patIdx) * PADDED_STATE_COUNT + state]; \
            sBuf2[patIdx][state] = evec2[(i + patIdx) * PADDED_STATE_COUNT + state]; \
        } \
        KW_LOCAL_FENCE; \
        for (int j = 0; j < BLOCK_PEELING_SIZE; j++) { \
            REAL q1 = sQ1[patIdx][i + j]; \
            REAL q2 = sQ2[patIdx][i + j]; \
            SPECTRAL_FMA(sBuf1[j][state], q1, sum1); \
            SPECTRAL_FMA(sBuf2[j][state], q2, sum2); \
        } \
        KW_LOCAL_FENCE; \
    }

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
 * the per-pattern max-exponent reduction.
 * XOR-based tree reduction: at each stride _s, thread k merges with its XOR
 * partner k^_s when the partner has a higher index and is in-bounds.  After
 * ceil(log2(N)) steps, sQ1[patIdx][0] holds the global max.  Correct for all
 * PADDED_STATE_COUNT values (power-of-2 and non-power-of-2 alike). */
#define SPECTRAL_WRITE_AUTO_SCALE_GPU() \
    { \
        REAL tmpPartial = sum1 * sum2; \
        int  expTmp; \
        REAL sigTmp = frexp(tmpPartial, &expTmp); \
        sQ1[patIdx][state] = (REAL)( \
            (pattern < totalPatterns && abs(expTmp) > SCALING_EXPONENT_THRESHOLD) \
            ? expTmp : 0); \
        KW_LOCAL_FENCE; \
        for (int _s = 1; _s < PADDED_STATE_COUNT; _s <<= 1) { \
            int _p = state ^ _s; \
            if (_p > state && _p < PADDED_STATE_COUNT) \
                if (sQ1[patIdx][_p] > sQ1[patIdx][state]) \
                    sQ1[patIdx][state] = sQ1[patIdx][_p]; \
            KW_LOCAL_FENCE; \
        } \
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
     * PP case fuses both children into one peel loop (half the barriers).
     * SP/SS cases fall through to single-child macros. */
#if !defined(SPECTRAL_CHILD1_STATES) && !defined(SPECTRAL_CHILD2_STATES)
    SPECTRAL_PHASE1_PARTIALS_DUAL_GPU(sBuf1, sP1, sQ1, ievc1, sBuf2, sP2, sQ2, ievc2)
#else
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
    SPECTRAL_PHASE1_PARTIALS_DUAL_GPU(sBuf1, sP1, sQ1, ievc1, sBuf2, sP2, sQ2, ievc2)
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
    SPECTRAL_PHASE1_PARTIALS_DUAL_GPU(sBuf1, sP1, sQ1, ievc1, sBuf2, sP2, sQ2, ievc2)
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

/* ── Growing (pre-order) macros — generic block-peel variants ──────────── */

/* Scale only sQ2 (sibling) by its eigenvalue exponentials. */
#define SPECTRAL_PHASE2_SIB_GPU() \
    KW_LOCAL_FENCE; \
    { \
        REAL ec2 = sDs2[state], es2 = sCs2[state]; \
        int  nb2 = state + (es2 > (REAL)0 ? 1 : -1); \
        sQ2[patIdx][state] = (es2 == (REAL)0) \
            ? ec2 * sQ2[patIdx][state] \
            : ec2 * sQ2[patIdx][state] + es2 * sQ2[patIdx][nb2]; \
    } \
    KW_LOCAL_FENCE;

/* Sibling Phase 3 + Hadamard: block-peel evec2·sQ2, then ⊙ sP1 → combined in sQ2.
 * Ends with KW_LOCAL_FENCE so the combined vector is visible for parent Phase 1. */
#define SPECTRAL_PHASE3_SIB_HADAMARD_GPU() \
    { \
        REAL tmp = (REAL)0; \
        for (int i = 0; i < PADDED_STATE_COUNT; i += BLOCK_PEELING_SIZE) { \
            if (patIdx < BLOCK_PEELING_SIZE) \
                sBuf2[patIdx][state] = evec2[(i + patIdx) * PADDED_STATE_COUNT + state]; \
            KW_LOCAL_FENCE; \
            for (int j = 0; j < BLOCK_PEELING_SIZE; j++) \
                SPECTRAL_FMA(sBuf2[j][state], sQ2[patIdx][i + j], tmp); \
            KW_LOCAL_FENCE; \
        } \
        sQ2[patIdx][state] = tmp * sP1[patIdx][state]; \
    } \
    KW_LOCAL_FENCE;

/* Scale sQ1 (parent) by transposed eigenvalue exponentials D^T.
 * For the backward (pre-order) pass, P^T = (U^{-1})^T · D^T · U^T.
 * D^T for a complex conjugate pair negates the sin coupling term relative
 * to the forward pass, so we subtract es1 instead of adding it. */
#define SPECTRAL_PHASE2_PAR_GPU() \
    KW_LOCAL_FENCE; \
    { \
        REAL ec1 = sDs1[state], es1 = sCs1[state]; \
        int  nb1 = state + (es1 > (REAL)0 ? 1 : -1); \
        sQ1[patIdx][state] = (es1 == (REAL)0) \
            ? ec1 * sQ1[patIdx][state] \
            : ec1 * sQ1[patIdx][state] - es1 * sQ1[patIdx][nb1]; \
    } \
    KW_LOCAL_FENCE;

/* Parent Phase 3: block-peel evec1·sQ1 → result, write to partials3. */
#define SPECTRAL_PHASE3_WRITE_PAR_GPU() \
    { \
        REAL sum = (REAL)0; \
        for (int i = 0; i < PADDED_STATE_COUNT; i += BLOCK_PEELING_SIZE) { \
            if (patIdx < BLOCK_PEELING_SIZE) \
                sBuf1[patIdx][state] = evec1[(i + patIdx) * PADDED_STATE_COUNT + state]; \
            KW_LOCAL_FENCE; \
            for (int j = 0; j < BLOCK_PEELING_SIZE; j++) { \
                REAL q = sQ1[patIdx][i + j]; \
                SPECTRAL_FMA(sBuf1[j][state], q, sum); \
            } \
            KW_LOCAL_FENCE; \
        } \
        if (pattern < totalPatterns) \
            partials3[u] = sum; \
    }

/* Load only sibling (child2) eigenvalue exponentials — for Top Root kernels. */
#define SPECTRAL_EIGENVALS_SIB_ONLY_GPU() \
    if (patIdx == 0) { \
        REAL t2  = distances2[matrix]; \
        REAL e2  = exp(eigenValues2[state] * t2); \
        REAL bt2 = eigenValues2[PADDED_STATE_COUNT + state] * t2; \
        REAL cv2, sv2; \
        SPECTRAL_SINCOS(bt2, sv2, cv2); \
        sDs2[state] = e2 * cv2; \
        sCs2[state] = e2 * sv2; \
    } \
    KW_LOCAL_FENCE;

/* Sibling Phase 3 + Hadamard + write (block-peel): evec2·sQ2 ⊙ sP1 → partials3. */
#define SPECTRAL_PHASE3_SIB_HADAMARD_WRITE_GPU() \
    { \
        REAL tmp = (REAL)0; \
        for (int i = 0; i < PADDED_STATE_COUNT; i += BLOCK_PEELING_SIZE) { \
            if (patIdx < BLOCK_PEELING_SIZE) \
                sBuf2[patIdx][state] = evec2[(i + patIdx) * PADDED_STATE_COUNT + state]; \
            KW_LOCAL_FENCE; \
            for (int j = 0; j < BLOCK_PEELING_SIZE; j++) \
                SPECTRAL_FMA(sBuf2[j][state], sQ2[patIdx][i + j], tmp); \
            KW_LOCAL_FENCE; \
        } \
        if (pattern < totalPatterns) \
            partials3[u] = tmp * sP1[patIdx][state]; \
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
    SPECTRAL_PHASE1_PARTIALS_DUAL_GPU(sBuf1, sP1, sQ1, ievc1, sBuf2, sP2, sQ2, ievc2)
    SPECTRAL_PHASE2_GPU()
    SPECTRAL_PHASE3_GPU()
    SPECTRAL_WRITE_AUTO_SCALE_GPU()
}

/* ── Growing (pre-order) kernels ─────────────────────────────────────────
 *
 * child1 = parent (backward P^T): ievc1=U^T (dEvecT), evec1=(U^-1)^T (dIevcT)
 * child2 = sibling (forward P):   ievc2=(U^-1)^T (dIevc), evec2=U^T (dEvec)
 * ────────────────────────────────────────────────────────────────────────── */

KW_GLOBAL_KERNEL void kernelPartialsPartialsGrowingSpectral(
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
    SPECTRAL_LOAD_PARTIALS1_GPU()           /* sP1 = parent pre-order */
    SPECTRAL_LOAD_PARTIALS2_GPU()           /* sP2 = sibling post-order */
    SPECTRAL_EIGENVALS_GPU()
    SPECTRAL_PHASE1_PARTIALS_GPU(sBuf2, sP2, sQ2, ievc2)
    SPECTRAL_PHASE2_SIB_GPU()
    SPECTRAL_PHASE3_SIB_HADAMARD_GPU()      /* sQ2 = P_sib·p_sib ⊙ p_par */
    SPECTRAL_PHASE1_PARTIALS_GPU(sBuf1, sQ2, sQ1, ievc1)
    SPECTRAL_PHASE2_PAR_GPU()
    SPECTRAL_PHASE3_WRITE_PAR_GPU()
}

KW_GLOBAL_KERNEL void kernelPartialsStatesGrowingSpectral(
        KW_GLOBAL_VAR REAL* KW_RESTRICT partials1,
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
    SPECTRAL_COMMON_SMEM_GPU()
    SPECTRAL_LOAD_PARTIALS1_GPU()           /* sP1 = parent pre-order */
    SPECTRAL_EIGENVALS_GPU()
    SPECTRAL_PHASE1_STATES_GPU(sQ2, ievc2, states2)
    SPECTRAL_PHASE2_SIB_GPU()
    SPECTRAL_PHASE3_SIB_HADAMARD_GPU()      /* sQ2 = P_sib·e_s ⊙ p_par */
    SPECTRAL_PHASE1_PARTIALS_GPU(sBuf1, sQ2, sQ1, ievc1)
    SPECTRAL_PHASE2_PAR_GPU()
    SPECTRAL_PHASE3_WRITE_PAR_GPU()
}

/* ── Top NotRoot / Root Growing kernels ──────────────────────────────────
 *
 * TOP, NotRoot PP: reuse kernelPartialsPartialsNoScaleSpectral (caller passes
 *                  backward matrices for child1). No separate kernel here.
 *
 * TOP, NotRoot PS: parent=partials(backward), sibling=states(forward).
 * TOP, Root PP:    sibling=partials forward, child1 is root (no transform).
 * TOP, Root PS:    sibling=states forward, child1 is root (no transform).
 * ────────────────────────────────────────────────────────────────────────── */

KW_GLOBAL_KERNEL void kernelPartialsStatesGrowingTopSpectral(
        KW_GLOBAL_VAR REAL* KW_RESTRICT partials1,
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
    SPECTRAL_COMMON_SMEM_GPU()
    SPECTRAL_LOAD_PARTIALS1_GPU()
    SPECTRAL_EIGENVALS_GPU()
    SPECTRAL_PHASE1_PARTIALS_GPU(sBuf1, sP1, sQ1, ievc1)   /* parent backward Ph1 */
    SPECTRAL_PHASE1_STATES_GPU(sQ2, ievc2, states2)          /* sibling forward Ph1 */
    SPECTRAL_PHASE2_GPU()
    SPECTRAL_PHASE3_GPU()
    SPECTRAL_WRITE_NO_SCALE_GPU()
}

KW_GLOBAL_KERNEL void kernelPartialsPartialsGrowingTopRootSpectral(
        KW_GLOBAL_VAR REAL* KW_RESTRICT partials1,
        KW_GLOBAL_VAR REAL* KW_RESTRICT partials2,
        KW_GLOBAL_VAR REAL* KW_RESTRICT partials3,
        KW_GLOBAL_VAR REAL* KW_RESTRICT ievc2,
        KW_GLOBAL_VAR REAL* KW_RESTRICT evec2,
        KW_GLOBAL_VAR REAL* KW_RESTRICT eigenValues2,
        KW_GLOBAL_VAR REAL* KW_RESTRICT distances2,
        int totalPatterns) {
    SPECTRAL_INDICES_GPU()
    SPECTRAL_COMMON_SMEM_GPU()
    SPECTRAL_LOAD_PARTIALS1_GPU()           /* sP1 = root pre-order (Hadamard factor) */
    SPECTRAL_LOAD_PARTIALS2_GPU()
    SPECTRAL_EIGENVALS_SIB_ONLY_GPU()
    SPECTRAL_PHASE1_PARTIALS_GPU(sBuf2, sP2, sQ2, ievc2)
    SPECTRAL_PHASE2_SIB_GPU()
    SPECTRAL_PHASE3_SIB_HADAMARD_WRITE_GPU()
}

KW_GLOBAL_KERNEL void kernelPartialsStatesGrowingTopRootSpectral(
        KW_GLOBAL_VAR REAL* KW_RESTRICT partials1,
        KW_GLOBAL_VAR int*  KW_RESTRICT states2,
        KW_GLOBAL_VAR REAL* KW_RESTRICT partials3,
        KW_GLOBAL_VAR REAL* KW_RESTRICT ievc2,
        KW_GLOBAL_VAR REAL* KW_RESTRICT evec2,
        KW_GLOBAL_VAR REAL* KW_RESTRICT eigenValues2,
        KW_GLOBAL_VAR REAL* KW_RESTRICT distances2,
        int totalPatterns) {
    SPECTRAL_INDICES_GPU()
    SPECTRAL_COMMON_SMEM_GPU()
    SPECTRAL_LOAD_PARTIALS1_GPU()           /* sP1 = root pre-order (Hadamard factor) */
    SPECTRAL_EIGENVALS_SIB_ONLY_GPU()
    SPECTRAL_PHASE1_STATES_GPU(sQ2, ievc2, states2)
    SPECTRAL_PHASE2_SIB_GPU()
    SPECTRAL_PHASE3_SIB_HADAMARD_WRITE_GPU()
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Adjoint cross-product kernels — generic N-state (OpenCL-compatible)
 *
 * Phase 1: one block per (ls, category); block = ADJOINT_BLOCK_SP_N threads.
 *   Partials: all threads cooperate to peel ievc into sIevcBuf[BP][S].
 *             "Padded iteration" ensures all threads reach every barrier.
 *   States:   direct column lookup; no barriers inside the pattern loop.
 *   Reduction: column-by-column tree reduction through sRedBuf[128].
 *
 * Phase 2: one block per category; block = PADDED_STATE_COUNT threads.
 *   Each thread handles one row ls of the gradient.
 *   Reads cross-row shared arrays (sExpat, sCosbt, etc.) written by all threads.
 *   atomicAdd via CAS-loop (OpenCL) / native (CUDA).
 * ═══════════════════════════════════════════════════════════════════════════*/

#if defined(FW_OPENCL) && defined(DOUBLE_PRECISION)
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#endif

#define ADJOINT_BLOCK_SP_N  128

#ifndef ADJOINT_ATOMIC_ADD_GPU  /* may already be defined if both IfDef files merged */
#ifdef CUDA
#define ADJOINT_ATOMIC_ADD_GPU(ptr, val)  atomicAdd((ptr), (REAL)(val))
#elif defined(DOUBLE_PRECISION)
#define ADJOINT_ATOMIC_ADD_GPU(ptr, val) \
    do { \
        __global long* _anp = (__global long*)(ptr); \
        long _ano, _ann; \
        do { _ano = *_anp; _ann = as_long(as_double(_ano) + (double)(val)); } \
        while (atom_cmpxchg(_anp, _ano, _ann) != _ano); \
    } while(0)
#else
#define ADJOINT_ATOMIC_ADD_GPU(ptr, val) \
    do { \
        __global int* _anp = (__global int*)(ptr); \
        int _ano, _ann; \
        do { _ano = *_anp; _ann = as_int(as_float(_ano) + (float)(val)); } \
        while (atomic_cmpxchg(_anp, _ano, _ann) != _ano); \
    } while(0)
#endif
#endif /* ADJOINT_ATOMIC_ADD_GPU */

/* ── Phase 1: Partials child ────────────────────────────────────────────── */
/* Uses "padded iteration": rounds totalPatterns up to next multiple of
 * ADJOINT_BLOCK_SP_N so that all threads reach every barrier inside the
 * ievc block-peel loop.  Threads with k >= totalPatterns contribute lv=0. */

KW_GLOBAL_KERNEL void kernelAdjointPhase1AllRealPartialsN(
        KW_GLOBAL_VAR REAL* KW_RESTRICT prePartials,
        KW_GLOBAL_VAR REAL* KW_RESTRICT postPartials,
        KW_GLOBAL_VAR REAL* KW_RESTRICT evecT,
        KW_GLOBAL_VAR REAL* KW_RESTRICT ievc,
        KW_GLOBAL_VAR REAL* KW_RESTRICT distances,
        KW_GLOBAL_VAR REAL* KW_RESTRICT patternWeights,
        KW_GLOBAL_VAR REAL* KW_RESTRICT categoryWeights,
        KW_GLOBAL_VAR REAL* KW_RESTRICT perSiteLikelihoods,
        KW_GLOBAL_VAR REAL* KW_RESTRICT dOpBuf,
        int totalPatterns) {

    const int tid = KW_LOCAL_ID_0;
    const int ls  = KW_GROUP_ID_0;
    const int cat = KW_GROUP_ID_1;

    KW_LOCAL_MEM REAL sEvecTCol[PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sIevcBuf [BLOCK_PEELING_SIZE][PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sCatW;
    KW_LOCAL_MEM REAL sRedBuf  [ADJOINT_BLOCK_SP_N];

    if (tid < PADDED_STATE_COUNT) sEvecTCol[tid] = evecT[tid * PADDED_STATE_COUNT + ls];
    if (tid == 0)                 sCatW = categoryWeights[cat];
    KW_LOCAL_FENCE;

    REAL regOp[PADDED_STATE_COUNT];
    for (int rs = 0; rs < PADDED_STATE_COUNT; rs++) regOp[rs] = (REAL)0;

    const int catOff  = cat * totalPatterns * PADDED_STATE_COUNT;
    const int maxIter = (totalPatterns + ADJOINT_BLOCK_SP_N - 1) / ADJOINT_BLOCK_SP_N;

    for (int iter = 0; iter < maxIter; iter++) {
        const int k     = iter * ADJOINT_BLOCK_SP_N + tid;
        const int valid = (k < totalPatterns);

        REAL lhsLs = (REAL)0;
        REAL lv    = (REAL)0;
        if (valid) {
            for (int j = 0; j < PADDED_STATE_COUNT; j++)
                SPECTRAL_FMA(sEvecTCol[j], prePartials[catOff + k*PADDED_STATE_COUNT + j], lhsLs);
            lv = lhsLs * (patternWeights[k] * sCatW / exp(perSiteLikelihoods[k]));
        }

        for (int i = 0; i < PADDED_STATE_COUNT; i += BLOCK_PEELING_SIZE) {
            const int elems = BLOCK_PEELING_SIZE * PADDED_STATE_COUNT;
            for (int e = tid; e < elems; e += ADJOINT_BLOCK_SP_N) {
                const int row = e / PADDED_STATE_COUNT;
                const int col = e % PADDED_STATE_COUNT;
                sIevcBuf[row][col] = ievc[(i + row) * PADDED_STATE_COUNT + col];
            }
            KW_LOCAL_FENCE;
            if (valid) {
                for (int j = 0; j < BLOCK_PEELING_SIZE; j++) {
                    const REAL pj = postPartials[catOff + k*PADDED_STATE_COUNT + i + j];
                    for (int rs = 0; rs < PADDED_STATE_COUNT; rs++)
                        regOp[rs] += lv * sIevcBuf[j][rs] * pj;
                }
            }
            KW_LOCAL_FENCE;
        }
    }

    /* Column-by-column tree reduction → write dOpBuf */
    for (int rs = 0; rs < PADDED_STATE_COUNT; rs++) {
        sRedBuf[tid] = regOp[rs];
        KW_LOCAL_FENCE;
        for (int stride = ADJOINT_BLOCK_SP_N >> 1; stride >= 1; stride >>= 1) {
            if (tid < stride) sRedBuf[tid] += sRedBuf[tid + stride];
            KW_LOCAL_FENCE;
        }
        if (tid == 0)
            dOpBuf[cat*PADDED_STATE_COUNT*PADDED_STATE_COUNT + ls*PADDED_STATE_COUNT + rs] = sRedBuf[0];
    }
}

/* ── Phase 1: States child ──────────────────────────────────────────────── */
/* No barriers inside the pattern loop → standard strided iteration is safe. */

KW_GLOBAL_KERNEL void kernelAdjointPhase1AllRealStatesN(
        KW_GLOBAL_VAR REAL* KW_RESTRICT prePartials,
        KW_GLOBAL_VAR int*  KW_RESTRICT tipStates,
        KW_GLOBAL_VAR REAL* KW_RESTRICT evecT,
        KW_GLOBAL_VAR REAL* KW_RESTRICT ievc,
        KW_GLOBAL_VAR REAL* KW_RESTRICT distances,
        KW_GLOBAL_VAR REAL* KW_RESTRICT patternWeights,
        KW_GLOBAL_VAR REAL* KW_RESTRICT categoryWeights,
        KW_GLOBAL_VAR REAL* KW_RESTRICT perSiteLikelihoods,
        KW_GLOBAL_VAR REAL* KW_RESTRICT dOpBuf,
        int totalPatterns) {

    const int tid = KW_LOCAL_ID_0;
    const int ls  = KW_GROUP_ID_0;
    const int cat = KW_GROUP_ID_1;

    KW_LOCAL_MEM REAL sEvecTCol[PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sCatW;
    KW_LOCAL_MEM REAL sRedBuf  [ADJOINT_BLOCK_SP_N];

    if (tid < PADDED_STATE_COUNT) sEvecTCol[tid] = evecT[tid * PADDED_STATE_COUNT + ls];
    if (tid == 0)                 sCatW = categoryWeights[cat];
    KW_LOCAL_FENCE;

    REAL regOp[PADDED_STATE_COUNT];
    for (int rs = 0; rs < PADDED_STATE_COUNT; rs++) regOp[rs] = (REAL)0;

    const int catOff = cat * totalPatterns * PADDED_STATE_COUNT;

    for (int k = tid; k < totalPatterns; k += ADJOINT_BLOCK_SP_N) {
        REAL lhsLs = (REAL)0;
        for (int j = 0; j < PADDED_STATE_COUNT; j++)
            SPECTRAL_FMA(sEvecTCol[j], prePartials[catOff + k*PADDED_STATE_COUNT + j], lhsLs);
        const REAL lv = lhsLs * (patternWeights[k] * sCatW / exp(perSiteLikelihoods[k]));

        const int s = tipStates[k];
        if (s < PADDED_STATE_COUNT) {
            for (int rs = 0; rs < PADDED_STATE_COUNT; rs++)
                regOp[rs] += lv * ievc[s * PADDED_STATE_COUNT + rs];
        } else {
            for (int rs = 0; rs < PADDED_STATE_COUNT; rs++) {
                REAL rhs = (REAL)0;
                for (int j = 0; j < PADDED_STATE_COUNT; j++)
                    rhs += ievc[j * PADDED_STATE_COUNT + rs];
                regOp[rs] += lv * rhs;
            }
        }
    }

    for (int rs = 0; rs < PADDED_STATE_COUNT; rs++) {
        sRedBuf[tid] = regOp[rs];
        KW_LOCAL_FENCE;
        for (int stride = ADJOINT_BLOCK_SP_N >> 1; stride >= 1; stride >>= 1) {
            if (tid < stride) sRedBuf[tid] += sRedBuf[tid + stride];
            KW_LOCAL_FENCE;
        }
        if (tid == 0)
            dOpBuf[cat*PADDED_STATE_COUNT*PADDED_STATE_COUNT + ls*PADDED_STATE_COUNT + rs] = sRedBuf[0];
    }
}

/* AllReal and Complex Phase 1 bodies are identical — the complex flag only
 * matters in Phase 2.  Provide all four required function names. */

KW_GLOBAL_KERNEL void kernelAdjointPhase1ComplexPartialsN(
        KW_GLOBAL_VAR REAL* KW_RESTRICT prePartials,
        KW_GLOBAL_VAR REAL* KW_RESTRICT postPartials,
        KW_GLOBAL_VAR REAL* KW_RESTRICT evecT,
        KW_GLOBAL_VAR REAL* KW_RESTRICT ievc,
        KW_GLOBAL_VAR REAL* KW_RESTRICT distances,
        KW_GLOBAL_VAR REAL* KW_RESTRICT patternWeights,
        KW_GLOBAL_VAR REAL* KW_RESTRICT categoryWeights,
        KW_GLOBAL_VAR REAL* KW_RESTRICT perSiteLikelihoods,
        KW_GLOBAL_VAR REAL* KW_RESTRICT dOpBuf,
        int totalPatterns) {

    const int tid = KW_LOCAL_ID_0;
    const int ls  = KW_GROUP_ID_0;
    const int cat = KW_GROUP_ID_1;

    KW_LOCAL_MEM REAL sEvecTCol[PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sIevcBuf [BLOCK_PEELING_SIZE][PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sCatW;
    KW_LOCAL_MEM REAL sRedBuf  [ADJOINT_BLOCK_SP_N];

    if (tid < PADDED_STATE_COUNT) sEvecTCol[tid] = evecT[tid * PADDED_STATE_COUNT + ls];
    if (tid == 0)                 sCatW = categoryWeights[cat];
    KW_LOCAL_FENCE;

    REAL regOp[PADDED_STATE_COUNT];
    for (int rs = 0; rs < PADDED_STATE_COUNT; rs++) regOp[rs] = (REAL)0;

    const int catOff  = cat * totalPatterns * PADDED_STATE_COUNT;
    const int maxIter = (totalPatterns + ADJOINT_BLOCK_SP_N - 1) / ADJOINT_BLOCK_SP_N;

    for (int iter = 0; iter < maxIter; iter++) {
        const int k     = iter * ADJOINT_BLOCK_SP_N + tid;
        const int valid = (k < totalPatterns);

        REAL lhsLs = (REAL)0;
        REAL lv    = (REAL)0;
        if (valid) {
            for (int j = 0; j < PADDED_STATE_COUNT; j++)
                SPECTRAL_FMA(sEvecTCol[j], prePartials[catOff + k*PADDED_STATE_COUNT + j], lhsLs);
            lv = lhsLs * (patternWeights[k] * sCatW / exp(perSiteLikelihoods[k]));
        }

        for (int i = 0; i < PADDED_STATE_COUNT; i += BLOCK_PEELING_SIZE) {
            const int elems = BLOCK_PEELING_SIZE * PADDED_STATE_COUNT;
            for (int e = tid; e < elems; e += ADJOINT_BLOCK_SP_N) {
                const int row = e / PADDED_STATE_COUNT;
                const int col = e % PADDED_STATE_COUNT;
                sIevcBuf[row][col] = ievc[(i + row) * PADDED_STATE_COUNT + col];
            }
            KW_LOCAL_FENCE;
            if (valid) {
                for (int j = 0; j < BLOCK_PEELING_SIZE; j++) {
                    const REAL pj = postPartials[catOff + k*PADDED_STATE_COUNT + i + j];
                    for (int rs = 0; rs < PADDED_STATE_COUNT; rs++)
                        regOp[rs] += lv * sIevcBuf[j][rs] * pj;
                }
            }
            KW_LOCAL_FENCE;
        }
    }

    for (int rs = 0; rs < PADDED_STATE_COUNT; rs++) {
        sRedBuf[tid] = regOp[rs];
        KW_LOCAL_FENCE;
        for (int stride = ADJOINT_BLOCK_SP_N >> 1; stride >= 1; stride >>= 1) {
            if (tid < stride) sRedBuf[tid] += sRedBuf[tid + stride];
            KW_LOCAL_FENCE;
        }
        if (tid == 0)
            dOpBuf[cat*PADDED_STATE_COUNT*PADDED_STATE_COUNT + ls*PADDED_STATE_COUNT + rs] = sRedBuf[0];
    }
}

KW_GLOBAL_KERNEL void kernelAdjointPhase1ComplexStatesN(
        KW_GLOBAL_VAR REAL* KW_RESTRICT prePartials,
        KW_GLOBAL_VAR int*  KW_RESTRICT tipStates,
        KW_GLOBAL_VAR REAL* KW_RESTRICT evecT,
        KW_GLOBAL_VAR REAL* KW_RESTRICT ievc,
        KW_GLOBAL_VAR REAL* KW_RESTRICT distances,
        KW_GLOBAL_VAR REAL* KW_RESTRICT patternWeights,
        KW_GLOBAL_VAR REAL* KW_RESTRICT categoryWeights,
        KW_GLOBAL_VAR REAL* KW_RESTRICT perSiteLikelihoods,
        KW_GLOBAL_VAR REAL* KW_RESTRICT dOpBuf,
        int totalPatterns) {

    const int tid = KW_LOCAL_ID_0;
    const int ls  = KW_GROUP_ID_0;
    const int cat = KW_GROUP_ID_1;

    KW_LOCAL_MEM REAL sEvecTCol[PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sCatW;
    KW_LOCAL_MEM REAL sRedBuf  [ADJOINT_BLOCK_SP_N];

    if (tid < PADDED_STATE_COUNT) sEvecTCol[tid] = evecT[tid * PADDED_STATE_COUNT + ls];
    if (tid == 0)                 sCatW = categoryWeights[cat];
    KW_LOCAL_FENCE;

    REAL regOp[PADDED_STATE_COUNT];
    for (int rs = 0; rs < PADDED_STATE_COUNT; rs++) regOp[rs] = (REAL)0;

    const int catOff = cat * totalPatterns * PADDED_STATE_COUNT;

    for (int k = tid; k < totalPatterns; k += ADJOINT_BLOCK_SP_N) {
        REAL lhsLs = (REAL)0;
        for (int j = 0; j < PADDED_STATE_COUNT; j++)
            SPECTRAL_FMA(sEvecTCol[j], prePartials[catOff + k*PADDED_STATE_COUNT + j], lhsLs);
        const REAL lv = lhsLs * (patternWeights[k] * sCatW / exp(perSiteLikelihoods[k]));

        const int s = tipStates[k];
        if (s < PADDED_STATE_COUNT) {
            for (int rs = 0; rs < PADDED_STATE_COUNT; rs++)
                regOp[rs] += lv * ievc[s * PADDED_STATE_COUNT + rs];
        } else {
            for (int rs = 0; rs < PADDED_STATE_COUNT; rs++) {
                REAL rhs = (REAL)0;
                for (int j = 0; j < PADDED_STATE_COUNT; j++)
                    rhs += ievc[j * PADDED_STATE_COUNT + rs];
                regOp[rs] += lv * rhs;
            }
        }
    }

    for (int rs = 0; rs < PADDED_STATE_COUNT; rs++) {
        sRedBuf[tid] = regOp[rs];
        KW_LOCAL_FENCE;
        for (int stride = ADJOINT_BLOCK_SP_N >> 1; stride >= 1; stride >>= 1) {
            if (tid < stride) sRedBuf[tid] += sRedBuf[tid + stride];
            KW_LOCAL_FENCE;
        }
        if (tid == 0)
            dOpBuf[cat*PADDED_STATE_COUNT*PADDED_STATE_COUNT + ls*PADDED_STATE_COUNT + rs] = sRedBuf[0];
    }
}

/* ── Phase 2: integral transform + atomicAdd ────────────────────────────── */
/* Block = PADDED_STATE_COUNT threads (one per ls); grid = (categoryCount).
 * Each thread writes its eigenvalue data to shared memory first, then all
 * threads read cross-ls values after the fence. */

KW_GLOBAL_KERNEL void kernelAdjointPhase2AllRealN(
        KW_GLOBAL_VAR REAL* KW_RESTRICT dOpBuf,
        KW_GLOBAL_VAR REAL* KW_RESTRICT eigenValues,
        KW_GLOBAL_VAR REAL* KW_RESTRICT distances,
        KW_GLOBAL_VAR REAL* KW_RESTRICT dGradient) {

    const int ls  = KW_LOCAL_ID_0;
    const int cat = KW_GROUP_ID_0;

    KW_LOCAL_MEM REAL sEvalR[PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sExpat[PADDED_STATE_COUNT];

    sEvalR[ls] = eigenValues[ls];
    KW_LOCAL_FENCE;

    const REAL t = distances[cat];
    sExpat[ls] = exp(sEvalR[ls] * t);
    KW_LOCAL_FENCE;

    const int S     = PADDED_STATE_COUNT;
    const int opOff = cat * S * S + ls * S;
    const REAL ea   = sExpat[ls];
    const REAL la   = sEvalR[ls];

    for (int rs = 0; rs < S; rs++) {
        const REAL coeff = (t * fabs(la - sEvalR[rs]) < (REAL)1e-12)
            ? t * ea : (ea - sExpat[rs]) / (la - sEvalR[rs]);
        ADJOINT_ATOMIC_ADD_GPU(&dGradient[ls*S+rs], dOpBuf[opOff+rs] * coeff);
    }
}

KW_GLOBAL_KERNEL void kernelAdjointPhase2ComplexN(
        KW_GLOBAL_VAR REAL* KW_RESTRICT dOpBuf,
        KW_GLOBAL_VAR REAL* KW_RESTRICT eigenValues,
        KW_GLOBAL_VAR REAL* KW_RESTRICT distances,
        KW_GLOBAL_VAR REAL* KW_RESTRICT dGradient) {

    const int ls  = KW_LOCAL_ID_0;
    const int cat = KW_GROUP_ID_0;

    KW_LOCAL_MEM REAL sEvalR [PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sEvalI [PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sExpat [PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sCosbt [PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sSinbt [PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sExpatC[PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sExpatS[PADDED_STATE_COUNT];

    sEvalR[ls] = eigenValues[ls];
    sEvalI[ls] = eigenValues[PADDED_STATE_COUNT + ls];
    KW_LOCAL_FENCE;

    /* All threads write sExpat/sincos so both fences are always reached. */
    const REAL t = distances[cat];
    const REAL e = exp(sEvalR[ls] * t);
    sExpat[ls] = e;
    REAL sv, cv;
    SPECTRAL_SINCOS(sEvalI[ls] * t, sv, cv);
    sCosbt[ls] = cv; sSinbt[ls] = sv;
    sExpatC[ls] = e * cv; sExpatS[ls] = e * sv;
    KW_LOCAL_FENCE;

    /* Second member of complex pair: its row is handled by ls-1. */
    if (sEvalI[ls] < (REAL)0) return;

    const int S     = PADDED_STATE_COUNT;
    const int opOff = cat * S * S + ls * S;
    const REAL ea   = sExpat[ls];
    const REAL la   = sEvalR[ls];
    const REAL li   = sEvalI[ls];

    if (li == (REAL)0) {
        /* ls is real */
        for (int rs = 0; rs < S; ) {
            const REAL ri = sEvalI[rs];
            if (ri == (REAL)0) {
                /* 1×1 */
                const REAL coeff = (t * fabs(la - sEvalR[rs]) < (REAL)1e-12)
                    ? t * ea : (ea - sExpat[rs]) / (la - sEvalR[rs]);
                ADJOINT_ATOMIC_ADD_GPU(&dGradient[ls*S+rs], dOpBuf[opOff+rs] * coeff);
                rs++;
            } else {
                /* 1×2 */
                const REAL sr  = sEvalR[rs] - la;
                const REAL den = sr*sr + ri*ri;
                REAL ic0, ic1;
                if (den < (REAL)1e-12) { ic0 = t; ic1 = (REAL)0; }
                else {
                    const REAL ex = sExpat[rs] / ea;
                    ic0 = (ex*(sr*sCosbt[rs]+ri*sSinbt[rs])-sr)/den;
                    ic1 = (ex*(sr*sSinbt[rs]-ri*sCosbt[rs])+ri)/den;
                }
                const REAL c0 = ea*ic0, c1 = ea*ic1;
                const REAL in0 = dOpBuf[opOff+rs], in1 = dOpBuf[opOff+rs+1];
                ADJOINT_ATOMIC_ADD_GPU(&dGradient[ls*S+rs],     c0*in0+c1*in1);
                ADJOINT_ATOMIC_ADD_GPU(&dGradient[ls*S+rs+1], -c1*in0+c0*in1);
                rs += 2;
            }
        }
    } else {
        /* ls is first of complex pair — read both rows from dOpBuf */
        const REAL ec   = sExpatC[ls], es = sExpatS[ls];
        const REAL cI   = sCosbt[ls],  sI = sSinbt[ls];
        const int opOff1 = cat * S * S + (ls + 1) * S;
        for (int rs = 0; rs < S; ) {
            const REAL ri = sEvalI[rs];
            if (ri == (REAL)0) {
                /* 2×1 */
                const REAL sr  = sEvalR[rs] - la;
                const REAL den = sr*sr + li*li;
                REAL ic0, ic1;
                if (den < (REAL)1e-12) { ic0 = t; ic1 = (REAL)0; }
                else {
                    const REAL ex = sExpat[rs] / ea;
                    ic0 = (ex*(sr*cI+li*sI)-sr)/den;
                    ic1 = (ex*(sr*sI-li*cI)+li)/den;
                }
                const REAL p0=ec*ic0+es*ic1, p1=ec*ic1-es*ic0;
                const REAL p2=es*ic0-ec*ic1, p3=es*ic1+ec*ic0;
                const REAL in0=dOpBuf[opOff+rs], in1=dOpBuf[opOff1+rs];
                ADJOINT_ATOMIC_ADD_GPU(&dGradient[ls*S+rs],     p0*in0+p1*in1);
                ADJOINT_ATOMIC_ADD_GPU(&dGradient[(ls+1)*S+rs], p2*in0+p3*in1);
                rs++;
            } else {
                /* 2×2 */
                const REAL rr=sEvalR[rs], ri2=ri;
                const REAL sr=rr-la, si1=li+ri2, si2=ri2-li;
                const REAL sr2=sr*sr;
                const REAL d1=sr2+si1*si1, d2=sr2+si2*si2;
                const REAL ex=(d1>=(REAL)1e-12||d2>=(REAL)1e-12) ? sExpat[rs]/ea : (REAL)0;
                const REAL clcr=cI*sCosbt[rs], slsr=sI*sSinbt[rs];
                const REAL clsr=cI*sSinbt[rs], slcr=sI*sCosbt[rs];
                REAL i1r,i1i;
                if (d1<(REAL)1e-12){i1r=t;i1i=(REAL)0;}
                else{
                    const REAL cs1=clcr-slsr, sn1=slcr+clsr;
                    i1r=(sr*(ex*cs1-(REAL)1)+si1*ex*sn1)/d1;
                    i1i=(sr*ex*sn1-si1*(ex*cs1-(REAL)1))/d1;
                }
                REAL i2r,i2i;
                if (d2<(REAL)1e-12){i2r=t;i2i=(REAL)0;}
                else{
                    const REAL cs2=clcr+slsr, sn2=clsr-slcr;
                    i2r=(sr*(ex*cs2-(REAL)1)+si2*ex*sn2)/d2;
                    i2i=(sr*ex*sn2-si2*(ex*cs2-(REAL)1))/d2;
                }
                const REAL pr=ec*i1r+es*i1i, pi_=ec*i1i-es*i1r;
                const REAL mr_=ec*i2r-es*i2i, mi_=ec*i2i+es*i2r;
                const REAL A=(REAL)0.5*(mr_+pr), B=(REAL)0.5*(mi_+pi_);
                const REAL C=(REAL)0.5*(pi_-mi_), D=(REAL)0.5*(mr_-pr);
                const REAL in00=dOpBuf[opOff+rs],  in01=dOpBuf[opOff+rs+1];
                const REAL in10=dOpBuf[opOff1+rs], in11=dOpBuf[opOff1+rs+1];
                ADJOINT_ATOMIC_ADD_GPU(&dGradient[ls*S+rs],          A*in00+B*in01+C*in10+D*in11);
                ADJOINT_ATOMIC_ADD_GPU(&dGradient[ls*S+rs+1],       -B*in00+A*in01-D*in10+C*in11);
                ADJOINT_ATOMIC_ADD_GPU(&dGradient[(ls+1)*S+rs],     -C*in00-D*in01+A*in10+B*in11);
                ADJOINT_ATOMIC_ADD_GPU(&dGradient[(ls+1)*S+rs+1],    D*in00-C*in01-B*in10+A*in11);
                rs += 2;
            }
        }
    }
}

#ifdef CUDA
} /* extern "C" */
#endif
