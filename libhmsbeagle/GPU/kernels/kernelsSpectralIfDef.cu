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
 * ISALLREAL1/ISALLREAL2: per-eigendecomposition runtime flags (from
 * hEigenDecompIsAllReal, threaded through as ordinary kernel arguments — see
 * kernelAdjointMergedN's isAllReal for the precedent this mirrors). When a
 * decomposition is all-real, the imaginary half of eigenValues[] is never
 * read: it may not even be allocated (kEigenValuesSize is only widened for
 * BEAGLE_FLAG_EIGEN_COMPLEX, not merely for spectral-representation use), so
 * reading it unconditionally is an out-of-bounds access, not just wasted
 * work. sCs == 0 (the real-eigenvalue convention already documented on
 * SPECTRAL_COMMON_SMEM_GPU above) is produced explicitly rather than relying
 * on cos(0)=1/sin(0)=0 falling out of an unread angle.
 * Ends with KW_LOCAL_FENCE so sP*, sScale, sDs*, sCs* are all visible. */
#define SPECTRAL_EIGENVALS_GPU(ISALLREAL1, ISALLREAL2) \
    if (patIdx == 0) { \
        REAL t1  = distances1[matrix]; \
        REAL e1  = exp(eigenValues1[state] * t1); \
        sDs1[state] = e1; \
        sCs1[state] = (REAL)0; \
        if (!(ISALLREAL1)) { \
            REAL bt1 = eigenValues1[PADDED_STATE_COUNT + state] * t1; \
            REAL cv1, sv1; \
            SPECTRAL_SINCOS(bt1, sv1, cv1); \
            sDs1[state] = e1 * cv1; \
            sCs1[state] = e1 * sv1; \
        } \
        REAL t2  = distances2[matrix]; \
        REAL e2  = exp(eigenValues2[state] * t2); \
        sDs2[state] = e2; \
        sCs2[state] = (REAL)0; \
        if (!(ISALLREAL2)) { \
            REAL bt2 = eigenValues2[PADDED_STATE_COUNT + state] * t2; \
            REAL cv2, sv2; \
            SPECTRAL_SINCOS(bt2, sv2, cv2); \
            sDs2[state] = e2 * cv2; \
            sCs2[state] = e2 * sv2; \
        } \
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
        int isAllReal1, int isAllReal2,
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

    SPECTRAL_EIGENVALS_GPU(isAllReal1, isAllReal2)    /* ends with KW_LOCAL_FENCE */

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
        int isAllReal1, int isAllReal2,
        int totalPatterns) {
    SPECTRAL_INDICES_GPU()
    SPECTRAL_COMMON_SMEM_GPU()
    SPECTRAL_LOAD_PARTIALS1_GPU()
    SPECTRAL_LOAD_PARTIALS2_GPU()
    SPECTRAL_EIGENVALS_GPU(isAllReal1, isAllReal2)
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
        int isAllReal1, int isAllReal2,
        int totalPatterns) {
    SPECTRAL_INDICES_GPU()
    SPECTRAL_COMMON_SMEM_GPU()
    SPECTRAL_LOAD_PARTIALS1_GPU()
    SPECTRAL_LOAD_PARTIALS2_GPU()
    SPECTRAL_LOAD_SCALE_GPU()
    SPECTRAL_EIGENVALS_GPU(isAllReal1, isAllReal2)
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
        int isAllReal1, int isAllReal2,
        int totalPatterns) {
    SPECTRAL_INDICES_GPU()
    SPECTRAL_COMMON_SMEM_GPU()
    SPECTRAL_LOAD_PARTIALS2_GPU()   /* no sP1: child 1 is States */
    SPECTRAL_EIGENVALS_GPU(isAllReal1, isAllReal2)
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
        int isAllReal1, int isAllReal2,
        int totalPatterns) {
    SPECTRAL_INDICES_GPU()
    SPECTRAL_COMMON_SMEM_GPU()
    SPECTRAL_LOAD_PARTIALS2_GPU()
    SPECTRAL_LOAD_SCALE_GPU()
    SPECTRAL_EIGENVALS_GPU(isAllReal1, isAllReal2)
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
        int isAllReal1, int isAllReal2,
        int totalPatterns) {
    SPECTRAL_INDICES_GPU()
    SPECTRAL_COMMON_SMEM_GPU()      /* no LOAD_PARTIALS: both children are States */
    SPECTRAL_EIGENVALS_GPU(isAllReal1, isAllReal2)
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
        int isAllReal1, int isAllReal2,
        int totalPatterns) {
    SPECTRAL_INDICES_GPU()
    SPECTRAL_COMMON_SMEM_GPU()
    SPECTRAL_LOAD_SCALE_GPU()
    SPECTRAL_EIGENVALS_GPU(isAllReal1, isAllReal2)
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

/* Load only sibling (child2) eigenvalue exponentials — for Top Root kernels.
 * ISALLREAL2: see SPECTRAL_EIGENVALS_GPU's comment above — same gating. */
#define SPECTRAL_EIGENVALS_SIB_ONLY_GPU(ISALLREAL2) \
    if (patIdx == 0) { \
        REAL t2  = distances2[matrix]; \
        REAL e2  = exp(eigenValues2[state] * t2); \
        sDs2[state] = e2; \
        sCs2[state] = (REAL)0; \
        if (!(ISALLREAL2)) { \
            REAL bt2 = eigenValues2[PADDED_STATE_COUNT + state] * t2; \
            REAL cv2, sv2; \
            SPECTRAL_SINCOS(bt2, sv2, cv2); \
            sDs2[state] = e2 * cv2; \
            sCs2[state] = e2 * sv2; \
        } \
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
        int isAllReal1, int isAllReal2,
        int totalPatterns) {
    SPECTRAL_INDICES_GPU()
    SPECTRAL_COMMON_SMEM_GPU()
    SPECTRAL_LOAD_PARTIALS1_GPU()
    SPECTRAL_LOAD_PARTIALS2_GPU()
    SPECTRAL_EIGENVALS_GPU(isAllReal1, isAllReal2)
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
        int isAllReal1, int isAllReal2,
        int totalPatterns) {
    SPECTRAL_INDICES_GPU()
    SPECTRAL_COMMON_SMEM_GPU()
    SPECTRAL_LOAD_PARTIALS1_GPU()           /* sP1 = parent pre-order */
    SPECTRAL_LOAD_PARTIALS2_GPU()           /* sP2 = sibling post-order */
    SPECTRAL_EIGENVALS_GPU(isAllReal1, isAllReal2)
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
        int isAllReal1, int isAllReal2,
        int totalPatterns) {
    SPECTRAL_INDICES_GPU()
    SPECTRAL_COMMON_SMEM_GPU()
    SPECTRAL_LOAD_PARTIALS1_GPU()           /* sP1 = parent pre-order */
    SPECTRAL_EIGENVALS_GPU(isAllReal1, isAllReal2)
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
        int isAllReal1, int isAllReal2,
        int totalPatterns) {
    SPECTRAL_INDICES_GPU()
    SPECTRAL_COMMON_SMEM_GPU()
    SPECTRAL_LOAD_PARTIALS1_GPU()
    SPECTRAL_EIGENVALS_GPU(isAllReal1, isAllReal2)
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
        int isAllReal2,
        int totalPatterns) {
    SPECTRAL_INDICES_GPU()
    SPECTRAL_COMMON_SMEM_GPU()
    SPECTRAL_LOAD_PARTIALS1_GPU()           /* sP1 = root pre-order (Hadamard factor) */
    SPECTRAL_LOAD_PARTIALS2_GPU()
    SPECTRAL_EIGENVALS_SIB_ONLY_GPU(isAllReal2)
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
        int isAllReal2,
        int totalPatterns) {
    SPECTRAL_INDICES_GPU()
    SPECTRAL_COMMON_SMEM_GPU()
    SPECTRAL_LOAD_PARTIALS1_GPU()           /* sP1 = root pre-order (Hadamard factor) */
    SPECTRAL_EIGENVALS_SIB_ONLY_GPU(isAllReal2)
    SPECTRAL_PHASE1_STATES_GPU(sQ2, ievc2, states2)
    SPECTRAL_PHASE2_SIB_GPU()
    SPECTRAL_PHASE3_SIB_HADAMARD_WRITE_GPU()
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Adjoint cross-product kernel — generic N-state (OpenCL-compatible)
 *
 * Single launch, all branches at once (see kernelAdjointMergedN below):
 * grid = (S, categoryCount, branchCount), block = ADJOINT_BLOCK_SP_N threads.
 * One block computes one gradient row-group (a single real row, or a
 * complex-conjugate pair) for one branch/category — pattern reduction and
 * the integral-transform + atomicAdd both happen in that one block, no
 * intermediate global scratch buffer. Per-branch buffers are resolved via
 * a device offset-queue against pooled origins (see ADJOINT_QUEUE_STRIDE
 * and kernelAdjointMergedN below) rather than passed individually.
 *
 * This superseded an earlier two-phase (Phase1/Phase2 + global `dOpBuf`
 * scratch) design, and an intermediate one-launch-per-branch / bucketed
 * (up to 4 launches) design — both were implemented, verified, and
 * benchmarked against this merged version before being removed; see
 * STATUS.md/TODO.md for the full history and benchmark numbers.
 * ═══════════════════════════════════════════════════════════════════════════*/

#if defined(FW_OPENCL) && defined(DOUBLE_PRECISION)
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#endif

#define ADJOINT_BLOCK_SP_N  128

/* CAS-retry float/double atomic add, in its own KW_DEVICE_FUNC-free OpenCL
 * helper function rather than inlined as a macro at the call site.
 *
 * This split is load-bearing, not stylistic: with the CAS loop inlined
 * directly into kernelAdjointMergedN (compiled at PADDED_STATE_COUNT=32,
 * i.e. any state count that pads above 16 states, e.g. the 17-state case),
 * clEnqueueNDRangeKernel succeeds but clFinish() never returns — reproduced
 * down to a single block (no cross-block contention at all) and shown by
 * bisection to require *both* things at once: the atomic_cmpxchg call AND a
 * retry loop around it (a lone, non-looping atomic_cmpxchg — or the same
 * loop shape swapped for a different atomic like atomic_xchg — completes
 * fine; only "loop whose exit condition reads back an atomic's result"
 * hangs, regardless of do/while vs for(;;) syntax). That is consistent with
 * a known class of ROCm/LLVM AMDGPU backend bugs (this is ROCm 4.0.1, circa
 * 2020) around convergent-instruction handling for loops inlined into large,
 * register-heavy kernels; it is not a BEAGLE logic bug — the 16-state
 * (unpadded, S=16) build of the identical macro works correctly. Moving the
 * loop into its own `__attribute__((noinline))` function keeps it out of
 * the miscompiled inlined context and reliably avoids the hang. */
#ifndef ADJOINT_ATOMIC_ADD_GPU  /* may already be defined if both IfDef files merged */
#ifdef CUDA
#define ADJOINT_ATOMIC_ADD_GPU(ptr, val)  atomicAdd((ptr), (REAL)(val))
#elif defined(DOUBLE_PRECISION)
__attribute__((noinline)) void adjointAtomicAddGpuDPHelper(__global long* _anp, double val) {
    long _ano, _ann;
    do { _ano = *_anp; _ann = as_long(as_double(_ano) + val); }
    while (atom_cmpxchg(_anp, _ano, _ann) != _ano);
}
#define ADJOINT_ATOMIC_ADD_GPU(ptr, val) \
    adjointAtomicAddGpuDPHelper((__global long*)(ptr), (double)(val))
#else
__attribute__((noinline)) void adjointAtomicAddGpuSPHelper(__global int* _anp, float val) {
    int _ano, _ann;
    do { _ano = *_anp; _ann = as_int(as_float(_ano) + val); }
    while (atomic_cmpxchg(_anp, _ano, _ann) != _ano);
}
#define ADJOINT_ATOMIC_ADD_GPU(ptr, val) \
    adjointAtomicAddGpuSPHelper((__global int*)(ptr), (float)(val))
#endif
#endif /* ADJOINT_ATOMIC_ADD_GPU */

#define ADJOINT_QUEUE_STRIDE 9

/* ── Pattern-parallel raw accumulation + single post-loop ievc rotation ────
 * First cut of this redesign parallelized threads over *output state* `rs`
 * instead of pattern (see git history / prior STATUS.md revision) — that
 * fixed register pressure and coalescing but, benchmarked, was a real
 * regression at moderate-to-large pattern counts (e.g. S=32, nPat=2000:
 * 0.803 -> 1.114 ms/branch, ~40% slower) because it cut per-block pattern
 * parallelism from 128-way down to (ADJOINT_BLOCK_SP_N/S)-way — 4-way at
 * S=32 — and the `lhs` projection (`EVECT_COL . prePartials[k,:]`, O(S))
 * still has to happen once per pattern regardless, so throughput dropped
 * ~32x for the dominant, patterns-scaling term while only removing an S-way
 * redundancy elsewhere. Reverted to the design below, which keeps full
 * 128-way pattern parallelism (matching the original kernel) and removes
 * *only* the redundant `ievc` projection — the part that was genuinely
 * S-fold redundant across the `S` per-row blocks (see STATUS.md's
 * O(S^3)-vs-O(S^2) derivation), without touching pattern-level parallelism
 * at all. Each thread still privately accumulates a RAW (not yet
 * ievc-rotated) REGOP[PADDED_STATE_COUNT] over its strided patterns —
 * register use is therefore unchanged from before at this step (still a
 * real cost at very large S, e.g. S=256; those sizes are untestable on
 * this machine today due to a separate, pre-existing GPU instance-creation
 * bug — see STATUS.md — so fixing it isn't verifiable here and is left as
 * a follow-on, not bundled into this already-large change). What *is* new:
 * no per-pattern `ievc` tile load/projection (removes the O(S) inner loop
 * and the `sIevcBuf`/`BLOCK_PEELING_SIZE` tiling it needed), and the
 * ievc rotation itself happens exactly once per block afterward
 * (ADJOINTN_ROTATE_ROW) instead of once per (block, pattern) — valid by
 * linearity since `ievc` is constant across every pattern in a segment. */
/* Extracted into its own `__attribute__((noinline))` helper (§5.4 of
 * GPU_ADJOINT_PLAN.md) — this is the single densest concentration of
 * `tid`-predicated, barrier-synchronized code in the kernel (a 32x-repeated
 * 128-wide tree reduction), shared by every code path (isAllReal, singleton,
 * pairleader alike) and never previously isolated by the noinline treatment
 * already proven for the SINGLETON/PAIRLEADER bodies above. Giving it its
 * own function/register-allocation scope is a diagnostic for whether it is
 * the specific miscompiled hot spot behind the PADDED_STATE_COUNT=32
 * get_local_id(0) corruption documented in CLUSTER_AGENT_FINDINGS.md §3. */
__attribute__((noinline)) void adjointAccumRowRaw(
        int tid, __local REAL* evectCol, bool isStates,
        REAL* regOp, __local REAL* rawRow,
        __global REAL* prePartials, __global REAL* postPartials,
        __global int* tipStates, __global REAL* patternWeights, REAL sCatW,
        __global REAL* perSiteLikelihoods, __local REAL* sRedBuf,
        int totalPatterns, int catOff) {
    for (int _rs_a = 0; _rs_a < PADDED_STATE_COUNT; _rs_a++) regOp[_rs_a] = (REAL)0;
    for (int _k_a = tid; _k_a < totalPatterns; _k_a += ADJOINT_BLOCK_SP_N) {
        REAL _lhs_a = (REAL)0;
        for (int _j_a = 0; _j_a < PADDED_STATE_COUNT; _j_a++)
            SPECTRAL_FMA(evectCol[_j_a], prePartials[catOff + _k_a*PADDED_STATE_COUNT + _j_a], _lhs_a);
        const REAL _lv_a = _lhs_a * (patternWeights[_k_a] * sCatW / exp(perSiteLikelihoods[_k_a]));
        if (isStates) {
            const int _st_a = tipStates[_k_a];
            if (_st_a >= PADDED_STATE_COUNT) {
                for (int _rs_a = 0; _rs_a < PADDED_STATE_COUNT; _rs_a++) regOp[_rs_a] += _lv_a;
            } else {
                regOp[_st_a] += _lv_a;
            }
        } else {
            for (int _rs_a = 0; _rs_a < PADDED_STATE_COUNT; _rs_a++)
                regOp[_rs_a] += _lv_a * postPartials[catOff + _k_a*PADDED_STATE_COUNT + _rs_a];
        }
    }
    for (int _rs_a = 0; _rs_a < PADDED_STATE_COUNT; _rs_a++) {
        sRedBuf[tid] = regOp[_rs_a];
        KW_LOCAL_FENCE;
        for (int _str_a = ADJOINT_BLOCK_SP_N >> 1; _str_a >= 1; _str_a >>= 1) {
            if (tid < _str_a) sRedBuf[tid] += sRedBuf[tid + _str_a];
            KW_LOCAL_FENCE;
        }
        if (tid == 0) rawRow[_rs_a] = sRedBuf[0];
        KW_LOCAL_FENCE;
    }
}

#define ADJOINTN_ACCUM_ROW_RAW(EVECT_COL, IS_STATES, REGOP, RAWROW) \
    adjointAccumRowRaw(tid, EVECT_COL, IS_STATES, REGOP, RAWROW, \
        prePartials, postPartials, tipStates, patternWeights, sCatW, \
        perSiteLikelihoods, sRedBuf, totalPatterns, catOff);

/* Rotate a raw (un-projected) row RAWROW[PADDED_STATE_COUNT] through `ievc`
 * once: ROTATED[rs'] = sum_rs RAWROW[rs] * ievc[rs*S + rs']. Valid for every
 * thread with tid (+ m*ADJOINT_BLOCK_SP_N) < PADDED_STATE_COUNT. The `ievc`
 * read for fixed `rs` is coalesced across threads (consecutive tid ==
 * consecutive rs' == consecutive addresses). O(S^2) total, done once per
 * block (not per pattern) — mathematically equivalent to the old
 * per-pattern `ievc` projection by linearity, since `ievc` is constant
 * across every pattern within one segment/branch. */
/* Extracted into its own noinline helper for the same reason and as part of
 * the same §5.4 diagnostic as adjointAccumRowRaw() above — smaller than that
 * one but shares the `tid`-indexed pattern with its own KW_LOCAL_FENCE. */
__attribute__((noinline)) void adjointRotateRow(
        int tid, __local REAL* rawRow, __global REAL* ievc, __local REAL* rotated) {
    const int _rrpt = (PADDED_STATE_COUNT + ADJOINT_BLOCK_SP_N - 1) / ADJOINT_BLOCK_SP_N;
    for (int _m = 0; _m < _rrpt; _m++) {
        const int _rsp = tid + _m * ADJOINT_BLOCK_SP_N;
        if (_rsp < PADDED_STATE_COUNT) {
            REAL _acc = (REAL)0;
            for (int _rs = 0; _rs < PADDED_STATE_COUNT; _rs++)
                SPECTRAL_FMA(rawRow[_rs], ievc[_rs*PADDED_STATE_COUNT + _rsp], _acc);
            rotated[_rsp] = _acc;
        }
    }
    KW_LOCAL_FENCE;
}

#define ADJOINTN_ROTATE_ROW(RAWROW, ROTATED) \
    adjointRotateRow(tid, RAWROW, ievc, ROTATED);

/* Apply the all-real integral transform to a single reduced row DEST[S] and
 * atomicAdd into dGradient row `ROW`. Verbatim math from
 * kernelAdjointPhase2AllRealN, tid==0 only (one block == one row already,
 * unlike Phase2 where one thread per row shared a block). */
#define ADJOINTN_APPLY_ALLREAL_ROW(DEST, ROW) \
    if (tid == 0) { \
        const int _S_p = PADDED_STATE_COUNT; \
        const REAL _la_p = sEvalR[ROW], _ea_p = sExpat[ROW]; \
        for (int _rs_p = 0; _rs_p < _S_p; _rs_p++) { \
            const REAL _co_p = (t * fabs(_la_p - sEvalR[_rs_p]) < (REAL)1e-12) \
                ? t * _ea_p : (_ea_p - sExpat[_rs_p]) / (_la_p - sEvalR[_rs_p]); \
            ADJOINT_ATOMIC_ADD_GPU(&dGradient[(ROW)*_S_p+_rs_p], (DEST)[_rs_p] * _co_p); \
        } \
    }

/* Apply the complex-eigenvalue integral transform. Verbatim math from
 * kernelAdjointPhase2ComplexN's two branches, tid==0 only, reading the
 * shared reduced rows sOpRow0[S] (row `ls`) and, for the pair-leader case,
 * sOpRow1[S] (row `ls+1`) instead of dOpBuf.
 *
 * Each branch's per-iteration body is pulled into its own
 * `__attribute__((noinline))` helper function, the same treatment already
 * applied to the atomic CAS-retry loop above and for the same reason: these
 * bodies carry a lot of REAL temporaries (the complex-pair branch alone has
 * ~30), and inlining them directly into the `for` loop inside
 * kernelAdjointMergedN (compiled at PADDED_STATE_COUNT=32) was observed to
 * produce silently-wrong (NaN) results that turned into a hard GPU page
 * fault the moment unrelated code (e.g. a debug printf) was added nearby —
 * i.e. correctness that depended on incidental register allocation/code
 * layout, the signature of the ROCm 4.0.1 AMDGPU-backend miscompilation
 * class documented above, not a BEAGLE logic bug. Keeping only the cheap
 * loop control (the `for` and the `_ri` branch dispatch) inlined and moving
 * the heavy arithmetic out-of-line avoids it, matching the CAS-loop fix. */
__attribute__((noinline)) void adjointSingletonRealStep(
        __global REAL* dGradient, int _S_c, int ls, int _rs_c, REAL t,
        REAL _ea_c, REAL _la_c,
        __local REAL* sEvalR, __local REAL* sExpat, __local REAL* sOpRow0) {
    const REAL _co_c = (t * fabs(_la_c - sEvalR[_rs_c]) < (REAL)1e-12)
        ? t * _ea_c : (_ea_c - sExpat[_rs_c]) / (_la_c - sEvalR[_rs_c]);
    ADJOINT_ATOMIC_ADD_GPU(&dGradient[ls*_S_c+_rs_c], sOpRow0[_rs_c] * _co_c);
}

__attribute__((noinline)) void adjointSingletonComplexStep(
        __global REAL* dGradient, int _S_c, int ls, int _rs_c, REAL t,
        REAL _ea_c, REAL _la_c, REAL _ri_c,
        __local REAL* sEvalR, __local REAL* sExpat, __local REAL* sCosbt,
        __local REAL* sSinbt, __local REAL* sOpRow0) {
    const REAL _sr_c = sEvalR[_rs_c] - _la_c;
    const REAL _dn_c = _sr_c*_sr_c + _ri_c*_ri_c;
    REAL _i0_c, _i1_c;
    if (_dn_c < (REAL)1e-12) { _i0_c = t; _i1_c = (REAL)0; }
    else {
        const REAL _ex_c = sExpat[_rs_c] / _ea_c;
        _i0_c = (_ex_c*(_sr_c*sCosbt[_rs_c]+_ri_c*sSinbt[_rs_c])-_sr_c)/_dn_c;
        _i1_c = (_ex_c*(_sr_c*sSinbt[_rs_c]-_ri_c*sCosbt[_rs_c])+_ri_c)/_dn_c;
    }
    const REAL _c0_c = _ea_c*_i0_c, _c1_c = _ea_c*_i1_c;
    const REAL _n0_c = sOpRow0[_rs_c], _n1_c = sOpRow0[_rs_c+1];
    ADJOINT_ATOMIC_ADD_GPU(&dGradient[ls*_S_c+_rs_c],     _c0_c*_n0_c+_c1_c*_n1_c);
    ADJOINT_ATOMIC_ADD_GPU(&dGradient[ls*_S_c+_rs_c+1], -_c1_c*_n0_c+_c0_c*_n1_c);
}

__attribute__((noinline)) void adjointSingletonLoop(
        int tid, int ls, REAL t, __global REAL* dGradient,
        __local REAL* sEvalR, __local REAL* sEvalI, __local REAL* sExpat,
        __local REAL* sCosbt, __local REAL* sSinbt, __local REAL* sOpRow0) {
    if (tid != 0) return;
    const int _S_c = PADDED_STATE_COUNT;
    const REAL _ea_c = sExpat[ls], _la_c = sEvalR[ls];
    for (int _rs_c = 0; _rs_c < _S_c; ) {
        const REAL _ri_c = sEvalI[_rs_c];
        if (_ri_c == (REAL)0) {
            adjointSingletonRealStep(dGradient, _S_c, ls, _rs_c, t, _ea_c, _la_c,
                                      sEvalR, sExpat, sOpRow0);
            _rs_c++;
        } else {
            adjointSingletonComplexStep(dGradient, _S_c, ls, _rs_c, t, _ea_c, _la_c, _ri_c,
                                         sEvalR, sExpat, sCosbt, sSinbt, sOpRow0);
            _rs_c += 2;
        }
    }
}

#define ADJOINTN_APPLY_COMPLEX_SINGLETON() \
    adjointSingletonLoop(tid, ls, t, dGradient, sEvalR, sEvalI, sExpat, sCosbt, sSinbt, sOpRow0);

__attribute__((noinline)) void adjointPairleaderRealStep(
        __global REAL* dGradient, int _S_q, int ls, int _rs_q, REAL t,
        REAL _lr_q, REAL _li_q, REAL _ec_q, REAL _es_q, REAL _cI_q, REAL _sI_q, REAL _ea_q,
        __local REAL* sEvalR, __local REAL* sExpat,
        __local REAL* sOpRow0, __local REAL* sOpRow1) {
    const REAL _sr_q = sEvalR[_rs_q] - _lr_q;
    const REAL _dn_q = _sr_q*_sr_q + _li_q*_li_q;
    REAL _i0_q, _i1_q;
    if (_dn_q < (REAL)1e-12) { _i0_q = t; _i1_q = (REAL)0; }
    else {
        const REAL _ex_q = sExpat[_rs_q] / _ea_q;
        _i0_q = (_ex_q*(_sr_q*_cI_q+_li_q*_sI_q)-_sr_q)/_dn_q;
        _i1_q = (_ex_q*(_sr_q*_sI_q-_li_q*_cI_q)+_li_q)/_dn_q;
    }
    const REAL _p0_q=_ec_q*_i0_q+_es_q*_i1_q, _p1_q=_ec_q*_i1_q-_es_q*_i0_q;
    const REAL _p2_q=_es_q*_i0_q-_ec_q*_i1_q, _p3_q=_es_q*_i1_q+_ec_q*_i0_q;
    const REAL _n0_q=sOpRow0[_rs_q], _n1_q=sOpRow1[_rs_q];
    ADJOINT_ATOMIC_ADD_GPU(&dGradient[ls*_S_q+_rs_q],     _p0_q*_n0_q+_p1_q*_n1_q);
    ADJOINT_ATOMIC_ADD_GPU(&dGradient[(ls+1)*_S_q+_rs_q], _p2_q*_n0_q+_p3_q*_n1_q);
}

__attribute__((noinline)) void adjointPairleaderComplexStep(
        __global REAL* dGradient, int _S_q, int ls, int _rs_q, REAL t,
        REAL _lr_q, REAL _li_q, REAL _ec_q, REAL _es_q, REAL _cI_q, REAL _sI_q, REAL _ea_q,
        REAL _ri_q,
        __local REAL* sEvalR, __local REAL* sExpat, __local REAL* sCosbt, __local REAL* sSinbt,
        __local REAL* sOpRow0, __local REAL* sOpRow1) {
    const REAL _rr_q=sEvalR[_rs_q], _ri2_q=_ri_q;
    const REAL _sr_q=_rr_q-_lr_q, _si1_q=_li_q+_ri2_q, _si2_q=_ri2_q-_li_q;
    const REAL _sr2_q=_sr_q*_sr_q;
    const REAL _d1_q=_sr2_q+_si1_q*_si1_q, _d2_q=_sr2_q+_si2_q*_si2_q;
    const REAL _ex_q=(_d1_q>=(REAL)1e-12||_d2_q>=(REAL)1e-12) ? sExpat[_rs_q]/_ea_q : (REAL)0;
    const REAL _clcr_q=_cI_q*sCosbt[_rs_q], _slsr_q=_sI_q*sSinbt[_rs_q];
    const REAL _clsr_q=_cI_q*sSinbt[_rs_q], _slcr_q=_sI_q*sCosbt[_rs_q];
    REAL _i1r_q,_i1i_q;
    if (_d1_q<(REAL)1e-12){_i1r_q=t;_i1i_q=(REAL)0;}
    else{
        const REAL _cs1_q=_clcr_q-_slsr_q, _sn1_q=_slcr_q+_clsr_q;
        _i1r_q=(_sr_q*(_ex_q*_cs1_q-(REAL)1)+_si1_q*_ex_q*_sn1_q)/_d1_q;
        _i1i_q=(_sr_q*_ex_q*_sn1_q-_si1_q*(_ex_q*_cs1_q-(REAL)1))/_d1_q;
    }
    REAL _i2r_q,_i2i_q;
    if (_d2_q<(REAL)1e-12){_i2r_q=t;_i2i_q=(REAL)0;}
    else{
        const REAL _cs2_q=_clcr_q+_slsr_q, _sn2_q=_clsr_q-_slcr_q;
        _i2r_q=(_sr_q*(_ex_q*_cs2_q-(REAL)1)+_si2_q*_ex_q*_sn2_q)/_d2_q;
        _i2i_q=(_sr_q*_ex_q*_sn2_q-_si2_q*(_ex_q*_cs2_q-(REAL)1))/_d2_q;
    }
    const REAL _pr_q=_ec_q*_i1r_q+_es_q*_i1i_q, _pi_q=_ec_q*_i1i_q-_es_q*_i1r_q;
    const REAL _mr_q=_ec_q*_i2r_q-_es_q*_i2i_q, _mi_q=_ec_q*_i2i_q+_es_q*_i2r_q;
    const REAL _A_q=(REAL)0.5*(_mr_q+_pr_q), _B_q=(REAL)0.5*(_mi_q+_pi_q);
    const REAL _C_q=(REAL)0.5*(_pi_q-_mi_q), _D_q=(REAL)0.5*(_mr_q-_pr_q);
    const REAL _n00_q=sOpRow0[_rs_q],  _n01_q=sOpRow0[_rs_q+1];
    const REAL _n10_q=sOpRow1[_rs_q], _n11_q=sOpRow1[_rs_q+1];
    ADJOINT_ATOMIC_ADD_GPU(&dGradient[ls*_S_q+_rs_q],          _A_q*_n00_q+_B_q*_n01_q+_C_q*_n10_q+_D_q*_n11_q);
    ADJOINT_ATOMIC_ADD_GPU(&dGradient[ls*_S_q+_rs_q+1],       -_B_q*_n00_q+_A_q*_n01_q-_D_q*_n10_q+_C_q*_n11_q);
    ADJOINT_ATOMIC_ADD_GPU(&dGradient[(ls+1)*_S_q+_rs_q],     -_C_q*_n00_q-_D_q*_n01_q+_A_q*_n10_q+_B_q*_n11_q);
    ADJOINT_ATOMIC_ADD_GPU(&dGradient[(ls+1)*_S_q+_rs_q+1],    _D_q*_n00_q-_C_q*_n01_q-_B_q*_n10_q+_A_q*_n11_q);
}

#define ADJOINTN_APPLY_COMPLEX_PAIRLEADER() \
    if (tid == 0) { \
        const int _S_q = PADDED_STATE_COUNT; \
        const REAL _lr_q = sEvalR[ls], _li_q = sEvalI[ls]; \
        const REAL _ec_q = sExpatC[ls], _es_q = sExpatS[ls]; \
        const REAL _cI_q = sCosbt[ls],  _sI_q = sSinbt[ls]; \
        const REAL _ea_q = sExpat[ls]; \
        for (int _rs_q = 0; _rs_q < _S_q; ) { \
            const REAL _ri_q = sEvalI[_rs_q]; \
            if (_ri_q == (REAL)0) { \
                adjointPairleaderRealStep(dGradient, _S_q, ls, _rs_q, t, \
                                           _lr_q, _li_q, _ec_q, _es_q, _cI_q, _sI_q, _ea_q, \
                                           sEvalR, sExpat, sOpRow0, sOpRow1); \
                _rs_q++; \
            } else if (_ri_q > (REAL)0) { \
                adjointPairleaderComplexStep(dGradient, _S_q, ls, _rs_q, t, \
                                              _lr_q, _li_q, _ec_q, _es_q, _cI_q, _sI_q, _ea_q, _ri_q, \
                                              sEvalR, sExpat, sCosbt, sSinbt, sOpRow0, sOpRow1); \
                _rs_q += 2; \
            } else { \
                _rs_q++; \
            } \
        } \
    }

/* ═══════════════════════════════════════════════════════════════════════════
 * Adjoint cross-product kernel — generic N-state, MERGED single launch
 * (Stage 3 of the single-launch redesign — see STATUS.md/TODO.md)
 *
 * One kernel, one launch covering every branch in a call regardless of
 * (isStates,isAllReal) — grid.z spans all `count` branches, not a bucket.
 * `isStates`/`isAllReal` are read per-block from the offset-queue record
 * (fields 2 and 8) and branched on at runtime; since both are uniform
 * across every thread in a block (same record for the whole block), this
 * costs a branch, not warp/wavefront divergence. Reuses the exact same
 * ADJOINTN_* macros as the fused/batched kernels above — no math
 * duplicated, only the dispatch is different.
 * ═══════════════════════════════════════════════════════════════════════════*/

KW_GLOBAL_KERNEL void kernelAdjointMergedN(
        KW_GLOBAL_VAR REAL* KW_RESTRICT partialsOrigin,
        KW_GLOBAL_VAR int*  KW_RESTRICT statesOrigin,
        KW_GLOBAL_VAR REAL* KW_RESTRICT evecTOrigin,
        KW_GLOBAL_VAR REAL* KW_RESTRICT ievcOrigin,
        KW_GLOBAL_VAR REAL* KW_RESTRICT evalOrigin,
        KW_GLOBAL_VAR REAL* KW_RESTRICT distOrigin,
        KW_GLOBAL_VAR REAL* KW_RESTRICT patternWeights,
        KW_GLOBAL_VAR REAL* KW_RESTRICT categoryWeights,
        KW_GLOBAL_VAR REAL* KW_RESTRICT perSiteLikelihoods,
        KW_GLOBAL_VAR REAL* KW_RESTRICT gradientOrigin,
        KW_GLOBAL_VAR unsigned int* KW_RESTRICT adjointQueue,
        int totalPatterns) {

    const int tid    = KW_LOCAL_ID_0;
    const int ls     = KW_GROUP_ID_0;
    const int cat    = KW_GROUP_ID_1;
    const int branch = KW_GROUP_ID_2;
    KW_GLOBAL_VAR const unsigned int* rec = adjointQueue + branch * ADJOINT_QUEUE_STRIDE;

    const bool isStates  = (rec[2] != 0u);
    const bool isAllReal = (rec[8] != 0u);

    KW_GLOBAL_VAR REAL* prePartials  = partialsOrigin + rec[0];
    KW_GLOBAL_VAR REAL* postPartials = partialsOrigin + rec[1];
    KW_GLOBAL_VAR int*  tipStates    = statesOrigin    + rec[1];
    KW_GLOBAL_VAR REAL* evecT        = evecTOrigin     + rec[3];
    KW_GLOBAL_VAR REAL* ievc         = ievcOrigin      + rec[4];
    KW_GLOBAL_VAR REAL* eigenValues  = evalOrigin      + rec[5];
    KW_GLOBAL_VAR REAL* distances    = distOrigin      + rec[6];
    KW_GLOBAL_VAR REAL* dGradient    = gradientOrigin  + rec[7];

    KW_LOCAL_MEM REAL sEvecTCol [PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sEvecTCol1[PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sCatW;
    KW_LOCAL_MEM REAL sRedBuf   [ADJOINT_BLOCK_SP_N];
    KW_LOCAL_MEM REAL sEvalR    [PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sEvalI    [PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sExpat    [PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sCosbt    [PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sSinbt    [PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sExpatC   [PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sExpatS   [PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sRawRow0  [PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sRawRow1  [PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sOpRow0   [PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sOpRow1   [PADDED_STATE_COUNT];

    if (tid < PADDED_STATE_COUNT) {
        sEvecTCol[tid] = evecT[tid * PADDED_STATE_COUNT + ls];
        sEvalR[tid]    = eigenValues[tid];
        sEvalI[tid]    = isAllReal ? (REAL)0 : eigenValues[PADDED_STATE_COUNT + tid];
    }
    if (tid == 0) sCatW = categoryWeights[cat];
    KW_LOCAL_FENCE;

    const REAL t = distances[cat];
    const int catOff  = cat * totalPatterns * PADDED_STATE_COUNT;
    REAL regOp[PADDED_STATE_COUNT];

    if (isAllReal) {
        if (tid < PADDED_STATE_COUNT) sExpat[tid] = exp(sEvalR[tid] * t);
        KW_LOCAL_FENCE;

        ADJOINTN_ACCUM_ROW_RAW(sEvecTCol, isStates, regOp, sRawRow0)
        ADJOINTN_ROTATE_ROW(sRawRow0, sOpRow0)
        ADJOINTN_APPLY_ALLREAL_ROW(sOpRow0, ls)
    } else {
        if (sEvalI[ls] < (REAL)0) return;
        const bool isPairLeader = (sEvalI[ls] > (REAL)0);
        if (isPairLeader && tid < PADDED_STATE_COUNT)
            sEvecTCol1[tid] = evecT[tid * PADDED_STATE_COUNT + (ls+1)];

        if (tid < PADDED_STATE_COUNT) {
            const REAL e = exp(sEvalR[tid] * t);
            sExpat[tid] = e;
            REAL sv, cv;
            SPECTRAL_SINCOS(sEvalI[tid] * t, sv, cv);
            sCosbt[tid] = cv; sSinbt[tid] = sv;
            sExpatC[tid] = e * cv; sExpatS[tid] = e * sv;
        }
        KW_LOCAL_FENCE;

        ADJOINTN_ACCUM_ROW_RAW(sEvecTCol, isStates, regOp, sRawRow0)
        ADJOINTN_ROTATE_ROW(sRawRow0, sOpRow0)

        if (isPairLeader) {
            ADJOINTN_ACCUM_ROW_RAW(sEvecTCol1, isStates, regOp, sRawRow1)
            ADJOINTN_ROTATE_ROW(sRawRow1, sOpRow1)
            ADJOINTN_APPLY_COMPLEX_PAIRLEADER()
        } else {
            ADJOINTN_APPLY_COMPLEX_SINGLETON()
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Batched scatter for updateTransitionMatrices' per-branch distance queue.
 * One thread per (branch, category) queue entry: replaces what used to be
 * one MemcpyHostToDevice call per branch with two flat uploads (destOffsets,
 * values) plus this single on-device scatter. No PADDED_STATE_COUNT/
 * STATE_COUNT dependence, so this is byte-identical in kernelsSpectralIfDef4.cu.
 * ═══════════════════════════════════════════════════════════════════════════*/
KW_GLOBAL_KERNEL void kernelScatterSpectralDistances(
        KW_GLOBAL_VAR REAL* KW_RESTRICT distOrigin,
        KW_GLOBAL_VAR unsigned int* KW_RESTRICT destOffsets,
        KW_GLOBAL_VAR REAL* KW_RESTRICT values,
        int totalCount) {
    int idx = KW_GROUP_ID_0 * KW_LOCAL_SIZE_0 + KW_LOCAL_ID_0;
    if (idx < totalCount) {
        distOrigin[destOffsets[idx]] = values[idx];
    }
}

#ifdef CUDA
} /* extern "C" */
#endif
