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
 * Spectral partial-likelihood GPU kernels for 4 states — one source for CUDA
 * and OpenCL, as for the regular kernels: framework differences go through the
 * KW_ macros of GPUImplDefs.h, and C-preprocessor #ifdef directives emulate
 * C++ templates. Like kernels4Derivatives.cu, this file has no preamble of its
 * own. For OpenCL, make_opencl_spectral_kernels.sh appends it to
 * GPUImplDefs.h, kernelsAll.cu and kernels4.cu; for CUDA, kernels4.cu includes
 * it (inside its extern "C") when CUDA_SPECTRAL is defined
 * (make_cuda_spectral_kernels.sh). The other state counts are
 * kernelsSpectralIfDef.cu, with kernelsX.cu.
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
 *   template <typename Direction>        │ DIRECTION macro argument:
 *       Direction = Forward / Backward   │     FORWARD / BACKWARD
 *
 * Two usage models:
 *
 *   MODEL A — generic device function (mirrors the template):
 *     kernelSpectralBody() is a KW_DEVICE_FUNC whose body selects code paths
 *     at compile time via #ifdef.  Compile once per define-combination to
 *     obtain a single specialisation.  Six compilations yield six binaries.
 *
 *   MODEL B — single-compilation named kernels:
 *     The KW_GLOBAL_KERNEL functions at the bottom of this file invoke
 *     the phase macros directly, so all variants coexist in one OpenCL
 *     program object or CUDA module — the model BEAGLE's GPU backends load.
 *
 * Phase-macro building blocks (used in both models):
 *
 *   SPECTRAL_INDICES_GPU()
 *   SPECTRAL_COMMON_SMEM_GPU()
 *   SPECTRAL_LOAD_PARTIALS1_GPU() / SPECTRAL_LOAD_PARTIALS2_GPU()
 *   SPECTRAL_LOAD_SCALE_GPU()
 *   SPECTRAL_EXP_TERMS_GPU(N)                 — e^{Dt} of child N: sDsN, sCsN, sNbN
 *   SPECTRAL_TO_EIGEN_GPU(DIR, N, X)          — unrolled: V^{-1} X or V^T X → sQN
 *   SPECTRAL_TO_EIGEN_STATES_GPU(DIR, N, ST)  — a matrix row for a tip state → sQN
 *   SPECTRAL_TO_EIGEN_BOTH_GPU(D1, X1, D2, X2) — both children behind one fence
 *   SPECTRAL_EXP_GPU(DIR, N)                  — sQN ← e^{Dt} sQN or e^{Dt}^T sQN
 *   SPECTRAL_EXP_BOTH_GPU(D1, D2)             — both children
 *   SPECTRAL_FROM_EIGEN_GPU(DIR, N, SUM)      — unrolled: V sQN or V^{-T} sQN
 *   SPECTRAL_FROM_EIGEN_BOTH_GPU(D1, D2)      — both children, into sum1 and sum2
 *   SPECTRAL_WRITE_NO_SCALE_GPU()
 *   SPECTRAL_WRITE_FIXED_SCALE_GPU()
 *   SPECTRAL_WRITE_AUTO_SCALE_GPU()
 */

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
 *   SPECTRAL_COMMON_SMEM_GPU  → sBuf1, sBuf2, sDs1, sCs1, sNb1, sDs2, sCs2,
 *                               sNb2, sQ1, sQ2
 *   SPECTRAL_LOAD_PARTIALS1   → sP1
 *   SPECTRAL_LOAD_PARTIALS2   → sP2
 *   SPECTRAL_LOAD_SCALE       → sScale
 *   SPECTRAL_FROM_EIGEN_BOTH  → sum1, sum2
 * ═══════════════════════════════════════════════════════════════════════════*/

/* ── Direction of a product ─────────────────────────────────────────────────
 * Every kernel is built from products of a branch's P = V e^{Dt} V^{-1} with
 * a vector, in one of two directions, as on the CPU (forwardEigenBasis /
 * backwardEigenBasis):
 *   FORWARD:  P x   = V e^{Dt} V^{-1} x      (post-order children, pre-order sibling)
 *   BACKWARD: P^T x = V^{-T} e^{Dt}^T V^T x  (pre-order parent)
 * Each product is three steps: SPECTRAL_TO_EIGEN*, SPECTRAL_EXP* and
 * SPECTRAL_FROM_EIGEN*. Their DIRECTION argument selects the child's matrices
 * and the sign of the rotation, so a step cannot pair one direction's
 * matrices with the other's rotation. Child N's matrices are the kernel
 * parameters ievcN, evecN (forward) or evecTN, ievcTN (backward), stored so
 * that the thread for state k reads matrix row j at M[j * S + k]:
 *   forward:  to eigen  ievc  = dIevc,  ievc [j*S+k] = V^{-1}[k,j]
 *             from      evec  = dEvec,  evec [j*S+k] = V[k,j]
 *   backward: to eigen  evecT = dEvecT, evecT[j*S+k] = V[j,k]
 *             from      ievcT = dIevcT, ievcT[j*S+k] = V^{-1}[j,k]
 * e^{Dt} is diagonal except for a 2x2 block for each complex conjugate pair
 * (i, i + 1) with first eigenvalue a + bi, c = e^{at} cos(bt), s = e^{at} sin(bt):
 *   forward:  y_i = c u_i + s u_{i+1},  y_{i+1} = c u_{i+1} - s u_i
 *   backward: the same with s negated (e^{Dt}^T).
 * As on the CPU, pairs are found by position, not by the sign of b: the host
 * stores eigenValues[2S + k] = +1 for the first of a pair, -1 for the second
 * and 0 for a real eigenvalue. */
#define SPECTRAL_TO_EIGEN_MATRIX_FORWARD(N)     ievc##N
#define SPECTRAL_TO_EIGEN_MATRIX_BACKWARD(N)    evecT##N
#define SPECTRAL_FROM_EIGEN_MATRIX_FORWARD(N)   evec##N
#define SPECTRAL_FROM_EIGEN_MATRIX_BACKWARD(N)  ievcT##N
#define SPECTRAL_ROTATION_FORWARD               ((REAL) 1)
#define SPECTRAL_ROTATION_BACKWARD              ((REAL) -1)

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
/* sBuf1/2: child 1/2's whole matrix (both products).
 * sDs/sCs/sNb: child 1/2's e^{Dt} per eigenstate, see SPECTRAL_EXP_TERMS_GPU.
 * sQ1/2: child 1/2's vector in the eigen basis.
 * sBuf holds all PADDED_STATE_COUNT rows, NOT BLOCK_PEELING_SIZE (which is
 * tuned for the non-spectral thread layout and may exceed PADDED_STATE_COUNT). */
#define SPECTRAL_COMMON_SMEM_GPU() \
    KW_LOCAL_MEM REAL sBuf1[PADDED_STATE_COUNT][PADDED_STATE_COUNT]; \
    KW_LOCAL_MEM REAL sBuf2[PADDED_STATE_COUNT][PADDED_STATE_COUNT]; \
    KW_LOCAL_MEM REAL sDs1[PADDED_STATE_COUNT]; \
    KW_LOCAL_MEM REAL sCs1[PADDED_STATE_COUNT]; \
    KW_LOCAL_MEM int  sNb1[PADDED_STATE_COUNT]; \
    KW_LOCAL_MEM REAL sDs2[PADDED_STATE_COUNT]; \
    KW_LOCAL_MEM REAL sCs2[PADDED_STATE_COUNT]; \
    KW_LOCAL_MEM int  sNb2[PADDED_STATE_COUNT]; \
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
/* One per pattern, loaded by the state-0 thread of its row: PATTERN_BLOCK_SIZE
 * (16) exceeds PADDED_STATE_COUNT (4), the threads of a row. */
#define SPECTRAL_LOAD_SCALE_GPU() \
    KW_LOCAL_MEM REAL sScale[PATTERN_BLOCK_SIZE]; \
    if (state == 0) \
        sScale[patIdx] = scalingFactors[KW_GROUP_ID_0 * PATTERN_BLOCK_SIZE + patIdx];

/* ── e^{Dt} of child N for this block's rate category ───────────────────── */
/* patIdx-0 threads, one per eigenstate k. eigenValuesN is [real parts |
 * imaginary parts | pair positions]; distancesN[matrix] = branch length *
 * category rate. The forward rotation of every eigenstate is then
 *   y_k = sDs[k] u_k + sCs[k] u_{sNb[k]}
 * with sDs = c, sCs = s for the first of a pair, -s for the second and 0 for
 * a real eigenvalue, and sNb[k] the other eigenstate of k's pair (k itself if
 * real). Both eigenstates of a pair take a and b from the first, as on the
 * CPU. No fence: the steps that read these start with one. */
#define SPECTRAL_EXP_TERMS_GPU(N) \
    if (patIdx == 0) { \
        const int  pos   = (int) eigenValues##N[2 * PADDED_STATE_COUNT + state]; \
        const int  first = (pos < 0) ? state - 1 : state; \
        const REAL t     = distances##N[matrix]; \
        const REAL e     = exp(eigenValues##N[first] * t); \
        REAL cv, sv; \
        SPECTRAL_SINCOS(eigenValues##N[PADDED_STATE_COUNT + first] * t, sv, cv); \
        sDs##N[state] = e * cv; \
        sCs##N[state] = (REAL) pos * e * sv; \
        sNb##N[state] = state + pos; \
    }

/* ── To the eigen basis: sQN = A X ──────────────────────────────────────── */
/* A = V^{-1} (FORWARD) or V^T (BACKWARD) of child N, through sBufN.
 * X: sP1, sP2, or another child's sQ holding an intermediate vector.
 * q[k] = Σ_j A[j*S + k] X[j]. For 4 states the 4×4 matrix is 16 contiguous
 * REALs: the 16 threads with patIdx<4 and state=0..3 load all of it in one
 * coalesced transaction (flat index patIdx*4+state = 0..15), matching the
 * LOAD_MATRIX_4_GPU pattern used by the non-spectral kernels. The 4-element
 * dot product is fully unrolled; no trailing fence is emitted because
 * SPECTRAL_EXP*_GPU, which always follows, opens with one.
 * Scoped in {} so q_loc does not collide across consecutive invocations. */
#define SPECTRAL_TO_EIGEN_GPU(DIRECTION, N, X) \
    { \
        if (patIdx < PADDED_STATE_COUNT) \
            sBuf##N[patIdx][state] = SPECTRAL_TO_EIGEN_MATRIX_##DIRECTION(N)[patIdx * PADDED_STATE_COUNT + state]; \
        KW_LOCAL_FENCE; \
        REAL q_loc = (REAL)0; \
        SPECTRAL_FMA(sBuf##N[0][state], X[patIdx][0], q_loc); \
        SPECTRAL_FMA(sBuf##N[1][state], X[patIdx][1], q_loc); \
        SPECTRAL_FMA(sBuf##N[2][state], X[patIdx][2], q_loc); \
        SPECTRAL_FMA(sBuf##N[3][state], X[patIdx][3], q_loc); \
        sQ##N[patIdx][state] = q_loc; \
    }

/* ── To the eigen basis for a tip state: sQN = A e_s ────────────────────── */
/* A[s*S + k] = (A e_s)[k]: threads k = 0..S-1 read consecutive addresses.
 * A missing state (s >= PADDED_STATE_COUNT) is the all-ones vector, whose
 * product is the sum of A's rows. */
#define SPECTRAL_TO_EIGEN_STATES_GPU(DIRECTION, N, STATES_ARR) \
    { \
        REAL q_loc = (REAL)0; \
        if (pattern < totalPatterns) { \
            int s = (STATES_ARR)[pattern]; \
            if (s < PADDED_STATE_COUNT) { \
                q_loc = SPECTRAL_TO_EIGEN_MATRIX_##DIRECTION(N)[s * PADDED_STATE_COUNT + state]; \
            } else { \
                for (int j = 0; j < PADDED_STATE_COUNT; j++) \
                    q_loc += SPECTRAL_TO_EIGEN_MATRIX_##DIRECTION(N)[j * PADDED_STATE_COUNT + state]; \
            } \
        } \
        sQ##N[patIdx][state] = q_loc; \
    }

/* ── To the eigen basis for both children, fused ────────────────────────── */
/* Loads both children's matrices behind a single fence instead of two. */
#define SPECTRAL_TO_EIGEN_BOTH_GPU(DIRECTION1, X1, DIRECTION2, X2) \
    { \
        if (patIdx < PADDED_STATE_COUNT) { \
            sBuf1[patIdx][state] = SPECTRAL_TO_EIGEN_MATRIX_##DIRECTION1(1)[patIdx * PADDED_STATE_COUNT + state]; \
            sBuf2[patIdx][state] = SPECTRAL_TO_EIGEN_MATRIX_##DIRECTION2(2)[patIdx * PADDED_STATE_COUNT + state]; \
        } \
        KW_LOCAL_FENCE; \
        REAL q_loc1 = (REAL)0, q_loc2 = (REAL)0; \
        SPECTRAL_FMA(sBuf1[0][state], X1[patIdx][0], q_loc1); \
        SPECTRAL_FMA(sBuf1[1][state], X1[patIdx][1], q_loc1); \
        SPECTRAL_FMA(sBuf1[2][state], X1[patIdx][2], q_loc1); \
        SPECTRAL_FMA(sBuf1[3][state], X1[patIdx][3], q_loc1); \
        SPECTRAL_FMA(sBuf2[0][state], X2[patIdx][0], q_loc2); \
        SPECTRAL_FMA(sBuf2[1][state], X2[patIdx][1], q_loc2); \
        SPECTRAL_FMA(sBuf2[2][state], X2[patIdx][2], q_loc2); \
        SPECTRAL_FMA(sBuf2[3][state], X2[patIdx][3], q_loc2); \
        sQ1[patIdx][state] = q_loc1; \
        sQ2[patIdx][state] = q_loc2; \
    }

/* ── e^{Dt} (FORWARD) or e^{Dt}^T (BACKWARD), in place on sQN ───────────── */
/* Fenced before (sQN and the e^{Dt} terms are written by other threads),
 * between the reads and the writes (the rotation of k reads its partner,
 * which another thread overwrites) and after. */
#define SPECTRAL_EXP_GPU(DIRECTION, N) \
    KW_LOCAL_FENCE; \
    { \
        const REAL y_loc = sDs##N[state] * sQ##N[patIdx][state] \
                         + SPECTRAL_ROTATION_##DIRECTION * sCs##N[state] * sQ##N[patIdx][sNb##N[state]]; \
        KW_LOCAL_FENCE; \
        sQ##N[patIdx][state] = y_loc; \
    } \
    KW_LOCAL_FENCE;

#define SPECTRAL_EXP_BOTH_GPU(DIRECTION1, DIRECTION2) \
    KW_LOCAL_FENCE; \
    { \
        const REAL y_loc1 = sDs1[state] * sQ1[patIdx][state] \
                          + SPECTRAL_ROTATION_##DIRECTION1 * sCs1[state] * sQ1[patIdx][sNb1[state]]; \
        const REAL y_loc2 = sDs2[state] * sQ2[patIdx][state] \
                          + SPECTRAL_ROTATION_##DIRECTION2 * sCs2[state] * sQ2[patIdx][sNb2[state]]; \
        KW_LOCAL_FENCE; \
        sQ1[patIdx][state] = y_loc1; \
        sQ2[patIdx][state] = y_loc2; \
    } \
    KW_LOCAL_FENCE;

/* ── From the eigen basis: SUM += (B sQN)[state] ────────────────────────── */
/* B = V (FORWARD) or V^{-T} (BACKWARD) of child N, through sBufN, with the
 * same coalesced 16-element load as SPECTRAL_TO_EIGEN_GPU; SUM is the
 * caller's accumulator. No trailing fence: a caller that then overwrites sQN
 * fences first. */
#define SPECTRAL_FROM_EIGEN_GPU(DIRECTION, N, SUM) \
    if (patIdx < PADDED_STATE_COUNT) \
        sBuf##N[patIdx][state] = SPECTRAL_FROM_EIGEN_MATRIX_##DIRECTION(N)[patIdx * PADDED_STATE_COUNT + state]; \
    KW_LOCAL_FENCE; \
    SPECTRAL_FMA(sBuf##N[0][state], sQ##N[patIdx][0], SUM); \
    SPECTRAL_FMA(sBuf##N[1][state], sQ##N[patIdx][1], SUM); \
    SPECTRAL_FMA(sBuf##N[2][state], sQ##N[patIdx][2], SUM); \
    SPECTRAL_FMA(sBuf##N[3][state], sQ##N[patIdx][3], SUM);

/* Both children in one pass with fully unrolled 4-element FMAs. No trailing
 * fence: the global-memory write that follows needs none. Declares sum1, sum2
 * at function scope so SPECTRAL_WRITE_*_GPU can use them. */
#define SPECTRAL_FROM_EIGEN_BOTH_GPU(DIRECTION1, DIRECTION2) \
    REAL sum1 = (REAL)0, sum2 = (REAL)0; \
    if (patIdx < PADDED_STATE_COUNT) { \
        sBuf1[patIdx][state] = SPECTRAL_FROM_EIGEN_MATRIX_##DIRECTION1(1)[patIdx * PADDED_STATE_COUNT + state]; \
        sBuf2[patIdx][state] = SPECTRAL_FROM_EIGEN_MATRIX_##DIRECTION2(2)[patIdx * PADDED_STATE_COUNT + state]; \
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

/* sScale is loaded before the fences of the products → valid here. */
#define SPECTRAL_WRITE_FIXED_SCALE_GPU() \
    if (pattern < totalPatterns) \
        partials3[u] = sum1 * sum2 * ((REAL)1 / sScale[patIdx]);

/* Auto-scaling: detect overflow/underflow per pattern, rescale if needed, and
 * write the per-pattern exponent to scalingFactors[matrix*totalPatterns+pattern]
 * as a signed char.  Reuses sQ1[patIdx][*] as scratch for the per-pattern
 * max-exponent reduction, after a fence: SPECTRAL_FROM_EIGEN_BOTH_GPU has none,
 * and the threads of a pattern row need not run in lockstep (CUDA since Volta).
 * Thread 0 of each pattern row does a linear scan so correctness does not
 * depend on PADDED_STATE_COUNT being a power of two. */
#define SPECTRAL_WRITE_AUTO_SCALE_GPU() \
    { \
        REAL tmpPartial = sum1 * sum2; \
        int  expTmp; \
        REAL sigTmp = frexp(tmpPartial, &expTmp); \
        KW_LOCAL_FENCE; \
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

    SPECTRAL_EXP_TERMS_GPU(1)
    SPECTRAL_EXP_TERMS_GPU(2)

    /* To the eigen basis, forward.
     * PP case fuses both children into one peel loop (half the barriers).
     * SP/SS cases fall through to single-child macros. */
#if !defined(SPECTRAL_CHILD1_STATES) && !defined(SPECTRAL_CHILD2_STATES)
    SPECTRAL_TO_EIGEN_BOTH_GPU(FORWARD, sP1, FORWARD, sP2)
#else
    #ifdef SPECTRAL_CHILD1_STATES
        SPECTRAL_TO_EIGEN_STATES_GPU(FORWARD, 1, states1)
    #else
        SPECTRAL_TO_EIGEN_GPU(FORWARD, 1, sP1)
    #endif
    #ifdef SPECTRAL_CHILD2_STATES
        SPECTRAL_TO_EIGEN_STATES_GPU(FORWARD, 2, states2)
    #else
        SPECTRAL_TO_EIGEN_GPU(FORWARD, 2, sP2)
    #endif
#endif

    SPECTRAL_EXP_BOTH_GPU(FORWARD, FORWARD)
    SPECTRAL_FROM_EIGEN_BOTH_GPU(FORWARD, FORWARD)   /* declares sum1, sum2 */

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
 * Each kernel invokes the phase macros directly so that all variants
 * coexist in one translation unit / OpenCL program object.  Each kernel has
 * a type-specific parameter list (no superfluous null pointers in the API).
 * ═══════════════════════════════════════════════════════════════════════════*/

/* ── PartialsPartials ──────────────────────────────────────────────────── */
/* Post-order: dest = (P_1 x_1) ⊙ (P_2 x_2), both forward. */

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
    SPECTRAL_EXP_TERMS_GPU(1)
    SPECTRAL_EXP_TERMS_GPU(2)
    SPECTRAL_TO_EIGEN_BOTH_GPU(FORWARD, sP1, FORWARD, sP2)
    SPECTRAL_EXP_BOTH_GPU(FORWARD, FORWARD)
    SPECTRAL_FROM_EIGEN_BOTH_GPU(FORWARD, FORWARD)
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
    SPECTRAL_EXP_TERMS_GPU(1)
    SPECTRAL_EXP_TERMS_GPU(2)
    SPECTRAL_TO_EIGEN_BOTH_GPU(FORWARD, sP1, FORWARD, sP2)
    SPECTRAL_EXP_BOTH_GPU(FORWARD, FORWARD)
    SPECTRAL_FROM_EIGEN_BOTH_GPU(FORWARD, FORWARD)
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
    SPECTRAL_EXP_TERMS_GPU(1)
    SPECTRAL_EXP_TERMS_GPU(2)
    SPECTRAL_TO_EIGEN_STATES_GPU(FORWARD, 1, states1)
    SPECTRAL_TO_EIGEN_GPU(FORWARD, 2, sP2)
    SPECTRAL_EXP_BOTH_GPU(FORWARD, FORWARD)
    SPECTRAL_FROM_EIGEN_BOTH_GPU(FORWARD, FORWARD)
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
    SPECTRAL_EXP_TERMS_GPU(1)
    SPECTRAL_EXP_TERMS_GPU(2)
    SPECTRAL_TO_EIGEN_STATES_GPU(FORWARD, 1, states1)
    SPECTRAL_TO_EIGEN_GPU(FORWARD, 2, sP2)
    SPECTRAL_EXP_BOTH_GPU(FORWARD, FORWARD)
    SPECTRAL_FROM_EIGEN_BOTH_GPU(FORWARD, FORWARD)
    SPECTRAL_WRITE_FIXED_SCALE_GPU()
}

/* ── StatesStates ──────────────────────────────────────────────────────── */
/* No sP1 or sP2 needed; sBuf1/sBuf2 are used only from the eigen basis. */

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
    SPECTRAL_EXP_TERMS_GPU(1)
    SPECTRAL_EXP_TERMS_GPU(2)
    SPECTRAL_TO_EIGEN_STATES_GPU(FORWARD, 1, states1)
    SPECTRAL_TO_EIGEN_STATES_GPU(FORWARD, 2, states2)
    SPECTRAL_EXP_BOTH_GPU(FORWARD, FORWARD)
    SPECTRAL_FROM_EIGEN_BOTH_GPU(FORWARD, FORWARD)
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
    SPECTRAL_EXP_TERMS_GPU(1)
    SPECTRAL_EXP_TERMS_GPU(2)
    SPECTRAL_TO_EIGEN_STATES_GPU(FORWARD, 1, states1)
    SPECTRAL_TO_EIGEN_STATES_GPU(FORWARD, 2, states2)
    SPECTRAL_EXP_BOTH_GPU(FORWARD, FORWARD)
    SPECTRAL_FROM_EIGEN_BOTH_GPU(FORWARD, FORWARD)
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
    SPECTRAL_EXP_TERMS_GPU(1)
    SPECTRAL_EXP_TERMS_GPU(2)
    SPECTRAL_TO_EIGEN_BOTH_GPU(FORWARD, sP1, FORWARD, sP2)
    SPECTRAL_EXP_BOTH_GPU(FORWARD, FORWARD)
    SPECTRAL_FROM_EIGEN_BOTH_GPU(FORWARD, FORWARD)
    SPECTRAL_WRITE_AUTO_SCALE_GPU()
}

/* ── Growing (pre-order) kernels ─────────────────────────────────────────
 *
 * partials1 = the parent's pre-order partials, which go BACKWARD through
 *             branch 1 (evecT1 = dEvecT, ievcT1 = dIevcT);
 * child 2   = the sibling (partials2 or states2), which goes FORWARD through
 *             branch 2 (ievc2 = dIevc, evec2 = dEvec).
 *
 *   BOTTOM:        dest = P_1^T (p_par ⊙ P_2 x_sib)
 *   TOP, NotRoot:  dest = (P_1^T p_par) ⊙ (P_2 x_sib)
 *   TOP, Root:     dest = p_root ⊙ (P_2 x_sib)   (no branch 1)
 * ────────────────────────────────────────────────────────────────────────── */

KW_GLOBAL_KERNEL void kernelPartialsPartialsGrowingSpectral(
        KW_GLOBAL_VAR REAL* KW_RESTRICT partials1,
        KW_GLOBAL_VAR REAL* KW_RESTRICT partials2,
        KW_GLOBAL_VAR REAL* KW_RESTRICT partials3,
        KW_GLOBAL_VAR REAL* KW_RESTRICT evecT1,
        KW_GLOBAL_VAR REAL* KW_RESTRICT ievcT1,
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
    SPECTRAL_EXP_TERMS_GPU(1)
    SPECTRAL_EXP_TERMS_GPU(2)
    SPECTRAL_TO_EIGEN_GPU(FORWARD, 2, sP2)
    SPECTRAL_EXP_GPU(FORWARD, 2)
    {
        REAL sibling = (REAL)0;
        SPECTRAL_FROM_EIGEN_GPU(FORWARD, 2, sibling)
        KW_LOCAL_FENCE;                      /* every thread has read sQ2 */
        sQ2[patIdx][state] = sibling * sP1[patIdx][state];   /* p_par ⊙ P_2 x_sib */
    }
    SPECTRAL_TO_EIGEN_GPU(BACKWARD, 1, sQ2)
    SPECTRAL_EXP_GPU(BACKWARD, 1)
    {
        REAL sum = (REAL)0;
        SPECTRAL_FROM_EIGEN_GPU(BACKWARD, 1, sum)
        if (pattern < totalPatterns)
            partials3[u] = sum;
    }
}

KW_GLOBAL_KERNEL void kernelPartialsStatesGrowingSpectral(
        KW_GLOBAL_VAR REAL* KW_RESTRICT partials1,
        KW_GLOBAL_VAR int*  KW_RESTRICT states2,
        KW_GLOBAL_VAR REAL* KW_RESTRICT partials3,
        KW_GLOBAL_VAR REAL* KW_RESTRICT evecT1,
        KW_GLOBAL_VAR REAL* KW_RESTRICT ievcT1,
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
    SPECTRAL_EXP_TERMS_GPU(1)
    SPECTRAL_EXP_TERMS_GPU(2)
    SPECTRAL_TO_EIGEN_STATES_GPU(FORWARD, 2, states2)
    SPECTRAL_EXP_GPU(FORWARD, 2)
    {
        REAL sibling = (REAL)0;
        SPECTRAL_FROM_EIGEN_GPU(FORWARD, 2, sibling)
        KW_LOCAL_FENCE;                      /* every thread has read sQ2 */
        sQ2[patIdx][state] = sibling * sP1[patIdx][state];   /* p_par ⊙ P_2 e_s */
    }
    SPECTRAL_TO_EIGEN_GPU(BACKWARD, 1, sQ2)
    SPECTRAL_EXP_GPU(BACKWARD, 1)
    {
        REAL sum = (REAL)0;
        SPECTRAL_FROM_EIGEN_GPU(BACKWARD, 1, sum)
        if (pattern < totalPatterns)
            partials3[u] = sum;
    }
}

KW_GLOBAL_KERNEL void kernelPartialsPartialsGrowingTopSpectral(
        KW_GLOBAL_VAR REAL* KW_RESTRICT partials1,
        KW_GLOBAL_VAR REAL* KW_RESTRICT partials2,
        KW_GLOBAL_VAR REAL* KW_RESTRICT partials3,
        KW_GLOBAL_VAR REAL* KW_RESTRICT evecT1,
        KW_GLOBAL_VAR REAL* KW_RESTRICT ievcT1,
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
    SPECTRAL_EXP_TERMS_GPU(1)
    SPECTRAL_EXP_TERMS_GPU(2)
    SPECTRAL_TO_EIGEN_BOTH_GPU(BACKWARD, sP1, FORWARD, sP2)
    SPECTRAL_EXP_BOTH_GPU(BACKWARD, FORWARD)
    SPECTRAL_FROM_EIGEN_BOTH_GPU(BACKWARD, FORWARD)
    SPECTRAL_WRITE_NO_SCALE_GPU()           /* (P_1^T p_par) ⊙ (P_2 p_sib) */
}

KW_GLOBAL_KERNEL void kernelPartialsStatesGrowingTopSpectral(
        KW_GLOBAL_VAR REAL* KW_RESTRICT partials1,
        KW_GLOBAL_VAR int*  KW_RESTRICT states2,
        KW_GLOBAL_VAR REAL* KW_RESTRICT partials3,
        KW_GLOBAL_VAR REAL* KW_RESTRICT evecT1,
        KW_GLOBAL_VAR REAL* KW_RESTRICT ievcT1,
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
    SPECTRAL_EXP_TERMS_GPU(1)
    SPECTRAL_EXP_TERMS_GPU(2)
    SPECTRAL_TO_EIGEN_GPU(BACKWARD, 1, sP1)
    SPECTRAL_TO_EIGEN_STATES_GPU(FORWARD, 2, states2)
    SPECTRAL_EXP_BOTH_GPU(BACKWARD, FORWARD)
    SPECTRAL_FROM_EIGEN_BOTH_GPU(BACKWARD, FORWARD)
    SPECTRAL_WRITE_NO_SCALE_GPU()           /* (P_1^T p_par) ⊙ (P_2 e_s) */
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
    SPECTRAL_EXP_TERMS_GPU(2)
    SPECTRAL_TO_EIGEN_GPU(FORWARD, 2, sP2)
    SPECTRAL_EXP_GPU(FORWARD, 2)
    {
        REAL sibling = (REAL)0;
        SPECTRAL_FROM_EIGEN_GPU(FORWARD, 2, sibling)
        if (pattern < totalPatterns)
            partials3[u] = sibling * sP1[patIdx][state];
    }
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
    SPECTRAL_EXP_TERMS_GPU(2)
    SPECTRAL_TO_EIGEN_STATES_GPU(FORWARD, 2, states2)
    SPECTRAL_EXP_GPU(FORWARD, 2)
    {
        REAL sibling = (REAL)0;
        SPECTRAL_FROM_EIGEN_GPU(FORWARD, 2, sibling)
        if (pattern < totalPatterns)
            partials3[u] = sibling * sP1[patIdx][state];
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Adjoint cross-product kernels — 4-state (OpenCL-compatible)
 *
 * Four variants: {Partials, States} × {AllReal, Complex}.
 * Thread layout: KW_LOCAL_ID_0 ∈ [0, ADJOINT_BLOCK_SP4), KW_GROUP_ID_0 = cat.
 * No barriers inside the pattern loop: S=4, so full evecT/ievc (16 elements)
 * fit in shared memory loaded before the loop.
 * Reduction: tree reduction through sRedBuf[128][16].
 * Atomic write: CAS-loop ADJOINT_ATOMIC_ADD_GPU (OpenCL) / native (CUDA).
 * ═══════════════════════════════════════════════════════════════════════════*/

#if defined(FW_OPENCL) && defined(DOUBLE_PRECISION)
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#endif

#define ADJOINT_BLOCK_SP4  128
#define ADJOINT_SS4        (PADDED_STATE_COUNT * PADDED_STATE_COUNT)

#ifdef CUDA
#define ADJOINT_ATOMIC_ADD_GPU(ptr, val)  atomicAdd((ptr), (REAL)(val))
#elif defined(DOUBLE_PRECISION)
#define ADJOINT_ATOMIC_ADD_GPU(ptr, val) \
    do { \
        __global long* _ap = (__global long*)(ptr); \
        long _ao, _an; \
        do { _ao = *_ap; _an = as_long(as_double(_ao) + (double)(val)); } \
        while (atom_cmpxchg(_ap, _ao, _an) != _ao); \
    } while(0)
#else
#define ADJOINT_ATOMIC_ADD_GPU(ptr, val) \
    do { \
        __global int* _ap = (__global int*)(ptr); \
        int _ao, _an; \
        do { _ao = *_ap; _an = as_int(as_float(_ao) + (float)(val)); } \
        while (atomic_cmpxchg(_ap, _ao, _an) != _ao); \
    } while(0)
#endif

/* Shared memory — all-real variant (no sincos arrays) */
#define ADJOINT4_ALLREAL_SMEM() \
    KW_LOCAL_MEM REAL sEvecT[ADJOINT_SS4]; \
    KW_LOCAL_MEM REAL sIevc [ADJOINT_SS4]; \
    KW_LOCAL_MEM REAL sEvalR[PADDED_STATE_COUNT]; \
    KW_LOCAL_MEM REAL sExpat[PADDED_STATE_COUNT]; \
    KW_LOCAL_MEM REAL sTime; \
    KW_LOCAL_MEM REAL sCatW; \
    KW_LOCAL_MEM REAL sRedBuf[ADJOINT_BLOCK_SP4][ADJOINT_SS4];

/* Shared memory — complex variant (adds sincos arrays) */
#define ADJOINT4_COMPLEX_SMEM() \
    KW_LOCAL_MEM REAL sEvecT[ADJOINT_SS4]; \
    KW_LOCAL_MEM REAL sIevc [ADJOINT_SS4]; \
    KW_LOCAL_MEM REAL sEvalR[PADDED_STATE_COUNT]; \
    KW_LOCAL_MEM REAL sEvalI[PADDED_STATE_COUNT]; \
    KW_LOCAL_MEM REAL sExpat[PADDED_STATE_COUNT]; \
    KW_LOCAL_MEM REAL sCosbt[PADDED_STATE_COUNT]; \
    KW_LOCAL_MEM REAL sSinbt[PADDED_STATE_COUNT]; \
    KW_LOCAL_MEM REAL sExpatC[PADDED_STATE_COUNT]; \
    KW_LOCAL_MEM REAL sExpatS[PADDED_STATE_COUNT]; \
    KW_LOCAL_MEM REAL sTime; \
    KW_LOCAL_MEM REAL sCatW; \
    KW_LOCAL_MEM REAL sRedBuf[ADJOINT_BLOCK_SP4][ADJOINT_SS4];

/* Load matrices and compute exponentials — all-real */
#define ADJOINT4_LOAD_ALLREAL(EVECT, IEVC, EVALUES, DIST, CW) \
    if (tid < ADJOINT_SS4) { sEvecT[tid] = (EVECT)[tid]; sIevc[tid] = (IEVC)[tid]; } \
    if (tid < PADDED_STATE_COUNT) sEvalR[tid] = (EVALUES)[tid]; \
    if (tid == 0) { sTime = (DIST)[cat]; sCatW = (CW)[cat]; } \
    KW_LOCAL_FENCE; \
    if (tid < PADDED_STATE_COUNT) sExpat[tid] = exp(sEvalR[tid] * sTime); \
    KW_LOCAL_FENCE;

/* Load matrices and compute exponentials + sincos — complex */
#define ADJOINT4_LOAD_COMPLEX(EVECT, IEVC, EVALUES, DIST, CW) \
    if (tid < ADJOINT_SS4) { sEvecT[tid] = (EVECT)[tid]; sIevc[tid] = (IEVC)[tid]; } \
    if (tid < PADDED_STATE_COUNT) { \
        sEvalR[tid] = (EVALUES)[tid]; \
        sEvalI[tid] = (EVALUES)[PADDED_STATE_COUNT + tid]; \
    } \
    if (tid == 0) { sTime = (DIST)[cat]; sCatW = (CW)[cat]; } \
    KW_LOCAL_FENCE; \
    if (tid < PADDED_STATE_COUNT) { \
        const REAL _e4 = exp(sEvalR[tid] * sTime); \
        sExpat[tid] = _e4; \
        REAL _sv4, _cv4; \
        SPECTRAL_SINCOS(sEvalI[tid] * sTime, _sv4, _cv4); \
        sCosbt[tid] = _cv4; sSinbt[tid] = _sv4; \
        sExpatC[tid] = _e4 * _cv4; sExpatS[tid] = _e4 * _sv4; \
    } \
    KW_LOCAL_FENCE;

/* Pattern accumulation — Partials child */
#define ADJOINT4_ACCUM_PARTIALS(PRE, POST, NP, CATOFF) \
    for (int _k4 = tid; _k4 < (NP); _k4 += ADJOINT_BLOCK_SP4) { \
        REAL _pre4[PADDED_STATE_COUNT], _post4[PADDED_STATE_COUNT]; \
        for (int _s4 = 0; _s4 < PADDED_STATE_COUNT; _s4++) { \
            _pre4 [_s4] = (PRE) [(CATOFF) + _k4 * PADDED_STATE_COUNT + _s4]; \
            _post4[_s4] = (POST)[(CATOFF) + _k4 * PADDED_STATE_COUNT + _s4]; \
        } \
        REAL _lhs4[PADDED_STATE_COUNT]; \
        for (int _ls4 = 0; _ls4 < PADDED_STATE_COUNT; _ls4++) { \
            REAL _q4 = (REAL)0; \
            for (int _j4 = 0; _j4 < PADDED_STATE_COUNT; _j4++) \
                SPECTRAL_FMA(sEvecT[_j4 * PADDED_STATE_COUNT + _ls4], _pre4[_j4], _q4); \
            _lhs4[_ls4] = _q4; \
        } \
        REAL _rhs4[PADDED_STATE_COUNT]; \
        for (int _rs4 = 0; _rs4 < PADDED_STATE_COUNT; _rs4++) { \
            REAL _q4 = (REAL)0; \
            for (int _j4 = 0; _j4 < PADDED_STATE_COUNT; _j4++) \
                SPECTRAL_FMA(sIevc[_j4 * PADDED_STATE_COUNT + _rs4], _post4[_j4], _q4); \
            _rhs4[_rs4] = _q4; \
        } \
        const REAL _sc4 = patternWeights[_k4] * sCatW / exp(perSiteLikelihoods[_k4]); \
        for (int _ls4 = 0; _ls4 < PADDED_STATE_COUNT; _ls4++) { \
            const REAL _lv4 = _lhs4[_ls4] * _sc4; \
            for (int _rs4 = 0; _rs4 < PADDED_STATE_COUNT; _rs4++) \
                regOp[_ls4 * PADDED_STATE_COUNT + _rs4] += _lv4 * _rhs4[_rs4]; \
        } \
    }

/* Pattern accumulation — States child */
#define ADJOINT4_ACCUM_STATES(PRE, STATES, NP, CATOFF) \
    for (int _k4 = tid; _k4 < (NP); _k4 += ADJOINT_BLOCK_SP4) { \
        REAL _pre4[PADDED_STATE_COUNT]; \
        for (int _s4 = 0; _s4 < PADDED_STATE_COUNT; _s4++) \
            _pre4[_s4] = (PRE)[(CATOFF) + _k4 * PADDED_STATE_COUNT + _s4]; \
        REAL _lhs4[PADDED_STATE_COUNT]; \
        for (int _ls4 = 0; _ls4 < PADDED_STATE_COUNT; _ls4++) { \
            REAL _q4 = (REAL)0; \
            for (int _j4 = 0; _j4 < PADDED_STATE_COUNT; _j4++) \
                SPECTRAL_FMA(sEvecT[_j4 * PADDED_STATE_COUNT + _ls4], _pre4[_j4], _q4); \
            _lhs4[_ls4] = _q4; \
        } \
        const int _st4 = (STATES)[_k4]; \
        REAL _rhs4[PADDED_STATE_COUNT]; \
        if (_st4 < PADDED_STATE_COUNT) { \
            for (int _rs4 = 0; _rs4 < PADDED_STATE_COUNT; _rs4++) \
                _rhs4[_rs4] = sIevc[_st4 * PADDED_STATE_COUNT + _rs4]; \
        } else { \
            for (int _rs4 = 0; _rs4 < PADDED_STATE_COUNT; _rs4++) { \
                REAL _sum4 = (REAL)0; \
                for (int _j4 = 0; _j4 < PADDED_STATE_COUNT; _j4++) \
                    _sum4 += sIevc[_j4 * PADDED_STATE_COUNT + _rs4]; \
                _rhs4[_rs4] = _sum4; \
            } \
        } \
        const REAL _sc4 = patternWeights[_k4] * sCatW / exp(perSiteLikelihoods[_k4]); \
        for (int _ls4 = 0; _ls4 < PADDED_STATE_COUNT; _ls4++) { \
            const REAL _lv4 = _lhs4[_ls4] * _sc4; \
            for (int _rs4 = 0; _rs4 < PADDED_STATE_COUNT; _rs4++) \
                regOp[_ls4 * PADDED_STATE_COUNT + _rs4] += _lv4 * _rhs4[_rs4]; \
        } \
    }

/* Tree reduction: regOp[16] per thread → sRedBuf[0][*] = total */
#define ADJOINT4_REDUCE() \
    for (int _i4 = 0; _i4 < ADJOINT_SS4; _i4++) sRedBuf[tid][_i4] = regOp[_i4]; \
    KW_LOCAL_FENCE; \
    for (int _str4 = ADJOINT_BLOCK_SP4 >> 1; _str4 >= 1; _str4 >>= 1) { \
        if (tid < _str4) \
            for (int _i4 = 0; _i4 < ADJOINT_SS4; _i4++) \
                sRedBuf[tid][_i4] += sRedBuf[tid + _str4][_i4]; \
        KW_LOCAL_FENCE; \
    }

/* Integral transform + atomicAdd — all-real eigenvalues */
#define ADJOINT4_APPLY_ALLREAL(GRAD) \
    if (tid == 0) { \
        const REAL _t4 = sTime; \
        const int _S4 = PADDED_STATE_COUNT; \
        for (int _ls4 = 0; _ls4 < _S4; _ls4++) { \
            const REAL _la4 = sEvalR[_ls4], _ea4 = sExpat[_ls4]; \
            for (int _rs4 = 0; _rs4 < _S4; _rs4++) { \
                const REAL _co4 = (_t4 * fabs(_la4 - sEvalR[_rs4]) < (REAL)1e-12) \
                    ? _t4 * _ea4 : (_ea4 - sExpat[_rs4]) / (_la4 - sEvalR[_rs4]); \
                ADJOINT_ATOMIC_ADD_GPU(&(GRAD)[_ls4*_S4+_rs4], \
                    sRedBuf[0][_ls4*_S4+_rs4] * _co4); \
            } \
        } \
    }

/* Integral transform + atomicAdd — complex eigenvalues */
#define ADJOINT4_APPLY_COMPLEX(GRAD) \
    if (tid == 0) { \
        const REAL _t4 = sTime; \
        const int _S4 = PADDED_STATE_COUNT; \
        for (int _ls4 = 0; _ls4 < _S4; ) { \
            const REAL _li4 = sEvalI[_ls4]; \
            if (_li4 == (REAL)0) { \
                for (int _rs4 = 0; _rs4 < _S4; ) { \
                    const REAL _ri4 = sEvalI[_rs4]; \
                    if (_ri4 == (REAL)0) { \
                        const REAL _co4 = (_t4 * fabs(sEvalR[_ls4]-sEvalR[_rs4]) < (REAL)1e-12) \
                            ? _t4*sExpat[_ls4] \
                            : (sExpat[_ls4]-sExpat[_rs4])/(sEvalR[_ls4]-sEvalR[_rs4]); \
                        ADJOINT_ATOMIC_ADD_GPU(&(GRAD)[_ls4*_S4+_rs4], \
                            sRedBuf[0][_ls4*_S4+_rs4]*_co4); \
                        _rs4++; \
                    } else { \
                        const REAL _sr4=sEvalR[_rs4]-sEvalR[_ls4]; \
                        const REAL _dn4=_sr4*_sr4+_ri4*_ri4; \
                        REAL _i04,_i14; \
                        if (_dn4<(REAL)1e-12){_i04=_t4;_i14=(REAL)0;} \
                        else{ \
                            const REAL _ex4=sExpat[_rs4]/sExpat[_ls4]; \
                            _i04=(_ex4*(_sr4*sCosbt[_rs4]+_ri4*sSinbt[_rs4])-_sr4)/_dn4; \
                            _i14=(_ex4*(_sr4*sSinbt[_rs4]-_ri4*sCosbt[_rs4])+_ri4)/_dn4; \
                        } \
                        const REAL _c04=sExpat[_ls4]*_i04, _c14=sExpat[_ls4]*_i14; \
                        const REAL _n04=sRedBuf[0][_ls4*_S4+_rs4]; \
                        const REAL _n14=sRedBuf[0][_ls4*_S4+_rs4+1]; \
                        ADJOINT_ATOMIC_ADD_GPU(&(GRAD)[_ls4*_S4+_rs4],    _c04*_n04+_c14*_n14); \
                        ADJOINT_ATOMIC_ADD_GPU(&(GRAD)[_ls4*_S4+_rs4+1], -_c14*_n04+_c04*_n14); \
                        _rs4 += 2; \
                    } \
                } \
                _ls4++; \
            } else { \
                const REAL _lr4=sEvalR[_ls4], _li24=_li4; \
                const REAL _ec4=sExpatC[_ls4], _es4=sExpatS[_ls4]; \
                const REAL _cI4=sCosbt[_ls4],  _sI4=sSinbt[_ls4]; \
                for (int _rs4 = 0; _rs4 < _S4; ) { \
                    const REAL _ri4 = sEvalI[_rs4]; \
                    if (_ri4 == (REAL)0) { \
                        const REAL _sr4=sEvalR[_rs4]-_lr4; \
                        const REAL _dn4=_sr4*_sr4+_li24*_li24; \
                        REAL _i04,_i14; \
                        if (_dn4<(REAL)1e-12){_i04=_t4;_i14=(REAL)0;} \
                        else{ \
                            const REAL _ex4=sExpat[_rs4]/sExpat[_ls4]; \
                            _i04=(_ex4*(_sr4*_cI4+_li24*_sI4)-_sr4)/_dn4; \
                            _i14=(_ex4*(_sr4*_sI4-_li24*_cI4)+_li24)/_dn4; \
                        } \
                        const REAL _p04=_ec4*_i04+_es4*_i14; \
                        const REAL _p14=_ec4*_i14-_es4*_i04; \
                        const REAL _p24=_es4*_i04-_ec4*_i14; \
                        const REAL _p34=_es4*_i14+_ec4*_i04; \
                        const REAL _n04=sRedBuf[0][_ls4*_S4+_rs4]; \
                        const REAL _n14=sRedBuf[0][(_ls4+1)*_S4+_rs4]; \
                        ADJOINT_ATOMIC_ADD_GPU(&(GRAD)[_ls4*_S4+_rs4],     _p04*_n04+_p14*_n14); \
                        ADJOINT_ATOMIC_ADD_GPU(&(GRAD)[(_ls4+1)*_S4+_rs4], _p24*_n04+_p34*_n14); \
                        _rs4++; \
                    } else { \
                        const REAL _rr4=sEvalR[_rs4], _ri24=_ri4; \
                        const REAL _sr4=_rr4-_lr4; \
                        const REAL _si14=_li24+_ri24, _si24=_ri24-_li24; \
                        const REAL _sr24=_sr4*_sr4; \
                        const REAL _d14=_sr24+_si14*_si14, _d24=_sr24+_si24*_si24; \
                        const REAL _ex4=(_d14>=(REAL)1e-12||_d24>=(REAL)1e-12) \
                            ? sExpat[_rs4]/sExpat[_ls4] : (REAL)0; \
                        const REAL _clcr4=_cI4*sCosbt[_rs4], _slsr4=_sI4*sSinbt[_rs4]; \
                        const REAL _clsr4=_cI4*sSinbt[_rs4], _slcr4=_sI4*sCosbt[_rs4]; \
                        REAL _i1r4,_i1i4; \
                        if (_d14<(REAL)1e-12){_i1r4=_t4;_i1i4=(REAL)0;} \
                        else{ \
                            const REAL _cs14=_clcr4-_slsr4, _sn14=_slcr4+_clsr4; \
                            const REAL _xc14=_ex4*_cs14-(REAL)1, _xs14=_ex4*_sn14; \
                            _i1r4=(_sr4*_xc14+_si14*_xs14)/_d14; \
                            _i1i4=(_sr4*_xs14-_si14*_xc14)/_d14; \
                        } \
                        REAL _i2r4,_i2i4; \
                        if (_d24<(REAL)1e-12){_i2r4=_t4;_i2i4=(REAL)0;} \
                        else{ \
                            const REAL _cs24=_clcr4+_slsr4, _sn24=_clsr4-_slcr4; \
                            const REAL _xc24=_ex4*_cs24-(REAL)1, _xs24=_ex4*_sn24; \
                            _i2r4=(_sr4*_xc24+_si24*_xs24)/_d24; \
                            _i2i4=(_sr4*_xs24-_si24*_xc24)/_d24; \
                        } \
                        const REAL _pr4=_ec4*_i1r4+_es4*_i1i4; \
                        const REAL _pi4=_ec4*_i1i4-_es4*_i1r4; \
                        const REAL _mr4=_ec4*_i2r4-_es4*_i2i4; \
                        const REAL _mi4=_ec4*_i2i4+_es4*_i2r4; \
                        const REAL _A4=(REAL)0.5*(_mr4+_pr4), _B4=(REAL)0.5*(_mi4+_pi4); \
                        const REAL _C4=(REAL)0.5*(_pi4-_mi4), _D4=(REAL)0.5*(_mr4-_pr4); \
                        const REAL _n004=sRedBuf[0][_ls4*_S4+_rs4]; \
                        const REAL _n014=sRedBuf[0][_ls4*_S4+_rs4+1]; \
                        const REAL _n104=sRedBuf[0][(_ls4+1)*_S4+_rs4]; \
                        const REAL _n114=sRedBuf[0][(_ls4+1)*_S4+_rs4+1]; \
                        ADJOINT_ATOMIC_ADD_GPU(&(GRAD)[_ls4*_S4+_rs4],         _A4*_n004+_B4*_n014+_C4*_n104+_D4*_n114); \
                        ADJOINT_ATOMIC_ADD_GPU(&(GRAD)[_ls4*_S4+_rs4+1],      -_B4*_n004+_A4*_n014-_D4*_n104+_C4*_n114); \
                        ADJOINT_ATOMIC_ADD_GPU(&(GRAD)[(_ls4+1)*_S4+_rs4],    -_C4*_n004-_D4*_n014+_A4*_n104+_B4*_n114); \
                        ADJOINT_ATOMIC_ADD_GPU(&(GRAD)[(_ls4+1)*_S4+_rs4+1],   _D4*_n004-_C4*_n014-_B4*_n104+_A4*_n114); \
                        _rs4 += 2; \
                    } \
                } \
                _ls4 += 2; \
            } \
        } \
    }

/* ── Named kernel functions ────────────────────────────────────────────── */

/* ═══════════════════════════════════════════════════════════════════════════
 * Adjoint cross-product kernel — 4-state, MERGED single launch across
 * branches (see STATUS.md/TODO.md — mirrors kernelAdjointMergedN's design
 * for the generic-N path). grid = (categoryCount, branchCount); every
 * branch's buffers are resolved via the same 9-field-per-branch device
 * offset-queue layout used by the N-state merged kernel. `kernelsSpectralIfDef.cu`
 * and `kernelsSpectralIfDef4.cu` are never compiled into the same OpenCL
 * program (one per state count — see `make_opencl_spectral_kernels.sh`), so
 * `ADJOINT_QUEUE_STRIDE` is redefined locally here rather than shared.
 * `isStates`/`isAllReal` are read per-block from the queue and branched on
 * at runtime (block-uniform, no divergence cost) instead of selecting among
 * 4 kernel functions. `ADJOINT4_COMPLEX_SMEM()` is a superset of
 * `ADJOINT4_ALLREAL_SMEM()`, so declaring it unconditionally covers both
 * cases. ═══════════════════════════════════════════════════════════════*/

#define ADJOINT_QUEUE_STRIDE 9

KW_GLOBAL_KERNEL void kernelAdjointMerged4(
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
    const int cat    = KW_GROUP_ID_0;
    const int branch = KW_GROUP_ID_1;
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

    ADJOINT4_COMPLEX_SMEM()

    REAL regOp[ADJOINT_SS4];
    for (int _i = 0; _i < ADJOINT_SS4; _i++) regOp[_i] = (REAL)0;
    const int catOff = cat * totalPatterns * PADDED_STATE_COUNT;

    if (isAllReal) {
        ADJOINT4_LOAD_ALLREAL(evecT, ievc, eigenValues, distances, categoryWeights)
        if (isStates) { ADJOINT4_ACCUM_STATES(prePartials, tipStates, totalPatterns, catOff) }
        else          { ADJOINT4_ACCUM_PARTIALS(prePartials, postPartials, totalPatterns, catOff) }
        ADJOINT4_REDUCE()
        ADJOINT4_APPLY_ALLREAL(dGradient)
    } else {
        ADJOINT4_LOAD_COMPLEX(evecT, ievc, eigenValues, distances, categoryWeights)
        if (isStates) { ADJOINT4_ACCUM_STATES(prePartials, tipStates, totalPatterns, catOff) }
        else          { ADJOINT4_ACCUM_PARTIALS(prePartials, postPartials, totalPatterns, catOff) }
        ADJOINT4_REDUCE()
        ADJOINT4_APPLY_COMPLEX(dGradient)
    }
}
