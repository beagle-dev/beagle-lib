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
 * GPU kernels for spectral (eigendecomposition-based) partial likelihood updates.
 *
 * These kernels compute the Felsenstein partial update
 *
 *     p_dest[j] = (P1 · c1)[j] * (P2 · c2)[j]
 *
 * implicitly via the spectral decomposition P = U diag(exp(Λ t)) U^{-1},
 * avoiding materialisation of the full transition matrices.  Each child c_i
 * is either a partial likelihood vector (Partials) or a discrete tip state
 * (States); the three public variants are PartialsPartials, StatesPartials,
 * and StatesStates.
 *
 * The computation is structured as three phases:
 *   1. matrix-vector multiply:  q = U^{-1} · c
 *      For a discrete state s this collapses to q[k] = U^{-1}[k, s] (one
 *      column of U^{-1}); for an ambiguous tip (s >= PADDED_STATE_COUNT)
 *      c is treated as all-ones and q[k] = Σ_j U^{-1}[k, j].
 *   2. scale/mix eigenvalues:   tmp[k] = expat_k * cos(imag_k * t) * q[k]
 *                                       ± expat_k * sin(imag_k * t) * q[neighbor]
 *      (real eigenvalue: just expat*q; complex conjugate pair: 2×2 rotation)
 *   3. matrix-vector multiply:  result = U · tmp
 *
 * Data layout (matching BeagleGPUImpl::setEigenDecomposition):
 *   dIevc[row * S + col] = (U^{-1})^T [row, col] = U^{-1}[col, row]
 *   dEvec[row * S + col] = U^T[row, col]          = U[col, row]
 *   dEigenValues[k]      = real part of eigenvalue k
 *   dEigenValues[S + k]  = imaginary part (0 for real; +b / -b for conjugate pair)
 *   distances[matrix]    = branchLength * categoryRate[matrix]
 *
 * The block-peel access pattern for phases 1 (Partials) and 3 is identical
 * to that of kernelPartialsPartialsNoScale in kernelsX.cu, which efficiently
 * coalesces global reads of the transposed-stored matrices.
 *
 * Grid:  dim3(ceil(totalPatterns / PATTERN_BLOCK_SIZE), kCategoryCount)
 * Block: dim3(PADDED_STATE_COUNT, PATTERN_BLOCK_SIZE)
 */

#ifdef CUDA

#include <type_traits>
#include "libhmsbeagle/GPU/GPUImplDefs.h"

/* ── FMA helper (fused multiply-add when available) ─────────────────────── */
#if (!defined DOUBLE_PRECISION && defined FP_FAST_FMAF) || \
    ( defined DOUBLE_PRECISION && defined FP_FAST_FMA)
    #define SPECTRAL_FMA(x, y, z) (z = fma(x, y, z))
#else
    #define SPECTRAL_FMA(x, y, z) (z += x * y)
#endif

/* ── Child-type tags ────────────────────────────────────────────────────── */
/* Passed as template arguments to select Phase-1 implementation at compile
 * time.  Zero-overhead: if constexpr eliminates the unused code path entirely
 * for each instantiation. */
struct Partials {};   /* child carries a partial likelihood vector */
struct States   {};   /* child carries a discrete tip state index  */

/* ── Convenience alias ──────────────────────────────────────────────────── */
template <typename A, typename B>
static constexpr bool IsSameType = std::is_same<A, B>::value;

/* ═════════════════════════════════════════════════════════════════════════
 * kernelSpectralBody — shared __device__ implementation
 *
 * Child1, Child2 : Partials or States
 * useScaling     : whether to divide by pre-computed scalingFactors
 *
 * Null-pointer convention (enforced by if constexpr, never dereferenced):
 *   Child1 = States   → partials1 == nullptr
 *   Child1 = Partials → states1   == nullptr
 *   (same for Child2 / partials2 / states2)
 *   useScaling = false → scalingFactors == nullptr
 * ═════════════════════════════════════════════════════════════════════════*/
template <typename Child1, typename Child2, bool useScaling>
__device__ void kernelSpectralBody(
        KW_GLOBAL_VAR REAL* KW_RESTRICT partials1,
        KW_GLOBAL_VAR int*  KW_RESTRICT states1,
        KW_GLOBAL_VAR REAL* KW_RESTRICT partials2,
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

    /* ── Thread / block indices ──────────────────────────────────────────── */
    const int state   = KW_LOCAL_ID_0;
    const int patIdx  = KW_LOCAL_ID_1;
    const int pattern = __umul24(KW_GROUP_ID_0, PATTERN_BLOCK_SIZE) + patIdx;
    const int matrix  = KW_GROUP_ID_1;

    const int deltaPartialsByState  = pattern * PADDED_STATE_COUNT;
    const int deltaPartialsByMatrix = matrix  * PADDED_STATE_COUNT * totalPatterns;
    const int u = state + deltaPartialsByState + deltaPartialsByMatrix;
    const int y = deltaPartialsByState + deltaPartialsByMatrix;

    /* ── Shared memory ───────────────────────────────────────────────────── */
    /* Partial inputs — only populated for Partials children; with
     * if constexpr the compiler eliminates dead stores/loads for States. */
    KW_LOCAL_MEM REAL sP1[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sP2[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];

    /* Matrix peel buffer — reused for ievc (phase 1) and evec (phase 3). */
    KW_LOCAL_MEM REAL sBuf1[BLOCK_PEELING_SIZE][PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sBuf2[BLOCK_PEELING_SIZE][PADDED_STATE_COUNT];

    /* Per-eigenstate scaling factors:
     *   sDs[k] = expat_k * cos(imagEV_k * t)   (always non-negative)
     *   sCs[k] = expat_k * sin(imagEV_k * t)
     *     == 0  → real eigenvalue
     *     >  0  → first  of complex conjugate pair (imagEV > 0)
     *     <  0  → second of complex conjugate pair (imagEV < 0) */
    KW_LOCAL_MEM REAL sDs1[PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sCs1[PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sDs2[PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sCs2[PADDED_STATE_COUNT];

    /* Intermediate eigenspace vectors: q = U^{-1}·c (phase 1 → phase 2),
     * then overwritten with scaled tmp (phase 2 → phase 3). */
    KW_LOCAL_MEM REAL sQ1[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sQ2[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];

    KW_LOCAL_MEM REAL sScale[PATTERN_BLOCK_SIZE];

    /* ── Load input partials into shared memory ───────────────────────────── */
    if constexpr (IsSameType<Child1, Partials>) {
        sP1[patIdx][state] = (pattern < totalPatterns) ? partials1[y + state] : REAL(0);
    }
    if constexpr (IsSameType<Child2, Partials>) {
        sP2[patIdx][state] = (pattern < totalPatterns) ? partials2[y + state] : REAL(0);
    }

    /* ── Load optional scaling denominators ─────────────────────────────── */
    if constexpr (useScaling) {
        if (patIdx == 0 && state < PATTERN_BLOCK_SIZE)
            sScale[state] = scalingFactors[KW_GROUP_ID_0 * PATTERN_BLOCK_SIZE + state];
    }

    /* ── Compute eigenvalue exponentials for this rate category ─────────── */
    /* Only the patIdx==0 row of threads computes (one thread per eigenstate).
     * eigenValues layout: [realEV_0 .. realEV_{S-1} | imagEV_0 .. imagEV_{S-1}] */
    if (patIdx == 0) {
        const REAL t1  = distances1[matrix];
        const REAL e1  = exp(eigenValues1[state] * t1);
        const REAL bt1 = eigenValues1[PADDED_STATE_COUNT + state] * t1;
        sDs1[state] = e1 * cos(bt1);
        sCs1[state] = e1 * sin(bt1);

        const REAL t2  = distances2[matrix];
        const REAL e2  = exp(eigenValues2[state] * t2);
        const REAL bt2 = eigenValues2[PADDED_STATE_COUNT + state] * t2;
        sDs2[state] = e2 * cos(bt2);
        sCs2[state] = e2 * sin(bt2);
    }
    KW_LOCAL_FENCE;  /* sP, sScale, sDs, sCs all visible to all threads */

    /* ── Phase 1: project child into eigenspace (q = U^{-1} · c) ────────── */
    REAL q1 = REAL(0), q2 = REAL(0);

    /* States children: direct column lookup of dIevc.
     * dIevc[row * S + col] = U^{-1}[col, row], so
     * dIevc[s * S + k] = U^{-1}[k, s] = (U^{-1} · e_s)[k].
     * Threads for k = 0..S-1 with fixed s access consecutive addresses → coalesced.
     * Ambiguous tip (s >= S): c = all-ones → q[k] = Σ_j U^{-1}[k, j]. */
    if constexpr (IsSameType<Child1, States>) {
        if (pattern < totalPatterns) {
            const int s1 = states1[pattern];
            if (s1 < PADDED_STATE_COUNT) {
                q1 = ievc1[s1 * PADDED_STATE_COUNT + state];
            } else {
                for (int j = 0; j < PADDED_STATE_COUNT; j++)
                    q1 += ievc1[j * PADDED_STATE_COUNT + state];
            }
        }
        sQ1[patIdx][state] = q1;
    }
    if constexpr (IsSameType<Child2, States>) {
        if (pattern < totalPatterns) {
            const int s2 = states2[pattern];
            if (s2 < PADDED_STATE_COUNT) {
                q2 = ievc2[s2 * PADDED_STATE_COUNT + state];
            } else {
                for (int j = 0; j < PADDED_STATE_COUNT; j++)
                    q2 += ievc2[j * PADDED_STATE_COUNT + state];
            }
        }
        sQ2[patIdx][state] = q2;
    }

    /* Partials children: block-peeled dot product q = Σ_j ievc[j*S+k] * p[j].
     * Loading ievc blocks for both Partials children per peel pass amortises
     * the fence overhead.  The if constexpr guards suppress each load/FMA for
     * whichever child is of type States. */
    if constexpr (IsSameType<Child1, Partials> || IsSameType<Child2, Partials>) {
        for (int i = 0; i < PADDED_STATE_COUNT; i += BLOCK_PEELING_SIZE) {
            if (patIdx < BLOCK_PEELING_SIZE) {
                if constexpr (IsSameType<Child1, Partials>)
                    sBuf1[patIdx][state] = ievc1[(i + patIdx) * PADDED_STATE_COUNT + state];
                if constexpr (IsSameType<Child2, Partials>)
                    sBuf2[patIdx][state] = ievc2[(i + patIdx) * PADDED_STATE_COUNT + state];
            }
            KW_LOCAL_FENCE;
            for (int j = 0; j < BLOCK_PEELING_SIZE; j++) {
                if constexpr (IsSameType<Child1, Partials>)
                    SPECTRAL_FMA(sBuf1[j][state], sP1[patIdx][i + j], q1);
                if constexpr (IsSameType<Child2, Partials>)
                    SPECTRAL_FMA(sBuf2[j][state], sP2[patIdx][i + j], q2);
            }
            KW_LOCAL_FENCE;
        }
        if constexpr (IsSameType<Child1, Partials>) sQ1[patIdx][state] = q1;
        if constexpr (IsSameType<Child2, Partials>) sQ2[patIdx][state] = q2;
    }
    KW_LOCAL_FENCE;  /* Phase 1 complete: sQ1, sQ2 visible for Phase 2 */

    /* ── Phase 2: eigenvalue scaling / complex conjugate pair mixing ──────── */
    /* Unified formula: tmp[k] = sDs[k]*q[k] + sCs[k]*q[neighbor(k)]
     * where neighbor = k+1 (first of pair, sCs>0), k-1 (second, sCs<0),
     * or doesn't matter (real, sCs==0 so the sCs term vanishes).
     * All threads read sQ before any thread overwrites it; the fence above
     * guarantees this ordering. */
    {
        const REAL ecos1 = sDs1[state], esin1 = sCs1[state];
        sQ1[patIdx][state] = (esin1 == REAL(0))
            ? ecos1 * sQ1[patIdx][state]
            : ecos1 * sQ1[patIdx][state]
              + esin1 * sQ1[patIdx][state + (esin1 > REAL(0) ? 1 : -1)];

        const REAL ecos2 = sDs2[state], esin2 = sCs2[state];
        sQ2[patIdx][state] = (esin2 == REAL(0))
            ? ecos2 * sQ2[patIdx][state]
            : ecos2 * sQ2[patIdx][state]
              + esin2 * sQ2[patIdx][state + (esin2 > REAL(0) ? 1 : -1)];
    }
    KW_LOCAL_FENCE;  /* Phase 2 complete: scaled sQ1, sQ2 visible for Phase 3 */

    /* ── Phase 3: project back to state space (result = U · tmp) ─────────── */
    /* Same block-peel as Phase 1; sBuf is reused for evec blocks.
     * result[state] = Σ_k evec[k*S+state] * sQ[patIdx][k] = (U·tmp)[state]. */
    REAL sum1 = REAL(0), sum2 = REAL(0);
    for (int i = 0; i < PADDED_STATE_COUNT; i += BLOCK_PEELING_SIZE) {
        if (patIdx < BLOCK_PEELING_SIZE) {
            sBuf1[patIdx][state] = evec1[(i + patIdx) * PADDED_STATE_COUNT + state];
            sBuf2[patIdx][state] = evec2[(i + patIdx) * PADDED_STATE_COUNT + state];
        }
        KW_LOCAL_FENCE;
        for (int j = 0; j < BLOCK_PEELING_SIZE; j++) {
            SPECTRAL_FMA(sBuf1[j][state], sQ1[patIdx][i + j], sum1);
            SPECTRAL_FMA(sBuf2[j][state], sQ2[patIdx][i + j], sum2);
        }
        KW_LOCAL_FENCE;
    }

    /* ── Write output (Hadamard product of the two branch contributions) ─── */
    if (pattern < totalPatterns) {
        if constexpr (useScaling)
            partials3[u] = sum1 * sum2 / sScale[patIdx];
        else
            partials3[u] = sum1 * sum2;
    }
}

/* ═════════════════════════════════════════════════════════════════════════
 * kernelSpectralGeneric — single public template kernel covering all cases.
 *
 * Explicit template instantiations:
 *   kernelSpectralGeneric<Partials, Partials, false>  ←→ PartialsPartials, no scale
 *   kernelSpectralGeneric<Partials, Partials, true>   ←→ PartialsPartials, fixed scale
 *   kernelSpectralGeneric<States,   Partials, false>  ←→ StatesPartials,   no scale
 *   kernelSpectralGeneric<States,   Partials, true>   ←→ StatesPartials,   fixed scale
 *   kernelSpectralGeneric<States,   States,   false>  ←→ StatesStates,     no scale
 *   kernelSpectralGeneric<States,   States,   true>   ←→ StatesStates,     fixed scale
 *
 * Template linkage is C++; this kernel is not accessible via GetFunction()
 * string lookup.  Use the named extern "C" wrappers below for that purpose.
 *
 * Null-pointer convention: pass nullptr for the pointer corresponding to the
 * unused child type (partials* for States children, states* for Partials
 * children, scalingFactors for useScaling=false).
 * ═════════════════════════════════════════════════════════════════════════*/
template <typename Child1, typename Child2, bool useScaling>
KW_GLOBAL_KERNEL void kernelSpectralGeneric(
        KW_GLOBAL_VAR REAL* KW_RESTRICT partials1,
        KW_GLOBAL_VAR int*  KW_RESTRICT states1,
        KW_GLOBAL_VAR REAL* KW_RESTRICT partials2,
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
    kernelSpectralBody<Child1, Child2, useScaling>(
        partials1, states1, partials2, states2, partials3,
        ievc1, evec1, eigenValues1, distances1,
        ievc2, evec2, eigenValues2, distances2,
        scalingFactors, totalPatterns);
}

/* ═════════════════════════════════════════════════════════════════════════
 * Named extern "C" kernels — accessible via GetFunction() string lookup.
 *
 * Each wrapper has a type-specific parameter list (no spurious null-pointer
 * arguments) and delegates to kernelSpectralBody with the appropriate
 * template instantiation.
 * ═════════════════════════════════════════════════════════════════════════*/
extern "C" {

/* ── PartialsPartials ──────────────────────────────────────────────────── */

KW_GLOBAL_KERNEL void kernelPartialsPartialsSpectralNoScale(
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
    kernelSpectralBody<Partials, Partials, false>(
        partials1, nullptr, partials2, nullptr, partials3,
        ievc1, evec1, eigenValues1, distances1,
        ievc2, evec2, eigenValues2, distances2,
        nullptr, totalPatterns);
}

KW_GLOBAL_KERNEL void kernelPartialsPartialsSpectralFixedScale(
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
    kernelSpectralBody<Partials, Partials, true>(
        partials1, nullptr, partials2, nullptr, partials3,
        ievc1, evec1, eigenValues1, distances1,
        ievc2, evec2, eigenValues2, distances2,
        scalingFactors, totalPatterns);
}

/* ── StatesPartials ────────────────────────────────────────────────────── */
/* Convention: states (tip) is child 1, partials is child 2.
 * The caller swaps the arguments when the partials child is actually child 1. */

KW_GLOBAL_KERNEL void kernelStatesPartialsSpectralNoScale(
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
    kernelSpectralBody<States, Partials, false>(
        nullptr, states1, partials2, nullptr, partials3,
        ievc1, evec1, eigenValues1, distances1,
        ievc2, evec2, eigenValues2, distances2,
        nullptr, totalPatterns);
}

KW_GLOBAL_KERNEL void kernelStatesPartialsSpectralFixedScale(
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
    kernelSpectralBody<States, Partials, true>(
        nullptr, states1, partials2, nullptr, partials3,
        ievc1, evec1, eigenValues1, distances1,
        ievc2, evec2, eigenValues2, distances2,
        scalingFactors, totalPatterns);
}

/* ── StatesStates ──────────────────────────────────────────────────────── */

KW_GLOBAL_KERNEL void kernelStatesStatesSpectralNoScale(
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
    kernelSpectralBody<States, States, false>(
        nullptr, states1, nullptr, states2, partials3,
        ievc1, evec1, eigenValues1, distances1,
        ievc2, evec2, eigenValues2, distances2,
        nullptr, totalPatterns);
}

KW_GLOBAL_KERNEL void kernelStatesStatesSpectralFixedScale(
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
    kernelSpectralBody<States, States, true>(
        nullptr, states1, nullptr, states2, partials3,
        ievc1, evec1, eigenValues1, distances1,
        ievc2, evec2, eigenValues2, distances2,
        scalingFactors, totalPatterns);
}

} // extern "C"

#endif // CUDA
