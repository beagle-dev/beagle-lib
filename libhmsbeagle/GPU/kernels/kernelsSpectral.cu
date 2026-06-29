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

/* ── PartialsPartials / auto-scaling ───────────────────────────────────── */
/* Auto-scales the output: detects per-pattern overflow/underflow via frexp,
 * finds the max exponent across states using sQ1 as scratch (free after
 * Phase 3), rescales partials3, and writes the exponent to scalingFactors. */

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

    const int state   = KW_LOCAL_ID_0;
    const int patIdx  = KW_LOCAL_ID_1;
    const int pattern = __umul24(KW_GROUP_ID_0, PATTERN_BLOCK_SIZE) + patIdx;
    const int matrix  = KW_GROUP_ID_1;

    const int deltaPartialsByState  = pattern * PADDED_STATE_COUNT;
    const int deltaPartialsByMatrix = matrix  * PADDED_STATE_COUNT * totalPatterns;
    const int u = state + deltaPartialsByState + deltaPartialsByMatrix;
    const int y = deltaPartialsByState + deltaPartialsByMatrix;

    KW_LOCAL_MEM REAL sP1[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sP2[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sBuf1[BLOCK_PEELING_SIZE][PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sBuf2[BLOCK_PEELING_SIZE][PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sDs1[PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sCs1[PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sDs2[PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sCs2[PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sQ1[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sQ2[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];

    sP1[patIdx][state] = (pattern < totalPatterns) ? partials1[y + state] : REAL(0);
    sP2[patIdx][state] = (pattern < totalPatterns) ? partials2[y + state] : REAL(0);

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
    KW_LOCAL_FENCE;

    REAL q1 = REAL(0), q2 = REAL(0);
    for (int i = 0; i < PADDED_STATE_COUNT; i += BLOCK_PEELING_SIZE) {
        if (patIdx < BLOCK_PEELING_SIZE) {
            sBuf1[patIdx][state] = ievc1[(i + patIdx) * PADDED_STATE_COUNT + state];
            sBuf2[patIdx][state] = ievc2[(i + patIdx) * PADDED_STATE_COUNT + state];
        }
        KW_LOCAL_FENCE;
        for (int j = 0; j < BLOCK_PEELING_SIZE; j++) {
            SPECTRAL_FMA(sBuf1[j][state], sP1[patIdx][i + j], q1);
            SPECTRAL_FMA(sBuf2[j][state], sP2[patIdx][i + j], q2);
        }
        KW_LOCAL_FENCE;
    }
    sQ1[patIdx][state] = q1;
    sQ2[patIdx][state] = q2;
    KW_LOCAL_FENCE;

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
    KW_LOCAL_FENCE;

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

    /* Auto-scale output.  sQ1 is reused as scratch for the per-pattern
     * max-exponent reduction; thread 0 of each pattern row does a linear
     * scan, keeping the logic correct for any PADDED_STATE_COUNT. */
    REAL tmpPartial = sum1 * sum2;
    int  expTmp;
    REAL sigTmp = frexp(tmpPartial, &expTmp);

    sQ1[patIdx][state] = REAL(
        (pattern < totalPatterns && abs(expTmp) > SCALING_EXPONENT_THRESHOLD)
        ? expTmp : 0);
    KW_LOCAL_FENCE;

    if (state == 0) {
        REAL maxVal = sQ1[patIdx][0];
        for (int i = 1; i < PADDED_STATE_COUNT; i++)
            if (sQ1[patIdx][i] > maxVal) maxVal = sQ1[patIdx][i];
        sQ1[patIdx][0] = maxVal;
    }
    KW_LOCAL_FENCE;

    const int maxExp = (int)sQ1[patIdx][0];
    if (pattern < totalPatterns)
        partials3[u] = (maxExp != 0) ? ldexp(sigTmp, expTmp - maxExp) : tmpPartial;
    if (state == 0 && pattern < totalPatterns)
        scalingFactors[matrix * totalPatterns + pattern] = (signed char)maxExp;
}

/* ── Growing (pre-order) kernels — generic block-peel (N≥16) ──────────────
 *
 * Same semantics as kernelsSpectral4.cu Growing kernels, but uses
 * BLOCK_PEELING_SIZE as the peel stride (instead of PADDED_STATE_COUNT).
 * Shared-memory peel buffers are [BLOCK_PEELING_SIZE][PADDED_STATE_COUNT].
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
    const int state = KW_LOCAL_ID_0, patIdx = KW_LOCAL_ID_1;
    const int pattern = __umul24(KW_GROUP_ID_0, PATTERN_BLOCK_SIZE) + patIdx;
    const int matrix  = KW_GROUP_ID_1;
    const int y = pattern * PADDED_STATE_COUNT + matrix * PADDED_STATE_COUNT * totalPatterns;
    const int u = state + y;

    KW_LOCAL_MEM REAL sP1[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sP2[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sBuf1[BLOCK_PEELING_SIZE][PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sBuf2[BLOCK_PEELING_SIZE][PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sDs1[PADDED_STATE_COUNT], sCs1[PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sDs2[PADDED_STATE_COUNT], sCs2[PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sQ1[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sQ2[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];

    sP1[patIdx][state] = (pattern < totalPatterns) ? partials1[y + state] : REAL(0);
    sP2[patIdx][state] = (pattern < totalPatterns) ? partials2[y + state] : REAL(0);
    if (patIdx == 0) {
        const REAL t1 = distances1[matrix];
        sDs1[state] = exp(eigenValues1[state] * t1) * cos(eigenValues1[PADDED_STATE_COUNT + state] * t1);
        sCs1[state] = exp(eigenValues1[state] * t1) * sin(eigenValues1[PADDED_STATE_COUNT + state] * t1);
        const REAL t2 = distances2[matrix];
        sDs2[state] = exp(eigenValues2[state] * t2) * cos(eigenValues2[PADDED_STATE_COUNT + state] * t2);
        sCs2[state] = exp(eigenValues2[state] * t2) * sin(eigenValues2[PADDED_STATE_COUNT + state] * t2);
    }
    KW_LOCAL_FENCE;

    /* Sibling Phase 1: sQ2 = ievc2 · sP2 (block-peel) */
    { REAL q = REAL(0);
      for (int i = 0; i < PADDED_STATE_COUNT; i += BLOCK_PEELING_SIZE) {
          if (patIdx < BLOCK_PEELING_SIZE) sBuf2[patIdx][state] = ievc2[(i + patIdx) * PADDED_STATE_COUNT + state];
          KW_LOCAL_FENCE;
          for (int j = 0; j < BLOCK_PEELING_SIZE; j++) SPECTRAL_FMA(sBuf2[j][state], sP2[patIdx][i + j], q);
          KW_LOCAL_FENCE;
      }
      sQ2[patIdx][state] = q; }
    KW_LOCAL_FENCE;

    /* Sibling Phase 2 */
    { const REAL ec = sDs2[state], es = sCs2[state];
      sQ2[patIdx][state] = (es == REAL(0)) ? ec * sQ2[patIdx][state]
          : ec * sQ2[patIdx][state] + es * sQ2[patIdx][state + (es > REAL(0) ? 1 : -1)]; }
    KW_LOCAL_FENCE;

    /* Sibling Phase 3 + Hadamard: evec2·sQ2 ⊙ sP1 → sQ2 (combined) */
    { REAL tmp = REAL(0);
      for (int i = 0; i < PADDED_STATE_COUNT; i += BLOCK_PEELING_SIZE) {
          if (patIdx < BLOCK_PEELING_SIZE) sBuf2[patIdx][state] = evec2[(i + patIdx) * PADDED_STATE_COUNT + state];
          KW_LOCAL_FENCE;
          for (int j = 0; j < BLOCK_PEELING_SIZE; j++) SPECTRAL_FMA(sBuf2[j][state], sQ2[patIdx][i + j], tmp);
          KW_LOCAL_FENCE;
      }
      sQ2[patIdx][state] = tmp * sP1[patIdx][state]; }
    KW_LOCAL_FENCE;

    /* Parent Phase 1 (backward): sQ1 = ievc1 · sQ2 (ievc1 = dEvecT = U) */
    { REAL q = REAL(0);
      for (int i = 0; i < PADDED_STATE_COUNT; i += BLOCK_PEELING_SIZE) {
          if (patIdx < BLOCK_PEELING_SIZE) sBuf1[patIdx][state] = ievc1[(i + patIdx) * PADDED_STATE_COUNT + state];
          KW_LOCAL_FENCE;
          for (int j = 0; j < BLOCK_PEELING_SIZE; j++) SPECTRAL_FMA(sBuf1[j][state], sQ2[patIdx][i + j], q);
          KW_LOCAL_FENCE;
      }
      sQ1[patIdx][state] = q; }
    KW_LOCAL_FENCE;

    /* Parent Phase 2 */
    { const REAL ec = sDs1[state], es = sCs1[state];
      sQ1[patIdx][state] = (es == REAL(0)) ? ec * sQ1[patIdx][state]
          : ec * sQ1[patIdx][state] + es * sQ1[patIdx][state + (es > REAL(0) ? 1 : -1)]; }
    KW_LOCAL_FENCE;

    /* Parent Phase 3: evec1·sQ1 → result (evec1 = dIevcT = U^-1) */
    { REAL sum = REAL(0);
      for (int i = 0; i < PADDED_STATE_COUNT; i += BLOCK_PEELING_SIZE) {
          if (patIdx < BLOCK_PEELING_SIZE) sBuf1[patIdx][state] = evec1[(i + patIdx) * PADDED_STATE_COUNT + state];
          KW_LOCAL_FENCE;
          for (int j = 0; j < BLOCK_PEELING_SIZE; j++) SPECTRAL_FMA(sBuf1[j][state], sQ1[patIdx][i + j], sum);
          KW_LOCAL_FENCE;
      }
      if (pattern < totalPatterns) partials3[u] = sum; }
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
    const int state = KW_LOCAL_ID_0, patIdx = KW_LOCAL_ID_1;
    const int pattern = __umul24(KW_GROUP_ID_0, PATTERN_BLOCK_SIZE) + patIdx;
    const int matrix  = KW_GROUP_ID_1;
    const int y = pattern * PADDED_STATE_COUNT + matrix * PADDED_STATE_COUNT * totalPatterns;
    const int u = state + y;

    KW_LOCAL_MEM REAL sP1[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sBuf1[BLOCK_PEELING_SIZE][PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sBuf2[BLOCK_PEELING_SIZE][PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sDs1[PADDED_STATE_COUNT], sCs1[PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sDs2[PADDED_STATE_COUNT], sCs2[PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sQ1[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sQ2[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];

    sP1[patIdx][state] = (pattern < totalPatterns) ? partials1[y + state] : REAL(0);
    if (patIdx == 0) {
        const REAL t1 = distances1[matrix];
        sDs1[state] = exp(eigenValues1[state] * t1) * cos(eigenValues1[PADDED_STATE_COUNT + state] * t1);
        sCs1[state] = exp(eigenValues1[state] * t1) * sin(eigenValues1[PADDED_STATE_COUNT + state] * t1);
        const REAL t2 = distances2[matrix];
        sDs2[state] = exp(eigenValues2[state] * t2) * cos(eigenValues2[PADDED_STATE_COUNT + state] * t2);
        sCs2[state] = exp(eigenValues2[state] * t2) * sin(eigenValues2[PADDED_STATE_COUNT + state] * t2);
    }
    KW_LOCAL_FENCE;

    /* Sibling Phase 1 (states lookup) */
    { REAL q = REAL(0);
      if (pattern < totalPatterns) {
          const int s = states2[pattern];
          if (s < PADDED_STATE_COUNT) q = ievc2[s * PADDED_STATE_COUNT + state];
          else for (int j = 0; j < PADDED_STATE_COUNT; j++) q += ievc2[j * PADDED_STATE_COUNT + state];
      }
      sQ2[patIdx][state] = q; }
    KW_LOCAL_FENCE;

    /* Sibling Phase 2 */
    { const REAL ec = sDs2[state], es = sCs2[state];
      sQ2[patIdx][state] = (es == REAL(0)) ? ec * sQ2[patIdx][state]
          : ec * sQ2[patIdx][state] + es * sQ2[patIdx][state + (es > REAL(0) ? 1 : -1)]; }
    KW_LOCAL_FENCE;

    /* Sibling Phase 3 + Hadamard: evec2·sQ2 ⊙ sP1 → sQ2 (combined) */
    { REAL tmp = REAL(0);
      for (int i = 0; i < PADDED_STATE_COUNT; i += BLOCK_PEELING_SIZE) {
          if (patIdx < BLOCK_PEELING_SIZE) sBuf2[patIdx][state] = evec2[(i + patIdx) * PADDED_STATE_COUNT + state];
          KW_LOCAL_FENCE;
          for (int j = 0; j < BLOCK_PEELING_SIZE; j++) SPECTRAL_FMA(sBuf2[j][state], sQ2[patIdx][i + j], tmp);
          KW_LOCAL_FENCE;
      }
      sQ2[patIdx][state] = tmp * sP1[patIdx][state]; }
    KW_LOCAL_FENCE;

    /* Parent Phase 1 (backward): sQ1 = ievc1 · sQ2 */
    { REAL q = REAL(0);
      for (int i = 0; i < PADDED_STATE_COUNT; i += BLOCK_PEELING_SIZE) {
          if (patIdx < BLOCK_PEELING_SIZE) sBuf1[patIdx][state] = ievc1[(i + patIdx) * PADDED_STATE_COUNT + state];
          KW_LOCAL_FENCE;
          for (int j = 0; j < BLOCK_PEELING_SIZE; j++) SPECTRAL_FMA(sBuf1[j][state], sQ2[patIdx][i + j], q);
          KW_LOCAL_FENCE;
      }
      sQ1[patIdx][state] = q; }
    KW_LOCAL_FENCE;

    /* Parent Phase 2 */
    { const REAL ec = sDs1[state], es = sCs1[state];
      sQ1[patIdx][state] = (es == REAL(0)) ? ec * sQ1[patIdx][state]
          : ec * sQ1[patIdx][state] + es * sQ1[patIdx][state + (es > REAL(0) ? 1 : -1)]; }
    KW_LOCAL_FENCE;

    /* Parent Phase 3: evec1·sQ1 → result */
    { REAL sum = REAL(0);
      for (int i = 0; i < PADDED_STATE_COUNT; i += BLOCK_PEELING_SIZE) {
          if (patIdx < BLOCK_PEELING_SIZE) sBuf1[patIdx][state] = evec1[(i + patIdx) * PADDED_STATE_COUNT + state];
          KW_LOCAL_FENCE;
          for (int j = 0; j < BLOCK_PEELING_SIZE; j++) SPECTRAL_FMA(sBuf1[j][state], sQ1[patIdx][i + j], sum);
          KW_LOCAL_FENCE;
      }
      if (pattern < totalPatterns) partials3[u] = sum; }
}

/* Top NotRoot PS: parent=partials(backward), sibling=states(forward). */
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
    const int state = KW_LOCAL_ID_0, patIdx = KW_LOCAL_ID_1;
    const int pattern = __umul24(KW_GROUP_ID_0, PATTERN_BLOCK_SIZE) + patIdx;
    const int matrix  = KW_GROUP_ID_1;
    const int y = pattern * PADDED_STATE_COUNT + matrix * PADDED_STATE_COUNT * totalPatterns;
    const int u = state + y;

    KW_LOCAL_MEM REAL sP1[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sBuf1[BLOCK_PEELING_SIZE][PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sBuf2[BLOCK_PEELING_SIZE][PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sDs1[PADDED_STATE_COUNT], sCs1[PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sDs2[PADDED_STATE_COUNT], sCs2[PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sQ1[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sQ2[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];

    sP1[patIdx][state] = (pattern < totalPatterns) ? partials1[y + state] : REAL(0);
    if (patIdx == 0) {
        const REAL t1 = distances1[matrix];
        sDs1[state] = exp(eigenValues1[state] * t1) * cos(eigenValues1[PADDED_STATE_COUNT + state] * t1);
        sCs1[state] = exp(eigenValues1[state] * t1) * sin(eigenValues1[PADDED_STATE_COUNT + state] * t1);
        const REAL t2 = distances2[matrix];
        sDs2[state] = exp(eigenValues2[state] * t2) * cos(eigenValues2[PADDED_STATE_COUNT + state] * t2);
        sCs2[state] = exp(eigenValues2[state] * t2) * sin(eigenValues2[PADDED_STATE_COUNT + state] * t2);
    }
    KW_LOCAL_FENCE;

    /* Parent Phase 1 (backward): sQ1 = ievc1 · sP1 */
    { REAL q = REAL(0);
      for (int i = 0; i < PADDED_STATE_COUNT; i += BLOCK_PEELING_SIZE) {
          if (patIdx < BLOCK_PEELING_SIZE) sBuf1[patIdx][state] = ievc1[(i + patIdx) * PADDED_STATE_COUNT + state];
          KW_LOCAL_FENCE;
          for (int j = 0; j < BLOCK_PEELING_SIZE; j++) SPECTRAL_FMA(sBuf1[j][state], sP1[patIdx][i + j], q);
          KW_LOCAL_FENCE;
      }
      sQ1[patIdx][state] = q; }

    /* Sibling Phase 1 (states lookup) */
    { REAL q = REAL(0);
      if (pattern < totalPatterns) {
          const int s = states2[pattern];
          if (s < PADDED_STATE_COUNT) q = ievc2[s * PADDED_STATE_COUNT + state];
          else for (int j = 0; j < PADDED_STATE_COUNT; j++) q += ievc2[j * PADDED_STATE_COUNT + state];
      }
      sQ2[patIdx][state] = q; }
    KW_LOCAL_FENCE;

    /* Phase 2: scale both sQ1 and sQ2 */
    { const REAL ec1 = sDs1[state], es1 = sCs1[state];
      sQ1[patIdx][state] = (es1 == REAL(0)) ? ec1 * sQ1[patIdx][state]
          : ec1 * sQ1[patIdx][state] + es1 * sQ1[patIdx][state + (es1 > REAL(0) ? 1 : -1)];
      const REAL ec2 = sDs2[state], es2 = sCs2[state];
      sQ2[patIdx][state] = (es2 == REAL(0)) ? ec2 * sQ2[patIdx][state]
          : ec2 * sQ2[patIdx][state] + es2 * sQ2[patIdx][state + (es2 > REAL(0) ? 1 : -1)]; }
    KW_LOCAL_FENCE;

    /* Phase 3: evec1·sQ1 → sum1, evec2·sQ2 → sum2 */
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
    if (pattern < totalPatterns) partials3[u] = sum1 * sum2;
}

/* Top Root PP: only sibling forward; root pre-order is Hadamard factor. */
KW_GLOBAL_KERNEL void kernelPartialsPartialsGrowingTopRootSpectral(
        KW_GLOBAL_VAR REAL* KW_RESTRICT partials1,
        KW_GLOBAL_VAR REAL* KW_RESTRICT partials2,
        KW_GLOBAL_VAR REAL* KW_RESTRICT partials3,
        KW_GLOBAL_VAR REAL* KW_RESTRICT ievc2,
        KW_GLOBAL_VAR REAL* KW_RESTRICT evec2,
        KW_GLOBAL_VAR REAL* KW_RESTRICT eigenValues2,
        KW_GLOBAL_VAR REAL* KW_RESTRICT distances2,
        int totalPatterns) {
    const int state = KW_LOCAL_ID_0, patIdx = KW_LOCAL_ID_1;
    const int pattern = __umul24(KW_GROUP_ID_0, PATTERN_BLOCK_SIZE) + patIdx;
    const int matrix  = KW_GROUP_ID_1;
    const int y = pattern * PADDED_STATE_COUNT + matrix * PADDED_STATE_COUNT * totalPatterns;
    const int u = state + y;

    KW_LOCAL_MEM REAL sP1[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sP2[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sBuf2[BLOCK_PEELING_SIZE][PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sDs2[PADDED_STATE_COUNT], sCs2[PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sQ2[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];

    sP1[patIdx][state] = (pattern < totalPatterns) ? partials1[y + state] : REAL(0);
    sP2[patIdx][state] = (pattern < totalPatterns) ? partials2[y + state] : REAL(0);
    if (patIdx == 0) {
        const REAL t2 = distances2[matrix];
        sDs2[state] = exp(eigenValues2[state] * t2) * cos(eigenValues2[PADDED_STATE_COUNT + state] * t2);
        sCs2[state] = exp(eigenValues2[state] * t2) * sin(eigenValues2[PADDED_STATE_COUNT + state] * t2);
    }
    KW_LOCAL_FENCE;

    /* Sibling Phase 1: sQ2 = ievc2 · sP2 */
    { REAL q = REAL(0);
      for (int i = 0; i < PADDED_STATE_COUNT; i += BLOCK_PEELING_SIZE) {
          if (patIdx < BLOCK_PEELING_SIZE) sBuf2[patIdx][state] = ievc2[(i + patIdx) * PADDED_STATE_COUNT + state];
          KW_LOCAL_FENCE;
          for (int j = 0; j < BLOCK_PEELING_SIZE; j++) SPECTRAL_FMA(sBuf2[j][state], sP2[patIdx][i + j], q);
          KW_LOCAL_FENCE;
      }
      sQ2[patIdx][state] = q; }
    KW_LOCAL_FENCE;

    /* Sibling Phase 2 */
    { const REAL ec = sDs2[state], es = sCs2[state];
      sQ2[patIdx][state] = (es == REAL(0)) ? ec * sQ2[patIdx][state]
          : ec * sQ2[patIdx][state] + es * sQ2[patIdx][state + (es > REAL(0) ? 1 : -1)]; }
    KW_LOCAL_FENCE;

    /* Sibling Phase 3 + Hadamard + write */
    { REAL tmp = REAL(0);
      for (int i = 0; i < PADDED_STATE_COUNT; i += BLOCK_PEELING_SIZE) {
          if (patIdx < BLOCK_PEELING_SIZE) sBuf2[patIdx][state] = evec2[(i + patIdx) * PADDED_STATE_COUNT + state];
          KW_LOCAL_FENCE;
          for (int j = 0; j < BLOCK_PEELING_SIZE; j++) SPECTRAL_FMA(sBuf2[j][state], sQ2[patIdx][i + j], tmp);
          KW_LOCAL_FENCE;
      }
      if (pattern < totalPatterns) partials3[u] = tmp * sP1[patIdx][state]; }
}

/* Top Root PS: sibling=states, root pre-order is Hadamard factor. */
KW_GLOBAL_KERNEL void kernelPartialsStatesGrowingTopRootSpectral(
        KW_GLOBAL_VAR REAL* KW_RESTRICT partials1,
        KW_GLOBAL_VAR int*  KW_RESTRICT states2,
        KW_GLOBAL_VAR REAL* KW_RESTRICT partials3,
        KW_GLOBAL_VAR REAL* KW_RESTRICT ievc2,
        KW_GLOBAL_VAR REAL* KW_RESTRICT evec2,
        KW_GLOBAL_VAR REAL* KW_RESTRICT eigenValues2,
        KW_GLOBAL_VAR REAL* KW_RESTRICT distances2,
        int totalPatterns) {
    const int state = KW_LOCAL_ID_0, patIdx = KW_LOCAL_ID_1;
    const int pattern = __umul24(KW_GROUP_ID_0, PATTERN_BLOCK_SIZE) + patIdx;
    const int matrix  = KW_GROUP_ID_1;
    const int y = pattern * PADDED_STATE_COUNT + matrix * PADDED_STATE_COUNT * totalPatterns;
    const int u = state + y;

    KW_LOCAL_MEM REAL sP1[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sBuf2[BLOCK_PEELING_SIZE][PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sDs2[PADDED_STATE_COUNT], sCs2[PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sQ2[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];

    sP1[patIdx][state] = (pattern < totalPatterns) ? partials1[y + state] : REAL(0);
    if (patIdx == 0) {
        const REAL t2 = distances2[matrix];
        sDs2[state] = exp(eigenValues2[state] * t2) * cos(eigenValues2[PADDED_STATE_COUNT + state] * t2);
        sCs2[state] = exp(eigenValues2[state] * t2) * sin(eigenValues2[PADDED_STATE_COUNT + state] * t2);
    }
    KW_LOCAL_FENCE;

    /* Sibling Phase 1 (states lookup) */
    { REAL q = REAL(0);
      if (pattern < totalPatterns) {
          const int s = states2[pattern];
          if (s < PADDED_STATE_COUNT) q = ievc2[s * PADDED_STATE_COUNT + state];
          else for (int j = 0; j < PADDED_STATE_COUNT; j++) q += ievc2[j * PADDED_STATE_COUNT + state];
      }
      sQ2[patIdx][state] = q; }
    KW_LOCAL_FENCE;

    /* Sibling Phase 2 */
    { const REAL ec = sDs2[state], es = sCs2[state];
      sQ2[patIdx][state] = (es == REAL(0)) ? ec * sQ2[patIdx][state]
          : ec * sQ2[patIdx][state] + es * sQ2[patIdx][state + (es > REAL(0) ? 1 : -1)]; }
    KW_LOCAL_FENCE;

    /* Sibling Phase 3 + Hadamard + write */
    { REAL tmp = REAL(0);
      for (int i = 0; i < PADDED_STATE_COUNT; i += BLOCK_PEELING_SIZE) {
          if (patIdx < BLOCK_PEELING_SIZE) sBuf2[patIdx][state] = evec2[(i + patIdx) * PADDED_STATE_COUNT + state];
          KW_LOCAL_FENCE;
          for (int j = 0; j < BLOCK_PEELING_SIZE; j++) SPECTRAL_FMA(sBuf2[j][state], sQ2[patIdx][i + j], tmp);
          KW_LOCAL_FENCE;
      }
      if (pattern < totalPatterns) partials3[u] = tmp * sP1[patIdx][state]; }
}

} // extern "C"

/* ══════════════════════════════════════════════════════════════════════════
 * Adjoint cross-product kernels (generic N-state, N ≥ 16)
 *
 * Two-phase design to avoid S²-sized shared memory for large S:
 *
 * Phase 1 — kernelAdjointPhase1*:
 *   Grid:  dim3(PADDED_STATE_COUNT, kCategoryCount)  — one block per (ls, cat)
 *   Block: dim3(ADJOINT_BLOCK_SP_N)
 *   Each block computes OP[ls][rs] for rs=0..S-1 via block-peeling and
 *   writes to dOpBuf[cat*S*S + ls*S + rs].  Uses warp shuffle reduction.
 *
 * Phase 2 — kernelAdjointPhase2*:
 *   Grid:  dim3(kCategoryCount)  — one block per category
 *   Block: dim3(PADDED_STATE_COUNT)  — one thread per ls
 *   Each thread ls reads OP[ls][0..S-1] from dOpBuf, applies the integral
 *   transform for row ls (including cross-row reads for complex 2×1/2×2 blocks),
 *   and atomicAdds to dGradient[S*S].
 *
 * Both phases are launched once per branch.  dGradient is accumulated across
 * all branch/category calls via atomicAdd.
 * ══════════════════════════════════════════════════════════════════════════*/

#define ADJOINT_BLOCK_SP_N  128
#define ADJOINT_NWARPS_SP_N (ADJOINT_BLOCK_SP_N / 32)

/* ── Phase 1 body — shared between Partials and States variants ─────────── */
/*
 * Design: one block per (ls, category).  Each thread strides over patterns.
 * sEvecTCol holds column ls of evecT (= U row-major; evecT[j*S+ls] = U[j,ls])
 * so every thread can compute lhsLs = Σ_j U[j,ls]*pre[k,j] = (U^T·pre)[ls].
 * sIevcBuf holds BLOCK_PEELING_SIZE rows of ievc at a time; threads
 * cooperate to load it, then each reads its own post[k] to extend regOp.
 */
template <typename Child>
__device__ void adjointPhase1Body(
        KW_GLOBAL_VAR REAL* KW_RESTRICT prePartials,
        KW_GLOBAL_VAR REAL* KW_RESTRICT postPartials,
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
    KW_LOCAL_MEM REAL sIevcBuf [BLOCK_PEELING_SIZE][PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sCatW;
    KW_LOCAL_MEM REAL sWarpOp[ADJOINT_NWARPS_SP_N][PADDED_STATE_COUNT];

    /* Load column ls of evecT: evecT[j*S+ls] = U[j,ls] = U^T[ls,j] */
    if (tid < PADDED_STATE_COUNT) sEvecTCol[tid] = evecT[tid * PADDED_STATE_COUNT + ls];
    if (tid == 0)                 sCatW = categoryWeights[cat];
    KW_LOCAL_FENCE;

    REAL regOp[PADDED_STATE_COUNT];
    for (int rs = 0; rs < PADDED_STATE_COUNT; rs++) regOp[rs] = REAL(0);

    const int catOff = cat * totalPatterns * PADDED_STATE_COUNT;

    for (int k = tid; k < totalPatterns; k += ADJOINT_BLOCK_SP_N) {
        /* lhsLs = Σ_j sEvecTCol[j] * pre[k,j] = (U^T · pre)[ls] */
        REAL lhsLs = REAL(0);
        for (int j = 0; j < PADDED_STATE_COUNT; j++)
            SPECTRAL_FMA(sEvecTCol[j], prePartials[catOff + k * PADDED_STATE_COUNT + j], lhsLs);

        const REAL sc = patternWeights[k] * sCatW / perSiteLikelihoods[k];
        const REAL lv = lhsLs * sc;

        if constexpr (IsSameType<Child, States>) {
            /* rhs[rs] = ievc[tipState, rs] (column lookup) */
            const int s = tipStates[k];
            if (s < PADDED_STATE_COUNT) {
                for (int rs = 0; rs < PADDED_STATE_COUNT; rs++)
                    regOp[rs] += lv * ievc[s * PADDED_STATE_COUNT + rs];
            } else {
                for (int rs = 0; rs < PADDED_STATE_COUNT; rs++) {
                    REAL rhs = REAL(0);
                    for (int j = 0; j < PADDED_STATE_COUNT; j++)
                        rhs += ievc[j * PADDED_STATE_COUNT + rs];
                    regOp[rs] += lv * rhs;
                }
            }
        } else {
            /* rhs[rs] = Σ_j ievc[j*S+rs] * post[k,j].
             * Peel over j: all threads cooperate to load sIevcBuf[BPS][S];
             * each then reads its own post[k] elements independently. */
            for (int i = 0; i < PADDED_STATE_COUNT; i += BLOCK_PEELING_SIZE) {
                /* Coalesced load: thread t loads element (t/S, t%S) of sIevcBuf */
                const int elems = BLOCK_PEELING_SIZE * PADDED_STATE_COUNT;
                for (int e = tid; e < elems; e += ADJOINT_BLOCK_SP_N) {
                    const int row = e / PADDED_STATE_COUNT;
                    const int col = e % PADDED_STATE_COUNT;
                    sIevcBuf[row][col] = ievc[(i + row) * PADDED_STATE_COUNT + col];
                }
                KW_LOCAL_FENCE;
                for (int j = 0; j < BLOCK_PEELING_SIZE; j++) {
                    const REAL pj = postPartials[catOff + k * PADDED_STATE_COUNT + i + j];
                    for (int rs = 0; rs < PADDED_STATE_COUNT; rs++)
                        regOp[rs] += lv * sIevcBuf[j][rs] * pj;
                }
                KW_LOCAL_FENCE;
            }
        }
    }

    /* ── Warp reduction ─────────────────────────────────────────────────── */
    const unsigned FMASK = 0xffffffff;
    for (int rs = 0; rs < PADDED_STATE_COUNT; rs++)
        for (int off = 16; off >= 1; off >>= 1)
            regOp[rs] += __shfl_down_sync(FMASK, regOp[rs], off);

    const int warpId = tid >> 5, laneId = tid & 31;
    if (laneId == 0)
        for (int rs = 0; rs < PADDED_STATE_COUNT; rs++) sWarpOp[warpId][rs] = regOp[rs];
    KW_LOCAL_FENCE;

    /* Write back — loop handles S > ADJOINT_BLOCK_SP_N (e.g. S=192, 256). */
    for (int rs = tid; rs < PADDED_STATE_COUNT; rs += ADJOINT_BLOCK_SP_N) {
        REAL sum = REAL(0);
        for (int w = 0; w < ADJOINT_NWARPS_SP_N; w++) sum += sWarpOp[w][rs];
        dOpBuf[cat * PADDED_STATE_COUNT * PADDED_STATE_COUNT + ls * PADDED_STATE_COUNT + rs] = sum;
    }
}

/* ── Phase 2 body — integral transform for one category ─────────────────── */
template <bool IsAllReal>
__device__ void adjointPhase2Body(
        KW_GLOBAL_VAR REAL* KW_RESTRICT dOpBuf,
        KW_GLOBAL_VAR REAL* KW_RESTRICT eigenValues,
        KW_GLOBAL_VAR REAL* KW_RESTRICT distances,
        KW_GLOBAL_VAR REAL* KW_RESTRICT dGradient) {

    const int ls  = KW_LOCAL_ID_0;   /* one thread per lhs eigenstate */
    const int cat = KW_GROUP_ID_0;

    KW_LOCAL_MEM REAL sEvalR[PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sEvalI[PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sExpat[PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sCosbt[PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sSinbt[PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sExpatC[PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sExpatS[PADDED_STATE_COUNT];

    sEvalR[ls] = eigenValues[ls];
    if constexpr (!IsAllReal) sEvalI[ls] = eigenValues[PADDED_STATE_COUNT + ls];
    KW_LOCAL_FENCE;

    /* All threads must write sExpat/sCosbt/sSinbt so both KW_LOCAL_FENCEs are
     * reached by every thread (divergent __syncthreads is undefined).
     * Second pair member exits AFTER the fence. */
    const REAL t = distances[cat];
    const REAL e = exp(sEvalR[ls] * t);
    sExpat[ls] = e;
    if constexpr (!IsAllReal) {
        const REAL bt = sEvalI[ls] * t;
        sCosbt[ls]  = cos(bt); sSinbt[ls]  = sin(bt);
        sExpatC[ls] = e * sCosbt[ls]; sExpatS[ls] = e * sSinbt[ls];
    }
    KW_LOCAL_FENCE;

    /* Second member of a complex conjugate pair: its rows are handled by ls-1. */
    if constexpr (!IsAllReal) {
        if (sEvalI[ls] < REAL(0)) return;
    }

    const int S     = PADDED_STATE_COUNT;
    const int opOff = cat * S * S + ls * S;
    const REAL ea   = sExpat[ls];
    const REAL la   = sEvalR[ls];

    if constexpr (IsAllReal) {
        for (int rs = 0; rs < S; rs++) {
            const REAL coeff = (t * fabsf(la - sEvalR[rs]) < REAL(1e-12))
                               ? t * ea : (ea - sExpat[rs]) / (la - sEvalR[rs]);
            atomicAdd(&dGradient[ls * S + rs], dOpBuf[opOff + rs] * coeff);
        }
    } else {
        const REAL li = sEvalI[ls];
        if (li == REAL(0)) {
            /* ls is real */
            for (int rs = 0; rs < S; ) {
                const REAL ri = sEvalI[rs];
                if (ri == REAL(0)) {
                    /* 1×1 */
                    const REAL coeff = (t * fabsf(la - sEvalR[rs]) < REAL(1e-12))
                                       ? t * ea : (ea - sExpat[rs]) / (la - sEvalR[rs]);
                    atomicAdd(&dGradient[ls*S+rs], dOpBuf[opOff+rs] * coeff);
                    rs++;
                } else {
                    /* 1×2 */
                    const REAL sr = sEvalR[rs] - la;
                    const REAL den = sr*sr + ri*ri;
                    REAL ic0, ic1;
                    if (den < REAL(1e-12)) { ic0 = t; ic1 = REAL(0); }
                    else {
                        const REAL ex = sExpat[rs] / ea;
                        ic0 = (ex*(sr*sCosbt[rs]+ri*sSinbt[rs])-sr)/den;
                        ic1 = (ex*(sr*sSinbt[rs]-ri*sCosbt[rs])+ri)/den;
                    }
                    const REAL c0 = ea*ic0, c1 = ea*ic1;
                    const REAL in0 = dOpBuf[opOff+rs], in1 = dOpBuf[opOff+rs+1];
                    atomicAdd(&dGradient[ls*S+rs],    c0*in0 + c1*in1);
                    atomicAdd(&dGradient[ls*S+rs+1], -c1*in0 + c0*in1);
                    rs += 2;
                }
            }
        } else {
            /* ls is first of complex pair — read both rows from dOpBuf */
            const REAL li2 = li, lr = la;
            const REAL ec  = sExpatC[ls], es = sExpatS[ls];
            const REAL cI  = sCosbt[ls],  sI = sSinbt[ls];
            /* Note: ls+1 row stored at dOpBuf[cat*S*S + (ls+1)*S + rs] */
            const int opOff1 = cat * S * S + (ls + 1) * S;
            for (int rs = 0; rs < S; ) {
                const REAL ri = sEvalI[rs];
                if (ri == REAL(0)) {
                    /* 2×1 */
                    const REAL sr = sEvalR[rs] - lr;
                    const REAL den = sr*sr + li2*li2;
                    REAL ic0, ic1;
                    if (den < REAL(1e-12)) { ic0 = t; ic1 = REAL(0); }
                    else {
                        const REAL ex = sExpat[rs] / ea;
                        ic0 = (ex*(sr*cI+li2*sI)-sr)/den;
                        ic1 = (ex*(sr*sI-li2*cI)+li2)/den;
                    }
                    const REAL p0=ec*ic0+es*ic1, p1=ec*ic1-es*ic0;
                    const REAL p2=es*ic0-ec*ic1, p3=es*ic1+ec*ic0;
                    const REAL in0=dOpBuf[opOff+rs], in1=dOpBuf[opOff1+rs];
                    atomicAdd(&dGradient[ls*S+rs],     p0*in0+p1*in1);
                    atomicAdd(&dGradient[(ls+1)*S+rs], p2*in0+p3*in1);
                    rs++;
                } else {
                    /* 2×2 */
                    const REAL rr=sEvalR[rs], ri2=ri;
                    const REAL sr=rr-lr, si1=li2+ri2, si2=ri2-li2;
                    const REAL sr2=sr*sr;
                    const REAL d1=sr2+si1*si1, d2=sr2+si2*si2;
                    const REAL ex=(d1>=REAL(1e-12)||d2>=REAL(1e-12)) ? sExpat[rs]/ea : REAL(0);
                    const REAL clcr=cI*sCosbt[rs], slsr=sI*sSinbt[rs];
                    const REAL clsr=cI*sSinbt[rs], slcr=sI*sCosbt[rs];
                    REAL i1r,i1i;
                    if (d1<REAL(1e-12)){i1r=t;i1i=REAL(0);}
                    else{
                        const REAL cs1=clcr-slsr,sn1=slcr+clsr;
                        i1r=(sr*(ex*cs1-1)+si1*ex*sn1)/d1;
                        i1i=(sr*ex*sn1-si1*(ex*cs1-1))/d1;
                    }
                    REAL i2r,i2i;
                    if (d2<REAL(1e-12)){i2r=t;i2i=REAL(0);}
                    else{
                        const REAL cs2=clcr+slsr,sn2=clsr-slcr;
                        i2r=(sr*(ex*cs2-1)+si2*ex*sn2)/d2;
                        i2i=(sr*ex*sn2-si2*(ex*cs2-1))/d2;
                    }
                    const REAL pr=ec*i1r+es*i1i, pi_=ec*i1i-es*i1r;
                    const REAL mr_=ec*i2r-es*i2i, mi_=ec*i2i+es*i2r;
                    const REAL A=REAL(0.5)*(mr_+pr), B=REAL(0.5)*(mi_+pi_);
                    const REAL C=REAL(0.5)*(pi_-mi_), D=REAL(0.5)*(mr_-pr);
                    const REAL in00=dOpBuf[opOff+rs],   in01=dOpBuf[opOff+rs+1];
                    const REAL in10=dOpBuf[opOff1+rs],  in11=dOpBuf[opOff1+rs+1];
                    atomicAdd(&dGradient[ls*S+rs],         A*in00+B*in01+C*in10+D*in11);
                    atomicAdd(&dGradient[ls*S+rs+1],      -B*in00+A*in01-D*in10+C*in11);
                    atomicAdd(&dGradient[(ls+1)*S+rs],    -C*in00-D*in01+A*in10+B*in11);
                    atomicAdd(&dGradient[(ls+1)*S+rs+1],   D*in00-C*in01-B*in10+A*in11);
                    rs += 2;
                }
            }
        }
    }
}

extern "C" {

/* ── Phase 1 kernels ────────────────────────────────────────────────────── */

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
    adjointPhase1Body<Partials>(prePartials, postPartials, nullptr,
        evecT, ievc, distances, patternWeights, categoryWeights,
        perSiteLikelihoods, dOpBuf, totalPatterns);
}

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
    adjointPhase1Body<States>(prePartials, nullptr, tipStates,
        evecT, ievc, distances, patternWeights, categoryWeights,
        perSiteLikelihoods, dOpBuf, totalPatterns);
}

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
    adjointPhase1Body<Partials>(prePartials, postPartials, nullptr,
        evecT, ievc, distances, patternWeights, categoryWeights,
        perSiteLikelihoods, dOpBuf, totalPatterns);
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
    adjointPhase1Body<States>(prePartials, nullptr, tipStates,
        evecT, ievc, distances, patternWeights, categoryWeights,
        perSiteLikelihoods, dOpBuf, totalPatterns);
}

/* ── Phase 2 kernels ────────────────────────────────────────────────────── */

KW_GLOBAL_KERNEL void kernelAdjointPhase2AllRealN(
        KW_GLOBAL_VAR REAL* KW_RESTRICT dOpBuf,
        KW_GLOBAL_VAR REAL* KW_RESTRICT eigenValues,
        KW_GLOBAL_VAR REAL* KW_RESTRICT distances,
        KW_GLOBAL_VAR REAL* KW_RESTRICT dGradient) {
    adjointPhase2Body<true>(dOpBuf, eigenValues, distances, dGradient);
}

KW_GLOBAL_KERNEL void kernelAdjointPhase2ComplexN(
        KW_GLOBAL_VAR REAL* KW_RESTRICT dOpBuf,
        KW_GLOBAL_VAR REAL* KW_RESTRICT eigenValues,
        KW_GLOBAL_VAR REAL* KW_RESTRICT distances,
        KW_GLOBAL_VAR REAL* KW_RESTRICT dGradient) {
    adjointPhase2Body<false>(dOpBuf, eigenValues, distances, dGradient);
}

} // extern "C" (adjoint N-state)

#endif // CUDA
