/*
 *  BeagleCPUAdjoint4StateSSE.hpp
 *  BEAGLE
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
 */

#ifndef BEAGLE_CPU_ADJOINT_4STATE_SSE_HPP
#define BEAGLE_CPU_ADJOINT_4STATE_SSE_HPP

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include <algorithm>
#include <cmath>
#include <limits>

#include "libhmsbeagle/beagle.h"
#include "libhmsbeagle/CPU/BeagleCPUAdjoint4StateSSE.h"
#include "libhmsbeagle/CPU/SSEDefinitions.h"
#include "libhmsbeagle/CPU/SSEKernels.h"

namespace beagle {
namespace cpu {

template <typename Base>
int BeagleCPUAdjoint4StateSSE<Base>::createInstance(int tipCount,
                                                    int partialsBufferCount,
                                                    int compactBufferCount,
                                                    int stateCount,
                                                    int patternCount,
                                                    int eigenDecompositionCount,
                                                    int matrixCount,
                                                    int categoryCount,
                                                    int scaleBufferCount,
                                                    int resourceNumber,
                                                    int pluginResourceNumber,
                                                    long preferenceFlags,
                                                    long requirementFlags) {
    const int returnCode = Base::createInstance(tipCount, partialsBufferCount, compactBufferCount, stateCount,
                                                patternCount, eigenDecompositionCount, matrixCount, categoryCount,
                                                scaleBufferCount, resourceNumber, pluginResourceNumber,
                                                preferenceFlags, requirementFlags);
    Adjoint4 stale = Adjoint4();
    stale.stale = true;
    gAdjoint4.assign(kEigenDecompCount, stale);
    return returnCode;
}

template <typename Base>
int BeagleCPUAdjoint4StateSSE<Base>::setEigenDecomposition(int eigenIndex,
                                                           const double* inEigenVectors,
                                                           const double* inInverseEigenVectors,
                                                           const double* inEigenValues) {
    const int returnCode = Base::setEigenDecomposition(eigenIndex, inEigenVectors, inInverseEigenVectors,
                                                       inEigenValues);
    if (returnCode == BEAGLE_SUCCESS) {
        gAdjoint4[eigenIndex].stale = true; // built by the next adjoint gradient that uses it
    }
    return returnCode;
}

template <typename Base>
void BeagleCPUAdjoint4StateSSE<Base>::prepareAdjoint(const int* branchEigenIndices, int count) {
    Base::prepareAdjoint(branchEigenIndices, count); // also readies the matrices of the spectral representation
    // those matrices have rows of stride 4 + T_PAD; V^{-1}'s pad column holds its row sums
    const int stride = kTransPaddedStateCount;
    for (int i = 0; i < count; ++i) {
        const int eigenIndex = gBranchEigenInfo[branchEigenIndices[i]].eigenIndex;
        Adjoint4& m = gAdjoint4[eigenIndex];
        if (!m.stale) {
            continue;
        }
        const double* evec = gEigenDecomposition->getEigenVectorsPtr(eigenIndex);
        const double* ivec = gEigenDecomposition->getInverseEigenVectorsPtr(eigenIndex);
        const double* tEvec = gEigenDecomposition->getBackwardsEigenVectorsPtr(eigenIndex);
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                m.inverseColumns[r * 4 + c] = tEvec[r * stride + c];
                m.transposeColumns[r * 4 + c] = evec[r * stride + c];
            }
            m.inverseColumns[16 + r] = ivec[r * stride + 4];
        }

        const double* eval = gEigenDecomposition->getEigenValuesPtr(eigenIndex);
        m.allReal = true;
        if (kFlags & BEAGLE_FLAG_EIGEN_COMPLEX) {
            for (int l = 0; l < 4; ++l) {
                m.allReal = m.allReal && (eval[4 + l] == 0.0);
            }
        }
        m.smallestDistance = std::numeric_limits<double>::infinity();
        for (int l = 0; l < 4; ++l) {
            m.eigenvalues[l] = eval[l];
            for (int r = 0; r < 4; ++r) {
                const double distance = eval[l] - eval[r];
                m.reciprocals[l * 4 + r] = (distance == 0.0) ? 0.0 : 1.0 / distance;
                m.equalLanes[l * 4 + r] = (distance == 0.0) ? ~0ULL : 0ULL;
                if (distance != 0.0) {
                    m.smallestDistance = std::min(m.smallestDistance, std::abs(distance));
                }
            }
        }
        m.stale = false;
    }
}

/*
 * As BeagleCPUAdjointSSE::calcAdjointCrossProductsRange, for real eigenvalues: per branch, the outer product
 * O = sum_k scale_k (V^T pre_k) (V^{-1} post_k)^T over the patterns, then gradient += O * K(t) entrywise with
 * K_lr = (e_l - e_r) / (lambda_l - lambda_r), or t e_l where t |lambda_l - lambda_r| < 1e-12
 */
template <typename Base>
void BeagleCPUAdjoint4StateSSE<Base>::calcAdjointCrossProductsRange(const int* postBufferIndices,
        const int* preBufferIndices,
        const int* branchEigenIndices,
        const double* categoryRates,
        const double* categoryWeights,
        const double* perSiteLikelihoods,
        double* buffer,
        int startNode,
        int endNode,
        int startPattern,
        int endPattern,
        int currentPartition,
        const int* postScaleIndices,
        const int* preScaleIndices,
        const int cumulativeScaleIndex,
        double* branchMarginalLk) {

    bool allReal = (postScaleIndices == nullptr && preScaleIndices == nullptr && branchMarginalLk == nullptr);
    for (int node = startNode; allReal && node < endNode; ++node) {
        allReal = gAdjoint4[gBranchEigenInfo[branchEigenIndices[node]].eigenIndex].allReal;
    }
    if (!allReal) {
        Base::calcAdjointCrossProductsRange(postBufferIndices, preBufferIndices, branchEigenIndices,
                                            categoryRates, categoryWeights, perSiteLikelihoods, buffer,
                                            startNode, endNode, startPattern, endPattern, currentPartition,
                                            postScaleIndices, preScaleIndices, cumulativeScaleIndex,
                                            branchMarginalLk);
        return;
    }

    const V_Real zero = VEC_SETZERO();
    const V_Real threshold = VEC_SPLAT(1e-12);
    const V_Real signBit = VEC_SPLAT(-0.0);
    V_Real gradient[8]; // rows l = 0..3, two vectors each
    for (int i = 0; i < 8; ++i) {
        gradient[i] = zero;
    }

    for (int category = 0; category < kCategoryCount; ++category) {
        const double categoryRate = categoryRates[category];
        const int infoOffset = category * kPartialsPaddedStateCount;
        const double categoryWeight = categoryWeights[category];

        for (int node = startNode; node < endNode; ++node) {
            const BranchEigenInfo& info = gBranchEigenInfo[branchEigenIndices[node]];
            const Adjoint4& m = gAdjoint4[info.eigenIndex];
            const double time = categoryRate * info.branchLength;

            const double* pre = gPartials[preBufferIndices[node]];
            const int* tipStates = gTipStates[postBufferIndices[node]];
            const double* post = (tipStates != nullptr) ? nullptr : gPartials[postBufferIndices[node]];

            V_Real outer[8];
            for (int i = 0; i < 8; ++i) {
                outer[i] = zero;
            }
            for (int k = startPattern; k < endPattern; ++k) {
                const int v = category * kPartialsPaddedStateCount * kPatternCount + kPartialsPaddedStateCount * k;
                const V_Real scale = VEC_SPLAT(gPatternWeights[k] * categoryWeight / perSiteLikelihoods[k]);
                V_Real lo, hi; // scale V^T pre
                simd::product4(m.transposeColumns, VEC_LOADU(pre + v), VEC_LOADU(pre + v + 2), lo, hi);
                lo = VEC_MULT(lo, scale);
                hi = VEC_MULT(hi, scale);
                V_Real rlo, rhi; // V^{-1} post, or a column of V^{-1} for a tip state (4 is missing)
                if (post != nullptr) {
                    simd::product4(m.inverseColumns, VEC_LOADU(post + v), VEC_LOADU(post + v + 2), rlo, rhi);
                } else {
                    rlo = VEC_LOAD(m.inverseColumns + 4 * tipStates[k]);
                    rhi = VEC_LOAD(m.inverseColumns + 4 * tipStates[k] + 2);
                }
                const V_Real a[4] = {VEC_SHUFFLE0(lo, lo), VEC_SHUFFLE1(lo, lo),
                                     VEC_SHUFFLE0(hi, hi), VEC_SHUFFLE1(hi, hi)};
                for (int l = 0; l < 4; ++l) {
                    outer[2 * l]     = VEC_MADD(a[l], rlo, outer[2 * l]);
                    outer[2 * l + 1] = VEC_MADD(a[l], rhi, outer[2 * l + 1]);
                }
            }

            const double* expat = info.expat + infoOffset;
            const V_Real elo = VEC_LOADU(expat);
            const V_Real ehi = VEC_LOADU(expat + 2);
            if (time * m.smallestDistance >= 1e-12) { // the degenerate entries are exactly the equal eigenvalues
                for (int l = 0; l < 4; ++l) {
                    const V_Real el = VEC_SPLAT(expat[l]);
                    const V_Real degenerate = VEC_SPLAT(time * expat[l]);
                    const V_Real clo = VEC_ADD(VEC_MULT(VEC_SUB(el, elo), VEC_LOAD(m.reciprocals + 4 * l)),
                            VEC_AND(VEC_LOAD((const double*) (m.equalLanes + 4 * l)), degenerate));
                    const V_Real chi = VEC_ADD(VEC_MULT(VEC_SUB(el, ehi), VEC_LOAD(m.reciprocals + 4 * l + 2)),
                            VEC_AND(VEC_LOAD((const double*) (m.equalLanes + 4 * l + 2)), degenerate));
                    gradient[2 * l]     = VEC_MADD(outer[2 * l], clo, gradient[2 * l]);
                    gradient[2 * l + 1] = VEC_MADD(outer[2 * l + 1], chi, gradient[2 * l + 1]);
                }
            } else {
                const V_Real t = VEC_SPLAT(time);
                const V_Real llo = VEC_LOAD(m.eigenvalues);
                const V_Real lhi = VEC_LOAD(m.eigenvalues + 2);
                for (int l = 0; l < 4; ++l) {
                    const V_Real el = VEC_SPLAT(expat[l]);
                    const V_Real ll = VEC_SPLAT(m.eigenvalues[l]);
                    const V_Real degenerate = VEC_SPLAT(time * expat[l]);
                    const V_Real dlo = VEC_CMPLT(VEC_MULT(t, VEC_ANDNOT(signBit, VEC_SUB(ll, llo))), threshold);
                    const V_Real dhi = VEC_CMPLT(VEC_MULT(t, VEC_ANDNOT(signBit, VEC_SUB(ll, lhi))), threshold);
                    const V_Real rlo = VEC_MULT(VEC_SUB(el, elo), VEC_LOAD(m.reciprocals + 4 * l));
                    const V_Real rhi = VEC_MULT(VEC_SUB(el, ehi), VEC_LOAD(m.reciprocals + 4 * l + 2));
                    const V_Real clo = VEC_OR(VEC_AND(dlo, degenerate), VEC_ANDNOT(dlo, rlo));
                    const V_Real chi = VEC_OR(VEC_AND(dhi, degenerate), VEC_ANDNOT(dhi, rhi));
                    gradient[2 * l]     = VEC_MADD(outer[2 * l], clo, gradient[2 * l]);
                    gradient[2 * l + 1] = VEC_MADD(outer[2 * l + 1], chi, gradient[2 * l + 1]);
                }
            }
        }
    }

    for (int i = 0; i < 8; ++i) {
        VEC_STOREU(buffer + 2 * i, VEC_ADD(VEC_LOADU(buffer + 2 * i), gradient[i]));
    }
}

} // namespace cpu
} // namespace beagle

#endif // BEAGLE_CPU_ADJOINT_4STATE_SSE_HPP
