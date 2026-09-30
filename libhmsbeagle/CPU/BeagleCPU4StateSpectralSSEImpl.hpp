/*
 *  BeagleCPU4StateSpectralSSEImpl.hpp
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

#ifndef BEAGLE_CPU_4STATE_SPECTRAL_SSE_IMPL_HPP
#define BEAGLE_CPU_4STATE_SPECTRAL_SSE_IMPL_HPP

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include <algorithm>
#include <cmath>
#include <iostream>

#include "libhmsbeagle/beagle.h"
#include "libhmsbeagle/CPU/BeagleCPU4StateSpectralSSEImpl.h"
#include "libhmsbeagle/CPU/SSEKernels.h"

namespace beagle {
namespace cpu {

namespace spectral_sse4 {

/*
 * (lo, hi) = exp(D t) (lo, hi): y_i = A_i u_i + B_i u_p(i), where A = exp(at) cos(bt) (exp(at) for a real
 * eigenvalue), B = exp(at) sin(bt) (0 for a real eigenvalue, negated for the second of a pair) and p(i) is the
 * other index of i's conjugate pair; B is negated going backward (P^T).
 */
template <bool Backward>
inline void exponential(const int pairs, const double* A, const double* B, V_Real& lo, V_Real& hi) {
    V_Real ylo = VEC_MULT(VEC_LOADU(A), lo);
    V_Real yhi = VEC_MULT(VEC_LOADU(A + 2), hi);
    if (pairs != 0) {
        V_Real plo, phi;
        if (pairs == 4) { // (1, 2): lane 1 of lo pairs with lane 0 of hi; B is 0 on lanes 0 and 3
            plo = VEC_SHUFFLE0(hi, hi);
            phi = VEC_SHUFFLE1(lo, lo);
        } else {          // (0, 1) and / or (2, 3); B is 0 on a real pair of lanes
            plo = VEC_SWAP(lo);
            phi = VEC_SWAP(hi);
        }
        const V_Real blo = VEC_MULT(VEC_LOADU(B), plo);
        const V_Real bhi = VEC_MULT(VEC_LOADU(B + 2), phi);
        if (Backward) {
            ylo = VEC_SUB(ylo, blo);
            yhi = VEC_SUB(yhi, bhi);
        } else {
            ylo = VEC_ADD(ylo, blo);
            yhi = VEC_ADD(yhi, bhi);
        }
    }
    lo = ylo;
    hi = yhi;
}

} // namespace spectral_sse4

template <int T_PAD, int P_PAD>
const char* BeagleCPU4StateSpectralSSEImpl<T_PAD, P_PAD>::getName() {
    return "CPU-4State-Spectral-SSE-Double";
}

template <int T_PAD, int P_PAD>
int BeagleCPU4StateSpectralSSEImpl<T_PAD, P_PAD>::createInstance(int tipCount,
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
    if (stateCount != 4) {
        return BEAGLE_ERROR_OUT_OF_RANGE;
    }
    const int returnCode = Base::createInstance(tipCount, partialsBufferCount, compactBufferCount, stateCount,
                                                patternCount, eigenDecompositionCount, matrixCount, categoryCount,
                                                scaleBufferCount, resourceNumber, pluginResourceNumber,
                                                preferenceFlags, requirementFlags);
    gMatrices4.assign(kEigenDecompCount, Matrices4());
    return returnCode;
}

template <int T_PAD, int P_PAD>
int BeagleCPU4StateSpectralSSEImpl<T_PAD, P_PAD>::setEigenDecomposition(int eigenIndex,
                                                                       const double* inEigenVectors,
                                                                       const double* inInverseEigenVectors,
                                                                       const double* inEigenValues) {
    const int returnCode = Base::setEigenDecomposition(eigenIndex, inEigenVectors, inInverseEigenVectors,
                                                       inEigenValues);
    if (returnCode != BEAGLE_SUCCESS) {
        return returnCode;
    }

    // the spectral decomposition's matrices have rows of stride 4 + T_PAD; V^{-1}'s pad column holds its row sums
    const int stride = 4 + T_PAD;
    const double* evec = gEigenDecomposition->getEigenVectorsPtr(eigenIndex);
    const double* ivec = gEigenDecomposition->getInverseEigenVectorsPtr(eigenIndex);
    const double* tEvec = gEigenDecomposition->getBackwardsEigenVectorsPtr(eigenIndex);
    const double* tIvec = gEigenDecomposition->getBackwardsInverseEigenVectorsPtr(eigenIndex);

    Matrices4& m = gMatrices4[eigenIndex];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            m.inverseColumns[i * 4 + j] = tEvec[i * stride + j];
            m.columns[i * 4 + j] = tIvec[i * stride + j];
            m.transposeColumns[i * 4 + j] = evec[i * stride + j];
            m.inverseTransposeColumns[i * 4 + j] = ivec[i * stride + j];
        }
        m.inverseColumns[16 + i] = ivec[i * stride + 4];
    }

    const double* eval = gEigenDecomposition->getEigenValuesPtr(eigenIndex);
    const double* imag = (kFlags & BEAGLE_FLAG_EIGEN_COMPLEX) ? eval + 4 : nullptr;
    m.pairs = 0;
    for (int i = 0; i < 3; ) { // the same classification as updateBranchEigenInfo
        if (imag != nullptr && imag[i] != 0.0) {
            m.pairs |= (i == 0) ? 1 : (i == 1) ? 4 : 2;
            i += 2;
        } else {
            ++i;
        }
    }
    return BEAGLE_SUCCESS;
}

template <int T_PAD, int P_PAD>
void BeagleCPU4StateSpectralSSEImpl<T_PAD, P_PAD>::postOrder(double* destP,
                                                            const int* states1, const double* partials1,
                                                            const int branchEigenIndex1,
                                                            const int* states2, const double* partials2,
                                                            const int branchEigenIndex2,
                                                            const double* scaleFactors,
                                                            int startPattern, int endPattern) {
    const bool singleChild = (states2 == nullptr && partials2 == nullptr);
    const BranchEigenInfo& info1 = gBranchEigenInfo[branchEigenIndex1];
    const BranchEigenInfo& info2 = gBranchEigenInfo[singleChild ? branchEigenIndex1 : branchEigenIndex2];
    const Matrices4& m1 = gMatrices4[info1.eigenIndex];
    const Matrices4& m2 = gMatrices4[info2.eigenIndex];

    // max(P x, 0) of one child, from its partials x or, when x is null, its tip state (4 is missing)
    auto propagate = [](const Matrices4& m, const double* A, const double* B, const double* x, const int state,
                        V_Real& lo, V_Real& hi) {
        V_Real ulo, uhi;
        if (x != nullptr) {
            simd::product4(m.inverseColumns, VEC_LOADU(x), VEC_LOADU(x + 2), ulo, uhi);
        } else {
            ulo = VEC_LOAD(m.inverseColumns + 4 * state);
            uhi = VEC_LOAD(m.inverseColumns + 4 * state + 2);
        }
        spectral_sse4::exponential<false>(m.pairs, A, B, ulo, uhi);
        simd::product4(m.columns, ulo, uhi, lo, hi);
        const V_Real zero = VEC_SETZERO();
        lo = VEC_MAX(zero, lo);
        hi = VEC_MAX(zero, hi);
    };

    for (int l = 0; l < kCategoryCount; l++) {
        const int catOffset = l * 4;
        const double* A1 = info1.expatcosbt + catOffset;
        const double* B1 = info1.expatsinbt + catOffset;
        const double* A2 = info2.expatcosbt + catOffset;
        const double* B2 = info2.expatsinbt + catOffset;

        for (int k = startPattern; k < endPattern; k++) {
            const int v = l * 4 * kPaddedPatternCount + 4 * k;
            double* dest = destP + v;

            V_Real lo, hi;
            propagate(m1, A1, B1, partials1 ? partials1 + v : nullptr, states1 ? states1[k] : 0, lo, hi);
            if (!singleChild) {
                V_Real lo2, hi2;
                propagate(m2, A2, B2, partials2 ? partials2 + v : nullptr, states2 ? states2[k] : 0, lo2, hi2);
                lo = VEC_MULT(lo, lo2);
                hi = VEC_MULT(hi, hi2);
            }
            if (scaleFactors != nullptr) {
                const V_Real oneOverScaleFactor = VEC_SPLAT(1.0 / scaleFactors[k]);
                lo = VEC_MULT(lo, oneOverScaleFactor);
                hi = VEC_MULT(hi, oneOverScaleFactor);
            }
            VEC_STOREU(dest, lo);
            VEC_STOREU(dest + 2, hi);
        }
    }
}

template <int T_PAD, int P_PAD>
void BeagleCPU4StateSpectralSSEImpl<T_PAD, P_PAD>::preOrder(double* destP, SpectralPreOrder type,
                                                           const double* partials1, const int branchEigenIndex1,
                                                           const int* states2, const double* partials2,
                                                           const int branchEigenIndex2,
                                                           int startPattern, int endPattern) {
    const bool hasSibling = (states2 != nullptr || partials2 != nullptr);
    const bool hasBranch = (type != SpectralPreOrder::TopRoot);
    const BranchEigenInfo* info1 = hasBranch ? &gBranchEigenInfo[branchEigenIndex1] : nullptr;
    const BranchEigenInfo* info2 = hasSibling ? &gBranchEigenInfo[branchEigenIndex2] : nullptr;
    const Matrices4* m1 = hasBranch ? &gMatrices4[info1->eigenIndex] : nullptr;
    const Matrices4* m2 = hasSibling ? &gMatrices4[info2->eigenIndex] : nullptr;
    const V_Real zero = VEC_SETZERO();

    for (int l = 0; l < kCategoryCount; l++) {
        const int catOffset = l * 4;

        for (int k = startPattern; k < endPattern; k++) {
            const int v = l * 4 * kPaddedPatternCount + 4 * k;
            double* dest = destP + v;
            const V_Real parentLo = VEC_LOADU(partials1 + v);
            const V_Real parentHi = VEC_LOADU(partials1 + v + 2);

            V_Real siblingLo, siblingHi; // max(P x, 0) of the sibling
            if (hasSibling) {
                const double* x = partials2 ? partials2 + v : nullptr;
                V_Real ulo, uhi;
                if (x != nullptr) {
                    simd::product4(m2->inverseColumns, VEC_LOADU(x), VEC_LOADU(x + 2), ulo, uhi);
                } else {
                    ulo = VEC_LOAD(m2->inverseColumns + 4 * states2[k]);
                    uhi = VEC_LOAD(m2->inverseColumns + 4 * states2[k] + 2);
                }
                spectral_sse4::exponential<false>(m2->pairs, info2->expatcosbt + catOffset,
                                                  info2->expatsinbt + catOffset, ulo, uhi);
                simd::product4(m2->columns, ulo, uhi, siblingLo, siblingHi);
                siblingLo = VEC_MAX(zero, siblingLo);
                siblingHi = VEC_MAX(zero, siblingHi);
            }

            if (type == SpectralPreOrder::TopRoot) { // max(P x, 0) * root prior
                VEC_STOREU(dest, VEC_MULT(siblingLo, parentLo));
                VEC_STOREU(dest + 2, VEC_MULT(siblingHi, parentHi));
                continue;
            }

            // P^T of the parent's partials (TOP, degree-2) or of max(P x, 0) * parent (BOTTOM)
            V_Real xlo = parentLo;
            V_Real xhi = parentHi;
            if (hasSibling && type == SpectralPreOrder::Bottom) {
                xlo = VEC_MULT(siblingLo, parentLo);
                xhi = VEC_MULT(siblingHi, parentHi);
            }
            V_Real ulo, uhi;
            simd::product4(m1->transposeColumns, xlo, xhi, ulo, uhi);
            spectral_sse4::exponential<true>(m1->pairs, info1->expatcosbt + catOffset,
                                             info1->expatsinbt + catOffset, ulo, uhi);
            V_Real lo, hi;
            simd::product4(m1->inverseTransposeColumns, ulo, uhi, lo, hi);
            lo = VEC_MAX(zero, lo);
            hi = VEC_MAX(zero, hi);
            if (hasSibling && type == SpectralPreOrder::Top) {
                lo = VEC_MULT(lo, siblingLo);
                hi = VEC_MULT(hi, siblingHi);
            }
            VEC_STOREU(dest, lo);
            VEC_STOREU(dest + 2, hi);
        }
    }
}

template <int T_PAD, int P_PAD>
void BeagleCPU4StateSpectralSSEImpl<T_PAD, P_PAD>::spectralPartialsPartials(double* destP,
        const double* partials1, const int branchEigenIndex1,
        const double* partials2, const int branchEigenIndex2,
        const double* scaleFactors,
        int startPattern, int endPattern, int currentPartition) {
    postOrder(destP, nullptr, partials1, branchEigenIndex1, nullptr, partials2, branchEigenIndex2,
              scaleFactors, startPattern, endPattern);
}

template <int T_PAD, int P_PAD>
void BeagleCPU4StateSpectralSSEImpl<T_PAD, P_PAD>::spectralStatesPartials(double* destP,
        const int* states1, const int branchEigenIndex1,
        const double* partials2, const int branchEigenIndex2,
        const double* scaleFactors,
        int startPattern, int endPattern, int currentPartition) {
    postOrder(destP, states1, nullptr, branchEigenIndex1, nullptr, partials2, branchEigenIndex2,
              scaleFactors, startPattern, endPattern);
}

template <int T_PAD, int P_PAD>
void BeagleCPU4StateSpectralSSEImpl<T_PAD, P_PAD>::spectralStatesStates(double* destP,
        const int* states1, const int branchEigenIndex1,
        const int* states2, const int branchEigenIndex2,
        const double* scaleFactors,
        int startPattern, int endPattern, int currentPartition) {
    postOrder(destP, states1, nullptr, branchEigenIndex1, states2, nullptr, branchEigenIndex2,
              scaleFactors, startPattern, endPattern);
}

template <int T_PAD, int P_PAD>
void BeagleCPU4StateSpectralSSEImpl<T_PAD, P_PAD>::spectralDegree2Partials(double* destP,
        const int* states1, const double* partials1,
        const int branchEigenIndex1,
        const double* scaleFactors,
        int startPattern, int endPattern, int currentPartition) {
    postOrder(destP, states1, states1 ? nullptr : partials1, branchEigenIndex1, nullptr, nullptr, -1,
              scaleFactors, startPattern, endPattern);
}

template <int T_PAD, int P_PAD>
void BeagleCPU4StateSpectralSSEImpl<T_PAD, P_PAD>::spectralPrePartialsPartials(double* destP,
        SpectralPreOrder type,
        const double* partials1, const int branchEigenIndex1,
        const double* partials2, const int branchEigenIndex2,
        int startPattern, int endPattern, int currentPartition) {
    preOrder(destP, type, partials1, branchEigenIndex1, nullptr, partials2, branchEigenIndex2,
             startPattern, endPattern);
}

template <int T_PAD, int P_PAD>
void BeagleCPU4StateSpectralSSEImpl<T_PAD, P_PAD>::spectralPrePartialsStates(double* destP,
        SpectralPreOrder type,
        const double* partials1, const int branchEigenIndex1,
        const int* states2, const int branchEigenIndex2,
        int startPattern, int endPattern, int currentPartition) {
    preOrder(destP, type, partials1, branchEigenIndex1, states2, nullptr, branchEigenIndex2,
             startPattern, endPattern);
}

template <int T_PAD, int P_PAD>
void BeagleCPU4StateSpectralSSEImpl<T_PAD, P_PAD>::calcDegree2PrePartials(double* destP,
        const double* partials1,
        const int branchEigenIndex1,
        int startPattern, int endPattern, int currentPartition) {
    if (branchEigenIndex1 < 0) { // top partials below the root: a copy
        Base::calcDegree2PrePartials(destP, partials1, branchEigenIndex1, startPattern, endPattern, currentPartition);
        return;
    }
    // the same for bottom and top partials: P^T of the parent's pre-order partials through branchEigenIndex1
    preOrder(destP, SpectralPreOrder::Bottom, partials1, branchEigenIndex1, nullptr, nullptr, -1,
             startPattern, endPattern);
}

///////////////////////////////////////////////////////////////////////////////
// BeagleCPU4StateSpectralSSEImplFactory public methods

inline BeagleImpl* BeagleCPU4StateSpectralSSEImplFactory::createImpl(int tipCount,
                                                                     int partialsBufferCount,
                                                                     int compactBufferCount,
                                                                     int stateCount,
                                                                     int patternCount,
                                                                     int eigenBufferCount,
                                                                     int matrixBufferCount,
                                                                     int categoryCount,
                                                                     int scaleBufferCount,
                                                                     int resourceNumber,
                                                                     int pluginResourceNumber,
                                                                     long preferenceFlags,
                                                                     long requirementFlags,
                                                                     int* errorCode) {

    if (stateCount != 4 || !CPUSupportsSSE()) {
        return NULL;
    }

    BeagleImpl* impl = new BeagleCPU4StateSpectralSSEImpl<T_PAD_DEFAULT, P_PAD_DEFAULT>();

    try {
        if (impl->createInstance(tipCount, partialsBufferCount, compactBufferCount, stateCount,
                                 patternCount, eigenBufferCount, matrixBufferCount,
                                 categoryCount, scaleBufferCount, resourceNumber,
                                 pluginResourceNumber,
                                 preferenceFlags, requirementFlags) == 0)
            return impl;
    }
    catch(...) {
        if (DEBUGGING_OUTPUT)
            std::cerr << "exception in initialize\n";
        delete impl;
        throw;
    }

    delete impl;

    return NULL;
}

inline const char* BeagleCPU4StateSpectralSSEImplFactory::getName() {
    return "CPU-4State-Spectral-SSE-Double";
}

inline const long BeagleCPU4StateSpectralSSEImplFactory::getFlags() {
    return BeagleCPUSpectralSSEImplFactory().getFlags();
}

} // namespace cpu
} // namespace beagle

#endif // BEAGLE_CPU_4STATE_SPECTRAL_SSE_IMPL_HPP
