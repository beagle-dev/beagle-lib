/*
 *  BeagleCPUSpectralSSEImpl.h
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

#ifndef __BeagleCPUSpectralSSEImpl__
#define __BeagleCPUSpectralSSEImpl__

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include "libhmsbeagle/CPU/BeagleCPUSpectralImpl.h"
#include "libhmsbeagle/CPU/BeagleCPUAdjointSSE.h"

#include <vector>

namespace beagle {
namespace cpu {

/*
 * Spectral CPU implementation, double precision, with SIMD across states (SSE2 on x86, NEON through sse2neon on
 * arm64). P x = V (exp(D t) (V^{-1} x)) is computed from the rows of the transposed matrices, so a block of states
 * stays in vector registers and no horizontal sums are needed. The adjoint gradient is BeagleCPUAdjointSSE's.
 * Results match BeagleCPUSpectralImpl up to rounding.
 */
template <int T_PAD, int P_PAD>
class BeagleCPUSpectralSSEImpl : public BeagleCPUAdjointSSE<BeagleCPUSpectralImpl<double, T_PAD, P_PAD>> {

    typedef BeagleCPUAdjointSSE<BeagleCPUSpectralImpl<double, T_PAD, P_PAD>> Base;

protected:
    using Base::gEigenDecomposition;
    using Base::kFlags;
    using Base::kStateCount;
    using Base::kCategoryCount;
    using Base::kPaddedPatternCount;
    using Base::kPartialsPaddedStateCount;
    using Base::kPartitionCount;
    using Base::gPartials;
    using typename Base::BranchEigenInfo;
    using Base::gBranchEigenInfo;

public:
    const char* getName() override;

    const long getFlags() override;

    int createInstance(int tipCount,
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
                       long requirementFlags) override;

    int setPatternPartitions(int partitionCount, const int* inPatternPartitions) override;

protected:
    void spectralPartialsPartials(double* destP,
                                  const double* partials1, const int branchEigenIndex1,
                                  const double* partials2, const int branchEigenIndex2,
                                  const double* scaleFactors,
                                  int startPattern, int endPattern, int currentPartition) override;

    void spectralStatesPartials(double* destP,
                                const int* states1, const int branchEigenIndex1,
                                const double* partials2, const int branchEigenIndex2,
                                const double* scaleFactors,
                                int startPattern, int endPattern, int currentPartition) override;

    void spectralStatesStates(double* destP,
                              const int* states1, const int branchEigenIndex1,
                              const int* states2, const int branchEigenIndex2,
                              const double* scaleFactors,
                              int startPattern, int endPattern, int currentPartition) override;

    void spectralDegree2Partials(double* destP,
                                 const int* states1, const double* partials1,
                                 const int branchEigenIndex1,
                                 const double* scaleFactors,
                                 int startPattern, int endPattern, int currentPartition) override;

    void spectralPrePartialsPartials(double* destP, SpectralPreOrder type,
                                     const double* partials1, const int branchEigenIndex1,
                                     const double* partials2, const int branchEigenIndex2,
                                     int startPattern, int endPattern, int currentPartition) override;

    void spectralPrePartialsStates(double* destP, SpectralPreOrder type,
                                   const double* partials1, const int branchEigenIndex1,
                                   const int* states2, const int branchEigenIndex2,
                                   int startPattern, int endPattern, int currentPartition) override;

    void calcDegree2PrePartials(double* destP,
                                const double* partials1,
                                const int branchEigenIndex1,
                                int startPattern,
                                int endPattern,
                                int currentPartition) override;

private:
    // Per partition: u, y and a third vector (z1 or w), each kSimdTmpStride doubles, room for a full last vector
    // when S is odd; the stride is 0 until createInstance sets it (its base may set the pattern partitions first)
    std::vector<double> gSimdTmp;
    int kSimdTmpStride = 0;

    double* simdTmp(int currentPartition) { return gSimdTmp.data() + currentPartition * 3 * kSimdTmpStride; }

    // y = exp(D t) V^{-1} x for one child and category, from its partials x or, when x is null, its tip state
    void forwardEigenBasis(double* y, double* u, const double* x, const int state,
                           const BranchEigenInfo& info, const int catOffset);

    // y = exp(D t)^T V^T x, the first half of P^T x = V^{-T} y
    void backwardEigenBasis(double* y, double* u, const double* x,
                            const BranchEigenInfo& info, const int catOffset);

    // Pre-order partials from the parent's pre-order partials1 through branchEigenIndex1 (P^T; none for TopRoot)
    // and the sibling (P x) from its partials or tip states; no sibling (both null) makes a degree-2 node
    void preOrder(double* destP, SpectralPreOrder type,
                  const double* partials1, const int branchEigenIndex1,
                  const int* states2, const double* partials2, const int branchEigenIndex2,
                  int startPattern, int endPattern, int currentPartition);

    // Two children, each from partials (states null) or a tip state (partials null); a null second child makes
    // a degree-2 node. dest = max(z1, 0) * max(z2, 0) [/ scale factor], with z = P x for each child.
    void postOrder(double* destP,
                   const int* states1, const double* partials1, const int branchEigenIndex1,
                   const int* states2, const double* partials2, const int branchEigenIndex2,
                   const double* scaleFactors,
                   int startPattern, int endPattern, int currentPartition);
};

class BeagleCPUSpectralSSEImplFactory : public BeagleImplFactory {
public:
    virtual BeagleImpl* createImpl(int tipCount,
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
                                   int* errorCode);

    virtual const char* getName();
    virtual const long getFlags();
};

} // namespace cpu
} // namespace beagle

// now include the file containing template function implementations
#include "libhmsbeagle/CPU/BeagleCPUSpectralSSEImpl.hpp"

#endif // __BeagleCPUSpectralSSEImpl__
