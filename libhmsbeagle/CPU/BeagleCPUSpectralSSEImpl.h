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

#include <vector>

namespace beagle {
namespace cpu {

/*
 * Spectral CPU implementation, double precision, with SIMD across states (SSE2 on x86, NEON through sse2neon on
 * arm64). P x = V (exp(D t) (V^{-1} x)) is computed from the rows of the transposed matrices, so a block of states
 * stays in vector registers and no horizontal sums are needed. Results match BeagleCPUSpectralImpl up to rounding.
 */
template <int T_PAD, int P_PAD>
class BeagleCPUSpectralSSEImpl : public BeagleCPUSpectralImpl<double, T_PAD, P_PAD> {

    typedef BeagleCPUSpectralImpl<double, T_PAD, P_PAD> Base;

protected:
    using Base::gEigenDecomposition;
    using Base::kFlags;
    using Base::kStateCount;
    using Base::kCategoryCount;
    using Base::kPaddedPatternCount;
    using Base::kPartialsPaddedStateCount;
    using Base::kPartitionCount;
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

    int setCPUThreadCount(int threadCount) override;

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

private:
    // Per partition: u, y and z1, each kSimdTmpStride doubles, room for a full last vector when S is odd
    std::vector<double> gSimdTmp;
    int kSimdTmpStride;

    double* simdTmp(int currentPartition) { return gSimdTmp.data() + currentPartition * 3 * kSimdTmpStride; }

    // y = exp(D t) V^{-1} x for one child and category, from its partials x or, when x is null, its tip state
    void forwardEigenBasis(double* y, double* u, const double* x, const int state,
                           const BranchEigenInfo& info, const int catOffset);

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
