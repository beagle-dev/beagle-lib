/*
 *  BeagleCPUAdjoint4StateSSE.h
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

#ifndef __BeagleCPUAdjoint4StateSSE__
#define __BeagleCPUAdjoint4StateSSE__

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include <vector>

namespace beagle {
namespace cpu {

/*
 * The adjoint gradient of the rate matrix for 4 states and real eigenvalues, double precision, SSE2 / NEON: the
 * 4 x 4 gradient and each branch's outer product stay in vector registers. Base is a BeagleCPUAdjointSSE (or
 * derives from one), whose adjoint gradient is used for complex eigenvalues and rescaled partials.
 */
template <typename Base>
class BeagleCPUAdjoint4StateSSE : public Base {

protected:
    using Base::gEigenDecomposition;
    using Base::kFlags;
    using Base::kTransPaddedStateCount;
    using Base::kCategoryCount;
    using Base::kPartialsPaddedStateCount;
    using Base::kPatternCount;
    using Base::kEigenDecompCount;
    using Base::gPartials;
    using Base::gTipStates;
    using Base::gPatternWeights;
    using typename Base::BranchEigenInfo;
    using Base::gBranchEigenInfo;

public:
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

    int setEigenDecomposition(int eigenIndex,
                              const double* inEigenVectors,
                              const double* inInverseEigenVectors,
                              const double* inEigenValues) override;

protected:
    void prepareAdjoint(const int* branchEigenIndices, int count) override;

    void calcAdjointCrossProductsRange(const int* postBufferIndices,
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
                                       double* branchMarginalLk) override;

private:
    // Per eigen decomposition, built on first use (prepareAdjoint), 4 x 4 row-major and 16-byte aligned: the columns
    // of V^{-1} (the rows of V^{-T}) followed by the row sums of V^{-1} (a missing state) and the columns of V^T (the
    // rows of V); for the integral kernel, 1 / (lambda_l - lambda_r) with 0 for equal eigenvalues, all-ones lanes
    // where lambda_l == lambda_r, the eigenvalues and the smallest distance between two different eigenvalues
    struct Adjoint4 {
        alignas(16) double inverseColumns[20];
        alignas(16) double transposeColumns[16];
        alignas(16) double reciprocals[16];
        alignas(16) unsigned long long equalLanes[16];
        alignas(16) double eigenvalues[4];
        double smallestDistance;
        bool allReal;
        bool stale;
    };
    std::vector<Adjoint4> gAdjoint4;
};

} // namespace cpu
} // namespace beagle

// now include the file containing template function implementations
#include "libhmsbeagle/CPU/BeagleCPUAdjoint4StateSSE.hpp"

#endif // __BeagleCPUAdjoint4StateSSE__
