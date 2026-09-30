/*
 *  BeagleCPUAdjointSSE.h
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

#ifndef __BeagleCPUAdjointSSE__
#define __BeagleCPUAdjointSSE__

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include <array>
#include <utility>
#include <vector>

namespace beagle {
namespace cpu {

/*
 * The adjoint gradient of the rate matrix in the eigen basis (calcAdjointCrossProductsRange) with SIMD across
 * states, double precision, for a CPU implementation Base whose eigen decomposition has the matrices of the
 * spectral representation: EigenDecompositionSpectral, or the adjoint storage of EigenDecompositionCube and
 * EigenDecompositionSquare, both with rows of stride kTransPaddedStateCount. With rescaled partials (scale
 * indices) Base's implementation is used.
 */
template <typename Base>
class BeagleCPUAdjointSSE : public Base {

protected:
    using Base::gEigenDecomposition;
    using Base::kFlags;
    using Base::kStateCount;
    using Base::kTransPaddedStateCount;
    using Base::kCategoryCount;
    using Base::kPartialsPaddedStateCount;
    using Base::kPatternCount;
    using Base::kPartitionCount;
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

    int setPatternPartitions(int partitionCount, const int* inPatternPartitions) override;

    int setEigenDecomposition(int eigenIndex,
                              const double* inEigenVectors,
                              const double* inInverseEigenVectors,
                              const double* inEigenValues) override;

protected:
    // Builds the adjoint plans of the eigen decompositions set since the last gradient
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
    // Per eigen decomposition, for the adjoint integral kernel: its real eigenvalues and the first index of each
    // complex conjugate pair, and a table of reciprocals that depend only on the eigenvalues (row stride
    // kAdjointStride): 1 / (lambda_l - lambda_r) for two real eigenvalues, the reciprocal denominator of a
    // block with a complex pair, 0 where that block is degenerate
    struct AdjointPlan {
        bool allReal;
        std::vector<int> realIndices;
        std::vector<int> pairIndices;
        std::vector<double> reciprocals;
        // Real eigenvalues only: the pairs (l, r) with lambda_l == lambda_r, including l == r (reciprocal 0), and
        // the smallest distance between two different eigenvalues. When time * that distance >= 1e-12 the
        // degenerate kernel entries are exactly the equal pairs.
        std::vector<std::pair<int, int>> equalPairs;
        double smallestDistance;
        // Two conjugate pairs of rows and columns (pl, pr): for each pl, five arrays over pr with stride pairStride,
        // padded with 0: the real part difference, the sum and the difference of the imaginary parts, and
        // 0.5 / d1, 0.5 / d2 of the two denominators (0 where degenerate, listed as (pl, pr, 1 or 2))
        int pairStride;
        std::vector<double> pairBlocks;
        std::vector<std::array<int, 3>> degenerateBlocks;
    };
    std::vector<AdjointPlan> gAdjointPlans;
    std::vector<bool> gAdjointPlanStale; // set by setEigenDecomposition, built on the next gradient
    int kAdjointStride = 0;
    // Per partition: the two vectors of an outer product, kAdjointVectorStride each (room for a full last vector
    // when S is odd); the outer product of several patterns, S rows of kAdjointStride; and exp(a t) cos(b t),
    // exp(a t) sin(b t) of each conjugate pair, kAdjointPairTmpStride each. The strides are 0 until
    // createInstance sets them (its base may set the pattern partitions first).
    int kAdjointVectorStride = 0;
    std::vector<double> gAdjointVectorTmp;
    std::vector<double> gAdjointOuterTmp;
    int kAdjointPairTmpStride = 0;
    std::vector<double> gAdjointPairTmp;

    void resizeAdjointTmp();

    void prepareAdjointPlan(int eigenIndex);

    // gradient += (outer product of the branch's pre- and post-order partials in the eigen basis) times the
    // integral kernel of the branch; View gives entries (l, r) and (l, r + 1) of that outer product
    template <typename View>
    void adjointKernel(double* gradient, const View& view, const AdjointPlan& plan, const double* eval,
                       const BranchEigenInfo& info, int infoOffset, double time, double* pairTmp);
};

} // namespace cpu
} // namespace beagle

// now include the file containing template function implementations
#include "libhmsbeagle/CPU/BeagleCPUAdjointSSE.hpp"

#endif // __BeagleCPUAdjointSSE__
