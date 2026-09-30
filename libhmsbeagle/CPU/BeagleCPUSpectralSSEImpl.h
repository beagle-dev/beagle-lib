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

#include <array>
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
    using Base::kPatternCount;
    using Base::kEigenDecompCount;
    using Base::gPartials;
    using Base::gTipStates;
    using Base::gPatternWeights;
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

    int setEigenDecomposition(int eigenIndex,
                              const double* inEigenVectors,
                              const double* inInverseEigenVectors,
                              const double* inEigenValues) override;

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

    // Adjoint gradient of the rate matrix in the eigen basis; with rescaled partials (scale indices) the scalar
    // implementation is used
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
    // Per partition: u, y and a third vector (z1 or w), each kSimdTmpStride doubles, room for a full last vector
    // when S is odd
    std::vector<double> gSimdTmp;
    int kSimdTmpStride;

    double* simdTmp(int currentPartition) { return gSimdTmp.data() + currentPartition * 3 * kSimdTmpStride; }

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
    int kAdjointStride;
    // Per partition: the outer product of several patterns, S rows of kAdjointStride
    std::vector<double> gAdjointOuterTmp;
    // Per partition: exp(a t) cos(b t) and exp(a t) sin(b t) of each conjugate pair, kAdjointPairTmpStride each
    std::vector<double> gAdjointPairTmp;
    int kAdjointPairTmpStride;

    void prepareAdjointPlan(int eigenIndex);

    // gradient += (outer product of the branch's pre- and post-order partials in the eigen basis) times the
    // integral kernel of the branch; View gives entries (l, r) and (l, r + 1) of that outer product
    template <typename View>
    void adjointKernel(double* gradient, const View& view, const AdjointPlan& plan, const double* eval,
                       const BranchEigenInfo& info, int infoOffset, double time, double* pairTmp);

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
