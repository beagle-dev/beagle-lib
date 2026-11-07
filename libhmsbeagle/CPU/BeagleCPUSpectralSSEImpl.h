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

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
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
    using Base::kBufferCount;
    using Base::kMatrixCount;
    using Base::kEigenDecompCount;
    using Base::kPartialsSize;
    using Base::gPartials;
    using typename Base::BranchEigenInfo;
    using Base::gBranchEigenInfo;

public:
    const char* getName() override;

    const long long getFlags() override;

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
                       long long preferenceFlags,
                       long long requirementFlags) override;

    int setPatternPartitions(int partitionCount, const int* inPatternPartitions) override;

    // Writes that invalidate the message and push-down caches (BEAGLE_SPECTRAL_CACHE=1)
    int setTipStates(int tipIndex, const int* inStates) override;

    int setTipPartials(int tipIndex, const double* inPartials) override;

    int setPartials(int bufferIndex, const double* inPartials) override;

    int setRootPrePartials(const int* bufferIndices, const int* stateFrequenciesIndices, int count) override;

    int setEigenDecomposition(int eigenIndex,
                              const double* inEigenVectors,
                              const double* inInverseEigenVectors,
                              const double* inEigenValues) override;

    int updateTransitionMatrices(int eigenIndex,
                                 const int* probabilityIndices,
                                 const int* firstDerivativeIndices,
                                 const int* secondDerivativeIndices,
                                 const double* edgeLengths,
                                 int count) override;

protected:
    // One operation at a time, recording its buffer and matrix indices for the kernels' caches
    int upPartials(bool byPartition,
                   const int* operations,
                   int count,
                   int cumulativeScaleIndex,
                   BeaglePartialsType partialsType) override;

    int upPrePartials(bool byPartition,
                      const int* operations,
                      int count,
                      int cumulativeScaleIndex,
                      BeaglePartialsType partialsType) override;

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

    // BEAGLE_SPECTRAL_CACHE=1: two caches keyed by partials buffer index, filled where the value is computed anyway.
    // - message: max(0, P_c x_c) of a child c, written by the post-order operation of its parent (and by a pre-order
    //   miss), read by the pre-order operation of its sibling;
    // - push-down: max(0, P_p^T top_p) of a parent p, written by the first TOP pre-order operation under p, read by
    //   the second.
    // An entry is used only if its stamp still matches: the write generation of the source partials (per pattern
    // partition), the matrix index and its generation, and the eigen index and its generation. Values are the same
    // arithmetic as without the cache, so results are bitwise identical. BEAGLE_SPECTRAL_CACHE_VERIFY=1 recomputes
    // every hit and counts mismatches; BEAGLE_SPECTRAL_CACHE_VERBOSE=1 reports the counts.
    struct CacheStamp {
        uint64_t partials = 0; // 0: empty
        uint64_t matrix = 0;
        uint64_t eigen = 0;
        int matrixIndex = -1;
        int eigenIndex = -1;
    };
    // The current operation of a pattern partition: post-order (child1, matrix1, child2, matrix2) or pre-order
    // (parent's pre-order buffer, parent's matrix, sibling, sibling's matrix)
    struct OpContext {
        int index1 = -1;
        int matrix1 = -1;
        int index2 = -1;
        int matrix2 = -1;
    };

    bool kUseSpectralCache = false;
    bool kVerifySpectralCache = false;
    bool kVerboseSpectralCache = false;
    std::atomic<uint64_t> gCacheCounter{0};
    std::atomic<int> kCacheBufferCount{-1}; // -1: not allocated, or emptied
    int kCacheMatrixCount = 0;
    int kCacheEigenCount = 0;
    int kCachePartitionCount = 0;
    std::mutex gCacheMutex;
    std::vector<uint64_t> gPartialsGeneration; // [buffer * partitions + partition]
    std::vector<uint64_t> gMatrixGeneration;   // [matrix]
    std::vector<uint64_t> gEigenGeneration;    // [eigen]
    // [buffer], kPartialsSize each, allocated on first use (under gCacheMutex: partition threads share a slot)
    struct CacheSlots {
        std::unique_ptr<std::atomic<double*>[]> pointers;
        std::vector<std::unique_ptr<double[]>> storage;
    };
    CacheSlots gMessageCache;
    CacheSlots gPushDownCache;
    std::vector<CacheStamp> gMessageStamp;  // [buffer * partitions + partition]
    std::vector<CacheStamp> gPushDownStamp; // [buffer * partitions + partition]
    std::vector<OpContext> gOpContext;      // [partition]
    std::atomic<long long> gMessageHits{0}, gMessageMisses{0}, gPushDownHits{0}, gPushDownMisses{0};
    std::atomic<long long> gCacheMismatches{0};

    // (Re)allocates the bookkeeping when buffer, matrix, eigen or partition counts change, emptying the caches
    void ensureCacheSizes();
    // A new write generation for a buffer: one pattern partition, or all of them (partition < 0)
    void bumpPartials(int buffer, int partition);
    CacheStamp makeStamp(int buffer, int partition, int matrixIndex, int eigenIndex) const;
    bool cacheValid(const CacheStamp& stamp, int buffer, int partition, int matrixIndex, int eigenIndex) const;
    double* cacheSlot(CacheSlots& cache, int buffer);
    void resetSlots(CacheSlots& cache, int bufferCount);

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
                                   long long preferenceFlags,
                                   long long requirementFlags,
                                   int* errorCode);

    virtual const char* getName();
    virtual const long long getFlags();
};

} // namespace cpu
} // namespace beagle

// now include the file containing template function implementations
#include "libhmsbeagle/CPU/BeagleCPUSpectralSSEImpl.hpp"

#endif // __BeagleCPUSpectralSSEImpl__
