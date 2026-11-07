/*
 *  BeagleCPUSpectralImpl.h
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

#ifndef __BeagleCPUSpectralImp__
#define __BeagleCPUSpectralImp__

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include "libhmsbeagle/CPU/BeagleCPUImpl.h"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

#define TEST_EB

namespace beagle {
    namespace cpu {

        // Pre-order partials at the bottom of a branch, at its top, or at the top of a branch below the root
        enum class SpectralPreOrder { Bottom, Top, TopRoot };

        BEAGLE_CPU_TEMPLATE
        class BeagleCPUSpectralImpl : public BeagleCPUImpl<BEAGLE_CPU_GENERIC> {

        protected:
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::gEigenDecomposition;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::kPartialsSize;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::kFlags;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::kTipCount;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::kMatrixCount;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::kBufferCount;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::gPartials;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::integrationTmp;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::gTransitionMatrices;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::kPatternCount;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::kEigenDecompCount;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::kMatrixSize;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::kPaddedPatternCount;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::kTransPaddedStateCount;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::kPartialsPaddedStateCount;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::kExtraPatterns;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::kStateCount;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::gTipStates;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::kCategoryCount;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::gScaleBuffers;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::gStateFrequencies;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::gCategoryWeights;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::gCategoryRates;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::grandNumeratorDerivTmp;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::grandDenominatorDerivTmp;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::gPatternWeights;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::outLogLikelihoodsTmp;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::realtypeMin;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::scalingExponentThreshold;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::gPatternPartitionsStartPatterns;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::accumulateDerivatives;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcEdgeLogDerivatives;

            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::accumulateScaleFactors;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::rescalePartials;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::rescalePartialsByPartition;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::autoRescalePartials;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::removeScaleFactors;

            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::gActiveScalingFactors;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::gAutoScaleBuffers;

            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::kThreadingEnabled;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::kAutoPartitioningEnabled;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::kPartitionCount;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::gPatternScaleTmp;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::gOuterProductTmp;

            using typename BeagleCPUImpl<BEAGLE_CPU_GENERIC>::BranchEigenInfo;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::gBranchEigenInfo;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::gPartialTmp1;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::gPartialTmp2;

        public:
            // BeagleCPUSpectralImpl();
            ~BeagleCPUSpectralImpl() override;
            const char *getName() override;

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

            // int setCPUThreadCount(int threadCount) override;

            // Writes that invalidate the message and push-down caches (BEAGLE_FLAG_SPECTRAL_CACHE)
            int setTipStates(int tipIndex, const int* inStates) override;

            int setTipPartials(int tipIndex, const double* inPartials) override;

            int setPartials(int bufferIndex, const double* inPartials) override;

            int setRootPrePartials(const int* bufferIndices, const int* stateFrequenciesIndices, int count) override;

            int setEigenDecomposition(int eigenIndex,
                                      const double* inEigenVectors,
                                      const double* inInverseEigenVectors,
                                      const double* inEigenValues) override;

            int setPatternPartitions(int partitionCount, const int* inPatternPartitions) override;

        protected:
            // BEAGLE_FLAG_SPECTRAL_CACHE: two caches keyed by partials buffer index, filled where the value is
            // computed anyway, for every spectral kernel family.
            // - message: max(0, P_c x_c) of a child c, written by the post-order operation of its parent (and by a
            //   pre-order miss), read by the pre-order operation of its sibling;
            // - push-down: max(0, P_p^T top_p) of a parent p, written by the first TOP pre-order operation under p,
            //   read by the second.
            // An entry is used only while its stamp matches: the write generation of the source partials (per
            // pattern partition), the matrix index and its generation, and the eigen index and its generation. The
            // kernels compute the cached values with the same arithmetic, so results are bitwise identical.
            // BEAGLE_SPECTRAL_CACHE_VERIFY=1 recomputes every hit and counts mismatches;
            // BEAGLE_SPECTRAL_CACHE_VERBOSE=1 reports the counts after each pre-order update.
            struct CacheStamp {
                uint64_t partials = 0; // 0: empty
                uint64_t matrix = 0;
                uint64_t eigen = 0;
                int matrixIndex = -1;
                int eigenIndex = -1;
            };
            // The current operation of a pattern partition: post-order (child1, matrix1, child2, matrix2) or
            // pre-order (parent's pre-order buffer, parent's matrix, sibling, sibling's matrix)
            struct OpContext {
                int index1 = -1;
                int matrix1 = -1;
                int index2 = -1;
                int matrix2 = -1;
            };
            // [buffer], kPartialsSize each, allocated on first use under gCacheMutex (partition threads share one)
            struct CacheSlots {
                std::unique_ptr<std::atomic<REALTYPE*>[]> pointers;
                std::vector<std::unique_ptr<REALTYPE[]>> storage;
            };
            // One post-order kernel call: each child's message goes to message1 / message2 + v (null: not cached)
            struct PostOrderCache {
                REALTYPE* message1 = nullptr;
                REALTYPE* message2 = nullptr;
                int buffer1 = -1;
                int buffer2 = -1;
            };
            // One pre-order kernel call: message / pushDown, when not null, is a valid entry to use; otherwise the
            // kernel computes the vector and passes it to computedMessage / computedPushDown, which store it (or,
            // verifying a hit, compare it with the entry)
            struct PreOrderCache {
                const REALTYPE* message = nullptr;
                const REALTYPE* pushDown = nullptr;
                REALTYPE* messageSlot = nullptr;
                REALTYPE* pushDownSlot = nullptr;
                int siblingBuffer = -1;
                int parentBuffer = -1;
                bool messageHit = false;
                bool pushDownHit = false;
                bool messageStored = false;  // a kernel that does not record a vector leaves its entry unstamped
                bool pushDownStored = false;
                long long mismatches = 0;

                void computedMessage(int v, const REALTYPE* values, int n) {
                    record(messageSlot, messageHit, messageStored, v, values, n);
                }
                void computedPushDown(int v, const REALTYPE* values, int n) {
                    record(pushDownSlot, pushDownHit, pushDownStored, v, values, n);
                }

            private:
                void record(REALTYPE* slot, bool hit, bool& stored, int v, const REALTYPE* values, int n) {
                    if (slot == nullptr) {
                        return;
                    }
                    if (hit) { // verifying
                        mismatches += (std::memcmp(slot + v, values, n * sizeof(REALTYPE)) != 0);
                    } else {
                        std::memcpy(slot + v, values, n * sizeof(REALTYPE));
                        stored = true;
                    }
                }
            };

            bool kUseSpectralCache = false;
            bool kVerifySpectralCache = false;
            bool kVerboseSpectralCache = false;

            PostOrderCache beginPostOrderCache(int branchEigenIndex1, int branchEigenIndex2, int partition);
            void endPostOrderCache(const PostOrderCache& cache, int branchEigenIndex1, int branchEigenIndex2,
                                   int partition);
            PreOrderCache beginPreOrderCache(SpectralPreOrder type, bool hasSibling, int branchEigenIndex1,
                                             int branchEigenIndex2, int partition);
            void endPreOrderCache(const PreOrderCache& cache, int branchEigenIndex1, int branchEigenIndex2,
                                  int partition);

            // TOP pre-order partials of one pattern and category from the caches: max(P_p^T top, 0) * max(P_s x, 0),
            // each factor from its cache entry or computed alone, by the arithmetic of the pair it replaces
            template <typename Sibling>
            void prePartialsTopCached(REALTYPE* dest, ptrdiff_t offset,
                                      const REALTYPE* parentPartials, const BranchEigenInfo& info1,
                                      const REALTYPE* inverseEigenVectors1, const REALTYPE* eigenVectors1,
                                      const REALTYPE* siblingPartials, int siblingState,
                                      const BranchEigenInfo& info2,
                                      const REALTYPE* inverseEigenVectors2, const REALTYPE* eigenVectors2,
                                      int matrixIncr, int catOffset, PreOrderCache& cache, int partition);

        private:
            std::atomic<uint64_t> gCacheCounter{0};
            std::atomic<int> kCacheBufferCount{-1}; // -1: not allocated, or emptied
            int kCacheMatrixCount = 0;
            int kCacheEigenCount = 0;
            int kCachePartitionCount = 0;
            std::mutex gCacheMutex;
            std::vector<uint64_t> gPartialsGeneration; // [buffer * partitions + partition]
            std::vector<uint64_t> gMatrixGeneration;   // [matrix]
            std::vector<uint64_t> gEigenGeneration;    // [eigen]
            CacheSlots gMessageCache;
            CacheSlots gPushDownCache;
            std::vector<CacheStamp> gMessageStamp;  // [buffer * partitions + partition]
            std::vector<CacheStamp> gPushDownStamp; // [buffer * partitions + partition]
            std::vector<OpContext> gOpContext;      // [partition]
            std::vector<REALTYPE> gCacheScratch;     // [partition], three vectors for prePartialsTopCached
            std::atomic<long long> gMessageHits{0}, gMessageMisses{0}, gPushDownHits{0}, gPushDownMisses{0};
            std::atomic<long long> gCacheMismatches{0};

            // (Re)allocates the bookkeeping when buffer, matrix, eigen or partition counts change, emptying the caches
            void ensureCacheSizes();
            // A new write generation for a buffer: one pattern partition, or all of them (partition < 0)
            void bumpPartials(int buffer, int partition);
            // Records an operation's buffers and matrices for its partition's kernels, or clears them (operation null)
            void setOpContext(int partition, const int* operation);
            CacheStamp makeStamp(int buffer, int partition, int matrixIndex) const;
            bool cacheValid(const CacheStamp& stamp, int buffer, int partition, int matrixIndex) const;
            REALTYPE* cacheSlot(CacheSlots& cache, int buffer);
            void resetSlots(CacheSlots& cache, int bufferCount);

        protected:
            EigenDecomposition<BEAGLE_CPU_EIGEN_GENERIC>* createEigenDecomposition(
                    int decompositionCount,
                    int stateCount,
                    int categoryCount,
                    long long flags) override;

        private:
            int kStateCountModFour;

        protected:
            int updateTransitionMatrices(int eigenIndex,
                                         const int* probabilityIndices,
                                         const int* firstDerivativeIndices,
                                         const int* secondDerivativeIndices,
                                         const double* edgeLengths,
                                         int count) override;

            int getTransitionMatrix(int matrixIndex,
                                    double* outMatrix) override;

            // a matrix buffer holds a branch's eigen info; dense storage exists only once the index is written densely
            bool denseMatricesOnDemand() const override { return true; }

            int upPartials(bool byPartition,
                           const int* operations,
                           int operationCount,
                           int cumulativeScalingIndex,
                           BeaglePartialsType partialsType) override;

            int upPrePartials(bool byPartition,
                              const int* operations,
                              int count,
                              int cumulativeScaleIndex,
                              BeaglePartialsType preorderType) override;

            template <typename T>
            int upPrePartialsImpl(bool byPartition,
                                  const int* operations,
                                  int count,
                                  int cumulativeScaleIndex);

            // One call per operation from upPartials and upPrePartialsImpl, so a subclass (e.g. SIMD) can replace
            // the kernels and keep the operation decoding and scaling. scaleFactors is nullptr unless fixed scale
            // factors are applied. For degree-2 nodes one of states1 and partials1 is nullptr. For
            // SpectralPreOrder::TopRoot, branchEigenIndex1 is negative and partials1 is the root prior.
            virtual void spectralPartialsPartials(REALTYPE *destP,
                                                  const REALTYPE *partials1, const int branchEigenIndex1,
                                                  const REALTYPE *partials2, const int branchEigenIndex2,
                                                  const REALTYPE *scaleFactors,
                                                  int startPattern, int endPattern, int currentPartition);

            virtual void spectralStatesPartials(REALTYPE *destP,
                                                const int *states1, const int branchEigenIndex1,
                                                const REALTYPE *partials2, const int branchEigenIndex2,
                                                const REALTYPE *scaleFactors,
                                                int startPattern, int endPattern, int currentPartition);

            virtual void spectralStatesStates(REALTYPE *destP,
                                              const int *states1, const int branchEigenIndex1,
                                              const int *states2, const int branchEigenIndex2,
                                              const REALTYPE *scaleFactors,
                                              int startPattern, int endPattern, int currentPartition);

            virtual void spectralDegree2Partials(REALTYPE *destP,
                                                 const int *states1, const REALTYPE *partials1,
                                                 const int branchEigenIndex1,
                                                 const REALTYPE *scaleFactors,
                                                 int startPattern, int endPattern, int currentPartition);

            virtual void spectralPrePartialsPartials(REALTYPE *destP, SpectralPreOrder type,
                                                     const REALTYPE *partials1, const int branchEigenIndex1,
                                                     const REALTYPE *partials2, const int branchEigenIndex2,
                                                     int startPattern, int endPattern, int currentPartition);

            virtual void spectralPrePartialsStates(REALTYPE *destP, SpectralPreOrder type,
                                                   const REALTYPE *partials1, const int branchEigenIndex1,
                                                   const int *states2, const int branchEigenIndex2,
                                                   int startPattern, int endPattern, int currentPartition);

            template <typename T>
            void calcPartialsPartials(REALTYPE *destPartials,
                                      const REALTYPE *partials1,
                                      const int branchEigenIndex1,
                                      const REALTYPE *partials2,
                                      const int branchEigenIndex2,
                                      const REALTYPE* scaleFactors,
                                      int startPattern,
                                      int endPattern,
                                      int currentPartition);

            template <typename T>
            void calcStatesStates(REALTYPE *destP,
                                  const int *states1,
                                  const int branchEigenIndex1,
                                  const int *states2,
                                  const int branchEigenIndex2,
                                  const REALTYPE* scaleFactors,
                                  int startPattern,
                                  int endPattern,
                                  int currentPartition);

            template <typename T>
            void calcStatesPartials(REALTYPE *destP,
                                    const int *states1,
                                    const int branchEigenIndex1,
                                    const REALTYPE *partials2,
                                    const int branchEigenIndex2,
                                    const REALTYPE* scaleFactors,
                                    int startPattern,
                                    int endPattern,
                                    int currentPartition);

            template <typename T, typename V>
            void calcPrePartialsPartials(REALTYPE *destP,
                                         const REALTYPE *partialsParent,
                                         const int branchEigenIndex1,
                                         const REALTYPE *partials2,
                                         const int branchEigenIndex2,
                                         int startPattern,
                                         int endPattern,
                                         int currentPartition);

            template <typename T, typename V>
            void calcPrePartialsStates(REALTYPE *destP,
                                       const REALTYPE *partials1,
                                       const int branchEigenIndex1,
                                       const int *states2,
                                       const int branchEigenIndex2,
                                       int startPattern,
                                       int endPattern,
                                       int currentPartition);

            // Degree-2 (single-child) nodes, signalled by a second child (post-order) or sibling (pre-order) of
            // BEAGLE_OP_NONE. First is States or Partials; T is NoScaling or WithScaling.
            template <typename First, typename T>
            void calcDegree2Partials(REALTYPE *destP,
                                     const int *states1,
                                     const REALTYPE *partials1,
                                     const int branchEigenIndex1,
                                     const REALTYPE* scaleFactors,
                                     int startPattern,
                                     int endPattern,
                                     int currentPartition);

            // Pre-order partials through the single branch branchEigenIndex1 (transposed); a copy of partials1
            // when branchEigenIndex1 is negative (top partials below the root).
            virtual void calcDegree2PrePartials(REALTYPE *destP,
                                        const REALTYPE *partials1,
                                        const int branchEigenIndex1,
                                        int startPattern,
                                        int endPattern,
                                        int currentPartition);


            // template <typename T>
            // void calcPrePartialsPartialsRoot(REALTYPE *destP,
            //                                  const REALTYPE *partialsParent,
            //                                  const int branchEigenIndexSelf,
            //                                  const REALTYPE *partialsSibling,
            //                                  const int branchEigenIndexSibling,
            //                                  int startPattern,
            //                                  int endPattern);

            // template <typename T>
            // void calcPrePartialsStatesRoot(REALTYPE *destP,
            //                                const REALTYPE *partials1,
            //                                const int branchEigenIndex1,
            //                                const int *states2,
            //                                const int branchEigenIndex2,
            //                                int startPattern,
            //                                int endPattern);

            template <typename First, typename Second>
            void expScaledMatrixVectorMultiple(REALTYPE* outPartials1, REALTYPE* outPartials2,
                                               const REALTYPE* inPartials1, const int state1,
                                               const REALTYPE* eigenValuesReal1,
                                               const REALTYPE* eigenValuesImag1,
                                               const REALTYPE* inverseEigenVectors1,
                                               const REALTYPE scaledBranchLength1,
                                               const REALTYPE* inPartials2, const int state2,
                                               const REALTYPE* eigenValuesReal2,
                                               const REALTYPE* eigenValuesImag2,
                                               const REALTYPE* inverseEigenVectors2,
                                               const REALTYPE scaledBranchLength2,
                                               const int matrixIncr);

            template <typename T, typename Direction>
            void expScaledMatrixVector(REALTYPE* outPartials,
                                       const REALTYPE* inPartials, const int state,
                                       const BranchEigenInfo& info,
                                       const REALTYPE* inverseEigenVectors,
                                       const int matrixIncr,
                                       const int catOffset);

#ifdef TEST_EB
            template <typename First, typename Second, typename Direction>
            void expScaledMatrixVectorMultiple2(REALTYPE* outPartials1, REALTYPE* outPartials2,
                                               const REALTYPE* inPartials1, const int state1,
                                               const BranchEigenInfo& info1,
                                               const REALTYPE* inverseEigenVectors1,
                                               const REALTYPE* inPartials2, const int state2,
                                               const BranchEigenInfo& info2,
                                               const REALTYPE* inverseEigenVectors2,
                                               const int matrixIncr,
                                               const int catOffset);
            template <typename First, typename Second, typename Direction>
            void expScaledMatrixVectorMultiple3(REALTYPE* outPartials1, REALTYPE* outPartials2,
                                               const REALTYPE* inPartials1, const int state1,
                                               const BranchEigenInfo& info1,
                                               const REALTYPE* inverseEigenVectors1,
                                               const REALTYPE* inPartials2, const int state2,
                                               const BranchEigenInfo& info2,
                                               const REALTYPE* inverseEigenVectors2,
                                               const int matrixIncr);
#endif
        };

        BEAGLE_CPU_FACTORY_TEMPLATE
        class BeagleCPUSpectralImplFactory : public BeagleImplFactory
        {
        public:
            virtual BeagleImpl *createImpl(int tipCount,
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
                                           int *errorCode);

            virtual const char *getName();
            virtual const long long getFlags();
        };

        struct NoRotation {};

        struct Top {};
        struct Bottom {};

        struct Forward {};
        struct Backward {};

        struct NoScaling {
            static constexpr bool useScaleFactors = false;
        };

        struct WithScaling {
            static constexpr bool useScaleFactors = true;
        };

        struct Root { };
        struct NotRoot { };
        struct NotUsed { };

    } // namespace cpu
} // namespace beagle

// now include the file containing template function implementations
#include "libhmsbeagle/CPU/BeagleCPUSpectralImpl.hpp"

#endif // __BeagleCPUSpectralImp__
