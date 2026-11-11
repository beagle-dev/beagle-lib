/*
 * @file BeagleGPUSpectralImpl.h
 *
 * Copyright 2026 Phylogenetic Likelihood Working Group
 *
 * This file is part of BEAGLE.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 *
 * @brief GPU spectral implementation header
 *
 * @author Marc Suchard
 */

#ifndef __BeagleGPUSpectralImpl__
#define __BeagleGPUSpectralImpl__

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include "libhmsbeagle/GPU/BeagleGPUImpl.h"

namespace beagle {
namespace gpu {

#ifdef CUDA
    namespace cuda {
#elif defined(FW_TINYGPU)
    namespace tinygpu {
#else
    namespace opencl {
#endif

BEAGLE_GPU_TEMPLATE
class BeagleGPUSpectralImpl : public BeagleGPUImpl<Real> {
private:
    GPUPtr  dSpectralDistancesOrigin;
    int     kSpectralDistancePoolCapacity; // matrix slots in dSpectralDistancesOrigin; ensureBufferCounts may leave
                                           // more than kMatrixCount, and dSpectralDistances that many handles
    GPUPtr* dSpectralDistances;
    int*    hEigenIndexForMatrix;
    /* Per-matrix element stride within dSpectralDistancesOrigin, i.e.
     * AlignMemOffset(kCategoryCount * sizeof(Real)) / sizeof(Real). Device
     * alignment padding can make this larger than kCategoryCount, so any code
     * that indexes into the pooled origin buffer by matrix index (e.g. the
     * adjoint offset-queue) must multiply by this, not by kCategoryCount. */
    unsigned int kSpectralDistanceStrideElements;
    /* Host-side mirror of the first kMatrixCount * kSpectralDistanceStrideElements
     * elements of dSpectralDistancesOrigin (same per-matrix layout). After
     * ensureBufferCounts either buffer may be longer (the pool has
     * kSpectralDistancePoolCapacity slots), and only that kMatrixCount prefix
     * is uploaded by flushSpectralDistances. Kept
     * persistent across calls so updateTransitionMatrices can patch just the
     * touched matrices' entries and mark the mirror dirty; the first kernel
     * that reads distances afterwards pushes the whole pooled buffer in a
     * single MemcpyHostToDevice call (flushSpectralDistances), however many
     * updateTransitionMatrices calls (e.g. one per eigen index) came before. */
    Real*   hSpectralDistances;
    bool    hSpectralDistancesDirty;

    /* Backward eigenvector buffers for the parent branch in pre-order.
     * dEvecT[ei]  = U   stored row-major (dEvecT[j*S+k] = U[j,k]).
     *               evecT1 of the Growing kernels (backward, to the eigen basis: U^T·p).
     * dIevcT[ei]  = U^-1 stored row-major (dIevcT[j*S+k] = U^-1[j,k]).
     *               ievcT1 of the Growing kernels (backward, from the eigen basis: (U^-1)^T·q).
     * Allocated as one contiguous block each; sub-pointers per eigen index. */
    GPUPtr  dEvecTOrigin;
    GPUPtr* dEvecT;
    GPUPtr  dIevcTOrigin;
    GPUPtr* dIevcT;

    /* Adjoint gradient buffers.
     * dGradient: accumulated S×S gradient (one per eigen-decomposition).
     * hEigenDecompIsAllReal: per-eigen flag; set in setEigenDecomposition. */
    GPUPtr  dGradientOrigin;
    GPUPtr* dGradient;
    bool*   hEigenDecompIsAllReal;

    /* The adjoint's per-site log likelihoods (unscaled), sized like
     * dIntegrationTmp but apart from it: getSiteLogLikelihoods reads
     * dIntegrationTmp, the last likelihood calculation's (scaled) values.
     * Row 0 is the root's; rows 1.. (kAdjointSiteLogLkRows in all, each
     * kAdjointSiteLogLkStride Reals) hold the rescaled branches' own
     * (kernelAdjointScaledSiteLogLk), grown with the most branches rescaled
     * in one call. */
    GPUPtr  dAdjointSiteLogLk;
    int          kAdjointSiteLogLkRows;
    unsigned int kAdjointSiteLogLkStride;

    /* Offset-queue for the batched adjoint kernel: one 13-unsigned-int record
     * per branch (see STATUS.md for fields 0-8; 9-12 are the branch's row in
     * dAdjointSiteLogLk, 0 if not rescaled, and its post, pre and root scale
     * buffers' offsets in the scale pool), addressed via
     * integer offsets into the pooled partials/states/evec/ievc/eigenvalue/
     * distance/gradient origins — never per-branch device pointers. Grown
     * lazily since branch count varies per calculateAdjointCrossProducts
     * call and isn't known at createInstance time. */
    static const int kAdjointQueueFieldsPerBranch = 13;
    GPUPtr        dAdjointQueue;
    unsigned int* hAdjointQueue;
    int           kAdjointQueueCapacity;

    /* The low-rank adjoint (more than 4 states, at most ADJOINT_LOWRANK_MAX_PATTERNS patterns:
     * kernelAdjointLowRank{Prep,Slab,Reduce}N), all set up in calculateAdjointCrossProducts. hAdjointQueuePos: a call's
     * branches' records, stably sorted by eigen decomposition. dAdjointLowRankScratch: a chunk's slot records, then one
     * S x S plane per slab; grown only. */
    std::vector<int> hAdjointQueuePos;
    GPUPtr           dAdjointLowRankScratch;
    size_t           kAdjointLowRankScratchReals;

    /* BeagleGPUImpl::kEigenDecompCount is private to the base class; kept
     * here too so calculateAdjointCrossProducts can size a "seen ei" set
     * for the dGradient zero-fill without re-deriving it. */
    int kSpectralEigenDecompCount;

    /* Level-batched traversals (more than 4 states): updatePartials and updatePrePartials group a call's operations
     * into levels (scheduleLevels) and launch each level's operations of one kind as one kernel, from one upload per
     * call of their records (SPECTRAL_LEVEL_QUEUE_STRIDE, GPUImplDefs.h), at the pools' offsets as the adjoint queue.
     * Every upload is ordered after the launches before it, so one queue serves both traversals. */
    GPUPtr        dLevelQueue;
    unsigned int* hLevelQueue;
    int           kLevelQueueCapacity;   // records
    // scheduleLevels' input (per operation: the two buffers it reads, child 2 BEAGLE_OP_NONE in degree 2, and the one
    // it writes) and
    // its output (hLevelOrder: the operations by level, each level in call order from hLevelStart[level])
    std::vector<int> hLevelReads, hLevelWrites, hLevelOf, hLevelStart, hLevelOrder;
    std::vector<int> hLevelLastWrite, hLevelLastRead;   // per buffer index

public:
    BeagleGPUSpectralImpl();
    ~BeagleGPUSpectralImpl() override;

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

    int updatePrePartials(const int* operations,
                          int operationCount,
                          int cumulativeScaleIndex,
                          BeaglePartialsType partialsType) override;

    // level-batched with more than 4 states, MANUAL scaling (or none) and no operation writing scale factors; else
    // BeagleGPUImpl's, one launch per operation
    int updatePartials(const int* operations,
                       int operationCount,
                       int cumulativeScalingIndex,
                       BeaglePartialsType partialsType) override;

    int updatePrePartialsByPartition(const int* operations,
                                     int operationCount,
                                     BeaglePartialsType partialsType) override;

    char* getInstanceName() override;
    int getInstanceDetails(BeagleInstanceDetails* returnInfo) override;

    /* Paths that read transition matrices which spectral never computes (only setters write dense matrices here),
     * and pattern partitions, which the spectral dispatch does not honour: BEAGLE_ERROR_NO_IMPLEMENTATION. */
    int setPatternPartitions(int partitionCount,
                             const int* inPatternPartitions) override;

    int convolveTransitionMatrices(const int* firstIndices,
                                   const int* secondIndices,
                                   const int* resultIndices,
                                   int matrixCount) override;

    int transposeTransitionMatrices(const int* inputIndices,
                                    const int* resultIndices,
                                    int matrixCount) override;

    int updateTransitionMatricesWithModelCategories(int* eigenIndices,
                                                    const int* probabilityIndices,
                                                    const int* firstDerivativeIndices,
                                                    const int* secondDerivativeIndices,
                                                    const double* edgeLengths,
                                                    int count) override;

    int updateTransitionMatricesWithMultipleModels(const int* eigenIndices,
                                                   const int* categoryRateIndices,
                                                   const int* probabilityIndices,
                                                   const int* firstDerivativeIndices,
                                                   const int* secondDerivativeIndices,
                                                   const double* edgeLengths,
                                                   int count) override;

    int updatePartialsByPartition(const int* operations,
                                  int operationCount) override;

    int calculateEdgeLogLikelihoods(const int* parentBufferIndices,
                                    const int* childBufferIndices,
                                    const int* probabilityIndices,
                                    const int* firstDerivativeIndices,
                                    const int* secondDerivativeIndices,
                                    const int* categoryWeightsIndices,
                                    const int* stateFrequenciesIndices,
                                    const int* cumulativeScaleIndices,
                                    int count,
                                    double* outSumLogLikelihood,
                                    double* outSumFirstDerivative,
                                    double* outSumSecondDerivative) override;

    int calculateEdgeLogLikelihoodsByPartition(const int* parentBufferIndices,
                                               const int* childBufferIndices,
                                               const int* probabilityIndices,
                                               const int* firstDerivativeIndices,
                                               const int* secondDerivativeIndices,
                                               const int* categoryWeightsIndices,
                                               const int* stateFrequenciesIndices,
                                               const int* cumulativeScaleIndices,
                                               const int* partitionIndices,
                                               int partitionCount,
                                               int count,
                                               double* outSumLogLikelihoodByPartition,
                                               double* outSumLogLikelihood,
                                               double* outSumFirstDerivativeByPartition,
                                               double* outSumFirstDerivative,
                                               double* outSumSecondDerivativeByPartition,
                                               double* outSumSecondDerivative) override;

    int calculateAdjointCrossProducts(const int* postBufferIndices,
                                      const int* preBufferIndices,
                                      const int* eigenIndices,
                                      const int* categoryRatesIndices,
                                      const int* categoryWeightsIndices,
                                      const int  rootPostOrderIndex,
                                      const int  stateFrequenciesIndex,
                                      int        count,
                                      double*    outSumDerivatives,
                                      double*    outSumSquaredDerivatives,
                                      const int* postScaleIndices = nullptr,
                                      const int* preScaleIndices = nullptr,
                                      const int  cumulativeScaleIndex = BEAGLE_OP_NONE) override;

protected:
    // matrices hold eigen-systems; the dense pool exists only once a setter writes a matrix
    bool denseMatricesOnDemand() const override { return true; }

    // ensureBufferCounts: the distances and eigen indices by matrix
    int growDerivedHostArrays(int newMatrixCount) override;
    int growDerivedPools(int newMatrixCount) override;

    void dispatchPrunePP(GPUPtr p1, GPUPtr p2, GPUPtr p3,
                          int c1MatIdx, int c2MatIdx,
                          GPUPtr scalingFactors, GPUPtr cumulativeScaling,
                          unsigned int startPattern, unsigned int endPattern,
                          int rescale, int streamIndex, int waitIndex) override;
    void dispatchPruneSP(GPUPtr s1, GPUPtr p2, GPUPtr p3,
                          int c1MatIdx, int c2MatIdx,
                          GPUPtr scalingFactors, GPUPtr cumulativeScaling,
                          unsigned int startPattern, unsigned int endPattern,
                          int rescale, int streamIndex, int waitIndex) override;
    void dispatchPruneSS(GPUPtr s1, GPUPtr s2, GPUPtr p3,
                          int c1MatIdx, int c2MatIdx,
                          GPUPtr scalingFactors, GPUPtr cumulativeScaling,
                          unsigned int startPattern, unsigned int endPattern,
                          int rescale, int streamIndex, int waitIndex) override;

    // degree-2 operations are computed: post-order by dispatchPruneDegree2, pre-order in updatePrePartials
    int checkSingleChild(const int* operations, int operationCount, int numOps) override { return BEAGLE_SUCCESS; }
    void dispatchPruneDegree2(GPUPtr c1, bool c1IsStates, GPUPtr p3,
                              int c1MatIdx,
                              GPUPtr scalingFactors, GPUPtr cumulativeScaling,
                              unsigned int startPattern, unsigned int endPattern,
                              int rescale, int streamIndex, int waitIndex) override;

private:
    void flushSpectralDistances();

    void dispatchGrowingSpectral(bool isTop, bool isRoot,
                                  GPUPtr partials1, GPUPtr c2, bool sibIsStates,
                                  GPUPtr partials3,
                                  int c1MatIdx, int c2MatIdx);

    // one pre-order operation (7 ints), with its own launch or copy
    void dispatchPrePartialsOperation(const int* operation, bool isTop);

    // Gives each of count operations (hLevelReads and hLevelWrites, buffer indices checked by the caller) a level from
    // 0: one past every earlier operation of the call that writes a buffer it reads or writes, or reads the buffer it
    // writes. So a level's operations may run together, after the levels before it. Fills hLevelOf, hLevelStart and
    // hLevelOrder; returns the number of levels.
    int scheduleLevels(int count);

    // hLevelReads (child 1, child 2) and hLevelWrites (the destination) of count operations (7 ints, post- or
    // pre-order alike); BEAGLE_ERROR_OUT_OF_RANGE for a buffer index that is not one
    int setLevelBuffers(const int* operations, int count);

    // room for count records in hLevelQueue and dLevelQueue
    void ensureLevelQueue(int count);

    // the queue's offsets of a branch through matrix index matIndex: eigenvectors, eigenvalues, distances
    void fillLevelBranch(unsigned int* fields, int matIndex);
};

BEAGLE_GPU_TEMPLATE
class BeagleGPUSpectralImplFactory : public BeagleImplFactory {
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

} // namespace cuda/tinygpu/opencl
} // namespace gpu
} // namespace beagle

#include "libhmsbeagle/GPU/BeagleGPUSpectralImpl.hpp"

#endif // __BeagleGPUSpectralImpl__
