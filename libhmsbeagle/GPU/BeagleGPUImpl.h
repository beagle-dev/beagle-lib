/*
 * @file BeagleGPUImpl.h
 *
 * Copyright 2009 Phylogenetic Likelihood Working Group
 *
 * This file is part of BEAGLE.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 *
 * @brief GPU implementation header
 *
 * @author Marc Suchard
 * @author Andrew Rambaut
 * @author Daniel Ayres
 */

#ifndef __BeagleGPUImpl__
#define __BeagleGPUImpl__

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include <map>
#include <vector>
#include <cstdint>

#include "libhmsbeagle/BeagleImpl.h"
#include "libhmsbeagle/GPU/GPUImplDefs.h"
#include "libhmsbeagle/GPU/GPUInterface.h"
#include "libhmsbeagle/GPU/KernelLauncher.h"

#define BEAGLE_GPU_GENERIC	Real
#define BEAGLE_GPU_TEMPLATE template <typename Real>

#ifdef CUDA
	using namespace cuda_device;
#elif defined(FW_TINYGPU)
	using namespace tinygpu_device;
#else
	using namespace opencl_device;
#endif

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
class BeagleGPUImpl : public BeagleImpl {


protected:
    long long kFlags;

    int kStateCount;
    int kPaddedStateCount;     //   * kPaddedStateCount is a multiple of 16

    GPUPtr* dStates;
    GPUPtr* dPartials;

    GPUPtr  dIntegrationTmp;
    int     kResultPaddedPatterns; // dIntegrationTmp holds kPaddedPatternCount + kResultPaddedPatterns
    GPUPtr  dPatternWeights;
    GPUPtr* dWeights;
    GPUPtr* dFrequencies;

private:
    int kInitialized;

    int kTipCount;
    int kPartialsBufferCount;
    int kCompactBufferCount;
    int kPatternCount;
    int kEigenDecompCount;

    int kTipPartialsBufferCount;
    int kInternalPartialsBufferCount;
    int kBufferCount;
    int kScaleBufferCount;

    int kSumSitesBlockCount;

    int kPartialsSize;
    int kMatrixSize;
    int kEigenValuesSize;
    int kScaleBufferSize;

    int kLastCompactBufferIndex;
    int kLastTipPartialsBufferIndex;

	int kSumIntervalBlockSize;
	int kSlabOpsPerBlock;
	int kSumAcrossBlockSize;
    GPUPtr dOutFirstDeriv;
    GPUPtr dOutSecondDeriv;
    GPUPtr dPartialsTmp;
    GPUPtr dFirstDerivTmp;
    GPUPtr dSecondDerivTmp;

    GPUPtr dSumLogLikelihood;
    GPUPtr dSumFirstDeriv;
    GPUPtr dSumSecondDeriv;

	GPUPtr dMultipleDerivatives;
	GPUPtr dMultipleDerivativeSum;

    GPUPtr dDistanceQueue;
    int kDistanceQueueLength; // Reals in dDistanceQueue and hDistanceQueue

    GPUPtr dPtrQueue;
#ifdef FW_TINYGPU
    GPUPtr dZeroWord; // one unsigned int 0: a one-index queue for a scale buffer passed as the kernel's origin
#endif
    int kPtrQueueLength;      // entries in dPtrQueue and hPtrQueue

	GPUPtr dDerivativeQueue;

    GPUPtr dMaxScalingFactors;
    GPUPtr dIndexMaxScalingFactors;

    GPUPtr dAccumulatedScalingFactors;

    GPUPtr* dScalingFactors;
    GPUPtr dScalingFactorsOrigin; // the pool behind dScalingFactors; 0 with CUDA dynamic scaling (one buffer each)
    int kScalePoolCapacity;          // slots in that pool, with the zero buffer of raw scalers; 0 while there is none
    unsigned int kScaleBufferStride; // elements between consecutive scale buffers in that pool
    GPUPtr scalingFactorsOrigin() const { // what kernels offset by scale index
        return dScalingFactorsOrigin != (GPUPtr)NULL ? dScalingFactorsOrigin : dScalingFactors[0];
    }

    GPUPtr* dMatrices;
    GPUPtr dMatricesOrigin;           // 0 until the first dense write when denseMatricesOnDemand()
    int kMatrixPoolCapacity;          // matrix slots in dMatricesOrigin; 0 while there is no pool
    std::vector<bool> hDenseWritten;  // per matrix index, when denseMatricesOnDemand()
    std::vector<bool> hActiveScalingFactors; // SCALING_AUTO, per internal buffer: its last operation wrote its factors

    // Every sub-buffer made by createSubPointer, by parent; per-index arrays (dPartials, dStates, ...) alias each
    // other and are swapped by reorderPatternsByPartition, so they cannot tell what to release
    std::map<GPUPtr, std::vector<GPUPtr> > hSubPointers;
    
    GPUPtr* dCompactBuffers;
    GPUPtr* dTipPartialsBuffers;

    bool kUsingMultiGrid;
    bool kDerivBuffersInitialised;
	int kMultipleDerivativesLength; // TODO Change to length (max node count used)
	bool kUsingAutoTranspose;

    int kNumPatternBlocks;
    int kSitesPerBlock;
    int kSitesPerIntegrateBlock;
    int kSumSitesBlockSize;
    size_t kOpOffsetsSize;
    unsigned int kIndexOffsetPat;
    unsigned int kIndexOffsetStates;
    unsigned int kIndexOffsetMat;
    unsigned int kEvecOffset;
    unsigned int kEvalOffset;
    unsigned int kWeightsOffset;
    unsigned int kFrequenciesOffset;
    GPUPtr  dPartialsPtrs;
    // GPUPtr  dPartitionOffsets;
    GPUPtr  dPatternsNewOrder;
    GPUPtr  dTipOffsets;
    GPUPtr  dTipTypes;
    GPUPtr  dPartialsOrigin;
    int kPartialsPoolCapacity;  // slots in dPartialsOrigin, which may exceed kPartialsBufferCount (pattern reorder)
    int kPartialsHandleCount;   // entries in dPartials and hPartialsOffsets, which may exceed kBufferCount
    GPUPtr  dStatesOrigin;
    GPUPtr  dStatesSortOrigin;
    GPUPtr  dPatternWeightsSort;
    GPUPtr* dStatesSort;
    unsigned int* hPartialsPtrs;
    unsigned int* hPartitionOffsets;
    unsigned int* hIntegratePartitionOffsets;
    unsigned int* hPartialsOffsets;
    unsigned int* hStatesOffsets;
    int* hTipOffsets;
    BeagleDeviceImplementationCodes kDeviceCode;
    long long kDeviceType;
    int kPartitionCount;
    int kMaxPartitionCount;
    int kPaddedPartitionBlocks;
    int kMaxPaddedPartitionBlocks;
    int kPaddedPartitionIntegrateBlocks;
    int kMaxPaddedPartitionIntegrateBlocks;
    bool kPartitionsInitialised;
    bool kPatternsReordered;
    int* hPatternPartitions;
    int* hPatternPartitionsStartPatterns;
    int* hPatternPartitionsStartBlocks;
    int* hIntegratePartitionsStartBlocks;
    int* hPatternsNewOrder;
    int* hGridOpIndices;

    int kExtraMatrixCount;

    unsigned int* hPtrQueue;

	unsigned int* hDerivativeQueue;

    Real* hPatternWeightsCache;

    Real* hDistanceQueue;

    Real* hWeightsCache;
    Real* hFrequenciesCache;
    Real* hLogLikelihoodsCache;
    Real* hPartialsCache;
    int* hStatesCache;
    Real* hMatrixCache;

    int* hRescalingTrigger;
    GPUPtr dRescalingTrigger;

    GPUPtr* dScalingFactorsMaster;

    int* hStreamIndices;

	GPUPtr dBastaMemory;
	GPUPtr dBastaLogL;
	GPUPtr dBastaDistance;
	GPUPtr dBastaOperationQueue;
	GPUPtr dCoalescentBuffers;
	int kCoalescentBufferLength;
	int kCoalescentBufferCount;
	int kBastaIntervalBlockCount;

	int* hBastaOperationQueue;
	Real* hBastaLogL;
	Real* hBastaDistance;
	Real* hBastazeroes;

	GPUPtr dEdgeLengthsGrad;

	Real* hEdgeLengthsGrad;
	Real* hGradZeros;
	size_t kGradZerosSize;

	bool kGradBuffersAllocated;
	bool kBastaBuffersAllocated; // allocateBastaBuffers or allocateBastaGradBuffers has run: ensureBufferCounts refuses to grow
	bool kBastaOpsUploaded;
	int  kBastaOpsCount;

	// the two population-size buffers, each [sizes[0..R-1] | integrals[0..R-1]] (empty until set), and the current
	// one; kept on the host, in the instance's precision
	std::vector<Real> hBastaPopulationSizes[2];
	int kCurrentPopulationSizeBuffer;

	// _v5: the current buffer's sizes and integrals as a row of kPaddedStateCount per intervalNumber (padding 0), the
	// rows of the kernels' sizes and integrals arguments; allocated by the first _v5 call, and again when
	// kCoalescentBufferLength changes (kBastaRowsLength_v5: the length they have); hBastaRows_v5 stages either table,
	// hBastaOffsets_v5 is each intervalNumber's offset (-1: no operation uses it)
	GPUPtr dBastaSizeRows_v5;
	GPUPtr dBastaIntegralRows_v5;
	int kBastaRowsLength_v5;
	std::vector<Real> hBastaRows_v5;
	std::vector<int> hBastaOffsets_v5;

	GPUPtr dAdjointPopSizeGrad;
	GPUPtr dHazardAdjoints;
	GPUPtr dTransformedAdjoints;

	// dRawEvec, dRawIevc, dEvecT and dInverseEvecT: NULL until ensureBastaEigenBuffers
	GPUPtr* dRawEvec;
	GPUPtr* dRawIevc;
	GPUPtr  dRawEvecOrigin;
	GPUPtr  dRawIevcOrigin;
	GPUPtr dAdjointIntervalNumbers;        // [intervalCount]    intervalNumber per interval (used by ComputeHazardAdjoints)
	GPUPtr dAdjointIntervalStarts;         // [intervalCount + 1] op range per interval (used by gather kernel)
	GPUPtr dAdjointMatTransIndices;        // [intervalCount]    destination matrix offset per interval
	GPUPtr dAdjointCoalOps;                // [intervalCount]    coalescent op index per interval, -1 if none
	int* hAdjointIntervalNumbers;
	int* hAdjointCoalOps;
	int* hAdjointMatTransIndices;
	int* hAdjointIntervalStarts;
	int kAdjointIntervalNumbersSize;
	Real* hTransformedAdjointsHost;

	Real* hPopSizeGradHost;
	GPUPtr dLoewnerEigenValues;
	GPUPtr dLoewnerBranchLengths;
	GPUPtr dLoewnerBlockStarts;
	GPUPtr dLoewnerBlockDims;
	GPUPtr dLoewnerOutRateGrad;
	Real* hLoewnerEigenValues;
	Real* hLoewnerBranchLengths;
	int* hLoewnerBlockStarts;
	int* hLoewnerBlockDims;
	int kLoewnerNumBlocks;
	int kLoewnerEvalSize;
	int kLoewnerMaxM;
	bool kAdjointGradBuffersAllocated;

	bool kAdjointGradMode;
	int kActiveEigenIndexFwd;

	GPUPtr dScratchYBar;            // [operationCount * PADDED_STATE_COUNT]
	GPUPtr dScratchX;               // same shape, used by legacy pipeline only
	GPUPtr dCoalRightYBar;          // [intervalCount * PADDED_STATE_COUNT], legacy pipeline only
	GPUPtr dCoalRightX;             // same shape, legacy pipeline only
	int kScratchOpCapacity;      // allocated op slots for dScratchYBar
	int kScratchXOpCapacity;     // allocated op slots for dScratchX (legacy only)
	int kCoalRightCapacity;      // current capacity of coalRight buffers, in interval slots
	int kAdjointMaxOpsPerInterval;

	uint64_t kAdjointMetaFingerprint;     // hash over (hBastaOperationQueue, intervalStarts)
	bool kAdjointMetaCached;
	int kAdjointMetaIntervalCount;

	uint64_t kLoewnerMetaFingerprint;     // hash over eigenvalues
	bool kLoewnerMetaCached;
	int kLoewnerMetaEvalSize;

	void* kAdjointGraphExec;
	bool kAdjointGraphValid;
	uint64_t kAdjointGraphFingerprint;
	int kAdjointGraphIntervalCount;
	int kAdjointGraphAnchor;
	bool kSlabAnchorActive;
	int kAdjointGraphNumEvBlocks;    // kSlabNumEvBlocks at capture time
	long kAdjointGraphReplayCount;    // diagnostics
	long kAdjointGraphRebuildCount;
	long kAdjointGraphFallbackCount;

	bool kCfgUseForwardSlab;
	bool kCfgUseAdjointSlab;
	bool kForwardSlabPipelineEnabled;
	void* kForwardSlabGraphExec;
	bool kForwardSlabGraphValid;
	uint64_t kForwardSlabGraphFingerprint;
	int kForwardSlabGraphIntervalCount;
	int kForwardSlabGraphNumEvBlocks;
	long kForwardSlabGraphReplayCount;
	long kForwardSlabGraphRebuildCount;
	long kForwardSlabGraphFallbackCount;


	bool kAdjointSlabPipelineEnabled;
	GPUPtr dPartialsTilde;
	size_t kPartialsTildeBytes;

	GPUPtr dGEigen;

	GPUPtr* dEvecT;
	GPUPtr* dInverseEvecT;
	GPUPtr  dEvecTOrigin;
	GPUPtr  dInverseEvecTOrigin;

	GPUPtr dBranchKb;
	GPUPtr dBranchKTop;
	GPUPtr dBranchTopBuf;
	GPUPtr dBranchBotBuf;
	GPUPtr dBranchOpFirst;
	GPUPtr dBranchTimeStart;
	GPUPtr dBranchT;

	GPUPtr dCoalDestBufs;
	GPUPtr dCoalLeftAccBufs;
	GPUPtr dCoalRightAccBufs;
	GPUPtr dCoalIntervals;

	int kMaxSlabDepth;
	int* hBranchSlabStart;
	int* hCoalSlabStart;


	GPUPtr dOpInBufOff;


	GPUPtr dOpKIn;
	GPUPtr dOpKAcc;
	GPUPtr dOpHasAcc;
	GPUPtr dOpReduceTable;
	int kBastaOpReduceCount;

	GPUPtr dIntervalOpStartCSR;
	GPUPtr dIntervalOpListCSR;


	GPUPtr dHazardEigenPerOp;
	size_t kHazardEigenPerOpBytes;

	GPUPtr dSlabBlockBranchIdx;
	GPUPtr dSlabBlockChunkStart;
	GPUPtr dSlabBlockChunkLen;
	GPUPtr dSlabBlockChunkIdx;


	GPUPtr dBranchFirstBlock;
	GPUPtr dBranchSlabList;
	GPUPtr dSlabCarryOut;
	GPUPtr dSlabCarryPrefix;
	GPUPtr dSlabAlpha;
	GPUPtr dSlabCtaCarry;
	GPUPtr dSlabCtaCarryBranch;
	GPUPtr dSlabAStash;

	GPUPtr dSlabYBottomEigen;
	size_t kSlabYBottomBytes;


	GPUPtr dIntervalBranchLengths;
	size_t kIntervalBLBytes;
	size_t kIntervalOpStartBytes;
	Real*  hIntervalBranchLengths;

	GPUPtr dLoewnerExp;
	size_t kLoewnerExpBytes;


	int kSlabBlockCap;
	size_t kSlabCarryBytes;
	size_t kSlabAStashBytes;


	int* hSlabBlockBranchIdx;
	int* hSlabBlockChunkStart;
	int* hSlabBlockChunkLen;
	int* hSlabBlockChunkIdx;
	int* hSlabBlockStart;
	int* hBranchFirstBlock;

	GPUPtr dForwardBufList;
	int kForwardBufCount;

	// Capacity tracking
	int kSlabBranchCount;
	int kSlabCoalCount;
	int kSlabBranchListCap;
	int kSlabBranchTimeCap;
	int kSlabSlabCap;
	int kSlabCtaCarryCap;
	int kSlabOpCap;
	int kSlabOpReduceCap;
	int kSlabForwardBufCap;


	int* hBranchKb;
	int* hBranchKTop;
	int* hBranchTopBuf;
	int* hBranchBotBuf;
	int* hBranchOpFirst;
	int* hBranchTimeStart;
	Real* hBranchT;
	int* hCoalDestBufs;
	int* hCoalLeftAccBufs;
	int* hCoalRightAccBufs;
	int* hCoalIntervals;
	int* hBranchSlabList;
	int* hOpInBufOff;
	int* hOpKIn;
	int* hOpKAcc;
	int* hOpHasAcc;
	int* hOpReduceTable;
	int* hIntervalOpStartCSR;
	int* hIntervalOpListCSR;
	int* hForwardBufList;


	double** hStashedEval;
	int*  kStashedEvalSize;
	bool* kStashedEvalHasComplex;
	bool kLoewnerInfoBootstrapped;


	int kSlabBranchCountActive;
	int kSlabCoalCountActive;
	int kSlabBranchOpTotalActive;
	int kSlabNumEvBlocks;


	bool kBastaSlabMetadataUploaded;
	bool kBastaPipelineAnnounced;


	void* kForwardGraphExec;
	bool kForwardGraphValid;
	uint64_t kForwardGraphFingerprint;
	int kForwardGraphIntervalCount;
	int kForwardGraphOperationCount;
	long kForwardGraphReplayCount;
	long kForwardGraphRebuildCount;
	long kForwardGraphFallbackCount;

    bool kPoolLost; // a pool was lost by a failed restore in growPool: every growth returns BEAGLE_ERROR_OUT_OF_MEMORY

#ifdef BEAGLE_DEBUG_MEMORY
    std::vector<int> hDebugFailAllocations; // BEAGLE_DEBUG_MEMORY_FAIL during one ensureBufferCounts call, else empty
    int kDebugAllocationCount;              // tryAllocate calls so far in that call
#endif

protected:
    GPUInterface*  gpu;
    KernelLauncher* kernels;

    int kMatrixCount;
    int kCategoryCount;
    int kPaddedPatternCount;

    GPUPtr* dEigenValues;
    GPUPtr* dEvec;
    GPUPtr* dIevc;

    // The pools behind dEigenValues, dEvec, dIevc, dWeights and dFrequencies; kernels that add an offset to the
    // start of a pool get these, not the first per-index handle (on OpenCL that may be a one-stride sub-buffer)
    GPUPtr dEigenValuesOrigin;
    GPUPtr dEvecOrigin;
    GPUPtr dIevcOrigin;
    GPUPtr dWeightsOrigin;
    GPUPtr dFrequenciesOrigin;

    double** hCategoryRates;

    // True when the matrix pool exists only once a matrix is written densely (spectral: matrices hold eigen-systems).
    // Then a dense read of an index that was never written densely returns BEAGLE_ERROR_OUT_OF_RANGE.
    virtual bool denseMatricesOnDemand() const { return false; }

    // ensureBufferCounts' steps for the structures of a derived class sized by the matrix count, called while the
    // counts are still the old ones when the matrix count grows to newMatrixCount. Each returns BEAGLE_SUCCESS, or an
    // error with the instance valid at the old counts. growDerivedHostArrays: larger host arrays, after the base
    // ones. growDerivedPools: device pools through growPool, after the base ones; a std::bad_alloc thrown while it
    // makes the handles of a pool that moved marks the pool lost.
    virtual int growDerivedHostArrays(int newMatrixCount) { return BEAGLE_SUCCESS; }
    virtual int growDerivedPools(int newMatrixCount) { return BEAGLE_SUCCESS; }

    // A sub-buffer of parent that is released by freeWithSubPointers(parent)
    GPUPtr createSubPointer(GPUPtr parent, size_t offset, size_t size);
    // Releases the sub-buffers made from parent by createSubPointer, then frees parent
    void freeWithSubPointers(GPUPtr parent);
    // Whether createSubPointer has made sub-buffers of parent that are not yet released
    bool hasSubPointers(GPUPtr parent) const { return hSubPointers.find(parent) != hSubPointers.end(); }
    // Moves the pool at origin (capacity slots of strideBytes) to a block of newCapacity slots that starts with its first
    // usedSlots slots; nothing changes when newCapacity <= capacity. BEAGLE_ERROR_OUT_OF_RANGE before any change when the
    // block would exceed INT_MAX elements. BEAGLE_ERROR_OUT_OF_MEMORY leaves the old pool, possibly restored at a new
    // address by the host-staged fallback (best effort: origin 0 and capacity 0 if that fails too). After such a failed
    // restore the pool stays lost (kPoolLost): every later growth returns BEAGLE_ERROR_OUT_OF_MEMORY, and the instance
    // must be finalized. On TinyGPU, a lost or failed instance returns BEAGLE_ERROR_GENERAL (BEAGLE_ERROR_OUT_OF_MEMORY
    // when GetOutOfMemory) and stages, frees and reports nothing.
    // Whenever the old block is freed, its sub-buffers are released and !hasSubPointers(origin): the caller then rebuilds
    // its handles with createSubPointer(origin, ...).
    int growPool(GPUPtr& origin, int& capacity, size_t strideBytes, int newCapacity, int usedSlots);

    /* Accessors for the pooled partials/states origins and their per-index
     * element offsets, and the per-eigen-decomposition matrix/eigenvalue
     * strides — needed by BeagleGPUSpectralImpl to build a single batched
     * offset-queue kernel argument (mirrors this class's own dPtrQueue
     * idiom) instead of resolving one GPUPtr per branch on the host. The
     * offset accessors (not a naive index*stride recompute) are required
     * for correctness across tip/pattern resorting, which swaps which
     * physical buffer a logical index maps to. */
    GPUPtr getPartialsOrigin() const { return dPartialsOrigin; }
    GPUPtr getStatesOrigin() const { return dStatesOrigin; }
    unsigned int getPartialsOffsetElements(int index) const { return hPartialsOffsets[index]; }
    unsigned int getStatesOffsetElements(int index) const { return hStatesOffsets[index]; }
    unsigned int getEvecStrideElements() const { return kEvecOffset; }
    unsigned int getEvalStrideElements() const { return kEvalOffset; }
    // For BeagleGPUFusedImpl's op table: the matrix pool and its per-index stride, and the scale pool (0 without scale
    // buffers; scalingFactorsOrigin() would dereference dScalingFactors[0]) and its per-index offset (as upPartials)
    GPUPtr getMatricesOrigin() const { return dMatricesOrigin; }
    unsigned int getMatrixStrideElements() const { return kIndexOffsetMat; }
    GPUPtr getScalingFactorsOriginRaw() const { return dScalingFactorsOrigin; }
    unsigned int getScaleBufferSizeElements() const { return (unsigned int) kScaleBufferSize; }
    // BeagleGPUSpectralImpl's adjoint: the valid scale indices, and the pool's buffer of zeros (raw scalers) after them
    int getScaleBufferCount() const { return kScaleBufferCount; }

    virtual void dispatchPrunePP(GPUPtr p1, GPUPtr p2, GPUPtr p3,
                                  int c1MatIdx, int c2MatIdx,
                                  GPUPtr scalingFactors, GPUPtr cumulativeScaling,
                                  unsigned int startPattern, unsigned int endPattern,
                                  int rescale, int streamIndex, int waitIndex);
    virtual void dispatchPruneSP(GPUPtr s1, GPUPtr p2, GPUPtr p3,
                                  int c1MatIdx, int c2MatIdx,
                                  GPUPtr scalingFactors, GPUPtr cumulativeScaling,
                                  unsigned int startPattern, unsigned int endPattern,
                                  int rescale, int streamIndex, int waitIndex);
    virtual void dispatchPruneSS(GPUPtr s1, GPUPtr s2, GPUPtr p3,
                                  int c1MatIdx, int c2MatIdx,
                                  GPUPtr scalingFactors, GPUPtr cumulativeScaling,
                                  unsigned int startPattern, unsigned int endPattern,
                                  int rescale, int streamIndex, int waitIndex);
    // A degree-2 (single-child) post-order operation, child 2 BEAGLE_OP_NONE: p3 = P(c1MatIdx) c1. Reached only from a
    // subclass whose checkSingleChild accepts degree 2 (BeagleGPUSpectralImpl)
    virtual void dispatchPruneDegree2(GPUPtr c1, bool c1IsStates, GPUPtr p3,
                                      int c1MatIdx,
                                      GPUPtr scalingFactors, GPUPtr cumulativeScaling,
                                      unsigned int startPattern, unsigned int endPattern,
                                      int rescale, int streamIndex, int waitIndex) {}

    // Degree-2 (single-child) nodes, a second child or sibling of BEAGLE_OP_NONE: BEAGLE_ERROR_NO_IMPLEMENTATION if an
    // operation has one, BEAGLE_SUCCESS otherwise. BeagleGPUSpectralImpl computes them and accepts every operation
    virtual int checkSingleChild(const int* operations,
                                 int operationCount,
                                 int numOps);

public:
    BeagleGPUImpl();

    virtual ~BeagleGPUImpl();

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
                       long long requirementFlags);

    int getInstanceDetails(BeagleInstanceDetails* retunInfo);

    int setCPUThreadCount(int threadCount);

    int setTipStates(int tipIndex,
                     const int* inStates);

    int setTipPartials(int tipIndex,
                       const double* inPartials);

    int setRootPrePartials(const int* bufferIndices,
                           const int* stateFrequenciesIndices,
                           int count);

    int setPartials(int bufferIndex,
                    const double* inPartials);

    int getPartials(int bufferIndex,
				    int scaleIndex,
                    double* outPartials);

    int getPartialsBatch(const int* bufferIndices,
                         const int* scaleIndices,
                         int count,
                         double* outPartials);

    int setEigenDecomposition(int eigenIndex,
                              const double* inEigenVectors,
                              const double* inInverseEigenVectors,
                              const double* inEigenValues);

    int setStateFrequencies(int stateFrequenciesIndex,
                            const double* inStateFrequencies);

    int setCategoryWeights(int categoryWeightsIndex,
                           const double* inCategoryWeights);

    int setPatternWeights(const double* inPatternWeights);

    int setPatternPartitions(int partitionCount,
                             const int* inPatternPartitions);

    int setCategoryRates(const double* inCategoryRates);

    int setCategoryRatesWithIndex(int categoryRatesIndex,
                                  const double* inCategoryRates);

    int setTransitionMatrix(int matrixIndex,
                            const double* inMatrix,
                            double paddedValue);

    int setDifferentialMatrix(int matrixIndex,
                              const double* inMatrix);

    int setTransitionMatrices(const int* matrixIndices,
                              const double* inMatrices,
                              const double* paddedValues,
                              int count);

    int getTransitionMatrix(int matrixIndex,
                            double* outMatrix);

  	int convolveTransitionMatrices(const int* firstIndices,
                                   const int* secondIndices,
                                   const int* resultIndices,
                                   int matrixCount);

	int addTransitionMatrices(const int* firstIndices,
							  const int* secondIndices,
							  const int* resultIndices,
							  int matrixCount);

    int transposeTransitionMatrices(const int* inputIndices,
                                    const int* resultIndices,
                                    int matrixCount);

    int updateTransitionMatrices(int eigenIndex,
                                 const int* probabilityIndices,
                                 const int* firstDerivativeIndices,
                                 const int* secondDerivativeIndices,
                                 const double* edgeLengths,
                                 int count);
    
     int updateTransitionMatricesGrad(const int* probabilityIndices,
                                      const double* edgeLengths,
                                      int count);

    int updateTransitionMatricesWithModelCategories(int* eigenIndices,
                                 const int* probabilityIndices,
                                 const int* firstDerivativeIndices,
                                 const int* secondDerivativeIndices,
                                 const double* edgeLengths,
                                 int count);

    int updateTransitionMatricesWithMultipleModels(const int* eigenIndices,
                                                   const int* categoryRateIndices,
                                                   const int* probabilityIndices,
                                                   const int* firstDerivativeIndices,
                                                   const int* secondDerivativeIndices,
                                                   const double* edgeLengths,
                                                   int count);

    int updatePartials(const int* operations,
                       int operationCount,
                       int cumulativeScalingIndex,
                       BeaglePartialsType partialsType);

    int updatePrePartials(const int *operations,
                          int count,
                          int cumulativeScaleIndex,
                          BeaglePartialsType preorderType);

	int calculateEdgeDerivative(const int *postBufferIndices,
								const int *preBufferIndices,
								const int rootBufferIndex,
								const int *firstDerivativeIndices,
								const int *secondDerivativeIndices,
								const int categoryWeightsIndex,
								const int categoryRatesIndex,
								const int stateFrequenciesIndex,
								const int *cumulativeScaleIndices,
								int count,
								double *outFirstDerivative,
								double *outDiagonalSecondDerivative);

    int calculateEdgeDerivatives(const int *postBufferIndices,
                                 const int *preBufferIndices,
                                 const int *derivativeMatrixIndices,
                                 const int *categoryWeightsIndices,
                                 const int *categoryRatesIndices,
                                 const int *cumulativeScaleIndices,
                                 int count,
                                 double *outDerivatives,
                                 double *outSumDerivatives,
                                 double *outSumSquaredDerivatives);

    int calculateAdjointCrossProducts(const int *postBufferIndices,
                                      const int *preBufferIndices,
                                      const int *eigenIndices,
                                      const int *categoryRatesIndices,
                                      const int *categoryWeightsIndices,
                                      const int rootPostOrderIndex,
                                      const int stateFrequenciesIndex,
                                      int count,
                                      double *outSumDerivatives,
                                      double *outSumSquaredDerivatives,
                                      const int *postScaleIndices = nullptr,
                                      const int *preScaleIndices = nullptr,
                                      const int cumulativeScaleIndex = BEAGLE_OP_NONE);

    int calculateAdjointDerivative(const BeagleBranchOperation *operations,
                                   const int categoryRatesIndex,
                                   const int categoryWeightsIndex,
                                   const int rootPostOrderIndex,
                                   const int stateFrequenciesIndex,
                                   const int operationCount,
                                   double *outSumDerivatives,
                                   double *outSumSquaredDerivatives);

    int updatePartialsByPartition(const int* operations,
                                  int operationCount);

    int updatePrePartialsByPartition(const int* operations,
                                     int operationCount,
                                     BeaglePartialsType preorderType);

    int waitForPartials(const int* destinationPartials,
                        int destinationPartialsCount);

    int accumulateScaleFactors(const int* scalingIndices,
                               int count,
                               int cumulativeScalingIndex);

    int accumulateScaleFactorsByPartition(const int* scalingIndices,
                                          int count,
                                          int cumulativeScalingIndex,
                                          int partitionIndex);

    int removeScaleFactors(const int* scalingIndices,
                           int count,
                           int cumulativeScalingIndex);

    int removeScaleFactorsByPartition(const int* scalingIndices,
                                      int count,
                                      int cumulativeScalingIndex,
                                      int partitionIndex);

    int resetScaleFactors(int cumulativeScalingIndex);

    int resetScaleFactorsByPartition(int cumulativeScalingIndex, int partitionIndex);

    int copyScaleFactors(int destScalingIndex,
                         int srcScalingIndex);

    int getScaleFactors(int srcScalingIndex,
                        double* scaleFactors);

    int calculateRootLogLikelihoods(const int* bufferIndices,
                                    const int* categoryWeightsIndices,
                                    const int* stateFrequenciesIndices,
                                    const int* cumulativeScaleIndices,
                                    int count,
                                    double* outSumLogLikelihood);

    int calculateRootLogLikelihoodsByPartition(const int* bufferIndices,
                                               const int* categoryWeightsIndices,
                                               const int* stateFrequenciesIndices,
                                               const int* cumulativeScaleIndices,
                                               const int* partitionIndices,
                                               int partitionCount,
                                               int count,
                                               double* outSumLogLikelihoodByPartition,
                                               double* outSumLogLikelihood);

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
                                    double* outSumSecondDerivative);

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
                                               double* outSumSecondDerivative);

	int calculateCrossProducts(const int *postBufferIndices,
							   const int *preBufferIndices,
							   const int *categoryRatesIndices,
							   const int *categoryWeightsIndices,
							   const double *edgeLengths,
							   int count,
							   double *outSumDerivatives,
							   double *outSumSquaredDerivatives);

    int getLogLikelihood(double* outSumLogLikelihood);

    int getDerivatives(double* outSumFirstDerivative,
                       double* outSumSecondDerivative);

    int getSiteLogLikelihoods(double* outLogLikelihoods);

    int getSiteDerivatives(double* outFirstDerivatives,
                           double* outSecondDerivatives);

    int ensureBufferCounts(int partialsBufferCount,
                           int matrixBufferCount,
                           int scaleBufferCount);

    int updateInnerBastaPartials(const int * operations, const int * intervals, int i, int begin, int end, GPUPtr sizes, GPUPtr coalescent);

    // inStride: the integers per operation of operations (BEAGLE_BASTA_OP_COUNT_V5 for _v5); the queue keeps 8
    void uploadBastaOperationQueue(const int* operations, int count, int inStride = BEAGLE_BASTA_OP_COUNT);

    int uploadBastaSlabMetadata(const int* packed, int packedLen);

    int getBastaSlabConstants(int* opsPerBlock, int* indexOffsetPat);

    int ensureLoewnerInfoUploaded(int eigenIndex);

    // BASTA's per-eigen raw and transposed eigenvectors (dRawEvec, dRawIevc, dEvecT, dInverseEvecT), which only the
    // BASTA calls read: allocated by the first call that reads them, and filled there for the decompositions already
    // set; from then on setEigenDecomposition uploads them too. Evec and Ievc are setEigenDecomposition's padded,
    // untransposed matrices.
    void ensureBastaEigenBuffers();
    void uploadBastaEigenBuffers(int eigenIndex, const Real* Evec, const Real* Ievc);

	int updateBastaPartials(const int* operations,
                            int operationCount,
                            const int* intervals,
                            int intervalCount,
                            int populationSizesIndex,
                            int coalescentIndex);

     int updateBastaPartialsGrad(const int* operations,
  		 					int operationCount,
  		 					const int* intervals,
  		 					int intervalCount,
                             int populationSizesIndex,
                             int coalescentIndex);

	int accumulateBastaPartials(const int* operations,
	     				  		int operationCount,
	     				  		const int* segments,
	     				  		int segmentCount,
                                const double* intervalLengths,
                                const int populationSizesIndex,
                                int coalescentIndex,
                                double* out);
    
     int accumulateBastaPartialsGrad(const int *operations,
                                     const int operationCount,
                                     const int *intervalStarts,
                                     const int intervalStartsCount,
                                     const double *intervalLengths,
                                     const int populationSizesIndex,
                                     const int coalescentIndex,
                                     int eigenIndex,
                                     int partialAdjointBufferBase,
                                     int matrixAdjointBufferBase,
                                     double *popSizeGradOut,
                                     double *out);

    int allocateBastaBuffers(int bufferCount,
                             int bufferLength,
                             int partialsCount,
                             int initial,
                             int numThreads);

    int allocateBastaGradBuffers(int partialsCount);

    int getBastaBuffer(int bufferIndex,
                       double* out);

    int accumulateEigenBasisGradient(int eigenIndex,
                                     int matrixAdjointBufferBase,
                                     const double* branchLengths,
                                     int matrixCount,
                                     int hasComplexEigenvalues,
                                     double* outRateGradient);

    int setBastaPopulationSizes(const double* combinedSizesIntegrals,
                                int requiredStorageSize);

    int setBastaPopulationSizesBuffer(const double* combinedSizesIntegrals,
                                      int requiredStorageSize,
                                      int bufferIndex);

    int setCurrentPopulationSizeBuffer(int bufferIndex);

    int updateBastaPartials_v5(const int* operations,
                               int operationCount,
                               const int* intervals,
                               int intervalCount,
                               int populationSizesIndex,
                               int coalescentIndex);

    int updateBastaPartialsGrad_v5(const int* operations,
                                   int operationCount,
                                   const int* intervals,
                                   int intervalCount,
                                   int populationSizesIndex,
                                   int coalescentIndex);

    int accumulateBastaPartials_v5(const int* operations,
                                   int operationCount,
                                   const int* intervalStarts,
                                   int intervalStartsCount,
                                   const double* intervalLengths,
                                   const int populationSizesIndex,
                                   int coalescentIndex,
                                   double* out);

    int accumulateBastaPartialsGrad_v5(const int* operations,
                                       int operationCount,
                                       const int* intervalStarts,
                                       int intervalStartsCount,
                                       const double* intervalLengths,
                                       const int populationSizesIndex,
                                       int coalescentIndex,
                                       int eigenIndex,
                                       int partialAdjointBufferBase,
                                       int matrixAdjointBufferBase,
                                       double* popSizeGradOut,
                                       double* out);

private:

    // the _v5 functions' checks and, once they pass, the row buffers; then the sizes (half 0) or integrals (half 1) of
    // the current population-size buffer, a row per intervalNumber, into rows
    int prepareBasta_v5(const int* operations,
                        int operationCount,
                        const int* intervalStarts,
                        int intervalStartsCount);

    void uploadBastaRows_v5(GPUPtr rows, int half);

    virtual char* getInstanceName();

    // Every device allocation growPool and ensureBufferCounts make
#ifdef BEAGLE_DEBUG_MEMORY
    // fails the allocations BEAGLE_DEBUG_MEMORY_FAIL lists; with BEAGLE_DEBUG_MEMORY_SCRIBBLE set, fills new memory with
    // 0x3f bytes
    GPUPtr tryAllocate(size_t bytes);
#else
    GPUPtr tryAllocate(size_t bytes) { return gpu->TryAllocateMemory(bytes); }
#endif

    int growBufferCounts(int partialsBufferCount, int matrixBufferCount, int scaleBufferCount); // ensureBufferCounts

    void  allocateMultiGridBuffers();

    int  reorderPatternsByPartition();

    int transposeTransitionMatricesOnTheFly(const int *operations,
                                            int operationCount,
                                            std::vector<int>& newOperation);

    int setMatrixBufferImpl(int matrixIndex,
                            const double* inMatrix,
                            double paddedValue,
                            bool transpose);
    int ensureDenseMatrixPool();
    bool isDenseMatrix(int matrixIndex) const; // a valid index with dense contents

    int upPartials(bool byPartition,
                   const int* operations,
                   int operationCount,
                   int cumulativeScalingIndex,
                   BeaglePartialsType partialsType);

	int upPrePartials(bool byPartition,
					  const int* operations,
					  int count,
					  int cumulativeScaleIndex,
					  BeaglePartialsType preorderType);

	int upPrePartialsBottom(bool byPartition,
					        const int* operations,
					        int count,
					        int cumulativeScaleIndex);

	int upPrePartialsTop(bool byPartition,
					     const int* operations,
					     int count,
					     int cumulativeScaleIndex);

	void initDerivatives(int replicates);

	int calcEdgeFirstDerivatives(const int *postBufferIndices, const int *preBufferIndices, const int *firstDerivativeIndices,
								 const int *categoryWeightsIndices, const int *scaleIndices, int count,
								 double *outFirstDerivatives,
								 double *outSumFirstDerivatives,
								 double *outSumSquaredFirstDerivatives);

	int calcCrossProducts(const int *postBufferIndices,
						  const int *preBufferIndices,
						  const int *categoryRateIndices,
						  const int *categoryWeightIndices,
						  const double* edgeLengths,
						  int totalCount,
						  double *outCrossProducts);
};

BEAGLE_GPU_TEMPLATE
class BeagleGPUImplFactory : public BeagleImplFactory {
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

template <typename Real>
void modifyFlagsForPrecision(long long* flags, Real r);

} // namspace device
}	// namespace gpu
}	// namespace beagle

#include "libhmsbeagle/GPU/BeagleGPUImpl.hpp"

#endif // __BeagleGPUImpl__
