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
#else
	using namespace opencl_device;
#endif

namespace beagle {
namespace gpu {

#ifdef CUDA
	namespace cuda {
#else
	namespace opencl {
#endif

BEAGLE_GPU_TEMPLATE
class BeagleGPUImpl : public BeagleImpl {


private:
    GPUInterface* gpu;
    KernelLauncher* kernels;

    int kInitialized;

    long kFlags;

    int kTipCount;
    int kPartialsBufferCount;
    int kCompactBufferCount;
    int kStateCount;
    int kPatternCount;
    int kEigenDecompCount;
    int kMatrixCount;
    int kCategoryCount;

    int kTipPartialsBufferCount;
    int kInternalPartialsBufferCount;
    int kBufferCount;
    int kScaleBufferCount;

    int kPaddedStateCount;
    int kPaddedPatternCount;    // total # of patterns with padding so that kPaddedPatternCount
                                //   * kPaddedStateCount is a multiple of 16
    int kSumSitesBlockCount;

    int kPartialsSize;
    int kMatrixSize;
    int kEigenValuesSize;
    int kScaleBufferSize;

    int kLastCompactBufferIndex;
    int kLastTipPartialsBufferIndex;

    int kResultPaddedPatterns;

	int kSumIntervalBlockSize;
	int kSlabOpsPerBlock;
	int kSpineT;
	int kSumAcrossBlockSize;
    GPUPtr dIntegrationTmp;
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

    GPUPtr dPatternWeights;

    GPUPtr dBranchLengths;

    GPUPtr dDistanceQueue;

    GPUPtr dPtrQueue;

	GPUPtr dDerivativeQueue;

    GPUPtr dMaxScalingFactors;
    GPUPtr dIndexMaxScalingFactors;

    GPUPtr dAccumulatedScalingFactors;

    GPUPtr* dEigenValues;
    GPUPtr* dEvec;
    GPUPtr* dIevc;

    GPUPtr* dWeights;
    GPUPtr* dFrequencies;

    GPUPtr* dScalingFactors;

    GPUPtr* dStates;

    GPUPtr* dPartials;
    GPUPtr* dMatrices;
    GPUPtr dMatricesOrigin;
    
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
    long kDeviceType;
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

    double** hCategoryRates; // Can keep in double-precision

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

	GPUPtr dPartialsGrad;
	GPUPtr dCoalescentGrad;


	GPUPtr dPartialsGradPopSize;
	GPUPtr dCoalescentGradPopSize;

	GPUPtr dBastaGradBuffers;
	GPUPtr dBastaGradBuffersPopSize;

	GPUPtr dEdgeLengthsGrad;
	GPUPtr dGradOut;


	GPUPtr dBastaGradNodeOps;
	int* hBastaGradNodeOps;

	Real* hEdgeLengthsGrad;
	Real* hGradZeros;
	size_t kGradZerosSize;


	int kGradAbStride;
	int kGradAbStridePopSize;
	int kCoalescentGradLength;
	bool kGradBuffersAllocated;
	bool kBastaOpsUploaded;
	int  kBastaOpsCount;

	GPUPtr dPartialAdjoint;
	size_t kAdjointPartialBytes;
	GPUPtr dMatrixAdjoint;
	GPUPtr dAdjointPopSizeGrad;
	GPUPtr dHazardAdjoints;
	GPUPtr dTransformedAdjoints;
	GPUPtr dRawEigenVectors;
	GPUPtr dRawInverseEigenVectors;
	GPUPtr dAdjointIntervalNumbers;        // [intervalCount]    intervalNumber per interval (used by ComputeHazardAdjoints)
	GPUPtr dAdjointIntervalStarts;         // [intervalCount + 1] op range per interval (used by gather kernel)
	GPUPtr dAdjointMatTransIndices;        // [intervalCount]    destination matrix offset per interval
	GPUPtr dAdjointCoalOps;                // [intervalCount]    coalescent op index per interval, -1 if none
	int* hAdjointIntervalNumbers;
	int* hAdjointCoalOps;
	int* hAdjointMatTransIndices;
	int* hAdjointIntervalStarts;
	int kAdjointIntervalNumbersSize;
	Real* hRawEigenVectors;
	Real* hRawInverseEigenVectors;
	Real* hMatrixAdjointHost;
	Real* hTransformedAdjointsHost;
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
	bool kRawEigenVectorsUploaded;

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
	int kAdjointGraphNumEvBlocks;    // kSlabNumEvBlocks at capture time
	long kAdjointGraphReplayCount;    // diagnostics
	long kAdjointGraphRebuildCount;
	long kAdjointGraphFallbackCount;

	bool kForwardSlabPipelineEnabled;
	void* kForwardSlabGraphExec;
	bool kForwardSlabGraphValid;
	uint64_t kForwardSlabGraphFingerprint;
	int kForwardSlabGraphIntervalCount;
	int kForwardSlabGraphNumEvBlocks;
	long kForwardSlabGraphReplayCount;
	long kForwardSlabGraphRebuildCount;
	long kForwardSlabGraphFallbackCount;


	bool   kAdjointSlabPipelineEnabled;
	GPUPtr dPartialsTilde;
	size_t kPartialsTildeBytes;

	GPUPtr dGEigen;


	GPUPtr dEvecT;
	GPUPtr dInverseEvecT;

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
	GPUPtr dSlabCtaCarry;
	GPUPtr dSlabCtaCarryBranch;
	GPUPtr dSlabAStash;

	GPUPtr dSlabYBottomEigen;
	size_t kSlabYBottomBytes;


	GPUPtr dIntervalBranchLengths;
	size_t kIntervalBLBytes;
	size_t kIntervalOpStartBytes;
	Real*  hIntervalBranchLengths;


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


	double* hStashedEigenValues;
	int kStashedEigenValueSize;
	bool kStashedHasComplex;
	bool kLoewnerInfoBootstrapped;


	int kSlabBranchCountActive;
	int kSlabCoalCountActive;
	int kSlabBranchOpTotalActive;
	int kSlabNumEvBlocks;


	bool kBastaSlabMetadataUploaded;


	void* kForwardGraphExec;
	bool kForwardGraphValid;
	uint64_t kForwardGraphFingerprint;
	int kForwardGraphIntervalCount;
	int kForwardGraphOperationCount;
	long kForwardGraphReplayCount;
	long kForwardGraphRebuildCount;
	long kForwardGraphFallbackCount;

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
                       long preferenceFlags,
                       long requirementFlags);

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
                       int cumulativeScalingIndex);

    int updatePrePartials(const int *operations,
                          int count,
                          int cumulativeScaleIndex);

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

    int updatePartialsByPartition(const int* operations,
                                  int operationCount);

    int updatePrePartialsByPartition(const int* operations,
                                     int operationCount);

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

    int updateInnerBastaPartials(const int * operations, const int * intervals, int i, int begin, int end, GPUPtr sizes, GPUPtr coalescent);

    void uploadBastaOperationQueue(const int* operations, int count);

    int uploadBastaSlabMetadata(const int* packed, int packedLen);

    int getBastaSlabConstants(int* opsPerBlock, int* indexOffsetPat);

    int ensureLoewnerInfoUploaded();

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
                                     double *out);

     int updateBastaPartialsPopSizeGrad(const int* operations,
                                    int operationCount,
                                    const int* intervals,
                                    int intervalCount,
                                    int populationSizesIndex,
                                    int coalescentIndex);

     int accumulateBastaPartialsPopSizeGrad(const int *operations,
                                     const int operationCount,
                                     const int *intervalStarts,
                                     const int intervalStartsCount,
                                     const double *intervalLengths,
                                     const int populationSizesIndex,
                                     const int coalescentIndex,
                                     double *out);

    int allocateBastaBuffers(int bufferCount,
                             int bufferLength,
                             int partialsCount,
                             int initial,
                             int numThreads);

    int allocateBastaGradBuffers(int partialsCount);

    int getBastaBuffer(int bufferIndex,
                       double* out);

    int getBastaMatrixAdjoint(int matrixIndex,
                              double* out);

    int getBastaPopulationSizeGradient(double* out);

    int setBastaExpmKernels(const double* kernels);

    int accumulateBastaExpmGradient(double* out);

    int transformBastaMatrixAdjoints(int matrixCount, double* out);

    int backTransformBastaEigenBasisGradient(const double* eigenBasisGrad, double* out);

    int accumulateEigenBasisGradient(const double* eigenValues,
                                     const double* branchLengths,
                                     int matrixCount,
                                     int hasComplexEigenvalues,
                                     double* outRateGradient);

private:

    char* getInstanceName();

    void  allocateMultiGridBuffers();

    int  reorderPatternsByPartition();

    std::vector<int> transposeTransitionMatricesOnTheFly(const int *operations,
                                                         int operationCount);

    int setMatrixBufferImpl(int matrixIndex,
                            const double* inMatrix,
                            double paddedValue,
                            bool transpose);

    int upPartials(bool byPartition,
                   const int* operations,
                   int operationCount,
                   int cumulativeScalingIndex);

	int upPrePartials(bool byPartition,
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
                                   long preferenceFlags,
                                   long requirementFlags,
                                   int* errorCode);

    virtual const char* getName();
    virtual const long getFlags();
};

template <typename Real>
void modifyFlagsForPrecision(long* flags, Real r);

} // namspace device
}	// namespace gpu
}	// namespace beagle

#include "libhmsbeagle/GPU/BeagleGPUImpl.hpp"

#endif // __BeagleGPUImpl__
