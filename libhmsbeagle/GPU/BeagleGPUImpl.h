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

#include "libhmsbeagle/BeagleImpl.h"
#include "libhmsbeagle/GPU/GPUImplDefs.h"
#include "libhmsbeagle/GPU/GPUInterface.h"
#include "libhmsbeagle/GPU/KernelLauncher.h"

#define BEAGLE_GPU_GENERIC	Real
#define BEAGLE_GPU_TEMPLATE template <typename Real>

#if defined(CUDA) && defined(CUDA_TENSOR_CORES)
    using namespace tensor_cores_device;
//    using namespace  cuda_device;
#elif defined(CUDA)
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
