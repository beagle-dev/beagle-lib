/*
 *  BeagleImpl.h
 *  BEAGLE
 *
 * Copyright 2009 Phylogenetic Likelihood Working Group
 *
 * This file is part of BEAGLE.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 *
 * @author Andrew Rambaut
 * @author Marc Suchard
 * @author Daniel Ayres
 */

#ifndef __beagle_impl__
#define __beagle_impl__

#include "libhmsbeagle/beagle.h"

#ifdef DOUBLE_PRECISION
#define REAL    double
#else
#define REAL    float
#endif

namespace beagle {

class BeagleImpl
{
public:
    virtual ~BeagleImpl(){}

    virtual int createInstance(int tipCount,
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
                               long requirementFlags) = 0;

    virtual int getInstanceDetails(BeagleInstanceDetails* returnInfo) = 0;

    virtual int setCPUThreadCount(int threadCount) = 0;

    virtual int setTipStates(int tipIndex,
                             const int* inStates) = 0;

    virtual int setTipPartials(int tipIndex,
                               const double* inPartials) = 0;

    virtual int setPartials(int bufferIndex,
                            const double* inPartials) = 0;

	virtual int setRootPrePartials(const int* bufferIndices,
                                   const int* stateFrequenciesIndices,
                                   int count) = 0;

    virtual int getPartials(int bufferIndex,
							int scaleIndex,
                            double* outPartials) = 0;

    virtual int setEigenDecomposition(int eigenIndex,
                                      const double* inEigenVectors,
                                      const double* inInverseEigenVectors,
                                      const double* inEigenValues) = 0;

    virtual int setStateFrequencies(int stateFrequenciesIndex,
                                  const double* inStateFrequencies) = 0;

    virtual int setCategoryWeights(int categoryWeightsIndex,
                                 const double* inCategoryWeights) = 0;

    virtual int setPatternWeights(const double* inPatternWeights) = 0;

    virtual int setPatternPartitions(int partitionCount,
                                     const int* inPatternPartitions) = 0;

    virtual int setCategoryRates(const double* inCategoryRates) = 0;

    virtual int setCategoryRatesWithIndex(int categoryRatesIndex,
                                          const double* inCategoryRates) = 0;

    virtual int setTransitionMatrix(int matrixIndex,
                                    const double* inMatrix,
                                    double paddedValue) = 0;

    virtual int setDifferentialMatrix(int matrixIndex,
                                    const double* inMatrix) = 0;

    virtual int setTransitionMatrices(const int* matrixIndices,
                                      const double* inMatrices,
                                      const double* paddedValues,
                                      int count) = 0;

    virtual int getTransitionMatrix(int matrixIndex,
                                    double* outMatrix) = 0;

	virtual int convolveTransitionMatrices(const int* firstIndices,
										   const int* secondIndices,
										   const int* resultIndices,
										   int matrixCount) = 0;

	virtual int addTransitionMatrices(const int* firstIndices,
									  const int* secondIndices,
									  const int* resultIndices,
									  int matrixCount) = 0;

	virtual int transposeTransitionMatrices(const int* inputIndices,
											const int* resultIndices,
											int matrixCount) = 0;

    virtual int updateTransitionMatrices(int eigenIndex,
                                         const int* probabilityIndices,
                                         const int* firstDerivativeIndices,
                                         const int* secondDerivativeIndices,
                                         const double* edgeLengths,
                                         int count) = 0;

    virtual int updateTransitionMatricesWithModelCategories(int* eigenIndices,
                                         const int* probabilityIndices,
                                         const int* firstDerivativeIndices,
                                         const int* secondDerivativeIndices,
                                         const double* edgeLengths,
                                         int count) = 0;

    virtual int updateTransitionMatricesWithMultipleModels(const int* eigenIndices,
                                                           const int* categoryRateIndices,
                                                           const int* probabilityIndices,
                                                           const int* firstDerivativeIndices,
                                                           const int* secondDerivativeIndices,
                                                           const double* edgeLengths,
                                                           int count) = 0;

    virtual int updatePartials(const int* operations,
                               int operationCount,
                               int cumulativeScalingIndex) = 0;

	virtual int updatePrePartials(const int *operations,
                                  int count,
                                  int cumulativeScaleIndex) = 0;

    virtual int updatePartialsByPartition(const int* operations,
                                          int operationCount) = 0;

    virtual int updatePrePartialsByPartition(const int* operations,
                                             int operationCount) = 0;

    virtual int waitForPartials(const int* destinationPartials,
                                int destinationPartialsCount) = 0;

    virtual int accumulateScaleFactors(const int* scalingIndices,
									   int count,
									   int cumulativeScalingIndex) = 0;

    virtual int accumulateScaleFactorsByPartition(const int* scaleIndices,
                                                  int count,
                                                  int cumulativeScaleIndex,
                                                  int partitionIndex) = 0;

    virtual int removeScaleFactors(const int* scalingIndices,
                                     int count,
                                     int cumulativeScalingIndex) = 0;

    virtual int removeScaleFactorsByPartition(const int* scaleIndices,
                                                  int count,
                                                  int cumulativeScaleIndex,
                                                  int partitionIndex) = 0;

    virtual int resetScaleFactors(int cumulativeScalingIndex) = 0;

    virtual int resetScaleFactorsByPartition(int cumulativeScaleIndex,
                                             int partitionIndex) = 0;

    virtual int copyScaleFactors(int destScalingIndex,
                                 int srcScalingIndex) = 0;

    virtual int getScaleFactors(int srcScalingIndex,
                                 double* scaleFactors) = 0;

    virtual int calculateRootLogLikelihoods(const int* bufferIndices,
                                            const int* categoryWeightsIndices,
                                            const int* stateFrequenciesIndices,
                                            const int* scalingFactorsIndices,
                                            int count,
                                            double* outSumLogLikelihood) = 0;

    virtual int calculateRootLogLikelihoodsByPartition(const int* bufferIndices,
                                                       const int* categoryWeightsIndices,
                                                       const int* stateFrequenciesIndices,
                                                       const int* cumulativeScaleIndices,
                                                       const int* partitionIndices,
                                                       int partitionCount,
                                                       int count,
                                                       double* outSumLogLikelihoodByPartition,
                                                       double* outSumLogLikelihood) = 0;

    virtual int calculateEdgeDerivatives(const int *postBufferIndices,
                                         const int *preBufferIndices,
                                         const int *derivativeMatrixIndices,
                                         const int *categoryWeightsIndices,
                                         const int *categoryRatesIndices,
                                         const int *cumulativeScaleIndices,
                                         int count,
                                         double *outDerivatives,
                                         double *outSumDerivatives,
                                         double *outSumSquaredDerivatives) = 0;

	virtual int calculateCrossProducts(const int *postBufferIndices,
			                           const int *preBufferIndices,
			                           const int *categoryRatesIndices,
			                           const int *categoryWeightsIndices,
			                           const double *edgeLengths,
			                           int count,
			                           double *outSumDerivatives,
			                           double *outSumSquaredDerivatives) = 0;

    virtual int calculateEdgeLogLikelihoods(const int* parentBufferIndices,
                                            const int* childBufferIndices,
                                            const int* probabilityIndices,
                                            const int* firstDerivativeIndices,
                                            const int* secondDerivativeIndices,
                                            const int* categoryWeightsIndices,
                                            const int* stateFrequenciesIndices,
                                            const int* scalingFactorsIndices,
                                            int count,
                                            double* outSumLogLikelihood,
                                            double* outSumFirstDerivative,
                                            double* outSumSecondDerivative) = 0;

    virtual int calculateEdgeLogLikelihoodsByPartition(const int* parentBufferIndices,
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
                                                       double* outSumSecondDerivative) = 0;

    virtual int getLogLikelihood(double* outSumLogLikelihood) = 0;

    virtual int getDerivatives(double* outSumFirstDerivative,
                               double* outSumSecondDerivative) = 0;

    virtual int getSiteLogLikelihoods(double* outLogLikelihoods) = 0;

    virtual int getSiteDerivatives(double* outFirstDerivatives,
                                   double* outSecondDerivatives) = 0;
//protected:
    int resourceNumber;
};

class BeagleImplFactory {
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
                                   int* errorCode) = 0; // pure virtual

    virtual const char* getName() = 0; // pure virtual

    virtual const long getFlags() = 0; // pure virtual
};

} // end namespace beagle

#endif // __beagle_impl__
