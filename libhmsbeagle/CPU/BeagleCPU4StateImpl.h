/*
 *  BeagleCPU4StateImpl.h
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

#ifndef __BeagleCPU4StateImpl__
#define __BeagleCPU4StateImpl__

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include "libhmsbeagle/CPU/BeagleCPUImpl.h"

#include <vector>

namespace beagle {
namespace cpu {

//const char* beagleCPU4StateImplDoubleName = "CPU-4State-Double";
//const char* beagleCPU4StateImplSingleName = "CPU-4State-Single";

BEAGLE_CPU_TEMPLATE
class BeagleCPU4StateImpl : public BeagleCPUImpl<BEAGLE_CPU_GENERIC> {

protected:
	using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::kFlags;
	using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::kTipCount;
	using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::gPartials;
	using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::integrationTmp;
	using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::gTransitionMatrices;
	using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::kPatternCount;
	using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::kMatrixSize;
	using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::kPaddedPatternCount;
	using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::kExtraPatterns;
	using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::kStateCount;
	using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::gTipStates;
	using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::kCategoryCount;
	using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::gScaleBuffers;
	using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::gStateFrequencies;
	using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::gCategoryWeights;
    using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::gCategoryRates;
    using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::grandNumeratorDerivTmp;
//    using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::grandNumeratorLowerBoundDerivTmp;
//    using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::grandNumeratorUpperBoundDerivTmp;
    using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::grandDenominatorDerivTmp;
//    using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::cLikelihoodTmp;
	using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::gPatternWeights;
	using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::outLogLikelihoodsTmp;
	using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::realtypeMin;
  using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::scalingExponentThreshold;
  using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::gPatternPartitionsStartPatterns;
  using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::accumulateDerivatives;
  using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcEdgeLogDerivatives;

public:
    virtual ~BeagleCPU4StateImpl();
    virtual const char* getName();


    virtual void calcStatesStates(REALTYPE* destP,
                                    const int* states1,
                                    const REALTYPE* matrices1,
                                    const int* states2,
                                    const REALTYPE* matrices2,
                                    int startPattern,
                                    int endPattern);

    virtual void calcStatesPartials(REALTYPE* destP,
                                    const int* states1,
                                    const REALTYPE* matrices1,
                                    const REALTYPE* partials2,
                                    const REALTYPE* matrices2,
                                    int startPattern,
                                    int endPattern);

    virtual void calcPartialsPartials(REALTYPE* destP,
                                      const REALTYPE* partials1,
                                      const REALTYPE* matrices1,
                                      const REALTYPE* partials2,
                                      const REALTYPE* matrices2,
                                      int startPattern,
                                      int endPattern);

	virtual void calcPrePartialsPartials(REALTYPE *destP,
                                         const REALTYPE *partialsParent,
                                         const REALTYPE *matricesSelf,
                                         const REALTYPE *partialsSibling,
                                         const REALTYPE *matricesSibling,
                                         int startPattern,
                                         int endPattern);

    virtual void calcPrePartialsStates(REALTYPE* destP,
                                       const REALTYPE* partials1,
                                       const REALTYPE* matrices1,
                                       const int* states2,
                                       const REALTYPE* matrices2,
                                       int startPattern,
                                       int endPattern);

    virtual void calcEdgeLogDerivativesStates(const int *tipStates,
                                             const REALTYPE *preOrderPartial,
                                             const int firstDerivativeIndex,
                                             const int secondDerivativeIndex,
                                             const double *categoryRates,
                                             const REALTYPE *categoryWeights,
                                             double *outDerivatives,
                                             double *outSumDerivatives,
                                             double *outSumSquaredDerivatives);

    virtual void calcEdgeLogDerivativesPartials(const REALTYPE *postOrderPartial,
                                               const REALTYPE *preOrderPartial,
                                               const int firstDerivativeIndex,
                                               const int secondDerivativeIndex,
                                               const double *categoryRates,
                                               const REALTYPE *categoryWeights,
                                               const int scalingFactorsIndex,
                                               double *outDerivatives,
                                               double *outSumDerivatives,
                                               double *outSumSquaredDerivatives);

    virtual void calcCrossProductsStates(const int *tipStates,
                                         const REALTYPE *preOrderPartial,
                                         const double *categoryRates,
                                         const REALTYPE *categoryWeights,
                                         const double edgeLength,
                                         double *outCrossProducts,
                                         double *outSumSquaredDerivatives);

    virtual void calcCrossProductsPartials(const REALTYPE *postOrderPartial,
                                           const REALTYPE *preOrderPartial,
                                           const double *categoryRates,
                                           const REALTYPE *categoryWeights,
                                           const double edgeLength,
                                           double *outCrossProducts,
                                           double *outSumSquaredDerivatives);

    virtual int calcRootLogLikelihoods(const int bufferIndex,
                                        const int categoryWeightsIndex,
                                        const int stateFrequenciesIndex,
                                        const int scalingFactorsIndex,
                                        double* outSumLogLikelihood);

    virtual void calcRootLogLikelihoodsByPartition(const int* bufferIndices,
                                                  const int* categoryWeightsIndices,
                                                  const int* stateFrequenciesIndices,
                                                  const int* cumulativeScaleIndices,
                                                  const int* partitionIndices,
                                                  int partitionCount,
                                                  double* outSumLogLikelihoodByPartition);

    virtual int calcRootLogLikelihoodsMulti(const int* bufferIndices,
                                             const int* categoryWeightsIndices,
                                             const int* stateFrequenciesIndices,
                                             const int* scaleBufferIndices,
                                             int count,
                                             double* outSumLogLikelihood);

    virtual int calcEdgeLogLikelihoods(const int parentBufferIndex,
                                        const int childBufferIndex,
                                        const int probabilityIndex,
                                        const int categoryWeightsIndex,
                                        const int stateFrequenciesIndex,
                                        const int scalingFactorsIndex,
                                        double* outSumLogLikelihood);

    virtual void calcEdgeLogLikelihoodsByPartition(const int* parentBufferIndices,
                                                  const int* childBufferIndices,
                                                  const int* probabilityIndices,
                                                  const int* categoryWeightsIndices,
                                                  const int* stateFrequenciesIndices,
                                                  const int* cumulativeScaleIndices,
                                                  const int* partitionIndices,
                                                  int partitionCount,
                                                  double* outSumLogLikelihoodByPartition);

    virtual void calcStatesStatesFixedScaling(REALTYPE *destP,
                                              const int *child0States,
                                              const REALTYPE *child0TransMat,
                                              const int *child1States,
                                              const REALTYPE *child1TransMat,
                                              const REALTYPE *scaleFactors,
                                              int startPattern,
                                              int endPattern);

    virtual void calcStatesPartialsFixedScaling(REALTYPE *destP,
                                                const int *child0States,
                                                const REALTYPE *child0TransMat,
                                                const REALTYPE *child1Partials,
                                                const REALTYPE *child1TransMat,
                                                const REALTYPE *scaleFactors,
                                                int startPattern,
                                                int endPattern);

    virtual void calcPartialsPartialsFixedScaling(REALTYPE *destP,
                                                  const REALTYPE *child0Partials,
                                                  const REALTYPE *child0TransMat,
                                                  const REALTYPE *child1Partials,
                                                  const REALTYPE *child1TransMat,
                                                  const REALTYPE *scaleFactors,
                                                  int startPattern,
                                                  int endPattern);

    virtual void calcPartialsPartialsAutoScaling(REALTYPE *destP,
                                                  const REALTYPE *child0Partials,
                                                  const REALTYPE *child0TransMat,
                                                  const REALTYPE *child1Partials,
                                                  const REALTYPE *child1TransMat,
                                                  int *activateScaling);


    inline int integrateOutStatesAndScale(const REALTYPE* integrationTmp,
                                           const int stateFrequenciesIndex,
                                           const int scalingFactorsIndex,
                                           double* outSumLogLikelihood);

    inline void integrateOutStatesAndScaleByPartition(const REALTYPE* integrationTmp,
                                                     const int* stateFrequenciesIndices,
                                                     const int* cumulativeScaleIndices,
                                                     const int* partitionIndices,
                                                     int partitionCount,
                                                     double* outSumLogLikelihoodByPartition);

    virtual void rescalePartials(REALTYPE *destP,
    		                     REALTYPE *scaleFactors,
                                 REALTYPE *cumulativeScaleFactors,
                                 const int  fillWithOnes);

    virtual void rescalePartialsByPartition(REALTYPE *destP,
                                            REALTYPE *scaleFactors,
                                            REALTYPE *cumulativeScaleFactors,
                                            const int fillWithOnes,
                                            const int partitionIndex);


};

BEAGLE_CPU_FACTORY_TEMPLATE
class BeagleCPU4StateImplFactory : public BeagleImplFactory {
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

}	// namespace cpu
}	// namespace beagle

// now include the file containing template function implementations
#include "libhmsbeagle/CPU/BeagleCPU4StateImpl.hpp"

#endif // __BeagleCPU4StateImpl__
