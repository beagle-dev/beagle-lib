/*
 *  BeagleCPU4StateSSEImpl.h
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
 * @author Marc Suchard
 */

#ifndef __BeagleCPU4StateSSEImpl__
#define __BeagleCPU4StateSSEImpl__

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include "libhmsbeagle/CPU/BeagleCPU4StateImpl.h"

#include <vector>

#define RESTRICT __restrict		/* may need to define this instead to 'restrict' */

#define T_PAD_4_SSE_DEFAULT 2 // Pad transition matrix with 2 rows for SSE
#define P_PAD_4_SSE_DEFAULT 0 // Partials padding not needed for 4 states SSE

#define BEAGLE_CPU_4_SSE_FLOAT       float, T_PAD, P_PAD
#define BEAGLE_CPU_4_SSE_DOUBLE      double, T_PAD, P_PAD
#define BEAGLE_CPU_4_SSE_TEMPLATE    template <int T_PAD, int P_PAD>

namespace beagle {
namespace cpu {

BEAGLE_CPU_TEMPLATE
class BeagleCPU4StateSSEImpl : public BeagleCPU4StateImpl<BEAGLE_CPU_GENERIC> {};


BEAGLE_CPU_4_SSE_TEMPLATE
class BeagleCPU4StateSSEImpl<BEAGLE_CPU_4_SSE_FLOAT> : public BeagleCPU4StateImpl<BEAGLE_CPU_4_SSE_FLOAT> {

protected:
    using BeagleCPUImpl<BEAGLE_CPU_4_SSE_FLOAT>::kTipCount;
    using BeagleCPUImpl<BEAGLE_CPU_4_SSE_FLOAT>::gPartials;
    using BeagleCPUImpl<BEAGLE_CPU_4_SSE_FLOAT>::integrationTmp;
    using BeagleCPUImpl<BEAGLE_CPU_4_SSE_FLOAT>::gTransitionMatrices;
    using BeagleCPUImpl<BEAGLE_CPU_4_SSE_FLOAT>::kPatternCount;
    using BeagleCPUImpl<BEAGLE_CPU_4_SSE_FLOAT>::kPaddedPatternCount;
    using BeagleCPUImpl<BEAGLE_CPU_4_SSE_FLOAT>::kExtraPatterns;
    using BeagleCPUImpl<BEAGLE_CPU_4_SSE_FLOAT>::kStateCount;
    using BeagleCPUImpl<BEAGLE_CPU_4_SSE_FLOAT>::gTipStates;
    using BeagleCPUImpl<BEAGLE_CPU_4_SSE_FLOAT>::kCategoryCount;
    using BeagleCPUImpl<BEAGLE_CPU_4_SSE_FLOAT>::gScaleBuffers;
    using BeagleCPUImpl<BEAGLE_CPU_4_SSE_FLOAT>::gCategoryWeights;
    using BeagleCPUImpl<BEAGLE_CPU_4_SSE_FLOAT>::gStateFrequencies;
    using BeagleCPUImpl<BEAGLE_CPU_4_SSE_FLOAT>::realtypeMin;
    using BeagleCPUImpl<BEAGLE_CPU_4_SSE_FLOAT>::outLogLikelihoodsTmp;
    using BeagleCPUImpl<BEAGLE_CPU_4_SSE_FLOAT>::gPatternWeights;
    using BeagleCPUImpl<BEAGLE_CPU_4_SSE_FLOAT>::gPatternPartitionsStartPatterns;

public:
    virtual const char* getName();

	virtual const long getFlags();

protected:
    virtual int getPaddedPatternsModulus();

private:

	virtual void calcStatesStates(float* destP,
                                  const int* states1,
                                  const float* matrices1,
                                  const int* states2,
                                  const float* matrices2);

    virtual void calcStatesPartials(float* destP,
                                    const int* states1,
                                    const float* __restrict matrices1,
                                    const float* __restrict partials2,
                                    const float* __restrict matrices2);

    virtual void calcStatesPartialsFixedScaling(float* destP,
                                                const int* states1,
                                                const float* __restrict matrices1,
                                                const float* __restrict partials2,
                                                const float* __restrict matrices2,
                                                const float* __restrict scaleFactors);

    virtual void calcPartialsPartials(float* __restrict destP,
                                      const float* __restrict partials1,
                                      const float* __restrict matrices1,
                                      const float* __restrict partials2,
                                      const float* __restrict matrices2);

    virtual void calcPartialsPartialsFixedScaling(float* __restrict destP,
                                                  const float* __restrict child0Partials,
                                                  const float* __restrict child0TransMat,
                                                  const float* __restrict child1Partials,
                                                  const float* __restrict child1TransMat,
                                                  const float* __restrict scaleFactors);

    virtual void calcPartialsPartialsAutoScaling(float* __restrict destP,
                                                 const float* __restrict partials1,
                                                 const float* __restrict matrices1,
                                                 const float* __restrict partials2,
                                                 const float* __restrict matrices2,
                                                 int* activateScaling);

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

};


BEAGLE_CPU_4_SSE_TEMPLATE
class BeagleCPU4StateSSEImpl<BEAGLE_CPU_4_SSE_DOUBLE> : public BeagleCPU4StateImpl<BEAGLE_CPU_4_SSE_DOUBLE> {

protected:
    using BeagleCPUImpl<BEAGLE_CPU_4_SSE_DOUBLE>::kTipCount;
    using BeagleCPUImpl<BEAGLE_CPU_4_SSE_DOUBLE>::gPartials;
    using BeagleCPUImpl<BEAGLE_CPU_4_SSE_DOUBLE>::integrationTmp;
    using BeagleCPUImpl<BEAGLE_CPU_4_SSE_DOUBLE>::gTransitionMatrices;
    using BeagleCPUImpl<BEAGLE_CPU_4_SSE_DOUBLE>::kPatternCount;
    using BeagleCPUImpl<BEAGLE_CPU_4_SSE_DOUBLE>::kPaddedPatternCount;
    using BeagleCPUImpl<BEAGLE_CPU_4_SSE_DOUBLE>::kExtraPatterns;
    using BeagleCPUImpl<BEAGLE_CPU_4_SSE_DOUBLE>::kStateCount;
    using BeagleCPUImpl<BEAGLE_CPU_4_SSE_DOUBLE>::gTipStates;
    using BeagleCPUImpl<BEAGLE_CPU_4_SSE_DOUBLE>::kCategoryCount;
    using BeagleCPUImpl<BEAGLE_CPU_4_SSE_DOUBLE>::gScaleBuffers;
    using BeagleCPUImpl<BEAGLE_CPU_4_SSE_DOUBLE>::gCategoryWeights;
    using BeagleCPUImpl<BEAGLE_CPU_4_SSE_DOUBLE>::gStateFrequencies;
    using BeagleCPUImpl<BEAGLE_CPU_4_SSE_DOUBLE>::realtypeMin;
    using BeagleCPUImpl<BEAGLE_CPU_4_SSE_DOUBLE>::outLogLikelihoodsTmp;
    using BeagleCPUImpl<BEAGLE_CPU_4_SSE_DOUBLE>::gPatternWeights;
    using BeagleCPUImpl<BEAGLE_CPU_4_SSE_DOUBLE>::gPatternPartitionsStartPatterns;
    using BeagleCPUImpl<BEAGLE_CPU_4_SSE_DOUBLE>::grandDenominatorDerivTmp;
    using BeagleCPUImpl<BEAGLE_CPU_4_SSE_DOUBLE>::grandNumeratorDerivTmp;
    using BeagleCPUImpl<BEAGLE_CPU_4_SSE_DOUBLE>::accumulateDerivatives;

public:
    virtual const char* getName();

	virtual const long getFlags();

protected:
    virtual int getPaddedPatternsModulus();

	virtual void accumulateDerivatives(double* outDerivatives,
									   double* outSumDerivatives,
									   double* outSumSquaredDerivatives);

private:

	template <bool DoDerivatives>
	void accumulateDerivativesDispatch1(double* outDerivatives,
										double* outSumDerivatives,
										double* outSumSquaredDerivatives);

	template <bool DoDerivatives, bool DoSum>
	void accumulateDerivativesDispatch2(double* outDerivatives,
										double* outSumDerivatives,
										double* outSumSquaredDerivatives);

	template <bool DoDerivatives, bool DoSum, bool DoSumSquared>
	void accumulateDerivativesImpl(double* outDerivatives,
								   double* outSumDerivatives,
								   double* outSumSquaredDerivatives);

    virtual void calcStatesStates(double* destP,
                                  const int* states1,
                                  const double* matrices1,
                                  const int* states2,
                                  const double* matrices2,
                                  int startPattern,
                                  int endPattern);

    virtual void calcStatesPartials(double* destP,
                                    const int* states1,
                                    const double* __restrict matrices1,
                                    const double* __restrict partials2,
                                    const double* __restrict matrices2,
                                    int startPattern,
                                    int endPattern);

    virtual void calcStatesPartialsFixedScaling(double* destP,
                                                const int* states1,
                                                const double* __restrict matrices1,
                                                const double* __restrict partials2,
                                                const double* __restrict matrices2,
                                                const double* __restrict scaleFactors,
                                                int startPattern,
                                                int endPattern);

    virtual void calcPartialsPartials(double* __restrict destP,
                                      const double* __restrict partials1,
                                      const double* __restrict matrices1,
                                      const double* __restrict partials2,
                                      const double* __restrict matrices2,
                                      int startPattern,
                                      int endPattern);

    virtual void calcPrePartialsPartials(double* __restrict destP,
                                      const double* __restrict partials1,
                                      const double* __restrict matrices1,
                                      const double* __restrict partials2,
                                      const double* __restrict matrices2,
                                      int startPattern,
                                      int endPattern);

    virtual void calcPrePartialsStates(double* __restrict destP,
                                         const double* __restrict partials1,
                                         const double* __restrict matrices1,
                                         const int*               states2,
                                         const double* __restrict matrices2,
                                         int startPattern,
                                         int endPattern);

	virtual void calcCrossProductsStates(const int* __restrict tipStates,
										 const double* __restrict preOrderPartial,
										 const double* __restrict categoryRates,
										 const double* __restrict categoryWeights,
										 const double edgeLength,
										 double* __restrict outCrossProducts,
										 double* __restrict outSumSquaredDerivatives);

	virtual void calcCrossProductsPartials(const double* __restrict postOrderPartial,
										   const double* __restrict preOrderPartial,
										   const double* __restrict categoryRates,
										   const double* __restrict categoryWeights,
										   const double edgeLength,
										   double* __restrict outCrossProducts,
										   double* __restrict outSumSquaredDerivatives);

    virtual void calcEdgeLogDerivativesPartials(const double* __restrict postOrderPartial,
                                                const double* __restrict preOrderPartial,
                                                const int firstDerivativeIndex,
                                                const int secondDerivativeIndex,
                                                const double* __restrict categoryRates,
                                                const double* __restrict categoryWeights,
                                                const int scalingFactorsIndex,
                                                double* outDerivatives,
                                                double* outSumDerivatives,
                                                double* outSumSquaredDerivatives);

    virtual void calcEdgeLogDerivativesStates(const int* tipStates,
                                              const double *__restrict preOrderPartial,
                                              const int firstDerivativeIndex,
                                              const int secondDerivativeIndex,
                                              const double* __restrict categoryRates,
                                              const double* __restrict categoryWeights,
                                              double *outDerivatives,
                                              double *outSumDerivatives,
                                              double *outSumSquaredDerivatives);

    virtual void calcPartialsPartialsFixedScaling(double* __restrict destP,
                                                  const double* __restrict child0Partials,
                                                  const double* __restrict child0TransMat,
                                                  const double* __restrict child1Partials,
                                                  const double* __restrict child1TransMat,
                                                  const double* __restrict scaleFactors,
                                                  int startPattern,
                                                  int endPattern);

    virtual void calcPartialsPartialsAutoScaling(double* __restrict destP,
                                                 const double* __restrict partials1,
                                                 const double* __restrict matrices1,
                                                 const double* __restrict partials2,
                                                 const double* __restrict matrices2,
                                                 int* activateScaling);

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

};


BEAGLE_CPU_FACTORY_TEMPLATE
class BeagleCPU4StateSSEImplFactory : public BeagleImplFactory {
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
#include "libhmsbeagle/CPU/BeagleCPU4StateSSEImpl.hpp"


#endif // __BeagleCPU4StateSSEImpl__
