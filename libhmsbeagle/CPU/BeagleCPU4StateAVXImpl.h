/*
 *  BeagleCPU4StateAVXImpl.h
 *  BEAGLE
 *
 * Copyright 2013 Phylogenetic Likelihood Working Group
 *
 * This file is part of BEAGLE.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 *
 * @author Marc Suchard
 */

#ifndef __BeagleCPU4StateAVXImpl__
#define __BeagleCPU4StateAVXImpl__

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include "libhmsbeagle/CPU/BeagleCPU4StateImpl.h"

#include <vector>

#define RESTRICT __restrict		/* may need to define this instead to 'restrict' */

#define T_PAD_4_AVX_DEFAULT 2 // Pad transition matrix with 2 rows for AVX
#define P_PAD_4_AVX_DEFAULT 0 // Partials padding not needed for 4 states AVX

#define BEAGLE_CPU_4_AVX_FLOAT       float, T_PAD, P_PAD
#define BEAGLE_CPU_4_AVX_DOUBLE      double, T_PAD, P_PAD
#define BEAGLE_CPU_4_AVX_TEMPLATE    template <int T_PAD, int P_PAD>

namespace beagle {
namespace cpu {

BEAGLE_CPU_TEMPLATE
class BeagleCPU4StateAVXImpl : public BeagleCPU4StateImpl<BEAGLE_CPU_GENERIC> {};


BEAGLE_CPU_4_AVX_TEMPLATE
class BeagleCPU4StateAVXImpl<BEAGLE_CPU_4_AVX_FLOAT> : public BeagleCPU4StateImpl<BEAGLE_CPU_4_AVX_FLOAT> {

protected:
    using BeagleCPUImpl<BEAGLE_CPU_4_AVX_FLOAT>::kTipCount;
    using BeagleCPUImpl<BEAGLE_CPU_4_AVX_FLOAT>::gPartials;
    using BeagleCPUImpl<BEAGLE_CPU_4_AVX_FLOAT>::integrationTmp;
    using BeagleCPUImpl<BEAGLE_CPU_4_AVX_FLOAT>::gTransitionMatrices;
    using BeagleCPUImpl<BEAGLE_CPU_4_AVX_FLOAT>::kPatternCount;
    using BeagleCPUImpl<BEAGLE_CPU_4_AVX_FLOAT>::kPaddedPatternCount;
    using BeagleCPUImpl<BEAGLE_CPU_4_AVX_FLOAT>::kExtraPatterns;
    using BeagleCPUImpl<BEAGLE_CPU_4_AVX_FLOAT>::kStateCount;
    using BeagleCPUImpl<BEAGLE_CPU_4_AVX_FLOAT>::gTipStates;
    using BeagleCPUImpl<BEAGLE_CPU_4_AVX_FLOAT>::kCategoryCount;
    using BeagleCPUImpl<BEAGLE_CPU_4_AVX_FLOAT>::gScaleBuffers;
    using BeagleCPUImpl<BEAGLE_CPU_4_AVX_FLOAT>::gCategoryWeights;
    using BeagleCPUImpl<BEAGLE_CPU_4_AVX_FLOAT>::gStateFrequencies;
    using BeagleCPUImpl<BEAGLE_CPU_4_AVX_FLOAT>::realtypeMin;
    using BeagleCPUImpl<BEAGLE_CPU_4_AVX_FLOAT>::outLogLikelihoodsTmp;
    using BeagleCPUImpl<BEAGLE_CPU_4_AVX_FLOAT>::gPatternWeights;

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

};


BEAGLE_CPU_4_AVX_TEMPLATE
class BeagleCPU4StateAVXImpl<BEAGLE_CPU_4_AVX_DOUBLE> : public BeagleCPU4StateImpl<BEAGLE_CPU_4_AVX_DOUBLE> {

protected:
    using BeagleCPUImpl<BEAGLE_CPU_4_AVX_DOUBLE>::kTipCount;
    using BeagleCPUImpl<BEAGLE_CPU_4_AVX_DOUBLE>::gPartials;
    using BeagleCPUImpl<BEAGLE_CPU_4_AVX_DOUBLE>::integrationTmp;
    using BeagleCPUImpl<BEAGLE_CPU_4_AVX_DOUBLE>::gTransitionMatrices;
    using BeagleCPUImpl<BEAGLE_CPU_4_AVX_DOUBLE>::kPatternCount;
    using BeagleCPUImpl<BEAGLE_CPU_4_AVX_DOUBLE>::kPaddedPatternCount;
    using BeagleCPUImpl<BEAGLE_CPU_4_AVX_DOUBLE>::kExtraPatterns;
    using BeagleCPUImpl<BEAGLE_CPU_4_AVX_DOUBLE>::kStateCount;
    using BeagleCPUImpl<BEAGLE_CPU_4_AVX_DOUBLE>::gTipStates;
    using BeagleCPUImpl<BEAGLE_CPU_4_AVX_DOUBLE>::kCategoryCount;
    using BeagleCPUImpl<BEAGLE_CPU_4_AVX_DOUBLE>::gScaleBuffers;
    using BeagleCPUImpl<BEAGLE_CPU_4_AVX_DOUBLE>::gCategoryWeights;
    using BeagleCPUImpl<BEAGLE_CPU_4_AVX_DOUBLE>::gStateFrequencies;
    using BeagleCPUImpl<BEAGLE_CPU_4_AVX_DOUBLE>::realtypeMin;
    using BeagleCPUImpl<BEAGLE_CPU_4_AVX_DOUBLE>::outLogLikelihoodsTmp;
    using BeagleCPUImpl<BEAGLE_CPU_4_AVX_DOUBLE>::gPatternWeights;

public:
    virtual const char* getName();

	virtual const long getFlags();

protected:
    virtual int getPaddedPatternsModulus();

private:

    virtual void calcStatesStates(double* destP,
                                  const int* states1,
                                  const double* matrices1,
                                  const int* states2,
                                  const double* matrices2);

    virtual void calcStatesPartials(double* destP,
                                    const int* states1,
                                    const double* __restrict matrices1,
                                    const double* __restrict partials2,
                                    const double* __restrict matrices2);

    virtual void calcStatesPartialsFixedScaling(double* destP,
                                                const int* states1,
                                                const double* __restrict matrices1,
                                                const double* __restrict partials2,
                                                const double* __restrict matrices2,
                                                const double* __restrict scaleFactors);

    virtual void calcPartialsPartials(double* __restrict destP,
                                      const double* __restrict partials1,
                                      const double* __restrict matrices1,
                                      const double* __restrict partials2,
                                      const double* __restrict matrices2);

    virtual void calcPartialsPartialsFixedScaling(double* __restrict destP,
                                                  const double* __restrict child0Partials,
                                                  const double* __restrict child0TransMat,
                                                  const double* __restrict child1Partials,
                                                  const double* __restrict child1TransMat,
                                                  const double* __restrict scaleFactors);

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

};


BEAGLE_CPU_FACTORY_TEMPLATE
class BeagleCPU4StateAVXImplFactory : public BeagleImplFactory {
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
#include "libhmsbeagle/CPU/BeagleCPU4StateAVXImpl.hpp"


#endif // __BeagleCPU4StateAVXImpl__
