/*
 *  BeagleCPUSSEImpl.h
 *  BEAGLE
 *
 * Copyright 2010 Phylogenetic Likelihood Working Group
 *
 * This file is part of BEAGLE.
 *
 * BEAGLE is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 *
 * BEAGLE is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with BEAGLE.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @author Marc Suchard
 */

#ifndef __BeagleCPUSSEImpl__
#define __BeagleCPUSSEImpl__

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include "libhmsbeagle/CPU/BeagleCPUImpl.h"

#include <vector>

#define RESTRICT __restrict		/* may need to define this instead to 'restrict' */


// Pad transition matrix rows with an extra 1.0 for ambiguous characters
#define T_PAD_SSE_EVEN  2   // for even state counts
#define T_PAD_SSE_ODD   1   // for odd state counts

// Partials padding
#define P_PAD_SSE_EVEN  0   // for even state counts
#define P_PAD_SSE_ODD   1   // for odd state counts


#define BEAGLE_CPU_SSE_FLOAT	float, T_PAD, P_PAD
#define BEAGLE_CPU_SSE_DOUBLE	double, T_PAD, P_PAD
#define BEAGLE_CPU_SSE_TEMPLATE	template <int T_PAD, int P_PAD>

namespace beagle {
namespace cpu {

BEAGLE_CPU_TEMPLATE
class BeagleCPUSSEImpl : public BeagleCPUImpl<BEAGLE_CPU_GENERIC> {
//    void inline innerPartialsPartals(
//    		const double* __restrict partials1,
//    		const double* __restrict matrices1,
//    		const double* __restrict partials2,
//    		const double* __restrict matrices2,
//    		V_Real& sum1_vec,
//    		V_Real& sum2_vec,
//    		V_Real& out,
//    		int& v,
//    		int& w);

};

BEAGLE_CPU_SSE_TEMPLATE
class BeagleCPUSSEImpl<BEAGLE_CPU_SSE_FLOAT> : public BeagleCPUImpl<BEAGLE_CPU_SSE_FLOAT> {

protected:
	using BeagleCPUImpl<BEAGLE_CPU_SSE_FLOAT>::kTipCount;
	using BeagleCPUImpl<BEAGLE_CPU_SSE_FLOAT>::gPartials;
	using BeagleCPUImpl<BEAGLE_CPU_SSE_FLOAT>::integrationTmp;
	using BeagleCPUImpl<BEAGLE_CPU_SSE_FLOAT>::gTransitionMatrices;
	using BeagleCPUImpl<BEAGLE_CPU_SSE_FLOAT>::kPatternCount;
	using BeagleCPUImpl<BEAGLE_CPU_SSE_FLOAT>::kPaddedPatternCount;
	using BeagleCPUImpl<BEAGLE_CPU_SSE_FLOAT>::kExtraPatterns;
	using BeagleCPUImpl<BEAGLE_CPU_SSE_FLOAT>::kStateCount;
	using BeagleCPUImpl<BEAGLE_CPU_SSE_FLOAT>::gTipStates;
	using BeagleCPUImpl<BEAGLE_CPU_SSE_FLOAT>::kCategoryCount;
	using BeagleCPUImpl<BEAGLE_CPU_SSE_FLOAT>::gScaleBuffers;
	using BeagleCPUImpl<BEAGLE_CPU_SSE_FLOAT>::gCategoryWeights;
	using BeagleCPUImpl<BEAGLE_CPU_SSE_FLOAT>::gStateFrequencies;
	using BeagleCPUImpl<BEAGLE_CPU_SSE_FLOAT>::realtypeMin;
	using BeagleCPUImpl<BEAGLE_CPU_SSE_FLOAT>::kMatrixSize;
	using BeagleCPUImpl<BEAGLE_CPU_SSE_FLOAT>::kPartialsPaddedStateCount;

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
                                    const float* matrices1,
                                    const float* partials2,
                                    const float* matrices2);

    virtual void calcPartialsPartials(float* __restrict destP,
                                      const float* __restrict partials1,
                                      const float* __restrict matrices1,
                                      const float* __restrict partials2,
                                      const float* __restrict matrices2);
    
    virtual void calcPartialsPartialsFixedScaling(float* __restrict destP,
                                      const float* __restrict partials1,
                                      const float* __restrict matrices1,
                                      const float* __restrict partials2,
                                      const float* __restrict matrices2,
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

    
BEAGLE_CPU_SSE_TEMPLATE
class BeagleCPUSSEImpl<BEAGLE_CPU_SSE_DOUBLE> : public BeagleCPUImpl<BEAGLE_CPU_SSE_DOUBLE> {

protected:
	using BeagleCPUImpl<BEAGLE_CPU_SSE_DOUBLE>::kTipCount;
	using BeagleCPUImpl<BEAGLE_CPU_SSE_DOUBLE>::gPartials;
	using BeagleCPUImpl<BEAGLE_CPU_SSE_DOUBLE>::integrationTmp;
	using BeagleCPUImpl<BEAGLE_CPU_SSE_DOUBLE>::gTransitionMatrices;
	using BeagleCPUImpl<BEAGLE_CPU_SSE_DOUBLE>::kPatternCount;
	using BeagleCPUImpl<BEAGLE_CPU_SSE_DOUBLE>::kPaddedPatternCount;
	using BeagleCPUImpl<BEAGLE_CPU_SSE_DOUBLE>::kExtraPatterns;
	using BeagleCPUImpl<BEAGLE_CPU_SSE_DOUBLE>::kStateCount;
	using BeagleCPUImpl<BEAGLE_CPU_SSE_DOUBLE>::gTipStates;
	using BeagleCPUImpl<BEAGLE_CPU_SSE_DOUBLE>::kCategoryCount;
	using BeagleCPUImpl<BEAGLE_CPU_SSE_DOUBLE>::gScaleBuffers;
	using BeagleCPUImpl<BEAGLE_CPU_SSE_DOUBLE>::gCategoryWeights;
	using BeagleCPUImpl<BEAGLE_CPU_SSE_DOUBLE>::gStateFrequencies;
	using BeagleCPUImpl<BEAGLE_CPU_SSE_DOUBLE>::realtypeMin;
	using BeagleCPUImpl<BEAGLE_CPU_SSE_DOUBLE>::kMatrixSize;
	using BeagleCPUImpl<BEAGLE_CPU_SSE_DOUBLE>::kPartialsPaddedStateCount;

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
                                    const double* matrices1,
                                    const double* partials2,
                                    const double* matrices2);

    virtual void calcPartialsPartials(double* __restrict destP,
                                      const double* __restrict partials1,
                                      const double* __restrict matrices1,
                                      const double* __restrict partials2,
                                      const double* __restrict matrices2);
    
    virtual void calcPartialsPartialsFixedScaling(double* __restrict destP,
                                      const double* __restrict partials1,
                                      const double* __restrict matrices1,
                                      const double* __restrict partials2,
                                      const double* __restrict matrices2,
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
class BeagleCPUSSEImplFactory : public BeagleImplFactory {
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
                                   long preferenceFlags,
                                   long requirementFlags,
                                   int* errorCode);

    virtual const char* getName();
    virtual const long getFlags();
};

}	// namespace cpu
}	// namespace beagle

// now include the file containing template function implementations
#include "libhmsbeagle/CPU/BeagleCPUSSEImpl.hpp"


#endif // __BeagleCPUSSEImpl__
