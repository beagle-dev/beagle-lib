/*
 *  BeagleCPU4StateImpl.h
 *  BEAGLE
 *
 * Copyright 2009 Phylogenetic Likelihood Working Group
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

template <typename REALTYPE>
class BeagleCPU4StateImpl : public BeagleCPUImpl<REALTYPE> {

protected:
	using BeagleCPUImpl<REALTYPE>::kFlags;
	using BeagleCPUImpl<REALTYPE>::kTipCount;
	using BeagleCPUImpl<REALTYPE>::gPartials;
	using BeagleCPUImpl<REALTYPE>::integrationTmp;
	using BeagleCPUImpl<REALTYPE>::gTransitionMatrices;
	using BeagleCPUImpl<REALTYPE>::kPatternCount;
	using BeagleCPUImpl<REALTYPE>::kPaddedPatternCount;
	using BeagleCPUImpl<REALTYPE>::kExtraPatterns;
	using BeagleCPUImpl<REALTYPE>::kStateCount;
	using BeagleCPUImpl<REALTYPE>::gTipStates;
	using BeagleCPUImpl<REALTYPE>::kCategoryCount;
	using BeagleCPUImpl<REALTYPE>::gScaleBuffers;
	using BeagleCPUImpl<REALTYPE>::gStateFrequencies;
	using BeagleCPUImpl<REALTYPE>::gCategoryWeights;
	using BeagleCPUImpl<REALTYPE>::gPatternWeights;
	using BeagleCPUImpl<REALTYPE>::outLogLikelihoodsTmp;
	using BeagleCPUImpl<REALTYPE>::realtypeMin;

public:
    virtual ~BeagleCPU4StateImpl();
    virtual const char* getName();


    virtual void calcStatesStates(REALTYPE* destP,
                                    const int* states1,
                                    const REALTYPE* matrices1,
                                    const int* states2,
                                    const REALTYPE* matrices2);
    
    virtual void calcStatesPartials(REALTYPE* destP,
                                    const int* states1,
                                    const REALTYPE* matrices1,
                                    const REALTYPE* partials2,
                                    const REALTYPE* matrices2);
    
    virtual void calcPartialsPartials(REALTYPE* destP,
                                    const REALTYPE* partials1,
                                    const REALTYPE* matrices1,
                                    const REALTYPE* partials2,
                                    const REALTYPE* matrices2);
    
    virtual int calcRootLogLikelihoods(const int bufferIndex,
                                        const int categoryWeightsIndex,
                                        const int stateFrequenciesIndex,
                                        const int scalingFactorsIndex,
                                        double* outSumLogLikelihood);
    
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
    
    virtual void calcStatesStatesFixedScaling(REALTYPE *destP,
                                           const int *child0States,
                                        const REALTYPE *child0TransMat,
                                           const int *child1States,
                                        const REALTYPE *child1TransMat,
                                        const REALTYPE *scaleFactors);

    virtual void calcStatesPartialsFixedScaling(REALTYPE *destP,
                                             const int *child0States,
                                          const REALTYPE *child0TransMat,
                                          const REALTYPE *child1Partials,
                                          const REALTYPE *child1TransMat,
                                          const REALTYPE *scaleFactors);

    virtual void calcPartialsPartialsFixedScaling(REALTYPE *destP,
                                            const REALTYPE *child0Partials,
                                            const REALTYPE *child0TransMat,
                                            const REALTYPE *child1Partials,
                                            const REALTYPE *child1TransMat,
                                            const REALTYPE *scaleFactors);
    
    inline int integrateOutStatesAndScale(const REALTYPE* integrationTmp,
                                           const int stateFrequenciesIndex,
                                           const int scalingFactorsIndex,
                                           double* outSumLogLikelihood);

};

template <typename REALTYPE>
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
