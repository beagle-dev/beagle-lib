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

	using BeagleCPUImpl<REALTYPE>::kTipCount;
	using BeagleCPUImpl<REALTYPE>::gPartials;
	using BeagleCPUImpl<REALTYPE>::integrationTmp;
	using BeagleCPUImpl<REALTYPE>::gTransitionMatrices;
	using BeagleCPUImpl<REALTYPE>::kPatternCount;
	using BeagleCPUImpl<REALTYPE>::kStateCount;
	using BeagleCPUImpl<REALTYPE>::gTipStates;
	using BeagleCPUImpl<REALTYPE>::kCategoryCount;
	using BeagleCPUImpl<REALTYPE>::gScaleBuffers;

public:
    virtual ~BeagleCPU4StateImpl();

protected:
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
    
    virtual void calcRootLogLikelihoods(const int bufferIndex,
                                    const double* inWeights,
                                    const double* inStateFrequencies,
                                    const int scalingFactorsIndex,
                                    double* outLogLikelihoods);
    
    virtual void calcEdgeLogLikelihoods(const int parentBufferIndex,
                                        const int childBufferIndex,
                                        const int probabilityIndex,
                                        const int firstDerivativeIndex,
                                        const int secondDerivativeIndex,
                                        const double* inWeights,
                                        const double* inStateFrequencies,
                                        const int scalingFactorsIndex,
                                        double* outLogLikelihoods,
                                        double* outFirstDerivatives,
                                        double* outSecondDerivatives);
    
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
    
    inline void integrateOutStatesAndScale(const REALTYPE* integrationTmp,
                                           const double* inStateFrequencies,
                                           const int scalingFactorsIndex,
                                           double* outLogLikelihoods);

    virtual const char* getName();
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
