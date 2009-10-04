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

class BeagleCPU4StateImpl : public BeagleCPUImpl {

public:
    virtual ~BeagleCPU4StateImpl();

protected:
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
    
    virtual void calcPartialsPartials(double* destP,
                                    const double* partials1,
                                    const double* matrices1,
                                    const double* partials2,
                                    const double* matrices2);
    
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
    
    virtual void calcStatesStatesFixedScaling(double *destP,
                                           const int *child0States,
                                        const double *child0TransMat,
                                           const int *child1States,
                                        const double *child1TransMat,
                                        const double *scaleFactors);

    virtual void calcStatesPartialsFixedScaling(double *destP,
                                             const int *child0States,
                                          const double *child0TransMat,
                                          const double *child1Partials,
                                          const double *child1TransMat,
                                          const double *scaleFactors);

    virtual void calcPartialsPartialsFixedScaling(double *destP,
                                            const double *child0Partials,
                                            const double *child0TransMat,
                                            const double *child1Partials,
                                            const double *child1TransMat,
                                            const double *scaleFactors);
    
    inline void integrateOutStatesAndScale(const double* integrationTmp,
                                           const double* inStateFrequencies,
                                           const int scalingFactorsIndex,
                                           double* outLogLikelihoods);
};

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

#endif // __BeagleCPU4StateImpl__
