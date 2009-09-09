/*
 *  BeagleCPU4StateImpl.cpp
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
 * @author Mark Holder
 */

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cstring>
#include <cmath>
#include <cassert>

#include "libhmsbeagle/beagle.h"
#include "libhmsbeagle/CPU/BeagleCPUImpl.h"
#include "libhmsbeagle/CPU/BeagleCPU4StateImpl.h"


#ifdef PAD_MATRICES
    #define OFFSET    5    // For easy conversion between 4/5
#else
    #define OFFSET    4
#endif


using namespace beagle;
using namespace beagle::cpu;

#if defined (BEAGLE_IMPL_DEBUGGING_OUTPUT) && BEAGLE_IMPL_DEBUGGING_OUTPUT
const bool DEBUGGING_OUTPUT = true;
#else
const bool DEBUGGING_OUTPUT = false;
#endif

BeagleCPU4StateImpl::~BeagleCPU4StateImpl() {
    // free all that stuff...
    // If you delete partials, make sure not to delete the last element
    // which is TEMP_SCRATCH_PARTIAL twice.
}

///////////////////////////////////////////////////////////////////////////////
// private methods

/*
 * Calculates partial likelihoods at a node when both children have states.
 */
void BeagleCPU4StateImpl::calcStatesStates(double* destP,
                                     const int* states1,
                                     const double* matrices1,
                                     const int* states2,
                                     const double* matrices2) {

    int v = 0;
    int w = 0;

    for (int l = 0; l < kCategoryCount; l++) {

        for (int k = 0; k < kPatternCount; k++) {

            const int state1 = states1[k];
            const int state2 = states2[k];

            destP[v    ] = matrices1[w            + state1] * 
                           matrices2[w            + state2];
            destP[v + 1] = matrices1[w + OFFSET*1 + state1] * 
                           matrices2[w + OFFSET*1 + state2];
            destP[v + 2] = matrices1[w + OFFSET*2 + state1] * 
                           matrices2[w + OFFSET*2 + state2];
            destP[v + 3] = matrices1[w + OFFSET*3 + state1] * 
                           matrices2[w + OFFSET*3 + state2];
           v += 4;
        }
        
        w += OFFSET*4;
    }
}

void BeagleCPU4StateImpl::calcStatesStatesFixedScaling(double* destP,
                                     const int* states1,
                                     const double* matrices1,
                                     const int* states2,
                                     const double* matrices2,
                                     const double* scaleFactors) {

    int v = 0;
    for (int l = 0; l < kCategoryCount; l++) {

        for (int k = 0; k < kPatternCount; k++) {

            int state1 = states1[k];
            int state2 = states2[k];

            int w = l * kMatrixSize;
            
            const double scaleFactor = scaleFactors[k];

            destP[v] = matrices1[w + state1] * matrices2[w + state2] / scaleFactor;
            v++;    w += 5;
            destP[v] = matrices1[w + state1] * matrices2[w + state2] / scaleFactor;
            v++;    w += 5;
            destP[v] = matrices1[w + state1] * matrices2[w + state2] / scaleFactor;
            v++;    w += 5;
            destP[v] = matrices1[w + state1] * matrices2[w + state2] / scaleFactor;
            v++;    w += 5;

        }
    }
}

/*
 * Calculates partial likelihoods at a node when one child has states and one has partials.
 */
void BeagleCPU4StateImpl::calcStatesPartials(double* destP,
                                       const int* states1,
                                       const double* matrices1,
                                       const double* partials2,
                                       const double* matrices2) {

    double sum0, sum1, sum2, sum3;
    int u = 0;
    int v = 0;
    int w = 0;
    
    for (int l = 0; l < kCategoryCount; l++) {
        for (int k = 0; k < kPatternCount; k++) {

            const int state1 = states1[k];

            sum0  =  matrices2[w               ] * partials2[v    ];
            sum0 +=  matrices2[w            + 1] * partials2[v + 1];
            sum0 +=  matrices2[w            + 2] * partials2[v + 2];
            sum0 +=  matrices2[w            + 3] * partials2[v + 3];

            sum1  =  matrices2[w + OFFSET*1    ] * partials2[v    ];
            sum1 +=  matrices2[w + OFFSET*1 + 1] * partials2[v + 1];
            sum1 +=  matrices2[w + OFFSET*1 + 2] * partials2[v + 2];
            sum1 +=  matrices2[w + OFFSET*1 + 3] * partials2[v + 3];

            sum2  =  matrices2[w + OFFSET*2    ] * partials2[v    ];
            sum2 +=  matrices2[w + OFFSET*2 + 1] * partials2[v + 1];
            sum2 +=  matrices2[w + OFFSET*2 + 2] * partials2[v + 2];
            sum2 +=  matrices2[w + OFFSET*2 + 3] * partials2[v + 3];

            sum3  =  matrices2[w + OFFSET*3    ] * partials2[v    ];
            sum3 +=  matrices2[w + OFFSET*3 + 1] * partials2[v + 1];
            sum3 +=  matrices2[w + OFFSET*3 + 2] * partials2[v + 2];
            sum3 +=  matrices2[w + OFFSET*3 + 3] * partials2[v + 3];
            
            destP[u    ] = matrices1[w            + state1] * sum0;
            destP[u + 1] = matrices1[w + OFFSET*1 + state1] * sum1;
            destP[u + 2] = matrices1[w + OFFSET*2 + state1] * sum2;
            destP[u + 3] = matrices1[w + OFFSET*3 + state1] * sum3;

            v += 4;
            u += 4;

        }
        w += OFFSET*4;
    }
}

void BeagleCPU4StateImpl::calcStatesPartialsFixedScaling(double* destP,
                                       const int* states1,
                                       const double* matrices1,
                                       const double* partials2,
                                       const double* matrices2,
                                       const double* scaleFactors) {
    int u = 0;
    int v = 0;

    for (int l = 0; l < kCategoryCount; l++) {
        for (int k = 0; k < kPatternCount; k++) {

            int state1 = states1[k];

            int w = l * kMatrixSize;
            
            const double scaleFactor = scaleFactors[k];

            destP[u] = matrices1[w + state1];

            double sum = matrices2[w] * partials2[v]; w++;
            sum +=  matrices2[w] * partials2[v + 1]; w++;
            sum +=  matrices2[w] * partials2[v + 2]; w++;
            sum +=  matrices2[w] * partials2[v + 3]; w++;
            w++; // increment for the extra column at the end
            destP[u] *= sum / scaleFactor;    u++;

            destP[u] = matrices1[w + state1];

            sum = matrices2[w] * partials2[v]; w++;
            sum +=  matrices2[w] * partials2[v + 1]; w++;
            sum +=  matrices2[w] * partials2[v + 2]; w++;
            sum +=  matrices2[w] * partials2[v + 3]; w++;
            w++; // increment for the extra column at the end
            destP[u] *= sum / scaleFactor;    u++;

            destP[u] = matrices1[w + state1];

            sum = matrices2[w] * partials2[v]; w++;
            sum +=  matrices2[w] * partials2[v + 1]; w++;
            sum +=  matrices2[w] * partials2[v + 2]; w++;
            sum +=  matrices2[w] * partials2[v + 3]; w++;
            w++; // increment for the extra column at the end
            destP[u] *= sum / scaleFactor;    u++;

            destP[u] = matrices1[w + state1];

            sum = matrices2[w] * partials2[v]; w++;
            sum +=  matrices2[w] * partials2[v + 1]; w++;
            sum +=  matrices2[w] * partials2[v + 2]; w++;
            sum +=  matrices2[w] * partials2[v + 3]; w++;
            w++; // increment for the extra column at the end
            destP[u] *= sum / scaleFactor;    u++;

            v += 4;

        }
    }
}

void BeagleCPU4StateImpl::calcPartialsPartials(double* destP,
                                         const double* partials1,
                                         const double* matrices1,
                                         const double* partials2,
                                         const double* matrices2) {

    double sum01, sum02;
    double sum11, sum12;
    double sum21, sum22;
    double sum31, sum32;
    
    int u = 0;
    int v = 0;
    int w = 0;

    for (int l = 0; l < kCategoryCount; l++) {
        for (int k = 0; k < kPatternCount; k++) {

            sum01  = matrices1[w               ] * partials1[v    ];
            sum01 += matrices1[w            + 1] * partials1[v + 1];
            sum01 += matrices1[w            + 2] * partials1[v + 2];
            sum01 += matrices1[w            + 3] * partials1[v + 3];

            sum02  = matrices2[w               ] * partials2[v    ];
            sum02 += matrices2[w            + 1] * partials2[v + 1];
            sum02 += matrices2[w            + 2] * partials2[v + 2];
            sum02 += matrices2[w            + 3] * partials2[v + 3];

            sum11  = matrices1[w + OFFSET*1    ] * partials1[v    ];
            sum11 += matrices1[w + OFFSET*1 + 1] * partials1[v + 1];
            sum11 += matrices1[w + OFFSET*1 + 2] * partials1[v + 2];
            sum11 += matrices1[w + OFFSET*1 + 3] * partials1[v + 3];

            sum12  = matrices2[w + OFFSET*1    ] * partials2[v    ];
            sum12 += matrices2[w + OFFSET*1 + 1] * partials2[v + 1];
            sum12 += matrices2[w + OFFSET*1 + 2] * partials2[v + 2];
            sum12 += matrices2[w + OFFSET*1 + 3] * partials2[v + 3];

            sum21  = matrices1[w + OFFSET*2    ] * partials1[v    ];
            sum21 += matrices1[w + OFFSET*2 + 1] * partials1[v + 1];
            sum21 += matrices1[w + OFFSET*2 + 2] * partials1[v + 2];
            sum21 += matrices1[w + OFFSET*2 + 3] * partials1[v + 3];
           
            sum22  = matrices2[w + OFFSET*2    ] * partials2[v    ];
            sum22 += matrices2[w + OFFSET*2 + 1] * partials2[v + 1];
            sum22 += matrices2[w + OFFSET*2 + 2] * partials2[v + 2];
            sum22 += matrices2[w + OFFSET*2 + 3] * partials2[v + 3];

            sum31  = matrices1[w + OFFSET*3    ] * partials1[v    ];
            sum31 += matrices1[w + OFFSET*3 + 1] * partials1[v + 1];
            sum31 += matrices1[w + OFFSET*3 + 2] * partials1[v + 2];
            sum31 += matrices1[w + OFFSET*3 + 3] * partials1[v + 3];

            sum32  = matrices2[w + OFFSET*3    ] * partials2[v    ];
            sum32 += matrices2[w + OFFSET*3 + 1] * partials2[v + 1];
            sum32 += matrices2[w + OFFSET*3 + 2] * partials2[v + 2];
            sum32 += matrices2[w + OFFSET*3 + 3] * partials2[v + 3];
            
            destP[u    ] = sum01 * sum02;
            destP[u + 1] = sum11 * sum12;
            destP[u + 2] = sum21 * sum22;
            destP[u + 3] = sum31 * sum32;

            u += 4;
            v += 4;

        }
        w += OFFSET*4;
    }
}

void BeagleCPU4StateImpl::calcPartialsPartialsFixedScaling(double* destP,
                                         const double* partials1,
                                         const double* matrices1,
                                         const double* partials2,
                                         const double* matrices2,
                                         const double* scaleFactors) {

    double sum1, sum2;
    int u = 0;
    int v = 0;

    for (int l = 0; l < kCategoryCount; l++) {
        for (int k = 0; k < kPatternCount; k++) {

            int w = l * kMatrixSize;

            const double scaleFactor = scaleFactors[k];
            
            sum1 = matrices1[w] * partials1[v];
            sum2 = matrices2[w] * partials2[v]; w++;
            sum1 += matrices1[w] * partials1[v + 1];
            sum2 += matrices2[w] * partials2[v + 1]; w++;
            sum1 += matrices1[w] * partials1[v + 2];
            sum2 += matrices2[w] * partials2[v + 2]; w++;
            sum1 += matrices1[w] * partials1[v + 3];
            sum2 += matrices2[w] * partials2[v + 3]; w++;
            w++; // increment for the extra column at the end
            destP[u] = sum1 * sum2 / scaleFactor; u++;

            sum1 = matrices1[w] * partials1[v];
            sum2 = matrices2[w] * partials2[v]; w++;
            sum1 += matrices1[w] * partials1[v + 1];
            sum2 += matrices2[w] * partials2[v + 1]; w++;
            sum1 += matrices1[w] * partials1[v + 2];
            sum2 += matrices2[w] * partials2[v + 2]; w++;
            sum1 += matrices1[w] * partials1[v + 3];
            sum2 += matrices2[w] * partials2[v + 3]; w++;
            w++; // increment for the extra column at the end
            destP[u] = sum1 * sum2 / scaleFactor; u++;

            sum1 = matrices1[w] * partials1[v];
            sum2 = matrices2[w] * partials2[v]; w++;
            sum1 += matrices1[w] * partials1[v + 1];
            sum2 += matrices2[w] * partials2[v + 1]; w++;
            sum1 += matrices1[w] * partials1[v + 2];
            sum2 += matrices2[w] * partials2[v + 2]; w++;
            sum1 += matrices1[w] * partials1[v + 3];
            sum2 += matrices2[w] * partials2[v + 3]; w++;
            w++; // increment for the extra column at the end
            destP[u] = sum1 * sum2 / scaleFactor; u++;

            sum1 = matrices1[w] * partials1[v];
            sum2 = matrices2[w] * partials2[v]; w++;
            sum1 += matrices1[w] * partials1[v + 1];
            sum2 += matrices2[w] * partials2[v + 1]; w++;
            sum1 += matrices1[w] * partials1[v + 2];
            sum2 += matrices2[w] * partials2[v + 2]; w++;
            sum1 += matrices1[w] * partials1[v + 3];
            sum2 += matrices2[w] * partials2[v + 3]; w++;
            w++; // increment for the extra column at the end
            destP[u] = sum1 * sum2 / scaleFactor; u++;
            
            v += 4;

        }
    }
}

void BeagleCPU4StateImpl::calcRootLogLikelihoods(const int bufferIndex,
                                                const double* inWeights,
                                                const double* inStateFrequencies,
                                                const int scalingFactorsIndex,
                                                double* outLogLikelihoods) {

     // We treat this as a special case so that we don't have convoluted logic
     //      at the end of the loop over patterns
     const double* rootPartials = gPartials[bufferIndex];
     assert(rootPartials);
     const double* wt = inWeights;
     int u = 0;
     int v = 0;
     for (int k = 0; k < kPatternCount; k++) {
        integrationTmp[v] = rootPartials[v] * wt[0]; v++;
        integrationTmp[v] = rootPartials[v] * wt[0]; v++;
        integrationTmp[v] = rootPartials[v] * wt[0]; v++;
        integrationTmp[v] = rootPartials[v] * wt[0]; v++;
     }
     for (int l = 1; l < kCategoryCount; l++) {
         u = 0;
         for (int k = 0; k < kPatternCount; k++) {
             integrationTmp[u] += rootPartials[v] * wt[l]; u++; v++;
             integrationTmp[u] += rootPartials[v] * wt[l]; u++; v++;
             integrationTmp[u] += rootPartials[v] * wt[l]; u++; v++;
             integrationTmp[u] += rootPartials[v] * wt[l]; u++; v++;
         }
     }
     u = 0;
     for (int k = 0; k < kPatternCount; k++) {
         double sum = inStateFrequencies[0] * integrationTmp[u]; u++;
         sum += inStateFrequencies[1] * integrationTmp[u]; u++;
         sum += inStateFrequencies[2] * integrationTmp[u]; u++;
         sum += inStateFrequencies[3] * integrationTmp[u]; u++;
         outLogLikelihoods[k] = log(sum);   // take the log
     }
     if (scalingFactorsIndex >=0) {
         const double *cumulativeScaleFactors = gScaleBuffers[scalingFactorsIndex];
         for(int k=0; k<kPatternCount; k++)
             outLogLikelihoods[k] += cumulativeScaleFactors[k];
     }
}

///////////////////////////////////////////////////////////////////////////////
// BeagleCPUImplFactory public methods

BeagleImpl* BeagleCPU4StateImplFactory::createImpl(int tipCount,
                                             int partialsBufferCount,
                                             int compactBufferCount,
                                             int stateCount,
                                             int patternCount,
                                             int eigenBufferCount,
                                             int matrixBufferCount,
                                             int categoryCount,
                                             int scaleBufferCount,
                                             long preferenceFlags,
                                             long requirementFlags) {

	if (stateCount != 4) {
		return NULL;
    }

   	BeagleImpl* impl = new BeagleCPU4StateImpl();

	try {
        if (impl->createInstance(tipCount, partialsBufferCount, compactBufferCount, stateCount,
                                 patternCount, eigenBufferCount, matrixBufferCount,
                                 categoryCount,scaleBufferCount, preferenceFlags, requirementFlags) == 0)
            return impl;
    }
    catch(...) {
        if (DEBUGGING_OUTPUT)
            std::cerr << "exception in initialize\n";
        delete impl;
        throw;
    }

    delete impl;

    return NULL;
}

const char* BeagleCPU4StateImplFactory::getName() {
    return "CPU-4State";
}

const long BeagleCPU4StateImplFactory::getFlags() {
    return BEAGLE_FLAG_ASYNCH | BEAGLE_FLAG_CPU | BEAGLE_FLAG_DOUBLE;
}

