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

#define PREFETCH_MATRIX(num,matrices,w) \
    double m##num##00, m##num##01, m##num##02, m##num##03, \
           m##num##10, m##num##11, m##num##12, m##num##13, \
           m##num##20, m##num##21, m##num##22, m##num##23, \
           m##num##30, m##num##31, m##num##32, m##num##33; \
    m##num##00 = matrices[w + OFFSET*0 + 0]; \
    m##num##01 = matrices[w + OFFSET*0 + 1]; \
    m##num##02 = matrices[w + OFFSET*0 + 2]; \
    m##num##03 = matrices[w + OFFSET*0 + 3]; \
    m##num##10 = matrices[w + OFFSET*1 + 0]; \
    m##num##11 = matrices[w + OFFSET*1 + 1]; \
    m##num##12 = matrices[w + OFFSET*1 + 2]; \
    m##num##13 = matrices[w + OFFSET*1 + 3]; \
    m##num##20 = matrices[w + OFFSET*2 + 0]; \
    m##num##21 = matrices[w + OFFSET*2 + 1]; \
    m##num##22 = matrices[w + OFFSET*2 + 2]; \
    m##num##23 = matrices[w + OFFSET*2 + 3]; \
    m##num##30 = matrices[w + OFFSET*3 + 0]; \
    m##num##31 = matrices[w + OFFSET*3 + 1]; \
    m##num##32 = matrices[w + OFFSET*3 + 2]; \
    m##num##33 = matrices[w + OFFSET*3 + 3];

#define PREFETCH_PARTIALS(num,partials,v) \
    double p##num##0, p##num##1, p##num##2, p##num##3; \
    p##num##0 = partials[v + 0]; \
    p##num##1 = partials[v + 1]; \
    p##num##2 = partials[v + 2]; \
    p##num##3 = partials[v + 3];

//#define DO_INTEGRATION(num) \
//    double sum##num##0, sum##num##1, sum##num##2, sum##num##3; \
//    sum##num##0  = m##num##00 * p##num##0; \
//    sum##num##1  = m##num##10 * p##num##0; \
//    sum##num##2  = m##num##20 * p##num##0; \
//    sum##num##3  = m##num##30 * p##num##0; \
// \
//    sum##num##0 += m##num##01 * p##num##1; \
//    sum##num##1 += m##num##11 * p##num##1; \
//    sum##num##2 += m##num##21 * p##num##1; \
//    sum##num##3 += m##num##31 * p##num##1; \
// \
//    sum##num##0 += m##num##02 * p##num##2; \
//    sum##num##1 += m##num##12 * p##num##2; \
//    sum##num##2 += m##num##22 * p##num##2; \
//    sum##num##3 += m##num##32 * p##num##2; \
// \
//    sum##num##0 += m##num##03 * p##num##3; \
//    sum##num##1 += m##num##13 * p##num##3; \
//    sum##num##2 += m##num##23 * p##num##3; \
//    sum##num##3 += m##num##33 * p##num##3;

#define DO_INTEGRATION(num) \
    double sum##num##0, sum##num##1, sum##num##2, sum##num##3; \
    sum##num##0  = m##num##00 * p##num##0 + \
                   m##num##01 * p##num##1 + \
                   m##num##02 * p##num##2 + \
                   m##num##03 * p##num##3;  \
 \
    sum##num##1  = m##num##10 * p##num##0 + \
                   m##num##11 * p##num##1 + \
                   m##num##12 * p##num##2 + \
                   m##num##13 * p##num##3;  \
 \
    sum##num##2  = m##num##20 * p##num##0 + \
                   m##num##21 * p##num##1 + \
                   m##num##22 * p##num##2 + \
                   m##num##23 * p##num##3;  \
\
    sum##num##3  = m##num##30 * p##num##0 + \
                   m##num##31 * p##num##1 + \
                   m##num##32 * p##num##2 + \
                   m##num##33 * p##num##3;

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
    int w = 0;
    
    for (int l = 0; l < kCategoryCount; l++) {
        
        for (int k = 0; k < kPatternCount; k++) {
            
            const int state1 = states1[k];
            const int state2 = states2[k];
            const double scaleFactor = scaleFactors[k];
            
            destP[v    ] = matrices1[w            + state1] * 
                           matrices2[w            + state2] / scaleFactor;
            destP[v + 1] = matrices1[w + OFFSET*1 + state1] * 
                           matrices2[w + OFFSET*1 + state2] / scaleFactor;
            destP[v + 2] = matrices1[w + OFFSET*2 + state1] * 
                           matrices2[w + OFFSET*2 + state2] / scaleFactor;
            destP[v + 3] = matrices1[w + OFFSET*3 + state1] * 
                           matrices2[w + OFFSET*3 + state2] / scaleFactor;
            v += 4;
        }
        
        w += OFFSET*4;
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

    int u = 0;
    int v = 0;
    int w = 0;
    
    for (int l = 0; l < kCategoryCount; l++) {
                
        PREFETCH_MATRIX(2,matrices2,w);
        
        for (int k = 0; k < kPatternCount; k++) {
            
            const int state1 = states1[k];
            
            PREFETCH_PARTIALS(2,partials2,v);
                        
            DO_INTEGRATION(2); // defines sum20, sum21, sum22, sum23;
                        
            destP[u    ] = matrices1[w            + state1] * sum20;
            destP[u + 1] = matrices1[w + OFFSET*1 + state1] * sum21;
            destP[u + 2] = matrices1[w + OFFSET*2 + state1] * sum22;
            destP[u + 3] = matrices1[w + OFFSET*3 + state1] * sum23;
            
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
    int w = 0;
    
    for (int l = 0; l < kCategoryCount; l++) {
                
        PREFETCH_MATRIX(2,matrices2,w);
        
        for (int k = 0; k < kPatternCount; k++) {
            
            const int state1 = states1[k];
            const double scaleFactor = scaleFactors[k];
            
            PREFETCH_PARTIALS(2,partials2,v);
            
            DO_INTEGRATION(2); // defines sum20, sum21, sum22, sum23
            
            destP[u    ] = matrices1[w            + state1] * sum20 / scaleFactor;
            destP[u + 1] = matrices1[w + OFFSET*1 + state1] * sum21 / scaleFactor;
            destP[u + 2] = matrices1[w + OFFSET*2 + state1] * sum22 / scaleFactor;
            destP[u + 3] = matrices1[w + OFFSET*3 + state1] * sum23 / scaleFactor;
            
            v += 4;
            u += 4;            
        }
        w += OFFSET*4;
    }   
}

void BeagleCPU4StateImpl::calcPartialsPartials(double* destP,
                                         const double* partials1,
                                         const double* matrices1,
                                         const double* partials2,
                                         const double* matrices2) {
    
    int w = 0;

    for (int l = 0; l < kCategoryCount; l++) {
        int x=4*kPatternCount*l;
                
        PREFETCH_MATRIX(1,matrices1,w);                
        PREFETCH_MATRIX(2,matrices2,w);
#pragma omp parallel for 
        for (int k = 0; k < kPatternCount; k++) {                   
            int u=x+k*4;
            int v=x+k*4;
            PREFETCH_PARTIALS(1,partials1,v);
            PREFETCH_PARTIALS(2,partials2,v);
            
            DO_INTEGRATION(1); // defines sum10, sum11, sum12, sum13
            DO_INTEGRATION(2); // defines sum20, sum21, sum22, sum23
            
            // Final results
            destP[u    ] = sum10 * sum20;
            destP[u + 1] = sum11 * sum21;
            destP[u + 2] = sum12 * sum22;
            destP[u + 3] = sum13 * sum23;

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
    
    int u = 0;
    int v = 0;
    int w = 0;
    
    for (int l = 0; l < kCategoryCount; l++) {
        
        PREFETCH_MATRIX(1,matrices1,w);
        PREFETCH_MATRIX(2,matrices2,w);
        
        for (int k = 0; k < kPatternCount; k++) {
                        
            // Prefetch scale factor
            const double scaleFactor = scaleFactors[k];
            
            PREFETCH_PARTIALS(1,partials1,v);
            PREFETCH_PARTIALS(2,partials2,v);
            
            DO_INTEGRATION(1); // defines sum10, sum11, sum12, sum13
            DO_INTEGRATION(2); // defines sum20, sum21, sum22, sum23
                                
            // Final results
            destP[u    ] = sum10 * sum20 / scaleFactor;
            destP[u + 1] = sum11 * sum21 / scaleFactor;
            destP[u + 2] = sum12 * sum22 / scaleFactor;
            destP[u + 3] = sum13 * sum23 / scaleFactor;
            
            u += 4;
            v += 4;            
        }
        w += OFFSET*4;
    }    
}

void inline BeagleCPU4StateImpl::integrateOutStatesAndScale(const double* integrationTmp,
                                                            const double* inStateFrequencies,
                                                            const int scalingFactorsIndex,
                                                            double* outLogLikelihoods) {
    
    register double freq0, freq1, freq2, freq3; // Is it a good idea to specify 'register'?
    freq0 = inStateFrequencies[0];   
    freq1 = inStateFrequencies[1];
    freq2 = inStateFrequencies[2];
    freq3 = inStateFrequencies[3];
    
    int u = 0;
    for(int k = 0; k < kPatternCount; k++) {
        double sumOverI =
        freq0 * integrationTmp[u    ] +
        freq1 * integrationTmp[u + 1] +
        freq2 * integrationTmp[u + 2] +
        freq3 * integrationTmp[u + 3];
        
        u += 4;        
        outLogLikelihoods[k] = log(sumOverI);
    }        
        
    if (scalingFactorsIndex != BEAGLE_OP_NONE) {
        const double* scalingFactors = gScaleBuffers[scalingFactorsIndex];
        for(int k=0; k < kPatternCount; k++)
            outLogLikelihoods[k] += scalingFactors[k];
    }             
}

void BeagleCPU4StateImpl::calcEdgeLogLikelihoods(const int parIndex,
                                                 const int childIndex,
                                                 const int probIndex,
                                                 const int firstDerivativeIndex,
                                                 const int secondDerivativeIndex,
                                                 const double* inWeights,
                                                 const double* inStateFrequencies,
                                                 const int scalingFactorsIndex,
                                                 double* outLogLikelihoods,
                                                 double* outFirstDerivatives,
                                                 double* outSecondDerivatives) {
    // TODO: implement derivatives for calculateEdgeLnL
    // TODO: implement rate categories for calculateEdgeLnL
    
    assert(parIndex >= kTipCount);
    
    const double* partialsParent = gPartials[parIndex];
    const double* transMatrix = gTransitionMatrices[probIndex];
    const double* wt = inWeights;
    
    memset(integrationTmp, 0, (kPatternCount * kStateCount)*sizeof(double));
    
    if (childIndex < kTipCount && gTipStates[childIndex]) { // Integrate against a state at the child
      
        const int* statesChild = gTipStates[childIndex];    
        int v = 0; // Index for parent partials
        int w = 0;
        for(int l = 0; l < kCategoryCount; l++) {
            int u = 0; // Index in resulting product-partials (summed over categories)
            const double weight = wt[l];
            for(int k = 0; k < kPatternCount; k++) {
                
                const int stateChild = statesChild[k]; 
                
                integrationTmp[u    ] += transMatrix[w            + stateChild] * partialsParent[v    ] * weight;                                               
                integrationTmp[u + 1] += transMatrix[w + OFFSET*1 + stateChild] * partialsParent[v + 1] * weight;
                integrationTmp[u + 2] += transMatrix[w + OFFSET*2 + stateChild] * partialsParent[v + 2] * weight;
                integrationTmp[u + 3] += transMatrix[w + OFFSET*3 + stateChild] * partialsParent[v + 3] * weight;
                
                u += 4;
                v += 4;                
            }
            w += OFFSET*4;
        }
        
    } else { // Integrate against a partial at the child
        
        const double* partialsChild = gPartials[childIndex];
        int v = 0;
        int w = 0;
        for(int l = 0; l < kCategoryCount; l++) {            
            int u = 0;
            const double weight = wt[l];        
            
            PREFETCH_MATRIX(1,transMatrix,w);
            
            for(int k = 0; k < kPatternCount; k++) {                
                                 
                const double* partials1 = partialsChild;
                
                PREFETCH_PARTIALS(1,partials1,v);
                
                DO_INTEGRATION(1);
                
                integrationTmp[u    ] += sum10 * partialsParent[v    ] * weight;
                integrationTmp[u + 1] += sum11 * partialsParent[v + 1] * weight;
                integrationTmp[u + 2] += sum12 * partialsParent[v + 2] * weight;
                integrationTmp[u + 3] += sum13 * partialsParent[v + 3] * weight;
                
                u += 4;
                v += 4;
            } 
            w += OFFSET*4;
        }
    }
    
    integrateOutStatesAndScale(integrationTmp, inStateFrequencies, scalingFactorsIndex, outLogLikelihoods);
}

void BeagleCPU4StateImpl::calcRootLogLikelihoods(const int bufferIndex,
                                                const double* inWeights,
                                                const double* inStateFrequencies,
                                                const int scalingFactorsIndex,
                                                double* outLogLikelihoods) {

    const double* rootPartials = gPartials[bufferIndex];
    assert(rootPartials);
    const double* wt = inWeights;
    
    int u = 0;
    int v = 0;
    const double wt0 = wt[0];
    for (int k = 0; k < kPatternCount; k++) {
        integrationTmp[v    ] = rootPartials[v    ] * wt0;
        integrationTmp[v + 1] = rootPartials[v + 1] * wt0;
        integrationTmp[v + 2] = rootPartials[v + 2] * wt0;
        integrationTmp[v + 3] = rootPartials[v + 3] * wt0;
        v += 4;
    }
    for (int l = 1; l < kCategoryCount; l++) {
        u = 0;
        const double wtl = wt[l];
        for (int k = 0; k < kPatternCount; k++) {
            integrationTmp[u    ] += rootPartials[v    ] * wtl;
            integrationTmp[u + 1] += rootPartials[v + 1] * wtl;
            integrationTmp[u + 2] += rootPartials[v + 2] * wtl;
            integrationTmp[u + 3] += rootPartials[v + 3] * wtl;
             
            u += 4;
            v += 4;
        }
    }
    
    integrateOutStatesAndScale(integrationTmp, inStateFrequencies, scalingFactorsIndex, outLogLikelihoods);
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
                                             int resourceNumber,
                                             long preferenceFlags,
                                             long requirementFlags,
                                             int* errorCode) {

    if (stateCount != 4) {
        return NULL;
    }

    BeagleImpl* impl = new BeagleCPU4StateImpl();

    try {
        if (impl->createInstance(tipCount, partialsBufferCount, compactBufferCount, stateCount,
                                 patternCount, eigenBufferCount, matrixBufferCount,
                                 categoryCount,scaleBufferCount, resourceNumber,
                                 preferenceFlags, requirementFlags) == 0)
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

