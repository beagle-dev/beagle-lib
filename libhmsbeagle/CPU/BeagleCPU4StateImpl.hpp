
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
 * @author Aaron Darling
 */

#ifndef BEAGLE_CPU_4STATE_IMPL_HPP
#define BEAGLE_CPU_4STATE_IMPL_HPP

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

#define EXPERIMENTAL_OPENMP

#ifdef PAD_MATRICES
    #define OFFSET    5    // For easy conversion between 4/5
#else
    #define OFFSET    4
#endif

#define PREFETCH_MATRIX(num,matrices,w) \
    REALTYPE m##num##00, m##num##01, m##num##02, m##num##03, \
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
    REALTYPE p##num##0, p##num##1, p##num##2, p##num##3; \
    p##num##0 = partials[v + 0]; \
    p##num##1 = partials[v + 1]; \
    p##num##2 = partials[v + 2]; \
    p##num##3 = partials[v + 3];

//#define DO_INTEGRATION(num) \
//    REALTYPE sum##num##0, sum##num##1, sum##num##2, sum##num##3; \
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
    REALTYPE sum##num##0, sum##num##1, sum##num##2, sum##num##3; \
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


namespace beagle {
namespace cpu {

template<typename REALTYPE>
inline const char* getBeagleCPU4StateName(){ return "CPU-4State-Unknown"; };

template<>
inline const char* getBeagleCPU4StateName<double>(){ return "CPU-4State-Double"; };

template<>
inline const char* getBeagleCPU4StateName<float>(){ return "CPU-4State-Single"; };

template <typename REALTYPE>
BeagleCPU4StateImpl<REALTYPE>::~BeagleCPU4StateImpl() {
    // free all that stuff...
    // If you delete partials, make sure not to delete the last element
    // which is TEMP_SCRATCH_PARTIAL twice.
}

///////////////////////////////////////////////////////////////////////////////
// private methods

/*
 * Calculates partial likelihoods at a node when both children have states.
 */
template <typename REALTYPE>
void BeagleCPU4StateImpl<REALTYPE>::calcStatesStates(REALTYPE* destP,
                                     const int* states1,
                                     const REALTYPE* matrices1,
                                     const int* states2,
                                     const REALTYPE* matrices2) {

#pragma omp parallel for num_threads(kCategoryCount)
    for (int l = 0; l < kCategoryCount; l++) {
        int v = l*4*kPaddedPatternCount;
        int w = l*4*OFFSET;

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
    }
}

template <typename REALTYPE>
void BeagleCPU4StateImpl<REALTYPE>::calcStatesStatesFixedScaling(REALTYPE* destP,
                                     const int* states1,
                                     const REALTYPE* matrices1,
                                     const int* states2,
                                     const REALTYPE* matrices2,
                                     const REALTYPE* scaleFactors) {
    
#pragma omp parallel for num_threads(kCategoryCount)
    for (int l = 0; l < kCategoryCount; l++) {
        int v = l*4*kPaddedPatternCount;
        int w = l*4*OFFSET;
        
        for (int k = 0; k < kPatternCount; k++) {
            
            const int state1 = states1[k];
            const int state2 = states2[k];
            const REALTYPE scaleFactor = scaleFactors[k];
            
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
    }
}

/*
 * Calculates partial likelihoods at a node when one child has states and one has partials.
 */
template <typename REALTYPE>
void BeagleCPU4StateImpl<REALTYPE>::calcStatesPartials(REALTYPE* destP,
                                       const int* states1,
                                       const REALTYPE* matrices1,
                                       const REALTYPE* partials2,
                                       const REALTYPE* matrices2) {

#pragma omp parallel for num_threads(kCategoryCount)
    for (int l = 0; l < kCategoryCount; l++) {
        int u = l*4*kPaddedPatternCount;
        int w = l*4*OFFSET;
                
        PREFETCH_MATRIX(2,matrices2,w);
        
        for (int k = 0; k < kPatternCount; k++) {
            
            const int state1 = states1[k];
            
            PREFETCH_PARTIALS(2,partials2,u);
                        
            DO_INTEGRATION(2); // defines sum20, sum21, sum22, sum23;
                        
            destP[u    ] = matrices1[w            + state1] * sum20;
            destP[u + 1] = matrices1[w + OFFSET*1 + state1] * sum21;
            destP[u + 2] = matrices1[w + OFFSET*2 + state1] * sum22;
            destP[u + 3] = matrices1[w + OFFSET*3 + state1] * sum23;
            
            u += 4;
        }
    }
}

template <typename REALTYPE>
void BeagleCPU4StateImpl<REALTYPE>::calcStatesPartialsFixedScaling(REALTYPE* destP,
                                       const int* states1,
                                       const REALTYPE* matrices1,
                                       const REALTYPE* partials2,
                                       const REALTYPE* matrices2,
                                       const REALTYPE* scaleFactors) {
    
#pragma omp parallel for num_threads(kCategoryCount)
    for (int l = 0; l < kCategoryCount; l++) {
        int u = l*4*kPaddedPatternCount;
        int w = l*4*OFFSET;
                
        PREFETCH_MATRIX(2,matrices2,w);
        
        for (int k = 0; k < kPatternCount; k++) {
            
            const int state1 = states1[k];
            const REALTYPE scaleFactor = scaleFactors[k];
            
            PREFETCH_PARTIALS(2,partials2,u);
            
            DO_INTEGRATION(2); // defines sum20, sum21, sum22, sum23
            
            destP[u    ] = matrices1[w            + state1] * sum20 / scaleFactor;
            destP[u + 1] = matrices1[w + OFFSET*1 + state1] * sum21 / scaleFactor;
            destP[u + 2] = matrices1[w + OFFSET*2 + state1] * sum22 / scaleFactor;
            destP[u + 3] = matrices1[w + OFFSET*3 + state1] * sum23 / scaleFactor;
            
            u += 4;            
        }
    }   
}

template <typename REALTYPE>
void BeagleCPU4StateImpl<REALTYPE>::calcPartialsPartials(REALTYPE* destP,
                                         const REALTYPE* partials1,
                                         const REALTYPE* matrices1,
                                         const REALTYPE* partials2,
                                         const REALTYPE* matrices2) {
    
 
#pragma omp parallel for num_threads(kCategoryCount)
    for (int l = 0; l < kCategoryCount; l++) {
        int u = l*4*kPaddedPatternCount;
        int w = l*4*OFFSET;
                
        PREFETCH_MATRIX(1,matrices1,w);                
        PREFETCH_MATRIX(2,matrices2,w);
        for (int k = 0; k < kPatternCount; k++) {                   
            PREFETCH_PARTIALS(1,partials1,u);
            PREFETCH_PARTIALS(2,partials2,u);
            
            DO_INTEGRATION(1); // defines sum10, sum11, sum12, sum13
            DO_INTEGRATION(2); // defines sum20, sum21, sum22, sum23
            
            // Final results
            destP[u    ] = sum10 * sum20;
            destP[u + 1] = sum11 * sum21;
            destP[u + 2] = sum12 * sum22;
            destP[u + 3] = sum13 * sum23;

            u += 4;

        }
    }
}

template <typename REALTYPE>
void BeagleCPU4StateImpl<REALTYPE>::calcPartialsPartialsFixedScaling(REALTYPE* destP,
                                         const REALTYPE* partials1,
                                         const REALTYPE* matrices1,
                                         const REALTYPE* partials2,
                                         const REALTYPE* matrices2,
                                         const REALTYPE* scaleFactors) {
    
#pragma omp parallel for num_threads(kCategoryCount)
    for (int l = 0; l < kCategoryCount; l++) {
        int u = l*4*kPaddedPatternCount;
        int w = l*4*OFFSET;
        
        PREFETCH_MATRIX(1,matrices1,w);
        PREFETCH_MATRIX(2,matrices2,w);
        
        for (int k = 0; k < kPatternCount; k++) {
                        
            // Prefetch scale factor
            const REALTYPE scaleFactor = scaleFactors[k];
            
            PREFETCH_PARTIALS(1,partials1,u);
            PREFETCH_PARTIALS(2,partials2,u);
            
            DO_INTEGRATION(1); // defines sum10, sum11, sum12, sum13
            DO_INTEGRATION(2); // defines sum20, sum21, sum22, sum23
                                
            // Final results
            destP[u    ] = sum10 * sum20 / scaleFactor;
            destP[u + 1] = sum11 * sum21 / scaleFactor;
            destP[u + 2] = sum12 * sum22 / scaleFactor;
            destP[u + 3] = sum13 * sum23 / scaleFactor;
            
            u += 4;
        }
    }    
}

template <typename REALTYPE>
void inline BeagleCPU4StateImpl<REALTYPE>::integrateOutStatesAndScale(const REALTYPE* integrationTmp,
                                                            const double* inStateFrequencies,
                                                            const int scalingFactorsIndex,
                                                            double* outLogLikelihoods) {
    
    register REALTYPE freq0, freq1, freq2, freq3; // Is it a good idea to specify 'register'?
    freq0 = inStateFrequencies[0];   
    freq1 = inStateFrequencies[1];
    freq2 = inStateFrequencies[2];
    freq3 = inStateFrequencies[3];
    
    int u = 0;
    for(int k = 0; k < kPatternCount; k++) {
        REALTYPE sumOverI =
        freq0 * integrationTmp[u    ] +
        freq1 * integrationTmp[u + 1] +
        freq2 * integrationTmp[u + 2] +
        freq3 * integrationTmp[u + 3];
        
        u += 4;        
        outLogLikelihoods[k] = log(sumOverI);
    }        
        
    if (scalingFactorsIndex != BEAGLE_OP_NONE) {
        const REALTYPE* scalingFactors = gScaleBuffers[scalingFactorsIndex];
        for(int k=0; k < kPatternCount; k++)
            outLogLikelihoods[k] += scalingFactors[k];
    }             
}

template <typename REALTYPE>
void BeagleCPU4StateImpl<REALTYPE>::calcEdgeLogLikelihoods(const int parIndex,
                                                 const int childIndex,
                                                 const int probIndex,
                                                 const double* inWeights,
                                                 const double* inStateFrequencies,
                                                 const int scalingFactorsIndex,
                                                 double* outLogLikelihoods) {
    // TODO: implement derivatives for calculateEdgeLnL
    
    assert(parIndex >= kTipCount);
    
    const REALTYPE* partialsParent = gPartials[parIndex];
    const REALTYPE* transMatrix = gTransitionMatrices[probIndex];
    const double* wt = inWeights;
    
    memset(integrationTmp, 0, (kPatternCount * kStateCount)*sizeof(REALTYPE));
    
    if (childIndex < kTipCount && gTipStates[childIndex]) { // Integrate against a state at the child
      
        const int* statesChild = gTipStates[childIndex];    
        int v = 0; // Index for parent partials
        int w = 0;
        for(int l = 0; l < kCategoryCount; l++) {
            int u = 0; // Index in resulting product-partials (summed over categories)
            const REALTYPE weight = wt[l];
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
            if (kExtraPatterns)
            	v += 4 * kExtraPatterns;
        }
        
    } else { // Integrate against a partial at the child
        
        const REALTYPE* partialsChild = gPartials[childIndex];
		#if 0//
        int v = 0;
		#endif
        int w = 0;
        for(int l = 0; l < kCategoryCount; l++) {            
            int u = 0;
			#if 1//
			int v = l*kPaddedPatternCount*4;
			#endif
            const REALTYPE weight = wt[l];
            
            PREFETCH_MATRIX(1,transMatrix,w);
            
            for(int k = 0; k < kPatternCount; k++) {                
                                 
                const REALTYPE* partials1 = partialsChild;
                
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
			#if 0//
            if (kExtraPatterns)
            	v += 4 * kExtraPatterns;
			#endif//
        }
    }
    
    integrateOutStatesAndScale(integrationTmp, inStateFrequencies, scalingFactorsIndex, outLogLikelihoods);
}

template <typename REALTYPE>
void BeagleCPU4StateImpl<REALTYPE>::calcRootLogLikelihoods(const int bufferIndex,
                                                const double* inWeights,
                                                const double* inStateFrequencies,
                                                const int scalingFactorsIndex,
                                                double* outLogLikelihoods) {

    const REALTYPE* rootPartials = gPartials[bufferIndex];
    assert(rootPartials);
    const double* wt = inWeights;
    
    int u = 0;
    int v = 0;
    const REALTYPE wt0 = wt[0];
    for (int k = 0; k < kPatternCount; k++) {
        integrationTmp[v    ] = rootPartials[v    ] * wt0;
        integrationTmp[v + 1] = rootPartials[v + 1] * wt0;
        integrationTmp[v + 2] = rootPartials[v + 2] * wt0;
        integrationTmp[v + 3] = rootPartials[v + 3] * wt0;
        v += 4;
    }
    for (int l = 1; l < kCategoryCount; l++) {
        u = 0;
        const REALTYPE wtl = wt[l];
        for (int k = 0; k < kPatternCount; k++) {
            integrationTmp[u    ] += rootPartials[v    ] * wtl;
            integrationTmp[u + 1] += rootPartials[v + 1] * wtl;
            integrationTmp[u + 2] += rootPartials[v + 2] * wtl;
            integrationTmp[u + 3] += rootPartials[v + 3] * wtl;
             
            u += 4;
            v += 4;
        }
		v += 4 * kExtraPatterns;
    }
    
    integrateOutStatesAndScale(integrationTmp, inStateFrequencies, scalingFactorsIndex, outLogLikelihoods);
}

template <typename REALTYPE>
void BeagleCPU4StateImpl<REALTYPE>::calcRootLogLikelihoodsMulti(const int* bufferIndices,
                                                                const double* inWeights,
                                                                const double* inStateFrequencies,
                                                                const int* scaleBufferIndices,
                                                                int count,
                                                                double* outLogLikelihoods) {
    
    std::vector<int> indexMaxScale(kPatternCount);
    std::vector<REALTYPE> maxScaleFactor(kPatternCount);
    
    for (int subsetIndex = 0 ; subsetIndex < count; ++subsetIndex ) {
        const int rootPartialIndex = bufferIndices[subsetIndex];
        const REALTYPE* rootPartials = gPartials[rootPartialIndex];
        const double* frequencies = inStateFrequencies + (subsetIndex * kStateCount);
        const double* wt = inWeights + subsetIndex * kCategoryCount;
        int u = 0;
        int v = 0;
        
        const REALTYPE wt0 = wt[0];
        for (int k = 0; k < kPatternCount; k++) {
            integrationTmp[v    ] = rootPartials[v    ] * wt0;
            integrationTmp[v + 1] = rootPartials[v + 1] * wt0;
            integrationTmp[v + 2] = rootPartials[v + 2] * wt0;
            integrationTmp[v + 3] = rootPartials[v + 3] * wt0;
            v += 4;
        }
        for (int l = 1; l < kCategoryCount; l++) {
            u = 0;
            const REALTYPE wtl = wt[l];
            for (int k = 0; k < kPatternCount; k++) {
                integrationTmp[u    ] += rootPartials[v    ] * wtl;
                integrationTmp[u + 1] += rootPartials[v + 1] * wtl;
                integrationTmp[u + 2] += rootPartials[v + 2] * wtl;
                integrationTmp[u + 3] += rootPartials[v + 3] * wtl;
                
                u += 4;
                v += 4;
            }
            v += 4 * kExtraPatterns;
        }
                
        register REALTYPE freq0, freq1, freq2, freq3; // Is it a good idea to specify 'register'?
        freq0 = frequencies[0];   
        freq1 = frequencies[1];
        freq2 = frequencies[2];
        freq3 = frequencies[3];
        
        u = 0;
        for (int k = 0; k < kPatternCount; k++) {
            REALTYPE sum = 
                freq0 * integrationTmp[u    ] +
                freq1 * integrationTmp[u + 1] +
                freq2 * integrationTmp[u + 2] +
                freq3 * integrationTmp[u + 3];
            
            u += 4;     
            
            // TODO: allow only some subsets to have scale indices
            if (scaleBufferIndices[0] != BEAGLE_OP_NONE) {
                
                const REALTYPE* cumulativeScaleFactors = gScaleBuffers[scaleBufferIndices[subsetIndex]];
                
                if (subsetIndex == 0) {
                    indexMaxScale[k] = 0;
                    maxScaleFactor[k] = cumulativeScaleFactors[k];
                    for (int j = 1; j < count; j++) {
                        REALTYPE tmpScaleFactor = gScaleBuffers[scaleBufferIndices[j]][k];
                        if (tmpScaleFactor > maxScaleFactor[k]) {
                            indexMaxScale[k] = j;
                            maxScaleFactor[k] = tmpScaleFactor;
                        }
                    }
                }
                
                if (subsetIndex != indexMaxScale[k])
                    sum *= exp((REALTYPE)(cumulativeScaleFactors[k] - maxScaleFactor[k]));
            }
            
            if (subsetIndex == 0)
                outLogLikelihoods[k] = sum;
            else if (subsetIndex == count - 1)
                outLogLikelihoods[k] = log(outLogLikelihoods[k] + sum);
            else
                outLogLikelihoods[k] += sum;
        }
    }
    
    if (scaleBufferIndices[0] != BEAGLE_OP_NONE) {
        for(int i=0; i<kPatternCount; i++)
            outLogLikelihoods[i] += maxScaleFactor[i];
    }
}
    

template <typename REALTYPE>
const char* BeagleCPU4StateImpl<REALTYPE>::getName() {
	return getBeagleCPU4StateName<REALTYPE>();
}

///////////////////////////////////////////////////////////////////////////////
// BeagleCPUImplFactory public methods

template <typename REALTYPE>
BeagleImpl* BeagleCPU4StateImplFactory<REALTYPE>::createImpl(int tipCount,
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

    BeagleImpl* impl = new BeagleCPU4StateImpl<REALTYPE>();

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

template <typename REALTYPE>
const char* BeagleCPU4StateImplFactory<REALTYPE>::getName() {
	return getBeagleCPU4StateName<REALTYPE>();
}

template <typename REALTYPE>
const long BeagleCPU4StateImplFactory<REALTYPE>::getFlags() {
    long flags =  BEAGLE_FLAG_ASYNCH | BEAGLE_FLAG_CPU;
    if (DOUBLE_PRECISION)
    	flags |= BEAGLE_FLAG_DOUBLE;
    else
    	flags |= BEAGLE_FLAG_SINGLE;
    return flags;
}

}	// namespace cpu
}	// namespace beagle

#endif // BEAGLE_CPU_4STATE_IMPL_HPP

