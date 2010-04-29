/*
 *  BeagleCPUImpl.cpp
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

///@TODO: wrap partials, eigen calcs, and transition matrices in a small structs
//      so that we can flag them. This would this would be helpful for
//      implementing:
//          1. an error-checking version that double-checks (to the extent
//              possible) that the client is using the API correctly.  This would
//              ideally be a  conditional compilation variant (so that we do
//              not normally incur runtime penalties, but can enable it to help
//              find bugs).
//          2. a multithreading impl that checks dependencies before queuing
//              partials.

///@API-ISSUE: adding an resizePartialsBufferArray(int newPartialsBufferCount) method
//      would be trivial for this impl, and would be easier for clients that want
//      to cache partial like calculations for a indeterminate number of trees.
///@API-ISSUE: adding a
//  void waitForPartials(int* instance;
//                  int instanceCount;
//                  int* parentPartialIndex;
//                  int partialCount;
//                  );
//  method that blocks until the partials are valid would be important for
//  clients (such as GARLI) that deal with big trees by overwriting some temporaries.
///@API-ISSUE: Swapping temporaries (we decided not to implement the following idea
//  but MTH did want to record it for posterity). We could add following
//  calls:
////////////////////////////////////////////////////////////////////////////////
// BeagleReturnCodes swapEigens(int instance, int *firstInd, int *secondInd, int count);
// BeagleReturnCodes swapTransitionMatrices(int instance, int *firstInd, int *secondInd, int count);
// BeagleReturnCodes swapPartials(int instance, int *firstInd, int *secondInd, int count);
////////////////////////////////////////////////////////////////////////////////
//  They would be optional for the client but could improve efficiency if:
//      1. The impl is load balancing, AND
//      2. The client code, uses the calls to synchronize the indexing of temporaries
//          between instances such that you can pass an instanceIndices list with
//          multiple entries to updatePartials.
//  These seem too nitty gritty and low-level, but they also make it easy to
//      write a wrapper geared toward MCMC (during a move, cache the old data
//      in an unused array, after a rejection swap back to the cached copy)

#ifndef BEAGLE_CPU_IMPL_GENERAL_HPP
#define BEAGLE_CPU_IMPL_GENERAL_HPP

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cstring>
#include <cmath>
#include <cassert>
#include <vector>
#include <cfloat>

#include "libhmsbeagle/beagle.h"
#include "libhmsbeagle/CPU/Precision.h"
#include "libhmsbeagle/CPU/BeagleCPUImpl.h"
#include "libhmsbeagle/CPU/EigenDecompositionCube.h"
#include "libhmsbeagle/CPU/EigenDecompositionSquare.h"

namespace beagle {
namespace cpu {

//#if defined (BEAGLE_IMPL_DEBUGGING_OUTPUT) && BEAGLE_IMPL_DEBUGGING_OUTPUT
//const bool DEBUGGING_OUTPUT = true;
//#else
//const bool DEBUGGING_OUTPUT = false;
//#endif

template<typename REALTYPE>
inline const char* getBeagleCPUName(){ return "CPU-Unknown"; };

template<>
inline const char* getBeagleCPUName<double>(){ return "CPU-Double"; };

template<>
inline const char* getBeagleCPUName<float>(){ return "CPU-Single"; };

template<typename REALTYPE>
inline const long getBeagleCPUFlags(){ return BEAGLE_FLAG_COMPUTATION_SYNCH; };

template<>
inline const long getBeagleCPUFlags<double>(){ return BEAGLE_FLAG_COMPUTATION_SYNCH |
                                                      BEAGLE_FLAG_THREADING_NONE |
                                                      BEAGLE_FLAG_PROCESSOR_CPU |
                                                      BEAGLE_FLAG_PRECISION_DOUBLE |
                                                      BEAGLE_FLAG_VECTOR_NONE; };

template<>
inline const long getBeagleCPUFlags<float>(){ return BEAGLE_FLAG_COMPUTATION_SYNCH |
                                                     BEAGLE_FLAG_THREADING_NONE |
                                                     BEAGLE_FLAG_PROCESSOR_CPU |
                                                     BEAGLE_FLAG_PRECISION_SINGLE |
                                                     BEAGLE_FLAG_VECTOR_NONE; };



template <typename REALTYPE>
BeagleCPUImpl<REALTYPE>::~BeagleCPUImpl() {
    // free all that stuff...
    // If you delete partials, make sure not to delete the last element
    // which is TEMP_SCRATCH_PARTIAL twice.

    for(unsigned int i=0; i<kEigenDecompCount; i++) {
	    if (gCategoryWeights[i] != NULL)
		    free(gCategoryWeights[i]);
        if (gStateFrequencies[i] != NULL)
		    free(gStateFrequencies[i]);
	}

	for(unsigned int i=0; i<kMatrixCount; i++) {
	    if (gTransitionMatrices[i] != NULL)
		    free(gTransitionMatrices[i]);
	}
    free(gTransitionMatrices);

	for(unsigned int i=0; i<kBufferCount; i++) {
	    if (gPartials[i] != NULL)
		    free(gPartials[i]);
	    if (gTipStates[i] != NULL)
		    free(gTipStates[i]);
	}
    free(gPartials);
    free(gTipStates);
    
    if (kFlags & BEAGLE_FLAG_SCALING_AUTO) {
        for(unsigned int i=0; i<kScaleBufferCount; i++) {
            if (gAutoScaleBuffers[i] != NULL)
                free(gAutoScaleBuffers[i]);
        }
        if (gAutoScaleBuffers)
            free(gAutoScaleBuffers);
        free(gActiveScalingFactors);
        if (gScaleBuffers[0] != NULL)
            free(gScaleBuffers[0]);
    } else {
        for(unsigned int i=0; i<kScaleBufferCount; i++) {
            if (gScaleBuffers[i] != NULL)
                free(gScaleBuffers[i]);
        }        
    }
    
    if (gScaleBuffers)
        free(gScaleBuffers);

	free(gCategoryRates);
    free(gPatternWeights);

	free(integrationTmp);
    free(firstDerivTmp);
    free(secondDerivTmp);

    free(outLogLikelihoodsTmp);
    free(outFirstDerivativesTmp);
    free(outSecondDerivativesTmp);

	free(ones);
	free(zeros);

	delete gEigenDecomposition;
}

template <typename REALTYPE>
int BeagleCPUImpl<REALTYPE>::createInstance(int tipCount,
                                  int partialsBufferCount,
                                  int compactBufferCount,
                                  int stateCount,
                                  int patternCount,
                                  int eigenDecompositionCount,
                                  int matrixCount,
                                  int categoryCount,
                                  int scaleBufferCount,
                                  int resourceNumber,
                                  long preferenceFlags,
                                  long requirementFlags) {
    if (DEBUGGING_OUTPUT)
        std::cerr << "in BeagleCPUImpl::initialize\n" ;

    if (DOUBLE_PRECISION) {
        realtypeMin = DBL_MIN;
        scalingExponentThreshhold = 200;
    } else {
        realtypeMin = FLT_MIN;
        scalingExponentThreshhold = 20;
    }

    kBufferCount = partialsBufferCount + compactBufferCount;
    kTipCount = tipCount;
    assert(kBufferCount > kTipCount);
    kStateCount = stateCount;
    kPatternCount = patternCount;
    
    kInternalPartialsBufferCount = kBufferCount - kTipCount;

    // Handle possible padding of pattern sites for vectorization
    int modulus = getPaddedPatternsModulus();
    kPaddedPatternCount = kPatternCount;
    int remainder = kPatternCount % modulus;
    if (remainder != 0) {
    	kPaddedPatternCount += modulus - remainder;
    }
    kExtraPatterns = kPaddedPatternCount - kPatternCount;

    kMatrixCount = matrixCount;
    kEigenDecompCount = eigenDecompositionCount;
	kCategoryCount = categoryCount;
    kScaleBufferCount = scaleBufferCount;
#ifdef PAD_MATRICES
    kMatrixSize = (PAD + kStateCount) * kStateCount;
#else
    kMatrixSize = kStateCount * kStateCount;
#endif

    int scaleBufferSize = kPaddedPatternCount;
    
    kFlags = 0;

    if (preferenceFlags & BEAGLE_FLAG_SCALING_AUTO || requirementFlags & BEAGLE_FLAG_SCALING_AUTO) {
        kFlags |= BEAGLE_FLAG_SCALING_AUTO;
        kFlags |= BEAGLE_FLAG_SCALERS_LOG;
        kScaleBufferCount = kInternalPartialsBufferCount;
    } else
    if (preferenceFlags & BEAGLE_FLAG_SCALING_ALWAYS || requirementFlags & BEAGLE_FLAG_SCALING_ALWAYS) {
        kFlags |= BEAGLE_FLAG_SCALING_ALWAYS;
    	kFlags |= BEAGLE_FLAG_SCALERS_LOG;
        kScaleBufferCount = kInternalPartialsBufferCount + 1; // +1 for temp buffer used by edgelikelihood
    } else if (preferenceFlags & BEAGLE_FLAG_SCALERS_LOG || requirementFlags & BEAGLE_FLAG_SCALERS_LOG) {
        kFlags |= BEAGLE_FLAG_SCALING_MANUAL;
    	kFlags |= BEAGLE_FLAG_SCALERS_LOG;
    } else {
        kFlags |= BEAGLE_FLAG_SCALING_MANUAL;
        kFlags |= BEAGLE_FLAG_SCALERS_RAW;
    }
    
    if (requirementFlags & BEAGLE_FLAG_EIGEN_COMPLEX || preferenceFlags & BEAGLE_FLAG_EIGEN_COMPLEX)
    	kFlags |= BEAGLE_FLAG_EIGEN_COMPLEX;
    else
        kFlags |= BEAGLE_FLAG_EIGEN_REAL;

    if (kFlags & BEAGLE_FLAG_EIGEN_COMPLEX)
    	gEigenDecomposition = new EigenDecompositionSquare<REALTYPE>(kEigenDecompCount,
    			kStateCount,kCategoryCount,kFlags);
    else
    	gEigenDecomposition = new EigenDecompositionCube<REALTYPE>(kEigenDecompCount,
    			kStateCount, kCategoryCount);

	gCategoryRates = (double*) malloc(sizeof(double) * kCategoryCount);
	if (gCategoryRates == NULL)
		throw std::bad_alloc();

	gPatternWeights = (double*) malloc(sizeof(double) * kPatternCount);
	if (gPatternWeights == NULL)
		throw std::bad_alloc();

    // TODO: if pattern padding is implemented this will create problems with setTipPartials
    kPartialsSize = kPaddedPatternCount * kStateCount * kCategoryCount;

    gPartials = (REALTYPE**) malloc(sizeof(REALTYPE*) * kBufferCount);
    if (gPartials == NULL)
     throw std::bad_alloc();

    gStateFrequencies = (REALTYPE**) calloc(sizeof(REALTYPE*), kEigenDecompCount);
    if (gStateFrequencies == NULL)
        throw std::bad_alloc();

    gCategoryWeights = (REALTYPE**) calloc(sizeof(REALTYPE*), kEigenDecompCount);
    if (gCategoryWeights == NULL)
        throw std::bad_alloc();

    // assigning kBufferCount to this array so that we can just check if a tipStateBuffer is
    // allocated
    gTipStates = (int**) malloc(sizeof(int*) * kBufferCount);
    if (gTipStates == NULL)
        throw std::bad_alloc();

    for (int i = 0; i < kBufferCount; i++) {
        gPartials[i] = NULL;
        gTipStates[i] = NULL;
    }

    for (int i = kTipCount; i < kBufferCount; i++) {
        gPartials[i] = (REALTYPE*) mallocAligned(sizeof(REALTYPE) * kPartialsSize);
        if (gPartials[i] == NULL)
            throw std::bad_alloc();
    }

    gScaleBuffers = NULL;

    gAutoScaleBuffers = NULL;

    if (kFlags & BEAGLE_FLAG_SCALING_AUTO) {
        gAutoScaleBuffers = (signed short**) malloc(sizeof(signed short*) * kScaleBufferCount);
        if (gAutoScaleBuffers == NULL)
            throw std::bad_alloc();        
        for (int i = 0; i < kScaleBufferCount; i++) {
            gAutoScaleBuffers[i] = (signed short*) malloc(sizeof(signed short) * scaleBufferSize);
            if (gAutoScaleBuffers[i] == 0L)
                throw std::bad_alloc();
        }
        gActiveScalingFactors = (int*) malloc(sizeof(int) * kInternalPartialsBufferCount);
        gScaleBuffers = (REALTYPE**) malloc(sizeof(REALTYPE*));
        gScaleBuffers[0] = (REALTYPE*) malloc(sizeof(REALTYPE) * scaleBufferSize);
    } else {
        gScaleBuffers = (REALTYPE**) malloc(sizeof(REALTYPE*) * kScaleBufferCount);
        if (gScaleBuffers == NULL)
            throw std::bad_alloc();
        
        for (int i = 0; i < kScaleBufferCount; i++) {
            gScaleBuffers[i] = (REALTYPE*) malloc(sizeof(REALTYPE) * scaleBufferSize);
            if (gScaleBuffers[i] == 0L)
                throw std::bad_alloc();
        }
    }
        

    gTransitionMatrices = (REALTYPE**) malloc(sizeof(REALTYPE*) * kMatrixCount);
    if (gTransitionMatrices == NULL)
        throw std::bad_alloc();
    for (int i = 0; i < kMatrixCount; i++) {
        gTransitionMatrices[i] = (REALTYPE*) mallocAligned(sizeof(REALTYPE) * kMatrixSize * kCategoryCount);
        if (gTransitionMatrices[i] == 0L)
            throw std::bad_alloc();
    }

    integrationTmp = (REALTYPE*) mallocAligned(sizeof(REALTYPE) * kPatternCount * kStateCount);
    firstDerivTmp = (REALTYPE*) malloc(sizeof(REALTYPE) * kPatternCount * kStateCount);
    secondDerivTmp = (REALTYPE*) malloc(sizeof(REALTYPE) * kPatternCount * kStateCount);

    outLogLikelihoodsTmp = (REALTYPE*) malloc(sizeof(REALTYPE) * kPatternCount * kStateCount);
    outFirstDerivativesTmp = (REALTYPE*) malloc(sizeof(REALTYPE) * kPatternCount * kStateCount);
    outSecondDerivativesTmp = (REALTYPE*) malloc(sizeof(REALTYPE) * kPatternCount * kStateCount);

    zeros = (REALTYPE*) malloc(sizeof(REALTYPE) * kPaddedPatternCount);
    ones = (REALTYPE*) malloc(sizeof(REALTYPE) * kPaddedPatternCount);
    for(int i = 0; i < kPaddedPatternCount; i++) {
    	zeros[i] = 0.0;
        ones[i] = 1.0;
    }

    return BEAGLE_SUCCESS;
}

template <typename REALTYPE>
const char* BeagleCPUImpl<REALTYPE>::getName() {
	return getBeagleCPUName<REALTYPE>();
}

template <typename REALTYPE>
const long BeagleCPUImpl<REALTYPE>::getFlags() {
	return getBeagleCPUFlags<REALTYPE>();
}

template <typename REALTYPE>
int BeagleCPUImpl<REALTYPE>::getInstanceDetails(BeagleInstanceDetails* returnInfo) {
    if (returnInfo != NULL) {
        returnInfo->resourceNumber = 0;
        returnInfo->flags = getFlags();
        returnInfo->flags |= kFlags;

        returnInfo->implName = (char*) getName();
    }

    return BEAGLE_SUCCESS;
}

template <typename REALTYPE>
int BeagleCPUImpl<REALTYPE>::setTipStates(int tipIndex,
                                const int* inStates) {
    if (tipIndex < 0 || tipIndex >= kTipCount)
        return BEAGLE_ERROR_OUT_OF_RANGE;
    gTipStates[tipIndex] = (int*) mallocAligned(sizeof(int) * kPaddedPatternCount);
    // TODO: What if this throws a memory full error?
	for (int j = 0; j < kPatternCount; j++) {
		gTipStates[tipIndex][j] = (inStates[j] < kStateCount ? inStates[j] : kStateCount);
	}
	for (int j = kPatternCount; j < kPaddedPatternCount; j++) {
		gTipStates[tipIndex][j] = kStateCount;
	}

    return BEAGLE_SUCCESS;
}

template <typename REALTYPE>
int BeagleCPUImpl<REALTYPE>::setTipPartials(int tipIndex,
                                  const double* inPartials) {
    if (tipIndex < 0 || tipIndex >= kTipCount)
        return BEAGLE_ERROR_OUT_OF_RANGE;
    if(gPartials[tipIndex] == NULL) {
        gPartials[tipIndex] = (REALTYPE*) mallocAligned(sizeof(REALTYPE) * kPartialsSize);
        // TODO: What if this throws a memory full error?
        if (gPartials[tipIndex] == 0L)
            return BEAGLE_ERROR_OUT_OF_MEMORY;
    }
    int singlePartialsSize = kPatternCount * kStateCount;
    REALTYPE *partials = gPartials[tipIndex];
    for (int i = 0; i < kCategoryCount; i++) {
    	beagleMemCpy(partials, inPartials, singlePartialsSize);
    	partials += singlePartialsSize;
    	// Pad extra buffer with zeros
    	for(int k = 0; k < kStateCount * (kPaddedPatternCount - kPatternCount); k++) {
    		*partials++ = 0;
    	}
    }

    return BEAGLE_SUCCESS;
}

template <typename REALTYPE>
int BeagleCPUImpl<REALTYPE>::setPartials(int bufferIndex,
                               const double* inPartials) {
    if (bufferIndex < 0 || bufferIndex >= kBufferCount)
        return BEAGLE_ERROR_OUT_OF_RANGE;
    if (gPartials[bufferIndex] == NULL) {
        gPartials[bufferIndex] = (REALTYPE*) malloc(sizeof(REALTYPE) * kPartialsSize);
        if (gPartials[bufferIndex] == 0L)
            return BEAGLE_ERROR_OUT_OF_MEMORY;
    }
	beagleMemCpy(gPartials[bufferIndex], inPartials, kPartialsSize);

    return BEAGLE_SUCCESS;
}

template <typename REALTYPE>
int BeagleCPUImpl<REALTYPE>::getPartials(int bufferIndex,
                               int cumulativeScaleIndex,
                               double* outPartials) {
	// TODO: Test with and without padding
    if (bufferIndex < 0 || bufferIndex >= kBufferCount)
        return BEAGLE_ERROR_OUT_OF_RANGE;

    if (kPatternCount == kPaddedPatternCount) {
    	beagleMemCpy(outPartials, gPartials[bufferIndex], kPartialsSize);
    } else { // Need to remove padding
    	double *offsetOutPartials;
    	REALTYPE* offsetBeaglePartials = gPartials[bufferIndex];
    	for(int i = 0; i < kCategoryCount; i++) {
    		beagleMemCpy(offsetOutPartials,offsetBeaglePartials,
    				kPatternCount * kStateCount);
    		offsetOutPartials += kPatternCount * kStateCount;
    		offsetBeaglePartials += kPaddedPatternCount * kStateCount;
    	}
    }

    if (cumulativeScaleIndex != BEAGLE_OP_NONE) {
    	REALTYPE* cumulativeScaleBuffer = gScaleBuffers[cumulativeScaleIndex];
    	int index = 0;
    	for(int k=0; k<kPatternCount; k++) {
    		REALTYPE scaleFactor = exp(cumulativeScaleBuffer[k]);
    		for(int i=0; i<kStateCount; i++) {
    			outPartials[index] *= scaleFactor;
    			index++;
    		}
    	}
    	// TODO: Do we assume the cumulativeScaleBuffer is on the log-scale?
    }

    return BEAGLE_SUCCESS;
}

template <typename REALTYPE>
int BeagleCPUImpl<REALTYPE>::setEigenDecomposition(int eigenIndex,
                                         const double* inEigenVectors,
                                         const double* inInverseEigenVectors,
                                         const double* inEigenValues) {

	gEigenDecomposition->setEigenDecomposition(eigenIndex, inEigenVectors, inInverseEigenVectors, inEigenValues);
	return BEAGLE_SUCCESS;
}

template <typename REALTYPE>
int BeagleCPUImpl<REALTYPE>::setCategoryRates(const double* inCategoryRates) {
	memcpy(gCategoryRates, inCategoryRates, sizeof(double) * kCategoryCount);
    return BEAGLE_SUCCESS;
}

template <typename REALTYPE>
int BeagleCPUImpl<REALTYPE>::setPatternWeights(const double* inPatternWeights) {
    memcpy(gPatternWeights, inPatternWeights, sizeof(double) * kPatternCount);
    return BEAGLE_SUCCESS;
}

template <typename REALTYPE>
    int BeagleCPUImpl<REALTYPE>::setStateFrequencies(int stateFrequenciesIndex,
                                                     const double* inStateFrequencies) {
    if (stateFrequenciesIndex < 0 || stateFrequenciesIndex >= kEigenDecompCount)
        return BEAGLE_ERROR_OUT_OF_RANGE;
    if (gStateFrequencies[stateFrequenciesIndex] == NULL) {
        gStateFrequencies[stateFrequenciesIndex] = (REALTYPE*) malloc(sizeof(REALTYPE) * kStateCount);
        if (gStateFrequencies[stateFrequenciesIndex] == 0L)
            return BEAGLE_ERROR_OUT_OF_MEMORY;
    }
    beagleMemCpy(gStateFrequencies[stateFrequenciesIndex], inStateFrequencies, kStateCount);

    return BEAGLE_SUCCESS;
}

template <typename REALTYPE>
int BeagleCPUImpl<REALTYPE>::setCategoryWeights(int categoryWeightsIndex,
                                                 const double* inCategoryWeights) {
    if (categoryWeightsIndex < 0 || categoryWeightsIndex >= kEigenDecompCount)
        return BEAGLE_ERROR_OUT_OF_RANGE;
    if (gCategoryWeights[categoryWeightsIndex] == NULL) {
        gCategoryWeights[categoryWeightsIndex] = (REALTYPE*) malloc(sizeof(REALTYPE) * kCategoryCount);
        if (gCategoryWeights[categoryWeightsIndex] == 0L)
            return BEAGLE_ERROR_OUT_OF_MEMORY;
    }
    beagleMemCpy(gCategoryWeights[categoryWeightsIndex], inCategoryWeights, kCategoryCount);

    return BEAGLE_SUCCESS;
}

template <typename REALTYPE>
int BeagleCPUImpl<REALTYPE>::getTransitionMatrix(int matrixIndex,
												 double* outMatrix) {
	// TODO Test with multiple rate categories
#ifdef PAD_MATRICES
	double* offsetOutMatrix = outMatrix;
	REALTYPE* offsetBeagleMatrix = gTransitionMatrices[matrixIndex];
	for(int i = 0; i < kCategoryCount; i++) {
		for(int j = 0; j < kStateCount; j++) {
			beagleMemCpy(offsetOutMatrix,offsetBeagleMatrix,kStateCount);
			offsetBeagleMatrix += kStateCount + PAD; // Skip padding
			offsetOutMatrix += kStateCount;
		}
	}
#else
	beagleMemCpy(outMatrix,gTransitionMatrices[matrixIndex],
			kMatrixSize * kCategoryCount);
#endif
	return BEAGLE_SUCCESS;
}

template <typename REALTYPE>
int BeagleCPUImpl<REALTYPE>::getSiteLogLikelihoods(double* outLogLikelihoods) {
    beagleMemCpy(outLogLikelihoods, outLogLikelihoodsTmp, kPatternCount);

    return BEAGLE_SUCCESS;
}

template <typename REALTYPE>
int BeagleCPUImpl<REALTYPE>::getSiteDerivatives(double* outFirstDerivatives,
                                                double* outSecondDerivatives) {
    beagleMemCpy(outFirstDerivatives, outFirstDerivativesTmp, kPatternCount);
    if (outSecondDerivatives != NULL)
        beagleMemCpy(outSecondDerivatives, outSecondDerivativesTmp, kPatternCount);

    return BEAGLE_SUCCESS;
}


template <typename REALTYPE>
int BeagleCPUImpl<REALTYPE>::setTransitionMatrix(int matrixIndex,
                                       const double* inMatrix) {

#ifdef PAD_MATRICES
    const double* offsetInMatrix = inMatrix;
    REALTYPE* offsetBeagleMatrix = gTransitionMatrices[matrixIndex];
    for(int i = 0; i < kCategoryCount; i++) {
        for(int j = 0; j < kStateCount; j++) {
            beagleMemCpy(offsetBeagleMatrix, offsetInMatrix, kStateCount);
            offsetBeagleMatrix[kStateCount] = 1.0;
            offsetBeagleMatrix += kStateCount + PAD; // Skip padding
            offsetInMatrix += kStateCount;
        }
    }
#else
    beagleMemCpy(gTransitionMatrices[matrixIndex], inMatrix,
                 kMatrixSize * kCategoryCount);
#endif
    return BEAGLE_SUCCESS;
}

template <typename REALTYPE>
int BeagleCPUImpl<REALTYPE>::updateTransitionMatrices(int eigenIndex,
                                            const int* probabilityIndices,
                                            const int* firstDerivativeIndices,
                                            const int* secondDerivativeIndices,
                                            const double* edgeLengths,
                                            int count) {
	gEigenDecomposition->updateTransitionMatrices(eigenIndex,probabilityIndices,firstDerivativeIndices,secondDerivativeIndices,
												  edgeLengths,gCategoryRates,gTransitionMatrices,count);
	return BEAGLE_SUCCESS;
}

template <typename REALTYPE>
int BeagleCPUImpl<REALTYPE>::updatePartials(const int* operations,
                                  int count,
                                  int cumulativeScaleIndex) {

    REALTYPE* cumulativeScaleBuffer = NULL;
    if (cumulativeScaleIndex != BEAGLE_OP_NONE)
        cumulativeScaleBuffer = gScaleBuffers[cumulativeScaleIndex];

    for (int op = 0; op < count; op++) {
        if (DEBUGGING_OUTPUT) {
            std::cerr << "op[0]= " << operations[0] << "\n";
            std::cerr << "op[1]= " << operations[1] << "\n";
            std::cerr << "op[2]= " << operations[2] << "\n";
            std::cerr << "op[3]= " << operations[3] << "\n";
            std::cerr << "op[4]= " << operations[4] << "\n";
            std::cerr << "op[5]= " << operations[5] << "\n";
            std::cerr << "op[6]= " << operations[6] << "\n";
        }

        const int parIndex = operations[op * 7];
        const int writeScalingIndex = operations[op * 7 + 1];
        const int readScalingIndex = operations[op * 7 + 2];
        const int child1Index = operations[op * 7 + 3];
        const int child1TransMatIndex = operations[op * 7 + 4];
        const int child2Index = operations[op * 7 + 5];
        const int child2TransMatIndex = operations[op * 7 + 6];

        const REALTYPE* partials1 = gPartials[child1Index];
        const REALTYPE* partials2 = gPartials[child2Index];

        const int* tipStates1 = gTipStates[child1Index];
        const int* tipStates2 = gTipStates[child2Index];

        const REALTYPE* matrices1 = gTransitionMatrices[child1TransMatIndex];
        const REALTYPE* matrices2 = gTransitionMatrices[child2TransMatIndex];

        REALTYPE* destPartials = gPartials[parIndex];

        int rescale = BEAGLE_OP_NONE;
        REALTYPE* scalingFactors = NULL;
        
        if (kFlags & BEAGLE_FLAG_SCALING_AUTO) {
            gActiveScalingFactors[parIndex - kTipCount] = 0;
            if (tipStates1 == 0 && tipStates2 == 0)
                rescale = 2;
        } else if (kFlags & BEAGLE_FLAG_SCALING_ALWAYS) {
            rescale = 1;
            scalingFactors = gScaleBuffers[parIndex - kTipCount];
        } else if (writeScalingIndex >= 0) {
            rescale = 1;
            scalingFactors = gScaleBuffers[writeScalingIndex];
        } else if (readScalingIndex >= 0) {
            rescale = 0;
            scalingFactors = gScaleBuffers[readScalingIndex];
        }

        if (DEBUGGING_OUTPUT) {
            std::cerr << "Rescale= " << rescale << " writeIndex= " << writeScalingIndex
                     << " readIndex = " << readScalingIndex << "\n";
        }

        if (tipStates1 != NULL) {
            if (tipStates2 != NULL ) {
                if (rescale == 0) { // Use fixed scaleFactors
                    calcStatesStatesFixedScaling(destPartials, tipStates1, matrices1, tipStates2, matrices2,
                                                 scalingFactors);
                } else {
                    // First compute without any scaling
                    calcStatesStates(destPartials, tipStates1, matrices1, tipStates2, matrices2);
                    if (rescale == 1) // Recompute scaleFactors
                        rescalePartials(destPartials,scalingFactors,cumulativeScaleBuffer,1);
                }
            } else {
                if (rescale == 0) {
                    calcStatesPartialsFixedScaling(destPartials, tipStates1, matrices1, partials2, matrices2,
                                                   scalingFactors);
                } else {
                    calcStatesPartials(destPartials, tipStates1, matrices1, partials2, matrices2);
                    if (rescale == 1)
                        rescalePartials(destPartials,scalingFactors,cumulativeScaleBuffer,0);
                }
            }
        } else {
            if (tipStates2 != NULL) {
                if (rescale == 0) {
                    calcStatesPartialsFixedScaling(destPartials,tipStates2,matrices2,partials1,matrices1,
                                                   scalingFactors);
                } else {
                    calcStatesPartials(destPartials, tipStates2, matrices2, partials1, matrices1);
                    if (rescale == 1)
                        rescalePartials(destPartials,scalingFactors,cumulativeScaleBuffer,0);
                }
            } else {
                if (rescale == 2) {
                    int sIndex = parIndex - kTipCount;
                    calcPartialsPartialsAutoScaling(destPartials,partials1,matrices1,partials2,matrices2,
                                                     &gActiveScalingFactors[sIndex]);
                    if (gActiveScalingFactors[sIndex])
                        autoRescalePartials(destPartials, gAutoScaleBuffers[sIndex]);

                } else if (rescale == 0) {
                    calcPartialsPartialsFixedScaling(destPartials,partials1,matrices1,partials2,matrices2,
                                                     scalingFactors);
                } else {
                    calcPartialsPartials(destPartials, partials1, matrices1, partials2, matrices2);
                    if (rescale == 1)
                        rescalePartials(destPartials,scalingFactors,cumulativeScaleBuffer,0);
                }
            }
        }
        
        if (kFlags & BEAGLE_FLAG_SCALING_ALWAYS) {
            int parScalingIndex = parIndex - kTipCount;
            int child1ScalingIndex = child1Index - kTipCount;
            int child2ScalingIndex = child2Index - kTipCount;
            if (child1ScalingIndex >= 0 && child2ScalingIndex >= 0) {
                int scalingIndices[2] = {child1ScalingIndex, child2ScalingIndex};
                accumulateScaleFactors(scalingIndices, 2, parScalingIndex);
            } else if (child1ScalingIndex >= 0) {
                int scalingIndices[1] = {child1ScalingIndex};
                accumulateScaleFactors(scalingIndices, 1, parScalingIndex);
            } else if (child2ScalingIndex >= 0) {
                int scalingIndices[1] = {child2ScalingIndex};
                accumulateScaleFactors(scalingIndices, 1, parScalingIndex);
            }
        }
        
        if (DEBUGGING_OUTPUT) {
            if (scalingFactors != NULL && rescale == 0) {
                for(int i=0; i<kPatternCount; i++)
                    fprintf(stderr,"old scaleFactor[%d] = %.5f\n",i,scalingFactors[i]);
            }
            fprintf(stderr,"Result partials:\n");
            for(int i = 0; i < kPartialsSize; i++)
                fprintf(stderr,"destP[%d] = %.5f\n",i,destPartials[i]);
        }
    }

    return BEAGLE_SUCCESS;
}


template <typename REALTYPE>
int BeagleCPUImpl<REALTYPE>::waitForPartials(const int* destinationPartials,
                                   int destinationPartialsCount) {
    return BEAGLE_SUCCESS;
}


template <typename REALTYPE>
    int BeagleCPUImpl<REALTYPE>::calculateRootLogLikelihoods(const int* bufferIndices,
                                                             const int* categoryWeightsIndices,
                                                             const int* stateFrequenciesIndices,
                                                             const int* cumulativeScaleIndices,
                                                             int count,
                                                             double* outSumLogLikelihood) {

    if (count == 1) {
        // We treat this as a special case so that we don't have convoluted logic
        //      at the end of the loop over patterns
        int cumulativeScalingFactorIndex;
        if (kFlags & BEAGLE_FLAG_SCALING_AUTO)
            cumulativeScalingFactorIndex = 0;
        else if (kFlags & BEAGLE_FLAG_SCALING_ALWAYS)
            cumulativeScalingFactorIndex = bufferIndices[0] - kTipCount; 
        else
            cumulativeScalingFactorIndex = cumulativeScaleIndices[0];
        return calcRootLogLikelihoods(bufferIndices[0], categoryWeightsIndices[0], stateFrequenciesIndices[0],
                               cumulativeScalingFactorIndex, outSumLogLikelihood);
    }
    else
    {
        return calcRootLogLikelihoodsMulti(bufferIndices, categoryWeightsIndices, stateFrequenciesIndices,
                                    cumulativeScaleIndices, count, outSumLogLikelihood);
    }
}

template <typename REALTYPE>
int BeagleCPUImpl<REALTYPE>::calcRootLogLikelihoodsMulti(const int* bufferIndices,
                                                         const int* categoryWeightsIndices,
                                                         const int* stateFrequenciesIndices,
                                                         const int* scaleBufferIndices,
                                                         int count,
                                                         double* outSumLogLikelihood) {
    // Here we do the 3 similar operations:
    //              1. to set the lnL to the contribution of the first subset,
    //              2. to add the lnL for other subsets up to the penultimate
    //              3. to add the final subset and take the lnL
    //      This form of the calc would not work when count == 1 because
    //              we need operation 1 and 3 in the preceding list.  This is not
    //              a problem, though as we deal with count == 1 in the previous
    //              branch.

    std::vector<int> indexMaxScale(kPatternCount);
    std::vector<REALTYPE> maxScaleFactor(kPatternCount);

    int returnCode = BEAGLE_SUCCESS;

    for (int subsetIndex = 0 ; subsetIndex < count; ++subsetIndex ) {
        const int rootPartialIndex = bufferIndices[subsetIndex];
        const REALTYPE* rootPartials = gPartials[rootPartialIndex];
        const REALTYPE* frequencies = gStateFrequencies[stateFrequenciesIndices[subsetIndex]];
        const REALTYPE* wt = gCategoryWeights[categoryWeightsIndices[subsetIndex]];
        int u = 0;
        int v = 0;
        for (int k = 0; k < kPatternCount; k++) {
            for (int i = 0; i < kStateCount; i++) {
                integrationTmp[u] = rootPartials[v] * (REALTYPE) wt[0];
                u++;
                v++;
            }
        }
        for (int l = 1; l < kCategoryCount; l++) {
            u = 0;
            for (int k = 0; k < kPatternCount; k++) {
                for (int i = 0; i < kStateCount; i++) {
                    integrationTmp[u] += rootPartials[v] * (REALTYPE) wt[l];
                    u++;
                    v++;
                }
            }
        }
        u = 0;
        for (int k = 0; k < kPatternCount; k++) {
            REALTYPE sum = 0.0;
            for (int i = 0; i < kStateCount; i++) {
                sum += ((REALTYPE)frequencies[i]) * integrationTmp[u];
                u++;
            }

            // TODO: allow only some subsets to have scale indices
            if (scaleBufferIndices[0] != BEAGLE_OP_NONE || (kFlags & BEAGLE_FLAG_SCALING_ALWAYS)) {
                int cumulativeScalingFactorIndex;
                if (kFlags & BEAGLE_FLAG_SCALING_ALWAYS)
                    cumulativeScalingFactorIndex = rootPartialIndex - kTipCount; 
                else
                    cumulativeScalingFactorIndex = scaleBufferIndices[subsetIndex];
                
                const REALTYPE* cumulativeScaleFactors = gScaleBuffers[cumulativeScalingFactorIndex];

                if (subsetIndex == 0) {
                    indexMaxScale[k] = 0;
                    maxScaleFactor[k] = cumulativeScaleFactors[k];
                    for (int j = 1; j < count; j++) {
                        REALTYPE tmpScaleFactor;
                        if (kFlags & BEAGLE_FLAG_SCALING_ALWAYS)
                            tmpScaleFactor = gScaleBuffers[bufferIndices[j] - kTipCount][k]; 
                        else
                            tmpScaleFactor = gScaleBuffers[scaleBufferIndices[j]][k];

                        if (tmpScaleFactor > maxScaleFactor[k]) {
                            indexMaxScale[k] = j;
                            maxScaleFactor[k] = tmpScaleFactor;
                        }
                    }
                }

                if (subsetIndex != indexMaxScale[k])
                    sum *= exp((REALTYPE)(cumulativeScaleFactors[k] - maxScaleFactor[k]));
            }

            if (subsetIndex == 0) {
                outLogLikelihoodsTmp[k] = sum;
            } else if (subsetIndex == count - 1) {
                REALTYPE tmpSum = outLogLikelihoodsTmp[k] + sum;

                if (!(tmpSum >= realtypeMin))
                    returnCode = BEAGLE_ERROR_FLOATING_POINT;

                outLogLikelihoodsTmp[k] = log(tmpSum);
            } else {
                outLogLikelihoodsTmp[k] += sum;
            }
        }
    }

    if (scaleBufferIndices[0] != BEAGLE_OP_NONE || (kFlags & BEAGLE_FLAG_SCALING_ALWAYS)) {
        for(int i=0; i<kPatternCount; i++)
            outLogLikelihoodsTmp[i] += maxScaleFactor[i];
    }

    *outSumLogLikelihood = 0.0;
    for (int i = 0; i < kPatternCount; i++) {
        *outSumLogLikelihood += outLogLikelihoodsTmp[i] * gPatternWeights[i];
    }

    return returnCode;

}

template <typename REALTYPE>
int BeagleCPUImpl<REALTYPE>::calcRootLogLikelihoods(const int bufferIndex,
                            const int categoryWeightsIndex,
                            const int stateFrequenciesIndex,
                            const int scalingFactorsIndex,
                            double* outSumLogLikelihood) {

    int returnCode = BEAGLE_SUCCESS;

    const REALTYPE* rootPartials = gPartials[bufferIndex];
    const REALTYPE* wt = gCategoryWeights[categoryWeightsIndex];
    const REALTYPE* freqs = gStateFrequencies[stateFrequenciesIndex];
    int u = 0;
    int v = 0;
    for (int k = 0; k < kPatternCount; k++) {
        for (int i = 0; i < kStateCount; i++) {
            integrationTmp[u] = rootPartials[v] * (REALTYPE) wt[0];
            u++;
            v++;
        }
    }
    for (int l = 1; l < kCategoryCount; l++) {
        u = 0;
        for (int k = 0; k < kPatternCount; k++) {
            for (int i = 0; i < kStateCount; i++) {
                integrationTmp[u] += rootPartials[v] * (REALTYPE) wt[l];
                u++;
                v++;
            }
        }
    }
    u = 0;
    for (int k = 0; k < kPatternCount; k++) {
    	REALTYPE sum = 0.0;
        for (int i = 0; i < kStateCount; i++) {
            sum += freqs[i] * integrationTmp[u];
            u++;
        }

        if (!(sum >= realtypeMin))
            returnCode = BEAGLE_ERROR_FLOATING_POINT;

        outLogLikelihoodsTmp[k] = log(sum);
    }

    if (scalingFactorsIndex >= 0) {
    	const REALTYPE* cumulativeScaleFactors = gScaleBuffers[scalingFactorsIndex];
    	for(int i=0; i<kPatternCount; i++) {
    		outLogLikelihoodsTmp[i] += cumulativeScaleFactors[i];
        }
    }

    *outSumLogLikelihood = 0.0;
    for (int i = 0; i < kPatternCount; i++) {
        *outSumLogLikelihood += outLogLikelihoodsTmp[i] * gPatternWeights[i];
    }

    // TODO: merge the three kPatternCount loops above into one

    return returnCode;
}

template <typename REALTYPE>
int BeagleCPUImpl<REALTYPE>::accumulateScaleFactors(const int* scalingIndices,
                                                int  count,
                                                int  cumulativeScalingIndex) {
    if (kFlags & BEAGLE_FLAG_SCALING_AUTO) {
        REALTYPE* cumulativeScaleBuffer = gScaleBuffers[0];
        for(int j=0; j<kPatternCount; j++)
            cumulativeScaleBuffer[j] =  0;
        for(int i=0; i<count; i++) {
            int sIndex = scalingIndices[i] - kTipCount;
            if (gActiveScalingFactors[sIndex]) {
                const signed short* scaleBuffer = gAutoScaleBuffers[sIndex];
                for(int j=0; j<kPatternCount; j++) {
                    cumulativeScaleBuffer[j] += M_LN2 * scaleBuffer[j];
                }
            }
        }
                
    } else {
        REALTYPE* cumulativeScaleBuffer = gScaleBuffers[cumulativeScalingIndex];
        for(int i=0; i<count; i++) {
            const REALTYPE* scaleBuffer = gScaleBuffers[scalingIndices[i]];
            for(int j=0; j<kPatternCount; j++) {
                if (kFlags & BEAGLE_FLAG_SCALERS_LOG)
                    cumulativeScaleBuffer[j] += scaleBuffer[j];
                else
                    cumulativeScaleBuffer[j] += log(scaleBuffer[j]);
            }
        }

        if (DEBUGGING_OUTPUT) {
            fprintf(stderr,"Accumulating %d scale buffers into #%d\n",count,cumulativeScalingIndex);
            for(int j=0; j<kPatternCount; j++) {
                fprintf(stderr,"cumulativeScaleBuffer[%d] = %2.5e\n",j,cumulativeScaleBuffer[j]);
            }
        }
    }
    
    return BEAGLE_SUCCESS;
}

template <typename REALTYPE>
int BeagleCPUImpl<REALTYPE>::removeScaleFactors(const int* scalingIndices,
                                            int  count,
                                            int  cumulativeScalingIndex) {
	REALTYPE* cumulativeScaleBuffer = gScaleBuffers[cumulativeScalingIndex];
    for(int i=0; i<count; i++) {
        const REALTYPE* scaleBuffer = gScaleBuffers[scalingIndices[i]];
        for(int j=0; j<kPatternCount; j++) {
            if (kFlags & BEAGLE_FLAG_SCALERS_LOG)
                cumulativeScaleBuffer[j] -= scaleBuffer[j];
            else
                cumulativeScaleBuffer[j] -= log(scaleBuffer[j]);
        }
    }

    return BEAGLE_SUCCESS;
}

template <typename REALTYPE>
int BeagleCPUImpl<REALTYPE>::resetScaleFactors(int cumulativeScalingIndex) {
    //memcpy(gScaleBuffers[cumulativeScalingIndex],zeros,sizeof(double) * kPatternCount);
    memset(gScaleBuffers[cumulativeScalingIndex], 0, sizeof(REALTYPE) * kPatternCount);
    return BEAGLE_SUCCESS;
}

template <typename REALTYPE>
    int BeagleCPUImpl<REALTYPE>::calculateEdgeLogLikelihoods(const int* parentBufferIndices,
                                                             const int* childBufferIndices,
                                                             const int* probabilityIndices,
                                                             const int* firstDerivativeIndices,
                                                             const int* secondDerivativeIndices,
                                                             const int* categoryWeightsIndices,
                                                             const int* stateFrequenciesIndices,
                                                             const int* cumulativeScaleIndices,
                                                             int count,
                                                             double* outSumLogLikelihood,
                                                             double* outSumFirstDerivative,
                                                             double* outSumSecondDerivative) {
    // TODO: implement for count > 1

    if (count == 1) {
        int cumulativeScalingFactorIndex;
        if (kFlags & BEAGLE_FLAG_SCALING_AUTO) {
            cumulativeScalingFactorIndex = 0;
        } else if (kFlags & BEAGLE_FLAG_SCALING_ALWAYS) {
            cumulativeScalingFactorIndex = kInternalPartialsBufferCount;
            int child1ScalingIndex = parentBufferIndices[0] - kTipCount;
            int child2ScalingIndex = childBufferIndices[0] - kTipCount;
            resetScaleFactors(cumulativeScalingFactorIndex);
            if (child1ScalingIndex >= 0 && child2ScalingIndex >= 0) {
                int scalingIndices[2] = {child1ScalingIndex, child2ScalingIndex};
                accumulateScaleFactors(scalingIndices, 2, cumulativeScalingFactorIndex);
            } else if (child1ScalingIndex >= 0) {
                int scalingIndices[1] = {child1ScalingIndex};
                accumulateScaleFactors(scalingIndices, 1, cumulativeScalingFactorIndex);
            } else if (child2ScalingIndex >= 0) {
                int scalingIndices[1] = {child2ScalingIndex};
                accumulateScaleFactors(scalingIndices, 1, cumulativeScalingFactorIndex);
            }
        } else {
            cumulativeScalingFactorIndex = cumulativeScaleIndices[0];
        }
        
		if (firstDerivativeIndices == NULL && secondDerivativeIndices == NULL)
			return calcEdgeLogLikelihoods(parentBufferIndices[0], childBufferIndices[0], probabilityIndices[0],
                                   categoryWeightsIndices[0], stateFrequenciesIndices[0], cumulativeScalingFactorIndex,
                                   outSumLogLikelihood);
		else if (secondDerivativeIndices == NULL)
			return calcEdgeLogLikelihoodsFirstDeriv(parentBufferIndices[0], childBufferIndices[0], probabilityIndices[0],
                                             firstDerivativeIndices[0], categoryWeightsIndices[0], stateFrequenciesIndices[0],
                                             cumulativeScalingFactorIndex, outSumLogLikelihood, outSumFirstDerivative);
		else
			return calcEdgeLogLikelihoodsSecondDeriv(parentBufferIndices[0], childBufferIndices[0], probabilityIndices[0],
                                              firstDerivativeIndices[0], secondDerivativeIndices[0], categoryWeightsIndices[0],
                                              stateFrequenciesIndices[0], cumulativeScalingFactorIndex, outSumLogLikelihood,
                                              outSumFirstDerivative, outSumSecondDerivative);
    } else {
        fprintf(stderr,"BeagleCPUImpl::calculateEdgeLogLikelihoods not yet implemented for count > 1\n");
        return BEAGLE_ERROR_OUT_OF_RANGE;
    }

}

template <typename REALTYPE>
int BeagleCPUImpl<REALTYPE>::calcEdgeLogLikelihoods(const int parIndex,
													 const int childIndex,
													 const int probIndex,
                                                     const int categoryWeightsIndex,
                                                     const int stateFrequenciesIndex,
													 const int scalingFactorsIndex,
                                                     double* outSumLogLikelihood) {

	assert(parIndex >= kTipCount);

    int returnCode = BEAGLE_SUCCESS;

	const REALTYPE* partialsParent = gPartials[parIndex];
	const REALTYPE* transMatrix = gTransitionMatrices[probIndex];
    const REALTYPE* wt = gCategoryWeights[categoryWeightsIndex];
    const REALTYPE* freqs = gStateFrequencies[stateFrequenciesIndex];

	memset(integrationTmp, 0, (kPatternCount * kStateCount)*sizeof(REALTYPE));

	if (childIndex < kTipCount && gTipStates[childIndex]) { // Integrate against a state at the child

		const int* statesChild = gTipStates[childIndex];
		int v = 0; // Index for parent partials

		for(int l = 0; l < kCategoryCount; l++) {
			int u = 0; // Index in resulting product-partials (summed over categories)
			const REALTYPE weight = wt[l];
			for(int k = 0; k < kPatternCount; k++) {

				const int stateChild = statesChild[k];  // DISCUSSION PT: Does it make sense to change the order of the partials,
				// so we can interchange the patterCount and categoryCount loop order?
				int w =  l * kMatrixSize;
				for(int i = 0; i < kStateCount; i++) {
					integrationTmp[u] += transMatrix[w + stateChild] * partialsParent[v + i] * weight;
					u++;
#ifdef PAD_MATRICES
					w += (kStateCount + PAD);
#else
					w += kStateCount;
#endif
				}
				v += kStateCount;
			}
		}

	} else { // Integrate against a partial at the child

		const REALTYPE* partialsChild = gPartials[childIndex];
		int v = 0;

		for(int l = 0; l < kCategoryCount; l++) {
			int u = 0;
			const REALTYPE weight = wt[l];
			for(int k = 0; k < kPatternCount; k++) {
				int w = l * kMatrixSize;
				for(int i = 0; i < kStateCount; i++) {
					double sumOverJ = 0.0;
					for(int j = 0; j < kStateCount; j++) {
						sumOverJ += transMatrix[w] * partialsChild[v + j];
						w++;
					}
#ifdef PAD_MATRICES
					// increment for the extra column at the end
					w += PAD;
#endif
					integrationTmp[u] += sumOverJ * partialsParent[v + i] * weight;
					u++;
				}
				v += kStateCount;
			}
		}
	}

	int u = 0;
	for(int k = 0; k < kPatternCount; k++) {
		REALTYPE sumOverI = 0.0;
		for(int i = 0; i < kStateCount; i++) {
			sumOverI += freqs[i] * integrationTmp[u];
			u++;
		}

        if (!(sumOverI >= realtypeMin))
            returnCode = BEAGLE_ERROR_FLOATING_POINT;

        outLogLikelihoodsTmp[k] = log(sumOverI);
	}


	if (scalingFactorsIndex != BEAGLE_OP_NONE) {
		const REALTYPE* scalingFactors = gScaleBuffers[scalingFactorsIndex];
		for(int k=0; k < kPatternCount; k++)
			outLogLikelihoodsTmp[k] += scalingFactors[k];
	}

    *outSumLogLikelihood = 0.0;
    for (int i = 0; i < kPatternCount; i++) {
        *outSumLogLikelihood += outLogLikelihoodsTmp[i] * gPatternWeights[i];
    }

    return returnCode;
}

template <typename REALTYPE>
int BeagleCPUImpl<REALTYPE>::calcEdgeLogLikelihoodsFirstDeriv(const int parIndex,
                                                               const int childIndex,
                                                               const int probIndex,
                                                               const int firstDerivativeIndex,
                                                               const int categoryWeightsIndex,
                                                               const int stateFrequenciesIndex,
                                                               const int scalingFactorsIndex,
                                                               double* outSumLogLikelihood,
                                                               double* outSumFirstDerivative) {

	assert(parIndex >= kTipCount);

    int returnCode = BEAGLE_SUCCESS;

	const REALTYPE* partialsParent = gPartials[parIndex];
	const REALTYPE* transMatrix = gTransitionMatrices[probIndex];
	const REALTYPE* firstDerivMatrix = gTransitionMatrices[firstDerivativeIndex];
    const REALTYPE* wt = gCategoryWeights[categoryWeightsIndex];
    const REALTYPE* freqs = gStateFrequencies[stateFrequenciesIndex];


	memset(integrationTmp, 0, (kPatternCount * kStateCount)*sizeof(REALTYPE));
	memset(firstDerivTmp, 0, (kPatternCount * kStateCount)*sizeof(REALTYPE));

	if (childIndex < kTipCount && gTipStates[childIndex]) { // Integrate against a state at the child

		const int* statesChild = gTipStates[childIndex];
		int v = 0; // Index for parent partials

		for(int l = 0; l < kCategoryCount; l++) {
			int u = 0; // Index in resulting product-partials (summed over categories)
			const REALTYPE weight = wt[l];
			for(int k = 0; k < kPatternCount; k++) {

				const int stateChild = statesChild[k];  // DISCUSSION PT: Does it make sense to change the order of the partials,
				// so we can interchange the patterCount and categoryCount loop order?
				int w =  l * kMatrixSize;
				for(int i = 0; i < kStateCount; i++) {
					integrationTmp[u] += transMatrix[w + stateChild] * partialsParent[v + i] * weight;
					firstDerivTmp[u] += firstDerivMatrix[w + stateChild] * partialsParent[v + i] * weight;
					u++;
#ifdef PAD_MATRICES
					w += (kStateCount + PAD);
#else
					w += kStateCount;
#endif
				}
				v += kStateCount;
			}
		}

	} else { // Integrate against a partial at the child

		const REALTYPE* partialsChild = gPartials[childIndex];
		int v = 0;

		for(int l = 0; l < kCategoryCount; l++) {
			int u = 0;
			const REALTYPE weight = wt[l];
			for(int k = 0; k < kPatternCount; k++) {
				int w = l * kMatrixSize;
				for(int i = 0; i < kStateCount; i++) {
					double sumOverJ = 0.0;
					double sumOverJD1 = 0.0;
					for(int j = 0; j < kStateCount; j++) {
						sumOverJ += transMatrix[w] * partialsChild[v + j];
						sumOverJD1 += firstDerivMatrix[w] * partialsChild[v + j];
						w++;
					}
#ifdef PAD_MATRICES
					// increment for the extra column at the end
					w += PAD;
#endif
					integrationTmp[u] += sumOverJ * partialsParent[v + i] * weight;
					firstDerivTmp[u] += sumOverJD1 * partialsParent[v + i] * weight;
					u++;
				}
				v += kStateCount;
			}
		}
	}

	int u = 0;
	for(int k = 0; k < kPatternCount; k++) {
		REALTYPE sumOverI = 0.0;
		REALTYPE sumOverID1 = 0.0;
		for(int i = 0; i < kStateCount; i++) {
			sumOverI += freqs[i] * integrationTmp[u];
			sumOverID1 += freqs[i] * firstDerivTmp[u];
			u++;
		}

        if (!(sumOverI >= realtypeMin))
            returnCode = BEAGLE_ERROR_FLOATING_POINT;

        outLogLikelihoodsTmp[k] = log(sumOverI);
		outFirstDerivativesTmp[k] = sumOverID1 / sumOverI;
	}


	if (scalingFactorsIndex != BEAGLE_OP_NONE) {
		const REALTYPE* scalingFactors = gScaleBuffers[scalingFactorsIndex];
		for(int k=0; k < kPatternCount; k++)
			outLogLikelihoodsTmp[k] += scalingFactors[k];
	}

    *outSumLogLikelihood = 0.0;
    *outSumFirstDerivative = 0.0;
    for (int i = 0; i < kPatternCount; i++) {
        *outSumLogLikelihood += outLogLikelihoodsTmp[i] * gPatternWeights[i];

        *outSumFirstDerivative += outFirstDerivativesTmp[i] * gPatternWeights[i];
    }

    return returnCode;
}

template <typename REALTYPE>
int BeagleCPUImpl<REALTYPE>::calcEdgeLogLikelihoodsSecondDeriv(const int parIndex,
                                                                const int childIndex,
                                                                const int probIndex,
                                                                const int firstDerivativeIndex,
                                                                const int secondDerivativeIndex,
                                                                const int categoryWeightsIndex,
                                                                const int stateFrequenciesIndex,
                                                                const int scalingFactorsIndex,
                                                                double* outSumLogLikelihood,
                                                                double* outSumFirstDerivative,
                                                                double* outSumSecondDerivative) {

	assert(parIndex >= kTipCount);

    int returnCode = BEAGLE_SUCCESS;

	const REALTYPE* partialsParent = gPartials[parIndex];
	const REALTYPE* transMatrix = gTransitionMatrices[probIndex];
	const REALTYPE* firstDerivMatrix = gTransitionMatrices[firstDerivativeIndex];
	const REALTYPE* secondDerivMatrix = gTransitionMatrices[secondDerivativeIndex];
    const REALTYPE* wt = gCategoryWeights[categoryWeightsIndex];
    const REALTYPE* freqs = gStateFrequencies[stateFrequenciesIndex];


	memset(integrationTmp, 0, (kPatternCount * kStateCount)*sizeof(REALTYPE));
	memset(firstDerivTmp, 0, (kPatternCount * kStateCount)*sizeof(REALTYPE));
	memset(secondDerivTmp, 0, (kPatternCount * kStateCount)*sizeof(REALTYPE));

	if (childIndex < kTipCount && gTipStates[childIndex]) { // Integrate against a state at the child

		const int* statesChild = gTipStates[childIndex];
		int v = 0; // Index for parent partials

		for(int l = 0; l < kCategoryCount; l++) {
			int u = 0; // Index in resulting product-partials (summed over categories)
			const REALTYPE weight = wt[l];
			for(int k = 0; k < kPatternCount; k++) {

				const int stateChild = statesChild[k];  // DISCUSSION PT: Does it make sense to change the order of the partials,
				// so we can interchange the patterCount and categoryCount loop order?
				int w =  l * kMatrixSize;
				for(int i = 0; i < kStateCount; i++) {
					integrationTmp[u] += transMatrix[w + stateChild] * partialsParent[v + i] * weight;
					firstDerivTmp[u] += firstDerivMatrix[w + stateChild] * partialsParent[v + i] * weight;
					secondDerivTmp[u] += secondDerivMatrix[w + stateChild] * partialsParent[v + i] * weight;
					u++;
#ifdef PAD_MATRICES
					w += (kStateCount + PAD);
#else
					w += kStateCount;
#endif
				}
				v += kStateCount;
			}
		}

	} else { // Integrate against a partial at the child

		const REALTYPE* partialsChild = gPartials[childIndex];
		int v = 0;

		for(int l = 0; l < kCategoryCount; l++) {
			int u = 0;
			const REALTYPE weight = wt[l];
			for(int k = 0; k < kPatternCount; k++) {
				int w = l * kMatrixSize;
				for(int i = 0; i < kStateCount; i++) {
					double sumOverJ = 0.0;
					double sumOverJD1 = 0.0;
					double sumOverJD2 = 0.0;
					for(int j = 0; j < kStateCount; j++) {
						sumOverJ += transMatrix[w] * partialsChild[v + j];
						sumOverJD1 += firstDerivMatrix[w] * partialsChild[v + j];
						sumOverJD2 += secondDerivMatrix[w] * partialsChild[v + j];
						w++;
					}
#ifdef PAD_MATRICES
					// increment for the extra column at the end
					w += PAD;
#endif
					integrationTmp[u] += sumOverJ * partialsParent[v + i] * weight;
					firstDerivTmp[u] += sumOverJD1 * partialsParent[v + i] * weight;
					secondDerivTmp[u] += sumOverJD2 * partialsParent[v + i] * weight;
					u++;
				}
				v += kStateCount;
			}
		}
	}

	int u = 0;
	for(int k = 0; k < kPatternCount; k++) {
		REALTYPE sumOverI = 0.0;
		REALTYPE sumOverID1 = 0.0;
		REALTYPE sumOverID2 = 0.0;
		for(int i = 0; i < kStateCount; i++) {
			sumOverI += freqs[i] * integrationTmp[u];
			sumOverID1 += freqs[i] * firstDerivTmp[u];
			sumOverID2 += freqs[i] * secondDerivTmp[u];
			u++;
		}

        if (!(sumOverI >= realtypeMin))
            returnCode = BEAGLE_ERROR_FLOATING_POINT;

        outLogLikelihoodsTmp[k] = log(sumOverI);
		outFirstDerivativesTmp[k] = sumOverID1 / sumOverI;
		outSecondDerivativesTmp[k] = sumOverID2 / sumOverI - outFirstDerivativesTmp[k] * outFirstDerivativesTmp[k];
	}


	if (scalingFactorsIndex != BEAGLE_OP_NONE) {
		const REALTYPE* scalingFactors = gScaleBuffers[scalingFactorsIndex];
		for(int k=0; k < kPatternCount; k++)
			outLogLikelihoodsTmp[k] += scalingFactors[k];
	}

    *outSumLogLikelihood = 0.0;
    *outSumFirstDerivative = 0.0;
    *outSumSecondDerivative = 0.0;
    for (int i = 0; i < kPatternCount; i++) {
        *outSumLogLikelihood += outLogLikelihoodsTmp[i] * gPatternWeights[i];

        *outSumFirstDerivative += outFirstDerivativesTmp[i] * gPatternWeights[i];

        *outSumSecondDerivative += outSecondDerivativesTmp[i] * gPatternWeights[i];
    }

    return returnCode;
}



template <typename REALTYPE>
int BeagleCPUImpl<REALTYPE>::block(void) {
	// Do nothing.
	return BEAGLE_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
// private methods

/*
 * Re-scales the partial likelihoods such that the largest is one.
 */
template <typename REALTYPE>
void BeagleCPUImpl<REALTYPE>::rescalePartials(REALTYPE* destP,
		REALTYPE* scaleFactors,
		REALTYPE* cumulativeScaleFactors,
                                       const int  fillWithOnes) {
    if (DEBUGGING_OUTPUT) {
        std::cerr << "destP (before rescale): \n";// << destP << "\n";
        for(int i=0; i<kPartialsSize; i++)
            fprintf(stderr,"destP[%d] = %.5f\n",i,destP[i]);
    }

    if (kStateCount == 4 && fillWithOnes != 0) {
//        if (ones == NULL) {
//            ones = (double*) malloc(sizeof(double) * kPatternCount);
//                for(int i = 0; i < kPatternCount; i++)
//                    ones[i] = 1.0;
//        }
        if (kFlags & BEAGLE_FLAG_SCALERS_LOG)
            memcpy(scaleFactors,zeros,sizeof(REALTYPE) * kPaddedPatternCount);
        else
            memcpy(scaleFactors,ones,sizeof(REALTYPE) * kPaddedPatternCount);
        // No accumulation necessary as cumulativeScaleFactors are on the log-scale
        if (DEBUGGING_OUTPUT)
            fprintf(stderr,"Ones copied!\n");
        return;
    }

    // TODO None of the code below has been optimized.
    for (int k = 0; k < kPatternCount; k++) {
    	REALTYPE max = 0;
        const int patternOffset = k * kStateCount;
        for (int l = 0; l < kCategoryCount; l++) {
            int offset = l * kPaddedPatternCount * kStateCount + patternOffset;
            for (int i = 0; i < kStateCount; i++) {
                if(destP[offset] > max)
                    max = destP[offset];
                offset++;
            }
        }
        
        if (max == 0)
            max = 1.0;
        
        for (int l = 0; l < kCategoryCount; l++) {
            int offset = l * kPaddedPatternCount * kStateCount + patternOffset;
            for (int i = 0; i < kStateCount; i++)
                destP[offset++] /= max;
        }

        if (kFlags & BEAGLE_FLAG_SCALERS_LOG) {
            REALTYPE logMax = log(max);
            scaleFactors[k] = logMax;
            if( cumulativeScaleFactors != NULL )
                cumulativeScaleFactors[k] += logMax;
        } else {
            scaleFactors[k] = max;
            if( cumulativeScaleFactors != NULL )
                cumulativeScaleFactors[k] += log(max);
        }
    }
    if (DEBUGGING_OUTPUT) {
        for(int i=0; i<kPatternCount; i++)
            fprintf(stderr,"new scaleFactor[%d] = %.5f\n",i,scaleFactors[i]);
    }
}
    
template <typename REALTYPE>
void BeagleCPUImpl<REALTYPE>::autoRescalePartials(REALTYPE* destP,
                                              signed short* scaleFactors) {
    
    
    for (int k = 0; k < kPatternCount; k++) {
        REALTYPE max = 0;
        const int patternOffset = k * kStateCount;
        for (int l = 0; l < kCategoryCount; l++) {
            int offset = l * kPaddedPatternCount * kStateCount + patternOffset;
            for (int i = 0; i < kStateCount; i++) {
                if(destP[offset] > max)
                    max = destP[offset];
                offset++;
            }
        }
        
        int expMax;
        frexp(max, &expMax);
        scaleFactors[k] = expMax;
        
        if (expMax != 0) {
            for (int l = 0; l < kCategoryCount; l++) {
                int offset = l * kPaddedPatternCount * kStateCount + patternOffset;
                for (int i = 0; i < kStateCount; i++)
                    destP[offset++] *= pow(2.0, -expMax);
            }
        }
    }
}

/*
 * Calculates partial likelihoods at a node when both children have states.
 */
template <typename REALTYPE>
void BeagleCPUImpl<REALTYPE>::calcStatesStates(REALTYPE* destP,
                                     const int* states1,
                                     const REALTYPE* matrices1,
                                     const int* states2,
                                     const REALTYPE* matrices2) {

#pragma omp parallel for num_threads(kCategoryCount)
    for (int l = 0; l < kCategoryCount; l++) {
        int v = l*kStateCount*kPatternCount;
        for (int k = 0; k < kPatternCount; k++) {
            const int state1 = states1[k];
            const int state2 = states2[k];
            if (DEBUGGING_OUTPUT) {
                std::cerr << "calcStatesStates s1 = " << state1 << '\n';
                std::cerr << "calcStatesStates s2 = " << state2 << '\n';
            }
            int w = l * kMatrixSize;
            for (int i = 0; i < kStateCount; i++) {
                destP[v] = matrices1[w + state1] * matrices2[w + state2];
                v++;
#ifdef PAD_MATRICES
                w += (kStateCount + PAD);
#else
                w += kStateCount;
#endif
            }
        }
    }
}

template <typename REALTYPE>
void BeagleCPUImpl<REALTYPE>::calcStatesStatesFixedScaling(REALTYPE* destP,
                                              const int* child1States,
                                           const REALTYPE* child1TransMat,
                                              const int* child2States,
                                           const REALTYPE* child2TransMat,
                                           const REALTYPE* scaleFactors) {
#pragma omp parallel for num_threads(kCategoryCount)
    for (int l = 0; l < kCategoryCount; l++) {
	int v = l*kStateCount*kPatternCount;
        for (int k = 0; k < kPatternCount; k++) {
            const int state1 = child1States[k];
            const int state2 = child2States[k];
            int w = l * kMatrixSize;
            REALTYPE scaleFactor = scaleFactors[k];
            for (int i = 0; i < kStateCount; i++) {
                destP[v] = child1TransMat[w + state1] *
                           child2TransMat[w + state2] / scaleFactor;
                v++;
#ifdef PAD_MATRICES
                w += (kStateCount + PAD);
#else
                w += kStateCount;
#endif
            }
        }
    }
}

/*
 * Calculates partial likelihoods at a node when one child has states and one has partials.
 */
template <typename REALTYPE>
void BeagleCPUImpl<REALTYPE>::calcStatesPartials(REALTYPE* destP,
                                       const int* states1,
                                       const REALTYPE* matrices1,
                                       const REALTYPE* partials2,
                                       const REALTYPE* matrices2) {
#pragma omp parallel for num_threads(kCategoryCount)
    for (int l = 0; l < kCategoryCount; l++) {
	int u = l*kStateCount*kPatternCount;
	int v = l*kStateCount*kPatternCount;
        for (int k = 0; k < kPatternCount; k++) {
            int state1 = states1[k];
            int w = l * kMatrixSize;
            for (int i = 0; i < kStateCount; i++) {
            	REALTYPE tmp = matrices1[w + state1];
            	REALTYPE sum = 0.0;
                for (int j = 0; j < kStateCount; j++) {
                    sum += matrices2[w] * partials2[v + j];
                    w++;
                }
#ifdef PAD_MATRICES
                // increment for the extra column at the end
                w += PAD;
#endif
                destP[u] = tmp * sum;
                u++;
            }
            v += kStateCount;
        }
    }
}

template <typename REALTYPE>
void BeagleCPUImpl<REALTYPE>::calcStatesPartialsFixedScaling(REALTYPE* destP,
                                                const int* states1,
                                             const REALTYPE* matrices1,
                                             const REALTYPE* partials2,
                                             const REALTYPE* matrices2,
                                             const REALTYPE* scaleFactors) {
#pragma omp parallel for num_threads(kCategoryCount)
    for (int l = 0; l < kCategoryCount; l++) {
	int u = l*kStateCount*kPatternCount;
	int v = l*kStateCount*kPatternCount;
        for (int k = 0; k < kPatternCount; k++) {
            int state1 = states1[k];
            int w = l * kMatrixSize;
            REALTYPE scaleFactor = scaleFactors[k];
            for (int i = 0; i < kStateCount; i++) {
            	REALTYPE tmp = matrices1[w + state1];
            	REALTYPE sum = 0.0;
                for (int j = 0; j < kStateCount; j++) {
                    sum += matrices2[w] * partials2[v + j];
                    w++;
                }
#ifdef PAD_MATRICES
                // increment for the extra column at the end
                w += PAD;
#endif
                destP[u] = tmp * sum / scaleFactor;
                u++;
            }
            v += kStateCount;
        }
    }
}

/*
 * Calculates partial likelihoods at a node when both children have partials.
 */
template <typename REALTYPE>
void BeagleCPUImpl<REALTYPE>::calcPartialsPartials(REALTYPE* destP,
                                         const REALTYPE* partials1,
                                         const REALTYPE* matrices1,
                                         const REALTYPE* partials2,
                                         const REALTYPE* matrices2) {
#pragma omp parallel for num_threads(kCategoryCount)
    for (int l = 0; l < kCategoryCount; l++) {
	int u = l*kStateCount*kPatternCount;
	int v = l*kStateCount*kPatternCount;
        for (int k = 0; k < kPatternCount; k++) {
            int w = l * kMatrixSize;
            for (int i = 0; i < kStateCount; i++) {
                REALTYPE sum1 = 0.0, sum2 = 0.0;
                for (int j = 0; j < kStateCount; j++) {
                    sum1 += matrices1[w] * partials1[v + j];
                    sum2 += matrices2[w] * partials2[v + j];
                    if (DEBUGGING_OUTPUT) {
                        if (k == 0)
                            printf("mat1[%d] = %.5f\n", w, matrices1[w]);
                        if (k == 1)
                            printf("mat2[%d] = %.5f\n", w, matrices2[w]);
                    }
                    w++;
                }
#ifdef PAD_MATRICES
                // increment for the extra column at the end
                w += PAD;
#endif
                destP[u] = sum1 * sum2;
                u++;
            }
            v += kStateCount;
        }
    }
}

template <typename REALTYPE>
void BeagleCPUImpl<REALTYPE>::calcPartialsPartialsFixedScaling(REALTYPE* destP,
                                               const REALTYPE* partials1,
                                               const REALTYPE* matrices1,
                                               const REALTYPE* partials2,
                                               const REALTYPE* matrices2,
                                               const REALTYPE* scaleFactors) {
#pragma omp parallel for num_threads(kCategoryCount)
    for (int l = 0; l < kCategoryCount; l++) {
	int u = l*kStateCount*kPatternCount;
	int v = l*kStateCount*kPatternCount;
        for (int k = 0; k < kPatternCount; k++) {
            int w = l * kMatrixSize;
            REALTYPE scaleFactor = scaleFactors[k];
            for (int i = 0; i < kStateCount; i++) {
                REALTYPE sum1 = 0.0, sum2 = 0.0;
                for (int j = 0; j < kStateCount; j++) {
                    sum1 += matrices1[w] * partials1[v + j];
                    sum2 += matrices2[w] * partials2[v + j];
                    w++;
                }
#ifdef PAD_MATRICES
                // increment for the extra column at the end
                w += PAD;
#endif
                destP[u] = sum1 * sum2 / scaleFactor;
                u++;
            }
            v += kStateCount;
        }
    }
}
    
template <typename REALTYPE>
void BeagleCPUImpl<REALTYPE>::calcPartialsPartialsAutoScaling(REALTYPE* destP,
                                                               const REALTYPE* partials1,
                                                               const REALTYPE* matrices1,
                                                               const REALTYPE* partials2,
                                                               const REALTYPE* matrices2,
                                                               int* activateScaling) {
    
#pragma omp parallel for num_threads(kCategoryCount)
    for (int l = 0; l < kCategoryCount; l++) {
        int u = l*kStateCount*kPatternCount;
        int v = l*kStateCount*kPatternCount;
        for (int k = 0; k < kPatternCount; k++) {
            int w = l * kMatrixSize;
            for (int i = 0; i < kStateCount; i++) {
                REALTYPE sum1 = 0.0, sum2 = 0.0;
                for (int j = 0; j < kStateCount; j++) {
                    sum1 += matrices1[w] * partials1[v + j];
                    sum2 += matrices2[w] * partials2[v + j];
                    w++;
                }
#ifdef PAD_MATRICES
                // increment for the extra column at the end
                w += PAD;
#endif
                destP[u] = sum1 * sum2;

                if (*activateScaling == 0) {
                    int expTmp;
                    frexp(destP[u], &expTmp);
                    if (abs(expTmp) > scalingExponentThreshhold) 
                        *activateScaling = 1;
                }
                
                u++;
            }
            v += kStateCount;
        }
    }
}

template <typename REALTYPE>
int BeagleCPUImpl<REALTYPE>::getPaddedPatternsModulus() {
	// Padding only necessary for SSE implementations that vectorize across patterns
	return 1;  // No padding
}

template<typename REALTYPE>
void* BeagleCPUImpl<REALTYPE>::mallocAligned(size_t size) {
	void *ptr = (void *) NULL;
#if (PAD == 1)	
	const size_t align = 32;
#else // PAD == 2
	const size_t align = 16;
#endif
	int res;

#if defined (__APPLE__)
	/*
	 presumably malloc on OS X always returns
	 a 16-byte aligned pointer
	 */
	ptr = malloc(size);
	if(ptr == (void*)NULL) {
		assert(0);
	}
#else
	res = posix_memalign(&ptr, align, size);
	if (res != 0) {
		assert(0);
	}
#endif

	return ptr;
}

///////////////////////////////////////////////////////////////////////////////
// BeagleCPUImplFactory public methods
template <typename REALTYPE>
BeagleImpl* BeagleCPUImplFactory<REALTYPE>::createImpl(int tipCount,
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

    BeagleImpl* impl = new BeagleCPUImpl<REALTYPE>();

    try {
        *errorCode =
            impl->createInstance(tipCount, partialsBufferCount, compactBufferCount, stateCount,
                                 patternCount, eigenBufferCount, matrixBufferCount,
                                 categoryCount,scaleBufferCount, resourceNumber, preferenceFlags, requirementFlags);
        if (*errorCode == BEAGLE_SUCCESS) {
            return impl;
        }
        delete impl;
        return NULL;
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
const char* BeagleCPUImplFactory<REALTYPE>::getName() {
	return getBeagleCPUName<REALTYPE>();
}

template <typename REALTYPE>
const long BeagleCPUImplFactory<REALTYPE>::getFlags() {
    long flags = BEAGLE_FLAG_COMPUTATION_SYNCH |
                 BEAGLE_FLAG_SCALING_MANUAL | BEAGLE_FLAG_SCALING_ALWAYS | BEAGLE_FLAG_SCALING_AUTO |
                 BEAGLE_FLAG_THREADING_NONE |
                 BEAGLE_FLAG_PROCESSOR_CPU |
                 BEAGLE_FLAG_VECTOR_NONE |
                 BEAGLE_FLAG_SCALERS_LOG | BEAGLE_FLAG_SCALERS_RAW |
                 BEAGLE_FLAG_EIGEN_COMPLEX | BEAGLE_FLAG_EIGEN_REAL;
	if (DOUBLE_PRECISION)
		flags |= BEAGLE_FLAG_PRECISION_DOUBLE;
	else
		flags |= BEAGLE_FLAG_PRECISION_SINGLE;
    return flags;
}

}	// namespace cpu
}	// namespace beagle

#endif // BEAGLE_CPU_IMPL_GENERAL_HPP

