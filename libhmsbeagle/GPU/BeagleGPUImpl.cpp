
/*
 *  BeagleGPUImpl.cpp
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
 * @author Marc Suchard
 * @author Andrew Rambaut
 * @author Daniel Ayres
 * @author Aaron Darling
 */
#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <cstring>

#include "libhmsbeagle/beagle.h"
#include "libhmsbeagle/GPU/GPUImplDefs.h"
#include "libhmsbeagle/GPU/BeagleGPUImpl.h"
#include "libhmsbeagle/GPU/GPUImplHelper.h"
#include "libhmsbeagle/GPU/KernelLauncher.h"
#include "libhmsbeagle/GPU/GPUInterface.h"

using namespace beagle;
using namespace beagle::gpu;

BeagleGPUImpl::~BeagleGPUImpl() {
		
	// free GPU memory
	
	if (kInitialized) {
		for (int i=0; i < kScaleBufferCount; i++)
			gpu->FreeMemory(dScalingFactors[i]);
		for (int i = 0; i < kBufferCount; i++) {        
			if (i < kTipCount) { // For the tips
				if (i < kCompactBufferCount)
					gpu->FreeMemory(dCompactBuffers[i]);
				if (i < kTipPartialsBufferCount)
					gpu->FreeMemory(dTipPartialsBuffers[i]);
			} else {
				gpu->FreeMemory(dPartials[i]);        
			}
		}
	}
	
	if (kHostMemoryUsed) {
		// free memory
		delete kernels;
		delete gpu;
    
		free(dPartials);
		free(dTipPartialsBuffers);
		free(dStates);
		free(dCompactBuffers);
		free(dMatrices);
		free(dEigenValues);
		free(dEvec);
		free(dIevc);
    
		free(dScalingFactors);
    
		free(hDistanceQueue);
		free(hPtrQueue);
    
		free(hWeightsCache);
		free(hFrequenciesCache);
		free(hPartialsCache);
		free(hStatesCache);
		free(hMatrixCache);
		free(hCategoryRates);
		
#ifndef DOUBLE_PRECISION
    free(hCategoryCache);
    free(hLogLikelihoodsCache);
#endif
    
	}
}

int BeagleGPUImpl::createInstance(int tipCount,
                                  int partialsBufferCount,
                                  int compactBufferCount,
                                  int stateCount,
                                  int patternCount,
                                  int eigenDecompositionCount,
                                  int matrixCount,
                                  int categoryCount,
                                  int scaleBufferCount,
                                  long preferenceFlags,
                                  long requirementFlags) {
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::createInstance\n");
#endif
    
    kHostMemoryUsed = 0;
    kInitialized = 0;
    
    kTipCount = tipCount;
    kPartialsBufferCount = partialsBufferCount;
    kCompactBufferCount = compactBufferCount;
    kStateCount = stateCount;
    kPatternCount = patternCount;
    kEigenDecompCount = eigenDecompositionCount;
    kMatrixCount = matrixCount;
    kCategoryCount = categoryCount;
    kScaleBufferCount = scaleBufferCount;
    
    kTipPartialsBufferCount = kTipCount - kCompactBufferCount;
    kBufferCount = kPartialsBufferCount + kCompactBufferCount;
    
    if (kStateCount <= 4)
        kPaddedStateCount = 4;
    else if (kStateCount <= 16)
        kPaddedStateCount = 16;
    else if (kStateCount <= 32)
        kPaddedStateCount = 32;
    else if (kStateCount <= 48)
    	kPaddedStateCount = 48;    
    else if (kStateCount <= 64)
        kPaddedStateCount = 64;
    else if (kStateCount <= 128)
        kPaddedStateCount = 128;
    else if (kStateCount <= 192)
        kPaddedStateCount = 192;
    else
        kPaddedStateCount = kStateCount + kStateCount % 16;
    
    // Abort for mismatched stateCount; remove when run-time stateCounts are complete
    if (kPaddedStateCount == 16 || kPaddedStateCount == 32) {
    	fprintf(stderr,"\tMismatch in model size in GPU implementation!\n\t\tkPaddedStateCount = %d (not yet implemented)\n",
    			kPaddedStateCount);    	
        return BEAGLE_ERROR_GENERAL;
    }
    
    // Make sure that kPaddedPatternCount + paddedPatterns is multiple of 4 for DNA model
    int paddedPatterns = 0;
    if (kPaddedStateCount == 4 && kPatternCount % 4 != 0)
        paddedPatterns = 4 - kPatternCount % 4;
    else
        paddedPatterns = 0;
    
    kPaddedPatternCount = kPatternCount + paddedPatterns;
        
    kPartialsSize = kPaddedPatternCount * kPaddedStateCount * kCategoryCount;
    kMatrixSize = kPaddedStateCount * kPaddedStateCount;
    kEigenValuesSize = kPaddedStateCount; // TODO: How are we going to handle complex eigenvalues?
    
    // TODO: only allocate if necessary on the fly
    hWeightsCache = (REAL*) calloc(kBufferCount, SIZE_REAL);
    hFrequenciesCache = (REAL*) calloc(kPaddedStateCount, SIZE_REAL);
    hPartialsCache = (REAL*) calloc(kPartialsSize, SIZE_REAL);
    hStatesCache = (int*) calloc(kPaddedPatternCount, SIZE_INT);
    hMatrixCache = (REAL*) calloc(2 * kMatrixSize + kEigenValuesSize, SIZE_REAL);
#ifndef DOUBLE_PRECISION
	hCategoryCache = (REAL*) malloc(kCategoryCount * SIZE_REAL);
    hLogLikelihoodsCache = (REAL*) malloc(kPatternCount * SIZE_REAL);
#endif
    
    kHostMemoryUsed = 1;
        
    kLastCompactBufferIndex = -1;
    kLastTipPartialsBufferIndex = -1;
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving BeagleGPUImpl::createInstance\n");
#endif
    
    return BEAGLE_SUCCESS;
}

int BeagleGPUImpl::initializeInstance(BeagleInstanceDetails* returnInfo) {
    
    // TODO: compute device memory requirements
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::initializeInstance\n");
#endif
    
    gpu = new GPUInterface();
    
    gpu->Initialize();
    
    int numDevices = 0;
    numDevices = gpu->GetDeviceCount();
    if (numDevices == 0) {
        fprintf(stderr, "Error: No GPU devices\n");
        return BEAGLE_ERROR_GENERAL;
    }
    if (resourceNumber > numDevices) {
        fprintf(stderr,"Error: Trying to initialize device # %d (which does not exist)\n",resourceNumber);
        return BEAGLE_ERROR_GENERAL;
    }
    
    // TODO: recompiling kernels for every instance, probably not ideal
    gpu->SetDevice(resourceNumber-1,kPaddedStateCount,kCategoryCount,kPaddedPatternCount);
    
    kernels = new KernelLauncher(gpu);
    
    dEvec = (GPUPtr*) calloc(sizeof(GPUPtr*),kEigenDecompCount);
    dIevc = (GPUPtr*) calloc(sizeof(GPUPtr*),kEigenDecompCount);
    dEigenValues = (GPUPtr*) calloc(sizeof(GPUPtr*),kEigenDecompCount);
                
    for(int i=0; i<kEigenDecompCount; i++) {
    	dEvec[i] = gpu->AllocateRealMemory(kMatrixSize);
    	dIevc[i] = gpu->AllocateRealMemory(kMatrixSize);
    	dEigenValues[i] = gpu->AllocateRealMemory(kEigenValuesSize);
    }
    
    dWeights = gpu->AllocateRealMemory(kBufferCount);
    
    dFrequencies = gpu->AllocateRealMemory(kPaddedStateCount);
    
    dIntegrationTmp = gpu->AllocateRealMemory(kPaddedPatternCount);
    dPartialsTmp = gpu->AllocateRealMemory(kPartialsSize);
    
    // Fill with 0s so 'free' does not choke if unallocated
    dPartials = (GPUPtr*) calloc(sizeof(GPUPtr), kBufferCount);
    
    // Internal nodes have 0s so partials are used
    dStates = (GPUPtr*) calloc(sizeof(GPUPtr), kBufferCount); 
    
    dCompactBuffers = (GPUPtr*) malloc(sizeof(GPUPtr) * kCompactBufferCount); 
    dTipPartialsBuffers = (GPUPtr*) malloc(sizeof(GPUPtr) * kTipPartialsBufferCount);
    
    dScalingFactors = (GPUPtr*) malloc(sizeof(GPUPtr) * kScaleBufferCount);
    for (int i=0; i < kScaleBufferCount; i++)
    	dScalingFactors[i] = gpu->AllocateRealMemory(kPaddedPatternCount);
    
    for (int i = 0; i < kBufferCount; i++) {        
        if (i < kTipCount) { // For the tips
            if (i < kCompactBufferCount)
                dCompactBuffers[i] = gpu->AllocateIntMemory(kPaddedPatternCount);
            if (i < kTipPartialsBufferCount)
                dTipPartialsBuffers[i] = gpu->AllocateRealMemory(kPartialsSize);
        } else {
            dPartials[i] = gpu->AllocateRealMemory(kPartialsSize);
        }
    }
    
    kLastCompactBufferIndex = kCompactBufferCount - 1;
    kLastTipPartialsBufferIndex = kTipPartialsBufferCount - 1;
    
    dMatrices = (GPUPtr*) malloc(sizeof(GPUPtr) * kMatrixCount);
    
    for (int i = 0; i < kMatrixCount; i++) {
        dMatrices[i] = gpu->AllocateRealMemory(kMatrixSize * kCategoryCount);
    }
    
    // No execution has more no kBufferCount events
    dBranchLengths = gpu->AllocateRealMemory(kBufferCount);
    
    dDistanceQueue = gpu->AllocateRealMemory(kMatrixCount * kCategoryCount);
    hDistanceQueue = (REAL*) malloc(SIZE_REAL * kMatrixCount * kCategoryCount);
    checkHostMemory(hDistanceQueue);
    
    int ptrQueueLength = kMatrixCount * kCategoryCount;
    if (kPartialsBufferCount > (kMatrixCount * kCategoryCount))
        ptrQueueLength = kPartialsBufferCount;
    
    dPtrQueue = gpu->AllocateMemory(sizeof(GPUPtr) * ptrQueueLength);
    hPtrQueue = (GPUPtr*) malloc(sizeof(GPUPtr) * ptrQueueLength);
    checkHostMemory(hPtrQueue);
    
	hCategoryRates = (REAL*) malloc(SIZE_REAL * kCategoryCount);
    checkHostMemory(hCategoryRates);
    
    if (returnInfo != NULL) {
        returnInfo->resourceNumber = resourceNumber;
        returnInfo->flags = BEAGLE_FLAG_SINGLE | BEAGLE_FLAG_ASYNCH | BEAGLE_FLAG_GPU;
    }
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::initializeInstance\n");
#endif
    
    kInitialized = 1;
    
    return BEAGLE_SUCCESS;
}

int BeagleGPUImpl::setTipStates(int tipIndex,
                                const int* inStates) {
    // TODO: test setTipStates
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::setTipStates\n");
#endif
    
    if (tipIndex < 0 || tipIndex >= kTipCount)
        return BEAGLE_ERROR_OUT_OF_RANGE;

    for(int i = 0; i < kPatternCount; i++)
        hStatesCache[i] = (inStates[i] < kStateCount ? inStates[i] : kPaddedStateCount);
    
    // Padded extra patterns
    for(int i = kPatternCount; i < kPaddedPatternCount; i++)
        hStatesCache[i] = kPaddedStateCount;

    if (dStates[tipIndex] == 0) {
        assert(kLastCompactBufferIndex >= 0 && kLastCompactBufferIndex < kCompactBufferCount);
        dStates[tipIndex] = dCompactBuffers[kLastCompactBufferIndex--];
    }
    // Copy to GPU device
    gpu->MemcpyHostToDevice(dStates[tipIndex], hStatesCache, SIZE_INT * kPaddedPatternCount);
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::setTipStates\n");
#endif
    
    return BEAGLE_SUCCESS;
}

int BeagleGPUImpl::setTipPartials(int tipIndex,
                                  const double* inPartials) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::setTipPartials\n");
#endif
    
    if (tipIndex < 0 || tipIndex >= kTipCount)
        return BEAGLE_ERROR_OUT_OF_RANGE;

    const double* inPartialsOffset = inPartials;
    REAL* tmpRealPartialsOffset = hPartialsCache;
    for (int i = 0; i < kPatternCount; i++) {
#ifdef DOUBLE_PRECISION
        memcpy(tmpRealPartialsOffset, inPartialsOffset, SIZE_REAL * kStateCount);
#else
        MEMCNV(tmpRealPartialsOffset, inPartialsOffset, kStateCount, REAL);
#endif
        tmpRealPartialsOffset += kPaddedStateCount;
        inPartialsOffset += kStateCount;
    }
    
    int partialsLength = kPaddedPatternCount * kPaddedStateCount;
    for (int i = 1; i < kCategoryCount; i++) {
        memcpy(hPartialsCache + i * partialsLength, hPartialsCache, partialsLength * SIZE_REAL);
    }    
    
    if (tipIndex < kTipCount) {
        if (dPartials[tipIndex] == 0) {
            assert(kLastTipPartialsBufferIndex >= 0 && kLastTipPartialsBufferIndex <
                   kTipPartialsBufferCount);
            dPartials[tipIndex] = dTipPartialsBuffers[kLastTipPartialsBufferIndex--];
        }
    }
    // Copy to GPU device
    gpu->MemcpyHostToDevice(dPartials[tipIndex], hPartialsCache, SIZE_REAL * kPartialsSize);
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::setTipPartials\n");
#endif
    
    return BEAGLE_SUCCESS;
}

int BeagleGPUImpl::setPartials(int bufferIndex,
                               const double* inPartials) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::setPartials\n");
#endif
    
    if (bufferIndex < 0 || bufferIndex >= kPartialsBufferCount)
        return BEAGLE_ERROR_OUT_OF_RANGE;

    const double* inPartialsOffset = inPartials;
    REAL* tmpRealPartialsOffset = hPartialsCache;
    for (int l = 0; l < kCategoryCount; l++) {
        for (int i = 0; i < kPatternCount; i++) {
#ifdef DOUBLE_PRECISION
            memcpy(tmpRealPartialsOffset, inPartialsOffset, SIZE_REAL * kStateCount);
#else
            MEMCNV(tmpRealPartialsOffset, inPartialsOffset, kStateCount, REAL);
#endif
            tmpRealPartialsOffset += kPaddedStateCount;
            inPartialsOffset += kStateCount;
        }
    }
    
    if (bufferIndex < kTipCount) {
        if (dPartials[bufferIndex] == 0) {
            assert(kLastTipPartialsBufferIndex >= 0 && kLastTipPartialsBufferIndex <
                   kTipPartialsBufferCount);
            dPartials[bufferIndex] = dTipPartialsBuffers[kLastTipPartialsBufferIndex--];
        }
    }
    // Copy to GPU device
    gpu->MemcpyHostToDevice(dPartials[bufferIndex], hPartialsCache, SIZE_REAL * kPartialsSize);
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::setPartials\n");
#endif
    
    return BEAGLE_SUCCESS;
}

int BeagleGPUImpl::getPartials(int bufferIndex,
							   int scaleIndex,
                               double* outPartials) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::getPartials\n");
#endif
    
    // TODO: test getPartials
    
    gpu->MemcpyDeviceToHost(hPartialsCache, dPartials[bufferIndex], SIZE_REAL * kPartialsSize);
    
    double* outPartialsOffset = outPartials;
    REAL* tmpRealPartialsOffset = hPartialsCache;
    
    for (int i = 0; i < kPatternCount; i++) {
#ifdef DOUBLE_PRECISION
        memcpy(outPartialsOffset, tmpRealPartialsOffset, SIZE_REAL * kStateCount);
#else
        MEMCNV(outPartialsOffset, tmpRealPartialsOffset, kStateCount, double);
#endif
        tmpRealPartialsOffset += kPaddedStateCount;
        outPartialsOffset += kStateCount;
    }
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::getPartials\n");
#endif
    
    return BEAGLE_SUCCESS;
}


int BeagleGPUImpl::setEigenDecomposition(int eigenIndex,
                                         const double* inEigenVectors,
                                         const double* inInverseEigenVectors,
                                         const double* inEigenValues) {
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\tEntering BeagleGPUImpl::setEigenDecomposition\n");
#endif
    
    // Native memory packing order (length): Ievc (state^2), Evec (state^2),
    //  Eval (state), EvalImag (state)
    
    REAL* Ievc, * tmpIevc, * Evec, * tmpEvec, * Eval;
    
    tmpIevc = Ievc = (REAL*) hMatrixCache;
    tmpEvec = Evec = Ievc + kMatrixSize;
    Eval = Evec + kMatrixSize;
    
    for (int i = 0; i < kStateCount; i++) {
#ifdef DOUBLE_PRECISION
        memcpy(tmpIevc, inInverseEigenVectors + i * kStateCount, SIZE_REAL * kStateCount);
        memcpy(tmpEvec, inEigenVectors + i * kStateCount, SIZE_REAL * kStateCount);
#else
        MEMCNV(tmpIevc, (inInverseEigenVectors + i * kStateCount), kStateCount, REAL);
        MEMCNV(tmpEvec, (inEigenVectors + i * kStateCount), kStateCount, REAL);
#endif
        tmpIevc += kPaddedStateCount;
        tmpEvec += kPaddedStateCount;
    }
    
    // Transposing matrices avoids incoherent memory read/writes  
    // TODO: Only need to tranpose sub-matrix of trueStateCount
    transposeSquareMatrix(Ievc, kPaddedStateCount); 
    transposeSquareMatrix(Evec, kPaddedStateCount);
    
#ifdef DOUBLE_PRECISION
    memcpy(Eval, inEigenValues, SIZE_REAL * kStateCount);
#else
    MEMCNV(Eval, inEigenValues, kStateCount, REAL);
#endif
    
#ifdef BEAGLE_DEBUG_VALUES
#ifdef DOUBLE_PRECISION
    fprintf(stderr, "Eval:\n");
    printfVectorD(Eval, kPaddedStateCount);
    fprintf(stderr, "Evec:\n");
    printfVectorD(Evec, kMatrixSize);
    fprintf(stderr, "Ievc:\n");
    printfVectorD(Ievc, kPaddedStateCount * kPaddedStateCount);
#else
    fprintf(stderr, "Eval =\n");
    printfVectorF(Eval, kPaddedStateCount);
    fprintf(stderr, "Evec =\n");
    printfVectorF(Evec, kMatrixSize);
    fprintf(stderr, "Ievc =\n");
    printfVectorF(Ievc, kPaddedStateCount * kPaddedStateCount);   
#endif
#endif
    
    // Copy to GPU device
    gpu->MemcpyHostToDevice(dIevc[eigenIndex], Ievc, SIZE_REAL * kMatrixSize);
    gpu->MemcpyHostToDevice(dEvec[eigenIndex], Evec, SIZE_REAL * kMatrixSize);
    gpu->MemcpyHostToDevice(dEigenValues[eigenIndex], Eval, SIZE_REAL * kPaddedStateCount);
    
#ifdef BEAGLE_DEBUG_VALUES
    fprintf(stderr, "dEigenValues =\n");
    gpu->PrintfDeviceVector(dEigenValues[eigenIndex], kPaddedStateCount);
    fprintf(stderr, "dEvec =\n");
    gpu->PrintfDeviceVector(dEvec[eigenIndex], kMatrixSize);
    fprintf(stderr, "dIevc =\n");
    gpu->PrintfDeviceVector(dIevc[eigenIndex], kPaddedStateCount * kPaddedStateCount);
#endif
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::setEigenDecomposition\n");
#endif
    
    return BEAGLE_SUCCESS;
}

int BeagleGPUImpl::setCategoryRates(const double* inCategoryRates) {
    // TODO: test setCategoryRates
#ifdef BEAGLE_DEBUG_FLOW
	fprintf(stderr, "\tEntering BeagleGPUImpl::updateCategoryRates\n");
#endif

#ifdef DOUBLE_PRECISION
	const double* categoryRates = inCategoryRates;
#else
	REAL* categoryRates = hCategoryCache;
	MEMCNV(categoryRates, inCategoryRates, kCategoryCount, REAL);
#endif
    
	memcpy(hCategoryRates, categoryRates, SIZE_REAL * kCategoryCount);
    
#ifdef BEAGLE_DEBUG_FLOW
	fprintf(stderr, "\tLeaving  BeagleGPUImpl::updateCategoryRates\n");
#endif
    
    return BEAGLE_SUCCESS;
}

int BeagleGPUImpl::setTransitionMatrix(int matrixIndex,
                                       const double* inMatrix) {
    // TODO: test setTransitionMatrix
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::setTransitionMatrix\n");
#endif
    
    const double* inMatrixOffset = inMatrix;
    REAL* tmpRealMatrixOffset = hMatrixCache;
    
    for (int l = 0; l < kCategoryCount; l++) {
        for (int i = 0; i < kStateCount; i++) {
#ifdef DOUBLE_PRECISION
            memcpy(tmpRealMatrixOffset, inMatrixOffset, SIZE_REAL * kStateCount);
#else
            MEMCNV(tmpRealMatrixOffset, inMatrixOffset, kStateCount, REAL);
#endif
            tmpRealMatrixOffset += kPaddedStateCount;
            inMatrixOffset += kStateCount;
        }
    }
        
    // Copy to GPU device
    gpu->MemcpyHostToDevice(dMatrices[matrixIndex], hMatrixCache,
                            SIZE_REAL * kMatrixSize * kCategoryCount);
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::setTransitionMatrix\n");
#endif
    
    return BEAGLE_SUCCESS;
}

int BeagleGPUImpl::updateTransitionMatrices(int eigenIndex,
                                            const int* probabilityIndices,
                                            const int* firstDerivativeIndices,
                                            const int* secondDervativeIndices,
                                            const double* edgeLengths,
                                            int count) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\tEntering BeagleGPUImpl::updateTransitionMatrices\n");
#endif
    
    // TODO: implement calculation of derivatives
    
    int totalCount = 0;
    
#ifdef CUDA
    for (int i = 0; i < count; i++) {        
		for (int j = 0; j < kCategoryCount; j++) {
            hPtrQueue[totalCount] = dMatrices[probabilityIndices[i]] + (j * kMatrixSize * sizeof(GPUPtr));
            hDistanceQueue[totalCount] = ((REAL) edgeLengths[i]) * hCategoryRates[j];
            totalCount++;
        }
    }
    
    gpu->MemcpyHostToDevice(dPtrQueue, hPtrQueue, sizeof(GPUPtr) * totalCount);
    gpu->MemcpyHostToDevice(dDistanceQueue, hDistanceQueue, SIZE_REAL * totalCount);
    
    // Set-up and call GPU kernel
    kernels->GetTransitionProbabilitiesSquare(dPtrQueue, dEvec[eigenIndex], dIevc[eigenIndex], 
                                              dEigenValues[eigenIndex], dDistanceQueue, totalCount);    
    
#else
    for (int i = 0; i < count; i++) {        
		for (int j = 0; j < kCategoryCount; j++) {
            hDistanceQueue[totalCount] = ((REAL) edgeLengths[i]) * hCategoryRates[j];
            totalCount++;
        }
    }
    
    gpu->MemcpyHostToDevice(dDistanceQueue, hDistanceQueue, SIZE_REAL * totalCount);
    
    // Set-up and call GPU kernel
    for (int i = 0; i < count; i++) {        
        kernels->GetTransitionProbabilitiesSquare(dMatrices[probabilityIndices[i]], dEvec[eigenIndex], dIevc[eigenIndex], 
                                                  dEigenValues[eigenIndex], dDistanceQueue, kCategoryCount,
                                                  i * kCategoryCount);
    }
        
#endif
    
#ifdef BEAGLE_DEBUG_VALUES
    for (int i = 0; i < count; i++) {
        fprintf(stderr, "dMatrices[probabilityIndices[%d]]  (hDQ = %1.5e, eL = %1.5e) =\n", i,hDistanceQueue[i], edgeLengths[i]);        
        gpu->PrintfDeviceVector(dMatrices[probabilityIndices[i]], kMatrixSize * kCategoryCount);
        for(int j=0; j<kCategoryCount; j++)
        	fprintf(stderr, " %1.5f",hCategoryRates[j]);
        fprintf(stderr,"\n");
    }
#endif    

#ifdef BEAGLE_DEBUG_SYNCH    
    gpu->Synchronize();
#endif
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::updateTransitionMatrices\n");
#endif
    
    return BEAGLE_SUCCESS;
}

int BeagleGPUImpl::updatePartials(const int* operations,
                                  int operationCount,
                                  int cumulativeScalingIndex) {
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::updatePartials\n");
#endif
    
    GPUPtr cumulativeScalingBuffer = 0;
    if (cumulativeScalingIndex != BEAGLE_OP_NONE)
        cumulativeScalingBuffer = dScalingFactors[cumulativeScalingIndex];
    
    // Serial version
    for (int op = 0; op < operationCount; op++) {
        const int parIndex = operations[op * 7];
        const int writeScalingIndex = operations[op * 7 + 1];
        const int readScalingIndex = operations[op * 7 + 2];
        const int child1Index = operations[op * 7 + 3];
        const int child1TransMatIndex = operations[op * 7 + 4];
        const int child2Index = operations[op * 7 + 5];
        const int child2TransMatIndex = operations[op * 7 + 6];
        
        GPUPtr matrices1 = dMatrices[child1TransMatIndex];
        GPUPtr matrices2 = dMatrices[child2TransMatIndex];
        
        GPUPtr partials1 = dPartials[child1Index];
        GPUPtr partials2 = dPartials[child2Index];
        
        GPUPtr partials3 = dPartials[parIndex];
        
        GPUPtr tipStates1 = dStates[child1Index];
        GPUPtr tipStates2 = dStates[child2Index];
        
        int rescale = BEAGLE_OP_NONE;
        GPUPtr scalingFactors;
        if (writeScalingIndex >= 0) {
            rescale = 1;
            scalingFactors = dScalingFactors[writeScalingIndex];
        } else if (readScalingIndex >= 0) {
            rescale = 0;
            scalingFactors = dScalingFactors[readScalingIndex];
        }
        
#ifdef BEAGLE_DEBUG_VALUES
        fprintf(stderr, "kPaddedPatternCount = %d\n", kPaddedPatternCount);
        fprintf(stderr, "kPatternCount = %d\n", kPatternCount);
        fprintf(stderr, "categoryCount  = %d\n", kCategoryCount);
        fprintf(stderr, "partialSize = %d\n", kPartialsSize);
        fprintf(stderr, "writeIndex = %d,  readIndex = %d, rescale = %d\n",writeScalingIndex,readScalingIndex,rescale);
        fprintf(stderr, "child1 = \n");
        if (tipStates1)
            gpu->PrintfDeviceInt(tipStates1, kPaddedPatternCount);
        else
            gpu->PrintfDeviceVector(partials1, kPartialsSize);
        fprintf(stderr, "child2 = \n");
        if (tipStates2)
            gpu->PrintfDeviceInt(tipStates2, kPaddedPatternCount);
        else
            gpu->PrintfDeviceVector(partials2, kPartialsSize);
        fprintf(stderr, "node index = %d\n", parIndex);       
#endif        
        
        if (tipStates1 != 0) {
            if (tipStates2 != 0 ) {
                kernels->StatesStatesPruningDynamicScaling(tipStates1, tipStates2, partials3,
                                                           matrices1, matrices2, scalingFactors,
                                                           cumulativeScalingBuffer,
                                                           kPaddedPatternCount, kCategoryCount,
                                                           rescale);
            } else {
                kernels->StatesPartialsPruningDynamicScaling(tipStates1, partials2, partials3,
                                                             matrices1, matrices2, scalingFactors,
                                                             cumulativeScalingBuffer, 
                                                             kPaddedPatternCount, kCategoryCount,
                                                             rescale);
            }
        } else {
            if (tipStates2 != 0) {
                kernels->StatesPartialsPruningDynamicScaling(tipStates2, partials1, partials3,
                                                             matrices2, matrices1, scalingFactors,
                                                             cumulativeScalingBuffer, 
                                                             kPaddedPatternCount, kCategoryCount,
                                                             rescale);
            } else {
                kernels->PartialsPartialsPruningDynamicScaling(partials1, partials2, partials3,
                                                               matrices1, matrices2, scalingFactors,
                                                               cumulativeScalingBuffer, 
                                                               kPaddedPatternCount, kCategoryCount,
                                                               rescale);
            }
        }

        
#ifdef BEAGLE_DEBUG_VALUES            
        if (rescale > -1) {
        	fprintf(stderr,"scalars = ");
        	gpu->PrintfDeviceVector(scalingFactors,kPaddedPatternCount);
        }
        fprintf(stderr, "parent = \n");
        int signal = 0;
        if (writeScalingIndex == -1)
        	gpu->PrintfDeviceVector(partials3, kPartialsSize);
        else
        	gpu->PrintfDeviceVector(partials3, kPartialsSize, 1.0,&signal);
//        if (signal) {
//        	fprintf(stderr,"mat1 = ");
//        	gpu->PrintfDeviceVector(matrices1,kMatrixSize);
//        	fprintf(stderr,"mat2 = ");
//        	gpu->PrintfDeviceVector(matrices2,kMatrixSize);
//        	exit(0);
//        }
        
#endif
    }
    
#ifdef BEAGLE_DEBUG_SYNCH    
    gpu->Synchronize();
#endif
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::updatePartials\n");
#endif
    
    return BEAGLE_SUCCESS;
}

int BeagleGPUImpl::waitForPartials(const int* destinationPartials,
                                   int destinationPartialsCount) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::waitForPartials\n");
#endif
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::waitForPartials\n");
#endif    
    
    return BEAGLE_SUCCESS;
}

int BeagleGPUImpl::accumulateScaleFactors(const int* scalingIndices,
										  int count,
										  int cumulativeScalingIndex) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::accumulateScaleFactors\n");
#endif
    
    for(int n = 0; n < count; n++)
        hPtrQueue[n] = dScalingFactors[scalingIndices[n]];
    gpu->MemcpyHostToDevice(dPtrQueue, hPtrQueue, sizeof(GPUPtr) * count);
    

#ifdef CUDA    
    // Compute scaling factors at the root
    kernels->AccumulateFactorsDynamicScaling(dPtrQueue, dScalingFactors[cumulativeScalingIndex],
                                             count, kPaddedPatternCount);
#else // OpenCL
    for (int i = 0; i < count; i++) {
        kernels->AccumulateFactorsDynamicScaling(dScalingFactors[scalingIndices[i]], dScalingFactors[cumulativeScalingIndex],
                                                 1, kPaddedPatternCount);
    }
#endif
    
#ifdef BEAGLE_DEBUG_SYNCH    
    gpu->Synchronize();
#endif

#ifdef BEAGLE_DEBUG_VALUES    
    fprintf(stderr, "scaling factors = ");
    gpu->PrintfDeviceVector(dScalingFactors[cumulativeScalingIndex], kPaddedPatternCount);
#endif
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::accumulateScaleFactors\n");
#endif   
    
    return BEAGLE_SUCCESS;
}

int BeagleGPUImpl::removeScaleFactors(const int* scalingIndices,
                                        int count,
                                        int cumulativeScalingIndex) {
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::removeScaleFactors\n");
#endif
    
    for(int n = 0; n < count; n++)
        hPtrQueue[n] = dScalingFactors[scalingIndices[n]];
    gpu->MemcpyHostToDevice(dPtrQueue, hPtrQueue, sizeof(GPUPtr) * count);
    
#ifdef CUDA    
    // Compute scaling factors at the root
    kernels->RemoveFactorsDynamicScaling(dPtrQueue, dScalingFactors[cumulativeScalingIndex],
                                         count, kPaddedPatternCount);
#else // OpenCL
    for (int i = 0; i < count; i++) {
        kernels->RemoveFactorsDynamicScaling(dScalingFactors[scalingIndices[i]], dScalingFactors[cumulativeScalingIndex],
                                             1, kPaddedPatternCount);
    }
    
#endif
    
#ifdef BEAGLE_DEBUG_SYNCH    
    gpu->Synchronize();
#endif

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::removeScaleFactors\n");
#endif        
    
    return BEAGLE_SUCCESS;
}

int BeagleGPUImpl::resetScaleFactors(int cumulativeScalingIndex) {
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::resetScaleFactors\n");
#endif

    REAL* zeroes = (REAL*) calloc(SIZE_REAL, kPaddedPatternCount);
    
    // Fill with zeroes
    gpu->MemcpyHostToDevice(dScalingFactors[cumulativeScalingIndex], zeroes,
                            SIZE_REAL * kPaddedPatternCount);
    
    free(zeroes);
    
#ifdef BEAGLE_DEBUG_SYNCH    
    gpu->Synchronize();
#endif
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::resetScaleFactors\n");
#endif    
    
    return BEAGLE_SUCCESS;
}

int BeagleGPUImpl::calculateRootLogLikelihoods(const int* bufferIndices,
                                               const double* inWeights,
                                               const double* inStateFrequencies,
                                               const int* scalingFactorsIndices,
                                               int count,
                                               double* outLogLikelihoods) {
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::calculateRootLogLikelihoods\n");
#endif

    if (count == 1) {         
        
    	// MAS:  I may have introduced a bug here.
#ifdef DOUBLE_PRECISION
    	const double* tmpWeights = inWeights;        
        memcpy(hFrequenciesCache, inStateFrequencies, kStateCount * SIZE_REAL);
#else
        REAL* tmpWeights = hWeightsCache;
        MEMCNV(hWeightsCache, inWeights, count * kCategoryCount, REAL);
        MEMCNV(hFrequenciesCache, inStateFrequencies, kStateCount, REAL);
#endif        
        gpu->MemcpyHostToDevice(dWeights, tmpWeights, SIZE_REAL * count * kCategoryCount);
        gpu->MemcpyHostToDevice(dFrequencies, hFrequenciesCache, SIZE_REAL * kPaddedStateCount);
   
        const int rootNodeIndex = bufferIndices[0];
        
        int cumulativeScalingFactor = scalingFactorsIndices[0];
        if (cumulativeScalingFactor != BEAGLE_OP_NONE) {
            kernels->IntegrateLikelihoodsDynamicScaling(dIntegrationTmp, dPartials[rootNodeIndex],
                                                        dWeights, dFrequencies,
                                                        dScalingFactors[cumulativeScalingFactor],
                                                        kPaddedPatternCount,
                                                        kCategoryCount);
        } else {
            kernels->IntegrateLikelihoods(dIntegrationTmp, dPartials[rootNodeIndex], dWeights,
                                          dFrequencies, kPaddedPatternCount, kCategoryCount);
        }
        
#ifdef DOUBLE_PRECISION
        gpu->MemcpyDeviceToHost(outLogLikelihoods, dIntegrationTmp, SIZE_REAL * kPatternCount);
#else
        gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dIntegrationTmp, SIZE_REAL * kPatternCount);
        MEMCNV(outLogLikelihoods, hLogLikelihoodsCache, kPatternCount, double);
#endif
        
    } else {
        // TODO: implement calculate root lnL for count > 1
        assert(false);
    }
    
#ifdef BEAGLE_DEBUG_VALUES
    fprintf(stderr, "parent = \n");
    gpu->PrintfDeviceVector(dIntegrationTmp, kPatternCount);    
#endif
    
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::calculateRootLogLikelihoods\n");
#endif
    
    return BEAGLE_SUCCESS;
}

int BeagleGPUImpl::calculateEdgeLogLikelihoods(const int* parentBufferIndices,
                                               const int* childBufferIndices,
                                               const int* probabilityIndices,
                                               const int* firstDerivativeIndices,
                                               const int* secondDerivativeIndices,
                                               const double* inWeights,
                                               const double* inStateFrequencies,
                                               const int* scalingFactorsIndices,
                                               int count,
                                               double* outLogLikelihoods,
                                               double* outFirstDerivatives,
                                               double* outSecondDerivatives) {

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::calculateEdgeLogLikelihoods\n");
#endif
    
    if (count == 1) { 
                 
#ifdef DOUBLE_PRECISION
        const double* tmpWeights = inWeights;
        memcpy(hFrequenciesCache, inStateFrequencies, kStateCount * SIZE_REAL);
#else
        REAL* tmpWeights = hWeightsCache;
        MEMCNV(hWeightsCache, inWeights, count * kCategoryCount, REAL);
        MEMCNV(hFrequenciesCache, inStateFrequencies, kStateCount, REAL);
#endif        
        gpu->MemcpyHostToDevice(dWeights, tmpWeights, SIZE_REAL * count * kCategoryCount);
        gpu->MemcpyHostToDevice(dFrequencies, hFrequenciesCache, SIZE_REAL * kPaddedStateCount);
        
        const int parIndex = parentBufferIndices[0];
        const int childIndex = childBufferIndices[0];
        const int probIndex = probabilityIndices[0];
        
        // TODO: implement derivatives for calculateEdgeLnL
        //        const int firstDerivIndex = firstDerivativeIndices[0];
        //        const int secondDerivIndex = secondDerivativeIndices[0];
        
        GPUPtr partialsParent = dPartials[parIndex];
        GPUPtr partialsChild = dPartials[childIndex];        
        GPUPtr statesChild = dStates[childIndex];
        GPUPtr transMatrix = dMatrices[probIndex];
        //        REAL* firstDerivMatrix = 0L;
        //        REAL* secondDerivMatrix = 0L;
        
        if (statesChild != 0) {
            // TODO: test calculateEdgeLnL when child is of tipStates kind
            kernels->StatesPartialsEdgeLikelihoods(dPartialsTmp, partialsParent, statesChild,
                                                   transMatrix, kPaddedPatternCount,
                                                   kCategoryCount);
        } else {
            kernels->PartialsPartialsEdgeLikelihoods(dPartialsTmp, partialsParent, partialsChild,
                                                     transMatrix, kPaddedPatternCount,
                                                     kCategoryCount);
        }        

        int cumulativeScalingFactor = scalingFactorsIndices[0];
        if (cumulativeScalingFactor != BEAGLE_OP_NONE) {
            kernels->IntegrateLikelihoodsDynamicScaling(dIntegrationTmp, dPartialsTmp, dWeights,
                                                        dFrequencies, dScalingFactors[cumulativeScalingFactor],
                                                        kPaddedPatternCount, kCategoryCount);
        } else {
            kernels->IntegrateLikelihoods(dIntegrationTmp, dPartialsTmp, dWeights, dFrequencies,
                                          kPaddedPatternCount, kCategoryCount);
        }
        
#ifdef DOUBLE_PRECISION
        gpu->MemcpyDeviceToHost(outLogLikelihoods, dIntegrationTmp, SIZE_REAL * kPatternCount);
#else
        gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dIntegrationTmp, SIZE_REAL * kPatternCount);
        MEMCNV(outLogLikelihoods, hLogLikelihoodsCache, kPatternCount, double);
#endif
        
    } else {
        // TODO: implement calculateEdgeLnL for count > 1
        assert(false);
    }
    
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::calculateEdgeLogLikelihoods\n");
#endif
    
    return BEAGLE_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
// BeagleGPUImplFactory public methods

BeagleImpl*  BeagleGPUImplFactory::createImpl(int tipCount,
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
    BeagleImpl* impl = new BeagleGPUImpl();
    try {
        if (impl->createInstance(tipCount, partialsBufferCount, compactBufferCount, stateCount,
                                 patternCount, eigenBufferCount, matrixBufferCount,
                                 categoryCount,scaleBufferCount,preferenceFlags, requirementFlags) == 0)
            return impl;
    }
    catch(...)
    {
        delete impl;
        throw;
    }
    delete impl;
    return NULL;
}

const char* BeagleGPUImplFactory::getName() {
    return "GPU";
}

const long BeagleGPUImplFactory::getFlags() {
   return BEAGLE_FLAG_ASYNCH | BEAGLE_FLAG_GPU | BEAGLE_FLAG_SINGLE;
}
