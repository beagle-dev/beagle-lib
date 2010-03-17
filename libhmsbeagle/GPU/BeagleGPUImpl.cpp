
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

BeagleGPUImpl::BeagleGPUImpl() {
    
    gpu = NULL;
    kernels = NULL;
    
    dIntegrationTmp = NULL;
    dOutFirstDeriv = NULL;
    dOutSecondDeriv = NULL;
    dPartialsTmp = NULL;
    dFirstDerivTmp = NULL;
    dSecondDerivTmp = NULL;
    
    dSumLogLikelihood = NULL;
    dSumFirstDeriv = NULL;
    dSumSecondDeriv = NULL;
    
    dPatternWeights = NULL;    
	
    dBranchLengths = NULL;
    
    dDistanceQueue = NULL;
    
    dPtrQueue = NULL;
	
    dMaxScalingFactors = NULL;
    dIndexMaxScalingFactors = NULL;
    
    dEigenValues = NULL;
    dEvec = NULL;
    dIevc = NULL;
    
    dWeights = NULL;
    dFrequencies = NULL; 
    
    dScalingFactors = NULL;
    
    dStates = NULL;
    
    dPartials = NULL;
    dMatrices = NULL;
    
    dCompactBuffers = NULL;
    dTipPartialsBuffers = NULL;
    
    hPtrQueue = NULL;
    
    hCategoryRates = NULL;
    
    hPatternWeightsCache = NULL;
    
    hDistanceQueue = NULL;
    
    hWeightsCache = NULL;
    hFrequenciesCache = NULL;
    hLogLikelihoodsCache = NULL;
    hPartialsCache = NULL;
    hStatesCache = NULL;
    hMatrixCache = NULL;
}

BeagleGPUImpl::~BeagleGPUImpl() {
    	
	if (kInitialized) {
        for (int i=0; i < kEigenDecompCount; i++) {
            gpu->FreeMemory(dEigenValues[i]);
            gpu->FreeMemory(dEvec[i]);
            gpu->FreeMemory(dIevc[i]);
            gpu->FreeMemory(dWeights[i]);
            gpu->FreeMemory(dFrequencies[i]);
        }
        for (int i=0; i < kMatrixCount; i++)
            gpu->FreeMemory(dMatrices[i]);
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
        
        gpu->FreeMemory(dIntegrationTmp);
        gpu->FreeMemory(dOutFirstDeriv);
        gpu->FreeMemory(dOutSecondDeriv);
        gpu->FreeMemory(dPartialsTmp);
        gpu->FreeMemory(dFirstDerivTmp);
        gpu->FreeMemory(dSecondDerivTmp);
        
        gpu->FreeMemory(dSumLogLikelihood);
        gpu->FreeMemory(dSumFirstDeriv);
        gpu->FreeMemory(dSumSecondDeriv);
        
        gpu->FreeMemory(dPatternWeights);

        gpu->FreeMemory(dBranchLengths);
        
        gpu->FreeMemory(dDistanceQueue);
        
        gpu->FreeMemory(dPtrQueue);
        
        gpu->FreeMemory(dMaxScalingFactors);
        gpu->FreeMemory(dIndexMaxScalingFactors);
        
        if (kFlags & BEAGLE_FLAG_SCALING_AUTO) {
            gpu->FreeMemory(dActiveScalingFactors);
            gpu->FreeMemory(dAccumulatedScalingFactors);
        }
	        
        free(dEigenValues);
        free(dEvec);
        free(dIevc);
    
        free(dWeights);
        free(dFrequencies);

        free(dScalingFactors);

        free(dStates);

        free(dPartials);
        free(dMatrices);

        free(dCompactBuffers);
        free(dTipPartialsBuffers);

        free(hPtrQueue);
    
        free(hCategoryRates);
        
        free(hPatternWeightsCache);
    
        free(hDistanceQueue);
    
        free(hWeightsCache);
        free(hFrequenciesCache);
        free(hLogLikelihoodsCache);
        free(hPartialsCache);
        free(hStatesCache);
        free(hMatrixCache);
    }
    
    if (kernels)
        delete kernels;        
    if (gpu) 
        delete gpu;    
    
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
                                  int iResourceNumber,
                                  long preferenceFlags,
                                  long requirementFlags) {
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::createInstance\n");
#endif
    
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
    
    resourceNumber = iResourceNumber;

    kTipPartialsBufferCount = kTipCount - kCompactBufferCount;
    kBufferCount = kPartialsBufferCount + kCompactBufferCount;

    kInternalPartialsBufferCount = kBufferCount - kTipCount;
    
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
        
    // Make sure that kPaddedPatternCount + paddedPatterns is multiple of 4 for DNA model
    int paddedPatterns = 0;
    if (kPaddedStateCount == 4 && kPatternCount % 4 != 0)
        paddedPatterns = 4 - kPatternCount % 4;
    else
        paddedPatterns = 0;
    
    // TODO Should do something similar for 4 < kStateCount <= 8 as well
    
    kPaddedPatternCount = kPatternCount + paddedPatterns;
    
    int scaleBufferSize = kPaddedPatternCount;
    
    kFlags = 0;
    
    if (preferenceFlags & BEAGLE_FLAG_SCALING_AUTO || requirementFlags & BEAGLE_FLAG_SCALING_AUTO) {
        kFlags |= BEAGLE_FLAG_SCALING_AUTO;
        kFlags |= BEAGLE_FLAG_SCALERS_LOG;
        kScaleBufferCount = kInternalPartialsBufferCount;
        scaleBufferSize *= kCategoryCount;
    } else if (preferenceFlags & BEAGLE_FLAG_SCALING_ALWAYS || requirementFlags & BEAGLE_FLAG_SCALING_ALWAYS) {
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
    
    if (preferenceFlags & BEAGLE_FLAG_EIGEN_COMPLEX || requirementFlags & BEAGLE_FLAG_EIGEN_COMPLEX) {
    	kFlags |= BEAGLE_FLAG_EIGEN_COMPLEX;
    } else {
        kFlags |= BEAGLE_FLAG_EIGEN_REAL;
    }    
    
    kSumSitesBlockCount = kPatternCount / SUM_SITES_BLOCK_SIZE;
    if (kPatternCount % SUM_SITES_BLOCK_SIZE != 0)
        kSumSitesBlockCount += 1;
        
    kPartialsSize = kPaddedPatternCount * kPaddedStateCount * kCategoryCount;
    kMatrixSize = kPaddedStateCount * kPaddedStateCount;

    if (kFlags & BEAGLE_FLAG_EIGEN_COMPLEX)
    	kEigenValuesSize = 2 * kPaddedStateCount;
    else
    	kEigenValuesSize = kPaddedStateCount;
        
    kLastCompactBufferIndex = -1;
    kLastTipPartialsBufferIndex = -1;
        
    gpu = new GPUInterface();
    
    gpu->Initialize();
    
    int numDevices = 0;
    numDevices = gpu->GetDeviceCount();
    if (numDevices == 0) {
        fprintf(stderr, "Error: No GPU devices\n");
        return BEAGLE_ERROR_NO_RESOURCE;
    }
    if (resourceNumber > numDevices) {
        fprintf(stderr,"Error: Trying to initialize device # %d (which does not exist)\n",resourceNumber);
        return BEAGLE_ERROR_NO_RESOURCE;
    }
    
    // TODO: recompiling kernels for every instance, probably not ideal
    gpu->SetDevice(resourceNumber-1,kPaddedStateCount,kCategoryCount,kPaddedPatternCount,kFlags);
      
    int ptrQueueLength = kMatrixCount * kCategoryCount * 3;
    if (kPartialsBufferCount > ptrQueueLength)
        ptrQueueLength = kPartialsBufferCount;
    
    unsigned int neededMemory = SIZE_REAL * (kMatrixSize * kEigenDecompCount + // dEvec
                                             kMatrixSize * kEigenDecompCount + // dIevc
                                             kEigenValuesSize * kEigenDecompCount + // dEigenValues
                                             kCategoryCount * kPartialsBufferCount + // dWeights
                                             kPaddedStateCount * kPartialsBufferCount + // dFrequencies
                                             kPaddedPatternCount + // dIntegrationTmp
                                             kPaddedPatternCount + // dOutFirstDeriv
                                             kPaddedPatternCount + // dOutSecondDeriv
                                             kPartialsSize + // dPartialsTmp
                                             kPartialsSize + // dFirstDerivTmp
                                             kPartialsSize + // dSecondDerivTmp
                                             kScaleBufferCount * kPaddedPatternCount + // dScalingFactors
                                             kPartialsBufferCount * kPartialsSize + // dTipPartialsBuffers + dPartials
                                             kMatrixCount * kMatrixSize * kCategoryCount + // dMatrices
                                             kBufferCount + // dBranchLengths
                                             kMatrixCount * kCategoryCount * 2) + // dDistanceQueue
    SIZE_INT * kCompactBufferCount * kPaddedPatternCount + // dCompactBuffers
    sizeof(GPUPtr) * ptrQueueLength; // dPtrQueue
    
    
    int availableMem = gpu->GetAvailableMemory();
    
#ifdef BEAGLE_DEBUG_VALUES
    fprintf(stderr, "     needed memory: %d\n", neededMemory);
    fprintf(stderr, "  available memory: %d\n", availableMem);
#endif     
    
    if (availableMem < neededMemory) 
        return BEAGLE_ERROR_OUT_OF_MEMORY;
    
    kernels = new KernelLauncher(gpu);
    
    // TODO: only allocate if necessary on the fly
    hWeightsCache = (REAL*) calloc(kCategoryCount * kPartialsBufferCount, SIZE_REAL);
    hFrequenciesCache = (REAL*) calloc(kPaddedStateCount * kPartialsBufferCount, SIZE_REAL);
    hPartialsCache = (REAL*) calloc(kPartialsSize, SIZE_REAL);
    hStatesCache = (int*) calloc(kPaddedPatternCount, SIZE_INT);
    if ((2 * kMatrixSize + kEigenValuesSize) > (kMatrixSize * kCategoryCount))
        hMatrixCache = (REAL*) calloc(2 * kMatrixSize + kEigenValuesSize, SIZE_REAL);
    else
        hMatrixCache = (REAL*) calloc(kMatrixSize * kCategoryCount, SIZE_REAL);
    hLogLikelihoodsCache = (REAL*) malloc(kPatternCount * SIZE_REAL);
    
    dEvec = (GPUPtr*) calloc(sizeof(GPUPtr),kEigenDecompCount);
    dIevc = (GPUPtr*) calloc(sizeof(GPUPtr),kEigenDecompCount);
    dEigenValues = (GPUPtr*) calloc(sizeof(GPUPtr),kEigenDecompCount);
    dWeights = (GPUPtr*) calloc(sizeof(GPUPtr),kEigenDecompCount);
    dFrequencies = (GPUPtr*) calloc(sizeof(GPUPtr),kEigenDecompCount);
    
    for(int i=0; i<kEigenDecompCount; i++) {
    	dEvec[i] = gpu->AllocateRealMemory(kMatrixSize);
    	dIevc[i] = gpu->AllocateRealMemory(kMatrixSize);
    	dEigenValues[i] = gpu->AllocateRealMemory(kEigenValuesSize);
        dWeights[i] = gpu->AllocateRealMemory(kCategoryCount);
        dFrequencies[i] = gpu->AllocateRealMemory(kPaddedStateCount);
    }
    
    
    dIntegrationTmp = gpu->AllocateRealMemory(kPaddedPatternCount);
    dOutFirstDeriv = gpu->AllocateRealMemory(kPaddedPatternCount);
    dOutSecondDeriv = gpu->AllocateRealMemory(kPaddedPatternCount);

    dPatternWeights = gpu->AllocateRealMemory(kPatternCount);
    
    dSumLogLikelihood = gpu->AllocateRealMemory(kSumSitesBlockCount);
    dSumFirstDeriv = gpu->AllocateRealMemory(kSumSitesBlockCount);
    dSumSecondDeriv = gpu->AllocateRealMemory(kSumSitesBlockCount);
    
    dPartialsTmp = gpu->AllocateRealMemory(kPartialsSize);
    dFirstDerivTmp = gpu->AllocateRealMemory(kPartialsSize);
    dSecondDerivTmp = gpu->AllocateRealMemory(kPartialsSize);
    
    // Fill with 0s so 'free' does not choke if unallocated
    dPartials = (GPUPtr*) calloc(sizeof(GPUPtr), kBufferCount);
    
    // Internal nodes have 0s so partials are used
    dStates = (GPUPtr*) calloc(sizeof(GPUPtr), kBufferCount); 
    
    dCompactBuffers = (GPUPtr*) malloc(sizeof(GPUPtr) * kCompactBufferCount); 
    dTipPartialsBuffers = (GPUPtr*) malloc(sizeof(GPUPtr) * kTipPartialsBufferCount);
    
    dScalingFactors = (GPUPtr*) malloc(sizeof(GPUPtr) * kScaleBufferCount);
    if (kFlags & BEAGLE_FLAG_SCALING_AUTO) {
        for (int i=0; i < kScaleBufferCount; i++)
            dScalingFactors[i] = gpu->AllocateMemory(sizeof(signed char) * scaleBufferSize); // TODO: char won't work for double-precision
    } else {
        for (int i=0; i < kScaleBufferCount; i++)
            dScalingFactors[i] = gpu->AllocateRealMemory(scaleBufferSize);
    }
    
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
    
    dDistanceQueue = gpu->AllocateRealMemory(kMatrixCount * kCategoryCount * 2);
    hDistanceQueue = (REAL*) malloc(SIZE_REAL * kMatrixCount * kCategoryCount * 2);
    checkHostMemory(hDistanceQueue);
    
    dPtrQueue = gpu->AllocateMemory(sizeof(GPUPtr) * ptrQueueLength);
    hPtrQueue = (GPUPtr*) malloc(sizeof(GPUPtr) * ptrQueueLength);
    checkHostMemory(hPtrQueue);

	hCategoryRates = (double*) malloc(sizeof(double) * kCategoryCount); // Keep in double-precision
    checkHostMemory(hCategoryRates);

	hPatternWeightsCache = (REAL*) malloc(sizeof(double) * kPatternCount);
    checkHostMemory(hPatternWeightsCache);
    
	dMaxScalingFactors = gpu->AllocateRealMemory(kPaddedPatternCount);
	dIndexMaxScalingFactors = gpu->AllocateIntMemory(kPaddedPatternCount);
    
    if (kFlags & BEAGLE_FLAG_SCALING_AUTO) {
        dActiveScalingFactors = gpu->AllocateMemory(sizeof(unsigned short) * kInternalPartialsBufferCount);
        dAccumulatedScalingFactors = gpu->AllocateMemory(sizeof(int) * scaleBufferSize);
    }

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving BeagleGPUImpl::createInstance\n");
#endif
    
    kInitialized = 1;
    
#ifdef BEAGLE_DEBUG_VALUES
    gpu->Synchronize();
    int usedMemory = availableMem - gpu->GetAvailableMemory();
    fprintf(stderr, "actual used memory: %d\n", usedMemory);
    fprintf(stderr, "        difference: %d\n\n", usedMemory-neededMemory);
#endif
    
    return BEAGLE_SUCCESS;
}

int BeagleGPUImpl::getInstanceDetails(BeagleInstanceDetails* returnInfo) {
    if (returnInfo != NULL) {
        returnInfo->resourceNumber = resourceNumber;
        returnInfo->flags = BEAGLE_FLAG_COMPUTATION_SYNCH |
                            BEAGLE_FLAG_PRECISION_SINGLE |
                            BEAGLE_FLAG_THREADING_NONE |
                            BEAGLE_FLAG_VECTOR_NONE |
                            BEAGLE_FLAG_PROCESSOR_GPU;
        
        returnInfo->flags |= kFlags;        
        
        returnInfo->implName = (char*) "CUDA";
    }
    return BEAGLE_SUCCESS;
}

int BeagleGPUImpl::setTipStates(int tipIndex,
                                const int* inStates) {

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
    // TODO: unscale the partials
    
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
    if (kFlags & BEAGLE_FLAG_EIGEN_COMPLEX)
    	memcpy(Eval+kPaddedStateCount,inEigenValues+kStateCount,SIZE_REAL*kStateCount);
#else
    MEMCNV(Eval, inEigenValues, kStateCount, REAL);
    if (kFlags & BEAGLE_FLAG_EIGEN_COMPLEX)
    	MEMCNV((Eval+kPaddedStateCount),(inEigenValues+kStateCount),kStateCount,REAL);
#endif
    
#ifdef BEAGLE_DEBUG_VALUES
#ifdef DOUBLE_PRECISION
    fprintf(stderr, "Eval:\n");
    printfVectorD(Eval, kEigenValuesSize);
    fprintf(stderr, "Evec:\n");
    printfVectorD(Evec, kMatrixSize);
    fprintf(stderr, "Ievc:\n");
    printfVectorD(Ievc, kPaddedStateCount * kPaddedStateCount);
#else
    fprintf(stderr, "Eval =\n");
    printfVectorF(Eval, kEigenValuesSize);
    fprintf(stderr, "Evec =\n");
    printfVectorF(Evec, kMatrixSize);
    fprintf(stderr, "Ievc =\n");
    printfVectorF(Ievc, kPaddedStateCount * kPaddedStateCount);   
#endif
#endif
    
    // Copy to GPU device
    gpu->MemcpyHostToDevice(dIevc[eigenIndex], Ievc, SIZE_REAL * kMatrixSize);
    gpu->MemcpyHostToDevice(dEvec[eigenIndex], Evec, SIZE_REAL * kMatrixSize);
    gpu->MemcpyHostToDevice(dEigenValues[eigenIndex], Eval, SIZE_REAL * kEigenValuesSize);
    
#ifdef BEAGLE_DEBUG_VALUES
    fprintf(stderr, "dEigenValues =\n");
    gpu->PrintfDeviceVector(dEigenValues[eigenIndex], kEigenValuesSize);
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

int BeagleGPUImpl::setStateFrequencies(int stateFrequenciesIndex,
                                       const double* inStateFrequencies) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::setStateFrequencies\n");
#endif
    
    if (stateFrequenciesIndex < 0 || stateFrequenciesIndex >= kEigenDecompCount)
        return BEAGLE_ERROR_OUT_OF_RANGE;
	
#ifdef DOUBLE_PRECISION
    memcpy(hFrequenciesCache, inStateFrequencies, kStateCount * SIZE_REAL);
#else
    MEMCNV(hFrequenciesCache, inStateFrequencies, kStateCount, REAL);
#endif
	
	gpu->MemcpyHostToDevice(dFrequencies[stateFrequenciesIndex], hFrequenciesCache,
                            SIZE_REAL * kPaddedStateCount);
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::setStateFrequencies\n");
#endif
    
    return BEAGLE_SUCCESS;
}

int BeagleGPUImpl::setCategoryWeights(int categoryWeightsIndex,
                                      const double* inCategoryWeights) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::setCategoryWeights\n");
#endif
    
    if (categoryWeightsIndex < 0 || categoryWeightsIndex >= kEigenDecompCount)
        return BEAGLE_ERROR_OUT_OF_RANGE;
    
#ifdef DOUBLE_PRECISION
	const double* tmpWeights = inCategoryWeights;        
#else
	REAL* tmpWeights = hWeightsCache;
	MEMCNV(hWeightsCache, inCategoryWeights, kCategoryCount, REAL);
#endif
    
	gpu->MemcpyHostToDevice(dWeights[categoryWeightsIndex], tmpWeights,
                            SIZE_REAL * kCategoryCount);
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::setCategoryWeights\n");
#endif
    
    return BEAGLE_SUCCESS;
}

int BeagleGPUImpl::setCategoryRates(const double* inCategoryRates) {

#ifdef BEAGLE_DEBUG_FLOW
	fprintf(stderr, "\tEntering BeagleGPUImpl::updateCategoryRates\n");
#endif

	const double* categoryRates = inCategoryRates;
	// Can keep these in double-precision until after multiplication by (double) branch-length

	memcpy(hCategoryRates, categoryRates, sizeof(double) * kCategoryCount);
    
#ifdef BEAGLE_DEBUG_FLOW
	fprintf(stderr, "\tLeaving  BeagleGPUImpl::updateCategoryRates\n");
#endif
    
    return BEAGLE_SUCCESS;
}

int BeagleGPUImpl::setPatternWeights(const double* inPatternWeights) {
    
#ifdef BEAGLE_DEBUG_FLOW
	fprintf(stderr, "\tEntering BeagleGPUImpl::setPatternWeights\n");
#endif
	
#ifdef DOUBLE_PRECISION
	const double* tmpWeights = inPatternWeights;        
#else
	REAL* tmpWeights = hPatternWeightsCache;
	MEMCNV(hPatternWeightsCache, inPatternWeights, kPatternCount, REAL);
#endif
	
	gpu->MemcpyHostToDevice(dPatternWeights, tmpWeights,
                            SIZE_REAL * kPatternCount);
    
#ifdef BEAGLE_DEBUG_FLOW
	fprintf(stderr, "\tLeaving  BeagleGPUImpl::setPatternWeights\n");
#endif
    
    return BEAGLE_SUCCESS;
}

int BeagleGPUImpl::getTransitionMatrix(int matrixIndex,
									   double* outMatrix) {
	fprintf(stderr,"BeagleGPUImpl::getTransitionMatrix is not yet implemeneted!");
	exit(-1);
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
                                            const int* secondDerivativeIndices,
                                            const double* edgeLengths,
                                            int count) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\tEntering BeagleGPUImpl::updateTransitionMatrices\n");
#endif
    
    // TODO: improve performance of calculation of derivatives
    int totalCount = 0;
    
#ifdef CUDA
    
	if (firstDerivativeIndices == NULL && secondDerivativeIndices == NULL) {
        for (int i = 0; i < count; i++) {        
            for (int j = 0; j < kCategoryCount; j++) {
                hPtrQueue[totalCount] = dMatrices[probabilityIndices[i]] + (j * kMatrixSize * sizeof(GPUPtr));
                hDistanceQueue[totalCount] = (REAL) (edgeLengths[i] * hCategoryRates[j]);
                totalCount++;
            }
        }
        
        gpu->MemcpyHostToDevice(dPtrQueue, hPtrQueue, sizeof(GPUPtr) * totalCount);
        gpu->MemcpyHostToDevice(dDistanceQueue, hDistanceQueue, SIZE_REAL * totalCount);
        
        // Set-up and call GPU kernel
        kernels->GetTransitionProbabilitiesSquare(dPtrQueue, dEvec[eigenIndex], dIevc[eigenIndex],
                                                  dEigenValues[eigenIndex], dDistanceQueue, totalCount);
	} else if (secondDerivativeIndices == NULL) {        
        
        totalCount = count * kCategoryCount;
        int ptrIndex = 0;
        for (int i = 0; i < count; i++) {        
            for (int j = 0; j < kCategoryCount; j++) {
                hPtrQueue[ptrIndex] = dMatrices[probabilityIndices[i]] + (j * kMatrixSize * sizeof(GPUPtr));
                hPtrQueue[ptrIndex + totalCount] = dMatrices[firstDerivativeIndices[i]] + (j * kMatrixSize * sizeof(GPUPtr));
                hDistanceQueue[ptrIndex] = (REAL) (edgeLengths[i]);
                hDistanceQueue[ptrIndex + totalCount] = (REAL) (hCategoryRates[j]);
                ptrIndex++;
            }
        }
        
        gpu->MemcpyHostToDevice(dPtrQueue, hPtrQueue, sizeof(GPUPtr) * totalCount * 2);
        gpu->MemcpyHostToDevice(dDistanceQueue, hDistanceQueue, SIZE_REAL * totalCount * 2);
        
        kernels->GetTransitionProbabilitiesSquareFirstDeriv(dPtrQueue, dEvec[eigenIndex], dIevc[eigenIndex],
                                                             dEigenValues[eigenIndex], dDistanceQueue, totalCount);        
        
    } else {
        totalCount = count * kCategoryCount;
        int ptrIndex = 0;
        for (int i = 0; i < count; i++) {        
            for (int j = 0; j < kCategoryCount; j++) {
                hPtrQueue[ptrIndex] = dMatrices[probabilityIndices[i]] + (j * kMatrixSize * sizeof(GPUPtr));
                hPtrQueue[ptrIndex + totalCount] = dMatrices[firstDerivativeIndices[i]] + (j * kMatrixSize * sizeof(GPUPtr));
                hPtrQueue[ptrIndex + totalCount * 2] = dMatrices[secondDerivativeIndices[i]] + (j * kMatrixSize * sizeof(GPUPtr));
                hDistanceQueue[ptrIndex] = (REAL) (edgeLengths[i]);
                hDistanceQueue[ptrIndex + totalCount] = (REAL) (hCategoryRates[j]);
                ptrIndex++;
            }
        }
        
        gpu->MemcpyHostToDevice(dPtrQueue, hPtrQueue, sizeof(GPUPtr) * totalCount * 3);
        gpu->MemcpyHostToDevice(dDistanceQueue, hDistanceQueue, SIZE_REAL * totalCount * 2);
        
        kernels->GetTransitionProbabilitiesSquareSecondDeriv(dPtrQueue, dEvec[eigenIndex], dIevc[eigenIndex],
                                                  dEigenValues[eigenIndex], dDistanceQueue, totalCount);        
    }
    
    
#else
    // TODO: update OpenCL implementation with derivs
    
    for (int i = 0; i < count; i++) {        
		for (int j = 0; j < kCategoryCount; j++) {
            hDistanceQueue[totalCount] = (REAL) (edgeLengths[i] * hCategoryRates[j]);
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
    for (int i = 0; i < 1; i++) {
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
        GPUPtr scalingFactors = NULL;
        
        if (kFlags & BEAGLE_FLAG_SCALING_AUTO) {
            rescale = 2;
            int sIndex = parIndex - kTipCount;
            scalingFactors = dScalingFactors[sIndex];
            
            // small hack; passing activeScalingFactors GPU pointer via cumulativeScalingBuffer parameter
            cumulativeScalingBuffer = dActiveScalingFactors + (sIndex * sizeof(unsigned short));
            gpu->MemsetShort(cumulativeScalingBuffer, 0, 1);
        } else if (kFlags & BEAGLE_FLAG_SCALING_ALWAYS) {
            rescale = 1;
            scalingFactors = dScalingFactors[parIndex - kTipCount];
        } else if (writeScalingIndex >= 0) {
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
    
    if (kFlags & BEAGLE_FLAG_SCALING_AUTO) {
        
        for(int n = 0; n < count; n++)
            hPtrQueue[n] = dScalingFactors[scalingIndices[n] - kTipCount];
        
        gpu->MemcpyHostToDevice(dPtrQueue, hPtrQueue, sizeof(GPUPtr) * count);
        
        kernels->AccumulateFactorsAutoScaling(dPtrQueue, dAccumulatedScalingFactors, dActiveScalingFactors, count, kPaddedPatternCount);        
                
    } else {        
        for(int n = 0; n < count; n++)
            hPtrQueue[n] = dScalingFactors[scalingIndices[n]];
        gpu->MemcpyHostToDevice(dPtrQueue, hPtrQueue, sizeof(GPUPtr) * count);
        

    #ifdef CUDA    
        // Compute scaling factors at the root
        kernels->AccumulateFactorsDynamicScaling(dPtrQueue, dScalingFactors[cumulativeScalingIndex], count, kPaddedPatternCount);
    #else // OpenCL
        for (int i = 0; i < count; i++) {
            kernels->AccumulateFactorsDynamicScaling(dScalingFactors[scalingIndices[i]], dScalingFactors[cumulativeScalingIndex],
                                                     1, kPaddedPatternCount);
        }
    #endif
}
    
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
                                               const int* categoryWeightsIndices,
                                               const int* stateFrequenciesIndices,
                                               const int* cumulativeScaleIndices,
                                               int count,
                                               double* outSumLogLikelihood) {
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::calculateRootLogLikelihoods\n");
#endif
    
    int returnCode = BEAGLE_SUCCESS;
		
    if (count == 1) {         
        const int rootNodeIndex = bufferIndices[0];
        const int categoryWeightsIndex = categoryWeightsIndices[0];
        const int stateFrequenciesIndex = stateFrequenciesIndices[0];
        

        GPUPtr dCumulativeScalingFactor;
        bool scale = 1;
        if (kFlags & BEAGLE_FLAG_SCALING_AUTO)
            dCumulativeScalingFactor = dAccumulatedScalingFactors;
        else if (kFlags & BEAGLE_FLAG_SCALING_ALWAYS)
            dCumulativeScalingFactor = dScalingFactors[bufferIndices[0] - kTipCount];
        else if (cumulativeScaleIndices[0] != BEAGLE_OP_NONE)
            dCumulativeScalingFactor = dScalingFactors[cumulativeScaleIndices[0]];
        else
            scale = 0;
        
        if (scale) {
            kernels->IntegrateLikelihoodsDynamicScaling(dIntegrationTmp, dPartials[rootNodeIndex],
                                                        dWeights[categoryWeightsIndex],
                                                        dFrequencies[stateFrequenciesIndex],
                                                        dCumulativeScalingFactor,
                                                        dPatternWeights,
                                                        kPaddedPatternCount,
                                                        kCategoryCount);
        } else {
            kernels->IntegrateLikelihoods(dIntegrationTmp, dPartials[rootNodeIndex],
                                          dWeights[categoryWeightsIndex],
                                          dFrequencies[stateFrequenciesIndex],
                                          dPatternWeights,
                                          kPaddedPatternCount, kCategoryCount);
        }
        
        kernels->SumSites1(dIntegrationTmp, dSumLogLikelihood,
                                    kPatternCount);

        gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dSumLogLikelihood, SIZE_REAL * kSumSitesBlockCount);

        *outSumLogLikelihood = 0.0;
        for (int i = 0; i < kSumSitesBlockCount; i++) {
            if (!(hLogLikelihoodsCache[i] - hLogLikelihoodsCache[i] == 0.0))
                returnCode = BEAGLE_ERROR_FLOATING_POINT;
            
            *outSumLogLikelihood += hLogLikelihoodsCache[i];
        }    
        
    } else {
		// TODO: evaluate peformance, maybe break up kernels below for each subsetIndex case
		
        if (kFlags & BEAGLE_FLAG_SCALING_ALWAYS) {
			for(int n = 0; n < count; n++) {
                int cumulativeScalingFactor = bufferIndices[n] - kTipCount; 
				hPtrQueue[n] = dScalingFactors[cumulativeScalingFactor];
            }
			gpu->MemcpyHostToDevice(dPtrQueue, hPtrQueue, sizeof(GPUPtr) * count);    
        } else if (cumulativeScaleIndices[0] != BEAGLE_OP_NONE) {
			for(int n = 0; n < count; n++)
				hPtrQueue[n] = dScalingFactors[cumulativeScaleIndices[n]];
			gpu->MemcpyHostToDevice(dPtrQueue, hPtrQueue, sizeof(GPUPtr) * count);
		}
		
		for (int subsetIndex = 0 ; subsetIndex < count; ++subsetIndex ) {

			const GPUPtr tmpDWeights = dWeights[categoryWeightsIndices[subsetIndex]];
			const GPUPtr tmpDFrequencies = dFrequencies[stateFrequenciesIndices[subsetIndex]];
			const int rootNodeIndex = bufferIndices[subsetIndex];

			if (cumulativeScaleIndices[0] != BEAGLE_OP_NONE || (kFlags & BEAGLE_FLAG_SCALING_ALWAYS)) {
				kernels->IntegrateLikelihoodsFixedScaleMulti(dIntegrationTmp, dPartials[rootNodeIndex], tmpDWeights,
															 tmpDFrequencies, dPtrQueue, dMaxScalingFactors,
															 dIndexMaxScalingFactors, dPatternWeights,
                                                             kPaddedPatternCount,
															 kCategoryCount, count, subsetIndex);
			} else {
                if (subsetIndex == 0) {
					kernels->IntegrateLikelihoodsMulti(dIntegrationTmp, dPartials[rootNodeIndex], tmpDWeights,
													   tmpDFrequencies, dPatternWeights,
                                                       kPaddedPatternCount, kCategoryCount, 0);
				} else if (subsetIndex == count - 1) { 
					kernels->IntegrateLikelihoodsMulti(dIntegrationTmp, dPartials[rootNodeIndex], tmpDWeights,
													   tmpDFrequencies, dPatternWeights,
                                                       kPaddedPatternCount, kCategoryCount, 1);
				} else {
					kernels->IntegrateLikelihoodsMulti(dIntegrationTmp, dPartials[rootNodeIndex], tmpDWeights,
													   tmpDFrequencies, dPatternWeights,
                                                       kPaddedPatternCount, kCategoryCount, 2);
				}
			}
			

            kernels->SumSites1(dIntegrationTmp, dSumLogLikelihood,
                                        kPatternCount);
                        
            gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dSumLogLikelihood, SIZE_REAL * kSumSitesBlockCount);
            
            *outSumLogLikelihood = 0.0;
            for (int i = 0; i < kSumSitesBlockCount; i++) {
                if (!(hLogLikelihoodsCache[i] - hLogLikelihoodsCache[i] == 0.0))
                    returnCode = BEAGLE_ERROR_FLOATING_POINT;
                
                *outSumLogLikelihood += hLogLikelihoodsCache[i];
            }    
		}
    }
    
#ifdef BEAGLE_DEBUG_VALUES
    fprintf(stderr, "parent = \n");
    gpu->PrintfDeviceVector(dIntegrationTmp, kPatternCount);    
#endif
    
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::calculateRootLogLikelihoods\n");
#endif
    
    return returnCode;
}

int BeagleGPUImpl::calculateEdgeLogLikelihoods(const int* parentBufferIndices,
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

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::calculateEdgeLogLikelihoods\n");
#endif
    
    int returnCode = BEAGLE_SUCCESS;
    
    if (count == 1) { 
                 
        
        const int parIndex = parentBufferIndices[0];
        const int childIndex = childBufferIndices[0];
        const int probIndex = probabilityIndices[0];
        
        const int categoryWeightsIndex = categoryWeightsIndices[0];
        const int stateFrequenciesIndex = stateFrequenciesIndices[0];
        
        
        GPUPtr partialsParent = dPartials[parIndex];
        GPUPtr partialsChild = dPartials[childIndex];        
        GPUPtr statesChild = dStates[childIndex];
        GPUPtr transMatrix = dMatrices[probIndex];
        
        
        GPUPtr dCumulativeScalingFactor;
        bool scale = 1;
        if (kFlags & BEAGLE_FLAG_SCALING_AUTO) {
            dCumulativeScalingFactor = dAccumulatedScalingFactors;
        } else if (kFlags & BEAGLE_FLAG_SCALING_ALWAYS) {
            int cumulativeScalingFactor = kInternalPartialsBufferCount;
            int child1ScalingIndex = parIndex - kTipCount;
            int child2ScalingIndex = childIndex - kTipCount;
            resetScaleFactors(cumulativeScalingFactor);
            if (child1ScalingIndex >= 0 && child2ScalingIndex >= 0) {
                int scalingIndices[2] = {child1ScalingIndex, child2ScalingIndex};
                accumulateScaleFactors(scalingIndices, 2, cumulativeScalingFactor);
            } else if (child1ScalingIndex >= 0) {
                int scalingIndices[1] = {child1ScalingIndex};
                accumulateScaleFactors(scalingIndices, 1, cumulativeScalingFactor);
            } else if (child2ScalingIndex >= 0) {
                int scalingIndices[1] = {child2ScalingIndex};
                accumulateScaleFactors(scalingIndices, 1, cumulativeScalingFactor);
            }
            dCumulativeScalingFactor = dScalingFactors[cumulativeScalingFactor];
        } else if (cumulativeScaleIndices[0] != BEAGLE_OP_NONE) {
            dCumulativeScalingFactor = dScalingFactors[cumulativeScaleIndices[0]];
        } else {
            scale = 0;
        }
        
        if (firstDerivativeIndices == NULL && secondDerivativeIndices == NULL) {
            if (statesChild != 0) {
                kernels->StatesPartialsEdgeLikelihoods(dPartialsTmp, partialsParent, statesChild,
                                                       transMatrix, kPaddedPatternCount,
                                                       kCategoryCount);
            } else {
                kernels->PartialsPartialsEdgeLikelihoods(dPartialsTmp, partialsParent, partialsChild,
                                                         transMatrix, kPaddedPatternCount,
                                                         kCategoryCount);
            }        
            
            
            if (scale) {
                kernels->IntegrateLikelihoodsDynamicScaling(dIntegrationTmp, dPartialsTmp, dWeights[categoryWeightsIndex],
                                                            dFrequencies[stateFrequenciesIndex],
                                                            dCumulativeScalingFactor,
                                                            dPatternWeights,
                                                            kPaddedPatternCount, kCategoryCount);
            } else {
                kernels->IntegrateLikelihoods(dIntegrationTmp, dPartialsTmp, dWeights[categoryWeightsIndex],
                                              dFrequencies[stateFrequenciesIndex], dPatternWeights,
                                              kPaddedPatternCount, kCategoryCount);
            }
            
            kernels->SumSites1(dIntegrationTmp, dSumLogLikelihood,
                                        kPatternCount);
            
            gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dSumLogLikelihood, SIZE_REAL * kSumSitesBlockCount);
            
            *outSumLogLikelihood = 0.0;
            for (int i = 0; i < kSumSitesBlockCount; i++) {
                if (!(hLogLikelihoodsCache[i] - hLogLikelihoodsCache[i] == 0.0))
                    returnCode = BEAGLE_ERROR_FLOATING_POINT;
                
                *outSumLogLikelihood += hLogLikelihoodsCache[i];
            }    
		} else if (secondDerivativeIndices == NULL) {
            // TODO: remove this "hack" for a proper version that only calculates firstDeriv
            
            const int firstDerivIndex = firstDerivativeIndices[0];
            GPUPtr firstDerivMatrix = dMatrices[firstDerivIndex];
            GPUPtr secondDerivMatrix = dMatrices[firstDerivIndex];
            
            if (statesChild != 0) {
                // TODO: test GPU derivative matrices for statesPartials (including extra ambiguity column)
                kernels->StatesPartialsEdgeLikelihoodsSecondDeriv(dPartialsTmp, dFirstDerivTmp, dSecondDerivTmp,
                                                                  partialsParent, statesChild,
                                                                  transMatrix, firstDerivMatrix, secondDerivMatrix,
                                                                  kPaddedPatternCount, kCategoryCount);
            } else {
                kernels->PartialsPartialsEdgeLikelihoodsSecondDeriv(dPartialsTmp, dFirstDerivTmp, dSecondDerivTmp,
                                                                    partialsParent, partialsChild,
                                                                    transMatrix, firstDerivMatrix, secondDerivMatrix,
                                                                    kPaddedPatternCount, kCategoryCount);
                
            }
                        
            if (scale) {
                kernels->IntegrateLikelihoodsDynamicScalingSecondDeriv(dIntegrationTmp, dOutFirstDeriv, dOutSecondDeriv,
                                                                       dPartialsTmp, dFirstDerivTmp, dSecondDerivTmp,
                                                                       dWeights[categoryWeightsIndex],
                                                                       dFrequencies[stateFrequenciesIndex],
                                                                       dCumulativeScalingFactor,
                                                                       dPatternWeights,
                                                                       kPaddedPatternCount, kCategoryCount);
            } else {
                kernels->IntegrateLikelihoodsSecondDeriv(dIntegrationTmp, dOutFirstDeriv, dOutSecondDeriv,
                                                         dPartialsTmp, dFirstDerivTmp, dSecondDerivTmp,
                                                         dWeights[categoryWeightsIndex],
                                                         dFrequencies[stateFrequenciesIndex],
                                                         dPatternWeights,
                                                         kPaddedPatternCount, kCategoryCount);
            }
            

            kernels->SumSites2(dIntegrationTmp, dSumLogLikelihood, dOutFirstDeriv, dSumFirstDeriv,
                                        kPatternCount);
            
            gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dSumLogLikelihood, SIZE_REAL * kSumSitesBlockCount);
            
            *outSumLogLikelihood = 0.0;
            for (int i = 0; i < kSumSitesBlockCount; i++) {
                if (!(hLogLikelihoodsCache[i] - hLogLikelihoodsCache[i] == 0.0))
                    returnCode = BEAGLE_ERROR_FLOATING_POINT;
                
                *outSumLogLikelihood += hLogLikelihoodsCache[i];
            }    
            
            gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dSumFirstDeriv, SIZE_REAL * kSumSitesBlockCount);
            
            *outSumFirstDerivative = 0.0;
            for (int i = 0; i < kSumSitesBlockCount; i++) {
                *outSumFirstDerivative += hLogLikelihoodsCache[i];
            }                
            
		} else {
            // TODO: improve performance of GPU implementation of derivatives for calculateEdgeLnL

            const int firstDerivIndex = firstDerivativeIndices[0];
            const int secondDerivIndex = secondDerivativeIndices[0];
            GPUPtr firstDerivMatrix = dMatrices[firstDerivIndex];
            GPUPtr secondDerivMatrix = dMatrices[secondDerivIndex];
            
            if (statesChild != 0) {
                // TODO: test GPU derivative matrices for statesPartials (including extra ambiguity column)
                kernels->StatesPartialsEdgeLikelihoodsSecondDeriv(dPartialsTmp, dFirstDerivTmp, dSecondDerivTmp,
                                                                  partialsParent, statesChild,
                                                                  transMatrix, firstDerivMatrix, secondDerivMatrix,
                                                                  kPaddedPatternCount, kCategoryCount);
            } else {
                kernels->PartialsPartialsEdgeLikelihoodsSecondDeriv(dPartialsTmp, dFirstDerivTmp, dSecondDerivTmp,
                                                                    partialsParent, partialsChild,
                                                                    transMatrix, firstDerivMatrix, secondDerivMatrix,
                                                                    kPaddedPatternCount, kCategoryCount);
                
            }
            
            if (scale) {
                kernels->IntegrateLikelihoodsDynamicScalingSecondDeriv(dIntegrationTmp, dOutFirstDeriv, dOutSecondDeriv,
                                                                       dPartialsTmp, dFirstDerivTmp, dSecondDerivTmp,
                                                                       dWeights[categoryWeightsIndex],
                                                                       dFrequencies[stateFrequenciesIndex],
                                                                       dCumulativeScalingFactor,
                                                                       dPatternWeights,
                                                                       kPaddedPatternCount, kCategoryCount);
            } else {
                kernels->IntegrateLikelihoodsSecondDeriv(dIntegrationTmp, dOutFirstDeriv, dOutSecondDeriv,
                                                         dPartialsTmp, dFirstDerivTmp, dSecondDerivTmp,
                                                         dWeights[categoryWeightsIndex],
                                                         dFrequencies[stateFrequenciesIndex],
                                                         dPatternWeights,
                                                         kPaddedPatternCount, kCategoryCount);
            }
            
            kernels->SumSites3(dIntegrationTmp, dSumLogLikelihood, dOutFirstDeriv, dSumFirstDeriv, dOutSecondDeriv, dSumSecondDeriv,
                              kPatternCount);
            
            gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dSumLogLikelihood, SIZE_REAL * kSumSitesBlockCount);
            
            *outSumLogLikelihood = 0.0;
            for (int i = 0; i < kSumSitesBlockCount; i++) {
                if (!(hLogLikelihoodsCache[i] - hLogLikelihoodsCache[i] == 0.0))
                    returnCode = BEAGLE_ERROR_FLOATING_POINT;
                
                *outSumLogLikelihood += hLogLikelihoodsCache[i];
            }    
            
            gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dSumFirstDeriv, SIZE_REAL * kSumSitesBlockCount);
            
            *outSumFirstDerivative = 0.0;
            for (int i = 0; i < kSumSitesBlockCount; i++) {
                *outSumFirstDerivative += hLogLikelihoodsCache[i];
            }   

            gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dSumSecondDeriv, SIZE_REAL * kSumSitesBlockCount);
            
            *outSumSecondDerivative = 0.0;
            for (int i = 0; i < kSumSitesBlockCount; i++) {
                *outSumSecondDerivative += hLogLikelihoodsCache[i];
            }   
        }
        
        
    } else {
        // TODO: implement calculateEdgeLnL for count > 1
        assert(false);
    }
    
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::calculateEdgeLogLikelihoods\n");
#endif
    
    return returnCode;
}

int BeagleGPUImpl::getSiteLogLikelihoods(double* outLogLikelihoods) {
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::getSiteLogLikelihoods\n");
#endif
    
#ifdef DOUBLE_PRECISION
    gpu->MemcpyDeviceToHost(outLogLikelihoods, dIntegrationTmp, SIZE_REAL * kPatternCount);
#else
    gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dIntegrationTmp, SIZE_REAL * kPatternCount);
    MEMCNV(outLogLikelihoods, hLogLikelihoodsCache, kPatternCount, double);
#endif    
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::getSiteLogLikelihoods\n");
#endif    
    
    return BEAGLE_SUCCESS;
}

int BeagleGPUImpl::getSiteDerivatives(double* outFirstDerivatives,
                                      double* outSecondDerivatives) {
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::getSiteDerivatives\n");
#endif
    
#ifdef DOUBLE_PRECISION
    gpu->MemcpyDeviceToHost(outFirstDerivatives, dOutFirstDeriv, SIZE_REAL * kPatternCount);
#else    
    gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dOutFirstDeriv, SIZE_REAL * kPatternCount);
    MEMCNV(outFirstDerivatives, hLogLikelihoodsCache, kPatternCount, double);
#endif                                        

    if (outSecondDerivatives != NULL) {
#ifdef DOUBLE_PRECISION
        gpu->MemcpyDeviceToHost(outSecondDerivatives, dOutSecondDeriv, SIZE_REAL * kPatternCount);
#else    
        gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dOutSecondDeriv, SIZE_REAL * kPatternCount);
        MEMCNV(outSecondDerivatives, hLogLikelihoodsCache, kPatternCount, double);
#endif                                        
    }
    
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::getSiteDerivatives\n");
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
                                              int resourceNumber,
                                              long preferenceFlags,
                                              long requirementFlags,
                                              int* errorCode) {
    BeagleImpl* impl = new BeagleGPUImpl();
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
    catch(...)
    {
        delete impl;
        *errorCode = BEAGLE_ERROR_GENERAL;
        throw;
    }
    delete impl;
    *errorCode = BEAGLE_ERROR_GENERAL;
    return NULL;
}

const char* BeagleGPUImplFactory::getName() {
    return "GPU";
}

const long BeagleGPUImplFactory::getFlags() {
   return BEAGLE_FLAG_COMPUTATION_SYNCH |
          BEAGLE_FLAG_PRECISION_SINGLE |
          BEAGLE_FLAG_SCALING_MANUAL | BEAGLE_FLAG_SCALING_ALWAYS | BEAGLE_FLAG_SCALING_AUTO |
          BEAGLE_FLAG_THREADING_NONE |
          BEAGLE_FLAG_VECTOR_NONE |
          BEAGLE_FLAG_PROCESSOR_GPU |
          BEAGLE_FLAG_SCALERS_LOG | BEAGLE_FLAG_SCALERS_RAW |
          BEAGLE_FLAG_EIGEN_COMPLEX | BEAGLE_FLAG_EIGEN_REAL;
}
