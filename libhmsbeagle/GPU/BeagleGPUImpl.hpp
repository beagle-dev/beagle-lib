
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
#include "libhmsbeagle/GPU/Precision.h"

namespace beagle {
namespace gpu {

BEAGLE_GPU_TEMPLATE
BeagleGPUImpl<BEAGLE_GPU_GENERIC>::BeagleGPUImpl() {
    
    gpu = NULL;
    kernels = NULL;
    
    dIntegrationTmp = (GPUPtr)NULL;
    dOutFirstDeriv = (GPUPtr)NULL;
    dOutSecondDeriv = (GPUPtr)NULL;
    dPartialsTmp = (GPUPtr)NULL;
    dFirstDerivTmp = (GPUPtr)NULL;
    dSecondDerivTmp = (GPUPtr)NULL;
    
    dSumLogLikelihood = (GPUPtr)NULL;
    dSumFirstDeriv = (GPUPtr)NULL;
    dSumSecondDeriv = (GPUPtr)NULL;
    
    dPatternWeights = (GPUPtr)NULL;    
	
    dBranchLengths = (GPUPtr)NULL;
    
    dDistanceQueue = (GPUPtr)NULL;
    
    dPtrQueue = (GPUPtr)NULL;
    
    dMaxScalingFactors = (GPUPtr)NULL;
    dIndexMaxScalingFactors = (GPUPtr)NULL;
    
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
    
    hRescalingTrigger = NULL;
    dRescalingTrigger = (GPUPtr)NULL;
    dScalingFactorsMaster = NULL;
}

BEAGLE_GPU_TEMPLATE
BeagleGPUImpl<BEAGLE_GPU_GENERIC>::~BeagleGPUImpl() {
    	
	if (kInitialized) {
        for (int i=0; i < kEigenDecompCount; i++) {
            gpu->FreeMemory(dEigenValues[i]);
            gpu->FreeMemory(dEvec[i]);
            gpu->FreeMemory(dIevc[i]);
            gpu->FreeMemory(dWeights[i]);
            gpu->FreeMemory(dFrequencies[i]);
        }

        gpu->FreeMemory(dMatrices[0]);
        
        if (kFlags & BEAGLE_FLAG_SCALING_DYNAMIC) {
            gpu->FreePinnedHostMemory(hRescalingTrigger);
            for (int i = 0; i < kScaleBufferCount; i++) {
                if (dScalingFactorsMaster[i] != 0)
                    gpu->FreeMemory(dScalingFactorsMaster[i]);
            }
            free(dScalingFactorsMaster);
        } else {
            if (kScaleBufferCount > 0)
                gpu->FreeMemory(dScalingFactors[0]);
        }
        
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

        gpu->FreeHostMemory(hPtrQueue);
        
        gpu->FreeHostMemory(hCategoryRates);
        
        gpu->FreeHostMemory(hPatternWeightsCache);
    
        gpu->FreeHostMemory(hDistanceQueue);
    
        gpu->FreeHostMemory(hWeightsCache);
        gpu->FreeHostMemory(hFrequenciesCache);
        gpu->FreeHostMemory(hPartialsCache);
        gpu->FreeHostMemory(hStatesCache);
                
        gpu->FreeHostMemory(hLogLikelihoodsCache);
        gpu->FreeHostMemory(hMatrixCache);
        
    }
    
    if (kernels)
        delete kernels;        
    if (gpu) 
        delete gpu;    
    
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::createInstance(int tipCount,
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
    else if (kStateCount <= 80)
		kPaddedStateCount = 80;
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
            
    // TODO Should do something similar for 4 < kStateCount <= 8 as well
    
    kPaddedPatternCount = kPatternCount + paddedPatterns;
    
    int resultPaddedPatterns = 0;
    int patternBlockSizeFour = (kFlags & BEAGLE_FLAG_PRECISION_DOUBLE ? PATTERN_BLOCK_SIZE_DP_4 : PATTERN_BLOCK_SIZE_SP_4);
    if (kPaddedStateCount == 4 && kPaddedPatternCount % patternBlockSizeFour != 0)
        resultPaddedPatterns = patternBlockSizeFour - kPaddedPatternCount % patternBlockSizeFour;
        
    kScaleBufferSize = kPaddedPatternCount;
    
    kFlags = 0;
    
    if (preferenceFlags & BEAGLE_FLAG_SCALING_AUTO || requirementFlags & BEAGLE_FLAG_SCALING_AUTO) {
        kFlags |= BEAGLE_FLAG_SCALING_AUTO;
        kFlags |= BEAGLE_FLAG_SCALERS_LOG;
        kScaleBufferCount = kInternalPartialsBufferCount;
        kScaleBufferSize *= kCategoryCount;
    } else if (preferenceFlags & BEAGLE_FLAG_SCALING_ALWAYS || requirementFlags & BEAGLE_FLAG_SCALING_ALWAYS) {
        kFlags |= BEAGLE_FLAG_SCALING_ALWAYS;
        kFlags |= BEAGLE_FLAG_SCALERS_LOG;
        kScaleBufferCount = kInternalPartialsBufferCount + 1; // +1 for temp buffer used by edgelikelihood
    } else if (preferenceFlags & BEAGLE_FLAG_SCALING_DYNAMIC || requirementFlags & BEAGLE_FLAG_SCALING_DYNAMIC) {
        kFlags |= BEAGLE_FLAG_SCALING_DYNAMIC;
        kFlags |= BEAGLE_FLAG_SCALERS_RAW;
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
    
    if (requirementFlags & BEAGLE_FLAG_INVEVEC_TRANSPOSED || preferenceFlags & BEAGLE_FLAG_INVEVEC_TRANSPOSED)
    	kFlags |= BEAGLE_FLAG_INVEVEC_TRANSPOSED;
    else
        kFlags |= BEAGLE_FLAG_INVEVEC_STANDARD;

    Real r = 0;
    modifyFlagsForPrecision(&kFlags, r);
    
    int sumSitesBlockSize = (kFlags & BEAGLE_FLAG_PRECISION_DOUBLE ? SUM_SITES_BLOCK_SIZE_DP : SUM_SITES_BLOCK_SIZE_SP);
    kSumSitesBlockCount = kPatternCount / sumSitesBlockSize;
    if (kPatternCount % sumSitesBlockSize != 0)
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
    
    unsigned int neededMemory = sizeof(Real) * (kMatrixSize * kEigenDecompCount + // dEvec
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
    sizeof(int) * kCompactBufferCount * kPaddedPatternCount + // dCompactBuffers
    sizeof(GPUPtr) * ptrQueueLength;  // dPtrQueue
    
    
    unsigned int availableMem = gpu->GetAvailableMemory();
    
#ifdef BEAGLE_DEBUG_VALUES
    fprintf(stderr, "     needed memory: %d\n", neededMemory);
    fprintf(stderr, "  available memory: %d\n", availableMem);
#endif     
    
    if (availableMem < neededMemory) 
        return BEAGLE_ERROR_OUT_OF_MEMORY;
    
    kernels = new KernelLauncher(gpu);
    
    // TODO: only allocate if necessary on the fly
    hWeightsCache = (Real*) gpu->CallocHost(kCategoryCount * kPartialsBufferCount, sizeof(Real));
    hFrequenciesCache = (Real*) gpu->CallocHost(kPaddedStateCount * kPartialsBufferCount, sizeof(Real));
    hPartialsCache = (Real*) gpu->CallocHost(kPartialsSize, sizeof(Real));
    hStatesCache = (int*) gpu->CallocHost(kPaddedPatternCount, sizeof(int));

    int hMatrixCacheSize = kMatrixSize * kCategoryCount * BEAGLE_CACHED_MATRICES_COUNT;
    if ((2 * kMatrixSize + kEigenValuesSize) > hMatrixCacheSize)
        hMatrixCacheSize = 2 * kMatrixSize + kEigenValuesSize;
    
    hLogLikelihoodsCache = (Real*) gpu->MallocHost(kPatternCount * sizeof(Real));
    hMatrixCache = (Real*) gpu->CallocHost(hMatrixCacheSize, sizeof(Real));
    
    dEvec = (GPUPtr*) calloc(sizeof(GPUPtr),kEigenDecompCount);
    dIevc = (GPUPtr*) calloc(sizeof(GPUPtr),kEigenDecompCount);
    dEigenValues = (GPUPtr*) calloc(sizeof(GPUPtr),kEigenDecompCount);
    dWeights = (GPUPtr*) calloc(sizeof(GPUPtr),kEigenDecompCount);
    dFrequencies = (GPUPtr*) calloc(sizeof(GPUPtr),kEigenDecompCount);
    
    dMatrices = (GPUPtr*) malloc(sizeof(GPUPtr) * kMatrixCount);
    dMatrices[0] = gpu->AllocateMemory(kMatrixCount * kMatrixSize * kCategoryCount * sizeof(Real));
    
#ifdef CUDA
    for (int i = 1; i < kMatrixCount; i++) {
        dMatrices[i] = dMatrices[i-1] + kMatrixSize * kCategoryCount * sizeof(Real);
    }
    
    if (kScaleBufferCount > 0) {
        if (kFlags & BEAGLE_FLAG_SCALING_AUTO) {        
            dScalingFactors = (GPUPtr*) malloc(sizeof(GPUPtr) * kScaleBufferCount);
            dScalingFactors[0] =  gpu->AllocateMemory(sizeof(signed char) * kScaleBufferSize * kScaleBufferCount); // TODO: char won't work for double-precision
            for (int i=1; i < kScaleBufferCount; i++)
                dScalingFactors[i] = dScalingFactors[i-1] + kScaleBufferSize * sizeof(signed char);
        } else if (kFlags & BEAGLE_FLAG_SCALING_DYNAMIC) {
            dScalingFactors = (GPUPtr*) calloc(sizeof(GPUPtr), kScaleBufferCount);
            dScalingFactorsMaster = (GPUPtr*) calloc(sizeof(GPUPtr), kScaleBufferCount);
            hRescalingTrigger = (int*) gpu->AllocatePinnedHostMemory(sizeof(int), false, true);
            dRescalingTrigger = gpu->GetDevicePointer((void*) hRescalingTrigger);
        } else {
            dScalingFactors = (GPUPtr*) malloc(sizeof(GPUPtr) * kScaleBufferCount);
            dScalingFactors[0] = gpu->AllocateMemory(kScaleBufferSize * kScaleBufferCount * sizeof(Real));
            for (int i=1; i < kScaleBufferCount; i++) {
                dScalingFactors[i] = dScalingFactors[i-1] + kScaleBufferSize * sizeof(Real);
            }
        }        
    }
#endif
    
    for(int i=0; i<kEigenDecompCount; i++) {
    	dEvec[i] = gpu->AllocateMemory(kMatrixSize * sizeof(Real));
    	dIevc[i] = gpu->AllocateMemory(kMatrixSize * sizeof(Real));
    	dEigenValues[i] = gpu->AllocateMemory(kEigenValuesSize * sizeof(Real));
        dWeights[i] = gpu->AllocateMemory(kCategoryCount * sizeof(Real));
        dFrequencies[i] = gpu->AllocateMemory(kPaddedStateCount * sizeof(Real));
    }
    
    
    dIntegrationTmp = gpu->AllocateMemory((kPaddedPatternCount + resultPaddedPatterns) * sizeof(Real));
    dOutFirstDeriv = gpu->AllocateMemory((kPaddedPatternCount + resultPaddedPatterns) * sizeof(Real));
    dOutSecondDeriv = gpu->AllocateMemory((kPaddedPatternCount + resultPaddedPatterns) * sizeof(Real));

    dPatternWeights = gpu->AllocateMemory(kPatternCount * sizeof(Real));
    
    dSumLogLikelihood = gpu->AllocateMemory(kSumSitesBlockCount * sizeof(Real));
    dSumFirstDeriv = gpu->AllocateMemory(kSumSitesBlockCount * sizeof(Real));
    dSumSecondDeriv = gpu->AllocateMemory(kSumSitesBlockCount * sizeof(Real));
    
    dPartialsTmp = gpu->AllocateMemory(kPartialsSize * sizeof(Real));
    dFirstDerivTmp = gpu->AllocateMemory(kPartialsSize * sizeof(Real));
    dSecondDerivTmp = gpu->AllocateMemory(kPartialsSize * sizeof(Real));
    
    // Fill with 0s so 'free' does not choke if unallocated
    dPartials = (GPUPtr*) calloc(sizeof(GPUPtr), kBufferCount);
    
    // Internal nodes have 0s so partials are used
    dStates = (GPUPtr*) calloc(sizeof(GPUPtr), kBufferCount); 
    
    dCompactBuffers = (GPUPtr*) malloc(sizeof(GPUPtr) * kCompactBufferCount); 
    dTipPartialsBuffers = (GPUPtr*) malloc(sizeof(GPUPtr) * kTipPartialsBufferCount);
    
    for (int i = 0; i < kBufferCount; i++) {        
        if (i < kTipCount) { // For the tips
            if (i < kCompactBufferCount)
                dCompactBuffers[i] = gpu->AllocateMemory(kPaddedPatternCount * sizeof(int));
            if (i < kTipPartialsBufferCount)
                dTipPartialsBuffers[i] = gpu->AllocateMemory(kPartialsSize * sizeof(Real));
        } else {
            dPartials[i] = gpu->AllocateMemory(kPartialsSize * sizeof(Real));
        }
    }
    
    kLastCompactBufferIndex = kCompactBufferCount - 1;
    kLastTipPartialsBufferIndex = kTipPartialsBufferCount - 1;
        
    // No execution has more no kBufferCount events
    dBranchLengths = gpu->AllocateMemory(kBufferCount * sizeof(Real));
    
    dDistanceQueue = gpu->AllocateMemory(kMatrixCount * kCategoryCount * 2 * sizeof(Real));
    hDistanceQueue = (Real*) gpu->MallocHost(sizeof(Real) * kMatrixCount * kCategoryCount * 2);
    checkHostMemory(hDistanceQueue);
    
    dPtrQueue = gpu->AllocateMemory(sizeof(unsigned int) * ptrQueueLength);
    hPtrQueue = (unsigned int*) gpu->MallocHost(sizeof(unsigned int) * ptrQueueLength);
    checkHostMemory(hPtrQueue);
    
	hCategoryRates = (double*) gpu->MallocHost(sizeof(double) * kCategoryCount); // Keep in double-precision
    checkHostMemory(hCategoryRates);

	hPatternWeightsCache = (Real*) gpu->MallocHost(sizeof(double) * kPatternCount);
    checkHostMemory(hPatternWeightsCache);
    
	dMaxScalingFactors = gpu->AllocateMemory(kPaddedPatternCount * sizeof(Real));
	dIndexMaxScalingFactors = gpu->AllocateMemory(kPaddedPatternCount * sizeof(int));
    
    if (kFlags & BEAGLE_FLAG_SCALING_AUTO) {
        dAccumulatedScalingFactors = gpu->AllocateMemory(sizeof(int) * kScaleBufferSize);
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

template<>
char* BeagleGPUImpl<double>::getInstanceName() {
	return (char*) "CUDA-Double";
}

template<>
char* BeagleGPUImpl<float>::getInstanceName() {
	return (char*) "CUDA-Single";
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::getInstanceDetails(BeagleInstanceDetails* returnInfo) {
    if (returnInfo != NULL) {
        returnInfo->resourceNumber = resourceNumber;
        returnInfo->flags = BEAGLE_FLAG_COMPUTATION_SYNCH |
                            BEAGLE_FLAG_THREADING_NONE |
                            BEAGLE_FLAG_VECTOR_NONE |
                            BEAGLE_FLAG_PROCESSOR_GPU;
        Real r = 0;
        modifyFlagsForPrecision(&(returnInfo->flags), r);

        returnInfo->flags |= kFlags;        
        
        returnInfo->implName = getInstanceName();
    }
    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::setTipStates(int tipIndex,
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
    gpu->MemcpyHostToDevice(dStates[tipIndex], hStatesCache, sizeof(int) * kPaddedPatternCount);
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::setTipStates\n");
#endif
    
    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::setTipPartials(int tipIndex,
                                  const double* inPartials) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::setTipPartials\n");
#endif
    
    if (tipIndex < 0 || tipIndex >= kTipCount)
        return BEAGLE_ERROR_OUT_OF_RANGE;

    const double* inPartialsOffset = inPartials;
    Real* tmpRealPartialsOffset = hPartialsCache;
    for (int i = 0; i < kPatternCount; i++) {
//#ifdef DOUBLE_PRECISION
//        memcpy(tmpRealPartialsOffset, inPartialsOffset, sizeof(Real) * kStateCount);
//#else
//        MEMCNV(tmpRealPartialsOffset, inPartialsOffset, kStateCount, Real);
//#endif
    	beagleMemCpy(tmpRealPartialsOffset, inPartialsOffset, kStateCount);
        tmpRealPartialsOffset += kPaddedStateCount;
        inPartialsOffset += kStateCount;
    }
    
    int partialsLength = kPaddedPatternCount * kPaddedStateCount;
    for (int i = 1; i < kCategoryCount; i++) {
        memcpy(hPartialsCache + i * partialsLength, hPartialsCache, partialsLength * sizeof(Real));
    }    
    
    if (tipIndex < kTipCount) {
        if (dPartials[tipIndex] == 0) {
            assert(kLastTipPartialsBufferIndex >= 0 && kLastTipPartialsBufferIndex <
                   kTipPartialsBufferCount);
            dPartials[tipIndex] = dTipPartialsBuffers[kLastTipPartialsBufferIndex--];
        }
    }
    // Copy to GPU device
    gpu->MemcpyHostToDevice(dPartials[tipIndex], hPartialsCache, sizeof(Real) * kPartialsSize);
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::setTipPartials\n");
#endif
    
    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::setPartials(int bufferIndex,
                               const double* inPartials) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::setPartials\n");
#endif
    
    if (bufferIndex < 0 || bufferIndex >= kPartialsBufferCount)
        return BEAGLE_ERROR_OUT_OF_RANGE;

    const double* inPartialsOffset = inPartials;
    Real* tmpRealPartialsOffset = hPartialsCache;
    for (int l = 0; l < kCategoryCount; l++) {
        for (int i = 0; i < kPatternCount; i++) {
//#ifdef DOUBLE_PRECISION
//            memcpy(tmpRealPartialsOffset, inPartialsOffset, sizeof(Real) * kStateCount);
//#else
//            MEMCNV(tmpRealPartialsOffset, inPartialsOffset, kStateCount, Real);
//#endif
        	beagleMemCpy(tmpRealPartialsOffset, inPartialsOffset, kStateCount);
            tmpRealPartialsOffset += kPaddedStateCount;
            inPartialsOffset += kStateCount;
        }
        tmpRealPartialsOffset += kPaddedStateCount * (kPaddedPatternCount - kPatternCount);
    }
    
    if (bufferIndex < kTipCount) {
        if (dPartials[bufferIndex] == 0) {
            assert(kLastTipPartialsBufferIndex >= 0 && kLastTipPartialsBufferIndex <
                   kTipPartialsBufferCount);
            dPartials[bufferIndex] = dTipPartialsBuffers[kLastTipPartialsBufferIndex--];
        }
    }
    // Copy to GPU device
    gpu->MemcpyHostToDevice(dPartials[bufferIndex], hPartialsCache, sizeof(Real) * kPartialsSize);
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::setPartials\n");
#endif
    
    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::getPartials(int bufferIndex,
							   int scaleIndex,
                               double* outPartials) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::getPartials\n");
#endif

    gpu->MemcpyDeviceToHost(hPartialsCache, dPartials[bufferIndex], sizeof(Real) * kPartialsSize);
    
    double* outPartialsOffset = outPartials;
    Real* tmpRealPartialsOffset = hPartialsCache;
    
    for (int i = 0; i < kPatternCount; i++) {
//#ifdef DOUBLE_PRECISION
//        memcpy(outPartialsOffset, tmpRealPartialsOffset, sizeof(Real) * kStateCount);
//#else
//        MEMCNV(outPartialsOffset, tmpRealPartialsOffset, kStateCount, double);
//#endif
    	beagleMemCpy(outPartialsOffset, tmpRealPartialsOffset, kStateCount);
        tmpRealPartialsOffset += kPaddedStateCount;
        outPartialsOffset += kStateCount;
    }
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::getPartials\n");
#endif
    
    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::setEigenDecomposition(int eigenIndex,
                                         const double* inEigenVectors,
                                         const double* inInverseEigenVectors,
                                         const double* inEigenValues) {
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\tEntering BeagleGPUImpl::setEigenDecomposition\n");
#endif
    
    // Native memory packing order (length): Ievc (state^2), Evec (state^2),
    //  Eval (state), EvalImag (state)
    
    Real* Ievc, * tmpIevc, * Evec, * tmpEvec, * Eval;
    
    tmpIevc = Ievc = (Real*) hMatrixCache;
    tmpEvec = Evec = Ievc + kMatrixSize;
    Eval = Evec + kMatrixSize;
    
    for (int i = 0; i < kStateCount; i++) {
//#ifdef DOUBLE_PRECISION
//        memcpy(tmpIevc, inInverseEigenVectors + i * kStateCount, sizeof(Real) * kStateCount);
//        memcpy(tmpEvec, inEigenVectors + i * kStateCount, sizeof(Real) * kStateCount);
//#else
//        MEMCNV(tmpIevc, (inInverseEigenVectors + i * kStateCount), kStateCount, Real);
//        MEMCNV(tmpEvec, (inEigenVectors + i * kStateCount), kStateCount, Real);
//#endif
    	beagleMemCpy(tmpIevc, inInverseEigenVectors + i * kStateCount, kStateCount);
    	beagleMemCpy(tmpEvec, inEigenVectors + i * kStateCount, kStateCount);
        tmpIevc += kPaddedStateCount;
        tmpEvec += kPaddedStateCount;
    }
    
    // Transposing matrices avoids incoherent memory read/writes  
    // TODO: Only need to tranpose sub-matrix of trueStateCount
    if (kFlags & BEAGLE_FLAG_INVEVEC_STANDARD)
        transposeSquareMatrix(Ievc, kPaddedStateCount); 
    transposeSquareMatrix(Evec, kPaddedStateCount);
    
//#ifdef DOUBLE_PRECISION
//    memcpy(Eval, inEigenValues, sizeof(Real) * kStateCount);
//    if (kFlags & BEAGLE_FLAG_EIGEN_COMPLEX)
//    	memcpy(Eval+kPaddedStateCount,inEigenValues+kStateCount,sizeof(Real)*kStateCount);
//#else
//    MEMCNV(Eval, inEigenValues, kStateCount, Real);
//    if (kFlags & BEAGLE_FLAG_EIGEN_COMPLEX)
//    	MEMCNV((Eval+kPaddedStateCount),(inEigenValues+kStateCount),kStateCount,Real);
//#endif
    beagleMemCpy(Eval, inEigenValues, kStateCount);
    if (kFlags & BEAGLE_FLAG_EIGEN_COMPLEX) {
    	beagleMemCpy(Eval + kPaddedStateCount, inEigenValues + kStateCount, kStateCount);
    }
    
#ifdef BEAGLE_DEBUG_VALUES
//#ifdef DOUBLE_PRECISION
//    fprintf(stderr, "Eval:\n");
//    printfVectorD(Eval, kEigenValuesSize);
//    fprintf(stderr, "Evec:\n");
//    printfVectorD(Evec, kMatrixSize);
//    fprintf(stderr, "Ievc:\n");
//    printfVectorD(Ievc, kPaddedStateCount * kPaddedStateCount);
//#else
    fprintf(stderr, "Eval =\n");
    printfVector(Eval, kEigenValuesSize);
    fprintf(stderr, "Evec =\n");
    printfVector(Evec, kMatrixSize);
    fprintf(stderr, "Ievc =\n");
    printfVector(Ievc, kPaddedStateCount * kPaddedStateCount);   
//#endif
#endif
    
    // Copy to GPU device
    gpu->MemcpyHostToDevice(dIevc[eigenIndex], Ievc, sizeof(Real) * kMatrixSize);
    gpu->MemcpyHostToDevice(dEvec[eigenIndex], Evec, sizeof(Real) * kMatrixSize);
    gpu->MemcpyHostToDevice(dEigenValues[eigenIndex], Eval, sizeof(Real) * kEigenValuesSize);
    
#ifdef BEAGLE_DEBUG_VALUES
    Real r = 0;
    fprintf(stderr, "dEigenValues =\n");
    gpu->PrintfDeviceVector(dEigenValues[eigenIndex], kEigenValuesSize, r);
    fprintf(stderr, "dEvec =\n");
    gpu->PrintfDeviceVector(dEvec[eigenIndex], kMatrixSize, r);
    fprintf(stderr, "dIevc =\n");
    gpu->PrintfDeviceVector(dIevc[eigenIndex], kPaddedStateCount * kPaddedStateCount, r);
#endif
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::setEigenDecomposition\n");
#endif
    
    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::setStateFrequencies(int stateFrequenciesIndex,
                                       const double* inStateFrequencies) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::setStateFrequencies\n");
#endif
    
    if (stateFrequenciesIndex < 0 || stateFrequenciesIndex >= kEigenDecompCount)
        return BEAGLE_ERROR_OUT_OF_RANGE;
	
//#ifdef DOUBLE_PRECISION
//    memcpy(hFrequenciesCache, inStateFrequencies, kStateCount * sizeof(Real));
//#else
//    MEMCNV(hFrequenciesCache, inStateFrequencies, kStateCount, Real);
//#endif
    beagleMemCpy(hFrequenciesCache, inStateFrequencies, kStateCount);
	
	gpu->MemcpyHostToDevice(dFrequencies[stateFrequenciesIndex], hFrequenciesCache,
                            sizeof(Real) * kPaddedStateCount);
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::setStateFrequencies\n");
#endif
    
    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::setCategoryWeights(int categoryWeightsIndex,
                                      const double* inCategoryWeights) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::setCategoryWeights\n");
#endif
    
    if (categoryWeightsIndex < 0 || categoryWeightsIndex >= kEigenDecompCount)
        return BEAGLE_ERROR_OUT_OF_RANGE;
    
//#ifdef DOUBLE_PRECISION
//	const double* tmpWeights = inCategoryWeights;
//#else
//	Real* tmpWeights = hWeightsCache;
//	MEMCNV(hWeightsCache, inCategoryWeights, kCategoryCount, Real);
//#endif
    const Real* tmpWeights = beagleCastIfNecessary(inCategoryWeights, hWeightsCache,
    		kCategoryCount);
    
	gpu->MemcpyHostToDevice(dWeights[categoryWeightsIndex], tmpWeights,
                            sizeof(Real) * kCategoryCount);
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::setCategoryWeights\n");
#endif
    
    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::setCategoryRates(const double* inCategoryRates) {

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

BEAGLE_GPU_TEMPLATE
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::setPatternWeights(const double* inPatternWeights) {
    
#ifdef BEAGLE_DEBUG_FLOW
	fprintf(stderr, "\tEntering BeagleGPUImpl::setPatternWeights\n");
#endif
	
//#ifdef DOUBLE_PRECISION
//	const double* tmpWeights = inPatternWeights;
//#else
//	Real* tmpWeights = hPatternWeightsCache;
//	MEMCNV(hPatternWeightsCache, inPatternWeights, kPatternCount, Real);
//#endif
	const Real* tmpWeights = beagleCastIfNecessary(inPatternWeights, hPatternWeightsCache, kPatternCount);
	
	gpu->MemcpyHostToDevice(dPatternWeights, tmpWeights,
                            sizeof(Real) * kPatternCount);
    
#ifdef BEAGLE_DEBUG_FLOW
	fprintf(stderr, "\tLeaving  BeagleGPUImpl::setPatternWeights\n");
#endif
    
    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::getTransitionMatrix(int matrixIndex,
									   double* outMatrix) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::getTransitionMatrix\n");
#endif
        
    gpu->MemcpyDeviceToHost(hMatrixCache, dMatrices[matrixIndex], sizeof(Real) * kMatrixSize * kCategoryCount);
    
    double* outMatrixOffset = outMatrix;
    Real* tmpRealMatrixOffset = hMatrixCache;
    
    for (int l = 0; l < kCategoryCount; l++) {
        
        transposeSquareMatrix(tmpRealMatrixOffset, kPaddedStateCount);
        
        for (int i = 0; i < kStateCount; i++) {
//#ifdef DOUBLE_PRECISION
//            memcpy(outMatrixOffset, tmpRealMatrixOffset, sizeof(Real) * kStateCount);
//#else
//            MEMCNV(outMatrixOffset, tmpRealMatrixOffset, kStateCount, double);
//#endif
        	beagleMemCpy(outMatrixOffset, tmpRealMatrixOffset, kStateCount);
            tmpRealMatrixOffset += kPaddedStateCount;
            outMatrixOffset += kStateCount;
        }
        tmpRealMatrixOffset += (kPaddedStateCount - kStateCount) * kPaddedStateCount;
    }
        
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::getTransitionMatrix\n");
#endif
    
    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::setTransitionMatrix(int matrixIndex,
                                       const double* inMatrix,
                                       double paddedValue) {
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::setTransitionMatrix\n");
#endif
    
    const double* inMatrixOffset = inMatrix;
    Real* tmpRealMatrixOffset = hMatrixCache;
    
    for (int l = 0; l < kCategoryCount; l++) {
        Real* transposeOffset = tmpRealMatrixOffset;
        
        for (int i = 0; i < kStateCount; i++) {
//#ifdef DOUBLE_PRECISION
//            memcpy(tmpRealMatrixOffset, inMatrixOffset, sizeof(Real) * kStateCount);
//#else
//            MEMCNV(tmpRealMatrixOffset, inMatrixOffset, kStateCount, Real);
//#endif
        	beagleMemCpy(tmpRealMatrixOffset, inMatrixOffset, kStateCount);
            tmpRealMatrixOffset += kPaddedStateCount;
            inMatrixOffset += kStateCount;
        }
        
        transposeSquareMatrix(transposeOffset, kPaddedStateCount);
        tmpRealMatrixOffset += (kPaddedStateCount - kStateCount) * kPaddedStateCount;
    }
        
    // Copy to GPU device
    gpu->MemcpyHostToDevice(dMatrices[matrixIndex], hMatrixCache,
                            sizeof(Real) * kMatrixSize * kCategoryCount);
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::setTransitionMatrix\n");
#endif
    
    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::setTransitionMatrices(const int* matrixIndices,
                                         const double* inMatrices,
                                         const double* paddedValues,
                                         int count) {
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::setTransitionMatrices\n");
#endif
    
    int k = 0;
    while (k < count) {
        const double* inMatrixOffset = inMatrices + k*kStateCount*kStateCount*kCategoryCount;
        Real* tmpRealMatrixOffset = hMatrixCache;
        int lumpedMatricesCount = 0;
        int matrixIndex = matrixIndices[k];
                
        do {
            for (int l = 0; l < kCategoryCount; l++) {
                Real* transposeOffset = tmpRealMatrixOffset;
                
                for (int i = 0; i < kStateCount; i++) {
//        #ifdef DOUBLE_PRECISION
//                    memcpy(tmpRealMatrixOffset, inMatrixOffset, sizeof(Real) * kStateCount);
//        #else
//                    MEMCNV(tmpRealMatrixOffset, inMatrixOffset, kStateCount, Real);
//        #endif
                	beagleMemCpy(tmpRealMatrixOffset, inMatrixOffset, kStateCount);
                    tmpRealMatrixOffset += kPaddedStateCount;
                    inMatrixOffset += kStateCount;
                }
                
                transposeSquareMatrix(transposeOffset, kPaddedStateCount);
                tmpRealMatrixOffset += (kPaddedStateCount - kStateCount) * kPaddedStateCount;
            } 
                    
            lumpedMatricesCount++;
            k++;
        } while ((k < count) && (matrixIndices[k] == matrixIndices[k-1] + 1) && (lumpedMatricesCount < BEAGLE_CACHED_MATRICES_COUNT));
        
        // Copy to GPU device
        gpu->MemcpyHostToDevice(dMatrices[matrixIndex], hMatrixCache,
                                sizeof(Real) * kMatrixSize * kCategoryCount * lumpedMatricesCount);
        
    }   
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::setTransitionMatrices\n");
#endif
    
    return BEAGLE_SUCCESS;
}

///////////////////////////
//---TODO: Epoch model---//
///////////////////////////

BEAGLE_GPU_TEMPLATE
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::convolveTransitionMatrices(const int* firstIndices,
		const int* secondIndices,
		const int* resultIndices,
		int matrixCount) {

#ifdef BEAGLE_DEBUG_FLOW
	fprintf(stderr, "\t Entering BeagleGPUImpl::convolveTransitionMatrices \n");
#endif

	int returnCode = BEAGLE_SUCCESS;

	if (matrixCount > 0) {

		for(int u = 0; u < matrixCount; u++) {
			if(firstIndices[u] == resultIndices[u] || secondIndices[u] == resultIndices[u]) {

#ifdef BEAGLE_DEBUG_FLOW
				fprintf(stderr, "In-place convolution is not allowed \n");
#endif

				returnCode = BEAGLE_ERROR_GENERAL;
				break;

			}//END: overwrite check
		}//END: u loop

		int totalMatrixCount = matrixCount * kCategoryCount;

		int ptrIndex = 0;
		int indexOffset = kMatrixSize * kCategoryCount;
		int categoryOffset = kMatrixSize;

#ifdef CUDA

		for (int i = 0; i < matrixCount; i++) {
			for (int j = 0; j < kCategoryCount; j++) {

				hPtrQueue[ptrIndex] = firstIndices[i] * indexOffset + j * categoryOffset;
				hPtrQueue[ptrIndex + totalMatrixCount] = secondIndices[i] * indexOffset + j * categoryOffset;
				hPtrQueue[ptrIndex + totalMatrixCount*2] = resultIndices[i] * indexOffset + j * categoryOffset;

				ptrIndex++;

			}//END: kCategoryCount loop
		}//END: matrices count loop

		gpu->MemcpyHostToDevice(dPtrQueue, hPtrQueue, sizeof(unsigned int) * totalMatrixCount * 3);

		kernels->ConvolveTransitionMatrices(dMatrices[0], dPtrQueue, totalMatrixCount);

	}//END: count check

#endif// END: if CUDA
#ifdef BEAGLE_DEBUG_FLOW
	fprintf(stderr, "\t Leaving BeagleGPUImpl::convolveTransitionMatrices \n");
#endif

	return returnCode;
}//END: convolveTransitionMatrices


BEAGLE_GPU_TEMPLATE
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::updateTransitionMatrices(int eigenIndex,
                                            const int* probabilityIndices,
                                            const int* firstDerivativeIndices,
                                            const int* secondDerivativeIndices,
                                            const double* edgeLengths,
                                            int count) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\tEntering BeagleGPUImpl::updateTransitionMatrices\n");
#endif
    if (count > 0) {
        // TODO: improve performance of calculation of derivatives
        int totalCount = 0;
        
        int indexOffset =  kMatrixSize * kCategoryCount;
        int categoryOffset = kMatrixSize;
        
    #ifdef CUDA
        
        if (firstDerivativeIndices == NULL && secondDerivativeIndices == NULL) {
            for (int i = 0; i < count; i++) {        
                for (int j = 0; j < kCategoryCount; j++) {
                    hPtrQueue[totalCount] = probabilityIndices[i] * indexOffset + j * categoryOffset;
                    hDistanceQueue[totalCount] = (Real) (edgeLengths[i] * hCategoryRates[j]);
                    totalCount++;
                }
            }
            
            gpu->MemcpyHostToDevice(dPtrQueue, hPtrQueue, sizeof(unsigned int) * totalCount);
            gpu->MemcpyHostToDevice(dDistanceQueue, hDistanceQueue, sizeof(Real) * totalCount);
            
            // Set-up and call GPU kernel
            kernels->GetTransitionProbabilitiesSquare(dMatrices[0], dPtrQueue, dEvec[eigenIndex], dIevc[eigenIndex],
                                                      dEigenValues[eigenIndex], dDistanceQueue, totalCount);
        } else if (secondDerivativeIndices == NULL) {        
            
            totalCount = count * kCategoryCount;
            int ptrIndex = 0;
            for (int i = 0; i < count; i++) {        
                for (int j = 0; j < kCategoryCount; j++) {
                    hPtrQueue[ptrIndex] = probabilityIndices[i] * indexOffset + j * categoryOffset;
                    hPtrQueue[ptrIndex + totalCount] = firstDerivativeIndices[i] * indexOffset + j * categoryOffset;
                    hDistanceQueue[ptrIndex] = (Real) (edgeLengths[i]);
                    hDistanceQueue[ptrIndex + totalCount] = (Real) (hCategoryRates[j]);
                    ptrIndex++;
                }
            }
            
            gpu->MemcpyHostToDevice(dPtrQueue, hPtrQueue, sizeof(unsigned int) * totalCount * 2);
            gpu->MemcpyHostToDevice(dDistanceQueue, hDistanceQueue, sizeof(Real) * totalCount * 2);
            
            kernels->GetTransitionProbabilitiesSquareFirstDeriv(dMatrices[0], dPtrQueue, dEvec[eigenIndex], dIevc[eigenIndex],
                                                                 dEigenValues[eigenIndex], dDistanceQueue, totalCount);        
            
        } else {
            totalCount = count * kCategoryCount;
            int ptrIndex = 0;
            for (int i = 0; i < count; i++) {        
                for (int j = 0; j < kCategoryCount; j++) {
                    hPtrQueue[ptrIndex] = probabilityIndices[i] * indexOffset + j * categoryOffset;
                    hPtrQueue[ptrIndex + totalCount] = firstDerivativeIndices[i] * indexOffset + j * categoryOffset;
                    hPtrQueue[ptrIndex + totalCount*2] = secondDerivativeIndices[i] * indexOffset + j * categoryOffset;
                                    hDistanceQueue[ptrIndex] = (Real) (edgeLengths[i]);
                    hDistanceQueue[ptrIndex + totalCount] = (Real) (hCategoryRates[j]);
                    ptrIndex++;
                }
            }
            
            gpu->MemcpyHostToDevice(dPtrQueue, hPtrQueue, sizeof(unsigned int) * totalCount * 3);
            gpu->MemcpyHostToDevice(dDistanceQueue, hDistanceQueue, sizeof(Real) * totalCount * 2);
            
            kernels->GetTransitionProbabilitiesSquareSecondDeriv(dMatrices[0], dPtrQueue, dEvec[eigenIndex], dIevc[eigenIndex],
                                                      dEigenValues[eigenIndex], dDistanceQueue, totalCount);        
        }
        
        
    #else
        // TODO: update OpenCL implementation with derivs
        
        for (int i = 0; i < count; i++) {        
            for (int j = 0; j < kCategoryCount; j++) {
                hDistanceQueue[totalCount] = (Real) (edgeLengths[i] * hCategoryRates[j]);
                totalCount++;
            }
        }
        
        gpu->MemcpyHostToDevice(dDistanceQueue, hDistanceQueue, sizeof(Real) * totalCount);
        
        // Set-up and call GPU kernel
        for (int i = 0; i < count; i++) {        
            kernels->GetTransitionProbabilitiesSquare(dMatrices[probabilityIndices[i]], dEvec[eigenIndex], dIevc[eigenIndex], 
                                                      dEigenValues[eigenIndex], dDistanceQueue, kCategoryCount,
                                                      i * kCategoryCount);
        }
            
    #endif
        
    #ifdef BEAGLE_DEBUG_VALUES
        Real r = 0;
        for (int i = 0; i < 1; i++) {
            fprintf(stderr, "dMatrices[probabilityIndices[%d]]  (hDQ = %1.5e, eL = %1.5e) =\n", i,hDistanceQueue[i], edgeLengths[i]);        
            gpu->PrintfDeviceVector(dMatrices[probabilityIndices[i]], kMatrixSize * kCategoryCount, r);
            for(int j=0; j<kCategoryCount; j++)
                fprintf(stderr, " %1.5f",hCategoryRates[j]);
            fprintf(stderr,"\n");
        }
    #endif

    #ifdef BEAGLE_DEBUG_SYNCH    
        gpu->Synchronize();
    #endif
    }
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::updateTransitionMatrices\n");
#endif
    
    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::updatePartials(const int* operations,
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
        GPUPtr scalingFactors = (GPUPtr)NULL;
        
        if (kFlags & BEAGLE_FLAG_SCALING_AUTO) {
            int sIndex = parIndex - kTipCount;
            
            if (tipStates1 == 0 && tipStates2 == 0) {
                rescale = 2;
                scalingFactors = dScalingFactors[sIndex];
            }
        } else if (kFlags & BEAGLE_FLAG_SCALING_ALWAYS) {
            rescale = 1;
            scalingFactors = dScalingFactors[parIndex - kTipCount];
        } else if ((kFlags & BEAGLE_FLAG_SCALING_MANUAL) && writeScalingIndex >= 0) {
            rescale = 1;
            scalingFactors = dScalingFactors[writeScalingIndex];
        } else if ((kFlags & BEAGLE_FLAG_SCALING_MANUAL) && readScalingIndex >= 0) {
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
        Real r = 0;
        if (tipStates1)
            gpu->PrintfDeviceInt(tipStates1, kPaddedPatternCount);
        else
            gpu->PrintfDeviceVector(partials1, kPartialsSize, r);
        fprintf(stderr, "child2 = \n");
        if (tipStates2)
            gpu->PrintfDeviceInt(tipStates2, kPaddedPatternCount);
        else
            gpu->PrintfDeviceVector(partials2, kPartialsSize, r);
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
                if (kFlags & BEAGLE_FLAG_SCALING_DYNAMIC) {
                    kernels->PartialsPartialsPruningDynamicCheckScaling(partials1, partials2, partials3,
                                                                   matrices1, matrices2, writeScalingIndex, readScalingIndex,
                                                                   cumulativeScalingIndex, dScalingFactors, dScalingFactorsMaster,
                                                                   kPaddedPatternCount, kCategoryCount,
                                                                   rescale, hRescalingTrigger, dRescalingTrigger, sizeof(Real));
                } else {
                    kernels->PartialsPartialsPruningDynamicScaling(partials1, partials2, partials3,
                                                                   matrices1, matrices2, scalingFactors,
                                                                   cumulativeScalingBuffer, 
                                                                   kPaddedPatternCount, kCategoryCount,
                                                                   rescale);
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
        
#ifdef BEAGLE_DEBUG_VALUES
        if (rescale > -1) {
        	fprintf(stderr,"scalars = ");
        	gpu->PrintfDeviceVector(scalingFactors,kPaddedPatternCount, r);
        }
        fprintf(stderr, "parent = \n");
        int signal = 0;
        if (writeScalingIndex == -1)
        	gpu->PrintfDeviceVector(partials3, kPartialsSize, r);
        else
        	gpu->PrintfDeviceVector(partials3, kPartialsSize, 1.0, &signal, r);
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

BEAGLE_GPU_TEMPLATE
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::waitForPartials(const int* /*destinationPartials*/,
                                   int /*destinationPartialsCount*/) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::waitForPartials\n");
#endif
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::waitForPartials\n");
#endif    
    
    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::accumulateScaleFactors(const int* scalingIndices,
										  int count,
										  int cumulativeScalingIndex) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::accumulateScaleFactors\n");
#endif
    
    if (kFlags & BEAGLE_FLAG_SCALING_DYNAMIC) {
        if (dScalingFactors[cumulativeScalingIndex] != dScalingFactorsMaster[cumulativeScalingIndex]) {
            gpu->MemcpyDeviceToDevice(dScalingFactorsMaster[cumulativeScalingIndex], dScalingFactors[cumulativeScalingIndex], sizeof(Real)*kScaleBufferSize);
            gpu->Synchronize();
            dScalingFactors[cumulativeScalingIndex] = dScalingFactorsMaster[cumulativeScalingIndex];            
        }
    } 
    
    if (kFlags & BEAGLE_FLAG_SCALING_AUTO) {
        
        for(int n = 0; n < count; n++) {
            int sIndex = scalingIndices[n] - kTipCount;
            hPtrQueue[n] = sIndex;
        }
        
        gpu->MemcpyHostToDevice(dPtrQueue, hPtrQueue, sizeof(unsigned int) * count);
        
        kernels->AccumulateFactorsAutoScaling(dScalingFactors[0], dPtrQueue, dAccumulatedScalingFactors, count, kPaddedPatternCount, kScaleBufferSize);
                
    } else {        
        for(int n = 0; n < count; n++)
            hPtrQueue[n] = scalingIndices[n] * kScaleBufferSize;
        gpu->MemcpyHostToDevice(dPtrQueue, hPtrQueue, sizeof(unsigned int) * count);
        

    #ifdef CUDA    
        // Compute scaling factors at the root
        kernels->AccumulateFactorsDynamicScaling(dScalingFactors[0], dPtrQueue, dScalingFactors[cumulativeScalingIndex], count, kPaddedPatternCount);
    #else // OpenCL
//        for (int i = 0; i < count; i++) {
//            kernels->AccumulateFactorsDynamicScaling(dScalingFactors[scalingIndices[i]], dScalingFactors[cumulativeScalingIndex],
//                                                     1, kPaddedPatternCount);
//        }
    #endif
    }
    
#ifdef BEAGLE_DEBUG_SYNCH    
    gpu->Synchronize();
#endif

#ifdef BEAGLE_DEBUG_VALUES
    Real r = 0;
    fprintf(stderr, "scaling factors = ");
    gpu->PrintfDeviceVector(dScalingFactors[cumulativeScalingIndex], kPaddedPatternCount, r);
#endif
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::accumulateScaleFactors\n");
#endif   
    
    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::removeScaleFactors(const int* scalingIndices,
                                        int count,
                                        int cumulativeScalingIndex) {
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::removeScaleFactors\n");
#endif
    
    if (kFlags & BEAGLE_FLAG_SCALING_DYNAMIC) {
        if (dScalingFactors[cumulativeScalingIndex] != dScalingFactorsMaster[cumulativeScalingIndex]) {
            gpu->MemcpyDeviceToDevice(dScalingFactorsMaster[cumulativeScalingIndex], dScalingFactors[cumulativeScalingIndex], sizeof(Real)*kScaleBufferSize);
            gpu->Synchronize();
            dScalingFactors[cumulativeScalingIndex] = dScalingFactorsMaster[cumulativeScalingIndex];
        }
    } 
    
    for(int n = 0; n < count; n++)
        hPtrQueue[n] = scalingIndices[n] * kScaleBufferSize;
    gpu->MemcpyHostToDevice(dPtrQueue, hPtrQueue, sizeof(unsigned int) * count);
    
#ifdef CUDA    
    // Compute scaling factors at the root
    kernels->RemoveFactorsDynamicScaling(dScalingFactors[0], dPtrQueue, dScalingFactors[cumulativeScalingIndex],
                                         count, kPaddedPatternCount);
#else // OpenCL
//    for (int i = 0; i < count; i++) {
//        kernels->RemoveFactorsDynamicScaling(dScalingFactors[scalingIndices[i]], dScalingFactors[cumulativeScalingIndex],
//                                             1, kPaddedPatternCount);
//    }    
#endif
    
#ifdef BEAGLE_DEBUG_SYNCH    
    gpu->Synchronize();
#endif

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::removeScaleFactors\n");
#endif        
    
    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::resetScaleFactors(int cumulativeScalingIndex) {
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::resetScaleFactors\n");
#endif

    if (kFlags & BEAGLE_FLAG_SCALING_DYNAMIC) {
        if (dScalingFactors[cumulativeScalingIndex] != dScalingFactorsMaster[cumulativeScalingIndex])
            dScalingFactors[cumulativeScalingIndex] = dScalingFactorsMaster[cumulativeScalingIndex];
        
        if (dScalingFactors[cumulativeScalingIndex] == 0) {
            dScalingFactors[cumulativeScalingIndex] = gpu->AllocateMemory(kScaleBufferSize * sizeof(Real));
            dScalingFactorsMaster[cumulativeScalingIndex] = dScalingFactors[cumulativeScalingIndex];
        }
    }
    
    Real* zeroes = (Real*) gpu->CallocHost(sizeof(Real), kPaddedPatternCount);
    
    // Fill with zeroes
    gpu->MemcpyHostToDevice(dScalingFactors[cumulativeScalingIndex], zeroes,
                            sizeof(Real) * kPaddedPatternCount);
    
    gpu->FreeHostMemory(zeroes);
    
#ifdef BEAGLE_DEBUG_SYNCH    
    gpu->Synchronize();
#endif
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::resetScaleFactors\n");
#endif    
    
    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::copyScaleFactors(int destScalingIndex,
                                    int srcScalingIndex) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::copyScaleFactors\n");
#endif

    if (kFlags & BEAGLE_FLAG_SCALING_DYNAMIC) {
        dScalingFactors[destScalingIndex] = dScalingFactors[srcScalingIndex];
    } else {
        gpu->MemcpyDeviceToDevice(dScalingFactors[destScalingIndex], dScalingFactors[srcScalingIndex], sizeof(Real)*kScaleBufferSize);
    }
#ifdef BEAGLE_DEBUG_SYNCH    
    gpu->Synchronize();
#endif
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::copyScaleFactors\n");
#endif   
    
    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::calculateRootLogLikelihoods(const int* bufferIndices,
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

#ifdef BEAGLE_DEBUG_VALUES
        Real r = 0;
        fprintf(stderr,"root partials = \n");
        gpu->PrintfDeviceVector(dPartials[rootNodeIndex], kPaddedPatternCount, r);
#endif

        if (scale) {
            kernels->IntegrateLikelihoodsDynamicScaling(dIntegrationTmp, dPartials[rootNodeIndex],
                                                        dWeights[categoryWeightsIndex],
                                                        dFrequencies[stateFrequenciesIndex],
                                                        dCumulativeScalingFactor,
                                                        kPaddedPatternCount,
                                                        kCategoryCount);
        } else {
            kernels->IntegrateLikelihoods(dIntegrationTmp, dPartials[rootNodeIndex],
                                          dWeights[categoryWeightsIndex],
                                          dFrequencies[stateFrequenciesIndex],
                                          kPaddedPatternCount, kCategoryCount);
        }
        
#ifdef BEAGLE_DEBUG_VALUES
        fprintf(stderr,"before SumSites1 = \n");
        gpu->PrintfDeviceVector(dIntegrationTmp, kPaddedPatternCount, r);
#endif

        kernels->SumSites1(dIntegrationTmp, dSumLogLikelihood, dPatternWeights,
                                    kPatternCount);

        gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dSumLogLikelihood, sizeof(Real) * kSumSitesBlockCount);

        *outSumLogLikelihood = 0.0;
        for (int i = 0; i < kSumSitesBlockCount; i++) {
            if (!(hLogLikelihoodsCache[i] - hLogLikelihoodsCache[i] == 0.0))
                returnCode = BEAGLE_ERROR_FLOATING_POINT;
            
            *outSumLogLikelihood += hLogLikelihoodsCache[i];
        }    
        
    } else {
		// TODO: evaluate performance, maybe break up kernels below for each subsetIndex case
		
        if (kFlags & BEAGLE_FLAG_SCALING_ALWAYS) {
			for(int n = 0; n < count; n++) {
                int cumulativeScalingFactor = bufferIndices[n] - kTipCount; 
				hPtrQueue[n] = cumulativeScalingFactor * kScaleBufferSize;
            }
			gpu->MemcpyHostToDevice(dPtrQueue, hPtrQueue, sizeof(unsigned int) * count);    
        } else if (cumulativeScaleIndices[0] != BEAGLE_OP_NONE) {
			for(int n = 0; n < count; n++)
				hPtrQueue[n] = cumulativeScaleIndices[n] * kScaleBufferSize;
			gpu->MemcpyHostToDevice(dPtrQueue, hPtrQueue, sizeof(unsigned int) * count);
		}
		
		for (int subsetIndex = 0 ; subsetIndex < count; ++subsetIndex ) {

			const GPUPtr tmpDWeights = dWeights[categoryWeightsIndices[subsetIndex]];
			const GPUPtr tmpDFrequencies = dFrequencies[stateFrequenciesIndices[subsetIndex]];
			const int rootNodeIndex = bufferIndices[subsetIndex];

			if (cumulativeScaleIndices[0] != BEAGLE_OP_NONE || (kFlags & BEAGLE_FLAG_SCALING_ALWAYS)) {
				kernels->IntegrateLikelihoodsFixedScaleMulti(dIntegrationTmp, dPartials[rootNodeIndex], tmpDWeights,
															 tmpDFrequencies, dScalingFactors[0], dPtrQueue, dMaxScalingFactors,
															 dIndexMaxScalingFactors,
                                                             kPaddedPatternCount,
															 kCategoryCount, count, subsetIndex);
			} else {
                if (subsetIndex == 0) {
					kernels->IntegrateLikelihoodsMulti(dIntegrationTmp, dPartials[rootNodeIndex], tmpDWeights,
													   tmpDFrequencies,
                                                       kPaddedPatternCount, kCategoryCount, 0);
				} else if (subsetIndex == count - 1) { 
					kernels->IntegrateLikelihoodsMulti(dIntegrationTmp, dPartials[rootNodeIndex], tmpDWeights,
													   tmpDFrequencies,
                                                       kPaddedPatternCount, kCategoryCount, 1);
				} else {
					kernels->IntegrateLikelihoodsMulti(dIntegrationTmp, dPartials[rootNodeIndex], tmpDWeights,
													   tmpDFrequencies,
                                                       kPaddedPatternCount, kCategoryCount, 2);
				}
			}
			

            kernels->SumSites1(dIntegrationTmp, dSumLogLikelihood, dPatternWeights,
                                        kPatternCount);
                        
            gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dSumLogLikelihood, sizeof(Real) * kSumSitesBlockCount);
            
            *outSumLogLikelihood = 0.0;
            for (int i = 0; i < kSumSitesBlockCount; i++) {
                if (!(hLogLikelihoodsCache[i] - hLogLikelihoodsCache[i] == 0.0))
                    returnCode = BEAGLE_ERROR_FLOATING_POINT;
                
                *outSumLogLikelihood += hLogLikelihoodsCache[i];
            }    
		}
    }
    
#ifdef BEAGLE_DEBUG_VALUES
    Real r = 0;
    fprintf(stderr, "parent = \n");
    gpu->PrintfDeviceVector(dIntegrationTmp, kPatternCount, r);
#endif
    
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::calculateRootLogLikelihoods\n");
#endif
    
    return returnCode;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::calculateEdgeLogLikelihoods(const int* parentBufferIndices,
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
                                                            kPaddedPatternCount, kCategoryCount);
            } else {
                kernels->IntegrateLikelihoods(dIntegrationTmp, dPartialsTmp, dWeights[categoryWeightsIndex],
                                              dFrequencies[stateFrequenciesIndex],
                                              kPaddedPatternCount, kCategoryCount);
            }
            
            kernels->SumSites1(dIntegrationTmp, dSumLogLikelihood, dPatternWeights,
                                        kPatternCount);
            
            gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dSumLogLikelihood, sizeof(Real) * kSumSitesBlockCount);
            
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
                                                                       kPaddedPatternCount, kCategoryCount);
            } else {
                kernels->IntegrateLikelihoodsSecondDeriv(dIntegrationTmp, dOutFirstDeriv, dOutSecondDeriv,
                                                         dPartialsTmp, dFirstDerivTmp, dSecondDerivTmp,
                                                         dWeights[categoryWeightsIndex],
                                                         dFrequencies[stateFrequenciesIndex],
                                                         kPaddedPatternCount, kCategoryCount);
            }
            

            kernels->SumSites2(dIntegrationTmp, dSumLogLikelihood, dOutFirstDeriv, dSumFirstDeriv, dPatternWeights,
                                        kPatternCount);
            
            gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dSumLogLikelihood, sizeof(Real) * kSumSitesBlockCount);
            
            *outSumLogLikelihood = 0.0;
            for (int i = 0; i < kSumSitesBlockCount; i++) {
                if (!(hLogLikelihoodsCache[i] - hLogLikelihoodsCache[i] == 0.0))
                    returnCode = BEAGLE_ERROR_FLOATING_POINT;
                
                *outSumLogLikelihood += hLogLikelihoodsCache[i];
            }    
            
            gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dSumFirstDeriv, sizeof(Real) * kSumSitesBlockCount);
            
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
                                                                       kPaddedPatternCount, kCategoryCount);
            } else {
                kernels->IntegrateLikelihoodsSecondDeriv(dIntegrationTmp, dOutFirstDeriv, dOutSecondDeriv,
                                                         dPartialsTmp, dFirstDerivTmp, dSecondDerivTmp,
                                                         dWeights[categoryWeightsIndex],
                                                         dFrequencies[stateFrequenciesIndex],
                                                         kPaddedPatternCount, kCategoryCount);
            }
            
            kernels->SumSites3(dIntegrationTmp, dSumLogLikelihood, dOutFirstDeriv, dSumFirstDeriv, dOutSecondDeriv, dSumSecondDeriv, dPatternWeights,
                              kPatternCount);
            
            gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dSumLogLikelihood, sizeof(Real) * kSumSitesBlockCount);
            
            *outSumLogLikelihood = 0.0;
            for (int i = 0; i < kSumSitesBlockCount; i++) {
                if (!(hLogLikelihoodsCache[i] - hLogLikelihoodsCache[i] == 0.0))
                    returnCode = BEAGLE_ERROR_FLOATING_POINT;
                
                *outSumLogLikelihood += hLogLikelihoodsCache[i];
            }    
            
            gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dSumFirstDeriv, sizeof(Real) * kSumSitesBlockCount);
            
            *outSumFirstDerivative = 0.0;
            for (int i = 0; i < kSumSitesBlockCount; i++) {
                *outSumFirstDerivative += hLogLikelihoodsCache[i];
            }   

            gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dSumSecondDeriv, sizeof(Real) * kSumSitesBlockCount);
            
            *outSumSecondDerivative = 0.0;
            for (int i = 0; i < kSumSitesBlockCount; i++) {
                *outSumSecondDerivative += hLogLikelihoodsCache[i];
            }   
        }
        
        
    } else { //count > 1
        if (firstDerivativeIndices == NULL && secondDerivativeIndices == NULL) {

            if (kFlags & BEAGLE_FLAG_SCALING_ALWAYS) {
                fprintf(stderr,"BeagleGPUImpl::calculateEdgeLogLikelihoods not yet implemented for count > 1 and SCALING_ALWAYS\n");
            } else if (cumulativeScaleIndices[0] != BEAGLE_OP_NONE) {
                for(int n = 0; n < count; n++)
                    hPtrQueue[n] = cumulativeScaleIndices[n] * kScaleBufferSize;
                gpu->MemcpyHostToDevice(dPtrQueue, hPtrQueue, sizeof(unsigned int) * count);
            }
            
            for (int subsetIndex = 0 ; subsetIndex < count; ++subsetIndex ) {
                
                const int parIndex = parentBufferIndices[subsetIndex];
                const int childIndex = childBufferIndices[subsetIndex];
                const int probIndex = probabilityIndices[subsetIndex];                
                                
                GPUPtr partialsParent = dPartials[parIndex];
                GPUPtr partialsChild = dPartials[childIndex];        
                GPUPtr statesChild = dStates[childIndex];
                GPUPtr transMatrix = dMatrices[probIndex];
                
                const GPUPtr tmpDWeights = dWeights[categoryWeightsIndices[subsetIndex]];
                const GPUPtr tmpDFrequencies = dFrequencies[stateFrequenciesIndices[subsetIndex]];
                
                if (statesChild != 0) {
                    kernels->StatesPartialsEdgeLikelihoods(dPartialsTmp, partialsParent, statesChild,
                                                           transMatrix, kPaddedPatternCount,
                                                           kCategoryCount);
                } else {
                    kernels->PartialsPartialsEdgeLikelihoods(dPartialsTmp, partialsParent, partialsChild,
                                                             transMatrix, kPaddedPatternCount,
                                                             kCategoryCount);
                }
                
                if (cumulativeScaleIndices[0] != BEAGLE_OP_NONE) {
                    kernels->IntegrateLikelihoodsFixedScaleMulti(dIntegrationTmp, dPartialsTmp, tmpDWeights,
                                                                 tmpDFrequencies, dScalingFactors[0], dPtrQueue, dMaxScalingFactors,
                                                                 dIndexMaxScalingFactors,
                                                                 kPaddedPatternCount,
                                                                 kCategoryCount, count, subsetIndex);
                } else {
                    if (subsetIndex == 0) {
                        kernels->IntegrateLikelihoodsMulti(dIntegrationTmp, dPartialsTmp, tmpDWeights,
                                                           tmpDFrequencies,
                                                           kPaddedPatternCount, kCategoryCount, 0);
                    } else if (subsetIndex == count - 1) { 
                        kernels->IntegrateLikelihoodsMulti(dIntegrationTmp, dPartialsTmp, tmpDWeights,
                                                           tmpDFrequencies,
                                                           kPaddedPatternCount, kCategoryCount, 1);
                    } else {
                        kernels->IntegrateLikelihoodsMulti(dIntegrationTmp, dPartialsTmp, tmpDWeights,
                                                           tmpDFrequencies,
                                                           kPaddedPatternCount, kCategoryCount, 2);
                    }
                }
                
                kernels->SumSites1(dIntegrationTmp, dSumLogLikelihood, dPatternWeights,
                                   kPatternCount);
                
                gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dSumLogLikelihood, sizeof(Real) * kSumSitesBlockCount);
                
                *outSumLogLikelihood = 0.0;
                for (int i = 0; i < kSumSitesBlockCount; i++) {
                    if (!(hLogLikelihoodsCache[i] - hLogLikelihoodsCache[i] == 0.0))
                        returnCode = BEAGLE_ERROR_FLOATING_POINT;
                    
                    *outSumLogLikelihood += hLogLikelihoodsCache[i];
                }    
            }

		} else {
            fprintf(stderr,"BeagleGPUImpl::calculateEdgeLogLikelihoods not yet implemented for count > 1 and derivatives\n");
            returnCode = BEAGLE_ERROR_GENERAL;
        }
    }
    
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::calculateEdgeLogLikelihoods\n");
#endif
    
    return returnCode;
}

template<>
int BeagleGPUImpl<float>::getSiteLogLikelihoods(double* outLogLikelihoods) {

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::getSiteLogLikelihoods\n");
#endif

//#ifdef DOUBLE_PRECISION
//    gpu->MemcpyDeviceToHost(outLogLikelihoods, dIntegrationTmp, sizeof(Real) * kPatternCount);
//#else
    gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dIntegrationTmp, sizeof(float) * kPatternCount);
//    MEMCNV(outLogLikelihoods, hLogLikelihoodsCache, kPatternCount, double);
    beagleMemCpy(outLogLikelihoods, hLogLikelihoodsCache, kPatternCount);
//#endif

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::getSiteLogLikelihoods\n");
#endif

    return BEAGLE_SUCCESS;
}

template<>
int BeagleGPUImpl<double>::getSiteLogLikelihoods(double* outLogLikelihoods) {
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::getSiteLogLikelihoods\n");
#endif
    
//#ifdef DOUBLE_PRECISION
    gpu->MemcpyDeviceToHost(outLogLikelihoods, dIntegrationTmp, sizeof(double) * kPatternCount);
//#else
//    gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dIntegrationTmp, sizeof(Real) * kPatternCount);
//    MEMCNV(outLogLikelihoods, hLogLikelihoodsCache, kPatternCount, double);
//#endif
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::getSiteLogLikelihoods\n");
#endif    
    
    return BEAGLE_SUCCESS;
}

template<>
int BeagleGPUImpl<double>::getSiteDerivatives(double* outFirstDerivatives,
                                      double* outSecondDerivatives) {

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::getSiteDerivatives\n");
#endif

//#ifdef DOUBLE_PRECISION
    gpu->MemcpyDeviceToHost(outFirstDerivatives, dOutFirstDeriv, sizeof(double) * kPatternCount);
//#else
//    gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dOutFirstDeriv, sizeof(Real) * kPatternCount);
//    MEMCNV(outFirstDerivatives, hLogLikelihoodsCache, kPatternCount, double);
//#endif

    if (outSecondDerivatives != NULL) {
//#ifdef DOUBLE_PRECISION
        gpu->MemcpyDeviceToHost(outSecondDerivatives, dOutSecondDeriv, sizeof(double) * kPatternCount);
//#else
//        gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dOutSecondDeriv, sizeof(Real) * kPatternCount);
//        MEMCNV(outSecondDerivatives, hLogLikelihoodsCache, kPatternCount, double);
//#endif
    }


#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::getSiteDerivatives\n");
#endif

    return BEAGLE_SUCCESS;
}

template<>
int BeagleGPUImpl<float>::getSiteDerivatives(double* outFirstDerivatives,
                                      double* outSecondDerivatives) {
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::getSiteDerivatives\n");
#endif
    
//#ifdef DOUBLE_PRECISION
//    gpu->MemcpyDeviceToHost(outFirstDerivatives, dOutFirstDeriv, sizeof(Real) * kPatternCount);
//#else
    gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dOutFirstDeriv, sizeof(float) * kPatternCount);
    beagleMemCpy(outFirstDerivatives, hLogLikelihoodsCache, kPatternCount);
//    MEMCNV(outFirstDerivatives, hLogLikelihoodsCache, kPatternCount, double);
//#endif

    if (outSecondDerivatives != NULL) {
//#ifdef DOUBLE_PRECISION
//        gpu->MemcpyDeviceToHost(outSecondDerivatives, dOutSecondDeriv, sizeof(Real) * kPatternCount);
//#else
        gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dOutSecondDeriv, sizeof(float) * kPatternCount);
        beagleMemCpy(outSecondDerivatives, hLogLikelihoodsCache, kPatternCount);
//        MEMCNV(outSecondDerivatives, hLogLikelihoodsCache, kPatternCount, double);
//#endif
    }
    
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::getSiteDerivatives\n");
#endif    
    
    return BEAGLE_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
// BeagleGPUImplFactory public methods

BEAGLE_GPU_TEMPLATE
BeagleImpl*  BeagleGPUImplFactory<BEAGLE_GPU_GENERIC>::createImpl(int tipCount,
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
    BeagleImpl* impl = new BeagleGPUImpl<BEAGLE_GPU_GENERIC>();
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

template<>
const char* BeagleGPUImplFactory<double>::getName() {
	return "GPU-DP";
}

template<>
const char* BeagleGPUImplFactory<float>::getName() {
	return "GPU-SP";
}

template<>
void modifyFlagsForPrecision(long *flags, double r) {
	*flags |= BEAGLE_FLAG_PRECISION_DOUBLE;
}

template<>
void modifyFlagsForPrecision(long *flags, float r) {
	*flags |= BEAGLE_FLAG_PRECISION_SINGLE;
}

BEAGLE_GPU_TEMPLATE
const long BeagleGPUImplFactory<BEAGLE_GPU_GENERIC>::getFlags() {
	long flags = BEAGLE_FLAG_COMPUTATION_SYNCH |
          BEAGLE_FLAG_SCALING_MANUAL | BEAGLE_FLAG_SCALING_ALWAYS | BEAGLE_FLAG_SCALING_AUTO | BEAGLE_FLAG_SCALING_DYNAMIC |
          BEAGLE_FLAG_THREADING_NONE |
          BEAGLE_FLAG_VECTOR_NONE |
          BEAGLE_FLAG_PROCESSOR_GPU |
          BEAGLE_FLAG_SCALERS_LOG | BEAGLE_FLAG_SCALERS_RAW |
          BEAGLE_FLAG_EIGEN_COMPLEX | BEAGLE_FLAG_EIGEN_REAL |
          BEAGLE_FLAG_INVEVEC_STANDARD | BEAGLE_FLAG_INVEVEC_TRANSPOSED;

	Real r = 0;
	modifyFlagsForPrecision(&flags, r);
	return flags;
}

} // end of gpu namespace
} // end of beagle namespace
