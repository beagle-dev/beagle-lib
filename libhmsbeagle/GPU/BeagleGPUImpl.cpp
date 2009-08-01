
/*
 *  BeagleGPUImpl.cpp
 *  BEAGLE
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

int currentDevice = -1;

BeagleGPUImpl::~BeagleGPUImpl() {
    
    // free memory
    delete kernels;
    delete gpu;
    
    free(dPartials);
    free(dTipPartialsBuffers);
    free(dStates);
    free(dCompactBuffers);
    free(dMatrices);
    
#ifdef DYNAMIC_SCALING
    free(dScalingFactors);
#endif
    
    free(hDistanceQueue);
    free(hPtrQueue);
    
    free(hWeightsCache);
    free(hFrequenciesCache);
    free(hPartialsCache);
    free(hStatesCache);
    free(hMatrixCache);
    free(hLogLikelihoodsCache);
}

int BeagleGPUImpl::createInstance(int tipCount,
                                  int partialsBufferCount,
                                  int compactBufferCount,
                                  int stateCount,
                                  int patternCount,
                                  int eigenDecompositionCount,
                                  int matrixCount,
                                  int categoryCount) {
    
    // TODO: Determine if GPU device satisfies memory requirements.
    
    // TODO: add support for eigenDecompositionCount > 1
    
    kTipCount = tipCount;
    kPartialsBufferCount = partialsBufferCount;
    kCompactBufferCount = compactBufferCount;
    kStateCount = stateCount;
    kPatternCount = patternCount;
    kEigenDecompCount = eigenDecompositionCount;
    kMatrixCount = matrixCount;
    kCategoryCount = categoryCount;
    
    kTipPartialsBufferCount = kTipCount - kCompactBufferCount;
    kBufferCount = kPartialsBufferCount + kCompactBufferCount;
    
    if (kStateCount <= 4)
        kPaddedStateCount = 4;
    else if (kStateCount <= 16)
        kPaddedStateCount = 16;
    else if (kStateCount <= 32)
        kPaddedStateCount = 32;
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
    
    kPaddedPatternCount = kPatternCount + paddedPatterns;
    
#ifdef DEBUG
    fprintf(stderr, "Padding patterns for 4-state model:\n");
    fprintf(stderr, "\ttruePatternCount = %d\n\tpaddedPatterns = %d\n", kPatternCount,
            paddedPatterns);
#endif // DEBUG
    
    kPartialsSize = kPaddedPatternCount * kPaddedStateCount * kCategoryCount;
    kMatrixSize = kPaddedStateCount * kPaddedStateCount;
    kEigenValuesSize = kPaddedStateCount;
    
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
    
    hTmpTipPartials = (REAL**) calloc(sizeof(REAL*), kTipCount);
    hTmpStates = (int**) calloc(sizeof(int*), kTipCount);
    
    kDoRescaling = 1;
    
    kLastCompactBufferIndex = -1;
    kLastTipPartialsBufferIndex = -1;
    
    return NO_ERROR;
}

int BeagleGPUImpl::initializeInstance(InstanceDetails* returnInfo) {
    
    // TODO: compute device memory requirements
    
#ifdef DEBUG_FLOW
    fprintf(stderr, "Entering initialize\n");
#endif
    
    gpu = new GPUInterface();
    
    int numDevices = 0;
    numDevices = gpu->GetDeviceCount();
    if (numDevices == 0) {
        fprintf(stderr, "Error: No GPU devices\n");
        return GENERAL_ERROR;
    }
    
    currentDevice++;
    if (currentDevice == numDevices)
        currentDevice = 0;
    
    // TODO: recompiling kernels for every instance, probably not ideal
    gpu->SetDevice(currentDevice);
    
    kernels = new KernelLauncher(gpu);
    
    gpu->PrintInfo();
    
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
    
#ifdef DYNAMIC_SCALING
    dScalingFactors = (GPUPtr*) malloc(sizeof(GPUPtr) * kBufferCount);
    dRootScalingFactors = gpu->AllocateRealMemory(kPaddedPatternCount);
#endif
    
    for (int i = 0; i < kBufferCount; i++) {        
        if (i < kTipCount) { // For the tips
            if (i < kCompactBufferCount)
                dCompactBuffers[i] = gpu->AllocateIntMemory(kPaddedPatternCount);
            if (i < kTipPartialsBufferCount)
                dTipPartialsBuffers[i] = gpu->AllocateRealMemory(kPartialsSize);
        } else {
            dPartials[i] = gpu->AllocateRealMemory(kPartialsSize);
#ifdef DYNAMIC_SCALING
            dScalingFactors[i] = gpu->AllocateRealMemory(kPaddedPatternCount);
#endif
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
    
    // loadTipPartialsAndStates
    for (int i = 0; i < kTipCount; i++) {
        if (hTmpTipPartials[i] != 0) {
            assert(kLastTipPartialsBufferIndex >= 0 && kLastTipPartialsBufferIndex < 
                   kTipPartialsBufferCount);
            dPartials[i] = dTipPartialsBuffers[kLastTipPartialsBufferIndex--];
            gpu->MemcpyHostToDevice(dPartials[i], hTmpTipPartials[i], SIZE_REAL * kPartialsSize);
        } else if (hTmpStates[i] != 0) {
            assert(kLastCompactBufferIndex >= 0 && kLastCompactBufferIndex < kCompactBufferCount);
            dStates[i] = dCompactBuffers[kLastCompactBufferIndex--];
            gpu->MemcpyHostToDevice(dStates[i], hTmpStates[i], SIZE_INT * kPaddedPatternCount);
        }
    }
    
    // freeTmpTipPartialsAndStates
    for (int i = 0; i < kTipCount; i++) {
        if (hTmpTipPartials[i] != 0)
            free(hTmpTipPartials[i]);
        else if (hTmpStates[i] != 0)
            free(hTmpStates[i]);
    }
    free(hTmpTipPartials);
    free(hTmpStates);
    
    kDeviceMemoryAllocated = 1;
    
#ifdef DEBUG_FLOW
    fprintf(stderr, "Exiting initialize\n");
#endif
    
    return NO_ERROR;
}

int BeagleGPUImpl::setPartials(int bufferIndex,
                               const double* inPartials) {
#ifdef DEBUG_FLOW
    fprintf(stderr, "Entering setPartials\n");
#endif
    
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
    
    if (kDeviceMemoryAllocated) {
        if (bufferIndex < kTipCount) {
            assert(kLastTipPartialsBufferIndex >= 0 && kLastTipPartialsBufferIndex <
                   kTipPartialsBufferCount);
            dPartials[bufferIndex] = dTipPartialsBuffers[kLastTipPartialsBufferIndex--];
        }
        // Copy to GPU device
        gpu->MemcpyHostToDevice(dPartials[bufferIndex], hPartialsCache, SIZE_REAL * kPartialsSize);
    } else {
        hTmpTipPartials[bufferIndex] = (REAL*) malloc(SIZE_REAL * kPartialsSize);
        checkHostMemory(hTmpTipPartials[bufferIndex]);
        memcpy(hTmpTipPartials[bufferIndex], hPartialsCache, SIZE_REAL * kPartialsSize);
    }
    
#ifdef DEBUG_FLOW
    fprintf(stderr, "Exiting setPartials\n");
#endif
    
    return NO_ERROR;
}

int BeagleGPUImpl::getPartials(int bufferIndex,
                               double* outPartials) {
#ifdef DEBUG_FLOW
    fprintf(stderr, "Entering getPartials\n");
#endif
    
    // TODO: test getPartials
    
    if (kDeviceMemoryAllocated) {
        gpu->MemcpyDeviceToHost(hPartialsCache, dPartials[bufferIndex], SIZE_REAL * kPartialsSize);
    } else {
        memcpy(hPartialsCache, hTmpTipPartials[bufferIndex], SIZE_REAL * kPartialsSize);
    }
    
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
    
#ifdef DEBUG_FLOW
    fprintf(stderr, "Exiting getPartials\n");
#endif
    
    return NO_ERROR;
}

int BeagleGPUImpl::setTipStates(int tipIndex,
                                const int* inStates) {
    // TODO: test setTipStates
    
#ifdef DEBUG_FLOW
    fprintf(stderr, "Entering setTipStates\n");
#endif
    
    for(int i = 0; i < kPatternCount; i++)
        hStatesCache[i] = (inStates[i] < kStateCount ? inStates[i] : kPaddedStateCount);
    
    // Padded extra patterns
    for(int i = kPatternCount; i < kPaddedPatternCount; i++)
        hStatesCache[i] = kPaddedStateCount;
    
    if (kDeviceMemoryAllocated) {
        assert(kLastCompactBufferIndex >= 0 && kLastCompactBufferIndex < kCompactBufferCount);
        dStates[tipIndex] = dCompactBuffers[kLastCompactBufferIndex--];
        // Copy to GPU device
        gpu->MemcpyHostToDevice(dStates[tipIndex], hStatesCache, SIZE_INT * kPaddedPatternCount);
    } else {
        hTmpStates[tipIndex] = (int*) malloc(SIZE_INT * kPaddedPatternCount);
        checkHostMemory(hTmpStates[tipIndex]);
        memcpy(hTmpStates[tipIndex], hStatesCache, SIZE_INT * kPaddedPatternCount);
    }
    
#ifdef DEBUG_FLOW
    fprintf(stderr, "Exiting setTipStates\n");
#endif
    
    return NO_ERROR;
}

int BeagleGPUImpl::setEigenDecomposition(int eigenIndex,
                                         const double* inEigenVectors,
                                         const double* inInverseEigenVectors,
                                         const double* inEigenValues) {
    
#ifdef DEBUG_FLOW
    fprintf(stderr,"Entering updateEigenDecomposition\n");
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
    transposeSquareMatrix(Ievc, kPaddedStateCount);
    
    // TODO: Only need to tranpose sub-matrix of trueStateCount
    transposeSquareMatrix(Evec, kPaddedStateCount);
    
#ifdef DOUBLE_PRECISION
    memcpy(Eval, inEigenValues, SIZE_REAL * STATE_COUNT);
#else
    MEMCNV(Eval, inEigenValues, STATE_COUNT, REAL);
#endif
    
#ifdef DEBUG_BEAGLE
#ifdef DOUBLE_PRECISION
    printfVectorD(Eval, kPaddedStateCount);
    printfVectorD(Evec, kMatrixSize);
    printfVectorD(Ievc, kPaddedStateCount * kPaddedStateCount);
#else
    printfVectorF(Eval, kPaddedStateCount);
    printfVectorF(Evec, kMatrixSize);
    printfVectorF(Ievc, kPaddedStateCount * kPaddedStateCount);
#endif
#endif
    
    // Copy to GPU device
    gpu->MemcpyHostToDevice(dIevc[eigenIndex], Ievc, SIZE_REAL * kMatrixSize);
    gpu->MemcpyHostToDevice(dEvec[eigenIndex], Evec, SIZE_REAL * kMatrixSize);
    gpu->MemcpyHostToDevice(dEigenValues[eigenIndex], Eval, SIZE_REAL * kPaddedStateCount);
    
#ifdef DEBUG_BEAGLE
    gpu->PrintfDeviceVector(dEigenValues[eigenIndex], kPaddedStateCount);
    gpu->PrintfDeviceVector(dEvec[eigenIndex], kMatrixSize);
    gpu->PrintfDeviceVector(dIevc[eigenIndex], kPaddedStateCount * kPaddedStateCount);
#endif
    
#ifdef DEBUG_FLOW
    fprintf(stderr, "Exiting updateEigenDecomposition\n");
#endif
    
    return NO_ERROR;
}

int BeagleGPUImpl::setCategoryRates(const double* inCategoryRates) {
    // TODO: test setCategoryRates
#ifdef DEBUG_FLOW
	fprintf(stderr, "Entering updateCategoryRates\n");
#endif

#ifdef DOUBLE_PRECISION
	double* categoryRates = inCategoryRates;
#else
	REAL* categoryRates = hCategoryCache;
	MEMCNV(categoryRates, inCategoryRates, kCategoryCount, REAL);
#endif
    
	memcpy(hCategoryRates, categoryRates, SIZE_REAL * kCategoryCount);
    
#ifdef DEBUG_FLOW
	fprintf(stderr, "Exiting updateCategoryRates\n");
#endif
    
    return NO_ERROR;
}

int BeagleGPUImpl::setTransitionMatrix(int matrixIndex,
                                       const double* inMatrix) {
    // TODO: test setTransitionMatrix
    
#ifdef DEBUG_FLOW
    fprintf(stderr, "Entering setTransitionMatrix\n");
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
    
#ifdef DEBUG_FLOW
    fprintf(stderr, "Exiting setTransitionMatrix\n");
#endif
    
    return NO_ERROR;
}

int BeagleGPUImpl::updateTransitionMatrices(int eigenIndex,
                                            const int* probabilityIndices,
                                            const int* firstDerivativeIndices,
                                            const int* secondDervativeIndices,
                                            const double* edgeLengths,
                                            int count) {
#ifdef DEBUG_FLOW
    fprintf(stderr,"Entering updateMatrices\n");
#endif
    
    // TODO: implement calculation of derivatives
    
    int totalCount = 0;
    for (int i = 0; i < count; i++) {        
		for (int j = 0; j < kCategoryCount; j++) {
            hPtrQueue[totalCount] = dMatrices[probabilityIndices[i]] +
                    (j * kMatrixSize * sizeof(GPUPtr));
            hDistanceQueue[totalCount] = ((REAL) edgeLengths[i]) * hCategoryRates[j];
            totalCount++;
        }
    }
    
    gpu->MemcpyHostToDevice(dPtrQueue, hPtrQueue, sizeof(GPUPtr) * totalCount);
    gpu->MemcpyHostToDevice(dDistanceQueue, hDistanceQueue, SIZE_REAL * totalCount);
    
    // Set-up and call GPU kernel
    kernels->GetTransitionProbabilitiesSquare(dPtrQueue, dEvec[eigenIndex], dIevc[eigenIndex], 
											  dEigenValues[eigenIndex], dDistanceQueue,totalCount);
    
#ifdef DEBUG_BEAGLE
    gpu->PrintfDeviceVector(hPtrQueue[0], kMatrixSize * kCategoryCount);
#endif
    
#ifdef DEBUG_FLOW
    fprintf(stderr, "Exiting updateMatrices\n");
#endif
    
    return NO_ERROR;
}

int BeagleGPUImpl::updatePartials(const int* operations,
                                  int operationCount,
                                  int rescale) {
    
#ifdef DEBUG_FLOW
    fprintf(stderr, "Entering updatePartials\n");
#endif
    
#ifdef DYNAMIC_SCALING
    if (kDoRescaling == 0) // Forces rescaling on first computation
        kDoRescaling = rescale;
#endif
    
    // Serial version
    for (int op = 0; op < operationCount; op++) {
        const int parIndex = operations[op * 6];
        const int scalingIndex = operations[op * 6 + 1];
        const int child1Index = operations[op * 6 + 2];
        const int child1TransMatIndex = operations[op * 6 + 3];
        const int child2Index = operations[op * 6 + 4];
        const int child2TransMatIndex = operations[op * 6 + 5];
        
        GPUPtr matrices1 = dMatrices[child1TransMatIndex];
        GPUPtr matrices2 = dMatrices[child2TransMatIndex];
        
        GPUPtr partials1 = dPartials[child1Index];
        GPUPtr partials2 = dPartials[child2Index];
        
        GPUPtr partials3 = dPartials[parIndex];
        
        GPUPtr tipStates1 = dStates[child1Index];
        GPUPtr tipStates2 = dStates[child2Index];
        
#ifdef DYNAMIC_SCALING
        GPUPtr scalingFactors = dScalingFactors[scalingIndex];
        
        if (tipStates1 != 0) {
            if (tipStates2 != 0 ) {
                kernels->StatesStatesPruningDynamicScaling(tipStates1, tipStates2, partials3,
                                                           matrices1, matrices2, scalingFactors,
                                                           kPaddedPatternCount, kCategoryCount,
                                                           kDoRescaling);
            } else {
                kernels->StatesPartialsPruningDynamicScaling(tipStates1, partials2, partials3,
                                                             matrices1, matrices2, scalingFactors,
                                                             kPaddedPatternCount, kCategoryCount,
                                                             kDoRescaling);
            }
        } else {
            if (tipStates2 != 0) {
                kernels->StatesPartialsPruningDynamicScaling(tipStates2, partials1, partials3,
                                                             matrices2, matrices1, scalingFactors,
                                                             kPaddedPatternCount, kCategoryCount,
                                                             kDoRescaling);
            } else {
                kernels->PartialsPartialsPruningDynamicScaling(partials1, partials2, partials3,
                                                               matrices1, matrices2, scalingFactors,
                                                               kPaddedPatternCount, kCategoryCount,
                                                               kDoRescaling);
            }
        }
#else
        if (tipStates1 != 0) {
            if (tipStates2 != 0 ) {
                kernels->StatesStatesPruning(tipStates1, tipStates2, partials3, matrices1,
                                             matrices2, kPaddedPatternCount, kCategoryCount);
            } else {
                kernels->StatesPartialsPruning(tipStates1, partials2, partials3, matrices1,
                                               matrices2, kPaddedPatternCount, kCategoryCount);
            }
        } else {
            if (tipStates2 != 0) {
                kernels->StatesPartialsPruning(tipStates2, partials1, partials3, matrices2,
                                               matrices1, kPaddedPatternCount, kCategoryCount);
            } else {
                kernels->PartialsPartialsPruning(partials1, partials2, partials3, matrices1,
                                                 matrices2, kPaddedPatternCount, kCategoryCount);
            }
        }
#endif // DYNAMIC_SCALING
        
#ifdef DEBUG_BEAGLE
        fprintf(stderr, "kPaddedPatternCount = %d\n", kPaddedPatternCount);
        fprintf(stderr, "kPatternCount = %d\n", kPatternCount);
        fprintf(stderr, "categoryCount  = %d\n", kCategoryCount);
        fprintf(stderr, "partialSize = %d\n", kPartialsSize);
        if (tipStates1)
            gpu->PrintfDeviceInt(tipStates1, kPaddedPatternCount);
        else
            gpu->PrintfDeviceVector(partials1, kPartialsSize);
        if (tipStates2)
            gpu->PrintfDeviceInt(tipStates2, kPaddedPatternCount);
        else
            gpu->PrintfDeviceVector(partials2, kPartialsSize);
        fprintf(stderr, "node index = %d\n", parIndex);
        gpu->PrintfDeviceVector(partials3, kPartialsSize);
#endif
    }
    
#ifdef DEBUG_FLOW
    fprintf(stderr, "Exiting updatePartials\n");
#endif
    
    return NO_ERROR;
}

int BeagleGPUImpl::waitForPartials(const int* destinationPartials,
                                   int destinationPartialsCount) {
    return NO_ERROR;
}

int BeagleGPUImpl::calculateRootLogLikelihoods(const int* bufferIndices,
                                               const double* inWeights,
                                               const double* inStateFrequencies,
                                               const int* scalingFactorsIndices,
                                               int* scalingFactorsCount,
                                               int count,
                                               double* outLogLikelihoods) {
    if (count == 1) { 
        
#ifdef DEBUG_FLOW
        fprintf(stderr, "Entering calculateLogLikelihoods\n");
#endif
        
        REAL* tmpWeights = hWeightsCache;
        REAL* tmpStateFrequencies = hFrequenciesCache;
        
#ifdef DOUBLE_PRECISION
        tmpWeights = inWeights;
        tmpStateFrequencies = inStateFrequencies;
#else
        MEMCNV(hWeightsCache, inWeights, count * kCategoryCount, REAL);
        MEMCNV(hFrequenciesCache, inStateFrequencies, kPaddedStateCount * kCategoryCount, REAL);
#endif        
        gpu->MemcpyHostToDevice(dWeights, tmpWeights, SIZE_REAL * count * kCategoryCount);
        gpu->MemcpyHostToDevice(dFrequencies, tmpStateFrequencies, SIZE_REAL * kPaddedStateCount
                                * kCategoryCount);
        
        const int rootNodeIndex = bufferIndices[0];
        
#ifdef DYNAMIC_SCALING
        if (kDoRescaling) {
            // Construct node-list for scalingFactors
            int length = scalingFactorsCount[0];
            for(int n = 0; n < length; n++)
                hPtrQueue[n] = dScalingFactors[scalingFactorsIndices[n]];
            
            gpu->MemcpyHostToDevice(dPtrQueue, hPtrQueue, sizeof(GPUPtr) * length);
            
            // Compute scaling factors at the root
            kernels->ComputeRootDynamicScaling(dPtrQueue, dRootScalingFactors, length,
                                               kPaddedPatternCount);
        }
        
        kDoRescaling = 0;
        
        kernels->IntegrateLikelihoodsDynamicScaling(dIntegrationTmp, dPartials[rootNodeIndex],
                                                    dWeights, dFrequencies,
                                                    dRootScalingFactors, kPaddedPatternCount,
                                                    kCategoryCount);
#else
        kernels->IntegrateLikelihoods(dIntegrationTmp, dPartials[rootNodeIndex], dWeights,
                                      dFrequencies, kPaddedPatternCount, kCategoryCount);
#endif // DYNAMIC_SCALING
        
#ifdef DOUBLE_PRECISION
        gpu->MemcpyDeviceToHost(outLogLikelihoods, dIntegrationTmp, SIZE_REAL * kPatternCount);
#else
        gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dIntegrationTmp, SIZE_REAL * kPatternCount);
        MEMCNV(outLogLikelihoods, hLogLikelihoodsCache, kPatternCount, double);
#endif
        
#ifdef DEBUG
        printf("logLike = ");
        printfVectorD(outLogLikelihoods, kPatternCount);
        exit(-1);
#endif
    } else {
        // TODO: implement calculate root lnL for count > 1
        assert(false);
    }
    
    
#ifdef DEBUG_FLOW
    fprintf(stderr, "Exiting calculateLogLikelihoods\n");
#endif
    
    return NO_ERROR;
}

int BeagleGPUImpl::calculateEdgeLogLikelihoods(const int* parentBufferIndices,
                                               const int* childBufferIndices,
                                               const int* probabilityIndices,
                                               const int* firstDerivativeIndices,
                                               const int* secondDerivativeIndices,
                                               const double* inWeights,
                                               const double* inStateFrequencies,
                                               const int* scalingFactorsIndices,
                                               int* scalingFactorsCount,
                                               int count,
                                               double* outLogLikelihoods,
                                               double* outFirstDerivatives,
                                               double* outSecondDerivatives) {
    
    if (count == 1) { 
        
#ifdef DEBUG_FLOW
        fprintf(stderr, "Entering calculateEdgeLogLikelihoods\n");
#endif
        
        REAL* tmpWeights = hWeightsCache;
        REAL* tmpStateFrequencies = hFrequenciesCache;
        
#ifdef DOUBLE_PRECISION
        tmpWeights = inWeights;
        tmpStateFrequencies = inStateFrequencies;
#else
        MEMCNV(hWeightsCache, inWeights, count * kCategoryCount, REAL);
        MEMCNV(hFrequenciesCache, inStateFrequencies, kPaddedStateCount * kCategoryCount, REAL);
#endif        
        gpu->MemcpyHostToDevice(dWeights, tmpWeights, SIZE_REAL * count * kCategoryCount);
        gpu->MemcpyHostToDevice(dFrequencies, tmpStateFrequencies,
                                SIZE_REAL * kPaddedStateCount * kCategoryCount);
        
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
        
#ifdef DYNAMIC_SCALING
        // TODO: fix calculateEdgLnL with dynamic scaling
        
        if (statesChild != 0) {
            kernels->StatesPartialsEdgeLikelihoods(dPartialsTmp, partialsParent, statesChild,
                                                   transMatrix, kPaddedPatternCount,
                                                   kCategoryCount);
        } else {
            kernels->PartialsPartialsEdgeLikelihoods(dPartialsTmp, partialsParent, partialsChild,
                                                     transMatrix, kPaddedPatternCount,
                                                     kCategoryCount);
        }
        
        if (kDoRescaling) {
            // Construct node-list for scalingFactors
            int length = scalingFactorsCount[0];
            for(int n = 0; n < length; n++)
                hPtrQueue[n] = dScalingFactors[scalingFactorsIndices[n]];
                        
            gpu->MemcpyHostToDevice(dPtrQueue, hPtrQueue, sizeof(GPUPtr) * length);
            
            // Accumulate scaling factors
            kernels->ComputeRootDynamicScaling(dPtrQueue, dRootScalingFactors, length,
                                               kPaddedPatternCount);
        }
        
        kDoRescaling = 0;
        
        kernels->IntegrateLikelihoodsDynamicScaling(dIntegrationTmp, dPartialsTmp, dWeights,
                                                    dFrequencies, dRootScalingFactors,
                                                    kPaddedPatternCount, kCategoryCount);
#else
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
        
        kernels->IntegrateLikelihoods(dIntegrationTmp, dPartialsTmp, dWeights, dFrequencies,
                                      kPaddedPatternCount, kCategoryCount);
        
#endif // DYNAMIC_SCALING
        
#ifdef DOUBLE_PRECISION
        gpu->MemcpyDeviceToHost(outLogLikelihoods, dIntegrationTmp, SIZE_REAL * kPatternCount);
#else
        gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dIntegrationTmp, SIZE_REAL * kPatternCount);
        MEMCNV(outLogLikelihoods, hLogLikelihoodsCache, kPatternCount, double);
#endif
        
#ifdef DEBUG
        printf("edgeLogLike = ");
        printfVectorD(outLogLikelihoods, kPatternCount);
        exit(-1);
#endif
    } else {
        // TODO: implement calculateEdgeLnL for count > 1
        assert(false);
    }
    
    
#ifdef DEBUG_FLOW
    fprintf(stderr, "Exiting calculateEdgeLogLikelihoods\n");
#endif
    
    return NO_ERROR;
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
                                              int categoryCount) {
    BeagleImpl* impl = new BeagleGPUImpl();
    try {
        if (impl->createInstance(tipCount, partialsBufferCount, compactBufferCount, stateCount,
                                 patternCount, eigenBufferCount, matrixBufferCount,
                                 categoryCount) == 0)
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
