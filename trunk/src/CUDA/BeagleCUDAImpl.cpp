
/*
 *  BeagleCUDAImpl.cpp
 *  BEAGLE
 *
 * @author Marc Suchard
 * @author Andrew Rambaut
 * @author Daniel Ayres
 */

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#include "beagle.h"
#include "BeagleCUDAImpl.h"
#include "CUDASharedFunctions.h"

int currentDevice = -1;

BeagleCUDAImpl::~BeagleCUDAImpl() {
    freeMemory();
}

int BeagleCUDAImpl::createInstance(int tipCount,
                                   int partialsBufferCount,
                                   int compactBufferCount,
                                   int stateCount,
                                   int patternCount,
                                   int eigenDecompositionCount,
                                   int matrixCount) {
    
    // TODO: Determine if CUDA device satisfies memory requirements.
    
    int numDevices = getGPUDeviceCount();
    if (numDevices == 0) {
        fprintf(stderr, "No GPU devices found");
        return GENERAL_ERROR;
    }
    
    // Static load balancing; each instance gets added to the next available device
    currentDevice++;
    if (currentDevice == numDevices)
        currentDevice = 0;
    
    printGPUInfo(currentDevice);
    
    kDevice = currentDevice;
    
    kTipCount = tipCount;
    kPartialsBufferCount = partialsBufferCount;
    kCompactBufferCount = compactBufferCount;
    kStateCount = stateCount;
    kPatternCount = patternCount;
    kEigenDecompCount = eigenDecompositionCount;
    kMatrixCount = matrixCount;
    
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
    
    kPartialsSize = kPaddedPatternCount * kPaddedStateCount;
    kMatrixSize = kPaddedStateCount * kPaddedStateCount;
    // TODO: change to 2 * kPaddedStateCount for complex models (?)
    kEigenValuesSize = kPaddedStateCount;
    
    // TODO: only allocate if necessary on the fly
    hWeightsCache = (REAL*) calloc(kBufferCount, SIZE_REAL);
    hFrequenciesCache = (REAL*) calloc(kPaddedStateCount, SIZE_REAL);
    hPartialsCache = (REAL*) calloc(kPartialsSize, SIZE_REAL);
    hStatesCache = (int*) calloc(kPaddedPatternCount, SIZE_INT);
    hMatrixCache = (REAL*) calloc(2 * kMatrixSize + kEigenValuesSize, SIZE_REAL);
#ifndef DOUBLE_PRECISION
    hLogLikelihoodsCache = (REAL*) malloc(kPatternCount * SIZE_REAL);
#endif
    
    hTmpTipPartials = (REAL**) calloc(sizeof(REAL*), kTipCount);
    hTmpStates = (int**) calloc(sizeof(int*), kTipCount);
    
    kDoRescaling = 1;
    
    kLastCompactBufferIndex = -1;
    kLastTipPartialsBufferIndex = -1;
    
    return NO_ERROR;
}

int BeagleCUDAImpl::initializeInstance(InstanceDetails* returnInfo) {
    
#ifdef DEBUG_FLOW
    fprintf(stderr, "Entering initialize\n");
#endif

    cudaSetDevice(kDevice);
    
    dEvec = allocateGPURealMemory(kMatrixSize);
    dIevc = allocateGPURealMemory(kMatrixSize);
    
    dEigenValues = allocateGPURealMemory(kEigenValuesSize);
    
    dWeights = allocateGPURealMemory(kBufferCount);
    
    dFrequencies = allocateGPURealMemory(kPaddedStateCount);
    
    dIntegrationTmp = allocateGPURealMemory(kPaddedPatternCount);
    
    dPartials = (REAL***) malloc(sizeof(REAL**) * 2);
    
    // Fill with 0s so 'free' does not choke if unallocated
    dPartials[0] = (REAL**) calloc(sizeof(REAL*), kBufferCount);
    
    // Internal nodes have 0s so partials are used
    dStates = (int **) calloc(sizeof(int*), kBufferCount); 
    
    dCompactBuffers = (int **) malloc(sizeof(int*) * kCompactBufferCount); 
    dTipPartialsBuffers = (REAL**) malloc(sizeof(REAL*) * kTipPartialsBufferCount);
    
#ifdef DYNAMIC_SCALING
    dScalingFactors = (REAL***) malloc(sizeof(REAL**) * 2);
    dScalingFactors[0] = (REAL**) malloc(sizeof(REAL*) * kBufferCount);
    dRootScalingFactors = allocateGPURealMemory(kPaddedPatternCount);
#endif
    
    for (int i = 0; i < kBufferCount; i++) {        
        if (i < kTipCount) { // For the tips
            if (i < kCompactBufferCount)
                dCompactBuffers[i] = allocateGPUIntMemory(kPaddedPatternCount);
            if (i < kTipPartialsBufferCount)
                dTipPartialsBuffers[i] = allocateGPURealMemory(kPartialsSize);
        } else {
            dPartials[0][i] = allocateGPURealMemory(kPartialsSize);
#ifdef DYNAMIC_SCALING
            dScalingFactors[0][i] = allocateGPURealMemory(kPaddedPatternCount);
#endif
        }
    }
    
    kLastCompactBufferIndex = kCompactBufferCount - 1;
    kLastTipPartialsBufferIndex = kTipPartialsBufferCount - 1;
    
    dMatrices = (REAL***) malloc(sizeof(REAL**) * 2);
    dMatrices[0] = (REAL**) malloc(sizeof(REAL*) * kMatrixCount);
    
    for (int i = 0; i < kMatrixCount; i++) {
        dMatrices[0][i] = allocateGPURealMemory(kMatrixSize);
    }
    
    // No execution has more no kBufferCount events
    dBranchLengths = allocateGPURealMemory(kBufferCount);
    
    dDistanceQueue = allocateGPURealMemory(kMatrixCount);
    hDistanceQueue = (REAL*) malloc(sizeof(REAL) * kMatrixCount);
    
    checkNativeMemory(hDistanceQueue);
    
    SAFE_CUDA(cudaMalloc((void**) &dPtrQueue, sizeof(REAL*) * kMatrixCount), dPtrQueue);
    hPtrQueue = (REAL**) malloc(sizeof(REAL*) * kMatrixCount);
    
    checkNativeMemory(hPtrQueue);
    
    loadTipPartialsAndStates();
    freeTmpTipPartialsAndStates();
    
    kDeviceMemoryAllocated = 1;
    
#ifdef DEBUG_FLOW
    fprintf(stderr, "Exiting initialize\n");
#endif
    
    return NO_ERROR;
}

int BeagleCUDAImpl::setPartials(int bufferIndex,
                                const double* inPartials) {
#ifdef DEBUG_FLOW
    fprintf(stderr,"Entering setTipPartials\n");
#endif
    
    const double* inPartialsOffset = inPartials;
    REAL* tmpRealPartialsOffset = hPartialsCache;
    
    for (int i = 0; i < kPatternCount; i++) {
#ifdef DOUBLE_PRECISION
        memcpy(tmpRealPartialsOffset, inPartialsOffset, SIZE_REAL * kStateCount);
#else
        MEMCPY(tmpRealPartialsOffset, inPartialsOffset, kStateCount, REAL);
#endif
        tmpRealPartialsOffset += kPaddedStateCount;
        inPartialsOffset += kStateCount;
    }
    
    if (kDeviceMemoryAllocated) {
        if (bufferIndex < kTipCount) {
            assert(kLastTipPartialsBufferIndex >= 0 && kLastTipPartialsBufferIndex <
                   kTipPartialsBufferCount);
            dPartials[0][bufferIndex] = dTipPartialsBuffers[kLastTipPartialsBufferIndex--];
        }
        // Copy to CUDA device
        SAFE_CUDA(cudaMemcpy(dPartials[0][bufferIndex], hPartialsCache, SIZE_REAL * kPartialsSize,
                             cudaMemcpyHostToDevice),
                  dPartials[0][bufferIndex]);
    } else {
        hTmpTipPartials[bufferIndex] = (REAL*) malloc(SIZE_REAL * kPartialsSize);
        checkNativeMemory(hTmpTipPartials[bufferIndex]);
        memcpy(hTmpTipPartials[bufferIndex], hPartialsCache, SIZE_REAL * kPartialsSize);
    }
    
#ifdef DEBUG_FLOW
    fprintf(stderr,"Exiting setTipPartials\n");
#endif
    
    return NO_ERROR;
}

int BeagleCUDAImpl::getPartials(int bufferIndex,
                                double* inPartials) {
    //TODO: implement getPartials
    assert (false);
}

int BeagleCUDAImpl::setTipStates(int tipIndex,
                                 const int* inStates) {
    //TODO: test setTipStates
    
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
        // Copy to CUDA device
        SAFE_CUDA(cudaMemcpy(dStates[tipIndex], hStatesCache, SIZE_INT * kPaddedPatternCount,
                             cudaMemcpyHostToDevice),
                  dStates[tipIndex]);
    } else {
        hTmpStates[tipIndex] = (int*) malloc(SIZE_INT * kPaddedPatternCount);
        checkNativeMemory(hTmpStates[tipIndex]);
        memcpy(hTmpStates[tipIndex], hStatesCache, SIZE_INT * kPaddedPatternCount);
    }
    
#ifdef DEBUG_FLOW
    fprintf(stderr, "Exiting setTipStates\n");
#endif
    
    return NO_ERROR;
}

int BeagleCUDAImpl::setEigenDecomposition(int matrixIndex,
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
        MEMCPY(tmpIevc, (inInverseEigenVectors + i * kStateCount), kStateCount, REAL);
        MEMCPY(tmpEvec, (inEigenVectors + i * kStateCount), kStateCount, REAL);
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
    MEMCPY(Eval, inEigenValues, STATE_COUNT, REAL);
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
    
    // Copy to CUDA device
    cudaMemcpy(dIevc, Ievc, SIZE_REAL * kMatrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dEvec, Evec, SIZE_REAL * kMatrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dEigenValues, Eval, SIZE_REAL * kPaddedStateCount, cudaMemcpyHostToDevice);
    
#ifdef DEBUG_BEAGLE
    printfCudaVector(dEigenValues, kPaddedStateCount);
    printfCudaVector(dEvec, kMatrixSize);
    printfCudaVector(dIevc, kPaddedStateCount * kPaddedStateCount);
#endif
    
#ifdef DEBUG_FLOW
    fprintf(stderr, "Exiting updateEigenDecomposition\n");
#endif
    
    return NO_ERROR;
}

int BeagleCUDAImpl::setTransitionMatrix(int matrixIndex,
                                        const double* inMatrix) {
    //TODO: implement setTransitionMatrix
    assert(false);
}

int BeagleCUDAImpl::updateTransitionMatrices(int eigenIndex,
                                             const int* probabilityIndices,
                                             const int* firstDerivativeIndices,
                                             const int* secondDervativeIndices,
                                             const double* edgeLengths,
                                             int count) {
#ifdef DEBUG_FLOW
    fprintf(stderr,"Entering updateMatrices\n");
#endif
    
    // TODO: calculate derivatives
    
    for (int i = 0; i < count; i++) {
        hPtrQueue[i] = dMatrices[0][probabilityIndices[i]];
        hDistanceQueue[i] = (REAL) edgeLengths[i];
    }

    cudaMemcpy(dPtrQueue, hPtrQueue, sizeof(REAL*) * count, cudaMemcpyHostToDevice);
    cudaMemcpy(dDistanceQueue, hDistanceQueue, SIZE_REAL * count, cudaMemcpyHostToDevice);

    // Set-up and call GPU kernel
    nativeGPUGetTransitionProbabilitiesSquare(dPtrQueue, dEvec, dIevc, dEigenValues, dDistanceQueue,
                                              count);
    
#ifdef DEBUG_BEAGLE
    printfCudaVector(hPtrQueue[0], kMatrixSize);
#endif
    
#ifdef DEBUG_FLOW
    fprintf(stderr, "Exiting updateMatrices\n");
#endif
    
    return NO_ERROR;
}

int BeagleCUDAImpl::updatePartials(const int* operations,
                                   int operationCount,
                                   int rescale) {
    // TODO: remove this categoryCount hack
    int categoryCount = 1;
    
#ifdef DEBUG_FLOW
    fprintf(stderr, "Entering updatePartials\n");
#endif
    
#ifdef DYNAMIC_SCALING
    if (kDoRescaling == 0) // Forces rescaling on first computation
        kDoRescaling = rescale;
#endif
    
    int die = 0;
    
    // Serial version
    for (int op = 0; op < operationCount; op++) {
        const int parIndex = operations[op * 5];
        const int child1Index = operations[op * 5 + 1];
        const int child1TransMatIndex = operations[op * 5 + 2];
        const int child2Index = operations[op * 5 + 3];
        const int child2TransMatIndex = operations[op * 5 + 4];
        
        REAL* matrices1 = dMatrices[0][child1TransMatIndex];
        REAL* matrices2 = dMatrices[0][child2TransMatIndex];
        
        REAL* partials1 = dPartials[0][child1Index];
        REAL* partials2 = dPartials[0][child2Index];
        
        REAL* partials3 = dPartials[0][parIndex];
        
        int* tipStates1 = dStates[child1Index];
        int* tipStates2 = dStates[child2Index];
        
#ifdef DYNAMIC_SCALING
        REAL* scalingFactors = dScalingFactors[0][parIndex];
        
        if (tipStates1 != 0) {
            if (tipStates2 != 0 ) {
                nativeGPUStatesStatesPruningDynamicScaling(tipStates1, tipStates2, partials3,
                                                           matrices1, matrices2, scalingFactors,
                                                           kPaddedPatternCount, categoryCount,
                                                           kDoRescaling);
            } else {
                nativeGPUStatesPartialsPruningDynamicScaling(tipStates1, partials2, partials3,
                                                             matrices1, matrices2, scalingFactors,
                                                             kPaddedPatternCount, categoryCount,
                                                             kDoRescaling);
                die = 1;
            }
        } else {
            if (tipStates2 != 0) {
                nativeGPUStatesPartialsPruningDynamicScaling(tipStates2, partials1, partials3,
                                                             matrices2, matrices1, scalingFactors,
                                                             kPaddedPatternCount, categoryCount,
                                                             kDoRescaling);
                die = 1;
            } else {
                nativeGPUPartialsPartialsPruningDynamicScaling(partials1, partials2, partials3,
                                                               matrices1, matrices2, scalingFactors,
                                                               kPaddedPatternCount, categoryCount,
                                                               kDoRescaling);
            }
        }
#else
        if (tipStates1 != 0) {
            if (tipStates2 != 0 ) {
                nativeGPUStatesStatesPruning(tipStates1, tipStates2, partials3, matrices1,
                                             matrices2, kPaddedPatternCount, categoryCount);
            } else {
                nativeGPUStatesPartialsPruning(tipStates1, partials2, partials3, matrices1,
                                               matrices2, kPaddedPatternCount, categoryCount);
                die = 1;
            }
        } else {
            if (tipStates2 != 0) {
                nativeGPUStatesPartialsPruning(tipStates2, partials1, partials3, matrices2,
                                               matrices1, kPaddedPatternCount, categoryCount);
                die = 1;
            } else {
                nativeGPUPartialsPartialsPruning(partials1, partials2, partials3, matrices1,
                                                 matrices2, kPaddedPatternCount, categoryCount);
            }
        }
#endif // DYNAMIC_SCALING
        
#ifdef DEBUG_BEAGLE
        fprintf(stderr, "kPaddedPatternCount = %d\n", kPaddedPatternCount);
        fprintf(stderr, "kPatternCount = %d\n", kPatternCount);
        fprintf(stderr, "categoryCount  = %d\n", categoryCount);
        fprintf(stderr, "partialSize = %d\n", kPartialsSize);
        if (tipStates1)
            printfCudaInt(tipStates1, kPaddedPatternCount);
        else
            printfCudaVector(partials1, kPartialsSize);
        if (tipStates2)
            printfCudaInt(tipStates2, kPaddedPatternCount);
        else
            printfCudaVector(partials2, kPartialsSize);
        fprintf(stderr, "node index = %d\n", parIndex);
        printfCudaVector(partials3, kPartialsSize);
        
        if(parIndex == 106)
            exit(-1);
#endif
    }
    
#ifdef DEBUG_FLOW
    fprintf(stderr, "Exiting updatePartials\n");
#endif
    
    return NO_ERROR;
}

int BeagleCUDAImpl::waitForPartials(const int* destinationPartials,
                                    int destinationPartialsCount) {
    return NO_ERROR;
}

int BeagleCUDAImpl::calculateRootLogLikelihoods(const int* bufferIndices,
                                                const double* inWeights,
                                                const double* inStateFrequencies,
                                                int count,
                                                double* outLogLikelihoods) {
    if (count == 1) { 

#ifdef DEBUG_FLOW
        fprintf(stderr, "Entering calculateLogLikelihoods\n");
#endif
        
        REAL* tmpWeights = hWeightsCache;
        REAL* tmpStateFrequencies = hFrequenciesCache;
                
#ifdef DOUBLE_PRECISION
        // TODO: fix const assigned no non-const
        tmpWeights = inWeights;
        tmpStateFrequencies = inStateFrequencies;
#else
        MEMCPY(hWeightsCache, inWeights, count, REAL);
        MEMCPY(hFrequenciesCache, inStateFrequencies, kPaddedStateCount, REAL);
#endif        
        cudaMemcpy(dWeights, tmpWeights, SIZE_REAL * count, cudaMemcpyHostToDevice);
        cudaMemcpy(dFrequencies, tmpStateFrequencies, SIZE_REAL * kPaddedStateCount,
                   cudaMemcpyHostToDevice);

        const int rootNodeIndex = bufferIndices[0];
        
#ifdef DYNAMIC_SCALING
        if (kDoRescaling) {
            // Construct node-list for scalingFactors
            int n;
            int length = kBufferCount - kTipCount;
            for(n = 0; n < length; n++)
                hPtrQueue[n] = dScalingFactors[0][n + kTipCount];
            
            cudaMemcpy(dPtrQueue, hPtrQueue, sizeof(REAL*) * length, cudaMemcpyHostToDevice);
            
            // Computer scaling factors at the root
            nativeGPUComputeRootDynamicScaling(dPtrQueue, dRootScalingFactors, length,
                                               kPaddedPatternCount);
        }
        
        kDoRescaling = 0;
        
        nativeGPUIntegrateLikelihoodsDynamicScaling(dIntegrationTmp, dPartials[0][rootNodeIndex],
                                                    dWeights, dFrequencies,
                                                    dRootScalingFactors, kPaddedPatternCount,
                                                    count, kBufferCount);
#else
        nativeGPUIntegrateLikelihoods(dIntegrationTmp, dPartials[0][rootNodeIndex],
                                      dWeights, dFrequencies, kPaddedPatternCount,
                                      count);
#endif // DYNAMIC_SCALING
        
#ifdef DOUBLE_PRECISION
        cudaMemcpy(outLogLikelihoods, dIntegrationTmp, SIZE_REAL * kPatternCount,
                   cudaMemcpyDeviceToHost);
#else
        cudaMemcpy(hLogLikelihoodsCache, dIntegrationTmp, SIZE_REAL * kPatternCount,
                   cudaMemcpyDeviceToHost);
        MEMCPY(outLogLikelihoods, hLogLikelihoodsCache, kPatternCount, double);
#endif
        
#ifdef DEBUG
        printf("logLike = ");
        printfVectorD(outLogLikelihoods, kPatternCount);
        exit(-1);
#endif
    } else {
        // TODO: implement calculate root lnL for multiple count
        assert(false);
    }

    
#ifdef DEBUG_FLOW
    fprintf(stderr, "Exiting calculateLogLikelihoods\n");
#endif
    
    return NO_ERROR;
}

int BeagleCUDAImpl::calculateEdgeLogLikelihoods(const int* parentBufferIndices,
                                                const int* childBufferIndices,
                                                const int* probabilityIndices,
                                                const int* firstDerivativeIndices,
                                                const int* secondDerivativeIndices,
                                                const double* inWeights,
                                                const double* inStateFrequencies,
                                                int count,
                                                double* outLogLikelihoods,
                                                double* outFirstDerivatives,
                                                double* outSecondDerivatives) {
    // TODO: implement calculateEdgeLogLikelihoods
    assert(false);
}


void BeagleCUDAImpl::checkNativeMemory(void* ptr) {
    if (ptr == NULL) {
        fprintf(stderr, "Unable to allocate some memory!\n");
        exit(-1);
    }
}

long BeagleCUDAImpl::memoryRequirement(int kTipCount,
                                       int stateCount) {
// TODO: compute device memory requirements
    
    // Evec, storedEvec
    // Ivec, storedIevc
    // EigenValues, storeEigenValues
    // Frequencies, storedFrequencies
    // categoryProportions, storedCategoryProportions
    // integrationTmp (kPaddedPatternCount)
    
    assert(false);
}

void BeagleCUDAImpl::freeMemory() {
    for (int i = 0; i < kBufferCount; i++) {
        if (i < kTipCount) {
            if (i < kCompactBufferCount)
                freeGPUMemory(dCompactBuffers[i]);
            if (i < kTipPartialsBufferCount)
                freeGPUMemory(dTipPartialsBuffers[i]);
        } else {
            freeGPUMemory(dPartials[0][i]);
#ifdef DYNAMIC_SCALING
            freeGPUMemory(dScalingFactors[0][i]);
#endif
        }
    }
    
    for (int i = 0; i < kMatrixCount; i++)
        freeGPUMemory(dMatrices[0][i]);
    
    freeGPUMemory(dEigenValues);
    freeGPUMemory(dEvec);
    freeGPUMemory(dIevc);
    
    freeGPUMemory(dWeights);
    freeGPUMemory(dFrequencies);
    freeGPUMemory(dIntegrationTmp);
    
    free(dPartials[0]);
    free(dPartials);

    free(dMatrices[0]);
    free(dMatrices);
    
#ifdef DYNAMIC_SCALING
    free(dScalingFactors[0]);
    free(dScalingFactors);
    freeGPUMemory(dRootScalingFactors);
#endif
    
    free(dStates);
    
    free(dCompactBuffers);
    free(dTipPartialsBuffers);
    
    freeGPUMemory(dBranchLengths);
    
    free(hDistanceQueue);
    free(hPtrQueue);
    freeGPUMemory(dDistanceQueue);
    freeGPUMemory(dPtrQueue);
    
    free(hWeightsCache);
    free(hFrequenciesCache);
    free(hPartialsCache);
    free(hStatesCache);
    free(hMatrixCache);
    free(hLogLikelihoodsCache);
}

void BeagleCUDAImpl::freeTmpTipPartialsAndStates() {
    for (int i = 0; i < kTipCount; i++) {
        if (hTmpTipPartials[i] != 0)
            free(hTmpTipPartials[i]);
        else if (hTmpStates[i] != 0)
            free(hTmpStates[i]);
    }
    
    free(hTmpTipPartials);
    free(hTmpStates);
}

void BeagleCUDAImpl::loadTipPartialsAndStates() {    
    for (int i = 0; i < kTipCount; i++) {
        if (hTmpTipPartials[i] != 0) {
            assert(kLastTipPartialsBufferIndex >= 0 && kLastTipPartialsBufferIndex < 
                   kTipPartialsBufferCount);
            dPartials[0][i] = dTipPartialsBuffers[kLastTipPartialsBufferIndex--];
            cudaMemcpy(dPartials[0][i], hTmpTipPartials[i], SIZE_REAL * kPartialsSize,
                       cudaMemcpyHostToDevice);
        } else if (hTmpStates[i] != 0) {
            assert(kLastCompactBufferIndex >= 0 && kLastCompactBufferIndex < kCompactBufferCount);
            dStates[i] = dCompactBuffers[kLastCompactBufferIndex--];
            cudaMemcpy(dStates[i], hTmpStates[i], SIZE_INT * kPaddedPatternCount,
                       cudaMemcpyHostToDevice);
        }
    }
}

void BeagleCUDAImpl::transposeSquareMatrix(REAL* mat,
                                           int size) {
    for (int i = 0; i < size - 1; i++) {
        for (int j = i + 1; j < size; j++) {
            REAL tmp = mat[i * size + j];
            mat[i * size + j] = mat[j * size + i];
            mat[j * size + i] = tmp;
        }
    }
}

int BeagleCUDAImpl::getGPUDeviceCount() {
    int cDevices;
    CUresult status;
    status = cuInit(0);
    if (CUDA_SUCCESS != status)
        return 0;
    status = cuDeviceGetCount(&cDevices);
    if (CUDA_SUCCESS != status)
        return 0;
    if (cDevices == 0) {
        return 0;
    }
    return cDevices;
}

void BeagleCUDAImpl::printGPUInfo(int device) {
    
    fprintf(stderr, "GPU Device Information:");
    
    char name[256];
    int totalGlobalMemory = 0;
    int clockSpeed = 0;
    
    // New CUDA functions in cutil.h do not work in JNI files
    getGPUInfo(device, name, &totalGlobalMemory, &clockSpeed);
    fprintf(stderr, "\nDevice #%d: %s\n", (device + 1), name);
    double mem = totalGlobalMemory / 1024.0 / 1024.0;
    double clo = clockSpeed / 1000000.0;
    fprintf(stderr, "\tGlobal Memory (MB) : %1.2f\n", mem);
    fprintf(stderr, "\tClock Speed (Ghz)  : %1.2f\n", clo);
}

void BeagleCUDAImpl::getGPUInfo(int iDevice,
                                char* name,
                                int* memory,
                                int* speed) {
    cudaDeviceProp deviceProp;
    memset(&deviceProp, 0, sizeof(deviceProp));
    cudaGetDeviceProperties(&deviceProp, iDevice);
    *memory = deviceProp.totalGlobalMem;
    *speed = deviceProp.clockRate;
    strcpy(name, deviceProp.name);
}


///////////////////////////////////////////////////////////////////////////////
// BeagleCUDAImplFactory public methods

BeagleImpl*  BeagleCUDAImplFactory::createImpl(int tipCount,
                                               int partialsBufferCount,
                                               int compactBufferCount,
                                               int stateCount,
                                               int patternCount,
                                               int eigenBufferCount,
                                               int matrixBufferCount) {
    BeagleImpl* impl = new BeagleCUDAImpl();
    try {
        if (impl->createInstance(tipCount, partialsBufferCount, compactBufferCount, stateCount,
                                 patternCount, eigenBufferCount, matrixBufferCount) == 0)
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

const char* BeagleCUDAImplFactory::getName() {
    return "CUDA";
}
