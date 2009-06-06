
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
    //freeNativeMemory();
}

int BeagleCUDAImpl::initialize(int tipCount,
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
    
    initializeDevice(currentDevice, tipCount, partialsBufferCount, compactBufferCount, stateCount,
                     patternCount, eigenDecompositionCount, matrixCount);
    
    return NO_ERROR;
}

void BeagleCUDAImpl::initializeDevice(int deviceNumber,
                                      int inTipCount,
                                      int inPartialsBufferCount,
                                      int inCompactBufferCount,
                                      int inStateCount,
                                      int inPatternCount,
                                      int inEigenDecompositionCount,
                                      int inMatrixCount) {
    
#ifdef DEBUG_FLOW
    fprintf(stderr, "Entering initialize\n");
#endif

    kDevice = deviceNumber;
    
    kTipCount = inTipCount;
    kBufferCount = inPartialsBufferCount + inCompactBufferCount;
    kTrueStateCount = inStateCount;
    kTruePatternCount = inPatternCount;
    kMatrixCount = inMatrixCount;
    
    if (kTrueStateCount <= 4)
        kStateCount = 4;
    else if (kTrueStateCount <= 16)
        kStateCount = 16;
    else if (kTrueStateCount <= 32)
        kStateCount = 32;
    else if (kTrueStateCount <= 64)
        kStateCount = 64;
    else if (kTrueStateCount <= 128)
        kStateCount = 128;
    else if (kTrueStateCount <= 192)
        kStateCount = 192;
    else
        kStateCount = kTrueStateCount + kTrueStateCount % 16;
    
    kPaddedPatternCount = 0;
    
    // Make sure that kPatternCount + kPaddedPatternCount is multiple of 4 for DNA model
    if (kStateCount == 4 && kTruePatternCount % 4 != 0)
        kPaddedPatternCount = 4 - kTruePatternCount % 4;
    else
        kPaddedPatternCount = 0;
#ifdef DEBUG
    fprintf(stderr, "Padding patterns for 4-state model:\n");
    fprintf(stderr, "\ttruePatternCount = %d\n\tpaddedPatterns = %d\n", kTruePatternCount,
            kPaddedPatternCount);
#endif // DEBUG

    kPatternCount = kTruePatternCount + kPaddedPatternCount;
    
    kPartialsSize = kPatternCount * kStateCount;
    kMatrixSize = kStateCount * kStateCount;
    kEigenValuesSize = kStateCount;  // TODO: change to 2 * kStateCount for complex models (?)
    
    hFrequenciesCache = (REAL*) calloc(kStateCount, SIZE_REAL);
    
    // TODO: Only allocate if necessary on the fly
    hPartialsCache = (REAL*) calloc(kPartialsSize, SIZE_REAL);
    hStatesCache = (int*) calloc(kPatternCount, SIZE_INT);
    
    hMatrixCache = (REAL*) calloc(2 * kMatrixSize + kEigenValuesSize, SIZE_REAL);
    
#ifndef DOUBLE_PRECISION
    hLogLikelihoodsCache = (REAL*) malloc(kTruePatternCount * SIZE_REAL);
#endif
    
    doRescaling = 1;
    sinceRescaling = 0;
    
#ifndef PRE_LOAD
    // Fill with 0 (= no states to load)
    hTmpStates = (int**) calloc(sizeof(int*), kTipCount);
    initializeInstanceMemory();
#else
    // initialize temporary storage before likelihood thread exists
    loaded = 0;
    hTmpPartials = (REAL**) malloc(sizeof(REAL*) * kTipCount);
    
    // TODO: Only need to allocate tipPartials or tipStates, not both
    // Should just fill with 0 (= no partials to load)
    for (int i = 0; i < kTipCount; i++) {
        hTmpPartials[i] = (REAL*) malloc(SIZE_REAL * kPartialsSize);
    }
    
    // Fill with 0 (= no states to load)
    hTmpStates = (int**) calloc(sizeof(int*), kTipCount);
    initializeInstanceMemory();
#endif
    
#ifdef DEBUG_FLOW
    fprintf(stderr, "Exiting initialize\n");
#endif
}

void BeagleCUDAImpl::initializeInstanceMemory() {
    cudaSetDevice(kDevice);
    int i;
    
    dEvec = allocateGPURealMemory(kMatrixSize);
    dIevc = allocateGPURealMemory(kMatrixSize);
    
    dEigenValues = allocateGPURealMemory(kEigenValuesSize);
    
    dFrequencies = allocateGPURealMemory(kStateCount);
    
    dIntegrationTmp = allocateGPURealMemory(kPatternCount);
    
    dPartials = (REAL***) malloc(sizeof(REAL**) * 2);
    
    // Fill with 0s so 'free' does not choke if unallocated
    dPartials[0] = (REAL**) calloc(sizeof(REAL*), kBufferCount);
    dPartials[1] = (REAL**) calloc(sizeof(REAL*), kBufferCount);
    
    // Internal nodes have 0s so partials are used
    dStates = (int **) calloc(sizeof(int*), kBufferCount); 
    
#ifdef DYNAMIC_SCALING
    dScalingFactors = (REAL***) malloc(sizeof(REAL**) * 2);
    dScalingFactors[0] = (REAL**) malloc(sizeof(REAL*) * kBufferCount);
    dScalingFactors[1] = (REAL**) malloc(sizeof(REAL*) * kBufferCount);
    dRootScalingFactors = allocateGPURealMemory(kPatternCount);
#endif
    
    for (i = 0; i < kBufferCount; i++) {        
        if (i < kTipCount) { // For the tips
            if (hTmpStates[i] == 0) // If no tipStates
                dPartials[0][i] = allocateGPURealMemory(kPartialsSize);
            else
                dStates[i] = allocateGPUIntMemory(kPatternCount);
        } else {
            dPartials[0][i] = allocateGPURealMemory(kPartialsSize);
            dPartials[1][i] = allocateGPURealMemory(kPartialsSize);
#ifdef DYNAMIC_SCALING
            dScalingFactors[0][i] = allocateGPURealMemory(kPatternCount);
            dScalingFactors[1][i] = allocateGPURealMemory(kPatternCount);
#endif
        }
    }
    
    dMatrices = (REAL***) malloc(sizeof(REAL**) * 2);
    dMatrices[0] = (REAL**) malloc(sizeof(REAL*) * kBufferCount);
    dMatrices[1] = (REAL**) malloc(sizeof(REAL*) * kBufferCount);
    
    for (i = 0; i < kBufferCount; i++) {
        dMatrices[0][i] = allocateGPURealMemory(kMatrixSize);
        dMatrices[1][i] = allocateGPURealMemory(kMatrixSize);
    }
    
    // No execution has more no kBufferCount events
    dNodeIndices = allocateGPUIntMemory(kBufferCount);
    hNodeIndices = (int*) malloc(sizeof(int) * kBufferCount);
    hDependencies = (int*) malloc(sizeof(int) * kBufferCount);
    dBranchLengths = allocateGPURealMemory(kBufferCount);
    
    checkNativeMemory(hNodeIndices);
    checkNativeMemory(hDependencies);
    
    dDistanceQueue = allocateGPURealMemory(kBufferCount);
    hDistanceQueue = (REAL*) malloc(sizeof(REAL) * kBufferCount);
    
    checkNativeMemory(hDistanceQueue);
    
    int len = 5;
    
    SAFE_CUDA(cudaMalloc((void**) &dPtrQueue, sizeof(REAL*) * kBufferCount * len), dPtrQueue);
    hPtrQueue = (REAL**) malloc(sizeof(REAL*) * kBufferCount * len);
    
    checkNativeMemory(hPtrQueue);
}

int BeagleCUDAImpl::setPartials(int bufferIndex,
                                const double* inPartials) {
#ifdef DEBUG_FLOW
    fprintf(stderr,"Entering setTipPartials\n");
#endif
    
    const double* inPartialsOffset = inPartials;
    REAL* tmpRealArrayOffset = hPartialsCache;
    
    for (int i = 0; i < kTruePatternCount; i++) {
#ifdef DOUBLE_PRECISION
        memcpy(tmpRealArrayOffset, inPartialsOffset, SIZE_REAL * kStateCount);
#else
        MEMCPY(tmpRealArrayOffset, inPartialsOffset, kStateCount, REAL);
#endif
        tmpRealArrayOffset += kStateCount;
        inPartialsOffset += kStateCount;
    }
    
#ifndef PRE_LOAD
    // Copy to CUDA device
    SAFE_CUDA(cudaMemcpy(dPartials[0][bufferIndex], hPartialsCache, SIZE_REAL * kPartialsSize,
                         cudaMemcpyHostToDevice),
              dPartials[0][bufferIndex]);
#else
    memcpy(hTmpPartials[bufferIndex], hPartialsCache, SIZE_REAL * kPartialsSize);
#endif // PRE_LOAD
    
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
    //TODO: update and test setTipStates
    
#ifdef DEBUG_FLOW
    fprintf(stderr, "Entering setTipStates\n");
#endif
    
    //  memcpy(hStatesCache,inStates,SIZE_INT*kTruePatternCount);
    for(int i = 0; i < kTruePatternCount; i++) {
        hStatesCache[i] = inStates[i];
        if (hStatesCache[i] >= STATE_COUNT)
            hStatesCache[i] = kStateCount;
    }
    // Padded extra patterns
    for(int i = 0; i < kPaddedPatternCount; i++)
        hStatesCache[kTruePatternCount + i] = kStateCount;
    
#ifndef PRE_LOAD
    // Copy to CUDA device
    SAFE_CUDA(cudaMemcpy(dStates[tipIndex], inStates, SIZE_INT * kPatternCount,
                         cudaMemcpyHostToDevice),
              dStates[tipIndex]);
#else
    
    hTmpStates[tipIndex] = (int*) malloc(SIZE_INT * kPatternCount);
    
    memcpy(hTmpStates[tipIndex], hStatesCache, SIZE_INT * kPatternCount);
#endif // PRE_LOAD
    
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
        
        tmpIevc += kStateCount;
        tmpEvec += kStateCount;
    }
    
    // Transposing matrices avoids incoherent memory read/writes    
    transposeSquareMatrix(Ievc, kStateCount);
    
    // TODO: Only need to tranpose sub-matrix of trueStateCount
    transposeSquareMatrix(Evec, kStateCount);
    
#ifdef DOUBLE_PRECISION
    memcpy(Eval, inEigenValues, SIZE_REAL * STATE_COUNT);
#else
    MEMCPY(Eval, inEigenValues, STATE_COUNT, REAL);
#endif
    
#ifdef DEBUG_BEAGLE
#ifdef DOUBLE_PRECISION
    printfVectorD(Eval, kStateCount);
    printfVectorD(Evec, kMatrixSize);
    printfVectorD(Ievc, kStateCount * kStateCount);
#else
    printfVectorF(Eval, kStateCount);
    printfVectorF(Evec, kMatrixSize);
    printfVectorF(Ievc, kStateCount * kStateCount);
#endif
#endif
    
    // Copy to CUDA device
    cudaMemcpy(dIevc, Ievc, SIZE_REAL * kMatrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dEvec, Evec, SIZE_REAL * kMatrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dEigenValues, Eval, SIZE_REAL * kStateCount, cudaMemcpyHostToDevice);
    
#ifdef DEBUG_BEAGLE
    printfCudaVector(dEigenValues, kStateCount);
    printfCudaVector(dEvec, kMatrixSize);
    printfCudaVector(dIevc, kStateCount * kStateCount);
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
    
    for (int i = 0; i < count; i++) {
        hPtrQueue[i] = dMatrices[0][probabilityIndices[i]];
        hDistanceQueue[i] = (REAL) edgeLengths[i];
    }
    
    cudaMemcpy(dDistanceQueue, hDistanceQueue, SIZE_REAL * count, cudaMemcpyHostToDevice);
    cudaMemcpy(dPtrQueue, hPtrQueue, sizeof(REAL*) * count, cudaMemcpyHostToDevice);
    
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
    if (doRescaling == 0) // Forces rescaling on first computation
        doRescaling = rescale;
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
                                                           kPatternCount, categoryCount,
                                                           doRescaling);
            } else {
                nativeGPUStatesPartialsPruningDynamicScaling(tipStates1, partials2, partials3,
                                                             matrices1, matrices2, scalingFactors,
                                                             kPatternCount, categoryCount,
                                                             doRescaling);
                die = 1;
            }
        } else {
            if (tipStates2 != 0) {
                nativeGPUStatesPartialsPruningDynamicScaling(tipStates2, partials1, partials3,
                                                             matrices2, matrices1, scalingFactors,
                                                             kPatternCount, categoryCount,
                                                             doRescaling);
                die = 1;
            } else {
                nativeGPUPartialsPartialsPruningDynamicScaling(partials1, partials2, partials3,
                                                               matrices1, matrices2, scalingFactors,
                                                               kPatternCount, categoryCount,
                                                               doRescaling);
            }
        }
#else
        if (tipStates1 != 0) {
            if (tipStates2 != 0 ) {
                nativeGPUStatesStatesPruning(tipStates1, tipStates2, partials3, matrices1,
                                             matrices2, kPatternCount, categoryCount);
            } else {
                nativeGPUStatesPartialsPruning(tipStates1, partials2, partials3, matrices1,
                                               matrices2, kPatternCount, categoryCount);
                die = 1;
            }
        } else {
            if (tipStates2 != 0) {
                nativeGPUStatesPartialsPruning(tipStates2, partials1, partials3, matrices2,
                                               matrices1, kPatternCount, categoryCount);
                die = 1;
            } else {
                nativeGPUPartialsPartialsPruning(partials1, partials2, partials3, matrices1,
                                                 matrices2, kPatternCount, categoryCount);
            }
        }
#endif // DYNAMIC_SCALING
        
#ifdef DEBUG_BEAGLE
        fprintf(stderr, "kPatternCount = %d\n", kPatternCount);
        fprintf(stderr, "kTruePatternCount = %d\n", kTruePatternCount);
        fprintf(stderr, "categoryCount  = %d\n", categoryCount);
        fprintf(stderr, "partialSize = %d\n", kPartialsSize);
        if (tipStates1)
            printfCudaInt(tipStates1, kPatternCount);
        else
            printfCudaVector(partials1, kPartialsSize);
        if (tipStates2)
            printfCudaInt(tipStates2, kPatternCount);
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
    // TODO: remove this categoryCount hack
    int categoryCount = 1;

#ifdef DOUBLE_PRECISION
	REAL* hWeights = inWeights;
#else
	REAL* hWeights = (REAL*) malloc(count * SIZE_REAL);

	MEMCPY(hWeights, inWeights, count, REAL);
#endif
    REAL* dWeights = allocateGPURealMemory(count);
	cudaMemcpy(dWeights, hWeights, SIZE_REAL * categoryCount, cudaMemcpyHostToDevice);
    

#ifdef DEBUG_FLOW
    fprintf(stderr,"Entering updateRootFreqencies\n");
#endif
    
#ifdef DEBUG_BEAGLE
    printfVectorD(inStateFrequencies, kStateCount);
#endif
    
#ifdef DOUBLE_PRECISION
    memcpy(hFrequenciesCache, inStateFrequencies, kStateCount * SIZE_REAL);
#else
    MEMCPY(hFrequenciesCache, inStateFrequencies, kStateCount, REAL);
#endif
    cudaMemcpy(dFrequencies, hFrequenciesCache, SIZE_REAL * kStateCount,
               cudaMemcpyHostToDevice);
#ifdef DEBUG_FLOW
    fprintf(stderr, "Exiting updateRootFrequencies\n");
#endif


#ifdef DEBUG_FLOW
    fprintf(stderr, "Entering calculateLogLikelihoods\n");
#endif

    if (count == 1) {   
        const int rootNodeIndex = bufferIndices[0];
        
    #ifdef DYNAMIC_SCALING
        if (doRescaling) {
            // Construct node-list for scalingFactors
            int n;
            int length = kBufferCount - kTipCount;
            for(n = 0; n < length; n++)
                hPtrQueue[n] = dScalingFactors[0][n + kTipCount];
            
            cudaMemcpy(dPtrQueue, hPtrQueue, sizeof(REAL*) * length, cudaMemcpyHostToDevice);
            
            // Computer scaling factors at the root
            nativeGPUComputeRootDynamicScaling(dPtrQueue, dRootScalingFactors, length,
                                               kPatternCount);
        }
        
        doRescaling = 0;
        
        nativeGPUIntegrateLikelihoodsDynamicScaling(dIntegrationTmp, dPartials[0][rootNodeIndex],
                                                    dWeights, dFrequencies,
                                                    dRootScalingFactors, kPatternCount,
                                                    categoryCount, kBufferCount);
    #else
        nativeGPUIntegrateLikelihoods(dIntegrationTmp, dPartials[0][rootNodeIndex],
                                      dWeights, dFrequencies, kPatternCount,
                                      categoryCount);
    #endif // DYNAMIC_SCALING
        
    #ifdef DOUBLE_PRECISION
        cudaMemcpy(outLogLikelihoods, dIntegrationTmp, SIZE_REAL * kTruePatternCount,
                   cudaMemcpyDeviceToHost);
    #else
        cudaMemcpy(hLogLikelihoodsCache, dIntegrationTmp, SIZE_REAL * kTruePatternCount,
                   cudaMemcpyDeviceToHost);
        MEMCPY(outLogLikelihoods, hLogLikelihoodsCache, kTruePatternCount, double);
    #endif
        
    #ifdef DEBUG
        printf("logLike = ");
        printfVectorD(outLogLikelihoods, kTruePatternCount);
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
    // integrationTmp (kPatternCount)
    
    assert(false);
}

void BeagleCUDAImpl::freeTmpPartialsOrStates() {
    int i;
    for (i = 0; i < kTipCount; i++) {
        free(hTmpPartials[i]);
        free(hTmpStates[i]);
    }
    
    free(hTmpPartials);
    free(hTmpStates);
    free(hPartialsCache);
    free(hStatesCache);
}

void BeagleCUDAImpl::freeNativeMemory() {
    int i;
    for (i = 0; i < kBufferCount; i++) {
        freeGPUMemory(dPartials[0][i]);
        freeGPUMemory(dPartials[1][i]);
#ifdef DYNAMIC_SCALING
        freeGPUMemory(dScalingFactors[0][i]);
        freeGPUMemory(dScalingFactors[1][i]);
#endif
        freeGPUMemory(dMatrices[0][i]);
        freeGPUMemory(dMatrices[1][i]);
        freeGPUMemory(dStates[i]);
    }
    
    freeGPUMemory(dEvec);
    freeGPUMemory(dIevc);
    
    free(dPartials[0]);
    free(dPartials[1]);
    free(dPartials);
    
#ifdef DYNAMIC_SCALING
    free(dScalingFactors[0]);
    free(dScalingFactors[1]);
    free(dScalingFactors);
#endif
    
    free(dMatrices[0]);
    free(dMatrices[1]);
    free(dMatrices);
    
    free(dStates);
    
    freeGPUMemory(dNodeIndices);
    free(hNodeIndices);
    free(hDependencies);
    freeGPUMemory(dBranchLengths);
    
    freeGPUMemory(dIntegrationTmp);
    
    free(hDistanceQueue);
    free(hPtrQueue);
    freeGPUMemory(dDistanceQueue);
    freeGPUMemory(dPtrQueue);
    
    // TODO: Free all caches
    free(hPartialsCache);
    free(hStatesCache);
}

void BeagleCUDAImpl::loadTipPartialsOrStates() {
    for (int i = 0; i < kTipCount   ; i++) {
        if (hTmpStates[i] == 0)
            cudaMemcpy(dPartials[0][i], hTmpPartials[i], SIZE_REAL * kPartialsSize,
                       cudaMemcpyHostToDevice);
        else
            cudaMemcpy(dStates[i], hTmpStates[i], SIZE_INT * kPatternCount, cudaMemcpyHostToDevice);
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
        if (impl->initialize(tipCount, partialsBufferCount, compactBufferCount, stateCount,
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
