
/*
 *  BeagleGPUImpl.cpp
 *  BEAGLE
 *
 * Copyright 2009 Phylogenetic Likelihood Working Group
 *
 * This file is part of BEAGLE.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
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
#include <vector>

#include "libhmsbeagle/beagle.h"
#include "libhmsbeagle/GPU/GPUImplDefs.h"
#include "libhmsbeagle/GPU/GPUImplHelper.h"
#include "libhmsbeagle/GPU/KernelLauncher.h"
#include "libhmsbeagle/GPU/GPUInterface.h"
#include "libhmsbeagle/GPU/Precision.h"
#include "BeagleGPUImpl.h"

namespace beagle {
namespace gpu {

#ifdef CUDA
    namespace cuda {
#else
    namespace opencl {
#endif

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

    dMultipleDerivatives = (GPUPtr)NULL;
    dMultipleDerivativeSum = (GPUPtr)NULL;

    dPatternWeights = (GPUPtr)NULL;

    dBranchLengths = (GPUPtr)NULL;

    dDistanceQueue = (GPUPtr)NULL;

    dPtrQueue = (GPUPtr)NULL;

    dDerivativeQueue = (GPUPtr)NULL;

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
            if (hCategoryRates[i] != NULL) {
                gpu->FreeHostMemory(hCategoryRates[i]);
            }
        }

        // TODO: free subpointers
        gpu->FreeMemory(dMatrices[0]);
        gpu->FreeMemory(dEigenValues[0]); // TODO Here is where my Mac / Intel-GPU are throwing bad-exception
        gpu->FreeMemory(dEvec[0]);        // TODO Should be save and then release just d*Origin?
        gpu->FreeMemory(dIevc[0]);
        gpu->FreeMemory(dWeights[0]);
        gpu->FreeMemory(dFrequencies[0]);


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

        if (kPartitionsInitialised) {
            free(hPatternPartitions);
            free(hPatternPartitionsStartPatterns);
            free(hIntegratePartitionsStartBlocks);
            free(hPatternPartitionsStartBlocks);
            free(hIntegratePartitionOffsets);
            if (kPatternsReordered) {
                free(hPatternsNewOrder);
                gpu->FreeMemory(dPatternsNewOrder);
                free(hTipOffsets);
                gpu->FreeMemory(dTipOffsets);
                gpu->FreeMemory(dTipTypes);
                gpu->FreeMemory(dPatternWeightsSort);
                if (kCompactBufferCount > 0) {
                    free(dStatesSort);
                    gpu->FreeMemory(dStatesSortOrigin);
                }

            }
        }

        if (kUsingMultiGrid || kPartitionsInitialised) {
        #ifdef FW_OPENCL
            gpu->UnmapMemory(dPartialsPtrs, hPartialsPtrs);
        #else
            gpu->FreeHostMemory(hPartialsPtrs);
            // gpu->FreePinnedHostMemory(hPartialsPtrs);
        #endif
            gpu->FreeMemory(dPartialsPtrs);
            // gpu->FreeMemory(dPartitionOffsets);
            free(hPartitionOffsets);
            free(hGridOpIndices);
        }

        gpu->FreeMemory(dPartialsOrigin);

        if (kCompactBufferCount > 0)
            gpu->FreeMemory(dStatesOrigin);

        gpu->FreeMemory(dIntegrationTmp);
        gpu->FreeMemory(dPartialsTmp);
        gpu->FreeMemory(dSumLogLikelihood);

        if (kDerivBuffersInitialised) {
            gpu->FreeMemory(dSumFirstDeriv);
            gpu->FreeMemory(dFirstDerivTmp);
            gpu->FreeMemory(dOutFirstDeriv);
            gpu->FreeMemory(dSumSecondDeriv);
            gpu->FreeMemory(dSecondDerivTmp);
            gpu->FreeMemory(dOutSecondDeriv);
        }

        if (kMultipleDerivativesLength > 0) {
            gpu->FreeMemory(dMultipleDerivatives);
            gpu->FreeMemory(dMultipleDerivativeSum);
        }

        gpu->FreeMemory(dPatternWeights);

        gpu->FreeMemory(dBranchLengths);

        gpu->FreeMemory(dDistanceQueue);

        gpu->FreeMemory(dPtrQueue);

        gpu->FreeMemory(dDerivativeQueue);

        gpu->FreeMemory(dMaxScalingFactors);
        gpu->FreeMemory(dIndexMaxScalingFactors);

        if (kFlags & BEAGLE_FLAG_SCALING_AUTO) {
            gpu->FreeMemory(dAccumulatedScalingFactors);
        }

        free(dEigenValues);
        free(dEvec);
        free(dIevc);

        free(hCategoryRates);

        free(dWeights);
        free(dFrequencies);

        free(dScalingFactors);

        free(dStates);

        free(dPartials);
        free(dMatrices);

        free(dCompactBuffers);
        free(dTipPartialsBuffers);

        free(hStreamIndices);

        free(hPartialsOffsets);
        free(hStatesOffsets);

        gpu->FreeHostMemory(hPtrQueue);

        gpu->FreeHostMemory(hDerivativeQueue);

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
                                  int globalResourceNumber,
                                  int pluginResourceNumber,
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

    kExtraMatrixCount = 0;

    kPartitionCount = 1;
    kMaxPartitionCount = kPartitionCount;
    kPartitionsInitialised = false;
    kPatternsReordered = false;

    resourceNumber = globalResourceNumber;

    kTipPartialsBufferCount = kTipCount - kCompactBufferCount;
    kBufferCount = kPartialsBufferCount + kCompactBufferCount;

    kInternalPartialsBufferCount = kBufferCount - kTipCount;

    if (kStateCount <= 4) {
        kPaddedStateCount = 4;
    } else if (kStateCount <= 16) {
        kPaddedStateCount = 16;
    } else if (kStateCount <= 32) {
        kPaddedStateCount = 32;
    } else if (kStateCount <= 48) {
        kPaddedStateCount = 48;
    } else if (kStateCount <= 64) {
        kPaddedStateCount = 64;
    } else if (kStateCount <= 80) {
        kPaddedStateCount = 80;
    } else if (kStateCount <= 128) {
        kPaddedStateCount = 128;
    } else if (kStateCount <= 192) {
        kPaddedStateCount = 192;
    } else if (kStateCount <= 256) {
        kPaddedStateCount = 256;
    } else {
        kPaddedStateCount = kStateCount + kStateCount % 16;
    }

    gpu = new GPUInterface();

    gpu->Initialize();

    int numDevices = 0;
    numDevices = gpu->GetDeviceCount();
    if (numDevices == 0) {
        fprintf(stderr, "Error: No GPU devices\n");
        return BEAGLE_ERROR_NO_RESOURCE;
    }
    if (pluginResourceNumber > numDevices) {
        fprintf(stderr,"Error: Trying to initialize device # %d (which does not exist)\n",resourceNumber);
        return BEAGLE_ERROR_NO_RESOURCE;
    }

    int paddedPatterns = 0;
    // Make sure that kPaddedPatternCount + paddedPatterns is multiple of 4 for DNA model
    if (kPaddedStateCount == 4 && kPatternCount % 4 != 0)
        paddedPatterns = 4 - kPatternCount % 4;
    // TODO Should do something similar for 4 < kStateCount <= 8 as well

    bool CPUImpl = false;

#ifdef BEAGLE_DEBUG_OPENCL_CORES
    gpu->CreateDevice(pluginResourceNumber);
#endif

    kDeviceType = gpu->GetDeviceTypeFlag(pluginResourceNumber);
    kDeviceCode = gpu->GetDeviceImplementationCode(pluginResourceNumber);

#ifdef FW_OPENCL

    // TODO: Apple OpenCL on CPU for state count > 128
    if (kDeviceCode == BEAGLE_OPENCL_DEVICE_APPLE_CPU && kPaddedStateCount > 128) {
        return BEAGLE_ERROR_NO_IMPLEMENTATION;
    }

    // TODO: AMD GPU implementation for high state and category counts
    if ((kDeviceCode == BEAGLE_OPENCL_DEVICE_APPLE_AMD_GPU ||
        kDeviceCode == BEAGLE_OPENCL_DEVICE_AMD_GPU) &&
        ((kPaddedStateCount > 64 && kCategoryCount > 2) ||
          (kPaddedStateCount == 192 && kCategoryCount > 1))) {
        return BEAGLE_ERROR_NO_IMPLEMENTATION;
    }

    if (kDeviceCode == BEAGLE_OPENCL_DEVICE_INTEL_CPU ||
        kDeviceCode == BEAGLE_OPENCL_DEVICE_INTEL_MIC ||
        kDeviceCode == BEAGLE_OPENCL_DEVICE_AMD_CPU ||
        kDeviceCode == BEAGLE_OPENCL_DEVICE_APPLE_CPU) {

        CPUImpl = true;

        int patternBlockSize = 0;
        int id = (kFlags & BEAGLE_FLAG_PRECISION_DOUBLE ?
                  kPaddedStateCount : (-1 * kPaddedStateCount));

        switch(id) {
            case   -4: patternBlockSize = PATTERN_BLOCK_SIZE_DP_4_CPU; break;
            case  -16: patternBlockSize = PATTERN_BLOCK_SIZE_DP_16;    break;
            case  -32: patternBlockSize = PATTERN_BLOCK_SIZE_DP_32;    break;
            case  -48: patternBlockSize = PATTERN_BLOCK_SIZE_DP_48;    break;
            case  -64: patternBlockSize = PATTERN_BLOCK_SIZE_DP_64;    break;
            case  -80: patternBlockSize = PATTERN_BLOCK_SIZE_DP_80;    break;
            case -128: patternBlockSize = PATTERN_BLOCK_SIZE_DP_128;   break;
            case -192: patternBlockSize = PATTERN_BLOCK_SIZE_DP_192;   break;
            case    4: patternBlockSize = PATTERN_BLOCK_SIZE_SP_4_CPU; break;
            case   16: patternBlockSize = PATTERN_BLOCK_SIZE_SP_16;    break;
            case   32: patternBlockSize = PATTERN_BLOCK_SIZE_SP_32;    break;
            case   48: patternBlockSize = PATTERN_BLOCK_SIZE_SP_48;    break;
            case   64: patternBlockSize = PATTERN_BLOCK_SIZE_SP_64;    break;
            case   80: patternBlockSize = PATTERN_BLOCK_SIZE_SP_80;    break;
            case  128: patternBlockSize = PATTERN_BLOCK_SIZE_SP_128;   break;
            case  192: patternBlockSize = PATTERN_BLOCK_SIZE_SP_192;   break;
        }

        // pad patterns for CPU/MIC implementation
        if (patternBlockSize != 0 && kPatternCount % patternBlockSize) {
            paddedPatterns = patternBlockSize - (kPatternCount % patternBlockSize);
        }
    }
#endif

    kPaddedPatternCount = kPatternCount + paddedPatterns;

    kResultPaddedPatterns = 0;

    if (!CPUImpl) {
        int patternBlockSizeFour = (kFlags & BEAGLE_FLAG_PRECISION_DOUBLE ? PATTERN_BLOCK_SIZE_DP_4 : PATTERN_BLOCK_SIZE_SP_4);
        if (kPaddedStateCount == 4 && kPaddedPatternCount % patternBlockSizeFour != 0)
            kResultPaddedPatterns = patternBlockSizeFour - kPaddedPatternCount % patternBlockSizeFour;
    }

#ifdef BEAGLE_DEBUG_VALUES
    printf("kPatternCount %d, paddedPatterns %d, kResultPaddedPatterns %d, kPaddedPatternCount %d\n", kPatternCount, paddedPatterns, kResultPaddedPatterns, kPaddedPatternCount);
#endif

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

    if (kDeviceCode == BEAGLE_OPENCL_DEVICE_APPLE_CPU)
        kFlags |= BEAGLE_FLAG_PARALLELOPS_STREAMS;
    else if (requirementFlags & BEAGLE_FLAG_PARALLELOPS_STREAMS || preferenceFlags & BEAGLE_FLAG_PARALLELOPS_STREAMS)
        kFlags |= BEAGLE_FLAG_PARALLELOPS_STREAMS;
    else if (requirementFlags & BEAGLE_FLAG_PARALLELOPS_GRID || preferenceFlags & BEAGLE_FLAG_PARALLELOPS_GRID)
        kFlags |= BEAGLE_FLAG_PARALLELOPS_GRID;

    if (preferenceFlags & BEAGLE_FLAG_COMPUTATION_ASYNCH || requirementFlags & BEAGLE_FLAG_COMPUTATION_ASYNCH) {
        kFlags |= BEAGLE_FLAG_COMPUTATION_ASYNCH;
    } else {
        kFlags |= BEAGLE_FLAG_COMPUTATION_SYNCH;
    }

    if (preferenceFlags & BEAGLE_FLAG_PREORDER_TRANSPOSE_AUTO || requirementFlags & BEAGLE_FLAG_PREORDER_TRANSPOSE_AUTO) {
        kFlags |= BEAGLE_FLAG_PREORDER_TRANSPOSE_AUTO;
    } else {
        kFlags |= BEAGLE_FLAG_PREORDER_TRANSPOSE_MANUAL;
    }

    Real r = 0;
    modifyFlagsForPrecision(&kFlags, r);

    kSumSitesBlockSize = (kFlags & BEAGLE_FLAG_PRECISION_DOUBLE ? SUM_SITES_BLOCK_SIZE_DP : SUM_SITES_BLOCK_SIZE_SP);
    kSumSitesBlockCount = kPatternCount / kSumSitesBlockSize;
    if (kPatternCount % kSumSitesBlockSize != 0)
        kSumSitesBlockCount += 1;

    kPartialsSize = kPaddedPatternCount * kPaddedStateCount * kCategoryCount;
    kMatrixSize = kPaddedStateCount * kPaddedStateCount;

    if (kFlags & BEAGLE_FLAG_EIGEN_COMPLEX)
        kEigenValuesSize = 2 * kPaddedStateCount;
    else
        kEigenValuesSize = kPaddedStateCount;

    kLastCompactBufferIndex = -1;
    kLastTipPartialsBufferIndex = -1;

    // TODO: recompiling kernels for every instance, probably not ideal
    gpu->SetDevice(pluginResourceNumber, kPaddedStateCount, kCategoryCount,
                   kPaddedPatternCount, kPatternCount, kTipCount, kFlags);

#ifdef FW_OPENCL
    kFlags |= gpu->GetDeviceTypeFlag(pluginResourceNumber);
#endif

    int ptrQueueLength = kMatrixCount * kCategoryCount * 3 * 3; // first '3' for derivatives, last '3' is for 3 ops for uTMWMM
    if (kInternalPartialsBufferCount > ptrQueueLength)
        ptrQueueLength = kInternalPartialsBufferCount;

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

    #ifdef CUDA
        size_t availableMem = gpu->GetAvailableMemory();
    #ifdef BEAGLE_DEBUG_VALUES
        fprintf(stderr, "     needed memory: %f MB\n", neededMemory/1000.0/1000);
        fprintf(stderr, "  available memory: %f MB\n", availableMem/1000.0/1000);
    #endif
        // TODO: fix memory check on CUDA and implement for OpenCL
        // if (availableMem < neededMemory)
        //     return BEAGLE_ERROR_OUT_OF_MEMORY;
    #endif

    kernels = new KernelLauncher(gpu);

    // TODO: only allocate if necessary on the fly
    hWeightsCache = (Real*) gpu->CallocHost(kCategoryCount, sizeof(Real));
    hFrequenciesCache = (Real*) gpu->CallocHost(kPaddedStateCount, sizeof(Real));
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

    size_t ptrIncrement = gpu->AlignMemOffset(kMatrixSize * kCategoryCount * sizeof(Real));
    kIndexOffsetMat = ptrIncrement/sizeof(Real);
    GPUPtr dMatricesOrigin = gpu->AllocateMemory(kMatrixCount * ptrIncrement);
    for (int i = 0; i < kMatrixCount; i++) {
        dMatrices[i] = gpu->CreateSubPointer(dMatricesOrigin, ptrIncrement*i, ptrIncrement);
    }

    if (kScaleBufferCount > 0) {
        if (kFlags & BEAGLE_FLAG_SCALING_AUTO) {
            dScalingFactors = (GPUPtr*) malloc(sizeof(GPUPtr) * kScaleBufferCount);
            ptrIncrement = gpu->AlignMemOffset(kScaleBufferSize * sizeof(signed char)); // TODO: char won't work for double-precision
            GPUPtr dScalingFactorsOrigin =  gpu->AllocateMemory(ptrIncrement * kScaleBufferCount);
            for (int i=0; i < kScaleBufferCount; i++)
                dScalingFactors[i] = gpu->CreateSubPointer(dScalingFactorsOrigin, ptrIncrement*i, ptrIncrement);
        } else if (kFlags & BEAGLE_FLAG_SCALING_DYNAMIC) {
#ifdef CUDA
            dScalingFactors = (GPUPtr*) calloc(sizeof(GPUPtr), kScaleBufferCount);
            dScalingFactorsMaster = (GPUPtr*) calloc(sizeof(GPUPtr), kScaleBufferCount);
            hRescalingTrigger = (int*) gpu->AllocatePinnedHostMemory(sizeof(int), false, true);
            dRescalingTrigger = gpu->GetDeviceHostPointer((void*) hRescalingTrigger);
#else
            return BEAGLE_ERROR_NO_IMPLEMENTATION;
#endif
        } else {
            // allocating extra buffer for zeroes to use with partitioned problems and mixed scaling on/off
            dScalingFactors = (GPUPtr*) malloc(sizeof(GPUPtr) * (kScaleBufferCount + 1));
            ptrIncrement = gpu->AlignMemOffset(kScaleBufferSize * sizeof(Real));
            kScaleBufferSize = ptrIncrement / sizeof(Real);
            GPUPtr dScalingFactorsOrigin = gpu->AllocateMemory(ptrIncrement * (kScaleBufferCount + 1));
            for (int i=0; i < (kScaleBufferCount + 1); i++) {
                dScalingFactors[i] = gpu->CreateSubPointer(dScalingFactorsOrigin, ptrIncrement*i, ptrIncrement);
            }
            Real* zeroes = (Real*) gpu->CallocHost(sizeof(Real), kPaddedPatternCount);
            // Fill with zeroes
            gpu->MemcpyHostToDevice(dScalingFactors[kScaleBufferCount], zeroes,
                                    sizeof(Real) * kPaddedPatternCount);
            gpu->FreeHostMemory(zeroes);
        }
    }

    ptrIncrement = gpu->AlignMemOffset(kMatrixSize * sizeof(Real));
    kEvecOffset  = ptrIncrement/sizeof(Real);
    GPUPtr dEvecOrigin = gpu->AllocateMemory(kEigenDecompCount * ptrIncrement);
    GPUPtr dIevcOrigin = gpu->AllocateMemory(kEigenDecompCount * ptrIncrement);
    for(int i=0; i<kEigenDecompCount; i++) {
        dEvec[i] = gpu->CreateSubPointer(dEvecOrigin, ptrIncrement*i, ptrIncrement);
        dIevc[i] = gpu->CreateSubPointer(dIevcOrigin, ptrIncrement*i, ptrIncrement);
    }

    ptrIncrement = gpu->AlignMemOffset(kEigenValuesSize * sizeof(Real));
    kEvalOffset  = ptrIncrement/sizeof(Real);
    GPUPtr dEigenValuesOrigin = gpu->AllocateMemory(kEigenDecompCount * ptrIncrement);
    for(int i=0; i<kEigenDecompCount; i++) {
        dEigenValues[i] = gpu->CreateSubPointer(dEigenValuesOrigin, ptrIncrement*i, ptrIncrement);
    }

    ptrIncrement = gpu->AlignMemOffset(kCategoryCount * sizeof(Real));
    kWeightsOffset = ptrIncrement/sizeof(Real);
    GPUPtr dWeightsOrigin = gpu->AllocateMemory(kEigenDecompCount * ptrIncrement);
    for(int i=0; i<kEigenDecompCount; i++) {
        dWeights[i] = gpu->CreateSubPointer(dWeightsOrigin, ptrIncrement*i, ptrIncrement);
    }

    ptrIncrement = gpu->AlignMemOffset(kPaddedStateCount * sizeof(Real));
    kFrequenciesOffset = ptrIncrement/sizeof(Real);
    GPUPtr dFrequenciesOrigin = gpu->AllocateMemory(kEigenDecompCount * ptrIncrement);
    for(int i=0; i<kEigenDecompCount; i++) {
        dFrequencies[i] = gpu->CreateSubPointer(dFrequenciesOrigin, ptrIncrement*i, ptrIncrement);
    }


    dIntegrationTmp = gpu->AllocateMemory((kPaddedPatternCount + kResultPaddedPatterns) * sizeof(Real));

    dPatternWeights = gpu->AllocateMemory(kPatternCount * sizeof(Real));

    dSumLogLikelihood = gpu->AllocateMemory(kSumSitesBlockCount * sizeof(Real));

    dPartialsTmp = gpu->AllocateMemory(kPartialsSize * sizeof(Real));

    kDerivBuffersInitialised = false;

    kMultipleDerivativesLength = 0;

    int bufferCountTotal = kBufferCount;
    int partialsBufferCountTotal = kPartialsBufferCount;

    // for potential partitioning reorder, TODO: allocate only when needed
    if (partialsBufferCountTotal < 2*kTipPartialsBufferCount) {
        partialsBufferCountTotal = 2*kTipPartialsBufferCount;
        if (bufferCountTotal < 2*kTipPartialsBufferCount) {
            bufferCountTotal = 2*kTipPartialsBufferCount;
        }
    }


    // Fill with 0s so 'free' does not choke if unallocated
    dPartials = (GPUPtr*) calloc(sizeof(GPUPtr), bufferCountTotal);

    ptrIncrement = gpu->AlignMemOffset(kPartialsSize * sizeof(Real));
    GPUPtr dPartialsTmpOrigin = gpu->AllocateMemory(partialsBufferCountTotal * ptrIncrement);
    dPartialsOrigin = gpu->CreateSubPointer(dPartialsTmpOrigin, 0, ptrIncrement);
    hPartialsOffsets = (unsigned int*) calloc(sizeof(unsigned int), bufferCountTotal);
    kIndexOffsetPat = gpu->AlignMemOffset(kPartialsSize * sizeof(Real)) / sizeof(Real);

    size_t ptrIncrementStates = gpu->AlignMemOffset(kPaddedPatternCount * sizeof(int));
    GPUPtr dStatesTmpOrigin;
    if (kCompactBufferCount > 0) {
        dStatesTmpOrigin = gpu->AllocateMemory(kCompactBufferCount * ptrIncrementStates);
        dStatesOrigin = gpu->CreateSubPointer(dStatesTmpOrigin, 0, ptrIncrementStates);
    } else {
        dStatesOrigin = (GPUPtr) NULL;
    }
    // Internal nodes have 0s so partials are used
    dStates = (GPUPtr*) calloc(sizeof(GPUPtr), kBufferCount);
    hStatesOffsets = (unsigned int*) calloc(sizeof(unsigned int), kTipCount);
    kIndexOffsetStates = gpu->AlignMemOffset(kPaddedPatternCount * sizeof(int)) / sizeof(int);

    dCompactBuffers = (GPUPtr*) malloc(sizeof(GPUPtr) * kCompactBufferCount);
    dTipPartialsBuffers = (GPUPtr*) malloc(sizeof(GPUPtr) * kTipPartialsBufferCount);

    hStreamIndices = (int*) malloc(sizeof(int) * kBufferCount);

    for (int i = 0; i < bufferCountTotal; i++) {
        if (i < kTipCount) { // For the tips
            if (i < kCompactBufferCount) {
                dCompactBuffers[i] = gpu->CreateSubPointer(dStatesTmpOrigin, ptrIncrementStates*i, ptrIncrementStates);
            }
            if (i < kTipPartialsBufferCount) {
                dTipPartialsBuffers[i] = gpu->CreateSubPointer(dPartialsTmpOrigin, ptrIncrement*i, ptrIncrement);
            }
        } else {
            int partialsSubIndex = i - (kTipCount - kTipPartialsBufferCount);
            dPartials[i] = gpu->CreateSubPointer(dPartialsTmpOrigin, ptrIncrement*partialsSubIndex, ptrIncrement);
            hPartialsOffsets[i] = kIndexOffsetPat*partialsSubIndex;
        }
    }

    kLastCompactBufferIndex = kCompactBufferCount - 1;
    kLastTipPartialsBufferIndex = kTipPartialsBufferCount - 1;

    // No execution has more no kBufferCount events
    dBranchLengths = gpu->AllocateMemory(kBufferCount * sizeof(Real));

    const int distanceQueueLength = std::max(
        kMatrixCount * kCategoryCount * 2, // for transition matrices
        kMatrixCount + kCategoryCount); // for cross-products

    dDistanceQueue = gpu->AllocateMemory(distanceQueueLength * sizeof(Real));
    hDistanceQueue = (Real*) gpu->MallocHost(distanceQueueLength *  sizeof(Real));
    checkHostMemory(hDistanceQueue);

    dPtrQueue = gpu->AllocateMemory(sizeof(unsigned int) * ptrQueueLength);
    hPtrQueue = (unsigned int*) gpu->MallocHost(sizeof(unsigned int) * ptrQueueLength);
    checkHostMemory(hPtrQueue);

    dDerivativeQueue = gpu->AllocateMemory(sizeof(unsigned int) * kBufferCount * 3);
    hDerivativeQueue = (unsigned int*) gpu->MallocHost(sizeof(unsigned int) * kBufferCount * 3);
    checkHostMemory(hDerivativeQueue);

    if (kPaddedStateCount == 4)
        kSitesPerIntegrateBlock = gpu->kernelResource->patternBlockSize;
    else
        kSitesPerIntegrateBlock = 1;
    kSitesPerBlock = gpu->kernelResource->patternBlockSize;
    if (kDeviceType == BEAGLE_FLAG_PROCESSOR_GPU && kPaddedStateCount == 4)
        kSitesPerBlock *= 4;
    kNumPatternBlocks = (kPaddedPatternCount + kSitesPerBlock - 1) / kSitesPerBlock;
    kPaddedPartitionBlocks = kNumPatternBlocks;
    kMaxPaddedPartitionBlocks = kPaddedPartitionBlocks;
    kPaddedPartitionIntegrateBlocks = (kPaddedPatternCount + kSitesPerIntegrateBlock - 1) / kSitesPerIntegrateBlock;
    kMaxPaddedPartitionIntegrateBlocks = kPaddedPartitionIntegrateBlocks;
    kUsingMultiGrid = false;


    if (kPaddedStateCount == 4 &&
           (kDeviceType == BEAGLE_FLAG_PROCESSOR_CPU ||
#ifdef FW_OPENCL
               kDeviceCode == BEAGLE_OPENCL_DEVICE_AMD_GPU ||
#endif
               kPaddedPatternCount < BEAGLE_MULTI_GRID_MAX ||
               kFlags & BEAGLE_FLAG_PARALLELOPS_GRID) &&
           !(kFlags & BEAGLE_FLAG_PARALLELOPS_STREAMS)) {
        kUsingMultiGrid = true;
        allocateMultiGridBuffers();

        int i;
        for (i=0; i < (kNumPatternBlocks-1); i++) {
            hPartitionOffsets[i*2    ] = i*kSitesPerBlock;
            hPartitionOffsets[i*2 + 1] = (i+1)*kSitesPerBlock;
        }
        hPartitionOffsets[i*2    ] = i*kSitesPerBlock;
        hPartitionOffsets[i*2 + 1] = kPatternCount;

        // size_t transferSize = sizeof(unsigned int) * kNumPatternBlocks * 2;
        // gpu->MemcpyHostToDevice(dPartitionOffsets, hPartitionOffsets, transferSize);
    } else {
        gpu->ResizeStreamCount(kTipCount/2 + 1);
        // gpu->ResizeStreamCount(1);
    }

    hCategoryRates = (double**) calloc(sizeof(double*),kEigenDecompCount); // Keep in double-precision
    hCategoryRates[0] = (double*) gpu->MallocHost(sizeof(double) * kCategoryCount);
    checkHostMemory(hCategoryRates[0]);

    hPatternWeightsCache = (Real*) gpu->MallocHost(sizeof(Real) * kPatternCount);
    checkHostMemory(hPatternWeightsCache);

    dMaxScalingFactors = gpu->AllocateMemory((kPaddedPatternCount + kResultPaddedPatterns) * sizeof(Real));
    dIndexMaxScalingFactors = gpu->AllocateMemory((kPaddedPatternCount + kResultPaddedPatterns) * sizeof(unsigned int));

    if (kFlags & BEAGLE_FLAG_SCALING_AUTO) {
        dAccumulatedScalingFactors = gpu->AllocateMemory(sizeof(int) * kScaleBufferSize);
    }

    kUsingAutoTranspose = (kPaddedStateCount > 4 &&
            kFlags & BEAGLE_FLAG_PREORDER_TRANSPOSE_AUTO);

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving BeagleGPUImpl::createInstance\n");
#endif

    kInitialized = 1;

#ifdef CUDA
#ifdef BEAGLE_DEBUG_VALUES
    gpu->SynchronizeHost();
    size_t usedMemory = availableMem - gpu->GetAvailableMemory();
    fprintf(stderr, "actual used memory: %f MB\n", usedMemory/1000.0/1000);
    fprintf(stderr, "        difference: %f MB\n\n", (usedMemory-neededMemory)/1000.0/1000);
#endif
#endif

    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
void BeagleGPUImpl<BEAGLE_GPU_GENERIC>::allocateMultiGridBuffers() {
    int ptrsPerOp = 8;
    kOpOffsetsSize = sizeof(unsigned int) * kInternalPartialsBufferCount * kPaddedPartitionBlocks * ptrsPerOp;
    // printf("kOpOffsetsSize size = %.2f KB\n\n", (kOpOffsetsSize)/1024.0);
    #ifdef FW_OPENCL
    dPartialsPtrs = (GPUPtr) gpu->AllocatePinnedHostMemory(kOpOffsetsSize, false, false);
    hPartialsPtrs = (unsigned int*)gpu->MapMemory(dPartialsPtrs, kOpOffsetsSize);
    #else
    dPartialsPtrs = gpu->AllocateMemory(kOpOffsetsSize);
    hPartialsPtrs = (unsigned int*) gpu->MallocHost(kOpOffsetsSize);
    // hPartialsPtrs = (unsigned int*) gpu->AllocatePinnedHostMemory(kOpOffsetsSize, true, false);
    #endif
    checkHostMemory(hPartialsPtrs);

    size_t allocationSize = sizeof(unsigned int) * kPaddedPartitionBlocks * 2;
    // dPartitionOffsets = gpu->AllocateMemory(allocationSize);
    hPartitionOffsets = (unsigned int*) malloc(allocationSize);
    checkHostMemory(hPartitionOffsets);

    hGridOpIndices = (int*) malloc(sizeof(int) * kInternalPartialsBufferCount * (ptrsPerOp-2));
}

#ifdef CUDA
template<>
char* BeagleGPUImpl<double>::getInstanceName() {
    return (char*) "CUDA-Double";
}

template<>
char* BeagleGPUImpl<float>::getInstanceName() {
    return (char*) "CUDA-Single";
}
#elif defined(FW_OPENCL)
template<>
char* BeagleGPUImpl<double>::getInstanceName() {
    return (char*) "OpenCL-Double";
}

template<>
char* BeagleGPUImpl<float>::getInstanceName() {
    return (char*) "OpenCL-Single";
}
#endif

BEAGLE_GPU_TEMPLATE
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::getInstanceDetails(BeagleInstanceDetails* returnInfo) {
    if (returnInfo != NULL) {
        returnInfo->resourceNumber = resourceNumber;
        returnInfo->flags = BEAGLE_FLAG_THREADING_NONE |
                            BEAGLE_FLAG_VECTOR_NONE;
        Real r = 0;
        modifyFlagsForPrecision(&(returnInfo->flags), r);

#ifdef CUDA
        kFlags |= BEAGLE_FLAG_FRAMEWORK_CUDA;
        kFlags |= BEAGLE_FLAG_PROCESSOR_GPU;
#elif defined(FW_OPENCL)
        kFlags |= BEAGLE_FLAG_FRAMEWORK_OPENCL;
#endif

        returnInfo->flags |= kFlags;

        returnInfo->implName = getInstanceName();
    }
    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::setCPUThreadCount(int threadCount) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::setCPUThreadCount\n");
#endif

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::setCPUThreadCount\n");
#endif

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
        dStates[tipIndex] = dCompactBuffers[kLastCompactBufferIndex];
        hStatesOffsets[tipIndex] = kIndexOffsetStates * kLastCompactBufferIndex;
        kLastCompactBufferIndex--;
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
            dPartials[tipIndex] = dTipPartialsBuffers[kLastTipPartialsBufferIndex];
            hPartialsOffsets[tipIndex] = kIndexOffsetPat*kLastTipPartialsBufferIndex;
            kLastTipPartialsBufferIndex--;
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
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::setRootPrePartials(const int* bufferIndices,
                       const int* stateFrequenciesIndices,
                       int count){
    return BEAGLE_ERROR_NO_IMPLEMENTATION;
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
            dPartials[bufferIndex] = dTipPartialsBuffers[kLastTipPartialsBufferIndex];
            hPartialsOffsets[bufferIndex] = kIndexOffsetPat*kLastTipPartialsBufferIndex;
            kLastTipPartialsBufferIndex--;
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

    double *outPartialsOffset = outPartials;
    Real *tmpRealPartialsOffset = hPartialsCache;

    for (int c = 0; c < kCategoryCount; c++) {
        for (int i = 0; i < kPatternCount; i++) {
            beagleMemCpy(outPartialsOffset, tmpRealPartialsOffset, kStateCount);
            tmpRealPartialsOffset += kPaddedStateCount;
            outPartialsOffset += kStateCount;
        }
        tmpRealPartialsOffset += kPaddedStateCount * (kPaddedPatternCount - kPatternCount);
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
//      memcpy(Eval+kPaddedStateCount,inEigenValues+kStateCount,sizeof(Real)*kStateCount);
//#else
//    MEMCNV(Eval, inEigenValues, kStateCount, Real);
//    if (kFlags & BEAGLE_FLAG_EIGEN_COMPLEX)
//      MEMCNV((Eval+kPaddedStateCount),(inEigenValues+kStateCount),kStateCount,Real);
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
//  const double* tmpWeights = inCategoryWeights;
//#else
//  Real* tmpWeights = hWeightsCache;
//  MEMCNV(hWeightsCache, inCategoryWeights, kCategoryCount, Real);
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

    memcpy(hCategoryRates[0], categoryRates, sizeof(double) * kCategoryCount);

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::updateCategoryRates\n");
#endif

    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::setCategoryRatesWithIndex(int categoryRatesIndex,
                                                                 const double* inCategoryRates) {

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::setCategoryRatesWithIndex\n");
#endif
    int returnCode = BEAGLE_SUCCESS;

    if (categoryRatesIndex < kEigenDecompCount) {
        if (hCategoryRates[categoryRatesIndex] == NULL) {
            hCategoryRates[categoryRatesIndex] = (double*) gpu->MallocHost(sizeof(double) * kCategoryCount);
            checkHostMemory(hCategoryRates[categoryRatesIndex]);
        }

        const double* categoryRates = inCategoryRates;
        // Can keep these in double-precision until after multiplication by (double) branch-length

        memcpy(hCategoryRates[categoryRatesIndex], categoryRates, sizeof(double) * kCategoryCount);
    } else {
        returnCode = BEAGLE_ERROR_OUT_OF_RANGE;
    }

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::setCategoryRatesWithIndex\n");
#endif

    return returnCode;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::setPatternWeights(const double* inPatternWeights) {

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::setPatternWeights\n");
#endif

//#ifdef DOUBLE_PRECISION
//  const double* tmpWeights = inPatternWeights;
//#else
//  Real* tmpWeights = hPatternWeightsCache;
//  MEMCNV(hPatternWeightsCache, inPatternWeights, kPatternCount, Real);
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
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::setPatternPartitions(int partitionCount,
                                                            const int* inPatternPartitions) {

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::setPatternPartitions\n");
#endif

    if (kPaddedStateCount != 4) {
        return BEAGLE_ERROR_NO_IMPLEMENTATION;
    }

    int returnCode = BEAGLE_SUCCESS;

    assert(partitionCount > 0);
    assert(inPatternPartitions != 0L);

    kPartitionCount = partitionCount;

    if (!kPartitionsInitialised) {
        hPatternPartitions = (int*) malloc(sizeof(int) * kPatternCount);
        checkHostMemory(hPatternPartitions);
    }
    if (!kPartitionsInitialised || kPartitionCount > kMaxPartitionCount) {
        if (kPartitionsInitialised) {
            free(hPatternPartitionsStartPatterns);
        }
        hPatternPartitionsStartPatterns = (int*) malloc(sizeof(int) * (kPartitionCount+1));
        checkHostMemory(hPatternPartitionsStartPatterns);
        free(hStreamIndices);
        hStreamIndices = (int*) malloc(sizeof(int) * kBufferCount * kPartitionCount);
        checkHostMemory(hStreamIndices);

        if ((kPaddedPatternCount >= BEAGLE_MULTI_GRID_MAX || kFlags & BEAGLE_FLAG_PARALLELOPS_STREAMS) && !(kFlags & BEAGLE_FLAG_PARALLELOPS_GRID)) {
            gpu->ResizeStreamCount((kTipCount/2 + 1) * kPartitionCount);
         }
    }

    memcpy(hPatternPartitions, inPatternPartitions, sizeof(int) * kPatternCount);

    bool reorderPatterns = false;
    int contiguousPartitions = 0;
    for (int i=0; i<kPatternCount; i++) {
        if (i > 0 && (hPatternPartitions[i] != hPatternPartitions[i-1])) {
            contiguousPartitions++;
        }
        if (contiguousPartitions != hPatternPartitions[i]) {
            reorderPatterns = true;
            break;
        }
    }

    if (reorderPatterns) {
        returnCode = reorderPatternsByPartition();
    } else {
        int currentPartition = hPatternPartitions[0];
        hPatternPartitionsStartPatterns[currentPartition] = 0;
        for (int i=0; i<kPatternCount; i++) {
            if (hPatternPartitions[i] != currentPartition) {
                currentPartition = hPatternPartitions[i];
                hPatternPartitionsStartPatterns[currentPartition] = i;
            }
        }
        hPatternPartitionsStartPatterns[currentPartition+1] = kPatternCount;
    }

    bool useMultiGrid = true;
    if (!kUsingMultiGrid && ((kPaddedPatternCount/kPartitionCount >= BEAGLE_MULTI_GRID_MAX && kDeviceCode == BEAGLE_CUDA_DEVICE_NVIDIA_GPU) || kFlags & BEAGLE_FLAG_PARALLELOPS_STREAMS) && !(kFlags & BEAGLE_FLAG_PARALLELOPS_GRID)) {
        useMultiGrid = false; // use streams for larger partitions on CUDA
    }

    int totalBlocks = 0;
    for (int i=0; i < kPartitionCount; i++) {
        int partitionStart = hPatternPartitionsStartPatterns[i];
        int partitionEnd = hPatternPartitionsStartPatterns[i+1];
        totalBlocks += (partitionEnd - partitionStart + kSitesPerBlock - 1) / kSitesPerBlock;
    }
    kPaddedPartitionBlocks = totalBlocks;

    if (useMultiGrid) {
        // kernels->SetupPartitioningKernelGrid(kPaddedPartitionBlocks);

        if (kUsingMultiGrid && kPaddedPartitionBlocks > kMaxPaddedPartitionBlocks) {
            #ifdef FW_OPENCL
            gpu->UnmapMemory(dPartialsPtrs, hPartialsPtrs);
            #else
            gpu->FreeHostMemory(hPartialsPtrs);
            #endif
            gpu->FreeMemory(dPartialsPtrs);
            // gpu->FreeMemory(dPartitionOffsets);
            free(hPartitionOffsets);

            allocateMultiGridBuffers();
        } else if (!kUsingMultiGrid) {
            allocateMultiGridBuffers();
            kUsingMultiGrid = true;
        }
    }

    // make sure we always allocate multi-grid buffers because of integration step
    if (!kUsingMultiGrid && !kPartitionsInitialised)
        allocateMultiGridBuffers();

    if (!kPartitionsInitialised || kPartitionCount > kMaxPartitionCount) {
        if (kPartitionsInitialised) {
            free(hPatternPartitionsStartBlocks);
        }
        hPatternPartitionsStartBlocks = (int*) malloc(sizeof(int) * (kPartitionCount+1));
        checkHostMemory(hPatternPartitionsStartBlocks);

    }

    int blockIndex = 0;
    for (int i=0; i < kPartitionCount; i++) {
        hPatternPartitionsStartBlocks[i] = blockIndex;
        int partitionStart = hPatternPartitionsStartPatterns[i];
        int partitionEnd = hPatternPartitionsStartPatterns[i+1];
        int partitionBlocks = (partitionEnd - partitionStart) / kSitesPerBlock;
        for (int j=0; j < partitionBlocks; j++) {
            int blockStart = partitionStart + j*kSitesPerBlock;
            hPartitionOffsets[blockIndex*2    ] = blockStart;
            hPartitionOffsets[blockIndex*2 + 1] = blockStart + kSitesPerBlock;
            blockIndex++;
        }
        int partitionRemainder = (partitionEnd - partitionStart) % kSitesPerBlock;
        if (partitionRemainder != 0) {
            int blockStart = partitionStart + partitionBlocks*kSitesPerBlock;
            hPartitionOffsets[blockIndex*2    ] = blockStart;
            hPartitionOffsets[blockIndex*2 + 1] = blockStart + partitionRemainder;
            blockIndex++;
        }
    }
    hPatternPartitionsStartBlocks[kPartitionCount] = blockIndex;

    // always using 'multi-grid' approach to root integration
    int totalIntegrateBlocks = 0;
    for (int i=0; i < kPartitionCount; i++) {
        int partitionStart = hPatternPartitionsStartPatterns[i];
        int partitionEnd = hPatternPartitionsStartPatterns[i+1];
        totalIntegrateBlocks += (partitionEnd - partitionStart + kSitesPerIntegrateBlock - 1) / kSitesPerIntegrateBlock;
    }
    kPaddedPartitionIntegrateBlocks = totalIntegrateBlocks;
    if (!kPartitionsInitialised || kPaddedPartitionIntegrateBlocks > kMaxPaddedPartitionIntegrateBlocks) {
        if (kPartitionsInitialised) {
            free(hIntegratePartitionOffsets);
        }
        hIntegratePartitionOffsets = (unsigned int*) malloc(sizeof(unsigned int) * kPaddedPartitionIntegrateBlocks * 2);
        checkHostMemory(hIntegratePartitionOffsets);
    }
    if (!kPartitionsInitialised || kPartitionCount > kMaxPartitionCount) {
        if (kPartitionsInitialised) {
            free(hIntegratePartitionsStartBlocks);
        }
        hIntegratePartitionsStartBlocks = (int*) malloc(sizeof(int) * (kPartitionCount+1));
        checkHostMemory(hIntegratePartitionsStartBlocks);
    }
    blockIndex = 0;
    for (int i=0; i < kPartitionCount; i++) {
        hIntegratePartitionsStartBlocks[i] = blockIndex;
        int partitionStart = hPatternPartitionsStartPatterns[i];
        int partitionEnd = hPatternPartitionsStartPatterns[i+1];
        int partitionBlocks = (partitionEnd - partitionStart) / kSitesPerIntegrateBlock;
        for (int j=0; j < partitionBlocks; j++) {
            int blockStart = partitionStart + j*kSitesPerIntegrateBlock;
            hIntegratePartitionOffsets[blockIndex*2    ] = blockStart;
            hIntegratePartitionOffsets[blockIndex*2 + 1] = blockStart + kSitesPerIntegrateBlock;
            blockIndex++;
        }
        int partitionRemainder = (partitionEnd - partitionStart) % kSitesPerIntegrateBlock;
        if (partitionRemainder != 0) {
            int blockStart = partitionStart + partitionBlocks*kSitesPerIntegrateBlock;
            hIntegratePartitionOffsets[blockIndex*2    ] = blockStart;
            hIntegratePartitionOffsets[blockIndex*2 + 1] = blockStart + partitionRemainder;
            blockIndex++;
        }
    }
    hIntegratePartitionsStartBlocks[kPartitionCount] = blockIndex;

    if (kPartitionCount > kMaxPartitionCount) {
        kMaxPartitionCount = kPartitionCount;
    }
    if (kPaddedPartitionBlocks > kMaxPaddedPartitionBlocks) {
        kMaxPaddedPartitionBlocks = kPaddedPartitionBlocks;
    }
    if (kPaddedPartitionIntegrateBlocks > kMaxPaddedPartitionIntegrateBlocks) {
        kMaxPaddedPartitionIntegrateBlocks = kPaddedPartitionIntegrateBlocks;
    }
    kPartitionsInitialised = true;

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::setPatternPartitions\n");
#endif

    return returnCode;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::reorderPatternsByPartition() {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::reorderPatternsByPartition\n");
#endif
    size_t newOrderSize = kPatternCount * sizeof(int);
    size_t tipOffsetsSize = (kTipCount * 2) * sizeof(int);

    if (!kPatternsReordered) {
        hPatternsNewOrder = (int*) malloc(newOrderSize);
        dPatternsNewOrder = gpu->AllocateMemory(newOrderSize);

        int* hTipTypes = (int*) calloc(sizeof(int), kTipCount);
        dTipTypes = gpu->AllocateMemory(kTipCount * sizeof(int));

        dStatesSort = (GPUPtr*) calloc(sizeof(GPUPtr), kTipCount);

        size_t ptrIncrementStates = gpu->AlignMemOffset(kPaddedPatternCount * sizeof(int));
        int lastCompactBufferIndex = kCompactBufferCount - 1;
        if (kCompactBufferCount > 0) {
            dStatesSortOrigin = gpu->AllocateMemory(kCompactBufferCount * ptrIncrementStates);
        } else {
            dStatesSortOrigin = (GPUPtr) NULL;
        }

        int internalBufferIndex = kTipCount;

        hTipOffsets = (int*) calloc(sizeof(int), kTipCount*2);
        for (int i=0; i < kTipCount; i++) {
            if (dStates[i]) {
                hTipTypes[i] = 1;

                hTipOffsets[i] = hStatesOffsets[i];
                dStatesSort[i] = gpu->CreateSubPointer(dStatesSortOrigin,
                                                       ptrIncrementStates * lastCompactBufferIndex,
                                                       ptrIncrementStates);
                hTipOffsets[i+kTipCount] = kIndexOffsetStates * lastCompactBufferIndex;
                lastCompactBufferIndex--;
            } else {
                hTipOffsets[i] = hPartialsOffsets[i];
                hTipOffsets[i+kTipCount] = hPartialsOffsets[internalBufferIndex];
                internalBufferIndex++;
            }
        }

        dTipOffsets  = gpu->AllocateMemory(tipOffsetsSize);
        gpu->MemcpyHostToDevice(dTipOffsets, hTipOffsets, tipOffsetsSize);

        dPatternWeightsSort = gpu->AllocateMemory(kPatternCount * sizeof(Real));

        gpu->MemcpyHostToDevice(dTipTypes, hTipTypes, kTipCount * sizeof(int));
        free(hTipTypes);
    } else {
        return BEAGLE_ERROR_NO_IMPLEMENTATION;
    }

    int* partitionSizes = (int*) malloc(kPartitionCount * sizeof(int));

    for (int i=0; i < kPartitionCount; i++) {
        hPatternPartitionsStartPatterns[i] = 0;
        partitionSizes[i] = 0;
    }

    for (int i=0; i < kPatternCount; i++) {
         hPatternsNewOrder[i] = partitionSizes[hPatternPartitions[i]]++;
    }

    for (int i=0; i < kPartitionCount; i++) {
        for (int j=0; j < i; j++) {
            hPatternPartitionsStartPatterns[i] += partitionSizes[j];
        }
    }
    hPatternPartitionsStartPatterns[kPartitionCount] = kPatternCount;

    for (int i=0; i < kPatternCount; i++) {
        hPatternsNewOrder[i] += hPatternPartitionsStartPatterns[hPatternPartitions[i]];
    }


    int currentPattern = 0;
    for (int i=0; i < kPartitionCount; i++) {
        for (int j=0; j < partitionSizes[i]; j++) {
            hPatternPartitions[currentPattern++] = i;
        }
    }

    gpu->MemcpyHostToDevice(dPatternsNewOrder, hPatternsNewOrder, newOrderSize);

    kernels->ReorderPatterns(dPartialsOrigin, dStatesOrigin, dStatesSortOrigin,
                             dTipOffsets, dTipTypes, dPatternsNewOrder,
                             dPatternWeights, dPatternWeightsSort,
                             kPatternCount, kPaddedPatternCount, kTipCount);

    int ibIndex = kTipCount;
    for (int i=0; i < kTipCount; i++) {
        if (dStates[i]) {
            GPUPtr tmpState      = dStates[i];

            dStates[i]           = dStatesSort[i];
            hStatesOffsets[i]    = hTipOffsets[i+kTipCount];

            dStatesSort[i]         = tmpState;
        } else {
            GPUPtr tmpPartial    = dPartials[i];

            dPartials[i]         = dPartials[ibIndex];
            hPartialsOffsets[i]  = hTipOffsets[i+kTipCount];

            dPartials[ibIndex]   = tmpPartial;
            hPartialsOffsets[ibIndex] = hTipOffsets[i];
            ibIndex++;
        }
        int tmpOffset            = hTipOffsets[i];
        hTipOffsets[i]           = hTipOffsets[i+kTipCount];
        hTipOffsets[i+kTipCount] = tmpOffset;
    }
    gpu->MemcpyHostToDevice(dTipOffsets, hTipOffsets, tipOffsetsSize);

    GPUPtr tmpPtr = dStatesOrigin;
    dStatesOrigin = dStatesSortOrigin;
    dStatesSortOrigin = tmpPtr;

    tmpPtr = dPatternWeights;
    dPatternWeights = dPatternWeightsSort;
    dPatternWeightsSort = tmpPtr;

    free(partitionSizes);

    kPatternsReordered = true;

#ifdef BEAGLE_DEBUG_SYNCH
    gpu->SynchronizeHost();
#endif

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving BeagleGPUImpl::reorderPatternsByPartition\n");
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

    setMatrixBufferImpl(matrixIndex, inMatrix, paddedValue, true);

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::setTransitionMatrix\n");
#endif

    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::setDifferentialMatrix(int matrixIndex,
                                       const double* inMatrix) {

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::setDifferentialMatrix\n");
#endif

    setMatrixBufferImpl(matrixIndex, inMatrix, 0.0, !kUsingAutoTranspose);

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::setDifferentialMatrix\n");
#endif

    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::setMatrixBufferImpl(int matrixIndex,
                                       const double* inMatrix,
                                       double paddedValue,
                                       bool transpose) {

    const double* inMatrixOffset = inMatrix;
    Real* tmpRealMatrixOffset = hMatrixCache;

    for (int l = 0; l < kCategoryCount; l++) {
        Real* transposeOffset = tmpRealMatrixOffset;

        for (int i = 0; i < kStateCount; i++) {
            beagleMemCpy(tmpRealMatrixOffset, inMatrixOffset, kStateCount);
            tmpRealMatrixOffset += kPaddedStateCount;
            inMatrixOffset += kStateCount;
        }

        if (transpose) {
            transposeSquareMatrix(transposeOffset, kPaddedStateCount);
        }
        tmpRealMatrixOffset += (kPaddedStateCount - kStateCount) * kPaddedStateCount;
    }

    // Copy to GPU device
    gpu->MemcpyHostToDevice(dMatrices[matrixIndex], hMatrixCache,
                            sizeof(Real) * kMatrixSize * kCategoryCount);

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

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t Leaving BeagleGPUImpl::convolveTransitionMatrices \n");
#endif

    return returnCode;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::addTransitionMatrices(const int* firstIndices,
                                                             const int* secondIndices,
                                                             const int* resultIndices,
                                                             int matrixCount) {
    return BEAGLE_ERROR_NO_IMPLEMENTATION;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::transposeTransitionMatrices(
        const int* inputIndices,
        const int* resultIndices,
        int matrixCount) {

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t Entering BeagleGPUImpl::transposeTransitionMatrices \n");
#endif

//    if (kPaddedStateCount == 4) {
//#ifdef BEAGLE_DEBUG_FLOW
//        fprintf(stderr, "Transposition not necessary for 4-state-count models.\n");
//#endif
//        return BEAGLE_ERROR_NO_IMPLEMENTATION;
//    }

    if (matrixCount > 0) {

        for (int u = 0; u < matrixCount; u++) {
            if (inputIndices[u] == resultIndices[u]) {

#ifdef BEAGLE_DEBUG_FLOW
                fprintf(stderr, "In-place transposition is not allowed.\n");
#endif

                return BEAGLE_ERROR_GENERAL;
            }
        }

        int totalMatrixCount = matrixCount * kCategoryCount;

        int ptrIndex = 0;
        int indexOffset = kMatrixSize * kCategoryCount;
        int categoryOffset = kMatrixSize;

        for (int i = 0; i < matrixCount; i++) {
            for (int j = 0; j < kCategoryCount; j++) {

                hPtrQueue[ptrIndex] = inputIndices[i] * indexOffset + j * categoryOffset;
                hPtrQueue[ptrIndex + totalMatrixCount] = resultIndices[i] * indexOffset + j * categoryOffset;

                ptrIndex++;
            }
        }

        gpu->MemcpyHostToDevice(dPtrQueue, hPtrQueue, sizeof(unsigned int) * totalMatrixCount * 2);

        kernels->TransposeTransitionMatrices(dMatrices[0], dPtrQueue, totalMatrixCount);
    }

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t Leaving BeagleGPUImpl::transposeTransitionMatrices \n");
#endif

    return BEAGLE_SUCCESS;
}

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

        int categoryOffset = kMatrixSize;

        const double* categoryRates = hCategoryRates[0];

        if (firstDerivativeIndices == NULL && secondDerivativeIndices == NULL) {
            for (int i = 0; i < count; i++) {
                for (int j = 0; j < kCategoryCount; j++) {
                    hPtrQueue[totalCount] = probabilityIndices[i] * kIndexOffsetMat + j * categoryOffset;
                    hDistanceQueue[totalCount] = (Real) (edgeLengths[i] * categoryRates[j]);
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
                    hPtrQueue[ptrIndex] = probabilityIndices[i] * kIndexOffsetMat + j * categoryOffset;
                    hPtrQueue[ptrIndex + totalCount] = firstDerivativeIndices[i] * kIndexOffsetMat + j * categoryOffset;
                    hDistanceQueue[ptrIndex] = (Real) (edgeLengths[i]);
                    hDistanceQueue[ptrIndex + totalCount] = (Real) (categoryRates[j]);
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
                    hPtrQueue[ptrIndex] = probabilityIndices[i] * kIndexOffsetMat + j * categoryOffset;
                    hPtrQueue[ptrIndex + totalCount] = firstDerivativeIndices[i] * kIndexOffsetMat + j * categoryOffset;
                    hPtrQueue[ptrIndex + totalCount*2] = secondDerivativeIndices[i] * kIndexOffsetMat + j * categoryOffset;
                                    hDistanceQueue[ptrIndex] = (Real) (edgeLengths[i]);
                    hDistanceQueue[ptrIndex + totalCount] = (Real) (categoryRates[j]);
                    ptrIndex++;
                }
            }

            gpu->MemcpyHostToDevice(dPtrQueue, hPtrQueue, sizeof(unsigned int) * totalCount * 3);
            gpu->MemcpyHostToDevice(dDistanceQueue, hDistanceQueue, sizeof(Real) * totalCount * 2);

            kernels->GetTransitionProbabilitiesSquareSecondDeriv(dMatrices[0], dPtrQueue, dEvec[eigenIndex], dIevc[eigenIndex],
                                                      dEigenValues[eigenIndex], dDistanceQueue, totalCount);
        }

#ifdef FW_OPENCL
            // todo: unclear why this is necessary to fix numerical instability, investigate further
            if (kDeviceCode == BEAGLE_OPENCL_DEVICE_AMD_GPU && kStateCount != 4) {
                gpu->SynchronizeHost();
            }
#endif

    #ifdef BEAGLE_DEBUG_VALUES
        Real r = 0;
        for (int i = 0; i < count; i++) {
            fprintf(stderr, "dMatrices[probabilityIndices[%d]]  (hDQ = %1.5e, eL = %1.5e) =\n", i,hDistanceQueue[i], edgeLengths[i]);
            gpu->PrintfDeviceVector(dMatrices[probabilityIndices[i]], kMatrixSize * kCategoryCount, r);
            for(int j=0; j<kCategoryCount; j++)
                fprintf(stderr, " %1.5f",categoryRates[j]);
            fprintf(stderr,"\n");
        }
    #endif

    #ifdef BEAGLE_DEBUG_SYNCH
        gpu->SynchronizeHost();
    #endif
    }

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::updateTransitionMatrices\n");
#endif

    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::updateTransitionMatricesWithModelCategories(int* eigenIndices,
                                            const int* probabilityIndices,
                                            const int* firstDerivativeIndices,
                                            const int* secondDerivativeIndices,
                                            const double* edgeLengths,
                                            int count) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\tEntering BeagleGPUImpl::updateTransitionMatrices\n");
#endif

    if (count > 0) {
        // TODO: improve performance of calculation of derivatives and of model cats

        int categoryOffset = kMatrixSize;

        if (firstDerivativeIndices == NULL && secondDerivativeIndices == NULL) {
            for (int i = 0; i < count; i++) {
                hDistanceQueue[i] = (Real) (edgeLengths[i]);
            }
            gpu->MemcpyHostToDevice(dDistanceQueue, hDistanceQueue, sizeof(Real) * count);

            for (int j = 0; j < kCategoryCount; j++) {
                for (int i = 0; i < count; i++) {
                    hPtrQueue[i] = probabilityIndices[i] * kIndexOffsetMat + j * categoryOffset;
                }
                gpu->MemcpyHostToDevice(dPtrQueue, hPtrQueue, sizeof(unsigned int) * count);

                int eigenIndex = eigenIndices[j];
                // Set-up and call GPU kernel
                kernels->GetTransitionProbabilitiesSquare(dMatrices[0], dPtrQueue, dEvec[eigenIndex], dIevc[eigenIndex],
                                                          dEigenValues[eigenIndex], dDistanceQueue, count);
            }

        } else if (secondDerivativeIndices == NULL) {

            for (int i = 0; i < count; i++) {
                hDistanceQueue[i] = (Real) (edgeLengths[i]);
                hDistanceQueue[i + count] = (Real) 1.0;
            }
            gpu->MemcpyHostToDevice(dDistanceQueue, hDistanceQueue, sizeof(Real) * count * 2);

            for (int j = 0; j < kCategoryCount; j++) {
                for (int i = 0; i < count; i++) {
                    hPtrQueue[i] = probabilityIndices[i] * kIndexOffsetMat + j * categoryOffset;
                    hPtrQueue[i + count] = firstDerivativeIndices[i] * kIndexOffsetMat + j * categoryOffset;
                }
                gpu->MemcpyHostToDevice(dPtrQueue, hPtrQueue, sizeof(unsigned int) * count * 2);

                int eigenIndex = eigenIndices[j];
                // Set-up and call GPU kernel
                kernels->GetTransitionProbabilitiesSquareFirstDeriv(dMatrices[0], dPtrQueue, dEvec[eigenIndex], dIevc[eigenIndex],
                                                          dEigenValues[eigenIndex], dDistanceQueue, count);
            }

        } else {

            for (int i = 0; i < count; i++) {
                hDistanceQueue[i] = (Real) (edgeLengths[i]);
                hDistanceQueue[i + count] = (Real) 1.0;
            }
            gpu->MemcpyHostToDevice(dDistanceQueue, hDistanceQueue, sizeof(Real) * count * 2);

            for (int j = 0; j < kCategoryCount; j++) {
                for (int i = 0; i < count; i++) {
                    hPtrQueue[i] = probabilityIndices[i] * kIndexOffsetMat + j * categoryOffset;
                    hPtrQueue[i + count] = firstDerivativeIndices[i] * kIndexOffsetMat + j * categoryOffset;
                    hPtrQueue[i + count*2] = secondDerivativeIndices[i] * kIndexOffsetMat + j * categoryOffset;
                }
                gpu->MemcpyHostToDevice(dPtrQueue, hPtrQueue, sizeof(unsigned int) * count * 3);

                int eigenIndex = eigenIndices[j];
                // Set-up and call GPU kernel
                kernels->GetTransitionProbabilitiesSquareSecondDeriv(dMatrices[0], dPtrQueue, dEvec[eigenIndex], dIevc[eigenIndex],
                                                          dEigenValues[eigenIndex], dDistanceQueue, count);
            }
        }

#ifdef FW_OPENCL
            // todo: unclear why this is necessary to fix numerical instability, investigate further
            if (kDeviceCode == BEAGLE_OPENCL_DEVICE_AMD_GPU && kStateCount != 4) {
                gpu->SynchronizeHost();
            }
#endif

    #ifdef BEAGLE_DEBUG_VALUES
        Real r = 0;
        for (int i = 0; i < count; i++) {
            fprintf(stderr, "dMatrices[probabilityIndices[%d]]  (hDQ = %1.5e, eL = %1.5e) =\n", i,hDistanceQueue[i], edgeLengths[i]);
            gpu->PrintfDeviceVector(dMatrices[probabilityIndices[i]], kMatrixSize * kCategoryCount, r);
            for(int j=0; j<kCategoryCount; j++)
                fprintf(stderr, " %1.5f",categoryRates[j]);
            fprintf(stderr,"\n");
        }
    #endif

    #ifdef BEAGLE_DEBUG_SYNCH
        gpu->SynchronizeHost();
    #endif
    }

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::updateTransitionMatrices\n");
#endif

    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::updateTransitionMatricesWithMultipleModels(const int* eigenIndices,
                                                                                  const int* categoryRateIndices,
                                                                                  const int* probabilityIndices,
                                                                                  const int* firstDerivativeIndices,
                                                                                  const int* secondDerivativeIndices,
                                                                                  const double* edgeLengths,
                                                                                  int count) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::updateTransitionMatricesWithMultipleModels\n");
#endif

    int returnCode = BEAGLE_SUCCESS;

    if (count > 0) {
        int totalCount = 0;

        int categoryOffset = kMatrixSize;

        if (firstDerivativeIndices == NULL && secondDerivativeIndices == NULL) {
            for (int i = 0; i < count; i++) {
                const double* categoryRates = hCategoryRates[categoryRateIndices[i]];
                for (int j = 0; j < kCategoryCount; j++) {
                    hPtrQueue[totalCount*3] = probabilityIndices[i] * kIndexOffsetMat + j * categoryOffset;
                    hPtrQueue[totalCount*3 + 1] = eigenIndices[i] * kEvecOffset;
                    hPtrQueue[totalCount*3 + 2] = eigenIndices[i] * kEvalOffset;
                    hDistanceQueue[totalCount] = (Real) (edgeLengths[i] * categoryRates[j]);
                    totalCount++;
                }
            }

            gpu->MemcpyHostToDevice(dPtrQueue, hPtrQueue, sizeof(unsigned int) * totalCount * 3);
            gpu->MemcpyHostToDevice(dDistanceQueue, hDistanceQueue, sizeof(Real) * totalCount);

            // Set-up and call GPU kernel
            kernels->GetTransitionProbabilitiesSquareMulti(dMatrices[0], dPtrQueue,
                                                           dEvec[0], dIevc[0],
                                                           dEigenValues[0],
                                                           dDistanceQueue, totalCount);

        } else {
            returnCode = BEAGLE_ERROR_NO_IMPLEMENTATION;
        }

    #ifdef BEAGLE_DEBUG_SYNCH
        gpu->SynchronizeHost();
    #endif
    }

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::updateTransitionMatricesWithMultipleModels\n");
#endif

    return returnCode;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::updatePartials(const int* operations,
                                                      int operationCount,
                                                      int cumulativeScalingIndex) {

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::updatePartials\n");
#endif

    bool byPartition = false;
    int returnCode = upPartials(byPartition,
                                operations,
                                operationCount,
                                cumulativeScalingIndex);

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::updatePartials\n");
#endif

    return returnCode;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::updatePrePartials(const int *operations,
                                                         int count,
                                                         int cumulativeScaleIndex) {

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::updatePrePartials\n");
#endif

    int returnCode = BEAGLE_ERROR_GENERAL;

    bool byPartition = false;
    returnCode = upPrePartials(byPartition, operations, count, cumulativeScaleIndex);

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::updatePrePartials\n");
#endif

    return returnCode;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::calculateEdgeDerivative(const int *postBufferIndices,
                                                               const int *preBufferIndices,
                                                               const int rootBufferIndex,
                                                               const int *firstDerivativeIndices,
                                                               const int *secondDerivativeIndices,
                                                               const int categoryWeightsIndex,
                                                               const int categoryRatesIndex,
                                                               const int stateFrequenciesIndex,
                                                               const int *cumulativeScaleIndices,
                                                               int count,
                                                               double *outFirstDerivative,
                                                               double *outDiagonalSecondDerivative) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::calculateEdgeDerivative\n");
#endif

    if (dOutFirstDeriv == (GPUPtr)NULL) {
        dOutFirstDeriv = gpu->AllocateMemory(kPaddedPatternCount * kBufferCount * 2 * sizeof(Real));
    }

    int returnCode = BEAGLE_ERROR_GENERAL;

    returnCode = calcEdgeFirstDerivatives(postBufferIndices, preBufferIndices,
                                          firstDerivativeIndices, &categoryWeightsIndex,
                                          cumulativeScaleIndices, count,
                                          outFirstDerivative, NULL, NULL);
    if (outDiagonalSecondDerivative != NULL) {
        int diagonalSecondDerivativeReturnCode =
                calcEdgeFirstDerivatives(postBufferIndices, preBufferIndices,
                        secondDerivativeIndices, &categoryWeightsIndex,
                        cumulativeScaleIndices, count,
                        outDiagonalSecondDerivative, NULL, NULL);
    }

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::calculateEdgeDerivative\n");
#endif

    return returnCode;
}


BEAGLE_GPU_TEMPLATE
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::calculateEdgeDerivatives(const int *postBufferIndices,
                                                                const int *preBufferIndices,
                                                                const int *derivativeMatrixIndices,
                                                                const int *categoryWeightsIndices,
                                                                const int *categoryRatesIndices,
                                                                const int *cumulativeScaleIndices,
                                                                int count,
                                                                double *outDerivatives,
                                                                double *outSumDerivatives,
                                                                double *outSumSquaredDerivatives) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::calculateEdgeDerivatives\n");
#endif

    if (dOutFirstDeriv == (GPUPtr)NULL) {
        dOutFirstDeriv = gpu->AllocateMemory(kPaddedPatternCount * kBufferCount * 2 * sizeof(Real));
    }

    int returnCode = BEAGLE_ERROR_GENERAL;

    returnCode = calcEdgeFirstDerivatives(postBufferIndices, preBufferIndices,
                                          derivativeMatrixIndices, categoryWeightsIndices,
                                          cumulativeScaleIndices, count,
                                          outDerivatives, outSumDerivatives, outSumSquaredDerivatives);
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::calculateEdgeDerivatives\n");
#endif

    return returnCode;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::updatePartialsByPartition(const int* operations,
                                                                 int operationCount) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::updatePartialsByPartition\n");
#endif

    bool byPartition = true;
    int returnCode = upPartials(byPartition,
                                operations,
                                operationCount,
                                BEAGLE_OP_NONE);

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::updatePartialsByPartition\n");
#endif

    return returnCode;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::updatePrePartialsByPartition(const int* operations,
                                                                    int operationCount) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::updatePrePartialsByPartition\n");
#endif

    bool byPartition = true;
    int returnCode = upPrePartials(byPartition,
                                   operations,
                                   operationCount,
                                   BEAGLE_OP_NONE);

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::updatePrePartialsByPartition\n");
#endif

    return returnCode;
}


BEAGLE_GPU_TEMPLATE
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::upPartials(bool byPartition,
                                                  const int* operations,
                                                  int operationCount,
                                                  int cumulativeScalingIndex) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::upPartials\n");
#endif

    GPUPtr cumulativeScalingBuffer = 0;
    if (cumulativeScalingIndex != BEAGLE_OP_NONE)
        cumulativeScalingBuffer = dScalingFactors[cumulativeScalingIndex];

    int numOps = BEAGLE_OP_COUNT;
    if (byPartition) {
        numOps = BEAGLE_PARTITION_OP_COUNT;
    }

    int gridLaunches = 0;
    int* gridStartOp;
    int* gridOpType;
    int* gridOpBlocks;
    int parentMinIndex = 0;
    int lastStreamIndex = 0;
    int gridOpIndex = 0;

    if (kUsingMultiGrid) {
        gridStartOp  = (int*) malloc(sizeof(int) * (operationCount + 1));
        gridOpType   = (int*) malloc(sizeof(int) * (operationCount + 1));
        gridOpBlocks = (int*) malloc(sizeof(int) * (operationCount + 1));
    }


    int anyRescale = BEAGLE_OP_NONE;
    if (kUsingMultiGrid && (kFlags & BEAGLE_FLAG_SCALING_MANUAL)) {
        for (int op = 0; op < operationCount; op++) {
            const int writeScalingIndex = operations[op * numOps + 1];
            const int readScalingIndex  = operations[op * numOps + 2];
            if (writeScalingIndex >= 0) {
                anyRescale = 1;
                break;
            } else if (readScalingIndex >= 0) {
                anyRescale = 0;
            }
        }
    }

    int streamIndex = -1;
    int waitIndex = -1;
    if (!kUsingMultiGrid || (anyRescale == 1 && kPartitionsInitialised)) {
        gpu->SynchronizeDevice();
        for (int i = 0; i < kBufferCount * kPartitionCount; i++) {
            hStreamIndices[i] = -1;
        }
    }

    for (int op = 0; op < operationCount; op++) {
        const int parIndex = operations[op * numOps];
        const int writeScalingIndex = operations[op * numOps + 1];
        const int readScalingIndex = operations[op * numOps + 2];
        const int child1Index = operations[op * numOps + 3];
        const int child1TransMatIndex = operations[op * numOps + 4];
        const int child2Index = operations[op * numOps + 5];
        const int child2TransMatIndex = operations[op * numOps + 6];
        int currentPartition = 0;
        if (byPartition) {
            currentPartition = operations[op * numOps + 7];
            cumulativeScalingIndex = operations[op * numOps + 8];
            if (cumulativeScalingIndex != BEAGLE_OP_NONE)
                cumulativeScalingBuffer = dScalingFactors[cumulativeScalingIndex];
            else
                cumulativeScalingBuffer = 0;
        }

        if (!kUsingMultiGrid || (anyRescale == 1 && kPartitionsInitialised)) {
            int pOffset = currentPartition * kBufferCount;
            waitIndex = hStreamIndices[child2Index + pOffset];
            if (hStreamIndices[child1Index + pOffset] != -1) {
                hStreamIndices[parIndex + pOffset] = hStreamIndices[child1Index + pOffset];
            } else if (hStreamIndices[child2Index + pOffset] != -1) {
                hStreamIndices[parIndex + pOffset] = hStreamIndices[child2Index + pOffset];
                waitIndex = hStreamIndices[child1Index + pOffset];
            } else {
                hStreamIndices[parIndex + pOffset] = lastStreamIndex++;
            }
            streamIndex = hStreamIndices[parIndex + pOffset];
        }

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
// printf("op[%d]: c1 %d (%d), c2 %d (%d), c1m %d, c2m %d, par %d, rescale %d, wsi %d, rsi %d, cp %d\n", op, child1Index, tipStates1, child2Index, tipStates2, child1TransMatIndex, child2TransMatIndex, parIndex, rescale, writeScalingIndex, readScalingIndex, currentPartition);
// printf("op[%03d]: c1 %03d (%d), c2 %03d (%d), par %03d, c1m %03d, c2m %03d, rescale %d, streamIndex %03d, waitIndex %03d\n", op, child1Index, (tipStates1?1:0), child2Index, (tipStates2?1:0), parIndex, child1TransMatIndex, child2TransMatIndex, rescale, streamIndex, waitIndex);

// printf("%03d %03d %03d %03d %03d\n", parIndex, child1Index, child2Index, streamIndex, waitIndex);


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

        int startPattern = 0;
        int endPattern = 0;

        if (kUsingMultiGrid && (anyRescale != 1)) {
            int startBlock = 0;
            int endBlock = kNumPatternBlocks;
            if (byPartition) {
                startBlock = hPatternPartitionsStartBlocks[currentPartition];
                endBlock = hPatternPartitionsStartBlocks[currentPartition+1];
            }
            int opBlockCount = endBlock - startBlock;

            int opType = 1;
            if (tipStates1 != 0 && tipStates2 != 0) {
                opType = 3;
            } else if (tipStates1 != 0 || tipStates2 != 0) {
                opType = 2;
            }
            if (rescale == 0) {
                opType *= -1;
            }

            bool newLaunch = false;

            if (op == 0) {
                newLaunch = true;
            } else if (opType != gridOpType[gridLaunches-1]) {
                newLaunch = true;
            } else if (child1Index >= parentMinIndex || child2Index >= parentMinIndex) {
                for (int i=gridStartOp[gridLaunches-1]; i < op; i++) {
                    int previousParentIndex = operations[i * numOps];
                    if (child1Index == previousParentIndex || child2Index == previousParentIndex) {
                        newLaunch = true;
                        break;
                    }
                }
            }

            if (newLaunch) {
                gridStartOp[gridLaunches] = op;
                gridOpBlocks[gridLaunches] = opBlockCount;
                gridOpType[gridLaunches] = opType;
                parentMinIndex = parIndex;

                if (!byPartition) {
                    hGridOpIndices[gridLaunches*6+0] = child1Index;
                    hGridOpIndices[gridLaunches*6+1] = child2Index;
                    hGridOpIndices[gridLaunches*6+2] = parIndex;
                    hGridOpIndices[gridLaunches*6+3] = child1TransMatIndex;
                    hGridOpIndices[gridLaunches*6+4] = child2TransMatIndex;
                    hGridOpIndices[gridLaunches*6+5] = readScalingIndex;
                }

                gridLaunches++;
            } else {
                gridOpBlocks[gridLaunches-1] += opBlockCount;
            }

            if (parIndex < parentMinIndex)
                parentMinIndex = parIndex;

            unsigned int c1Off, c2Off;
            unsigned int c1MOff   = child1TransMatIndex * kIndexOffsetMat;
            unsigned int c2MOff   = child2TransMatIndex * kIndexOffsetMat;
            unsigned int paOff    = hPartialsOffsets[parIndex];
            unsigned int scaleOff = 0;
            if (rescale == 0) {
                scaleOff = readScalingIndex * kScaleBufferSize;
            }


            if (abs(opType) == 1) {
                c1Off  = hPartialsOffsets[child1Index];
                c2Off  = hPartialsOffsets[child2Index];
            } else if (abs(opType) == 2) {
                if (tipStates1 != 0) {
                    c1Off  = hStatesOffsets[child1Index];
                    c2Off  = hPartialsOffsets[child2Index];
                } else {
                    c1Off  = hStatesOffsets[child2Index];
                    c2Off  = hPartialsOffsets[child1Index];
                    unsigned int tmpOff = c1MOff; c1MOff = c2MOff; c2MOff = tmpOff;
                }
            } else {
                c1Off  = hStatesOffsets[child1Index];
                c2Off  = hStatesOffsets[child2Index];
            }


            for (int i=startBlock; i < endBlock; i++) {
                hPartialsPtrs[gridOpIndex++] = hPartitionOffsets[i*2];
                hPartialsPtrs[gridOpIndex++] = hPartitionOffsets[i*2+1];
                hPartialsPtrs[gridOpIndex++] = c1Off;
                hPartialsPtrs[gridOpIndex++] = c2Off;
                hPartialsPtrs[gridOpIndex++] = paOff;
                hPartialsPtrs[gridOpIndex++] = c1MOff;
                hPartialsPtrs[gridOpIndex++] = c2MOff;
                hPartialsPtrs[gridOpIndex++] = scaleOff;

// printf("block %d, hPP = %d %d %d %d %d %d %d %d\n", i,
//        hPartialsPtrs[gridOpIndex-8],
//        hPartialsPtrs[gridOpIndex-7],
//        hPartialsPtrs[gridOpIndex-6],
//        hPartialsPtrs[gridOpIndex-5],
//        hPartialsPtrs[gridOpIndex-4],
//        hPartialsPtrs[gridOpIndex-3],
//        hPartialsPtrs[gridOpIndex-2],
//        hPartialsPtrs[gridOpIndex-1]);

            }
        } else {
            if (byPartition) {
                startPattern = hPatternPartitionsStartPatterns[currentPartition];
                endPattern = hPatternPartitionsStartPatterns[currentPartition+1];
            }


            if (tipStates1 != 0) {
                if (tipStates2 != 0 ) {
                    kernels->StatesStatesPruningDynamicScaling(tipStates1, tipStates2, partials3,
                                                               matrices1, matrices2, scalingFactors,
                                                               cumulativeScalingBuffer,
                                                               startPattern, endPattern,
                                                               kPaddedPatternCount, kCategoryCount,
                                                               rescale,
                                                               streamIndex, waitIndex);
                } else {
                    kernels->StatesPartialsPruningDynamicScaling(tipStates1, partials2, partials3,
                                                                 matrices1, matrices2, scalingFactors,
                                                                 cumulativeScalingBuffer,
                                                                 startPattern, endPattern,
                                                                 kPaddedPatternCount, kCategoryCount,
                                                                 rescale,
                                                                 streamIndex, waitIndex);
                }
            } else {
                if (tipStates2 != 0) {
                    kernels->StatesPartialsPruningDynamicScaling(tipStates2, partials1, partials3,
                                                                 matrices2, matrices1, scalingFactors,
                                                                 cumulativeScalingBuffer,
                                                                 startPattern, endPattern,
                                                                 kPaddedPatternCount, kCategoryCount,
                                                                 rescale,
                                                                 streamIndex, waitIndex);
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
                                                                       startPattern, endPattern,
                                                                       kPaddedPatternCount, kCategoryCount,
                                                                       rescale,
                                                                       streamIndex, waitIndex);
                    }
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
    } //end for loop over operationCount


    if (kUsingMultiGrid && (anyRescale != 1)) {
        size_t transferSize = sizeof(unsigned int) * gridOpIndex;
        #ifdef FW_OPENCL
        gpu->UnmapMemory(dPartialsPtrs, hPartialsPtrs);
        #else
        gpu->MemcpyHostToDevice(dPartialsPtrs, hPartialsPtrs, transferSize);
        #endif
        gridStartOp[gridLaunches] = operationCount;
        int gridStart = 0;
        for (int i=0; i < gridLaunches; i++) {
            int gridSize = gridOpBlocks[i];
            int rescaleMulti = BEAGLE_OP_NONE;
            GPUPtr scalingFactorsMulti = (GPUPtr)NULL;
            if (gridOpType[i] < 0) {
                scalingFactorsMulti = dScalingFactors[0];
                rescaleMulti = 0;
                gridOpType[i] *= -1;
            }

            if (((gridStartOp[i+1] - gridStartOp[i]) == 1) && !byPartition && (kDeviceCode != BEAGLE_OPENCL_DEVICE_AMD_GPU)) {
                int child1Index         = hGridOpIndices[i*6+0];
                int child2Index         = hGridOpIndices[i*6+1];
                int parIndex            = hGridOpIndices[i*6+2];
                int child1TransMatIndex = hGridOpIndices[i*6+3];
                int child2TransMatIndex = hGridOpIndices[i*6+4];
                if (rescaleMulti == 0) {
                    scalingFactorsMulti = dScalingFactors[hGridOpIndices[i*6+5]];
                }
                cumulativeScalingBuffer = 0;

                GPUPtr tipStates1 = dStates[child1Index];
                GPUPtr tipStates2 = dStates[child2Index];
                GPUPtr partials1 = dPartials[child1Index];
                GPUPtr partials2 = dPartials[child2Index];
                GPUPtr partials3 = dPartials[parIndex];
                GPUPtr matrices1 = dMatrices[child1TransMatIndex];
                GPUPtr matrices2 = dMatrices[child2TransMatIndex];

                if (gridOpType[i] == 1) {
                        kernels->PartialsPartialsPruningDynamicScaling(partials1, partials2, partials3,
                                                                       matrices1, matrices2, scalingFactorsMulti,
                                                                       cumulativeScalingBuffer,
                                                                       0, 0,
                                                                       kPaddedPatternCount, kCategoryCount,
                                                                       rescaleMulti,
                                                                       -1, -1);
                } else if (gridOpType[i] == 2) {
                    if (tipStates1 != 0) {
                        kernels->StatesPartialsPruningDynamicScaling(tipStates1, partials2, partials3,
                                                                     matrices1, matrices2, scalingFactorsMulti,
                                                                     cumulativeScalingBuffer,
                                                                     0, 0,
                                                                     kPaddedPatternCount, kCategoryCount,
                                                                     rescaleMulti,
                                                                     -1, -1);
                    } else {
                        kernels->StatesPartialsPruningDynamicScaling(tipStates2, partials1, partials3,
                                                                     matrices2, matrices1, scalingFactorsMulti,
                                                                     cumulativeScalingBuffer,
                                                                     0, 0,
                                                                     kPaddedPatternCount, kCategoryCount,
                                                                     rescaleMulti,
                                                                     -1, -1);
                    }
                } else {
                    kernels->StatesStatesPruningDynamicScaling(tipStates1, tipStates2, partials3,
                                                               matrices1, matrices2, scalingFactorsMulti,
                                                               cumulativeScalingBuffer,
                                                               0, 0,
                                                               kPaddedPatternCount, kCategoryCount,
                                                               rescaleMulti,
                                                               -1, -1);
                }
            } else {
                if (gridOpType[i] == 1) {
                    kernels->PartialsPartialsPruningMulti(dPartialsOrigin, dMatrices[0],
                                                          scalingFactorsMulti,
                                                          dPartialsPtrs,
                                                          kPaddedPatternCount,
                                                          gridStart, gridSize,
                                                          rescaleMulti);
                } else if (gridOpType[i] == 2) {
                    kernels->StatesPartialsPruningMulti(dStatesOrigin, dPartialsOrigin, dMatrices[0],
                                                        scalingFactorsMulti,
                                                        dPartialsPtrs,
                                                        kPaddedPatternCount,
                                                        gridStart, gridSize,
                                                        rescaleMulti);
                } else {
                    kernels->StatesStatesPruningMulti(dStatesOrigin, dPartialsOrigin, dMatrices[0],
                                                      scalingFactorsMulti,
                                                      dPartialsPtrs,
                                                      kPaddedPatternCount,
                                                      gridStart, gridSize,
                                                      rescaleMulti);
                }
            }
            gridStart += gridSize;
        }

        #ifdef FW_OPENCL
        hPartialsPtrs = (unsigned int*)gpu->MapMemory(dPartialsPtrs, kOpOffsetsSize);
        #endif

    }

    if (!kUsingMultiGrid || (anyRescale == 1 && kPartitionsInitialised)) {
        gpu->SynchronizeDevice();
    }

    if (kUsingMultiGrid) {
        free(gridStartOp);
        free(gridOpType);
        free(gridOpBlocks);
    }

#ifdef BEAGLE_DEBUG_SYNCH
    gpu->SynchronizeHost();
#endif

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::upPartials\n");
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
std::vector<int>
BeagleGPUImpl<BEAGLE_GPU_GENERIC>::transposeTransitionMatricesOnTheFly(const int *operations,
                                                                       int operationCount) {

    if (kExtraMatrixCount < operationCount) {

        size_t ptrIncrement = gpu->AlignMemOffset(kMatrixSize * kCategoryCount * sizeof(Real));
        GPUPtr dMatricesOrigin = gpu->AllocateMemory((kMatrixCount + operationCount) * ptrIncrement);

        gpu->MemcpyDeviceToDevice(dMatricesOrigin, dMatrices[0],
                (kMatrixCount + kExtraMatrixCount) * ptrIncrement);

        gpu->FreeMemory(dMatrices[0]);
        free(dMatrices);

        dMatrices = (GPUPtr*) malloc(sizeof(GPUPtr) * (kMatrixCount + operationCount));

        for (int i = 0; i < kMatrixCount + operationCount; ++i) {
            dMatrices[i] = gpu->CreateSubPointer(dMatricesOrigin, ptrIncrement * i, ptrIncrement);

        }

        kExtraMatrixCount = operationCount;
    }

    std::vector<int> newOperation(operations, operations + (7 * operationCount)); // make copy
    std::vector<int> oldMatrices(operationCount);
    std::vector<int> newMatrices(operationCount);

    int currentMatrix = kMatrixCount;
    for (int op = 0; op < operationCount; ++op) {
        oldMatrices[op] = newOperation[op * 7 + 4];
        newMatrices[op] = currentMatrix;
        newOperation[op * 7 + 4] = currentMatrix;
        ++currentMatrix;
    }

    transposeTransitionMatrices(oldMatrices.data(), newMatrices.data(), operationCount);

    return newOperation;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::upPrePartials(bool byPartition,
                                                     const int* inOperations,
                                                     int operationCount,
                                                     int cumulativeScaleIndex) {

    const int* operations = inOperations;
    std::vector<int> newOperations;

    if (kUsingAutoTranspose) {
        newOperations = transposeTransitionMatricesOnTheFly(operations, operationCount);
        operations = newOperations.data();
    }

    // Below is the old serial version of upPartials (as a far starting point)
    for (int op = 0; op < operationCount; op++) {
        const int parIndex = operations[op * 7];                // Self
        const int writeScalingIndex = operations[op * 7 + 1];
        const int readScalingIndex = operations[op * 7 + 2];
        const int child1Index = operations[op * 7 + 3];         // Parent
        const int child1TransMatIndex = operations[op * 7 + 4];
        const int child2Index = operations[op * 7 + 5];         // Sibling
        const int child2TransMatIndex = operations[op * 7 + 6];

        GPUPtr matrices1 = dMatrices[child1TransMatIndex];
        GPUPtr matrices2 = dMatrices[child2TransMatIndex];

        GPUPtr partials1 = dPartials[child1Index];
        GPUPtr partials2 = dPartials[child2Index];
        GPUPtr partials3 = dPartials[parIndex];

        GPUPtr tipStates1 = dStates[child1Index];
        GPUPtr tipStates2 = dStates[child2Index];

        if (tipStates2 != 0) {
            kernels->PartialsStatesGrowing(partials1, tipStates2, partials3,
                                           matrices1, matrices2,
                                           kPaddedPatternCount, kCategoryCount,
                                           sizeof(Real));
        } else {
            kernels->PartialsPartialsGrowing(partials1, partials2, partials3,
                                             matrices1, matrices2,
                                             kPaddedPatternCount, kCategoryCount,
                                             sizeof(Real));
        }
    }

//    GPUPtr cumulativeScalingBuffer = 0;
//    if (cumulativeScalingIndex != BEAGLE_OP_NONE)
//        cumulativeScalingBuffer = dScalingFactors[cumulativeScalingIndex];
//
//    // Serial version
//    for (int op = 0; op < operationCount; op++) {
//        const int parIndex = operations[op * 7];
//        const int writeScalingIndex = operations[op * 7 + 1];
//        const int readScalingIndex = operations[op * 7 + 2];
//        const int child1Index = operations[op * 7 + 3];
//        const int child1TransMatIndex = operations[op * 7 + 4];
//        const int child2Index = operations[op * 7 + 5];
//        const int child2TransMatIndex = operations[op * 7 + 6];
//
//        GPUPtr matrices1 = dMatrices[child1TransMatIndex];
//        GPUPtr matrices2 = dMatrices[child2TransMatIndex];
//
//        GPUPtr partials1 = dPartials[child1Index];
//        GPUPtr partials2 = dPartials[child2Index];
//
//        GPUPtr partials3 = dPartials[parIndex];
//
//        GPUPtr tipStates1 = dStates[child1Index];
//        GPUPtr tipStates2 = dStates[child2Index];
//
//        int rescale = BEAGLE_OP_NONE;
//        GPUPtr scalingFactors = (GPUPtr)NULL;
//
//        if (kFlags & BEAGLE_FLAG_SCALING_AUTO) {
//            int sIndex = parIndex - kTipCount;
//
//            if (tipStates1 == 0 && tipStates2 == 0) {
//                rescale = 2;
//                scalingFactors = dScalingFactors[sIndex];
//            }
//        } else if (kFlags & BEAGLE_FLAG_SCALING_ALWAYS) {
//            rescale = 1;
//            scalingFactors = dScalingFactors[parIndex - kTipCount];
//        } else if ((kFlags & BEAGLE_FLAG_SCALING_MANUAL) && writeScalingIndex >= 0) {
//            rescale = 1;
//            scalingFactors = dScalingFactors[writeScalingIndex];
//        } else if ((kFlags & BEAGLE_FLAG_SCALING_MANUAL) && readScalingIndex >= 0) {
//            rescale = 0;
//            scalingFactors = dScalingFactors[readScalingIndex];
//        }
//
//#ifdef BEAGLE_DEBUG_VALUES
//        fprintf(stderr, "kPaddedPatternCount = %d\n", kPaddedPatternCount);
//        fprintf(stderr, "kPatternCount = %d\n", kPatternCount);
//        fprintf(stderr, "categoryCount  = %d\n", kCategoryCount);
//        fprintf(stderr, "partialSize = %d\n", kPartialsSize);
//        fprintf(stderr, "writeIndex = %d,  readIndex = %d, rescale = %d\n",writeScalingIndex,readScalingIndex,rescale);
//        fprintf(stderr, "child1 = \n");
//        Real r = 0;
//        if (tipStates1)
//            gpu->PrintfDeviceInt(tipStates1, kPaddedPatternCount);
//        else
//            gpu->PrintfDeviceVector(partials1, kPartialsSize, r);
//        fprintf(stderr, "child2 = \n");
//        if (tipStates2)
//            gpu->PrintfDeviceInt(tipStates2, kPaddedPatternCount);
//        else
//            gpu->PrintfDeviceVector(partials2, kPartialsSize, r);
//        fprintf(stderr, "node index = %d\n", parIndex);
//#endif
//
//        if (tipStates1 != 0) {
//            if (tipStates2 != 0 ) {
//                kernels->StatesStatesPruningDynamicScaling(tipStates1, tipStates2, partials3,
//                                                           matrices1, matrices2, scalingFactors,
//                                                           cumulativeScalingBuffer,
//                                                           kPaddedPatternCount, kCategoryCount,
//                                                           rescale);
//            } else {
//                kernels->StatesPartialsPruningDynamicScaling(tipStates1, partials2, partials3,
//                                                             matrices1, matrices2, scalingFactors,
//                                                             cumulativeScalingBuffer,
//                                                             kPaddedPatternCount, kCategoryCount,
//                                                             rescale);
//            }
//        } else {
//            if (tipStates2 != 0) {
//                kernels->StatesPartialsPruningDynamicScaling(tipStates2, partials1, partials3,
//                                                             matrices2, matrices1, scalingFactors,
//                                                             cumulativeScalingBuffer,
//                                                             kPaddedPatternCount, kCategoryCount,
//                                                             rescale);
//            } else {
//                if (kFlags & BEAGLE_FLAG_SCALING_DYNAMIC) {
//                    kernels->PartialsPartialsPruningDynamicCheckScaling(partials1, partials2, partials3,
//                                                                        matrices1, matrices2, writeScalingIndex, readScalingIndex,
//                                                                        cumulativeScalingIndex, dScalingFactors, dScalingFactorsMaster,
//                                                                        kPaddedPatternCount, kCategoryCount,
//                                                                        rescale, hRescalingTrigger, dRescalingTrigger, sizeof(Real));
//                } else {
//                    kernels->PartialsPartialsPruningDynamicScaling(partials1, partials2, partials3,
//                                                                   matrices1, matrices2, scalingFactors,
//                                                                   cumulativeScalingBuffer,
//                                                                   kPaddedPatternCount, kCategoryCount,
//                                                                   rescale);
//                }
//            }
//        }
//
//        if (kFlags & BEAGLE_FLAG_SCALING_ALWAYS) {
//            int parScalingIndex = parIndex - kTipCount;
//            int child1ScalingIndex = child1Index - kTipCount;
//            int child2ScalingIndex = child2Index - kTipCount;
//            if (child1ScalingIndex >= 0 && child2ScalingIndex >= 0) {
//                int scalingIndices[2] = {child1ScalingIndex, child2ScalingIndex};
//                accumulateScaleFactors(scalingIndices, 2, parScalingIndex);
//            } else if (child1ScalingIndex >= 0) {
//                int scalingIndices[1] = {child1ScalingIndex};
//                accumulateScaleFactors(scalingIndices, 1, parScalingIndex);
//            } else if (child2ScalingIndex >= 0) {
//                int scalingIndices[1] = {child2ScalingIndex};
//                accumulateScaleFactors(scalingIndices, 1, parScalingIndex);
//            }
//        }
//    }
//

#ifdef BEAGLE_DEBUG_SYNCH
    gpu->Synchronize();
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
            gpu->SynchronizeDevice();
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
        for(int n = 0; n < count; n++) {
            hPtrQueue[n] = scalingIndices[n] * kScaleBufferSize;
        }

        gpu->MemcpyHostToDevice(dPtrQueue, hPtrQueue, sizeof(unsigned int) * count);

        // Compute scaling factors at the root
        kernels->AccumulateFactorsDynamicScaling(dScalingFactors[0], dPtrQueue, dScalingFactors[cumulativeScalingIndex], count, kPaddedPatternCount);
    }

#ifdef BEAGLE_DEBUG_SYNCH
    gpu->SynchronizeHost();
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
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::accumulateScaleFactorsByPartition(const int* scalingIndices,
                                                                         int count,
                                                                         int cumulativeScalingIndex,
                                                                         int partitionIndex) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::accumulateScaleFactorsByPartition\n");
#endif

    if (kFlags & BEAGLE_FLAG_SCALING_DYNAMIC) {
        return BEAGLE_ERROR_NO_IMPLEMENTATION;
    } else if (kFlags & BEAGLE_FLAG_SCALING_AUTO) {
        return BEAGLE_ERROR_NO_IMPLEMENTATION;
    }

    int startPattern = hPatternPartitionsStartPatterns[partitionIndex];
    int endPattern = hPatternPartitionsStartPatterns[partitionIndex + 1];

    for(int n = 0; n < count; n++)
        hPtrQueue[n] = scalingIndices[n] * kScaleBufferSize;
    gpu->MemcpyHostToDevice(dPtrQueue, hPtrQueue, sizeof(unsigned int) * count);


    // Compute scaling factors at the root
    kernels->AccumulateFactorsDynamicScalingByPartition(dScalingFactors[0],
                                                        dPtrQueue,
                                                        dScalingFactors[cumulativeScalingIndex],
                                                        count,
                                                        startPattern,
                                                        endPattern);


#ifdef BEAGLE_DEBUG_SYNCH
    gpu->SynchronizeHost();
#endif

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::accumulateScaleFactorsByPartition\n");
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
            gpu->SynchronizeDevice();
            dScalingFactors[cumulativeScalingIndex] = dScalingFactorsMaster[cumulativeScalingIndex];
        }
    }

    for(int n = 0; n < count; n++)
        hPtrQueue[n] = scalingIndices[n] * kScaleBufferSize;
    gpu->MemcpyHostToDevice(dPtrQueue, hPtrQueue, sizeof(unsigned int) * count);

    // Compute scaling factors at the root
    kernels->RemoveFactorsDynamicScaling(dScalingFactors[0], dPtrQueue, dScalingFactors[cumulativeScalingIndex],
                                         count, kPaddedPatternCount);

#ifdef BEAGLE_DEBUG_SYNCH
    gpu->SynchronizeHost();
#endif

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::removeScaleFactors\n");
#endif

    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::removeScaleFactorsByPartition(const int* scalingIndices,
                                                                     int count,
                                                                     int cumulativeScalingIndex,
                                                                     int partitionIndex) {

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::removeScaleFactorsByPartition\n");
#endif

    if (kFlags & BEAGLE_FLAG_SCALING_DYNAMIC) {
        return BEAGLE_ERROR_NO_IMPLEMENTATION;
    }

    int startPattern = hPatternPartitionsStartPatterns[partitionIndex];
    int endPattern = hPatternPartitionsStartPatterns[partitionIndex + 1];

    for(int n = 0; n < count; n++)
        hPtrQueue[n] = scalingIndices[n] * kScaleBufferSize;
    gpu->MemcpyHostToDevice(dPtrQueue, hPtrQueue, sizeof(unsigned int) * count);

    // Compute scaling factors at the root
    kernels->RemoveFactorsDynamicScalingByPartition(dScalingFactors[0],
                                                    dPtrQueue,
                                                    dScalingFactors[cumulativeScalingIndex],
                                                    count,
                                                    startPattern,
                                                    endPattern);

#ifdef BEAGLE_DEBUG_SYNCH
    gpu->SynchronizeHost();
#endif

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::removeScaleFactorsByPartition\n");
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
    gpu->SynchronizeHost();
#endif

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::resetScaleFactors\n");
#endif

    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::resetScaleFactorsByPartition(int cumulativeScalingIndex,
                                                                    int partitionIndex) {

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::resetScaleFactorsByPartition\n");
#endif

    if (kFlags & BEAGLE_FLAG_SCALING_DYNAMIC) {
        return BEAGLE_ERROR_NO_IMPLEMENTATION;
    }

// printf("TEST THIS FUNCTION ON CUDA & OPENCL\n");
// exit(-1);

if (0) {
    int startPattern = hPatternPartitionsStartPatterns[partitionIndex];
    int endPattern = hPatternPartitionsStartPatterns[partitionIndex + 1];
    int partitionPatternCount = endPattern - startPattern;
    printf("partitionPatternCount %d\n", partitionPatternCount );

    size_t ptrIncrement = sizeof(Real);

    Real* zeroes = (Real*) gpu->CallocHost(ptrIncrement, partitionPatternCount);

    GPUPtr dScalingFactorPartition = gpu->CreateSubPointer(dScalingFactors[cumulativeScalingIndex],
                                                           gpu->AlignMemOffset(ptrIncrement * startPattern),
                                                           ptrIncrement * partitionPatternCount);
    printf("created subpointer for %d\n", partitionPatternCount);
    // Fill with zeroes
    gpu->MemcpyHostToDevice(dScalingFactorPartition,
                            zeroes,
                            ptrIncrement * partitionPatternCount);

    gpu->FreeHostMemory(zeroes);
} else if (0) {
    Real* zeroes = (Real*) gpu->CallocHost(sizeof(Real), kPaddedPatternCount);

    GPUPtr dScalingFactorPartition = dScalingFactors[cumulativeScalingIndex];

    // Fill with zeroes
    gpu->MemcpyHostToDevice(dScalingFactorPartition,
                            zeroes,
                            sizeof(Real) * kPaddedPatternCount);

    gpu->FreeHostMemory(zeroes);


} else {
    int startPattern = hPatternPartitionsStartPatterns[partitionIndex];
    int endPattern = hPatternPartitionsStartPatterns[partitionIndex + 1];

    kernels->ResetFactorsDynamicScalingByPartition(dScalingFactors[cumulativeScalingIndex],
                                                   startPattern,
                                                   endPattern);
}
#ifdef BEAGLE_DEBUG_SYNCH
    gpu->SynchronizeHost();
#endif

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::resetScaleFactorsByPartition\n");
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
    gpu->SynchronizeHost();
#endif

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::copyScaleFactors\n");
#endif

    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::getScaleFactors(int srcScalingIndex,
                                                       double* scaleFactors) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::getScaleFactors\n");
#endif

    // TBD
    return BEAGLE_ERROR_NO_IMPLEMENTATION;

#ifdef BEAGLE_DEBUG_SYNCH
    gpu->SynchronizeHost();
#endif

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::getScaleFactors\n");
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

        if (kFlags & BEAGLE_FLAG_COMPUTATION_SYNCH) {
            gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dSumLogLikelihood, sizeof(Real) * kSumSitesBlockCount);

            *outSumLogLikelihood = 0.0;
            for (int i = 0; i < kSumSitesBlockCount; i++) {
                if (hLogLikelihoodsCache[i] != hLogLikelihoodsCache[i])
                    returnCode = BEAGLE_ERROR_FLOATING_POINT;

                *outSumLogLikelihood += hLogLikelihoodsCache[i];
            }
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

            if (kFlags & BEAGLE_FLAG_COMPUTATION_SYNCH) {
                gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dSumLogLikelihood, sizeof(Real) * kSumSitesBlockCount);

                *outSumLogLikelihood = 0.0;
                for (int i = 0; i < kSumSitesBlockCount; i++) {
                    if (hLogLikelihoodsCache[i] != hLogLikelihoodsCache[i])
                        returnCode = BEAGLE_ERROR_FLOATING_POINT;

                    *outSumLogLikelihood += hLogLikelihoodsCache[i];
                }
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
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::calculateRootLogLikelihoodsByPartition(
                                                                const int* bufferIndices,
                                                                const int* categoryWeightsIndices,
                                                                const int* stateFrequenciesIndices,
                                                                const int* cumulativeScaleIndices,
                                                                const int* partitionIndices,
                                                                int partitionCount,
                                                                int count,
                                                                double* outSumLogLikelihoodByPartition,
                                                                double* outSumLogLikelihood) {

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::calculateRootLogLikelihoodsByPartition\n");
#endif

    if (count != 1 || kFlags & BEAGLE_FLAG_SCALING_AUTO || kFlags & BEAGLE_FLAG_SCALING_ALWAYS) {
        return BEAGLE_ERROR_NO_IMPLEMENTATION;
    }

    int returnCode = BEAGLE_SUCCESS;

    int gridOpIndex = 0;
    int gridSize = 0;

    bool scale = false;

    for (int p = 0; p < partitionCount; p++) {
        if (cumulativeScaleIndices[p] != BEAGLE_OP_NONE) {
            scale = true;
        }
    }

    for (int p = 0; p < partitionCount; p++) {
        int pIndex = partitionIndices[p];

        int startBlock = hIntegratePartitionsStartBlocks[pIndex];
        int endBlock = hIntegratePartitionsStartBlocks[pIndex+1];

        gridSize += endBlock - startBlock;

        const int rootNodeIndex = bufferIndices[p];
        const int categoryWeightsIndex = categoryWeightsIndices[p];
        const int stateFrequenciesIndex = stateFrequenciesIndices[p];
              int cumulativeScalingIndex = kScaleBufferCount; // default to buffer with zeroes

        if ((scale == true) && (cumulativeScaleIndices[p] != BEAGLE_OP_NONE)) {
            cumulativeScalingIndex = cumulativeScaleIndices[p];
        }

        unsigned int rNOff      = hPartialsOffsets[rootNodeIndex];
        unsigned int cWOff      = categoryWeightsIndex  * kWeightsOffset;
        unsigned int sFOff      = stateFrequenciesIndex * kFrequenciesOffset;
        unsigned int scaleOff   = cumulativeScalingIndex * kScaleBufferSize;

        for (int i=startBlock; i < endBlock; i++) {
            hPartialsPtrs[gridOpIndex++] = hIntegratePartitionOffsets[i*2];
            hPartialsPtrs[gridOpIndex++] = hIntegratePartitionOffsets[i*2+1];
            hPartialsPtrs[gridOpIndex++] = rNOff;
            hPartialsPtrs[gridOpIndex++] = cWOff;
            hPartialsPtrs[gridOpIndex++] = sFOff;
            hPartialsPtrs[gridOpIndex++] = scaleOff;

// printf("block %d, hPP = %d %d %d %d %d %d\n", i,
//        hPartialsPtrs[gridOpIndex-6],
//        hPartialsPtrs[gridOpIndex-5],
//        hPartialsPtrs[gridOpIndex-4],
//        hPartialsPtrs[gridOpIndex-3],
//        hPartialsPtrs[gridOpIndex-2],
//        hPartialsPtrs[gridOpIndex-1]);
        }

        // Real r = 0;
        // fprintf(stderr,"root partials = \n");
        // gpu->PrintfDeviceVector(dPartials[rootNodeIndex], kPaddedPatternCount, r);

    }

    size_t transferSize = sizeof(unsigned int) * gridOpIndex;
    #ifdef FW_OPENCL
    gpu->UnmapMemory(dPartialsPtrs, hPartialsPtrs);
    #else
    gpu->MemcpyHostToDevice(dPartialsPtrs, hPartialsPtrs, transferSize);
    #endif

    if (scale == true) {
        kernels->IntegrateLikelihoodsDynamicScalingPartition(dIntegrationTmp,
                                                             dPartialsOrigin,
                                                             dWeights[0],
                                                             dFrequencies[0],
                                                             dScalingFactors[0],
                                                             dPartialsPtrs,
                                                             kPaddedPatternCount,
                                                             kCategoryCount,
                                                             gridSize);
    } else {
        kernels->IntegrateLikelihoodsPartition(dIntegrationTmp,
                                               dPartialsOrigin,
                                               dWeights[0],
                                               dFrequencies[0],
                                               dPartialsPtrs,
                                               kPaddedPatternCount,
                                               kCategoryCount,
                                               gridSize);

            // kernels->IntegrateLikelihoods(dIntegrationTmp, dPartials[bufferIndices[0]],
            //                               dWeights[0],
            //                               dFrequencies[0],
            //                               kPaddedPatternCount, kCategoryCount);
    }

// Real r = 0;
// fprintf(stderr,"before SumSites1 = \n");
// gpu->PrintfDeviceVector(dIntegrationTmp, kPaddedPatternCount, r);


    #ifdef FW_OPENCL
    hPartialsPtrs = (unsigned int*)gpu->MapMemory(dPartialsPtrs, kOpOffsetsSize);
    #endif

    *outSumLogLikelihood = 0.0;

    for (int p = 0; p < partitionCount; p++) {
        int pIndex = partitionIndices[p];
        int startPattern = hPatternPartitionsStartPatterns[pIndex];
        int endPattern = hPatternPartitionsStartPatterns[pIndex + 1];
        int partitionPatternCount = endPattern - startPattern;
        int partitionSumSitesBlockCount = partitionPatternCount / kSumSitesBlockSize;
        if (partitionPatternCount % kSumSitesBlockSize != 0)
            partitionSumSitesBlockCount += 1;

        kernels->SumSites1Partition(dIntegrationTmp,
                                    dSumLogLikelihood,
                                    dPatternWeights,
                                    startPattern,
                                    endPattern,
                                    partitionSumSitesBlockCount);

        gpu->MemcpyDeviceToHost(hLogLikelihoodsCache,
                                dSumLogLikelihood,
                                sizeof(Real) * partitionSumSitesBlockCount);

        outSumLogLikelihoodByPartition[p] = 0.0;
        for (int i = 0; i < partitionSumSitesBlockCount; i++) {
            if (hLogLikelihoodsCache[i] != hLogLikelihoodsCache[i])
                returnCode = BEAGLE_ERROR_FLOATING_POINT;
            outSumLogLikelihoodByPartition[p] += hLogLikelihoodsCache[i];
        }
        *outSumLogLikelihood += outSumLogLikelihoodByPartition[p];
    }

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::calculateRootLogLikelihoodsByPartition\n");
#endif

    return returnCode;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::calculateCrossProducts(const int *postBufferIndices,
                                                              const int *preBufferIndices,
                                                              const int *categoryRatesIndices,
                                                              const int *categoryWeightsIndices,
                                                              const double *edgeLengths,
                                                              int count,
                                                              double *outSumDerivatives,
                                                              double *outSumSquaredDerivatives) {

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::calculateCrossProducts\n");
#endif

    int returnCode = calcCrossProducts(
            postBufferIndices, preBufferIndices,
            categoryRatesIndices, categoryWeightsIndices, edgeLengths, count, outSumDerivatives);

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::calculateCrossProducts\n");
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

    if (firstDerivativeIndices != NULL && !kDerivBuffersInitialised) {
        dSumFirstDeriv = gpu->AllocateMemory(kSumSitesBlockCount * sizeof(Real));
        dSumSecondDeriv = gpu->AllocateMemory(kSumSitesBlockCount * sizeof(Real));

        dFirstDerivTmp = gpu->AllocateMemory(kPartialsSize * sizeof(Real));
        dSecondDerivTmp = gpu->AllocateMemory(kPartialsSize * sizeof(Real));

        dOutFirstDeriv = gpu->AllocateMemory((kPaddedPatternCount + kResultPaddedPatterns) * sizeof(Real));
        dOutSecondDeriv = gpu->AllocateMemory((kPaddedPatternCount + kResultPaddedPatterns) * sizeof(Real));

        kDerivBuffersInitialised = true;
    }

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

            if (kFlags & BEAGLE_FLAG_COMPUTATION_SYNCH) {
                kernels->SumSites1(dIntegrationTmp, dSumLogLikelihood, dPatternWeights,
                                            kPatternCount);

                gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dSumLogLikelihood, sizeof(Real) * kSumSitesBlockCount);

                *outSumLogLikelihood = 0.0;
                for (int i = 0; i < kSumSitesBlockCount; i++) {
                    if (hLogLikelihoodsCache[i] != hLogLikelihoodsCache[i])
                        returnCode = BEAGLE_ERROR_FLOATING_POINT;

                    *outSumLogLikelihood += hLogLikelihoodsCache[i];
                }
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
            if (kFlags & BEAGLE_FLAG_COMPUTATION_SYNCH) {
                gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dSumLogLikelihood, sizeof(Real) * kSumSitesBlockCount);

                *outSumLogLikelihood = 0.0;
                for (int i = 0; i < kSumSitesBlockCount; i++) {
                    if (hLogLikelihoodsCache[i] != hLogLikelihoodsCache[i])
                        returnCode = BEAGLE_ERROR_FLOATING_POINT;

                    *outSumLogLikelihood += hLogLikelihoodsCache[i];
                }

                gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dSumFirstDeriv, sizeof(Real) * kSumSitesBlockCount);

                *outSumFirstDerivative = 0.0;
                for (int i = 0; i < kSumSitesBlockCount; i++) {
                    *outSumFirstDerivative += hLogLikelihoodsCache[i];
                }
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

            if (kFlags & BEAGLE_FLAG_COMPUTATION_SYNCH) {
                gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dSumLogLikelihood, sizeof(Real) * kSumSitesBlockCount);

                *outSumLogLikelihood = 0.0;
                for (int i = 0; i < kSumSitesBlockCount; i++) {
                    if (hLogLikelihoodsCache[i] != hLogLikelihoodsCache[i])
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

                if (kFlags & BEAGLE_FLAG_COMPUTATION_SYNCH) {
                    gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dSumLogLikelihood, sizeof(Real) * kSumSitesBlockCount);

                    *outSumLogLikelihood = 0.0;
                    for (int i = 0; i < kSumSitesBlockCount; i++) {
                        if (hLogLikelihoodsCache[i] != hLogLikelihoodsCache[i])
                            returnCode = BEAGLE_ERROR_FLOATING_POINT;

                        *outSumLogLikelihood += hLogLikelihoodsCache[i];
                    }
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

BEAGLE_GPU_TEMPLATE
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::calculateEdgeLogLikelihoodsByPartition(
                                                    const int* parentBufferIndices,
                                                    const int* childBufferIndices,
                                                    const int* probabilityIndices,
                                                    const int* firstDerivativeIndices,
                                                    const int* secondDerivativeIndices,
                                                    const int* categoryWeightsIndices,
                                                    const int* stateFrequenciesIndices,
                                                    const int* cumulativeScaleIndices,
                                                    const int* partitionIndices,
                                                    int partitionCount,
                                                    int count,
                                                    double* outSumLogLikelihoodByPartition,
                                                    double* outSumLogLikelihood,
                                                    double* outSumFirstDerivativeByPartition,
                                                    double* outSumFirstDerivative,
                                                    double* outSumSecondDerivativeByPartition,
                                                    double* outSumSecondDerivative) {

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::calculateEdgeLogLikelihoodsByPartition\n");
#endif

    if (firstDerivativeIndices != NULL && !kDerivBuffersInitialised) {
        dSumFirstDeriv = gpu->AllocateMemory(kSumSitesBlockCount * sizeof(Real));
        dSumSecondDeriv = gpu->AllocateMemory(kSumSitesBlockCount * sizeof(Real));

        dFirstDerivTmp = gpu->AllocateMemory(kPartialsSize * sizeof(Real));
        dSecondDerivTmp = gpu->AllocateMemory(kPartialsSize * sizeof(Real));

        dOutFirstDeriv = gpu->AllocateMemory((kPaddedPatternCount + kResultPaddedPatterns) * sizeof(Real));
        dOutSecondDeriv = gpu->AllocateMemory((kPaddedPatternCount + kResultPaddedPatterns) * sizeof(Real));

        kDerivBuffersInitialised = true;
    }

    if (count != 1 ||  firstDerivativeIndices != NULL ||  secondDerivativeIndices != NULL ||
        kFlags & BEAGLE_FLAG_SCALING_AUTO || kFlags & BEAGLE_FLAG_SCALING_ALWAYS) {
        return BEAGLE_ERROR_NO_IMPLEMENTATION;
    }

    int returnCode = BEAGLE_SUCCESS;

    int gridOpIndex = 0;
    int gridSize = 0;
    int statesChild = -1;

    for (int p = 0; p < partitionCount; p++) {
        int pIndex = partitionIndices[p];

        int startBlock = hPatternPartitionsStartBlocks[pIndex];
        int endBlock = hPatternPartitionsStartBlocks[pIndex+1];

        gridSize += endBlock - startBlock;

        const int parIndex              = parentBufferIndices[p];
        const int childIndex            = childBufferIndices[p];
        const int probIndex             = probabilityIndices[p];

        if (dStates[childIndex] != 0){
            if (statesChild == 0) {
                return BEAGLE_ERROR_NO_IMPLEMENTATION;
            }
            statesChild = 1;
        } else {
            if (statesChild == 1) {
                return BEAGLE_ERROR_NO_IMPLEMENTATION;
            }
            statesChild = 0;
        }

        unsigned int pBOff      = hPartialsOffsets[parIndex];
        unsigned int childOff;
        if (statesChild != 0)
            childOff            = hStatesOffsets  [childIndex];
        else
            childOff            = hPartialsOffsets[childIndex];
        unsigned int probOff    = probIndex              * kIndexOffsetMat;

        for (int i=startBlock; i < endBlock; i++) {
            hPartialsPtrs[gridOpIndex++] = hPartitionOffsets[i*2];
            hPartialsPtrs[gridOpIndex++] = hPartitionOffsets[i*2+1];
            hPartialsPtrs[gridOpIndex++] = pBOff;
            hPartialsPtrs[gridOpIndex++] = childOff;
            hPartialsPtrs[gridOpIndex++] = probOff;
        }
    }

    size_t transferSize = sizeof(unsigned int) * gridOpIndex;
    #ifdef FW_OPENCL
    gpu->UnmapMemory(dPartialsPtrs, hPartialsPtrs);
    #else
    gpu->MemcpyHostToDevice(dPartialsPtrs, hPartialsPtrs, transferSize);
    #endif

    if (statesChild != 0) {
        kernels->StatesPartialsEdgeLikelihoodsByPartition(dPartialsTmp,
                                                          dPartialsOrigin,
                                                          dStatesOrigin,
                                                          dMatrices[0],
                                                          dPartialsPtrs,
                                                          kPaddedPatternCount,
                                                          gridSize);
    } else {
        kernels->PartialsPartialsEdgeLikelihoodsByPartition(dPartialsTmp,
                                                            dPartialsOrigin,
                                                            dMatrices[0],
                                                            dPartialsPtrs,
                                                            kPaddedPatternCount,
                                                            gridSize);
    }

    #ifdef FW_OPENCL
    hPartialsPtrs = (unsigned int*)gpu->MapMemory(dPartialsPtrs, kOpOffsetsSize);
    #endif

    gridOpIndex = 0;
    gridSize = 0;

    bool scale = false;

    for (int p = 0; p < partitionCount; p++) {
        if (cumulativeScaleIndices[p] != BEAGLE_OP_NONE) {
            scale = true;
        }
    }

    for (int p = 0; p < partitionCount; p++) {
        int pIndex = partitionIndices[p];

        int startBlock = hIntegratePartitionsStartBlocks[pIndex];
        int endBlock = hIntegratePartitionsStartBlocks[pIndex+1];

        gridSize += endBlock - startBlock;

        const int categoryWeightsIndex  = categoryWeightsIndices[p];
        const int stateFrequenciesIndex = stateFrequenciesIndices[p];

        int cumulativeScalingIndex = kScaleBufferCount; // default to buffer with zeroes

        if ((scale == true) && (cumulativeScaleIndices[p] != BEAGLE_OP_NONE)) {
            cumulativeScalingIndex = cumulativeScaleIndices[p];
        }

        unsigned int cWOff      = categoryWeightsIndex   * kWeightsOffset;
        unsigned int sFOff      = stateFrequenciesIndex  * kFrequenciesOffset;
        unsigned int scaleOff   = cumulativeScalingIndex * kScaleBufferSize;

        for (int i=startBlock; i < endBlock; i++) {
            hPartialsPtrs[gridOpIndex++] = hIntegratePartitionOffsets[i*2];
            hPartialsPtrs[gridOpIndex++] = hIntegratePartitionOffsets[i*2+1];
            hPartialsPtrs[gridOpIndex++] = 0;
            hPartialsPtrs[gridOpIndex++] = cWOff;
            hPartialsPtrs[gridOpIndex++] = sFOff;
            hPartialsPtrs[gridOpIndex++] = scaleOff;
        }
    }

    transferSize = sizeof(unsigned int) * gridOpIndex;
    #ifdef FW_OPENCL
    gpu->UnmapMemory(dPartialsPtrs, hPartialsPtrs);
    #else
    gpu->MemcpyHostToDevice(dPartialsPtrs, hPartialsPtrs, transferSize);
    #endif

    if (scale == true) {
        kernels->IntegrateLikelihoodsDynamicScalingPartition(dIntegrationTmp,
                                                             dPartialsTmp,
                                                             dWeights[0],
                                                             dFrequencies[0],
                                                             dScalingFactors[0],
                                                             dPartialsPtrs,
                                                             kPaddedPatternCount,
                                                             kCategoryCount,
                                                             gridSize);
    } else {
        kernels->IntegrateLikelihoodsPartition(dIntegrationTmp,
                                               dPartialsTmp,
                                               dWeights[0],
                                               dFrequencies[0],
                                               dPartialsPtrs,
                                               kPaddedPatternCount,
                                               kCategoryCount,
                                               gridSize);
    }

    #ifdef FW_OPENCL
    hPartialsPtrs = (unsigned int*)gpu->MapMemory(dPartialsPtrs, kOpOffsetsSize);
    #endif

    *outSumLogLikelihood = 0.0;

    for (int p = 0; p < partitionCount; p++) {
        int pIndex = partitionIndices[p];
        int startPattern = hPatternPartitionsStartPatterns[pIndex];
        int endPattern = hPatternPartitionsStartPatterns[pIndex + 1];
        int partitionPatternCount = endPattern - startPattern;
        int partitionSumSitesBlockCount = partitionPatternCount / kSumSitesBlockSize;
        if (partitionPatternCount % kSumSitesBlockSize != 0)
            partitionSumSitesBlockCount += 1;

        kernels->SumSites1Partition(dIntegrationTmp,
                                    dSumLogLikelihood,
                                    dPatternWeights,
                                    startPattern,
                                    endPattern,
                                    partitionSumSitesBlockCount);

        gpu->MemcpyDeviceToHost(hLogLikelihoodsCache,
                                dSumLogLikelihood,
                                sizeof(Real) * partitionSumSitesBlockCount);

        outSumLogLikelihoodByPartition[p] = 0.0;
        for (int i = 0; i < partitionSumSitesBlockCount; i++) {
            if (hLogLikelihoodsCache[i] != hLogLikelihoodsCache[i])
                returnCode = BEAGLE_ERROR_FLOATING_POINT;
            outSumLogLikelihoodByPartition[p] += hLogLikelihoodsCache[i];
        }
        *outSumLogLikelihood += outSumLogLikelihoodByPartition[p];
    }

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::calculateEdgeLogLikelihoodsByPartition\n");
#endif

    return returnCode;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::getLogLikelihood(double* outSumLogLikelihood) {

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::getLogLikelihood\n");
#endif

    int returnCode = BEAGLE_SUCCESS;

    gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dSumLogLikelihood, sizeof(Real) * kSumSitesBlockCount);

    *outSumLogLikelihood = 0.0;
    for (int i = 0; i < kSumSitesBlockCount; i++) {
        if (hLogLikelihoodsCache[i] != hLogLikelihoodsCache[i])
            returnCode = BEAGLE_ERROR_FLOATING_POINT;

        *outSumLogLikelihood += hLogLikelihoodsCache[i];
    }

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::getLogLikelihood\n");
#endif

    return returnCode;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::getDerivatives(double* outSumFirstDerivative,
                                                      double* outSumSecondDerivative) {

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::getDerivatives\n");
#endif

    gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dSumFirstDeriv, sizeof(Real) * kSumSitesBlockCount);

    *outSumFirstDerivative = 0.0;
    for (int i = 0; i < kSumSitesBlockCount; i++) {
        *outSumFirstDerivative += hLogLikelihoodsCache[i];
    }

    if (outSumSecondDerivative != NULL) {
        gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dSumSecondDeriv, sizeof(Real) * kSumSitesBlockCount);

        *outSumSecondDerivative = 0.0;
        for (int i = 0; i < kSumSitesBlockCount; i++) {
            *outSumSecondDerivative += hLogLikelihoodsCache[i];
        }
    }

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::getDerivatives\n");
#endif

    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::getSiteLogLikelihoods(double* outLogLikelihoods) {

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::getSiteLogLikelihoods\n");
#endif

// TODO: copy directly to outLogLikelihoods when GPU is running in double precision
    gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dIntegrationTmp, sizeof(Real) * kPatternCount);

    if (kPatternsReordered) {
        Real* outLogLikelihoodsOriginalOrder = (Real*) malloc(sizeof(Real) * kPatternCount);

        for (int i=0; i < kPatternCount; i++) {
            outLogLikelihoodsOriginalOrder[i] = hLogLikelihoodsCache[hPatternsNewOrder[i]];
        }
        beagleMemCpy(outLogLikelihoods, outLogLikelihoodsOriginalOrder, kPatternCount);
        free(outLogLikelihoodsOriginalOrder);
    } else {
        beagleMemCpy(outLogLikelihoods, hLogLikelihoodsCache, kPatternCount);
    }

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::getSiteLogLikelihoods\n");
#endif

    return BEAGLE_SUCCESS;
}


BEAGLE_GPU_TEMPLATE
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::getSiteDerivatives(double* outFirstDerivatives,
                                      double* outSecondDerivatives) {

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::getSiteDerivatives\n");
#endif

    gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dOutFirstDeriv, sizeof(Real) * kPatternCount);
    beagleMemCpy(outFirstDerivatives, hLogLikelihoodsCache, kPatternCount);

    if (outSecondDerivatives != NULL) {
        gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dOutSecondDeriv, sizeof(Real) * kPatternCount);
        beagleMemCpy(outSecondDerivatives, hLogLikelihoodsCache, kPatternCount);
    }

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::getSiteDerivatives\n");
#endif

    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::calcEdgeFirstDerivatives(const int *postBufferIndices,
                                                                const int *preBufferIndices,
                                                                const int *firstDerivativeIndices,
                                                                const int *categoryWeightsIndices,
                                                                const int *scaleIndices,
                                                                int totalCount,
                                                                double *outFirstDerivatives,
                                                                double *outSumFirstDerivatives,
                                                                double *outSumSquaredFirstDerivatives) {

    int instructionOffset = 0;
    int statesTipsCount = 0;

    if (kCompactBufferCount > 0) {
        for (int i = 0; i < totalCount; ++i) {

            if (postBufferIndices[i] < kCompactBufferCount) {
                hDerivativeQueue[instructionOffset++] = hStatesOffsets[postBufferIndices[i]];
                hDerivativeQueue[instructionOffset++] = hPartialsOffsets[preBufferIndices[i]];
                hDerivativeQueue[instructionOffset++] = firstDerivativeIndices[i] * kIndexOffsetMat;
                ++statesTipsCount;
            }
        }
    }

    for (int i = 0; i < totalCount; i++) {
        if (postBufferIndices[i] >= kCompactBufferCount) {
            hDerivativeQueue[instructionOffset++] = hPartialsOffsets[postBufferIndices[i]];
            hDerivativeQueue[instructionOffset++] = hPartialsOffsets[preBufferIndices[i]];
            hDerivativeQueue[instructionOffset++] = firstDerivativeIndices[i] * kIndexOffsetMat;
        }
    }

    gpu->MemcpyHostToDevice(dDerivativeQueue, hDerivativeQueue, sizeof(unsigned int) * 3 * totalCount);

    initDerivatives(1);

    if (statesTipsCount > 0) {
        kernels->PartialsStatesEdgeFirstDerivatives(
                dMultipleDerivatives,
                dStatesOrigin,
                dPartialsOrigin,
                dMatrices[0],
                dDerivativeQueue,
                dWeights[0], // TODO Use categoryWeightsIndices
                0, statesTipsCount, kPaddedPatternCount, kCategoryCount, false);
    }

    kernels->PartialsPartialsEdgeFirstDerivatives(
            dMultipleDerivatives,
            dPartialsOrigin,
            dMatrices[0],
            dDerivativeQueue,
            dWeights[0], // TODO Use categoryWeightsIndices
            statesTipsCount, (totalCount - statesTipsCount), kPaddedPatternCount, kCategoryCount, true);

    std::vector<Real> hTmp(totalCount * kPaddedPatternCount); // TODO Use existing buffer

    if (outFirstDerivatives != NULL) {

        gpu->MemcpyDeviceToHost(hTmp.data(), dMultipleDerivatives, sizeof(Real) * kPaddedPatternCount * totalCount);

        for (int i = 0; i < totalCount; ++i) {
            beagleMemCpy(outFirstDerivatives + i * kPatternCount,
                         hTmp.data() + i * kPaddedPatternCount,
                         kPatternCount);
        }
    }

    if (outSumFirstDerivatives != NULL || outSumSquaredFirstDerivatives != NULL) {

        int length = 0;
        if (outSumFirstDerivatives != NULL) {
            kernels->MultipleNodeSiteReduction(dMultipleDerivativeSum,
                                               dMultipleDerivatives,
                                               dPatternWeights,
                                               length,
                                               kPaddedPatternCount,
                                               totalCount);
            length += totalCount;
        }

        if (outSumSquaredFirstDerivatives != NULL) {
            kernels->MultipleNodeSiteSquaredReduction(dMultipleDerivativeSum,
                                                      dMultipleDerivatives,
                                                      dPatternWeights,
                                                      length,
                                                      kPaddedPatternCount,
                                                      totalCount);
            length += totalCount;
        }

        gpu->MemcpyDeviceToHost(hTmp.data(), dMultipleDerivativeSum, sizeof(Real) * length);

        length = 0;
        if (outSumFirstDerivatives != NULL) {
            beagleMemCpy(outSumFirstDerivatives, hTmp.data(), totalCount);
            length += totalCount;
        }

        if (outSumSquaredFirstDerivatives != NULL) {
            beagleMemCpy(outSumSquaredFirstDerivatives, hTmp.data() + length, totalCount);
        }
    }

    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
void BeagleGPUImpl<BEAGLE_GPU_GENERIC>::initDerivatives(int replicates) {

    int minSize = std::max(kPaddedPatternCount * kBufferCount,
            kPaddedPatternCount * kPaddedPatternCount * replicates);

    if (kMultipleDerivativesLength < minSize) {

        if (dMultipleDerivatives != (GPUPtr)NULL) {
            gpu->FreeMemory(dMultipleDerivatives);
        }

        dMultipleDerivatives = gpu->AllocateMemory(minSize * sizeof(Real));

        if (dMultipleDerivativeSum == (GPUPtr)NULL) {
            dMultipleDerivativeSum = gpu->AllocateMemory(kBufferCount * sizeof(Real));
        }

        kMultipleDerivativesLength = minSize;
    }
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::calcCrossProducts(const int *postBufferIndices,
                                                         const int *preBufferIndices,
                                                         const int *categoryRateIndices,
                                                         const int *categoryWeightIndices,
                                                         const double* edgeLengths,
                                                         int totalCount,
                                                         double *outCrossProducts) {

    int instructionOffset = 0;
    int statesTipsCount = 0;

    if (kCompactBufferCount > 0) {
        for (int i = 0; i < totalCount; ++i) {
            if (postBufferIndices[i] < kCompactBufferCount) {
                hDerivativeQueue[instructionOffset++] = hStatesOffsets[postBufferIndices[i]];
                hDerivativeQueue[instructionOffset++] = hPartialsOffsets[preBufferIndices[i]];
                ++statesTipsCount;
            }
        }
    }

    for (int i = 0; i < totalCount; i++) {
        if (postBufferIndices[i] >= kCompactBufferCount) {
            hDerivativeQueue[instructionOffset++] = hPartialsOffsets[postBufferIndices[i]];
            hDerivativeQueue[instructionOffset++] = hPartialsOffsets[preBufferIndices[i]];
        }
    }

    gpu->MemcpyHostToDevice(dDerivativeQueue, hDerivativeQueue, sizeof(unsigned int) * 2 * totalCount);

    const double* categoryRates = hCategoryRates[0]; // TODO parameterize index
    const GPUPtr categoryWeights = dWeights[0];

    int lengthCount = 0;
    for (int i = 0; i < totalCount; i++) {
        hDistanceQueue[lengthCount++] = (Real) edgeLengths[i];
    }
    for (int i = 0; i < kCategoryCount; i++) {
        hDistanceQueue[lengthCount++] = (Real) categoryRates[i];
    }

    unsigned int nodeBlocks = 8; // TODO Determine relatively good values
    unsigned int patternBlocks = 16; // TODO Determine relatively good value

    gpu->MemcpyHostToDevice(dDistanceQueue, hDistanceQueue, sizeof(Real) * lengthCount);

    initDerivatives(nodeBlocks * patternBlocks);

    bool accumulate = false;
    if (statesTipsCount > 0) {
       kernels->PartialsStatesCrossProducts(
               dMultipleDerivatives,
               dStatesOrigin,
               dPartialsOrigin,
               dDistanceQueue,
               dDerivativeQueue,
               categoryWeights, dPatternWeights,
               0, statesTipsCount, totalCount,
               kPaddedPatternCount, kCategoryCount, accumulate,
               nodeBlocks, patternBlocks);

        accumulate = true;
    }

    kernels->PartialsPartialsCrossProducts(
        dMultipleDerivatives,
        dPartialsOrigin,
        dDistanceQueue,
        dDerivativeQueue,
        categoryWeights, dPatternWeights,
        statesTipsCount, (totalCount - statesTipsCount), totalCount,
        kPaddedPatternCount, kCategoryCount, accumulate,
        nodeBlocks, patternBlocks
    );

    int replicates = nodeBlocks * patternBlocks;

    std::vector<Real> hTmp(kPaddedStateCount * kPaddedStateCount * replicates); // TODO Use existing buffer
    gpu->MemcpyDeviceToHost(hTmp.data(), dMultipleDerivatives, sizeof(Real) * kPaddedStateCount * kPaddedStateCount * replicates);

    for (int r = 1; r < replicates; r++) {
        for (int i = 0; i < kPaddedStateCount * kPaddedStateCount; i++) {
            hTmp[i] += hTmp[r * kPaddedStateCount * kPaddedStateCount + i];
        }
    }

    for (int i = 0; i < kStateCount; ++i) {
        beagleMemCpy(outCrossProducts + i * kStateCount,
                     hTmp.data() + i * kPaddedStateCount,
                     kStateCount);
    }


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
                                              int pluginResourceNumber,
                                              long preferenceFlags,
                                              long requirementFlags,
                                              int* errorCode) {
    BeagleImpl* impl = new BeagleGPUImpl<BEAGLE_GPU_GENERIC>();
    try {
        *errorCode =
            impl->createInstance(tipCount, partialsBufferCount, compactBufferCount, stateCount,
                                 patternCount, eigenBufferCount, matrixBufferCount,
                                 categoryCount,scaleBufferCount, resourceNumber, pluginResourceNumber, preferenceFlags, requirementFlags);
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

#ifdef CUDA
template<>
const char* BeagleGPUImplFactory<double>::getName() {
    return "GPU-DP-CUDA";
}

template<>
const char* BeagleGPUImplFactory<float>::getName() {
    return "GPU-SP-CUDA";
}
#elif defined(FW_OPENCL)
template<>
const char* BeagleGPUImplFactory<double>::getName() {
    return "DP-OpenCL";

}
template<>
const char* BeagleGPUImplFactory<float>::getName() {
    return "SP-OpenCL";
}
#endif

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
    long flags = BEAGLE_FLAG_COMPUTATION_SYNCH | BEAGLE_FLAG_COMPUTATION_ASYNCH |
          BEAGLE_FLAG_SCALING_MANUAL | BEAGLE_FLAG_SCALING_ALWAYS | BEAGLE_FLAG_SCALING_AUTO | BEAGLE_FLAG_SCALING_DYNAMIC |
          BEAGLE_FLAG_THREADING_NONE |
          BEAGLE_FLAG_VECTOR_NONE |
          BEAGLE_FLAG_SCALERS_LOG | BEAGLE_FLAG_SCALERS_RAW |
          BEAGLE_FLAG_EIGEN_COMPLEX | BEAGLE_FLAG_EIGEN_REAL |
          BEAGLE_FLAG_INVEVEC_STANDARD | BEAGLE_FLAG_INVEVEC_TRANSPOSED |
          BEAGLE_FLAG_PREORDER_TRANSPOSE_MANUAL | BEAGLE_FLAG_PREORDER_TRANSPOSE_AUTO |
          BEAGLE_FLAG_PARALLELOPS_GRID | BEAGLE_FLAG_PARALLELOPS_STREAMS;

#ifdef CUDA
    flags |= BEAGLE_FLAG_FRAMEWORK_CUDA |
             BEAGLE_FLAG_PROCESSOR_GPU;
#elif defined(FW_OPENCL)
    flags |= BEAGLE_FLAG_FRAMEWORK_OPENCL |
             BEAGLE_FLAG_PROCESSOR_CPU | BEAGLE_FLAG_PROCESSOR_GPU | BEAGLE_FLAG_PROCESSOR_OTHER;
#endif

    Real r = 0;
    modifyFlagsForPrecision(&flags, r);
    return flags;
}

} // end of device namespace
} // end of gpu namespace
} // end of beagle namespace
