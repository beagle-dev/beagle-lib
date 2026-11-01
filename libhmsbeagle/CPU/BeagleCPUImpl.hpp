/*
 *  BeagleCPUImpl.cpp
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
 * @author Andrew Rambaut
 * @author Marc Suchard
 * @author Daniel Ayres
 * @author Mark Holder
 */

///@TODO: wrap partials, eigen calcs, and transition matrices in a small structs
//      so that we can flag them. This would be helpful for
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
#include <algorithm>
#include <climits>
#include <cstdint>
#include <chrono>

#include "libhmsbeagle/beagle.h"
#include "libhmsbeagle/BufferCountRule.h"
#include "libhmsbeagle/CPU/Precision.h"
#include "libhmsbeagle/CPU/BeagleCPUImpl.h"
#include "libhmsbeagle/CPU/EigenDecompositionCube.h"
#include "libhmsbeagle/CPU/EigenDecompositionSquare.h"

// BEAGLE_INDEX_CHECKS (CMake option, default OFF): range-check buffer indices at the entry points
#ifdef BEAGLE_INDEX_CHECKS
#define BEAGLE_CHECK_INDICES(outOfRange) if (outOfRange) return BEAGLE_ERROR_OUT_OF_RANGE
#else
#define BEAGLE_CHECK_INDICES(outOfRange)
#endif

// When defined, calcAdjointCrossProducts (below) recomputes each branch's own
// per-site likelihood locally, via AdjointIntegralPlan::branchLikelihoodInEigenBasis,
// instead of reusing the shared root-based perSiteLikelihoods. This used to be
// (mis)placed in BeagleCPUSpectralImpl.hpp, past the point where BeagleCPUImpl.hpp
// (the actual owner of both #ifdef sites below) had already been preprocessed via
// BeagleCPUImpl.h's own #include of this file -- so it never took effect. Toggle it
// here instead, where it's actually reachable.
// #define USE_BRANCH_LIKELIHOOD

namespace beagle {
namespace cpu {

//#if definBUGGING_OUTPUT = false;
//#endif

BEAGLE_CPU_FACTORY_TEMPLATE
inline const char* getBeagleCPUName(){ return "CPU-Unknown"; };

template<>
inline const char* getBeagleCPUName<double>(){ return "CPU-Double"; };

template<>
inline const char* getBeagleCPUName<float>(){ return "CPU-Single"; };

BEAGLE_CPU_FACTORY_TEMPLATE
inline const long long getBeagleCPUFlags(){ return BEAGLE_FLAG_COMPUTATION_SYNCH; };

template<>
inline const long long getBeagleCPUFlags<double>(){ return BEAGLE_FLAG_COMPUTATION_SYNCH |
                                                      BEAGLE_FLAG_PROCESSOR_CPU |
                                                      BEAGLE_FLAG_PRECISION_DOUBLE |
                                                      BEAGLE_FLAG_VECTOR_NONE |
                                                      BEAGLE_FLAG_FRAMEWORK_CPU; };

template<>
inline const long long getBeagleCPUFlags<float>(){ return BEAGLE_FLAG_COMPUTATION_SYNCH |
                                                     BEAGLE_FLAG_PROCESSOR_CPU |
                                                     BEAGLE_FLAG_PRECISION_SINGLE |
                                                     BEAGLE_FLAG_VECTOR_NONE |
                                                     BEAGLE_FLAG_FRAMEWORK_CPU; };



BEAGLE_CPU_TEMPLATE
BeagleCPUImpl<BEAGLE_CPU_GENERIC>::~BeagleCPUImpl() {
    // free all that stuff...
    // If you delete partials, make sure not to delete the last element
    // which is TEMP_SCRATCH_PARTIAL twice.

    for(unsigned int i=0; i<kEigenDecompCount; i++) {
        if (gCategoryWeights[i] != NULL)
            free(gCategoryWeights[i]);
        if (gStateFrequencies[i] != NULL)
            free(gStateFrequencies[i]);
        if (gCategoryRates[i] != NULL)
            free(gCategoryRates[i]);
    }
    free(gCategoryWeights);
    free(gStateFrequencies);

    for(unsigned int i=0; i<kMatrixCount; i++) {
        if (gTransitionMatrices[i] != NULL)
            free(gTransitionMatrices[i]);
    }
    free(gTransitionMatrices);

    for (size_t i = 0; i < gEigenInfoChunks.size(); i++) {
        if (gEigenInfoChunks[i] != NULL)
            free(gEigenInfoChunks[i]);
    }

    if (gPartialsGrad != NULL) {
        free(gPartialsGrad);
    }

    if (coalescentGrad != NULL) {
        free(coalescentGrad);
    }

    for(unsigned int i=0; i<kBufferCount; i++) {
#ifndef BEAGLE_CACHE_FRIENDLY
        if (gPartials[i] != NULL)
            free(gPartials[i]);
#endif
        if (gTipStates[i] != NULL)
            free(gTipStates[i]);
    }
#ifdef BEAGLE_CACHE_FRIENDLY
    free(gPartialsCache);
#endif
    free(gPartials);
    free(gTipStates);

    if (kFlags & BEAGLE_FLAG_SCALING_AUTO) {
        for (int i = 0; i < kScaleBufferCount; i++) {
            if (gAutoScaleBuffers[i] != NULL)
                free(gAutoScaleBuffers[i]);
        }
        if (gAutoScaleBuffers)
            free(gAutoScaleBuffers);
        free(gActiveScalingFactors);
        if (gScaleBuffers[0] != NULL)
            free(gScaleBuffers[0]);
    } else {
        for (int i = 0; i < kScaleBufferCount; i++) {
            if (gScaleBuffers[i] != NULL)
                free(gScaleBuffers[i]);
        }
    }

    if (gScaleBuffers)
        free(gScaleBuffers);

    free(gCategoryRates);
    free(gPatternWeights);

    if (kPartitionsInitialised) {
        free(gPatternPartitions);
        free(gPatternPartitionsStartPatterns);
        if (kPatternsReordered) {
            free(gPatternsNewOrder);
        }
    }

    free(integrationTmp);
    free(firstDerivTmp);
    free(secondDerivTmp);

//    free(cLikelihoodTmp);
    free(grandDenominatorDerivTmp);
//    free(grandNumeratorUpperBoundDerivTmp);
//    free(grandNumeratorLowerBoundDerivTmp);
    free(grandNumeratorDerivTmp);

    if (crossProductNumeratorTmp != nullptr) {
        free(crossProductNumeratorTmp);
    }

    free(outLogLikelihoodsTmp);
    free(outFirstDerivativesTmp);
    free(outSecondDerivativesTmp);

    free(ones);
    free(zeros);

    delete gEigenDecomposition;

    stopThreads();

    // allocated also when auto-partitioning ends up disabled
    free(gAutoPartitionOperations);
    free(gAutoPartitionIndices);
    free(gAutoPartitionOutSumLogLikelihoods);
}

BEAGLE_CPU_TEMPLATE
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::stopThreads() {
    if (!kThreadingEnabled)
        return;

    // Send stop signal to all threads and join them...
    for (int i = 0; i < kNumThreads; i++) {
        threadData* td = &gThreads[i];
        std::unique_lock<std::mutex> l(td->m);
        td->stop = true;
        td->cv.notify_one();
    }

    // Join all the threads
    for (int i = 0; i < kNumThreads; i++) {
        threadData* td = &gThreads[i];
        td->t.join();
    }

    delete[] gThreads;
    delete[] gFutures;

    for (int i=0; i<kNumThreads; i++) {
        free(gThreadOperations[i]);
    }
    free(gThreadOperations);
    free(gThreadOpCounts);
    gThreadOperations = NULL;
    gThreadOpCounts = NULL;

    kThreadingEnabled = false;
}

BEAGLE_CPU_TEMPLATE
EigenDecomposition<BEAGLE_CPU_EIGEN_GENERIC>* BeagleCPUImpl<BEAGLE_CPU_GENERIC>::createEigenDecomposition(
        int decompositionCount,
        int stateCount,
        int categoryCount,
        long long flags) {

    EigenDecomposition<BEAGLE_CPU_EIGEN_GENERIC>* ed;

    if (kFlags & BEAGLE_FLAG_EIGEN_COMPLEX)
        ed = new EigenDecompositionSquare<BEAGLE_CPU_EIGEN_GENERIC>(kEigenDecompCount,
                kStateCount,kCategoryCount,kFlags);
    else
        ed = new EigenDecompositionCube<BEAGLE_CPU_EIGEN_GENERIC>(kEigenDecompCount,
                kStateCount, kCategoryCount,kFlags);

    return ed;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::createInstance(int tipCount,
                                  int partialsBufferCount,
                                  int compactBufferCount,
                                  int stateCount,
                                  int patternCount,
                                  int eigenDecompositionCount,
                                  int matrixCount,
                                  int categoryCount,
                                  int scaleBufferCount,
                                  int resourceNumber,
                                  int pluginResourceNumber,
                                  long long preferenceFlags,
                                  long long requirementFlags) {
    if (DEBUGGING_OUTPUT)
        std::cerr << "in BeagleCPUImpl::initialize\n" ;

    if (DOUBLE_PRECISION) {
        realtypeMin = DBL_MIN;
        scalingExponentThreshold = 200;
    } else {
        realtypeMin = FLT_MIN;
        scalingExponentThreshold = 20;
    }

    kBufferCount = partialsBufferCount + compactBufferCount;
    kCompactBufferCount = compactBufferCount;
    kTipCount = tipCount;
    assert(kBufferCount > kTipCount);
    kStateCount = stateCount;
    kPatternCount = patternCount;

    kPartitionCount = 1;
    kMaxPartitionCount = kPartitionCount;
    kPartitionsInitialised = false;
    kPatternsReordered = false;

    kInternalPartialsBufferCount = kBufferCount - kTipCount;

    kTransPaddedStateCount = kStateCount + T_PAD;
    kPartialsPaddedStateCount = kStateCount + P_PAD;

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

    kMatrixSize = (T_PAD + kStateCount) * kStateCount;

    int scaleBufferSize = kPaddedPatternCount;

    kFlags = 0;

    if (preferenceFlags & BEAGLE_FLAG_SCALING_AUTO || requirementFlags & BEAGLE_FLAG_SCALING_AUTO) {
        kFlags |= BEAGLE_FLAG_SCALING_AUTO;
        kFlags |= BEAGLE_FLAG_SCALERS_LOG;
        kScaleBufferCount = kInternalPartialsBufferCount;
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

    if (requirementFlags & BEAGLE_FLAG_EIGEN_COMPLEX || preferenceFlags & BEAGLE_FLAG_EIGEN_COMPLEX)
        kFlags |= BEAGLE_FLAG_EIGEN_COMPLEX;
    else
        kFlags |= BEAGLE_FLAG_EIGEN_REAL;

    if (requirementFlags & BEAGLE_FLAG_INVEVEC_TRANSPOSED || preferenceFlags & BEAGLE_FLAG_INVEVEC_TRANSPOSED)
        kFlags |= BEAGLE_FLAG_INVEVEC_TRANSPOSED;
    else
        kFlags |= BEAGLE_FLAG_INVEVEC_STANDARD;

    if (requirementFlags & BEAGLE_FLAG_THREADING_CPP || preferenceFlags & BEAGLE_FLAG_THREADING_CPP)
        kFlags |= BEAGLE_FLAG_THREADING_CPP;
    else if (requirementFlags & BEAGLE_FLAG_THREADING_OPENMP || preferenceFlags & BEAGLE_FLAG_THREADING_OPENMP)
        kFlags |= BEAGLE_FLAG_THREADING_OPENMP;
    else
        kFlags |= BEAGLE_FLAG_THREADING_NONE;

    gEigenDecomposition = createEigenDecomposition(kEigenDecompCount, kStateCount, kCategoryCount, kFlags);

    gCategoryRates = (double**) calloc(sizeof(double), kEigenDecompCount);
    if (gCategoryRates == NULL)
        throw std::bad_alloc();

    gPatternWeights = (double*) malloc(sizeof(double) * kPatternCount);
    if (gPatternWeights == NULL)
        throw std::bad_alloc();

    // TODO: if pattern padding is implemented this will create problems with setTipPartials
    kPartialsSize = kPaddedPatternCount * kPartialsPaddedStateCount * kCategoryCount;

    gPartials = (REALTYPE**) malloc(sizeof(REALTYPE*) * kBufferCount);
    if (gPartials == NULL)
     throw std::bad_alloc();


    gPartialsGrad = NULL;
    kPartialsGradCount = 0;
    coalescentGrad = NULL;
    kCoalescentGradLength = 0;
    kBastaBuffersAllocated = false;
    kAdjointBuffersAllocated = false;
    kUseAdjointGradient = false;
    kAdjointReversePassDone = false;
    kUnclampedForwardDone = false;
    kBastaNumThreads = 1;

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

#ifdef BEAGLE_CACHE_FRIENDLY
    gPartialsCache = (REALTYPE*) mallocAligned(sizeof(REALTYPE) * kPartialsSize * (kBufferCount - kTipCount));
#endif

    for (int i = kTipCount; i < kBufferCount; i++) {
#ifdef BEAGLE_CACHE_FRIENDLY
        gPartials[i] = gPartialsCache + (i - kTipCount) * kPartialsSize;
#else
        gPartials[i] = (REALTYPE*) mallocAligned(sizeof(REALTYPE) * kPartialsSize);
#endif
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


            if (kFlags & BEAGLE_FLAG_SCALING_DYNAMIC) {
                for (int j=0; j < scaleBufferSize; j++) {
                    gScaleBuffers[i][j] = 1.0;
                }
            }
        }
    }


    gTransitionMatrices = (REALTYPE**) malloc(sizeof(REALTYPE*) * kMatrixCount);
    if (gTransitionMatrices == NULL)
        throw std::bad_alloc();
    const bool matricesOnDemand = denseMatricesOnDemand();
    for (int i = 0; i < kMatrixCount; i++) {
        gTransitionMatrices[i] = NULL;
        if (!matricesOnDemand) {
            gTransitionMatrices[i] = (REALTYPE*) mallocAligned(sizeof(REALTYPE) * kMatrixSize * kCategoryCount);
            if (gTransitionMatrices[i] == 0L)
                throw std::bad_alloc();
        }
    }

    integrationTmp = (REALTYPE*) mallocAligned(sizeof(REALTYPE) * kPatternCount * kStateCount);
    firstDerivTmp = (REALTYPE*) malloc(sizeof(REALTYPE) * kPatternCount * kStateCount);
    secondDerivTmp = (REALTYPE*) malloc(sizeof(REALTYPE) * kPatternCount * kStateCount);

//    cLikelihoodTmp = (REALTYPE*) mallocAligned(sizeof(REALTYPE) * kPatternCount * kCategoryCount);
    grandDenominatorDerivTmp = (REALTYPE*) mallocAligned(sizeof(REALTYPE) * kPaddedPatternCount); // TODO Deprecate in favor of integrationTmp
    grandNumeratorDerivTmp = (REALTYPE*) mallocAligned(sizeof(REALTYPE) * kPaddedPatternCount);
    crossProductNumeratorTmp = nullptr;
//    grandNumeratorLowerBoundDerivTmp = (REALTYPE*) mallocAligned(sizeof(REALTYPE) * kPatternCount);
//    grandNumeratorUpperBoundDerivTmp = (REALTYPE*) mallocAligned(sizeof(REALTYPE) * kPatternCount);

    outLogLikelihoodsTmp = (REALTYPE*) malloc(sizeof(REALTYPE) * kPatternCount * kStateCount);
    outFirstDerivativesTmp = (REALTYPE*) malloc(sizeof(REALTYPE) * kPatternCount * kStateCount);
    outSecondDerivativesTmp = (REALTYPE*) malloc(sizeof(REALTYPE) * kPatternCount * kStateCount);

    zeros = (REALTYPE*) malloc(sizeof(REALTYPE) * kPaddedPatternCount);
    ones = (REALTYPE*) malloc(sizeof(REALTYPE) * kPaddedPatternCount);
    for(int i = 0; i < kPaddedPatternCount; i++) {
        zeros[i] = 0.0;
        ones[i] = 1.0;
    }

    kThreadingEnabled = false;
    kAutoPartitioningEnabled = false;
    gThreadOperations = NULL;
    gThreadOpCounts = NULL;
    gAutoPartitionOperations = NULL;
    gAutoPartitionIndices = NULL;
    gAutoPartitionOutSumLogLikelihoods = NULL;
    if (kFlags & BEAGLE_FLAG_THREADING_CPP) {
        int hardwareThreads = std::thread::hardware_concurrency();
        if (kStateCount <= 4) {
            kMinPatternCount = BEAGLE_CPU_ASYNC_MIN_PATTERN_COUNT_LOW;
            if (hardwareThreads < BEAGLE_CPU_ASYNC_HW_THREAD_COUNT_THRESHOLD) {
                kMinPatternCount = BEAGLE_CPU_ASYNC_MIN_PATTERN_COUNT_HIGH;
            } else if (kPatternCount < BEAGLE_CPU_ASYNC_LIMIT_PATTERN_COUNT) {
                hardwareThreads = BEAGLE_CPU_ASYNC_HW_THREAD_COUNT_THRESHOLD;
            }
        } else {
            // todo: assess minimum pattern count for efficient auto-threading
            //       for higher state-count values
            kMinPatternCount = 2;
        }
        if (kPatternCount >= kMinPatternCount && hardwareThreads > 2) {
            int partitionCount = kPatternCount/(kMinPatternCount/2);
            if (partitionCount > hardwareThreads/2) {
                partitionCount = hardwareThreads/2;
            }

            std::vector<int> partitionForPattern(kPatternCount);
            for (int p=0; p<kPatternCount; p++) {
                partitionForPattern[p] = (p * partitionCount) / kPatternCount;
            }
            setPatternPartitions(partitionCount, partitionForPattern.data());

            gAutoPartitionOperations = (int*) malloc(sizeof(int) * kBufferCount * kPartitionCount * BEAGLE_PARTITION_OP_COUNT);

            if (kPatternCount >= kMinPatternCount*4) {
                gAutoPartitionIndices = (int*) malloc(sizeof(int) * partitionCount);
                for (int i=0; i<partitionCount; i++) {
                    gAutoPartitionIndices[i] = i;
                }
                gAutoPartitionOutSumLogLikelihoods = (double*) malloc(sizeof(double) * partitionCount);
                kAutoRootPartitioningEnabled = false; //TODO: XJ: need to change this back to true
            }
            // TODO: XJ: need to change below back to true
            kAutoPartitioningEnabled = false;
        }
    }

    gBranchEigenInfo.resize(kMatrixCount);

    gPartialTmp1.resize(kPartialsPaddedStateCount * kPartitionCount);
    gPartialTmp2.resize(kPartialsPaddedStateCount * kPartitionCount);

    // chunk layout: time[32 * C], then expat, cosbt, sinbt, expatcosbt and expatsinbt, each [32 * C * (S + P_PAD)],
    // every sub-array starting at a multiple of 64 bytes from the chunk base
    const size_t chunkMatrices = size_t(1) << kEigenInfoChunkShift;
    const size_t alignedLength = 64 / sizeof(REALTYPE);
    const size_t subArrayLength[6] = {
            chunkMatrices * kCategoryCount,
            chunkMatrices * kCategoryCount * kPartialsPaddedStateCount,
            chunkMatrices * kCategoryCount * kPartialsPaddedStateCount,
            chunkMatrices * kCategoryCount * kPartialsPaddedStateCount,
            chunkMatrices * kCategoryCount * kPartialsPaddedStateCount,
            chunkMatrices * kCategoryCount * kPartialsPaddedStateCount};
    size_t chunkLength = 0;
    for (int i = 0; i < 6; i++) {
        kEigenInfoChunkOffset[i] = chunkLength;
        chunkLength += (subArrayLength[i] + alignedLength - 1) / alignedLength * alignedLength;
    }
    kEigenInfoChunkSize = chunkLength * sizeof(REALTYPE);
    gEigenInfoChunks.assign((kMatrixCount + chunkMatrices - 1) >> kEigenInfoChunkShift, NULL);

    return BEAGLE_SUCCESS;
}

BEAGLE_CPU_TEMPLATE
const char* BeagleCPUImpl<BEAGLE_CPU_GENERIC>::getName() {
    return getBeagleCPUName<BEAGLE_CPU_FACTORY_GENERIC>();
}

BEAGLE_CPU_TEMPLATE
const long long BeagleCPUImpl<BEAGLE_CPU_GENERIC>::getFlags() {
    return getBeagleCPUFlags<BEAGLE_CPU_FACTORY_GENERIC>();
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::getInstanceDetails(BeagleInstanceDetails* returnInfo) {
    if (returnInfo != NULL) {
        returnInfo->resourceNumber = 0;
        returnInfo->flags = getFlags();
        returnInfo->flags |= kFlags;

        returnInfo->implName = (char*) getName();
    }

    return BEAGLE_SUCCESS;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::setCPUThreadCount(int threadCount) {

    if (threadCount < 1)
        return BEAGLE_ERROR_OUT_OF_RANGE;

    // the previous pool and auto-partition buffers; setPatternPartitions starts a new pool
    stopThreads();
    free(gAutoPartitionOperations);
    free(gAutoPartitionIndices);
    free(gAutoPartitionOutSumLogLikelihoods);
    gAutoPartitionOperations = NULL;
    gAutoPartitionIndices = NULL;
    gAutoPartitionOutSumLogLikelihoods = NULL;
    kAutoPartitioningEnabled = false;
    kAutoRootPartitioningEnabled = false;
    if (kFlags & BEAGLE_FLAG_THREADING_CPP) {
        int hardwareThreads = std::thread::hardware_concurrency();
        if (kStateCount <= 4) {
            kMinPatternCount = BEAGLE_CPU_ASYNC_MIN_PATTERN_COUNT_LOW;
            if (hardwareThreads < BEAGLE_CPU_ASYNC_HW_THREAD_COUNT_THRESHOLD) {
                kMinPatternCount = BEAGLE_CPU_ASYNC_MIN_PATTERN_COUNT_HIGH;
            }
        } else {
            // todo: assess minimum pattern count for efficient auto-threading
            //       for higher state-count values
            kMinPatternCount = 2;
        }
        if (kPatternCount >= kMinPatternCount && hardwareThreads > 2) {
            int partitionCount = kPatternCount/(kMinPatternCount/2);
            if (partitionCount > threadCount) {
                partitionCount = threadCount;
            }

            std::vector<int> partitionForPattern(kPatternCount);
            for (int p=0; p<kPatternCount; p++) {
                partitionForPattern[p] = (p * partitionCount) / kPatternCount;
            }
            setPatternPartitions(partitionCount, partitionForPattern.data());

            gAutoPartitionOperations = (int*) malloc(sizeof(int) * kBufferCount * kPartitionCount * BEAGLE_PARTITION_OP_COUNT);

            if (kPatternCount >= kMinPatternCount*4) {
                gAutoPartitionIndices = (int*) malloc(sizeof(int) * partitionCount);
                for (int i=0; i<partitionCount; i++) {
                    gAutoPartitionIndices[i] = i;
                }
                gAutoPartitionOutSumLogLikelihoods = (double*) malloc(sizeof(double) * partitionCount);
                kAutoRootPartitioningEnabled = true;
            }

            kAutoPartitioningEnabled = true;
        }
    }

    gPartialTmp1.resize(kPartialsPaddedStateCount * kPartitionCount);
    gPartialTmp2.resize(kPartialsPaddedStateCount * kPartitionCount);

    return BEAGLE_SUCCESS;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::ensureBufferCounts(int partialsBufferCount,
                                                          int matrixBufferCount,
                                                          int scaleBufferCount) {
    // BASTA's arrays sized by these counts (gEdgeLengthsGrad, the legacy gPartialsGrad) do not grow with them: once
    // allocateBastaBuffers or allocateBastaGradBuffers has run, a growth is refused and changes nothing
    if (kBastaBuffersAllocated) {
        BufferCounts counts;
        int ruleCode;
        if (ensureBufferCountsRule(partialsBufferCount, matrixBufferCount, scaleBufferCount, kTipCount,
                                   kCompactBufferCount, kBufferCount, kMatrixCount, kScaleBufferCount, kFlags,
                                   &counts, &ruleCode))
            return BEAGLE_ERROR_NO_IMPLEMENTATION;
    }
    return growBufferCounts(partialsBufferCount, matrixBufferCount, scaleBufferCount);
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::growBufferCounts(int partialsBufferCount,
                                                        int matrixBufferCount,
                                                        int scaleBufferCount) {
    if (partialsBufferCount < 0 || matrixBufferCount < 0 || scaleBufferCount < 0)
        return BEAGLE_ERROR_OUT_OF_RANGE;

    // new counts; a request at or below a current count leaves that kind unchanged
    const long long requestedBuffers = (long long) partialsBufferCount + kCompactBufferCount;
    const int newBuffer = (requestedBuffers > kBufferCount) ? (int) std::min<long long>(requestedBuffers, INT_MAX)
                                                            : kBufferCount;
    const int newInternal = newBuffer - kTipCount;
    const int newMatrix = (matrixBufferCount > kMatrixCount) ? matrixBufferCount : kMatrixCount;
    long long requestedScale = scaleBufferCount; // as in createInstance
    if (kFlags & BEAGLE_FLAG_SCALING_AUTO)
        requestedScale = newInternal;
    else if (kFlags & BEAGLE_FLAG_SCALING_ALWAYS)
        requestedScale = (long long) newInternal + 1; // + 1 for the temporary buffer of the edge likelihoods
    const int newScale = (requestedScale > kScaleBufferCount) ? (int) std::min<long long>(requestedScale, INT_MAX)
                                                              : kScaleBufferCount;

    if (requestedBuffers > INT_MAX || requestedScale > INT_MAX)
        return BEAGLE_ERROR_OUT_OF_RANGE;

    const bool growPartials = newBuffer > kBufferCount;
    const bool growMatrices = newMatrix > kMatrixCount;
    const bool growScale = newScale > kScaleBufferCount;
    if (!growPartials && !growMatrices && !growScale)
        return BEAGLE_SUCCESS;

    // counts whose sizes overflow, here or in later int arithmetic (the operation arrays of the threads)
    const long long operationsLength = (long long) BEAGLE_PARTITION_OP_COUNT * newBuffer * kPartitionCount;
    if ((growPartials && (operationsLength > INT_MAX || (size_t) operationsLength > SIZE_MAX / sizeof(int) ||
                          (size_t) newBuffer > SIZE_MAX / sizeof(REALTYPE*))) ||
        (growMatrices && ((size_t) newMatrix > SIZE_MAX / sizeof(REALTYPE*) ||
                          (size_t) newMatrix > gBranchEigenInfo.max_size())) ||
        (growScale && (size_t) newScale > SIZE_MAX / sizeof(REALTYPE*)))
        return BEAGLE_ERROR_OUT_OF_RANGE;

    // Phase 1: allocate everything into locals, leaving the instance untouched until all of it succeeded

    const size_t chunkMatrices = size_t(1) << kEigenInfoChunkShift;
    const int scaleBufferSize = kPaddedPatternCount;
    const bool autoScaling = (kFlags & BEAGLE_FLAG_SCALING_AUTO) != 0;

    REALTYPE** newPartials = NULL;
    int** newTipStates = NULL;
    REALTYPE** newScaleBuffers = NULL;
    signed short** newAutoScaleBuffers = NULL;
    int* newActiveScalingFactors = NULL;
    REALTYPE** newTransitionMatrices = NULL;
    int** newThreadOperations = NULL;  // one replacement array per thread
    int* newAutoPartitionOperations = NULL;
    bool ok = true;

    if (growPartials) {
        newPartials = (REALTYPE**) calloc(newBuffer, sizeof(REALTYPE*)); // new entries NULL until allocated
        newTipStates = (int**) calloc(newBuffer, sizeof(int*));
        ok = (newPartials != NULL && newTipStates != NULL);
        if (ok) {
            memcpy(newPartials, gPartials, sizeof(REALTYPE*) * kBufferCount);
            memcpy(newTipStates, gTipStates, sizeof(int*) * kBufferCount);
            for (int i = kBufferCount; ok && i < newBuffer; i++) { // all internal, since kBufferCount > kTipCount
                newPartials[i] = (REALTYPE*) mallocAligned(sizeof(REALTYPE) * kPartialsSize);
                ok = (newPartials[i] != NULL);
            }
        }
    }

    if (ok && growScale) {
        if (autoScaling) {
            newAutoScaleBuffers = (signed short**) calloc(newScale, sizeof(signed short*));
            newActiveScalingFactors = (int*) malloc(sizeof(int) * newInternal);
            ok = (newAutoScaleBuffers != NULL && newActiveScalingFactors != NULL);
            if (ok) {
                memcpy(newAutoScaleBuffers, gAutoScaleBuffers, sizeof(signed short*) * kScaleBufferCount);
                for (int i = kScaleBufferCount; ok && i < newScale; i++) {
                    newAutoScaleBuffers[i] = (signed short*) malloc(sizeof(signed short) * scaleBufferSize);
                    ok = (newAutoScaleBuffers[i] != NULL);
                }
                memcpy(newActiveScalingFactors, gActiveScalingFactors, sizeof(int) * kInternalPartialsBufferCount);
                for (int i = kInternalPartialsBufferCount; i < newInternal; i++)
                    newActiveScalingFactors[i] = 0;
            }
        } else {
            newScaleBuffers = (REALTYPE**) calloc(newScale, sizeof(REALTYPE*));
            ok = (newScaleBuffers != NULL);
            if (ok) {
                memcpy(newScaleBuffers, gScaleBuffers, sizeof(REALTYPE*) * kScaleBufferCount);
                for (int i = kScaleBufferCount; ok && i < newScale; i++) {
                    newScaleBuffers[i] = (REALTYPE*) malloc(sizeof(REALTYPE) * scaleBufferSize);
                    ok = (newScaleBuffers[i] != NULL);
                    if (ok && (kFlags & BEAGLE_FLAG_SCALING_DYNAMIC)) {
                        for (int j = 0; j < scaleBufferSize; j++)
                            newScaleBuffers[i][j] = 1.0;
                    }
                }
            }
        }
    }

    if (ok && growMatrices) {
        newTransitionMatrices = (REALTYPE**) calloc(newMatrix, sizeof(REALTYPE*)); // NULL: dense on demand
        ok = (newTransitionMatrices != NULL);
        if (ok) {
            memcpy(newTransitionMatrices, gTransitionMatrices, sizeof(REALTYPE*) * kMatrixCount);
            if (!denseMatricesOnDemand()) {
                for (int i = kMatrixCount; ok && i < newMatrix; i++) {
                    newTransitionMatrices[i] = (REALTYPE*) mallocAligned(sizeof(REALTYPE) * kMatrixSize * kCategoryCount);
                    ok = (newTransitionMatrices[i] != NULL);
                }
            }
        }
        if (ok) {
            try { // capacity only; the contents and sizes are unchanged until phase 2
                gBranchEigenInfo.reserve(newMatrix);
                gEigenInfoChunks.reserve((newMatrix + chunkMatrices - 1) >> kEigenInfoChunkShift);
            } catch (std::bad_alloc&) {
                ok = false;
            }
        }
    }

    // operation lists sized by the buffer count; their contents are rebuilt on every call
    const size_t operationsSize = sizeof(int) * (size_t) operationsLength;
    if (ok && growPartials && kThreadingEnabled) {
        newThreadOperations = (int**) calloc(kNumThreads, sizeof(int*));
        ok = (newThreadOperations != NULL);
        for (int i = 0; ok && i < kNumThreads; i++) {
            newThreadOperations[i] = (int*) malloc(operationsSize);
            ok = (newThreadOperations[i] != NULL);
        }
    }
    if (ok && growPartials && kAutoPartitioningEnabled) {
        newAutoPartitionOperations = (int*) malloc(operationsSize);
        ok = (newAutoPartitionOperations != NULL);
    }

    if (!ok) {
        if (newPartials != NULL) {
            for (int i = kBufferCount; i < newBuffer; i++)
                free(newPartials[i]);
        }
        free(newPartials);
        free(newTipStates);
        if (newAutoScaleBuffers != NULL) {
            for (int i = kScaleBufferCount; i < newScale; i++)
                free(newAutoScaleBuffers[i]);
        }
        free(newAutoScaleBuffers);
        free(newActiveScalingFactors);
        if (newScaleBuffers != NULL) {
            for (int i = kScaleBufferCount; i < newScale; i++)
                free(newScaleBuffers[i]);
        }
        free(newScaleBuffers);
        if (newTransitionMatrices != NULL) {
            for (int i = kMatrixCount; i < newMatrix; i++)
                free(newTransitionMatrices[i]);
        }
        free(newTransitionMatrices);
        if (newThreadOperations != NULL) {
            for (int i = 0; i < kNumThreads; i++)
                free(newThreadOperations[i]);
        }
        free(newThreadOperations);
        free(newAutoPartitionOperations);
        return BEAGLE_ERROR_OUT_OF_MEMORY;
    }

    // Phase 2: commit; nothing here can fail. Buffers keep their addresses, so every index valid before keeps its
    // contents, and BranchEigenInfo pointers into eigen-info chunks stay valid when the structs move.

    if (growPartials) {
        free(gPartials);
        free(gTipStates);
        gPartials = newPartials;
        gTipStates = newTipStates;
    }
    if (growScale) {
        if (autoScaling) {
            free(gAutoScaleBuffers);
            free(gActiveScalingFactors);
            gAutoScaleBuffers = newAutoScaleBuffers;
            gActiveScalingFactors = newActiveScalingFactors;
        } else {
            free(gScaleBuffers);
            gScaleBuffers = newScaleBuffers;
        }
    }
    if (growMatrices) {
        free(gTransitionMatrices);
        gTransitionMatrices = newTransitionMatrices;
        gBranchEigenInfo.resize(newMatrix); // within the reserved capacity; new entries are "never updated"
        gEigenInfoChunks.resize((newMatrix + chunkMatrices - 1) >> kEigenInfoChunkShift, NULL);
    }
    if (newThreadOperations != NULL) {
        for (int i = 0; i < kNumThreads; i++) {
            free(gThreadOperations[i]);
            gThreadOperations[i] = newThreadOperations[i];
        }
        free(newThreadOperations);
    }
    if (newAutoPartitionOperations != NULL) {
        free(gAutoPartitionOperations);
        gAutoPartitionOperations = newAutoPartitionOperations;
    }

    kBufferCount = newBuffer;
    kInternalPartialsBufferCount = newInternal;
    kMatrixCount = newMatrix;
    kScaleBufferCount = newScale;

    return BEAGLE_SUCCESS;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::setTipStates(int tipIndex,
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

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::setTipPartials(int tipIndex,
                                  const double* inPartials) {
    if (tipIndex < 0 || tipIndex >= kTipCount)
        return BEAGLE_ERROR_OUT_OF_RANGE;
    if(gPartials[tipIndex] == NULL) {
        gPartials[tipIndex] = (REALTYPE*) mallocAligned(sizeof(REALTYPE) * kPartialsSize);
        // TODO: What if this throws a memory full error?
        if (gPartials[tipIndex] == 0L)
            return BEAGLE_ERROR_OUT_OF_MEMORY;
    }

    const double* inPartialsOffset;
    REALTYPE* tmpRealPartialsOffset = gPartials[tipIndex];
    for (int l = 0; l < kCategoryCount; l++) {
        inPartialsOffset = inPartials;
        for (int i = 0; i < kPatternCount; i++) {
            beagleMemCpy(tmpRealPartialsOffset, inPartialsOffset, kStateCount);
            tmpRealPartialsOffset += kStateCount;
            // Pad extra buffer with zeros
            for(int k = kStateCount; k < kPartialsPaddedStateCount; k++) {
                *tmpRealPartialsOffset++ = 0;
            }
            inPartialsOffset += kStateCount;
        }
        // Pad extra buffer with zeros
        for(int k = 0; k < kPartialsPaddedStateCount * (kPaddedPatternCount - kPatternCount); k++) {
            *tmpRealPartialsOffset++ = 0;
        }
    }

    return BEAGLE_SUCCESS;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::setRootPrePartials(const int *bufferIndices,
                                                          const int *stateFrequenciesIndices,
                                                          int count) {
    if (count == 1) {
        // the state frequencies in every pattern, whether or not the patterns are partitioned
        int stateFrequenciesIndex = stateFrequenciesIndices[0];
        int bufferIndex = bufferIndices[0];
        if (bufferIndex < 0 || bufferIndex >= kBufferCount)
            return BEAGLE_ERROR_OUT_OF_RANGE;
        if (gPartials[bufferIndex] == NULL) {
            gPartials[bufferIndex] = (REALTYPE *) malloc(sizeof(REALTYPE) * kPartialsSize);
            if (gPartials[bufferIndex] == 0L)
                return BEAGLE_ERROR_OUT_OF_MEMORY;
        }
        const REALTYPE *inPartialsOffset = gStateFrequencies[stateFrequenciesIndex];
        REALTYPE *tmpRealPartialsOffset = gPartials[bufferIndex];
        for (int l = 0; l < kCategoryCount; l++) {
            for (int i = 0; i < kPatternCount; i++) {
                beagleMemCpy(tmpRealPartialsOffset, inPartialsOffset, kStateCount);
                tmpRealPartialsOffset += kStateCount;
                // Pad extra buffer with zeros
                for(int k = kStateCount; k < kPartialsPaddedStateCount; k++) {
                    *tmpRealPartialsOffset++ = 0;
                }
            }
            // Pad extra buffer with zeros
            for (int k = 0; k < kPartialsPaddedStateCount * (kPaddedPatternCount - kPatternCount); k++) {
                *tmpRealPartialsOffset++ = 0;
            }
        }

        return BEAGLE_SUCCESS;
    }
    return BEAGLE_ERROR_NO_IMPLEMENTATION;

}


BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::setPartials(int bufferIndex,
                               const double* inPartials) {
    if (bufferIndex < 0 || bufferIndex >= kBufferCount)
        return BEAGLE_ERROR_OUT_OF_RANGE;
    if (gPartials[bufferIndex] == NULL) {
        gPartials[bufferIndex] = (REALTYPE*) malloc(sizeof(REALTYPE) * kPartialsSize);
        if (gPartials[bufferIndex] == 0L)
            return BEAGLE_ERROR_OUT_OF_MEMORY;
    }

    const double* inPartialsOffset = inPartials;
    REALTYPE* tmpRealPartialsOffset = gPartials[bufferIndex];
    for (int l = 0; l < kCategoryCount; l++) {
        for (int i = 0; i < kPatternCount; i++) {
            beagleMemCpy(tmpRealPartialsOffset, inPartialsOffset, kStateCount);
            tmpRealPartialsOffset += kStateCount;
            // Pad extra buffer with zeros
            for(int k = kStateCount; k < kPartialsPaddedStateCount; k++) {
                *tmpRealPartialsOffset++ = 0;
            }
            inPartialsOffset += kStateCount;
        }
        // Pad extra buffer with zeros
        for(int k = 0; k < kPartialsPaddedStateCount * (kPaddedPatternCount - kPatternCount); k++) {
            *tmpRealPartialsOffset++ = 0;
        }
    }

    return BEAGLE_SUCCESS;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::getPartials(int bufferIndex,
                               int cumulativeScaleIndex,
                               double* outPartials) {

    // TODO: Test with and without padding
    if (bufferIndex < 0 || bufferIndex >= kBufferCount)
        return BEAGLE_ERROR_OUT_OF_RANGE;
    BEAGLE_CHECK_INDICES(cumulativeScaleIndex != BEAGLE_OP_NONE && badScale(cumulativeScaleIndex));

    if ((kPatternCount == kPaddedPatternCount) && (kStateCount == kPartialsPaddedStateCount)) {
        beagleMemCpy(outPartials, gPartials[bufferIndex], kPartialsSize);
    } else if (kStateCount == kPartialsPaddedStateCount) {
        double *offsetOutPartials = outPartials;
        REALTYPE* offsetBeaglePartials = gPartials[bufferIndex];
        for(int l = 0; l < kCategoryCount; l++) {
            beagleMemCpy(offsetOutPartials, offsetBeaglePartials, kPatternCount * kStateCount);
            offsetOutPartials += kPatternCount * kStateCount;
            offsetBeaglePartials += kPaddedPatternCount * kStateCount;
        }
    } else {
        double *offsetOutPartials = outPartials;
        REALTYPE* offsetBeaglePartials = gPartials[bufferIndex];
        for(int l = 0; l < kCategoryCount; l++) {
            for (int i = 0; i < kPatternCount; i++) {
                beagleMemCpy(offsetOutPartials, offsetBeaglePartials, kStateCount);
                offsetOutPartials += kStateCount;
                offsetBeaglePartials += kPartialsPaddedStateCount;
            }
            offsetBeaglePartials += (kPaddedPatternCount - kPatternCount) * kPartialsPaddedStateCount;
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

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::getPartialsBatch(const int* bufferIndices,
                                        const int* scaleIndices,
                                        int count,
                                        double* outPartials) {
    const int stride = kPatternCount * kStateCount * kCategoryCount;
    for (int i = 0; i < count; i++) {
        int returnValue = getPartials(bufferIndices[i], scaleIndices[i], outPartials);
        if (returnValue != BEAGLE_SUCCESS)
            return returnValue;
        outPartials += stride;
    }
    return BEAGLE_SUCCESS;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::setEigenDecomposition(int eigenIndex,
                                         const double* inEigenVectors,
                                         const double* inInverseEigenVectors,
                                         const double* inEigenValues) {

    gEigenDecomposition->setEigenDecomposition(eigenIndex, inEigenVectors, inInverseEigenVectors, inEigenValues);
    return BEAGLE_SUCCESS;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::setCategoryRates(const double* inCategoryRates) {
    int categoryRatesIndex=0;
    if (gCategoryRates[categoryRatesIndex] == NULL) {
        gCategoryRates[categoryRatesIndex] = (double*) malloc(sizeof(double) * kCategoryCount);
        if (gCategoryRates[categoryRatesIndex] == 0L)
            return BEAGLE_ERROR_OUT_OF_MEMORY;
    }
    memcpy(gCategoryRates[categoryRatesIndex], inCategoryRates, sizeof(double) * kCategoryCount);
    return BEAGLE_SUCCESS;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::setCategoryRatesWithIndex(int categoryRatesIndex,
                                                                 const double* inCategoryRates) {
    if (categoryRatesIndex < 0 || categoryRatesIndex >= kEigenDecompCount)
        return BEAGLE_ERROR_OUT_OF_RANGE;
    if (gCategoryRates[categoryRatesIndex] == NULL) {
        gCategoryRates[categoryRatesIndex] = (double*) malloc(sizeof(double) * kCategoryCount);
        if (gCategoryRates[categoryRatesIndex] == 0L)
            return BEAGLE_ERROR_OUT_OF_MEMORY;
    }
    memcpy(gCategoryRates[categoryRatesIndex], inCategoryRates, sizeof(double) * kCategoryCount);
    return BEAGLE_SUCCESS;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::setPatternWeights(const double* inPatternWeights) {
    assert(inPatternWeights != 0L);
    memcpy(gPatternWeights, inPatternWeights, sizeof(double) * kPatternCount);
    return BEAGLE_SUCCESS;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::setPatternPartitions(int partitionCount,
                                                            const int* inPatternPartitions) {

    int returnCode = BEAGLE_SUCCESS;

    assert(partitionCount > 0);
    assert(inPatternPartitions != 0L);

    kPartitionCount = partitionCount;
    gAdjointPartitionBuffers.resize(kStateCount * kStateCount * kPartitionCount, REALTYPE(0));
    free(grandDenominatorDerivTmp);
    free(grandNumeratorDerivTmp);
    free(firstDerivTmp);
    free(secondDerivTmp);
    grandDenominatorDerivTmp = (REALTYPE*) mallocAligned(sizeof(REALTYPE) * (kPaddedPatternCount + 1) * kPartitionCount);
    grandNumeratorDerivTmp = (REALTYPE*) mallocAligned(sizeof(REALTYPE) * (kPaddedPatternCount + 1)* kPartitionCount);
    firstDerivTmp =  (REALTYPE*) mallocAligned(sizeof(REALTYPE) * kPaddedPatternCount * kStateCount * kPartitionCount);
    secondDerivTmp =  (REALTYPE*) mallocAligned(sizeof(REALTYPE) * kPaddedPatternCount * kStateCount * kPartitionCount);


    if (!kPartitionsInitialised) {
        gPatternPartitions = (int*) malloc(sizeof(int) * kPatternCount);
        if (gPatternPartitions == NULL)
            throw std::bad_alloc();

        if (kAutoPartitioningEnabled) {
            free(gAutoPartitionOperations);
            if (kAutoRootPartitioningEnabled) {
                free(gAutoPartitionIndices);
                free(gAutoPartitionOutSumLogLikelihoods);
                kAutoRootPartitioningEnabled = false;
            }
            kAutoPartitioningEnabled = false;
        }
    }
    if (!kPartitionsInitialised || partitionCount > kMaxPartitionCount) {
        if (kPartitionsInitialised) {
            free(gPatternPartitionsStartPatterns);
        }
        gPatternPartitionsStartPatterns = (int*) malloc(sizeof(int) * (partitionCount+1));
        if (gPatternPartitionsStartPatterns == NULL)
            throw std::bad_alloc();

        kMaxPartitionCount = partitionCount;
    }

    stopThreads();

    if (kFlags & BEAGLE_FLAG_THREADING_CPP) {
        kNumThreads = partitionCount;

        gThreads = new threadData[kNumThreads];
        for (int i = 0; i < kNumThreads; i++) {
            gThreads[i].t = std::thread(&BeagleCPUImpl<BEAGLE_CPU_GENERIC>::threadWaiting, this, &gThreads[i]);
        }

        gFutures = new std::shared_future<void>[kNumThreads];
        if (gFutures == NULL)
            throw std::bad_alloc();

        gThreadOperations = (int**) malloc(sizeof(int*) * kNumThreads);
        for (int i=0; i<kNumThreads; i++) {
            gThreadOperations[i] = (int*) malloc(sizeof(int) * BEAGLE_PARTITION_OP_COUNT * kBufferCount * partitionCount);
        }

        gThreadOpCounts = (int*) malloc(sizeof(int) * kNumThreads);

        kThreadingEnabled = true;
    }


    memcpy(gPatternPartitions, inPatternPartitions, sizeof(int) * kPatternCount);

    bool reorderPatterns = false;
    int contiguousPartitions = 0;
    for (int i=0; i<kPatternCount; i++) {
        if (i > 0 && (gPatternPartitions[i] != gPatternPartitions[i-1])) {
            contiguousPartitions++;
        }
        if (contiguousPartitions != gPatternPartitions[i]) {
            reorderPatterns = true;
            break;
        }
    }

    if (reorderPatterns) {
        returnCode = reorderPatternsByPartition();
    } else {
        int currentPartition = gPatternPartitions[0];
        gPatternPartitionsStartPatterns[currentPartition] = 0;
        for (int i=0; i<kPatternCount; i++) {
            if (gPatternPartitions[i] != currentPartition) {
                currentPartition = gPatternPartitions[i];
                gPatternPartitionsStartPatterns[currentPartition] = i;
            }
        }
        gPatternPartitionsStartPatterns[currentPartition+1] = kPatternCount;
    }

    kPartitionsInitialised = true;

    return returnCode;
}

BEAGLE_CPU_TEMPLATE
    int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::setStateFrequencies(int stateFrequenciesIndex,
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

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::setCategoryWeights(int categoryWeightsIndex,
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

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::getTransitionMatrix(int matrixIndex,
                                                 double* outMatrix) {
    BEAGLE_CHECK_INDICES(badMatrix(matrixIndex));
    // TODO Test with multiple rate categories
if (T_PAD != 0) {
    double* offsetOutMatrix = outMatrix;
    REALTYPE* offsetBeagleMatrix = gTransitionMatrices[matrixIndex];
    for(int i = 0; i < kCategoryCount; i++) {
        for(int j = 0; j < kStateCount; j++) {
            beagleMemCpy(offsetOutMatrix,offsetBeagleMatrix,kStateCount);
            offsetBeagleMatrix += kTransPaddedStateCount; // Skip padding
            offsetOutMatrix += kStateCount;
        }
    }
} else {
    beagleMemCpy(outMatrix,gTransitionMatrices[matrixIndex],
            kMatrixSize * kCategoryCount);
}
    return BEAGLE_SUCCESS;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::getLogLikelihood(double* outSumLogLikelihood) {

    int returnCode = BEAGLE_SUCCESS;

    *outSumLogLikelihood = 0.0;
    for(int k=0; k < kPatternCount; k++) {
        *outSumLogLikelihood += outLogLikelihoodsTmp[k] * gPatternWeights[k];
    }

    if (*outSumLogLikelihood != *outSumLogLikelihood)
        returnCode = BEAGLE_ERROR_FLOATING_POINT;

    return returnCode;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::getDerivatives(double* outSumFirstDerivative,
                                                      double* outSumSecondDerivative) {

    *outSumFirstDerivative = 0.0;
    for (int i = 0; i < kPatternCount; i++) {
        *outSumFirstDerivative += outFirstDerivativesTmp[i] * gPatternWeights[i];
    }

    if (outSumSecondDerivative != NULL) {
        *outSumSecondDerivative = 0.0;
        for (int i = 0; i < kPatternCount; i++) {
            *outSumSecondDerivative += outSecondDerivativesTmp[i] * gPatternWeights[i];
        }
    }

    return BEAGLE_SUCCESS;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::getSiteLogLikelihoods(double* outLogLikelihoods) {
    if (kPatternsReordered) {
        REALTYPE* outLogLikelihoodsOriginalOrder = (REALTYPE*) malloc(sizeof(REALTYPE) * kPatternCount);
        for (int i=0; i < kPatternCount; i++) {
            outLogLikelihoodsOriginalOrder[i] = outLogLikelihoodsTmp[gPatternsNewOrder[i]];
        }
        beagleMemCpy(outLogLikelihoods, outLogLikelihoodsOriginalOrder, kPatternCount);
        free(outLogLikelihoodsOriginalOrder);
    } else {
        beagleMemCpy(outLogLikelihoods, outLogLikelihoodsTmp, kPatternCount);
    }

    return BEAGLE_SUCCESS;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::getSiteDerivatives(double* outFirstDerivatives,
                                                double* outSecondDerivatives) {
    beagleMemCpy(outFirstDerivatives, outFirstDerivativesTmp, kPatternCount);
    if (outSecondDerivatives != NULL)
        beagleMemCpy(outSecondDerivatives, outSecondDerivativesTmp, kPatternCount);

    return BEAGLE_SUCCESS;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::setTransitionMatrix(int matrixIndex,
                                       const double* inMatrix,
                                       double paddedValue) {

    if (matrixIndex < 0 || matrixIndex >= kMatrixCount)
        return BEAGLE_ERROR_OUT_OF_RANGE;
    REALTYPE* matrix = writableMatrix(matrixIndex);
    if (matrix == NULL)
        return BEAGLE_ERROR_OUT_OF_MEMORY;

if (T_PAD != 0) {
    const double* offsetInMatrix = inMatrix;
    REALTYPE* offsetBeagleMatrix = matrix;
    for(int i = 0; i < kCategoryCount; i++) {
        for(int j = 0; j < kStateCount; j++) {
            beagleMemCpy(offsetBeagleMatrix, offsetInMatrix, kStateCount);
            offsetBeagleMatrix[kStateCount] = paddedValue;
            offsetBeagleMatrix += kTransPaddedStateCount; // Skip padding
            offsetInMatrix += kStateCount;
        }
    }
} else {
    beagleMemCpy(matrix, inMatrix,
                 kMatrixSize * kCategoryCount);
}
    return BEAGLE_SUCCESS;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::setDifferentialMatrix(int matrixIndex,
                                       const double* inMatrix) {

    return setTransitionMatrix(matrixIndex, inMatrix, 0.0);
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::setTransitionMatrices(const int* matrixIndices,
                                                             const double* inMatrices,
                                                             const double* paddedValues,
                                                             int count) {
    for (int k = 0; k < count; k++) {
        if (matrixIndices[k] < 0 || matrixIndices[k] >= kMatrixCount)
            return BEAGLE_ERROR_OUT_OF_RANGE;
    }

    for (int k = 0; k < count; k++) {
        const double* inMatrix = inMatrices + k*kStateCount*kStateCount*kCategoryCount;
        int matrixIndex = matrixIndices[k];
        REALTYPE* matrix = writableMatrix(matrixIndex);
        if (matrix == NULL)
            return BEAGLE_ERROR_OUT_OF_MEMORY;

if (T_PAD != 0) {
        const double* offsetInMatrix = inMatrix;
        REALTYPE* offsetBeagleMatrix = matrix;
        for(int i = 0; i < kCategoryCount; i++) {
            for(int j = 0; j < kStateCount; j++) {
                beagleMemCpy(offsetBeagleMatrix, offsetInMatrix, kStateCount);
                offsetBeagleMatrix[kStateCount] = paddedValues[k];
                offsetBeagleMatrix += kTransPaddedStateCount; // Skip padding
                offsetInMatrix += kStateCount;
            }
        }
} else {
        beagleMemCpy(matrix, inMatrix,
                     kMatrixSize * kCategoryCount);
}
    }

    return BEAGLE_SUCCESS;
}

//TODO: move to EigenDecompositionSquare

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::convolveTransitionMatrices(const int* firstIndices,
        const int* secondIndices,
        const int* resultIndices,
        int matrixCount) {

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t Entering BeagleCPUImpl::convolveTransitionMatrices \n");
#endif

    int returnCode = BEAGLE_SUCCESS;

    BEAGLE_CHECK_INDICES(badMatrices(firstIndices, matrixCount) || badMatrices(secondIndices, matrixCount) ||
                         badMatrices(resultIndices, matrixCount));

    for (int u = 0; u < matrixCount; u++) {

        if(firstIndices[u] == resultIndices[u] || secondIndices[u] == resultIndices[u]) {

#ifdef BEAGLE_DEBUG_FLOW
            fprintf(stderr, "In-place convolution is not allowed \n");
#endif

            returnCode = BEAGLE_ERROR_OUT_OF_RANGE;
            break;

        }//END: overwrite check

        if (missingDenseMatrix(&firstIndices[u], 1) || missingDenseMatrix(&secondIndices[u], 1)) {
            returnCode = BEAGLE_ERROR_OUT_OF_RANGE;
            break;
        }

        REALTYPE* C = writableMatrix(resultIndices[u]);
        if (C == NULL) {
            returnCode = BEAGLE_ERROR_OUT_OF_MEMORY;
            break;
        }
        REALTYPE* A = gTransitionMatrices[firstIndices[u]];
        REALTYPE* B = gTransitionMatrices[secondIndices[u]];

        int n = 0;
        for (int l = 0; l < kCategoryCount; l++) {

            for (int i = 0; i < kStateCount; i++) {
                for (int j = 0; j < kStateCount; j++) {

                    REALTYPE sum = 0.0;
                    for (int k = 0; k < kStateCount; k++) {
                        sum += A[k + kTransPaddedStateCount * i] * B[j + kTransPaddedStateCount * k];
                    }
//                  printf("%.30f %.30f %d: \n", C[n], sum, l);
                    C[n] = sum;
                    n++;

                }//END: j loop

                if (T_PAD != 0) {

                    //                      A[n] = 1.0;
                    //                      B[n] = 1.0;
                    C[n] = 1.0;

                    n += T_PAD;

                }//END: padding check

            }//END: i loop

            A += kStateCount * kTransPaddedStateCount;
            B += kStateCount * kTransPaddedStateCount;

        }//END: rates loop

    }//END: u loop

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t Leaving BeagleCPUImpl::convolveTransitionMatrices \n");
#endif

    return returnCode;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::addTransitionMatrices(const int* firstIndices,
                                                             const int* secondIndices,
                                                             const int* resultIndices,
                                                             int matrixCount) {
    return BEAGLE_ERROR_NO_IMPLEMENTATION;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::transposeTransitionMatrices(
        const int* inputIndices,
        const int* resultIndices,
        int matrixCount) {

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t Entering BeagleCPUImpl::transposeTransitionMatrices \n");
#endif

    int returnCode = BEAGLE_SUCCESS;

    BEAGLE_CHECK_INDICES(badMatrices(inputIndices, matrixCount) || badMatrices(resultIndices, matrixCount));

    for (int u = 0; u < matrixCount; u++) {

        if (inputIndices[u] == resultIndices[u]) {

#ifdef BEAGLE_DEBUG_FLOW
            fprintf(stderr, "In-place transpose is not allowed.\n");
#endif

            returnCode = BEAGLE_ERROR_OUT_OF_RANGE;
            break;

        }

        if (missingDenseMatrix(&inputIndices[u], 1)) {
            returnCode = BEAGLE_ERROR_OUT_OF_RANGE;
            break;
        }

        REALTYPE* A = gTransitionMatrices[inputIndices[u]];
        REALTYPE* C = writableMatrix(resultIndices[u]);
        if (C == NULL) {
            returnCode = BEAGLE_ERROR_OUT_OF_MEMORY;
            break;
        }

        for (int l = 0; l < kCategoryCount; l++) {

            for (int i = 0; i < kStateCount; i++) {
                for (int j = 0; j < kStateCount; j++) {

                    C[kTransPaddedStateCount * j + i] = A[kTransPaddedStateCount * i + j];

                }
            }

            A += kStateCount * kTransPaddedStateCount;
            C += kStateCount * kTransPaddedStateCount;
        }
    }

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t Leaving BeagleCPUImpl::transposeTransitionMatrices \n");
#endif

    return returnCode;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::updateTransitionMatrices(
        int eigenIndex,
        const int* probabilityIndices,
        const int* firstDerivativeIndices,
        const int* secondDerivativeIndices,
        const double* edgeLengths,
        int count)
    {
    BEAGLE_CHECK_INDICES(badDerivativeMatrices(firstDerivativeIndices, secondDerivativeIndices, count));

    int returnCode = updateBranchEigenInfo(eigenIndex, probabilityIndices, edgeLengths, count);
    if (returnCode != BEAGLE_SUCCESS)
        return returnCode;

    for (int u = 0; u < count; u++) {
        if (!gEdgeLengthsGrad.empty()) {
            gEdgeLengthsGrad[probabilityIndices[u]] = edgeLengths[u];
        }
    }

    bool threading = (kBastaNumThreads > 1);
    if (threading) {
        gEigenDecomposition->updateTransitionMatricesParallel(
            eigenIndex,
            probabilityIndices,
            firstDerivativeIndices,
            secondDerivativeIndices,
            edgeLengths,
            gCategoryRates[0],
            gTransitionMatrices,
            count,
            kBastaNumThreads);
    } else {
        gEigenDecomposition->updateTransitionMatrices(
            eigenIndex,
            probabilityIndices,
            firstDerivativeIndices,
            secondDerivativeIndices,
            edgeLengths,
            gCategoryRates[0],
            gTransitionMatrices,
            count);
    }
    return BEAGLE_SUCCESS;
    }

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::updateTransitionMatricesGrad(const int* probabilityIndices,
                                                                    const double* edgeLengths,
                                                                    int count) {
    if (!kAdjointBuffersAllocated && gPartialsGrad == NULL) // allocateBastaGradBuffers has not run, as on the GPU
        return BEAGLE_ERROR_GENERAL;

    std::fill(gEdgeLengthsGrad.begin(), gEdgeLengthsGrad.end(), 0.0);
    for (int u = 0; u < count; u++) {
        gEdgeLengthsGrad[probabilityIndices[u]] = edgeLengths[u];
    }
    return BEAGLE_SUCCESS;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::updateBranchEigenInfo(int eigenIndex,
                                                             const int* probabilityIndices,
                                                             const double* edgeLengths,
                                                             int count) {
    BEAGLE_CHECK_INDICES(badMatrices(probabilityIndices, count));

    for (int i = 0; i < count; i++) {
        const int index = probabilityIndices[i];
        REALTYPE* chunk = gEigenInfoChunks[index >> kEigenInfoChunkShift];
        if (chunk == NULL && (chunk = allocateEigenInfoChunk(index >> kEigenInfoChunkShift)) == NULL)
            return BEAGLE_ERROR_OUT_OF_MEMORY;

        auto& info = gBranchEigenInfo[index];
        info.branchLength = edgeLengths[i];
        info.eigenIndex = eigenIndex;
        info.categoryRatesIndex = 0; // TODO: Implement category rates index

        const int slot = index & ((1 << kEigenInfoChunkShift) - 1);
        const size_t slotOffset = size_t(slot) * (kCategoryCount * kPartialsPaddedStateCount);

        REALTYPE* BEAGLE_RESTRICT times = chunk + kEigenInfoChunkOffset[0] + size_t(slot) * kCategoryCount;
        REALTYPE* BEAGLE_RESTRICT expat = chunk + kEigenInfoChunkOffset[1] + slotOffset;
        REALTYPE* BEAGLE_RESTRICT cosbt = chunk + kEigenInfoChunkOffset[2] + slotOffset;
        REALTYPE* BEAGLE_RESTRICT sinbt = chunk + kEigenInfoChunkOffset[3] + slotOffset;
        REALTYPE* BEAGLE_RESTRICT expatcosbt = chunk + kEigenInfoChunkOffset[4] + slotOffset;
        REALTYPE* BEAGLE_RESTRICT expatsinbt = chunk + kEigenInfoChunkOffset[5] + slotOffset;
        const REALTYPE* BEAGLE_RESTRICT eval = gEigenDecomposition->getEigenValuesPtr(eigenIndex);

        const bool isComplex = kFlags & BEAGLE_FLAG_EIGEN_COMPLEX;

        for (int j = 0; j < kCategoryCount; ++j) {
            const REALTYPE time = info.branchLength * gCategoryRates[info.categoryRatesIndex][j];
            times[j] = time;

            const int offset = j * kPartialsPaddedStateCount;

            for (int k = 0; k < kStateCount; ++k) {
                const REALTYPE e = std::exp(time * eval[k]);
                expat[offset + k] = e;

                const REALTYPE imag = isComplex ? eval[kStateCount + k] : REALTYPE(0);
                if (std::abs(imag) != REALTYPE(0)){
                    const REALTYPE c = std::cos(time * imag);
                    const REALTYPE s = std::sin(time * imag);
                    cosbt[offset + k]      = c;
                    sinbt[offset + k]      = s;
                    expatcosbt[offset + k] = e * c;
                    expatsinbt[offset + k] = e * s;
                    ++k;
                    expat[offset + k]      = e;
                    cosbt[offset + k]      = c;
                    sinbt[offset + k]      = -s;
                    expatcosbt[offset + k] = e * c;
                    expatsinbt[offset + k] = e * -s;
                } else {
                    cosbt[offset + k]      = REALTYPE(1);
                    sinbt[offset + k]      = REALTYPE(0);
                    expatcosbt[offset + k] = e;
                    expatsinbt[offset + k] = REALTYPE(0);
                }
            }
        }

        info.eval = eval;
        info.time = times;
        info.expat = expat;
        info.cosbt = cosbt;
        info.sinbt = sinbt;
        info.expatcosbt = expatcosbt;
        info.expatsinbt = expatsinbt;
    }

    return BEAGLE_SUCCESS;
}


BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::updateTransitionMatricesWithModelCategories(int* eigenIndices,
                                            const int* probabilityIndices,
                                            const int* firstDerivativeIndices,
                                            const int* secondDerivativeIndices,
                                            const double* edgeLengths,
                                            int count) {
    BEAGLE_CHECK_INDICES(badMatrices(probabilityIndices, count) ||
                         badDerivativeMatrices(firstDerivativeIndices, secondDerivativeIndices, count));

    gEigenDecomposition->updateTransitionMatricesWithModelCategories(eigenIndices,probabilityIndices,firstDerivativeIndices,secondDerivativeIndices,
                                                  edgeLengths,gTransitionMatrices,count);
    return BEAGLE_SUCCESS;
}


BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::updateTransitionMatricesWithMultipleModels(const int* eigenIndices,
                                                                                  const int* categoryRateIndices,
                                                                                  const int* probabilityIndices,
                                                                                  const int* firstDerivativeIndices,
                                                                                  const int* secondDerivativeIndices,
                                                                                  const double* edgeLengths,
                                                                                  int count) {
    BEAGLE_CHECK_INDICES(badMatrices(probabilityIndices, count) ||
                         badDerivativeMatrices(firstDerivativeIndices, secondDerivativeIndices, count));

    // TODO: move loop to within gEigenDecomposition

    for (int i = 0; i < count; i++) {
        // printf("uTMWMM %d %d %f %d\n", eigenIndices[i], probabilityIndices[i], edgeLengths[i], categoryRateIndices[i]);

        const int* firstDeriv  = NULL;
        const int* secondDeriv = NULL;
        if (firstDerivativeIndices != NULL && secondDerivativeIndices == NULL) {
            firstDeriv = &firstDerivativeIndices[i];
        } else if (firstDerivativeIndices != NULL && secondDerivativeIndices != NULL) {
            firstDeriv  = &firstDerivativeIndices[i];
            secondDeriv = &secondDerivativeIndices[i];
        }

        gEigenDecomposition->updateTransitionMatrices(eigenIndices[i],
                                                      &probabilityIndices[i],
                                                      firstDeriv,
                                                      secondDeriv,
                                                      &edgeLengths[i],
                                                      gCategoryRates[categoryRateIndices[i]],
                                                      gTransitionMatrices,
                                                      1);
    }

    return BEAGLE_SUCCESS;
}


BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::updatePartials(const int* operations,
                                                      int count,
                                                      int cumulativeScaleIndex,
                                                      BeaglePartialsType partialsType) {

    int returnCode = BEAGLE_ERROR_GENERAL;

    if (partialsType & BEAGLE_PARTIALS_TOP) {
        fprintf(stderr, "Post-order top partials are not currently supported on the CPU implementation.\n");
        return returnCode;
    }

    BEAGLE_CHECK_INDICES(badOperations(operations, count, false, cumulativeScaleIndex, false, false));

    if (kAutoPartitioningEnabled) {
        autoPartitionPartialsOperations(operations,
                                        gAutoPartitionOperations,
                                        count,
                                        cumulativeScaleIndex);
        count *= kPartitionCount;
        returnCode = upPartialsByPartitionAsync((const int*) gAutoPartitionOperations,
                                                count);
    } else {
        bool byPartition = false;
        returnCode = upPartials(byPartition,
                                operations,
                                count,
                                cumulativeScaleIndex,
                                partialsType);
    }

    return returnCode;
}


BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::updatePrePartials(const int *operations,
                                                         int count,
                                                         int cumulativeScaleIndex,
                                                         BeaglePartialsType partialsType) {
    int returnCode = BEAGLE_ERROR_GENERAL;

    BEAGLE_CHECK_INDICES(badOperations(operations, count, false, cumulativeScaleIndex, true,
                                       partialsType & BEAGLE_PARTIALS_TOP));

    if (kAutoPartitioningEnabled) {
        autoPartitionPartialsOperations(operations,
                                        gAutoPartitionOperations,
                                        count,
                                        cumulativeScaleIndex);
        count *= kPartitionCount;
        returnCode = upPrePartialsByPartitionAsync((const int*) gAutoPartitionOperations,
                                                count,
                                                partialsType);
    } else {
        bool byPartition = false;
        returnCode = upPrePartials(byPartition,
                                   operations,
                                   count,
                                   cumulativeScaleIndex,
                                   partialsType);
    }

    return returnCode;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::allocateBastaBuffers(int bufferCount,
                                                            int bufferLength,
                                                            int partialsCount,
                                                            int initial,
                                                            int numThreads) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleCPUImpl::allocateBastaBuffers\n");
#endif

    kBastaBuffersAllocated = true;
    kCoalescentBufferLength = bufferLength;
    kCoalescentBufferCount = bufferCount;
    kBastaNumThreads = numThreads;

    gCoalescentBuffers.resize(kCoalescentBufferCount * kCoalescentBufferLength);
    gBastaBuffers.resize(kPartialsPaddedStateCount * kCoalescentBufferLength * 4);

    gEdgeLengthsGrad.resize(kMatrixCount, 0);

    if (partialsCount > kBufferCount) {
        // the legacy gradient's partials (allocateBastaGradBuffers) follow the buffer count; they grow first, so
        // that a failure below leaves them large enough
        if (gPartialsGrad != NULL && kPartialsGradCount < partialsCount) {
            REALTYPE* temp = (REALTYPE*) realloc(gPartialsGrad, sizeof(REALTYPE) * (size_t) partialsCount *
                                                 kStateCount * kStateCount * kPartialsPaddedStateCount);
            if (temp == NULL)
                return BEAGLE_ERROR_OUT_OF_MEMORY;
            gPartialsGrad = temp;
            kPartialsGradCount = partialsCount;
        }

        // the partials, tip states, scale buffers (they follow the partials under SCALING_AUTO and SCALING_ALWAYS),
        // operation lists and counts grow as in beagleEnsureBufferCounts; through growBufferCounts, since
        // ensureBufferCounts refuses an instance that has BASTA buffers
        int returnCode = growBufferCounts(partialsCount - kCompactBufferCount, 0, 0);
        if (returnCode != BEAGLE_SUCCESS)
            return returnCode;
    }

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleCPUImpl::allocateBastaBuffers\n");
#endif

    return BEAGLE_SUCCESS;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::getBastaBuffer(int bufferIndex,
                                                      double* out) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleCPUImpl::getBastaBuffer\n");
#endif

    if (bufferIndex >= 0 && bufferIndex < kCoalescentBufferCount) {
        beagleMemCpy(out, gCoalescentBuffers.data() + bufferIndex * kCoalescentBufferLength,
                     kCoalescentBufferLength);
    } else if (bufferIndex >= kCoalescentBufferCount && bufferIndex < kCoalescentBufferCount + 4) {
        bufferIndex -= kCoalescentBufferCount;
        beagleMemCpy(out, gBastaBuffers.data() + bufferIndex * kPartialsPaddedStateCount, kStateCount);
    } else if (bufferIndex == -1) {
        for (int i = 0; i < 4; ++i) {
            beagleMemCpy(out + i * kStateCount,
                         gBastaBuffers.data() + i * kPartialsPaddedStateCount,
                         kStateCount);
        }
    }

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleCPUImpl::getBastaBuffer\n");
#endif

    return BEAGLE_SUCCESS;
}

BEAGLE_CPU_TEMPLATE
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::updateInnerBastaPartials(const int* operations,
                                                                 const int begin,
                                                                 const int end,
                                                                 const REALTYPE* sizes,
                                                                 REALTYPE* coalescent) {
    for (int op = begin; op < end; ++op) {
        const int numOps = BEAGLE_BASTA_OP_COUNT;

        const int destinationPartialIndex = operations[op * numOps];
        const int child1PartialIndex = operations[op * numOps + 1];
        const int child1TransMatIndex = operations[op * numOps + 2];
        const int child2PartialIndex = operations[op * numOps + 3];
        const int child2TransMatIndex = operations[op * numOps + 4];
        const int accumulation1PartialIndex = operations[op * numOps + 5];
        const int accumulation2PartialIndex = operations[op * numOps + 6];
        const int intervalNumber = operations[op * numOps + 7];

        const int matrixIncr = kStateCount + T_PAD;
        const int stateCountModFour = (kStateCount / 4) * 4;

        REALTYPE *destPartial = gPartials[destinationPartialIndex];
        // child 1
        const REALTYPE *partial1 = gPartials[child1PartialIndex];
        const REALTYPE *matrix1 = gTransitionMatrices[child1TransMatIndex];

        for (int i = 0; i < kStateCount; i++) {
            const REALTYPE *matrices1Ptr = matrix1 + i * matrixIncr;
            REALTYPE sum1A = 0.0;
            REALTYPE sum1B = 0.0;

            int j = 0;
            for (; j < stateCountModFour; j += 4) {
                sum1A += matrices1Ptr[j + 0] * partial1[j + 0];
                sum1B += matrices1Ptr[j + 1] * partial1[j + 1];
                sum1A += matrices1Ptr[j + 2] * partial1[j + 2];
                sum1B += matrices1Ptr[j + 3] * partial1[j + 3];

            }

            for (; j < kStateCount; j++) {
                sum1A += matrices1Ptr[j] * partial1[j];
            }

            destPartial[i] = sum1A + sum1B;
        }

        if (child2PartialIndex >= 0) {

            // child 2
            const REALTYPE *partial2 = gPartials[child2PartialIndex];
            const REALTYPE *matrix2 = gTransitionMatrices[child2TransMatIndex];

            REALTYPE *accumulation1 = gPartials[accumulation1PartialIndex];
            REALTYPE *accumulation2 = gPartials[accumulation2PartialIndex];

            REALTYPE prob = REALTYPE(0);

            for (int i = 0; i < kStateCount; i++) {
                const REALTYPE *matrices2Ptr = matrix2 + i * matrixIncr;
                REALTYPE sum2A = 0.0;
                REALTYPE sum2B = 0.0;

                int j = 0;
                for (; j < stateCountModFour; j += 4) {
                    sum2A += matrices2Ptr[j + 0] * partial2[j + 0];
                    sum2B += matrices2Ptr[j + 1] * partial2[j + 1];
                    sum2A += matrices2Ptr[j + 2] * partial2[j + 2];
                    sum2B += matrices2Ptr[j + 3] * partial2[j + 3];
                }

                for (; j < kStateCount; j++) {
                    sum2A += matrices2Ptr[j] * partial2[j];
                }

                REALTYPE child1 = destPartial[i];
                REALTYPE child2 = sum2A + sum2B;
                REALTYPE parent = child1 * child2 / sizes[i];

                accumulation1[i] = child1;
                accumulation2[i] = child2;
                destPartial[i] = parent;

                prob += parent;
            }

            for (int i = 0; i < kStateCount; i++) {
                destPartial[i] /= prob;
            }

            coalescent[intervalNumber] = prob;
        }
    }
}

BEAGLE_CPU_TEMPLATE
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::updateInnerBastaPartials2(const int* operations,
                                                                 const int op,
                                                                 const REALTYPE* sizes,
                                                                 REALTYPE* coalescent) {
    const int numOps = BEAGLE_BASTA_OP_COUNT;

    const int destinationPartialIndex = operations[op * numOps];
    const int child1PartialIndex = operations[op * numOps + 1];
    const int child1TransMatIndex = operations[op * numOps + 2];
    const int child2PartialIndex = operations[op * numOps + 3];
    const int child2TransMatIndex = operations[op * numOps + 4];
    const int accumulation1PartialIndex = operations[op * numOps + 5];
    const int accumulation2PartialIndex = operations[op * numOps + 6];
    const int intervalNumber = operations[op * numOps + 7];

    const int matrixIncr = kStateCount + T_PAD;
    const int stateCountModFour = (kStateCount / 4) * 4;

    REALTYPE *destPartial = gPartials[destinationPartialIndex];

    // child 1
    const REALTYPE *partial1 = gPartials[child1PartialIndex];
    const REALTYPE *matrix1 = gTransitionMatrices[child1TransMatIndex];

    for (int i = 0; i < kStateCount; i++) {
        const REALTYPE *matrices1Ptr = matrix1 + i * matrixIncr;
        REALTYPE sum1A = 0.0;
        REALTYPE sum1B = 0.0;

        int j = 0;
        for (; j < stateCountModFour; j += 4) {
            sum1A += matrices1Ptr[j + 0] * partial1[j + 0];
            sum1B += matrices1Ptr[j + 1] * partial1[j + 1];
            sum1A += matrices1Ptr[j + 2] * partial1[j + 2];
            sum1B += matrices1Ptr[j + 3] * partial1[j + 3];
        }

        for (; j < kStateCount; j++) {
            sum1A += matrices1Ptr[j] * partial1[j];
        }

        destPartial[i] = sum1A + sum1B;
    }

    if (child2PartialIndex >= 0) {

        // child 2
        const REALTYPE *partial2 = gPartials[child2PartialIndex];
        const REALTYPE *matrix2 = gTransitionMatrices[child2TransMatIndex];

        REALTYPE *accumulation1 = gPartials[accumulation1PartialIndex];
        REALTYPE *accumulation2 = gPartials[accumulation2PartialIndex];

        REALTYPE prob = REALTYPE(0);

        for (int i = 0; i < kStateCount; i++) {
            const REALTYPE *matrices2Ptr = matrix2 + i * matrixIncr;
            REALTYPE sum2A = 0.0;
            REALTYPE sum2B = 0.0;

            int j = 0;
            for (; j < stateCountModFour; j += 4) {
                sum2A += matrices2Ptr[j + 0] * partial2[j + 0];
                sum2B += matrices2Ptr[j + 1] * partial2[j + 1];
                sum2A += matrices2Ptr[j + 2] * partial2[j + 2];
                sum2B += matrices2Ptr[j + 3] * partial2[j + 3];
            }

            for (; j < kStateCount; j++) {
                sum2A += matrices2Ptr[j] * partial2[j];
            }

            REALTYPE child1 = destPartial[i];
            REALTYPE child2 = sum2A + sum2B;
            REALTYPE parent = child1 * child2 / sizes[i];

            accumulation1[i] = child1;
            accumulation2[i] = child2;
            destPartial[i] = parent;

            prob += parent;
        }

        for (int i = 0; i < kStateCount; i++) {
            destPartial[i] /= prob;
        }

        coalescent[intervalNumber] = prob;
    }
}

BEAGLE_CPU_TEMPLATE
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::updateInnerBastaPartialsGrad(const int* operations,
                                                                    const int begin,
                                                                    const int end,
                                                                    const REALTYPE* sizes,
                                                                    const REALTYPE* coalescent) {
    // gPartialsGrad[(node * SÂ² + a*S + b) * S_pad + i]
    // coalescentGrad[(a*S + b) * L + interval]
    const int S = kStateCount;
    const int S2 = S * S;
    const int S_pad = kPartialsPaddedStateCount;
    const int L = kCoalescentGradLength;

    for (int op = begin; op < end; ++op) {

        const int numOps = BEAGLE_BASTA_OP_COUNT;

        const int destinationPartialIndex = operations[op * numOps];
        const int child1PartialIndex = operations[op * numOps + 1];
        const int child1TransMatIndex = operations[op * numOps + 2];
        const int child2PartialIndex = operations[op * numOps + 3];
        const int child2TransMatIndex = operations[op * numOps + 4];
        const int accumulation1PartialIndex = operations[op * numOps + 5];
        const int accumulation2PartialIndex = operations[op * numOps + 6];
        const int intervalNumber = operations[op * numOps + 7];

        const int matrixIncr = S + T_PAD;

        REALTYPE *destPartial = gPartials[destinationPartialIndex];

        const REALTYPE distance1 = gEdgeLengthsGrad[child1TransMatIndex];

        const REALTYPE *matrix1 = gTransitionMatrices[child1TransMatIndex];
        const REALTYPE *leftAcc = (child2PartialIndex >= 0)? gPartials[accumulation1PartialIndex]: gPartials[destinationPartialIndex];

        const int childGradBase  = child1PartialIndex * S2 * S_pad;
        const int destGradBase   = destinationPartialIndex * S2 * S_pad;

        for (int a = 0; a < S; a++) {
            for (int b = 0; b < S; b++) {
                const int ab = a * S + b;
                const REALTYPE *childGrad = gPartialsGrad + childGradBase + ab * S_pad;
                REALTYPE *destGrad = gPartialsGrad + destGradBase + ab * S_pad;
                for (int i = 0; i < S; i++) {
                    REALTYPE sum = 0.0;
                    const REALTYPE *matRow = matrix1 + i * matrixIncr;
                    for (int j = 0; j < S; j++) {
                        sum += matRow[j] * childGrad[j];
                    }
                    destGrad[i] = sum;
                }
                destGrad[b] += leftAcc[a] * distance1;
            }
        }

        if (child2PartialIndex >= 0) {
            const REALTYPE *matrix2 = gTransitionMatrices[child2TransMatIndex];
            const REALTYPE distance2 = gEdgeLengthsGrad[child2TransMatIndex];

            REALTYPE *accumulation1 = gPartials[accumulation1PartialIndex];
            REALTYPE *accumulation2 = gPartials[accumulation2PartialIndex];

            const int child2GradBase = child2PartialIndex * S2 * S_pad;
            const int acc1GradBase   = accumulation1PartialIndex * S2 * S_pad;
            const int acc2GradBase   = accumulation2PartialIndex * S2 * S_pad;

            for (int a = 0; a < S; a++) {
                for (int b = 0; b < S; b++) {
                    const int ab = a * S + b;
                    REALTYPE J = coalescent[intervalNumber];

                    const REALTYPE *childGrad2 = gPartialsGrad + child2GradBase + ab * S_pad;

                    REALTYPE *destGrad  = gPartialsGrad + destGradBase + ab * S_pad;
                    REALTYPE *accGrad1  = gPartialsGrad + acc1GradBase + ab * S_pad;
                    REALTYPE *accGrad2  = gPartialsGrad + acc2GradBase + ab * S_pad;

                    REALTYPE partial_J_ab = REALTYPE(0);
                    for (int i = 0; i < S; i++) {
                        REALTYPE rightGradI = 0.0;
                        const REALTYPE *matRow2 = matrix2 + i * matrixIncr;
                        for (int j = 0; j < S; j++) {
                            rightGradI += matRow2[j] * childGrad2[j];
                        }
                        accGrad2[i] = rightGradI;
                    }
                    accGrad2[b] += accumulation2[a] * distance2;

                    for (int i = 0; i < S; i++) {
                        REALTYPE rightGrad = accGrad2[i];
                        REALTYPE leftGrad = destGrad[i];
                        REALTYPE left = accumulation1[i];
                        REALTYPE right = accumulation2[i];

                        REALTYPE entry = (leftGrad * right + rightGrad * left) / sizes[i];
                        partial_J_ab += entry;

                        destGrad[i] = entry / J;
                        accGrad1[i] = leftGrad;
                    }

                    for (int i = 0; i < S; i++) {
                        REALTYPE entry = destPartial[i];
                        destGrad[i] -= partial_J_ab * entry / J;
                    }
                    coalescentGrad[ab * L + intervalNumber] = partial_J_ab;
                }
            }
        }
    }
}



BEAGLE_CPU_TEMPLATE
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::matrixMultiply(const REALTYPE* matrix1,
                                                    const REALTYPE* matrix2,
                                                    REALTYPE* output,
                                                    int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            REALTYPE sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += matrix1[i*n + k] * matrix2[k*n + j];
            }
            output[i*n + j] = sum;
        }
    }
} 

template <typename Integer, typename Function>
inline void for_each(Integer begin, Integer end, Function function, int global_num_threads) {
    std::vector<std::thread> threads;
    threads.reserve(global_num_threads);

    // Calculate the chunk size for each thread
    size_t chunk_size = (end - begin) / global_num_threads;
    size_t remainder = (end - begin) % global_num_threads;

    // Perform the operation in parallel using threads
    for (size_t i = 0; i < global_num_threads; ++i) {
        Integer thread_begin = begin + i * chunk_size;
        Integer thread_end = thread_begin + chunk_size;
        if (i == global_num_threads - 1) {
            // Include the remainder in the last thread
            thread_end += remainder;
        }

        threads.emplace_back([thread_begin, thread_end, &function] {
//                threads.emplace_back([&] {
                    function(thread_begin, thread_end);
//            for (int j = thread_begin; j < thread_end; ++j) {
//                function(j);
//            }
        });
    }

    // Wait for all threads to finish
    for (auto &thread : threads) {
        thread.join();
    }
}


    BEAGLE_CPU_TEMPLATE
    int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::updateBastaPartials(const int* operations,
                                                                 const int count,
                                                                 const int* intervals,
                                                                 const int intervalCount,
                                                               const int populationSizesIndex,
                                                               const int coalescentIndex) {
        int returnCode = BEAGLE_SUCCESS;

        kUnclampedForwardDone = false;

        REALTYPE* coalescent = gCoalescentBuffers.data() + kCoalescentBufferLength * coalescentIndex;
        std::fill(coalescent, coalescent + kCoalescentBufferLength, REALTYPE(0));

        const REALTYPE* sizes = gStateFrequencies[populationSizesIndex];

        bool THREADING = (kBastaNumThreads > 1);
        int CUT_POINT = 16;
        int numThreads = kBastaNumThreads;
        if (THREADING) {

            for (int i = 0; i < intervalCount - 1; ++i) {

                const int begin = intervals[i];
                const int end = intervals[i + 1];

                if (end - begin > CUT_POINT) {
#pragma omp parallel for num_threads(numThreads)
                    for (int op = begin; op < end; ++op) {
                        updateInnerBastaPartials2(operations, op, sizes, coalescent);
                    }

                } else {
                    updateInnerBastaPartials(operations, begin, end, sizes, coalescent);
                }
            }
        } else {
            updateInnerBastaPartials(operations, 0, count, sizes, coalescent);
        }

        return returnCode;
    }

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::allocateBastaGradBuffers(int partialsCount) {

    if (partialsCount < 0) {
        kUseAdjointGradient = true;
        partialsCount = -partialsCount;
    }

    const int S  = kStateCount;
    const int S2 = S * S;
    const int S_pad = kPartialsPaddedStateCount;
    const int bufferLength = kCoalescentBufferLength;

    kBastaBuffersAllocated = true;
    gEdgeLengthsGrad.resize(kMatrixCount, 0);

    if (kUseAdjointGradient) {
        if (!kAdjointBuffersAllocated) {
            gCoalescentDetails.resize(bufferLength);

            gAdjointWorkE.resize(S_pad);
            gAdjointWorkF.resize(S_pad);
            gAdjointWorkG.resize(S_pad);
            gAdjointWorkH.resize(S_pad);
            gAdjointWorkW.resize(S_pad);
            gAdjointWorkIexp.resize(S2);
            gAdjointWorkTemp.resize(S2);

            kAdjointBuffersAllocated = true;
        }
    } else {
        gBastaGradBuffers.resize(4 * (size_t)S2 * S_pad * bufferLength);

        const size_t partialsGradSize = (size_t)kBufferCount * S2 * S_pad;
        const size_t coalescentGradSize = (size_t)S2 * bufferLength;

        if (gPartialsGrad == NULL) {
            gPartialsGrad = (REALTYPE*) mallocAligned(sizeof(REALTYPE) * partialsGradSize);
            if (gPartialsGrad == NULL) throw std::bad_alloc();
            kPartialsGradCount = kBufferCount;
        } else if (partialsCount > kPartialsGradCount) {
            const size_t newSize = (size_t)partialsCount * S2 * S_pad;
            REALTYPE* temp = (REALTYPE*) realloc(gPartialsGrad, sizeof(REALTYPE) * newSize);
            if (temp == NULL) throw std::bad_alloc();
            gPartialsGrad = temp;
            kPartialsGradCount = partialsCount;
        }

        if (coalescentGrad == NULL) {
            coalescentGrad = (REALTYPE*) mallocAligned(sizeof(REALTYPE) * coalescentGradSize);
            if (coalescentGrad == NULL) throw std::bad_alloc();
            kCoalescentGradLength = bufferLength;
        } else if (bufferLength > kCoalescentGradLength) {
            REALTYPE* temp = (REALTYPE*) realloc(coalescentGrad, sizeof(REALTYPE) * coalescentGradSize);
            if (temp == NULL) throw std::bad_alloc();
            coalescentGrad = temp;
            kCoalescentGradLength = bufferLength;
        }
    }

    return BEAGLE_SUCCESS;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::updateBastaPartialsGrad(const int* operations,
  														   const int count,
  														   const int* intervals,
  														   const int intervalCount,
                                                           const int populationSizesIndex,
                                                           const int coalescentIndex) {
	int returnCode = BEAGLE_SUCCESS;
    if (!kAdjointBuffersAllocated && gPartialsGrad == NULL) // allocateBastaGradBuffers has not run, as on the GPU
        return BEAGLE_ERROR_GENERAL;

    const REALTYPE* coalescent = gCoalescentBuffers.data() + kCoalescentBufferLength * coalescentIndex;
    const REALTYPE* sizes = gStateFrequencies[populationSizesIndex];

    if (kAdjointBuffersAllocated) {
        kAdjointReversePassDone = false;
    } else {
        const size_t partialsGradSize = (size_t)kBufferCount * kStateCount * kStateCount * kPartialsPaddedStateCount;
        std::fill(gPartialsGrad, gPartialsGrad + partialsGradSize, REALTYPE(0));

        const size_t coalescentGradSize = (size_t)kStateCount * kStateCount * kCoalescentGradLength;
        std::fill(coalescentGrad, coalescentGrad + coalescentGradSize, REALTYPE(0));

        for (int i = 0; i < intervalCount - 1; ++i) {
            const int begin = intervals[i];
            const int end = intervals[i + 1];
            updateInnerBastaPartialsGrad(operations, begin, end, sizes, coalescent);
        }
    }

	return returnCode;
}

BEAGLE_CPU_TEMPLATE
REALTYPE BeagleCPUImpl<BEAGLE_CPU_GENERIC>::reduceAcrossIntervals(
        REALTYPE* e, REALTYPE* f,
        REALTYPE* g, REALTYPE* h,
        int interval, REALTYPE length,
        const REALTYPE* sizes,
        const REALTYPE* coalescent) {

    e += interval * kPartialsPaddedStateCount;
    f += interval * kPartialsPaddedStateCount;
    g += interval * kPartialsPaddedStateCount;
    h += interval * kPartialsPaddedStateCount;
    REALTYPE sum = REALTYPE(0);
    for (int k = 0; k < kStateCount; ++k) {
        sum += (e[k] * e[k] - f[k] +
                g[k] * g[k] - h[k]) / sizes[k];
    }

    REALTYPE logL = -length * sum / 4;

    REALTYPE prob = coalescent[interval];
    if (prob != REALTYPE(0)) {
        logL += log(prob);
    }

    return logL;
}


BEAGLE_CPU_TEMPLATE
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::reduceAcrossIntervalsGrad(
        REALTYPE* e, REALTYPE* f,
        REALTYPE* g, REALTYPE* h,
        REALTYPE* eGrad, REALTYPE* fGrad,
        REALTYPE* gGrad, REALTYPE* hGrad,
        REALTYPE* resultGrad,
        int interval, REALTYPE length,
        const REALTYPE* sizes,
        const REALTYPE* coalescent) {

    const int S = kStateCount;
    const int S_pad = kPartialsPaddedStateCount;
    const int L = kCoalescentBufferLength;

    const int eOffset = interval * S_pad;
    e += eOffset;
    g += eOffset;

    for (int ab = 0; ab < S * S; ++ab) {
        REALTYPE sum = REALTYPE(0);
        const int efghOff = (ab * L + interval) * S_pad;
        REALTYPE* eGrad_ab = eGrad + efghOff;
        REALTYPE* fGrad_ab = fGrad + efghOff;
        REALTYPE* gGrad_ab = gGrad + efghOff;
        REALTYPE* hGrad_ab = hGrad + efghOff;

        for (int k = 0; k < S; ++k) {
            sum += (2 * e[k] * eGrad_ab[k] - fGrad_ab[k] +
                    2 * g[k] * gGrad_ab[k] - hGrad_ab[k]) / sizes[k];
        }
        resultGrad[ab] = -length * sum / 4;
        REALTYPE prob = coalescent[interval];
        if (prob != REALTYPE(0)) {
            resultGrad[ab] += coalescentGrad[ab * L + interval] / prob;
        }
    }
}

BEAGLE_CPU_TEMPLATE
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::reduceWithinInterval(REALTYPE* e, REALTYPE* f,
                                                             REALTYPE* g, REALTYPE* h,
                                                             int startBuffer1, int startBuffer2,
                                                             int endBuffer1, int endBuffer2,
                                                             int interval) {

    e += interval * kPartialsPaddedStateCount;
    f += interval * kPartialsPaddedStateCount;
    g += interval * kPartialsPaddedStateCount;
    h += interval * kPartialsPaddedStateCount;

    const REALTYPE* startPartials = gPartials[startBuffer1];
    const REALTYPE* endPartials = gPartials[endBuffer1];

    for (int i = 0; i < kStateCount; ++i) {
        REALTYPE startP = startPartials[i];
        e[i] += startP;
        f[i] += startP * startP;

        REALTYPE endP = endPartials[i];
        g[i] += endP;
        h[i] += endP * endP;
    }

    if (startBuffer2 >= 0) {

        const REALTYPE* startPartials = gPartials[startBuffer2];
        const REALTYPE* endPartials = gPartials[endBuffer2];

        for (int i = 0; i < kStateCount; ++i) {
            REALTYPE startP = startPartials[i];
            e[i] += startP;
            f[i] += startP * startP;

            REALTYPE endP = endPartials[i];
            g[i] += endP;
            h[i] += endP * endP;
        }
    }
}


BEAGLE_CPU_TEMPLATE
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::reduceWithinIntervalGrad(REALTYPE* eGrad, REALTYPE* fGrad,
                                                             REALTYPE* gGrad, REALTYPE* hGrad,
                                                             int startBuffer1, int startBuffer2,
                                                             int endBuffer1, int endBuffer2,
                                                             int interval) {

    const int S = kStateCount;
    const int S2 = S * S;
    const int S_pad = kPartialsPaddedStateCount;
    const int L = kCoalescentBufferLength;

    const REALTYPE* startPartials = gPartials[startBuffer1];
    const REALTYPE* endPartials = gPartials[endBuffer1];


    const int startGradBase1 = startBuffer1 * S2 * S_pad;
    const int endGradBase1   = endBuffer1 * S2 * S_pad;

    for (int ab = 0; ab < S2; ++ab) {
        const int efghOffset = (ab * L + interval) * S_pad;
        REALTYPE* eGrad_ab = eGrad + efghOffset;
        REALTYPE* fGrad_ab = fGrad + efghOffset;
        REALTYPE* gGrad_ab = gGrad + efghOffset;
        REALTYPE* hGrad_ab = hGrad + efghOffset;

        const REALTYPE* startPGradPtr = gPartialsGrad + startGradBase1 + ab * S_pad;
        const REALTYPE* endPGradPtr   = gPartialsGrad + endGradBase1 + ab * S_pad;

        for (int i = 0; i < S; ++i) {
            REALTYPE startPGrad = startPGradPtr[i];
            REALTYPE startP = startPartials[i];
            eGrad_ab[i] += startPGrad;
            fGrad_ab[i] += 2 * startP * startPGrad;

            REALTYPE endPGrad = endPGradPtr[i];
            REALTYPE endP = endPartials[i];
            gGrad_ab[i] += endPGrad;
            hGrad_ab[i] += 2 * endP * endPGrad;
        }
    }

    if (startBuffer2 >= 0) {
        const REALTYPE* startPartials2 = gPartials[startBuffer2];
        const REALTYPE* endPartials2 = gPartials[endBuffer2];

        const int startGradBase2 = startBuffer2 * S2 * S_pad;
        const int endGradBase2   = endBuffer2 * S2 * S_pad;

        for (int ab = 0; ab < S2; ++ab) {
            const int efghOffset = (ab * L + interval) * S_pad;
            REALTYPE* eGrad_ab = eGrad + efghOffset;
            REALTYPE* fGrad_ab = fGrad + efghOffset;
            REALTYPE* gGrad_ab = gGrad + efghOffset;
            REALTYPE* hGrad_ab = hGrad + efghOffset;

            const REALTYPE* startPGradPtr = gPartialsGrad + startGradBase2 + ab * S_pad;
            const REALTYPE* endPGradPtr   = gPartialsGrad + endGradBase2 + ab * S_pad;

            for (int i = 0; i < S; ++i) {
                REALTYPE startPGrad = startPGradPtr[i];
                REALTYPE startP = startPartials2[i];
                eGrad_ab[i] += startPGrad;
                fGrad_ab[i] += 2 * startP * startPGrad;

                REALTYPE endPGrad = endPGradPtr[i];
                REALTYPE endP = endPartials2[i];
                gGrad_ab[i] += endPGrad;
                hGrad_ab[i] += 2 * endP * endPGrad;
            }
        }
    }
}

BEAGLE_CPU_TEMPLATE
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::adjointPushReductionAdjoints(
        const REALTYPE* partials, REALTYPE* partialAdj,
        const REALTYPE* adjSum, const REALTYPE* adjSq) {
    if (partialAdj == NULL || partials == NULL) return;
    for (int i = 0; i < kStateCount; ++i) {
        partialAdj[i] += adjSum[i] + REALTYPE(2) * partials[i] * adjSq[i];
    }
}

BEAGLE_CPU_TEMPLATE
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::adjointAddMatTVecAndOuterInPlace(
        REALTYPE* transposeVecTarget, REALTYPE* outerTarget,
        const REALTYPE* matrix, const REALTYPE* left, const REALTYPE* right,
        int matrixIncr) {
    const int S = kStateCount;
    for (int i = 0; i < S; ++i) {
        REALTYPE leftVal = left[i];
        if (leftVal == REALTYPE(0)) continue;
        const REALTYPE* matRow = matrix + i * matrixIncr;
        REALTYPE* outerRow = outerTarget + i * matrixIncr;
        for (int j = 0; j < S; ++j) {
            transposeVecTarget[j] += matRow[j] * leftVal;
            outerRow[j] += leftVal * right[j];
        }
    }
}


BEAGLE_CPU_TEMPLATE
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::adjointAddMatTVecInPlace(REALTYPE* transposeVecTarget,
																const REALTYPE* matrix,
        														const REALTYPE* left,
																int matrixIncr) {
    if (transposeVecTarget == NULL) return;
    const int S = kStateCount;
    for (int i = 0; i < S; ++i) {
        REALTYPE leftVal = left[i];
        if (leftVal == REALTYPE(0)) continue;
        const REALTYPE* matRow = matrix + i * matrixIncr;
        for (int j = 0; j < S; ++j) {
            transposeVecTarget[j] += matRow[j] * leftVal;
        }
    }
}


BEAGLE_CPU_TEMPLATE
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::adjointAddOuterInPlace( REALTYPE* outerTarget,
														const REALTYPE* left,
														const REALTYPE* right,
        												int matrixIncr) {
    if (outerTarget == NULL) return;
    const int S = kStateCount;
    for (int i = 0; i < S; ++i) {
        REALTYPE leftVal = left[i];
        if (leftVal == REALTYPE(0)) continue;
        REALTYPE* outerRow = outerTarget + i * matrixIncr;
        for (int j = 0; j < S; ++j) {
            outerRow[j] += leftVal * right[j];
        }
    }
}

BEAGLE_CPU_TEMPLATE
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::adjointReversePass(
        const int* operations, int count,
        const int* intervals, int intervalCount,
        const double* intervalLengths,
        const REALTYPE* sizes, const REALTYPE* coalescent,
        int eigenIndex,
        int partialAdjointBufferBase,
        int matrixAdjointBufferBase,
        REALTYPE* popSizeGradOut) {

    const int S = kStateCount;
    const int S_pad = kPartialsPaddedStateCount;
    const int numOps = BEAGLE_BASTA_OP_COUNT;
    const int matrixIncr = S + T_PAD;

    const bool bastaThreading = (kBastaNumThreads > 1);
    const int bastaNumThreads = bastaThreading ? kBastaNumThreads : 1;
    const int CUT_POINT_REVERSE = 16;

    if (partialAdjointBufferBase < 0 || matrixAdjointBufferBase < 0) {
        return;
    }
    if (partialAdjointBufferBase >= kBufferCount || matrixAdjointBufferBase >= kMatrixCount) {
        return;
    }

    const int adjPartialCount = kBufferCount - partialAdjointBufferBase;
    const int adjMatrixCount  = kMatrixCount - matrixAdjointBufferBase;


#pragma omp parallel for num_threads(bastaNumThreads) if(bastaThreading)
    for (int b = 0; b < adjPartialCount; ++b) {
        REALTYPE* adj = gPartials[partialAdjointBufferBase + b];
        if (adj != NULL) {
            std::fill(adj, adj + S_pad, REALTYPE(0));
        }
    }


#pragma omp parallel for num_threads(bastaNumThreads) if(bastaThreading)
    for (int m = 0; m < adjMatrixCount; ++m) {
        REALTYPE* adj = gTransitionMatrices[matrixAdjointBufferBase + m];
        if (adj != NULL) {
            std::fill(adj, adj + (size_t)S * matrixIncr, REALTYPE(0));
        }
    }

    if (popSizeGradOut != NULL) {
        std::fill(popSizeGradOut, popSizeGradOut + S, REALTYPE(0));
    }


    if (!kUnclampedForwardDone) {
        std::vector<int> indices(kMatrixCount);
        std::vector<double> edgeLengths(kMatrixCount);
        int activeCount = 0;
        for (int m = 0; m < kMatrixCount; ++m) {
            if (gEdgeLengthsGrad[m] != REALTYPE(0)) {
                indices[activeCount] = m;
                edgeLengths[activeCount] = (double)gEdgeLengthsGrad[m];
                activeCount++;
            }
        }

        if (activeCount > 0 && eigenIndex >= 0) {
            gEigenDecomposition->updateTransitionMatrices(
                eigenIndex, indices.data(), NULL, NULL,
                edgeLengths.data(), gCategoryRates[0],
                gTransitionMatrices, activeCount);
        }

        REALTYPE* coal = const_cast<REALTYPE*>(coalescent);
        memset(coal, 0, kCoalescentBufferLength * sizeof(REALTYPE));
        if (bastaThreading) {
            for (int iv = 0; iv < intervalCount - 1; ++iv) {
                const int ivBegin = intervals[iv];
                const int ivEnd = intervals[iv + 1];
                if (ivEnd - ivBegin > CUT_POINT_REVERSE) {
#pragma omp parallel for num_threads(bastaNumThreads)
                    for (int op = ivBegin; op < ivEnd; ++op) {
                        updateInnerBastaPartials2(operations, op, sizes, coal);
                    }
                } else {
                    updateInnerBastaPartials(operations, ivBegin, ivEnd, sizes, coal);
                }
            }
        } else {
            updateInnerBastaPartials(operations, 0, count, sizes, coal);
        }

        std::fill(gBastaBuffers.begin(), gBastaBuffers.end(), REALTYPE(0));
        REALTYPE* eB = gBastaBuffers.data();
        REALTYPE* fB = eB + S_pad * kCoalescentBufferLength;
        REALTYPE* gB = fB + S_pad * kCoalescentBufferLength;
        REALTYPE* hB = gB + S_pad * kCoalescentBufferLength;
#pragma omp parallel for num_threads(bastaNumThreads) if(bastaThreading)
        for (int iv = 0; iv < intervalCount - 1; ++iv) {
            const int ivBegin = intervals[iv];
            const int ivEnd = intervals[iv + 1];
            for (int op = ivBegin; op < ivEnd; ++op) {
                reduceWithinInterval(eB, fB, gB, hB,
                    operations[op * numOps + 1], operations[op * numOps + 3],
                    operations[op * numOps + 5], operations[op * numOps + 6],
                    operations[op * numOps + 7]);
            }
        }
        kUnclampedForwardDone = true;
    }

    const REALTYPE* e = gBastaBuffers.data();
    const REALTYPE* f = e + S_pad * kCoalescentBufferLength;
    const REALTYPE* g = f + S_pad * kCoalescentBufferLength;
    const REALTYPE* h = g + S_pad * kCoalescentBufferLength;

    REALTYPE* adjE = gAdjointWorkE.data();
    REALTYPE* adjF = gAdjointWorkF.data();
    REALTYPE* adjG = gAdjointWorkG.data();
    REALTYPE* adjH = gAdjointWorkH.data();
    REALTYPE* adjW = gAdjointWorkW.data();

    auto partialAdjOf = [&](int bufferIndex) -> REALTYPE* {
        return (bufferIndex < 0) ? NULL : gPartials[partialAdjointBufferBase + bufferIndex];
    };
    auto matrixAdjOf = [&](int matIndex) -> REALTYPE* {
        return (matIndex < 0) ? NULL : gTransitionMatrices[matrixAdjointBufferBase + matIndex];
    };

    gAdjointYBar.assign((size_t)2 * count * S, REALTYPE(0));
    REALTYPE* ybarStore = gAdjointYBar.data();
    auto ybarSlot = [&](int opIdx, int slot) -> REALTYPE* {
        return ybarStore + ((size_t)2 * opIdx + slot) * S;
    };

    for (int interval = intervalCount - 2; interval >= 0; --interval) {
        const int start = intervals[interval];
        const int end = intervals[interval + 1];

        const int intervalNumber = operations[start * numOps + 7];
        const REALTYPE length = (REALTYPE)intervalLengths[interval];

        const REALTYPE* eI = e + intervalNumber * S_pad;
        const REALTYPE* fI = f + intervalNumber * S_pad;
        const REALTYPE* gI = g + intervalNumber * S_pad;
        const REALTYPE* hI = h + intervalNumber * S_pad;

        for (int i = 0; i < S; ++i) {
            REALTYPE invSize = REALTYPE(1) / sizes[i];
            adjE[i] = -length * eI[i] / (REALTYPE(2) * sizes[i]);
            adjF[i] = length * invSize / REALTYPE(4);
            adjG[i] = -length * gI[i] / (REALTYPE(2) * sizes[i]);
            adjH[i] = length * invSize / REALTYPE(4);
            if (popSizeGradOut != NULL) {
                popSizeGradOut[i] += length * (eI[i]*eI[i] - fI[i] + gI[i]*gI[i] - hI[i]) /
                                  (REALTYPE(4) * sizes[i] * sizes[i]);
            }
        }

#pragma omp parallel for num_threads(bastaNumThreads) if(bastaThreading && (end - start) > CUT_POINT_REVERSE)
        for (int idx = start; idx < end; ++idx) {
            const int accBuf1 = operations[idx * numOps + 5];
            const int accBuf2 = operations[idx * numOps + 6];
            const int inBuf1 = operations[idx * numOps + 1];
            const int inBuf2 = operations[idx * numOps + 3];

            if (accBuf1 >= 0)
                adjointPushReductionAdjoints(gPartials[accBuf1], partialAdjOf(accBuf1), adjG, adjH);
            if (accBuf2 >= 0)
                adjointPushReductionAdjoints(gPartials[accBuf2], partialAdjOf(accBuf2), adjG, adjH);
            if (inBuf1 >= 0)
                adjointPushReductionAdjoints(gPartials[inBuf1], partialAdjOf(inBuf1), adjE, adjF);
            if (inBuf2 >= 0)
                adjointPushReductionAdjoints(gPartials[inBuf2], partialAdjOf(inBuf2), adjE, adjF);
        }

        int coalOpIdx = -1;
        for (int idx = start; idx < end; ++idx) {
            if (operations[idx * numOps + 3] >= 0) {
                coalOpIdx = idx;
                break;
            }
        }

        if (coalOpIdx >= 0) {
            const int destBuf = operations[coalOpIdx * numOps];
            const int inBuf1 = operations[coalOpIdx * numOps + 1];
            const int inBuf2 = operations[coalOpIdx * numOps + 3];
            const int inMat1 = operations[coalOpIdx * numOps + 2];
            const int inMat2 = operations[coalOpIdx * numOps + 4];
            const int accBuf1 = operations[coalOpIdx * numOps + 5];
            const int accBuf2 = operations[coalOpIdx * numOps + 6];
            const int intNum = operations[coalOpIdx * numOps + 7];

            const REALTYPE* leftEnd = gPartials[accBuf1];
            const REALTYPE* rightEnd = gPartials[accBuf2];
            REALTYPE J = coalescent[intNum];

            REALTYPE* zAdj = partialAdjOf(destBuf);
            REALTYPE* accAdj1 = partialAdjOf(accBuf1);
            REALTYPE* accAdj2 = partialAdjOf(accBuf2);

            REALTYPE* ybL = ybarSlot(coalOpIdx, 0);
            REALTYPE* ybR = ybarSlot(coalOpIdx, 1);

            REALTYPE dotZW = REALTYPE(0);
            for (int i = 0; i < S; ++i) {
                REALTYPE w_i = leftEnd[i] * rightEnd[i] / sizes[i];
                dotZW += zAdj[i] * w_i;
            }
            REALTYPE adjJ = REALTYPE(1) / J - dotZW / (J * J);

            for (int i = 0; i < S; ++i) {
                adjW[i] = zAdj[i] / J + adjJ;
                ybL[i] = accAdj1[i] + adjW[i] * rightEnd[i] / sizes[i];
                ybR[i] = accAdj2[i] + adjW[i] * leftEnd[i] / sizes[i];
                if (popSizeGradOut != NULL) {
                    popSizeGradOut[i] -= adjW[i] * leftEnd[i] * rightEnd[i] / (sizes[i] * sizes[i]);
                }
            }

            adjointAddMatTVecInPlace(partialAdjOf(inBuf1),
                    gTransitionMatrices[inMat1], ybL, matrixIncr);
            adjointAddMatTVecInPlace(partialAdjOf(inBuf2),
                    gTransitionMatrices[inMat2], ybR, matrixIncr);
        }


#pragma omp parallel for num_threads(bastaNumThreads) if(bastaThreading && (end - start) > CUT_POINT_REVERSE)
        for (int idx = start; idx < end; ++idx) {
            if (operations[idx * numOps + 3] >= 0) continue;

            const int inBuf1 = operations[idx * numOps + 1];
            const int inMat1 = operations[idx * numOps + 2];
            const int accBuf1 = operations[idx * numOps + 5];

            const REALTYPE* yBar = partialAdjOf(accBuf1);
            REALTYPE* yb = ybarSlot(idx, 0);
            for (int i = 0; i < S; ++i) yb[i] = yBar[i];

            adjointAddMatTVecInPlace(partialAdjOf(inBuf1),
                    gTransitionMatrices[inMat1], yBar, matrixIncr);
        }
    }


#pragma omp parallel for num_threads(bastaNumThreads) if(bastaThreading)
    for (int interval = 0; interval <= intervalCount - 2; ++interval) {
        const int start = intervals[interval];
        const int end = intervals[interval + 1];

        for (int idx = start; idx < end; ++idx) {
            const int inBuf2 = operations[idx * numOps + 3];
            if (inBuf2 >= 0) {
                const int inBuf1 = operations[idx * numOps + 1];
                const int inMat1 = operations[idx * numOps + 2];
                const int inMat2 = operations[idx * numOps + 4];
                adjointAddOuterInPlace(matrixAdjOf(inMat1),
                        ybarSlot(idx, 0), gPartials[inBuf1], matrixIncr);
                adjointAddOuterInPlace(matrixAdjOf(inMat2),
                        ybarSlot(idx, 1), gPartials[inBuf2], matrixIncr);
            } else {
                const int inBuf1 = operations[idx * numOps + 1];
                const int inMat1 = operations[idx * numOps + 2];
                adjointAddOuterInPlace(matrixAdjOf(inMat1),
                        ybarSlot(idx, 0), gPartials[inBuf1], matrixIncr);
            }
        }
    }
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::accumulateEigenBasisGradient(
        int eigenIndex,
        int matrixAdjointBufferBase,
        const double* branchLengths,
        int matrixCount, int hasComplexEigenvalues, double* outRateGradient) {

    if (!kAdjointBuffersAllocated)
        return BEAGLE_ERROR_GENERAL;
    if (matrixAdjointBufferBase < 0)
        return BEAGLE_ERROR_OUT_OF_RANGE;

    const REALTYPE* evec = NULL;
    const REALTYPE* ievc = NULL;
    const REALTYPE* evals = NULL;
    int eigRc = gEigenDecomposition->getRawEigenDecomposition(eigenIndex, &evec, &ievc, &evals);
    if (eigRc != BEAGLE_SUCCESS || evec == NULL || ievc == NULL || evals == NULL) {
        return eigRc != BEAGLE_SUCCESS ? eigRc : BEAGLE_ERROR_GENERAL;
    }

    const int S = kStateCount;
    const int S2 = S * S;
    const int matrixIncr = S + T_PAD;
	const int M = matrixCount;
	if (M < 0 || matrixAdjointBufferBase + M > kMatrixCount) {
    	return BEAGLE_ERROR_OUT_OF_RANGE;
	}


    std::vector<int> blockStartsVec(S), blockDimsVec(S);
    int* blockStarts = blockStartsVec.data();
    int* blockDimsArr = blockDimsVec.data();
    int numBlocks = 0;
    for (int i = 0; i < S; i++) {
        blockStarts[numBlocks] = i;
        if (hasComplexEigenvalues && std::abs(evals[S + i]) > 1e-12) {
            blockDimsArr[numBlocks] = 2; i++;
        } else {
            blockDimsArr[numBlocks] = 1;
        }
        numBlocks++;
    }

    REALTYPE* eigenBasisGradShared = gAdjointWorkTemp.data();
    memset(eigenBasisGradShared, 0, S2 * sizeof(REALTYPE));

    const bool bastaThreading = (kBastaNumThreads > 1);
    const int bastaNumThreads = bastaThreading ? kBastaNumThreads : 1;


#pragma omp parallel num_threads(bastaNumThreads)
    {
        std::vector<REALTYPE> localTempVec(S2);
        std::vector<REALTYPE> transformedVecLocal(S2);
        std::vector<REALTYPE> eigenBasisGradVec(S2, REALTYPE(0));
        REALTYPE* localTemp = localTempVec.data();
        REALTYPE* transformed = transformedVecLocal.data();
        REALTYPE* eigenBasisGrad = eigenBasisGradVec.data();

#pragma omp for
    for (int m = 0; m < M; ++m) {
        const REALTYPE* matAdj = gTransitionMatrices[matrixAdjointBufferBase + m];
        if (matAdj == NULL) continue;

        memset(localTemp, 0, S2 * sizeof(REALTYPE));
        for (int i = 0; i < S; ++i)
            for (int j = 0; j < S; ++j) {
                REALTYPE val = matAdj[i * matrixIncr + j];
                if (val == REALTYPE(0)) continue;
                for (int b = 0; b < S; ++b) localTemp[i * S + b] += val * ievc[b * S + j];
            }


        memset(transformed, 0, S2 * sizeof(REALTYPE));
        for (int i = 0; i < S; ++i)
            for (int a = 0; a < S; ++a) {
                REALTYPE v = evec[i * S + a];
                if (v == REALTYPE(0)) continue;
                for (int b = 0; b < S; ++b) transformed[a * S + b] += v * localTemp[i * S + b];
            }

        REALTYPE t = (REALTYPE)branchLengths[m];
        for (int lb = 0; lb < numBlocks; lb++) {
            int ls = blockStarts[lb], ld = blockDimsArr[lb];
            for (int rb = 0; rb < numBlocks; rb++) {
                int rs = blockStarts[rb], rd = blockDimsArr[rb];

                if (ld == 1 && rd == 1) {
                    REALTYPE la = evals[ls], lbb = evals[rs];
                    REALTYPE tla = t*la, tlb = t*lbb;
                    REALTYPE coeff = (std::abs(tla-tlb) < 1e-12) ? t*std::exp(tla)
                        : (std::exp(tla) - std::exp(tlb)) / (la - lbb);
                    eigenBasisGrad[ls*S+rs] += transformed[ls*S+rs] * coeff;

                } else if (ld == 1 && rd == 2) {
                    REALTYPE la=evals[ls],rr=evals[rs],ri=evals[S+rs];
                    REALTYPE sr=rr-la,den=sr*sr+ri*ri,scale=std::exp(t*la),ic0,ic1;
                    if(den<1e-12){ic0=t;ic1=0;}else{REALTYPE ex=std::exp(t*sr),cs=std::cos(t*ri),sn=std::sin(t*ri);
                        ic0=(ex*(sr*cs+ri*sn)-sr)/den;ic1=(ex*(sr*sn-ri*cs)+ri)/den;}
                    REALTYPE c0=scale*ic0,c1=scale*ic1;
                    REALTYPE in0=transformed[ls*S+rs],in1=transformed[ls*S+rs+1];
                    eigenBasisGrad[ls*S+rs]+=c0*in0+c1*in1; eigenBasisGrad[ls*S+rs+1]+=-c1*in0+c0*in1;

                } else if (ld == 2 && rd == 1) {
                    REALTYPE lr=evals[ls],li=evals[S+ls],rb2=evals[rs];
                    REALTYPE sr=rb2-lr,den=sr*sr+li*li,ic0,ic1;
                    REALTYPE cI=std::cos(t*li),sI=std::sin(t*li);
                    if(den<1e-12){ic0=t;ic1=0;}else{REALTYPE ex=std::exp(t*sr);
                        ic0=(ex*(sr*cI+li*sI)-sr)/den;ic1=(ex*(sr*sI-li*cI)+li)/den;}
                    REALTYPE eR=std::exp(t*lr);
                    REALTYPE l00=eR*cI,l01=-eR*sI,l10=eR*sI,l11=eR*cI;
                    REALTYPE p0=l00*ic0-l01*ic1,p1=l00*ic1+l01*ic0;
                    REALTYPE p2=l10*ic0-l11*ic1,p3=l10*ic1+l11*ic0;
                    REALTYPE in0=transformed[ls*S+rs],in1=transformed[(ls+1)*S+rs];
                    eigenBasisGrad[ls*S+rs]+=p0*in0+p1*in1; eigenBasisGrad[(ls+1)*S+rs]+=p2*in0+p3*in1;

                } else {
                    REALTYPE lr=evals[ls],li=evals[S+ls],rr=evals[rs],ri=evals[S+rs];
                    REALTYPE sr=rr-lr,si1=li+ri,si2=ri-li;
                    REALTYPE e1=std::exp(t*lr),tli=t*li;
                    REALTYPE er=e1*std::cos(tli),ei=e1*std::sin(tli);
                    REALTYPE ex=std::exp(t*sr);
                    REALTYPE int1r,int1i,int2r,int2i;
                    REALTYPE d1=sr*sr+si1*si1;
                    if(d1<1e-12){int1r=t;int1i=0;}
					else{
                        REALTYPE cs1=std::cos(t*si1),sn1=std::sin(t*si1);
                        int1r=(sr*(ex*cs1-1)+si1*ex*sn1)/d1;int1i=(sr*ex*sn1-si1*(ex*cs1-1))/d1;
					}
                    REALTYPE d2=sr*sr+si2*si2;
                    if(d2<1e-12){int2r=t;int2i=0;}
					else{
                        REALTYPE cs2=std::cos(t*si2),sn2=std::sin(t*si2);
                        int2r=(sr*(ex*cs2-1)+si2*ex*sn2)/d2;int2i=(sr*ex*sn2-si2*(ex*cs2-1))/d2;
					}
                    REALTYPE pr=er*int1r+ei*int1i,pi_=er*int1i-ei*int1r;
                    REALTYPE mr_=er*int2r-ei*int2i,mi_=er*int2i+ei*int2r;
                    REALTYPE A=REALTYPE(0.5)*(mr_+pr),B=REALTYPE(0.5)*(mi_+pi_);
                    REALTYPE C=REALTYPE(0.5)*(pi_-mi_),D=REALTYPE(0.5)*(mr_-pr);
                    REALTYPE in00=transformed[ls*S+rs],in01=transformed[ls*S+rs+1];
                    REALTYPE in10=transformed[(ls+1)*S+rs],in11=transformed[(ls+1)*S+rs+1];
                    eigenBasisGrad[ls*S+rs] += A*in00+B*in01+C*in10+D*in11;
                    eigenBasisGrad[ls*S+rs+1] += -B*in00+A*in01-D*in10+C*in11;
                    eigenBasisGrad[(ls+1)*S+rs] += -C*in00-D*in01+A*in10+B*in11;
                    eigenBasisGrad[(ls+1)*S+rs+1] += D*in00-C*in01-B*in10+A*in11;
                }
            }
        }
    }

#pragma omp critical
        {
            for (int k = 0; k < S2; ++k) eigenBasisGradShared[k] += eigenBasisGrad[k];
        }
    }

    REALTYPE* backTemp = gAdjointWorkIexp.data();
    memset(backTemp, 0, S2 * sizeof(REALTYPE));
    for (int i = 0; i < S; ++i)
        for (int a = 0; a < S; ++a) {
            REALTYPE w = ievc[a * S + i];
            if (w == REALTYPE(0)) continue;
            for (int b = 0; b < S; ++b) backTemp[i * S + b] += w * eigenBasisGradShared[a * S + b];
        }

    memset(outRateGradient, 0, S2 * sizeof(double));
    for (int i = 0; i < S; ++i)
        for (int b = 0; b < S; ++b) {
            REALTYPE val = backTemp[i * S + b];
            if (val == REALTYPE(0)) continue;
            for (int j = 0; j < S; ++j) outRateGradient[i * S + j] += val * evec[j * S + b];
        }

    for (int i = 0; i < S; ++i)
        for (int j = i + 1; j < S; ++j)
            std::swap(outRateGradient[i * S + j], outRateGradient[j * S + i]);

    return BEAGLE_SUCCESS;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::accumulateBastaPartials(const int* operations,
	     				  									   const int operationCount,
							  	     				  		   const int* intervalStarts,
	     				  									   const int intervalStartsCount,
                                                               const double* intervalLengths,
                                                               const int populationSizesIndex,
                                                               const int coalescentIndex,
                                                               double* out) {
	int returnCode = BEAGLE_SUCCESS;

    std::fill(gBastaBuffers.begin(), gBastaBuffers.end(), REALTYPE(0));

    REALTYPE* e = gBastaBuffers.data() + 0 * kPartialsPaddedStateCount * kCoalescentBufferLength;
    REALTYPE* f = gBastaBuffers.data() + 1 * kPartialsPaddedStateCount * kCoalescentBufferLength;
    REALTYPE* g = gBastaBuffers.data() + 2 * kPartialsPaddedStateCount * kCoalescentBufferLength;
    REALTYPE* h = gBastaBuffers.data() + 3 * kPartialsPaddedStateCount * kCoalescentBufferLength;

    const REALTYPE* coalescent = gCoalescentBuffers.data() + kCoalescentBufferLength * coalescentIndex;
    const REALTYPE* sizes = gStateFrequencies[populationSizesIndex];
    bool threading = (kBastaNumThreads > 1);
    int numThreads = threading ? kBastaNumThreads : 1;
#pragma omp parallel for num_threads(numThreads)
    for (int interval = 0; interval < intervalStartsCount - 1; ++interval) { // TODO execute in parallel (no race conditions)
        const int start = intervalStarts[interval];
        const int end = intervalStarts[interval + 1];

        for (int op = start; op < end; ++op) { // TODO execute in parallel (has race conditions on e,f,g,h

            const int numOps = BEAGLE_BASTA_OP_COUNT;

            const int destinationPartialIndex = operations[op * numOps];
            const int child1PartialIndex = operations[op * numOps + 1];
            const int child1TransMatIndex = operations[op * numOps + 2];
            const int child2PartialIndex = operations[op * numOps + 3];
            const int child2TransMatIndex = operations[op * numOps + 4];
            const int accumulation1PartialIndex = operations[op * numOps + 5];
            const int accumulation2PartialIndex = operations[op * numOps + 6];
            const int intervalNumber = operations[op * numOps + 7];

            reduceWithinInterval(e, f, g, h,
                                 child1PartialIndex, child2PartialIndex,
                                 accumulation1PartialIndex, accumulation2PartialIndex,
                                 intervalNumber);
        }
    }
    REALTYPE logL = 0.0;

#pragma omp parallel num_threads(numThreads)
    {
	    REALTYPE threadLogL = 0.0;

#pragma omp for
	    for (int i = 0; i < intervalStartsCount - 1; ++i) {
	        const int op = intervalStarts[i];
	        const int numOps = BEAGLE_BASTA_OP_COUNT;
	        const int intervalNumber = operations[op * numOps + 7];

	        threadLogL += reduceAcrossIntervals(
                e, f, g, h,
                intervalNumber,
                intervalLengths[i],
                sizes,
                coalescent
            );
	    }

#pragma omp critical
	    {
	        logL += threadLogL;
	    }
    }
    out[0] = logL;

    if (kAdjointBuffersAllocated) {
        kUnclampedForwardDone = true;
    }

	return returnCode;
}


BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::accumulateBastaPartialsGrad(const int* operations,
	     				  									   const int operationCount,
					  	     				  		   		   const int* intervalStarts,
	     				  									   const int intervalStartsCount,
                                                               const double* intervalLengths,
                                                               const int populationSizesIndex,
                                                               const int coalescentIndex,
                                                               int eigenIndex,
                                                               int partialAdjointBufferBase,
                                                               int matrixAdjointBufferBase,
                                                               double* popSizeGradOut,
                                                               double* out) {
	int returnCode = BEAGLE_SUCCESS;
    if (!kAdjointBuffersAllocated && gPartialsGrad == NULL) // allocateBastaGradBuffers has not run, as on the GPU
        return BEAGLE_ERROR_GENERAL;

    const int S = kStateCount;
    const int S2 = S * S;

    if (kAdjointBuffersAllocated) {
        const REALTYPE* coalescent = gCoalescentBuffers.data() + kCoalescentBufferLength * coalescentIndex;
        const REALTYPE* sizes = gStateFrequencies[populationSizesIndex];

        std::vector<REALTYPE> popSizeScratch;
        REALTYPE* popSizeWork = NULL;
        if (popSizeGradOut != NULL) {
            popSizeScratch.assign(S, REALTYPE(0));
            popSizeWork = popSizeScratch.data();
        }
        adjointReversePass(operations, operationCount, intervalStarts, intervalStartsCount,
                           intervalLengths, sizes, coalescent,
                           eigenIndex,
                           partialAdjointBufferBase,
                           matrixAdjointBufferBase,
                           popSizeWork);
        if (popSizeGradOut != NULL) {
            beagleMemCpy(popSizeGradOut, popSizeWork, S);
        }
    } else {
        const int S_pad = kPartialsPaddedStateCount;
        const int L = kCoalescentBufferLength;

        REALTYPE* e = gBastaBuffers.data() + 0 * S_pad * L;
        REALTYPE* f = gBastaBuffers.data() + 1 * S_pad * L;
        REALTYPE* g = gBastaBuffers.data() + 2 * S_pad * L;
        REALTYPE* h = gBastaBuffers.data() + 3 * S_pad * L;

        std::fill(gBastaGradBuffers.begin(), gBastaGradBuffers.end(), REALTYPE(0));

        const size_t efghSliceSize = (size_t)S2 * L * S_pad;
        REALTYPE* eGrad = gBastaGradBuffers.data() + 0 * efghSliceSize;
        REALTYPE* fGrad = gBastaGradBuffers.data() + 1 * efghSliceSize;
        REALTYPE* gGrad = gBastaGradBuffers.data() + 2 * efghSliceSize;
        REALTYPE* hGrad = gBastaGradBuffers.data() + 3 * efghSliceSize;

        const REALTYPE* coalescent = gCoalescentBuffers.data() + L * coalescentIndex;
        const REALTYPE* sizes = gStateFrequencies[populationSizesIndex];

        for (int interval = 0; interval < intervalStartsCount - 1; ++interval) {
            const int start = intervalStarts[interval];
            const int end = intervalStarts[interval + 1];
            for (int op = start; op < end; ++op) {
                const int numOps = BEAGLE_BASTA_OP_COUNT;
                const int child1PartialIndex = operations[op * numOps + 1];
                const int child2PartialIndex = operations[op * numOps + 3];
                const int accumulation1PartialIndex = operations[op * numOps + 5];
                const int accumulation2PartialIndex = operations[op * numOps + 6];
                const int intervalNumber = operations[op * numOps + 7];
                reduceWithinIntervalGrad(eGrad, fGrad, gGrad, hGrad,
                                     child1PartialIndex, child2PartialIndex,
                                     accumulation1PartialIndex, accumulation2PartialIndex,
                                     intervalNumber);
            }
        }

        std::fill(out, out + S2, 0.0);
        std::vector<REALTYPE> tempGradLocal(S2);
        for (int i = 0; i < intervalStartsCount - 1; ++i) {
            const int op = intervalStarts[i];
            const int numOps = BEAGLE_BASTA_OP_COUNT;
            const int intervalNumber = operations[op * numOps + 7];
            reduceAcrossIntervalsGrad(e, f, g, h, eGrad, fGrad, gGrad, hGrad,
                                      tempGradLocal.data(), intervalNumber, intervalLengths[i],
                                      sizes, gCoalescentBuffers.data() + kCoalescentBufferLength * coalescentIndex);
            for (int ab = 0; ab < S2; ++ab) {
                out[ab] += tempGradLocal[ab];
            }
        }
    }

	return returnCode;
}



BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::uploadBastaSlabMetadata(const int* packed,
                                                                int packedLen) {
    (void)packed;
    (void)packedLen;
    return BEAGLE_ERROR_NO_IMPLEMENTATION;
}


BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::getBastaSlabConstants(int* opsPerBlock,
                                                               int* indexOffsetPat) {
    (void)opsPerBlock;
    (void)indexOffsetPat;
    return BEAGLE_ERROR_NO_IMPLEMENTATION;
}


//BEAGLE_CPU_TEMPLATE
//int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calculateEdgeDerivative(const int *postBufferIndices,
//                                                               const int *preBufferIndices,
//                                                               const int rootBufferIndex,
//                                                               const int *firstDerivativeIndices,
//                                                               const int *secondDerivativeIndices,
//                                                               const int categoryWeightsIndex,
//                                                               const int categoryRatesIndex,
//                                                               const int stateFrequenciesIndex,
//                                                               const int *cumulativeScaleIndices,
//                                                               int count,
//                                                               double *outFirstDerivative,
//                                                               double *outDiagonalSecondDerivative) {
//
//    bool byPartition = false;
////    int returnCode = calcEdgeDerivative(byPartition, postBufferIndices, preBufferIndices, rootBufferIndex,
////                                    firstDerivativeIndices, secondDerivativeIndices, categoryWeightsIndex,
////                                    categoryRatesIndex, stateFrequenciesIndex,
////                                    cumulativeScaleIndices, count, outFirstDerivative,
////                                    outDiagonalSecondDerivative, 0, kPatternCount);
//    fprintf(stderr, "Depricated");
//    return 0;
//}

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calculateEdgeDerivatives(const int *postBufferIndices,
                                                                   const int *preBufferIndices,
                                                                   const int *derivativeMatrixIndices,
                                                                   const int *categoryWeightsIndices,
                                                                   const int *categoryRatesIndices,
                                                                   const int *cumulativeScaleIndices,
                                                                   int count,
                                                                   double *outDerivatives,
                                                                   double *outSumDerivatives,
                                                                   double *outSumSquaredDerivatives) {
    int returnCode = BEAGLE_ERROR_GENERAL;

    BEAGLE_CHECK_INDICES(badPartialsIndices(postBufferIndices, count) || badPartialsIndices(preBufferIndices, count) ||
                         badMatrices(derivativeMatrixIndices, count));

    if (missingDenseMatrix(derivativeMatrixIndices, count))
        return BEAGLE_ERROR_OUT_OF_RANGE;

    if (kAutoPartitioningEnabled && kPartitionCount > 1) {



        int numOps  = BEAGLE_OP_COUNT;
        int numOpsP = BEAGLE_PARTITION_OP_COUNT;

        int perThreadCount = (count - (count % kPartitionCount)) / kPartitionCount;

#ifdef BEAGLE_DEBUG_FLOW
        std::cerr<<"Mark 0" << ", Original count = "<< count << ", perThreadCount = " << perThreadCount <<std::endl;
#endif


        if (gThreadOpCounts == NULL) {
            gThreadOpCounts = (int*) calloc(kNumThreads, sizeof(int));
        }



        std::fill(gThreadOpCounts, gThreadOpCounts + kNumThreads, 0);

#ifdef BEAGLE_DEBUG_FLOW
        //        std::cerr<<"Mark 1"<<std::endl;
#endif

        if (gThreadOperations == NULL) {

#ifdef BEAGLE_DEBUG_FLOW
            std::cerr<<"Mark 1.5"<<std::endl;
#endif
            gThreadOperations = (int**) calloc(kNumThreads, sizeof(int*));
        }

        for (int i=0; i<kPartitionCount; i++) {
            if (gThreadOperations[i] == NULL) {
                gThreadOperations[i] = (int*) malloc(sizeof(int) * BEAGLE_PARTITION_OP_COUNT * kBufferCount * kPartitionCount);
            }
        }

#ifdef BEAGLE_DEBUG_FLOW
        std::cerr<<"Mark 2"<<std::endl;
#endif

        for (int i = 0; i < count; i++) {
            const int partitionIndex = (i * kPartitionCount) / count;

            gThreadOperations[partitionIndex][gThreadOpCounts[partitionIndex] * numOpsP] = postBufferIndices[i];
            gThreadOperations[partitionIndex][gThreadOpCounts[partitionIndex] * numOpsP + 1] = preBufferIndices[i];
            gThreadOperations[partitionIndex][gThreadOpCounts[partitionIndex] * numOpsP + 2] = derivativeMatrixIndices[i];
            gThreadOperations[partitionIndex][gThreadOpCounts[partitionIndex] * numOpsP + 3] = categoryWeightsIndices[0];
            gThreadOperations[partitionIndex][gThreadOpCounts[partitionIndex] * numOpsP + 4] = i;
            gThreadOperations[partitionIndex][gThreadOpCounts[partitionIndex] * numOpsP + 5] = partitionIndex;
            gThreadOpCounts[partitionIndex]++;

        }


#ifdef BEAGLE_DEBUG_FLOW
        std::cerr<<"Mark 3"<<std::endl;
#endif

//
//        count *= kPartitionCount;
//        returnCode = upPartialsByPartitionAsync((const int*) gAutoPartitionOperations,
//                                                count);

        for (int i=0; i<kPartitionCount; i++) {
            auto b1485 = std::bind(&BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcEdgeLogDerivativesByAutoPartitionAsync, this,
                              (const int*) gThreadOperations[i],
                              gThreadOpCounts[i],
                              outDerivatives,
                              outSumDerivatives,
                              outSumSquaredDerivatives);
            std::packaged_task<void()> threadTask([b = std::move(b1485)]() mutable { b(); });

            gFutures[i] = threadTask.get_future();
            threadData* td = &gThreads[i];

            std::unique_lock<std::mutex> l(td->m);
            td->jobs.push(std::move(threadTask));
            l.unlock();

            gThreads[i].cv.notify_one();
        }

        for (int i=0; i<kNumThreads; i++) {
            gFutures[i].wait();
        }

        returnCode = BEAGLE_SUCCESS;

    } else {
        returnCode = calcEdgeLogDerivatives(
                postBufferIndices, preBufferIndices,
                derivativeMatrixIndices, NULL,
                categoryWeightsIndices, categoryRatesIndices, cumulativeScaleIndices,
                count,
                outDerivatives,
                outSumDerivatives, outSumSquaredDerivatives);
    }

    return returnCode;
}

        BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calculateCrossProducts(const int *postBufferIndices,
                                                              const int *preBufferIndices,
                                                              const int *categoryRatesIndices,
                                                              const int *categoryWeightsIndices,
                                                              const double *edgeLengths,
                                                              int count,
                                                              double *outSumDerivatives,
                                                              double *outSumSquaredDerivatives) {
    BEAGLE_CHECK_INDICES(badPartialsIndices(postBufferIndices, count) || badPartialsIndices(preBufferIndices, count));

    return calcCrossProducts(
            postBufferIndices, preBufferIndices,
            categoryRatesIndices,
            categoryWeightsIndices,
            edgeLengths,
            count,
            outSumDerivatives,
            outSumSquaredDerivatives);
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calculateAdjointCrossProducts(const int *postBufferIndices,
                                                                     const int *preBufferIndices,
                                                                     const int *transitionIndices,
                                                                     const int *categoryRatesIndices,
                                                                     const int *categoryWeightsIndices,
                                                                     const int rootPostOrderIndex,
                                                                     const int stateFrequenciesIndex,
                                                                     int count,
                                                                     double *outSumDerivatives,
                                                                     double *outSumSquaredDerivatives,
                                                                     const int *postScaleIndices,
                                                                     const int *preScaleIndices,
                                                                     const int cumulativeScaleIndex) {
#ifdef BEAGLE_INDEX_CHECKS
    if (badPartials(rootPostOrderIndex)) // read whatever the count
        return BEAGLE_ERROR_OUT_OF_RANGE;
    if (count > 0) {
        if (badPartialsIndices(postBufferIndices, count) || badPartialsIndices(preBufferIndices, count) ||
            badMatrices(transitionIndices, count))
            return BEAGLE_ERROR_OUT_OF_RANGE;
        // a node's scale indices, and then the root's cumulative one, are read only when not NONE
        bool rescaled = false;
        for (int i = 0; i < count; i++) {
            const int post = postScaleIndices ? postScaleIndices[i] : BEAGLE_OP_NONE;
            const int pre = preScaleIndices ? preScaleIndices[i] : BEAGLE_OP_NONE;
            if ((post != BEAGLE_OP_NONE && badScale(post)) || (pre != BEAGLE_OP_NONE && badScale(pre)))
                return BEAGLE_ERROR_OUT_OF_RANGE;
            rescaled |= (post != BEAGLE_OP_NONE || pre != BEAGLE_OP_NONE);
        }
        if (rescaled && cumulativeScaleIndex != BEAGLE_OP_NONE && badScale(cumulativeScaleIndex))
            return BEAGLE_ERROR_OUT_OF_RANGE;
    }
#endif
    return calcAdjointCrossProducts(
            postBufferIndices, preBufferIndices,
            transitionIndices,
            categoryRatesIndices,
            categoryWeightsIndices,
            rootPostOrderIndex,
            stateFrequenciesIndex,
            count,
            outSumDerivatives,
            outSumSquaredDerivatives,
            postScaleIndices,
            preScaleIndices,
            cumulativeScaleIndex);
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calculateAdjointDerivative(
        const BeagleBranchOperation *operations,
        const int categoryRatesIndex,
        const int categoryWeightsIndex,
        const int rootPostOrderIndex,
        const int stateFrequenciesIndex,
        const int operationCount,
        double *outSumDerivatives,
        double *outSumSquaredDerivatives) {
    std::vector<int> post(operationCount), pre(operationCount), eigen(operationCount);
    for (int i = 0; i < operationCount; ++i) {
        post[i]  = operations[i].postOrderPartials;
        pre[i]   = operations[i].preOrderPartials;
        eigen[i] = operations[i].branchTransitionMatrix;
    }
    BEAGLE_CHECK_INDICES(badPartials(rootPostOrderIndex) ||
                         (operationCount > 0 &&
                          (badPartialsIndices(post.data(), operationCount) ||
                           badPartialsIndices(pre.data(), operationCount) ||
                           badMatrices(eigen.data(), operationCount))));
    return calcAdjointCrossProducts(
            post.data(), pre.data(), eigen.data(),
            &categoryRatesIndex, &categoryWeightsIndex,
            rootPostOrderIndex, stateFrequenciesIndex,
            operationCount, outSumDerivatives, outSumSquaredDerivatives,
            NULL, NULL, BEAGLE_OP_NONE);
}

inline double proportionRepeated(const int* branchEigenIndices, const int start, const int end) {
    double sum = 0.0;
    int last = branchEigenIndices[start];
    for (int i = start + 1; i < end; ++i) {
        const int index = branchEigenIndices[i];
        if (index == last) {
            ++sum;
        } else {
            last = index;
        }
    }
    return sum / (double) (end - start);
}

template <typename REALTYPE>
void computePerSiteLikelihoods(
        const REALTYPE* BEAGLE_RESTRICT rootPostOrderPartial,
        const REALTYPE* BEAGLE_RESTRICT stateFrequencies,
        const REALTYPE* BEAGLE_RESTRICT categoryWeights,
        const int kStateCount,
        const int kStateCountModFour,
        const int patternStart,
        const int patternEnd,
        const int kPatternCount,
        const int kCategoryCount,
        const int kPartialsPaddedStateCount,
        REALTYPE* BEAGLE_RESTRICT perSiteLikelihoods) {

    for (int pattern = patternStart; pattern < patternEnd; ++pattern) {

        REALTYPE likelihood = REALTYPE(0);
        for (int category = 0; category < kCategoryCount; ++category) {
            const int patternIndex = category * kPatternCount + pattern;
            const int v = patternIndex * kPartialsPaddedStateCount;

            int k = 0;
            REALTYPE categoryLikelihood1 = REALTYPE(0), categoryLikelihood2 = REALTYPE(0);
            #pragma clang loop vectorize(enable)
            for ( ; k < kStateCountModFour; k += 4) {
                categoryLikelihood1 += rootPostOrderPartial[v + k + 0] * stateFrequencies[k + 0];
                categoryLikelihood2 += rootPostOrderPartial[v + k + 1] * stateFrequencies[k + 1];
                categoryLikelihood1 += rootPostOrderPartial[v + k + 2] * stateFrequencies[k + 2];
                categoryLikelihood2 += rootPostOrderPartial[v + k + 3] * stateFrequencies[k + 3];
            }

            for ( ; k < kStateCount; k++) {
                categoryLikelihood1 += rootPostOrderPartial[v + k] * stateFrequencies[k];
            }

            const REALTYPE weight = categoryWeights[category];
            likelihood += (categoryLikelihood1 + categoryLikelihood2) * weight;
        }

        perSiteLikelihoods[pattern] = likelihood;
    }
}

BEAGLE_CPU_TEMPLATE
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::prepareAdjoint(const int *branchEigenIndices, int count) {
    for (int i = 0; i < count; ++i) {
        gEigenDecomposition->prepareAdjoint(gBranchEigenInfo[branchEigenIndices[i]].eigenIndex);
    }
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcAdjointCrossProducts(const int *postBufferIndices,
                                                                const int *preBufferIndices,
                                                                const int *transitionIndices,
                                                                const int *categoryRatesIndices,
                                                                const int *categoryWeightsIndices,
                                                                const int rootPostOrderIndex,
                                                                const int stateFrequenciesIndex,
                                                                int count,
                                                                double *outSumDerivatives,
                                                                double *outSumSquaredDerivatives,
                                                                const int *postScaleIndices,
                                                                const int *preScaleIndices,
                                                                const int cumulativeScaleIndex) {

    int returnCode = BEAGLE_SUCCESS;

    prepareAdjoint(transitionIndices, count);

    const double *categoryRates    = gCategoryRates[categoryRatesIndices[0]];
    const REALTYPE *categoryWeights = gCategoryWeights[categoryWeightsIndices[0]];
    const REALTYPE *stateFrequencies = gStateFrequencies[stateFrequenciesIndex];

    REALTYPE* buffer;
    std::vector<REALTYPE> realTypeBuffer;
    if constexpr (!std::is_same_v<REALTYPE, double>) {
        realTypeBuffer.resize(kStateCount * kStateCount * kCategoryCount);
        buffer = realTypeBuffer.data();
    } else {
        buffer = outSumDerivatives;
    }

    if (static_cast<int>(gPatternScaleTmp.size()) < kPatternCount) {
        gPatternScaleTmp.resize(kPatternCount);
    }
    REALTYPE* perSiteLikelihoods = gPatternScaleTmp.data();

#ifndef USE_BRANCH_LIKELIHOOD
    computePerSiteLikelihoods(gPartials[rootPostOrderIndex],
        stateFrequencies, categoryWeights,
        kStateCount, (kStateCount / 4) * 4, 0, kPatternCount, kPatternCount, kCategoryCount, kPartialsPaddedStateCount,
        perSiteLikelihoods);
#endif

    const int outerProductSize = kPartitionCount * kStateCount * kStateCount;
    if (static_cast<int>(gOuterProductTmp.size()) < outerProductSize) {
        gOuterProductTmp.resize(outerProductSize);
    }

    // Per-partition scratch for the (possibly branch-specific, rescaling-corrected)
    // per-site likelihoods; sized generously so calcAdjointCrossProductsRange's
    // currentPartition-indexed slice is always in-bounds.
    const int correctedLkSize = kPartitionCount * kPatternCount;
    if (static_cast<int>(gAdjointScaleCorrectedLk.size()) < correctedLkSize) {
        gAdjointScaleCorrectedLk.resize(correctedLkSize);
    }

#ifdef USE_BRANCH_LIKELIHOOD
    // [node][pattern], absolute indices -- see calcAdjointBranchMarginalLikelihoods.
    const int branchLkSize = count * kPatternCount;
    if (static_cast<int>(gAdjointBranchLikelihoodTmp.size()) < branchLkSize) {
        gAdjointBranchLikelihoodTmp.resize(branchLkSize);
    }
    REALTYPE* branchMarginalLk = gAdjointBranchLikelihoodTmp.data();
    // Precomputed once, single-threaded, covering every node/pattern -- see
    // calcAdjointBranchMarginalLikelihoods for why this can't be folded into the
    // (possibly multi-threaded) main loop below.
    calcAdjointBranchMarginalLikelihoods(postBufferIndices, preBufferIndices, transitionIndices,
                                         categoryRates, categoryWeights, branchMarginalLk,
                                         0, count, 0, kPatternCount, 0);
#else
    REALTYPE* branchMarginalLk = nullptr;
#endif

    if (kAutoPartitioningEnabled && kPartitionCount > 1) {
        returnCode = calcAdjointCrossProductsByPartitionAsync(
            postBufferIndices, preBufferIndices, transitionIndices,
            categoryRates, categoryWeights, perSiteLikelihoods,
            buffer, count, postScaleIndices, preScaleIndices, cumulativeScaleIndex,
            branchMarginalLk);
    } else {
        std::fill(buffer, buffer + kStateCount * kStateCount, REALTYPE(0));
        calcAdjointCrossProductsRange(postBufferIndices, preBufferIndices, transitionIndices,
                                      categoryRates, categoryWeights, perSiteLikelihoods,
                                      buffer,
                                      0, count, 0, kPatternCount, 0,
                                      postScaleIndices, preScaleIndices, cumulativeScaleIndex,
                                      branchMarginalLk);
    }

    if constexpr (!std::is_same_v<REALTYPE, double>) {
        beagleMemCpy(outSumDerivatives, buffer, kStateCount * kStateCount * kCategoryCount);
    }

    return returnCode;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcAdjointCrossProductsByNodeAsync(
        const int *postBufferIndices,
        const int *preBufferIndices,
        const int *branchEigenIndices,
        const double *categoryRates,
        const REALTYPE *categoryWeights,
        const REALTYPE *perSiteLikelihoods,
        REALTYPE *buffer,
        int count,
        const int *postScaleIndices,
        const int *preScaleIndices,
        const int cumulativeScaleIndex,
        REALTYPE *branchMarginalLk) {

    const int gradSize = kStateCount * kStateCount;

    std::fill(buffer, buffer + gradSize, REALTYPE(0));
    if (kNumThreads > 1) {
        std::fill(gAdjointPartitionBuffers.begin(), gAdjointPartitionBuffers.end(), REALTYPE(0));
    }

    const int chunkSize = (count + kNumThreads - 1) / kNumThreads;

    for (int i = 0; i < kNumThreads; ++i) {
        const int startNode = i * chunkSize;
        const int endNode   = std::min(startNode + chunkSize, count);
        auto b = [this, i, gradSize, buffer,
                  postBufferIndices, preBufferIndices, branchEigenIndices,
                  categoryRates, categoryWeights, perSiteLikelihoods,
                  startNode, endNode,
                  postScaleIndices, preScaleIndices, cumulativeScaleIndex, branchMarginalLk]() {
            REALTYPE* partBuffer = (i == 0) ? buffer
                                            : gAdjointPartitionBuffers.data() + i * gradSize;
            if (startNode < endNode) {
                calcAdjointCrossProductsRange(postBufferIndices, preBufferIndices, branchEigenIndices,
                                              categoryRates, categoryWeights, perSiteLikelihoods,
                                              partBuffer,
                                              startNode, endNode, 0, kPatternCount, i,
                                              postScaleIndices, preScaleIndices, cumulativeScaleIndex,
                                              branchMarginalLk);
            }
        };
        std::packaged_task<void()> threadTask(std::move(b));

        gFutures[i] = threadTask.get_future();
        threadData* td = &gThreads[i];
        std::unique_lock<std::mutex> l(td->m);
        td->jobs.push(std::move(threadTask));
        l.unlock();
        gThreads[i].cv.notify_one();
    }

    for (int i = 0; i < kNumThreads; ++i) {
        gFutures[i].wait();
    }

    if (kNumThreads > 1) {
        for (int t = 1; t < kNumThreads; ++t) {
            const REALTYPE* partBuffer = gAdjointPartitionBuffers.data() + t * gradSize;
            for (int i = 0; i < gradSize; ++i) {
                buffer[i] += partBuffer[i];
            }
        }
    }

    return BEAGLE_SUCCESS;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcAdjointCrossProductsByPartitionAsync(
        const int *postBufferIndices,
        const int *preBufferIndices,
        const int *branchEigenIndices,
        const double *categoryRates,
        const REALTYPE *categoryWeights,
        const REALTYPE *perSiteLikelihoods,
        REALTYPE *buffer,
        int count,
        const int *postScaleIndices,
        const int *preScaleIndices,
        const int cumulativeScaleIndex,
        REALTYPE *branchMarginalLk) {

    const int gradSize = kStateCount * kStateCount;
    const double prop = proportionRepeated(branchEigenIndices, 0, count);

    std::fill(buffer, buffer + gradSize, REALTYPE(0));
    if (kNumThreads > 1) {
        std::fill(gAdjointPartitionBuffers.begin(), gAdjointPartitionBuffers.end(), REALTYPE(0));
    }

    memset(gThreadOpCounts, 0, sizeof(int) * kNumThreads);
    for (int p = 0; p < kPartitionCount; ++p) {
        const int t = p % kNumThreads;
        gThreadOperations[t][gThreadOpCounts[t]] = p;
        ++gThreadOpCounts[t];
    }

    for (int i = 0; i < kNumThreads; ++i) {
        auto b = [this, i, count, gradSize, buffer, prop,
                  postBufferIndices, preBufferIndices, branchEigenIndices,
                  categoryRates, categoryWeights, perSiteLikelihoods,
                  postScaleIndices, preScaleIndices, cumulativeScaleIndex, branchMarginalLk]() {
            for (int opIdx = 0; opIdx < gThreadOpCounts[i]; ++opIdx) {
                const int p = gThreadOperations[i][opIdx];
                const int startPattern = gPatternPartitionsStartPatterns[p];
                const int endPattern   = gPatternPartitionsStartPatterns[p + 1];
                REALTYPE* partBuffer = (i == 0) ? buffer
                                                : gAdjointPartitionBuffers.data() + p * gradSize;
                calcAdjointCrossProductsRange(postBufferIndices, preBufferIndices, branchEigenIndices,
                                              categoryRates, categoryWeights, perSiteLikelihoods,
                                              partBuffer,
                                              0, count, startPattern, endPattern, p,
                                              postScaleIndices, preScaleIndices, cumulativeScaleIndex,
                                              branchMarginalLk);
            }
        };
        std::packaged_task<void()> threadTask(std::move(b));

        gFutures[i] = threadTask.get_future();
        threadData* td = &gThreads[i];
        std::unique_lock<std::mutex> l(td->m);
        td->jobs.push(std::move(threadTask));
        l.unlock();
        gThreads[i].cv.notify_one();
    }

    for (int i = 0; i < kNumThreads; ++i) {
        gFutures[i].wait();
    }

    if (kNumThreads > 1) {
        for (int t = 1; t < kNumThreads; ++t) {
            for (int opIdx = 0; opIdx < gThreadOpCounts[t]; ++opIdx) {
                const int p = gThreadOperations[t][opIdx];
                const REALTYPE* partBuffer = gAdjointPartitionBuffers.data() + p * gradSize;
                for (int i = 0; i < gradSize; ++i) {
                    buffer[i] += partBuffer[i];
                }
            }
        }
    }

    return BEAGLE_SUCCESS;
}

BEAGLE_CPU_TEMPLATE
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcAdjointCrossProductsRange(
        const int *postBufferIndices,
        const int *preBufferIndices,
        const int *branchEigenIndices,
        const double *categoryRates,
        const REALTYPE *categoryWeights,
        const REALTYPE *perSiteLikelihoods,
        REALTYPE *buffer,
        int startNode,
        int endNode,
        int startPattern,
        int endPattern,
        int currentPartition,
        const int *postScaleIndices,
        const int *preScaleIndices,
        const int cumulativeScaleIndex,
        REALTYPE *branchMarginalLk) {

#ifndef USE_BRANCH_LIKELIHOOD
    // If a branch's post-order and/or pre-order partials were independently
    // rescaled (see beagleCalculateAdjointCrossProductDerivative doc), the
    // "scale = categoryWeight / Lk" term below implicitly assumes UNSCALED
    // post/pre partials -- using the raw (scaled) Lk without compensation
    // would leave each branch's contribution off by that branch's own
    // (post-order-scale * pre-order-cumulative-scale / root-cumulative-scale)
    // factor. Correct for this by building a per-node "effective Lk" =
    // Lk_measured * rootCumulativeScale / (nodeOwnPostScale * nodePreCumulativeScale),
    // and using THAT (instead of the raw shared perSiteLikelihoods) for any
    // node with a non-NONE scale index. postScaleIndices[node]/preScaleIndices[node]
    // hold RAW/LOG-format buffer indices exactly as populated by the caller's
    // post-order pass (postScaleIndices, one factor per node) and pre-order
    // cumulative-scale bookkeeping (preScaleIndices, already log-summed via
    // beagleAccumulateScaleFactors); cumulativeScaleIndex is the same root
    // cumulative-scale buffer already used for the ordinary log-likelihood.
    // (Under USE_BRANCH_LIKELIHOOD this is moot: branchMarginalLk is recomputed
    // directly from each branch's own -- already rescaled -- partials, so it
    // self-corrects for rescaling with no separate bookkeeping needed; see
    // calcAdjointBranchMarginalLikelihoods.)
    const bool applyScaleCorrection = (postScaleIndices != NULL) || (preScaleIndices != NULL);
    REALTYPE* correctedLk = applyScaleCorrection
            ? gAdjointScaleCorrectedLk.data() + currentPartition * kPatternCount
            : NULL;
#endif

    const double prop = proportionRepeated(branchEigenIndices, startNode, endNode);
    const bool useMultiCollector = ((endPattern - startPattern) > 1 || prop > 0.1);

    REALTYPE* outerProductTmp = gOuterProductTmp.data() + currentPartition * kStateCount * kStateCount;

    for (int category = 0; category < kCategoryCount; ++category) {

        const REALTYPE categoryRate   = static_cast<REALTYPE>(categoryRates[category]);
        const int      infoOffset     = category * kPartialsPaddedStateCount;
        const REALTYPE categoryWeight = categoryWeights[category];

        auto iterateOverNodes = [&](auto& collector) {

            auto setupCollector = [&](const int branchEigenIndex) {
                const BranchEigenInfo& info = gBranchEigenInfo[branchEigenIndex];
                AdjointIntegralPlan<REALTYPE>* plan =
                    gEigenDecomposition->getAdjointMethodsPtr(info.eigenIndex);
                collector.setTime(categoryRate * info.branchLength,
                    info.expat + infoOffset,
                    info.cosbt + infoOffset, info.sinbt + infoOffset,
                    info.expatcosbt + infoOffset, info.expatsinbt + infoOffset);
                collector.setPlan(plan);
                collector.flush();
            };

            setupCollector(branchEigenIndices[0]);
            int lastBranchEigenIndex = branchEigenIndices[0];

            for (int nodeNum = startNode; nodeNum < endNode; ++nodeNum) {
                const int branchEigenIndex = branchEigenIndices[nodeNum];
                if (branchEigenIndex != lastBranchEigenIndex) {
                    collector.accumulateEigenBasisGradient();
                    setupCollector(branchEigenIndex);
                    lastBranchEigenIndex = branchEigenIndex;
                }

                const REALTYPE *preOrderPartial = gPartials[preBufferIndices[nodeNum]];
                const int *tipStates = gTipStates[postBufferIndices[nodeNum]];

#ifdef USE_BRANCH_LIKELIHOOD
                const REALTYPE *nodeLikelihoods = branchMarginalLk + nodeNum * kPatternCount;
#else
                const REALTYPE *nodeLikelihoods = perSiteLikelihoods;
                if (applyScaleCorrection) {
                    const int postIdx = postScaleIndices ? postScaleIndices[nodeNum] : BEAGLE_OP_NONE;
                    const int preIdx  = preScaleIndices  ? preScaleIndices[nodeNum]  : BEAGLE_OP_NONE;
                    if (postIdx == BEAGLE_OP_NONE && preIdx == BEAGLE_OP_NONE) {
                        // Neither this branch's post-order nor pre-order partial was
                        // independently rescaled (e.g. a tip feeding straight off the
                        // root): no correction needed for this node.
                        nodeLikelihoods = perSiteLikelihoods;
                    } else {
                        const REALTYPE* postScaleBuffer = (postIdx != BEAGLE_OP_NONE) ? gScaleBuffers[postIdx] : NULL;
                        const REALTYPE* preScaleBuffer  = (preIdx  != BEAGLE_OP_NONE) ? gScaleBuffers[preIdx]  : NULL;
                        const REALTYPE* rootScaleBuffer = (cumulativeScaleIndex != BEAGLE_OP_NONE) ? gScaleBuffers[cumulativeScaleIndex] : NULL;
                        for (int k = startPattern; k < endPattern; ++k) {
                            // postScaleBuffer/preScaleBuffer/rootScaleBuffer are all
                            // CUMULATIVE (always LOG, built by
                            // beagleAccumulateScaleFactors) buffers -- NOT plain
                            // individual scale-write buffers, whose RAW/LOG format
                            // instead follows kFlags. A node's own post-order buffer
                            // generally carries more than just its own scale-write:
                            // it inherits every rescaled descendant's scale factor
                            // too, so postScaleIndices[node] must be built the same
                            // way as preScaleIndices (accumulateScaleFactors over
                            // every rescaled node at-or-below it), not passed as a
                            // single individual buffer index.
                            REALTYPE logCorrection = REALTYPE(0);
                            if (postScaleBuffer != NULL) {
                                logCorrection += postScaleBuffer[k];
                            }
                            if (preScaleBuffer != NULL) {
                                logCorrection += preScaleBuffer[k];
                            }
                            if (rootScaleBuffer != NULL) {
                                logCorrection -= rootScaleBuffer[k];
                            }
                            // TODO Simplify scale-factors here
                            correctedLk[k] = perSiteLikelihoods[k] * std::exp(-logCorrection);
                        }
                        nodeLikelihoods = correctedLk;
                    }
                }
#endif

                if (tipStates != nullptr) {
                    calcAdjointCrossProducts<States, WithRotation>(
                        nullptr, tipStates, preOrderPartial,
                        branchEigenIndex, nodeLikelihoods,
                        category, categoryWeight,
                        collector, startPattern, endPattern, currentPartition);
                } else {
                    const REALTYPE *postOrderPartial = gPartials[postBufferIndices[nodeNum]];
                    calcAdjointCrossProducts<Partials, WithRotation>(
                        postOrderPartial, nullptr, preOrderPartial,
                        branchEigenIndex, nodeLikelihoods,
                        category, categoryWeight,
                        collector, startPattern, endPattern, currentPartition);
                }
            }

            collector.accumulateEigenBasisGradient();
        };

        if (useMultiCollector) {
            MultipleCollector<REALTYPE> collector(buffer, outerProductTmp, kStateCount, kStateCount);
            iterateOverNodes(collector);
        } else {
            SimpleCollector<REALTYPE> collector(buffer, kStateCount, kStateCount);
            iterateOverNodes(collector);
        }
    }
}

BEAGLE_CPU_TEMPLATE template <typename T, typename V, typename CC>
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcAdjointCrossProducts(
        const REALTYPE* inPartialsPost, const int* tipsPost,
        const REALTYPE* inPartialsPre,
        const int branchEigenIndex,
        const REALTYPE* perSiteLikelihoods,
        const int category, const REALTYPE categoryWeight,
        CC& collector,
        int startPattern,
        int endPattern,
        int currentPartition) {

    const int matrixIncr = kStateCount + T_PAD;
    const int stateCountModFour = (kStateCount / 4) * 4;

    const BranchEigenInfo& info = gBranchEigenInfo[branchEigenIndex];

    const REALTYPE* ievc = gEigenDecomposition->getInverseEigenVectorsPtr(info.eigenIndex);
    const REALTYPE* tievc = gEigenDecomposition->getBackwardsInverseEigenVectorsPtr(info.eigenIndex);

    const int v = category * kPartialsPaddedStateCount * kPatternCount + kPartialsPaddedStateCount * startPattern;

    const REALTYPE* preOrderPartials = &inPartialsPre[v];
    const REALTYPE* postOrderPartials = &inPartialsPost[v];

    for (int k = startPattern; k < endPattern; k++) {

        REALTYPE *lhs, *rhs;

        if constexpr (std::is_same_v<V, WithRotation>) {

            lhs = gPartialTmp1.data() + currentPartition * kPartialsPaddedStateCount;
            rhs = gPartialTmp2.data() + currentPartition * kPartialsPaddedStateCount;

            if constexpr (std::is_same_v<T, States>) {

                const int state2 = tipsPost[k];

                matVecDual<Partials, States>(
                    tievc, preOrderPartials, 0,
                    ievc, nullptr, state2,
                    matrixIncr,
                    [lhs, rhs](int i, REALTYPE s1, REALTYPE s2) {
                        lhs[i] = s1;
                        rhs[i] = s2;
                    });

            } else if constexpr (std::is_same_v<T, Partials>) {

                matVecDual<Partials, Partials>(
                    tievc, preOrderPartials, 0,
                    ievc, postOrderPartials, 0,
                    matrixIncr,
                    [lhs, rhs](int i, REALTYPE s1, REALTYPE s2) {
                        lhs[i] = s1;
                        rhs[i] = s2;
                    });

            } else {
                fprintf(stderr, "Unknown input type\n");
                exit(-1);
            }

        } else {
            static_assert(always_false<V>::value, "Unsupported lack of rotation");
        }

        // perSiteLikelihoods is ordinarily gPatternScaleTmp itself (see caller), but
        // calcAdjointCrossProductsRange may substitute a branch-specific array here
        // instead -- either rescaling-corrected (default path) or freshly recomputed
        // from this branch's own partials (USE_BRANCH_LIKELIHOOD, via
        // calcAdjointBranchMarginalLikelihoods) -- always use the parameter, not the
        // member, so that substitution takes effect.
        const REALTYPE scale = gPatternWeights[k] * categoryWeight / perSiteLikelihoods[k];

        collector.accumulateScaledOuterProducts(lhs, rhs, scale);

        preOrderPartials  += kPartialsPaddedStateCount;
        postOrderPartials += kPartialsPaddedStateCount;
    }

}

BEAGLE_CPU_TEMPLATE
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcAdjointBranchMarginalLikelihoods(
        const int *postBufferIndices,
        const int *preBufferIndices,
        const int *branchEigenIndices,
        const double *categoryRates,
        const REALTYPE *categoryWeights,
        REALTYPE *branchMarginalLk,
        int startNode,
        int endNode,
        int startPattern,
        int endPattern,
        int currentPartition) {

    // USE_BRANCH_LIKELIHOOD's per-branch normalizer must be the branch's MARGINAL
    // (category-weighted) site likelihood -- matching what the default path's
    // perSiteLikelihoods already holds -- not any single category's own conditional
    // likelihood. calcAdjointCrossProductsRange's main loop is category-outer/
    // node-inner (to group consecutive same-branchEigenIndex nodes), so a given
    // node's marginal sum isn't complete until every category has been visited;
    // precompute it here, once per node, before that loop runs.
    for (int nodeNum = startNode; nodeNum < endNode; ++nodeNum) {
        std::fill(branchMarginalLk + nodeNum * kPatternCount + startPattern,
                  branchMarginalLk + nodeNum * kPatternCount + endPattern,
                  REALTYPE(0));
    }

    const int matrixIncr = kStateCount + T_PAD;

    REALTYPE* lhs = gPartialTmp1.data() + currentPartition * kPartialsPaddedStateCount;
    REALTYPE* rhs = gPartialTmp2.data() + currentPartition * kPartialsPaddedStateCount;

    CollectorBase<REALTYPE> collector(nullptr, kStateCount, kStateCount);

    for (int category = 0; category < kCategoryCount; ++category) {

        const REALTYPE categoryRate   = static_cast<REALTYPE>(categoryRates[category]);
        const int      infoOffset     = category * kPartialsPaddedStateCount;
        const REALTYPE categoryWeight = categoryWeights[category];
        const int      v = category * kPartialsPaddedStateCount * kPatternCount + kPartialsPaddedStateCount * startPattern;

        for (int nodeNum = startNode; nodeNum < endNode; ++nodeNum) {

            const BranchEigenInfo& info = gBranchEigenInfo[branchEigenIndices[nodeNum]];
            collector.setTime(categoryRate * info.branchLength,
                info.expat + infoOffset, info.cosbt + infoOffset, info.sinbt + infoOffset,
                info.expatcosbt + infoOffset, info.expatsinbt + infoOffset);
            collector.setPlan(gEigenDecomposition->getAdjointMethodsPtr(info.eigenIndex));

            const REALTYPE* ievc  = gEigenDecomposition->getInverseEigenVectorsPtr(info.eigenIndex);
            const REALTYPE* tievc = gEigenDecomposition->getBackwardsInverseEigenVectorsPtr(info.eigenIndex);

            const REALTYPE* preOrderPartials  = &gPartials[preBufferIndices[nodeNum]][v];
            const int* tipStates = gTipStates[postBufferIndices[nodeNum]];
            const REALTYPE* postOrderPartials = (tipStates != nullptr) ? nullptr : &gPartials[postBufferIndices[nodeNum]][v];

            REALTYPE* outRow = branchMarginalLk + nodeNum * kPatternCount;

            for (int k = startPattern; k < endPattern; ++k) {

                if (tipStates != nullptr) {
                    const int state2 = tipStates[k];
                    matVecDual<Partials, States>(
                        tievc, preOrderPartials, 0,
                        ievc, nullptr, state2,
                        matrixIncr,
                        [lhs, rhs](int i, REALTYPE s1, REALTYPE s2) {
                            lhs[i] = s1;
                            rhs[i] = s2;
                        });
                } else {
                    matVecDual<Partials, Partials>(
                        tievc, preOrderPartials, 0,
                        ievc, postOrderPartials, 0,
                        matrixIncr,
                        [lhs, rhs](int i, REALTYPE s1, REALTYPE s2) {
                            lhs[i] = s1;
                            rhs[i] = s2;
                        });
                    postOrderPartials += kPartialsPaddedStateCount;
                }

                const REALTYPE branchLk = collector.plan->branchLikelihoodInEigenBasis(collector, lhs, rhs);
                outRow[k] += categoryWeight * branchLk;

                preOrderPartials += kPartialsPaddedStateCount;
            }
        }
    }
}

BEAGLE_CPU_TEMPLATE template <typename First, typename Second, typename Body>
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::matVecDual(
        const REALTYPE* BEAGLE_RESTRICT mat1, const REALTYPE* BEAGLE_RESTRICT vec1, const int state1,
        const REALTYPE* BEAGLE_RESTRICT mat2, const REALTYPE* BEAGLE_RESTRICT vec2, const int state2,
        const int matrixIncr,
        Body body) {

    if constexpr (
        !(std::is_same_v<First,  None> || std::is_same_v<First,  States> || std::is_same_v<First,  Partials>) ||
        !(std::is_same_v<Second, None> || std::is_same_v<Second, States> || std::is_same_v<Second, Partials>)
        ) {
        static_assert(always_false<First>::value, "Unsupported type T");
    }

    const int stateCountModFour = (kStateCount / 4) * 4;

    for (int i = 0; i < kStateCount; i++) {

        const REALTYPE* BEAGLE_RESTRICT row1 = mat1 + i * matrixIncr;
        const REALTYPE* BEAGLE_RESTRICT row2 = mat2 + i * matrixIncr;

        REALTYPE sum1A, sum1B, sum2A, sum2B;
        if constexpr (std::is_same_v<First, States>) {
            sum1A = row1[state1];
            sum1B = REALTYPE(0);
        } else if constexpr (std::is_same_v<First, Partials>) {
            sum1A = REALTYPE(0);
            sum1B = REALTYPE(0);
        }

        if constexpr (std::is_same_v<Second, States>) {
            sum2A = row2[state2];
            sum2B = REALTYPE(0);
        } else if constexpr (std::is_same_v<Second, Partials>) {
            sum2A = REALTYPE(0);
            sum2B = REALTYPE(0);
        }

        int j = 0;
        #pragma clang loop vectorize(enable)
        for ( ; j < stateCountModFour; j += 4) {
            if constexpr (std::is_same_v<First, Partials>) {
                sum1A += row1[j + 0] * vec1[j + 0];
                sum1B += row1[j + 1] * vec1[j + 1];

                sum1A += row1[j + 2] * vec1[j + 2];
                sum1B += row1[j + 3] * vec1[j + 3];
            }
            if constexpr (std::is_same_v<Second, Partials>) {
                sum2A += row2[j + 0] * vec2[j + 0];
                sum2B += row2[j + 1] * vec2[j + 1];

                sum2A += row2[j + 2] * vec2[j + 2];
                sum2B += row2[j + 3] * vec2[j + 3];
            }
        }

        for ( ; j < kStateCount; j++) {
            if constexpr (std::is_same_v<First, Partials>) {
                sum1A += row1[j] * vec1[j];
            }
            if constexpr (std::is_same_v<Second, Partials>) {
                sum2A += row2[j] * vec2[j];
            }
        }

        body(i, sum1A + sum1B, sum2A + sum2B);
    }
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::updatePartialsByPartition(const int* operations,
                                                                 int count) {

    int returnCode = BEAGLE_ERROR_GENERAL;

    BEAGLE_CHECK_INDICES(badOperations(operations, count, true, BEAGLE_OP_NONE, false, false));

    if (kThreadingEnabled) {
        returnCode = upPartialsByPartitionAsync(operations,
                                                count);
    } else {
        bool byPartition = true;
        returnCode = upPartials(byPartition,
                                operations,
                                count,
                                BEAGLE_OP_NONE,
                                BEAGLE_PARTIALS_BOTTOM);
    }

    return returnCode;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::updatePrePartialsByPartition(const int* operations,
                                                                    int count,
                                                                    BeaglePartialsType partialsType) {

    int returnCode = BEAGLE_ERROR_GENERAL;

    BEAGLE_CHECK_INDICES(badOperations(operations, count, true, BEAGLE_OP_NONE, true,
                                       partialsType & BEAGLE_PARTIALS_TOP));

    if (kThreadingEnabled) {
        returnCode = upPrePartialsByPartitionAsync(operations,
                                                   count,
                                                   partialsType);
    } else {
        bool byPartition = true;
        returnCode = upPrePartials(byPartition,
                                  operations,
                                  count,
                                  BEAGLE_OP_NONE,
                                  partialsType);
    }

    return returnCode;
}

BEAGLE_CPU_TEMPLATE
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::autoPartitionPartialsOperations(const int* operations,
                                                                        int* partitionOperations,
                                                                        int count,
                                                                        int cumulativeScaleIndex) {

    int numOps  = BEAGLE_OP_COUNT;
    int numOpsP = BEAGLE_PARTITION_OP_COUNT;

    for (int i=0; i<count; i++) {
        for (int j=0; j<kPartitionCount; j++) {
            for (int k=0; k<numOps; k++) {
                partitionOperations[i*kPartitionCount*numOpsP + j*numOpsP + k] = operations[i*numOps + k];
            }
                partitionOperations[i*kPartitionCount*numOpsP + j*numOpsP + numOps    ] = j;
                partitionOperations[i*kPartitionCount*numOpsP + j*numOpsP + numOps + 1] = cumulativeScaleIndex;
        }
    }
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::upPartialsByPartitionAsync(const int* operations,
                                                                  int count) {

    int numOps = BEAGLE_PARTITION_OP_COUNT;

    memset(gThreadOpCounts, 0, sizeof(int) * kNumThreads);

    for (int i=0; i<count; i++) {
        int t = operations[i * numOps + 7] % kNumThreads;
        for (int j=0; j<numOps; j++) {
            gThreadOperations[t][gThreadOpCounts[t]*numOps + j] = operations[i*numOps + j];
        }
        gThreadOpCounts[t]++;
    }

    for (int i=0; i<kNumThreads; i++) {
        auto b1658 = std::bind(&BeagleCPUImpl<BEAGLE_CPU_GENERIC>::upPartials, this,
                      true,
                      (const int*) gThreadOperations[i],
                      gThreadOpCounts[i],
                      BEAGLE_OP_NONE,
                      BEAGLE_PARTIALS_BOTTOM);
        std::packaged_task<void()> threadTask([b = std::move(b1658)]() mutable { b(); });

        gFutures[i] = threadTask.get_future();
        threadData* td = &gThreads[i];

        std::unique_lock<std::mutex> l(td->m);
        td->jobs.push(std::move(threadTask));
        l.unlock();

        gThreads[i].cv.notify_one();
    }

    for (int i=0; i<kNumThreads; i++) {
        gFutures[i].wait();
    }

    return BEAGLE_SUCCESS;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::upPrePartialsByPartitionAsync(const int* operations,
                                                                     int count,
                                                                     BeaglePartialsType partialsType) {

    int numOps = BEAGLE_PARTITION_OP_COUNT;

    memset(gThreadOpCounts, 0, sizeof(int) * kNumThreads);

    for (int i=0; i<count; i++) {
        int t = operations[i * numOps + 7] % kNumThreads;
        for (int j=0; j<numOps; j++) {
            gThreadOperations[t][gThreadOpCounts[t]*numOps + j] = operations[i*numOps + j];
        }
        gThreadOpCounts[t]++;
    }

    for (int i=0; i<kNumThreads; i++) {
        auto b1700 = std::bind(&BeagleCPUImpl<BEAGLE_CPU_GENERIC>::upPrePartials, this,
                          true,
                          (const int*) gThreadOperations[i],
                          gThreadOpCounts[i],
                          BEAGLE_OP_NONE,
                          partialsType);
        std::packaged_task<void()> threadTask([b = std::move(b1700)]() mutable { b(); });

        gFutures[i] = threadTask.get_future();
        threadData* td = &gThreads[i];

        std::unique_lock<std::mutex> l(td->m);
        td->jobs.push(std::move(threadTask));
        l.unlock();

        gThreads[i].cv.notify_one();
    }

    for (int i=0; i<kNumThreads; i++) {
        gFutures[i].wait();
    }

    return BEAGLE_SUCCESS;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::upPartials(bool byPartition,
                                                  const int* operations,
                                                  int count,
                                                  int cumulativeScaleIndex,
                                                  BeaglePartialsType partialsType) {

    REALTYPE* cumulativeScaleBuffer = NULL;
    if (cumulativeScaleIndex != BEAGLE_OP_NONE)
        cumulativeScaleBuffer = gScaleBuffers[cumulativeScaleIndex];

    for (int op = 0; op < count; op++) {

        int numOps = BEAGLE_OP_COUNT;
        if (byPartition)
            numOps = BEAGLE_PARTITION_OP_COUNT;

        if (DEBUGGING_OUTPUT) {
            fprintf(stderr, "op[%d] = ", op);
            for (int j = 0; j < numOps; j++) {
                std::cerr << operations[op*numOps+j] << " ";
            }
            fprintf(stderr, "\n");
        }

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
            cumulativeScaleIndex = operations[op * numOps + 8];
            if (cumulativeScaleIndex != BEAGLE_OP_NONE)
                cumulativeScaleBuffer = gScaleBuffers[cumulativeScaleIndex];
            else
                cumulativeScaleBuffer = NULL;
        }

        // a second child of BEAGLE_OP_NONE makes this a degree-2 (single-child) node
        const bool singleChild = (child2Index == BEAGLE_OP_NONE);

        const REALTYPE* partials1 = gPartials[child1Index];
        const REALTYPE* partials2 = singleChild ? NULL : gPartials[child2Index];

        const int* tipStates1 = gTipStates[child1Index];
        const int* tipStates2 = singleChild ? NULL : gTipStates[child2Index];

        const REALTYPE* matrices1 = gTransitionMatrices[child1TransMatIndex];
        const REALTYPE* matrices2 = singleChild ? NULL : gTransitionMatrices[child2TransMatIndex];

        REALTYPE* destPartials = gPartials[parIndex];

        int startPattern = 0;
        int endPattern = kPatternCount;
        if (byPartition) {
            startPattern = gPatternPartitionsStartPatterns[currentPartition];
            endPattern = gPatternPartitionsStartPatterns[currentPartition + 1];
        }

        int rescale = BEAGLE_OP_NONE;
        REALTYPE* scalingFactors = NULL;

        if (kFlags & BEAGLE_FLAG_SCALING_AUTO) {
            gActiveScalingFactors[parIndex - kTipCount] = 0;
            if (tipStates1 == 0 && tipStates2 == 0)
                rescale = 2;
        } else if (kFlags & BEAGLE_FLAG_SCALING_ALWAYS) {
            rescale = 1;
            scalingFactors = gScaleBuffers[parIndex - kTipCount];
        } else if (kFlags & BEAGLE_FLAG_SCALING_DYNAMIC) { // TODO: this is a quick and dirty implementation just so it returns correct results
            if (tipStates1 == 0 && tipStates2 == 0) {
                rescale = 1;
                removeScaleFactors(&readScalingIndex, 1, cumulativeScaleIndex);
                scalingFactors = gScaleBuffers[writeScalingIndex];
            }
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

        if (singleChild) {
            // auto-scaling (rescale == 2) is not applied to a single child, which is left unscaled
            const REALTYPE* fixedScaleFactors = (rescale == 0) ? scalingFactors : NULL;
            if (tipStates1 != NULL) {
                calcStatesDegree2(destPartials, tipStates1, matrices1, fixedScaleFactors, startPattern, endPattern);
            } else {
                calcPartialsDegree2(destPartials, partials1, matrices1, fixedScaleFactors, startPattern, endPattern);
            }
            if (rescale == 1) { // Recompute scaleFactors
                if (byPartition) {
                    rescalePartialsByPartition(destPartials,scalingFactors,cumulativeScaleBuffer,0, currentPartition);
                } else {
                    rescalePartials(destPartials,scalingFactors,cumulativeScaleBuffer,0);
                }
            }
        } else if (tipStates1 != NULL) {
            if (tipStates2 != NULL ) {
                if (rescale == 0) { // Use fixed scaleFactors
                    calcStatesStatesFixedScaling(destPartials, tipStates1, matrices1, tipStates2,
                                                 matrices2, scalingFactors, startPattern, endPattern);
                } else {
                    // First compute without any scaling
                    calcStatesStates(destPartials, tipStates1, matrices1, tipStates2, matrices2,
                                     startPattern, endPattern);
                    if (rescale == 1) { // Recompute scaleFactors
                        if (byPartition) {
                            rescalePartialsByPartition(destPartials,scalingFactors,cumulativeScaleBuffer,0, currentPartition);
                        } else {
                            rescalePartials(destPartials,scalingFactors,cumulativeScaleBuffer,0);
                        }
                    }
                }
            } else {
                if (rescale == 0) {
                    calcStatesPartialsFixedScaling(destPartials, tipStates1, matrices1, partials2,
                                                   matrices2, scalingFactors, startPattern, endPattern);
                } else {
                    calcStatesPartials(destPartials, tipStates1, matrices1, partials2, matrices2,
                                       startPattern, endPattern);
                    if (rescale == 1) { // Recompute scaleFactors
                        if (byPartition) {
                            rescalePartialsByPartition(destPartials,scalingFactors,cumulativeScaleBuffer,0, currentPartition);
                        } else {
                            rescalePartials(destPartials,scalingFactors,cumulativeScaleBuffer,0);
                        }
                    }
                }
            }
        } else {
            if (tipStates2 != NULL) {
                if (rescale == 0) {
                    calcStatesPartialsFixedScaling(destPartials,tipStates2,matrices2,partials1,matrices1,
                                                   scalingFactors, startPattern, endPattern);
                } else {
                    calcStatesPartials(destPartials, tipStates2, matrices2, partials1, matrices1,
                                       startPattern, endPattern);
                    if (rescale == 1) {// Recompute scaleFactors
                        if (byPartition) {
                            rescalePartialsByPartition(destPartials,scalingFactors,cumulativeScaleBuffer,0, currentPartition);
                        } else {
                            rescalePartials(destPartials,scalingFactors,cumulativeScaleBuffer,0);
                        }
                    }
                }
            } else {
                if (rescale == 2) {
                    int sIndex = parIndex - kTipCount;
                    calcPartialsPartialsAutoScaling(destPartials,partials1,matrices1,partials2,matrices2,
                                                     &gActiveScalingFactors[sIndex]);
                    if (gActiveScalingFactors[sIndex])
                        autoRescalePartials(destPartials, gAutoScaleBuffers[sIndex]);

                } else if (rescale == 0) {
                    calcPartialsPartialsFixedScaling(destPartials,partials1,matrices1,partials2,
                                                     matrices2,scalingFactors,startPattern,endPattern);
                } else {
                    calcPartialsPartials(destPartials, partials1, matrices1, partials2, matrices2,
                                         startPattern, endPattern);
                    if (rescale == 1) {// Recompute scaleFactors
                        if (byPartition) {
                            rescalePartialsByPartition(destPartials,scalingFactors,cumulativeScaleBuffer,0, currentPartition);
                        } else {
                            rescalePartials(destPartials,scalingFactors,cumulativeScaleBuffer,0);
                        }
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

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::upPrePartials(bool byPartition,
                                                     const int* operations,
                                                     int count,
                                                     int cumulativeScaleIndex,
                                                     BeaglePartialsType partialsType) {
    if (partialsType & BEAGLE_PARTIALS_TOP) {
        return upPrePartialsTop(byPartition, operations, count, cumulativeScaleIndex);
    } else {
        return upPrePartialsBottom(byPartition, operations, count, cumulativeScaleIndex);
    }
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::upPrePartialsBottom(bool byPartition,
                                                           const int* operations,
                                                           int count,
                                                           int cumulativeScaleIndex) {

    REALTYPE* cumulativeScaleBuffer = NULL;  // don't need to normalize/transform back preOrderPartials, off by constant rescaling factor is fine
//    if (cumulativeScaleIndex != BEAGLE_OP_NONE)
//        cumulativeScaleBuffer = gScaleBuffers[cumulativeScaleIndex];

    for (int op = 0; op < count; op++) {

        int numOps = BEAGLE_OP_COUNT;
        if (byPartition)
            numOps = BEAGLE_PARTITION_OP_COUNT;

        if (DEBUGGING_OUTPUT) {
            fprintf(stderr, "op[%d] = ", op);
            for (int j = 0; j < numOps; j++) {
                std::cerr << operations[op*numOps+j] << " ";
            }
            fprintf(stderr, "\n");
        }

        // create a list of partial likelihood update operations
        // the order is [dest, destScaling, source1, matrix1, source2, matrix2]
        // destPartials point to the pre-order partials
        // partials1 = pre-order partials of the parent node
        // matrices1 = Ptr matrices of the current node (to the parent node)
        // partials2 = post-order partials of the sibling node
        // matrices2 = Ptr matrices of the sibling node (to the parent node)
        const int parIndex = operations[op * numOps];
        const int writeScalingIndex = operations[op * numOps + 1];
        const int readScalingIndex = operations[op * numOps + 2];
        const int parentIndex = operations[op * numOps + 3];
        const int parentTransMatIndex = operations[op * numOps + 4];
        const int siblingIndex = operations[op * numOps + 5];
        const int siblingTransMatIndex = operations[op * numOps + 6];
        int currentPartition = 0;
        if (byPartition) {
            currentPartition = operations[op * numOps + 7];
            cumulativeScaleIndex = operations[op * numOps + 8];
//            if (cumulativeScaleIndex != BEAGLE_OP_NONE)
//                cumulativeScaleBuffer = gScaleBuffers[cumulativeScaleIndex];
//            else
//                cumulativeScaleBuffer = NULL;
        }


        // a sibling of BEAGLE_OP_NONE makes the parent a degree-2 (single-child) node
        const bool singleChild = (siblingIndex == BEAGLE_OP_NONE);

        /// non-root nodes, can be a tip
        const REALTYPE *partials1 = gPartials[parentIndex];
        const REALTYPE *partials2 = singleChild ? NULL : gPartials[siblingIndex];

        const int *tipStates2 = singleChild ? NULL : gTipStates[siblingIndex];

        const REALTYPE *matrices1 = gTransitionMatrices[parentTransMatIndex];
        const REALTYPE *matrices2 = singleChild ? NULL : gTransitionMatrices[siblingTransMatIndex];


        REALTYPE *destPartials = gPartials[parIndex];

        int startPattern = 0;
        int endPattern = kPatternCount;
        if (byPartition) {
            startPattern = gPatternPartitionsStartPatterns[currentPartition];
            endPattern = gPatternPartitionsStartPatterns[currentPartition + 1];
        }

        int rescale = BEAGLE_OP_NONE;
        REALTYPE *scalingFactors = NULL;

        if (kFlags & BEAGLE_FLAG_SCALING_AUTO) {
            gActiveScalingFactors[parIndex - kTipCount] = 0;
//            if (siblingStates == 0)
                rescale = 2;
        } else if (kFlags & BEAGLE_FLAG_SCALING_ALWAYS) {
            rescale = 1;
            scalingFactors = gScaleBuffers[parIndex - kTipCount];
        } else if (kFlags &
                   BEAGLE_FLAG_SCALING_DYNAMIC) { // TODO: this is a quick and dirty implementation just so it returns correct results
//            if (siblingStates == 0) {
                rescale = 1;
                removeScaleFactors(&readScalingIndex, 1, cumulativeScaleIndex);
                scalingFactors = gScaleBuffers[writeScalingIndex];
//            }
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

        /// destPartials point to the pre-order partials
        /// partials1 = pre-order partials of the parent node
        /// matrices1 = Ptr matrices of the current node (to the parent node)
        /// partials2 = post-order partials of the sibling node
        /// matrices2 = Ptr matrices of the sibling node (to the parent node)
        /// comment out all conditions that's not implemented

        if (singleChild) {
            calcDegree2PrePartials(destPartials, partials1, matrices1, startPattern, endPattern);

            if (rescale == 1) {// Recompute scaleFactors
                if (byPartition) {
                    rescalePartialsByPartition(destPartials, scalingFactors, cumulativeScaleBuffer, 0,
                                               currentPartition);
                } else {
                    rescalePartials(destPartials, scalingFactors, cumulativeScaleBuffer, 0);
                }
            }

        } else if (tipStates2 != NULL) {
            calcPrePartialsStates(destPartials, partials1, matrices1, tipStates2, matrices2,
                                  startPattern, endPattern);

            if (rescale == 1) {// Recompute scaleFactors
                if (byPartition) {
                    rescalePartialsByPartition(destPartials, scalingFactors, cumulativeScaleBuffer, 0,
                                               currentPartition);
                } else {
                    rescalePartials(destPartials, scalingFactors, cumulativeScaleBuffer, 0);
                }
            }

        } else {
//            if (rescale == 2) {
//                //                    int sIndex = parIndex - kTipCount;
//                //                    calcPartialsPartialsAutoScaling(destPartials,partials1,matrices1,partials2,matrices2,
//                //                                                    &gActiveScalingFactors[sIndex]);
//                //                    if (gActiveScalingFactors[sIndex])
//                //                        autoRescalePartials(destPartials, gAutoScaleBuffers[sIndex]);
//
//            } else if (rescale == 0) {
//                //                    calcPartialsPartialsFixedScaling(destPartials,partials1,matrices1,partials2,
//                //                                                     matrices2,scalingFactors,startPattern,endPattern);
//            } else {

                calcPrePartialsPartials(destPartials, partials1, matrices1, partials2, matrices2,
                                        startPattern, endPattern);

                if (rescale == 1) {// Recompute scaleFactors
                    if (byPartition) {
                        rescalePartialsByPartition(destPartials, scalingFactors, cumulativeScaleBuffer, 0,
                                                   currentPartition);
                    } else {
                        rescalePartials(destPartials, scalingFactors, cumulativeScaleBuffer, 0);
                    }
                }
//            }
        }


//        if (kFlags & BEAGLE_FLAG_SCALING_ALWAYS) {
//            int parScalingIndex = parIndex - kTipCount;
//            int child1ScalingIndex = parentIndex - kTipCount;
//            int child2ScalingIndex = siblingIndex - kTipCount;
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

        if (DEBUGGING_OUTPUT) {
            if (scalingFactors != NULL && rescale == 0) {
                for (int i = 0; i < kPatternCount; i++)
                    fprintf(stderr, "old scaleFactor[%d] = %.5f\n", i, scalingFactors[i]);
            }
            fprintf(stderr, "Result partials:\n");
            for (int i = 0; i < kPartialsSize; i++)
                fprintf(stderr, "destP[%d] = %.5f\n", i, destPartials[i]);
        }
    }

    return BEAGLE_SUCCESS;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::upPrePartialsTop(bool byPartition,
                                                        const int* operations,
                                                        int count,
                                                        int cumulativeScaleIndex) {

    for (int op = 0; op < count; op++) {

        int numOps = BEAGLE_OP_COUNT;
        if (byPartition) {
            numOps = BEAGLE_PARTITION_OP_COUNT;
        }

        if (DEBUGGING_OUTPUT) {
            fprintf(stderr, "op[%d] = ", op);
            for (int j = 0; j < numOps; j++) {
                std::cerr << operations[op*numOps+j] << " ";
            }
            fprintf(stderr, "\n");
        }

        // create a list of partial likelihood update operations
        // the order is [dest, destScaling, source1, matrix1, source2, matrix2]
        // destPartials point to the pre-order partials
        // partials1 = pre-order partials of the parent node
        // matrices1 = Ptr matrices of the current node (to the parent node)
        // partials2 = post-order partials of the sibling node
        // matrices2 = Ptr matrices of the sibling node (to the parent node)
        const int parIndex = operations[op * numOps];
        const int writeScalingIndex = operations[op * numOps + 1];
        const int readScalingIndex = operations[op * numOps + 2];
        const int parentIndex = operations[op * numOps + 3];
        const int parentTransMatIndex = operations[op * numOps + 4];
        const int siblingIndex = operations[op * numOps + 5];
        const int siblingTransMatIndex = operations[op * numOps + 6];
        int currentPartition = 0;
        if (byPartition) {
            currentPartition = operations[op * numOps + 7];
        }

        // a sibling of BEAGLE_OP_NONE makes the parent a degree-2 (single-child) node
        const bool singleChild = (siblingIndex == BEAGLE_OP_NONE);

        /// non-root nodes, can be a tip
        const REALTYPE *partials1 = gPartials[parentIndex];
        const REALTYPE *partials2 = singleChild ? NULL : gPartials[siblingIndex];

        const int *tipStates2 = singleChild ? NULL : gTipStates[siblingIndex];

        const REALTYPE *matrices1 = (parentTransMatIndex >= 0) ? gTransitionMatrices[parentTransMatIndex] : NULL;
        const REALTYPE *matrices2 = singleChild ? NULL : gTransitionMatrices[siblingTransMatIndex];

        REALTYPE *destPartials = gPartials[parIndex];

        int startPattern = 0;
        int endPattern = kPatternCount;
        if (byPartition) {
            startPattern = gPatternPartitionsStartPatterns[currentPartition];
            endPattern = gPatternPartitionsStartPatterns[currentPartition + 1];
        }

        int rescale = BEAGLE_OP_NONE;

        if (singleChild) {
            calcDegree2PrePartials(destPartials, partials1, matrices1, startPattern, endPattern);
        } else if (tipStates2 != NULL) {
            if (matrices1 == NULL) { // Parent node is root
                calcPrePartialsStatesTopRoot(destPartials, partials1, matrices1, tipStates2, matrices2,
                                              startPattern, endPattern);
            } else {
                calcPrePartialsStatesTop(destPartials, partials1, matrices1, tipStates2, matrices2,
                                         startPattern, endPattern);
            }
        } else {
            if (matrices1 == NULL) { // Parent node is root
                calcPrePartialsPartialsTopRoot(destPartials, partials1, matrices1, partials2, matrices2,
                                               startPattern, endPattern);
            } else {
                calcPrePartialsPartialsTop(destPartials, partials1, matrices1, partials2, matrices2,
                                           startPattern, endPattern);
            }
        }
    }

    return BEAGLE_SUCCESS;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcCrossProducts(const int *postBufferIndices,
                                                         const int *preBufferIndices,
                                                         const int *categoryRatesIndices,
                                                         const int *categoryWeightsIndices,
                                                         const double *edgeLengths,
                                                         int count,
                                                         double *outSumDerivatives,
                                                         double *outSumSquaredDerivatives) {

    int returnCode = BEAGLE_SUCCESS;

    const int secondDerivativeIndex = BEAGLE_OP_NONE;
    const double *categoryRates = gCategoryRates[categoryRatesIndices[0]]; // TODO Generalize
    const REALTYPE *categoryWeights = gCategoryWeights[categoryWeightsIndices[0]]; // TODO Generalize

    // TODO remove unnecessary? allocation
    // if (crossProductNumeratorTmp == nullptr) {
    //     crossProductNumeratorTmp = (REALTYPE*) mallocAligned(sizeof(REALTYPE) * kPaddedPatternCount
    //             * kStateCount * kStateCount);
    // }

//    std::fill(outCrossProducts, outCrossProducts + kStateCount * kStateCount, 0.0); // TODO Remove

    for (int nodeNum = 0; nodeNum < count; nodeNum++) {

        const double edgeLength = edgeLengths[nodeNum];

        const REALTYPE *preOrderPartial = gPartials[preBufferIndices[nodeNum]];
        const int *tipStates = gTipStates[postBufferIndices[nodeNum]];

//        const int firstDerivativeIndex = firstDerivativeIndices[nodeNum];
        const int scalingFactorsIndex = -1; // cumulativeScaleIndices[nodeNum];

        const int patternOffset = nodeNum * kPatternCount;
//        double* outDerivativesForNode = (outDerivatives == NULL) ?
//                NULL : outDerivatives + patternOffset;
//        double* outSumDerivativesForNode = (outSumDerivatives == NULL) ?
//                NULL : outSumDerivatives + nodeNum;
//        double* outSumSquaredDerivativesForNode = (outSumSquaredDerivatives == NULL) ?
//                NULL : outSumSquaredDerivatives + nodeNum;
//
//        resetDerivativeTemporaries();
//
        if (tipStates != NULL) {

            calcCrossProductsStates(tipStates, preOrderPartial,
                                    categoryRates,
                                    categoryWeights,
                                    edgeLength,
                                    outSumDerivatives,
                                    outSumSquaredDerivatives);

        } else {

            const REALTYPE *postOrderPartial = gPartials[postBufferIndices[nodeNum]];
            calcCrossProductsPartials(postOrderPartial, preOrderPartial,
                                      categoryRates,
                                      categoryWeights,
                                      edgeLength,
                                      outSumDerivatives,
                                      outSumSquaredDerivatives);
        }
//
//        accumulateDerivatives(outDerivativesForNode,
//                outSumDerivativesForNode,
//                outSumSquaredDerivativesForNode);

//        std::cout << "A: " << outCrossProducts[0] << std::endl;
//        std::fill(outCrossProducts, outCrossProducts + kStateCount * kStateCount, 0.0); // TODO Remove

    }

    return returnCode;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcEdgeLogDerivativesByAutoPartitionAsync(const int *operations,
                                                                                  int count,
                                                                                  double *outDerivatives,
                                                                                  double *outSumDerivatives,
                                                                                  double *outSumSquaredDerivatives) {

    int returnCode = BEAGLE_SUCCESS;

    int numOps  = BEAGLE_OP_COUNT;
    int numOpsP = BEAGLE_PARTITION_OP_COUNT;

    const int secondDerivativeIndex = BEAGLE_OP_NONE;
    const double *categoryRates = NULL; // gCategoryRates[categoryRatesIndices[0]]; // TODO Generalize
    const REALTYPE *categoryWeights = gCategoryWeights[operations[3]]; // TODO Generalize



    for (int nodeNum = 0; nodeNum < count; nodeNum++) {

        const REALTYPE *preOrderPartial = gPartials[operations[nodeNum * numOpsP + 1]];
        const int *tipStates = gTipStates[operations[nodeNum * numOpsP]];

        const int firstDerivativeIndex = operations[nodeNum * numOpsP + 2];
        const int scalingFactorsIndex = -1; // cumulativeScaleIndices[nodeNum];

        const int patternOffset = operations[nodeNum * numOpsP + 4] * kPatternCount;
        const int threadOffset = operations[nodeNum * numOpsP + 5] * ((kPaddedPatternCount + 1) / 2 * 2);


#ifdef BEAGLE_DEBUG_FLOW
        std::cerr<<"Job = " << operations[nodeNum * numOpsP + 4] <<std::endl;
        std::cerr<<"Post index = " << operations[nodeNum * numOpsP] <<std::endl;
        std::cerr<<"Pre index = " << operations[nodeNum * numOpsP + 1] <<std::endl;
        std::cerr<<"First Derivative index = " << operations[nodeNum * numOpsP + 2] <<std::endl;
#endif



        double* outDerivativesForNode = (outDerivatives == NULL) ?
                                        NULL : outDerivatives + patternOffset;
        double* outSumDerivativesForNode = (outSumDerivatives == NULL) ?
                                           NULL : outSumDerivatives + operations[nodeNum * numOpsP + 4];
        double* outSumSquaredDerivativesForNode = (outSumSquaredDerivatives == NULL) ?
                                                  NULL : outSumSquaredDerivatives + operations[nodeNum * numOpsP + 4];

        resetDerivativeTemporaries(threadOffset);

#ifdef BEAGLE_DEBUG_FLOW
        std::cerr<<"Almost almost finished Job = " << operations[nodeNum * numOpsP + 4] <<std::endl;
#endif

        if (tipStates != NULL) {

            calcEdgeLogDerivativesStates(tipStates, preOrderPartial, firstDerivativeIndex,
                                         secondDerivativeIndex, categoryRates, categoryWeights,
                                         outDerivativesForNode,
                                         outSumDerivativesForNode,
                                         outSumSquaredDerivativesForNode, threadOffset);

        } else {

            const REALTYPE *postOrderPartial = gPartials[operations[nodeNum * numOpsP]];
            calcEdgeLogDerivativesPartials(postOrderPartial, preOrderPartial, firstDerivativeIndex,
                                           secondDerivativeIndex, categoryRates, categoryWeights,
                                           scalingFactorsIndex,
                                           outDerivativesForNode,
                                           outSumDerivativesForNode,
                                           outSumSquaredDerivativesForNode, threadOffset);
        }

#ifdef BEAGLE_DEBUG_FLOW
        std::cerr<<"Almost finished Job = " << operations[nodeNum * numOpsP + 4] <<std::endl;
#endif

        accumulateDerivatives(outDerivativesForNode,
                              outSumDerivativesForNode,
                              outSumSquaredDerivativesForNode, threadOffset);

#ifdef BEAGLE_DEBUG_FLOW
        std::cerr<<"Finished Job = " << operations[nodeNum * numOpsP + 4] << ", current / count = " << nodeNum << "/" << count <<std::endl;
#endif

    }

#ifdef BEAGLE_DEBUG_FLOW
    std::cerr<<"Reached return!" <<std::endl;
#endif

    return returnCode;
}



BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcEdgeLogDerivatives(const int *postBufferIndices,
                                                              const int *preBufferIndices,
                                                              const int *firstDerivativeIndices,
                                                              const int *secondDerivativeIndices,
                                                              const int *categoryWeightsIndices,
                                                              const int *categoryRatesIndices,
                                                              const int *cumulativeScaleIndices,
                                                              int count,
                                                              double *outDerivatives,
                                                              double *outSumDerivatives,
                                                              double *outSumSquaredDerivatives) {

    int returnCode = BEAGLE_SUCCESS;

    const int secondDerivativeIndex = BEAGLE_OP_NONE;
    const double *categoryRates = NULL; // gCategoryRates[categoryRatesIndices[0]]; // TODO Generalize
    const REALTYPE *categoryWeights = gCategoryWeights[categoryWeightsIndices[0]]; // TODO Generalize

    for (int nodeNum = 0; nodeNum < count; nodeNum++) {

        const REALTYPE *preOrderPartial = gPartials[preBufferIndices[nodeNum]];
        const int *tipStates = gTipStates[postBufferIndices[nodeNum]];

        const int firstDerivativeIndex = firstDerivativeIndices[nodeNum];
        const int scalingFactorsIndex = -1; // cumulativeScaleIndices[nodeNum];

        const int patternOffset = nodeNum * kPatternCount;
        double* outDerivativesForNode = (outDerivatives == NULL) ?
                NULL : outDerivatives + patternOffset;
        double* outSumDerivativesForNode = (outSumDerivatives == NULL) ?
                NULL : outSumDerivatives + nodeNum;
        double* outSumSquaredDerivativesForNode = (outSumSquaredDerivatives == NULL) ?
                NULL : outSumSquaredDerivatives + nodeNum;

        resetDerivativeTemporaries(0);

        if (tipStates != NULL) {

            calcEdgeLogDerivativesStates(tipStates, preOrderPartial, firstDerivativeIndex,
                                         secondDerivativeIndex, categoryRates, categoryWeights,
                                         outDerivativesForNode,
                                         outSumDerivativesForNode,
                                         outSumSquaredDerivativesForNode, 0);

        } else {

            const REALTYPE *postOrderPartial = gPartials[postBufferIndices[nodeNum]];
            calcEdgeLogDerivativesPartials(postOrderPartial, preOrderPartial, firstDerivativeIndex,
                                           secondDerivativeIndex, categoryRates, categoryWeights,
                                           scalingFactorsIndex,
                                           outDerivativesForNode,
                                           outSumDerivativesForNode,
                                           outSumSquaredDerivativesForNode, 0);
        }

        accumulateDerivatives(outDerivativesForNode,
                              outSumDerivativesForNode,
                              outSumSquaredDerivativesForNode, 0);

    }

    return returnCode;
}

BEAGLE_CPU_TEMPLATE template <bool DoDerivatives, bool DoSum, bool DoSumSquared>
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::accumulateDerivativesImpl(double *outDerivatives,
                                                                  double *outSumDerivatives,
                                                                  double *outSumSquaredDerivatives,
                                                                  int offset) {

    REALTYPE sum = 0.0;
    REALTYPE sumSquared = 0.0;

    for (int k = 0; k < kPatternCount; k++) {
        REALTYPE derivative = grandNumeratorDerivTmp[offset + k] / grandDenominatorDerivTmp[offset + k];
        if (DoDerivatives) {
            outDerivatives[k] = derivative;
        }
        if constexpr (DoSum) {
            sum += derivative * static_cast<REALTYPE>(gPatternWeights[k]);
        }
        if constexpr (DoSumSquared) {
            sumSquared += derivative * derivative * static_cast<REALTYPE>(gPatternWeights[k]);
        }
    }

    if (DoSum) {
        *outSumDerivatives = sum;
    }

    if (DoSumSquared) {
        *outSumSquaredDerivatives = sumSquared;
    }
}

BEAGLE_CPU_TEMPLATE template <bool DoDerivatives, bool DoSum>
void
BeagleCPUImpl<BEAGLE_CPU_GENERIC>::accumulateDerivativesDispatch2(double *outDerivatives,
                                                                  double *outSumDerivatives,
                                                                  double *outSumSquaredDerivatives,
                                                                  int offset) {

    if (outSumSquaredDerivatives == NULL) {
        accumulateDerivativesImpl<DoDerivatives, DoSum, false>(
                outDerivatives, outSumDerivatives, outSumSquaredDerivatives, offset);
    } else {
        accumulateDerivativesImpl<DoDerivatives, DoSum, true>(
                outDerivatives, outSumDerivatives, outSumSquaredDerivatives, offset);
    }
}

BEAGLE_CPU_TEMPLATE template <bool DoDerivatives>
void
BeagleCPUImpl<BEAGLE_CPU_GENERIC>::accumulateDerivativesDispatch1(double *outDerivatives,
                                                                  double *outSumDerivatives,
                                                                  double *outSumSquaredDerivatives,
                                                                  int offset) {

    if (outSumDerivatives == NULL) {
        accumulateDerivativesDispatch2<DoDerivatives, false>(
                outDerivatives, outSumDerivatives, outSumSquaredDerivatives, offset);
    } else {
        accumulateDerivativesDispatch2<DoDerivatives, true>(
                outDerivatives, outSumDerivatives, outSumSquaredDerivatives, offset);
    }
}

BEAGLE_CPU_TEMPLATE
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::resetDerivativeTemporaries(int offset) {
        std::fill(grandNumeratorDerivTmp + offset, grandNumeratorDerivTmp + offset + kPaddedPatternCount, 0);
        std::fill(grandDenominatorDerivTmp + offset, grandDenominatorDerivTmp + offset + kPaddedPatternCount, 0);
}


BEAGLE_CPU_TEMPLATE
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::accumulateDerivatives(double *outDerivatives,
                                                              double *outSumDerivatives,
                                                              double *outSumSquaredDerivatives,
                                                              int offset) {
    if (outDerivatives == NULL) {
        accumulateDerivativesDispatch1<false>(
                outDerivatives, outSumDerivatives, outSumSquaredDerivatives, offset);
    } else {
        accumulateDerivativesDispatch1<true>(
                outDerivatives, outSumDerivatives, outSumSquaredDerivatives, offset);
    }
}

BEAGLE_CPU_TEMPLATE
void
BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcEdgeLogDerivativesStates(const int *tipStates, const REALTYPE *preOrderPartial,
                                                                const int firstDerivativeIndex,
                                                                const int secondDerivativeIndex,
                                                                const double *categoryRates,
                                                                const REALTYPE *categoryWeights,
                                                                double *outDerivatives,
                                                                double *outSumDerivatives,
                                                                double *outSumSquaredDerivatives,
                                                                int cacheOffset) {

    const REALTYPE *firstDerivMatrix = gTransitionMatrices[firstDerivativeIndex];

    for (int category = 0; category < kCategoryCount; category++) {

        for (int pattern = 0; pattern < kPatternCount; pattern++) {

            const int patternIndex = category * kPatternCount + pattern;
            const int state = tipStates[pattern];

            REALTYPE numerator = 0.0;
            REALTYPE denominator = preOrderPartial[patternIndex * kPartialsPaddedStateCount + (state % kStateCount)];
            // TODO (state % kStateCount) is not correct; should imply missing character
            // TODO See calcCrossProductsStates() for possible solution

            for (int k = 0; k < kStateCount; k++) {
                numerator += firstDerivMatrix[category * kMatrixSize + k * kTransPaddedStateCount + state] *
                             preOrderPartial[patternIndex * kPartialsPaddedStateCount + k];
            }

            grandNumeratorDerivTmp[cacheOffset + pattern] += categoryWeights[category] * numerator;
            grandDenominatorDerivTmp[cacheOffset + pattern] += categoryWeights[category] * denominator;
        }
    }
}

BEAGLE_CPU_TEMPLATE
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcCrossProductsStates(const int *tipStates,
                                                                const REALTYPE *preOrderPartial,
                                                                const double *categoryRates,
                                                                const REALTYPE *categoryWeights,
                                                                const double edgeLength,
                                                                double *outCrossProducts,
                                                                double *outSumSquaredDerivatives) {

    for (int pattern = 0; pattern < kPatternCount; pattern++) {

        std::vector<REALTYPE> tmp(kStateCount * kStateCount, 0.0); // TODO Handle temporary memory better

        REALTYPE patternDenominator = 0.0;

        const int state = tipStates[pattern];

        if (state < kStateCount) {

            for (int category = 0; category < kCategoryCount; category++) {

                const REALTYPE scale = (REALTYPE) categoryRates[category] * edgeLength;

                const REALTYPE weight = categoryWeights[category];
                const int patternIndex = category * kPatternCount + pattern;
                const int v = patternIndex * kPartialsPaddedStateCount;

                REALTYPE denominator = preOrderPartial[v + state];
                patternDenominator += denominator * weight;

                for (int k = 0; k < kStateCount; k++) {
                    tmp[k * kStateCount + state] += preOrderPartial[v + k] * weight * scale;
                }
            }

            const auto patternWeight = gPatternWeights[pattern] / patternDenominator;
            for (int k = 0; k < kStateCount; k++) {
                outCrossProducts[k * kStateCount + state] += tmp[k * kStateCount + state] * patternWeight;
            }

        } else { // Missing character

            for (int category = 0; category < kCategoryCount; category++) {

                const REALTYPE scale = (REALTYPE) categoryRates[category] * edgeLength;

                const REALTYPE weight = categoryWeights[category];
                const int patternIndex = category * kPatternCount + pattern;
                const int v = patternIndex * kPartialsPaddedStateCount;

                REALTYPE denominator = 0.0;
                for (int k = 0; k < kStateCount; k++) {
                    denominator += preOrderPartial[v + k];
                }
                patternDenominator += denominator * weight;

                for (int k = 0; k < kStateCount; k++) {
                    for (int j = 0; j < kStateCount; j++) {
                        tmp[k * kStateCount + j] += preOrderPartial[v + k] * weight * scale;
                    }
                }
            }

            const auto patternWeight = gPatternWeights[pattern] / patternDenominator;
            for (int k = 0; k < kStateCount; k++) {
                for (int j = 0; j < kStateCount; j++) {
                    outCrossProducts[k * kStateCount + j] += tmp[k * kStateCount + j] * patternWeight;
                }
            }
        }
    }
}

BEAGLE_CPU_TEMPLATE
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcCrossProductsPartials(const REALTYPE *postOrderPartial,
                                                                  const REALTYPE *preOrderPartial,
                                                                  const double *categoryRates,
                                                                  const REALTYPE *categoryWeights,
                                                                  const double edgeLength,
                                                                  double *outCrossProducts,
                                                                  double *outSumSquaredDerivatives) {

    for (int pattern = 0; pattern < kPatternCount; pattern++) {

        std::vector<REALTYPE> tmp(kStateCount * kStateCount, 0.0); // TODO Handle temporary memory better

        REALTYPE patternDenominator = 0.0;

        for (int category = 0; category < kCategoryCount; category++) {

            const REALTYPE scale = (REALTYPE) categoryRates[category] * edgeLength;

            const REALTYPE weight = categoryWeights[category];
            const int patternIndex = category * kPatternCount + pattern; // Bad memory access
            const int v = patternIndex * kPartialsPaddedStateCount;

            REALTYPE denominator = 0.0;
            for (int k = 0; k < kStateCount; k++) {
                denominator += postOrderPartial[v + k] * preOrderPartial[v + k];
            }
            patternDenominator += denominator * weight;

            for (int k = 0; k < kStateCount; k++) {
                for (int j = 0; j < kStateCount; j++) {
                    tmp[k * kStateCount + j] += preOrderPartial[v + k] * postOrderPartial[v + j] * weight * scale;
                }
            }
        }

        const auto patternWeight = gPatternWeights[pattern] / patternDenominator;
        for (int k = 0; k < kStateCount; k++) {
            for (int j = 0; j < kStateCount; j++) {
                outCrossProducts[k * kStateCount + j] += tmp[k * kStateCount + j] * patternWeight;
            }
        }
    }
}

BEAGLE_CPU_TEMPLATE
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcEdgeLogDerivativesPartials(const REALTYPE *postOrderPartial,
                                                                       const REALTYPE *preOrderPartial,
                                                                       const int firstDerivativeIndex,
                                                                       const int secondDerivativeIndex,
                                                                       const double *categoryRates,
                                                                       const REALTYPE *categoryWeights,
                                                                       const int scalingFactorsIndex,
                                                                       double *outDerivatives,
                                                                       double *outSumDerivatives,
                                                                       double *outSumSquaredDerivatives,
                                                                       int cacheOffset) {

    const REALTYPE *firstDerivMatrix = gTransitionMatrices[firstDerivativeIndex];

    for (int category = 0; category < kCategoryCount; category++) {
        const REALTYPE weight = categoryWeights[category];

        for (int pattern = 0; pattern < kPatternCount; pattern++) {

            int w = category * kMatrixSize;

            const int patternIndex = category * kPatternCount + pattern;
            const int v = patternIndex * kPartialsPaddedStateCount;

            REALTYPE numerator = 0.0;
            REALTYPE denominator = 0.0;

            for (int k = 0; k < kStateCount; k++) {

                REALTYPE sumOverEndState = 0.0;
                for (int j = 0; j < kStateCount; j++) {
                    sumOverEndState += firstDerivMatrix[w]
                                       * postOrderPartial[v + j]; // fix padded index
                    w++;
                }
                w += T_PAD;

                numerator += sumOverEndState * preOrderPartial[v + k];
                denominator += postOrderPartial[v + k] * preOrderPartial[v + k];
            }

            grandNumeratorDerivTmp[cacheOffset + pattern] += weight * numerator;
            grandDenominatorDerivTmp[cacheOffset + pattern] += weight * denominator;
        }
    }
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::waitForPartials(const int* destinationPartials,
                                   int destinationPartialsCount) {
    return BEAGLE_SUCCESS;
}

BEAGLE_CPU_TEMPLATE
    int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calculateRootLogLikelihoods(const int *bufferIndices,
                                                                       const int *categoryWeightsIndices,
                                                                       const int *stateFrequenciesIndices,
                                                                       const int *cumulativeScaleIndices,
                                                                       int count,
                                                                       double *outSumLogLikelihood) {

    BEAGLE_CHECK_INDICES(badRootIndices(bufferIndices, categoryWeightsIndices, cumulativeScaleIndices, count));

    if (count == 1) {
        // We treat this as a special case so that we don't have convoluted logic
        //      at the end of the loop over patterns
        int cumulativeScalingFactorIndex;
        if (kFlags & BEAGLE_FLAG_SCALING_AUTO) {
            cumulativeScalingFactorIndex = 0;
        } else if (kFlags & BEAGLE_FLAG_SCALING_ALWAYS) {
            cumulativeScalingFactorIndex = bufferIndices[0] - kTipCount;
        } else {
            cumulativeScalingFactorIndex = cumulativeScaleIndices[0];
        }

        if (kAutoRootPartitioningEnabled) {
            calcRootLogLikelihoodsByAutoPartitionAsync(bufferIndices,
                                                       categoryWeightsIndices,
                                                       stateFrequenciesIndices,
                                                       cumulativeScaleIndices,
                                                       gAutoPartitionIndices,
                                                       gAutoPartitionOutSumLogLikelihoods);

            *outSumLogLikelihood = 0.0;

            for (int i = 0; i < kPartitionCount; i++) {
                *outSumLogLikelihood += gAutoPartitionOutSumLogLikelihoods[i];
            }

            if (*outSumLogLikelihood != *outSumLogLikelihood) {
                return BEAGLE_ERROR_FLOATING_POINT;
            } else {
                return BEAGLE_SUCCESS;
            }
        } else {

            if (categoryWeightsIndices[0] >= 0) {
                return calcRootLogLikelihoods(bufferIndices[0], categoryWeightsIndices[0], stateFrequenciesIndices[0],
                                              cumulativeScalingFactorIndex, outSumLogLikelihood);
            } else {
                return calcRootLogLikelihoodsPerCategory(
                        bufferIndices[0], stateFrequenciesIndices[0], cumulativeScalingFactorIndex, outSumLogLikelihood);
            }
        }
    } else {
        return calcRootLogLikelihoodsMulti(bufferIndices, categoryWeightsIndices, stateFrequenciesIndices,
                                           cumulativeScaleIndices, count, outSumLogLikelihood);
    }
}

BEAGLE_CPU_TEMPLATE
    int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calculateRootLogLikelihoodsByPartition(
                                                                  const int* bufferIndices,
                                                                  const int* categoryWeightsIndices,
                                                                  const int* stateFrequenciesIndices,
                                                                  const int* cumulativeScaleIndices,
                                                                  const int* partitionIndices,
                                                                  int partitionCount,
                                                                  int count,
                                                                  double* outSumLogLikelihoodByPartition,
                                                                  double* outSumLogLikelihood) {

    int returnCode = BEAGLE_SUCCESS;

    if (count == 1) {
        if (kFlags & BEAGLE_FLAG_SCALING_AUTO) {
            returnCode = BEAGLE_ERROR_NO_IMPLEMENTATION;
        } else if (kFlags & BEAGLE_FLAG_SCALING_ALWAYS) {
            returnCode = BEAGLE_ERROR_NO_IMPLEMENTATION;
        } else {
#ifdef BEAGLE_INDEX_CHECKS
            for (int p = 0; p < partitionCount; p++) {
                if (badPartition(partitionIndices[p]) || badPartials(bufferIndices[p]) ||
                    badRootScale(cumulativeScaleIndices[p], rootReadsNegativeScale()))
                    return BEAGLE_ERROR_OUT_OF_RANGE;
            }
#endif
            if (kThreadingEnabled) {
                calcRootLogLikelihoodsByPartitionAsync(bufferIndices, categoryWeightsIndices, stateFrequenciesIndices, cumulativeScaleIndices, partitionIndices, partitionCount, outSumLogLikelihoodByPartition);
            } else {
                calcRootLogLikelihoodsByPartition(bufferIndices, categoryWeightsIndices, stateFrequenciesIndices, cumulativeScaleIndices, partitionIndices, partitionCount, outSumLogLikelihoodByPartition);
            }

            *outSumLogLikelihood = 0.0;

            for (int i = 0; i < partitionCount; i++) {
                *outSumLogLikelihood += outSumLogLikelihoodByPartition[i];
            }

            if (*outSumLogLikelihood != *outSumLogLikelihood)
                returnCode = BEAGLE_ERROR_FLOATING_POINT;
        }
    } else {
        returnCode = BEAGLE_ERROR_NO_IMPLEMENTATION;
    }

    return returnCode;
}

BEAGLE_CPU_TEMPLATE
    void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcRootLogLikelihoodsByPartitionAsync(
                                                        const int* bufferIndices,
                                                        const int* categoryWeightsIndices,
                                                        const int* stateFrequenciesIndices,
                                                        const int* cumulativeScaleIndices,
                                                        const int* partitionIndices,
                                                        int partitionCount,
                                                        double* outSumLogLikelihoodByPartition) {


    int partitionsPerThreadFloor = partitionCount / kNumThreads;
    int partitionsRemainder = partitionCount % kNumThreads;
    int currentPartitionIndex = 0;
    int threadsUsed = (partitionCount < kNumThreads ? partitionCount : kNumThreads);
    for (int i=0; i<threadsUsed; i++) {
        int partitionCountThread = partitionsPerThreadFloor;
        if (partitionsRemainder) {
            partitionCountThread++;
            partitionsRemainder--;
        }

        std::packaged_task<void()> threadTask(
            std::bind(&BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcRootLogLikelihoodsByPartition, this,
                      &bufferIndices[currentPartitionIndex], &categoryWeightsIndices[currentPartitionIndex],
                      &stateFrequenciesIndices[currentPartitionIndex], &cumulativeScaleIndices[currentPartitionIndex],
                      &partitionIndices[currentPartitionIndex], partitionCountThread,
                      &outSumLogLikelihoodByPartition[currentPartitionIndex]));

        gFutures[i] = threadTask.get_future();
        threadData* td = &gThreads[i];

        std::unique_lock<std::mutex> l(td->m);
        td->jobs.push(std::move(threadTask));
        l.unlock();

        gThreads[i].cv.notify_one();

        currentPartitionIndex += partitionCountThread;
    }

    for (int i=0; i<kNumThreads; i++) {
        gFutures[i].wait();
    }

}

BEAGLE_CPU_TEMPLATE
    void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcRootLogLikelihoodsByAutoPartitionAsync(
                                                        const int* bufferIndices,
                                                        const int* categoryWeightsIndices,
                                                        const int* stateFrequenciesIndices,
                                                        const int* cumulativeScaleIndices,
                                                        const int* partitionIndices,
                                                        double* outSumLogLikelihoodByPartition) {

    for (int i=0; i<kNumThreads; i++) {

        std::packaged_task<void()> threadTask(
            std::bind(&BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcRootLogLikelihoodsByPartition, this,
                      bufferIndices, categoryWeightsIndices,
                      stateFrequenciesIndices, cumulativeScaleIndices,
                      &partitionIndices[i], 1,
                      &outSumLogLikelihoodByPartition[i]));

        gFutures[i] = threadTask.get_future();
        threadData* td = &gThreads[i];

        std::unique_lock<std::mutex> l(td->m);
        td->jobs.push(std::move(threadTask));
        l.unlock();

        gThreads[i].cv.notify_one();

    }

    for (int i=0; i<kNumThreads; i++) {
        gFutures[i].wait();
    }

}


BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcRootLogLikelihoodsMulti(const int* bufferIndices,
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
            v += P_PAD;
        }
        for (int l = 1; l < kCategoryCount; l++) {
            u = 0;
            for (int k = 0; k < kPatternCount; k++) {
                for (int i = 0; i < kStateCount; i++) {
                    integrationTmp[u] += rootPartials[v] * (REALTYPE) wt[l];
                    u++;
                    v++;
                }
                v += P_PAD;
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

    if (*outSumLogLikelihood != *outSumLogLikelihood)
        returnCode = BEAGLE_ERROR_FLOATING_POINT;

    return returnCode;

}

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcRootLogLikelihoodsPerCategory(
        const int bufferIndex,
        const int stateFrequenciesIndex,
        const int scalingFactorsIndex,
        double* outLogLikelihoodPerCategory) {

    int returnCode = BEAGLE_SUCCESS;

    const REALTYPE* rootPartials = gPartials[bufferIndex];
    const REALTYPE* freqs = gStateFrequencies[stateFrequenciesIndex];

    int u = 0;
    int v = 0;
    for (int l = 0; l < kCategoryCount; l++) {
        for (int k = 0; k < kPatternCount; k++) {
            REALTYPE sum = 0.0;
            for (int i = 0; i < kStateCount; i++) {
                sum += rootPartials[v] * freqs[i];
                v++;
            }
            outLogLikelihoodPerCategory[u] = log(sum);
            u++;
            v += P_PAD;
        }
    }

    if (scalingFactorsIndex >= 0) {
        const REALTYPE* cumulativeScaleFactors = gScaleBuffers[scalingFactorsIndex];
        int u = 0;
        for (int l = 0; l < kCategoryCount; l++) {
            for (int i = 0; i < kPatternCount; i++) {
                outLogLikelihoodPerCategory[u] += cumulativeScaleFactors[i];
                u++;
            }
        }
    }

    return returnCode;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcRootLogLikelihoods(const int bufferIndex,
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
        v += P_PAD;
    }
    for (int l = 1; l < kCategoryCount; l++) {
        u = 0;
        for (int k = 0; k < kPatternCount; k++) {
            for (int i = 0; i < kStateCount; i++) {
                integrationTmp[u] += rootPartials[v] * (REALTYPE) wt[l];
                u++;
                v++;
            }
            v += P_PAD;
        }
    }
    u = 0;
    for (int k = 0; k < kPatternCount; k++) {
        REALTYPE sum = 0.0;
        for (int i = 0; i < kStateCount; i++) {
            sum += freqs[i] * integrationTmp[u];
            u++;
        }

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

    if (*outSumLogLikelihood != *outSumLogLikelihood)
        returnCode = BEAGLE_ERROR_FLOATING_POINT;

    // TODO: merge the three kPatternCount loops above into one

    return returnCode;
}

BEAGLE_CPU_TEMPLATE
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcRootLogLikelihoodsByPartition(
                                                         const int* bufferIndices,
                                                         const int* categoryWeightsIndices,
                                                         const int* stateFrequenciesIndices,
                                                         const int* cumulativeScaleIndices,
                                                         const int* partitionIndices,
                                                         int partitionCount,
                                                         double* outSumLogLikelihoodByPartition) {

    for (int p = 0; p < partitionCount; p++) {
        int pIndex = partitionIndices[p];

        int startPattern = gPatternPartitionsStartPatterns[pIndex];
        int endPattern = gPatternPartitionsStartPatterns[pIndex + 1];

        const REALTYPE* rootPartials = gPartials[bufferIndices[p]];
        const REALTYPE* wt = gCategoryWeights[categoryWeightsIndices[p]];
        const REALTYPE* freqs = gStateFrequencies[stateFrequenciesIndices[p]];
        const int scalingFactorsIndex = cumulativeScaleIndices[p];
        int u = startPattern * kStateCount;
        int v = startPattern * kPartialsPaddedStateCount;
        for (int k = startPattern; k < endPattern; k++) {
            for (int i = 0; i < kStateCount; i++) {
                integrationTmp[u] = rootPartials[v] * (REALTYPE) wt[0];
                u++;
                v++;
            }
            v += P_PAD;
        }
        for (int l = 1; l < kCategoryCount; l++) {
            u = startPattern * kStateCount;
            v += ((kPatternCount - endPattern) + startPattern) * kPartialsPaddedStateCount;
            for (int k = startPattern; k < endPattern; k++) {
                for (int i = 0; i < kStateCount; i++) {
                    integrationTmp[u] += rootPartials[v] * (REALTYPE) wt[l];
                    u++;
                    v++;
                }
                v += P_PAD;
            }
        }
        u = startPattern * kStateCount;
        for (int k = startPattern; k < endPattern; k++) {
            REALTYPE sum = 0.0;
            for (int i = 0; i < kStateCount; i++) {
                sum += freqs[i] * integrationTmp[u];
                u++;
            }

            outLogLikelihoodsTmp[k] = log(sum);
        }

        if (scalingFactorsIndex >= 0) {
            const REALTYPE* cumulativeScaleFactors = gScaleBuffers[scalingFactorsIndex];
            for(int i=startPattern; i<endPattern; i++) {
                outLogLikelihoodsTmp[i] += cumulativeScaleFactors[i];
            }
        }

        outSumLogLikelihoodByPartition[p] = 0.0;
        for (int i = startPattern; i < endPattern; i++) {
            outSumLogLikelihoodByPartition[p] += outLogLikelihoodsTmp[i] * gPatternWeights[i];
        }

    }

}

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::accumulateScaleFactors(const int* scalingIndices,
                                                int  count,
                                                int  cumulativeScalingIndex) {
#ifdef BEAGLE_INDEX_CHECKS
    // under SCALING_AUTO the indices are internal partials buffers and the cumulative index is not used
    for (int i = 0; i < count; i++) {
        if ((kFlags & BEAGLE_FLAG_SCALING_AUTO) ? (scalingIndices[i] < kTipCount || badPartials(scalingIndices[i]))
                                                : badScale(scalingIndices[i]))
            return BEAGLE_ERROR_OUT_OF_RANGE;
    }
    if (!(kFlags & BEAGLE_FLAG_SCALING_AUTO) && badScale(cumulativeScalingIndex))
        return BEAGLE_ERROR_OUT_OF_RANGE;
#endif
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

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::accumulateScaleFactorsByPartition(const int* scalingIndices,
                                                                         int count,
                                                                         int cumulativeScalingIndex,
                                                                         int partitionIndex) {
    if (kFlags & BEAGLE_FLAG_SCALING_AUTO) {
        return BEAGLE_ERROR_NO_IMPLEMENTATION;
    } else {
        BEAGLE_CHECK_INDICES(badPartition(partitionIndex) || badScale(cumulativeScalingIndex));
#ifdef BEAGLE_INDEX_CHECKS
        for (int i = 0; i < count; i++) {
            if (badScale(scalingIndices[i]))
                return BEAGLE_ERROR_OUT_OF_RANGE;
        }
#endif

        int startPattern = gPatternPartitionsStartPatterns[partitionIndex];
        int endPattern = gPatternPartitionsStartPatterns[partitionIndex + 1];

        REALTYPE* cumulativeScaleBuffer = gScaleBuffers[cumulativeScalingIndex];
        for(int i=0; i<count; i++) {
            const REALTYPE* scaleBuffer = gScaleBuffers[scalingIndices[i]];
            for(int j=startPattern; j<endPattern; j++) {
                if (kFlags & BEAGLE_FLAG_SCALERS_LOG)
                    cumulativeScaleBuffer[j] += scaleBuffer[j];
                else
                    cumulativeScaleBuffer[j] += log(scaleBuffer[j]);
            }
        }

    }

    return BEAGLE_SUCCESS;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::removeScaleFactors(const int* scalingIndices,
                                                          int  count,
                                                          int  cumulativeScalingIndex) {
    BEAGLE_CHECK_INDICES(badScale(cumulativeScalingIndex));
#ifdef BEAGLE_INDEX_CHECKS
    for (int i = 0; i < count; i++) {
        if (badScale(scalingIndices[i]))
            return BEAGLE_ERROR_OUT_OF_RANGE;
    }
#endif
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

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::removeScaleFactorsByPartition(const int* scalingIndices,
                                                                     int count,
                                                                     int cumulativeScalingIndex,
                                                                     int partitionIndex) {
    BEAGLE_CHECK_INDICES(badPartition(partitionIndex) || badScale(cumulativeScalingIndex));
#ifdef BEAGLE_INDEX_CHECKS
    for (int i = 0; i < count; i++) {
        if (badScale(scalingIndices[i]))
            return BEAGLE_ERROR_OUT_OF_RANGE;
    }
#endif

    int startPattern = gPatternPartitionsStartPatterns[partitionIndex];
    int endPattern = gPatternPartitionsStartPatterns[partitionIndex + 1];

    REALTYPE* cumulativeScaleBuffer = gScaleBuffers[cumulativeScalingIndex];
    for(int i=0; i<count; i++) {
        const REALTYPE* scaleBuffer = gScaleBuffers[scalingIndices[i]];
        for(int j=startPattern; j<endPattern; j++) {
            if (kFlags & BEAGLE_FLAG_SCALERS_LOG)
                cumulativeScaleBuffer[j] -= scaleBuffer[j];
            else
                cumulativeScaleBuffer[j] -= log(scaleBuffer[j]);
        }
    }

    return BEAGLE_SUCCESS;
}


BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::resetScaleFactors(int cumulativeScalingIndex) {
    BEAGLE_CHECK_INDICES(badScale(cumulativeScalingIndex));
    //memcpy(gScaleBuffers[cumulativeScalingIndex],zeros,sizeof(double) * kPatternCount);

     if (kFlags & BEAGLE_FLAG_SCALING_AUTO) {
         memset(gScaleBuffers[cumulativeScalingIndex], 0, sizeof(signed short) * kPaddedPatternCount);
     } else {
         memset(gScaleBuffers[cumulativeScalingIndex], 0, sizeof(REALTYPE) * kPaddedPatternCount);
     }
    return BEAGLE_SUCCESS;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::resetScaleFactorsByPartition(int cumulativeScalingIndex,
                                                                    int partitionIndex) {

     if (kFlags & BEAGLE_FLAG_SCALING_AUTO) {
        return BEAGLE_ERROR_NO_IMPLEMENTATION;
     } else {
        BEAGLE_CHECK_INDICES(badPartition(partitionIndex) || badScale(cumulativeScalingIndex));
        int startPattern = gPatternPartitionsStartPatterns[partitionIndex];
        int endPattern = gPatternPartitionsStartPatterns[partitionIndex + 1];

        REALTYPE* cumulativeBuffer = gScaleBuffers[cumulativeScalingIndex];

        memset(&cumulativeBuffer[startPattern], 0, sizeof(REALTYPE) * (endPattern - startPattern));
     }
    return BEAGLE_SUCCESS;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::copyScaleFactors(int destScalingIndex,
                                                        int srcScalingIndex) {
    BEAGLE_CHECK_INDICES(badScale(destScalingIndex) || badScale(srcScalingIndex));
    memcpy(gScaleBuffers[destScalingIndex],gScaleBuffers[srcScalingIndex],sizeof(REALTYPE) * kPatternCount);

    return BEAGLE_SUCCESS;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::getScaleFactors(int srcScalingIndex,
                                                       double* scaleFactors) {
    // Do nothing
    return BEAGLE_SUCCESS;
}

BEAGLE_CPU_TEMPLATE
    int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calculateEdgeLogLikelihoods(const int* parentBufferIndices,
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
    BEAGLE_CHECK_INDICES(badEdgeIndices(parentBufferIndices, childBufferIndices, probabilityIndices,
                                        firstDerivativeIndices, secondDerivativeIndices, cumulativeScaleIndices, count));

    if (missingDenseMatrix(probabilityIndices, count) ||
        (firstDerivativeIndices != NULL && missingDenseMatrix(firstDerivativeIndices, count)) ||
        (secondDerivativeIndices != NULL && missingDenseMatrix(secondDerivativeIndices, count)))
        return BEAGLE_ERROR_OUT_OF_RANGE;

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

            if (kAutoRootPartitioningEnabled) {
                calcEdgeLogLikelihoodsByAutoPartitionAsync(parentBufferIndices,
                                                           childBufferIndices,
                                                           probabilityIndices,
                                                           categoryWeightsIndices,
                                                           stateFrequenciesIndices,
                                                           cumulativeScaleIndices,
                                                           gAutoPartitionIndices,
                                                           gAutoPartitionOutSumLogLikelihoods);

                *outSumLogLikelihood = 0.0;

                for (int i = 0; i < kPartitionCount; i++) {
                    *outSumLogLikelihood += gAutoPartitionOutSumLogLikelihoods[i];
                }

                if (*outSumLogLikelihood != *outSumLogLikelihood) {
                    return BEAGLE_ERROR_FLOATING_POINT;
                } else {
                    return BEAGLE_SUCCESS;
                }
            } else {
                return calcEdgeLogLikelihoods(parentBufferIndices[0], childBufferIndices[0], probabilityIndices[0],
                                       categoryWeightsIndices[0], stateFrequenciesIndices[0], cumulativeScalingFactorIndex,
                                       outSumLogLikelihood);
            }
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
        if ((kFlags & BEAGLE_FLAG_SCALING_AUTO) || (kFlags & BEAGLE_FLAG_SCALING_ALWAYS)) {
            fprintf(stderr,"BeagleCPUImpl::calculateEdgeLogLikelihoods not yet implemented for count > 1 and auto/always scaling\n");
        }

        if (firstDerivativeIndices == NULL && secondDerivativeIndices == NULL) {
            return calcEdgeLogLikelihoodsMulti(parentBufferIndices, childBufferIndices, probabilityIndices,
                                          categoryWeightsIndices, stateFrequenciesIndices, cumulativeScaleIndices, count,
                                          outSumLogLikelihood);
        } else {
            fprintf(stderr,"BeagleCPUImpl::calculateEdgeLogLikelihoods not yet implemented for count > 1 and derivatives\n");
        }
    }
    return BEAGLE_SUCCESS;
}

BEAGLE_CPU_TEMPLATE
    int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calculateEdgeLogLikelihoodsByPartition(
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

    int returnCode = BEAGLE_SUCCESS;

    BEAGLE_CHECK_INDICES(denseMatricesOnDemand() &&
                         (badMatrices(probabilityIndices, partitionCount) ||
                          (firstDerivativeIndices != NULL && badMatrices(firstDerivativeIndices, partitionCount)) ||
                          (secondDerivativeIndices != NULL && badMatrices(secondDerivativeIndices, partitionCount))));

    if (missingDenseMatrix(probabilityIndices, partitionCount) ||
        (firstDerivativeIndices != NULL && missingDenseMatrix(firstDerivativeIndices, partitionCount)) ||
        (secondDerivativeIndices != NULL && missingDenseMatrix(secondDerivativeIndices, partitionCount)))
        return BEAGLE_ERROR_OUT_OF_RANGE;

    if (count == 1) {
        if (kFlags & BEAGLE_FLAG_SCALING_AUTO) {
            returnCode =  BEAGLE_ERROR_NO_IMPLEMENTATION;
        } else if (kFlags & BEAGLE_FLAG_SCALING_ALWAYS) {
            returnCode = BEAGLE_ERROR_NO_IMPLEMENTATION;
        } else {
#ifdef BEAGLE_INDEX_CHECKS
            // implemented without derivatives or with both; first derivatives alone are NO_IMPLEMENTATION
            if (!(firstDerivativeIndices != NULL && secondDerivativeIndices == NULL)) {
                for (int p = 0; p < partitionCount; p++) {
                    if (badPartition(partitionIndices[p]) || badPartials(parentBufferIndices[p]) ||
                        badPartials(childBufferIndices[p]) || badMatrix(probabilityIndices[p]) ||
                        (firstDerivativeIndices != NULL && badMatrix(firstDerivativeIndices[p])) ||
                        (secondDerivativeIndices != NULL && badMatrix(secondDerivativeIndices[p])) ||
                        (cumulativeScaleIndices[p] != BEAGLE_OP_NONE && badScale(cumulativeScaleIndices[p])))
                        return BEAGLE_ERROR_OUT_OF_RANGE;
                }
            }
#endif

            if (firstDerivativeIndices == NULL && secondDerivativeIndices == NULL) {

                if (kThreadingEnabled) {
                    calcEdgeLogLikelihoodsByPartitionAsync(parentBufferIndices,
                                                           childBufferIndices,
                                                           probabilityIndices,
                                                           categoryWeightsIndices,
                                                           stateFrequenciesIndices,
                                                           cumulativeScaleIndices,
                                                           partitionIndices,
                                                           partitionCount,
                                                           outSumLogLikelihoodByPartition);
                } else {
                    calcEdgeLogLikelihoodsByPartition(parentBufferIndices,
                                                      childBufferIndices,
                                                      probabilityIndices,
                                                      categoryWeightsIndices,
                                                      stateFrequenciesIndices,
                                                      cumulativeScaleIndices,
                                                      partitionIndices,
                                                      partitionCount,
                                                      outSumLogLikelihoodByPartition);
                }


            } else if (secondDerivativeIndices == NULL) {
                return BEAGLE_ERROR_NO_IMPLEMENTATION;
            } else {

                calcEdgeLogLikelihoodsSecondDerivByPartition(
                                                parentBufferIndices,
                                                childBufferIndices,
                                                probabilityIndices,
                                                firstDerivativeIndices,
                                                secondDerivativeIndices,
                                                categoryWeightsIndices,
                                                stateFrequenciesIndices,
                                                cumulativeScaleIndices,
                                                partitionIndices,
                                                partitionCount,
                                                outSumLogLikelihoodByPartition,
                                                outSumFirstDerivativeByPartition,
                                                outSumSecondDerivativeByPartition);

                *outSumFirstDerivative  = 0.0;
                *outSumSecondDerivative = 0.0;

                for (int i = 0; i < partitionCount; i++) {
                    *outSumFirstDerivative  += outSumFirstDerivativeByPartition[i];
                    *outSumSecondDerivative += outSumSecondDerivativeByPartition[i];
                }

                if (*outSumFirstDerivative  != *outSumFirstDerivative ||
                    *outSumSecondDerivative != *outSumSecondDerivative) {
                    returnCode = BEAGLE_ERROR_FLOATING_POINT;
                }
            }

            *outSumLogLikelihood = 0.0;

            for (int i = 0; i < partitionCount; i++) {
                *outSumLogLikelihood += outSumLogLikelihoodByPartition[i];
            }

            if (*outSumLogLikelihood != *outSumLogLikelihood) {
                returnCode = BEAGLE_ERROR_FLOATING_POINT;
            }

        }
    } else {
        returnCode = BEAGLE_ERROR_NO_IMPLEMENTATION;
    }

    return returnCode;
}

BEAGLE_CPU_TEMPLATE
    void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcEdgeLogLikelihoodsByPartitionAsync(
                                                        const int* parentBufferIndices,
                                                        const int* childBufferIndices,
                                                        const int* probabilityIndices,
                                                        const int* categoryWeightsIndices,
                                                        const int* stateFrequenciesIndices,
                                                        const int* cumulativeScaleIndices,
                                                        const int* partitionIndices,
                                                        int partitionCount,
                                                        double* outSumLogLikelihoodByPartition) {

    int partitionsPerThreadFloor = partitionCount / kNumThreads;
    int partitionsRemainder = partitionCount % kNumThreads;
    int currentPartitionIndex = 0;
    int threadsUsed = (partitionCount < kNumThreads ? partitionCount : kNumThreads);
    for (int i=0; i<threadsUsed; i++) {
        int partitionCountThread = partitionsPerThreadFloor;
        if (partitionsRemainder) {
            partitionCountThread++;
            partitionsRemainder--;
        }

        std::packaged_task<void()> threadTask(
            std::bind(&BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcEdgeLogLikelihoodsByPartition, this,
                      &parentBufferIndices[currentPartitionIndex],
                      &childBufferIndices[currentPartitionIndex],
                      &probabilityIndices[currentPartitionIndex],
                      &categoryWeightsIndices[currentPartitionIndex],
                      &stateFrequenciesIndices[currentPartitionIndex],
                      &cumulativeScaleIndices[currentPartitionIndex],
                      &partitionIndices[currentPartitionIndex],
                      partitionCountThread,
                      &outSumLogLikelihoodByPartition[currentPartitionIndex]));

        gFutures[i] = threadTask.get_future();
        threadData* td = &gThreads[i];

        std::unique_lock<std::mutex> l(td->m);
        td->jobs.push(std::move(threadTask));
        l.unlock();

        gThreads[i].cv.notify_one();

        currentPartitionIndex += partitionCountThread;
    }

    for (int i=0; i<kNumThreads; i++) {
        gFutures[i].wait();
    }

}

BEAGLE_CPU_TEMPLATE
    void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcEdgeLogLikelihoodsByAutoPartitionAsync(
                                                        const int* parentBufferIndices,
                                                        const int* childBufferIndices,
                                                        const int* probabilityIndices,
                                                        const int* categoryWeightsIndices,
                                                        const int* stateFrequenciesIndices,
                                                        const int* cumulativeScaleIndices,
                                                        const int* partitionIndices,
                                                        double* outSumLogLikelihoodByPartition) {

    for (int i=0; i<kNumThreads; i++) {

        std::packaged_task<void()> threadTask(
            std::bind(&BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcEdgeLogLikelihoodsByPartition, this,
                      parentBufferIndices,
                      childBufferIndices,
                      probabilityIndices,
                      categoryWeightsIndices,
                      stateFrequenciesIndices,
                      cumulativeScaleIndices,
                      &partitionIndices[i],
                      1,
                      &outSumLogLikelihoodByPartition[i]));

        gFutures[i] = threadTask.get_future();
        threadData* td = &gThreads[i];

        std::unique_lock<std::mutex> l(td->m);
        td->jobs.push(std::move(threadTask));
        l.unlock();

        gThreads[i].cv.notify_one();
    }

    for (int i=0; i<kNumThreads; i++) {
        gFutures[i].wait();
    }

}



BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcEdgeLogLikelihoods(const int parIndex,
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

                    w += kTransPaddedStateCount;
                }
                v += kPartialsPaddedStateCount;
            }
        }

    } else { // Integrate against a partial at the child

        const REALTYPE* partialsChild = gPartials[childIndex];
        int v = 0;
        int stateCountModFour = (kStateCount / 4) * 4;

        for(int l = 0; l < kCategoryCount; l++) {
            int u = 0;
            const REALTYPE weight = wt[l];
            for(int k = 0; k < kPatternCount; k++) {
                int w = l * kMatrixSize;
                const REALTYPE* partialsChildPtr = &partialsChild[v];
                for(int i = 0; i < kStateCount; i++) {
                    double sumOverJA = 0.0, sumOverJB = 0.0;
                    int j = 0;
                    const REALTYPE* transMatrixPtr = &transMatrix[w];
                    for (; j < stateCountModFour; j += 4) {
                        sumOverJA += transMatrixPtr[j + 0] * partialsChildPtr[j + 0];
                        sumOverJB += transMatrixPtr[j + 1] * partialsChildPtr[j + 1];
                        sumOverJA += transMatrixPtr[j + 2] * partialsChildPtr[j + 2];
                        sumOverJB += transMatrixPtr[j + 3] * partialsChildPtr[j + 3];

                    }
                    for (; j < kStateCount; j++) {
                        sumOverJA += transMatrixPtr[j] * partialsChildPtr[j];
                    }
                    //                        for(int j = 0; j < kStateCount; j++) {
                    //                            sumOverJ += transMatrix[w] * partialsChild[v + j];
                    //                            w++;
                    //                        }
                    integrationTmp[u] += (sumOverJA + sumOverJB) * partialsParent[v + i] * weight;
                    u++;

                    w += kStateCount;

                    // increment for the extra column at the end
                    w += T_PAD;
                }
                v += kPartialsPaddedStateCount;
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

    if (*outSumLogLikelihood != *outSumLogLikelihood)
        returnCode = BEAGLE_ERROR_FLOATING_POINT;

    return returnCode;
}

BEAGLE_CPU_TEMPLATE
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcEdgeLogLikelihoodsByPartition(
                                                  const int* parentBufferIndices,
                                                  const int* childBufferIndices,
                                                  const int* probabilityIndices,
                                                  const int* categoryWeightsIndices,
                                                  const int* stateFrequenciesIndices,
                                                  const int* cumulativeScaleIndices,
                                                  const int* partitionIndices,
                                                  int partitionCount,
                                                  double* outSumLogLikelihoodByPartition) {


    for (int p = 0; p < partitionCount; p++) {
        int pIndex = partitionIndices[p];

        int startPattern = gPatternPartitionsStartPatterns[pIndex];
        int endPattern = gPatternPartitionsStartPatterns[pIndex + 1];

        memset(&integrationTmp[startPattern*kStateCount], 0, ((endPattern - startPattern) * kStateCount)*sizeof(REALTYPE));

        const int parIndex = parentBufferIndices[p];
        const int childIndex = childBufferIndices[p];
        const int probIndex = probabilityIndices[p];
        const int categoryWeightsIndex = categoryWeightsIndices[p];
        const int stateFrequenciesIndex = stateFrequenciesIndices[p];
        const int scalingFactorsIndex = cumulativeScaleIndices[p];

        assert(parIndex >= kTipCount);

        const REALTYPE* partialsParent = gPartials[parIndex];
        const REALTYPE* transMatrix = gTransitionMatrices[probIndex];
        const REALTYPE* wt = gCategoryWeights[categoryWeightsIndex];
        const REALTYPE* freqs = gStateFrequencies[stateFrequenciesIndex];

        if (childIndex < kTipCount && gTipStates[childIndex]) { // Integrate against a state at the child
            const int* statesChild = gTipStates[childIndex];
            int v = startPattern * kPartialsPaddedStateCount; // Index for parent partials

            for(int l = 0; l < kCategoryCount; l++) {
                int u = startPattern * kStateCount; // Index in resulting product-partials (summed over categories)
                const REALTYPE weight = wt[l];
                for(int k = startPattern; k < endPattern; k++) {

                    const int stateChild = statesChild[k];  // DISCUSSION PT: Does it make sense to change the order of the partials,
                    // so we can interchange the patterCount and categoryCount loop order?
                    int w =  l * kMatrixSize;
                    for(int i = 0; i < kStateCount; i++) {
                        integrationTmp[u] += transMatrix[w + stateChild] * partialsParent[v + i] * weight;
                        u++;

                        w += kTransPaddedStateCount;
                    }
                    v += kPartialsPaddedStateCount;
                }
                v += ((kPatternCount - endPattern) + startPattern) * kPartialsPaddedStateCount;
            }

        } else { // Integrate against a partial at the child
            const REALTYPE* partialsChild = gPartials[childIndex];
            int v = startPattern * kPartialsPaddedStateCount;
            int stateCountModFour = (kStateCount / 4) * 4;

            for(int l = 0; l < kCategoryCount; l++) {
                int u = startPattern * kStateCount;
                const REALTYPE weight = wt[l];
                for(int k = startPattern; k < endPattern; k++) {
                    int w = l * kMatrixSize;
                    const REALTYPE* partialsChildPtr = &partialsChild[v];
                    for(int i = 0; i < kStateCount; i++) {
                        double sumOverJA = 0.0, sumOverJB = 0.0;
                        int j = 0;
                        const REALTYPE* transMatrixPtr = &transMatrix[w];
                        for (; j < stateCountModFour; j += 4) {
                            sumOverJA += transMatrixPtr[j + 0] * partialsChildPtr[j + 0];
                            sumOverJB += transMatrixPtr[j + 1] * partialsChildPtr[j + 1];
                            sumOverJA += transMatrixPtr[j + 2] * partialsChildPtr[j + 2];
                            sumOverJB += transMatrixPtr[j + 3] * partialsChildPtr[j + 3];

                        }
                        for (; j < kStateCount; j++) {
                            sumOverJA += transMatrixPtr[j] * partialsChildPtr[j];
                        }
                        //                        for(int j = 0; j < kStateCount; j++) {
                        //                            sumOverJ += transMatrix[w] * partialsChild[v + j];
                        //                            w++;
                        //                        }
                        integrationTmp[u] += (sumOverJA + sumOverJB) * partialsParent[v + i] * weight;
                        u++;

                        w += kStateCount;

                        // increment for the extra column at the end
                        w += T_PAD;
                    }
                    v += kPartialsPaddedStateCount;
                }
                v += ((kPatternCount - endPattern) + startPattern) * kPartialsPaddedStateCount;
            }
        }

        int u = startPattern * kStateCount;
        for(int k = startPattern; k < endPattern; k++) {
            REALTYPE sumOverI = 0.0;
            for(int i = 0; i < kStateCount; i++) {
                sumOverI += freqs[i] * integrationTmp[u];
                u++;
            }

            outLogLikelihoodsTmp[k] = log(sumOverI);
        }


        if (scalingFactorsIndex != BEAGLE_OP_NONE) {
            const REALTYPE* scalingFactors = gScaleBuffers[scalingFactorsIndex];
            for(int k=startPattern; k < endPattern; k++)
                outLogLikelihoodsTmp[k] += scalingFactors[k];
        }

        outSumLogLikelihoodByPartition[p] = 0.0;
        for (int i = startPattern; i < endPattern; i++) {
            outSumLogLikelihoodByPartition[p] += outLogLikelihoodsTmp[i] * gPatternWeights[i];
        }

    }
}

BEAGLE_CPU_TEMPLATE
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcEdgeLogLikelihoodsSecondDerivByPartition(
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
                                                  double* outSumLogLikelihoodByPartition,
                                                  double* outSumFirstDerivativeByPartition,
                                                  double* outSumSecondDerivativeByPartition) {


    for (int p = 0; p < partitionCount; p++) {
        int pIndex = partitionIndices[p];

        int startPattern = gPatternPartitionsStartPatterns[pIndex];
        int endPattern = gPatternPartitionsStartPatterns[pIndex + 1];

        memset(&integrationTmp[startPattern*kStateCount], 0, ((endPattern - startPattern) * kStateCount)*sizeof(REALTYPE));
        memset(&firstDerivTmp[startPattern*kStateCount], 0, ((endPattern - startPattern) * kStateCount)*sizeof(REALTYPE));
        memset(&secondDerivTmp[startPattern*kStateCount], 0, ((endPattern - startPattern) * kStateCount)*sizeof(REALTYPE));

        const int parIndex = parentBufferIndices[p];
        const int childIndex = childBufferIndices[p];
        const int probIndex = probabilityIndices[p];
        const int firstDerivativeIndex = firstDerivativeIndices[p];
        const int secondDerivativeIndex = secondDerivativeIndices[p];
        const int categoryWeightsIndex = categoryWeightsIndices[p];
        const int stateFrequenciesIndex = stateFrequenciesIndices[p];
        const int scalingFactorsIndex = cumulativeScaleIndices[p];

        assert(parIndex >= kTipCount);

        const REALTYPE* partialsParent = gPartials[parIndex];
        const REALTYPE* transMatrix = gTransitionMatrices[probIndex];
        const REALTYPE* firstDerivMatrix = gTransitionMatrices[firstDerivativeIndex];
        const REALTYPE* secondDerivMatrix = gTransitionMatrices[secondDerivativeIndex];
        const REALTYPE* wt = gCategoryWeights[categoryWeightsIndex];
        const REALTYPE* freqs = gStateFrequencies[stateFrequenciesIndex];

        if (childIndex < kTipCount && gTipStates[childIndex]) { // Integrate against a state at the child

            const int* statesChild = gTipStates[childIndex];
            int v = startPattern * kPartialsPaddedStateCount; // Index for parent partials

            for(int l = 0; l < kCategoryCount; l++) {
                int u = startPattern * kStateCount; // Index in resulting product-partials (summed over categories)
                const REALTYPE weight = wt[l];
                for(int k = startPattern; k < endPattern; k++) {

                    const int stateChild = statesChild[k];  // DISCUSSION PT: Does it make sense to change the order of the partials,
                    // so we can interchange the patterCount and categoryCount loop order?
                    int w =  l * kMatrixSize;
                    for(int i = 0; i < kStateCount; i++) {
                        integrationTmp[u] += transMatrix[w + stateChild] * partialsParent[v + i] * weight;
                        firstDerivTmp[u] += firstDerivMatrix[w + stateChild] * partialsParent[v + i] * weight;
                        secondDerivTmp[u] += secondDerivMatrix[w + stateChild] * partialsParent[v + i] * weight;
                        u++;

                        w += kTransPaddedStateCount;
                    }
                    v += kPartialsPaddedStateCount;
                }
                v += ((kPatternCount - endPattern) + startPattern) * kPartialsPaddedStateCount;
            }

        } else { // Integrate against a partial at the child

            const REALTYPE* partialsChild = gPartials[childIndex];
            int v = startPattern * kPartialsPaddedStateCount;

            for(int l = 0; l < kCategoryCount; l++) {
                int u = startPattern * kStateCount;
                const REALTYPE weight = wt[l];
                for(int k = startPattern; k < endPattern; k++) {
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

                        // increment for the extra column at the end
                        w += T_PAD;

                        integrationTmp[u] += sumOverJ * partialsParent[v + i] * weight;
                        firstDerivTmp[u] += sumOverJD1 * partialsParent[v + i] * weight;
                        secondDerivTmp[u] += sumOverJD2 * partialsParent[v + i] * weight;
                        u++;
                    }
                    v += kPartialsPaddedStateCount;
                }
                v += ((kPatternCount - endPattern) + startPattern) * kPartialsPaddedStateCount;
            }
        }

        int u = startPattern * kStateCount;
        for(int k = startPattern; k < endPattern; k++) {
            REALTYPE sumOverI = 0.0;
            REALTYPE sumOverID1 = 0.0;
            REALTYPE sumOverID2 = 0.0;
            for(int i = 0; i < kStateCount; i++) {
                sumOverI += freqs[i] * integrationTmp[u];
                sumOverID1 += freqs[i] * firstDerivTmp[u];
                sumOverID2 += freqs[i] * secondDerivTmp[u];
                u++;
            }

            outLogLikelihoodsTmp[k] = log(sumOverI);
            outFirstDerivativesTmp[k] = sumOverID1 / sumOverI;
            outSecondDerivativesTmp[k] = sumOverID2 / sumOverI - outFirstDerivativesTmp[k] * outFirstDerivativesTmp[k];
        }


        if (scalingFactorsIndex != BEAGLE_OP_NONE) {
            const REALTYPE* scalingFactors = gScaleBuffers[scalingFactorsIndex];
            for(int k=startPattern; k < endPattern; k++)
                outLogLikelihoodsTmp[k] += scalingFactors[k];
        }


        outSumLogLikelihoodByPartition[p] = 0.0;
        outSumFirstDerivativeByPartition[p] = 0.0;
        outSumSecondDerivativeByPartition[p] = 0.0;
        for (int i = startPattern; i < endPattern; i++) {
            outSumLogLikelihoodByPartition[p]    += outLogLikelihoodsTmp[i]    * gPatternWeights[i];
            outSumFirstDerivativeByPartition[p]  += outFirstDerivativesTmp[i]  * gPatternWeights[i];
            outSumSecondDerivativeByPartition[p] += outSecondDerivativesTmp[i] * gPatternWeights[i];
        }

    }
}





BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcEdgeLogLikelihoodsMulti(const int* parentBufferIndices,
                                                                   const int* childBufferIndices,
                                                                   const int* probabilityIndices,
                                                                   const int* categoryWeightsIndices,
                                                                   const int* stateFrequenciesIndices,
                                                                   const int* scalingFactorsIndices,
                                                                   int count,
                                                                   double* outSumLogLikelihood) {

    std::vector<int> indexMaxScale(kPatternCount);
    std::vector<REALTYPE> maxScaleFactor(kPatternCount);

    int returnCode = BEAGLE_SUCCESS;

    for (int subsetIndex = 0 ; subsetIndex < count; ++subsetIndex ) {
        const REALTYPE* partialsParent = gPartials[parentBufferIndices[subsetIndex]];
        const REALTYPE* transMatrix = gTransitionMatrices[probabilityIndices[subsetIndex]];
        const REALTYPE* wt = gCategoryWeights[categoryWeightsIndices[subsetIndex]];
        const REALTYPE* freqs = gStateFrequencies[stateFrequenciesIndices[subsetIndex]];
        int childIndex = childBufferIndices[subsetIndex];

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

                        w += kTransPaddedStateCount;
                    }
                    v += kPartialsPaddedStateCount;
                }
            }
        } else {
            const REALTYPE* partialsChild = gPartials[childIndex];
            int v = 0;
            int stateCountModFour = (kStateCount / 4) * 4;

            for(int l = 0; l < kCategoryCount; l++) {
                int u = 0;
                const REALTYPE weight = wt[l];
                for(int k = 0; k < kPatternCount; k++) {
                    int w = l * kMatrixSize;
                    const REALTYPE* partialsChildPtr = &partialsChild[v];
                    for(int i = 0; i < kStateCount; i++) {
                        double sumOverJA = 0.0, sumOverJB = 0.0;
                        int j = 0;
                        const REALTYPE* transMatrixPtr = &transMatrix[w];
                        for (; j < stateCountModFour; j += 4) {
                            sumOverJA += transMatrixPtr[j + 0] * partialsChildPtr[j + 0];
                            sumOverJB += transMatrixPtr[j + 1] * partialsChildPtr[j + 1];
                            sumOverJA += transMatrixPtr[j + 2] * partialsChildPtr[j + 2];
                            sumOverJB += transMatrixPtr[j + 3] * partialsChildPtr[j + 3];

                        }
                        for (; j < kStateCount; j++) {
                            sumOverJA += transMatrixPtr[j] * partialsChildPtr[j];
                        }
//                        for(int j = 0; j < kStateCount; j++) {
//                            sumOverJ += transMatrix[w] * partialsChild[v + j];
//                            w++;
//                        }
                        integrationTmp[u] += (sumOverJA + sumOverJB) * partialsParent[v + i] * weight;
                        u++;

                        w += kStateCount;

                        // increment for the extra column at the end
                        w += T_PAD;
                    }
                    v += kPartialsPaddedStateCount;
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

            if (scalingFactorsIndices[0] != BEAGLE_OP_NONE) {
                int cumulativeScalingFactorIndex;
                cumulativeScalingFactorIndex = scalingFactorsIndices[subsetIndex];

                const REALTYPE* cumulativeScaleFactors = gScaleBuffers[cumulativeScalingFactorIndex];

                if (subsetIndex == 0) {
                    indexMaxScale[k] = 0;
                    maxScaleFactor[k] = cumulativeScaleFactors[k];
                    for (int j = 1; j < count; j++) {
                        REALTYPE tmpScaleFactor;
                        tmpScaleFactor = gScaleBuffers[scalingFactorsIndices[j]][k];

                        if (tmpScaleFactor > maxScaleFactor[k]) {
                            indexMaxScale[k] = j;
                            maxScaleFactor[k] = tmpScaleFactor;
                        }
                    }
                }

                if (subsetIndex != indexMaxScale[k])
                    sumOverI *= exp((REALTYPE)(cumulativeScaleFactors[k] - maxScaleFactor[k]));
            }



            if (subsetIndex == 0) {
                outLogLikelihoodsTmp[k] = sumOverI;
            } else if (subsetIndex == count - 1) {
                REALTYPE tmpSum = outLogLikelihoodsTmp[k] + sumOverI;

                outLogLikelihoodsTmp[k] = log(tmpSum);
            } else {
                outLogLikelihoodsTmp[k] += sumOverI;
            }

        }

    }

    if (scalingFactorsIndices[0] != BEAGLE_OP_NONE) {
        for(int i=0; i<kPatternCount; i++)
            outLogLikelihoodsTmp[i] += maxScaleFactor[i];
    }


    *outSumLogLikelihood = 0.0;
    for (int i = 0; i < kPatternCount; i++) {
        *outSumLogLikelihood += outLogLikelihoodsTmp[i] * gPatternWeights[i];
    }

    if (*outSumLogLikelihood != *outSumLogLikelihood)
        returnCode = BEAGLE_ERROR_FLOATING_POINT;

    return returnCode;
}


BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcEdgeLogLikelihoodsFirstDeriv(const int parIndex,
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

                    w += kTransPaddedStateCount;
                }
                v += kPartialsPaddedStateCount;
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

                    // increment for the extra column at the end
                    w += T_PAD;

                    integrationTmp[u] += sumOverJ * partialsParent[v + i] * weight;
                    firstDerivTmp[u] += sumOverJD1 * partialsParent[v + i] * weight;
                    u++;
                }
                v += kPartialsPaddedStateCount;
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

    if (*outSumLogLikelihood != *outSumLogLikelihood)
        returnCode = BEAGLE_ERROR_FLOATING_POINT;

    return returnCode;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcEdgeLogLikelihoodsSecondDeriv(const int parIndex,
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

                    w += kTransPaddedStateCount;
                }
                v += kPartialsPaddedStateCount;
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
                    REALTYPE sumOverJ = 0.0;
                    REALTYPE sumOverJD1 = 0.0;
                    REALTYPE sumOverJD2 = 0.0;
                    for(int j = 0; j < kStateCount; j++) {
                        sumOverJ += transMatrix[w] * partialsChild[v + j];
                        sumOverJD1 += firstDerivMatrix[w] * partialsChild[v + j];
                        sumOverJD2 += secondDerivMatrix[w] * partialsChild[v + j];
                        w++;
                    }

                    // increment for the extra column at the end
                    w += T_PAD;

                    integrationTmp[u] += sumOverJ * partialsParent[v + i] * weight;
                    firstDerivTmp[u] += sumOverJD1 * partialsParent[v + i] * weight;
                    secondDerivTmp[u] += sumOverJD2 * partialsParent[v + i] * weight;
                    u++;
                }
                v += kPartialsPaddedStateCount;
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

    if (*outSumLogLikelihood != *outSumLogLikelihood)
        returnCode = BEAGLE_ERROR_FLOATING_POINT;

    return returnCode;
}



BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::block(void) {
    // Do nothing.
    return BEAGLE_SUCCESS;
}

/*
 * Re-scales the partial likelihoods such that the largest is one.
 */
BEAGLE_CPU_TEMPLATE
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::rescalePartials(REALTYPE* destP,
        REALTYPE* scaleFactors,
        REALTYPE* cumulativeScaleFactors,
        const int  fillWithOnes) {
    if (DEBUGGING_OUTPUT) {
        std::cerr << "destP (before rescale): \n";// << destP << "\n";
        for(int i=0; i<kPartialsSize; i++)
            fprintf(stderr,"destP[%d] = %.5f\n",i,destP[i]);
    }

    // TODO None of the code below has been optimized.
    for (int k = 0; k < kPatternCount; k++) {
        REALTYPE max = 0;
        const int patternOffset = k * kPartialsPaddedStateCount;
        for (int l = 0; l < kCategoryCount; l++) {
            int offset = l * kPaddedPatternCount * kPartialsPaddedStateCount + patternOffset;
            for (int i = 0; i < kStateCount; i++) {
                if(destP[offset] > max)
                    max = destP[offset];
                offset++;
            }
        }

        if (max == 0)
            max = 1.0;

        REALTYPE oneOverMax = REALTYPE(1.0) / max;
        for (int l = 0; l < kCategoryCount; l++) {
            int offset = l * kPaddedPatternCount * kPartialsPaddedStateCount + patternOffset;
            for (int i = 0; i < kStateCount; i++)
                destP[offset++] *= oneOverMax;
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

BEAGLE_CPU_TEMPLATE
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::rescalePartialsByPartition(REALTYPE* destP,
                                                                   REALTYPE* scaleFactors,
                                                                   REALTYPE* cumulativeScaleFactors,
                                                                   const int fillWithOnes,
                                                                   const int partitionIndex) {

    int startPattern = gPatternPartitionsStartPatterns[partitionIndex];
    int endPattern = gPatternPartitionsStartPatterns[partitionIndex + 1];

    // TODO None of the code below has been optimized.
    for (int k = startPattern; k < endPattern; k++) {
        REALTYPE max = 0;
        const int patternOffset = k * kPartialsPaddedStateCount;
        for (int l = 0; l < kCategoryCount; l++) {
            int offset = l * kPaddedPatternCount * kPartialsPaddedStateCount + patternOffset;
            for (int i = 0; i < kStateCount; i++) {
                if(destP[offset] > max)
                    max = destP[offset];
                offset++;
            }
        }

        if (max == 0)
            max = 1.0;

        REALTYPE oneOverMax = REALTYPE(1.0) / max;
        for (int l = 0; l < kCategoryCount; l++) {
            int offset = l * kPaddedPatternCount * kPartialsPaddedStateCount + patternOffset;
            for (int i = 0; i < kStateCount; i++)
                destP[offset++] *= oneOverMax;
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
}

BEAGLE_CPU_TEMPLATE
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::autoRescalePartials(REALTYPE* destP,
                                              signed short* scaleFactors) {


    for (int k = 0; k < kPatternCount; k++) {
        REALTYPE max = 0;
        const int patternOffset = k * kPartialsPaddedStateCount;
        for (int l = 0; l < kCategoryCount; l++) {
            int offset = l * kPaddedPatternCount * kPartialsPaddedStateCount + patternOffset;
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
                int offset = l * kPaddedPatternCount * kPartialsPaddedStateCount + patternOffset;
                for (int i = 0; i < kStateCount; i++)
                    destP[offset++] *= static_cast<REALTYPE>(pow(2.0, -expMax));
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
// private methods


BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::reorderPatternsByPartition() {

    if (!kPatternsReordered) {
        gPatternsNewOrder = (int*) malloc(kPatternCount * sizeof(int));
    } else {
        return BEAGLE_ERROR_NO_IMPLEMENTATION;
    }

    int* partitionSizes = (int*) malloc(kPartitionCount * sizeof(int));
    double* sortedPatternWeights = (double*) malloc(sizeof(double) * kPatternCount);

    for (int i=0; i < kPartitionCount; i++) {
        gPatternPartitionsStartPatterns[i] = 0;
        partitionSizes[i] = 0;
    }

    for (int i=0; i < kPatternCount; i++) {
         gPatternsNewOrder[i] = partitionSizes[gPatternPartitions[i]]++;
    }

    for (int i=0; i < kPartitionCount; i++) {
        for (int j=0; j < i; j++) {
            gPatternPartitionsStartPatterns[i] += partitionSizes[j];
        }
    }
    gPatternPartitionsStartPatterns[kPartitionCount] = kPatternCount;

    for (int i=0; i < kPatternCount; i++) {
        gPatternsNewOrder[i] += gPatternPartitionsStartPatterns[gPatternPartitions[i]];
        sortedPatternWeights[gPatternsNewOrder[i]] = gPatternWeights[i];
    }


    int currentPattern = 0;
    for (int i=0; i < kPartitionCount; i++) {
        for (int j=0; j < partitionSizes[i]; j++) {
            gPatternPartitions[currentPattern++] = i;
        }
    }

    free(partitionSizes);
    free(gPatternWeights);
    gPatternWeights = sortedPatternWeights;

    REALTYPE* sortedPartials = (REALTYPE*) mallocAligned(sizeof(REALTYPE) * kPartialsSize);
    int* sortedTips = (int*) mallocAligned(sizeof(int) * kPaddedPatternCount);

    for (int tip=0; tip < kTipCount; tip++) {
        if (gTipStates[tip] == NULL) {
            REALTYPE* unsortedPartials = gPartials[tip];
            for (int l=0; l < kCategoryCount; l++) {
                for (int i=0; i < kPatternCount; i++) {
                    for (int j=0; j < kStateCount; j++) {
                        int sortIndex = l*kStateCount*kPatternCount + gPatternsNewOrder[i]*kStateCount + j;
                        int pIndex = l*kStateCount*kPatternCount + i*kStateCount + j;
                        sortedPartials[sortIndex] = unsortedPartials[pIndex];
                    }
                }
            }
            gPartials[tip] = sortedPartials;
            sortedPartials = unsortedPartials;
        } else {
            int* unsortedTips = gTipStates[tip];
            for (int i=0; i < kPatternCount; i++) {
                int sortIndex = gPatternsNewOrder[i];
                int pIndex = i;
                sortedTips[sortIndex] = unsortedTips[pIndex];
            }
            gTipStates[tip] = sortedTips;
            sortedTips = unsortedTips;
        }
    }

    free(sortedPartials);
    free(sortedTips);

    kPatternsReordered = true;

    return BEAGLE_SUCCESS;
}


/*
 * Calculates partial likelihoods at a node when both children have states.
 */
BEAGLE_CPU_TEMPLATE
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcStatesStates(REALTYPE* destP,
                                                         const int* states1,
                                                         const REALTYPE* matrices1,
                                                         const int* states2,
                                                         const REALTYPE* matrices2,
                                                         int startPattern,
                                                         int endPattern) {

#pragma omp parallel for num_threads(kCategoryCount)
    for (int l = 0; l < kCategoryCount; l++) {
        int v = l*kPartialsPaddedStateCount*kPatternCount + kPartialsPaddedStateCount*startPattern;
        for (int k = startPattern; k < endPattern; k++) {
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

                w += kTransPaddedStateCount;
            }
            if (P_PAD) {
                for (int pad = 0; pad < P_PAD; pad++)  {
                    destP[v] = 0.0;
                    v++;
                }
            }
        }
    }
}

BEAGLE_CPU_TEMPLATE
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcStatesStatesFixedScaling(REALTYPE* destP,
                                                                     const int* child1States,
                                                                     const REALTYPE* child1TransMat,
                                                                     const int* child2States,
                                                                     const REALTYPE* child2TransMat,
                                                                     const REALTYPE* scaleFactors,
                                                                     int startPattern,
                                                                     int endPattern) {

#pragma omp parallel for num_threads(kCategoryCount)
    for (int l = 0; l < kCategoryCount; l++) {
    int v = l*kPartialsPaddedStateCount*kPatternCount + kPartialsPaddedStateCount*startPattern;
        for (int k = startPattern; k < endPattern; k++) {
            const int state1 = child1States[k];
            const int state2 = child2States[k];
            int w = l * kMatrixSize;
            REALTYPE scaleFactor = scaleFactors[k];
            for (int i = 0; i < kStateCount; i++) {
                destP[v] = child1TransMat[w + state1] *
                           child2TransMat[w + state2] / scaleFactor;
                v++;

                w += kTransPaddedStateCount;
            }
            if (P_PAD) {
                for (int pad = 0; pad < P_PAD; pad++)  {
                    destP[v] = 0.0;
                    v++;
                }
            }
        }
    }
}

/*
 * Calculates partial likelihoods at a node when one child has states and one has partials.
 */
BEAGLE_CPU_TEMPLATE
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcStatesPartials(REALTYPE* destP,
                                                           const int* states1,
                                                           const REALTYPE* matrices1,
                                                           const REALTYPE* partials2,
                                                           const REALTYPE* matrices2,
                                                           int startPattern,
                                                           int endPattern) {

    int matrixIncr = kStateCount;

    // increment for the extra column at the end
    matrixIncr += T_PAD;


    int stateCountModFour = (kStateCount / 4) * 4;

#pragma omp parallel for num_threads(kCategoryCount)
    for (int l = 0; l < kCategoryCount; l++) {
        int v = l*kPartialsPaddedStateCount*kPatternCount + kPartialsPaddedStateCount*startPattern;
        int matrixOffset = l*kMatrixSize;
        const REALTYPE* partials2Ptr = &partials2[v];
        REALTYPE* destPtr = &destP[v];
        for (int k = startPattern; k < endPattern; k++) {
            int w = l * kMatrixSize;
            int state1 = states1[k];
            for (int i = 0; i < kStateCount; i++) {
                const REALTYPE* matrices2Ptr = matrices2 + matrixOffset + i * matrixIncr;
                REALTYPE tmp = matrices1[w + state1];
                REALTYPE sumA = 0.0;
                REALTYPE sumB = 0.0;
                int j = 0;
                for (; j < stateCountModFour; j += 4) {
                    sumA += matrices2Ptr[j + 0] * partials2Ptr[j + 0];
                    sumB += matrices2Ptr[j + 1] * partials2Ptr[j + 1];
                    sumA += matrices2Ptr[j + 2] * partials2Ptr[j + 2];
                    sumB += matrices2Ptr[j + 3] * partials2Ptr[j + 3];
                }
                for (; j < kStateCount; j++) {
                    sumA += matrices2Ptr[j] * partials2Ptr[j];
                }

                w += matrixIncr;

                *(destPtr++) = tmp * (sumA + sumB);
            }
            if (P_PAD) {
                for (int pad = 0; pad < P_PAD; pad++)  {
                    *(destPtr++) = 0.0;
                }
            }
            partials2Ptr += kPartialsPaddedStateCount;
        }
    }
}

BEAGLE_CPU_TEMPLATE
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcStatesPartialsFixedScaling(REALTYPE* destP,
                                                                       const int* states1,
                                                                       const REALTYPE* matrices1,
                                                                       const REALTYPE* partials2,
                                                                       const REALTYPE* matrices2,
                                                                       const REALTYPE* scaleFactors,
                                                                       int startPattern,
                                                                       int endPattern) {

    int matrixIncr = kStateCount;

    // increment for the extra column at the end
    matrixIncr += T_PAD;

    int stateCountModFour = (kStateCount / 4) * 4;

#pragma omp parallel for num_threads(kCategoryCount)
    for (int l = 0; l < kCategoryCount; l++) {
        int v = l*kPartialsPaddedStateCount*kPatternCount + kPartialsPaddedStateCount*startPattern;
        int matrixOffset = l*kMatrixSize;
        const REALTYPE* partials2Ptr = &partials2[v];
        REALTYPE* destPtr = &destP[v];
        for (int k = startPattern; k < endPattern; k++) {
            int w = l * kMatrixSize;
            int state1 = states1[k];
            REALTYPE oneOverScaleFactor = REALTYPE(1.0) / scaleFactors[k];
            for (int i = 0; i < kStateCount; i++) {
                const REALTYPE* matrices2Ptr = matrices2 + matrixOffset + i * matrixIncr;
                REALTYPE tmp = matrices1[w + state1];
                REALTYPE sumA = 0.0;
                REALTYPE sumB = 0.0;
                int j = 0;
                for (; j < stateCountModFour; j += 4) {
                    sumA += matrices2Ptr[j + 0] * partials2Ptr[j + 0];
                    sumB += matrices2Ptr[j + 1] * partials2Ptr[j + 1];
                    sumA += matrices2Ptr[j + 2] * partials2Ptr[j + 2];
                    sumB += matrices2Ptr[j + 3] * partials2Ptr[j + 3];
                }
                for (; j < kStateCount; j++) {
                    sumA += matrices2Ptr[j] * partials2Ptr[j];
                }

                w += matrixIncr;

                *(destPtr++) = tmp * (sumA + sumB) * oneOverScaleFactor;
            }
            if (P_PAD) {
                for (int pad = 0; pad < P_PAD; pad++)  {
                    *(destPtr++) = 0.0;
                }
            }
            partials2Ptr += kPartialsPaddedStateCount;
        }
    }
}

BEAGLE_CPU_TEMPLATE
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcPreStatesPartialsFixedScaling(REALTYPE* destP,
                                                                       const int* states1,
                                                                       const REALTYPE* matrices1,
                                                                       const REALTYPE* partials2,
                                                                       const REALTYPE* matrices2,
                                                                       const REALTYPE* scaleFactors,
                                                                       int startPattern,
                                                                       int endPattern) {

    int matrixIncr = kStateCount;

    // increment for the extra column at the end
    matrixIncr += T_PAD;

    int stateCountModFour = (kStateCount / 4) * 4;
//
//#pragma omp parallel for num_threads(kCategoryCount)
//    for (int l = 0; l < kCategoryCount; l++) {
//        int v = l*kPartialsPaddedStateCount*kPatternCount + kPartialsPaddedStateCount*startPattern;
//        int matrixOffset = l*kMatrixSize;
//        const REALTYPE* partials2Ptr = &partials2[v];
//        REALTYPE* destPtr = &destP[v];
//        for (int k = startPattern; k < endPattern; k++) {
//            int w = l * kMatrixSize;
//            int state1 = states1[k];
//            REALTYPE oneOverScaleFactor = REALTYPE(1.0) / scaleFactors[k];
//            for (int i = 0; i < kStateCount; i++) {
//                const REALTYPE* matrices2Ptr = matrices2 + matrixOffset + i * matrixIncr;
//                REALTYPE tmp = matrices1[w + state1];
//                REALTYPE sumA = 0.0;
//                REALTYPE sumB = 0.0;
//                int j = 0;
//                for (; j < stateCountModFour; j += 4) {
//                    sumA += matrices2Ptr[j + 0] * partials2Ptr[j + 0];
//                    sumB += matrices2Ptr[j + 1] * partials2Ptr[j + 1];
//                    sumA += matrices2Ptr[j + 2] * partials2Ptr[j + 2];
//                    sumB += matrices2Ptr[j + 3] * partials2Ptr[j + 3];
//                }
//                for (; j < kStateCount; j++) {
//                    sumA += matrices2Ptr[j] * partials2Ptr[j];
//                }
//
//                w += matrixIncr;
//
//                *(destPtr++) = tmp * (sumA + sumB) * oneOverScaleFactor;
//            }
//            destPtr += P_PAD;
//            partials2Ptr += kPartialsPaddedStateCount;
//        }
//    }
}

/*
 * Calculates partial likelihoods at a node when both children have partials.
 */
BEAGLE_CPU_TEMPLATE
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcPartialsPartials(REALTYPE* destP,
                                                             const REALTYPE* partials1,
                                                             const REALTYPE* matrices1,
                                                             const REALTYPE* partials2,
                                                             const REALTYPE* matrices2,
                                                             int startPattern,
                                                             int endPattern) {
    int matrixIncr = kStateCount;

    // increment for the extra column at the end
    matrixIncr += T_PAD;

    int stateCountModFour = (kStateCount / 4) * 4;

#pragma omp parallel for num_threads(kCategoryCount)
    for (int l = 0; l < kCategoryCount; l++) {
        int v = l*kPartialsPaddedStateCount*kPatternCount + kPartialsPaddedStateCount*startPattern;
        int matrixOffset = l*kMatrixSize;
        const REALTYPE* partials1Ptr = &partials1[v];
        const REALTYPE* partials2Ptr = &partials2[v];
        REALTYPE* destPtr = &destP[v];
        for (int k = startPattern; k < endPattern; k++) {

            for (int i = 0; i < kStateCount; i++) {
                const REALTYPE* matrices1Ptr = matrices1 + matrixOffset + i * matrixIncr;
                const REALTYPE* matrices2Ptr = matrices2 + matrixOffset + i * matrixIncr;
                REALTYPE sum1A = 0.0, sum2A = 0.0;
                REALTYPE sum1B = 0.0, sum2B = 0.0;
                int j = 0;
                for (; j < stateCountModFour; j += 4) {
                    sum1A += matrices1Ptr[j + 0] * partials1Ptr[j + 0];
                    sum2A += matrices2Ptr[j + 0] * partials2Ptr[j + 0];

                    sum1B += matrices1Ptr[j + 1] * partials1Ptr[j + 1];
                    sum2B += matrices2Ptr[j + 1] * partials2Ptr[j + 1];

                    sum1A += matrices1Ptr[j + 2] * partials1Ptr[j + 2];
                    sum2A += matrices2Ptr[j + 2] * partials2Ptr[j + 2];

                    sum1B += matrices1Ptr[j + 3] * partials1Ptr[j + 3];
                    sum2B += matrices2Ptr[j + 3] * partials2Ptr[j + 3];
                }

                for (; j < kStateCount; j++) {
                    sum1A += matrices1Ptr[j] * partials1Ptr[j];
                    sum2A += matrices2Ptr[j] * partials2Ptr[j];
                }

                *(destPtr++) = (sum1A + sum1B) * (sum2A + sum2B);
            }
            destPtr += P_PAD;
            partials1Ptr += kPartialsPaddedStateCount;
            partials2Ptr += kPartialsPaddedStateCount;
        }
    }
}

/*
 * Calculates Pre-order partial likelihoods at a node when both parent and sibling have partials.
 */
BEAGLE_CPU_TEMPLATE
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcPrePartialsPartials(REALTYPE* destP,
                                                                const REALTYPE* partials1,
                                                                const REALTYPE* matrices1,
                                                                const REALTYPE* partials2,
                                                                const REALTYPE* matrices2,
                                                                int startPattern,
                                                                int endPattern) {
    int matrixIncr = kStateCount;

    // increment for the extra column at the end
    matrixIncr += T_PAD;

    int stateCountModFour = (kStateCount / 4) * 4;
    REALTYPE* tmpdestPtr = destP;
    //clean up the partial first, set every entry to 0
//    std::fill(tmpdestPtr, tmpdestPtr + kPartialsSize, 0);

#pragma omp parallel for num_threads(kCategoryCount)
    for (int l = 0; l < kCategoryCount; l++) {
        int v = l*kPartialsPaddedStateCount*kPaddedPatternCount + kPartialsPaddedStateCount*startPattern;
        std::fill(destP + v, destP + l*kPartialsPaddedStateCount*kPaddedPatternCount + kPartialsPaddedStateCount*endPattern, 0);
        int matrixOffset = l*kMatrixSize;
        const REALTYPE* partials1Ptr = &partials1[v];
        const REALTYPE* partials2Ptr = &partials2[v];
        REALTYPE* destPtr = &destP[v];
        tmpdestPtr = destPtr;
        for (int k = startPattern; k < endPattern; k++) {
            for (int i = 0; i < kStateCount; i++) {
                const REALTYPE* matrices1Ptr = matrices1 + matrixOffset + i * matrixIncr;
                const REALTYPE* matrices2Ptr = matrices2 + matrixOffset + i * matrixIncr;
                REALTYPE sum2A = 0.0, sum2B = 0.0;
                int j = 0;
                for (; j < stateCountModFour; j += 4) {
                    sum2A += matrices2Ptr[j + 0] * partials2Ptr[j + 0];

                    sum2B += matrices2Ptr[j + 1] * partials2Ptr[j + 1];

                    sum2A += matrices2Ptr[j + 2] * partials2Ptr[j + 2];

                    sum2B += matrices2Ptr[j + 3] * partials2Ptr[j + 3];
                }

                for (; j < kStateCount; j++) {
                    sum2A += matrices2Ptr[j] * partials2Ptr[j];
                }

                // sum2A + sum2B = M_j P_j
                // Now 2nd loop
                tmpdestPtr = destPtr;
                REALTYPE  MjPj = (sum2A + sum2B) * partials1Ptr[i];

                for (j = 0; j < stateCountModFour; j += 4) {
                    *(tmpdestPtr++) += matrices1Ptr[j + 0] * MjPj;

                    *(tmpdestPtr++) += matrices1Ptr[j + 1] * MjPj;

                    *(tmpdestPtr++) += matrices1Ptr[j + 2] * MjPj;

                    *(tmpdestPtr++) += matrices1Ptr[j + 3] * MjPj;
                }

                for (; j < kStateCount; j++) {
                    *(tmpdestPtr++) += matrices1Ptr[j] * MjPj;
                }
            }
            destPtr +=kPartialsPaddedStateCount;
            partials1Ptr += kPartialsPaddedStateCount;
            partials2Ptr += kPartialsPaddedStateCount;
        }
    }
}

BEAGLE_CPU_TEMPLATE
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcPrePartialsPartialsTopRoot(REALTYPE* destP,
                                                                const REALTYPE* partials1,
                                                                const REALTYPE* matrices1,
                                                                const REALTYPE* partials2,
                                                                const REALTYPE* matrices2,
                                                                int startPattern,
                                                                int endPattern) {
    int matrixIncr = kStateCount;

    // increment for the extra column at the end
    matrixIncr += T_PAD;

    int stateCountModFour = (kStateCount / 4) * 4;

#pragma omp parallel for num_threads(kCategoryCount)
    for (int l = 0; l < kCategoryCount; l++) {
        int v = l*kPartialsPaddedStateCount*kPaddedPatternCount + kPartialsPaddedStateCount*startPattern;
        int matrixOffset = l*kMatrixSize;
        const REALTYPE* partials1Ptr = &partials1[v];
        const REALTYPE* partials2Ptr = &partials2[v];
        REALTYPE* destPtr = &destP[v];
        for (int k = startPattern; k < endPattern; k++) {
            for (int i = 0; i < kStateCount; i++) {
                const REALTYPE* matrices2Ptr = matrices2 + matrixOffset + i * matrixIncr;

                REALTYPE sum2A = 0.0, sum2B = 0.0;
                int j = 0;
                for (; j < stateCountModFour; j += 4) {
                    sum2A += matrices2Ptr[j + 0] * partials2Ptr[j + 0];
                    sum2B += matrices2Ptr[j + 1] * partials2Ptr[j + 1];
                    sum2A += matrices2Ptr[j + 2] * partials2Ptr[j + 2];
                    sum2B += matrices2Ptr[j + 3] * partials2Ptr[j + 3];
                }

                for (; j < kStateCount; j++) {
                    sum2A += matrices2Ptr[j] * partials2Ptr[j];
                }

                *(destPtr++) = (sum2A + sum2B) * partials1Ptr[i];
            }

            destPtr += P_PAD;
            partials1Ptr += kPartialsPaddedStateCount;
            partials2Ptr += kPartialsPaddedStateCount;
        }
    }
}

#define PRE_CACHE_FRIENDLY

BEAGLE_CPU_TEMPLATE
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcPrePartialsPartialsTop(REALTYPE* destP,
                                                                const REALTYPE* partials1,
                                                                const REALTYPE* matrices1,
                                                                const REALTYPE* partials2,
                                                                const REALTYPE* matrices2,
                                                                int startPattern,
                                                                int endPattern) {
    const int matrixIncr = kStateCount + T_PAD;
    const int stateCountModFour = (kStateCount / 4) * 4;

#pragma omp parallel for num_threads(kCategoryCount)
    for (int l = 0; l < kCategoryCount; l++) {
        int v = l * kPartialsPaddedStateCount * kPatternCount + kPartialsPaddedStateCount * startPattern;
#ifdef PRE_CACHE_FRIENDLY
        // only these patterns: other pattern partitions may be computed at the same time
        std::fill(destP + v, destP + v + kPartialsPaddedStateCount * (endPattern - startPattern), 0);
#endif
        int matrixOffset = l * kMatrixSize;
        const REALTYPE* partials1Ptr = &partials1[v];
        const REALTYPE* partials2Ptr = &partials2[v];
        REALTYPE* destPtr = &destP[v];

        for (int k = startPattern; k < endPattern; k++) {
            REALTYPE* destPtri = destPtr;
#ifdef PRE_CACHE_FRIENDLY
            for (int i = 0; i < kStateCount; i++) {
                // scatter (M2[i,:].P2) * P1[i] via row i of M1 — contiguous
                const REALTYPE Pk = partials1Ptr[i];
                const REALTYPE* m1row = matrices1 + matrixOffset + i * matrixIncr;
                REALTYPE* tmpdestPtr = destPtr;
                int j = 0;
                for (; j < stateCountModFour; j += 4) {
                    *(tmpdestPtr++) += m1row[j + 0] * Pk;
                    *(tmpdestPtr++) += m1row[j + 1] * Pk;
                    *(tmpdestPtr++) += m1row[j + 2] * Pk;
                    *(tmpdestPtr++) += m1row[j + 3] * Pk;
                }

                for (; j < kStateCount; j++) {
                    *(tmpdestPtr++) += m1row[j] * Pk;
                }
            }

            for (int i = 0; i < kStateCount; i++) {
                // sum2 = M2[i,:] . P2 — row i of M2 is contiguous
                const REALTYPE* matrices2Ptr = matrices2 + matrixOffset + i * matrixIncr;
                REALTYPE sum2A = 0.0, sum2B = 0.0;
                int j = 0;
                for (; j < stateCountModFour; j += 4) {
                    sum2A += matrices2Ptr[j + 0] * partials2Ptr[j + 0];
                    sum2B += matrices2Ptr[j + 1] * partials2Ptr[j + 1];
                    sum2A += matrices2Ptr[j + 2] * partials2Ptr[j + 2];
                    sum2B += matrices2Ptr[j + 3] * partials2Ptr[j + 3];
                }

                for (; j < kStateCount; j++) {
                    sum2A += matrices2Ptr[j] * partials2Ptr[j];
                }

                *(destPtri++) *= (sum2A + sum2B);
            }
#else
            for (int i = 0; i < kStateCount; i++) {
                // sum2 = M2[i,:] . P2 — row i of M2 is contiguous
                const REALTYPE* matrices2Ptr = matrices2 + matrixOffset + i * matrixIncr;
                REALTYPE sum2A = 0.0, sum2B = 0.0;
                int j = 0;
                for (; j < stateCountModFour; j += 4) {
                    sum2A += matrices2Ptr[j + 0] * partials2Ptr[j + 0];
                    sum2B += matrices2Ptr[j + 1] * partials2Ptr[j + 1];
                    sum2A += matrices2Ptr[j + 2] * partials2Ptr[j + 2];
                    sum2B += matrices2Ptr[j + 3] * partials2Ptr[j + 3];
                }

                for (; j < kStateCount; j++) {
                    sum2A += matrices2Ptr[j] * partials2Ptr[j];
                }

                REALTYPE sum1A = 0.0;

                for (j = 0; j < kStateCount; ++j) {
                    sum1A += matrices1[matrixOffset + j * matrixIncr + i] * partials1Ptr[j];
                }

                *(destPtri++) = (sum2A + sum2B) * sum1A;
            }
#endif

            destPtr      += kPartialsPaddedStateCount;
            partials1Ptr += kPartialsPaddedStateCount;
            partials2Ptr += kPartialsPaddedStateCount;
        }
    }
}

/*
* Calculates Pre-order partial likelihoods at a node when sibling node being a tip.
*/
BEAGLE_CPU_TEMPLATE
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcPrePartialsStates(REALTYPE* destP,
                                                              const REALTYPE* partials1,
                                                              const REALTYPE* matrices1,
                                                              const int* states2,
                                                              const REALTYPE* matrices2,
                                                              int startPattern,
                                                              int endPattern) {
    int matrixIncr = kStateCount;

    // increment for the extra column at the end
    matrixIncr += T_PAD;

    int stateCountModFour = (kStateCount / 4) * 4;
    REALTYPE* tmpdestPtr = destP;
    //clean up the partial first, set every entry to 0
//    std::fill(tmpdestPtr, tmpdestPtr + kPartialsSize, 0);

#pragma omp parallel for num_threads(kCategoryCount)
    for (int l = 0; l < kCategoryCount; l++) {
        int v = l*kPartialsPaddedStateCount*kPaddedPatternCount + kPartialsPaddedStateCount*startPattern;
        std::fill(destP + v, destP + l*kPartialsPaddedStateCount*kPaddedPatternCount + kPartialsPaddedStateCount*endPattern, 0);
        int matrixOffset = l*kMatrixSize;
        const REALTYPE* partials1Ptr = &partials1[v];

        REALTYPE* destPtr = &destP[v];
        tmpdestPtr = destPtr;
        for (int k = startPattern; k < endPattern; k++) {
            int w = l * kMatrixSize;
            int state2 = states2[k];
            for (int i = 0; i < kStateCount; i++) {
                const REALTYPE* matrices1Ptr = matrices1 + matrixOffset + i * matrixIncr;

                tmpdestPtr = destPtr;
                const REALTYPE  MjPj = matrices2[w + state2] * partials1Ptr[i];

                int j = 0;
                for (; j < stateCountModFour; j += 4) {
                    *(tmpdestPtr++) += matrices1Ptr[j + 0] * MjPj;

                    *(tmpdestPtr++) += matrices1Ptr[j + 1] * MjPj;

                    *(tmpdestPtr++) += matrices1Ptr[j + 2] * MjPj;

                    *(tmpdestPtr++) += matrices1Ptr[j + 3] * MjPj;
                }

                for (; j < kStateCount; j++) {
                    *(tmpdestPtr++) += matrices1Ptr[j] * MjPj;
                }

                w += matrixIncr;
            }
            destPtr +=kPartialsPaddedStateCount;
            partials1Ptr += kPartialsPaddedStateCount;
        }
    }
}

BEAGLE_CPU_TEMPLATE
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcPrePartialsStatesTop(REALTYPE* destP,
                                                              const REALTYPE* partials1,
                                                              const REALTYPE* matrices1,
                                                              const int* states2,
                                                              const REALTYPE* matrices2,
                                                              int startPattern,
                                                              int endPattern) {
    int matrixIncr = kStateCount + T_PAD;
    int stateCountModFour = (kStateCount / 4) * 4;

#pragma omp parallel for num_threads(kCategoryCount)
    for (int l = 0; l < kCategoryCount; l++) {
        int v = l*kPartialsPaddedStateCount*kPatternCount + kPartialsPaddedStateCount*startPattern;
#ifdef PRE_CACHE_FRIENDLY
        // only these patterns: other pattern partitions may be computed at the same time
        std::fill(destP + v, destP + v + kPartialsPaddedStateCount * (endPattern - startPattern), 0);
#endif
        int matrixOffset = l*kMatrixSize;
        const REALTYPE* partials1Ptr = &partials1[v];
        REALTYPE* destPtr = &destP[v];

        for (int k = startPattern; k < endPattern; k++) {
            REALTYPE* destPtri = destPtr;
            int w = l * kMatrixSize;
            int state2 = states2[k];
#ifdef PRE_CACHE_FRIENDLY
            for (int i = 0; i < kStateCount; i++) {
                // scatter (M2[i,:].P2) * P1[i] via row i of M1 — contiguous
                const REALTYPE Pk = partials1Ptr[i];
                const REALTYPE* m1row = matrices1 + matrixOffset + i * matrixIncr;
                REALTYPE* tmpdestPtr = destPtr;
                int j = 0;
                for (; j < stateCountModFour; j += 4) {
                    *(tmpdestPtr++) += m1row[j + 0] * Pk;
                    *(tmpdestPtr++) += m1row[j + 1] * Pk;
                    *(tmpdestPtr++) += m1row[j + 2] * Pk;
                    *(tmpdestPtr++) += m1row[j + 3] * Pk;
                }

                for (; j < kStateCount; j++) {
                    *(tmpdestPtr++) += m1row[j] * Pk;
                }
            }

            const REALTYPE sum2A = matrices2[w + state2];
            for (int i = 0; i < kStateCount; i++) {
                const REALTYPE sum2A = matrices2[w + state2];
                *(destPtri++) *= sum2A;

                w += matrixIncr;
            }
#else
            for (int i = 0; i < kStateCount; i++) {
                const REALTYPE* matrices1Ptr = matrices1 + matrixOffset + i * matrixIncr;

                const REALTYPE sum2A = matrices2[w + state2];

                REALTYPE sum1A = 0.0;
                int j;
                for (j = 0; j < kStateCount; ++j) {
                    sum1A += matrices1[matrixOffset + j * matrixIncr + i] * partials1Ptr[j];
                }

                *(destPtri++) = sum2A * sum1A;

                w += matrixIncr;
            }
#endif
            destPtr +=kPartialsPaddedStateCount;
            partials1Ptr += kPartialsPaddedStateCount;
        }
    }
}

BEAGLE_CPU_TEMPLATE
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcPrePartialsStatesTopRoot(REALTYPE* destP,
                                                              const REALTYPE* partials1,
                                                              const REALTYPE* matrices1,
                                                              const int* states2,
                                                              const REALTYPE* matrices2,
                                                              int startPattern,
                                                              int endPattern) {
    int matrixIncr = kStateCount + T_PAD;
    int stateCountModFour = (kStateCount / 4) * 4;

#pragma omp parallel for num_threads(kCategoryCount)
    for (int l = 0; l < kCategoryCount; l++) {
        int v = l*kPartialsPaddedStateCount*kPaddedPatternCount + kPartialsPaddedStateCount*startPattern;
        const REALTYPE* partials1Ptr = &partials1[v];

        REALTYPE* destPtr = &destP[v];

        for (int k = startPattern; k < endPattern; k++) {
            int w = l * kMatrixSize;
            int state2 = states2[k];
            for (int i = 0; i < kStateCount; i++) {

                destPtr[i] = matrices2[w + state2] * partials1Ptr[i];

                w += matrixIncr;
            }
            destPtr += kPartialsPaddedStateCount;
            partials1Ptr += kPartialsPaddedStateCount;
        }
    }
}

BEAGLE_CPU_TEMPLATE
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcStatesDegree2(REALTYPE* destP,
                                                          const int* states1,
                                                          const REALTYPE* matrices1,
                                                          const REALTYPE* scaleFactors,
                                                          int startPattern,
                                                          int endPattern) {
    if (scaleFactors != NULL) {
        calcDegree2Partials<States, true>(destP, states1, NULL, matrices1, scaleFactors, startPattern, endPattern);
    } else {
        calcDegree2Partials<States, false>(destP, states1, NULL, matrices1, NULL, startPattern, endPattern);
    }
}

BEAGLE_CPU_TEMPLATE
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcPartialsDegree2(REALTYPE* destP,
                                                            const REALTYPE* partials1,
                                                            const REALTYPE* matrices1,
                                                            const REALTYPE* scaleFactors,
                                                            int startPattern,
                                                            int endPattern) {
    if (scaleFactors != NULL) {
        calcDegree2Partials<Partials, true>(destP, NULL, partials1, matrices1, scaleFactors, startPattern, endPattern);
    } else {
        calcDegree2Partials<Partials, false>(destP, NULL, partials1, matrices1, NULL, startPattern, endPattern);
    }
}

/*
 * Calculates partial likelihoods at a degree-2 node, whose only child has states or partials.
 */
BEAGLE_CPU_TEMPLATE template <typename First, bool FixedScaling>
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcDegree2Partials(REALTYPE* destP,
                                                            const int* states1,
                                                            const REALTYPE* partials1,
                                                            const REALTYPE* matrices1,
                                                            const REALTYPE* scaleFactors,
                                                            int startPattern,
                                                            int endPattern) {

    static_assert(std::is_same_v<First, States> || std::is_same_v<First, Partials>, "Unsupported child type");

    const int matrixIncr = kStateCount + T_PAD;
    const int stateCountModFour = (kStateCount / 4) * 4;

#pragma omp parallel for num_threads(kCategoryCount)
    for (int l = 0; l < kCategoryCount; l++) {
        const int v = l*kPartialsPaddedStateCount*kPaddedPatternCount + kPartialsPaddedStateCount*startPattern;
        const REALTYPE* matrices1Ptr = matrices1 + l*kMatrixSize;
        const REALTYPE* partials1Ptr = std::is_same_v<First, Partials> ? &partials1[v] : nullptr;
        REALTYPE* destPtr = &destP[v];

        for (int k = startPattern; k < endPattern; k++) {

            REALTYPE oneOverScaleFactor = REALTYPE(1.0);
            if constexpr (FixedScaling) {
                oneOverScaleFactor = REALTYPE(1.0) / scaleFactors[k];
            }

            for (int i = 0; i < kStateCount; i++) {
                const REALTYPE* row = matrices1Ptr + i * matrixIncr;
                REALTYPE sum;
                if constexpr (std::is_same_v<First, States>) {
                    const int state1 = states1[k];
                    sum = (state1 < kStateCount) ? row[state1] : REALTYPE(1.0); // ambiguous state
                } else {
                    REALTYPE sumA = 0.0, sumB = 0.0;
                    int j = 0;
                    for (; j < stateCountModFour; j += 4) {
                        sumA += row[j + 0] * partials1Ptr[j + 0];
                        sumB += row[j + 1] * partials1Ptr[j + 1];
                        sumA += row[j + 2] * partials1Ptr[j + 2];
                        sumB += row[j + 3] * partials1Ptr[j + 3];
                    }
                    for (; j < kStateCount; j++) {
                        sumA += row[j] * partials1Ptr[j];
                    }
                    sum = sumA + sumB;
                }
                if constexpr (FixedScaling) {
                    sum *= oneOverScaleFactor;
                }
                *(destPtr++) = sum;
            }
            for (int pad = 0; pad < P_PAD; pad++) {
                *(destPtr++) = 0.0;
            }
            if constexpr (std::is_same_v<First, Partials>) {
                partials1Ptr += kPartialsPaddedStateCount;
            }
        }
    }
}

/*
 * Calculates pre-order partial likelihoods through a single branch, from a parent with no sibling.
 */
BEAGLE_CPU_TEMPLATE
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcDegree2PrePartials(REALTYPE* destP,
                                                               const REALTYPE* partials1,
                                                               const REALTYPE* matrices1,
                                                               int startPattern,
                                                               int endPattern) {
    if (matrices1 == NULL) { // top partials below the root: nothing to propagate through
        for (int l = 0; l < kCategoryCount; l++) {
            const int v = l*kPartialsPaddedStateCount*kPaddedPatternCount + kPartialsPaddedStateCount*startPattern;
            std::copy(partials1 + v, partials1 + v + kPartialsPaddedStateCount*(endPattern - startPattern), destP + v);
        }
        return;
    }

    const int matrixIncr = kStateCount + T_PAD;
    const int stateCountModFour = (kStateCount / 4) * 4;

#pragma omp parallel for num_threads(kCategoryCount)
    for (int l = 0; l < kCategoryCount; l++) {
        const int v = l*kPartialsPaddedStateCount*kPaddedPatternCount + kPartialsPaddedStateCount*startPattern;
        std::fill(destP + v, destP + v + kPartialsPaddedStateCount*(endPattern - startPattern), 0);
        const REALTYPE* matrices1Ptr = matrices1 + l*kMatrixSize;
        const REALTYPE* partials1Ptr = &partials1[v];
        REALTYPE* destPtr = &destP[v];

        for (int k = startPattern; k < endPattern; k++) {
            for (int i = 0; i < kStateCount; i++) {
                // scatter P1[i] via row i of M1 -- contiguous
                const REALTYPE Pk = partials1Ptr[i];
                const REALTYPE* row = matrices1Ptr + i * matrixIncr;
                REALTYPE* tmpdestPtr = destPtr;
                int j = 0;
                for (; j < stateCountModFour; j += 4) {
                    *(tmpdestPtr++) += row[j + 0] * Pk;
                    *(tmpdestPtr++) += row[j + 1] * Pk;
                    *(tmpdestPtr++) += row[j + 2] * Pk;
                    *(tmpdestPtr++) += row[j + 3] * Pk;
                }
                for (; j < kStateCount; j++) {
                    *(tmpdestPtr++) += row[j] * Pk;
                }
            }
            destPtr += kPartialsPaddedStateCount;
            partials1Ptr += kPartialsPaddedStateCount;
        }
    }
}

BEAGLE_CPU_TEMPLATE
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcPartialsPartialsFixedScaling(REALTYPE* destP,
                                                                         const REALTYPE* partials1,
                                                                         const REALTYPE* matrices1,
                                                                         const REALTYPE* partials2,
                                                                         const REALTYPE* matrices2,
                                                                         const REALTYPE* scaleFactors,
                                                                         int startPattern,
                                                                         int endPattern) {

    int matrixIncr = kStateCount;

    // increment for the extra column at the end
    matrixIncr += T_PAD;

    int stateCountModFour = (kStateCount / 4) * 4;

#pragma omp parallel for num_threads(kCategoryCount)
    for (int l = 0; l < kCategoryCount; l++) {
        int v = l*kPartialsPaddedStateCount*kPatternCount + kPartialsPaddedStateCount*startPattern;
        int matrixOffset = l*kMatrixSize;
        const REALTYPE* partials1Ptr = &partials1[v];
        const REALTYPE* partials2Ptr = &partials2[v];
        REALTYPE* destPtr = &destP[v];
        for (int k = startPattern; k < endPattern; k++) {
            REALTYPE oneOverScaleFactor = REALTYPE(1.0) / scaleFactors[k];
            for (int i = 0; i < kStateCount; i++) {
                const REALTYPE* matrices1Ptr = matrices1 + matrixOffset + i * matrixIncr;
                const REALTYPE* matrices2Ptr = matrices2 + matrixOffset + i * matrixIncr;
                REALTYPE sum1A = 0.0, sum2A = 0.0;
                REALTYPE sum1B = 0.0, sum2B = 0.0;
                int j = 0;
                for (; j < stateCountModFour; j += 4) {
                    sum1A += matrices1Ptr[j + 0] * partials1Ptr[j + 0];
                    sum2A += matrices2Ptr[j + 0] * partials2Ptr[j + 0];

                    sum1B += matrices1Ptr[j + 1] * partials1Ptr[j + 1];
                    sum2B += matrices2Ptr[j + 1] * partials2Ptr[j + 1];

                    sum1A += matrices1Ptr[j + 2] * partials1Ptr[j + 2];
                    sum2A += matrices2Ptr[j + 2] * partials2Ptr[j + 2];

                    sum1B += matrices1Ptr[j + 3] * partials1Ptr[j + 3];
                    sum2B += matrices2Ptr[j + 3] * partials2Ptr[j + 3];
                }

                for (; j < kStateCount; j++) {
                    sum1A += matrices1Ptr[j] * partials1Ptr[j];
                    sum2A += matrices2Ptr[j] * partials2Ptr[j];
                }

                *(destPtr++) = (sum1A + sum1B) * (sum2A + sum2B) * oneOverScaleFactor;
            }
            destPtr += P_PAD;
            partials1Ptr += kPartialsPaddedStateCount;
            partials2Ptr += kPartialsPaddedStateCount;
        }
    }
}

BEAGLE_CPU_TEMPLATE
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcPartialsPartialsAutoScaling(REALTYPE* destP,
                                                               const REALTYPE* partials1,
                                                               const REALTYPE* matrices1,
                                                               const REALTYPE* partials2,
                                                               const REALTYPE* matrices2,
                                                               int* activateScaling) {

#pragma omp parallel for num_threads(kCategoryCount)
    for (int l = 0; l < kCategoryCount; l++) {
        int u = l*kPartialsPaddedStateCount*kPatternCount;
        int v = l*kPartialsPaddedStateCount*kPatternCount;
        for (int k = 0; k < kPatternCount; k++) {
            int w = l * kMatrixSize;
            for (int i = 0; i < kStateCount; i++) {
                REALTYPE sum1 = 0.0, sum2 = 0.0;
                for (int j = 0; j < kStateCount; j++) {
                    sum1 += matrices1[w] * partials1[v + j];
                    sum2 += matrices2[w] * partials2[v + j];
                    w++;
                }

                // increment for the extra column at the end
                w += T_PAD;

                destP[u] = sum1 * sum2;

                if (*activateScaling == 0) {
                    int expTmp;
                    frexp(destP[u], &expTmp);
                    if (abs(expTmp) > scalingExponentThreshold)
                        *activateScaling = 1;
                }

                u++;
            }
            u += P_PAD;
            v += kPartialsPaddedStateCount;
        }
    }
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::getPaddedPatternsModulus() {
    // Padding only necessary for SSE implementations that vectorize across patterns
    return 1;  // No padding
}

BEAGLE_CPU_TEMPLATE
void* BeagleCPUImpl<BEAGLE_CPU_GENERIC>::mallocAligned(size_t size) {
    void *ptr = (void *) NULL;

#if defined (__APPLE__) || defined(WIN32)
    /*
     presumably malloc on OS X always returns
     a 16-byte aligned pointer
     */
    /* Windows malloc() always gives 16-byte alignment */
    assert(size > 0);
    ptr = malloc(size);
    if(ptr == (void*)NULL) {
        assert(0);
    }
#else
    #if (T_PAD == 1)
        const size_t align = 32;
    #else // T_PAD == 2
        const size_t align = 32; // Changed from 16 (under SSE) to ensure AVX alignment
    #endif
    int res;
    res = posix_memalign(&ptr, align, size);
    if (res != 0) {
        assert(0);
    }
#endif

    return ptr;
}

// allocates the zero-filled eigen-info chunk for matrices [32 * chunkIndex, 32 * chunkIndex + 32); NULL if out of memory
BEAGLE_CPU_TEMPLATE
REALTYPE* BeagleCPUImpl<BEAGLE_CPU_GENERIC>::allocateEigenInfoChunk(int chunkIndex) {
    REALTYPE* chunk = (REALTYPE*) mallocAligned(kEigenInfoChunkSize);
    if (chunk == NULL)
        return NULL;
    memset(chunk, 0, kEigenInfoChunkSize);
    gEigenInfoChunks[chunkIndex] = chunk;
    return chunk;
}

BEAGLE_CPU_TEMPLATE
REALTYPE* BeagleCPUImpl<BEAGLE_CPU_GENERIC>::writableMatrix(int matrixIndex) {
    if (gTransitionMatrices[matrixIndex] == NULL)
        gTransitionMatrices[matrixIndex] = (REALTYPE*) mallocAligned(sizeof(REALTYPE) * kMatrixSize * kCategoryCount);
    return gTransitionMatrices[matrixIndex];
}

BEAGLE_CPU_TEMPLATE
bool BeagleCPUImpl<BEAGLE_CPU_GENERIC>::missingDenseMatrix(const int* matrixIndices, int count) {
    if (!denseMatricesOnDemand())
        return false;
    for (int i = 0; i < count; i++) {
        if (gTransitionMatrices[matrixIndices[i]] == NULL)
            return true;
    }
    return false;
}

#ifdef BEAGLE_INDEX_CHECKS
BEAGLE_CPU_TEMPLATE
bool BeagleCPUImpl<BEAGLE_CPU_GENERIC>::badScale(int index) const {
    const int scaleBufferEntries = (kFlags & BEAGLE_FLAG_SCALING_AUTO) ? 1 : kScaleBufferCount;
    return index < 0 || index >= scaleBufferEntries;
}

BEAGLE_CPU_TEMPLATE
bool BeagleCPUImpl<BEAGLE_CPU_GENERIC>::badMatrices(const int* indices, int count) const {
    for (int i = 0; i < count; i++) {
        if (badMatrix(indices[i]))
            return true;
    }
    return false;
}

BEAGLE_CPU_TEMPLATE
bool BeagleCPUImpl<BEAGLE_CPU_GENERIC>::badPartialsIndices(const int* indices, int count) const {
    for (int i = 0; i < count; i++) {
        if (badPartials(indices[i]))
            return true;
    }
    return false;
}

// calculateRootLogLikelihoods. count == 1 reads the caller's scale index only under MANUAL or DYNAMIC scaling or with
// auto root partitioning; under ALWAYS it uses the root's own scale buffer at bufferIndex - kTipCount. A negative
// index is skipped by the generic root kernels, but read by the 4-state ones unless it is BEAGLE_OP_NONE; the
// per-category path (a negative category weights index) is generic everywhere. count != 1 reads every scale index
// unless the first is BEAGLE_OP_NONE.
BEAGLE_CPU_TEMPLATE
bool BeagleCPUImpl<BEAGLE_CPU_GENERIC>::badRootIndices(const int* bufferIndices, const int* categoryWeightsIndices,
                                                       const int* cumulativeScaleIndices, int count) const {
    const bool autoOrAlways = kFlags & (BEAGLE_FLAG_SCALING_AUTO | BEAGLE_FLAG_SCALING_ALWAYS);
    if (count == 1) {
        if (badPartials(bufferIndices[0]))
            return true;
        const bool readsNegative = rootReadsNegativeScale() &&
                                   (kAutoRootPartitioningEnabled || categoryWeightsIndices[0] >= 0);
        if (kAutoRootPartitioningEnabled || !autoOrAlways)
            return badRootScale(cumulativeScaleIndices[0], readsNegative);
        if (kFlags & BEAGLE_FLAG_SCALING_ALWAYS)
            return badRootScale(bufferIndices[0] - kTipCount, readsNegative);
        return false;
    }
    for (int i = 0; i < count; i++) {
        // under SCALING_ALWAYS each root's scale buffer is at bufferIndex - kTipCount
        if (badPartials(bufferIndices[i]) || ((kFlags & BEAGLE_FLAG_SCALING_ALWAYS) && bufferIndices[i] < kTipCount))
            return true;
    }
    if (count > 0 && !(kFlags & BEAGLE_FLAG_SCALING_ALWAYS) && cumulativeScaleIndices[0] != BEAGLE_OP_NONE) {
        for (int i = 0; i < count; i++) {
            if (badScale(cumulativeScaleIndices[i]))
                return true;
        }
    }
    return false;
}

// calculateEdgeLogLikelihoods; spectral instances also look up every matrix index for dense contents
BEAGLE_CPU_TEMPLATE
bool BeagleCPUImpl<BEAGLE_CPU_GENERIC>::badEdgeIndices(const int* parentBufferIndices, const int* childBufferIndices,
                                                       const int* probabilityIndices,
                                                       const int* firstDerivativeIndices,
                                                       const int* secondDerivativeIndices,
                                                       const int* cumulativeScaleIndices, int count) {
    if (denseMatricesOnDemand() &&
        (badMatrices(probabilityIndices, count) ||
         (firstDerivativeIndices != NULL && badMatrices(firstDerivativeIndices, count)) ||
         (secondDerivativeIndices != NULL && badMatrices(secondDerivativeIndices, count))))
        return true;
    const bool noDerivatives = (firstDerivativeIndices == NULL && secondDerivativeIndices == NULL);
    if (count == 1) {
        if (badPartials(parentBufferIndices[0]) || badPartials(childBufferIndices[0]) ||
            badMatrix(probabilityIndices[0]) ||
            (firstDerivativeIndices != NULL && badMatrix(firstDerivativeIndices[0])) ||
            (secondDerivativeIndices != NULL && badMatrix(secondDerivativeIndices[0])))
            return true;
        const bool readsScale = (kAutoRootPartitioningEnabled && noDerivatives) ||
                                !(kFlags & (BEAGLE_FLAG_SCALING_AUTO | BEAGLE_FLAG_SCALING_ALWAYS));
        return readsScale && cumulativeScaleIndices[0] != BEAGLE_OP_NONE && badScale(cumulativeScaleIndices[0]);
    }
    if (!noDerivatives || count <= 0)
        return false; // not implemented for count > 1 with derivatives
    if (badPartialsIndices(parentBufferIndices, count) || badPartialsIndices(childBufferIndices, count) ||
        badMatrices(probabilityIndices, count))
        return true;
    if (cumulativeScaleIndices[0] != BEAGLE_OP_NONE) {
        for (int i = 0; i < count; i++) {
            if (badScale(cumulativeScaleIndices[i]))
                return true;
        }
    }
    return false;
}

// derivative matrices are written only by the real eigen decomposition (EigenDecompositionCube), and the second
// derivatives only together with the first
BEAGLE_CPU_TEMPLATE
bool BeagleCPUImpl<BEAGLE_CPU_GENERIC>::badDerivativeMatrices(const int* firstDerivativeIndices,
                                                              const int* secondDerivativeIndices,
                                                              int count) const {
    if ((kFlags & BEAGLE_FLAG_EIGEN_COMPLEX) || firstDerivativeIndices == NULL)
        return false;
    return badMatrices(firstDerivativeIndices, count) ||
           (secondDerivativeIndices != NULL && badMatrices(secondDerivativeIndices, count));
}

// post-order (updatePartials*) or pre-order (updatePrePartials*) operations, in the 7- or 9-int layout
BEAGLE_CPU_TEMPLATE
bool BeagleCPUImpl<BEAGLE_CPU_GENERIC>::badOperations(const int* operations, int count, bool byPartition,
                                                      int cumulativeScaleIndex, bool preOrder, bool top) {
    const int numOps = byPartition ? BEAGLE_PARTITION_OP_COUNT : BEAGLE_OP_COUNT;
    const bool spectral = denseMatricesOnDemand();
    // a post-order call reads its cumulative scale buffer once, before any operation
    if (!preOrder && !byPartition && cumulativeScaleIndex != BEAGLE_OP_NONE && badScale(cumulativeScaleIndex))
        return true;
    for (int op = 0; op < count; op++) {
        const int* operation = operations + op * numOps;
        const int destination = operation[0];
        const int writeScale = operation[1];
        const int readScale = operation[2];
        const int child1 = operation[3];
        const int matrix1 = operation[4];
        const int child2 = operation[5];
        const int matrix2 = operation[6];
        const bool singleChild = (child2 == BEAGLE_OP_NONE);
        const int cumulative = byPartition ? operation[8] : cumulativeScaleIndex;

        if (byPartition && badPartition(operation[7]))
            return true;
        if (badPartials(destination) || badPartials(child1) ||
            (!singleChild && (badPartials(child2) || badMatrix(matrix2))))
            return true;
        // a negative matrix on the parent side means none for a pre-order TOP op, and for a spectral single-child
        // pre-order op
        const bool matrix1Optional = preOrder && (top || (spectral && singleChild));
        if (!(matrix1Optional && matrix1 < 0) && badMatrix(matrix1))
            return true;

        if (preOrder && (top || spectral))
            continue; // these pre-order paths read no scale buffer
        if (!preOrder && cumulative != BEAGLE_OP_NONE && badScale(cumulative))
            return true;
        if (kFlags & (BEAGLE_FLAG_SCALING_AUTO | BEAGLE_FLAG_SCALING_ALWAYS)) {
            if (destination < kTipCount) // the destination's scale state is at destination - kTipCount
                return true;
        } else if (kFlags & BEAGLE_FLAG_SCALING_DYNAMIC) {
            // post-order rescales when no child is a tip with states; pre-order always
            const bool rescales = preOrder ||
                    (gTipStates[child1] == NULL && (singleChild || gTipStates[child2] == NULL));
            if (rescales && (badScale(writeScale) || badScale(readScale) || badScale(cumulative)))
                return true;
        } else if (writeScale >= 0 ? badScale(writeScale) : (readScale >= 0 && badScale(readScale))) {
            return true;
        }
    }
    return false;
}
#endif

BEAGLE_CPU_TEMPLATE
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::threadWaiting(threadData* tData)
{
    std::unique_lock<std::mutex> l(tData->m, std::defer_lock);
    while (true)
    {
        l.lock();

        // Wait until the queue won't be empty or stop is signaled
        tData->cv.wait(l, [tData] () {
            return (tData->stop || !tData->jobs.empty());
            });

        // Stop was signaled, let's exit the thread
        if (tData->stop) { return; }

        // Pop one task from the queue...
        std::packaged_task<void()> j = std::move(tData->jobs.front());
        tData->jobs.pop();

        l.unlock();

        // Execute the task!
        j();
    }
}

///////////////////////////////////////////////////////////////////////////////
// BeagleCPUImplFactory public methods
BEAGLE_CPU_FACTORY_TEMPLATE
BeagleImpl* BeagleCPUImplFactory<BEAGLE_CPU_FACTORY_GENERIC>::createImpl(int tipCount,
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
                                             long long preferenceFlags,
                                             long long requirementFlags,
                                             int* errorCode) {

    BeagleImpl* impl = new BeagleCPUImpl<REALTYPE, T_PAD_DEFAULT, P_PAD_DEFAULT>();

    try {
        *errorCode =
            impl->createInstance(tipCount, partialsBufferCount, compactBufferCount, stateCount,
                                 patternCount, eigenBufferCount, matrixBufferCount,
                                 categoryCount,scaleBufferCount, resourceNumber,
                                 pluginResourceNumber,
                                 preferenceFlags, requirementFlags);
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


BEAGLE_CPU_FACTORY_TEMPLATE
const char* BeagleCPUImplFactory<BEAGLE_CPU_FACTORY_GENERIC>::getName() {
    return getBeagleCPUName<BEAGLE_CPU_FACTORY_GENERIC>();
}

BEAGLE_CPU_FACTORY_TEMPLATE
const long long BeagleCPUImplFactory<BEAGLE_CPU_FACTORY_GENERIC>::getFlags() {
    long long flags = BEAGLE_FLAG_COMPUTATION_SYNCH |
                 BEAGLE_FLAG_SCALING_MANUAL | BEAGLE_FLAG_SCALING_ALWAYS | BEAGLE_FLAG_SCALING_AUTO | BEAGLE_FLAG_SCALING_DYNAMIC |
                 BEAGLE_FLAG_THREADING_NONE | BEAGLE_FLAG_THREADING_CPP |
                 BEAGLE_FLAG_PROCESSOR_CPU |
                 BEAGLE_FLAG_VECTOR_NONE |
                 BEAGLE_FLAG_SCALERS_LOG | BEAGLE_FLAG_SCALERS_RAW |
                 BEAGLE_FLAG_EIGEN_COMPLEX | BEAGLE_FLAG_EIGEN_REAL |
                 BEAGLE_FLAG_INVEVEC_STANDARD | BEAGLE_FLAG_INVEVEC_TRANSPOSED |
                 BEAGLE_FLAG_PREORDER_TRANSPOSE_MANUAL | BEAGLE_FLAG_PREORDER_TRANSPOSE_AUTO |
                 BEAGLE_FLAG_FRAMEWORK_CPU;
#ifdef BEAGLE_OPENMP_ENABLED
    flags |= BEAGLE_FLAG_THREADING_OPENMP;
#endif
    if (DOUBLE_PRECISION)
        flags |= BEAGLE_FLAG_PRECISION_DOUBLE;
    else
        flags |= BEAGLE_FLAG_PRECISION_SINGLE;
    return flags;
}

}   // namespace cpu
}   // namespace beagle

#endif // BEAGLE_CPU_IMPL_GENERAL_HPP

