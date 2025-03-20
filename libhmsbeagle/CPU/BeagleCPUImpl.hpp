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

BEAGLE_CPU_FACTORY_TEMPLATE
inline const char* getBeagleCPUName(){ return "CPU-Unknown"; };

template<>
inline const char* getBeagleCPUName<double>(){ return "CPU-Double"; };

template<>
inline const char* getBeagleCPUName<float>(){ return "CPU-Single"; };

BEAGLE_CPU_FACTORY_TEMPLATE
inline const long getBeagleCPUFlags(){ return BEAGLE_FLAG_COMPUTATION_SYNCH; };

template<>
inline const long getBeagleCPUFlags<double>(){ return BEAGLE_FLAG_COMPUTATION_SYNCH |
                                                      BEAGLE_FLAG_PROCESSOR_CPU |
                                                      BEAGLE_FLAG_PRECISION_DOUBLE |
                                                      BEAGLE_FLAG_VECTOR_NONE |
                                                      BEAGLE_FLAG_FRAMEWORK_CPU; };

template<>
inline const long getBeagleCPUFlags<float>(){ return BEAGLE_FLAG_COMPUTATION_SYNCH |
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

    if (kThreadingEnabled) {
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
    }

    if (kAutoPartitioningEnabled) {
        free(gAutoPartitionOperations);
        if (kAutoRootPartitioningEnabled) {
            free(gAutoPartitionIndices);
            free(gAutoPartitionOutSumLogLikelihoods);
        }
    }
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
                                  long preferenceFlags,
                                  long requirementFlags) {
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
    else
        kFlags |= BEAGLE_FLAG_THREADING_NONE;

    if (kFlags & BEAGLE_FLAG_EIGEN_COMPLEX)
        gEigenDecomposition = new EigenDecompositionSquare<BEAGLE_CPU_EIGEN_GENERIC>(kEigenDecompCount,
                kStateCount,kCategoryCount,kFlags);
    else
        gEigenDecomposition = new EigenDecompositionCube<BEAGLE_CPU_EIGEN_GENERIC>(kEigenDecompCount,
                kStateCount, kCategoryCount,kFlags);

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
    for (int i = 0; i < kMatrixCount; i++) {
        gTransitionMatrices[i] = (REALTYPE*) mallocAligned(sizeof(REALTYPE) * kMatrixSize * kCategoryCount);
        if (gTransitionMatrices[i] == 0L)
            throw std::bad_alloc();
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
            int* patternPartitions = (int*) malloc(sizeof(int) * kPatternCount);
            int partitionSize = kPatternCount/partitionCount;
            for (int i=0; i<kPatternCount; i++) {
                int sitePartition = i/partitionSize;
                if (sitePartition > partitionCount - 1)
                    sitePartition = partitionCount - 1;
                patternPartitions[i] = sitePartition;
            }
            setPatternPartitions(partitionCount, patternPartitions);

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

    return BEAGLE_SUCCESS;
}

BEAGLE_CPU_TEMPLATE
const char* BeagleCPUImpl<BEAGLE_CPU_GENERIC>::getName() {
    return getBeagleCPUName<BEAGLE_CPU_FACTORY_GENERIC>();
}

BEAGLE_CPU_TEMPLATE
const long BeagleCPUImpl<BEAGLE_CPU_GENERIC>::getFlags() {
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

    kThreadingEnabled = false;
    kAutoPartitioningEnabled = false;
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

            int* patternPartitions = (int*) malloc(sizeof(int) * kPatternCount);
            int partitionSize = kPatternCount/partitionCount;
            for (int i=0; i<kPatternCount; i++) {
                int sitePartition = i/partitionSize;
                if (sitePartition > partitionCount - 1)
                    sitePartition = partitionCount - 1;
                patternPartitions[i] = sitePartition;
            }
            setPatternPartitions(partitionCount, patternPartitions);

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
        // We treat this as a special case so that we don't have convoluted logic
        //      at the end of the loop over patterns
        if (kAutoRootPartitioningEnabled) {
//            calcRootLogLikelihoodsByAutoPartitionAsync(bufferIndices,
//                                                       categoryWeightsIndices,
//                                                       stateFrequenciesIndices,
//                                                       cumulativeScaleIndices,
//                                                       gAutoPartitionIndices,
//                                                       gAutoPartitionOutSumLogLikelihoods);
//
//            *outSumLogLikelihood = 0.0;
//
//            for (int i = 0; i < kPartitionCount; i++) {
//                *outSumLogLikelihood += gAutoPartitionOutSumLogLikelihoods[i];
//            }
//
//            if (*outSumLogLikelihood != *outSumLogLikelihood) {
//                return BEAGLE_ERROR_FLOATING_POINT;
//            } else {
//                return BEAGLE_SUCCESS;
//            }
            return BEAGLE_ERROR_NO_IMPLEMENTATION;
        } else {
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
                    tmpRealPartialsOffset += kPartialsPaddedStateCount;
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

    if (kThreadingEnabled) {
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

        kThreadingEnabled = false;
    }

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

if (T_PAD != 0) {
    const double* offsetInMatrix = inMatrix;
    REALTYPE* offsetBeagleMatrix = gTransitionMatrices[matrixIndex];
    for(int i = 0; i < kCategoryCount; i++) {
        for(int j = 0; j < kStateCount; j++) {
            beagleMemCpy(offsetBeagleMatrix, offsetInMatrix, kStateCount);
            offsetBeagleMatrix[kStateCount] = paddedValue;
            offsetBeagleMatrix += kTransPaddedStateCount; // Skip padding
            offsetInMatrix += kStateCount;
        }
    }
} else {
    beagleMemCpy(gTransitionMatrices[matrixIndex], inMatrix,
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
        const double* inMatrix = inMatrices + k*kStateCount*kStateCount*kCategoryCount;
        int matrixIndex = matrixIndices[k];

if (T_PAD != 0) {
        const double* offsetInMatrix = inMatrix;
        REALTYPE* offsetBeagleMatrix = gTransitionMatrices[matrixIndex];
        for(int i = 0; i < kCategoryCount; i++) {
            for(int j = 0; j < kStateCount; j++) {
                beagleMemCpy(offsetBeagleMatrix, offsetInMatrix, kStateCount);
                offsetBeagleMatrix[kStateCount] = paddedValues[k];
                offsetBeagleMatrix += kTransPaddedStateCount; // Skip padding
                offsetInMatrix += kStateCount;
            }
        }
} else {
        beagleMemCpy(gTransitionMatrices[matrixIndex], inMatrix,
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

    for (int u = 0; u < matrixCount; u++) {

        if(firstIndices[u] == resultIndices[u] || secondIndices[u] == resultIndices[u]) {

#ifdef BEAGLE_DEBUG_FLOW
            fprintf(stderr, "In-place convolution is not allowed \n");
#endif

            returnCode = BEAGLE_ERROR_OUT_OF_RANGE;
            break;

        }//END: overwrite check

        REALTYPE* C = gTransitionMatrices[resultIndices[u]];
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

    for (int u = 0; u < matrixCount; u++) {

        if (inputIndices[u] == resultIndices[u]) {

#ifdef BEAGLE_DEBUG_FLOW
            fprintf(stderr, "In-place transpose is not allowed.\n");
#endif

            returnCode = BEAGLE_ERROR_OUT_OF_RANGE;
            break;

        }

        REALTYPE* A = gTransitionMatrices[inputIndices[u]];
        REALTYPE* C = gTransitionMatrices[resultIndices[u]];

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
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::updateTransitionMatrices(int eigenIndex,
                                            const int* probabilityIndices,
                                            const int* firstDerivativeIndices,
                                            const int* secondDerivativeIndices,
                                            const double* edgeLengths,
                                            int count) {
    // for (int i = 0; i < count; i++) {
    //     printf("uTM %d %d %f %d\n", eigenIndex, probabilityIndices[i], edgeLengths[i], 0);
    // }

    gEigenDecomposition->updateTransitionMatrices(eigenIndex,probabilityIndices,firstDerivativeIndices,secondDerivativeIndices,
                                                  edgeLengths,gCategoryRates[0],gTransitionMatrices,count);
    return BEAGLE_SUCCESS;
}


BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::updateTransitionMatricesWithModelCategories(int* eigenIndices,
                                            const int* probabilityIndices,
                                            const int* firstDerivativeIndices,
                                            const int* secondDerivativeIndices,
                                            const double* edgeLengths,
                                            int count) {

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
                                                      int cumulativeScaleIndex) {

    int returnCode = BEAGLE_ERROR_GENERAL;

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
                                cumulativeScaleIndex);
    }

    return returnCode;
}


BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::updatePrePartials(const int *operations,
                                                         int count,
                                                         int cumulativeScaleIndex) {
    int returnCode = BEAGLE_ERROR_GENERAL;

    bool byPartition = false;
    returnCode = upPrePartials(byPartition, operations, count, cumulativeScaleIndex);

    return returnCode;
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
    return calcEdgeLogDerivatives(
            postBufferIndices, preBufferIndices,
            derivativeMatrixIndices, NULL,
            categoryWeightsIndices,categoryRatesIndices, cumulativeScaleIndices,
            count,
            outDerivatives,
            outSumDerivatives, outSumSquaredDerivatives);
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
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::updatePartialsByPartition(const int* operations,
                                                                 int count) {

    int returnCode = BEAGLE_ERROR_GENERAL;

    if (kThreadingEnabled) {
        returnCode = upPartialsByPartitionAsync(operations,
                                                count);
    } else {
        bool byPartition = true;
        returnCode = upPartials(byPartition,
                                operations,
                                count,
                                BEAGLE_OP_NONE);
    }

    return returnCode;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUImpl<BEAGLE_CPU_GENERIC>::updatePrePartialsByPartition(const int* operations,
                                                                    int count) {

    int returnCode = BEAGLE_ERROR_GENERAL;

    if (kThreadingEnabled) {
//        TODO
//        returnCode = upPrePartialsByPartitionAsync(operations,
//                                                   count);
    } else {
        bool byPartition = true;
        returnCode = upPrePartials(byPartition,
                                  operations,
                                  count,
                                  BEAGLE_OP_NONE);
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
        std::packaged_task<void()> threadTask(
            std::bind(&BeagleCPUImpl<BEAGLE_CPU_GENERIC>::upPartials, this,
                      true,
                      (const int*) gThreadOperations[i],
                      gThreadOpCounts[i],
                      BEAGLE_OP_NONE));

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
                                                  int cumulativeScaleIndex) {

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

        const REALTYPE* partials1 = gPartials[child1Index];
        const REALTYPE* partials2 = gPartials[child2Index];

        const int* tipStates1 = gTipStates[child1Index];
        const int* tipStates2 = gTipStates[child2Index];

        const REALTYPE* matrices1 = gTransitionMatrices[child1TransMatIndex];
        const REALTYPE* matrices2 = gTransitionMatrices[child2TransMatIndex];

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

        if (tipStates1 != NULL) {
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


        /// non-root nodes, can be a tip
        const REALTYPE *partials1 = gPartials[parentIndex];
        const REALTYPE *partials2 = gPartials[siblingIndex];

        const int *tipStates2 = gTipStates[siblingIndex];

        const REALTYPE *matrices1 = gTransitionMatrices[parentTransMatIndex];
        const REALTYPE *matrices2 = gTransitionMatrices[siblingTransMatIndex];


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

        if (tipStates2 != NULL) {
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

    if (crossProductNumeratorTmp == nullptr) {
        crossProductNumeratorTmp = (REALTYPE*) mallocAligned(sizeof(REALTYPE) * kPaddedPatternCount
                * kStateCount * kStateCount);
    }

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

        resetDerivativeTemporaries();

        if (tipStates != NULL) {

            calcEdgeLogDerivativesStates(tipStates, preOrderPartial, firstDerivativeIndex,
                                        secondDerivativeIndex, categoryRates, categoryWeights,
                                        outDerivativesForNode,
                                        outSumDerivativesForNode,
                                        outSumSquaredDerivativesForNode);

        } else {

            const REALTYPE *postOrderPartial = gPartials[postBufferIndices[nodeNum]];
            calcEdgeLogDerivativesPartials(postOrderPartial, preOrderPartial, firstDerivativeIndex,
                                       secondDerivativeIndex, categoryRates, categoryWeights,
                                       scalingFactorsIndex,
                                       outDerivativesForNode,
                                       outSumDerivativesForNode,
                                       outSumSquaredDerivativesForNode);
        }

        accumulateDerivatives(outDerivativesForNode,
                outSumDerivativesForNode,
                outSumSquaredDerivativesForNode);

    }

    return returnCode;
}

BEAGLE_CPU_TEMPLATE template <bool DoDerivatives, bool DoSum, bool DoSumSquared>
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::accumulateDerivativesImpl(
        double* outDerivatives,
        double* outSumDerivatives,
        double* outSumSquaredDerivatives) {

    REALTYPE sum = 0.0;
    REALTYPE sumSquared = 0.0;

    for (int k = 0; k < kPatternCount; k++) {
        REALTYPE derivative = grandNumeratorDerivTmp[k] / grandDenominatorDerivTmp[k];
        if (DoDerivatives) {
            outDerivatives[k] = derivative;
        }
        if (DoSum) { // TODO Confirm that these are compile-time
            sum += derivative * gPatternWeights[k];
        }
        if (DoSumSquared) {
            sumSquared += derivative * derivative * gPatternWeights[k];
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
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::accumulateDerivativesDispatch2(
        double* outDerivatives,
        double* outSumDerivatives,
        double* outSumSquaredDerivatives) {

    if (outSumSquaredDerivatives == NULL) {
        accumulateDerivativesImpl<DoDerivatives, DoSum, false>(
                outDerivatives, outSumDerivatives, outSumSquaredDerivatives);
    } else {
        accumulateDerivativesImpl<DoDerivatives, DoSum, true>(
                outDerivatives, outSumDerivatives, outSumSquaredDerivatives);
    }
}

BEAGLE_CPU_TEMPLATE template <bool DoDerivatives>
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::accumulateDerivativesDispatch1(
        double* outDerivatives,
        double* outSumDerivatives,
        double* outSumSquaredDerivatives) {

    if (outSumDerivatives == NULL) {
        accumulateDerivativesDispatch2<DoDerivatives, false>(
                outDerivatives, outSumDerivatives, outSumSquaredDerivatives);
    } else {
        accumulateDerivativesDispatch2<DoDerivatives, true>(
                outDerivatives, outSumDerivatives, outSumSquaredDerivatives);
    }
}

BEAGLE_CPU_TEMPLATE
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::resetDerivativeTemporaries() {
        std::fill(grandNumeratorDerivTmp, grandNumeratorDerivTmp + kPaddedPatternCount, 0);
        std::fill(grandDenominatorDerivTmp, grandDenominatorDerivTmp + kPaddedPatternCount, 0);
}


BEAGLE_CPU_TEMPLATE
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::accumulateDerivatives(double* outDerivatives,
                                                              double* outSumDerivatives,
                                                              double* outSumSquaredDerivatives) {
    if (outDerivatives == NULL) {
        accumulateDerivativesDispatch1<false>(
                outDerivatives, outSumDerivatives, outSumSquaredDerivatives);
    } else {
        accumulateDerivativesDispatch1<true>(
                outDerivatives, outSumDerivatives, outSumSquaredDerivatives);
    }
}

BEAGLE_CPU_TEMPLATE
void BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcEdgeLogDerivativesStates(const int *tipStates,
                                                                     const REALTYPE *preOrderPartial,
                                                                     const int firstDerivativeIndex,
                                                                     const int secondDerivativeIndex,
                                                                     const double *categoryRates,
                                                                     const REALTYPE *categoryWeights,
                                                                     double *outDerivatives,
                                                                     double *outSumDerivatives,
                                                                     double *outSumSquaredDerivatives) {

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

            grandNumeratorDerivTmp[pattern] += categoryWeights[category] * numerator;
            grandDenominatorDerivTmp[pattern] += categoryWeights[category] * denominator;
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
//                                                                       const REALTYPE *cumulativeScaleBuffer,
                                                                       double *outDerivatives,
                                                                       double *outSumDerivatives,
                                                                       double *outSumSquaredDerivatives) {

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

            grandNumeratorDerivTmp[pattern] += weight * numerator;
            grandDenominatorDerivTmp[pattern] += weight * denominator;
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

    if (count == 1) {
        if (kFlags & BEAGLE_FLAG_SCALING_AUTO) {
            returnCode =  BEAGLE_ERROR_NO_IMPLEMENTATION;
        } else if (kFlags & BEAGLE_FLAG_SCALING_ALWAYS) {
            returnCode = BEAGLE_ERROR_NO_IMPLEMENTATION;
        } else {

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
                    destP[offset++] *= pow(2.0, -expMax);
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
    std::fill(tmpdestPtr, tmpdestPtr + kPartialsSize, 0);

#pragma omp parallel for num_threads(kCategoryCount)
    for (int l = 0; l < kCategoryCount; l++) {
        int v = l*kPartialsPaddedStateCount*kPatternCount + kPartialsPaddedStateCount*startPattern;
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
    std::fill(tmpdestPtr, tmpdestPtr + kPartialsSize, 0);

#pragma omp parallel for num_threads(kCategoryCount)
    for (int l = 0; l < kCategoryCount; l++) {
        int v = l*kPartialsPaddedStateCount*kPatternCount + kPartialsPaddedStateCount*startPattern;
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
                                             long preferenceFlags,
                                             long requirementFlags,
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
const long BeagleCPUImplFactory<BEAGLE_CPU_FACTORY_GENERIC>::getFlags() {
    long flags = BEAGLE_FLAG_COMPUTATION_SYNCH |
                 BEAGLE_FLAG_SCALING_MANUAL | BEAGLE_FLAG_SCALING_ALWAYS | BEAGLE_FLAG_SCALING_AUTO | BEAGLE_FLAG_SCALING_DYNAMIC |
                 BEAGLE_FLAG_THREADING_NONE | BEAGLE_FLAG_THREADING_CPP |
                 BEAGLE_FLAG_PROCESSOR_CPU |
                 BEAGLE_FLAG_VECTOR_NONE |
                 BEAGLE_FLAG_SCALERS_LOG | BEAGLE_FLAG_SCALERS_RAW |
                 BEAGLE_FLAG_EIGEN_COMPLEX | BEAGLE_FLAG_EIGEN_REAL |
                 BEAGLE_FLAG_INVEVEC_STANDARD | BEAGLE_FLAG_INVEVEC_TRANSPOSED |
                 BEAGLE_FLAG_PREORDER_TRANSPOSE_MANUAL | BEAGLE_FLAG_PREORDER_TRANSPOSE_AUTO |
                 BEAGLE_FLAG_FRAMEWORK_CPU;
    if (DOUBLE_PRECISION)
        flags |= BEAGLE_FLAG_PRECISION_DOUBLE;
    else
        flags |= BEAGLE_FLAG_PRECISION_SINGLE;
    return flags;
}

}   // namespace cpu
}   // namespace beagle

#endif // BEAGLE_CPU_IMPL_GENERAL_HPP

