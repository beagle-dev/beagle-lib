/*
 *  BeagleCPUHierarchicalImpl.hpp
 *  BEAGLE
 *
 * Copyright 2026 Phylogenetic Likelihood Working Group
 *
 * This file is part of BEAGLE.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */

#ifndef BEAGLE_CPU_HIERARCHICAL_IMPL_HPP
#define BEAGLE_CPU_HIERARCHICAL_IMPL_HPP

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include <iostream>

#include "libhmsbeagle/beagle.h"
#include "libhmsbeagle/CPU/BeagleCPUImpl.h"
#include "libhmsbeagle/CPU/BeagleCPUHierarchicalImpl.h"

namespace beagle {
namespace cpu {

BEAGLE_CPU_FACTORY_TEMPLATE
inline const char* getBeagleCPUHierarchicalName(){ return "CPU-Hierarchical-Unknown"; };

template<>
inline const char* getBeagleCPUHierarchicalName<double>(){ return "CPU-Hierarchical-Double"; };

template<>
inline const char* getBeagleCPUHierarchicalName<float>(){ return "CPU-Hierarchical-Single"; };

BEAGLE_CPU_FACTORY_TEMPLATE
inline const long getBeagleCPUHierarchicalFlags() {
    long flags = BEAGLE_FLAG_COMPUTATION_SYNCH |
                 BEAGLE_FLAG_SCALING_MANUAL | BEAGLE_FLAG_SCALING_ALWAYS | BEAGLE_FLAG_SCALING_DYNAMIC |
                 BEAGLE_FLAG_THREADING_NONE | BEAGLE_FLAG_THREADING_CPP |
                 BEAGLE_FLAG_PROCESSOR_CPU |
                 BEAGLE_FLAG_VECTOR_NONE |
                 BEAGLE_FLAG_SCALERS_LOG | BEAGLE_FLAG_SCALERS_RAW |
                 BEAGLE_FLAG_HIERARCHICAL_REPRESENTATION |
                 BEAGLE_FLAG_FRAMEWORK_CPU;
    if (DOUBLE_PRECISION)
        flags |= BEAGLE_FLAG_PRECISION_DOUBLE;
    else
        flags |= BEAGLE_FLAG_PRECISION_SINGLE;
    return flags;
}

BEAGLE_CPU_TEMPLATE
BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::~BeagleCPUHierarchicalImpl() {
}

BEAGLE_CPU_TEMPLATE
const char* BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::getName() {
    return getBeagleCPUHierarchicalName<BEAGLE_CPU_FACTORY_GENERIC>();
}

BEAGLE_CPU_TEMPLATE
const long BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::getFlags() {
    return getBeagleCPUHierarchicalFlags<BEAGLE_CPU_FACTORY_GENERIC>();
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::createInstance(int tipCount,
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
    if ((preferenceFlags | requirementFlags) & BEAGLE_FLAG_SCALING_AUTO) {
        fprintf(stderr, "BEAGLE_FLAG_SCALING_AUTO is not yet implemented for hierarchical representation\n");
        return BEAGLE_ERROR_NO_IMPLEMENTATION;
    }
    int returnCode = BeagleCPUImpl<BEAGLE_CPU_GENERIC>::createInstance(tipCount, partialsBufferCount,
                                                                       compactBufferCount, stateCount,
                                                                       patternCount, eigenDecompositionCount,
                                                                       matrixCount, categoryCount,
                                                                       scaleBufferCount, resourceNumber,
                                                                       pluginResourceNumber,
                                                                       preferenceFlags, requirementFlags);
    kFlags &= ~(BEAGLE_FLAG_EIGEN_REAL | BEAGLE_FLAG_EIGEN_COMPLEX |
                BEAGLE_FLAG_INVEVEC_STANDARD | BEAGLE_FLAG_INVEVEC_TRANSPOSED);
    gSlots.assign(kMatrixCount, HierarchicalSlot<REALTYPE>());
    gW.assign((size_t) (kStateCount - 1) * kPaddedPatternCount, REALTYPE(0));
    gV.assign((size_t) (kStateCount - 1) * kPaddedPatternCount, REALTYPE(0));
    gInvScaleTmp.assign(kPaddedPatternCount, REALTYPE(0));
    const int modulus = getPaddedPatternsModulus();
    int tile = (int) (kHierarchicalTileBudget / (3 * (size_t) kStateCount * sizeof(REALTYPE)));
    tile = std::max(64, std::min(tile, kPaddedPatternCount));
    kTileSize = std::max(modulus, (tile / modulus) * modulus);
    gSiteScaleTmp.assign(kPaddedPatternCount, REALTYPE(0));
    gSiteAutoScaleTmp.assign(kPaddedPatternCount, 0.0);
    return returnCode;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::getPaddedPatternsModulus() {
    return 64 / sizeof(REALTYPE);
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::ensurePartials(int bufferIndex) {
    if (gPartials[bufferIndex] == NULL) {
        gPartials[bufferIndex] = (REALTYPE*) mallocAligned(sizeof(REALTYPE) * kPartialsSize);
        if (gPartials[bufferIndex] == 0L)
            return BEAGLE_ERROR_OUT_OF_MEMORY;
    }
    return BEAGLE_SUCCESS;
}

BEAGLE_CPU_TEMPLATE
void BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::transposeToSiteMajor(REALTYPE* dest,
                                                                         const double* src,
                                                                         int srcCategoryStride,
                                                                         int srcPatternStride) {
    for (int l = 0; l < kCategoryCount; l++) {
        const double* category = src + l * srcCategoryStride;
        for (int i = 0; i < kPartialsPaddedStateCount; i++) {
            REALTYPE* row = dest + pIdx(l, 0, i);
            if (i < kStateCount) {
                for (int k = 0; k < kPatternCount; k++)
                    row[k] = (REALTYPE) category[k * srcPatternStride + i];
            } else {
                for (int k = 0; k < kPatternCount; k++)
                    row[k] = 0;
            }
            for (int k = kPatternCount; k < kPaddedPatternCount; k++)
                row[k] = 0;
        }
    }
}

BEAGLE_CPU_TEMPLATE
void BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::transposeFromSiteMajor(double* dest,
                                                                           const REALTYPE* src) {
    for (int l = 0; l < kCategoryCount; l++) {
        double* category = dest + l * kPatternCount * kStateCount;
        for (int i = 0; i < kStateCount; i++) {
            const REALTYPE* row = src + pIdx(l, 0, i);
            for (int k = 0; k < kPatternCount; k++)
                category[k * kStateCount + i] = (double) row[k];
        }
    }
}

BEAGLE_CPU_TEMPLATE
    void BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::rescaleRange(REALTYPE* destP,
                                                                     REALTYPE* scaleFactors,
                                                                     REALTYPE* cumulativeScaleFactors,
                                                                     int startPattern,
                                                                     int endPattern) {
        REALTYPE* siteMax = gSiteScaleTmp.data();
        for (int k = startPattern; k < endPattern; k++)
            siteMax[k] = 0;
        for (int l = 0; l < kCategoryCount; l++) {
            for (int i = 0; i < kStateCount; i++) {
                const REALTYPE* row = destP + pIdx(l, 0, i);
                for (int k = startPattern; k < endPattern; k++)
                    siteMax[k] = (row[k] > siteMax[k]) ? row[k] : siteMax[k];
            }
        }

    for (int k = startPattern; k < endPattern; k++) {
        REALTYPE max = siteMax[k];
        if (max == 0)
            max = 1.0;
        if (kFlags & BEAGLE_FLAG_SCALERS_LOG) {
            REALTYPE logMax = log(max);
            scaleFactors[k] = logMax;
            if (cumulativeScaleFactors != NULL)
                cumulativeScaleFactors[k] += logMax;
        } else {
            scaleFactors[k] = max;
            if (cumulativeScaleFactors != NULL)
                cumulativeScaleFactors[k] += log(max);
        }
        siteMax[k] = REALTYPE(1.0) / max;
    }

    for (int l = 0; l < kCategoryCount; l++) {
        for (int i = 0; i < kStateCount; i++) {
            REALTYPE* row = destP + pIdx(l, 0, i);
            for (int k = startPattern; k < endPattern; k++)
                row[k] *= siteMax[k];
        }
    }
}

BEAGLE_CPU_TEMPLATE
void BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::rescalePartials(REALTYPE* destP,
                                                                    REALTYPE* scaleFactors,
                                                                    REALTYPE* cumulativeScaleFactors,
                                                                    const int fillWithOnes) {
    rescaleRange(destP, scaleFactors, cumulativeScaleFactors, 0, kPatternCount);
}

BEAGLE_CPU_TEMPLATE
void BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::rescalePartialsByPartition(REALTYPE* destP,
                                                                               REALTYPE* scaleFactors,
                                                                               REALTYPE* cumulativeScaleFactors,
                                                                               const int fillWithOnes,
                                                                               const int partitionIndex) {
    rescaleRange(destP, scaleFactors, cumulativeScaleFactors,
                 gPatternPartitionsStartPatterns[partitionIndex],
                 gPatternPartitionsStartPatterns[partitionIndex + 1]);
}

BEAGLE_CPU_TEMPLATE
void BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::autoRescalePartials(REALTYPE* destP,
                                                                        signed short* scaleFactors) {
    REALTYPE* siteMax = gSiteScaleTmp.data();
    double* factor = gSiteAutoScaleTmp.data();
    for (int k = 0; k < kPatternCount; k++)
        siteMax[k] = 0;
    for (int l = 0; l < kCategoryCount; l++) {
        for (int i = 0; i < kStateCount; i++) {
            const REALTYPE* row = destP + pIdx(l, 0, i);
            for (int k = 0; k < kPatternCount; k++)
                siteMax[k] = (row[k] > siteMax[k]) ? row[k] : siteMax[k];
        }
    }

    for (int k = 0; k < kPatternCount; k++) {
        int expMax;
        frexp(siteMax[k], &expMax);
        scaleFactors[k] = expMax;
        factor[k] = (expMax != 0) ? pow(2.0, -expMax) : 1.0;
    }

    for (int l = 0; l < kCategoryCount; l++) {
        for (int i = 0; i < kStateCount; i++) {
            REALTYPE* row = destP + pIdx(l, 0, i);
            for (int k = 0; k < kPatternCount; k++)
                row[k] *= factor[k];
        }
    }
}

BEAGLE_CPU_TEMPLATE
EigenDecomposition<BEAGLE_CPU_EIGEN_GENERIC>* BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::createEigenDecomposition(
        int decompositionCount,
        int stateCount,
        int categoryCount,
        long flags) {
    return nullptr;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::setTipPartials(int tipIndex,
                                                                  const double* inPartials) {
    if (tipIndex < 0 || tipIndex >= kTipCount)
        return BEAGLE_ERROR_OUT_OF_RANGE;
    int returnCode = ensurePartials(tipIndex);
    if (returnCode != BEAGLE_SUCCESS)
        return returnCode;
    transposeToSiteMajor(gPartials[tipIndex], inPartials, 0, kStateCount);
    return BEAGLE_SUCCESS;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::setPartials(int bufferIndex,
                                                               const double* inPartials) {
    if (bufferIndex < 0 || bufferIndex >= kBufferCount)
        return BEAGLE_ERROR_OUT_OF_RANGE;
    int returnCode = ensurePartials(bufferIndex);
    if (returnCode != BEAGLE_SUCCESS)
        return returnCode;
    transposeToSiteMajor(gPartials[bufferIndex], inPartials, kPatternCount * kStateCount, kStateCount);
    return BEAGLE_SUCCESS;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::setRootPrePartials(const int* bufferIndices,
                                                                      const int* stateFrequenciesIndices,
                                                                      int count) {
    if (count != 1)
        return BEAGLE_ERROR_NO_IMPLEMENTATION;
    int bufferIndex = bufferIndices[0];
    if (bufferIndex < 0 || bufferIndex >= kBufferCount)
        return BEAGLE_ERROR_OUT_OF_RANGE;
    int returnCode = ensurePartials(bufferIndex);
    if (returnCode != BEAGLE_SUCCESS)
        return returnCode;
    const REALTYPE* frequencies = gStateFrequencies[stateFrequenciesIndices[0]];
    REALTYPE* dest = gPartials[bufferIndex];
    for (int l = 0; l < kCategoryCount; l++) {
        for (int i = 0; i < kPartialsPaddedStateCount; i++) {
            REALTYPE* row = dest + pIdx(l, 0, i);
            const REALTYPE value = (i < kStateCount) ? frequencies[i] : REALTYPE(0);
            for (int k = 0; k < kPatternCount; k++)
                row[k] = value;
            for (int k = kPatternCount; k < kPaddedPatternCount; k++)
                row[k] = 0;
        }
    }
    return BEAGLE_SUCCESS;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::getPartials(int bufferIndex,
                                                               int cumulativeScaleIndex,
                                                               double* outPartials) {
    if (bufferIndex < 0 || bufferIndex >= kBufferCount)
        return BEAGLE_ERROR_OUT_OF_RANGE;
    BEAGLE_CHECK_INDICES(cumulativeScaleIndex != BEAGLE_OP_NONE && this->badScale(cumulativeScaleIndex));

    transposeFromSiteMajor(outPartials, gPartials[bufferIndex]);

    if (cumulativeScaleIndex != BEAGLE_OP_NONE) {
        REALTYPE* cumulativeScaleBuffer = gScaleBuffers[cumulativeScaleIndex];
        int index = 0;
        for (int k = 0; k < kPatternCount; k++) {
            REALTYPE scaleFactor = exp(cumulativeScaleBuffer[k]);
            for (int i = 0; i < kStateCount; i++) {
                outPartials[index] *= scaleFactor;
                index++;
            }
        }
    }

    return BEAGLE_SUCCESS;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::setEigenDecomposition(int eigenIndex,
                                                                         const double* inEigenVectors,
                                                                         const double* inInverseEigenVectors,
                                                                         const double* inEigenValues) {
    return BEAGLE_ERROR_NO_IMPLEMENTATION;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::setTransitionMatrix(int matrixIndex,
                                                                       const double* inMatrix,
                                                                       double paddedValue) {
    if (matrixIndex < 0 || matrixIndex >= kMatrixCount)
        return BEAGLE_ERROR_OUT_OF_RANGE;
    if ((int) gSlots.size() < kMatrixCount)
        gSlots.resize(kMatrixCount);
    return unpackHierarchicalBundle(inMatrix, kStateCount, kCategoryCount, gSlots[matrixIndex]);
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::setTransitionMatrices(const int* matrixIndices,
                                                                         const double* inMatrices,
                                                                         const double* paddedValues,
                                                                         int count) {
    const int bundleLength = hierarchicalBundleLength(kStateCount, kCategoryCount);
    for (int k = 0; k < count; k++) {
        int returnCode = setTransitionMatrix(matrixIndices[k], inMatrices + k * bundleLength, 0.0);
        if (returnCode != BEAGLE_SUCCESS)
            return returnCode;
    }
    return BEAGLE_SUCCESS;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::getTransitionMatrix(int matrixIndex,
                                                                       double* outMatrix) {
    if (matrixIndex < 0 || matrixIndex >= kMatrixCount ||
        matrixIndex >= (int) gSlots.size() || !gSlots[matrixIndex].set || gStateFrequencies[0] == NULL)
        return BEAGLE_ERROR_OUT_OF_RANGE;
    hierarchicalTransitionMatrices(gSlots[matrixIndex], gStateFrequencies[0], kStateCount, kCategoryCount, outMatrix);
    return BEAGLE_SUCCESS;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::updateTransitionMatrices(int eigenIndex,
                                                                            const int* probabilityIndices,
                                                                            const int* firstDerivativeIndices,
                                                                            const int* secondDerivativeIndices,
                                                                            const double* edgeLengths,
                                                                            int count) {
    return BEAGLE_ERROR_NO_IMPLEMENTATION;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::updateTransitionMatricesWithModelCategories(
        int* eigenIndices,
        const int* probabilityIndices,
        const int* firstDerivativeIndices,
        const int* secondDerivativeIndices,
        const double* edgeLengths,
        int count) {
    return BEAGLE_ERROR_NO_IMPLEMENTATION;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::updateTransitionMatricesWithMultipleModels(
        const int* eigenIndices,
        const int* categoryRateIndices,
        const int* probabilityIndices,
        const int* firstDerivativeIndices,
        const int* secondDerivativeIndices,
        const double* edgeLengths,
        int count) {
    return BEAGLE_ERROR_NO_IMPLEMENTATION;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::updatePartials(const int* operations,
                                                                  int operationCount,
                                                                  int cumulativeScalingIndex,
                                                                  BeaglePartialsType partialsType) {
    if (!slotsReady(operations, operationCount, BEAGLE_OP_COUNT))
        return BEAGLE_ERROR_OUT_OF_RANGE;
    return BeagleCPUImpl<BEAGLE_CPU_GENERIC>::updatePartials(operations, operationCount,
                                                             cumulativeScalingIndex, partialsType);
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::updatePrePartials(const int* operations,
                                                                     int operationCount,
                                                                     int cumulativeScalingIndex,
                                                                     BeaglePartialsType partialsType) {
    return BEAGLE_ERROR_NO_IMPLEMENTATION;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::updatePartialsByPartition(const int* operations,
                                                                             int operationCount) {
    if (!slotsReady(operations, operationCount, BEAGLE_PARTITION_OP_COUNT))
        return BEAGLE_ERROR_OUT_OF_RANGE;
    return BeagleCPUImpl<BEAGLE_CPU_GENERIC>::updatePartialsByPartition(operations, operationCount);
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::updatePrePartialsByPartition(const int* operations,
                                                                                int operationCount,
                                                                                BeaglePartialsType partialsType) {
    return BEAGLE_ERROR_NO_IMPLEMENTATION;
}

BEAGLE_CPU_TEMPLATE
double BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::rootLogLikelihoodRange(const int bufferIndex,
                                                                            const int categoryWeightsIndex,
                                                                            const int stateFrequenciesIndex,
                                                                            const int scaleBufferIndex,
                                                                            int startPattern,
                                                                            int endPattern) {
    const REALTYPE* rootPartials = gPartials[bufferIndex];
    const REALTYPE* wt = gCategoryWeights[categoryWeightsIndex];
    const REALTYPE* freqs = gStateFrequencies[stateFrequenciesIndex];
    REALTYPE* site = outLogLikelihoodsTmp;

    for (int i = 0; i < kStateCount; i++) {
        REALTYPE* tmp = integrationTmp + (size_t) i * kPatternCount;
        const REALTYPE w0 = wt[0];
        const REALTYPE* row = rootPartials + pIdx(0, 0, i);
        for (int k = startPattern; k < endPattern; k++)
            tmp[k] = row[k] * w0;
        for (int l = 1; l < kCategoryCount; l++) {
            const REALTYPE wl = wt[l];
            row = rootPartials + pIdx(l, 0, i);
            for (int k = startPattern; k < endPattern; k++)
                tmp[k] += row[k] * wl;
        }
    }

    for (int k = startPattern; k < endPattern; k++)
        site[k] = 0.0;
    for (int i = 0; i < kStateCount; i++) {
        const REALTYPE f = freqs[i];
        const REALTYPE* tmp = integrationTmp + (size_t) i * kPatternCount;
        for (int k = startPattern; k < endPattern; k++)
            site[k] += f * tmp[k];
    }
    for (int k = startPattern; k < endPattern; k++)
        site[k] = log(site[k]);

    if (scaleBufferIndex >= 0) {
        const REALTYPE* cumulativeScaleFactors = gScaleBuffers[scaleBufferIndex];
        for (int k = startPattern; k < endPattern; k++)
            site[k] += cumulativeScaleFactors[k];
    }

    double sum = 0.0;
    for (int k = startPattern; k < endPattern; k++)
        sum += site[k] * gPatternWeights[k];
    return sum;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::calcRootLogLikelihoods(const int bufferIndex,
                                                                          const int categoryWeightsIndex,
                                                                          const int stateFrequenciesIndex,
                                                                          const int scaleBufferIndex,
                                                                          double* outSumLogLikelihood) {
    *outSumLogLikelihood = rootLogLikelihoodRange(bufferIndex, categoryWeightsIndex, stateFrequenciesIndex,
                                                  scaleBufferIndex, 0, kPatternCount);
    if (*outSumLogLikelihood != *outSumLogLikelihood)
        return BEAGLE_ERROR_FLOATING_POINT;
    return BEAGLE_SUCCESS;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::calcRootLogLikelihoodsPerCategory(const int bufferIndex,
                                                                                     const int stateFrequenciesIndex,
                                                                                     const int scaleBufferIndex,
                                                                                     double* outSumLogLikelihood) {
    return BEAGLE_ERROR_NO_IMPLEMENTATION;
}

BEAGLE_CPU_TEMPLATE
void BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::calcRootLogLikelihoodsByPartition(
        const int* bufferIndices,
        const int* categoryWeightsIndices,
        const int* stateFrequenciesIndices,
        const int* cumulativeScaleIndices,
        const int* partitionIndices,
        int partitionCount,
        double* outSumLogLikelihoodByPartition) {
    for (int p = 0; p < partitionCount; p++) {
        const int pIndex = partitionIndices[p];
        outSumLogLikelihoodByPartition[p] = rootLogLikelihoodRange(bufferIndices[p], categoryWeightsIndices[p],
                                                                   stateFrequenciesIndices[p],
                                                                   cumulativeScaleIndices[p],
                                                                   gPatternPartitionsStartPatterns[pIndex],
                                                                   gPatternPartitionsStartPatterns[pIndex + 1]);
    }
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::calcRootLogLikelihoodsMulti(const int* bufferIndices,
                                                                               const int* categoryWeightsIndices,
                                                                               const int* stateFrequenciesIndices,
                                                                               const int* scaleBufferIndices,
                                                                               int count,
                                                                               double* outSumLogLikelihood) {
    return BEAGLE_ERROR_NO_IMPLEMENTATION;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::calculateEdgeLogLikelihoods(const int* parentBufferIndices,
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
    return BEAGLE_ERROR_NO_IMPLEMENTATION;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::calculateEdgeLogLikelihoodsByPartition(
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
    return BEAGLE_ERROR_NO_IMPLEMENTATION;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::calculateEdgeDerivatives(const int* postBufferIndices,
                                                                            const int* preBufferIndices,
                                                                            const int* derivativeMatrixIndices,
                                                                            const int* categoryWeightsIndices,
                                                                            const int* categoryRatesIndices,
                                                                            const int* cumulativeScaleIndices,
                                                                            int count,
                                                                            double* outDerivatives,
                                                                            double* outSumDerivatives,
                                                                            double* outSumSquaredDerivatives) {
    return BEAGLE_ERROR_NO_IMPLEMENTATION;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::calculateCrossProducts(const int* postBufferIndices,
                                                                          const int* preBufferIndices,
                                                                          const int* categoryRatesIndices,
                                                                          const int* categoryWeightsIndices,
                                                                          const double* edgeLengths,
                                                                          int count,
                                                                          double* outSumDerivatives,
                                                                          double* outSumSquaredDerivatives) {
    return BEAGLE_ERROR_NO_IMPLEMENTATION;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::calculateAdjointCrossProducts(const int* postBufferIndices,
                                                                                 const int* preBufferIndices,
                                                                                 const int* eigenIndices,
                                                                                 const int* categoryRatesIndices,
                                                                                 const int* categoryWeightsIndices,
                                                                                 const int rootPostOrderIndex,
                                                                                 const int stateFrequenciesIndex,
                                                                                 int count,
                                                                                 double* outSumDerivatives,
                                                                                 double* outSumSquaredDerivatives,
                                                                                 const int* postScaleIndices,
                                                                                 const int* preScaleIndices,
                                                                                 const int cumulativeScaleIndex) {
    return BEAGLE_ERROR_NO_IMPLEMENTATION;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::calculateAdjointDerivative(const BeagleBranchOperation* operations,
                                                                              const int categoryRatesIndex,
                                                                              const int categoryWeightsIndex,
                                                                              const int rootPostOrderIndex,
                                                                              const int stateFrequenciesIndex,
                                                                              const int operationCount,
                                                                              double* outSumDerivatives,
                                                                              double* outSumSquaredDerivatives) {
    return BEAGLE_ERROR_NO_IMPLEMENTATION;
}

BEAGLE_CPU_TEMPLATE
bool BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::slotsReady(const int* operations, int count, int numOps) {
    if (gStateFrequencies[0] == NULL)
        return false;
    for (int op = 0; op < count; op++) {
        const int* operation = operations + op * numOps;
        const int child2 = operation[5];
        const int last = (child2 == BEAGLE_OP_NONE) ? 4 : 6;
        for (int m = 4; m <= last; m += 2) {
            const int matrix = operation[m];
            if (matrix < 0 || matrix >= (int) gSlots.size() || !gSlots[matrix].set)
                return false;
        }
    }
    return true;
}

BEAGLE_CPU_TEMPLATE template <int Mode, typename Source>
void BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::sweep(const HierarchicalSlot<REALTYPE>& slot,
                                                          const Source& source,
                                                          REALTYPE* destP,
                                                          const REALTYPE* invScale,
                                                          int category,
                                                          int startPattern,
                                                          int endPattern) {
    hierarchicalSweep<Mode>(slot, kStateCount, category, gStateFrequencies[0], source.category(category),
                            destP + pIdx(category, 0, 0), kPaddedPatternCount, invScale,
                            gW.data(), gV.data(), kPaddedPatternCount, startPattern, endPattern);
}

BEAGLE_CPU_TEMPLATE template <typename Source1, typename Source2>
void BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::hierarchicalPartials(REALTYPE* destP,
                                                                         const Source1& source1,
                                                                         const HierarchicalSlot<REALTYPE>& slot1,
                                                                         const Source2* source2,
                                                                         const HierarchicalSlot<REALTYPE>* slot2,
                                                                         const REALTYPE* scaleFactors,
                                                                         int startPattern,
                                                                         int endPattern) {
    REALTYPE* invScale = NULL;
    if (scaleFactors != NULL) {
        invScale = gInvScaleTmp.data();
        for (int k = startPattern; k < endPattern; k++)
            invScale[k] = REALTYPE(1.0) / scaleFactors[k];
    }

    for (int k0 = startPattern; k0 < endPattern; k0 += kTileSize) {
        const int k1 = std::min(k0 + kTileSize, endPattern);
        for (int l = 0; l < kCategoryCount; l++) {
            if (source2 == NULL) {
                if (invScale != NULL)
                    sweep<kHierarchicalWriteScaled>(slot1, source1, destP, invScale, l, k0, k1);
                else
                    sweep<kHierarchicalWrite>(slot1, source1, destP, invScale, l, k0, k1);
            } else {
                sweep<kHierarchicalWrite>(slot1, source1, destP, invScale, l, k0, k1);
                if (invScale != NULL)
                    sweep<kHierarchicalMultiplyScaled>(*slot2, *source2, destP, invScale, l, k0, k1);
                else
                    sweep<kHierarchicalMultiply>(*slot2, *source2, destP, invScale, l, k0, k1);
            }
        }
    }
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::upPartials(bool byPartition,
                                                              const int* operations,
                                                              int count,
                                                              int cumulativeScaleIndex,
                                                              BeaglePartialsType partialsType) {

    if (partialsType & BEAGLE_PARTIALS_TOP)
        return BEAGLE_ERROR_NO_IMPLEMENTATION;

    const int numOps = byPartition ? BEAGLE_PARTITION_OP_COUNT : BEAGLE_OP_COUNT;
    if (!slotsReady(operations, count, numOps))
        return BEAGLE_ERROR_OUT_OF_RANGE;

    REALTYPE* cumulativeScaleBuffer = NULL;
    if (cumulativeScaleIndex != BEAGLE_OP_NONE)
        cumulativeScaleBuffer = gScaleBuffers[cumulativeScaleIndex];

    for (int op = 0; op < count; op++) {
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

        const bool singleChild = (child2Index == BEAGLE_OP_NONE);

        const REALTYPE* partials1 = gPartials[child1Index];
        const REALTYPE* partials2 = singleChild ? NULL : gPartials[child2Index];

        const int* tipStates1 = gTipStates[child1Index];
        const int* tipStates2 = singleChild ? NULL : gTipStates[child2Index];

        const HierarchicalSlot<REALTYPE>& slot1 = gSlots[child1TransMatIndex];
        const HierarchicalSlot<REALTYPE>* slot2 = singleChild ? NULL : &gSlots[child2TransMatIndex];

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
            if (tipStates1 == 0 && tipStates2 == 0 && !singleChild)
                rescale = 2;
        } else if (kFlags & BEAGLE_FLAG_SCALING_ALWAYS) {
            rescale = 1;
            scalingFactors = gScaleBuffers[parIndex - kTipCount];
        } else if (kFlags & BEAGLE_FLAG_SCALING_DYNAMIC) {
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

        const REALTYPE* fixedScaleFactors = (rescale == 0) ? scalingFactors : NULL;
        const HierarchicalPartialsSource<REALTYPE> partialsSource2 =
                { partials2, kPaddedPatternCount, kPartialsPaddedStateCount * kPaddedPatternCount };
        const HierarchicalStatesSource<REALTYPE> statesSource2 = { tipStates2, kStateCount };

        auto withChild1 = [&](const auto& source1) {
            if (singleChild)
                hierarchicalPartials(destPartials, source1, slot1,
                                     (const HierarchicalStatesSource<REALTYPE>*) NULL, slot2,
                                     fixedScaleFactors, startPattern, endPattern);
            else if (tipStates2 != NULL)
                hierarchicalPartials(destPartials, source1, slot1, &statesSource2, slot2,
                                     fixedScaleFactors, startPattern, endPattern);
            else
                hierarchicalPartials(destPartials, source1, slot1, &partialsSource2, slot2,
                                     fixedScaleFactors, startPattern, endPattern);
        };

        if (tipStates1 != NULL) {
            const HierarchicalStatesSource<REALTYPE> source1 = { tipStates1, kStateCount };
            withChild1(source1);
        } else {
            const HierarchicalPartialsSource<REALTYPE> source1 =
                    { partials1, kPaddedPatternCount, kPartialsPaddedStateCount * kPaddedPatternCount };
            withChild1(source1);
        }

        if (rescale == 1) {
            if (byPartition)
                rescalePartialsByPartition(destPartials, scalingFactors, cumulativeScaleBuffer, 0, currentPartition);
            else
                rescalePartials(destPartials, scalingFactors, cumulativeScaleBuffer, 0);
        } else if (rescale == 2) {
            fprintf(stderr, "BEAGLE_FLAG_SCALING_AUTO is not yet implemented for hierarchical representation\n");
            return BEAGLE_ERROR_NO_IMPLEMENTATION;
        }

        if (kFlags & BEAGLE_FLAG_SCALING_ALWAYS) {
            int parScalingIndex = parIndex - kTipCount;
            int child1ScalingIndex = child1Index - kTipCount;
            int child2ScalingIndex = singleChild ? -1 : child2Index - kTipCount;
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
    }

    return BEAGLE_SUCCESS;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::upPrePartials(bool byPartition,
                                                                 const int* operations,
                                                                 int count,
                                                                 int cumulativeScaleIndex,
                                                                 BeaglePartialsType preorderType) {
    return BEAGLE_ERROR_NO_IMPLEMENTATION;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::reorderPatternsByPartition() {

    if (!kPatternsReordered) {
        gPatternsNewOrder = (int*) malloc(kPatternCount * sizeof(int));
    } else {
        return BEAGLE_ERROR_NO_IMPLEMENTATION;
    }

    int* partitionSizes = (int*) malloc(kPartitionCount * sizeof(int));
    double* sortedPatternWeights = (double*) malloc(sizeof(double) * kPatternCount);

    for (int i = 0; i < kPartitionCount; i++) {
        gPatternPartitionsStartPatterns[i] = 0;
        partitionSizes[i] = 0;
    }

    for (int i = 0; i < kPatternCount; i++) {
        gPatternsNewOrder[i] = partitionSizes[gPatternPartitions[i]]++;
    }

    for (int i = 0; i < kPartitionCount; i++) {
        for (int j = 0; j < i; j++) {
            gPatternPartitionsStartPatterns[i] += partitionSizes[j];
        }
    }
    gPatternPartitionsStartPatterns[kPartitionCount] = kPatternCount;

    for (int i = 0; i < kPatternCount; i++) {
        gPatternsNewOrder[i] += gPatternPartitionsStartPatterns[gPatternPartitions[i]];
        sortedPatternWeights[gPatternsNewOrder[i]] = gPatternWeights[i];
    }

    int currentPattern = 0;
    for (int i = 0; i < kPartitionCount; i++) {
        for (int j = 0; j < partitionSizes[i]; j++) {
            gPatternPartitions[currentPattern++] = i;
        }
    }

    free(partitionSizes);
    free(gPatternWeights);
    gPatternWeights = sortedPatternWeights;

    REALTYPE* sortedPartials = (REALTYPE*) mallocAligned(sizeof(REALTYPE) * kPartialsSize);
    int* sortedTips = (int*) mallocAligned(sizeof(int) * kPaddedPatternCount);

    for (int tip = 0; tip < kTipCount; tip++) {
        if (gTipStates[tip] == NULL) {
            REALTYPE* unsortedPartials = gPartials[tip];
            if (unsortedPartials == NULL)
                continue;
            for (int l = 0; l < kCategoryCount; l++) {
                for (int i = 0; i < kPartialsPaddedStateCount; i++) {
                    const REALTYPE* from = unsortedPartials + pIdx(l, 0, i);
                    REALTYPE* to = sortedPartials + pIdx(l, 0, i);
                    for (int k = 0; k < kPatternCount; k++)
                        to[gPatternsNewOrder[k]] = from[k];
                    for (int k = kPatternCount; k < kPaddedPatternCount; k++)
                        to[k] = from[k];
                }
            }
            gPartials[tip] = sortedPartials;
            sortedPartials = unsortedPartials;
        } else {
            int* unsortedTips = gTipStates[tip];
            for (int i = 0; i < kPatternCount; i++) {
                sortedTips[gPatternsNewOrder[i]] = unsortedTips[i];
            }
            for (int i = kPatternCount; i < kPaddedPatternCount; i++) {
                sortedTips[i] = unsortedTips[i];
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

BEAGLE_CPU_FACTORY_TEMPLATE
BeagleImpl* BeagleCPUHierarchicalImplFactory<BEAGLE_CPU_FACTORY_GENERIC>::createImpl(int tipCount,
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

    BeagleImpl* impl = new BeagleCPUHierarchicalImpl<REALTYPE, T_PAD_DEFAULT, P_PAD_DEFAULT>();

    try {
        if (impl->createInstance(tipCount, partialsBufferCount, compactBufferCount, stateCount,
                                 patternCount, eigenBufferCount, matrixBufferCount,
                                 categoryCount, scaleBufferCount, resourceNumber,
                                 pluginResourceNumber,
                                 preferenceFlags, requirementFlags) == 0)
            return impl;
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
const char* BeagleCPUHierarchicalImplFactory<BEAGLE_CPU_FACTORY_GENERIC>::getName() {
    return getBeagleCPUHierarchicalName<BEAGLE_CPU_FACTORY_GENERIC>();
}

BEAGLE_CPU_FACTORY_TEMPLATE
const long BeagleCPUHierarchicalImplFactory<BEAGLE_CPU_FACTORY_GENERIC>::getFlags() {
    return getBeagleCPUHierarchicalFlags<BEAGLE_CPU_FACTORY_GENERIC>();
}

}
}

#endif
