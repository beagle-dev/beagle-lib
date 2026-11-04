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

#include <chrono>
#include <cstdlib>
#include <cstring>
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
                 BEAGLE_FLAG_HIERARCHICAL_AGGREGATE_ON | BEAGLE_FLAG_HIERARCHICAL_AGGREGATE_OFF |
                 BEAGLE_FLAG_FRAMEWORK_CPU;
    if (DOUBLE_PRECISION)
        flags |= BEAGLE_FLAG_PRECISION_DOUBLE;
    else
        flags |= BEAGLE_FLAG_PRECISION_SINGLE;
    return flags;
}

BEAGLE_CPU_TEMPLATE
BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::~BeagleCPUHierarchicalImpl() {
    for (REALTYPE* ragged : gRagged)
        free(ragged);
    for (auto& tips : gTipRagged)
        for (REALTYPE* ragged : tips)
            free(ragged);
    free(gRaggedW);
    free(gRaggedV);
}

BEAGLE_CPU_TEMPLATE
const char* BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::getName() {
    return getBeagleCPUHierarchicalName<BEAGLE_CPU_FACTORY_GENERIC>();
}

BEAGLE_CPU_TEMPLATE
const long BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::getFlags() {
    return getBeagleCPUHierarchicalFlags<BEAGLE_CPU_FACTORY_GENERIC>() &
           ~(BEAGLE_FLAG_HIERARCHICAL_AGGREGATE_ON | BEAGLE_FLAG_HIERARCHICAL_AGGREGATE_OFF);
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
    const long aggregateFlags = (preferenceFlags | requirementFlags) &
                                (BEAGLE_FLAG_HIERARCHICAL_AGGREGATE_ON | BEAGLE_FLAG_HIERARCHICAL_AGGREGATE_OFF);
    int returnCode = BeagleCPUImpl<BEAGLE_CPU_GENERIC>::createInstance(tipCount, partialsBufferCount,
                                                                       compactBufferCount, stateCount,
                                                                       patternCount, eigenDecompositionCount,
                                                                       matrixCount, categoryCount,
                                                                       scaleBufferCount, resourceNumber,
                                                                       pluginResourceNumber,
                                                                       preferenceFlags, requirementFlags);
    if (aggregateFlags == (BEAGLE_FLAG_HIERARCHICAL_AGGREGATE_ON | BEAGLE_FLAG_HIERARCHICAL_AGGREGATE_OFF)) {
        fprintf(stderr, "BEAGLE_FLAG_HIERARCHICAL_AGGREGATE_ON and BEAGLE_FLAG_HIERARCHICAL_AGGREGATE_OFF exclude each other\n");
        return BEAGLE_ERROR_OUT_OF_RANGE;
    }
    kFlags &= ~(BEAGLE_FLAG_EIGEN_REAL | BEAGLE_FLAG_EIGEN_COMPLEX |
                BEAGLE_FLAG_INVEVEC_STANDARD | BEAGLE_FLAG_INVEVEC_TRANSPOSED);
    gSlots.assign(kMatrixCount, HierarchicalSlot<REALTYPE>());
    gModels.assign(kEigenDecompCount, HierarchicalModel());
    gFrequenciesDouble.assign(kEigenDecompCount, std::vector<double>());
    gPiNodeTmp.assign(2 * kStateCount - 1, 0.0);
    gPrZ0Tmp.assign(2 * kStateCount - 1, 0.0);
    gSumHTmp.assign(2 * kStateCount - 1, 0.0);
    gFrequencyStamp = 0;
    gFrequencyStamps.assign(kEigenDecompCount, 0);
    gTopologyParents.clear();
    gTopologyIds.clear();
    gNextTopologyId = 0;
    gPiNodeByFrequency.assign(kEigenDecompCount, std::vector<double>());
    gPiNodeTopology.assign(kEigenDecompCount, -1);
    gPiNodeStamp.assign(kEigenDecompCount, 0);
    kKernel = kHierarchicalKernelShape;
    const char* kernel = getenv("BEAGLE_HRATE_KERNEL");
    if (kernel != NULL && std::strcmp(kernel, "tiled") == 0)
        kKernel = kHierarchicalKernelTiled;
    else if (kernel != NULL && std::strcmp(kernel, "dense") == 0)
        kKernel = kHierarchicalKernelDense;
    else if (kernel != NULL && std::strcmp(kernel, "fused") == 0)
        kKernel = kHierarchicalKernelFused;
    kGroupLanes = hierarchicalGroupLanes(kStateCount, sizeof(REALTYPE));
    if (const char* groupEnv = getenv("BEAGLE_HRATE_GROUP")) {
        const int g = atoi(groupEnv);
        const int line = (int) (128 / sizeof(REALTYPE));
        if (g >= line && g <= kHierarchicalGroupMaxLanes && g % line == 0 &&
            (size_t) (kStateCount - 1) * g * sizeof(REALTYPE) <= kHierarchicalGroupScratchBytes)
            kGroupLanes = g;
    }
    kGroupOrder = kHierarchicalOrderCategory;
    if (const char* orderEnv = getenv("BEAGLE_HRATE_ORDER"))
        if (std::strcmp(orderEnv, "group") == 0)
            kGroupOrder = kHierarchicalOrderGroup;
    const bool fusedDefault = kKernel == kHierarchicalKernelShape && kStateCount >= kHierarchicalFusedDefaultMinStates &&
                              kPatternCount >= kHierarchicalFusedMinPatterns;
    kFusedBand = (kKernel == kHierarchicalKernelFused || fusedDefault) && kGroupLanes > 0;
    const char* gatherEnv = getenv("BEAGLE_HRATE_GATHER");
    kColumnGather = kFusedBand && kStateCount <= kHierarchicalColumnMaxStates &&
                    !(gatherEnv != NULL && std::strcmp(gatherEnv, "off") == 0);
    kTableBudget = kHierarchicalTableBudget;
    if (const char* budgetEnv = getenv("BEAGLE_HRATE_TABLE_BUDGET"))
        kTableBudget = (size_t) std::strtoull(budgetEnv, NULL, 10);
    gTableBytes = 0;
    kShapeMaxStates = kHierarchicalShapeMaxStates;
    if (const char* shapeEnv = getenv("BEAGLE_HRATE_SHAPE_MAX"))
        kShapeMaxStates = std::max(0, std::min(atoi(shapeEnv), kHierarchicalShapeBandLimit));
    kSmallBand = (kKernel == kHierarchicalKernelShape || kKernel == kHierarchicalKernelDense) &&
                 kStateCount <= kShapeMaxStates && hierarchicalShapeCompiled(kStateCount);
    const char* rescale = getenv("BEAGLE_HRATE_RESCALE");
    kRescaleRows = kStateCount * kCategoryCount <= kHierarchicalRescaleMaxRows &&
                   !(rescale != NULL && std::strcmp(rescale, "legacy") == 0);
    gW.assign((size_t) (kStateCount - 1) * kPaddedPatternCount, REALTYPE(0));
    gV.assign((size_t) (kStateCount - 1) * kPaddedPatternCount, REALTYPE(0));
    gInvScaleTmp.assign(kPaddedPatternCount, REALTYPE(0));
    const int modulus = getPaddedPatternsModulus();
    int tile = (int) (kHierarchicalTileBudget / (3 * (size_t) kStateCount * sizeof(REALTYPE)));
    if (const char* tileEnv = getenv("BEAGLE_HRATE_TILE"))
        if (atoi(tileEnv) > 0)
            tile = atoi(tileEnv);
    tile = std::max(64, std::min(tile, kPaddedPatternCount));
    kTileSize = std::max(modulus, (tile / modulus) * modulus);
    gSiteScaleTmp.assign(kPaddedPatternCount, REALTYPE(0));
    gSiteAutoScaleTmp.assign(kPaddedPatternCount, 0.0);
    kAggregateMode = kHierarchicalAggregateAuto;
    if (aggregateFlags == BEAGLE_FLAG_HIERARCHICAL_AGGREGATE_ON)
        kAggregateMode = kHierarchicalAggregateOn;
    else if (aggregateFlags == BEAGLE_FLAG_HIERARCHICAL_AGGREGATE_OFF)
        kAggregateMode = kHierarchicalAggregateOff;
    if (const char* aggregateEnv = getenv("BEAGLE_HRATE_AGGREGATE")) {
        if (std::strcmp(aggregateEnv, "0") == 0)
            kAggregateMode = kHierarchicalAggregateOff;
        else if (std::strcmp(aggregateEnv, "1") == 0)
            kAggregateMode = kHierarchicalAggregateOn;
    }
    if (kAggregateMode == kHierarchicalAggregateOn)
        kFlags |= BEAGLE_FLAG_HIERARCHICAL_AGGREGATE_ON;
    else if (kAggregateMode == kHierarchicalAggregateOff)
        kFlags |= BEAGLE_FLAG_HIERARCHICAL_AGGREGATE_OFF;
    kMayAggregate = kAggregateMode == kHierarchicalAggregateOn ||
                    (kAggregateMode == kHierarchicalAggregateAuto && kStateCount >= kHierarchicalAggregateMinStates &&
                     !kSmallBand && !kFusedBand);
    const char* verboseEnv = getenv("BEAGLE_HRATE_VERBOSE");
    kVerbose = verboseEnv != NULL && std::strcmp(verboseEnv, "0") != 0;
    gTipPlanesEqual.assign(kTipCount, 0);
    gRagged.assign(kBufferCount, nullptr);
    gRaggedCapacity.assign(kBufferCount, 0);
    gRaggedGeneration.assign(kBufferCount, 0);
    gRaggedPre.assign(kBufferCount, 0);
    gRaggedTopology.assign(kBufferCount, -1);
    gBufferTopology.assign(kBufferCount, -1);
    const char* supportEnv = getenv("BEAGLE_HRATE_SUPPORT");
    kSupportBuild = !(supportEnv != NULL && std::strcmp(supportEnv, "0") == 0);
    gModelStamp.assign(kEigenDecompCount, 0);
    gBuildInputs.assign(kEigenDecompCount, nullptr);
    gBuildInputsModelStamp.assign(kEigenDecompCount, 0);
    gBuildInputsFrequencyStamp.assign(kEigenDecompCount, 0);
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
void BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::finishPattern(REALTYPE* dest,
                                                                  int n,
                                                                  int rescale,
                                                                  const REALTYPE* fixedScaleFactors,
                                                                  REALTYPE* scaleFactors,
                                                                  REALTYPE* cumulativeScaleFactors,
                                                                  int k) {
    if (rescale == 0) {
        const REALTYPE inv = REALTYPE(1.0) / fixedScaleFactors[k];
        for (int i = 0; i < n; i++)
            dest[i] *= inv;
    } else if (rescale == 1) {
        REALTYPE max = 0;
        for (int i = 0; i < n; i++)
            max = dest[i] > max ? dest[i] : max;
        if (max == 0)
            max = 1.0;
        const REALTYPE inv = REALTYPE(1.0) / max;
        for (int i = 0; i < n; i++)
            dest[i] *= inv;
        const REALTYPE logMax = log(max);
        scaleFactors[k] = (kFlags & BEAGLE_FLAG_SCALERS_LOG) ? logMax : max;
        if (cumulativeScaleFactors != NULL)
            cumulativeScaleFactors[k] += logMax;
    }
}

BEAGLE_CPU_TEMPLATE
void BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::aggregatedPartials(int parIndex,
                                                                       int child1Index,
                                                                       const int* states1,
                                                                       const HierarchicalSlot<REALTYPE>& slot1,
                                                                       bool singleChild,
                                                                       int child2Index,
                                                                       const int* states2,
                                                                       const HierarchicalSlot<REALTYPE>* slot2,
                                                                       int rescale,
                                                                       const REALTYPE* fixedScaleFactors,
                                                                       REALTYPE* scaleFactors,
                                                                       REALTYPE* cumulativeScaleFactors,
                                                                       int startPattern,
                                                                       int endPattern) {
    const int live = liveSlotOf(slot1.topologyId);
    REALTYPE* destRagged = gRagged[parIndex];
    const REALTYPE* ragged1 = states1 ? NULL : (child1Index < kTipCount ? gTipRagged[live][child1Index]
                                                                        : gRagged[child1Index]);
    const REALTYPE* ragged2 = singleChild || states2 ? NULL : (child2Index < kTipCount ? gTipRagged[live][child2Index]
                                                                                       : gRagged[child2Index]);
    for (int k = startPattern; k < endPattern; k++) {
        const HierarchicalReduced<REALTYPE>& red = gClasses[gClassOfPattern[k]].reduced[live];
        const int Sc = red.stateCount;
        const size_t off = gPatternOffset[k];
        REALTYPE* dest = destRagged + off;
        const REALTYPE* rows1 = ragged1 ? ragged1 + off : NULL;
        const REALTYPE* rows2 = ragged2 ? ragged2 + off : NULL;
        const int s1 = states1 ? states1[k] : -1;
        const int s2 = states2 ? states2[k] : -1;
        if (Sc == 1) {
            for (int l = 0; l < kCategoryCount; l++)
                dest[l] = (rows1 ? rows1[l] : REALTYPE(1)) * (rows2 ? rows2[l] : REALTYPE(1));
        } else {
            REALTYPE* W = gRaggedW + off;
            REALTYPE* V = gRaggedV + off;
            hierarchicalLaneSweep<false>(slot1, red, kStateCount, kCategoryCount, s1, rows1, dest, W, V);
            if (!singleChild)
                hierarchicalLaneSweep<true>(*slot2, red, kStateCount, kCategoryCount, s2, rows2, dest, W, V);
        }
        finishPattern(dest, Sc * kCategoryCount, rescale, fixedScaleFactors, scaleFactors, cumulativeScaleFactors, k);
    }
}

BEAGLE_CPU_TEMPLATE
void BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::rescaleRange(REALTYPE* destP,
                                                                 REALTYPE* scaleFactors,
                                                                 REALTYPE* cumulativeScaleFactors,
                                                                 int startPattern,
                                                                 int endPattern) {
    REALTYPE* siteMax = gSiteScaleTmp.data();
    if (kRescaleRows) {
        REALTYPE* rows[kHierarchicalRescaleMaxRows];
        for (int l = 0; l < kCategoryCount; l++)
            for (int i = 0; i < kStateCount; i++)
                rows[l * kStateCount + i] = destP + pIdx(l, 0, i);
        HierarchicalRowsDispatch<REALTYPE>::rescale(kStateCount * kCategoryCount, rows, siteMax, startPattern,
                                                    endPattern);
        for (int k = startPattern; k < endPattern; k++) {
            const REALTYPE max = siteMax[k];
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
        }
        return;
    }

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
int BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::setTipStates(int tipIndex,
                                                                const int* inStates) {
    int returnCode = BeagleCPUImpl<BEAGLE_CPU_GENERIC>::setTipStates(tipIndex, inStates);
    if (returnCode != BEAGLE_SUCCESS)
        return returnCode;
    if (kPatternsReordered) {
        std::vector<int> unsorted(gTipStates[tipIndex], gTipStates[tipIndex] + kPatternCount);
        for (int k = 0; k < kPatternCount; k++)
            gTipStates[tipIndex][gPatternsNewOrder[k]] = unsorted[k];
    }
    gTipStamp++;
    return BEAGLE_SUCCESS;
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
    if (tipIndex < (int) gTipPlanesEqual.size())
        gTipPlanesEqual[tipIndex] = 1;
    gTipStamp++;
    return BEAGLE_SUCCESS;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::setPatternWeights(const double* inPatternWeights) {
    if (!kPatternsReordered)
        return BeagleCPUImpl<BEAGLE_CPU_GENERIC>::setPatternWeights(inPatternWeights);
    for (int k = 0; k < kPatternCount; k++)
        gPatternWeights[gPatternsNewOrder[k]] = inPatternWeights[k];
    return BEAGLE_SUCCESS;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::setPatternPartitions(int partitionCount,
                                                                        const int* inPatternPartitions) {
    const bool reordered = kPatternsReordered;
    if (gAggregated)
        expandRagged(-1);
    int returnCode = BeagleCPUImpl<BEAGLE_CPU_GENERIC>::setPatternPartitions(partitionCount, inPatternPartitions);
    if (kPatternsReordered != reordered)
        invalidateClasses(false);
    return returnCode;
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
    if (bufferIndex < kTipCount) {
        if (bufferIndex < (int) gTipPlanesEqual.size())
            gTipPlanesEqual[bufferIndex] = 0;
        gTipStamp++;
    } else {
        gBufferTopology[bufferIndex] = -1;
        gRaggedGeneration[bufferIndex] = 0;
        if (gAggregated) {
            REALTYPE* ragged = ensureRagged(bufferIndex);
            if (ragged == NULL)
                return BEAGLE_ERROR_OUT_OF_MEMORY;
            compressRange(gPartials[bufferIndex], ragged, gCurrentLive, 0, kPatternCount);
            gRaggedGeneration[bufferIndex] = gClassGeneration;
            gRaggedPre[bufferIndex] = 0;
            gRaggedTopology[bufferIndex] = gLiveId[gCurrentLive];
            if (kVerbose)
                fprintf(stderr, "hrate aggregation: setPartials(%d) compressed, max deviation from constancy %.3g\n",
                        bufferIndex, constancyDeviation(gPartials[bufferIndex], gCurrentLive));
        }
    }
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
    if (bufferIndex >= kTipCount) {
        gBufferTopology[bufferIndex] = -1;
        gRaggedGeneration[bufferIndex] = 0;
    }
    const double* piNode = gAggregated && bufferIndex >= kTipCount ?
                           nodeFrequencies(stateFrequenciesIndices[0], gLiveOps[gCurrentLive]) : NULL;
    if (piNode != NULL) {
        REALTYPE* ragged = ensureRagged(bufferIndex);
        if (ragged == NULL)
            return BEAGLE_ERROR_OUT_OF_MEMORY;
        gRaggedTopology[bufferIndex] = gLiveId[gCurrentLive];
        for (int k = 0; k < kPatternCount; k++) {
            const HierarchicalReduced<REALTYPE>& red = gClasses[gClassOfPattern[k]].reduced[gCurrentLive];
            REALTYPE* p = ragged + gPatternOffset[k];
            for (int x = 0; x < red.stateCount; x++)
                for (int l = 0; l < kCategoryCount; l++)
                    p[x * kCategoryCount + l] = (REALTYPE) piNode[red.localToFull[x]];
        }
        gRaggedGeneration[bufferIndex] = gClassGeneration;
        gRaggedPre[bufferIndex] = 1;
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

    if (raggedReady(bufferIndex)) {
        const REALTYPE* ragged = gRagged[bufferIndex];
        const int live = liveSlotOf(gRaggedTopology[bufferIndex]);
        std::vector<int> rowOf;
        int cached = -1;
        for (int k = 0; k < kPatternCount; k++) {
            const int c = gClassOfPattern[k];
            if (c != cached) {
                hierarchicalRowOfState(gLiveOps[live], kStateCount, gClasses[c].reduced[live], rowOf);
                cached = c;
            }
            const REALTYPE* p = ragged + gPatternOffset[k];
            for (int l = 0; l < kCategoryCount; l++) {
                double* out = outPartials + ((size_t) l * kPatternCount + k) * kStateCount;
                for (int i = 0; i < kStateCount; i++)
                    out[i] = (double) p[rowOf[i] * kCategoryCount + l];
            }
        }
    } else {
        transposeFromSiteMajor(outPartials, gPartials[bufferIndex]);
    }

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
    if (eigenIndex < 0 || eigenIndex >= kEigenDecompCount)
        return BEAGLE_ERROR_OUT_OF_RANGE;
    int returnCode = decodeHierarchicalModel(inEigenVectors, inEigenValues, kStateCount, gModels[eigenIndex]);
    gModelStamp[eigenIndex]++;
    if (returnCode == BEAGLE_SUCCESS && kMayAggregate)
        gModels[eigenIndex].ops.topologyId = topologyIdOf(gModels[eigenIndex].ops);
    if (returnCode == BEAGLE_SUCCESS && kSmallBand)
        hierarchicalDetectShape(gModels[eigenIndex].ops, kStateCount);
    else if (returnCode == BEAGLE_SUCCESS && kColumnGather)
        hierarchicalComputeMrca(gModels[eigenIndex].ops, kStateCount);
    return returnCode;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::setStateFrequencies(int stateFrequenciesIndex,
                                                                       const double* inStateFrequencies) {
    int returnCode = BeagleCPUImpl<BEAGLE_CPU_GENERIC>::setStateFrequencies(stateFrequenciesIndex,
                                                                            inStateFrequencies);
    if (returnCode != BEAGLE_SUCCESS)
        return returnCode;
    gFrequenciesDouble[stateFrequenciesIndex].assign(inStateFrequencies, inStateFrequencies + kStateCount);
    gFrequencyStamps[stateFrequenciesIndex]++;
    if (stateFrequenciesIndex == 0)
        gFrequencyStamp++;
    return BEAGLE_SUCCESS;
}

BEAGLE_CPU_TEMPLATE
std::shared_ptr<const HierarchicalBuildInputs> BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::buildInputs(
        int modelIndex,
        int categoryRatesIndex) {
    const HierarchicalModel& model = gModels[modelIndex];
    const double* rates = gCategoryRates[categoryRatesIndex];
    std::shared_ptr<const HierarchicalBuildInputs>& cached = gBuildInputs[modelIndex];
    if (cached && gBuildInputsModelStamp[modelIndex] == gModelStamp[modelIndex] &&
        gBuildInputsFrequencyStamp[modelIndex] == gFrequencyStamp &&
        std::equal(cached->rates.begin(), cached->rates.end(), rates))
        return cached;
    std::shared_ptr<HierarchicalBuildInputs> inputs = std::make_shared<HierarchicalBuildInputs>();
    inputs->lambda = model.lambda;
    inputs->piNode.resize(2 * kStateCount - 1);
    hierarchicalNodeFrequencies(model, gFrequenciesDouble[0].data(), kStateCount, inputs->piNode.data());
    for (int m = kStateCount; m < 2 * kStateCount - 1; m++)
        if (!std::isfinite(inputs->piNode[m]) || inputs->piNode[m] <= 0.0)
            return nullptr;
    inputs->rates.assign(rates, rates + kCategoryCount);
    cached = inputs;
    gBuildInputsModelStamp[modelIndex] = gModelStamp[modelIndex];
    gBuildInputsFrequencyStamp[modelIndex] = gFrequencyStamp;
    return cached;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::buildBundles(int modelIndex,
                                                                int categoryRatesIndex,
                                                                const int* probabilityIndices,
                                                                const double* edgeLengths,
                                                                int count) {
    if (modelIndex < 0 || modelIndex >= kEigenDecompCount || !gModels[modelIndex].set ||
        categoryRatesIndex < 0 || categoryRatesIndex >= kEigenDecompCount ||
        gCategoryRates[categoryRatesIndex] == NULL || gFrequenciesDouble[0].empty())
        return BEAGLE_ERROR_OUT_OF_RANGE;
    const HierarchicalModel& model = gModels[modelIndex];
    const int live = kSupportBuild && gAggregated ? liveSlotOf(model.ops.topologyId) : -1;
    if (live >= 0) {
        const std::shared_ptr<const HierarchicalBuildInputs> inputs = buildInputs(modelIndex, categoryRatesIndex);
        if (!inputs)
            return BEAGLE_ERROR_OUT_OF_RANGE;
        for (int i = 0; i < count; i++) {
            const int index = probabilityIndices[i];
            if (index < 0 || index >= kMatrixCount || !std::isfinite(edgeLengths[i]) || edgeLengths[i] < 0.0)
                return BEAGLE_ERROR_OUT_OF_RANGE;
            hierarchicalBuildSupportBundle(model.ops, gSupport[live], kStateCount, inputs, kCategoryCount,
                                           edgeLengths[i], gPrZ0Tmp.data(), gSlots[index]);
        }
        return BEAGLE_SUCCESS;
    }
    double* piNode = gPiNodeTmp.data();
    hierarchicalNodeFrequencies(model, gFrequenciesDouble[0].data(), kStateCount, piNode);
    for (int m = kStateCount; m < 2 * kStateCount - 1; m++)
        if (!std::isfinite(piNode[m]) || piNode[m] <= 0.0)
            return BEAGLE_ERROR_OUT_OF_RANGE;
    const double* categoryRates = gCategoryRates[categoryRatesIndex];
    for (int i = 0; i < count; i++) {
        const int index = probabilityIndices[i];
        if (index < 0 || index >= kMatrixCount || !std::isfinite(edgeLengths[i]) || edgeLengths[i] < 0.0)
            return BEAGLE_ERROR_OUT_OF_RANGE;
        hierarchicalBuildBundle(model, piNode, categoryRates, kCategoryCount, edgeLengths[i],
                                gPrZ0Tmp.data(), kMayAggregate, gSlots[index]);
    }
    return BEAGLE_SUCCESS;
}

BEAGLE_CPU_TEMPLATE
void BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::completeSlot(HierarchicalSlot<REALTYPE>& slot) {
    if (slot.supportGeneration == 0)
        return;
    const std::shared_ptr<const HierarchicalBuildInputs> inputs = slot.inputs;
    hierarchicalBuildBundle(*slot.ops, inputs->lambda.data(), kStateCount, inputs->piNode.data(),
                            inputs->rates.data(), kCategoryCount, slot.edgeLength, gPrZ0Tmp.data(), kMayAggregate,
                            slot);
}

BEAGLE_CPU_TEMPLATE
void BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::completeSlots(const int* operations, int count, int numOps) {
    for (int op = 0; op < count; op++) {
        const int* operation = operations + op * numOps;
        const int last = operation[5] == BEAGLE_OP_NONE ? 4 : 6;
        for (int m = 4; m <= last; m += 2) {
            HierarchicalSlot<REALTYPE>& slot = gSlots[operation[m]];
            if (slot.supportGeneration == 0)
                continue;
            const int live = gAggregated ? liveSlotOf(slot.topologyId) : -1;
            if (live < 0 || gSupport[live].generation != slot.supportGeneration)
                completeSlot(slot);
        }
    }
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::setTransitionMatrix(int matrixIndex,
                                                                       const double* inMatrix,
                                                                       double paddedValue) {
    if (matrixIndex < 0 || matrixIndex >= kMatrixCount)
        return BEAGLE_ERROR_OUT_OF_RANGE;
    if ((int) gSlots.size() < kMatrixCount)
        gSlots.resize(kMatrixCount);
    int returnCode = unpackHierarchicalBundle(inMatrix, kStateCount, kCategoryCount, gSlots[matrixIndex]);
    if (returnCode == BEAGLE_SUCCESS && kMayAggregate) {
        gSlots[matrixIndex].owned->topologyId = topologyIdOf(*gSlots[matrixIndex].owned);
        gSlots[matrixIndex].topologyId = gSlots[matrixIndex].owned->topologyId;
    }
    if (returnCode == BEAGLE_SUCCESS && kSmallBand)
        hierarchicalDetectShape(*gSlots[matrixIndex].owned, kStateCount);
    else if (returnCode == BEAGLE_SUCCESS && kColumnGather)
        hierarchicalComputeMrca(*gSlots[matrixIndex].owned, kStateCount);
    return returnCode;
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
    completeSlot(gSlots[matrixIndex]);
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
    if (firstDerivativeIndices != NULL || secondDerivativeIndices != NULL)
        return BEAGLE_ERROR_NO_IMPLEMENTATION;
    return buildBundles(eigenIndex, 0, probabilityIndices, edgeLengths, count);
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
    if (firstDerivativeIndices != NULL || secondDerivativeIndices != NULL)
        return BEAGLE_ERROR_NO_IMPLEMENTATION;
    for (int i = 0; i < count; i++) {
        int returnCode = buildBundles(eigenIndices[i], categoryRateIndices[i], probabilityIndices + i,
                                      edgeLengths + i, 1);
        if (returnCode != BEAGLE_SUCCESS)
            return returnCode;
    }
    return BEAGLE_SUCCESS;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::updatePartials(const int* operations,
                                                                  int operationCount,
                                                                  int cumulativeScalingIndex,
                                                                  BeaglePartialsType partialsType) {
    if (!slotsReady(operations, operationCount, BEAGLE_OP_COUNT))
        return BEAGLE_ERROR_OUT_OF_RANGE;
    prepareClasses(operations, operationCount, BEAGLE_OP_COUNT);
    completeSlots(operations, operationCount, BEAGLE_OP_COUNT);
    if (!gAggregated)
        prepareTables(operations, operationCount, BEAGLE_OP_COUNT);
    preparePrZ0(operations, operationCount, BEAGLE_OP_COUNT);
    int returnCode = prepareRagged(operations, operationCount, BEAGLE_OP_COUNT);
    if (returnCode != BEAGLE_SUCCESS)
        return returnCode;
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
    prepareClasses(operations, operationCount, BEAGLE_PARTITION_OP_COUNT);
    completeSlots(operations, operationCount, BEAGLE_PARTITION_OP_COUNT);
    if (!gAggregated)
        prepareTables(operations, operationCount, BEAGLE_PARTITION_OP_COUNT);
    preparePrZ0(operations, operationCount, BEAGLE_PARTITION_OP_COUNT);
    int returnCode = prepareRagged(operations, operationCount, BEAGLE_PARTITION_OP_COUNT);
    if (returnCode != BEAGLE_SUCCESS)
        return returnCode;
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

    const int live = raggedReady(bufferIndex) ? liveSlotOf(gRaggedTopology[bufferIndex]) : -1;
    const double* piNode = live >= 0 ? nodeFrequencies(stateFrequenciesIndex, gLiveOps[live]) : NULL;
    if (piNode != NULL) {
        const REALTYPE* ragged = gRagged[bufferIndex];
        for (int k = startPattern; k < endPattern; k++) {
            const HierarchicalReduced<REALTYPE>& red = gClasses[gClassOfPattern[k]].reduced[live];
            const REALTYPE* p = ragged + gPatternOffset[k];
            REALTYPE sum = 0.0;
            for (int x = 0; x < red.stateCount; x++) {
                REALTYPE row = p[x * kCategoryCount] * wt[0];
                for (int l = 1; l < kCategoryCount; l++)
                    row += p[x * kCategoryCount + l] * wt[l];
                sum += (REALTYPE) piNode[red.localToFull[x]] * row;
            }
            site[k] = log(sum);
        }
    } else {
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
    }

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
            if (matrix < 0 || matrix >= (int) gSlots.size() || !gSlots[matrix].set || gSlots[matrix].ops == nullptr)
                return false;
        }
    }
    return true;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::smallPlan(bool states1,
                                                             const HierarchicalSlot<REALTYPE>& slot1,
                                                             bool singleChild,
                                                             bool states2,
                                                             const HierarchicalSlot<REALTYPE>* slot2) {
    const HierarchicalOps& ops1 = *slot1.ops;
    if (ops1.mrca.empty())
        return kHierarchicalSmallNone;
    if (singleChild) {
        if (states1)
            return kHierarchicalSmallColSingle;
        return ops1.shapeId >= 0 ? kHierarchicalSmallRegSingle : kHierarchicalSmallNone;
    }
    const HierarchicalOps& ops2 = *slot2->ops;
    if (ops2.mrca.empty())
        return kHierarchicalSmallNone;
    const bool dense = kKernel == kHierarchicalKernelDense;
    if (states1 && states2)
        return kHierarchicalSmallColCol;
    if (states2)
        return (!dense && ops1.shapeId >= 0) ? kHierarchicalSmallRegCol : kHierarchicalSmallDenseCol;
    if (states1)
        return (!dense && ops2.shapeId >= 0) ? kHierarchicalSmallColReg : kHierarchicalSmallColDense;
    if (!dense && ops1.shapeId >= 0 && ops1.shapeId == ops2.shapeId &&
        (&ops1 == &ops2 || ops1.tipOfSlot == ops2.tipOfSlot))
        return kHierarchicalSmallRegPair;
    return kHierarchicalSmallDensePair;
}

BEAGLE_CPU_TEMPLATE
bool BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::tableReady(const HierarchicalSlot<REALTYPE>& slot) {
    return slot.tableSet && slot.tableStamp == slot.stamp && slot.tableFrequencyStamp == gFrequencyStamp;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::topologyIdOf(const HierarchicalOps& ops) {
    for (size_t i = 0; i < gTopologyParents.size(); i++)
        if (gTopologyParents[i] == ops.parentOf)
            return gTopologyIds[i];
    if (gTopologyParents.size() == kHierarchicalTopologyCacheSize) {
        gTopologyParents.erase(gTopologyParents.begin());
        gTopologyIds.erase(gTopologyIds.begin());
    }
    gTopologyParents.push_back(ops.parentOf);
    gTopologyIds.push_back(gNextTopologyId);
    return gNextTopologyId++;
}

BEAGLE_CPU_TEMPLATE
const double* BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::nodeFrequencies(int frequencyIndex,
                                                                             const HierarchicalOps& ops) {
    if (frequencyIndex < 0 || frequencyIndex >= kEigenDecompCount || gFrequenciesDouble[frequencyIndex].empty())
        return NULL;
    std::vector<double>& piNode = gPiNodeByFrequency[frequencyIndex];
    if (ops.topologyId < 0 || piNode.empty() || gPiNodeTopology[frequencyIndex] != ops.topologyId ||
        gPiNodeStamp[frequencyIndex] != gFrequencyStamps[frequencyIndex]) {
        piNode.resize(2 * kStateCount - 1);
        hierarchicalNodeFrequencies(ops, gFrequenciesDouble[frequencyIndex].data(), kStateCount, piNode.data());
        gPiNodeTopology[frequencyIndex] = ops.topologyId;
        gPiNodeStamp[frequencyIndex] = gFrequencyStamps[frequencyIndex];
    }
    return piNode.data();
}

BEAGLE_CPU_TEMPLATE
bool BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::prZ0Ready(const HierarchicalSlot<REALTYPE>& slot) {
    return slot.prZ0Stamp == slot.stamp && (!slot.owned || slot.prZ0FrequencyStamp == gFrequencyStamp);
}

BEAGLE_CPU_TEMPLATE
void BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::preparePrZ0(const int* operations, int count, int numOps) {
    if (!kMayAggregate)
        return;
    for (int op = 0; op < count; op++) {
        const int* operation = operations + op * numOps;
        const int last = operation[5] == BEAGLE_OP_NONE ? 4 : 6;
        for (int m = 4; m <= last; m += 2) {
            HierarchicalSlot<REALTYPE>& slot = gSlots[operation[m]];
            if (prZ0Ready(slot))
                continue;
            hierarchicalRecoverInternalPrZ0(*slot.ops, kStateCount, kCategoryCount,
                                            nodeFrequencies(0, *slot.ops), gFrequencyStamp, slot);
        }
    }
}

BEAGLE_CPU_TEMPLATE
void BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::invalidateClasses(bool expand) {
    if (expand && gAggregated)
        expandRagged(-1);
    gClassesValid = false;
    gAggregateOn = false;
    gAggregated = false;
    gClasses.clear();
    gClassOfPattern.clear();
    gPatternCapacity.clear();
    gPatternOffset.clear();
    gRaggedSize = 0;
    std::fill(gLiveId.begin(), gLiveId.end(), -1);
    for (auto& support : gSupport)
        support.generation = 0;
    gClassGeneration++;
}

BEAGLE_CPU_TEMPLATE
void BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::observedSets(std::vector<int>& offsets,
                                                                 std::vector<int>& values) {
    const int S = kStateCount;
    offsets.assign(kPatternCount + 1, 0);
    values.clear();
    std::vector<char> mark(S, 0);
    for (int k = 0; k < kPatternCount; k++) {
        const size_t start = values.size();
        for (int t = 0; t < kTipCount; t++) {
            if (gTipStates[t] != NULL) {
                const int x = gTipStates[t][k];
                if (x < S && !mark[x]) {
                    mark[x] = 1;
                    values.push_back(x);
                }
                continue;
            }
            const REALTYPE* tip = gPartials[t];
            if (tip == NULL)
                continue;
            const int planes = (t < (int) gTipPlanesEqual.size() && gTipPlanesEqual[t]) ? 1 : kCategoryCount;
            for (int l = 0; l < planes; l++) {
                const REALTYPE first = tip[pIdx(l, k, 0)];
                bool constant = true, zero = false;
                for (int i = 0; i < S; i++) {
                    const REALTYPE v = tip[pIdx(l, k, i)];
                    constant = constant && v == first;
                    zero = zero || v == REALTYPE(0);
                }
                if (constant)
                    continue;
                for (int i = 0; i < S; i++) {
                    if (mark[i] || (zero && tip[pIdx(l, k, i)] == REALTYPE(0)))
                        continue;
                    mark[i] = 1;
                    values.push_back(i);
                }
            }
        }
        for (size_t i = start; i < values.size(); i++)
            mark[values[i]] = 0;
        std::sort(values.begin() + start, values.end());
        offsets[k + 1] = (int) values.size();
    }
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::liveSlotOf(int topologyId) const {
    if (topologyId < 0)
        return -1;
    for (int s = 0; s < kHierarchicalLiveTopologies; s++)
        if (gLiveId[s] == topologyId)
            return s;
    return -1;
}

BEAGLE_CPU_TEMPLATE
void BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::buildSupport(int live) {
    hierarchicalSupport(gLiveOps[live], kStateCount, gClasses, live, gSupport[live]);
    gSupport[live].generation = ++gSupportClock;
}

BEAGLE_CPU_TEMPLATE
void BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::refreshPiRed() {
    if (!gAggregateOn || gClasses.empty())
        return;
    for (int s = 0; s < kHierarchicalLiveTopologies; s++) {
        if (gLiveId[s] < 0 || gLiveFrequencyStamp[s] == gFrequencyStamp)
            continue;
        const double* piNode = nodeFrequencies(0, gLiveOps[s]);
        if (piNode == NULL)
            continue;
        for (auto& c : gClasses) {
            HierarchicalReduced<REALTYPE>& red = c.reduced[s];
            for (int x = 0; x < red.stateCount; x++)
                red.piRed[x] = (REALTYPE) piNode[red.localToFull[x]];
        }
        gLiveFrequencyStamp[s] = gFrequencyStamp;
    }
}

BEAGLE_CPU_TEMPLATE
REALTYPE* BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::ensureRagged(int bufferIndex) {
    if (gRaggedCapacity[bufferIndex] < gRaggedSize) {
        free(gRagged[bufferIndex]);
        gRagged[bufferIndex] = (REALTYPE*) mallocAligned(sizeof(REALTYPE) * gRaggedSize);
        gRaggedCapacity[bufferIndex] = gRagged[bufferIndex] == NULL ? 0 : gRaggedSize;
    }
    return gRagged[bufferIndex];
}

BEAGLE_CPU_TEMPLATE
REALTYPE* BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::ensureTipRagged(int live, int tipIndex) {
    std::vector<REALTYPE*>& tips = gTipRagged[live];
    std::vector<size_t>& capacity = gTipRaggedCapacity[live];
    if (tips.empty()) {
        tips.assign(kTipCount, nullptr);
        capacity.assign(kTipCount, 0);
    }
    if (capacity[tipIndex] < gRaggedSize) {
        free(tips[tipIndex]);
        tips[tipIndex] = (REALTYPE*) mallocAligned(sizeof(REALTYPE) * gRaggedSize);
        capacity[tipIndex] = tips[tipIndex] == NULL ? 0 : gRaggedSize;
    }
    return tips[tipIndex];
}

BEAGLE_CPU_TEMPLATE
bool BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::raggedReady(int bufferIndex) const {
    return gAggregated && bufferIndex >= kTipCount && bufferIndex < (int) gRagged.size() &&
           gRagged[bufferIndex] != NULL && gRaggedGeneration[bufferIndex] == gClassGeneration &&
           !gRaggedPre[bufferIndex] && liveSlotOf(gRaggedTopology[bufferIndex]) >= 0;
}

BEAGLE_CPU_TEMPLATE
void BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::compressRange(const REALTYPE* dense,
                                                                  REALTYPE* ragged,
                                                                  int live,
                                                                  int startPattern,
                                                                  int endPattern) {
    for (int k = startPattern; k < endPattern; k++) {
        const HierarchicalReduced<REALTYPE>& red = gClasses[gClassOfPattern[k]].reduced[live];
        REALTYPE* p = ragged + gPatternOffset[k];
        for (int x = 0; x < red.stateCount; x++)
            for (int l = 0; l < kCategoryCount; l++)
                p[x * kCategoryCount + l] = dense[pIdx(l, k, red.repOfRow[x])];
    }
}

BEAGLE_CPU_TEMPLATE
double BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::constancyDeviation(const REALTYPE* dense, int live) {
    double deviation = 0.0;
    std::vector<int> rowOf;
    int cached = -1;
    for (int k = 0; k < kPatternCount; k++) {
        const int c = gClassOfPattern[k];
        const HierarchicalReduced<REALTYPE>& red = gClasses[c].reduced[live];
        if (c != cached) {
            hierarchicalRowOfState(gLiveOps[live], kStateCount, red, rowOf);
            cached = c;
        }
        for (int l = 0; l < kCategoryCount; l++)
            for (int i = 0; i < kStateCount; i++) {
                const double v = dense[pIdx(l, k, i)];
                const double rep = dense[pIdx(l, k, red.repOfRow[rowOf[i]])];
                if (v != rep)
                    deviation = std::max(deviation, std::fabs(v - rep) / std::max(std::fabs(rep), 1e-300));
            }
    }
    return deviation;
}

BEAGLE_CPU_TEMPLATE
bool BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::ensureRaggedScratch() {
    if (gRaggedScratchSize >= gRaggedSize)
        return true;
    free(gRaggedW);
    free(gRaggedV);
    gRaggedW = (REALTYPE*) mallocAligned(sizeof(REALTYPE) * gRaggedSize);
    gRaggedV = (REALTYPE*) mallocAligned(sizeof(REALTYPE) * gRaggedSize);
    gRaggedScratchSize = (gRaggedW == NULL || gRaggedV == NULL) ? 0 : gRaggedSize;
    return gRaggedScratchSize != 0;
}

BEAGLE_CPU_TEMPLATE
bool BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::buildRaggedTips(int live) {
    for (int t = 0; t < kTipCount; t++) {
        if (gTipStates[t] != NULL || gPartials[t] == NULL)
            continue;
        REALTYPE* ragged = ensureTipRagged(live, t);
        if (ragged == NULL)
            return false;
        compressRange(gPartials[t], ragged, live, 0, kPatternCount);
    }
    return true;
}

BEAGLE_CPU_TEMPLATE
void BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::expandRagged(int topologyId) {
    std::vector<int> order(kPatternCount);
    for (int k = 0; k < kPatternCount; k++)
        order[k] = k;
    std::stable_sort(order.begin(), order.end(),
                     [this](int a, int b) { return gClassOfPattern[a] < gClassOfPattern[b]; });
    std::vector<int> rowOf;
    for (int s = 0; s < kHierarchicalLiveTopologies; s++) {
        if (gLiveId[s] < 0 || (topologyId >= 0 && gLiveId[s] != topologyId))
            continue;
        std::vector<int> buffers;
        for (int b = kTipCount; b < kBufferCount; b++) {
            if (gRagged[b] == NULL || gRaggedGeneration[b] != gClassGeneration || gRaggedTopology[b] != gLiveId[s])
                continue;
            if (gRaggedPre[b])
                gRaggedGeneration[b] = 0;
            else if (gPartials[b] != NULL)
                buffers.push_back(b);
        }
        if (buffers.empty())
            continue;
        int cached = -1;
        for (int k : order) {
            const int c = gClassOfPattern[k];
            if (c != cached) {
                hierarchicalRowOfState(gLiveOps[s], kStateCount, gClasses[c].reduced[s], rowOf);
                cached = c;
            }
            for (int b : buffers) {
                const REALTYPE* p = gRagged[b] + gPatternOffset[k];
                REALTYPE* dense = gPartials[b];
                for (int l = 0; l < kCategoryCount; l++)
                    for (int i = 0; i < kStateCount; i++)
                        dense[pIdx(l, k, i)] = p[rowOf[i] * kCategoryCount + l];
            }
        }
        for (int b : buffers)
            gRaggedGeneration[b] = 0;
    }
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::prepareRagged(const int* operations, int count, int numOps) {
    if (!gAggregated)
        return BEAGLE_SUCCESS;
    for (int op = 0; op < count; op++) {
        const int parIndex = operations[op * numOps];
        if (parIndex < kTipCount || parIndex >= kBufferCount)
            continue;
        if (ensureRagged(parIndex) == NULL)
            return BEAGLE_ERROR_OUT_OF_MEMORY;
        gRaggedGeneration[parIndex] = gClassGeneration;
        gRaggedPre[parIndex] = 0;
        gRaggedTopology[parIndex] = gOpTopology[op];
    }
    for (int op = 0; op < count; op++) {
        const int* operation = operations + op * numOps;
        const int topology = gOpTopology[op];
        for (int m = 3; m <= 5; m += 2) {
            const int child = operation[m];
            if (child < kTipCount || child >= kBufferCount || gPartials[child] == NULL ||
                (gRaggedGeneration[child] == gClassGeneration && !gRaggedPre[child] &&
                 gRaggedTopology[child] == topology))
                continue;
            REALTYPE* ragged = ensureRagged(child);
            if (ragged == NULL)
                return BEAGLE_ERROR_OUT_OF_MEMORY;
            compressRange(gPartials[child], ragged, liveSlotOf(topology), 0, kPatternCount);
            gRaggedGeneration[child] = gClassGeneration;
            gRaggedPre[child] = 0;
            gRaggedTopology[child] = topology;
        }
    }
    return BEAGLE_SUCCESS;
}

BEAGLE_CPU_TEMPLATE
void BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::opTopologies(const int* operations, int count, int numOps) {
    gOpTopology.resize(count);
    for (int op = 0; op < count; op++) {
        const int* operation = operations + op * numOps;
        const HierarchicalSlot<REALTYPE>& slot1 = gSlots[operation[4]];
        int topology = slot1.topologyId;
        if (topology < 0 || slot1.ops->topologyId != topology)
            topology = kHierarchicalMixedTopology;
        if (operation[5] != BEAGLE_OP_NONE) {
            const HierarchicalSlot<REALTYPE>& slot2 = gSlots[operation[6]];
            if (slot2.topologyId != topology || slot2.ops->topologyId != topology)
                topology = kHierarchicalMixedTopology;
        }
        gOpTopology[op] = topology;
    }
}

BEAGLE_CPU_TEMPLATE
bool BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::buildClasses(const HierarchicalOps& ops) {
    const auto start = std::chrono::steady_clock::now();
    invalidateClasses(true);
    std::vector<int> offsets, values;
    observedSets(offsets, values);
    hierarchicalClassify(offsets, values, kPatternCount, gClasses, gClassOfPattern);
    const double* piNode = nodeFrequencies(0, ops);
    for (auto& c : gClasses)
        hierarchicalBuildReduced(ops, kStateCount, piNode, c.observed, c.reduced[0]);
    long sumSc = 0;
    int maxSc = 0;
    for (int k = 0; k < kPatternCount; k++) {
        const int Sc = gClasses[gClassOfPattern[k]].reduced[0].stateCount;
        sumSc += Sc;
        maxSc = std::max(maxSc, Sc);
    }
    const double r = (double) sumSc / ((double) kPatternCount * kStateCount);
    const double threshold = hierarchicalAggregateThreshold(kCategoryCount);
    gAggregateOn = kAggregateMode == kHierarchicalAggregateOn || r < threshold;
    gClassesValid = true;
    gClassTipStamp = gTipStamp;
    const size_t classCount = gClasses.size();
    bool allocated = true;
    if (gAggregateOn) {
        gLiveId[0] = ops.topologyId;
        gLiveOps[0] = ops;
        gLiveFrequencyStamp[0] = gFrequencyStamp;
        gLiveUse[0] = ++gUseClock;
        gCurrentLive = 0;
        buildSupport(0);
        std::vector<int> added(1, 0);
        allocated = layoutRagged(added);
    } else {
        gClasses.clear();
        gClassOfPattern.clear();
    }
    if (kVerbose) {
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        fprintf(stderr, "hrate aggregation: %d patterns, %zu classes, mean S_c %.2f, max S_c %d, r %.4f, "
                "threshold %.3f, %s, build %.2f ms\n", kPatternCount, classCount,
                (double) sumSc / kPatternCount, maxSc, r, threshold,
                gAggregateOn ? (kAggregateMode == kHierarchicalAggregateOn ? "on (forced)" : "on") : "off", ms);
    }
    if (!allocated) {
        if (kVerbose)
            fprintf(stderr, "hrate aggregation: off, the ragged buffers could not be allocated\n");
        invalidateClasses(false);
        return false;
    }
    return true;
}

BEAGLE_CPU_TEMPLATE
bool BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::layoutRagged(std::vector<int>& added) {
    std::vector<int> need(kPatternCount, 1);
    size_t needSize = 0, haveSize = 0;
    bool grow = gPatternCapacity.size() != (size_t) kPatternCount;
    for (int k = 0; k < kPatternCount; k++) {
        const HierarchicalClass<REALTYPE>& c = gClasses[gClassOfPattern[k]];
        for (int s = 0; s < kHierarchicalLiveTopologies; s++)
            if (gLiveId[s] >= 0)
                need[k] = std::max(need[k], c.reduced[s].stateCount);
        needSize += (size_t) need[k];
        if (!grow) {
            haveSize += (size_t) gPatternCapacity[k];
            grow = need[k] > gPatternCapacity[k];
        }
    }
    const bool fresh = gPatternCapacity.size() != (size_t) kPatternCount;
    if (!grow && 2 * haveSize <= 3 * needSize) {
        for (int s : added)
            if (!buildRaggedTips(s))
                return false;
        return true;
    }
    std::vector<size_t> offset(kPatternCount + 1);
    size_t total = 0;
    for (int k = 0; k < kPatternCount; k++) {
        offset[k] = total;
        const size_t size = (size_t) kCategoryCount * need[k];
        total += (size + kHierarchicalRaggedAlign - 1) / kHierarchicalRaggedAlign * kHierarchicalRaggedAlign;
    }
    offset[kPatternCount] = total;
    if (!fresh) {
        std::vector<int> buffers;
        std::vector<REALTYPE*> moved;
        for (int b = kTipCount; b < kBufferCount; b++) {
            if (gRagged[b] == NULL || gRaggedGeneration[b] != gClassGeneration || gRaggedPre[b] ||
                liveSlotOf(gRaggedTopology[b]) < 0)
                continue;
            REALTYPE* to = (REALTYPE*) mallocAligned(sizeof(REALTYPE) * total);
            if (to == NULL) {
                for (REALTYPE* p : moved)
                    free(p);
                return false;
            }
            buffers.push_back(b);
            moved.push_back(to);
        }
        for (int b = kTipCount; b < kBufferCount; b++)
            if (gRaggedGeneration[b] == gClassGeneration && gRaggedPre[b])
                gRaggedGeneration[b] = 0;
        for (size_t j = 0; j < buffers.size(); j++) {
            const int b = buffers[j];
            const int s = liveSlotOf(gRaggedTopology[b]);
            for (int k = 0; k < kPatternCount; k++)
                std::memcpy(moved[j] + offset[k], gRagged[b] + gPatternOffset[k],
                            sizeof(REALTYPE) * kCategoryCount * gClasses[gClassOfPattern[k]].reduced[s].stateCount);
            free(gRagged[b]);
            gRagged[b] = moved[j];
            gRaggedCapacity[b] = total;
        }
    }
    gPatternCapacity = need;
    gPatternOffset = offset;
    gRaggedSize = total;
    if (!ensureRaggedScratch())
        return false;
    for (int s = 0; s < kHierarchicalLiveTopologies; s++)
        if (gLiveId[s] >= 0 && !buildRaggedTips(s))
            return false;
    return true;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::addLiveTopology(const HierarchicalOps& ops,
                                                                   const std::vector<int>& keep) {
    int live = -1;
    bool liveHeld = true;
    for (int s = 0; s < kHierarchicalLiveTopologies; s++) {
        if (gLiveId[s] < 0) {
            live = s;
            break;
        }
        if (std::find(keep.begin(), keep.end(), gLiveId[s]) != keep.end())
            continue;
        bool held = false;
        for (int b = kTipCount; b < kBufferCount && !held; b++)
            held = gRaggedTopology[b] == gLiveId[s] && gRaggedGeneration[b] == gClassGeneration &&
                   gRagged[b] != NULL && !gRaggedPre[b];
        if (live < 0 || (liveHeld && !held) || (liveHeld == held && gLiveUse[s] < gLiveUse[live])) {
            live = s;
            liveHeld = held;
        }
    }
    if (live < 0)
        return -1;
    if (gLiveId[live] >= 0) {
        if (kVerbose && liveHeld)
            fprintf(stderr, "hrate aggregation: character tree %d leaves the live set, its buffers are expanded\n",
                    gLiveId[live]);
        expandRagged(gLiveId[live]);
    }
    gLiveId[live] = ops.topologyId;
    gLiveOps[live] = ops;
    gLiveFrequencyStamp[live] = gFrequencyStamp;
    const double* piNode = nodeFrequencies(0, ops);
    for (auto& c : gClasses)
        hierarchicalBuildReduced(ops, kStateCount, piNode, c.observed, c.reduced[live]);
    buildSupport(live);
    return live;
}

BEAGLE_CPU_TEMPLATE
bool BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::aggregateCall(const int* operations,
                                                                  int count,
                                                                  int numOps,
                                                                  const char*& reason) {
    std::vector<int> ids;
    for (int op = 0; op < count; op++) {
        const int topology = gOpTopology[op];
        if (topology < 0) {
            reason = "an operation mixes character trees";
            return false;
        }
        if (std::find(ids.begin(), ids.end(), topology) == ids.end())
            ids.push_back(topology);
    }
    if ((int) ids.size() > kHierarchicalLiveTopologies) {
        reason = "the operations use more character trees than the live set holds";
        return false;
    }
    std::vector<int>& provenance = gProvenanceTmp;
    provenance = gBufferTopology;
    for (int op = 0; op < count; op++) {
        const int* operation = operations + op * numOps;
        const int topology = gOpTopology[op];
        for (int m = 3; m <= 5; m += 2) {
            const int child = operation[m];
            if (child < kTipCount || child >= kBufferCount)
                continue;
            if (provenance[child] != topology && provenance[child] != -1) {
                reason = "a child was computed under another character tree";
                return false;
            }
        }
        if (operation[0] >= kTipCount && operation[0] < kBufferCount)
            provenance[operation[0]] = topology;
    }
    std::vector<int> added;
    for (int op = 0; op < count; op++) {
        const int topology = gOpTopology[op];
        if (liveSlotOf(topology) >= 0)
            continue;
        const auto start = std::chrono::steady_clock::now();
        const int live = addLiveTopology(*gSlots[operations[op * numOps + 4]].ops, ids);
        if (live < 0) {
            reason = "no live slot is free";
            return false;
        }
        added.push_back(live);
        if (kVerbose) {
            long sumSc = 0;
            for (int k = 0; k < kPatternCount; k++)
                sumSc += gClasses[gClassOfPattern[k]].reduced[live].stateCount;
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            fprintf(stderr, "hrate aggregation: character tree %d in live slot %d, mean S_c %.2f, r %.4f, build %.2f ms\n",
                    topology, live, (double) sumSc / kPatternCount,
                    (double) sumSc / ((double) kPatternCount * kStateCount), ms);
        }
    }
    if (!added.empty() && !layoutRagged(added)) {
        reason = "the ragged buffers could not be allocated";
        invalidateClasses(true);
        return false;
    }
    for (int op = 0; op < count; op++)
        gLiveUse[liveSlotOf(gOpTopology[op])] = ++gUseClock;
    gCurrentLive = liveSlotOf(gOpTopology[count - 1]);
    return true;
}

BEAGLE_CPU_TEMPLATE
void BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::stampBuffers(const int* operations, int count, int numOps) {
    for (int op = 0; op < count; op++) {
        const int* operation = operations + op * numOps;
        int topology = gOpTopology[op];
        for (int m = 3; m <= 5; m += 2) {
            const int child = operation[m];
            if (child >= kTipCount && child < kBufferCount && gBufferTopology[child] != topology &&
                gBufferTopology[child] != -1)
                topology = kHierarchicalMixedTopology;
        }
        if (operation[0] >= kTipCount && operation[0] < kBufferCount)
            gBufferTopology[operation[0]] = topology;
    }
}

BEAGLE_CPU_TEMPLATE
void BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::prepareClasses(const int* operations, int count, int numOps) {
    if (!kMayAggregate || count == 0)
        return;
    opTopologies(operations, count, numOps);
    if (!gClassesValid || gClassTipStamp != gTipStamp) {
        int first = 0;
        while (first < count && gOpTopology[first] < 0)
            first++;
        if (first == count)
            invalidateClasses(true);
        else
            buildClasses(*gSlots[operations[first * numOps + 4]].ops);
    }
    if (gClassesValid && gAggregateOn) {
        const char* reason = NULL;
        const bool aggregate = aggregateCall(operations, count, numOps, reason);
        if (!aggregate && gAggregated) {
            if (kVerbose)
                fprintf(stderr, "hrate aggregation: dense for this call, %s\n", reason);
            expandRagged(-1);
        }
        gAggregated = aggregate;
        if (aggregate)
            refreshPiRed();
    }
    stampBuffers(operations, count, numOps);
}

BEAGLE_CPU_TEMPLATE
void BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::buildTable(HierarchicalSlot<REALTYPE>& slot) {
    if (tableReady(slot))
        return;
    if (!slot.tableSet) {
        const size_t bytes = (size_t) kCategoryCount * kStateCount * (kStateCount + 1) * sizeof(REALTYPE);
        if (!kSmallBand && gTableBytes + bytes > kTableBudget)
            return;
        gTableBytes += bytes;
    }
    hierarchicalColumnTable(slot, gStateFrequencies[0], kStateCount, kCategoryCount, gSumHTmp.data());
    slot.tableStamp = slot.stamp;
    slot.tableFrequencyStamp = gFrequencyStamp;
    slot.tableSet = true;
}

BEAGLE_CPU_TEMPLATE
void BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::prepareTables(const int* operations, int count, int numOps) {
    if (!kSmallBand && !kColumnGather)
        return;
    for (int op = 0; op < count; op++) {
        const int* operation = operations + op * numOps;
        const int child1 = operation[3];
        const int child2 = operation[5];
        const bool singleChild = child2 == BEAGLE_OP_NONE;
        HierarchicalSlot<REALTYPE>& slot1 = gSlots[operation[4]];
        HierarchicalSlot<REALTYPE>* slot2 = singleChild ? NULL : &gSlots[operation[6]];
        const bool states1 = gTipStates[child1] != NULL;
        const bool states2 = !singleChild && gTipStates[child2] != NULL;
        if (kColumnGather && !kSmallBand) {
            if (states1 && !slot1.ops->mrca.empty())
                buildTable(slot1);
            if (states2 && !slot2->ops->mrca.empty())
                buildTable(*slot2);
            continue;
        }
        const int plan = smallPlan(states1, slot1, singleChild, states2, slot2);
        const bool need1 = plan == kHierarchicalSmallColSingle || plan == kHierarchicalSmallColCol ||
                           plan == kHierarchicalSmallColReg || plan == kHierarchicalSmallDenseCol ||
                           plan == kHierarchicalSmallColDense || plan == kHierarchicalSmallDensePair;
        const bool need2 = plan == kHierarchicalSmallColCol || plan == kHierarchicalSmallRegCol ||
                           plan == kHierarchicalSmallDenseCol || plan == kHierarchicalSmallColDense ||
                           plan == kHierarchicalSmallDensePair;
        if (need1)
            buildTable(slot1);
        if (need2)
            buildTable(*slot2);
    }
}

BEAGLE_CPU_TEMPLATE
bool BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::fusedColumnPartials(REALTYPE* destP,
                                                                        const REALTYPE* partials1,
                                                                        const int* states1,
                                                                        const HierarchicalSlot<REALTYPE>& slot1,
                                                                        bool singleChild,
                                                                        const REALTYPE* partials2,
                                                                        const int* states2,
                                                                        const HierarchicalSlot<REALTYPE>* slot2,
                                                                        const REALTYPE* fixedScaleFactors,
                                                                        REALTYPE* scaleFactors,
                                                                        REALTYPE* cumulativeScaleFactors,
                                                                        int startPattern,
                                                                        int endPattern) {
    if (!kColumnGather)
        return false;
    const bool column1 = states1 != NULL && tableReady(slot1);
    const bool column2 = !singleChild && states2 != NULL && tableReady(*slot2);
    if (!column1 && !column2)
        return false;
    const int S = kStateCount;
    const size_t categoryTable = (size_t) S * (S + 1);
    const REALTYPE* pi = gStateFrequencies[0];
    const bool scaled = fixedScaleFactors != NULL;

    auto sweepColumn = [&](const HierarchicalSlot<REALTYPE>& slot, const REALTYPE* partials, const int* states,
                           const int* columnStates, const REALTYPE* table, int l, REALTYPE* dest, int k0, int k1) {
        if (states != NULL) {
            const HierarchicalStatesSource<REALTYPE> source = { states, S };
            if (scaled)
                hierarchicalGroupSweepColumn<true>(slot, S, l, pi, source, columnStates, table, dest,
                                                   kPaddedPatternCount, fixedScaleFactors, kGroupLanes, k0, k1);
            else
                hierarchicalGroupSweepColumn<false>(slot, S, l, pi, source, columnStates, table, dest,
                                                    kPaddedPatternCount, fixedScaleFactors, kGroupLanes, k0, k1);
        } else {
            const HierarchicalPartialsSource<REALTYPE> source =
                    { partials + pIdx(l, 0, 0), kPaddedPatternCount, kPartialsPaddedStateCount * kPaddedPatternCount };
            if (scaled)
                hierarchicalGroupSweepColumn<true>(slot, S, l, pi, source, columnStates, table, dest,
                                                   kPaddedPatternCount, fixedScaleFactors, kGroupLanes, k0, k1);
            else
                hierarchicalGroupSweepColumn<false>(slot, S, l, pi, source, columnStates, table, dest,
                                                    kPaddedPatternCount, fixedScaleFactors, kGroupLanes, k0, k1);
        }
    };

    for (int k0 = startPattern; k0 < endPattern; k0 += kTileSize) {
        const int k1 = std::min(k0 + kTileSize, endPattern);
        for (int l = 0; l < kCategoryCount; l++) {
            REALTYPE* dest = destP + pIdx(l, 0, 0);
            const REALTYPE* table1 = column1 ? slot1.columnTable.data() + l * categoryTable : NULL;
            const REALTYPE* table2 = column2 ? slot2->columnTable.data() + l * categoryTable : NULL;
            if (singleChild || (column1 && column2)) {
                if (scaled)
                    hierarchicalColumnRows<true>(S, states1, table1, column2 ? states2 : NULL, table2, dest,
                                                 kPaddedPatternCount, fixedScaleFactors, k0, k1);
                else
                    hierarchicalColumnRows<false>(S, states1, table1, column2 ? states2 : NULL, table2, dest,
                                                  kPaddedPatternCount, fixedScaleFactors, k0, k1);
            } else if (column2) {
                sweepColumn(slot1, partials1, states1, states2, table2, l, dest, k0, k1);
            } else {
                sweepColumn(*slot2, partials2, states2, states1, table1, l, dest, k0, k1);
            }
        }
        if (scaleFactors != NULL)
            rescaleRange(destP, scaleFactors, cumulativeScaleFactors, k0, k1);
    }
    return true;
}

BEAGLE_CPU_TEMPLATE
bool BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::smallPartials(REALTYPE* destP,
                                                                  const REALTYPE* partials1,
                                                                  const int* states1,
                                                                  const HierarchicalSlot<REALTYPE>& slot1,
                                                                  bool singleChild,
                                                                  const REALTYPE* partials2,
                                                                  const int* states2,
                                                                  const HierarchicalSlot<REALTYPE>* slot2,
                                                                  const REALTYPE* scaleFactors,
                                                                  int startPattern,
                                                                  int endPattern) {
    const int plan = smallPlan(states1 != NULL, slot1, singleChild, states2 != NULL, slot2);
    if (plan == kHierarchicalSmallNone)
        return false;
    const bool need1 = plan == kHierarchicalSmallColSingle || plan == kHierarchicalSmallColCol ||
                       plan == kHierarchicalSmallColReg || plan == kHierarchicalSmallDenseCol ||
                       plan == kHierarchicalSmallColDense || plan == kHierarchicalSmallDensePair;
    const bool need2 = plan == kHierarchicalSmallColCol || plan == kHierarchicalSmallRegCol ||
                       plan == kHierarchicalSmallDenseCol || plan == kHierarchicalSmallColDense ||
                       plan == kHierarchicalSmallDensePair;
    if ((need1 && !tableReady(slot1)) || (need2 && !tableReady(*slot2)))
        return false;
    switch (kStateCount) {
        case 2:
            return smallPartialsS<2>(plan, destP, partials1, states1, slot1, partials2, states2, slot2,
                                     scaleFactors, startPattern, endPattern);
        case 3:
            return smallPartialsS<3>(plan, destP, partials1, states1, slot1, partials2, states2, slot2,
                                     scaleFactors, startPattern, endPattern);
        case 4:
            return smallPartialsS<4>(plan, destP, partials1, states1, slot1, partials2, states2, slot2,
                                     scaleFactors, startPattern, endPattern);
        case 5:
            return smallPartialsS<5>(plan, destP, partials1, states1, slot1, partials2, states2, slot2,
                                     scaleFactors, startPattern, endPattern);
        case 6:
            return smallPartialsS<6>(plan, destP, partials1, states1, slot1, partials2, states2, slot2,
                                     scaleFactors, startPattern, endPattern);
        case 8:
            return smallPartialsS<8>(plan, destP, partials1, states1, slot1, partials2, states2, slot2,
                                     scaleFactors, startPattern, endPattern);
        default:
            return false;
    }
}

BEAGLE_CPU_TEMPLATE template <int S>
bool BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::smallPartialsS(int plan,
                                                                   REALTYPE* destP,
                                                                   const REALTYPE* partials1,
                                                                   const int* states1,
                                                                   const HierarchicalSlot<REALTYPE>& slot1,
                                                                   const REALTYPE* partials2,
                                                                   const int* states2,
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
    for (int l = 0; l < kCategoryCount; l++) {
        if (invScale != NULL)
            smallCategory<S, true>(plan, l, destP, partials1, states1, slot1, partials2, states2, slot2, invScale,
                                   startPattern, endPattern);
        else
            smallCategory<S, false>(plan, l, destP, partials1, states1, slot1, partials2, states2, slot2, invScale,
                                    startPattern, endPattern);
    }
    return true;
}

BEAGLE_CPU_TEMPLATE template <int S, bool Scaled>
void BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::smallCategory(int plan,
                                                                  int category,
                                                                  REALTYPE* destP,
                                                                  const REALTYPE* partials1,
                                                                  const int* states1,
                                                                  const HierarchicalSlot<REALTYPE>& slot1,
                                                                  const REALTYPE* partials2,
                                                                  const int* states2,
                                                                  const HierarchicalSlot<REALTYPE>* slot2,
                                                                  const REALTYPE* invScale,
                                                                  int startPattern,
                                                                  int endPattern) {
    typedef HierarchicalShapeDispatch<REALTYPE, S, Scaled> Dispatch;
    const int l = category;
    const int width = S + 1;
    const int blockLength = hierarchicalBlockLength(S);
    const REALTYPE* pi = gStateFrequencies[0];

    auto coef = [&](const HierarchicalSlot<REALTYPE>& slot, HierarchicalLaneCoef<REALTYPE, S>& c) {
        const HierarchicalOps& ops = *slot.ops;
        const REALTYPE* H = slot.values.data() + (size_t) l * blockLength;
        const REALTYPE* prZ0 = H + (S - 1);
        for (int i = 0; i < S; i++) {
            c.pi[i] = pi[ops.tipOfSlot[i]];
            c.z[i] = prZ0[ops.tipOfSlot[i]];
        }
        for (int i = 0; i < S - 1; i++)
            c.h[i] = H[ops.intOfSlot[i] - S];
    };
    auto slotRows = [&](const HierarchicalSlot<REALTYPE>& slot, const REALTYPE* partials, const REALTYPE** rows) {
        for (int i = 0; i < S; i++)
            rows[i] = partials + pIdx(l, 0, slot.ops->tipOfSlot[i]);
    };
    auto slotDest = [&](const HierarchicalSlot<REALTYPE>& slot, REALTYPE** rows) {
        for (int i = 0; i < S; i++)
            rows[i] = destP + pIdx(l, 0, slot.ops->tipOfSlot[i]);
    };
    auto stateRows = [&](const REALTYPE* partials, const REALTYPE** rows) {
        for (int x = 0; x < S; x++)
            rows[x] = partials + pIdx(l, 0, x);
    };
    auto table = [&](const HierarchicalSlot<REALTYPE>& slot) {
        return slot.columnTable.data() + (size_t) l * S * width;
    };
    auto permuted = [&](const HierarchicalSlot<REALTYPE>& order, const REALTYPE* T, REALTYPE* out) {
        for (int x = 0; x < S; x++)
            for (int y = 0; y < width; y++)
                out[x * width + y] = T[order.ops->tipOfSlot[x] * width + y];
    };

    HierarchicalLaneCoef<REALTYPE, S> c1, c2;
    const REALTYPE* L1[S];
    const REALTYPE* L2[S];
    REALTYPE* D[S];
    REALTYPE Tp[S * (S + 1)];
    switch (plan) {
        case kHierarchicalSmallRegSingle:
            coef(slot1, c1);
            slotRows(slot1, partials1, L1);
            slotDest(slot1, D);
            Dispatch::single(slot1.ops->shapeId, L1, c1, D, invScale, startPattern, endPattern);
            break;
        case kHierarchicalSmallColSingle:
            for (int x = 0; x < S; x++)
                D[x] = destP + pIdx(l, 0, x);
            hierarchicalColSingle<REALTYPE, S, Scaled>(states1, table(slot1), D, invScale, startPattern, endPattern);
            break;
        case kHierarchicalSmallRegPair:
            coef(slot1, c1);
            coef(*slot2, c2);
            slotRows(slot1, partials1, L1);
            slotRows(*slot2, partials2, L2);
            slotDest(slot1, D);
            Dispatch::pair(slot1.ops->shapeId, L1, c1, L2, c2, D, invScale, startPattern, endPattern);
            break;
        case kHierarchicalSmallDensePair:
            stateRows(partials1, L1);
            stateRows(partials2, L2);
            for (int x = 0; x < S; x++)
                D[x] = destP + pIdx(l, 0, x);
            hierarchicalDensePair<REALTYPE, S, Scaled>(L1, table(slot1), L2, table(*slot2), D, invScale,
                                                       startPattern, endPattern);
            break;
        case kHierarchicalSmallRegCol:
            coef(slot1, c1);
            slotRows(slot1, partials1, L1);
            slotDest(slot1, D);
            permuted(slot1, table(*slot2), Tp);
            Dispatch::column(slot1.ops->shapeId, L1, c1, states2, Tp, D, invScale, startPattern, endPattern);
            break;
        case kHierarchicalSmallColReg:
            coef(*slot2, c2);
            slotRows(*slot2, partials2, L2);
            slotDest(*slot2, D);
            permuted(*slot2, table(slot1), Tp);
            Dispatch::column(slot2->ops->shapeId, L2, c2, states1, Tp, D, invScale, startPattern, endPattern);
            break;
        case kHierarchicalSmallDenseCol:
            stateRows(partials1, L1);
            for (int x = 0; x < S; x++)
                D[x] = destP + pIdx(l, 0, x);
            hierarchicalDenseColPair<REALTYPE, S, Scaled>(L1, table(slot1), states2, table(*slot2), D, invScale,
                                                          startPattern, endPattern);
            break;
        case kHierarchicalSmallColDense:
            stateRows(partials2, L2);
            for (int x = 0; x < S; x++)
                D[x] = destP + pIdx(l, 0, x);
            hierarchicalDenseColPair<REALTYPE, S, Scaled>(L2, table(*slot2), states1, table(slot1), D, invScale,
                                                          startPattern, endPattern);
            break;
        case kHierarchicalSmallColCol:
            for (int x = 0; x < S; x++)
                D[x] = destP + pIdx(l, 0, x);
            hierarchicalColColPair<REALTYPE, S, Scaled>(states1, table(slot1), states2, table(*slot2), D, invScale,
                                                        startPattern, endPattern);
            break;
        default:
            break;
    }
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

BEAGLE_CPU_TEMPLATE template <typename Source1, typename Source2>
void BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::groupSweep(REALTYPE* destP,
                                                               const Source1& source1,
                                                               const HierarchicalSlot<REALTYPE>& slot1,
                                                               const Source2* source2,
                                                               const HierarchicalSlot<REALTYPE>* slot2,
                                                               const REALTYPE* fixedScaleFactors,
                                                               int category,
                                                               int k0,
                                                               int k1) {
    const REALTYPE* pi = gStateFrequencies[0];
    REALTYPE* dest = destP + pIdx(category, 0, 0);
    if (source2 == NULL) {
        if (fixedScaleFactors != NULL)
            hierarchicalGroupSweepSingle<true>(slot1, kStateCount, category, pi, source1.category(category), dest,
                                               kPaddedPatternCount, fixedScaleFactors, kGroupLanes, k0, k1);
        else
            hierarchicalGroupSweepSingle<false>(slot1, kStateCount, category, pi, source1.category(category), dest,
                                                kPaddedPatternCount, fixedScaleFactors, kGroupLanes, k0, k1);
    } else {
        if (fixedScaleFactors != NULL)
            hierarchicalGroupSweepPair<true>(slot1, *slot2, kStateCount, category, pi, source1.category(category),
                                             source2->category(category), dest, kPaddedPatternCount,
                                             fixedScaleFactors, kGroupLanes, k0, k1);
        else
            hierarchicalGroupSweepPair<false>(slot1, *slot2, kStateCount, category, pi, source1.category(category),
                                              source2->category(category), dest, kPaddedPatternCount,
                                              fixedScaleFactors, kGroupLanes, k0, k1);
    }
}

BEAGLE_CPU_TEMPLATE template <typename Source1, typename Source2>
void BeagleCPUHierarchicalImpl<BEAGLE_CPU_GENERIC>::hierarchicalPartialsFused(REALTYPE* destP,
                                                                              const Source1& source1,
                                                                              const HierarchicalSlot<REALTYPE>& slot1,
                                                                              const Source2* source2,
                                                                              const HierarchicalSlot<REALTYPE>* slot2,
                                                                              const REALTYPE* fixedScaleFactors,
                                                                              REALTYPE* scaleFactors,
                                                                              REALTYPE* cumulativeScaleFactors,
                                                                              int startPattern,
                                                                              int endPattern) {
    for (int k0 = startPattern; k0 < endPattern; k0 += kTileSize) {
        const int k1 = std::min(k0 + kTileSize, endPattern);
        if (kGroupOrder == kHierarchicalOrderCategory) {
            for (int l = 0; l < kCategoryCount; l++)
                groupSweep(destP, source1, slot1, source2, slot2, fixedScaleFactors, l, k0, k1);
        } else {
            for (int k = k0; k < k1; k += kGroupLanes) {
                const int kn = std::min(k + kGroupLanes, k1);
                for (int l = 0; l < kCategoryCount; l++)
                    groupSweep(destP, source1, slot1, source2, slot2, fixedScaleFactors, l, k, kn);
            }
        }
        if (scaleFactors != NULL)
            rescaleRange(destP, scaleFactors, cumulativeScaleFactors, k0, k1);
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

        bool rescaled = false;
        auto run = [&](const auto& source1, const auto* source2) {
            if (kFusedBand) {
                hierarchicalPartialsFused(destPartials, source1, slot1, source2, slot2, fixedScaleFactors,
                                          rescale == 1 ? scalingFactors : NULL,
                                          rescale == 1 ? cumulativeScaleBuffer : NULL, startPattern, endPattern);
                rescaled = rescale == 1;
            } else {
                hierarchicalPartials(destPartials, source1, slot1, source2, slot2, fixedScaleFactors, startPattern,
                                     endPattern);
            }
        };
        const HierarchicalStatesSource<REALTYPE>* noSource = NULL;

        if (gAggregated) {
            aggregatedPartials(parIndex, child1Index, tipStates1, slot1, singleChild, child2Index, tipStates2, slot2,
                               rescale, fixedScaleFactors, rescale == 1 ? scalingFactors : NULL,
                               rescale == 1 ? cumulativeScaleBuffer : NULL, startPattern, endPattern);
            rescaled = rescale == 1;
        } else if (kSmallBand && smallPartials(destPartials, partials1, tipStates1, slot1, singleChild, partials2, tipStates2,
                                        slot2, fixedScaleFactors, startPattern, endPattern)) {
        } else if (kColumnGather &&
                   fusedColumnPartials(destPartials, partials1, tipStates1, slot1, singleChild, partials2, tipStates2,
                                       slot2, fixedScaleFactors, rescale == 1 ? scalingFactors : NULL,
                                       rescale == 1 ? cumulativeScaleBuffer : NULL, startPattern, endPattern)) {
            rescaled = rescale == 1;
        } else if (tipStates1 != NULL) {
            const HierarchicalStatesSource<REALTYPE> source1 = { tipStates1, kStateCount };
            if (singleChild)
                run(source1, noSource);
            else if (tipStates2 != NULL)
                run(source1, &statesSource2);
            else
                run(source1, &partialsSource2);
        } else {
            const HierarchicalPartialsSource<REALTYPE> source1 =
                    { partials1, kPaddedPatternCount, kPartialsPaddedStateCount * kPaddedPatternCount };
            if (singleChild)
                run(source1, noSource);
            else if (tipStates2 != NULL)
                run(source1, &statesSource2);
            else
                run(source1, &partialsSource2);
        }

        if (rescale == 1 && !rescaled) {
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
