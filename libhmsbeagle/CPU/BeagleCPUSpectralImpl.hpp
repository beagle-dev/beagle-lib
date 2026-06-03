
/*
 *  BeagleCPUSpectralImpl.cpp
 *  BEAGLE
 *
 * Copyright 2026 Phylogenetic Likelihood Working Group
 *
 * This file is part of BEAGLE.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 *
 * @author Marc Suchard
 */

#ifndef BEAGLE_CPU_SPECTRAL_IMPL_HPP
#define BEAGLE_CPU_SPECTRAL_IMPL_HPP

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cstring>
#include <cmath>
#include <cassert>
#include <array>
#include <type_traits>

#include "libhmsbeagle/beagle.h"
#include "libhmsbeagle/CPU/BeagleCPUImpl.h"
#include "libhmsbeagle/CPU/BeagleCPUSpectralImpl.h"
#include "libhmsbeagle/CPU/EigenDecompositionSpectral.h"
#include "libhmsbeagle/CPU/AdjointMethods.h"

#define UNROLL_MV

namespace beagle {
namespace cpu {

BEAGLE_CPU_FACTORY_TEMPLATE
inline const char* getBeagleCPUSpectralName(){ return "CPU-Spectral-Unknown"; };

template<>
inline const char* getBeagleCPUSpectralName<double>(){ return "CPU-Spectral-Double"; };

template<>
inline const char* getBeagleCPUSpectralName<float>(){ return "CPU-Spectral-Single"; };

BEAGLE_CPU_TEMPLATE
int BeagleCPUSpectralImpl<BEAGLE_CPU_GENERIC>::createInstance(int tipCount,
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

    int returnCode = BeagleCPUImpl<BEAGLE_CPU_GENERIC>::createInstance(tipCount, partialsBufferCount, compactBufferCount,
                                                              stateCount, patternCount, eigenDecompositionCount,
                                                              matrixCount, categoryCount, scaleBufferCount,
                                                              resourceNumber, pluginResourceNumber,
                                                              preferenceFlags, requirementFlags);
    gBranchEigenInfo.resize(kMatrixCount);
    gPartialTmp1.resize(kStateCount);
    gPartialTmp2.resize(kStateCount);

#ifdef TEST_EB
    gTime.resize(kMatrixCount * kCategoryCount);

    const int eigenComputationLength = kMatrixCount * kCategoryCount * kPartialsPaddedStateCount;
    gExpAt.resize(eigenComputationLength);
    gCosBt.resize(eigenComputationLength);
    gSinBt.resize(eigenComputationLength);
    gExpAtCosBt.resize(eigenComputationLength);
    gExpAtSinBt.resize(eigenComputationLength);
#endif

    kStateCountModFour = (kStateCount / 4) * 4;

    return returnCode;
}

BEAGLE_CPU_TEMPLATE
BeagleCPUSpectralImpl<BEAGLE_CPU_GENERIC>::~BeagleCPUSpectralImpl() {
    // Do nothing
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUSpectralImpl<BEAGLE_CPU_GENERIC>::updateTransitionMatrices(int eigenIndex,
                                                                        const int* probabilityIndices,
                                                                        const int* firstDerivativeIndices,
                                                                        const int* secondDerivativeIndices,
                                                                        const double* edgeLengths,
                                                                        int count) {
    for (int i = 0; i < count; i++) {
        const int index = probabilityIndices[i];
        auto& info = gBranchEigenInfo[index];
        info.branchLength = edgeLengths[i];
        info.eigenIndex = eigenIndex;
        info.categoryRatesIndex = 0; // TODO: Implement category rates index

#ifdef TEST_EB
        info.time = &gTime[index * kCategoryCount];
        info.expat = &gExpAt[index * (kCategoryCount * kPartialsPaddedStateCount)];
        info.cosbt = &gCosBt[index * (kCategoryCount * kPartialsPaddedStateCount)];
        info.sinbt = &gSinBt[index * (kCategoryCount * kPartialsPaddedStateCount)];
        info.expatcosbt = &gExpAtCosBt[index * (kCategoryCount * kPartialsPaddedStateCount)];
        info.expatsinbt = &gExpAtSinBt[index * (kCategoryCount * kPartialsPaddedStateCount)];

        if (kCategoryCount > 1) {
            fprintf(stderr, "Multiple categories are not yet implemented");
            exit(-1);
        }

        const REALTYPE* eval = gEigenDecomposition->getEigenValuesPtr(eigenIndex);

        info.eval = eval;

        for (int j = 0; j < kCategoryCount; ++j) {
            const REALTYPE time = info.branchLength * gCategoryRates[info.categoryRatesIndex][j];
            info.time[j] = time;

            const int offset = j * kPartialsPaddedStateCount;

            for (int i = 0; i < kStateCount; ++i) {
                const REALTYPE e = std::exp(time * eval[i]);
                info.expat[offset + i] = e;

                const REALTYPE imag = eval[kStateCount + i];
                if (std::abs(imag) != REALTYPE(0)){
                    const REALTYPE c = std::cos(time * imag);
                    const REALTYPE s = std::sin(time * imag);
                    info.cosbt[offset + i]      = c;
                    info.sinbt[offset + i]      = s;
                    info.expatcosbt[offset + i] = e * c;
                    info.expatsinbt[offset + i] = e * s;
                    ++i;
                } else {
                    info.cosbt[offset + i] = REALTYPE(1);
                    info.sinbt[offset + i] = REALTYPE(0);
                }
            }
        }
#endif
    }

    return BEAGLE_SUCCESS;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUSpectralImpl<BEAGLE_CPU_GENERIC>::upPartials(bool byPartition,
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

        const REALTYPE* partials1 = gPartials[child1Index];
        const REALTYPE* partials2 = gPartials[child2Index];

        const int* tipStates1 = gTipStates[child1Index];
        const int* tipStates2 = gTipStates[child2Index];

        const int branchEigenIndex1 = child1TransMatIndex;
        const int branchEigenIndex2 = child2TransMatIndex;

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
                    calcStatesStates<WithScaling>(destPartials, tipStates1, branchEigenIndex1, tipStates2, branchEigenIndex2,
                                                  scalingFactors, startPattern, endPattern);
                } else {
                    // First compute without any scaling
                    calcStatesStates<NoScaling>(destPartials, tipStates1, branchEigenIndex1, tipStates2, branchEigenIndex2,
                                                nullptr, startPattern, endPattern);
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
                    calcStatesPartials<WithScaling>(destPartials, tipStates1, branchEigenIndex1, partials2, branchEigenIndex2,
                                                    scalingFactors, startPattern, endPattern);
                } else {
                    calcStatesPartials<NoScaling>(destPartials, tipStates1, branchEigenIndex1, partials2, branchEigenIndex2,
                                                  nullptr, startPattern, endPattern);
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
                    calcStatesPartials<WithScaling>(destPartials, tipStates2, branchEigenIndex2, partials1, branchEigenIndex1,
                                                    scalingFactors, startPattern, endPattern);
                } else {
                    calcStatesPartials<NoScaling>(destPartials, tipStates2, branchEigenIndex2, partials1, branchEigenIndex1,
                                                  nullptr, startPattern, endPattern);
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
                    // calcPartialsPartialsAutoScaling(destPartials,partials1,matrices1,partials2,matrices2,
                    //                                  &gActiveScalingFactors[sIndex]);
                    fprintf(stderr, "calcPartialsPartialsAutoScaling is not yet implemented for spectral representation\n");
                    exit(-1);
                    if (gActiveScalingFactors[sIndex]) {
                        autoRescalePartials(destPartials, gAutoScaleBuffers[sIndex]);
                    }

                } else if (rescale == 0) {
                    calcPartialsPartials<WithScaling>(destPartials, partials1, branchEigenIndex1, partials2, branchEigenIndex2,
                                                      scalingFactors, startPattern, endPattern, true);
                } else {
                    calcPartialsPartials<NoScaling>(destPartials, partials1, branchEigenIndex1, partials2, branchEigenIndex2,
                                                    nullptr, startPattern, endPattern, true);
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
int BeagleCPUSpectralImpl<BEAGLE_CPU_GENERIC>::upPrePartials(
        bool byPartition,
        const int* operations,
        int count,
        int cumulativeScaleIndex,
        BeaglePartialsType partialsType) {

    if (partialsType & BEAGLE_PARTIALS_TOP) {
        return upPrePartialsImpl<Top>(byPartition, operations, count, cumulativeScaleIndex);
    } else {
        return upPrePartialsImpl<Bottom>(byPartition, operations, count, cumulativeScaleIndex);
    }
}

BEAGLE_CPU_TEMPLATE template <typename T>
int BeagleCPUSpectralImpl<BEAGLE_CPU_GENERIC>::upPrePartialsImpl(
        bool byPartition,
        const int* operations,
        int count,
        int cumulativeScaleIndex) {

    for (int op = 0; op < count; op++) {

        int numOps = BEAGLE_OP_COUNT;
        if (byPartition) {
            numOps = BEAGLE_PARTITION_OP_COUNT;
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

        /// non-root nodes, can be a tip
        const REALTYPE *partials1 = gPartials[parentIndex];
        const REALTYPE *partials2 = gPartials[siblingIndex];

        const int *tipStates2 = gTipStates[siblingIndex];

        const int branchEigenIndex1 = parentTransMatIndex;
        const int branchEigenIndex2 = siblingTransMatIndex;

        REALTYPE *destPartials = gPartials[parIndex];

        int startPattern = 0;
        int endPattern = kPatternCount;
        if (byPartition) {
            startPattern = gPatternPartitionsStartPatterns[currentPartition];
            endPattern = gPatternPartitionsStartPatterns[currentPartition + 1];
        }

        if (tipStates2 != NULL) {
            if constexpr (std::is_same_v<T, Top>) {
                if (branchEigenIndex1 < 0) { // Parent node is root
                    calcPrePartialsStates<T, Root>(destPartials, partials1, branchEigenIndex1, tipStates2, branchEigenIndex2,
                                              startPattern, endPattern);
                } else {
                    calcPrePartialsStates<T, NotRoot>(destPartials, partials1, branchEigenIndex1, tipStates2, branchEigenIndex2,
                        startPattern, endPattern);
                }
            } else { // T == Bottom
                calcPrePartialsStates<T, NotUsed>(destPartials, partials1, branchEigenIndex1, tipStates2, branchEigenIndex2,
                    startPattern, endPattern);
            }
        } else {
            if constexpr (std::is_same_v<T, Top>) {
                if (branchEigenIndex1 < 0) { // Parent node is root
                    calcPrePartialsPartials<T, Root>(destPartials, partials1, branchEigenIndex1, partials2, branchEigenIndex2,
                        startPattern, endPattern);
                } else {
                    calcPrePartialsPartials<T, NotRoot>(destPartials, partials1, branchEigenIndex1, partials2, branchEigenIndex2,
                        startPattern, endPattern);
                }
            } else { // T == Bottom
                calcPrePartialsPartials<T, NotUsed>(destPartials, partials1, branchEigenIndex1, partials2, branchEigenIndex2,
                    startPattern, endPattern);
            }
        }
    }

    return BEAGLE_SUCCESS;
}

#define INIT(ifo,evr,evi,evec,ievc,bl,cr) \
    const BranchEigenInfo& ifo##_1 = gBranchEigenInfo[branchEigenIndex1]; \
    const BranchEigenInfo& ifo##_2 = gBranchEigenInfo[branchEigenIndex2]; \
    \
    const REALTYPE* evr##1 = gEigenDecomposition->getEigenValuesPtr(ifo##_1.eigenIndex); \
    const REALTYPE* evr##2 = gEigenDecomposition->getEigenValuesPtr(ifo##_2.eigenIndex); \
    \
    const REALTYPE* evi##1 = evr##1 + kStateCount; \
    const REALTYPE* evi##2 = evr##2 + kStateCount; \
    \
    const REALTYPE* evec##1 = gEigenDecomposition->getEigenVectorsPtr(ifo##_1.eigenIndex); \
    const REALTYPE* evec##2 = gEigenDecomposition->getEigenVectorsPtr(ifo##_2.eigenIndex); \
    \
    const REALTYPE* ievc##1 = gEigenDecomposition->getInverseEigenVectorsPtr(ifo##_1.eigenIndex); \
    const REALTYPE* ievc##2 = gEigenDecomposition->getInverseEigenVectorsPtr(ifo##_2.eigenIndex); \
    \
    const REALTYPE bl##1 = ifo##_1.branchLength; \
    const REALTYPE bl##2 = ifo##_2.branchLength; \
    \
    const double* cr##1 = gCategoryRates[ifo##_1.categoryRatesIndex]; \
    const double* cr##2 = gCategoryRates[ifo##_2.categoryRatesIndex];

 // TODO: do not init with eigenIndex == -1 \

#define INIT_TRANSPOSE1(ifo,evr,evi,evec,ievc,bl,cr) \
    const BranchEigenInfo& ifo##_1 = gBranchEigenInfo[branchEigenIndex1]; \
    const BranchEigenInfo& ifo##_2 = gBranchEigenInfo[branchEigenIndex2]; \
    \
    const REALTYPE* evr##1 = gEigenDecomposition->getBackwardsEigenValuesPtr(ifo##_1.eigenIndex); \
    const REALTYPE* evr##2 = gEigenDecomposition->getEigenValuesPtr(ifo##_2.eigenIndex); \
    \
    const REALTYPE* evi##1 = evr##1 + kStateCount; \
    const REALTYPE* evi##2 = evr##2 + kStateCount; \
    \
    const REALTYPE* evec##1 = gEigenDecomposition->getBackwardsEigenVectorsPtr(ifo##_1.eigenIndex); \
    const REALTYPE* evec##2 = gEigenDecomposition->getEigenVectorsPtr(ifo##_2.eigenIndex); \
    \
    const REALTYPE* ievc##1 = gEigenDecomposition->getBackwardsInverseEigenVectorsPtr(ifo##_1.eigenIndex); \
    const REALTYPE* ievc##2 = gEigenDecomposition->getInverseEigenVectorsPtr(ifo##_2.eigenIndex); \
    \
    const REALTYPE bl##1 = ifo##_1.branchLength; \
    const REALTYPE bl##2 = ifo##_2.branchLength; \
    \
    const double* cr##1 = gCategoryRates[ifo##_1.categoryRatesIndex]; \
    const double* cr##2 = gCategoryRates[ifo##_2.categoryRatesIndex];

// #define FORM_EXPAT(et, evr, sbl) \
//     const REALTYPE et##1 = std::exp(evr##1[i] * sbl##1); \
//     const REALTYPE et##2 = std::exp(evr##2[i] * sbl##2);

// #define FORM_EXPAT_COS_SIN(et, evi, sbl) \
//     int i2 = i + 1; \
//     const REALTYPE b1 = evi##1[i]; \
//     const REALTYPE et##cosbt1 = et##1 * std::cos(sbl##1 * b1); \
//     const REALTYPE et##sinbt1 = et##1 * std::sin(sbl##1 * b1); \
//     \
//     const REALTYPE b2 = evi##2[i]; \
//     const REALTYPE et##cosbt2 = et##2 * std::cos(sbl##2 * b2); \
//     const REALTYPE et##sinbt2 = et##2 * std::sin(sbl##2 * b2);

#define MATRIX_VECTOR(output, mat, vec) \
    for (int i = 0; i < kStateCount; i++) { \
        REALTYPE sum1 = 0.0, sum2 = 0.0; \
        for (int j = 0; j < kStateCount; j++) { \
            sum1 += mat##1[i * matrixIncr + j] * vec##1[j]; \
            sum2 += mat##2[i * matrixIncr + j] * vec##2[j]; \
        } \
        output; \
    }

#define MATRIX_VECTOR_SINGLE(output, mat, vec, sum) \
    for (int i = 0; i < kStateCount; i++) { \
        REALTYPE sum = 0.0; \
        for (int j = 0; j < kStateCount; j++) { \
            sum += mat[i * matrixIncr + j] * vec[j]; \
        } \
        output; \
    }

#define MATRIX_VECTOR_HADAMARD_PRODUCT(out, mat, vec) \
    MATRIX_VECTOR(out[i] = sum1 * sum2, mat, vec)

#define MATRIX_VECTOR_HADAMARD_PRODUCT_SCALE(out, mat, vec, scale) \
    MATRIX_VECTOR(out[i] = sum1 * sum2 * scale, mat, vec)

#ifdef TEST_EB
BEAGLE_CPU_TEMPLATE template <typename First, typename Second, typename Direction>
void BeagleCPUSpectralImpl<BEAGLE_CPU_GENERIC>::expScaledMatrixVectorMultiple2(
        REALTYPE* out1, REALTYPE* out2,
        const REALTYPE* partials1, const int state1,
            const BranchEigenInfo& info1,
            const REALTYPE* matrix1,
        const REALTYPE* partials2, const int state2,
            const BranchEigenInfo& info2,
            const REALTYPE* matrix2,
            const int matrixIncr) {

    if constexpr (
        !(std::is_same_v<First,  None> || std::is_same_v<First,  States> || std::is_same_v<First,  Partials>) ||
        !(std::is_same_v<Second, None> || std::is_same_v<Second, States> || std::is_same_v<Second, Partials>)
        ) {
        static_assert(always_false<First>::value, "Unsupported type T");
    }

    for (int i = 0; i < kStateCount; i++) {

        REALTYPE expat1;
        REALTYPE expat2;
        if constexpr (!std::is_same_v<First, None>) {
            expat1 = info1.expat[i]; // std::exp(real1[i] * time1);
        }
        if constexpr (!std::is_same_v<Second, None>) {
            expat2 = info2.expat[i]; // std::exp(real2[i] * time2);
        }

        if ((std::is_same_v<First,  None> || info1.sinbt[i] == 0.0) &&
            (std::is_same_v<Second, None> || info2.sinbt[i] == 0.0)) {
            // All real
            const int row_i = i * matrixIncr;
            REALTYPE sum1a, sum1b, sum2a, sum2b;
            if constexpr (std::is_same_v<First, States>) {
                sum1a = matrix1[row_i + state1];
                sum1b = REALTYPE(0);
            } else if constexpr (std::is_same_v<First, Partials>) {
                sum1a = REALTYPE(0);
                sum1b = REALTYPE(0);
            }

            if constexpr (std::is_same_v<Second, States>) {
                sum2a = matrix2[row_i + state2];
                sum2b = REALTYPE(0);
            } else if constexpr (std::is_same_v<Second, Partials>) {
                sum2a = REALTYPE(0);
                sum2b = REALTYPE(0);
            }

            int j = 0;
            for (; j < kStateCountModFour; j += 4) {
                if constexpr (std::is_same_v<First, Partials>) {
                    sum1a += matrix1[row_i + j + 0] * partials1[j + 0];
                    sum1b += matrix1[row_i + j + 1] * partials1[j + 1];
                    sum1a += matrix1[row_i + j + 2] * partials1[j + 2];
                    sum1b += matrix1[row_i + j + 3] * partials1[j + 3];
                }
                if constexpr (std::is_same_v<Second, Partials>) {
                    sum2a += matrix2[row_i + j + 0] * partials2[j + 0];
                    sum2b += matrix2[row_i + j + 1] * partials2[j + 1];
                    sum2a += matrix2[row_i + j + 2] * partials2[j + 2];
                    sum2b += matrix2[row_i + j + 3] * partials2[j + 3];
                }
            }
            for (; j < kStateCount; j++) {
                if constexpr (std::is_same_v<First, Partials>) {
                    sum1a += matrix1[row_i + j] * partials1[j];
                }
                if constexpr (std::is_same_v<Second, Partials>) {
                    sum2a += matrix2[row_i + j] * partials2[j];
                }
            }

            if constexpr (!std::is_same_v<First, None>) {
                out1[i] = expat1 * (sum1a + sum1b);
            }
            if constexpr (!std::is_same_v<Second, None>) {
                out2[i] = expat2 * (sum2a + sum2b);
            }

        } else {
            // At least one complex conjugate pair
            //
            // 2x2 conjugate block
            // If A is 2x2 with complex conjugate pair eigenvalues a +/- bi, then
            // exp(At) = exp(at)*( cos(bt)I + \frac{sin(bt)}{b}(A - aI)).
            int i2 = i + 1;
            REALTYPE expatcosbt1, expatcosbt2;
            REALTYPE expatsinbt1, expatsinbt2;

            if constexpr (!std::is_same_v<First, None>) {
                if constexpr (std::is_same_v<Direction, Backward>) {
                    expatcosbt1 = info1.expatcosbt[i]; // expat1 * std::cos(time1 * imag1[i]);
                    expatsinbt1 = -info1.expatsinbt[i]; // expat1 * std::sin(time1 * imag1[i]);
                } else {
                    expatcosbt1 = info1.expatcosbt[i]; // expat1 * std::cos(time1 * imag1[i]);
                    expatsinbt1 = info1.expatsinbt[i]; // expat1 * std::sin(time1 * imag1[i]);
                }
            }

            if constexpr (!std::is_same_v<Second, None>) {
                expatcosbt2 = info2.expatcosbt[i]; // expat2 * std::cos(time2 * imag2[i]);
                expatsinbt2 = info2.expatsinbt[i]; // expat2 * std::sin(time2 * imag2[i]);
            }

            const int row_i  = i  * matrixIncr;
            const int row_i2 = i2 * matrixIncr;
            REALTYPE s1Aa, s1Ab, s1Ba, s1Bb;
            REALTYPE s2Aa, s2Ab, s2Ba, s2Bb;
            if constexpr (std::is_same_v<First, States>) {
                s1Aa = expatcosbt1 * matrix1[row_i  + state1] +
                           expatsinbt1 * matrix1[row_i2 + state1];
                s1Ab = REALTYPE(0);
                s1Ba = expatcosbt1 * matrix1[row_i2 + state1] -
                           expatsinbt1 * matrix1[row_i  + state1];
                s1Bb = REALTYPE(0);
            } else if constexpr (std::is_same_v<First, Partials>) {
                s1Aa = REALTYPE(0); s1Ab = REALTYPE(0);
                s1Ba = REALTYPE(0); s1Bb = REALTYPE(0);
            }

            if constexpr (std::is_same_v<Second, States>) {
                s2Aa = expatcosbt2 * matrix2[row_i  + state2] +
                           expatsinbt2 * matrix2[row_i2 + state2];
                s2Ab = REALTYPE(0);
                s2Ba = expatcosbt2 * matrix2[row_i2 + state2] -
                           expatsinbt2 * matrix2[row_i  + state2];
                s2Bb = REALTYPE(0);
            } else if constexpr (std::is_same_v<Second, Partials>) {
                s2Aa = REALTYPE(0); s2Ab = REALTYPE(0);
                s2Ba = REALTYPE(0); s2Bb = REALTYPE(0);
            }

            int j = 0;
            for (; j < kStateCountModFour; j += 4) {
                if constexpr (std::is_same_v<First, Partials>) {
                    s1Aa += (expatcosbt1 * matrix1[row_i  + j+0] + expatsinbt1 * matrix1[row_i2 + j+0]) * partials1[j+0];
                    s1Ab += (expatcosbt1 * matrix1[row_i  + j+1] + expatsinbt1 * matrix1[row_i2 + j+1]) * partials1[j+1];
                    s1Aa += (expatcosbt1 * matrix1[row_i  + j+2] + expatsinbt1 * matrix1[row_i2 + j+2]) * partials1[j+2];
                    s1Ab += (expatcosbt1 * matrix1[row_i  + j+3] + expatsinbt1 * matrix1[row_i2 + j+3]) * partials1[j+3];

                    s1Ba += (expatcosbt1 * matrix1[row_i2 + j+0] - expatsinbt1 * matrix1[row_i  + j+0]) * partials1[j+0];
                    s1Bb += (expatcosbt1 * matrix1[row_i2 + j+1] - expatsinbt1 * matrix1[row_i  + j+1]) * partials1[j+1];
                    s1Ba += (expatcosbt1 * matrix1[row_i2 + j+2] - expatsinbt1 * matrix1[row_i  + j+2]) * partials1[j+2];
                    s1Bb += (expatcosbt1 * matrix1[row_i2 + j+3] - expatsinbt1 * matrix1[row_i  + j+3]) * partials1[j+3];
                }

                if constexpr (std::is_same_v<Second, Partials>) {
                    s2Aa += (expatcosbt2 * matrix2[row_i  + j+0] + expatsinbt2 * matrix2[row_i2 + j+0]) * partials2[j+0];
                    s2Ab += (expatcosbt2 * matrix2[row_i  + j+1] + expatsinbt2 * matrix2[row_i2 + j+1]) * partials2[j+1];
                    s2Aa += (expatcosbt2 * matrix2[row_i  + j+2] + expatsinbt2 * matrix2[row_i2 + j+2]) * partials2[j+2];
                    s2Ab += (expatcosbt2 * matrix2[row_i  + j+3] + expatsinbt2 * matrix2[row_i2 + j+3]) * partials2[j+3];

                    s2Ba += (expatcosbt2 * matrix2[row_i2 + j+0] - expatsinbt2 * matrix2[row_i  + j+0]) * partials2[j+0];
                    s2Bb += (expatcosbt2 * matrix2[row_i2 + j+1] - expatsinbt2 * matrix2[row_i  + j+1]) * partials2[j+1];
                    s2Ba += (expatcosbt2 * matrix2[row_i2 + j+2] - expatsinbt2 * matrix2[row_i  + j+2]) * partials2[j+2];
                    s2Bb += (expatcosbt2 * matrix2[row_i2 + j+3] - expatsinbt2 * matrix2[row_i  + j+3]) * partials2[j+3];
                }
            }
            for (; j < kStateCount; j++) {
                if constexpr (std::is_same_v<First, Partials>) {
                    s1Aa += (expatcosbt1 * matrix1[row_i  + j] + expatsinbt1 * matrix1[row_i2 + j]) * partials1[j];
                    s1Ba += (expatcosbt1 * matrix1[row_i2 + j] - expatsinbt1 * matrix1[row_i  + j]) * partials1[j];
                }

                if constexpr (std::is_same_v<Second, Partials>) {
                    s2Aa += (expatcosbt2 * matrix2[row_i  + j] + expatsinbt2 * matrix2[row_i2 + j]) * partials2[j];
                    s2Ba += (expatcosbt2 * matrix2[row_i2 + j] - expatsinbt2 * matrix2[row_i  + j]) * partials2[j];
                }
            }

            if constexpr (!std::is_same_v<First, None>) {
                out1[i]  = s1Aa + s1Ab;
                out1[i2] = s1Ba + s1Bb;
            }

            if constexpr (!std::is_same_v<Second, None>) {
                out2[i]  = s2Aa + s2Ab;
                out2[i2] = s2Ba + s2Bb;
            }

            i++; // processed two conjugate rows
        }
    }
}
#endif

BEAGLE_CPU_TEMPLATE template <typename First, typename Second>
void BeagleCPUSpectralImpl<BEAGLE_CPU_GENERIC>::expScaledMatrixVectorMultiple(
        REALTYPE* out1, REALTYPE* out2,
        const REALTYPE* partials1, const int state1,
            const REALTYPE* real1,
            const REALTYPE* imag1,
            const REALTYPE* matrix1,
            const REALTYPE time1,
        const REALTYPE* partials2, const int state2,
            const REALTYPE* real2,
            const REALTYPE* imag2,
            const REALTYPE* matrix2,
            const REALTYPE time2,
        // REALTYPE* temp1, REALTYPE* temp2,
            const int matrixIncr) {

    if constexpr (
        !(std::is_same_v<First,  None> || std::is_same_v<First,  States> || std::is_same_v<First,  Partials>) ||
        !(std::is_same_v<Second, None> || std::is_same_v<Second, States> || std::is_same_v<Second, Partials>)
        ) {
        static_assert(always_false<First>::value, "Unsupported type T");
    }

    for (int i = 0; i < kStateCount; i++) {

        REALTYPE expat1;
        REALTYPE expat2;
        if constexpr (!std::is_same_v<First, None>) {
            expat1 = std::exp(real1[i] * time1);
        }
        if constexpr (!std::is_same_v<Second, None>) {
            expat2 = std::exp(real2[i] * time2);
        }

        if ((std::is_same_v<First,  None> || imag1[i] == 0.0) &&
            (std::is_same_v<Second, None> || imag2[i] == 0.0)) {
            // All real
            REALTYPE sum1, sum2;
            if constexpr (std::is_same_v<First, States>) {
                sum1 = matrix1[i * matrixIncr + state1];
            } else if constexpr (std::is_same_v<First, Partials>) {
                sum1 = 0.0;
            }

            if constexpr (std::is_same_v<Second, States>) {
                sum2 = matrix2[i * matrixIncr + state2];
            } else if constexpr (std::is_same_v<Second, Partials>) {
                sum2 = 0.0;
            }

            for (int j = 0; j < kStateCount; j++) {
                if constexpr (std::is_same_v<First, Partials>) {
                    sum1 += matrix1[i * matrixIncr + j] * partials1[j];
                }
                if constexpr (std::is_same_v<Second, Partials>) {
                    sum2 += matrix2[i * matrixIncr + j] * partials2[j];
                }
            }

            if constexpr (!std::is_same_v<First, None>) {
                out1[i] = expat1 * sum1;
            }
            if constexpr (!std::is_same_v<Second, None>) {
                out2[i] = expat2 * sum2;
            }

        } else {
            // At least one complex conjugate pair
            //
            // 2x2 conjugate block
            // If A is 2x2 with complex conjugate pair eigenvalues a +/- bi, then
            // exp(At) = exp(at)*( cos(bt)I + \frac{sin(bt)}{b}(A - aI)).
            int i2 = i + 1;
            REALTYPE expatcosbt1, expatcosbt2;
            REALTYPE expatsinbt1, expatsinbt2;

            if constexpr (!std::is_same_v<First, None>) {
                expatcosbt1 = expat1 * std::cos(time1 * imag1[i]);
                expatsinbt1 = expat1 * std::sin(time1 * imag1[i]);
            }

            if constexpr (!std::is_same_v<Second, None>) {
                expatcosbt2 = expat2 * std::cos(time2 * imag2[i]);
                expatsinbt2 = expat2 * std::sin(time2 * imag2[i]);
            }

            REALTYPE sum1A, sum1B, sum2A, sum2B;
            if constexpr (std::is_same_v<First, States>) {
                sum1A = expatcosbt1 * matrix1[i * matrixIncr + state1] +
                            expatsinbt1 * matrix1[i2 * matrixIncr + state1];

                sum1B = expatcosbt1 * matrix1[i2 * matrixIncr + state1] -
                            expatsinbt1 * matrix1[i * matrixIncr + state1];
            } else if constexpr (std::is_same_v<First, Partials>) {
                sum1A = 0.0; sum1B = 0.0;
            }

            if constexpr (std::is_same_v<Second, States>) {
                sum2A = expatcosbt2 * matrix2[i * matrixIncr + state2] +
                            expatsinbt2 * matrix2[i2 * matrixIncr + state2];

                sum2B = expatcosbt2 * matrix2[i2 * matrixIncr + state2] -
                            expatsinbt2 * matrix2[i * matrixIncr + state2];
            } else if constexpr (std::is_same_v<Second, Partials>) {
                sum2A = 0.0; sum2B = 0.0;
            }

            for (int j = 0; j < kStateCount; j++) {
                if constexpr (std::is_same_v<First, Partials>) {
                    sum1A += (expatcosbt1 * matrix1[i * matrixIncr + j] +
                                expatsinbt1 * matrix1[i2 * matrixIncr + j]) * partials1[j];

                    sum1B += (expatcosbt1 * matrix1[i2 * matrixIncr + j] -
                                expatsinbt1 * matrix1[i * matrixIncr + j]) * partials1[j];
                }

                if constexpr (std::is_same_v<Second, Partials>) {
                    sum2A += (expatcosbt2 * matrix2[i * matrixIncr + j] +
                                expatsinbt2 * matrix2[i2 * matrixIncr + j]) * partials2[j];

                    sum2B += (expatcosbt2 * matrix2[i2 * matrixIncr + j] -
                                expatsinbt2 * matrix2[i * matrixIncr + j]) * partials2[j];
                }
            }

            if constexpr (!std::is_same_v<First, None>) {
                out1[i] = sum1A;
                out1[i2] = sum1B;
            }

            if constexpr (!std::is_same_v<Second, None>) {
                out2[i] = sum2A;
                out2[i2] = sum2B;
            }

            i++; // processed two conjugate rows
        }
    }
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUSpectralImpl<BEAGLE_CPU_GENERIC>::calculateAdjointCrossProducts(
        const int *postBufferIndices,
        const int *preBufferIndices,
        const int *branchEigenIndices,
        const int *categoryRatesIndices,
        const int *categoryWeightsIndices,
        const int rootPostOrderIndex,
        const int stateFrequenciesIndex,
        int count,
        double *outSumDerivatives,
        double *outSumSquaredDerivatives) {

    int returnCode = BEAGLE_SUCCESS;

    const int secondDerivativeIndex = BEAGLE_OP_NONE;
    const double *categoryRates = gCategoryRates[categoryRatesIndices[0]]; // TODO Generalize
    const REALTYPE *categoryWeights = gCategoryWeights[categoryWeightsIndices[0]]; // TODO Generalize

    REALTYPE* buffer;
    std::vector<REALTYPE> realTypeBuffer;
    if constexpr (!std::is_same_v<REALTYPE, double>) {
        realTypeBuffer.resize(kStateCount * kStateCount * kCategoryCount);
        buffer = realTypeBuffer.data();
    } else {
        buffer = outSumDerivatives;
    }

    for (int nodeNum = 0; nodeNum < count; nodeNum++) {

        const int branchEigenIndex = branchEigenIndices[nodeNum];
        const double edgeLength = gBranchEigenInfo[branchEigenIndex].branchLength;

        const REALTYPE *preOrderPartial = gPartials[preBufferIndices[nodeNum]];

        const int *tipStates = gTipStates[postBufferIndices[nodeNum]];
        const REALTYPE *postOrderPartial = (tipStates == nullptr) ? gPartials[postBufferIndices[nodeNum]] : nullptr;

        const int patternOffset = nodeNum * kPatternCount;

        if (tipStates != nullptr) {

            calcAdjointCrossProducts<States,WithRotation>(
                postOrderPartial, tipStates, preOrderPartial,
                branchEigenIndex,
                categoryRates, categoryWeights,
                edgeLength,
                buffer, nullptr);

        } else {

            calcAdjointCrossProducts<Partials,WithRotation>(
                postOrderPartial, tipStates, preOrderPartial,
                branchEigenIndex,
                categoryRates, categoryWeights,
                edgeLength,
                buffer, nullptr);
        }
    }

    if constexpr (!std::is_same_v<REALTYPE, double>) {
        beagleMemCpy(outSumDerivatives, buffer, kStateCount * kStateCount * kCategoryCount);
    }

    return returnCode;
}

template <typename REALTYPE>
void printVector(const char* prompt, const int index, const REALTYPE* vec, const int dim) {
    fprintf(stderr, "%s", prompt);
    fprintf(stderr, "%d:", index);
    for (int i = 0; i < dim; ++i) {
        fprintf(stderr, " %.5e", vec[i]);
    }
    fprintf(stderr, "\n");
}


template <typename REALTYPE>
inline REALTYPE branchLikelihoodInEigenBasis(
        const REALTYPE* lhs,
        const REALTYPE* rhs,
        const REALTYPE* evalR,
        const REALTYPE* evalI,
        const REALTYPE time,
        const int kStateCount) {

    // fprintf(stderr, "beagle-time = %f\n", time);

    // printVector("eval-real ", 0, evalR, kStateCount);
    // printVector("eval-imag ", 0, evalI, kStateCount);

    REALTYPE sum = REALTYPE(0);
    for (int i = 0; i < kStateCount; ++i) {
        const REALTYPE real = evalR[i];
        const REALTYPE imag = evalI[i];
        const REALTYPE expReal = std::exp(time * real); // TODO cache;

        if (imag == 0.0) {
            sum += lhs[i] * expReal * rhs[i];
        } else {

            const REALTYPE c = expReal * std::cos(time * imag); // TODO cache
            const REALTYPE s = expReal * std::sin(time * imag); // TODO cache
            const REALTYPE x = rhs[i];
            const REALTYPE y = rhs[i + 1];

            sum += lhs[i] * (c * x + s * y) + lhs[i + 1] * (c * y - s * x);

            ++i;
        }
    }

    return sum;
}

BEAGLE_CPU_TEMPLATE template <typename T, typename V>
void BeagleCPUSpectralImpl<BEAGLE_CPU_GENERIC>::calcAdjointCrossProducts(
        const REALTYPE* inPartialsPost, const int* tipsPost,
        const REALTYPE* inPartialsPre,
        const int branchEigenIndex,
        const double* categoryRates,
        const REALTYPE* categoryWeights,
        const double edgeLength,
        REALTYPE* first,
        REALTYPE* second) {

    // TODO follow calcCrossProductsPartials()

    // for (int pattern = 0; pattern < kPatternCount; pattern++) {

    //     std::vector<REALTYPE> tmp(kStateCount * kStateCount, 0.0);

    //     REALTYPE patternDenominator = 0.0;

    //     for (int category = 0; category < kCategoryCount; category++) {

    //         const REALTYPE scale = (REALTYPE) categoryRates[category] * edgeLength;
    //         const REALTYPE weight = categoryWeights[category];

    //         const int patternIndex = category * kPatternCount + pattern;
    //         const int v = patternIndex * kPartialsPaddedStateCount;

    //         REALTYPE denominator = 0.0;

    //         // compute denominator for this pattern and category

    //         patternDenominator += denominator * weight;

    //         // accumulate cross products for this pattern and category into tmp[k * kStateCount + j]

    //     }

    //     const auto patternWeight = gPatternWeights[pattern] / patternDenominator;

    //     // accumulate into outCrossProducts[k * kStateCount + j] from tmp[k * kStateCount + j] with pattern
    // }

    const int startPattern = 0;
    const int endPattern = kPatternCount;

    const int matrixIncr = kStateCount + T_PAD;
    const int stateCountModFour = (kStateCount / 4) * 4;

    const BranchEigenInfo& info = gBranchEigenInfo[branchEigenIndex];

    // fprintf(stderr, "matrix = %d @ %f\n", branchEigenIndex, info.branchLength);

    const REALTYPE* evalR = gEigenDecomposition->getEigenValuesPtr(info.eigenIndex);
    const REALTYPE* evalI = evalR + kStateCount;

    const REALTYPE* evec = gEigenDecomposition->getEigenVectorsPtr(info.eigenIndex);
    const REALTYPE* ievc = gEigenDecomposition->getInverseEigenVectorsPtr(info.eigenIndex);

    const REALTYPE* tevalR = gEigenDecomposition->getBackwardsEigenValuesPtr(info.eigenIndex);
    const REALTYPE* tevalI = tevalR + kStateCount;

    const REALTYPE* tevec = gEigenDecomposition->getBackwardsEigenVectorsPtr(info.eigenIndex);
    const REALTYPE* tievc = gEigenDecomposition->getBackwardsInverseEigenVectorsPtr(info.eigenIndex);

    const REALTYPE time = info.branchLength;
    const double* rates = gCategoryRates[info.categoryRatesIndex];
    const REALTYPE* weights = gCategoryWeights[info.categoryRatesIndex];

    AdjointMethods<REALTYPE>* adj = gEigenDecomposition->getAdjointMethodsPtr(info.eigenIndex);

    if (kCategoryCount > 1) {
        fprintf(stderr, "kCategoryCount > 1 not yet implemented\n");
    }

#if defined(_OPENMP)
    #pragma omp parallel for num_threads(kCategoryCount)
#endif
    for (int l = 0; l < kCategoryCount; l++) {

        int v = l * kPartialsPaddedStateCount * kPatternCount + kPartialsPaddedStateCount * startPattern;

        const REALTYPE scaledTime = static_cast<REALTYPE>(rates[l]) * time;
#ifdef TEST_EB
        adj->setTime(scaledTime,
            info.expat,
            info.cosbt, info.sinbt,
            info.expatcosbt, info.expatsinbt);
#else
        adj->setTime(scaledTime);
#endif

        const REALTYPE* preOrderPartials = &inPartialsPre[v];
        const REALTYPE* postOrderPartials = &inPartialsPost[v];

        for (int k = startPattern; k < endPattern; k++) {

            REALTYPE *lhs, *rhs;

            if constexpr (std::is_same_v<V, WithRotation>) {

                lhs = gPartialTmp1.data();
                rhs = gPartialTmp2.data();

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

                // printVector("post-order-rotated ", 0, rhs, kStateCount);
                // printVector("pre-order-rotated  ", 0, lhs, kStateCount);
            } else {
                static_assert(always_false<V>::value, "Unsupported lack of rotation");
            }

            const REALTYPE denominator = adj->branchLikelihoodInEigenBasis(lhs, rhs);

            const REALTYPE scale = gPatternWeights[k] * weights[l] / denominator;
            // fprintf(stderr, "likelihood = %.5e for %f with scale %f\n", denominator, scaledTime, scale);

            adj->accumulateEigenBasisGradient(1, lhs, rhs, scale, first, kStateCount);

            // printVector("beagle-acc ", 0, first, kStateCount * kStateCount);

            preOrderPartials  += kPartialsPaddedStateCount;
            postOrderPartials += kPartialsPaddedStateCount;
        }
    }
}

#ifdef UNROLL_MV
BEAGLE_CPU_TEMPLATE template <typename First, typename Second, typename Body>
void BeagleCPUSpectralImpl<BEAGLE_CPU_GENERIC>::matVecDual(
        const REALTYPE* mat1, const REALTYPE* vec1, const int state1,
        const REALTYPE* mat2, const REALTYPE* vec2, const int state2,
        const int matrixIncr,
        Body body) {                       // deduced as a concrete lambda type → zero overhead

    if constexpr (
        !(std::is_same_v<First,  None> || std::is_same_v<First,  States> || std::is_same_v<First,  Partials>) ||
        !(std::is_same_v<Second, None> || std::is_same_v<Second, States> || std::is_same_v<Second, Partials>)
        ) {
        static_assert(always_false<First>::value, "Unsupported type T");
    }

    for (int i = 0; i < kStateCount; i++) {
        REALTYPE sum1A, sum1B, sum2A, sum2B;
        if constexpr (std::is_same_v<First, States>) {
            sum1A = mat1[i * matrixIncr + state1];
            sum1B = REALTYPE(0);
        } else if constexpr (std::is_same_v<First, Partials>) {
            sum1A = REALTYPE(0);
            sum1B = REALTYPE(0);
        }

        if constexpr (std::is_same_v<Second, States>) {
            sum2A = mat2[i * matrixIncr + state2];
            sum2B = REALTYPE(0);
        } else if constexpr (std::is_same_v<Second, Partials>) {
            sum2A = REALTYPE(0);
            sum2B = REALTYPE(0);
        }

        int j = 0;
        for ( ; j < kStateCountModFour; j += 4) {
            if constexpr (std::is_same_v<First, Partials>) {
                sum1A += mat1[i * matrixIncr + j + 0] * vec1[j + 0];
                sum1B += mat1[i * matrixIncr + j + 1] * vec1[j + 1];

                sum1A += mat1[i * matrixIncr + j + 2] * vec1[j + 2];
                sum1B += mat1[i * matrixIncr + j + 3] * vec1[j + 3];
            }
            if constexpr (std::is_same_v<Second, Partials>) {
                sum2A += mat2[i * matrixIncr + j + 0] * vec2[j + 0];
                sum2B += mat2[i * matrixIncr + j + 1] * vec2[j + 1];

                sum2A += mat2[i * matrixIncr + j + 2] * vec2[j + 2];
                sum2B += mat2[i * matrixIncr + j + 3] * vec2[j + 3];
            }
        }

        for ( ; j < kStateCount; j++) {
            if constexpr (std::is_same_v<First, Partials>) {
                sum1A += mat1[i * matrixIncr + j] * vec1[j];
            }
            if constexpr (std::is_same_v<Second, Partials>) {
                sum2A += mat2[i * matrixIncr + j] * vec2[j];
            }
        }

        body(i, sum1A + sum1B, sum2A + sum2B);
    }
}
#else
BEAGLE_CPU_TEMPLATE template <typename First, typename Second, typename Body>
void BeagleCPUSpectralImpl<BEAGLE_CPU_GENERIC>::matVecDual(
        const REALTYPE* mat1, const REALTYPE* vec1, const int state1,
        const REALTYPE* mat2, const REALTYPE* vec2, const int state2,
        const int N, const int matrixIncr,
        Body body) {                       // deduced as a concrete lambda type → zero overhead

    if constexpr (
        !(std::is_same_v<First,  None> || std::is_same_v<First,  States> || std::is_same_v<First,  Partials>) ||
        !(std::is_same_v<Second, None> || std::is_same_v<Second, States> || std::is_same_v<Second, Partials>)
        ) {
        static_assert(always_false<First>::value, "Unsupported type T");
    }

    for (int i = 0; i < N; i++) {
        REALTYPE sum1, sum2;
        if constexpr (std::is_same_v<First, States>) {
            sum1 = mat1[i * matrixIncr + state1];
        } else if constexpr (std::is_same_v<First, Partials>) {
            sum1 = REALTYPE(0);
        }

        if constexpr (std::is_same_v<Second, States>) {
            sum2 = mat2[i * matrixIncr + state2];
        } else if constexpr (std::is_same_v<Second, Partials>) {
            sum2 = REALTYPE(0);
        }

        for (int j = 0; j < N; j++) {
            if constexpr (std::is_same_v<First, Partials>) {
                sum1 += mat1[i * matrixIncr + j] * vec1[j];
            }
            if constexpr (std::is_same_v<Second, Partials>) {
                sum2 += mat2[i * matrixIncr + j] * vec2[j];
            }
        }

        body(i, sum1, sum2);
    }
}
#endif

BEAGLE_CPU_TEMPLATE template <typename T>
void BeagleCPUSpectralImpl<BEAGLE_CPU_GENERIC>::calcPartialsPartials(
        REALTYPE *destPartials,
        const REALTYPE *partials1,
        const int branchEigenIndex1,
        const REALTYPE *partials2,
        const int branchEigenIndex2,
        const REALTYPE *scaleFactors,
        int startPattern,
        int endPattern,
        bool isComplex) {

    // TODO: we can optimize for info1.eigenIndex == info2.eigenIndex

    INIT(info,
         eigenValuesReal, eigenValuesImag,
         eigenVectors, inverseEigenVectors,
         branchLength, categoryRate);

    const int matrixIncr = kStateCount + T_PAD;
    const int stateCountModFour = (kStateCount / 4) * 4;

#if defined(_OPENMP)
    #pragma omp parallel for num_threads(kCategoryCount)
#endif
    for (int l = 0; l < kCategoryCount; l++) {
        int v = l * kPartialsPaddedStateCount * kPatternCount + kPartialsPaddedStateCount * startPattern;

        const REALTYPE scaledBranchLength1 = static_cast<REALTYPE>(categoryRate1[l]) * branchLength1;
        const REALTYPE scaledBranchLength2 = static_cast<REALTYPE>(categoryRate2[l]) * branchLength2;

        const REALTYPE* partials1Ptr = &partials1[v];
        const REALTYPE* partials2Ptr = &partials2[v];
        REALTYPE* destPtr = &destPartials[v];

        for (int k = startPattern; k < endPattern; k++) {
#ifdef TEST_EB
           expScaledMatrixVectorMultiple2<Partials,Partials,Forward>(
                gPartialTmp1.data(), gPartialTmp2.data(),
                partials1Ptr, 0, info_1, inverseEigenVectors1,
                partials2Ptr, 0, info_2, inverseEigenVectors2,
                matrixIncr);
#else
            expScaledMatrixVectorMultiple<Partials,Partials>(
                gPartialTmp1.data(), gPartialTmp2.data(),
                partials1Ptr, 0, eigenValuesReal1, eigenValuesImag1,
                inverseEigenVectors1, scaledBranchLength1,
                partials2Ptr, 0, eigenValuesReal2, eigenValuesImag2,
                inverseEigenVectors2, scaledBranchLength2,
                matrixIncr);
#endif

            if constexpr (T::useScaleFactors) {
                const REALTYPE oneOverScaleFactor = REALTYPE(1.0) / scaleFactors[k];
                MATRIX_VECTOR_HADAMARD_PRODUCT_SCALE(destPtr, eigenVectors, gPartialTmp, oneOverScaleFactor);
            } else {
                // MATRIX_VECTOR_HADAMARD_PRODUCT(destPtr, eigenVectors, gPartialTmp);
                matVecDual<Partials, Partials>(
                    eigenVectors1, gPartialTmp1.data(), 0,
                    eigenVectors2, gPartialTmp2.data(), 0,
                    matrixIncr,
                    [&](int i, REALTYPE sum1, REALTYPE sum2) {
                        destPtr[i] = sum1 * sum2;
                    });
            }

            destPtr += kPartialsPaddedStateCount;
            partials1Ptr += kPartialsPaddedStateCount;
            partials2Ptr += kPartialsPaddedStateCount;
        }
    }
}

BEAGLE_CPU_TEMPLATE template <typename T>
void BeagleCPUSpectralImpl<BEAGLE_CPU_GENERIC>::calcStatesPartials(
        REALTYPE* destPartials,
        const int* states1,
        const int branchEigenIndex1,
        const REALTYPE* partials2,
        const int branchEigenIndex2,
        const REALTYPE* scaleFactors,
        int startPattern,
        int endPattern) {

    INIT(info,
         eigenValuesReal, eigenValuesImag,
         eigenVectors, inverseEigenVectors,
         branchLength, categoryRate);

    const int matrixIncr = kStateCount + T_PAD;
    const int stateCountModFour = (kStateCount / 4) * 4;

#if defined(_OPENMP)
#pragma omp parallel for num_threads(kCategoryCount)
#endif
    for (int l = 0; l < kCategoryCount; l++) {
        int v = l * kPartialsPaddedStateCount * kPatternCount + kPartialsPaddedStateCount * startPattern;

        const REALTYPE scaledBranchLength1 = static_cast<REALTYPE>(categoryRate1[l]) * branchLength1;
        const REALTYPE scaledBranchLength2 = static_cast<REALTYPE>(categoryRate2[l]) * branchLength2;

        const REALTYPE* partials2Ptr = &partials2[v];
        REALTYPE* destPtr = &destPartials[v];

        for (int k = startPattern; k < endPattern; k++) {

            const int state1 = states1[k];

#ifdef TEST_EB
           expScaledMatrixVectorMultiple2<States,Partials,Forward>(
                gPartialTmp1.data(), gPartialTmp2.data(),
                nullptr, state1, info_1, inverseEigenVectors1,
                partials2Ptr, 0, info_2, inverseEigenVectors2,
                matrixIncr);
#else
            expScaledMatrixVectorMultiple<States,Partials>(
                gPartialTmp1.data(), gPartialTmp2.data(),
                nullptr, state1, eigenValuesReal1, eigenValuesImag1,
                inverseEigenVectors1, scaledBranchLength1,
                partials2Ptr, 0, eigenValuesReal2, eigenValuesImag2,
                inverseEigenVectors2, scaledBranchLength2,
                matrixIncr);
#endif

            if constexpr (T::useScaleFactors) {
                const REALTYPE oneOverScaleFactor = REALTYPE(1.0) / scaleFactors[k];
                MATRIX_VECTOR_HADAMARD_PRODUCT_SCALE(destPtr, eigenVectors, gPartialTmp, oneOverScaleFactor);
            } else {
                // MATRIX_VECTOR_HADAMARD_PRODUCT(destPtr, eigenVectors, gPartialTmp);
                matVecDual<Partials, Partials>(
                    eigenVectors1, gPartialTmp1.data(), 0,
                    eigenVectors2, gPartialTmp2.data(), 0,
                    matrixIncr,
                    [&](int i, REALTYPE sum1, REALTYPE sum2) {
                        destPtr[i] = sum1 * sum2;
                    });
            }

            destPtr += kPartialsPaddedStateCount;
            partials2Ptr += kPartialsPaddedStateCount;
        }
    }
}


BEAGLE_CPU_TEMPLATE template <typename T>
void BeagleCPUSpectralImpl<BEAGLE_CPU_GENERIC>::calcStatesStates(
    REALTYPE* destPartials,
    const int* states1,
    const int branchEigenIndex1,
    const int* states2,
    const int branchEigenIndex2,
    const REALTYPE* scaleFactors,
    int startPattern,
    int endPattern) {

    INIT(info,
         eigenValuesReal, eigenValuesImag,
         eigenVectors, inverseEigenVectors,
         branchLength, categoryRate);

    const int matrixIncr = kStateCount + T_PAD;
    const int stateCountModFour = (kStateCount / 4) * 4;

#if defined(_OPENMP)
#pragma omp parallel for num_threads(kCategoryCount)
#endif
    for (int l = 0; l < kCategoryCount; l++) {
        int v = l * kPartialsPaddedStateCount * kPatternCount + kPartialsPaddedStateCount * startPattern;

        const REALTYPE scaledBranchLength1 = static_cast<REALTYPE>(categoryRate1[l]) * branchLength1;
        const REALTYPE scaledBranchLength2 = static_cast<REALTYPE>(categoryRate2[l]) * branchLength2;

        REALTYPE* destPtr = &destPartials[v];

        for (int k = startPattern; k < endPattern; k++) {

            const int state1 = states1[k];
            const int state2 = states2[k];

#ifdef TEST_EB
             expScaledMatrixVectorMultiple2<States,States,Forward>(
                gPartialTmp1.data(), gPartialTmp2.data(),
                nullptr, state1, info_1, inverseEigenVectors1,
                nullptr, state2, info_2, inverseEigenVectors2,
                matrixIncr);
#else
            expScaledMatrixVectorMultiple<States,States>(
                gPartialTmp1.data(), gPartialTmp2.data(),
                nullptr, state1, eigenValuesReal1, eigenValuesImag1,
                inverseEigenVectors1, scaledBranchLength1,
                nullptr, state2, eigenValuesReal2, eigenValuesImag2,
                inverseEigenVectors2, scaledBranchLength2,
                matrixIncr);
#endif

            if constexpr (T::useScaleFactors) {
                const REALTYPE oneOverScaleFactor = REALTYPE(1.0) / scaleFactors[k];
                MATRIX_VECTOR_HADAMARD_PRODUCT_SCALE(destPtr, eigenVectors, gPartialTmp, oneOverScaleFactor);
            } else {
                // MATRIX_VECTOR_HADAMARD_PRODUCT(destPtr, eigenVectors, gPartialTmp);
                matVecDual<Partials, Partials>(
                    eigenVectors1, gPartialTmp1.data(), 0,
                    eigenVectors2, gPartialTmp2.data(), 0,
                    matrixIncr,
                    [&](int i, REALTYPE sum1, REALTYPE sum2) {
                        destPtr[i] = sum1 * sum2;
                    });
            }

            destPtr += kPartialsPaddedStateCount;
        }
    }
}

BEAGLE_CPU_TEMPLATE template <typename T, typename V>
void BeagleCPUSpectralImpl<BEAGLE_CPU_GENERIC>::calcPrePartialsPartials(
            REALTYPE *destPartials,
            const REALTYPE *partials1,
            const int branchEigenIndex1,
            const REALTYPE *partials2,
            const int branchEigenIndex2,
            int startPattern,
            int endPattern) {

    INIT_TRANSPOSE1(info,
        eigenValuesReal, eigenValuesImag,
        eigenVectors, inverseEigenVectors,
        branchLength, categoryRate);

    const int matrixIncr = kStateCount + T_PAD;
    const int stateCountModFour = (kStateCount / 4) * 4;

#if defined(_OPENMP)
    #pragma omp parallel for num_threads(kCategoryCount)
#endif
    for (int l = 0; l < kCategoryCount; l++) {
        int v = l * kPartialsPaddedStateCount * kPatternCount + kPartialsPaddedStateCount * startPattern;

        const REALTYPE scaledBranchLength1 = static_cast<REALTYPE>(categoryRate1[l]) * branchLength1;
        const REALTYPE scaledBranchLength2 = static_cast<REALTYPE>(categoryRate2[l]) * branchLength2;

        const REALTYPE* partials1Ptr = &partials1[v];
        const REALTYPE* partials2Ptr = &partials2[v];
        REALTYPE* destPtr = &destPartials[v];

        REALTYPE* intermediate = gPartialTmp2.data();
        REALTYPE* parent = gPartialTmp1.data();

        for (int k = startPattern; k < endPattern; k++) {

            if constexpr (std::is_same_v<T, Bottom>) {

                // first step
#ifdef TEST_EB
               expScaledMatrixVectorMultiple2<Partials,None,Forward>(
                    intermediate, nullptr,
                    partials2Ptr, 0, info_2, inverseEigenVectors2,
                    nullptr, 0, info_1, nullptr,
                    matrixIncr);
#else
                expScaledMatrixVectorMultiple<Partials,None>(
                    intermediate, nullptr,
                    partials2Ptr, 0, eigenValuesReal2, eigenValuesImag2,
                    inverseEigenVectors2, scaledBranchLength2,
                    nullptr, 0, nullptr, nullptr,
                    nullptr, 0.0,
                    matrixIncr);
#endif

                MATRIX_VECTOR_SINGLE(parent[i] = sum2 * partials1Ptr[i], // fused Hadamard product
                    eigenVectors2, intermediate, sum2)

                // second step
#ifdef TEST_EB
                expScaledMatrixVectorMultiple2<Partials,None,Backward>(
                    intermediate, nullptr,
                    parent, 0, info_1, inverseEigenVectors1,
                    nullptr, 0, info_2, nullptr,
                    matrixIncr);
#else
                expScaledMatrixVectorMultiple<Partials,None>(
                    intermediate, nullptr,
                    parent, 0, eigenValuesReal1, eigenValuesImag1,
                    inverseEigenVectors1, scaledBranchLength1,
                    nullptr, 0, nullptr, nullptr,
                    nullptr, 0.0,
                    matrixIncr);
#endif

                MATRIX_VECTOR_SINGLE(destPtr[i] = sum1, eigenVectors1, intermediate, sum1);

            } else if constexpr (std::is_same_v<T, Top>) {

                if constexpr (std::is_same_v<V, Root>) {

#ifdef TEST_EB
                    expScaledMatrixVectorMultiple2<Partials,None,Forward>(
                        intermediate, nullptr,
                        partials2Ptr, 0, info_2, inverseEigenVectors2,
                        nullptr, 0, info_1, nullptr,
                        matrixIncr);
#else
                    expScaledMatrixVectorMultiple<Partials,None>(
                        intermediate, nullptr,
                        partials2Ptr, 0, eigenValuesReal2, eigenValuesImag2,
                        inverseEigenVectors2, scaledBranchLength2,
                        nullptr, 0, nullptr, nullptr,
                        nullptr, 0.0,
                        matrixIncr);
#endif

                    MATRIX_VECTOR_SINGLE(destPtr[i] = sum2 * partials1Ptr[i], // fused Hadamard product
                        eigenVectors2, intermediate, sum2)

                } else {

#ifdef TEST_EB

                //   printVector("", 0, eigenValuesReal1, 2 * kStateCount);
                //   printVector("", 1, info_1.eval, 2 * kStateCount);
                  expScaledMatrixVectorMultiple2<Partials,Partials,Backward>(
                        gPartialTmp1.data(), gPartialTmp2.data(),
                        partials1Ptr, 0, info_1, inverseEigenVectors1,
                        partials2Ptr, 0, info_2, inverseEigenVectors2,
                        matrixIncr);
#else
                    expScaledMatrixVectorMultiple<Partials,Partials>(
                        gPartialTmp1.data(), gPartialTmp2.data(),
                        partials1Ptr, 0, eigenValuesReal1, eigenValuesImag1,
                        inverseEigenVectors1, scaledBranchLength1,
                        partials2Ptr, 0, eigenValuesReal2, eigenValuesImag2,
                        inverseEigenVectors2, scaledBranchLength2,
                        matrixIncr);
#endif

                    // MATRIX_VECTOR_HADAMARD_PRODUCT(destPtr, eigenVectors, gPartialTmp);
                    matVecDual<Partials, Partials>(
                        eigenVectors1, gPartialTmp1.data(), 0,
                        eigenVectors2, gPartialTmp2.data(), 0,
                        matrixIncr,
                        [&](int i, REALTYPE sum1, REALTYPE sum2) {
                            destPtr[i] = sum1 * sum2;
                        });

                }
            } else {
                static_assert(!sizeof(T), "calcPrePartialsPartials called with unknown type");
            }

            // end
            destPtr += kPartialsPaddedStateCount;
            partials1Ptr += kPartialsPaddedStateCount;
            partials2Ptr += kPartialsPaddedStateCount;
        }
    }
}

BEAGLE_CPU_TEMPLATE template <typename T, typename V>
void BeagleCPUSpectralImpl<BEAGLE_CPU_GENERIC>::calcPrePartialsStates(
            REALTYPE *destPartials,
            const REALTYPE *partials1,
            const int branchEigenIndex1,
            const int *states2,
            const int branchEigenIndex2,
            int startPattern,
            int endPattern) {

    INIT_TRANSPOSE1(info,
        eigenValuesReal, eigenValuesImag,
        eigenVectors, inverseEigenVectors,
        branchLength, categoryRate);

    const int matrixIncr = kStateCount + T_PAD;
    const int stateCountModFour = (kStateCount / 4) * 4;

#if defined(_OPENMP)
    #pragma omp parallel for num_threads(kCategoryCount)
#endif
    for (int l = 0; l < kCategoryCount; l++) {
        int v = l * kPartialsPaddedStateCount * kPatternCount + kPartialsPaddedStateCount * startPattern;

        const REALTYPE scaledBranchLength1 = static_cast<REALTYPE>(categoryRate1[l]) * branchLength1;
        const REALTYPE scaledBranchLength2 = static_cast<REALTYPE>(categoryRate2[l]) * branchLength2;

        const REALTYPE* partials1Ptr = &partials1[v];
        REALTYPE* destPtr = &destPartials[v];

        REALTYPE* intermediate = gPartialTmp2.data();
        REALTYPE* parent = gPartialTmp1.data();

        for (int k = startPattern; k < endPattern; k++) {

            const int state2 = states2[k];

            if constexpr (std::is_same_v<T, Bottom>) {

                // first step
#ifdef TEST_EB
                expScaledMatrixVectorMultiple2<States,None,Forward>(
                    intermediate, nullptr,
                    nullptr, state2, info_2, inverseEigenVectors2,
                    nullptr, 0, info_1, nullptr,
                    matrixIncr);
#else
                expScaledMatrixVectorMultiple<States,None>(
                    intermediate, nullptr,
                    nullptr, state2, eigenValuesReal2, eigenValuesImag2,
                    inverseEigenVectors2, scaledBranchLength2,
                    nullptr, 0, nullptr, nullptr,
                    nullptr, 0.0,
                    matrixIncr);
#endif

                MATRIX_VECTOR_SINGLE(parent[i] = sum2 * partials1Ptr[i], // fused Hadamard product
                    eigenVectors2, intermediate, sum2)

                // second step
#ifdef TEST_EB
                expScaledMatrixVectorMultiple2<Partials,None,Backward>(
                    intermediate, nullptr,
                    parent, 0, info_1, inverseEigenVectors1,
                    nullptr, 0, info_2, nullptr,
                    matrixIncr);
#else
                expScaledMatrixVectorMultiple<Partials,None>(
                    intermediate, nullptr,
                    parent, 0, eigenValuesReal1, eigenValuesImag1,
                    inverseEigenVectors1, scaledBranchLength1,
                    nullptr, 0, nullptr, nullptr,
                    nullptr, 0.0,
                    matrixIncr);
#endif

                MATRIX_VECTOR_SINGLE(destPtr[i] = sum1, eigenVectors1, intermediate, sum1);

            } else if constexpr (std::is_same_v<T, Top>) {

                if constexpr (std::is_same_v<V, Root>) {
#ifdef TEST_EB
                   expScaledMatrixVectorMultiple2<States,None,Forward>(
                        intermediate, nullptr,
                        nullptr, state2, info_2, inverseEigenVectors2,
                        nullptr, 0, info_1, nullptr,
                        matrixIncr);
#else
                    expScaledMatrixVectorMultiple<States,None>(
                        intermediate, nullptr,
                        nullptr, state2, eigenValuesReal2, eigenValuesImag2,
                        inverseEigenVectors2, scaledBranchLength2,
                        nullptr, 0, nullptr, nullptr,
                        nullptr, 0.0,
                        matrixIncr);
#endif

                    MATRIX_VECTOR_SINGLE(destPtr[i] = sum2 * partials1Ptr[i], // fused Hadamard product
                        eigenVectors2, intermediate, sum2)

                } else {

#ifdef TEST_EB
                    expScaledMatrixVectorMultiple2<Partials,States,Backward>(
                        gPartialTmp1.data(), gPartialTmp2.data(),
                        partials1Ptr, 0, info_1, inverseEigenVectors1,
                        nullptr, state2, info_2, inverseEigenVectors2,
                        matrixIncr);
#else
                    expScaledMatrixVectorMultiple<Partials,States>(
                        gPartialTmp1.data(), gPartialTmp2.data(),
                        partials1Ptr, 0, eigenValuesReal1, eigenValuesImag1,
                        inverseEigenVectors1, scaledBranchLength1,
                        nullptr, state2, eigenValuesReal2, eigenValuesImag2,
                        inverseEigenVectors2, scaledBranchLength2,
                        matrixIncr);
#endif

                    // MATRIX_VECTOR_HADAMARD_PRODUCT(destPtr, eigenVectors, gPartialTmp);
                    matVecDual<Partials, Partials>(
                        eigenVectors1, gPartialTmp1.data(), 0,
                        eigenVectors2, gPartialTmp2.data(), 0,
                        matrixIncr,
                        [&](int index, REALTYPE sum1, REALTYPE sum2) {
                            destPtr[index] = sum1 * sum2;
                        });
                }

            } else {
                static_assert(!sizeof(T), "calcPrePartialsStates called with unknown type");
            }

            // end
            destPtr += kPartialsPaddedStateCount;
            partials1Ptr += kPartialsPaddedStateCount;
        }
    }
}

BEAGLE_CPU_TEMPLATE
EigenDecomposition<BEAGLE_CPU_EIGEN_GENERIC>* BeagleCPUSpectralImpl<BEAGLE_CPU_GENERIC>::createEigenDecomposition(
        int decompositionCount,
        int stateCount,
        int categoryCount,
        long flags) {

    return new EigenDecompositionSpectral<BEAGLE_CPU_EIGEN_GENERIC>(
        decompositionCount, stateCount, categoryCount, flags);
}

BEAGLE_CPU_TEMPLATE
const char* BeagleCPUSpectralImpl<BEAGLE_CPU_GENERIC>::getName() {
	return getBeagleCPUSpectralName<BEAGLE_CPU_FACTORY_GENERIC>();
}

///////////////////////////////////////////////////////////////////////////////
// BeagleCPUImplFactory public methods

BEAGLE_CPU_FACTORY_TEMPLATE
BeagleImpl* BeagleCPUSpectralImplFactory<BEAGLE_CPU_FACTORY_GENERIC>::createImpl(int tipCount,
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

    BeagleImpl* impl = new BeagleCPUSpectralImpl<REALTYPE, T_PAD_DEFAULT, P_PAD_DEFAULT>();

    try {
        if (impl->createInstance(tipCount, partialsBufferCount, compactBufferCount, stateCount,
                                 patternCount, eigenBufferCount, matrixBufferCount,
                                 categoryCount,scaleBufferCount, resourceNumber,
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
const char* BeagleCPUSpectralImplFactory<BEAGLE_CPU_FACTORY_GENERIC>::getName() {
	return getBeagleCPUSpectralName<BEAGLE_CPU_FACTORY_GENERIC>();
}

BEAGLE_CPU_FACTORY_TEMPLATE
const long BeagleCPUSpectralImplFactory<BEAGLE_CPU_FACTORY_GENERIC>::getFlags() {
    long flags =  BEAGLE_FLAG_COMPUTATION_SYNCH |
                  BEAGLE_FLAG_SCALING_MANUAL | BEAGLE_FLAG_SCALING_ALWAYS | BEAGLE_FLAG_SCALING_AUTO |
                  BEAGLE_FLAG_THREADING_NONE | BEAGLE_FLAG_THREADING_CPP |
                  BEAGLE_FLAG_PROCESSOR_CPU |
                  BEAGLE_FLAG_VECTOR_NONE |
                  BEAGLE_FLAG_SCALERS_LOG | BEAGLE_FLAG_SCALERS_RAW |
                  BEAGLE_FLAG_EIGEN_COMPLEX | BEAGLE_FLAG_EIGEN_REAL |
                  BEAGLE_FLAG_INVEVEC_STANDARD | BEAGLE_FLAG_INVEVEC_TRANSPOSED |
                  BEAGLE_FLAG_PREORDER_TRANSPOSE_MANUAL | BEAGLE_FLAG_PREORDER_TRANSPOSE_AUTO |
                  BEAGLE_FLAG_SPECTRAL_REPRESENTATION |
                  BEAGLE_FLAG_FRAMEWORK_CPU;

    if (DOUBLE_PRECISION)
    	flags |= BEAGLE_FLAG_PRECISION_DOUBLE;
    else
    	flags |= BEAGLE_FLAG_PRECISION_SINGLE;
    return flags;
}

}	// namespace cpu
}	// namespace beagle

/*

__attribute__((always_inline)) / [[clang::always_inline]]

*/

#endif // BEAGLE_CPU_SPECTRAL_IMPL_HPP
