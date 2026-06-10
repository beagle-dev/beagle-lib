
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

// #define USE_BRANCH_LIKELIHOOD
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
    kStateCountModFour = (kStateCount / 4) * 4;

    return returnCode;
}

BEAGLE_CPU_TEMPLATE
BeagleCPUSpectralImpl<BEAGLE_CPU_GENERIC>::~BeagleCPUSpectralImpl() {
    // Do nothing
}

// BEAGLE_CPU_TEMPLATE
// int BeagleCPUSpectralImpl<BEAGLE_CPU_GENERIC>::setCPUThreadCount(int threadCount) {

//     int returnCode = BeagleCPUImpl<BEAGLE_CPU_GENERIC>::setCPUThreadCount(threadCount);

//     return returnCode;
// }


BEAGLE_CPU_TEMPLATE
int BeagleCPUSpectralImpl<BEAGLE_CPU_GENERIC>::updateTransitionMatrices(int eigenIndex,
                                                                        const int* probabilityIndices,
                                                                        const int* firstDerivativeIndices,
                                                                        const int* secondDerivativeIndices,
                                                                        const double* edgeLengths,
                                                                        int count) {
    return this->updateBranchEigenInfo(eigenIndex, probabilityIndices, edgeLengths, count);
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

    // fprintf(stderr, "Call to upPartials\n");

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

        // fprintf(stderr, "start %d, end %d\n", startPattern, endPattern);

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
                                                  scalingFactors, startPattern, endPattern, currentPartition);
                } else {
                    // First compute without any scaling
                    calcStatesStates<NoScaling>(destPartials, tipStates1, branchEigenIndex1, tipStates2, branchEigenIndex2,
                                                nullptr, startPattern, endPattern, currentPartition);
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
                                                    scalingFactors, startPattern, endPattern, currentPartition);
                } else {
                    calcStatesPartials<NoScaling>(destPartials, tipStates1, branchEigenIndex1, partials2, branchEigenIndex2,
                                                  nullptr, startPattern, endPattern, currentPartition);
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
                                                    scalingFactors, startPattern, endPattern, currentPartition);
                } else {
                    calcStatesPartials<NoScaling>(destPartials, tipStates2, branchEigenIndex2, partials1, branchEigenIndex1,
                                                  nullptr, startPattern, endPattern, currentPartition);
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
                                                      scalingFactors, startPattern, endPattern, currentPartition);
                } else {
                    calcPartialsPartials<NoScaling>(destPartials, partials1, branchEigenIndex1, partials2, branchEigenIndex2,
                                                    nullptr, startPattern, endPattern, currentPartition);
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
                    calcPrePartialsStates<T, Root>(destPartials, partials1, 0, tipStates2, branchEigenIndex2,
                                              startPattern, endPattern, currentPartition);
                } else {
                    calcPrePartialsStates<T, NotRoot>(destPartials, partials1, branchEigenIndex1, tipStates2, branchEigenIndex2,
                        startPattern, endPattern, currentPartition);
                }
            } else { // T == Bottom
                calcPrePartialsStates<T, NotUsed>(destPartials, partials1, branchEigenIndex1, tipStates2, branchEigenIndex2,
                    startPattern, endPattern, currentPartition);
            }
        } else {
            if constexpr (std::is_same_v<T, Top>) {
                if (branchEigenIndex1 < 0) { // Parent node is root
                    calcPrePartialsPartials<T, Root>(destPartials, partials1, 0, partials2, branchEigenIndex2,
                        startPattern, endPattern, currentPartition);
                } else {
                    calcPrePartialsPartials<T, NotRoot>(destPartials, partials1, branchEigenIndex1, partials2, branchEigenIndex2,
                        startPattern, endPattern, currentPartition);
                }
            } else { // T == Bottom
                calcPrePartialsPartials<T, NotUsed>(destPartials, partials1, branchEigenIndex1, partials2, branchEigenIndex2,
                    startPattern, endPattern, currentPartition);
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
    const REALTYPE* evr##2 = gEigenDecomposition->getEigenValuesPtr(ifo##_2.eigenIndex); \
    \
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

// #define MATRIX_VECTOR(output, mat, vec) \
//     for (int i = 0; i < kStateCount; i++) { \
//         REALTYPE sum1 = 0.0, sum2 = 0.0; \
//         for (int j = 0; j < kStateCount; j++) { \
//             sum1 += mat##1[i * matrixIncr + j] * vec##1[j]; \
//             sum2 += mat##2[i * matrixIncr + j] * vec##2[j]; \
//         } \
//         output; \
//     }

#define MATRIX_VECTOR_SINGLE(output, mat, vec, sum) \
    for (int i = 0; i < kStateCount; i++) { \
        REALTYPE sum = 0.0; \
        for (int j = 0; j < kStateCount; j++) { \
            sum += mat[i * matrixIncr + j] * vec[j]; \
        } \
        output; \
    }

// #define MATRIX_VECTOR_HADAMARD_PRODUCT(out, mat, vec) \
//     MATRIX_VECTOR(out[i] = sum1 * sum2, mat, vec)

// #define MATRIX_VECTOR_HADAMARD_PRODUCT_SCALE(out, mat, vec, scale) \
//     MATRIX_VECTOR(out[i] = sum1 * sum2 * scale, mat, vec)

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
        int currentPartition) {

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

            REALTYPE* __restrict__ tmp1 = gPartialTmp1.data() + currentPartition * kPartialsPaddedStateCount;
            REALTYPE* __restrict__ tmp2 = gPartialTmp2.data() + currentPartition * kPartialsPaddedStateCount;

            expScaledMatrixVectorMultiple2<Partials,Partials,Forward>(
                tmp1, tmp2,
                partials1Ptr, 0, info_1, inverseEigenVectors1,
                partials2Ptr, 0, info_2, inverseEigenVectors2,
                matrixIncr);

            if constexpr (T::useScaleFactors) {
                const REALTYPE oneOverScaleFactor = REALTYPE(1.0) / scaleFactors[k];
                this->template matVecDual<Partials, Partials>(
                    eigenVectors1, tmp1, 0,
                    eigenVectors2, tmp2, 0,
                    matrixIncr,
                    [&](int i, REALTYPE sum1, REALTYPE sum2) {
                        destPtr[i] = sum1 * sum2 * oneOverScaleFactor;
                        // *(destPtr++) = sum1 * sum2 * oneOverScaleFactor;
                    });


                fprintf(stderr, "Hit a scaling factor: %.4e\n", scaleFactors[k]);
                exit(-1);
            } else {
                this->template matVecDual<Partials, Partials>(
                    eigenVectors1, tmp1, 0,
                    eigenVectors2, tmp2, 0,
                    matrixIncr,
                    [&](int i, REALTYPE sum1, REALTYPE sum2) {
                        destPtr[i] = sum1 * sum2;
                        // *(destPtr++) = sum1 * sum2;
                    });
            }

            // destPtr += P_PAD;

            //     *(destPtr++) = (sum1A + sum1B) * (sum2A + sum2B);
            // }
            // destPtr += P_PAD;

            destPtr += kPartialsPaddedStateCount;
            partials1Ptr += kPartialsPaddedStateCount;
            partials2Ptr += kPartialsPaddedStateCount;

            // fprintf(stderr, "pat %d\n", k);
        }
        // fprintf(stderr, "Done\n");
        // exit(-1);
        // }
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
        int endPattern,
        int currentPartition) {

            fprintf(stderr, "calcStatesPartials\n");
            exit(-1);

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

            REALTYPE* __restrict__ tmp1 = gPartialTmp1.data() + currentPartition * kPartialsPaddedStateCount;
            REALTYPE* __restrict__ tmp2 = gPartialTmp2.data() + currentPartition * kPartialsPaddedStateCount;

            expScaledMatrixVectorMultiple2<States,Partials,Forward>(
                tmp1, tmp2,
                nullptr, state1, info_1, inverseEigenVectors1,
                partials2Ptr, 0, info_2, inverseEigenVectors2,
                matrixIncr);

            if constexpr (T::useScaleFactors) {
                const REALTYPE oneOverScaleFactor = REALTYPE(1.0) / scaleFactors[k];
                this->template matVecDual<Partials, Partials>(
                    eigenVectors1, tmp1, 0,
                    eigenVectors2, tmp2, 0,
                    matrixIncr,
                    [&](int i, REALTYPE sum1, REALTYPE sum2) {
                        destPtr[i] = sum1 * sum2 * oneOverScaleFactor;
                    });
            } else {
                this->template matVecDual<Partials, Partials>(
                    eigenVectors1, tmp1, 0,
                    eigenVectors2, tmp2, 0,
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
        int endPattern,
        int currentPartition) {

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

        // const REALTYPE scaledBranchLength1 = static_cast<REALTYPE>(categoryRate1[l]) * branchLength1;
        // const REALTYPE scaledBranchLength2 = static_cast<REALTYPE>(categoryRate2[l]) * branchLength2;

        REALTYPE* __restrict__ destPtr = &destPartials[v];

        REALTYPE* __restrict__ tmp1 = gPartialTmp1.data() + currentPartition * kPartialsPaddedStateCount;
        REALTYPE* __restrict__ tmp2 = gPartialTmp2.data() + currentPartition * kPartialsPaddedStateCount;

        for (int k = startPattern; k < endPattern; k++) {

            const int state1 = states1[k];
            const int state2 = states2[k];

            expScaledMatrixVectorMultiple2<States,States,Forward>(
                tmp1, tmp2,
                nullptr, state1, info_1, inverseEigenVectors1,
                nullptr, state2, info_2, inverseEigenVectors2,
                matrixIncr);

            if constexpr (T::useScaleFactors) {
                const REALTYPE oneOverScaleFactor = REALTYPE(1.0) / scaleFactors[k];
                this->template matVecDual<Partials, Partials>(
                    eigenVectors1, tmp1, 0,
                    eigenVectors2, tmp2, 0,
                    matrixIncr,
                    [&](int i, REALTYPE sum1, REALTYPE sum2) {
                        destPtr[i] = sum1 * sum2 * oneOverScaleFactor;
                    });
            } else {
                this->template matVecDual<Partials, Partials>(
                    eigenVectors1, tmp1, 0,
                    eigenVectors2, tmp2, 0,
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
            int endPattern,
            int currentPartition) {

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

        const REALTYPE* __restrict__ partials1Ptr = &partials1[v];
        const REALTYPE* __restrict__ partials2Ptr = &partials2[v];
        REALTYPE* __restrict__ destPtr = &destPartials[v];

        REALTYPE* __restrict__ intermediate = gPartialTmp2.data() + currentPartition * kPartialsPaddedStateCount;
        REALTYPE* __restrict__ parent = gPartialTmp1.data() + currentPartition * kPartialsPaddedStateCount;

        for (int k = startPattern; k < endPattern; k++) {

            if constexpr (std::is_same_v<T, Bottom>) {

                // first step
                expScaledMatrixVectorMultiple2<Partials,None,Forward>(
                    intermediate, nullptr,
                    partials2Ptr, 0, info_2, inverseEigenVectors2,
                    nullptr, 0, info_1, nullptr,
                    matrixIncr);

                MATRIX_VECTOR_SINGLE(parent[i] = sum2 * partials1Ptr[i], // fused Hadamard product
                    eigenVectors2, intermediate, sum2)

                // second step
                expScaledMatrixVectorMultiple2<Partials,None,Backward>(
                    intermediate, nullptr,
                    parent, 0, info_1, inverseEigenVectors1,
                    nullptr, 0, info_2, nullptr,
                    matrixIncr);

                MATRIX_VECTOR_SINGLE(destPtr[i] = sum1, eigenVectors1, intermediate, sum1);

            } else if constexpr (std::is_same_v<T, Top>) {

                if constexpr (std::is_same_v<V, Root>) {

                    expScaledMatrixVectorMultiple2<Partials,None,Forward>(
                        intermediate, nullptr,
                        partials2Ptr, 0, info_2, inverseEigenVectors2,
                        nullptr, 0, info_1, nullptr,
                        matrixIncr);

                    MATRIX_VECTOR_SINGLE(destPtr[i] = sum2 * partials1Ptr[i], // fused Hadamard product
                        eigenVectors2, intermediate, sum2)

                } else {

                //   printVector("", 0, eigenValuesReal1, 2 * kStateCount);
                //   printVector("", 1, info_1.eval, 2 * kStateCount);
                    expScaledMatrixVectorMultiple2<Partials,Partials,Backward>(
                        intermediate, parent,
                        partials1Ptr, 0, info_1, inverseEigenVectors1,
                        partials2Ptr, 0, info_2, inverseEigenVectors2,
                        matrixIncr);

                    this->template matVecDual<Partials, Partials>(
                        eigenVectors1, intermediate, 0,
                        eigenVectors2, parent, 0,
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
            int endPattern,
            int currentPartition) {

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

        const REALTYPE* __restrict__ partials1Ptr = &partials1[v];
        REALTYPE* __restrict__ destPtr = &destPartials[v];

        REALTYPE* __restrict__ intermediate = gPartialTmp2.data() + currentPartition * kPartialsPaddedStateCount;
        REALTYPE* __restrict__ parent = gPartialTmp1.data() + currentPartition * kPartialsPaddedStateCount;

        for (int k = startPattern; k < endPattern; k++) {

            const int state2 = states2[k];

            if constexpr (std::is_same_v<T, Bottom>) {

                // first step
                expScaledMatrixVectorMultiple2<States,None,Forward>(
                    intermediate, nullptr,
                    nullptr, state2, info_2, inverseEigenVectors2,
                    nullptr, 0, info_1, nullptr,
                    matrixIncr);

                MATRIX_VECTOR_SINGLE(parent[i] = sum2 * partials1Ptr[i], // fused Hadamard product
                    eigenVectors2, intermediate, sum2)

                // second step
                expScaledMatrixVectorMultiple2<Partials,None,Backward>(
                    intermediate, nullptr,
                    parent, 0, info_1, inverseEigenVectors1,
                    nullptr, 0, info_2, nullptr,
                    matrixIncr);

                MATRIX_VECTOR_SINGLE(destPtr[i] = sum1, eigenVectors1, intermediate, sum1);

            } else if constexpr (std::is_same_v<T, Top>) {

                if constexpr (std::is_same_v<V, Root>) {

                    expScaledMatrixVectorMultiple2<States,None,Forward>(
                        intermediate, nullptr,
                        nullptr, state2, info_2, inverseEigenVectors2,
                        nullptr, 0, info_1, nullptr,
                        matrixIncr);

                    MATRIX_VECTOR_SINGLE(destPtr[i] = sum2 * partials1Ptr[i], // fused Hadamard product
                        eigenVectors2, intermediate, sum2)

                } else {

                    expScaledMatrixVectorMultiple2<Partials,States,Backward>(
                        intermediate, parent,
                        partials1Ptr, 0, info_1, inverseEigenVectors1,
                        nullptr, state2, info_2, inverseEigenVectors2,
                        matrixIncr);

                    this->template matVecDual<Partials, Partials>(
                        eigenVectors1, intermediate, 0,
                        eigenVectors2, parent, 0,
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
