
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

#include "libhmsbeagle/beagle.h"
#include "libhmsbeagle/CPU/BeagleCPUImpl.h"
#include "libhmsbeagle/CPU/BeagleCPUSpectralImpl.h"
#include "libhmsbeagle/CPU/EigenDecompositionSpectral.h"

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
    gIvecRowSums.resize(kEigenDecompCount * kStateCount);

    return returnCode;
}

BEAGLE_CPU_TEMPLATE
int BeagleCPUSpectralImpl<BEAGLE_CPU_GENERIC>::setEigenDecomposition(int eigenIndex,
                                         const double* inEigenVectors,
                                         const double* inInverseEigenVectors,
                                         const double* inEigenValues) {

    BeagleCPUImpl<BEAGLE_CPU_GENERIC>::setEigenDecomposition(eigenIndex, inEigenVectors, inInverseEigenVectors, inEigenValues);

    // TODO: could only compute when necessary
    const REALTYPE* inverseEigenVectorsPtr = gEigenDecomposition->getInverseEigenVectorsPtr(eigenIndex);
    for (int i = 0; i < kStateCount; i++) {
        REALTYPE rowSum = 0.0;
        for (int j = 0; j < kStateCount; j++) {
            rowSum += inverseEigenVectorsPtr[i * kStateCount + j];
        }
        gIvecRowSums[eigenIndex * kStateCount + i] = rowSum;
        // fprintf(stderr, "EigenIndex= %d i= %d rowSum= %f\n", eigenIndex, i, rowSum);
    }

    return BEAGLE_SUCCESS;
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
        auto& info = gBranchEigenInfo[probabilityIndices[i]];
        info.branchLength = edgeLengths[i];
        info.eigenIndex = eigenIndex;
        info.categoryRatesIndex = 0; // TODO: Implement category rates index
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

#define FORM_EXPAT(et, evr, sbl) \
    const REALTYPE et##1 = std::exp(evr##1[i] * sbl##1); \
    const REALTYPE et##2 = std::exp(evr##2[i] * sbl##2);

#define FORM_EXPAT_COS_SIN(et, evi, sbl) \
    int i2 = i + 1; \
    const REALTYPE b1 = evi##1[i]; \
    const REALTYPE et##cosbt1 = et##1 * std::cos(sbl##1 * b1); \
    const REALTYPE et##sinbt1 = et##1 * std::sin(sbl##1 * b1); \
    \
    const REALTYPE b2 = evi##2[i]; \
    const REALTYPE et##cosbt2 = et##2 * std::cos(sbl##2 * b2); \
    const REALTYPE et##sinbt2 = et##2 * std::sin(sbl##2 * b2);

#define MATRIX_VECTOR(output, mat, vec) \
    for (int i = 0; i < kStateCount; i++) { \
        REALTYPE sum1 = 0.0, sum2 = 0.0; \
        for (int j = 0; j < kStateCount; j++) { \
            sum1 += mat##1[i * matrixIncr + j] * vec##1[j]; \
            sum2 += mat##2[i * matrixIncr + j] * vec##2[j]; \
        } \
        output; \
    }

#define MATRIX_VECTOR_HADAMARD_PRODUCT(out, mat, vec) \
    MATRIX_VECTOR(out[i] = sum1 * sum2, mat, vec)

#define MATRIX_VECTOR_HADAMARD_PRODUCT_SCALE(out, mat, vec, scale) \
    MATRIX_VECTOR(out[i] = sum1 * sum2 * scale, mat, vec)

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

#pragma omp parallel for num_threads(kCategoryCount)
    for (int l = 0; l < kCategoryCount; l++) {
        int v = l * kPartialsPaddedStateCount * kPatternCount + kPartialsPaddedStateCount * startPattern;

        const REALTYPE scaledBranchLength1 = static_cast<REALTYPE>(categoryRate1[l]) * branchLength1;
        const REALTYPE scaledBranchLength2 = static_cast<REALTYPE>(categoryRate2[l]) * branchLength2;

        const REALTYPE* partials1Ptr = &partials1[v];
        const REALTYPE* partials2Ptr = &partials2[v];
        REALTYPE* destPtr = &destPartials[v];

        for (int k = startPattern; k < endPattern; k++) {

            for (int i = 0; i < kStateCount; i++) {

                FORM_EXPAT(expat, eigenValuesReal, scaledBranchLength);

                if (eigenValuesImag1[i] == 0.0 && eigenValuesImag2[i] == 0.0) {
                    // Both real
                    REALTYPE sum1 = 0.0, sum2 = 0.0;
                    for (int j = 0; j < kStateCount; j++) {
                        sum1 += inverseEigenVectors1[i * matrixIncr + j] * partials1Ptr[j];
                        sum2 += inverseEigenVectors2[i * matrixIncr + j] * partials2Ptr[j];
                    }

                    gPartialTmp1[i] = expat1 * sum1;
                    gPartialTmp2[i] = expat2 * sum2;

                } else {
                    // At least one complex conjugate pair
                    //
                    // 2x2 conjugate block
                    // If A is 2x2 with complex conjugate pair eigenvalues a +/- bi, then
                    // exp(At) = exp(at)*( cos(bt)I + \frac{sin(bt)}{b}(A - aI)).

                    FORM_EXPAT_COS_SIN(expat, eigenValuesImag, scaledBranchLength);

                    REALTYPE sum1A = 0.0, sum1B = 0.0, sum2A = 0.0, sum2B = 0.0;
                    for (int j = 0; j < kStateCount; j++) {
                        sum1A += (expatcosbt1 * inverseEigenVectors1[i * matrixIncr + j] +
                                    expatsinbt1 * inverseEigenVectors1[i2 * matrixIncr + j]) * partials1Ptr[j];

                        sum1B += (expatcosbt1 * inverseEigenVectors1[i2 * matrixIncr + j] -
                                    expatsinbt1 * inverseEigenVectors1[i * matrixIncr + j]) * partials1Ptr[j];

                        sum2A += (expatcosbt2 * inverseEigenVectors2[i * matrixIncr + j] +
                                    expatsinbt2 * inverseEigenVectors2[i2 * matrixIncr + j]) * partials2Ptr[j];

                        sum2B += (expatcosbt2 * inverseEigenVectors2[i2 * matrixIncr + j] -
                                    expatsinbt2 * inverseEigenVectors2[i * matrixIncr + j]) * partials2Ptr[j];
                    }

                    gPartialTmp1[i] = sum1A;
                    gPartialTmp1[i2] = sum1B;

                    gPartialTmp2[i] = sum2A;
                    gPartialTmp2[i2] = sum2B;

                    i++; // processed two conjugate rows
                }
            }

            if constexpr (T::useScaleFactors) {
                const REALTYPE oneOverScaleFactor = REALTYPE(1.0) / scaleFactors[k];
                MATRIX_VECTOR_HADAMARD_PRODUCT_SCALE(destPtr, eigenVectors, gPartialTmp, oneOverScaleFactor);
            } else {
                MATRIX_VECTOR_HADAMARD_PRODUCT(destPtr, eigenVectors, gPartialTmp);
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

#pragma omp parallel for num_threads(kCategoryCount)
    for (int l = 0; l < kCategoryCount; l++) {
        int v = l * kPartialsPaddedStateCount * kPatternCount + kPartialsPaddedStateCount * startPattern;

        const REALTYPE scaledBranchLength1 = static_cast<REALTYPE>(categoryRate1[l]) * branchLength1;
        const REALTYPE scaledBranchLength2 = static_cast<REALTYPE>(categoryRate2[l]) * branchLength2;

        const REALTYPE* partials2Ptr = &partials2[v];
        REALTYPE* destPtr = &destPartials[v];

        for (int k = startPattern; k < endPattern; k++) {

            const int state1 = states1[k];

            for (int i = 0; i < kStateCount; i++) {

                FORM_EXPAT(expat, eigenValuesReal, scaledBranchLength);

                if (eigenValuesImag1[i] == 0.0 && eigenValuesImag2[i] == 0.0) {
                    // Both real
                    gPartialTmp1[i] = expat1 * inverseEigenVectors1[i * matrixIncr + state1];

                    REALTYPE sum2 = 0.0;
                    for (int j = 0; j < kStateCount; j++) {
                        sum2 += inverseEigenVectors2[i * matrixIncr + j] * partials2Ptr[j];
                    }

                    gPartialTmp2[i] = expat2 * sum2;

                } else {
                    // At least one complex conjugate pair
                    //
                    // 2x2 conjugate block
                    // If A is 2x2 with complex conjugate pair eigenvalues a +/- bi, then
                    // exp(At) = exp(at)*( cos(bt)I + \frac{sin(bt)}{b}(A - aI)).

                    FORM_EXPAT_COS_SIN(expat, eigenValuesImag, scaledBranchLength);

                    gPartialTmp1[i] = expatcosbt1 * inverseEigenVectors1[i * matrixIncr + state1] +
                                    expatsinbt1 * inverseEigenVectors1[i2 * matrixIncr + state1];

                    gPartialTmp1[i2] = expatcosbt1 * inverseEigenVectors1[i2 * matrixIncr + state1] -
                                expatsinbt1 * inverseEigenVectors1[i * matrixIncr + state1];

                    REALTYPE sum2A = 0.0, sum2B = 0.0;
                    for (int j = 0; j < kStateCount; j++) {
                        sum2A += (expatcosbt2 * inverseEigenVectors2[i * matrixIncr + j] +
                                    expatsinbt2 * inverseEigenVectors2[i2 * matrixIncr + j]) * partials2Ptr[j];

                        sum2B += (expatcosbt2 * inverseEigenVectors2[i2 * matrixIncr + j] -
                                    expatsinbt2 * inverseEigenVectors2[i * matrixIncr + j]) * partials2Ptr[j];
                    }

                    gPartialTmp2[i] = sum2A;
                    gPartialTmp2[i2] = sum2B;

                    i++; // processed two conjugate rows
                }
            }

            if constexpr (T::useScaleFactors) {
                const REALTYPE oneOverScaleFactor = REALTYPE(1.0) / scaleFactors[k];
                MATRIX_VECTOR_HADAMARD_PRODUCT_SCALE(destPtr, eigenVectors, gPartialTmp, oneOverScaleFactor);
            } else {
                MATRIX_VECTOR_HADAMARD_PRODUCT(destPtr, eigenVectors, gPartialTmp);
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

#pragma omp parallel for num_threads(kCategoryCount)
    for (int l = 0; l < kCategoryCount; l++) {
        int v = l * kPartialsPaddedStateCount * kPatternCount + kPartialsPaddedStateCount * startPattern;

        const REALTYPE scaledBranchLength1 = static_cast<REALTYPE>(categoryRate1[l]) * branchLength1;
        const REALTYPE scaledBranchLength2 = static_cast<REALTYPE>(categoryRate2[l]) * branchLength2;

        REALTYPE* destPtr = &destPartials[v];

        for (int k = startPattern; k < endPattern; k++) {

            const int state1 = states1[k];
            const int state2 = states2[k];

            for (int i = 0; i < kStateCount; i++) {

                FORM_EXPAT(expat, eigenValuesReal, scaledBranchLength);

                if (eigenValuesImag1[i] == 0.0 && eigenValuesImag2[i] == 0.0) {
                    // Both real eigenvalues
                    // gPartialTmp1[i] = expat1 * inverseEigenVectors1[i * matrixIncr + state1];
                    // gPartialTmp2[i] = expat2 * inverseEigenVectors2[i * matrixIncr + state2];

                    if (state1 < kStateCount) {
                        gPartialTmp1[i] = expat1 * inverseEigenVectors1[i * matrixIncr + state1];
                    } else {
                        gPartialTmp1[i] = expat1 * inverseEigenVectors1[i * matrixIncr + state1];
                    }

                    if (state2 < kStateCount) {
                        gPartialTmp2[i] = expat2 * inverseEigenVectors2[i * matrixIncr + state2];
                    } else {
                        gPartialTmp2[i] = expat2 * inverseEigenVectors2[i * matrixIncr + state2];
                    }
                } else {
                    // At least one complex conjugate pair
                    //
                    // 2x2 conjugate block
                    // If A is 2x2 with complex conjugate pair eigenvalues a +/- bi, then
                    // exp(At) = exp(at)*( cos(bt)I + \frac{sin(bt)}{b}(A - aI)).

                    FORM_EXPAT_COS_SIN(expat, eigenValuesImag, scaledBranchLength);

                    gPartialTmp1[i] = expatcosbt1 * inverseEigenVectors1[i * matrixIncr + state1] +
                                    expatsinbt1 * inverseEigenVectors1[i2 * matrixIncr + state1];
                    gPartialTmp1[i2] = expatcosbt1 * inverseEigenVectors1[i2 * matrixIncr + state1] -
                                expatsinbt1 * inverseEigenVectors1[i * matrixIncr + state1];

                    gPartialTmp2[i] = expatcosbt2 * inverseEigenVectors2[i * matrixIncr + state2] +
                                expatsinbt2 * inverseEigenVectors2[i2 * matrixIncr + state2];
                    gPartialTmp2[i2] = expatcosbt2 * inverseEigenVectors2[i2 * matrixIncr + state2] -
                                expatsinbt2 * inverseEigenVectors2[i * matrixIncr + state2];

                    i++; // processed two conjugate rows
                }
            }

            if constexpr (T::useScaleFactors) {
                const REALTYPE oneOverScaleFactor = REALTYPE(1.0) / scaleFactors[k];
                MATRIX_VECTOR_HADAMARD_PRODUCT_SCALE(destPtr, eigenVectors, gPartialTmp, oneOverScaleFactor);
            } else {
                MATRIX_VECTOR_HADAMARD_PRODUCT(destPtr, eigenVectors, gPartialTmp);
            }

            destPtr += kPartialsPaddedStateCount;
        }
    }
}

BEAGLE_CPU_TEMPLATE
void BeagleCPUSpectralImpl<BEAGLE_CPU_GENERIC>::calcPrePartialsPartials(REALTYPE* destP,
                                                                   const REALTYPE* partials1,
                                                                   const REALTYPE* matrices1,
                                                                   const REALTYPE* partials2,
                                                                   const REALTYPE* matrices2,
                                                                   int startPattern,
                                                                   int endPattern) {

    fprintf(stderr, "calcPrePartialsPartials is not yet implemented for spectral representation\n");
    exit(-1);
}

BEAGLE_CPU_TEMPLATE
EigenDecomposition<BEAGLE_CPU_EIGEN_GENERIC>* BeagleCPUSpectralImpl<BEAGLE_CPU_GENERIC>::createEigenDecomposition(
        int decompositionCount,
        int stateCount,
        int categoryCount,
        long flags) {

    return new EigenDecompositionSpectral<BEAGLE_CPU_EIGEN_GENERIC>(
        decompositionCount, stateCount, categoryCount, flags);
        // return new EigenDecompositionSquare<BEAGLE_CPU_EIGEN_GENERIC>(
        // decompositionCount, stateCount, categoryCount, flags);
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

#endif // BEAGLE_CPU_SPECTRAL_IMPL_HPP
