
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

#define EXPERIMENTAL_OPENMP


#define OFFSET    (4 + T_PAD)    // For easy conversion between 4/5

#define PREFETCH_MATRIX(num,matrices,w) \
    REALTYPE m##num##00, m##num##01, m##num##02, m##num##03, \
           m##num##10, m##num##11, m##num##12, m##num##13, \
           m##num##20, m##num##21, m##num##22, m##num##23, \
           m##num##30, m##num##31, m##num##32, m##num##33; \
    m##num##00 = matrices[w + OFFSET*0 + 0]; \
    m##num##01 = matrices[w + OFFSET*0 + 1]; \
    m##num##02 = matrices[w + OFFSET*0 + 2]; \
    m##num##03 = matrices[w + OFFSET*0 + 3]; \
    m##num##10 = matrices[w + OFFSET*1 + 0]; \
    m##num##11 = matrices[w + OFFSET*1 + 1]; \
    m##num##12 = matrices[w + OFFSET*1 + 2]; \
    m##num##13 = matrices[w + OFFSET*1 + 3]; \
    m##num##20 = matrices[w + OFFSET*2 + 0]; \
    m##num##21 = matrices[w + OFFSET*2 + 1]; \
    m##num##22 = matrices[w + OFFSET*2 + 2]; \
    m##num##23 = matrices[w + OFFSET*2 + 3]; \
    m##num##30 = matrices[w + OFFSET*3 + 0]; \
    m##num##31 = matrices[w + OFFSET*3 + 1]; \
    m##num##32 = matrices[w + OFFSET*3 + 2]; \
    m##num##33 = matrices[w + OFFSET*3 + 3];

#define PREFETCH_PARTIALS(num,partials,v) \
    REALTYPE p##num##0, p##num##1, p##num##2, p##num##3; \
    p##num##0 = partials[v + 0]; \
    p##num##1 = partials[v + 1]; \
    p##num##2 = partials[v + 2]; \
    p##num##3 = partials[v + 3];

#define PREFETCH_MATRIX_COLUMN(num,matrices,v) \
    REALTYPE sum##num##0, sum##num##1, sum##num##2, sum##num##3; \
    sum##num##0 = matrices[v    ]; \
    sum##num##1 = matrices[v + OFFSET*1]; \
    sum##num##2 = matrices[v + OFFSET*2]; \
    sum##num##3 = matrices[v + OFFSET*3];

#define PREFETCH_MATRIX_TRANSPOSE(num,matrices,w) \
    REALTYPE m##num##00, m##num##01, m##num##02, m##num##03, \
           m##num##10, m##num##11, m##num##12, m##num##13, \
           m##num##20, m##num##21, m##num##22, m##num##23, \
           m##num##30, m##num##31, m##num##32, m##num##33; \
    m##num##00 = matrices[w + OFFSET*0 + 0]; \
    m##num##10 = matrices[w + OFFSET*0 + 1]; \
    m##num##20 = matrices[w + OFFSET*0 + 2]; \
    m##num##30 = matrices[w + OFFSET*0 + 3]; \
    m##num##01 = matrices[w + OFFSET*1 + 0]; \
    m##num##11 = matrices[w + OFFSET*1 + 1]; \
    m##num##21 = matrices[w + OFFSET*1 + 2]; \
    m##num##31 = matrices[w + OFFSET*1 + 3]; \
    m##num##02 = matrices[w + OFFSET*2 + 0]; \
    m##num##12 = matrices[w + OFFSET*2 + 1]; \
    m##num##22 = matrices[w + OFFSET*2 + 2]; \
    m##num##32 = matrices[w + OFFSET*2 + 3]; \
    m##num##03 = matrices[w + OFFSET*3 + 0]; \
    m##num##13 = matrices[w + OFFSET*3 + 1]; \
    m##num##23 = matrices[w + OFFSET*3 + 2]; \
    m##num##33 = matrices[w + OFFSET*3 + 3];

#define DO_SCHUR_PRODUCT(qnum, pnum, snum) \
    p##qnum##0 = p##pnum##0 * sum##snum##0; \
    p##qnum##1 = p##pnum##1 * sum##snum##1; \
    p##qnum##2 = p##pnum##2 * sum##snum##2; \
    p##qnum##3 = p##pnum##3 * sum##snum##3;

#define INNER_PRODUCT(lhs, rhs) \
    p##lhs##0 * p##rhs##0 +        \
    p##lhs##1 * p##rhs##1 +        \
    p##lhs##2 * p##rhs##2 +        \
    p##lhs##3 * p##rhs##3

//#define DO_INTEGRATION(num) \
//    REALTYPE sum##num##0, sum##num##1, sum##num##2, sum##num##3; \
//    sum##num##0  = m##num##00 * p##num##0; \
//    sum##num##1  = m##num##10 * p##num##0; \
//    sum##num##2  = m##num##20 * p##num##0; \
//    sum##num##3  = m##num##30 * p##num##0; \
// \
//    sum##num##0 += m##num##01 * p##num##1; \
//    sum##num##1 += m##num##11 * p##num##1; \
//    sum##num##2 += m##num##21 * p##num##1; \
//    sum##num##3 += m##num##31 * p##num##1; \
// \
//    sum##num##0 += m##num##02 * p##num##2; \
//    sum##num##1 += m##num##12 * p##num##2; \
//    sum##num##2 += m##num##22 * p##num##2; \
//    sum##num##3 += m##num##32 * p##num##2; \
// \
//    sum##num##0 += m##num##03 * p##num##3; \
//    sum##num##1 += m##num##13 * p##num##3; \
//    sum##num##2 += m##num##23 * p##num##3; \
//    sum##num##3 += m##num##33 * p##num##3;

#define DO_INTEGRATION(num) \
    REALTYPE sum##num##0, sum##num##1, sum##num##2, sum##num##3; \
    sum##num##0  = m##num##00 * p##num##0 + \
                   m##num##01 * p##num##1 + \
                   m##num##02 * p##num##2 + \
                   m##num##03 * p##num##3;  \
 \
    sum##num##1  = m##num##10 * p##num##0 + \
                   m##num##11 * p##num##1 + \
                   m##num##12 * p##num##2 + \
                   m##num##13 * p##num##3;  \
 \
    sum##num##2  = m##num##20 * p##num##0 + \
                   m##num##21 * p##num##1 + \
                   m##num##22 * p##num##2 + \
                   m##num##23 * p##num##3;  \
\
    sum##num##3  = m##num##30 * p##num##0 + \
                   m##num##31 * p##num##1 + \
                   m##num##32 * p##num##2 + \
                   m##num##33 * p##num##3;


namespace beagle {
namespace cpu {

BEAGLE_CPU_FACTORY_TEMPLATE
inline const char* getBeagleCPUSpectralName(){ return "CPU-Spectral-Unknown"; };

template<>
inline const char* getBeagleCPUSpectralName<double>(){ return "CPU-Spectral-Double"; };

template<>
inline const char* getBeagleCPUSpectralName<float>(){ return "CPU-Spectral-Single"; };

BEAGLE_CPU_TEMPLATE
BeagleCPUSpectralImpl<BEAGLE_CPU_GENERIC>::~BeagleCPUSpectralImpl() {
    // free all that stuff...
    // If you delete partials, make sure not to delete the last element
    // which is TEMP_SCRATCH_PARTIAL twice.
}

///////////////////////////////////////////////////////////////////////////////
// private methods

BEAGLE_CPU_TEMPLATE
int BeagleCPUSpectralImpl<BEAGLE_CPU_GENERIC>::updateTransitionMatrices(int eigenIndex,
                                                                        const int* probabilityIndices,
                                                                        const int* firstDerivativeIndices,
                                                                        const int* secondDerivativeIndices,
                                                                        const double* edgeLengths,
                                                                        int count) {
    // for (int i = 0; i < count; i++) {
    //     fprintf(stderr, "uTM %d %d %f %d\n", eigenIndex, probabilityIndices[i], edgeLengths[i], 0);
    // }
    // fprintf(stderr, "\n");

    if (gBranchEigenInfo.size() < kMatrixCount) {
        gBranchEigenInfo.resize(kMatrixCount);
    }

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

        // const REALTYPE* matrices1 = gTransitionMatrices[child1TransMatIndex];
        // const REALTYPE* matrices2 = gTransitionMatrices[child2TransMatIndex];

        // const REALTYPE* eigenValues1 = gEigenDecomposition->getEigenValuesPtr(child1TransMatIndex);
        // const REALTYPE* eigenValues2 = gEigenDecomposition->getEigenValuesPtr(child2TransMatIndex);

        // const REALTYPE* eigenVectors1 = gEigenDecomposition->getEigenVectorsPtr(child1TransMatIndex);
        // const REALTYPE* eigenVectors2 = gEigenDecomposition->getEigenVectorsPtr(child2TransMatIndex);

        // const REALTYPE* inverseEigenVectors1 = gEigenDecomposition->getInverseEigenVectorsPtr(child1TransMatIndex);
        // const REALTYPE* inverseEigenVectors2 = gEigenDecomposition->getInverseEigenVectorsPtr(child2TransMatIndex);

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
                    // calcStatesStatesFixedScaling(destPartials, tipStates1, matrices1, tipStates2,
                    //                              matrices2, scalingFactors, startPattern, endPattern);
                    fprintf(stderr, "calcStatesStatesFixedScaling is not yet implemented for spectral representation\n");
                    exit(-1);
                } else {
                    // First compute without any scaling
                    // calcStatesStates(destPartials, tipStates1, matrices1, tipStates2, matrices2,
                    //                  startPattern, endPattern);
                    fprintf(stderr, "calcStatesStates is not yet implemented for spectral representation\n");
                    exit(-1);
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
                    // calcStatesPartialsFixedScaling(destPartials, tipStates1, matrices1, partials2,
                    //                                matrices2, scalingFactors, startPattern, endPattern);
                    fprintf(stderr, "calcStatesPartialsFixedScaling is not yet implemented for spectral representation\n");
                    exit(-1);
                } else {
                    // calcStatesPartials(destPartials, tipStates1, matrices1, partials2, matrices2,
                    //                    startPattern, endPattern);
                    fprintf(stderr, "calcStatesPartials is not yet implemented for spectral representation\n");
                    exit(-1);
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
                    // calcStatesPartialsFixedScaling(destPartials,tipStates2,matrices2,partials1,matrices1,
                    //                                scalingFactors, startPattern, endPattern);
                    fprintf(stderr, "calcStatesPartialsFixedScaling is not yet implemented for spectral representation\n");
                    exit(-1);
                } else {
                    // calcStatesPartials(destPartials, tipStates2, matrices2, partials1, matrices1,
                    //                    startPattern, endPattern);
                    fprintf(stderr, "calcStatesPartials is not yet implemented for spectral representation\n");
                    exit(-1);
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
                    // calcPartialsPartialsFixedScaling(destPartials,partials1,matrices1,partials2,
                    //                                  matrices2,scalingFactors,startPattern,endPattern);
                    fprintf(stderr, "calcPartialsPartialsFixedScaling is not yet implemented for spectral representation\n");
                    exit(-1);
                } else {
                    calcPartialsPartials(destPartials, 
                                        //  partials1, eigenValues1, eigenVectors1, inverseEigenVectors1,
                                        //  partials2, eigenValues2, eigenVectors2, inverseEigenVectors2,
                                         partials1, branchEigenIndex1,
                                         partials2, branchEigenIndex2,
                                         startPattern, endPattern, true);
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
void BeagleCPUSpectralImpl<BEAGLE_CPU_GENERIC>::calcPartialsPartials(
    REALTYPE *destPartials, 
    const REALTYPE *partials1, 
    const int branchEigenIndex1,
    const REALTYPE *partials2,
    const int branchEigenIndex2,
    int startPattern, 
    int endPattern, 
    bool isComplex) {
  
    if (gPartialTmp1.size() < kStateCount) {
        gPartialTmp1.resize(kStateCount);
    }

    if (gPartialTmp2.size() < kStateCount) {
        gPartialTmp2.resize(kStateCount);
    }

    const BranchEigenInfo& info1 = gBranchEigenInfo[branchEigenIndex1];
    const BranchEigenInfo& info2 = gBranchEigenInfo[branchEigenIndex2];

    // TODO: we can optimize for info1.eigenIndex == info2.eigenIndex
    
    const REALTYPE* eigenValuesReal1 = gEigenDecomposition->getEigenValuesPtr(info1.eigenIndex);
    const REALTYPE* eigenValuesReal2 = gEigenDecomposition->getEigenValuesPtr(info2.eigenIndex);

    const REALTYPE* eigenValuesImag1 = eigenValuesReal1 + kStateCount;
    const REALTYPE* eigenValuesImag2 = eigenValuesReal2 + kStateCount;

    const REALTYPE* eigenVectors1 = gEigenDecomposition->getEigenVectorsPtr(info1.eigenIndex);
    const REALTYPE* eigenVectors2 = gEigenDecomposition->getEigenVectorsPtr(info2.eigenIndex);

    const REALTYPE* inverseEigenVectors1 = gEigenDecomposition->getInverseEigenVectorsPtr(info1.eigenIndex);
    const REALTYPE* inverseEigenVectors2 = gEigenDecomposition->getInverseEigenVectorsPtr(info2.eigenIndex);

    const REALTYPE branchLength1 = info1.branchLength;
    const REALTYPE branchLength2 = info2.branchLength;

    const double* categoryRate1 = gCategoryRates[info1.categoryRatesIndex];
    const double* categoryRate2 = gCategoryRates[info2.categoryRatesIndex];

    const int matrixIncr = kStateCount; // TODO: no padding`
    const int stateCountModFour = (kStateCount / 4) * 4;

    for (int i = 0; i < kStateCount; i++) {
        fprintf(stderr, "eigenValuesImag1[%d] = %f\n", i, eigenValuesImag1[i]);
        fprintf(stderr, "eigenValuesImag2[%d] = %f\n", i, eigenValuesImag2[i]);
    }
    fprintf(stderr, "\n");

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

                const REALTYPE expat1 = exp(eigenValuesReal1[i] * scaledBranchLength1);
                const REALTYPE expat2 = exp(eigenValuesReal2[i] * scaledBranchLength2);

                if (eigenValuesImag1[i] == 0.0 && eigenValuesImag2[i] == 0.0) {
                    // Both real 
                    REALTYPE sum1 = 0.0, sum2 = 0.0;
                    for (int j = 0; j < kStateCount; j++) {
                        sum1 += inverseEigenVectors1[i * matrixIncr + j] * partials1Ptr[j];
                        sum2 += inverseEigenVectors2[i * matrixIncr + j] * partials2Ptr[j];
                    }

                    gPartialTmp1[i] = expat1 * sum1;
                    gPartialTmp2[i] = expat2 * sum2;

                } else if (eigenValuesImag1[i] != 0.0 && eigenValuesImag2[i] != 0.0) {
                    // Both complex conjugate pairs
                    //
                    // 2x2 conjugate block
                    // If A is 2x2 with complex conjugate pair eigenvalues a +/- bi, then
                    // exp(At) = exp(at)*( cos(bt)I + \frac{sin(bt)}{b}(A - aI)).

                    int i2 = i + 1;                   
                    const REALTYPE b1 = eigenValuesImag1[i];                                 
                    const REALTYPE expatcosbt1 = expat1 * cos(scaledBranchLength1 * b1);
                    const REALTYPE expatsinbt1 = expat1 * sin(scaledBranchLength1 * b1);

                    const REALTYPE b2 = eigenValuesImag2[i];
                    const REALTYPE expatcosbt2 = expat2 * cos(scaledBranchLength2 * b2);                    
                    const REALTYPE expatsinbt2 = expat2 * sin(scaledBranchLength2 * b2);

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
                    
                } else {
                    fprintf(stderr, "Error: Mismatched eigenvalue types in calcPartialsPartials; not yet implemented\n");
                    exit(-1);
                }
            }

            for (int i = 0; i < kStateCount; i++) {
                REALTYPE sum1 = 0.0, sum2 = 0.0;
                for (int j = 0; j < kStateCount; j++) {
                    sum1 += eigenVectors1[i * matrixIncr + j] * gPartialTmp1[j];
                    sum2 += eigenVectors2[i * matrixIncr + j] * gPartialTmp2[j];
                }

                destPtr[i] = sum1 * sum2;
            }

            
//                 REALTYPE sum1A = 0.0, sum2A = 0.0;
//                 REALTYPE sum1B = 0.0, sum2B = 0.0;
//                 int j = 0;
//                 for (; j < stateCountModFour; j += 4) {
//                     sum1A += matrices1Ptr[j + 0] * partials1Ptr[j + 0];
//                     sum2A += matrices2Ptr[j + 0] * partials2Ptr[j + 0];

//                     sum1B += matrices1Ptr[j + 1] * partials1Ptr[j + 1];
//                     sum2B += matrices2Ptr[j + 1] * partials2Ptr[j + 1];

//                     sum1A += matrices1Ptr[j + 2] * partials1Ptr[j + 2];
//                     sum2A += matrices2Ptr[j + 2] * partials2Ptr[j + 2];

//                     sum1B += matrices1Ptr[j + 3] * partials1Ptr[j + 3];
//                     sum2B += matrices2Ptr[j + 3] * partials2Ptr[j + 3];
//                 }

//                 for (; j < kStateCount; j++) {
//                     sum1A += matrices1Ptr[j] * partials1Ptr[j];
//                     sum2A += matrices2Ptr[j] * partials2Ptr[j];
//                 }

            destPtr += P_PAD;
            partials1Ptr += kPartialsPaddedStateCount;
            partials2Ptr += kPartialsPaddedStateCount;
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

#pragma omp parallel for num_threads(kCategoryCount)
    for (int l = 0; l < kCategoryCount; l++) {
        int u = l*4*kPaddedPatternCount;
        if (startPattern != 0) {
            u += 4*startPattern;
        }
        int w = l*4*OFFSET;


        PREFETCH_MATRIX(2,matrices2,w); // m200, m201, ..., m233
        PREFETCH_MATRIX_TRANSPOSE(1,matrices1,w); //m100, m101, ..., m133
        for (int k = startPattern; k < endPattern; k++) {
            PREFETCH_PARTIALS(2,partials2,u); // p20, p21, p22, p23
            PREFETCH_PARTIALS(1,partials1,u); // p10, p11, p12, p13

            DO_INTEGRATION(2); // defines sum20, sum21, sum22, sum23
            DO_SCHUR_PRODUCT(1, 1, 2); // reWrites p10, p11, p12, p13

            DO_INTEGRATION(1); // defines sum10, sum11, sum12, sum13

            // Final results
            destP[u    ] = sum10;
            destP[u + 1] = sum11;
            destP[u + 2] = sum12;
            destP[u + 3] = sum13;

            u += 4;

        }
    }
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

