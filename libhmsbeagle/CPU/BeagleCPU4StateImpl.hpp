
/*
 *  BeagleCPU4StateImpl.cpp
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
 * @author Aaron Darling
 */

#ifndef BEAGLE_CPU_4STATE_IMPL_HPP
#define BEAGLE_CPU_4STATE_IMPL_HPP

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
#include "libhmsbeagle/CPU/BeagleCPU4StateImpl.h"

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
inline const char* getBeagleCPU4StateName(){ return "CPU-4State-Unknown"; };

template<>
inline const char* getBeagleCPU4StateName<double>(){ return "CPU-4State-Double"; };

template<>
inline const char* getBeagleCPU4StateName<float>(){ return "CPU-4State-Single"; };

BEAGLE_CPU_TEMPLATE
BeagleCPU4StateImpl<BEAGLE_CPU_GENERIC>::~BeagleCPU4StateImpl() {
    // free all that stuff...
    // If you delete partials, make sure not to delete the last element
    // which is TEMP_SCRATCH_PARTIAL twice.
}

///////////////////////////////////////////////////////////////////////////////
// private methods

/*
 * Calculates partial likelihoods at a node when both children have states.
 */
BEAGLE_CPU_TEMPLATE
void BeagleCPU4StateImpl<BEAGLE_CPU_GENERIC>::calcStatesStates(REALTYPE* destP,
                                                               const int* states1,
                                                               const REALTYPE* matrices1,
                                                               const int* states2,
                                                               const REALTYPE* matrices2,
                                                               int startPattern,
                                                               int endPattern) {

#pragma omp parallel for num_threads(kCategoryCount)
    for (int l = 0; l < kCategoryCount; l++) {
        int v = l*4*kPaddedPatternCount;
        if (startPattern != 0) {
          v += 4*startPattern;
        }

        int w = l*4*OFFSET;

        for (int k = startPattern; k < endPattern; k++) {

            const int state1 = states1[k];
            const int state2 = states2[k];

            destP[v    ] = matrices1[w            + state1] *
                           matrices2[w            + state2];
            destP[v + 1] = matrices1[w + OFFSET*1 + state1] *
                           matrices2[w + OFFSET*1 + state2];
            destP[v + 2] = matrices1[w + OFFSET*2 + state1] *
                           matrices2[w + OFFSET*2 + state2];
            destP[v + 3] = matrices1[w + OFFSET*3 + state1] *
                           matrices2[w + OFFSET*3 + state2];
           v += 4;
        }
    }
}

BEAGLE_CPU_TEMPLATE
void BeagleCPU4StateImpl<BEAGLE_CPU_GENERIC>::calcStatesStatesFixedScaling(REALTYPE* destP,
                                                                           const int* states1,
                                                                           const REALTYPE* matrices1,
                                                                           const int* states2,
                                                                           const REALTYPE* matrices2,
                                                                           const REALTYPE* scaleFactors,
                                                                           int startPattern,
                                                                           int endPattern) {

#pragma omp parallel for num_threads(kCategoryCount)
    for (int l = 0; l < kCategoryCount; l++) {
        int v = l*4*kPaddedPatternCount;
        if (startPattern != 0) {
          v += 4*startPattern;
        }

        int w = l*4*OFFSET;

        for (int k = startPattern; k < endPattern; k++) {

            const int state1 = states1[k];
            const int state2 = states2[k];
            const REALTYPE scaleFactor = scaleFactors[k];

            destP[v    ] = matrices1[w            + state1] *
                           matrices2[w            + state2] / scaleFactor;
            destP[v + 1] = matrices1[w + OFFSET*1 + state1] *
                           matrices2[w + OFFSET*1 + state2] / scaleFactor;
            destP[v + 2] = matrices1[w + OFFSET*2 + state1] *
                           matrices2[w + OFFSET*2 + state2] / scaleFactor;
            destP[v + 3] = matrices1[w + OFFSET*3 + state1] *
                           matrices2[w + OFFSET*3 + state2] / scaleFactor;
            v += 4;
        }
    }
}

/*
 * Calculates partial likelihoods at a node when one child has states and one has partials.
 */
BEAGLE_CPU_TEMPLATE
void BeagleCPU4StateImpl<BEAGLE_CPU_GENERIC>::calcStatesPartials(REALTYPE* destP,
                                                                 const int* states1,
                                                                 const REALTYPE* matrices1,
                                                                 const REALTYPE* partials2,
                                                                 const REALTYPE* matrices2,
                                                                 int startPattern,
                                                                 int endPattern) {

#pragma omp parallel for num_threads(kCategoryCount)
    for (int l = 0; l < kCategoryCount; l++) {
        int u = l*4*kPaddedPatternCount;
        if (startPattern != 0) {
          u += 4*startPattern;
        }

        int w = l*4*OFFSET;

        PREFETCH_MATRIX(2,matrices2,w);

        for (int k = startPattern; k < endPattern; k++) {

            const int state1 = states1[k];

            PREFETCH_PARTIALS(2,partials2,u);

            DO_INTEGRATION(2); // defines sum20, sum21, sum22, sum23;

            destP[u    ] = matrices1[w            + state1] * sum20;
            destP[u + 1] = matrices1[w + OFFSET*1 + state1] * sum21;
            destP[u + 2] = matrices1[w + OFFSET*2 + state1] * sum22;
            destP[u + 3] = matrices1[w + OFFSET*3 + state1] * sum23;

            u += 4;
        }
    }
}

BEAGLE_CPU_TEMPLATE
void BeagleCPU4StateImpl<BEAGLE_CPU_GENERIC>::calcStatesPartialsFixedScaling(REALTYPE* destP,
                                                                             const int* states1,
                                                                             const REALTYPE* matrices1,
                                                                             const REALTYPE* partials2,
                                                                             const REALTYPE* matrices2,
                                                                             const REALTYPE* scaleFactors,
                                                                             int startPattern,
                                                                             int endPattern) {

#pragma omp parallel for num_threads(kCategoryCount)
    for (int l = 0; l < kCategoryCount; l++) {
        int u = l*4*kPaddedPatternCount;
        if (startPattern != 0) {
          u += 4*startPattern;
        }

        int w = l*4*OFFSET;

        PREFETCH_MATRIX(2,matrices2,w);

        for (int k = startPattern; k < endPattern; k++) {

            const int state1 = states1[k];
            const REALTYPE scaleFactor = scaleFactors[k];

            PREFETCH_PARTIALS(2,partials2,u);

            DO_INTEGRATION(2); // defines sum20, sum21, sum22, sum23

            destP[u    ] = matrices1[w            + state1] * sum20 / scaleFactor;
            destP[u + 1] = matrices1[w + OFFSET*1 + state1] * sum21 / scaleFactor;
            destP[u + 2] = matrices1[w + OFFSET*2 + state1] * sum22 / scaleFactor;
            destP[u + 3] = matrices1[w + OFFSET*3 + state1] * sum23 / scaleFactor;

            u += 4;
        }
    }
}

BEAGLE_CPU_TEMPLATE
void BeagleCPU4StateImpl<BEAGLE_CPU_GENERIC>::calcPartialsPartials(REALTYPE* destP,
                                                                   const REALTYPE* partials1,
                                                                   const REALTYPE* matrices1,
                                                                   const REALTYPE* partials2,
                                                                   const REALTYPE* matrices2,
                                                                   int startPattern,
                                                                   int endPattern) {


#pragma omp parallel for num_threads(kCategoryCount)
    for (int l = 0; l < kCategoryCount; l++) {
        int u = l*4*kPaddedPatternCount;
        if (startPattern != 0) {
          u += 4*startPattern;
        }
        int w = l*4*OFFSET;

        PREFETCH_MATRIX(1,matrices1,w);
        PREFETCH_MATRIX(2,matrices2,w);
        for (int k = startPattern; k < endPattern; k++) {
            PREFETCH_PARTIALS(1,partials1,u);
            PREFETCH_PARTIALS(2,partials2,u);

            DO_INTEGRATION(1); // defines sum10, sum11, sum12, sum13
            DO_INTEGRATION(2); // defines sum20, sum21, sum22, sum23

            // Final results
            destP[u    ] = sum10 * sum20;
            destP[u + 1] = sum11 * sum21;
            destP[u + 2] = sum12 * sum22;
            destP[u + 3] = sum13 * sum23;

            u += 4;

        }
    }
}

BEAGLE_CPU_TEMPLATE
void BeagleCPU4StateImpl<BEAGLE_CPU_GENERIC>::calcPrePartialsPartials(REALTYPE* destP,
                                                                   const REALTYPE* partials1,
                                                                   const REALTYPE* matrices1,
                                                                   const REALTYPE* partials2,
                                                                   const REALTYPE* matrices2,
                                                                   int startPattern,
                                                                   int endPattern) {


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
void BeagleCPU4StateImpl<BEAGLE_CPU_GENERIC>::calcPrePartialsStates(REALTYPE* destP,
                                                                    const REALTYPE* partials1,
                                                                    const REALTYPE* matrices1,
                                                                    const int* states2,
                                                                    const REALTYPE* matrices2,
                                                                    int startPattern,
                                                                    int endPattern) {


#pragma omp parallel for num_threads(kCategoryCount)
    for (int l = 0; l < kCategoryCount; l++) {
        int u = l*4*kPaddedPatternCount;
        if (startPattern != 0) {
            u += 4*startPattern;
        }
        int w = l*4*OFFSET;


        PREFETCH_MATRIX_TRANSPOSE(1, matrices1, w); //m100, m101, ..., m133
        for (int k = startPattern; k < endPattern; k++) {
            PREFETCH_PARTIALS(1, partials1, u); // p10, p11, p12, p13

            const int state2 = states2[k];
            PREFETCH_MATRIX_COLUMN(2, matrices2, w + state2); // sum20, sum21, sum22, sum23

            DO_SCHUR_PRODUCT(1, 1, 2); // reWrites p10, p11, p12, p13

            DO_INTEGRATION(1); // defines sum10, sum11, sum12, sum13

            // Final results
            destP[u] = sum10;
            destP[u + 1] = sum11;
            destP[u + 2] = sum12;
            destP[u + 3] = sum13;

            u += 4;

        }
    }
}

BEAGLE_CPU_TEMPLATE
void BeagleCPU4StateImpl<BEAGLE_CPU_GENERIC>::calcPartialsPartialsAutoScaling(REALTYPE* destP,
                                                                    const REALTYPE* partials1,
                                                                    const REALTYPE* matrices1,
                                                                    const REALTYPE* partials2,
                                                                    const REALTYPE* matrices2,
                                                                    int* activateScaling) {


#pragma omp parallel for num_threads(kCategoryCount)
    for (int l = 0; l < kCategoryCount; l++) {
        int u = l*4*kPaddedPatternCount;
        int w = l*4*OFFSET;

        PREFETCH_MATRIX(1,matrices1,w);
        PREFETCH_MATRIX(2,matrices2,w);
        for (int k = 0; k < kPatternCount; k++) {
            PREFETCH_PARTIALS(1,partials1,u);
            PREFETCH_PARTIALS(2,partials2,u);

            DO_INTEGRATION(1); // defines sum10, sum11, sum12, sum13
            DO_INTEGRATION(2); // defines sum20, sum21, sum22, sum23

            // Final results
            destP[u    ] = sum10 * sum20;
            destP[u + 1] = sum11 * sum21;
            destP[u + 2] = sum12 * sum22;
            destP[u + 3] = sum13 * sum23;

            if (*activateScaling == 0) {
                int expTmp;
                int expMax;
                frexp(destP[u], &expMax);
                frexp(destP[u + 1], &expTmp);
                if (abs(expTmp) > abs(expMax))
                    expMax = expTmp;
                frexp(destP[u + 2], &expTmp);
                if (abs(expTmp) > abs(expMax))
                    expMax = expTmp;
                frexp(destP[u + 3], &expTmp);
                if (abs(expTmp) > abs(expMax))
                    expMax = expTmp;

                if(abs(expMax) > scalingExponentThreshold) {
                    *activateScaling = 1;
                }
            }

            u += 4;

        }
    }
}


BEAGLE_CPU_TEMPLATE
void BeagleCPU4StateImpl<BEAGLE_CPU_GENERIC>::calcPartialsPartialsFixedScaling(REALTYPE* destP,
                                                                               const REALTYPE* partials1,
                                                                               const REALTYPE* matrices1,
                                                                               const REALTYPE* partials2,
                                                                               const REALTYPE* matrices2,
                                                                               const REALTYPE* scaleFactors,
                                                                               int startPattern,
                                                                               int endPattern) {

#pragma omp parallel for num_threads(kCategoryCount)
    for (int l = 0; l < kCategoryCount; l++) {
        int u = l*4*kPaddedPatternCount;
        if (startPattern != 0) {
          u += 4*startPattern;
        }

        int w = l*4*OFFSET;

        PREFETCH_MATRIX(1,matrices1,w);
        PREFETCH_MATRIX(2,matrices2,w);

        for (int k = startPattern; k < endPattern; k++) {

            // Prefetch scale factor
            const REALTYPE scaleFactor = scaleFactors[k];

            PREFETCH_PARTIALS(1,partials1,u);
            PREFETCH_PARTIALS(2,partials2,u);

            DO_INTEGRATION(1); // defines sum10, sum11, sum12, sum13
            DO_INTEGRATION(2); // defines sum20, sum21, sum22, sum23

            // Final results
            destP[u    ] = sum10 * sum20 / scaleFactor;
            destP[u + 1] = sum11 * sum21 / scaleFactor;
            destP[u + 2] = sum12 * sum22 / scaleFactor;
            destP[u + 3] = sum13 * sum23 / scaleFactor;

            u += 4;
        }
    }
}

BEAGLE_CPU_TEMPLATE
int inline BeagleCPU4StateImpl<BEAGLE_CPU_GENERIC>::integrateOutStatesAndScale(const REALTYPE* integrationTmp,
                                                                      const int stateFrequenciesIndex,
                                                            const int scalingFactorsIndex,
                                                            double* outSumLogLikelihood) {

    int returnCode = BEAGLE_SUCCESS;

    REALTYPE freq0, freq1, freq2, freq3;
    freq0 = gStateFrequencies[stateFrequenciesIndex][0];
    freq1 = gStateFrequencies[stateFrequenciesIndex][1];
    freq2 = gStateFrequencies[stateFrequenciesIndex][2];
    freq3 = gStateFrequencies[stateFrequenciesIndex][3];

    int u = 0;
    for(int k = 0; k < kPatternCount; k++) {
        REALTYPE sumOverI =
        freq0 * integrationTmp[u    ] +
        freq1 * integrationTmp[u + 1] +
        freq2 * integrationTmp[u + 2] +
        freq3 * integrationTmp[u + 3];

        u += 4;

        outLogLikelihoodsTmp[k] = log(sumOverI);
    }

    if (scalingFactorsIndex != BEAGLE_OP_NONE) {
        const REALTYPE* scalingFactors = gScaleBuffers[scalingFactorsIndex];
        for(int k=0; k < kPatternCount; k++) {
            outLogLikelihoodsTmp[k] += scalingFactors[k];
        }
    }

    *outSumLogLikelihood = 0.0;
    for(int k=0; k < kPatternCount; k++) {
        *outSumLogLikelihood += outLogLikelihoodsTmp[k] * gPatternWeights[k];
    }

    if (*outSumLogLikelihood != *outSumLogLikelihood)
        returnCode = BEAGLE_ERROR_FLOATING_POINT;

    return returnCode;
}

BEAGLE_CPU_TEMPLATE
void inline BeagleCPU4StateImpl<BEAGLE_CPU_GENERIC>::integrateOutStatesAndScaleByPartition(
                                                              const REALTYPE* integrationTmp,
                                                              const int* stateFrequenciesIndices,
                                                              const int* cumulativeScaleIndices,
                                                              const int* partitionIndices,
                                                              int partitionCount,
                                                              double* outSumLogLikelihoodByPartition) {


    for (int p = 0; p < partitionCount; p++) {
      int pIndex = partitionIndices[p];
      int startPattern = gPatternPartitionsStartPatterns[pIndex];
      int endPattern = gPatternPartitionsStartPatterns[pIndex + 1];

      const int stateFrequenciesIndex = stateFrequenciesIndices[p];
      const int scalingFactorsIndex = cumulativeScaleIndices[p];

      REALTYPE freq0, freq1, freq2, freq3;
      freq0 = gStateFrequencies[stateFrequenciesIndex][0];
      freq1 = gStateFrequencies[stateFrequenciesIndex][1];
      freq2 = gStateFrequencies[stateFrequenciesIndex][2];
      freq3 = gStateFrequencies[stateFrequenciesIndex][3];

      int u = startPattern * 4;
      for(int k = startPattern; k < endPattern; k++) {
          REALTYPE sumOverI =
          freq0 * integrationTmp[u    ] +
          freq1 * integrationTmp[u + 1] +
          freq2 * integrationTmp[u + 2] +
          freq3 * integrationTmp[u + 3];

          u += 4;

          outLogLikelihoodsTmp[k] = log(sumOverI);
      }

      if (scalingFactorsIndex != BEAGLE_OP_NONE) {
          const REALTYPE* scalingFactors = gScaleBuffers[scalingFactorsIndex];
          for(int k=startPattern; k < endPattern; k++) {
              outLogLikelihoodsTmp[k] += scalingFactors[k];
          }
      }

      outSumLogLikelihoodByPartition[p] = 0.0;
      for(int k=startPattern; k < endPattern; k++) {
          outSumLogLikelihoodByPartition[p] += outLogLikelihoodsTmp[k] * gPatternWeights[k];
      }
    }

}

#define FAST_MAX(x,y)	(x > y ? x : y)
//#define BEAGLE_TEST_OPTIMIZATION
/*
 * Re-scales the partial likelihoods such that the largest is one.
 */
BEAGLE_CPU_TEMPLATE
void BeagleCPU4StateImpl<BEAGLE_CPU_GENERIC>::rescalePartials(REALTYPE* destP,
		REALTYPE* scaleFactors,
		REALTYPE* cumulativeScaleFactors,
        const int  fillWithOnes) {

	bool useLogScalars = kFlags & BEAGLE_FLAG_SCALERS_LOG;

    for (int k = 0; k < kPatternCount; k++) {
    	REALTYPE max = 0;
        const int patternOffset = k * 4;
        for (int l = 0; l < kCategoryCount; l++) {
            int offset = l * kPaddedPatternCount * 4 + patternOffset;

#ifdef BEAGLE_TEST_OPTIMIZATION
            REALTYPE max01 = FAST_MAX(destP[offset + 0], destP[offset + 1]);
            REALTYPE max23 = FAST_MAX(destP[offset + 2], destP[offset + 3]);
            max = FAST_MAX(max, max01);
            max = FAST_MAX(max, max23);
#else
			#pragma unroll
            for (int i = 0; i < 4; i++) {
                if(destP[offset] > max)
                    max = destP[offset];
                offset++;
            }
#endif
        }

        if (max == 0)
            max = REALTYPE(1.0);

        REALTYPE oneOverMax = REALTYPE(1.0) / max;
        for (int l = 0; l < kCategoryCount; l++) {
            int offset = l * kPaddedPatternCount * 4 + patternOffset;
			#pragma unroll
            for (int i = 0; i < 4; i++)
                destP[offset++] *= oneOverMax;
        }

        if (useLogScalars) {
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
void BeagleCPU4StateImpl<BEAGLE_CPU_GENERIC>::rescalePartialsByPartition(
                                                                    REALTYPE* destP,
                                                                    REALTYPE* scaleFactors,
                                                                    REALTYPE* cumulativeScaleFactors,
                                                                    const int fillWithOnes,
                                                                    const int partitionIndex) {

    bool useLogScalars = kFlags & BEAGLE_FLAG_SCALERS_LOG;

    int startPattern = gPatternPartitionsStartPatterns[partitionIndex];
    int endPattern = gPatternPartitionsStartPatterns[partitionIndex + 1];

    for (int k = startPattern; k < endPattern; k++) {
      REALTYPE max = 0;
        const int patternOffset = k * 4;
        for (int l = 0; l < kCategoryCount; l++) {
            int offset = l * kPaddedPatternCount * 4 + patternOffset;

#ifdef BEAGLE_TEST_OPTIMIZATION
            REALTYPE max01 = FAST_MAX(destP[offset + 0], destP[offset + 1]);
            REALTYPE max23 = FAST_MAX(destP[offset + 2], destP[offset + 3]);
            max = FAST_MAX(max, max01);
            max = FAST_MAX(max, max23);
#else
      #pragma unroll
            for (int i = 0; i < 4; i++) {
                if(destP[offset] > max)
                    max = destP[offset];
                offset++;
            }
#endif
        }

        if (max == 0)
            max = REALTYPE(1.0);

        REALTYPE oneOverMax = REALTYPE(1.0) / max;
        for (int l = 0; l < kCategoryCount; l++) {
            int offset = l * kPaddedPatternCount * 4 + patternOffset;
      #pragma unroll
            for (int i = 0; i < 4; i++)
                destP[offset++] *= oneOverMax;
        }

        if (useLogScalars) {
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
int BeagleCPU4StateImpl<BEAGLE_CPU_GENERIC>::calcEdgeLogLikelihoods(const int parIndex,
                                                           const int childIndex,
                                                           const int probIndex,
                                                           const int categoryWeightsIndex,
                                                           const int stateFrequenciesIndex,
                                                           const int scalingFactorsIndex,
                                                           double* outSumLogLikelihood) {
    // TODO: implement derivatives for calculateEdgeLnL

    assert(parIndex >= kTipCount);

    const REALTYPE* partialsParent = gPartials[parIndex];
    const REALTYPE* transMatrix = gTransitionMatrices[probIndex];
    const REALTYPE* wt = gCategoryWeights[categoryWeightsIndex];


    memset(integrationTmp, 0, (kPatternCount * kStateCount)*sizeof(REALTYPE));

    if (childIndex < kTipCount && gTipStates[childIndex]) { // Integrate against a state at the child

        const int* statesChild = gTipStates[childIndex];
        int v = 0; // Index for parent partials
        int w = 0;
        for(int l = 0; l < kCategoryCount; l++) {
            int u = 0; // Index in resulting product-partials (summed over categories)
            const REALTYPE weight = wt[l];
            for(int k = 0; k < kPatternCount; k++) {

                const int stateChild = statesChild[k];

                integrationTmp[u    ] += transMatrix[w            + stateChild] * partialsParent[v    ] * weight;
                integrationTmp[u + 1] += transMatrix[w + OFFSET*1 + stateChild] * partialsParent[v + 1] * weight;
                integrationTmp[u + 2] += transMatrix[w + OFFSET*2 + stateChild] * partialsParent[v + 2] * weight;
                integrationTmp[u + 3] += transMatrix[w + OFFSET*3 + stateChild] * partialsParent[v + 3] * weight;

                u += 4;
                v += 4;
            }
            w += OFFSET*4;
            if (kExtraPatterns)
            	v += 4 * kExtraPatterns;
        }

    } else { // Integrate against a partial at the child

        const REALTYPE* partialsChild = gPartials[childIndex];
		#if 0//
        int v = 0;
		#endif
        int w = 0;
        for(int l = 0; l < kCategoryCount; l++) {
            int u = 0;
			#if 1//
			int v = l*kPaddedPatternCount*4;
			#endif
            const REALTYPE weight = wt[l];

            PREFETCH_MATRIX(1,transMatrix,w);

            for(int k = 0; k < kPatternCount; k++) {

                const REALTYPE* partials1 = partialsChild;

                PREFETCH_PARTIALS(1,partials1,v);

                DO_INTEGRATION(1);

                integrationTmp[u    ] += sum10 * partialsParent[v    ] * weight;
                integrationTmp[u + 1] += sum11 * partialsParent[v + 1] * weight;
                integrationTmp[u + 2] += sum12 * partialsParent[v + 2] * weight;
                integrationTmp[u + 3] += sum13 * partialsParent[v + 3] * weight;

                u += 4;
                v += 4;
            }
            w += OFFSET*4;
			#if 0//
            if (kExtraPatterns)
            	v += 4 * kExtraPatterns;
			#endif//
        }
    }

    return integrateOutStatesAndScale(integrationTmp, stateFrequenciesIndex, scalingFactorsIndex, outSumLogLikelihood);
}

BEAGLE_CPU_TEMPLATE
void BeagleCPU4StateImpl<BEAGLE_CPU_GENERIC>::calcEdgeLogLikelihoodsByPartition(
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

        assert(parIndex >= kTipCount);

        const REALTYPE* partialsParent = gPartials[parIndex];
        const REALTYPE* transMatrix = gTransitionMatrices[probIndex];
        const REALTYPE* wt = gCategoryWeights[categoryWeightsIndex];

        if (childIndex < kTipCount && gTipStates[childIndex]) { // Integrate against a state at the child

            const int* statesChild = gTipStates[childIndex];
            int v = startPattern * 4; // Index for parent partials
            int w = 0;
            for(int l = 0; l < kCategoryCount; l++) {
                int u = startPattern * 4; // Index in resulting product-partials (summed over categories)
                const REALTYPE weight = wt[l];
                for(int k = startPattern; k < endPattern; k++) {

                    const int stateChild = statesChild[k];

                    integrationTmp[u    ] += transMatrix[w            + stateChild] * partialsParent[v    ] * weight;
                    integrationTmp[u + 1] += transMatrix[w + OFFSET*1 + stateChild] * partialsParent[v + 1] * weight;
                    integrationTmp[u + 2] += transMatrix[w + OFFSET*2 + stateChild] * partialsParent[v + 2] * weight;
                    integrationTmp[u + 3] += transMatrix[w + OFFSET*3 + stateChild] * partialsParent[v + 3] * weight;

                    u += 4;
                    v += 4;
                }
                w += OFFSET*4;
                if (kExtraPatterns)
                  v += 4 * kExtraPatterns;
                v += ((kPatternCount - endPattern) + startPattern) * 4;
            }
        } else { // Integrate against a partial at the child
            const REALTYPE* partialsChild = gPartials[childIndex];
        #if 0//
            int v = 0;
        #endif
            int w = 0;
            for(int l = 0; l < kCategoryCount; l++) {
                int u = startPattern * 4;
          #if 1//
          int v = l*kPaddedPatternCount*4 + startPattern * 4;
          #endif
                const REALTYPE weight = wt[l];

                PREFETCH_MATRIX(1,transMatrix,w);

                for(int k = startPattern; k < endPattern; k++) {

                    const REALTYPE* partials1 = partialsChild;

                    PREFETCH_PARTIALS(1,partials1,v);

                    DO_INTEGRATION(1);

                    integrationTmp[u    ] += sum10 * partialsParent[v    ] * weight;
                    integrationTmp[u + 1] += sum11 * partialsParent[v + 1] * weight;
                    integrationTmp[u + 2] += sum12 * partialsParent[v + 2] * weight;
                    integrationTmp[u + 3] += sum13 * partialsParent[v + 3] * weight;

                    u += 4;
                    v += 4;
                }
                w += OFFSET*4;
          #if 0//
                if (kExtraPatterns)
                  v += 4 * kExtraPatterns;
                v += ((kPatternCount - endPattern) + startPattern) * 4;
          #endif//
            }
        }
    }

    integrateOutStatesAndScaleByPartition(integrationTmp, stateFrequenciesIndices, cumulativeScaleIndices, partitionIndices, partitionCount, outSumLogLikelihoodByPartition);
}

BEAGLE_CPU_TEMPLATE
int BeagleCPU4StateImpl<BEAGLE_CPU_GENERIC>::calcRootLogLikelihoods(const int bufferIndex,
                                                           const int categoryWeightsIndex,
                                                           const int stateFrequenciesIndex,
                                                const int scalingFactorsIndex,
                                                double* outSumLogLikelihood) {

    const REALTYPE* rootPartials = gPartials[bufferIndex];
    assert(rootPartials);
    const REALTYPE* wt = gCategoryWeights[categoryWeightsIndex];

    int u = 0;
    int v = 0;
    const REALTYPE wt0 = wt[0];
    for (int k = 0; k < kPatternCount; k++) {
        integrationTmp[v    ] = rootPartials[v    ] * wt0;
        integrationTmp[v + 1] = rootPartials[v + 1] * wt0;
        integrationTmp[v + 2] = rootPartials[v + 2] * wt0;
        integrationTmp[v + 3] = rootPartials[v + 3] * wt0;
        v += 4;
    }
    for (int l = 1; l < kCategoryCount; l++) {
        u = 0;
        const REALTYPE wtl = wt[l];
        for (int k = 0; k < kPatternCount; k++) {
            integrationTmp[u    ] += rootPartials[v    ] * wtl;
            integrationTmp[u + 1] += rootPartials[v + 1] * wtl;
            integrationTmp[u + 2] += rootPartials[v + 2] * wtl;
            integrationTmp[u + 3] += rootPartials[v + 3] * wtl;

            u += 4;
            v += 4;
        }
		v += 4 * kExtraPatterns;
    }

    return integrateOutStatesAndScale(integrationTmp, stateFrequenciesIndex, scalingFactorsIndex, outSumLogLikelihood);
}

BEAGLE_CPU_TEMPLATE
void BeagleCPU4StateImpl<BEAGLE_CPU_GENERIC>::calcEdgeLogDerivativesStates(const int *tipStates,
                                                                           const REALTYPE *preOrderPartial,
                                                                           const int firstDerivativeIndex,
                                                                           const int secondDerivativeIndex,
                                                                           const double *categoryRates,
                                                                           const REALTYPE *categoryWeights,
                                                                           double *outDerivatives,
                                                                           double *outSumDerivatives,
                                                                           double *outSumSquaredDerivatives) {

    for (int category = 0; category < kCategoryCount; category++) {

        const REALTYPE *firstDerivMatrix = gTransitionMatrices[firstDerivativeIndex] + category * kMatrixSize;

        for (int pattern = 0; pattern < kPatternCount; pattern++) {

            const int patternIndex = category * kPatternCount + pattern;
            const int localPatternOffset = patternIndex * 4;

            const int state = tipStates[pattern];

            PREFETCH_MATRIX_COLUMN(0, firstDerivMatrix, state);

            REALTYPE numerator =
                    sum00 * preOrderPartial[localPatternOffset] +
                    sum01 * preOrderPartial[localPatternOffset + 1] +
                    sum02 * preOrderPartial[localPatternOffset + 2] +
                    sum03 * preOrderPartial[localPatternOffset + 3];

            REALTYPE denominator = preOrderPartial[localPatternOffset + (state & 3)];

            grandNumeratorDerivTmp[pattern] += categoryWeights[category] * numerator;
            grandDenominatorDerivTmp[pattern] += categoryWeights[category] * denominator;
        }
    }
}

BEAGLE_CPU_TEMPLATE
void BeagleCPU4StateImpl<BEAGLE_CPU_GENERIC>::calcEdgeLogDerivativesPartials(const REALTYPE *postOrderPartial,
                                                                             const REALTYPE *preOrderPartial,
                                                                             const int firstDerivativeIndex,
                                                                             const int secondDerivativeIndex,
                                                                             const double *categoryRates,
                                                                             const REALTYPE *categoryWeights,
                                                                             const int scalingFactorsIndex,
                                                                             double *outDerivatives,
                                                                             double *outSumDerivatives,
                                                                             double *outSumSquaredDerivatives) {

    const REALTYPE* transMatrix = gTransitionMatrices[firstDerivativeIndex];

    int w = 0;
    for(int l = 0; l < kCategoryCount; l++) {

        int v = l*kPaddedPatternCount*4;

        const REALTYPE weight = categoryWeights[l];

        PREFETCH_MATRIX(1,transMatrix,w); // TODO Use _TRANSPOSE and then reverse integration below

        for(int k = 0; k < kPatternCount; k++) {

            PREFETCH_PARTIALS(1, postOrderPartial,v);
            PREFETCH_PARTIALS(0, preOrderPartial, v);


            DO_INTEGRATION(1);

//            grandNumeratorDerivTmp[k] += (sum10 * prePartials[v] + sum11 * prePartials[v + 1]
//                    + sum12 * prePartials[v + 2] + sum13 * prePartials[v + 3]) * weight;
//            grandDenominatorDerivTmp[k] += (postPartials[v] * prePartials[v] + postPartials[v + 1] * prePartials[v + 1]
//                    + postPartials[v + 2] * prePartials[v + 2] + postPartials[v + 3] * prePartials[v + 3]) * weight;

            grandDenominatorDerivTmp[k] += (p10 * p00 + p11 * p01 + p12 * p02 + p13 * p03) * weight;
            grandNumeratorDerivTmp[k] += (sum10 * p00 + sum11 * p01 + sum12 * p02 + sum13 * p03) * weight;

            v += 4;
        }
        w += OFFSET*4;
    }
}

//#define OLD_CP_STATES

BEAGLE_CPU_TEMPLATE
void BeagleCPU4StateImpl<BEAGLE_CPU_GENERIC>::calcCrossProductsStates(const int *tipStates,
                                                                const REALTYPE *preOrderPartial,
                                                                const double *categoryRates,
                                                                const REALTYPE *categoryWeights,
                                                                const double edgeLength,
                                                                double *outCrossProducts,
                                                                double *outSumSquaredDerivatives) {
#ifdef OLD_CP_STATES
    return BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcCrossProductsStates(tipStates, preOrderPartial, categoryRates,
            categoryWeights, edgeLength, outCrossProducts, outSumSquaredDerivatives);
#else
    std::array<REALTYPE, 16> acrossPatterns;
    acrossPatterns.fill((REALTYPE) 0);

    std::array<REALTYPE, 16> withinPattern;

    for (int pattern = 0; pattern < kPatternCount; pattern++) {

        withinPattern.fill((REALTYPE) 0);
        REALTYPE patternDenominator = 0.0;

        const int state = tipStates[pattern];

        if (state < kStateCount) {

            for (int category = 0; category < kCategoryCount; category++) {

                const REALTYPE scale = (REALTYPE) categoryRates[category] * edgeLength;

                const REALTYPE weight = categoryWeights[category];
                const int patternIndex = category * kPatternCount + pattern;
                const int v = patternIndex * 4;

                REALTYPE denominator = preOrderPartial[v + state];
                patternDenominator += denominator * weight;

                PREFETCH_PARTIALS(pre, preOrderPartial, v);

                withinPattern[0 * 4 + state] += ppre0 * weight * scale; // TODO Work with transpose(withinPattern)
                withinPattern[1 * 4 + state] += ppre1 * weight * scale;
                withinPattern[2 * 4 + state] += ppre2 * weight * scale;
                withinPattern[3 * 4 + state] += ppre3 * weight * scale;
            }

            const REALTYPE patternWeight = gPatternWeights[pattern] / patternDenominator;
            acrossPatterns[0 * 4 + state] += withinPattern[0 * 4 + state] * patternWeight;
            acrossPatterns[1 * 4 + state] += withinPattern[1 * 4 + state] * patternWeight;
            acrossPatterns[2 * 4 + state] += withinPattern[2 * 4 + state] * patternWeight;
            acrossPatterns[3 * 4 + state] += withinPattern[3 * 4 + state] * patternWeight;

        } else { // Missing character

            for (int category = 0; category < kCategoryCount; category++) {

                const REALTYPE scale = (REALTYPE) categoryRates[category] * edgeLength;

                const REALTYPE weight = categoryWeights[category];
                const int patternIndex = category * kPatternCount + pattern;
                const int v = patternIndex * 4;

                REALTYPE denominator = 0.0;
                for (int k = 0; k < 4; k++) {
                    denominator += preOrderPartial[v + k];
                }
                patternDenominator += denominator * weight;

                for (int k = 0; k < 4; k++) {
                    for (int j = 0; j < 4; j++) {
                        withinPattern[k * 4 + j] += preOrderPartial[v + k] * weight * scale;
                    }
                }
            }

            const REALTYPE patternWeight = gPatternWeights[pattern] / patternDenominator;
            for (int k = 0; k < 16; k++) {
                acrossPatterns[k] += withinPattern[k] * patternWeight;
            }
        }
    }

    for (int k = 0; k < 16; k++) {
        outCrossProducts[k] += acrossPatterns[k]; // TODO transpose back on store
    }
#endif
}

//#define OLD_CP_PARTIALS

BEAGLE_CPU_TEMPLATE
void BeagleCPU4StateImpl<BEAGLE_CPU_GENERIC>::calcCrossProductsPartials(const REALTYPE *postOrderPartial,
                                                                        const REALTYPE *preOrderPartial,
                                                                        const double *categoryRates,
                                                                        const REALTYPE *categoryWeights,
                                                                        const double edgeLength,
                                                                        double *outCrossProducts,
                                                                        double *outSumSquaredDerivatives) {


#ifdef OLD_CP_PARTIALS
    return BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcCrossProductsPartials(postOrderPartial, preOrderPartial,
            categoryRates, categoryWeights, edgeLength, outCrossProducts, outSumSquaredDerivatives);
#else

    std::array<REALTYPE, 16> acrossPatterns;
    acrossPatterns.fill((REALTYPE) 0.0);

    std::array<REALTYPE, 16> withinPattern;

    for (int pattern = 0; pattern < kPatternCount; pattern++) {

        withinPattern.fill((REALTYPE) 0.0);

        REALTYPE patternDenominator = 0.0;

        for (int category = 0; category < kCategoryCount; category++) {

            const REALTYPE scale = (REALTYPE) categoryRates[category] * edgeLength;
            const REALTYPE weight = categoryWeights[category];
            const int patternIndex = category * kPatternCount + pattern; // Bad memory access
            const int v = patternIndex * 4;

            PREFETCH_PARTIALS(pre, preOrderPartial, v);
            PREFETCH_PARTIALS(post, postOrderPartial, v);

            REALTYPE denominator = INNER_PRODUCT(pre, post);

            patternDenominator += denominator * weight;

            REALTYPE weightScale = weight * scale;

            ppost0 *= weightScale;
            ppost1 *= weightScale;
            ppost2 *= weightScale;
            ppost3 *= weightScale;

            withinPattern[0 * 4 + 0] += ppre0 * ppost0;
            withinPattern[0 * 4 + 1] += ppre0 * ppost1;
            withinPattern[0 * 4 + 2] += ppre0 * ppost2;
            withinPattern[0 * 4 + 3] += ppre0 * ppost3;

            withinPattern[1 * 4 + 0] += ppre1 * ppost0;
            withinPattern[1 * 4 + 1] += ppre1 * ppost1;
            withinPattern[1 * 4 + 2] += ppre1 * ppost2;
            withinPattern[1 * 4 + 3] += ppre1 * ppost3;

            withinPattern[2 * 4 + 0] += ppre2 * ppost0;
            withinPattern[2 * 4 + 1] += ppre2 * ppost1;
            withinPattern[2 * 4 + 2] += ppre2 * ppost2;
            withinPattern[2 * 4 + 3] += ppre2 * ppost3;

            withinPattern[3 * 4 + 0] += ppre3 * ppost0;
            withinPattern[3 * 4 + 1] += ppre3 * ppost1;
            withinPattern[3 * 4 + 2] += ppre3 * ppost2;
            withinPattern[3 * 4 + 3] += ppre3 * ppost3;
        }

        const auto patternWeight =  gPatternWeights[pattern] / patternDenominator;
        for (int k = 0; k < 16; k++) {
            acrossPatterns[k] += withinPattern[k] * patternWeight;
        }
    }

    for (int k = 0; k < 16; k++) {
        outCrossProducts[k] += acrossPatterns[k];
    }
#endif
}

BEAGLE_CPU_TEMPLATE
void BeagleCPU4StateImpl<BEAGLE_CPU_GENERIC>::calcRootLogLikelihoodsByPartition(
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
        assert(rootPartials);
        const REALTYPE* wt = gCategoryWeights[categoryWeightsIndices[p]];

        int v = startPattern * 4;
        const REALTYPE wt0 = wt[0];
        for (int k = startPattern; k < endPattern; k++) {
            integrationTmp[v    ] = rootPartials[v    ] * wt0;
            integrationTmp[v + 1] = rootPartials[v + 1] * wt0;
            integrationTmp[v + 2] = rootPartials[v + 2] * wt0;
            integrationTmp[v + 3] = rootPartials[v + 3] * wt0;
            v += 4;
        }
        for (int l = 1; l < kCategoryCount; l++) {
            int u = startPattern * 4;
            v += ((kPatternCount - endPattern) + startPattern) * 4;
            const REALTYPE wtl = wt[l];
            for (int k = startPattern; k < endPattern; k++) {
                integrationTmp[u    ] += rootPartials[v    ] * wtl;
                integrationTmp[u + 1] += rootPartials[v + 1] * wtl;
                integrationTmp[u + 2] += rootPartials[v + 2] * wtl;
                integrationTmp[u + 3] += rootPartials[v + 3] * wtl;

                u += 4;
                v += 4;
            }
        v += 4 * kExtraPatterns;
        }
    }
    integrateOutStatesAndScaleByPartition(integrationTmp, stateFrequenciesIndices, cumulativeScaleIndices, partitionIndices, partitionCount, outSumLogLikelihoodByPartition);
}

BEAGLE_CPU_TEMPLATE
int BeagleCPU4StateImpl<BEAGLE_CPU_GENERIC>::calcRootLogLikelihoodsMulti(const int* bufferIndices,
                                                                const int* categoryWeightsIndices,
                                                                const int* stateFrequenciesIndices,
                                                                const int* scaleBufferIndices,
                                                                int count,
                                                                double* outSumLogLikelihood) {

    int returnCode = BEAGLE_SUCCESS;

    std::vector<int> indexMaxScale(kPatternCount);
    std::vector<REALTYPE> maxScaleFactor(kPatternCount);

    for (int subsetIndex = 0 ; subsetIndex < count; ++subsetIndex ) {
        const int rootPartialIndex = bufferIndices[subsetIndex];
        const REALTYPE* rootPartials = gPartials[rootPartialIndex];
        const REALTYPE* frequencies = gStateFrequencies[stateFrequenciesIndices[subsetIndex]];
        const REALTYPE* wt = gCategoryWeights[categoryWeightsIndices[subsetIndex]];
        int u = 0;
        int v = 0;

        const REALTYPE wt0 = wt[0];
        for (int k = 0; k < kPatternCount; k++) {
            integrationTmp[v    ] = rootPartials[v    ] * wt0;
            integrationTmp[v + 1] = rootPartials[v + 1] * wt0;
            integrationTmp[v + 2] = rootPartials[v + 2] * wt0;
            integrationTmp[v + 3] = rootPartials[v + 3] * wt0;
            v += 4;
        }
        for (int l = 1; l < kCategoryCount; l++) {
            u = 0;
            const REALTYPE wtl = wt[l];
            for (int k = 0; k < kPatternCount; k++) {
                integrationTmp[u    ] += rootPartials[v    ] * wtl;
                integrationTmp[u + 1] += rootPartials[v + 1] * wtl;
                integrationTmp[u + 2] += rootPartials[v + 2] * wtl;
                integrationTmp[u + 3] += rootPartials[v + 3] * wtl;

                u += 4;
                v += 4;
            }
            v += 4 * kExtraPatterns;
        }

        REALTYPE freq0, freq1, freq2, freq3;
        freq0 = frequencies[0];
        freq1 = frequencies[1];
        freq2 = frequencies[2];
        freq3 = frequencies[3];

        u = 0;
        for (int k = 0; k < kPatternCount; k++) {
            REALTYPE sum =
                freq0 * integrationTmp[u    ] +
                freq1 * integrationTmp[u + 1] +
                freq2 * integrationTmp[u + 2] +
                freq3 * integrationTmp[u + 3];

            u += 4;

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
const char* BeagleCPU4StateImpl<BEAGLE_CPU_GENERIC>::getName() {
	return getBeagleCPU4StateName<BEAGLE_CPU_FACTORY_GENERIC>();
}

//        template<typename REALTYPE, int T_PAD, int P_PAD>
//        void BeagleCPU4StateImpl<REALTYPE, T_PAD, P_PAD>::calcCrossProductsPartials(const REALTYPE *postOrderPartial,
//                                                                                    const REALTYPE *preOrderPartial,
//                                                                                    const double *categoryRates,
//                                                                                    const REALTYPE *categoryWeights,
//                                                                                    const double edgeLength,
//                                                                                    double *outCrossProducts,
//                                                                                    double *outSumSquaredDerivatives) {
//            BeagleCPUImpl::calcCrossProductsPartials(postOrderPartial, preOrderPartial, categoryRates, categoryWeights,
//                                                     edgeLength, outCrossProducts, outSumSquaredDerivatives);
//        }

///////////////////////////////////////////////////////////////////////////////
// BeagleCPUImplFactory public methods

BEAGLE_CPU_FACTORY_TEMPLATE
BeagleImpl* BeagleCPU4StateImplFactory<BEAGLE_CPU_FACTORY_GENERIC>::createImpl(int tipCount,
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

    if (stateCount != 4) {
        return NULL;
    }

    BeagleImpl* impl = new BeagleCPU4StateImpl<REALTYPE, T_PAD_DEFAULT, P_PAD_DEFAULT>();

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
const char* BeagleCPU4StateImplFactory<BEAGLE_CPU_FACTORY_GENERIC>::getName() {
	return getBeagleCPU4StateName<BEAGLE_CPU_FACTORY_GENERIC>();
}

BEAGLE_CPU_FACTORY_TEMPLATE
const long BeagleCPU4StateImplFactory<BEAGLE_CPU_FACTORY_GENERIC>::getFlags() {
    long flags =  BEAGLE_FLAG_COMPUTATION_SYNCH |
                  BEAGLE_FLAG_SCALING_MANUAL | BEAGLE_FLAG_SCALING_ALWAYS | BEAGLE_FLAG_SCALING_AUTO |
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

}	// namespace cpu
}	// namespace beagle

#endif // BEAGLE_CPU_4STATE_IMPL_HPP

