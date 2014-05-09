/*
 *
 * Copyright 2009 Phylogenetic Likelihood Working Group
 *
 * This file is part of BEAGLE.
 *
 * BEAGLE is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 *
 * BEAGLE is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with BEAGLE.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @author Marc Suchard
 * @author Daniel Ayres
 * @author Andrew Rambaut
 */

#define STATE_COUNT 4

#ifdef CUDA
    #include "libhmsbeagle/GPU/GPUImplDefs.h"
    #include "libhmsbeagle/GPU/kernels/kernelsAll.cu" // This file includes the non-state-count specific kernels
    extern "C" {
#endif 

#define multBy4(x)  ((x) << 2)
#define multBy16(x) ((x) << 4)
#define LIKE_PATTERN_BLOCK_SIZE PATTERN_BLOCK_SIZE

///////////////////////////////////////////////////////////////////////////////
// kernel macros CPU

#define DETERMINE_INDICES_4_CPU()\
    int patIdx = KW_LOCAL_ID_0;\
    int matrix = KW_GROUP_ID_1;\
    int pattern = KW_GROUP_ID_0 * KW_LOCAL_SIZE_0 + patIdx;\
    int deltaPartialsByState = pattern * PADDED_STATE_COUNT;\
    int deltaPartialsByMatrix = matrix * PADDED_STATE_COUNT * totalPatterns;\
    int deltaMatrix = matrix * PADDED_STATE_COUNT * PADDED_STATE_COUNT;\
    int deltaPartials = deltaPartialsByMatrix + deltaPartialsByState;
    
#define SUM_PARTIALS_PARTIALS_4_CPU()\
    REAL sum1[PADDED_STATE_COUNT];\
    REAL sum2[PADDED_STATE_COUNT];\
    KW_GLOBAL_VAR REAL* KW_RESTRICT sMatrix1 = matrices1 + deltaMatrix;\
    KW_GLOBAL_VAR REAL* KW_RESTRICT sMatrix2 = matrices2 + deltaMatrix;\
    KW_GLOBAL_VAR REAL* KW_RESTRICT sPartials1 = partials1 + deltaPartials;\
    KW_GLOBAL_VAR REAL* KW_RESTRICT sPartials2 = partials2 + deltaPartials;\
    for(int i = 0; i < PADDED_STATE_COUNT; i++) {\
        sum1[i] = sMatrix1[0 * PADDED_STATE_COUNT + i] * sPartials1[0];\
        sum2[i] = sMatrix2[0 * PADDED_STATE_COUNT + i] * sPartials2[0];\
    }\
    for (int s = 1; s < PADDED_STATE_COUNT; s++) {\
        for(int i = 0; i < PADDED_STATE_COUNT; i++) {\
            FMA(sMatrix1[s * PADDED_STATE_COUNT + i],  sPartials1[s], sum1[i]);\
            FMA(sMatrix2[s * PADDED_STATE_COUNT + i],  sPartials2[s], sum2[i]);\
        }\
    }

#define SUM_STATES_PARTIALS_4_CPU()\
    REAL sum1[PADDED_STATE_COUNT];\
    REAL sum2[PADDED_STATE_COUNT];\
    KW_GLOBAL_VAR REAL* KW_RESTRICT sMatrix1 = matrices1 + deltaMatrix;\
    KW_GLOBAL_VAR REAL* KW_RESTRICT sMatrix2 = matrices2 + deltaMatrix;\
    KW_GLOBAL_VAR REAL* KW_RESTRICT sPartials2 = partials2 + deltaPartials;\
    int state1 = states1[pattern];\
    if (state1 < PADDED_STATE_COUNT) {\
        for(int i = 0; i < PADDED_STATE_COUNT; i++) {\
            sum1[i] = sMatrix1[state1 * PADDED_STATE_COUNT + i];\
        }\
    } else {\
        for(int i = 0; i < PADDED_STATE_COUNT; i++) {\
            sum1[i] = 1.0;\
        }\
    }\
    for(int i = 0; i < PADDED_STATE_COUNT; i++) {\
        sum2[i] = sMatrix2[0 * PADDED_STATE_COUNT + i] * sPartials2[0];\
    }\
    for (int s = 1; s < PADDED_STATE_COUNT; s++) {\
        for(int i = 0; i < PADDED_STATE_COUNT; i++) {\
            FMA(sMatrix2[s * PADDED_STATE_COUNT + i],  sPartials2[s], sum2[i]);\
        }\
    }

#define SUM_PARTIALS_SINGLE_4_CPU()\
    REAL sum1[PADDED_STATE_COUNT];\
    KW_GLOBAL_VAR REAL* KW_RESTRICT sMatrix1 = matrices1 + deltaMatrix;\
    KW_GLOBAL_VAR REAL* KW_RESTRICT sPartials1 = partials1 + deltaPartials;\
    KW_GLOBAL_VAR REAL* KW_RESTRICT sPartials2 = partials2 + deltaPartials;\
    for(int i = 0; i < PADDED_STATE_COUNT; i++) {\
        sum1[i] = sMatrix1[0 * PADDED_STATE_COUNT + i] * sPartials1[0];\
    }\
    for (int s = 1; s < PADDED_STATE_COUNT; s++) {\
        for(int i = 0; i < PADDED_STATE_COUNT; i++) {\
            FMA(sMatrix1[s * PADDED_STATE_COUNT + i],  sPartials1[s], sum1[i]);\
        }\
    }

#define SUM_STATES_SINGLE_4_CPU()\
    REAL sum1[PADDED_STATE_COUNT];\
    KW_GLOBAL_VAR REAL* KW_RESTRICT sMatrix1 = matrices1 + deltaMatrix;\
    KW_GLOBAL_VAR REAL* KW_RESTRICT sPartials2 = partials2 + deltaPartials;\
    int state1 = dChildStates[pattern];\
    if (state1 < PADDED_STATE_COUNT) {\
        for(int i = 0; i < PADDED_STATE_COUNT; i++) {\
            sum1[i] = sMatrix1[state1 * PADDED_STATE_COUNT + i];\
        }\
    } else {\
        for(int i = 0; i < PADDED_STATE_COUNT; i++) {\
            sum1[i] = 1.0;\
        }\
    }

#define SUM_PARTIALS_DERIV_4_CPU()\
    REAL sum1[PADDED_STATE_COUNT];\
    REAL sumFirstDeriv[PADDED_STATE_COUNT], sumSecondDeriv[PADDED_STATE_COUNT];\
    KW_GLOBAL_VAR REAL* KW_RESTRICT sMatrix1 = matrices1 + deltaMatrix;\
    KW_GLOBAL_VAR REAL* KW_RESTRICT sMatrixFirstDeriv = dFirstDerivMatrix + deltaMatrix;\
    KW_GLOBAL_VAR REAL* KW_RESTRICT sMatrixSecondDeriv = dSecondDerivMatrix + deltaMatrix;\
    KW_GLOBAL_VAR REAL* KW_RESTRICT sPartials1 = partials1 + deltaPartials;\
    KW_GLOBAL_VAR REAL* KW_RESTRICT sPartials2 = partials2 + deltaPartials;\
    for(int i = 0; i < PADDED_STATE_COUNT; i++) {\
        sum1[i]           = sMatrix1[          0 * PADDED_STATE_COUNT + i] * sPartials1[0];\
        sumFirstDeriv[i]  = sMatrixFirstDeriv[ 0 * PADDED_STATE_COUNT + i] * sPartials1[0];\
        sumSecondDeriv[i] = sMatrixSecondDeriv[0 * PADDED_STATE_COUNT + i] * sPartials1[0];\
    }\
    for (int s = 1; s < PADDED_STATE_COUNT; s++) {\
        for(int i = 0; i < PADDED_STATE_COUNT; i++) {\
            FMA(sMatrix1[          s * PADDED_STATE_COUNT + i],  sPartials1[s], sum1[i]);\
            FMA(sMatrixFirstDeriv[ s * PADDED_STATE_COUNT + i],  sPartials1[s], sumFirstDeriv[i]);\
            FMA(sMatrixSecondDeriv[s * PADDED_STATE_COUNT + i],  sPartials1[s], sumSecondDeriv[i]);\
        }\
    }

#define SUM_STATES_DERIV_4_CPU()\
    REAL sum1[PADDED_STATE_COUNT];\
    REAL sumFirstDeriv[PADDED_STATE_COUNT], sumSecondDeriv[PADDED_STATE_COUNT];\
    KW_GLOBAL_VAR REAL* KW_RESTRICT sMatrix1 = matrices1 + deltaMatrix;\
    KW_GLOBAL_VAR REAL* KW_RESTRICT sMatrixFirstDeriv = dFirstDerivMatrix + deltaMatrix;\
    KW_GLOBAL_VAR REAL* KW_RESTRICT sMatrixSecondDeriv = dSecondDerivMatrix + deltaMatrix;\
    KW_GLOBAL_VAR REAL* KW_RESTRICT sPartials2 = partials2 + deltaPartials;\
    int state1 = dChildStates[pattern];\
    if (state1 < PADDED_STATE_COUNT) {\
        for(int i = 0; i < PADDED_STATE_COUNT; i++) {\
            sum1[i] =           sMatrix1[          state1 * PADDED_STATE_COUNT + i];\
            sumFirstDeriv[i]  = sMatrixFirstDeriv[ state1 * PADDED_STATE_COUNT + i];\
            sumSecondDeriv[i] = sMatrixSecondDeriv[state1 * PADDED_STATE_COUNT + i];\
        }\
    } else {\
        for(int i = 0; i < PADDED_STATE_COUNT; i++) {\
            sum1[i] = 1.0;\
            sumFirstDeriv[i] = 0.0;\
            sumSecondDeriv[i] = 0.0;\
        }\
    }

#define FIND_MAX_PARTIALS_4_CPU()\
    int patIdx = KW_LOCAL_ID_0;\
    int pattern = KW_GROUP_ID_0 * PATTERN_BLOCK_SIZE + patIdx;\
    int deltaPartialsByState = pattern * PADDED_STATE_COUNT;\
    REAL maxPartial = 0;\
    for(int m = 0; m < matrixCount; m++) {\
        int deltaPartialsByMatrix = m * PADDED_STATE_COUNT * PATTERN_BLOCK_SIZE * KW_NUM_GROUPS_0;\
        int deltaPartials = deltaPartialsByMatrix + deltaPartialsByState;\
        for(int i = 0; i < PADDED_STATE_COUNT; i++) {\
            REAL iPartial = allPartials[deltaPartials + i];\
            if (iPartial > maxPartial)\
                maxPartial = iPartial;\
        }\
    }

#define SCALE_PARTIALS_4_CPU()\
    for(int m = 0; m < matrixCount; m++) {\
        int deltaPartialsByMatrix = m * PADDED_STATE_COUNT * PATTERN_BLOCK_SIZE * KW_NUM_GROUPS_0;\
        int deltaPartials = deltaPartialsByMatrix + deltaPartialsByState;\
        for(int i = 0; i < PADDED_STATE_COUNT; i++) {\
            allPartials[deltaPartials + i] /= maxPartial;\
        }\
    }

#define INTEGRATE_PARTIALS_4_CPU()\
    int pat = KW_LOCAL_ID_0;\
    int pattern = KW_GROUP_ID_0 * LIKE_PATTERN_BLOCK_SIZE + pat;\
    int u = pattern * PADDED_STATE_COUNT;\
    int delta = patternCount * PADDED_STATE_COUNT;\
    REAL sum[PADDED_STATE_COUNT];\
    REAL sumTotal = 0;\
    for(int i = 0; i < PADDED_STATE_COUNT; i++) {\
        sum[i] = dRootPartials[i + u] * dWeights[0];\
    }\
    for(int r = 1; r < matrixCount; r++) {\
        for(int i = 0; i < PADDED_STATE_COUNT; i++) {\
            FMA(dRootPartials[i + u + delta * r],  dWeights[r], sum[i]);\
        }\
    }\
    for(int i = 0; i < PADDED_STATE_COUNT; i++) {\
        sum[i] *= dFrequencies[i];\
        sumTotal += sum[i];\
    }

#define INTEGRATE_PARTIALS_DERIV_4_CPU()\
    int pat = KW_LOCAL_ID_0;\
    int pattern = KW_GROUP_ID_0 * LIKE_PATTERN_BLOCK_SIZE + pat;\
    int u = pattern * PADDED_STATE_COUNT;\
    int delta = patternCount * PADDED_STATE_COUNT;\
    REAL sum[PADDED_STATE_COUNT];\
    REAL sumD1[PADDED_STATE_COUNT], sumD2[PADDED_STATE_COUNT];\
    REAL sumTotal = 0, sumTotalD1 = 0, sumTotalD2 = 0;\
    REAL tmpLogLike, tmpFirstDeriv;\
    for(int i = 0; i < PADDED_STATE_COUNT; i++) {\
        sum[i]   = dRootPartials[   i + u] * dWeights[0];\
        sumD1[i] = dRootFirstDeriv[ i + u] * dWeights[0];\
        sumD2[i] = dRootSecondDeriv[i + u] * dWeights[0];\
    }\
    for(int r = 1; r < matrixCount; r++) {\
        for(int i = 0; i < PADDED_STATE_COUNT; i++) {\
            FMA(dRootPartials[   i + u + delta * r],  dWeights[r], sum[i]);\
            FMA(dRootFirstDeriv[ i + u + delta * r],  dWeights[r], sumD1[i]);\
            FMA(dRootSecondDeriv[i + u + delta * r],  dWeights[r], sumD2[i]);\
        }\
    }\
    for(int i = 0; i < PADDED_STATE_COUNT; i++) {\
        sum[i]     *= dFrequencies[i];\
        sumD1[i]   *= dFrequencies[i];\
        sumD2[i]   *= dFrequencies[i];\
        sumTotal   += sum[i];\
        sumTotalD1 += sumD1[i];\
        sumTotalD2 += sumD2[i];\
    }

///////////////////////////////////////////////////////////////////////////////
// kernel macros GPU

// Do not use | (instead of +) for any term involing PATTERN_BLOCK_SIZE
// as this should be adjustable
#define DETERMINE_INDICES_4_GPU()\
    int tx = KW_LOCAL_ID_0;\
    int state = tx & 0x3;\
    int pat = tx >> 2;\
    int patIdx = KW_LOCAL_ID_1;\
    int matrix = KW_GROUP_ID_1;\
    int pattern = __umul24(KW_GROUP_ID_0, PATTERN_BLOCK_SIZE * 4) + multBy4(patIdx) + pat;\
    int deltaPartialsByState = multBy16(KW_GROUP_ID_0 * PATTERN_BLOCK_SIZE + patIdx);\
    int deltaPartialsByMatrix = __umul24(matrix, multBy4(totalPatterns));\
    int x2 = multBy16(matrix);\
    int u = tx + deltaPartialsByState + deltaPartialsByMatrix;

#define LOAD_PARTIALS_PARTIALS_4_GPU()\
    int y = deltaPartialsByState + deltaPartialsByMatrix;\
    KW_LOCAL_MEM REAL sPartials1[PATTERN_BLOCK_SIZE * 4 * 4];\
    KW_LOCAL_MEM REAL sPartials2[PATTERN_BLOCK_SIZE * 4 * 4];\
    /* copy PADDED_STATE_COUNT * PATTERN_BLOCK_SIZE lengthed partials*/\
    if (pattern < totalPatterns) {\
        sPartials1[multBy16(patIdx) | tx] = partials1[y | tx]; /*All coalesced memory*/\
        sPartials2[multBy16(patIdx) | tx] = partials2[y | tx];\
    } else {\
        sPartials1[multBy16(patIdx) | tx] = 0;\
        sPartials2[multBy16(patIdx) | tx] = 0;\
    }

#define LOAD_PARTIALS_SINGLE_4_GPU()\
    int y = deltaPartialsByState + deltaPartialsByMatrix;\
    KW_LOCAL_MEM REAL sPartials2[PATTERN_BLOCK_SIZE * 4 * 4];\
    if (pattern < totalPatterns) {\
        sPartials2[multBy16(patIdx) | tx] = partials2[y | tx];\
    } else {\
        sPartials2[multBy16(patIdx) | tx] = 0;\
    }

#define LOAD_SCALING_4_GPU()\
    KW_LOCAL_MEM REAL fixedScalingFactors[PATTERN_BLOCK_SIZE * 4];\
    if (patIdx < 4) { /* need to load 4*PATTERN_BLOCK_SIZE factors for this block*/\
        fixedScalingFactors[patIdx * PATTERN_BLOCK_SIZE + tx] = \
            scalingFactors[KW_GROUP_ID_0 * PATTERN_BLOCK_SIZE * 4 + patIdx * PATTERN_BLOCK_SIZE + tx];\
    }

#define LOAD_MATRIX_4_GPU()\
    KW_GLOBAL_VAR REAL* KW_RESTRICT matrix1 = matrices1 + x2; /*Points to *this* matrix*/\
    KW_GLOBAL_VAR REAL* KW_RESTRICT matrix2 = matrices2 + x2;\
    KW_LOCAL_MEM REAL sMatrix1[16]; /*Load values into shared memory*/\
    KW_LOCAL_MEM REAL sMatrix2[16];\
    if (patIdx == 0 ) {\
        sMatrix1[tx] = matrix1[tx]; /*All coalesced memory reads*/\
        sMatrix2[tx] = matrix2[tx];\
    }\
    KW_LOCAL_FENCE;
    
#define LOAD_MATRIX_SINGLE_4_GPU()\
    KW_GLOBAL_VAR REAL* KW_RESTRICT matrix1 = matrices1 + x2;\
    KW_LOCAL_MEM REAL sMatrix1[16];\
    if (patIdx == 0 ) {\
        sMatrix1[tx] = matrix1[tx];\
    }\
    KW_LOCAL_FENCE;

#define LOAD_MATRIX_DERIV_4_GPU()\
    KW_GLOBAL_VAR REAL* KW_RESTRICT matrix1 = matrices1 + x2;\
    KW_GLOBAL_VAR REAL* KW_RESTRICT matrixFirstDeriv = dFirstDerivMatrix + x2;\
    KW_GLOBAL_VAR REAL* KW_RESTRICT matrixSecondDeriv = dSecondDerivMatrix + x2;\
    KW_LOCAL_MEM REAL sMatrix1[16];\
    KW_LOCAL_MEM REAL sMatrixFirstDeriv[16];\
    KW_LOCAL_MEM REAL sMatrixSecondDeriv[16];\
    if (patIdx == 0) {\
        sMatrix1[tx] = matrix1[tx];\
        sMatrixFirstDeriv[tx] = matrixFirstDeriv[tx];\
        sMatrixSecondDeriv[tx] = matrixSecondDeriv[tx];\
    }\
    KW_LOCAL_FENCE;

#define SUM_PARTIALS_PARTIALS_4_GPU()\
    REAL sum1, sum2;\
    int i = pat;\
    int patIdx16pat4 = multBy16(patIdx) | (tx & 0xC);\
    sum1 = sMatrix1[multBy4(i) | state] * sPartials1[patIdx16pat4 | i];\
    sum2 = sMatrix2[multBy4(i) | state] * sPartials2[patIdx16pat4 | i];\
    i = (++i) & 0x3;\
    FMA(   sMatrix1[multBy4(i) | state],  sPartials1[patIdx16pat4 | i], sum1);\
    FMA(   sMatrix2[multBy4(i) | state],  sPartials2[patIdx16pat4 | i], sum2);\
    i = (++i) & 0x3;\
    FMA(   sMatrix1[multBy4(i) | state],  sPartials1[patIdx16pat4 | i], sum1);\
    FMA(   sMatrix2[multBy4(i) | state],  sPartials2[patIdx16pat4 | i], sum2);\
    i = (++i) & 0x3;\
    FMA(   sMatrix1[multBy4(i) | state],  sPartials1[patIdx16pat4 | i], sum1);\
    FMA(   sMatrix2[multBy4(i) | state],  sPartials2[patIdx16pat4 | i], sum2);

#define SUM_STATES_PARTIALS_4_GPU()\
    REAL sum1 = 1, sum2;\
    int state1 = states1[pattern];\
    if (state1 < PADDED_STATE_COUNT)\
        sum1 = sMatrix1[state1 * 4 + state];\
    int i = pat;\
    int patIdx16pat4 = multBy16(patIdx) | (tx & 0xC);\
    sum2  = sMatrix2[multBy4(i) | state] * sPartials2[patIdx16pat4 | i];\
    i = (++i) & 0x3;\
    FMA(    sMatrix2[multBy4(i) | state],  sPartials2[patIdx16pat4 | i], sum2);\
    i = (++i) & 0x3;\
    FMA(    sMatrix2[multBy4(i) | state],  sPartials2[patIdx16pat4 | i], sum2);\
    i = (++i) & 0x3;\
    FMA(    sMatrix2[multBy4(i) | state],  sPartials2[patIdx16pat4 | i], sum2);

#define SUM_PARTIALS_SINGLE_4_GPU()\
    REAL sum1;\
    int i = pat;\
    int patIdx16pat4 = multBy16(patIdx) | (tx & 0xC);\
    sum1  = sMatrix1[multBy4(i) | state] * sPartials1[patIdx16pat4 | i];\
    i = (++i) & 0x3;\
    FMA(    sMatrix1[multBy4(i) | state],  sPartials1[patIdx16pat4 | i], sum1);\
    i = (++i) & 0x3;\
    FMA(    sMatrix1[multBy4(i) | state],  sPartials1[patIdx16pat4 | i], sum1);\
    i = (++i) & 0x3;\
    FMA(    sMatrix1[multBy4(i) | state],  sPartials1[patIdx16pat4 | i], sum1);

#define SUM_STATES_SINGLE_4_GPU()\
    REAL sum1;\
    int state1 = dChildStates[pattern];\
    if (state1 < PADDED_STATE_COUNT)\
        sum1 = sMatrix1[state1 * 4 + state];\
    else\
        sum1 = 1.0;

#define SUM_PARTIALS_DERIV_4_GPU()\
    REAL sum1, sumFirstDeriv, sumSecondDeriv;\
    int i = pat;\
    int patIdx16pat4 = multBy16(patIdx) | (tx & 0xC);\
    sum1           = sMatrix1[          multBy4(i) | state] * sPartials1[patIdx16pat4 | i];\
    sumFirstDeriv  = sMatrixFirstDeriv[ multBy4(i) | state] * sPartials1[patIdx16pat4 | i];\
    sumSecondDeriv = sMatrixSecondDeriv[multBy4(i) | state] * sPartials1[patIdx16pat4 | i];\
    i = (++i) & 0x3;\
    FMA(sMatrix1[          multBy4(i) | state], sPartials1[patIdx16pat4 | i], sum1);\
    FMA(sMatrixFirstDeriv[ multBy4(i) | state], sPartials1[patIdx16pat4 | i], sumFirstDeriv);\
    FMA(sMatrixSecondDeriv[multBy4(i) | state], sPartials1[patIdx16pat4 | i], sumSecondDeriv);\
    i = (++i) & 0x3;\
    FMA(sMatrix1[          multBy4(i) | state], sPartials1[patIdx16pat4 | i], sum1);\
    FMA(sMatrixFirstDeriv[ multBy4(i) | state], sPartials1[patIdx16pat4 | i], sumFirstDeriv);\
    FMA(sMatrixSecondDeriv[multBy4(i) | state], sPartials1[patIdx16pat4 | i], sumSecondDeriv);\
    i = (++i) & 0x3;\
    FMA(sMatrix1[          multBy4(i) | state], sPartials1[patIdx16pat4 | i], sum1);\
    FMA(sMatrixFirstDeriv[ multBy4(i) | state], sPartials1[patIdx16pat4 | i], sumFirstDeriv);\
    FMA(sMatrixSecondDeriv[multBy4(i) | state], sPartials1[patIdx16pat4 | i], sumSecondDeriv);

#define SUM_STATES_DERIV_4_GPU()\
    REAL sum1, sumFirstDeriv, sumSecondDeriv;\
    int state1 = dChildStates[pattern];\
    if (state1 < PADDED_STATE_COUNT) {\
        sum1 = sMatrix1[state1 * 4 + state];\
        sumFirstDeriv = sMatrixFirstDeriv[state1 * 4 + state];\
        sumSecondDeriv = sMatrixSecondDeriv[state1 * 4 + state];\
    } else {\
        sum1 = 1.0;\
        sumFirstDeriv = 0.0;\
        sumSecondDeriv = 0.0;\
    }

#define FIND_MAX_PARTIALS_STATE_4_GPU()\
    int tx = KW_LOCAL_ID_0;\
    int state = tx & 0x3;\
    int pat = tx >> 2;\
    int patIdx = KW_GROUP_ID_0;\
    int pattern = (patIdx << 2) + pat;\
    int matrix = KW_LOCAL_ID_1;\
    /* TODO: Assumes matrixCount < MATRIX_BLOCK_SIZE*/\
    /* Patterns are always padded, so no reading/writing past end possible*/\
    /* Find start of patternBlock for thread-block*/\
    int partialsOffset = (matrix * KW_NUM_GROUPS_0 + patIdx) << 4; /* 16;*/\
    KW_LOCAL_MEM REAL partials[MATRIX_BLOCK_SIZE][16]; /* 4 patterns at a time*/\
    KW_LOCAL_MEM REAL storedPartials[MATRIX_BLOCK_SIZE][16];\
    KW_LOCAL_MEM REAL matrixMax[4];\
    if (matrix < matrixCount)\
        partials[matrix][tx] = allPartials[partialsOffset + tx];          \
    storedPartials[matrix][tx] = partials[matrix][tx];\
    KW_LOCAL_FENCE;\
    /* Unrolled parallel max-reduction*/\
    if (state < 2) {\
        REAL compare1 = partials[matrix][tx];\
        REAL compare2 = partials[matrix][tx + 2];\
        if (compare2 > compare1)\
            partials[matrix][tx] = compare2;\
    }\
    KW_LOCAL_FENCE;\
    if (state < 1) {\
        REAL compare1 = partials[matrix][tx];\
        REAL compare2 = partials[matrix][tx + 1];\
        if (compare2 > compare1)\
            partials[matrix][tx] = compare2;\
    }\
    KW_LOCAL_FENCE;

#define FIND_MAX_PARTIALS_MATRIX_4_GPU()\
    matrixMax[pat] = 0;\
    for(int m = 0; m < matrixCount; m++) {\
        if (partials[m][tx] > matrixMax[pat])\
            matrixMax[pat] = partials[m][tx];\
    }

#define SCALE_PARTIALS_4_GPU()\
    KW_LOCAL_FENCE;\
    if (matrix < matrixCount)\
        allPartials[partialsOffset + tx] = storedPartials[matrix][tx] / matrixMax[pat];

#define INTEGRATE_PARTIALS_4_GPU()\
    int state = KW_LOCAL_ID_0;\
    int pat = KW_LOCAL_ID_1;\
    int pattern = KW_GROUP_ID_0 * LIKE_PATTERN_BLOCK_SIZE + KW_LOCAL_ID_1;\
    int u = state + pattern * PADDED_STATE_COUNT;\
    int delta = patternCount * PADDED_STATE_COUNT;\
    KW_LOCAL_MEM REAL stateFreq[4];\
    /* TODO: Currently assumes MATRIX_BLOCK_SIZE >= matrixCount */\
    KW_LOCAL_MEM REAL matrixProp[MATRIX_BLOCK_SIZE];\
    KW_LOCAL_MEM REAL sum[LIKE_PATTERN_BLOCK_SIZE][4];\
    /* Load shared memory */\
    if (pat == 0) {\
        stateFreq[state] = dFrequencies[state];\
    }\
    sum[pat][state] = 0;\
    /* TODO: Assumes matrixCount < LIKE_PATTERN_BLOCK_SIZE * 4 */\
    if (pat * 4 + state < matrixCount) {\
        matrixProp[pat * 4 + state] = dWeights[pat * 4 + state];\
    }\
    KW_LOCAL_FENCE;\
    for(int r = 0; r < matrixCount; r++) {\
        FMA(dRootPartials[u + delta * r], matrixProp[r], sum[pat][state]);\
    }\
    sum[pat][state] *= stateFreq[state];\
    KW_LOCAL_FENCE;\
    if (state < 2)\
        sum[pat][state] += sum[pat][state + 2];\
    KW_LOCAL_FENCE;\
    if (state < 1) {\
        sum[pat][state] += sum[pat][state + 1];\
    }

#define INTEGRATE_PARTIALS_DERIV_4_GPU()\
    int state = KW_LOCAL_ID_0;\
    int pat = KW_LOCAL_ID_1;\
    int pattern = KW_GROUP_ID_0 * LIKE_PATTERN_BLOCK_SIZE + KW_LOCAL_ID_1;\
    REAL tmpLogLike, tmpFirstDeriv;\
    KW_LOCAL_MEM REAL stateFreq[4];\
    /* TODO: Currently assumes MATRIX_BLOCK_SIZE >= matrixCount */\
    KW_LOCAL_MEM REAL matrixProp[MATRIX_BLOCK_SIZE];\
    KW_LOCAL_MEM REAL sum[LIKE_PATTERN_BLOCK_SIZE][4];\
    KW_LOCAL_MEM REAL sumD1[LIKE_PATTERN_BLOCK_SIZE][4];\
    KW_LOCAL_MEM REAL sumD2[LIKE_PATTERN_BLOCK_SIZE][4];\
    /* Load shared memory */\
    if (pat == 0) {\
        stateFreq[state] = dFrequencies[state];\
    }\
    sum[pat][state]   = 0;\
    sumD1[pat][state] = 0;\
    sumD2[pat][state] = 0;\
    /* TODO: Assumes matrixCount < LIKE_PATTERN_BLOCK_SIZE * 4 */\
    if (pat * 4 + state < matrixCount) {\
        matrixProp[pat * 4 + state] = dWeights[pat * 4 + state];\
    }\
    KW_LOCAL_FENCE;\
    int u = state + pattern * PADDED_STATE_COUNT;\
    int delta = patternCount * PADDED_STATE_COUNT;\
    for(int r = 0; r < matrixCount; r++) {\
        FMA(dRootPartials[u + delta * r]   , matrixProp[r], sum[pat][state]);\
        FMA(dRootFirstDeriv[u + delta * r] , matrixProp[r], sumD1[pat][state]);\
        FMA(dRootSecondDeriv[u + delta * r], matrixProp[r], sumD2[pat][state]);\
    }\
    sum[pat][state]   *= stateFreq[state];\
    sumD1[pat][state] *= stateFreq[state];\
    sumD2[pat][state] *= stateFreq[state];\
    KW_LOCAL_FENCE;\
    if (state < 2) {\
        sum[pat][state]   += sum[pat][state + 2];\
        sumD1[pat][state] += sumD1[pat][state + 2];\
        sumD2[pat][state] += sumD2[pat][state + 2];\
    }\
    KW_LOCAL_FENCE;\
    if (state < 1) {\
        sum[pat][state]   += sum[pat][state + 1];\
        sumD1[pat][state] += sumD1[pat][state + 1];\
        sumD2[pat][state] += sumD2[pat][state + 1];\
    }

///////////////////////////////////////////////////////////////////////////////

KW_GLOBAL_KERNEL void kernelPartialsPartialsNoScale(KW_GLOBAL_VAR REAL* KW_RESTRICT partials1,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT partials2,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT partials3,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT matrices1,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT matrices2,
                                                    int totalPatterns) {
#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    DETERMINE_INDICES_4_CPU()
    SUM_PARTIALS_PARTIALS_4_CPU();
    for(int i = 0; i < PADDED_STATE_COUNT; i++) {
        partials3[deltaPartials + i] = sum1[i] * sum2[i];
    }
#else // GPU implementation
    DETERMINE_INDICES_4_GPU();
    LOAD_PARTIALS_PARTIALS_4_GPU();
    LOAD_MATRIX_4_GPU();
    if (pattern < totalPatterns) { // Remove padded threads!
        SUM_PARTIALS_PARTIALS_4_GPU();
        partials3[u] = sum1 * sum2;
    }
#endif // FW_OPENCL_CPU

#ifdef KERNEL_PRINT_ENABLED
    printf("matrix = %d, pat = %d for tx = %d and state = %d :  u = %d\n",
           matrix, pattern, tx, state, u);
#endif
}

KW_GLOBAL_KERNEL void kernelPartialsPartialsFixedScale(KW_GLOBAL_VAR REAL* KW_RESTRICT partials1,
                                                       KW_GLOBAL_VAR REAL* KW_RESTRICT partials2,
                                                       KW_GLOBAL_VAR REAL* KW_RESTRICT partials3,
                                                       KW_GLOBAL_VAR REAL* KW_RESTRICT matrices1,
                                                       KW_GLOBAL_VAR REAL* KW_RESTRICT matrices2,
                                                       KW_GLOBAL_VAR REAL* KW_RESTRICT scalingFactors,
                                                       int totalPatterns) {
#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    DETERMINE_INDICES_4_CPU()
    SUM_PARTIALS_PARTIALS_4_CPU();
    REAL oneOverScaling = 1.0/scalingFactors[pattern];
    for(int i = 0; i < PADDED_STATE_COUNT; i++) {
        partials3[deltaPartials + i] = sum1[i] * sum2[i] * oneOverScaling;
    }
#else // GPU implementation
    DETERMINE_INDICES_4_GPU();
    LOAD_PARTIALS_PARTIALS_4_GPU();
    LOAD_SCALING_4_GPU();
    LOAD_MATRIX_4_GPU();
    if (pattern < totalPatterns) { // Remove padded threads!
        SUM_PARTIALS_PARTIALS_4_GPU();
        partials3[u] = sum1 * sum2 / fixedScalingFactors[patIdx * 4 + pat];
    }
#endif // FW_OPENCL_CPU
}
    
KW_GLOBAL_KERNEL void kernelStatesPartialsNoScale(KW_GLOBAL_VAR int* KW_RESTRICT states1,
                                                  KW_GLOBAL_VAR REAL* KW_RESTRICT partials2,
                                                  KW_GLOBAL_VAR REAL* KW_RESTRICT partials3,
                                                  KW_GLOBAL_VAR REAL* KW_RESTRICT matrices1,
                                                  KW_GLOBAL_VAR REAL* KW_RESTRICT matrices2,
                                                  int totalPatterns) {
#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    DETERMINE_INDICES_4_CPU()
    SUM_STATES_PARTIALS_4_CPU();
    for(int i = 0; i < PADDED_STATE_COUNT; i++) {
        partials3[deltaPartials + i] = sum1[i] * sum2[i];
    }
#else // GPU implementation
    DETERMINE_INDICES_4_GPU();
    LOAD_PARTIALS_SINGLE_4_GPU();
    LOAD_MATRIX_4_GPU();
    if (pattern < totalPatterns) { // Remove padded threads!
        SUM_STATES_PARTIALS_4_GPU();
        partials3[u] = sum1 * sum2;
    }
#endif // FW_OPENCL_CPU
}

KW_GLOBAL_KERNEL void kernelStatesPartialsFixedScale(KW_GLOBAL_VAR int* KW_RESTRICT states1,
                                                     KW_GLOBAL_VAR REAL* KW_RESTRICT partials2,
                                                     KW_GLOBAL_VAR REAL* KW_RESTRICT partials3,
                                                     KW_GLOBAL_VAR REAL* KW_RESTRICT matrices1,
                                                     KW_GLOBAL_VAR REAL* KW_RESTRICT matrices2,
                                                     KW_GLOBAL_VAR REAL* KW_RESTRICT scalingFactors,
                                                     int totalPatterns) {

#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    DETERMINE_INDICES_4_CPU()
    SUM_STATES_PARTIALS_4_CPU();
    REAL oneOverScaling = 1.0/scalingFactors[pattern];
    for(int i = 0; i < PADDED_STATE_COUNT; i++) {
        partials3[deltaPartials + i] = sum1[i] * sum2[i] * oneOverScaling;
    }
#else // GPU implementation
    DETERMINE_INDICES_4_GPU();
    LOAD_PARTIALS_SINGLE_4_GPU();
    LOAD_SCALING_4_GPU();
    LOAD_MATRIX_4_GPU();
    if (pattern < totalPatterns) { // Remove padded threads!
        SUM_STATES_PARTIALS_4_GPU();
        partials3[u] = sum1 * sum2 / fixedScalingFactors[patIdx * 4 + pat];
    }
#endif // FW_OPENCL_CPU
}

KW_GLOBAL_KERNEL void kernelStatesStatesNoScale(KW_GLOBAL_VAR int* KW_RESTRICT states1,
                                                KW_GLOBAL_VAR int* KW_RESTRICT states2,
                                                KW_GLOBAL_VAR REAL* KW_RESTRICT partials3,
                                                KW_GLOBAL_VAR REAL* KW_RESTRICT matrices1,
                                                KW_GLOBAL_VAR REAL* KW_RESTRICT matrices2,
                                                int totalPatterns) {
#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    DETERMINE_INDICES_4_CPU()
    KW_GLOBAL_VAR REAL* KW_RESTRICT sMatrix1 = matrices1 + deltaMatrix;
    KW_GLOBAL_VAR REAL* KW_RESTRICT sMatrix2 = matrices2 + deltaMatrix;
    int state1 = states1[pattern];
    int state2 = states2[pattern];
    if (state1 < PADDED_STATE_COUNT && state2 < PADDED_STATE_COUNT) {
        for(int i = 0; i < PADDED_STATE_COUNT; i++) {
            partials3[deltaPartials + i] = sMatrix1[state1 * 4 + i] * sMatrix2[state2 * 4 + i];
        }
    } else if (state1 < PADDED_STATE_COUNT) {
        for(int i = 0; i < PADDED_STATE_COUNT; i++) {
            partials3[deltaPartials + i] = sMatrix1[state1 * 4 + i];
        }
    } else if (state2 < PADDED_STATE_COUNT) {
        for(int i = 0; i < PADDED_STATE_COUNT; i++) {
            partials3[deltaPartials + i] = sMatrix2[state2 * 4 + i];
        }
    } else {
        partials3[deltaPartials + 0] = 1.0; // unrolled to work around Apple OpenCL bug
        partials3[deltaPartials + 1] = 1.0;
        partials3[deltaPartials + 2] = 1.0;
        partials3[deltaPartials + 3] = 1.0;
    }
#else // GPU implementation
    DETERMINE_INDICES_4_GPU();
    LOAD_MATRIX_4_GPU();
    if (pattern < totalPatterns) {
        int state1 = states1[pattern];
        int state2 = states2[pattern];
        if (state1 < PADDED_STATE_COUNT && state2 < PADDED_STATE_COUNT) {
            partials3[u] = sMatrix1[state1 * 4 + state] * sMatrix2[state2 * 4 + state];
        } else if (state1 < PADDED_STATE_COUNT) {
            partials3[u] = sMatrix1[state1 * 4 + state];
        } else if (state2 < PADDED_STATE_COUNT) {
            partials3[u] = sMatrix2[state2 * 4 + state];
        } else {
            partials3[u] = 1.0;
        }
    }
#endif // FW_OPENCL_CPU
}

KW_GLOBAL_KERNEL void kernelStatesStatesFixedScale(KW_GLOBAL_VAR int* KW_RESTRICT states1,
                                                   KW_GLOBAL_VAR int* KW_RESTRICT states2,
                                                   KW_GLOBAL_VAR REAL* KW_RESTRICT partials3,
                                                   KW_GLOBAL_VAR REAL* KW_RESTRICT matrices1,
                                                   KW_GLOBAL_VAR REAL* KW_RESTRICT matrices2,
                                                   KW_GLOBAL_VAR REAL* KW_RESTRICT scalingFactors,
                                                   int totalPatterns) {
#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    DETERMINE_INDICES_4_CPU()
    KW_GLOBAL_VAR REAL* KW_RESTRICT sMatrix1 = matrices1 + deltaMatrix;
    KW_GLOBAL_VAR REAL* KW_RESTRICT sMatrix2 = matrices2 + deltaMatrix;
    int state1 = states1[pattern];
    int state2 = states2[pattern];
    REAL oneOverScaling = 1.0/scalingFactors[pattern];
    if (state1 < PADDED_STATE_COUNT && state2 < PADDED_STATE_COUNT) {
        for(int i = 0; i < PADDED_STATE_COUNT; i++) {
            partials3[deltaPartials + i] = sMatrix1[state1 * 4 + i] * sMatrix2[state2 * 4 + i]
                                           * oneOverScaling;
        }
    } else if (state1 < PADDED_STATE_COUNT) {
        for(int i = 0; i < PADDED_STATE_COUNT; i++) {
            partials3[deltaPartials + i] = sMatrix1[state1 * 4 + i] * oneOverScaling;
        }
    } else if (state2 < PADDED_STATE_COUNT) {
        for(int i = 0; i < PADDED_STATE_COUNT; i++) {
            partials3[deltaPartials + i] = sMatrix2[state2 * 4 + i] * oneOverScaling;
        }
    } else {
        for(int i = 0; i < PADDED_STATE_COUNT; i++) {
            partials3[deltaPartials + i] = 1.0 * oneOverScaling;
        }
    }
#else // GPU implementation
    DETERMINE_INDICES_4_GPU();
    LOAD_SCALING_4_GPU();
    LOAD_MATRIX_4_GPU();
    if (pattern < totalPatterns) {
        int state1 = states1[pattern];
        int state2 = states2[pattern];
        if (state1 < PADDED_STATE_COUNT && state2 < PADDED_STATE_COUNT) {
            partials3[u] = sMatrix1[state1 * 4 + state] * sMatrix2[state2 * 4 + state]
                           / fixedScalingFactors[patIdx * 4 + pat];
        } else if (state1 < PADDED_STATE_COUNT) {
            partials3[u] = sMatrix1[state1 * 4 + state] / fixedScalingFactors[patIdx * 4 + pat];
        } else if (state2 < PADDED_STATE_COUNT) {
            partials3[u] = sMatrix2[state2 * 4 + state] / fixedScalingFactors[patIdx * 4 + pat];
        } else {
            partials3[u] = 1.0 / fixedScalingFactors[patIdx * 4 + pat];
        }
    }
#endif // FW_OPENCL_CPU
}

// Find a scaling factor for each pattern
KW_GLOBAL_KERNEL void kernelPartialsDynamicScaling(KW_GLOBAL_VAR REAL* KW_RESTRICT allPartials,
                                                   KW_GLOBAL_VAR REAL* KW_RESTRICT scalingFactors,
                                                   int matrixCount) {
#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    FIND_MAX_PARTIALS_4_CPU();
    if (maxPartial == 0)
        maxPartial = 1.0;
    scalingFactors[pattern] = maxPartial;
    SCALE_PARTIALS_4_CPU();
#else // GPU implementation
    FIND_MAX_PARTIALS_STATE_4_GPU();
    // Could also parallel-reduce here.
    if (state == 0 && matrix == 0) {
        FIND_MAX_PARTIALS_MATRIX_4_GPU();
        if (matrixMax[pat] == 0)
            matrixMax[pat] = 1.0;
        scalingFactors[pattern] = matrixMax[pat]; // TODO: Are these incoherent writes?
    }
    SCALE_PARTIALS_4_GPU();
#endif // FW_OPENCL_CPU
}

KW_GLOBAL_KERNEL void kernelPartialsDynamicScalingScalersLog(KW_GLOBAL_VAR REAL* KW_RESTRICT allPartials,
                                                             KW_GLOBAL_VAR REAL* KW_RESTRICT scalingFactors,
                                                             int matrixCount) {
#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    FIND_MAX_PARTIALS_4_CPU();
    if (maxPartial == 0) {
        maxPartial = 1.0;
        scalingFactors[pattern] = 0.0;
    } else {
        scalingFactors[pattern] = log(maxPartial);
    }
    SCALE_PARTIALS_4_CPU();
#else // GPU implementation
    FIND_MAX_PARTIALS_STATE_4_GPU();
    if (state == 0 && matrix == 0) {
        FIND_MAX_PARTIALS_MATRIX_4_GPU();
        if (matrixMax[pat] == 0) {
            matrixMax[pat] = 1.0;
            scalingFactors[pattern] = 0.0;
        } else {
            scalingFactors[pattern] = log(matrixMax[pat]);
        }
    }
    SCALE_PARTIALS_4_GPU();
#endif // FW_OPENCL_CPU
}

// Find a scaling factor for each pattern and accumulate into buffer
KW_GLOBAL_KERNEL void kernelPartialsDynamicScalingAccumulate(KW_GLOBAL_VAR REAL* KW_RESTRICT allPartials,
                                                             KW_GLOBAL_VAR REAL* KW_RESTRICT scalingFactors,
                                                             KW_GLOBAL_VAR REAL* KW_RESTRICT cumulativeScaling,
                                                             int matrixCount) {
#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    FIND_MAX_PARTIALS_4_CPU();
    if (maxPartial == 0)
        maxPartial = 1.0;
    scalingFactors[pattern] = maxPartial;
    cumulativeScaling[pattern] += log(maxPartial);
    SCALE_PARTIALS_4_CPU();
#else // GPU implementation
    FIND_MAX_PARTIALS_STATE_4_GPU();
    if (state == 0 && matrix == 0) {
        FIND_MAX_PARTIALS_MATRIX_4_GPU();
        if (matrixMax[pat] == 0)
            matrixMax[pat] = 1.0;
        scalingFactors[pattern] = matrixMax[pat]; 
        cumulativeScaling[pattern] += log(matrixMax[pat]);
    }
    SCALE_PARTIALS_4_GPU();
#endif // FW_OPENCL_CPU
}

KW_GLOBAL_KERNEL void kernelPartialsDynamicScalingAccumulateScalersLog(KW_GLOBAL_VAR REAL* KW_RESTRICT allPartials,
                                                                       KW_GLOBAL_VAR REAL* KW_RESTRICT scalingFactors,
                                                                       KW_GLOBAL_VAR REAL* KW_RESTRICT cumulativeScaling,
                                                                       int matrixCount) {
#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    FIND_MAX_PARTIALS_4_CPU();
    if (maxPartial == 0) {
        maxPartial = 1.0;
        scalingFactors[pattern] = 0.0;
    } else {
        REAL logMax = log(maxPartial);
        scalingFactors[pattern] = logMax;
        cumulativeScaling[pattern] += logMax;
    }
    SCALE_PARTIALS_4_CPU();
#else // GPU implementation
    FIND_MAX_PARTIALS_STATE_4_GPU();
    if (state == 0 && matrix == 0) {
        FIND_MAX_PARTIALS_MATRIX_4_GPU();
        if (matrixMax[pat] == 0) {
            matrixMax[pat] = 1.0;
            scalingFactors[pattern] = 0.0;
        } else {
            REAL logMax = log(matrixMax[pat]);
            scalingFactors[pattern] = logMax;
            cumulativeScaling[pattern] += logMax; // TODO: Fix, this is both a read and write
        }
    }
    SCALE_PARTIALS_4_GPU();
#endif // FW_OPENCL_CPU
}

KW_GLOBAL_KERNEL void kernelIntegrateLikelihoods(KW_GLOBAL_VAR REAL* KW_RESTRICT dResult,
                                                 KW_GLOBAL_VAR REAL* KW_RESTRICT dRootPartials,
                                                 KW_GLOBAL_VAR REAL* KW_RESTRICT dWeights,
                                                 KW_GLOBAL_VAR REAL* KW_RESTRICT dFrequencies,
                                                 int matrixCount,
                                                 int patternCount) {
#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    INTEGRATE_PARTIALS_4_CPU();
    dResult[pattern] = log(sumTotal);
#else // GPU implementation
    INTEGRATE_PARTIALS_4_GPU();
    if (state == 0)
        dResult[pattern] = log(sum[pat][state]);
#endif // FW_OPENCL_CPU
}

KW_GLOBAL_KERNEL void kernelIntegrateLikelihoodsFixedScale(KW_GLOBAL_VAR REAL* KW_RESTRICT dResult,
                                                           KW_GLOBAL_VAR REAL* KW_RESTRICT dRootPartials,
                                                           KW_GLOBAL_VAR REAL* KW_RESTRICT dWeights,
                                                           KW_GLOBAL_VAR REAL* KW_RESTRICT dFrequencies,
                                                           KW_GLOBAL_VAR REAL* KW_RESTRICT dRootScalingFactors,
                                                           int matrixCount,
                                                           int patternCount) {
#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    INTEGRATE_PARTIALS_4_CPU();
    dResult[pattern] = log(sumTotal) + dRootScalingFactors[pattern];
#else // GPU implementation
    INTEGRATE_PARTIALS_4_GPU();
    if (state == 0)
        dResult[pattern] = log(sum[pat][state]) + dRootScalingFactors[pattern];
#endif // FW_OPENCL_CPU
}

KW_GLOBAL_KERNEL void kernelIntegrateLikelihoodsMulti(KW_GLOBAL_VAR REAL* KW_RESTRICT dResult,
                                                      KW_GLOBAL_VAR REAL* KW_RESTRICT dRootPartials,
                                                      KW_GLOBAL_VAR REAL* KW_RESTRICT dWeights,
                                                      KW_GLOBAL_VAR REAL* KW_RESTRICT dFrequencies,
                                                      int matrixCount,
                                                      int patternCount,
                                                      int takeLog) {
#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    INTEGRATE_PARTIALS_4_CPU();
    if (takeLog == 0)
        dResult[pattern] = sumTotal;
    else if (takeLog == 1)
        dResult[pattern] = log(dResult[pattern] + sumTotal);
    else 
        dResult[pattern] += sumTotal;
#else // GPU implementation
    INTEGRATE_PARTIALS_4_GPU();
    if (state == 0) {
        if (takeLog == 0)
            dResult[pattern] = sum[pat][state];
        else if (takeLog == 1)
            dResult[pattern] = log(dResult[pattern] + sum[pat][state]);
        else 
            dResult[pattern] += sum[pat][state];
    }
#endif // FW_OPENCL_CPU
}

KW_GLOBAL_KERNEL void kernelIntegrateLikelihoodsFixedScaleMulti(KW_GLOBAL_VAR REAL* KW_RESTRICT dResult,
                                                                KW_GLOBAL_VAR REAL* KW_RESTRICT dRootPartials,
                                                                KW_GLOBAL_VAR REAL* KW_RESTRICT dWeights,
                                                                KW_GLOBAL_VAR REAL* KW_RESTRICT dFrequencies,
                                                                KW_GLOBAL_VAR REAL* KW_RESTRICT dScalingFactors,
                                                                KW_GLOBAL_VAR unsigned int* KW_RESTRICT dPtrQueue,
                                                                KW_GLOBAL_VAR REAL* KW_RESTRICT dMaxScalingFactors,
                                                                KW_GLOBAL_VAR unsigned int* KW_RESTRICT dIndexMaxScalingFactors,
                                                                int matrixCount,
                                                                int patternCount,
                                                                int subsetCount,
                                                                int subsetIndex) {
#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    INTEGRATE_PARTIALS_4_CPU();
    REAL cumulativeScalingFactor = (dScalingFactors + dPtrQueue[subsetIndex])[pattern];
    if (subsetIndex == 0) {
        int indexMaxScalingFactor = 0;
        REAL maxScalingFactor = cumulativeScalingFactor;
        for (int j = 1; j < subsetCount; j++) {
            REAL tmpScalingFactor = (dScalingFactors + dPtrQueue[j])[pattern];
            if (tmpScalingFactor > maxScalingFactor) {
                indexMaxScalingFactor = j;
                maxScalingFactor = tmpScalingFactor;
            }
        }
        dIndexMaxScalingFactors[pattern] = indexMaxScalingFactor;
        dMaxScalingFactors[pattern] = maxScalingFactor; 
        if (indexMaxScalingFactor != 0)
            sumTotal *= exp((REAL)(cumulativeScalingFactor - maxScalingFactor));
        dResult[pattern] = sumTotal;
    } else {
        if (subsetIndex != dIndexMaxScalingFactors[pattern])
            sumTotal *= exp((REAL)(cumulativeScalingFactor - dMaxScalingFactors[pattern]));
        if (subsetIndex == subsetCount - 1)
            dResult[pattern] = (log(dResult[pattern] + sumTotal) + dMaxScalingFactors[pattern]);
        else
            dResult[pattern] += sumTotal;
    }
#else // GPU implementation
    INTEGRATE_PARTIALS_4_GPU();
    REAL cumulativeScalingFactor = (dScalingFactors + dPtrQueue[subsetIndex])[pattern];
    if (subsetIndex == 0) {
        int indexMaxScalingFactor = 0;
        REAL maxScalingFactor = cumulativeScalingFactor;
        for (int j = 1; j < subsetCount; j++) {
            REAL tmpScalingFactor = (dScalingFactors + dPtrQueue[j])[pattern];
            if (tmpScalingFactor > maxScalingFactor) {
                indexMaxScalingFactor = j;
                maxScalingFactor = tmpScalingFactor;
            }
        }
        dIndexMaxScalingFactors[pattern] = indexMaxScalingFactor;
        dMaxScalingFactors[pattern] = maxScalingFactor; 
        if (indexMaxScalingFactor != 0)
            sum[pat][state] *= exp((REAL)(cumulativeScalingFactor - maxScalingFactor));
        if (state == 0)
            dResult[pattern] = sum[pat][state];
        KW_LOCAL_FENCE;
    } else {
        if (subsetIndex != dIndexMaxScalingFactors[pattern])
            sum[pat][state] *= exp((REAL)(cumulativeScalingFactor - dMaxScalingFactors[pattern]));
        if (state == 0) {
            if (subsetIndex == subsetCount - 1)
                dResult[pattern] = (log(dResult[pattern] + sum[pat][state]) + dMaxScalingFactors[pattern]);
            else
                dResult[pattern] += sum[pat][state];
        }
    }
#endif // FW_OPENCL_CPU
}

////////////////////////////////////////////////////////////////////////////////////////////////
// edge and deriv kernels

KW_GLOBAL_KERNEL void kernelPartialsPartialsEdgeLikelihoods(KW_GLOBAL_VAR REAL* KW_RESTRICT dPartialsTmp,
                                                            KW_GLOBAL_VAR REAL* KW_RESTRICT partials1,
                                                            KW_GLOBAL_VAR REAL* KW_RESTRICT partials2,
                                                            KW_GLOBAL_VAR REAL* KW_RESTRICT matrices1,
                                                            int totalPatterns) {
#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    DETERMINE_INDICES_4_CPU();
    SUM_PARTIALS_SINGLE_4_CPU();
    for(int i = 0; i < PADDED_STATE_COUNT; i++) {
        dPartialsTmp[deltaPartials + i] = sum1[i] * sPartials2[i];
    }
#else // GPU implementation
    DETERMINE_INDICES_4_GPU();
    LOAD_PARTIALS_PARTIALS_4_GPU();
    LOAD_MATRIX_SINGLE_4_GPU();
    if (pattern < totalPatterns) {
        SUM_PARTIALS_SINGLE_4_GPU();
        dPartialsTmp[u] = sum1 * sPartials2[patIdx16pat4 | state];
    }
#endif // FW_OPENCL_CPU
}

KW_GLOBAL_KERNEL void kernelPartialsPartialsEdgeLikelihoodsSecondDeriv(KW_GLOBAL_VAR REAL* KW_RESTRICT dPartialsTmp,
                                                                       KW_GLOBAL_VAR REAL* KW_RESTRICT dFirstDerivTmp,
                                                                       KW_GLOBAL_VAR REAL* KW_RESTRICT dSecondDerivTmp,
                                                                       KW_GLOBAL_VAR REAL* KW_RESTRICT partials1,
                                                                       KW_GLOBAL_VAR REAL* KW_RESTRICT partials2,
                                                                       KW_GLOBAL_VAR REAL* KW_RESTRICT matrices1,
                                                                       KW_GLOBAL_VAR REAL* KW_RESTRICT dFirstDerivMatrix,
                                                                       KW_GLOBAL_VAR REAL* KW_RESTRICT dSecondDerivMatrix,
                                                                       int totalPatterns) {
#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    DETERMINE_INDICES_4_CPU();
    SUM_PARTIALS_DERIV_4_CPU();
    for(int i = 0; i < PADDED_STATE_COUNT; i++) {
        dPartialsTmp[deltaPartials + i]    = sum1[i]           * sPartials2[i];
        dFirstDerivTmp[deltaPartials + i]  = sumFirstDeriv[i]  * sPartials2[i];
        dSecondDerivTmp[deltaPartials + i] = sumSecondDeriv[i] * sPartials2[i];
    }
#else // GPU implementation
    DETERMINE_INDICES_4_GPU();
    LOAD_PARTIALS_PARTIALS_4_GPU()
    LOAD_MATRIX_DERIV_4_GPU();
    if (pattern < totalPatterns) {
        SUM_PARTIALS_DERIV_4_GPU();
        dPartialsTmp[u]    = sum1           * sPartials2[patIdx16pat4 | state];
        dFirstDerivTmp[u]  = sumFirstDeriv  * sPartials2[patIdx16pat4 | state];
        dSecondDerivTmp[u] = sumSecondDeriv * sPartials2[patIdx16pat4 | state];
    }
#endif // FW_OPENCL_CPU
}

KW_GLOBAL_KERNEL void kernelStatesPartialsEdgeLikelihoods(KW_GLOBAL_VAR REAL* KW_RESTRICT dPartialsTmp,
                                                          KW_GLOBAL_VAR REAL* KW_RESTRICT partials2,
                                                          KW_GLOBAL_VAR int* KW_RESTRICT dChildStates,
                                                          KW_GLOBAL_VAR REAL* KW_RESTRICT matrices1,
                                                          int totalPatterns) {
#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    DETERMINE_INDICES_4_CPU();
    SUM_STATES_SINGLE_4_CPU();
    for(int i = 0; i < PADDED_STATE_COUNT; i++) {
        dPartialsTmp[deltaPartials + i] = sum1[i] * sPartials2[i];
    }
#else // GPU implementation
    DETERMINE_INDICES_4_GPU();
    LOAD_PARTIALS_SINGLE_4_GPU();
    LOAD_MATRIX_SINGLE_4_GPU();
    if (pattern < totalPatterns) {
        SUM_STATES_SINGLE_4_GPU();
        int patIdx16pat4 = multBy16(patIdx) | (tx & 0xC);
        dPartialsTmp[u] = sum1 * sPartials2[patIdx16pat4 | state];
    }
#endif // FW_OPENCL_CPU
}

KW_GLOBAL_KERNEL void kernelStatesPartialsEdgeLikelihoodsSecondDeriv(KW_GLOBAL_VAR REAL* KW_RESTRICT dPartialsTmp,
                                                                     KW_GLOBAL_VAR REAL* KW_RESTRICT dFirstDerivTmp,
                                                                     KW_GLOBAL_VAR REAL* KW_RESTRICT dSecondDerivTmp,
                                                                     KW_GLOBAL_VAR REAL* KW_RESTRICT partials2,
                                                                     KW_GLOBAL_VAR int* KW_RESTRICT dChildStates,
                                                                     KW_GLOBAL_VAR REAL* KW_RESTRICT matrices1,
                                                                     KW_GLOBAL_VAR REAL* KW_RESTRICT dFirstDerivMatrix,
                                                                     KW_GLOBAL_VAR REAL* KW_RESTRICT dSecondDerivMatrix,
                                                                     int totalPatterns) {
#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    DETERMINE_INDICES_4_CPU();
    SUM_STATES_DERIV_4_CPU();
    for(int i = 0; i < PADDED_STATE_COUNT; i++) {
        dPartialsTmp[deltaPartials + i]    = sum1[i]           * sPartials2[i];
        dFirstDerivTmp[deltaPartials + i]  = sumFirstDeriv[i]  * sPartials2[i];
        dSecondDerivTmp[deltaPartials + i] = sumSecondDeriv[i] * sPartials2[i];
    }
#else // GPU implementation
    DETERMINE_INDICES_4_GPU();
    LOAD_PARTIALS_SINGLE_4_GPU();
    LOAD_MATRIX_DERIV_4_GPU();
    if (pattern < totalPatterns) { // Remove padded threads!
        SUM_STATES_DERIV_4_GPU();
        int patIdx16pat4 = multBy16(patIdx) | (tx & 0xC);
        dPartialsTmp[u]    = sum1           * sPartials2[patIdx16pat4 | state];
        dFirstDerivTmp[u]  = sumFirstDeriv  * sPartials2[patIdx16pat4 | state];
        dSecondDerivTmp[u] = sumSecondDeriv * sPartials2[patIdx16pat4 | state];
    }
#endif // FW_OPENCL_CPU
}

KW_GLOBAL_KERNEL void kernelIntegrateLikelihoodsSecondDeriv(KW_GLOBAL_VAR REAL* KW_RESTRICT dResult,
                                                            KW_GLOBAL_VAR REAL* KW_RESTRICT dFirstDerivResult,
                                                            KW_GLOBAL_VAR REAL* KW_RESTRICT dSecondDerivResult,
                                                            KW_GLOBAL_VAR REAL* KW_RESTRICT dRootPartials,
                                                            KW_GLOBAL_VAR REAL* KW_RESTRICT dRootFirstDeriv,
                                                            KW_GLOBAL_VAR REAL* KW_RESTRICT dRootSecondDeriv,
                                                            KW_GLOBAL_VAR REAL* KW_RESTRICT dWeights,
                                                            KW_GLOBAL_VAR REAL* KW_RESTRICT dFrequencies,
                                                            int matrixCount,
                                                            int patternCount) {
#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    INTEGRATE_PARTIALS_DERIV_4_CPU();
    tmpLogLike = sumTotal;
    dResult[pattern] = log(tmpLogLike);
    tmpFirstDeriv = sumTotalD1 / tmpLogLike;
    dFirstDerivResult[pattern] = tmpFirstDeriv;
    dSecondDerivResult[pattern] = (sumTotalD2 / tmpLogLike - tmpFirstDeriv * tmpFirstDeriv);
#else // GPU implementation
    INTEGRATE_PARTIALS_DERIV_4_GPU();
    if (state == 0) {
        tmpLogLike = sum[pat][state];
        dResult[pattern] = log(tmpLogLike);
        tmpFirstDeriv = sumD1[pat][state] / tmpLogLike;
        dFirstDerivResult[pattern] = tmpFirstDeriv;
        dSecondDerivResult[pattern] = (sumD2[pat][state] / tmpLogLike - tmpFirstDeriv * tmpFirstDeriv);
    }
#endif // FW_OPENCL_CPU
}

KW_GLOBAL_KERNEL void kernelIntegrateLikelihoodsFixedScaleSecondDeriv(KW_GLOBAL_VAR REAL* KW_RESTRICT dResult,
                                                                      KW_GLOBAL_VAR REAL* KW_RESTRICT dFirstDerivResult,
                                                                      KW_GLOBAL_VAR REAL* KW_RESTRICT dSecondDerivResult,
                                                                      KW_GLOBAL_VAR REAL* KW_RESTRICT dRootPartials,
                                                                      KW_GLOBAL_VAR REAL* KW_RESTRICT dRootFirstDeriv,
                                                                      KW_GLOBAL_VAR REAL* KW_RESTRICT dRootSecondDeriv,
                                                                      KW_GLOBAL_VAR REAL* KW_RESTRICT dWeights,
                                                                      KW_GLOBAL_VAR REAL* KW_RESTRICT dFrequencies,
                                                                      KW_GLOBAL_VAR REAL* KW_RESTRICT dRootScalingFactors,
                                                                      int matrixCount,
                                                                      int patternCount) {
#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    INTEGRATE_PARTIALS_DERIV_4_CPU();
    tmpLogLike = sumTotal;
    dResult[pattern] = log(tmpLogLike) + dRootScalingFactors[pattern];
    tmpFirstDeriv = sumTotalD1 / tmpLogLike;
    dFirstDerivResult[pattern] = tmpFirstDeriv;
    dSecondDerivResult[pattern] = (sumTotalD2 / tmpLogLike - tmpFirstDeriv * tmpFirstDeriv);
#else // GPU implementation
    INTEGRATE_PARTIALS_DERIV_4_GPU();
    if (state == 0) {
        tmpLogLike = sum[pat][state];
        dResult[pattern] = log(tmpLogLike) + dRootScalingFactors[pattern];
        tmpFirstDeriv = sumD1[pat][state] / tmpLogLike;
        dFirstDerivResult[pattern] = tmpFirstDeriv;
        dSecondDerivResult[pattern] = (sumD2[pat][state] / tmpLogLike - tmpFirstDeriv * tmpFirstDeriv);
    }
#endif // FW_OPENCL_CPU
}

////////////////////////////////////////////////////////////////////////////////////////////////
// scaling experiments kernels

KW_GLOBAL_KERNEL void kernelPartialsPartialsCheckScale(KW_GLOBAL_VAR REAL* partials1,
                                                                  KW_GLOBAL_VAR REAL* partials2,
                                                                  KW_GLOBAL_VAR REAL* partials3,
                                                                  KW_GLOBAL_VAR REAL* matrices1,
                                                                  KW_GLOBAL_VAR REAL* matrices2,
                                                                  KW_GLOBAL_VAR int* dRescalingTrigger,
                                                                  int totalPatterns) {
        REAL sum1;
        REAL sum2;
        int i;

        DETERMINE_INDICES_4_GPU();

        int patIdx16pat4 = multBy16(patIdx) | (tx & 0xC);
        int y = deltaPartialsByState + deltaPartialsByMatrix;
        
        KW_GLOBAL_VAR REAL* matrix1 = matrices1 + x2; // Points to *this* matrix
        KW_GLOBAL_VAR REAL* matrix2 = matrices2 + x2;

    #ifdef KERNEL_PRINT_ENABLED
        printf("matrix = %d, pat = %d for tx = %d and state = %d :  u = %d\n", matrix, pattern, tx,
               state, u);
    #endif

        // Load values into shared memory
        KW_LOCAL_MEM REAL sMatrix1[16];
        KW_LOCAL_MEM REAL sMatrix2[16];

        KW_LOCAL_MEM REAL sPartials1[PATTERN_BLOCK_SIZE * 4 * 4];
        KW_LOCAL_MEM REAL sPartials2[PATTERN_BLOCK_SIZE * 4 * 4];

        // copy PADDED_STATE_COUNT * PATTERN_BLOCK_SIZE lengthed partials
        if (pattern < totalPatterns) {
            sPartials1[multBy16(patIdx) | tx] = partials1[y | tx]; // All coalesced memory reads
            sPartials2[multBy16(patIdx) | tx] = partials2[y | tx];
        } else {
            sPartials1[multBy16(patIdx) | tx] = 0;
            sPartials2[multBy16(patIdx) | tx] = 0;
        }

        if (patIdx == 0 ) {
            sMatrix1[tx] = matrix1[tx]; // All coalesced memory reads
            sMatrix2[tx] = matrix2[tx];
        }

        KW_LOCAL_FENCE;

        if (pattern < totalPatterns) { // Remove padded threads!

            i = pat;
            sum1  = sMatrix1[multBy4(i) | state] * sPartials1[patIdx16pat4 | i];
            sum2  = sMatrix2[multBy4(i) | state] * sPartials2[patIdx16pat4 | i];

            i = (++i) & 0x3;
            sum1 += sMatrix1[multBy4(i) | state] * sPartials1[patIdx16pat4 | i];
            sum2 += sMatrix2[multBy4(i) | state] * sPartials2[patIdx16pat4 | i];

            i = (++i) & 0x3;
            sum1 += sMatrix1[multBy4(i) | state] * sPartials1[patIdx16pat4 | i];
            sum2 += sMatrix2[multBy4(i) | state] * sPartials2[patIdx16pat4 | i];

            i = (++i) & 0x3;
            sum1 += sMatrix1[multBy4(i) | state] * sPartials1[patIdx16pat4 | i];
            sum2 += sMatrix2[multBy4(i) | state] * sPartials2[patIdx16pat4 | i];
            
            REAL tmpPartial = sum1 * sum2;
            
            partials3[u] = tmpPartial;

            if (tmpPartial < SCALING_THRESHOLD_LOWER || tmpPartial > SCALING_THRESHOLD_UPPER)
                *dRescalingTrigger = 1;
            
//            union {float f; long l;} fl;
//            fl.f = sum1 * sum2;;
//
//          partials3[u] = fl.f;
//            
//            int expTmp  = ((fl.l >> 23) & 0x000000ff) - 0x7e;
//            
//            if (abs(expTmp) > SCALING_EXPONENT_THRESHOLD)
//                *dRescalingTrigger = 1;
        }

    }

KW_GLOBAL_KERNEL void kernelPartialsPartialsFixedCheckScale(KW_GLOBAL_VAR REAL* partials1,
                                                      KW_GLOBAL_VAR REAL* partials2,
                                                      KW_GLOBAL_VAR REAL* partials3,
                                                      KW_GLOBAL_VAR REAL* matrices1,
                                                      KW_GLOBAL_VAR REAL* matrices2,
                                                      KW_GLOBAL_VAR REAL* scalingFactors,
                                                      KW_GLOBAL_VAR int* dRescalingTrigger,
                                                      int totalPatterns) {
    REAL sum1;
    REAL sum2;
    int i;

    DETERMINE_INDICES_4_GPU();
    int y = deltaPartialsByState + deltaPartialsByMatrix;
    KW_GLOBAL_VAR REAL* matrix1 = matrices1 + x2; // Points to *this* matrix
    KW_GLOBAL_VAR REAL* matrix2 = matrices2 + x2;

#ifdef KERNEL_PRINT_ENABLED
    printf("matrix = %d, pat = %d for tx = %d and state = %d :  u = %d\n", matrix, pattern, tx,
           state, u);
#endif

    // Load values into shared memory
    KW_LOCAL_MEM REAL sMatrix1[16];
    KW_LOCAL_MEM REAL sMatrix2[16];

    KW_LOCAL_MEM REAL sPartials1[PATTERN_BLOCK_SIZE * 4 * 4];
    KW_LOCAL_MEM REAL sPartials2[PATTERN_BLOCK_SIZE * 4 * 4];

    KW_LOCAL_MEM REAL fixedScalingFactors[PATTERN_BLOCK_SIZE * 4];

    // copy PADDED_STATE_COUNT*PATTERN_BLOCK_SIZE lengthed partials
    if (pattern < totalPatterns) {
        sPartials1[patIdx * 16 + tx] = partials1[y + tx]; // All coalesced memory reads
        sPartials2[patIdx * 16 + tx] = partials2[y + tx];
    } else {
        sPartials1[patIdx * 16 + tx] = 0;
        sPartials2[patIdx * 16 + tx] = 0;
    }

    if (patIdx < 4) // need to load 4*PATTERN_BLOCK_SIZE factors for this block
        fixedScalingFactors[patIdx * PATTERN_BLOCK_SIZE + tx] =
            scalingFactors[KW_GROUP_ID_0 * PATTERN_BLOCK_SIZE * 4 + patIdx * PATTERN_BLOCK_SIZE + tx];

    if (patIdx == 0 ) {
        sMatrix1[tx] = matrix1[tx]; // All coalesced memory reads
        sMatrix2[tx] = matrix2[tx];
    }

    KW_LOCAL_FENCE;

    if (pattern < totalPatterns) { // Remove padded threads!

        i = pat;
        sum1  = sMatrix1[i * 4 + state] * sPartials1[patIdx * 16 + pat * 4 + i];
        sum2  = sMatrix2[i * 4 + state] * sPartials2[patIdx * 16 + pat * 4 + i];

        i = (++i) & 0x3;
        sum1 += sMatrix1[i * 4 + state] * sPartials1[patIdx * 16 + pat * 4 + i];
        sum2 += sMatrix2[i * 4 + state] * sPartials2[patIdx * 16 + pat * 4 + i];

        i = (++i) & 0x3;
        sum1 += sMatrix1[i * 4 + state] * sPartials1[patIdx * 16 + pat * 4 + i];
        sum2 += sMatrix2[i * 4 + state] * sPartials2[patIdx * 16 + pat * 4 + i];

        i = (++i) & 0x3;
        sum1 += sMatrix1[i * 4 + state] * sPartials1[patIdx * 16 + pat * 4 + i];
        sum2 += sMatrix2[i * 4 + state] * sPartials2[patIdx * 16 + pat * 4 + i];
        
        REAL tmpPartial = sum1 * sum2 * fixedScalingFactors[patIdx * 4 + pat];
        
        partials3[u] = tmpPartial;

        if (tmpPartial < SCALING_THRESHOLD_LOWER || tmpPartial > SCALING_THRESHOLD_UPPER)
            *dRescalingTrigger = 1;

    }

}

KW_GLOBAL_KERNEL void kernelPartialsPartialsAutoScale(KW_GLOBAL_VAR REAL* partials1,
                                                KW_GLOBAL_VAR REAL* partials2,
                                                KW_GLOBAL_VAR REAL* partials3,
                                                KW_GLOBAL_VAR REAL* matrices1,
                                                KW_GLOBAL_VAR REAL* matrices2,
                                                KW_GLOBAL_VAR signed char* scalingFactors,
                                                int totalPatterns) {
    REAL sum1;
    REAL sum2;
    int i;

    DETERMINE_INDICES_4_GPU();

    int patIdx16pat4 = multBy16(patIdx) | (tx & 0xC);
    int y = deltaPartialsByState + deltaPartialsByMatrix;
    int myIdx = multBy16(patIdx) + tx; // threadId in block
    
    KW_GLOBAL_VAR REAL* matrix1 = matrices1 + x2; // Points to *this* matrix
    KW_GLOBAL_VAR REAL* matrix2 = matrices2 + x2;

#ifdef KERNEL_PRINT_ENABLED
    printf("matrix = %d, pat = %d for tx = %d and state = %d :  u = %d\n", matrix, pattern, tx,
           state, u);
#endif

    // Load values into shared memory
    KW_LOCAL_MEM REAL sMatrix1[16];
    KW_LOCAL_MEM REAL sMatrix2[16];

    KW_LOCAL_MEM REAL sPartials1[PATTERN_BLOCK_SIZE * 4 * 4];
    KW_LOCAL_MEM REAL sPartials2[PATTERN_BLOCK_SIZE * 4 * 4];

    // copy PADDED_STATE_COUNT * PATTERN_BLOCK_SIZE lengthed partials
    if (pattern < totalPatterns) {
        sPartials1[multBy16(patIdx) | tx] = partials1[y | tx]; // All coalesced memory reads
        sPartials2[multBy16(patIdx) | tx] = partials2[y | tx];
    } else {
        sPartials1[multBy16(patIdx) | tx] = 0;
        sPartials2[multBy16(patIdx) | tx] = 0;
    }

    if (patIdx == 0 ) {
        sMatrix1[tx] = matrix1[tx]; // All coalesced memory reads
        sMatrix2[tx] = matrix2[tx];
    }

    KW_LOCAL_FENCE;

    i = pat;
    sum1  = sMatrix1[multBy4(i) | state] * sPartials1[patIdx16pat4 | i];
    sum2  = sMatrix2[multBy4(i) | state] * sPartials2[patIdx16pat4 | i];

    i = (++i) & 0x3;
    sum1 += sMatrix1[multBy4(i) | state] * sPartials1[patIdx16pat4 | i];
    sum2 += sMatrix2[multBy4(i) | state] * sPartials2[patIdx16pat4 | i];

    i = (++i) & 0x3;
    sum1 += sMatrix1[multBy4(i) | state] * sPartials1[patIdx16pat4 | i];
    sum2 += sMatrix2[multBy4(i) | state] * sPartials2[patIdx16pat4 | i];

    i = (++i) & 0x3;
    sum1 += sMatrix1[multBy4(i) | state] * sPartials1[patIdx16pat4 | i];
    sum2 += sMatrix2[multBy4(i) | state] * sPartials2[patIdx16pat4 | i];
    
    REAL tmpPartial = sum1 * sum2;
    int expTmp;
    REAL sigTmp = frexp(tmpPartial, &expTmp);        

    KW_LOCAL_FENCE;
    
    if (pattern < totalPatterns) {
        if (abs(expTmp) > SCALING_EXPONENT_THRESHOLD) {
            // now using sPartials2 to hold scaling trigger boolean
            sPartials2[patIdx16pat4] = 1;
        } else {
            partials3[u] = tmpPartial;
            sPartials2[patIdx16pat4] = 0;
            sPartials1[myIdx] = 0;
        }
    } 
    
    KW_LOCAL_FENCE;
    
    int scalingActive = sPartials2[patIdx16pat4];
        
    if (scalingActive) {
        // now using sPartials1 to store max unscaled partials3
        sPartials1[myIdx] = tmpPartial;
    }
        
    KW_LOCAL_FENCE;
        
    // Unrolled parallel max-reduction
    if (scalingActive && state < 2) {
        REAL compare = sPartials1[myIdx + 2];
        if (compare >  sPartials1[myIdx])
            sPartials1[myIdx] = compare;
    }
     
    KW_LOCAL_FENCE;
            
    if (scalingActive && state < 1) {
        REAL maxPartial = sPartials1[myIdx + 1];
        if (maxPartial < sPartials1[myIdx])
            maxPartial = sPartials1[myIdx];
        int expMax;
        frexp(maxPartial, &expMax);
        sPartials1[myIdx] = expMax;
    }

    KW_LOCAL_FENCE;
            
    if (scalingActive) 
        partials3[u] = ldexp(sigTmp, expTmp - sPartials1[patIdx16pat4]);
        
    if ((myIdx < PATTERN_BLOCK_SIZE * 4) && (myIdx + (KW_GROUP_ID_0 * PATTERN_BLOCK_SIZE * 4) < totalPatterns))
        scalingFactors[(KW_GROUP_ID_0 * PATTERN_BLOCK_SIZE * 4) + (matrix * totalPatterns) + myIdx] = sPartials1[multBy4(myIdx)];
}


KW_GLOBAL_KERNEL void kernelPartialsDynamicScalingAccumulateReciprocal(KW_GLOBAL_VAR REAL* allPartials,
                                                       KW_GLOBAL_VAR REAL* scalingFactors,
                                                       KW_GLOBAL_VAR REAL* cumulativeScaling,
                                                       int matrixCount) {
    int tx = KW_LOCAL_ID_0;
    
    int state = tx & 0x3;
    int pat = tx >> 2;
                             
    int patIdx = KW_GROUP_ID_0;
    
    int pattern = (patIdx << 2) + pat;
    int matrix = KW_LOCAL_ID_1;
    // TODO: Assumes matrixCount < MATRIX_BLOCK_SIZ
    
    // Patterns are always padded, so no reading/writing past end possible
    // Find start of patternBlock for thread-block
    int partialsOffset = (matrix * KW_NUM_GROUPS_0 + patIdx) << 4; //* 16;

    KW_LOCAL_MEM REAL partials[MATRIX_BLOCK_SIZE][16]; // 4 patterns at a time
    KW_LOCAL_MEM REAL storedPartials[MATRIX_BLOCK_SIZE][16];

    KW_LOCAL_MEM REAL matrixMax[4];
    
    if (matrix < matrixCount)
        partials[matrix][tx] = allPartials[partialsOffset + tx];          

    storedPartials[matrix][tx] = partials[matrix][tx];
           
    KW_LOCAL_FENCE;
    
    // Unrolled parallel max-reduction
    if (state < 2) {
        REAL compare1 = partials[matrix][tx];
        REAL compare2 = partials[matrix][tx + 2];
        if (compare2 > compare1)
            partials[matrix][tx] = compare2;
    }
    KW_LOCAL_FENCE;
    
    if (state < 1) {
        REAL compare1 = partials[matrix][tx];
        REAL compare2 = partials[matrix][tx + 1];
        if (compare2 > compare1)
            partials[matrix][tx] = compare2;
    }
    KW_LOCAL_FENCE;
 
    // Could also parallel-reduce here.
    if (state == 0 && matrix == 0) {
        matrixMax[pat] = 0;
        int m;
        for(m = 0; m < matrixCount; m++) {
            if (partials[m][tx] > matrixMax[pat])
                matrixMax[pat] = partials[m][tx];
        }
        
        if (matrixMax[pat] == 0)
            matrixMax[pat] = 1.0;
   
        scalingFactors[pattern] = 1/matrixMax[pat]; 
        cumulativeScaling[pattern] += log(matrixMax[pat]);
    }

    KW_LOCAL_FENCE;

    if (matrix < matrixCount)
        allPartials[partialsOffset + tx] = storedPartials[matrix][tx] / matrixMax[pat];
        
}

KW_GLOBAL_KERNEL void kernelPartialsDynamicScalingAccumulateDifference(KW_GLOBAL_VAR REAL* allPartials,
                                                                 KW_GLOBAL_VAR REAL* scalingFactors,
                                                                 KW_GLOBAL_VAR REAL* existingScalingFactors,
                                                                 KW_GLOBAL_VAR REAL* cumulativeScaling,
                                                                 int matrixCount) {
    int tx = KW_LOCAL_ID_0;
    
    int state = tx & 0x3;
    int pat = tx >> 2;
                             
    int patIdx = KW_GROUP_ID_0;
    
    int pattern = (patIdx << 2) + pat;
    int matrix = KW_LOCAL_ID_1;
    // TODO: Assumes matrixCount < MATRIX_BLOCK_SIZ
    
    // Patterns are always padded, so no reading/writing past end possible
    // Find start of patternBlock for thread-block
    int partialsOffset = (matrix * KW_NUM_GROUPS_0 + patIdx) << 4; //* 16;

    KW_LOCAL_MEM REAL partials[MATRIX_BLOCK_SIZE][16]; // 4 patterns at a time
    KW_LOCAL_MEM REAL storedPartials[MATRIX_BLOCK_SIZE][16];

    KW_LOCAL_MEM REAL matrixMax[4];
    
    if (matrix < matrixCount)
        partials[matrix][tx] = allPartials[partialsOffset + tx];          

    storedPartials[matrix][tx] = partials[matrix][tx];
           
    KW_LOCAL_FENCE;
    
    // Unrolled parallel max-reduction
    if (state < 2) {
        REAL compare1 = partials[matrix][tx];
        REAL compare2 = partials[matrix][tx + 2];
        if (compare2 > compare1)
            partials[matrix][tx] = compare2;
    }
    KW_LOCAL_FENCE;
    
    if (state < 1) {
        REAL compare1 = partials[matrix][tx];
        REAL compare2 = partials[matrix][tx + 1];
        if (compare2 > compare1)
            partials[matrix][tx] = compare2;
    }
    KW_LOCAL_FENCE;
 
    // Could also parallel-reduce here.
    if (state == 0 && matrix == 0) {
        matrixMax[pat] = 0;
        int m;
        for(m = 0; m < matrixCount; m++) {
            if (partials[m][tx] > matrixMax[pat])
                matrixMax[pat] = partials[m][tx];
        }
        
        if (matrixMax[pat] == 0)
            matrixMax[pat] = 1.0;
   
        REAL currentFactors = existingScalingFactors[pattern];
        scalingFactors[pattern] = 1/matrixMax[pat] * currentFactors; 
        cumulativeScaling[pattern] += (log(matrixMax[pat]));
    }

    KW_LOCAL_FENCE;

    if (matrix < matrixCount)
        allPartials[partialsOffset + tx] = storedPartials[matrix][tx] / matrixMax[pat];
        
}

KW_GLOBAL_KERNEL void kernelIntegrateLikelihoodsAutoScaling(KW_GLOBAL_VAR REAL* dResult,
                                                     KW_GLOBAL_VAR REAL* dRootPartials,
                                                     KW_GLOBAL_VAR REAL* dWeights,
                                                     KW_GLOBAL_VAR REAL* dFrequencies,
                                                     KW_GLOBAL_VAR int* dRootScalingFactors,
                                                     int matrixCount,
                                                     int patternCount) {
     int state   = KW_LOCAL_ID_0;
    int pat = KW_LOCAL_ID_1;
    int pattern = KW_GROUP_ID_0 * LIKE_PATTERN_BLOCK_SIZE + KW_LOCAL_ID_1;
    
    KW_LOCAL_MEM REAL stateFreq[4];
    
    // TODO: Currently assumes MATRIX_BLOCK_SIZE >= matrixCount
    KW_LOCAL_MEM REAL matrixProp[MATRIX_BLOCK_SIZE];
    KW_LOCAL_MEM REAL sum[LIKE_PATTERN_BLOCK_SIZE][4];

    // Load shared memory

    if (pat == 0) {
        stateFreq[state] = dFrequencies[state];
    }
    
    sum[pat][state] = 0;
    
    // TODO: Assumes matrixCount < LIKE_PATTERN_BLOCK_SIZE * 4
    if (pat * 4 + state < matrixCount) {
        matrixProp[pat * 4 + state] = dWeights[pat * 4 + state];
    }

    KW_LOCAL_FENCE;

    int u = state + pattern * PADDED_STATE_COUNT;
    int delta = patternCount * PADDED_STATE_COUNT;

    short maxScaleFactor = dRootScalingFactors[pattern];
    for(int r = 1; r < matrixCount; r++) {
        int tmpFactor = dRootScalingFactors[pattern + (r * patternCount)];
        if (tmpFactor > maxScaleFactor)
            maxScaleFactor = tmpFactor;
    }

    for(int r = 0; r < matrixCount; r++) {
        int tmpFactor = dRootScalingFactors[pattern + (r * patternCount)];
        if (tmpFactor != maxScaleFactor) {
            // TODO: verify which of the two methods below is faster
            int expTmp;
            sum[pat][state] += ldexp(frexp(dRootPartials[u + delta * r], &expTmp), expTmp + (tmpFactor - maxScaleFactor)) * matrixProp[r];
//            sum[pat][state] += dRootPartials[u + delta * r] * pow(2.0, tmpFactor - maxScaleFactor) * matrixProp[r];
        } else {
            sum[pat][state] += dRootPartials[u + delta * r] * matrixProp[r];
        }
    }

    sum[pat][state] *= stateFreq[state];
    KW_LOCAL_FENCE;
    if (state < 2)
        sum[pat][state] += sum[pat][state + 2];
    KW_LOCAL_FENCE;
    if (state < 1) {
        sum[pat][state] += sum[pat][state + 1];
    }

    if (state == 0)
        dResult[pattern] = (log(sum[pat][state]) + (M_LN2 * maxScaleFactor));
}

#ifdef CUDA
} // extern "C"
#endif //CUDA
