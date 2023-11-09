/*
 *
 * Copyright 2009 Phylogenetic Likelihood Working Group
 *
 * This file is part of BEAGLE.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 *
 * @author Marc Suchard
 * @author Daniel Ayres
 */

#ifdef CUDA
    #include "libhmsbeagle/GPU/GPUImplDefs.h"
    #include "libhmsbeagle/GPU/kernels/kernelsAll.cu" // This file includes the non-state-count specific kernels
    #include <mma.h>
    extern "C" {
#endif

///////////////////////////////////////////////////////////////////////////////
// kernel macros CPU

#define DETERMINE_INDICES_X_CPU()\
    int state = KW_LOCAL_ID_0;\
    int patIdx = get_global_id(1);\
    int pattern = __umul24(KW_GROUP_ID_0,PATTERN_BLOCK_SIZE) + patIdx;\
    int matrix = KW_GROUP_ID_2;\
    int patternCount = totalPatterns;\
    int deltaPartialsByState = pattern * PADDED_STATE_COUNT;\
    int deltaPartialsByMatrix = matrix * PADDED_STATE_COUNT * patternCount;\
    int deltaMatrix = matrix * PADDED_STATE_COUNT * PADDED_STATE_COUNT;\
    int u = state + deltaPartialsByState + deltaPartialsByMatrix;

#define SUM_PARTIALS_PARTIALS_X_CPU()\
    REAL sum1 = 0, sum2 = 0;\
    int deltaPartials = deltaPartialsByMatrix + deltaPartialsByState;\
    KW_GLOBAL_VAR REAL* KW_RESTRICT sMatrix1 = matrices1 + deltaMatrix;\
    KW_GLOBAL_VAR REAL* KW_RESTRICT sMatrix2 = matrices2 + deltaMatrix;\
    KW_GLOBAL_VAR REAL* KW_RESTRICT sPartials1 = partials1 + deltaPartials;\
    KW_GLOBAL_VAR REAL* KW_RESTRICT sPartials2 = partials2 + deltaPartials;\
    for(int i = 0; i < PADDED_STATE_COUNT; i++) {\
        FMA(sMatrix1[i * PADDED_STATE_COUNT + state],  sPartials1[i], sum1);\
        FMA(sMatrix2[i * PADDED_STATE_COUNT + state],  sPartials2[i], sum2);\
    }

#define SUM_STATES_PARTIALS_X_CPU()\
    REAL sum1 = 0, sum2 = 0;\
    int deltaPartials = deltaPartialsByMatrix + deltaPartialsByState;\
    KW_GLOBAL_VAR REAL* KW_RESTRICT sMatrix1 = matrices1 + deltaMatrix;\
    KW_GLOBAL_VAR REAL* KW_RESTRICT sMatrix2 = matrices2 + deltaMatrix;\
    KW_GLOBAL_VAR REAL* KW_RESTRICT sPartials2 = partials2 + deltaPartials;\
    int state1 = states1[pattern];\
    if (state1 < PADDED_STATE_COUNT)\
        sum1 = sMatrix1[state1 * PADDED_STATE_COUNT + state];\
    else\
        sum1 = 1.0;\
    for(int i = 0; i < PADDED_STATE_COUNT; i++) {\
        FMA(sMatrix2[i * PADDED_STATE_COUNT + state],  sPartials2[i], sum2);\
    }

#define FIND_MAX_PARTIALS_X_CPU()\
    int patIdx = KW_LOCAL_ID_0;\
    int pattern = KW_GROUP_ID_0 * PATTERN_BLOCK_SIZE + patIdx;\
    int deltaPartialsByState = pattern * PADDED_STATE_COUNT;\
    REAL max = 0;\
    for(int m = 0; m < matrixCount; m++) {\
        int deltaPartialsByMatrix = m * PADDED_STATE_COUNT * PATTERN_BLOCK_SIZE * KW_NUM_GROUPS_0;\
        int deltaPartials = deltaPartialsByMatrix + deltaPartialsByState;\
        for(int i = 0; i < PADDED_STATE_COUNT; i++) {\
            REAL iPartial = allPartials[deltaPartials + i];\
            if (iPartial > max)\
                max = iPartial;\
        }\
    }

#define SCALE_PARTIALS_X_CPU()\
    for(int m = 0; m < matrixCount; m++) {\
        int deltaPartialsByMatrix = m * PADDED_STATE_COUNT * PATTERN_BLOCK_SIZE * KW_NUM_GROUPS_0;\
        int deltaPartials = deltaPartialsByMatrix + deltaPartialsByState;\
        for(int i = 0; i < PADDED_STATE_COUNT; i++) {\
            allPartials[deltaPartials + i] /= max;\
        }\
    }

#define INTEGRATE_PARTIALS_X_CPU()\
    int pattern = KW_GROUP_ID_0;\
    int u = pattern * PADDED_STATE_COUNT;\
    int delta = patternCount * PADDED_STATE_COUNT;\
    REAL sumTotal = 0;\
    for (int i = 0; i < PADDED_STATE_COUNT; i++) {\
        REAL sumState = dRootPartials[i + u] * dWeights[0];\
        for(int r = 1; r < matrixCount; r++) {\
            FMA(dRootPartials[i + u + delta * r],  dWeights[r], sumState);\
        }\
        sumState *= dFrequencies[i];\
        sumTotal += sumState;\
    }

#define INTEGRATE_PARTIALS_DERIV_X_CPU()\
    int pattern = KW_GROUP_ID_0;\
    int u = pattern * PADDED_STATE_COUNT;\
    int delta = patternCount * PADDED_STATE_COUNT;\
    REAL sumTotal = 0, sumTotalD1 = 0, sumTotalD2 = 0;\
    REAL tmpLogLike, tmpFirstDeriv;\
    for (int i = 0; i < PADDED_STATE_COUNT; i++) {\
        REAL sumState = dRootPartials[   i + u] * dWeights[0];\
        REAL sumD1    = dRootFirstDeriv[ i + u] * dWeights[0];\
        REAL sumD2    = dRootSecondDeriv[i + u] * dWeights[0];\
        for(int r = 1; r < matrixCount; r++) {\
            FMA(dRootPartials[   i + u + delta * r],  dWeights[r], sumState);\
            FMA(dRootFirstDeriv[ i + u + delta * r],  dWeights[r], sumD1);\
            FMA(dRootSecondDeriv[i + u + delta * r],  dWeights[r], sumD2);\
        }\
        sumState   *= dFrequencies[i];\
        sumD1      *= dFrequencies[i];\
        sumD2      *= dFrequencies[i];\
        sumTotal   += sumState;\
        sumTotalD1 += sumD1;\
        sumTotalD2 += sumD2;\
    }

///////////////////////////////////////////////////////////////////////////////
// kernel macros GPU

#define DETERMINE_INDICES_X_GPU()\
    int state = KW_LOCAL_ID_0;\
    int patIdx = KW_LOCAL_ID_1;\
    int pattern = __umul24(KW_GROUP_ID_0,PATTERN_BLOCK_SIZE) + patIdx;\
    int matrix = KW_GROUP_ID_1;\
    int patternCount = totalPatterns;\
    int deltaPartialsByState = pattern * PADDED_STATE_COUNT;\
    int deltaPartialsByMatrix = matrix * PADDED_STATE_COUNT * patternCount;\
    int deltaMatrix = matrix * PADDED_STATE_COUNT * PADDED_STATE_COUNT;\
    int u = state + deltaPartialsByState + deltaPartialsByMatrix;

#define LOAD_SCALING_X_GPU()\
    KW_LOCAL_MEM REAL fixedScalingFactors[PATTERN_BLOCK_SIZE];\
    if (patIdx == 0 && state < PATTERN_BLOCK_SIZE ) {\
        /* TODO: If PATTERN_BLOCK_SIZE > PADDED_STATE_COUNT, there is a bug here */\
        fixedScalingFactors[state] = scalingFactors[KW_GROUP_ID_0 * PATTERN_BLOCK_SIZE + state];\
    }

#define SUM_PARTIALS_PARTIALS_X_GPU()\
    KW_GLOBAL_VAR REAL* KW_RESTRICT matrix1 = matrices1 + deltaMatrix; /* Points to *this* matrix */\
    KW_GLOBAL_VAR REAL* KW_RESTRICT matrix2 = matrices2 + deltaMatrix;\
    /* Load values into shared memory */\
    KW_LOCAL_MEM REAL sMatrix1[BLOCK_PEELING_SIZE][PADDED_STATE_COUNT];\
    KW_LOCAL_MEM REAL sMatrix2[BLOCK_PEELING_SIZE][PADDED_STATE_COUNT];\
    KW_LOCAL_MEM REAL sPartials1[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];\
    KW_LOCAL_MEM REAL sPartials2[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];\
    int y = deltaPartialsByState + deltaPartialsByMatrix;\
    /* copy PADDED_STATE_COUNT*PATTERN_BLOCK_SIZE lengthed partials */\
    /* These are all coherent global memory reads; checked in Profiler */\
    if (pattern < totalPatterns) {\
        sPartials1[patIdx][state] = partials1[y + state];\
        sPartials2[patIdx][state] = partials2[y + state];\
    } else {\
        sPartials1[patIdx][state] = 0;\
        sPartials2[patIdx][state] = 0;\
    }\
    REAL sum1 = 0, sum2 = 0;\
    for (int i = 0; i < PADDED_STATE_COUNT; i += BLOCK_PEELING_SIZE) {\
        /* load one row of matrices */\
        if (patIdx < BLOCK_PEELING_SIZE) {\
            /* These are all coherent global memory reads. */\
            sMatrix1[patIdx][state] = matrix1[patIdx * PADDED_STATE_COUNT + state];\
            sMatrix2[patIdx][state] = matrix2[patIdx * PADDED_STATE_COUNT + state];\
            /* sMatrix now filled with starting in state and ending in i */\
            matrix1 += BLOCK_PEELING_SIZE * PADDED_STATE_COUNT;\
            matrix2 += BLOCK_PEELING_SIZE * PADDED_STATE_COUNT;\
        }\
        KW_LOCAL_FENCE;\
        for(int j = 0; j < BLOCK_PEELING_SIZE; j++) {\
            FMA(sMatrix1[j][state],  sPartials1[patIdx][i + j], sum1);\
            FMA(sMatrix2[j][state],  sPartials2[patIdx][i + j], sum2);\
        }\
        KW_LOCAL_FENCE;\
    }

#define SUM_STATES_PARTIALS_X_GPU()\
    KW_LOCAL_MEM REAL sMatrix2[BLOCK_PEELING_SIZE][PADDED_STATE_COUNT];\
    KW_LOCAL_MEM REAL sPartials2[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];\
    int y = deltaPartialsByState + deltaPartialsByMatrix;\
    if (pattern < totalPatterns) {\
        sPartials2[patIdx][state] = partials2[y + state];\
    } else {\
        sPartials2[patIdx][state] = 0;\
    }\
    KW_GLOBAL_VAR REAL* KW_RESTRICT matrix2 = matrices2 + deltaMatrix;\
    REAL sum1 = 0, sum2 = 0;\
    if (pattern < totalPatterns) {\
        int state1 = states1[pattern]; /* Coalesced; no need to share */\
        KW_GLOBAL_VAR REAL* KW_RESTRICT matrix1 = matrices1 + deltaMatrix + state1 * PADDED_STATE_COUNT;\
        if (state1 < PADDED_STATE_COUNT)\
            sum1 = matrix1[state];\
        else\
            sum1 = 1.0;\
    }\
    for (int i = 0; i < PADDED_STATE_COUNT; i += BLOCK_PEELING_SIZE) {\
        if (patIdx < BLOCK_PEELING_SIZE) {\
            sMatrix2[patIdx][state] = matrix2[patIdx * PADDED_STATE_COUNT + state];\
            matrix2 += BLOCK_PEELING_SIZE * PADDED_STATE_COUNT;\
        }\
        KW_LOCAL_FENCE;\
        for(int j = 0; j < BLOCK_PEELING_SIZE; j++) {\
            FMA(sMatrix2[j][state], sPartials2[patIdx][i + j], sum2);\
        }\
        KW_LOCAL_FENCE;\
    }

#define LOAD_PARTIALS_SCALING_X_GPU()\
    int state = KW_LOCAL_ID_0;\
    int matrix = KW_LOCAL_ID_1;\
    int pattern = KW_GROUP_ID_0;\
    int patternCount = KW_NUM_GROUPS_0;\
    int offsetPartials = matrix * patternCount * PADDED_STATE_COUNT\
                         + pattern * PADDED_STATE_COUNT + state;\
    /* TODO: Currently assumes MATRIX_BLOCK_SIZE > matrixCount; FIX!!! */\
    KW_LOCAL_MEM REAL partials[MATRIX_BLOCK_SIZE][PADDED_STATE_COUNT];\
    KW_LOCAL_MEM REAL storedPartials[MATRIX_BLOCK_SIZE][PADDED_STATE_COUNT];\
    KW_LOCAL_MEM REAL max;\
    if (matrix < matrixCount)\
        partials[matrix][state] = allPartials[offsetPartials];\
    else\
        partials[matrix][state] = 0;\
    storedPartials[matrix][state] = partials[matrix][state];\
    KW_LOCAL_FENCE;

#define FIND_MAX_PARTIALS_STATE_POWER_OF_TWO_X_GPU()\
    /* parallelized reduction, only works for powers-of-2 */\
    for (int i = PADDED_STATE_COUNT / 2; i > 0; i >>= 1) {\
        if (state < i) {\
            REAL compare1 = partials[matrix][state];\
            REAL compare2 = partials[matrix][state + i];\
            if (compare2 > compare1)\
                partials[matrix][state] = compare2;\
        }\
        KW_LOCAL_FENCE;\
    }

#define FIND_MAX_PARTIALS_STATE_X_GPU()\
    /* not power-of-2 */\
    for (int i = SMALLEST_POWER_OF_TWO / 2; i > 0; i >>= 1) {\
        if (state < i && state + i < PADDED_STATE_COUNT ) {\
            REAL compare1 = partials[matrix][state];\
            REAL compare2 = partials[matrix][state + i];\
            if (compare2 > compare1)\
               partials[matrix][state] = compare2;\
        }\
        KW_LOCAL_FENCE;\
    }

#define FIND_MAX_PARTIALS_MATRIX_X_GPU()\
    max = 0;\
    for(int m = 0; m < matrixCount; m++) {\
        if (partials[m][0] > max)\
            max = partials[m][0];\
    }

#define SCALE_PARTIALS_X_GPU()\
    KW_LOCAL_FENCE;\
    if (matrix < matrixCount)\
        allPartials[offsetPartials] = storedPartials[matrix][state] / max;

#define INTEGRATE_PARTIALS_X_GPU()\
    int state   = KW_LOCAL_ID_0;\
    int pattern = KW_GROUP_ID_0;\
    KW_LOCAL_MEM REAL stateFreq[PADDED_STATE_COUNT];\
    /* TODO: Currently assumes MATRIX_BLOCK_SIZE >> matrixCount */\
    KW_LOCAL_MEM REAL matrixProp[MATRIX_BLOCK_SIZE];\
    KW_LOCAL_MEM REAL sum[PADDED_STATE_COUNT];\
    /* Load shared memory */\
    stateFreq[state] = dFrequencies[state];\
    sum[state] = 0;\
    for(int matrixEdge = 0; matrixEdge < matrixCount; matrixEdge += PADDED_STATE_COUNT) {\
        int x = matrixEdge + state;\
        if (x < matrixCount)\
            matrixProp[x] = dWeights[x];\
    }\
    KW_LOCAL_FENCE;\
    int u = state + pattern * PADDED_STATE_COUNT;\
    int delta = patternCount * PADDED_STATE_COUNT;\
    for(int r = 0; r < matrixCount; r++) {\
        FMA(dRootPartials[u + delta * r], matrixProp[r], sum[state]);\
    }\
    sum[state] *= stateFreq[state];\
    KW_LOCAL_FENCE;

#define INTEGRATE_PARTIALS_DERIV_X_GPU()\
    int state   = KW_LOCAL_ID_0;\
    int pattern = KW_GROUP_ID_0;\
    REAL tmpLogLike, tmpFirstDeriv;\
    KW_LOCAL_MEM REAL stateFreq[PADDED_STATE_COUNT];\
    KW_LOCAL_MEM REAL matrixProp[MATRIX_BLOCK_SIZE];\
    KW_LOCAL_MEM REAL sum[PADDED_STATE_COUNT];\
    KW_LOCAL_MEM REAL sumD1[PADDED_STATE_COUNT];\
    KW_LOCAL_MEM REAL sumD2[PADDED_STATE_COUNT];\
    stateFreq[state] = dFrequencies[state];\
    sum[state]   = 0;\
    sumD1[state] = 0;\
    sumD2[state] = 0;\
    for(int matrixEdge = 0; matrixEdge < matrixCount; matrixEdge += PADDED_STATE_COUNT) {\
        int x = matrixEdge + state;\
        if (x < matrixCount)\
            matrixProp[x] = dWeights[x];\
    }\
    KW_LOCAL_FENCE;\
    int u = state + pattern * PADDED_STATE_COUNT;\
    int delta = patternCount * PADDED_STATE_COUNT;\
    for(int r = 0; r < matrixCount; r++) {\
        FMA(dRootPartials[   u + delta * r], matrixProp[r], sum[state]  );\
        FMA(dRootFirstDeriv[ u + delta * r], matrixProp[r], sumD1[state]);\
        FMA(dRootSecondDeriv[u + delta * r], matrixProp[r], sumD2[state]);\
    }\
    sum[state]   *= stateFreq[state];\
    sumD1[state] *= stateFreq[state];\
    sumD2[state] *= stateFreq[state];\
    KW_LOCAL_FENCE;

#define SUM_STATES_POWER_OF_TWO_X_GPU()\
    /* parallelized reduction, only works for powers-of-2 */\
    for (int i = PADDED_STATE_COUNT / 2; i > 0; i >>= 1) {\
        if (state < i) {\
            sum[state] += sum[state + i];\
        }\
        KW_LOCAL_FENCE;\
    }

#define SUM_STATES_X_GPU()\
    /* not power-of-2 */\
    for (int i = SMALLEST_POWER_OF_TWO / 2; i > 0; i >>= 1) {\
        if (state < i && state + i < PADDED_STATE_COUNT ) {\
            sum[state] += sum[state + i];\
        }\
        KW_LOCAL_FENCE;\
    }

#define SUM_STATES_DERIVS_POWER_OF_TWO_X_GPU()\
    for (int i = PADDED_STATE_COUNT / 2; i > 0; i >>= 1) {\
        if (state < i) {\
            sum[state]   += sum[state + i];\
            sumD1[state] += sumD1[state + i];\
            sumD2[state] += sumD2[state + i];\
        }\
        KW_LOCAL_FENCE;\
    }

#define SUM_STATES_DERIVS_X_GPU()\
    for (int i = SMALLEST_POWER_OF_TWO / 2; i > 0; i >>= 1) {\
        if (state < i && state + i < PADDED_STATE_COUNT ) {\
            sum[state]   += sum[state + i];\
            sumD1[state] += sumD1[state + i];\
            sumD2[state] += sumD2[state + i];\
        }\
        KW_LOCAL_FENCE;\
    }

///////////////////////////////////////////////////////////////////////////////

//KW_GLOBAL_KERNEL void kernelPartialsPartialsEdgeFirstDerivatives(KW_GLOBAL_VAR REAL* KW_RESTRICT partials1,
//                                                                 KW_GLOBAL_VAR REAL* KW_RESTRICT partials2,
//                                                                 KW_GLOBAL_VAR REAL* KW_RESTRICT matrices1,
//                                                                 int totalPatterns, int categoryCount) {
//#ifdef FW_OPENCL_CPU // CPU/MIC implementation
//    // Not implemented
//#else // GPU implementation
////    DETERMINE_INDICES_X_GPU();
////
////    KW_GLOBAL_VAR REAL* KW_RESTRICT matrix1 = matrices1 + deltaMatrix; /* Points to *this* matrix */
////    KW_GLOBAL_VAR REAL* KW_RESTRICT matrix2 = matrices2 + deltaMatrix;
////
////    /* Load values into shared memory */
////    KW_LOCAL_MEM REAL sMatrix1[BLOCK_PEELING_SIZE][PADDED_STATE_COUNT];
////    KW_LOCAL_MEM REAL sMatrix2[BLOCK_PEELING_SIZE][PADDED_STATE_COUNT];
////
////    KW_LOCAL_MEM REAL sPartials1[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];
////    KW_LOCAL_MEM REAL sPartials2[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];
////
////    int y = deltaPartialsByState + deltaPartialsByMatrix;
////
////    /* copy PADDED_STATE_COUNT*PATTERN_BLOCK_SIZE lengthed partials */
////    /* These are all coherent global memory reads; checked in Profiler */
////    if (pattern < totalPatterns) {
////        sPartials1[patIdx][state] = partials1[y + state];
////        sPartials2[patIdx][state] = partials2[y + state];
////    } else {
////        sPartials1[patIdx][state] = 0;
////        sPartials2[patIdx][state] = 0;
////    }
////
////    REAL sum2 = 0;
////    for (int i = 0; i < PADDED_STATE_COUNT; i += BLOCK_PEELING_SIZE) {
////        /* load one row of matrices */
////        if (patIdx < BLOCK_PEELING_SIZE) {
////            /* These are all coherent global memory reads. */
////            sMatrix2[patIdx][state] = matrix2[patIdx * PADDED_STATE_COUNT + state];
////            /* sMatrix now filled with starting in state and ending in i */
////            matrix2 += BLOCK_PEELING_SIZE * PADDED_STATE_COUNT;
////        }
////
////        KW_LOCAL_FENCE;
////
////        for(int j = 0; j < BLOCK_PEELING_SIZE; j++) {
////            FMA(sMatrix2[j][state],  sPartials2[patIdx][i + j], sum2);
////        }
////
////        KW_LOCAL_FENCE;
////    }
////
////    sPartials1[patIdx][state] *= sum2;
////
////    KW_LOCAL_FENCE; // TODO Remove?
////
////    REAL sum1 = 0;
////    for (int i = 0; i < PADDED_STATE_COUNT; i += BLOCK_PEELING_SIZE) {
////        /* load one row of matrices */
////        if (patIdx < BLOCK_PEELING_SIZE) {
////            /* These are all coherent global memory reads. */
////            sMatrix1[patIdx][state] = matrix1[patIdx * PADDED_STATE_COUNT + state];
////            /* sMatrix now filled with starting in state and ending in i */
////            matrix1 += BLOCK_PEELING_SIZE * PADDED_STATE_COUNT;
////        }
////
////        KW_LOCAL_FENCE;
////
////        for(int j = 0; j < BLOCK_PEELING_SIZE; j++) {
////            FMA(sMatrix1[j][state],  sPartials1[patIdx][i + j], sum1);
////        }
////
////        KW_LOCAL_FENCE;
////    }
////
////    if (pattern < totalPatterns) {
////        partials3[u] = sum1;
////    }
//
//#endif
//}

KW_GLOBAL_KERNEL void kernelPartialsPartialsNoScale(KW_GLOBAL_VAR REAL* KW_RESTRICT partials1,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT partials2,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT partials3,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT matrices1,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT matrices2,
                                                    int totalPatterns) {
#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    DETERMINE_INDICES_X_CPU();
    SUM_PARTIALS_PARTIALS_X_CPU();
    partials3[u] = sum1 * sum2;
#else // GPU implementation
    DETERMINE_INDICES_X_GPU();
    SUM_PARTIALS_PARTIALS_X_GPU();
    if (pattern < totalPatterns)
        partials3[u] = sum1 * sum2;
#endif // FW_OPENCL_CPU
}

KW_GLOBAL_KERNEL void kernelPartialsPartialsFixedScale(KW_GLOBAL_VAR REAL* KW_RESTRICT partials1,
                                                       KW_GLOBAL_VAR REAL* KW_RESTRICT partials2,
                                                       KW_GLOBAL_VAR REAL* KW_RESTRICT partials3,
                                                       KW_GLOBAL_VAR REAL* KW_RESTRICT matrices1,
                                                       KW_GLOBAL_VAR REAL* KW_RESTRICT matrices2,
                                                       KW_GLOBAL_VAR REAL* KW_RESTRICT scalingFactors,
                                                       int totalPatterns) {
#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    DETERMINE_INDICES_X_CPU();
    SUM_PARTIALS_PARTIALS_X_CPU();
    partials3[u] = sum1 * sum2 / scalingFactors[pattern];
#else // GPU implementation
    DETERMINE_INDICES_X_GPU();
    LOAD_SCALING_X_GPU();
    SUM_PARTIALS_PARTIALS_X_GPU();
    if (pattern < totalPatterns)
        partials3[u] = sum1 * sum2 / fixedScalingFactors[patIdx];
#endif // FW_OPENCL_CPU
}

KW_GLOBAL_KERNEL void kernelStatesPartialsNoScale(KW_GLOBAL_VAR int* KW_RESTRICT states1,
                                                  KW_GLOBAL_VAR REAL* KW_RESTRICT partials2,
                                                  KW_GLOBAL_VAR REAL* KW_RESTRICT partials3,
                                                  KW_GLOBAL_VAR REAL* KW_RESTRICT matrices1,
                                                  KW_GLOBAL_VAR REAL* KW_RESTRICT matrices2,
                                                  int totalPatterns) {
#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    DETERMINE_INDICES_X_CPU();
    SUM_STATES_PARTIALS_X_CPU();
    partials3[u] = sum1 * sum2;
#else // GPU implementation
    DETERMINE_INDICES_X_GPU();
    SUM_STATES_PARTIALS_X_GPU();
    if (pattern < totalPatterns)
        partials3[u] = sum1 * sum2;
#endif // FW_OPENCL_CPU
}


KW_GLOBAL_KERNEL void kernelPartialsPartialsNoScaleTensorCores(KW_GLOBAL_VAR REAL* KW_RESTRICT partials1,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT partials2,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT partials3,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT matrices1,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT matrices2,
//                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT tmpAcc,
                                                    int totalPatterns) {
#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    DETERMINE_INDICES_X_CPU();
    SUM_PARTIALS_PARTIALS_X_CPU();
    partials3[u] = sum1 * sum2;
#else // GPU implementation
//    DETERMINE_INDICES_X_GPU();
//    SUM_PARTIALS_PARTIALS_X_GPU();
//    if (pattern < totalPatterns)
//        partials3[u] = sum1 * sum2;
// TODO: Change constant 8 patterns for all state counts
    const int NEW_PATTERN_BLOCK_SIZE = 8;
    DETERMINE_INDICES_X_GPU();
    int patternBlock = __umul24(KW_GROUP_ID_0,NEW_PATTERN_BLOCK_SIZE);

    KW_GLOBAL_VAR REAL* KW_RESTRICT matrix1 = matrices1 + deltaMatrix; /* Points to *this* matrix */
    KW_GLOBAL_VAR REAL* KW_RESTRICT matrix2 = matrices2 + deltaMatrix;
    /* Load values into shared memory */
    KW_LOCAL_MEM REAL sMatrix1[4 * PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sMatrix2[4 * PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sPartials1[PADDED_STATE_COUNT * NEW_PATTERN_BLOCK_SIZE];
    KW_LOCAL_MEM REAL sPartials2[PADDED_STATE_COUNT * NEW_PATTERN_BLOCK_SIZE];

    int y = patternBlock * PADDED_STATE_COUNT + deltaPartialsByMatrix;

    const int WMMA_M = 8;
    const int WMMA_N = 8;
    const int WMMA_K = 4;
    const int PATTERN_SPAN = NEW_PATTERN_BLOCK_SIZE/2;
    const int MEM_OFFSET = PATTERN_SPAN * PADDED_STATE_COUNT;

    int warpSize = 32; // TODO: Check if its better to get this from Cuda API
    int warpState = state/warpSize;
    int warpPattern = patIdx;
    int warpIdx = warpState + warpPattern * (PADDED_STATE_COUNT/warpSize);

    int sMatrixRow, partialsCol, sMatrixCol, partialsRow;

//   TODO: Declare right before usage
    double a1, b1, a2,b2, res11 = 0, res12 = 0, res21 = 0, res22 = 0;

    int partialsOffset = warpIdx % (NEW_PATTERN_BLOCK_SIZE / WMMA_N);

    // Indices to permute ShM for sMatrix
    // X -> threadIdx.x or state and Y -> threadIdx.y or patIdx
    // (int(X/8): Splits 32 values into groups of 8.
    // ((Y & 1) * -2 + 1)): For strip-mined layout: If patIdx is even increment by 1 else by -1
    // & 0x03 To cycle within the limits [0,1,2,3] i.e., [0, ... , PADDED_STATE_COUNT/WMMA_M]
#define GET_SMEM_ROW_SMATRIX(X) ((X / WMMA_K) & 0x03)
#define GET_BANK_GROUP_SMATRIX(X,Y) ((Y + (X/WMMA_K) * ((Y & 1) * -2 + 1)) & ((PADDED_STATE_COUNT/WMMA_K) - 1)) // 0x03 should be generalized to & PADDED_STATE_COUNT/WMMA_M - 1
#define GET_SMEM_COL_SMATRIX(X,Y) (GET_BANK_GROUP_SMATRIX(X,Y) * WMMA_K + (X % WMMA_K))
#define GET_SMEM_OFFSET_SMATRIX(X,Y) (GET_SMEM_ROW_SMATRIX(X) * PADDED_STATE_COUNT + GET_SMEM_COL_SMATRIX(X, Y))

    // Indices to permute ShM for partials
    // X -> threadIdx.x or state and Y -> threadIdx.y or patIdx
    // (int(X/8): Splits 32 values into groups of 4.
    // ((Y & 1) * -2 + 1)): For strip-mined layout: If patIdx is even increment by 1 else by -1
    // & 0x07 To cycle within the limits [0,1,2,3,4,5,6,7] i.e., [0, ... , PADDED_STATE_COUNT/WMMA_K]
#define GET_SMEM_ROW_PARTIALS(X) ((X / WMMA_K) & 0x07)
#define GET_BANK_GROUP_PARTIALS(X,Y) ((Y + (X/WMMA_K) * ((Y & 1) * -2 + 1)) & ((PADDED_STATE_COUNT/WMMA_K) - 1)) // 0x07 should be generalized to & PADDED_STATE_COUNT/WMMA_K - 1
#define GET_SMEM_COL_PARTIALS(X,Y) (GET_BANK_GROUP_PARTIALS(X,Y) * WMMA_K + (X % WMMA_K))
#define GET_SMEM_OFFSET_PARTIALS(X,Y) (GET_SMEM_ROW_PARTIALS(X) * PADDED_STATE_COUNT + GET_SMEM_COL_PARTIALS(X, Y))

    // Load PADDED_STATE_COUNT * PATTERN_BLOCK_SIZE partials
    if(pattern < totalPatterns) {
        sPartials1[GET_SMEM_OFFSET_PARTIALS(state, patIdx)] = partials1[y + patIdx * PADDED_STATE_COUNT + state];
        sPartials2[GET_SMEM_OFFSET_PARTIALS(state, patIdx)] = partials2[y + patIdx * PADDED_STATE_COUNT + state];
    } else {
        sPartials1[GET_SMEM_OFFSET_PARTIALS(state, patIdx)] = 0;
        sPartials2[GET_SMEM_OFFSET_PARTIALS(state, patIdx)] = 0;
    }

    int pattern_span = patIdx + 4;
    if(pattern + 4 < totalPatterns) {
        sPartials1[GET_SMEM_OFFSET_PARTIALS(state, pattern_span)] = partials1[y + (pattern_span) * PADDED_STATE_COUNT + state];
        sPartials2[GET_SMEM_OFFSET_PARTIALS(state, pattern_span)] = partials2[y + (pattern_span) * PADDED_STATE_COUNT + state];
    } else {
        sPartials1[GET_SMEM_OFFSET_PARTIALS(state, pattern_span)] = 0;
        sPartials2[GET_SMEM_OFFSET_PARTIALS(state, pattern_span)] = 0;
    }

//    tmpAcc[patIdx * PADDED_STATE_COUNT + state] = partials1[y + patIdx * PADDED_STATE_COUNT + state];

    for (int i = 0; i < PADDED_STATE_COUNT; i += WMMA_K) {
        sMatrixRow = warpIdx % (PADDED_STATE_COUNT / WMMA_M);
        sMatrixCol = i;
        partialsRow = i;
        partialsCol = warpIdx / (PADDED_STATE_COUNT / WMMA_M);

        sMatrix1[GET_SMEM_OFFSET_SMATRIX(state, patIdx)] = matrix1[sMatrixCol * PADDED_STATE_COUNT + patIdx * PADDED_STATE_COUNT + state];
        sMatrix2[GET_SMEM_OFFSET_SMATRIX(state, patIdx)] = matrix2[sMatrixCol * PADDED_STATE_COUNT + patIdx * PADDED_STATE_COUNT + state];

        KW_LOCAL_FENCE;

        // reg_row* and reg_col* are according to memory layout in ShM
        int laneid = state % 32;

        int reg_row = state % 4;
        int reg_col = (warpIdx * WMMA_M) + (laneid / 4);

        // TODO: More book keeping if PATTERN_BLOCK_SIZE > WMMA_M
        int reg_row_partials = laneid / 4;
        int reg_col_partials = partialsRow + state % 4;

        a1 = sMatrix1[GET_SMEM_OFFSET_SMATRIX(reg_col, reg_row)];
        b1 = sPartials1[GET_SMEM_OFFSET_PARTIALS(reg_col_partials, reg_row_partials)];

        asm("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
            : "=d"(res11), "=d"(res12)
            : "d"(a1), "d"(b1), "d"(res11), "d"(res12));

        a2 = sMatrix2[GET_SMEM_OFFSET_SMATRIX(reg_col, reg_row)];
        b2 = sPartials2[GET_SMEM_OFFSET_PARTIALS(reg_col_partials, reg_row_partials)];

        asm("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
            : "=d"(res21), "=d"(res22)
            : "d"(a2), "d"(b2), "d"(res21), "d"(res22));

        KW_LOCAL_FENCE;
    }

    u = patternBlock * PADDED_STATE_COUNT + deltaPartialsByMatrix;

    // TODO: ((state * 2) % 8) is 'pattern' within thread-block. Create better variables for readability
    if (patternBlock + ((state * 2) % 8) < totalPatterns)
        partials3[u + patIdx * (PADDED_STATE_COUNT/warpSize) * WMMA_M + ((state * 2) % 8) * PADDED_STATE_COUNT + (state * 2)/8] = res11 * res21;

    if (patternBlock + ((state * 2) % 8) + 1 < totalPatterns)
        partials3[u + PADDED_STATE_COUNT + patIdx * (PADDED_STATE_COUNT/warpSize) * WMMA_M + ((state * 2) % 8) * PADDED_STATE_COUNT + (state * 2)/8] = res12 * res22;

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
    DETERMINE_INDICES_X_CPU();
    SUM_STATES_PARTIALS_X_CPU();
    partials3[u] = sum1 * sum2 / scalingFactors[pattern];
#else // GPU implementation
    DETERMINE_INDICES_X_GPU();
    LOAD_SCALING_X_GPU();
    SUM_STATES_PARTIALS_X_GPU();
    if (pattern < totalPatterns)
        partials3[u] = sum1 * sum2 / fixedScalingFactors[patIdx];
#endif // FW_OPENCL_CPU
}

KW_GLOBAL_KERNEL void kernelStatesStatesNoScale(KW_GLOBAL_VAR int* KW_RESTRICT states1,
                                                KW_GLOBAL_VAR int* KW_RESTRICT states2,
                                                KW_GLOBAL_VAR REAL* KW_RESTRICT partials3,
                                                KW_GLOBAL_VAR REAL* KW_RESTRICT matrices1,
                                                KW_GLOBAL_VAR REAL* KW_RESTRICT matrices2,
                                                int totalPatterns) {
#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    DETERMINE_INDICES_X_CPU();
    int state1 = states1[pattern];
    int state2 = states2[pattern];
    KW_GLOBAL_VAR REAL* KW_RESTRICT matrix1 = matrices1 + deltaMatrix + state1 * PADDED_STATE_COUNT;
    KW_GLOBAL_VAR REAL* KW_RESTRICT matrix2 = matrices2 + deltaMatrix + state2 * PADDED_STATE_COUNT;
    if (state1 < PADDED_STATE_COUNT && state2 < PADDED_STATE_COUNT) {
        partials3[u] = matrix1[state] * matrix2[state];
    } else if (state1 < PADDED_STATE_COUNT) {
        partials3[u] = matrix1[state];
    } else if (state2 < PADDED_STATE_COUNT) {
        partials3[u] = matrix2[state];
    } else {
        partials3[u] = 1.0;
    }
#else // GPU implementation
    DETERMINE_INDICES_X_GPU();
    int state1 = states1[pattern];
    int state2 = states2[pattern];
    KW_GLOBAL_VAR REAL* KW_RESTRICT matrix1 = matrices1 + deltaMatrix + state1 * PADDED_STATE_COUNT;
    KW_GLOBAL_VAR REAL* KW_RESTRICT matrix2 = matrices2 + deltaMatrix + state2 * PADDED_STATE_COUNT;
    if (pattern < totalPatterns) {
        if (state1 < PADDED_STATE_COUNT && state2 < PADDED_STATE_COUNT) {
            partials3[u] = matrix1[state] * matrix2[state];
        } else if (state1 < PADDED_STATE_COUNT) {
            partials3[u] = matrix1[state];
        } else if (state2 < PADDED_STATE_COUNT) {
            partials3[u] = matrix2[state];
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
    DETERMINE_INDICES_X_CPU();
    int state1 = states1[pattern];
    int state2 = states2[pattern];
    KW_GLOBAL_VAR REAL* KW_RESTRICT matrix1 = matrices1 + deltaMatrix + state1 * PADDED_STATE_COUNT;
    KW_GLOBAL_VAR REAL* KW_RESTRICT matrix2 = matrices2 + deltaMatrix + state2 * PADDED_STATE_COUNT;
    if (state1 < PADDED_STATE_COUNT && state2 < PADDED_STATE_COUNT) {
        partials3[u] = matrix1[state] * matrix2[state] / scalingFactors[pattern];
    } else if (state1 < PADDED_STATE_COUNT) {
        partials3[u] = matrix1[state] / scalingFactors[pattern];
    } else if (state2 < PADDED_STATE_COUNT) {
        partials3[u] = matrix2[state] / scalingFactors[pattern];
    } else {
        partials3[u] = 1.0 / scalingFactors[pattern];
    }
#else // GPU implementation
    DETERMINE_INDICES_X_GPU();
    int state1 = states1[pattern];
    int state2 = states2[pattern];
    KW_GLOBAL_VAR REAL* KW_RESTRICT matrix1 = matrices1 + deltaMatrix + state1 * PADDED_STATE_COUNT;
    KW_GLOBAL_VAR REAL* KW_RESTRICT matrix2 = matrices2 + deltaMatrix + state2 * PADDED_STATE_COUNT;
    LOAD_SCALING_X_GPU();
    KW_LOCAL_FENCE;
    if (pattern < totalPatterns) {
        if (state1 < PADDED_STATE_COUNT && state2 < PADDED_STATE_COUNT) {
            partials3[u] = matrix1[state] * matrix2[state] / fixedScalingFactors[patIdx];
        } else if (state1 < PADDED_STATE_COUNT) {
            partials3[u] = matrix1[state] / fixedScalingFactors[patIdx];
        } else if (state2 < PADDED_STATE_COUNT) {
            partials3[u] = matrix2[state] / fixedScalingFactors[patIdx];
        } else {
            partials3[u] = 1.0 / fixedScalingFactors[patIdx];
        }
    }
#endif // FW_OPENCL_CPU
}

// Find a scaling factor for each pattern
KW_GLOBAL_KERNEL void kernelPartialsDynamicScaling(KW_GLOBAL_VAR REAL* KW_RESTRICT allPartials,
                                                   KW_GLOBAL_VAR REAL* KW_RESTRICT scalingFactors,
                                                   int matrixCount) {
#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    FIND_MAX_PARTIALS_X_CPU();
    if (max == 0)
        max = 1.0;
    scalingFactors[pattern] = max;
    SCALE_PARTIALS_X_CPU();
#else // GPU implementation
    LOAD_PARTIALS_SCALING_X_GPU();
#ifdef IS_POWER_OF_TWO
    FIND_MAX_PARTIALS_STATE_POWER_OF_TWO_X_GPU();
#else // not power-of-2
    FIND_MAX_PARTIALS_STATE_X_GPU();
#endif // IS_POWER_OF_TWO
    if (state == 0 && matrix == 0) {
        FIND_MAX_PARTIALS_MATRIX_X_GPU();
        if (max == 0)
        	max = 1.0;
        scalingFactors[pattern] = max; // TODO: These are incoherent memory writes!!!
    }
    SCALE_PARTIALS_X_GPU();
#endif // FW_OPENCL_CPU
}

KW_GLOBAL_KERNEL void kernelPartialsDynamicScalingScalersLog(KW_GLOBAL_VAR REAL* KW_RESTRICT allPartials,
                                                             KW_GLOBAL_VAR REAL* KW_RESTRICT scalingFactors,
                                                             int matrixCount) {
#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    FIND_MAX_PARTIALS_X_CPU();
    if (max == 0) {
        max = 1.0;
        scalingFactors[pattern] = 0.0;
    } else {
        scalingFactors[pattern] = log(max);
    }
    SCALE_PARTIALS_X_CPU();
#else // GPU implementation
    LOAD_PARTIALS_SCALING_X_GPU();
#ifdef IS_POWER_OF_TWO
    FIND_MAX_PARTIALS_STATE_POWER_OF_TWO_X_GPU();
#else // not power-of-2
    FIND_MAX_PARTIALS_STATE_X_GPU();
#endif // IS_POWER_OF_TWO
    if (state == 0 && matrix == 0) {
        FIND_MAX_PARTIALS_MATRIX_X_GPU();
        if (max == 0) {
            max = 1.0;
            scalingFactors[pattern] = 0.0;
        } else {
            scalingFactors[pattern] = log(max);
        }
    }
    SCALE_PARTIALS_X_GPU();
#endif // FW_OPENCL_CPU
}



// Find a scaling factor for each pattern and accumulate into buffer
KW_GLOBAL_KERNEL void kernelPartialsDynamicScalingAccumulate(KW_GLOBAL_VAR REAL* KW_RESTRICT allPartials,
                                                             KW_GLOBAL_VAR REAL* KW_RESTRICT scalingFactors,
                                                             KW_GLOBAL_VAR REAL* KW_RESTRICT cumulativeScaling,
                                                             int matrixCount) {
#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    FIND_MAX_PARTIALS_X_CPU();
    if (max == 0)
        max = 1.0;
    scalingFactors[pattern] = max;
    cumulativeScaling[pattern] += log(max);
    SCALE_PARTIALS_X_CPU();
#else // GPU implementation
    LOAD_PARTIALS_SCALING_X_GPU();
#ifdef IS_POWER_OF_TWO
    FIND_MAX_PARTIALS_STATE_POWER_OF_TWO_X_GPU();
#else // not power-of-2
    FIND_MAX_PARTIALS_STATE_X_GPU();
#endif // IS_POWER_OF_TWO
    if (state == 0 && matrix == 0) {
        FIND_MAX_PARTIALS_MATRIX_X_GPU();
        if (max == 0)
            max = 1.0;
        scalingFactors[pattern] = max;
        #ifdef CUDA
            atomicAdd(&cumulativeScaling[pattern], log(max));
        #else
            cumulativeScaling[pattern] += log(max);
        #endif
    }
    SCALE_PARTIALS_X_GPU();
#endif // FW_OPENCL_CPU
}

KW_GLOBAL_KERNEL void kernelPartialsDynamicScalingAccumulateScalersLog(KW_GLOBAL_VAR REAL* KW_RESTRICT allPartials,
                                                                       KW_GLOBAL_VAR REAL* KW_RESTRICT scalingFactors,
                                                                       KW_GLOBAL_VAR REAL* KW_RESTRICT cumulativeScaling,
                                                                       int matrixCount) {
#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    FIND_MAX_PARTIALS_X_CPU();
    if (max == 0) {
        max = 1.0;
        scalingFactors[pattern] = 0.0;
    } else {
        REAL logMax = log(max);
        scalingFactors[pattern] = logMax;
        cumulativeScaling[pattern] += logMax;
    }
    SCALE_PARTIALS_X_CPU();
#else // GPU implementation
    LOAD_PARTIALS_SCALING_X_GPU();
#ifdef IS_POWER_OF_TWO
    FIND_MAX_PARTIALS_STATE_POWER_OF_TWO_X_GPU();
#else // not power-of-2
    FIND_MAX_PARTIALS_STATE_X_GPU();
#endif // IS_POWER_OF_TWO
    if (state == 0 && matrix == 0) {
        FIND_MAX_PARTIALS_MATRIX_X_GPU();
        if (max == 0) {
            max = 1.0;
            scalingFactors[pattern] = 0.0;
        } else {
            REAL logMax = log(max);
            scalingFactors[pattern] = logMax;
            #ifdef CUDA
                atomicAdd(&cumulativeScaling[pattern], logMax);
            #else
                cumulativeScaling[pattern] += logMax;
            #endif
        }
    }
    SCALE_PARTIALS_X_GPU();
#endif // FW_OPENCL_CPU
}

KW_GLOBAL_KERNEL void kernelIntegrateLikelihoods(KW_GLOBAL_VAR REAL* KW_RESTRICT dResult,
                                                 KW_GLOBAL_VAR REAL* KW_RESTRICT dRootPartials,
                                                 KW_GLOBAL_VAR REAL* KW_RESTRICT dWeights,
                                                 KW_GLOBAL_VAR REAL* KW_RESTRICT dFrequencies,
                                                 int matrixCount,
                                                 int patternCount) {
#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    INTEGRATE_PARTIALS_X_CPU();
    dResult[pattern] = log(sumTotal);
#else // GPU implementation
    INTEGRATE_PARTIALS_X_GPU();
    #ifdef IS_POWER_OF_TWO
        SUM_STATES_POWER_OF_TWO_X_GPU();
    #else // not power-of-2
        SUM_STATES_X_GPU();
    #endif // IS_POWER_OF_TWO
    if (state == 0)
        dResult[pattern] = log(sum[state]);
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
    INTEGRATE_PARTIALS_X_CPU();
    dResult[pattern] = log(sumTotal) + dRootScalingFactors[pattern];
#else // GPU implementation
    INTEGRATE_PARTIALS_X_GPU();
    #ifdef IS_POWER_OF_TWO
        SUM_STATES_POWER_OF_TWO_X_GPU();
    #else // not power-of-2
        SUM_STATES_X_GPU();
    #endif // IS_POWER_OF_TWO
    if (state == 0)
        dResult[pattern] = log(sum[state]) + dRootScalingFactors[pattern];
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
    INTEGRATE_PARTIALS_X_CPU();
    if (takeLog == 0)
        dResult[pattern] = sumTotal;
    else if (takeLog == 1)
        dResult[pattern] = log(dResult[pattern] + sumTotal);
    else
        dResult[pattern] += sumTotal;
#else // GPU implementation
    INTEGRATE_PARTIALS_X_GPU();
    #ifdef IS_POWER_OF_TWO
        SUM_STATES_POWER_OF_TWO_X_GPU();
    #else // not power-of-2
        SUM_STATES_X_GPU();
    #endif // IS_POWER_OF_TWO
    if (state == 0) {
        if (takeLog == 0)
            dResult[pattern] = sum[state];
        else if (takeLog == 1)
            dResult[pattern] = log(dResult[pattern] + sum[state]);
        else
            dResult[pattern] += sum[state];
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
    INTEGRATE_PARTIALS_X_CPU();
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
    INTEGRATE_PARTIALS_X_GPU();
    #ifdef IS_POWER_OF_TWO
        SUM_STATES_POWER_OF_TWO_X_GPU();
    #else // not power-of-2
        SUM_STATES_X_GPU();
    #endif // IS_POWER_OF_TWO
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
            sum[state] *= exp((REAL)(cumulativeScalingFactor - maxScalingFactor));
        if (state == 0)
            dResult[pattern] = sum[state];
        KW_LOCAL_FENCE;
    } else {
        if (subsetIndex != dIndexMaxScalingFactors[pattern])
            sum[state] *= exp((REAL)(cumulativeScalingFactor - dMaxScalingFactors[pattern]));
        if (state == 0) {
            if (subsetIndex == subsetCount - 1)
                dResult[pattern] = (log(dResult[pattern] + sum[state]) + dMaxScalingFactors[pattern]);
            else
                dResult[pattern] += sum[state];
        }
    }
#endif // FW_OPENCL_CPU
}

////////////////////////////////////////////////////////////////////////////////////////////////
// edge and deriv kernels

KW_GLOBAL_KERNEL void kernelPartialsPartialsEdgeLikelihoods(KW_GLOBAL_VAR REAL* KW_RESTRICT dPartialsTmp,
                                                            KW_GLOBAL_VAR REAL* KW_RESTRICT dParentPartials,
                                                            KW_GLOBAL_VAR REAL* KW_RESTRICT dChildParials,
                                                            KW_GLOBAL_VAR REAL* KW_RESTRICT dTransMatrix,
                                                            int totalPatterns) {

#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    DETERMINE_INDICES_X_CPU();
    int deltaPartials = deltaPartialsByMatrix + deltaPartialsByState;
    KW_GLOBAL_VAR REAL* KW_RESTRICT sMatrix1 = dTransMatrix + deltaMatrix;
    KW_GLOBAL_VAR REAL* KW_RESTRICT sPartials1 = dParentPartials + deltaPartials;
    KW_GLOBAL_VAR REAL* KW_RESTRICT sPartials2 = dChildParials + deltaPartials;
    REAL sum1 = 0;
    for(int i = 0; i < PADDED_STATE_COUNT; i++) {
        FMA(sMatrix1[i * PADDED_STATE_COUNT + state],  sPartials1[i], sum1);
    }
    dPartialsTmp[u] = sum1 * sPartials2[state];
#else // GPU implementation
    DETERMINE_INDICES_X_GPU();
    KW_GLOBAL_VAR REAL* KW_RESTRICT matrix1 = dTransMatrix + deltaMatrix;
    int y = deltaPartialsByState + deltaPartialsByMatrix;
    KW_LOCAL_MEM REAL sMatrix1[BLOCK_PEELING_SIZE][PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sPartials1[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sPartials2[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];
    if (pattern < totalPatterns) {
        sPartials1[patIdx][state] = dParentPartials[y + state];
        sPartials2[patIdx][state] = dChildParials[y + state];
    } else {
        sPartials1[patIdx][state] = 0;
        sPartials2[patIdx][state] = 0;
    }
    REAL sum1 = 0;
    int i;
    for (i = 0; i < PADDED_STATE_COUNT; i += BLOCK_PEELING_SIZE) {
        if (patIdx < BLOCK_PEELING_SIZE) {
            sMatrix1[patIdx][state] = matrix1[patIdx * PADDED_STATE_COUNT + state];
            matrix1 += BLOCK_PEELING_SIZE * PADDED_STATE_COUNT;
        }
        KW_LOCAL_FENCE;
        int j;
        for(j = 0; j < BLOCK_PEELING_SIZE; j++) {
            FMA(sMatrix1[j][state], sPartials1[patIdx][i + j], sum1);
        }
        KW_LOCAL_FENCE;
    }
    if (pattern < totalPatterns)
        dPartialsTmp[u] = sum1 * sPartials2[patIdx][state];
#endif // FW_OPENCL_CPU
}


KW_GLOBAL_KERNEL void
#ifdef CUDA
__launch_bounds__(PATTERN_BLOCK_SIZE * PADDED_STATE_COUNT)
#endif
kernelPartialsPartialsEdgeLikelihoodsSecondDeriv(KW_GLOBAL_VAR REAL* KW_RESTRICT dPartialsTmp,
                                                 KW_GLOBAL_VAR REAL* KW_RESTRICT dFirstDerivTmp,
                                                 KW_GLOBAL_VAR REAL* KW_RESTRICT dSecondDerivTmp,
                                                 KW_GLOBAL_VAR REAL* KW_RESTRICT dParentPartials,
                                                 KW_GLOBAL_VAR REAL* KW_RESTRICT dChildParials,
                                                 KW_GLOBAL_VAR REAL* KW_RESTRICT dTransMatrix,
                                                 KW_GLOBAL_VAR REAL* KW_RESTRICT dFirstDerivMatrix,
                                                 KW_GLOBAL_VAR REAL* KW_RESTRICT dSecondDerivMatrix,
                                                 int totalPatterns) {
#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    DETERMINE_INDICES_X_CPU();
    int deltaPartials = deltaPartialsByMatrix + deltaPartialsByState;
    KW_GLOBAL_VAR REAL* KW_RESTRICT sMatrix1 = dTransMatrix + deltaMatrix;
    KW_GLOBAL_VAR REAL* KW_RESTRICT sMatrixFirstDeriv = dFirstDerivMatrix + deltaMatrix;
    KW_GLOBAL_VAR REAL* KW_RESTRICT sMatrixSecondDeriv = dSecondDerivMatrix + deltaMatrix;
    KW_GLOBAL_VAR REAL* KW_RESTRICT sPartials1 = dParentPartials + deltaPartials;
    KW_GLOBAL_VAR REAL* KW_RESTRICT sPartials2 = dChildParials + deltaPartials;
    REAL sum1 = 0;
    REAL sumFirstDeriv = 0;
    REAL sumSecondDeriv = 0;
    for(int i = 0; i < PADDED_STATE_COUNT; i++) {
        FMA(sMatrix1[          i * PADDED_STATE_COUNT + state], sPartials1[i], sum1);
        FMA(sMatrixFirstDeriv[ i * PADDED_STATE_COUNT + state], sPartials1[i], sumFirstDeriv);
        FMA(sMatrixSecondDeriv[i * PADDED_STATE_COUNT + state], sPartials1[i], sumSecondDeriv);
    }
    dPartialsTmp[u]    = sum1           * sPartials2[state];
    dFirstDerivTmp[u]  = sumFirstDeriv  * sPartials2[state];
    dSecondDerivTmp[u] = sumSecondDeriv * sPartials2[state];
#else // GPU implementation
    DETERMINE_INDICES_X_GPU();
    KW_GLOBAL_VAR REAL* KW_RESTRICT matrix1 = dTransMatrix + deltaMatrix; // Points to *this* matrix
    KW_GLOBAL_VAR REAL* KW_RESTRICT matrixFirstDeriv = dFirstDerivMatrix + deltaMatrix;
    KW_GLOBAL_VAR REAL* KW_RESTRICT matrixSecondDeriv = dSecondDerivMatrix + deltaMatrix;
    int y = deltaPartialsByState + deltaPartialsByMatrix;
    KW_LOCAL_MEM REAL sMatrix1[BLOCK_PEELING_SIZE/2][PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sMatrixFirstDeriv[BLOCK_PEELING_SIZE/2][PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sMatrixSecondDeriv[BLOCK_PEELING_SIZE/2][PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sPartials1[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sPartials2[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];
    if (pattern < totalPatterns) {
        sPartials1[patIdx][state] = dParentPartials[y + state];
        sPartials2[patIdx][state] = dChildParials[y + state];
    } else {
        sPartials1[patIdx][state] = 0;
        sPartials2[patIdx][state] = 0;
    }
    REAL sum1 = 0;
    REAL sumFirstDeriv = 0;
    REAL sumSecondDeriv = 0;
    int i;
    for (i = 0; i < PADDED_STATE_COUNT; i += BLOCK_PEELING_SIZE/2) {
        if (patIdx < BLOCK_PEELING_SIZE/2) {
            sMatrix1[patIdx][state] = matrix1[patIdx * PADDED_STATE_COUNT + state];
            sMatrixFirstDeriv[patIdx][state] = matrixFirstDeriv[patIdx * PADDED_STATE_COUNT + state];
            sMatrixSecondDeriv[patIdx][state] = matrixSecondDeriv[patIdx * PADDED_STATE_COUNT + state];
            matrix1 += BLOCK_PEELING_SIZE/2 * PADDED_STATE_COUNT;
            matrixFirstDeriv += BLOCK_PEELING_SIZE/2 * PADDED_STATE_COUNT;
            matrixSecondDeriv += BLOCK_PEELING_SIZE/2 * PADDED_STATE_COUNT;
        }
        KW_LOCAL_FENCE;
        int j;
        for(j = 0; j < BLOCK_PEELING_SIZE/2; j++) {
            FMA(sMatrix1[j][state]          , sPartials1[patIdx][i + j], sum1          );
            FMA(sMatrixFirstDeriv[j][state] , sPartials1[patIdx][i + j], sumFirstDeriv );
            FMA(sMatrixSecondDeriv[j][state], sPartials1[patIdx][i + j], sumSecondDeriv);
        }
        KW_LOCAL_FENCE;
    }
    if (pattern < totalPatterns) {
        dPartialsTmp[u] = sum1 * sPartials2[patIdx][state];
        dFirstDerivTmp[u] = sumFirstDeriv * sPartials2[patIdx][state];
        dSecondDerivTmp[u] = sumSecondDeriv * sPartials2[patIdx][state];
    }
#endif // FW_OPENCL_CPU
}

KW_GLOBAL_KERNEL void kernelStatesPartialsEdgeLikelihoods(KW_GLOBAL_VAR REAL* KW_RESTRICT dPartialsTmp,
                                                          KW_GLOBAL_VAR REAL* KW_RESTRICT dParentPartials,
                                                          KW_GLOBAL_VAR int* KW_RESTRICT dChildStates,
                                                          KW_GLOBAL_VAR REAL* KW_RESTRICT dTransMatrix,
                                                          int totalPatterns) {
#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    DETERMINE_INDICES_X_CPU();
    int deltaPartials = deltaPartialsByMatrix + deltaPartialsByState;
    KW_GLOBAL_VAR REAL* KW_RESTRICT sMatrix1 = dTransMatrix + deltaMatrix;
    KW_GLOBAL_VAR REAL* KW_RESTRICT sPartials2 = dParentPartials + deltaPartials;
    REAL sum1 = 0;
    int state1 = dChildStates[pattern];
    if (state1 < PADDED_STATE_COUNT)
        sum1 = sMatrix1[state1 * PADDED_STATE_COUNT + state];
    else
        sum1 = 1.0;
    dPartialsTmp[u] = sum1 * sPartials2[state];
#else // GPU implementation
    DETERMINE_INDICES_X_GPU();
    int y = deltaPartialsByState + deltaPartialsByMatrix;
    KW_LOCAL_MEM REAL sPartials2[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];
    if (pattern < totalPatterns) {
        sPartials2[patIdx][state] = dParentPartials[y + state];
    } else {
        sPartials2[patIdx][state] = 0;
    }
    REAL sum1 = 0;
    if (pattern < totalPatterns) {
        int state1 = dChildStates[pattern];
        KW_GLOBAL_VAR REAL* KW_RESTRICT matrix1 = dTransMatrix + deltaMatrix + state1 * PADDED_STATE_COUNT;
        if (state1 < PADDED_STATE_COUNT)
            sum1 = matrix1[state];
        else
            sum1 = 1.0;
    }
    if (pattern < totalPatterns)
        dPartialsTmp[u] = sum1 * sPartials2[patIdx][state];
#endif // FW_OPENCL_CPU
}

KW_GLOBAL_KERNEL void kernelStatesPartialsEdgeLikelihoodsSecondDeriv(KW_GLOBAL_VAR REAL* KW_RESTRICT dPartialsTmp,
                                                                     KW_GLOBAL_VAR REAL* KW_RESTRICT dFirstDerivTmp,
                                                                     KW_GLOBAL_VAR REAL* KW_RESTRICT dSecondDerivTmp,
                                                                     KW_GLOBAL_VAR REAL* KW_RESTRICT dParentPartials,
                                                                     KW_GLOBAL_VAR int* KW_RESTRICT dChildStates,
                                                                     KW_GLOBAL_VAR REAL* KW_RESTRICT dTransMatrix,
                                                                     KW_GLOBAL_VAR REAL* KW_RESTRICT dFirstDerivMatrix,
                                                                     KW_GLOBAL_VAR REAL* KW_RESTRICT dSecondDerivMatrix,
                                                                     int totalPatterns) {
#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    DETERMINE_INDICES_X_CPU();
    int deltaPartials = deltaPartialsByMatrix + deltaPartialsByState;
    KW_GLOBAL_VAR REAL* KW_RESTRICT sMatrix1 = dTransMatrix + deltaMatrix;
    KW_GLOBAL_VAR REAL* KW_RESTRICT sMatrixFirstDeriv = dFirstDerivMatrix + deltaMatrix;
    KW_GLOBAL_VAR REAL* KW_RESTRICT sMatrixSecondDeriv = dSecondDerivMatrix + deltaMatrix;
    KW_GLOBAL_VAR REAL* KW_RESTRICT sPartials2 = dParentPartials + deltaPartials;
    REAL sum1 = 0;
    REAL sumFirstDeriv = 0;
    REAL sumSecondDeriv = 0;
    int state1 = dChildStates[pattern];
    if (state1 < PADDED_STATE_COUNT) {
        sum1           = sMatrix1[          state1 * PADDED_STATE_COUNT + state];
        sumFirstDeriv  = sMatrixFirstDeriv[ state1 * PADDED_STATE_COUNT + state];
        sumSecondDeriv = sMatrixSecondDeriv[state1 * PADDED_STATE_COUNT + state];
    } else {
        sum1 = 1.0;
    }
    dPartialsTmp[u]    = sum1           * sPartials2[state];
    dFirstDerivTmp[u]  = sumFirstDeriv  * sPartials2[state];
    dSecondDerivTmp[u] = sumSecondDeriv * sPartials2[state];
#else // GPU implementation
    DETERMINE_INDICES_X_GPU();
    int y = deltaPartialsByState + deltaPartialsByMatrix;
    KW_LOCAL_MEM REAL sPartials2[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];
    if (pattern < totalPatterns) {
        sPartials2[patIdx][state] = dParentPartials[y + state];
    } else {
        sPartials2[patIdx][state] = 0;
    }
    REAL sum1 = 0;
    REAL sumFirstDeriv = 0;
    REAL sumSecondDeriv = 0;
    if (pattern < totalPatterns) {
        int state1 = dChildStates[pattern]; // Coalesced; no need to share
        KW_GLOBAL_VAR REAL* KW_RESTRICT matrix1 = dTransMatrix + deltaMatrix + state1 * PADDED_STATE_COUNT;
        KW_GLOBAL_VAR REAL* KW_RESTRICT matrixFirstDeriv = dFirstDerivMatrix + deltaMatrix + state1 * PADDED_STATE_COUNT;
        KW_GLOBAL_VAR REAL* KW_RESTRICT matrixSecondDeriv = dSecondDerivMatrix + deltaMatrix + state1 * PADDED_STATE_COUNT;
        if (state1 < PADDED_STATE_COUNT) {
            sum1 = matrix1[state];
            sumFirstDeriv = matrixFirstDeriv[state];
            sumSecondDeriv = matrixSecondDeriv[state];
        } else {
            sum1 = 1.0;
            sumFirstDeriv = 0.0;
            sumSecondDeriv = 0.0;
        }
    }
    if (pattern < totalPatterns) {
        dPartialsTmp[u] = sum1 * sPartials2[patIdx][state];
        dFirstDerivTmp[u] = sumFirstDeriv * sPartials2[patIdx][state];
        dSecondDerivTmp[u] = sumSecondDeriv * sPartials2[patIdx][state];
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
    INTEGRATE_PARTIALS_DERIV_X_CPU();
    tmpLogLike = sumTotal;
    dResult[pattern] = log(tmpLogLike);
    tmpFirstDeriv = sumTotalD1 / tmpLogLike;
    dFirstDerivResult[pattern] = tmpFirstDeriv;
    dSecondDerivResult[pattern] = (sumTotalD2 / tmpLogLike - tmpFirstDeriv * tmpFirstDeriv);
#else // GPU implementation
    INTEGRATE_PARTIALS_DERIV_X_GPU();
#ifdef IS_POWER_OF_TWO
    SUM_STATES_DERIVS_POWER_OF_TWO_X_GPU();
#else // not power-of-2
    SUM_STATES_DERIVS_X_GPU();
#endif // IS_POWER_OF_TWO
    if (state == 0) {
        tmpLogLike = sum[state];
        dResult[pattern] = log(tmpLogLike);
        tmpFirstDeriv = sumD1[state] / tmpLogLike;
        dFirstDerivResult[pattern] = tmpFirstDeriv;
        dSecondDerivResult[pattern] = (sumD2[state] / tmpLogLike - tmpFirstDeriv * tmpFirstDeriv);
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
    INTEGRATE_PARTIALS_DERIV_X_CPU();
    tmpLogLike = sumTotal;
    dResult[pattern] = log(tmpLogLike) + dRootScalingFactors[pattern];
    tmpFirstDeriv = sumTotalD1 / tmpLogLike;
    dFirstDerivResult[pattern] = tmpFirstDeriv;
    dSecondDerivResult[pattern] = (sumTotalD2 / tmpLogLike - tmpFirstDeriv * tmpFirstDeriv);
#else // GPU implementation
    INTEGRATE_PARTIALS_DERIV_X_GPU();
#ifdef IS_POWER_OF_TWO
    SUM_STATES_DERIVS_POWER_OF_TWO_X_GPU();
#else // not power-of-2
    SUM_STATES_DERIVS_X_GPU();
#endif // IS_POWER_OF_TWO
    if (state == 0) {
        tmpLogLike = sum[state];
        dResult[pattern] = log(tmpLogLike) + dRootScalingFactors[pattern];
        tmpFirstDeriv = sumD1[state] / tmpLogLike;
        dFirstDerivResult[pattern] = tmpFirstDeriv;
        dSecondDerivResult[pattern] = (sumD2[state] / tmpLogLike - tmpFirstDeriv * tmpFirstDeriv);
    }
#endif // FW_OPENCL_CPU
}


////////////////////////////////////////////////////////////////////////////////////////////////
// scaling experiments kernels

KW_GLOBAL_KERNEL void kernelPartialsPartialsAutoScale(KW_GLOBAL_VAR REAL* partials1,
                                                             KW_GLOBAL_VAR REAL* partials2,
                                                             KW_GLOBAL_VAR REAL* partials3,
                                                             KW_GLOBAL_VAR REAL* matrices1,
                                                             KW_GLOBAL_VAR REAL* matrices2,
                                                             KW_GLOBAL_VAR signed char* scalingFactors,
                                                             int totalPatterns) {
    REAL sum1 = 0;
    REAL sum2 = 0;
    int i;

    DETERMINE_INDICES_X_GPU();

    KW_GLOBAL_VAR REAL* matrix1 = matrices1 + deltaMatrix; // Points to *this* matrix
    KW_GLOBAL_VAR REAL* matrix2 = matrices2 + deltaMatrix;

    int y = deltaPartialsByState + deltaPartialsByMatrix;

    // Load values into shared memory
    KW_LOCAL_MEM REAL sMatrix1[BLOCK_PEELING_SIZE][PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sMatrix2[BLOCK_PEELING_SIZE][PADDED_STATE_COUNT];

    KW_LOCAL_MEM REAL sPartials1[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sPartials2[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];

    // copy PADDED_STATE_COUNT*PATTERN_BLOCK_SIZE lengthed partials
    if (pattern < totalPatterns) {
        // These are all coherent global memory reads; checked in Profiler
        sPartials1[patIdx][state] = partials1[y + state];
        sPartials2[patIdx][state] = partials2[y + state];
    } else {
        sPartials1[patIdx][state] = 0;
        sPartials2[patIdx][state] = 0;
    }

    for (i = 0; i < PADDED_STATE_COUNT; i += BLOCK_PEELING_SIZE) {
        // load one row of matrices
        if (patIdx < BLOCK_PEELING_SIZE) {
            // These are all coherent global memory reads.
            sMatrix1[patIdx][state] = matrix1[patIdx * PADDED_STATE_COUNT + state];
            sMatrix2[patIdx][state] = matrix2[patIdx * PADDED_STATE_COUNT + state];

            // sMatrix now filled with starting in state and ending in i
            matrix1 += BLOCK_PEELING_SIZE * PADDED_STATE_COUNT;
            matrix2 += BLOCK_PEELING_SIZE * PADDED_STATE_COUNT;
        }
        KW_LOCAL_FENCE;

        int j;
        for(j = 0; j < BLOCK_PEELING_SIZE; j++) {
            sum1 += sMatrix1[j][state] * sPartials1[patIdx][i + j];
            sum2 += sMatrix2[j][state] * sPartials2[patIdx][i + j];
        }

        KW_LOCAL_FENCE; // GTX280 FIX HERE

    }

    REAL tmpPartial = sum1 * sum2;
    int expTmp;
    REAL sigTmp = frexp(tmpPartial, &expTmp);

    if (pattern < totalPatterns) {
        if (abs(expTmp) > SCALING_EXPONENT_THRESHOLD) {
            // now using sPartials2 to hold scaling trigger boolean
            sPartials2[patIdx][0] = 1;
        } else {
            partials3[u] = tmpPartial;
            sPartials2[patIdx][0] = 0;
            sPartials1[patIdx][0] = 0;
        }
    }

    KW_LOCAL_FENCE;

    int scalingActive = sPartials2[patIdx][0];

    if (scalingActive) {
        // now using sPartials1 to store max unscaled partials3
        sPartials1[patIdx][state] = tmpPartial;
    }

    KW_LOCAL_FENCE;

    // Unrolled parallel max-reduction
    if (scalingActive && state < 2) {
        REAL compare = sPartials1[patIdx][state + 2];
        if (compare >  sPartials1[patIdx][state])
            sPartials1[patIdx][state] = compare;
    }

    KW_LOCAL_FENCE;

    if (scalingActive && state < 1) {
        REAL maxPartial = sPartials1[patIdx][1];
        if (maxPartial < sPartials1[patIdx][0])
            maxPartial = sPartials1[patIdx][0];
        int expMax;
        frexp(maxPartial, &expMax);
        sPartials1[patIdx][0] = expMax;
    }

    KW_LOCAL_FENCE;
    
    if (scalingActive)
        partials3[u] = ldexp(sigTmp, expTmp - sPartials1[patIdx][0]);

    int myIdx = (patIdx * PADDED_STATE_COUNT) + state; // threadId in block
    if ((myIdx < PATTERN_BLOCK_SIZE) && (myIdx + (KW_GROUP_ID_0 * PATTERN_BLOCK_SIZE) < totalPatterns))
        scalingFactors[(KW_GROUP_ID_0 * PATTERN_BLOCK_SIZE) + (matrix * totalPatterns) + myIdx] = sPartials1[myIdx][0];

}

KW_GLOBAL_KERNEL void kernelIntegrateLikelihoodsAutoScaling(KW_GLOBAL_VAR REAL* dResult,
                                                     KW_GLOBAL_VAR REAL* dRootPartials,
                                                     KW_GLOBAL_VAR REAL* dWeights,
                                                     KW_GLOBAL_VAR REAL* dFrequencies,
                                                     KW_GLOBAL_VAR int* dRootScalingFactors,
                                                     int matrixCount,
                                                     int patternCount) {
    int state   = KW_LOCAL_ID_0;
    int pattern = KW_GROUP_ID_0;
//    int patternCount = KW_NUM_GROUPS_0;

    KW_LOCAL_MEM REAL stateFreq[PADDED_STATE_COUNT];
    // TODO: Currently assumes MATRIX_BLOCK_SIZE >> matrixCount
    KW_LOCAL_MEM REAL matrixProp[MATRIX_BLOCK_SIZE];
    KW_LOCAL_MEM REAL matrixScalers[MATRIX_BLOCK_SIZE];
    KW_LOCAL_MEM REAL sum[PADDED_STATE_COUNT];

    // Load shared memory

    stateFreq[state] = dFrequencies[state];
    sum[state] = 0;

    for(int matrixEdge = 0; matrixEdge < matrixCount; matrixEdge += PADDED_STATE_COUNT) {
        int x = matrixEdge + state;
        if (x < matrixCount) {
            matrixProp[x] = dWeights[x];
            matrixScalers[x] = dRootScalingFactors[pattern + (x * patternCount)];
        }
    }

    KW_LOCAL_FENCE;

    int u = state + pattern * PADDED_STATE_COUNT;
    int delta = patternCount * PADDED_STATE_COUNT;

    short maxScaleFactor = matrixScalers[0];
    for(int r = 1; r < matrixCount; r++) {
        int tmpFactor = matrixScalers[r];
        if (tmpFactor > maxScaleFactor)
            maxScaleFactor = tmpFactor;
    }

    for(int r = 0; r < matrixCount; r++) {
        int tmpFactor = matrixScalers[r];
        if (tmpFactor != maxScaleFactor) {
            int expTmp;
            sum[state] += ldexp(frexp(dRootPartials[u + delta * r], &expTmp), expTmp + (tmpFactor - maxScaleFactor)) * matrixProp[r];
        } else {
            sum[state] += dRootPartials[u + delta * r] * matrixProp[r];
        }
    }

    sum[state] *= stateFreq[state];
    KW_LOCAL_FENCE;

#ifdef IS_POWER_OF_TWO
    // parallelized reduction *** only works for powers-of-2 ****
    for (int i = PADDED_STATE_COUNT / 2; i > 0; i >>= 1) {
        if (state < i) {
#else
    for (int i = SMALLEST_POWER_OF_TWO / 2; i > 0; i >>= 1) {
        if (state < i && state + i < PADDED_STATE_COUNT ) {
#endif // IS_POWER_OF_TWO
            sum[state] += sum[state + i];
        }
        KW_LOCAL_FENCE;
    }

    if (state == 0)
        dResult[pattern] = (log(sum[state]) + (M_LN2 * maxScaleFactor));
}

#ifdef CUDA
    #include "kernelsXDerivatives.cu"
#endif // CUDA

#ifdef CUDA
} // extern "C"
#endif //CUDA
