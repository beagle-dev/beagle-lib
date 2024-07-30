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
    #include <stdlib.h>
    #include <string.h>
    #include <stdio.h>
    #include <math.h>
    extern "C" {
#elif defined(FW_OPENCL)
    #ifdef DOUBLE_PRECISION
        #pragma OPENCL EXTENSION cl_khr_fp64: enable
    #endif
    #define __umul24(x, y) (x * y)
#endif //FW_OPENCL

#if (!defined DOUBLE_PRECISION && defined FP_FAST_FMAF) || (defined DOUBLE_PRECISION && defined FP_FAST_FMA)
    #define FMA(x, y, z) (z = fma(x, y, z))
#else //FP_FAST_FMA
    #define FMA(x, y, z) (z += x * y)
#endif //FP_FAST_FMA

#if (defined CUDA) && (defined DOUBLE_PRECISION) &&  (__CUDA_ARCH__ < 600)
    __device__ double atomicAdd(double* address, double val)
    {
        unsigned long long int* address_as_ull =
                                  (unsigned long long int*)address;
        unsigned long long int old = *address_as_ull, assumed;

        do {
            assumed = old;
            old = atomicCAS(address_as_ull, assumed,
                            __double_as_longlong(val +
                                   __longlong_as_double(assumed)));

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
        } while (assumed != old);

        return __longlong_as_double(old);
    }
#endif

///////////////////////////////////////////////////////////////////////////////

KW_GLOBAL_KERNEL void kernelReorderPatterns(      KW_GLOBAL_VAR REAL*             dPartials,
                                                  KW_GLOBAL_VAR int*              dStates,
                                                  KW_GLOBAL_VAR int*              dStatesSort,
                                            const KW_GLOBAL_VAR int*  KW_RESTRICT dTipOffsets,
                                            const KW_GLOBAL_VAR int*  KW_RESTRICT dTipTypes,
                                            const KW_GLOBAL_VAR int*  KW_RESTRICT dPatternsNewOrder,
                                            const KW_GLOBAL_VAR REAL* KW_RESTRICT dPatternWeights,
                                                  KW_GLOBAL_VAR REAL* KW_RESTRICT dPatternWeightsSort,
                                                                int               patternCount,
                                                                int               paddedPatternCount) {
#ifdef FW_OPENCL_CPU
    int state      = 0;
    int pattern    = KW_LOCAL_ID_0 + KW_GROUP_ID_0 * KW_LOCAL_SIZE_0;
#else
    int state      = KW_LOCAL_ID_0;
    int pattern    = KW_LOCAL_ID_1 + KW_GROUP_ID_0 * KW_LOCAL_SIZE_1;
#endif
    int stateCount = PADDED_STATE_COUNT;
    int category   = KW_GROUP_ID_1;
    int tip        = KW_GROUP_ID_2;
    int tipCount   = KW_NUM_GROUPS_2;

    if (pattern < patternCount) {
        int patternSorted  = dPatternsNewOrder[pattern];

        if (dTipTypes[tip] == 0) {
            int categoryOffset = category * stateCount * paddedPatternCount;

            int sortIndex   = categoryOffset + patternSorted * stateCount;
            int originIndex = categoryOffset + pattern       * stateCount;

            const KW_GLOBAL_VAR REAL* KW_RESTRICT partialOriginal = dPartials + dTipOffsets[tip];
                  KW_GLOBAL_VAR REAL* KW_RESTRICT partialSorted   = dPartials + dTipOffsets[tip+tipCount];

#ifdef FW_OPENCL_CPU
            for (int i=0; i < stateCount; i++) {
                partialSorted[sortIndex+i] = partialOriginal[originIndex+i];
            }
#else
            sortIndex += state;
            originIndex += state;
            partialSorted[sortIndex] = partialOriginal[originIndex];
#endif
        } else if (state == 0) {
            const KW_GLOBAL_VAR int* KW_RESTRICT stateOriginal = dStates     + dTipOffsets[tip];
                  KW_GLOBAL_VAR int* KW_RESTRICT stateSorted   = dStatesSort + dTipOffsets[tip+tipCount];

            stateSorted[patternSorted] = stateOriginal[pattern];
        }

        if (state == 0 && category == 0 && tip == 0) {
            dPatternWeightsSort[patternSorted] = dPatternWeights[pattern];
        }
    }
}

KW_GLOBAL_KERNEL void kernelMatrixMulADBMulti(KW_GLOBAL_VAR REAL* dMatrices,
                                              KW_GLOBAL_VAR unsigned int* offsets,
                                              KW_GLOBAL_VAR REAL* Alist,
                                              KW_GLOBAL_VAR REAL* Dlist,
                                              KW_GLOBAL_VAR REAL* Blist,
                                              KW_GLOBAL_VAR REAL* distanceQueue,
                                              int length,
                                              int wB,
                                              int totalMatrix) {

    int wMatrix = KW_GROUP_ID_0 % totalMatrix;
    int offIndex = wMatrix * 3;

    // Block index
    int bx = KW_GROUP_ID_0 / totalMatrix;
    int by = KW_GROUP_ID_1;

    // Thread index
    int tx = KW_LOCAL_ID_0;
    int ty = KW_LOCAL_ID_1;
    int BLOCKS = KW_NUM_GROUPS_1;

    KW_GLOBAL_VAR REAL* C = dMatrices + offsets[offIndex];
    KW_GLOBAL_VAR REAL* B = Blist + offsets[offIndex + 1]; // dEvec
    KW_GLOBAL_VAR REAL* A = Alist + offsets[offIndex + 1]; // dIevc
    KW_GLOBAL_VAR REAL* D = Dlist + offsets[offIndex + 2]; // dEigenValues
    REAL distance = distanceQueue[wMatrix];

    const int EDGE = PADDED_STATE_COUNT - (BLOCKS - 1) * MULTIPLY_BLOCK_SIZE;

    // Step size used to iterate through the sub-matrices of A
    int aStep = MULTIPLY_BLOCK_SIZE;

    // Step size used to iterate through the sub-matrices of B
    int bStep = MULTIPLY_BLOCK_SIZE * PADDED_STATE_COUNT;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    REAL Csub = 0;

    int a = PADDED_STATE_COUNT * MULTIPLY_BLOCK_SIZE * by;
    int b = MULTIPLY_BLOCK_SIZE * bx;
    int d = 0; //MULTIPLY_BLOCK_SIZE * bx;

    KW_LOCAL_MEM REAL As[MULTIPLY_BLOCK_SIZE][MULTIPLY_BLOCK_SIZE];
    KW_LOCAL_MEM REAL Bs[MULTIPLY_BLOCK_SIZE][MULTIPLY_BLOCK_SIZE];
    KW_LOCAL_MEM REAL Ds[MULTIPLY_BLOCK_SIZE];

    for (int i = 0; i < BLOCKS - 1; i++) {

        if (ty == 0)
            Ds[tx] = exp(D[d + tx] * distance);

        As[ty][tx] = A[a + PADDED_STATE_COUNT * ty + tx];
        Bs[ty][tx] = B[b + PADDED_STATE_COUNT * ty + tx];

        KW_LOCAL_FENCE;

        for (int k = 0; k < MULTIPLY_BLOCK_SIZE; ++k)
            Csub += As[ty][k] * Ds[k] * Bs[k][tx];

        KW_LOCAL_FENCE;

        a += aStep;
        b += bStep;
        d += MULTIPLY_BLOCK_SIZE;
    }

    // Last block is too long
    if (tx < EDGE && ty < EDGE) {
        if (ty == 0)
            Ds[tx] = exp(D[d + tx] * distance);

        As[ty][tx] = A[a + PADDED_STATE_COUNT * ty + tx];
        Bs[ty][tx] = B[b + PADDED_STATE_COUNT * ty + tx];

    } else {

        if (ty == 0)
            Ds[tx] = 0;

        As[ty][tx] = 0;
        Bs[ty][tx] = 0;
    }

    KW_LOCAL_FENCE;

    for (int k = 0; k < EDGE; k++)
        Csub += As[ty][k] * Ds[k] * Bs[k][tx];

    KW_LOCAL_FENCE;

    // Write the block sub-matrix to device memory;
    // each thread writes one element

    if ((tx < EDGE || bx < BLOCKS - 1) && (ty < EDGE || by < BLOCKS - 1)) { // It's OK to write
        if (Csub < 0)
            C[PADDED_STATE_COUNT* MULTIPLY_BLOCK_SIZE * by + MULTIPLY_BLOCK_SIZE * bx +
              PADDED_STATE_COUNT * ty + tx] = 0;
        else
            C[PADDED_STATE_COUNT* MULTIPLY_BLOCK_SIZE * by + MULTIPLY_BLOCK_SIZE * bx +
              PADDED_STATE_COUNT * ty + tx] = Csub;
    }
}

KW_GLOBAL_KERNEL void kernelMatrixMulADB(KW_GLOBAL_VAR REAL* dMatrices,
                                   KW_GLOBAL_VAR unsigned int* listC,
                                   KW_GLOBAL_VAR REAL* A,
                                   KW_GLOBAL_VAR REAL* D,
                                   KW_GLOBAL_VAR REAL* B,
                                   KW_GLOBAL_VAR REAL* distanceQueue,
                                   int length,
                                   int wB,
                                   int totalMatrix) {

    int wMatrix = KW_GROUP_ID_0 % totalMatrix;

    // Block index
    int bx = KW_GROUP_ID_0 / totalMatrix;
    int by = KW_GROUP_ID_1;

    // Thread index
    int tx = KW_LOCAL_ID_0;
    int ty = KW_LOCAL_ID_1;
    int BLOCKS = KW_NUM_GROUPS_1;

#ifdef CUDA
    KW_LOCAL_MEM REAL* C;
    KW_LOCAL_MEM REAL distance;
    if (tx == 0 && ty == 0) {
        C = dMatrices + listC[wMatrix]; // Non-coalescent read
        distance = distanceQueue[wMatrix]; // Non-coalescent read
    }
#elif defined(FW_OPENCL)
    KW_GLOBAL_VAR REAL* C;
    REAL distance;
    C = dMatrices + listC[wMatrix];
    distance = distanceQueue[wMatrix];
#endif

    KW_LOCAL_FENCE;

    const int EDGE = PADDED_STATE_COUNT - (BLOCKS - 1) * MULTIPLY_BLOCK_SIZE;

    // Step size used to iterate through the sub-matrices of A
    int aStep = MULTIPLY_BLOCK_SIZE;

    // Step size used to iterate through the sub-matrices of B
    int bStep = MULTIPLY_BLOCK_SIZE * PADDED_STATE_COUNT;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    REAL Csub = 0;

    int a = PADDED_STATE_COUNT * MULTIPLY_BLOCK_SIZE * by;
    int b = MULTIPLY_BLOCK_SIZE * bx;
    int d = 0; //MULTIPLY_BLOCK_SIZE * bx;

    KW_LOCAL_MEM REAL As[MULTIPLY_BLOCK_SIZE][MULTIPLY_BLOCK_SIZE];
    KW_LOCAL_MEM REAL Bs[MULTIPLY_BLOCK_SIZE][MULTIPLY_BLOCK_SIZE];
    KW_LOCAL_MEM REAL Ds[MULTIPLY_BLOCK_SIZE];

    for (int i = 0; i < BLOCKS - 1; i++) {

        if (ty == 0)
            Ds[tx] = exp(D[d + tx] * distance);

        As[ty][tx] = A[a + PADDED_STATE_COUNT * ty + tx];
        Bs[ty][tx] = B[b + PADDED_STATE_COUNT * ty + tx];

        KW_LOCAL_FENCE;

        for (int k = 0; k < MULTIPLY_BLOCK_SIZE; ++k)
            Csub += As[ty][k] * Ds[k] * Bs[k][tx];

        KW_LOCAL_FENCE;

        a += aStep;
        b += bStep;
        d += MULTIPLY_BLOCK_SIZE;
    }

    // Last block is too long
    if (tx < EDGE && ty < EDGE) {
        if (ty == 0)
            Ds[tx] = exp(D[d + tx] * distance);

        As[ty][tx] = A[a + PADDED_STATE_COUNT * ty + tx];
        Bs[ty][tx] = B[b + PADDED_STATE_COUNT * ty + tx];

    } else {

        if (ty == 0)
            Ds[tx] = 0;

        As[ty][tx] = 0;
        Bs[ty][tx] = 0;
    }

    KW_LOCAL_FENCE;

    for (int k = 0; k < EDGE; k++)
        Csub += As[ty][k] * Ds[k] * Bs[k][tx];

    KW_LOCAL_FENCE;

    // Write the block sub-matrix to device memory;
    // each thread writes one element

    if ((tx < EDGE || bx < BLOCKS - 1) && (ty < EDGE || by < BLOCKS - 1)) { // It's OK to write
        if (Csub < 0)
            C[PADDED_STATE_COUNT* MULTIPLY_BLOCK_SIZE * by + MULTIPLY_BLOCK_SIZE * bx +
              PADDED_STATE_COUNT * ty + tx] = 0;
        else
            C[PADDED_STATE_COUNT* MULTIPLY_BLOCK_SIZE * by + MULTIPLY_BLOCK_SIZE * bx +
              PADDED_STATE_COUNT * ty + tx] = Csub;
    }
}

KW_GLOBAL_KERNEL void kernelMatrixMulADBFirstDeriv(KW_GLOBAL_VAR REAL* dMatrices,
                                           KW_GLOBAL_VAR unsigned int* listC,
                                           KW_GLOBAL_VAR REAL* A,
                                           KW_GLOBAL_VAR REAL* D,
                                           KW_GLOBAL_VAR REAL* B,
                                           KW_GLOBAL_VAR REAL* distanceQueue,
                                           int length,
                                           int wB,
                                           int totalMatrix) {

    int wMatrix = KW_GROUP_ID_0 % totalMatrix;

    // Block index
    int bx = KW_GROUP_ID_0 / totalMatrix;
    int by = KW_GROUP_ID_1;

    // Thread index
    int tx = KW_LOCAL_ID_0;
    int ty = KW_LOCAL_ID_1;
    int BLOCKS = KW_NUM_GROUPS_1;

#ifdef CUDA
    KW_LOCAL_MEM REAL* C;
    KW_LOCAL_MEM REAL* CFirstDeriv;
    KW_LOCAL_MEM REAL distanceLength;
    KW_LOCAL_MEM REAL distanceRate;
    if (tx == 0 && ty == 0) {
        C = dMatrices + listC[wMatrix];
        CFirstDeriv = dMatrices + listC[wMatrix + totalMatrix];
        distanceLength = distanceQueue[wMatrix]; // Non-coalescent read
        distanceRate = distanceQueue[wMatrix + totalMatrix]; // Non-coalescent read
    }
#elif defined(FW_OPENCL)
    KW_GLOBAL_VAR REAL* C;
    KW_GLOBAL_VAR REAL* CFirstDeriv;
    REAL distanceLength;
    REAL distanceRate;
    C = dMatrices + listC[wMatrix];
    CFirstDeriv = dMatrices + listC[wMatrix + totalMatrix];
    distanceLength = distanceQueue[wMatrix];
    distanceRate = distanceQueue[wMatrix + totalMatrix];
#endif

    KW_LOCAL_FENCE;

    const int EDGE = PADDED_STATE_COUNT - (BLOCKS - 1) * MULTIPLY_BLOCK_SIZE;

    // Step size used to iterate through the sub-matrices of A
    int aStep = MULTIPLY_BLOCK_SIZE;

    // Step size used to iterate through the sub-matrices of B
    int bStep = MULTIPLY_BLOCK_SIZE * PADDED_STATE_COUNT;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    REAL Csub = 0;
    REAL CFirstDerivSub = 0;

    int a = PADDED_STATE_COUNT * MULTIPLY_BLOCK_SIZE * by;
    int b = MULTIPLY_BLOCK_SIZE * bx;
    int d = 0; //MULTIPLY_BLOCK_SIZE * bx;

    KW_LOCAL_MEM REAL As[MULTIPLY_BLOCK_SIZE][MULTIPLY_BLOCK_SIZE];
    KW_LOCAL_MEM REAL Bs[MULTIPLY_BLOCK_SIZE][MULTIPLY_BLOCK_SIZE];
    KW_LOCAL_MEM REAL Ds[MULTIPLY_BLOCK_SIZE][2];

    for (int i = 0; i < BLOCKS - 1; i++) {

        if (ty == 0) {
            REAL scaledEigenTmp = D[d + tx] * distanceRate;
            Ds[tx][0] = exp(scaledEigenTmp * distanceLength);
            Ds[tx][1] = scaledEigenTmp * Ds[tx][0];
        }

        As[ty][tx] = A[a + PADDED_STATE_COUNT * ty + tx];
        Bs[ty][tx] = B[b + PADDED_STATE_COUNT * ty + tx];

        KW_LOCAL_FENCE;

        for (int k = 0; k < MULTIPLY_BLOCK_SIZE; ++k) {
            Csub += As[ty][k] * Ds[k][0] * Bs[k][tx];
            CFirstDerivSub += As[ty][k] * Ds[k][1] * Bs[k][tx];
        }

        KW_LOCAL_FENCE;

        a += aStep;
        b += bStep;
        d += MULTIPLY_BLOCK_SIZE;
    }

    // Last block is too long
    if (tx < EDGE && ty < EDGE) {
        if (ty == 0) {
            REAL scaledEigenTmp = D[d + tx] * distanceRate;
            Ds[tx][0] = exp(scaledEigenTmp * distanceLength);
            Ds[tx][1] = scaledEigenTmp * Ds[tx][0];
                }

        As[ty][tx] = A[a + PADDED_STATE_COUNT * ty + tx];
        Bs[ty][tx] = B[b + PADDED_STATE_COUNT * ty + tx];

    } else {

        if (ty == 0) {
            Ds[tx][0] = 0;
            Ds[tx][1] = 0;
        }

        As[ty][tx] = 0;
        Bs[ty][tx] = 0;
    }

    KW_LOCAL_FENCE;

    for (int k = 0; k < EDGE; k++) {
        Csub += As[ty][k] * Ds[k][0] * Bs[k][tx];
        CFirstDerivSub += As[ty][k] * Ds[k][1] * Bs[k][tx];
    }

    KW_LOCAL_FENCE;

    // Write the block sub-matrix to device memory;
    // each thread writes one element

    if ((tx < EDGE || bx < BLOCKS - 1) && (ty < EDGE || by < BLOCKS - 1)) { // It's OK to write
        if (Csub < 0)
            C[PADDED_STATE_COUNT* MULTIPLY_BLOCK_SIZE * by + MULTIPLY_BLOCK_SIZE * bx +
              PADDED_STATE_COUNT * ty + tx] = 0;
        else
            C[PADDED_STATE_COUNT* MULTIPLY_BLOCK_SIZE * by + MULTIPLY_BLOCK_SIZE * bx +
              PADDED_STATE_COUNT * ty + tx] = Csub;

        CFirstDeriv[PADDED_STATE_COUNT* MULTIPLY_BLOCK_SIZE * by + MULTIPLY_BLOCK_SIZE * bx +
          PADDED_STATE_COUNT * ty + tx] = CFirstDerivSub;
    }
}

KW_GLOBAL_KERNEL void kernelMatrixMulADBSecondDeriv(KW_GLOBAL_VAR REAL* dMatrices,
                                           KW_GLOBAL_VAR unsigned int* listC,
                                           KW_GLOBAL_VAR REAL* A,
                                           KW_GLOBAL_VAR REAL* D,
                                           KW_GLOBAL_VAR REAL* B,
                                           KW_GLOBAL_VAR REAL* distanceQueue,
                                           int length,
                                           int wB,
                                           int totalMatrix) {

    int wMatrix = KW_GROUP_ID_0 % totalMatrix;

    // Block index
    int bx = KW_GROUP_ID_0 / totalMatrix;
    int by = KW_GROUP_ID_1;

    // Thread index
    int tx = KW_LOCAL_ID_0;
    int ty = KW_LOCAL_ID_1;
    int BLOCKS = KW_NUM_GROUPS_1;

#ifdef CUDA
    KW_LOCAL_MEM REAL* C;
    KW_LOCAL_MEM REAL* CFirstDeriv;
    KW_LOCAL_MEM REAL* CSecondDeriv;
    KW_LOCAL_MEM REAL distanceLength;
    KW_LOCAL_MEM REAL distanceRate;
    if (tx == 0 && ty == 0) {
        C = dMatrices + listC[wMatrix];
        CFirstDeriv = dMatrices + listC[wMatrix + totalMatrix];
        CSecondDeriv = dMatrices + listC[wMatrix + totalMatrix * 2];
        distanceLength = distanceQueue[wMatrix]; // Non-coalescent read
        distanceRate = distanceQueue[wMatrix + totalMatrix]; // Non-coalescent read
    }
#elif defined(FW_OPENCL)
    KW_GLOBAL_VAR REAL* C;
    KW_GLOBAL_VAR REAL* CFirstDeriv;
    KW_GLOBAL_VAR REAL* CSecondDeriv;
    REAL distanceLength;
    REAL distanceRate;
    C = dMatrices + listC[wMatrix];
    CFirstDeriv = dMatrices + listC[wMatrix + totalMatrix];
    CSecondDeriv = dMatrices + listC[wMatrix + totalMatrix * 2];
    distanceLength = distanceQueue[wMatrix];
    distanceRate = distanceQueue[wMatrix + totalMatrix];
#endif

    KW_LOCAL_FENCE;

    const int EDGE = PADDED_STATE_COUNT - (BLOCKS - 1) * MULTIPLY_BLOCK_SIZE;

    // Step size used to iterate through the sub-matrices of A
    int aStep = MULTIPLY_BLOCK_SIZE;

    // Step size used to iterate through the sub-matrices of B
    int bStep = MULTIPLY_BLOCK_SIZE * PADDED_STATE_COUNT;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    REAL Csub = 0;
    REAL CFirstDerivSub = 0;
    REAL CSecondDerivSub = 0;

    int a = PADDED_STATE_COUNT * MULTIPLY_BLOCK_SIZE * by;
    int b = MULTIPLY_BLOCK_SIZE * bx;
    int d = 0; //MULTIPLY_BLOCK_SIZE * bx;

    KW_LOCAL_MEM REAL As[MULTIPLY_BLOCK_SIZE][MULTIPLY_BLOCK_SIZE];
    KW_LOCAL_MEM REAL Bs[MULTIPLY_BLOCK_SIZE][MULTIPLY_BLOCK_SIZE];
    KW_LOCAL_MEM REAL Ds[MULTIPLY_BLOCK_SIZE][3];

    for (int i = 0; i < BLOCKS - 1; i++) {

        if (ty == 0) {
            REAL scaledEigenTmp = D[d + tx] * distanceRate;
            Ds[tx][0] = exp(scaledEigenTmp * distanceLength);
            Ds[tx][1] = scaledEigenTmp * Ds[tx][0];
            Ds[tx][2] = scaledEigenTmp * Ds[tx][1];
        }

        As[ty][tx] = A[a + PADDED_STATE_COUNT * ty + tx];
        Bs[ty][tx] = B[b + PADDED_STATE_COUNT * ty + tx];

        KW_LOCAL_FENCE;

        for (int k = 0; k < MULTIPLY_BLOCK_SIZE; ++k) {
            Csub += As[ty][k] * Ds[k][0] * Bs[k][tx];
            CFirstDerivSub += As[ty][k] * Ds[k][1] * Bs[k][tx];
            CSecondDerivSub += As[ty][k] * Ds[k][2] * Bs[k][tx];
        }

        KW_LOCAL_FENCE;

        a += aStep;
        b += bStep;
        d += MULTIPLY_BLOCK_SIZE;
    }

    // Last block is too long
    if (tx < EDGE && ty < EDGE) {
        if (ty == 0) {
            REAL scaledEigenTmp = D[d + tx] * distanceRate;
            Ds[tx][0] = exp(scaledEigenTmp * distanceLength);
            Ds[tx][1] = scaledEigenTmp * Ds[tx][0];
            Ds[tx][2] = scaledEigenTmp * Ds[tx][1];
                }

        As[ty][tx] = A[a + PADDED_STATE_COUNT * ty + tx];
        Bs[ty][tx] = B[b + PADDED_STATE_COUNT * ty + tx];

    } else {

        if (ty == 0) {
            Ds[tx][0] = 0;
            Ds[tx][1] = 0;
            Ds[tx][2] = 0;
        }

        As[ty][tx] = 0;
        Bs[ty][tx] = 0;
    }

    KW_LOCAL_FENCE;

    for (int k = 0; k < EDGE; k++) {
        Csub += As[ty][k] * Ds[k][0] * Bs[k][tx];
        CFirstDerivSub += As[ty][k] * Ds[k][1] * Bs[k][tx];
        CSecondDerivSub += As[ty][k] * Ds[k][2] * Bs[k][tx];
    }

    KW_LOCAL_FENCE;

    // Write the block sub-matrix to device memory;
    // each thread writes one element

    if ((tx < EDGE || bx < BLOCKS - 1) && (ty < EDGE || by < BLOCKS - 1)) { // It's OK to write
        if (Csub < 0)
            C[PADDED_STATE_COUNT* MULTIPLY_BLOCK_SIZE * by + MULTIPLY_BLOCK_SIZE * bx +
              PADDED_STATE_COUNT * ty + tx] = 0;
        else
            C[PADDED_STATE_COUNT* MULTIPLY_BLOCK_SIZE * by + MULTIPLY_BLOCK_SIZE * bx +
              PADDED_STATE_COUNT * ty + tx] = Csub;

        CFirstDeriv[PADDED_STATE_COUNT* MULTIPLY_BLOCK_SIZE * by + MULTIPLY_BLOCK_SIZE * bx +
          PADDED_STATE_COUNT * ty + tx] = CFirstDerivSub;

        CSecondDeriv[PADDED_STATE_COUNT* MULTIPLY_BLOCK_SIZE * by + MULTIPLY_BLOCK_SIZE * bx +
          PADDED_STATE_COUNT * ty + tx] = CSecondDerivSub;
    }
}

KW_GLOBAL_KERNEL void kernelMatrixConvolution(KW_GLOBAL_VAR REAL* dMatrices,
								        KW_GLOBAL_VAR unsigned int* list,
								        int totalMatrixCount
								        ) {

	    int wMatrix = KW_GROUP_ID_0 % totalMatrixCount;

	    // Block index
	    int bx = KW_GROUP_ID_0 / totalMatrixCount;
	    int by = KW_GROUP_ID_1;

	    // Thread index
	    int tx = KW_LOCAL_ID_0;
	    int ty = KW_LOCAL_ID_1;
	    int BLOCKS = KW_NUM_GROUPS_1;


#ifdef CUDA
        KW_LOCAL_MEM REAL* A;
        KW_LOCAL_MEM REAL* B;
        KW_LOCAL_MEM REAL* C;
        if (tx == 0 && ty == 0) {
            A = dMatrices + list[wMatrix]; // Non-coalescent read
            B = dMatrices + list[wMatrix + totalMatrixCount]; // Non-coalescent read
            C = dMatrices + list[wMatrix + totalMatrixCount*2]; // Non-coalescent read
        }
#elif defined(FW_OPENCL)
        KW_GLOBAL_VAR REAL* A;
        KW_GLOBAL_VAR REAL* B;
        KW_GLOBAL_VAR REAL* C;
        A = dMatrices + list[wMatrix];
        B = dMatrices + list[wMatrix + totalMatrixCount];
        C = dMatrices + list[wMatrix + totalMatrixCount*2];
#endif

	    KW_LOCAL_FENCE;

	    const int EDGE = PADDED_STATE_COUNT - (BLOCKS - 1) * MULTIPLY_BLOCK_SIZE;

	    // Step size used to iterate through the sub-matrices of A
	    int aStep = MULTIPLY_BLOCK_SIZE;

	    // Step size used to iterate through the sub-matrices of B
	    int bStep = MULTIPLY_BLOCK_SIZE * PADDED_STATE_COUNT;

	    // Csub is used to store the element of the block sub-matrix
	    // that is computed by the thread
	    REAL Csub = 0;

	    int a = PADDED_STATE_COUNT * MULTIPLY_BLOCK_SIZE * by;
	    int b = MULTIPLY_BLOCK_SIZE * bx;

	    KW_LOCAL_MEM REAL As[MULTIPLY_BLOCK_SIZE][MULTIPLY_BLOCK_SIZE];
	    KW_LOCAL_MEM REAL Bs[MULTIPLY_BLOCK_SIZE][MULTIPLY_BLOCK_SIZE];

	    for (int i = 0; i < BLOCKS - 1; i++) {

	        As[ty][tx] = A[a + PADDED_STATE_COUNT * ty + tx];
	        Bs[ty][tx] = B[b + PADDED_STATE_COUNT * ty + tx];

	        KW_LOCAL_FENCE;

	        for (int k = 0; k < MULTIPLY_BLOCK_SIZE; ++k)
	            Csub += As[ty][k]  * Bs[k][tx];

	        KW_LOCAL_FENCE;

	        a += aStep;
	        b += bStep;
	    }//END: BLOCKS loop

	    // Last block is too long
	    if (tx < EDGE && ty < EDGE) {

	#ifndef KERNEL_PRINT_ENABLED
	        KW_LOCAL_FENCE;
	#endif

	        As[ty][tx] = A[a + PADDED_STATE_COUNT * ty + tx];
	        Bs[ty][tx] = B[b + PADDED_STATE_COUNT * ty + tx];

	    } else {

	        As[ty][tx] = 0;
	        Bs[ty][tx] = 0;

	    }//END: EDGE check

	    KW_LOCAL_FENCE;

	    for (int k = 0; k < EDGE; k++) {
	        Csub += As[ty][k] *  Bs[k][tx];
	    }

	    KW_LOCAL_FENCE;

	    // Write the block sub-matrix to device memory;
	    // each thread writes one element

	    if ((tx < EDGE || bx < BLOCKS - 1) && (ty < EDGE || by < BLOCKS - 1)) { // It's OK to write
	        if (Csub < 0) {

	        	C[PADDED_STATE_COUNT* MULTIPLY_BLOCK_SIZE * by + MULTIPLY_BLOCK_SIZE * bx +
	              PADDED_STATE_COUNT * ty + tx] = 0;

	        } else {

	        	C[PADDED_STATE_COUNT* MULTIPLY_BLOCK_SIZE * by + MULTIPLY_BLOCK_SIZE * bx +
	              PADDED_STATE_COUNT * ty + tx] = Csub;

	        }//END: Csub check
	    }//END: EDGE check

}//END: kernelMatrixConvolution

KW_GLOBAL_KERNEL void kernelMatrixTranspose(KW_GLOBAL_VAR REAL* dMatrices,
                                            KW_GLOBAL_VAR unsigned int* list,
                                            int totalMatrixCount) {

	    int wMatrix = KW_GROUP_ID_0 % totalMatrixCount;

	    // Block index
	    int bx = KW_GROUP_ID_0 / totalMatrixCount;
	    int by = KW_GROUP_ID_1;

	    // Thread index
	    int tx = KW_LOCAL_ID_0;
	    int ty = KW_LOCAL_ID_1;

#ifdef CUDA
        KW_LOCAL_MEM REAL* A;
        KW_LOCAL_MEM REAL* C;
        if (tx == 0 && ty == 0) {
            A = dMatrices + list[wMatrix]; // Non-coalescent read
            C = dMatrices + list[wMatrix + totalMatrixCount]; // Non-coalescent read
        }
#elif defined(FW_OPENCL)
        KW_GLOBAL_VAR REAL* A;
        KW_GLOBAL_VAR REAL* C;
        A = dMatrices + list[wMatrix];
        C = dMatrices + list[wMatrix + totalMatrixCount];
#endif

	    KW_LOCAL_FENCE;

        const int rowOffset = MULTIPLY_BLOCK_SIZE * bx;
        const int colOffset = MULTIPLY_BLOCK_SIZE * by;

        const int row = rowOffset + tx;
        const int col = colOffset + ty;

	    KW_LOCAL_MEM REAL As[MULTIPLY_BLOCK_SIZE][MULTIPLY_BLOCK_SIZE];

	    if (row < PADDED_STATE_COUNT && col < PADDED_STATE_COUNT) {
	        As[ty][tx] = A[PADDED_STATE_COUNT * colOffset + rowOffset +
                           PADDED_STATE_COUNT * ty + tx];
	    }

	    KW_LOCAL_FENCE;

	    if (row < PADDED_STATE_COUNT && col < PADDED_STATE_COUNT) {
		    C[PADDED_STATE_COUNT * rowOffset + colOffset +
		      PADDED_STATE_COUNT * ty + tx] = As[tx][ty];
	    }
}

KW_GLOBAL_KERNEL void kernelMatrixMulADBComplexMulti(KW_GLOBAL_VAR REAL* dMatrices,
                                   KW_GLOBAL_VAR unsigned int* offsets,
                                   KW_GLOBAL_VAR REAL* Alist,
                                   KW_GLOBAL_VAR REAL* Dlist,
                                   KW_GLOBAL_VAR REAL* Blist,
                                   KW_GLOBAL_VAR REAL* distanceQueue,
                                   int length,
                                   int wB,
                                   int totalMatrix) {
#if !(defined(FW_OPENCL_APPLEAMDGPU) && defined(DOUBLE_PRECISION)) // TODO: fix this issue
    int wMatrix = KW_GROUP_ID_0 % totalMatrix;
    int offIndex = wMatrix * 3;

    // Block index
    int bx = KW_GROUP_ID_0 / totalMatrix;
    int by = KW_GROUP_ID_1;
    int BLOCKS = KW_NUM_GROUPS_1;

    // Thread index
    int tx = KW_LOCAL_ID_0;
    int ty = KW_LOCAL_ID_1;

    KW_GLOBAL_VAR REAL* C = dMatrices + offsets[offIndex];
    KW_GLOBAL_VAR REAL* B = Blist + offsets[offIndex + 1]; // dEvec
    KW_GLOBAL_VAR REAL* A = Alist + offsets[offIndex + 1]; // dIevc
    KW_GLOBAL_VAR REAL* D = Dlist + offsets[offIndex + 2]; // dEigenValues
    REAL distance = distanceQueue[wMatrix];

    const int EDGE = PADDED_STATE_COUNT - (BLOCKS - 1) * MULTIPLY_BLOCK_SIZE;

    // Step size used to iterate through the sub-matrices of A
    int aStep = MULTIPLY_BLOCK_SIZE;

    // Step size used to iterate through the sub-matrices of B
    int bStep = MULTIPLY_BLOCK_SIZE * PADDED_STATE_COUNT;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    REAL Csub = 0;

    int a = PADDED_STATE_COUNT * MULTIPLY_BLOCK_SIZE * by;
    int b = MULTIPLY_BLOCK_SIZE * bx;
    int d = 0; //MULTIPLY_BLOCK_SIZE * bx;

    KW_LOCAL_MEM REAL As[MULTIPLY_BLOCK_SIZE][MULTIPLY_BLOCK_SIZE];
    KW_LOCAL_MEM REAL Bs[MULTIPLY_BLOCK_SIZE + 2][MULTIPLY_BLOCK_SIZE];
    KW_LOCAL_MEM REAL Cs[MULTIPLY_BLOCK_SIZE];
    KW_LOCAL_MEM REAL Ds[MULTIPLY_BLOCK_SIZE];
    KW_LOCAL_MEM REAL Es[MULTIPLY_BLOCK_SIZE + 2];

#ifdef CUDA
    REAL* B0  = &Bs[1][0];
    REAL* Bm1 = &Bs[0][0];
    REAL* Bp1 = &Bs[2][0];
    REAL* E0  = &Es[1];
#elif defined(FW_OPENCL)
    KW_LOCAL_MEM REAL* B0  = &Bs[1][0];
    KW_LOCAL_MEM REAL* Bm1 = &Bs[0][0];
    KW_LOCAL_MEM REAL* Bp1 = &Bs[2][0];
    KW_LOCAL_MEM REAL* E0  = &Es[1];
#endif

    // Zero first row of Bs and Es
    if (ty == 0) {
        Bs[0][tx] = 0;
        if (tx == 0) {
            Es[0] = 0;
        }
    }

    while (d + MULTIPLY_BLOCK_SIZE < PADDED_STATE_COUNT) {

//      READ_SCHUR_VALUES();
        if (ty == 0) {
            Ds[tx] = exp(D[d + tx] * distance);
            Cs[tx] = D[d + PADDED_STATE_COUNT + tx] * distance;
            if (Cs[tx]) {
                REAL expat = Ds[tx];
                REAL cosbt = cos(Cs[tx]);
#ifdef FW_OPENCL_AMDGPU
                Cs[tx] = -expat * sin(Cs[tx] + 0.0);
#else
                Cs[tx] = -expat * sin(Cs[tx]);
#endif
                Ds[tx] *= cosbt;
            }
        }

        // Block read A and B sub-matrices
        As[ty][tx] = A[a + PADDED_STATE_COUNT * ty + tx];
        B0[ty * MULTIPLY_BLOCK_SIZE + tx] = B[b + PADDED_STATE_COUNT * ty + tx];

        // Read extra row of B for Bp1
        if (ty == 0) {
            B0[MULTIPLY_BLOCK_SIZE * MULTIPLY_BLOCK_SIZE + tx] =
                    B[b + PADDED_STATE_COUNT * MULTIPLY_BLOCK_SIZE + tx];
        }

        // All necessary values loaded
        KW_LOCAL_FENCE;

//      POPULATE_SCHUR_BAND(MULTIPLY_BLOCK_SIZE);
        if (ty == 0 && tx == 0) {
            for(int k=0; k<MULTIPLY_BLOCK_SIZE; k++) {
                if (Cs[k] && !Es[k]) {
                    E0[k] = Cs[k];
                } else {
                    E0[k] = 0;
                }
            }
        }


        KW_LOCAL_FENCE;

//      DO_MULTIPLICATION(MULTIPLY_BLOCK_SIZE);
        for (int k = 0; k < MULTIPLY_BLOCK_SIZE; k++) {
            Csub += As[ty][k] * (
                    Ds[k] * B0 [k * MULTIPLY_BLOCK_SIZE + tx]
                  + E0[k] * Bp1[k * MULTIPLY_BLOCK_SIZE + tx]
                  - Es[k] * Bm1[k * MULTIPLY_BLOCK_SIZE + tx]
            );
        }


        // Move last entries in B0 and E0 to first entries in Bs and Es
        if (ty == 0) {
            Bm1[tx] = Bm1[MULTIPLY_BLOCK_SIZE*MULTIPLY_BLOCK_SIZE + tx];
            if (tx == 0) {
                Es[0] = Es[MULTIPLY_BLOCK_SIZE];
            }
        }

        KW_LOCAL_FENCE;

        // Increment sub-matrices
        a += aStep;
        b += bStep;
        d += MULTIPLY_BLOCK_SIZE;

    }

    if (tx < EDGE && ty < EDGE) { // Last block is too long

//      READ_SCHUR_VALUES();
        if (ty == 0) {
            Ds[tx] = exp(D[d + tx] * distance);
            Cs[tx] = D[d + PADDED_STATE_COUNT + tx] * distance;
            if (Cs[tx]) {
                REAL expat = Ds[tx];
                REAL cosbt = cos(Cs[tx]);
#ifdef FW_OPENCL_AMDGPU
                Cs[tx] = -expat * sin(Cs[tx] + 0.0);
#else
                Cs[tx] = -expat * sin(Cs[tx]);
#endif
                Ds[tx] *= cosbt;
            }
        }

        As[ty][tx] = A[a + PADDED_STATE_COUNT * ty + tx];
        B0[ty * MULTIPLY_BLOCK_SIZE + tx] = B[b + PADDED_STATE_COUNT * ty + tx];

    } else {
        if (ty == 0) {
            Ds[tx] = 0;
            Cs[tx] = 0;
        }
        As[ty][tx] = 0;
        B0[ty * MULTIPLY_BLOCK_SIZE + tx] = 0;
    }

    // Zero last row of Bs and Es (only for unrolled iteration at end)
    if (ty == 0) {
        Bs[MULTIPLY_BLOCK_SIZE+1][tx] = 0;
    }

    // All necessary values loaded
    KW_LOCAL_FENCE;

//  POPULATE_SCHUR_BAND(EDGE);
    if (ty == 0 && tx == 0) {
        for(int k=0; k<EDGE; k++) {
            if (Cs[k] && !Es[k]) {
                E0[k] = Cs[k];
            } else {
                E0[k] = 0;
            }
        }
    }

    KW_LOCAL_FENCE;

    // Do matrix multiplication
//  DO_MULTIPLICATION(EDGE);
    for (int k = 0; k < EDGE; k++) {
        Csub += As[ty][k] * (
                Ds[k] * B0 [k * MULTIPLY_BLOCK_SIZE + tx]
              + E0[k] * Bp1[k * MULTIPLY_BLOCK_SIZE + tx]
              - Es[k] * Bm1[k * MULTIPLY_BLOCK_SIZE + tx]
        );
    }


    KW_LOCAL_FENCE;

    // Write the block sub-matrix to device memory;
    // each thread writes one element

    if (Csub < 0)
        Csub = 0;

    if ((tx < EDGE || bx < BLOCKS - 1) && (ty < EDGE || by < BLOCKS - 1)) { // It's OK to write
        C[PADDED_STATE_COUNT* MULTIPLY_BLOCK_SIZE * by + MULTIPLY_BLOCK_SIZE * bx +
              PADDED_STATE_COUNT * ty + tx] = Csub;
    }
#endif
}


KW_GLOBAL_KERNEL void kernelMatrixMulADBComplex(KW_GLOBAL_VAR REAL* dMatrices,
                                   KW_GLOBAL_VAR unsigned int* listC,
                                   KW_GLOBAL_VAR REAL* A,
                                   KW_GLOBAL_VAR REAL* D,
                                   KW_GLOBAL_VAR REAL* B,
                                   KW_GLOBAL_VAR REAL* distanceQueue,
                                   int length,
                                   int wB,
                                   int totalMatrix) {
#if !(defined(FW_OPENCL_APPLEAMDGPU) && defined(DOUBLE_PRECISION)) // TODO: fix this issue
    int wMatrix = KW_GROUP_ID_0 % totalMatrix;

    // Block index
    int bx = KW_GROUP_ID_0 / totalMatrix;
    int by = KW_GROUP_ID_1;
    int BLOCKS = KW_NUM_GROUPS_1;

    // Thread index
    int tx = KW_LOCAL_ID_0;
    int ty = KW_LOCAL_ID_1;

#ifdef CUDA
    KW_LOCAL_MEM REAL* C;
    KW_LOCAL_MEM REAL distance;
    if (tx == 0 && ty == 0) {
        C = dMatrices + listC[wMatrix];
        distance = distanceQueue[wMatrix]; // Non-coalescent read
    }
#elif defined(FW_OPENCL)
    KW_GLOBAL_VAR REAL* C;
    REAL distance;
    C = dMatrices + listC[wMatrix];
    distance = distanceQueue[wMatrix];
#endif

    KW_LOCAL_FENCE;

    const int EDGE = PADDED_STATE_COUNT - (BLOCKS - 1) * MULTIPLY_BLOCK_SIZE;

    // Step size used to iterate through the sub-matrices of A
    int aStep = MULTIPLY_BLOCK_SIZE;

    // Step size used to iterate through the sub-matrices of B
    int bStep = MULTIPLY_BLOCK_SIZE * PADDED_STATE_COUNT;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    REAL Csub = 0;

    int a = PADDED_STATE_COUNT * MULTIPLY_BLOCK_SIZE * by;
    int b = MULTIPLY_BLOCK_SIZE * bx;
    int d = 0; //MULTIPLY_BLOCK_SIZE * bx;

    KW_LOCAL_MEM REAL As[MULTIPLY_BLOCK_SIZE][MULTIPLY_BLOCK_SIZE];
    KW_LOCAL_MEM REAL Bs[MULTIPLY_BLOCK_SIZE + 2][MULTIPLY_BLOCK_SIZE];
    KW_LOCAL_MEM REAL Cs[MULTIPLY_BLOCK_SIZE];
    KW_LOCAL_MEM REAL Ds[MULTIPLY_BLOCK_SIZE];
    KW_LOCAL_MEM REAL Es[MULTIPLY_BLOCK_SIZE + 2];

#ifdef CUDA
   	REAL* B0  = &Bs[1][0];
   	REAL* Bm1 = &Bs[0][0];
   	REAL* Bp1 = &Bs[2][0];
   	REAL* E0  = &Es[1];
#elif defined(FW_OPENCL)
   	KW_LOCAL_MEM REAL* B0  = &Bs[1][0];
   	KW_LOCAL_MEM REAL* Bm1 = &Bs[0][0];
   	KW_LOCAL_MEM REAL* Bp1 = &Bs[2][0];
   	KW_LOCAL_MEM REAL* E0  = &Es[1];
#endif

   	// Zero first row of Bs and Es
   	if (ty == 0) {
   		Bs[0][tx] = 0;
   		if (tx == 0) {
   			Es[0] = 0;
   		}
   	}

    while (d + MULTIPLY_BLOCK_SIZE < PADDED_STATE_COUNT) {

//      READ_SCHUR_VALUES();
		if (ty == 0) {
			Ds[tx] = exp(D[d + tx] * distance);
			Cs[tx] = D[d + PADDED_STATE_COUNT + tx] * distance;
			if (Cs[tx]) {
            	REAL expat = Ds[tx];
            	REAL cosbt = cos(Cs[tx]);
#ifdef FW_OPENCL_AMDGPU
                Cs[tx] = -expat * sin(Cs[tx] + 0.0);
#else
                Cs[tx] = -expat * sin(Cs[tx]);
#endif
            	Ds[tx] *= cosbt;
            }
        }

        // Block read A and B sub-matrices
        As[ty][tx] = A[a + PADDED_STATE_COUNT * ty + tx];
        B0[ty * MULTIPLY_BLOCK_SIZE + tx] = B[b + PADDED_STATE_COUNT * ty + tx];

        // Read extra row of B for Bp1
        if (ty == 0) {
        	B0[MULTIPLY_BLOCK_SIZE * MULTIPLY_BLOCK_SIZE + tx] =
        			B[b + PADDED_STATE_COUNT * MULTIPLY_BLOCK_SIZE + tx];
        }

        // All necessary values loaded
    	KW_LOCAL_FENCE;

//    	POPULATE_SCHUR_BAND(MULTIPLY_BLOCK_SIZE);
		if (ty == 0 && tx == 0) {
			for(int k=0; k<MULTIPLY_BLOCK_SIZE; k++) {
				if (Cs[k] && !Es[k]) {
					E0[k] = Cs[k];
				} else {
					E0[k] = 0;
				}
			}
		}


    	KW_LOCAL_FENCE;

//      DO_MULTIPLICATION(MULTIPLY_BLOCK_SIZE);
		for (int k = 0; k < MULTIPLY_BLOCK_SIZE; k++) {
			Csub += As[ty][k] * (
					Ds[k] * B0 [k * MULTIPLY_BLOCK_SIZE + tx]
				  + E0[k] * Bp1[k * MULTIPLY_BLOCK_SIZE + tx]
				  - Es[k] * Bm1[k * MULTIPLY_BLOCK_SIZE + tx]
			);
		}


        // Move last entries in B0 and E0 to first entries in Bs and Es
        if (ty == 0) {
        	Bm1[tx] = Bm1[MULTIPLY_BLOCK_SIZE*MULTIPLY_BLOCK_SIZE + tx];
        	if (tx == 0) {
        		Es[0] = Es[MULTIPLY_BLOCK_SIZE];
        	}
        }

        KW_LOCAL_FENCE;

        // Increment sub-matrices
        a += aStep;
        b += bStep;
        d += MULTIPLY_BLOCK_SIZE;

    }

    if (tx < EDGE && ty < EDGE) { // Last block is too long

//      READ_SCHUR_VALUES();
		if (ty == 0) {
			Ds[tx] = exp(D[d + tx] * distance);
			Cs[tx] = D[d + PADDED_STATE_COUNT + tx] * distance;
			if (Cs[tx]) {
            	REAL expat = Ds[tx];
            	REAL cosbt = cos(Cs[tx]);
#ifdef FW_OPENCL_AMDGPU
            	Cs[tx] = -expat * sin(Cs[tx] + 0.0);
#else
                Cs[tx] = -expat * sin(Cs[tx]);
#endif
            	Ds[tx] *= cosbt;
            }
        }

        As[ty][tx] = A[a + PADDED_STATE_COUNT * ty + tx];
        B0[ty * MULTIPLY_BLOCK_SIZE + tx] = B[b + PADDED_STATE_COUNT * ty + tx];

    } else {
    	if (ty == 0) {
    		Ds[tx] = 0;
    		Cs[tx] = 0;
    	}
    	As[ty][tx] = 0;
    	B0[ty * MULTIPLY_BLOCK_SIZE + tx] = 0;
    }

	// Zero last row of Bs and Es (only for unrolled iteration at end)
    if (ty == 0) {
    	Bs[MULTIPLY_BLOCK_SIZE+1][tx] = 0;
    }

    // All necessary values loaded
	KW_LOCAL_FENCE;

//	POPULATE_SCHUR_BAND(EDGE);
    if (ty == 0 && tx == 0) {
        for(int k=0; k<EDGE; k++) {
            if (Cs[k] && !Es[k]) {
                E0[k] = Cs[k];
            } else {
                E0[k] = 0;
            }
        }
    }

	KW_LOCAL_FENCE;

	// Do matrix multiplication
//	DO_MULTIPLICATION(EDGE);
    for (int k = 0; k < EDGE; k++) {
        Csub += As[ty][k] * (
                Ds[k] * B0 [k * MULTIPLY_BLOCK_SIZE + tx]
              + E0[k] * Bp1[k * MULTIPLY_BLOCK_SIZE + tx]
              - Es[k] * Bm1[k * MULTIPLY_BLOCK_SIZE + tx]
        );
    }


    KW_LOCAL_FENCE;

    // Write the block sub-matrix to device memory;
    // each thread writes one element

    if (Csub < 0)
    	Csub = 0;

    if ((tx < EDGE || bx < BLOCKS - 1) && (ty < EDGE || by < BLOCKS - 1)) { // It's OK to write
        C[PADDED_STATE_COUNT* MULTIPLY_BLOCK_SIZE * by + MULTIPLY_BLOCK_SIZE * bx +
              PADDED_STATE_COUNT * ty + tx] = Csub;
    }
#endif
}

KW_GLOBAL_KERNEL void kernelSumSites1(KW_GLOBAL_VAR REAL* dArray,
                                      KW_GLOBAL_VAR REAL* dSum,
                                      KW_GLOBAL_VAR REAL* dPatternWeights,
                                      int patternCount) {
#ifdef FW_OPENCL_CPU

    REAL sum = 0;

    int pattern = KW_GROUP_ID_0 * SUM_SITES_BLOCK_SIZE;
    int maxPattern = (KW_GROUP_ID_0 + 1) * SUM_SITES_BLOCK_SIZE;

    if (maxPattern > patternCount)
        maxPattern = patternCount;

    while (pattern < maxPattern) {
        FMA(dArray[pattern],  dPatternWeights[pattern], sum);
        pattern++;
    }

    dSum[KW_GROUP_ID_0] = sum;

#else

    KW_LOCAL_MEM REAL sum[SUM_SITES_BLOCK_SIZE];

    int tx = KW_LOCAL_ID_0;
    int pattern = KW_LOCAL_ID_0 + KW_GROUP_ID_0 * SUM_SITES_BLOCK_SIZE;

    if (pattern < patternCount)
        sum[tx] = dArray[pattern] * dPatternWeights[pattern];
    else
        sum[tx] = 0.0;

    KW_LOCAL_FENCE;

    for (unsigned int s = SUM_SITES_BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tx < s)
            sum[tx] += sum[tx + s];
        KW_LOCAL_FENCE;
    }

    if (tx == 0)
        dSum[KW_GROUP_ID_0] = sum[0];

#endif
}

KW_GLOBAL_KERNEL void kernelSumSites1Partition(KW_GLOBAL_VAR REAL* dArray,
                                               KW_GLOBAL_VAR REAL* dSum,
                                               KW_GLOBAL_VAR REAL* dPatternWeights,
                                               int startPattern,
                                               int endPattern) {
#ifdef FW_OPENCL_CPU

    REAL sum = 0;

    int pattern = startPattern + KW_GROUP_ID_0 * SUM_SITES_BLOCK_SIZE;
    int maxPattern = startPattern + (KW_GROUP_ID_0 + 1) * SUM_SITES_BLOCK_SIZE;

    if (maxPattern > endPattern)
        maxPattern = endPattern;

    while (pattern < maxPattern) {
        FMA(dArray[pattern],  dPatternWeights[pattern], sum);
        pattern++;
    }

    dSum[KW_GROUP_ID_0] = sum;

#else

    KW_LOCAL_MEM REAL sum[SUM_SITES_BLOCK_SIZE];

    int tx = KW_LOCAL_ID_0;
    int pattern = startPattern + KW_LOCAL_ID_0 + KW_GROUP_ID_0 * SUM_SITES_BLOCK_SIZE;

    if (pattern < endPattern)
        sum[tx] = dArray[pattern] * dPatternWeights[pattern];
    else
        sum[tx] = 0.0;

    KW_LOCAL_FENCE;

    for (unsigned int s = SUM_SITES_BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tx < s)
            sum[tx] += sum[tx + s];
        KW_LOCAL_FENCE;
    }

    if (tx == 0)
        dSum[KW_GROUP_ID_0] = sum[0];

#endif
}

// KW_GLOBAL_KERNEL void kernelSumSites1Partition(KW_GLOBAL_VAR REAL*         dArray,
//                                                KW_GLOBAL_VAR REAL*         dSum,
//                                                KW_GLOBAL_VAR REAL*         dPatternWeights,
//                                                KW_GLOBAL_VAR unsigned int* dPtrOffsets) {

//     int opIndexPtr = KW_GROUP_ID_0 * 2;
//     int startPattern = dPtrOffsets[opIndexPtr    ];
//     int endPattern   = dPtrOffsets[opIndexPtr + 1];

// #ifdef FW_OPENCL_CPU

//     REAL sum = 0;

//     int pattern = startPattern + KW_GROUP_ID_0 * SUM_SITES_BLOCK_SIZE;

//     while (pattern < endPattern) {
//         FMA(dArray[pattern],  dPatternWeights[pattern], sum);
//         pattern++;
//     }

//     dSum[KW_GROUP_ID_0] = sum;

// #else

//     KW_LOCAL_MEM REAL sum[SUM_SITES_BLOCK_SIZE];

//     int tx = KW_LOCAL_ID_0;
//     int pattern = startPattern + KW_LOCAL_ID_0 + KW_GROUP_ID_0 * SUM_SITES_BLOCK_SIZE;

//     if (pattern < endPattern)
//         sum[tx] = dArray[pattern] * dPatternWeights[pattern];
//     else
//         sum[tx] = 0.0;

//     KW_LOCAL_FENCE;

//     for (unsigned int s = SUM_SITES_BLOCK_SIZE / 2; s > 0; s >>= 1) {
//         if (tx < s)
//             sum[tx] += sum[tx + s];
//         KW_LOCAL_FENCE;
//     }

//     if (tx == 0)
//         dSum[KW_GROUP_ID_0] = sum[0];

// #endif
// }

KW_GLOBAL_KERNEL void kernelSumSites2(KW_GLOBAL_VAR REAL* dArray1,
                                      KW_GLOBAL_VAR REAL* dSum1,
                                      KW_GLOBAL_VAR REAL* dArray2,
                                      KW_GLOBAL_VAR REAL* dSum2,
                                      KW_GLOBAL_VAR REAL* dPatternWeights,
                                      int patternCount) {

#ifdef FW_OPENCL_CPU

    REAL sum1 = 0, sum2 = 0;

    int pattern = KW_GROUP_ID_0 * SUM_SITES_BLOCK_SIZE;
    int maxPattern = (KW_GROUP_ID_0 + 1) * SUM_SITES_BLOCK_SIZE;

    if (maxPattern > patternCount)
        maxPattern = patternCount;

    while (pattern < maxPattern) {
        FMA(dArray1[pattern],  dPatternWeights[pattern], sum1);
        FMA(dArray2[pattern],  dPatternWeights[pattern], sum2);
        pattern++;
    }

    dSum1[KW_GROUP_ID_0] = sum1;
    dSum2[KW_GROUP_ID_0] = sum2;

#else

    KW_LOCAL_MEM REAL sum1[SUM_SITES_BLOCK_SIZE];
    KW_LOCAL_MEM REAL sum2[SUM_SITES_BLOCK_SIZE];

    int tx = KW_LOCAL_ID_0;
    int pattern = KW_LOCAL_ID_0 + KW_GROUP_ID_0 * SUM_SITES_BLOCK_SIZE;

    if (pattern < patternCount) {
        REAL pWeight = dPatternWeights[pattern];
        sum1[tx] = dArray1[pattern] * pWeight;
        sum2[tx] = dArray2[pattern] * pWeight;
    } else {
        sum1[tx] = 0.0;
        sum2[tx] = 0.0;
    }

    KW_LOCAL_FENCE;

    for (unsigned int s = SUM_SITES_BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tx < s) {
            sum1[tx] += sum1[tx + s];
            sum2[tx] += sum2[tx + s];
        }
        KW_LOCAL_FENCE;
    }

    if (tx == 0) {
        dSum1[KW_GROUP_ID_0] = sum1[0];
        dSum2[KW_GROUP_ID_0] = sum2[0];
    }

#endif
}

KW_GLOBAL_KERNEL void kernelSumSites3(KW_GLOBAL_VAR REAL* dArray1,
                                      KW_GLOBAL_VAR REAL* dSum1,
                                      KW_GLOBAL_VAR REAL* dArray2,
                                      KW_GLOBAL_VAR REAL* dSum2,
                                      KW_GLOBAL_VAR REAL* dArray3,
                                      KW_GLOBAL_VAR REAL* dSum3,
                                      KW_GLOBAL_VAR REAL* dPatternWeights,
                                      int patternCount) {

#ifdef FW_OPENCL_CPU

    REAL sum1 = 0, sum2 = 0, sum3 = 0;

    int pattern = KW_GROUP_ID_0 * SUM_SITES_BLOCK_SIZE;
    int maxPattern = (KW_GROUP_ID_0 + 1) * SUM_SITES_BLOCK_SIZE;

    if (maxPattern > patternCount)
        maxPattern = patternCount;

    while (pattern < maxPattern) {
        FMA(dArray1[pattern],  dPatternWeights[pattern], sum1);
        FMA(dArray2[pattern],  dPatternWeights[pattern], sum2);
        FMA(dArray3[pattern],  dPatternWeights[pattern], sum3);

        pattern++;
    }

    dSum1[KW_GROUP_ID_0] = sum1;
    dSum2[KW_GROUP_ID_0] = sum2;
    dSum3[KW_GROUP_ID_0] = sum3;

#else

    KW_LOCAL_MEM REAL sum1[SUM_SITES_BLOCK_SIZE];
    KW_LOCAL_MEM REAL sum2[SUM_SITES_BLOCK_SIZE];
    KW_LOCAL_MEM REAL sum3[SUM_SITES_BLOCK_SIZE];

    int tx = KW_LOCAL_ID_0;
    int pattern = KW_LOCAL_ID_0 + KW_GROUP_ID_0 * SUM_SITES_BLOCK_SIZE;

    if (pattern < patternCount) {
        REAL pWeight = dPatternWeights[pattern];
        sum1[tx] = dArray1[pattern] * pWeight;
        sum2[tx] = dArray2[pattern] * pWeight;
        sum3[tx] = dArray3[pattern] * pWeight;
    } else {
        sum1[tx] = 0.0;
        sum2[tx] = 0.0;
        sum3[tx] = 0.0;
    }

    KW_LOCAL_FENCE;

    for (unsigned int s = SUM_SITES_BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tx < s) {
            sum1[tx] += sum1[tx + s];
            sum2[tx] += sum2[tx + s];
            sum3[tx] += sum3[tx + s];
        }
        KW_LOCAL_FENCE;
    }

    if (tx == 0) {
        dSum1[KW_GROUP_ID_0] = sum1[0];
        dSum2[KW_GROUP_ID_0] = sum2[0];
        dSum3[KW_GROUP_ID_0] = sum3[0];
    }

#endif
}

KW_GLOBAL_KERNEL void kernelAccumulateFactors(KW_GLOBAL_VAR REAL* dScalingFactors,
                                              KW_GLOBAL_VAR unsigned int* dNodePtrQueue,
                                              KW_GLOBAL_VAR REAL* rootScaling,
                                              int nodeCount,
                                              int patternCount) {

    int pattern = KW_LOCAL_ID_0 + KW_GROUP_ID_0 * PATTERN_BLOCK_SIZE;

    REAL total = 0;
    KW_GLOBAL_VAR REAL* nodeScales;

    int n;
    for(n = 0; n < nodeCount; n++) {
//      if (KW_LOCAL_ID_0 == 0) // TODO Why does this not work???
        nodeScales = dScalingFactors + dNodePtrQueue[n];
//      KW_LOCAL_FENCE;

    #ifdef KERNEL_PRINT_ENABLED
        if (pattern == 1)
            printf("added %1.2e\n", nodeScales[pattern]);
    #endif
        REAL factor = nodeScales[pattern];
        if (factor != 1.0) {
            total += log(factor);
        }
    }

#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    rootScaling[pattern] += total;
#else // GPU implementation
    if (pattern < patternCount)
        rootScaling[pattern] += total;
#endif // FW_OPENCL_CPU
}

KW_GLOBAL_KERNEL void kernelAccumulateFactorsByPartition(KW_GLOBAL_VAR REAL* dScalingFactors,
                                                         KW_GLOBAL_VAR unsigned int* dNodePtrQueue,
                                                         KW_GLOBAL_VAR REAL* rootScaling,
                                                         int nodeCount,
                                                         int startPattern,
                                                         int endPattern) {

    int pattern = startPattern + KW_LOCAL_ID_0 + KW_GROUP_ID_0 * PATTERN_BLOCK_SIZE;

    REAL total = 0;
    KW_GLOBAL_VAR REAL* nodeScales;

    int n;
    for(n = 0; n < nodeCount; n++) {
        nodeScales = dScalingFactors + dNodePtrQueue[n];

        REAL factor = nodeScales[pattern];
        if (factor != 1.0) {
            total += log(factor);
        }
    }

    if (pattern < endPattern) {
        rootScaling[pattern] += total;
    }
}

KW_GLOBAL_KERNEL void kernelAccumulateFactorsScalersLog(KW_GLOBAL_VAR REAL* dScalingFactors,
                                                 KW_GLOBAL_VAR unsigned int* dNodePtrQueue,
                                                 KW_GLOBAL_VAR REAL* rootScaling,
                                                 int nodeCount,
                                                 int patternCount) {
    int pattern = KW_LOCAL_ID_0 + KW_GROUP_ID_0 * PATTERN_BLOCK_SIZE;

    REAL total = 0;
    KW_GLOBAL_VAR REAL* nodeScales;

    int n;
    for(n = 0; n < nodeCount; n++) {
//      if (KW_LOCAL_ID_0 == 0) // TODO Why does this not work???
        nodeScales = dScalingFactors + dNodePtrQueue[n];
//      KW_LOCAL_FENCE;

#ifdef KERNEL_PRINT_ENABLED
        if (pattern == 1)
            printf("added %1.2e\n", nodeScales[pattern]);
#endif
        total += nodeScales[pattern];
    }

#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    rootScaling[pattern] += total;
#else // GPU implementation
    if (pattern < patternCount)
        rootScaling[pattern] += total;
#endif // FW_OPENCL_CPU
}

KW_GLOBAL_KERNEL void kernelAccumulateFactorsScalersLogByPartition(
                                                KW_GLOBAL_VAR REAL* dScalingFactors,
                                                KW_GLOBAL_VAR unsigned int* dNodePtrQueue,
                                                KW_GLOBAL_VAR REAL* rootScaling,
                                                int nodeCount,
                                                int startPattern,
                                                int endPattern) {

    int pattern = startPattern + KW_LOCAL_ID_0 + KW_GROUP_ID_0 * PATTERN_BLOCK_SIZE;

    REAL total = 0;
    KW_GLOBAL_VAR REAL* nodeScales;

    int n;
    for(n = 0; n < nodeCount; n++) {
        nodeScales = dScalingFactors + dNodePtrQueue[n];

        total += nodeScales[pattern];
    }

    if (pattern < endPattern) {
        rootScaling[pattern] += total;
    }
}

KW_GLOBAL_KERNEL void kernelRemoveFactors(KW_GLOBAL_VAR REAL* dScalingFactors,
                                    KW_GLOBAL_VAR unsigned int* dNodePtrQueue,
                                                   KW_GLOBAL_VAR REAL* rootScaling,
                                                   int nodeCount,
                                                   int patternCount) {
    int pattern = KW_LOCAL_ID_0 + KW_GROUP_ID_0 * PATTERN_BLOCK_SIZE;

    REAL total = 0;
    KW_GLOBAL_VAR REAL* nodeScales;

    int n;
    for(n = 0; n < nodeCount; n++) {
//      if (KW_LOCAL_ID_0 == 0) // TODO Why does this not work???
        nodeScales = dScalingFactors + dNodePtrQueue[n];
//      KW_LOCAL_FENCE;

#ifdef KERNEL_PRINT_ENABLED
        if (pattern == 1)
            printf("added %1.2e\n", nodeScales[pattern]);
#endif
        REAL factor = nodeScales[pattern];
        if (factor != 1.0) {
            total += log(factor);
        }
    }

#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    rootScaling[pattern] -= total;
#else // GPU implementation
    if (pattern < patternCount)
        rootScaling[pattern] -= total;
#endif // FW_OPENCL_CPU
}

KW_GLOBAL_KERNEL void kernelRemoveFactorsByPartition(KW_GLOBAL_VAR REAL* dScalingFactors,
                                                     KW_GLOBAL_VAR unsigned int* dNodePtrQueue,
                                                     KW_GLOBAL_VAR REAL* rootScaling,
                                                     int nodeCount,
                                                     int startPattern,
                                                     int endPattern) {
    int pattern = startPattern + KW_LOCAL_ID_0 + KW_GROUP_ID_0 * PATTERN_BLOCK_SIZE;

    REAL total = 0;
    KW_GLOBAL_VAR REAL* nodeScales;

    int n;
    for(n = 0; n < nodeCount; n++) {
        nodeScales = dScalingFactors + dNodePtrQueue[n];

        REAL factor = nodeScales[pattern];
        if (factor != 1.0) {
            total += log(factor);
        }
    }

    if (pattern < endPattern) {
        rootScaling[pattern] -= total;
    }
}

KW_GLOBAL_KERNEL void kernelRemoveFactorsScalersLog(KW_GLOBAL_VAR REAL* dScalingFactors,
                                             KW_GLOBAL_VAR unsigned int* dNodePtrQueue,
                                             KW_GLOBAL_VAR REAL* rootScaling,
                                             int nodeCount,
                                             int patternCount) {
    int pattern = KW_LOCAL_ID_0 + KW_GROUP_ID_0 * PATTERN_BLOCK_SIZE;

    REAL total = 0;
    KW_GLOBAL_VAR REAL* nodeScales;

    int n;
    for(n = 0; n < nodeCount; n++) {
//      if (KW_LOCAL_ID_0 == 0) // TODO Why does this not work???
        nodeScales = dScalingFactors + dNodePtrQueue[n];
//      KW_LOCAL_FENCE;

#ifdef KERNEL_PRINT_ENABLED
        if (pattern == 1)
            printf("added %1.2e\n", nodeScales[pattern]);
#endif

        total += nodeScales[pattern];
    }

#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    rootScaling[pattern] -= total;
#else // GPU implementation
    if (pattern < patternCount)
        rootScaling[pattern] -= total;
#endif // FW_OPENCL_CPU
}

KW_GLOBAL_KERNEL void kernelRemoveFactorsScalersLogByPartition(KW_GLOBAL_VAR REAL* dScalingFactors,
                                                               KW_GLOBAL_VAR unsigned int* dNodePtrQueue,
                                                               KW_GLOBAL_VAR REAL* rootScaling,
                                                               int nodeCount,
                                                               int startPattern,
                                                               int endPattern) {
    int pattern = startPattern + KW_LOCAL_ID_0 + KW_GROUP_ID_0 * PATTERN_BLOCK_SIZE;

    REAL total = 0;
    KW_GLOBAL_VAR REAL* nodeScales;

    int n;
    for(n = 0; n < nodeCount; n++) {
        nodeScales = dScalingFactors + dNodePtrQueue[n];

        total += nodeScales[pattern];
    }

    if (pattern < endPattern)
        rootScaling[pattern] -= total;

}

KW_GLOBAL_KERNEL void kernelResetFactorsByPartition(KW_GLOBAL_VAR REAL* dScalingFactors,
                                                    int startPattern,
                                                    int endPattern) {
    int pattern = startPattern + KW_LOCAL_ID_0 + KW_GROUP_ID_0 * PATTERN_BLOCK_SIZE;

    if (pattern < endPattern) {
        dScalingFactors[pattern] = 0.0;
    }
}


KW_GLOBAL_KERNEL void kernelPartialsDynamicScalingSlow(KW_GLOBAL_VAR REAL* allPartials,
                                                 KW_GLOBAL_VAR REAL* scalingFactors,
                                                 int matrixCount) {
    int state = KW_LOCAL_ID_0;
    int pattern = KW_GROUP_ID_0;
    int patternCount = KW_NUM_GROUPS_0;

    KW_LOCAL_MEM REAL partials[PADDED_STATE_COUNT];

    KW_LOCAL_MEM REAL max;

    if (state == 0)
        max = 0.0;

    int m;
    for(m = 0; m < matrixCount; m++) {
        partials[state] = allPartials[m * patternCount * PADDED_STATE_COUNT + pattern *
                                      PADDED_STATE_COUNT + state];
        KW_LOCAL_FENCE;

#ifdef IS_POWER_OF_TWO
    // parallelized reduction *** only works for powers-of-2 ****
    for (int i = PADDED_STATE_COUNT / 2; i > 0; i >>= 1) {
        if (state < i) {
#else
    for (int i = SMALLEST_POWER_OF_TWO / 2; i > 0; i >>= 1) {
        if (state < i && state + i < PADDED_STATE_COUNT ) {
#endif // IS_POWER_OF_TWO
                REAL compare1 = partials[state];
                REAL compare2 = partials[state + i];
                if(compare2 > compare1)
                    partials[state] = compare2;
            }
            KW_LOCAL_FENCE;
        }
        if(state == 0) {
            if( partials[0] > max)
                max = partials[0];
        }
    }

    if(state == 0) {
        if (max == 0)
        	max = 1.0;
        scalingFactors[pattern] = max;
    }


    KW_LOCAL_FENCE;

    for(m = 0; m < matrixCount; m++)
        allPartials[m * patternCount * PADDED_STATE_COUNT + pattern * PADDED_STATE_COUNT +
                    state] /= max;

}

KW_GLOBAL_KERNEL void kernelPartialsDynamicScalingSlowScalersLog(KW_GLOBAL_VAR REAL* allPartials,
                                                          KW_GLOBAL_VAR REAL* scalingFactors,
                                                          int matrixCount) {
    int state = KW_LOCAL_ID_0;
    int pattern = KW_GROUP_ID_0;
    int patternCount = KW_NUM_GROUPS_0;

    KW_LOCAL_MEM REAL partials[PADDED_STATE_COUNT];

    KW_LOCAL_MEM REAL max;

    if (state == 0)
        max = 0.0;

    int m;
    for(m = 0; m < matrixCount; m++) {
        partials[state] = allPartials[m * patternCount * PADDED_STATE_COUNT + pattern *
                                      PADDED_STATE_COUNT + state];
        KW_LOCAL_FENCE;

#ifdef IS_POWER_OF_TWO
    // parallelized reduction *** only works for powers-of-2 ****
    for (int i = PADDED_STATE_COUNT / 2; i > 0; i >>= 1) {
        if (state < i) {
#else
    for (int i = SMALLEST_POWER_OF_TWO / 2; i > 0; i >>= 1) {
        if (state < i && state + i < PADDED_STATE_COUNT ) {
#endif // IS_POWER_OF_TWO
                REAL compare1 = partials[state];
                REAL compare2 = partials[state + i];
                if(compare2 > compare1)
                    partials[state] = compare2;
            }
            KW_LOCAL_FENCE;
        }
        if(state == 0) {
            if( partials[0] > max)
                max = partials[0];
        }
    }

    if(state == 0) {
        if (max == 0) {
        	max = 1.0;
            scalingFactors[pattern] = 0.0;
        } else {
            scalingFactors[pattern] = log(max);
        }
    }


    KW_LOCAL_FENCE;

    for(m = 0; m < matrixCount; m++)
        allPartials[m * patternCount * PADDED_STATE_COUNT + pattern * PADDED_STATE_COUNT +
                    state] /= max;

}

KW_GLOBAL_KERNEL void kernelMultipleNodeSiteReduction(KW_GLOBAL_VAR REAL* dOut,
                                                      KW_GLOBAL_VAR REAL* dIn,
                                                      KW_GLOBAL_VAR REAL* dPatternWeights,
                                                      int outOffset,
                                                      int patternCount) {
#ifdef FW_OPENCL_CPU
    // TODO
#else

    KW_LOCAL_MEM REAL reduce[MULTI_NODE_SUM_BLOCK_SIZE];

    int tx = KW_LOCAL_ID_0;
    int node = KW_GROUP_ID_0;
    int offset = patternCount * node;
    int pattern = tx;

    REAL sum = 0;

    while (pattern < patternCount) {
        FMA(dIn[offset + pattern], dPatternWeights[pattern], sum);
        pattern += MULTI_NODE_SUM_BLOCK_SIZE;
    }

    reduce[tx] = sum;

    KW_LOCAL_FENCE;

    for (unsigned int s = MULTI_NODE_SUM_BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tx < s) {
            reduce[tx] += reduce[tx + s];
        }
        KW_LOCAL_FENCE;
    }

    if (tx == 0) {
        dOut[outOffset + node] = reduce[0];
    }
#endif
}

KW_GLOBAL_KERNEL void kernelMultipleNodeSiteSquaredReduction(KW_GLOBAL_VAR REAL* dOut,
                                                             KW_GLOBAL_VAR REAL* dIn,
                                                             KW_GLOBAL_VAR REAL* dPatternWeights,
                                                             int outOffset,
                                                             int patternCount) {
#ifdef FW_OPENCL_CPU
    // TODO
#else

    KW_LOCAL_MEM REAL reduce[MULTI_NODE_SUM_BLOCK_SIZE];

    int tx = KW_LOCAL_ID_0;
    int node = KW_GROUP_ID_0;
    int offset = patternCount * node;
    int pattern = tx;

    REAL sum = 0;

    while (pattern < patternCount) {
        REAL value = dIn[offset + pattern];
        FMA(value * value, dPatternWeights[pattern], sum);
        pattern += MULTI_NODE_SUM_BLOCK_SIZE;
    }

    reduce[tx] = sum;

    KW_LOCAL_FENCE;

    for (unsigned int s = MULTI_NODE_SUM_BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tx < s) {
            reduce[tx] += reduce[tx + s];
        }
        KW_LOCAL_FENCE;
    }

    if (tx == 0) {
        dOut[outOffset + node] = reduce[0];
    }
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////
// scaling experiments kernels

KW_GLOBAL_KERNEL void kernelAccumulateFactorsAutoScaling(KW_GLOBAL_VAR signed char* dScalingFactors,
                                                   KW_GLOBAL_VAR unsigned int* dNodePtrQueue,
                                                   KW_GLOBAL_VAR int* rootScaling,
                                                   int nodeCount,
                                                   int patternCount,
                                                   int scaleBufferSize) {
    int pattern = KW_LOCAL_ID_0 + KW_GROUP_ID_0 * PATTERN_BLOCK_SIZE;
    int index = pattern + KW_GROUP_ID_1 * patternCount;

    int total = 0;
    KW_GLOBAL_VAR signed char* nodeScales;

    int n;
    for(n = 0; n < nodeCount; n++) {
//        int sIndex = dNodePtrQueue[n];
        nodeScales = dScalingFactors + dNodePtrQueue[n] * scaleBufferSize;

        total += nodeScales[index];
    }

    if (pattern < patternCount)
        rootScaling[index] = total;
}


/*
 * BASTA kernels
 */

KW_GLOBAL_KERNEL void kernelInnerBastaPartialsCoalescent(KW_GLOBAL_VAR REAL* KW_RESTRICT partials,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT matrices,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT operations,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT sizes,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT coalescent,
													int intervalNumber,
													int start,
                                                    int numOps,
                                                    int totalPatterns) {

    #define PATTERN_BLOCK_SIZE_B 32
    #define BLOCK_PEELING_SIZE_B PADDED_STATE_COUNT

    int state = KW_LOCAL_ID_0;
    int patIdx = KW_LOCAL_ID_1;
    int pattern = __umul24(KW_GROUP_ID_0,PATTERN_BLOCK_SIZE_B) + patIdx;

    KW_LOCAL_MEM REAL sMatrix1[BLOCK_PEELING_SIZE_B][PADDED_STATE_COUNT];

    KW_LOCAL_MEM REAL sPartials1[PATTERN_BLOCK_SIZE_B][PADDED_STATE_COUNT];
    int op = pattern + start;
    int maxOp = start + totalPatterns - 1;
    int desIndex = operations[op * numOps];
    int child1PartialIndex = operations[op * numOps + 1];
    int child1TransIndex = operations[maxOp * numOps + 2];
    int child2PartialIndex = operations[op * numOps + 3];
    int child2TransIndex = operations[op * numOps + 4];
    int accumulation1PartialIndex = operations[op * numOps + 5];
    int accumulation2PartialIndex = operations[op * numOps + 6];

    KW_LOCAL_MEM REAL sPartials3[PATTERN_BLOCK_SIZE_B][PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL popSizes[PADDED_STATE_COUNT];

    if (pattern < totalPatterns) {
        KW_GLOBAL_VAR REAL* KW_RESTRICT partials1 = partials + child1PartialIndex;
        sPartials1[patIdx][state] = partials1[state];
    } else {
        sPartials1[patIdx][state] = 0;
    }
    REAL sum1 = 0;

    KW_GLOBAL_VAR REAL* KW_RESTRICT matrix1 = matrices + child1TransIndex;
    for (int i = 0; i < PADDED_STATE_COUNT; i += BLOCK_PEELING_SIZE_B) {
        /* load one row of matrices */
        if (patIdx < BLOCK_PEELING_SIZE_B) {
            /* These are all coherent global memory reads. */
            sMatrix1[patIdx][state] = matrix1[patIdx * PADDED_STATE_COUNT + state];
            /* sMatrix now filled with starting in state and ending in i */
            matrix1 += BLOCK_PEELING_SIZE_B * PADDED_STATE_COUNT;
        }
        KW_LOCAL_FENCE;
        for(int j = 0; j < BLOCK_PEELING_SIZE_B; j++) {
            FMA(sMatrix1[j][state], sPartials1[patIdx][i + j], sum1);
        }
        KW_LOCAL_FENCE;
    }


    if (pattern < totalPatterns) {
        KW_GLOBAL_VAR REAL* KW_RESTRICT partials3 = partials + desIndex;
        partials3[state] = sum1;
    }


    /* copy PADDED_STATE_COUNT*PATTERN_BLOCK_SIZE lengthed partials */
    /* These are all coherent global memory reads; checked in Profiler */
    if (child2PartialIndex >= 0) {
        KW_LOCAL_MEM REAL sPartials2[PATTERN_BLOCK_SIZE_B][PADDED_STATE_COUNT];

        if (pattern < totalPatterns) {
            KW_GLOBAL_VAR REAL* KW_RESTRICT partials2 = partials + child2PartialIndex;
            sPartials2[patIdx][state] = partials2[state];
            popSizes[state] = sizes[state];
        } else {
            sPartials2[patIdx][state] = 0;
            popSizes[state] = 0;
        }

        if (patIdx == 0) {
            popSizes[state] = sizes[state];
        }

        REAL sum2 = 0;

        for (int i = 0; i < PADDED_STATE_COUNT; i += BLOCK_PEELING_SIZE_B) {
            for(int j = 0; j < BLOCK_PEELING_SIZE_B; j++) {
                FMA(sMatrix1[j][state],  sPartials2[patIdx][i + j], sum2);
            }
            KW_LOCAL_FENCE;
        }


	    if (pattern < totalPatterns) {
	        KW_GLOBAL_VAR REAL* KW_RESTRICT partials3 = partials + desIndex;
	        KW_GLOBAL_VAR REAL* KW_RESTRICT accumulation1 = partials + accumulation1PartialIndex;
	        KW_GLOBAL_VAR REAL* KW_RESTRICT accumulation2 = partials + accumulation2PartialIndex;
		    accumulation1[state] = sum1;
		    accumulation2[state] = sum2;
		    if (popSizes[state] > 0) {
                partials3[state] = sum1 * sum2 / popSizes[state];
            } else {
                partials3[state] = 0;
            }
	        sPartials3[patIdx][state] = partials3[state];

#ifdef IS_POWER_OF_TWO
	        // parallelized reduction *** only works for powers-of-2 ****
	        for (int i = PADDED_STATE_COUNT / 2; i > 0; i >>= 1) {
	            if (state < i) {
#else
	        for (int i = SMALLEST_POWER_OF_TWO / 2; i > 0; i >>= 1) {
	            if (state < i && state + i < PADDED_STATE_COUNT) {
#endif // IS_POWER_OF_TWO
	                sPartials3[patIdx][state] += sPartials3[patIdx][state + i];
	            }
	            KW_LOCAL_FENCE;
	        }
			REAL denominator = sPartials3[patIdx][0];
		    partials3[state] = partials3[state] / denominator;
			coalescent[intervalNumber] = denominator;
	   }
    }
}



//KW_GLOBAL_KERNEL void kernelBastaReduceWithinInterval(KW_GLOBAL_VAR REAL* e,
                                                    //KW_GLOBAL_VAR REAL*  f,
                                                    //KW_GLOBAL_VAR REAL*  g,
                                                    //KW_GLOBAL_VAR REAL*  h,
                                                    //KW_GLOBAL_VAR REAL* KW_RESTRICT startPartials1,
                                                    //KW_GLOBAL_VAR REAL* KW_RESTRICT startPartials2,
                                                    //KW_GLOBAL_VAR REAL* KW_RESTRICT endPartials1,
                                                    //KW_GLOBAL_VAR REAL* KW_RESTRICT endPartials2,
													//int intervalNumber,
                                                    //int child2PartialIndex,
                                                    //int renew) {

    //#define SUM_SITES_BLOCK_SIZE_B 64
	//#define SUM_PARTIAL_BLOCK_SIZE_B 4
    //int state = KW_LOCAL_ID_0;
    //int u = KW_LOCAL_ID_0 + KW_GROUP_ID_0 * SUM_SITES_BLOCK_SIZE_B;
    //int y = intervalNumber * PADDED_STATE_COUNT;
	//KW_LOCAL_MEM REAL sPartials1[SUM_PARTIAL_BLOCK_SIZE_B][PADDED_STATE_COUNT];

    //if (state < PADDED_STATE_COUNT) {
		//sPartials1[0][state] = startPartials1[u];
		//sPartials1[1][state] = startPartials1[u] * startPartials1[u];
		//sPartials1[2][state] = endPartials1[u];
		//sPartials1[3][state] = endPartials1[u] * endPartials1[u];
    //} else {
        //sPartials1[0][state] = 0;
		//sPartials1[1][state] = 0;
		//sPartials1[2][state] = 0;
		//sPartials1[3][state] = 0;
    //}

    //if (renew == 0) {
        //e[y + state] = sPartials1[0][state];
        //f[y + state] = sPartials1[1][state];
        //g[y + state] = sPartials1[2][state];
        //h[y + state] = sPartials1[3][state];
    //} else {
        //e[y + state] += sPartials1[0][state];
        //f[y + state] += sPartials1[1][state];
        //g[y + state] += sPartials1[2][state];
        //h[y + state] += sPartials1[3][state];
    //}

	//KW_LOCAL_MEM REAL sPartials2[SUM_PARTIAL_BLOCK_SIZE_B][PADDED_STATE_COUNT];
    //if (child2PartialIndex >= 0) {
    	//if (state < PADDED_STATE_COUNT) {
			//sPartials2[0][state] = startPartials2[u];
			//sPartials2[1][state] = startPartials2[u] * startPartials2[u];
			//sPartials2[2][state] = endPartials2[u];
			//sPartials2[3][state] = endPartials2[u] * endPartials2[u];
    	//} else {
        	//sPartials2[0][state] = 0;
			//sPartials2[1][state] = 0;
			//sPartials2[2][state] = 0;
			//sPartials2[3][state] = 0;
    	///}
	//e[y + state] += sPartials2[0][state];
	//f[y + state] += sPartials2[1][state];
	//g[y + state] += sPartials2[2][state];
	//h[y + state] += sPartials2[3][state];
	//}
//}


KW_GLOBAL_KERNEL void kernelPreProcessBastaFlags(KW_GLOBAL_VAR REAL* KW_RESTRICT intervals,
                                                                    KW_GLOBAL_VAR REAL* flags,
                                                                    KW_GLOBAL_VAR REAL* blockEnds,
                                                                    int operationCount,
                                                                    int numBlocks) {
#define SUM_PARTIAL_BLOCK_SIZE_B 32
        int opIdx = KW_LOCAL_ID_0;
        int opBlock = KW_GROUP_ID_0;
        int opNumber = KW_LOCAL_ID_0 + opBlock * SUM_PARTIAL_BLOCK_SIZE_B;
        int blockSize = SUM_PARTIAL_BLOCK_SIZE_B;

        if (opNumber < operationCount) {
            if (opNumber == 0 || intervals[opNumber] != intervals[opNumber - 1]) {
                flags[opNumber] = 1;
            } else {
                flags[opNumber] = 0;
            }
        }

        KW_LOCAL_FENCE;

        if (opBlock < numBlocks && opIdx == 0) {
            int start_index = opBlock * blockSize;
            int end_index = min(start_index + blockSize - 1, operationCount - 1);
            blockEnds[opBlock] = intervals[end_index];
        }
}


KW_GLOBAL_KERNEL void kernelAccumulateCarryOut(KW_GLOBAL_VAR REAL* e,
                                                            KW_GLOBAL_VAR REAL* f,
                                                            KW_GLOBAL_VAR REAL* g,
                                                            KW_GLOBAL_VAR REAL* h,
                                                            KW_GLOBAL_VAR REAL* eRes,
                                                            KW_GLOBAL_VAR REAL* fRes,
                                                            KW_GLOBAL_VAR REAL* gRes,
                                                            KW_GLOBAL_VAR REAL* hRes,
                                                            KW_GLOBAL_VAR REAL* keys,
                                                            int numBlocks) {
#define SUM_INTERVAL_BLOCK_SIZE 32
            int tid = KW_LOCAL_ID_0;
            int group = KW_GROUP_ID_0;
            int state = tid % PADDED_STATE_COUNT;
            int blockIdx = tid / PADDED_STATE_COUNT;
            int blockNumber = group * SUM_INTERVAL_BLOCK_SIZE + blockIdx;
            int u = state + blockNumber * PADDED_STATE_COUNT;

            KW_LOCAL_MEM REAL sResE[SUM_INTERVAL_BLOCK_SIZE * PADDED_STATE_COUNT];
            KW_LOCAL_MEM REAL sResF[SUM_INTERVAL_BLOCK_SIZE * PADDED_STATE_COUNT];
            KW_LOCAL_MEM REAL sResG[SUM_INTERVAL_BLOCK_SIZE * PADDED_STATE_COUNT];
            KW_LOCAL_MEM REAL sResH[SUM_INTERVAL_BLOCK_SIZE * PADDED_STATE_COUNT];
            KW_LOCAL_MEM REAL sSegmentEnds[SUM_INTERVAL_BLOCK_SIZE];
            if (blockNumber < numBlocks && (state < PADDED_STATE_COUNT)) {
                sSegmentEnds[blockIdx] = keys[blockNumber];
                sResE[tid] = eRes[u];
                sResF[tid] = fRes[u];
                sResG[tid] = gRes[u];
                sResH[tid] = hRes[u];
            } else {
                sSegmentEnds[blockIdx] = -1;
                sResE[tid] = 0;
                sResF[tid] = 0;
                sResG[tid] = 0;
                sResH[tid] = 0;
            }

            KW_LOCAL_FENCE;

            if (blockNumber < numBlocks && (state < PADDED_STATE_COUNT)) {
                int blockInterval = sSegmentEnds[blockIdx];
                atomicAdd(&e[blockInterval * PADDED_STATE_COUNT + state], sResE[tid]);
                atomicAdd(&f[blockInterval * PADDED_STATE_COUNT + state], sResF[tid]);
                atomicAdd(&g[blockInterval * PADDED_STATE_COUNT + state], sResG[tid]);
                atomicAdd(&h[blockInterval * PADDED_STATE_COUNT + state], sResH[tid]);
            }
            KW_LOCAL_FENCE;
}


KW_GLOBAL_KERNEL void kernelBastaReduceWithinInterval(KW_GLOBAL_VAR REAL* KW_RESTRICT operations,
                                                            KW_GLOBAL_VAR REAL* KW_RESTRICT partials,
                                                            KW_GLOBAL_VAR REAL* e,
                                                            KW_GLOBAL_VAR REAL* f,
                                                            KW_GLOBAL_VAR REAL* g,
                                                            KW_GLOBAL_VAR REAL* h,
                                                            KW_GLOBAL_VAR REAL* eRes,
                                                            KW_GLOBAL_VAR REAL* fRes,
                                                            KW_GLOBAL_VAR REAL* gRes,
                                                            KW_GLOBAL_VAR REAL* hRes,
                                                            KW_GLOBAL_VAR REAL* intervals,
                                                            KW_GLOBAL_VAR REAL* flags,
                                                            int numOps,
                                                            int start,
                                                            int end) {

#define SUM_PARTIAL_BLOCK_SIZE_B 32

        int state = KW_LOCAL_ID_0;
        int opIdx = KW_LOCAL_ID_1;
        int opBlock = KW_GROUP_ID_0;
        int opNumber = KW_LOCAL_ID_1 + opBlock * SUM_PARTIAL_BLOCK_SIZE_B;
        int opCount = end - start;
        // int opCountLast = opCount % SUM_PARTIAL_BLOCK_SIZE_B;
        // int blockSize = (SUM_PARTIAL_BLOCK_SIZE_B > opCountLast) ? opCountLast : SUM_PARTIAL_BLOCK_SIZE_B;
        int blockSize = SUM_PARTIAL_BLOCK_SIZE_B;
        KW_LOCAL_MEM REAL sPartialsE[SUM_PARTIAL_BLOCK_SIZE_B][PADDED_STATE_COUNT];
        KW_LOCAL_MEM REAL sPartialsF[SUM_PARTIAL_BLOCK_SIZE_B][PADDED_STATE_COUNT];
        KW_LOCAL_MEM REAL sPartialsG[SUM_PARTIAL_BLOCK_SIZE_B][PADDED_STATE_COUNT];
        KW_LOCAL_MEM REAL sPartialsH[SUM_PARTIAL_BLOCK_SIZE_B][PADDED_STATE_COUNT];
        KW_LOCAL_MEM REAL sSegmentKeys[SUM_PARTIAL_BLOCK_SIZE_B];
        KW_LOCAL_MEM REAL sSegmentFlags[SUM_PARTIAL_BLOCK_SIZE_B];

        if (opNumber < opCount && (state < PADDED_STATE_COUNT)) {
            int op = opNumber + start;
            int child1PartialIndex = operations[op * numOps + 1];
            int accumulation1PartialIndex = operations[op * numOps + 5];
            KW_GLOBAL_VAR REAL* startPartials1 = partials + child1PartialIndex;
            KW_GLOBAL_VAR REAL* endPartials1 = partials + accumulation1PartialIndex;
            KW_GLOBAL_VAR REAL* partialIntervals = intervals + start;
            sSegmentFlags[opIdx] = flags[op];
            sSegmentKeys[opIdx] = partialIntervals[op];
            sPartialsE[opIdx][state] = startPartials1[state];
            sPartialsF[opIdx][state] = startPartials1[state] * startPartials1[state];
            sPartialsG[opIdx][state] = endPartials1[state];
            sPartialsH[opIdx][state] = endPartials1[state] * endPartials1[state];
        } else {
            sPartialsE[opIdx][state] = 0;
            sPartialsF[opIdx][state] = 0;
            sPartialsG[opIdx][state] = 0;
            sPartialsH[opIdx][state] = 0;
            sSegmentKeys[opIdx] = -1;
            sSegmentFlags[opIdx] = 0;
        }

        KW_LOCAL_FENCE;


        int intervalNumber = sSegmentKeys[opIdx];
        int y = intervalNumber * PADDED_STATE_COUNT;
        int u = opBlock * PADDED_STATE_COUNT;

        if (opNumber < opCount && state < PADDED_STATE_COUNT) {
            int op = opNumber + start;
            int child2PartialIndex = operations[op * numOps + 3];
            int accumulation2PartialIndex = operations[op * numOps + 6];
            KW_GLOBAL_VAR REAL* startPartials2 = partials + child2PartialIndex;
            KW_GLOBAL_VAR REAL* endPartials2 = partials + accumulation2PartialIndex;

            if (child2PartialIndex >= 0) {
                e[y + state] = startPartials2[state];
                f[y + state] = startPartials2[state] * startPartials2[state];
                g[y + state] = endPartials2[state];
                h[y + state] = endPartials2[state] * endPartials2[state];
            }
        }

        for (int stride = 1; stride < blockSize; stride *= 2) {
            int k = (opIdx + 1) * 2 * stride - 1;
            if (k < blockSize) {
                if (sSegmentFlags[k] == 0) {
                    sPartialsE[k][state] += sPartialsE[k - stride][state];
                    sPartialsF[k][state] += sPartialsF[k - stride][state];
                    sPartialsG[k][state] += sPartialsG[k - stride][state];
                    sPartialsH[k][state] += sPartialsH[k - stride][state];
                    sSegmentFlags[k] = sSegmentFlags[k - stride];
                }
            }
            KW_LOCAL_FENCE;
        }

        for (int stride = blockSize / 2; stride > 0; stride /= 2) {
            int k = (opIdx + 1) * 2 * stride - 1;
            if (k  + stride < blockSize) {
                if (sSegmentFlags[k + stride] == 0) {
                    sPartialsE[k + stride][state] += sPartialsE[k][state];
                    sPartialsF[k + stride][state] += sPartialsF[k][state];
                    sPartialsG[k + stride][state] += sPartialsG[k][state];
                    sPartialsH[k + stride][state] += sPartialsH[k][state];
                    sSegmentFlags[k + stride] = sSegmentFlags[k];
                }
            }
            KW_LOCAL_FENCE;
        }

        //if (sSegmentKeys[opIdx] != sSegmentKeys[opIdx + 1]) {
            //if (sSegmentFlags[opIdx] == 0) {
                //sPartialsE[opIdx][state] += sPartialsE[opIdx - 1][state];
                //sPartialsF[opIdx][state] += sPartialsF[opIdx - 1][state];
                //sPartialsG[opIdx][state] += sPartialsG[opIdx - 1][state];
                //sPartialsH[opIdx][state] += sPartialsH[opIdx - 1][state];
            //}
        //}

        if (opIdx == blockSize - 1 && (opNumber < opCount -1) || opNumber == opCount -1) {
            eRes[u + state] = sPartialsE[opIdx][state];
            fRes[u + state] = sPartialsF[opIdx][state];
            gRes[u + state] = sPartialsG[opIdx][state];
            hRes[u + state] = sPartialsH[opIdx][state];
        }

        if (sSegmentKeys[opIdx] != sSegmentKeys[opIdx + 1] && (opIdx < blockSize - 1) && (opNumber < opCount -1)) {
            e[y + state] += sPartialsE[opIdx][state];
            f[y + state] += sPartialsF[opIdx][state];
            g[y + state] += sPartialsG[opIdx][state];
            h[y + state] += sPartialsH[opIdx][state];
       }
}

KW_GLOBAL_KERNEL void kernelBastaReduceAcrossInterval(KW_GLOBAL_VAR REAL* KW_RESTRICT e,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT f,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT g,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT h,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT distance,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT dLogL,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT sizes,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT coalescent,
													int intervalStartsCount) {

#define SUM_INTERVAL_BLOCK_SIZE 32

        int intervalCount = intervalStartsCount - 1;
        int tid = KW_LOCAL_ID_0;
        int tidTotal = __umul24(KW_GROUP_ID_0, SUM_INTERVAL_BLOCK_SIZE * PADDED_STATE_COUNT) + tid;
        int state = tid % PADDED_STATE_COUNT;
        int intervalIdx = tid / PADDED_STATE_COUNT;
        int intervalNumber = __umul24(KW_GROUP_ID_0, SUM_INTERVAL_BLOCK_SIZE) + intervalIdx;
        int u = state + intervalNumber * PADDED_STATE_COUNT;

        KW_LOCAL_MEM REAL sPartials1[SUM_INTERVAL_BLOCK_SIZE * PADDED_STATE_COUNT];
        KW_LOCAL_MEM REAL sPartials2[SUM_INTERVAL_BLOCK_SIZE];

        if (intervalNumber < intervalCount && (sizes[state] > 0)) {
            sPartials1[tid] = (e[u] * e[u] - f[u] +
                                 g[u] * g[u] - h[u]) * distance[intervalNumber] / sizes[state];
        } else {
            sPartials1[tid] = 0;
        }
        KW_LOCAL_FENCE;

        if (tidTotal < intervalCount && (coalescent[u] != 0)) {
            sPartials2[tid] = logf(coalescent[u]);
        } else {
            sPartials2[tid] = 0;
        }
        KW_LOCAL_FENCE;


        for (int i = SUM_INTERVAL_BLOCK_SIZE * PADDED_STATE_COUNT / 2; i > 0; i >>= 1) {
            if (tid < i) {
                sPartials1[tid] += sPartials1[tid + i];
            }
            KW_LOCAL_FENCE;
        }

        for (int i = SUM_INTERVAL_BLOCK_SIZE * PADDED_STATE_COUNT / 2; i > 0; i >>= 1) {
            if (tid < i) {
                sPartials2[tid] += sPartials2[tid + i];
            }
            KW_LOCAL_FENCE;
        }

        if (tid == 0) {
            REAL temp = - sPartials1[0] / 4 + sPartials2[0];
            dLogL[KW_GROUP_ID_0] = temp;
        }
    }


#ifdef CUDA
} // extern "C"
#endif //CUDA
