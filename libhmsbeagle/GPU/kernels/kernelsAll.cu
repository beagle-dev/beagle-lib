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
		#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
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

#if (defined FW_OPENCL) && (defined DOUBLE_PRECISION)
    double atomicAdd(__global double* address, double val) {
    __global long* address_as_ull =
        (__global long*)address;
    long old = *address_as_ull;
    long assumed;

    do {
        assumed = old;
        old = atom_cmpxchg(address_as_ull, assumed,
            as_long(val + as_double(assumed)));
    } while (assumed != old);

    return as_double(old);
}

//void atomicAdd(volatile global float* addr, const float val) {
//    private float old, sum;
//    do {
//        old = *addr;
//        sum = old+val;
//    } while(atomic_cmpxchg((volatile global int*)addr, as_int(old), as_int(sum))!=as_int(old));
//}
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


KW_DEVICE_FUNC void bastaTiledDualMatVec(KW_GLOBAL_VAR REAL* KW_RESTRICT matrix1Ptr,
                                         KW_GLOBAL_VAR REAL* KW_RESTRICT matrix2Ptr,
                                         REAL (*sMatrix1)[PADDED_STATE_COUNT],
                                         REAL (*sMatrix2)[PADDED_STATE_COUNT],
                                         REAL* sVec1,
                                         REAL* sVec2,
                                         int state,
                                         int patIdx,
                                         int sameTransIndex,
                                         int isCoalescent,
                                         REAL* pSum1,
                                         REAL* pSum2){

    for (int i = 0; i < PADDED_STATE_COUNT; i += BLOCK_PEELING_SIZE_SCA) {
        if (patIdx < BLOCK_PEELING_SIZE_SCA) {
            sMatrix1[patIdx][state] = matrix1Ptr[patIdx * PADDED_STATE_COUNT + state];
            matrix1Ptr += BLOCK_PEELING_SIZE_SCA * PADDED_STATE_COUNT;
            if (!sameTransIndex) {
                sMatrix2[patIdx][state] = matrix2Ptr[patIdx * PADDED_STATE_COUNT + state];
                matrix2Ptr += BLOCK_PEELING_SIZE_SCA * PADDED_STATE_COUNT;
            }
        }
        KW_LOCAL_FENCE;
        REAL (*secondMatrix)[PADDED_STATE_COUNT] = (sameTransIndex == 1) ? sMatrix1 : sMatrix2;
        for (int j = 0; j < BLOCK_PEELING_SIZE_SCA; j++) {
            FMA(sMatrix1[j][state], sVec1[i + j], *pSum1);
            if (isCoalescent) {
                FMA(secondMatrix[j][state], sVec2[i + j], *pSum2);
            }
        }
        KW_LOCAL_FENCE;
    }
}



KW_DEVICE_FUNC REAL bastaParallelReduce(REAL* sReduction,
                                        REAL value,
                                        int state,
                                        int patIdx){
    sReduction[state] = value;
    (void) patIdx;
    KW_LOCAL_FENCE;
#ifdef IS_POWER_OF_TWO
    for (int stride = PADDED_STATE_COUNT / 2; stride > 0; stride >>= 1) {
        if (state < stride) {
#else
    for (int stride = SMALLEST_POWER_OF_TWO / 2; stride > 0; stride >>= 1) {
        if (state < stride && state + stride < PADDED_STATE_COUNT) {
#endif
            sReduction[state] += sReduction[state + stride];
        }
        KW_LOCAL_FENCE;
    }
    return sReduction[0];
}



/*
 * BASTA kernels
 */

KW_GLOBAL_KERNEL void kernelInnerBastaPartialsCoalescent(KW_GLOBAL_VAR REAL* KW_RESTRICT partials,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT matrices,
                                                    KW_GLOBAL_VAR int* KW_RESTRICT operations,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT sizes,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT coalescent,
													int start,
                                                    int numOps,
                                                    int totalPatterns) {

    int state = KW_LOCAL_ID_0;
    int patIdx = KW_LOCAL_ID_1;
    int pattern = __umul24(KW_GROUP_ID_0,BASTA_SUM_ACROSS_BLOCK_SIZE) + patIdx;
    int op = pattern + start;
    int maxOp = start + totalPatterns - 1;
    int sameTransIndex = 1;
    KW_LOCAL_MEM REAL sMatrix1[BLOCK_PEELING_SIZE_SCA][PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sMatrix2[BLOCK_PEELING_SIZE_SCA][PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sPartials1[BASTA_SUM_ACROSS_BLOCK_SIZE][PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sPartials2[PADDED_STATE_COUNT];

    int desIndex = operations[op * numOps];
    int child1PartialIndex = operations[op * numOps + 1];
    int child1TransIndex = operations[maxOp * numOps + 2];
    int child2PartialIndex = operations[op * numOps + 3];
    int child2TransIndex = operations[maxOp * numOps + 4];
    int accumulation1PartialIndex = operations[op * numOps + 5];
    int accumulation2PartialIndex = operations[op * numOps + 6];
    int intervalNumber = operations[op * numOps + 7];

    int isCoalescent = (child2PartialIndex >= 0);

    KW_GLOBAL_VAR REAL* KW_RESTRICT partials1 = partials + child1PartialIndex;
    KW_GLOBAL_VAR REAL* KW_RESTRICT partials3 = partials + desIndex;

    if (pattern < totalPatterns) {
        sPartials1[patIdx][state] = partials1[state];
    } else {
        sPartials1[patIdx][state] = 0;
    }
    REAL sum1 = 0;

    if (pattern < totalPatterns && isCoalescent) {
        KW_GLOBAL_VAR REAL* KW_RESTRICT partials2 = partials + child2PartialIndex;
        sPartials2[state] = partials2[state];
    }

    REAL sum2 = 0;

    KW_LOCAL_FENCE;

    bastaTiledDualMatVec(matrices + child1TransIndex, matrices + child2TransIndex,
                         sMatrix1, sMatrix2, sPartials1[patIdx], sPartials2,
                         state, patIdx, sameTransIndex, isCoalescent, &sum1, &sum2);


    if (pattern < totalPatterns) {
        partials3[state] = sum1;
    }

	if (pattern < totalPatterns && isCoalescent) {
	    KW_GLOBAL_VAR REAL* KW_RESTRICT accumulation1 = partials + accumulation1PartialIndex;
	    KW_GLOBAL_VAR REAL* KW_RESTRICT accumulation2 = partials + accumulation2PartialIndex;
		accumulation1[state] = sum1;
		accumulation2[state] = sum2;
		REAL popSize = sizes[state];
		if (popSize > 0) {
            partials3[state] = sum1 * sum2 / popSize;
        } else {
            partials3[state] = 0;
        }
		REAL denominator = bastaParallelReduce(sPartials2, partials3[state], state, patIdx);
		partials3[state] = partials3[state] / denominator;


		coalescent[intervalNumber] = denominator;
    }
}

KW_GLOBAL_KERNEL void kernelBastaPrecomputeDiagonals(
    KW_GLOBAL_VAR REAL* KW_RESTRICT eval,
    KW_GLOBAL_VAR REAL* KW_RESTRICT branchLengths,
    KW_GLOBAL_VAR REAL* KW_RESTRICT diagBuffer,
    int intervalCount) {
    
    int interval = KW_GROUP_ID_0;
    int state = KW_LOCAL_ID_0;
    
    if (interval < intervalCount && state < PADDED_STATE_COUNT) {
        REAL t = branchLengths[interval];
        REAL lambda = eval[state];
        diagBuffer[interval * PADDED_STATE_COUNT + state] = exp(lambda * t);
    }
}

KW_GLOBAL_KERNEL void kernelBastaPartialsWithPrecomputedDiag(
    KW_GLOBAL_VAR REAL* KW_RESTRICT partials,
    KW_GLOBAL_VAR REAL* KW_RESTRICT evec,
    KW_GLOBAL_VAR REAL* KW_RESTRICT ievc,
    KW_GLOBAL_VAR REAL* KW_RESTRICT diagBuffer,
    KW_GLOBAL_VAR int* KW_RESTRICT operations,
    KW_GLOBAL_VAR REAL* KW_RESTRICT sizes,
    KW_GLOBAL_VAR REAL* KW_RESTRICT coalescent,
    int start,
    int numOps,
    int totalPatterns,
    int intervalIdx) {

    int state = KW_LOCAL_ID_0;
    int patIdx = KW_LOCAL_ID_1;
    int pattern = __umul24(KW_GROUP_ID_0, BASTA_SUM_ACROSS_BLOCK_SIZE) + patIdx;
    int op = pattern + start;
    
    KW_LOCAL_MEM REAL sMatrix[BLOCK_PEELING_SIZE_SCA][PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sDiag[PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sPartials1[BASTA_SUM_ACROSS_BLOCK_SIZE][PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sPartials2[PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sTransformed1[BASTA_SUM_ACROSS_BLOCK_SIZE][PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sTransformed2[PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL popSizes[PADDED_STATE_COUNT];

    if (patIdx == 0) {
        popSizes[state] = sizes[state];
        sDiag[state] = diagBuffer[intervalIdx * PADDED_STATE_COUNT + state];
    }
    
    KW_LOCAL_FENCE;

    int desIndex = operations[op * numOps];
    int child1PartialIndex = operations[op * numOps + 1];
    int child2PartialIndex = operations[op * numOps + 3];
    int accumulation1PartialIndex = operations[op * numOps + 5];
    int accumulation2PartialIndex = operations[op * numOps + 6];
    int intervalNumber = operations[op * numOps + 7];
    
    int isCoalescent = (child2PartialIndex >= 0);
    
    KW_GLOBAL_VAR REAL* KW_RESTRICT partials1 = partials + child1PartialIndex;
    KW_GLOBAL_VAR REAL* KW_RESTRICT partials3 = partials + desIndex;

    if (pattern < totalPatterns) {
        sPartials1[patIdx][state] = partials1[state];
    } else {
        sPartials1[patIdx][state] = 0;
    }

    if (pattern < totalPatterns && isCoalescent) {
        KW_GLOBAL_VAR REAL* KW_RESTRICT partials2 = partials + child2PartialIndex;
        sPartials2[state] = partials2[state];
    }
    
    KW_LOCAL_FENCE;
    
    REAL temp1 = 0, temp2 = 0;
    
    KW_GLOBAL_VAR REAL* KW_RESTRICT evecRow = evec + state * PADDED_STATE_COUNT;
    for (int i = 0; i < PADDED_STATE_COUNT; i += BLOCK_PEELING_SIZE_SCA) {
        if (patIdx < BLOCK_PEELING_SIZE_SCA) {
            sMatrix[patIdx][state] = evecRow[i + patIdx];
        }
        KW_LOCAL_FENCE;
        
        for (int j = 0; j < BLOCK_PEELING_SIZE_SCA; j++) {
            FMA(sMatrix[j][state], sPartials1[patIdx][i + j], temp1);
            if (isCoalescent) {
                FMA(sMatrix[j][state], sPartials2[i + j], temp2);
            }
        }
        KW_LOCAL_FENCE;
    }

    sTransformed1[patIdx][state] = sDiag[state] * temp1;
    if (isCoalescent) {
        sTransformed2[state] = sDiag[state] * temp2;
    }
    
    KW_LOCAL_FENCE;
    
    REAL sum1 = 0, sum2 = 0;
    
    KW_GLOBAL_VAR REAL* KW_RESTRICT ievcRow = ievc + state * PADDED_STATE_COUNT;
    for (int i = 0; i < PADDED_STATE_COUNT; i += BLOCK_PEELING_SIZE_SCA) {
        if (patIdx < BLOCK_PEELING_SIZE_SCA) {
            sMatrix[patIdx][state] = ievcRow[i + patIdx];
        }
        KW_LOCAL_FENCE;
        
        for (int j = 0; j < BLOCK_PEELING_SIZE_SCA; j++) {
            FMA(sMatrix[j][state], sTransformed1[patIdx][i + j], sum1);
            if (isCoalescent) {
                FMA(sMatrix[j][state], sTransformed2[i + j], sum2);
            }
        }
        KW_LOCAL_FENCE;
    }

    if (pattern < totalPatterns) {
        partials3[state] = sum1;
    }

    if (pattern < totalPatterns && isCoalescent) {
        KW_GLOBAL_VAR REAL* KW_RESTRICT accumulation1 = partials + accumulation1PartialIndex;
        KW_GLOBAL_VAR REAL* KW_RESTRICT accumulation2 = partials + accumulation2PartialIndex;
        
        accumulation1[state] = sum1;
        accumulation2[state] = sum2;
        if (popSizes[state] > 0) {
            partials3[state] = sum1 * sum2 / popSizes[state];
        } else {
            partials3[state] = 0;
        }
        sPartials2[state] = partials3[state];
        
        KW_LOCAL_FENCE;
        
#ifdef IS_POWER_OF_TWO
        for (int i = PADDED_STATE_COUNT / 2; i > 0; i >>= 1) {
            if (state < i) {
#else
        for (int i = SMALLEST_POWER_OF_TWO / 2; i > 0; i >>= 1) {
            if (state < i && state + i < PADDED_STATE_COUNT) {
#endif
                sPartials2[state] += sPartials2[state + i];
            }
            KW_LOCAL_FENCE;
        }
        
        REAL denominator = sPartials2[0];
        partials3[state] = partials3[state] / denominator;
        coalescent[intervalNumber] = denominator;
    }
}


KW_GLOBAL_KERNEL void kernelBastaReduceWithinIntervalMerged(KW_GLOBAL_VAR int* KW_RESTRICT operations,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT partials,
                                                    KW_GLOBAL_VAR REAL* dBastaMemory,
                                                    int numOps,
                                                    int start,
                                                    int end,
                                                    int numBlocks,
                                                    int kCoalescentBufferLength) {

#define OPS_PER_THREAD 8

    // Thread and block indices
    int state = KW_LOCAL_ID_0;
    int threadId = KW_LOCAL_ID_1;
    int blockY = KW_GROUP_ID_0;

    int halfBlocks = numBlocks / 2;
    int doEF = (blockY < halfBlocks) ? 1 : 0;

    if (!doEF) {blockY = blockY - halfBlocks;}
    int threadGlobalY = blockY * BASTA_SUM_INTERVAL_BLOCK_SIZE + threadId;
    int opStart = start + threadGlobalY * OPS_PER_THREAD;
    int opEnd = opStart + OPS_PER_THREAD;
    int opBlockStart = OPS_PER_THREAD * blockY * BASTA_SUM_INTERVAL_BLOCK_SIZE;
	int opBlockEnd = opBlockStart + OPS_PER_THREAD * BASTA_SUM_INTERVAL_BLOCK_SIZE;
    if (opEnd > end) opEnd = end;
	if (opBlockEnd > end) opBlockEnd = end;


    KW_GLOBAL_VAR REAL* e = dBastaMemory;
    KW_GLOBAL_VAR REAL* f = e + PADDED_STATE_COUNT * kCoalescentBufferLength;
    KW_GLOBAL_VAR REAL* g = f + PADDED_STATE_COUNT * kCoalescentBufferLength;
    KW_GLOBAL_VAR REAL* h = g + PADDED_STATE_COUNT * kCoalescentBufferLength;
// 	KW_LOCAL_MEM int shared_child1PartialIndex[BASTA_SUM_INTERVAL_BLOCK_SIZE * OPS_PER_THREAD];
// 	KW_LOCAL_MEM int shared_child2PartialIndex[BASTA_SUM_INTERVAL_BLOCK_SIZE * OPS_PER_THREAD];
// 	KW_LOCAL_MEM int shared_accumulation1PartialIndex[BASTA_SUM_INTERVAL_BLOCK_SIZE * OPS_PER_THREAD];
// 	KW_LOCAL_MEM int shared_accumulation2PartialIndex[BASTA_SUM_INTERVAL_BLOCK_SIZE * OPS_PER_THREAD];
// 	KW_LOCAL_MEM int shared_segmentKey[BASTA_SUM_INTERVAL_BLOCK_SIZE * OPS_PER_THREAD];
    int currentSegmentKey = -1;
    int carryOutSegmentKey = -1;
    REAL partialA = 0;
    REAL partialB = 0;


    int next_op = opStart;
    int nextSegmentKey = -1;
    REAL nextA_val1 = 0, nextB_val1 = 0;
    REAL nextA_val2 = 0, nextB_val2 = 0;



     if (state < PADDED_STATE_COUNT && next_op < opEnd) {
        int op = next_op;
		int child1PartialIndex = operations[op * numOps + 1];
		int child2PartialIndex = operations[op * numOps + 3];
		int accumulation1PartialIndex = operations[op * numOps + 5];
		int accumulation2PartialIndex = operations[op * numOps + 6];
		int segmentKey = operations[op * numOps + 7];


        KW_GLOBAL_VAR REAL* part1A = (doEF)? (partials + child1PartialIndex):(partials + accumulation1PartialIndex);

        KW_GLOBAL_VAR REAL* part2A = (doEF)? (partials + child2PartialIndex):(partials + accumulation2PartialIndex);

        REAL val1A = part1A[state];
        REAL val2A = 0;

        if (child2PartialIndex >= 0) {
            val2A = part2A[state];
        }

        nextA_val1 = val1A;
        nextB_val1 = val1A * val1A;
        nextA_val2 = val2A;
        nextB_val2 = val2A * val2A;
        nextSegmentKey = segmentKey;
    }

    for (int idx = opStart; idx < opEnd; ++idx) {
        REAL currA_val1 = nextA_val1;
        REAL currB_val1 = nextB_val1;
        REAL currA_val2 = nextA_val2;
        REAL currB_val2 = nextB_val2;

        int segmentKey = nextSegmentKey;

        next_op = idx + 1;
        if (state < PADDED_STATE_COUNT && next_op < opEnd) {
            int op = next_op;
			int child1PartialIndex = operations[op * numOps + 1];
			int child2PartialIndex = operations[op * numOps + 3];
			int accumulation1PartialIndex = operations[op * numOps + 5];
			int accumulation2PartialIndex = operations[op * numOps + 6];
			int segmentKeyNext = operations[op * numOps + 7];

        	KW_GLOBAL_VAR REAL* part1A = (doEF)? (partials + child1PartialIndex):(partials + accumulation1PartialIndex);

        	KW_GLOBAL_VAR REAL* part2A = (doEF)? (partials + child2PartialIndex):(partials + accumulation2PartialIndex);

        	REAL val1A = part1A[state];
        	REAL val2A = 0;

        	if (child2PartialIndex >= 0) {
            	val2A = part2A[state];
        	}

        	nextA_val1 = val1A;
        	nextB_val1 = val1A * val1A;
        	nextA_val2 = val2A;
        	nextB_val2 = val2A * val2A;

            nextSegmentKey = segmentKeyNext;
        } else {
            nextA_val1 = nextB_val1 = 0;
            nextA_val2 = nextB_val2 = 0;
            nextSegmentKey = -1;
        }

        int isNewSegment = (segmentKey != currentSegmentKey) ? 1 : 0;

        if (isNewSegment == 1 && idx != opStart) {
        	int w = currentSegmentKey * PADDED_STATE_COUNT + state;

                if (doEF) {
                    // partialA => e, partialB => f
                    atomicAdd(&e[w], partialA);
                    atomicAdd(&f[w], partialB);
                } else {
                    // partialA => g, partialB => h
                    atomicAdd(&g[w], partialA);
                    atomicAdd(&h[w], partialB);
                }

            partialA = 0;
            partialB = 0;
            }

        partialA += (currA_val1 + currA_val2);
        partialB += (currB_val1 + currB_val2);

        currentSegmentKey = segmentKey;
    }

	KW_LOCAL_FENCE;

	carryOutSegmentKey = currentSegmentKey;
    REAL carryOutA = partialA;
    REAL carryOutB = partialB;


    KW_LOCAL_MEM REAL sCarryOutA[BASTA_SUM_INTERVAL_BLOCK_SIZE][PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sCarryOutB[BASTA_SUM_INTERVAL_BLOCK_SIZE][PADDED_STATE_COUNT];

	KW_LOCAL_MEM REAL sSegmentFlags[BASTA_SUM_INTERVAL_BLOCK_SIZE];
	KW_LOCAL_MEM REAL sCarryOutSegmentKeys[BASTA_SUM_INTERVAL_BLOCK_SIZE + 1];


    if (state < PADDED_STATE_COUNT) {
        sCarryOutA[threadId][state] = carryOutA;
        sCarryOutB[threadId][state] = carryOutB;
        sCarryOutSegmentKeys[threadId] = carryOutSegmentKey;
    }

	KW_LOCAL_FENCE;
	if (state == 0 && opStart < end) {
		if (threadId == 0) {
	    	sSegmentFlags[threadId] = 1;
		} else {
	    	int prevSegmentKey = sCarryOutSegmentKeys[threadId - 1];
	    	int currSegmentKey = sCarryOutSegmentKeys[threadId];

	    	if (currSegmentKey != prevSegmentKey) {
	        	sSegmentFlags[threadId] = 1;
	    	} else {
	        	sSegmentFlags[threadId] = 0;
	    	}
		}
	}

	KW_LOCAL_FENCE;

    int n = BASTA_SUM_INTERVAL_BLOCK_SIZE;
    for (int stride = 1; stride < n; stride *= 2) {
        int index = (threadId + 1) * 2 * stride - 1;
        if (index < n) {
            if (sSegmentFlags[index] == 0) {
                sCarryOutA[index][state] += sCarryOutA[index - stride][state];
                sCarryOutB[index][state] += sCarryOutB[index - stride][state];
                if (state == 0) {
                    sSegmentFlags[index] = sSegmentFlags[index - stride];
                }
            }
        }
        KW_LOCAL_FENCE;
    }

    for (int stride = n / 2; stride >= 1; stride /= 2) {
        int index = (threadId + 1) * 2 * stride - 1;
        if (index + stride < n) {
            if (sSegmentFlags[index + stride] == 0) {
                sCarryOutA[index + stride][state] += sCarryOutA[index][state];
                sCarryOutB[index + stride][state] += sCarryOutB[index][state];
                if (state == 0) {
                    sSegmentFlags[index + stride] = sSegmentFlags[index];
                }
            }
        }
        KW_LOCAL_FENCE;
    }


    if (threadId == BASTA_SUM_INTERVAL_BLOCK_SIZE - 1 || sCarryOutSegmentKeys[threadId] != sCarryOutSegmentKeys[threadId + 1]) {
        int reducedKey = sCarryOutSegmentKeys[threadId];
        if (reducedKey >= 0) {
    		int u = reducedKey * PADDED_STATE_COUNT + state;

            if (doEF) {
    			atomicAdd(&e[u], sCarryOutA[threadId][state]);
    			atomicAdd(&f[u], sCarryOutB[threadId][state]);
            } else {
    			atomicAdd(&g[u], sCarryOutA[threadId][state]);
    			atomicAdd(&h[u], sCarryOutB[threadId][state]);
            }
		}
    }
}



KW_GLOBAL_KERNEL void kernelBastaReduceWithinIntervalMergedSlab(
                                                    KW_GLOBAL_VAR int* KW_RESTRICT reduceOps,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT partials,
                                                    KW_GLOBAL_VAR REAL* dBastaMemory,
                                                    int start,
                                                    int end,
                                                    int numBlocks,
                                                    int kCoalescentBufferLength) {

    const int BASTA_REDUCE_OPS_PER_THREAD = 8;
    const int BASTA_REDUCE_STRIDE         = 5;

    int state = KW_LOCAL_ID_0;
    int threadId = KW_LOCAL_ID_1;
    int blockY = KW_GROUP_ID_0;

    int halfBlocks = numBlocks / 2;
    int doEF = (blockY < halfBlocks) ? 1 : 0;

    if (!doEF) { blockY = blockY - halfBlocks; }
    int threadGlobalY = blockY * BASTA_SUM_INTERVAL_BLOCK_SIZE + threadId;
    int opStart = start + threadGlobalY * BASTA_REDUCE_OPS_PER_THREAD;
    int opEnd = opStart + BASTA_REDUCE_OPS_PER_THREAD;
    int opBlockStart = BASTA_REDUCE_OPS_PER_THREAD * blockY * BASTA_SUM_INTERVAL_BLOCK_SIZE;
    int opBlockEnd = opBlockStart + BASTA_REDUCE_OPS_PER_THREAD * BASTA_SUM_INTERVAL_BLOCK_SIZE;
    if (opEnd > end) opEnd = end;
    if (opBlockEnd > end) opBlockEnd = end;

    KW_GLOBAL_VAR REAL* e = dBastaMemory;
    KW_GLOBAL_VAR REAL* f = e + PADDED_STATE_COUNT * kCoalescentBufferLength;
    KW_GLOBAL_VAR REAL* g = f + PADDED_STATE_COUNT * kCoalescentBufferLength;
    KW_GLOBAL_VAR REAL* h = g + PADDED_STATE_COUNT * kCoalescentBufferLength;

    int currentSegmentKey = -1;
    int carryOutSegmentKey = -1;
    REAL partialA = 0;
    REAL partialB = 0;

    int next_op = opStart;
    int nextSegmentKey = -1;
    REAL nextA_val1 = 0, nextB_val1 = 0;
    REAL nextA_val2 = 0, nextB_val2 = 0;

    if (state < PADDED_STATE_COUNT && next_op < opEnd) {
        int op = next_op;
        int child1PartialIndex = reduceOps[op * BASTA_REDUCE_STRIDE + 0];
        int child2PartialIndex = reduceOps[op * BASTA_REDUCE_STRIDE + 1];
        int accumulation1PartialIndex = reduceOps[op * BASTA_REDUCE_STRIDE + 2];
        int accumulation2PartialIndex = reduceOps[op * BASTA_REDUCE_STRIDE + 3];
        int segmentKey = reduceOps[op * BASTA_REDUCE_STRIDE + 4];

        KW_GLOBAL_VAR REAL* part1A = (doEF) ? (partials + child1PartialIndex) : (partials + accumulation1PartialIndex);
        KW_GLOBAL_VAR REAL* part2A = (doEF) ? (partials + child2PartialIndex) : (partials + accumulation2PartialIndex);

        REAL val1A = part1A[state];
        REAL val2A = 0;
        if (child2PartialIndex >= 0) {
            val2A = part2A[state];
        }

        nextA_val1 = val1A;
        nextB_val1 = val1A * val1A;
        nextA_val2 = val2A;
        nextB_val2 = val2A * val2A;
        nextSegmentKey = segmentKey;
    }

    for (int idx = opStart; idx < opEnd; ++idx) {
        REAL currA_val1 = nextA_val1;
        REAL currB_val1 = nextB_val1;
        REAL currA_val2 = nextA_val2;
        REAL currB_val2 = nextB_val2;
        int segmentKey = nextSegmentKey;

        next_op = idx + 1;
        if (state < PADDED_STATE_COUNT && next_op < opEnd) {
            int op = next_op;
            int child1PartialIndex = reduceOps[op * BASTA_REDUCE_STRIDE + 0];
            int child2PartialIndex = reduceOps[op * BASTA_REDUCE_STRIDE + 1];
            int accumulation1PartialIndex = reduceOps[op * BASTA_REDUCE_STRIDE + 2];
            int accumulation2PartialIndex = reduceOps[op * BASTA_REDUCE_STRIDE + 3];
            int segmentKeyNext = reduceOps[op * BASTA_REDUCE_STRIDE + 4];

            KW_GLOBAL_VAR REAL* part1A = (doEF) ? (partials + child1PartialIndex) : (partials + accumulation1PartialIndex);
            KW_GLOBAL_VAR REAL* part2A = (doEF) ? (partials + child2PartialIndex) : (partials + accumulation2PartialIndex);

            REAL val1A = part1A[state];
            REAL val2A = 0;
            if (child2PartialIndex >= 0) {
                val2A = part2A[state];
            }

            nextA_val1 = val1A;
            nextB_val1 = val1A * val1A;
            nextA_val2 = val2A;
            nextB_val2 = val2A * val2A;
            nextSegmentKey = segmentKeyNext;
        } else {
            nextA_val1 = nextB_val1 = 0;
            nextA_val2 = nextB_val2 = 0;
            nextSegmentKey = -1;
        }

        int isNewSegment = (segmentKey != currentSegmentKey) ? 1 : 0;

        if (isNewSegment == 1 && idx != opStart) {
            int w = currentSegmentKey * PADDED_STATE_COUNT + state;

            if (doEF) {
                atomicAdd(&e[w], partialA);
                atomicAdd(&f[w], partialB);
            } else {
                atomicAdd(&g[w], partialA);
                atomicAdd(&h[w], partialB);
            }

            partialA = 0;
            partialB = 0;
        }

        partialA += (currA_val1 + currA_val2);
        partialB += (currB_val1 + currB_val2);

        currentSegmentKey = segmentKey;
    }

    KW_LOCAL_FENCE;

    carryOutSegmentKey = currentSegmentKey;
    REAL carryOutA = partialA;
    REAL carryOutB = partialB;

    KW_LOCAL_MEM REAL sCarryOutA[BASTA_SUM_INTERVAL_BLOCK_SIZE][PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sCarryOutB[BASTA_SUM_INTERVAL_BLOCK_SIZE][PADDED_STATE_COUNT];

    KW_LOCAL_MEM REAL sSegmentFlags[BASTA_SUM_INTERVAL_BLOCK_SIZE];
    KW_LOCAL_MEM REAL sCarryOutSegmentKeys[BASTA_SUM_INTERVAL_BLOCK_SIZE + 1];

    if (state < PADDED_STATE_COUNT) {
        sCarryOutA[threadId][state] = carryOutA;
        sCarryOutB[threadId][state] = carryOutB;
        sCarryOutSegmentKeys[threadId] = carryOutSegmentKey;
    }

    KW_LOCAL_FENCE;
    if (state == 0 && opStart < end) {
        if (threadId == 0) {
            sSegmentFlags[threadId] = 1;
        } else {
            int prevSegmentKey = sCarryOutSegmentKeys[threadId - 1];
            int currSegmentKey = sCarryOutSegmentKeys[threadId];

            if (currSegmentKey != prevSegmentKey) {
                sSegmentFlags[threadId] = 1;
            } else {
                sSegmentFlags[threadId] = 0;
            }
        }
    }

    KW_LOCAL_FENCE;

    int n = BASTA_SUM_INTERVAL_BLOCK_SIZE;
    for (int stride = 1; stride < n; stride *= 2) {
        int index = (threadId + 1) * 2 * stride - 1;
        if (index < n) {
            if (sSegmentFlags[index] == 0) {
                sCarryOutA[index][state] += sCarryOutA[index - stride][state];
                sCarryOutB[index][state] += sCarryOutB[index - stride][state];
                if (state == 0) {
                    sSegmentFlags[index] = sSegmentFlags[index - stride];
                }
            }
        }
        KW_LOCAL_FENCE;
    }

    for (int stride = n / 2; stride >= 1; stride /= 2) {
        int index = (threadId + 1) * 2 * stride - 1;
        if (index + stride < n) {
            if (sSegmentFlags[index + stride] == 0) {
                sCarryOutA[index + stride][state] += sCarryOutA[index][state];
                sCarryOutB[index + stride][state] += sCarryOutB[index][state];
                if (state == 0) {
                    sSegmentFlags[index + stride] = sSegmentFlags[index];
                }
            }
        }
        KW_LOCAL_FENCE;
    }


    if (threadId == BASTA_SUM_INTERVAL_BLOCK_SIZE - 1 || sCarryOutSegmentKeys[threadId] != sCarryOutSegmentKeys[threadId + 1]) {
        int reducedKey = sCarryOutSegmentKeys[threadId];
        if (reducedKey >= 0) {
            int u = reducedKey * PADDED_STATE_COUNT + state;

            if (doEF) {
                atomicAdd(&e[u], sCarryOutA[threadId][state]);
                atomicAdd(&f[u], sCarryOutB[threadId][state]);
            } else {
                atomicAdd(&g[u], sCarryOutA[threadId][state]);
                atomicAdd(&h[u], sCarryOutB[threadId][state]);
            }
        }
    }
}



KW_GLOBAL_KERNEL void kernelBastaReduceWithinIntervalGrad(KW_GLOBAL_VAR int*  KW_RESTRICT operations,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT partials,
                                                    KW_GLOBAL_VAR REAL* dBastaGradBuffers,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT partialsGrad,
                                                    KW_GLOBAL_VAR int*  KW_RESTRICT gradNodeOps,
                                                    int numOps,
                                                    int start,
                                                    int end,
                                                    int numBlocks,
                                                    int kCoalescentBufferLength,
                                                    int gradAbStride) {


    int state = KW_LOCAL_ID_0;
    int threadId = KW_LOCAL_ID_1;
    int blockY = KW_GROUP_ID_0;
    int ab = KW_GROUP_ID_1;

    int halfBlocks = numBlocks / 2;
    int doEF = (blockY < halfBlocks) ? 1 : 0;

    if (!doEF) { blockY = blockY - halfBlocks; }
    int threadGlobalY = blockY * BASTA_SUM_INTERVAL_BLOCK_SIZE + threadId;
    int opStart  = start + threadGlobalY * OPS_PER_THREAD;
    int opEnd    = opStart + OPS_PER_THREAD;
    if (opEnd > end) opEnd = end;

    int S2 = PADDED_STATE_COUNT * PADDED_STATE_COUNT;
    int gradSlice = S2 * kCoalescentBufferLength * PADDED_STATE_COUNT;
    KW_GLOBAL_VAR REAL* e = dBastaGradBuffers;
    KW_GLOBAL_VAR REAL* f = e + gradSlice;
    KW_GLOBAL_VAR REAL* g = f + gradSlice;
    KW_GLOBAL_VAR REAL* h = g + gradSlice;

    int abBase = ab * gradAbStride;

    int currentSegmentKey = -1;
    REAL partialA = 0;
    REAL partialB = 0;

    int next_op = opStart;
    int nextSegmentKey = -1;
    REAL nextA_val1 = 0, nextB_val1 = 0;
    REAL nextA_val2 = 0, nextB_val2 = 0;

    if (state < PADDED_STATE_COUNT && next_op < opEnd) {
        int op = next_op;
        int child1PartialIndex = operations[op * numOps + 1];
        int child2PartialIndex = operations[op * numOps + 3];
        int accumulation1PartialIndex = operations[op * numOps + 5];
        int accumulation2PartialIndex = operations[op * numOps + 6];
        int segmentKey = operations[op * numOps + 7];

        KW_GLOBAL_VAR REAL* part1A = doEF ? (partials + child1PartialIndex)
                                          : (partials + accumulation1PartialIndex);
        KW_GLOBAL_VAR REAL* part2A = doEF ? (partials + child2PartialIndex)
                                          : (partials + accumulation2PartialIndex);

        REAL val1A = part1A[state];
        REAL val2A = (child2PartialIndex >= 0) ? part2A[state] : 0;

        int nb = op * 5;
        int node1 = doEF ? gradNodeOps[nb + 1] : gradNodeOps[nb + 3];
        int node2 = doEF ? gradNodeOps[nb + 2] : gradNodeOps[nb + 4];
        REAL grad1 = partialsGrad[abBase + node1 * PADDED_STATE_COUNT + state];
        REAL grad2 = (child2PartialIndex >= 0)
                     ? partialsGrad[abBase + node2 * PADDED_STATE_COUNT + state] : 0;
        nextA_val1 = grad1;
        nextB_val1 = 2 * val1A * grad1;
        nextA_val2 = grad2;
        nextB_val2 = 2 * val2A * grad2;
        nextSegmentKey = ab * kCoalescentBufferLength + segmentKey;
    }

    for (int idx = opStart; idx < opEnd; ++idx) {
        REAL currA_val1 = nextA_val1;
        REAL currB_val1 = nextB_val1;
        REAL currA_val2 = nextA_val2;
        REAL currB_val2 = nextB_val2;
        int  segmentKey = nextSegmentKey;

        next_op = idx + 1;
        if (state < PADDED_STATE_COUNT && next_op < opEnd) {
            int op = next_op;
            int child1PartialIndex = operations[op * numOps + 1];
            int child2PartialIndex = operations[op * numOps + 3];
            int accumulation1PartialIndex = operations[op * numOps + 5];
            int accumulation2PartialIndex = operations[op * numOps + 6];
            int segmentKeyNext = operations[op * numOps + 7];

            KW_GLOBAL_VAR REAL* part1A = doEF ? (partials + child1PartialIndex)
                                              : (partials + accumulation1PartialIndex);
            KW_GLOBAL_VAR REAL* part2A = doEF ? (partials + child2PartialIndex)
                                              : (partials + accumulation2PartialIndex);

            REAL val1A = part1A[state];
            REAL val2A = (child2PartialIndex >= 0) ? part2A[state] : 0;

            int nb = op * 5;
            int node1 = doEF ? gradNodeOps[nb + 1] : gradNodeOps[nb + 3];
            int node2 = doEF ? gradNodeOps[nb + 2] : gradNodeOps[nb + 4];
            REAL grad1 = partialsGrad[abBase + node1 * PADDED_STATE_COUNT + state];
            REAL grad2 = (child2PartialIndex >= 0)
                         ? partialsGrad[abBase + node2 * PADDED_STATE_COUNT + state] : 0;
            nextA_val1 = grad1;
            nextB_val1 = 2 * val1A * grad1;
            nextA_val2 = grad2;
            nextB_val2 = 2 * val2A * grad2;
            nextSegmentKey = ab * kCoalescentBufferLength + segmentKeyNext;
        } else {
            nextA_val1 = nextB_val1 = 0;
            nextA_val2 = nextB_val2 = 0;
            nextSegmentKey = -1;
        }

        if (segmentKey != currentSegmentKey && idx != opStart) {
            int w = currentSegmentKey * PADDED_STATE_COUNT + state;
            if (doEF) {
                atomicAdd(&e[w], partialA);
                atomicAdd(&f[w], partialB);
            } else {
                atomicAdd(&g[w], partialA);
                atomicAdd(&h[w], partialB);
            }
            partialA = 0;
            partialB = 0;
        }

        partialA += (currA_val1 + currA_val2);
        partialB += (currB_val1 + currB_val2);
        currentSegmentKey = segmentKey;
    }

    KW_LOCAL_FENCE;

    REAL carryOutA = partialA;
    REAL carryOutB = partialB;
    int  carryOutSegmentKey = currentSegmentKey;

    KW_LOCAL_MEM REAL sCarryOutA[BASTA_SUM_INTERVAL_BLOCK_SIZE][PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sCarryOutB[BASTA_SUM_INTERVAL_BLOCK_SIZE][PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sSegmentFlags[BASTA_SUM_INTERVAL_BLOCK_SIZE];
    KW_LOCAL_MEM REAL sCarryOutSegmentKeys[BASTA_SUM_INTERVAL_BLOCK_SIZE + 1];

    if (state < PADDED_STATE_COUNT) {
        sCarryOutA[threadId][state] = carryOutA;
        sCarryOutB[threadId][state] = carryOutB;
        sCarryOutSegmentKeys[threadId] = carryOutSegmentKey;
    }

    KW_LOCAL_FENCE;
    if (state == 0 && opStart < end) {
        if (threadId == 0) {
            sSegmentFlags[threadId] = 1;
        } else {
            int prevKey = sCarryOutSegmentKeys[threadId - 1];
            int currKey = sCarryOutSegmentKeys[threadId];
            sSegmentFlags[threadId] = (currKey != prevKey) ? 1 : 0;
        }
    }

    KW_LOCAL_FENCE;

    int n = BASTA_SUM_INTERVAL_BLOCK_SIZE;
    for (int stride = 1; stride < n; stride *= 2) {
        int index = (threadId + 1) * 2 * stride - 1;
        if (index < n) {
            if (sSegmentFlags[index] == 0) {
                sCarryOutA[index][state] += sCarryOutA[index - stride][state];
                sCarryOutB[index][state] += sCarryOutB[index - stride][state];
                if (state == 0) {
                    sSegmentFlags[index] = sSegmentFlags[index - stride];
                }
            }
        }
        KW_LOCAL_FENCE;
    }

    for (int stride = n / 2; stride >= 1; stride /= 2) {
        int index = (threadId + 1) * 2 * stride - 1;
        if (index + stride < n) {
            if (sSegmentFlags[index + stride] == 0) {
                sCarryOutA[index + stride][state] += sCarryOutA[index][state];
                sCarryOutB[index + stride][state] += sCarryOutB[index][state];
                if (state == 0) {
                    sSegmentFlags[index + stride] = sSegmentFlags[index];
                }
            }
        }
        KW_LOCAL_FENCE;
    }

    if (threadId == BASTA_SUM_INTERVAL_BLOCK_SIZE - 1 ||
        sCarryOutSegmentKeys[threadId] != sCarryOutSegmentKeys[threadId + 1]) {
        int reducedKey = sCarryOutSegmentKeys[threadId];
        if (reducedKey >= 0) {
            int u = reducedKey * PADDED_STATE_COUNT + state;
            if (doEF) {
                atomicAdd(&e[u], sCarryOutA[threadId][state]);
                atomicAdd(&f[u], sCarryOutB[threadId][state]);
            } else {
                atomicAdd(&g[u], sCarryOutA[threadId][state]);
                atomicAdd(&h[u], sCarryOutB[threadId][state]);
            }
        }
    }
}


KW_DEVICE_FUNC REAL bastaReduceAcrossTwoPasses(REAL* sPartials1,
                                            REAL pass1Val,
                                            REAL pass2Val,
                                            int tid){
    sPartials1[tid] = pass1Val;
    KW_LOCAL_FENCE;

    for (int i = BASTA_SUM_ACROSS_BLOCK_SIZE * PADDED_STATE_COUNT / 2; i > 0; i >>= 1) {
        if (tid < i) {
            sPartials1[tid] += sPartials1[tid + i];
        }
        KW_LOCAL_FENCE;
    }

    REAL temp = -sPartials1[0] / 4;

    sPartials1[tid] = pass2Val;
    KW_LOCAL_FENCE;

    for (int i = BASTA_SUM_ACROSS_BLOCK_SIZE * PADDED_STATE_COUNT / 2; i > 0; i >>= 1) {
        if (tid < i) {
            sPartials1[tid] += sPartials1[tid + i];
        }
        KW_LOCAL_FENCE;
    }

    return temp + sPartials1[0];
}


KW_GLOBAL_KERNEL void kernelBastaReduceAcrossInterval(KW_GLOBAL_VAR REAL* KW_RESTRICT dBastaMemory,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT distance,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT dLogL,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT sizes,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT coalescent,
													int intervalStartsCount,
													int kCoalescentBufferLength) {


        int intervalCount = intervalStartsCount - 1;
        int tid = KW_LOCAL_ID_0;
        int tidTotal = __umul24(KW_GROUP_ID_0, BASTA_SUM_ACROSS_BLOCK_SIZE * PADDED_STATE_COUNT) + tid;
        int state = tid % PADDED_STATE_COUNT;
        int intervalIdx = tid / PADDED_STATE_COUNT;
        int intervalNumber = __umul24(KW_GROUP_ID_0, BASTA_SUM_ACROSS_BLOCK_SIZE) + intervalIdx;
        int u = state + intervalNumber * PADDED_STATE_COUNT;

	    KW_GLOBAL_VAR REAL* e = dBastaMemory;
	    KW_GLOBAL_VAR REAL* f = e + PADDED_STATE_COUNT * kCoalescentBufferLength;
	    KW_GLOBAL_VAR REAL* g = f + PADDED_STATE_COUNT * kCoalescentBufferLength;
	    KW_GLOBAL_VAR REAL* h = g + PADDED_STATE_COUNT * kCoalescentBufferLength;

        KW_LOCAL_MEM REAL sPartials1[BASTA_SUM_ACROSS_BLOCK_SIZE * PADDED_STATE_COUNT];

        REAL pass1Val = 0;
        if (intervalNumber < intervalCount && sizes[state] > 0) {
            pass1Val = (e[u] * e[u] - f[u] + g[u] * g[u] - h[u]) * distance[intervalNumber] / sizes[state];
        }

        REAL pass2Val = 0;
        if (tidTotal < intervalCount && coalescent[tidTotal] > 0) {
            pass2Val = log(coalescent[tidTotal]);
        }

        REAL result = bastaReduceAcrossTwoPasses(sPartials1, pass1Val, pass2Val, tid);

        if (tid == 0) {
            dLogL[KW_GROUP_ID_0] = result;
        }
    }


KW_GLOBAL_KERNEL void kernelBastaReduceAcrossIntervalGrad(KW_GLOBAL_VAR REAL* KW_RESTRICT dBastaMemory,
                                                        KW_GLOBAL_VAR REAL* KW_RESTRICT dBastaGradBuffers,
                                                        KW_GLOBAL_VAR REAL* KW_RESTRICT distance,
                                                        KW_GLOBAL_VAR REAL* KW_RESTRICT sizes,
                                                        KW_GLOBAL_VAR REAL* KW_RESTRICT coalescent,
                                                        KW_GLOBAL_VAR REAL* KW_RESTRICT coalescentGrad,
                                                        KW_GLOBAL_VAR REAL* KW_RESTRICT dGradOut,
                                                        int intervalCount,
                                                        int stateCount,
                                                        int kCoalescentBufferLength){
    int tid = KW_LOCAL_ID_0;
    int tidTotal = __umul24(KW_GROUP_ID_0, BASTA_SUM_ACROSS_BLOCK_SIZE * PADDED_STATE_COUNT) + tid;
    int state = tid % PADDED_STATE_COUNT;
    int intervalIdx = tid / PADDED_STATE_COUNT;
    int intervalNumber = __umul24(KW_GROUP_ID_0, BASTA_SUM_ACROSS_BLOCK_SIZE) + intervalIdx;
    int ab = KW_GROUP_ID_1;
    int u = state + intervalNumber * PADDED_STATE_COUNT;

    int S2 = stateCount * stateCount;
    int L  = kCoalescentBufferLength;

    KW_GLOBAL_VAR REAL* e = dBastaMemory;
    KW_GLOBAL_VAR REAL* g = e + 2 * PADDED_STATE_COUNT * L;

    int sliceSize = S2 * L * PADDED_STATE_COUNT;
    int w = (ab * L + intervalNumber) * PADDED_STATE_COUNT + state;
    KW_GLOBAL_VAR REAL* eGr = dBastaGradBuffers;
    KW_GLOBAL_VAR REAL* fGr = eGr + sliceSize;
    KW_GLOBAL_VAR REAL* gGr = fGr + sliceSize;
    KW_GLOBAL_VAR REAL* hGr = gGr + sliceSize;

    KW_LOCAL_MEM REAL sPartials1[BASTA_SUM_ACROSS_BLOCK_SIZE * PADDED_STATE_COUNT];

    REAL pass1Val = 0;
    if (intervalNumber < intervalCount && sizes[state] > 0) {
        pass1Val = (2 * e[u] * eGr[w] - fGr[w] +
                    2 * g[u] * gGr[w] - hGr[w]) * distance[intervalNumber] / sizes[state];
    }

    REAL pass2Val = 0;
    if (tidTotal < intervalCount && coalescent[tidTotal] > 0) {
        pass2Val = coalescentGrad[ab * L + tidTotal] / coalescent[tidTotal];
    }

    REAL result = bastaReduceAcrossTwoPasses(sPartials1, pass1Val, pass2Val, tid);

    if (tid == 0) {
        dGradOut[ab * KW_NUM_GROUPS_0 + KW_GROUP_ID_0] = result;
    }
}

//Need to remove
KW_GLOBAL_KERNEL void kernelBastaPartialsGradMigrationAB(KW_GLOBAL_VAR REAL* KW_RESTRICT partials,
                                                        KW_GLOBAL_VAR REAL* KW_RESTRICT partialsGrad,
                                                        KW_GLOBAL_VAR REAL* KW_RESTRICT matrices,
                                                        KW_GLOBAL_VAR int*  KW_RESTRICT operations,
                                                        KW_GLOBAL_VAR int*  KW_RESTRICT gradNodeOps,
                                                        KW_GLOBAL_VAR REAL* KW_RESTRICT sizes,
                                                        KW_GLOBAL_VAR REAL* KW_RESTRICT coalescent,
                                                        KW_GLOBAL_VAR REAL* KW_RESTRICT coalescentGrad,
                                                        KW_GLOBAL_VAR REAL* KW_RESTRICT edgeLengthsGrad,
                                                        int   start,
                                                        int   numOps,
                                                        int   totalPatterns,
                                                        int   stateCount,
                                                        int   gradAbStride,
                                                        int   coalLength,
                                                        int   matrixIndex){

        int state = KW_LOCAL_ID_0;
        int patIdx = KW_LOCAL_ID_1;
        int pattern = __umul24(KW_GROUP_ID_0, BASTA_SUM_ACROSS_BLOCK_SIZE) + patIdx;
        int op = pattern + start;
        int ab = KW_GROUP_ID_1;

        int a = ab / stateCount;
        int b = ab % stateCount;

        REAL distance = edgeLengthsGrad[matrixIndex];

        int sameTransIndex = 1;
        KW_LOCAL_MEM REAL sMatrix1[BLOCK_PEELING_SIZE_SCA][PADDED_STATE_COUNT];
        KW_LOCAL_MEM REAL sMatrix2[BLOCK_PEELING_SIZE_SCA][PADDED_STATE_COUNT];
        KW_LOCAL_MEM REAL sChildGrad1[BASTA_SUM_ACROSS_BLOCK_SIZE][PADDED_STATE_COUNT];
        KW_LOCAL_MEM REAL sChildGrad2[PADDED_STATE_COUNT];
        KW_LOCAL_MEM REAL sReduction[PADDED_STATE_COUNT];

        int opsBase = op * numOps;
        int desPartialOff = operations[opsBase + 0];
        int child2PartialOff = operations[opsBase + 3];
        int acc1PartialOff = operations[opsBase + 5];
        int acc2PartialOff = operations[opsBase + 6];
        int intervalNumber = operations[opsBase + 7];

        int nodeBase = op * 5;
        int destNode = gradNodeOps[nodeBase + 0];
        int child1Node = gradNodeOps[nodeBase + 1];
        int child2Node = gradNodeOps[nodeBase + 2];
        int acc1Node = gradNodeOps[nodeBase + 3];
        int acc2Node = gradNodeOps[nodeBase + 4];

        int isCoalescent = (pattern < totalPatterns) && (child2PartialOff >= 0);
        int abBase = ab * gradAbStride;

        if (pattern < totalPatterns) {
            sChildGrad1[patIdx][state] = partialsGrad[abBase + child1Node * PADDED_STATE_COUNT + state];
        } else {
            sChildGrad1[patIdx][state] = 0;
        }

        if (isCoalescent) {
            sChildGrad2[state] = partialsGrad[abBase + child2Node * PADDED_STATE_COUNT + state];
        }

        REAL sum1 = 0;
        REAL sum2 = 0;
        KW_LOCAL_FENCE;


        int commonMatOff = matrixIndex * PADDED_STATE_COUNT * PADDED_STATE_COUNT;
        bastaTiledDualMatVec(matrices + commonMatOff, matrices + commonMatOff,
                             sMatrix1, sMatrix2, sChildGrad1[patIdx], sChildGrad2,
                             state, patIdx, sameTransIndex, isCoalescent, &sum1, &sum2);


        if (pattern < totalPatterns && state == b) {
            REAL leftAccA = (partials + acc1PartialOff)[a];
            sum1 += leftAccA * distance;
        }
        if (isCoalescent && state == b) {
            sum2 += (partials + acc2PartialOff)[a] * distance;
        }

        if (pattern < totalPatterns && !isCoalescent) {
            partialsGrad[abBase + destNode * PADDED_STATE_COUNT + state] = sum1;
        }


        REAL entry = 0;
        if (isCoalescent) {
            REAL popSize = sizes[state];
            if (state < stateCount && popSize > 0) {
                entry = (sum1 * (partials + acc2PartialOff)[state]
                       + sum2 * (partials + acc1PartialOff)[state]) / popSize;
            }

            REAL partial_J_ab = bastaParallelReduce(sReduction, entry, state, patIdx);

            REAL J = coalescent[intervalNumber];
            REAL destPartialI = (partials + desPartialOff)[state];
            REAL destGradFinal = 0;
            if (state < stateCount && J != 0) {
                destGradFinal = entry / J - partial_J_ab * destPartialI / J;
            }

            partialsGrad[abBase + destNode * PADDED_STATE_COUNT + state] = destGradFinal;
            partialsGrad[abBase + acc1Node * PADDED_STATE_COUNT + state] = sum1;
            partialsGrad[abBase + acc2Node * PADDED_STATE_COUNT + state] = sum2;

            if (state == 0) {
                coalescentGrad[ab * coalLength + intervalNumber] = partial_J_ab;
            }
    }
}


#if PADDED_STATE_COUNT < 32
    #define BASTA_MATVEC_K_TILE PADDED_STATE_COUNT
#else
    #define BASTA_MATVEC_K_TILE 32
#endif


KW_DEVICE_FUNC REAL bastaTransposeMatVec(KW_GLOBAL_VAR REAL* KW_RESTRICT matrixPtr,
    								REAL (*sMatrix)[BASTA_MATVEC_K_TILE + 1],
    								REAL* sVec,
    								int state, int patIdx, int stateCount) {

    REAL transSum = 0;

    const int linearTid = patIdx * PADDED_STATE_COUNT + state;
    const int blockThreads = PADDED_STATE_COUNT * BASTA_SUM_ACROSS_BLOCK_SIZE;
    const int tileElems = PADDED_STATE_COUNT * BASTA_MATVEC_K_TILE;

    for (int kTile = 0; kTile < PADDED_STATE_COUNT;
                        kTile += BASTA_MATVEC_K_TILE) {

        for (int e = linearTid; e < tileElems; e += blockThreads) {
            int row = e / BASTA_MATVEC_K_TILE;
            int col = e - row * BASTA_MATVEC_K_TILE;
            int srcCol = kTile + col;
            REAL val = (row < stateCount && srcCol < stateCount)? matrixPtr[row * PADDED_STATE_COUNT + srcCol] : (REAL) 0.0;
            sMatrix[row][col] = val;
        }

        KW_LOCAL_FENCE;

        if (state < stateCount) {
            int kMax = BASTA_MATVEC_K_TILE;
            if (kTile + kMax > stateCount) kMax = stateCount - kTile;
            for (int k = 0; k < kMax; k++) {
                FMA(sMatrix[state][k], sVec[kTile + k], transSum);
            }
        }
        KW_LOCAL_FENCE;
    }

    return transSum;
}


KW_DEVICE_FUNC void bastaTransposeDualMatVecSameMat(
    KW_GLOBAL_VAR REAL* KW_RESTRICT mPtr,
    REAL (*sMatrix)[BASTA_MATVEC_K_TILE + 1],
    REAL* sVec1,
    REAL* sVec2,
    int state, int patIdx, int stateCount,
    REAL* outTrans1,
    REAL* outTrans2) {

    REAL t1 = (REAL) 0.0;
    REAL t2 = (REAL) 0.0;

    const int linearTid = patIdx * PADDED_STATE_COUNT + state;
    const int blockThreads = PADDED_STATE_COUNT * BASTA_SUM_ACROSS_BLOCK_SIZE;
    const int tileElems = PADDED_STATE_COUNT * BASTA_MATVEC_K_TILE;

    for (int kTile = 0; kTile < PADDED_STATE_COUNT;
                        kTile += BASTA_MATVEC_K_TILE) {
        for (int e = linearTid; e < tileElems; e += blockThreads) {
            int row = e / BASTA_MATVEC_K_TILE;
            int col = e - row * BASTA_MATVEC_K_TILE;
            int srcCol = kTile + col;
            REAL val = (row < stateCount && srcCol < stateCount)
					   ? mPtr[row * PADDED_STATE_COUNT + srcCol]
                       : (REAL) 0.0;
            sMatrix[row][col] = val;
        }
        KW_LOCAL_FENCE;

        if (state < stateCount) {
            int kMax = BASTA_MATVEC_K_TILE;
            if (kTile + kMax > stateCount) kMax = stateCount - kTile;
            if (patIdx == 0) {
                for (int k = 0; k < kMax; k++) {
                    FMA(sMatrix[state][k], sVec1[kTile + k], t1);
                }
            } else if (patIdx == 1) {
                for (int k = 0; k < kMax; k++) {
                    FMA(sMatrix[state][k], sVec2[kTile + k], t2);
                }
            }
        }
        KW_LOCAL_FENCE;
    }

    if (patIdx == 0) *outTrans1 = t1;
    if (patIdx == 1) *outTrans2 = t2;
}


KW_GLOBAL_KERNEL void kernelComputeHazardAdjoints(
    KW_GLOBAL_VAR REAL* KW_RESTRICT dBastaMemory,
    KW_GLOBAL_VAR REAL* KW_RESTRICT intervalLengths,
    KW_GLOBAL_VAR int*  KW_RESTRICT intervalNumbers,
    KW_GLOBAL_VAR REAL* KW_RESTRICT sizes,
    KW_GLOBAL_VAR REAL* KW_RESTRICT dHazardAdjoints,
    KW_GLOBAL_VAR REAL* KW_RESTRICT dPopSizeGrad,
    int numIntervals,
    int kCoalescentBufferLength) {

    int state = KW_LOCAL_ID_0;
    int interval = KW_GROUP_ID_0 * KW_LOCAL_SIZE_1 + KW_LOCAL_ID_1;
    if (interval >= numIntervals || state >= PADDED_STATE_COUNT) return;

    int intervalNumber = intervalNumbers[interval];
    REAL len = intervalLengths[interval];
    REAL sz = sizes[state];
    if (sz == 0) return;
    REAL invSz = 1.0 / sz;

    int u = intervalNumber * PADDED_STATE_COUNT + state;
    int stride = PADDED_STATE_COUNT * kCoalescentBufferLength;
    REAL eVal = dBastaMemory[u];
    REAL fVal = dBastaMemory[stride + u];
    REAL gVal = dBastaMemory[2 * stride + u];
    REAL hVal = dBastaMemory[3 * stride + u];

    int base = interval * 4 * PADDED_STATE_COUNT + state;
    int adjStride = PADDED_STATE_COUNT;
    dHazardAdjoints[base] = -len * eVal * invSz / 2.0;
    dHazardAdjoints[base + adjStride] =  len * invSz / 4.0;
    dHazardAdjoints[base + 2 * adjStride] = -len * gVal * invSz / 2.0;
    dHazardAdjoints[base + 3 * adjStride] =  len * invSz / 4.0;

    REAL popContrib = len * (eVal*eVal - fVal + gVal*gVal - hVal) / (4.0 * sz * sz);
    atomicAdd(&dPopSizeGrad[state], popContrib);
}



KW_GLOBAL_KERNEL void kernelAdjointBastaPartials(KW_GLOBAL_VAR REAL* KW_RESTRICT partials,
    										KW_GLOBAL_VAR REAL* KW_RESTRICT partialAdj,
    										KW_GLOBAL_VAR REAL* KW_RESTRICT matrices,
    										KW_GLOBAL_VAR REAL* KW_RESTRICT scratchYBar,
    										KW_GLOBAL_VAR REAL* KW_RESTRICT scratchX,
    										KW_GLOBAL_VAR REAL* KW_RESTRICT coalRightYBar,
    										KW_GLOBAL_VAR REAL* KW_RESTRICT coalRightX,
    										KW_GLOBAL_VAR int*  KW_RESTRICT operations,
    										KW_GLOBAL_VAR REAL* KW_RESTRICT sizes,
   											KW_GLOBAL_VAR REAL* KW_RESTRICT coalescent,
    										KW_GLOBAL_VAR REAL* KW_RESTRICT dHazardAdjoints,
    										KW_GLOBAL_VAR REAL* KW_RESTRICT dPopSizeGrad,
    										int interval,
    										int start,
    										int totalPatterns,
    										int matTransIndex,
    										int stateCount,
    										int coalOp) {

    int state = KW_LOCAL_ID_0;
    int patIdx = KW_LOCAL_ID_1;
    int validState = (state < stateCount) ? 1 : 0;
    int pattern = __umul24(KW_GROUP_ID_0, BASTA_SUM_ACROSS_BLOCK_SIZE) + patIdx;
    int op = pattern + start;
    int numOps = BASTA_OP_COUNT;

    KW_LOCAL_MEM REAL sMatrix[PADDED_STATE_COUNT][BASTA_MATVEC_K_TILE + 1];
    KW_LOCAL_MEM REAL sYBar[BASTA_SUM_ACROSS_BLOCK_SIZE][PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sReduction[PADDED_STATE_COUNT];

    const int hBase   = interval * 4 * PADDED_STATE_COUNT + state;
    const int hStride = PADDED_STATE_COUNT;
    const REAL adjE = dHazardAdjoints[hBase];
    const REAL adjF = dHazardAdjoints[hBase + hStride];
    const REAL adjG = dHazardAdjoints[hBase + 2 * hStride];
    const REAL adjH = dHazardAdjoints[hBase + 3 * hStride];

    int accBuf1Off = -1, accBuf2Off = -1, inBuf1Off = -1, inBuf2Off = -1;
    REAL pAcc1 = 0.0, pAcc2 = 0.0, pIn1 = 0.0, pIn2 = 0.0;
    int childMissing2 = 0;
    if (pattern < totalPatterns) {
        inBuf2Off = operations[op * numOps + 3];
        childMissing2 = (inBuf2Off < 0);

        if (validState) {
            accBuf1Off = operations[op * numOps + 5];
            accBuf2Off = operations[op * numOps + 6];
            inBuf1Off  = operations[op * numOps + 1];

            pAcc1 = partials[accBuf1Off + state];
            if (accBuf2Off >= 0) pAcc2 = partials[accBuf2Off + state];
            pIn1  = partials[inBuf1Off + state];
            if (inBuf2Off >= 0)  pIn2 = partials[inBuf2Off + state];

            partialAdj[accBuf1Off + state] += adjG + 2.0 * pAcc1 * adjH;
            if (accBuf2Off >= 0)
                partialAdj[accBuf2Off + state] += adjG + 2.0 * pAcc2 * adjH;
            partialAdj[inBuf1Off + state] += adjE + 2.0 * pIn1 * adjF;
            if (inBuf2Off >= 0)
                partialAdj[inBuf2Off + state] += adjE + 2.0 * pIn2 * adjF;
        }
    }

    KW_LOCAL_FENCE;


    int coalBlockStart = start + __umul24(KW_GROUP_ID_0, BASTA_SUM_ACROSS_BLOCK_SIZE);
    int isCoalBlock = (coalOp >= 0 && coalOp >= coalBlockStart &&
                       coalOp < coalBlockStart + BASTA_SUM_ACROSS_BLOCK_SIZE) ? 1 : 0;

    if (isCoalBlock) {
        int destOff = operations[coalOp * numOps];
        int inBuf1 = operations[coalOp * numOps + 1];
        int inMat1 = operations[coalOp * numOps + 2];
        int inBuf2 = operations[coalOp * numOps + 3];
        int inMat2 = operations[coalOp * numOps + 4];
        int accBuf1 = operations[coalOp * numOps + 5];
        int accBuf2 = operations[coalOp * numOps + 6];
        int intNum = operations[coalOp * numOps + 7];

        REAL leftEnd  = validState ? partials[accBuf1 + state] : 0.0;
        REAL rightEnd = validState ? partials[accBuf2 + state] : 0.0;
        REAL sz = validState ? sizes[state] : 0.0;
        REAL J = coalescent[intNum];

        REAL zAdj = validState ? partialAdj[destOff + state] : 0.0;
        REAL w_i = (sz > 0) ? leftEnd * rightEnd / sz : 0.0;
        REAL dotZW = bastaParallelReduce(sReduction, zAdj * w_i, state, patIdx);

        REAL adjJ = 1.0 / J - dotZW / (J * J);
        REAL adjW = validState ? (zAdj / J + adjJ) : 0.0;
        REAL leftEndBar  = validState ? (partialAdj[accBuf1 + state] + ((sz > 0) ? adjW * rightEnd / sz : 0.0)) : 0.0;
        REAL rightEndBar = validState ? (partialAdj[accBuf2 + state] + ((sz > 0) ? adjW * leftEnd / sz : 0.0)) : 0.0;

        REAL leftStart  = validState ? partials[inBuf1 + state] : 0.0;
        REAL rightStart = validState ? partials[inBuf2 + state] : 0.0;

        if (patIdx == 0) {
            sYBar[0][state] = leftEndBar;
            sYBar[1][state] = rightEndBar;
        }
        KW_LOCAL_FENCE;

        REAL trans1 = (REAL) 0.0;
        REAL trans2 = (REAL) 0.0;
        bastaTransposeDualMatVecSameMat(matrices + inMat1, sMatrix,
                                        sYBar[0], sYBar[1],
                                        state, patIdx, stateCount,
                                        &trans1, &trans2);

        if (patIdx == 0 && validState) {
            partialAdj[inBuf1 + state] += trans1;
            if (sz > 0)
                atomicAdd(&dPopSizeGrad[state], -adjW * leftEnd * rightEnd / (sz * sz));
        }
        if (patIdx == 1 && validState) {
            partialAdj[inBuf2 + state] += trans2;
        }

        if (patIdx == 0) {
            scratchYBar[coalOp * PADDED_STATE_COUNT + state] = leftEndBar;
            scratchX[coalOp * PADDED_STATE_COUNT + state] = leftStart;
            coalRightYBar[interval * PADDED_STATE_COUNT + state] = rightEndBar;
            coalRightX[interval * PADDED_STATE_COUNT + state] = rightStart;
        }
        KW_LOCAL_FENCE;
    }

    int myInBuf1Off = 0;
    REAL myYBar = 0.0;
    REAL myX = 0.0;

    if (childMissing2 && validState) {
        myInBuf1Off = inBuf1Off;
        myYBar = partialAdj[accBuf1Off + state];
        myX = pIn1;
    }

    sYBar[patIdx][state] = myYBar;
    KW_LOCAL_FENCE;

    REAL transResult = bastaTransposeMatVec(matrices + matTransIndex, sMatrix,
                                            sYBar[patIdx], state, patIdx, stateCount);

    if (childMissing2 && validState) {
        partialAdj[myInBuf1Off + state] += transResult;
    }

    if (childMissing2) {
        scratchYBar[op * PADDED_STATE_COUNT + state] = myYBar;
        scratchX[op * PADDED_STATE_COUNT + state] = myX;
    }
}



#define EIGEN_TILE 16

KW_GLOBAL_KERNEL void kernelTiledBatchGemmABt(
    KW_GLOBAL_VAR REAL* KW_RESTRICT A,
    KW_GLOBAL_VAR REAL* KW_RESTRICT B,
    KW_GLOBAL_VAR REAL* KW_RESTRICT C,
    int matrixCount) {

    int tilesPerRow = (PADDED_STATE_COUNT + EIGEN_TILE - 1) / EIGEN_TILE;
    int m = KW_GROUP_ID_0 / tilesPerRow;
    int tileR = KW_GROUP_ID_0 % tilesPerRow;
    int tileC = KW_GROUP_ID_1;
    if (m >= matrixCount) return;

    int tx = KW_LOCAL_ID_0;
    int ty = KW_LOCAL_ID_1;
    int row = tileR * EIGEN_TILE + tx;
    int col = tileC * EIGEN_TILE + ty;

    KW_LOCAL_MEM REAL sA[EIGEN_TILE][EIGEN_TILE + 1];
    KW_LOCAL_MEM REAL sB[EIGEN_TILE][EIGEN_TILE + 1];

    int S = PADDED_STATE_COUNT;
    int S2 = S * S;
    REAL acc = 0.0;

    for (int k = 0; k < S; k += EIGEN_TILE) {
        // Load A[m][row, k+ty] into shared memory
        sA[tx][ty] = (row < S && (k + ty) < S) ? A[m * S2 + row * S + k + ty] : 0.0;
        // Load B[col, k+tx] for B^T access: B^T[k+tx, col] = B[col, k+tx]
        sB[tx][ty] = (col < S && (k + tx) < S) ? B[col * S + k + tx] : 0.0;
        KW_LOCAL_FENCE;
        #pragma unroll
        for (int j = 0; j < EIGEN_TILE; j++)
            acc += sA[tx][j] * sB[j][ty];
        KW_LOCAL_FENCE;
    }

    if (row < S && col < S)
        C[m * S2 + row * S + col] = acc;
}


KW_GLOBAL_KERNEL void kernelTiledBatchGemmAtB(
    KW_GLOBAL_VAR REAL* KW_RESTRICT A,
    KW_GLOBAL_VAR REAL* KW_RESTRICT B,
    KW_GLOBAL_VAR REAL* KW_RESTRICT C,
    int matrixCount) {

    int tilesPerRow = (PADDED_STATE_COUNT + EIGEN_TILE - 1) / EIGEN_TILE;
    int m = KW_GROUP_ID_0 / tilesPerRow;
    int tileR = KW_GROUP_ID_0 % tilesPerRow;
    int tileC = KW_GROUP_ID_1;
    if (m >= matrixCount) return;

    int tx = KW_LOCAL_ID_0;
    int ty = KW_LOCAL_ID_1;
    int row = tileR * EIGEN_TILE + tx;
    int col = tileC * EIGEN_TILE + ty;

    KW_LOCAL_MEM REAL sA[EIGEN_TILE][EIGEN_TILE + 1];
    KW_LOCAL_MEM REAL sB[EIGEN_TILE][EIGEN_TILE + 1];

    int S = PADDED_STATE_COUNT;
    int S2 = S * S;
    REAL acc = 0.0;

    for (int k = 0; k < S; k += EIGEN_TILE) {
        // Load A^T[row, k+ty] = A[k+ty, row]  (transpose read from row-major A)
        sA[tx][ty] = (row < S && (k + ty) < S) ? A[(k + ty) * S + row] : 0.0;
        // Load B[m][k+tx, col]  (normal row-major read)
        sB[tx][ty] = (col < S && (k + tx) < S) ? B[m * S2 + (k + tx) * S + col] : 0.0;
        KW_LOCAL_FENCE;
        #pragma unroll
        for (int j = 0; j < EIGEN_TILE; j++)
            acc += sA[tx][j] * sB[j][ty];
        KW_LOCAL_FENCE;
    }

    if (row < S && col < S)
        C[m * S2 + row * S + col] = acc;
}


KW_GLOBAL_KERNEL void kernelReduceMatrices(
    KW_GLOBAL_VAR REAL* KW_RESTRICT src,
    KW_GLOBAL_VAR REAL* KW_RESTRICT dst,
    int matrixCount) {

    int idx = KW_GROUP_ID_0 * KW_LOCAL_SIZE_0 + KW_LOCAL_ID_0;
    int S2 = PADDED_STATE_COUNT * PADDED_STATE_COUNT;
    if (idx >= S2) return;

    REAL sum = 0.0;
    for (int m = 0; m < matrixCount; m++)
        sum += src[m * S2 + idx];
    dst[idx] = sum;
}


KW_GLOBAL_KERNEL void kernelApplyComplexLoewnerInPlace(
    KW_GLOBAL_VAR REAL* KW_RESTRICT transformed,
    KW_GLOBAL_VAR REAL* KW_RESTRICT eigenValues,
    KW_GLOBAL_VAR REAL* KW_RESTRICT branchLengths,
    KW_GLOBAL_VAR int*  KW_RESTRICT blockStarts,
    KW_GLOBAL_VAR int*  KW_RESTRICT blockDims,
    int matrixCount,
    int stateCount,
    int numBlocks) {

    int m = KW_GROUP_ID_0;

    int flatIdx = KW_GROUP_ID_1 * KW_LOCAL_SIZE_0 + KW_LOCAL_ID_0;
    int leftBlock = flatIdx / numBlocks;
    int rightBlock = flatIdx % numBlocks;
    if (m >= matrixCount || leftBlock >= numBlocks) return;

    REAL t = branchLengths[m];
    int ls = blockStarts[leftBlock], ld = blockDims[leftBlock];
    int rs = blockStarts[rightBlock], rd = blockDims[rightBlock];

    if (ld == 0 || rd == 0) return;
    int S2 = PADDED_STATE_COUNT * PADDED_STATE_COUNT;
    KW_GLOBAL_VAR REAL* mat = transformed + m * S2;


    if (ld == 1 && rd == 1) {
        REAL la = eigenValues[ls];
        REAL lb = eigenValues[rs];
        REAL tla = t * la, tlb = t * lb;
        REAL coeff;
        if (fabs(tla - tlb) < 1e-12) {
            coeff = t * exp(tla);
        } else {
            coeff = (exp(tla) - exp(tlb)) / (la - lb);
        }
        mat[ls * PADDED_STATE_COUNT + rs] *= coeff;


    } else if (ld == 1 && rd == 2) {
        REAL la = eigenValues[ls];
        REAL rb_real = eigenValues[rs];
        REAL rb_imag = eigenValues[stateCount + rs];
        REAL shift_real = rb_real - la;
        REAL denom = shift_real * shift_real + rb_imag * rb_imag;
        REAL scale = exp(t * la);
        REAL ic0, ic1;
        if (denom < 1e-12) {
            ic0 = t; ic1 = 0.0;
        } else {
            REAL ex = exp(t * shift_real);
            REAL cs = cos(t * rb_imag), sn = sin(t * rb_imag);
            ic0 = (ex * (shift_real * cs + rb_imag * sn) - shift_real) / denom;
            ic1 = (ex * (shift_real * sn - rb_imag * cs) + rb_imag) / denom;
        }
        REAL c0 = scale * ic0, c1 = scale * ic1;
        REAL in0 = mat[ls * PADDED_STATE_COUNT + rs];
        REAL in1 = mat[ls * PADDED_STATE_COUNT + rs + 1];

        mat[ls * PADDED_STATE_COUNT + rs] = c0 * in0 + c1 * in1;
        mat[ls * PADDED_STATE_COUNT + rs + 1] = -c1 * in0 + c0 * in1;


    } else if (ld == 2 && rd == 1) {
        REAL la_real = eigenValues[ls];
        REAL la_imag = eigenValues[stateCount + ls];
        REAL rb = eigenValues[rs];
        REAL shift_real = rb - la_real;
        REAL denom = shift_real * shift_real + la_imag * la_imag;
        REAL ic0, ic1;
        if (denom < 1e-12) {
            ic0 = t; ic1 = 0.0;
        } else {
            REAL ex = exp(t * shift_real);
            REAL cs = cos(t * la_imag), sn = sin(t * la_imag);
            ic0 = (ex * (shift_real * cs + la_imag * sn) - shift_real) / denom;
            ic1 = (ex * (shift_real * sn - la_imag * cs) + la_imag) / denom;
        }

        REAL expR = exp(t * la_real);
        REAL cosI = cos(t * la_imag), sinI = sin(t * la_imag);
        REAL l00 = expR * cosI, l01 = -expR * sinI;
        REAL l10 = expR * sinI, l11 = expR * cosI;

        REAL p0 = l00 * ic0 - l01 * ic1;
        REAL p1 = l00 * ic1 + l01 * ic0;
        REAL p2 = l10 * ic0 - l11 * ic1;
        REAL p3 = l10 * ic1 + l11 * ic0;
        REAL in0 = mat[ls * PADDED_STATE_COUNT + rs];
        REAL in1 = mat[(ls+1) * PADDED_STATE_COUNT + rs];
        mat[ls * PADDED_STATE_COUNT + rs] = p0 * in0 + p1 * in1;
        mat[(ls+1) * PADDED_STATE_COUNT + rs] = p2 * in0 + p3 * in1;


    } else {
        REAL la_real = eigenValues[ls];
        REAL la_imag = eigenValues[stateCount + ls];
        REAL rb_real = eigenValues[rs];
        REAL rb_imag = eigenValues[stateCount + rs];


        REAL sr1 = rb_real - la_real, si1 = la_imag + rb_imag;
        REAL sr2 = rb_real - la_real, si2 = rb_imag - la_imag;

        REAL exp1r, exp1i, exp2r, exp2i;
        REAL int1r, int1i, int2r, int2i;


        REAL e1 = exp(t * la_real);
        exp1r = e1 * cos(t * (-la_imag)); exp1i = e1 * sin(t * (-la_imag));
        exp2r = e1 * cos(t * la_imag);    exp2i = e1 * sin(t * la_imag);


        REAL d1 = sr1 * sr1 + si1 * si1;
        if (d1 < 1e-12) {
            int1r = t; int1i = 0.0;
        } else {
            REAL ex1 = exp(t * sr1);
            REAL cs1 = cos(t * si1), sn1 = sin(t * si1);
            int1r = (sr1 * (ex1 * cs1 - 1.0) + si1 * ex1 * sn1) / d1;
            int1i = (sr1 * ex1 * sn1 - si1 * (ex1 * cs1 - 1.0)) / d1;
        }


        REAL d2 = sr2 * sr2 + si2 * si2;
        if (d2 < 1e-12) {
            int2r = t; int2i = 0.0;
        } else {
            REAL ex2 = exp(t * sr2);
            REAL cs2 = cos(t * si2), sn2 = sin(t * si2);
            int2r = (sr2 * (ex2 * cs2 - 1.0) + si2 * ex2 * sn2) / d2;
            int2i = (sr2 * ex2 * sn2 - si2 * (ex2 * cs2 - 1.0)) / d2;
        }


        REAL pr = exp1r * int1r - exp1i * int1i;
        REAL pi = exp1r * int1i + exp1i * int1r;

        REAL mr = exp2r * int2r - exp2i * int2i;
        REAL mi = exp2r * int2i + exp2i * int2r;


        REAL p00 = pr, p01 = pi, p10 = -pi, p11 = pr;
        REAL m00 = mr, m01 = mi, m10 = -mi, m11 = mr;


        REAL c[16];
        REAL basis[4][4] = {
            {0.5, 0.0, 0.5, 0.0},
            {0.0, 0.5, 0.0, 0.5},
            {0.0, -0.5, 0.0, 0.5},
            {0.5, 0.0, -0.5, 0.0}
        };
        for (int col = 0; col < 4; col++) {
            REAL u = basis[col][0], v = basis[col][1];
            REAL p = basis[col][2], q = basis[col][3];

            REAL ou = m00*u + m01*v, ov = m10*u + m11*v;
            REAL op = p00*p + p01*q, oq = p10*p + p11*q;
            c[col] = ou + op;
            c[4+col] = ov + oq;
            c[8+col] = -ov + oq;
            c[12+col] = ou - op;
        }

        REAL in00 = mat[ls * PADDED_STATE_COUNT + rs];
        REAL in01 = mat[ls * PADDED_STATE_COUNT + rs + 1];
        REAL in10 = mat[(ls+1) * PADDED_STATE_COUNT + rs];
        REAL in11 = mat[(ls+1) * PADDED_STATE_COUNT + rs + 1];

        mat[ls * PADDED_STATE_COUNT + rs] = c[0]*in00 + c[1]*in01 + c[2]*in10 + c[3]*in11;
        mat[ls * PADDED_STATE_COUNT + rs + 1] = c[4]*in00 + c[5]*in01 + c[6]*in10 + c[7]*in11;
        mat[(ls+1) * PADDED_STATE_COUNT + rs] = c[8]*in00 + c[9]*in01 + c[10]*in10 + c[11]*in11;
        mat[(ls+1) * PADDED_STATE_COUNT + rs + 1] = c[12]*in00 + c[13]*in01 + c[14]*in10 + c[15]*in11;
    }
}


KW_GLOBAL_KERNEL void kernelTiledSingleGemmAtB(
    KW_GLOBAL_VAR REAL* KW_RESTRICT A,
    KW_GLOBAL_VAR REAL* KW_RESTRICT B,
    KW_GLOBAL_VAR REAL* KW_RESTRICT C) {

    int tileR = KW_GROUP_ID_0;
    int tileC = KW_GROUP_ID_1;

    int tx = KW_LOCAL_ID_0;
    int ty = KW_LOCAL_ID_1;
    int row = tileR * EIGEN_TILE + tx;
    int col = tileC * EIGEN_TILE + ty;

    KW_LOCAL_MEM REAL sA[EIGEN_TILE][EIGEN_TILE + 1];
    KW_LOCAL_MEM REAL sB[EIGEN_TILE][EIGEN_TILE + 1];

    int S = PADDED_STATE_COUNT;
    REAL acc = 0.0;

    for (int k = 0; k < S; k += EIGEN_TILE) {
        sA[tx][ty] = (row < S && (k + ty) < S) ? A[(k + ty) * S + row] : 0.0;
        sB[tx][ty] = (col < S && (k + tx) < S) ? B[(k + tx) * S + col] : 0.0;
        KW_LOCAL_FENCE;
        #pragma unroll
        for (int j = 0; j < EIGEN_TILE; j++)
            acc += sA[tx][j] * sB[j][ty];
        KW_LOCAL_FENCE;
    }

    if (row < S && col < S)
        C[row * S + col] = acc;
}


KW_GLOBAL_KERNEL void kernelTiledSingleGemmABt(
    KW_GLOBAL_VAR REAL* KW_RESTRICT A,
    KW_GLOBAL_VAR REAL* KW_RESTRICT B,
    KW_GLOBAL_VAR REAL* KW_RESTRICT C) {

    int tileR = KW_GROUP_ID_0;
    int tileC = KW_GROUP_ID_1;

    int tx = KW_LOCAL_ID_0;
    int ty = KW_LOCAL_ID_1;
    int row = tileR * EIGEN_TILE + tx;
    int col = tileC * EIGEN_TILE + ty;

    KW_LOCAL_MEM REAL sA[EIGEN_TILE][EIGEN_TILE + 1];
    KW_LOCAL_MEM REAL sB[EIGEN_TILE][EIGEN_TILE + 1];

    int S = PADDED_STATE_COUNT;
    REAL acc = 0.0;

    for (int k = 0; k < S; k += EIGEN_TILE) {
        sA[tx][ty] = (row < S && (k + ty) < S) ? A[row * S + k + ty] : 0.0;
        sB[tx][ty] = (col < S && (k + tx) < S) ? B[col * S + k + tx] : 0.0;
        KW_LOCAL_FENCE;
        #pragma unroll
        for (int j = 0; j < EIGEN_TILE; j++)
            acc += sA[tx][j] * sB[j][ty];
        KW_LOCAL_FENCE;
    }

    if (row < S && col < S)
        C[row * S + col] = acc;
}


KW_GLOBAL_KERNEL void kernelAccumulateMatrixAdjoints(
    KW_GLOBAL_VAR REAL* KW_RESTRICT scratchYBar,
    KW_GLOBAL_VAR REAL* KW_RESTRICT scratchX,
    KW_GLOBAL_VAR REAL* KW_RESTRICT coalRightYBar,
    KW_GLOBAL_VAR REAL* KW_RESTRICT coalRightX,
    KW_GLOBAL_VAR int*  KW_RESTRICT intervalStarts,
    KW_GLOBAL_VAR int*  KW_RESTRICT matTransIndices,
    KW_GLOBAL_VAR int*  KW_RESTRICT coalOps,
    KW_GLOBAL_VAR REAL* KW_RESTRICT matrixAdj,
    int intervalCount) {

    int interval = KW_GROUP_ID_0;
    if (interval >= intervalCount) return;

    int tilesPerRow = (PADDED_STATE_COUNT + EIGEN_TILE - 1) / EIGEN_TILE;
    int tileFlat = KW_GROUP_ID_1;
    int tileR = tileFlat / tilesPerRow;
    int tileC = tileFlat - tileR * tilesPerRow;

    int tx = KW_LOCAL_ID_0;
    int ty = KW_LOCAL_ID_1;
    int row = tileR * EIGEN_TILE + tx;
    int col = tileC * EIGEN_TILE + ty;

    int start  = intervalStarts[interval];
    int end = intervalStarts[interval + 1];
    int matIdx = matTransIndices[interval];
    int coalOp = coalOps[interval];
    int numOps = end - start;

    KW_LOCAL_MEM REAL sYBarTile[EIGEN_TILE][EIGEN_TILE + 1];
    KW_LOCAL_MEM REAL sXTile[EIGEN_TILE][EIGEN_TILE + 1];

    REAL acc = 0.0;
    int S = PADDED_STATE_COUNT;

    for (int opTile = 0; opTile < numOps; opTile += EIGEN_TILE) {
        int opIdxA = opTile + ty;
        if (opIdxA < numOps && row < S) {
            int op = start + opIdxA;
            sYBarTile[tx][ty] = scratchYBar[op * S + row];
        } else {
            sYBarTile[tx][ty] = 0.0;
        }

        int opIdxB = opTile + tx;
        if (opIdxB < numOps && col < S) {
            int op = start + opIdxB;
            sXTile[tx][ty] = scratchX[op * S + col];
        } else {
            sXTile[tx][ty] = 0.0;
        }

        KW_LOCAL_FENCE;

        for (int k = 0; k < EIGEN_TILE; k++) {
            acc += sYBarTile[tx][k] * sXTile[k][ty];
        }

        KW_LOCAL_FENCE;
    }

    if (coalOp >= 0) {
        REAL yR = (row < S) ? coalRightYBar[interval * S + row] : 0.0;
        REAL xC = (col < S) ? coalRightX   [interval * S + col] : 0.0;
        acc += yR * xC;
    }

    if (row < S && col < S) {
        matrixAdj[matIdx + row * S + col] = acc;
    }
}


KW_GLOBAL_KERNEL void kernelAccumulateMatrixAdjointsSlabEigen(
    KW_GLOBAL_VAR REAL* KW_RESTRICT scratchYBar,
    KW_GLOBAL_VAR REAL* KW_RESTRICT partialsTilde,
    KW_GLOBAL_VAR int*  KW_RESTRICT opInBufOff,
    KW_GLOBAL_VAR int*  KW_RESTRICT intervalOpStart,
    KW_GLOBAL_VAR int*  KW_RESTRICT intervalOpList,
    KW_GLOBAL_VAR REAL* KW_RESTRICT matrixAdj,
    int intervalCount) {

    int interval = KW_GROUP_ID_0;
    if (interval >= intervalCount) return;

    int tilesPerRow = (PADDED_STATE_COUNT + EIGEN_TILE - 1) / EIGEN_TILE;
    int tileFlat = KW_GROUP_ID_1;
    int tileR = tileFlat / tilesPerRow;
    int tileC = tileFlat - tileR * tilesPerRow;

    int tx = KW_LOCAL_ID_0;
    int ty = KW_LOCAL_ID_1;
    int row = tileR * EIGEN_TILE + tx;
    int col = tileC * EIGEN_TILE + ty;

    int start = intervalOpStart[interval];
    int end = intervalOpStart[interval + 1];
    int numOps = end - start;

    KW_LOCAL_MEM REAL sYBarTile[EIGEN_TILE][EIGEN_TILE + 1];
    KW_LOCAL_MEM REAL sXTile[EIGEN_TILE][EIGEN_TILE + 1];

    REAL acc = 0.0;
    int S = PADDED_STATE_COUNT;
    int matBase = interval * S * S;

    for (int opTile = 0; opTile < numOps; opTile += EIGEN_TILE) {
        int opIdxA = opTile + ty;
        if (opIdxA < numOps && row < S) {
            int op = intervalOpList[start + opIdxA];
            sYBarTile[tx][ty] = scratchYBar[op * S + row];
        } else {
            sYBarTile[tx][ty] = 0.0;
        }

        int opIdxB = opTile + tx;
        if (opIdxB < numOps && col < S) {
            int op = intervalOpList[start + opIdxB];
            int xtBase = opInBufOff[op];
            sXTile[tx][ty] = partialsTilde[xtBase + col];
        } else {
            sXTile[tx][ty] = 0.0;
        }

        KW_LOCAL_FENCE;

        for (int k = 0; k < EIGEN_TILE; k++) {
            acc += sYBarTile[tx][k] * sXTile[k][ty];
        }

        KW_LOCAL_FENCE;
    }

    if (row < S && col < S) {
        matrixAdj[matBase + row * S + col] = acc;
    }
}




#ifndef BASTA_HEIG_OPS_PER_BLOCK
#define BASTA_HEIG_OPS_PER_BLOCK 8
#endif

KW_GLOBAL_KERNEL void kernelProjectHazardsToEigen(
    KW_GLOBAL_VAR REAL* KW_RESTRICT partials,
    KW_GLOBAL_VAR REAL* KW_RESTRICT hazardAdjoints,
    KW_GLOBAL_VAR REAL* KW_RESTRICT evecT,
    KW_GLOBAL_VAR int*  KW_RESTRICT opUOff,
    KW_GLOBAL_VAR int*  KW_RESTRICT opKIn,
    KW_GLOBAL_VAR int*  KW_RESTRICT opKAcc,
    KW_GLOBAL_VAR int*  KW_RESTRICT opHasAcc,
    KW_GLOBAL_VAR REAL* KW_RESTRICT hazardEigenPerOp,
    int opCount,
    int stateCount) {

    int p = KW_LOCAL_ID_1;
    int a  = KW_LOCAL_ID_0;
    int globalOp = KW_GROUP_ID_0 * BASTA_HEIG_OPS_PER_BLOCK + p;

    int validOp = (globalOp < opCount) ? 1 : 0;
    int validA  = (a < stateCount) ? 1 : 0;

    KW_LOCAL_MEM REAL sH[BASTA_HEIG_OPS_PER_BLOCK][PADDED_STATE_COUNT + 1];

    if (validOp) {
        if (validA) {
            int u_n = opUOff[globalOp];
            int k_in = opKIn[globalOp];
            int hasAcc = opHasAcc[globalOp];
            int k_acc = hasAcc ? opKAcc[globalOp] : 0;

            int inBase = k_in * 4 * PADDED_STATE_COUNT;
            int accBase = k_acc * 4 * PADDED_STATE_COUNT;

            REAL p_a = partials[u_n + a];
            REAL h_a = hazardAdjoints[inBase + a]
                     + (REAL) 2.0 * p_a *
                       hazardAdjoints[inBase + PADDED_STATE_COUNT + a];
            if (hasAcc) {
                h_a += hazardAdjoints[accBase + 2 * PADDED_STATE_COUNT + a]
                     + (REAL) 2.0 * p_a *
                       hazardAdjoints[accBase + 3 * PADDED_STATE_COUNT + a];
            }
            sH[p][a] = h_a;
        } else if (a < PADDED_STATE_COUNT) {
            sH[p][a] = (REAL) 0.0;
        }
    } else if (a < PADDED_STATE_COUNT) {
        sH[p][a] = (REAL) 0.0;
    }
    KW_LOCAL_FENCE;


    if (validOp && validA) {
        REAL acc = (REAL) 0.0;
        for (int i = 0; i < stateCount; ++i) {
            acc += evecT[a * PADDED_STATE_COUNT + i] * sH[p][i];
        }
        hazardEigenPerOp[globalOp * PADDED_STATE_COUNT + a] = acc;
    } else if (validOp && a < PADDED_STATE_COUNT) {
        hazardEigenPerOp[globalOp * PADDED_STATE_COUNT + a] = (REAL) 0.0;
    }
}



KW_GLOBAL_KERNEL void kernelProjectPartialsToEigenbasis(
    KW_GLOBAL_VAR REAL* KW_RESTRICT partials,
    KW_GLOBAL_VAR REAL* KW_RESTRICT inverseEvec,
    KW_GLOBAL_VAR REAL* KW_RESTRICT partialsTilde,
    KW_GLOBAL_VAR int*  KW_RESTRICT bufferIndices,
    int bufferCount,
    int stateCount,
    int stride) {

    int buf  = KW_GROUP_ID_0;
    int a    = KW_LOCAL_ID_0;
    if (buf >= bufferCount || a >= PADDED_STATE_COUNT) return;

    int u = bufferIndices[buf];
    KW_GLOBAL_VAR REAL* xPtr = partials + u * stride;

    KW_LOCAL_MEM REAL sX[PADDED_STATE_COUNT];
    if (a < stateCount) sX[a] = xPtr[a];
    else                sX[a] = (REAL) 0.0;
    KW_LOCAL_FENCE;

    REAL acc = (REAL) 0.0;
    if (a < stateCount) {
        for (int i = 0; i < stateCount; ++i) {
            REAL c = inverseEvec[a * PADDED_STATE_COUNT + i] * sX[i];
            acc = acc + c;
        }
    }
    partialsTilde[u * stride + a] = acc;
}


KW_GLOBAL_KERNEL void kernelCoalescentSlab(
    KW_GLOBAL_VAR REAL* KW_RESTRICT partials,
    KW_GLOBAL_VAR REAL* KW_RESTRICT partialAdj,
    KW_GLOBAL_VAR REAL* KW_RESTRICT sizes,
    KW_GLOBAL_VAR REAL* KW_RESTRICT coalescent,
    KW_GLOBAL_VAR REAL* KW_RESTRICT popSizeGrad,
    KW_GLOBAL_VAR int*  KW_RESTRICT coalDestBufs,
    KW_GLOBAL_VAR int*  KW_RESTRICT coalLeftAccBufs,
    KW_GLOBAL_VAR int*  KW_RESTRICT coalRightAccBufs,
    KW_GLOBAL_VAR int*  KW_RESTRICT coalIntervals,
    int coalSlabOffset,
    int coalCount,
    int stateCount,
    int stride) {

    int c = KW_GROUP_ID_0 + coalSlabOffset;
    int i = KW_LOCAL_ID_0;
    if (KW_GROUP_ID_0 >= coalCount) return;
    int validState = (i < stateCount) ? 1 : 0;

    int destOff = coalDestBufs[c];
    int leftAccOff = coalLeftAccBufs[c];
    int rightAccOff = coalRightAccBufs[c];
    int interval = coalIntervals[c];

    REAL leftEnd = validState ? partials[leftAccOff  + i] : (REAL) 0.0;
    REAL rightEnd = validState ? partials[rightAccOff + i] : (REAL) 0.0;
    REAL N_i = validState ? sizes[i] : (REAL) 0.0;
    REAL J = coalescent[interval];

    REAL z_i = validState ? partialAdj[destOff + i] : (REAL) 0.0;
    REAL w_i = (N_i > 0) ? (leftEnd * rightEnd / N_i) : (REAL) 0.0;

    KW_LOCAL_MEM REAL sRed[PADDED_STATE_COUNT];
    sRed[i] = z_i * w_i;
    KW_LOCAL_FENCE;
    for (int s = PADDED_STATE_COUNT >> 1; s > 0; s >>= 1) {
        if (i < s) sRed[i] += sRed[i + s];
        KW_LOCAL_FENCE;
    }
    REAL dotZW = sRed[0];

    REAL adjJ = (J != (REAL) 0.0) ? ((REAL) 1.0 / J - dotZW / (J * J)) : (REAL) 0.0;
    REAL adjW_i = validState ? (z_i / J + adjJ) : (REAL) 0.0;

    if (validState) {
        REAL leftBar = (N_i > 0) ? (adjW_i * rightEnd / N_i) : (REAL) 0.0;
        REAL rightBar = (N_i > 0) ? (adjW_i * leftEnd  / N_i) : (REAL) 0.0;
        partialAdj[leftAccOff + i] = leftBar;
        partialAdj[rightAccOff + i] = rightBar;

        if (N_i > 0) {
            atomicAdd(&popSizeGrad[i], -adjW_i * leftEnd * rightEnd / (N_i * N_i));
        }
    }
}




KW_GLOBAL_KERNEL void kernelAdjointBranchSlabLocal(
    KW_GLOBAL_VAR REAL* KW_RESTRICT partials,
    KW_GLOBAL_VAR REAL* KW_RESTRICT partialAdj,
    KW_GLOBAL_VAR REAL* KW_RESTRICT scratchYBar,
    KW_GLOBAL_VAR REAL* KW_RESTRICT evecT,
    KW_GLOBAL_VAR REAL* KW_RESTRICT eigenValues,
    KW_GLOBAL_VAR int*  KW_RESTRICT blockStarts,
    KW_GLOBAL_VAR int*  KW_RESTRICT blockDims,
    KW_GLOBAL_VAR REAL* KW_RESTRICT hazardAdjoints,
    KW_GLOBAL_VAR REAL* KW_RESTRICT hazardEigenPerOp,

    KW_GLOBAL_VAR int*  KW_RESTRICT slabBlockBranchIdx,
    KW_GLOBAL_VAR int*  KW_RESTRICT slabBlockChunkStart,
    KW_GLOBAL_VAR int*  KW_RESTRICT slabBlockChunkLen,
    KW_GLOBAL_VAR int*  KW_RESTRICT slabBlockChunkIdx,

    KW_GLOBAL_VAR int*  KW_RESTRICT branchKb,
    KW_GLOBAL_VAR int*  KW_RESTRICT branchKTop,
    KW_GLOBAL_VAR int*  KW_RESTRICT branchTopBuf,
    KW_GLOBAL_VAR int*  KW_RESTRICT branchOpFirst,
    KW_GLOBAL_VAR int*  KW_RESTRICT branchTimeStart,
    KW_GLOBAL_VAR REAL* KW_RESTRICT branchT,

    KW_GLOBAL_VAR REAL* KW_RESTRICT slabCarryOut,
    KW_GLOBAL_VAR REAL* KW_RESTRICT slabAStash,
    KW_GLOBAL_VAR REAL* KW_RESTRICT slabYBottomEigen,
    int blockBase,
    int numBlocks,
    int numEvBlocks,
    int stateCount,
    int stride) {

    if (KW_GROUP_ID_0 >= numBlocks) return;
    int gGlobal = KW_GROUP_ID_0 + blockBase;

    int a = KW_LOCAL_ID_0;
    int t = KW_LOCAL_ID_1;
    int validA = (a < stateCount) ? 1 : 0;

    int branchIdx = slabBlockBranchIdx [gGlobal];
    int chunkStart = slabBlockChunkStart[gGlobal];
    int chunkLen = slabBlockChunkLen  [gGlobal];
    int chunkIdx = slabBlockChunkIdx  [gGlobal];

    int K_b = branchKb[branchIdx];
    int kTop = branchKTop[branchIdx];
    int topBufOff = branchTopBuf[branchIdx];
    int opFirst = branchOpFirst[branchIdx];
    int timeBase = branchTimeStart[branchIdx];


    int blockStart_a = a;
    int blockDim_a = 1;
    if (validA) {
        for (int eb = 0; eb < PADDED_STATE_COUNT; ++eb) {
            int s0 = blockStarts[eb];
            int d = blockDims[eb];
            if (a >= s0 && a < s0 + d) {
                blockStart_a = s0;
                blockDim_a = d;
                break;
            }
        }
    }
    REAL la_re = validA ? eigenValues[blockStart_a] : (REAL) 0.0;
    REAL la_im = (validA && blockDim_a == 2)? eigenValues[stateCount + blockStart_a] : (REAL) 0.0;

    KW_LOCAL_MEM REAL sBoundary[PADDED_STATE_COUNT];
    if (t == 0 && validA) {
        REAL y0 = (REAL) 0.0;
        const int hbBase = kTop * 4 * PADDED_STATE_COUNT;
        for (int i = 0; i < stateCount; ++i) {
            REAL p_i = partials  [topBufOff + i];
            REAL adjG_i = hazardAdjoints[hbBase + 2 * PADDED_STATE_COUNT + i];
            REAL adjH_i = hazardAdjoints[hbBase + 3 * PADDED_STATE_COUNT + i];
            REAL ybar_i = partialAdj[topBufOff + i] + adjG_i + (REAL) 2.0 * p_i * adjH_i;
            REAL c_b = evecT[a * PADDED_STATE_COUNT + i] * ybar_i;
            y0 = y0 + c_b;
        }
        sBoundary[a] = y0;
        if (chunkIdx == 0) {
            scratchYBar[opFirst * PADDED_STATE_COUNT + a] = y0;
        }
    }
    KW_LOCAL_FENCE;

    KW_LOCAL_MEM REAL sH[BASTA_SLAB_OPS_PER_BLOCK][PADDED_STATE_COUNT + 1];
    KW_LOCAL_MEM REAL sChunk[BASTA_SLAB_OPS_PER_BLOCK][PADDED_STATE_COUNT + 1];

    int n = chunkStart + t + 1;
    int valid_p = (t < chunkLen) && validA;

    REAL t_n = (REAL) 0.0;
    REAL A_re = (REAL) 1.0,  A_im    = (REAL) 0.0;
    REAL Ainv_re = (REAL) 1.0,  Ainv_im = (REAL) 0.0;
    REAL h_a = (REAL) 0.0;

    if (valid_p) {
        t_n = branchT[timeBase + (n - 1)];
        REAL eR = exp( t_n * la_re);
        REAL eRI = exp(-t_n * la_re);
        if (blockDim_a == 1) {
            A_re = eR;
			A_im = (REAL) 0.0;
            Ainv_re = eRI;
			Ainv_im = (REAL) 0.0;
        } else {
            REAL c = cos(t_n * la_im);
            REAL s = sin(t_n * la_im);
            A_re = eR  * c;    A_im = eR  * s;
            Ainv_re = eRI * c;    Ainv_im = -eRI * s;
        }

        int op_n = opFirst + (n - 1);
        h_a = hazardEigenPerOp[op_n * PADDED_STATE_COUNT + a];

        slabAStash[(op_n * PADDED_STATE_COUNT + a) * 2 + 0] = A_re;
        slabAStash[(op_n * PADDED_STATE_COUNT + a) * 2 + 1] = A_im;
    }

    sH[t][a] = h_a;
    KW_LOCAL_FENCE;

    REAL myS = (REAL) 0.0;
    if (valid_p) {
        if (blockDim_a == 1) {
            myS = h_a * Ainv_re;
        } else {
            REAL h_re = sH[t][blockStart_a];
            REAL h_im = sH[t][blockStart_a + 1];
            if (a == blockStart_a) {
                myS = Ainv_re * h_re - Ainv_im * h_im;
            } else {
                myS = Ainv_re * h_im + Ainv_im * h_re;
            }
        }
    }
    sChunk[t][a] = myS;
    KW_LOCAL_FENCE;

    /* Hillis-Steele inclusive scan */
    for (int stride_pow = 1; stride_pow < BASTA_SLAB_OPS_PER_BLOCK; stride_pow <<= 1) {
        REAL prev = (REAL) 0.0;
        if (t >= stride_pow) prev = sChunk[t - stride_pow][a];
        KW_LOCAL_FENCE;
        if (t >= stride_pow) sChunk[t][a] = sChunk[t][a] + prev;
        KW_LOCAL_FENCE;
    }

    if (valid_p) {
        REAL y_n = (REAL) 0.0;
        if (blockDim_a == 1) {
            REAL S_n = sChunk[t][a];
            y_n = A_re * (sBoundary[a] + S_n);
        } else {
            REAL S_re = sChunk[t][blockStart_a];
            REAL S_im = sChunk[t][blockStart_a + 1];
            REAL sum_re = sBoundary[blockStart_a] + S_re;
            REAL sum_im = sBoundary[blockStart_a + 1] + S_im;
            if (a == blockStart_a) {
                y_n = A_re * sum_re - A_im * sum_im;
            } else {
                y_n = A_re * sum_im + A_im * sum_re;
            }
        }
        if (n < K_b) {
            int op = opFirst + n;
            scratchYBar[op * PADDED_STATE_COUNT + a] = y_n;
        }
        if (n == K_b) {
            slabYBottomEigen[branchIdx * PADDED_STATE_COUNT + a] = y_n;
        }
    }
    KW_LOCAL_FENCE;

    if (t == 0 && validA) {
        slabCarryOut[gGlobal * PADDED_STATE_COUNT + a] = sChunk[chunkLen - 1][a];
    }
}




KW_GLOBAL_KERNEL void kernelAdjointBranchSlabSpine(
    KW_GLOBAL_VAR REAL* KW_RESTRICT slabCarryOut,
    KW_GLOBAL_VAR REAL* KW_RESTRICT slabCarryPrefix,
    KW_GLOBAL_VAR int*  KW_RESTRICT slabBlockBranchIdx,
    int blockBase,
    int numBlocks,
    int stateCount) {

    int a = KW_LOCAL_ID_0;
    int t = KW_LOCAL_ID_1;

    int ctaOff = (int)KW_GROUP_ID_0 * BASTA_SPINE_T;
    int g = blockBase + ctaOff + t;
    int valid  = (a < stateCount) & (ctaOff + t < numBlocks);

    KW_LOCAL_MEM REAL sBuf[BASTA_SPINE_T][PADDED_STATE_COUNT + 1];
    KW_LOCAL_MEM int sBid[BASTA_SPINE_T];

    /* Load. */
    sBuf[t][a] = valid ? slabCarryOut[g * PADDED_STATE_COUNT + a] : (REAL)0.0;
    if (a == 0) sBid[t] = (ctaOff + t < numBlocks) ? slabBlockBranchIdx[g] : -1;
    KW_LOCAL_FENCE;

    /* Hillis-Steele segmented inclusive scan */
    for (int s = 1; s < BASTA_SPINE_T; s <<= 1) {
        REAL add = (t >= s && sBid[t] >= 0 && sBid[t] == sBid[t - s])
                   ? sBuf[t - s][a] : (REAL)0.0;
        KW_LOCAL_FENCE;
        sBuf[t][a] += add;
        KW_LOCAL_FENCE;
    }

    REAL prefix = (t > 0 && sBid[t] >= 0 && sBid[t] == sBid[t - 1])
                  ? sBuf[t - 1][a] : (REAL)0.0;
    if (valid) slabCarryPrefix[g * PADDED_STATE_COUNT + a] = prefix;

    KW_LOCAL_FENCE;

    /* cross-CTA carry correction */
    if (t == 0 && a < stateCount && ctaOff > 0) {
        int g0 = blockBase + ctaOff;
        int firstBranch = slabBlockBranchIdx[g0];
        int prevBranch = slabBlockBranchIdx[g0 - 1];

        if (firstBranch == prevBranch) {
            REAL crossCarry = (REAL)0.0;
            for (int gi = g0 - 1; gi >= blockBase; gi--) {
                if (slabBlockBranchIdx[gi] != firstBranch) break;
                crossCarry += slabCarryOut[gi * PADDED_STATE_COUNT + a];
            }

            for (int tt = 0; tt < BASTA_SPINE_T && ctaOff + tt < numBlocks; tt++) {
                int gfix = blockBase + ctaOff + tt;
                if (slabBlockBranchIdx[gfix] != firstBranch) break;
                slabCarryPrefix[gfix * PADDED_STATE_COUNT + a] += crossCarry;
            }
        }
    }
}



KW_GLOBAL_KERNEL void kernelAdjointBranchSlabApply(
    KW_GLOBAL_VAR REAL* KW_RESTRICT partialAdj,
    KW_GLOBAL_VAR REAL* KW_RESTRICT scratchYBar,
    KW_GLOBAL_VAR REAL* KW_RESTRICT inverseEvecT,
    KW_GLOBAL_VAR REAL* KW_RESTRICT slabAStash,
    KW_GLOBAL_VAR REAL* KW_RESTRICT slabCarryPrefix,
    KW_GLOBAL_VAR REAL* KW_RESTRICT slabYBottomEigen,
    KW_GLOBAL_VAR int*  KW_RESTRICT slabBlockBranchIdx,
    KW_GLOBAL_VAR int*  KW_RESTRICT slabBlockChunkStart,
    KW_GLOBAL_VAR int*  KW_RESTRICT slabBlockChunkLen,
    KW_GLOBAL_VAR int*  KW_RESTRICT slabBlockChunkIdx,
    KW_GLOBAL_VAR int*  KW_RESTRICT branchKb,
    KW_GLOBAL_VAR int*  KW_RESTRICT branchBotBuf,
    KW_GLOBAL_VAR int*  KW_RESTRICT branchOpFirst,
    KW_GLOBAL_VAR int*  KW_RESTRICT blockStarts,
    KW_GLOBAL_VAR int*  KW_RESTRICT blockDims,
    int blockBase,
    int numBlocks,
    int numEvBlocks,
    int opsPerBlock,
    int stateCount,
    int stride) {

    if (KW_GROUP_ID_0 >= numBlocks) return;
    int gGlobal = KW_GROUP_ID_0 + blockBase;

    int a = KW_LOCAL_ID_0;
    int t = KW_LOCAL_ID_1;
    int validA = (a < stateCount) ? 1 : 0;

    int branchIdx = slabBlockBranchIdx[gGlobal];
    int chunkStart = slabBlockChunkStart[gGlobal];
    int chunkLen  = slabBlockChunkLen[gGlobal];
    int chunkIdx  = slabBlockChunkIdx [gGlobal];

    int K_b = branchKb[branchIdx];
    int botBuf = branchBotBuf [branchIdx];
    int opFirst = branchOpFirst[branchIdx];

    int numChunks = (K_b + opsPerBlock - 1) / opsPerBlock;
    int isLastChunk = (chunkIdx == numChunks - 1);


    int blockStart_a = a;
    int blockDim_a   = 1;
    if (validA) {
        for (int eb = 0; eb < PADDED_STATE_COUNT; ++eb) {
            int s0 = blockStarts[eb];
            int d = blockDims[eb];
            if (a >= s0 && a < s0 + d) {
                blockStart_a = s0;
                blockDim_a = d;
                break;
            }
        }
    }

    KW_LOCAL_MEM REAL sPrefix[PADDED_STATE_COUNT + 1];
    if (t == 0 && validA) {
        sPrefix[a] = slabCarryPrefix[gGlobal * PADDED_STATE_COUNT + a];
    }
    KW_LOCAL_FENCE;

    if (chunkIdx > 0) {
        int n = chunkStart + t + 1;
        if (t < chunkLen && n < K_b && validA) {
            int op_n = opFirst + (n - 1);
            REAL A_re = slabAStash[(op_n * PADDED_STATE_COUNT + a) * 2 + 0];
            REAL A_im = slabAStash[(op_n * PADDED_STATE_COUNT + a) * 2 + 1];
            REAL pre_re, pre_im;
            REAL corr;
            if (blockDim_a == 1) {
                pre_re = sPrefix[a];
                corr = A_re * pre_re;
            } else {
                pre_re = sPrefix[blockStart_a];
                pre_im = sPrefix[blockStart_a + 1];
                if (a == blockStart_a) {
                    corr = A_re * pre_re - A_im * pre_im;
                } else {
                    corr = A_re * pre_im + A_im * pre_re;
                }
            }
            scratchYBar[(opFirst + n) * PADDED_STATE_COUNT + a] += corr;
        }
    }

    if (isLastChunk) {
        KW_LOCAL_MEM REAL sYbottom[PADDED_STATE_COUNT + 1];
        if (t == 0 && validA) {
            REAL y_K = slabYBottomEigen[branchIdx * PADDED_STATE_COUNT + a];
            if (chunkIdx > 0) {
                int op_K = opFirst + (K_b - 1);
                REAL A_re = slabAStash[(op_K * PADDED_STATE_COUNT + a) * 2 + 0];
                REAL A_im = slabAStash[(op_K * PADDED_STATE_COUNT + a) * 2 + 1];
                REAL corr;
                if (blockDim_a == 1) {
                    corr = A_re * sPrefix[a];
                } else {
                    if (a == blockStart_a) {
                        corr = A_re * sPrefix[blockStart_a]
                             - A_im * sPrefix[blockStart_a + 1];
                    } else {
                        corr = A_re * sPrefix[blockStart_a + 1]
                             + A_im * sPrefix[blockStart_a];
                    }
                }
                y_K = y_K + corr;
            }
            sYbottom[a] = y_K;
        }
        KW_LOCAL_FENCE;

        if (t == 0 && validA) {
            REAL ybar_a = (REAL) 0.0;
            for (int aa = 0; aa < stateCount; ++aa) {
                REAL c_bp = inverseEvecT[a * PADDED_STATE_COUNT + aa] * sYbottom[aa];
                ybar_a = ybar_a + c_bp;
            }
            partialAdj[botBuf + a] = ybar_a;
        }
    }
}




KW_GLOBAL_KERNEL void kernelForwardBranchSlab(
    KW_GLOBAL_VAR REAL* KW_RESTRICT partials,
    KW_GLOBAL_VAR REAL* KW_RESTRICT partialsTilde,
    KW_GLOBAL_VAR REAL* KW_RESTRICT evecT,
    KW_GLOBAL_VAR REAL* KW_RESTRICT eigenValues,
    KW_GLOBAL_VAR int*  KW_RESTRICT blockStarts,
    KW_GLOBAL_VAR int*  KW_RESTRICT blockDims,

    KW_GLOBAL_VAR int*  KW_RESTRICT slabBlockBranchIdx,
    KW_GLOBAL_VAR int*  KW_RESTRICT slabBlockChunkStart,
    KW_GLOBAL_VAR int*  KW_RESTRICT slabBlockChunkLen,
    KW_GLOBAL_VAR int*  KW_RESTRICT slabBlockChunkIdx,

    KW_GLOBAL_VAR int*  KW_RESTRICT branchKb,
    KW_GLOBAL_VAR int*  KW_RESTRICT branchTopBuf,
    KW_GLOBAL_VAR int*  KW_RESTRICT branchBotBuf,
    KW_GLOBAL_VAR int*  KW_RESTRICT branchOpFirst,
    KW_GLOBAL_VAR int*  KW_RESTRICT branchTimeStart,
    KW_GLOBAL_VAR REAL* KW_RESTRICT branchT,
    KW_GLOBAL_VAR int*  KW_RESTRICT opInBufOff,
    int blockBase,
    int numBlocks,
    int numEvBlocks,
    int stateCount,
    int stride) {

    if (KW_GROUP_ID_0 >= numBlocks) return;
    int gGlobal = KW_GROUP_ID_0 + blockBase;

    int s = KW_LOCAL_ID_0;
    int t = KW_LOCAL_ID_1;
    int validS = (s < stateCount) ? 1 : 0;

    int branchIdx = slabBlockBranchIdx [gGlobal];
    int chunkStart = slabBlockChunkStart[gGlobal];
    int chunkLen = slabBlockChunkLen[gGlobal];
    int chunkIdx = slabBlockChunkIdx[gGlobal];

    int K_b = branchKb[branchIdx];
    int topBuf = branchTopBuf [branchIdx];
    int botBuf = branchBotBuf [branchIdx];
    int opFirst = branchOpFirst[branchIdx];
    int timeBase = branchTimeStart[branchIdx];

    int blockStart_s = s;
    int blockDim_s = 1;
    if (validS) {
        for (int eb = 0; eb < PADDED_STATE_COUNT; ++eb) {
            int s0 = blockStarts[eb];
            int dd = blockDims[eb];
            if (s >= s0 && s < s0 + dd) {
                blockStart_s = s0;
                blockDim_s   = dd;
                break;
            }
        }
    }
    REAL la_re = validS ? eigenValues[blockStart_s] : (REAL) 0.0;
    REAL la_im = (validS && blockDim_s == 2)
                 ? -eigenValues[stateCount + blockStart_s] : (REAL) 0.0;

    KW_LOCAL_MEM REAL sBotEigen[PADDED_STATE_COUNT + 1];
    KW_LOCAL_MEM REAL sTauUp   [BASTA_SLAB_OPS_PER_BLOCK];

    REAL totalT = (K_b > 0) ? branchT[timeBase + K_b - 1] : (REAL) 0.0;

    int n_op    = chunkStart + t + 1;
    int doWrite = (t < chunkLen) && (n_op < K_b);

    if (t == 0) {
        sBotEigen[s] = validS ? partialsTilde[botBuf + s] : (REAL) 0.0;
    }
    if (s == 0 && t < BASTA_SLAB_OPS_PER_BLOCK) {
        sTauUp[t] = doWrite ? (totalT - branchT[timeBase + n_op - 1])
                            : (REAL) 0.0;
    }
    KW_LOCAL_FENCE;
    KW_LOCAL_MEM REAL sTilde[BASTA_SLAB_OPS_PER_BLOCK][PADDED_STATE_COUNT + 1];

    REAL tilde_s = (REAL) 0.0;
    if (doWrite && validS) {
        REAL tau_up = sTauUp[t];
        REAL eR     = exp(tau_up * la_re);
        if (blockDim_s == 1) {
            tilde_s = eR * sBotEigen[s];
        } else {
            REAL c = cos(tau_up * la_im);
            REAL si = sin(tau_up * la_im);
            REAL p_re = sBotEigen[blockStart_s];
            REAL p_im = sBotEigen[blockStart_s + 1];
            if (s == blockStart_s) {
                tilde_s = eR * (c * p_re - si * p_im);
            } else {
                tilde_s = eR * (si * p_re + c  * p_im);
            }
        }
    }

    if (t < BASTA_SLAB_OPS_PER_BLOCK && s < PADDED_STATE_COUNT) {
        sTilde[t][s] = tilde_s;
    }

    KW_LOCAL_MEM REAL sTildeU0[PADDED_STATE_COUNT + 1];
    if (chunkIdx == 0 && t == 0 && s < PADDED_STATE_COUNT) {
        REAL tildeU0_s = (REAL) 0.0;
        if (validS) {
            REAL eR = exp(totalT * la_re);
            if (blockDim_s == 1) {
                tildeU0_s = eR * sBotEigen[s];
            } else {
                REAL c = cos(totalT * la_im);
                REAL si = sin(totalT * la_im);
                REAL p_re = sBotEigen[blockStart_s];
                REAL p_im = sBotEigen[blockStart_s + 1];
                if (s == blockStart_s) {
                    tildeU0_s = eR * (c * p_re - si * p_im);
                } else {
                    tildeU0_s = eR * (si * p_re + c  * p_im);
                }
            }
        }
        sTildeU0[s] = tildeU0_s;
    }
    KW_LOCAL_FENCE;

    KW_LOCAL_MEM REAL sV[BLOCK_PEELING_SIZE_SCA][PADDED_STATE_COUNT + 1];
    REAL r_acc = (REAL) 0.0;
    REAL r_acc_0 = (REAL) 0.0;
    int  do_u0 = (chunkIdx == 0) && (t == 0);

    KW_GLOBAL_VAR REAL* KW_RESTRICT vPtr = evecT;
    for (int i = 0; i < PADDED_STATE_COUNT; i += BLOCK_PEELING_SIZE_SCA) {
        if (t < BLOCK_PEELING_SIZE_SCA && s < PADDED_STATE_COUNT) {
            sV[t][s] = vPtr[t * PADDED_STATE_COUNT + s];
        }
        KW_LOCAL_FENCE;

        if (doWrite && validS) {
            for (int j = 0; j < BLOCK_PEELING_SIZE_SCA; ++j) {
                FMA(sV[j][s], sTilde[t][i + j], r_acc);
            }
        }
        if (do_u0 && validS) {
            for (int j = 0; j < BLOCK_PEELING_SIZE_SCA; ++j) {
                FMA(sV[j][s], sTildeU0[i + j], r_acc_0);
            }
        }
        vPtr += BLOCK_PEELING_SIZE_SCA * PADDED_STATE_COUNT;
        KW_LOCAL_FENCE;
    }

    if (doWrite && validS) {
        int uBufOff = opInBufOff[opFirst + (n_op - 1)];
        partials[uBufOff + s] = r_acc;
    }
    if (do_u0 && validS) {
        partials[topBuf + s] = r_acc_0;
    }
}


KW_GLOBAL_KERNEL void kernelForwardCoalescentSlab(
    KW_GLOBAL_VAR REAL* KW_RESTRICT partials,
    KW_GLOBAL_VAR REAL* KW_RESTRICT partialsTilde,
    KW_GLOBAL_VAR REAL* KW_RESTRICT inverseEvecT,
    KW_GLOBAL_VAR REAL* KW_RESTRICT sizes,
    KW_GLOBAL_VAR REAL* KW_RESTRICT coalescent,
    KW_GLOBAL_VAR int*  KW_RESTRICT coalDestBufs,
    KW_GLOBAL_VAR int*  KW_RESTRICT coalLeftAccBufs,
    KW_GLOBAL_VAR int*  KW_RESTRICT coalRightAccBufs,
    KW_GLOBAL_VAR int*  KW_RESTRICT coalIntervals,
    int coalSlabOffset,
    int coalCount,
    int stateCount,
    int stride) {

    if (KW_GROUP_ID_0 >= coalCount) return;
    int c = KW_GROUP_ID_0 + coalSlabOffset;
    int s = KW_LOCAL_ID_0;
    int validS = (s < stateCount) ? 1 : 0;

    int destOff = coalDestBufs    [c];
    int leftOff = coalLeftAccBufs [c];
    int rightOff = coalRightAccBufs[c];
    int interval = coalIntervals   [c];

    REAL N_s = validS ? sizes[s]              : (REAL) 0.0;
    REAL leftEnd = validS ? partials[leftOff  + s] : (REAL) 0.0;
    REAL rightEnd = validS ? partials[rightOff + s] : (REAL) 0.0;

    REAL raw = (N_s > (REAL) 0.0) ? (leftEnd * rightEnd / N_s) : (REAL) 0.0;

    KW_LOCAL_MEM REAL sRed[PADDED_STATE_COUNT + 1];
    sRed[s] = raw;
    KW_LOCAL_FENCE;
#ifdef IS_POWER_OF_TWO
    for (int stridePow = PADDED_STATE_COUNT >> 1; stridePow > 0; stridePow >>= 1) {
        if (s < stridePow) {
            sRed[s] += sRed[s + stridePow];
        }
        KW_LOCAL_FENCE;
    }
#else
    for (int stridePow = SMALLEST_POWER_OF_TWO / 2; stridePow > 0; stridePow >>= 1) {
        if (s < stridePow && s + stridePow < PADDED_STATE_COUNT) {
            sRed[s] += sRed[s + stridePow];
        }
        KW_LOCAL_FENCE;
    }
#endif
    REAL phi = sRed[0];

    REAL r_c = (phi != (REAL) 0.0) ? (raw / phi) : (REAL) 0.0;
    if (validS) {
        partials[destOff + s] = r_c;
    }
    if (s == 0) {
        coalescent[interval] = phi;
    }

    KW_LOCAL_MEM REAL sRc[PADDED_STATE_COUNT + 1];
    sRc[s] = r_c;
    KW_LOCAL_FENCE;

    REAL p_c = (REAL) 0.0;
    if (validS) {
        for (int ee = 0; ee < stateCount; ++ee) {
            p_c += inverseEvecT[ee * PADDED_STATE_COUNT + s] * sRc[ee];
        }
    }
    if (validS) {
        partialsTilde[destOff + s] = p_c;
    }
}




#ifdef CUDA
} // extern "C"
#endif //CUDA
