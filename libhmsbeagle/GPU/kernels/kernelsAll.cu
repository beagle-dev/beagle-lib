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
 */

#ifdef CUDA
    #include "libhmsbeagle/GPU/GPUImplDefs.h"
    #include <stdlib.h>
    #include <string.h>
    #include <stdio.h>
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

///////////////////////////////////////////////////////////////////////////////

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
            	Cs[tx] = -expat * sin(Cs[tx]);
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
            	Cs[tx] = -expat * sin(Cs[tx] + 0.0);
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

#ifdef CUDA
} // extern "C"
#endif //CUDA
