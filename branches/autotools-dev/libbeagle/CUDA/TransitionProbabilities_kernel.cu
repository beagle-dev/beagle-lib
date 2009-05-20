/*
 * @author Marc Suchard
 */
#ifndef _Included_TransitionProbabilitiesKernel
#define _Included_TransitionProbabilitiesKernel

/**************INCLUDES***********/
#include <stdio.h>
#include <string.h>
#include <cuda_runtime_api.h>
#include "CUDASharedFunctions.h"

/**************CODE***********/
#ifdef __cplusplus
extern "C" {
#endif

void checkCUDAError(const char *msg);

__global__ void matrixMulADB(REAL** listC, REAL* A, REAL* D, REAL* B,
		REAL* distanceQueue, int length, int wB, int totalMatrix) {

	__shared__ REAL *C;
	__shared__ REAL distance;

	int wMatrix = blockIdx.x % totalMatrix;

	// Block index
	int bx = blockIdx.x / totalMatrix;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int BLOCKS = gridDim.y;

	if (tx == 0 && ty == 0) {
		C = listC[wMatrix]; // Non-coalescent read
		distance = distanceQueue[wMatrix]; // Non-coalescent read
	}

	__syncthreads();

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

	__shared__ REAL As[MULTIPLY_BLOCK_SIZE][MULTIPLY_BLOCK_SIZE];
	__shared__ REAL Bs[MULTIPLY_BLOCK_SIZE][MULTIPLY_BLOCK_SIZE];
	__shared__ REAL Ds[MULTIPLY_BLOCK_SIZE];

	for (int i = 0; i < BLOCKS - 1; i++) {

		if (ty == 0)
			Ds[tx] = exp(D[d + tx] * distance);

		As[ty][tx] = A[a + PADDED_STATE_COUNT * ty + tx];
		Bs[ty][tx] = B[b + PADDED_STATE_COUNT * ty + tx];

		__syncthreads();

		for (int k = 0; k < MULTIPLY_BLOCK_SIZE; ++k)
			Csub += As[ty][k] * Ds[k] * Bs[k][tx];

		__syncthreads();

		a += aStep;
		b += bStep;
		d += MULTIPLY_BLOCK_SIZE;
	}

	// Last block is too long
	if (tx < EDGE && ty < EDGE) {
		if (ty == 0)
			Ds[tx] = exp(D[d + tx] * distance);

#ifndef KERNEL_PRINT_ENABLED
		__syncthreads();
#endif

		As[ty][tx] = A[a + PADDED_STATE_COUNT * ty + tx];
		Bs[ty][tx] = B[b + PADDED_STATE_COUNT * ty + tx];

	} else {

		if (ty == 0)
			Ds[tx] = 0;

		As[ty][tx] = 0;
		Bs[ty][tx] = 0;
	}

	__syncthreads();

	for (int k = 0; k < EDGE; k++)
		Csub += As[ty][k] * Ds[k] * Bs[k][tx];

	__syncthreads();

	// Write the block sub-matrix to device memory;
	// each thread writes one element

	if ((tx < EDGE || bx < BLOCKS - 1) && (ty < EDGE || by < BLOCKS - 1)) { // It's OK to write
		if (Csub < 0)
			C[PADDED_STATE_COUNT* MULTIPLY_BLOCK_SIZE * by
					+ MULTIPLY_BLOCK_SIZE * bx + PADDED_STATE_COUNT * ty + tx]
					= 0;
		else
			C[PADDED_STATE_COUNT* MULTIPLY_BLOCK_SIZE * by
					+ MULTIPLY_BLOCK_SIZE * bx + PADDED_STATE_COUNT * ty + tx]
					= Csub;

	}
}

extern "C" void nativeGPUGetTransitionProbabilitiesSquare(REAL **dPtrQueue, REAL *dEvec,
		REAL *dIevc, REAL *dEigenValues, REAL *distanceQueue, int totalMatrix) {

#ifdef DEBUG
	fprintf(stderr,"Starting GPU TP\n");
	cudaThreadSynchronize();
	checkCUDAError("TP kernel pre-invocation");
#endif

	dim3 block(MULTIPLY_BLOCK_SIZE, MULTIPLY_BLOCK_SIZE);
	dim3 grid(PADDED_STATE_COUNT/MULTIPLY_BLOCK_SIZE, PADDED_STATE_COUNT/MULTIPLY_BLOCK_SIZE);
	if (PADDED_STATE_COUNT% MULTIPLY_BLOCK_SIZE != 0) {
		grid.x += 1;
		grid.y += 1;
	}
	grid.x *= totalMatrix;

	matrixMulADB<<<grid,block>>>(dPtrQueue, dIevc, dEigenValues, dEvec, distanceQueue, PADDED_STATE_COUNT, PADDED_STATE_COUNT, totalMatrix); // Transposed (interchanged Ievc and Evec)

#ifdef DEBUG
	fprintf(stderr,"Ending GPU TP\n");
	cudaThreadSynchronize();
	checkCUDAError("TP kernel invocation");
#endif
}

#ifdef __cplusplus
}
#endif
#endif

