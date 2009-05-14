/*
 * @author Marc Suchard
 */
#ifndef _Included_PeelingKernel
#define _Included_PeelingKernel

/**************INCLUDES***********/
#include <stdio.h>
#include <cuda_runtime_api.h>
#include "CUDASharedFunctions.h"

/**************CODE***********/
#ifdef __cplusplus
extern "C" {
#endif

void checkCUDAError(const char *msg);

REAL *ones = NULL; // TODO Memory leak, need to free at some point.

__global__ void kernelPartialsPartialsFixedScaling(REAL* partials1, REAL* partials2,
		REAL* partials3, REAL* matrices1, REAL* matrices2, REAL* scalingFactors, int totalPatterns) {

	REAL sum1 = 0;
	REAL sum2 = 0;
	int i;

	//  OLD
	// blockIdx.y = matrix;
	// blockIdx.x = pattern;  NOW, pattern = blockIdx.x*PATTERN_BLOCK_SIZE + threadIdx.y
	// threadIdx.x = state
	// blockDim.x = PADDED_STATE_COUNT;
	// gridDim.y = matrixCount;
	// gridDim.x = patternCount; NOW, meaningless

	int state = threadIdx.x;
	int patIdx = threadIdx.y;

	int pattern = __umul24(blockIdx.x,PATTERN_BLOCK_SIZE) + patIdx;
	int matrix = blockIdx.y;
	int patternCount = totalPatterns; // gridDim.x;

	int deltaPartialsByState = __umul24(pattern,PADDED_STATE_COUNT);
	int deltaPartialsByMatrix = __umul24(matrix, __umul24( PADDED_STATE_COUNT, patternCount));

	int x = __umul24(matrix, PADDED_STATE_COUNT * PADDED_STATE_COUNT);

	REAL *matrix1 = matrices1 + x; // Points to *this* matrix
	REAL *matrix2 = matrices2 + x;

	int y = deltaPartialsByState + deltaPartialsByMatrix;
	int u = state + deltaPartialsByState + deltaPartialsByMatrix;

	// Load values into shared memory
	__shared__ REAL sMatrix1[BLOCK_PEELING_SIZE][PADDED_STATE_COUNT];
	__shared__ REAL sMatrix2[BLOCK_PEELING_SIZE][PADDED_STATE_COUNT];

	__shared__ REAL sPartials1[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];
	__shared__ REAL sPartials2[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];

	__shared__ REAL fixedScalingFactors[PATTERN_BLOCK_SIZE];

	// copy PADDED_STATE_COUNT*PATTERN_BLOCK_SIZE lengthed partials
	if (pattern < totalPatterns) {
		sPartials1[patIdx][state] = partials1[y + state]; // These are all coherent global memory reads; checked in Profiler
		sPartials2[patIdx][state] = partials2[y + state];
	} else {
		sPartials1[patIdx][state] = 0;
		sPartials2[patIdx][state] = 0;
	}

	if (patIdx == 0 && state < PATTERN_BLOCK_SIZE )
		fixedScalingFactors[state] = scalingFactors[blockIdx.x*PATTERN_BLOCK_SIZE + state]; // TODO If PATTERN_BLOCK_SIZE > PADDED_STATE_COUNT, there is a bug here

	for (i = 0; i < PADDED_STATE_COUNT; i+=BLOCK_PEELING_SIZE) {
		// load one row of matrices
		if (patIdx < BLOCK_PEELING_SIZE) {
			sMatrix1[patIdx][state] = matrix1[patIdx*PADDED_STATE_COUNT + state]; // These are all coherent global memory reads.
			sMatrix2[patIdx][state] = matrix2[patIdx*PADDED_STATE_COUNT + state];

			// sMatrix now filled with starting in state and ending in i
			matrix1 += BLOCK_PEELING_SIZE*PADDED_STATE_COUNT;
			matrix2 += BLOCK_PEELING_SIZE*PADDED_STATE_COUNT;
		}
		__syncthreads();

		int j;
		for(j=0; j<BLOCK_PEELING_SIZE; j++) {
			sum1 += sMatrix1[j][state] * sPartials1[patIdx][i+j];
			sum2 += sMatrix2[j][state] * sPartials2[patIdx][i+j];
		}

		__syncthreads(); // GTX280 FIX HERE

	}

	if (pattern < totalPatterns)
		partials3[u] = sum1 * sum2 / fixedScalingFactors[patIdx];

}
/*
 * Find a scaling factor for each pattern
 */
__global__ void kernelPartialsDynamicScaling(REAL *allPartials, REAL *scalingFactors, int matrixCount) {

	int state = threadIdx.x;
	int matrix = threadIdx.y;
	int pattern = blockIdx.x;
	int patternCount = gridDim.x;

	int deltaPartialsByMatrix = __umul24(matrix, __umul24( PADDED_STATE_COUNT, patternCount));

	__shared__ REAL partials[MATRIX_BLOCK_SIZE][PADDED_STATE_COUNT]; // TODO Currently assumes MATRIX_BLOCK_SIZE > matrixCount; FIX!!!

	__shared__ REAL max;

	if ( matrix < matrixCount )
		partials[matrix][state] = allPartials[matrix*patternCount*PADDED_STATE_COUNT + pattern*PADDED_STATE_COUNT + state];
	else
		partials[matrix][state] = 0;

	__syncthreads();

	int i;
	for (i=PADDED_STATE_COUNT/2; i>0; i>>=1) { // parallelized reduction; assumes PADDED_STATE_COUNT is power of 2.
		if (state < i) {
			REAL compare1 = partials[matrix][state];
			REAL compare2 = partials[matrix][state+i];
			if (compare2 > compare1)
			partials[matrix][state] = compare2;
		}
		__syncthreads();
	}

	if (state == 0 && matrix == 0) {
		max = 0;
		int m;
		for(m=0; m<matrixCount; m++) {
			if (partials[m][0] > max)
				max = partials[m][0];
		}

		scalingFactors[pattern] = max; // TODO These are incoherent memory writes!!!
	}

	__syncthreads();

	if ( matrix < matrixCount )
		allPartials[matrix*patternCount*PADDED_STATE_COUNT + pattern*PADDED_STATE_COUNT + state] /= max;

	__syncthreads();

}


__global__ void kernelPartialsDynamicScalingSlow(REAL *allPartials, REAL *scalingFactors, int matrixCount) {

	int state = threadIdx.x;
	int matrix = threadIdx.y;
	int pattern = blockIdx.x;
	int patternCount = gridDim.x;

	int deltaPartialsByMatrix = __umul24(matrix, __umul24( PADDED_STATE_COUNT, patternCount));

	__shared__ REAL partials[PADDED_STATE_COUNT];

	__shared__ REAL max;

	if (state == 0)
		max = 0.0;

	int m;
	for(m=0; m<matrixCount; m++) {

		partials[state] = allPartials[m*patternCount*PADDED_STATE_COUNT + pattern*PADDED_STATE_COUNT + state];
		__syncthreads();

#ifdef IS_POWER_OF_TWO
	for (int i=PADDED_STATE_COUNT/2; i>0; i>>=1) { // parallelized reduction *** only works for powers-of-2 ****
		if (state < i) {
#else
	for (int i=SMALLEST_POWER_OF_TWO/2; i>0; i>>=1) {
		if (state < i && state+i < PADDED_STATE_COUNT ) {
#endif // IS_POWER_OF_TWO
				REAL compare1 = partials[state];
				REAL compare2 = partials[state+i];
				if( compare2 > compare1)
					partials[state] = compare2;
			}
			__syncthreads();
		}
		if( state == 0) {
			if( partials[0] > max)
				max = partials[0];
		}
	}

	if( state == 0 )
		scalingFactors[pattern] = max;

	__syncthreads();

	for(m=0; m<matrixCount; m++)
		allPartials[m*patternCount*PADDED_STATE_COUNT + pattern*PADDED_STATE_COUNT + state] /= max;

}


__global__ void kernelPartialsPartialsByPatternBlockCoherent(REAL* partials1, REAL* partials2,
		REAL* partials3, REAL* matrices1, REAL* matrices2, int totalPatterns) {

	REAL sum1 = 0;
	REAL sum2 = 0;
	int i;

	//  OLD
	// blockIdx.y = matrix;
	// blockIdx.x = pattern;  NOW, pattern = blockIdx.x*PATTERN_BLOCK_SIZE + threadIdx.y
	// threadIdx.x = state
	// blockDim.x = PADDED_STATE_COUNT;
	// gridDim.y = matrixCount;
	// gridDim.x = patternCount; NOW, meaningless

	int state = threadIdx.x;
	int patIdx = threadIdx.y;

	int pattern = __umul24(blockIdx.x,PATTERN_BLOCK_SIZE) + patIdx;
	int matrix = blockIdx.y;
	int patternCount = totalPatterns; // gridDim.x;

	int deltaPartialsByState = __umul24(pattern,PADDED_STATE_COUNT);
	int deltaPartialsByMatrix = __umul24(matrix, __umul24( PADDED_STATE_COUNT, patternCount));

	int x = __umul24(matrix, PADDED_STATE_COUNT * PADDED_STATE_COUNT);

	REAL *matrix1 = matrices1 + x; // Points to *this* matrix
	REAL *matrix2 = matrices2 + x;

	int y = deltaPartialsByState + deltaPartialsByMatrix;
	int u = state + deltaPartialsByState + deltaPartialsByMatrix;

	// Load values into shared memory
	__shared__ REAL sMatrix1[BLOCK_PEELING_SIZE][PADDED_STATE_COUNT];
	__shared__ REAL sMatrix2[BLOCK_PEELING_SIZE][PADDED_STATE_COUNT];

	__shared__ REAL sPartials1[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];
	__shared__ REAL sPartials2[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];

	// copy PADDED_STATE_COUNT*PATTERN_BLOCK_SIZE lengthed partials
	if (pattern < totalPatterns) {
		sPartials1[patIdx][state] = partials1[y + state]; // These are all coherent global memory reads; checked in Profiler
		sPartials2[patIdx][state] = partials2[y + state];
	} else {
		sPartials1[patIdx][state] = 0;
		sPartials2[patIdx][state] = 0;
	}

	for (i = 0; i < PADDED_STATE_COUNT; i+=BLOCK_PEELING_SIZE) {
		// load one row of matrices
		if (patIdx < BLOCK_PEELING_SIZE) {
			sMatrix1[patIdx][state] = matrix1[patIdx*PADDED_STATE_COUNT + state]; // These are all coherent global memory reads.
			sMatrix2[patIdx][state] = matrix2[patIdx*PADDED_STATE_COUNT + state];

			// sMatrix now filled with starting in state and ending in i
			matrix1 += BLOCK_PEELING_SIZE*PADDED_STATE_COUNT;
			matrix2 += BLOCK_PEELING_SIZE*PADDED_STATE_COUNT;
		}
		__syncthreads();

		int j;
		for(j=0; j<BLOCK_PEELING_SIZE; j++) {
			sum1 += sMatrix1[j][state] * sPartials1[patIdx][i+j];
			sum2 += sMatrix2[j][state] * sPartials2[patIdx][i+j];
		}

		__syncthreads(); // GTX280 FIX HERE

	}

	if (pattern < totalPatterns)
		partials3[u] = sum1 * sum2;
}


#if (PADDED_STATE_COUNT == 4)
__global__ void kernelPartialsPartialsByPatternBlockCoherentSmall(REAL* partials1, REAL* partials2,
		REAL* partials3, REAL* matrices1, REAL* matrices2, int totalPatterns) {

	REAL sum1 = 0;
	REAL sum2 = 0;
	int i;

	int tx = threadIdx.x;
	int state = tx % 4;
	int pat = tx / 4;
	int patIdx = threadIdx.y;
	int matrix = blockIdx.y;
	int patternCount = totalPatterns; // gridDim.x;

	int pattern = __umul24(blockIdx.x,PATTERN_BLOCK_SIZE*4) + patIdx*4 + pat; // read 4 patterns at a time, since 4 * 4 = 16




//	int deltaPartialsByState = __umul24(pattern,PADDED_STATE_COUNT);
	int deltaPartialsByState = 4 * 4 * (blockIdx.x * PATTERN_BLOCK_SIZE + patIdx);
	int deltaPartialsByMatrix = __umul24(matrix, __umul24( PADDED_STATE_COUNT, patternCount));

	int x2 = __umul24(matrix, PADDED_STATE_COUNT * PADDED_STATE_COUNT);

	REAL *matrix1 = matrices1 + x2; // Points to *this* matrix
	REAL *matrix2 = matrices2 + x2;


	int y = deltaPartialsByState + deltaPartialsByMatrix;
	int u = tx + deltaPartialsByState + deltaPartialsByMatrix;

#ifdef KERNEL_PRINT_ENABLED
	printf("matrix = %d, pat = %d for tx = %d and state = %d :  u = %d\n",matrix,pattern,tx,state,u);
#endif

	// Load values into shared memory
	__shared__ REAL sMatrix1[16];
	__shared__ REAL sMatrix2[16];

	__shared__ REAL sPartials1[PATTERN_BLOCK_SIZE*4*4];
	__shared__ REAL sPartials2[PATTERN_BLOCK_SIZE*4*4];

	// copy PADDED_STATE_COUNT*PATTERN_BLOCK_SIZE lengthed partials
	if (pattern < totalPatterns) {
		sPartials1[patIdx*16 + tx] = partials1[y + tx]; // All coalesced memory reads
		sPartials2[patIdx*16 + tx] = partials2[y + tx];
	} else {
		sPartials1[patIdx*16 + tx] = 0;
		sPartials2[patIdx*16 + tx] = 0;
	}

	if (patIdx == 0 ) {
		sMatrix1[tx] = matrix1[tx]; // All coalesced memory reads
		sMatrix2[tx] = matrix2[tx];
	}

	__syncthreads();

	if (pattern < totalPatterns) { // Remove padded threads!
		for(i=0; i<PADDED_STATE_COUNT; i++) {
			sum1 += sMatrix1[i*4 + state] * sPartials1[patIdx*16 + pat*4 + i];
			sum2 += sMatrix2[i*4 + state] * sPartials2[patIdx*16 + pat*4 + i];
		}
		partials3[u] = sum1 * sum2;
	}

}

__global__ void kernelPartialsPartialsSmallFixedScaling(REAL* partials1, REAL* partials2,
		REAL* partials3, REAL* matrices1, REAL* matrices2, REAL *scalingFactors, int totalPatterns) {

	REAL sum1 = 0;
	REAL sum2 = 0;
	int i;

	int tx = threadIdx.x;
	int state = tx % 4;
	int pat = tx / 4;
	int patIdx = threadIdx.y;
	int matrix = blockIdx.y;
	int patternCount = totalPatterns; // gridDim.x;

	int pattern = __umul24(blockIdx.x,PATTERN_BLOCK_SIZE*4) + patIdx*4 + pat; // read 4 patterns at a time, since 4 * 4 = 16

//	int deltaPartialsByState = __umul24(pattern,PADDED_STATE_COUNT);
	int deltaPartialsByState = 4 * 4 * (blockIdx.x * PATTERN_BLOCK_SIZE + patIdx);
	int deltaPartialsByMatrix = __umul24(matrix, __umul24( PADDED_STATE_COUNT, patternCount));

	int x2 = __umul24(matrix, PADDED_STATE_COUNT * PADDED_STATE_COUNT);

	REAL *matrix1 = matrices1 + x2; // Points to *this* matrix
	REAL *matrix2 = matrices2 + x2;


	int y = deltaPartialsByState + deltaPartialsByMatrix;
	int u = tx + deltaPartialsByState + deltaPartialsByMatrix;

#ifdef KERNEL_PRINT_ENABLED
	printf("matrix = %d, pat = %d for tx = %d and state = %d :  u = %d\n",matrix,pattern,tx,state,u);
#endif

	// Load values into shared memory
	__shared__ REAL sMatrix1[16];
	__shared__ REAL sMatrix2[16];

	__shared__ REAL sPartials1[PATTERN_BLOCK_SIZE*4*4];
	__shared__ REAL sPartials2[PATTERN_BLOCK_SIZE*4*4];

	__shared__ REAL fixedScalingFactors[PATTERN_BLOCK_SIZE*4];

	// copy PADDED_STATE_COUNT*PATTERN_BLOCK_SIZE lengthed partials
	if (pattern < totalPatterns) {
		sPartials1[patIdx*16 + tx] = partials1[y + tx]; // All coalesced memory reads
		sPartials2[patIdx*16 + tx] = partials2[y + tx];
	} else {
		sPartials1[patIdx*16 + tx] = 0;
		sPartials2[patIdx*16 + tx] = 0;
	}

	if (patIdx < 4) // need to load 4*PATTERN_BLOCK_SIZE factors for this block
		fixedScalingFactors[patIdx*PATTERN_BLOCK_SIZE + tx] = scalingFactors[blockIdx.x*PATTERN_BLOCK_SIZE*4 + patIdx*PATTERN_BLOCK_SIZE + tx];

	if (patIdx == 0 ) {
		sMatrix1[tx] = matrix1[tx]; // All coalesced memory reads
		sMatrix2[tx] = matrix2[tx];
	}

	__syncthreads();

	if (pattern < totalPatterns) { // Remove padded threads!
		for(i=0; i<PADDED_STATE_COUNT; i++) {
			sum1 += sMatrix1[i*4 + state] * sPartials1[patIdx*16 + pat*4 + i];
			sum2 += sMatrix2[i*4 + state] * sPartials2[patIdx*16 + pat*4 + i];
		}
		partials3[u] = sum1 * sum2 / fixedScalingFactors[patIdx*4 + pat];
	}

}


__global__ void kernelStatesStatesByPatternBlockCoherentSmall(int* states1, int* states2,
		REAL* partials3, REAL* matrices1, REAL* matrices2, int totalPatterns) {

	int tx = threadIdx.x;
	int state = tx % 4;
	int pat = tx / 4;
	int patIdx = threadIdx.y;
	int matrix = blockIdx.y;
	int patternCount = totalPatterns; // gridDim.x;

	int pattern = __umul24(blockIdx.x,PATTERN_BLOCK_SIZE*4) + patIdx*4 + pat; // read 4 patterns at a time, since 4 * 4 = 16

	int deltaPartialsByState = 4 * 4 * (blockIdx.x * PATTERN_BLOCK_SIZE + patIdx);
	int deltaPartialsByMatrix = __umul24(matrix, __umul24( PADDED_STATE_COUNT, patternCount));

	int x2 = __umul24(matrix, PADDED_STATE_COUNT * PADDED_STATE_COUNT);

	REAL *matrix1 = matrices1 + x2; // Points to *this* matrix
	REAL *matrix2 = matrices2 + x2;

	int u = tx + deltaPartialsByState + deltaPartialsByMatrix;

#ifdef KERNEL_PRINT_ENABLED
	printf("matrix = %d, pat = %d for tx = %d and state = %d :  u = %d\n",matrix,pattern,tx,state,u);
#endif

	// Load values into shared memory
	__shared__ REAL sMatrix1[16];
	__shared__ REAL sMatrix2[16];

//	__shared__ INT sStates1[PATTERN_BLOCK_SIZE*4];
//	__shared__ INT sStates2[PATTERN_BLOCK_SIZE*4];
//
//	if (pattern < totalPatterns) {
//		if (patIdx < PATTERN_BLOCK_SIZE/4) {
//			sStates1[patIdx*16 + tx] = states1[blockIdx.x*PATTERN_BLOCK_SIZE*4 + patIdx*16 + tx];
//			sStates2[patIdx*16 + tx] = states2[blockIdx.x*PATTERN_BLOCK_SIZE*4 + patIdx*16 + tx];
//		} else {
//			sStates1[patIdx*16 + tx] = PADDED_STATE_COUNT;
//			sStates2[patIdx*16 + tx] = PADDED_STATE_COUNT;
//		}
//	}

	if (patIdx == 0 ) {
		sMatrix1[tx] = matrix1[tx]; // All coalesced memory reads
		sMatrix2[tx] = matrix2[tx];
	}

	__syncthreads();

	if (pattern < totalPatterns) {
		int state1 = states1[pattern];
		int state2 = states2[pattern];
//		int state1 = sStates1[patIdx*4 + pat];
//		int state2 = sStates2[patIdx*4 + pat];

		if ( state1 < PADDED_STATE_COUNT && state2 < PADDED_STATE_COUNT) {
			partials3[u] = sMatrix1[state1*4 + state] *sMatrix2[state2*4 + state];
		} else if (state1 < PADDED_STATE_COUNT) {
			partials3[u] = sMatrix1[state1*4 + state];
		} else if (state2 < PADDED_STATE_COUNT) {
			partials3[u] = sMatrix2[state2*4 + state];
		} else {
			partials3[u] = 1.0;
		}
	}
}

__global__ void kernelStatesPartialsByPatternBlockCoherentSmall(int* states1, REAL* partials2,
		REAL* partials3, REAL* matrices1, REAL* matrices2, int totalPatterns) {

	REAL sum1 = 0;
	REAL sum2 = 0;
	int i;

	int tx = threadIdx.x;
	int state = tx % 4;
	int pat = tx / 4;
	int patIdx = threadIdx.y;
	int matrix = blockIdx.y;
	int patternCount = totalPatterns; // gridDim.x;

	int pattern = __umul24(blockIdx.x,PATTERN_BLOCK_SIZE*4) + patIdx*4 + pat; // read 4 patterns at a time, since 4 * 4 = 16

	int deltaPartialsByState = 4 * 4 * (blockIdx.x * PATTERN_BLOCK_SIZE + patIdx);
	int deltaPartialsByMatrix = __umul24(matrix, __umul24( PADDED_STATE_COUNT, patternCount));

	int x2 = __umul24(matrix, PADDED_STATE_COUNT * PADDED_STATE_COUNT);

	REAL *matrix1 = matrices1 + x2; // Points to *this* matrix
	REAL *matrix2 = matrices2 + x2;


	int y = deltaPartialsByState + deltaPartialsByMatrix;
	int u = tx + deltaPartialsByState + deltaPartialsByMatrix;

#ifdef KERNEL_PRINT_ENABLED
	printf("matrix = %d, pat = %d for tx = %d and state = %d :  u = %d\n",matrix,pattern,tx,state,u);
#endif

	// Load values into shared memory
	__shared__ REAL sMatrix1[16];
	__shared__ REAL sMatrix2[16];

//	__shared__ INT sStates1[PATTERN_BLOCK_SIZE*4];
	__shared__ REAL sPartials2[PATTERN_BLOCK_SIZE*4*4];

	// copy PADDED_STATE_COUNT*PATTERN_BLOCK_SIZE lengthed partials
	if (pattern < totalPatterns) {
		sPartials2[patIdx*16 + tx] = partials2[y + tx];
//		if (patIdx < PATTERN_BLOCK_SIZE/4)
//			sStates1[patIdx*16 + tx] = states1[blockIdx.x*PATTERN_BLOCK_SIZE*4 + patIdx*16 + tx];
	} else {
		sPartials2[patIdx*16 + tx] = 0;
//		if (patIdx < PATTERN_BLOCK_SIZE/4)
//			sStates1[patIdx*16 + tx] = PADDED_STATE_COUNT;
	}

	if (patIdx == 0 ) {
		sMatrix1[tx] = matrix1[tx]; // All coalesced memory reads
		sMatrix2[tx] = matrix2[tx];
	}

	__syncthreads();

	if (pattern < totalPatterns) { // Remove padded threads!
//		int state1 = sStates1[patIdx*4 + pat];
		int state1 = states1[pattern];

		if (state1 < PADDED_STATE_COUNT)
			sum1 = sMatrix1[state1*4 + state];
		else
			sum1 = 1.0;

		for(i=0; i<PADDED_STATE_COUNT; i++) {
			sum2 += sMatrix2[i*4 + state] * sPartials2[patIdx*16 + pat*4 + i];
		}
		partials3[u] = sum1 * sum2;
	}

}



#endif // PADDED_STATE_COUNT == 4

extern "C" void nativeGPUPartialsPartialsPruningDynamicScaling(
	REAL* partials1, REAL* partials2, REAL* partials3, REAL* matrices1, REAL* matrices2, REAL *scalingFactors,
	const unsigned int patternCount, const unsigned int matrixCount, int doRescaling) {

#ifdef DEBUG
	fprintf(stderr,"Entering GPU PP\n");
	cudaThreadSynchronize();
	checkCUDAError("PP kernel pre-invocation");
#endif

#if (PADDED_STATE_COUNT == 4)
	dim3 grid(patternCount/(PATTERN_BLOCK_SIZE*4), matrixCount);
	if (patternCount % (PATTERN_BLOCK_SIZE*4) != 0)
		grid.x +=1;
	dim3 block(16,PATTERN_BLOCK_SIZE);
#else
	dim3 grid(patternCount/PATTERN_BLOCK_SIZE, matrixCount);
	if (patternCount % PATTERN_BLOCK_SIZE != 0)
		grid.x += 1;
	dim3 block(PADDED_STATE_COUNT,PATTERN_BLOCK_SIZE);
#endif

	if (doRescaling)	{
		// Compute partials without any rescaling
#if (PADDED_STATE_COUNT == 4)
		kernelPartialsPartialsByPatternBlockCoherentSmall<<<grid, block>>>(partials1, partials2, partials3, matrices1, matrices2, patternCount);
#else
		kernelPartialsPartialsByPatternBlockCoherent<<<grid, block>>>(partials1, partials2, partials3, matrices1, matrices2, patternCount);
#endif

		cudaThreadSynchronize();

		// Rescale partials and save scaling factors

#ifndef SLOW_REWEIGHING
		dim3 grid2(patternCount,matrixCount/MATRIX_BLOCK_SIZE);
		if (matrixCount % MATRIX_BLOCK_SIZE != 0)
			grid2.y += 1;
		if (grid2.y > 1) {
			fprintf(stderr,"Not yet implemented! Try slow reweighing.\n");
			exit(0);
		}
		dim3 block2(PADDED_STATE_COUNT,MATRIX_BLOCK_SIZE);
		kernelPartialsDynamicScaling<<<grid2,block2>>>(partials3,scalingFactors,matrixCount); // TODO Totally incoherent for PADDED_STATE_COUNT == 4
#else
		dim3 grid2(patternCount,1);
		dim3 block2(PADDED_STATE_COUNT);
		kernelPartialsDynamicScalingSlow<<<grid2,block2>>>(partials3,scalingFactors,matrixCount);

#endif

	} else {

	// Compute partials with known rescalings
#if (PADDED_STATE_COUNT == 4)
		kernelPartialsPartialsSmallFixedScaling<<<grid,block>>>(partials1,partials2,partials3,matrices1,matrices2, scalingFactors, patternCount);
#else
		kernelPartialsPartialsFixedScaling<<<grid,block>>>(partials1,partials2,partials3,matrices1,matrices2, scalingFactors, patternCount);
#endif

	}

#ifdef DEBUG
	cudaThreadSynchronize();
	checkCUDAError("PP kernel invocation");
	fprintf(stderr,"Completed GPU PP\n");
#endif

}

extern "C" void nativeGPUPartialsPartialsPruning(
	REAL* partials1, REAL* partials2, REAL* partials3, REAL* matrices1, REAL* matrices2,
	const unsigned int patternCount, const unsigned int matrixCount) {

#ifdef DEBUG
	fprintf(stderr,"Entering GPU PP\n");
	cudaThreadSynchronize();
	checkCUDAError("PP kernel pre-invocation");
#endif


#if (PADDED_STATE_COUNT == 4)
	dim3 block(16,PATTERN_BLOCK_SIZE);
	dim3 grid(patternCount/(PATTERN_BLOCK_SIZE*4), matrixCount);
	if (patternCount % (PATTERN_BLOCK_SIZE*4) != 0)
		grid.x +=1;

	kernelPartialsPartialsByPatternBlockCoherentSmall<<<grid, block>>>(partials1, partials2, partials3, matrices1, matrices2, patternCount);
#else
	dim3 grid(patternCount/PATTERN_BLOCK_SIZE, matrixCount);
	if (patternCount % PATTERN_BLOCK_SIZE != 0)
		grid.x += 1;
	dim3 block(PADDED_STATE_COUNT,PATTERN_BLOCK_SIZE);

	kernelPartialsPartialsByPatternBlockCoherent<<<grid, block>>>(partials1, partials2, partials3, matrices1, matrices2, patternCount);
#endif

#ifdef DEBUG
	cudaThreadSynchronize();
	checkCUDAError("PP kernel invocation");
	fprintf(stderr,"Completed GPU PP\n");
#endif

}

__global__ void kernelStatesPartialsByPatternBlockCoherent(int* states1, REAL* partials2,
		REAL* partials3, REAL* matrices1, REAL* matrices2, int totalPatterns) {

	REAL sum1 = 0;
	REAL sum2 = 0;
	int i;

	int state = threadIdx.x;
	int patIdx = threadIdx.y;

	int pattern = __umul24(blockIdx.x,PATTERN_BLOCK_SIZE) + patIdx;
	int matrix = blockIdx.y;
	int patternCount = totalPatterns; // gridDim.x;

	int deltaPartialsByState = __umul24(pattern,PADDED_STATE_COUNT);
	int deltaPartialsByMatrix = __umul24(matrix, __umul24( PADDED_STATE_COUNT, patternCount));

	int x = __umul24(matrix, PADDED_STATE_COUNT * PADDED_STATE_COUNT);

//	REAL *matrix1 = matrices1 + x; // Points to *this* matrix
	REAL *matrix2 = matrices2 + x;

	int y = deltaPartialsByState + deltaPartialsByMatrix;
	int u = state + deltaPartialsByState + deltaPartialsByMatrix;

	// Load values into shared memory
//	__shared__ REAL sMatrix1[BLOCK_PEELING_SIZE][PADDED_STATE_COUNT];
	__shared__ REAL sMatrix2[BLOCK_PEELING_SIZE][PADDED_STATE_COUNT];

//	__shared__ REAL sPartials1[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];
	__shared__ REAL sPartials2[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];

	// copy PADDED_STATE_COUNT*PATTERN_BLOCK_SIZE lengthed partials
	if (pattern < totalPatterns) {
//		sPartials1[patIdx][state] = partials1[y + state]; // These are all coherent global memory reads; checked in Profiler
		sPartials2[patIdx][state] = partials2[y + state];
	} else {
//		sPartials1[patIdx][state] = 0;
		sPartials2[patIdx][state] = 0;
	}

	for (i = 0; i < PADDED_STATE_COUNT; i+=BLOCK_PEELING_SIZE) {
		// load one row of matrices
		if (patIdx < BLOCK_PEELING_SIZE) {
//			sMatrix1[patIdx][state] = matrix1[patIdx*PADDED_STATE_COUNT + state]; // These are all coherent global memory reads.
			sMatrix2[patIdx][state] = matrix2[patIdx*PADDED_STATE_COUNT + state];

			// sMatrix now filled with starting in state and ending in i
//			matrix1 += BLOCK_PEELING_SIZE*PADDED_STATE_COUNT;
			matrix2 += BLOCK_PEELING_SIZE*PADDED_STATE_COUNT;
		}
		__syncthreads();

		// NOT DONE!!! TODO TODO TODO

		int j;
		for(j=0; j<BLOCK_PEELING_SIZE; j++) {
//			sum1 += sMatrix1[j][state] * sPartials1[patIdx][i+j];
			sum2 += sMatrix2[j][state] * sPartials2[patIdx][i+j];
		}

		__syncthreads(); // GTX280 FIX HERE

	}

	if (pattern < totalPatterns)
		partials3[u] = sum1 * sum2;
}

__global__ void kernelStatesStates(INT* states1, INT* states2, REAL* partials3, REAL* matrices1, REAL* matrices2) {

	// blockIdx.y = matrix; blockIdx.x = pattern; threadIdx.x = state
	// blockDim.x = PADDED_STATE_COUNT; gridDim.y = matrixCount; gridDim.x = patternCount;

	int deltaPartialsByPattern = blockIdx.x * PADDED_STATE_COUNT;
	int deltaPartialsByMatrix = blockIdx.y * PADDED_STATE_COUNT * gridDim.x;

	int matrixOffset = blockIdx.y * PADDED_STATE_COUNT * PADDED_STATE_COUNT;

	int u = threadIdx.x + deltaPartialsByPattern + deltaPartialsByMatrix;

	int state1 = (int) states1[blockIdx.x];
	int state2 = (int) states2[blockIdx.x];

	// matrices are in column-form: to0from0, to0from1, ..., to1from0, to1from1, ..., toXfromX
	// to = child, from = this node

	if ( state1 < PADDED_STATE_COUNT && state2 < PADDED_STATE_COUNT) {
		partials3[u] = matrices1[matrixOffset + state1*blockDim.x + threadIdx.x] *
		               matrices2[matrixOffset + state2*blockDim.x + threadIdx.x];
	} else if (state1 < PADDED_STATE_COUNT) {
		partials3[u] = matrices1[matrixOffset + state1*blockDim.x + threadIdx.x];
	} else if (state2 < PADDED_STATE_COUNT) {
		partials3[u] = matrices2[matrixOffset + state2*blockDim.x + threadIdx.x];
	} else {
		partials3[u] = 1.0;
	}

#ifdef KERNEL_PRINT_ENABLED_SS
	printf("GPU u = %d, x = %d, p %1.2e = %1.2e * %1.2e, state1 = %d, state2 = %d\n",u,x,partials3[u],matrices1[x+state1],matrices2[x+state2],state1,state2);
#endif

}

__global__ void kernelIntegratePartials(REAL *partials1, REAL *partials3, REAL *proportions, int matrixCount) {

	// threadIdx.x = state; threadIdx.y = pattern
	// blockDim.x = PADDED_STATE_COUNT; blockDim.y = patternCount

	__shared__ REAL sum[PADDED_STATE_COUNT];

	int u = threadIdx.x +  blockIdx.x * blockDim.x;
	int delta = blockDim.x * gridDim.x;

#ifdef KERNEL_PRINT_ENABLED_2
	printf("blockDim.x = %d, blockDim.y = %d: %d\n",blockDim.x,gridDim.x,u);
#endif

//	partials3[u] = 0; // move to shared memory
	sum[threadIdx.x] = 0;

	for(int r=0; r<matrixCount; r++) {
		int v = u + delta * r;
//		partials3[u]
		sum[threadIdx.x]
		    += partials1[v] * proportions[r]; // move proportions to shared memory
#ifdef KERNEL_PRINT_ENABLED_2
		printf("out %d += in %d * f %d\n",u,v,r);
#endif

	}

	partials3[u] = sum[threadIdx.x];
}

void nativeGPUIntegratePartials(
	REAL* partials1, REAL* proportions, REAL* partials3,
	const unsigned int patternCount, const unsigned int matrixCount) {

	dim3 grid(patternCount);
	dim3 block(PADDED_STATE_COUNT);

	kernelIntegratePartials<<<grid, block>>>(partials1, partials3, proportions, matrixCount);

#ifdef DEBUG
	fprintf(stderr,"Completed GPU IP\n");
#endif

}

void nativeGPUStatesStatesPruningDynamicScaling(
	INT* states1, INT* states2, REAL* partials3, REAL* matrices1, REAL* matrices2, REAL* scalingFactors,
	const unsigned int patternCount, const unsigned int matrixCount, int doRescaling) {

#if (PADDED_STATE_COUNT == 4)
	dim3 block(16,PATTERN_BLOCK_SIZE);
	dim3 grid(patternCount/(PATTERN_BLOCK_SIZE*4), matrixCount);
	if (patternCount % (PATTERN_BLOCK_SIZE*4) != 0)
		grid.x +=1;

	kernelStatesStatesByPatternBlockCoherentSmall<<<grid, block>>>(states1, states2, partials3, matrices1, matrices2, patternCount);

//	dim3 grid(patternCount, matrixCount);
//	dim3 block(PADDED_STATE_COUNT,1);
//
//	kernelStatesStates<<<grid, block>>>(states1, states2, partials3, matrices1, matrices2);
#else
//	dim3 grid(patternCount/PATTERN_BLOCK_SIZE, matrixCount);
//	if (patternCount % PATTERN_BLOCK_SIZE != 0)
//		grid.x += 1;
//	dim3 block(PADDED_STATE_COUNT,PATTERN_BLOCK_SIZE);
//
//	kernelStatesStatesByPatternBlockCoherent<<<grid, block>>>(states1, states2, partials3, matrices1, matrices2, patternCount);

	dim3 grid(patternCount, matrixCount);
	dim3 block(PADDED_STATE_COUNT,1);

	kernelStatesStates<<<grid, block>>>(states1, states2, partials3, matrices1, matrices2);
#endif

	if (doRescaling) {
		if (ones == NULL) {
			ones = (REAL *)malloc(SIZE_REAL*patternCount);
			for(int i=0; i<patternCount; i++)
				ones[i] = 1.0;
		}
		cudaMemcpy(scalingFactors,ones,sizeof(REAL*)*patternCount, cudaMemcpyHostToDevice);
	}

#ifdef DEBUG
	fprintf(stderr,"Completed GPU SS\n");
#endif

}

void nativeGPUStatesPartialsPruningDynamicScaling(
	INT* states1, REAL* partials2, REAL* partials3, REAL* matrices1, REAL* matrices2, REAL *scalingFactors,
	const unsigned int patternCount, const unsigned int matrixCount, int doRescaling) {

#if (PADDED_STATE_COUNT == 4)
	dim3 block(16,PATTERN_BLOCK_SIZE);
	dim3 grid(patternCount/(PATTERN_BLOCK_SIZE*4), matrixCount);
	if (patternCount % (PATTERN_BLOCK_SIZE*4) != 0)
		grid.x +=1;

	kernelStatesPartialsByPatternBlockCoherentSmall<<<grid, block>>>(states1, partials2, partials3, matrices1, matrices2, patternCount);
#else
	dim3 grid(patternCount/PATTERN_BLOCK_SIZE, matrixCount);
	if (patternCount % PATTERN_BLOCK_SIZE != 0)
		grid.x += 1;
	dim3 block(PADDED_STATE_COUNT,PATTERN_BLOCK_SIZE);

	kernelStatesPartialsByPatternBlockCoherent<<<grid, block>>>(states1, partials2, partials3, matrices1, matrices2, patternCount);
#endif

	if (doRescaling) {
		if (ones == NULL) {
			ones = (REAL *)malloc(SIZE_REAL*patternCount);
			for(int i=0; i<patternCount; i++)
				ones[i] = 1.0;
		}
		cudaMemcpy(scalingFactors,ones,sizeof(REAL*)*patternCount, cudaMemcpyHostToDevice);
	}

#ifdef DEBUG
	fprintf(stderr,"Completed GPU SP\n");
#endif

}

__global__ void kernelGPUIntegrateLikelihoods(REAL *dResult, REAL *dRootPartials, REAL *dCategoryProportions, REAL *dFrequencies, int matrixCount) {

	int state   = threadIdx.x;
	int pattern = blockIdx.x;
	int patternCount = gridDim.x;

	__shared__ REAL stateFreq[PADDED_STATE_COUNT];
	__shared__ REAL matrixProp[MATRIX_BLOCK_SIZE]; // TODO Currently assumes MATRIX_BLOCK_SIZE >> matrixCount
	__shared__ REAL sum[PADDED_STATE_COUNT];

	// Load shared memory

	stateFreq[state] = dFrequencies[state];
	sum[state] = 0;

	for(int matrixEdge=0; matrixEdge < matrixCount; matrixEdge += PADDED_STATE_COUNT) {
		int x = matrixEdge + state;
		if (x < matrixCount)
			matrixProp[x] = dCategoryProportions[x];
//		__syncthreads(); // TODO REMOVE???
	}

	__syncthreads();

	int u = state + pattern*PADDED_STATE_COUNT;
	int delta = patternCount*PADDED_STATE_COUNT;;

	for(int r=0; r<matrixCount; r++) {
		sum[state] += dRootPartials[u + delta*r] * matrixProp[r];
	}

	sum[state] *= stateFreq[state];
	__syncthreads();

//	if (state == 0) { // Can parallelize this reduction -- see below
//		REAL final = 0;
//		for(int i=0; i<PADDED_STATE_COUNT; i++) {
//			final += sum[i];
//		}
//		dResult[pattern] = log(final);
//	}

#ifdef IS_POWER_OF_TWO
	for (int i=PADDED_STATE_COUNT/2; i>0; i>>=1) { // parallelized reduction *** only works for powers-of-2 ****
		if (state < i) {
#else
	for (int i=SMALLEST_POWER_OF_TWO/2; i>0; i>>=1) {
		if (state < i && state+i < PADDED_STATE_COUNT ) {
#endif // IS_POWER_OF_TWO
			sum[state] += sum[state+i];
		}
		__syncthreads();
	}

	if (state == 0)
		dResult[pattern] = log(sum[state]);
}

__global__ void kernelGPUComputeRootDynamicScaling(REAL **dNodePtrQueue, REAL *rootScaling, int nodeCount, int patternCount) {

	int pattern = threadIdx.x + blockIdx.x*PATTERN_BLOCK_SIZE;

	REAL total = 0;

//	__shared__ REAL *nodeScales;
	REAL *nodeScales;

	int n;
	for(n=0; n<nodeCount; n++) {
//		if (threadIdx.x == 0) // TODO Why does this not work???
			nodeScales = dNodePtrQueue[n];
//		__syncthreads();

#ifdef KERNEL_PRINT_ENABLED
		if (pattern == 1)
			printf("added %1.2e\n",nodeScales[pattern]);
#endif
		REAL factor = nodeScales[pattern];
		if (factor > 0) // Small check here for 0s in unscaled nodes, TODO Check performance hit (if any)
			total += log(factor);
	}

	if (pattern < patternCount)
		rootScaling[pattern] = total;

}

extern "C" void nativeGPUComputeRootDynamicScaling(REAL **dNodePtrQueue, REAL *dRootScalingFactors, int nodeCount, int patternCount) {

	dim3 grid(patternCount/PATTERN_BLOCK_SIZE);
	if (patternCount % PATTERN_BLOCK_SIZE != 0)
		grid.x +=1;
	dim3 block(PATTERN_BLOCK_SIZE);
	kernelGPUComputeRootDynamicScaling<<<grid,block>>>(dNodePtrQueue, dRootScalingFactors, nodeCount, patternCount);

}

__global__ void kernelGPUIntegrateLikelihoodsDynamicScaling(REAL *dResult, REAL *dRootPartials, REAL *dCategoryProportions, REAL *dFrequencies,
		REAL *dRootScalingFactors, int matrixCount, int nodeCount) {

	int state   = threadIdx.x;
	int pattern = blockIdx.x;
	int patternCount = gridDim.x;

	__shared__ REAL stateFreq[PADDED_STATE_COUNT];
	__shared__ REAL matrixProp[MATRIX_BLOCK_SIZE]; // TODO Currently assumes MATRIX_BLOCK_SIZE >> matrixCount
	__shared__ REAL sum[PADDED_STATE_COUNT];

	// Load shared memory

	stateFreq[state] = dFrequencies[state];
	sum[state] = 0;

	for(int matrixEdge=0; matrixEdge < matrixCount; matrixEdge += PADDED_STATE_COUNT) {
		int x = matrixEdge + state;
		if (x < matrixCount)
			matrixProp[x] = dCategoryProportions[x];
//		__syncthreads(); // TODO REMOVE???
	}

	__syncthreads();

	int u = state + pattern*PADDED_STATE_COUNT;
	int delta = patternCount*PADDED_STATE_COUNT;;

	for(int r=0; r<matrixCount; r++) {
		sum[state] += dRootPartials[u + delta*r] * matrixProp[r];
	}

	sum[state] *= stateFreq[state];
	__syncthreads();

//	if (state == 0) { // Can parallelize this reduction -- see below
//		REAL final = 0;
//		for(int i=0; i<PADDED_STATE_COUNT; i++) {
//			final += sum[i];
//		}
//
//		dResult[pattern] = log(final) + dRootScalingFactors[pattern];
//	}

#ifdef IS_POWER_OF_TWO
	for (int i=PADDED_STATE_COUNT/2; i>0; i>>=1) { // parallelized reduction *** only works for powers-of-2 ****
		if (state < i) {
#else
	for (int i=SMALLEST_POWER_OF_TWO/2; i>0; i>>=1) {
		if (state < i && state+i < PADDED_STATE_COUNT ) {
#endif // IS_POWER_OF_TWO
			sum[state] += sum[state+i];
		}
		__syncthreads();
	}

	if (state == 0)
		dResult[pattern] = log(sum[state]) + dRootScalingFactors[pattern];

}


extern "C" void nativeGPUIntegrateLikelihoodsDynamicScaling(REAL *dResult, REAL *dRootPartials, REAL *dCategoryProportions, REAL *dFrequencies,
		REAL *dRootScalingFactors,
		int patternCount, int matrixCount, int nodeCount) {

#ifdef DEBUG
	fprintf(stderr,"Entering IL\n");
#endif

	dim3 grid(patternCount);
	dim3 block(PADDED_STATE_COUNT);

	kernelGPUIntegrateLikelihoodsDynamicScaling<<<grid, block>>>(dResult, dRootPartials,dCategoryProportions, dFrequencies, dRootScalingFactors, matrixCount, nodeCount);

//	cudaThreadSynchronize();

#ifdef DEBUG
	fprintf(stderr,"Exiting IL\n");
#endif

}

extern "C"
void nativeGPUIntegrateLikelihoods(REAL *dResult, REAL *dRootPartials, REAL *dCategoryProportions, REAL *dFrequencies,
		int patternCount, int matrixCount) {

#ifdef DEBUG
	fprintf(stderr,"Entering IL\n");
#endif

	dim3 grid(patternCount);
	dim3 block(PADDED_STATE_COUNT);

	kernelGPUIntegrateLikelihoods<<<grid, block>>>(dResult, dRootPartials,dCategoryProportions, dFrequencies, matrixCount);

#ifdef DEBUG
	fprintf(stderr,"Exiting IL\n");
#endif

}

#ifdef __cplusplus
}
#endif
#endif
//
