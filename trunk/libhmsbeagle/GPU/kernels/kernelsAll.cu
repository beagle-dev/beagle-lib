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

#include "libhmsbeagle/GPU/GPUImplDefs.h"

#define DETERMINE_INDICES() \
    int state = threadIdx.x; \
    int patIdx = threadIdx.y; \
    int pattern = __umul24(blockIdx.x,PATTERN_BLOCK_SIZE) + patIdx; \
    int matrix = blockIdx.y; \
    int patternCount = totalPatterns; \
    int deltaPartialsByState = pattern * PADDED_STATE_COUNT; \
    int deltaPartialsByMatrix = matrix * PADDED_STATE_COUNT * patternCount; \
    int deltaMatrix = matrix * PADDED_STATE_COUNT * PADDED_STATE_COUNT; \
    int u = state + deltaPartialsByState + deltaPartialsByMatrix;

extern "C" {

__global__ void kernelMatrixMulADB(REAL* dMatrices,
                                   unsigned int* listC,
                                   REAL* A,
                                   REAL* D,
                                   REAL* B,
                                   REAL* distanceQueue,
                                   int length,
                                   int wB,
                                   int totalMatrix) {

    __shared__ REAL* C;
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
        C = dMatrices + listC[wMatrix]; // Non-coalescent read
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
            C[PADDED_STATE_COUNT* MULTIPLY_BLOCK_SIZE * by + MULTIPLY_BLOCK_SIZE * bx +
              PADDED_STATE_COUNT * ty + tx] = 0;
        else
            C[PADDED_STATE_COUNT* MULTIPLY_BLOCK_SIZE * by + MULTIPLY_BLOCK_SIZE * bx +
              PADDED_STATE_COUNT * ty + tx] = Csub;
    }
}

__global__ void kernelMatrixMulADBFirstDeriv(REAL* dMatrices,
                                           unsigned int* listC,
                                           REAL* A,
                                           REAL* D,
                                           REAL* B,
                                           REAL* distanceQueue,
                                           int length,
                                           int wB,
                                           int totalMatrix) {

    __shared__ REAL* C;
    __shared__ REAL* CFirstDeriv;
    __shared__ REAL distanceLength;
    __shared__ REAL distanceRate;

    int wMatrix = blockIdx.x % totalMatrix;

    // Block index
    int bx = blockIdx.x / totalMatrix;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int BLOCKS = gridDim.y;

    if (tx == 0 && ty == 0) {
        C = dMatrices + listC[wMatrix];
        CFirstDeriv = dMatrices + listC[wMatrix + totalMatrix];
        distanceLength = distanceQueue[wMatrix]; // Non-coalescent read
        distanceRate = distanceQueue[wMatrix + totalMatrix]; // Non-coalescent read
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
    REAL CFirstDerivSub = 0;

    int a = PADDED_STATE_COUNT * MULTIPLY_BLOCK_SIZE * by;
    int b = MULTIPLY_BLOCK_SIZE * bx;
    int d = 0; //MULTIPLY_BLOCK_SIZE * bx;

    __shared__ REAL As[MULTIPLY_BLOCK_SIZE][MULTIPLY_BLOCK_SIZE];
    __shared__ REAL Bs[MULTIPLY_BLOCK_SIZE][MULTIPLY_BLOCK_SIZE];
    __shared__ REAL Ds[MULTIPLY_BLOCK_SIZE][2];

    for (int i = 0; i < BLOCKS - 1; i++) {

        if (ty == 0) {
            REAL scaledEigenTmp = D[d + tx] * distanceRate;
            Ds[tx][0] = exp(scaledEigenTmp * distanceLength);
            Ds[tx][1] = scaledEigenTmp * Ds[tx][0];
        }

        As[ty][tx] = A[a + PADDED_STATE_COUNT * ty + tx];
        Bs[ty][tx] = B[b + PADDED_STATE_COUNT * ty + tx];

        __syncthreads();

        for (int k = 0; k < MULTIPLY_BLOCK_SIZE; ++k) {
            Csub += As[ty][k] * Ds[k][0] * Bs[k][tx];
            CFirstDerivSub += As[ty][k] * Ds[k][1] * Bs[k][tx];
        }

        __syncthreads();

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

#ifndef KERNEL_PRINT_ENABLED
        __syncthreads();
#endif

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

    __syncthreads();

    for (int k = 0; k < EDGE; k++) {
        Csub += As[ty][k] * Ds[k][0] * Bs[k][tx];
        CFirstDerivSub += As[ty][k] * Ds[k][1] * Bs[k][tx];
    }

    __syncthreads();

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

__global__ void kernelMatrixMulADBSecondDeriv(REAL* dMatrices,
                                           unsigned int* listC,
                                           REAL* A,
                                           REAL* D,
                                           REAL* B,
                                           REAL* distanceQueue,
                                           int length,
                                           int wB,
                                           int totalMatrix) {

    __shared__ REAL* C;
    __shared__ REAL* CFirstDeriv;
    __shared__ REAL* CSecondDeriv;
    __shared__ REAL distanceLength;
    __shared__ REAL distanceRate;

    int wMatrix = blockIdx.x % totalMatrix;

    // Block index
    int bx = blockIdx.x / totalMatrix;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int BLOCKS = gridDim.y;

    if (tx == 0 && ty == 0) {
        C = dMatrices + listC[wMatrix];
        CFirstDeriv = dMatrices + listC[wMatrix + totalMatrix];
        CSecondDeriv = dMatrices + listC[wMatrix + totalMatrix * 2];
        distanceLength = distanceQueue[wMatrix]; // Non-coalescent read
        distanceRate = distanceQueue[wMatrix + totalMatrix]; // Non-coalescent read
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
    REAL CFirstDerivSub = 0;
    REAL CSecondDerivSub = 0;

    int a = PADDED_STATE_COUNT * MULTIPLY_BLOCK_SIZE * by;
    int b = MULTIPLY_BLOCK_SIZE * bx;
    int d = 0; //MULTIPLY_BLOCK_SIZE * bx;

    __shared__ REAL As[MULTIPLY_BLOCK_SIZE][MULTIPLY_BLOCK_SIZE];
    __shared__ REAL Bs[MULTIPLY_BLOCK_SIZE][MULTIPLY_BLOCK_SIZE];
    __shared__ REAL Ds[MULTIPLY_BLOCK_SIZE][3];

    for (int i = 0; i < BLOCKS - 1; i++) {

        if (ty == 0) {
            REAL scaledEigenTmp = D[d + tx] * distanceRate;
            Ds[tx][0] = exp(scaledEigenTmp * distanceLength);
            Ds[tx][1] = scaledEigenTmp * Ds[tx][0];
            Ds[tx][2] = scaledEigenTmp * Ds[tx][1];
        }

        As[ty][tx] = A[a + PADDED_STATE_COUNT * ty + tx];
        Bs[ty][tx] = B[b + PADDED_STATE_COUNT * ty + tx];

        __syncthreads();

        for (int k = 0; k < MULTIPLY_BLOCK_SIZE; ++k) {
            Csub += As[ty][k] * Ds[k][0] * Bs[k][tx];
            CFirstDerivSub += As[ty][k] * Ds[k][1] * Bs[k][tx];
            CSecondDerivSub += As[ty][k] * Ds[k][2] * Bs[k][tx];
        }

        __syncthreads();

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

#ifndef KERNEL_PRINT_ENABLED
        __syncthreads();
#endif

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

    __syncthreads();

    for (int k = 0; k < EDGE; k++) {
        Csub += As[ty][k] * Ds[k][0] * Bs[k][tx];
        CFirstDerivSub += As[ty][k] * Ds[k][1] * Bs[k][tx];
        CSecondDerivSub += As[ty][k] * Ds[k][2] * Bs[k][tx];
    }

    __syncthreads();

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

#define READ_SCHUR_VALUES() \
		if (ty == 0) { \
			Ds[tx] = exp(D[d + tx] * distance); \
			Cs[tx] = D[d + PADDED_STATE_COUNT + tx] * distance; \
			if (Cs[tx]) { \
            	REAL expat = Ds[tx]; \
            	REAL cosbt = cos(Cs[tx]); \
            	Cs[tx] = -expat * sin(Cs[tx]); \
            	Ds[tx] *= cosbt; \
            } \
        }
// end READ_SCHUR_VALUES

#define POPULATE_SCHUR_BAND(limit) \
		if (ty == 0 && tx == 0) { \
			for(int k=0; k<limit; k++) { \
				if (Cs[k] && !Es[k]) { \
					E0[k] = Cs[k]; \
				} else { \
					E0[k] = 0; \
				} \
			} \
		}
// end POPULATE_SCHUR_BAND(limit)

#define DO_MULTIPLICATION(limit) \
		for (int k = 0; k < limit; k++) { \
			Csub += As[ty][k] * ( \
					Ds[k] * B0 [k * MULTIPLY_BLOCK_SIZE + tx] \
				  + E0[k] * Bp1[k * MULTIPLY_BLOCK_SIZE + tx] \
				  - Es[k] * Bm1[k * MULTIPLY_BLOCK_SIZE + tx] \
			); \
		}
// end DO_MULTIPLICATION(limit)

__global__ void kernelMatrixMulADBComplex(REAL* dMatrices,
                                   unsigned int* listC,
                                   REAL* A,
                                   REAL* D,
                                   REAL* B,
                                   REAL* distanceQueue,
                                   int length,
                                   int wB,
                                   int totalMatrix) {

    __shared__ REAL* C;
    __shared__ REAL distance;

    int wMatrix = blockIdx.x % totalMatrix;

    // Block index
    int bx = blockIdx.x / totalMatrix;
    int by = blockIdx.y;
    int BLOCKS = gridDim.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    if (tx == 0 && ty == 0) {
        C = dMatrices + listC[wMatrix];
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
    __shared__ REAL Bs[MULTIPLY_BLOCK_SIZE + 2][MULTIPLY_BLOCK_SIZE];
    __shared__ REAL Cs[MULTIPLY_BLOCK_SIZE];
    __shared__ REAL Ds[MULTIPLY_BLOCK_SIZE];
    __shared__ REAL Es[MULTIPLY_BLOCK_SIZE + 2];

   	REAL* B0  = &Bs[1][0];
   	REAL* Bm1 = &Bs[0][0];
   	REAL* Bp1 = &Bs[2][0];
   	REAL* E0  = &Es[1];

   	// Zero first row of Bs and Es
   	if (ty == 0) {
   		Bs[0][tx] = 0;
   		if (tx == 0) {
   			Es[0] = 0;
   		}
   	}

    while (d + MULTIPLY_BLOCK_SIZE < PADDED_STATE_COUNT) {

        READ_SCHUR_VALUES();

        // Block read A and B sub-matrices
        As[ty][tx] = A[a + PADDED_STATE_COUNT * ty + tx];
        B0[ty * MULTIPLY_BLOCK_SIZE + tx] = B[b + PADDED_STATE_COUNT * ty + tx];

        // Read extra row of B for Bp1
        if (ty == 0) {
        	B0[MULTIPLY_BLOCK_SIZE * MULTIPLY_BLOCK_SIZE + tx] =
        			B[b + PADDED_STATE_COUNT * MULTIPLY_BLOCK_SIZE + tx];
        }

        // All necessary values loaded
    	__syncthreads();

    	POPULATE_SCHUR_BAND(MULTIPLY_BLOCK_SIZE);

    	__syncthreads();

        DO_MULTIPLICATION(MULTIPLY_BLOCK_SIZE);

        // Move last entries in B0 and E0 to first entries in Bs and Es
        if (ty == 0) {
        	Bm1[tx] = Bm1[MULTIPLY_BLOCK_SIZE*MULTIPLY_BLOCK_SIZE + tx];
        	if (tx == 0) {
        		Es[0] = Es[MULTIPLY_BLOCK_SIZE];
        	}
        }

        __syncthreads();

        // Increment sub-matrices
        a += aStep;
        b += bStep;
        d += MULTIPLY_BLOCK_SIZE;

    }

    if (tx < EDGE && ty < EDGE) { // Last block is too long

        READ_SCHUR_VALUES();

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
	__syncthreads();

	POPULATE_SCHUR_BAND(EDGE);

	__syncthreads();

	// Do matrix multiplication
	DO_MULTIPLICATION(EDGE);

    __syncthreads();

    // Write the block sub-matrix to device memory;
    // each thread writes one element

    if (Csub < 0)
    	Csub = 0;

    if ((tx < EDGE || bx < BLOCKS - 1) && (ty < EDGE || by < BLOCKS - 1)) { // It's OK to write
        C[PADDED_STATE_COUNT* MULTIPLY_BLOCK_SIZE * by + MULTIPLY_BLOCK_SIZE * bx +
              PADDED_STATE_COUNT * ty + tx] = Csub;
    }
}

__global__ void kernelSumSites1(REAL* dArray,
                                REAL* dSum,
                                REAL* dPatternWeights,
                                int patternCount) {

    __shared__ REAL sum[SUM_SITES_BLOCK_SIZE];

    int tx = threadIdx.x;
    int pattern = threadIdx.x + blockIdx.x * SUM_SITES_BLOCK_SIZE;
    
    if (pattern < patternCount)
        sum[tx] = dArray[pattern] * dPatternWeights[pattern];
    else
        sum[tx] = 0.0;
        
    __syncthreads();
    
    for (unsigned int s = SUM_SITES_BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tx < s)
            sum[tx] += sum[tx + s];
        __syncthreads();
    }
    
    if (tx == 0)
        dSum[blockIdx.x] = sum[0];
}

__global__ void kernelSumSites2(REAL* dArray1,
                                REAL* dSum1,
                                REAL* dArray2,
                                REAL* dSum2,
                                REAL* dPatternWeights,
                                int patternCount) {

    __shared__ REAL sum1[SUM_SITES_BLOCK_SIZE];
    __shared__ REAL sum2[SUM_SITES_BLOCK_SIZE];

    int tx = threadIdx.x;
    int pattern = threadIdx.x + blockIdx.x * SUM_SITES_BLOCK_SIZE;
    
    if (pattern < patternCount) {
        REAL pWeight = dPatternWeights[pattern];
        sum1[tx] = dArray1[pattern] * pWeight;
        sum2[tx] = dArray2[pattern] * pWeight;
    } else {
        sum1[tx] = 0.0;
        sum2[tx] = 0.0;
    }
        
    __syncthreads();
    
    for (unsigned int s = SUM_SITES_BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tx < s) {
            sum1[tx] += sum1[tx + s];
            sum2[tx] += sum2[tx + s];
        }
        __syncthreads();
    }
    
    if (tx == 0) {
        dSum1[blockIdx.x] = sum1[0];
        dSum2[blockIdx.x] = sum2[0];
    }
}

__global__ void kernelSumSites3(REAL* dArray1,
                                REAL* dSum1,
                                REAL* dArray2,
                                REAL* dSum2,
                                REAL* dArray3,
                                REAL* dSum3,
                                REAL* dPatternWeights,
                                int patternCount) {

    __shared__ REAL sum1[SUM_SITES_BLOCK_SIZE];
    __shared__ REAL sum2[SUM_SITES_BLOCK_SIZE];
    __shared__ REAL sum3[SUM_SITES_BLOCK_SIZE];

    int tx = threadIdx.x;
    int pattern = threadIdx.x + blockIdx.x * SUM_SITES_BLOCK_SIZE;
    
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
        
    __syncthreads();
    
    for (unsigned int s = SUM_SITES_BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tx < s) {
            sum1[tx] += sum1[tx + s];
            sum2[tx] += sum2[tx + s];
            sum3[tx] += sum3[tx + s];
        }
        __syncthreads();
    }
    
    if (tx == 0) {
        dSum1[blockIdx.x] = sum1[0];
        dSum2[blockIdx.x] = sum2[0];
        dSum3[blockIdx.x] = sum3[0];
    }
}


__global__ void kernelAccumulateFactors(REAL* dScalingFactors,
                                        unsigned int* dNodePtrQueue,
                                                   REAL* rootScaling,
                                                   int nodeCount,
                                                   int patternCount) {
    int pattern = threadIdx.x + blockIdx.x * PATTERN_BLOCK_SIZE;

    REAL total = 0;
    REAL* nodeScales;

    int n;
    for(n = 0; n < nodeCount; n++) {
//      if (threadIdx.x == 0) // TODO Why does this not work???
        nodeScales = dScalingFactors + dNodePtrQueue[n];
//      __syncthreads();

#ifdef KERNEL_PRINT_ENABLED
        if (pattern == 1)
            printf("added %1.2e\n", nodeScales[pattern]);
#endif
        REAL factor = nodeScales[pattern];
        if (factor != 1.0) {
            total += log(factor);
        }
    }

    if (pattern < patternCount)
        rootScaling[pattern] += total;
}

__global__ void kernelAccumulateFactorsScalersLog(REAL* dScalingFactors,
                                                 unsigned int* dNodePtrQueue,
                                                 REAL* rootScaling,
                                                 int nodeCount,
                                                 int patternCount) {
    int pattern = threadIdx.x + blockIdx.x * PATTERN_BLOCK_SIZE;

    REAL total = 0;
    REAL* nodeScales;

    int n;
    for(n = 0; n < nodeCount; n++) {
//      if (threadIdx.x == 0) // TODO Why does this not work???
        nodeScales = dScalingFactors + dNodePtrQueue[n];
//      __syncthreads();

#ifdef KERNEL_PRINT_ENABLED
        if (pattern == 1)
            printf("added %1.2e\n", nodeScales[pattern]);
#endif
        total += nodeScales[pattern];
    }

    if (pattern < patternCount)
        rootScaling[pattern] += total;
}

__global__ void kernelAccumulateFactorsAutoScaling(signed char* dScalingFactors,
                                                   unsigned int* dNodePtrQueue,
                                                   int* rootScaling,
                                                   int nodeCount,
                                                   int patternCount,
                                                   int scaleBufferSize) {
    int pattern = threadIdx.x + blockIdx.x * PATTERN_BLOCK_SIZE;
    int index = pattern + blockIdx.y * patternCount;

    int total = 0;
    signed char* nodeScales;

    int n;
    for(n = 0; n < nodeCount; n++) {
//        int sIndex = dNodePtrQueue[n];
        nodeScales = dScalingFactors + dNodePtrQueue[n] * scaleBufferSize;

        total += nodeScales[index];
    }

    if (pattern < patternCount)
        rootScaling[index] = total;
}


__global__ void kernelRemoveFactors(REAL* dScalingFactors,
                                    unsigned int* dNodePtrQueue,
                                                   REAL* rootScaling,
                                                   int nodeCount,
                                                   int patternCount) {
    int pattern = threadIdx.x + blockIdx.x * PATTERN_BLOCK_SIZE;

    REAL total = 0;
    REAL* nodeScales;

    int n;
    for(n = 0; n < nodeCount; n++) {
//      if (threadIdx.x == 0) // TODO Why does this not work???
        nodeScales = dScalingFactors + dNodePtrQueue[n];
//      __syncthreads();

#ifdef KERNEL_PRINT_ENABLED
        if (pattern == 1)
            printf("added %1.2e\n", nodeScales[pattern]);
#endif
        REAL factor = nodeScales[pattern];
        if (factor != 1.0) {
            total += log(factor);
        }
    }

    if (pattern < patternCount)
        rootScaling[pattern] -= total;
} 

__global__ void kernelRemoveFactorsScalersLog(REAL* dScalingFactors,
                                             unsigned int* dNodePtrQueue,
                                             REAL* rootScaling,
                                             int nodeCount,
                                             int patternCount) {
    int pattern = threadIdx.x + blockIdx.x * PATTERN_BLOCK_SIZE;

    REAL total = 0;
    REAL* nodeScales;

    int n;
    for(n = 0; n < nodeCount; n++) {
//      if (threadIdx.x == 0) // TODO Why does this not work???
        nodeScales = dScalingFactors + dNodePtrQueue[n];
//      __syncthreads();

#ifdef KERNEL_PRINT_ENABLED
        if (pattern == 1)
            printf("added %1.2e\n", nodeScales[pattern]);
#endif

        total += nodeScales[pattern];
    }

    if (pattern < patternCount)
        rootScaling[pattern] -= total;
} 

__global__ void kernelPartialsDynamicScalingSlow(REAL* allPartials,
                                                 REAL* scalingFactors,
                                                 int matrixCount) {
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
    for(m = 0; m < matrixCount; m++) {
        partials[state] = allPartials[m * patternCount * PADDED_STATE_COUNT + pattern *
                                      PADDED_STATE_COUNT + state];
        __syncthreads();

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
            __syncthreads();
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


    __syncthreads();

    for(m = 0; m < matrixCount; m++)
        allPartials[m * patternCount * PADDED_STATE_COUNT + pattern * PADDED_STATE_COUNT +
                    state] /= max;

}

__global__ void kernelPartialsDynamicScalingSlowScalersLog(REAL* allPartials,
                                                          REAL* scalingFactors,
                                                          int matrixCount) {
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
    for(m = 0; m < matrixCount; m++) {
        partials[state] = allPartials[m * patternCount * PADDED_STATE_COUNT + pattern *
                                      PADDED_STATE_COUNT + state];
        __syncthreads();

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
            __syncthreads();
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


    __syncthreads();

    for(m = 0; m < matrixCount; m++)
        allPartials[m * patternCount * PADDED_STATE_COUNT + pattern * PADDED_STATE_COUNT +
                    state] /= max;

}


} // extern "C"

