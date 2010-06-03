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

__global__ void kernelMatrixMulADB(REAL** listC,
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
        C = (REAL*) *((int*)listC + wMatrix); // Non-coalescent read
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

__global__ void kernelMatrixMulADBFirstDeriv(REAL** listC,
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
        C = (REAL*) *((int*)listC + wMatrix); // Non-coalescent read
        CFirstDeriv = (REAL*) *((int*)listC + wMatrix + totalMatrix); // Non-coalescent read
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

__global__ void kernelMatrixMulADBSecondDeriv(REAL** listC,
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
        C = (REAL*) *((int*)listC + wMatrix); // Non-coalescent read
        CFirstDeriv = (REAL*) *((int*)listC + wMatrix + totalMatrix); // Non-coalescent read
        CSecondDeriv = (REAL*) *((int*)listC + wMatrix + totalMatrix * 2); // Non-coalescent read
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

__global__ void kernelMatrixMulADBComplex(REAL** listC,
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
        C = (REAL*) *((int*)listC + wMatrix); // Non-coalescent read
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

    for (int i = 0; i < BLOCKS - 1; i++) {

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

__global__ void kernelAccumulateFactors(REAL** dNodePtrQueue,
                                                   REAL* rootScaling,
                                                   int nodeCount,
                                                   int patternCount) {
    int pattern = threadIdx.x + blockIdx.x * PATTERN_BLOCK_SIZE;

    REAL total = 0;
    REAL* nodeScales;

    int n;
    for(n = 0; n < nodeCount; n++) {
//      if (threadIdx.x == 0) // TODO Why does this not work???
            nodeScales = (REAL*) *((int*)dNodePtrQueue + n);
//      __syncthreads();

#ifdef KERNEL_PRINT_ENABLED
        if (pattern == 1)
            printf("added %1.2e\n", nodeScales[pattern]);
#endif
        REAL factor = nodeScales[pattern];
        if (factor != 1.0) {
#ifdef LSCALER
            total += factor;
#else
            total += log(factor);
#endif
        }
    }

    if (pattern < patternCount)
        rootScaling[pattern] += total;
}

__global__ void kernelRemoveFactors(REAL** dNodePtrQueue,
                                                   REAL* rootScaling,
                                                   int nodeCount,
                                                   int patternCount) {
    int pattern = threadIdx.x + blockIdx.x * PATTERN_BLOCK_SIZE;

    REAL total = 0;
    REAL* nodeScales;

    int n;
    for(n = 0; n < nodeCount; n++) {
//      if (threadIdx.x == 0) // TODO Why does this not work???
            nodeScales = (REAL*) *((int*)dNodePtrQueue + n);
//      __syncthreads();

#ifdef KERNEL_PRINT_ENABLED
        if (pattern == 1)
            printf("added %1.2e\n", nodeScales[pattern]);
#endif
        REAL factor = nodeScales[pattern];
        if (factor != 1.0) {
#ifdef LSCALER
            total += factor;
#else
            total += log(factor);
#endif
        }
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
#ifdef LSCALER
        scalingFactors[pattern] = log(max);
#else
        scalingFactors[pattern] = max;
#endif
    }


    __syncthreads();

    for(m = 0; m < matrixCount; m++)
        allPartials[m * patternCount * PADDED_STATE_COUNT + pattern * PADDED_STATE_COUNT +
                    state] /= max;

}

} // extern "C"

