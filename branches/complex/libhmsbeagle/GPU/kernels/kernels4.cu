#define STATE_COUNT 4

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

#include "libhmsbeagle/GPU/GPUImplDefs.h"
#include "libhmsbeagle/GPU/kernels/kernelsAll.cu" // This file includes the non-state-count specific kernels

#define DETERMINE_INDICES_4() \
    int tx = threadIdx.x; \
    int state = tx & 0x3; \
    int pat = tx >> 2; \
    int patIdx = threadIdx.y; \
    int matrix = blockIdx.y; \
    int pattern = __umul24(blockIdx.x, PATTERN_BLOCK_SIZE * 4) + patIdx * 4 + pat; \
    int deltaPartialsByState = 4 * 4 * (blockIdx.x * PATTERN_BLOCK_SIZE + patIdx); \
    int deltaPartialsByMatrix = __umul24(matrix, __umul24( PADDED_STATE_COUNT, totalPatterns)); \
    int x2 = __umul24(matrix, PADDED_STATE_COUNT * PADDED_STATE_COUNT); \
    int u = tx + deltaPartialsByState + deltaPartialsByMatrix;

extern "C" {

__global__ void kernelPartialsPartialsNoScale(REAL* partials1,
                                                                  REAL* partials2,
                                                                  REAL* partials3,
                                                                  REAL* matrices1,
                                                                  REAL* matrices2,
                                                                  int totalPatterns) {
		REAL sum1;
	    REAL sum2;
	    int i;

	    DETERMINE_INDICES_4();
	    int y = deltaPartialsByState + deltaPartialsByMatrix;
	    
	    REAL* matrix1 = matrices1 + x2; // Points to *this* matrix
	    REAL* matrix2 = matrices2 + x2;

	#ifdef KERNEL_PRINT_ENABLED
	    printf("matrix = %d, pat = %d for tx = %d and state = %d :  u = %d\n", matrix, pattern, tx,
	           state, u);
	#endif

	    // Load values into shared memory
	    __shared__ REAL sMatrix1[16];
	    __shared__ REAL sMatrix2[16];

	    __shared__ REAL sPartials1[PATTERN_BLOCK_SIZE * 4 * 4];
	    __shared__ REAL sPartials2[PATTERN_BLOCK_SIZE * 4 * 4];

	    // copy PADDED_STATE_COUNT * PATTERN_BLOCK_SIZE lengthed partials
	    if (pattern < totalPatterns) {
	        sPartials1[patIdx * 16 + tx] = partials1[y + tx]; // All coalesced memory reads
	        sPartials2[patIdx * 16 + tx] = partials2[y + tx];
	    } else {
	        sPartials1[patIdx * 16 + tx] = 0;
	        sPartials2[patIdx * 16 + tx] = 0;
	    }

	    if (patIdx == 0 ) {
	        sMatrix1[tx] = matrix1[tx]; // All coalesced memory reads
	        sMatrix2[tx] = matrix2[tx];
	    }

	    __syncthreads();

	    if (pattern < totalPatterns) { // Remove padded threads!

 	        i = pat;                       
	        sum1  = sMatrix1[i * 4 + state] * sPartials1[patIdx * 16 + pat * 4 + i];
	        sum2  = sMatrix2[i * 4  + state] * sPartials2[patIdx * 16 + pat * 4 + i];

	        i = (++i) & 0x3;
	        sum1 += sMatrix1[i * 4 + state] * sPartials1[patIdx * 16 + pat * 4 + i];
	        sum2 += sMatrix2[i * 4 + state] * sPartials2[patIdx * 16 + pat * 4 + i];

	        i = (++i) & 0x3;
	        sum1 += sMatrix1[i * 4 + state] * sPartials1[patIdx * 16 + pat * 4 + i];
	        sum2 += sMatrix2[i * 4 + state] * sPartials2[patIdx * 16 + pat * 4 + i];

	        i = (++i) & 0x3;
	        sum1 += sMatrix1[i * 4 + state] * sPartials1[patIdx * 16 + pat * 4 + i];
	        sum2 += sMatrix2[i * 4 + state] * sPartials2[patIdx * 16 + pat * 4 + i];

	        partials3[u] = sum1 * sum2;
	    }

	}

__global__ void kernelPartialsPartialsFixedScale(REAL* partials1,
                                                                      REAL* partials2,
                                                                      REAL* partials3,
                                                                      REAL* matrices1,
                                                                      REAL* matrices2,
                                                                      REAL* scalingFactors,
                                                                      int totalPatterns) {
    REAL sum1;
    REAL sum2;
    int i;

    DETERMINE_INDICES_4();
    int y = deltaPartialsByState + deltaPartialsByMatrix;
    REAL* matrix1 = matrices1 + x2; // Points to *this* matrix
    REAL* matrix2 = matrices2 + x2;

#ifdef KERNEL_PRINT_ENABLED
    printf("matrix = %d, pat = %d for tx = %d and state = %d :  u = %d\n", matrix, pattern, tx,
           state, u);
#endif

    // Load values into shared memory
    __shared__ REAL sMatrix1[16];
    __shared__ REAL sMatrix2[16];

    __shared__ REAL sPartials1[PATTERN_BLOCK_SIZE * 4 * 4];
    __shared__ REAL sPartials2[PATTERN_BLOCK_SIZE * 4 * 4];

    __shared__ REAL fixedScalingFactors[PATTERN_BLOCK_SIZE * 4];

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
            scalingFactors[blockIdx.x * PATTERN_BLOCK_SIZE * 4 + patIdx * PATTERN_BLOCK_SIZE + tx];

    if (patIdx == 0 ) {
        sMatrix1[tx] = matrix1[tx]; // All coalesced memory reads
        sMatrix2[tx] = matrix2[tx];
    }

    __syncthreads();

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
        
        partials3[u] = sum1 * sum2 / fixedScalingFactors[patIdx * 4 + pat];
    }

}


__global__ void kernelStatesPartialsNoScale(int* states1,
                                                                REAL* partials2,
                                                                REAL* partials3,
                                                                REAL* matrices1,
                                                                REAL* matrices2,
                                                                int totalPatterns) {
    REAL sum1 = 1;
    REAL sum2;
    int i;

    DETERMINE_INDICES_4();
    int y = deltaPartialsByState + deltaPartialsByMatrix;
    REAL* matrix1 = matrices1 + x2; // Points to *this* matrix
    REAL* matrix2 = matrices2 + x2;


#ifdef KERNEL_PRINT_ENABLED
    printf("matrix = %d, pat = %d for tx = %d and state = %d :  u = %d\n", matrix, pattern, tx,
           state, u);
#endif

    // Load values into shared memory
    __shared__ REAL sMatrix1[16];
    __shared__ REAL sMatrix2[16];

    __shared__ REAL sPartials2[PATTERN_BLOCK_SIZE * 4 * 4];

    // copy PADDED_STATE_COUNT * PATTERN_BLOCK_SIZE lengthed partials
    if (pattern < totalPatterns) {
        sPartials2[patIdx * 16 + tx] = partials2[y + tx];
    } else {
        sPartials2[patIdx * 16 + tx] = 0;
    }

    if (patIdx == 0) {
        sMatrix1[tx] = matrix1[tx]; // All coalesced memory reads
        sMatrix2[tx] = matrix2[tx];
    }

    __syncthreads();

    if (pattern < totalPatterns) { // Remove padded threads!

        int state1 = states1[pattern];

        if (state1 < PADDED_STATE_COUNT)
            sum1 = sMatrix1[state1 * 4 + state];

        i = pat;
        sum2  = sMatrix2[i * 4 + state] * sPartials2[patIdx * 16 + pat * 4 + i];
        i = (++i) & 0x3;
        sum2 += sMatrix2[i * 4 + state] * sPartials2[patIdx * 16 + pat * 4 + i];
        i = (++i) & 0x3;
        sum2 += sMatrix2[i * 4 + state] * sPartials2[patIdx * 16 + pat * 4 + i];
        i = (++i) & 0x3;
        sum2 += sMatrix2[i * 4 + state] * sPartials2[patIdx * 16 + pat * 4 + i];
        partials3[u] = sum1 * sum2;
    }

}


__global__ void kernelStatesStatesNoScale(int* states1,
                                                              int* states2,
                                                              REAL* partials3,
                                                              REAL* matrices1,
                                                              REAL* matrices2,
                                                              int totalPatterns) {

	DETERMINE_INDICES_4();
    REAL* matrix1 = matrices1 + x2; // Points to *this* matrix
    REAL* matrix2 = matrices2 + x2;
    
#ifdef KERNEL_PRINT_ENABLED
    printf("matrix = %d, pat = %d for tx = %d and state = %d :  u = %d\n", matrix, pattern, tx,
           state, u);
#endif

    // Load values into shared memory
    __shared__ REAL sMatrix1[16];
    __shared__ REAL sMatrix2[16];

    if (patIdx == 0 ) {
        sMatrix1[tx] = matrix1[tx]; // All coalesced memory reads
        sMatrix2[tx] = matrix2[tx];
    }

    __syncthreads();

    if (pattern < totalPatterns) {
        int state1 = states1[pattern];
        int state2 = states2[pattern];

        if ( state1 < PADDED_STATE_COUNT && state2 < PADDED_STATE_COUNT) {
            partials3[u] = sMatrix1[state1 * 4 + state] * sMatrix2[state2 * 4 + state];
        } else if (state1 < PADDED_STATE_COUNT) {
            partials3[u] = sMatrix1[state1 * 4 + state];
        } else if (state2 < PADDED_STATE_COUNT) {
            partials3[u] = sMatrix2[state2 * 4 + state];
        } else {
            partials3[u] = 1.0;
        }
    }
}

__global__ void kernelPartialsPartialsEdgeLikelihoods(REAL* dPartialsTmp,
                                                              REAL* dParentPartials,
                                                              REAL* dChildParials,
                                                              REAL* dTransMatrix,
                                                              int totalPatterns) {
	   REAL sum1 = 0;

	    int i;

	    DETERMINE_INDICES_4();
	    int y = deltaPartialsByState + deltaPartialsByMatrix;
	    REAL* matrix1 = dTransMatrix + x2; // Points to *this* matrix

	#ifdef KERNEL_PRINT_ENABLED
	    printf("matrix = %d, pat = %d for tx = %d and state = %d :  u = %d\n", matrix, pattern, tx,
	           state, u);
	#endif

	    // Load values into shared memory
	    __shared__ REAL sMatrix1[16];

	    __shared__ REAL sPartials1[PATTERN_BLOCK_SIZE * 4 * 4];
	    __shared__ REAL sPartials2[PATTERN_BLOCK_SIZE * 4 * 4];

	    // copy PADDED_STATE_COUNT * PATTERN_BLOCK_SIZE lengthed partials
	    if (pattern < totalPatterns) {
	        sPartials1[patIdx * 16 + tx] = dParentPartials[y + tx]; // All coalesced memory reads
	        sPartials2[patIdx * 16 + tx] = dChildParials[y + tx];
	    } else {
	        sPartials1[patIdx * 16 + tx] = 0;
	        sPartials2[patIdx * 16 + tx] = 0;
	    }

	    if (patIdx == 0 ) {
	        sMatrix1[tx] = matrix1[tx]; // All coalesced memory reads
	    }

	    __syncthreads();

	    if (pattern < totalPatterns) { // Remove padded threads!
//	        for(i = 0; i < PADDED_STATE_COUNT; i++) {
//	            sum1 += sMatrix1[i * 4 + state] * sPartials1[patIdx * 16 + pat * 4 + i];
//	        }
	        i = pat;
	        sum1  = sMatrix1[i * 4 + state] * sPartials1[patIdx * 16 + pat * 4 + i];
	        i = (++i) & 0x3;
	        sum1 += sMatrix1[i * 4 + state] * sPartials1[patIdx * 16 + pat * 4 + i];
	        i = (++i) & 0x3;
	        sum1 += sMatrix1[i * 4 + state] * sPartials1[patIdx * 16 + pat * 4 + i];
	        i = (++i) & 0x3;
	        sum1 += sMatrix1[i * 4 + state] * sPartials1[patIdx * 16 + pat * 4 + i];
	        
	        dPartialsTmp[u] = sum1 * sPartials2[patIdx * 16 + pat * 4 + state];
	    }    

	}

__global__ void kernelStatesPartialsEdgeLikelihoods(REAL* dPartialsTmp,
                                                         REAL* dParentPartials,
                                                         int* dChildStates,
                                                         REAL* dTransMatrix,
                                                         int totalPatterns) {
    REAL sum1 = 0;

    DETERMINE_INDICES_4();
    int y = deltaPartialsByState + deltaPartialsByMatrix;
    REAL* matrix1 = dTransMatrix + x2; // Points to *this* matrix
    
#ifdef KERNEL_PRINT_ENABLED
    printf("matrix = %d, pat = %d for tx = %d and state = %d :  u = %d\n", matrix, pattern, tx,
           state, u);
#endif

    // Load values into shared memory
    __shared__ REAL sMatrix1[16];

    __shared__ REAL sPartials2[PATTERN_BLOCK_SIZE * 4 * 4];

    // copy PADDED_STATE_COUNT * PATTERN_BLOCK_SIZE lengthed partials
    if (pattern < totalPatterns) {
        sPartials2[patIdx * 16 + tx] = dParentPartials[y + tx];
    } else {
        sPartials2[patIdx * 16 + tx] = 0;
    }

    if (patIdx == 0) {
        sMatrix1[tx] = matrix1[tx]; // All coalesced memory reads
    }

    __syncthreads();

    if (pattern < totalPatterns) { // Remove padded threads!
        int state1 = dChildStates[pattern];

        if (state1 < PADDED_STATE_COUNT)
            sum1 = sMatrix1[state1 * 4 + state];
        else
            sum1 = 1.0;

        dPartialsTmp[u] = sum1 * sPartials2[patIdx * 16 + pat * 4 + state];
    }
}

/*
 * Find a scaling factor for each pattern
 */
__global__ void kernelPartialsDynamicScaling(REAL* allPartials,
                                             REAL* scalingFactors,
                                             int matrixCount) {
                                             
    int tx = threadIdx.x;
    
    int state = tx & 0x3;
    int pat = tx >> 2;
                             
    int patIdx = blockIdx.x;
    
    int pattern = (patIdx << 2) + pat;
    int matrix = threadIdx.y;
    // TODO: Assumes matrixCount < MATRIX_BLOCK_SIZ
    
    // Patterns are always padded, so no reading/writing past end possible
    // Find start of patternBlock for thread-block
    int partialsOffset = (matrix * gridDim.x + patIdx) << 4; //* 16;

    __shared__ REAL partials[MATRIX_BLOCK_SIZE][16]; // 4 patterns at a time
    __shared__ REAL storedPartials[MATRIX_BLOCK_SIZE][16];

    __shared__ REAL matrixMax[4];
    
    if (matrix < matrixCount)
        partials[matrix][tx] = allPartials[partialsOffset + tx];          

    storedPartials[matrix][tx] = partials[matrix][tx];
           
    __syncthreads();
    
    // Unrolled parallel max-reduction
    if (state < 2) {
        REAL compare1 = partials[matrix][tx];
        REAL compare2 = partials[matrix][tx + 2];
        if (compare2 > compare1)
            partials[matrix][tx] = compare2;
    }
    __syncthreads();
    
    if (state < 1) {
        REAL compare1 = partials[matrix][tx];
        REAL compare2 = partials[matrix][tx + 1];
        if (compare2 > compare1)
            partials[matrix][tx] = compare2;
    }
    __syncthreads();
 
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
   
#ifdef LSCALER
        scalingFactors[pattern] = log(matrixMax[pat]);
#else
        scalingFactors[pattern] = matrixMax[pat]; // TODO: Are these incoherent writes?
#endif
    }

    // Attempt at a parallel reduction that (1) does not work and (2) is slower
//    if (state == 0) {    
//        for (int i = MATRIX_BLOCK_SIZE / 2; i > 0; i >>= 1) {
//            if (matrix < i) {
//                REAL compare1 = partials[matrix][tx];
//                REAL compare2 = partials[matrix+i][tx];
//                if (compare2 > compare1)
//                    partials[matrix][tx] = compare2;              
//            }
//            __syncthreads();
//        }         
//        
//        if (matrix == 0) {
//            matrixMax[pat] = partials[matrix][tx];
//            if (matrixMax[pat] == 0)
//                matrixMax[pat] = 1.0;
//                
//            scalingFactors[pattern] = matrixMax[pat];
//        }
//    }

    __syncthreads();

    if (matrix < matrixCount)
        allPartials[partialsOffset + tx] = storedPartials[matrix][tx] / matrixMax[pat];
}


/*
 * Find a scaling factor for each pattern and accumulate into buffer
 */
__global__ void kernelPartialsDynamicScalingAccumulate(REAL* allPartials,
                                                       REAL* scalingFactors,
                                                       REAL* cumulativeScaling,
                                                       int matrixCount) {
    int tx = threadIdx.x;
    
    int state = tx & 0x3;
    int pat = tx >> 2;
                             
    int patIdx = blockIdx.x;
    
    int pattern = (patIdx << 2) + pat;
    int matrix = threadIdx.y;
    // TODO: Assumes matrixCount < MATRIX_BLOCK_SIZ
    
    // Patterns are always padded, so no reading/writing past end possible
    // Find start of patternBlock for thread-block
    int partialsOffset = (matrix * gridDim.x + patIdx) << 4; //* 16;

    __shared__ REAL partials[MATRIX_BLOCK_SIZE][16]; // 4 patterns at a time
    __shared__ REAL storedPartials[MATRIX_BLOCK_SIZE][16];

    __shared__ REAL matrixMax[4];
    
    if (matrix < matrixCount)
        partials[matrix][tx] = allPartials[partialsOffset + tx];          

    storedPartials[matrix][tx] = partials[matrix][tx];
           
    __syncthreads();
    
    // Unrolled parallel max-reduction
    if (state < 2) {
        REAL compare1 = partials[matrix][tx];
        REAL compare2 = partials[matrix][tx + 2];
        if (compare2 > compare1)
            partials[matrix][tx] = compare2;
    }
    __syncthreads();
    
    if (state < 1) {
        REAL compare1 = partials[matrix][tx];
        REAL compare2 = partials[matrix][tx + 1];
        if (compare2 > compare1)
            partials[matrix][tx] = compare2;
    }
    __syncthreads();
 
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
   
#ifdef LSCALER
        REAL logMax = log(matrixMax[pat]);
        scalingFactors[pattern] = logMax;
        cumulativeScaling[pattern] += logMax; // TODO: Fix, this is both a read and write
#else
        scalingFactors[pattern] = matrixMax[pat]; 
        cumulativeScaling[pattern] += log(matrixMax[pat]);
#endif
    }

    __syncthreads();

    if (matrix < matrixCount)
        allPartials[partialsOffset + tx] = storedPartials[matrix][tx] / matrixMax[pat];
        
}

#define LIKE_PATTERN_BLOCK_SIZE PATTERN_BLOCK_SIZE

__global__ void kernelIntegrateLikelihoodsFixedScale(REAL* dResult,
                                                     REAL* dRootPartials,
                                                     REAL *dWeights,
                                                     REAL *dFrequencies,
                                                     REAL *dRootScalingFactors,
                                                     int matrixCount,
                                                     int patternCount) {
    int state   = threadIdx.x;
    int pat = threadIdx.y;
    int pattern = blockIdx.x * LIKE_PATTERN_BLOCK_SIZE + threadIdx.y;
    
    __shared__ REAL stateFreq[4];
    
    // TODO: Currently assumes MATRIX_BLOCK_SIZE >= matrixCount
    __shared__ REAL matrixProp[MATRIX_BLOCK_SIZE];
    __shared__ REAL sum[LIKE_PATTERN_BLOCK_SIZE][4];

    // Load shared memory

    if (pat == 0) {
        stateFreq[state] = dFrequencies[state];
    }
    
    sum[pat][state] = 0;
    
    // TODO: Assumes matrixCount < LIKE_PATTERN_BLOCK_SIZE * 4
    if (pat * LIKE_PATTERN_BLOCK_SIZE + state < matrixCount) {
        matrixProp[pat * LIKE_PATTERN_BLOCK_SIZE + state] = dWeights[pat * 4 + state];
    }

    __syncthreads();

    int u = state + pattern * PADDED_STATE_COUNT;
    int delta = patternCount * PADDED_STATE_COUNT;;

    for(int r = 0; r < matrixCount; r++) {
        sum[pat][state] += dRootPartials[u + delta * r] * matrixProp[r];
    }

    sum[pat][state] *= stateFreq[state];
        
    if (state < 2)
        sum[pat][state] += sum[pat][state + 2];
    __syncthreads();
    if (state < 1) {
        sum[pat][state] += sum[pat][state + 1];
    }
    __syncthreads();
    
    if (state == 0)
        dResult[pattern] = log(sum[pat][state]) + dRootScalingFactors[pattern];
}

__global__ void kernelIntegrateLikelihoods(REAL* dResult,
                                              REAL* dRootPartials,
                                              REAL* dWeights,
                                              REAL* dFrequencies,
                                              int matrixCount,
                                              int patternCount) {
    int state   = threadIdx.x;
    int pat = threadIdx.y;
    int pattern = blockIdx.x * LIKE_PATTERN_BLOCK_SIZE + threadIdx.y;
    
    __shared__ REAL stateFreq[4];
    
    // TODO: Currently assumes MATRIX_BLOCK_SIZE >= matrixCount
    __shared__ REAL matrixProp[MATRIX_BLOCK_SIZE];
    __shared__ REAL sum[LIKE_PATTERN_BLOCK_SIZE][4];

    // Load shared memory

    if (pat == 0) {
        stateFreq[state] = dFrequencies[state];
    }
    
    sum[pat][state] = 0;
    
    // TODO: Assumes matrixCount < LIKE_PATTERN_BLOCK_SIZE * 4
    if (pat * LIKE_PATTERN_BLOCK_SIZE + state < matrixCount) {
        matrixProp[pat * LIKE_PATTERN_BLOCK_SIZE + state] = dWeights[pat * 4 + state];
    }

    __syncthreads();

    int u = state + pattern * PADDED_STATE_COUNT;
    int delta = patternCount * PADDED_STATE_COUNT;;

    for(int r = 0; r < matrixCount; r++) {
        sum[pat][state] += dRootPartials[u + delta * r] * matrixProp[r];
    }

    sum[pat][state] *= stateFreq[state];
        
    if (state < 2)
        sum[pat][state] += sum[pat][state + 2];
    __syncthreads();
    if (state < 1) {
        sum[pat][state] += sum[pat][state + 1];
    }
    __syncthreads();
    
    if (state == 0)
        dResult[pattern] = log(sum[pat][state]);
        
}

} // extern "C"

