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

} // extern "C"

