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
#include "libhmsbeagle/GPU/kernels/kernelsAll.cu" // This file includes the non-state-count specific kernels

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

__global__ void kernelPartialsPartialsNoScale(REAL* partials1,
                                                             REAL* partials2,
                                                             REAL* partials3,
                                                             REAL* matrices1,
                                                             REAL* matrices2,
                                                             int totalPatterns) {
    REAL sum1 = 0;
    REAL sum2 = 0;
    int i;

    DETERMINE_INDICES();

    REAL* matrix1 = matrices1 + deltaMatrix; // Points to *this* matrix
    REAL* matrix2 = matrices2 + deltaMatrix;

    int y = deltaPartialsByState + deltaPartialsByMatrix;

    // Load values into shared memory
    __shared__ REAL sMatrix1[BLOCK_PEELING_SIZE][PADDED_STATE_COUNT];
    __shared__ REAL sMatrix2[BLOCK_PEELING_SIZE][PADDED_STATE_COUNT];

    __shared__ REAL sPartials1[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];
    __shared__ REAL sPartials2[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];

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
        __syncthreads();

        int j;
        for(j = 0; j < BLOCK_PEELING_SIZE; j++) {
            sum1 += sMatrix1[j][state] * sPartials1[patIdx][i + j];
            sum2 += sMatrix2[j][state] * sPartials2[patIdx][i + j];
        }

        __syncthreads(); // GTX280 FIX HERE

    }

    if (pattern < totalPatterns)
        partials3[u] = sum1 * sum2;
}

__global__ void kernelPartialsPartialsAutoScale(REAL* partials1,
                                                             REAL* partials2,
                                                             REAL* partials3,
                                                             REAL* matrices1,
                                                             REAL* matrices2,
                                                             signed char* scalingFactors,
                                                             int totalPatterns) {
    REAL sum1 = 0;
    REAL sum2 = 0;
    int i;

    DETERMINE_INDICES();

    REAL* matrix1 = matrices1 + deltaMatrix; // Points to *this* matrix
    REAL* matrix2 = matrices2 + deltaMatrix;

    int y = deltaPartialsByState + deltaPartialsByMatrix;

    // Load values into shared memory
    __shared__ REAL sMatrix1[BLOCK_PEELING_SIZE][PADDED_STATE_COUNT];
    __shared__ REAL sMatrix2[BLOCK_PEELING_SIZE][PADDED_STATE_COUNT];

    __shared__ REAL sPartials1[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];
    __shared__ REAL sPartials2[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];

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
        __syncthreads();

        int j;
        for(j = 0; j < BLOCK_PEELING_SIZE; j++) {
            sum1 += sMatrix1[j][state] * sPartials1[patIdx][i + j];
            sum2 += sMatrix2[j][state] * sPartials2[patIdx][i + j];
        }

        __syncthreads(); // GTX280 FIX HERE

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
        
    __syncthreads();
    
    int scalingActive = sPartials2[patIdx][0];
        
    if (scalingActive) {
        // now using sPartials1 to store max unscaled partials3
        sPartials1[patIdx][state] = tmpPartial;
    }
            
    __syncthreads();
            
    // Unrolled parallel max-reduction
    if (scalingActive && state < 2) {
        REAL compare = sPartials1[patIdx][state + 2];
        if (compare >  sPartials1[patIdx][state])
            sPartials1[patIdx][state] = compare;
    }
    
    __syncthreads();
            
    if (scalingActive && state < 1) {
        REAL maxPartial = sPartials1[patIdx][1];
        if (maxPartial < sPartials1[patIdx][0])
            maxPartial = sPartials1[patIdx][0];
        int expMax;
        frexp(maxPartial, &expMax);
        sPartials1[patIdx][0] = expMax;
    }

    __syncthreads();
    
    if (scalingActive)
        partials3[u] = ldexp(sigTmp, expTmp - sPartials1[patIdx][0]);

    int myIdx = (patIdx * PADDED_STATE_COUNT) + state; // threadId in block
    if ((myIdx < PATTERN_BLOCK_SIZE) && (myIdx + __umul24(blockIdx.x, PATTERN_BLOCK_SIZE) < totalPatterns))
        scalingFactors[(blockIdx.x * PATTERN_BLOCK_SIZE) + (matrix * totalPatterns) + myIdx] = sPartials1[myIdx][0];

}

__global__ void kernelPartialsPartialsFixedScale(REAL* partials1,
                                                                 REAL* partials2,
                                                                 REAL* partials3,
                                                                 REAL* matrices1,
                                                                 REAL* matrices2,
                                                                 REAL* scalingFactors,
                                                                 int totalPatterns) {
    REAL sum1 = 0;
    REAL sum2 = 0;
    int i;

    DETERMINE_INDICES();

    REAL* matrix1 = matrices1 + deltaMatrix; // Points to *this* matrix
    REAL* matrix2 = matrices2 + deltaMatrix;

    int y = deltaPartialsByState + deltaPartialsByMatrix;

    // Load values into shared memory
    __shared__ REAL sMatrix1[BLOCK_PEELING_SIZE][PADDED_STATE_COUNT];
    __shared__ REAL sMatrix2[BLOCK_PEELING_SIZE][PADDED_STATE_COUNT];

    __shared__ REAL sPartials1[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];
    __shared__ REAL sPartials2[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];

    __shared__ REAL fixedScalingFactors[PATTERN_BLOCK_SIZE];

    // copy PADDED_STATE_COUNT*PATTERN_BLOCK_SIZE lengthed partials
    if (pattern < totalPatterns) {
        // These are all coherent global memory reads; checked in Profiler
        sPartials1[patIdx][state] = partials1[y + state];
        sPartials2[patIdx][state] = partials2[y + state];
    } else {
        sPartials1[patIdx][state] = 0;
        sPartials2[patIdx][state] = 0;
    }

    if (patIdx == 0 && state < PATTERN_BLOCK_SIZE )
        // TODO: If PATTERN_BLOCK_SIZE > PADDED_STATE_COUNT, there is a bug here
        fixedScalingFactors[state] = scalingFactors[blockIdx.x * PATTERN_BLOCK_SIZE + state];

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
        __syncthreads();

        int j;
        for(j = 0; j < BLOCK_PEELING_SIZE; j++) {
            sum1 += sMatrix1[j][state] * sPartials1[patIdx][i + j];
            sum2 += sMatrix2[j][state] * sPartials2[patIdx][i + j];
        }

        __syncthreads(); // GTX280 FIX HERE

    }

    if (pattern < totalPatterns)
        partials3[u] = sum1 * sum2 / fixedScalingFactors[patIdx];

}

__global__ void kernelStatesPartialsNoScale(int* states1,
                                                           REAL* partials2,
                                                           REAL* partials3,
                                                           REAL* matrices1,
                                                           REAL* matrices2,
                                                           int totalPatterns) {
    REAL sum1 = 0;
    REAL sum2 = 0;
    int i;

    DETERMINE_INDICES();

    int y = deltaPartialsByState + deltaPartialsByMatrix;

    // Load values into shared memory
    __shared__ REAL sMatrix2[BLOCK_PEELING_SIZE][PADDED_STATE_COUNT];

    __shared__ REAL sPartials2[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];

    // copy PADDED_STATE_COUNT*PATTERN_BLOCK_SIZE lengthed partials
    if (pattern < totalPatterns) {
        sPartials2[patIdx][state] = partials2[y + state];
    } else {
        sPartials2[patIdx][state] = 0;
    }

    REAL* matrix2 = matrices2 + deltaMatrix;

    if (pattern < totalPatterns) {
        int state1 = states1[pattern]; // Coalesced; no need to share

        REAL* matrix1 = matrices1 + deltaMatrix + state1 * PADDED_STATE_COUNT;

        if (state1 < PADDED_STATE_COUNT)
            sum1 = matrix1[state];
        else
            sum1 = 1.0;
    }

    for (i = 0; i < PADDED_STATE_COUNT; i += BLOCK_PEELING_SIZE) {
        // load one row of matrices
        if (patIdx < BLOCK_PEELING_SIZE) {
            sMatrix2[patIdx][state] = matrix2[patIdx * PADDED_STATE_COUNT + state];

            // sMatrix now filled with starting in state and ending in i
            matrix2 += BLOCK_PEELING_SIZE * PADDED_STATE_COUNT;
        }
        __syncthreads();

        int j;
        for(j = 0; j < BLOCK_PEELING_SIZE; j++) {
            sum2 += sMatrix2[j][state] * sPartials2[patIdx][i + j];
        }

        __syncthreads(); // GTX280 FIX HERE

    }

    if (pattern < totalPatterns)
        partials3[u] = sum1 * sum2;
}

__global__ void kernelStatesPartialsFixedScale(int* states1,
                                                               REAL* partials2,
                                                               REAL* partials3,
                                                               REAL* matrices1,
                                                               REAL* matrices2,
                                                               REAL* scalingFactors,
                                                               int totalPatterns) {
    REAL sum1 = 0;
    REAL sum2 = 0;
    int i;

    DETERMINE_INDICES();

    int y = deltaPartialsByState + deltaPartialsByMatrix;

    // Load values into shared memory
    __shared__ REAL sMatrix2[BLOCK_PEELING_SIZE][PADDED_STATE_COUNT];

    __shared__ REAL sPartials2[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];

    __shared__ REAL fixedScalingFactors[PATTERN_BLOCK_SIZE];

    // copy PADDED_STATE_COUNT*PATTERN_BLOCK_SIZE lengthed partials
    if (pattern < totalPatterns) {
        sPartials2[patIdx][state] = partials2[y + state];
    } else {
        sPartials2[patIdx][state] = 0;
    }

    REAL* matrix2 = matrices2 + deltaMatrix;

    if (pattern < totalPatterns) {
        int state1 = states1[pattern]; // Coalesced; no need to share

        REAL* matrix1 = matrices1 + deltaMatrix + state1 * PADDED_STATE_COUNT;

        if (state1 < PADDED_STATE_COUNT)
            sum1 = matrix1[state];
        else
            sum1 = 1.0;
    }

    if (patIdx == 0 && state < PATTERN_BLOCK_SIZE )
        // TODO: If PATTERN_BLOCK_SIZE > PADDED_STATE_COUNT, there is a bug here
        fixedScalingFactors[state] = scalingFactors[blockIdx.x * PATTERN_BLOCK_SIZE + state];

    for (i = 0; i < PADDED_STATE_COUNT; i += BLOCK_PEELING_SIZE) {
        // load one row of matrices
        if (patIdx < BLOCK_PEELING_SIZE) {
            sMatrix2[patIdx][state] = matrix2[patIdx * PADDED_STATE_COUNT + state];

            // sMatrix now filled with starting in state and ending in i
            matrix2 += BLOCK_PEELING_SIZE * PADDED_STATE_COUNT;
        }
        __syncthreads();

        int j;
        for(j = 0; j < BLOCK_PEELING_SIZE; j++) {
            sum2 += sMatrix2[j][state] * sPartials2[patIdx][i + j];
        }

        __syncthreads(); // GTX280 FIX HERE

    }

    if (pattern < totalPatterns)
        partials3[u] = sum1 * sum2 / fixedScalingFactors[patIdx];
}

__global__ void kernelStatesStatesNoScale(int* states1,
                                                         int* states2,
                                                         REAL* partials3,
                                                         REAL* matrices1,
                                                         REAL* matrices2,
                                                         int totalPatterns) {
    DETERMINE_INDICES();

    // Load values into shared memory
//  __shared__ REAL sMatrix1[PADDED_STATE_COUNT];
//  __shared__ REAL sMatrix2[PADDED_STATE_COUNT];

    int state1 = states1[pattern];
    int state2 = states2[pattern];

    // Points to *this* matrix
    REAL* matrix1 = matrices1 + deltaMatrix + state1 * PADDED_STATE_COUNT;
    REAL* matrix2 = matrices2 + deltaMatrix + state2 * PADDED_STATE_COUNT;

//  if (patIdx == 0) {
//      sMatrix1[state] = matrix1[state];
//      sMatrix2[state] = matrix2[state];
//  }

    __syncthreads();

    if (pattern < totalPatterns) {

        if ( state1 < PADDED_STATE_COUNT && state2 < PADDED_STATE_COUNT) {
//          partials3[u] = sMatrix1[state] * sMatrix2[state];
            partials3[u] = matrix1[state] * matrix2[state];
        } else if (state1 < PADDED_STATE_COUNT) {
//          partials3[u] = sMatrix1[state];
            partials3[u] = matrix1[state];
        } else if (state2 < PADDED_STATE_COUNT) {
//          partials3[u] = sMatrix2[state];
            partials3[u] = matrix2[state];
        } else {
            partials3[u] = 1.0;
        }
    }
}

__global__ void kernelStatesStatesFixedScale(int* states1,
                                                             int* states2,
                                                             REAL* partials3,
                                                             REAL* matrices1,
                                                             REAL* matrices2,
                                                             REAL* scalingFactors,
                                                             int totalPatterns) {
    DETERMINE_INDICES();

    // Load values into shared memory
    // Prefetching into shared memory gives no performance gain
    // TODO: Double-check.
//  __shared__ REAL sMatrix1[PADDED_STATE_COUNT];
//  __shared__ REAL sMatrix2[PADDED_STATE_COUNT];

    __shared__ REAL fixedScalingFactors[PATTERN_BLOCK_SIZE];

    int state1 = states1[pattern];
    int state2 = states2[pattern];

    // Points to *this* matrix
    REAL* matrix1 = matrices1 + deltaMatrix + state1 * PADDED_STATE_COUNT;
    REAL* matrix2 = matrices2 + deltaMatrix + state2 * PADDED_STATE_COUNT;

//  if (patIdx == 0) {
//      sMatrix1[state] = matrix1[state];
//      sMatrix2[state] = matrix2[state];
//  }

    // TODO: If PATTERN_BLOCK_SIZE > PADDED_STATE_COUNT, there is a bug here
    if (patIdx == 0 && state < PATTERN_BLOCK_SIZE )
        fixedScalingFactors[state] = scalingFactors[blockIdx.x * PATTERN_BLOCK_SIZE + state];

    __syncthreads();

    if (pattern < totalPatterns) {
        if (state1 < PADDED_STATE_COUNT && state2 < PADDED_STATE_COUNT) {
//          partials3[u] = sMatrix1[state] * sMatrix2[state];
            partials3[u] = matrix1[state] * matrix2[state] / fixedScalingFactors[patIdx];
        } else if (state1 < PADDED_STATE_COUNT) {
//          partials3[u] = sMatrix1[state];
            partials3[u] = matrix1[state] / fixedScalingFactors[patIdx];
        } else if (state2 < PADDED_STATE_COUNT) {
//          partials3[u] = sMatrix2[state];
            partials3[u] = matrix2[state] / fixedScalingFactors[patIdx];
        } else {
            partials3[u] = 1.0 / fixedScalingFactors[patIdx];
        }
    }
}

__global__ void kernelPartialsPartialsEdgeLikelihoods(REAL* dPartialsTmp,
                                                         REAL* dParentPartials,
                                                         REAL* dChildParials,
                                                         REAL* dTransMatrix,
                                                         int patternCount) {
    REAL sum1 = 0;

    int i;

    int state = threadIdx.x;
    int patIdx = threadIdx.y;
    int pattern = __umul24(blockIdx.x,PATTERN_BLOCK_SIZE) + patIdx;
    int matrix = blockIdx.y;
    int totalPatterns = patternCount;
    int deltaPartialsByState = pattern * PADDED_STATE_COUNT;
    int deltaPartialsByMatrix = matrix * PADDED_STATE_COUNT * totalPatterns;
    int deltaMatrix = matrix * PADDED_STATE_COUNT * PADDED_STATE_COUNT;
    int u = state + deltaPartialsByState + deltaPartialsByMatrix;

    REAL* matrix1 = dTransMatrix + deltaMatrix; // Points to *this* matrix

    int y = deltaPartialsByState + deltaPartialsByMatrix;

    // Load values into shared memory
    __shared__ REAL sMatrix1[BLOCK_PEELING_SIZE][PADDED_STATE_COUNT];

    __shared__ REAL sPartials1[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];
    __shared__ REAL sPartials2[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];

    // copy PADDED_STATE_COUNT*PATTERN_BLOCK_SIZE lengthed partials
    if (pattern < totalPatterns) {
        // These are all coherent global memory reads; checked in Profiler
        sPartials1[patIdx][state] = dParentPartials[y + state];
        sPartials2[patIdx][state] = dChildParials[y + state];
    } else {
        sPartials1[patIdx][state] = 0;
        sPartials2[patIdx][state] = 0;
    }

    for (i = 0; i < PADDED_STATE_COUNT; i += BLOCK_PEELING_SIZE) {
        // load one row of matrices
        if (patIdx < BLOCK_PEELING_SIZE) {
            // These are all coherent global memory reads.
            sMatrix1[patIdx][state] = matrix1[patIdx * PADDED_STATE_COUNT + state];

            // sMatrix now filled with starting in state and ending in i
            matrix1 += BLOCK_PEELING_SIZE * PADDED_STATE_COUNT;
        }
        __syncthreads();

        int j;
        for(j = 0; j < BLOCK_PEELING_SIZE; j++) {
            sum1 += sMatrix1[j][state] * sPartials1[patIdx][i + j];
        }

        __syncthreads(); // GTX280 FIX HERE

    }

    if (pattern < totalPatterns)
        dPartialsTmp[u] = sum1 * sPartials2[patIdx][state];
}


__global__ void
__launch_bounds__(PATTERN_BLOCK_SIZE * PADDED_STATE_COUNT)
kernelPartialsPartialsEdgeLikelihoodsSecondDeriv(REAL* dPartialsTmp,
                                                              REAL* dFirstDerivTmp,
                                                              REAL* dSecondDerivTmp,
                                                              REAL* dParentPartials,
                                                              REAL* dChildParials,
                                                              REAL* dTransMatrix,
                                                              REAL* dFirstDerivMatrix,
                                                              REAL* dSecondDerivMatrix,
                                                              int patternCount) {
    REAL sum1 = 0;
    REAL sumFirstDeriv = 0;
    REAL sumSecondDeriv = 0;

    int i;

    int state = threadIdx.x;
    int patIdx = threadIdx.y;
    int pattern = __umul24(blockIdx.x,PATTERN_BLOCK_SIZE) + patIdx;
    int matrix = blockIdx.y;
    int totalPatterns = patternCount;
    int deltaPartialsByState = pattern * PADDED_STATE_COUNT;
    int deltaPartialsByMatrix = matrix * PADDED_STATE_COUNT * totalPatterns;
    int deltaMatrix = matrix * PADDED_STATE_COUNT * PADDED_STATE_COUNT;
    int u = state + deltaPartialsByState + deltaPartialsByMatrix;

    REAL* matrix1 = dTransMatrix + deltaMatrix; // Points to *this* matrix
    REAL* matrixFirstDeriv = dFirstDerivMatrix + deltaMatrix;
    REAL* matrixSecondDeriv = dSecondDerivMatrix + deltaMatrix;

    int y = deltaPartialsByState + deltaPartialsByMatrix;

    // Load values into shared memory
    __shared__ REAL sMatrix1[BLOCK_PEELING_SIZE/2][PADDED_STATE_COUNT];
    __shared__ REAL sMatrixFirstDeriv[BLOCK_PEELING_SIZE/2][PADDED_STATE_COUNT];
    __shared__ REAL sMatrixSecondDeriv[BLOCK_PEELING_SIZE/2][PADDED_STATE_COUNT];

    __shared__ REAL sPartials1[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];
    __shared__ REAL sPartials2[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];

    // copy PADDED_STATE_COUNT*PATTERN_BLOCK_SIZE lengthed partials
    if (pattern < totalPatterns) {
        // These are all coherent global memory reads; checked in Profiler
        sPartials1[patIdx][state] = dParentPartials[y + state];
        sPartials2[patIdx][state] = dChildParials[y + state];
    } else {
        sPartials1[patIdx][state] = 0;
        sPartials2[patIdx][state] = 0;
    }

    for (i = 0; i < PADDED_STATE_COUNT; i += BLOCK_PEELING_SIZE/2) {
        // load one row of matrices
        if (patIdx < BLOCK_PEELING_SIZE/2) {
            // These are all coherent global memory reads.
            sMatrix1[patIdx][state] = matrix1[patIdx * PADDED_STATE_COUNT + state];
	        sMatrixFirstDeriv[patIdx][state] = matrixFirstDeriv[patIdx * PADDED_STATE_COUNT + state];
	        sMatrixSecondDeriv[patIdx][state] = matrixSecondDeriv[patIdx * PADDED_STATE_COUNT + state];

            // sMatrix now filled with starting in state and ending in i
            matrix1 += BLOCK_PEELING_SIZE/2 * PADDED_STATE_COUNT;
            matrixFirstDeriv += BLOCK_PEELING_SIZE/2 * PADDED_STATE_COUNT;
            matrixSecondDeriv += BLOCK_PEELING_SIZE/2 * PADDED_STATE_COUNT;
        }
        __syncthreads();

        int j;
        for(j = 0; j < BLOCK_PEELING_SIZE/2; j++) {
            sum1 += sMatrix1[j][state] * sPartials1[patIdx][i + j];
            sumFirstDeriv += sMatrixFirstDeriv[j][state] * sPartials1[patIdx][i + j];
            sumSecondDeriv += sMatrixSecondDeriv[j][state] * sPartials1[patIdx][i + j];
        }

        __syncthreads(); // GTX280 FIX HERE

    }

    if (pattern < totalPatterns) {
        dPartialsTmp[u] = sum1 * sPartials2[patIdx][state];
        dFirstDerivTmp[u] = sumFirstDeriv * sPartials2[patIdx][state];
        dSecondDerivTmp[u] = sumSecondDeriv * sPartials2[patIdx][state];
    }
}

__global__ void kernelStatesPartialsEdgeLikelihoods(REAL* dPartialsTmp,
                                                    REAL* dParentPartials,
                                                    int* dChildStates,
                                                    REAL* dTransMatrix,
                                                    int patternCount) {
    REAL sum1 = 0;

    int state = threadIdx.x;
    int patIdx = threadIdx.y;
    int pattern = __umul24(blockIdx.x,PATTERN_BLOCK_SIZE) + patIdx;
    int matrix = blockIdx.y;
    int totalPatterns = patternCount;
    int deltaPartialsByState = pattern * PADDED_STATE_COUNT;
    int deltaPartialsByMatrix = matrix * PADDED_STATE_COUNT * patternCount;
    int deltaMatrix = matrix * PADDED_STATE_COUNT * PADDED_STATE_COUNT;
    int u = state + deltaPartialsByState + deltaPartialsByMatrix;

    int y = deltaPartialsByState + deltaPartialsByMatrix;

    // Load values into shared memory
    __shared__ REAL sPartials2[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];

    // copy PADDED_STATE_COUNT*PATTERN_BLOCK_SIZE lengthed partials
    if (pattern < totalPatterns) {
        sPartials2[patIdx][state] = dParentPartials[y + state];
    } else {
        sPartials2[patIdx][state] = 0;
    }

    if (pattern < totalPatterns) {
        int state1 = dChildStates[pattern]; // Coalesced; no need to share

        REAL* matrix1 = dTransMatrix + deltaMatrix + state1 * PADDED_STATE_COUNT;

        if (state1 < PADDED_STATE_COUNT)
            sum1 = matrix1[state];
        else
            sum1 = 1.0;
    }

    if (pattern < totalPatterns)
        dPartialsTmp[u] = sum1 * sPartials2[patIdx][state];                         
}

__global__ void kernelStatesPartialsEdgeLikelihoodsSecondDeriv(REAL* dPartialsTmp,
                                                              REAL* dFirstDerivTmp,
                                                              REAL* dSecondDerivTmp,
                                                              REAL* dParentPartials,
                                                              int* dChildStates,
                                                              REAL* dTransMatrix,
                                                              REAL* dFirstDerivMatrix,
                                                              REAL* dSecondDerivMatrix,
                                                              int patternCount) {
    REAL sum1 = 0;
    REAL sumFirstDeriv = 0;
    REAL sumSecondDeriv = 0;

    int state = threadIdx.x;
    int patIdx = threadIdx.y;
    int pattern = __umul24(blockIdx.x,PATTERN_BLOCK_SIZE) + patIdx;
    int matrix = blockIdx.y;
    int totalPatterns = patternCount;
    int deltaPartialsByState = pattern * PADDED_STATE_COUNT;
    int deltaPartialsByMatrix = matrix * PADDED_STATE_COUNT * patternCount;
    int deltaMatrix = matrix * PADDED_STATE_COUNT * PADDED_STATE_COUNT;
    int u = state + deltaPartialsByState + deltaPartialsByMatrix;

    int y = deltaPartialsByState + deltaPartialsByMatrix;

    // Load values into shared memory
    __shared__ REAL sPartials2[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];

    // copy PADDED_STATE_COUNT*PATTERN_BLOCK_SIZE lengthed partials
    if (pattern < totalPatterns) {
        sPartials2[patIdx][state] = dParentPartials[y + state];
    } else {
        sPartials2[patIdx][state] = 0;
    }

    if (pattern < totalPatterns) {
        int state1 = dChildStates[pattern]; // Coalesced; no need to share

        REAL* matrix1 = dTransMatrix + deltaMatrix + state1 * PADDED_STATE_COUNT;
        REAL* matrixFirstDeriv = dFirstDerivMatrix + deltaMatrix + state1 * PADDED_STATE_COUNT;
        REAL* matrixSecondDeriv = dSecondDerivMatrix + deltaMatrix + state1 * PADDED_STATE_COUNT;

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
}


/*
 * Find a scaling factor for each pattern
 */
__global__ void kernelPartialsDynamicScaling(REAL* allPartials,
                                             REAL* scalingFactors,
                                             int matrixCount) {
    int state = threadIdx.x;
    int matrix = threadIdx.y;
    int pattern = blockIdx.x;
    int patternCount = gridDim.x;

    int deltaPartialsByMatrix = __umul24(matrix, __umul24(PADDED_STATE_COUNT, patternCount));
    
    int offsetPartials = matrix * patternCount * PADDED_STATE_COUNT + pattern * PADDED_STATE_COUNT + state;

    // TODO: Currently assumes MATRIX_BLOCK_SIZE > matrixCount; FIX!!!
    __shared__ REAL partials[MATRIX_BLOCK_SIZE][PADDED_STATE_COUNT];
    __shared__ REAL storedPartials[MATRIX_BLOCK_SIZE][PADDED_STATE_COUNT];

    __shared__ REAL max;

    if (matrix < matrixCount)
        partials[matrix][state] = allPartials[offsetPartials];
    else
        partials[matrix][state] = 0;
        
    storedPartials[matrix][state] = partials[matrix][state];

    __syncthreads();

#ifdef IS_POWER_OF_TWO
    // parallelized reduction *** only works for powers-of-2 ****
    for (int i = PADDED_STATE_COUNT / 2; i > 0; i >>= 1) {
        if (state < i) {
#else
    for (int i = SMALLEST_POWER_OF_TWO / 2; i > 0; i >>= 1) {
        if (state < i && state + i < PADDED_STATE_COUNT ) {
#endif // IS_POWER_OF_TWO
    // parallelized reduction; assumes PADDED_STATE_COUNT is power of 2.
            REAL compare1 = partials[matrix][state];
            REAL compare2 = partials[matrix][state + i];
            if (compare2 > compare1)
            partials[matrix][state] = compare2;
        }
        __syncthreads();
    }

    if (state == 0 && matrix == 0) {
        max = 0;
        int m;
        for(m = 0; m < matrixCount; m++) {
            if (partials[m][0] > max)
                max = partials[m][0];
        }
        
        if (max == 0)
        	max = 1.0;

        scalingFactors[pattern] = max; // TODO: These are incoherent memory writes!!!
    }

    __syncthreads();

    if (matrix < matrixCount)
        allPartials[offsetPartials] ///= max;
                    = storedPartials[matrix][state] / max;

    __syncthreads();
}


/*
 * Find a scaling factor for each pattern
 */
__global__ void kernelPartialsDynamicScalingScalersLog(REAL* allPartials,
                                                      REAL* scalingFactors,
                                                      int matrixCount) {
    int state = threadIdx.x;
    int matrix = threadIdx.y;
    int pattern = blockIdx.x;
    int patternCount = gridDim.x;

    int deltaPartialsByMatrix = __umul24(matrix, __umul24(PADDED_STATE_COUNT, patternCount));
    
    int offsetPartials = matrix * patternCount * PADDED_STATE_COUNT + pattern * PADDED_STATE_COUNT + state;

    // TODO: Currently assumes MATRIX_BLOCK_SIZE > matrixCount; FIX!!!
    __shared__ REAL partials[MATRIX_BLOCK_SIZE][PADDED_STATE_COUNT];
    __shared__ REAL storedPartials[MATRIX_BLOCK_SIZE][PADDED_STATE_COUNT];

    __shared__ REAL max;

    if (matrix < matrixCount)
        partials[matrix][state] = allPartials[offsetPartials];
    else
        partials[matrix][state] = 0;
        
    storedPartials[matrix][state] = partials[matrix][state];

    __syncthreads();

#ifdef IS_POWER_OF_TWO
    // parallelized reduction *** only works for powers-of-2 ****
    for (int i = PADDED_STATE_COUNT / 2; i > 0; i >>= 1) {
        if (state < i) {
#else
    for (int i = SMALLEST_POWER_OF_TWO / 2; i > 0; i >>= 1) {
        if (state < i && state + i < PADDED_STATE_COUNT ) {
#endif // IS_POWER_OF_TWO
    // parallelized reduction; assumes PADDED_STATE_COUNT is power of 2.
            REAL compare1 = partials[matrix][state];
            REAL compare2 = partials[matrix][state + i];
            if (compare2 > compare1)
            partials[matrix][state] = compare2;
        }
        __syncthreads();
    }

    if (state == 0 && matrix == 0) {
        max = 0;
        int m;
        for(m = 0; m < matrixCount; m++) {
            if (partials[m][0] > max)
                max = partials[m][0];
        }
        
        if (max == 0) {
        	max = 1.0;
            scalingFactors[pattern] = 0.0;
        } else {
            scalingFactors[pattern] = log(max);
        }
    }

    __syncthreads();

    if (matrix < matrixCount)
        allPartials[offsetPartials] ///= max;
                    = storedPartials[matrix][state] / max;

    __syncthreads();
}



/*
 * Find a scaling factor for each pattern and accumulate into buffer
 */
__global__ void kernelPartialsDynamicScalingAccumulate(REAL* allPartials,
                                                       REAL* scalingFactors,
                                                       REAL* cumulativeScaling,
                                                       int matrixCount) {
    int state = threadIdx.x;
    int matrix = threadIdx.y;
    int pattern = blockIdx.x;
    int patternCount = gridDim.x;

    int deltaPartialsByMatrix = __umul24(matrix, __umul24(PADDED_STATE_COUNT, patternCount));

    // TODO: Currently assumes MATRIX_BLOCK_SIZE > matrixCount; FIX!!!
    __shared__ REAL partials[MATRIX_BLOCK_SIZE][PADDED_STATE_COUNT];

    __shared__ REAL max;

    if (matrix < matrixCount)
        partials[matrix][state] = allPartials[matrix * patternCount * PADDED_STATE_COUNT + pattern *
                                              PADDED_STATE_COUNT + state];
    else
        partials[matrix][state] = 0;

    __syncthreads();
  
#ifdef IS_POWER_OF_TWO
    // parallelized reduction *** only works for powers-of-2 ****
    for (int i = PADDED_STATE_COUNT / 2; i > 0; i >>= 1) {
        if (state < i) {
#else
    for (int i = SMALLEST_POWER_OF_TWO / 2; i > 0; i >>= 1) {
        if (state < i && state + i < PADDED_STATE_COUNT ) {
#endif // IS_POWER_OF_TWO        
            REAL compare1 = partials[matrix][state];
            REAL compare2 = partials[matrix][state + i];
            if (compare2 > compare1)
            partials[matrix][state] = compare2;
        }
        __syncthreads();
    }

    if (state == 0 && matrix == 0) {
        max = 0;
        int m;
        for(m = 0; m < matrixCount; m++) {
            if (partials[m][0] > max)
                max = partials[m][0];
        }
        
        if (max == 0)
        	max = 1.0;

        scalingFactors[pattern] = max; // TODO: These are incoherent memory writes!!!
        cumulativeScaling[pattern] += log(max);

    }

    __syncthreads();

    if (matrix < matrixCount)
        allPartials[matrix * patternCount * PADDED_STATE_COUNT + pattern * PADDED_STATE_COUNT +
                    state] /= max;

    __syncthreads();
}

/*
 * Find a scaling factor for each pattern and accumulate into buffer
 */
__global__ void kernelPartialsDynamicScalingAccumulateScalersLog(REAL* allPartials,
                                                                REAL* scalingFactors,
                                                                REAL* cumulativeScaling,
                                                                int matrixCount) {
    int state = threadIdx.x;
    int matrix = threadIdx.y;
    int pattern = blockIdx.x;
    int patternCount = gridDim.x;

    int deltaPartialsByMatrix = __umul24(matrix, __umul24(PADDED_STATE_COUNT, patternCount));

    // TODO: Currently assumes MATRIX_BLOCK_SIZE > matrixCount; FIX!!!
    __shared__ REAL partials[MATRIX_BLOCK_SIZE][PADDED_STATE_COUNT];

    __shared__ REAL max;

    if (matrix < matrixCount)
        partials[matrix][state] = allPartials[matrix * patternCount * PADDED_STATE_COUNT + pattern *
                                              PADDED_STATE_COUNT + state];
    else
        partials[matrix][state] = 0;

    __syncthreads();
  
#ifdef IS_POWER_OF_TWO
    // parallelized reduction *** only works for powers-of-2 ****
    for (int i = PADDED_STATE_COUNT / 2; i > 0; i >>= 1) {
        if (state < i) {
#else
    for (int i = SMALLEST_POWER_OF_TWO / 2; i > 0; i >>= 1) {
        if (state < i && state + i < PADDED_STATE_COUNT ) {
#endif // IS_POWER_OF_TWO        
            REAL compare1 = partials[matrix][state];
            REAL compare2 = partials[matrix][state + i];
            if (compare2 > compare1)
            partials[matrix][state] = compare2;
        }
        __syncthreads();
    }

    if (state == 0 && matrix == 0) {
        max = 0;
        int m;
        for(m = 0; m < matrixCount; m++) {
            if (partials[m][0] > max)
                max = partials[m][0];
        }
        
        if (max == 0) {
        	max = 1.0;
            scalingFactors[pattern] = 0.0;
        } else {
            REAL logMax = log(max);
            scalingFactors[pattern] = logMax;
            cumulativeScaling[pattern] += logMax;
        }

    }

    __syncthreads();

    if (matrix < matrixCount)
        allPartials[matrix * patternCount * PADDED_STATE_COUNT + pattern * PADDED_STATE_COUNT +
                    state] /= max;

    __syncthreads();
}


__global__ void kernelIntegrateLikelihoodsFixedScale(REAL* dResult,
                                                            REAL* dRootPartials,
                                                            REAL *dWeights,
                                                            REAL *dFrequencies,
                                                            REAL *dRootScalingFactors,
                                                            int matrixCount,
                                                            int patternCount) {
    int state   = threadIdx.x;
    int pattern = blockIdx.x;
//    int patternCount = gridDim.x;

    __shared__ REAL stateFreq[PADDED_STATE_COUNT];
    // TODO: Currently assumes MATRIX_BLOCK_SIZE >> matrixCount
    __shared__ REAL matrixProp[MATRIX_BLOCK_SIZE];
    __shared__ REAL sum[PADDED_STATE_COUNT];

    // Load shared memory

    stateFreq[state] = dFrequencies[state];
    sum[state] = 0;

    for(int matrixEdge = 0; matrixEdge < matrixCount; matrixEdge += PADDED_STATE_COUNT) {
        int x = matrixEdge + state;
        if (x < matrixCount)
            matrixProp[x] = dWeights[x];
    }

    __syncthreads();

    int u = state + pattern * PADDED_STATE_COUNT;
    int delta = patternCount * PADDED_STATE_COUNT;;

    for(int r = 0; r < matrixCount; r++) {
        sum[state] += dRootPartials[u + delta * r] * matrixProp[r];
    }

    sum[state] *= stateFreq[state];
    __syncthreads();

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
        __syncthreads();
    }

    if (state == 0)
        dResult[pattern] = (log(sum[state]) + dRootScalingFactors[pattern]);
}

__global__ void kernelIntegrateLikelihoodsAutoScaling(REAL* dResult,
                                                     REAL* dRootPartials,
                                                     REAL* dWeights,
                                                     REAL* dFrequencies,
                                                     int* dRootScalingFactors,
                                                     int matrixCount,
                                                     int patternCount) {
    int state   = threadIdx.x;
    int pattern = blockIdx.x;
//    int patternCount = gridDim.x;

    __shared__ REAL stateFreq[PADDED_STATE_COUNT];
    // TODO: Currently assumes MATRIX_BLOCK_SIZE >> matrixCount
    __shared__ REAL matrixProp[MATRIX_BLOCK_SIZE];
    __shared__ REAL matrixScalers[MATRIX_BLOCK_SIZE];
    __shared__ REAL sum[PADDED_STATE_COUNT];

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

    __syncthreads();

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
    __syncthreads();

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
        __syncthreads();
    }

    if (state == 0)
        dResult[pattern] = (log(sum[state]) + (M_LN2 * maxScaleFactor));
}


__global__ void kernelIntegrateLikelihoodsFixedScaleSecondDeriv(REAL* dResult,
                                              REAL* dFirstDerivResult,
                                              REAL* dSecondDerivResult,
                                              REAL* dRootPartials,
                                              REAL* dRootFirstDeriv,
                                              REAL* dRootSecondDeriv,
                                              REAL* dWeights,
                                              REAL* dFrequencies,
                                              REAL *dRootScalingFactors,
                                              int matrixCount,
                                              int patternCount) {
    int state   = threadIdx.x;
    int pattern = blockIdx.x;
//    int patternCount = gridDim.x;

    REAL tmpLogLike = 0.0;
    REAL tmpFirstDeriv = 0.0;

    __shared__ REAL stateFreq[PADDED_STATE_COUNT];
    // TODO: Currently assumes MATRIX_BLOCK_SIZE >> matrixCount
    __shared__ REAL matrixProp[MATRIX_BLOCK_SIZE];
    __shared__ REAL sum[PADDED_STATE_COUNT];
    __shared__ REAL sumD1[PADDED_STATE_COUNT];
    __shared__ REAL sumD2[PADDED_STATE_COUNT];

    // Load shared memory

    stateFreq[state] = dFrequencies[state];
    sum[state] = 0;
    sumD1[state] = 0;
    sumD2[state] = 0;

    for(int matrixEdge = 0; matrixEdge < matrixCount; matrixEdge += PADDED_STATE_COUNT) {
        int x = matrixEdge + state;
        if (x < matrixCount)
            matrixProp[x] = dWeights[x];
    }

    __syncthreads();

    int u = state + pattern * PADDED_STATE_COUNT;
    int delta = patternCount * PADDED_STATE_COUNT;;

    for(int r = 0; r < matrixCount; r++) {
        sum[state] += dRootPartials[u + delta * r] * matrixProp[r];
        sumD1[state] += dRootFirstDeriv[u + delta * r] * matrixProp[r];
        sumD2[state] += dRootSecondDeriv[u + delta * r] * matrixProp[r];
    }

    sum[state] *= stateFreq[state];
    sumD1[state] *= stateFreq[state];
    sumD2[state] *= stateFreq[state];    
    __syncthreads();

#ifdef IS_POWER_OF_TWO
    // parallelized reduction *** only works for powers-of-2 ****
    for (int i = PADDED_STATE_COUNT / 2; i > 0; i >>= 1) {
        if (state < i) {
#else
    for (int i = SMALLEST_POWER_OF_TWO / 2; i > 0; i >>= 1) {
        if (state < i && state + i < PADDED_STATE_COUNT ) {
#endif // IS_POWER_OF_TWO
            sum[state] += sum[state + i];
            sumD1[state] += sumD1[state + i];
            sumD2[state] += sumD2[state + i];
        }
        __syncthreads();
    }

    if (state == 0) {
        tmpLogLike = sum[state];
        dResult[pattern] = (log(tmpLogLike) + dRootScalingFactors[pattern]);
        
        tmpFirstDeriv = sumD1[state] / tmpLogLike;
        dFirstDerivResult[pattern] = tmpFirstDeriv;
        
        dSecondDerivResult[pattern] = (sumD2[state] / tmpLogLike - tmpFirstDeriv * tmpFirstDeriv);
    }
}


__global__ void kernelIntegrateLikelihoods(REAL* dResult,
                                              REAL* dRootPartials,
                                              REAL* dWeights,
                                              REAL* dFrequencies,
                                              int matrixCount,
                                              int patternCount) {
    int state   = threadIdx.x;
    int pattern = blockIdx.x;
//    int patternCount = gridDim.x;

    __shared__ REAL stateFreq[PADDED_STATE_COUNT];
    // TODO: Currently assumes MATRIX_BLOCK_SIZE >> matrixCount
    __shared__ REAL matrixProp[MATRIX_BLOCK_SIZE];
    __shared__ REAL sum[PADDED_STATE_COUNT];

    // Load shared memory

    stateFreq[state] = dFrequencies[state];
    sum[state] = 0;

    for(int matrixEdge = 0; matrixEdge < matrixCount; matrixEdge += PADDED_STATE_COUNT) {
        int x = matrixEdge + state;
        if (x < matrixCount)
            matrixProp[x] = dWeights[x];
    }

    __syncthreads();

    int u = state + pattern * PADDED_STATE_COUNT;
    int delta = patternCount * PADDED_STATE_COUNT;

    for(int r = 0; r < matrixCount; r++) {
        sum[state] += dRootPartials[u + delta * r] * matrixProp[r];
    }

    sum[state] *= stateFreq[state];
    __syncthreads();

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
        __syncthreads();
    }

    if (state == 0)
        dResult[pattern] = log(sum[state]);
}

__global__ void kernelIntegrateLikelihoodsSecondDeriv(REAL* dResult,
                                              REAL* dFirstDerivResult,
                                              REAL* dSecondDerivResult,
                                              REAL* dRootPartials,
                                              REAL* dRootFirstDeriv,
                                              REAL* dRootSecondDeriv,
                                              REAL* dWeights,
                                              REAL* dFrequencies,
                                              int matrixCount,
                                              int patternCount) {
    int state   = threadIdx.x;
    int pattern = blockIdx.x;
//    int patternCount = gridDim.x;

    REAL tmpLogLike = 0.0;
    REAL tmpFirstDeriv = 0.0;

    __shared__ REAL stateFreq[PADDED_STATE_COUNT];
    // TODO: Currently assumes MATRIX_BLOCK_SIZE >> matrixCount
    __shared__ REAL matrixProp[MATRIX_BLOCK_SIZE];
    __shared__ REAL sum[PADDED_STATE_COUNT];
    __shared__ REAL sumD1[PADDED_STATE_COUNT];
    __shared__ REAL sumD2[PADDED_STATE_COUNT];

    // Load shared memory

    stateFreq[state] = dFrequencies[state];
    sum[state] = 0;
    sumD1[state] = 0;
    sumD2[state] = 0;

    for(int matrixEdge = 0; matrixEdge < matrixCount; matrixEdge += PADDED_STATE_COUNT) {
        int x = matrixEdge + state;
        if (x < matrixCount)
            matrixProp[x] = dWeights[x];
    }

    __syncthreads();

    int u = state + pattern * PADDED_STATE_COUNT;
    int delta = patternCount * PADDED_STATE_COUNT;

    for(int r = 0; r < matrixCount; r++) {
        sum[state] += dRootPartials[u + delta * r] * matrixProp[r];
        sumD1[state] += dRootFirstDeriv[u + delta * r] * matrixProp[r];
        sumD2[state] += dRootSecondDeriv[u + delta * r] * matrixProp[r];
    }

    sum[state] *= stateFreq[state];
    sumD1[state] *= stateFreq[state];
    sumD2[state] *= stateFreq[state];
    __syncthreads();

#ifdef IS_POWER_OF_TWO
    // parallelized reduction *** only works for powers-of-2 ****
    for (int i = PADDED_STATE_COUNT / 2; i > 0; i >>= 1) {
        if (state < i) {
#else
    for (int i = SMALLEST_POWER_OF_TWO / 2; i > 0; i >>= 1) {
        if (state < i && state + i < PADDED_STATE_COUNT ) {
#endif // IS_POWER_OF_TWO
            sum[state] += sum[state + i];
            sumD1[state] += sumD1[state + i];
            sumD2[state] += sumD2[state + i];
        }
        __syncthreads();
    }

    if (state == 0) {
        tmpLogLike = sum[state];
        dResult[pattern] = log(tmpLogLike);
        
        tmpFirstDeriv = sumD1[state] / tmpLogLike;
        dFirstDerivResult[pattern] = tmpFirstDeriv;
        
        dSecondDerivResult[pattern] = (sumD2[state] / tmpLogLike - tmpFirstDeriv * tmpFirstDeriv);
    }
}


__global__ void kernelIntegrateLikelihoodsMulti(REAL* dResult,
                                              REAL* dRootPartials,
                                              REAL* dWeights,
                                              REAL* dFrequencies,
                                              int matrixCount,
                                              int patternCount,
											  int takeLog) {
    int state   = threadIdx.x;
    int pattern = blockIdx.x;
//    int patternCount = gridDim.x;

    __shared__ REAL stateFreq[PADDED_STATE_COUNT];
    // TODO: Currently assumes MATRIX_BLOCK_SIZE >> matrixCount
    __shared__ REAL matrixProp[MATRIX_BLOCK_SIZE];
    __shared__ REAL sum[PADDED_STATE_COUNT];

    // Load shared memory

    stateFreq[state] = dFrequencies[state];
    sum[state] = 0;

    for(int matrixEdge = 0; matrixEdge < matrixCount; matrixEdge += PADDED_STATE_COUNT) {
        int x = matrixEdge + state;
        if (x < matrixCount)
            matrixProp[x] = dWeights[x];
    }

    __syncthreads();

    int u = state + pattern * PADDED_STATE_COUNT;
    int delta = patternCount * PADDED_STATE_COUNT;

    for(int r = 0; r < matrixCount; r++) {
        sum[state] += dRootPartials[u + delta * r] * matrixProp[r];
    }

    sum[state] *= stateFreq[state];
    __syncthreads();

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
        __syncthreads();
    }

    if (state == 0) {
		if (takeLog == 0)
			dResult[pattern] = sum[state]; 
		else if (takeLog == 1)
			dResult[pattern] = log(dResult[pattern] + sum[state]);
		else
			dResult[pattern] += sum[state]; 
	}

}

__global__ void kernelIntegrateLikelihoodsFixedScaleMulti(REAL* dResult,
											  REAL* dRootPartials,
                                              REAL* dWeights,
                                              REAL* dFrequencies,
                                              REAL* dScalingFactors,
											  unsigned int* dPtrQueue,
											  REAL* dMaxScalingFactors,
											  REAL* dIndexMaxScalingFactors,
                                              int matrixCount,
                                              int patternCount,
											  int subsetCount,
											  int subsetIndex) {
    int state   = threadIdx.x;
    int pattern = blockIdx.x;
//    int patternCount = gridDim.x;

    __shared__ REAL stateFreq[PADDED_STATE_COUNT];
    // TODO: Currently assumes MATRIX_BLOCK_SIZE >> matrixCount
    __shared__ REAL matrixProp[MATRIX_BLOCK_SIZE];
    __shared__ REAL sum[PADDED_STATE_COUNT];

    // Load shared memory

    stateFreq[state] = dFrequencies[state];
    sum[state] = 0;

    for(int matrixEdge = 0; matrixEdge < matrixCount; matrixEdge += PADDED_STATE_COUNT) {
        int x = matrixEdge + state;
        if (x < matrixCount)
            matrixProp[x] = dWeights[x];
    }

    __syncthreads();

    int u = state + pattern * PADDED_STATE_COUNT;
    int delta = patternCount * PADDED_STATE_COUNT;

    for(int r = 0; r < matrixCount; r++) {
        sum[state] += dRootPartials[u + delta * r] * matrixProp[r];
    }

    sum[state] *= stateFreq[state];
    __syncthreads();

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
        __syncthreads();
    }
	
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
}


} // extern "C"

