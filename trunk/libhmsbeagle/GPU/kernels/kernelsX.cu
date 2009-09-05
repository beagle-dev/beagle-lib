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

} // extern "C"

