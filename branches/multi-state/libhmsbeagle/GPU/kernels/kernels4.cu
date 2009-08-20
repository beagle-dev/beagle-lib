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

#define PATTERN_BLOCK_SIZE  16

extern "C" {


__global__ void kernelPartialsPartialsNoScale4(REAL* partials1,
                                                                  REAL* partials2,
                                                                  REAL* partials3,
                                                                  REAL* matrices1,
                                                                  REAL* matrices2,
                                                                  int totalPatterns) {
    REAL sum1 = 0;
    REAL sum2 = 0;
    int i;

    int tx = threadIdx.x;
    int state = tx % 4;
    int pat = tx / 4;
    int patIdx = threadIdx.y;
    int matrix = blockIdx.y;
    int patternCount = totalPatterns; // gridDim.x;

    // read 4 patterns at a time, since 4 * 4 = 16
    int pattern = __umul24(blockIdx.x, PATTERN_BLOCK_SIZE * 4) + patIdx * 4 + pat;

//  int deltaPartialsByState = __umul24(pattern, PADDED_STATE_COUNT);
    int deltaPartialsByState = 4 * 4 * (blockIdx.x * PATTERN_BLOCK_SIZE + patIdx);
    int deltaPartialsByMatrix = __umul24(matrix, __umul24( 4, patternCount));

    int x2 = __umul24(matrix, 4 * 4);

    REAL* matrix1 = matrices1 + x2; // Points to *this* matrix
    REAL* matrix2 = matrices2 + x2;

    int y = deltaPartialsByState + deltaPartialsByMatrix;
    int u = tx + deltaPartialsByState + deltaPartialsByMatrix;

#ifdef KERNEL_PRINT_ENABLED
    printf("matrix = %d, pat = %d for tx = %d and state = %d :  u = %d\n", matrix, pattern, tx,
           state, u);
#endif

    // Load values into shared memory
    __shared__ REAL sMatrix1[16];
    __shared__ REAL sMatrix2[16];

    __shared__ REAL sPartials1[PATTERN_BLOCK_SIZE * 4 * 4];
    __shared__ REAL sPartials2[PATTERN_BLOCK_SIZE * 4 * 4];

    // copy 4 * PATTERN_BLOCK_SIZE lengthed partials
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
        sum1 = sMatrix1[i * 4 + state] * sPartials1[patIdx * 16 + pat * 4];
        sum1 += sMatrix1[i * 4 + state] * sPartials1[patIdx * 16 + pat * 4 + 1];
        sum1 += sMatrix1[i * 4 + state] * sPartials1[patIdx * 16 + pat * 4 + 2];
        sum1 += sMatrix1[i * 4 + state] * sPartials1[patIdx * 16 + pat * 4 + 3];
        sum2 = sMatrix2[i * 4 + state] * sPartials2[patIdx * 16 + pat * 4];
        sum2 += sMatrix2[i * 4 + state] * sPartials2[patIdx * 16 + pat * 4 + 1];
        sum2 += sMatrix2[i * 4 + state] * sPartials2[patIdx * 16 + pat * 4 + 2];
        sum2 += sMatrix2[i * 4 + state] * sPartials2[patIdx * 16 + pat * 4 + 3];
        partials3[u] = sum1 * sum2;
    }

}

__global__ void kernelPartialsPartialsFixedScale4(REAL* partials1,
                                                                      REAL* partials2,
                                                                      REAL* partials3,
                                                                      REAL* matrices1,
                                                                      REAL* matrices2,
                                                                      REAL* scalingFactors,
                                                                      int totalPatterns) {
    REAL sum1 = 0;
    REAL sum2 = 0;
    int i;

    int tx = threadIdx.x;
    int state = tx % 4;
    int pat = tx / 4;
    int patIdx = threadIdx.y;
    int matrix = blockIdx.y;
    int patternCount = totalPatterns; // gridDim.x;

    // read 4 patterns at a time, since 4 * 4 = 16
    int pattern = __umul24(blockIdx.x, PATTERN_BLOCK_SIZE * 4) + patIdx * 4 + pat;

//  int deltaPartialsByState = __umul24(pattern,PADDED_STATE_COUNT);
    int deltaPartialsByState = 4 * 4 * (blockIdx.x * PATTERN_BLOCK_SIZE + patIdx);
    int deltaPartialsByMatrix = __umul24(matrix, __umul24( PADDED_STATE_COUNT, patternCount));

    int x2 = __umul24(matrix, 4 * 4);

    REAL* matrix1 = matrices1 + x2; // Points to *this* matrix
    REAL* matrix2 = matrices2 + x2;


    int y = deltaPartialsByState + deltaPartialsByMatrix;
    int u = tx + deltaPartialsByState + deltaPartialsByMatrix;

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

    if (patIdx == 0) {
        sMatrix1[tx] = matrix1[tx]; // All coalesced memory reads
        sMatrix2[tx] = matrix2[tx];
    }

    __syncthreads();

    if (pattern < totalPatterns) { // Remove padded threads!

        sum1 = sMatrix1[i * 4 + state] * sPartials1[patIdx * 16 + pat * 4];
        sum1 += sMatrix1[i * 4 + state] * sPartials1[patIdx * 16 + pat * 4 + 1];
        sum1 += sMatrix1[i * 4 + state] * sPartials1[patIdx * 16 + pat * 4 + 2];
        sum1 += sMatrix1[i * 4 + state] * sPartials1[patIdx * 16 + pat * 4 + 3];
        sum2 = sMatrix2[i * 4 + state] * sPartials2[patIdx * 16 + pat * 4];
        sum2 += sMatrix2[i * 4 + state] * sPartials2[patIdx * 16 + pat * 4 + 1];
        sum2 += sMatrix2[i * 4 + state] * sPartials2[patIdx * 16 + pat * 4 + 2];
        sum2 += sMatrix2[i * 4 + state] * sPartials2[patIdx * 16 + pat * 4 + 3];
        partials3[u] = sum1 * sum2 / fixedScalingFactors[patIdx * 4 + pat];
    }

}

__global__ void kernelStatesPartialsNoScale4(int* states1,
                                                                REAL* partials2,
                                                                REAL* partials3,
                                                                REAL* matrices1,
                                                                REAL* matrices2,
                                                                int totalPatterns) {
    REAL sum1 = 0;
    REAL sum2 = 0;
    int i;

    int tx = threadIdx.x;
    int state = tx % 4;
    int pat = tx / 4;
    int patIdx = threadIdx.y;
    int matrix = blockIdx.y;
    int patternCount = totalPatterns; // gridDim.x;

    // read 4 patterns at a time, since 4 * 4 = 16
    int pattern = __umul24(blockIdx.x, PATTERN_BLOCK_SIZE * 4) + patIdx * 4 + pat;

    int deltaPartialsByState = 4 * 4 * (blockIdx.x * PATTERN_BLOCK_SIZE + patIdx);
    int deltaPartialsByMatrix = __umul24(matrix, __umul24( 4, patternCount));

    int x2 = __umul24(matrix, 4 * 4);

    REAL* matrix1 = matrices1 + x2; // Points to *this* matrix
    REAL* matrix2 = matrices2 + x2;


    int y = deltaPartialsByState + deltaPartialsByMatrix;
    int u = tx + deltaPartialsByState + deltaPartialsByMatrix;

#ifdef KERNEL_PRINT_ENABLED
    printf("matrix = %d, pat = %d for tx = %d and state = %d :  u = %d\n", matrix, pattern, tx,
           state, u);
#endif

    // Load values into shared memory
    __shared__ REAL sMatrix1[16];
    __shared__ REAL sMatrix2[16];

//  __shared__ INT sStates1[PATTERN_BLOCK_SIZE * 4];
    __shared__ REAL sPartials2[PATTERN_BLOCK_SIZE * 4 * 4];

    // copy 4 * PATTERN_BLOCK_SIZE lengthed partials
    if (pattern < totalPatterns) {
        sPartials2[patIdx * 16 + tx] = partials2[y + tx];
//      if (patIdx < PATTERN_BLOCK_SIZE / 4)
//          sStates1[patIdx * 16 + tx] = states1[blockIdx.x * PATTERN_BLOCK_SIZE * 4 + patIdx * 16 +
//                                               tx];
    } else {
        sPartials2[patIdx * 16 + tx] = 0;
//      if (patIdx < PATTERN_BLOCK_SIZE / 4)
//          sStates1[patIdx * 16 + tx] = PADDED_STATE_COUNT;
    }

    if (patIdx == 0) {
        sMatrix1[tx] = matrix1[tx]; // All coalesced memory reads
        sMatrix2[tx] = matrix2[tx];
    }

    __syncthreads();

    if (pattern < totalPatterns) { // Remove padded threads!
//      int state1 = sStates1[patIdx * 4 + pat];
        int state1 = states1[pattern];

        if (state1 < 4)
            sum1 = sMatrix1[state1 * 4 + state];
        else
            sum1 = 1.0;

        sum2 = sMatrix2[i * 4 + state] * sPartials2[patIdx * 16 + pat * 4];
        sum2 += sMatrix2[i * 4 + state] * sPartials2[patIdx * 16 + pat * 4 + 1];
        sum2 += sMatrix2[i * 4 + state] * sPartials2[patIdx * 16 + pat * 4 + 2];
        sum2 += sMatrix2[i * 4 + state] * sPartials2[patIdx * 16 + pat * 4 + 3];
        partials3[u] = sum1 * sum2;
    }

}

__global__ void kernelStatesStatesNoScale4(int* states1,
                                                              int* states2,
                                                              REAL* partials3,
                                                              REAL* matrices1,
                                                              REAL* matrices2,
                                                              int totalPatterns) {

    int tx = threadIdx.x;
    int state = tx % 4;
    int pat = tx / 4;
    int patIdx = threadIdx.y;
    int matrix = blockIdx.y;
    int patternCount = totalPatterns; // gridDim.x;

    // read 4 patterns at a time, since 4 * 4 = 16
    int pattern = __umul24(blockIdx.x, PATTERN_BLOCK_SIZE * 4) + patIdx * 4 + pat;

    int deltaPartialsByState = 4 * 4 * (blockIdx.x * PATTERN_BLOCK_SIZE + patIdx);
    int deltaPartialsByMatrix = __umul24(matrix, __umul24(4, patternCount));

    int x2 = __umul24(matrix, 4 * 4);

    REAL* matrix1 = matrices1 + x2; // Points to *this* matrix
    REAL* matrix2 = matrices2 + x2;

    int u = tx + deltaPartialsByState + deltaPartialsByMatrix;

#ifdef KERNEL_PRINT_ENABLED
    printf("matrix = %d, pat = %d for tx = %d and state = %d :  u = %d\n", matrix, pattern, tx,
           state, u);
#endif

    // Load values into shared memory
    __shared__ REAL sMatrix1[16];
    __shared__ REAL sMatrix2[16];

//  __shared__ INT sStates1[PATTERN_BLOCK_SIZE * 4];
//  __shared__ INT sStates2[PATTERN_BLOCK_SIZE * 4];
//
//  if (pattern < totalPatterns) {
//      if (patIdx < PATTERN_BLOCK_SIZE/4) {
//          sStates1[patIdx * 16 + tx] = states1[blockIdx.x * PATTERN_BLOCK_SIZE*4 + patIdx * 16 +
//                                               tx];
//          sStates2[patIdx * 16 + tx] = states2[blockIdx.x * PATTERN_BLOCK_SIZE*4 + patIdx * 16 +
//                                               tx];
//      } else {
//          sStates1[patIdx * 16 + tx] = PADDED_STATE_COUNT;
//          sStates2[patIdx * 16 + tx] = PADDED_STATE_COUNT;
//      }
//  }

    if (patIdx == 0 ) {
        sMatrix1[tx] = matrix1[tx]; // All coalesced memory reads
        sMatrix2[tx] = matrix2[tx];
    }

    __syncthreads();

    if (pattern < totalPatterns) {
        int state1 = states1[pattern];
        int state2 = states2[pattern];
//      int state1 = sStates1[patIdx * 4 + pat];
//      int state2 = sStates2[patIdx * 4 + pat];

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

} // extern "C"

