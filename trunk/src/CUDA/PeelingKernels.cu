/*
 * @author Marc Suchard
 * @author Daniel Ayres
 */
#ifndef _Included_PeelingKernels
#define _Included_PeelingKernels

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

__global__ void kernelPartialsPartialsByPatternBlockCoherent(REAL* partials1,
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

__global__ void kernelPartialsPartialsByPatternBlockFixedScaling(REAL* partials1,
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

__global__ void kernelStatesPartialsByPatternBlockCoherent(int* states1,
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

__global__ void kernelStatesPartialsByPatternBlockFixedScaling(int* states1,
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

__global__ void kernelStatesStatesByPatternBlockCoherent(int* states1,
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

__global__ void kernelStatesStatesByPatternBlockFixedScaling(int* states1,
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

#if (PADDED_STATE_COUNT == 4)
__global__ void kernelPartialsPartialsByPatternBlockCoherentSmall(REAL* partials1,
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
    int deltaPartialsByMatrix = __umul24(matrix, __umul24( PADDED_STATE_COUNT, patternCount));

    int x2 = __umul24(matrix, PADDED_STATE_COUNT * PADDED_STATE_COUNT);

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
        for(i = 0; i < PADDED_STATE_COUNT; i++) {
            sum1 += sMatrix1[i * 4 + state] * sPartials1[patIdx * 16 + pat * 4 + i];
            sum2 += sMatrix2[i * 4 + state] * sPartials2[patIdx * 16 + pat * 4 + i];
        }
        partials3[u] = sum1 * sum2;
    }

}

__global__ void kernelPartialsPartialsByPatternBlockSmallFixedScaling(REAL* partials1,
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

    int x2 = __umul24(matrix, PADDED_STATE_COUNT * PADDED_STATE_COUNT);

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
        for(i = 0; i < PADDED_STATE_COUNT; i++) {
            sum1 += sMatrix1[i * 4 + state] * sPartials1[patIdx * 16 + pat * 4 + i];
            sum2 += sMatrix2[i * 4 + state] * sPartials2[patIdx * 16 + pat * 4 + i];
        }
        partials3[u] = sum1 * sum2 / fixedScalingFactors[patIdx * 4 + pat];
    }

}

__global__ void kernelStatesPartialsByPatternBlockCoherentSmall(int* states1,
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
    int deltaPartialsByMatrix = __umul24(matrix, __umul24( PADDED_STATE_COUNT, patternCount));

    int x2 = __umul24(matrix, PADDED_STATE_COUNT * PADDED_STATE_COUNT);

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

    // copy PADDED_STATE_COUNT * PATTERN_BLOCK_SIZE lengthed partials
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

        if (state1 < PADDED_STATE_COUNT)
            sum1 = sMatrix1[state1 * 4 + state];
        else
            sum1 = 1.0;

        for(i=0; i<PADDED_STATE_COUNT; i++) {
            sum2 += sMatrix2[i * 4 + state] * sPartials2[patIdx * 16 + pat * 4 + i];
        }
        partials3[u] = sum1 * sum2;
    }

}

__global__ void kernelStatesStatesByPatternBlockCoherentSmall(int* states1,
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
    int deltaPartialsByMatrix = __umul24(matrix, __umul24(PADDED_STATE_COUNT, patternCount));

    int x2 = __umul24(matrix, PADDED_STATE_COUNT * PADDED_STATE_COUNT);

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
#endif // PADDED_STATE_COUNT == 4

__global__ void kernelGPUIntegrateLikelihoodsDynamicScaling(REAL* dResult,
                                                            REAL* dRootPartials,
                                                            REAL *dCategoryProportions,
                                                            REAL *dFrequencies,
                                                            REAL *dRootScalingFactors,
                                                            int matrixCount,
                                                            int nodeCount) {
    int state   = threadIdx.x;
    int pattern = blockIdx.x;
    int patternCount = gridDim.x;

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
            matrixProp[x] = dCategoryProportions[x];
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
        dResult[pattern] = log(sum[state]) + dRootScalingFactors[pattern];
}

__global__ void kernelGPUComputeRootDynamicScaling(REAL** dNodePtrQueue,
                                                   REAL *rootScaling,
                                                   int nodeCount,
                                                   int patternCount) {
    int pattern = threadIdx.x + blockIdx.x * PATTERN_BLOCK_SIZE;

    REAL total = 0;
    REAL* nodeScales;

    int n;
    for(n = 0; n < nodeCount; n++) {
//      if (threadIdx.x == 0) // TODO Why does this not work???
            nodeScales = dNodePtrQueue[n];
//      __syncthreads();

#ifdef KERNEL_PRINT_ENABLED
        if (pattern == 1)
            printf("added %1.2e\n", nodeScales[pattern]);
#endif
        REAL factor = nodeScales[pattern];
        if (factor != 1.0)
            total += log(factor);
    }

    if (pattern < patternCount)
        rootScaling[pattern] = total;
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

    // TODO: Currently assumes MATRIX_BLOCK_SIZE > matrixCount; FIX!!!
    __shared__ REAL partials[MATRIX_BLOCK_SIZE][PADDED_STATE_COUNT];

    __shared__ REAL max;

    if (matrix < matrixCount)
        partials[matrix][state] = allPartials[matrix * patternCount * PADDED_STATE_COUNT + pattern *
                                              PADDED_STATE_COUNT + state];
    else
        partials[matrix][state] = 0;

    __syncthreads();

    int i;
    // parallelized reduction; assumes PADDED_STATE_COUNT is power of 2.
    for (i = PADDED_STATE_COUNT / 2; i > 0; i >>= 1) {
        if (state < i) {
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

        scalingFactors[pattern] = max; // TODO: These are incoherent memory writes!!!
    }

    __syncthreads();

    if (matrix < matrixCount)
        allPartials[matrix * patternCount * PADDED_STATE_COUNT + pattern * PADDED_STATE_COUNT +
                    state] /= max;

    __syncthreads();
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

    if(state == 0)
        scalingFactors[pattern] = max;

    __syncthreads();

    for(m = 0; m < matrixCount; m++)
        allPartials[m * patternCount * PADDED_STATE_COUNT + pattern * PADDED_STATE_COUNT +
                    state] /= max;

}

__global__ void kernelGPUIntegrateLikelihoods(REAL* dResult,
                                              REAL* dRootPartials,
                                              REAL* dCategoryProportions,
                                              REAL* dFrequencies,
                                              int matrixCount) {
    int state   = threadIdx.x;
    int pattern = blockIdx.x;
    int patternCount = gridDim.x;

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
            matrixProp[x] = dCategoryProportions[x];
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

#if (PADDED_STATE_COUNT == 4)
__global__ void kernelPartialsPartialsGPUEdgeLikelihoodsSmall(REAL* dResult,
                                                              REAL* dPartialsTmp,
                                                              REAL* dParentPartials,
                                                              REAL* dChildParials,
                                                              REAL* dTransMatrix,
                                                              REAL* dWeights,
                                                              REAL* dFrequencies,
                                                              int patternCount,
                                                              int count) {
    
    REAL sum1 = 0;
//    REAL sum2 = 0;
    int i;

    int tx = threadIdx.x;
    int state = tx % 4;
    int pat = tx / 4;
    int patIdx = threadIdx.y;
    int matrix = blockIdx.y;
    int totalPatterns = patternCount; // gridDim.x;

    // read 4 patterns at a time, since 4 * 4 = 16 
    int pattern = __umul24(blockIdx.x, PATTERN_BLOCK_SIZE * 4) + patIdx * 4 + pat;

//  int deltaPartialsByState = __umul24(pattern, PADDED_STATE_COUNT);
    int deltaPartialsByState = 4 * 4 * (blockIdx.x * PATTERN_BLOCK_SIZE + patIdx);
    int deltaPartialsByMatrix = __umul24(matrix, __umul24( PADDED_STATE_COUNT, totalPatterns));

    int x2 = __umul24(matrix, PADDED_STATE_COUNT * PADDED_STATE_COUNT);

    REAL* matrix1 = dTransMatrix + x2; // Points to *this* matrix
//    REAL* matrix2 = matrices2 + x2;

    int y = deltaPartialsByState + deltaPartialsByMatrix;
    int u = tx + deltaPartialsByState + deltaPartialsByMatrix;

#ifdef KERNEL_PRINT_ENABLED
    printf("matrix = %d, pat = %d for tx = %d and state = %d :  u = %d\n", matrix, pattern, tx,
           state, u);
#endif

    // Load values into shared memory
    __shared__ REAL sMatrix1[16];
//    __shared__ REAL sMatrix2[16];

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
//        sMatrix2[tx] = matrix2[tx];
    }

    __syncthreads();

    if (pattern < totalPatterns) { // Remove padded threads!
        for(i = 0; i < PADDED_STATE_COUNT; i++) {
            sum1 += sMatrix1[i * 4 + state] * sPartials1[patIdx * 16 + pat * 4 + i];
//            sum2 += sMatrix2[i * 4 + state] * sPartials2[patIdx * 16 + pat * 4 + i];
        }
        dPartialsTmp[u] = sum1 *  sPartials2[patIdx * 16 + pat * 4 + state];
    }    

}
#endif // PADDED_STATE_COUNT == 4

#endif
