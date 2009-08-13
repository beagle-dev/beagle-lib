/*
 * @author Marc Suchard
 * @author Daniel Ayres
 */
 
#define STATE_COUNT 4
 
#include "libhmsbeagle/GPU/GPUImplDefs.h" 

__kernel void kernelMatrixMulADB(__global float* listC,
                                   __global float* A,
                                   __global float* D,
                                   __global float* B,
                                   __global float* distanceQueue,
                                   int length,
                                   int wB,
                                   int index) {

//    __global float* C;
    __local float distance;

    int totalMatrix = 1;
    
    int wMatrix = get_group_id(0) % totalMatrix;

    // Block index
    int bx = get_group_id(0) / totalMatrix;
    int by = get_group_id(1);

    // Thread index
    int tx = get_local_id(0);
    int ty = get_local_id(1);
    int BLOCKS = get_num_groups(1);

    if (tx == 0 && ty == 0) {
//        C = listC[wMatrix]; // Non-coalescent read
        distance = distanceQueue[index]; // Non-coalescent read
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const int EDGE = PADDED_STATE_COUNT - (BLOCKS - 1) * MULTIPLY_BLOCK_SIZE;

    // Step size used to iterate through the sub-matrices of A
    int aStep = MULTIPLY_BLOCK_SIZE;

    // Step size used to iterate through the sub-matrices of B
    int bStep = MULTIPLY_BLOCK_SIZE * PADDED_STATE_COUNT;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    int a = PADDED_STATE_COUNT * MULTIPLY_BLOCK_SIZE * by;
    int b = MULTIPLY_BLOCK_SIZE * bx;
    int d = 0; //MULTIPLY_BLOCK_SIZE * bx;

    __local float As[MULTIPLY_BLOCK_SIZE][MULTIPLY_BLOCK_SIZE];
    __local float Bs[MULTIPLY_BLOCK_SIZE][MULTIPLY_BLOCK_SIZE];
    __local float Ds[MULTIPLY_BLOCK_SIZE];

    for (int i = 0; i < BLOCKS - 1; i++) {

        if (ty == 0)
            Ds[tx] = exp(D[d + tx] * distance);

        As[ty][tx] = A[a + PADDED_STATE_COUNT * ty + tx];
        Bs[ty][tx] = B[b + PADDED_STATE_COUNT * ty + tx];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < MULTIPLY_BLOCK_SIZE; ++k)
            Csub += As[ty][k] * Ds[k] * Bs[k][tx];

        barrier(CLK_LOCAL_MEM_FENCE);

        a += aStep;
        b += bStep;
        d += MULTIPLY_BLOCK_SIZE;
    }

    // Last block is too long
    if (tx < EDGE && ty < EDGE) {
        if (ty == 0)
            Ds[tx] = exp(D[d + tx] * distance);

        As[ty][tx] = A[a + PADDED_STATE_COUNT * ty + tx];
        Bs[ty][tx] = B[b + PADDED_STATE_COUNT * ty + tx];

    } else {

        if (ty == 0)
            Ds[tx] = 0;

        As[ty][tx] = 0;
        Bs[ty][tx] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k = 0; k < EDGE; k++)
        Csub += As[ty][k] * Ds[k] * Bs[k][tx];

    barrier(CLK_LOCAL_MEM_FENCE);

    // Write the block sub-matrix to device memory;
    // each thread writes one element

    if ((tx < EDGE || bx < BLOCKS - 1) && (ty < EDGE || by < BLOCKS - 1)) { // It's OK to write
        if (Csub < 0)
            listC[PADDED_STATE_COUNT* MULTIPLY_BLOCK_SIZE * by + MULTIPLY_BLOCK_SIZE * bx +
              PADDED_STATE_COUNT * ty + tx] = 0;
        else
            listC[PADDED_STATE_COUNT* MULTIPLY_BLOCK_SIZE * by + MULTIPLY_BLOCK_SIZE * bx +
              PADDED_STATE_COUNT * ty + tx] = Csub;
    }
}

__kernel void kernelPartialsPartialsByPatternBlockCoherent(__global float* partials1,
                                                             __global float* partials2,
                                                             __global float* partials3,
                                                             __global float* matrices1,
                                                             __global float* matrices2,
                                                             int totalPatterns) {
    float sum1 = 0;
    float sum2 = 0;
    int i;

    int state = get_local_id(0);
    int patIdx = get_local_id(1);
    int pattern = (get_group_id(0) * PATTERN_BLOCK_SIZE) + patIdx;
    int matrix = get_group_id(1);
    int patternCount = totalPatterns;
    int deltaPartialsByState = pattern * PADDED_STATE_COUNT;
    int deltaPartialsByMatrix = matrix * PADDED_STATE_COUNT * patternCount;
    int deltaMatrix = matrix * PADDED_STATE_COUNT * PADDED_STATE_COUNT;
    int u = state + deltaPartialsByState + deltaPartialsByMatrix;

    __global float* matrix1 = matrices1 + deltaMatrix; // Points to *this* matrix
    __global float* matrix2 = matrices2 + deltaMatrix;

    int y = deltaPartialsByState + deltaPartialsByMatrix;

    // Load values into shared memory
    __local float sMatrix1[BLOCK_PEELING_SIZE][PADDED_STATE_COUNT];
    __local float sMatrix2[BLOCK_PEELING_SIZE][PADDED_STATE_COUNT];

    __local float sPartials1[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];
    __local float sPartials2[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];

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
        barrier(CLK_LOCAL_MEM_FENCE);

        int j;
        for(j = 0; j < BLOCK_PEELING_SIZE; j++) {
            sum1 += sMatrix1[j][state] * sPartials1[patIdx][i + j];
            sum2 += sMatrix2[j][state] * sPartials2[patIdx][i + j];
        }

        barrier(CLK_LOCAL_MEM_FENCE); // GTX280 FIX HERE

    }

    if (pattern < totalPatterns)
        partials3[u] = sum1 * sum2;
}

__kernel void kernelPartialsPartialsByPatternBlockFixedScaling(__global float* partials1,
                                                                 __global float* partials2,
                                                                 __global float* partials3,
                                                                 __global float* matrices1,
                                                                 __global float* matrices2,
                                                                 __global float* scalingFactors,
                                                                 int totalPatterns) {
    float sum1 = 0;
    float sum2 = 0;
    int i;

    int state = get_local_id(0);
    int patIdx = get_local_id(1);
    int pattern = (get_group_id(0) * PATTERN_BLOCK_SIZE) + patIdx;
    int matrix = get_group_id(1);
    int patternCount = totalPatterns;
    int deltaPartialsByState = pattern * PADDED_STATE_COUNT;
    int deltaPartialsByMatrix = matrix * PADDED_STATE_COUNT * patternCount;
    int deltaMatrix = matrix * PADDED_STATE_COUNT * PADDED_STATE_COUNT;
    int u = state + deltaPartialsByState + deltaPartialsByMatrix;

    __global float* matrix1 = matrices1 + deltaMatrix; // Points to *this* matrix
    __global float* matrix2 = matrices2 + deltaMatrix;

    int y = deltaPartialsByState + deltaPartialsByMatrix;

    // Load values into shared memory
    __local float sMatrix1[BLOCK_PEELING_SIZE][PADDED_STATE_COUNT];
    __local float sMatrix2[BLOCK_PEELING_SIZE][PADDED_STATE_COUNT];

    __local float sPartials1[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];
    __local float sPartials2[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];

    __local float fixedScalingFactors[PATTERN_BLOCK_SIZE];

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
        fixedScalingFactors[state] = scalingFactors[get_group_id(0) * PATTERN_BLOCK_SIZE + state];

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
        barrier(CLK_LOCAL_MEM_FENCE);

        int j;
        for(j = 0; j < BLOCK_PEELING_SIZE; j++) {
            sum1 += sMatrix1[j][state] * sPartials1[patIdx][i + j];
            sum2 += sMatrix2[j][state] * sPartials2[patIdx][i + j];
        }

        barrier(CLK_LOCAL_MEM_FENCE); // GTX280 FIX HERE

    }

    if (pattern < totalPatterns)
        partials3[u] = sum1 * sum2 / fixedScalingFactors[patIdx];

}

__kernel void kernelStatesPartialsByPatternBlockCoherent(__global int* states1,
                                                           __global float* partials2,
                                                           __global float* partials3,
                                                           __global float* matrices1,
                                                           __global float* matrices2,
                                                           int totalPatterns) {
    float sum1 = 0;
    float sum2 = 0;
    int i;

    int state = get_local_id(0);
    int patIdx = get_local_id(1);
    int pattern = (get_group_id(0) * PATTERN_BLOCK_SIZE) + patIdx;
    int matrix = get_group_id(1);
    int patternCount = totalPatterns;
    int deltaPartialsByState = pattern * PADDED_STATE_COUNT;
    int deltaPartialsByMatrix = matrix * PADDED_STATE_COUNT * patternCount;
    int deltaMatrix = matrix * PADDED_STATE_COUNT * PADDED_STATE_COUNT;
    int u = state + deltaPartialsByState + deltaPartialsByMatrix;

    int y = deltaPartialsByState + deltaPartialsByMatrix;

    // Load values into shared memory
    __local float sMatrix2[BLOCK_PEELING_SIZE][PADDED_STATE_COUNT];

    __local float sPartials2[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];

    // copy PADDED_STATE_COUNT*PATTERN_BLOCK_SIZE lengthed partials
    if (pattern < totalPatterns) {
        sPartials2[patIdx][state] = partials2[y + state];
    } else {
        sPartials2[patIdx][state] = 0;
    }

    __global float* matrix2 = matrices2 + deltaMatrix;

    if (pattern < totalPatterns) {
        int state1 = states1[pattern]; // Coalesced; no need to share

        __global float* matrix1 = matrices1 + deltaMatrix + state1 * PADDED_STATE_COUNT;

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
        barrier(CLK_LOCAL_MEM_FENCE);

        int j;
        for(j = 0; j < BLOCK_PEELING_SIZE; j++) {
            sum2 += sMatrix2[j][state] * sPartials2[patIdx][i + j];
        }

        barrier(CLK_LOCAL_MEM_FENCE); // GTX280 FIX HERE

    }

    if (pattern < totalPatterns)
        partials3[u] = sum1 * sum2;
}

__kernel void kernelStatesPartialsByPatternBlockFixedScaling(__global int* states1,
                                                               __global float* partials2,
                                                               __global float* partials3,
                                                               __global float* matrices1,
                                                               __global float* matrices2,
                                                               __global float* scalingFactors,
                                                               int totalPatterns) {
    float sum1 = 0;
    float sum2 = 0;
    int i;

    int state = get_local_id(0);
    int patIdx = get_local_id(1);
    int pattern = (get_group_id(0) * PATTERN_BLOCK_SIZE) + patIdx;
    int matrix = get_group_id(1);
    int patternCount = totalPatterns;
    int deltaPartialsByState = pattern * PADDED_STATE_COUNT;
    int deltaPartialsByMatrix = matrix * PADDED_STATE_COUNT * patternCount;
    int deltaMatrix = matrix * PADDED_STATE_COUNT * PADDED_STATE_COUNT;
    int u = state + deltaPartialsByState + deltaPartialsByMatrix;

    int y = deltaPartialsByState + deltaPartialsByMatrix;

    // Load values into shared memory
    __local float sMatrix2[BLOCK_PEELING_SIZE][PADDED_STATE_COUNT];

    __local float sPartials2[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];

    __local float fixedScalingFactors[PATTERN_BLOCK_SIZE];

    // copy PADDED_STATE_COUNT*PATTERN_BLOCK_SIZE lengthed partials
    if (pattern < totalPatterns) {
        sPartials2[patIdx][state] = partials2[y + state];
    } else {
        sPartials2[patIdx][state] = 0;
    }

    __global float* matrix2 = matrices2 + deltaMatrix;

    if (pattern < totalPatterns) {
        int state1 = states1[pattern]; // Coalesced; no need to share

        __global float* matrix1 = matrices1 + deltaMatrix + state1 * PADDED_STATE_COUNT;

        if (state1 < PADDED_STATE_COUNT)
            sum1 = matrix1[state];
        else
            sum1 = 1.0;
    }

    if (patIdx == 0 && state < PATTERN_BLOCK_SIZE )
        // TODO: If PATTERN_BLOCK_SIZE > PADDED_STATE_COUNT, there is a bug here
        fixedScalingFactors[state] = scalingFactors[get_group_id(0) * PATTERN_BLOCK_SIZE + state];

    for (i = 0; i < PADDED_STATE_COUNT; i += BLOCK_PEELING_SIZE) {
        // load one row of matrices
        if (patIdx < BLOCK_PEELING_SIZE) {
            sMatrix2[patIdx][state] = matrix2[patIdx * PADDED_STATE_COUNT + state];

            // sMatrix now filled with starting in state and ending in i
            matrix2 += BLOCK_PEELING_SIZE * PADDED_STATE_COUNT;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        int j;
        for(j = 0; j < BLOCK_PEELING_SIZE; j++) {
            sum2 += sMatrix2[j][state] * sPartials2[patIdx][i + j];
        }

        barrier(CLK_LOCAL_MEM_FENCE); // GTX280 FIX HERE

    }

    if (pattern < totalPatterns)
        partials3[u] = sum1 * sum2 / fixedScalingFactors[patIdx];
}

__kernel void kernelStatesStatesByPatternBlockCoherent(__global int* states1,
                                                         __global int* states2,
                                                         __global float* partials3,
                                                         __global float* matrices1,
                                                         __global float* matrices2,
                                                         int totalPatterns) {
    int state = get_local_id(0);
    int patIdx = get_local_id(1);
    int pattern = (get_group_id(0) * PATTERN_BLOCK_SIZE) + patIdx;
    int matrix = get_group_id(1);
    int patternCount = totalPatterns;
    int deltaPartialsByState = pattern * PADDED_STATE_COUNT;
    int deltaPartialsByMatrix = matrix * PADDED_STATE_COUNT * patternCount;
    int deltaMatrix = matrix * PADDED_STATE_COUNT * PADDED_STATE_COUNT;
    int u = state + deltaPartialsByState + deltaPartialsByMatrix;


    // Load values into shared memory
//  __local float sMatrix1[PADDED_STATE_COUNT];
//  __local float sMatrix2[PADDED_STATE_COUNT];

    int state1 = states1[pattern];
    int state2 = states2[pattern];

    // Points to *this* matrix
    __global float* matrix1 = matrices1 + deltaMatrix + state1 * PADDED_STATE_COUNT;
    __global float* matrix2 = matrices2 + deltaMatrix + state2 * PADDED_STATE_COUNT;

//  if (patIdx == 0) {
//      sMatrix1[state] = matrix1[state];
//      sMatrix2[state] = matrix2[state];
//  }

    barrier(CLK_LOCAL_MEM_FENCE);

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

__kernel void kernelStatesStatesByPatternBlockFixedScaling(__global int* states1,
                                                             __global int* states2,
                                                             __global float* partials3,
                                                             __global float* matrices1,
                                                             __global float* matrices2,
                                                             __global float* scalingFactors,
                                                             int totalPatterns) {
    int state = get_local_id(0);
    int patIdx = get_local_id(1);
    int pattern = (get_group_id(0) * PATTERN_BLOCK_SIZE) + patIdx;
    int matrix = get_group_id(1);
    int patternCount = totalPatterns;
    int deltaPartialsByState = pattern * PADDED_STATE_COUNT;
    int deltaPartialsByMatrix = matrix * PADDED_STATE_COUNT * patternCount;
    int deltaMatrix = matrix * PADDED_STATE_COUNT * PADDED_STATE_COUNT;
    int u = state + deltaPartialsByState + deltaPartialsByMatrix;

    // Load values into shared memory
    // Prefetching into shared memory gives no performance gain
    // TODO: Double-check.
//  __local float sMatrix1[PADDED_STATE_COUNT];
//  __local float sMatrix2[PADDED_STATE_COUNT];

    __local float fixedScalingFactors[PATTERN_BLOCK_SIZE];

    int state1 = states1[pattern];
    int state2 = states2[pattern];

    // Points to *this* matrix
    __global float* matrix1 = matrices1 + deltaMatrix + state1 * PADDED_STATE_COUNT;
    __global float* matrix2 = matrices2 + deltaMatrix + state2 * PADDED_STATE_COUNT;

//  if (patIdx == 0) {
//      sMatrix1[state] = matrix1[state];
//      sMatrix2[state] = matrix2[state];
//  }

    // TODO: If PATTERN_BLOCK_SIZE > PADDED_STATE_COUNT, there is a bug here
    if (patIdx == 0 && state < PATTERN_BLOCK_SIZE )
        fixedScalingFactors[state] = scalingFactors[get_group_id(0) * PATTERN_BLOCK_SIZE + state];

    barrier(CLK_LOCAL_MEM_FENCE);

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


__kernel void kernelPartialsPartialsEdgeLikelihoods(__global float* dPartialsTmp,
                                                         __global float* dParentPartials,
                                                         __global float* dChildParials,
                                                         __global float* dTransMatrix,
                                                         int patternCount) {
    float sum1 = 0;

    int i;

    int state = get_local_id(0);
    int patIdx = get_local_id(1);
    int pattern = (get_group_id(0) * PATTERN_BLOCK_SIZE) + patIdx;
    int matrix = get_group_id(1);
    int totalPatterns = patternCount;
    int deltaPartialsByState = pattern * PADDED_STATE_COUNT;
    int deltaPartialsByMatrix = matrix * PADDED_STATE_COUNT * totalPatterns;
    int deltaMatrix = matrix * PADDED_STATE_COUNT * PADDED_STATE_COUNT;
    int u = state + deltaPartialsByState + deltaPartialsByMatrix;

    __global float* matrix1 = dTransMatrix + deltaMatrix; // Points to *this* matrix

    int y = deltaPartialsByState + deltaPartialsByMatrix;

    // Load values into shared memory
    __local float sMatrix1[BLOCK_PEELING_SIZE][PADDED_STATE_COUNT];

    __local float sPartials1[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];
    __local float sPartials2[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];

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
        barrier(CLK_LOCAL_MEM_FENCE);

        int j;
        for(j = 0; j < BLOCK_PEELING_SIZE; j++) {
            sum1 += sMatrix1[j][state] * sPartials1[patIdx][i + j];
        }

        barrier(CLK_LOCAL_MEM_FENCE); // GTX280 FIX HERE

    }

    if (pattern < totalPatterns)
        dPartialsTmp[u] = sum1 * sPartials2[patIdx][state];
}

__kernel void kernelStatesPartialsEdgeLikelihoods(__global float* dPartialsTmp,
                                                    __global float* dParentPartials,
                                                    __global int* dChildStates,
                                                    __global float* dTransMatrix,
                                                    int patternCount) {
    float sum1 = 0;

    int state = get_local_id(0);
    int patIdx = get_local_id(1);
    int pattern = (get_group_id(0) * PATTERN_BLOCK_SIZE) + patIdx;
    int matrix = get_group_id(1);
    int totalPatterns = patternCount;
    int deltaPartialsByState = pattern * PADDED_STATE_COUNT;
    int deltaPartialsByMatrix = matrix * PADDED_STATE_COUNT * patternCount;
    int deltaMatrix = matrix * PADDED_STATE_COUNT * PADDED_STATE_COUNT;
    int u = state + deltaPartialsByState + deltaPartialsByMatrix;

    int y = deltaPartialsByState + deltaPartialsByMatrix;

    // Load values into shared memory
    __local float sPartials2[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];

    // copy PADDED_STATE_COUNT*PATTERN_BLOCK_SIZE lengthed partials
    if (pattern < totalPatterns) {
        sPartials2[patIdx][state] = dParentPartials[y + state];
    } else {
        sPartials2[patIdx][state] = 0;
    }

    if (pattern < totalPatterns) {
        int state1 = dChildStates[pattern]; // Coalesced; no need to share

        __global float* matrix1 = dTransMatrix + deltaMatrix + state1 * PADDED_STATE_COUNT;

        if (state1 < PADDED_STATE_COUNT)
            sum1 = matrix1[state];
        else
            sum1 = 1.0;
    }

    if (pattern < totalPatterns)
        dPartialsTmp[u] = sum1 * sPartials2[patIdx][state];                         
}


__kernel void kernelPartialsPartialsByPatternBlockCoherentSmall(__global float* partials1,
                                                                  __global float* partials2,
                                                                  __global float* partials3,
                                                                  __global float* matrices1,
                                                                  __global float* matrices2,
                                                                  int totalPatterns) {
    float sum1 = 0;
    float sum2 = 0;
    int i;

    int tx = get_local_id(0);
    int state = tx % 4;
    int pat = tx / 4;
    int patIdx = get_local_id(1);
    int matrix = get_group_id(1);
    int patternCount = totalPatterns; // get_num_groups(0);

    // read 4 patterns at a time, since 4 * 4 = 16 
    int pattern = (get_group_id(0) * PATTERN_BLOCK_SIZE * 4) + patIdx * 4 + pat;

//  int deltaPartialsByState = __umul24(pattern, PADDED_STATE_COUNT);
    int deltaPartialsByState = 4 * 4 * (get_group_id(0) * PATTERN_BLOCK_SIZE + patIdx);
    int deltaPartialsByMatrix = (matrix * ( PADDED_STATE_COUNT * patternCount));

    int x2 = (matrix * PADDED_STATE_COUNT * PADDED_STATE_COUNT);

    __global float* matrix1 = matrices1 + x2; // Points to *this* matrix
    __global float* matrix2 = matrices2 + x2;

    int y = deltaPartialsByState + deltaPartialsByMatrix;
    int u = tx + deltaPartialsByState + deltaPartialsByMatrix;

    // Load values into shared memory
    __local float sMatrix1[16];
    __local float sMatrix2[16];

    __local float sPartials1[PATTERN_BLOCK_SIZE * 4 * 4];
    __local float sPartials2[PATTERN_BLOCK_SIZE * 4 * 4];

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

    barrier(CLK_LOCAL_MEM_FENCE);

    if (pattern < totalPatterns) { // Remove padded threads!
        for(i = 0; i < PADDED_STATE_COUNT; i++) {
            sum1 += sMatrix1[i * 4 + state] * sPartials1[patIdx * 16 + pat * 4 + i];
            sum2 += sMatrix2[i * 4 + state] * sPartials2[patIdx * 16 + pat * 4 + i];
        }
        partials3[u] = sum1 * sum2;
    }

}

__kernel void kernelPartialsPartialsByPatternBlockSmallFixedScaling(__global float* partials1,
                                                                      __global float* partials2,
                                                                      __global float* partials3,
                                                                      __global float* matrices1,
                                                                      __global float* matrices2,
                                                                      __global float* scalingFactors,
                                                                      int totalPatterns) {
    float sum1 = 0;
    float sum2 = 0;
    int i;

    int tx = get_local_id(0);
    int state = tx % 4;
    int pat = tx / 4;
    int patIdx = get_local_id(1);
    int matrix = get_group_id(1);
    int patternCount = totalPatterns; // get_num_groups(0);

    // read 4 patterns at a time, since 4 * 4 = 16
    int pattern = (get_group_id(0) * PATTERN_BLOCK_SIZE * 4) + patIdx * 4 + pat;

//  int deltaPartialsByState = __umul24(pattern,PADDED_STATE_COUNT);
    int deltaPartialsByState = 4 * 4 * (get_group_id(0) * PATTERN_BLOCK_SIZE + patIdx);
    int deltaPartialsByMatrix = (matrix * ( PADDED_STATE_COUNT * patternCount));

    int x2 = (matrix * PADDED_STATE_COUNT * PADDED_STATE_COUNT);

    __global float* matrix1 = matrices1 + x2; // Points to *this* matrix
    __global float* matrix2 = matrices2 + x2;


    int y = deltaPartialsByState + deltaPartialsByMatrix;
    int u = tx + deltaPartialsByState + deltaPartialsByMatrix;

    // Load values into shared memory
    __local float sMatrix1[16];
    __local float sMatrix2[16];

    __local float sPartials1[PATTERN_BLOCK_SIZE * 4 * 4];
    __local float sPartials2[PATTERN_BLOCK_SIZE * 4 * 4];

    __local float fixedScalingFactors[PATTERN_BLOCK_SIZE * 4];

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
            scalingFactors[get_group_id(0) * PATTERN_BLOCK_SIZE * 4 + patIdx * PATTERN_BLOCK_SIZE + tx];

    if (patIdx == 0) {
        sMatrix1[tx] = matrix1[tx]; // All coalesced memory reads
        sMatrix2[tx] = matrix2[tx];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (pattern < totalPatterns) { // Remove padded threads!
        for(i = 0; i < PADDED_STATE_COUNT; i++) {
            sum1 += sMatrix1[i * 4 + state] * sPartials1[patIdx * 16 + pat * 4 + i];
            sum2 += sMatrix2[i * 4 + state] * sPartials2[patIdx * 16 + pat * 4 + i];
        }
        partials3[u] = sum1 * sum2 / fixedScalingFactors[patIdx * 4 + pat];
    }

}

__kernel void kernelStatesPartialsByPatternBlockCoherentSmall(__global int* states1,
                                                                __global float* partials2,
                                                                __global float* partials3,
                                                                __global float* matrices1,
                                                                __global float* matrices2,
                                                                int totalPatterns) {
    float sum1 = 0;
    float sum2 = 0;
    int i;

    int tx = get_local_id(0);
    int state = tx % 4;
    int pat = tx / 4;
    int patIdx = get_local_id(1);
    int matrix = get_group_id(1);
    int patternCount = totalPatterns; // get_num_groups(0);

    // read 4 patterns at a time, since 4 * 4 = 16
    int pattern = (get_group_id(0) * PATTERN_BLOCK_SIZE * 4) + patIdx * 4 + pat;

    int deltaPartialsByState = 4 * 4 * (get_group_id(0) * PATTERN_BLOCK_SIZE + patIdx);
    int deltaPartialsByMatrix = (matrix * ( PADDED_STATE_COUNT * patternCount));

    int x2 = (matrix * PADDED_STATE_COUNT * PADDED_STATE_COUNT);

    __global float* matrix1 = matrices1 + x2; // Points to *this* matrix
    __global float* matrix2 = matrices2 + x2;


    int y = deltaPartialsByState + deltaPartialsByMatrix;
    int u = tx + deltaPartialsByState + deltaPartialsByMatrix;

    // Load values into shared memory
    __local float sMatrix1[16];
    __local float sMatrix2[16];

//  __local INT sStates1[PATTERN_BLOCK_SIZE * 4];
    __local float sPartials2[PATTERN_BLOCK_SIZE * 4 * 4];

    // copy PADDED_STATE_COUNT * PATTERN_BLOCK_SIZE lengthed partials
    if (pattern < totalPatterns) {
        sPartials2[patIdx * 16 + tx] = partials2[y + tx];
//      if (patIdx < PATTERN_BLOCK_SIZE / 4)
//          sStates1[patIdx * 16 + tx] = states1[get_group_id(0) * PATTERN_BLOCK_SIZE * 4 + patIdx * 16 +
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

    barrier(CLK_LOCAL_MEM_FENCE);

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

__kernel void kernelStatesStatesByPatternBlockCoherentSmall(__global int* states1,
                                                              __global int* states2,
                                                              __global float* partials3,
                                                              __global float* matrices1,
                                                              __global float* matrices2,
                                                              int totalPatterns) {

    int tx = get_local_id(0);
    int state = tx % 4;
    int pat = tx / 4;
    int patIdx = get_local_id(1);
    int matrix = get_group_id(1);
    int patternCount = totalPatterns; // get_num_groups(0);

    // read 4 patterns at a time, since 4 * 4 = 16
    int pattern = (get_group_id(0) * PATTERN_BLOCK_SIZE * 4) + patIdx * 4 + pat;

    int deltaPartialsByState = 4 * 4 * (get_group_id(0) * PATTERN_BLOCK_SIZE + patIdx);
    int deltaPartialsByMatrix = (matrix * (PADDED_STATE_COUNT * patternCount));

    int x2 = (matrix * PADDED_STATE_COUNT * PADDED_STATE_COUNT);

    __global float* matrix1 = matrices1 + x2; // Points to *this* matrix
    __global float* matrix2 = matrices2 + x2;

    int u = tx + deltaPartialsByState + deltaPartialsByMatrix;

    // Load values into shared memory
    __local float sMatrix1[16];
    __local float sMatrix2[16];

//  __local INT sStates1[PATTERN_BLOCK_SIZE * 4];
//  __local INT sStates2[PATTERN_BLOCK_SIZE * 4];
//
//  if (pattern < totalPatterns) {
//      if (patIdx < PATTERN_BLOCK_SIZE/4) {
//          sStates1[patIdx * 16 + tx] = states1[get_group_id(0) * PATTERN_BLOCK_SIZE*4 + patIdx * 16 +
//                                               tx];
//          sStates2[patIdx * 16 + tx] = states2[get_group_id(0) * PATTERN_BLOCK_SIZE*4 + patIdx * 16 +
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

    barrier(CLK_LOCAL_MEM_FENCE);

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

__kernel void kernelPartialsPartialsEdgeLikelihoodsSmall(__global float* dPartialsTmp,
                                                              __global float* dParentPartials,
                                                              __global float* dChildParials,
                                                              __global float* dTransMatrix,
                                                              int patternCount) {
    float sum1 = 0;

    int i;

    int tx = get_local_id(0);
    int state = tx % 4;
    int pat = tx / 4;
    int patIdx = get_local_id(1);
    int matrix = get_group_id(1);
    int totalPatterns = patternCount; // get_num_groups(0);

    // read 4 patterns at a time, since 4 * 4 = 16 
    int pattern = (get_group_id(0) * PATTERN_BLOCK_SIZE * 4) + patIdx * 4 + pat;

    int deltaPartialsByState = 4 * 4 * (get_group_id(0) * PATTERN_BLOCK_SIZE + patIdx);
    int deltaPartialsByMatrix = (matrix * ( PADDED_STATE_COUNT * totalPatterns));

    int x2 = (matrix * PADDED_STATE_COUNT * PADDED_STATE_COUNT);

    __global float* matrix1 = dTransMatrix + x2; // Points to *this* matrix

    int y = deltaPartialsByState + deltaPartialsByMatrix;
    int u = tx + deltaPartialsByState + deltaPartialsByMatrix;

    // Load values into shared memory
    __local float sMatrix1[16];

    __local float sPartials1[PATTERN_BLOCK_SIZE * 4 * 4];
    __local float sPartials2[PATTERN_BLOCK_SIZE * 4 * 4];

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

    barrier(CLK_LOCAL_MEM_FENCE);

    if (pattern < totalPatterns) { // Remove padded threads!
        for(i = 0; i < PADDED_STATE_COUNT; i++) {
            sum1 += sMatrix1[i * 4 + state] * sPartials1[patIdx * 16 + pat * 4 + i];
        }
        dPartialsTmp[u] = sum1 * sPartials2[patIdx * 16 + pat * 4 + state];
    }    

}

__kernel void kernelStatesPartialsEdgeLikelihoodsSmall(__global float* dPartialsTmp,
                                                         __global float* dParentPartials,
                                                         __global int* dChildStates,
                                                         __global float* dTransMatrix,
                                                         int patternCount) {
    float sum1 = 0;

    int tx = get_local_id(0);
    int state = tx % 4;
    int pat = tx / 4;
    int patIdx = get_local_id(1);
    int matrix = get_group_id(1);
    int totalPatterns = patternCount; // get_num_groups(0);

    // read 4 patterns at a time, since 4 * 4 = 16
    int pattern = (get_group_id(0) * PATTERN_BLOCK_SIZE * 4) + patIdx * 4 + pat;

    int deltaPartialsByState = 4 * 4 * (get_group_id(0) * PATTERN_BLOCK_SIZE + patIdx);
    int deltaPartialsByMatrix = (matrix * ( PADDED_STATE_COUNT * patternCount));

    int x2 = (matrix * PADDED_STATE_COUNT * PADDED_STATE_COUNT);

    __global float* matrix1 = dTransMatrix + x2; // Points to *this* matrix

    int y = deltaPartialsByState + deltaPartialsByMatrix;
    int u = tx + deltaPartialsByState + deltaPartialsByMatrix;

    // Load values into shared memory
    __local float sMatrix1[16];

    __local float sPartials2[PATTERN_BLOCK_SIZE * 4 * 4];

    // copy PADDED_STATE_COUNT * PATTERN_BLOCK_SIZE lengthed partials
    if (pattern < totalPatterns) {
        sPartials2[patIdx * 16 + tx] = dParentPartials[y + tx];
    } else {
        sPartials2[patIdx * 16 + tx] = 0;
    }

    if (patIdx == 0) {
        sMatrix1[tx] = matrix1[tx]; // All coalesced memory reads
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (pattern < totalPatterns) { // Remove padded threads!
        int state1 = dChildStates[pattern];

        if (state1 < PADDED_STATE_COUNT)
            sum1 = sMatrix1[state1 * 4 + state];
        else
            sum1 = 1.0;

        dPartialsTmp[u] = sum1 * sPartials2[patIdx * 16 + pat * 4 + state];
    }
}

__kernel void kernelIntegrateLikelihoodsDynamicScaling(__global float* dResult,
                                                            __global float* dRootPartials,
                                                            float *dWeights,
                                                            float *dFrequencies,
                                                            float *dRootScalingFactors,
                                                            int count) {
    int state   = get_local_id(0);
    int pattern = get_group_id(0);
    int patternCount = get_num_groups(0);

    __local float stateFreq[PADDED_STATE_COUNT];
    // TODO: Currently assumes MATRIX_BLOCK_SIZE >> matrixCount
    __local float matrixProp[MATRIX_BLOCK_SIZE];
    __local float sum[PADDED_STATE_COUNT];

    // Load shared memory

    stateFreq[state] = dFrequencies[state];
    sum[state] = 0;

    for(int matrixEdge = 0; matrixEdge < count; matrixEdge += PADDED_STATE_COUNT) {
        int x = matrixEdge + state;
        if (x < count)
            matrixProp[x] = dWeights[x];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    int u = state + pattern * PADDED_STATE_COUNT;
    int delta = patternCount * PADDED_STATE_COUNT;;

    for(int r = 0; r < count; r++) {
        sum[state] += dRootPartials[u + delta * r] * matrixProp[r];
    }

    sum[state] *= stateFreq[state];
    barrier(CLK_LOCAL_MEM_FENCE);

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
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (state == 0)
        dResult[pattern] = log(sum[state]) + dRootScalingFactors[pattern];
}


__kernel void kernelAccumulateFactorsDynamicScaling(__global float** dNodePtrQueue,
                                                   __global float* rootScaling,
                                                   int nodeCount,
                                                   int patternCount) {
    int pattern = get_local_id(0) + get_group_id(0) * PATTERN_BLOCK_SIZE;

    float total = 0;
    __global float* nodeScales;

    int n;
    for(n = 0; n < nodeCount; n++) {
//      if (threadIdx.x == 0) // TODO Why does this not work???
            nodeScales = dNodePtrQueue[n];
//      __syncthreads();

        float factor = nodeScales[pattern];
        if (factor != 1.0)
            total += log(factor);
    }

    if (pattern < patternCount)
        rootScaling[pattern] += total;
}

__kernel void kernelRemoveFactorsDynamicScaling(__global float** dNodePtrQueue,
                                                   __global float* rootScaling,
                                                   int nodeCount,
                                                   int patternCount) {
    int pattern = get_local_id(0) + get_group_id(0) * PATTERN_BLOCK_SIZE;

    float total = 0;
    __global float* nodeScales;

    int n;
    for(n = 0; n < nodeCount; n++) {
//      if (threadIdx.x == 0) // TODO Why does this not work???
            nodeScales = dNodePtrQueue[n];
//      __syncthreads();

        float factor = nodeScales[pattern];
        if (factor != 1.0)
            total += log(factor);
    }

    if (pattern < patternCount)
        rootScaling[pattern] -= total;
}

/*
 * Find a scaling factor for each pattern
 */
__kernel void kernelPartialsDynamicScaling(__global float* allPartials,
                                             __global float* scalingFactors,
                                             int matrixCount) {
    int state = get_local_id(0);
    int matrix = get_local_id(1);
    int pattern = get_group_id(0);
    int patternCount = get_num_groups(0);

    int deltaPartialsByMatrix = (matrix * (PADDED_STATE_COUNT * patternCount));

    // TODO: Currently assumes MATRIX_BLOCK_SIZE > matrixCount; FIX!!!
    __local float partials[MATRIX_BLOCK_SIZE][PADDED_STATE_COUNT];

    __local float max;

    if (matrix < matrixCount)
        partials[matrix][state] = allPartials[matrix * patternCount * PADDED_STATE_COUNT + pattern *
                                              PADDED_STATE_COUNT + state];
    else
        partials[matrix][state] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    int i;
    // parallelized reduction; assumes PADDED_STATE_COUNT is power of 2.
    for (i = PADDED_STATE_COUNT / 2; i > 0; i >>= 1) {
        if (state < i) {
            float compare1 = partials[matrix][state];
            float compare2 = partials[matrix][state + i];
            if (compare2 > compare1)
            partials[matrix][state] = compare2;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
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

    barrier(CLK_LOCAL_MEM_FENCE);

    if (matrix < matrixCount)
        allPartials[matrix * patternCount * PADDED_STATE_COUNT + pattern * PADDED_STATE_COUNT +
                    state] /= max;

    barrier(CLK_LOCAL_MEM_FENCE);
}

/*
 * Find a scaling factor for each pattern and accumulate into buffer
 */
__kernel void kernelPartialsDynamicScalingAccumulate(__global float* allPartials,
                                                       __global float* scalingFactors,
                                                       __global float* cumulativeScaling,
                                                       int matrixCount) {
    int state = get_local_id(0);
    int matrix = get_local_id(1);
    int pattern = get_group_id(0);
    int patternCount = get_num_groups(0);

    int deltaPartialsByMatrix = (matrix * (PADDED_STATE_COUNT * patternCount));

    // TODO: Currently assumes MATRIX_BLOCK_SIZE > matrixCount; FIX!!!
    __local float partials[MATRIX_BLOCK_SIZE][PADDED_STATE_COUNT];

    __local float max;

    if (matrix < matrixCount)
        partials[matrix][state] = allPartials[matrix * patternCount * PADDED_STATE_COUNT + pattern *
                                              PADDED_STATE_COUNT + state];
    else
        partials[matrix][state] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    int i;
    // parallelized reduction; assumes PADDED_STATE_COUNT is power of 2.
    for (i = PADDED_STATE_COUNT / 2; i > 0; i >>= 1) {
        if (state < i) {
            float compare1 = partials[matrix][state];
            float compare2 = partials[matrix][state + i];
            if (compare2 > compare1)
            partials[matrix][state] = compare2;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (state == 0 && matrix == 0) {
        max = 0;
        int m;
        for(m = 0; m < matrixCount; m++) {
            if (partials[m][0] > max)
                max = partials[m][0];
        }

        scalingFactors[pattern] = max; // TODO: These are incoherent memory writes!!!
        cumulativeScaling[pattern] += log(max);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (matrix < matrixCount)
        allPartials[matrix * patternCount * PADDED_STATE_COUNT + pattern * PADDED_STATE_COUNT +
                    state] /= max;

    barrier(CLK_LOCAL_MEM_FENCE);
}



__kernel void kernelPartialsDynamicScalingSlow(__global float* allPartials,
                                                 __global float* scalingFactors,
                                                 int matrixCount) {
    int state = get_local_id(0);
    int matrix = get_local_id(1);
    int pattern = get_group_id(0);
    int patternCount = get_num_groups(0);

    int deltaPartialsByMatrix = (matrix * ( PADDED_STATE_COUNT * patternCount));

    __local float partials[PADDED_STATE_COUNT];

    __local float max;

    if (state == 0)
        max = 0.0;

    int m;
    for(m = 0; m < matrixCount; m++) {
        partials[state] = allPartials[m * patternCount * PADDED_STATE_COUNT + pattern *
                                      PADDED_STATE_COUNT + state];
        barrier(CLK_LOCAL_MEM_FENCE);

#ifdef IS_POWER_OF_TWO
    // parallelized reduction *** only works for powers-of-2 ****
    for (int i = PADDED_STATE_COUNT / 2; i > 0; i >>= 1) {
        if (state < i) {
#else
    for (int i = SMALLEST_POWER_OF_TWO / 2; i > 0; i >>= 1) {
        if (state < i && state + i < PADDED_STATE_COUNT ) {
#endif // IS_POWER_OF_TWO
                float compare1 = partials[state];
                float compare2 = partials[state + i];
                if(compare2 > compare1)
                    partials[state] = compare2;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if(state == 0) {
            if( partials[0] > max)
                max = partials[0];
        }
    }

    if(state == 0)
        scalingFactors[pattern] = max;

    barrier(CLK_LOCAL_MEM_FENCE);

    for(m = 0; m < matrixCount; m++)
        allPartials[m * patternCount * PADDED_STATE_COUNT + pattern * PADDED_STATE_COUNT +
                    state] /= max;

}

__kernel void kernelIntegrateLikelihoods(__global float* dResult,
                                              __global float* dRootPartials,
                                              __global float* dWeights,
                                              __global float* dFrequencies,
                                              int count) {
    int state   = get_local_id(0);
    int pattern = get_group_id(0);
    int patternCount = get_num_groups(0);

    __local float stateFreq[PADDED_STATE_COUNT];
    // TODO: Currently assumes MATRIX_BLOCK_SIZE >> matrixCount
    __local float matrixProp[MATRIX_BLOCK_SIZE];
    __local float sum[PADDED_STATE_COUNT];

    // Load shared memory

    stateFreq[state] = dFrequencies[state];
    sum[state] = 0;

    for(int matrixEdge = 0; matrixEdge < count; matrixEdge += PADDED_STATE_COUNT) {
        int x = matrixEdge + state;
        if (x < count)
            matrixProp[x] = dWeights[x];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    int u = state + pattern * PADDED_STATE_COUNT;
    int delta = patternCount * PADDED_STATE_COUNT;

    for(int r = 0; r < count; r++) {
        sum[state] += dRootPartials[u + delta * r] * matrixProp[r];
    }

    sum[state] *= stateFreq[state];
    barrier(CLK_LOCAL_MEM_FENCE);

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
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (state == 0)
        dResult[pattern] = log(sum[state]);
}
