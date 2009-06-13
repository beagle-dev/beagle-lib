/*
 * @author Marc Suchard
 */

/**************INCLUDES***********/
#include <stdio.h>
#include <string.h>
#include <cuda_runtime_api.h>
#include "CUDASharedFunctions.h"

#include "TransitionKernels.cu"

/**************CODE***********/
#ifdef __cplusplus
extern "C" {
#endif

void nativeGPUGetTransitionProbabilitiesSquare(REAL** dPtrQueue,
                                               REAL* dEvec,
                                               REAL* dIevc,
                                               REAL* dEigenValues,
                                               REAL* distanceQueue,
                                               int totalMatrix) {
#ifdef DEBUG
    fprintf(stderr, "Starting GPU TP\n");
    cudaThreadSynchronize();
    checkCUDAError("TP kernel pre-invocation");
#endif

    dim3 block(MULTIPLY_BLOCK_SIZE, MULTIPLY_BLOCK_SIZE);
    dim3 grid(PADDED_STATE_COUNT / MULTIPLY_BLOCK_SIZE, PADDED_STATE_COUNT / MULTIPLY_BLOCK_SIZE);
    if (PADDED_STATE_COUNT % MULTIPLY_BLOCK_SIZE != 0) {
        grid.x += 1;
        grid.y += 1;
    }
    grid.x *= totalMatrix;

    // Transposed (interchanged Ievc and Evec)
    matrixMulADB<<<grid, block>>>(dPtrQueue, dIevc, dEigenValues, dEvec, distanceQueue,
                                  PADDED_STATE_COUNT, PADDED_STATE_COUNT, totalMatrix);

#ifdef DEBUG
    fprintf(stderr, "Ending GPU TP\n");
    cudaThreadSynchronize();
    checkCUDAError("TP kernel invocation");
#endif
}

#ifdef __cplusplus
}
#endif
