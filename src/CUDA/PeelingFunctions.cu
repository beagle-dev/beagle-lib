/*
 * @author Marc Suchard
 */
#ifndef _Included_PeelingFunctions
#define _Included_PeelingFunctions

/**************INCLUDES***********/
#include <stdio.h>
#include <cuda_runtime_api.h>
#include "BeagleCUDAImpl.h"
#include "CUDASharedFunctions.h"

#include "PeelingKernels.cu"

/**************CODE***********/
#ifdef __cplusplus
extern "C" {
#endif

REAL* ones = NULL; // TODO: Memory leak, need to free at some point.

void nativeGPUPartialsPartialsPruningDynamicScaling(REAL* partials1,
                                                    REAL* partials2,
                                                    REAL* partials3,
                                                    REAL* matrices1,
                                                    REAL* matrices2,
                                                    REAL* scalingFactors,
                                                    const unsigned int patternCount,
                                                    const unsigned int matrixCount,
                                                    int doRescaling) {
#ifdef DEBUG
    fprintf(stderr, "Entering GPU PP\n");
    cudaThreadSynchronize();
    checkCUDAError("PP kernel pre-invocation");
#endif

#if (PADDED_STATE_COUNT == 4)
    dim3 grid(patternCount / (PATTERN_BLOCK_SIZE * 4), matrixCount);
    if (patternCount % (PATTERN_BLOCK_SIZE * 4) != 0)
        grid.x += 1;
    dim3 block(16, PATTERN_BLOCK_SIZE);
#else
    dim3 grid(patternCount / PATTERN_BLOCK_SIZE, matrixCount);
    if (patternCount % PATTERN_BLOCK_SIZE != 0)
        grid.x += 1;
    dim3 block(PADDED_STATE_COUNT, PATTERN_BLOCK_SIZE);
#endif

    if (doRescaling)    {
        // Compute partials without any rescaling
#if (PADDED_STATE_COUNT == 4)
        kernelPartialsPartialsByPatternBlockCoherentSmall<<<grid, block>>>(partials1, partials2,
                                                                           partials3, matrices1,
                                                                           matrices2, patternCount);
#else
        kernelPartialsPartialsByPatternBlockCoherent<<<grid, block>>>(partials1, partials2,
                                                                      partials3, matrices1,
                                                                      matrices2, patternCount);
#endif

        cudaThreadSynchronize();

        // Rescale partials and save scaling factors
        nativeGPURescalePartials(partials3, scalingFactors, patternCount, matrixCount, 0);

    } else {

    // Compute partials with known rescalings
#if (PADDED_STATE_COUNT == 4)
        kernelPartialsPartialsByPatternBlockSmallFixedScaling<<<grid, block>>>(partials1, partials2,
                                                                               partials3, matrices1,
                                                                               matrices2,
                                                                               scalingFactors,
                                                                               patternCount);
#else
        kernelPartialsPartialsByPatternBlockFixedScaling<<<grid, block>>>(partials1, partials2,
                                                                          partials3, matrices1,
                                                                          matrices2, scalingFactors,
                                                                          patternCount);
#endif

    }

#ifdef DEBUG
    cudaThreadSynchronize();
    checkCUDAError("PP kernel invocation");
    fprintf(stderr, "Completed GPU PP\n");
#endif

}

void nativeGPUPartialsPartialsPruning(REAL* partials1,
                                      REAL* partials2,
                                      REAL* partials3,
                                      REAL* matrices1,
                                      REAL* matrices2,
                                      const unsigned int patternCount,
                                      const unsigned int matrixCount) {
#ifdef DEBUG
    fprintf(stderr, "Entering GPU PP\n");
    cudaThreadSynchronize();
    checkCUDAError("PP kernel pre-invocation");
#endif


#if (PADDED_STATE_COUNT == 4)
    dim3 block(16, PATTERN_BLOCK_SIZE);
    dim3 grid(patternCount / (PATTERN_BLOCK_SIZE * 4), matrixCount);
    if (patternCount % (PATTERN_BLOCK_SIZE * 4) != 0)
        grid.x += 1;

    kernelPartialsPartialsByPatternBlockCoherentSmall<<<grid, block>>>(partials1, partials2,
                                                                       partials3, matrices1,
                                                                       matrices2, patternCount);
#else
    dim3 grid(patternCount / PATTERN_BLOCK_SIZE, matrixCount);
    if (patternCount % PATTERN_BLOCK_SIZE != 0)
        grid.x += 1;
    dim3 block(PADDED_STATE_COUNT, PATTERN_BLOCK_SIZE);

    kernelPartialsPartialsByPatternBlockCoherent<<<grid, block>>>(partials1, partials2, partials3,
                                                                  matrices1, matrices2,
                                                                  patternCount);
#endif

#ifdef DEBUG
    cudaThreadSynchronize();
    checkCUDAError("PP kernel invocation");
    fprintf(stderr, "Completed GPU PP\n");
#endif

}

void nativeGPURescalePartials(REAL* partials3,
                              REAL* scalingFactors,
                              int patternCount,
                              int matrixCount,
                              int fillWithOnes) {
    // Rescale partials and save scaling factors
//#if (PADDED_STATE_COUNT == 4) 
    if (fillWithOnes != 0) {
        if (ones == NULL) {
            ones = (REAL*) malloc(SIZE_REAL * patternCount);
            for(int i = 0; i < patternCount; i++)
                ones[i] = 1.0;
        }
        cudaMemcpy(scalingFactors, ones, sizeof(REAL*) * patternCount, cudaMemcpyHostToDevice);
        return;
    }
//#endif

#ifndef SLOW_REWEIGHING
    dim3 grid2(patternCount, matrixCount / MATRIX_BLOCK_SIZE);
    if (matrixCount % MATRIX_BLOCK_SIZE != 0)
        grid2.y += 1;
    if (grid2.y > 1) {
        fprintf(stderr, "Not yet implemented! Try slow reweighing.\n");
        exit(0);
    }
    dim3 block2(PADDED_STATE_COUNT, MATRIX_BLOCK_SIZE);
    // TODO: Totally incoherent for PADDED_STATE_COUNT == 4
    kernelPartialsDynamicScaling<<<grid2, block2>>>(partials3, scalingFactors, matrixCount);
#else
    dim3 grid2(patternCount, 1);
    dim3 block2(PADDED_STATE_COUNT);
    kernelPartialsDynamicScalingSlow<<<grid2, block2>>>(partials3, scalingFactors, matrixCount);
#endif
}

void nativeGPUStatesStatesPruningDynamicScaling(INT* states1,
                                                INT* states2,
                                                REAL* partials3,
                                                REAL* matrices1,
                                                REAL* matrices2,
                                                REAL* scalingFactors,
                                                const unsigned int patternCount,
                                                const unsigned int matrixCount,
                                                int doRescaling) {
#if (PADDED_STATE_COUNT == 4)
    dim3 grid(patternCount/(PATTERN_BLOCK_SIZE * 4), matrixCount);
    if (patternCount % (PATTERN_BLOCK_SIZE * 4) != 0)
        grid.x += 1;
    dim3 block(16, PATTERN_BLOCK_SIZE);
#else
    dim3 grid(patternCount / PATTERN_BLOCK_SIZE, matrixCount);
    if (patternCount % PATTERN_BLOCK_SIZE != 0)
        grid.x += 1;
    dim3 block(PADDED_STATE_COUNT, PATTERN_BLOCK_SIZE);
#endif

    if (doRescaling)    {
        // Compute partials without any rescaling
#if (PADDED_STATE_COUNT == 4)
        kernelStatesStatesByPatternBlockCoherentSmall<<<grid, block>>>(states1, states2, partials3,
                                                                       matrices1, matrices2,
                                                                       patternCount);
#else
        kernelStatesStatesByPatternBlockCoherent<<<grid, block>>>(states1, states2, partials3,
                                                                  matrices1, matrices2,
                                                                  patternCount);
#endif
        cudaThreadSynchronize();

        // Rescale partials and save scaling factors
        // If PADDED_STATE_COUNT == 4, just with ones.
        nativeGPURescalePartials(partials3, scalingFactors, patternCount, matrixCount, 1);

    } else {

        // Compute partials with known rescalings
#if (PADDED_STATE_COUNT == 4)
        kernelStatesStatesByPatternBlockCoherentSmall<<<grid, block>>>(states1, states2, partials3,
                                                                       matrices1, matrices2,
                                                                       patternCount);
#else
        kernelStatesStatesByPatternBlockFixedScaling<<<grid, block>>>(states1, states2, partials3,
                                                                      matrices1, matrices2,
                                                                      scalingFactors, patternCount);
#endif
    }

#ifdef DEBUG
    fprintf(stderr, "Completed GPU SP\n");
#endif
}

void nativeGPUStatesPartialsPruningDynamicScaling(INT* states1,
                                                  REAL* partials2,
                                                  REAL* partials3,
                                                  REAL* matrices1,
                                                  REAL* matrices2,
                                                  REAL* scalingFactors,
                                                  const unsigned int patternCount,
                                                  const unsigned int matrixCount,
                                                  int doRescaling) {
#if (PADDED_STATE_COUNT == 4)
    dim3 grid(patternCount / (PATTERN_BLOCK_SIZE * 4), matrixCount);
    if (patternCount % (PATTERN_BLOCK_SIZE * 4) != 0)
        grid.x += 1;
    dim3 block(16, PATTERN_BLOCK_SIZE);
#else
    dim3 grid(patternCount / PATTERN_BLOCK_SIZE, matrixCount);
    if (patternCount % PATTERN_BLOCK_SIZE != 0)
        grid.x += 1;
    dim3 block(PADDED_STATE_COUNT, PATTERN_BLOCK_SIZE);
#endif

    if (doRescaling)    {
        // Compute partials without any rescaling
#if (PADDED_STATE_COUNT == 4)
        kernelStatesPartialsByPatternBlockCoherentSmall<<<grid, block>>>(states1, partials2,
                                                                         partials3, matrices1,
                                                                         matrices2, patternCount);
#else
        kernelStatesPartialsByPatternBlockCoherent<<<grid, block>>>(states1, partials2, partials3,
                                                                    matrices1, matrices2,
                                                                    patternCount);
#endif
        cudaThreadSynchronize();

        // Rescale partials and save scaling factors
        nativeGPURescalePartials(partials3, scalingFactors, patternCount, matrixCount, 1);
    } else {

        // Compute partials with known rescalings
#if (PADDED_STATE_COUNT == 4)
        kernelStatesPartialsByPatternBlockCoherentSmall<<<grid, block>>>(states1, partials2,
                                                                         partials3, matrices1,
                                                                         matrices2, patternCount);
#else
        kernelStatesPartialsByPatternBlockFixedScaling<<<grid, block>>>(states1, partials2,
                                                                        partials3, matrices1,
                                                                        matrices2, scalingFactors,
                                                                        patternCount);
#endif
    }

#ifdef DEBUG
    fprintf(stderr,"Completed GPU SP\n");
#endif

}

void nativeGPUComputeRootDynamicScaling(REAL** dNodePtrQueue,
                                        REAL* dRootScalingFactors,
                                        int nodeCount,
                                        int patternCount) {
    dim3 grid(patternCount / PATTERN_BLOCK_SIZE);
    if (patternCount % PATTERN_BLOCK_SIZE != 0)
        grid.x += 1;
    dim3 block(PATTERN_BLOCK_SIZE);

    kernelGPUComputeRootDynamicScaling<<<grid, block>>>(dNodePtrQueue, dRootScalingFactors,
                                                        nodeCount, patternCount);
}

void nativeGPUIntegrateLikelihoodsDynamicScaling(REAL* dResult,
                                                 REAL* dRootPartials,
                                                 REAL* dCategoryProportions,
                                                 REAL* dFrequencies,
                                                 REAL* dRootScalingFactors,
                                                 int patternCount,
                                                 int matrixCount,
                                                 int nodeCount) {

#ifdef DEBUG
    fprintf(stderr, "Entering IL\n");
#endif

    dim3 grid(patternCount);
    dim3 block(PADDED_STATE_COUNT);

    kernelGPUIntegrateLikelihoodsDynamicScaling<<<grid, block>>>(dResult, dRootPartials,
                                                                 dCategoryProportions, dFrequencies,
                                                                 dRootScalingFactors, matrixCount,
                                                                 nodeCount);

#ifdef DEBUG
    fprintf(stderr, "Exiting IL\n");
#endif
}

void nativeGPUIntegrateLikelihoods(REAL* dResult,
                                   REAL* dRootPartials,
                                   REAL* dCategoryProportions,
                                   REAL* dFrequencies,
                                   int patternCount,
                                   int matrixCount) {
#ifdef DEBUG
    fprintf(stderr,"Entering IL\n");
#endif

    dim3 grid(patternCount);
    dim3 block(PADDED_STATE_COUNT);

    kernelGPUIntegrateLikelihoods<<<grid, block>>>(dResult, dRootPartials, dCategoryProportions,
                                                   dFrequencies, matrixCount);

#ifdef DEBUG
    fprintf(stderr, "Exiting IL\n");
#endif

}

void nativeGPUStatesPartialsPruning(int* states1,
                                    REAL* partials2,
                                    REAL* partials3,
                                    REAL* matrices1,
                                    REAL* matrices2,
                                    const unsigned int patternCount,
                                    const unsigned int matrixCount) {
#ifdef DEBUG
    fprintf(stderr, "Entering GPU PP\n");
    cudaThreadSynchronize();
    checkCUDAError("PP kernel pre-invocation");
#endif


#if (PADDED_STATE_COUNT == 4)
    dim3 block(16, PATTERN_BLOCK_SIZE);
    dim3 grid(patternCount / (PATTERN_BLOCK_SIZE * 4), matrixCount);
    if (patternCount % (PATTERN_BLOCK_SIZE * 4) != 0)
        grid.x += 1;

    kernelStatesPartialsByPatternBlockCoherentSmall<<<grid, block>>>(states1, partials2, partials3,
                                                                     matrices1, matrices2,
                                                                     patternCount);
#else
    dim3 grid(patternCount / PATTERN_BLOCK_SIZE, matrixCount);
    if (patternCount % PATTERN_BLOCK_SIZE != 0)
        grid.x += 1;
    dim3 block(PADDED_STATE_COUNT, PATTERN_BLOCK_SIZE);

    kernelStatesPartialsByPatternBlockCoherent<<<grid, block>>>(states1, partials2, partials3,
                                                                matrices1, matrices2, patternCount);
#endif

#ifdef DEBUG
    cudaThreadSynchronize();
    checkCUDAError("PP kernel invocation");
    fprintf(stderr, "Completed GPU PP\n");
#endif

}

void nativeGPUStatesStatesPruning(int* states1,
                                  int* states2,
                                  REAL* partials3,
                                  REAL* matrices1,
                                  REAL* matrices2,
                                  const unsigned int patternCount,
                                  const unsigned int matrixCount) {
#ifdef DEBUG
    fprintf(stderr, "Entering GPU PP\n");
    cudaThreadSynchronize();
    checkCUDAError("PP kernel pre-invocation");
#endif


#if (PADDED_STATE_COUNT == 4)
    dim3 block(16, PATTERN_BLOCK_SIZE);
    dim3 grid(patternCount/(PATTERN_BLOCK_SIZE * 4), matrixCount);
    if (patternCount % (PATTERN_BLOCK_SIZE * 4) != 0)
        grid.x += 1;

    kernelStatesStatesByPatternBlockCoherentSmall<<<grid, block>>>(states1, states2, partials3,
                                                                   matrices1, matrices2,
                                                                   patternCount);
#else
    dim3 grid(patternCount / PATTERN_BLOCK_SIZE, matrixCount);
    if (patternCount % PATTERN_BLOCK_SIZE != 0)
        grid.x += 1;
    dim3 block(PADDED_STATE_COUNT, PATTERN_BLOCK_SIZE);

    kernelStatesStatesByPatternBlockCoherent<<<grid, block>>>(states1, states2, partials3,
                                                              matrices1, matrices2, patternCount);
#endif

#ifdef DEBUG
    cudaThreadSynchronize();
    checkCUDAError("PP kernel invocation");
    fprintf(stderr, "Completed GPU PP\n");
#endif

}

#ifdef __cplusplus
}
#endif
#endif
//
