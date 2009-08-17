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

/**************INCLUDES***********/
#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include <cstdio>
#include <cstdlib>

#include "libhmsbeagle/GPU/GPUImplDefs.h"
#include "libhmsbeagle/GPU/KernelLauncher.h"

/**************CODE***********/

REAL* ones = NULL; // TODO: Memory leak, need to free at some point.

KernelLauncher::KernelLauncher(GPUInterface* inGpu) {
    // TODO: copy the gpu instance?
    gpu = inGpu;

    fMatrixMulADB =
            gpu->GetFunction("kernelMatrixMulADB");
    
    fPartialsPartialsByPatternBlockCoherent =
            gpu->GetFunction("kernelPartialsPartialsByPatternBlockCoherent");

    fPartialsPartialsByPatternBlockFixedScaling =
            gpu->GetFunction("kernelPartialsPartialsByPatternBlockFixedScaling");
    
    fStatesPartialsByPatternBlockCoherent =
            gpu->GetFunction("kernelStatesPartialsByPatternBlockCoherent");
    
    fStatesPartialsByPatternBlockFixedScaling =
            gpu->GetFunction("kernelStatesPartialsByPatternBlockFixedScaling");
    
    fStatesStatesByPatternBlockCoherent =
            gpu->GetFunction("kernelStatesStatesByPatternBlockCoherent");
    
    fStatesStatesByPatternBlockFixedScaling =
            gpu->GetFunction("kernelStatesStatesByPatternBlockFixedScaling");
    
    fPartialsPartialsEdgeLikelihoods =
            gpu->GetFunction("kernelPartialsPartialsEdgeLikelihoods");
    
    fStatesPartialsEdgeLikelihoods =
            gpu->GetFunction("kernelStatesPartialsEdgeLikelihoods");
    
#if (PADDED_STATE_COUNT == 4)
    fPartialsPartialsByPatternBlockCoherentSmall = 
            gpu->GetFunction("kernelPartialsPartialsByPatternBlockCoherentSmall");
    
    fPartialsPartialsByPatternBlockSmallFixedScaling =
            gpu->GetFunction("kernelPartialsPartialsByPatternBlockSmallFixedScaling");
    
    fStatesPartialsByPatternBlockCoherentSmall =
            gpu->GetFunction("kernelStatesPartialsByPatternBlockCoherentSmall");
    
    fStatesStatesByPatternBlockCoherentSmall =
            gpu->GetFunction("kernelStatesStatesByPatternBlockCoherentSmall");
    
    fPartialsPartialsEdgeLikelihoodsSmall =
            gpu->GetFunction("kernelPartialsPartialsEdgeLikelihoodsSmall");
    
    fStatesPartialsEdgeLikelihoodsSmall =
            gpu->GetFunction("kernelStatesPartialsEdgeLikelihoodsSmall");
#endif // PADDED_STATE_COUNT == 4
    
    fIntegrateLikelihoodsDynamicScaling =
            gpu->GetFunction("kernelIntegrateLikelihoodsDynamicScaling");
    
    fAccumulateFactorsDynamicScaling =
            gpu->GetFunction("kernelAccumulateFactorsDynamicScaling");

    fRemoveFactorsDynamicScaling =
        gpu->GetFunction("kernelRemoveFactorsDynamicScaling");
    
    fPartialsDynamicScaling =
            gpu->GetFunction("kernelPartialsDynamicScaling");
    
    fPartialsDynamicScalingAccumulate =
            gpu->GetFunction("kernelPartialsDynamicScalingAccumulate");
    
    fPartialsDynamicScalingSlow =
            gpu->GetFunction("kernelPartialsDynamicScalingSlow");
    
    fIntegrateLikelihoods =
            gpu->GetFunction("kernelIntegrateLikelihoods");
}

KernelLauncher::~KernelLauncher() {
}

#ifdef CUDA
void KernelLauncher::GetTransitionProbabilitiesSquare(GPUPtr dPtrQueue,
                                                      GPUPtr dEvec,
                                                      GPUPtr dIevc,
                                                      GPUPtr dEigenValues,
                                                      GPUPtr distanceQueue,
                                                      int totalMatrix) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::GetTransitionProbabilitiesSquare\n");
#endif
    
    Dim3Int block(MULTIPLY_BLOCK_SIZE, MULTIPLY_BLOCK_SIZE);
    Dim3Int grid(PADDED_STATE_COUNT / MULTIPLY_BLOCK_SIZE,
                 PADDED_STATE_COUNT / MULTIPLY_BLOCK_SIZE);
    if (PADDED_STATE_COUNT % MULTIPLY_BLOCK_SIZE != 0) {
        grid.x += 1;
        grid.y += 1;
    }

    grid.x *= totalMatrix;

    // Transposed (interchanged Ievc and Evec)    
    int parameterCount = 8;
    gpu->LaunchKernelIntParams(fMatrixMulADB,
                               block, grid,
                               parameterCount,
                               dPtrQueue, dIevc, dEigenValues, dEvec, distanceQueue,
                               PADDED_STATE_COUNT, PADDED_STATE_COUNT, totalMatrix);
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::GetTransitionProbabilitiesSquare\n");
#endif
}

#else //OpenCL
void KernelLauncher::GetTransitionProbabilitiesSquare(GPUPtr dPtr,
                                                      GPUPtr dEvec,
                                                      GPUPtr dIevc,
                                                      GPUPtr dEigenValues,
                                                      GPUPtr distanceQueue,
                                                      int totalMatrix,
                                                      int index) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::GetTransitionProbabilitiesSquare\n");
#endif
    
    Dim3Int block(MULTIPLY_BLOCK_SIZE, MULTIPLY_BLOCK_SIZE);
    Dim3Int grid(PADDED_STATE_COUNT / MULTIPLY_BLOCK_SIZE,
                 PADDED_STATE_COUNT / MULTIPLY_BLOCK_SIZE);
    if (PADDED_STATE_COUNT % MULTIPLY_BLOCK_SIZE != 0) {
        grid.x += 1;
        grid.y += 1;
    }
    
    grid.x *= totalMatrix;
    
    // Transposed (interchanged Ievc and Evec)    
    int parameterCount = 9;
    gpu->LaunchKernelIntParams(fMatrixMulADB,
                               block, grid,
                               parameterCount,
                               dPtr, dIevc, dEigenValues, dEvec, distanceQueue,
                               PADDED_STATE_COUNT, PADDED_STATE_COUNT, totalMatrix,
                               index);
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::GetTransitionProbabilitiesSquare\n");
#endif
}
#endif


void KernelLauncher::PartialsPartialsPruningDynamicScaling(GPUPtr partials1,
                                                           GPUPtr partials2,
                                                           GPUPtr partials3,
                                                           GPUPtr matrices1,
                                                           GPUPtr matrices2,
                                                           GPUPtr scalingFactors,
                                                           GPUPtr cumulativeScaling,
                                                           const unsigned int patternCount,
                                                           const unsigned int categoryCount,
                                                           int doRescaling) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::PartialsPartialsPruningDynamicScaling\n");
#endif
    
#if (PADDED_STATE_COUNT == 4)
    Dim3Int block(16, PATTERN_BLOCK_SIZE);
    Dim3Int grid(patternCount / (PATTERN_BLOCK_SIZE * 4), categoryCount);
    if (patternCount % (PATTERN_BLOCK_SIZE * 4) != 0)
        grid.x += 1;
#else
    Dim3Int block(PADDED_STATE_COUNT, PATTERN_BLOCK_SIZE);
    Dim3Int grid(patternCount / PATTERN_BLOCK_SIZE, categoryCount);
    if (patternCount % PATTERN_BLOCK_SIZE != 0)
        grid.x += 1;
#endif    
    if (doRescaling != 0)    {
        // Compute partials without any rescaling
#if (PADDED_STATE_COUNT == 4)        
        GPUFunction fHandle = fPartialsPartialsByPatternBlockCoherentSmall;
#else
        GPUFunction fHandle = fPartialsPartialsByPatternBlockCoherent;
#endif
        
        int parameterCount = 6;
        gpu->LaunchKernelIntParams(fHandle,
                                   block, grid,
                                   parameterCount,
                                   partials1, partials2, partials3, matrices1, matrices2,
                                   patternCount);
        
        // Rescale partials and save scaling factors
        if (doRescaling > 0) {
            gpu->Synchronize();
            KernelLauncher::RescalePartials(partials3, scalingFactors, cumulativeScaling,
                                            patternCount, categoryCount, 0);
        }
        
    } else {
        
        // Compute partials with known rescalings
#if (PADDED_STATE_COUNT == 4)        
        GPUFunction fHandle = fPartialsPartialsByPatternBlockSmallFixedScaling;        
#else
        GPUFunction fHandle = fPartialsPartialsByPatternBlockFixedScaling;   
#endif
        
        int parameterCount = 7;
        gpu->LaunchKernelIntParams(fHandle,
                                   block, grid,
                                   parameterCount,
                                   partials1, partials2, partials3, matrices1, matrices2,
                                   scalingFactors, patternCount);        
    }
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::PartialsPartialsPruningDynamicScaling\n");
#endif
    
}


void KernelLauncher::StatesPartialsPruningDynamicScaling(GPUPtr states1,
                                                         GPUPtr partials2,
                                                         GPUPtr partials3,
                                                         GPUPtr matrices1,
                                                         GPUPtr matrices2,
                                                         GPUPtr scalingFactors,
                                                         GPUPtr cumulativeScaling,
                                                         const unsigned int patternCount,
                                                         const unsigned int categoryCount,
                                                         int doRescaling) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::StatesPartialsPruningDynamicScaling\n");
#endif    
    
#if (PADDED_STATE_COUNT == 4)
    Dim3Int block(16, PATTERN_BLOCK_SIZE);
    Dim3Int grid(patternCount / (PATTERN_BLOCK_SIZE * 4), categoryCount);
    if (patternCount % (PATTERN_BLOCK_SIZE * 4) != 0)
        grid.x += 1;
#else
    Dim3Int block(PADDED_STATE_COUNT, PATTERN_BLOCK_SIZE);
    Dim3Int grid(patternCount / PATTERN_BLOCK_SIZE, categoryCount);
    if (patternCount % PATTERN_BLOCK_SIZE != 0)
        grid.x += 1;
#endif
    
    if (doRescaling != 0)    {
        // Compute partials without any rescaling
#if (PADDED_STATE_COUNT == 4)
        GPUFunction fHandle = fStatesPartialsByPatternBlockCoherentSmall;
#else
        GPUFunction fHandle = fStatesPartialsByPatternBlockCoherent;
#endif
        int parameterCount = 6;
        gpu->LaunchKernelIntParams(fHandle,
                                   block, grid,
                                   parameterCount,
                                   states1, partials2, partials3, matrices1, matrices2,
                                   patternCount);
        
        // Rescale partials and save scaling factors
        if (doRescaling > 0) {
            gpu->Synchronize();
            KernelLauncher::RescalePartials(partials3, scalingFactors, cumulativeScaling,
                                            patternCount, categoryCount, 1);
        }
    } else {
        
        // Compute partials with known rescalings
#if (PADDED_STATE_COUNT == 4)        
        int parameterCount = 6;
        gpu->LaunchKernelIntParams(fStatesPartialsByPatternBlockCoherentSmall,
                                   block, grid,
                                   parameterCount,
                                   states1, partials2, partials3, matrices1, matrices2,
                                   patternCount);
#else        
        int parameterCount = 7;
        gpu->LaunchKernelIntParams(fStatesPartialsByPatternBlockFixedScaling,
                                   block, grid,
                                   parameterCount,
                                   states1, partials2, partials3, matrices1, matrices2,
                                   scalingFactors, patternCount);
#endif
    }
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\tLeaving  KernelLauncher::StatesPartialsPruningDynamicScaling\n");
#endif
    
}

void KernelLauncher::StatesStatesPruningDynamicScaling(GPUPtr states1,
                                                       GPUPtr states2,
                                                       GPUPtr partials3,
                                                       GPUPtr matrices1,
                                                       GPUPtr matrices2,
                                                       GPUPtr scalingFactors,
                                                       GPUPtr cumulativeScaling,
                                                       const unsigned int patternCount,
                                                       const unsigned int categoryCount,
                                                       int doRescaling) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::StatesStatesPruningDynamicScaling\n");
#endif    
    
    
#if (PADDED_STATE_COUNT == 4)
    Dim3Int block(16, PATTERN_BLOCK_SIZE);
    Dim3Int grid(patternCount / (PATTERN_BLOCK_SIZE * 4), categoryCount);
    if (patternCount % (PATTERN_BLOCK_SIZE * 4) != 0)
        grid.x += 1;
#else
    Dim3Int block(PADDED_STATE_COUNT, PATTERN_BLOCK_SIZE);
    Dim3Int grid(patternCount / PATTERN_BLOCK_SIZE, categoryCount);
    if (patternCount % PATTERN_BLOCK_SIZE != 0)
        grid.x += 1;
#endif
    
    if (doRescaling != 0)    {
        // Compute partials without any rescaling
#if (PADDED_STATE_COUNT == 4)
        GPUFunction fHandle = fStatesStatesByPatternBlockCoherentSmall;
#else
        GPUFunction fHandle = fStatesStatesByPatternBlockCoherent;
#endif
        int parameterCount = 6;
        gpu->LaunchKernelIntParams(fHandle,
                                   block, grid,
                                   parameterCount,
                                   states1, states2, partials3, matrices1, matrices2, patternCount);
        
        // Rescale partials and save scaling factors
        // If PADDED_STATE_COUNT == 4, just with ones.
        if (doRescaling > 0) {
            gpu->Synchronize();
            KernelLauncher::RescalePartials(partials3, scalingFactors, cumulativeScaling,
                                            patternCount, categoryCount, 1);
        }
        
    } else {
        
        // Compute partials with known rescalings
#if (PADDED_STATE_COUNT == 4)
        int parameterCount = 6;
        gpu->LaunchKernelIntParams(fStatesStatesByPatternBlockCoherentSmall,
                                   block, grid,
                                   parameterCount,
                                   states1, states2, partials3, matrices1, matrices2, patternCount);
#else
        int parameterCount = 7;
        gpu->LaunchKernelIntParams(fStatesStatesByPatternBlockFixedScaling,
                                   block, grid,
                                   parameterCount,
                                   states1, states2, partials3, matrices1, matrices2,
                                   scalingFactors, patternCount);
#endif
    }
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::StatesStatesPruningDynamicScaling\n");
#endif
}

void KernelLauncher::IntegrateLikelihoodsDynamicScaling(GPUPtr dResult,
                                                        GPUPtr dRootPartials,
                                                        GPUPtr dWeights,
                                                        GPUPtr dFrequencies,
                                                        GPUPtr dRootScalingFactors,
                                                        int patternCount,
                                                        int categoryCount) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::IntegrateLikelihoodsDynamicScaling\n");
#endif
    
    Dim3Int block(PADDED_STATE_COUNT);
    Dim3Int grid(patternCount);
    
    int parameterCount = 6;
    gpu->LaunchKernelIntParams(fIntegrateLikelihoodsDynamicScaling,
                               block, grid,
                               parameterCount,
                               dResult, dRootPartials, dWeights, dFrequencies, dRootScalingFactors,
                               categoryCount);    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::IntegrateLikelihoodsDynamicScaling\n");
#endif
}


void KernelLauncher::PartialsPartialsEdgeLikelihoods(GPUPtr dPartialsTmp,
                                                     GPUPtr dParentPartials,
                                                     GPUPtr dChildParials,
                                                     GPUPtr dTransMatrix,
                                                     int patternCount,
                                                     int categoryCount) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\tEntering KernelLauncher::PartialsPartialsEdgeLikelihoods\n");
#endif
    
#if (PADDED_STATE_COUNT == 4)
    Dim3Int block(16, PATTERN_BLOCK_SIZE);
    Dim3Int grid(patternCount / (PATTERN_BLOCK_SIZE * 4), categoryCount);
    if (patternCount % (PATTERN_BLOCK_SIZE * 4) != 0)
        grid.x += 1;
    
    GPUFunction fHandle = fPartialsPartialsEdgeLikelihoodsSmall;
#else
    Dim3Int block(PADDED_STATE_COUNT, PATTERN_BLOCK_SIZE);
    Dim3Int grid(patternCount / PATTERN_BLOCK_SIZE, categoryCount);
    if (patternCount % PATTERN_BLOCK_SIZE != 0)
        grid.x += 1;
    
    // TODO: test kernelPartialsPartialsEdgeLikelihoods    
    GPUFunction fHandle = fPartialsPartialsEdgeLikelihoods;
#endif
    
    int parameterCount = 5;
    gpu->LaunchKernelIntParams(fHandle,
                               block, grid,
                               parameterCount,
                               dPartialsTmp, dParentPartials, dChildParials, dTransMatrix,
                               patternCount);
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::PartialsPartialsEdgeLikelihoods\n");
#endif
    
}

void KernelLauncher::StatesPartialsEdgeLikelihoods(GPUPtr dPartialsTmp,
                                                   GPUPtr dParentPartials,
                                                   GPUPtr dChildStates,
                                                   GPUPtr dTransMatrix,
                                                   int patternCount,
                                                   int categoryCount) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\tEntering KernelLauncher::StatesPartialsEdgeLikelihoods\n");
#endif
    
    // TODO: test KernelLauncher::StatesPartialsEdgeLikelihoods
    
#if (PADDED_STATE_COUNT == 4)
    Dim3Int block(16, PATTERN_BLOCK_SIZE);
    Dim3Int grid(patternCount / (PATTERN_BLOCK_SIZE * 4), categoryCount);
    if (patternCount % (PATTERN_BLOCK_SIZE * 4) != 0)
        grid.x += 1;
    
    GPUFunction fHandle = fStatesPartialsEdgeLikelihoodsSmall;
#else
    Dim3Int block(PADDED_STATE_COUNT, PATTERN_BLOCK_SIZE);
    Dim3Int grid(patternCount / PATTERN_BLOCK_SIZE, categoryCount);
    if (patternCount % PATTERN_BLOCK_SIZE != 0)
        grid.x += 1;
    
    GPUFunction fHandle = fStatesPartialsEdgeLikelihoods;    
#endif
    
    int parameterCount = 5;
    gpu->LaunchKernelIntParams(fHandle,
                               block, grid,
                               parameterCount,
                               dPartialsTmp, dParentPartials, dChildStates, dTransMatrix,
                               patternCount);  
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::StatesPartialsEdgeLikelihoods\n");
#endif
    
}

void KernelLauncher::AccumulateFactorsDynamicScaling(GPUPtr dNodePtrQueue,
                                               GPUPtr dRootScalingFactors,
                                               int nodeCount,
                                               int patternCount) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::AccumulateFactorsDynamicScaling\n");
#endif
    
    Dim3Int block(PATTERN_BLOCK_SIZE);
    Dim3Int grid(patternCount / PATTERN_BLOCK_SIZE);
    if (patternCount % PATTERN_BLOCK_SIZE != 0)
        grid.x += 1;
    
    int parameterCount = 4;
    gpu->LaunchKernelIntParams(fAccumulateFactorsDynamicScaling,
                               block, grid,
                               parameterCount,
                               dNodePtrQueue, dRootScalingFactors, nodeCount, patternCount);
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::AccumulateFactorsDynamicScaling\n");
#endif
    
}

void KernelLauncher::RemoveFactorsDynamicScaling(GPUPtr dNodePtrQueue,
                                                     GPUPtr dRootScalingFactors,
                                                     int nodeCount,
                                                     int patternCount) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::RemoveFactorsDynamicScaling\n");
#endif    
    
    Dim3Int block(PATTERN_BLOCK_SIZE);
    Dim3Int grid(patternCount / PATTERN_BLOCK_SIZE);
    if (patternCount % PATTERN_BLOCK_SIZE != 0)
        grid.x += 1;
    
    int parameterCount = 4;
    gpu->LaunchKernelIntParams(fRemoveFactorsDynamicScaling,
                               block, grid,
                               parameterCount,
                               dNodePtrQueue, dRootScalingFactors, nodeCount, patternCount);

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::RemoveFactorsDynamicScaling\n");
#endif        

}


void KernelLauncher::RescalePartials(GPUPtr partials3,
                                     GPUPtr scalingFactors,
                                     GPUPtr cumulativeScaling, 
                                     int patternCount,
                                     int categoryCount,
                                     int fillWithOnes) {
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::RescalePartials\n");
#endif    
    
    // TODO: remove fillWithOnes and leave it up to client?
    
    // Rescale partials and save scaling factors
    //#if (PADDED_STATE_COUNT == 4) 
    if (fillWithOnes != 0) {
        if (ones == NULL) {
            ones = (REAL*) malloc(SIZE_REAL * patternCount);
            for(int i = 0; i < patternCount; i++)
                ones[i] = 1.0;
        }
        gpu->MemcpyHostToDevice(scalingFactors, ones, SIZE_REAL * patternCount);
        return;
    }
    //#endif
    
#ifndef SLOW_REWEIGHING
    Dim3Int grid(patternCount, categoryCount / MATRIX_BLOCK_SIZE);
    if (categoryCount % MATRIX_BLOCK_SIZE != 0)
        grid.y += 1;
    if (grid.y > 1) {
        fprintf(stderr, "Not yet implemented! Try slow reweighing.\n");
        exit(0);
    }
    Dim3Int block(PADDED_STATE_COUNT, MATRIX_BLOCK_SIZE);
    // TODO: Totally incoherent for PADDED_STATE_COUNT == 4

    
    GPUFunction fHandle = fPartialsDynamicScaling;
    if (cumulativeScaling != 0)
        fHandle = fPartialsDynamicScalingAccumulate;
#else
    Dim3Int grid(patternCount, 1);
    Dim3Int block(PADDED_STATE_COUNT);
    
    GPUFunction fHandle = fPartialsDynamicScalingSlow;
    
    // TODO: add support for accumulate scaling as you rescale for SLOW_REWEIGHING
#endif

    if (cumulativeScaling != 0) {
        int parameterCount = 4;
        gpu->LaunchKernelIntParams(fHandle,
                                   block, grid,
                                   parameterCount,
                                   partials3, scalingFactors, cumulativeScaling,
                                   categoryCount);
    } else {
        int parameterCount = 3;
        gpu->LaunchKernelIntParams(fHandle,
                                   block, grid,
                                   parameterCount,
                                   partials3, scalingFactors, categoryCount);        
    }
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::RescalePartials\n");
#endif        
}

void KernelLauncher::IntegrateLikelihoods(GPUPtr dResult,
                                          GPUPtr dRootPartials,
                                          GPUPtr dWeights,
                                          GPUPtr dFrequencies,
                                          int patternCount,
                                          int categoryCount) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\tEntering KernelLauncher::IntegrateLikelihoods\n");
#endif
    
    Dim3Int block(PADDED_STATE_COUNT);
    Dim3Int grid(patternCount);
    
    int parameterCount = 5;
    gpu->LaunchKernelIntParams(fIntegrateLikelihoods,
                               block, grid,
                               parameterCount,
                               dResult, dRootPartials, dWeights, dFrequencies, categoryCount);
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::IntegrateLikelihoods\n");
#endif
    
}
