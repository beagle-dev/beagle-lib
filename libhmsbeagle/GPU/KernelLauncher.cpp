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
    SetupKernelBlocksAndGrids(); // Delegate, so can be overridden as necessary
    LoadKernels(); // Delegate, so can be overridden as necessary
}

KernelLauncher::~KernelLauncher() {
}

void KernelLauncher::SetupKernelBlocksAndGrids() {

    kPaddedStateCount = gpu->kernelResource->paddedStateCount;
    kCategoryCount = gpu->kernelResource->categoryCount;
    kPatternCount = gpu->kernelResource->patternCount;
    kMultiplyBlockSize = gpu->kernelResource->multiplyBlockSize;
    kPatternBlockSize = gpu->kernelResource->patternBlockSize;
    kSlowReweighing = gpu->kernelResource->slowReweighing;
    kMatrixBlockSize = gpu->kernelResource->matrixBlockSize;
    
    // Set up block/grid for transition matrices computation
    bgTransitionProbabilitiesBlock = Dim3Int(kMultiplyBlockSize, kMultiplyBlockSize);
    bgTransitionProbabilitiesGrid = Dim3Int(
            kPaddedStateCount/kMultiplyBlockSize, 
            kPaddedStateCount/kMultiplyBlockSize);
    if(kPaddedStateCount % kMultiplyBlockSize != 0) {
        bgTransitionProbabilitiesGrid.x += 1;
        bgTransitionProbabilitiesGrid.y += 1;
    }
    
    // Set up block/grid for peeling computation
    if (kPaddedStateCount == 4) {
        bgPeelingBlock = Dim3Int(16, kPatternBlockSize);
        bgPeelingGrid  = Dim3Int(kPatternCount / (kPatternBlockSize * 4), kCategoryCount);
        if (kPatternCount % (kPatternBlockSize * 4) != 0)
            bgPeelingGrid.x += 1;
    } else {
        bgPeelingBlock = Dim3Int(kPaddedStateCount, kPatternBlockSize);
        bgPeelingGrid  = Dim3Int(kPatternCount / kPatternBlockSize, kCategoryCount);
        if (kPatternCount % kPatternBlockSize != 0)
            bgPeelingGrid.x += 1;
    } 
    
    // Set up block/grid for likelihood computation
    if (kPaddedStateCount == 4) {
        int likePatternBlockSize = kPatternBlockSize;
        bgLikelihoodBlock = Dim3Int(4,likePatternBlockSize);
        bgLikelihoodGrid = Dim3Int(kPatternCount/likePatternBlockSize);
        if (kPatternCount % likePatternBlockSize != 0)
            bgLikelihoodGrid.x += 1;
    } else {
        bgLikelihoodBlock = Dim3Int(kPaddedStateCount);
        bgLikelihoodGrid  = Dim3Int(kPatternCount);
    }
    
    // Set up block/grid for scale factor accumulation
    bgAccumulateBlock = Dim3Int(kPatternBlockSize);
    bgAccumulateGrid  = Dim3Int(kPatternCount / kPatternBlockSize);
    if (kPatternCount % kPatternBlockSize != 0)
        bgAccumulateGrid.x += 1;
 
    // Set up block/grid for scaling partials
    if (kSlowReweighing) {
        bgScaleBlock = Dim3Int(kPaddedStateCount);
        bgScaleGrid  = Dim3Int(kPatternCount);        
    } else {
        if (kPaddedStateCount == 4) {
            bgScaleBlock = Dim3Int(16, kMatrixBlockSize);
            bgScaleGrid  = Dim3Int(kPatternCount / 4, kCategoryCount/kMatrixBlockSize);
            if (kPatternCount % 4 != 0) {
                bgScaleGrid.x += 1; // 
                fprintf(stderr,"PATTERNS SHOULD BE PADDED! Inform Marc, please.\n");
                exit(-1);
            }
        } else { 
            bgScaleBlock = Dim3Int(kPaddedStateCount, kMatrixBlockSize);
            bgScaleGrid  = Dim3Int(kPatternCount, kCategoryCount/kMatrixBlockSize);
        }
        if (kCategoryCount % kMatrixBlockSize != 0)
            bgScaleGrid.y += 1;
        if (bgScaleGrid.y > 1) { 
            fprintf(stderr, "Not yet implemented! Try slow reweighing.\n");
            exit(-1);
        }        
    }
}

void KernelLauncher::LoadKernels() {
    fMatrixMulADB = gpu->GetFunction("kernelMatrixMulADB");

    fPartialsPartialsByPatternBlockCoherent = gpu->GetFunction(
            "kernelPartialsPartialsNoScale");

    fPartialsPartialsByPatternBlockFixedScaling = gpu->GetFunction(
            "kernelPartialsPartialsFixedScale");

    fStatesPartialsByPatternBlockCoherent = gpu->GetFunction(
            "kernelStatesPartialsNoScale");

    fStatesStatesByPatternBlockCoherent = gpu->GetFunction(
            "kernelStatesStatesNoScale");

    if (kPaddedStateCount != 4) {
        fStatesPartialsByPatternBlockFixedScaling = gpu->GetFunction(
                "kernelStatesPartialsFixedScale");

        fStatesStatesByPatternBlockFixedScaling = gpu->GetFunction(
                "kernelStatesStatesFixedScale");
    }

    fPartialsPartialsEdgeLikelihoods = gpu->GetFunction(
            "kernelPartialsPartialsEdgeLikelihoods");

    fStatesPartialsEdgeLikelihoods = gpu->GetFunction(
            "kernelStatesPartialsEdgeLikelihoods");

    fIntegrateLikelihoodsDynamicScaling = gpu->GetFunction(
            "kernelIntegrateLikelihoodsFixedScale");

    fAccumulateFactorsDynamicScaling = gpu->GetFunction(
            "kernelAccumulateFactors");

    fRemoveFactorsDynamicScaling = gpu->GetFunction("kernelRemoveFactors");

    if (!kSlowReweighing) {
        fPartialsDynamicScaling = gpu->GetFunction(
                "kernelPartialsDynamicScaling");

        fPartialsDynamicScalingAccumulate = gpu->GetFunction(
                "kernelPartialsDynamicScalingAccumulate");
    } else {
        fPartialsDynamicScaling = gpu->GetFunction(
                "kernelPartialsDynamicScalingSlow");

        fPartialsDynamicScalingAccumulate = gpu->GetFunction(
                "kernelPartialsDynamicScalingAccumulate"); // TODO Write kernel 
    }

    fIntegrateLikelihoods = gpu->GetFunction("kernelIntegrateLikelihoods");
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

   bgTransitionProbabilitiesGrid.x *= totalMatrix;

    // Transposed (interchanged Ievc and Evec)    
    int parameterCount = 8;
    gpu->LaunchKernelIntParams(fMatrixMulADB,
                               bgTransitionProbabilitiesBlock, bgTransitionProbabilitiesGrid,
                               parameterCount,
                               dPtrQueue, dIevc, dEigenValues, dEvec, distanceQueue,
                               kPaddedStateCount, kPaddedStateCount,
                               totalMatrix);

    bgTransitionProbabilitiesGrid.x /= totalMatrix; // Reset value

    
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
    
    Dim3Int block(kMultiplyBlockSize, kMultiplyBlockSize); // TODO Construct once
    Dim3Int grid(kPaddedStateCount / kMultiplyBlockSize,
            kPaddedStateCount / kMultiplyBlockSize);
    if (kPaddedStateCount % kMultiplyBlockSize != 0) {
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
                               kPaddedStateCount, kPaddedStateCount, totalMatrix,
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
    
    if (doRescaling != 0)    {
        
        // Compute partials without any rescaling        
        gpu->LaunchKernelIntParams(fPartialsPartialsByPatternBlockCoherent,
                                   bgPeelingBlock, bgPeelingGrid,
                                   6,
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
        gpu->LaunchKernelIntParams(fPartialsPartialsByPatternBlockFixedScaling,
                                   bgPeelingBlock, bgPeelingGrid,
                                   7,
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
           
    if (doRescaling != 0)    {
        
        // Compute partials without any rescaling
        gpu->LaunchKernelIntParams(fStatesPartialsByPatternBlockCoherent,
                                   bgPeelingBlock, bgPeelingGrid,
                                   6,
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
        if (kPaddedStateCount == 4) { // Ignore rescaling
            
            gpu->LaunchKernelIntParams(fStatesPartialsByPatternBlockCoherent,
                                       bgPeelingBlock, bgPeelingGrid,
                                       6,
                                       states1, partials2, partials3, matrices1, matrices2,
                                       patternCount);
        } else {       

            gpu->LaunchKernelIntParams(fStatesPartialsByPatternBlockFixedScaling,
                                   bgPeelingBlock, bgPeelingGrid,
                                   7,
                                   states1, partials2, partials3, matrices1, matrices2,
                                   scalingFactors, patternCount);
        }
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
       
    if (doRescaling != 0)    {
        
        // Compute partials without any rescaling
        gpu->LaunchKernelIntParams(fStatesStatesByPatternBlockCoherent,
                                   bgPeelingBlock, bgPeelingGrid,
                                   6,
                                   states1, states2, partials3, matrices1, matrices2, patternCount);
        
        // Rescale partials and save scaling factors     
        if (doRescaling > 0) {
            gpu->Synchronize();
            KernelLauncher::RescalePartials(partials3, scalingFactors, cumulativeScaling,
                                            patternCount, categoryCount, 1);
        }
        
    } else {
        
        // Compute partials with known rescalings
        if (kPaddedStateCount == 4) {

            gpu->LaunchKernelIntParams(fStatesStatesByPatternBlockCoherent,
                                   bgPeelingBlock, bgPeelingGrid,
                                   6,
                                   states1, states2, partials3, matrices1, matrices2, patternCount);
        } else {

            gpu->LaunchKernelIntParams(fStatesStatesByPatternBlockFixedScaling,
                                   bgPeelingBlock, bgPeelingGrid,
                                   7,
                                   states1, states2, partials3, matrices1, matrices2,
                                   scalingFactors, patternCount);
        }
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
    
    gpu->LaunchKernelIntParams(fIntegrateLikelihoodsDynamicScaling,
                               bgLikelihoodBlock, bgLikelihoodGrid,
                               7,
                               dResult, dRootPartials, dWeights, dFrequencies, dRootScalingFactors,
                               categoryCount,patternCount);    
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
        
    gpu->LaunchKernelIntParams(fPartialsPartialsEdgeLikelihoods,
                               bgPeelingBlock, bgPeelingGrid,
                               5,
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
    
    gpu->LaunchKernelIntParams(fStatesPartialsEdgeLikelihoods,
                               bgPeelingBlock, bgPeelingGrid,
                               5,
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
    
    int parameterCount = 4;
    gpu->LaunchKernelIntParams(fAccumulateFactorsDynamicScaling,
                               bgAccumulateBlock, bgAccumulateGrid,
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
       
    int parameterCount = 4;
    gpu->LaunchKernelIntParams(fRemoveFactorsDynamicScaling,
                               bgAccumulateBlock, bgAccumulateGrid,
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
    if (kPaddedStateCount == 4) {
        if (fillWithOnes != 0) {
            if (ones == NULL) {
                ones = (REAL*) malloc(SIZE_REAL * patternCount);
                for(int i = 0; i < patternCount; i++)
                    ones[i] = 1.0;
            }
            gpu->MemcpyHostToDevice(scalingFactors, ones, SIZE_REAL * patternCount);
            return;
        }
    }
        
    // TODO: Totally incoherent for kPaddedStateCount == 4
        
    if (cumulativeScaling != 0) {
        
        if (kSlowReweighing) {        
            fprintf(stderr,"Simultaneous slow reweighing and accumulation is not yet implemented.\n");
            exit(-1);
            // TODO: add support for accumulate scaling as you rescale for SLOW_REWEIGHING                
        }
        
        int parameterCount = 4;
        gpu->LaunchKernelIntParams(fPartialsDynamicScalingAccumulate,
                                   bgScaleBlock, bgScaleGrid,
                                   parameterCount,
                                   partials3, scalingFactors, cumulativeScaling,
                                   categoryCount);
    } else {
        int parameterCount = 3;     
        gpu->LaunchKernelIntParams(fPartialsDynamicScaling,
                                   bgScaleBlock, bgScaleGrid,
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
   
    int parameterCount = 6;
    gpu->LaunchKernelIntParams(fIntegrateLikelihoods,
                               bgLikelihoodBlock, bgLikelihoodGrid,
                               parameterCount,
                               dResult, dRootPartials, dWeights, dFrequencies, 
                               categoryCount, patternCount);
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::IntegrateLikelihoods\n");
#endif
    
}
