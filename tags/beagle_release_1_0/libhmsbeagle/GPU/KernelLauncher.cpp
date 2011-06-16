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

#include "libhmsbeagle/beagle.h"
#include "libhmsbeagle/GPU/GPUImplDefs.h"
#include "libhmsbeagle/GPU/KernelLauncher.h"

/**************CODE***********/

REAL* ones = NULL; // TODO: Memory leak, need to free at some point.

KernelLauncher::KernelLauncher(GPUInterface* inGpu) {
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
    kSumSitesBlockSize = SUM_SITES_BLOCK_SIZE;
    kFlags = gpu->kernelResource->flags;
    
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
    if (kFlags & BEAGLE_FLAG_SCALING_AUTO)
        bgAccumulateGrid  = Dim3Int(kPatternCount / kPatternBlockSize, kCategoryCount);
    else
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
    
    // Set up block/grid for site likelihood accumulation
    bgSumSitesBlock = Dim3Int(kSumSitesBlockSize);
    bgSumSitesGrid  = Dim3Int(kPatternCount / kSumSitesBlockSize);
    if (kPatternCount % kSumSitesBlockSize != 0)
        bgSumSitesGrid.x += 1;
    
}

void KernelLauncher::LoadKernels() {
	
	if (kFlags & BEAGLE_FLAG_EIGEN_COMPLEX) {
		fMatrixMulADB = gpu->GetFunction("kernelMatrixMulADBComplex");
	} else {
		fMatrixMulADB = gpu->GetFunction("kernelMatrixMulADB");
	}

    fMatrixMulADBFirstDeriv = gpu->GetFunction("kernelMatrixMulADBFirstDeriv");
    
    fMatrixMulADBSecondDeriv = gpu->GetFunction("kernelMatrixMulADBSecondDeriv");
    
    fPartialsPartialsByPatternBlockCoherent = gpu->GetFunction(
            "kernelPartialsPartialsNoScale");
    
    fPartialsPartialsByPatternBlockAutoScaling = gpu->GetFunction(
                "kernelPartialsPartialsAutoScale");

    fPartialsPartialsByPatternBlockFixedScaling = gpu->GetFunction(
            "kernelPartialsPartialsFixedScale");
    
    if (kPaddedStateCount == 4) { // TODO Temporary hack until kernels are written
    fPartialsPartialsByPatternBlockCheckScaling = gpu->GetFunction(
            "kernelPartialsPartialsCheckScale");

   
    fPartialsPartialsByPatternBlockFixedCheckScaling = gpu->GetFunction(
           "kernelPartialsPartialsFixedCheckScale");
    }
    
    fStatesPartialsByPatternBlockCoherent = gpu->GetFunction(
            "kernelStatesPartialsNoScale");

    fStatesStatesByPatternBlockCoherent = gpu->GetFunction(
            "kernelStatesStatesNoScale");

//    if (kPaddedStateCount != 4) {
        fStatesPartialsByPatternBlockFixedScaling = gpu->GetFunction(
                "kernelStatesPartialsFixedScale");

        fStatesStatesByPatternBlockFixedScaling = gpu->GetFunction(
                "kernelStatesStatesFixedScale");
//    }

    fPartialsPartialsEdgeLikelihoods = gpu->GetFunction(
            "kernelPartialsPartialsEdgeLikelihoods");
    
    fPartialsPartialsEdgeLikelihoodsSecondDeriv = gpu->GetFunction(
            "kernelPartialsPartialsEdgeLikelihoodsSecondDeriv");

    fStatesPartialsEdgeLikelihoods = gpu->GetFunction(
            "kernelStatesPartialsEdgeLikelihoods");

    fStatesPartialsEdgeLikelihoodsSecondDeriv = gpu->GetFunction(
            "kernelStatesPartialsEdgeLikelihoodsSecondDeriv");
    
    fIntegrateLikelihoodsDynamicScalingSecondDeriv = gpu->GetFunction(
            "kernelIntegrateLikelihoodsFixedScaleSecondDeriv");
    
    if (kFlags & BEAGLE_FLAG_SCALING_AUTO)
        fIntegrateLikelihoodsDynamicScaling = gpu->GetFunction("kernelIntegrateLikelihoodsAutoScaling");
    else
        fIntegrateLikelihoodsDynamicScaling = gpu->GetFunction(
                                                               "kernelIntegrateLikelihoodsFixedScale");    
        
    if (kFlags & BEAGLE_FLAG_SCALERS_LOG) {
        fAccumulateFactorsDynamicScaling = gpu->GetFunction(
                                                            "kernelAccumulateFactorsScalersLog");
        fRemoveFactorsDynamicScaling = gpu->GetFunction("kernelRemoveFactorsScalersLog");
    } else {
        fAccumulateFactorsDynamicScaling = gpu->GetFunction(
                                                            "kernelAccumulateFactors");
        fRemoveFactorsDynamicScaling = gpu->GetFunction("kernelRemoveFactors");
    }
    
    fAccumulateFactorsAutoScaling = gpu->GetFunction("kernelAccumulateFactorsAutoScaling");

    if (!kSlowReweighing) {
        if (kFlags & BEAGLE_FLAG_SCALERS_LOG) {
            fPartialsDynamicScaling = gpu->GetFunction(
                   "kernelPartialsDynamicScalingScalersLog");
            
            fPartialsDynamicScalingAccumulate = gpu->GetFunction(
                    "kernelPartialsDynamicScalingAccumulateScalersLog");            
        } else {
            fPartialsDynamicScaling = gpu->GetFunction(
                    "kernelPartialsDynamicScaling");

            fPartialsDynamicScalingAccumulate = gpu->GetFunction(
                    "kernelPartialsDynamicScalingAccumulate");            
        }
    } else {
        if (kFlags & BEAGLE_FLAG_SCALERS_LOG) {
            fPartialsDynamicScaling = gpu->GetFunction(
                    "kernelPartialsDynamicScalingSlowScalersLog");

            fPartialsDynamicScalingAccumulate = gpu->GetFunction(
                    "kernelPartialsDynamicScalingAccumulateScalersLog");            
        } else {
            fPartialsDynamicScaling = gpu->GetFunction(
                    "kernelPartialsDynamicScalingSlow");

            fPartialsDynamicScalingAccumulate = gpu->GetFunction(
                    "kernelPartialsDynamicScalingAccumulate"); // TODO Write kernel 
        }
    }
    
    if (kPaddedStateCount == 4) { // TODO Temporary
    fPartialsDynamicScalingAccumulateDifference = gpu->GetFunction(
           "kernelPartialsDynamicScalingAccumulateDifference");

    fPartialsDynamicScalingAccumulateReciprocal = gpu->GetFunction(
           "kernelPartialsDynamicScalingAccumulateReciprocal");
    }
    
    fIntegrateLikelihoods = gpu->GetFunction("kernelIntegrateLikelihoods");
    
    fIntegrateLikelihoodsSecondDeriv = gpu->GetFunction("kernelIntegrateLikelihoodsSecondDeriv");
	
    fIntegrateLikelihoodsMulti = gpu->GetFunction("kernelIntegrateLikelihoodsMulti");
	
	fIntegrateLikelihoodsFixedScaleMulti = gpu->GetFunction("kernelIntegrateLikelihoodsFixedScaleMulti");
    
    fSumSites1 = gpu->GetFunction("kernelSumSites1");
    fSumSites2 = gpu->GetFunction("kernelSumSites2");
    fSumSites3 = gpu->GetFunction("kernelSumSites3");
}

#ifdef CUDA
void KernelLauncher::GetTransitionProbabilitiesSquare(GPUPtr dMatrices,
                                                      GPUPtr dPtrQueue,
                                                      GPUPtr dEvec,
                                                      GPUPtr dIevc,
                                                      GPUPtr dEigenValues,
                                                      GPUPtr distanceQueue,
                                                      unsigned int totalMatrix) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::GetTransitionProbabilitiesSquare\n");
#endif

   bgTransitionProbabilitiesGrid.x *= totalMatrix;

    // Transposed (interchanged Ievc and Evec)    
    int parameterCountV = 6;
    int totalParameterCount = 9;
    gpu->LaunchKernel(fMatrixMulADB,
                               bgTransitionProbabilitiesBlock, bgTransitionProbabilitiesGrid,
                               parameterCountV, totalParameterCount,
                               dMatrices, dPtrQueue, dIevc, dEigenValues, dEvec, distanceQueue,
                               kPaddedStateCount, kPaddedStateCount,
                               totalMatrix);

    bgTransitionProbabilitiesGrid.x /= totalMatrix; // Reset value

    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::GetTransitionProbabilitiesSquare\n");
#endif
}

void KernelLauncher::GetTransitionProbabilitiesSquareFirstDeriv(GPUPtr dMatrices,
                                                                GPUPtr dPtrQueue,
                                                                 GPUPtr dEvec,
                                                                 GPUPtr dIevc,
                                                                 GPUPtr dEigenValues,
                                                                 GPUPtr distanceQueue,
                                                                 unsigned int totalMatrix) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::GetTransitionProbabilitiesSquareFirstDeriv\n");
#endif
    
    bgTransitionProbabilitiesGrid.x *= totalMatrix;
    
    // Transposed (interchanged Ievc and Evec)    
    int parameterCountV = 6;
    int totalParameterCount = 9;
    gpu->LaunchKernel(fMatrixMulADBFirstDeriv,
                               bgTransitionProbabilitiesBlock, bgTransitionProbabilitiesGrid,
                               parameterCountV, totalParameterCount,
                               dMatrices, dPtrQueue, dIevc, dEigenValues, dEvec, distanceQueue,
                               kPaddedStateCount, kPaddedStateCount,
                               totalMatrix);
    
    bgTransitionProbabilitiesGrid.x /= totalMatrix; // Reset value
    
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::GetTransitionProbabilitiesSquareFirstDeriv\n");
#endif
}

void KernelLauncher::GetTransitionProbabilitiesSquareSecondDeriv(GPUPtr dMatrices,
                                                                 GPUPtr dPtrQueue,
                                                      GPUPtr dEvec,
                                                      GPUPtr dIevc,
                                                      GPUPtr dEigenValues,
                                                      GPUPtr distanceQueue,
                                                      unsigned int totalMatrix) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::GetTransitionProbabilitiesSquareSecondDeriv\n");
#endif
    
    bgTransitionProbabilitiesGrid.x *= totalMatrix;
    
    // Transposed (interchanged Ievc and Evec)    
    int parameterCountV = 6;
    int totalParameterCount = 9;
    gpu->LaunchKernel(fMatrixMulADBSecondDeriv,
                               bgTransitionProbabilitiesBlock, bgTransitionProbabilitiesGrid,
                               parameterCountV, totalParameterCount,
                               dMatrices, dPtrQueue, dIevc, dEigenValues, dEvec, distanceQueue,
                               kPaddedStateCount, kPaddedStateCount,
                               totalMatrix);
    
    bgTransitionProbabilitiesGrid.x /= totalMatrix; // Reset value
    
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::GetTransitionProbabilitiesSquareSecondDeriv\n");
#endif
}


#else //OpenCL
void KernelLauncher::GetTransitionProbabilitiesSquare(GPUPtr dPtr,
                                                      GPUPtr dEvec,
                                                      GPUPtr dIevc,
                                                      GPUPtr dEigenValues,
                                                      GPUPtr distanceQueue,
                                                      unsigned int totalMatrix,
                                                      unsigned int index) {
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
    int parameterCountV = 5;
    int totalParameterCount = 9;
    gpu->LaunchKernel(fMatrixMulADB,
                               block, grid,
                               parameterCountV, totalParameterCount,
                               dPtr, dIevc, dEigenValues, dEvec, distanceQueue,
                               kPaddedStateCount, kPaddedStateCount, totalMatrix,
                               index);
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::GetTransitionProbabilitiesSquare\n");
#endif
}
#endif


void KernelLauncher::PartialsPartialsPruningDynamicCheckScaling(GPUPtr partials1,
                                                           GPUPtr partials2,
                                                           GPUPtr partials3,
                                                           GPUPtr matrices1,
                                                           GPUPtr matrices2,
                                                           int writeScalingIndex,
                                                           int readScalingIndex,
                                                           int cumulativeScalingIndex,
                                                           GPUPtr* dScalingFactors,
                                                           GPUPtr* dScalingFactorsMaster,
                                                           unsigned int patternCount,
                                                           unsigned int categoryCount,
                                                           int doRescaling,
                                                           int* hRescalingTrigger,
                                                           GPUPtr dRescalingTrigger,
                                                           int sizeReal) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::PartialsPartialsPruningDynamicCheckScaling\n");
#endif

    if (dScalingFactors[readScalingIndex] == 0) {
        *hRescalingTrigger = 0;
        // Compute partials without any rescaling but check values
        gpu->LaunchKernel(fPartialsPartialsByPatternBlockCheckScaling,
                          bgPeelingBlock, bgPeelingGrid,
                          6, 7,
                          partials1, partials2, partials3, matrices1, matrices2, dRescalingTrigger,
                          patternCount);            

        gpu->Synchronize();
//        printf("hRescalingTrigger (no factors) %d\n", *hRescalingTrigger);
        if (*hRescalingTrigger) { // check if any partials need rescaling
            if (dScalingFactors[writeScalingIndex] != dScalingFactorsMaster[writeScalingIndex])
                dScalingFactors[writeScalingIndex] = dScalingFactorsMaster[writeScalingIndex];

            if (dScalingFactors[writeScalingIndex] == 0) {
                dScalingFactors[writeScalingIndex] = gpu->AllocateMemory(patternCount * sizeReal);
                dScalingFactorsMaster[writeScalingIndex] = dScalingFactors[writeScalingIndex];
            }
            
            if (dScalingFactors[cumulativeScalingIndex] != dScalingFactorsMaster[cumulativeScalingIndex]) {
                gpu->MemcpyDeviceToDevice(dScalingFactorsMaster[cumulativeScalingIndex], dScalingFactors[cumulativeScalingIndex], sizeReal *patternCount);
                gpu->Synchronize();
                dScalingFactors[cumulativeScalingIndex] = dScalingFactorsMaster[cumulativeScalingIndex];
            }
            
            gpu->LaunchKernel(fPartialsDynamicScalingAccumulateReciprocal,
                              bgScaleBlock, bgScaleGrid,
                              3, 4,
                              partials3, dScalingFactors[writeScalingIndex], dScalingFactors[cumulativeScalingIndex],
                              categoryCount);
        }
    } else {
        *hRescalingTrigger = 0;
        // Compute partials with known rescalings        
        gpu->LaunchKernel(fPartialsPartialsByPatternBlockFixedCheckScaling,
                          bgPeelingBlock, bgPeelingGrid,
                          7, 8,
                          partials1, partials2, partials3, matrices1, matrices2,
                          dScalingFactors[readScalingIndex], dRescalingTrigger,
                          patternCount);        
        
        gpu->Synchronize();
//        printf("hRescalingTrigger (existing factors) %d\n", *hRescalingTrigger);
        if (*hRescalingTrigger) { // check if any partials need rescaling
            if (dScalingFactors[writeScalingIndex] != dScalingFactorsMaster[writeScalingIndex])
                dScalingFactors[writeScalingIndex] = dScalingFactorsMaster[writeScalingIndex];
            
            if (dScalingFactors[writeScalingIndex] == 0) {
                dScalingFactors[writeScalingIndex] = gpu->AllocateRealMemory(patternCount);
                dScalingFactorsMaster[writeScalingIndex] = dScalingFactors[writeScalingIndex];
            }
            
            if (dScalingFactors[cumulativeScalingIndex] != dScalingFactorsMaster[cumulativeScalingIndex]) {
                gpu->MemcpyDeviceToDevice(dScalingFactorsMaster[cumulativeScalingIndex], dScalingFactors[cumulativeScalingIndex], sizeReal * patternCount);
                gpu->Synchronize();
                dScalingFactors[cumulativeScalingIndex] = dScalingFactorsMaster[cumulativeScalingIndex];
            }
            
            gpu->LaunchKernel(fPartialsDynamicScalingAccumulateDifference,
                              bgScaleBlock, bgScaleGrid,
                              4, 5,
                              partials3, dScalingFactors[writeScalingIndex], dScalingFactors[readScalingIndex], dScalingFactors[cumulativeScalingIndex],
                              categoryCount);
        } else if (readScalingIndex != writeScalingIndex) {
            dScalingFactors[writeScalingIndex] = dScalingFactors[readScalingIndex];
        }
    }
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::PartialsPartialsPruningDynamicCheckScaling\n");
#endif
    
}

void KernelLauncher::PartialsPartialsPruningDynamicScaling(GPUPtr partials1,
                                                           GPUPtr partials2,
                                                           GPUPtr partials3,
                                                           GPUPtr matrices1,
                                                           GPUPtr matrices2,
                                                           GPUPtr scalingFactors,
                                                           GPUPtr cumulativeScaling,
                                                           unsigned int patternCount,
                                                           unsigned int categoryCount,
                                                           int doRescaling) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::PartialsPartialsPruningDynamicScaling\n");
#endif
    
    if (doRescaling == 2) { // auto-rescaling
        gpu->LaunchKernel(fPartialsPartialsByPatternBlockAutoScaling,
                          bgPeelingBlock, bgPeelingGrid,
                          6, 7,
                          partials1, partials2, partials3, matrices1, matrices2, scalingFactors,
                          patternCount);        
    } else if (doRescaling != 0) {
        
        // Compute partials without any rescaling        
        gpu->LaunchKernel(fPartialsPartialsByPatternBlockCoherent,
                          bgPeelingBlock, bgPeelingGrid,
                          5, 6,
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
        gpu->LaunchKernel(fPartialsPartialsByPatternBlockFixedScaling,
                          bgPeelingBlock, bgPeelingGrid,
                          6, 7,
                          partials1, partials2, partials3, matrices1, matrices2,
                          scalingFactors,
                          patternCount);        
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
                                                         unsigned int patternCount,
                                                         unsigned int categoryCount,
                                                         int doRescaling) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::StatesPartialsPruningDynamicScaling\n");
#endif    
           
    if (doRescaling != 0)    {
        
        // Compute partials without any rescaling
        gpu->LaunchKernel(fStatesPartialsByPatternBlockCoherent,
                                   bgPeelingBlock, bgPeelingGrid,
                                   5, 6,
                                   states1, partials2, partials3, matrices1, matrices2,
                                   patternCount);
        
        // Rescale partials and save scaling factors
        if (doRescaling > 0) {
            gpu->Synchronize();
            KernelLauncher::RescalePartials(partials3, scalingFactors, cumulativeScaling,
                                            patternCount, categoryCount,
#ifdef BEAGLE_FILL_4_STATE_SCALAR_SP
											1
#else
                                            0
#endif
                                            );
        }
    } else {
        
        // Compute partials with known rescalings
#ifdef BEAGLE_FILL_4_STATE_SCALAR_SP
        if (kPaddedStateCount == 4) { // Ignore rescaling

            gpu->LaunchKernel(fStatesPartialsByPatternBlockCoherent,
                                       bgPeelingBlock, bgPeelingGrid,
                                       5, 6,
                                       states1, partials2, partials3, matrices1, matrices2,
                                       patternCount);
        } else {
#endif

            gpu->LaunchKernel(fStatesPartialsByPatternBlockFixedScaling,
                                   bgPeelingBlock, bgPeelingGrid,
                                   6, 7,
                                   states1, partials2, partials3, matrices1, matrices2,
                                   scalingFactors,
                                   patternCount);
#ifdef BEAGLE_FILL_4_STATE_SCALAR_SP
        }
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
                                                       unsigned int patternCount,
                                                       unsigned int categoryCount,
                                                       int doRescaling) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::StatesStatesPruningDynamicScaling\n");
#endif    
       
    if (doRescaling != 0)    {
        
        // Compute partials without any rescaling
        gpu->LaunchKernel(fStatesStatesByPatternBlockCoherent,
                                   bgPeelingBlock, bgPeelingGrid,
                                   5, 6,
                                   states1, states2, partials3, matrices1, matrices2,
                                   patternCount);
        
        // Rescale partials and save scaling factors     
        if (doRescaling > 0) {
            gpu->Synchronize();
            KernelLauncher::RescalePartials(partials3, scalingFactors, cumulativeScaling,
                                            patternCount, categoryCount,
#ifdef BEAGLE_FILL_4_STATE_SCALAR_SS
                                            1
#else
                                            0
#endif
                                            );
        }
        
    } else {
        
        // Compute partials with known rescalings
#ifdef BEAGLE_FILL_4_STATE_SCALAR_SS
        if (kPaddedStateCount == 4) {

            gpu->LaunchKernel(fStatesStatesByPatternBlockCoherent,
                                   bgPeelingBlock, bgPeelingGrid,
                                   5, 6,
                                   states1, states2, partials3, matrices1, matrices2,
                                   patternCount);
        } else {
#endif

            gpu->LaunchKernel(fStatesStatesByPatternBlockFixedScaling,
                                   bgPeelingBlock, bgPeelingGrid,
                                   6, 7,
                                   states1, states2, partials3, matrices1, matrices2,
                                   scalingFactors,
                                   patternCount);
#ifdef BEAGLE_FILL_4_STATE_SCALAR_SS
        }
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
                                                        unsigned int patternCount,
                                                        unsigned int categoryCount) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::IntegrateLikelihoodsDynamicScaling\n");
#endif
    
    gpu->LaunchKernel(fIntegrateLikelihoodsDynamicScaling,
                               bgLikelihoodBlock, bgLikelihoodGrid,
                               5, 7,
                               dResult, dRootPartials, dWeights, dFrequencies, dRootScalingFactors,
                               categoryCount,patternCount);    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::IntegrateLikelihoodsDynamicScaling\n");
#endif
}

void KernelLauncher::IntegrateLikelihoodsDynamicScalingSecondDeriv(GPUPtr dResult,
                                                                   GPUPtr dFirstDerivResult,
                                                                   GPUPtr dSecondDerivResult,
                                                                   GPUPtr dRootPartials,
                                                                   GPUPtr dRootFirstDeriv,
                                                                   GPUPtr dRootSecondDeriv,
                                                                   GPUPtr dWeights,
                                                                   GPUPtr dFrequencies,
                                                                   GPUPtr dRootScalingFactors,
                                                                   unsigned int patternCount,
                                                                   unsigned int categoryCount) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::IntegrateLikelihoodsDynamicScalingSecondDeriv\n");
#endif
    
    gpu->LaunchKernel(fIntegrateLikelihoodsDynamicScalingSecondDeriv,
                               bgLikelihoodBlock, bgLikelihoodGrid,
                               9, 11,
                               dResult, dFirstDerivResult, dSecondDerivResult,
                               dRootPartials, dRootFirstDeriv, dRootSecondDeriv,
                               dWeights, dFrequencies, dRootScalingFactors,
                               categoryCount, patternCount);    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::IntegrateLikelihoodsDynamicScalingSecondDeriv\n");
#endif
}

void KernelLauncher::PartialsPartialsEdgeLikelihoods(GPUPtr dPartialsTmp,
                                                     GPUPtr dParentPartials,
                                                     GPUPtr dChildParials,
                                                     GPUPtr dTransMatrix,
                                                     unsigned int patternCount,
                                                     unsigned int categoryCount) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\tEntering KernelLauncher::PartialsPartialsEdgeLikelihoods\n");
#endif
        
    gpu->LaunchKernel(fPartialsPartialsEdgeLikelihoods,
                               bgPeelingBlock, bgPeelingGrid,
                               4, 5,
                               dPartialsTmp, dParentPartials, dChildParials, dTransMatrix,
                               patternCount);
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::PartialsPartialsEdgeLikelihoods\n");
#endif
    
}

void KernelLauncher::PartialsPartialsEdgeLikelihoodsSecondDeriv(GPUPtr dPartialsTmp,
                                                                GPUPtr dFirstDerivTmp,
                                                                GPUPtr dSecondDerivTmp,
                                                                GPUPtr dParentPartials,
                                                                GPUPtr dChildParials,
                                                                GPUPtr dTransMatrix,
                                                                GPUPtr dFirstDerivMatrix,
                                                                GPUPtr dSecondDerivMatrix,
                                                                unsigned int patternCount,
                                                                unsigned int categoryCount) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\tEntering KernelLauncher::PartialsPartialsEdgeLikelihoodsSecondDeriv\n");
#endif
    
    gpu->LaunchKernel(fPartialsPartialsEdgeLikelihoodsSecondDeriv,
                               bgPeelingBlock, bgPeelingGrid,
                               8, 9,
                               dPartialsTmp, dFirstDerivTmp, dSecondDerivTmp,
                               dParentPartials, dChildParials,
                               dTransMatrix, dFirstDerivMatrix, dSecondDerivMatrix,
                               patternCount);
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::PartialsPartialsEdgeLikelihoodsSecondDeriv\n");
#endif
    
}

void KernelLauncher::StatesPartialsEdgeLikelihoods(GPUPtr dPartialsTmp,
                                                   GPUPtr dParentPartials,
                                                   GPUPtr dChildStates,
                                                   GPUPtr dTransMatrix,
                                                   unsigned int patternCount,
                                                   unsigned int categoryCount) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\tEntering KernelLauncher::StatesPartialsEdgeLikelihoods\n");
#endif
    
    gpu->LaunchKernel(fStatesPartialsEdgeLikelihoods,
                               bgPeelingBlock, bgPeelingGrid,
                               4, 5,
                               dPartialsTmp, dParentPartials, dChildStates, dTransMatrix,
                               patternCount);  
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::StatesPartialsEdgeLikelihoods\n");
#endif
    
}

void KernelLauncher::StatesPartialsEdgeLikelihoodsSecondDeriv(GPUPtr dPartialsTmp,
                                                   GPUPtr dFirstDerivTmp,
                                                   GPUPtr dSecondDerivTmp,
                                                   GPUPtr dParentPartials,
                                                   GPUPtr dChildStates,
                                                   GPUPtr dTransMatrix,
                                                   GPUPtr dFirstDerivMatrix,
                                                   GPUPtr dSecondDerivMatrix,
                                                   unsigned int patternCount,
                                                   unsigned int categoryCount) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\tEntering KernelLauncher::StatesPartialsEdgeLikelihoodsSecondDeriv\n");
#endif
    
    gpu->LaunchKernel(fStatesPartialsEdgeLikelihoodsSecondDeriv,
                               bgPeelingBlock, bgPeelingGrid,
                               8, 9,
                               dPartialsTmp, dFirstDerivTmp, dSecondDerivTmp,
                               dParentPartials, dChildStates,
                               dTransMatrix, dFirstDerivMatrix, dSecondDerivMatrix,
                               patternCount);
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::StatesPartialsEdgeLikelihoodsSecondDeriv\n");
#endif
    
}

void KernelLauncher::AccumulateFactorsDynamicScaling(GPUPtr dScalingFactors,
                                                     GPUPtr dNodePtrQueue,
                                                     GPUPtr dRootScalingFactors,
                                                     unsigned int nodeCount,
                                                     unsigned int patternCount) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::AccumulateFactorsDynamicScaling\n");
#endif
    
    int parameterCountV = 3;
    int totalParameterCount = 5;
    gpu->LaunchKernel(fAccumulateFactorsDynamicScaling,
                               bgAccumulateBlock, bgAccumulateGrid,
                               parameterCountV, totalParameterCount,
                               dScalingFactors, dNodePtrQueue, dRootScalingFactors,
                               nodeCount, patternCount);
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::AccumulateFactorsDynamicScaling\n");
#endif
    
}

void KernelLauncher::AccumulateFactorsAutoScaling(GPUPtr dScalingFactors,
                                                  GPUPtr dNodePtrQueue,
                                                  GPUPtr dRootScalingFactors,
                                                  unsigned int nodeCount,
                                                  unsigned int patternCount,
                                                  unsigned int scaleBufferSize) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::AccumulateFactorsAutoScaling\n");
#endif
    
    int parameterCountV = 3;
    int totalParameterCount = 6;
    gpu->LaunchKernel(fAccumulateFactorsAutoScaling,
                      bgAccumulateBlock, bgAccumulateGrid,
                      parameterCountV, totalParameterCount,
                      dScalingFactors, dNodePtrQueue, dRootScalingFactors,
                      nodeCount, patternCount, scaleBufferSize);

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::AccumulateFactorsAutoScaling\n");
#endif
    
}


void KernelLauncher::RemoveFactorsDynamicScaling(GPUPtr dScalingFactors,
                                                 GPUPtr dNodePtrQueue,
                                                     GPUPtr dRootScalingFactors,
                                                     unsigned int nodeCount,
                                                     unsigned int patternCount) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::RemoveFactorsDynamicScaling\n");
#endif    
       
    int parameterCountV = 3;
    int totalParameterCount = 5;
    gpu->LaunchKernel(fRemoveFactorsDynamicScaling,
                               bgAccumulateBlock, bgAccumulateGrid,
                               parameterCountV, totalParameterCount,
                               dScalingFactors, dNodePtrQueue, dRootScalingFactors,
                               nodeCount, patternCount);

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::RemoveFactorsDynamicScaling\n");
#endif        

}


void KernelLauncher::RescalePartials(GPUPtr partials3,
                                     GPUPtr scalingFactors,
                                     GPUPtr cumulativeScaling, 
                                     unsigned int patternCount,
                                     unsigned int categoryCount,
                                     unsigned int fillWithOnes) {
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::RescalePartials\n");
#endif    
    
    // TODO: remove fillWithOnes and leave it up to client?
    
    // Rescale partials and save scaling factors
    if (kPaddedStateCount == 4) {
        if (fillWithOnes != 0) {
//            if (ones == NULL) {
//                ones = (REAL*) malloc(SIZE_REAL * patternCount);
//                if (kFlags & BEAGLE_FLAG_SCALERS_LOG) {
//                    for(int i = 0; i < patternCount; i++)
//                        ones[i] = 0.0;
//                } else {
//                    for(int i = 0; i < patternCount; i++)
//                        ones[i] = 1.0;
//                }
//            }
//            gpu->MemcpyHostToDevice(scalingFactors, ones, SIZE_REAL * patternCount);
//            return;
        	fprintf(stderr,"Old legacy code; should not get here!\n");
        	exit(0);
        }
    }
        
    // TODO: Totally incoherent for kPaddedStateCount == 4
        
    if (cumulativeScaling != 0) {
        
        if (kSlowReweighing) {        
            fprintf(stderr,"Simultaneous slow reweighing and accumulation is not yet implemented.\n");
            exit(-1);
            // TODO: add support for accumulate scaling as you rescale for SLOW_REWEIGHING                
        }
        
        int parameterCountV = 3;
        int totalParameterCount = 4;
        gpu->LaunchKernel(fPartialsDynamicScalingAccumulate,
                                   bgScaleBlock, bgScaleGrid,
                                   parameterCountV, totalParameterCount,
                                   partials3, scalingFactors, cumulativeScaling,
                                   categoryCount);
    } else {
        int parameterCountV = 2;
        int totalParameterCount = 3;
        gpu->LaunchKernel(fPartialsDynamicScaling,
                                   bgScaleBlock, bgScaleGrid,
                                   parameterCountV, totalParameterCount,
                                   partials3, scalingFactors,
                                   categoryCount);
    }
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::RescalePartials\n");
#endif
}


void KernelLauncher::IntegrateLikelihoods(GPUPtr dResult,
                                          GPUPtr dRootPartials,
                                          GPUPtr dWeights,
                                          GPUPtr dFrequencies,
                                          unsigned int patternCount,
                                          unsigned int categoryCount) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\tEntering KernelLauncher::IntegrateLikelihoods\n");
#endif
   
    int parameterCountV = 4;
    int totalParameterCount = 6;
    gpu->LaunchKernel(fIntegrateLikelihoods,
                               bgLikelihoodBlock, bgLikelihoodGrid,
                               parameterCountV, totalParameterCount,
                               dResult, dRootPartials, dWeights, dFrequencies,
                               categoryCount, patternCount);
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::IntegrateLikelihoods\n");
#endif
    
}

void KernelLauncher::IntegrateLikelihoodsSecondDeriv(GPUPtr dResult,
                                          GPUPtr dFirstDerivResult,
                                          GPUPtr dSecondDerivResult,
                                          GPUPtr dRootPartials,
                                          GPUPtr dRootFirstDeriv,
                                          GPUPtr dRootSecondDeriv,
                                          GPUPtr dWeights,
                                          GPUPtr dFrequencies,
                                          unsigned int patternCount,
                                          unsigned int categoryCount) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\tEntering KernelLauncher::IntegrateLikelihoodsSecondDeriv\n");
#endif
    
    int parameterCountV = 8;
    int totalParameterCount = 10;
    gpu->LaunchKernel(fIntegrateLikelihoodsSecondDeriv,
                               bgLikelihoodBlock, bgLikelihoodGrid,
                               parameterCountV, totalParameterCount,
                               dResult, dFirstDerivResult, dSecondDerivResult,
                               dRootPartials, dRootFirstDeriv, dRootSecondDeriv,
                               dWeights, dFrequencies,
                               categoryCount, patternCount);
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::IntegrateLikelihoodsSecondDeriv\n");
#endif
    
}


void KernelLauncher::IntegrateLikelihoodsMulti(GPUPtr dResult,
											   GPUPtr dRootPartials,
											   GPUPtr dWeights,
											   GPUPtr dFrequencies,
											   unsigned int patternCount,
											   unsigned int categoryCount,
											   unsigned int takeLog) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\tEntering KernelLauncher::IntegrateLikelihoodsNoLog\n");
#endif
	
    int parameterCountV = 4;
    int totalParameterCount = 7;
    gpu->LaunchKernel(fIntegrateLikelihoodsMulti,
                               bgLikelihoodBlock, bgLikelihoodGrid,
                               parameterCountV, totalParameterCount,
                               dResult, dRootPartials, dWeights, dFrequencies,
                               categoryCount, patternCount, takeLog);
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::IntegrateLikelihoodsNoLog\n");
#endif
    
}

void KernelLauncher::IntegrateLikelihoodsFixedScaleMulti(GPUPtr dResult,
														 GPUPtr dRootPartials,
														 GPUPtr dWeights,
														 GPUPtr dFrequencies,
                                                         GPUPtr dScalingFactors,
														 GPUPtr dPtrQueue,
														 GPUPtr dMaxScalingFactors,
														 GPUPtr dIndexMaxScalingFactors,
														 unsigned int patternCount,
														 unsigned int categoryCount,
														 unsigned int subsetCount,
														 unsigned int subsetIndex) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::IntegrateLikelihoodsFixedScaleMulti\n");
#endif
    
    gpu->LaunchKernel(fIntegrateLikelihoodsFixedScaleMulti,
                               bgLikelihoodBlock, bgLikelihoodGrid,
                               8, 12,
                               dResult, dRootPartials, dWeights, dFrequencies, dScalingFactors, dPtrQueue,
							   dMaxScalingFactors, dIndexMaxScalingFactors,
                               categoryCount, patternCount, subsetCount, subsetIndex);    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::IntegrateLikelihoodsFixedScaleMulti\n");
#endif
}

void KernelLauncher::SumSites1(GPUPtr dArray1,
                              GPUPtr dSum1,
                              GPUPtr dPatternWeights,
                              unsigned int patternCount) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::SumSites1\n");
#endif
    
    int parameterCountV = 3;
    int totalParameterCount = 4;
    gpu->Synchronize();  
    gpu->LaunchKernel(fSumSites1,
                      bgSumSitesBlock, bgSumSitesGrid,
                      parameterCountV, totalParameterCount,
                      dArray1, dSum1, dPatternWeights,
                      patternCount);
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::SumSites1\n");
#endif
    
}

void KernelLauncher::SumSites2(GPUPtr dArray1,
                              GPUPtr dSum1,
                              GPUPtr dArray2,
                              GPUPtr dSum2,
                              GPUPtr dPatternWeights,
                              unsigned int patternCount) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::SumSites2\n");
#endif
    int parameterCountV = 5;
    int totalParameterCount = 6;
    gpu->Synchronize();
    gpu->LaunchKernel(fSumSites2,
                      bgSumSitesBlock, bgSumSitesGrid,
                      parameterCountV, totalParameterCount,
                      dArray1, dSum1, dArray2, dSum2, dPatternWeights,
                      patternCount);
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::SumSites2\n");
#endif
    
}

void KernelLauncher::SumSites3(GPUPtr dArray1,
                              GPUPtr dSum1,
                              GPUPtr dArray2,
                              GPUPtr dSum2,
                              GPUPtr dArray3,
                              GPUPtr dSum3,
                              GPUPtr dPatternWeights,
                              unsigned int patternCount) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::SumSites3\n");
#endif
    
    int parameterCountV = 7;
    int totalParameterCount = 8;
    gpu->Synchronize();
    gpu->LaunchKernel(fSumSites3,
                      bgSumSitesBlock, bgSumSitesGrid,
                      parameterCountV, totalParameterCount,
                      dArray1, dSum1, dArray2, dSum2, dArray3, dSum3, dPatternWeights,
                      patternCount);        
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::SumSites3\n");
#endif
    
}




