/*
 *
 * Copyright 2009 Phylogenetic Likelihood Working Group
 *
 * This file is part of BEAGLE.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
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
#include <cstring>

#include "libhmsbeagle/beagle.h"
#include "libhmsbeagle/GPU/GPUImplDefs.h"
#include "libhmsbeagle/GPU/KernelLauncher.h"

/**************CODE***********/

#ifdef CUDA
namespace cuda_device {
#else
namespace opencl_device {
#endif

REAL* ones = NULL; // TODO: Memory leak, need to free at some point.

KernelLauncher::KernelLauncher(GPUInterface* inGpu) {
    gpu = inGpu;
    SetupKernelBlocksAndGrids(); // Delegate, so can be overridden as necessary
    LoadKernels(); // Delegate, so can be overridden as necessary
}

KernelLauncher::~KernelLauncher() {
}

void KernelLauncher::SetupKernelBlocksAndGrids() {
    kCPUImplementation = false;
    kAppleCPUImplementation = false;

#ifdef FW_OPENCL
    BeagleDeviceImplementationCodes deviceCode = gpu->GetDeviceImplementationCode(-1);
    if (deviceCode == BEAGLE_OPENCL_DEVICE_INTEL_CPU ||
        deviceCode == BEAGLE_OPENCL_DEVICE_INTEL_MIC ||
        deviceCode == BEAGLE_OPENCL_DEVICE_AMD_CPU) {
        kCPUImplementation = true;
    } else if (deviceCode == BEAGLE_OPENCL_DEVICE_APPLE_CPU) {
        kCPUImplementation = true;
        kAppleCPUImplementation = true;
    }
#endif

    kPaddedStateCount = gpu->kernelResource->paddedStateCount;
    kCategoryCount = gpu->kernelResource->categoryCount;
    kPatternCount = gpu->kernelResource->patternCount;
    kUnpaddedPatternCount = gpu->kernelResource->unpaddedPatternCount;
    kMultiplyBlockSize = gpu->kernelResource->multiplyBlockSize;
    kPatternBlockSize = gpu->kernelResource->patternBlockSize;
    kSlowReweighing = gpu->kernelResource->slowReweighing;
    kMatrixBlockSize = gpu->kernelResource->matrixBlockSize;
    kSumSitesBlockSize = SUM_SITES_BLOCK_SIZE;
    kFlags = gpu->kernelResource->flags;

    // Set up block/grid for transition matrices computation
    bgTransitionProbabilitiesBlock = Dim3Int(kMultiplyBlockSize, kMultiplyBlockSize);
    bgTransitionProbabilitiesGrid = Dim3Int(kPaddedStateCount/kMultiplyBlockSize,
                                            kPaddedStateCount/kMultiplyBlockSize);
    if(kPaddedStateCount % kMultiplyBlockSize != 0) {
        bgTransitionProbabilitiesGrid.x += 1;
        bgTransitionProbabilitiesGrid.y += 1;
    }

    // Set up block/grid for peeling computation
    if (kPaddedStateCount == 4) {
        if (kCPUImplementation) {
            bgPeelingBlock = Dim3Int(kPatternBlockSize, 1);
            bgPeelingGrid  = Dim3Int(kPatternCount / kPatternBlockSize, kCategoryCount);
        } else {
            bgPeelingBlock = Dim3Int(16, kPatternBlockSize);
            bgPeelingGrid  = Dim3Int(kPatternCount / (kPatternBlockSize * 4),
                                     kCategoryCount);
            if (kPatternCount % (kPatternBlockSize * 4) != 0) {
                bgPeelingGrid.x += 1;
            }
        }
    } else {
        if (kAppleCPUImplementation) {
            bgPeelingBlock = Dim3Int(kPaddedStateCount, 1, 1);
            bgPeelingGrid  = Dim3Int(kPatternCount / kPatternBlockSize, kPatternBlockSize, kCategoryCount);
        } else if (kCPUImplementation) {
            bgPeelingBlock = Dim3Int(kPaddedStateCount, kPatternBlockSize, 1);
            bgPeelingGrid  = Dim3Int(kPatternCount / kPatternBlockSize, 1, kCategoryCount);
        } else {
            bgPeelingBlock = Dim3Int(kPaddedStateCount, kPatternBlockSize);
            bgPeelingGrid  = Dim3Int(kPatternCount / kPatternBlockSize, kCategoryCount);
        }
        if (!kCPUImplementation && (kPatternCount % kPatternBlockSize != 0)) {
            bgPeelingGrid.x += 1;
        }
    }

    // Set up block/grid for likelihood computation
    if (kPaddedStateCount == 4) {
        int likePatternBlockSize = kPatternBlockSize;
        if (kCPUImplementation) {
            bgLikelihoodBlock = Dim3Int(likePatternBlockSize,1);
        } else {
            bgLikelihoodBlock = Dim3Int(4,likePatternBlockSize);
        }
        bgLikelihoodGrid = Dim3Int(kPatternCount/likePatternBlockSize);
        if (kPatternCount % likePatternBlockSize != 0)
            bgLikelihoodGrid.x += 1;
    } else {
        if (kCPUImplementation) {
            bgLikelihoodBlock = Dim3Int(1);
        } else {
            bgLikelihoodBlock = Dim3Int(kPaddedStateCount);
        }
        bgLikelihoodGrid  = Dim3Int(kPatternCount);
    }

    // Set up block/grid for cross-product computation
    if (kPaddedStateCount == 4) {
        bgCrossProductBlock = Dim3Int(16,1,1);
        bgCrossProductGrid = Dim3Int(1,1,1);
    } else {
        bgCrossProductBlock = Dim3Int(256, 1, 1);
        const int array = kPaddedStateCount / 16;
        bgCrossProductGrid = Dim3Int(1,1, array * array);
    }

    // Set up block/grid for derivative computation
    if (kPaddedStateCount == 4) {
        if (kCPUImplementation) {
            bgDerivativeBlock = Dim3Int(kPatternBlockSize, 1);
            bgDerivativeGrid  = Dim3Int(kPatternCount / kPatternBlockSize, 1);
        } else {
            bgDerivativeBlock = Dim3Int(16, kPatternBlockSize);
            bgDerivativeGrid  = Dim3Int(kPatternCount / (kPatternBlockSize * 4),
                                        1);
            if (kPatternCount % (kPatternBlockSize * 4) != 0) {
                bgDerivativeGrid.x += 1;
            }
        }
    } else {
        if (kAppleCPUImplementation) {
            bgDerivativeBlock = Dim3Int(kPaddedStateCount, 1, 1);
            bgDerivativeGrid  = Dim3Int(kPatternCount / kPatternBlockSize, kPatternBlockSize, 1);
        } else if (kCPUImplementation) {
            bgDerivativeBlock = Dim3Int(kPaddedStateCount, kPatternBlockSize, 1);
            bgDerivativeGrid  = Dim3Int(kPatternCount / kPatternBlockSize, 1, 1);
        } else {
            bgDerivativeBlock = Dim3Int(kPaddedStateCount, kPatternBlockSize);
            bgDerivativeGrid  = Dim3Int(kPatternCount / kPatternBlockSize, 1);
        }
        if (!kCPUImplementation && (kPatternCount % kPatternBlockSize != 0)) {
            bgDerivativeGrid.x += 1;
        }
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
    if (kCPUImplementation) {
        bgScaleBlock = Dim3Int(kPatternBlockSize);
        bgScaleGrid  = Dim3Int(kPatternCount/kPatternBlockSize);
    } else {
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

    // Set up block for site likelihood accumulation
    if (kCPUImplementation) {
        bgSumSitesBlock = Dim3Int(1);
    } else {
        bgSumSitesBlock = Dim3Int(kSumSitesBlockSize);
    }
    bgSumSitesGrid  = Dim3Int(kUnpaddedPatternCount / kSumSitesBlockSize);
    if (kUnpaddedPatternCount % kSumSitesBlockSize != 0)
        bgSumSitesGrid.x += 1;

    // Set up block for multiple node accumulation
    if (kCPUImplementation) {
        bgMultiNodeSumBlock = Dim3Int(1);
    } else {
        bgMultiNodeSumBlock = Dim3Int(MULTI_NODE_SUM_BLOCK_SIZE);
    }
    bgMultiNodeSumGrid  = Dim3Int(1);

    // Set up block for reordering partials
    if (kAppleCPUImplementation) {
        bgReorderPatternsBlock = Dim3Int(REORDER_BLOCK_SIZE_APPLECPU);
    } else if (kCPUImplementation) {
        bgReorderPatternsBlock = Dim3Int(REORDER_BLOCK_SIZE_CPU);
    } else {
        bgReorderPatternsBlock = Dim3Int(kPaddedStateCount, REORDER_BLOCK_SIZE);
    }
    bgReorderPatternsGrid = Dim3Int((kUnpaddedPatternCount + REORDER_BLOCK_SIZE - 1) / REORDER_BLOCK_SIZE, kCategoryCount);

    //Set up block for basta partials

    bgBastaPeelingBlock = Dim3Int(kPaddedStateCount, 4);
    bgBastaPeelingGrid = Dim3Int(30,1);

    bgBastaReductionBlock = Dim3Int(kPaddedStateCount, 4);
    bgBastaReductionGrid = Dim3Int(2271, 1);

    bgBastaPreBlock = Dim3Int(32);
    bgBastaPreGrid = Dim3Int(1);

    bgBastaSumBlock = Dim3Int(128);
    bgBastaSumGrid = Dim3Int(1200);
}



void KernelLauncher::LoadKernels() {

#ifdef FW_OPENCL_TESTING
	fMatrixMulADB = gpu->GetFunction("kernelMatrixMulADB");
    fPartialsPartialsByPatternBlockCoherent = gpu->GetFunction(
            "kernelPartialsPartialsNoScale");
    fIntegrateLikelihoods = gpu->GetFunction("kernelIntegrateLikelihoods");
    fSumSites1 = gpu->GetFunction("kernelSumSites1");
#else

	fMatrixConvolution = gpu ->GetFunction("kernelMatrixConvolution");

	fMatrixTranspose = gpu->GetFunction("kernelMatrixTranspose");

    if (kFlags & BEAGLE_FLAG_EIGEN_COMPLEX) {
        fMatrixMulADBMulti = gpu->GetFunction("kernelMatrixMulADBComplexMulti");
    } else {
        fMatrixMulADBMulti = gpu->GetFunction("kernelMatrixMulADBMulti");
    }

    fMatrixMulADBFirstDeriv = gpu->GetFunction("kernelMatrixMulADBFirstDeriv");

    fMatrixMulADBSecondDeriv = gpu->GetFunction("kernelMatrixMulADBSecondDeriv");

	if (kFlags & BEAGLE_FLAG_EIGEN_COMPLEX) {
		fMatrixMulADB = gpu->GetFunction("kernelMatrixMulADBComplex");
	} else {
		fMatrixMulADB = gpu->GetFunction("kernelMatrixMulADB");
	}

    fPartialsPartialsByPatternBlockCoherent = gpu->GetFunction(
            "kernelPartialsPartialsNoScale");

    fPartialsPartialsByPatternBlockFixedScaling = gpu->GetFunction(
            "kernelPartialsPartialsFixedScale");

    fPartialsPartialsByPatternBlockAutoScaling = gpu->GetFunction(
                "kernelPartialsPartialsAutoScale");

    fPartialsPartialsGrowing = gpu->GetFunction(
            "kernelPartialsPartialsGrowing");

    fPartialsStatesGrowing = gpu->GetFunction(
            "kernelPartialsStatesGrowing");

    fPartialsPartialsEdgeFirstDerivatives = gpu->GetFunction(
            "kernelPartialsPartialsEdgeFirstDerivatives");

    fPartialsStatesEdgeFirstDerivatives = gpu->GetFunction(
            "kernelPartialsStatesEdgeFirstDerivatives");

    fMultipleNodeSiteReduction = gpu->GetFunction(
            "kernelMultipleNodeSiteReduction");

    fMultipleNodeSiteSquaredReduction = gpu->GetFunction(
            "kernelMultipleNodeSiteSquaredReduction");

    fPartialsPartialsCrossProducts = gpu->GetFunction(
            "kernelPartialsPartialsCrossProducts");

    fPartialsStatesCrossProducts = gpu->GetFunction(
            "kernelPartialsStatesCrossProducts");

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

    if (kFlags & BEAGLE_FLAG_SCALING_AUTO) {
        fIntegrateLikelihoodsDynamicScaling = gpu->GetFunction("kernelIntegrateLikelihoodsAutoScaling");
    } else {
        fIntegrateLikelihoodsDynamicScaling = gpu->GetFunction("kernelIntegrateLikelihoodsFixedScale");
    }

    if (kFlags & BEAGLE_FLAG_SCALERS_LOG) {
        fAccumulateFactorsDynamicScaling = gpu->GetFunction("kernelAccumulateFactorsScalersLog");
        fRemoveFactorsDynamicScaling = gpu->GetFunction("kernelRemoveFactorsScalersLog");
    } else {
        fAccumulateFactorsDynamicScaling = gpu->GetFunction("kernelAccumulateFactors");
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

    fInnerBastaPartialsCoalescent = gpu->GetFunction("kernelInnerBastaPartialsCoalescent");
    fReduceWithinInterval = gpu->GetFunction("kernelBastaReduceWithinInterval");
    fReduceWithinIntervalSerial = gpu->GetFunction("kernelBastaReduceWithinIntervalSerial");
    fReduceWithinIntervalMerged = gpu->GetFunction("kernelBastaReduceWithinIntervalMerged");
    fReduceAcrossInterval = gpu->GetFunction("kernelBastaReduceAcrossInterval");
    fPreProcessBastaFlags = gpu->GetFunction("kernelPreProcessBastaFlags");
    fAccumulateCarryOut = gpu->GetFunction("kernelAccumulateCarryOut");
    fAccumulateCarryOutFinal = gpu->GetFunction("kernelAccumulateCarryOutFinal");


    fReorderPatterns = gpu->GetFunction("kernelReorderPatterns");

    // partitioning and multi-op kernels
    if (kPaddedStateCount == 4) {
        fPartialsPartialsByPatternBlockCoherentMulti = gpu->GetFunction(
                "kernelPartialsPartialsNoScaleMulti");

        fPartialsPartialsByPatternBlockCoherentPartition = gpu->GetFunction(
                "kernelPartialsPartialsNoScalePartition");

        fPartialsPartialsByPatternBlockFixedScalingMulti = gpu->GetFunction(
                "kernelPartialsPartialsFixedScaleMulti");

        fPartialsPartialsByPatternBlockFixedScalingPartition = gpu->GetFunction(
                "kernelPartialsPartialsFixedScalePartition");

        fStatesPartialsByPatternBlockCoherentMulti = gpu->GetFunction(
                "kernelStatesPartialsNoScaleMulti");

        fStatesPartialsByPatternBlockCoherentPartition = gpu->GetFunction(
                "kernelStatesPartialsNoScalePartition");

        fStatesStatesByPatternBlockCoherentMulti = gpu->GetFunction(
                "kernelStatesStatesNoScaleMulti");

        fStatesStatesByPatternBlockCoherentPartition = gpu->GetFunction(
                "kernelStatesStatesNoScalePartition");

        fStatesPartialsByPatternBlockFixedScalingMulti = gpu->GetFunction(
                "kernelStatesPartialsFixedScaleMulti");

        fStatesPartialsByPatternBlockFixedScalingPartition = gpu->GetFunction(
                "kernelStatesPartialsFixedScalePartition");

        fStatesStatesByPatternBlockFixedScalingMulti = gpu->GetFunction(
                "kernelStatesStatesFixedScaleMulti");

        fStatesStatesByPatternBlockFixedScalingPartition = gpu->GetFunction(
                "kernelStatesStatesFixedScalePartition");

        fPartialsPartialsEdgeLikelihoodsByPartition = gpu->GetFunction(
                "kernelPartialsPartialsEdgeLikelihoodsByPartition");

        fStatesPartialsEdgeLikelihoodsByPartition = gpu->GetFunction(
                "kernelStatesPartialsEdgeLikelihoodsByPartition");

        fIntegrateLikelihoodsDynamicScalingPartition = gpu->GetFunction(
                "kernelIntegrateLikelihoodsFixedScalePartition");

        fResetFactorsDynamicScalingByPartition = gpu->GetFunction(
                "kernelResetFactorsByPartition");

        if (kFlags & BEAGLE_FLAG_SCALERS_LOG) {
            fAccumulateFactorsDynamicScalingByPartition = gpu->GetFunction(
                "kernelAccumulateFactorsScalersLogByPartition");
            fRemoveFactorsDynamicScalingByPartition = gpu->GetFunction(
                "kernelRemoveFactorsScalersLogByPartition");
        } else {
            fRemoveFactorsDynamicScalingByPartition = gpu->GetFunction(
                "kernelRemoveFactorsByPartition");
            fAccumulateFactorsDynamicScalingByPartition = gpu->GetFunction(
                "kernelAccumulateFactorsByPartition");
        }

        if (!kSlowReweighing) {
            if (kFlags & BEAGLE_FLAG_SCALERS_LOG) {
                fPartialsDynamicScalingByPartition = gpu->GetFunction(
                       "kernelPartialsDynamicScalingScalersLogByPartition");
                fPartialsDynamicScalingAccumulateByPartition = gpu->GetFunction(
                        "kernelPartialsDynamicScalingAccumulateScalersLogByPartition");
            } else {
                fPartialsDynamicScalingByPartition = gpu->GetFunction(
                        "kernelPartialsDynamicScalingByPartition");
                fPartialsDynamicScalingAccumulateByPartition = gpu->GetFunction(
                        "kernelPartialsDynamicScalingAccumulateByPartition");
            }
        } else {
            if (kFlags & BEAGLE_FLAG_SCALERS_LOG) {
                fPartialsDynamicScalingByPartition = gpu->GetFunction(
                       "kernelPartialsDynamicScalingSlowScalersLogByPartition"); // TODO write kernel
                fPartialsDynamicScalingAccumulateByPartition = gpu->GetFunction(
                        "kernelPartialsDynamicScalingAccumulateScalersLogByPartition");
            } else {
                fPartialsDynamicScalingByPartition = gpu->GetFunction(
                        "kernelPartialsDynamicScalingSlowByPartition"); // TODO write kernel
                fPartialsDynamicScalingAccumulateByPartition = gpu->GetFunction(
                        "kernelPartialsDynamicScalingAccumulateByPartition");
            }
        }

        fIntegrateLikelihoodsPartition = gpu->GetFunction("kernelIntegrateLikelihoodsPartition");

        fSumSites1Partition = gpu->GetFunction("kernelSumSites1Partition");
    }
#endif // !FW_OPENCL_TESTING
}

// void KernelLauncher::SetupPartitioningKernelGrid(unsigned int partitionBlockCount) {
// #ifdef BEAGLE_DEBUG_FLOW
//     fprintf(stderr, "\t \t Entering KernelLauncher::SetupPartitioningKernelBlocksAndGrids \n");
// #endif

//     bgPeelingGrid.x = partitionBlockCount;

// #ifdef BEAGLE_DEBUG_FLOW
//     fprintf(stderr, "\t \t Leaving  KernelLauncher::SetupPartitioningKernelBlocksAndGrids \n");
// #endif
// }

void KernelLauncher::ReorderPatterns(GPUPtr dPartials,
                                     GPUPtr dStates,
                                     GPUPtr dStatesSort,
                                     GPUPtr dTipOffsets,
                                     GPUPtr dTipTypes,
                                     GPUPtr dPatternsNewOrder,
                                     GPUPtr dPatternWeights,
                                     GPUPtr dPatternWeightsSort,
                                     int    patternCount,
                                     int    paddedPatternCount,
                                     int    tipCount) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::ReorderPatterns\n");
#endif

    bgReorderPatternsGrid.z = tipCount;

    int parameterCountV = 8;
    int totalParameterCount = 10;
    gpu->LaunchKernel(fReorderPatterns,
                      bgReorderPatternsBlock, bgReorderPatternsGrid,
                      parameterCountV, totalParameterCount,
                      dPartials, dStates, dStatesSort, dTipOffsets, dTipTypes,
                      dPatternsNewOrder, dPatternWeights, dPatternWeightsSort,
                      patternCount, paddedPatternCount);

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::ReorderPatterns\n");
#endif
}


void KernelLauncher::ConvolveTransitionMatrices(GPUPtr dMatrices,
		                                        GPUPtr dPtrQueue,
		                                        unsigned int totalMatrixCount) {

#ifdef BEAGLE_DEBUG_FLOW
	fprintf(stderr, "\t \t Entering KernelLauncher::ConvolveMatrices \n");
#endif

	bgTransitionProbabilitiesGrid.x *= totalMatrixCount;

	//Dim3Int grid(1);// = totalMatrixCount;
	//Dim3Int block(4,4);// = totalMatrixCount;

	int parameterCountV = 2;
	int totalParameterCount = 3;

	gpu->LaunchKernel(fMatrixConvolution, bgTransitionProbabilitiesBlock,
			bgTransitionProbabilitiesGrid, parameterCountV,
			totalParameterCount, dMatrices, dPtrQueue, totalMatrixCount);

	bgTransitionProbabilitiesGrid.x /= totalMatrixCount; // Reset value

#ifdef BEAGLE_DEBUG_FLOW
	fprintf(stderr, "\t \t Leaving  KernelLauncher::ConvolveMatrices \n");
#endif
}

void KernelLauncher::TransposeTransitionMatrices(GPUPtr dMatrices,
		                                         GPUPtr dPtrQueue,
		                                         unsigned int totalMatrixCount) {

#ifdef BEAGLE_DEBUG_FLOW
	fprintf(stderr, "\t \t Entering KernelLauncher::TransposeMatrices \n");
#endif

	bgTransitionProbabilitiesGrid.x *= totalMatrixCount;

	int parameterCountV = 2;
	int totalParameterCount = 3;

	gpu->LaunchKernel(fMatrixTranspose, bgTransitionProbabilitiesBlock,
			bgTransitionProbabilitiesGrid, parameterCountV,
			totalParameterCount, dMatrices, dPtrQueue, totalMatrixCount);

    gpu->SynchronizeDevice();

	bgTransitionProbabilitiesGrid.x /= totalMatrixCount; // Reset value

#ifdef BEAGLE_DEBUG_FLOW
	fprintf(stderr, "\t \t Leaving  KernelLauncher::TransposeMatrices \n");
#endif
}

void KernelLauncher::GetTransitionProbabilitiesSquareMulti(GPUPtr dMatrices,
                                                           GPUPtr dPtrQueue,
                                                           GPUPtr dEvec,
                                                           GPUPtr dIevc,
                                                           GPUPtr dEigenValues,
                                                           GPUPtr distanceQueue,
                                                           unsigned int totalMatrix) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::GetTransitionProbabilitiesSquareMulti\n");
#endif

    bgTransitionProbabilitiesGrid.x *= totalMatrix;

    // Transposed (interchanged Ievc and Evec)
    int parameterCountV = 6;
    int totalParameterCount = 9;
    gpu->LaunchKernel(fMatrixMulADBMulti,
                      bgTransitionProbabilitiesBlock, bgTransitionProbabilitiesGrid,
                      parameterCountV, totalParameterCount,
                      dMatrices, dPtrQueue, dIevc, dEigenValues, dEvec, distanceQueue,
                      kPaddedStateCount, kPaddedStateCount,
                      totalMatrix);

    bgTransitionProbabilitiesGrid.x /= totalMatrix; // Reset value

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::GetTransitionProbabilitiesSquareMulti\n");
#endif
}

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

void KernelLauncher::PartialsStatesEdgeFirstDerivatives(GPUPtr out,
                                                        GPUPtr states0,
                                                        GPUPtr partials0,
                                                        GPUPtr matrices0,
                                                        GPUPtr instructions,
                                                        GPUPtr weights,
                                                        unsigned int instructionOffset,
                                                        unsigned int nodeCount,
                                                        unsigned int patternCount,
                                                        unsigned int categoryCount,
                                                        bool synchronize) {

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::PartialsStatesEdgeFirstDerivatives\n");
#endif
    unsigned int saved = bgDerivativeGrid.y;
    bgDerivativeGrid.y = nodeCount;

    gpu->LaunchKernel(fPartialsStatesEdgeFirstDerivatives,
                      bgDerivativeBlock, bgDerivativeGrid,
                      6, 9,
                      out, states0, partials0, matrices0, instructions, weights,
                      instructionOffset, patternCount, categoryCount);

    if (synchronize) {
        gpu->SynchronizeDevice();
    }

    bgDerivativeGrid.y = saved;
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving KernelLauncher::PartialsStatesEdgeFirstDerivatives\n");
#endif
}

void KernelLauncher::PartialsPartialsEdgeFirstDerivatives(GPUPtr out,
                                                          GPUPtr partials0,
                                                          GPUPtr matrices0,
                                                          GPUPtr instructions,
                                                          GPUPtr weights,
                                                          unsigned int instructionOffset,
                                                          unsigned int nodeCount,
                                                          unsigned int patternCount,
                                                          unsigned int categoryCount,
                                                          bool synchronize) {

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::PartialsPartialsEdgeFirstDerivatives\n");
#endif

    unsigned int saved = bgDerivativeGrid.y;
    bgDerivativeGrid.y = nodeCount;

//    fprintf(stderr, "Executing for %d nodes\n", nodeCount);
//    fprintf(stderr, "block = %d %d\n", bgDerivativeBlock.x, bgDerivativeBlock.y);
//    fprintf(stderr, "grid  = %d %d\n", bgDerivativeGrid.x, bgDerivativeGrid.y);

    gpu->LaunchKernel(fPartialsPartialsEdgeFirstDerivatives,
                      bgDerivativeBlock, bgDerivativeGrid,
                      5, 8,
                      out, partials0, matrices0, instructions, weights,
                      instructionOffset, patternCount, categoryCount);

    if (synchronize) {
        gpu->SynchronizeDevice();
    }

    bgDerivativeGrid.y = saved;

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving KernelLauncher::PartialsPartialsEdgeFirstDerivatives\n");
#endif
}

void KernelLauncher::PartialsStatesCrossProducts(GPUPtr out,
                                                 GPUPtr states0,
                                                   GPUPtr partials,
                                                   GPUPtr lengths,
                                                   GPUPtr instructions,
                                                   GPUPtr categoryWeights,
                                                   GPUPtr patternWeights,
                                                   unsigned int instructionOffset,
                                                   unsigned int nodeCount,
                                                   unsigned int rateOffset,
                                                   unsigned int patternCount,
                                                   unsigned int categoryCount,
                                                   bool accumulate,
                                                   unsigned int nodeBlocks,
                                                   unsigned int patternBlocks,
                                                   unsigned int missingState) {

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::PartialsStatesCrossProducts\n");
#endif

    Dim3Int grid = bgCrossProductGrid;  // Dim3Int(patternBlocks, nodeBlocks, 1);
    grid.x = patternBlocks;
    grid.y = nodeBlocks;

//    fprintf(stderr, "Executing for %d nodes\n", nodeCount);
//    fprintf(stderr, "block = %d %d\n", bgCrossProductBlock.x, bgCrossProductBlock.y);
//    fprintf(stderr, "grid  = %d %d\n", grid.x, grid.y);

    gpu->LaunchKernel(fPartialsStatesCrossProducts,
                      bgCrossProductBlock, grid,
                      7, 14,
                      out, states0, partials, lengths, instructions, 
                      categoryWeights, patternWeights,
                      instructionOffset, 
                      patternCount, nodeCount, categoryCount, rateOffset, accumulate, missingState);

    gpu->SynchronizeDevice();

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving KernelLauncher::PartialsStatesCrossProducts\n");
#endif
}

void KernelLauncher::PartialsPartialsCrossProducts(GPUPtr out,
                                                   GPUPtr partials,
                                                   GPUPtr lengths,
                                                   GPUPtr instructions,
                                                   GPUPtr categoryWeights,
                                                   GPUPtr patternWeights,
                                                   unsigned int instructionOffset,
                                                   unsigned int nodeCount,
                                                   unsigned int rateOffset,
                                                   unsigned int patternCount,
                                                   unsigned int categoryCount,
                                                   bool accumulate,
                                                   unsigned int nodeBlocks,
                                                   unsigned int patternBlocks) {

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::PartialsPartialsCrossProducts\n");
#endif

    Dim3Int grid = bgCrossProductGrid;  // Dim3Int(patternBlocks, nodeBlocks, 1);
    grid.x = patternBlocks;
    grid.y = nodeBlocks;

//    fprintf(stderr, "Executing for %d nodes\n", nodeCount);
//    fprintf(stderr, "block = %d %d\n", bgCrossProductBlock.x, bgCrossProductBlock.y);
//    fprintf(stderr, "grid  = %d %d %d\n", grid.x, grid.y, grid.z);
//    fprintf(stderr, "accumulate = %d\n", accumulate);

    gpu->LaunchKernel(fPartialsPartialsCrossProducts,
                      bgCrossProductBlock, grid,
                      6, 12,
                      out, partials, lengths, instructions,
                      categoryWeights, patternWeights,
                      instructionOffset,
                      patternCount, nodeCount, categoryCount, rateOffset, accumulate);

    gpu->SynchronizeDevice();

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving KernelLauncher::PartialsPartialsCrossProducts\n");
#endif
}

void KernelLauncher::MultipleNodeSiteReduction(GPUPtr outSiteValues,
                                               GPUPtr inSiteValues,
                                               GPUPtr weights,
                                               unsigned int outputOffset,
                                               unsigned int stride,
                                               unsigned int count) {

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::MultipleNodeSiteReduction\n");
#endif

    unsigned int saved = bgMultiNodeSumGrid.y;
    bgMultiNodeSumGrid.x = count;

//    fprintf(stderr, "Executing for %d nodes\n", count);
//    fprintf(stderr, "block = %d %d\n", bgMultiNodeSumBlock.x, bgMultiNodeSumBlock.y);
//    fprintf(stderr, "grid  = %d %d\n", bgMultiNodeSumGrid.x, bgMultiNodeSumGrid.y);

    gpu->LaunchKernel(fMultipleNodeSiteReduction,
                      bgMultiNodeSumBlock, bgMultiNodeSumGrid,
                      3, 5,
                      outSiteValues, inSiteValues, weights, outputOffset, stride);
    gpu->SynchronizeDevice();

    bgMultiNodeSumGrid.x = saved;

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving KernelLauncher::MultipleNodeSiteReduction\n");
#endif
}

void KernelLauncher::MultipleNodeSiteSquaredReduction(GPUPtr outSiteValues,
                                                      GPUPtr inSiteValues,
                                                      GPUPtr weights,
                                                      unsigned int outputOffset,
                                                      unsigned int stride,
                                                      unsigned int count) {

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::MultipleNodeSiteSquaredReduction\n");
#endif

    unsigned int saved = bgMultiNodeSumGrid.y;
    bgMultiNodeSumGrid.x = count;

//    fprintf(stderr, "Executing for %d nodes\n", count);
//    fprintf(stderr, "block = %d %d\n", bgMultiNodeSumBlock.x, bgMultiNodeSumBlock.y);
//    fprintf(stderr, "grid  = %d %d\n", bgMultiNodeSumGrid.x, bgMultiNodeSumGrid.y);

    gpu->LaunchKernel(fMultipleNodeSiteSquaredReduction,
                      bgMultiNodeSumBlock, bgMultiNodeSumGrid,
                      3, 5,
                      outSiteValues, inSiteValues, weights, outputOffset, stride);
    gpu->SynchronizeDevice();

    bgMultiNodeSumGrid.x = saved;

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving KernelLauncher::MultipleNodeSiteSquaredReduction\n");
#endif
}

void KernelLauncher::PartialsStatesGrowing(GPUPtr partials1,
                                           GPUPtr states2,
                                           GPUPtr partials3,
                                           GPUPtr matrices1,
                                           GPUPtr matrices2,
                                           unsigned int patternCount,
                                           unsigned int categoryCount,
                                           int sizeReal) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::PartialsStatesGrowing\n");
#endif

    gpu->LaunchKernel(fPartialsStatesGrowing,
                      bgPeelingBlock, bgPeelingGrid,
                      5, 6,
                      partials1, states2, partials3, matrices1, matrices2,
                      patternCount);
    gpu->SynchronizeDevice();

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving KernelLauncher::PartialsStatesGrowing\n");
#endif
}

void KernelLauncher::PartialsPartialsGrowing(GPUPtr partials1,
                                             GPUPtr partials2,
                                             GPUPtr partials3,
                                             GPUPtr matrices1,
                                             GPUPtr matrices2,
                                             unsigned int patternCount,
                                             unsigned int categoryCount,
                                             int sizeReal) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::PartialsPartialsGrowing\n");
#endif

    gpu->LaunchKernel(fPartialsPartialsGrowing,
                      bgPeelingBlock, bgPeelingGrid,
                      5, 6,
                      partials1, partials2, partials3, matrices1, matrices2,
                      patternCount);
    gpu->SynchronizeDevice();

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving KernelLauncher::PartialsPartialsGrowing\n");
#endif
}


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

        gpu->SynchronizeDevice();
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
                gpu->SynchronizeDevice();
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

        gpu->SynchronizeDevice();
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
                gpu->SynchronizeDevice();
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

void KernelLauncher::PartialsPartialsPruningMulti(GPUPtr partials,
                                                  GPUPtr matrices,
                                                  GPUPtr scalingFactors,
                                                  GPUPtr ptrOffsets,
                                                  unsigned int patternCount,
                                                  int gridStartOp,
                                                  int gridSize,
                                                  int doRescaling) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::PartialsPartialsPruningMulti\n");
#endif

    int tmpGridx = bgPeelingGrid.x;
    bgPeelingGrid.x = gridSize;

    if (doRescaling != 0) {
        gpu->LaunchKernel(fPartialsPartialsByPatternBlockCoherentMulti,
                          bgPeelingBlock, bgPeelingGrid,
                          3, 5,
                          partials, matrices, ptrOffsets,
                          gridStartOp, patternCount);
    } else {
        gpu->LaunchKernel(fPartialsPartialsByPatternBlockFixedScalingMulti,
                          bgPeelingBlock, bgPeelingGrid,
                          4, 6,
                          partials, matrices, scalingFactors, ptrOffsets,
                          gridStartOp, patternCount);
    }

    bgPeelingGrid.x = tmpGridx;

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::PartialsPartialsPruningMulti\n");
#endif
}

void KernelLauncher::PartialsPartialsPruningDynamicScaling(GPUPtr partials1,
                                                           GPUPtr partials2,
                                                           GPUPtr partials3,
                                                           GPUPtr matrices1,
                                                           GPUPtr matrices2,
                                                           GPUPtr scalingFactors,
                                                           GPUPtr cumulativeScaling,
                                                           unsigned int startPattern,
                                                           unsigned int endPattern,
                                                           unsigned int patternCount,
                                                           unsigned int categoryCount,
                                                           int doRescaling,
                                                           int streamIndex,
                                                           int waitIndex) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::PartialsPartialsPruningDynamicScaling\n");
#endif

    int tmpGridx = bgPeelingGrid.x;
    if (endPattern != 0) {
        int launchPatternCount = endPattern - startPattern;
        int blockPatternCount = kPatternBlockSize;
        if (kPaddedStateCount == 4 && !kCPUImplementation) {
            blockPatternCount *= 4;
        }
        bgPeelingGrid.x = (launchPatternCount + blockPatternCount - 1) / blockPatternCount;
    }

    if (doRescaling == 2) { // auto-rescaling
        bgPeelingGrid.x = tmpGridx;
        gpu->LaunchKernel(fPartialsPartialsByPatternBlockAutoScaling,
                          bgPeelingBlock, bgPeelingGrid,
                          6, 7,
                          partials1, partials2, partials3, matrices1, matrices2, scalingFactors,
                          patternCount);
    } else if (doRescaling != 0) {
        // Compute partials without any rescaling

        if (endPattern != 0) {
            gpu->LaunchKernelConcurrent(fPartialsPartialsByPatternBlockCoherentPartition,
                                        bgPeelingBlock, bgPeelingGrid,
                                        streamIndex, waitIndex,
                                        5, 8,
                                        partials1, partials2, partials3, matrices1, matrices2,
                                        startPattern, endPattern, patternCount);

        } else {
            gpu->LaunchKernelConcurrent(fPartialsPartialsByPatternBlockCoherent,
                                        bgPeelingBlock, bgPeelingGrid,
                                        streamIndex, waitIndex,
                                        5, 6,
                                        partials1, partials2, partials3, matrices1, matrices2,
                                        patternCount);
        }

        // Rescale partials and save scaling factors
        if (doRescaling > 0) {
            if (endPattern == 0 ) {
                KernelLauncher::RescalePartials(partials3, scalingFactors, cumulativeScaling,
                                                patternCount, categoryCount, 0, streamIndex, -1);
            } else {
                KernelLauncher::RescalePartialsByPartition(partials3, scalingFactors, cumulativeScaling,
                                                patternCount, categoryCount, 0, streamIndex, -1,
                                                startPattern, endPattern);
            }
        }

    } else {
        // Compute partials with known rescalings
        if (endPattern != 0) {
            gpu->LaunchKernelConcurrent(fPartialsPartialsByPatternBlockFixedScalingPartition,
                                        bgPeelingBlock, bgPeelingGrid,
                                        streamIndex, waitIndex,
                                        6, 9,
                                        partials1, partials2, partials3, matrices1, matrices2,
                                        scalingFactors,
                                        startPattern, endPattern, patternCount);
        } else {
            gpu->LaunchKernelConcurrent(fPartialsPartialsByPatternBlockFixedScaling,
                              bgPeelingBlock, bgPeelingGrid,
                              streamIndex, waitIndex,
                              6, 7,
                              partials1, partials2, partials3, matrices1, matrices2,
                              scalingFactors,
                              patternCount);
        }
    }

    bgPeelingGrid.x = tmpGridx;

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::PartialsPartialsPruningDynamicScaling\n");
#endif

}


void KernelLauncher::StatesPartialsPruningMulti(GPUPtr states,
                                                GPUPtr partials,
                                                GPUPtr matrices,
                                                GPUPtr scalingFactors,
                                                GPUPtr ptrOffsets,
                                                unsigned int patternCount,
                                                int gridStartOp,
                                                int gridSize,
                                                int doRescaling) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::StatesPartialsPruningMulti\n");
#endif

    int tmpGridx = bgPeelingGrid.x;
    bgPeelingGrid.x = gridSize;

    if (doRescaling != 0) {
        gpu->LaunchKernel(fStatesPartialsByPatternBlockCoherentMulti,
                          bgPeelingBlock, bgPeelingGrid,
                          4, 6,
                          states, partials, matrices, ptrOffsets,
                          gridStartOp, patternCount);
    } else {
        gpu->LaunchKernel(fStatesPartialsByPatternBlockFixedScalingMulti,
                          bgPeelingBlock, bgPeelingGrid,
                          5, 7,
                          states, partials, matrices, scalingFactors, ptrOffsets,
                          gridStartOp, patternCount);
    }

    bgPeelingGrid.x = tmpGridx;

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::StatesPartialsPruningMulti\n");
#endif
}


void KernelLauncher::StatesPartialsPruningDynamicScaling(GPUPtr states1,
                                                         GPUPtr partials2,
                                                         GPUPtr partials3,
                                                         GPUPtr matrices1,
                                                         GPUPtr matrices2,
                                                         GPUPtr scalingFactors,
                                                         GPUPtr cumulativeScaling,
                                                         unsigned int startPattern,
                                                         unsigned int endPattern,
                                                         unsigned int patternCount,
                                                         unsigned int categoryCount,
                                                         int doRescaling,
                                                         int streamIndex,
                                                         int waitIndex) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::StatesPartialsPruningDynamicScaling\n");
#endif

#ifdef FW_OPENCL
    // fix for Apple CPU OpenCL limitations
    size_t blockX = bgPeelingBlock.x;
    size_t gridX  = bgPeelingGrid.x;
    bool AppleCPUImplementation = false;
    if (kAppleCPUImplementation &&
        kPaddedStateCount == 4) {
        bgPeelingBlock.x = 1;
        bgPeelingGrid.x  = gridX * blockX;
        AppleCPUImplementation = true;
    }
#endif

    int tmpGridx = bgPeelingGrid.x;
    if (endPattern != 0) {
        int launchPatternCount = endPattern - startPattern;
        int blockPatternCount = kPatternBlockSize;
        if (kAppleCPUImplementation)
            blockPatternCount = 1;
        if (kPaddedStateCount == 4 && !kCPUImplementation) {
            blockPatternCount *= 4;
        }
        int tmpGridx = bgPeelingGrid.x;
        bgPeelingGrid.x = (launchPatternCount + blockPatternCount - 1) / blockPatternCount;
    }

    if (doRescaling != 0)    {
        // Compute partials without any rescaling
        if (endPattern != 0) {
            gpu->LaunchKernelConcurrent(fStatesPartialsByPatternBlockCoherentPartition,
                                        bgPeelingBlock, bgPeelingGrid,
                                        streamIndex, waitIndex,
                                        5, 8,
                                        states1, partials2, partials3, matrices1, matrices2,
                                        startPattern, endPattern, patternCount);

        } else {
            gpu->LaunchKernelConcurrent(fStatesPartialsByPatternBlockCoherent,
                                        bgPeelingBlock, bgPeelingGrid,
                                        streamIndex, waitIndex,
                                        5, 6,
                                        states1, partials2, partials3, matrices1, matrices2,
                                        patternCount);
        }

        // Rescale partials and save scaling factors
        if (doRescaling > 0) {
            if (endPattern == 0 ) {
                KernelLauncher::RescalePartials(partials3, scalingFactors, cumulativeScaling,
                                                patternCount, categoryCount,
#ifdef BEAGLE_FILL_4_STATE_SCALAR_SP
                                                1
#else
                                                0
#endif
                                                , streamIndex, -1);
            } else {
                KernelLauncher::RescalePartialsByPartition(partials3, scalingFactors, cumulativeScaling,
                                                patternCount, categoryCount,
#ifdef BEAGLE_FILL_4_STATE_SCALAR_SP
                                                1
#else
                                                0
#endif
                                                , streamIndex, -1, startPattern, endPattern);
            }
        }

    } else {

        // Compute partials with known rescalings
#ifdef BEAGLE_FILL_4_STATE_SCALAR_SP
        if (kPaddedStateCount == 4) { // Ignore rescaling
            if (endPattern != 0) {
                gpu->LaunchKernelConcurrent(fStatesPartialsByPatternBlockCoherentPartition,
                                            bgPeelingBlock, bgPeelingGrid,
                                            streamIndex, waitIndex,
                                            5, 8,
                                            states1, partials2, partials3, matrices1, matrices2,
                                            startPattern, endPattern, patternCount);

            } else {
                gpu->LaunchKernelConcurrent(fStatesPartialsByPatternBlockCoherent,
                                            bgPeelingBlock, bgPeelingGrid,
                                            streamIndex, waitIndex,
                                            5, 6,
                                            states1, partials2, partials3, matrices1, matrices2,
                                            patternCount);
            }
        } else {
#endif
        if (endPattern != 0) {
            gpu->LaunchKernelConcurrent(fStatesPartialsByPatternBlockFixedScalingPartition,
                                        bgPeelingBlock, bgPeelingGrid,
                                        streamIndex, waitIndex,
                                        6, 9,
                                        states1, partials2, partials3, matrices1, matrices2,
                                        scalingFactors,
                                        startPattern, endPattern, patternCount);
        } else {
            gpu->LaunchKernelConcurrent(fStatesPartialsByPatternBlockFixedScaling,
                                   bgPeelingBlock, bgPeelingGrid,
                                   streamIndex, waitIndex,
                                   6, 7,
                                   states1, partials2, partials3, matrices1, matrices2,
                                   scalingFactors,
                                   patternCount);
        }
#ifdef BEAGLE_FILL_4_STATE_SCALAR_SP
        }
#endif
    }

    bgPeelingGrid.x = tmpGridx;

#ifdef FW_OPENCL
    // restore values if used fix for Apple CPU OpenCL limitations
    if (AppleCPUImplementation) {
        bgPeelingBlock.x = blockX;
        bgPeelingGrid.x  = gridX;
    }
#endif

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\tLeaving  KernelLauncher::StatesPartialsPruningDynamicScaling\n");
#endif

}

void KernelLauncher::StatesStatesPruningMulti(GPUPtr states,
                                              GPUPtr partials,
                                              GPUPtr matrices,
                                              GPUPtr scalingFactors,
                                              GPUPtr ptrOffsets,
                                              unsigned int patternCount,
                                              int gridStartOp,
                                              int gridSize,
                                              int doRescaling) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::StatesStatesPruningMulti\n");
#endif

    int tmpGridx = bgPeelingGrid.x;
    bgPeelingGrid.x = gridSize;

    if (doRescaling != 0) {
        gpu->LaunchKernel(fStatesStatesByPatternBlockCoherentMulti,
                          bgPeelingBlock, bgPeelingGrid,
                          4, 6,
                          states, partials, matrices, ptrOffsets,
                          gridStartOp, patternCount);
    } else {
        gpu->LaunchKernel(fStatesStatesByPatternBlockFixedScalingMulti,
                          bgPeelingBlock, bgPeelingGrid,
                          5, 7,
                          states, partials, matrices, scalingFactors, ptrOffsets,
                          gridStartOp, patternCount);
    }

    bgPeelingGrid.x = tmpGridx;

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::StatesStatesPruningMulti\n");
#endif
}


void KernelLauncher::StatesStatesPruningDynamicScaling(GPUPtr states1,
                                                       GPUPtr states2,
                                                       GPUPtr partials3,
                                                       GPUPtr matrices1,
                                                       GPUPtr matrices2,
                                                       GPUPtr scalingFactors,
                                                       GPUPtr cumulativeScaling,
                                                       unsigned int startPattern,
                                                       unsigned int endPattern,
                                                       unsigned int patternCount,
                                                       unsigned int categoryCount,
                                                       int doRescaling,
                                                       int streamIndex,
                                                       int waitIndex) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::StatesStatesPruningDynamicScaling\n");
#endif

#ifdef FW_OPENCL
    // fix for Apple CPU OpenCL limitations
    size_t blockX = bgPeelingBlock.x;
    size_t gridX  = bgPeelingGrid.x;
    bool AppleCPUImplementation = false;
    if (kAppleCPUImplementation &&
        kPaddedStateCount == 4) {
        bgPeelingBlock.x = 1;
        bgPeelingGrid.x  = gridX * blockX;
        AppleCPUImplementation = true;
    }
#endif

    int tmpGridx = bgPeelingGrid.x;
    if (endPattern != 0) {
        int launchPatternCount = endPattern - startPattern;
        int blockPatternCount = kPatternBlockSize;
        if (kAppleCPUImplementation)
            blockPatternCount = 1;
        if (kPaddedStateCount == 4 && !kCPUImplementation) {
            blockPatternCount *= 4;
        }
        int tmpGridx = bgPeelingGrid.x;
        bgPeelingGrid.x = (launchPatternCount + blockPatternCount - 1) / blockPatternCount;
    }

    if (doRescaling != 0)    {

        // Compute partials without any rescaling
        if (endPattern != 0) {
            gpu->LaunchKernelConcurrent(fStatesStatesByPatternBlockCoherentPartition,
                                        bgPeelingBlock, bgPeelingGrid,
                                        streamIndex, waitIndex,
                                        5, 8,
                                        states1, states2, partials3, matrices1, matrices2,
                                        startPattern, endPattern, patternCount);
        } else {
            gpu->LaunchKernelConcurrent(fStatesStatesByPatternBlockCoherent,
                                        bgPeelingBlock, bgPeelingGrid,
                                        streamIndex, waitIndex,
                                        5, 6,
                                        states1, states2, partials3, matrices1, matrices2,
                                        patternCount);
        }

        // Rescale partials and save scaling factors
        if (doRescaling > 0) {
            if (endPattern == 0 ) {
                KernelLauncher::RescalePartials(partials3, scalingFactors, cumulativeScaling,
                                                patternCount, categoryCount,
#ifdef BEAGLE_FILL_4_STATE_SCALAR_SS
                                                1
#else
                                                0
#endif
                                                , streamIndex, -1);
            } else {
                KernelLauncher::RescalePartialsByPartition(partials3, scalingFactors, cumulativeScaling,
                                                patternCount, categoryCount,
#ifdef BEAGLE_FILL_4_STATE_SCALAR_SS
                                                1
#else
                                                0
#endif
                                                , streamIndex, -1, startPattern, endPattern);
            }
        }

    } else {

        // Compute partials with known rescalings
#ifdef BEAGLE_FILL_4_STATE_SCALAR_SS
        if (kPaddedStateCount == 4) {
            if (endPattern != 0) {
                gpu->LaunchKernelConcurrent(fStatesStatesByPatternBlockCoherentPartition,
                                            bgPeelingBlock, bgPeelingGrid,
                                            streamIndex, waitIndex,
                                            5, 8,
                                            states1, states2, partials3, matrices1, matrices2,
                                            startPattern, endPattern, patternCount);
            } else {
                gpu->LaunchKernelConcurrent(fStatesStatesByPatternBlockCoherent,
                                            bgPeelingBlock, bgPeelingGrid,
                                            streamIndex, waitIndex,
                                            5, 6,
                                            states1, states2, partials3, matrices1, matrices2,
                                            patternCount);
            }
        } else {
#endif
            if (endPattern != 0) {
                gpu->LaunchKernelConcurrent(fStatesStatesByPatternBlockFixedScalingPartition,
                                            bgPeelingBlock, bgPeelingGrid,
                                            streamIndex, waitIndex,
                                            6, 9,
                                            states1, states2, partials3, matrices1, matrices2,
                                            scalingFactors,
                                            startPattern, endPattern, patternCount);
            } else {
                gpu->LaunchKernelConcurrent(fStatesStatesByPatternBlockFixedScaling,
                                       bgPeelingBlock, bgPeelingGrid,
                                       streamIndex, waitIndex,
                                       6, 7,
                                       states1, states2, partials3, matrices1, matrices2,
                                       scalingFactors,
                                       patternCount);
            }
#ifdef BEAGLE_FILL_4_STATE_SCALAR_SS
        }
#endif
    }

    bgPeelingGrid.x = tmpGridx;

#ifdef FW_OPENCL
    // restore values if used fix for Apple CPU OpenCL limitations
    if (AppleCPUImplementation) {
        bgPeelingBlock.x = blockX;
        bgPeelingGrid.x  = gridX;
    }
#endif

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

void KernelLauncher::IntegrateLikelihoodsDynamicScalingPartition(GPUPtr dResult,
                                                                 GPUPtr dRootPartials,
                                                                 GPUPtr dWeights,
                                                                 GPUPtr dFrequencies,
                                                                 GPUPtr dRootScalingFactors,
                                                                 GPUPtr dPtrOffsets,
                                                                 unsigned int patternCount,
                                                                 unsigned int categoryCount,
                                                                 int gridSize) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::IntegrateLikelihoodsDynamicScalingPartition\n");
#endif

    Dim3Int bgLikelihoodPartitionGrid = Dim3Int(gridSize);

    gpu->LaunchKernel(fIntegrateLikelihoodsDynamicScalingPartition,
                      bgLikelihoodBlock, bgLikelihoodPartitionGrid,
                      6, 8,
                      dResult, dRootPartials, dWeights, dFrequencies, dRootScalingFactors, dPtrOffsets,
                      categoryCount, patternCount);

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::IntegrateLikelihoodsDynamicScalingPartition\n");
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

void KernelLauncher::PartialsPartialsEdgeLikelihoodsByPartition(GPUPtr dPartialsTmp,
                                                                GPUPtr dPartialsOrigin,
                                                                GPUPtr dMatricesOrigin,
                                                                GPUPtr dPtrOffsets,
                                                                unsigned int patternCount,
                                                                int gridSize) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\tEntering KernelLauncher::PartialsPartialsEdgeLikelihoodsByPartition\n");
#endif

    int tmpGridx = bgPeelingGrid.x;
    bgPeelingGrid.x = gridSize;

    gpu->LaunchKernel(fPartialsPartialsEdgeLikelihoodsByPartition,
                      bgPeelingBlock, bgPeelingGrid,
                      4, 5,
                      dPartialsTmp, dPartialsOrigin, dMatricesOrigin, dPtrOffsets,
                      patternCount);

    bgPeelingGrid.x = tmpGridx;

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::PartialsPartialsEdgeLikelihoodsByPartition\n");
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

#ifdef FW_OPENCL
    // fix for Apple CPU OpenCL limitations
    size_t blockX = bgPeelingBlock.x;
    size_t gridX  = bgPeelingGrid.x;
    bool AppleCPUImplementation = false;
    if (kAppleCPUImplementation &&
        kPaddedStateCount == 4) {
        bgPeelingBlock.x = 1;
        bgPeelingGrid.x  = gridX * blockX;
        AppleCPUImplementation = true;
    }
#endif

    gpu->LaunchKernel(fStatesPartialsEdgeLikelihoods,
                               bgPeelingBlock, bgPeelingGrid,
                               4, 5,
                               dPartialsTmp, dParentPartials, dChildStates, dTransMatrix,
                               patternCount);

#ifdef FW_OPENCL
    // restore values if used fix for Apple CPU OpenCL limitations
    if (AppleCPUImplementation) {
        bgPeelingBlock.x = blockX;
        bgPeelingGrid.x  = gridX;
    }
#endif

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::StatesPartialsEdgeLikelihoods\n");
#endif

}

void KernelLauncher::StatesPartialsEdgeLikelihoodsByPartition(GPUPtr dPartialsTmp,
                                                              GPUPtr dPartialsOrigin,
                                                              GPUPtr dStatesOrigin,
                                                              GPUPtr dMatricesOrigin,
                                                              GPUPtr dPtrOffsets,
                                                              unsigned int patternCount,
                                                              int gridSize) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\tEntering KernelLauncher::StatesPartialsEdgeLikelihoodsByPartition\n");
#endif

    int tmpGridx = bgPeelingGrid.x;
    bgPeelingGrid.x = gridSize;

    gpu->LaunchKernel(fStatesPartialsEdgeLikelihoodsByPartition,
                      bgPeelingBlock, bgPeelingGrid,
                      5, 6,
                      dPartialsTmp, dPartialsOrigin, dStatesOrigin, dMatricesOrigin, dPtrOffsets,
                      patternCount);

    bgPeelingGrid.x = tmpGridx;

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::StatesPartialsEdgeLikelihoodsByPartition\n");
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

#ifdef FW_OPENCL
    // fix for Apple CPU OpenCL limitations
    size_t blockX = bgPeelingBlock.x;
    size_t gridX  = bgPeelingGrid.x;
    bool AppleCPUImplementation = false;
    if (kAppleCPUImplementation &&
        kPaddedStateCount == 4) {
        bgPeelingBlock.x = 1;
        bgPeelingGrid.x  = gridX * blockX;
        AppleCPUImplementation = true;
    }
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

#ifdef FW_OPENCL
    // restore values if used fix for Apple CPU OpenCL limitations
    if (AppleCPUImplementation) {
        bgPeelingBlock.x = blockX;
        bgPeelingGrid.x  = gridX;
    }
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


void KernelLauncher::AccumulateFactorsDynamicScalingByPartition(GPUPtr dScalingFactors,
                                                                GPUPtr dNodePtrQueue,
                                                                GPUPtr dRootScalingFactors,
                                                                unsigned int nodeCount,
                                                                int startPattern,
                                                                int endPattern) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::AccumulateFactorsDynamicScalingByPartition\n");
#endif

    int partitionPatternCount = endPattern - startPattern;
    Dim3Int bgAccumulatePartitionGrid  = Dim3Int(partitionPatternCount / kPatternBlockSize);

    if (partitionPatternCount % kPatternBlockSize != 0)
        bgAccumulatePartitionGrid.x += 1;

    int parameterCountV = 3;
    int totalParameterCount = 6;
    gpu->LaunchKernel(fAccumulateFactorsDynamicScalingByPartition,
                      bgAccumulateBlock, bgAccumulatePartitionGrid,
                      parameterCountV, totalParameterCount,
                      dScalingFactors, dNodePtrQueue, dRootScalingFactors,
                      nodeCount, startPattern, endPattern);

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::AccumulateFactorsDynamicScalingByPartition\n");
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

void KernelLauncher::RemoveFactorsDynamicScalingByPartition(GPUPtr dScalingFactors,
                                                            GPUPtr dNodePtrQueue,
                                                            GPUPtr dRootScalingFactors,
                                                            unsigned int nodeCount,
                                                            int startPattern,
                                                            int endPattern) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::RemoveFactorsDynamicScalingByPartition\n");
#endif

    int partitionPatternCount = endPattern - startPattern;
    Dim3Int bgAccumulatePartitionGrid  = Dim3Int(partitionPatternCount / kPatternBlockSize);

    if (partitionPatternCount % kPatternBlockSize != 0)
        bgAccumulatePartitionGrid.x += 1;

    int parameterCountV = 3;
    int totalParameterCount = 6;
    gpu->LaunchKernel(fRemoveFactorsDynamicScalingByPartition,
                      bgAccumulateBlock, bgAccumulatePartitionGrid,
                      parameterCountV, totalParameterCount,
                      dScalingFactors, dNodePtrQueue, dRootScalingFactors,
                      nodeCount, startPattern, endPattern);

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::RemoveFactorsDynamicScalingByPartition\n");
#endif

}

void KernelLauncher::ResetFactorsDynamicScalingByPartition(GPUPtr dScalingFactors,
                                                           int startPattern,
                                                           int endPattern) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::ResetFactorsDynamicScalingByPartition\n");
#endif

    int partitionPatternCount = endPattern - startPattern;
    Dim3Int bgAccumulatePartitionGrid  = Dim3Int(partitionPatternCount / kPatternBlockSize);

    if (partitionPatternCount % kPatternBlockSize != 0)
        bgAccumulatePartitionGrid.x += 1;

    int parameterCountV = 1;
    int totalParameterCount = 3;
    gpu->LaunchKernel(fResetFactorsDynamicScalingByPartition,
                      bgAccumulateBlock, bgAccumulatePartitionGrid,
                      parameterCountV, totalParameterCount,
                      dScalingFactors,
                      startPattern, endPattern);

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::ResetFactorsDynamicScalingByPartition\n");
#endif

}

void KernelLauncher::RescalePartials(GPUPtr partials3,
                                     GPUPtr scalingFactors,
                                     GPUPtr cumulativeScaling,
                                     unsigned int patternCount,
                                     unsigned int categoryCount,
                                     unsigned int fillWithOnes,
                                     int streamIndex,
                                     int waitIndex) {

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::RescalePartials\n");
#endif

    // TODO: remove fillWithOnes and leave it up to client?
// printf("RESCALE ON STREAM %d\n",streamIndex );
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
        gpu->LaunchKernelConcurrent(fPartialsDynamicScalingAccumulate,
                                   bgScaleBlock, bgScaleGrid,
                                   streamIndex, waitIndex,
                                   parameterCountV, totalParameterCount,
                                   partials3, scalingFactors, cumulativeScaling,
                                   categoryCount);
    } else {
        int parameterCountV = 2;
        int totalParameterCount = 3;
        gpu->LaunchKernelConcurrent(fPartialsDynamicScaling,
                                   bgScaleBlock, bgScaleGrid,
                                   streamIndex, waitIndex,
                                   parameterCountV, totalParameterCount,
                                   partials3, scalingFactors,
                                   categoryCount);
    }

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::RescalePartials\n");
#endif
}

void KernelLauncher::RescalePartialsByPartition(GPUPtr partials3,
                                                GPUPtr scalingFactors,
                                                GPUPtr cumulativeScaling,
                                                unsigned int patternCount,
                                                unsigned int categoryCount,
                                                unsigned int fillWithOnes,
                                                int streamIndex,
                                                int waitIndex,
                                                int startPattern,
                                                int endPattern) {

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::RescalePartialsByPartition\n");
#endif

    int partitionPatternCount = endPattern - startPattern;

    Dim3Int bgScalePartitionGrid = bgScaleGrid;

    if (kCPUImplementation) {
        bgScalePartitionGrid.x  = partitionPatternCount/kPatternBlockSize;
        if (partitionPatternCount % kPatternBlockSize != 0)
            bgScalePartitionGrid.x += 1;
    } else {
        if (kSlowReweighing) {
            fprintf(stderr,"Slow reweighing and partitioning not yet implemented\n");
            exit(-1);
        } else {
            if (kPaddedStateCount == 4) {
                bgScalePartitionGrid.x  = partitionPatternCount / 4;
                if (partitionPatternCount % 4 != 0) {
                    bgScalePartitionGrid.x += 1; //
                    // fprintf(stderr,"PATTERNS SHOULD BE PADDED! Inform Marc, please.\n");
                    // exit(-1);
                }
            } else {
                fprintf(stderr,"Partitioning and state count != 4 not implemented\n");
                exit(-1);
            }
        }
    }

// printf("p %d, s %2d, e %2d, bgScaleGrid.x %2d, bgScalePartitionGrid.x %2d\n", partials3, startPattern, endPattern, bgScaleGrid.x, bgScalePartitionGrid.x );

    if (kPaddedStateCount == 4) {
        if (fillWithOnes != 0) {
            fprintf(stderr,"Old legacy code; should not get here!\n");
            exit(0);
        }
    }

    // TODO: Totally incoherent for kPaddedStateCount == 4
    if (cumulativeScaling != 0) {
        int parameterCountV = 3;
        int totalParameterCount = 7;
        gpu->LaunchKernelConcurrent(fPartialsDynamicScalingAccumulateByPartition,
                                    bgScaleBlock, bgScalePartitionGrid,
                                    streamIndex, waitIndex,
                                    parameterCountV, totalParameterCount,
                                    partials3, scalingFactors, cumulativeScaling,
                                    categoryCount, startPattern, endPattern, patternCount);
    } else {
        int parameterCountV = 2;
        int totalParameterCount = 6;
        gpu->LaunchKernelConcurrent(fPartialsDynamicScalingByPartition,
                                    bgScaleBlock, bgScalePartitionGrid,
                                    streamIndex, waitIndex,
                                    parameterCountV, totalParameterCount,
                                    partials3, scalingFactors,
                                    categoryCount, startPattern, endPattern, patternCount);
    }

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::RescalePartialsByPartition\n");
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

void KernelLauncher::IntegrateLikelihoodsPartition(GPUPtr dResult,
                                                   GPUPtr dRootPartials,
                                                   GPUPtr dWeights,
                                                   GPUPtr dFrequencies,
                                                   GPUPtr dPtrOffsets,
                                                   unsigned int patternCount,
                                                   unsigned int categoryCount,
                                                   int gridSize) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\tEntering KernelLauncher::IntegrateLikelihoodsPartition\n");
#endif

    Dim3Int bgLikelihoodPartitionGrid = Dim3Int(gridSize);

    int parameterCountV = 5;
    int totalParameterCount = 7;
    gpu->LaunchKernel(fIntegrateLikelihoodsPartition,
                      bgLikelihoodBlock, bgLikelihoodPartitionGrid,
                      parameterCountV, totalParameterCount,
                      dResult, dRootPartials, dWeights, dFrequencies, dPtrOffsets,
                      categoryCount, patternCount);

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::IntegrateLikelihoodsPartition\n");
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
    gpu->LaunchKernel(fSumSites1,
                      bgSumSitesBlock, bgSumSitesGrid,
                      parameterCountV, totalParameterCount,
                      dArray1, dSum1, dPatternWeights,
                      patternCount);

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::SumSites1\n");
#endif

}

void KernelLauncher::SumSites1Partition(GPUPtr dArray1,
                                        GPUPtr dSum1,
                                        GPUPtr dPatternWeights,
                                        int startPattern,
                                        int endPattern,
                                        int blockCount) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::SumSites1Partition\n");
#endif

    Dim3Int bgSumSitesPartitionGrid = Dim3Int(blockCount);

    int parameterCountV = 3;
    int totalParameterCount = 5;
    gpu->LaunchKernel(fSumSites1Partition,
                      bgSumSitesBlock, bgSumSitesPartitionGrid,
                      parameterCountV, totalParameterCount,
                      dArray1, dSum1, dPatternWeights,
                      startPattern, endPattern);

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::SumSites1Partition\n");
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
    gpu->LaunchKernel(fSumSites3,
                      bgSumSitesBlock, bgSumSitesGrid,
                      parameterCountV, totalParameterCount,
                      dArray1, dSum1, dArray2, dSum2, dArray3, dSum3, dPatternWeights,
                      patternCount);

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::SumSites3\n");
#endif

}


void KernelLauncher::InnerBastaPartialsCoalescent(GPUPtr partials,
                              GPUPtr matrices,
                              GPUPtr operations,
                              const GPUPtr sizes,
                              GPUPtr coalescent,
                              unsigned int start,
                              unsigned int numOps,
                              unsigned int patternCount) {
#ifdef BEAGLE_DEBUG_FLOW
        fprintf(stderr, "\t\tEntering KernelLauncher::InnerBastaPartialsCoalescent\n");
#endif

        int parameterCountV = 5;
        int totalParameterCount = 8;
        gpu->LaunchKernel(fInnerBastaPartialsCoalescent,
                          bgBastaPeelingBlock, bgBastaPeelingGrid,
                          parameterCountV, totalParameterCount,
                          partials, matrices, operations, sizes, coalescent,
                          start, numOps, patternCount);

#ifdef BEAGLE_DEBUG_FLOW
        fprintf(stderr, "\t\tLeaving  KernelLauncher::InnerBastaPartialsCoalescent\n");
#endif

    }

// void KernelLauncher::InnerBastaPartialsCoalescent(GPUPtr partials1,
//                           GPUPtr partials2,
//                           GPUPtr partials3,
//                           GPUPtr matrices1,
//                           GPUPtr matrices2,
//                           GPUPtr accumulation1,
//                           GPUPtr accumulation2,
//                           const GPUPtr sizes,
//                           GPUPtr coalescent,
//                           unsigned int intervalNUmber,
//                           unsigned int patternCount,
//                           unsigned int child2Index) {
// #ifdef BEAGLE_DEBUG_FLOW
//         fprintf(stderr, "\t\tEntering KernelLauncher::InnerBastaPartialsCoalescent\n");
// #endif
//
//         int parameterCountV = 9;
//         int totalParameterCount = 12;
//         gpu->LaunchKernel(fInnerBastaPartialsCoalescent,
//                           bgBastaPeelingBlock, bgBastaPeelingGrid,
//                           parameterCountV, totalParameterCount,
//                           partials1, partials2, partials3, matrices1, matrices2, accumulation1, accumulation2, sizes, coalescent,
//                           intervalNUmber, patternCount, child2Index);
//
// #ifdef BEAGLE_DEBUG_FLOW
//         fprintf(stderr, "\t\tLeaving  KernelLauncher::InnerBastaPartialsCoalescent\n");
// #endif
//
// }

    void KernelLauncher::reduceWithinInterval(GPUPtr operations,
                              GPUPtr partials,
                              GPUPtr dBastaBlockResMemory,
                              GPUPtr intervals,
                              unsigned int numOps,
                              unsigned int start,
                              unsigned int end,
                              unsigned int numSubinterval) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::ReduceWithinInterval\n");
#endif

    int parameterCountV = 4;
    int totalParameterCount = 8;
    gpu->LaunchKernel(fReduceWithinInterval,
                      bgBastaReductionBlock, bgBastaReductionGrid,
                      parameterCountV, totalParameterCount,
                      operations, partials, dBastaBlockResMemory, intervals, numOps, start, end, numSubinterval);

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::ReduceWithinInterval\n");
#endif

}


    void KernelLauncher::reduceWithinIntervalMerged(GPUPtr operations,
                                                    GPUPtr partials,
                                                    GPUPtr dBastaMemory,
                                                    unsigned int numOps,
                                                    unsigned int start,
                                                    unsigned int end,
                                                    unsigned int numBlocks,
                                                    unsigned int kCoalescentBufferLength) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::ReduceWithinInterval\n");
#endif
    int parameterCountV = 3;
    int totalParameterCount = 8;
    gpu->LaunchKernel(fReduceWithinIntervalMerged,
                      bgBastaReductionBlock, bgBastaReductionGrid,
                      parameterCountV, totalParameterCount,
                      operations, partials, dBastaMemory, numOps, start, end, numBlocks, kCoalescentBufferLength);
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::ReduceWithinInterval\n");
#endif

}

    void KernelLauncher::reduceWithinIntervalSerial(GPUPtr operations,
                                                    GPUPtr partials,
                                                    GPUPtr distance,
                                                    GPUPtr dLogL,
                                                    GPUPtr sizes,
                                                    GPUPtr coalescent,
                                                    unsigned int numOps,
                                                    int start,
                                                    unsigned int end,
                                                    unsigned int intervalNUmber) {

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::ReduceWithinInterval\n");
#endif

    int parameterCountV = 6;
    int totalParameterCount = 10;
    gpu->LaunchKernel(fReduceWithinIntervalSerial,
                      bgBastaSumBlock, bgBastaSumGrid,
                      parameterCountV, totalParameterCount,
                      operations, partials, distance, dLogL, sizes, coalescent, numOps, start, end, intervalNUmber);

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::ReduceWithinInterval\n");
#endif

}

    void KernelLauncher::preProcessBastaFlags(GPUPtr dBastaInterval,
                              GPUPtr dBastaFlags,
                              GPUPtr dBlockSegmentKeysEnd,
                              unsigned int operationCount,
                              unsigned int  numBlocks) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::ReduceAcrossinInterval\n");
#endif
    int parameterCountV = 3;
    int totalParameterCount = 5;

    gpu->LaunchKernel(fPreProcessBastaFlags,
                      bgBastaPreBlock, bgBastaPreGrid,
                      parameterCountV, totalParameterCount,
                      dBastaInterval, dBastaFlags, dBlockSegmentKeysEnd, operationCount, numBlocks);
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::ReduceAcrossinInterval\n");
#endif

}


    void KernelLauncher::accumulateCarryOut(GPUPtr dBastaBlockResMemory,
                                            GPUPtr dBastaFinalResMemory,
                                            GPUPtr dBastaFlags,
                                            unsigned int numSubinterval,
                                            unsigned int  numSubintervalFinal) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::ReduceAcrossinInterval\n");
#endif
    int parameterCountV = 3;
    int totalParameterCount = 5;

    gpu->LaunchKernel(fAccumulateCarryOut,
                      bgBastaReductionBlock, bgBastaReductionGrid,
                      parameterCountV, totalParameterCount,
                      dBastaBlockResMemory, dBastaFinalResMemory, dBastaFlags, numSubinterval, numSubintervalFinal);
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::ReduceAcrossinInterval\n");
#endif

}

    void KernelLauncher::accumulateCarryOutFinal(GPUPtr dBastaFinalResMemory,
                                        GPUPtr dBastaMemory,
                                        GPUPtr dBastaFlags,
                                        unsigned int numSubinterval,
                                        unsigned int  numSubintervalFinal,
                                        unsigned int kCoalescentBufferLength) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::ReduceAcrossinInterval\n");
#endif
    int parameterCountV = 3;
    int totalParameterCount = 6;

    gpu->LaunchKernel(fAccumulateCarryOutFinal,
                      bgBastaReductionBlock, bgBastaReductionGrid,
                      parameterCountV, totalParameterCount,
                      dBastaFinalResMemory, dBastaMemory, dBastaFlags, numSubinterval, numSubintervalFinal, kCoalescentBufferLength);
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::ReduceAcrossinInterval\n");
#endif

}

void KernelLauncher::reduceAcrossIntervals(GPUPtr dBastaMemory,
                              GPUPtr distance,
                              GPUPtr dLogL,
                              const GPUPtr sizes,
                              GPUPtr coalescent,
                              unsigned int intervalNumber,
                              unsigned int kCoalescentBufferLength) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tEntering KernelLauncher::ReduceAcrossinInterval\n");
#endif
    int parameterCountV = 5;
    int totalParameterCount = 7;
    gpu->LaunchKernel(fReduceAcrossInterval,
                      bgBastaSumBlock, bgBastaSumGrid,
                      parameterCountV, totalParameterCount,
                      dBastaMemory, distance, dLogL, sizes, coalescent, intervalNumber, kCoalescentBufferLength);
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\tLeaving  KernelLauncher::ReduceAcrossinInterval\n");
#endif

}

}; // namespace



