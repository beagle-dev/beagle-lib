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
 * @brief GPU kernel launcher
 *
 * @author Marc Suchard
 * @author Daniel Ayres
 */

#ifndef __KernelLauncher__
#define __KernelLauncher__

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include "libhmsbeagle/GPU/GPUImplDefs.h"
#include "libhmsbeagle/GPU/GPUInterface.h"

class KernelLauncher {
private:
    GPUInterface* gpu;
    
    GPUFunction fMatrixMulADB;
    GPUFunction fMatrixMulADBFirstDeriv;
    GPUFunction fMatrixMulADBSecondDeriv;

    GPUFunction fPartialsPartialsByPatternBlockCoherent;
    GPUFunction fPartialsPartialsByPatternBlockAutoScaling;
    GPUFunction fPartialsPartialsByPatternBlockFixedScaling;
    GPUFunction fPartialsPartialsByPatternBlockCheckScaling;
    GPUFunction fPartialsPartialsByPatternBlockFixedCheckScaling;
    GPUFunction fStatesPartialsByPatternBlockCoherent;
    GPUFunction fStatesPartialsByPatternBlockFixedScaling;
    GPUFunction fStatesStatesByPatternBlockCoherent;
    GPUFunction fStatesStatesByPatternBlockFixedScaling;
    GPUFunction fPartialsPartialsEdgeLikelihoods;
    GPUFunction fPartialsPartialsEdgeLikelihoodsSecondDeriv;
    GPUFunction fStatesPartialsEdgeLikelihoods;
    GPUFunction fStatesPartialsEdgeLikelihoodsSecondDeriv;
        
    GPUFunction fIntegrateLikelihoodsDynamicScaling;
    GPUFunction fIntegrateLikelihoodsDynamicScalingSecondDeriv;
    GPUFunction fAccumulateFactorsDynamicScaling;
    GPUFunction fAccumulateFactorsAutoScaling;
    GPUFunction fRemoveFactorsDynamicScaling;
    GPUFunction fPartialsDynamicScaling;
    GPUFunction fPartialsDynamicScalingAccumulate;
    GPUFunction fPartialsDynamicScalingAccumulateDifference;
    GPUFunction fPartialsDynamicScalingAccumulateReciprocal;
    GPUFunction fPartialsDynamicScalingSlow;
    GPUFunction fIntegrateLikelihoods;
    GPUFunction fIntegrateLikelihoodsSecondDeriv;
	GPUFunction fIntegrateLikelihoodsMulti;
	GPUFunction fIntegrateLikelihoodsFixedScaleMulti;
    GPUFunction fIntegrateLikelihoodsAutoScaling;

    GPUFunction fSumSites1;
    GPUFunction fSumSites2;
    GPUFunction fSumSites3;
    
    Dim3Int bgTransitionProbabilitiesBlock;
    Dim3Int bgTransitionProbabilitiesGrid;
    Dim3Int bgPeelingBlock;
    Dim3Int bgPeelingGrid;
    Dim3Int bgLikelihoodBlock;
    Dim3Int bgLikelihoodGrid;
    Dim3Int bgAccumulateBlock;
    Dim3Int bgAccumulateGrid;
    Dim3Int bgScaleBlock;
    Dim3Int bgScaleGrid;
    Dim3Int bgSumSitesBlock;
    Dim3Int bgSumSitesGrid;
    
    unsigned int kPaddedStateCount;
    unsigned int kCategoryCount;
    unsigned int kPatternCount;
    unsigned int kPatternBlockSize;
    unsigned int kMatrixBlockSize;
    unsigned int kSlowReweighing;  
    unsigned int kMultiplyBlockSize;
    unsigned int kSumSitesBlockSize;
    long kFlags;
    
public:
    KernelLauncher(GPUInterface* inGpu);
    
    ~KernelLauncher();
    
// Kernel links
#ifdef CUDA
    void GetTransitionProbabilitiesSquare(GPUPtr dMatrices,
                                          GPUPtr dPtrQueue,
                                          GPUPtr dEvec,
                                          GPUPtr dIevc,
                                          GPUPtr dEigenValues,
                                          GPUPtr distanceQueue,
                                          unsigned int totalMatrix);

    void GetTransitionProbabilitiesSquareFirstDeriv(GPUPtr dMatrices,
                                                    GPUPtr dPtrQueue,
                                                     GPUPtr dEvec,
                                                     GPUPtr dIevc,
                                                     GPUPtr dEigenValues,
                                                     GPUPtr distanceQueue,
                                                     unsigned int totalMatrix);    
    
    void GetTransitionProbabilitiesSquareSecondDeriv(GPUPtr dMatrices,
                                                     GPUPtr dPtrQueue,
                                          GPUPtr dEvec,
                                          GPUPtr dIevc,
                                          GPUPtr dEigenValues,
                                          GPUPtr distanceQueue,
                                          unsigned int totalMatrix);

#else //OpenCL
    void GetTransitionProbabilitiesSquare(GPUPtr dPtr,
                                          GPUPtr dEvec,
                                          GPUPtr dIevc,
                                          GPUPtr dEigenValues,
                                          GPUPtr distanceQueue,
                                          unsigned int totalMatrix,
                                          unsigned int index);    
#endif
    
    void PartialsPartialsPruningDynamicCheckScaling(GPUPtr partials1,
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
                                                    int sizeReal);
    
    void PartialsPartialsPruningDynamicScaling(GPUPtr partials1,
                                               GPUPtr partials2,
                                               GPUPtr partials3,
                                               GPUPtr matrices1,
                                               GPUPtr matrices2,
                                               GPUPtr scalingFactors,
                                               GPUPtr cumulativeScaling,
                                               unsigned int patternCount,
                                               unsigned int categoryCount,
                                               int doRescaling);
    
    void StatesPartialsPruningDynamicScaling(GPUPtr states1,
                                             GPUPtr partials2,
                                             GPUPtr partials3,
                                             GPUPtr matrices1,
                                             GPUPtr matrices2,
                                             GPUPtr scalingFactors,
                                             GPUPtr cumulativeScaling,
                                             unsigned int patternCount,
                                             unsigned int categoryCount,
                                             int doRescaling);
    
    void StatesStatesPruningDynamicScaling(GPUPtr states1,
                                           GPUPtr states2,
                                           GPUPtr partials3,
                                           GPUPtr matrices1,
                                           GPUPtr matrices2,
                                           GPUPtr scalingFactors,
                                           GPUPtr cumulativeScaling,
                                           unsigned int patternCount,
                                           unsigned int categoryCount,
                                           int doRescaling);
    
    void IntegrateLikelihoodsDynamicScaling(GPUPtr dResult,
                                            GPUPtr dRootPartials,
                                            GPUPtr dWeights,
                                            GPUPtr dFrequencies,
                                            GPUPtr dRootScalingFactors,
                                            unsigned int patternCount,
                                            unsigned int categoryCount);
    
    void IntegrateLikelihoodsAutoScaling(GPUPtr dResult,
                                            GPUPtr dRootPartials,
                                            GPUPtr dWeights,
                                            GPUPtr dFrequencies,
                                            GPUPtr dRootScalingFactors,
                                            unsigned int patternCount,
                                            unsigned int categoryCount);
    
    void IntegrateLikelihoodsDynamicScalingSecondDeriv(GPUPtr dResult,
                                                       GPUPtr dFirstDerivResult,
                                                       GPUPtr dSecondDerivResult,
                                                       GPUPtr dRootPartials,
                                                       GPUPtr dRootFirstDeriv,
                                                       GPUPtr dRootSecondDeriv,
                                                       GPUPtr dWeights,
                                                       GPUPtr dFrequencies,
                                                       GPUPtr dRootScalingFactors,
                                                       unsigned int patternCount,
                                                       unsigned int categoryCount);
    
    void PartialsPartialsEdgeLikelihoods(GPUPtr dPartialsTmp,
                                         GPUPtr dParentPartials,
                                         GPUPtr dChildParials,
                                         GPUPtr dTransMatrix,
                                         unsigned int patternCount,
                                         unsigned int categoryCount);
    
    void PartialsPartialsEdgeLikelihoodsSecondDeriv(GPUPtr dPartialsTmp,
                                                    GPUPtr dFirstDerivTmp,
                                                    GPUPtr dSecondDerivTmp,
                                                    GPUPtr dParentPartials,
                                                    GPUPtr dChildParials,
                                                    GPUPtr dTransMatrix,
                                                    GPUPtr dFirstDerivMatrix,
                                                    GPUPtr dSecondDerivMatrix,
                                                    unsigned int patternCount,
                                                    unsigned int categoryCount);
    
    
    void StatesPartialsEdgeLikelihoods(GPUPtr dPartialsTmp,
                                       GPUPtr dParentPartials,
                                       GPUPtr dChildStates,
                                       GPUPtr dTransMatrix,
                                       unsigned int patternCount,
                                       unsigned int categoryCount);
    
    void StatesPartialsEdgeLikelihoodsSecondDeriv(GPUPtr dPartialsTmp,
                                                  GPUPtr dFirstDerivTmp,
                                                  GPUPtr dSecondDerivTmp,
                                                  GPUPtr dParentPartials,
                                                  GPUPtr dChildStates,
                                                  GPUPtr dTransMatrix,
                                                  GPUPtr dFirstDerivMatrix,
                                                  GPUPtr dSecondDerivMatrix,
                                                  unsigned int patternCount,
                                                  unsigned int categoryCount);
    
    void AccumulateFactorsDynamicScaling(GPUPtr dScalingFactors,
                                         GPUPtr dNodePtrQueue,
                                         GPUPtr dRootScalingFactors,
                                         unsigned int nodeCount,
                                         unsigned int patternCount);

    void AccumulateFactorsAutoScaling(GPUPtr dScalingFactors,
                                      GPUPtr dNodePtrQueue,
                                      GPUPtr dRootScalingFactors,
                                      unsigned int nodeCount,
                                      unsigned int patternCount,
                                      unsigned int scaleBufferSize);
    
    void RemoveFactorsDynamicScaling(GPUPtr dScalingFactors,
                                     GPUPtr dNodePtrQueue,
                                     GPUPtr dRootScalingFactors,
                                     unsigned int nodeCount,
                                     unsigned int patternCount);    
    
    void RescalePartials(GPUPtr partials3,
                         GPUPtr scalingFactors,
                         GPUPtr cumulativeScaling,
                         unsigned int patternCount,
                         unsigned int categoryCount,
                         unsigned int fillWithOnes);

    void IntegrateLikelihoods(GPUPtr dResult,
                              GPUPtr dRootPartials,
                              GPUPtr dWeights,
                              GPUPtr dFrequencies,
                              unsigned int patternCount,
                              unsigned int categoryCount);
    
    void IntegrateLikelihoodsSecondDeriv(GPUPtr dResult,
                                         GPUPtr dFirstDerivResult,
                                         GPUPtr dSecondDerivResult,
                                         GPUPtr dRootPartials,
                                         GPUPtr dRootFirstDeriv,
                                         GPUPtr dRootSecondDeriv,
                                         GPUPtr dWeights,
                                         GPUPtr dFrequencies,
                                         unsigned int patternCount,
                                         unsigned int categoryCount);
    
	void IntegrateLikelihoodsMulti(GPUPtr dResult,
								   GPUPtr dRootPartials,
								   GPUPtr dWeights,
								   GPUPtr dFrequencies,
								   unsigned int patternCount,
								   unsigned int categoryCount,
								   unsigned int takeLog);
	
	void IntegrateLikelihoodsFixedScaleMulti(GPUPtr dResult,
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
											 unsigned int subsetIndex);
    
    void SumSites1(GPUPtr dArray1,
                  GPUPtr dSum1,
                  GPUPtr dPatternWeights,
                  unsigned int patternCount);
    
    void SumSites2(GPUPtr dArray1,
                  GPUPtr dSum1,
                  GPUPtr dArray2,
                  GPUPtr dSum2,
                  GPUPtr dPatternWeights,
                  unsigned int patternCount);
    
    void SumSites3(GPUPtr dArray1,
                  GPUPtr dSum1,
                  GPUPtr dArray2,
                  GPUPtr dSum2,
                  GPUPtr dArray3,
                  GPUPtr dSum3,
                  GPUPtr dPatternWeights,
                  unsigned int patternCount);
	
    void SetupKernelBlocksAndGrids();
    
protected:
    void LoadKernels();

};
#endif // __KernelLauncher__
