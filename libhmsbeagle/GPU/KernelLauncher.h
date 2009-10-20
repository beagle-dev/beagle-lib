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
    GPUFunction fMatrixMulAB;
    GPUFunction fMatrixMulDComplexB;
    GPUFunction fMatrixMulADBComplex;

    GPUFunction fPartialsPartialsByPatternBlockCoherent;
    GPUFunction fPartialsPartialsByPatternBlockFixedScaling;
    GPUFunction fStatesPartialsByPatternBlockCoherent;
    GPUFunction fStatesPartialsByPatternBlockFixedScaling;
    GPUFunction fStatesStatesByPatternBlockCoherent;
    GPUFunction fStatesStatesByPatternBlockFixedScaling;
    GPUFunction fPartialsPartialsEdgeLikelihoods;
    GPUFunction fStatesPartialsEdgeLikelihoods;
    
    GPUFunction fPartialsPartialsByPatternBlockCoherentSmall;
    GPUFunction fPartialsPartialsByPatternBlockSmallFixedScaling;
    GPUFunction fStatesPartialsByPatternBlockCoherentSmall;
    GPUFunction fStatesStatesByPatternBlockCoherentSmall;
    GPUFunction fPartialsPartialsEdgeLikelihoodsSmall;
    GPUFunction fStatesPartialsEdgeLikelihoodsSmall;
    
    GPUFunction fIntegrateLikelihoodsDynamicScaling;
    GPUFunction fAccumulateFactorsDynamicScaling;
    GPUFunction fRemoveFactorsDynamicScaling;
    GPUFunction fPartialsDynamicScaling;
    GPUFunction fPartialsDynamicScalingAccumulate;
    GPUFunction fPartialsDynamicScalingSlow;
    GPUFunction fIntegrateLikelihoods;    
    
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
    
    int kPaddedStateCount;
    int kCategoryCount;
    int kPatternCount;
    int kPatternBlockSize;
    int kMatrixBlockSize;
    int kSlowReweighing;  
    int kMultiplyBlockSize;
    long kFlags;
    
public:
    KernelLauncher(GPUInterface* inGpu);
    
    ~KernelLauncher();
    
// Kernel links
#ifdef CUDA
    void GetTransitionProbabilitiesSquare(GPUPtr dPtrQueue,
                                          GPUPtr dEvec,
                                          GPUPtr dIevc,
                                          GPUPtr dEigenValues,
                                          GPUPtr distanceQueue,
                                          int totalMatrix);

    void GetTransitionProbabilitiesComplex(GPUPtr dPtrQueue,
                                           GPUPtr dEvec,
                                           GPUPtr dIevc,
                                           GPUPtr dEigenValues,
                                           GPUPtr distanceQueue,
                                           GPUPtr dComplex,
                                           int totalMatrix);

#else //OpenCL
    void GetTransitionProbabilitiesSquare(GPUPtr dPtr,
                                          GPUPtr dEvec,
                                          GPUPtr dIevc,
                                          GPUPtr dEigenValues,
                                          GPUPtr distanceQueue,
                                          int totalMatrix,
                                          int index);    
#endif
    
    void PartialsPartialsPruningDynamicScaling(GPUPtr partials1,
                                               GPUPtr partials2,
                                               GPUPtr partials3,
                                               GPUPtr matrices1,
                                               GPUPtr matrices2,
                                               GPUPtr scalingFactors,
                                               GPUPtr cumulativeScaling,
                                               const unsigned int patternCount,
                                               const unsigned int categoryCount,
                                               int doRescaling);
    
    void StatesPartialsPruningDynamicScaling(GPUPtr states1,
                                             GPUPtr partials2,
                                             GPUPtr partials3,
                                             GPUPtr matrices1,
                                             GPUPtr matrices2,
                                             GPUPtr scalingFactors,
                                             GPUPtr cumulativeScaling,
                                             const unsigned int patternCount,
                                             const unsigned int categoryCount,
                                             int doRescaling);
    
    void StatesStatesPruningDynamicScaling(GPUPtr states1,
                                           GPUPtr states2,
                                           GPUPtr partials3,
                                           GPUPtr matrices1,
                                           GPUPtr matrices2,
                                           GPUPtr scalingFactors,
                                           GPUPtr cumulativeScaling,
                                           const unsigned int patternCount,
                                           const unsigned int categoryCount,
                                           int doRescaling);
    
    void IntegrateLikelihoodsDynamicScaling(GPUPtr dResult,
                                            GPUPtr dRootPartials,
                                            GPUPtr dWeights,
                                            GPUPtr dFrequencies,
                                            GPUPtr dRootScalingFactors,
                                            int patternCount,
                                            int categoryCount);
    
    void PartialsPartialsEdgeLikelihoods(GPUPtr dPartialsTmp,
                                         GPUPtr dParentPartials,
                                         GPUPtr dChildParials,
                                         GPUPtr dTransMatrix,
                                         int patternCount,
                                         int categoryCount);
    
    void StatesPartialsEdgeLikelihoods(GPUPtr dPartialsTmp,
                                       GPUPtr dParentPartials,
                                       GPUPtr dChildStates,
                                       GPUPtr dTransMatrix,
                                       int patternCount,
                                       int categoryCount);
    
    void AccumulateFactorsDynamicScaling(GPUPtr dNodePtrQueue,
                                         GPUPtr dRootScalingFactors,
                                         int nodeCount,
                                         int patternCount);

    void RemoveFactorsDynamicScaling(GPUPtr dNodePtrQueue,
                                     GPUPtr dRootScalingFactors,
                                     int nodeCount,
                                     int patternCount);    
    
    void RescalePartials(GPUPtr partials3,
                         GPUPtr scalingFactors,
                         GPUPtr cumulativeScaling,
                         int patternCount,
                         int categoryCount,
                         int fillWithOnes);
        
    void IntegrateLikelihoods(GPUPtr dResult,
                              GPUPtr dRootPartials,
                              GPUPtr dWeights,
                              GPUPtr dFrequencies,
                              int patternCount,
                              int categoryCount);
    
    void SetupKernelBlocksAndGrids();
    
protected:
    void LoadKernels();

};
#endif // __KernelLauncher__
