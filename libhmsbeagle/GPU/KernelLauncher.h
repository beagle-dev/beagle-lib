/*
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
    
public:
    KernelLauncher(GPUInterface* inGpu);
    
    ~KernelLauncher();
    
// Kernel links
    void GetTransitionProbabilitiesSquare(GPUPtr dPtrQueue,
                                          GPUPtr dEvec,
                                          GPUPtr dIevc,
                                          GPUPtr dEigenValues,
                                          GPUPtr distanceQueue,
                                          int totalMatrix);
    
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
    
    void PartialsPartialsPruning(GPUPtr partials1,
                                 GPUPtr partials2,
                                 GPUPtr partials3,
                                 GPUPtr matrices1,
                                 GPUPtr matrices2,
                                 const unsigned int patternCount,
                                 const unsigned int categoryCount);
    
    void StatesPartialsPruning(GPUPtr states1,
                               GPUPtr partials2,
                               GPUPtr partials3,
                               GPUPtr matrices1,
                               GPUPtr matrices2,
                               const unsigned int patternCount,
                               const unsigned int categoryCount);
    
    void StatesStatesPruning(GPUPtr states1,
                             GPUPtr states2,
                             GPUPtr partials3,
                             GPUPtr matrices1,
                             GPUPtr matrices2,
                             const unsigned int patternCount,
                             const unsigned int categoryCount);
    
    void IntegrateLikelihoods(GPUPtr dResult,
                              GPUPtr dRootPartials,
                              GPUPtr dWeights,
                              GPUPtr dFrequencies,
                              int patternCount,
                              int categoryCount);
};
#endif // __KernelLauncher__
