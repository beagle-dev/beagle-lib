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
#include "libbeagle-lib/config.h"
#endif

#include "libbeagle-lib/GPU/GPUImplDefs.h"
#include "libbeagle-lib/GPU/GPUInterface.h"

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
    GPUFunction fComputeRootDynamicScaling;
    GPUFunction fPartialsDynamicScaling;
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
                                               const unsigned int patternCount,
                                               int doRescaling);
    
    void StatesPartialsPruningDynamicScaling(GPUPtr states1,
                                             GPUPtr partials2,
                                             GPUPtr partials3,
                                             GPUPtr matrices1,
                                             GPUPtr matrices2,
                                             GPUPtr scalingFactors,
                                             const unsigned int patternCount,
                                             int doRescaling);
    
    void StatesStatesPruningDynamicScaling(GPUPtr states1,
                                           GPUPtr states2,
                                           GPUPtr partials3,
                                           GPUPtr matrices1,
                                           GPUPtr matrices2,
                                           GPUPtr scalingFactors,
                                           const unsigned int patternCount,
                                           int doRescaling);
    
    void IntegrateLikelihoodsDynamicScaling(GPUPtr dResult,
                                            GPUPtr dRootPartials,
                                            GPUPtr dWeights,
                                            GPUPtr dFrequencies,
                                            GPUPtr dRootScalingFactors,
                                            int patternCount,
                                            int count);
    
    void PartialsPartialsEdgeLikelihoods(GPUPtr dPartialsTmp,
                                         GPUPtr dParentPartials,
                                         GPUPtr dChildParials,
                                         GPUPtr dTransMatrix,
                                         int patternCount);
    
    void StatesPartialsEdgeLikelihoods(GPUPtr dPartialsTmp,
                                       GPUPtr dParentPartials,
                                       GPUPtr dChildStates,
                                       GPUPtr dTransMatrix,
                                       int patternCount);
    
    void ComputeRootDynamicScaling(GPUPtr dNodePtrQueue,
                                   GPUPtr dRootScalingFactors,
                                   int nodeCount,
                                   int patternCount);
    
    void RescalePartials(GPUPtr partials3,
                         GPUPtr scalingFactors,
                         int patternCount,
                         int fillWithOnes);
    
    void PartialsPartialsPruning(GPUPtr partials1,
                                 GPUPtr partials2,
                                 GPUPtr partials3,
                                 GPUPtr matrices1,
                                 GPUPtr matrices2,
                                 const unsigned int patternCount);
    
    void StatesPartialsPruning(GPUPtr states1,
                               GPUPtr partials2,
                               GPUPtr partials3,
                               GPUPtr matrices1,
                               GPUPtr matrices2,
                               const unsigned int patternCount);
    
    void StatesStatesPruning(GPUPtr states1,
                             GPUPtr states2,
                             GPUPtr partials3,
                             GPUPtr matrices1,
                             GPUPtr matrices2,
                             const unsigned int patternCount);
    
    void IntegrateLikelihoods(GPUPtr dResult,
                              GPUPtr dRootPartials,
                              GPUPtr dWeights,
                              GPUPtr dFrequencies,
                              int patternCount,
                              int count);
};
#endif // __KernelLauncher__
