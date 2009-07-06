/*
 * @file BeagleGPUImpl.h
 * 
 * @brief GPU implementation header
 *
 * @author Marc Suchard
 * @author Andrew Rambaut
 * @author Daniel Ayres
 */

#ifndef __BeagleGPUImpl__
#define __BeagleGPUImpl__

#ifdef HAVE_CONFIG_H
#include "libbeagle-lib/config.h"
#endif

#include "libbeagle-lib/BeagleImpl.h"
#include "libbeagle-lib/GPU/GPUImplDefs.h"
#include "libbeagle-lib/GPU/GPUInterface.h"
#include "libbeagle-lib/GPU/KernelLauncher.h"

namespace beagle {
namespace gpu {

class BeagleGPUImpl : public BeagleImpl {
private:
    GPUInterface* gpu;
    KernelLauncher* kernels;
    
    int kDevice;
    
    int kDeviceMemoryAllocated;
    
    int kTipCount;
    int kPartialsBufferCount;
    int kCompactBufferCount;
    int kStateCount;
    int kPatternCount;
    int kEigenDecompCount;
    int kMatrixCount;
 
    int kTipPartialsBufferCount;
    int kBufferCount;
    
    int kPaddedStateCount;
    int kPaddedPatternCount;    // total # of patterns with padding so that kPaddedPatternCount
                                //   * kPaddedStateCount is a multiple of 16

    int kPartialsSize;
    int kMatrixSize;
    int kEigenValuesSize;
    
    int kDoRescaling;

    int kLastCompactBufferIndex;
    int kLastTipPartialsBufferIndex;
    
    GPUPtr dEigenValues;
    GPUPtr dEvec;
    GPUPtr dIevc;
    
    GPUPtr dWeights;
    GPUPtr dFrequencies; 
    GPUPtr dIntegrationTmp;
    GPUPtr dPartialsTmp;
    
    GPUPtr* dPartials;
    GPUPtr* dMatrices;
    
    REAL** hTmpTipPartials;
    int** hTmpStates;
    
    GPUPtr* dScalingFactors;
    GPUPtr dRootScalingFactors;
    
    GPUPtr* dStates;
    
    GPUPtr* dCompactBuffers;
    GPUPtr* dTipPartialsBuffers;
    
    GPUPtr dBranchLengths;
    
    REAL* hDistanceQueue;
    GPUPtr dDistanceQueue;
    
    GPUPtr* hPtrQueue;
    GPUPtr dPtrQueue;
    
    REAL* hWeightsCache;
    REAL* hFrequenciesCache;
    REAL* hLogLikelihoodsCache;
    REAL* hPartialsCache;
    int* hStatesCache;
    REAL* hMatrixCache;
    
public:
    virtual ~BeagleGPUImpl();
    
    int createInstance(int tipCount,
                       int partialsBufferCount,
                       int compactBufferCount,
                       int stateCount,
                       int patternCount,
                       int eigenDecompositionCount,
                       int matrixCount);
    
    int initializeInstance(InstanceDetails* retunInfo);
    
    int setPartials(int bufferIndex,
                    const double* inPartials);
    
    int getPartials(int bufferIndex,
                    double* outPartials);
    
    int setTipStates(int tipIndex,
                     const int* inStates);
    
    int setEigenDecomposition(int eigenIndex,
                              const double* inEigenVectors,
                              const double* inInverseEigenVectors,
                              const double* inEigenValues);
    
    int setTransitionMatrix(int matrixIndex,
                            const double* inMatrix);
    
    int updateTransitionMatrices(int eigenIndex,
                                 const int* probabilityIndices,
                                 const int* firstDerivativeIndices,
                                 const int* secondDervativeIndices,
                                 const double* edgeLengths,
                                 int count);
    
    int updatePartials(const int* operations,
                       int operationCount,
                       int rescale);
    
    int waitForPartials(const int* destinationPartials,
                        int destinationPartialsCount);
    
    int calculateRootLogLikelihoods(const int* bufferIndices,
                                    const double* inWeights,
                                    const double* inStateFrequencies,
                                    int count,
                                    double* outLogLikelihoods);
    
    int calculateEdgeLogLikelihoods(const int* parentBufferIndices,
                                    const int* childBufferIndices,
                                    const int* probabilityIndices,
                                    const int* firstDerivativeIndices,
                                    const int* secondDerivativeIndices,
                                    const double* inWeights,
                                    const double* inStateFrequencies,
                                    int count,
                                    double* outLogLikelihoods,
                                    double* outFirstDerivatives,
                                    double* outSecondDerivatives);    
};

class BeagleGPUImplFactory : public BeagleImplFactory {
public:
    virtual BeagleImpl* createImpl(int tipCount,
                                   int partialsBufferCount,
                                   int compactBufferCount,
                                   int stateCount,
                                   int patternCount,
                                   int eigenBufferCount,
                                   int matrixBufferCount);

    virtual const char* getName();
};

}	// namespace gpu
}	// namespace beagle

#endif // __BeagleGPUImpl__
