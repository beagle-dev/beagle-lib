/*
 * @file BeagleCUDAImpl.h
 * 
 * @brief CUDA implementation header
 *
 * @author Marc Suchard
 * @author Andrew Rambaut
 * @author Daniel Ayres
 */

#include "libbeagle-lib/BeagleImpl.h"
#include "libbeagle-lib/CUDA/CUDASharedFunctions.h"

namespace beagle {
namespace cuda {

class BeagleCUDAImpl : public BeagleImpl {
private:
    
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
    
    REAL* dEigenValues;
    REAL* dEvec;
    REAL* dIevc;
    
    REAL* dWeights;
    REAL* dFrequencies; 
    REAL* dIntegrationTmp;
    REAL* dPartialsTmp;
    
    REAL** dPartials;
    REAL** dMatrices;
    
    REAL** hTmpTipPartials;
    int** hTmpStates;
    
    REAL** dScalingFactors;
    REAL* dRootScalingFactors;
    
    int** dStates;
    
    int** dCompactBuffers;
    REAL** dTipPartialsBuffers;
    
    REAL* dBranchLengths;
    
    REAL* hDistanceQueue;
    REAL* dDistanceQueue;
    
    REAL** hPtrQueue;
    REAL** dPtrQueue;
    
    REAL* hWeightsCache;
    REAL* hFrequenciesCache;
    REAL* hLogLikelihoodsCache;
    REAL* hPartialsCache;
    int* hStatesCache;
    REAL* hMatrixCache;
    
public:
    virtual ~BeagleCUDAImpl();
    
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
    
private:
    void checkNativeMemory(void* ptr);
    
    void loadTipPartialsAndStates();
    
    void freeMemory();
    void freeTmpTipPartialsAndStates();
    
    void initializeDevice(int deviceNumber,
                          int inTipCount,
                          int inPartialsBufferCount,
                          int inCompactBufferCount,
                          int inStateCount,
                          int inPatternCount,
                          int inEigenDecompositionCount,
                          int inMatrixCount);
    
    void initializeInstanceMemory();
    
    int getGPUDeviceCount();
    
    void printGPUInfo(int device);
    
    void getGPUInfo(int iDevice,
                    char* name,
                    int* memory,
                    int* speed);

/**
  * @brief Transposes a square matrix in place
  */
    void transposeSquareMatrix(REAL* mat,
                               int size);
    
/**
 * @brief Computes the device memory requirements
 */    
    long memoryRequirement(int taxaCount,
                           int stateCount);
    
};

class BeagleCUDAImplFactory : public BeagleImplFactory {
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

}	// namespace cuda
}	// namespace beagle

// Kernel links
extern "C" {
    void nativeGPUGetTransitionProbabilitiesSquare(REAL** dPtrQueue,
                                                   REAL* dEvec,
                                                   REAL* dIevc,
                                                   REAL* dEigenValues,
                                                   REAL* distanceQueue,
                                                   int totalMatrix);
    
    void nativeGPUPartialsPartialsPruningDynamicScaling(REAL* partials1,
                                                        REAL* partials2,
                                                        REAL* partials3,
                                                        REAL* matrices1,
                                                        REAL* matrices2,
                                                        REAL* scalingFactors,
                                                        const unsigned int patternCount,
                                                        const unsigned int matrixCount,
                                                        int doRescaling);
    
    void nativeGPUStatesPartialsPruningDynamicScaling(INT* states1,
                                                      REAL* partials2,
                                                      REAL* partials3,
                                                      REAL* matrices1,
                                                      REAL* matrices2,
                                                      REAL* scalingFactors,
                                                      const unsigned int patternCount,
                                                      const unsigned int matrixCount,
                                                      int doRescaling);
    
    void nativeGPUStatesStatesPruningDynamicScaling(INT* states1,
                                                    INT* states2,
                                                    REAL* partials3,
                                                    REAL* matrices1,
                                                    REAL* matrices2,
                                                    REAL* scalingFactors,
                                                    const unsigned int patternCount,
                                                    const unsigned int matrixCount,
                                                    int doRescaling);

    void nativeGPUIntegrateLikelihoodsDynamicScaling(REAL* dResult,
                                                     REAL* dRootPartials,
                                                     REAL* dCategoryProportions,
                                                     REAL* dFrequencies,
                                                     REAL* dRootScalingFactors,
                                                     int patternCount,
                                                     int matrixCount);
    
    void nativeGPUPartialsPartialsEdgeLikelihoods(REAL* dPartialsTmp,
                                                  REAL* dParentPartials,
                                                  REAL* dChildParials,
                                                  REAL* dTransMatrix,
                                                  int patternCount,
                                                  int count);
    
    void nativeGPUStatesPartialsEdgeLikelihoods(REAL* dPartialsTmp,
                                                REAL* dParentPartials,
                                                INT* dChildStates,
                                                REAL* dTransMatrix,
                                                int patternCount,
                                                int count);
    
    void nativeGPUComputeRootDynamicScaling(REAL** dNodePtrQueue,
                                            REAL* dRootScalingFactors,
                                            int nodeCount,
                                            int patternCount);
    
    void nativeGPURescalePartials(REAL* partials3,
                                  REAL* scalingFactors,
                                  int patternCount,
                                  int matrixCount,
                                  int fillWithOnes);
    
    void nativeGPUPartialsPartialsPruning(REAL* partials1,
                                          REAL* partials2,
                                          REAL* partials3,
                                          REAL* matrices1,
                                          REAL* matrices2,
                                          const unsigned int patternCount,
                                          const unsigned int matrixCount);
    
    void nativeGPUStatesPartialsPruning(INT* states1,
                                        REAL* partials2,
                                        REAL* partials3,
                                        REAL* matrices1,
                                        REAL* matrices2,
                                        const unsigned int patternCount,
                                        const unsigned int matrixCount);
    
    void nativeGPUStatesStatesPruning(INT* states1,
                                      INT* states2,
                                      REAL* partials3,
                                      REAL* matrices1,
                                      REAL* matrices2,
                                      const unsigned int patternCount,
                                      const unsigned int matrixCount);
    
    void nativeGPUIntegrateLikelihoods(REAL* dResult,
                                       REAL* dRootPartials,
                                       REAL* dCategoryProportions,
                                       REAL* dFrequencies,
                                       int patternCount,
                                       int matrixCount);
} // extern "C"

