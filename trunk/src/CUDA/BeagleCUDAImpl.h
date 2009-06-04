/*
 *  BeagleCUDAImpl.h
 *  BEAGLE
 *
 * @author Marc Suchard
 * @author Andrew Rambaut
 * @author Daniel Ayres
 */

#include "BeagleImpl.h"
#include "CUDASharedFunctions.h"

class BeagleCUDAImpl : public BeagleImpl {
private:
	
	int device;
	
#ifdef PRE_LOAD
	int loaded;
#endif
	
	int nodeCount;
	int patternCount;
	int truePatternCount;
	int partialsSize;
	int matrixSize;
	int categoryCount;
	int matrixCount;
	int taxaCount;
	
	int paddedStates;	// # of states to pad so that "true" + padded states
						// = PADDED_STATE_COUNT (a multiple of 16, except for DNA models)
	int paddedPatterns; // # of patterns to pad so that (patternCount + paddedPatterns)
						// * PADDED_STATE_COUNT is a multiple of 16
	
	//	REAL* dCMatrix;
	//	REAL* dStoredMatrix;
	REAL* dEigenValues;
	REAL* dStoredEigenValues;
	REAL* dEvec;
	REAL* dStoredEvec;
	REAL* dIevc;
	REAL* dStoredIevc;
	
	REAL* dFrequencies;
	REAL* dStoredFrequencies;
	REAL* dCategoryProportions;
	REAL* dStoredCategoryProportions;
	//	REAL* dCategoryRates;
	//	REAL* dStoredCategoryRates;
	
	REAL* hCategoryRates;
	REAL *hStoredCategoryRates;
	
	REAL* dIntegrationTmp;
	
	REAL*** dPartials;
	REAL*** dMatrices;
	
	REAL** hTmpPartials;
	int** hTmpStates;
	
	REAL*** dScalingFactors;
	REAL*** dStoredScalingFactors;
	REAL *dRootScalingFactors;
	REAL *dStoredRootScalingFactors;
	
	int** dStates;
	
	int* hCurrentMatricesIndices;
	int* hStoredMatricesIndices;
	int* hCurrentPartialsIndices;
	int* hStoredPartialsIndices;
	
#ifdef DYNAMIC_SCALING
	int *hCurrentScalingFactorsIndices;
	int *hStoredScalingFactorsIndices;
#endif
	
	int* dCurrentMatricesIndices;
	int* dStoredMatricesIndices;
	int* dCurrentPartialsIndices;
	int* dStoredPartialsIndices;
	
	int* dNodeIndices;
	int* hNodeIndices;
	int* hDependencies;
	REAL* dBranchLengths;
	
	REAL* dExtraCache;
	
	REAL* hDistanceQueue;
	REAL* dDistanceQueue;
	
	REAL** hPtrQueue;
	REAL** dPtrQueue;
	
	REAL *hFrequenciesCache;
	REAL *hCategoryCache;
	REAL *hLogLikelihoodsCache;
	REAL *hPartialsCache;
	int  *hStatesCache;
	REAL *hMatrixCache;
	//	REAL *hNodeCache;
	
	int doRescaling;
	int doStore;
	int doRestore;
		
	int sinceRescaling;
	int storedDoRescaling;
	int storedSinceRescaling;
	
public:
	virtual ~BeagleCUDAImpl();
	
	int initialize(int tipCount,
				   int partialBufferCount,
				   int compactBufferCount,
				   int stateCount,
				   int patternCount,
				   int eigenDecompositionCount,
				   int matrixCount);
	
	int setPartials(int bufferIndex,
					const double* inPartials);
	
	int getPartials(int bufferIndex,
					double *outPartials);
	
	int setTipStates(int tipIndex,
					 const int* inStates);
	
	int setStateFrequencies(const double* inStateFrequencies);
	
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
									const double* weights,
									const double* stateFrequencies,
									int count,
									double* outLogLikelihoods);
	
	int calculateEdgeLogLikelihoods(const int* parentBufferIndices,
									const int* childBufferIndices,
									const int* probabilityIndices,
									const int* firstDerivativeIndices,
									const int* secondDerivativeIndices,
									const double* weights,
									const double* stateFrequencies,
									int count,
									double* outLogLikelihoods,
									double* outFirstDerivatives,
									double* outSecondDerivatives);
	
private:
    void updateStatesStates(int nodeIndex1,
							int nodeIndex2,
							int nodeIndex3);
	
    void updateStatesPartials(int nodeIndex1,
							  int nodeIndex2,
							  int nodeIndex3);
	
    void updatePartialsPartials(int nodeIndex1,
								int nodeIndex2,
								int nodeIndex3);
    
	void loadTipPartialsOrStates();
	
    void freeNativeMemory();
    void freeTmpPartialsOrStates();
	
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
					char *name,
					int *memory,
					int *speed);
	
    void transposeSquareMatrix(REAL *mat,
							   int size);
	
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

// Kernel links
extern "C" void nativeGPUIntegrateLikelihoods(REAL *dResult,
											  REAL *dRootPartials,
											  REAL *dCategoryProportions,
											  REAL *dFrequencies,
											  int patternCount,
											  int matrixCount);

extern "C" void nativeGPUPartialsPartialsPruning(REAL* partials1,
												 REAL* partials2,
												 REAL* partials3,
												 REAL* matrices1,
												 REAL* matrices2,
												 const unsigned int patternCount,
												 const unsigned int matrixCount);

extern "C" void nativeGPUStatesStatesPruning(INT* states1,
											 INT* states2,
											 REAL* partials3,
											 REAL* matrices1,
											 REAL* matrices2,
											 const unsigned int patternCount,
											 const unsigned int matrixCount);

extern "C" void nativeGPUStatesPartialsPruning(INT* states1,
											   REAL* partials2,
											   REAL* partials3,
											   REAL* matrices1,
											   REAL* matrices2,
											   const unsigned int patternCount,
											   const unsigned int matrixCount);


extern "C" void nativeGPUGetTransitionProbabilitiesSquare(REAL **dPtrQueue,
														  REAL *dEvec,
														  REAL *dIevc,
														  REAL *dEigenValues,
														  REAL *distanceQueue,
														  int totalMatrix);

extern "C" void nativeGPUPartialsPartialsPruningDynamicScaling(REAL* partials1,
															   REAL* partials2,
															   REAL* partials3,
															   REAL* matrices1,
															   REAL* matrices2,
															   REAL *scalingFactors,
															   const unsigned int patternCount,
															   const unsigned int matrixCount,
															   int doRescaling);

extern "C" void nativeGPUStatesStatesPruningDynamicScaling(INT* states1,
														   INT* states2,
														   REAL* partials3,
														   REAL* matrices1,
														   REAL* matrices2,
														   REAL* scalingFactors,
														   const unsigned int patternCount,
														   const unsigned int matrixCount,
														   int doRescaling);

extern "C" void nativeGPUStatesPartialsPruningDynamicScaling(INT* states1,
															 REAL* partials2,
															 REAL* partials3,
															 REAL* matrices1,
															 REAL* matrices2,
															 REAL* scalingFactors,
															 const unsigned int patternCount,
															 const unsigned int matrixCount,
															 int doRescaling);

extern "C" void nativeGPUComputeRootDynamicScaling(REAL **dNodePtrQueue,
												   REAL *dRootScalingFactors,
												   int nodeCount,
												   int patternCount);

extern "C" void nativeGPUIntegrateLikelihoodsDynamicScaling(REAL *dResult,
															REAL *dRootPartials,
															REAL *dCategoryProportions,
															REAL *dFrequencies,
															REAL *dRootScalingFactors,
															int patternCount,
															int matrixCount,
															int nodeCount);

extern "C" void nativeGPURescalePartials(REAL *partials3,
										 REAL *scalingFactors,
										 int patternCount,
										 int matrixCount,
										 int fillWithOnes);
