/*
 *  BeagleCUDAImpl.h
 *  BEAGLE
 *
 * @author Marc Suchard
 * @author Andrew Rambaut
 *
 */

#include "BeagleImpl.h"
#include "CUDASharedFunctions.h"
#include "Queue.h"

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

	int paddedStates; // # of states to pad so that "true" + padded states = PADDED_STATE_COUNT (a multiple of 16, except for DNA models)
	int paddedPatterns; // # of patterns to pad so that (patternCount + paddedPatterns) * PADDED_STATE_COUNT is a multiple of 16

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

#ifdef LAZY_STORE
//	queue doStoreRestoreQueue;
	Queue doStoreRestoreQueue;
#endif

	int sinceRescaling;
	int storedDoRescaling;
	int storedSinceRescaling;

public:
	virtual int initialize(
					int nodeCount,
					int tipCount,
					int stateCount,
					int patternCount,
					int categoryCount,
					int matrixCount);

	virtual void finalize();

	virtual void setTipPartials(
						int tipIndex,
						double* inPartials);

	virtual void setTipStates(
					  int tipIndex,
					  int* inStates);

	virtual void setStateFrequencies(double* inStateFrequencies);

	virtual void setEigenDecomposition(
							   int matrixIndex,
							   double** inEigenVectors,
							   double** inInverseEigenVectors,
							   double* inEigenValues);

	virtual void setCategoryRates(double* inCategoryRates);

	virtual void setCategoryProportions(double* inCategoryProportions);

	virtual void calculateProbabilityTransitionMatrices(
												int* nodeIndices,
												double* branchLengths,
												int count);

	virtual void calculatePartials(
						   int* operations,
						   int* dependencies,
						   int count,
						   int rescale);

	virtual void calculateLogLikelihoods(
								 int rootNodeIndex,
								 double* outLogLikelihoods);

	virtual void storeState();

	virtual void restoreState();

private:
    void updateStatesStates(int nodeIndex1, int nodeIndex2, int nodeIndex3);
    void updateStatesPartials(int nodeIndex1, int nodeIndex2, int nodeIndex3);
    void updatePartialsPartials(int nodeIndex1, int nodeIndex2, int nodeIndex3);

    void handleStoreRestoreQueue();
    void doRestoreState();
    void doStoreState();
    void loadTipPartialsOrStates();

    void freeNativeMemory();
    void freeTmpPartialsOrStates();

    void initializeDevice(int deviceNumber,
				          int inNodeCount,
				          int inStateTipCount,
				          int inPatternCount,
				          int inCategoryCount,
				          int inMatrixCount);
    void initializeInstanceMemory();


    int getGPUDeviceCount();
    void printGPUInfo(int device);
    void getGPUInfo(int iDevice, char *name, int *memory, int *speed);

    void transposeSquareMatrix(REAL *mat, int size);

    long memoryRequirement(int taxaCount, int stateCount);

};

class BeagleCUDAImplFactory : public BeagleImplFactory {
    public:
    	virtual BeagleImpl* createImpl(
								int nodeCount,
    							int tipCount,
    							int stateCount,
    							int patternCount,
    							int categoryCount,
    							int matrixCount);
    	virtual const char* getName();
};

// Kernel links

extern "C" void nativeGPUIntegrateLikelihoods(REAL *dResult, REAL *dRootPartials, REAL *dCategoryProportions, REAL *dFrequencies,
		int patternCount, int matrixCount);

extern "C" void nativeGPUPartialsPartialsPruning(
	REAL* partials1, REAL* partials2, REAL* partials3, REAL* matrices1, REAL* matrices2,
	const unsigned int patternCount, const unsigned int matrixCount);

extern "C" void nativeGPUStatesStatesPruning(
	INT* states1, INT* states2, REAL* partials3, REAL* matrices1, REAL* matrices2,
	const unsigned int patternCount, const unsigned int matrixCount);

extern "C" void nativeGPUStatesPartialsPruning(
	INT* states1, REAL* partials2, REAL* partials3, REAL* matrices1, REAL* matrices2,
	const unsigned int patternCount, const unsigned int matrixCount);


extern "C" void nativeGPUGetTransitionProbabilitiesSquare(REAL **dPtrQueue, REAL *dEvec,
		REAL *dIevc, REAL *dEigenValues, REAL *distanceQueue, int totalMatrix);

extern "C" void nativeGPUPartialsPartialsPruningDynamicScaling(
	REAL* partials1, REAL* partials2, REAL* partials3, REAL* matrices1, REAL* matrices2, REAL *scalingFactors,
	const unsigned int patternCount, const unsigned int matrixCount, int doRescaling);

extern "C" void nativeGPUStatesStatesPruningDynamicScaling(
	INT* states1, INT* states2, REAL* partials3, REAL* matrices1, REAL* matrices2, REAL* scalingFactors,
	const unsigned int patternCount, const unsigned int matrixCount, int doRescaling);

extern "C" void nativeGPUStatesPartialsPruningDynamicScaling(
	INT* states1, REAL* partials2, REAL* partials3, REAL* matrices1, REAL* matrices2, REAL* scalingFactors,
	const unsigned int patternCount, const unsigned int matrixCount, int doRescaling);

extern "C" void nativeGPUComputeRootDynamicScaling(REAL **dNodePtrQueue, REAL *dRootScalingFactors,
		int nodeCount, int patternCount);

extern "C" void nativeGPUIntegrateLikelihoodsDynamicScaling(REAL *dResult, REAL *dRootPartials, REAL *dCategoryProportions, REAL *dFrequencies,
		REAL *dRootScalingFactors,
		int patternCount, int matrixCount, int nodeCount);

extern "C" void nativeGPURescalePartials(REAL *partials3,REAL *scalingFactors,
		int patternCount, int matrixCount, int fillWithOnes);
