/*
 * @author Marc Suchard
 * @author Andrew Rambaut
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#include "CUDASharedFunctions.h"
#include "beagle.h"

#define CMATRIX_SIZE		2 * PADDED_STATE_COUNT * PADDED_STATE_COUNT + 2 * PADDED_STATE_COUNT // Using memory saving format
#define MATRIX_SIZE     	PADDED_STATE_COUNT * PADDED_STATE_COUNT
#define MATRIX_CACHE_SIZE	PADDED_STATE_COUNT * PADDED_STATE_COUNT * PADDED_STATE_COUNT
#define EVAL_SIZE			PADDED_STATE_COUNT // Change to 2 * PADDED_STATE_COUNT for complex models
#define	RESTORE_VALUE	1
#define STORE_VALUE		2
#define STORE_RESTORE_MAX_LENGTH	2

#define DEVICE_NUMBER	0 // TODO Send info from wrapper
#define INSTANCE	0 // TODO Send info from wrapper

#ifdef LAZY_STORE
#define CHECK_LAZY_STORE(instance)	\
									if (!queueEmpty(&thread[instance].doStoreRestoreQueue)) \
										handleStoreRestoreQueue(instance);
#else
#define CHECK_LAZY_STORE
#endif // LAZY_STORE

typedef struct {

	int device;

#ifdef PRE_LOAD
	int loaded;
#endif

	int trueStateCount; // the "true" stateCount (without padding)
	int nodeCount;
	int patternCount;
	int truePatternCount;
	int partialsSize;
	int matrixSize;
	int matrixCount;
	int taxaCount;

	int paddedStates; // # of states to pad so that "true" + padded states = PADDED_STATE_COUNT (a multiple of 16, except for DNA models)
	int paddedPatterns; // # of patterns to pad so that (patternCount + paddedPatterns) * PADDED_STATE_COUNT is a multiple of 16

	REAL* dCMatrix;
	REAL* dStoredMatrix;
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
	REAL* dCategoryRates;
	REAL* dStoredCategoryRates;

	REAL* hCategoryRates; // TODO Remove when updateMatrices is done in parallel
	REAL *hStoredCategoryRates;

	REAL* dIntegrationTmp;

	REAL*** dPartials;
	REAL*** dMatrices;

	REAL** hTmpPartials;

	REAL*** dScalingFactors;
	REAL*** dStoredScalingFactors;
	REAL *dRootScalingFactors;
	REAL *dStoredRootScalingFactors;

	INT** dStates;

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
	REAL *hPartialsCache;
	REAL *hMatrixCache;
	REAL *hNodeCache;

	int doRescaling;
	int doStore;
	int doRestore;

#ifdef LAZY_STORE
	queue doStoreRestoreQueue;
#endif

	int sinceRescaling;

	int storedDoRescaling;
	int storedSinceRescaling;

} threadVariables;

threadVariables* thread;

int numThreads = 0;


void checkNativeMemory(void *ptr) {
	if (ptr == NULL) {
		fprintf(stderr, "Unable to allocate some memory!\n");
		exit(-1);
	}
}

void initializeInstanceMemory(int instance) {

	cudaSetDevice(thread[instance].device);
	int i;

	thread[instance].dCMatrix = allocateGPURealMemory(MATRIX_CACHE_SIZE);
	thread[instance].dStoredMatrix = allocateGPURealMemory(MATRIX_CACHE_SIZE);
	thread[instance].dEvec = allocateGPURealMemory(MATRIX_SIZE);
	thread[instance].dIevc = allocateGPURealMemory(MATRIX_SIZE);
	thread[instance].dStoredEvec = allocateGPURealMemory(MATRIX_SIZE);
	thread[instance].dStoredIevc = allocateGPURealMemory(MATRIX_SIZE);

	thread[instance].dEigenValues = allocateGPURealMemory(EVAL_SIZE);
	thread[instance].dStoredEigenValues = allocateGPURealMemory(EVAL_SIZE);

	thread[instance].dFrequencies = allocateGPURealMemory(PADDED_STATE_COUNT);
	thread[instance].dStoredFrequencies = allocateGPURealMemory(PADDED_STATE_COUNT);

	thread[instance].dCategoryRates = allocateGPURealMemory(
			thread[instance].matrixCount);
	thread[instance].hCategoryRates = (REAL *) malloc(sizeof(REAL)
			* thread[instance].matrixCount);
	thread[instance].dStoredCategoryRates = allocateGPURealMemory(
			thread[instance].matrixCount);
	thread[instance].hStoredCategoryRates = (REAL *) malloc(sizeof(REAL)
			* thread[instance].matrixCount);

	checkNativeMemory(thread[instance].hCategoryRates);
	checkNativeMemory(thread[instance].hStoredCategoryRates);

	thread[instance].dCategoryProportions = allocateGPURealMemory(
			thread[instance].matrixCount);
	thread[instance].dStoredCategoryProportions = allocateGPURealMemory(
			thread[instance].matrixCount);

	thread[instance].dIntegrationTmp = allocateGPURealMemory(
			thread[instance].patternCount);

	thread[instance].dPartials = (REAL ***) malloc(sizeof(REAL**) * 2);
	thread[instance].dPartials[0] = (REAL **) malloc(sizeof(REAL*)
			* thread[instance].nodeCount);
	thread[instance].dPartials[1] = (REAL **) malloc(sizeof(REAL*)
			* thread[instance].nodeCount);

#ifdef DYNAMIC_SCALING
	thread[instance].dScalingFactors = (REAL ***)malloc(sizeof(REAL**) * 2);
	thread[instance].dScalingFactors[0] = (REAL **)malloc(sizeof(REAL*) * thread[instance].nodeCount);
	thread[instance].dScalingFactors[1] = (REAL **)malloc(sizeof(REAL*) * thread[instance].nodeCount);
	thread[instance].dRootScalingFactors = allocateGPURealMemory(thread[instance].patternCount);
	thread[instance].dStoredRootScalingFactors = allocateGPURealMemory(thread[instance].patternCount);
#endif

	for (i = 0; i < thread[instance].nodeCount; i++) {
		thread[instance].dPartials[0][i] = allocateGPURealMemory(
				thread[instance].partialsSize);
		thread[instance].dPartials[1][i] = allocateGPURealMemory(
				thread[instance].partialsSize);

#ifdef DYNAMIC_SCALING
		thread[instance].dScalingFactors[0][i] = allocateGPURealMemory(thread[instance].patternCount);
		thread[instance].dScalingFactors[1][i] = allocateGPURealMemory(thread[instance].patternCount);
#endif
	}

	thread[instance].hCurrentMatricesIndices = (int *) malloc(sizeof(int)
			* thread[instance].nodeCount);
	thread[instance].hStoredMatricesIndices = (int *) malloc(sizeof(int)
			* thread[instance].nodeCount);
	for (i = 0; i < thread[instance].nodeCount; i++) {
		thread[instance].hCurrentMatricesIndices[i] = 0;
		thread[instance].hStoredMatricesIndices[i] = 0;
	}

	checkNativeMemory(thread[instance].hCurrentMatricesIndices);
	checkNativeMemory(thread[instance].hStoredMatricesIndices);

	thread[instance].hCurrentPartialsIndices = (int *) malloc(sizeof(int)
			* thread[instance].nodeCount);
	thread[instance].hStoredPartialsIndices = (int *) malloc(sizeof(int)
			* thread[instance].nodeCount);

#ifdef DYNAMIC_SCALING
	thread[instance].hCurrentScalingFactorsIndices = (int *)malloc(sizeof(int) * thread[instance].nodeCount);
	thread[instance].hStoredScalingFactorsIndices = (int *)malloc(sizeof(int) * thread[instance].nodeCount);
#endif

	for (i = 0; i < thread[instance].nodeCount; i++) {
		thread[instance].hCurrentPartialsIndices[i] = 0;
		thread[instance].hStoredPartialsIndices[i] = 0;
#ifdef DYNAMIC_SCALING
		thread[instance].hCurrentScalingFactorsIndices[i] = 0;
		thread[instance].hStoredScalingFactorsIndices[i] = 0;
#endif
	}

	checkNativeMemory(thread[instance].hCurrentPartialsIndices);
	checkNativeMemory(thread[instance].hStoredPartialsIndices);

#ifdef DYNAMIC_SCALING
	checkNativeMemory(thread[instance].hCurrentScalingFactorsIndices);
	checkNativeMemory(thread[instance].hStoredScalingFactorsIndices);
#endif

	//	thread[instance].dCurrentMatricesIndices = allocateGPUIntMemory(thread[instance].nodeCount);
	//	thread[instance].dStoredMatricesIndices = allocateGPUIntMemory(thread[instance].nodeCount);
	//	cudaMemcpy(thread[instance].dCurrentMatricesIndices,thread[instance].hCurrentMatricesIndices,sizeof(int)*thread[instance].nodeCount,cudaMemcpyHostToDevice);

	//	thread[instance].dCurrentPartialsIndices = allocateGPUIntMemory(thread[instance].nodeCount);
	//	thread[instance].dStoredPartialsIndices = allocateGPUIntMemory(thread[instance].nodeCount);
	//	cudaMemcpy(thread[instance].dCurrentPartialsIndices,thread[instance].hCurrentPartialsIndices,sizeof(int)*thread[instance].nodeCount,cudaMemcpyHostToDevice);

	thread[instance].dMatrices = (REAL ***) malloc(sizeof(REAL**) * 2);
	thread[instance].dMatrices[0] = (REAL **) malloc(sizeof(REAL*)
			* thread[instance].nodeCount);
	thread[instance].dMatrices[1] = (REAL **) malloc(sizeof(REAL*)
			* thread[instance].nodeCount);

	for (i = 0; i < thread[instance].nodeCount; i++) {
		thread[instance].dMatrices[0][i] = allocateGPURealMemory(MATRIX_SIZE
				* thread[instance].matrixCount);
		thread[instance].dMatrices[1][i] = allocateGPURealMemory(MATRIX_SIZE
				* thread[instance].matrixCount);
	}

	thread[instance].dStates = (INT **) malloc(sizeof(INT*)
			* thread[instance].nodeCount);
	for (i = 0; i < thread[instance].nodeCount; i++) {
		thread[instance].dStates[i] = 0; // Allocate in setStates only if state info is provided
	}

	thread[instance].dNodeIndices = allocateGPUIntMemory(
			thread[instance].nodeCount); // No execution has more no nodeCount events
	thread[instance].hNodeIndices = (int *) malloc(sizeof(int)
			* thread[instance].nodeCount);
	thread[instance].hDependencies = (int *) malloc(sizeof(int)
			* thread[instance].nodeCount);
	thread[instance].dBranchLengths = allocateGPURealMemory(
			thread[instance].nodeCount);

	checkNativeMemory(thread[instance].hNodeIndices);
	checkNativeMemory(thread[instance].hDependencies);

	thread[instance].dDistanceQueue = allocateGPURealMemory(
			thread[instance].nodeCount * thread[instance].matrixCount);
	thread[instance].hDistanceQueue = (REAL *) malloc(sizeof(REAL)
			* thread[instance].nodeCount * thread[instance].matrixCount);

	checkNativeMemory(thread[instance].hDistanceQueue);

	int len = max(5, thread[instance].matrixCount);
	SAFE_CUDA(cudaMalloc((void**) &thread[instance].dPtrQueue, sizeof(REAL*)*thread[instance].nodeCount*len),thread[instance].dPtrQueue);
	thread[instance].hPtrQueue = (REAL **) malloc(sizeof(REAL*)
			* thread[instance].nodeCount * len);

	checkNativeMemory(thread[instance].hPtrQueue);

}

void initializeDevice(int instance, int deviceNumber,
		int inNodeCount, int inStateTipCount, int inPatternCount,
		int inMatrixCount) {

#ifdef DEBUG_FLOW
	fprintf(stderr,"Entering initialize\n");
#endif

	// Increase instance storage
	numThreads++;
	thread = (threadVariables*) realloc(thread, numThreads
			* sizeof(threadVariables));

	int i;
	thread[instance].device = deviceNumber;
	thread[instance].trueStateCount = STATE_COUNT;
	thread[instance].nodeCount = inNodeCount;
	thread[instance].taxaCount = (thread[instance].nodeCount + 1) / 2;
	thread[instance].truePatternCount = inPatternCount;
	thread[instance].matrixCount = inMatrixCount;

	thread[instance].paddedStates = 0;
	thread[instance].paddedPatterns = 0;

#if (PADDED_STATE_COUNT == 4)  // DNA model
	// Make sure that patternCount + paddedPatterns is multiple of 4
	if (thread[instance].truePatternCount % 4 != 0)
	thread[instance].paddedPatterns = 4 - thread[instance].truePatternCount % 4;
	else
	thread[instance].paddedPatterns = 0;
#ifdef DEBUG
	fprintf(stderr,"Padding patterns for 4-state model:\n");
	fprintf(stderr,"\ttruePatternCount = %d\n\tpaddedPatterns = %d\n",thread[instance].truePatternCount,thread[instance].paddedPatterns);
#endif // DEBUG
#endif // DNA model
	thread[instance].patternCount = thread[instance].truePatternCount
			+ thread[instance].paddedPatterns;

	thread[instance].partialsSize = thread[instance].patternCount * PADDED_STATE_COUNT
			* thread[instance].matrixCount;

	thread[instance].hFrequenciesCache = NULL;
	thread[instance].hPartialsCache = NULL;
	thread[instance].hMatrixCache = NULL;
	thread[instance].hNodeCache = NULL;

	thread[instance].doRescaling = 1;
	thread[instance].sinceRescaling = 0;

#ifndef PRE_LOAD
	initializeInstanceMemory(instance);

#else
	// initialize temporary storage before likelihood thread exists

	thread[instance].loaded = 0;

	thread[instance].hTmpPartials = (REAL **) malloc(sizeof(REAL*)
			* thread[instance].taxaCount);

	for (i = 0; i < thread[instance].taxaCount; i++) {
		thread[instance].hTmpPartials[i] = (REAL *) malloc(
				thread[instance].partialsSize * SIZE_REAL); // TODO Don't forget to free these
	}
#endif

#ifdef LAZY_STORE
	thread[instance].doStore = 0;
	thread[instance].doRestore = 0;
	initQueue(&(thread[instance].doStoreRestoreQueue));
#endif // LAZY_STORE

#ifdef DEBUG_FLOW
	fprintf(stderr,"Exiting initialize\n");
#endif

}

void initialize(int nodeCount,
				int tipCount,
				int patternCount,
				int categoryCount,
				int matrixCount) {

	int numDevices = printGPUInfo();
	initializeDevice(INSTANCE, DEVICE_NUMBER, nodeCount, tipCount, patternCount, categoryCount);
}

void freeTmpPartials(int instance) {
	int i;
	for (i = 0; i < thread[instance].taxaCount; i++) { // TODO divide by 2
		free(thread[instance].hTmpPartials[i]);
	}
}

void freeNativeMemory(int instance) {
	int i;
	for (i = 0; i < thread[instance].nodeCount; i++) {
		freeGPUMemory(thread[instance].dPartials[0][i]);
		freeGPUMemory(thread[instance].dPartials[1][i]);
#ifdef DYNAMIC_SCALING
		freeGPUMemory(thread[instance].dScalingFactors[0][i]);
		freeGPUMemory(thread[instance].dScalingFactors[1][i]);
#endif
		freeGPUMemory(thread[instance].dMatrices[0][i]);
		freeGPUMemory(thread[instance].dMatrices[1][i]);
		freeGPUMemory(thread[instance].dStates[i]);
	}

	freeGPUMemory(thread[instance].dCMatrix);
	freeGPUMemory(thread[instance].dStoredMatrix);
	freeGPUMemory(thread[instance].dEvec);
	freeGPUMemory(thread[instance].dIevc);

	free(thread[instance].dPartials[0]);
	free(thread[instance].dPartials[1]);
	free(thread[instance].dPartials);

#ifdef DYNAMIC_SCALING
	free(thread[instance].dScalingFactors[0]);
	free(thread[instance].dScalingFactors[1]);
	free(thread[instance].dScalingFactors);
#endif

	free(thread[instance].dMatrices[0]);
	free(thread[instance].dMatrices[1]);
	free(thread[instance].dMatrices);

	free(thread[instance].dStates);

	free(thread[instance].hCurrentMatricesIndices);
	free(thread[instance].hStoredMatricesIndices);
	free(thread[instance].hCurrentPartialsIndices);
	free(thread[instance].hStoredPartialsIndices);

#ifdef DYNAMIC_SCALING
	free(thread[instance].hCurrentScalingFactorsIndices);
	free(thread[instance].hStoredScalingFactorsIndices);
#endif

	freeGPUMemory(thread[instance].dNodeIndices);
	free(thread[instance].hNodeIndices);
	free(thread[instance].hDependencies);
	freeGPUMemory(thread[instance].dBranchLengths);

	freeGPUMemory(thread[instance].dIntegrationTmp);

	free(thread[instance].hDistanceQueue);
	free(thread[instance].hPtrQueue);
	freeGPUMemory(thread[instance].dDistanceQueue);
	freeGPUMemory(thread[instance].dPtrQueue);
}

REAL *callocBEAGLE(int length, int instance) {
	REAL *ptr = (REAL *) calloc(length, SIZE_REAL);
	if (ptr == NULL) {
		fprintf(stderr,"Unable to allocate native memory!");
		exit(-1);
	}
	return ptr;
}

void finalize() {
	freeNativeMemory(INSTANCE);
}

void setTipStates(int tipIndex,
				  int* inStates) {
#ifdef DEBUG_FLOW
	fprintf(stderr,"Entering setTipStates\n");
#endif

	fprintf(stderr,"Unsupported operation!\n");
	exit(-1);

#ifdef DEBUG_FLOW
	fprintf(stderr,"Exiting setTipStates\n");
#endif
}

void setTipPartials(int tipIndex,
					REAL* inPartials) {

#ifdef DEBUG_FLOW
	fprintf(stderr,"Entering setTipPartials\n");
#endif

	int instance = INSTANCE;

	REAL *inPartialsOffset = inPartials;

	int length = thread[instance].patternCount * PADDED_STATE_COUNT;

	if (thread[instance].hPartialsCache == NULL)
		thread[instance].hPartialsCache = callocBEAGLE(length
				* thread[instance].matrixCount, instance);

	REAL *tmpRealArrayOffset = thread[instance].hPartialsCache;

	int s, p;
	for (p = 0; p < thread[instance].truePatternCount; p++) {
		memcpy(tmpRealArrayOffset,inPartialsOffset, SIZE_REAL*STATE_COUNT);

		tmpRealArrayOffset += PADDED_STATE_COUNT;
		inPartialsOffset += STATE_COUNT;
	}
	// Pad patterns as necessary
	// these should already/always be zero'd
	//for (p = 0; p < thread[instance].paddedPatterns * PADDED_STATE_COUNT; p++)
	//	tmpRealArrayOffset[p] = 0;

	// Replicate 1st copy "times" times
	int i;
	for (i = 1; i < thread[instance].matrixCount; i++) {
		memcpy(thread[instance].hPartialsCache + i * length,
				thread[instance].hPartialsCache, length * SIZE_REAL);
	}

#ifndef PRE_LOAD
	// Copy to CUDA device
	SAFE_CUDA(cudaMemcpy(thread[instance].dPartials[0][tipIndex],
					thread[instance].hPartialsCache,
					SIZE_REAL*length*thread[instance].matrixCount, cudaMemcpyHostToDevice),thread[instance].dPartials[0][tipIndex]);

#ifdef DEBUG_6
	printf("Setting tip for node %d : ",tipIndex);
	printfCudaVector(thread[instance].dPartials[0][tipIndex],thread[instance].partialsSize);
	printf("patternCount = %d, PADDED_STATE_COUNT = %d, matrixCount = %d\n",thread[instance].patternCount,PADDED_STATE_COUNT,thread[instance].matrixCount);
#endif

#else
	memcpy(thread[instance].hTmpPartials[tipIndex],
			thread[instance].hPartialsCache, SIZE_REAL
					* thread[instance].partialsSize);
#endif // PRE_LOAD

#ifdef DEBUG_FLOW
	fprintf(stderr,"Exiting setTipPartials\n");
#endif
}

void loadTipPartials(int instance) {
	int i;
	for (i = 0; i < thread[instance].taxaCount; i++) {
		cudaMemcpy(thread[instance].dPartials[0][i],
				thread[instance].hTmpPartials[i], SIZE_REAL
						* thread[instance].partialsSize, cudaMemcpyHostToDevice);
	}
}

void setStateFrequencies(REAL* inFrequencies) {

#ifdef DEBUG_FLOW
	fprintf(stderr,"Entering updateRootFreqencies\n");
#endif

	int instance = INSTANCE;

	CHECK_LAZY_STORE(instance);

#ifdef DEBUG_BEAGLE
	printfVector(inFrequencies,PADDED_STATE_COUNT);
//	exit(-1);
#endif

#ifdef PRE_LOAD
	if (thread[instance].loaded == 0) {
		initializeInstanceMemory(instance);
		loadTipPartials(instance);
		freeTmpPartials(instance);
		thread[instance].loaded = 1;
	}
#endif // PRE_LOAD

	if (thread[instance].hFrequenciesCache == NULL) {
		thread[instance].hFrequenciesCache = callocBEAGLE(PADDED_STATE_COUNT,
				instance);
	}

	memcpy(thread[instance].hFrequenciesCache,inFrequencies,STATE_COUNT*SIZE_REAL);

	cudaMemcpy(thread[instance].dFrequencies,thread[instance].hFrequenciesCache,
			SIZE_REAL*PADDED_STATE_COUNT,cudaMemcpyHostToDevice);

#ifdef DEBUG_FLOW
	fprintf(stderr,"Exiting updateRootFrequencies\n");
#endif
}

/*
 * Transposes a square matrix in place
 */
void transposeSquareMatrix(REAL *mat, int size) {
	int i, j;
	for (i = 0; i < size - 1; i++) {
		for (j = i + 1; j < size; j++) {
			REAL tmp = mat[i * size + j];
			mat[i * size + j] = mat[j * size + i];
			mat[j * size + i] = tmp;
		}
	}
}

void setEigenDecomposition(int matrixIndex,
						   REAL** inEigenVectors,
						   REAL** inInverseEigenVectors,
						   REAL* inEigenValues) {

#ifdef DEBUG_FLOW
	fprintf(stderr,"Entering updateEigenDecomposition\n");
#endif

	int instance = INSTANCE;

	CHECK_LAZY_STORE(instance);

	// Native memory packing order (length): Ievc (state^2), Evec (state^2), Eval (state), EvalImag (state)

	int length = 2 * (MATRIX_SIZE + PADDED_STATE_COUNT); // Storing extra space for complex eigenvalues

	if (thread[instance].hMatrixCache == NULL)
		thread[instance].hMatrixCache = callocBEAGLE(length, instance);

	REAL *Ievc, *tmpIevc, *Evec, *tmpEvec, *Eval, *EvalImag;

	tmpIevc = Ievc = (REAL *) thread[instance].hMatrixCache;
	tmpEvec = Evec = Ievc + MATRIX_SIZE;
	Eval = Evec + MATRIX_SIZE;

	int i, j;
	for (i = 0; i < STATE_COUNT; i++) {
		memcpy(tmpIevc,inInverseEigenVectors[i],SIZE_REAL*STATE_COUNT);
		memcpy(tmpEvec,inEigenVectors[i],SIZE_REAL*STATE_COUNT);

		tmpIevc += PADDED_STATE_COUNT;
		tmpEvec += PADDED_STATE_COUNT;
	}

	transposeSquareMatrix(Ievc, PADDED_STATE_COUNT); // Transposing matrices avoids incoherent memory read/writes
	transposeSquareMatrix(Evec, PADDED_STATE_COUNT); // TODO Only need to tranpose sub-matrix of trueStateCount

	memcpy(Eval,inEigenValues,SIZE_REAL*STATE_COUNT);

#ifdef DEBUG_BEAGLE
	printfVector(Eval,PADDED_STATE_COUNT);
	printfVector(Evec,MATRIX_SIZE);
	printfVector(Ievc,PADDED_STATE_COUNT*PADDED_STATE_COUNT);
#endif

	// Copy to CUDA device
	cudaMemcpy(thread[instance].dIevc,Ievc, SIZE_REAL*MATRIX_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(thread[instance].dEvec,Evec, SIZE_REAL*MATRIX_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(thread[instance].dEigenValues,Eval, SIZE_REAL*PADDED_STATE_COUNT, cudaMemcpyHostToDevice);

#ifdef DEBUG_FLOW
	fprintf(stderr,"Exiting updateEigenDecomposition\n");
#endif

}

void setCategoryRates(REAL* inCategoryRates) {
#ifdef DEBUG_FLOW
	fprintf(stderr,"Entering updateCategoryRates\n");
#endif

	int instance = INSTANCE;

	CHECK_LAZY_STORE(instance);

	if (thread[instance].hMatrixCache == NULL) { // TODO Is necessary?
		thread[instance].hMatrixCache = callocBEAGLE(
				thread[instance].matrixCount, instance);
	}

	cudaMemcpy(thread[instance].dCategoryRates, inCategoryRates,
			SIZE_REAL*thread[instance].matrixCount, cudaMemcpyHostToDevice);

#ifdef DEBUG_BEAGLE
	printfCudaVector(thread[instance].dCategoryRates,thread[instance].matrixCount);
#endif

	memcpy(thread[instance].hCategoryRates, inCategoryRates,
			SIZE_REAL*thread[instance].matrixCount); // TODO Can remove?

#ifdef DEBUG_FLOW
	fprintf(stderr,"Exiting updateCategoryRates\n");
#endif
}

void setCategoryProportions(REAL* inCategoryProportions) {
#ifdef DEBUG_FLOW
	fprintf(stderr,"Entering updateCategoryProportions\n");
#endif

	int instance = INSTANCE;

	CHECK_LAZY_STORE(instance);

	cudaMemcpy(thread[instance].dCategoryProportions, inCategoryProportions,
			SIZE_REAL*thread[instance].matrixCount, cudaMemcpyHostToDevice);

#ifdef DEBUG_BEAGLE
	printfCudaVector(thread[instance].dCategoryProportions,thread[instance].matrixCount);
#endif

#ifdef DEBUG_FLOW
	fprintf(stderr,"Exiting updateCategoryProportions\n");
#endif
}

void calculateProbabilityTransitionMatrices(int* nodeIndices,
                                            REAL* branchLengths,
                                            int count) {
#ifdef DEBUG_FLOW
	fprintf(stderr,"Entering updateMatrices\n");
#endif

	int instance = INSTANCE;

	CHECK_LAZY_STORE(instance);

	int x, total = 0;
	for (x = 0; x < count; x++) {
		int nodeIndex = nodeIndices[x];

		thread[instance].hCurrentMatricesIndices[nodeIndex] = 1
				- thread[instance].hCurrentMatricesIndices[nodeIndex];

		int l;
		for (l = 0; l < thread[instance].matrixCount; l++) {
			thread[instance].hPtrQueue[total]
					= thread[instance].dMatrices[thread[instance].hCurrentMatricesIndices[nodeIndex]][nodeIndex]
							+ l * MATRIX_SIZE;
			thread[instance].hDistanceQueue[total] = branchLengths[x]
					* thread[instance].hCategoryRates[l];
			total++;
		}
	}

	cudaMemcpy(thread[instance].dDistanceQueue,
			thread[instance].hDistanceQueue, SIZE_REAL*total,
			cudaMemcpyHostToDevice);
	cudaMemcpy(thread[instance].dPtrQueue, thread[instance].hPtrQueue,
			sizeof(REAL*) * total, cudaMemcpyHostToDevice);

	nativeGPUGetTransitionProbabilitiesSquare(thread[instance].dPtrQueue,
			thread[instance].dEvec, thread[instance].dIevc,
			thread[instance].dEigenValues, thread[instance].dDistanceQueue,
			total);

#ifdef DEBUG_BEAGLE
	printfCudaVector(thread[instance].hPtrQueue[0],MATRIX_SIZE);
	//exit(-1);
#endif

#ifdef DEBUG_FLOW
	fprintf(stderr,"Exiting updateMatrices\n");
#endif
}

//#define SERIAL_PARTIALS

void calculatePartials(int* operations,
					   int* dependencies,
					   int count,
					   int rescale) {

#ifdef DEBUG_FLOW
	fprintf(stderr,"Entering updatePartials\n");
#endif

	int instance = INSTANCE;

	CHECK_LAZY_STORE(instance);

#ifdef DYNAMIC_SCALING
	if (thread[instance].doRescaling == 0) // Forces rescaling on first computation
		thread[instance].doRescaling = rescale;
#endif

	// Serial version
	int op, x = 0, y = 0;
	for (op = 0; op < count; op++) {
		int nodeIndex1 = operations[x];
		x++;
		int nodeIndex2 = operations[x];
		x++;
		int nodeIndex3 = operations[x];
		x++;

		REAL *matrices1 = thread[instance].dMatrices[thread[instance].hCurrentMatricesIndices[nodeIndex1]][nodeIndex1];
		REAL *matrices2 = thread[instance].dMatrices[thread[instance].hCurrentMatricesIndices[nodeIndex2]][nodeIndex2];

		REAL *partials1 = thread[instance].dPartials[thread[instance].hCurrentPartialsIndices[nodeIndex1]][nodeIndex1];
		REAL *partials2 = thread[instance].dPartials[thread[instance].hCurrentPartialsIndices[nodeIndex2]][nodeIndex2];

		thread[instance].hCurrentPartialsIndices[nodeIndex3] = 1
				- thread[instance].hCurrentPartialsIndices[nodeIndex3];
		REAL *partials3 = thread[instance].dPartials[thread[instance].hCurrentPartialsIndices[nodeIndex3]][nodeIndex3];

#ifdef DYNAMIC_SCALING

		if( thread[instance].doRescaling )
			thread[instance].hCurrentScalingFactorsIndices[nodeIndex3] = 1 - thread[instance].hCurrentScalingFactorsIndices[nodeIndex3];

		REAL* scalingFactors = thread[instance].dScalingFactors[thread[instance].hCurrentScalingFactorsIndices[nodeIndex3]][nodeIndex3];

		nativeGPUPartialsPartialsPruningDynamicScaling(
				partials1,partials2, partials3,
				matrices1,matrices2,
				scalingFactors,
				thread[instance].patternCount, thread[instance].matrixCount, thread[instance].doRescaling);
#else
		nativeGPUPartialsPartialsPruning(partials1, partials2, partials3,
				matrices1, matrices2, thread[instance].patternCount,
				thread[instance].matrixCount);

#ifdef DEBUG_BEAGLE
		fprintf(stderr,"patternCount = %d\n",thread[instance].patternCount);
		fprintf(stderr,"truePatternCount = %d\n",thread[instance].truePatternCount);
		fprintf(stderr,"matrixCount  = %d\n",thread[instance].matrixCount);
		fprintf(stderr,"partialSize = %d\n",thread[instance].partialsSize);//		printfCudaVector(partials1,thread[instance].partialsSize);
		printfCudaVector(partials1,thread[instance].partialsSize);
		printfCudaVector(partials2,thread[instance].partialsSize);
		printfCudaVector(partials3,thread[instance].partialsSize);
//		exit(-1);
#endif
#endif // DYNAMIC_SCALING

	}

#ifdef DEBUG_FLOW
	fprintf(stderr,"Exiting updatePartials\n");
#endif

}

void calculateLogLikelihoods(int rootNodeIndex,
							 REAL* outLogLikelihoods) {
#ifdef DEBUG_FLOW
	fprintf(stderr,"Entering calculateLogLikelihoods\n");
#endif
	int instance = INSTANCE;

	CHECK_LAZY_STORE(instance);

#ifdef DYNAMIC_SCALING

	if (thread[instance].doRescaling) {
		// Construct node-list for scalingFactors
		int n;
		int length = thread[instance].nodeCount - thread[instance].taxaCount;
		for(n=0; n<length; n++)
			thread[instance].hPtrQueue[n] = thread[instance].dScalingFactors[thread[instance].hCurrentScalingFactorsIndices[n+thread[instance].taxaCount]][n+thread[instance].taxaCount];

		cudaMemcpy(thread[instance].dPtrQueue,thread[instance].hPtrQueue,sizeof(REAL*)*length, cudaMemcpyHostToDevice);

		// Computer scaling factors at the root
		nativeGPUComputeRootDynamicScaling(thread[instance].dPtrQueue,thread[instance].dRootScalingFactors,length,thread[instance].patternCount);
	}

	thread[instance].doRescaling = 0;

	nativeGPUIntegrateLikelihoodsDynamicScaling(thread[instance].dIntegrationTmp, thread[instance].dPartials[thread[instance].hCurrentPartialsIndices[rootNodeIndex]][rootNodeIndex],
			thread[instance].dCategoryProportions, thread[instance].dFrequencies,
			thread[instance].dRootScalingFactors,
			thread[instance].patternCount, thread[instance].matrixCount, thread[instance].nodeCount);

#else
	nativeGPUIntegrateLikelihoods(
			thread[instance].dIntegrationTmp,
			thread[instance].dPartials[thread[instance].hCurrentPartialsIndices[rootNodeIndex]][rootNodeIndex],
			thread[instance].dCategoryProportions,
			thread[instance].dFrequencies, thread[instance].patternCount,
			thread[instance].matrixCount);
#endif // DYNAMIC_SCALING

#ifdef DEBUG
	fprintf(stderr,"logLike = ");
	printfCudaVector(thread[instance].dIntegrationTmp,thread[instance].truePatternCount);
//	exit(-1);
#endif

	cudaMemcpy(outLogLikelihoods,thread[instance].dIntegrationTmp,
			SIZE_REAL*thread[instance].truePatternCount, cudaMemcpyDeviceToHost);

#ifdef DEBUG_FLOW
	fprintf(stderr,"Exiting calculateLogLikelihoods\n");
#endif
}

void handleStoreRestoreQueue(int instance) {

#ifdef DEBUG_FLOW
	fprintf(stderr,"Entering handleStoreRestoreQueue: ");
	printQueue(&thread[instance].doStoreRestoreQueue);
#endif

	while (!queueEmpty(&thread[instance].doStoreRestoreQueue)) {
		int command = deQueue(&thread[instance].doStoreRestoreQueue);
		switch (command) {
		case STORE_VALUE:
			doStore(instance);
			break;
		case RESTORE_VALUE:
			doRestore(instance);
			break;
		default:
			fprintf(stderr,"Illegal command in Store/Restore queue!");
			exit(-1);
		}
	}

#ifdef DEBUG_FLOW
	fprintf(stderr,"Exiting handleStoreRestoreQueue\n");
#endif
}

void doStore(int instance) {

#ifdef DEBUG_FLOW
	fprintf(stderr,"Entering doStore\n");
#endif

	storeGPURealMemoryArray(thread[instance].dStoredEigenValues,
			thread[instance].dEigenValues, EVAL_SIZE);
	storeGPURealMemoryArray(thread[instance].dStoredEvec,
			thread[instance].dEvec, MATRIX_SIZE);
	storeGPURealMemoryArray(thread[instance].dStoredIevc,
			thread[instance].dIevc, MATRIX_SIZE);

	storeGPURealMemoryArray(thread[instance].dStoredFrequencies,
			thread[instance].dFrequencies, PADDED_STATE_COUNT);
	storeGPURealMemoryArray(thread[instance].dStoredCategoryRates,
			thread[instance].dCategoryRates, thread[instance].matrixCount);
	memcpy(thread[instance].hStoredCategoryRates,
			thread[instance].hCategoryRates, thread[instance].matrixCount
					* sizeof(REAL));
	storeGPURealMemoryArray(thread[instance].dStoredCategoryProportions,
			thread[instance].dCategoryProportions, thread[instance].matrixCount);

	memcpy(thread[instance].hStoredMatricesIndices,
			thread[instance].hCurrentMatricesIndices, sizeof(int)
					* thread[instance].nodeCount);
	memcpy(thread[instance].hStoredPartialsIndices,
			thread[instance].hCurrentPartialsIndices, sizeof(int)
					* thread[instance].nodeCount);

#ifdef DYNAMIC_SCALING
	memcpy(thread[instance].hStoredScalingFactorsIndices, thread[instance].hCurrentScalingFactorsIndices, sizeof(int) * thread[instance].nodeCount);
	storeGPURealMemoryArray(thread[instance].dStoredRootScalingFactors, thread[instance].dRootScalingFactors, thread[instance].patternCount);
#endif

#ifdef DEBUG_FLOW
	fprintf(stderr,"Exiting doStore\n");
#endif
}

void storeState() {
#ifdef DEBUG_FLOW
	fprintf(stderr,"Entering storeState\n");
#endif

	int instance = INSTANCE;

#ifdef LAZY_STORE
	enQueue(&thread[instance].doStoreRestoreQueue, STORE_VALUE);
#else
	doStore(instance);
#endif

#ifdef DEBUG_FLOW
	fprintf(stderr,"Exiting storeState\n");
#endif
}

void doRestore(int instance) {
#ifdef DEBUG_FLOW
	fprintf(stderr,"Entering doRestore\n");
#endif
	// Rather than copying the stored stuff back, just swap the pointers...
	REAL* tmp = thread[instance].dCMatrix;
	thread[instance].dCMatrix = thread[instance].dStoredMatrix;
	thread[instance].dStoredMatrix = tmp;

	tmp = thread[instance].dEvec;
	thread[instance].dEvec = thread[instance].dStoredEvec;
	thread[instance].dStoredEvec = tmp;

	tmp = thread[instance].dIevc;
	thread[instance].dIevc = thread[instance].dStoredIevc;
	thread[instance].dStoredIevc = tmp;

	tmp = thread[instance].dEigenValues;
	thread[instance].dEigenValues = thread[instance].dStoredEigenValues;
	thread[instance].dStoredEigenValues = tmp;

	tmp = thread[instance].dFrequencies;
	thread[instance].dFrequencies = thread[instance].dStoredFrequencies;
	thread[instance].dStoredFrequencies = tmp;

	tmp = thread[instance].dCategoryRates;
	thread[instance].dCategoryRates = thread[instance].dStoredCategoryRates;
	thread[instance].dStoredCategoryRates = tmp;

	tmp = thread[instance].hCategoryRates;
	thread[instance].hCategoryRates = thread[instance].hStoredCategoryRates;
	thread[instance].hStoredCategoryRates = tmp;

	tmp = thread[instance].dCategoryProportions;
	thread[instance].dCategoryProportions
			= thread[instance].dStoredCategoryProportions;
	thread[instance].dStoredCategoryProportions = tmp;

	int* tmp2 = thread[instance].hCurrentMatricesIndices;
	thread[instance].hCurrentMatricesIndices
			= thread[instance].hStoredMatricesIndices;
	thread[instance].hStoredMatricesIndices = tmp2;

	tmp2 = thread[instance].hCurrentPartialsIndices;
	thread[instance].hCurrentPartialsIndices
			= thread[instance].hStoredPartialsIndices;
	thread[instance].hStoredPartialsIndices = tmp2;

#ifdef DYNAMIC_SCALING
	tmp2 = thread[instance].hCurrentScalingFactorsIndices;
	thread[instance].hCurrentScalingFactorsIndices = thread[instance].hStoredScalingFactorsIndices;
	thread[instance].hStoredScalingFactorsIndices = tmp2;
	tmp = thread[instance].dRootScalingFactors;
	thread[instance].dRootScalingFactors = thread[instance].dStoredRootScalingFactors;
	thread[instance].dStoredRootScalingFactors = tmp;
#endif

#ifdef DEBUG_FLOW
	fprintf(stderr,"Exiting doRestore\n");
#endif

}

void restoreState() {
#ifdef DEBUG_FLOW
	fprintf(stderr,"Entering restoreState\n");
#endif

	int instance = INSTANCE;

#ifdef LAZY_STORE
	enQueue(&thread[instance].doStoreRestoreQueue, RESTORE_VALUE);
#else
	doRestore(instance);
#endif

#ifdef DEBUG_FLOW
	fprintf(stderr,"Exiting restoreState\n");
#endif

}

int printGPUInfo() {
	char* nativeName = "*** Marc is too lazy to write this function!";

	int cDevices;
	CUresult status;
	status = cuInit(0);
	if (CUDA_SUCCESS != status)
		return 0;
	status = cuDeviceGetCount(&cDevices);
	if (CUDA_SUCCESS != status)
		return 0;
	if (cDevices == 0) {
		return 0;
	}

	fprintf(stderr,"GPU Device Information:");

	int iDevice;
	for (iDevice = 0; iDevice < cDevices; iDevice++) {

		char name[256];
		int totalGlobalMemory = 0;
		int clockSpeed = 0;

		// New CUDA functions in cutil.h do not work in JNI files
		getGPUInfo(iDevice, name, &totalGlobalMemory, &clockSpeed);
		fprintf(stderr,"\nDevice #%d: %s\n",(iDevice+1),name);
		double mem = totalGlobalMemory / 1024.0 / 1024.0;
		double clo = clockSpeed / 1000000.0;
		fprintf(stderr,"\tGlobal Memory (MB) : %1.2f\n",mem);
		fprintf(stderr,"\tClock Speed (Ghz)  : %1.2f\n",clo);
	}

	if (cDevices == 0)
		fprintf(stderr,"None found.\n");

	return cDevices;
}

