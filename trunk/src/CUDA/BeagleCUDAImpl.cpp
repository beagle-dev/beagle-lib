
/*
 *  BeagleCUDAImpl.cpp
 *  BEAGLE
 *
 * @author Marc Suchard
 * @author Andrew Rambaut
 * @author Daniel Ayres
 */

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#include "beagle.h"
#include "BeagleCUDAImpl.h"
#include "CUDASharedFunctions.h"

#define MATRIX_SIZE     PADDED_STATE_COUNT * PADDED_STATE_COUNT
#define EVAL_SIZE		PADDED_STATE_COUNT // Change to 2 * PADDED_STATE_COUNT for complex models

int currentDevice = -1;

BeagleCUDAImpl::~BeagleCUDAImpl() {
	//freeNativeMemory();
}

int BeagleCUDAImpl::initialize(int tipCount,
							   int partialBufferCount,
							   int compactBufferCount,
							   int stateCount,
							   int patternCount,
							   int eigenDecompositionCount,
							   int matrixCount) {
	
	// TODO Determine if CUDA device satisfies memory requirements.
	
	int numDevices = getGPUDeviceCount();
	if (numDevices == 0) {
		fprintf(stderr,"No GPU devices found");
		return GENERAL_ERROR;
	}
	
	// Static load balancing; each instance gets added to the next available device
	currentDevice++;
	if (currentDevice == numDevices)
		currentDevice = 0;
	printGPUInfo(currentDevice);
	
	initializeDevice(currentDevice,
					 tipCount,
					 partialBufferCount,
					 compactBufferCount,
					 stateCount,
					 patternCount,
					 eigenDecompositionCount,
					 matrixCount);
	
	return NO_ERROR;
}

int BeagleCUDAImpl::setPartials(int bufferIndex,
								const double* inPartials) {
#ifdef DEBUG_FLOW
	fprintf(stderr,"Entering setTipPartials\n");
#endif
	
	const double* inPartialsOffset = inPartials;
	REAL* tmpRealArrayOffset = hPartialsCache;
	
	for (int i = 0; i < truePatternCount; i++) {
#ifdef DOUBLE_PRECISION
		memcpy(tmpRealArrayOffset, inPartialsOffset, SIZE_REAL * kStateCount);
#else
		MEMCPY(tmpRealArrayOffset, inPartialsOffset, kStateCount, REAL);
#endif
		tmpRealArrayOffset += PADDED_STATE_COUNT;
		inPartialsOffset += kStateCount;
	}
	
#ifndef PRE_LOAD
	// Copy to CUDA device
	SAFE_CUDA(cudaMemcpy(dPartials[0][bufferIndex],
						 hPartialsCache,
						 SIZE_REAL * partialsSize,
						 cudaMemcpyHostToDevice),
			  dPartials[0][bufferIndex]);
#else
	memcpy(hTmpPartials[bufferIndex], hPartialsCache, SIZE_REAL * partialsSize);
#endif // PRE_LOAD
	
#ifdef DEBUG_FLOW
	fprintf(stderr,"Exiting setTipPartials\n");
#endif
	
	return NO_ERROR;
}

int BeagleCUDAImpl::getPartials(int bufferIndex,
								double* inPartials) {
	//TODO: implement getPartials
	assert (false);
}

int BeagleCUDAImpl::setTipStates(int tipIndex,
								 const int* inStates) {
	//TODO: update and test setTipStates
	
#ifdef DEBUG_FLOW
	fprintf(stderr, "Entering setTipStates\n");
#endif
	
	//	memcpy(hStatesCache,inStates,SIZE_INT*truePatternCount);
	for(int i = 0; i < truePatternCount; i++) {
		hStatesCache[i] = inStates[i];
		if (hStatesCache[i] >= STATE_COUNT)
			hStatesCache[i] = PADDED_STATE_COUNT;
	}
	// Padded extra patterns
	for(int i = 0; i < paddedPatterns; i++)
		hStatesCache[truePatternCount + i] = PADDED_STATE_COUNT;
	
#ifndef PRE_LOAD
	// Copy to CUDA device
	SAFE_CUDA(cudaMemcpy(dStates[tipIndex],
						 inStates,
						 SIZE_INT * patternCount,
						 cudaMemcpyHostToDevice),
			  dStates[tipIndex]);
#else
	
	hTmpStates[tipIndex] = (int*) malloc(SIZE_INT * patternCount);
	
	memcpy(hTmpStates[tipIndex],
		   hStatesCache,
		   SIZE_INT * patternCount);
#endif // PRE_LOAD
	
#ifdef DEBUG_FLOW
	fprintf(stderr, "Exiting setTipStates\n");
#endif
	
	return NO_ERROR;
}

int BeagleCUDAImpl::setEigenDecomposition(int matrixIndex,
										  const double* inEigenVectors,
										  const double* inInverseEigenVectors,
										  const double* inEigenValues) {
	
#ifdef DEBUG_FLOW
	fprintf(stderr,"Entering updateEigenDecomposition\n");
#endif
	
	// Native memory packing order (length): Ievc (state^2), Evec (state^2),
	//	Eval (state), EvalImag (state)
	
	REAL* Ievc, * tmpIevc, * Evec, * tmpEvec, * Eval, * EvalImag;
	
	tmpIevc = Ievc = (REAL*) hMatrixCache;
	tmpEvec = Evec = Ievc + MATRIX_SIZE;
	Eval = Evec + MATRIX_SIZE;
	
	for (int i = 0; i < kStateCount; i++) {
#ifdef DOUBLE_PRECISION
		memcpy(tmpIevc, inInverseEigenVectors + i * kStateCount, SIZE_REAL * kStateCount);
		memcpy(tmpEvec, inEigenVectors + i * kStateCount, SIZE_REAL * kStateCount);
#else
		MEMCPY(tmpIevc, (inInverseEigenVectors + i * kStateCount), kStateCount, REAL);
		MEMCPY(tmpEvec, (inEigenVectors + i * kStateCount), kStateCount, REAL);
#endif
		
		tmpIevc += PADDED_STATE_COUNT;
		tmpEvec += PADDED_STATE_COUNT;
	}
	
	// Transposing matrices avoids incoherent memory read/writes	
	transposeSquareMatrix(Ievc, PADDED_STATE_COUNT);
	
	// TODO: Only need to tranpose sub-matrix of trueStateCount
	transposeSquareMatrix(Evec, PADDED_STATE_COUNT);
	
#ifdef DOUBLE_PRECISION
	memcpy(Eval, inEigenValues, SIZE_REAL * STATE_COUNT);
#else
	MEMCPY(Eval, inEigenValues, STATE_COUNT, REAL);
#endif
	
#ifdef DEBUG_BEAGLE
#ifdef DOUBLE_PRECISION
	printfVectorD(Eval, PADDED_STATE_COUNT);
	printfVectorD(Evec, MATRIX_SIZE);
	printfVectorD(Ievc, PADDED_STATE_COUNT * PADDED_STATE_COUNT);
#else
	printfVectorF(Eval, PADDED_STATE_COUNT);
	printfVectorF(Evec, MATRIX_SIZE);
	printfVectorF(Ievc, PADDED_STATE_COUNT * PADDED_STATE_COUNT);
#endif
#endif
	
	// Copy to CUDA device
	cudaMemcpy(dIevc,Ievc, SIZE_REAL * MATRIX_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(dEvec,Evec, SIZE_REAL * MATRIX_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(dEigenValues,Eval, SIZE_REAL * PADDED_STATE_COUNT, cudaMemcpyHostToDevice);
	
#ifdef DEBUG_BEAGLE
	printfCudaVector(dEigenValues, PADDED_STATE_COUNT);
	printfCudaVector(dEvec, MATRIX_SIZE);
	printfCudaVector(dIevc, PADDED_STATE_COUNT * PADDED_STATE_COUNT);
#endif
	
#ifdef DEBUG_FLOW
	fprintf(stderr, "Exiting updateEigenDecomposition\n");
#endif
	
	return NO_ERROR;
}

int BeagleCUDAImpl::setTransitionMatrix(int matrixIndex,
										const double* inMatrix) {
	//TODO: implement setTransitionMatrix
	assert(false);
}

int BeagleCUDAImpl::updateTransitionMatrices(int eigenIndex,
											 const int* probabilityIndices,
											 const int* firstDerivativeIndices,
											 const int* secondDervativeIndices,
											 const double* edgeLengths,
											 int count) {
#ifdef DEBUG_FLOW
	fprintf(stderr,"Entering updateMatrices\n");
#endif
	
	for (int i = 0; i < count; i++) {
		hPtrQueue[i] = dMatrices[0][probabilityIndices[i]];
		hDistanceQueue[i] = (REAL) edgeLengths[i];
	}
	
	cudaMemcpy(dDistanceQueue, hDistanceQueue, SIZE_REAL * count, cudaMemcpyHostToDevice);
	cudaMemcpy(dPtrQueue, hPtrQueue, sizeof(REAL*) * count, cudaMemcpyHostToDevice);
	
	// Set-up and call GPU kernel
	nativeGPUGetTransitionProbabilitiesSquare(dPtrQueue, dEvec, dIevc,
											  dEigenValues, dDistanceQueue, total);
	
#ifdef DEBUG_BEAGLE
	printfCudaVector(hPtrQueue[0], MATRIX_SIZE);
#endif
	
#ifdef DEBUG_FLOW
	fprintf(stderr, "Exiting updateMatrices\n");
#endif
	
	return NO_ERROR;
}

int BeagleCUDAImpl::updatePartials(const int* operations,
								   int operationCount,
								   int rescale) {
#ifdef DEBUG_FLOW
	fprintf(stderr, "Entering updatePartials\n");
#endif
	
#ifdef DYNAMIC_SCALING
	if (doRescaling == 0) // Forces rescaling on first computation
		doRescaling = rescale;
#endif
	
	int die = 0;
	
	// Serial version
	int op, x = 0, y = 0;
	for (op = 0; op < operationCount; op++) {
		int nodeIndex1 = operations[x];
		x++;
		int nodeIndex2 = operations[x];
		x++;
		int nodeIndex3 = operations[x];
		x++;
		
		REAL* matrices1 = dMatrices[hCurrentMatricesIndices[nodeIndex1]][nodeIndex1];
		REAL* matrices2 = dMatrices[hCurrentMatricesIndices[nodeIndex2]][nodeIndex2];
		
		REAL* partials1 = dPartials[hCurrentPartialsIndices[nodeIndex1]][nodeIndex1];
		REAL* partials2 = dPartials[hCurrentPartialsIndices[nodeIndex2]][nodeIndex2];
		
		hCurrentPartialsIndices[nodeIndex3] = 1 - hCurrentPartialsIndices[nodeIndex3];
		REAL* partials3 = dPartials[hCurrentPartialsIndices[nodeIndex3]][nodeIndex3];
		
		int* tipStates1 = dStates[nodeIndex1];
		int* tipStates2 = dStates[nodeIndex2];
		
#ifdef DYNAMIC_SCALING
		if( doRescaling )
			hCurrentScalingFactorsIndices[nodeIndex3] =
			1 - hCurrentScalingFactorsIndices[nodeIndex3];
		
		REAL* scalingFactors = 
		dScalingFactors[hCurrentScalingFactorsIndices[nodeIndex3]][nodeIndex3];
		
		if (tipStates1 != 0) {
			if (tipStates2 != 0 ) {
				nativeGPUStatesStatesPruningDynamicScaling(tipStates1, tipStates2, partials3,
														   matrices1, matrices2, scalingFactors,
														   patternCount, categoryCount,
														   doRescaling);
			} else {
				nativeGPUStatesPartialsPruningDynamicScaling(tipStates1, partials2, partials3,
															 matrices1, matrices2, scalingFactors,
															 patternCount, categoryCount,
															 doRescaling);
				die = 1;
			}
		} else {
			if (tipStates2 != 0) {
				nativeGPUStatesPartialsPruningDynamicScaling(tipStates2, partials1, partials3,
															 matrices2, matrices1, scalingFactors,
															 patternCount, categoryCount,
															 doRescaling);
				die = 1;
			} else {
				nativeGPUPartialsPartialsPruningDynamicScaling(partials1, partials2, partials3,
															   matrices1, matrices2, scalingFactors,
															   patternCount, categoryCount,
															   doRescaling);
			}
		}
		
		//		nativeGPUPartialsPartialsPruningDynamicScaling(
		//				partials1,partials2, partials3,
		//				matrices1,matrices2,
		//				scalingFactors,
		//				patternCount, categoryCount, doRescaling);
#else
		if (tipStates1 != 0) {
			if (tipStates2 != 0 ) {
				nativeGPUStatesStatesPruning(tipStates1, tipStates2, partials3, matrices1,
											 matrices2, patternCount, categoryCount);
			} else {
				nativeGPUStatesPartialsPruning(tipStates1, partials2, partials3, matrices1,
											   matrices2, patternCount, categoryCount);
				die = 1;
			}
		} else {
			if (tipStates2 != 0) {
				nativeGPUStatesPartialsPruning(tipStates2, partials1, partials3, matrices2,
											   matrices1, patternCount, categoryCount);
				die = 1;
			} else {
				nativeGPUPartialsPartialsPruning(partials1, partials2, partials3, matrices1,
												 matrices2, patternCount, categoryCount);
			}
		}
		
		//		nativeGPUPartialsPartialsPruning(partials1, partials2, partials3,
		//				matrices1, matrices2, patternCount,
		//				categoryCount);
#endif // DYNAMIC_SCALING
		
#ifdef DEBUG_BEAGLE
		fprintf(stderr, "patternCount = %d\n", patternCount);
		fprintf(stderr, "truePatternCount = %d\n", truePatternCount);
		fprintf(stderr, "categoryCount  = %d\n", categoryCount);
		fprintf(stderr, "partialSize = %d\n", partialsSize);
		if (tipStates1)
			printfCudaInt(tipStates1, patternCount);
		else
			printfCudaVector(partials1, partialsSize);
		if (tipStates2)
			printfCudaInt(tipStates2, patternCount);
		else
			printfCudaVector(partials2, partialsSize);
		fprintf(stderr, "node index = %d\n", nodeIndex3);
		printfCudaVector(partials3, partialsSize);
		
		if(nodeIndex3 == 106)
			exit(-1);
#endif
	}
	
#ifdef DEBUG_FLOW
	fprintf(stderr, "Exiting updatePartials\n");
#endif
	
	return NO_ERROR;
}

int BeagleCUDAImpl::waitForPartials(const int* destinationPartials,
									int destinationPartialsCount) {
	return NO_ERROR;
}

int BeagleCUDAImpl::calculateRootLogLikelihoods(const int* bufferIndices,
												const double* weights,
												const double* stateFrequencies,
												int count,
												double* outLogLikelihoods) {
#ifdef DEBUG_FLOW
	fprintf(stderr,"Entering calculateLogLikelihoods\n");
#endif

	if (count == 1) {	
		const int rootNodeIndex = bufferIndices[0];
		
	#ifdef DYNAMIC_SCALING
		if (doRescaling) {
			// Construct node-list for scalingFactors
			int n;
			int length = nodeCount - taxaCount;
			for(n=0; n<length; n++)
				hPtrQueue[n] = 
					dScalingFactors[hCurrentScalingFactorsIndices[n+taxaCount]][n+taxaCount];
			
			cudaMemcpy(dPtrQueue,hPtrQueue,sizeof(REAL*)*length, cudaMemcpyHostToDevice);
			
			// Computer scaling factors at the root
			nativeGPUComputeRootDynamicScaling(dPtrQueue,
											   dRootScalingFactors,
											   length,
											   patternCount);
		}
		
		doRescaling = 0;
		
		nativeGPUIntegrateLikelihoodsDynamicScaling(dIntegrationTmp,
								dPartials[hCurrentPartialsIndices[rootNodeIndex]][rootNodeIndex],
								dCategoryProportions, dFrequencies,
								dRootScalingFactors,
								patternCount, categoryCount, nodeCount);
		
	#else
		nativeGPUIntegrateLikelihoods(dIntegrationTmp,
								  dPartials[hCurrentPartialsIndices[rootNodeIndex]][rootNodeIndex],
								  dCategoryProportions,
								  dFrequencies, patternCount,
								  categoryCount);
	#endif // DYNAMIC_SCALING
		
	#ifdef DOUBLE_PRECISION
		cudaMemcpy(outLogLikelihoods,dIntegrationTmp,
				   SIZE_REAL*truePatternCount, cudaMemcpyDeviceToHost);
	#else
		cudaMemcpy(hLogLikelihoodsCache,dIntegrationTmp,
				   SIZE_REAL*truePatternCount, cudaMemcpyDeviceToHost);
		MEMCPY(outLogLikelihoods,hLogLikelihoodsCache,truePatternCount,double);
	#endif
		
	#ifdef DEBUG
		printf("logLike = ");
		printfVectorD(outLogLikelihoods,truePatternCount);
		exit(-1);
	#endif
	} else {
		// TODO: implement calculate root lnL for multiple count 
	}

	
#ifdef DEBUG_FLOW
	fprintf(stderr,"Exiting calculateLogLikelihoods\n");
#endif
	
	return NO_ERROR;
}

int BeagleCUDAImpl::calculateEdgeLogLikelihoods(const int* parentBufferIndices,
												const int* childBufferIndices,
												const int* probabilityIndices,
												const int* firstDerivativeIndices,
												const int* secondDerivativeIndices,
												const double* weights,
												const double* stateFrequencies,
												int count,
												double* outLogLikelihoods,
												double* outFirstDerivatives,
												double* outSecondDerivatives) {
	// TODO: implement calculateEdgeLogLikelihoods
	assert(false);
}


///////////////////////////////////////////////////////////////////////////////
// private methods

void checkNativeMemory(void *ptr) {
	if (ptr == NULL) {
		fprintf(stderr, "Unable to allocate some memory!\n");
		exit(-1);
	}
}

/*
 * Computes the device memory requirements
 * TODO
 */
long BeagleCUDAImpl::memoryRequirement(int taxaCount, int stateCount) {
	// Evec, storedEvec
	// Ivec, storedIevc
	// EigenValues, storeEigenValues
	// Frequencies, storedFrequencies
	// categoryProportions, storedCategoryProportions
	// integrationTmp (patternCount)
	
	return 0;
}

void BeagleCUDAImpl::initializeInstanceMemory() {
	
	cudaSetDevice(device);
	int i;
	
	//	dCMatrix = allocateGPURealMemory(MATRIX_CACHE_SIZE);
	//	dStoredMatrix = allocateGPURealMemory(MATRIX_CACHE_SIZE);
	dEvec = allocateGPURealMemory(MATRIX_SIZE);
	dIevc = allocateGPURealMemory(MATRIX_SIZE);
	dStoredEvec = allocateGPURealMemory(MATRIX_SIZE);
	dStoredIevc = allocateGPURealMemory(MATRIX_SIZE);
	
	dEigenValues = allocateGPURealMemory(EVAL_SIZE);
	dStoredEigenValues = allocateGPURealMemory(EVAL_SIZE);
	
	dFrequencies = allocateGPURealMemory(PADDED_STATE_COUNT);
	dStoredFrequencies = allocateGPURealMemory(PADDED_STATE_COUNT);
	
	//	dCategoryRates = allocateGPURealMemory(
	//			categoryCount);
	hCategoryRates = (REAL *) malloc(sizeof(REAL)
									 * categoryCount);
	//	dStoredCategoryRates = allocateGPURealMemory(
	//			categoryCount);
	hStoredCategoryRates = (REAL *) malloc(sizeof(REAL)
										   * categoryCount);
	
	checkNativeMemory(hCategoryRates);
	checkNativeMemory(hStoredCategoryRates);
	
	dCategoryProportions = allocateGPURealMemory(
												 categoryCount);
	dStoredCategoryProportions = allocateGPURealMemory(
													   categoryCount);
	
	dIntegrationTmp = allocateGPURealMemory(
											patternCount);
	
	dPartials = (REAL ***) malloc(sizeof(REAL**) * 2);
	dPartials[0] = (REAL **) calloc(sizeof(REAL*),nodeCount);	// Fill with 0s so 'free'
	//	does not choke if unallocated
	dPartials[1] = (REAL **) calloc(sizeof(REAL*),nodeCount);
	
	dStates = (int **) calloc(sizeof(int*),nodeCount);	// Internal nodes have 0s so
	// partials are used
	
#ifdef DYNAMIC_SCALING
	dScalingFactors = (REAL ***) malloc(sizeof(REAL**) * 2);
	dScalingFactors[0] = (REAL **) malloc(sizeof(REAL*) * nodeCount);
	dScalingFactors[1] = (REAL **) malloc(sizeof(REAL*) * nodeCount);
	dRootScalingFactors = allocateGPURealMemory(patternCount);
	dStoredRootScalingFactors = allocateGPURealMemory(patternCount);
#endif
	
	for (i = 0; i < nodeCount; i++) {
		
		if (i < taxaCount) { // For the tips
			if (hTmpStates[i] == 0) // If no tipStates
				dPartials[0][i] = allocateGPURealMemory(partialsSize);
			else
				dStates[i] = allocateGPUIntMemory(patternCount);
		} else {
			dPartials[0][i] = allocateGPURealMemory(partialsSize);
			dPartials[1][i] = allocateGPURealMemory(partialsSize);
#ifdef DYNAMIC_SCALING
			dScalingFactors[0][i] = allocateGPURealMemory(patternCount);
			dScalingFactors[1][i] = allocateGPURealMemory(patternCount);
#endif
		}
	}
	
	hCurrentMatricesIndices = (int *) malloc(sizeof(int)
											 * nodeCount);
	hStoredMatricesIndices = (int *) malloc(sizeof(int)
											* nodeCount);
	for (i = 0; i < nodeCount; i++) {
		hCurrentMatricesIndices[i] = 0;
		hStoredMatricesIndices[i] = 0;
	}
	
	checkNativeMemory(hCurrentMatricesIndices);
	checkNativeMemory(hStoredMatricesIndices);
	
	hCurrentPartialsIndices = (int *) malloc(sizeof(int)
											 * nodeCount);
	hStoredPartialsIndices = (int *) malloc(sizeof(int)
											* nodeCount);
	
#ifdef DYNAMIC_SCALING
	hCurrentScalingFactorsIndices = (int *)malloc(sizeof(int) * nodeCount);
	hStoredScalingFactorsIndices = (int *)malloc(sizeof(int) * nodeCount);
#endif
	
	for (i = 0; i < nodeCount; i++) {
		hCurrentPartialsIndices[i] = 0;
		hStoredPartialsIndices[i] = 0;
#ifdef DYNAMIC_SCALING
		hCurrentScalingFactorsIndices[i] = 0;
		hStoredScalingFactorsIndices[i] = 0;
#endif
	}
	
	checkNativeMemory(hCurrentPartialsIndices);
	checkNativeMemory(hStoredPartialsIndices);
	
#ifdef DYNAMIC_SCALING
	checkNativeMemory(hCurrentScalingFactorsIndices);
	checkNativeMemory(hStoredScalingFactorsIndices);
#endif
	
	//	dCurrentMatricesIndices = allocateGPUIntMemory(nodeCount);
	//	dStoredMatricesIndices = allocateGPUIntMemory(nodeCount);
	//	cudaMemcpy(	dCurrentMatricesIndices,
	//				hCurrentMatricesIndices,
	//				sizeof(int)*nodeCount,cudaMemcpyHostToDevice);
	
	//	dCurrentPartialsIndices = allocateGPUIntMemory(nodeCount);
	//	dStoredPartialsIndices = allocateGPUIntMemory(nodeCount);
	//	cudaMemcpy(	dCurrentPartialsIndices,
	//				hCurrentPartialsIndices,
	//				sizeof(int)*nodeCount,cudaMemcpyHostToDevice);
	
	dMatrices = (REAL ***) malloc(sizeof(REAL**) * 2);
	dMatrices[0] = (REAL **) malloc(sizeof(REAL*)
									* nodeCount);
	dMatrices[1] = (REAL **) malloc(sizeof(REAL*)
									* nodeCount);
	
	for (i = 0; i < nodeCount; i++) {
		dMatrices[0][i] = allocateGPURealMemory(MATRIX_SIZE
												* categoryCount);
		dMatrices[1][i] = allocateGPURealMemory(MATRIX_SIZE
												* categoryCount);
	}
	
	dNodeIndices = allocateGPUIntMemory(
										nodeCount); // No execution has more no nodeCount events
	hNodeIndices = (int *) malloc(sizeof(int)
								  * nodeCount);
	hDependencies = (int *) malloc(sizeof(int)
								   * nodeCount);
	dBranchLengths = allocateGPURealMemory(
										   nodeCount);
	
	checkNativeMemory(hNodeIndices);
	checkNativeMemory(hDependencies);
	
	dDistanceQueue = allocateGPURealMemory(
										   nodeCount * categoryCount);
	hDistanceQueue = (REAL *) malloc(sizeof(REAL)
									 * nodeCount * categoryCount);
	
	checkNativeMemory(hDistanceQueue);
	
	int len = 5;
	if (categoryCount > 5);
	len = categoryCount;
	
	SAFE_CUDA(cudaMalloc((void**) &dPtrQueue, sizeof(REAL*)*nodeCount*len),dPtrQueue);
	hPtrQueue = (REAL **) malloc(sizeof(REAL*)
								 * nodeCount * len);
	
	checkNativeMemory(hPtrQueue);
	
}

void BeagleCUDAImpl::initializeDevice(int deviceNumber,
									  int inTipCount,
									  int inPartialsBufferCount,
									  int inCompactBufferCount,
									  int inStateCount,
									  int inPatternCount,
									  int inEigenDecompositionCount,
									  int inMatrixCount) {
	
#ifdef DEBUG_FLOW
	fprintf(stderr,"Entering initialize\n");
#endif
	kStateCount = inStateCount;
	
	device = deviceNumber;
	nodeCount = inPartialsBufferCount + inCompactBufferCount;
	taxaCount = inTipCount;
	truePatternCount = inPatternCount;
	matrixCount = inMatrixCount;
	
	paddedStates = 0;
	paddedPatterns = 0;
	
#if (PADDED_STATE_COUNT == 4)  // DNA model
	// Make sure that patternCount + paddedPatterns is multiple of 4
	if (truePatternCount % 4 != 0)
		paddedPatterns = 4 - truePatternCount % 4;
	else
		paddedPatterns = 0;
#ifdef DEBUG
	fprintf(stderr,"Padding patterns for 4-state model:\n");
	fprintf(stderr,"\ttruePatternCount = %d\n\tpaddedPatterns = %d\n",
			truePatternCount,paddedPatterns);
#endif // DEBUG
#endif // DNA model
	patternCount = truePatternCount + paddedPatterns;
	
	partialsSize = patternCount * PADDED_STATE_COUNT;
	
	hFrequenciesCache = (REAL*)calloc(PADDED_STATE_COUNT, SIZE_REAL);
	
	// TODO Only allocate if necessary on the fly
	hPartialsCache = (REAL*)calloc(partialsSize,SIZE_REAL);
	hStatesCache = (int *)calloc(patternCount,SIZE_INT);
	
	hMatrixCache = (REAL*)calloc(2*MATRIX_SIZE + EVAL_SIZE, SIZE_REAL);
	
	//	hNodeCache = NULL;
	
#ifndef DOUBLE_PRECISION
	hLogLikelihoodsCache = (REAL*)malloc(truePatternCount*SIZE_REAL);
#endif
	
	doRescaling = 1;
	sinceRescaling = 0;
	
#ifndef PRE_LOAD
	initializeInstanceMemory();
#else
	// initialize temporary storage before likelihood thread exists
	loaded = 0;
	hTmpPartials = (REAL**) malloc(sizeof(REAL*) * taxaCount);
	
	// TODO: Only need to allocate tipPartials or tipStates, not both
	// Should just fill with 0 (= no partials to load)
	for (int i = 0; i < taxaCount; i++) {
		hTmpPartials[i] = (REAL*) malloc(SIZE_REAL * partialsSize);
	}
	
	// Fill with 0 (= no states to load)
	hTmpStates = (int**) calloc(sizeof(int*), taxaCount);
	
#endif
	
#ifdef DEBUG_FLOW
	fprintf(stderr,"Exiting initialize\n");
#endif
}

void BeagleCUDAImpl::freeTmpPartialsOrStates() {
	int i;
	for (i = 0; i < taxaCount; i++) {
		free(hTmpPartials[i]);
		free(hTmpStates[i]);
	}
	
	free(hTmpPartials);
	free(hTmpStates);
	free(hPartialsCache);
	free(hStatesCache);
}

void BeagleCUDAImpl::freeNativeMemory() {
	int i;
	for (i = 0; i < nodeCount; i++) {
		freeGPUMemory(dPartials[0][i]);
		freeGPUMemory(dPartials[1][i]);
#ifdef DYNAMIC_SCALING
		freeGPUMemory(dScalingFactors[0][i]);
		freeGPUMemory(dScalingFactors[1][i]);
#endif
		freeGPUMemory(dMatrices[0][i]);
		freeGPUMemory(dMatrices[1][i]);
		freeGPUMemory(dStates[i]);
	}
	
	//	freeGPUMemory(dCMatrix);
	//	freeGPUMemory(dStoredMatrix);
	freeGPUMemory(dEvec);
	freeGPUMemory(dIevc);
	
	free(dPartials[0]);
	free(dPartials[1]);
	free(dPartials);
	
#ifdef DYNAMIC_SCALING
	free(dScalingFactors[0]);
	free(dScalingFactors[1]);
	free(dScalingFactors);
#endif
	
	free(dMatrices[0]);
	free(dMatrices[1]);
	free(dMatrices);
	
	free(dStates);
	
	free(hCurrentMatricesIndices);
	free(hStoredMatricesIndices);
	free(hCurrentPartialsIndices);
	free(hStoredPartialsIndices);
	
#ifdef DYNAMIC_SCALING
	free(hCurrentScalingFactorsIndices);
	free(hStoredScalingFactorsIndices);
#endif
	
	freeGPUMemory(dNodeIndices);
	free(hNodeIndices);
	free(hDependencies);
	freeGPUMemory(dBranchLengths);
	
	freeGPUMemory(dIntegrationTmp);
	
	free(hDistanceQueue);
	free(hPtrQueue);
	freeGPUMemory(dDistanceQueue);
	freeGPUMemory(dPtrQueue);
	
	// TODO: Free all caches
	free(hPartialsCache);
	free(hStatesCache);
}

REAL *callocBEAGLE(int length, int instance) {
	REAL *ptr = (REAL *) calloc(length, SIZE_REAL);
	if (ptr == NULL) {
		fprintf(stderr,"Unable to allocate native memory!");
		exit(-1);
	}
	return ptr;
}

void BeagleCUDAImpl::loadTipPartialsOrStates() {
	
	for (int i = 0; i < taxaCount; i++) {
		if (hTmpStates[i] == 0)
			cudaMemcpy(dPartials[0][i],hTmpPartials[i], SIZE_REAL
					   * partialsSize, cudaMemcpyHostToDevice);
		else
			cudaMemcpy(dStates[i],hTmpStates[i], SIZE_INT
					   * patternCount, cudaMemcpyHostToDevice);
	}
}

/*
 * Transposes a square matrix in place
 */
void BeagleCUDAImpl::transposeSquareMatrix(REAL* mat, int size) {
	for (int i = 0; i < size - 1; i++) {
		for (int j = i + 1; j < size; j++) {
			REAL tmp = mat[i * size + j];
			mat[i * size + j] = mat[j * size + i];
			mat[j * size + i] = tmp;
		}
	}
}

int BeagleCUDAImpl::getGPUDeviceCount() {
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
	return cDevices;
}

void BeagleCUDAImpl::printGPUInfo(int device) {
	
	fprintf(stderr,"GPU Device Information:");
	
	char name[256];
	int totalGlobalMemory = 0;
	int clockSpeed = 0;
	
	// New CUDA functions in cutil.h do not work in JNI files
	getGPUInfo(device, name, &totalGlobalMemory, &clockSpeed);
	fprintf(stderr,"\nDevice #%d: %s\n",(device+1),name);
	double mem = totalGlobalMemory / 1024.0 / 1024.0;
	double clo = clockSpeed / 1000000.0;
	fprintf(stderr,"\tGlobal Memory (MB) : %1.2f\n",mem);
	fprintf(stderr,"\tClock Speed (Ghz)  : %1.2f\n",clo);
}

void BeagleCUDAImpl::getGPUInfo(int iDevice, char *name, int *memory, int *speed) {
	cudaDeviceProp deviceProp;
	memset(&deviceProp, 0, sizeof(deviceProp));
	cudaGetDeviceProperties(&deviceProp, iDevice);
	*memory = deviceProp.totalGlobalMem;
	*speed = deviceProp.clockRate;
	strcpy(name, deviceProp.name);
}


///////////////////////////////////////////////////////////////////////////////
// BeagleCUDAImplFactory public methods

BeagleImpl*  BeagleCUDAImplFactory::createImpl(int tipCount,
											   int partialsBufferCount,
											   int compactBufferCount,
											   int stateCount,
											   int patternCount,
											   int eigenBufferCount,
											   int matrixBufferCount) {
	BeagleImpl* impl = new BeagleCUDAImpl();
	try {
		if (impl->initialize(tipCount,
							 partialsBufferCount,
							 compactBufferCount,
							 stateCount,
							 patternCount,
							 eigenBufferCount,
							 matrixBufferCount) == 0)
			return impl;
	}
	catch(...)
	{
		delete impl;
		throw;
	}
	delete impl;
	return NULL;
}

const char* BeagleCUDAImplFactory::getName() {
	return "CUDA";
}
