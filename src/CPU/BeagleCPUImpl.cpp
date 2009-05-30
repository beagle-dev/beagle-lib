/*
 *  BeagleCPUImpl.cpp
 *  BEAGLE
 *
 * @author Andrew Rambaut
 * @author Marc Suchard
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "BeagleCPUImpl.h"



int BeagleCPUImpl::initialize(int bufferCount, int maxTipStateCount,
		int stateCount, int patternCount, int eigenDecompositionCount,
		int matrixCount) {
	kBufferCount = bufferCount;
	kTipCount = maxTipStateCount;
	assert(bufferCount > kTipCount);
	kStateCount = stateCount;
	kPatternCount = patternCount;
	kMatrixCount = matrixCount;

	kMatrixSize = (1 + stateCount) * stateCount;

	cMatrices = (double **) malloc(sizeof(double *) * eigenDecompositionCount);
	if (cMatrices == 0L)
		throw std::bad_alloc();
	storedCMatrices = (double **) malloc(sizeof(double *)
			* eigenDecompositionCount);
	if (storedCMatrices == 0L)
		throw std::bad_alloc();
	eigenValues
			= (double **) malloc(sizeof(double *) * eigenDecompositionCount);
	if (eigenValues == 0L)
		throw std::bad_alloc();
	storedEigenValues = (double **) malloc(sizeof(double *)
			* eigenDecompositionCount);
	if (storedEigenValues == 0L)
		throw std::bad_alloc();

	for (int i = 0; i < eigenDecompositionCount; i++) {
		cMatrices[i] = (double *) malloc(sizeof(double) * stateCount
				* stateCount * stateCount);
		if (cMatrices[i] == 0L)
			throw std::bad_alloc();
		storedCMatrices[i] = (double *) malloc(sizeof(double) * stateCount
				* stateCount * stateCount);
		if (storedCMatrices[i] == 0L)
			throw std::bad_alloc();
		eigenValues[i] = (double *) malloc(sizeof(double) * stateCount);
		if (eigenValues[i] == 0L)
			throw std::bad_alloc();
		storedEigenValues[i] = (double *) malloc(sizeof(double) * stateCount);
		if (storedEigenValues[i] == 0L)
			throw std::bad_alloc();
	}

	branchLengths.resize(kMatrixCount);
	storedBranchLengths.resize(kMatrixCount);

	// a temporary array used in calculating log likelihoods
	integrationTmp.resize(patternCount * stateCount);

	kPartialsSize = kPatternCount * stateCount;

	partials.assign(kBufferCount, 0L);
	tipStates.assign(kTipCount, 0L);

	for (int i = 0; i < kTipCount; i++) {
		tipStates[i] = (int *) malloc(sizeof(int) * kPatternCount);
	}
	for (int i = kTipCount; i < kBufferCount; i++) {
		partials[i] = (double *) malloc(sizeof(double) * kPartialsSize);
		if (partials[i] == 0L)
			throw std::bad_alloc();
	}

	std::vector<double> emptyMat(kMatrixSize);
	transitionMatrices.assign(kMatrixCount, emptyMat);

	fprintf(stderr, "done through here!\n");
	return SUCCESS;
}

BeagleCPUImpl::~BeagleCPUImpl() {
	// free all that stuff...
}

int BeagleCPUImpl::setPartials(int bufferIndex, const double* inPartials) {
	assert(partials[bufferIndex] == 0L);
	partials[bufferIndex] = (double *) malloc(sizeof(double) * kPartialsSize);
	if (partials[bufferIndex] == 0L)
		return OUT_OF_MEMORY_ERROR;
	memcpy(partials[bufferIndex], inPartials, sizeof(double) * kPartialsSize);
	return NO_ERROR;
}

int BeagleCPUImpl::getPartials(int bufferIndex, double *outPartials) {
	memcpy(outPartials, partials[bufferIndex], sizeof(double) * kPartialsSize);
	return NO_ERROR;
}

int BeagleCPUImpl::setTipStates(int tipIndex, const int* inStates) {
	tipStates[tipIndex] = (int *) malloc(sizeof(int) * kPatternCount);
	for (int i = 0; i < kPatternCount; i++) {
		tipStates[tipIndex][i] = (inStates[i] < kStateCount ? inStates[i]
				: kStateCount);
	}
	return NO_ERROR;
}

int BeagleCPUImpl::setEigenDecomposition(int eigenIndex,
		const double** inEigenVectors, const double** inInverseEigenVectors,
		const double* inEigenValues) {
	int l = 0;
	for (int i = 0; i < kStateCount; i++) {
		eigenValues[eigenIndex][i] = inEigenValues[i];

		for (int j = 0; j < kStateCount; j++) {
			for (int k = 0; k < kStateCount; k++) {
				cMatrices[eigenIndex][l] = inEigenVectors[i][k]
						* inInverseEigenVectors[k][j];
				l++;
			}
		}
	}
	return NO_ERROR;
}

int BeagleCPUImpl::setTransitionMatrix(int matrixIndex, const double* inMatrix) {
	assert(false);
}

int BeagleCPUImpl::updateTransitionMatrices(int eigenIndex,
		const int* probabilityIndices, const int* firstDerivativeIndices,
		const int* secondDervativeIndices, const double* edgeLengths, int count) {
	std::vector<double> tmp;
	tmp.resize(kStateCount);


	for (int u = 0; u < count; u++) {
		int nodeIndex = probabilityIndices[u];

		int n = 0;
		int matrixIndex = 0;
		for (int i = 0; i < kStateCount; i++) {
			tmp[i] = exp(eigenValues[eigenIndex][i] * edgeLengths[u]);
		}

		int m = 0;
		for (int i = 0; i < kStateCount; i++) {
			for (int j = 0; j < kStateCount; j++) {
				double sum = 0.0;
				for (int k = 0; k < kStateCount; k++) {
					sum += cMatrices[eigenIndex][m] * tmp[k];
					m++;
				}
				if (sum > 0)
					transitionMatrices[nodeIndex][nodeIndex][n] = sum;
				else
					transitionMatrices[nodeIndex][nodeIndex][n] = 0;

				n++;
			}
			transitionMatrices[nodeIndex][nodeIndex][n] = 1.0;
			n++;
		}
		if (kMatrixCount > 1) {
			eigenIndex++;
		}

	}
	return NO_ERROR;
}

int BeagleCPUImpl::updatePartials(int* operations, int count, int rescale) {

	int x = 0;
	for (int op = 0; op < count; op++) {
		const int parIndex = operations[x++];
		const int child1Index = operations[x++];
		const int child1TransMatIndex = operations[x++];
		const int child1Index = operations[x++];
		const int child1TransMatIndex = operations[x++];

		assert(parIndex < partials.size());
		assert(parIndex >= tipStates.size());
		assert(child1Index < partials.size());
		assert(child2Index < partials.size());
		assert(child1TransMatIndex < transitionMatrices.size());
		assert(child2TransMatIndex < transitionMatrices.size());

		const double * child1TransMat = transitionMatrices[child1TransMatIndex];
		assert(child1TransMat);
		const double * child2TransMat = transitionMatrices[child2TransMatIndex];
		assert(child2TransMat);
		double * destPartial = partials[parIndex];
		assert(destPartial);

		if (child1Index < kTipCount) {
			if (child2Index < kTipCount) {
				updateStatesStates(destPartial, tipStates[child1Index], child1TransMat, tipStates[child2Index], child2TransMat);
			} else {
				updateStatesPartials(destPartial, tipStates[child1Index], child1TransMat, partials[child2Index], child2TransMat);
			}
		} else {
			if (nodeIndex2 < kTipCount) {
				updateStatesPartials(destPartial, tipStates[child2Index], child2TransMat, partials[child1Index], child1TransMat);
			} else {
				updatePartialsPartials(destPartial, partials[child1Index], child1TransMat, partials[child2Index], child2TransMat);
			}
		}
	}
	return NO_ERROR;
}



/*
 * Calculates partial likelihoods at a node when both children have states.
 */
int BeagleCPUImpl::updateStatesStates(	double * destP,
										const int * child1States,
										const double *child1TransMat,
										const int * child2States,
										const double *child2TransMat)
{


	int v = 0;
	for (int k = 0; k < kPatternCount; k++) {

		const int state0 = child0States[k];
		const int state2 = child2States[k];
		int w = 0;
		for (int i = 0; i < kStateCount; i++) {
			destP[v] = child0TransMat[w + state0] * child2TransMat[w + state2];
			v++;
			w += (kStateCount + 1);
		}

	}
	return NO_ERROR;
}

/*
 * Calculates partial likelihoods at a node when one child has states and one has partials.
 */
int BeagleCPUImpl::updateStatesPartials(double * destP, const int * states1, const double *matrices1, const double * partials2, const double *matrices2)
{
	int u = 0;
	int v = 0;

	for (int l = 0; l < kCategoryCount; l++) {
		for (int k = 0; k < kPatternCount; k++) {

			int state1 = states1[k];

			int w = l * kMatrixSize;

			for (int i = 0; i < kStateCount; i++) {

				double tmp = matrices1[w + state1];

				double sum = 0.0;
				for (int j = 0; j < kStateCount; j++) {
					sum += matrices2[w] * partials2[v + j];
					w++;
				}

				// increment for the extra column at the end
				w++;

				destP[u] = tmp * sum;
				u++;
			}

			v += kStateCount;
		}
	}
	return NO_ERROR;
}

int BeagleCPUImpl::updatePartialsPartials(double * destP, const double * partials1, const double *matrices1, const double * partials2, const double *matrices2)
{
	double sum1, sum2;



	int u = 0;
	int v = 0;

	for (int k = 0; k < kPatternCount; k++) {

		int w = 0;

		for (int i = 0; i < kStateCount; i++) {

			sum1 = sum2 = 0.0;

			for (int j = 0; j < kStateCount; j++) {
				sum1 += matrices1[w] * partials1[v + j];
				sum2 += matrices2[w] * partials2[v + j];
				w++;
			}

			// increment for the extra column at the end
			w++;

			destP[u] = sum1 * sum2;

			u++;
		}
		v += kStateCount;
	}
	return NO_ERROR;
}

int BeagleCPUImpl::calculateRootLogLikelihoods(
	const int* bufferIndices,
	int count,
	const double* weights,
	const double** stateFrequencies,
	double* outLogLikelihoods)
{
	// this shouldn't be set to 0 here (we should do it in the loop)
	for (int k = 0; k < kPatternCount; k++)
		outLogLikelihoods[k] = 0.0;

	for (int subsetIndex = 0 ; subsetIndex < count; ++subsetIndex ) {
		assert(subsetIndex < partials.size());
		const int rootPartialIndex = bufferIndices[subsetIndex];
		const double * rootPartials = partials[rootPartialIndex];
		assert(rootPartials);
		const double * frequencies = stateFrequencies[weights];
		const double wt = weights[subsetIndex];
		int u = 0;
		int v = 0;
		for (int k = 0; k < kPatternCount; k++) {

			double sum = 0.0;
			for (int i = 0; i < stateCount; i++) {

				sum += frequencies[i] * rootPartials[v];
				u++;
				v++;
			}
			outLogLikelihoods[k] += sum*wt;
		}
	}

	// this shouldn't be set to 0 here (we should do it in the loop)
	for (int k = 0; k < kPatternCount; k++)
		outLogLikelihoods[k] = log(outLogLikelihoods[k]);
	return NO_ERROR;
}

int BeagleCPUImpl::calculateEdgeLogLikelihoods(const int * parentBufferIndices,
								 const int* childBufferIndices,
								 const int* probabilityIndices,
								 const int* firstDerivativeIndices,
								 const int* secondDerivativeIndices,
								 int count,
								 const double* weights,
								 const double** stateFrequencies,
								 double* outLogLikelihoods,
								 double* outFirstDerivatives,
								 double* outSecondDerivatives)
{
	assert(0);
}

BeagleImpl* BeagleCPUImplFactory::createImpl(int bufferCount, int tipCount,
		int kStateCount, int patternCount, int eigenDecompositionCount,
		int matrixCount) {
	BeagleImpl* impl = new BeagleCPUImpl();
	try {
		if(impl->initialize(nodeCount,tipCount,kStateCount,patternCount,categoryCount,matrixCount))
		return impl;
	}
	except(...)
	{
		delete impl;
		throw;
	}
	delete impl;
	return NULL;
}

const char* BeagleCPUImplFactory::getName() {
	return "CPU";
}

