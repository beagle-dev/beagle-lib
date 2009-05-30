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
#include <iostream>
#include <string.h>
#include <math.h>

#include "beagle.h"
#include "BeagleCPUImpl.h"



int BeagleCPUImpl::initialize(int tipCount, int partialBufferCount, int compactBufferCount,
		int stateCount, int patternCount, int eigenDecompositionCount,
		int matrixCount) {
	std::cerr << "in BeagleCPUImpl::initialize\n" ;
	kBufferCount = partialBufferCount + compactBufferCount;
	kTipCount = tipCount;
	assert(kBufferCount > kTipCount);
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


	////BEGIN edge Like Hack
	TEMP_SCRATCH_PARTIAL = (double *) malloc(sizeof(double) * kPartialsSize);  /// used in our hack edgeLike func
	partials.push_back(TEMP_SCRATCH_PARTIAL);
	TEMP_IDENTITY_MATRIX.assign(kStateCount*kStateCount, 0.0);
	unsigned el = 0;
	for (unsigned diag = 0 ; diag < kStateCount; ++diag)
		{
		TEMP_IDENTITY_MATRIX[el] = 1.0;
		el += kStateCount;
		}
	////END edge Like Hack

	return NO_ERROR;
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
		 std::vector<double> & transitionMat = transitionMatrices[probabilityIndices[u]];
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
					transitionMat[n] = sum;
				else
					transitionMat[n] = 0;

				n++;
			}
			transitionMat[n] = 1.0;
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
		const int child2Index = operations[x++];
		const int child2TransMatIndex = operations[x++];

		assert(parIndex < partials.size());
		assert(parIndex >= tipStates.size());
		assert(child1Index < partials.size());
		assert(child2Index < partials.size());
		assert(child1TransMatIndex < transitionMatrices.size());
		assert(child2TransMatIndex < transitionMatrices.size());

		const double * child1TransMat = &(transitionMatrices[child1TransMatIndex][0]);
		assert(child1TransMat);
		const double * child2TransMat = &(transitionMatrices[child2TransMatIndex][0]);
		assert(child2TransMat);
		double * destPartial = partials[parIndex];
		assert(destPartial);

		if (child1Index < kTipCount && tipStates[child1Index]) {
			if (child2Index < kTipCount && tipStates[child2Index]) {
				updateStatesStates(destPartial, tipStates[child1Index], child1TransMat, tipStates[child2Index], child2TransMat);
			} else {
				updateStatesPartials(destPartial, tipStates[child1Index], child1TransMat, partials[child2Index], child2TransMat);
			}
		} else {
			if (child2Index < kTipCount && tipStates[child2Index] ) {
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
void BeagleCPUImpl::updateStatesStates(	double * destP,
										const int * child1States,
										const double *child1TransMat,
										const int * child2States,
										const double *child2TransMat)
{


	int v = 0;
	for (int k = 0; k < kPatternCount; k++) {

		const int state1 = child1States[k];
		const int state2 = child2States[k];
		int w = 0;
		for (int i = 0; i < kStateCount; i++) {
			destP[v] = child1TransMat[w + state1] * child2TransMat[w + state2];
			v++;
			w += (kStateCount + 1);
		}
	}
}

/*
 * Calculates partial likelihoods at a node when one child has states and one has partials.
 */
void BeagleCPUImpl::updateStatesPartials(double * destP, const int * states1, const double *matrices1, const double * partials2, const double *matrices2)
{
	int u = 0;
	int v = 0;

	for (int k = 0; k < kPatternCount; k++) {

		int state1 = states1[k];

		int w = 0;

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

void BeagleCPUImpl::updatePartialsPartials(double * destP, const double * partials1, const double *matrices1, const double * partials2, const double *matrices2)
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
}

int BeagleCPUImpl::calculateRootLogLikelihoods(
	const int* bufferIndices,
	const double* weights,
	const double** stateFrequencies,
	int count,
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
		const double * frequencies = stateFrequencies[subsetIndex];
		const double wt = weights[subsetIndex];
		int u = 0;
		int v = 0;
		for (int k = 0; k < kPatternCount; k++) {

			double sum = 0.0;
			for (int i = 0; i < kStateCount; i++) {

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
								 const double* weights,
								 const double** stateFrequencies,
								 int count,
								 double* outLogLikelihoods,
								 double* outFirstDerivatives,
								 double* outSecondDerivatives)
{

	//@@@@ this impl is a real hack !!!!!!!!
	assert(firstDerivativeIndices == 0L);
	assert(secondDerivativeIndices == 0L);
	assert(outFirstDerivatives == 0L);
	assert(outSecondDerivatives == 0L);

	assert(count == 1);
	int parIndex = parentBufferIndices[0];
	int childIndex = childBufferIndices[0];
	if (parIndex < childIndex)
		{
		childIndex = parIndex;
		parIndex = childBufferIndices[0];
		}

	assert(parIndex >= kTipCount);

	memcpy(TEMP_SCRATCH_PARTIAL, partials[parIndex], sizeof(double) * kPartialsSize);
	const double * fakeEdgeMat = &TEMP_IDENTITY_MATRIX[0];
	const std::vector<double> & realMat = transitionMatrices[probabilityIndices[0]];
	const double * edgeTransMat = &(realMat[0]);

	if (childIndex < kTipCount && tipStates[childIndex] ) {
		updateStatesPartials(TEMP_SCRATCH_PARTIAL, tipStates[childIndex], edgeTransMat, partials[parIndex], fakeEdgeMat);
	} else {
		updatePartialsPartials(TEMP_SCRATCH_PARTIAL, partials[childIndex], edgeTransMat, partials[parIndex], fakeEdgeMat);
	}


	int c = partials.size() - 1;
	return calculateRootLogLikelihoods(&c,  weights, stateFrequencies, 1, outLogLikelihoods);
}

BeagleImpl* BeagleCPUImplFactory::createImpl(
                int tipCount,				/**< Number of tip data elements (input) */
				int partialsBufferCount,	/**< Number of partials buffers to create (input) */
				int compactBufferCount,		/**< Number of compact state representation buffers to create (input) */
				int stateCount,				/**< Number of states in the continuous-time Markov chain (input) */
				int patternCount,			/**< Number of site patterns to be handled by the instance (input) */
				int eigenBufferCount,		/**< Number of rate matrix eigen-decomposition buffers to allocate (input) */
				int matrixBufferCount		/**< Number of rate matrix buffers (input) */
                ) {
	BeagleImpl* impl = new BeagleCPUImpl();
	try {
		if (impl->initialize(tipCount, partialsBufferCount, compactBufferCount, stateCount, patternCount, eigenBufferCount, matrixBufferCount) == 0)
			return impl;
	}
	catch(...)
	{
		std::cerr << "exception in initialize\n";
		delete impl;
		throw;
	}
	delete impl;
	return NULL;
}

const char* BeagleCPUImplFactory::getName() {
	return "CPU";
}

