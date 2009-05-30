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

#define MATRIX_SIZE (STATE_COUNT + 1) * STATE_COUNT

#if (STATE_COUNT==4)
#	define IS_NUCLEOTIDES
#endif


int BeagleCPUImpl::initialize(	int bufferCount,
					int maxTipStateCount,
					int stateCount,
					int patternCount,
					int eigenDecompositionCount,
					int matrixCount)
{
	
	kBufferCount = bufferCount;
	kTipCount = maxTipStateCount;
	assert(bufferCount > kTipCount);
	kStateCount = stateCount;
	kPatternCount = patternCount;
	kMatrixCount = matrixCount;


	kMatrixSize = (1+stateCount)*stateCount;
	
	
	cMatrices = (double **)malloc(sizeof(double *) * eigenDecompositionCount);
	if (cMatrices == 0L)
		throw std::bad_alloc();
	storedCMatrices = (double **)malloc(sizeof(double *) * eigenDecompositionCount);
	if (storedCMatrices == 0L)
		throw std::bad_alloc();
	eigenValues = (double **)malloc(sizeof(double *) * eigenDecompositionCount);
	if (eigenValues == 0L)
		throw std::bad_alloc();
	storedEigenValues = (double **)malloc(sizeof(double *) * eigenDecompositionCount);
	if (storedEigenValues == 0L)
		throw std::bad_alloc();

	for (int i = 0; i < eigenDecompositionCount; i++) {
		cMatrices[i] = (double *)malloc(sizeof(double) * stateCount * stateCount * stateCount);
		if (cMatrices[i] == 0L)
			throw std::bad_alloc();
		storedCMatrices[i] = (double *)malloc(sizeof(double) * stateCount * stateCount * stateCount);
		if (storedCMatrices[i] == 0L)
			throw std::bad_alloc();
		eigenValues[i] = (double *)malloc(sizeof(double) * stateCount);
		if (eigenValues[i] == 0L)
			throw std::bad_alloc();
		storedEigenValues[i] = (double *)malloc(sizeof(double) * stateCount);
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
    useTipPartials = false;

	for (int i = 0; i < kTipCount; i++) {
		tipStates[i] = (int *)malloc(sizeof(int) * kPatternCount);
	}
	for (int i = kTipCount; i < kBufferCount; i++) {
		partials[i] = (double *)malloc(sizeof(double) * kPartialsSize);
		if (partials[i] == 0L)
			throw std::bad_alloc();
	}



	std::vector<double> emptyMat(kMatrixSize);
	transitionMatrices.assign(kMatrixCount, emptyMat);
	
	fprintf(stderr,"done through here!\n");
	return SUCCESS;
}

BeagleCPUImpl::~BeagleCPUImpl()
{
	// free all that stuff...
}

void BeagleCPUImpl::setPartials(int bufferIndex, const double* inPartials)
{
	assert(partials[bufferIndex] == 0L);
	partials[bufferIndex] = (double *)malloc(sizeof(double) * kPartialsSize);
	if (partials[i] == 0L)
		return OUT_OF_MEMORY_ERROR;
	memcpy(partials[bufferIndex], inPartials, sizeof(double) * kPartialsSize);
	return NO_ERROR;
}


int BeagleCPUImpl::getPartials(int bufferIndex, double *outPartials)
{
	memcpy(outPartials, partials[bufferIndex], sizeof(double)*kPartialsSize)
	return NO_ERROR;
}

void BeagleCPUImpl::setTipStates(
				  int tipIndex,
				  const int* inStates)
{
    tipStates[tipIndex] = (int *)malloc(sizeof(int) * kPatternCount * kCategoryCount);
	int k = 0;
	for (int i = 0; i < kCategoryCount; i++) {
		for (int j = 0; j < kPatternCount; j++) {
			tipStates[tipIndex][k] = (inStates[j] < stateCount ? inStates[j] : stateCount);
			k++;
		}
	}
}

void BeagleCPUImpl::setStateFrequencies(const double* inStateFrequencies)
{
	memcpy(frequencies, inStateFrequencies, sizeof(double) * stateCount);
}

void BeagleCPUImpl::setEigenDecomposition(
						   int eigenIndex,
						   const double** inEigenVectors,
						   const double** inInverseEigenVectors,
						   const double* inEigenValues)
{

	int l =0;
	for (int i = 0; i < stateCount; i++) {
		eigenValues[matrixIndex][i] = inEigenValues[i];

		for (int j = 0; j < stateCount; j++) {
			for (int k = 0; k < stateCount; k++) {
				cMatrices[matrixIndex][l] = inEigenVectors[i][k] * inInverseEigenVectors[k][j];
				l++;
			}
		}
	}
}


int BeagleCPUImpl::setTransitionMatrix(int matrixIndex, const double* inMatrix)
{
assert(false);
}


void BeagleCPUImpl::updateTransitionMatrices(
											int eigenIndex,
											const int* probabilityIndices,
											const int* firstDerivativeIndices,
											const int* secondDervativeIndices,
											const double* edgeLengths,
                                            int count)
{
	static double tmp[stateCount];

    for (int u = 0; u < count; u++) {
        int nodeIndex = nodeIndices[u];

		currentMatricesIndices[nodeIndex] = 1 - currentMatricesIndices[nodeIndex];

		int n = 0;
		int matrixIndex = 0;
		for (int l = 0; l < kCategoryCount; l++) {
			for (int i = 0; i < stateCount; i++) {
				tmp[i] =  exp(eigenValues[matrixIndex][i] * branchLengths[u] * categoryRates[l]);
			}

			int m = 0;
			for (int i = 0; i < stateCount; i++) {
				for (int j = 0; j < stateCount; j++) {
					double sum = 0.0;
					for (int k = 0; k < stateCount; k++) {
						sum += cMatrices[matrixIndex][m] * tmp[k];
						m++;
					}
					if (sum > 0)
						matrices[currentMatricesIndices[nodeIndex]][nodeIndex][n] = sum;
					else
						matrices[currentMatricesIndices[nodeIndex]][nodeIndex][n] = 0;

					n++;
				}
				matrices[currentMatricesIndices[nodeIndex]][nodeIndex][n] = 1.0;
				n++;
			}
			if (kMatrixCount > 1) {
				matrixIndex ++;
			}
		}

    }

}

void BeagleCPUImpl::updatePartials(
					   int* operations,
					   int count,
					   int rescale)
{

    int x = 0;
	for (int op = 0; op < count; op++) {
		int nodeIndex1 = operations[x];
		x++;
		int nodeIndex2 = operations[x];
		x++;
		int nodeIndex3 = operations[x];
		x++;
		currentPartialsIndices[nodeIndex3] = 1 - currentPartialsIndices[nodeIndex3];

        if (useTipPartials) {
            updatePartialsPartials(nodeIndex1, nodeIndex2, nodeIndex3);
        } else {
		    if (nodeIndex1 < kTipCount) {
			    if (nodeIndex2 < kTipCount) {
			    	updateStatesStates(nodeIndex1, nodeIndex2, nodeIndex3);
			    } else {
			    	updateStatesPartials(nodeIndex1, nodeIndex2, nodeIndex3);
			    }
		    } else {
			    if (nodeIndex2 < kTipCount) {
			    	updateStatesPartials(nodeIndex2, nodeIndex1, nodeIndex3);
			    } else {
			    	updatePartialsPartials(nodeIndex1, nodeIndex2, nodeIndex3);
			    }
		    }
		}
	}
}

/*
 * Calculates partial likelihoods at a node when both children have states.
 */
void BeagleCPUImpl::updateStatesStates(int nodeIndex1, int nodeIndex2, int nodeIndex3)
{
	double* matrices1 = matrices[currentMatricesIndices[nodeIndex1]][nodeIndex1];
	double* matrices2 = matrices[currentMatricesIndices[nodeIndex2]][nodeIndex2];

	int* states1 = tipStates[nodeIndex1];
	int* states2 = tipStates[nodeIndex2];

	double* partials3 = partials[currentPartialsIndices[nodeIndex3]][nodeIndex3];

#ifdef IS_NUCLEOTIDES

	int v = 0;
	for (int l = 0; l < kCategoryCount; l++) {

		for (int k = 0; k < kPatternCount; k++) {

			int state1 = states1[k];
			int state2 = states2[k];

			int w = l * MATRIX_SIZE;

			partials3[v] = matrices1[w + state1] * matrices2[w + state2];
			v++;	w += (stateCount + 1);
			partials3[v] = matrices1[w + state1] * matrices2[w + state2];
			v++;	w += (stateCount + 1);
			partials3[v] = matrices1[w + state1] * matrices2[w + state2];
			v++;	w += (stateCount + 1);
			partials3[v] = matrices1[w + state1] * matrices2[w + state2];
			v++;	w += (stateCount + 1);

		}
	}

#else

	int v = 0;
	for (int l = 0; l < kCategoryCount; l++) {

		for (int k = 0; k < kPatternCount; k++) {

			int state1 = states1[k];
			int state2 = states2[k];

			int w = l * kMatrixSize;

			for (int i = 0; i < stateCount; i++) {

				partials3[v] = matrices1[w + state1] * matrices2[w + state2];

				v++;
				w += (stateCount + 1);
			}

		}
	}
#endif
}

/*
 * Calculates partial likelihoods at a node when one child has states and one has partials.
 */
void BeagleCPUImpl::updateStatesPartials(int nodeIndex1, int nodeIndex2, int nodeIndex3)
{
	double* matrices1 = matrices[currentMatricesIndices[nodeIndex1]][nodeIndex1];
	double* matrices2 = matrices[currentMatricesIndices[nodeIndex2]][nodeIndex2];

	int* states1 = tipStates[nodeIndex1];
	double* partials2 = partials[currentPartialsIndices[nodeIndex2]][nodeIndex2];

	double* partials3 = partials[currentPartialsIndices[nodeIndex3]][nodeIndex3];

#ifdef IS_NUCLEOTIDES

	int u = 0;
	int v = 0;

	for (int l = 0; l < kCategoryCount; l++) {
		for (int k = 0; k < kPatternCount; k++) {

			int state1 = states1[k];

			int w = l * MATRIX_SIZE;

			partials3[u] = matrices1[w + state1];

			double sum = matrices2[w] * partials2[v]; w++;
			sum +=	matrices2[w] * partials2[v + 1]; w++;
			sum +=	matrices2[w] * partials2[v + 2]; w++;
			sum +=	matrices2[w] * partials2[v + 3]; w++;
			w++; // increment for the extra column at the end
			partials3[u] *= sum;	u++;

			partials3[u] = matrices1[w + state1];

			sum = matrices2[w] * partials2[v]; w++;
			sum +=	matrices2[w] * partials2[v + 1]; w++;
			sum +=	matrices2[w] * partials2[v + 2]; w++;
			sum +=	matrices2[w] * partials2[v + 3]; w++;
			w++; // increment for the extra column at the end
			partials3[u] *= sum;	u++;

			partials3[u] = matrices1[w + state1];

			sum = matrices2[w] * partials2[v]; w++;
			sum +=	matrices2[w] * partials2[v + 1]; w++;
			sum +=	matrices2[w] * partials2[v + 2]; w++;
			sum +=	matrices2[w] * partials2[v + 3]; w++;
			w++; // increment for the extra column at the end
			partials3[u] *= sum;	u++;

			partials3[u] = matrices1[w + state1];

			sum = matrices2[w] * partials2[v]; w++;
			sum +=	matrices2[w] * partials2[v + 1]; w++;
			sum +=	matrices2[w] * partials2[v + 2]; w++;
			sum +=	matrices2[w] * partials2[v + 3]; w++;
			w++; // increment for the extra column at the end
			partials3[u] *= sum;	u++;

			v += 4;

		}
	}

#else
	int u = 0;
	int v = 0;

	for (int l = 0; l < kCategoryCount; l++) {
		for (int k = 0; k < kPatternCount; k++) {

			int state1 = states1[k];

			int w = l * kMatrixSize;

			for (int i = 0; i < stateCount; i++) {

				double tmp = matrices1[w + state1];

				double sum = 0.0;
				for (int j = 0; j < stateCount; j++) {
					sum += matrices2[w] * partials2[v + j];
					w++;
				}

				// increment for the extra column at the end
				w++;

				partials3[u] = tmp * sum;
				u++;
			}

			v += stateCount;
		}
	}
#endif
}

void BeagleCPUImpl::updatePartialsPartials(int nodeIndex1, int nodeIndex2, int nodeIndex3)
{
	double* matrices1 = matrices[currentMatricesIndices[nodeIndex1]][nodeIndex1];
	double* matrices2 = matrices[currentMatricesIndices[nodeIndex2]][nodeIndex2];

	double* partials1 = partials[currentPartialsIndices[nodeIndex1]][nodeIndex1];
	double* partials2 = partials[currentPartialsIndices[nodeIndex2]][nodeIndex2];

	double* partials3 = partials[currentPartialsIndices[nodeIndex3]][nodeIndex3];

	/* fprintf(stdout, "*** operation %d: %d, %d -> %d\n", op, nodeIndex1, nodeIndex2, nodeIndex3); */

	double sum1, sum2;

#ifdef IS_NUCLEOTIDES

	int u = 0;
	int v = 0;

	for (int l = 0; l < kCategoryCount; l++) {
		for (int k = 0; k < kPatternCount; k++) {

			int w = l * MATRIX_SIZE;

			sum1 = matrices1[w] * partials1[v];
			sum2 = matrices2[w] * partials2[v]; w++;
			sum1 += matrices1[w] * partials1[v + 1];
			sum2 += matrices2[w] * partials2[v + 1]; w++;
			sum1 += matrices1[w] * partials1[v + 2];
			sum2 += matrices2[w] * partials2[v + 2]; w++;
			sum1 += matrices1[w] * partials1[v + 3];
			sum2 += matrices2[w] * partials2[v + 3]; w++;
			w++; // increment for the extra column at the end
			partials3[u] = sum1 * sum2; u++;

			sum1 = matrices1[w] * partials1[v];
			sum2 = matrices2[w] * partials2[v]; w++;
			sum1 += matrices1[w] * partials1[v + 1];
			sum2 += matrices2[w] * partials2[v + 1]; w++;
			sum1 += matrices1[w] * partials1[v + 2];
			sum2 += matrices2[w] * partials2[v + 2]; w++;
			sum1 += matrices1[w] * partials1[v + 3];
			sum2 += matrices2[w] * partials2[v + 3]; w++;
			w++; // increment for the extra column at the end
			partials3[u] = sum1 * sum2; u++;

			sum1 = matrices1[w] * partials1[v];
			sum2 = matrices2[w] * partials2[v]; w++;
			sum1 += matrices1[w] * partials1[v + 1];
			sum2 += matrices2[w] * partials2[v + 1]; w++;
			sum1 += matrices1[w] * partials1[v + 2];
			sum2 += matrices2[w] * partials2[v + 2]; w++;
			sum1 += matrices1[w] * partials1[v + 3];
			sum2 += matrices2[w] * partials2[v + 3]; w++;
			w++; // increment for the extra column at the end
			partials3[u] = sum1 * sum2; u++;

			sum1 = matrices1[w] * partials1[v];
			sum2 = matrices2[w] * partials2[v]; w++;
			sum1 += matrices1[w] * partials1[v + 1];
			sum2 += matrices2[w] * partials2[v + 1]; w++;
			sum1 += matrices1[w] * partials1[v + 2];
			sum2 += matrices2[w] * partials2[v + 2]; w++;
			sum1 += matrices1[w] * partials1[v + 3];
			sum2 += matrices2[w] * partials2[v + 3]; w++;
			w++; // increment for the extra column at the end
			partials3[u] = sum1 * sum2; u++;

			v += 4;

		}
	}

#else

	int u = 0;
	int v = 0;

	for (int l = 0; l < kCategoryCount; l++) {

		for (int k = 0; k < kPatternCount; k++) {

			int w = l * kMatrixSize;

			for (int i = 0; i < stateCount; i++) {

				sum1 = sum2 = 0.0;

				for (int j = 0; j < stateCount; j++) {
					sum1 += matrices1[w] * partials1[v + j];
					sum2 += matrices2[w] * partials2[v + j];
					w++;
				}

				// increment for the extra column at the end
				w++;

				partials3[u] = sum1 * sum2;

				u++;
			}
			v += stateCount;
		}
	}

#endif
}

void BeagleCPUImpl::calculateRootLogLikelihoods(
	const int* bufferIndices,
	int count,
	const double* weights,
	const double** stateFrequencies,
	double* outLogLikelihoods)
{

	double* rootPartials = partials[currentPartialsIndices[rootNodeIndex]][rootNodeIndex];
//    printArray("rootPartials", rootPartials, kPatternCount * stateCount);
//    printArray("frequencies", frequencies, stateCount);
//    printArray("categoryProportions", categoryProportions, kCategoryCount);

	int u = 0;
	int v = 0;
	for (int k = 0; k < kPatternCount; k++) {

		for (int i = 0; i < stateCount; i++) {

			integrationTmp[u] = rootPartials[v] * categoryProportions[0];
			u++;
			v++;
		}
	}


	for (int l = 1; l < kCategoryCount; l++) {
		u = 0;

		for (int k = 0; k < kPatternCount; k++) {

			for (int i = 0; i < stateCount; i++) {

				integrationTmp[u] += rootPartials[v] * categoryProportions[l];
				u++;
				v++;
			}
		}
	}

//    printArray("integrationTmp", integrationTmp, kPatternCount * stateCount);

	u = 0;
	for (int k = 0; k < kPatternCount; k++) {

		double sum = 0.0;
		for (int i = 0; i < stateCount; i++) {

			sum += frequencies[i] * integrationTmp[u];
			u++;
		}
		outLogLikelihoods[k] = log(sum);
	}

//    printArray("outLogLikelihoods", outLogLikelihoods, kPatternCount);
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
//not implemented, yet
}

BeagleImpl*  BeagleCPUImplFactory::createImpl(
						int bufferCount,
						int tipCount,
						int stateCount,
						int patternCount,
						int eigenDecompositionCount,
						int matrixCount) {
	BeagleImpl* impl = new BeagleCPUImpl();
	try {
		if(impl->initialize(nodeCount,tipCount,stateCount,patternCount,categoryCount,matrixCount))
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

