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
#define IS_NUCLEOTIDES
#endif

int BeagleCPUImpl::initialize(
				int nodeCount,
				int tipCount,
				int stateCount,
				int patternCount,
				int categoryCount,
				int matrixCount)
{
	kNodeCount = nodeCount;
	kTipCount = tipCount;

	kPatternCount = patternCount;
	kMatrixCount = matrixCount;
	kCategoryCount = categoryCount;

	cMatrices = (double **)malloc(sizeof(double *) * kMatrixCount);
	storedCMatrices = (double **)malloc(sizeof(double *) * kMatrixCount);
	eigenValues = (double **)malloc(sizeof(double *) * kMatrixCount);
	storedEigenValues = (double **)malloc(sizeof(double *) * kMatrixCount);

	for (int i = 0; i < kMatrixCount; i++) {
		cMatrices[i] = (double *)malloc(sizeof(double) * STATE_COUNT * STATE_COUNT * STATE_COUNT);
		storedCMatrices[i] = (double *)malloc(sizeof(double) * STATE_COUNT * STATE_COUNT * STATE_COUNT);
		eigenValues[i] = (double *)malloc(sizeof(double) * STATE_COUNT);
		storedEigenValues[i] = (double *)malloc(sizeof(double) * STATE_COUNT);
	}

	frequencies = (double *)malloc(sizeof(double) * STATE_COUNT);
	storedFrequencies = (double *)malloc(sizeof(double) * STATE_COUNT);

	categoryRates = (double *)malloc(sizeof(double) * kCategoryCount);
	storedCategoryRates = (double *)malloc(sizeof(double) * kCategoryCount);

	categoryProportions = (double *)malloc(sizeof(double) * kCategoryCount);
	storedCategoryProportions = (double *)malloc(sizeof(double) * kCategoryCount);

	branchLengths = (double *)malloc(sizeof(double) * kNodeCount);
	storedBranchLengths = (double *)malloc(sizeof(double) * kNodeCount);

	// a temporary array used in calculating log likelihoods
	integrationTmp = (double *)malloc(sizeof(double) * patternCount * STATE_COUNT);

	kPartialsSize = kPatternCount * STATE_COUNT * kCategoryCount;

	partials = (double ***)malloc(sizeof(double**) * 2);
	partials[0] = (double **)malloc(sizeof(double*) * kNodeCount);
	partials[1] = (double **)malloc(sizeof(double*) * kNodeCount);

	tipStates = (int **)malloc(sizeof(int*) * kTipCount);
	for (int i = 0; i < kTipCount; i++) {
		partials[0][i] = NULL;
		partials[1][i] = NULL;
		tipStates[i] = NULL;
    }
    useTipPartials = false;

	for (int i = kTipCount; i < kNodeCount; i++) {
		partials[0][i] = (double *)malloc(sizeof(double) * kPartialsSize);
		partials[1][i] = (double *)malloc(sizeof(double) * kPartialsSize);
	}

  	currentMatricesIndices = (int *)malloc(sizeof(int) * kNodeCount);
  	memset(currentMatricesIndices, 0, sizeof(int) * kNodeCount);
  	storedMatricesIndices = (int *)malloc(sizeof(int) * kNodeCount);

  	currentPartialsIndices = (int *)malloc(sizeof(int) * kNodeCount);
  	memset(currentPartialsIndices, 0, sizeof(int) * kNodeCount);
  	storedPartialsIndices = (int *)malloc(sizeof(int) * kNodeCount);

	matrices = (double ***)malloc(sizeof(double**) * 2);
	matrices[0] = (double **)malloc(sizeof(double*) * kNodeCount);
	matrices[1] = (double **)malloc(sizeof(double*) * kNodeCount);
	for (int i = 0; i < kNodeCount; i++) {
		matrices[0][i] = (double *)malloc(sizeof(double) * kCategoryCount * MATRIX_SIZE);
		matrices[1][i] = (double *)malloc(sizeof(double) * kCategoryCount * MATRIX_SIZE);
	}

	fprintf(stderr,"done through here!\n");
	return SUCCESS;
}

void BeagleCPUImpl::finalize()
{
	// free all that stuff...
}

void BeagleCPUImpl::setTipPartials(
					int tipIndex,
					double* inPartials)
{
	partials[0][tipIndex] = (double *)malloc(sizeof(double) * kPartialsSize);
	int k = 0;
	for (int i = 0; i < kCategoryCount; i++) {
		// set the partials identically for each matrix
		memcpy(partials[0][tipIndex] + k, inPartials, sizeof(double) * kPatternCount * STATE_COUNT);
		k += kPatternCount * STATE_COUNT;
	}
    useTipPartials = true;
}

void BeagleCPUImpl::setTipStates(
				  int tipIndex,
				  int* inStates)
{
    tipStates[tipIndex] = (int *)malloc(sizeof(int) * kPatternCount * kCategoryCount);
	int k = 0;
	for (int i = 0; i < kCategoryCount; i++) {
		for (int j = 0; j < kPatternCount; j++) {
			tipStates[tipIndex][k] = (inStates[j] < STATE_COUNT ? inStates[j] : STATE_COUNT);
			k++;
		}
	}
}

void BeagleCPUImpl::setStateFrequencies(double* inStateFrequencies)
{
	memcpy(frequencies, inStateFrequencies, sizeof(double) * STATE_COUNT);
}

void BeagleCPUImpl::setEigenDecomposition(
						   int matrixIndex,
						   double** inEigenVectors,
						   double** inInverseEigenVectors,
						   double* inEigenValues)
{

	int l =0;
	for (int i = 0; i < STATE_COUNT; i++) {
		eigenValues[matrixIndex][i] = inEigenValues[i];

		for (int j = 0; j < STATE_COUNT; j++) {
			for (int k = 0; k < STATE_COUNT; k++) {
				cMatrices[matrixIndex][l] = inEigenVectors[i][k] * inInverseEigenVectors[k][j];
				l++;
			}
		}
	}
}

void BeagleCPUImpl::setCategoryRates(double* inCategoryRates)
{
	memcpy(categoryRates, inCategoryRates, sizeof(double) * kCategoryCount);
}

void BeagleCPUImpl::setCategoryProportions(double* inCategoryProportions)
{
	memcpy(categoryProportions, inCategoryProportions, sizeof(double) * kCategoryCount);
}

void BeagleCPUImpl::calculateProbabilityTransitionMatrices(
                                            int* nodeIndices,
                                            double* branchLengths,
                                            int count)
{
	static double tmp[STATE_COUNT];

    for (int u = 0; u < count; u++) {
        int nodeIndex = nodeIndices[u];

		currentMatricesIndices[nodeIndex] = 1 - currentMatricesIndices[nodeIndex];

		int n = 0;
		int matrixIndex = 0;
		for (int l = 0; l < kCategoryCount; l++) {
			for (int i = 0; i < STATE_COUNT; i++) {
				tmp[i] =  exp(eigenValues[matrixIndex][i] * branchLengths[u] * categoryRates[l]);
			}

			int m = 0;
			for (int i = 0; i < STATE_COUNT; i++) {
				for (int j = 0; j < STATE_COUNT; j++) {
					double sum = 0.0;
					for (int k = 0; k < STATE_COUNT; k++) {
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

void BeagleCPUImpl::calculatePartials(
					   int* operations,
					   int* dependencies,
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
			v++;	w += (STATE_COUNT + 1);
			partials3[v] = matrices1[w + state1] * matrices2[w + state2];
			v++;	w += (STATE_COUNT + 1);
			partials3[v] = matrices1[w + state1] * matrices2[w + state2];
			v++;	w += (STATE_COUNT + 1);
			partials3[v] = matrices1[w + state1] * matrices2[w + state2];
			v++;	w += (STATE_COUNT + 1);

		}
	}

#else

	int v = 0;
	for (int l = 0; l < kCategoryCount; l++) {

		for (int k = 0; k < kPatternCount; k++) {

			int state1 = states1[k];
			int state2 = states2[k];

			int w = l * MATRIX_SIZE;

			for (int i = 0; i < STATE_COUNT; i++) {

				partials3[v] = matrices1[w + state1] * matrices2[w + state2];

				v++;
				w += (STATE_COUNT + 1);
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

			int w = l * MATRIX_SIZE;

			for (int i = 0; i < STATE_COUNT; i++) {

				double tmp = matrices1[w + state1];

				double sum = 0.0;
				for (int j = 0; j < STATE_COUNT; j++) {
					sum += matrices2[w] * partials2[v + j];
					w++;
				}

				// increment for the extra column at the end
				w++;

				partials3[u] = tmp * sum;
				u++;
			}

			v += STATE_COUNT;
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

			int w = l * MATRIX_SIZE;

			for (int i = 0; i < STATE_COUNT; i++) {

				sum1 = sum2 = 0.0;

				for (int j = 0; j < STATE_COUNT; j++) {
					sum1 += matrices1[w] * partials1[v + j];
					sum2 += matrices2[w] * partials2[v + j];
					w++;
				}

				// increment for the extra column at the end
				w++;

				partials3[u] = sum1 * sum2;

				u++;
			}
			v += STATE_COUNT;
		}
	}

#endif
}

void BeagleCPUImpl::calculateLogLikelihoods(
							 int rootNodeIndex,
							 double* outLogLikelihoods)
{

	double* rootPartials = partials[currentPartialsIndices[rootNodeIndex]][rootNodeIndex];
//    printArray("rootPartials", rootPartials, kPatternCount * STATE_COUNT);
//    printArray("frequencies", frequencies, STATE_COUNT);
//    printArray("categoryProportions", categoryProportions, kCategoryCount);

	int u = 0;
	int v = 0;
	for (int k = 0; k < kPatternCount; k++) {

		for (int i = 0; i < STATE_COUNT; i++) {

			integrationTmp[u] = rootPartials[v] * categoryProportions[0];
			u++;
			v++;
		}
	}


	for (int l = 1; l < kCategoryCount; l++) {
		u = 0;

		for (int k = 0; k < kPatternCount; k++) {

			for (int i = 0; i < STATE_COUNT; i++) {

				integrationTmp[u] += rootPartials[v] * categoryProportions[l];
				u++;
				v++;
			}
		}
	}

//    printArray("integrationTmp", integrationTmp, kPatternCount * STATE_COUNT);

	u = 0;
	for (int k = 0; k < kPatternCount; k++) {

		double sum = 0.0;
		for (int i = 0; i < STATE_COUNT; i++) {

			sum += frequencies[i] * integrationTmp[u];
			u++;
		}
		outLogLikelihoods[k] = log(sum);
	}

//    printArray("outLogLikelihoods", outLogLikelihoods, kPatternCount);
}

// store the current state of all partials and matrices
void BeagleCPUImpl::storeState()
{
	for (int i = 0; i < kMatrixCount; i++) {
		memcpy(storedCMatrices[i], cMatrices[i], sizeof(double) * STATE_COUNT * STATE_COUNT * STATE_COUNT);
		memcpy(storedEigenValues[i], eigenValues[i], sizeof(double) * STATE_COUNT);
	}

	memcpy(storedFrequencies, frequencies, sizeof(double) * STATE_COUNT);
	memcpy(storedCategoryRates, categoryRates, sizeof(double) * kCategoryCount);
	memcpy(storedCategoryProportions, categoryProportions, sizeof(double) * kCategoryCount);
	memcpy(storedBranchLengths, branchLengths, sizeof(double) * kNodeCount);

	memcpy(storedMatricesIndices, currentMatricesIndices, sizeof(int) * kNodeCount);
	memcpy(storedPartialsIndices, currentPartialsIndices, sizeof(int) * kNodeCount);
}

// restore the stored state after a rejected move
void BeagleCPUImpl::restoreState()
{
	// Rather than copying the stored stuff back, just swap the pointers...
	double** tmp = cMatrices;
	cMatrices = storedCMatrices;
	storedCMatrices = tmp;

	tmp = eigenValues;
	eigenValues = storedEigenValues;
	storedEigenValues = tmp;

	double *tmp1 = frequencies;
	frequencies = storedFrequencies;
	storedFrequencies = tmp1;

	tmp1 = categoryRates;
	categoryRates = storedCategoryRates;
	storedCategoryRates = tmp1;

	tmp1 = categoryProportions;
	categoryProportions = storedCategoryProportions;
	storedCategoryProportions = tmp1;

	tmp1 = branchLengths;
	branchLengths = storedBranchLengths;
	storedBranchLengths = tmp1;

	int* tmp2 = currentMatricesIndices;
	currentMatricesIndices = storedMatricesIndices;
	storedMatricesIndices = tmp2;

	tmp2 = currentPartialsIndices;
	currentPartialsIndices = storedPartialsIndices;
	storedPartialsIndices = tmp2;
}

BeagleImpl*  BeagleCPUImplFactory::createImpl(
						int nodeCount,
						int tipCount,
						int stateCount,
						int patternCount,
						int categoryCount,
						int matrixCount) {
	BeagleImpl* impl = new BeagleCPUImpl();
	if(impl->initialize(nodeCount,tipCount,stateCount,patternCount,categoryCount,matrixCount))
		return impl;
	impl->finalize();
	delete impl;
	return NULL;
}

const char* BeagleCPUImplFactory::getName() {
	return "CPU";
}



