/*
 * @author Andrew Rambaut
 * @author Marc Suchard
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "beagle.h"

#define MATRIX_SIZE (STATE_COUNT + 1) * STATE_COUNT
#if (STATE_COUNT==4)
#define IS_NUCLEOTIDES
#endif

int kNodeCount;
int kTipCount;
int kPatternCount;
int kPartialsSize;
int kMatrixCount;
int kCategoryCount;

REAL** cMatrices;
REAL** storedCMatrices;
REAL** eigenValues;
REAL** storedEigenValues;

REAL* frequencies;
REAL* storedFrequencies;
REAL* categoryProportions;
REAL* storedCategoryProportions;
REAL* categoryRates;
REAL* storedCategoryRates;

REAL* branchLengths;
REAL* storedBranchLengths;

REAL* integrationTmp;

REAL*** partials;
int** states;
REAL*** matrices;

int* currentMatricesIndices;
int* storedMatricesIndices;
int* currentPartialsIndices;
int* storedPartialsIndices;

void updateStatesStates(int nodeIndex1, int nodeIndex2, int nodeIndex3);
void updateStatesPartials(int nodeIndex1, int nodeIndex2, int nodeIndex3);
void updatePartialsPartials(int nodeIndex1, int nodeIndex2, int nodeIndex3);

void printArray(char* name, REAL *array, int length) {
	fprintf(stdout, "%s:", name);
	for (int i = 0; i < length; i++) {
		fprintf(stdout, " %f", array[i]);
	}
	fprintf(stdout, "\n");
}

// nodeCount the number of nodes in the tree
// tipCount the number of tips in the tree
// patternCount the number of site patterns
// categoryCount the number of rate scalers per branch
// matrixCount the number of Q-matrices per branch (should be 1 or categoryCount)
void initialize(
				int nodeCount,
				int tipCount,
				int patternCount,
				int categoryCount,
				int matrixCount)
{
	kNodeCount = nodeCount;
	kTipCount = tipCount;

	kPatternCount = patternCount;
	kMatrixCount = matrixCount;
	kCategoryCount = categoryCount;

	cMatrices = (REAL **)malloc(sizeof(REAL *) * kMatrixCount);
	storedCMatrices = (REAL **)malloc(sizeof(REAL *) * kMatrixCount);
	eigenValues = (REAL **)malloc(sizeof(REAL *) * kMatrixCount);
	storedEigenValues = (REAL **)malloc(sizeof(REAL *) * kMatrixCount);

	for (int i = 0; i < kMatrixCount; i++) {
		cMatrices[i] = (REAL *)malloc(sizeof(REAL) * STATE_COUNT * STATE_COUNT * STATE_COUNT);
		storedCMatrices[i] = (REAL *)malloc(sizeof(REAL) * STATE_COUNT * STATE_COUNT * STATE_COUNT);
		eigenValues[i] = (REAL *)malloc(sizeof(REAL) * STATE_COUNT);
		storedEigenValues[i] = (REAL *)malloc(sizeof(REAL) * STATE_COUNT);
	}

	frequencies = (REAL *)malloc(sizeof(REAL) * STATE_COUNT);
	storedFrequencies = (REAL *)malloc(sizeof(REAL) * STATE_COUNT);

	categoryRates = (REAL *)malloc(sizeof(REAL) * kCategoryCount);
	storedCategoryRates = (REAL *)malloc(sizeof(REAL) * kCategoryCount);

	categoryProportions = (REAL *)malloc(sizeof(REAL) * kCategoryCount);
	storedCategoryProportions = (REAL *)malloc(sizeof(REAL) * kCategoryCount);

	branchLengths = (REAL *)malloc(sizeof(REAL) * kNodeCount);
	storedBranchLengths = (REAL *)malloc(sizeof(REAL) * kNodeCount);

	// a temporary array used in calculating log likelihoods
	integrationTmp = (REAL *)malloc(sizeof(REAL) * patternCount * STATE_COUNT);

	kPartialsSize = kPatternCount * STATE_COUNT * kCategoryCount;

	partials = (REAL ***)malloc(sizeof(REAL**) * 2);
	partials[0] = (REAL **)malloc(sizeof(REAL*) * nodeCount);
	partials[1] = (REAL **)malloc(sizeof(REAL*) * nodeCount);

	states = (int **)malloc(sizeof(int*) * nodeCount);

	for (int i = 0; i < nodeCount; i++) {
		partials[0][i] = (REAL *)malloc(sizeof(REAL) * kPartialsSize);
		partials[1][i] = (REAL *)malloc(sizeof(REAL) * kPartialsSize);
		states[i] = (int *)malloc(sizeof(int) * kPatternCount * kCategoryCount);
	}

  	currentMatricesIndices = (int *)malloc(sizeof(int) * kNodeCount);
  	memset(currentMatricesIndices, 0, sizeof(int) * kNodeCount);
  	storedMatricesIndices = (int *)malloc(sizeof(int) * kNodeCount);

  	currentPartialsIndices = (int *)malloc(sizeof(int) * kNodeCount);
  	memset(currentPartialsIndices, 0, sizeof(int) * kNodeCount);
  	storedPartialsIndices = (int *)malloc(sizeof(int) * kNodeCount);

	matrices = (REAL ***)malloc(sizeof(REAL**) * 2);
	matrices[0] = (REAL **)malloc(sizeof(REAL*) * kNodeCount);
	matrices[1] = (REAL **)malloc(sizeof(REAL*) * kNodeCount);
	for (int i = 0; i < kNodeCount; i++) {
		matrices[0][i] = (REAL *)malloc(sizeof(REAL) * kCategoryCount * MATRIX_SIZE);
		matrices[1][i] = (REAL *)malloc(sizeof(REAL) * kCategoryCount * MATRIX_SIZE);
	}

}

// finalize and dispose of memory allocation if needed
void finalize()
{
	// free all that stuff...
}

// set the partials for a given tip
//
// tipIndex the index of the tip
// inPartials the array of partials, stateCount x patternCount
void setTipPartials(
					int tipIndex,
					REAL* inPartials)
{
	int k = 0;
	for (int i = 0; i < kCategoryCount; i++) {
		// set the partials identically for each matrix
		memcpy(partials[0][tipIndex] + k, inPartials, sizeof(REAL) * kPatternCount * STATE_COUNT);
		k += kPatternCount * STATE_COUNT;
	}
}

// set the states for a given tip when data is unambiguous
//
// tipIndex the index of the tip
// inStates the array of states: 0 to stateCount - 1, missing = stateCount
void setTipStates(
				  int tipIndex,
				  int* inStates)
{
	int k = 0;
	for (int i = 0; i < kCategoryCount; i++) {
		for (int j = 0; j < kPatternCount; j++) {
			states[tipIndex][k] = (inStates[j] < STATE_COUNT ? inStates[j] : STATE_COUNT);
			k++;
		}
	}
}

// set the vector of state frequencies
//
// inStateFrequencies an array containing the state frequencies
void setStateFrequencies(REAL* inStateFrequencies)
{
	memcpy(frequencies, inStateFrequencies, sizeof(REAL) * STATE_COUNT);
}

// sets the Eigen decomposition of a given substitution matrix
//
// matrixIndex the matrix index to update
// eigenVectors an array containing the Eigen Vectors
// inverseEigenVectors an array containing the inverse Eigen Vectors
// eigenValues an array containing the Eigen Values
void setEigenDecomposition(
						   int matrixIndex,
						   REAL** inEigenVectors,
						   REAL** inInverseEigenVectors,
						   REAL* inEigenValues)
{
	
	fprintf(stdout, "setEigenDecomposition\n");

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

// set the vector of category rates
//
// categoryRates an array containing categoryCount rate scalers
void setCategoryRates(REAL* inCategoryRates)
{
	memcpy(categoryRates, inCategoryRates, sizeof(REAL) * kCategoryCount);
}

// set the vector of category proportions
//
// categoryProportions an array containing categoryCount proportions (which sum to 1.0)
void setCategoryProportions(REAL* inCategoryProportions)
{
	memcpy(categoryProportions, inCategoryProportions, sizeof(REAL) * kCategoryCount);
}

// calculate a transition probability matrices for a given node. This will calculate
// for all categories (and all matrices if more than one is being used).
//
// nodeIndex the node that requires the transition probability matrices
// branchLength the expected length of this branch in substitutions per site
void calculateProbabilityTransitionMatrices(int nodeIndex, REAL branchLength)
{
	static REAL tmp[STATE_COUNT];

//	currentMatricesIndices[nodeIndex] = 1 - currentMatricesIndices[nodeIndex];

	int n = 0;
	int matrixIndex = 0;
	for (int l = 0; l < kCategoryCount; l++) {
		for (int i = 0; i < STATE_COUNT; i++) {
			tmp[i] =  exp(eigenValues[matrixIndex][i] * branchLength * categoryRates[l]);
		}

		int m = 0;
		for (int i = 0; i < STATE_COUNT; i++) {
			for (int j = 0; j < STATE_COUNT; j++) {
				REAL sum = 0.0;
				for (int k = 0; k < STATE_COUNT; k++) {
					sum += cMatrices[matrixIndex][m] * tmp[k];
					m++;
				}
				matrices[currentMatricesIndices[nodeIndex]][nodeIndex][n] = sum;
				
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

// calculate partials using an array of operations
//
// operations an array of triplets of indices: the two source partials and the destination
// dependencies an array of indices specify which operations are dependent on which (optional)
// the number of operations
void calculatePartials(
					   int* operations,
					   int* dependencies,
					   int operationCount)
{

    int x = 0;
	for (int op = 0; op < operationCount; op++) {
		int nodeIndex1 = operations[x];
		x++;
		int nodeIndex2 = operations[x];
		x++;
		int nodeIndex3 = operations[x];
		x++;
//		currentPartialsIndices[nodeIndex3] = 1 - currentPartialsIndices[nodeIndex3];

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

/*
 * Calculates partial likelihoods at a node when both children have states.
 */
void updateStatesStates(int nodeIndex1, int nodeIndex2, int nodeIndex3)
{
	REAL* matrices1 = matrices[currentMatricesIndices[nodeIndex1]][nodeIndex1];
	REAL* matrices2 = matrices[currentMatricesIndices[nodeIndex2]][nodeIndex2];

	int* states1 = states[nodeIndex1];
	int* states2 = states[nodeIndex2];

	REAL* partials3 = partials[currentPartialsIndices[nodeIndex3]][nodeIndex3];

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
void updateStatesPartials(int nodeIndex1, int nodeIndex2, int nodeIndex3)
{
	REAL* matrices1 = matrices[currentMatricesIndices[nodeIndex1]][nodeIndex1];
	REAL* matrices2 = matrices[currentMatricesIndices[nodeIndex2]][nodeIndex2];

	int* states1 = states[nodeIndex1];
	REAL* partials2 = partials[currentPartialsIndices[nodeIndex2]][nodeIndex2];

	REAL* partials3 = partials[currentPartialsIndices[nodeIndex3]][nodeIndex3];

    #ifdef IS_NUCLEOTIDES

	int u = 0;
	int v = 0;

	for (int l = 0; l < kCategoryCount; l++) {
		for (int k = 0; k < kPatternCount; k++) {

			int state1 = states1[k];

			int w = l * MATRIX_SIZE;

			partials3[u] = matrices1[w + state1];

			REAL sum = matrices2[w] * partials2[v]; w++;
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

				REAL tmp = matrices1[w + state1];

				REAL sum = 0.0;
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

void updatePartialsPartials(int nodeIndex1, int nodeIndex2, int nodeIndex3)
{
	REAL* matrices1 = matrices[currentMatricesIndices[nodeIndex1]][nodeIndex1];
	REAL* matrices2 = matrices[currentMatricesIndices[nodeIndex2]][nodeIndex2];

	REAL* partials1 = partials[currentPartialsIndices[nodeIndex1]][nodeIndex1];
	REAL* partials2 = partials[currentPartialsIndices[nodeIndex2]][nodeIndex2];

	REAL* partials3 = partials[currentPartialsIndices[nodeIndex3]][nodeIndex3];

	/* fprintf(stdout, "*** operation %d: %d, %d -> %d\n", op, nodeIndex1, nodeIndex2, nodeIndex3); */

	REAL sum1, sum2;

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

// calculate the site log likelihoods at a particular node
//
// rootNodeIndex the index of the root
// outLogLikelihoods an array into which the site log likelihoods will be put
void calculateLogLikelihoods(
							 int rootNodeIndex,
							 REAL* outLogLikelihoods)
{

	REAL* rootPartials = partials[currentPartialsIndices[rootNodeIndex]][rootNodeIndex];

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

	u = 0;
	for (int k = 0; k < kPatternCount; k++) {

		REAL sum = 0.0;
		for (int i = 0; i < STATE_COUNT; i++) {

			sum += frequencies[i] * integrationTmp[u];
			u++;
		}
		outLogLikelihoods[k] = log(sum);
	}
}

// store the current state of all partials and matrices
void storeState()
{
	for (int i = 0; i < kMatrixCount; i++) {
		memcpy(storedCMatrices[i], cMatrices[i], sizeof(REAL) * STATE_COUNT * STATE_COUNT * STATE_COUNT);
		memcpy(storedEigenValues[i], eigenValues[i], sizeof(REAL) * STATE_COUNT);
	}

	memcpy(storedFrequencies, frequencies, sizeof(REAL) * STATE_COUNT);
	memcpy(storedCategoryRates, categoryRates, sizeof(REAL) * kCategoryCount);
	memcpy(storedCategoryProportions, categoryProportions, sizeof(REAL) * kCategoryCount);
	memcpy(storedBranchLengths, branchLengths, sizeof(REAL) * kNodeCount);

	memcpy(storedMatricesIndices, currentMatricesIndices, sizeof(int) * kNodeCount);
	memcpy(storedPartialsIndices, currentPartialsIndices, sizeof(int) * kNodeCount);
}

// restore the stored state after a rejected move
void restoreState()
{
	// Rather than copying the stored stuff back, just swap the pointers...
	REAL** tmp = cMatrices;
	cMatrices = storedCMatrices;
	storedCMatrices = tmp;

	tmp = eigenValues;
	eigenValues = storedEigenValues;
	storedEigenValues = tmp;

	REAL *tmp1 = frequencies;
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

