/*
 *  BeagleCPUImpl.h
 *  BEAGLE
 *
 * @author Andrew Rambaut
 * @author Marc Suchard
 *
 */

#include "BeagleImpl.h"

class BeagleCPUImpl : public BeagleImpl {
private:
int kNodeCount;
int kTipCount;
int kPatternCount;
int kPartialsSize;
int kMatrixCount;
int kCategoryCount;

double** cMatrices;
double** storedCMatrices;
double** eigenValues;
double** storedEigenValues;

double* frequencies;
double* storedFrequencies;
double* categoryProportions;
double* storedCategoryProportions;
double* categoryRates;
double* storedCategoryRates;

double* branchLengths;
double* storedBranchLengths;

double* integrationTmp;

double*** partials;
int** tipStates;
double*** matrices;

bool useTipPartials;

int* currentMatricesIndices;
int* storedMatricesIndices;
int* currentPartialsIndices;
int* storedPartialsIndices;
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

	void calculatePartials(
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


};

