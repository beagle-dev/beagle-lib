/*
 *  BeagleImpl.h
 *  BEAGLE
 *
 * @author Andrew Rambaut
 * @author Marc Suchard
 *
 */

#ifndef __beagle_impl__
#define __beagle_impl__

#ifdef DOUBLE_PRECISION
	#define REAL	double
#else
	#define REAL	float
#endif

#define SUCCESS	1
#define ERROR	0

class BeagleImpl {
public:
	virtual int initialize(
					int nodeCount,
					int tipCount,
					int stateCount,
					int patternCount,
					int categoryCount,
					int matrixCount) = 0;

	virtual void finalize() = 0;

	virtual void setTipPartials(
						int tipIndex,
						double* inPartials) = 0;

	virtual void setTipStates(
					  int tipIndex,
					  int* inStates) = 0;

	virtual void setStateFrequencies(double* inStateFrequencies) = 0;

	virtual void setEigenDecomposition(
							   int matrixIndex,
							   double** inEigenVectors,
							   double** inInverseEigenVectors,
							   double* inEigenValues) = 0;

	virtual void setCategoryRates(double* inCategoryRates) = 0;

	virtual void setCategoryProportions(double* inCategoryProportions) = 0;

	virtual void calculateProbabilityTransitionMatrices(
												int* nodeIndices,
												double* branchLengths,
												int count) = 0;

	virtual void calculatePartials(
						   int* operations,
						   int* dependencies,
						   int count,
						   int rescale) = 0;

	virtual void calculateLogLikelihoods(
								 int rootNodeIndex,
								 double* outLogLikelihoods) = 0;

	virtual void storeState() = 0;

	virtual void restoreState() = 0;

};

#endif // __beagle_impl__
