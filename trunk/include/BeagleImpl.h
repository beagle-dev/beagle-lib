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
			int bufferCount,
			int tipCount,
			int stateCount,
			int patternCount,
			int eigenDecompositionCount,
			int matrixCount) = 0;

	virtual void setPartials(
			int* instance,
			int instanceCount,
			int bufferIndex,
			const double* inPartials) = 0;

	virtual void getPartials(
			int* instance,
			int bufferIndex,
			double *outPartials) = 0;

	virtual int setTipStates(
			int* instance,
			int instanceCount,
			int tipIndex,
			const int* inStates) = 0;

	virtual int setStateFrequencies(
			int* instance,
			const double* inStateFrequencies) = 0;

	virtual int setEigenDecomposition(
			int* instance,
			int instanceCount,
			int eigenIndex,
			const double** inEigenVectors,
			const double** inInverseEigenVectors,
			const double* inEigenValues) = 0;

	virtual int setTransitionMatrix( int* instance,
			int matrixIndex,
			const double* inMatrix) = 0;

	virtual int updateTransitionMatrices(
			int* instance,
			int instanceCount,
			int eigenIndex,
			const int* probabilityIndices,
			const int* firstDerivativeIndices,
			const int* secondDervativeIndices,
			const double* edgeLengths,
			int count) = 0;

	virtual int updatePartials(
			int* instance,
			int instanceCount,
			int* operations,
			int operationCount,
			int rescale) = 0;

	virtual int calculateRootLogLikelihoods(
			int* instance,
			int instanceCount,
			const int* bufferIndices,
			int count,
			const double* weights,
			const double** stateFrequencies,
			double* outLogLikelihoods) = 0;

	virtual int calculateEdgeLogLikelihoods(
			int* instance,
			int instanceCount,
			const int* parentBufferIndices,
			const int* childBufferIndices,
			const int* probabilityIndices,
			const int* firstDerivativeIndices,
			const int* secondDerivativeIndices,
			int count,
			const double* weights,
			const double** stateFrequencies,
			double* outLogLikelihoods,
			double* outFirstDerivatives,
			double* outSecondDerivatives) = 0;
};

class BeagleImplFactory {
public:
	virtual BeagleImpl* createImpl(
			int bufferCount,
			int tipCount,
			int stateCount,
			int patternCount,
			int eigenDecompositionCount,
			int matrixCount) = 0; // pure virtual

	virtual const char* getName() = 0; // pure virtual
};

#endif // __beagle_impl__
