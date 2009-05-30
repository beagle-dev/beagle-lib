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
	virtual ~BeagleCPUImpl();
	// initialization of instance,  returnInfo can be null				
	int initialize(	int bufferCount,
					int tipCount,
					int stateCount,
					int patternCount,
					int eigenDecompositionCount,
					int matrixCount);			

	// set the partials for a given tip
	//
	// tipIndex the index of the tip
	// inPartials the array of partials, stateCount x patternCount
	int setPartials(int bufferIndex, const double* inPartials);

	int getPartials(int bufferIndex, double *outPartials);

	// set the states for a given tip
	//
	// tipIndex the index of the tip
	// inStates the array of states: 0 to stateCount - 1, missing = stateCount
	int setTipStates(int tipIndex, const int* inStates);

	// set the vector of state frequencies
	//
	// stateFrequencies an array containing the state frequencies
	int setStateFrequencies(int* instance, const double* inStateFrequencies);
	
	// sets the Eigen decomposition for a given matrix
	//
	// matrixIndex the matrix index to update
	// eigenVectors an array containing the Eigen Vectors
	// inverseEigenVectors an array containing the inverse Eigen Vectors
	// eigenValues an array containing the Eigen Values
	int setEigenDecomposition(	int eigenIndex,
							  	const double** inEigenVectors,
							  	const double** inInverseEigenVectors,
						 		const double* inEigenValues);
	
	int setTransitionMatrix(int matrixIndex, const double* inMatrix);
																					   
	
	// calculate a transition probability matrices for a given list of node. This will
	// calculate for all categories (and all matrices if more than one is being used).
	//
	// nodeIndices an array of node indices that require transition probability matrices
	// edgeLengths an array of expected lengths in substitutions per site
	// count the number of elements in the above arrays
	int updateTransitionMatrices(	int eigenIndex,
									const int* probabilityIndices,
									const int* firstDerivativeIndices,
									const int* secondDervativeIndices,
									const double* edgeLengths,
									int count);                                                   
	
	// calculate or queue for calculation partials using an array of operations
	//
	// operations an array of triplets of indices: the two source partials and the destination
	// dependencies an array of indices specify which operations are dependent on which (optional)
	// count the number of operations
	// rescale indicate if partials should be rescaled during peeling
	int updatePartials(	int* operations,					
					   int operationCount,
					   int rescale);
	
	// calculate the site log likelihoods at a particular node
	//
	// rootNodeIndex the index of the root
	// outLogLikelihoods an array into which the site log likelihoods will be put
	int calculateRootLogLikelihoods(const int* bufferIndices,
									int count,
									const double* weights,
									const double** stateFrequencies,		                     
									double* outLogLikelihoods);
	
	// possible nulls: firstDerivativeIndices, secondDerivativeIndices,
	//                 outFirstDerivatives, outSecondDerivatives 
	int calculateEdgeLogLikelihoods(
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
								 double* outSecondDerivatives);

	private:

    void updateStatesStates(int nodeIndex1, int nodeIndex2, int nodeIndex3);
    void updateStatesPartials(int nodeIndex1, int nodeIndex2, int nodeIndex3);
    void updatePartialsPartials(int nodeIndex1, int nodeIndex2, int nodeIndex3);


};

class BeagleCPUImplFactory : public BeagleImplFactory {
    public:
    	virtual BeagleImpl* createImpl(
								int bufferCount,
								int tipCount,
								int stateCount,
								int patternCount,
								int eigenDecompositionCount,
								int matrixCount);

    	virtual const char* getName();
};
