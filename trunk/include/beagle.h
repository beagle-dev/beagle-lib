/*
 *  beagle.h
 *  BEAGLE
 *
 * @author Likelihood API Working Group
 *
 */

#ifndef __beagle__
#define __beagle__

enum BeagleFlags {
	DOUBLE	=1<<0,
	SINGLE	=1<<1,
	ASYNCH	=1<<2,
	SYNCH	=1<<3,
	
	CPU		=1<<16, 
	GPU		=1<<17, 
	FPGA	=1<<18,
	SSE		=1<<19,
	CELL	=1<<20
}


typedef struct {
	int resourceNumber,
	int flags,	
} InstanceDetails;

typedef struct {
	char* name;
	long flag;
} Resource;

typedef struct {
	Resource* list;
	int length;
} ResourceList;



// returns a list of computing resources
ResourceList* getResourceList();

// Create a single instance
// This can be called multiple times to create multiple data partition instances
// each returning a unique identifier.
//
// nodeCount the number of nodes in the tree
// tipCount the number of tips in the tree
// stateCount the number of states
// patternCount the number of site patterns
//
// returns the unique instance identifier (-1 if failed)
int createInstance(
				int bufferCount,
				int tipCount,
				int stateCount,
				int patternCount,
				int eigenDecompositionCount,
				int matrixCount,
				int* resourceList,
				int resourceCount,
				int preferenceFlags,
				int requirementFlags,				
				);

// initialization of instance,  returnInfo can be null				
int initializeInstance(
						int *instance, 
						int instanceCount,
						InstanceDetails* returnInfo);			

// finalize and dispose of memory allocation if needed
int finalize(int *instance, int instanceCount);

// set the partials for a given tip
//
// tipIndex the index of the tip
// inPartials the array of partials, stateCount x patternCount
int setPartials(
                    int* instance,
                    int instanceCount,
					int bufferIndex,
					const double* inPartials);

int getPartials(int* instance, int bufferIndex, double *outPartials);

// set the states for a given tip
//
// tipIndex the index of the tip
// inStates the array of states: 0 to stateCount - 1, missing = stateCount
int setTipStates(
                  int* instance,
                  int instanceCount,
				  int tipIndex,
				  const int* inStates);

// set the vector of state frequencies
//
// stateFrequencies an array containing the state frequencies
int setStateFrequencies(
                         int* instance,
                         const double* inStateFrequencies);

// sets the Eigen decomposition for a given matrix
//
// matrixIndex the matrix index to update
// eigenVectors an array containing the Eigen Vectors
// inverseEigenVectors an array containing the inverse Eigen Vectors
// eigenValues an array containing the Eigen Values
int setEigenDecomposition(
                           int* instance,
                           int instanceCount,
						   int eigenIndex,
						   const double** inEigenVectors,
						   const double** inInverseEigenVectors,
						   const double* inEigenValues);

int setTransitionMatrix(	int* instance,
                			int matrixIndex,
                			const double* inMatrix);
                                                                                   

// calculate a transition probability matrices for a given list of node. This will
// calculate for all categories (and all matrices if more than one is being used).
//
// nodeIndices an array of node indices that require transition probability matrices
// edgeLengths an array of expected lengths in substitutions per site
// count the number of elements in the above arrays
int updateTransitionMatrices(
                                            int* instance,
                                            int instanceCount,
                                            int eigenIndex,
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
int updatePartials(
                       int* instance,
                       int instanceCount,
					   int* operations,					
					   int operationCount,
					   int rescale);

// calculate the site log likelihoods at a particular node
//
// rootNodeIndex the index of the root
// outLogLikelihoods an array into which the site log likelihoods will be put
int calculateRootLogLikelihoods(
                             int* instance,
                             int instanceCount,
		                     const int* bufferIndices,
		                     int count,
		                     const double* weights,
		                     const double** stateFrequencies,		                     
			                 double* outLogLikelihoods);

// possible nulls: firstDerivativeIndices, secondDerivativeIndices,
//                 outFirstDerivatives, outSecondDerivatives 
int calculateEdgeLogLikelihoods(
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
			                 double* outSecondDerivatives);

#endif // __beagle__


queuing and asychronis (sp?) calls.