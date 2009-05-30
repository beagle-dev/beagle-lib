/**
 * @file	beagle.h
 *
 * @brief This file documents the API as well as header for the Broad-platform Evolutionary Analysis General Likelihood Evaluator
 *
 * LONG COMMENTS HERE
 *
 * OVERVIEW:
 *
 * KEY ELEMENTS:  INSTANCE, BUFFERS, etc.
 *
 * @author Likelihood API Working Group
 *
 */

#ifndef __beagle__
#define __beagle__

enum BeagleReturnCodes {
	NO_ERROR = 0,
	OUT_OF_MEMORY_ERROR
};

enum BeagleFlags {
	DOUBLE	=1<<0, /**< Request/require double precision computation */
	SINGLE	=1<<1, /**< same */
	ASYNCH	=1<<2,
	SYNCH	=1<<3,
	CPU		=1<<16,
	GPU		=1<<17,
	FPGA	=1<<18,
	SSE		=1<<19,
	CELL	=1<<20
};


/**
 * @brief Structure includes information about a specific instance
 *
 * LONG DESCRIPTION
 *
 */
typedef struct {
	int resourceNumber; /**< Resource upon which instance is running */
	int flags; 			/**< Bit-flags that characterize this instance's resource */
} InstanceDetails;

typedef struct {
	char* name;
	long flag;
} Resource;

typedef struct {
	Resource* list;
	int length;
} ResourceList;


/**
 * @brief
 *
 * LONG DESCRIPTION
 *
 * @return A list of resources available to the library as a ResourceList array
 */
// returns a list of computing resources
ResourceList* getResourceList();

/**
 * Create a single instance
 * This can be called multiple times to create multiple data partition instances
 * each returning a unique identifier.
 *
 * nodeCount the number of nodes in the tree
 * tipCount the number of tips in the tree
 * stateCount the number of states
 * patternCount the number of site patterns
 *
 *
 *
 * @return the unique instance identifier (-1 if failed)
 *
 */
int createInstance(
			    int tipCount,				/**< Number of tip data elements (input) */
				int partialsBufferCount,	/**< Number of partials buffers to create (input) */
				int compactBufferCount,		/**< Number of compact state representation buffers to create (input) */
				int stateCount,				/**< Number of states in the continuous-time Markov chain (input) */
				int patternCount,			/**< Number of site patterns to be handled by the instance (input) */
				int eigenBufferCount,		/**< Number of rate matrix eigen-decomposition buffers to allocate (input) */
				int matrixBufferCount,		/**< Number of rate matrix buffers (input) */
				int* resourceList,			/**< List of potential resource on which this instance is allowed (input, NULL implies no restriction */
				int resourceCount,			/**< Length of resourceList list (input) */
				int preferenceFlags,		/**< Bit-flags indicating preferred implementation charactertistics, see BeagleFlags (input) */
				int requirementFlags		/**< Bit-flags indicating required implementation characteristics, see BeagleFlags (input) */
				);

// initialization of instance,  returnInfo can be null
int initializeInstance(
						int instance,		/**< Instance number to initialize (input) */
						InstanceDetails* returnInfo);


// finalize and dispose of memory allocation if needed
int finalize(int *instance, int instanceCount);

// set the partials for a given tip
//
// tipIndex the index of the tip
// inPartials the array of partials, stateCount x patternCount
int setPartials(
                    int instance,				/**< Instance number in which to set a partialsBuffer (input) */
             		int bufferIndex,			/**< Index of destination partialsBuffer (input) */
					const double* inPartials);	/**< Partials values to set (input) */

int getPartials(
		int instance,	/**< Instance number from which to get */
		int bufferIndex,
		double *outPartials);

// set the states for a given tip
//
// tipIndex the index of the tip
// inStates the array of states: 0 to stateCount - 1, missing = stateCount
int setTipStates(
                  int instance,
				  int tipIndex,
				  const int* inStates);


// sets the Eigen decomposition for a given matrix
//
// matrixIndex the matrix index to update
// eigenVectors an array containing the Eigen Vectors
// inverseEigenVectors an array containing the inverse Eigen Vectors
// eigenValues an array containing the Eigen Values
int setEigenDecomposition(
                           int instance,
						   int eigenIndex,
						   const double** inEigenVectors,
						   const double** inInverseEigenVectors,
						   const double* inEigenValues);

int setTransitionMatrix(	int instance,
                			int matrixIndex,
                			const double* inMatrix);


// calculate a transition probability matrices for a given list of node. This will
// calculate for all categories (and all matrices if more than one is being used).
//
// nodeIndices an array of node indices that require transition probability matrices
// edgeLengths an array of expected lengths in substitutions per site
// count the number of elements in the above arrays
int updateTransitionMatrices(
                                            int instance,
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
/**
 * @brief Update or enqueue to update partialsBuffer
 *
 * LONG DESCRIPTION
 *
 * Format of operations list: {destinationPartialsIndex,
 *                             child1PartialsIndex
 *                             child1TransitionMatrixIndex,
 *                             child2PartialsIndex,
 *                             child2TransitionMatrixIndex}
 *
 */
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
                             int instance,
		                     const int* bufferIndices,
		                     int count,
		                     const double* weights,
		                     const double** stateFrequencies,
			                 double* outLogLikelihoods);

// possible nulls: firstDerivativeIndices, secondDerivativeIndices,
//                 outFirstDerivatives, outSecondDerivatives
int calculateEdgeLogLikelihoods(
							 int instance,
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



