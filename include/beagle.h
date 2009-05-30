/**
 * @file	beagle.h
 *
 * @brief This file documents the API as well as header for the
 * Broad-platform Evolutionary Analysis General Likelihood Evaluator
 *
 * OVERVIEW
 *
 * DEFINITION OF KEY CONCEPTS: INSTANCE, BUFFER, etc.
 *
 * @author Likelihood API Working Group
 *
 * @author Daniel Ayres
 * @author Adam Bazinet
 * @author Peter Beerli
 * @author Michael Cummings
 * @author Mark Holder
 * @author John Huelsenbeck
 * @author Paul Lewis
 * @author Michael Ott
 * @author Andrew Rambaut
 * @author Marc Suchard
 * @author David Swofford
 * @author Derrick Zwickl
 *
 */

#ifndef __beagle__
#define __beagle__

enum BeagleReturnCodes {
	NO_ERROR = 0,
	GENERAL_ERROR = -1,
	OUT_OF_MEMORY_ERROR = -2
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
 * @brief Create a single instance
 *
 * This function creates a single instance of the BEAGLE library and can be called
 * multiple times to create multiple data partition instances each returning a unique
 * identifier.
 *
 * @return the unique instance identifier (<0 if failed, see BeagleReturnCodes)
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
					const double* inPartials);	/**< Pointer to partials values to set (input) */

int getPartials(
		int instance,			/**< Instance number from which to get partialsBuffer (input) */
		int bufferIndex, 		/**< Index of source partialsBuffer (input) */
		double *outPartials		/**< Pointer to which to receive partialsBuffer (output) */
		);

// set the states for a given tip
//
// tipIndex the index of the tip
// inStates the array of states: 0 to stateCount - 1, missing = stateCount
int setTipStates(
                  int instance,			/**< Instance number (input) */
				  int tipIndex,			/**< Index of destination compressedBuffer (input) */
				  const int* inStates); /**< Pointer to compressed states (input) */



// sets the Eigen decomposition for a given matrix
//
// matrixIndex the matrix index to update
// eigenVectors an array containing the Eigen Vectors
// inverseEigenVectors an array containing the inverse Eigen Vectors
// eigenValues an array containing the Eigen Values
int setEigenDecomposition(
                           int instance,							/**< Instance number (input) */
						   int eigenIndex,							/**< Index of eigen-decomposition buffer (input) */
						   const double** inEigenVectors, 			/**< 2D matrix of eigen-vectors (input) */
						   const double** inInverseEigenVectors,	/**< 2D matrix of inverse-eigen-vectors (input) */
						   const double* inEigenValues); 			/**< 2D vector of eigenvalues*/

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
                                            int instance,	/**< Instance number (input) */
                                            int eigenIndex,	/**<  Index of eigen-decomposition buffer (input) */
                                            const int* probabilityIndices, /**<  List of indices of transition probability matrices to update (input) */
                                            const int* firstDerivativeIndices, /**< List of indices of first derivative matrices to update (input, NULL implies no calculation) */
                                            const int* secondDervativeIndices, /**< List of indices of second derivative matrices to update (input, NULL implies no calculation) */
                                            const double* edgeLengths, /**< List of edge lengths with which to perform calculations (input) */
                                            int count); /**< Length of lists */

/**
 * @brief Calculate or queue for calculation partials using a list of operations
 *
 * This function either calculates or queues for calculation a list partials. Implementations
 * supporting SYCH may queue these calculations while other implementations perform these
 * operations immediately.  Implementations supporting GPU may perform all operations in the list
 * simultaneously.
 *
 * Operations list is a list of 5-tuple integer indices, with one 5-tuple per operation.
 * Format of 5-tuple operation: {destinationPartials,
 *                               child1Partials,
 *                               child1TransitionMatrix,
 *                               child2Partials,
 *                               child2TransitionMatrix}
 *
 * @return error code
 */
int updatePartials(
                       int* instance, 		/**< List of instances for which to update partials buffers (input) */
                       int instanceCount, 	/**< Length of instance list (input) */
					   int* operations, 	/**< List of 5-tuples specifying operations (input) */
					   int operationCount, 	/**< Number of operations (input) */
					   int rescale); 		/**< Specify whether (=1) or not (=0) to recalculate scaling factors */

/**
 * @brief Calculate site log likelihoods at a root node
 *
 * This function integrates a list of partials at a node with respect to a set of partials-weights and
 * state frequencies to return the log likelihoods for each site
 *
 * @return error code
 */
int calculateRootLogLikelihoods(
                             int instance, /**< Instance number (input) */
		                     const int* bufferIndices, /**< List of partialsBuffer indices to integrate (input) */
		                     const double* weights, /**< List of weights to apply to each partialsBuffer (input) */
		                     const double** stateFrequencies, /**< List of state frequencies for each partialsBuffer (input)
															   * If list length is one, the same state frequencies are used
															   * for each partialsBuffer
															   */
		                     int count, /*< Number of partialsBuffer to integrate (input) */
			                 double* outLogLikelihoods); /**< Pointer to destination for resulting log likelihoods (output) */

/*
 * @brief Calculate site log likelihoods and derivatives along an edge
 *
 * This function integrates at list of partials at a parent and child node with respect
 * to a set of partials-weights and state frequencies to return the log likelihoods
 * and first and second derivatives for each site
 *
 * @return error code
 */
int calculateEdgeLogLikelihoods(
							 int instance, /**< Instance number (input) */
		                     const int* parentBufferIndices, /**< List of indices of parent partialsBuffers (input) */
		                     const int* childBufferIndices, /**< List of indices of child partialsBuffers (input) */
		                     const int* probabilityIndices , /**< List indices of transition probability matrices for this edge (input) */
		                     const int* firstDerivativeIndices, /**< List indices of first derivative matrices (input) */
		                     const int* secondDerivativeIndices, /**< List indices of second derivative matrices (input) */
		                     const double* weights, /**< List of weights to apply to each partialsBuffer (input) */
		                     const double** stateFrequencies, /**< List of state frequencies for each partialsBuffer (input)
															   * If list length is one, the same state frequencies are used
															   * for each partialsBuffer
															   */
		                     int count, /**< Number of partialsBuffers (input) */
		                     double* outLogLikelihoods, /**< Pointer to destination for resulting log likelihoods (output) */
			                 double* outFirstDerivatives, /**< Pointer to destination for resulting first derivatives (output) */
			                 double* outSecondDerivatives); /**< Pointer to destination for resulting second derivatives (output) */

#endif // __beagle__



