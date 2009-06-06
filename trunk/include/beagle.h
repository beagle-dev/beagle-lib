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

/**
 * @anchor BEAGLE_RETURN_CODES
 *
 * @brief Error return codes
 *
 * This enumerates all possible BEAGLE return codes.  Error codes are always negative.
 */
enum BeagleReturnCodes {
    NO_ERROR = 0,
    GENERAL_ERROR = -1,
    OUT_OF_MEMORY_ERROR = -2,
    UNIDENTIFIED_EXCEPTION_ERROR = -3,
    UNINITIALIZED_INSTANCE_ERROR = -4,  /**< the instance index is out of range,
                                          *  or the instance has not been created */
    OUT_OF_RANGE_ERROR = -5             /**< one of the indices specfied exceeded the range of the
                                          *  array */
};

/**
 * @anchor BEAGLE_FLAGS
 *
 * @brief Hardware and implementation capability flags
 *
 * This enumerates all possible hardware and implementation capability flags.
 * Each capability is a bit in a 'long'
 */
enum BeagleFlags {
    DOUBLE	=1<<0,  /**< Request/require double precision computation */
    SINGLE	=1<<1,  /**< Request/require single precision computation */
    ASYNCH	=1<<2,  /**< Request/require asynchronous computation */
    SYNCH	=1<<3,  /**< Request/require synchronous computation */
    CPU		=1<<16, /**< Request/require CPU */
    GPU		=1<<17, /**< Request/require GPU */
    FPGA	=1<<18, /**< Request/require FPGA */
    SSE		=1<<19, /**< Request/require SSE */
    CELL	=1<<20  /**< Request/require Cell */
};

/**
 * @brief Information about a specific instance
 */
typedef struct {
    int resourceNumber; /**< Resource upon which instance is running */
    long flags;         /**< Bit-flags that characterize the activate
                          *  capabilities of the resource for this instance */
} InstanceDetails;

/**
 * @brief Description of a hardware resource
 */
typedef struct {
    char* name; /**< Name of resource as a NULL-terminated character string */
    long flags; /**< Bit-flags of capabilities on resource */
} Resource;

/**
 * @brief List of hardware resources
 */
typedef struct {
    Resource* list; /**< Pointer list of resources */
    int length;     /**< Length of list */
} ResourceList;

/**
 * @brief
 *
 * LONG DESCRIPTION
 *
 * @return A list of resources available to the library as a ResourceList array
 */
ResourceList* getResourceList();

/**
 * @brief Create a single instance
 *
 * This function creates a single instance of the BEAGLE library and can be called
 * multiple times to create multiple data partition instances each returning a unique
 * identifier.
 *
 * @return the unique instance identifier (<0 if failed, see @ref BEAGLE_RETURN_CODES
 * "BeagleReturnCodes")
 */
int createInstance(int tipCount,            /**< Number of tip data elements (input) */
                   int partialsBufferCount, /**< Number of partials buffers to create (input) */
                   int compactBufferCount,  /**< Number of compact state representation buffers to
                                              *  create (input) */
                   int stateCount,          /**< Number of states in the continuous-time Markov
                                              *  chain (input) */
                   int patternCount,        /**< Number of site patterns to be handled by the
                                              *  instance (input) */
                   int eigenBufferCount,    /**< Number of rate matrix eigen-decomposition buffers
                                              *  to allocate (input) */
                   int matrixBufferCount,   /**< Number of rate matrix buffers (input) */
                   int* resourceList,       /**< List of potential resource on which this instance
                                              *  is allowed (input, NULL implies no restriction */
                   int resourceCount,       /**< Length of resourceList list (input) */
                   long preferenceFlags,    /**< Bit-flags indicating preferred implementation
                                              *  charactertistics, see BeagleFlags (input) */
                   long requirementFlags    /**< Bit-flags indicating required implementation
                                              *  characteristics, see BeagleFlags (input) */
                   );

/**
 * @brief Initialize the instance
 *
 * This function initializes the instance by selecting the hardware upon this instance will run,
 * allocating memory and populating this memory of values that may have been set.
 *
 * @returns Information about the implementation and hardware on which this instance will run
 */
int initializeInstance(int instance,                /**< Instance number to initialize (input) */
                       InstanceDetails* returnInfo  /**< Pointer to return hardware details */
                       );

/**
 * @brief Finalize this instance
 *
 * This function finalizes the instance by releasing allocated memory
 *
 * @return error code
 */
int finalize(int instance   /**< Instance number */
             );

/**
 * @brief Set an instance partials buffer
 *
 * This function copies an array of partials into an instance buffer.
 *
 * @return error code
 */
int setPartials(int instance,               /**< Instance number in which to set a partialsBuffer
                                              *  (input) */
                int bufferIndex,            /**< Index of destination partialsBuffer (input) */
                const double* inPartials    /**< Pointer to partials values to set (input) */
                );

/**
 * @brief Get partials from an instance buffer
 *
 * This function copies an instance buffer into the array outPartials
 *
 * @return error code
 */
int getPartials(int instance,       /**< Instance number from which to get partialsBuffer
                                      *  (input) */
                int bufferIndex,    /**< Index of source partialsBuffer (input) */
                double* outPartials /**< Pointer to which to receive partialsBuffer (output) */
                );

/**
 * @brief Set the compressed state representation for tip node
 *
 * This function copies a compressed state representation into a instance buffer.
 * Compressed state representation is an array of states: 0 to stateCount - 1 (missing = stateCount)
 *
 * @return error code
 */
int setTipStates(int instance,          /**< Instance number (input) */
                 int tipIndex,          /**< Index of destination compressedBuffer (input) */
                 const int* inStates    /**< Pointer to compressed states (input) */
                 );

/**
 * @brief Set an eigen-decomposition buffer
 *
 * This function copies an eigen-decomposition into a instance buffer.
 * 
 * @return error code
 */
int setEigenDecomposition(int instance,                         /**< Instance number (input) */
                          int eigenIndex,                       /**< Index of eigen-decomposition
                                                                  *  buffer (input) */
                          const double* inEigenVectors,         /**< Flattened matrix (stateCount x
                                                                  *  stateCount) of eigen-vectors
                                                                  *  (input) */
                          const double* inInverseEigenVectors,  /**< Flattened matrix (stateCount x
                                                                  *  stateCount) of inverse-eigen-
                                                                  *  vectors (input) */
                          const double* inEigenValues           /**< Vector of eigenvalues */
                          );

/**
 * @brief Set a finite-time transition probability matrix
 *
 * This function copies a finite-time transition probability matrix into a matrix buffer.
 *
 * @return error code
 */
int setTransitionMatrix(int instance,           /**< Instance number (input)  */
                        int matrixIndex,        /**< Index of matrix buffer (input) */
                        const double* inMatrix  /**< Pointer to source transition
                                                  *  probability matrix (input) */
                        );

/**
 * @brief Calculate a list of transition probability matrices
 *
 * This function calculates a list of transition probabilities matrices and their first and
 * second derivatives (if requested).
 *
 * @return error code
 */
int updateTransitionMatrices(int instance,                      /**< Instance number (input) */
                             int eigenIndex,                    /**< Index of eigen-decomposition
                                                                  *  buffer (input) */
                             const int* probabilityIndices,     /**< List of indices of transition
                                                                  *  probability matrices to update
                                                                  *  (input) */
                             const int* firstDerivativeIndices, /**< List of indices of first
                                                                  *  derivative matrices to update
                                                                  *  (input, NULL implies no
                                                                  *  calculation) */
                             const int* secondDervativeIndices, /**< List of indices of second
                                                                  *  derivative matrices to update
                                                                  *  (input, NULL implies no
                                                                  *  calculation) */
                             const double* edgeLengths,         /**< List of edge lengths with which
                                                                  *  to perform calculations
                                                                  *  (input) */
                             int count                          /**< Length of lists */
                             );

/**
 * @brief Calculate or queue for calculation partials using a list of operations
 *
 * This function either calculates or queues for calculation a list partials. Implementations
 * supporting SYNCH may queue these calculations while other implementations perform these
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
int updatePartials(const int* instance,     /**< List of instances for which to update partials
                                              *  buffers (input) */
                   int instanceCount,       /**< Length of instance list (input) */
                   const int* operations,   /**< List of 5-tuples specifying operations (input) */
                   int operationCount,      /**< Number of operations (input) */
                   int rescale              /**< Specify whether (=1) or not (=0) to recalculate
                                              *  scaling factors */
                   );

/**
* @brief Block until all calculations that write to the specified partials have completed.
*
* This function is optional and only has to be called by clients that "recycle" partials.
*
* If used, this function must be called after an updatePartials call and must refer to
* indices of "destinationPartials" that were used in a previous updatePartials
* call.  The library will block until those partials have been calculated.
*
* @return error code
*/
int waitForPartials(const int* instance,            /**< List of instances for which to update
                                                      *  partials buffers (input) */
                    int instanceCount,              /**< Length of instance list (input) */
                    const int* destinationPartials, /**< List of the indices of destinationPartials
                                                      *  that must be calculated before the function
                                                      *  returns */
                    int destinationPartialsCount    /**< Number of destinationPartials (input) */
                    );

/**
 * @brief Calculate site log likelihoods at a root node
 *
 * This function integrates a list of partials at a node with respect to a set of partials-weights
 * and state frequencies to return the log likelihoods for each site
 *
 * @return error code
 */
int calculateRootLogLikelihoods(int instance,                   /**< Instance number (input) */
                                const int* bufferIndices,       /**< List of partialsBuffer
                                                                  *  indices to integrate (input) */
                                const double* weights,          /**< List of weights to apply to
                                                                  *  each partialsBuffer (input) */
                                const double* stateFrequencies, /**< List of state frequencies for
                                                                  *  each partialsBuffer (input).
                                                                  *  There should be one set for
                                                                  *  each of parentBufferIndices */
                                int count,                      /**< Number of partialsBuffer to
                                                                  *  integrate (input) */
                                double* outLogLikelihoods       /**< Pointer to destination for
                                                                  *  resulting log likelihoods
                                                                  *  (output) */
                                );

/**
 * @brief Calculate site log likelihoods and derivatives along an edge
 *
 * This function integrates at list of partials at a parent and child node with respect
 * to a set of partials-weights and state frequencies to return the log likelihoods
 * and first and second derivatives for each site
 *
 * @return error code
 */
int calculateEdgeLogLikelihoods(int instance,                       /**< Instance number (input) */
                                const int* parentBufferIndices,     /**< List of indices of parent
                                                                      *  partialsBuffers (input) */
                                const int* childBufferIndices,      /**< List of indices of child
                                                                      *  partialsBuffers (input) */
                                const int* probabilityIndices ,     /**< List indices of transition
                                                                      *  probability matrices for
                                                                      *  this edge (input) */
                                const int* firstDerivativeIndices,  /**< List indices of first
                                                                      *  derivative matrices
                                                                      *  (input) */
                                const int* secondDerivativeIndices, /**< List indices of second
                                                                      *  derivative matrices
                                                                      *  (input) */
                                const double* weights,              /**< List of weights to apply to
                                                                      *  each partialsBuffer
                                                                      *  (input) */
                                const double* stateFrequencies,     /**< List of state frequencies
                                                                      *  for each partialsBuffer
                                                                      *  (input).  There should be
                                                                      *  one set for each of
                                                                      *  parentBufferIndices */
                                int count,                          /**< Number of partialsBuffers
                                                                      *  (input) */
                                double* outLogLikelihoods,          /**< Pointer to destination for
                                                                      *  resulting log likelihoods
                                                                      *  (output) */
                                double* outFirstDerivatives,        /**< Pointer to destination for
                                                                      *  resulting first derivatives
                                                                      *  (output) */
                                double* outSecondDerivatives        /**< Pointer to destination for
                                                                      *  resulting second
                                                                      *  derivatives (output) */
                                );

#endif // __beagle__
