/**
 * @file beagle.h
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
 * @author Aaron Darling
 * @author Mark Holder
 * @author John Huelsenbeck
 * @author Paul Lewis
 * @author Michael Ott
 * @author Andrew Rambaut
 * @author Fredrik Ronquist
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
    BEAGLE_SUCCESS                      = 0,   /**< Success */
    BEAGLE_ERROR_GENERAL                = -1,  /**< Unspecified error */
    BEAGLE_ERROR_OUT_OF_MEMORY          = -2,  /**< Not enough memory could be allocated */
    BEAGLE_ERROR_UNIDENTIFIED_EXCEPTION = -3,  /**< Unspecified exception */
    BEAGLE_ERROR_UNINITIALIZED_INSTANCE = -4,  /**< The instance index is out of range,
                                                *   or the instance has not been created */
    BEAGLE_ERROR_OUT_OF_RANGE           = -5   /**< One of the indices specified exceeded the range of the
                                                *   array */
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
    BEAGLE_FLAG_DOUBLE = 1 << 0,    /**< double precision computation */
    BEAGLE_FLAG_SINGLE = 1 << 1,    /**< single precision computation */
    BEAGLE_FLAG_ASYNCH = 1 << 2,    /**< asynchronous computation */
    BEAGLE_FLAG_SYNCH  = 1 << 3,    /**< synchronous computation */
    BEAGLE_FLAG_CPU    = 1 << 16,   /**< CPU */
    BEAGLE_FLAG_GPU    = 1 << 17,   /**< GPU */
    BEAGLE_FLAG_FPGA   = 1 << 18,   /**< FPGA */
    BEAGLE_FLAG_SSE    = 1 << 19,   /**< SSE */
    BEAGLE_FLAG_CELL   = 1 << 20    /**< Cell */
};

/**
 * @anchor BEAGLE_OP_CODES
 *
 * @brief Operation codes
 *
 * This enumerates all possible BEAGLE operation codes.
 */
enum BeagleOpCodes {
	BEAGLE_OP_COUNT    = 7,	/**< Total number of integers per beagleUpdatePartials operation */
	BEAGLE_OP_NONE     = -1	/**< Specify no use for indexed buffer */
};

/**
 * @brief Information about a specific instance
 */
typedef struct {
    int resourceNumber; /**< Resource upon which instance is running */
    long flags;         /**< Bit-flags that characterize the activate
                         *   capabilities of the resource for this instance */
} BeagleInstanceDetails;

/**
 * @brief Description of a hardware resource
 */
typedef struct {
    char* name;         /**< Name of resource as a NULL-terminated character string */
    char* description;  /**< Description of resource as a NULL-terminated character string */
    long flags;         /**< Bit-flags of capabilities on resource */
} BeagleResource;

/**
 * @brief List of hardware resources
 */
typedef struct {
    BeagleResource* list; /**< Pointer list of resources */
    int length;     /**< Length of list */
} BeagleResourceList;

/* using C calling conventions so that C programs can successfully link the beagle library
 * (brace is closed at the end of this file)
 */
#ifdef __cplusplus
extern "C" {
#endif
    
/**
 * @brief
 *
 * LONG DESCRIPTION
 *
 * @return A list of resources available to the library as a ResourceList array
 */
BeagleResourceList* beagleGetResourceList();

/**
 * @brief Create a single instance
 *
 * This function creates a single instance of the BEAGLE library and can be called
 * multiple times to create multiple data partition instances each returning a unique
 * identifier.
 *
 * @param tipCount              Number of tip data elements (input)
 * @param partialsBufferCount   Number of partials buffers to create (input)
 * @param compactBufferCount    Number of compact state representation buffers to create (input)
 * @param stateCount            Number of states in the continuous-time Markov chain (input)
 * @param patternCount          Number of site patterns to be handled by the instance (input)
 * @param eigenBufferCount      Number of rate matrix eigen-decomposition buffers to allocate
 *                               (input)
 * @param matrixBufferCount     Number of rate matrix buffers (input)
 * @param categoryCount         Number of rate categories (input)
 * @param scaleBufferCount		Number of scale buffers to create (input)
 * @param resourceList          List of potential resources on which this instance is allowed
 *                               (input, NULL implies no restriction)
 * @param resourceCount         Length of resourceList list (input)
 * @param preferenceFlags       Bit-flags indicating preferred implementation charactertistics,
 *                               see BeagleFlags (input)
 * @param requirementFlags      Bit-flags indicating required implementation characteristics,
 *                               see BeagleFlags (input)
 *
 * @return the unique instance identifier (<0 if failed, see @ref BEAGLE_RETURN_CODES
 * "BeagleReturnCodes")
 */
int beagleCreateInstance(int tipCount,
                         int partialsBufferCount,
                         int compactBufferCount,
                         int stateCount,
                         int patternCount,
                         int eigenBufferCount,
                         int matrixBufferCount,
                         int categoryCount,
                         int scaleBufferCount,
                         int* resourceList,
                         int resourceCount,
                         long preferenceFlags,
                         long requirementFlags);

/**
 * @brief Initialize the instance
 *
 * This function initializes the instance by selecting the hardware upon this instance will run,
 * allocating memory and populating this memory of values that may have been set.
 *
 * @param instance      Instance number to initialize (input)
 * @param returnInfo    Pointer to return hardware details
 *
 * @returns Information about the implementation and hardware on which this instance will run
 */
int beagleInitializeInstance(int instance,
                             BeagleInstanceDetails* returnInfo);

/**
 * @brief Finalize this instance
 *
 * This function finalizes the instance by releasing allocated memory
 *
 * @param instance  Instance number
 *
 * @return error code
 */
int beagleFinalizeInstance(int instance);

/**
 * @brief Set the compact state representation for tip node
 *
 * This function copies a compact state representation into an instance buffer.
 * Compact state representation is an array of states: 0 to stateCount - 1 (missing = stateCount).
 * The inStates array should be patternCount in length (replication across categoryCount is not
 * required).
 *
 * @param instance  Instance number (input)
 * @param tipIndex  Index of destination compactBuffer (input)
 * @param inStates  Pointer to compact states (input)
 *
 * @return error code
 */
int beagleSetTipStates(int instance,
                       int tipIndex,
                       const int* inStates);

/**
 * @brief Set an instance partials buffer for tip node
 *
 * This function copies an array of partials into an instance buffer. The inPartials array should
 * be stateCount * patternCount in length. For most applications this will be used
 * to set the partial likelihoods for the observed states. Internally, the partials will be copied
 * categoryCount times.
 *
 * @param instance      Instance number in which to set a partialsBuffer (input)
 * @param tipIndex      Index of destination partialsBuffer (input)
 * @param inPartials    Pointer to partials values to set (input)
 *
 * @return error code
 */
int beagleSetTipPartials(int instance,
                         int tipIndex,
                         const double* inPartials);

/**
 * @brief Set an instance partials buffer
 *
 * This function copies an array of partials into an instance buffer. The inPartials array should
 * be stateCount * patternCount * categoryCount in length. 
 *
 * @param instance      Instance number in which to set a partialsBuffer (input)
 * @param bufferIndex   Index of destination partialsBuffer (input)
 * @param inPartials    Pointer to partials values to set (input)
 *
 * @return error code
 */
int beagleSetPartials(int instance,
                      int bufferIndex,
                      const double* inPartials);

/**
 * @brief Get partials from an instance buffer
 *
 * This function copies an instance buffer into the array outPartials. The outPartials array should
 * be stateCount * patternCount * categoryCount in length.
 *
 * @param instance      Instance number from which to get partialsBuffer (input)
 * @param bufferIndex   Index of source partialsBuffer (input)
 * @param scaleIndex  	Index of scaleBuffer to apply to partialsBuffer (input)
 * @param outPartials   Pointer to which to receive partialsBuffer (output)
 *
 * @return error code
 */
int beagleGetPartials(int instance,
                      int bufferIndex,
                      int scaleIndex,
                      double* outPartials);

/**
 * @brief Set an eigen-decomposition buffer
 *
 * This function copies an eigen-decomposition into an instance buffer.
 *
 * @param instance              Instance number (input)
 * @param eigenIndex            Index of eigen-decomposition buffer (input)
 * @param inEigenVectors        Flattened matrix (stateCount x stateCount) of eigen-vectors (input)
 * @param inInverseEigenVectors Flattened matrix (stateCount x stateCount) of inverse-eigen- vectors
 *                               (input)
 * @param inEigenValues         Vector of eigenvalues
 *
 * @return error code
 */
int beagleSetEigenDecomposition(int instance,
                                int eigenIndex,
                                const double* inEigenVectors,
                                const double* inInverseEigenVectors,
                                const double* inEigenValues);

/**
 * @brief Set category rates
 *
 * This function sets the vector of category rates for an instance.
 *
 * @param instance              Instance number (input)
 * @param inCategoryRates       Array containing categoryCount rate scalers (input)
 *
 * @return error code
 */
int beagleSetCategoryRates(int instance,
                           const double* inCategoryRates);

/**
 * @brief Calculate a list of transition probability matrices
 *
 * This function calculates a list of transition probabilities matrices and their first and
 * second derivatives (if requested).
 *
 * @param instance                  Instance number (input)
 * @param eigenIndex                Index of eigen-decomposition buffer (input)
 * @param probabilityIndices        List of indices of transition probability matrices to update
 *                                   (input)
 * @param firstDerivativeIndices    List of indices of first derivative matrices to update
 *                                   (input, NULL implies no calculation)
 * @param secondDervativeIndices    List of indices of second derivative matrices to update
 *                                   (input, NULL implies no calculation)
 * @param edgeLengths               List of edge lengths with which to perform calculations (input)
 * @param count                     Length of lists
 *
 * @return error code
 */
int beagleUpdateTransitionMatrices(int instance,
                                   int eigenIndex,
                                   const int* probabilityIndices,
                                   const int* firstDerivativeIndices,
                                   const int* secondDervativeIndices,
                                   const double* edgeLengths,
                                   int count);

/**
 * @brief Set a finite-time transition probability matrix
 *
 * This function copies a finite-time transition probability matrix into a matrix buffer. This function
 * is used when the application wishes to explicitly set the transition probability matrix rather than
 * using the beagleSetEigenDecomposition and beagleUpdateTransitionMatrices functions. The inMatrix array should be
 * of size stateCount * stateCount * categoryCount and will contain one matrix for each rate category.
 *
 * @param instance      Instance number (input)
 * @param matrixIndex   Index of matrix buffer (input)
 * @param inMatrix      Pointer to source transition probability matrix (input)
 *
 * @return error code
 */
int beagleSetTransitionMatrix(int instance,
                              int matrixIndex,
                              const double* inMatrix);

/**
 * @brief Calculate or queue for calculation partials using a list of operations
 *
 * This function either calculates or queues for calculation a list partials. Implementations
 * supporting SYNCH may queue these calculations while other implementations perform these
 * operations immediately.  Implementations supporting GPU may perform all operations in the list
 * simultaneously.
 *
 * Operations list is a list of 7-tuple integer indices, with one 7-tuple per operation.
 * Format of 7-tuple operation: {destinationPartials,
 *                               destinationScaleWrite,
 *                               destinationScaleRead,
 *                               child1Partials,
 *                               child1TransitionMatrix,
 *                               child2Partials,
 *                               child2TransitionMatrix}
 *
 * @param instance                  List of instances for which to update partials buffers (input)
 * @param instanceCount             Length of instance list (input)
 * @param operations                List of 7-tuples specifying operations (input)
 * @param operationCount            Number of operations (input)
 * @param cumulativeScaleIndex   	Index number of scaleBuffer to store accumulated factors (input)
 *
 * @return error code
 */
int beagleUpdatePartials(const int* instance,
                         int instanceCount,
                         const int* operations,
                         int operationCount,
                         int cumulativeScaleIndex);

/**
 * @brief Block until all calculations that write to the specified partials have completed.
 *
 * This function is optional and only has to be called by clients that "recycle" partials.
 *
 * If used, this function must be called after an beagleUpdatePartials call and must refer to
 * indices of "destinationPartials" that were used in a previous beagleUpdatePartials
 * call.  The library will block until those partials have been calculated.
 *
 * @param instance                  List of instances for which to update partials buffers (input)
 * @param instanceCount             Length of instance list (input)
 * @param destinationPartials       List of the indices of destinationPartials that must be
 *                                   calculated before the function returns
 * @param destinationPartialsCount  Number of destinationPartials (input)
 *
 * @return error code
 */
int beagleWaitForPartials(const int* instance,
                          int instanceCount,
                          const int* destinationPartials,
                          int destinationPartialsCount);

/**
 * @brief Accumulate scale factors
 *
 * This function adds (log) scale factors from a list of scaleBuffers to a cumulative scale
 * buffer. It is used to calculate the marginal scaling at a specific node for each site.
 *
 * @param instance                  Instance number (input)
 * @param scaleIndices            	List of scaleBuffers to add (input)
 * @param count                     Number of scaleBuffers in list (input)
 * @param cumulativeScaleIndex      Index number of scaleBuffer to accumulate factors into (input)
 */
int beagleAccumulateScaleFactors(int instance,
                                 const int* scaleIndices,
                                 int count,
                                 int cumulativeScaleIndex);

/**
 * @brief Remove scale factors
 *
 * This function removes (log) scale factors from a cumulative scale buffer. The
 * scale factors to be removed are indicated in a list of scaleBuffers.
 *
 * @param instance                  Instance number (input)
 * @param scaleIndices            	List of scaleBuffers to remove (input)
 * @param count                     Number of scaleBuffers in list (input)
 * @param cumulativeScaleIndex    	Index number of scaleBuffer containing accumulated factors (input)
 */
int beagleRemoveScaleFactors(int instance,
                             const int* scaleIndices,
                             int count,
                             int cumulativeScaleIndex);

/**
 * @brief Reset scalefactors
 *
 * This function resets a cumulative scale buffer.
 *
 * @param instance                  Instance number (input)
 * @param cumulativeScaleIndex    	Index number of cumulative scaleBuffer (input)
 */
int beagleResetScaleFactors(int instance,
                            int cumulativeScaleIndex);

/**
 * @brief Calculate site log likelihoods at a root node
 *
 * This function integrates a list of partials at a node with respect to a set of partials-weights
 * and state frequencies to return the log likelihoods for each site
 *
 * @param instance               Instance number (input)
 * @param bufferIndices          List of partialsBuffer indices to integrate (input)
 * @param inWeights              List of weights to apply to each partialsBuffer (input). There
 *                                should be one categoryCount sized set for each of
 *                                parentBufferIndices
 * @param inStateFrequencies     List of state frequencies for each partialsBuffer (input). There
 *                                should be one set for each of parentBufferIndices
 * @param cumulativeScaleIndices List of scaleBuffers containing accumulated factors to apply to
 *                                each partialsBuffer (input). There should be one index for each
 *                                of parentBufferIndices
 * @param count                  Number of partialsBuffer to integrate (input)
 * @param outLogLikelihoods      Pointer to destination for resulting log likelihoods (output)
 *
 * @return error code
 */
int beagleCalculateRootLogLikelihoods(int instance,
                                      const int* bufferIndices,
                                      const double* inWeights,
                                      const double* inStateFrequencies,
                                      const int* cumulativeScaleIndices,
                                      int count,
                                      double* outLogLikelihoods);

/**
 * @brief Calculate site log likelihoods and derivatives along an edge
 *
 * This function integrates a list of partials at a parent and child node with respect
 * to a set of partials-weights and state frequencies to return the log likelihoods
 * and first and second derivatives for each site
 *
 * @param instance                  Instance number (input)
 * @param parentBufferIndices       List of indices of parent partialsBuffers (input)
 * @param childBufferIndices        List of indices of child partialsBuffers (input)
 * @param probabilityIndices        List indices of transition probability matrices for this edge
 *                                   (input)
 * @param firstDerivativeIndices    List indices of first derivative matrices (input)
 * @param secondDerivativeIndices   List indices of second derivative matrices (input)
 * @param inWeights                 List of weights to apply to each partialsBuffer (input)
 * @param inStateFrequencies        List of state frequencies for each partialsBuffer (input). There
 *                                   should be one set for each of parentBufferIndices
 * @param cumulativeScaleIndices    List of scaleBuffers containing accumulated factors to apply to
 *                                   each partialsBuffer (input). There should be one index for each
 *                                   of parentBufferIndices
 * @param count                     Number of partialsBuffers (input)
 * @param outLogLikelihoods         Pointer to destination for resulting log likelihoods (output)
 * @param outFirstDerivatives       Pointer to destination for resulting first derivatives (output)
 * @param outSecondDerivatives      Pointer to destination for resulting second derivatives (output)
 *
 * @return error code
 */
int beagleCalculateEdgeLogLikelihoods(int instance,
                                      const int* parentBufferIndices,
                                      const int* childBufferIndices,
                                      const int* probabilityIndices,
                                      const int* firstDerivativeIndices,
                                      const int* secondDerivativeIndices,
                                      const double* inWeights,
                                      const double* inStateFrequencies,
                                      const int* cumulativeScaleIndices,
                                      int count,
                                      double* outLogLikelihoods,
                                      double* outFirstDerivatives,
                                      double* outSecondDerivatives);

/* using C calling conventions so that C programs can successfully link the beagle library
 * (closing brace)
 */
#ifdef __cplusplus
}
#endif

#endif // __beagle__
