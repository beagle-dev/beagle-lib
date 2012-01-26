/**
 * @file beagle.h
 *
 * Copyright 2009-2011 Phylogenetic Likelihood Working Group
 *
 * This file is part of BEAGLE.
 *
 * BEAGLE is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 *
 * BEAGLE is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with BEAGLE.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @brief This file documents the API as well as header for the
 * Broad-platform Evolutionary Analysis General Likelihood Evaluator
 *
 * KEY CONCEPTS
 * 
 * The key to BEAGLE performance lies in delivering fine-scale
 * parallelization while minimizing data transfer and memory copy overhead.
 * To accomplish this, the library lacks the concept or data structure for
 * a tree, in spite of the intended use for phylogenetic analysis. Instead,
 * BEAGLE acts directly on flexibly indexed data storage (called buffers)
 * for observed character states and partial likelihoods. The client
 * program can set the input buffers to reflect the data and can calculate
 * the likelihood of a particular phylogeny by invoking likelihood
 * calculations on the appropriate input and output buffers in the correct
 * order. Because of this design simplicity, the library can support many
 * different tree inference algorithms and likelihood calculation on a
 * variety of models. Arbitrary numbers of states can be used, as can
 * nonreversible substitution matrices via complex eigen decompositions,
 * and mixture models with multiple rate categories and/or multiple eigen
 * decompositions. Finally, BEAGLE application programming interface (API)
 * calls can be asynchronous, allowing the calling program to implement
 * other coarse-scale parallelization schemes such as evaluating
 * independent genes or running concurrent Markov chains.
 * 
 * USAGE
 * 
 * To use the library, a client program first creates an instance of BEAGLE
 * by calling beagleCreateInstance; multiple instances per client are
 * possible and encouraged. All additional functions are called with a
 * reference to this instance. The client program can optionally request
 * that an instance run on certain hardware (e.g., a GPU) or have
 * particular features (e.g., double-precision math). Next, the client
 * program must specify the data dimensions and specify key aspects of the
 * phylogenetic model. Character state data are then loaded and can be in
 * the form of discrete observed states or partial likelihoods for
 * ambiguous characters. The observed data are usually unchanging and
 * loaded only once at the start to minimize memory copy overhead. The
 * character data can be compressed into unique “site patterns” and
 * associated weights for each. The parameters of the substitution process
 * can then be specified, including the equilibrium state frequencies, the
 * rates for one or more substitution rate categories and their weights,
 * and finally, the eigen decomposition for the substitution process.
 * 
 * In order to calculate the likelihood of a particular tree, the client
 * program then specifies a series of integration operations that
 * correspond to steps in Felsenstein’s algorithm. Finite-time transition
 * probabilities for each edge are loaded directly if considering a
 * nondiagonalizable model or calculated in parallel from the eigen
 * decomposition and edge lengths specified. This is performed within
 * BEAGLE’s memory space to minimize data transfers. A single function call
 * will then request one or more integration operations to calculate
 * partial likelihoods over some or all nodes. The operations are performed
 * in the order they are provided, typically dictated by a postorder
 * traversal of the tree topology. The client needs only specify nodes for
 * which the partial likelihoods need updating, but it is up to the calling
 * software to keep track of these dependencies. The final step in
 * evaluating the phylogenetic model is done using an API call that yields
 * a single log likelihood for the model given the data.
 * 
 * Aspects of the BEAGLE API design support both maximum likelihood (ML)
 * and Bayesian phylogenetic tree inference. For ML inference, API calls
 * can calculate first and second derivatives of the likelihood with
 * respect to the lengths of edges (branches). In both cases, BEAGLE
 * provides the ability to cache and reuse previously computed partial
 * likelihood results, which can yield a tremendous speedup over
 * recomputing the entire likelihood every time a new phylogenetic model is
 * evaluated.
 *
 * @author Likelihood API Working Group
 *
 * @author Daniel Ayres
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

#include "libhmsbeagle/platform.h"

/**
 * @anchor BEAGLE_RETURN_CODES
 *
 * @brief Error return codes
 *
 * This enumerates all possible BEAGLE return codes.  Error codes are always negative.
 */
enum BeagleReturnCodes {
    BEAGLE_SUCCESS                      =  0,  /**< Success */
    BEAGLE_ERROR_GENERAL                = -1,  /**< Unspecified error */
    BEAGLE_ERROR_OUT_OF_MEMORY          = -2,  /**< Not enough memory could be allocated */
    BEAGLE_ERROR_UNIDENTIFIED_EXCEPTION = -3,  /**< Unspecified exception */
    BEAGLE_ERROR_UNINITIALIZED_INSTANCE = -4,  /**< The instance index is out of range,
                                                *   or the instance has not been created */
    BEAGLE_ERROR_OUT_OF_RANGE           = -5,  /**< One of the indices specified exceeded the range of the
                                                *   array */
    BEAGLE_ERROR_NO_RESOURCE            = -6,  /**< No resource matches requirements */
    BEAGLE_ERROR_NO_IMPLEMENTATION      = -7,  /**< No implementation matches requirements */
    BEAGLE_ERROR_FLOATING_POINT         = -8   /**< Floating-point range exceeded */
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
    BEAGLE_FLAG_PRECISION_SINGLE    = 1 << 0,    /**< Single precision computation */
    BEAGLE_FLAG_PRECISION_DOUBLE    = 1 << 1,    /**< Double precision computation */

    BEAGLE_FLAG_COMPUTATION_SYNCH   = 1 << 2,    /**< Synchronous computation (blocking) */
    BEAGLE_FLAG_COMPUTATION_ASYNCH  = 1 << 3,    /**< Asynchronous computation (non-blocking) */
    
    BEAGLE_FLAG_EIGEN_REAL          = 1 << 4,    /**< Real eigenvalue computation */
    BEAGLE_FLAG_EIGEN_COMPLEX       = 1 << 5,    /**< Complex eigenvalue computation */

    BEAGLE_FLAG_SCALING_MANUAL      = 1 << 6,    /**< Manual scaling */
    BEAGLE_FLAG_SCALING_AUTO        = 1 << 7,    /**< Auto-scaling on */
    BEAGLE_FLAG_SCALING_ALWAYS      = 1 << 8,    /**< Scale at every updatePartials */
    BEAGLE_FLAG_SCALING_DYNAMIC     = 1 << 19,   /**< Manual scaling with dynamic checking  */
    
    BEAGLE_FLAG_SCALERS_RAW         = 1 << 9,    /**< Save raw scalers */
    BEAGLE_FLAG_SCALERS_LOG         = 1 << 10,   /**< Save log scalers */
    
    BEAGLE_FLAG_INVEVEC_STANDARD    = 1 << 20,   /**< Inverse eigen vectors passed to BEAGLE have not been transposed */
    BEAGLE_FLAG_INVEVEC_TRANSPOSED  = 1 << 21,   /**< Inverse eigen vectors passed to BEAGLE have been transposed */
    
    BEAGLE_FLAG_VECTOR_SSE          = 1 << 11,   /**< SSE computation */
    BEAGLE_FLAG_VECTOR_NONE         = 1 << 12,   /**< No vector computation */
    
    BEAGLE_FLAG_THREADING_OPENMP    = 1 << 13,   /**< OpenMP threading */
    BEAGLE_FLAG_THREADING_NONE      = 1 << 14,   /**< No threading */
    
    BEAGLE_FLAG_PROCESSOR_CPU       = 1 << 15,   /**< Use CPU as main processor */
    BEAGLE_FLAG_PROCESSOR_GPU       = 1 << 16,   /**< Use GPU as main processor */
    BEAGLE_FLAG_PROCESSOR_FPGA      = 1 << 17,   /**< Use FPGA as main processor */
    BEAGLE_FLAG_PROCESSOR_CELL      = 1 << 18,   /**< Use Cell as main processor */
    
    BEAGLE_FLAG_FRAMEWORK_CUDA      = 1 << 22,   /**< Use CUDA implementation with GPU resources */
    BEAGLE_FLAG_FRAMEWORK_OPENCL    = 1 << 23    /**< Use OpenCL implementation with GPU resources */
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
    char* resourceName; /**< Name of resource on which this instance is running as a NULL-terminated
					     *   character string */
    char* implName;     /**< Name of implementation on which this instance is running as a
                         *   NULL-terminated character string */
    char* implDescription; /**< Description of implementation with details such as how auto-scaling is performed */
    long flags;         /**< Bit-flags that characterize the activate
                         *   capabilities of the resource and implementation for this instance */
} BeagleInstanceDetails;

/**
 * @brief Description of a hardware resource
 */
typedef struct {
    char* name;         /**< Name of resource as a NULL-terminated character string */
    char* description;  /**< Description of resource as a NULL-terminated character string */
    long  supportFlags; /**< Bit-flags of supported capabilities on resource */
    long  requiredFlags;/**< Bit-flags that identify resource type */
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
 * @brief Get list of hardware resources
 *
 * This function returns a pointer to a BeagleResourceList struct, which includes
 * a BeagleResource array describing the available hardware resources.
 *
 * @return A list of hardware resources available to the library as a BeagleResourceList
 */
BEAGLE_DLLEXPORT BeagleResourceList* beagleGetResourceList(void);

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
 * @param eigenBufferCount      Number of rate matrix eigen-decomposition, category weight, and
 *                               state frequency buffers to allocate (input)
 * @param matrixBufferCount     Number of transition probability matrix buffers (input)
 * @param categoryCount         Number of rate categories (input)
 * @param scaleBufferCount		Number of scale buffers to create, ignored for auto scale or always scale (input)
 * @param resourceList          List of potential resources on which this instance is allowed
 *                               (input, NULL implies no restriction)
 * @param resourceCount         Length of resourceList list (input)
 * @param preferenceFlags       Bit-flags indicating preferred implementation characteristics,
 *                               see BeagleFlags (input)
 * @param requirementFlags      Bit-flags indicating required implementation characteristics,
 *                               see BeagleFlags (input)
 * @param returnInfo            Pointer to return implementation and resource details
 *
 * @return the unique instance identifier (<0 if failed, see @ref BEAGLE_RETURN_CODES
 * "BeagleReturnCodes")
 */
// TODO: if setting your own matrices, might not need eigen buffers allocated, but still need
//        category weight and state frequency buffers
BEAGLE_DLLEXPORT int beagleCreateInstance(int tipCount,
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
                         long requirementFlags,
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
BEAGLE_DLLEXPORT int beagleFinalizeInstance(int instance);

/**
 * @brief Finalize the library
 *
 * This function finalizes the library and releases all allocated memory.
 * This function is automatically called under GNU C via __attribute__ ((destructor)).
 *
 * @return error code
 */
BEAGLE_DLLEXPORT int beagleFinalize(void);
        
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
BEAGLE_DLLEXPORT int beagleSetTipStates(int instance,
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
BEAGLE_DLLEXPORT int beagleSetTipPartials(int instance,
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
BEAGLE_DLLEXPORT int beagleSetPartials(int instance,
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
BEAGLE_DLLEXPORT int beagleGetPartials(int instance,
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
BEAGLE_DLLEXPORT int beagleSetEigenDecomposition(int instance,
                                int eigenIndex,
                                const double* inEigenVectors,
                                const double* inInverseEigenVectors,
                                const double* inEigenValues);

/**
 * @brief Set a state frequency buffer
 *
 * This function copies a state frequency array into an instance buffer.
 *
 * @param instance              Instance number (input)
 * @param stateFrequenciesIndex Index of state frequencies buffer (input)
 * @param inStateFrequencies    State frequencies array (stateCount) (input)
 *
 * @return error code
 */
BEAGLE_DLLEXPORT int beagleSetStateFrequencies(int instance,
                                         int stateFrequenciesIndex,
                                         const double* inStateFrequencies);    
    
/**
 * @brief Set a category weights buffer
 *
 * This function copies a category weights array into an instance buffer.
 *
 * @param instance              Instance number (input)
 * @param categoryWeightsIndex  Index of category weights buffer (input)
 * @param inCategoryWeights     Category weights array (categoryCount) (input)
 *
 * @return error code
 */
BEAGLE_DLLEXPORT int beagleSetCategoryWeights(int instance,
                                        int categoryWeightsIndex,
                                        const double* inCategoryWeights);

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
BEAGLE_DLLEXPORT int beagleSetCategoryRates(int instance,
                           const double* inCategoryRates);
/**
 * @brief Set pattern weights
 *
 * This function sets the vector of pattern weights for an instance.
 *
 * @param instance              Instance number (input)
 * @param inPatternWeights      Array containing patternCount weights (input)
 *
 * @return error code
 */
BEAGLE_DLLEXPORT int beagleSetPatternWeights(int instance,
                                       const double* inPatternWeights);
    
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
 * @param secondDerivativeIndices    List of indices of second derivative matrices to update
 *                                   (input, NULL implies no calculation)
 * @param edgeLengths               List of edge lengths with which to perform calculations (input)
 * @param count                     Length of lists
 *
 * @return error code
 */
BEAGLE_DLLEXPORT int beagleUpdateTransitionMatrices(int instance,
                                   int eigenIndex,
                                   const int* probabilityIndices,
                                   const int* firstDerivativeIndices,
                                   const int* secondDerivativeIndices,
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
 * @param paddedValue   Value to be used for padding for ambiguous states (e.g. 1 for probability matrices, 0 for derivative matrices) (input)
 *
 * @return error code
 */
BEAGLE_DLLEXPORT int beagleSetTransitionMatrix(int instance,
                              int matrixIndex,
                              const double* inMatrix,
                              double paddedValue);

/**
 * @brief Get a finite-time transition probability matrix
 *
 * This function copies a finite-time transition matrix buffer into the array outMatrix. The
 * outMatrix array should be of size stateCount * stateCount * categoryCount and will be filled
 * with one matrix for each rate category.
 *
 * @param instance	   Instance number (input)
 * @param matrixIndex  Index of matrix buffer (input)
 * @param outMatrix    Pointer to destination transition probability matrix (output)
 *
 * @return error code
 */
BEAGLE_DLLEXPORT int beagleGetTransitionMatrix(int instance,
								int matrixIndex,
								double* outMatrix);
    
/**
 * @brief Set multiple transition matrices
 *
 * This function copies multiple transition matrices into matrix buffers. This function
 * is used when the application wishes to explicitly set the transition matrices rather than
 * using the beagleSetEigenDecomposition and beagleUpdateTransitionMatrices functions. The inMatrices array should be
 * of size stateCount * stateCount * categoryCount * count.
 *
 * @param instance      Instance number (input)
 * @param matrixIndices Indices of matrix buffers (input)
 * @param inMatrices    Pointer to source transition matrices (input)
 * @param paddedValues  Values to be used for padding for ambiguous states (e.g. 1 for probability matrices, 0 for derivative matrices) (input)
 * @param count         Number of transition matrices (input)
 *
 * @return error code
 */
BEAGLE_DLLEXPORT int beagleSetTransitionMatrices(int instance,
                                                 const int* matrixIndices,
                                                 const double* inMatrices,
                                                 const double* paddedValues,
                                                 int count);

    
/**
 * @brief A list of integer indices which specify a partial likelihoods operation.
 */
typedef struct {
	int destinationPartials;    /**< index of destination, or parent, partials buffer  */
	int destinationScaleWrite;  /**< index of scaling buffer to write to (if set to BEAGLE_OP_NONE then calculation of new scalers is disabled)  */
	int destinationScaleRead;   /**< index of scaling buffer to read from (if set to BEAGLE_OP_NONE then use of existing scale factors is disabled)  */
	int child1Partials;         /**< index of first child partials buffer */
	int child1TransitionMatrix; /**< index of transition matrix of first partials child buffer  */
	int child2Partials;         /**< index of second child partials buffer */
	int child2TransitionMatrix; /**< index of transition matrix of second partials child buffer */
} BeagleOperation;

/**
 * @brief Calculate or queue for calculation partials using a list of operations
 *
 * This function either calculates or queues for calculation a list partials. Implementations
 * supporting ASYNCH may queue these calculations while other implementations perform these
 * operations immediately and in order.
*
 * @param instance                  Instance number (input)
 * @param operations                BeagleOperation list specifying operations (input)
 * @param operationCount            Number of operations (input)
 * @param cumulativeScaleIndex   	Index number of scaleBuffer to store accumulated factors (input)
 *
 * @return error code
 */
BEAGLE_DLLEXPORT int beagleUpdatePartials(const int instance,
                         const BeagleOperation* operations,
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
 * @param instance                  Instance number (input)
 * @param destinationPartials       List of the indices of destinationPartials that must be
 *                                   calculated before the function returns
 * @param destinationPartialsCount  Number of destinationPartials (input)
 *
 * @return error code
 */
BEAGLE_DLLEXPORT int beagleWaitForPartials(const int instance,
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
BEAGLE_DLLEXPORT int beagleAccumulateScaleFactors(int instance,
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
BEAGLE_DLLEXPORT int beagleRemoveScaleFactors(int instance,
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
BEAGLE_DLLEXPORT int beagleResetScaleFactors(int instance,
                            int cumulativeScaleIndex);

/**
 * @brief Copy scale factors
 *
 * This function copies scale factors from one buffer to another.
 *
 * @param instance                  Instance number (input)
 * @param destScalingIndex          Destination scaleBuffer (input)
 * @param srcScalingIndex           Source scaleBuffer (input)
 */
BEAGLE_DLLEXPORT int beagleCopyScaleFactors(int instance,
                                            int destScalingIndex,
                                            int srcScalingIndex);
    
/**
 * @brief Calculate site log likelihoods at a root node
 *
 * This function integrates a list of partials at a node with respect to a set of partials-weights
 * and state frequencies to return the log likelihood sum
 *
 * @param instance                 Instance number (input)
 * @param bufferIndices            List of partialsBuffer indices to integrate (input)
 * @param categoryWeightsIndices   List of weights to apply to each partialsBuffer (input). There
 *                                  should be one categoryCount sized set for each of
 *                                  parentBufferIndices
 * @param stateFrequenciesIndices  List of state frequencies for each partialsBuffer (input). There
 *                                  should be one set for each of parentBufferIndices
 * @param cumulativeScaleIndices   List of scaleBuffers containing accumulated factors to apply to
 *                                  each partialsBuffer (input). There should be one index for each
 *                                  of parentBufferIndices
 * @param count                    Number of partialsBuffer to integrate (input)
 * @param outSumLogLikelihood      Pointer to destination for resulting log likelihood (output)
 *
 * @return error code
 */
// TODO: only need one state frequency index
BEAGLE_DLLEXPORT int beagleCalculateRootLogLikelihoods(int instance,
                                      const int* bufferIndices,
                                      const int* categoryWeightsIndices,
                                      const int* stateFrequenciesIndices,
                                      const int* cumulativeScaleIndices,
                                      int count,
                                      double* outSumLogLikelihood);

/**
 * @brief Calculate site log likelihoods and derivatives along an edge
 *
 * This function integrates a list of partials at a parent and child node with respect
 * to a set of partials-weights and state frequencies to return the log likelihood
 * and first and second derivative sums
 *
 * @param instance                  Instance number (input)
 * @param parentBufferIndices       List of indices of parent partialsBuffers (input)
 * @param childBufferIndices        List of indices of child partialsBuffers (input)
 * @param probabilityIndices        List indices of transition probability matrices for this edge
 *                                   (input)
 * @param firstDerivativeIndices    List indices of first derivative matrices (input)
 * @param secondDerivativeIndices   List indices of second derivative matrices (input)
 * @param categoryWeightsIndices    List of weights to apply to each partialsBuffer (input)
 * @param stateFrequenciesIndices   List of state frequencies for each partialsBuffer (input). There
 *                                   should be one set for each of parentBufferIndices
 * @param cumulativeScaleIndices    List of scaleBuffers containing accumulated factors to apply to
 *                                   each partialsBuffer (input). There should be one index for each
 *                                   of parentBufferIndices
 * @param count                     Number of partialsBuffers (input)
 * @param outSumLogLikelihood       Pointer to destination for resulting log likelihood (output)
 * @param outSumFirstDerivative     Pointer to destination for resulting first derivative (output)
 * @param outSumSecondDerivative    Pointer to destination for resulting second derivative (output)
 *
 * @return error code
 */
BEAGLE_DLLEXPORT int beagleCalculateEdgeLogLikelihoods(int instance,
                                      const int* parentBufferIndices,
                                      const int* childBufferIndices,
                                      const int* probabilityIndices,
                                      const int* firstDerivativeIndices,
                                      const int* secondDerivativeIndices,
                                      const int* categoryWeightsIndices,
                                      const int* stateFrequenciesIndices,
                                      const int* cumulativeScaleIndices,
                                      int count,
                                      double* outSumLogLikelihood,
                                      double* outSumFirstDerivative,
                                      double* outSumSecondDerivative);

/**
 * @brief Get site log likelihoods for last beagleCalculateRootLogLikelihoods or
 *         beagleCalculateEdgeLogLikelihoods call
 *
 * This function returns the log likelihoods for each site 
 *
 * @param instance               Instance number (input)
 * @param outLogLikelihoods      Pointer to destination for resulting log likelihoods (output)
 *
 * @return error code
 */
BEAGLE_DLLEXPORT int beagleGetSiteLogLikelihoods(int instance,
                                       double* outLogLikelihoods);

/**
 * @brief Get site derivatives for last beagleCalculateEdgeLogLikelihoods call
 *
 * This function returns the derivatives for each site 
 *
 * @param instance               Instance number (input)
 * @param outFirstDerivatives    Pointer to destination for resulting first derivatives (output)
 * @param outSecondDerivatives   Pointer to destination for resulting second derivatives (output)
 *
 * @return error code
 */
BEAGLE_DLLEXPORT int beagleGetSiteDerivatives(int instance,
                                    double* outFirstDerivatives,
                                    double* outSecondDerivatives);    
    
/* using C calling conventions so that C programs can successfully link the beagle library
 * (closing brace)
 */
#ifdef __cplusplus
}
#endif

#endif // __beagle__
