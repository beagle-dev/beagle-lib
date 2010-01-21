/*
 * @file beagle.h
 *
 * Copyright 2009 Phylogenetic Likelihood Working Group
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
 * OVERVIEW
 *
 * DEFINITION OF KEY CONCEPTS: INSTANCE, BUFFER, etc.
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
    BEAGLE_SUCCESS                      = 0,   /**< Success */
    BEAGLE_ERROR_GENERAL                = -1,  /**< Unspecified error */
    BEAGLE_ERROR_OUT_OF_MEMORY          = -2,  /**< Not enough memory could be allocated */
    BEAGLE_ERROR_UNIDENTIFIED_EXCEPTION = -3,  /**< Unspecified exception */
    BEAGLE_ERROR_UNINITIALIZED_INSTANCE = -4,  /**< The instance index is out of range,
                                                *   or the instance has not been created */
    BEAGLE_ERROR_OUT_OF_RANGE           = -5,  /**< One of the indices specified exceeded the range of the
                                                *   array */
    BEAGLE_ERROR_NO_RESOURCE            = -6,  /**< No resource matches requirements */
    BEAGLE_ERROR_NO_IMPLEMENTATION      = -7   /**< No implementation matches requirements */
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
    
    BEAGLE_FLAG_SCALERS_RAW         = 1 << 9,    /**< Save raw scalers */
    BEAGLE_FLAG_SCALERS_LOG         = 1 << 10,   /**< Save log scalers */
    
    BEAGLE_FLAG_VECTOR_SSE          = 1 << 11,   /**< SSE computation */
    BEAGLE_FLAG_VECTOR_NONE         = 1 << 12,   /**< No vector computation */
    
    BEAGLE_FLAG_THREADING_OPENMP    = 1 << 13,   /**< OpenMP threading */
    BEAGLE_FLAG_THREADING_NONE      = 1 << 14,   /**< No threading */
    
    BEAGLE_FLAG_PROCESSOR_CPU       = 1 << 15,   /**< Use CPU as main processor */
    BEAGLE_FLAG_PROCESSOR_GPU       = 1 << 16,   /**< Use GPU as main processor */
    BEAGLE_FLAG_PROCESSOR_FPGA      = 1 << 17,  /**< Use FPGA as main processor */
    BEAGLE_FLAG_PROCESSOR_CELL      = 1 << 18,  /**< Use Cell as main processor */
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
    long  requiredFlags;/**< Bit-flags of required capabilities on resource */
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
 * @param preferenceFlags       Bit-flags indicating preferred implementation charactertistics,
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
 * @param eigenIndex            Index of state frequencies buffer (input)
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
 * @param eigenIndex            Index of category weights buffer (input)
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
 *
 * @return error code
 */
BEAGLE_DLLEXPORT int beagleSetTransitionMatrix(int instance,
                              int matrixIndex,
                              const double* inMatrix);

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
 * @brief Calculate or queue for calculation partials using a list of operations
 *
 * This function either calculates or queues for calculation a list partials. Implementations
 * supporting ASYNCH may queue these calculations while other implementations perform these
 * operations immediately and in order.
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
 * @param instance                  Instance number (input)
 * @param operations                List of 7-tuples specifying operations (input)
 * @param operationCount            Number of operations (input)
 * @param cumulativeScaleIndex   	Index number of scaleBuffer to store accumulated factors (input)
 *
 * @return error code
 */
BEAGLE_DLLEXPORT int beagleUpdatePartials(const int instance,
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
