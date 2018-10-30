/**
 * @file Beagle.java
 *
 * Copyright 2009-2016 Phylogenetic Likelihood Working Group
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
 * To accomplish this, the library lacks the concept of data structure for
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

package beagle;

import java.io.Serializable;

/**
 * Beagle - An interface exposing the BEAGLE likelihood evaluation library.
 *
 * This interface mirrors the beagle.h API but it for a single instance only.
 * It is intended to be used by JNI wrappers of the BEAGLE library and for
 * Java implementations for testing purposes. BeagleFactory handles the creation
 * of specific istances.
 *
 * @author Andrew Rambaut
 * @author Marc A. Suchard
 * @version $Id:$
 */

public interface Beagle extends Serializable {

    public static int OPERATION_TUPLE_SIZE = 7;
    public static int PARTITION_OPERATION_TUPLE_SIZE = 9;
    public static int NONE = -1;


    /**
     * Finalize this instance
     *
     * This function finalizes the instance by releasing allocated memory
     */
    void finalize() throws Throwable;


    /**
     * Set number of threads for native CPU implementation
     *
     * This function sets the number of worker threads to be used with a native
     * CPU implementation. It should only be called after beagleCreateInstance and
     * requires the THREADING_CPP flag to be set. It has no effect on GPU-based
     * implementations. It has no effect with the default THREADING_NONE setting.
     * If THREADING_CPP is set and this function is not called BEAGLE will use 
     * a heuristic to set an appropriate number of threads.
     *
     * @param threadCount          Number of threads (input)
     */
    void setCPUThreadCount(int threadCount);

    /**
     * Set the weights for each pattern
     * @param patternWeights    Array containing patternCount weights
     */
    void setPatternWeights(final double[] patternWeights);

    /**
     * Set pattern partition assignments
     *
     * This function sets the vector of pattern partition indices for an instance. It should
     * only be called after setTipPartials.
     *
     * @param partitionCount        Number of partitions
     * @param patternPartitions     Array containing partitionCount partition indices (input)
     */
    void setPatternPartitions(int partitionCount, final int[] patternPartitions);

    /**
     * Set the compressed state representation for tip node
     *
     * This function copies a compact state representation into an instance buffer.
     * Compact state representation is an array of states: 0 to stateCount - 1 (missing = stateCount).
     * The inStates array should be patternCount in length (replication across categoryCount is not
     * required).
     *
     * @param tipIndex   Index of destination partialsBuffer (input)
     * @param inStates   Pointer to compressed states (input)
     */
    void setTipStates(
            int tipIndex,
            final int[] inStates);

    /**
     * Get the compressed state representation for tip node
     *
     * This function copies a compact state representation from an instance buffer.
     * Compact state representation is an array of states: 0 to stateCount - 1 (missing = stateCount).
     * The inStates array should be patternCount in length (replication across categoryCount is not
     * required).
     *
     * @param tipIndex   Index of destination partialsBuffer (input)
     * @param outStates   Pointer to compressed states (input)
     */
    void getTipStates(
            int tipIndex,
            final int[] outStates);

    /**
     * Set an instance partials buffer
     *
     * This function copies an array of partials into an instance buffer. The inPartials array should
     * be stateCount * patternCount in length. For most applications this will be used
     * to set the partial likelihoods for the observed states. Internally, the partials will be copied
     * categoryCount times.
     *
     * @param tipIndex   Index of destination partialsBuffer (input)
     * @param  inPartials   Pointer to partials values to set (input)
     */
    void setTipPartials(
            int tipIndex,
            final double[] inPartials);

    /**
     * Set an instance partials buffer
     *
     * This function copies an array of partials into an instance buffer. The inPartials array should
     * be stateCount * patternCount * categoryCount in length.
     *
     * @param bufferIndex   Index of destination partialsBuffer (input)
     * @param  inPartials   Pointer to partials values to set (input)
     */
    void setPartials(
            int bufferIndex,
            final double[] inPartials);

    /**
     * Get partials from an instance buffer
     *
     * This function copies an array of partials from an instance buffer. The inPartials array should
     * be stateCount * patternCount * categoryCount in length.
     *
     * @param bufferIndex   Index of destination partialsBuffer (input)
     * @param scaleIndex    Index of scaleBuffer to apply to partials (input)
     * @param  outPartials  Pointer to which to receive partialsBuffer (output)
     */
    void getPartials(
            int bufferIndex,
            int scaleIndex,
            final double[] outPartials);
                        
    /**
     * Get scale factors from instance buffer on log-scale
     *
     * This function copies an array of scale factors from an instance buffer. The outFactors array should
     * be patternCount in length.
     *   
     * @param scaleIndex    Index of scaleBuffer to get (input)
     * @param  outFactors  Pointer to which to receive partialsBuffer (output)
     */
    void getLogScaleFactors(           
            int scaleIndex,
            final double[] outFactors);            

    /**
     * Set an eigen-decomposition buffer
     *
     * This function copies an eigen-decomposition into a instance buffer.
     *
     * @param eigenIndex                Index of eigen-decomposition buffer (input)
     * @param inEigenVectors            Flattened matrix (stateCount x stateCount) of eigen-vectors (input)
     * @param inInverseEigenVectors     Flattened matrix (stateCount x stateCount) of inverse-eigen-vectors (input)
     * @param inEigenValues             Vector of eigenvalues
     */
    void setEigenDecomposition(
            int eigenIndex,
            final double[] inEigenVectors,
            final double[] inInverseEigenVectors,
            final double[] inEigenValues);

    /**
     * Set a set of state frequences. These will probably correspond to an
     * eigen-system.
     *
     * @param stateFrequenciesIndex the index of the frequency buffer
     * @param stateFrequencies the array of frequences (stateCount)
     */
    void setStateFrequencies(int stateFrequenciesIndex,
                             final double[] stateFrequencies);

    /**
     * Set a set of category weights. These will probably correspond to an
     * eigen-system.
     *
     * @param categoryWeightsIndex the index of the buffer
     * @param categoryWeights the array of weights
     */
    void setCategoryWeights(int categoryWeightsIndex,
                            final double[] categoryWeights);

    /**
     * Set default category rates buffer
     *
     * This function sets the default vector of category rates for an instance.
     *
     * @param inCategoryRates       Array containing categoryCount rate scalers (input)
     */
    void setCategoryRates(final double[] inCategoryRates);

    /**
     * Set a category rates buffer
     *
     * This function sets the vector of category rates for a given buffer in an instance.
     *
     * @param categoryRatesIndex    the index of the buffer
     * @param inCategoryRates       Array containing categoryCount rate scalers (input)
     */
    void setCategoryRatesWithIndex(int categoryRatesIndex,
                                   final double[] inCategoryRates);

    /**
     * Convolve lists of transition probability matrices
     *
     * This function convolves two lists of transition probability matrices.
     *
     * @param firstIndices              List of indices of the first transition probability matrices to convolve (input)
     * @param secondIndices             List of indices of the second transition probability matrices to convolve (input)
     * @param resultIndices             List of indices of resulting transition probability matrices (input)
     * @param matrixCount               Lenght of lists
     */
    void convolveTransitionMatrices(
            final int[] firstIndices,
            final int[] secondIndices,
            final int[] resultIndices,
            int matrixCount);
    
    /**
     * Calculate a list of transition probability matrices
     *
     * This function calculates a list of transition probabilities matrices and their first and
     * second derivatives (if requested).
     *
     * @param eigenIndex                Index of eigen-decomposition buffer (input)
     * @param probabilityIndices        List of indices of transition probability matrices to update (input)
     * @param firstDerivativeIndices    List of indices of first derivative matrices to update (input, NULL implies no calculation)
     * @param secondDervativeIndices    List of indices of second derivative matrices to update (input, NULL implies no calculation)
     * @param edgeLengths               List of edge lengths with which to perform calculations (input)
     * @param count                     Length of lists
     */
    void updateTransitionMatrices(
            int eigenIndex,
            final int[] probabilityIndices,
            final int[] firstDerivativeIndices,
            final int[] secondDervativeIndices,
            final double[] edgeLengths,
            int count);

    /**
     * Calculate a list of transition probability matrices with multiple models
     *
     * This function calculates a list of transition probabilities matrices and their first and
     * second derivatives (if requested).
     *
     * @param eigenIndices              List of indices of eigen-decomposition buffers (input)
     * @param categoryRateIndices       List of indices of category-rate buffers (input)
     * @param probabilityIndices        List of indices of transition probability matrices to update (input)
     * @param firstDerivativeIndices    List of indices of first derivative matrices to update (input, NULL implies no calculation)
     * @param secondDervativeIndices    List of indices of second derivative matrices to update (input, NULL implies no calculation)
     * @param edgeLengths               List of edge lengths with which to perform calculations (input)
     * @param count                     Length of lists
     */
    void updateTransitionMatricesWithMultipleModels(
            final int[] eigenIndices,
            final int[] categoryRateIndices,
            final int[] probabilityIndices,
            final int[] firstDerivativeIndices,
            final int[] secondDervativeIndices,
            final double[] edgeLengths,
            int count);

    /**
     * This function copies a finite-time transition probability matrix into a matrix buffer. This function
     * is used when the application wishes to explicitly set the transition probability matrix rather than
     * using the setEigenDecomposition and updateTransitionMatrices functions. The inMatrix array should be
     * of size stateCount * stateCount * categoryCount and will contain one matrix for each rate category.
     *
     * This function copies a finite-time transition probability matrix into a matrix buffer.
     * @param matrixIndex   Index of matrix buffer (input)
     * @param inMatrix          Pointer to source transition probability matrix (input)
     * @param paddedValue   Value to be used for padding for ambiguous states (e.g. 1 for probability matrices, 0 for derivative matrices) (input)
     */
    void setTransitionMatrix(
            int matrixIndex,			/**< Index of matrix buffer (input) */
            final double[] inMatrix, 	/**< Pointer to source transition probability matrix (input) */
            double paddedValue);

    /**
     * Get a finite-time transition probability matrix
     *
     * This function copies a finite-time transition matrix buffer into the array outMatrix. The
     * outMatrix array should be of size stateCount * stateCount * categoryCount and will be filled
     * with one matrix for each rate category.
     *
     * @param matrixIndex  Index of matrix buffer (input)
     * @param outMatrix    Pointer to destination transition probability matrix (output)
     *
     */
    void getTransitionMatrix(int matrixIndex,
                             double[] outMatrix);

    /**
     * Calculate or queue for calculation partials using a list of operations
     *
     * This function either calculates or queues for calculation a list partials. Implementations
     * supporting ASYNCH may queue these calculations while other implementations perform these
     * operations immediately and in order.
     *
     * If partitions have been set via setPatternPartitions, operationCount should be a
     * multiple of partitionCount.
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
     * @param operations            List of 7-tuples specifying operations (input)
     * @param operationCount        Number of operations (input)
     * @param cumulativeScaleIndex  Index number of scaleBuffer to store accumulated factors (input)
     *
     */
    void updatePartials(
            final int[] operations,
            int operationCount,
            int cumulativeScaleIndex);

    /**
     * Calculate or queue for calculating partials by partition using a list of operations
     *
     * This function either calculates or queues for calculation a list partials. Implementations
     * supporting ASYNCH may queue these calculations while other implementations perform these
     * operations immediately and in order.
     *
     * If partitions have been set via setPatternPartitions, operationCount should be a
     * multiple of partitionCount.
     *
     * Operations list is a list of 9-tuple integer indices, with one 9-tuple per operation.
     * Format of 9-tuple operation: {destinationPartials,
     *                               destinationScaleWrite,
     *                               destinationScaleRead,
     *                               child1Partials,
     *                               child1TransitionMatrix,
     *                               child2Partials,
     *                               child2TransitionMatrix,
     *                               partition,
     *                               cumulativeScaleIndex}
     *
     * @param operations            List of 9-tuples specifying operations (input)
     * @param operationCount        Number of operations (input)
     *
     */
    void updatePartialsByPartition(
            final int[] operations,
            int operationCount);

    /**
     * Accumulate scale factors
     *
     * This function adds (log) scale factors from a list of scaleBuffers to a cumulative scale
     * buffer. It is used to calculate the marginal scaling at a specific node for each site.
     *
     * @param scaleIndices            	List of scaleBuffers to add (input)
     * @param count                     Number of scaleBuffers in list (input)
     * @param cumulativeScaleIndex      Index number of scaleBuffer to accumulate factors into (input)
     */
    void accumulateScaleFactors(
            final int[] scaleIndices,
            final int count,
            final int cumulativeScaleIndex
    );

    /**
     * Accumulate scale factors by partition
     *
     * This function adds (log) scale factors from a list of scaleBuffers to a cumulative scale
     * buffer. It is used to calculate the marginal scaling at a specific node for each site.
     *
     * @param scaleIndices            	List of scaleBuffers to add (input)
     * @param count                     Number of scaleBuffers in list (input)
     * @param cumulativeScaleIndex      Index number of scaleBuffer to accumulate factors into (input)
     * @param partitionIndex            Index number of partition (input)
     */
    void accumulateScaleFactorsByPartition(
            final int[] scaleIndices,
            int count,
            int cumulativeScaleIndex,
            int partitionIndex
    );

    /**
     * Remove scale factors
     *
     * This function removes (log) scale factors from a cumulative scale buffer. The
     * scale factors to be removed are indicated in a list of scaleBuffers.
     *
     * @param scaleIndices            	List of scaleBuffers to remove (input)
     * @param count                     Number of scaleBuffers in list (input)
     * @param cumulativeScaleIndex    	Index number of scaleBuffer containing accumulated factors (input)
     */
    void removeScaleFactors(
            final int[] scaleIndices,
            final int count,
            final int cumulativeScaleIndex
    );

    /**
     * Remove scale factors by partition
     *
     * This function removes (log) scale factors from a cumulative scale buffer. The
     * scale factors to be removed are indicated in a list of scaleBuffers.
     *
     * @param scaleIndices            	List of scaleBuffers to remove (input)
     * @param count                     Number of scaleBuffers in list (input)
     * @param cumulativeScaleIndex    	Index number of scaleBuffer containing accumulated factors (input)
     * @param partitionIndex            Index number of partition (input)
     */
    void removeScaleFactorsByPartition(
            final int[] scaleIndices,
            final int count,
            final int cumulativeScaleIndex,
            final int partitionIndex
    );

    /**
     * Copy scale factors
     *
     * This function copies scale factors from one buffer to another.
     *
     * @param destScalingIndex          Destination scaleBuffer (input)
     * @param srcScalingIndex           Source scaleBuffer (input)
     */
    void copyScaleFactors(
        int destScalingIndex,
        int srcScalingIndex
    );    

    /**
     * Reset scalefactors
     *
     * This function resets a cumulative scale buffer.
     *
     * @param cumulativeScaleIndex    	Index number of cumulative scaleBuffer (input)
     */
    void resetScaleFactors(int cumulativeScaleIndex);

    /**
     * Reset scalefactors by partition
     *
     * This function resets a cumulative scale buffer.
     *
     * @param cumulativeScaleIndex    	Index number of cumulative scaleBuffer (input)
     * @param partitionIndex            Index number of partition (input)
     */
    void resetScaleFactorsByPartition(int cumulativeScaleIndex, int partitionIndex);

    /**
     * Calculate site log likelihoods at a root node
     *
     * This function integrates a list of partials at a node with respect to a set of partials-weights and
     * state frequencies to return the log likelihoods for each site
     *
     * @param bufferIndices             List of partialsBuffer indices to integrate (input)
     * @param categoryWeightsIndices    List of indices of category weights to apply to each partialsBuffer (input)
     *                                      should be one categoryCount sized set for each of
     *                                      parentBufferIndices
     * @param stateFrequenciesIndices   List of indices of state frequencies for each partialsBuffer (input)
     *                                      should be one set for each of parentBufferIndices
     * @param cumulativeScaleIndices    List of scalingFactors indices to accumulate over (input). There
     *                                      should be one set for each of parentBufferIndices
     * @param count                     Number of partialsBuffer to integrate (input)
     * @param outSumLogLikelihood       Pointer to destination for resulting sum of log likelihoods (output)
     */

    void calculateRootLogLikelihoods(int[] bufferIndices,
                                     int[] categoryWeightsIndices,
                                     int[] stateFrequenciesIndices,
                                     int[] cumulativeScaleIndices,
                                     int count,
                                     double[] outSumLogLikelihood);

    /**
     * Calculate site log likelihoods at a root node by partition
     *
     * This function integrates a list of partials at a node with respect to a set of partials-weights and
     * state frequencies to return the log likelihoods for each site
     *
     * @param bufferIndices             List of partialsBuffer indices to integrate (input)
     * @param categoryWeightsIndices    List of indices of category weights to apply to each partialsBuffer (input)
     *                                      should be one categoryCount sized set for each of
     *                                      parentBufferIndices
     * @param stateFrequenciesIndices   List of indices of state frequencies for each partialsBuffer (input)
     *                                      should be one set for each of parentBufferIndices
     * @param cumulativeScaleIndices    List of scalingFactors indices to accumulate over (input). There
     *                                      should be one set for each of parentBufferIndices
     * @param partitionIndices          List of partition indices indicating which sites in each 
     *                                  partialsBuffer should be used (input). There should be one 
     *                                  index for each of bufferIndices
     * @param partitionCount            Number of partialsBuffer to integrate (input)
     * @param count                     Number of sets of partitions to integrate across (input)
     * @param outSumLogLikelihoodByPartition     Pointer to destination for resulting sum of per partition log likelihoods (output)
     * @param outSumLogLikelihood       Pointer to destination for resulting sum of log likelihoods (output)
     */

    void calculateRootLogLikelihoodsByPartition(int[] bufferIndices,
                                     int[] categoryWeightsIndices,
                                     int[] stateFrequenciesIndices,
                                     int[] cumulativeScaleIndices,
                                     int[] partitionIndices,
                                     int partitionCount,
                                     int count,
                                     double[] outSumLogLikelihoodByPartition,
                                     double[] outSumLogLikelihood);

    /**
     * Calculate site log likelihoods and derivatives along an edge
     *
     * This function integrates at list of partials at a parent and child node with respect
     * to a set of partials-weights and state frequencies to return the log likelihoods
     * and first and second derivatives for each site
     *
     * @param parentBufferIndices       List of indices of parent partialsBuffers (input)
     * @param childBufferIndices        List of indices of child partialsBuffers (input)
     * @param probabilityIndices        List indices of transition probability matrices for this edge (input)
     * @param firstDerivativeIndices    List indices of first derivative matrices (input)
     * @param secondDerivativeIndices   List indices of second derivative matrices (input)
     * @param categoryWeightsIndices    List of indices of category weights to apply to each partialsBuffer (input)
     * @param stateFrequenciesIndices   List of indices of state frequencies for each partialsBuffer (input)
     *                                      There should be one set for each of parentBufferIndices
     * @param cumulativeScaleIndices    List of scalingFactors indices to accumulate over (input). There
     *                                      There should be one set for each of parentBufferIndices
     * @param count                     Number of partialsBuffers (input)
     * @param outSumLogLikelihood       Pointer to destination for resulting sum of log likelihoods (output)
     * @param outSumFirstDerivative     Pointer to destination for resulting sum of first derivatives (output)
     * @param outSumSecondDerivative    Pointer to destination for resulting sum of second derivatives (output)
     */

    /*void calculateEdgeLogLikelihoods(int[] parentBufferIndices,
                                     int[] childBufferIndices,
                                     int[] probabilityIndices,
                                     int[] firstDerivativeIndices,
                                     int[] secondDerivativeIndices,
                                     int[] categoryWeightsIndices,
                                     int[] stateFrequenciesIndices,
                                     int[] cumulativeScaleIndices,
                                     int count,
                                     double[] outSumLogLikelihood,
                                     double[] outSumFirstDerivative,
                                     double[] outSumSecondDerivative);*/

    /**
     * Return the individual log likelihoods for each site pattern.
     *
     * @param outLogLikelihoods an array in which the likelihoods will be put
     */
    void getSiteLogLikelihoods(double[] outLogLikelihoods);

    /**
     * Get a details class for this instance
     * @return
     */
    public InstanceDetails getDetails();
}