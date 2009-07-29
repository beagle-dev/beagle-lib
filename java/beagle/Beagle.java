/*
 * Beagle.java
 *
 */

package beagle;

import java.util.List;

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

public interface Beagle {


    /**
     * Finalize this instance
     *
     * This function finalizes the instance by releasing allocated memory
     */
    void finalize();

    /**
     * Set an instance partials buffer
     *
     * This function copies an array of partials into an instance buffer.
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
     * This function copies an instance buffer into the array outPartials
     *
     * @param bufferIndex   Index of destination partialsBuffer (input)
     * @param  outPartials  Pointer to which to receive partialsBuffer (output)
     */
    void getPartials(
            int bufferIndex,
            final double []outPartials);

    /**
     * Set the compressed state representation for tip node
     *
     * This function copies a compressed state representation into a instance buffer.
     * Compressed state representation is an array of states: 0 to stateCiunt - 1 (missing = stateCount)
     *
     * @param tipIndex   Index of destination partialsBuffer (input)
     * @param inStates   Pointer to compressed states (input)
     */
    void setTipStates(
            int tipIndex,
            final int[] inStates);

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
     * Set category rates
     *
     * This function sets the vector of category rates for an instance.
     *
     * @param inCategoryRates       Array containing categoryCount rate scalers (input)
     */
    void setCategoryRates(final double[] inCategoryRates);

    /**
     * Set a finite-time transition probability matrix
     *
     * This function copies a finite-time transition probability matrix into a matrix buffer.
     * @param matrixIndex   Index of matrix buffer (input)
     * @param inMatrix          Pointer to source transition probability matrix (input)
     */
    void setTransitionMatrix(
            int matrixIndex,			/**< Index of matrix buffer (input) */
            final double[] inMatrix);	/**< Pointer to source transition probability matrix (input) */

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
     * Calculate or queue for calculation partials using a list of operations
     *
     * This function either calculates or queues for calculation a list partials. Implementations
     * supporting SYNCH may queue these calculations while other implementations perform these
     * operations immediately.  Implementations supporting GPU may perform all operations in the list
     * simultaneously.
     *
     * Operations list is a list of 6-tuple integer indices, with one 6-tuple per operation.
     * Format of 6-tuple operation: {destinationPartials,
     *                               destinationScalingFactors, (this index must be > tipCount)
     *                               child1Partials,
     *                               child1TransitionMatrix,
     *                               child2Partials,
     *                               child2TransitionMatrix}
     *
     * @param operations        List of 6-tuples specifying operations (input)
     * @param operationCount    Number of operations (input)
     * @param rescale           Specify whether (=1) or not (=0) to recalculate scaling factors
     *
     */
    void updatePartials(
            final int[] operations,
            int operationCount,
            boolean rescale);

    /**
     * Calculate site log likelihoods at a root node
     *
     * This function integrates a list of partials at a node with respect to a set of partials-weights and
     * state frequencies to return the log likelihoods for each site
     *
     * @param bufferIndices         List of partialsBuffer indices to integrate (input)
     * @param weights             List of weights to apply to each partialsBuffer (input). There
     *                               should be one categoryCount sized set for each of
     *                               parentBufferIndices
     * @param stateFrequencies    List of state frequencies for each partialsBuffer (input). There
     *                               should be one set for each of parentBufferIndices
     * @param scalingFactorsIndices List of scalingFactors indices to accumulate over (input). There
     *                               should be one set for each of parentBufferIndices
     * @param scalingFactorsCount   List of scalingFactorsIndices sizes for each partialsBuffer (input)
     * @param outLogLikelihoods     Pointer to destination for resulting log likelihoods (output)
     */
    void calculateRootLogLikelihoods(
            final int[] bufferIndices,
            final double[] weights,
            final double[] stateFrequencies,
            final int[] scalingFactorsIndices,
            final int[] scalingFactorsCount,
            final double[] outLogLikelihoods);

    /*
    * Calculate site log likelihoods and derivatives along an edge
    *
    * This function integrates at list of partials at a parent and child node with respect
    * to a set of partials-weights and state frequencies to return the log likelihoods
    * and first and second derivatives for each site
    *
    * @param parentBufferIndices List of indices of parent partialsBuffers (input)
    * @param childBufferIndices        List of indices of child partialsBuffers (input)
    * @param probabilityIndices        List indices of transition probability matrices for this edge (input)
    * @param firstDerivativeIndices    List indices of first derivative matrices (input)
    * @param secondDerivativeIndices   List indices of second derivative matrices (input)
    * @param weights                   List of weights to apply to each partialsBuffer (input)
    * @param stateFrequencies          List of state frequencies for each partialsBuffer (input)
    *                                      There should be one set for each of parentBufferIndices
    * @param count                     Number of partialsBuffers (input)
    * @param outLogLikelihoods         Pointer to destination for resulting log likelihoods (output)
    * @param outFirstDerivatives       Pointer to destination for resulting first derivatives (output)
    * @param outSecondDerivatives      Pointer to destination for resulting second derivatives (output)
    */
//    void calculateEdgeLogLikelihoods(
//            final int[] parentBufferIndices,
//            final int[] childBufferIndices,
//            final int[] probabilityIndices,
//            final int[] firstDerivativeIndices,
//            final int[] secondDerivativeIndices,
//            final double[] weights,
//            final double[] stateFrequencies,
//            int count,
//            final double[] outLogLikelihoods,
//            final double[] outFirstDerivatives,
//            final double[] outSecondDerivatives);

}