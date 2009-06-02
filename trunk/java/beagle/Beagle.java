/*
 * Beagle.java
 *
 */

package beagle;

/**
 * Beagle - An interface exposing the BEAGLE likelihood evaluation library.
 *
 * @author Andrew Rambaut
 * @author Marc A. Suchard
 * @version $Id:$
 */

public interface Beagle {

    public void initialize(int tipCount,
                           int partialsBufferCount,
                           int compactBufferCount,
                           int stateCount,
                           int patternCount,
                           int eigenBufferCount,
                           int matrixBufferCount,
                           final int[] resourceList,
                           int resourceCount,
                           int preferenceFlags,
                           int requirementFlags);

    public void finalize() throws Throwable;

    public void setPartials(int bufferIndex, final double[] partials);

    public void setTipStates(int tipIndex, final int[] states);

    public void setEigenDecomposition(int eigenIndex,
                                      final double[] eigenVectors,
                                      final double[] inverseEigenValues,
                                      final double[] eigenValues);

    public void setTransitionMatrix(int matrixIndex, final double[] inMatrix);


    public void updateTransitionMatrices(int eigenIndex,
                                         final int[] probabilityIndices,
                                         final int[] firstDerivativeIndices,
                                         final int[] secondDervativeIndices,
                                         final double[] edgeLengths,
                                         int count);

    public void updatePartials(final int[] operations, int operationCount, boolean rescale);

    public void calculateRootLogLikelihoods(final int[] bufferIndices,
                                            final double[] weights,
                                            final double[] stateFrequencies,
                                            double[] outLogLikelihoods);
}