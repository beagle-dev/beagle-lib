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

    boolean canHandleTipPartials();

    boolean canHandleTipStates();

    boolean canHandleDynamicRescaling();

    public void initialize(
								  int nodeCount,
								  int tipCount,
								  int patternCount,
								  int categoryCount,
								  int matrixCount);

    public void finalize() throws Throwable;

    public void setTipPartials(int tipIndex, double[] partials);

    public void setTipStates(int tipIndex, int[] states);

	public void setStateFrequencies(double[] stateFrequencies);

    public void setEigenDecomposition(
											 int matrixIndex,
											 double[][] eigenVectors,
											 double[][] inverseEigenValues,
											 double[] eigenValues);

    public void setCategoryRates(double[] categoryRates);

    public void setCategoryProportions(double[] categoryProportions);

    public void calculateProbabilityTransitionMatrices(int[] nodeIndices, double[] branchLengths, int count);

    public void calculatePartials(int[] operations, int[] dependencies, int operationCount, boolean rescale);

    public void calculateLogLikelihoods(int rootNodeIndex, double[] outLogLikelihoods);

    public void storeState();

    public void restoreState();
}