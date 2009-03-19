/*
 * BeagleJNIWrapper.java
 *
 */

package beagle;

import java.util.Map;

/*
 * BeagleJNIWrapper.java
 *
 * @author Andrew Rambaut
 *
 */

public class BeagleJNIWrapper implements Beagle {
    public static final String LIBRARY_NAME = "BEAGLE";

    public boolean canHandleTipPartials() {
        return true;
    }

    public boolean canHandleTipStates() {
        return true;
    }

    public boolean canHandleDynamicRescaling() {
        return true;
    }

    public native void initialize(
								  int nodeCount,
								  int tipCount,
								  int patternCount,
								  int categoryCount,
								  int matrixCount);

    public native void finalize();

    public native void setTipPartials(int tipIndex, double[] partials);

    public native void setTipStates(int tipIndex, int[] states);

	public native void setStateFrequencies(double[] stateFrequencies);

    public native void setEigenDecomposition(
											 int matrixIndex,
											 double[][] eigenVectors,
											 double[][] inverseEigenValues,
											 double[] eigenValues);

    public native void setCategoryRates(double[] categoryRates);

    public native void setCategoryProportions(double[] categoryProportions);

    public native void calculateProbabilityTransitionMatrices(int nodeIndex, double branchLength);

    public native void calculatePartials(int[] operations, int[] dependencies, int operationCount);

    public native void calculateLogLikelihoods(int rootNodeIndex, double[] outLogLikelihoods);

    public native void storeState();

    public native void restoreState();

    /* Library loading routines */

    public static class BeagleLoader implements BeagleFactory.BeagleLoader {

        public String getLibraryName(Map<String, Object> configuration) {
            int stateCount = (Integer)configuration.get(BeagleFactory.STATE_COUNT);
            boolean singlePrecision = (Boolean)configuration.get(BeagleFactory.SINGLE_PRECISION);
            String name = LIBRARY_NAME + "-" + stateCount + (singlePrecision ? "-S": "-D");
            return name;
        }

        public Beagle createInstance(Map<String, Object> configuration) {


            try {
                String name = getLibraryName(configuration);
                System.loadLibrary(name);
            } catch (UnsatisfiedLinkError e) {
                return null;
            }

            return new BeagleJNIWrapper();
        }
    }
}