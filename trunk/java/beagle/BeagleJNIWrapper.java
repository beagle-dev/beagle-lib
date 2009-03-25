/*
 * BeagleJNIjava
 *
 */

package beagle;

import java.util.Map;

/*
 * BeagleJNIjava
 *
 * @author Andrew Rambaut
 *
 */

public class BeagleJNIWrapper implements Beagle {
    public static final String LIBRARY_NAME = "BEAGLE";

    private int instance;
    private final int stateCount;

    public BeagleJNIWrapper(int stateCount) {
        this.stateCount = stateCount;
    }

    public boolean canHandleTipPartials() {
        return true;
    }

    public boolean canHandleTipStates() {
        return true;
    }

    public boolean canHandleDynamicRescaling() {
        return true;
    }

    public void initialize(int nodeCount, int tipCount, int patternCount, int categoryCount, int matrixCount) {
        instance = initialize(nodeCount, tipCount, stateCount, patternCount, categoryCount, matrixCount);
    }

    public void finalize() throws Throwable {
        finalize(instance);
    }

    public void setTipPartials(int tipIndex, double[] partials) {
        setTipPartials(instance, tipIndex, partials);
    }

    public void setTipStates(int tipIndex, int[] states) {
        setTipStates(instance, tipIndex, states);
    }

    public void setStateFrequencies(double[] stateFrequencies) {
        setStateFrequencies(instance, stateFrequencies);
    }

    public void setEigenDecomposition(int matrixIndex, double[][] eigenVectors, double[][] inverseEigenValues, double[] eigenValues) {
        setEigenDecomposition(instance, matrixIndex, eigenVectors, inverseEigenValues, eigenValues);
    }

    public void setCategoryRates(double[] categoryRates) {
        setCategoryRates(instance, categoryRates);
    }

    public void setCategoryProportions(double[] categoryProportions) {
        setCategoryProportions(instance, categoryProportions);
    }

    public void calculateProbabilityTransitionMatrices(int[] nodeIndices, double[] branchLengths, int count) {
        calculateProbabilityTransitionMatrices(instance, nodeIndices, branchLengths, count);
    }

    public void calculatePartials(int[] operations, int[] dependencies, int operationCount) {
        calculatePartials(instance, operations, dependencies, operationCount);
    }

    public void calculateLogLikelihoods(int rootNodeIndex, double[] outLogLikelihoods) {
        calculateLogLikelihoods(instance, rootNodeIndex, outLogLikelihoods);
    }

    public void storeState() {
        storeState(instance);
    }

    public void restoreState() {
        restoreState(instance);
    }

    public native int initialize(
            int nodeCount,
            int tipCount,
            int stateCount,
            int patternCount,
            int categoryCount,
            int matrixCount);

    public native void finalize(int instance);

    public native void setTipPartials(int instance, int tipIndex, double[] partials);

    public native void setTipStates(int instance, int tipIndex, int[] states);

    public native void setStateFrequencies(int instance, double[] stateFrequencies);

    public native void setEigenDecomposition(
            int instance,
            int matrixIndex,
            double[][] eigenVectors,
            double[][] inverseEigenValues,
            double[] eigenValues);

    public native void setCategoryRates(int instance, double[] categoryRates);

    public native void setCategoryProportions(int instance, double[] categoryProportions);

    public native void calculateProbabilityTransitionMatrices(int instance, int[] nodeIndices, double[] branchLengths, int count);

    public native void calculatePartials(int instance, int[] operations, int[] dependencies, int operationCount);

    public native void calculateLogLikelihoods(int instance, int rootNodeIndex, double[] outLogLikelihoods);

    public native void storeState(int instance);

    public native void restoreState(int instance);

    /* Library loading routines */

    public static class BeagleLoader implements BeagleFactory.BeagleLoader {

        public String getLibraryName(Map<String, Object> configuration) {
            int stateCount = (Integer)configuration.get(BeagleFactory.STATE_COUNT);
            return LIBRARY_NAME + "-" + stateCount;
        }

        public Beagle createInstance(Map<String, Object> configuration) {
            try {
                String name = getLibraryName(configuration);
                System.loadLibrary(name);
            } catch (UnsatisfiedLinkError e) {
                return null;
            }
            
            int stateCount = (Integer)configuration.get(BeagleFactory.STATE_COUNT);

            return new BeagleJNIWrapper(stateCount);
        }
    }
}