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
                            int requirementFlags) {

        // create instance and store instance handle
        instance = createInstance(tipCount,
                partialsBufferCount, compactBufferCount,
                stateCount, patternCount, eigenBufferCount, matrixBufferCount,
                resourceList, resourceCount, preferenceFlags, requirementFlags);
    }

    public void finalize() throws Throwable {
        finalize(instance);
    }

    public void setPartials(int bufferIndex, final double[] partials) {
        setPartials(instance, bufferIndex, partials);
    }

    public void setTipStates(int tipIndex, final int[] states) {
        setTipStates(instance, tipIndex, states);
    }

    public void setEigenDecomposition(int eigenIndex,
                                      final double[] eigenVectors,
                                      final double[] inverseEigenValues,
                                      final double[] eigenValues) {
        setEigenDecomposition(instance, eigenIndex, eigenVectors, inverseEigenValues, eigenValues);
    }

    public void setTransitionMatrix(int matrixIndex, final double[] inMatrix) {
        setTransitionMatrix(instance, matrixIndex, inMatrix);
    }


    public void updateTransitionMatrices(int eigenIndex,
                                         final int[] probabilityIndices,
                                         final int[] firstDerivativeIndices,
                                         final int[] secondDervativeIndices,
                                         final double[] edgeLengths,
                                         int count) {
        updateTransitionMatrices(instance,
                eigenIndex, probabilityIndices,
                firstDerivativeIndices, secondDervativeIndices,
                edgeLengths, count);
    }

    public void updatePartials(final int[] operations, int operationCount, boolean rescale) {
        int[] instances = { instance };
        updatePartials(instances, instances.length, operations, operationCount, rescale ? 1 : 0);
    }

    public void calculateRootLogLikelihoods(final int[] bufferIndices,
                                            final double[] weights,
                                            final double[] stateFrequencies,
                                            double[] outLogLikelihoods) {
        calculateRootLogLikelihoods(instance, bufferIndices, weights, stateFrequencies, bufferIndices.length, outLogLikelihoods);
    }

//    public native ResourceList* getResourceList();
//
//    public native int initializeInstance(
//						int instance,
//						InstanceDetails* returnInfo);

    public native int createInstance(int tipCount,
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

    public native int initialize(int instance);
    
    public native int finalize(int instance);

    public native int setPartials(int instance, int bufferIndex, final double[] inPartials);

    public native int getPartials(int instance, int bufferIndex, final double[] outPartials);

    public native void setTipStates(int instance, int tipIndex, final int[] inStates);

    public native void setEigenDecomposition(int instance,
                                             int eigenIndex,
                                             final double[] eigenVectors,
                                             final double[] inverseEigenValues,
                                             final double[] eigenValues);

    public native int setTransitionMatrix(int instance, int matrixIndex, final double[] inMatrix);

    public native int updateTransitionMatrices(int instance, int eigenIndex,
                                               final int[] probabilityIndices,
                                               final int[] firstDerivativeIndices,
                                               final int[] secondDervativeIndices,
                                               final double[] edgeLengths,
                                               int count);

    public native int updatePartials(final int[] instance,
                                     int instanceCount,
                                     final int[] operations,
                                     int operationCount,
                                     int rescale);

    public native int calculateRootLogLikelihoods(int instance,
                                                  final int[] bufferIndices,
                                                  final double[] weights,
                                                  final double[] stateFrequencies,
                                                  int count,
                                                  double[] outLogLikelihoods);

    /* Library loading routines */

    public static class BeagleLoader implements BeagleFactory.BeagleLoader {

        public String getLibraryName(Map<String, Object> configuration) {
            return LIBRARY_NAME;
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