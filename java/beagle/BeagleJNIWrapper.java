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

public class BeagleJNIWrapper {
    public static final String LIBRARY_NAME = "BEAGLE";

    /**
     * private constructor to enforce singleton instance
     */
    private BeagleJNIWrapper() {
    }


    //    public native ResourceList* getResourceList();
//
//    public native int initializeInstance(
//						int instance,
//						InstanceDetails* returnInfo);

    public native int createInstance(
            int tipCount,
            int partialsBufferCount,
            int compactBufferCount,
            int stateCount,
            int patternCount,
            int eigenBufferCount,
            int matrixBufferCount,
            int categoryCount,
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

    public static void loadBeagleLibrary() throws UnsatisfiedLinkError {
        try {
            System.loadLibrary(LIBRARY_NAME);
        } catch (UnsatisfiedLinkError e) {
            throw new UnsatisfiedLinkError("Failed to load BEAGLE library:" + e.getMessage());
        }
        INSTANCE = new BeagleJNIWrapper();
    }

    public static BeagleJNIWrapper INSTANCE;
}