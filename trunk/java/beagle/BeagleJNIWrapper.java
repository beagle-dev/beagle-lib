/*
 * BeagleJNIjava
 *
 */

package beagle;


/*
 * BeagleJNIjava
 *
 * @author Andrew Rambaut
 * @author Marc Suchard
 *
 */

public class BeagleJNIWrapper {

    public static final String LIBRARY_NAME = getPlatformSpecificLibraryName();

    /**
     * private constructor to enforce singleton instance
     */
    private BeagleJNIWrapper() {
    }

    public native ResourceDetails[] getResourceList();

    public native int createInstance(
            int tipCount,
            int partialsBufferCount,
            int compactBufferCount,
            int stateCount,
            int patternCount,
            int eigenBufferCount,
            int matrixBufferCount,
            int categoryCount,
            int scaleBufferCount,
            final int[] resourceList,
            int resourceCount,
            long preferenceFlags,
            long requirementFlags);

    public native int initializeInstance(
            int instance,
            InstanceDetails returnInfo);

    public native int finalize(int instance);

    public native int setTipStates(int instance, int tipIndex, final int[] inStates);

    public native int setTipPartials(int instance, int tipIndex, final double[] inPartials);

    public native int setPartials(int instance, int bufferIndex, final double[] inPartials);

    public native int getPartials(int instance, int bufferIndex, int scaleIndex,
                                  final double[] outPartials);


    public native int setEigenDecomposition(int instance,
                                            int eigenIndex,
                                            final double[] eigenVectors,
                                            final double[] inverseEigenValues,
                                            final double[] eigenValues);

    public native int setCategoryRates(int instance, final double[] inCategoryRates);

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
                                     int cumulativeScalingIndex);

    public native int waitForPartials(final int[] instance,
                                      int instanceCount,
                                      final int[] destinationPartials,
                                      int destinationPartialsCount);

    public native int accumulateScaleFactors(final int instance,
                                             final int[] scaleIndices,
                                             final int count,
                                             final int cumulativeScalingIndex);

    public native int removeScaleFactors(final int instance,
                                         final int[] scaleIndices,
                                         final int count,
                                         final int cumulativeScalingIndex);

    public native int resetScaleFactors(final int instance,
                                        final int cumulativeScalingIndex);

    public native int calculateRootLogLikelihoods(int instance,
                                                  final int[] bufferIndices,
                                                  final double[] inWeights,
                                                  final double[] inStateFrequencies,
                                                  final int[] scalingFactorsIndices,
                                                  int count,
                                                  final double[] outLogLikelihoods);

    public native int calculateEdgeLogLikelihoods(int instance,
                                                  final int[] parentBufferIndices,
                                                  final int[] childBufferIndices,
                                                  final int[] probabilityIndices,
                                                  final int[] firstDerivativeIndices,
                                                  final int[] secondDerivativeIndices,
                                                  final double[] inWeights,
                                                  final double[] inStateFrequencies,
                                                  final int[] scalingFactorsIndices,
                                                  int count,
                                                  final double[] outLogLikelihoods,
                                                  final double[] outFirstDerivatives,
                                                  final double[] outSecondDerivatives);

    /* Library loading routines */

    private static final String getPlatformSpecificLibraryName()
    {
        String osName = System.getProperty("os.name").toLowerCase();
        String osArch = System.getProperty("os.arch").toLowerCase();
        if (osName.startsWith("windows")) {
            if(osArch.equals("i386")) return "hmsbeagle32";
            if(osArch.startsWith("amd64")||osArch.startsWith("x86_64")) return "hmsbeagle64";
        }
        return "hmsbeagle-jni";
    }

    public static void loadBeagleLibrary() throws UnsatisfiedLinkError {
        String path = "";
        if (System.getProperty("beagle.library.path") != null) {
            path = System.getProperty("beagle.library.path");
            if (path.length() > 0 && !path.endsWith("/")) {
                path += "/";
            }
        }

        System.loadLibrary(path + LIBRARY_NAME);
        INSTANCE = new BeagleJNIWrapper();
    }

    public static BeagleJNIWrapper INSTANCE;
}
