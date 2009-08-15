package beagle;

public class BeagleNative4StateImpl extends GeneralBeagleImpl {

    public static final boolean DEBUG = false;

    public BeagleNative4StateImpl(final int tipCount, final int partialsBufferCount, final int compactBufferCount, final int patternCount, final int eigenBufferCount, final int matrixBufferCount, final int categoryCount, final int scaleBufferCount) {
        super(tipCount, partialsBufferCount, compactBufferCount, 4, patternCount, eigenBufferCount, matrixBufferCount, categoryCount, scaleBufferCount);
//        Logger.getLogger("beagle").info("Constructing double-precision 4-state Java BEAGLE implementation.");
    }

    protected void updateStatesStates(int bufferIndex1, int matrixIndex1, int bufferIndex2, int matrixIndex2, int bufferIndex3)
    {
        double[] matrices1 = matrices[matrixIndex1];
        double[] matrices2 = matrices[matrixIndex2];

        int[] states1 = tipStates[bufferIndex1];
        int[] states2 = tipStates[bufferIndex2];

        double[] partials3 = partials[bufferIndex3];

        calculateStatesStates(states1, matrices1, states2, matrices2, patternCount, categoryCount, partials3);
    }

    protected void updateStatesPartials(int bufferIndex1, int matrixIndex1, int bufferIndex2, int matrixIndex2, int bufferIndex3)
    {
        double[] matrices1 = matrices[matrixIndex1];
        double[] matrices2 = matrices[matrixIndex2];

        int[] states1 = tipStates[bufferIndex1];
        double[] partials2 = partials[bufferIndex2];

        double[] partials3 = partials[bufferIndex3];

        calculateStatesPartials(states1, matrices1, partials2, matrices2, patternCount, categoryCount, partials3);
    }

    protected void updatePartialsPartials(int bufferIndex1, int matrixIndex1, int bufferIndex2, int matrixIndex2, int bufferIndex3)
    {
        double[] matrices1 = matrices[matrixIndex1];
        double[] matrices2 = matrices[matrixIndex2];

        double[] partials1 = partials[bufferIndex1];
        double[] partials2 = partials[bufferIndex2];

        double[] partials3 = partials[bufferIndex3];

        calculatePartialsPartials(partials1, matrices1, partials2, matrices2, patternCount, categoryCount, partials3);
    }

    @Override
    public void calculateRootLogLikelihoods(int[] bufferIndices, double[] weights, double[] stateFrequencies, int[] scaleIndices, int count, double[] outLogLikelihoods) {

        double[] rootPartials = partials[bufferIndices[0]];
        calculateLogLikelihoods(rootPartials,  weights, stateFrequencies, patternCount,  categoryCount, tmpPartials, outLogLikelihoods);
    }

    // these extra calls provide direct access to the low-level calculations
    protected native void calculateStatesStates(    int[] states1, double[] matrices1,
                                                    int[] states2, double[] matrices2,
                                                    int patternCount, int categoryCount,
                                                    double[] partials3);
    protected native void calculateStatesPartials(	int[] states1, double[] matrices1,
                                                      double[] partials2, double[] matrices2,
                                                      int patternCount, int categoryCount,
                                                      double[] partials3);
    protected native void calculatePartialsPartials(double[] partials1, double[] matrices1,
                                                    double[] partials2, double[] matrices2,
                                                    int patternCount, int categoryCount,
                                                    double[] partials3);
    public native void calculateLogLikelihoods(	double[] partials, double[] weights,
                                                   double[] stateFrequencies,
                                                   int patternCount, int categoryCount,
                                                   double[] outPartials, double[] outLogLikelihoods);

    public static boolean isAvailable() { return isNativeAvailable; }

    private static boolean isNativeAvailable = false;

    static {

        try {
            System.loadLibrary(BeagleJNIWrapper.LIBRARY_NAME);

//            System.err.println(BeagleJNIWrapper.LIBRARY_NAME + " found");

            isNativeAvailable = true;
        } catch (UnsatisfiedLinkError e) {
//            System.err.println(BeagleJNIWrapper.LIBRARY_NAME + " not found");
        }

    }
}