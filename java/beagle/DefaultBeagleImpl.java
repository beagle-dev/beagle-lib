package beagle;

/**
 * @author Andrew Rambaut
 * @version $Id$
 */
public class DefaultBeagleImpl implements Beagle {
    public DefaultBeagleImpl(final int tipCount, final int partialsBufferCount, final int compactBufferCount, final int stateCount, final int patternCount, final int eigenBufferCount, final int matrixBufferCount) {
        this.tipCount = tipCount;
        this.partialsBufferCount = partialsBufferCount;
        this.compactBufferCount = compactBufferCount;
        this.stateCount = stateCount;
        this.patternCount = patternCount;
        this.eigenBufferCount = eigenBufferCount;
        this.matrixBufferCount = matrixBufferCount;
    }

    public void finalize() {
    }

    public void setPartials(final int bufferIndex, final double[] inPartials) {
    }

    public void getPartials(final int bufferIndex, final double[] outPartials) {
    }

    public void setTipStates(final int tipIndex, final int[] inStates) {
    }

    public void setEigenDecomposition(final int eigenIndex, final double[] inEigenVectors, final double[] inInverseEigenVectors, final double[] inEigenValues) {
    }

    public void setTransitionMatrix(final int matrixIndex, final double[] inMatrix) {
    }

    public void updateTransitionMatrices(final int eigenIndex, final int[] probabilityIndices, final int[] firstDerivativeIndices, final int[] secondDervativeIndices, final double[] edgeLengths, final int count) {
    }

    public void updatePartials(final int[] operations, final int operationCount, final boolean rescale) {
    }

    public void calculateRootLogLikelihoods(final int[] bufferIndices, final double[] weights, final double[] stateFrequencies, final double[] outLogLikelihoods) {
    }

    int tipCount;

    int partialsBufferCount;
    int compactBufferCount;
    int stateCount;
    int patternCount;
    int eigenBufferCount;
    int matrixBufferCount;
}
