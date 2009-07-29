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

public class BeagleJNIImpl implements Beagle {

    private final int instance;

    public BeagleJNIImpl(int tipCount,
                         int partialsBufferCount,
                         int compactBufferCount,
                         int stateCount,
                         int patternCount,
                         int eigenBufferCount,
                         int matrixBufferCount,
                         final int[] resourceList,
                         int preferenceFlags,
                         int requirementFlags) {
        this.instance = BeagleJNIWrapper.INSTANCE.createInstance(
                tipCount,
                partialsBufferCount,
                compactBufferCount,
                stateCount,
                patternCount,
                eigenBufferCount,
                matrixBufferCount,
                resourceList,
                resourceList.length,
                preferenceFlags,
                requirementFlags);
        BeagleJNIWrapper.INSTANCE.initialize(instance);
    }

    public void finalize() {
        BeagleJNIWrapper.INSTANCE.finalize(instance);
    }

    public void setPartials(int bufferIndex, final double[] partials) {
        BeagleJNIWrapper.INSTANCE.setPartials(instance, bufferIndex, partials);
    }

    public void getPartials(int bufferIndex, final double []outPartials) {
        BeagleJNIWrapper.INSTANCE.getPartials(instance, bufferIndex, outPartials);
    }

    public void setTipStates(int tipIndex, final int[] states) {
        BeagleJNIWrapper.INSTANCE.setTipStates(instance, tipIndex, states);
    }

    public void setEigenDecomposition(int eigenIndex,
                                      final double[] eigenVectors,
                                      final double[] inverseEigenValues,
                                      final double[] eigenValues) {
        BeagleJNIWrapper.INSTANCE.setEigenDecomposition(instance, eigenIndex, eigenVectors, inverseEigenValues, eigenValues);
    }

    public void setTransitionMatrix(int matrixIndex, final double[] inMatrix) {
        BeagleJNIWrapper.INSTANCE.setTransitionMatrix(instance, matrixIndex, inMatrix);
    }


    public void updateTransitionMatrices(int eigenIndex,
                                         final int[] probabilityIndices,
                                         final int[] firstDerivativeIndices,
                                         final int[] secondDervativeIndices,
                                         final double[] edgeLengths,
                                         int count) {
        BeagleJNIWrapper.INSTANCE.updateTransitionMatrices(instance,
                eigenIndex, probabilityIndices,
                firstDerivativeIndices, secondDervativeIndices,
                edgeLengths, count);
    }

    public void updatePartials(final int[] operations, final int operationCount, final boolean rescale) {
        int[] instances = { instance };
        BeagleJNIWrapper.INSTANCE.updatePartials(instances, instances.length, operations, operationCount, rescale ? 1 : 0);
    }

    public void calculateRootLogLikelihoods(final int[] bufferIndices, final double[] weights, final double[] stateFrequencies, final int count, final double[] outLogLikelihoods) {
        BeagleJNIWrapper.INSTANCE.calculateRootLogLikelihoods(instance, bufferIndices, weights, stateFrequencies, bufferIndices.length, outLogLikelihoods);
    }

    public void calculateEdgeLogLikelihoods(final int[] parentBufferIndices, final int[] childBufferIndices, final int[] probabilityIndices, final int[] firstDerivativeIndices, final int[] secondDerivativeIndices, final double[] weights, final double[] stateFrequencies, final int count, final double[] outLogLikelihoods, final double[] outFirstDerivatives, final double[] outSecondDerivatives) {
//        BeagleJNIWrapper.INSTANCE.calculateEdgeLogLikelihoods(instance, bufferIndices, weights, stateFrequencies, bufferIndices.length, outLogLikelihoods);
    }


}