/*
 * BeagleJNIjava
 *
 */

package beagle;

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
                         int categoryCount,
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
                categoryCount,
                resourceList,
                (resourceList != null? resourceList.length: 0),
                preferenceFlags,
                requirementFlags);

        InstanceDetails[] details = new InstanceDetails[1];

        BeagleJNIWrapper.INSTANCE.initializeInstance(instance, details);
    }

    public void finalize() throws Throwable {
        super.finalize();
        int errCode = BeagleJNIWrapper.INSTANCE.finalize(instance);
        assert(errCode == 0);
    }

    public void setPartials(int bufferIndex, final double[] partials) {
        int errCode = BeagleJNIWrapper.INSTANCE.setPartials(instance, bufferIndex, partials);
        assert(errCode == 0);
    }

    public void getPartials(int bufferIndex, final double []outPartials) {
        int errCode = BeagleJNIWrapper.INSTANCE.getPartials(instance, bufferIndex, outPartials);
        assert(errCode == 0);
    }

    public void setTipStates(int tipIndex, final int[] states) {
       int errCode = BeagleJNIWrapper.INSTANCE.setTipStates(instance, tipIndex, states);
        assert(errCode == 0);
    }

    public void setEigenDecomposition(int eigenIndex,
                                      final double[] eigenVectors,
                                      final double[] inverseEigenValues,
                                      final double[] eigenValues) {
        int errCode = BeagleJNIWrapper.INSTANCE.setEigenDecomposition(instance, eigenIndex, eigenVectors, inverseEigenValues, eigenValues);
        assert(errCode == 0);
    }

    public void setCategoryRates(double[] inCategoryRates) {
        int errCode = BeagleJNIWrapper.INSTANCE.setCategoryRates(instance, inCategoryRates);
        assert(errCode == 0);
    }

    public void setTransitionMatrix(int matrixIndex, final double[] inMatrix) {
        int errCode = BeagleJNIWrapper.INSTANCE.setTransitionMatrix(instance, matrixIndex, inMatrix);
        assert(errCode == 0);
    }


    public void updateTransitionMatrices(int eigenIndex,
                                         final int[] probabilityIndices,
                                         final int[] firstDerivativeIndices,
                                         final int[] secondDervativeIndices,
                                         final double[] edgeLengths,
                                         int count) {
        int errCode = BeagleJNIWrapper.INSTANCE.updateTransitionMatrices(instance,
                eigenIndex, probabilityIndices,
                firstDerivativeIndices, secondDervativeIndices,
                edgeLengths, count);
        assert(errCode == 0);
    }

    public void updatePartials(final int[] operations, final int operationCount, final boolean rescale) {
        int[] instances = { instance };
        int errCode = BeagleJNIWrapper.INSTANCE.updatePartials(instances, instances.length, operations, operationCount, rescale ? 1 : 0);
    }

    public void calculateRootLogLikelihoods(int[] bufferIndices,
                                            double[] weights,
                                            double[] stateFrequencies,
                                            int[] scalingFactorsIndices,
                                            int[] scalingFactorsCount,
                                            double[] outLogLikelihoods) {
        int errCode = BeagleJNIWrapper.INSTANCE.calculateRootLogLikelihoods(instance, bufferIndices, weights, stateFrequencies, scalingFactorsIndices, scalingFactorsCount, bufferIndices.length, outLogLikelihoods);
        assert(errCode == 0);
    }

    public void calculateEdgeLogLikelihoods(final int[] parentBufferIndices, 
                                            final int[] childBufferIndices,
                                            final int[] probabilityIndices,
                                            final int[] firstDerivativeIndices,
                                            final int[] secondDerivativeIndices,
                                            final double[] weights,
                                            final double[] stateFrequencies,
                                            final int[] scalingFactorsIndices,
                                            final int[] scalingFactorsCount,
                                            final double[] outLogLikelihoods,
                                            final double[] outFirstDerivatives,
                                            final double[] outSecondDerivatives) {
        int errCode = BeagleJNIWrapper.INSTANCE.calculateEdgeLogLikelihoods(instance,
                parentBufferIndices,
                childBufferIndices,
                probabilityIndices,
                firstDerivativeIndices,
                secondDerivativeIndices,
                weights,
                stateFrequencies,
                scalingFactorsIndices,
                scalingFactorsCount,
                parentBufferIndices.length,
                outLogLikelihoods,
                outFirstDerivatives,
                outSecondDerivatives);
        assert(errCode == 0);
    }


}