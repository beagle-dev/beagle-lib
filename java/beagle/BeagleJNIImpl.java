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

public class BeagleJNIImpl implements Beagle {

    private final int instance;
    private final InstanceDetails details = new InstanceDetails();

    public BeagleJNIImpl(int tipCount,
                         int partialsBufferCount,
                         int compactBufferCount,
                         int stateCount,
                         int patternCount,
                         int eigenBufferCount,
                         int matrixBufferCount,
                         int categoryCount,
                         int scaleBufferCount,
                         final int[] resourceList,
                         long preferenceFlags,
                         long requirementFlags) {

        this.instance = BeagleJNIWrapper.INSTANCE.createInstance(
                tipCount,
                partialsBufferCount,
                compactBufferCount,
                stateCount,
                patternCount,
                eigenBufferCount,
                matrixBufferCount,
                categoryCount,
                scaleBufferCount,
                resourceList,
                (resourceList != null? resourceList.length: 0),
                preferenceFlags,
                requirementFlags);


        BeagleJNIWrapper.INSTANCE.initializeInstance(instance, details);
    }

    public void finalize() throws Throwable {
        super.finalize();
        int errCode = BeagleJNIWrapper.INSTANCE.finalize(instance);
        if (errCode != 0) {
            throw new BeagleException("finalize", errCode);
        }
    }

    public void setTipStates(int tipIndex, final int[] states) {
       int errCode = BeagleJNIWrapper.INSTANCE.setTipStates(instance, tipIndex, states);
        if (errCode != 0) {
            throw new BeagleException("setTipStates", errCode);
        }
    }

    public void setTipPartials(int tipIndex, final double[] partials) {
        int errCode = BeagleJNIWrapper.INSTANCE.setTipPartials(instance, tipIndex, partials);
        if (errCode != 0) {
            throw new BeagleException("setTipPartials", errCode);
        }
    }

    public void setPartials(int bufferIndex, final double[] partials) {
        int errCode = BeagleJNIWrapper.INSTANCE.setPartials(instance, bufferIndex, partials);
        if (errCode != 0) {
            throw new BeagleException("setPartials", errCode);
        }
    }

    public void getPartials(int bufferIndex, int scaleIndex, final double []outPartials) {
        int errCode = BeagleJNIWrapper.INSTANCE.getPartials(instance, bufferIndex, scaleIndex, outPartials);
        if (errCode != 0) {
            throw new BeagleException("getPartials", errCode);
        }
    }

    public void setEigenDecomposition(int eigenIndex,
                                      final double[] eigenVectors,
                                      final double[] inverseEigenValues,
                                      final double[] eigenValues) {
        int errCode = BeagleJNIWrapper.INSTANCE.setEigenDecomposition(instance, eigenIndex, eigenVectors, inverseEigenValues, eigenValues);
        if (errCode != 0) {
            throw new BeagleException("setEigenDecomposition", errCode);
        }
    }

    public void setCategoryRates(double[] inCategoryRates) {
        int errCode = BeagleJNIWrapper.INSTANCE.setCategoryRates(instance, inCategoryRates);
        if (errCode != 0) {
            throw new BeagleException("setCategoryRates", errCode);
        }
    }

    public void setTransitionMatrix(int matrixIndex, final double[] inMatrix) {
        int errCode = BeagleJNIWrapper.INSTANCE.setTransitionMatrix(instance, matrixIndex, inMatrix);
        if (errCode != 0) {
            throw new BeagleException("setTransitionMatrix", errCode);
        }
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
        if (errCode != 0) {
            throw new BeagleException("updateTransitionMatrices", errCode);
        }
    }


    public void updatePartials(final int[] operations, final int operationCount, final int cumulativeScaleIndex) {
        int[] instances = { instance };
        int errCode = BeagleJNIWrapper.INSTANCE.updatePartials(instances, instances.length, operations, operationCount, cumulativeScaleIndex);
        if (errCode != 0) {
            throw new BeagleException("updatePartials", errCode);
        }
    }

    public void accumulateScaleFactors(final int[] scaleIndices, final int count, final int cumulativeScaleIndex) {
        int errCode = BeagleJNIWrapper.INSTANCE.accumulateScaleFactors(instance, scaleIndices, count, cumulativeScaleIndex);
        if (errCode != 0) {
            throw new BeagleException("accumulateScaleFactors", errCode);
        }
    }

    public void removeScaleFactors(int[] scaleIndices, int count, int cumulativeScaleIndex) {
        int errCode = BeagleJNIWrapper.INSTANCE.removeScaleFactors(instance, scaleIndices, count, cumulativeScaleIndex);
        if (errCode != 0) {
            throw new BeagleException("removeScaleFactors", errCode);
        }
    }

    public void resetScaleFactors(int cumulativeScaleIndex) {
        int errCode = BeagleJNIWrapper.INSTANCE.resetScaleFactors(instance, cumulativeScaleIndex);
        if (errCode != 0) {
            throw new BeagleException("resetScaleFactors", errCode);
        }
    }

    public void calculateRootLogLikelihoods(int[] bufferIndices,
                                            double[] inWeights,
                                            double[] inStateFrequencies,
                                            int[] scaleIndices,
                                            int count,
                                            double[] outLogLikelihoods) {
        int errCode = BeagleJNIWrapper.INSTANCE.calculateRootLogLikelihoods(instance, bufferIndices, inWeights,
                inStateFrequencies, scaleIndices, count, outLogLikelihoods);
        if (errCode != 0) {
            throw new BeagleException("calculateRootLogLikelihoods", errCode);
        }
    }

    public void calculateEdgeLogLikelihoods(final int[] parentBufferIndices,
                                            final int[] childBufferIndices,
                                            final int[] probabilityIndices,
                                            final int[] firstDerivativeIndices,
                                            final int[] secondDerivativeIndices,
                                            final double[] weights,
                                            final double[] stateFrequencies,
                                            final int[] scaleIndices,
                                            int count,
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
                scaleIndices,
                count,
                outLogLikelihoods,
                outFirstDerivatives,
                outSecondDerivatives);
        if (errCode != 0) {
            throw new BeagleException("calculateEdgeLogLikelihoods", errCode);
        }
    }

    public InstanceDetails getDetails() {
        return details;
    }
}