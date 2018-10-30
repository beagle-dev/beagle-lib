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

    private int instance = -1;
    private InstanceDetails details = new InstanceDetails();

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

        instance = BeagleJNIWrapper.INSTANCE.createInstance(
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
                requirementFlags,
                details);

        if (instance < 0) {
            details = null; // To communicate that no instance has been created!
            throw new BeagleException("create", instance);
        }
    }

    public void finalize() throws Throwable {
        super.finalize();
        int errCode = BeagleJNIWrapper.INSTANCE.finalize(instance);
        if (errCode != 0) {
            throw new BeagleException("finalize", errCode);
        }
    }

    public void setCPUThreadCount(int threadCount) {
        int errCode = BeagleJNIWrapper.INSTANCE.setCPUThreadCount(instance, threadCount);
        if (errCode != 0) {
            throw new BeagleException("setCPUThreadCount", errCode);
        }
    }

    public void setPatternWeights(final double[] patternWeights) {
        int errCode = BeagleJNIWrapper.INSTANCE.setPatternWeights(instance, patternWeights);
        if (errCode != 0) {
            throw new BeagleException("setPatternWeights", errCode);
        }
    }

    public void setPatternPartitions(int partitionCount, final int[] patternPartitions) {
        int errCode = BeagleJNIWrapper.INSTANCE.setPatternPartitions(instance, partitionCount, patternPartitions);
        if (errCode != 0) {
            throw new BeagleException("setPatternPartitions", errCode);
        }
    }

    public void setTipStates(int tipIndex, final int[] states) {
        int errCode = BeagleJNIWrapper.INSTANCE.setTipStates(instance, tipIndex, states);
        if (errCode != 0) {
            throw new BeagleException("setTipStates", errCode);
        }
    }

    public void getTipStates(int tipIndex, final int[] states) {
        int errCode = BeagleJNIWrapper.INSTANCE.getTipStates(instance, tipIndex, states);
        if (errCode != 0) {
            throw new BeagleException("getTipStates", errCode);
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
    
    public void getLogScaleFactors(int scaleIndex, final double[] outFactors) {
        int errCode = BeagleJNIWrapper.INSTANCE.getLogScaleFactors(instance, scaleIndex, outFactors);
        if (errCode != 0) {
            throw new BeagleException("getScaleFactors", errCode);
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

    public void setStateFrequencies(int stateFrequenciesIndex,
                                    final double[] stateFrequencies) {
        int errCode = BeagleJNIWrapper.INSTANCE.setStateFrequencies(instance,
                stateFrequenciesIndex, stateFrequencies);
        if (errCode != 0) {
            throw new BeagleException("setStateFrequencies", errCode);
        }
    }

    public void setCategoryWeights( int categoryWeightsIndex,
                                    final double[] categoryWeights) {
        int errCode = BeagleJNIWrapper.INSTANCE.setCategoryWeights(instance,
                categoryWeightsIndex, categoryWeights);
        if (errCode != 0) {
            throw new BeagleException("setCategoryWeights", errCode);
        }
    }

    public void setCategoryRates(double[] inCategoryRates) {
        int errCode = BeagleJNIWrapper.INSTANCE.setCategoryRates(instance, inCategoryRates);
        if (errCode != 0) {
            throw new BeagleException("setCategoryRates", errCode);
        }
    }

    public void setCategoryRatesWithIndex( int categoryRatesIndex,
                                           double[] inCategoryRates) {
        int errCode = BeagleJNIWrapper.INSTANCE.setCategoryRatesWithIndex(instance, 
                                                                          categoryRatesIndex,
                                                                          inCategoryRates);
        if (errCode != 0) {
            throw new BeagleException("setCategoryRatesWithIndex", errCode);
        }
    }

    public void setTransitionMatrix(int matrixIndex, final double[] inMatrix, double paddedValue) {
        int errCode = BeagleJNIWrapper.INSTANCE.setTransitionMatrix(instance, matrixIndex, inMatrix, paddedValue);
        if (errCode != 0) {
            throw new BeagleException("setTransitionMatrix", errCode);
        }
    }

    public void getTransitionMatrix(int matrixIndex, final double[] outMatrix) {
        int errCode = BeagleJNIWrapper.INSTANCE.getTransitionMatrix(instance, matrixIndex, outMatrix);
        if (errCode != 0) {
            throw new BeagleException("getTransitionMatrix", errCode);
        }
    }

	// /////////////////////////
	// ---TODO: Epoch model---//
	// /////////////////////////

	public void convolveTransitionMatrices(final int[] firstIndices, 
                                           final int[] secondIndices,
                                           final int[] resultIndices, 
                                           int matrixCount) {

        int errCode = BeagleJNIWrapper.INSTANCE.convolveTransitionMatrices(instance,
                                                                           firstIndices, 
                                                                           secondIndices,
                                                                           resultIndices, 
                                                                           matrixCount);
        if (errCode != 0) {
            throw new BeagleException("convolveTransitionMatrices", errCode);
        }
		
	}//END: convolveTransitionMatrices    
    
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

    public void updateTransitionMatricesWithMultipleModels(
                                         final int[] eigenIndices,
                                         final int[] categoryRateIndices,
                                         final int[] probabilityIndices,
                                         final int[] firstDerivativeIndices,
                                         final int[] secondDervativeIndices,
                                         final double[] edgeLengths,
                                         int count) {
        int errCode = BeagleJNIWrapper.INSTANCE.updateTransitionMatricesWithMultipleModels(
                instance,
                eigenIndices,
                categoryRateIndices,
                probabilityIndices,
                firstDerivativeIndices,
                secondDervativeIndices,
                edgeLengths,
                count);
        if (errCode != 0) {
            throw new BeagleException("updateTransitionMatricesWithMultipleModels", errCode);
        }
    }


    public void updatePartials(final int[] operations, final int operationCount, final int cumulativeScaleIndex) {
        int errCode = BeagleJNIWrapper.INSTANCE.updatePartials(instance, operations, operationCount, cumulativeScaleIndex);
        if (errCode != 0) {
            throw new BeagleException("updatePartials", errCode);
        }
    }

    public void updatePartialsByPartition(final int[] operations, final int operationCount) {
        int errCode = BeagleJNIWrapper.INSTANCE.updatePartialsByPartition(instance, operations, operationCount);
        if (errCode != 0) {
            throw new BeagleException("updatePartialsByPartition", errCode);
        }
    }

    public void accumulateScaleFactors(final int[] scaleIndices, final int count, final int cumulativeScaleIndex) {
        int errCode = BeagleJNIWrapper.INSTANCE.accumulateScaleFactors(instance, scaleIndices, count, cumulativeScaleIndex);
        if (errCode != 0) {
            throw new BeagleException("accumulateScaleFactors", errCode);
        }
    }

    public void accumulateScaleFactorsByPartition(final int[] scaleIndices, final int count, final int cumulativeScaleIndex, final int partitionIndex) {
        int errCode = BeagleJNIWrapper.INSTANCE.accumulateScaleFactorsByPartition(instance, scaleIndices, count, cumulativeScaleIndex, partitionIndex);
        if (errCode != 0) {
            throw new BeagleException("accumulateScaleFactorsByPartition", errCode);
        }
    }

    public void removeScaleFactors(int[] scaleIndices, int count, int cumulativeScaleIndex) {
        int errCode = BeagleJNIWrapper.INSTANCE.removeScaleFactors(instance, scaleIndices, count, cumulativeScaleIndex);
        if (errCode != 0) {
            throw new BeagleException("removeScaleFactors", errCode);
        }
    }

    public void removeScaleFactorsByPartition(int[] scaleIndices, int count, int cumulativeScaleIndex, int partitionIndex) {
        int errCode = BeagleJNIWrapper.INSTANCE.removeScaleFactorsByPartition(instance, scaleIndices, count, cumulativeScaleIndex, partitionIndex);
        if (errCode != 0) {
            throw new BeagleException("removeScaleFactorsByPartition", errCode);
        }
    }

    public void copyScaleFactors(int destScalingIndex, int srcScalingIndex) {
        int errCode = BeagleJNIWrapper.INSTANCE.copyScaleFactors(instance, destScalingIndex, srcScalingIndex);
        if (errCode != 0) {
            throw new BeagleException("copyScaleFactors", errCode);
        }
    }

    public void resetScaleFactors(int cumulativeScaleIndex) {
        int errCode = BeagleJNIWrapper.INSTANCE.resetScaleFactors(instance, cumulativeScaleIndex);
        if (errCode != 0) {
            throw new BeagleException("resetScaleFactors", errCode);
        }
    }

    public void resetScaleFactorsByPartition(int cumulativeScaleIndex, int partitionIndex) {
        int errCode = BeagleJNIWrapper.INSTANCE.resetScaleFactorsByPartition(instance, cumulativeScaleIndex, partitionIndex);
        if (errCode != 0) {
            throw new BeagleException("resetScaleFactorsByPartition", errCode);
        }
    }

    public void calculateRootLogLikelihoods(int[] bufferIndices,
                                            final int[] categoryWeightsIndices,
                                            final int[] stateFrequenciesIndices,
                                            final int[] cumulativeScaleIndices,
                                            int count,
                                            final double[] outSumLogLikelihood) {
        int errCode = BeagleJNIWrapper.INSTANCE.calculateRootLogLikelihoods(instance,
                bufferIndices,
                categoryWeightsIndices,
                stateFrequenciesIndices,
                cumulativeScaleIndices,
                count,
                outSumLogLikelihood);
        // We probably don't want the Floating Point error to throw an exception...
        if (errCode != 0 && errCode != BeagleErrorCode.FLOATING_POINT_ERROR.getErrCode()) {
            throw new BeagleException("calculateRootLogLikelihoods", errCode);
        }
    }

    public void calculateRootLogLikelihoodsByPartition(int[] bufferIndices,
                                            final int[] categoryWeightsIndices,
                                            final int[] stateFrequenciesIndices,
                                            final int[] cumulativeScaleIndices,
                                            final int[] partitionIndices,
                                            int partitionCount,
                                            int count,
                                            final double[] outSumLogLikelihoodByPartition,
                                            final double[] outSumLogLikelihood) {
        int errCode = BeagleJNIWrapper.INSTANCE.calculateRootLogLikelihoodsByPartition(instance,
                bufferIndices,
                categoryWeightsIndices,
                stateFrequenciesIndices,
                cumulativeScaleIndices,
                partitionIndices,
                partitionCount,
                count,
                outSumLogLikelihoodByPartition,
                outSumLogLikelihood);
        // We probably don't want the Floating Point error to throw an exception...
        if (errCode != 0 && errCode != BeagleErrorCode.FLOATING_POINT_ERROR.getErrCode()) {
            throw new BeagleException("calculateRootLogLikelihoodsByPartition", errCode);
        }
    }

    /*public void calculateEdgeLogLikelihoods(final int[] parentBufferIndices,
                                            final int[] childBufferIndices,
                                            final int[] probabilityIndices,
                                            final int[] firstDerivativeIndices,
                                            final int[] secondDerivativeIndices,
                                            final int[] categoryWeightsIndices,
                                            final int[] stateFrequenciesIndices,
                                            final int[] cumulativeScaleIndices,
                                            int count,
                                            final double[] outSumLogLikelihood,
                                            final double[] outSumFirstDerivative,
                                            final double[] outSumSecondDerivative) {
        int errCode = BeagleJNIWrapper.INSTANCE.calculateEdgeLogLikelihoods(instance,
                parentBufferIndices,
                childBufferIndices,
                probabilityIndices,
                firstDerivativeIndices,
                secondDerivativeIndices,
                categoryWeightsIndices,
                stateFrequenciesIndices,
                cumulativeScaleIndices,
                count,
                outSumLogLikelihood,
                outSumFirstDerivative,
                outSumSecondDerivative);
        if (errCode != 0) {
            throw new BeagleException("calculateEdgeLogLikelihoods", errCode);
        }
    }*/

    public void getSiteLogLikelihoods(final double[] outLogLikelihoods) {
        int errCode = BeagleJNIWrapper.INSTANCE.getSiteLogLikelihoods(instance,
                outLogLikelihoods);
        if (errCode != 0) {
            throw new BeagleException("getSiteLogLikelihoods", errCode);
        }
    }

    public InstanceDetails getDetails() {
        return details;
    }
}