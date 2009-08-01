package beagle;

import java.util.logging.Logger;

public class GeneralBeagleImpl implements Beagle {

    public static final boolean DEBUG = false;
    public static final boolean SCALING = false;
    public static final boolean DYNAMIC_SCALING = true;

    protected final int tipCount;
    protected final int partialsBufferCount;
    protected final int compactBufferCount;
    protected final int stateCount;
    protected final int patternCount;
    protected final int eigenBufferCount;
    protected final int matrixBufferCount;
    protected final int categoryCount;

    protected int partialsSize;
    protected int matrixSize;

    protected double[][] cMatrices;
    protected double[][] eigenValues;

    protected double[] categoryRates;

    protected double[][] partials;

    protected int[][] tipStates;

    protected double[][] matrices;


//    protected double[][][] scalingFactors;
//    protected double[] rootScalingFactors;
//    protected int[] currentScalingFactorsIndices;

    /**
     * Constructor
     *
     * @param stateCount number of states
     */
    public GeneralBeagleImpl(final int tipCount,
                             final int partialsBufferCount,
                             final int compactBufferCount,
                             final int stateCount,
                             final int patternCount,
                             final int eigenBufferCount,
                             final int matrixBufferCount,
                             final int categoryCount) {

        this.tipCount = tipCount;
        this.partialsBufferCount = partialsBufferCount;
        this.compactBufferCount = compactBufferCount;
        this.stateCount = stateCount;
        this.patternCount = patternCount;
        this.eigenBufferCount = eigenBufferCount;
        this.matrixBufferCount = matrixBufferCount;
        this.categoryCount = categoryCount;

        Logger.getLogger("beagle").info("Constructing double-precision Java BEAGLE implementation.");

        if (patternCount < 1) {
            throw new IllegalArgumentException("Pattern count must be at least 1");
        }

        if (categoryCount < 1) {
            throw new IllegalArgumentException("Category count must be at least 1");
        }

        cMatrices = new double[eigenBufferCount][stateCount * stateCount * stateCount];

        eigenValues = new double[eigenBufferCount][stateCount];

        categoryRates = new double[categoryCount];

        partialsSize = patternCount * stateCount * categoryCount;

        tipStates = new int[compactBufferCount][];
        partials = new double[partialsBufferCount][];
        for (int i = tipCount; i < partialsBufferCount; i++) {
            partials[i] = new double[partialsSize];
        }

//        if (SCALING) {
//            scalingFactors = new double[2][nodeCount][];
//            for (int i = tipCount; i < nodeCount; i++) {
//                scalingFactors[0][i] = new double[patternCount];
//                scalingFactors[1][i] = new double[patternCount];
//            }
//            rootScalingFactors = new double[patternCount];
//
//            if (DYNAMIC_SCALING) {
//                currentScalingFactorsIndices = new int[nodeCount];
//                doRescaling = true;
//            }
//        }

        matrixSize = (stateCount + 1) * stateCount;

        matrices = new double[matrixBufferCount][categoryCount * matrixSize];
    }

    public void finalize() throws Throwable {
        super.finalize();
    }

    public void setPartials(final int bufferIndex, final double[] partials) {
        if (this.partials[bufferIndex] == null) {
            this.partials[bufferIndex] = new double[partialsSize];
        }
        for (int i = 0; i < categoryCount; i++) {
            System.arraycopy(partials, 0, this.partials[bufferIndex], 0, partialsSize);
        }
    }

    public void getPartials(final int bufferIndex, final double[] partials) {
        System.arraycopy(this.partials[bufferIndex], 0, partials, 0, partialsSize);
    }

    /**
     * Sets partials for a tip - these are numbered from 0 and remain
     * constant throughout the run.
     *
     * @param tipIndex the tip index
     * @param states   an array of patternCount state indices
     */
    public void setTipStates(int tipIndex, int[] states) {
        assert(this.tipStates[tipIndex] == null);
        tipStates[tipIndex] = new int[patternCount];
        int k = 0;
        for (int state : states) {
            this.tipStates[tipIndex][k] = (state < stateCount ? state : stateCount);
            k++;
        }

    }

    public void setEigenDecomposition(int eigenIndex, double[] eigenVectors, double[] inverseEigenValues, double[] eigenValues) {
        int l =0;
        for (int i = 0; i < stateCount; i++) {
            for (int j = 0; j < stateCount; j++) {
                for (int k = 0; k < stateCount; k++) {
                    cMatrices[eigenIndex][l] = eigenVectors[(i * stateCount) + k] * inverseEigenValues[(k * stateCount) + j];
                    l++;
                }
            }
        }
        System.arraycopy(eigenValues, 0, this.eigenValues[eigenIndex], 0, eigenValues.length);
    }

    public void setCategoryRates(double[] categoryRates) {
        System.arraycopy(categoryRates, 0, this.categoryRates, 0, this.categoryRates.length);
    }

    public void setTransitionMatrix(final int matrixIndex, final double[] inMatrix) {
        System.arraycopy(inMatrix, 0, this.matrices[matrixIndex], 0, this.matrixSize);
    }

    public void updateTransitionMatrices(final int eigenIndex,
                                         final int[] probabilityIndices,
                                         final int[] firstDerivativeIndices,
                                         final int[] secondDervativeIndices,
                                         final double[] edgeLengths,
                                         final int count) {
        for (int u = 0; u < count; u++) {
            int matrixIndex = probabilityIndices[u];

            if (DEBUG) System.err.println("Updating matrix for node " + matrixIndex);

            double[] tmp = new double[stateCount];

            int n = 0;
            for (int l = 0; l < categoryCount; l++) {
//	    if (DEBUG) System.err.println("1: Rate "+l+" = "+categoryRates[l]);
                for (int i = 0; i < stateCount; i++) {
                    tmp[i] =  Math.exp(eigenValues[eigenIndex][i] * edgeLengths[u] * categoryRates[l]);
                }
//            if (DEBUG) System.err.println(new dr.math.matrixAlgebra.Vector(tmp));
                //        if (DEBUG) System.exit(-1);

                int m = 0;
                for (int i = 0; i < stateCount; i++) {
                    for (int j = 0; j < stateCount; j++) {
                        double sum = 0.0;
                        for (int k = 0; k < stateCount; k++) {
                            sum += cMatrices[eigenIndex][m] * tmp[k];
                            m++;
                        }
                        //	    if (DEBUG) System.err.println("1: matrices[][]["+n+"] = "+sum);
                        if (sum > 0)
                            matrices[matrixIndex][n] = sum;
                        else
                            matrices[matrixIndex][n] = 0; // TODO Decision: set to -sum (as BEAST does)
                        n++;
                    }
                    matrices[matrixIndex][n] = 1.0;
                    n++;
                }

//            if (DEBUG) System.err.println(new dr.math.matrixAlgebra.Vector(matrices[currentMatricesIndices[nodeIndex]][nodeIndex]));
//            if (DEBUG) System.exit(0);
            }
        }
    }

    /**
     * Calculate or queue for calculation partials using a list of operations
     *
     * This function either calculates or queues for calculation a list partials. Implementations
     * supporting SYNCH may queue these calculations while other implementations perform these
     * operations immediately.  Implementations supporting GPU may perform all operations in the list
     * simultaneously.
     *
     * Operations list is a list of 6-tuple integer indices, with one 6-tuple per operation.
     * Format of 6-tuple operation: {destinationPartials,
     *                               destinationScalingFactors, (this index must be > tipCount)
     *                               child1Partials,
     *                               child1TransitionMatrix,
     *                               child2Partials,
     *                               child2TransitionMatrix}
     *
     * @param operations        List of 6-tuples specifying operations (input)
     * @param operationCount    Number of operations (input)
     * @param rescale           Specify whether (=1) or not (=0) to recalculate scaling factors
     *
     */
    public void updatePartials(final int[] operations, final int operationCount, final boolean rescale) {
//        if (SCALING) {
//            if (DYNAMIC_SCALING) {
//                if (!doRescaling) // Forces rescaling on first computation
//                    doRescaling = rescale;
////        		System.err.println("do dynamic = "+doRescaling);
////        		System.exit(-1);
//            }
//        }

        int x = 0;
        for (int op = 0; op < operationCount; op++) {
            int bufferIndex3 = operations[x];
            int bufferIndex1 = operations[x + 2];
            int matrixIndex1 = operations[x + 3];
            int bufferIndex2 = operations[x + 4];
            int matrixIndex2 = operations[x + 5];

            x += 6;

            if (compactBufferCount == 0) {
                updatePartialsPartials(bufferIndex1, matrixIndex1, bufferIndex2, matrixIndex2, bufferIndex3);
            } else {
                if (bufferIndex1 < tipCount && tipStates[bufferIndex1] != null) {
                    if (bufferIndex2 < tipCount && tipStates[bufferIndex2] != null) {
                        updateStatesStates(bufferIndex1, matrixIndex1, bufferIndex2, matrixIndex2, bufferIndex3);
                    } else {
                        updateStatesPartials(bufferIndex1, matrixIndex1, bufferIndex2, matrixIndex2, bufferIndex3);
                    }
                } else {
                    if (bufferIndex2 < tipCount && tipStates[bufferIndex2] != null) {
                        updateStatesPartials(bufferIndex2, matrixIndex2, bufferIndex1, matrixIndex1, bufferIndex3);
                    } else {
                        updatePartialsPartials(bufferIndex1, matrixIndex1, bufferIndex2, matrixIndex2, bufferIndex3);
                    }
                }
            }

//            if (SCALING) {
//                if (DYNAMIC_SCALING) {
//                    if (doRescaling) {
//                        currentScalingFactorsIndices[bufferIndex2] = 1 - currentScalingFactorsIndices[bufferIndex2];
//                        scalePartials(bufferIndex2,currentScalingFactorsIndices);
//                    }
//                } else {
//                    scalePartials(bufferIndex2,currentPartialsIndices);
//                }
//            }
        }
    }

    /**
     * Calculates partial likelihoods at a node when both children have states.
     */
    protected void updateStatesStates(int bufferIndex1, int matrixIndex1, int bufferIndex2, int matrixIndex2, int bufferIndex3)
    {
        double[] matrices1 = matrices[matrixIndex1];
        double[] matrices2 = matrices[matrixIndex2];

        int[] states1 = tipStates[bufferIndex1];
        int[] states2 = tipStates[bufferIndex2];

        double[] partials3 = partials[bufferIndex3];

        int v = 0;

        for (int l = 0; l < categoryCount; l++) {

            for (int k = 0; k < patternCount; k++) {

                int state1 = states1[k];
                int state2 = states2[k];

                int w = l * matrixSize;

                for (int i = 0; i < stateCount; i++) {

                    partials3[v] = matrices1[w + state1] * matrices2[w + state2];

                    v++;
                    w += (stateCount + 1);
                }

            }
        }
    }

    /**
     * Calculates partial likelihoods at a node when one child has states and one has partials.
     */
    protected void updateStatesPartials(int bufferIndex1, int matrixIndex1, int bufferIndex2, int matrixIndex2, int bufferIndex3)
    {
        double[] matrices1 = matrices[matrixIndex1];
        double[] matrices2 = matrices[matrixIndex2];

        int[] states1 = tipStates[bufferIndex1];
        double[] partials2 = partials[bufferIndex2];

        double[] partials3 = partials[bufferIndex3];

        double sum, tmp;

        int u = 0;
        int v = 0;

        for (int l = 0; l < categoryCount; l++) {

            for (int k = 0; k < patternCount; k++) {

                int state1 = states1[k];

                int w = l * matrixSize;

                for (int i = 0; i < stateCount; i++) {

                    tmp = matrices1[w + state1];

                    sum = 0.0;
                    for (int j = 0; j < stateCount; j++) {
                        sum += matrices2[w] * partials2[v + j];
                        w++;
                    }

                    // increment for the extra column at the end
                    w++;

                    partials3[u] = tmp * sum;
                    u++;
                }

                v += stateCount;
            }
        }
    }

    protected void updatePartialsPartials(int bufferIndex1, int matrixIndex1, int bufferIndex2, int matrixIndex2, int bufferIndex3)
    {
        double[] matrices1 = matrices[matrixIndex1];
        double[] matrices2 = matrices[matrixIndex2];

        double[] partials1 = partials[bufferIndex1];
        double[] partials2 = partials[bufferIndex2];

        double[] partials3 = partials[bufferIndex3];

        double sum1, sum2;

        int u = 0;
        int v = 0;

        for (int l = 0; l < categoryCount; l++) {

            for (int k = 0; k < patternCount; k++) {

                int w = l * matrixSize;

                for (int i = 0; i < stateCount; i++) {

                    sum1 = sum2 = 0.0;

                    for (int j = 0; j < stateCount; j++) {
                        sum1 += matrices1[w] * partials1[v + j];
                        sum2 += matrices2[w] * partials2[v + j];

                        w++;
                    }

                    // increment for the extra column at the end
                    w++;

                    partials3[u] = sum1 * sum2;
                    u++;
                }
                v += stateCount;

            }

            if (DEBUG) {
//    	    	System.err.println("1:PP node = "+nodeIndex3);
//    	    	for(int p=0; p<partials3.length; p++) {
//    	    		System.err.println("1:PP\t"+partials3[p]);
//    	    	}
//                System.err.println("node = "+nodeIndex3);
//                System.err.println(new dr.math.matrixAlgebra.Vector(partials3));
//                System.err.println(new dr.math.matrixAlgebra.Vector(scalingFactors[currentPartialsIndices[nodeIndex3]][nodeIndex3]));
                //System.exit(-1);
            }
        }
    }

    public void calculateRootLogLikelihoods(int[] bufferIndices, double[] weights, double[] stateFrequencies, int[] scalingFactorsIndices, int[] scalingFactorsCount, double[] outLogLikelihoods) {

        assert(bufferIndices.length == 0); // @todo implement integration across multiple subtrees

        double[] rootPartials = partials[bufferIndices[0]];

        double[] tmp = new double[patternCount * stateCount];

        int u = 0;
        int v = 0;
        for (int k = 0; k < patternCount; k++) {

            for (int i = 0; i < stateCount; i++) {

                tmp[u] = rootPartials[v] * weights[0];
                u++;
                v++;
            }
        }


        for (int l = 1; l < categoryCount; l++) {
            u = 0;

            for (int k = 0; k < patternCount; k++) {

                for (int i = 0; i < stateCount; i++) {

                    tmp[u] += rootPartials[v] * weights[l];
                    u++;
                    v++;
                }
            }
        }

//        if (SCALING) {
//            if (DYNAMIC_SCALING) {
//                if (doRescaling) {
//                    calculateRootScalingFactors(currentScalingFactorsIndices);
//                }
//            } else {
//                calculateRootScalingFactors(currentPartialsIndices);
//            }
//        }

        u = 0;
        for (int k = 0; k < patternCount; k++) {

            double sum = 0.0;
            for (int i = 0; i < stateCount; i++) {

                sum += stateFrequencies[i] * tmp[u];
                u++;
            }

            if (SCALING) {
//                    outLogLikelihoods[k] = Math.log(sum) + rootScalingFactors[k];
            } else {
                outLogLikelihoods[k] = Math.log(sum);
            }

            if (DEBUG) {
                System.err.println("log lik "+k+" = " + outLogLikelihoods[k]);
            }
        }
//        if (DEBUG) System.exit(-1);
    }

    public void calculateEdgeLogLikelihoods(int[] parentBufferIndices, int[] childBufferIndices, int[] probabilityIndices, int[] firstDerivativeIndices, int[] secondDerivativeIndices, double[] weights, double[] stateFrequencies, int[] scalingFactorsIndices, int[] scalingFactorsCount, double[] outLogLikelihoods, double[] outFirstDerivatives, double[] outSecondDerivatives) {
        throw new UnsupportedOperationException("calculateEdgeLogLikelihoods not implemented in GeneralBeagleImpl");
    }


//    private void calculateRootScalingFactors(int[] indices) {
//        for (int k = 0; k < patternCount; k++) {
//            double logScalingFactor = 0.0;
//            for (int i = tipCount; i < nodeCount; i++) {
//                logScalingFactor += scalingFactors[indices[i]][i][k];
//            }
//            if (DEBUG) System.err.println("1:SF "+logScalingFactor+" for "+ k);
//            rootScalingFactors[k] = logScalingFactor;
//        }
//
//    }

}