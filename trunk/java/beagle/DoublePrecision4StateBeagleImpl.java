package beagle;

import java.util.logging.Logger;

public class DoublePrecision4StateBeagleImpl implements Beagle {

    public static final boolean DEBUG = false;

    protected final int stateCount;
    protected int nodeCount;
    protected int tipCount;
    protected int patternCount;
    protected int partialsSize;
    protected int categoryCount;
    protected int matrixSize;
    protected int matrixCount;

    protected double[][] cMatrices;
    protected double[][] storedCMatrices;
    protected double[][] eigenValues;
    protected double[][] storedEigenValues;

    protected double[] frequencies;
    protected double[] storedFrequencies;
    protected double[] categoryProportions;
    protected double[] storedCategoryProportions;
    protected double[] categoryRates;
    protected double[] storedCategoryRates;

    protected double[][][] partials;

    protected int[][] states;

    protected double[][][] matrices;

    protected int[] currentMatricesIndices;
    protected int[] storedMatricesIndices;
    protected int[] currentPartialsIndices;
    protected int[] storedPartialsIndices;

    protected boolean useScaling = false;

    protected double[][][] scalingFactors;


    /**
     * Constructor
     *
     */
    public DoublePrecision4StateBeagleImpl() {
        this.stateCount = 4;
        Logger.getLogger("beagle").info("Constructing double-precision 4-state Java BEAGLE implementation.");
    }

    public boolean canHandleTipPartials() {
        return true;
    }

    public boolean canHandleTipStates() {
        return true;
    }

    public boolean canHandleDynamicRescaling() {
        return true;
    }


    public void initialize(
            int nodeCount,
            int tipCount,
            int patternCount,
            int categoryCount,
            int matrixCount) {

        this.nodeCount = nodeCount;
        if (nodeCount < 3) {
            throw new IllegalArgumentException("Node count must be at least 3");
        }

        this.tipCount = tipCount;
        if (tipCount < 2 || tipCount >= nodeCount) {
            throw new IllegalArgumentException("Tip count must be at least 2 and less than node count");
        }

        this.patternCount = patternCount;
        if (patternCount < 1) {
            throw new IllegalArgumentException("Pattern count must be at least 1");
        }

        this.categoryCount = categoryCount;
        if (categoryCount < 1) {
            throw new IllegalArgumentException("Category count must be at least 1");
        }

        this.matrixCount = matrixCount;
        if (matrixCount != 1 && matrixCount != categoryCount) {
            throw new IllegalArgumentException("Matrix count must be 1 or equal to category count");
        }

        cMatrices = new double[matrixCount][stateCount * stateCount * stateCount];
        storedCMatrices = new double[matrixCount][stateCount * stateCount * stateCount];

        eigenValues = new double[matrixCount][stateCount];
        storedEigenValues = new double[matrixCount][stateCount];

        frequencies = new double[stateCount];
        storedFrequencies = new double[stateCount];

        categoryRates = new double[categoryCount];
        storedCategoryRates = new double[categoryCount];

        categoryProportions = new double[categoryCount];
        storedCategoryProportions = new double[categoryCount];

        partialsSize = patternCount * stateCount * categoryCount;

        partials = new double[2][nodeCount][partialsSize];

        scalingFactors = new double[2][nodeCount][patternCount];

        states = new int[nodeCount][patternCount * categoryCount];

        matrixSize = stateCount * stateCount;
//        matrixSize = (stateCount + 1) * stateCount;

        matrices = new double[2][nodeCount][categoryCount * matrixSize];

        currentMatricesIndices = new int[nodeCount];
        storedMatricesIndices = new int[nodeCount];

        currentPartialsIndices = new int[nodeCount];
        storedPartialsIndices = new int[nodeCount];
    }

    /**
     * cleans up and deallocates arrays.
     */
    public void finalize() throws Throwable  {
        super.finalize();

        nodeCount = 0;
        patternCount = 0;
        matrixCount = 0;

        partials = null;
        currentPartialsIndices = null;
        storedPartialsIndices = null;
        states = null;
        matrices = null;
        currentMatricesIndices = null;
        storedMatricesIndices = null;
    }


    /**
     * Sets partials for a tip
     */
    public void setTipPartials(int tipIndex, double[] partials) {

        int k = 0;
        for (int i = 0; i < categoryCount; i++) {
            System.arraycopy(partials, 0, this.partials[0][tipIndex], k, partials.length);
            k += partials.length;
        }
    }

    /**
     * Sets partials for a tip - these are numbered from 0 and remain
     * constant throughout the run.
     *
     * @param tipIndex the tip index
     * @param states   an array of patternCount state indices
     */
    public void setTipStates(int tipIndex, int[] states) {
        int k = 0;
        for (int i = 0; i < categoryCount; i++) {
            for (int j = 0; j < states.length; j++) {
                this.states[tipIndex][k] = (states[j] < stateCount ? states[j] : stateCount);
                k++;
            }
        }
    }

    public void setStateFrequencies(double[] stateFrequencies) {
        System.arraycopy(stateFrequencies, 0, frequencies, 0, frequencies.length);
    }

    public void setEigenDecomposition(int matrixIndex, double[][] eigenVectors, double[][] inverseEigenValues, double[] eigenValues) {
        int l =0;
        for (int i = 0; i < stateCount; i++) {
            for (int j = 0; j < stateCount; j++) {
                for (int k = 0; k < stateCount; k++) {
                    cMatrices[matrixIndex][l] = eigenVectors[i][k] * inverseEigenValues[k][j];
                    l++;
                }
            }
        }
        System.arraycopy(eigenValues, 0, this.eigenValues[matrixIndex], 0, eigenValues.length);
    }

    public void setCategoryRates(double[] categoryRates) {
        System.arraycopy(categoryRates, 0, this.categoryRates, 0, this.categoryRates.length);
    }

    public void setCategoryProportions(double[] categoryProportions) {
        System.arraycopy(categoryProportions, 0, this.categoryProportions, 0, this.categoryProportions.length);
    }

    public void calculateProbabilityTransitionMatrices(int[] nodeIndices, double[] branchLengths, int count) {
        for (int u = 0; u < count; u++) {
            int nodeIndex = nodeIndices[u];

//            if (DEBUG) System.err.println("Updating matrix for node " + nodeIndex);

            currentMatricesIndices[nodeIndex] = 1 - currentMatricesIndices[nodeIndex];

//            if (DEBUG && nodeIndex == 0) {
//                System.err.println(matrices[currentMatricesIndices[0]][0][0]);
//                System.err.println(matrices[currentMatricesIndices[0]][0][184]);
//            }

            double[] tmp = new double[stateCount];

            int n = 0;
            int matrixIndex = 0;
            for (int l = 0; l < categoryCount; l++) {
//	    if (DEBUG) System.err.println("1: Rate "+l+" = "+categoryRates[l]);
                for (int i = 0; i < stateCount; i++) {
                    tmp[i] =  Math.exp(eigenValues[matrixIndex][i] * branchLengths[u] * categoryRates[l]);
                }
//            if (DEBUG) System.err.println(new dr.math.matrixAlgebra.Vector(tmp));
                //        if (DEBUG) System.exit(-1);

                int m = 0;
                for (int i = 0; i < stateCount; i++) {
                    for (int j = 0; j < stateCount; j++) {
                        double sum = 0.0;
                        for (int k = 0; k < stateCount; k++) {
                            sum += cMatrices[matrixIndex][m] * tmp[k];
                            m++;
                        }
                        //	    if (DEBUG) System.err.println("1: matrices[][]["+n+"] = "+sum);
                        matrices[currentMatricesIndices[nodeIndex]][nodeIndex][n] = sum;
                        n++;
                    }
//                    matrices[currentMatricesIndices[nodeIndex]][nodeIndex][n] = 1.0;
//                    n++;
                }

                if (matrixCount > 1) {
                    matrixIndex ++;
                }
//            if (DEBUG) System.err.println(new dr.math.matrixAlgebra.Vector(matrices[currentMatricesIndices[nodeIndex]][nodeIndex]));
//            if (DEBUG) System.exit(0);
            }
        }
    }


    public void calculatePartials(int[] operations, int[] dependencies, int operationCount) {

        int x = 0;
        for (int op = 0; op < operationCount; op++) {
            int nodeIndex1 = operations[x];
            x++;
            int nodeIndex2 = operations[x];
            x++;
            int nodeIndex3 = operations[x];
            x++;

            currentPartialsIndices[nodeIndex3] = 1 - currentPartialsIndices[nodeIndex3];

            if (nodeIndex1 < tipCount) {
                if (nodeIndex2 < tipCount) {
                    updateStatesStates(nodeIndex1, nodeIndex2, nodeIndex3);
                } else {
                    updateStatesPartials(nodeIndex1, nodeIndex2, nodeIndex3);
                }
            } else {
                if (nodeIndex2 < tipCount) {
                    updateStatesPartials(nodeIndex2, nodeIndex1, nodeIndex3);
                } else {
                    updatePartialsPartials(nodeIndex1, nodeIndex2, nodeIndex3);
                }
            }

            if (useScaling) {
                scalePartials(nodeIndex3);
            }
        }
    }

    /**
     * Calculates partial likelihoods at a node when both children have states.
     */
    private void updateStatesStates(int nodeIndex1, int nodeIndex2, int nodeIndex3)
    {
        double[] matrices1 = matrices[currentMatricesIndices[nodeIndex1]][nodeIndex1];
        double[] matrices2 = matrices[currentMatricesIndices[nodeIndex2]][nodeIndex2];

        int[] states1 = states[nodeIndex1];
        int[] states2 = states[nodeIndex2];

        double[] partials3 = partials[currentPartialsIndices[nodeIndex3]][nodeIndex3];

        // copied from NucleotideLikelihoodCore
        int u = 0;
        int v = 0;
        for (int j = 0; j < categoryCount; j++) {

            for (int k = 0; k < patternCount; k++) {

                int w = u;

                int state1 = states1[k];
                int state2 = states2[k];

                if (state1 < 4 && state2 < 4) {

                    partials3[v] = matrices1[w + state1] * matrices2[w + state2];
                    v++;	w += 4;
                    partials3[v] = matrices1[w + state1] * matrices2[w + state2];
                    v++;	w += 4;
                    partials3[v] = matrices1[w + state1] * matrices2[w + state2];
                    v++;	w += 4;
                    partials3[v] = matrices1[w + state1] * matrices2[w + state2];
                    v++;	w += 4;

                } else if (state1 < 4) {
                    // child 2 has a gap or unknown state so don't use it

                    partials3[v] = matrices1[w + state1];
                    v++;	w += 4;
                    partials3[v] = matrices1[w + state1];
                    v++;	w += 4;
                    partials3[v] = matrices1[w + state1];
                    v++;	w += 4;
                    partials3[v] = matrices1[w + state1];
                    v++;	w += 4;

                } else if (state2 < 4) {
                    // child 2 has a gap or unknown state so don't use it
                    partials3[v] = matrices2[w + state2];
                    v++;	w += 4;
                    partials3[v] = matrices2[w + state2];
                    v++;	w += 4;
                    partials3[v] = matrices2[w + state2];
                    v++;	w += 4;
                    partials3[v] = matrices2[w + state2];
                    v++;	w += 4;

                } else {
                    // both children have a gap or unknown state so set partials to 1
                    partials3[v] = 1.0;
                    v++;
                    partials3[v] = 1.0;
                    v++;
                    partials3[v] = 1.0;
                    v++;
                    partials3[v] = 1.0;
                    v++;
                }
            }

            u += 16;
        }
    }

    /**
     * Calculates partial likelihoods at a node when one child has states and one has partials.
     * @param nodeIndex1
     * @param nodeIndex2
     * @param nodeIndex3
     */
    private void updateStatesPartials(int nodeIndex1, int nodeIndex2, int nodeIndex3)
    {
        double[] matrices1 = matrices[currentMatricesIndices[nodeIndex1]][nodeIndex1];
        double[] matrices2 = matrices[currentMatricesIndices[nodeIndex2]][nodeIndex2];

        int[] states1 = states[nodeIndex1];
        double[] partials2 = partials[currentPartialsIndices[nodeIndex2]][nodeIndex2];

        double[] partials3 = partials[currentPartialsIndices[nodeIndex3]][nodeIndex3];

        // copied from NucleotideLikelihoodCore
        int u = 0;
        int v = 0;
        int w = 0;

        for (int l = 0; l < categoryCount; l++) {
            for (int k = 0; k < patternCount; k++) {

                int state1 = states1[k];

                if (state1 < 4) {

                    double sum;

                    sum =	matrices2[w] * partials2[v];
                    sum +=	matrices2[w + 1] * partials2[v + 1];
                    sum +=	matrices2[w + 2] * partials2[v + 2];
                    sum +=	matrices2[w + 3] * partials2[v + 3];
                    partials3[u] = matrices1[w + state1] * sum;	u++;

                    sum =	matrices2[w + 4] * partials2[v];
                    sum +=	matrices2[w + 5] * partials2[v + 1];
                    sum +=	matrices2[w + 6] * partials2[v + 2];
                    sum +=	matrices2[w + 7] * partials2[v + 3];
                    partials3[u] = matrices1[w + 4 + state1] * sum;	u++;

                    sum =	matrices2[w + 8] * partials2[v];
                    sum +=	matrices2[w + 9] * partials2[v + 1];
                    sum +=	matrices2[w + 10] * partials2[v + 2];
                    sum +=	matrices2[w + 11] * partials2[v + 3];
                    partials3[u] = matrices1[w + 8 + state1] * sum;	u++;

                    sum =	matrices2[w + 12] * partials2[v];
                    sum +=	matrices2[w + 13] * partials2[v + 1];
                    sum +=	matrices2[w + 14] * partials2[v + 2];
                    sum +=	matrices2[w + 15] * partials2[v + 3];
                    partials3[u] = matrices1[w + 12 + state1] * sum;	u++;

                    v += 4;

                } else {
                    // Child 1 has a gap or unknown state so don't use it

                    double sum;

                    sum =	matrices2[w] * partials2[v];
                    sum +=	matrices2[w + 1] * partials2[v + 1];
                    sum +=	matrices2[w + 2] * partials2[v + 2];
                    sum +=	matrices2[w + 3] * partials2[v + 3];
                    partials3[u] = sum;	u++;

                    sum =	matrices2[w + 4] * partials2[v];
                    sum +=	matrices2[w + 5] * partials2[v + 1];
                    sum +=	matrices2[w + 6] * partials2[v + 2];
                    sum +=	matrices2[w + 7] * partials2[v + 3];
                    partials3[u] = sum;	u++;

                    sum =	matrices2[w + 8] * partials2[v];
                    sum +=	matrices2[w + 9] * partials2[v + 1];
                    sum +=	matrices2[w + 10] * partials2[v + 2];
                    sum +=	matrices2[w + 11] * partials2[v + 3];
                    partials3[u] = sum;	u++;

                    sum =	matrices2[w + 12] * partials2[v];
                    sum +=	matrices2[w + 13] * partials2[v + 1];
                    sum +=	matrices2[w + 14] * partials2[v + 2];
                    sum +=	matrices2[w + 15] * partials2[v + 3];
                    partials3[u] = sum;	u++;

                    v += 4;
                }
            }

            w += 16;
        }
    }

    private void updatePartialsPartials(int nodeIndex1, int nodeIndex2, int nodeIndex3)
    {
        double[] matrices1 = matrices[currentMatricesIndices[nodeIndex1]][nodeIndex1];
        double[] matrices2 = matrices[currentMatricesIndices[nodeIndex2]][nodeIndex2];

        double[] partials1 = partials[currentPartialsIndices[nodeIndex1]][nodeIndex1];
        double[] partials2 = partials[currentPartialsIndices[nodeIndex2]][nodeIndex2];

        double[] partials3 = partials[currentPartialsIndices[nodeIndex3]][nodeIndex3];

        // copied from NucleotideLikelihoodCore

        double sum1, sum2;

        int u = 0;
        int v = 0;
        int w = 0;

        for (int l = 0; l < categoryCount; l++) {
            for (int k = 0; k < patternCount; k++) {

                sum1 = matrices1[w] * partials1[v];
                sum2 = matrices2[w] * partials2[v];
                sum1 += matrices1[w + 1] * partials1[v + 1];
                sum2 += matrices2[w + 1] * partials2[v + 1];
                sum1 += matrices1[w + 2] * partials1[v + 2];
                sum2 += matrices2[w + 2] * partials2[v + 2];
                sum1 += matrices1[w + 3] * partials1[v + 3];
                sum2 += matrices2[w + 3] * partials2[v + 3];
                partials3[u] = sum1 * sum2; u++;

                sum1 = matrices1[w + 4] * partials1[v];
                sum2 = matrices2[w + 4] * partials2[v];
                sum1 += matrices1[w + 5] * partials1[v + 1];
                sum2 += matrices2[w + 5] * partials2[v + 1];
                sum1 += matrices1[w + 6] * partials1[v + 2];
                sum2 += matrices2[w + 6] * partials2[v + 2];
                sum1 += matrices1[w + 7] * partials1[v + 3];
                sum2 += matrices2[w + 7] * partials2[v + 3];
                partials3[u] = sum1 * sum2; u++;

                sum1 = matrices1[w + 8] * partials1[v];
                sum2 = matrices2[w + 8] * partials2[v];
                sum1 += matrices1[w + 9] * partials1[v + 1];
                sum2 += matrices2[w + 9] * partials2[v + 1];
                sum1 += matrices1[w + 10] * partials1[v + 2];
                sum2 += matrices2[w + 10] * partials2[v + 2];
                sum1 += matrices1[w + 11] * partials1[v + 3];
                sum2 += matrices2[w + 11] * partials2[v + 3];
                partials3[u] = sum1 * sum2; u++;

                sum1 = matrices1[w + 12] * partials1[v];
                sum2 = matrices2[w + 12] * partials2[v];
                sum1 += matrices1[w + 13] * partials1[v + 1];
                sum2 += matrices2[w + 13] * partials2[v + 1];
                sum1 += matrices1[w + 14] * partials1[v + 2];
                sum2 += matrices2[w + 14] * partials2[v + 2];
                sum1 += matrices1[w + 15] * partials1[v + 3];
                sum2 += matrices2[w + 15] * partials2[v + 3];
                partials3[u] = sum1 * sum2; u++;

                v += 4;
            }

            w += 16;
        }
    }


    /**
     * Scale the partials at a given node. This uses a scaling suggested by Ziheng Yang in
     * Yang (2000) J. Mol. Evol. 51: 423-432
     * <p/>
     * This function looks over the partial likelihoods for each state at each pattern
     * and finds the largest. If this is less than the scalingThreshold (currently set
     * to 1E-40) then it rescales the partials for that pattern by dividing by this number
     * (i.e., normalizing to between 0, 1). It then stores the log of this scaling.
     * This is called for every internal node after the partials are calculated so provides
     * most of the performance hit. Ziheng suggests only doing this on a proportion of nodes
     * but this sounded like a headache to organize (and he doesn't use the threshold idea
     * which improves the performance quite a bit).
     *
     * @param nodeIndex
     */
    protected void scalePartials(int nodeIndex) {
        int u = 0;

        for (int i = 0; i < patternCount; i++) {

            double scaleFactor = 0.0;
            int v = u;
            for (int k = 0; k < categoryCount; k++) {
                for (int j = 0; j < stateCount; j++) {
                    if (partials[currentPartialsIndices[nodeIndex]][nodeIndex][v] > scaleFactor) {
                        scaleFactor = partials[currentPartialsIndices[nodeIndex]][nodeIndex][v];
                    }
                    v++;
                }
                v += (patternCount - 1) * stateCount;
            }

            if (scaleFactor < 1E+40) {

                v = u;
                for (int k = 0; k < categoryCount; k++) {
                    for (int j = 0; j < stateCount; j++) {
                        partials[currentPartialsIndices[nodeIndex]][nodeIndex][v] /= scaleFactor;
                        v++;
                    }
                    v += (patternCount - 1) * stateCount;
                }
                scalingFactors[currentPartialsIndices[nodeIndex]][nodeIndex][i] = Math.log(scaleFactor);

            } else {
                scalingFactors[currentPartialsIndices[nodeIndex]][nodeIndex][i] = 0.0;
            }
            u += stateCount;


        }
    }


    /**
     * This function returns the scaling factor for that pattern by summing over
     * the log scalings used at each node. If scaling is off then this just returns
     * a 0.
     *
     * @return the log scaling factor
     */
    public double getLogScalingFactor(int pattern) {
        double logScalingFactor = 0.0;
        if (useScaling) {
            for (int i = 0; i < nodeCount; i++) {
                logScalingFactor += scalingFactors[currentPartialsIndices[i]][i][pattern];
                if (DEBUG && pattern == 1) System.err.println("Adding "+scalingFactors[currentPartialsIndices[i]][i][pattern]);
            }
        }

        if (DEBUG) System.err.println("1:SF "+logScalingFactor+" for "+pattern);
        return logScalingFactor;
    }


    /**
     * Calculates pattern log likelihoods at a node.
     *
     * @param rootNodeIndex the index of the root node
     * @param outLogLikelihoods an array into which the log likelihoods will go
     */
    public void calculateLogLikelihoods(int rootNodeIndex, double[] outLogLikelihoods) {

        // @todo I have a feeling this could be done in a single set of nested loops.

        double[] rootPartials = partials[currentPartialsIndices[rootNodeIndex]][rootNodeIndex];

        double[] tmp = new double[patternCount * stateCount];

        int u = 0;
        int v = 0;
        for (int k = 0; k < patternCount; k++) {

            for (int i = 0; i < stateCount; i++) {

                tmp[u] = rootPartials[v] * categoryProportions[0];
                u++;
                v++;
            }
        }


        for (int l = 1; l < categoryCount; l++) {
            u = 0;

            for (int k = 0; k < patternCount; k++) {

                for (int i = 0; i < stateCount; i++) {

                    tmp[u] += rootPartials[v] * categoryProportions[l];
                    u++;
                    v++;
                }
            }
        }

        u = 0;
        for (int k = 0; k < patternCount; k++) {

            double sum = 0.0;
            for (int i = 0; i < stateCount; i++) {

                sum += frequencies[i] * tmp[u];
                u++;
            }
            outLogLikelihoods[k] = Math.log(sum) + getLogScalingFactor(k);
            if (DEBUG) {
                System.err.println("log lik "+k+" = "+outLogLikelihoods[k]);
            }
        }
        if (DEBUG) System.exit(-1);
    }

    /**
     * Store current state
     */
    public void storeState() {

        for (int i = 0; i < matrixCount; i++) {
            System.arraycopy(cMatrices[i], 0, storedCMatrices[i], 0, cMatrices[i].length);
            System.arraycopy(eigenValues[i], 0, storedEigenValues[i], 0, eigenValues[i].length);
        }

        System.arraycopy(frequencies, 0, storedFrequencies, 0, frequencies.length);
        System.arraycopy(categoryRates, 0, storedCategoryRates, 0, categoryRates.length);
        System.arraycopy(categoryProportions, 0, storedCategoryProportions, 0, categoryProportions.length);

        System.arraycopy(currentMatricesIndices, 0, storedMatricesIndices, 0, nodeCount);
        System.arraycopy(currentPartialsIndices, 0, storedPartialsIndices, 0, nodeCount);
    }

    /**
     * Restore the stored state
     */
    public void restoreState() {
        // Rather than copying the stored stuff back, just swap the pointers...
        double[][] tmp = cMatrices;
        cMatrices = storedCMatrices;
        storedCMatrices = tmp;

        tmp = eigenValues;
        eigenValues = storedEigenValues;
        storedEigenValues = tmp;

        double[] tmp1 = frequencies;
        frequencies = storedFrequencies;
        storedFrequencies = tmp1;

        tmp1 = categoryRates;
        categoryRates = storedCategoryRates;
        storedCategoryRates = tmp1;

        tmp1 = categoryProportions;
        categoryProportions = storedCategoryProportions;
        storedCategoryProportions = tmp1;

        int[] tmp3 = currentMatricesIndices;
        currentMatricesIndices = storedMatricesIndices;
        storedMatricesIndices = tmp3;

        int[] tmp4 = currentPartialsIndices;
        currentPartialsIndices = storedPartialsIndices;
        storedPartialsIndices = tmp4;
    }
}