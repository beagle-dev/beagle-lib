package beagle;

import java.util.logging.Logger;

public class FourStateBeagleImpl extends DefaultBeagleImpl {

    public static final boolean DEBUG = false;

    public FourStateBeagleImpl(final int tipCount, final int partialsBufferCount, final int compactBufferCount, final int stateCount, final int patternCount, final int eigenBufferCount, final int matrixBufferCount) {
        super(tipCount, partialsBufferCount, compactBufferCount, stateCount, patternCount, eigenBufferCount, matrixBufferCount);
    }

//    /**
//     * Constructor
//     *
//     */
//    public FourStateBeagleImpl() {
//        super(4);
//        Logger.getLogger("beagle").info("Constructing double-precision 4-state Java BEAGLE implementation.");
//    }
//
//
//    /**
//     * Calculates partial likelihoods at a node when both children have states.
//     */
//    protected void updateStatesStates(int nodeIndex1, int nodeIndex2, int nodeIndex3)
//    {
//        double[] matrices1 = matrices[currentMatricesIndices[nodeIndex1]][nodeIndex1];
//        double[] matrices2 = matrices[currentMatricesIndices[nodeIndex2]][nodeIndex2];
//
//        int[] states1 = tipStates[nodeIndex1];
//        int[] states2 = tipStates[nodeIndex2];
//
//        double[] partials3 = partials[currentPartialsIndices[nodeIndex3]][nodeIndex3];
//
//        // copied from NucleotideLikelihoodCore
//        int v = 0;
//        for (int j = 0; j < categoryCount; j++) {
//
//            for (int k = 0; k < patternCount; k++) {
//
//                int state1 = states1[k];
//                int state2 = states2[k];
//
//                int w = j * 20;
//
//                partials3[v] = matrices1[w + state1] * matrices2[w + state2];
//                v++;	w += 5;
//                partials3[v] = matrices1[w + state1] * matrices2[w + state2];
//                v++;	w += 5;
//                partials3[v] = matrices1[w + state1] * matrices2[w + state2];
//                v++;	w += 5;
//                partials3[v] = matrices1[w + state1] * matrices2[w + state2];
//                v++;	w += 5;
//            }
//        }
//    }
//
//    /**
//     * Calculates partial likelihoods at a node when one child has states and one has partials.
//     * @param nodeIndex1
//     * @param nodeIndex2
//     * @param nodeIndex3
//     */
//    protected void updateStatesPartials(int nodeIndex1, int nodeIndex2, int nodeIndex3)
//    {
//        double[] matrices1 = matrices[currentMatricesIndices[nodeIndex1]][nodeIndex1];
//        double[] matrices2 = matrices[currentMatricesIndices[nodeIndex2]][nodeIndex2];
//
//        int[] states1 = tipStates[nodeIndex1];
//        double[] partials2 = partials[currentPartialsIndices[nodeIndex2]][nodeIndex2];
//
//        double[] partials3 = partials[currentPartialsIndices[nodeIndex3]][nodeIndex3];
//
//        // copied from NucleotideLikelihoodCore
//        int u = 0;
//        int v = 0;
//
//        for (int l = 0; l < categoryCount; l++) {
//            for (int k = 0; k < patternCount; k++) {
//
//                int state1 = states1[k];
//
//                int w = l * 20;
//
//                partials3[u] = matrices1[w + state1];
//
//                double sum = matrices2[w] * partials2[v]; w++;
//                sum +=	matrices2[w] * partials2[v + 1]; w++;
//                sum +=	matrices2[w] * partials2[v + 2]; w++;
//                sum +=	matrices2[w] * partials2[v + 3]; w++;
//                w++; // increment for the extra column at the end
//                partials3[u] *= sum;	u++;
//
//                partials3[u] = matrices1[w + state1];
//
//                sum = matrices2[w] * partials2[v]; w++;
//                sum +=	matrices2[w] * partials2[v + 1]; w++;
//                sum +=	matrices2[w] * partials2[v + 2]; w++;
//                sum +=	matrices2[w] * partials2[v + 3]; w++;
//                w++; // increment for the extra column at the end
//                partials3[u] *= sum;	u++;
//
//                partials3[u] = matrices1[w + state1];
//
//                sum = matrices2[w] * partials2[v]; w++;
//                sum +=	matrices2[w] * partials2[v + 1]; w++;
//                sum +=	matrices2[w] * partials2[v + 2]; w++;
//                sum +=	matrices2[w] * partials2[v + 3]; w++;
//                w++; // increment for the extra column at the end
//                partials3[u] *= sum;	u++;
//
//                partials3[u] = matrices1[w + state1];
//
//                sum = matrices2[w] * partials2[v]; w++;
//                sum +=	matrices2[w] * partials2[v + 1]; w++;
//                sum +=	matrices2[w] * partials2[v + 2]; w++;
//                sum +=	matrices2[w] * partials2[v + 3];
//                partials3[u] *= sum;	u++;
//
//                v += 4;
//
//            }
//        }
//    }
//
//    protected void updatePartialsPartials(int nodeIndex1, int nodeIndex2, int nodeIndex3)
//    {
//        double[] matrices1 = matrices[currentMatricesIndices[nodeIndex1]][nodeIndex1];
//        double[] matrices2 = matrices[currentMatricesIndices[nodeIndex2]][nodeIndex2];
//
//        double[] partials1 = partials[currentPartialsIndices[nodeIndex1]][nodeIndex1];
//        double[] partials2 = partials[currentPartialsIndices[nodeIndex2]][nodeIndex2];
//
//        double[] partials3 = partials[currentPartialsIndices[nodeIndex3]][nodeIndex3];
//
//        // copied from NucleotideLikelihoodCore
//
//        double sum1, sum2;
//
//        int u = 0;
//        int v = 0;
//
//        for (int l = 0; l < categoryCount; l++) {
//            for (int k = 0; k < patternCount; k++) {
//
//                int w = l * 20;
//
//                sum1 = matrices1[w] * partials1[v];
//                sum2 = matrices2[w] * partials2[v]; w++;
//                sum1 += matrices1[w] * partials1[v + 1];
//                sum2 += matrices2[w] * partials2[v + 1]; w++;
//                sum1 += matrices1[w] * partials1[v + 2];
//                sum2 += matrices2[w] * partials2[v + 2]; w++;
//                sum1 += matrices1[w] * partials1[v + 3];
//                sum2 += matrices2[w] * partials2[v + 3]; w++;
//                w++; // increment for the extra column at the end
//                partials3[u] = sum1 * sum2; u++;
//
//                sum1 = matrices1[w] * partials1[v];
//                sum2 = matrices2[w] * partials2[v]; w++;
//                sum1 += matrices1[w] * partials1[v + 1];
//                sum2 += matrices2[w] * partials2[v + 1]; w++;
//                sum1 += matrices1[w] * partials1[v + 2];
//                sum2 += matrices2[w] * partials2[v + 2]; w++;
//                sum1 += matrices1[w] * partials1[v + 3];
//                sum2 += matrices2[w] * partials2[v + 3]; w++;
//                w++; // increment for the extra column at the end
//                partials3[u] = sum1 * sum2; u++;
//
//                sum1 = matrices1[w] * partials1[v];
//                sum2 = matrices2[w] * partials2[v]; w++;
//                sum1 += matrices1[w] * partials1[v + 1];
//                sum2 += matrices2[w] * partials2[v + 1]; w++;
//                sum1 += matrices1[w] * partials1[v + 2];
//                sum2 += matrices2[w] * partials2[v + 2]; w++;
//                sum1 += matrices1[w] * partials1[v + 3];
//                sum2 += matrices2[w] * partials2[v + 3]; w++;
//                w++; // increment for the extra column at the end
//                partials3[u] = sum1 * sum2; u++;
//
//                sum1 = matrices1[w] * partials1[v];
//                sum2 = matrices2[w] * partials2[v]; w++;
//                sum1 += matrices1[w] * partials1[v + 1];
//                sum2 += matrices2[w] * partials2[v + 1]; w++;
//                sum1 += matrices1[w] * partials1[v + 2];
//                sum2 += matrices2[w] * partials2[v + 2]; w++;
//                sum1 += matrices1[w] * partials1[v + 3];
//                sum2 += matrices2[w] * partials2[v + 3];
//                partials3[u] = sum1 * sum2; u++;
//
//                v += 4;
//            }
//        }
//    }


}