package beagle;

import java.util.logging.Logger;

public class FourStateBeagleImpl extends GeneralBeagleImpl {

    public static final boolean DEBUG = false;

    public FourStateBeagleImpl(final int tipCount, final int partialsBufferCount, final int compactBufferCount, final int patternCount, final int eigenBufferCount, final int matrixBufferCount, final int categoryCount) {
        super(tipCount, partialsBufferCount, compactBufferCount, 4, patternCount, eigenBufferCount, matrixBufferCount, categoryCount);
        Logger.getLogger("beagle").info("Constructing double-precision 4-state Java BEAGLE implementation.");
    }

    protected void updateStatesStates(int bufferIndex1, int matrixIndex1, int bufferIndex2, int matrixIndex2, int bufferIndex3)
    {
        double[] matrices1 = matrices[matrixIndex1];
        double[] matrices2 = matrices[matrixIndex2];

        int[] states1 = tipStates[bufferIndex1];
        int[] states2 = tipStates[bufferIndex2];

        double[] partials3 = partials[bufferIndex3];

        // copied from NucleotideLikelihoodCore
        int v = 0;
        for (int j = 0; j < categoryCount; j++) {

            for (int k = 0; k < patternCount; k++) {

                int state1 = states1[k];
                int state2 = states2[k];

                int w = j * 20;

                partials3[v] = matrices1[w + state1] * matrices2[w + state2];
                v++;	w += 5;
                partials3[v] = matrices1[w + state1] * matrices2[w + state2];
                v++;	w += 5;
                partials3[v] = matrices1[w + state1] * matrices2[w + state2];
                v++;	w += 5;
                partials3[v] = matrices1[w + state1] * matrices2[w + state2];
                v++;	w += 5;
            }
        }
    }

    protected void updateStatesPartials(int bufferIndex1, int matrixIndex1, int bufferIndex2, int matrixIndex2, int bufferIndex3)
    {
        double[] matrices1 = matrices[matrixIndex1];
        double[] matrices2 = matrices[matrixIndex2];

        int[] states1 = tipStates[bufferIndex1];
        double[] partials2 = partials[bufferIndex2];

        double[] partials3 = partials[bufferIndex3];

        // copied from NucleotideLikelihoodCore
        int u = 0;
        int v = 0;

        for (int l = 0; l < categoryCount; l++) {
            for (int k = 0; k < patternCount; k++) {

                int state1 = states1[k];

                int w = l * 20;

                partials3[u] = matrices1[w + state1];

                double sum = matrices2[w] * partials2[v]; w++;
                sum +=	matrices2[w] * partials2[v + 1]; w++;
                sum +=	matrices2[w] * partials2[v + 2]; w++;
                sum +=	matrices2[w] * partials2[v + 3]; w++;
                w++; // increment for the extra column at the end
                partials3[u] *= sum;	u++;

                partials3[u] = matrices1[w + state1];

                sum = matrices2[w] * partials2[v]; w++;
                sum +=	matrices2[w] * partials2[v + 1]; w++;
                sum +=	matrices2[w] * partials2[v + 2]; w++;
                sum +=	matrices2[w] * partials2[v + 3]; w++;
                w++; // increment for the extra column at the end
                partials3[u] *= sum;	u++;

                partials3[u] = matrices1[w + state1];

                sum = matrices2[w] * partials2[v]; w++;
                sum +=	matrices2[w] * partials2[v + 1]; w++;
                sum +=	matrices2[w] * partials2[v + 2]; w++;
                sum +=	matrices2[w] * partials2[v + 3]; w++;
                w++; // increment for the extra column at the end
                partials3[u] *= sum;	u++;

                partials3[u] = matrices1[w + state1];

                sum = matrices2[w] * partials2[v]; w++;
                sum +=	matrices2[w] * partials2[v + 1]; w++;
                sum +=	matrices2[w] * partials2[v + 2]; w++;
                sum +=	matrices2[w] * partials2[v + 3];
                partials3[u] *= sum;	u++;

                v += 4;

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

        // copied from NucleotideLikelihoodCore

        double sum1, sum2;

        int u = 0;
        int v = 0;

        for (int l = 0; l < categoryCount; l++) {
            for (int k = 0; k < patternCount; k++) {

                int w = l * 20;

                sum1 = matrices1[w] * partials1[v];
                sum2 = matrices2[w] * partials2[v]; w++;
                sum1 += matrices1[w] * partials1[v + 1];
                sum2 += matrices2[w] * partials2[v + 1]; w++;
                sum1 += matrices1[w] * partials1[v + 2];
                sum2 += matrices2[w] * partials2[v + 2]; w++;
                sum1 += matrices1[w] * partials1[v + 3];
                sum2 += matrices2[w] * partials2[v + 3]; w++;
                w++; // increment for the extra column at the end
                partials3[u] = sum1 * sum2; u++;

                sum1 = matrices1[w] * partials1[v];
                sum2 = matrices2[w] * partials2[v]; w++;
                sum1 += matrices1[w] * partials1[v + 1];
                sum2 += matrices2[w] * partials2[v + 1]; w++;
                sum1 += matrices1[w] * partials1[v + 2];
                sum2 += matrices2[w] * partials2[v + 2]; w++;
                sum1 += matrices1[w] * partials1[v + 3];
                sum2 += matrices2[w] * partials2[v + 3]; w++;
                w++; // increment for the extra column at the end
                partials3[u] = sum1 * sum2; u++;

                sum1 = matrices1[w] * partials1[v];
                sum2 = matrices2[w] * partials2[v]; w++;
                sum1 += matrices1[w] * partials1[v + 1];
                sum2 += matrices2[w] * partials2[v + 1]; w++;
                sum1 += matrices1[w] * partials1[v + 2];
                sum2 += matrices2[w] * partials2[v + 2]; w++;
                sum1 += matrices1[w] * partials1[v + 3];
                sum2 += matrices2[w] * partials2[v + 3]; w++;
                w++; // increment for the extra column at the end
                partials3[u] = sum1 * sum2; u++;

                sum1 = matrices1[w] * partials1[v];
                sum2 = matrices2[w] * partials2[v]; w++;
                sum1 += matrices1[w] * partials1[v + 1];
                sum2 += matrices2[w] * partials2[v + 1]; w++;
                sum1 += matrices1[w] * partials1[v + 2];
                sum2 += matrices2[w] * partials2[v + 2]; w++;
                sum1 += matrices1[w] * partials1[v + 3];
                sum2 += matrices2[w] * partials2[v + 3];
                partials3[u] = sum1 * sum2; u++;

                v += 4;
            }
        }
    }


}