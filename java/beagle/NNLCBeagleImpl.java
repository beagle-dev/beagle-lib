package beagle;

import benchmark.NativeNucleotideLikelihoodCore;

import java.util.logging.Logger;

public class NNLCBeagleImpl extends GeneralBeagleImpl {

    public static final boolean DEBUG = false;

    public final NativeNucleotideLikelihoodCore nnlc;

    public NNLCBeagleImpl(final int tipCount, final int partialsBufferCount, final int compactBufferCount, final int patternCount, final int eigenBufferCount, final int matrixBufferCount, final int categoryCount, final int scaleBufferCount) {
        super(tipCount, partialsBufferCount, compactBufferCount, 4, patternCount, eigenBufferCount, matrixBufferCount, categoryCount, scaleBufferCount);
//        Logger.getLogger("beagle").info("Constructing double-precision 4-state Java BEAGLE implementation.");
        nnlc = new NativeNucleotideLikelihoodCore();
        nnlc.initialize(tipCount, patternCount, categoryCount, true);
        
    }

    protected void updateStatesStates(int bufferIndex1, int matrixIndex1, int bufferIndex2, int matrixIndex2, int bufferIndex3)
    {
        double[] matrices1 = matrices[matrixIndex1];
        double[] matrices2 = matrices[matrixIndex2];

        int[] states1 = tipStates[bufferIndex1];
        int[] states2 = tipStates[bufferIndex2];

        double[] partials3 = partials[bufferIndex3];

        nnlc.calculateStatesStatesPruning(states1, matrices1, states2, matrices2, partials3);
    }

    protected void updateStatesPartials(int bufferIndex1, int matrixIndex1, int bufferIndex2, int matrixIndex2, int bufferIndex3)
    {
        double[] matrices1 = matrices[matrixIndex1];
        double[] matrices2 = matrices[matrixIndex2];

        int[] states1 = tipStates[bufferIndex1];
        double[] partials2 = partials[bufferIndex2];

        double[] partials3 = partials[bufferIndex3];

        nnlc.calculateStatesPartialsPruning(states1, matrices1, partials2, matrices2, partials3);
    }

    protected void updatePartialsPartials(int bufferIndex1, int matrixIndex1, int bufferIndex2, int matrixIndex2, int bufferIndex3)
    {
        double[] matrices1 = matrices[matrixIndex1];
        double[] matrices2 = matrices[matrixIndex2];

        double[] partials1 = partials[bufferIndex1];
        double[] partials2 = partials[bufferIndex2];

        double[] partials3 = partials[bufferIndex3];

        nnlc.calculatePartialsPartialsPruning(partials1, matrices1, partials2, matrices2, partials3);
    }

    @Override
    public void calculateRootLogLikelihoods(int[] bufferIndices, double[] weights, double[] stateFrequencies, int[] scaleIndices, int count, double[] outLogLikelihoods) {

        double[] rootPartials = partials[bufferIndices[0]];

        int u = 0;
        int v = 0;
        for (int k = 0; k < patternCount; k++) {

            tmpPartials[u] = rootPartials[v] * weights[0]; u++; v++;
            tmpPartials[u] = rootPartials[v] * weights[0]; u++; v++;
            tmpPartials[u] = rootPartials[v] * weights[0]; u++; v++;
            tmpPartials[u] = rootPartials[v] * weights[0]; u++; v++;
        }


        for (int j = 1; j < categoryCount; j++) {
            u = 0;
            for (int k = 0; k < patternCount; k++) {
                tmpPartials[u] += rootPartials[v] * weights[j]; u++; v++;
                tmpPartials[u] += rootPartials[v] * weights[j]; u++; v++;
                tmpPartials[u] += rootPartials[v] * weights[j]; u++; v++;
                tmpPartials[u] += rootPartials[v] * weights[j]; u++; v++;

            }
        }

        v = 0;
        for (int k = 0; k < patternCount; k++) {
            double sum = stateFrequencies[0] * tmpPartials[v];	v++;
            sum += stateFrequencies[1] * tmpPartials[v];	v++;
            sum += stateFrequencies[2] * tmpPartials[v];	v++;
            sum += stateFrequencies[3] * tmpPartials[v];	v++;
            outLogLikelihoods[k] = Math.log(sum);
        }

    }
}