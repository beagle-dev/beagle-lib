package benchmark;

import beagle.Beagle;
import beagle.BeagleFactory;

/**
 * Created by IntelliJ IDEA.
 * User: rambaut
 * Date: Aug 1, 2009
 * Time: 11:48:51 PM
 * To change this template use File | Settings | File Templates.
 */
public class BenchmarkLikelihoodCore {
    private final static int STATE_COUNT = 4;

    private final static String human = "AGAAATATGTCTGATAAAAGAGTTACTTTGATAGAGTAAATAATAGGAGCTTAAACCCCCTTATTTCTACTAGGACTATGAGAATCGAACCCATCCCTGAGAATCCAAAATTCTCCGTGCCACCTATCACACCCCATCCTAAGTAAGGTCAGCTAAATAAGCTATCGGGCCCATACCCCGAAAATGTTGGTTATACCCTTCCCGTACTAAGAAATTTAGGTTAAATACAGACCAAGAGCCTTCAAAGCCCTCAGTAAGTTG-CAATACTTAATTTCTGTAAGGACTGCAAAACCCCACTCTGCATCAACTGAACGCAAATCAGCCACTTTAATTAAGCTAAGCCCTTCTAGACCAATGGGACTTAAACCCACAAACACTTAGTTAACAGCTAAGCACCCTAATCAAC-TGGCTTCAATCTAAAGCCCCGGCAGG-TTTGAAGCTGCTTCTTCGAATTTGCAATTCAATATGAAAA-TCACCTCGGAGCTTGGTAAAAAGAGGCCTAACCCCTGTCTTTAGATTTACAGTCCAATGCTTCA-CTCAGCCATTTTACCACAAAAAAGGAAGGAATCGAACCCCCCAAAGCTGGTTTCAAGCCAACCCCATGGCCTCCATGACTTTTTCAAAAGGTATTAGAAAAACCATTTCATAACTTTGTCAAAGTTAAATTATAGGCT-AAATCCTATATATCTTA-CACTGTAAAGCTAACTTAGCATTAACCTTTTAAGTTAAAGATTAAGAGAACCAACACCTCTTTACAGTGA";
    private final static String chimp = "AGAAATATGTCTGATAAAAGAATTACTTTGATAGAGTAAATAATAGGAGTTCAAATCCCCTTATTTCTACTAGGACTATAAGAATCGAACTCATCCCTGAGAATCCAAAATTCTCCGTGCCACCTATCACACCCCATCCTAAGTAAGGTCAGCTAAATAAGCTATCGGGCCCATACCCCGAAAATGTTGGTTACACCCTTCCCGTACTAAGAAATTTAGGTTAAGCACAGACCAAGAGCCTTCAAAGCCCTCAGCAAGTTA-CAATACTTAATTTCTGTAAGGACTGCAAAACCCCACTCTGCATCAACTGAACGCAAATCAGCCACTTTAATTAAGCTAAGCCCTTCTAGATTAATGGGACTTAAACCCACAAACATTTAGTTAACAGCTAAACACCCTAATCAAC-TGGCTTCAATCTAAAGCCCCGGCAGG-TTTGAAGCTGCTTCTTCGAATTTGCAATTCAATATGAAAA-TCACCTCAGAGCTTGGTAAAAAGAGGCTTAACCCCTGTCTTTAGATTTACAGTCCAATGCTTCA-CTCAGCCATTTTACCACAAAAAAGGAAGGAATCGAACCCCCTAAAGCTGGTTTCAAGCCAACCCCATGACCTCCATGACTTTTTCAAAAGATATTAGAAAAACTATTTCATAACTTTGTCAAAGTTAAATTACAGGTT-AACCCCCGTATATCTTA-CACTGTAAAGCTAACCTAGCATTAACCTTTTAAGTTAAAGATTAAGAGGACCGACACCTCTTTACAGTGA";
    private final static String gorilla = "AGAAATATGTCTGATAAAAGAGTTACTTTGATAGAGTAAATAATAGAGGTTTAAACCCCCTTATTTCTACTAGGACTATGAGAATTGAACCCATCCCTGAGAATCCAAAATTCTCCGTGCCACCTGTCACACCCCATCCTAAGTAAGGTCAGCTAAATAAGCTATCGGGCCCATACCCCGAAAATGTTGGTCACATCCTTCCCGTACTAAGAAATTTAGGTTAAACATAGACCAAGAGCCTTCAAAGCCCTTAGTAAGTTA-CAACACTTAATTTCTGTAAGGACTGCAAAACCCTACTCTGCATCAACTGAACGCAAATCAGCCACTTTAATTAAGCTAAGCCCTTCTAGATCAATGGGACTCAAACCCACAAACATTTAGTTAACAGCTAAACACCCTAGTCAAC-TGGCTTCAATCTAAAGCCCCGGCAGG-TTTGAAGCTGCTTCTTCGAATTTGCAATTCAATATGAAAT-TCACCTCGGAGCTTGGTAAAAAGAGGCCCAGCCTCTGTCTTTAGATTTACAGTCCAATGCCTTA-CTCAGCCATTTTACCACAAAAAAGGAAGGAATCGAACCCCCCAAAGCTGGTTTCAAGCCAACCCCATGACCTTCATGACTTTTTCAAAAGATATTAGAAAAACTATTTCATAACTTTGTCAAGGTTAAATTACGGGTT-AAACCCCGTATATCTTA-CACTGTAAAGCTAACCTAGCGTTAACCTTTTAAGTTAAAGATTAAGAGTATCGGCACCTCTTTGCAGTGA";

    private static int[] getStates(String sequence) {
        int[] states = new int[sequence.length()];

        for (int i = 0; i < sequence.length(); i++) {
            switch (sequence.charAt(i)) {
                case 'A':
                    states[i] = 0;
                    break;
                case 'C':
                    states[i] = 1;
                    break;
                case 'G':
                    states[i] = 2;
                    break;
                case 'T':
                    states[i] = 3;
                    break;
                default:
                    states[i] = 4;
                    break;
            }
        }
        return states;
    }

    private static double[] getPartials(String sequence) {
        double[] partials = new double[sequence.length() * 4];

        int k = 0;
        for (int i = 0; i < sequence.length(); i++) {
            switch (sequence.charAt(i)) {
                case 'A':
                    partials[k++] = 1;
                    partials[k++] = 0;
                    partials[k++] = 0;
                    partials[k++] = 0;
                    break;
                case 'C':
                    partials[k++] = 0;
                    partials[k++] = 1;
                    partials[k++] = 0;
                    partials[k++] = 0;
                    break;
                case 'G':
                    partials[k++] = 0;
                    partials[k++] = 0;
                    partials[k++] = 1;
                    partials[k++] = 0;
                    break;
                case 'T':
                    partials[k++] = 0;
                    partials[k++] = 0;
                    partials[k++] = 0;
                    partials[k++] = 1;
                    break;
                default:
                    partials[k++] = 1;
                    partials[k++] = 1;
                    partials[k++] = 1;
                    partials[k++] = 1;
                    break;
            }
        }
        return partials;
    }

    public static void main(String[] argv) {

        // is nucleotides...
        int stateCount = 4;

        // get the number of site patterns
        int nPatterns = human.length();

        // create an instance of the BEAGLE library
//        LikelihoodCore instance = new NativeNucleotideLikelihoodCore();
        LikelihoodCore instance = new NucleotideLikelihoodCore();
        if (instance == null) {
            System.err.println("Failed to obtain LikelihoodCore instance");
            System.exit(1);
        }
        instance.initialize(5, nPatterns, 1, true);

        instance.setNodeStates(0, getStates(human));
        instance.setNodeStates(1, getStates(chimp));
        instance.setNodeStates(2, getStates(gorilla));
        instance.setNodePartials(3, new double[nPatterns * STATE_COUNT]);
        instance.setNodePartials(4, new double[nPatterns * STATE_COUNT]);
        // set the sequences for each tip using partial likelihood arrays
//        instance.setPartials(0, getPartials(human));
//        instance.setPartials(1, getPartials(chimp));
//        instance.setPartials(2, getPartials(gorilla));

        // create base frequency array
        final double[] freqs = { 0.25, 0.25, 0.25, 0.25 };

        // create an array containing site category weights
        final double[] weights = { 1.0 };

        // an eigen decomposition for the JC69 model
        final double[] evec = {
                1.0,  2.0,  0.0,  0.5,
                1.0,  -2.0,  0.5,  0.0,
                1.0,  2.0, 0.0,  -0.5,
                1.0,  -2.0,  -0.5,  0.0
        };

        final double[] ivec = {
                0.25,  0.25,  0.25,  0.25,
                0.125,  -0.125,  0.125,  -0.125,
                0.0,  1.0,  0.0,  -1.0,
                1.0,  0.0,  -1.0,  0.0
        };

        double[] eval = { 0.0, -1.3333333333333333, -1.3333333333333333, -1.3333333333333333 };

        double[] cMatrix = new double[STATE_COUNT * STATE_COUNT * STATE_COUNT];
        
        // set the Eigen decomposition
        int l =0;
        for (int i = 0; i < STATE_COUNT; i++) {
            for (int j = 0; j < STATE_COUNT; j++) {
                for (int k = 0; k < STATE_COUNT; k++) {
                    cMatrix[l] = evec[(i * stateCount) + k] * ivec[(k * stateCount) + j];
                    l++;
                }
            }
        }

        // a list of indices and edge lengths
        int[] nodeIndices = { 0, 1, 2, 3 };
        double[] edgeLengths = { 0.1, 0.1, 0.2, 0.1 };

        // populate the transition matrices for the above edge lengths
        double[] tmp = new double[STATE_COUNT];

        double[] pMatrix = new double[STATE_COUNT * STATE_COUNT];

//	    if (DEBUG) System.err.println("1: Rate "+l+" = "+categoryRates[l]);
        for (int u = 0; u < edgeLengths.length; u++) {
            int n = 0;
            for (int i = 0; i < STATE_COUNT; i++) {
                tmp[i] =  Math.exp(eval[i] * edgeLengths[u]);
            }
//            if (DEBUG) System.err.println(new dr.math.matrixAlgebra.Vector(tmp));
            //        if (DEBUG) System.exit(-1);

            int m = 0;
            for (int i = 0; i < STATE_COUNT; i++) {
                for (int j = 0; j < STATE_COUNT; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < STATE_COUNT; k++) {
                        sum += cMatrix[m] * tmp[k];
                        m++;
                    }
                    //	    if (DEBUG) System.err.println("1: matrices[][]["+n+"] = "+sum);
                    if (sum > 0)
                        pMatrix[n] = sum;
                    else
                        pMatrix[n] = 0; // TODO Decision: set to -sum (as BEAST does)
                    n++;
                }
            }
            instance.setNodeMatrix(nodeIndices[u], 0, pMatrix);
        }


        int count = 10000000;
        System.out.println("Running " + count + " iterations...");
        long time0 = System.nanoTime();
        for (int i = 0; i < count; i++) {
            // update the partials
            instance.calculatePartials(0,1,3);
            instance.calculatePartials(2,3,4);
        }
        long time1 = System.nanoTime();
        System.out.println("Time = " + ((double)(time1 - time0) / 1000000000));

        double[] rootPartials = new double[nPatterns * STATE_COUNT];

        // calculate the site likelihoods at the root node
        instance.getPartials(4, rootPartials);

        double[] patternLogLik = new double[nPatterns];

        instance.calculateLogLikelihoods(rootPartials, freqs, patternLogLik);

        double logL = 0.0;
        for (int i = 0; i < nPatterns; i++) {
//            System.out.println("site lnL[" + i + "] = " + patternLogLik[i]);
            logL += patternLogLik[i];
        }

        System.out.println();
        System.out.println("logL = " + logL + " (PAUP logL = -1574.63623)");
    }
}