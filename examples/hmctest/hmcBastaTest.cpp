/*
 *  tinyTest.c
 *  BEAGLE
 *
 *  Created by Andrew Rambaut on 20/03/2009.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <iostream>
#include <libhmsbeagle/BeagleImpl.h>
#include <cmath>
#include <vector>

//#define JC

#ifdef _WIN32
#include <vector>
#endif

#include "libhmsbeagle/beagle.h"

char *human = (char*)"A";
char *chimp = (char*)"G";
char *gorilla = (char*)"G";

//char *human = (char*)"G";
//char *chimp = (char*)"G";
//char *gorilla = (char*)"A";


//char *human = (char*)"GAGAAATATGTCTGATAAAAGAGTTACTTTGATAGAGTAAATAATAGGAGCTTAAACCCCCTTATTTCTACTAGGACTATGAGAATCGAACCCATCCCTGAGAATCCAAAATTCTCCGTGCCACCTATCACACCCCATCCTAAGTAAGGTCAGCTAAATAAGCTATCGGGCCCATACCCCGAAAATGTTGGTTATACCCTTCCCGTACTAAGAAATTTAGGTTAAATACAGACCAAGAGCCTTCAAAGCCCTCAGTAAGTTG-CAATACTTAATTTCTGTAAGGACTGCAAAACCCCACTCTGCATCAACTGAACGCAAATCAGCCACTTTAATTAAGCTAAGCCCTTCTAGACCAATGGGACTTAAACCCACAAACACTTAGTTAACAGCTAAGCACCCTAATCAAC-TGGCTTCAATCTAAAGCCCCGGCAGG-TTTGAAGCTGCTTCTTCGAATTTGCAATTCAATATGAAAA-TCACCTCGGAGCTTGGTAAAAAGAGGCCTAACCCCTGTCTTTAGATTTACAGTCCAATGCTTCA-CTCAGCCATTTTACCACAAAAAAGGAAGGAATCGAACCCCCCAAAGCTGGTTTCAAGCCAACCCCATGGCCTCCATGACTTTTTCAAAAGGTATTAGAAAAACCATTTCATAACTTTGTCAAAGTTAAATTATAGGCT-AAATCCTATATATCTTA-CACTGTAAAGCTAACTTAGCATTAACCTTTTAAGTTAAAGATTAAGAGAACCAACACCTCTTTACAGTGA";
//char *chimp = (char*)"GGGAAATATGTCTGATAAAAGAATTACTTTGATAGAGTAAATAATAGGAGTTCAAATCCCCTTATTTCTACTAGGACTATAAGAATCGAACTCATCCCTGAGAATCCAAAATTCTCCGTGCCACCTATCACACCCCATCCTAAGTAAGGTCAGCTAAATAAGCTATCGGGCCCATACCCCGAAAATGTTGGTTACACCCTTCCCGTACTAAGAAATTTAGGTTAAGCACAGACCAAGAGCCTTCAAAGCCCTCAGCAAGTTA-CAATACTTAATTTCTGTAAGGACTGCAAAACCCCACTCTGCATCAACTGAACGCAAATCAGCCACTTTAATTAAGCTAAGCCCTTCTAGATTAATGGGACTTAAACCCACAAACATTTAGTTAACAGCTAAACACCCTAATCAAC-TGGCTTCAATCTAAAGCCCCGGCAGG-TTTGAAGCTGCTTCTTCGAATTTGCAATTCAATATGAAAA-TCACCTCAGAGCTTGGTAAAAAGAGGCTTAACCCCTGTCTTTAGATTTACAGTCCAATGCTTCA-CTCAGCCATTTTACCACAAAAAAGGAAGGAATCGAACCCCCTAAAGCTGGTTTCAAGCCAACCCCATGACCTCCATGACTTTTTCAAAAGATATTAGAAAAACTATTTCATAACTTTGTCAAAGTTAAATTACAGGTT-AACCCCCGTATATCTTA-CACTGTAAAGCTAACCTAGCATTAACCTTTTAAGTTAAAGATTAAGAGGACCGACACCTCTTTACAGTGA";
//char *gorilla = (char*)"AGAAAATATGTCTGATAAAAGAGTTACTTTGATAGAGTAAATAATAGAGGTTTAAACCCCCTTATTTCTACTAGGACTATGAGAATTGAACCCATCCCTGAGAATCCAAAATTCTCCGTGCCACCTGTCACACCCCATCCTAAGTAAGGTCAGCTAAATAAGCTATCGGGCCCATACCCCGAAAATGTTGGTCACATCCTTCCCGTACTAAGAAATTTAGGTTAAACATAGACCAAGAGCCTTCAAAGCCCTTAGTAAGTTA-CAACACTTAATTTCTGTAAGGACTGCAAAACCCTACTCTGCATCAACTGAACGCAAATCAGCCACTTTAATTAAGCTAAGCCCTTCTAGATCAATGGGACTCAAACCCACAAACATTTAGTTAACAGCTAAACACCCTAGTCAAC-TGGCTTCAATCTAAAGCCCCGGCAGG-TTTGAAGCTGCTTCTTCGAATTTGCAATTCAATATGAAAT-TCACCTCGGAGCTTGGTAAAAAGAGGCCCAGCCTCTGTCTTTAGATTTACAGTCCAATGCCTTA-CTCAGCCATTTTACCACAAAAAAGGAAGGAATCGAACCCCCCAAAGCTGGTTTCAAGCCAACCCCATGACCTTCATGACTTTTTCAAAAGATATTAGAAAAACTATTTCATAACTTTGTCAAGGTTAAATTACGGGTT-AAACCCCGTATATCTTA-CACTGTAAAGCTAACCTAGCGTTAACCTTTTAAGTTAAAGATTAAGAGTATCGGCACCTCTTTGCAGTGA";

int* getStates(char *sequence, int repeats) {
    int n = strlen(sequence);
    int *states = (int*) malloc(sizeof(int) * n * repeats);

    int k = 0;
    for (int r = 0; r < repeats; ++r) {
        for (int i = 0; i < n; i++) {
            switch (sequence[i]) {
                case 'A':
                    states[k++] = 0;
                    break;
                case 'C':
                    states[k++] = 1;
                    break;
                case 'G':
                    states[k++] = 2;
                    break;
                case 'T':
                    states[k++] = 3;
                    break;
                default:
                    states[k++] = 4;
                    break;
            }
        }
    }
    return states;
}

double* getPartials(char *sequence, int repeats) {
    int n = strlen(sequence);
    double *partials = (double*)malloc(sizeof(double) * n * 4);

    int k = 0;
    for (int i = 0; i < n; i++) {
        switch (sequence[i]) {
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

void printFlags(long inFlags) {
    if (inFlags & BEAGLE_FLAG_PROCESSOR_CPU)      fprintf(stdout, " PROCESSOR_CPU");
    if (inFlags & BEAGLE_FLAG_PROCESSOR_GPU)      fprintf(stdout, " PROCESSOR_GPU");
    if (inFlags & BEAGLE_FLAG_PROCESSOR_FPGA)     fprintf(stdout, " PROCESSOR_FPGA");
    if (inFlags & BEAGLE_FLAG_PROCESSOR_CELL)     fprintf(stdout, " PROCESSOR_CELL");
    if (inFlags & BEAGLE_FLAG_PRECISION_DOUBLE)   fprintf(stdout, " PRECISION_DOUBLE");
    if (inFlags & BEAGLE_FLAG_PRECISION_SINGLE)   fprintf(stdout, " PRECISION_SINGLE");
    if (inFlags & BEAGLE_FLAG_COMPUTATION_ASYNCH) fprintf(stdout, " COMPUTATION_ASYNCH");
    if (inFlags & BEAGLE_FLAG_COMPUTATION_SYNCH)  fprintf(stdout, " COMPUTATION_SYNCH");
    if (inFlags & BEAGLE_FLAG_EIGEN_REAL)         fprintf(stdout, " EIGEN_REAL");
    if (inFlags & BEAGLE_FLAG_EIGEN_COMPLEX)      fprintf(stdout, " EIGEN_COMPLEX");
    if (inFlags & BEAGLE_FLAG_SCALING_MANUAL)     fprintf(stdout, " SCALING_MANUAL");
    if (inFlags & BEAGLE_FLAG_SCALING_AUTO)       fprintf(stdout, " SCALING_AUTO");
    if (inFlags & BEAGLE_FLAG_SCALING_ALWAYS)     fprintf(stdout, " SCALING_ALWAYS");
    if (inFlags & BEAGLE_FLAG_SCALING_DYNAMIC)    fprintf(stdout, " SCALING_DYNAMIC");
    if (inFlags & BEAGLE_FLAG_SCALERS_RAW)        fprintf(stdout, " SCALERS_RAW");
    if (inFlags & BEAGLE_FLAG_SCALERS_LOG)        fprintf(stdout, " SCALERS_LOG");
    if (inFlags & BEAGLE_FLAG_VECTOR_NONE)        fprintf(stdout, " VECTOR_NONE");
    if (inFlags & BEAGLE_FLAG_VECTOR_SSE)         fprintf(stdout, " VECTOR_SSE");
    if (inFlags & BEAGLE_FLAG_VECTOR_AVX)         fprintf(stdout, " VECTOR_AVX");
    if (inFlags & BEAGLE_FLAG_THREADING_NONE)     fprintf(stdout, " THREADING_NONE");
    if (inFlags & BEAGLE_FLAG_THREADING_OPENMP)   fprintf(stdout, " THREADING_OPENMP");
    if (inFlags & BEAGLE_FLAG_FRAMEWORK_CPU)      fprintf(stdout, " FRAMEWORK_CPU");
    if (inFlags & BEAGLE_FLAG_FRAMEWORK_CUDA)     fprintf(stdout, " FRAMEWORK_CUDA");
    if (inFlags & BEAGLE_FLAG_FRAMEWORK_OPENCL)   fprintf(stdout, " FRAMEWORK_OPENCL");
}

int main( int argc, const char* argv[] )
{
    // print resource list
    BeagleResourceList* rList;
    rList = beagleGetResourceList();
    fprintf(stdout, "Available resources:\n");
    for (int i = 0; i < rList->length; i++) {
        fprintf(stdout, "\tResource %i:\n\t\tName : %s\n", i, rList->list[i].name);
        fprintf(stdout, "\t\tDesc : %s\n", rList->list[i].description);
        fprintf(stdout, "\t\tFlags:");
        printFlags(rList->list[i].supportFlags);
        fprintf(stdout, "\n");
    }
    fprintf(stdout, "\n");

//    bool scaling = true;
    bool scaling = false; // disable scaling for now

    bool doJC = true;

    bool singlePrecision = false;
    bool useSSE = true;

    // is nucleotides...
    int stateCount = 4;

    int nRepeats = 1;

    // get the number of site patterns
    int nPatterns = strlen(human) * nRepeats;

    // change # rate category to 2
//    int rateCategoryCount = 4;
    int rateCategoryCount = 1;

    int scaleCount = (scaling ? 7 : 0);

    bool useGpu = argc > 1 && strcmp(argv[1] , "--gpu") == 0;

    bool useTipStates = false;

    int whichDevice = -1;
    if (useGpu) {
        if (argc > 2) {
            whichDevice = atol(argv[2]);
            if (whichDevice < 0) {
                whichDevice = -1;
            }
        }
    }

    BeagleInstanceDetails instDetails;

    long preferenceFlags = BEAGLE_FLAG_SCALERS_RAW;

    if (useGpu) {
        preferenceFlags |= BEAGLE_FLAG_PROCESSOR_GPU;
    } else {
        preferenceFlags |= BEAGLE_FLAG_PROCESSOR_CPU;
    }

    if (singlePrecision) {
        preferenceFlags |= BEAGLE_FLAG_PRECISION_SINGLE;
    } else {
        preferenceFlags |= BEAGLE_FLAG_PRECISION_DOUBLE;
    }

    long requirementFlags = BEAGLE_FLAG_EIGEN_REAL;
    if (useSSE) {
        requirementFlags |= BEAGLE_FLAG_VECTOR_SSE;
    } else {
        requirementFlags |= BEAGLE_FLAG_VECTOR_NONE;
    }

    // create an instance of the BEAGLE library
    int instance = beagleCreateInstance(
            0,				/**< Number of tip data elements (input) */
            20,	            /**< Number of partials buffers to create (input) */
            0,		        /**< Number of compact state representation buffers to create (input) */
            stateCount,		/**< Number of states in the continuous-time Markov chain (input) */
            1,		/**< Number of site patterns to be handled by the instance (input) */
            2,		        /**< Number of rate matrix eigen-decomposition buffers to allocate (input) */
            4,		    /**< Number of rate matrix buffers (input) */
            rateCategoryCount,/**< Number of rate categories (input) */
            1,       /**< Number of scaling buffers */
             NULL, /**< List of potential resource on which this instance is allowed (input, NULL implies no restriction */
            whichDevice >= 0 ? 1 : 0,			    /**< Length of resourceList list (input) */
            0,
            32, /**< Bit-flags indicating required implementation characteristics, see BeagleFlags (input) */
            &instDetails);

    beagleAllocateBastaBuffers(instance,5,4);

    if (instance < 0) {
        fprintf(stderr, "Failed to obtain BEAGLE instance\n\n");
        exit(1);
    }

    int rNumber = instDetails.resourceNumber;
    fprintf(stdout, "Using resource %i:\n", rNumber);
    fprintf(stdout, "\tRsrc Name : %s\n",instDetails.resourceName);
    fprintf(stdout, "\tImpl Name : %s\n", instDetails.implName);
    fprintf(stdout, "\tImpl Desc : %s\n", instDetails.implDescription);
    fprintf(stdout, "\n");

    if (useTipStates) {
        // set the sequences for each tip using state likelihood arrays
        int *humanStates = getStates(human, nRepeats);
        int *chimpStates = getStates(chimp, nRepeats);
        int *gorillaStates = getStates(gorilla, nRepeats);

        beagleSetTipStates(instance, 0, humanStates);
        beagleSetTipStates(instance, 1, chimpStates);
        beagleSetTipStates(instance, 2, gorillaStates);

        free(humanStates);
        free(chimpStates);
        free(gorillaStates);

    } else {
        // set the sequences for each tip using partial likelihood arrays
        double *humanPartials = getPartials(human, nRepeats);
        double *chimpPartials = getPartials(chimp, nRepeats);
        double *gorillaPartials = getPartials(gorilla, nRepeats);

        beagleSetPartials(instance, 0, humanPartials);
        beagleSetPartials(instance, 1, chimpPartials);
        beagleSetPartials(instance, 2, gorillaPartials);

        free(humanPartials);
        free(chimpPartials);
        free(gorillaPartials);
    }

#ifdef _WIN32
    std::vector<double> rates(rateCategoryCount);
#else
    double rates[rateCategoryCount];
#endif




    rates[0] = 1.0;

    beagleSetCategoryRates(instance, &rates[0]);

    double* patternWeights = (double*) malloc(sizeof(double) * nPatterns);

    for (int i = 0; i < nPatterns; i++) {
        patternWeights[i] = 1.0;
    }

    beagleSetPatternWeights(instance, patternWeights);

    // create base frequency array
//    double freqs[4] = { 0.1, 0.3, 0.2, 0.4 };
    double freqs[4] = { 0.25, 0.25, 0.25, 0.25 };

    beagleSetStateFrequencies(instance, 0, freqs);

    // create an array containing site category weights
#ifdef _WIN32
    std::vector<double> weights(rateCategoryCount);
#else
    double weights[rateCategoryCount];
#endif
    for (int i = 0; i < rateCategoryCount; i++) {
        weights[i] = 1.0/rateCategoryCount;
//        weights[i] = 2.0 * double(i + 1)/ double(rateCategoryCount * (rateCategoryCount + 1));
    }

    beagleSetCategoryWeights(instance, 0, &weights[0]);


    ///eigen decomposition of the HKY85 model
    double evec[4 * 4] = {-0.6532830113402044, 0.4999979842955143, -0.27059808322435847, 0.5000000001499998, 0.2705993899275025, -0.4999994206514062, -0.6532813709797044, 0.4999999998499999, -0.27059644259896054, -0.5000005794441943, 0.65328170474785, 0.5000000000499999, 0.6532800643812152, 0.5000020156000878, 0.2705977493031389, 0.49999999994999994};

    double ivec[4 * 4] = {-0.6532830113402049, 0.270599389927502, -0.27059644259896076, 0.6532800643812157, 0.49999798429551423, -0.49999942065140635, -0.5000005794441948, 0.500002015600088, -0.2705980832243587, -0.6532813709797037, 0.6532817047478499, 0.2705977493031391, 0.5000000001499999, 0.49999999985000004, 0.5000000000500006, 0.49999999994999994};

    ///array of real parts + array of imaginary parts
    double eval[8] = {-1.0000000001414213, -1.0000000000999998, -0.9999999998585786, 1.0000023031864202E-10, 0.0, 0.0, 0.0, 0.0};

    ///Q^T matrix
//    double QT[4 * 4] = {
//            -1.2857138,  0.1428570,  0.1428570,  0.1428570,
//            0.4285712, -0.9999997,  0.4285714,  0.4285713,
//            0.2857142,  0.2857143, -1.1428568,  0.2857142,
//            0.5714284,  0.5714284,  0.5714284, -0.8571426
//    };

    double Q[4 * 4 * 1] = {
            -0.7499999999, 0.2499999999, 0.2500000001, 0.2499999999,
            0.2500000001, -0.7500000001, 0.2500000001, 0.2499999999,
            0.2500000001, 0.2499999999, -0.7499999999, 0.2499999999,
            0.2500000001, 0.2499999999, 0.2500000001, -0.7500000001
    };


    std::vector<double> scaledQ(4 * 4 * 1);
//   std::vector<double> scaledQT(4 * 4 * 2);

    for (int rate = 0; rate < rateCategoryCount; ++rate) {
        for (int entry = 0; entry < stateCount * stateCount; ++entry) {
            scaledQ[entry + rate * stateCount * stateCount] = Q[entry + rate * stateCount * stateCount] * rates[rate];
        }
    }

    // set the Eigen decomposition
    beagleSetEigenDecomposition(instance, 0, evec, ivec, eval);

    // a list of indices and edge lengths
    int transitionMatrixIndices[3] = { 0, 1, 2};
    double edgeLengths[3] = {0, 0.6, 0.7};
    //double edgeLengths[4] = { 1.0, 1.0, 1.0, 1.0};

    // tell BEAGLE to populate the transition matrices for the above edge lengths
    beagleUpdateTransitionMatrices(instance,     // instance
                                   0,             // eigenIndex
                                   transitionMatrixIndices,   // probabilityIndices
                                   NULL,          // firstDerivativeIndices
                                   NULL,          // secondDervativeIndices
                                    edgeLengths,   // edgeLengths
                                   3);            // count

    beagleSetTransitionMatrix(instance, 3, scaledQ.data(), 0.0);


    double* matrix1 = (double*) malloc(sizeof(double) * stateCount * stateCount * rateCategoryCount);
    double* matrix2 = (double*) malloc(sizeof(double) * stateCount * stateCount * rateCategoryCount);
    double* matrix3 = (double*) malloc(sizeof(double) * stateCount * stateCount * rateCategoryCount);

    beagleGetTransitionMatrix(instance, 0, matrix1);
    beagleGetTransitionMatrix(instance, 1, matrix2);
    beagleGetTransitionMatrix(instance, 2, matrix3);

    int nodeId = 0;
    std::cout << "Matrix for node " << nodeId << std::endl;
    double* mat = matrix1;
    {
        int offset = 0;
        for (int r = 0; r < rateCategoryCount; r++) {
            std::cout << "  rate category" << r + 1 << ": \n";
            for (int i = 0; i < stateCount; i++) {
                for (int j = 0; j < stateCount; j++) {
                    std::cout << mat[offset++] << ", ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }

    std::cout << "Matrix for node " << nodeId << std::endl;
    double* mat2 = matrix2;
    {
        int offset = 0;
        for (int r = 0; r < rateCategoryCount; r++) {
            std::cout << "  rate category" << r + 1 << ": \n";
            for (int i = 0; i < stateCount; i++) {
                for (int j = 0; j < stateCount; j++) {
                    std::cout << mat2[offset++] << ", ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }

    std::cout << "Matrix for node " << nodeId << std::endl;
    double* mat3 = matrix3;
    {
        int offset = 0;
        for (int r = 0; r < rateCategoryCount; r++) {
            std::cout << "  rate category" << r + 1 << ": \n";
            for (int i = 0; i < stateCount; i++) {
                for (int j = 0; j < stateCount; j++) {
                    std::cout << mat3[offset++] << ", ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }

    // create a list of partial likelihood update operations
    // the order is [dest, destScaling, source1, matrix1, source2, matrix2]
    BastaOperation operations[4] = {3, 2, 0, -1, -1, 3, -1, 0, 4, 0, 1, 1, 1, 5, 6, 1, 7, 3, 1, -1, -1, 7, -1, 1, 8, 4, 2, 7, 2, 9, 10, 2};

    int rootIndex = 4;
    int intervals[4] = {0,1,3,4};
    int intervalSize = 4;
    // update the partials
    beagleUpdateBastaPartials(instance,
                         operations,
                         4,
                         intervals,
                         intervalSize,
                         0,
                         0);


    double * nodePartials = (double*) malloc(sizeof(double) * stateCount * nPatterns * rateCategoryCount);

    for(int i = 0; i < 5; i++){
        beagleGetPartials(instance, 3+i, BEAGLE_OP_NONE, nodePartials);

        std::cout<<"node Partial for node "<< 4-i << ": \n";

        int l = 0;
        for(int s = 0; s < rateCategoryCount; s++){
            std::cout<<"  rate category"<< s+1<< ": \n";
                for(int j=0; j < stateCount; j++){
                    std::cout<<nodePartials[l++]<<", ";
                }
                std::cout<<std::endl;
            std::cout<<std::endl;
        }
    }

    double coalescent[4] = {0,0,0,0};
    beagleGetBastaBuffer(instance, 0, coalescent);
    int a = 0;
    for(int j=0; j < intervalSize; j++) {
        std::cout << "coalescent:" << coalescent[a++] << std::endl;
    }
    //double *logL = new double(0.0);
    double *logL = new double[1];
    beagleAccumulateBastaPartials(instance, operations, 4, intervals, intervalSize,edgeLengths,0,0,logL);
    std::cout << "logL:" << logL[0] << std::endl;

    beagleUpdateTransitionMatricesGrad(instance,transitionMatrixIndices, edgeLengths,3);
    beagleUpdateBastaPartialsGrad(instance, operations, 4, intervals, intervalSize, 0, 0);

    double* gradients = (double*) malloc(sizeof(double) * stateCount * stateCount);
    beagleAccumulateBastaPartialsGrad(instance,operations,4,intervals,intervalSize,edgeLengths,0,0, gradients);

    std::cout << "check gradients  :";
    for (int i = 0; i < stateCount * stateCount; ++i) {
        std::cout << " " << gradients[i];
    }

    free(logL);
    free(gradients);
    free(matrix1);
    free(matrix2);
    free(matrix3);

    beagleFinalizeInstance(instance);

#ifdef _WIN32
    std::cout << "\nPress ENTER to exit...\n";
    fflush( stdout);
    fflush( stderr);
    getchar();
#endif

}

//Gradient:
// [0.5460353604288144, 0.07696854752416527, 1.530787625539162, 0.0769685475746027]
// [0.5305122750471234, 0.061445462517505, 0.7326799740643887, 0.0614454625580404]
// [0.8643317616354624, 0.09249163248364844, 0.28382997589489234, 0.0924916325448268]
// [0.5305122752959315, 0.0614454625381937, 0.7326799741367679, 0.0614454625787291]