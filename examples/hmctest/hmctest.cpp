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

char *human = (char*)"GAGT";
char *chimp = (char*)"GAGG";
char *gorilla = (char*)"AAAT";

//char *human = (char*)"GAGAAATATGTCTGATAAAAGAGTTACTTTGATAGAGTAAATAATAGGAGCTTAAACCCCCTTATTTCTACTAGGACTATGAGAATCGAACCCATCCCTGAGAATCCAAAATTCTCCGTGCCACCTATCACACCCCATCCTAAGTAAGGTCAGCTAAATAAGCTATCGGGCCCATACCCCGAAAATGTTGGTTATACCCTTCCCGTACTAAGAAATTTAGGTTAAATACAGACCAAGAGCCTTCAAAGCCCTCAGTAAGTTG-CAATACTTAATTTCTGTAAGGACTGCAAAACCCCACTCTGCATCAACTGAACGCAAATCAGCCACTTTAATTAAGCTAAGCCCTTCTAGACCAATGGGACTTAAACCCACAAACACTTAGTTAACAGCTAAGCACCCTAATCAAC-TGGCTTCAATCTAAAGCCCCGGCAGG-TTTGAAGCTGCTTCTTCGAATTTGCAATTCAATATGAAAA-TCACCTCGGAGCTTGGTAAAAAGAGGCCTAACCCCTGTCTTTAGATTTACAGTCCAATGCTTCA-CTCAGCCATTTTACCACAAAAAAGGAAGGAATCGAACCCCCCAAAGCTGGTTTCAAGCCAACCCCATGGCCTCCATGACTTTTTCAAAAGGTATTAGAAAAACCATTTCATAACTTTGTCAAAGTTAAATTATAGGCT-AAATCCTATATATCTTA-CACTGTAAAGCTAACTTAGCATTAACCTTTTAAGTTAAAGATTAAGAGAACCAACACCTCTTTACAGTGA";
//char *chimp = (char*)"GGGAAATATGTCTGATAAAAGAATTACTTTGATAGAGTAAATAATAGGAGTTCAAATCCCCTTATTTCTACTAGGACTATAAGAATCGAACTCATCCCTGAGAATCCAAAATTCTCCGTGCCACCTATCACACCCCATCCTAAGTAAGGTCAGCTAAATAAGCTATCGGGCCCATACCCCGAAAATGTTGGTTACACCCTTCCCGTACTAAGAAATTTAGGTTAAGCACAGACCAAGAGCCTTCAAAGCCCTCAGCAAGTTA-CAATACTTAATTTCTGTAAGGACTGCAAAACCCCACTCTGCATCAACTGAACGCAAATCAGCCACTTTAATTAAGCTAAGCCCTTCTAGATTAATGGGACTTAAACCCACAAACATTTAGTTAACAGCTAAACACCCTAATCAAC-TGGCTTCAATCTAAAGCCCCGGCAGG-TTTGAAGCTGCTTCTTCGAATTTGCAATTCAATATGAAAA-TCACCTCAGAGCTTGGTAAAAAGAGGCTTAACCCCTGTCTTTAGATTTACAGTCCAATGCTTCA-CTCAGCCATTTTACCACAAAAAAGGAAGGAATCGAACCCCCTAAAGCTGGTTTCAAGCCAACCCCATGACCTCCATGACTTTTTCAAAAGATATTAGAAAAACTATTTCATAACTTTGTCAAAGTTAAATTACAGGTT-AACCCCCGTATATCTTA-CACTGTAAAGCTAACCTAGCATTAACCTTTTAAGTTAAAGATTAAGAGGACCGACACCTCTTTACAGTGA";
//char *gorilla = (char*)"AGAAAATATGTCTGATAAAAGAGTTACTTTGATAGAGTAAATAATAGAGGTTTAAACCCCCTTATTTCTACTAGGACTATGAGAATTGAACCCATCCCTGAGAATCCAAAATTCTCCGTGCCACCTGTCACACCCCATCCTAAGTAAGGTCAGCTAAATAAGCTATCGGGCCCATACCCCGAAAATGTTGGTCACATCCTTCCCGTACTAAGAAATTTAGGTTAAACATAGACCAAGAGCCTTCAAAGCCCTTAGTAAGTTA-CAACACTTAATTTCTGTAAGGACTGCAAAACCCTACTCTGCATCAACTGAACGCAAATCAGCCACTTTAATTAAGCTAAGCCCTTCTAGATCAATGGGACTCAAACCCACAAACATTTAGTTAACAGCTAAACACCCTAGTCAAC-TGGCTTCAATCTAAAGCCCCGGCAGG-TTTGAAGCTGCTTCTTCGAATTTGCAATTCAATATGAAAT-TCACCTCGGAGCTTGGTAAAAAGAGGCCCAGCCTCTGTCTTTAGATTTACAGTCCAATGCCTTA-CTCAGCCATTTTACCACAAAAAAGGAAGGAATCGAACCCCCCAAAGCTGGTTTCAAGCCAACCCCATGACCTTCATGACTTTTTCAAAAGATATTAGAAAAACTATTTCATAACTTTGTCAAGGTTAAATTACGGGTT-AAACCCCGTATATCTTA-CACTGTAAAGCTAACCTAGCGTTAACCTTTTAAGTTAAAGATTAAGAGTATCGGCACCTCTTTGCAGTGA";


int* getStates(char *sequence) {
	int n = strlen(sequence);
	int *states = (int*) malloc(sizeof(int) * n);
    
	for (int i = 0; i < n; i++) {
		switch (sequence[i]) {
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

double* getPartials(char *sequence) {
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

int main( int argc, const char* argv[] )
{ 
    
    bool scaling = false;
//    bool scaling = false; // disable scaling for now

    bool doJC = true;

    // is nucleotides...
    int stateCount = 4;
	
    // get the number of site patterns
	int nPatterns = strlen(human);

    // change # rate category to 2
//    int rateCategoryCount = 4;
    int rateCategoryCount = 2;
    
    int scaleCount = (scaling ? 7 : 0);

    bool useGpu = argc > 1 && strcmp(argv[1] , "--gpu") == 0;

    BeagleInstanceDetails instDetails;

    /// Doubled the size of partials buffer from 5 to 10
    
    // create an instance of the BEAGLE library
	int instance = beagleCreateInstance(
                                  3,				/**< Number of tip data elements (input) */
                                  10,	            /**< Number of partials buffers to create (input) */
                                  0,		        /**< Number of compact state representation buffers to create (input) */
                                  stateCount,		/**< Number of states in the continuous-time Markov chain (input) */
                                  nPatterns,		/**< Number of site patterns to be handled by the instance (input) */
                                  1,		        /**< Number of rate matrix eigen-decomposition buffers to allocate (input) */
                                  6 * 2,		    /**< Number of rate matrix buffers (input) */
                                  rateCategoryCount,/**< Number of rate categories (input) */
                                  scaleCount,       /**< Number of scaling buffers */
                                  NULL,			    /**< List of potential resource on which this instance is allowed (input, NULL implies no restriction */
                                  0,			    /**< Length of resourceList list (input) */
                            useGpu ?
                                  BEAGLE_FLAG_PROCESSOR_GPU | BEAGLE_FLAG_PRECISION_SINGLE | BEAGLE_FLAG_SCALERS_RAW:
                                  BEAGLE_FLAG_PROCESSOR_CPU | BEAGLE_FLAG_SCALERS_RAW,             	/**< Bit-flags indicating preferred implementation charactertistics, see BeagleFlags (input) */
                                  BEAGLE_FLAG_EIGEN_REAL,                 /**< Bit-flags indicating required implementation characteristics, see BeagleFlags (input) */
                                  &instDetails);
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
    
    
//    beagleSetTipStates(instance, 0, getStates(human));
//    beagleSetTipStates(instance, 1, getStates(chimp));
//    beagleSetTipStates(instance, 2, getStates(gorilla));
    
    // set the sequences for each tip using partial likelihood arrays
    double *humanPartials   = getPartials(human);
    double *chimpPartials   = getPartials(chimp);
    double *gorillaPartials = getPartials(gorilla);
    
	beagleSetTipPartials(instance, 0, humanPartials);
	beagleSetTipPartials(instance, 1, chimpPartials);
	beagleSetTipPartials(instance, 2, gorillaPartials);
    
#ifdef _WIN32
	std::vector<double> rates(rateCategoryCount);
#else
	double rates[rateCategoryCount];
#endif
//    for (int i = 0; i < rateCategoryCount; i++) {
//        rates[i] = 1.0;
////        rates[i] = 3.0 * (i + 1) / (2 * rateCategoryCount + 1);
//    }
    rates[0] = 0.14251623900062188;
    rates[1] = 1.857483760999378;
    
	beagleSetCategoryRates(instance, &rates[0]);
    
	double* patternWeights = (double*) malloc(sizeof(double) * nPatterns);
    
    for (int i = 0; i < nPatterns; i++) {
        patternWeights[i] = 1.0;
    }    

    beagleSetPatternWeights(instance, patternWeights);
	
    // create base frequency array
	double freqs[4] = { 0.1, 0.3, 0.2, 0.4 };
//    double freqs[4] = { 0.25, 0.25, 0.25, 0.25 };
    
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
    
//#ifndef JC
//	// an eigen decomposition for the 4-state 1-step circulant infinitesimal generator
//	double evec[4 * 4] = {
//			 -0.5,  0.6906786606674509,   0.15153543380548623, 0.5,
//			  0.5, -0.15153543380548576,  0.6906786606674498,  0.5,
//			 -0.5, -0.6906786606674498,  -0.15153543380548617, 0.5,
//			  0.5,  0.15153543380548554, -0.6906786606674503,  0.5
//	};
//
//	double ivec[4 * 4] = {
//			 -0.5,  0.5, -0.5,  0.5,
//			  0.6906786606674505, -0.15153543380548617, -0.6906786606674507,   0.15153543380548645,
//			  0.15153543380548568, 0.6906786606674509,  -0.15153543380548584, -0.6906786606674509,
//			  0.5,  0.5,  0.5,  0.5
//	};
//
//	double eval[8] = { -2.0, -1.0, -1.0, 0, 0, 1, -1, 0 };
//#else
//	// an eigen decomposition for the JC69 model
//	double evec[4 * 4] = {
//        1.0,  2.0,  0.0,  0.5,
//        1.0,  -2.0,  0.5,  0.0,
//        1.0,  2.0, 0.0,  -0.5,
//        1.0,  -2.0,  -0.5,  0.0
//	};
//
//	double ivec[4 * 4] = {
//        0.25,  0.25,  0.25,  0.25,
//        0.125,  -0.125,  0.125,  -0.125,
//        0.0,  1.0,  0.0,  -1.0,
//        1.0,  0.0,  -1.0,  0.0
//	};
//
//	double eval[8] = { 0.0, -1.3333333333333333, -1.3333333333333333, -1.3333333333333333,
//			0.0, 0.0, 0.0, 0.0 };
//#endif

    ///eigen decomposition of the HKY85 model
    double evec[4 * 4] = {
            0.9819805,  0.040022305,  0.04454354,  -0.5,
            -0.1091089, -0.002488732, 0.81606029,  -0.5,
            -0.1091089, -0.896939683, -0.11849713, -0.5,
            -0.1091089,  0.440330814, -0.56393254, -0.5
    };

    double ivec[4 * 4] = {
            0.9165151, -0.3533241, -0.1573578, -0.4058332,
            0.0,  0.2702596, -0.8372848,  0.5670252,
            0.0,  0.8113638, -0.2686725, -0.5426913,
            -0.2, -0.6, -0.4, -0.8
    };

    ///array of real parts + array of imaginary parts
    double eval[8] = { -1.42857105618099456, -1.42857095607719153, -1.42857087221423851, 0.0,
                       0.0, 0.0, 0.0, 0.0 };

    ///Q^T matrix
    double QT[4 * 4] = {
            -1.2857138,  0.1428570,  0.1428570,  0.1428570,
            0.4285712, -0.9999997,  0.4285714,  0.4285713,
            0.2857142,  0.2857143, -1.1428568,  0.2857142,
            0.5714284,  0.5714284,  0.5714284, -0.8571426
    };

    double Q[4 * 4 * 2] = {
            -1.285714,  0.4285712,  0.2857142,  0.5714284,
            0.142857, -0.9999997,  0.2857143,  0.5714284,
            0.142857,  0.4285714, -1.1428568,  0.5714284,
            0.142857,  0.4285713,  0.2857142, -0.8571426,
            -1.285714,  0.4285712,  0.2857142,  0.5714284,
            0.142857, -0.9999997,  0.2857143,  0.5714284,
            0.142857,  0.4285714, -1.1428568,  0.5714284,
            0.142857,  0.4285713,  0.2857142, -0.8571426
    };

    double Q2[4 * 4 * 2] = {
            1.8367333, -0.6122443, -0.4081629, -0.8163261,
            -0.2040814,  1.4285705, -0.4081632, -0.8163259,
            -0.2040814, -0.6122447,  1.6326522, -0.8163261,
            -0.2040814, -0.6122446, -0.4081630,  1.2244890,
            1.8367333, -0.6122443, -0.4081629, -0.8163261,
            -0.2040814,  1.4285705, -0.4081632, -0.8163259,
            -0.2040814, -0.6122447,  1.6326522, -0.8163261,
            -0.2040814, -0.6122446, -0.4081630,  1.2244890
    };

    std::vector<double> scaledQ(4 * 4 * 2);

    for (int rate = 0; rate < rateCategoryCount; ++rate) {
        for (int entry = 0; entry < stateCount * stateCount; ++entry) {
            scaledQ[entry + rate * stateCount * stateCount] = Q[entry + rate * stateCount * stateCount] * rates[rate];
        }
    }

    beagleSetTransitionMatrix(instance, 4, Q, 0.0);
    beagleSetTransitionMatrix(instance, 5, Q2, 0.0);

    // set the Eigen decomposition
	beagleSetEigenDecomposition(instance, 0, evec, ivec, eval);
    
    // a list of indices and edge lengths
	int nodeIndices[4] = { 0, 1, 2, 3 };
	double edgeLengths[4] = { 0.6, 0.6, 1.3, 0.7};
    
    // tell BEAGLE to populate the transition matrices for the above edge lengths
	beagleUpdateTransitionMatrices(instance,     // instance
	                         0,             // eigenIndex
	                         nodeIndices,   // probabilityIndices
	                         NULL,          // firstDerivativeIndices
	                         NULL,          // secondDervativeIndices
	                         edgeLengths,   // edgeLengths
	                         4);            // count

	int transposeIndices[4] = { 6, 7, 8, 9 };

    double* matrix1 = (double*) malloc(sizeof(double) * stateCount * stateCount * rateCategoryCount);
    double* matrix2 = (double*) malloc(sizeof(double) * stateCount * stateCount * rateCategoryCount);

    beagleGetTransitionMatrix(instance, 0, matrix1);

    beagleTransposeTransitionMatrices(instance, nodeIndices, transposeIndices, 4);

    beagleGetTransitionMatrix(instance, 6, matrix2);

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

    std::cout << "Matrix-transpose for node " << nodeId << std::endl;
    mat = matrix2;
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
    
    // create a list of partial likelihood update operations
    // the order is [dest, destScaling, source1, matrix1, source2, matrix2]
	BeagleOperation operations[2] = {
		3, (scaling ? 0 : BEAGLE_OP_NONE), BEAGLE_OP_NONE, 0, 0, 1, 1,
		4, (scaling ? 1 : BEAGLE_OP_NONE), BEAGLE_OP_NONE, 2, 2, 3, 3
	};

	int rootIndex = 4;
    
    // update the partials
	beagleUpdatePartials(instance,      // instance
                   operations,     // eigenIndex
                   2,              // operationCount
                   BEAGLE_OP_NONE);          // cumulative scaling index

    ///XJ: I decided to store the pre-order partials vector in reverse order as those of post-orders
    ///This means that the two indices to the partials of root nodes are adjacent.
    ///For any node, the indices of the two partials sum to 2*(partialsBufferCount + compactBufferCount) - 1


    int categoryWeightsIndex = 0;
    int stateFrequencyIndex = 0;
    int transpose = (stateCount == 4 || !useGpu) ? 0 : 6;
    // create a list of partial likelihood update operations
    // the order is [dest, destScaling, source1, matrix1, source2, matrix2]
    // destPartials point to the pre-order partials
    // partials1 = pre-order partials of the parent node
    // matrices1 = Ptr matrices of the current node (to the parent node)
    // partials2 = post-order partials of the sibling node
    // matrices2 = Ptr matrices of the sibling node (to the parent node)
    BeagleOperation pre_order_operations[4] = {
            6, (scaling ? 3 : BEAGLE_OP_NONE), BEAGLE_OP_NONE, 5, 3 + transpose, 2, 2,
            7, (scaling ? 4 : BEAGLE_OP_NONE), BEAGLE_OP_NONE, 5, 2 + transpose, 3, 3,
            8, (scaling ? 5 : BEAGLE_OP_NONE), BEAGLE_OP_NONE, 6, 1 + transpose, 0, 0,
            9, (scaling ? 6 : BEAGLE_OP_NONE), BEAGLE_OP_NONE, 6, 0 + transpose, 1, 1,
    };

    int rootPreIndex = 5;

    double *patternLogLik = (double*)malloc(sizeof(double) * nPatterns);

    int cumulativeScalingIndex = (scaling ? 2 : BEAGLE_OP_NONE);

    if (scaling) {
        int scalingFactorsCount = 2;
        int scalingFactorsIndices[2] = {0, 1};

        beagleResetScaleFactors(instance,
                                cumulativeScalingIndex);

        beagleAccumulateScaleFactors(instance,
                                     scalingFactorsIndices,
                                     scalingFactorsCount,
                                     cumulativeScalingIndex);
    }

    double logL = 0.0;

    // calculate the site likelihoods at the root node
    beagleCalculateRootLogLikelihoods(instance,               // instance
                                      (const int *)&rootIndex,// bufferIndices
                                      &categoryWeightsIndex,                // weights
                                      &stateFrequencyIndex,                  // stateFrequencies
                                      &cumulativeScalingIndex,// cumulative scaling index
                                      1,                      // count
                                      &logL);         // outLogLikelihoods


    std::vector<double> siteLogLikelihoods(nPatterns);
    beagleGetSiteLogLikelihoods(instance, siteLogLikelihoods.data());

    std::cerr << "site-log-like:";
    for (double logLike : siteLogLikelihoods) {
        std::cerr << " " << logLike;
    }
    std::cerr << std::endl;

    double * seerootPartials = (double*) malloc(sizeof(double) * stateCount * nPatterns * rateCategoryCount);
    int offset = 0;
    for (int c = 0; c < rateCategoryCount; ++c) {
        for (int p = 0; p < nPatterns; ++p) {
            for (int s = 0; s < stateCount; ++s) {
                seerootPartials[offset++] = freqs[s];
            }
        }
    }
    beagleSetPartials(instance, rootPreIndex, seerootPartials);
    fprintf(stderr, "Setting preroot: %d\n", rootPreIndex);

//    beagleSetRootPrePartials(instance, // TODO Remove from API -- not necessary?
//                             (const int *) &rootPreIndex,               // bufferIndices
//                             &stateFrequencyIndex,                  // stateFrequencies
//                             1);                                    // count

    // update the pre-order partials
    beagleUpdatePrePartials(instance,
                            pre_order_operations,
                            4,
                            BEAGLE_OP_NONE);

    fprintf(stdout, "logL = %.5f (R = -18.04619478977292)\n\n", logL);

    int postBufferIndices[4] = {1, 0, 2, 3};
    int preBufferIndices[4] = {8, 9, 7, 6};
    int firstDervIndices[4] = {4, 4, 4, 4};
    int secondDervIndices[4] = {5, 5, 5, 5};
    int categoryRatesIndex = categoryWeightsIndex;
    double* gradient = (double*) malloc(sizeof(double) * nPatterns * 4);
    double* diagonalHessian = (double*) malloc(sizeof(double) * nPatterns * 4);

    beagleCalculateEdgeDerivative(instance, postBufferIndices, preBufferIndices,
                                  rootIndex, firstDervIndices, secondDervIndices,
                                  categoryWeightsIndex, categoryRatesIndex,
                                  stateFrequencyIndex, &cumulativeScalingIndex,
                                  4, gradient, NULL);

    std::cout<<"Gradient: \n";
    for (int i = 0; i < 4; i++) {
        for (int m = 0; m < nPatterns; m++) {
            std::cout<<gradient[i * nPatterns + m]<<"  ";
        }
        std::cout<<std::endl;
    }

//    std::cout<<"Diagonal Hessian: \n";
//    for (int i = 0; i < 4; i++) {
//        for (int m = 0; m < nPatterns; m++) {
//            std::cout<<diagonalHessian[i * nPatterns + m]<<"  ";
//        }
//        std::cout<<std::endl;
//    }


//  print pre-order partials and edge length log-likelihood gradient to screen
//  TODO: implement gradient calculation according to beagleCalculateEdgeLogLikelihoods() in beagle.cpp
//  need to consider rate variation case


    double * seeprePartials  = (double*) malloc(sizeof(double) * stateCount * nPatterns * rateCategoryCount);
    double * seepostPartials = (double*) malloc(sizeof(double) * stateCount * nPatterns * rateCategoryCount);

    double * tmpNumerator = (double*)   malloc(sizeof(double)  * nPatterns * rateCategoryCount);

    double * grand_denominator = (double*) malloc(sizeof(double)  * nPatterns);
    double * grand_numerator = (double*) malloc(sizeof(double)  * nPatterns);
    /// state frequencies stored in freqs
    /// category weights stored in weights


    beagleGetPartials(instance, rootIndex, BEAGLE_OP_NONE, seerootPartials);
    for(int i = 0; i < 5; i++){
        for(int m = 0; m < nPatterns; m++){
            grand_denominator[m] = 0;
            grand_numerator[m] = 0;
        }
        int postBufferIndex = 4-i;
        int preBufferIndex = 5+i;
        beagleGetPartials(instance, preBufferIndex, BEAGLE_OP_NONE, seeprePartials);
        beagleGetPartials(instance, postBufferIndex, BEAGLE_OP_NONE, seepostPartials);

        double * prePartialsPtr = seeprePartials;
        double * postPartialsPtr = seepostPartials;

        double denominator = 0;
        double numerator = 0;

        double tmp = 0;
        int k, j, l, m, s, t;
        std::cout<<"Gradient for branch (of node) "<< 4 -i <<": \n";

        ///get likelihood for each rate category first
        double clikelihood[rateCategoryCount * nPatterns];
        l = 0; j = 0;
        for(s = 0; s < rateCategoryCount; s++){
            for(m = 0; m < nPatterns; m++){
                double clikelihood_tmp = 0.0;
                for(k=0; k < stateCount; k++){
                    clikelihood_tmp += freqs[k] * seerootPartials[l++];
                }
                clikelihood[j++] = clikelihood_tmp;
            }
        }

        ///now calculate weights
        t = 0;
        for(s = 0; s < rateCategoryCount; s++){
            double ws = weights[s];
            double rs = rates[s];
            for(m=0; m < nPatterns; m++){
                l = 0;
                numerator = 0;
                denominator = 0;
                for(k = 0; k < stateCount; k++){
                    tmp = 0.0;
                    for(j=0; j < stateCount; j++){
                        tmp += QT[l++]*prePartialsPtr[j];
                    }
                    numerator += tmp * postPartialsPtr[k];
                    denominator += postPartialsPtr[k] * prePartialsPtr[k];
                }
                postPartialsPtr += stateCount;
                prePartialsPtr  += stateCount;
                tmpNumerator[t] = ws * rs * numerator / denominator * clikelihood[t];
                //std::cout<< tmpNumerator[t]<<",  "<<ws*clikelihood[t]<<"  \n";
                grand_numerator[m] += tmpNumerator[t];
                grand_denominator[m] += ws * clikelihood[t];
                t++;
                std::cout<<numerator / denominator <<"  ";
            }
            std::cout<<std::endl;
        }

//        std::cout << "site-rate like";
//        for (s = 0; s < rateCategoryCount; ++s) {
//            double ws = weights[s];
//            for (m = 0; m < nPatterns; ++m) {
//                double like = 0;
//                for (k = 0; k < stateCount; ++k) {
//                    double product = seeprePartials[t] * seepostPartials[t];
//                    like += product;
//                    ++t;
//                }
//                std::cout << " " << like;
//            }
//        }
//        std::cout << std::endl;
//
//        int noCategory = -1;
//
//        std::vector<double> logLikelihoodPerCategory(nPatterns * rateCategoryCount);
//
//        beagleCalculateRootLogLikelihoods(instance,               // instance
//                                          (const int *)&rootIndex,// bufferIndices
//                                          (const int *)&noCategory,                // weights
//                                          &stateFrequencyIndex,                  // stateFrequencies
//                                          &cumulativeScalingIndex,// cumulative scaling index
//                                          1,                      // count
//                                          logLikelihoodPerCategory.data());         // outLogLikelihoods
//        std::cout << "siteLogLikelihood =";
//        for (int i = 0; i < nPatterns * rateCategoryCount; ++i) {
//            std::cout << " " << exp(logLikelihoodPerCategory[i]);
//        }
//        std::cout << std::endl;




//        BEAGLE_DLLEXPORT int beagleCalculateEdgeLogDerivatives(int instance,
//                                                               const int *postBufferIndices,
//                                                               const int *preBufferIndices,
//                                                               const int *firstDerivativeIndices,
//                                                               const int *secondDerivativeIndices,
//                                                               const int *categoryWeightsIndices,
//                                                               const int *categoryRatesIndices,
//                                                               const int *cumulativeScaleIndices,
//                                                               int count,
//                                                               const double *siteLogLikelihoods,
//                                                               double *outLogFirstDerivative,
//                                                               double *outLogDiagonalSecondDerivative);

//        exit(-1);


//        std::cout<<"  Grand numerator:\n    ";
//        for(m=0; m < nPatterns; m++){
//            std::cout<<grand_numerator[m]<< "  ";
//        }
//        std::cout<<"\n  Grand denominator:\n    ";
//        for(m=0; m < nPatterns; m++){
//            std::cout<<grand_denominator[m] << "  ";
//        }
//        std::cout<<"\n  Grand derivative:\n    ";
        for(m=0; m < nPatterns; m++){
            std::cout<<grand_numerator[m] / grand_denominator[m] << "  ";
        }

        std::cout<<std::endl;
//        for(m=0; m < nPatterns; m++){
//            l = 0;
//            numerator = 0;
//            denominator = 0;
//            for(k = 0; k < stateCount; k++){
//                tmp = 0.0;
//                for(j=0; j < stateCount; j++){
//                    tmp += QT[l++]*prePartialsPtr[j];
//                }
//                numerator += tmp * postPartialsPtr[k];
//                denominator += postPartialsPtr[k] * prePartialsPtr[k];
//            }
//            postPartialsPtr += stateCount;
//            prePartialsPtr  += stateCount;
//            std::cout<<numerator / denominator <<"  ";
//        }
//        std::cout<<std::endl;

        std::cout<<"Pre-order Partial for node "<< 4-i << ": \n";

        l = 0;
        for(s = 0; s < rateCategoryCount; s++){
            std::cout<<"  rate category"<< s+1<< ": \n";
            for(k = 0; k<nPatterns; k++){
                for(j=0; j < stateCount; j++){
                    std::cout<<seeprePartials[l++]<<", ";
                }
                std::cout<<std::endl;
            }
            std::cout<<std::endl;
        }

    }

    std::vector<double> firstBuffer(nPatterns);

    beagleSetTransitionMatrix(instance, 4, scaledQ.data(), 0.0);


    beagleCalculateEdgeLogDerivatives(instance,
                                      postBufferIndices, preBufferIndices,
                                      firstDervIndices,
                                      NULL,
                                      &categoryWeightsIndex,
                                      &categoryRatesIndex,
                                      &cumulativeScalingIndex,
                                      1,
                                      siteLogLikelihoods.data(),
                                      firstBuffer.data(),
                                      NULL);

    std::cout << "check:";
    for (double x : firstBuffer) {
        std::cout << " " << x;
    }
    std::cout << std::endl;


    free(patternWeights);
    
	free(patternLogLik);
	free(humanPartials);
	free(chimpPartials);
	free(gorillaPartials);
    free(seepostPartials);
    free(seeprePartials);
    free(seerootPartials);
    free(tmpNumerator);
    free(grand_denominator);
    free(grand_numerator);
    free(gradient);
    free(diagonalHessian);
    free(matrix1);
    free(matrix2);
    
    beagleFinalizeInstance(instance);

#ifdef _WIN32
    std::cout << "\nPress ENTER to exit...\n";
    fflush( stdout);
    fflush( stderr);
    getchar();
#endif
    
}
