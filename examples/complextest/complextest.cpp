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

//#define JC

#ifdef _WIN32
	#include <vector>
#endif

#include "libhmsbeagle/beagle.h"

char *human = (char*)"AGAAATATGTCTGATAAAAGAGTTACTTTGATAGAGTAAATAATAGGAGCTTAAACCCCCTTATTTCTACTAGGACTATGAGAATCGAACCCATCCCTGAGAATCCAAAATTCTCCGTGCCACCTATCACACCCCATCCTAAGTAAGGTCAGCTAAATAAGCTATCGGGCCCATACCCCGAAAATGTTGGTTATACCCTTCCCGTACTAAGAAATTTAGGTTAAATACAGACCAAGAGCCTTCAAAGCCCTCAGTAAGTTG-CAATACTTAATTTCTGTAAGGACTGCAAAACCCCACTCTGCATCAACTGAACGCAAATCAGCCACTTTAATTAAGCTAAGCCCTTCTAGACCAATGGGACTTAAACCCACAAACACTTAGTTAACAGCTAAGCACCCTAATCAAC-TGGCTTCAATCTAAAGCCCCGGCAGG-TTTGAAGCTGCTTCTTCGAATTTGCAATTCAATATGAAAA-TCACCTCGGAGCTTGGTAAAAAGAGGCCTAACCCCTGTCTTTAGATTTACAGTCCAATGCTTCA-CTCAGCCATTTTACCACAAAAAAGGAAGGAATCGAACCCCCCAAAGCTGGTTTCAAGCCAACCCCATGGCCTCCATGACTTTTTCAAAAGGTATTAGAAAAACCATTTCATAACTTTGTCAAAGTTAAATTATAGGCT-AAATCCTATATATCTTA-CACTGTAAAGCTAACTTAGCATTAACCTTTTAAGTTAAAGATTAAGAGAACCAACACCTCTTTACAGTGA";
char *chimp = (char*)"AGAAATATGTCTGATAAAAGAATTACTTTGATAGAGTAAATAATAGGAGTTCAAATCCCCTTATTTCTACTAGGACTATAAGAATCGAACTCATCCCTGAGAATCCAAAATTCTCCGTGCCACCTATCACACCCCATCCTAAGTAAGGTCAGCTAAATAAGCTATCGGGCCCATACCCCGAAAATGTTGGTTACACCCTTCCCGTACTAAGAAATTTAGGTTAAGCACAGACCAAGAGCCTTCAAAGCCCTCAGCAAGTTA-CAATACTTAATTTCTGTAAGGACTGCAAAACCCCACTCTGCATCAACTGAACGCAAATCAGCCACTTTAATTAAGCTAAGCCCTTCTAGATTAATGGGACTTAAACCCACAAACATTTAGTTAACAGCTAAACACCCTAATCAAC-TGGCTTCAATCTAAAGCCCCGGCAGG-TTTGAAGCTGCTTCTTCGAATTTGCAATTCAATATGAAAA-TCACCTCAGAGCTTGGTAAAAAGAGGCTTAACCCCTGTCTTTAGATTTACAGTCCAATGCTTCA-CTCAGCCATTTTACCACAAAAAAGGAAGGAATCGAACCCCCTAAAGCTGGTTTCAAGCCAACCCCATGACCTCCATGACTTTTTCAAAAGATATTAGAAAAACTATTTCATAACTTTGTCAAAGTTAAATTACAGGTT-AACCCCCGTATATCTTA-CACTGTAAAGCTAACCTAGCATTAACCTTTTAAGTTAAAGATTAAGAGGACCGACACCTCTTTACAGTGA";
char *gorilla = (char*)"AGAAATATGTCTGATAAAAGAGTTACTTTGATAGAGTAAATAATAGAGGTTTAAACCCCCTTATTTCTACTAGGACTATGAGAATTGAACCCATCCCTGAGAATCCAAAATTCTCCGTGCCACCTGTCACACCCCATCCTAAGTAAGGTCAGCTAAATAAGCTATCGGGCCCATACCCCGAAAATGTTGGTCACATCCTTCCCGTACTAAGAAATTTAGGTTAAACATAGACCAAGAGCCTTCAAAGCCCTTAGTAAGTTA-CAACACTTAATTTCTGTAAGGACTGCAAAACCCTACTCTGCATCAACTGAACGCAAATCAGCCACTTTAATTAAGCTAAGCCCTTCTAGATCAATGGGACTCAAACCCACAAACATTTAGTTAACAGCTAAACACCCTAGTCAAC-TGGCTTCAATCTAAAGCCCCGGCAGG-TTTGAAGCTGCTTCTTCGAATTTGCAATTCAATATGAAAT-TCACCTCGGAGCTTGGTAAAAAGAGGCCCAGCCTCTGTCTTTAGATTTACAGTCCAATGCCTTA-CTCAGCCATTTTACCACAAAAAAGGAAGGAATCGAACCCCCCAAAGCTGGTTTCAAGCCAACCCCATGACCTTCATGACTTTTTCAAAAGATATTAGAAAAACTATTTCATAACTTTGTCAAGGTTAAATTACGGGTT-AAACCCCGTATATCTTA-CACTGTAAAGCTAACCTAGCGTTAACCTTTTAAGTTAAAGATTAAGAGTATCGGCACCTCTTTGCAGTGA";

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
    
    bool scaling = true;
    
    bool doJC = true;

    // is nucleotides...
    int stateCount = 4;
	
    // get the number of site patterns
	int nPatterns = strlen(human);
    
    int rateCategoryCount = 4;
    
    int scaleCount = (scaling ? 3 : 0);
    
    BeagleInstanceDetails instDetails;
    
    // create an instance of the BEAGLE library
	int instance = beagleCreateInstance(
                                  3,				/**< Number of tip data elements (input) */
                                  5,	            /**< Number of partials buffers to create (input) */
                                  0,		        /**< Number of compact state representation buffers to create (input) */
                                  stateCount,		/**< Number of states in the continuous-time Markov chain (input) */
                                  nPatterns,		/**< Number of site patterns to be handled by the instance (input) */
                                  1,		        /**< Number of rate matrix eigen-decomposition buffers to allocate (input) */
                                  4,		        /**< Number of rate matrix buffers (input) */
                                  rateCategoryCount,/**< Number of rate categories (input) */
                                  scaleCount,       /**< Number of scaling buffers */
                                  NULL,			    /**< List of potential resource on which this instance is allowed (input, NULL implies no restriction */
                                  0,			    /**< Length of resourceList list (input) */
                                  BEAGLE_FLAG_PROCESSOR_GPU,             	/**< Bit-flags indicating preferred implementation charactertistics, see BeagleFlags (input) */
                                  BEAGLE_FLAG_EIGEN_COMPLEX,                 /**< Bit-flags indicating required implementation characteristics, see BeagleFlags (input) */
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
    for (int i = 0; i < rateCategoryCount; i++) {
        rates[i] = 1.0;
    }
    
	beagleSetCategoryRates(instance, &rates[0]);
    
	double* patternWeights = (double*) malloc(sizeof(double) * nPatterns);
    
    for (int i = 0; i < nPatterns; i++) {
        patternWeights[i] = 1.0;
    }    

    beagleSetPatternWeights(instance, patternWeights);
	
    // create base frequency array
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
    }    
    
    beagleSetCategoryWeights(instance, 0, &weights[0]);
    
#ifndef JC
	// an eigen decomposition for the 4-state 1-step circulant infinitesimal generator
	double evec[4 * 4] = {
			 -0.5,  0.6906786606674509,   0.15153543380548623, 0.5,
			  0.5, -0.15153543380548576,  0.6906786606674498,  0.5,
			 -0.5, -0.6906786606674498,  -0.15153543380548617, 0.5,
			  0.5,  0.15153543380548554, -0.6906786606674503,  0.5
	};

	double ivec[4 * 4] = {
			 -0.5,  0.5, -0.5,  0.5,
			  0.6906786606674505, -0.15153543380548617, -0.6906786606674507,   0.15153543380548645,
			  0.15153543380548568, 0.6906786606674509,  -0.15153543380548584, -0.6906786606674509,
			  0.5,  0.5,  0.5,  0.5
	};

	double eval[8] = { -2.0, -1.0, -1.0, 0, 0, 1, -1, 0 };
#else
	// an eigen decomposition for the JC69 model
	double evec[4 * 4] = {
        1.0,  2.0,  0.0,  0.5,
        1.0,  -2.0,  0.5,  0.0,
        1.0,  2.0, 0.0,  -0.5,
        1.0,  -2.0,  -0.5,  0.0
	};
    
	double ivec[4 * 4] = {
        0.25,  0.25,  0.25,  0.25,
        0.125,  -0.125,  0.125,  -0.125,
        0.0,  1.0,  0.0,  -1.0,
        1.0,  0.0,  -1.0,  0.0
	};
    
	double eval[8] = { 0.0, -1.3333333333333333, -1.3333333333333333, -1.3333333333333333,
			0.0, 0.0, 0.0, 0.0 };
#endif
    // set the Eigen decomposition
	beagleSetEigenDecomposition(instance, 0, evec, ivec, eval);
    
    // a list of indices and edge lengths
	int nodeIndices[4] = { 0, 1, 2, 3 };
	double edgeLengths[4] = { 0.1, 0.1, 0.2, 0.1 };
    
    // tell BEAGLE to populate the transition matrices for the above edge lengths
	beagleUpdateTransitionMatrices(instance,     // instance
	                         0,             // eigenIndex
	                         nodeIndices,   // probabilityIndices
	                         NULL,          // firstDerivativeIndices
	                         NULL,          // secondDervativeIndices
	                         edgeLengths,   // edgeLengths
	                         4);            // count
    
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
    
	int categoryWeightsIndex = 0;
    int stateFrequencyIndex = 0;
    
	double logL = 0.0;
    
    // calculate the site likelihoods at the root node
	beagleCalculateRootLogLikelihoods(instance,               // instance
	                            (const int *)&rootIndex,// bufferIndices
	                            &categoryWeightsIndex,                // weights
	                            &stateFrequencyIndex,                  // stateFrequencies
                                &cumulativeScalingIndex,// cumulative scaling index
	                            1,                      // count
	                            &logL);         // outLogLikelihoods
        
#ifndef JC
	fprintf(stdout, "logL = %.5f (BEAST = -1665.38544)\n\n", logL);
#else
	fprintf(stdout, "logL = %.5f (PAUP = -1574.63624)\n\n", logL);
#endif
	
    free(patternWeights);
    
	free(patternLogLik);
	free(humanPartials);
	free(chimpPartials);
	free(gorillaPartials);
    
    beagleFinalizeInstance(instance);

#ifdef _WIN32
    std::cout << "\nPress ENTER to exit...\n";
    fflush( stdout);
    fflush( stderr);
    getchar();
#endif
    
}
