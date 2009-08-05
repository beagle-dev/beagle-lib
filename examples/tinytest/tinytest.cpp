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

#include "libhmsbeagle/beagle.h"

char *human = "AGAAATATGTCTGATAAAAGAGTTACTTTGATAGAGTAAATAATAGGAGCTTAAACCCCCTTATTTCTACTAGGACTATGAGAATCGAACCCATCCCTGAGAATCCAAAATTCTCCGTGCCACCTATCACACCCCATCCTAAGTAAGGTCAGCTAAATAAGCTATCGGGCCCATACCCCGAAAATGTTGGTTATACCCTTCCCGTACTAAGAAATTTAGGTTAAATACAGACCAAGAGCCTTCAAAGCCCTCAGTAAGTTG-CAATACTTAATTTCTGTAAGGACTGCAAAACCCCACTCTGCATCAACTGAACGCAAATCAGCCACTTTAATTAAGCTAAGCCCTTCTAGACCAATGGGACTTAAACCCACAAACACTTAGTTAACAGCTAAGCACCCTAATCAAC-TGGCTTCAATCTAAAGCCCCGGCAGG-TTTGAAGCTGCTTCTTCGAATTTGCAATTCAATATGAAAA-TCACCTCGGAGCTTGGTAAAAAGAGGCCTAACCCCTGTCTTTAGATTTACAGTCCAATGCTTCA-CTCAGCCATTTTACCACAAAAAAGGAAGGAATCGAACCCCCCAAAGCTGGTTTCAAGCCAACCCCATGGCCTCCATGACTTTTTCAAAAGGTATTAGAAAAACCATTTCATAACTTTGTCAAAGTTAAATTATAGGCT-AAATCCTATATATCTTA-CACTGTAAAGCTAACTTAGCATTAACCTTTTAAGTTAAAGATTAAGAGAACCAACACCTCTTTACAGTGA";
char *chimp = "AGAAATATGTCTGATAAAAGAATTACTTTGATAGAGTAAATAATAGGAGTTCAAATCCCCTTATTTCTACTAGGACTATAAGAATCGAACTCATCCCTGAGAATCCAAAATTCTCCGTGCCACCTATCACACCCCATCCTAAGTAAGGTCAGCTAAATAAGCTATCGGGCCCATACCCCGAAAATGTTGGTTACACCCTTCCCGTACTAAGAAATTTAGGTTAAGCACAGACCAAGAGCCTTCAAAGCCCTCAGCAAGTTA-CAATACTTAATTTCTGTAAGGACTGCAAAACCCCACTCTGCATCAACTGAACGCAAATCAGCCACTTTAATTAAGCTAAGCCCTTCTAGATTAATGGGACTTAAACCCACAAACATTTAGTTAACAGCTAAACACCCTAATCAAC-TGGCTTCAATCTAAAGCCCCGGCAGG-TTTGAAGCTGCTTCTTCGAATTTGCAATTCAATATGAAAA-TCACCTCAGAGCTTGGTAAAAAGAGGCTTAACCCCTGTCTTTAGATTTACAGTCCAATGCTTCA-CTCAGCCATTTTACCACAAAAAAGGAAGGAATCGAACCCCCTAAAGCTGGTTTCAAGCCAACCCCATGACCTCCATGACTTTTTCAAAAGATATTAGAAAAACTATTTCATAACTTTGTCAAAGTTAAATTACAGGTT-AACCCCCGTATATCTTA-CACTGTAAAGCTAACCTAGCATTAACCTTTTAAGTTAAAGATTAAGAGGACCGACACCTCTTTACAGTGA";
char *gorilla = "AGAAATATGTCTGATAAAAGAGTTACTTTGATAGAGTAAATAATAGAGGTTTAAACCCCCTTATTTCTACTAGGACTATGAGAATTGAACCCATCCCTGAGAATCCAAAATTCTCCGTGCCACCTGTCACACCCCATCCTAAGTAAGGTCAGCTAAATAAGCTATCGGGCCCATACCCCGAAAATGTTGGTCACATCCTTCCCGTACTAAGAAATTTAGGTTAAACATAGACCAAGAGCCTTCAAAGCCCTTAGTAAGTTA-CAACACTTAATTTCTGTAAGGACTGCAAAACCCTACTCTGCATCAACTGAACGCAAATCAGCCACTTTAATTAAGCTAAGCCCTTCTAGATCAATGGGACTCAAACCCACAAACATTTAGTTAACAGCTAAACACCCTAGTCAAC-TGGCTTCAATCTAAAGCCCCGGCAGG-TTTGAAGCTGCTTCTTCGAATTTGCAATTCAATATGAAAT-TCACCTCGGAGCTTGGTAAAAAGAGGCCCAGCCTCTGTCTTTAGATTTACAGTCCAATGCCTTA-CTCAGCCATTTTACCACAAAAAAGGAAGGAATCGAACCCCCCAAAGCTGGTTTCAAGCCAACCCCATGACCTTCATGACTTTTTCAAAAGATATTAGAAAAACTATTTCATAACTTTGTCAAGGTTAAATTACGGGTT-AAACCCCGTATATCTTA-CACTGTAAAGCTAACCTAGCGTTAACCTTTTAAGTTAAAGATTAAGAGTATCGGCACCTCTTTGCAGTGA";

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
    // print resource list
    ResourceList* rList;
    rList = getResourceList();
    fprintf(stdout, "Available resources:\n");
    for (int i = 0; i < rList->length; i++) {
        fprintf(stdout, "\tResource %i:\n\t\tName : %s\n", i, rList->list[i].name);
        fprintf(stdout, "\t\tFlags:");
        if (rList->list[i].flags & DOUBLE)
            fprintf(stdout, " DOUBLE");
        if (rList->list[i].flags & SINGLE)
            fprintf(stdout, " SINGLE");
        if (rList->list[i].flags & ASYNCH)
            fprintf(stdout, " ASYNCH");
        if (rList->list[i].flags & SYNCH)
            fprintf(stdout, " SYNCH");
        if (rList->list[i].flags & CPU)
            fprintf(stdout, " CPU");
        if (rList->list[i].flags & GPU)
            fprintf(stdout, " GPU");
        if (rList->list[i].flags & FPGA)
            fprintf(stdout, " FPGA");
        if (rList->list[i].flags & SSE)
            fprintf(stdout, " SSE");
        if (rList->list[i].flags & CELL)
            fprintf(stdout, " CELL");
        fprintf(stdout, "\n");
        
    }    
    
    
    // is nucleotides...
    int stateCount = 4;
	
    // get the number of site patterns
	int nPatterns = strlen(human);
    
    // create an instance of the BEAGLE library
	int instance = createInstance(
                                  3,				/**< Number of tip data elements (input) */
                                  5,	            /**< Number of partials buffers to create (input) */
                                  0,		        /**< Number of compact state representation buffers to create (input) */
                                  stateCount,		/**< Number of states in the continuous-time Markov chain (input) */
                                  nPatterns,		/**< Number of site patterns to be handled by the instance (input) */
                                  1,		        /**< Number of rate matrix eigen-decomposition buffers to allocate (input) */
                                  4,		        /**< Number of rate matrix buffers (input) */
                                  1,              /**< Number of rate categories (input) */
                                  0,              /**< Number of scaling buffers */
                                  NULL,			/**< List of potential resource on which this instance is allowed (input, NULL implies no restriction */
                                  0,			    /**< Length of resourceList list (input) */
                                  GPU,		        /**< Bit-flags indicating preferred implementation charactertistics, see BeagleFlags (input) */
                                  0		        /**< Bit-flags indicating required implementation characteristics, see BeagleFlags (input) */
                                  );
    if (instance < 0) {
	    fprintf(stderr, "Failed to obtain BEAGLE instance\n\n");
	    exit(1);
    }
    
    // initialize the instance
    int error = initializeInstance(instance, NULL);
	
    if (error < 0) {
	    fprintf(stderr, "Failed to initialize BEAGLE instance\n\n");
	    exit(1);
    }
    
    //	setTipStates(instance, 0, getStates(human));
    //	setTipStates(instance, 1, getStates(chimp));
    //	setTipStates(instance, 2, getStates(gorilla));
    
    // set the sequences for each tip using partial likelihood arrays
	setTipPartials(instance, 0, getPartials(human));
	setTipPartials(instance, 1, getPartials(chimp));
	setTipPartials(instance, 2, getPartials(gorilla));
    
	double rates[1] = { 1.0 };
	setCategoryRates(instance, rates);
	
    // create base frequency array
	double freqs[4] = { 0.25, 0.25, 0.25, 0.25 };
    
    // create an array containing site category weights
	const double weights[1] = { 1.0 };
    
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
    
	double eval[4] = { 0.0, -1.3333333333333333, -1.3333333333333333, -1.3333333333333333 };
    
    // set the Eigen decomposition
	setEigenDecomposition(instance, 0, evec, ivec, eval);
    
    // a list of indices and edge lengths
	int nodeIndices[4] = { 0, 1, 2, 3 };
	double edgeLengths[4] = { 0.1, 0.1, 0.2, 0.1 };
    
    // tell BEAGLE to populate the transition matrices for the above edge lengths
	updateTransitionMatrices(instance,     // instance
	                         0,             // eigenIndex
	                         nodeIndices,   // probabilityIndices
	                         NULL,          // firstDerivativeIndices
	                         NULL,          // secondDervativeIndices
	                         edgeLengths,   // edgeLengths
	                         4);            // count
    
    // create a list of partial likelihood update operations
    // the order is [dest, destScaling, source1, matrix1, source2, matrix2]
	int operations[7 * 2] = {
		3, NONE, NONE, 0, 0, 1, 1,
		4, NONE, NONE, 2, 2, 3, 3
	};
	int rootIndex = 4;
    
    // update the partials
	updatePartials( &instance,      // instance
                   1,              // instanceCount
                   operations,     // eigenIndex
                   2,              // operationCount
                   0);             // rescale ? 0 = no
    
	double *patternLogLik = (double*)malloc(sizeof(double) * nPatterns);

    int cumulativeScalingFactorIndex = NONE;
    
    // calculate the site likelihoods at the root node
	calculateRootLogLikelihoods(instance,               // instance
	                            (const int *)&rootIndex,// bufferIndices
	                            weights,                // weights
	                            freqs,                 // stateFrequencies
                                &cumulativeScalingFactorIndex,
	                            1,                      // count
	                            patternLogLik);         // outLogLikelihoods
    
	double logL = 0.0;
	for (int i = 0; i < nPatterns; i++) {
//		std::cerr << "site lnL[" << i << "] = " << patternLogLik[i] << '\n';
		logL += patternLogLik[i];
	}
    
	fprintf(stdout, "logL = %.5f (PAUP logL = -1574.63623)\n\n", logL);
    
    finalize(instance);
    
}
