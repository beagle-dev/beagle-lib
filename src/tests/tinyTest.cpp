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

#include "beagle.h"

char *human = "AGAAATATGTCTGATAAAAGAGTTACTTTGATAGAGTAAATAATAGGAGCTTAAACCCCCTTATTTCTACTAGGACTATGAGAATCGAACCCATCCCTGAGAATCCAAAATTCTCCGTGCCACCTATCACACCCCATCCTAAGTAAGGTCAGCTAAATAAGCTATCGGGCCCATACCCCGAAAATGTTGGTTATACCCTTCCCGTACTAAGAAATTTAGGTTAAATACAGACCAAGAGCCTTCAAAGCCCTCAGTAAGTTG-CAATACTTAATTTCTGTAAGGACTGCAAAACCCCACTCTGCATCAACTGAACGCAAATCAGCCACTTTAATTAAGCTAAGCCCTTCTAGACCAATGGGACTTAAACCCACAAACACTTAGTTAACAGCTAAGCACCCTAATCAAC-TGGCTTCAATCTAAAGCCCCGGCAGG-TTTGAAGCTGCTTCTTCGAATTTGCAATTCAATATGAAAA-TCACCTCGGAGCTTGGTAAAAAGAGGCCTAACCCCTGTCTTTAGATTTACAGTCCAATGCTTCA-CTCAGCCATTTTACCACAAAAAAGGAAGGAATCGAACCCCCCAAAGCTGGTTTCAAGCCAACCCCATGGCCTCCATGACTTTTTCAAAAGGTATTAGAAAAACCATTTCATAACTTTGTCAAAGTTAAATTATAGGCT-AAATCCTATATATCTTA-CACTGTAAAGCTAACTTAGCATTAACCTTTTAAGTTAAAGATTAAGAGAACCAACACCTCTTTACAGTGA";
char *chimp = "AGAAATATGTCTGATAAAAGAATTACTTTGATAGAGTAAATAATAGGAGTTCAAATCCCCTTATTTCTACTAGGACTATAAGAATCGAACTCATCCCTGAGAATCCAAAATTCTCCGTGCCACCTATCACACCCCATCCTAAGTAAGGTCAGCTAAATAAGCTATCGGGCCCATACCCCGAAAATGTTGGTTACACCCTTCCCGTACTAAGAAATTTAGGTTAAGCACAGACCAAGAGCCTTCAAAGCCCTCAGCAAGTTA-CAATACTTAATTTCTGTAAGGACTGCAAAACCCCACTCTGCATCAACTGAACGCAAATCAGCCACTTTAATTAAGCTAAGCCCTTCTAGATTAATGGGACTTAAACCCACAAACATTTAGTTAACAGCTAAACACCCTAATCAAC-TGGCTTCAATCTAAAGCCCCGGCAGG-TTTGAAGCTGCTTCTTCGAATTTGCAATTCAATATGAAAA-TCACCTCAGAGCTTGGTAAAAAGAGGCTTAACCCCTGTCTTTAGATTTACAGTCCAATGCTTCA-CTCAGCCATTTTACCACAAAAAAGGAAGGAATCGAACCCCCTAAAGCTGGTTTCAAGCCAACCCCATGACCTCCATGACTTTTTCAAAAGATATTAGAAAAACTATTTCATAACTTTGTCAAAGTTAAATTACAGGTT-AACCCCCGTATATCTTA-CACTGTAAAGCTAACCTAGCATTAACCTTTTAAGTTAAAGATTAAGAGGACCGACACCTCTTTACAGTGA";
char *gorilla = "AGAAATATGTCTGATAAAAGAGTTACTTTGATAGAGTAAATAATAGAGGTTTAAACCCCCTTATTTCTACTAGGACTATGAGAATTGAACCCATCCCTGAGAATCCAAAATTCTCCGTGCCACCTGTCACACCCCATCCTAAGTAAGGTCAGCTAAATAAGCTATCGGGCCCATACCCCGAAAATGTTGGTCACATCCTTCCCGTACTAAGAAATTTAGGTTAAACATAGACCAAGAGCCTTCAAAGCCCTTAGTAAGTTA-CAACACTTAATTTCTGTAAGGACTGCAAAACCCTACTCTGCATCAACTGAACGCAAATCAGCCACTTTAATTAAGCTAAGCCCTTCTAGATCAATGGGACTCAAACCCACAAACATTTAGTTAACAGCTAAACACCCTAGTCAAC-TGGCTTCAATCTAAAGCCCCGGCAGG-TTTGAAGCTGCTTCTTCGAATTTGCAATTCAATATGAAAT-TCACCTCGGAGCTTGGTAAAAAGAGGCCCAGCCTCTGTCTTTAGATTTACAGTCCAATGCCTTA-CTCAGCCATTTTACCACAAAAAAGGAAGGAATCGAACCCCCCAAAGCTGGTTTCAAGCCAACCCCATGACCTTCATGACTTTTTCAAAAGATATTAGAAAAACTATTTCATAACTTTGTCAAGGTTAAATTACGGGTT-AAACCCCGTATATCTTA-CACTGTAAAGCTAACCTAGCGTTAACCTTTTAAGTTAAAGATTAAGAGTATCGGCACCTCTTTGCAGTGA";

int* getStates(char *sequence) {
	int n = strlen(sequence);
	int *states = (int*)malloc(sizeof(int) * n);

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
	int nPatterns = strlen(human);

	int instance = createInstance(
			    3,				/**< Number of tip data elements (input) */
				5,	            /**< Number of partials buffers to create (input) */
				0,		        /**< Number of compact state representation buffers to create (input) */
				4,				/**< Number of states in the continuous-time Markov chain (input) */
				nPatterns,		/**< Number of site patterns to be handled by the instance (input) */
				1,		        /**< Number of rate matrix eigen-decomposition buffers to allocate (input) */
				4,		        /**< Number of rate matrix buffers (input) */
				NULL,			/**< List of potential resource on which this instance is allowed (input, NULL implies no restriction */
				0,			    /**< Length of resourceList list (input) */
				0,		        /**< Bit-flags indicating preferred implementation charactertistics, see BeagleFlags (input) */
				0		        /**< Bit-flags indicating required implementation characteristics, see BeagleFlags (input) */
				);
    if (instance < 0) {
	    fprintf(stderr, "Failed to obtain BEAGLE instance\n\n");
	    exit(1);
    }
    int error = initializeInstance(instance, NULL);

//	setTipStates(0, getStates(human));
//	setTipStates(1, getStates(chimp));
//	setTipStates(2, getStates(gorilla));

	setPartials(instance, 0, getPartials(human));
	setPartials(instance, 1, getPartials(chimp));
	setPartials(instance, 2, getPartials(gorilla));

    int stateCount = 4;

	double freqs[4] = { 0.25, 0.25, 0.25, 0.25 };

	double rates[1] = { 1.0 };

	double props[1] = { 1.0 };

	// an eigen decomposition for the JC69 model
	double evec[4][4] = {
		{ 1.0,  2.0,  0.0,  0.5},
		{ 1.0,  -2.0,  0.5,  0.0},
		{ 1.0,  2.0, 0.0,  -0.5},
		{ 1.0,  -2.0,  -0.5,  0.0}
	};
	double ivec[4][4] = {
		{ 0.25,  0.25,  0.25,  0.25},
		{ 0.125,  -0.125,  0.125,  -0.125},
		{ 0.0,  1.0,  0.0,  -1.0},
		{ 1.0,  0.0,  -1.0,  0.0}
	};
	double eval[4] = { 0.0, -1.3333333333333333, -1.3333333333333333, -1.3333333333333333 };

	double **evecP = (double **)malloc(sizeof(double*) * stateCount);
	double **ivecP = (double **)malloc(sizeof(double*) * stateCount);

	for (int i = 0; i < stateCount; i++) {
		evecP[i] = (double *)malloc(sizeof(double) * stateCount);
		ivecP[i] = (double *)malloc(sizeof(double) * stateCount);
		for (int j = 0; j < stateCount; j++) {
			evecP[i][j] = evec[i][j];
			ivecP[i][j] = ivec[i][j];
		}
	}

	setEigenDecomposition(instance, 0, (const double **)evecP, (const double **)ivecP, (const double *)eval);

	int nodeIndices[4] = { 0, 1, 2, 3 };
	double branchLengths[4] = { 0.1, 0.1, 0.2, 0.1 };

	updateTransitionMatrices(instance,     // instance
	                         0,             // eigenIndex
	                         nodeIndices,   // probabilityIndices
	                         NULL,          // firstDerivativeIndices
	                         NULL,          // secondDervativeIndices
	                         branchLengths, // edgeLengths
	                         4);            // count

	int operations[5 * 3] = {
		3, 0, 0, 1, 1,
		4, 2, 2, 3, 3
	};
	int rootIndex = 4;

	updatePartials( &instance,      // instance
	                1,              // instanceCount
	                operations,     // eigenIndex
	                4,              // operationCount
	                0);

	double *patternLogLik = (double*)malloc(sizeof(double) * nPatterns);

	calculateRootLogLikelihoods(instance,           // instance
	                            (const int *)&rootIndex,         // bufferIndices
	                            (const double *)props,              // weights
	                            (const double **)&freqs,             // stateFrequencies
	                            1,                  // count
	                            patternLogLik);     // outLogLikelihoods

	double logL = 0.0;
	for (int i = 0; i < nPatterns; i++) {
		logL += patternLogLik[i];
	}

	fprintf(stdout, "logL = %.5f (PAUP logL = -1574.63623)\n\n", logL);



}
