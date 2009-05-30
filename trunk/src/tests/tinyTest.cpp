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
			    5,          // bufferCount
				3,          // tipCount
				4,          // stateCount
				nPatterns,  // patternCount
				1,          // eigenDecompositionCount
				4,          // matrixCount (nodes-1)
				null,       // resourceList
				0,          // resourceCount
				0,          // preferenceFlags
				0          // requirementFlags
				);

    int error = initializeInstance(&1, 1, NULL);

//	setTipStates(0, getStates(human));
//	setTipStates(1, getStates(chimp));
//	setTipStates(2, getStates(gorilla));

	setPartials(&instance, 1, 0, getPartials(human));
	setPartials(&instance, 1, 1, getPartials(chimp));
	setPartials(&instance, 1, 2, getPartials(gorilla));

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

	double **evecP = (double **)malloc(sizeof(double*) * STATE_COUNT);
	double **ivecP = (double **)malloc(sizeof(double*) * STATE_COUNT);

	for (int i = 0; i < STATE_COUNT; i++) {
		evecP[i] = (double *)malloc(sizeof(double) * STATE_COUNT);
		ivecP[i] = (double *)malloc(sizeof(double) * STATE_COUNT);
		for (int j = 0; j < STATE_COUNT; j++) {
			evecP[i][j] = evec[i][j];
			ivecP[i][j] = ivec[i][j];
		}
	}

	setEigenDecomposition(&instance, 1, 0, evecP, ivecP, eval);

	int nodeIndices[4] = { 0, 1, 2, 3 };
	double branchLengths[4] = { 0.1, 0.1, 0.2, 0.1 };

	updateTransitionMatrices(&instance,     // instance
	                         1,             // instanceCount
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

	updatePartials( &instance,     // instance
	                1,             // instanceCount
	                operations,    // eigenIndex
	                nodeIndices,   // probabilityIndices
	                NULL,          // firstDerivativeIndices
	                NULL,          // secondDervativeIndices
	                branchLengths, // edgeLengths
	                4);

	double *patternLogLik = (double*)malloc(sizeof(double) * nPatterns);

	calculateRootLogLikelihoods(&instance,          // instance
	                            1,                  // instanceCount
	                            &rootIndex,         // bufferIndices
	                            1,                  // count
	                            props,              // weights
	                            &freqs,             // stateFrequencies
	                            patternLogLik);     // outLogLikelihoods

	double logL = 0.0;
	for (int i = 0; i < nPatterns; i++) {
		logL += patternLogLik[i];
	}

	fprintf(stdout, "logL = %.5f (PAUP logL = -1574.63623)\n\n", logL);



}
