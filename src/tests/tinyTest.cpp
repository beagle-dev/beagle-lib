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

	int instance = initialize(5, 3, 4, nPatterns, 1, 1);

//	setTipStates(0, getStates(human));
//	setTipStates(1, getStates(chimp));
//	setTipStates(2, getStates(gorilla));

	setTipPartials(instance, 0, getPartials(human));
	setTipPartials(instance, 1, getPartials(chimp));
	setTipPartials(instance, 2, getPartials(gorilla));

	double freqs[4] = { 0.25, 0.25, 0.25, 0.25 };
	setStateFrequencies(instance, freqs);

	double rates[1] = { 1.0 };
	setCategoryRates(instance, rates);

	double props[1] = { 1.0 };
	setCategoryProportions(instance, props);

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

	setEigenDecomposition(instance, 0, evecP, ivecP, eval);

	int nodeIndices[4] = { 0, 1, 2, 3 };
	double branchLengths[4] = { 0.1, 0.1, 0.2, 0.1 };

	calculateProbabilityTransitionMatrices(instance, nodeIndices, branchLengths, 4);

	int operations[5 * 3] = {
		0, 1, 3,
		2, 3, 4
	};

	calculatePartials(instance, operations, NULL, 2, 0);

	double *patternLogLik = (double*)malloc(sizeof(double) * nPatterns);

	calculateLogLikelihoods(instance, 4, patternLogLik);

	double logL = 0.0;
	for (int i = 0; i < nPatterns; i++) {
		logL += patternLogLik[i];
	}

	fprintf(stdout, "logL = %.5f (PAUP logL = -1574.63623)\n\n", logL);



}
