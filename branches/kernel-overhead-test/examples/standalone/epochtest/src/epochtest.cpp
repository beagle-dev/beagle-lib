#include <libhmsbeagle-1/libhmsbeagle/beagle.h>
#include <iostream>
#include <vector>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

char
		*SimSeq1 =
				(char*) "TCAAGTGAGGTACACACGATTCATAAGATACCGAGAGGGACGTGGGATCATGTCCGTTGGAGGCATGCTCAACTGGCGTAAGTGGTAAACTGGATGTCTCTAGTAGTGGGCAAGCGTCGTGAGGCGCAAAAGTACTTGTGGTAAGGACCAGAGATGCACAGGCTATCTTCTTTAATCTAATGCGATACTCGTTGAAGTCTATCTTTGAGATAGCTAAGATCACATAGGGACTTTGATAGCCAACAGTGTGCATGGCACTGTAATAATTTTTTGGCAGACCACACTTGGGTGGGGAGAGTAGAGAAAGTGAATCGATGAGCAGGTTCCGCGGCTGGGGGGGTTAGGGATATCACAAGATTGTCTCTTAAACACGGGGTTTAGGGTTACTTACAAGAAGATGAATGCCGGAGCAGGGGGTCATCTTAAGGAATCGGGGAATTTGTCCTCGGGCAAGAATTGACACCTGCCTCATTATTATTCATGAGACTACAAAAAGGGGATAGTTGAATTACATTAGCGCGCGGTCGAACTCCGACTGAAAATAATAAGCGTAAATGGAACCCAGCCACAGTGACTGGTAAGCAAATGTTTAGTTTAAACAACCAGGGTAATGGTGCATTTCAGAAACTGGGGACAAGTCGATTACGGCGGTTGATTACCGGGTCCTGTTAGGTGTAGGGAGCCCATCTACTGGCACATTTTTTTGTGCGTATCGACGGGGACCCAGTGAAAAGGAGTAACTGATAAAGTACAGTGAATGAATAATGTAAAAGCGGGACGAGTAAAAGCATACATAGTGCGTGAAAGGTGTCAGTTTACCTACCTGGGACAATCTCTGTACGTGAGTAAGTTTGGATCCAGCGGAAAACCACAGGTGAGAGGGCTTCTGAAACGCATGCCCGATTAGATAACTGATAATGTAAACCAGGACGCTTCAGCTGTGAATGCCTGTATAAAATCCCTGGGTTTTCAGGAATGGAAATAACGCTGACCAGCTATT";

char
		*SimSeq2 =
				(char*) "ATAAAAAAGGGGTGCAAAAGAACACGTGCAATTAAAATAAGGCCGGTCGATGTATAAGGTTACATCGCCAGTGAGTTATCTGGATCAATCTTGTTTCTTCTTAGAAAACGAAACATCTAAAGAGTTCATTGAGGAGTATGGCGGAGTTTATGGGATGGTTCTCGATCTTCTGCAGACACGAGAAGCAGCAGCAAAACTATATGATCAAGCGAATGAAGATAAAATTGGGGATTGATCAATGGGGGCTGCATACGGACTTGCACCATAGTAAGCAGCTAACGTGTCATAATAATGGTGATTAGAAGAGTAGGTTAGTGAGAATTAATAATCGGCGGAAATTGGGATTATACCTTATGCTGGCGCCGATAAGAGGTAATGAGTGACACAAATAAAGGGTAAATATAATCGGACGGAGCGTAATTCGGAGAGACTAATGGACGGGATGTAAAGTTCACACTAGGACTGTAACTGTTGGAACGTAAGAAAAAATCGAGAAGGAGCAAATGGGGTATAGGATGTACCGCAAGGTCTTACATTAAAAAGACGAAGTGGATGGCGGGTCTGAGTGAACAGGTTGTTCATTTGGGACGATGACCAGGTTATGGAACCAATAAGAAGTTATAGTAGAACAGGATAAATCCACAGAATTTCATCATGAGAGAGTTAAAATGACTATGGCGGCCCTATCTGTGGCAGAGAAACCATTAACTCGCCAGTGAATTCTCCTTGCGAGGCTGGGAGCATCAGCGTGAGACGAACAAAGATTGCATCGGTTAGATAAAACCAGTAATGCGTATCAGACCAAGTGTGTCAAAGAGTTTATTCAGAGGGCGGGACACATGGATCTAAGATCAGCTCGACCGGCGGATTCCAGTTAGTGGAGATATGCGGGGCTCTTATTGAATGGGTCCATCAGTGGTTATGGAGAAATTCGATTAACAGTAATCTTCAGGCGAGATGTGTGATCCTTTCAAACGGACAGAAGCGCGAGGTAAAATCC";

char
		*SimSeq3 =
				(char*) "ACAGGGTGCGAACCCGAGAGAATTGATAAAATCCAAATGGGTATGGTGGGTATGAACAAATGAATCATAAATAAATCGTCTGCATCGAGTTTATCCCTCGTTTGATGGCGCGGAGTTCAACAACTTATTGATCAGAAAAGGGAGAATTTGTGAGACCGCAGTTCTCTTCCTACCAGAGCAGAATGCATTGACCAAGCTCCATACGCACACAAACCCCTTTGTGATCGGGAATTGATCACAAAACGTCATACAGGAAGTAGCCTCGTCATCGGCAATCAATAGAAAAAGACCACAAGGAACGGAAGGGAAGTTCAAGGAAGAATAAGAGTAGTAAGGAACGATGTGTATATTGGTAAATTACGGCAACCAGAAGCGATGAGCCGTGCGTTCATAAGACAGACGCTATGGAACGGGGCATGACCTGGAGAGACTAATGATTAGTATATAGAATCCGACCAAGCGTGGTGGCCATTCTAATATGAGGAGACGTTGAGAAAAATGAGCCAGGAAGGGTAGCATGCCAGAAAACTCTATGCAGGAAGTAATTTAGACATATCCCTTCTGGTGCAGTAATTTGCTTATCCAAGACATTGATTCGACTCTGGGGGTCACATGGTGGTCTCATGGAATGACGAGGACTCACCGGACTATACTTGAACATAGGTGAGACAATCATGACGAATTCGCCCAAGGCGGTGCTGTTATTAGTATTGGACCGGTATCGCTGCACAAATTCAAGTAGATCGTTTGGGGGAGGATTAGGGTCACCCGGAATGAGCAATTGCGTGAATAAGAAACCCACTGAACTCGGAAGATTGATAGTTTAGTGGGTCGGATCTGTACGCTCTGGCAGAGCTCAGTTAGGAGACTTCAAAAGGAAGAATCCTACGCTTCGCCTGCGAGGTTAGGCAGATATGGGGAATGAGATGTTTCAGGTGATAATAGCTTGATGACACAAAATTAAGTCATCCCAAATCAGCAGGCAAACAGAGCAGAGCAC";

void printMatrix(int matrixIndex, int stateCount, int nRateCats, int instance) {

	double* outMatrix = (double*) malloc(sizeof(double) * stateCount
			* stateCount * nRateCats);

	beagleGetTransitionMatrix(instance, // instance,
			matrixIndex, // matrixIndex
			outMatrix // outMatrix
	);

	for (int row = 0; row < stateCount; row++) {
		printf("| ");
		for (int col = 0; col < stateCount; col++)
			printf("%f ", outMatrix[col + row * stateCount]);
		printf("|\n");
	}
	printf("\n");

	memset(outMatrix, 0, sizeof(double) * stateCount * stateCount * nRateCats);
	free(outMatrix);
}//END: printMatrix

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
}//END: getStates

double* getPartials(char *sequence) {
	int n = strlen(sequence);
	double *partials = (double*) malloc(sizeof(double) * n * 4);

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
}//END: getPartials

int main(int argc, const char* argv[]) {

	// is nucleotides
	int stateCount = 4;

	// get the number of site patterns
	int nPatterns = strlen(SimSeq1);

	int rateCategoryCount = 4;

	int nRateCats = rateCategoryCount;
	int nRootCount = 1;
	int nPartBuffs = 4 + nRootCount;
	int scaleCount = 0;
	int nConvMatBuff = 2; // number of extra transition probability matrices buffers to create
	int nRateMatBuff = 6 + nConvMatBuff;
	int nRateMatEigDecBuf = 2;

	// initialize the instance
	BeagleInstanceDetails instDetails;

	// create an instance of the BEAGLE library
	int instance = beagleCreateInstance(3, // Number of tip data elements
			nPartBuffs, // Number of partials buffers to create
			0, // Number of compact state representation buffers to create
			stateCount, // Number of states in the continuous-time Markov chain
			nPatterns, // Number of site patterns to be handled by the instance
			nRateMatEigDecBuf, // Number of rate matrix eigen-decomposition buffers to allocate
			nRateMatBuff, // Number of rate matrix buffers
			nRateCats, // Number of rate categories
			scaleCount, // Number of scaling buffers
			NULL, // List of potential resource on which this instance is allowed (NULL implies no restriction
			0, // Length of resourceList list
			BEAGLE_FLAG_PROCESSOR_CPU, // Bit-flags indicating preferred implementation charactertistics, see BeagleFlags
			BEAGLE_FLAG_PRECISION_DOUBLE, // Bit-flags indicating required implementation characteristics, see BeagleFlags
			&instDetails);

	if (instance < 0) {

		fprintf(stderr, "Failed to obtain BEAGLE instance \n \n");
		exit(BEAGLE_ERROR_UNINITIALIZED_INSTANCE);

	} else {

		fprintf(stdout, "Using resource %i: \n", instDetails.resourceNumber);
		fprintf(stdout, "\t Rsrc Name : %s \n", instDetails.resourceName);
		fprintf(stdout, "\t Impl Name : %s \n", instDetails.implName);
		fprintf(stdout, "\n");

	}

	// set the sequences for each tip using partial likelihood arrays
	double *SimSeq1Partials = getPartials(SimSeq1);
	double *SimSeq2Partials = getPartials(SimSeq2);
	double *SimSeq3Partials = getPartials(SimSeq3);

	beagleSetTipPartials(instance, 0, SimSeq3Partials);
	beagleSetTipPartials(instance, 1, SimSeq2Partials);
	beagleSetTipPartials(instance, 2, SimSeq1Partials);

	// alpha = 0.5
	double rates[4] = { 0.02907775442778477, 0.2807145339257214,
			0.9247730548197041, 2.76543465682679 };

	// create base frequency array
	double freqs[4] = { 0.25, 0.25, 0.25, 0.25 };

	// create an array containing site category weights
	double* weights = (double*) malloc(sizeof(double) * rateCategoryCount);

	for (int i = 0; i < rateCategoryCount; i++) {
		weights[i] = 1.0 / rateCategoryCount;
	}

	double* patternWeights = (double*) malloc(sizeof(double) * nPatterns);

	for (int i = 0; i < nPatterns; i++) {
		patternWeights[i] = 1.0;
	}

	// an eigen decomposition for the HKY kappa=1 subst model
	double evecHKY1[4 * 4] = { 1.0, 2.0, 0.0, 0.5, 1.0, -2.0, 0.5, 0.0, 1.0,
			2.0, 0.0, -0.5, 1.0, -2.0, -0.5, 0.0 };

	double ivecHKY1[4 * 4] = { 0.25, 0.25, 0.25, 0.25, 0.125, -0.125, 0.125,
			-0.125, 0.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 0.0 };

	double evalHKY1[4] = { 0.0, -1.3333333333333333, -1.3333333333333333,
			-1.3333333333333333 };

	// set the Eigen decomposition for buffer 0
	beagleSetEigenDecomposition(instance, 0, evecHKY1, ivecHKY1, evalHKY1);

	// an eigen decomposition for the HKY kappa=20 subst model
	double evecHKY20[4 * 4] = { 1.0, 2.0, 0.0, 0.5, 1.0, -2.0, 0.5, 0.0, 1.0,
			2.0, 0.0, -0.5, 1.0, -2.0, -0.5, 0.0 };

	double ivecHKY20[4 * 4] = { 0.25, 0.25, 0.25, 0.25, 0.125, -0.125, 0.125,
			-0.125, 0.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 0.0 };

	double evalHKY20[4] = { 0.0, -0.18181818181818182, -1.9090909084,
			-1.9090909097818183 };

	// set the Eigen decomposition for buffer 1
	beagleSetEigenDecomposition(instance, 1, evecHKY20, ivecHKY20, evalHKY20);

	beagleSetStateFrequencies(instance, 0, freqs);

	beagleSetCategoryWeights(instance, 0, weights);

	beagleSetPatternWeights(instance, patternWeights);

	double epochTransitionTimes[1] = { 20 };

	// a list of matrix indices and edge lengths for eigen buffer 0
	int probabilityIndicesEigenBuffer0[2] = { 4, 6 };
	double edgeLengthsEigenBuffer0[2] = { epochTransitionTimes[0] - 10.0083,
			epochTransitionTimes[0] };
	int listLengthEigenBuffer0 = 2;

	// a list of matrix indices and edge lengths for eigen buffer 1
	int probabilityIndicesEigenBuffer1[4] = { 1, 3, 5, 7 };
	double edgeLengthsEigenBuffer1[4] = { 25.2487, 18.48981, 45.2487
			- epochTransitionTimes[0] + 10.0083, 73.7468
			- epochTransitionTimes[0] };
	int listLengthEigenBuffer1 = 4;

	int* rootIndices = (int*) malloc(sizeof(int) * nRootCount);
	int* categoryWeightsIndices = (int*) malloc(sizeof(int) * nRootCount);
	int* stateFrequencyIndices = (int*) malloc(sizeof(int) * nRootCount);
	int* cumulativeScalingIndices = (int*) malloc(sizeof(int) * nRootCount);

	for (int i = 0; i < nRootCount; i++) {

		rootIndices[i] = 4 + i;
		categoryWeightsIndices[i] = 0;
		stateFrequencyIndices[i] = 0;
		cumulativeScalingIndices[i] = BEAGLE_OP_NONE;

		beagleSetCategoryRates(instance, &rates[i]);

		// tell BEAGLE to populate the transition matrices for the above edge lengths
		beagleUpdateTransitionMatrices(instance, // instance
				0, // eigenIndex
				probabilityIndicesEigenBuffer0, // probabilityIndices
				NULL, // firstDerivativeIndices
				NULL, // secondDerivativeIndices
				edgeLengthsEigenBuffer0, // edgeLengths
				listLengthEigenBuffer0); // count

		beagleUpdateTransitionMatrices(instance, // instance
				1, // eigenIndex
				probabilityIndicesEigenBuffer1, // probabilityIndices
				NULL, // firstDerivativeIndices
				NULL, // secondDerivativeIndices
				edgeLengthsEigenBuffer1, // edgeLengths
				listLengthEigenBuffer1); // count

		int firstIndices[2] = { 4, 6 };
		int secondIndices[2] = { 5, 7 };
		int resultIndices[2] = { 0, 2 };

		beagleConvolveTransitionMatrices(instance, // instance
				firstIndices, // first indices
				secondIndices, // second indices
				resultIndices, // result indices
				2 // matrixCount
		);

		// for (int j = 0; j <= (6 + 1); j++) {
		// printf("Matrix index %i: \n", j);
		// printMatrix(j, stateCount, nRateCats, instance);
		// }

		// create a list of partial likelihood update operations
		// the order is [dest, destScaling, source1, matrix1, source2, matrix2]
		BeagleOperation operations[2] = {

		{ 3, // destination or parent partials buffer
				BEAGLE_OP_NONE, // scaling buffer to write to
				BEAGLE_OP_NONE, // scaling buffer to read from
				0, // first child partials buffer
				0, // transition matrix of first partials child buffer
				1, // second child partials buffer
				1 // transition matrix of second partials child buffer
				},

				{ rootIndices[i], // destination or parent partials buffer
						BEAGLE_OP_NONE, // scaling buffer to write to
						BEAGLE_OP_NONE, // scaling buffer to read from
						2, // first child partials buffer
						2, // transition matrix of first partials child buffer
						3, // second child partials buffer
						3 // transition matrix of second partials child buffer
				}

		};

		// update the partials
		beagleUpdatePartials(instance, // instance
				operations, // eigenIndex
				2, // operationCount
				cumulativeScalingIndices[i]); // cumulative scaling index

	}//END: nRootCount loop

	double *patternLogLik = (double*) malloc(sizeof(double) * nPatterns);
	double logL = 0.0;
	int returnCode = 0;

	// calculate the site likelihoods at the root node
	returnCode = beagleCalculateRootLogLikelihoods(instance, // instance
			(const int *) rootIndices, // bufferIndices
			(const int *) categoryWeightsIndices, // weights
			(const int *) stateFrequencyIndices, // stateFrequencies
			cumulativeScalingIndices, // cumulative scaling index
			nRootCount, // count
			&logL); // outLogLikelihoods

	beagleGetSiteLogLikelihoods(instance, patternLogLik);

	double sumLogL = 0.0;
	for (int i = 0; i < nPatterns; i++) {
		sumLogL += patternLogLik[i] * patternWeights[i];
	}

	fprintf(stdout, "logL = %.5f \n", logL);
	fprintf(stdout, "sumLogL = %.5f \n", sumLogL);

	free(weights);
	free(patternWeights);
	free(rootIndices);
	free(categoryWeightsIndices);
	free(stateFrequencyIndices);
	free(cumulativeScalingIndices);

	free(patternLogLik);

	free(SimSeq1Partials);
	free(SimSeq2Partials);
	free(SimSeq3Partials);

	beagleFinalizeInstance(instance);

	return (EXIT_SUCCESS);
}//END: main

