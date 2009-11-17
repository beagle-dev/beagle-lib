/*
 *  tinyTest.cpp
 *  BEAGLE
 *
 *  Created by Andrew Rambaut on 20/03/2009.
 *
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <iostream>

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
    // print resource list
    BeagleResourceList* rList;
    rList = beagleGetResourceList();
    fprintf(stdout, "Available resources:\n");
    for (int i = 0; i < rList->length; i++) {
        fprintf(stdout, "\tResource %i:\n\t\tName : %s\n", i, rList->list[i].name);
        fprintf(stdout, "\t\tDesc : %s\n", rList->list[i].description);
        fprintf(stdout, "\t\tFlags:");
        if (rList->list[i].supportFlags & BEAGLE_FLAG_DOUBLE) fprintf(stdout, " DOUBLE");
        if (rList->list[i].supportFlags & BEAGLE_FLAG_SINGLE) fprintf(stdout, " SINGLE");
        if (rList->list[i].supportFlags & BEAGLE_FLAG_ASYNCH) fprintf(stdout, " ASYNCH");
        if (rList->list[i].supportFlags & BEAGLE_FLAG_SYNCH)  fprintf(stdout, " SYNCH");
        if (rList->list[i].supportFlags & BEAGLE_FLAG_COMPLEX)fprintf(stdout, " COMPLEX");
        if (rList->list[i].supportFlags & BEAGLE_FLAG_LSCALER)fprintf(stdout, " LSCALER");
        if (rList->list[i].supportFlags & BEAGLE_FLAG_CPU)    fprintf(stdout, " CPU");
        if (rList->list[i].supportFlags & BEAGLE_FLAG_GPU)    fprintf(stdout, " GPU");
        if (rList->list[i].supportFlags & BEAGLE_FLAG_FPGA)   fprintf(stdout, " FPGA");
        if (rList->list[i].supportFlags & BEAGLE_FLAG_SSE)    fprintf(stdout, " SSE");
        if (rList->list[i].supportFlags & BEAGLE_FLAG_CELL)   fprintf(stdout, " CELL");
        fprintf(stdout, "\n");
    }    
    fprintf(stdout, "\n");    
    
    bool scaling = true;
	bool gRates = false; // generalized rate categories, separate root buffers
    
    // is nucleotides...
    int stateCount = 4;
	
    // get the number of site patterns
	int nPatterns = strlen(human);
    
    int rateCategoryCount = 4;
	
	int nRateCats = (gRates ? 1 : rateCategoryCount);
	int nRootCount = (!gRates ? 1 : rateCategoryCount);
	int nPartBuffs = 4 + nRootCount;
    int scaleCount = (scaling ? 2 + nRootCount : 0);
    
    // create an instance of the BEAGLE library
	int instance = beagleCreateInstance(
                                  3,				/**< Number of tip data elements (input) */
                                  nPartBuffs,       /**< Number of partials buffers to create (input) */
                                  0,		        /**< Number of compact state representation buffers to create (input) */
                                  stateCount,		/**< Number of states in the continuous-time Markov chain (input) */
                                  nPatterns,		/**< Number of site patterns to be handled by the instance (input) */
                                  1,		        /**< Number of rate matrix eigen-decomposition buffers to allocate (input) */
                                  4,		        /**< Number of rate matrix buffers (input) */
								  nRateCats,		/**< Number of rate categories (input) */
                                  scaleCount,       /**< Number of scaling buffers */
                                  NULL,			    /**< List of potential resource on which this instance is allowed (input, NULL implies no restriction */
                                  0,			    /**< Length of resourceList list (input) */
                                  BEAGLE_FLAG_GPU,	/**< Bit-flags indicating preferred implementation charactertistics, see BeagleFlags (input) */
                                  0                 /**< Bit-flags indicating required implementation characteristics, see BeagleFlags (input) */
                                  );
    if (instance < 0) {
	    fprintf(stderr, "Failed to obtain BEAGLE instance\n\n");
	    exit(1);
    }
    
    // initialize the instance
    BeagleInstanceDetails instDetails;
    int error = beagleInitializeInstance(instance, &instDetails);
	
    if (error < 0) {
	    fprintf(stderr, "Failed to initialize BEAGLE instance\n\n");
	    exit(1);
    }
    
    int rNumber = instDetails.resourceNumber;
    fprintf(stdout, "Using resource %i:\n", rNumber);
    fprintf(stdout, "\tRsrc Name : %s\n",instDetails.resourceName);
    fprintf(stdout, "\tImpl Name : %s\n", instDetails.implName);
    fprintf(stdout, "\tFlags:");
    if (instDetails.flags & BEAGLE_FLAG_DOUBLE) fprintf(stdout, " DOUBLE");
    if (instDetails.flags & BEAGLE_FLAG_SINGLE) fprintf(stdout, " SINGLE");
    if (instDetails.flags & BEAGLE_FLAG_ASYNCH) fprintf(stdout, " ASYNCH");
    if (instDetails.flags & BEAGLE_FLAG_SYNCH)  fprintf(stdout, " SYNCH");
    if (instDetails.flags & BEAGLE_FLAG_COMPLEX)fprintf(stdout, " COMPLEX");
    if (instDetails.flags & BEAGLE_FLAG_LSCALER)fprintf(stdout, " LSCALER");
    if (instDetails.flags & BEAGLE_FLAG_CPU)    fprintf(stdout, " CPU");
    if (instDetails.flags & BEAGLE_FLAG_GPU)    fprintf(stdout, " GPU");
    if (instDetails.flags & BEAGLE_FLAG_FPGA)   fprintf(stdout, " FPGA");
    if (instDetails.flags & BEAGLE_FLAG_SSE)    fprintf(stdout, " SSE");
    if (instDetails.flags & BEAGLE_FLAG_CELL)   fprintf(stdout, " CELL");
    fprintf(stdout, "\n\n");
    
    
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
    
//#ifdef _WIN32
//	std::vector<double> rates(rateCategoryCount);
//#else
//	double rates[rateCategoryCount];
//#endif
//    for (int i = 0; i < rateCategoryCount; i++) {
//        rates[i] = 1.0;
//    }
	double rates[4] = { 0.03338775, 0.25191592, 0.82026848, 2.89442785 };
    
	
    // create base frequency array
	double freqs[16] = { 0.25, 0.25, 0.25, 0.25,
						 0.25, 0.25, 0.25, 0.25,
						 0.25, 0.25, 0.25, 0.25,
		                 0.25, 0.25, 0.25, 0.25 };
    
    // create an array containing site category weights
#ifdef _WIN32
	std::vector<double> weights(rateCategoryCount);
#else
	double weights[rateCategoryCount];
#endif
    for (int i = 0; i < rateCategoryCount; i++) {
        weights[i] = 1.0/rateCategoryCount;
    }    
    
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
	beagleSetEigenDecomposition(instance, 0, evec, ivec, eval);
    
    // a list of indices and edge lengths
	int nodeIndices[4] = { 0, 1, 2, 3 };
	double edgeLengths[4] = { 0.1, 0.1, 0.2, 0.1 };
	
	int rootIndex[nRootCount];
    int cumulativeScalingIndex[nRootCount];
	
	for (int i = 0; i < nRootCount; i++) {
		
		rootIndex[i] = 4 + i;
		cumulativeScalingIndex[i] = (scaling ? 2 + i : BEAGLE_OP_NONE);
		
		beagleSetCategoryRates(instance, &rates[i]);
		
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
		int operations[BEAGLE_OP_COUNT * 2] = {
			3, (scaling ? 0 : BEAGLE_OP_NONE), BEAGLE_OP_NONE, 0, 0, 1, 1,
			rootIndex[i], (scaling ? 1 : BEAGLE_OP_NONE), BEAGLE_OP_NONE, 2, 2, 3, 3
		};
		
		if (scaling)
			beagleResetScaleFactors(instance, cumulativeScalingIndex[i]);
		
		// update the partials
		beagleUpdatePartials(&instance,      // instance
					   1,              // instanceCount
					   operations,     // eigenIndex
					   2,              // operationCount
					   cumulativeScalingIndex[i]);// cumulative scaling index
	}
		 
	double *patternLogLik = (double*)malloc(sizeof(double) * nPatterns);
    
    
    // calculate the site likelihoods at the root node
	beagleCalculateRootLogLikelihoods(instance,               // instance
	                            (const int *)rootIndex,// bufferIndices
	                            weights,                // weights
	                            freqs,                  // stateFrequencies
								cumulativeScalingIndex,// cumulative scaling index
	                            nRootCount,                      // count
	                            patternLogLik);         // outLogLikelihoods
    
	double logL = 0.0;
	for (int i = 0; i < nPatterns; i++) {
//		std::cerr << "site lnL[" << i << "] = " << patternLogLik[i] << '\n';
		logL += patternLogLik[i];
	}
    
	fprintf(stdout, "logL = %.5f (PAUP logL = -1498.89812)\n\n", logL);
// no rate heterogeneity:	
//	fprintf(stdout, "logL = %.5f (PAUP logL = -1574.63623)\n\n", logL);
	
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
