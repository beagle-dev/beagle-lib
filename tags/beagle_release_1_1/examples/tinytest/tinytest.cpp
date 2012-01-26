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
    if (inFlags & BEAGLE_FLAG_THREADING_NONE)     fprintf(stdout, " THREADING_NONE");
    if (inFlags & BEAGLE_FLAG_THREADING_OPENMP)   fprintf(stdout, " THREADING_OPENMP");
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
    
    bool manualScaling = false;
    bool autoScaling = false;
	bool gRates = false; // generalized rate categories, separate root buffers
    
    // is nucleotides...
    int stateCount = 4;
	
    // get the number of site patterns
	int nPatterns = strlen(human);
    
    int rateCategoryCount = 4;
	
	int nRateCats = (gRates ? 1 : rateCategoryCount);
	int nRootCount = (!gRates ? 1 : rateCategoryCount);
	int nPartBuffs = 4 + nRootCount;
    int scaleCount = (manualScaling ? 2 + nRootCount : 0);
    
    // initialize the instance
    BeagleInstanceDetails instDetails;
    
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
                                  BEAGLE_FLAG_PRECISION_DOUBLE | BEAGLE_FLAG_PROCESSOR_GPU | (autoScaling ? BEAGLE_FLAG_SCALING_AUTO : 0),	/**< Bit-flags indicating preferred implementation charactertistics, see BeagleFlags (input) */
                                  0,                /**< Bit-flags indicating required implementation characteristics, see BeagleFlags (input) */
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
    fprintf(stdout, "\tFlags:");
    printFlags(instDetails.flags);
    fprintf(stdout, "\n\n");
    
    if (!(instDetails.flags & BEAGLE_FLAG_SCALING_AUTO))
        autoScaling = false;
    
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

	double* weights = (double*) malloc(sizeof(double) * rateCategoryCount);

    for (int i = 0; i < rateCategoryCount; i++) {
        weights[i] = 1.0/rateCategoryCount;
    }    

	double* patternWeights = (double*) malloc(sizeof(double) * nPatterns);
    
    for (int i = 0; i < nPatterns; i++) {
        patternWeights[i] = 1.0;
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
    
    beagleSetStateFrequencies(instance, 0, freqs);
    
    beagleSetCategoryWeights(instance, 0, weights);
    
    beagleSetPatternWeights(instance, patternWeights);
    
    // a list of indices and edge lengths
	int nodeIndices[4] = { 0, 1, 2, 3 };
	double edgeLengths[4] = { 0.1, 0.1, 0.2, 0.1 };
	
	int* rootIndices = (int*) malloc(sizeof(int) * nRootCount);
    int* categoryWeightsIndices = (int*) malloc(sizeof(int) * nRootCount);
    int* stateFrequencyIndices = (int*) malloc(sizeof(int) * nRootCount);
	int* cumulativeScalingIndices = (int*) malloc(sizeof(int) * nRootCount);
	
	for (int i = 0; i < nRootCount; i++) {
		
		rootIndices[i] = 4 + i;
        categoryWeightsIndices[i] = 0;
        stateFrequencyIndices[i] = 0;
		cumulativeScalingIndices[i] = (manualScaling ? 2 + i : BEAGLE_OP_NONE);
		
		beagleSetCategoryRates(instance, &rates[i]);
		
		// tell BEAGLE to populate the transition matrices for the above edge lengths
		beagleUpdateTransitionMatrices(instance,     // instance
								 0,             // eigenIndex
								 nodeIndices,   // probabilityIndices
								 NULL,          // firstDerivativeIndices
								 NULL,          // secondDerivativeIndices
								 edgeLengths,   // edgeLengths
								 4);            // count
		
		// create a list of partial likelihood update operations
		// the order is [dest, destScaling, source1, matrix1, source2, matrix2]
		BeagleOperation operations[2] = {
			3, (manualScaling ? 0 : BEAGLE_OP_NONE), BEAGLE_OP_NONE, 0, 0, 1, 1,
			rootIndices[i], (manualScaling ? 1 : BEAGLE_OP_NONE), BEAGLE_OP_NONE, 2, 2, 3, 3
		};
		
		if (manualScaling)
			beagleResetScaleFactors(instance, cumulativeScalingIndices[i]);
		
		// update the partials
		beagleUpdatePartials(instance,      // instance
					   operations,     // eigenIndex
					   2,              // operationCount
					   cumulativeScalingIndices[i]);// cumulative scaling index
	}
		 
    if (autoScaling) {
        int scaleIndices[2] = {3, 4};
        beagleAccumulateScaleFactors(instance, scaleIndices, 2, BEAGLE_OP_NONE);
    }
    
	double *patternLogLik = (double*)malloc(sizeof(double) * nPatterns);
	double logL = 0.0;    
    int returnCode = 0;
    
    // calculate the site likelihoods at the root node
	returnCode = beagleCalculateRootLogLikelihoods(instance,               // instance
	                            (const int *)rootIndices,// bufferIndices
	                            (const int *)categoryWeightsIndices,                // weights
	                            (const int *)stateFrequencyIndices,                  // stateFrequencies
								cumulativeScalingIndices,// cumulative scaling index
	                            nRootCount,                      // count
	                            &logL);         // outLogLikelihoods
    
    if (returnCode < 0) {
	    fprintf(stderr, "Failed to calculate root likelihood\n\n");
    } else {

        beagleGetSiteLogLikelihoods(instance, patternLogLik);
        double sumLogL = 0.0;
        for (int i = 0; i < nPatterns; i++) {
            sumLogL += patternLogLik[i] * patternWeights[i];
//            std::cerr << "site lnL[" << i << "] = " << patternLogLik[i] << '\n';
        }
      
        fprintf(stdout, "logL = %.5f (PAUP logL = -1498.89812)\n", logL);
        fprintf(stdout, "sumLogL = %.5f\n", sumLogL);  
    }
    
// no rate heterogeneity:	
//	fprintf(stdout, "logL = %.5f (PAUP logL = -1574.63623)\n\n", logL);
	
    free(weights);
    free(patternWeights);    
    free(rootIndices);
    free(categoryWeightsIndices);
    free(stateFrequencyIndices);
    free(cumulativeScalingIndices);    
    
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
