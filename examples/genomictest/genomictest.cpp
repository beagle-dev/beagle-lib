/*
 *  genomictest.cpp
 *  Created by Aaron Darling on 14/06/2009.
 *  Based on tinyTest.cpp by Andrew Rambaut.
 */
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <sys/time.h>
#include "libhmsbeagle/beagle.h"

int ntaxa = 29;
int nsites = 10000;



double* getRandomTipPartials( int nsites, int stateCount )
{
	double *partials = (double*) malloc(sizeof(double) * nsites * stateCount);
	for( int i=0; i<nsites*stateCount; i+=stateCount )
	{
		int s = rand()%stateCount;
		partials[i+s]=1.0;
	}
	return partials;
}


void runBeagle(int resource)
{
    // is nucleotides...
    int stateCount = 4;
    
    int rateCategoryCount = 4;
    
    int scaleCount = ntaxa;

    // create an instance of the BEAGLE library
	int instance = beagleCreateInstance(
			    ntaxa,			  /**< Number of tip data elements (input) */
				2*ntaxa-1,	      /**< Number of partials buffers to create (input) */
				0,		          /**< Number of compact state representation buffers to create (input) */
				stateCount,		  /**< Number of states in the continuous-time Markov chain (input) */
				nsites,			  /**< Number of site patterns to be handled by the instance (input) */
				1,		          /**< Number of rate matrix eigen-decomposition buffers to allocate (input) */
				2*ntaxa-2,	      /**< Number of rate matrix buffers (input) */
                rateCategoryCount,/**< Number of rate categories */
                scaleCount,          /**< scaling buffers */
				&resource,		  /**< List of potential resource on which this instance is allowed (input, NULL implies no restriction */
				1,			      /**< Length of resourceList list (input) */
				0,		          /**< Bit-flags indicating preferred implementation charactertistics, see BeagleFlags (input) */
				0		          /**< Bit-flags indicating required implementation characteristics, see BeagleFlags (input) */
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
    BeagleResourceList* rList = beagleGetResourceList();
    fprintf(stdout, "Using resource %i:\n", rNumber);
    fprintf(stdout, "\tName : %s\n", rList->list[rNumber].name);
    fprintf(stdout, "\tDesc : %s\n", rList->list[rNumber].description);
    fprintf(stdout, "\n");      
    
    // set the sequences for each tip using partial likelihood arrays
	srand(42);	// fix the random seed...
	for(int i=0; i<ntaxa; i++)
	{
		beagleSetTipPartials(instance, i, getRandomTipPartials(nsites, stateCount));
	}
    
	double rates[rateCategoryCount];
    for (int i = 0; i < rateCategoryCount; i++) {
        rates[i] = 1.0;
    }
    
	beagleSetCategoryRates(instance, rates);
	
    // create base frequency array
	double freqs[4] = { 0.25, 0.25, 0.25, 0.25 };

    // create an array containing site category weights
	double weights[rateCategoryCount];
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
	int* nodeIndices = new int[ntaxa*2-2];
	for(int i=0; i<ntaxa*2-2; i++) nodeIndices[i]=i;
	double* edgeLengths = new double[ntaxa*2-2];
	for(int i=0; i<ntaxa*2-2; i++) edgeLengths[i]=0.1;

    // tell BEAGLE to populate the transition matrices for the above edge lengths
	beagleUpdateTransitionMatrices(instance,     // instance
	                         0,             // eigenIndex
	                         nodeIndices,   // probabilityIndices
	                         NULL,          // firstDerivativeIndices
	                         NULL,          // secondDervativeIndices
	                         edgeLengths,   // edgeLengths
	                         ntaxa*2-2);            // count

    // create a list of partial likelihood update operations
    // the order is [dest, destScaling, source1, matrix1, source2, matrix2]
	int* operations = new int[(ntaxa-1)*BEAGLE_OP_COUNT];
    int* scalingFactorsIndices = new int[(ntaxa-1)]; // internal nodes
	for(int i=0; i<ntaxa-1; i++){
		operations[BEAGLE_OP_COUNT*i+0] = ntaxa+i;
        operations[BEAGLE_OP_COUNT*i+1] = i;
        operations[BEAGLE_OP_COUNT*i+2] = BEAGLE_OP_NONE;
		operations[BEAGLE_OP_COUNT*i+3] = i*2;
		operations[BEAGLE_OP_COUNT*i+4] = i*2;
		operations[BEAGLE_OP_COUNT*i+5] = i*2+1;
		operations[BEAGLE_OP_COUNT*i+6] = i*2+1;
        
        scalingFactorsIndices[i] = i;
	}	

	int rootIndex = ntaxa*2-2;

    // start timing!
	struct timeval start, end;
	gettimeofday(&start,NULL);

    // update the partials
	beagleUpdatePartials( &instance,      // instance
	                1,              // instanceCount
	                operations,     // eigenIndex
	                ntaxa-1,              // operationCount
	                BEAGLE_OP_NONE);             // cumulative scaling index

	double *patternLogLik = (double*)malloc(sizeof(double) * nsites);
    

    int scalingFactorsCount = ntaxa-1;
    
    int cumulativeScalingFactorIndex = ntaxa-1;

    beagleResetScaleFactors(instance,
                           cumulativeScalingFactorIndex);
    
    beagleAccumulateScaleFactors(instance,
                           scalingFactorsIndices,
                           scalingFactorsCount,
                           cumulativeScalingFactorIndex);

    
    // calculate the site likelihoods at the root node
	beagleCalculateRootLogLikelihoods(instance,               // instance
	                            (const int *)&rootIndex,// bufferIndices
	                            weights,                // weights
	                            freqs,                 // stateFrequencies
                                &cumulativeScalingFactorIndex,
	                            1,                      // count
	                            patternLogLik);         // outLogLikelihoods

	// end timing!
	gettimeofday(&end,NULL);

	double logL = 0.0;
	for (int i = 0; i < nsites; i++) {
		logL += patternLogLik[i];
	}


	fprintf(stdout, "logL = %.5f \n", logL);
	double timediff =  end.tv_sec - start.tv_sec + (double)(end.tv_usec-start.tv_usec)/1000000.0;
	std::cout << "Took " << timediff << " seconds\n\n";
	beagleFinalizeInstance(instance);
}

int main( int argc, const char* argv[] )
{
	std::cout << "Simulating genomic DNA with " << ntaxa << " taxa and " << nsites << " site patterns\n";

	BeagleResourceList* rl = beagleGetResourceList();
	if(rl != NULL){
		for(int i=0; i<rl->length; i++){
			runBeagle(i);
		}
	}else{
		runBeagle(NULL);
	}
}
