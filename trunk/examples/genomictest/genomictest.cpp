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

double* getRandomTipPartials( int nsites, int stateCount )
{
	double *partials = (double*) calloc(sizeof(double), nsites * stateCount); // 'malloc' was a bug
	for( int i=0; i<nsites*stateCount; i+=stateCount )
	{
		int s = rand()%stateCount;
		partials[i+s]=1.0;
	}
	return partials;
}


void runBeagle(int resource, 
               int stateCount, 
               int ntaxa, 
               int nsites, 
               bool scaling, 
               int rateCategoryCount)
{
    
    int scaleCount = (scaling ? ntaxa : 0);

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
    double freqs[stateCount];
    for (int i=0; i<stateCount; i++) {
        freqs[i] = 1.0 / stateCount;
    }

    // create an array containing site category weights
	double weights[rateCategoryCount];
    for (int i = 0; i < rateCategoryCount; i++) {
        weights[i] = 1.0/rateCategoryCount;
    } 

	// an eigen decomposition for the general state-space JC69 model
    // If stateCount = 2^n is a power-of-two, then Sylvester matrix H_n describes
    // the eigendecomposition of the infinitesimal rate matrix
     
    double* Hn = (double*)malloc(sizeof(double)*stateCount*stateCount);
    Hn[0*stateCount+0] = 1.0; Hn[0*stateCount+1] =  1.0; 
    Hn[1*stateCount+0] = 1.0; Hn[1*stateCount+1] = -1.0; // H_1
 
    for (int k=2; k < stateCount; k <<= 1) {
        // H_n = H_1 (Kronecker product) H_{n-1}
        for (int i=0; i<k; i++) {
            for (int j=i; j<k; j++) {
                double Hijold = Hn[i*stateCount + j];
                Hn[i    *stateCount + j + k] =  Hijold;
                Hn[(i+k)*stateCount + j    ] =  Hijold;
                Hn[(i+k)*stateCount + j + k] = -Hijold;
                
                Hn[j    *stateCount + i + k] = Hn[i    *stateCount + j + k];
                Hn[(j+k)*stateCount + i    ] = Hn[(i+k)*stateCount + j    ];
                Hn[(j+k)*stateCount + i + k] = Hn[(i+k)*stateCount + j + k];                                
            }
        }        
    }
    double* evec = Hn;
    
    // Since evec is Hadamard, ivec = (evec)^t / stateCount;    
    double ivec[stateCount * stateCount];
    for (int i=0; i<stateCount; i++) {
        for (int j=i; j<stateCount; j++) {
            ivec[i*stateCount+j] = evec[j*stateCount+i] / stateCount;
            ivec[j*stateCount+i] = ivec[i*stateCount+j]; // Symmetric
        }
    }

    double eval[stateCount];
    eval[0] = 0.0;
    for (int i=1; i<stateCount; i++) {
        eval[i] = -stateCount / (stateCount - 1.0);
    }

    // set the Eigen decomposition
	beagleSetEigenDecomposition(instance, 0, evec, ivec, eval);

    // a list of indices and edge lengths
	int* nodeIndices = new int[ntaxa*2-2];
	for(int i=0; i<ntaxa*2-2; i++) nodeIndices[i]=i;
	double* edgeLengths = new double[ntaxa*2-2];
	for(int i=0; i<ntaxa*2-2; i++) edgeLengths[i]=0.1;

    // create a list of partial likelihood update operations
    // the order is [dest, destScaling, source1, matrix1, source2, matrix2]
	int* operations = new int[(ntaxa-1)*BEAGLE_OP_COUNT];
    int* scalingFactorsIndices = new int[(ntaxa-1)]; // internal nodes
	for(int i=0; i<ntaxa-1; i++){
		operations[BEAGLE_OP_COUNT*i+0] = ntaxa+i;
        operations[BEAGLE_OP_COUNT*i+1] = (scaling ? i : BEAGLE_OP_NONE);
        operations[BEAGLE_OP_COUNT*i+2] = BEAGLE_OP_NONE;
		operations[BEAGLE_OP_COUNT*i+3] = i*2;
		operations[BEAGLE_OP_COUNT*i+4] = i*2;
		operations[BEAGLE_OP_COUNT*i+5] = i*2+1;
		operations[BEAGLE_OP_COUNT*i+6] = i*2+1;
        
        scalingFactorsIndices[i] = i;
	}	

	int rootIndex = ntaxa*2-2;

    // start timing!
	struct timeval time1, time2, time3;
	gettimeofday(&time1,NULL);
    
    // tell BEAGLE to populate the transition matrices for the above edge lengths
	beagleUpdateTransitionMatrices(instance,     // instance
                                   0,             // eigenIndex
                                   nodeIndices,   // probabilityIndices
                                   NULL,          // firstDerivativeIndices
                                   NULL,          // secondDervativeIndices
                                   edgeLengths,   // edgeLengths
                                   ntaxa*2-2);            // count    

    gettimeofday(&time2, NULL);
    
    // update the partials
	beagleUpdatePartials( &instance,      // instance
	                1,              // instanceCount
	                operations,     // eigenIndex
	                ntaxa-1,              // operationCount
	                BEAGLE_OP_NONE);             // cumulative scaling index

	double *patternLogLik = (double*)malloc(sizeof(double) * nsites);
    

    int scalingFactorsCount = ntaxa-1;
    
    int cumulativeScalingFactorIndex = (scaling ? ntaxa-1 : BEAGLE_OP_NONE);
    
    
    if (scaling) {
        beagleResetScaleFactors(instance,
                               cumulativeScalingFactorIndex);
        
        beagleAccumulateScaleFactors(instance,
                               scalingFactorsIndices,
                               scalingFactorsCount,
                               cumulativeScalingFactorIndex);
    }
    
    // calculate the site likelihoods at the root node
	beagleCalculateRootLogLikelihoods(instance,               // instance
	                            (const int *)&rootIndex,// bufferIndices
	                            weights,                // weights
	                            freqs,                 // stateFrequencies
                                &cumulativeScalingFactorIndex,
	                            1,                      // count
	                            patternLogLik);         // outLogLikelihoods

	// end timing!
	gettimeofday(&time3,NULL);

	double logL = 0.0;
	for (int i = 0; i < nsites; i++) {
		logL += patternLogLik[i];
	}


	fprintf(stdout, "logL = %.5f \n", logL);
	double timediff1 =  time2.tv_sec - time1.tv_sec + (double)(time2.tv_usec-time1.tv_usec)/1000000.0;
    double timediff2 =  time3.tv_sec - time2.tv_sec + (double)(time3.tv_usec-time2.tv_usec)/1000000.0;
	std::cout << "Took " << timediff1 << " and\n";
    std::cout << "     " << timediff2 << " seconds\n\n";
	beagleFinalizeInstance(instance);
    free(evec);
}

void abort(std::string msg) {
	std::cerr << msg << "\nAborting..." << std::endl;
	std::exit(1);
}

void helpMessage() {
	std::cerr << "Usage:\n\n";
	std::cerr << "genomictest [--help] [--states <integer>] [--taxa <integer>] [--sites <integer>] [--rates <integer>] [--scale]\n\n";
    std::cerr << "If --help is specified, this usage message is shown\n\n";
    std::cerr << "If --scale is specified, BEAGLE will rescale the partials during computation\n\n";
	std::exit(0);
}


void interpretCommandLineParameters(int argc, const char* argv[],
                                    int* stateCount,
                                    int* ntaxa,
                                    int* nsites,
                                    bool* scaling,
                                    int* rateCategoryCount)	{
    bool expecting_stateCount = false;
	bool expecting_ntaxa = false;
	bool expecting_nsites = false;
	bool expecting_rateCategoryCount = false;
	
    for (unsigned i = 1; i < argc; ++i) {
		std::string option = argv[i];
        
        if (expecting_stateCount) {
            *stateCount = (unsigned)atoi(option.c_str());
            expecting_stateCount = false;
        } else if (expecting_ntaxa) {
            *ntaxa = (unsigned)atoi(option.c_str());
            expecting_ntaxa = false;
        } else if (expecting_nsites) {
            *nsites = (unsigned)atoi(option.c_str());
            expecting_nsites = false;
        } else if (expecting_rateCategoryCount) {
            *rateCategoryCount = (unsigned)atoi(option.c_str());
            expecting_rateCategoryCount = false;
        } else if (option == "--help") {
			helpMessage();
        } else if (option == "--scale") {
            *scaling = true;
        } else if (option == "--states") {
            expecting_stateCount = true;
        } else if (option == "--taxa") {
            expecting_ntaxa = true;
        } else if (option == "--sites") {
            expecting_nsites = true;
        } else if (option == "--rates") {
            expecting_rateCategoryCount = true;
        } else {
			std::string msg("Unknown command line parameter \"");
			msg.append(option);			
			abort(msg.c_str());
        }
    }
    
	if (expecting_stateCount)
		abort("read last command line option without finding value associated with --states");
    
	if (expecting_ntaxa)
		abort("read last command line option without finding value associated with --taxa");
    
	if (expecting_nsites)
		abort("read last command line option without finding value associated with --sites");
	
	if (expecting_rateCategoryCount)
		abort("read last command line option without finding value associated with --rates");
	
	if (*stateCount < 2 || 
        (*stateCount & (*stateCount-1)) != 0)
		abort("invalid number of states (must be a power-of-two) supplied on the command line");
        
	if (*ntaxa < 2)
		abort("invalid number of taxa supplied on the command line");
      
	if (*nsites < 1)
		abort("invalid number of sites supplied on the command line");
    
    if (*rateCategoryCount < 1) {
        abort("invalid number of rates supplied on the command line");
    }
}

int main( int argc, const char* argv[] )
{
    // Default values
    int stateCount = 4;
    int ntaxa = 29;
    int nsites = 10000;
    bool scaling = false;
    int rateCategoryCount = 4;
    
    interpretCommandLineParameters(argc, argv, &stateCount, &ntaxa, &nsites, &scaling, &rateCategoryCount);
    
	std::cout << "Simulating genomic ";
    if (stateCount == 4)
        std::cout << "DNA";
    else
        std::cout << stateCount << "-state data";
    std::cout << " with " << ntaxa << " taxa and " << nsites << " site patterns\n";

	BeagleResourceList* rl = beagleGetResourceList();
	if(rl != NULL){
		for(int i=0; i<rl->length; i++){
			runBeagle(i,
                      stateCount,
                      ntaxa,
                      nsites,
                      scaling,
                      rateCategoryCount);                      
		}
	}else{
		runBeagle(NULL,
                  stateCount,
                  ntaxa,
                  nsites,
                  scaling,
                  rateCategoryCount);
	}
}
