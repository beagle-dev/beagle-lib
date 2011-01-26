/*
 *  genomictest.cpp
 *  Created by Aaron Darling on 14/06/2009.
 *  @author Aaron Darling
 *  @author Daniel Ayres
 *  Based on tinyTest.cpp by Andrew Rambaut.
 */
#include <cstdio>
#include <string>
#include <cstdlib>
#include <iostream>
#include <iomanip>

#ifdef _WIN32
	#include <vector>
	#include <winsock.h>
#else
	#include <sys/time.h>
#endif

#include "libhmsbeagle/beagle.h"

#ifdef _WIN32
	//From January 1, 1601 (UTC). to January 1,1970
	#define FACTOR 0x19db1ded53e8000 

	int gettimeofday(struct timeval *tp,void * tz) {
		FILETIME f;
		ULARGE_INTEGER ifreq;
		LONGLONG res; 
		GetSystemTimeAsFileTime(&f);
		ifreq.HighPart = f.dwHighDateTime;
		ifreq.LowPart = f.dwLowDateTime;

		res = ifreq.QuadPart - FACTOR;
		tp->tv_sec = (long)((LONGLONG)res/10000000);
		tp->tv_usec = (long)((LONGLONG)res% 10000000000); // Micro Seonds

		return 0;
	}
#endif

double cpuTimeUpdateTransitionMatrices, cpuTimeUpdatePartials, cpuTimeAccumulateScaleFactors, cpuTimeCalculateRootLogLikelihoods, cpuTimeTotal;

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

void printTiming(double timingValue,
                 int timePrecision,
                 bool printSpeedup,
                 double cpuTimingValue,
                 int speedupPrecision,
                 bool printPercent,
                 double totalTime,
                 int percentPrecision) {
	std::cout << std::setprecision(timePrecision) << timingValue << "s";
    if (printSpeedup) std::cout << " (" << std::setprecision(speedupPrecision) << cpuTimingValue/timingValue << "x CPU)";
    if (printPercent) std::cout << " (" << std::setw(3+percentPrecision) << std::setfill('0') << std::setprecision(percentPrecision) << (double)(timingValue/totalTime)*100 << "%)";
    std::cout << "\n";
}

double getTimeDiff(struct timeval t1,
                   struct timeval t2) {
    return ((t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec-t1.tv_usec)/1000000.0);
}

void runBeagle(int resource, 
               int stateCount, 
               int ntaxa, 
               int nsites, 
               bool manualScaling, 
               bool autoScaling,
               bool dynamicScaling,
               int rateCategoryCount,
               int nreps,
               bool fullTiming,
               bool preferDoublePrecision)
{
    
    int scaleCount = ((manualScaling || dynamicScaling) ? ntaxa : 0);
    
    BeagleInstanceDetails instDetails;
    
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
                (preferDoublePrecision ? BEAGLE_FLAG_PRECISION_DOUBLE : BEAGLE_FLAG_PRECISION_SINGLE),		          /**< Bit-flags indicating preferred implementation charactertistics, see BeagleFlags (input) */
				(dynamicScaling ? BEAGLE_FLAG_SCALING_DYNAMIC : 0) | (autoScaling ? BEAGLE_FLAG_SCALING_AUTO : 0),	  /**< Bit-flags indicating required implementation characteristics, see BeagleFlags (input) */
				&instDetails);
    if (instance < 0) {
	    fprintf(stderr, "Failed to obtain BEAGLE instance\n\n");
	    return;
    }
        
    int rNumber = instDetails.resourceNumber;
    fprintf(stdout, "Using resource %i:\n", rNumber);
    fprintf(stdout, "\tRsrc Name : %s\n",instDetails.resourceName);
    fprintf(stdout, "\tImpl Name : %s\n", instDetails.implName);    
    
    if (!(instDetails.flags & BEAGLE_FLAG_SCALING_AUTO))
        autoScaling = false;
    
    // set the sequences for each tip using partial likelihood arrays
	srand(42);	// fix the random seed...
	for(int i=0; i<ntaxa; i++)
	{
        double* tmpPartials = getRandomTipPartials(nsites, stateCount);
		beagleSetTipPartials(instance, i, tmpPartials);
        free(tmpPartials);
	}
    
#ifdef _WIN32
	std::vector<double> rates(rateCategoryCount);
#else
    double rates[rateCategoryCount];
#endif
	
    for (int i = 0; i < rateCategoryCount; i++) {
        rates[i] = 1.0;
    }
    
	beagleSetCategoryRates(instance, &rates[0]);
    
	double* patternWeights = (double*) malloc(sizeof(double) * nsites);
    
    for (int i = 0; i < nsites; i++) {
        patternWeights[i] = 1.0;
    }    

    beagleSetPatternWeights(instance, patternWeights);
    
    free(patternWeights);
	
    // create base frequency array

#ifdef _WIN32
	std::vector<double> freqs(stateCount);
#else
    double freqs[stateCount];
#endif
   
    for (int i=0; i<stateCount; i++) {
        freqs[i] = 1.0 / stateCount;
    }

    beagleSetStateFrequencies(instance, 0, &freqs[0]);
    
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
#ifdef _WIN32
	std::vector<double> ivec(stateCount * stateCount);
#else
    double ivec[stateCount * stateCount];
#endif

    for (int i=0; i<stateCount; i++) {
        for (int j=i; j<stateCount; j++) {
            ivec[i*stateCount+j] = evec[j*stateCount+i] / stateCount;
            ivec[j*stateCount+i] = ivec[i*stateCount+j]; // Symmetric
        }
    }

#ifdef _WIN32
	std::vector<double> eval(stateCount);
#else
    double eval[stateCount];
#endif
   
    eval[0] = 0.0;
    for (int i=1; i<stateCount; i++) {
        eval[i] = -stateCount / (stateCount - 1.0);
    }

    // set the Eigen decomposition
	beagleSetEigenDecomposition(instance, 0, &evec[0], &ivec[0], &eval[0]);

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
        operations[BEAGLE_OP_COUNT*i+1] = ((manualScaling || dynamicScaling) ? i : BEAGLE_OP_NONE);
        operations[BEAGLE_OP_COUNT*i+2] = (dynamicScaling ? i : BEAGLE_OP_NONE);
		operations[BEAGLE_OP_COUNT*i+3] = i*2;
		operations[BEAGLE_OP_COUNT*i+4] = i*2;
		operations[BEAGLE_OP_COUNT*i+5] = i*2+1;
		operations[BEAGLE_OP_COUNT*i+6] = i*2+1;

        scalingFactorsIndices[i] = i;
        
        if (autoScaling)
            scalingFactorsIndices[i] += ntaxa;
	}	

	int rootIndex = ntaxa*2-2;

    // start timing!
	struct timeval time1, time2, time3, time4, time5;
    double bestTimeUpdateTransitionMatrices, bestTimeUpdatePartials, bestTimeAccumulateScaleFactors, bestTimeCalculateRootLogLikelihoods, bestTimeTotal;
    
    double logL = 0.0;
    
    int cumulativeScalingFactorIndex = ((manualScaling || dynamicScaling) ? ntaxa-1 : BEAGLE_OP_NONE);

    if (dynamicScaling)
        beagleResetScaleFactors(instance, cumulativeScalingFactorIndex);
    
    for (int i=0; i<nreps; i++){
        gettimeofday(&time1,NULL);

        // tell BEAGLE to populate the transition matrices for the above edge lengths
        beagleUpdateTransitionMatrices(instance,     // instance
                                       0,             // eigenIndex
                                       nodeIndices,   // probabilityIndices
                                       NULL,          // firstDerivativeIndices
                                       NULL,          // secondDerivativeIndices
                                       edgeLengths,   // edgeLengths
                                       ntaxa*2-2);            // count    

        gettimeofday(&time2, NULL);
        
        // update the partials
        beagleUpdatePartials( instance,      // instance
                        (BeagleOperation*)operations,     // eigenIndex
                        ntaxa-1,              // operationCount
                        (dynamicScaling ? ntaxa-1 : BEAGLE_OP_NONE));             // cumulative scaling index

        gettimeofday(&time3, NULL);

        int scalingFactorsCount = ntaxa-1;
                
        if (manualScaling) {
            beagleResetScaleFactors(instance,
                                    cumulativeScalingFactorIndex);
            
            beagleAccumulateScaleFactors(instance,
                                   scalingFactorsIndices,
                                   scalingFactorsCount,
                                   cumulativeScalingFactorIndex);
        } else if (autoScaling) {
            beagleAccumulateScaleFactors(instance, scalingFactorsIndices, scalingFactorsCount, BEAGLE_OP_NONE);
        }

        gettimeofday(&time4, NULL);
        
        int categoryWeightsIndex = 0;
        int stateFrequencyIndex = 0;
        
        // calculate the site likelihoods at the root node
        beagleCalculateRootLogLikelihoods(instance,               // instance
                                    (const int *)&rootIndex,// bufferIndices
                                    &categoryWeightsIndex,                // weights
                                    &stateFrequencyIndex,                 // stateFrequencies
                                    &cumulativeScalingFactorIndex,
                                    1,                      // count
                                    &logL);         // outLogLikelihoods

        // end timing!
        gettimeofday(&time5,NULL);
        
        if (i == 0 || getTimeDiff(time1, time2) < bestTimeUpdateTransitionMatrices)
            bestTimeUpdateTransitionMatrices = getTimeDiff(time1, time2);
        if (i == 0 || getTimeDiff(time2, time3) < bestTimeUpdatePartials)
            bestTimeUpdatePartials = getTimeDiff(time2, time3);
        if (i == 0 || getTimeDiff(time3, time4) < bestTimeAccumulateScaleFactors)
            bestTimeAccumulateScaleFactors = getTimeDiff(time3, time4);
        if (i == 0 || getTimeDiff(time4, time5) < bestTimeUpdateTransitionMatrices)
            bestTimeCalculateRootLogLikelihoods = getTimeDiff(time4, time5);
        if (i == 0 || getTimeDiff(time1, time5) < bestTimeTotal)
            bestTimeTotal = getTimeDiff(time1, time5);
        
    }

    if (resource == 0) {
        cpuTimeUpdateTransitionMatrices = bestTimeUpdateTransitionMatrices;
        cpuTimeUpdatePartials = bestTimeUpdatePartials;
        cpuTimeAccumulateScaleFactors = bestTimeAccumulateScaleFactors;
        cpuTimeCalculateRootLogLikelihoods = bestTimeCalculateRootLogLikelihoods;
        cpuTimeTotal = bestTimeTotal;
    }
    
	fprintf(stdout, "logL = %.5f \n", logL);
    
    std::cout.setf(std::ios::showpoint);
    std::cout.setf(std::ios::floatfield, std::ios::fixed);
    int timePrecision = 6;
    int speedupPrecision = 2;
    int percentPrecision = 2;
	std::cout << "best run: ";
    printTiming(bestTimeTotal, timePrecision, resource, cpuTimeTotal, speedupPrecision, 0, 0, 0);
    if (fullTiming) {
        std::cout << " transMats:  ";
        printTiming(bestTimeUpdateTransitionMatrices, timePrecision, resource, cpuTimeUpdateTransitionMatrices, speedupPrecision, 1, bestTimeTotal, percentPrecision);
        std::cout << " partials:   ";
        printTiming(bestTimeUpdatePartials, timePrecision, resource, cpuTimeUpdatePartials, speedupPrecision, 1, bestTimeTotal, percentPrecision);
        if (manualScaling || autoScaling) {
            std::cout << " accScalers: ";
            printTiming(bestTimeAccumulateScaleFactors, timePrecision, resource, cpuTimeAccumulateScaleFactors, speedupPrecision, 1, bestTimeTotal, percentPrecision);
        }
        std::cout << " rootLnL:    ";
        printTiming(bestTimeCalculateRootLogLikelihoods, timePrecision, resource, cpuTimeCalculateRootLogLikelihoods, speedupPrecision, 1, bestTimeTotal, percentPrecision);
    }
    std::cout << "\n";
    
	beagleFinalizeInstance(instance);
    free(evec);
}

void abort(std::string msg) {
	std::cerr << msg << "\nAborting..." << std::endl;
	std::exit(1);
}

void helpMessage() {
	std::cerr << "Usage:\n\n";
	std::cerr << "genomictest [--help] [--states <integer>] [--taxa <integer>] [--sites <integer>] [--rates <integer>] [--manualscale] [--autoscale] [--dynamicscale] [--rsrc <integer>] [--reps <integer>] [--full-timing]\n\n";
    std::cerr << "If --help is specified, this usage message is shown\n\n";
    std::cerr << "If --manualscale, --autoscale, or --dynamicscale is specified, BEAGLE will rescale the partials during computation\n\n";
    std::cerr << "If --full-timing is specified, you will see more detailed timing results (requires BEAGLE_DEBUG_SYNCH defined to report accurate values)\n\n";
	std::exit(0);
}


void interpretCommandLineParameters(int argc, const char* argv[],
                                    int* stateCount,
                                    int* ntaxa,
                                    int* nsites,
                                    bool* manualScaling,
                                    bool* autoScaling,
                                    bool* dynamicScaling,
                                    int* rateCategoryCount,
                                    int* rsrc,
                                    int* nreps,
                                    bool* fullTiming,
                                    bool* preferDoublePrecision)	{
    bool expecting_stateCount = false;
	bool expecting_ntaxa = false;
	bool expecting_nsites = false;
	bool expecting_rateCategoryCount = false;
	bool expecting_nreps = false;
	bool expecting_rsrc = false;
	
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
        } else if (expecting_rsrc) {
            *rsrc = (unsigned)atoi(option.c_str());
            expecting_rsrc = false;            
        } else if (expecting_nreps) {
            *nreps = (unsigned)atoi(option.c_str());
            expecting_nreps = false;
        } else if (option == "--help") {
			helpMessage();
        } else if (option == "--manualscale") {
            *manualScaling = true;
        } else if (option == "--autoscale") {
        	*autoScaling = true;
        } else if (option == "--dynamicscale") {
        	*dynamicScaling = true;
        } else if (option == "--doubleprecision") {
        	*preferDoublePrecision = true;
        } else if (option == "--states") {
            expecting_stateCount = true;
        } else if (option == "--taxa") {
            expecting_ntaxa = true;
        } else if (option == "--sites") {
            expecting_nsites = true;
        } else if (option == "--rates") {
            expecting_rateCategoryCount = true;
        } else if (option == "--rsrc") {
            expecting_rsrc = true;
        } else if (option == "--reps") {
            expecting_nreps = true;
        } else if (option == "--full-timing") {
            *fullTiming = true;
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

	if (expecting_rsrc)
		abort("read last command line option without finding value associated with --rsrc");
    
	if (expecting_nreps)
		abort("read last command line option without finding value associated with --reps");

	if (*stateCount < 2 || 
        (*stateCount & (*stateCount-1)) != 0)
		abort("invalid number of states (must be a power-of-two) supplied on the command line");
        
	if (*ntaxa < 2)
		abort("invalid number of taxa supplied on the command line");
      
	if (*nsites < 1)
		abort("invalid number of sites supplied on the command line");
    
    if (*rateCategoryCount < 1) {
        abort("invalid number of rates supplied on the command line");
        
    if (*nreps < 1)
        abort("invalid number of reps supplied on the command line");
    }
}

int main( int argc, const char* argv[] )
{
    // Default values
    int stateCount = 4;
    int ntaxa = 29;
    int nsites = 10000;
    bool manualScaling = false;
    bool autoScaling = false;
    bool dynamicScaling = false;
    bool preferDoublePrecision = false;

    int rsrc = -1;
    int nreps = 5;
    bool fullTiming = false;
    
    int rateCategoryCount = 4;
    
    interpretCommandLineParameters(argc, argv, &stateCount, &ntaxa, &nsites, &manualScaling, &autoScaling, &dynamicScaling, &rateCategoryCount, &rsrc, &nreps, &fullTiming, &preferDoublePrecision);
    
	std::cout << "\nSimulating genomic ";
    if (stateCount == 4)
        std::cout << "DNA";
    else
        std::cout << stateCount << "-state data";
    std::cout << " with " << ntaxa << " taxa and " << nsites << " site patterns (" << nreps << " rep" << (nreps > 1 ? "s" : "");
    std::cout << (manualScaling ? ", manual scaling":(autoScaling ? ", auto scaling":(dynamicScaling ? ", dynamic scaling":""))) << ")\n\n";

    if (rsrc != -1) {
        runBeagle(rsrc,
                  stateCount,
                  ntaxa,
                  nsites,
                  manualScaling,
                  autoScaling,
                  dynamicScaling,
                  rateCategoryCount,
                  nreps,
                  fullTiming,
                  preferDoublePrecision);
    } else {
        BeagleResourceList* rl = beagleGetResourceList();
        if(rl != NULL){
            for(int i=0; i<rl->length; i++){
                runBeagle(i,
                          stateCount,
                          ntaxa,
                          nsites,
                          manualScaling,
                          autoScaling,
                          dynamicScaling,
                          rateCategoryCount,
                          nreps,
                          fullTiming,
                          preferDoublePrecision);
            }
        }else{
            runBeagle(0,
                      stateCount,
                      ntaxa,
                      nsites,
                      manualScaling,
                      autoScaling,
                      dynamicScaling,
                      rateCategoryCount,
                      nreps,
                      fullTiming,
                      preferDoublePrecision);
        }
	}

#ifdef _WIN32
    std::cout << "\nPress ENTER to exit...\n";
    fflush( stdout);
    fflush( stderr);
    getchar();
#endif
}
