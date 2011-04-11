/*
 *  genomictest.cpp
 *  Created by Aaron Darling on 14/06/2009.
 *  @author Aaron Darling
 *  @author Daniel Ayres
 *  Based on tinyTest.cpp by Andrew Rambaut.
 */
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>

#ifdef _WIN32
	#include <vector>
	#include <winsock.h>
	#include <string>
#else
	#include <sys/time.h>
#endif

#include "libhmsbeagle/beagle.h"
#include "linalg.h"

#define MAX_DIFF    0.01        //max discrepancy in scoring between reps
#define	GT_RAND_MAX 0x7fffffff

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
		tp->tv_usec =(long)(((LONGLONG)res%10000000)/10); // Micro Seconds

		return 0;
	}
#endif

double cpuTimeUpdateTransitionMatrices, cpuTimeUpdatePartials, cpuTimeAccumulateScaleFactors, cpuTimeCalculateRootLogLikelihoods, cpuTimeTotal;

static unsigned int rand_state = 1;

int gt_rand_r(unsigned int *seed)
{
    *seed = *seed * 1103515245 + 12345;
    return (*seed % ((unsigned int)GT_RAND_MAX + 1));
}

int gt_rand(void)
{
    return (gt_rand_r(&rand_state));
}

void gt_srand(unsigned int seed)
{
    rand_state = seed;
}

void abort(std::string msg) {
	std::cerr << msg << "\nAborting..." << std::endl;
	std::exit(1);
}

double* getRandomTipPartials( int nsites, int stateCount )
{
	double *partials = (double*) calloc(sizeof(double), nsites * stateCount); // 'malloc' was a bug
	for( int i=0; i<nsites*stateCount; i+=stateCount )
	{
		int s = gt_rand()%stateCount;
		partials[i+s]=1.0;
	}
	return partials;
}

int* getRandomTipStates( int nsites, int stateCount )
{
	int *states = (int*) calloc(sizeof(int), nsites); 
	for( int i=0; i<nsites; i++ )
	{
		int s = gt_rand()%stateCount;
		states[i]=s;
	}
	return states;
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
               bool requireDoublePrecision,
               bool requireSSE,
               int compactTipCount,
               int randomSeed,
               int rescaleFrequency,
               bool unrooted,
               bool calcderivs,
               bool logscalers,
               int eigenCount,
               bool eigencomplex,
               bool ievectrans,
               bool setmatrix)
{
    
    int edgeCount = ntaxa*2-2;
    int internalCount = ntaxa-1;
    int partialCount = ((ntaxa+internalCount)-compactTipCount)*eigenCount;
    int scaleCount = ((manualScaling || dynamicScaling) ? ntaxa : 0);
    
    BeagleInstanceDetails instDetails;
    
    // create an instance of the BEAGLE library
	int instance = beagleCreateInstance(
			    ntaxa,			  /**< Number of tip data elements (input) */
				partialCount, /**< Number of partials buffers to create (input) */
                compactTipCount,	/**< Number of compact state representation buffers to create (input) */
				stateCount,		  /**< Number of states in the continuous-time Markov chain (input) */
				nsites,			  /**< Number of site patterns to be handled by the instance (input) */
				eigenCount,		          /**< Number of rate matrix eigen-decomposition buffers to allocate (input) */
                (calcderivs ? (3*edgeCount*eigenCount) : edgeCount*eigenCount),/**< Number of rate matrix buffers (input) */
                rateCategoryCount,/**< Number of rate categories */
                scaleCount*eigenCount,          /**< scaling buffers */
				&resource,		  /**< List of potential resource on which this instance is allowed (input, NULL implies no restriction */
				1,			      /**< Length of resourceList list (input) */
                0,         /**< Bit-flags indicating preferred implementation charactertistics, see BeagleFlags (input) */
                (ievectrans ? BEAGLE_FLAG_INVEVEC_TRANSPOSED : BEAGLE_FLAG_INVEVEC_STANDARD) |
                (logscalers ? BEAGLE_FLAG_SCALERS_LOG : BEAGLE_FLAG_SCALERS_RAW) |
                (eigencomplex ? BEAGLE_FLAG_EIGEN_COMPLEX : BEAGLE_FLAG_EIGEN_REAL) |
                (dynamicScaling ? BEAGLE_FLAG_SCALING_DYNAMIC : 0) | 
                (autoScaling ? BEAGLE_FLAG_SCALING_AUTO : 0) |
                (requireDoublePrecision ? BEAGLE_FLAG_PRECISION_DOUBLE : BEAGLE_FLAG_PRECISION_SINGLE) |
                (requireSSE ? BEAGLE_FLAG_VECTOR_SSE : BEAGLE_FLAG_VECTOR_NONE),	  /**< Bit-flags indicating required implementation characteristics, see BeagleFlags (input) */
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
	gt_srand(randomSeed);	// fix the random seed...
    for(int i=0; i<ntaxa; i++)
    {
        if (i >= compactTipCount) {
            double* tmpPartials = getRandomTipPartials(nsites, stateCount);
            beagleSetTipPartials(instance, i, tmpPartials);
            free(tmpPartials);
        } else {
            int* tmpStates = getRandomTipStates(nsites, stateCount);
            beagleSetTipStates(instance, i, tmpStates);
            free(tmpStates);                
        }
    }
    
#ifdef _WIN32
	std::vector<double> rates(rateCategoryCount);
#else
    double rates[rateCategoryCount];
#endif
	
    for (int i = 0; i < rateCategoryCount; i++) {
        rates[i] = gt_rand() / (double) GT_RAND_MAX;
    }
    
	beagleSetCategoryRates(instance, &rates[0]);
    
	double* patternWeights = (double*) malloc(sizeof(double) * nsites);
    
    for (int i = 0; i < nsites; i++) {
        patternWeights[i] = gt_rand() / (double) GT_RAND_MAX;
    }    

    beagleSetPatternWeights(instance, patternWeights);
    
    free(patternWeights);
	
    // create base frequency array

#ifdef _WIN32
	std::vector<double> freqs(stateCount);
#else
    double freqs[stateCount];
#endif
    
    // create an array containing site category weights
#ifdef _WIN32
	std::vector<double> weights(rateCategoryCount);
#else
    double weights[rateCategoryCount];
#endif

    for (int eigenIndex=0; eigenIndex < eigenCount; eigenIndex++) {
        for (int i = 0; i < rateCategoryCount; i++) {
            weights[i] = gt_rand() / (double) GT_RAND_MAX;
        } 
    
        beagleSetCategoryWeights(instance, eigenIndex, &weights[0]);
    }
    
    double* eval;
    if (!eigencomplex)
        eval = (double*)malloc(sizeof(double)*stateCount);
    else
        eval = (double*)malloc(sizeof(double)*stateCount*2);
    double* evec = (double*)malloc(sizeof(double)*stateCount*stateCount);
    double* ivec = (double*)malloc(sizeof(double)*stateCount*stateCount);
    
    for (int eigenIndex=0; eigenIndex < eigenCount; eigenIndex++) {
        if (!eigencomplex && ((stateCount & (stateCount-1)) == 0)) {
            
            for (int i=0; i<stateCount; i++) {
                freqs[i] = 1.0 / stateCount;
            }

            // an eigen decomposition for the general state-space JC69 model
            // If stateCount = 2^n is a power-of-two, then Sylvester matrix H_n describes
            // the eigendecomposition of the infinitesimal rate matrix
             
            double* Hn = evec;
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
            
            // Since evec is Hadamard, ivec = (evec)^t / stateCount;    
            for (int i=0; i<stateCount; i++) {
                for (int j=i; j<stateCount; j++) {
                    ivec[i*stateCount+j] = evec[j*stateCount+i] / stateCount;
                    ivec[j*stateCount+i] = ivec[i*stateCount+j]; // Symmetric
                }
            }
           
            eval[0] = 0.0;
            for (int i=1; i<stateCount; i++) {
                eval[i] = -stateCount / (stateCount - 1.0);
            }
       
        } else if (!eigencomplex) {
            for (int i=0; i<stateCount; i++) {
                freqs[i] = gt_rand() / (double) GT_RAND_MAX;
            }
        
            double** qmat=New2DArray<double>(stateCount, stateCount);    
            double* relNucRates = new double[(stateCount * stateCount - stateCount) / 2];
            
            int rnum=0;
            for(int i=0;i<stateCount;i++){
                for(int j=i+1;j<stateCount;j++){
                    relNucRates[rnum] = gt_rand() / (double) GT_RAND_MAX;
                    qmat[i][j]=relNucRates[rnum] * freqs[j];
                    qmat[j][i]=relNucRates[rnum] * freqs[i];
                    rnum++;
                }
            }

            //set diags to sum rows to 0
            double sum;
            for(int x=0;x<stateCount;x++){
                sum=0.0;
                for(int y=0;y<stateCount;y++){
                    if(x!=y) sum+=qmat[x][y];
                        }
                qmat[x][x]=-sum;
            } 
            
            double* eigvalsimag=new double[stateCount];
            double** eigvecs=New2DArray<double>(stateCount, stateCount);//eigenvecs
            double** teigvecs=New2DArray<double>(stateCount, stateCount);//temp eigenvecs
            double** inveigvecs=New2DArray<double>(stateCount, stateCount);//inv eigenvecs    
            int* iwork=new int[stateCount];
            double* work=new double[stateCount];
            
            EigenRealGeneral(stateCount, qmat, eval, eigvalsimag, eigvecs, iwork, work);
            memcpy(*teigvecs, *eigvecs, stateCount*stateCount*sizeof(double));
            InvertMatrix(teigvecs, stateCount, work, iwork, inveigvecs);
            
            for(int x=0;x<stateCount;x++){
                for(int y=0;y<stateCount;y++){
                    evec[x * stateCount + y] = eigvecs[x][y];
                    if (ievectrans)
                        ivec[x * stateCount + y] = inveigvecs[y][x];
                    else
                        ivec[x * stateCount + y] = inveigvecs[x][y];
                }
            } 
            
            Delete2DArray(qmat);
            delete relNucRates;
            
            delete eigvalsimag;
            Delete2DArray(eigvecs);
            Delete2DArray(teigvecs);
            Delete2DArray(inveigvecs);
            delete iwork;
            delete work;
        } else if (eigencomplex && stateCount==4 && eigenCount==1) {
            // create base frequency array
            double temp_freqs[4] = { 0.25, 0.25, 0.25, 0.25 };
            
            // an eigen decomposition for the 4-state 1-step circulant infinitesimal generator
            double temp_evec[4 * 4] = {
                -0.5,  0.6906786606674509,   0.15153543380548623, 0.5,
                0.5, -0.15153543380548576,  0.6906786606674498,  0.5,
                -0.5, -0.6906786606674498,  -0.15153543380548617, 0.5,
                0.5,  0.15153543380548554, -0.6906786606674503,  0.5
            };
            
            double temp_ivec[4 * 4] = {
                -0.5,  0.5, -0.5,  0.5,
                0.6906786606674505, -0.15153543380548617, -0.6906786606674507,   0.15153543380548645,
                0.15153543380548568, 0.6906786606674509,  -0.15153543380548584, -0.6906786606674509,
                0.5,  0.5,  0.5,  0.5
            };
            
            double temp_eval[8] = { -2.0, -1.0, -1.0, 0, 0, 1, -1, 0 };
            
            for(int x=0;x<stateCount;x++){
                freqs[x] = temp_freqs[x];
                eval[x] = temp_eval[x];
                eval[x+stateCount] = temp_eval[x+stateCount];
                for(int y=0;y<stateCount;y++){
                    evec[x * stateCount + y] = temp_evec[x * stateCount + y];
                    if (ievectrans)
                        ivec[x * stateCount + y] = temp_ivec[x + y * stateCount];
                    else
                        ivec[x * stateCount + y] = temp_ivec[x * stateCount + y];
                }
            } 
        } else {
            abort("should not be here");
        }
            
        beagleSetStateFrequencies(instance, eigenIndex, &freqs[0]);
        
        if (!setmatrix) {
            // set the Eigen decomposition
            beagleSetEigenDecomposition(instance, eigenIndex, &evec[0], &ivec[0], &eval[0]);
        }
    }
    
    free(eval);
    free(evec);
    free(ivec);


    
    // a list of indices and edge lengths
	int* edgeIndices = new int[edgeCount*eigenCount];
	int* edgeIndicesD1 = new int[edgeCount*eigenCount];
	int* edgeIndicesD2 = new int[edgeCount*eigenCount];
	for(int i=0; i<edgeCount*eigenCount; i++) {
        edgeIndices[i]=i;
        edgeIndicesD1[i]=(edgeCount*eigenCount)+i;
        edgeIndicesD2[i]=2*(edgeCount*eigenCount)+i;
    }
	double* edgeLengths = new double[edgeCount];
	for(int i=0; i<edgeCount; i++) {
        edgeLengths[i]=gt_rand() / (double) GT_RAND_MAX;
    }
    
    // create a list of partial likelihood update operations
    // the order is [dest, destScaling, source1, matrix1, source2, matrix2]
	int* operations = new int[(internalCount)*BEAGLE_OP_COUNT*eigenCount];
    int* scalingFactorsIndices = new int[(internalCount)*eigenCount]; // internal nodes
	for(int i=0; i<internalCount*eigenCount; i++){
		operations[BEAGLE_OP_COUNT*i+0] = ntaxa+i;
        operations[BEAGLE_OP_COUNT*i+1] = (dynamicScaling ? i : BEAGLE_OP_NONE);
        operations[BEAGLE_OP_COUNT*i+2] = (dynamicScaling ? i : BEAGLE_OP_NONE);
        
        int child1Index;
        if (((i % internalCount)*2) < ntaxa)
            child1Index = (i % internalCount)*2;
        else
            child1Index = i*2 - internalCount * (int)(i / internalCount);
        operations[BEAGLE_OP_COUNT*i+3] = child1Index;
        operations[BEAGLE_OP_COUNT*i+4] = child1Index;

        int child2Index;
        if (((i % internalCount)*2+1) < ntaxa)
            child2Index = (i % internalCount)*2+1;
        else
            child2Index = i*2+1 - internalCount * (int)(i / internalCount);
		operations[BEAGLE_OP_COUNT*i+5] = child2Index;
		operations[BEAGLE_OP_COUNT*i+6] = child2Index;

        scalingFactorsIndices[i] = i;
        
//        printf("i %d dest %d c1 %d c2 %d\n", i, ntaxa+i, child1Index, child2Index);
        
        if (autoScaling)
            scalingFactorsIndices[i] += ntaxa;
	}	

    int* rootIndices = new int[eigenCount];
	int* lastTipIndices = new int[eigenCount];
    int* categoryWeightsIndices = new int[eigenCount];
    int* stateFrequencyIndices = new int[eigenCount];
    int* cumulativeScalingFactorIndices = new int[eigenCount];
    
    for (int eigenIndex=0; eigenIndex < eigenCount; eigenIndex++) {
        rootIndices[eigenIndex] = ntaxa+(internalCount*(eigenIndex+1))-1;//ntaxa*2-2;
        lastTipIndices[eigenIndex] = ntaxa-1;
        categoryWeightsIndices[eigenIndex] = eigenIndex;
        stateFrequencyIndices[eigenIndex] = 0;
        cumulativeScalingFactorIndices[eigenIndex] = ((manualScaling || dynamicScaling) ? (scaleCount*eigenCount-1)-eigenCount+eigenIndex+1 : BEAGLE_OP_NONE);
        
        if (dynamicScaling)
            beagleResetScaleFactors(instance, cumulativeScalingFactorIndices[eigenIndex]);
    }

    // start timing!
	struct timeval time1, time2, time3, time4, time5;
    double bestTimeUpdateTransitionMatrices, bestTimeUpdatePartials, bestTimeAccumulateScaleFactors, bestTimeCalculateRootLogLikelihoods, bestTimeTotal;
    
    double logL = 0.0;
    double deriv1 = 0.0;
    double deriv2 = 0.0;
    
    double previousLogL = 0.0;
    double previousDeriv1 = 0.0;
    double previousDeriv2 = 0.0;

    for (int i=0; i<nreps; i++){
        if (manualScaling && (!(i % rescaleFrequency) || !((i-1) % rescaleFrequency))) {
            for(int j=0; j<internalCount*eigenCount; j++){
                operations[BEAGLE_OP_COUNT*j+1] = (((manualScaling && !(i % rescaleFrequency))) ? j : BEAGLE_OP_NONE);
                operations[BEAGLE_OP_COUNT*j+2] = (((manualScaling && (i % rescaleFrequency))) ? j : BEAGLE_OP_NONE);
            }
        }
        
        gettimeofday(&time1,NULL);

        for (int eigenIndex=0; eigenIndex < eigenCount; eigenIndex++) {
            if (!setmatrix) {
                // tell BEAGLE to populate the transition matrices for the above edge lengths
                beagleUpdateTransitionMatrices(instance,     // instance
                                               eigenIndex,             // eigenIndex
                                               &edgeIndices[eigenIndex*edgeCount],   // probabilityIndices
                                               (calcderivs ? &edgeIndicesD1[eigenIndex*edgeCount] : NULL), // firstDerivativeIndices
                                               (calcderivs ? &edgeIndicesD2[eigenIndex*edgeCount] : NULL), // secondDerivativeIndices
                                               edgeLengths,   // edgeLengths
                                               edgeCount);            // count
            } else {
                double* inMatrix = new double[stateCount*stateCount*rateCategoryCount];
                for (int matrixIndex=0; matrixIndex < edgeCount; matrixIndex++) {
                    for(int z=0;z<rateCategoryCount;z++){
                        for(int x=0;x<stateCount;x++){
                            for(int y=0;y<stateCount;y++){
                                inMatrix[z*stateCount*stateCount + x*stateCount + y] = gt_rand() / (double) GT_RAND_MAX;
                            }
                        } 
                    }
                    beagleSetTransitionMatrix(instance, edgeIndices[eigenIndex*edgeCount + matrixIndex], inMatrix, 1);
                    if (calcderivs) {
                        beagleSetTransitionMatrix(instance, edgeIndicesD1[eigenIndex*edgeCount + matrixIndex], inMatrix, 0);
                        beagleSetTransitionMatrix(instance, edgeIndicesD2[eigenIndex*edgeCount + matrixIndex], inMatrix, 0);
                    }
                }
            }
        }

        gettimeofday(&time2, NULL);
        
        // update the partials
        beagleUpdatePartials( instance,      // instance
                        (BeagleOperation*)operations,     // eigenIndex
                        internalCount*eigenCount,              // operationCount
                        (dynamicScaling ? internalCount : BEAGLE_OP_NONE));             // cumulative scaling index

        gettimeofday(&time3, NULL);

        int scalingFactorsCount = internalCount;
                
        for (int eigenIndex=0; eigenIndex < eigenCount; eigenIndex++) {
            if (manualScaling && !(i % rescaleFrequency)) {
                beagleResetScaleFactors(instance,
                                        cumulativeScalingFactorIndices[eigenIndex]);
                
                beagleAccumulateScaleFactors(instance,
                                       &scalingFactorsIndices[eigenIndex*internalCount],
                                       scalingFactorsCount,
                                       cumulativeScalingFactorIndices[eigenIndex]);
            } else if (autoScaling) {
                beagleAccumulateScaleFactors(instance, &scalingFactorsIndices[eigenIndex*internalCount], scalingFactorsCount, BEAGLE_OP_NONE);
            }
        }
        
        gettimeofday(&time4, NULL);
                
        // calculate the site likelihoods at the root node
        if (!unrooted) {
            beagleCalculateRootLogLikelihoods(instance,               // instance
                                        rootIndices,// bufferIndices
                                        categoryWeightsIndices,                // weights
                                        stateFrequencyIndices,                 // stateFrequencies
                                        cumulativeScalingFactorIndices,
                                        eigenCount,                      // count
                                        &logL);         // outLogLikelihoods
        } else {
            // calculate the site likelihoods at the root node
            beagleCalculateEdgeLogLikelihoods(instance,               // instance
                                              rootIndices,// bufferIndices
                                              lastTipIndices,
                                              lastTipIndices,
                                              (calcderivs ? edgeIndicesD1 : NULL),
                                              (calcderivs ? edgeIndicesD2 : NULL),
                                              categoryWeightsIndices,                // weights
                                              stateFrequencyIndices,                 // stateFrequencies
                                              cumulativeScalingFactorIndices,
                                              eigenCount,                      // count
                                              &logL,    // outLogLikelihood
                                              (calcderivs ? &deriv1 : NULL),
                                              (calcderivs ? &deriv2 : NULL));
        }
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
        
        if (!(logL - logL == 0.0))
            abort("error: invalid lnL");
        
        if (i > 0 && abs(logL - previousLogL) > MAX_DIFF)
            abort("error: large lnL difference between reps");
        
        if (calcderivs) {
            if (!(deriv1 - deriv1 == 0.0) || !(deriv2 - deriv2 == 0.0))
                abort("error: invalid deriv");
            
            if (i > 0 && ((abs(deriv1 - previousDeriv1) > MAX_DIFF) || (abs(deriv2 - previousDeriv2) > MAX_DIFF)) )
                abort("error: large deriv difference between reps");
        }

        previousLogL = logL;
        previousDeriv1 = deriv1;
        previousDeriv2 = deriv2;        
    }

    if (resource == 0) {
        cpuTimeUpdateTransitionMatrices = bestTimeUpdateTransitionMatrices;
        cpuTimeUpdatePartials = bestTimeUpdatePartials;
        cpuTimeAccumulateScaleFactors = bestTimeAccumulateScaleFactors;
        cpuTimeCalculateRootLogLikelihoods = bestTimeCalculateRootLogLikelihoods;
        cpuTimeTotal = bestTimeTotal;
    }
    
    if (!calcderivs)
        fprintf(stdout, "logL = %.5f \n", logL);
    else
        fprintf(stdout, "logL = %.5f d1 = %.5f d2 = %.5f\n", logL, deriv1, deriv2);
    
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
}

void helpMessage() {
	std::cerr << "Usage:\n\n";
	std::cerr << "genomictest [--help] [--states <integer>] [--taxa <integer>] [--sites <integer>] [--rates <integer>] [--manualscale] [--autoscale] [--dynamicscale] [--rsrc <integer>] [--reps <integer>] [--doubleprecision] [--SSE] [--compact-tips] [--seed <integer>] [--rescale-frequency <integer>] [--full-timing] [--unrooted] [--calcderivs] [--logscalers] [--eigencount <integer>] [--eigencomplex] [--ievectrans] [--setmatrix]\n\n";
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
                                    bool* requireDoublePrecision,
                                    bool* requireSSE,
                                    int* compactTipCount,
                                    int* randomSeed,
                                    int* rescaleFrequency,
                                    bool* unrooted,
                                    bool* calcderivs,
                                    bool* logscalers,
                                    int* eigenCount,
                                    bool* eigencomplex,
                                    bool* ievectrans,
                                    bool* setmatrix)	{
    bool expecting_stateCount = false;
	bool expecting_ntaxa = false;
	bool expecting_nsites = false;
	bool expecting_rateCategoryCount = false;
	bool expecting_nreps = false;
	bool expecting_rsrc = false;
	bool expecting_compactTipCount = false;
	bool expecting_seed = false;
    bool expecting_rescaleFrequency = false;
    bool expecting_eigenCount = false;
	
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
        } else if (expecting_compactTipCount) {
            *compactTipCount = (unsigned)atoi(option.c_str());
            expecting_compactTipCount = false;
        } else if (expecting_seed) {
            *randomSeed = (unsigned)atoi(option.c_str());
            expecting_seed = false;
        } else if (expecting_rescaleFrequency) {
            *rescaleFrequency = (unsigned)atoi(option.c_str());
            expecting_rescaleFrequency = false;
        } else if (expecting_eigenCount) {
            *eigenCount = (unsigned)atoi(option.c_str());
            expecting_eigenCount = false;
        } else if (option == "--help") {
			helpMessage();
        } else if (option == "--manualscale") {
            *manualScaling = true;
        } else if (option == "--autoscale") {
        	*autoScaling = true;
        } else if (option == "--dynamicscale") {
        	*dynamicScaling = true;
        } else if (option == "--doubleprecision") {
        	*requireDoublePrecision = true;
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
        } else if (option == "--compact-tips") {
            expecting_compactTipCount = true;
        } else if (option == "--rescale-frequency") {
            expecting_rescaleFrequency = true;
        } else if (option == "--seed") {
            expecting_seed = true;
        } else if (option == "--full-timing") {
            *fullTiming = true;
        } else if (option == "--SSE") {
        	*requireSSE = true;
        } else if (option == "--unrooted") {
        	*unrooted = true;
        } else if (option == "--calcderivs") {
        	*calcderivs = true;
        } else if (option == "--logscalers") {
        	*logscalers = true;
        } else if (option == "--eigencount") {
        	expecting_eigenCount = true;
        } else if (option == "--eigencomplex") {
        	*eigencomplex = true;
        } else if (option == "--ievectrans") {
        	*ievectrans = true;
        } else if (option == "--setmatrix") {
        	*setmatrix = true;
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
    
	if (expecting_seed)
		abort("read last command line option without finding value associated with --seed");
    
	if (expecting_rescaleFrequency)
		abort("read last command line option without finding value associated with --rescale-frequency");

	if (expecting_compactTipCount)
		abort("read last command line option without finding value associated with --compact-tips");

    if (expecting_eigenCount)
		abort("read last command line option without finding value associated with --eigencount");
    
	if (*stateCount < 2)
		abort("invalid number of states supplied on the command line");
        
	if (*ntaxa < 2)
		abort("invalid number of taxa supplied on the command line");
      
	if (*nsites < 1)
		abort("invalid number of sites supplied on the command line");
    
    if (*rateCategoryCount < 1)
        abort("invalid number of rates supplied on the command line");
        
    if (*nreps < 1)
        abort("invalid number of reps supplied on the command line");

    if (*randomSeed < 1)
        abort("invalid number for seed supplied on the command line");   
        
    if (*manualScaling && *rescaleFrequency < 1)
        abort("invalid number for rescale-frequency supplied on the command line");   
    
    if (*compactTipCount < 0 || *compactTipCount > *ntaxa)
        abort("invalid number for compact-tips supplied on the command line");
    
    if (*calcderivs && !(*unrooted))
        abort("calcderivs option requires unrooted tree option");
    
    if (*eigenCount < 1)
        abort("invalid number for eigencount supplied on the command line");
    
    if (*eigencomplex && (*stateCount != 4 || *eigenCount != 1))
        abort("eigencomplex option only works with stateCount=4 and eigenCount=1");
}

int main( int argc, const char* argv[] )
{
    // Default values
    int stateCount = 4;
    int ntaxa = 16;
    int nsites = 10000;
    bool manualScaling = false;
    bool autoScaling = false;
    bool dynamicScaling = false;
    bool requireDoublePrecision = false;
    bool requireSSE = false;
    bool unrooted = false;
    bool calcderivs = false;
    int compactTipCount = 0;
    int randomSeed = 1;
    int rescaleFrequency = 1;
    bool logscalers = false;
    int eigenCount = 1;
    bool eigencomplex = false;
    bool ievectrans = false;
    bool setmatrix = false;

    int rsrc = -1;
    int nreps = 5;
    bool fullTiming = false;
    
    int rateCategoryCount = 4;
    
    interpretCommandLineParameters(argc, argv, &stateCount, &ntaxa, &nsites, &manualScaling, &autoScaling,
                                   &dynamicScaling, &rateCategoryCount, &rsrc, &nreps, &fullTiming,
                                   &requireDoublePrecision, &requireSSE, &compactTipCount, &randomSeed,
                                   &rescaleFrequency, &unrooted, &calcderivs, &logscalers,
                                   &eigenCount, &eigencomplex, &ievectrans, &setmatrix);
    
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
                  requireDoublePrecision,
                  requireSSE,
                  compactTipCount,
                  randomSeed,
                  rescaleFrequency,
                  unrooted,
                  calcderivs,
                  logscalers,
                  eigenCount,
                  eigencomplex,
                  ievectrans,
                  setmatrix);
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
                          requireDoublePrecision,
                          requireSSE,
                          compactTipCount,
                          randomSeed,
                          rescaleFrequency,
                          unrooted,
                          calcderivs,
                          logscalers,
                          eigenCount,
                          eigencomplex,
                          ievectrans,
                          setmatrix);
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
                      requireDoublePrecision,
                      requireSSE,
                      compactTipCount,
                      randomSeed,
                      rescaleFrequency,
                      unrooted,
                      calcderivs,
                      logscalers,
                      eigenCount,
                      eigencomplex,
                      ievectrans,
                      setmatrix);
        }
	}

//#ifdef _WIN32
//    std::cout << "\nPress ENTER to exit...\n";
//    fflush( stdout);
//    fflush( stderr);
//    getchar();
//#endif
}
