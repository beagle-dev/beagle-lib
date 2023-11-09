/*
 *  beagleBenchmark.cpp
 *  Resource/implementation benchmarking
 *
 * Copyright 2009 Phylogenetic Likelihood Working Group
 *
 * This file is part of BEAGLE.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 *
 * @author Daniel Ayres
 * Based on synthetictest.cpp by Daniel Ayres, Aaron Darling.
 */

#include "libhmsbeagle/benchmark/BeagleBenchmark.h"

namespace beagle {
namespace benchmark {

#ifdef _WIN32
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

double getTimeDiff(struct timeval t1,
                   struct timeval t2) {
    return ((double)(t2.tv_sec - t1.tv_sec)*1000.0 + (double)(t2.tv_usec-t1.tv_usec)/1000.0);
}

int gt_rand_r(unsigned int *seed)
{
    *seed = *seed * 1103515245 + 12345;
    return (*seed % ((unsigned int)GT_RAND_MAX + 1));
}

int gt_rand(unsigned int *seed)
{
    return (gt_rand_r(seed));
}

void gt_srand(unsigned int *seed, unsigned int newSeed)
{
    *seed = newSeed;
}


double* getRandomTipPartials( int nsites, int stateCount, unsigned int *seed )
{
    double *partials = (double*) calloc(sizeof(double), nsites * stateCount);
    for( int i=0; i<nsites*stateCount; i+=stateCount )
    {
        int s = gt_rand(seed)%stateCount;
        partials[i+s]=1.0;
    }
    return partials;
}

int* getRandomTipStates( int nsites, int stateCount, unsigned int *seed )
{
    int *states = (int*) calloc(sizeof(int), nsites);
    for( int i=0; i<nsites; i++ )
    {
        int s = gt_rand(seed)%stateCount;
        states[i]=s;
    }
    return states;
}

// void printFlags(long inFlags) {
//     if (inFlags & BEAGLE_FLAG_PROCESSOR_CPU)      fprintf(stdout, " PROCESSOR_CPU");
//     if (inFlags & BEAGLE_FLAG_PROCESSOR_GPU)      fprintf(stdout, " PROCESSOR_GPU");
//     if (inFlags & BEAGLE_FLAG_PROCESSOR_FPGA)     fprintf(stdout, " PROCESSOR_FPGA");
//     if (inFlags & BEAGLE_FLAG_PROCESSOR_CELL)     fprintf(stdout, " PROCESSOR_CELL");
//     if (inFlags & BEAGLE_FLAG_PRECISION_DOUBLE)   fprintf(stdout, " PRECISION_DOUBLE");
//     if (inFlags & BEAGLE_FLAG_PRECISION_SINGLE)   fprintf(stdout, " PRECISION_SINGLE");
//     if (inFlags & BEAGLE_FLAG_COMPUTATION_ASYNCH) fprintf(stdout, " COMPUTATION_ASYNCH");
//     if (inFlags & BEAGLE_FLAG_COMPUTATION_SYNCH)  fprintf(stdout, " COMPUTATION_SYNCH");
//     if (inFlags & BEAGLE_FLAG_EIGEN_REAL)         fprintf(stdout, " EIGEN_REAL");
//     if (inFlags & BEAGLE_FLAG_EIGEN_COMPLEX)      fprintf(stdout, " EIGEN_COMPLEX");
//     if (inFlags & BEAGLE_FLAG_SCALING_MANUAL)     fprintf(stdout, " SCALING_MANUAL");
//     if (inFlags & BEAGLE_FLAG_SCALING_AUTO)       fprintf(stdout, " SCALING_AUTO");
//     if (inFlags & BEAGLE_FLAG_SCALING_ALWAYS)     fprintf(stdout, " SCALING_ALWAYS");
//     if (inFlags & BEAGLE_FLAG_SCALING_DYNAMIC)    fprintf(stdout, " SCALING_DYNAMIC");
//     if (inFlags & BEAGLE_FLAG_SCALERS_RAW)        fprintf(stdout, " SCALERS_RAW");
//     if (inFlags & BEAGLE_FLAG_SCALERS_LOG)        fprintf(stdout, " SCALERS_LOG");
//     if (inFlags & BEAGLE_FLAG_VECTOR_NONE)        fprintf(stdout, " VECTOR_NONE");
//     if (inFlags & BEAGLE_FLAG_VECTOR_SSE)         fprintf(stdout, " VECTOR_SSE");
//     if (inFlags & BEAGLE_FLAG_VECTOR_AVX)         fprintf(stdout, " VECTOR_AVX");
//     if (inFlags & BEAGLE_FLAG_THREADING_NONE)     fprintf(stdout, " THREADING_NONE");
//     if (inFlags & BEAGLE_FLAG_THREADING_OPENMP)   fprintf(stdout, " THREADING_OPENMP");
//     if (inFlags & BEAGLE_FLAG_THREADING_CPP)      fprintf(stdout, " THREADING_CPP");
//     if (inFlags & BEAGLE_FLAG_FRAMEWORK_CPU)      fprintf(stdout, " FRAMEWORK_CPU");
//     if (inFlags & BEAGLE_FLAG_FRAMEWORK_CUDA)     fprintf(stdout, " FRAMEWORK_CUDA");
//     if (inFlags & BEAGLE_FLAG_FRAMEWORK_OPENCL)   fprintf(stdout, " FRAMEWORK_OPENCL");
// }

int benchmarkResource(int resource,
                         int stateCount,
                         int ntaxa,
                         int nsites,
                         bool manualScaling,
                         int rateCategoryCount,
                         int nreps,
                         int compactTipCount,
                         int rescaleFrequency,
                         bool unrooted,
                         bool calcderivs,
                         int eigenCount,
                         int partitionCount,
                         long preferenceFlags,
                         long requirementFlags,
                         int* resourceNumber,
                         char** implName,
                         long* benchedFlags,
                         double* benchmarkResult,
                         bool instOnly) {

    int edgeCount = ntaxa*2-2;
    int internalCount = ntaxa-1;
    int partialCount = ((ntaxa+internalCount)-compactTipCount)*eigenCount;
    int scaleCount = ((manualScaling) ? ntaxa : 0);

    int modelCount = eigenCount * partitionCount;

    BeagleInstanceDetails instDetails;

    // create an instance of the BEAGLE library
    int instance = beagleCreateInstance(
                ntaxa,            /**< Number of tip data elements (input) */
                partialCount, /**< Number of partials buffers to create (input) */
                compactTipCount,    /**< Number of compact state representation buffers to create (input) */
                stateCount,       /**< Number of states in the continuous-time Markov chain (input) */
                nsites,           /**< Number of site patterns to be handled by the instance (input) */
                modelCount,               /**< Number of rate matrix eigen-decomposition buffers to allocate (input) */
                (calcderivs ? (3*edgeCount*modelCount) : edgeCount*modelCount),/**< Number of rate matrix buffers (input) */
                rateCategoryCount,/**< Number of rate categories */
                scaleCount*eigenCount,          /**< scaling buffers */
                &resource,        /**< List of potential resource on which this instance is allowed (input, NULL implies no restriction */
                1,                /**< Length of resourceList list (input) */
                preferenceFlags,         /**< Bit-flags indicating preferred implementation charactertistics, see BeagleFlags (input) */
                requirementFlags,   /**< Bit-flags indicating required implementation characteristics, see BeagleFlags (input) */
                &instDetails);

    if (instance < 0) {
        return BEAGLE_ERROR_NO_IMPLEMENTATION;
    }

    *resourceNumber = instDetails.resourceNumber;
    *benchedFlags = instDetails.flags;
    *implName = instDetails.implName;

    if (instOnly) {
        return BEAGLE_SUCCESS;
    }

    // fprintf(stdout, "Using resource %d:\n", *resourceNumber);
    // fprintf(stdout, "\tRsrc Name : %s\n",instDetails.resourceName);
    // fprintf(stdout, "\tImpl Name : %s\n", instDetails.implName);
    // fprintf(stdout, "\tFlags:");
    // printFlags(instDetails.flags);
    // fprintf(stdout, "\n\n");

    // set the sequences for each tip using partial likelihood arrays
    unsigned int seed;
    gt_srand(&seed, 1);   // fix the random seed...
    for(int i=0; i<ntaxa; i++)
    {
        if (compactTipCount == 0 || (i >= (compactTipCount-1) && i != (ntaxa-1))) {
            double* tmpPartials = getRandomTipPartials(nsites, stateCount, &seed);
            beagleSetTipPartials(instance, i, tmpPartials);
            free(tmpPartials);
        } else {
            int* tmpStates = getRandomTipStates(nsites, stateCount, &seed);
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
        rates[i] = gt_rand(&seed) / (double) GT_RAND_MAX;
    }

    if (partitionCount > 1) {
        for (int i=0; i < partitionCount; i++) {
            beagleSetCategoryRatesWithIndex(instance, i, &rates[0]);
        }
    } else {
        beagleSetCategoryRates(instance, &rates[0]);
    }


    double* patternWeights = (double*) malloc(sizeof(double) * nsites);

    for (int i = 0; i < nsites; i++) {
        patternWeights[i] = gt_rand(&seed) / (double) GT_RAND_MAX;
    }

    beagleSetPatternWeights(instance, patternWeights);

    int* patternPartitions;
    double* partitionLogLs;
    double* partitionD1;
    double* partitionD2;

    if (partitionCount > 1) {
        partitionLogLs = (double*) malloc(sizeof(double) * partitionCount);
        partitionD1 = (double*) malloc(sizeof(double) * partitionCount);
        partitionD2 = (double*) malloc(sizeof(double) * partitionCount);
        patternPartitions = (int*) malloc(sizeof(int) * nsites);
        int partitionSize = nsites/partitionCount;
        for (int i = 0; i < nsites; i++) {
            int sitePartition = i/partitionSize;
            if (sitePartition > partitionCount - 1)
                sitePartition = partitionCount - 1;
            patternPartitions[i] = sitePartition;
        }
    }


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
            weights[i] = gt_rand(&seed) / (double) GT_RAND_MAX;
        }

        beagleSetCategoryWeights(instance, eigenIndex, &weights[0]);
    }

    double* eval;
    eval = (double*)malloc(sizeof(double)*stateCount);
    double* evec = (double*)malloc(sizeof(double)*stateCount*stateCount);
    double* ivec = (double*)malloc(sizeof(double)*stateCount*stateCount);

    for (int eigenIndex=0; eigenIndex < modelCount; eigenIndex++) {
        if ((stateCount & (stateCount-1)) == 0) {

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

        } else {
            for (int i=0; i<stateCount; i++) {
                freqs[i] = gt_rand(&seed) / (double) GT_RAND_MAX;
            }

            double** qmat=New2DArray<double>(stateCount, stateCount);
            double* relNucRates = new double[(stateCount * stateCount - stateCount) / 2];

            int rnum=0;
            for(int i=0;i<stateCount;i++){
                for(int j=i+1;j<stateCount;j++){
                    relNucRates[rnum] = gt_rand(&seed) / (double) GT_RAND_MAX;
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
                    ivec[x * stateCount + y] = inveigvecs[x][y];
                }
            }

            Delete2DArray(qmat);
            delete[] relNucRates;

            delete[] eigvalsimag;
            Delete2DArray(eigvecs);
            Delete2DArray(teigvecs);
            Delete2DArray(inveigvecs);
            delete[] iwork;
            delete[] work;
        }

        beagleSetStateFrequencies(instance, eigenIndex, &freqs[0]);

        // set the Eigen decomposition
        beagleSetEigenDecomposition(instance, eigenIndex, &evec[0], &ivec[0], &eval[0]);
    }

    free(eval);
    free(evec);
    free(ivec);


    // a list of indices and edge lengths
    int* edgeIndices = new int[edgeCount*modelCount];
    int* edgeIndicesD1 = new int[edgeCount*modelCount];
    int* edgeIndicesD2 = new int[edgeCount*modelCount];
    for(int i=0; i<edgeCount*modelCount; i++) {
        edgeIndices[i]=i;
        edgeIndicesD1[i]=(edgeCount*modelCount)+i;
        edgeIndicesD2[i]=2*(edgeCount*modelCount)+i;
    }
    double* edgeLengths = new double[edgeCount*modelCount];
    for(int i=0; i<edgeCount; i++) {
        edgeLengths[i]=gt_rand(&seed) / (double) GT_RAND_MAX;
    }

    // create a list of partial likelihood update operations
    // the order is [dest, destScaling, source1, matrix1, source2, matrix2]
    int operationCount = internalCount*modelCount;
    int beagleOpCount = BEAGLE_OP_COUNT;
    if (partitionCount > 1)
        beagleOpCount = BEAGLE_PARTITION_OP_COUNT;
    int* operations = new int[beagleOpCount*operationCount];
    int unpartOpsCount = internalCount*eigenCount;
    int* scalingFactorsIndices = new int[unpartOpsCount]; // internal nodes


    for(int i=0; i<unpartOpsCount; i++){
        int child1Index;
        if (((i % internalCount)*2) < ntaxa)
            child1Index = (i % internalCount)*2;
        else
            child1Index = i*2 - internalCount * (int)(i / internalCount);
        int child2Index;
        if (((i % internalCount)*2+1) < ntaxa)
            child2Index = (i % internalCount)*2+1;
        else
            child2Index = i*2+1 - internalCount * (int)(i / internalCount);

        for (int j=0; j<partitionCount; j++) {
            int op = partitionCount*i + j;
            operations[op*beagleOpCount+0] = ntaxa+i;
            operations[op*beagleOpCount+1] = BEAGLE_OP_NONE;
            operations[op*beagleOpCount+2] = BEAGLE_OP_NONE;
            operations[op*beagleOpCount+3] = child1Index;
            operations[op*beagleOpCount+4] = child1Index + j*edgeCount;
            operations[op*beagleOpCount+5] = child2Index;
            operations[op*beagleOpCount+6] = child2Index + j*edgeCount;
            if (partitionCount > 1) {
                operations[op*beagleOpCount+7] = j;
                operations[op*beagleOpCount+8] = BEAGLE_OP_NONE;
            }
        }

        scalingFactorsIndices[i] = i;

    }

    int* rootIndices = new int[eigenCount * partitionCount];
    int* lastTipIndices = new int[eigenCount * partitionCount];
    int* lastTipIndicesD1 = new int[eigenCount * partitionCount];
    int* lastTipIndicesD2 = new int[eigenCount * partitionCount];
    int* categoryWeightsIndices = new int[eigenCount * partitionCount];
    int* stateFrequencyIndices = new int[eigenCount * partitionCount];
    int* cumulativeScalingFactorIndices = new int[eigenCount * partitionCount];
    int* partitionIndices = new int[partitionCount];

    for (int eigenIndex=0; eigenIndex < eigenCount; eigenIndex++) {
        int pOffset = partitionCount*eigenIndex;

        for (int partitionIndex=0; partitionIndex < partitionCount; partitionIndex++) {
            if (eigenIndex == 0)
                partitionIndices[partitionIndex] = partitionIndex;
            rootIndices[partitionIndex + pOffset] = ntaxa+(internalCount*(eigenIndex+1))-1;//ntaxa*2-2;
            lastTipIndices[partitionIndex + pOffset] = ntaxa-1;
            lastTipIndicesD1[partitionIndex + pOffset] = (ntaxa-1) + (edgeCount*modelCount);
            lastTipIndicesD2[partitionIndex + pOffset] = (ntaxa-1) + 2*(edgeCount*modelCount);
            categoryWeightsIndices[partitionIndex + pOffset] = eigenIndex;
            stateFrequencyIndices[partitionIndex + pOffset] = 0;
            cumulativeScalingFactorIndices[partitionIndex + pOffset] = ((manualScaling) ? (scaleCount*eigenCount-1)-eigenCount+eigenIndex+1 : BEAGLE_OP_NONE);
        }
    }

    // start timing!
    struct timeval time0, time5;
    double bestTimeTotal;

    double logL = 0.0;
    double deriv1 = 0.0;
    double deriv2 = 0.0;

    double previousLogL = 0.0;
    double previousDeriv1 = 0.0;
    double previousDeriv2 = 0.0;

    int* eigenIndices = new int[edgeCount * modelCount];
    int* categoryRateIndices = new int[edgeCount * modelCount];
    for (int eigenIndex=0; eigenIndex < modelCount; eigenIndex++) {
        for(int j=0; j<edgeCount; j++) {
            eigenIndices[eigenIndex*edgeCount + j] = eigenIndex;
            categoryRateIndices[eigenIndex*edgeCount + j] = eigenIndex;
            edgeLengths[eigenIndex*edgeCount + j] = edgeLengths[j];
        }
    }

    for (int i=0; i<nreps; i++){

        if (manualScaling && (!(i % rescaleFrequency) || !((i-1) % rescaleFrequency))) {
            for(int j=0; j<operationCount; j++){
                int sIndex = j / partitionCount;
                operations[beagleOpCount*j+1] = (((manualScaling && !(i % rescaleFrequency))) ? sIndex : BEAGLE_OP_NONE);
                operations[beagleOpCount*j+2] = (((manualScaling && (i % rescaleFrequency))) ? sIndex : BEAGLE_OP_NONE);
            }
        }

        gettimeofday(&time0,NULL);

        if (partitionCount > 1 && i==0) {
            if (beagleSetPatternPartitions(instance, partitionCount, patternPartitions) != BEAGLE_SUCCESS) {
                return BEAGLE_ERROR_GENERAL;
            }
        }

        if (partitionCount > 1) {
            int totalEdgeCount = edgeCount * modelCount;
            beagleUpdateTransitionMatricesWithMultipleModels(
                                           instance,     // instance
                                           eigenIndices,   // eigenIndex
                                           categoryRateIndices,   // category rate index
                                           edgeIndices,   // probabilityIndices
                                           (calcderivs ? edgeIndicesD1 : NULL), // firstDerivativeIndices
                                           (calcderivs ? edgeIndicesD2 : NULL), // secondDerivativeIndices
                                           edgeLengths,   // edgeLengths
                                           totalEdgeCount);            // count
        } else {
            for (int eigenIndex=0; eigenIndex < modelCount; eigenIndex++) {
                // tell BEAGLE to populate the transition matrices for the above edge lengths
                beagleUpdateTransitionMatrices(instance,     // instance
                                               eigenIndex,             // eigenIndex
                                               &edgeIndices[eigenIndex*edgeCount],   // probabilityIndices
                                               (calcderivs ? &edgeIndicesD1[eigenIndex*edgeCount] : NULL), // firstDerivativeIndices
                                               (calcderivs ? &edgeIndicesD2[eigenIndex*edgeCount] : NULL), // secondDerivativeIndices
                                               edgeLengths,   // edgeLengths
                                               edgeCount);            // count
            }

        }

            // update the partials
            if (partitionCount > 1) {
                beagleUpdatePartialsByPartition( instance,                   // instance
                                (BeagleOperationByPartition*)operations,     // operations
                                internalCount*eigenCount*partitionCount);    // operationCount
            } else {
                beagleUpdatePartials( instance,      // instance
                                (BeagleOperation*)operations,     // operations
                                internalCount*eigenCount,              // operationCount
                                BEAGLE_OP_NONE);             // cumulative scaling index
            }

        int scalingFactorsCount = internalCount;

        for (int eigenIndex=0; eigenIndex < eigenCount; eigenIndex++) {
            if (manualScaling && !(i % rescaleFrequency)) {
                beagleResetScaleFactors(instance,
                                        cumulativeScalingFactorIndices[eigenIndex]);

                beagleAccumulateScaleFactors(instance,
                                       &scalingFactorsIndices[eigenIndex*internalCount],
                                       scalingFactorsCount,
                                       cumulativeScalingFactorIndices[eigenIndex]);
            }
        }

        // calculate the site likelihoods at the root node
        if (!unrooted) {
            if (partitionCount > 1) {
                beagleCalculateRootLogLikelihoodsByPartition(
                                            instance,               // instance
                                            rootIndices,// bufferIndices
                                            categoryWeightsIndices,                // weights
                                            stateFrequencyIndices,                 // stateFrequencies
                                            cumulativeScalingFactorIndices,
                                            partitionIndices,
                                            partitionCount,
                                            eigenCount,                      // count
                                            partitionLogLs,
                                            &logL);         // outLogLikelihoods
            } else {
                beagleCalculateRootLogLikelihoods(instance,               // instance
                                            rootIndices,// bufferIndices
                                            categoryWeightsIndices,                // weights
                                            stateFrequencyIndices,                 // stateFrequencies
                                            cumulativeScalingFactorIndices,
                                            eigenCount,                      // count
                                            &logL);         // outLogLikelihoods
            }
        } else {
            if (partitionCount > 1) {
                beagleCalculateEdgeLogLikelihoodsByPartition(
                                                  instance,
                                                  rootIndices,
                                                  lastTipIndices,
                                                  lastTipIndices,
                                                  (calcderivs ? lastTipIndicesD1 : NULL),
                                                  (calcderivs ? lastTipIndicesD2 : NULL),
                                                  categoryWeightsIndices,
                                                  stateFrequencyIndices,
                                                  cumulativeScalingFactorIndices,
                                                  partitionIndices,
                                                  partitionCount,
                                                  eigenCount,
                                                  partitionLogLs,
                                                  &logL,
                                                  (calcderivs ? partitionD1 : NULL),
                                                  (calcderivs ? &deriv1 : NULL),
                                                  (calcderivs ? partitionD2 : NULL),
                                                  (calcderivs ? &deriv2 : NULL));
            } else {
                beagleCalculateEdgeLogLikelihoods(instance,               // instance
                                                  rootIndices,// bufferIndices
                                                  lastTipIndices,
                                                  lastTipIndices,
                                                  (calcderivs ? lastTipIndicesD1 : NULL),
                                                  (calcderivs ? lastTipIndicesD2 : NULL),
                                                  categoryWeightsIndices,                // weights
                                                  stateFrequencyIndices,                 // stateFrequencies
                                                  cumulativeScalingFactorIndices,
                                                  eigenCount,                      // count
                                                  &logL,    // outLogLikelihood
                                                  (calcderivs ? &deriv1 : NULL),
                                                  (calcderivs ? &deriv2 : NULL));
            }

        }
        // end timing!
        gettimeofday(&time5,NULL);

        if (i == 0 || getTimeDiff(time0, time5) < bestTimeTotal) {
            bestTimeTotal = getTimeDiff(time0, time5);
        }
    }

    // fprintf(stdout, "logL = %.5f \n", logL);
    // fprintf(stdout, "time = %.5f \n\n", getTimeDiff(time0, time5));

    beagleFinalizeInstance(instance);

    *benchmarkResult = bestTimeTotal;

    return BEAGLE_SUCCESS;

}


}   // namespace benchmark
}   // namespace beagle

