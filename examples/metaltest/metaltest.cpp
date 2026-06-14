/*
 * metaltest.cpp
 *
 * Exercises the BEAGLE Metal GPU backend on macOS.
 * Computes the tree log-likelihood for a 3-taxon, 5-site alignment
 * using an HKY85 model with single precision on an Apple GPU.
 *
 * Usage:
 *   ./metaltest            # requires Metal backend
 *   ./metaltest --cpu      # fall back to CPU (for reference values)
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>

#include "libhmsbeagle/beagle.h"

static char* human   = (char*)"GAGTC";
static char* chimp   = (char*)"GAGGC";
static char* gorilla = (char*)"AAAT-";

static int* getStates(char* seq) {
    int n = (int)strlen(seq);
    int* s = (int*)malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) {
        switch (seq[i]) {
            case 'A': s[i] = 0; break;
            case 'C': s[i] = 1; break;
            case 'G': s[i] = 2; break;
            case 'T': s[i] = 3; break;
            default:  s[i] = 4; break;
        }
    }
    return s;
}

static void printFlags(long f) {
    if (f & BEAGLE_FLAG_PROCESSOR_CPU)      printf(" PROCESSOR_CPU");
    if (f & BEAGLE_FLAG_PROCESSOR_GPU)      printf(" PROCESSOR_GPU");
    if (f & BEAGLE_FLAG_PRECISION_DOUBLE)   printf(" PRECISION_DOUBLE");
    if (f & BEAGLE_FLAG_PRECISION_SINGLE)   printf(" PRECISION_SINGLE");
    if (f & BEAGLE_FLAG_FRAMEWORK_CPU)      printf(" FRAMEWORK_CPU");
    if (f & BEAGLE_FLAG_FRAMEWORK_CUDA)     printf(" FRAMEWORK_CUDA");
    if (f & BEAGLE_FLAG_FRAMEWORK_OPENCL)   printf(" FRAMEWORK_OPENCL");
    if (f & BEAGLE_FLAG_FRAMEWORK_METAL)    printf(" FRAMEWORK_METAL");
}

int main(int argc, const char* argv[])
{
    bool useCPU = (argc > 1 && strcmp(argv[1], "--cpu") == 0);

    BeagleResourceList* rList = beagleGetResourceList();
    printf("Available resources:\n");
    for (int i = 0; i < rList->length; i++) {
        printf("\t%d: %s\n", i, rList->list[i].name);
        printf("\t   Flags:");
        printFlags(rList->list[i].supportFlags);
        printf("\n");
    }
    printf("\n");

    const int stateCount   = 4;
    const int nPatterns    = (int)strlen(human);  // 5
    const int rateCatCount = 1;
    // Tree: ((human:0.1, chimp:0.1):0.1, gorilla:0.2)
    // Buffers: tips 0,1,2; internal 3; root 4.  Matrices: 0=human,1=chimp,2=gorilla,3=stem.
    const int nMatrices    = 4;
    const int nPartials    = 5;

    long prefFlags, reqFlags;
    if (useCPU) {
        prefFlags = BEAGLE_FLAG_PROCESSOR_CPU | BEAGLE_FLAG_PRECISION_DOUBLE;
        reqFlags  = 0;
        printf("Mode: CPU (double precision, for reference)\n\n");
    } else {
        prefFlags = BEAGLE_FLAG_PROCESSOR_GPU |
                    BEAGLE_FLAG_PRECISION_SINGLE |
                    BEAGLE_FLAG_FRAMEWORK_METAL;
        reqFlags  = BEAGLE_FLAG_FRAMEWORK_METAL | BEAGLE_FLAG_PRECISION_SINGLE;
        printf("Mode: Metal GPU (single precision)\n\n");
    }

    BeagleInstanceDetails instDetails;
    int instance = beagleCreateInstance(
        3,             // tipCount
        nPartials,     // partialsBufferCount
        3,             // compactBufferCount (tip states)
        stateCount,
        nPatterns,
        1,             // eigenBufferCount
        nMatrices,     // matrixBufferCount
        rateCatCount,
        0,             // scaleBufferCount
        NULL, 0,
        prefFlags, reqFlags,
        &instDetails);

    if (instance < 0) {
        fprintf(stderr, "Failed to create BEAGLE instance (error %d)\n", instance);
        if (!useCPU)
            fprintf(stderr, "Hint: is hmsbeagle-metal loaded? Try running with --cpu for a CPU reference.\n");
        return 1;
    }

    printf("Resource %d: %s\n", instDetails.resourceNumber, instDetails.resourceName);
    printf("Impl:       %s\n", instDetails.implName);
    printf("\n");

    // Tip states
    int* hs = getStates(human);
    int* cs = getStates(chimp);
    int* gs = getStates(gorilla);
    beagleSetTipStates(instance, 0, hs);
    beagleSetTipStates(instance, 1, cs);
    beagleSetTipStates(instance, 2, gs);
    free(hs); free(cs); free(gs);

    // Rates, weights, frequencies
    double rates[1]   = { 1.0 };
    double weights[1] = { 1.0 };
    double freqs[4]   = { 0.1, 0.3, 0.2, 0.4 };
    double* patWts = (double*)malloc(sizeof(double) * nPatterns);
    for (int i = 0; i < nPatterns; i++) patWts[i] = 1.0;

    beagleSetCategoryRates(instance, rates);
    beagleSetCategoryWeights(instance, 0, weights);
    beagleSetStateFrequencies(instance, 0, freqs);
    beagleSetPatternWeights(instance, patWts);
    free(patWts);

    // HKY85 eigen decomposition (same as hmctest)
    double evec[16] = {
         0.9819805,   0.040022305,  0.04454354, -0.5,
        -0.1091089,  -0.002488732,  0.81606029, -0.5,
        -0.1091089,  -0.896939683, -0.11849713, -0.5,
        -0.1091089,   0.440330814, -0.56393254, -0.5
    };
    double ivec[16] = {
         0.9165151, -0.3533241, -0.1573578, -0.4058332,
         0.0,        0.2702596, -0.8372848,  0.5670252,
         0.0,        0.8113638, -0.2686725, -0.5426913,
        -0.2,       -0.6,       -0.4,       -0.8
    };
    double eval[8] = {
        -1.42857105618099456, -1.42857095607719153, -1.42857087221423851, 0.0,
         0.0, 0.0, 0.0, 0.0
    };
    beagleSetEigenDecomposition(instance, 0, evec, ivec, eval);

    // Transition matrices: 4 branches
    int    nodeIdx[4]  = { 0, 1, 2, 3 };
    double edgeLen[4]  = { 0.1, 0.1, 0.2, 0.1 };
    beagleUpdateTransitionMatrices(instance, 0, nodeIdx, NULL, NULL, edgeLen, 4);

    // Partial likelihood operations:
    //   op0: buf3 = partial(tip0, mat0) x partial(tip1, mat1)   [human+chimp clade]
    //   op1: buf4 = partial(tip2, mat2) x partial(buf3, mat3)   [root]
    BeagleOperation ops[2] = {
        { 3, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 0, 0, 1, 1 },
        { 4, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 2, 2, 3, 3 }
    };
    beagleUpdatePartials(instance, ops, 2, BEAGLE_OP_NONE);

    // Root log-likelihood
    int    rootIdx      = 4;
    int    catWtIdx     = 0;
    int    freqIdx      = 0;
    int    scaleIdx     = BEAGLE_OP_NONE;
    double logL         = 0.0;
    beagleCalculateRootLogLikelihoods(instance,
                                      &rootIdx, &catWtIdx, &freqIdx,
                                      &scaleIdx, 1, &logL);

    printf("log-likelihood = %.8f\n", logL);
    if (!useCPU)
        printf("(expected approx. -19.38 for this 3-taxon HKY85 example)\n");

    // Per-site log-likelihoods
    double siteLikelihoods[5];
    beagleGetSiteLogLikelihoods(instance, siteLikelihoods);
    printf("site log-likelihoods:");
    for (int i = 0; i < nPatterns; i++)
        printf("  %.6f", siteLikelihoods[i]);
    printf("\n");

    beagleFinalizeInstance(instance);
    return 0;
}
