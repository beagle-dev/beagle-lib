//
// Created by Xiang Ji on 2/28/18.
//

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
#include <iostream>
#include <libhmsbeagle/BeagleImpl.h>

//#define JC

#ifdef _WIN32
#include <vector>
#endif

#include "libhmsbeagle/beagle.h"

char *D4Brazi82 = (char*) "AACCCCCCGGGGGTTTTTTTT";
char *D4ElSal83 = (char*) "AACCCCTTGGGGGCTTTTTTT";
char *D4ElSal94 = (char*) "AACCGTCTAGGGGCCCTTTTT";
char *D4Indon76 = (char*) "AGCTCCCTGAGGGCCTACCTT";
char *D4Indon77 = (char*) "AACTCCCTGAACGCCTACTCT";


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

//    bool scaling = true; disable scaling for now
    bool scaling = false;

    bool doJC = true;

    // is nucleotides...
    int stateCount = 4;

    // get the number of site patterns
    int nPatterns = strlen(D4Brazi82);

    // change # rate category to 2
//    int rateCategoryCount = 4;
    int rateCategoryCount = 2;
//    int rateCategoryCount = 1;

    int scaleCount = (scaling ? 4 : 0);

    BeagleInstanceDetails instDetails;

    /// Doubled the size of partials buffer from 5 to 10

    // create an instance of the BEAGLE library
    int instance = beagleCreateInstance(
            5,				/**< Number of tip data elements (input) */
            30,	            /**< Number of partials buffers to create (input) */
            0,		        /**< Number of compact state representation buffers to create (input) */
            stateCount,		/**< Number of states in the continuous-time Markov chain (input) */
            nPatterns,		/**< Number of site patterns to be handled by the instance (input) */
            1,		        /**< Number of rate matrix eigen-decomposition buffers to allocate (input) */
            8,		        /**< Number of rate matrix buffers (input) */
            rateCategoryCount,/**< Number of rate categories (input) */
            10,       /**< Number of scaling buffers */
            NULL,			    /**< List of potential resource on which this instance is allowed (input, NULL implies no restriction */
            0,			    /**< Length of resourceList list (input) */
            BEAGLE_FLAG_PROCESSOR_CPU,             	/**< Bit-flags indicating preferred implementation charactertistics, see BeagleFlags (input) */
            BEAGLE_FLAG_EIGEN_REAL,                 /**< Bit-flags indicating required implementation characteristics, see BeagleFlags (input) */
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
    fprintf(stdout, "\n");


//    beagleSetTipStates(instance, 0, getStates(human));
//    beagleSetTipStates(instance, 1, getStates(chimp));
//    beagleSetTipStates(instance, 2, getStates(gorilla));

    // set the sequences for each tip using partial likelihood arrays
    double *D4Brazi82Partials   = getPartials(D4Brazi82);
    double *D4ElSal83Partials   = getPartials(D4ElSal83);
    double *D4ElSal94Partials   = getPartials(D4ElSal94);
    double *D4Indon76Partials   = getPartials(D4Indon76);
    double *D4Indon77Partials   = getPartials(D4Indon77);

    beagleSetTipPartials(instance, 0, D4ElSal94Partials);
    beagleSetTipPartials(instance, 1, D4Indon76Partials);
    beagleSetTipPartials(instance, 2, D4Brazi82Partials);
    beagleSetTipPartials(instance, 3, D4ElSal83Partials);
    beagleSetTipPartials(instance, 4, D4Indon77Partials);

#ifdef _WIN32
    std::vector<double> rates(rateCategoryCount);
#else
    double rates[rateCategoryCount];
#endif
    rates[0] = 0.14251623900062188;
    rates[1] = 1.85748376099937812;
//    rates[0] = 1.0;
//    for (int i = 0; i < rateCategoryCount; i++) {
////        rates[i] = 1.0;
//        rates[i] = 3.0 * (i + 1) / (2 * rateCategoryCount + 1);
//    }

    beagleSetCategoryRates(instance, &rates[0]);

    double patternWeights[21] = {454,2,267,3,1,4,1,1,3,3,1,1,397,1,1,6,1,2,4,2,330};

//    for (int i = 0; i < nPatterns; i++) {
//        patternWeights[i] = 1.0;
//    }

    beagleSetPatternWeights(instance, patternWeights);

    // create base frequency array
    double freqs[4] = { 0.1, 0.3, 0.2, 0.4 };
//    double freqs[4] = { 0.25, 0.25, 0.25, 0.25 };

    beagleSetStateFrequencies(instance, 0, freqs);

    // create an array containing site category weights
#ifdef _WIN32
    std::vector<double> weights(rateCategoryCount);
#else
    double weights[rateCategoryCount];
#endif
    for (int i = 0; i < rateCategoryCount; i++) {
        weights[i] = 1.0/rateCategoryCount;
//        weights[i] = 2.0 * double(i + 1)/ double(rateCategoryCount * (rateCategoryCount + 1));
    }

    beagleSetCategoryWeights(instance, 0, &weights[0]);

//#ifndef JC
//	// an eigen decomposition for the 4-state 1-step circulant infinitesimal generator
//	double evec[4 * 4] = {
//			 -0.5,  0.6906786606674509,   0.15153543380548623, 0.5,
//			  0.5, -0.15153543380548576,  0.6906786606674498,  0.5,
//			 -0.5, -0.6906786606674498,  -0.15153543380548617, 0.5,
//			  0.5,  0.15153543380548554, -0.6906786606674503,  0.5
//	};
//
//	double ivec[4 * 4] = {
//			 -0.5,  0.5, -0.5,  0.5,
//			  0.6906786606674505, -0.15153543380548617, -0.6906786606674507,   0.15153543380548645,
//			  0.15153543380548568, 0.6906786606674509,  -0.15153543380548584, -0.6906786606674509,
//			  0.5,  0.5,  0.5,  0.5
//	};
//
//	double eval[8] = { -2.0, -1.0, -1.0, 0, 0, 1, -1, 0 };
//#else
//	// an eigen decomposition for the JC69 model
//	double evec[4 * 4] = {
//        1.0,  2.0,  0.0,  0.5,
//        1.0,  -2.0,  0.5,  0.0,
//        1.0,  2.0, 0.0,  -0.5,
//        1.0,  -2.0,  -0.5,  0.0
//	};
//
//	double ivec[4 * 4] = {
//        0.25,  0.25,  0.25,  0.25,
//        0.125,  -0.125,  0.125,  -0.125,
//        0.0,  1.0,  0.0,  -1.0,
//        1.0,  0.0,  -1.0,  0.0
//	};
//
//	double eval[8] = { 0.0, -1.3333333333333333, -1.3333333333333333, -1.3333333333333333,
//			0.0, 0.0, 0.0, 0.0 };
//#endif

    ///eigen decomposition of the HKY85 model
    double evec[4 * 4] = {
            0.9819805,  0.040022305,  0.04454354,  -0.5,
            -0.1091089, -0.002488732, 0.81606029,  -0.5,
            -0.1091089, -0.896939683, -0.11849713, -0.5,
            -0.1091089,  0.440330814, -0.56393254, -0.5
    };

    double ivec[4 * 4] = {
            0.9165151, -0.3533241, -0.1573578, -0.4058332,
            0.0,  0.2702596, -0.8372848,  0.5670252,
            0.0,  0.8113638, -0.2686725, -0.5426913,
            -0.2, -0.6, -0.4, -0.8
    };

    ///array of real parts + array of imaginary parts
    double eval[8] = { -1.42857105618099456, -1.42857095607719153, -1.42857087221423851, 0.0,
                       0.0, 0.0, 0.0, 0.0 };

    ///Q^T matrix
    double QT[4 * 4] = {
            -1.2857138,  0.1428570,  0.1428570,  0.1428570,
            0.4285712, -0.9999997,  0.4285714,  0.4285713,
            0.2857142,  0.2857143, -1.1428568,  0.2857142,
            0.5714284,  0.5714284,  0.5714284, -0.8571426
    };

    // set the Eigen decomposition
    beagleSetEigenDecomposition(instance, 0, evec, ivec, eval);

    // a list of indices and edge lengths
    int nodeIndices[8] = { 0, 1, 2, 3, 4, 5, 6, 7};
    double edgeLengths[8] = { 25.81403421468474, 7.814034214684739, 36.80326223293307, 37.80326223293307,
                              282.8618556834007, 244.56614874230695, 221.57692072405862, 29.481672726408988};
//    double edgeLengths[8] = { 0.81403421468474, 0.814034214684739, 0.80326223293307, 0.80326223293307,
//                              0.8618556834007, 0.56614874230695, 0.57692072405862, 0.481672726408988};

    // tell BEAGLE to populate the transition matrices for the above edge lengths
    beagleUpdateTransitionMatrices(instance,     // instance
                                   0,             // eigenIndex
                                   nodeIndices,   // probabilityIndices
                                   NULL,          // firstDerivativeIndices
                                   NULL,          // secondDervativeIndices
                                   edgeLengths,   // edgeLengths
                                   8);            // count

    // create a list of partial likelihood update operations
    // the order is [dest, destScaling, source1, matrix1, source2, matrix2]
    BeagleOperation operations[4] = {
             9, (scaling ? 0 : BEAGLE_OP_NONE), BEAGLE_OP_NONE, 0, 0, 1, 1,
            10, (scaling ? 1 : BEAGLE_OP_NONE), BEAGLE_OP_NONE, 2, 2, 3, 3,
            11, (scaling ? 2 : BEAGLE_OP_NONE), BEAGLE_OP_NONE, 9, 5, 10, 6,
            12, (scaling ? 3 : BEAGLE_OP_NONE), BEAGLE_OP_NONE, 11, 7, 4, 4
    };

    int rootIndex = 12;

    // update the partials
    beagleUpdatePartials(instance,      // instance
                         operations,     // operations
                         4,              // operationCount
                         BEAGLE_OP_NONE);          // cumulative scaling index

    int categoryWeightsIndex = 0;
    int stateFrequencyIndex = 0;
    // create a list of partial likelihood update operations
    // the order is [dest, destScaling, source1, matrix1, source2, matrix2]
    // destPartials point to the pre-order partials
    // partials1 = pre-order partials of the parent node
    // matrices1 = Ptr matrices of the current node (to the parent node)
    // partials2 = post-order partials of the sibling node
    // matrices2 = Ptr matrices of the sibling node (to the parent node)
    BeagleOperation pre_order_operations[8] = {
            29, (scaling ? 1 : BEAGLE_OP_NONE), BEAGLE_OP_NONE, 13, 7, 4, 4,
            27, (scaling ? 1 : BEAGLE_OP_NONE), BEAGLE_OP_NONE, 29, 5, 10, 6,
            22, (scaling ? 1 : BEAGLE_OP_NONE), BEAGLE_OP_NONE, 27, 0, 1, 1,
            23, (scaling ? 1 : BEAGLE_OP_NONE), BEAGLE_OP_NONE, 27, 1, 0, 0,
            28, (scaling ? 1 : BEAGLE_OP_NONE), BEAGLE_OP_NONE, 29, 6, 9, 5,
            24, (scaling ? 1 : BEAGLE_OP_NONE), BEAGLE_OP_NONE, 28, 2, 3, 3,
            25, (scaling ? 1 : BEAGLE_OP_NONE), BEAGLE_OP_NONE, 28, 3, 2, 2,
            26, (scaling ? 1 : BEAGLE_OP_NONE), BEAGLE_OP_NONE, 13, 4, 11, 7
    };

    int rootPreIndex = 13;

    beagleSetRootPrePartials(instance,
                             (const int *) &rootPreIndex,               // bufferIndices
                             &stateFrequencyIndex,                  // stateFrequencies
                             1);                                    // count

    // update the pre-order partials
    beagleUpdatePrePartials(instance,
                            pre_order_operations,
                            8,
                            BEAGLE_OP_NONE);

//  print pre-order partials and edge length log-likelihood gradient to screen
//  TODO: implement gradient calculation according to beagleCalculateEdgeLogLikelihoods() in beagle.cpp
//  need to consider rate variation case


    double * seeprePartials = (double*) malloc(sizeof(double) * stateCount * nPatterns * rateCategoryCount);
    double * seepreParentPartials = (double*) malloc(sizeof(double) * stateCount * nPatterns * rateCategoryCount);
    double * seepostPartials = (double*) malloc(sizeof(double) * stateCount * nPatterns *rateCategoryCount);
    double * seerootPartials = (double*) malloc(sizeof(double) * stateCount * nPatterns * rateCategoryCount);
    double * seePtrSelf = (double*) malloc(sizeof(double) * stateCount * stateCount * rateCategoryCount);
    double * seePtrSibling = (double*) malloc(sizeof(double) * stateCount * stateCount * rateCategoryCount);

    double * tmpNumerator = (double*)   malloc(sizeof(double)  * nPatterns * rateCategoryCount);

    double * grand_denominator = (double*) malloc(sizeof(double)  * nPatterns);
    double * grand_numerator = (double*) malloc(sizeof(double)  * nPatterns);
    /// state frequencies stored in freqs
    /// category weights stored in weights


    beagleGetPartials(instance, rootIndex, BEAGLE_OP_NONE, seerootPartials);
    for(int i = 0; i < 8; i++){
        for(int m = 0; m < nPatterns; m++){
            grand_denominator[m] = 0;
            grand_numerator[m] = 0;
        }
        int postBufferIndices = i > 4 ? i + 4 : i;
        int preBufferIndices = i == 8 ? 13 : i + 22;
        beagleGetPartials(instance, preBufferIndices, BEAGLE_OP_NONE, seeprePartials);
        beagleGetPartials(instance, 13, BEAGLE_OP_NONE, seepreParentPartials);
        beagleGetPartials(instance, postBufferIndices, BEAGLE_OP_NONE, seepostPartials);

        beagleGetTransitionMatrix(instance, i, seePtrSelf);
        beagleGetTransitionMatrix(instance, 4, seePtrSibling);

        //Show Ptr matrix
//        std::cout<<"Show Ptr matrix of node: "<<i<<"\n";
//        for(int j = 0; j < rateCategoryCount; j++){
//            for(int i = 0; i < stateCount * stateCount ; i++){
//                std::cout<<seePtrSelf[j * stateCount * stateCount + i]<<"  ";
//            }
//            std::cout<<std::endl;
//        }
//        std::cout<<"\nShow Ptr matrix of sibling node: 4\n";
//        for(int j = 0; j < rateCategoryCount; j++){
//            for(int i = 0; i < stateCount * stateCount ; i++){
//                std::cout<<seePtrSibling[j * stateCount * stateCount + i]<<"  ";
//            }
//            std::cout<<std::endl;
//        }
//
//        std::cout<<"\nPre Partial for parent node: ";
//        for(int j = 0; j < rateCategoryCount; j++){
//            for(int i = 0; i < nPatterns * stateCount; i++){
//                std::cout<<seepreParentPartials[j * nPatterns * stateCount + i]<<"  ";
//            }
//            std::cout<<std::endl;
//        }
//
//        std::cout<<"\nPre Partial for current node: \n";
//        for(int j = 0; j < rateCategoryCount; j++){
//            for(int i = 0; i < nPatterns * stateCount; i++){
//                std::cout<<seeprePartials[j * nPatterns * stateCount + i]<<"  ";
//            }
//            std::cout<<std::endl;
//        }
//
//        std::cout<<"\nPost Partial for current node: \n";
//        for(int j = 0; j < rateCategoryCount; j++){
//            for(int i = 0; i < nPatterns * stateCount; i++){
//                std::cout<<seepostPartials[j * nPatterns * stateCount + i]<<"  ";
//            }
//            std::cout<<std::endl;
//        }

        double * prePartialsPtr = seeprePartials;
        double * postPartialsPtr = seepostPartials;

        double denominator = 0;
        double numerator = 0;

        double tmp = 0;
        int k, j, l, m, s, t;
        std::cout<<"Gradient for branch (of node) "<< i <<": \n";

        ///get likelihood for each rate category first
        double clikelihood[rateCategoryCount * nPatterns];
        l = 0; j = 0;
        for(s = 0; s < rateCategoryCount; s++){
            for(m = 0; m < nPatterns; m++){
                double clikelihood_tmp = 0.0;
                for(k=0; k < stateCount; k++){
                    clikelihood_tmp += freqs[k] * seerootPartials[l++];
                }
                clikelihood[j++] = clikelihood_tmp;
            }
        }

        ///now calculate weights
        t = 0;
        for(s = 0; s < rateCategoryCount; s++){
            double ws = weights[s];
            double rs = rates[s];
            for(m=0; m < nPatterns; m++){
                l = 0;
                numerator = 0;
                denominator = 0;
                for(k = 0; k < stateCount; k++){
                    tmp = 0.0;
                    for(j=0; j < stateCount; j++){
                        tmp += QT[l++]*prePartialsPtr[j];
                    }
                    numerator += tmp * postPartialsPtr[k];
                    denominator += postPartialsPtr[k] * prePartialsPtr[k];
                }
                postPartialsPtr += stateCount;
                prePartialsPtr  += stateCount;
                tmpNumerator[t] = ws * rs * numerator / denominator * clikelihood[t];
                //std::cout<< tmpNumerator[t]<<",  "<<ws*clikelihood[t]<<"  \n";
                grand_numerator[m] += tmpNumerator[t];
                grand_denominator[m] += ws * clikelihood[t];
                t++;
                std::cout<<numerator / denominator <<"  ";
            }
            std::cout<<std::endl;
        }

        double gradient = 0.0;
        for(m=0; m < nPatterns; m++){
            gradient += grand_numerator[m] / grand_denominator[m] * patternWeights[m];
            std::cout<<grand_numerator[m] / grand_denominator[m]<< "  ";
        }

        std::cout<< std::endl <<gradient * edgeLengths[i] <<std::endl;
//        for(m=0; m < nPatterns; m++){
//            l = 0;
//            numerator = 0;
//            denominator = 0;
//            for(k = 0; k < stateCount; k++){
//                tmp = 0.0;
//                for(j=0; j < stateCount; j++){
//                    tmp += QT[l++]*prePartialsPtr[j];
//                }
//                numerator += tmp * postPartialsPtr[k];
//                denominator += postPartialsPtr[k] * prePartialsPtr[k];
//            }
//            postPartialsPtr += stateCount;
//            prePartialsPtr  += stateCount;
//            std::cout<<numerator / denominator <<"  ";
//        }
//        std::cout<<std::endl;

//        std::cout<<"Pre-order Partial for node "<< 4-i << ": \n";
//
//        l = 0;
//        for(s = 0; s < rateCategoryCount; s++){
//            std::cout<<"  rate category"<< s+1<< ": \n";
//            for(k = 0; k<100; k++){
//                for(j=0; j < stateCount; j++){
//                    std::cout<< k * stateCount + j<< "   " <<seeprePartials[l++]<<", \n";
//                }
//                //std::cout<<std::endl;
//            }
//            std::cout<<std::endl;
//        }

    }

    double *patternLogLik = (double*)malloc(sizeof(double) * nPatterns);

    int cumulativeScalingIndex = (scaling ? 2 : BEAGLE_OP_NONE);

    if (scaling) {
        int scalingFactorsCount = 2;
        int scalingFactorsIndices[2] = {0, 1};

        beagleResetScaleFactors(instance,
                                cumulativeScalingIndex);

        beagleAccumulateScaleFactors(instance,
                                     scalingFactorsIndices,
                                     scalingFactorsCount,
                                     cumulativeScalingIndex);
    }




    double logL = 0.0;

    // calculate the site likelihoods at the root node
    beagleCalculateRootLogLikelihoods(instance,               // instance
                                      (const int *)&rootIndex,// bufferIndices
                                      &categoryWeightsIndex,                // weights
                                      &stateFrequencyIndex,                  // stateFrequencies
                                      &cumulativeScalingIndex,// cumulative scaling index
                                      1,                      // count
                                      &logL);         // outLogLikelihoods

#ifndef JC
    fprintf(stdout, "logL = %.5f (R = -11773.29178724702)\n\n", logL);
#else
    fprintf(stdout, "logL = %.5f (PAUP = -1574.63624)\n\n", logL);
#endif

//    free(patternWeights);

    free(patternLogLik);
    free(D4Brazi82Partials);
    free(D4ElSal83Partials);
    free(D4ElSal94Partials);
    free(D4Indon76Partials);
    free(D4Indon77Partials);

    beagleFinalizeInstance(instance);

#ifdef _WIN32
    std::cout << "\nPress ENTER to exit...\n";
    fflush( stdout);
    fflush( stderr);
    getchar();
#endif

}
