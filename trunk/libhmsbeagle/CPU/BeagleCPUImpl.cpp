/*
 *  BeagleCPUImpl.cpp
 *  BEAGLE
 *
 * Copyright 2009 Phylogenetic Likelihood Working Group
 *
 * This file is part of BEAGLE.
 *
 * BEAGLE is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 *
 * BEAGLE is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with BEAGLE.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @author Andrew Rambaut
 * @author Marc Suchard
 * @author Daniel Ayres
 * @author Mark Holder
 */

///@TODO: wrap partials, eigen calcs, and transition matrices in a small structs
//      so that we can flag them. This would this would be helpful for
//      implementing:
//          1. an error-checking version that double-checks (to the extent
//              possible) that the client is using the API correctly.  This would
//              ideally be a  conditional compilation variant (so that we do
//              not normally incur runtime penalties, but can enable it to help
//              find bugs).
//          2. a multithreading impl that checks dependencies before queuing
//              partials.

///@API-ISSUE: adding an resizePartialsBufferArray(int newPartialsBufferCount) method
//      would be trivial for this impl, and would be easier for clients that want
//      to cache partial like calculations for a indeterminate number of trees.
///@API-ISSUE: adding a
//  void waitForPartials(int* instance;
//                  int instanceCount;
//                  int* parentPartialIndex;
//                  int partialCount;
//                  );
//  method that blocks until the partials are valid would be important for
//  clients (such as GARLI) that deal with big trees by overwriting some temporaries.
///@API-ISSUE: Swapping temporaries (we decided not to implement the following idea
//  but MTH did want to record it for posterity). We could add following
//  calls:
////////////////////////////////////////////////////////////////////////////////
// BeagleReturnCodes swapEigens(int instance, int *firstInd, int *secondInd, int count);
// BeagleReturnCodes swapTransitionMatrices(int instance, int *firstInd, int *secondInd, int count);
// BeagleReturnCodes swapPartials(int instance, int *firstInd, int *secondInd, int count);
////////////////////////////////////////////////////////////////////////////////
//  They would be optional for the client but could improve efficiency if:
//      1. The impl is load balancing, AND
//      2. The client code, uses the calls to synchronize the indexing of temporaries
//          between instances such that you can pass an instanceIndices list with
//          multiple entries to updatePartials.
//  These seem too nitty gritty and low-level, but they also make it easy to
//      write a wrapper geared toward MCMC (during a move, cache the old data
//      in an unused array, after a rejection swap back to the cached copy)

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cstring>
#include <cmath>
#include <cassert>
#include <vector>

#include "libhmsbeagle/beagle.h"
#include "libhmsbeagle/CPU/BeagleCPUImpl.h"
#include "libhmsbeagle/CPU/EigenDecompositionCube.h"
#include "libhmsbeagle/CPU/EigenDecompositionSquare.h"

using namespace beagle;
using namespace beagle::cpu;

#if defined (BEAGLE_IMPL_DEBUGGING_OUTPUT) && BEAGLE_IMPL_DEBUGGING_OUTPUT
const bool DEBUGGING_OUTPUT = true;
#else
const bool DEBUGGING_OUTPUT = false;
#endif

BeagleCPUImpl::~BeagleCPUImpl() {
    // free all that stuff...
    // If you delete partials, make sure not to delete the last element
    // which is TEMP_SCRATCH_PARTIAL twice.

	for(unsigned int i=0; i<kMatrixCount; i++) {
	    if (gTransitionMatrices[i] != NULL)
		    free(gTransitionMatrices[i]);
	}
    free(gTransitionMatrices);

	for(unsigned int i=0; i<kScaleBufferCount; i++) {
	    if (gScaleBuffers[i] != NULL)
		    free(gScaleBuffers[i]);
	}

	for(unsigned int i=0; i<kBufferCount; i++) {
	    if (gPartials[i] != NULL)
		    free(gPartials[i]);
	    if (gTipStates[i] != NULL)
		    free(gTipStates[i]);
	}
    free(gPartials);
    free(gTipStates);
    
    if (gScaleBuffers)
        free(gScaleBuffers);

	free(gCategoryRates);
	free(integrationTmp);

	free(ones);
	free(zeros);

	delete gEigenDecomposition;
}

int BeagleCPUImpl::createInstance(int tipCount,
                                  int partialsBufferCount,
                                  int compactBufferCount,
                                  int stateCount,
                                  int patternCount,
                                  int eigenDecompositionCount,
                                  int matrixCount,
                                  int categoryCount,
                                  int scaleBufferCount,
                                  int resourceNumber,
                                  long preferenceFlags,
                                  long requirementFlags) {
    if (DEBUGGING_OUTPUT)
        std::cerr << "in BeagleCPUImpl::initialize\n" ;

    kBufferCount = partialsBufferCount + compactBufferCount;
    kTipCount = tipCount;
    assert(kBufferCount > kTipCount);
    kStateCount = stateCount;
    kPatternCount = patternCount;
    kMatrixCount = matrixCount;
    kEigenDecompCount = eigenDecompositionCount;
	kCategoryCount = categoryCount;
    kScaleBufferCount = scaleBufferCount;
#ifdef PAD_MATRICES
    kMatrixSize = (1 + kStateCount) * kStateCount;
#else
    kMatrixSize = kStateCount * kStateCount;
#endif
    
    kFlags = 0;
    
    if (requirementFlags & BEAGLE_FLAG_COMPLEX || preferenceFlags & BEAGLE_FLAG_COMPLEX)
    	kFlags |= BEAGLE_FLAG_COMPLEX;

    if (kFlags & BEAGLE_FLAG_COMPLEX)
    	gEigenDecomposition = new EigenDecompositionSquare<double>(kEigenDecompCount,
    			kStateCount,kCategoryCount,kFlags);
    else
    	gEigenDecomposition = new EigenDecompositionCube<double>(kEigenDecompCount,
    			kStateCount, kCategoryCount);

	gCategoryRates = (double*) malloc(sizeof(double) * kCategoryCount);

    kPartialsSize = kPatternCount * kStateCount * kCategoryCount;

    gPartials = (double**) malloc(sizeof(double*) * kBufferCount);
    if (gPartials == NULL)
     throw std::bad_alloc();

    // assigning kBufferCount to this array so that we can just check if a tipStateBuffer is
    // allocated
    gTipStates = (int**) malloc(sizeof(int*) * kBufferCount);
    if (gTipStates == NULL)
        throw std::bad_alloc();

    for (int i = 0; i < kBufferCount; i++) {
        gPartials[i] = NULL;
        gTipStates[i] = NULL;
    }

    for (int i = kTipCount; i < kBufferCount; i++) {
        gPartials[i] = (double*) malloc(sizeof(double) * kPartialsSize);
        if (gPartials[i] == NULL)
            throw std::bad_alloc();
    }

    gScaleBuffers = NULL;
    gScaleBuffers = (double**) malloc(sizeof(double*) * kScaleBufferCount);
    if (gScaleBuffers == NULL)
         throw std::bad_alloc();

    for (int i = 0; i < kScaleBufferCount; i++) {
        gScaleBuffers[i] = (double*) malloc(sizeof(double) * kPartialsSize);
        if (gScaleBuffers[i] == 0L)
            throw std::bad_alloc();
    }

    gTransitionMatrices = (double**) malloc(sizeof(double*) * kMatrixCount);
    if (gTransitionMatrices == NULL)
        throw std::bad_alloc();
    for (int i = 0; i < kMatrixCount; i++) {
        gTransitionMatrices[i] = (double*) malloc(sizeof(double) * kMatrixSize * kCategoryCount);
        if (gTransitionMatrices[i] == 0L)
            throw std::bad_alloc();
    }

    integrationTmp = (double*) malloc(sizeof(double) * kPatternCount * kStateCount);
//    matrixTmp = (double*) malloc(sizeof(double) * kStateCount);

    zeros = (double*) malloc(sizeof(double) * kPatternCount);
    ones = (double*) malloc(sizeof(double) * kPatternCount);
    for(int i = 0; i < kPatternCount; i++) {
    	zeros[i] = 0.0;
        ones[i] = 1.0;
    }

    return BEAGLE_SUCCESS;
}

int BeagleCPUImpl::getInstanceDetails(BeagleInstanceDetails* returnInfo) {
    if (returnInfo != NULL) {
        returnInfo->resourceNumber = 0;
        returnInfo->flags = BEAGLE_FLAG_DOUBLE | BEAGLE_FLAG_ASYNCH | BEAGLE_FLAG_CPU;
    }

    return BEAGLE_SUCCESS;
}

int BeagleCPUImpl::setTipStates(int tipIndex,
                                const int* inStates) {
    if (tipIndex < 0 || tipIndex >= kTipCount)
        return BEAGLE_ERROR_OUT_OF_RANGE;
    gTipStates[tipIndex] = (int*) malloc(sizeof(int) * kPatternCount);
	for (int j = 0; j < kPatternCount; j++) {
		gTipStates[tipIndex][j] = (inStates[j] < kStateCount ? inStates[j] : kStateCount);
	}

    return BEAGLE_SUCCESS;
}

int BeagleCPUImpl::setTipPartials(int tipIndex,
                                  const double* inPartials) {
    if (tipIndex < 0 || tipIndex >= kTipCount)
        return BEAGLE_ERROR_OUT_OF_RANGE;
    assert(gPartials[tipIndex] == 0L);
    gPartials[tipIndex] = (double*) malloc(sizeof(double) * kPartialsSize);
    if (gPartials[tipIndex] == 0L)
        return BEAGLE_ERROR_OUT_OF_MEMORY;
    int singlePartialsSize = kPatternCount * kStateCount;
    for (int i = 0; i < kCategoryCount; i++)
        memcpy(gPartials[tipIndex] + i * singlePartialsSize, inPartials,
               sizeof(double) * singlePartialsSize);

    return BEAGLE_SUCCESS;
}

int BeagleCPUImpl::setPartials(int bufferIndex,
                               const double* inPartials) {
    if (bufferIndex < 0 || bufferIndex >= kBufferCount)
        return BEAGLE_ERROR_OUT_OF_RANGE;
    if (gPartials[bufferIndex] == NULL) {
        gPartials[bufferIndex] = (double*) malloc(sizeof(double) * kPartialsSize);
        if (gPartials[bufferIndex] == 0L)
            return BEAGLE_ERROR_OUT_OF_MEMORY;
    }
    memcpy(gPartials[bufferIndex], inPartials, sizeof(double) * kPartialsSize);

    return BEAGLE_SUCCESS;
}

int BeagleCPUImpl::getPartials(int bufferIndex,
                               int scaleIndex,
                               double* outPartials) {
    if (bufferIndex < 0 || bufferIndex >= kBufferCount)
        return BEAGLE_ERROR_OUT_OF_RANGE;
    
    //TODO: unscale the partials
    memcpy(outPartials, gPartials[bufferIndex], sizeof(double) * kPartialsSize);

    return BEAGLE_SUCCESS;
}

int BeagleCPUImpl::setEigenDecomposition(int eigenIndex,
                                         const double* inEigenVectors,
                                         const double* inInverseEigenVectors,
                                         const double* inEigenValues) {

	gEigenDecomposition->setEigenDecomposition(eigenIndex, inEigenVectors, inInverseEigenVectors, inEigenValues);
	return BEAGLE_SUCCESS;
}

int BeagleCPUImpl::setCategoryRates(const double* inCategoryRates) {
	memcpy(gCategoryRates, inCategoryRates, sizeof(double) * kCategoryCount);
    return BEAGLE_SUCCESS;
}

int BeagleCPUImpl::setTransitionMatrix(int matrixIndex,
                                       const double* inMatrix) {
    // TODO: test CPU setTransitionMatrix
    memcpy(gTransitionMatrices[matrixIndex], inMatrix,
           sizeof(double) * kMatrixSize * kCategoryCount);
    return BEAGLE_SUCCESS;
}

int BeagleCPUImpl::updateTransitionMatrices(int eigenIndex,
                                            const int* probabilityIndices,
                                            const int* firstDerivativeIndices,
                                            const int* secondDervativeIndices,
                                            const double* edgeLengths,
                                            int count) {
	gEigenDecomposition->updateTransitionMatrices(eigenIndex,probabilityIndices,firstDerivativeIndices,secondDervativeIndices,
												  edgeLengths,gCategoryRates,gTransitionMatrices,count);
	return BEAGLE_SUCCESS;
}

//int BeagleCPUImpl::updateTransitionMatrices(int eigenIndex,
//                                            const int* probabilityIndices,
//                                            const int* firstDerivativeIndices,
//                                            const int* secondDervativeIndices,
//                                            const double* edgeLengths,
//                                            int count) {
//    for (int u = 0; u < count; u++) {
//        double* transitionMat = gTransitionMatrices[probabilityIndices[u]];
//        int n = 0;
//        for (int l = 0; l < kCategoryCount; l++) {
//
//        	if (kFlags | BEAGLE_FLAG_COMPLEX) {
//        		for (int i = 0; i < kStateCount; i++ ) {
//        			if (gEigenValues[eigenIndex][i+kStateCount] == 0)
//        				;
//        		}        		
//        	} else { // Only real diagonalization        	
//        		for (int i = 0; i < kStateCount; i++) {
//        			matrixTmp[i] = exp(gEigenValues[eigenIndex][i] * edgeLengths[u] * gCategoryRates[l]);
//        		}
//        	}
//            int m = 0;
//            for (int i = 0; i < kStateCount; i++) {
//                for (int j = 0; j < kStateCount; j++) {
//                    double sum = 0.0;
//                    for (int k = 0; k < kStateCount; k++) {
//                        sum += gCMatrices[eigenIndex][m] * matrixTmp[k];
//                        m++;
//                    }
//                    if (sum > 0)
//                        transitionMat[n] = sum;
//                    else
//                        transitionMat[n] = 0;
//                    n++;
//                }
//#ifdef PAD_MATRICES
//                transitionMat[n] = 1.0;
//                n++;
//#endif
//            }
//        }
//
//        if (DEBUGGING_OUTPUT) {
//            fprintf(stderr,"transitionMat index=%d brlen=%.5f\n", probabilityIndices[u], edgeLengths[u]);
//            for ( int w = 0; w < (20 > kMatrixSize ? 20 : kMatrixSize); ++w)
//                fprintf(stderr,"transitionMat[%d] = %.5f\n", w, transitionMat[w]);
//        }
//    }
//
//    return BEAGLE_SUCCESS;
//}

int BeagleCPUImpl::updatePartials(const int* operations,
                                  int count,
                                  int cumulativeScaleIndex) {

    double* cumulativeScaleBuffer = NULL;
    if (cumulativeScaleIndex != BEAGLE_OP_NONE)
        cumulativeScaleBuffer = gScaleBuffers[cumulativeScaleIndex];

    for (int op = 0; op < count; op++) {
        if (DEBUGGING_OUTPUT) {
            std::cerr << "op[0]= " << operations[0] << "\n";
            std::cerr << "op[1]= " << operations[1] << "\n";
            std::cerr << "op[2]= " << operations[2] << "\n";
            std::cerr << "op[3]= " << operations[3] << "\n";
            std::cerr << "op[4]= " << operations[4] << "\n";
            std::cerr << "op[5]= " << operations[5] << "\n";
            std::cerr << "op[6]= " << operations[6] << "\n";
        }

        const int parIndex = operations[op * 7];
        const int writeScalingIndex = operations[op * 7 + 1];
        const int readScalingIndex = operations[op * 7 + 2];
        const int child1Index = operations[op * 7 + 3];
        const int child1TransMatIndex = operations[op * 7 + 4];
        const int child2Index = operations[op * 7 + 5];
        const int child2TransMatIndex = operations[op * 7 + 6];

        const double* partials1 = gPartials[child1Index];
        const double* partials2 = gPartials[child2Index];

        const int* tipStates1 = gTipStates[child1Index];
        const int* tipStates2 = gTipStates[child2Index];

        const double* matrices1 = gTransitionMatrices[child1TransMatIndex];
        const double* matrices2 = gTransitionMatrices[child2TransMatIndex];

        double* destPartials = gPartials[parIndex];

        int rescale = BEAGLE_OP_NONE;
        double* scalingFactors = NULL;
        if (writeScalingIndex >= 0) {
            rescale = 1;
            scalingFactors = gScaleBuffers[writeScalingIndex];
        } else if (readScalingIndex >= 0) {
            rescale = 0;
            scalingFactors = gScaleBuffers[readScalingIndex];
        }
        
        if (DEBUGGING_OUTPUT) {
            std::cerr << "Rescale= " << rescale << " writeIndex= " << writeScalingIndex 
                     << " readIndex = " << readScalingIndex << "\n";
        }

        if (tipStates1 != NULL) {
            if (tipStates2 != NULL ) {
                if (rescale == 0) { // Use fixed scaleFactors
                    calcStatesStatesFixedScaling(destPartials, tipStates1, matrices1, tipStates2, matrices2,
                                                 scalingFactors);
                } else { 
                    // First compute without any scaling
                    calcStatesStates(destPartials, tipStates1, matrices1, tipStates2, matrices2);//,
//                                     scalingFactors, cumulativeScalingBuffer, rescale);
                    if (rescale == 1) // Recompute scaleFactors
                        rescalePartials(destPartials,scalingFactors,cumulativeScaleBuffer,1);
                }
            } else {
                if (rescale == 0) {
                    calcStatesPartialsFixedScaling(destPartials, tipStates1, matrices1, partials2, matrices2,
                                                   scalingFactors);
                } else {
                    calcStatesPartials(destPartials, tipStates1, matrices1, partials2, matrices2);//,
//                                   scalingFactors, cumulativeScalingBuffer, rescale);
                    if (rescale == 1)
                        rescalePartials(destPartials,scalingFactors,cumulativeScaleBuffer,0);
                }
            }
        } else {
            if (tipStates2 != NULL) {
                if (rescale == 0) {
                    calcStatesPartialsFixedScaling(destPartials,tipStates2,matrices2,partials1,matrices1,
                                                   scalingFactors);
                } else {
                    calcStatesPartials(destPartials, tipStates2, matrices2, partials1, matrices1);//,
//                                   scalingFactors, cumulativeScalingBuffer, rescale);
                    if (rescale == 1)
                        rescalePartials(destPartials,scalingFactors,cumulativeScaleBuffer,0);
                }
            } else {
                if (rescale == 0) {
                    calcPartialsPartialsFixedScaling(destPartials,partials1,matrices1,partials2,matrices2,
                                                     scalingFactors);
                } else {
                    calcPartialsPartials(destPartials, partials1, matrices1, partials2, matrices2);//,
//                                   scalingFactors, cumulativeScalingBuffer, rescale);
                    if (rescale == 1)
                        rescalePartials(destPartials,scalingFactors,cumulativeScaleBuffer,0);
                }
            }
        }
        if (DEBUGGING_OUTPUT) {
            if (scalingFactors != NULL && rescale == 0) {
                for(int i=0; i<kPatternCount; i++)
                    fprintf(stderr,"old scaleFactor[%d] = %.5f\n",i,scalingFactors[i]);
            }
            fprintf(stderr,"Result partials:\n");
            for(int i = 0; i < kPartialsSize; i++)
                fprintf(stderr,"destP[%d] = %.5f\n",i,destPartials[i]);
        }
    }

    return BEAGLE_SUCCESS;
}


int BeagleCPUImpl::waitForPartials(const int* destinationPartials,
                                   int destinationPartialsCount) {
    return BEAGLE_SUCCESS;
}


int BeagleCPUImpl::calculateRootLogLikelihoods(const int* bufferIndices,
                                               const double* inWeights,
                                               const double* inStateFrequencies,
                                               const int* scaleBufferIndices,
                                               int count,
                                               double* outLogLikelihoods) {

    if (count == 1) {
        // We treat this as a special case so that we don't have convoluted logic
        //      at the end of the loop over patterns
        calcRootLogLikelihoods(bufferIndices[0], inWeights, inStateFrequencies, scaleBufferIndices[0], outLogLikelihoods);
    }
    else
    {
        // Here we do the 3 similar operations:
        //              1. to set the lnL to the contribution of the first subset,
        //              2. to add the lnL for other subsets up to the penultimate
        //              3. to add the final subset and take the lnL
        //      This form of the calc would not work when count == 1 because
        //              we need operation 1 and 3 in the preceding list.  This is not
        //              a problem, though as we deal with count == 1 in the previous
        //              branch.
        
        int indexMaxScale;
		std::vector<int> maxScaleFactor(kPatternCount);
        
        for (int subsetIndex = 0 ; subsetIndex < count; ++subsetIndex ) {
            const int rootPartialIndex = bufferIndices[subsetIndex];
            const double* rootPartials = gPartials[rootPartialIndex];
            const double* frequencies = inStateFrequencies + (subsetIndex * kStateCount);
            const double* wt = inWeights + subsetIndex * kCategoryCount;
            int u = 0;
            int v = 0;
            for (int k = 0; k < kPatternCount; k++) {
                for (int i = 0; i < kStateCount; i++) {
                    integrationTmp[u] = rootPartials[v] * wt[0];
                    u++;
                    v++;
                }
            }
            for (int l = 1; l < kCategoryCount; l++) {
                u = 0;
                for (int k = 0; k < kPatternCount; k++) {
                    for (int i = 0; i < kStateCount; i++) {
                        integrationTmp[u] += rootPartials[v] * wt[l];
                        u++;
                        v++;
                    }
                }
            }
            u = 0;
            for (int k = 0; k < kPatternCount; k++) {
                double sum = 0.0;
                for (int i = 0; i < kStateCount; i++) {
                    sum += frequencies[i] * integrationTmp[u];
                    u++;
                }
                
                if (scaleBufferIndices != NULL) {
                    if (subsetIndex == 0) {
                        indexMaxScale = 0;
                        maxScaleFactor[k] = scaleBufferIndices[k];
                        for (int j = 1; j < count; j++) {
                            if (scaleBufferIndices[j * kPatternCount + k] > maxScaleFactor[k]) {
                                indexMaxScale = j;
                                maxScaleFactor[k] = scaleBufferIndices[j * kPatternCount + k];
                            }
                        }
                    }
                    
                    if (subsetIndex != indexMaxScale)
                        sum *= exp((double)(scaleBufferIndices[k] - maxScaleFactor[k]));
                }
                
                if (subsetIndex == 0)
                    outLogLikelihoods[k] = sum;
                else if (subsetIndex == count - 1)
                    outLogLikelihoods[k] = log(outLogLikelihoods[k] + sum);
                else
                    outLogLikelihoods[k] += sum;
            }
        }
        
        if (scaleBufferIndices != NULL) {
            for(int i=0; i<kPatternCount; i++)
                outLogLikelihoods[i] += maxScaleFactor[i];
        }
    }

    return BEAGLE_SUCCESS;
}

void BeagleCPUImpl::calcRootLogLikelihoods(const int bufferIndex,
                            const double* inWeights,
                            const double* inStateFrequencies,
                            const int scalingFactorsIndex,
                            double* outLogLikelihoods) {

    const double* rootPartials = gPartials[bufferIndex];
    const double* wt = inWeights;
    int u = 0;
    int v = 0;
    for (int k = 0; k < kPatternCount; k++) {
        for (int i = 0; i < kStateCount; i++) {
            integrationTmp[u] = rootPartials[v] * wt[0];
            u++;
            v++;
        }
    }
    for (int l = 1; l < kCategoryCount; l++) {
        u = 0;
        for (int k = 0; k < kPatternCount; k++) {
            for (int i = 0; i < kStateCount; i++) {
                integrationTmp[u] += rootPartials[v] * wt[l];
                u++;
                v++;
            }
        }
    }
    u = 0;
    for (int k = 0; k < kPatternCount; k++) {
        double sum = 0.0;
        for (int i = 0; i < kStateCount; i++) {
            sum += inStateFrequencies[i] * integrationTmp[u];
            u++;
        }
        outLogLikelihoods[k] = log(sum);   // take the log
    }
    
    if (scalingFactorsIndex >= 0) {
    	const double* cumulativeScaleFactors = gScaleBuffers[scalingFactorsIndex];
    	for(int i=0; i<kPatternCount; i++)
    		outLogLikelihoods[i] += cumulativeScaleFactors[i];
    }
}

int BeagleCPUImpl::accumulateScaleFactors(const int* scalingIndices,
                                                int  count,
                                                int  cumulativeScalingIndex) {
    
    double* cumulativeScaleBuffer = gScaleBuffers[cumulativeScalingIndex];
    for(int i=0; i<count; i++) {
        const double* scaleBuffer = gScaleBuffers[scalingIndices[i]];
        for(int j=0; j<kPatternCount; j++) 
            cumulativeScaleBuffer[j] += log(scaleBuffer[j]);
    }
    
    if (DEBUGGING_OUTPUT) {
        fprintf(stderr,"Accumulating %d scale buffers into #%d\n",count,cumulativeScalingIndex);
        for(int j=0; j<kPatternCount; j++) {
            fprintf(stderr,"cumulativeScaleBuffer[%d] = %2.5e\n",j,cumulativeScaleBuffer[j]);
        }
    }
    return BEAGLE_SUCCESS;
}

int BeagleCPUImpl::removeScaleFactors(const int* scalingIndices,
                                            int  count,
                                            int  cumulativeScalingIndex) {
    double* cumulativeScaleBuffer = gScaleBuffers[cumulativeScalingIndex];
    for(int i=0; i<count; i++) {
        const double* scaleBuffer = gScaleBuffers[scalingIndices[i]];
        for(int j=0; j<kPatternCount; j++) 
            cumulativeScaleBuffer[j] -= log(scaleBuffer[j]);
    }    
    
    return BEAGLE_SUCCESS;
}

int BeagleCPUImpl::resetScaleFactors(int cumulativeScalingIndex) {
    //memcpy(gScaleBuffers[cumulativeScalingIndex],zeros,sizeof(double) * kPatternCount);
    memset(gScaleBuffers[cumulativeScalingIndex], 0, sizeof(double) * kPatternCount);
    return BEAGLE_SUCCESS;
}

int BeagleCPUImpl::calculateEdgeLogLikelihoods(const int* parentBufferIndices,
                                               const int* childBufferIndices,
                                               const int* probabilityIndices,
                                               const int* firstDerivativeIndices,
                                               const int* secondDerivativeIndices,
                                               const double* inWeights,
                                               const double* inStateFrequencies,
                                               const int* scalingFactorsIndices,
                                               int count,
                                               double* outLogLikelihoods,
                                               double* outFirstDerivatives,
                                               double* outSecondDerivatives) {
    // TODO: implement for count > 1 
    
    assert(count == 1);
    assert(firstDerivativeIndices == 0L);
    assert(secondDerivativeIndices == 0L);
    assert(outFirstDerivatives == 0L);
    assert(outSecondDerivatives == 0L);
    
    if (count == 1) {
        calcEdgeLogLikelihoods(parentBufferIndices[0], childBufferIndices[0], probabilityIndices[0], BEAGLE_OP_NONE, BEAGLE_OP_NONE, inWeights, inStateFrequencies, scalingFactorsIndices[0], outLogLikelihoods, outFirstDerivatives, outSecondDerivatives);
    } else {
        fprintf(stderr,"BeagleCPUImpl::calculateEdgeLogLikelihoods not yet implemented for count > 1\n");
        return BEAGLE_ERROR_OUT_OF_RANGE;
    }
        
    return BEAGLE_SUCCESS;    
}

void BeagleCPUImpl::calcEdgeLogLikelihoods(const int parIndex,
                                           const int childIndex,
                                           const int probIndex,
                                           const int firstDerivativeIndex,
                                           const int secondDerivativeIndex,
                                           const double* inWeights,
                                           const double* inStateFrequencies,
                                           const int scalingFactorsIndex,
                                           double* outLogLikelihoods,
                                           double* outFirstDerivatives,
                                           double* outSecondDerivatives) {
    // TODO: implement derivatives for calculateEdgeLnL
    // TODO: implement rate categories for calculateEdgeLnL

    assert(parIndex >= kTipCount);

    const double* partialsParent = gPartials[parIndex];
    const double* transMatrix = gTransitionMatrices[probIndex];
    const double* wt = inWeights;
        
    memset(integrationTmp, 0, (kPatternCount * kStateCount)*sizeof(double));
    
    if (childIndex < kTipCount && gTipStates[childIndex]) { // Integrate against a state at the child
        
        const int* statesChild = gTipStates[childIndex];    
        int v = 0; // Index for parent partials
        
        for(int l = 0; l < kCategoryCount; l++) {
            int u = 0; // Index in resulting product-partials (summed over categories)
            const double weight = wt[l];
            for(int k = 0; k < kPatternCount; k++) {
                
                const int stateChild = statesChild[k];  // DISCUSSION PT: Does it make sense to change the order of the partials,
                                                        // so we can interchange the patterCount and categoryCount loop order?
                int w =  l * kMatrixSize;
                for(int i = 0; i < kStateCount; i++) {
                    integrationTmp[u] += transMatrix[w + stateChild] * partialsParent[v + i] * weight;
                    u++;
#ifdef PAD_MATRICES
                    w += (kStateCount + 1);
#else
                    w += kStateCount;
#endif
                }
                v += kStateCount;
            }
        }
        
    } else { // Integrate against a partial at the child
        
        const double* partialsChild = gPartials[childIndex];
        int v = 0;
        
        for(int l = 0; l < kCategoryCount; l++) {
            int u = 0;
            const double weight = wt[l];
            for(int k = 0; k < kPatternCount; k++) {                
                int w = l * kMatrixSize;                 
                for(int i = 0; i < kStateCount; i++) {
                    double sumOverJ = 0.0;
                    for(int j = 0; j < kStateCount; j++) {
                        sumOverJ += transMatrix[w] * partialsChild[v + j];
                        w++;
                    }
#ifdef PAD_MATRICES
                    // increment for the extra column at the end
                    w++;
#endif
                    integrationTmp[u] += sumOverJ * partialsParent[v + i] * weight;
                    u++;
                }
                v += kStateCount;
            }
        }
    }
        
    int u = 0;
    for(int k = 0; k < kPatternCount; k++) {
        double sumOverI = 0.0;
        for(int i = 0; i < kStateCount; i++) {
            sumOverI += inStateFrequencies[i] * integrationTmp[u];
            u++;
        }
        outLogLikelihoods[k] = log(sumOverI);
    }        

    
    if (scalingFactorsIndex != BEAGLE_OP_NONE) {
        const double* scalingFactors = gScaleBuffers[scalingFactorsIndex];
        for(int k=0; k < kPatternCount; k++)
            outLogLikelihoods[k] += scalingFactors[k];
    }
}

int BeagleCPUImpl::block(void) {
	// Do nothing.
	return BEAGLE_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
// private methods

/*
 * Re-scales the partial likelihoods such that the largest is one.
 */
void BeagleCPUImpl::rescalePartials(double* destP,
                                    double* scaleFactors,
                                    double* cumulativeScaleFactors,
                                       const int  fillWithOnes) {
    if (DEBUGGING_OUTPUT) {
        std::cerr << "destP (before rescale): \n";// << destP << "\n";
        for(int i=0; i<kPartialsSize; i++)
            fprintf(stderr,"destP[%d] = %.5f\n",i,destP[i]);
    }
    
    if (kStateCount == 4 && fillWithOnes != 0) {
//        if (ones == NULL) {
//            ones = (double*) malloc(sizeof(double) * kPatternCount);
//                for(int i = 0; i < kPatternCount; i++)
//                    ones[i] = 1.0;
//        }
        memcpy(scaleFactors,ones,sizeof(double) * kPatternCount);
        // No accumulation necessary as cumulativeScaleFactors are on the log-scale
        if (DEBUGGING_OUTPUT)
            fprintf(stderr,"Ones copied!\n");
        return;
    }

    // TODO None of the code below has been optimized.
    for (int k = 0; k < kPatternCount; k++) {
        double max = 0;
        const int patternOffset = k * kStateCount;
        for (int l = 0; l < kCategoryCount; l++) {
            int offset = l * kPatternCount * kStateCount + patternOffset;
            for (int i = 0; i < kStateCount; i++) {
                if(destP[offset] > max)
                    max = destP[offset];
                offset++;
            }
        }
        for (int l = 0; l < kCategoryCount; l++) {
            int offset = l * kPatternCount * kStateCount + patternOffset;
            for (int i = 0; i < kStateCount; i++)
                destP[offset++] /= max;
        }
        if (max == 0)
            max = 1.0;
        scaleFactors[k] = max;
        if( cumulativeScaleFactors != NULL )
            cumulativeScaleFactors[k] += log(max);
    }
    if (DEBUGGING_OUTPUT) {
        for(int i=0; i<kPatternCount; i++)
            fprintf(stderr,"new scaleFactor[%d] = %.5f\n",i,scaleFactors[i]);
    }
}

/*
 * Calculates partial likelihoods at a node when both children have states.
 */
void BeagleCPUImpl::calcStatesStates(double* destP,
                                     const int* states1,
                                     const double* matrices1,
                                     const int* states2,
                                     const double* matrices2) {

    int v = 0;
    for (int l = 0; l < kCategoryCount; l++) {
        for (int k = 0; k < kPatternCount; k++) {
            const int state1 = states1[k];
            const int state2 = states2[k];
            if (DEBUGGING_OUTPUT) {
                std::cerr << "calcStatesStates s1 = " << state1 << '\n';
                std::cerr << "calcStatesStates s2 = " << state2 << '\n';
            }
            int w = l * kMatrixSize;
            for (int i = 0; i < kStateCount; i++) {
                destP[v] = matrices1[w + state1] * matrices2[w + state2];
                v++;
#ifdef PAD_MATRICES
                w += (kStateCount + 1);
#else
                w += kStateCount;
#endif
            }
        }
    }
}

void BeagleCPUImpl::calcStatesStatesFixedScaling(double* destP,
                                              const int* child1States,
                                           const double* child1TransMat,
                                              const int* child2States,
                                           const double* child2TransMat,
                                           const double* scaleFactors) {
    int v = 0;
    for (int l = 0; l < kCategoryCount; l++) {
        for (int k = 0; k < kPatternCount; k++) {
            const int state1 = child1States[k];
            const int state2 = child2States[k];
            int w = l * kMatrixSize;
            double scaleFactor = scaleFactors[k];
            for (int i = 0; i < kStateCount; i++) {
                destP[v] = child1TransMat[w + state1] *
                           child2TransMat[w + state2] / scaleFactor;
                v++;
#ifdef PAD_MATRICES
                w += (kStateCount + 1);
#else
                w += kStateCount;
#endif
            }
        }
    }
}

/*
 * Calculates partial likelihoods at a node when one child has states and one has partials.
 */
void BeagleCPUImpl::calcStatesPartials(double* destP,
                                       const int* states1,
                                       const double* matrices1,
                                       const double* partials2,
                                       const double* matrices2) {
    int u = 0;
    int v = 0;
    for (int l = 0; l < kCategoryCount; l++) {
        for (int k = 0; k < kPatternCount; k++) {
            int state1 = states1[k];
            int w = l * kMatrixSize;
            for (int i = 0; i < kStateCount; i++) {
                double tmp = matrices1[w + state1];
                double sum = 0.0;
                for (int j = 0; j < kStateCount; j++) {
                    sum += matrices2[w] * partials2[v + j];
                    w++;
                }
#ifdef PAD_MATRICES
                // increment for the extra column at the end
                w++;
#endif
                destP[u] = tmp * sum;
                u++;
            }
            v += kStateCount;
        }
    }
}

void BeagleCPUImpl::calcStatesPartialsFixedScaling(double* destP,
                                                const int* states1,
                                             const double* matrices1,
                                             const double* partials2,
                                             const double* matrices2,
                                             const double* scaleFactors) {
    int u = 0;
    int v = 0;
    for (int l = 0; l < kCategoryCount; l++) {
        for (int k = 0; k < kPatternCount; k++) {
            int state1 = states1[k];
            int w = l * kMatrixSize;
            double scaleFactor = scaleFactors[k];
            for (int i = 0; i < kStateCount; i++) {
                double tmp = matrices1[w + state1];
                double sum = 0.0;
                for (int j = 0; j < kStateCount; j++) {
                    sum += matrices2[w] * partials2[v + j];
                    w++;
                }
#ifdef PAD_MATRICES
                // increment for the extra column at the end
                w++;
#endif
                destP[u] = tmp * sum / scaleFactor;
                u++;
            }
            v += kStateCount;
        }
    }
}

/*
 * Calculates partial likelihoods at a node when both children have partials.
 */
void BeagleCPUImpl::calcPartialsPartials(double* destP,
                                         const double* partials1,
                                         const double* matrices1,
                                         const double* partials2,
                                         const double* matrices2) {
    double sum1, sum2;
    int u = 0;
    int v = 0;
    for (int l = 0; l < kCategoryCount; l++) {
        for (int k = 0; k < kPatternCount; k++) {
            int w = l * kMatrixSize;
            for (int i = 0; i < kStateCount; i++) {
                sum1 = sum2 = 0.0;
                for (int j = 0; j < kStateCount; j++) {
                    sum1 += matrices1[w] * partials1[v + j];
                    sum2 += matrices2[w] * partials2[v + j];
                    if (DEBUGGING_OUTPUT) {
                        if (k == 0)
                            printf("mat1[%d] = %.5f\n", w, matrices1[w]);
                        if (k == 1)
                            printf("mat2[%d] = %.5f\n", w, matrices2[w]);
                    }
                    w++;
                }
#ifdef PAD_MATRICES
                // increment for the extra column at the end
                w++;
#endif
                destP[u] = sum1 * sum2;
                u++;
            }
            v += kStateCount;
        }
    }
}

void BeagleCPUImpl::calcPartialsPartialsFixedScaling(double* destP,
                                               const double* partials1,
                                               const double* matrices1,
                                               const double* partials2,
                                               const double* matrices2,
                                               const double* scaleFactors) {
    double sum1, sum2;
    int u = 0;
    int v = 0;
    for (int l = 0; l < kCategoryCount; l++) {
        for (int k = 0; k < kPatternCount; k++) {
            int w = l * kMatrixSize;
            double scaleFactor = scaleFactors[k];
            for (int i = 0; i < kStateCount; i++) {
                sum1 = sum2 = 0.0;
                for (int j = 0; j < kStateCount; j++) {
                    sum1 += matrices1[w] * partials1[v + j];
                    sum2 += matrices2[w] * partials2[v + j];
                    w++;
                }
#ifdef PAD_MATRICES
                // increment for the extra column at the end
                w++;
#endif
                destP[u] = sum1 * sum2 / scaleFactor;
                u++;
            }
            v += kStateCount;
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
// BeagleCPUImplFactory public methods

BeagleImpl* BeagleCPUImplFactory::createImpl(int tipCount,
                                             int partialsBufferCount,
                                             int compactBufferCount,
                                             int stateCount,
                                             int patternCount,
                                             int eigenBufferCount,
                                             int matrixBufferCount,
                                             int categoryCount,
                                             int scaleBufferCount,
                                             int resourceNumber,
                                             long preferenceFlags,
                                             long requirementFlags,
                                             int* errorCode) {
    BeagleImpl* impl = new BeagleCPUImpl();

    try {
        *errorCode = 
            impl->createInstance(tipCount, partialsBufferCount, compactBufferCount, stateCount,
                                 patternCount, eigenBufferCount, matrixBufferCount,
                                 categoryCount,scaleBufferCount, resourceNumber, preferenceFlags, requirementFlags);
        if (*errorCode == BEAGLE_SUCCESS) {
            return impl;
        }
        delete impl;
        return NULL;
    }
    catch(...) {
        if (DEBUGGING_OUTPUT)
            std::cerr << "exception in initialize\n";
        delete impl;
        throw;
    }

    delete impl;

    return NULL;
}

const char* BeagleCPUImplFactory::getName() {
    return "CPU";
}

const long BeagleCPUImplFactory::getFlags() {
    return BEAGLE_FLAG_ASYNCH | BEAGLE_FLAG_CPU | BEAGLE_FLAG_DOUBLE | BEAGLE_FLAG_COMPLEX;
}

