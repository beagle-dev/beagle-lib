/*
 *  BeagleCPUImpl.cpp
 *  BEAGLE
 *
 * @author Andrew Rambaut
 * @author Marc Suchard
 * @author Daniel Ayres
 * @author Mark Holder
 */

///@TODO: deal with underflow
///@TODO: get rid of malloc (use vectors to make sure that memory is freed)
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

#include "libhmsbeagle/beagle.h"
#include "libhmsbeagle/CPU/BeagleCPUImpl.h"

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
	
	for(int i=0; i<partials.size(); i++) {
		free(partials[i]);	
	}
	
	for(int i=0; i<kEigenDecompCount; i++) {
		free(cMatrices[i]);
		free(eigenValues[i]);
	}
	
	free(cMatrices);
	free(eigenValues);
			
	free(categoryRates);
	free(integrationTmp);
	
	
}

int BeagleCPUImpl::createInstance(int tipCount,
                                  int partialsBufferCount,
                                  int compactBufferCount,
                                  int stateCount,
                                  int patternCount,
                                  int eigenDecompositionCount,
                                  int matrixCount,
                                  int categoryCount,
                                  int scaleBufferCount) {
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
    
    kMatrixSize = (1 + kStateCount) * kStateCount;
    
    cMatrices = (double**) malloc(sizeof(double*) * eigenDecompositionCount);
    if (cMatrices == 0L)
        throw std::bad_alloc();
    
    eigenValues = (double**) malloc(sizeof(double*) * eigenDecompositionCount);
    if (eigenValues == 0L)
        throw std::bad_alloc();
    
    for (int i = 0; i < eigenDecompositionCount; i++) {
        cMatrices[i] = (double*) malloc(sizeof(double) * kStateCount * kStateCount * kStateCount);
        if (cMatrices[i] == 0L)
            throw std::bad_alloc();
        
        eigenValues[i] = (double*) malloc(sizeof(double) * kStateCount);
        if (eigenValues[i] == 0L)
            throw std::bad_alloc();
    }
    
	categoryRates = (double*) malloc(sizeof(double) * kCategoryCount);
    
    kPartialsSize = kPatternCount * kStateCount * kCategoryCount;
    
    partials.assign(kBufferCount, 0L);
    tipStates.assign(kTipCount, 0L);
    
    for (int i = kTipCount; i < kBufferCount; i++) {
        partials[i] = (double*) malloc(sizeof(double) * kPartialsSize);
        if (partials[i] == 0L)
            throw std::bad_alloc();
    }
    
    std::vector<double> emptyMat(kMatrixSize * kCategoryCount);
    transitionMatrices.assign(kMatrixCount, emptyMat);
    
	integrationTmp = (double*) malloc(sizeof(double) * kPatternCount * kStateCount);
    
    return BEAGLE_SUCCESS;
}

int BeagleCPUImpl::initializeInstance(BeagleInstanceDetails* returnInfo) {
    if (returnInfo != NULL) {
        returnInfo->resourceNumber = 0;
        returnInfo->flags = BEAGLE_FLAG_SINGLE | BEAGLE_FLAG_ASYNCH | BEAGLE_FLAG_CPU;
    }
    
    return BEAGLE_SUCCESS;
}

int BeagleCPUImpl::setTipStates(int tipIndex,
                                const int* inStates) {
    if (tipIndex < 0 || tipIndex >= kTipCount)
        return BEAGLE_ERROR_OUT_OF_RANGE;
    tipStates[tipIndex] = (int*) malloc(sizeof(int) * kPatternCount);
	for (int j = 0; j < kPatternCount; j++) {
		tipStates[tipIndex][j] = (inStates[j] < kStateCount ? inStates[j] : kStateCount);
	}        
    
    return BEAGLE_SUCCESS;
}

int BeagleCPUImpl::setTipPartials(int tipIndex,
                                  const double* inPartials) {
    if (tipIndex < 0 || tipIndex >= kTipCount)
        return BEAGLE_ERROR_OUT_OF_RANGE;
    assert(partials[tipIndex] == 0L);
    partials[tipIndex] = (double*) malloc(sizeof(double) * kPartialsSize);
    if (partials[tipIndex] == 0L)
        return BEAGLE_ERROR_OUT_OF_MEMORY;
    int singlePartialsSize = kPatternCount * kStateCount;
    for (int i = 0; i < kCategoryCount; i++)
        memcpy(partials[tipIndex] + i * singlePartialsSize, inPartials,
               sizeof(double) * singlePartialsSize);
    
    return BEAGLE_SUCCESS;
}

int BeagleCPUImpl::setPartials(int bufferIndex,
                               const double* inPartials) {
    if (bufferIndex < 0 || bufferIndex >= kBufferCount)
        return BEAGLE_ERROR_OUT_OF_RANGE;
    assert(partials[bufferIndex] == 0L);
    partials[bufferIndex] = (double*) malloc(sizeof(double) * kPartialsSize);
    if (partials[bufferIndex] == 0L)
        return BEAGLE_ERROR_OUT_OF_MEMORY;
    memcpy(partials[bufferIndex], inPartials, sizeof(double) * kPartialsSize);
    
    return BEAGLE_SUCCESS;
}

int BeagleCPUImpl::getPartials(int bufferIndex,
							   int scaleIndex,
                               double* outPartials) {
    if (bufferIndex < 0 || bufferIndex >= kBufferCount)
        return BEAGLE_ERROR_OUT_OF_RANGE;
    memcpy(outPartials, partials[bufferIndex], sizeof(double) * kPartialsSize);
    
    return BEAGLE_SUCCESS;
}

int BeagleCPUImpl::setEigenDecomposition(int eigenIndex,
                                         const double* inEigenVectors,
                                         const double* inInverseEigenVectors,
                                         const double* inEigenValues) {
    int l = 0;
    for (int i = 0; i < kStateCount; i++) {
        eigenValues[eigenIndex][i] = inEigenValues[i];
        for (int j = 0; j < kStateCount; j++) {
            for (int k = 0; k < kStateCount; k++) {
                cMatrices[eigenIndex][l] = inEigenVectors[(i * kStateCount) + k]
                        * inInverseEigenVectors[(k * kStateCount) + j];
                l++;
            }
        }
    }
    
    return BEAGLE_SUCCESS;
}

int BeagleCPUImpl::setCategoryRates(const double* inCategoryRates) {
	memcpy(categoryRates, inCategoryRates, sizeof(double) * kCategoryCount);
    return BEAGLE_SUCCESS;
}

int BeagleCPUImpl::setTransitionMatrix(int matrixIndex,
                                       const double* inMatrix) {
    // TODO: test CPU setTransitionMatrix
    memcpy(&(transitionMatrices[matrixIndex][0]), inMatrix,
           sizeof(double) * kMatrixSize * kCategoryCount);
    return BEAGLE_SUCCESS;
}

int BeagleCPUImpl::updateTransitionMatrices(int eigenIndex,
                                            const int* probabilityIndices,
                                            const int* firstDerivativeIndices,
                                            const int* secondDervativeIndices,
                                            const double* edgeLengths,
                                            int count) {
    std::vector<double> tmp;
    tmp.resize(kStateCount);
    
    for (int u = 0; u < count; u++) {
        std::vector<double> & transitionMat = transitionMatrices[probabilityIndices[u]];
        int n = 0;
        for (int l = 0; l < kCategoryCount; l++) {
            
            for (int i = 0; i < kStateCount; i++) {
                tmp[i] = exp(eigenValues[eigenIndex][i] * edgeLengths[u] * categoryRates[l]);
            }
            
            int m = 0;
            for (int i = 0; i < kStateCount; i++) {
                for (int j = 0; j < kStateCount; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < kStateCount; k++) {
                        sum += cMatrices[eigenIndex][m] * tmp[k];
                        m++;
                    }
                    if (sum > 0)
                        transitionMat[n] = sum;
                    else
                        transitionMat[n] = 0;
                    n++;
                }
                transitionMat[n] = 1.0;
                n++;
            }
        }
        
        if (DEBUGGING_OUTPUT) {
            printf("transitionMat index=%d brlen=%.5f\n", probabilityIndices[u], edgeLengths[u]);
            for ( int w = 0; w < 20; ++w)
                printf("transitionMat[%d] = %.5f\n", w, transitionMat[w]);
        }
    }
    
    return BEAGLE_SUCCESS;
}

int BeagleCPUImpl::updatePartials(const int* operations,
                                  int count,
                                  int cumulativeScalingIndex) {
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
//      const int writeScalingIndex = operations[op * 7 + 1];
//      const int readScalingIndex = operations[op * 7 + 2];
        const int child1Index = operations[op * 7 + 3];
        const int child1TransMatIndex = operations[op * 7 + 4];
        const int child2Index = operations[op * 7 + 5];
        const int child2TransMatIndex = operations[op * 7 + 6];
        
        assert(parIndex < partials.size());
        assert(parIndex >= tipStates.size());
        assert(child1Index < partials.size());
        assert(child2Index < partials.size());
        assert(child1TransMatIndex < transitionMatrices.size());
        assert(child2TransMatIndex < transitionMatrices.size());
        
        const double* child1TransMat = &(transitionMatrices[child1TransMatIndex][0]);
        assert(child1TransMat);
        const double* child2TransMat = &(transitionMatrices[child2TransMatIndex][0]);
        assert(child2TransMat);
        double* destPartial = partials[parIndex];
        assert(destPartial);
        
        if (child1Index < kTipCount && tipStates[child1Index]) {
            if (child2Index < kTipCount && tipStates[child2Index]) {
                calcStatesStates(destPartial, tipStates[child1Index], child1TransMat,
                                 tipStates[child2Index], child2TransMat);
            } else {
                calcStatesPartials(destPartial, tipStates[child1Index], child1TransMat,
                                   partials[child2Index], child2TransMat);
            }
        } else {
            if (child2Index < kTipCount && tipStates[child2Index] ) {
                calcStatesPartials(destPartial, tipStates[child2Index], child2TransMat,
                                   partials[child1Index], child1TransMat);
            } else {
                calcPartialsPartials(destPartial, partials[child1Index], child1TransMat,
                                     partials[child2Index], child2TransMat);
            }
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
                                               const int* scalingFactorsIndices,
                                               int count,
                                               double* outLogLikelihoods) {

    if (count == 1) {
        // We treat this as a special case so that we don't have convoluted logic
        //      at the end of the loop over patterns
                
        calcRootLogLikelihoods(bufferIndices[0], inWeights, inStateFrequencies, scalingFactorsIndices[0], outLogLikelihoods);
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
        for (int subsetIndex = 0 ; subsetIndex < count; ++subsetIndex ) {
            assert(subsetIndex < partials.size());
            const int rootPartialIndex = bufferIndices[subsetIndex];
            const double* rootPartials = partials[rootPartialIndex];
            assert(rootPartials);
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
                if (subsetIndex == 0)
                    outLogLikelihoods[k] = sum;
                else if (subsetIndex == count - 1)
                    outLogLikelihoods[k] = log(outLogLikelihoods[k] + sum);
                else
                    outLogLikelihoods[k] += sum;
            }
        }
    }
    

    
    return BEAGLE_SUCCESS;
}

void BeagleCPUImpl::calcRootLogLikelihoods(const int bufferIndex,
                            const double* inWeights,
                            const double* inStateFrequencies,
                            const int scalingFactorsIndex,
                            double* outLogLikelihoods) {

    const double* rootPartials = partials[bufferIndex];
    assert(rootPartials);
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
}

int BeagleCPUImpl::accumulateScaleFactors(const int* scalingIndices,
										  int count,
										  int cumulativeScalingIndex) {
    // TODO: implement accumulateScaleFactors CPU
    return BEAGLE_SUCCESS;
}

int BeagleCPUImpl::removeScaleFactors(const int* scalingIndices,
										  int count,
										  int cumulativeScalingIndex) {
    // TODO: implement removeScaleFactors CPU
    return BEAGLE_SUCCESS;
}

int BeagleCPUImpl::resetScaleFactors(int cumulativeScalingIndex) {
    // TODO: implement resetScaleFactors CPU
    return BEAGLE_SUCCESS;
}

int BeagleCPUImpl::calculateEdgeLogLikelihoods(const int * parentBufferIndices,
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
    // TODO: implement calculateEdgeLnL for count > 1
    // TODO: test calculateEdgeLnL when child is of tipStates kind
    // TODO: implement derivatives for calculateEdgeLnL
    // TODO: implement rate categories for calculateEdgeLnL
    
    assert(firstDerivativeIndices == 0L);
    assert(secondDerivativeIndices == 0L);
    assert(outFirstDerivatives == 0L);
    assert(outSecondDerivatives == 0L);
    
    assert(count == 1);
    
    int parIndex = parentBufferIndices[0];
    int childIndex = childBufferIndices[0];
    int probIndex = probabilityIndices[0];
    
    assert(parIndex >= kTipCount);
    
    const double* partialsParent = partials[parIndex];
    const std::vector<double> transMatrix = transitionMatrices[probIndex];
    const double* wt = inWeights;    
    
    if (childIndex < kTipCount && tipStates[childIndex]) {
        const int* statesChild = tipStates[childIndex];
        int v = 0;
        for (int k = 0; k < kPatternCount; k++) {
            int stateChild = statesChild[k];
            double sumK = 0.0;
            for (int i = 0; i < kStateCount; i++) {
                int w = i * kStateCount + 1;
                for (int l = 0; l < kCategoryCount; l++) {
                    sumK += inStateFrequencies[i] * partialsParent[v + i + l * kPatternCount * kStateCount] * transMatrix[w + stateChild + l * kMatrixSize] * wt[l];
                }
            }
            outLogLikelihoods[k] = log(sumK);
            v += kStateCount;
        }
    } else {
        const double* partialsChild = partials[childIndex];
        
        int v = 0;
        for (int k = 0; k < kPatternCount; k++) {
            int w = 0;
            double sumK = 0.0;
            for (int i = 0; i < kStateCount; i++) {
                double sumI[kCategoryCount];
                for (int l = 0; l < kCategoryCount; l++)
                    sumI[l] = 0.0;
                for (int j = 0; j < kStateCount; j++) {
                    for (int l = 0; l < kCategoryCount; l++) {
                        sumI[l] += transMatrix[w + l * kMatrixSize] * partialsChild[v + j+ (l * kPatternCount * kStateCount)];
                    }
                    w++;
                }
                w++;    // increment for the extra column at the end
                for (int l = 0; l < kCategoryCount; l++) {
                    sumK += inStateFrequencies[i] * partialsParent[v + i + l * kPatternCount * kStateCount] * sumI[l] * wt[l];
                }
            }
            outLogLikelihoods[k] = log(sumK);
            v += kStateCount;
        }
    }
    
    return BEAGLE_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
// private methods

/*
 * Calculates partial likelihoods at a node when both children have states.
 */
void BeagleCPUImpl::calcStatesStates(double* destP,
                                     const int* child1States,
                                     const double* child1TransMat,
                                     const int* child2States,
                                     const double*child2TransMat) {

    int v = 0;
    for (int l = 0; l < kCategoryCount; l++) {
        for (int k = 0; k < kPatternCount; k++) {
            const int state1 = child1States[k];
            const int state2 = child2States[k];
            if (DEBUGGING_OUTPUT) {
                std::cerr << "calcStatesStates s1 = " << state1 << '\n';
                std::cerr << "calcStatesStates s2 = " << state2 << '\n';
            }
            int w = l * kMatrixSize;
            for (int i = 0; i < kStateCount; i++) {
                destP[v] = child1TransMat[w + state1] * child2TransMat[w + state2];
                v++;
                w += (kStateCount + 1);
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
            std::cerr << "calcStatesPartials s1 = " << state1 << '\n';
            int w = l * kMatrixSize;
            for (int i = 0; i < kStateCount; i++) {
                double tmp = matrices1[w + state1];
                double sum = 0.0;
                for (int j = 0; j < kStateCount; j++) {
                    sum += matrices2[w] * partials2[v + j];
                    w++;
                }
                // increment for the extra column at the end
                w++;
                destP[u] = tmp * sum;
                u++;
            }
            v += kStateCount;
        }
    }
}

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
                // increment for the extra column at the end
                w++;
                destP[u] = sum1 * sum2;
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
                                             int scaleBufferCount) {
    BeagleImpl* impl = new BeagleCPUImpl();
    
    try {
        if (impl->createInstance(tipCount, partialsBufferCount, compactBufferCount, stateCount,
                                 patternCount, eigenBufferCount, matrixBufferCount,
                                 categoryCount,scaleBufferCount) == 0)
            return impl;
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

