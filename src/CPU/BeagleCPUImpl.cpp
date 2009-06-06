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
///@TODO: reimplement calculateEdgeLogLikelihoods()
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

///@API-ISSUE: adding an resizePartialBufferArray(int newPartialBufferCount) method
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




#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <math.h>

#include "beagle.h"
#include "BeagleCPUImpl.h"

#if defined (BEAGLE_IMPL_DEBUGGING_OUTPUT) && BEAGLE_IMPL_DEBUGGING_OUTPUT
    const bool DEBUGGING_OUTPUT = true;
#else
    const bool DEBUGGING_OUTPUT = false;
#endif

BeagleCPUImpl::~BeagleCPUImpl() {
    // free all that stuff...
    // If you delete partials, make sure not to delete the last element
    // which is TEMP_SCRATCH_PARTIAL twice.
}

int BeagleCPUImpl::initialize(int tipCount,
                              int partialBufferCount,
                              int compactBufferCount,
                              int stateCount,
                              int patternCount,
                              int eigenDecompositionCount,
                              int matrixCount) {
    if (DEBUGGING_OUTPUT)
        std::cerr << "in BeagleCPUImpl::initialize\n" ;
    
    kBufferCount = partialBufferCount + compactBufferCount;
    kTipCount = tipCount;
    assert(kBufferCount > kTipCount);
    kStateCount = stateCount;
    kPatternCount = patternCount;
    kMatrixCount = matrixCount;
    kEigenDecompCount = eigenDecompositionCount;

    kMatrixSize = (1 + stateCount) * stateCount;

    cMatrices = (double**) malloc(sizeof(double*) * eigenDecompositionCount);
    if (cMatrices == 0L)
        throw std::bad_alloc();

    eigenValues = (double**) malloc(sizeof(double*) * eigenDecompositionCount);
    if (eigenValues == 0L)
        throw std::bad_alloc();

    for (int i = 0; i < eigenDecompositionCount; i++) {
        cMatrices[i] = (double*) malloc(sizeof(double) * stateCount * stateCount * stateCount);
        if (cMatrices[i] == 0L)
            throw std::bad_alloc();
        
        eigenValues[i] = (double*) malloc(sizeof(double) * stateCount);
        if (eigenValues[i] == 0L)
            throw std::bad_alloc();
    }


    kPartialsSize = kPatternCount * stateCount;

    partials.assign(kBufferCount, 0L);
    tipStates.assign(kTipCount, 0L);

    for (int i = kTipCount; i < kBufferCount; i++) {
        partials[i] = (double*) malloc(sizeof(double) * kPartialsSize);
        if (partials[i] == 0L)
            throw std::bad_alloc();
    }

    std::vector<double> emptyMat(kMatrixSize);
    transitionMatrices.assign(kMatrixCount, emptyMat);

    ////BEGIN edge Like Hack
    /// used in our hack edgeLike func
    TEMP_SCRATCH_PARTIAL = (double*) malloc(sizeof(double) * kPartialsSize); 
    partials.push_back(TEMP_SCRATCH_PARTIAL);
    TEMP_IDENTITY_MATRIX.assign(kStateCount * (kStateCount + 1), 0.0);
    unsigned el = 0;
    for (unsigned diag = 0 ; diag < kStateCount + 1; ++diag)
        {
        TEMP_IDENTITY_MATRIX[el] = 1.0;
        //  kStateCount is the cell below if we  weren't padding
        //  kStateCount + 1 is the cell below, if we  weren't padding
        //  kStateCount + 2 is the cell below and to the right. (this keeps us on the diag.)
        el += kStateCount + 2;
        }
    ////END edge Like Hack

    return NO_ERROR;
}

int BeagleCPUImpl::setPartials(int bufferIndex,
                               const double* inPartials) {
    assert(partials[bufferIndex] == 0L);
    partials[bufferIndex] = (double*) malloc(sizeof(double) * kPartialsSize);
    if (partials[bufferIndex] == 0L)
        return OUT_OF_MEMORY_ERROR;
    memcpy(partials[bufferIndex], inPartials, sizeof(double) * kPartialsSize);
    
    return NO_ERROR;
}

int BeagleCPUImpl::getPartials(int bufferIndex,
                               double* outPartials) {
    memcpy(outPartials, partials[bufferIndex], sizeof(double) * kPartialsSize);
    
    return NO_ERROR;
}

int BeagleCPUImpl::setTipStates(int tipIndex,
                                const int* inStates) {
    tipStates[tipIndex] = (int*) malloc(sizeof(int) * kPatternCount);
    for (int i = 0; i < kPatternCount; i++) {
        tipStates[tipIndex][i] = (inStates[i] < kStateCount ? inStates[i]
                : kStateCount);
    }
    
    return NO_ERROR;
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
    
    return NO_ERROR;
}

int BeagleCPUImpl::setTransitionMatrix(int matrixIndex,
                                       const double* inMatrix) {
    assert(false);
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
        
        for (int i = 0; i < kStateCount; i++) {
            tmp[i] = exp(eigenValues[eigenIndex][i] * edgeLengths[u]);
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

        if (DEBUGGING_OUTPUT) {
            printf("transitionMat index=%d brlen=%.5f\n", probabilityIndices[u], edgeLengths[u]);
            for ( int w = 0; w < 20; ++w)
                printf("transitionMat[%d] = %.5f\n", w, transitionMat[w]);
        }
    }

    return NO_ERROR;
}

int BeagleCPUImpl::updatePartials(const int* operations,
                                  int count,
                                  int rescale) {
    for (int op = 0; op < count; op++) {
        if (DEBUGGING_OUTPUT) {
            std::cerr << "op[0]= " << operations[0] << "\n";
            std::cerr << "op[1]= " << operations[1] << "\n";
            std::cerr << "op[2]= " << operations[2] << "\n";
            std::cerr << "op[3]= " << operations[3] << "\n";
            std::cerr << "op[4]= " << operations[4] << "\n";
        }
        const int parIndex = operations[op * 5];
        const int child1Index = operations[op * 5 + 1];
        const int child1TransMatIndex = operations[op * 5 + 2];
        const int child2Index = operations[op * 5 + 3];
        const int child2TransMatIndex = operations[op * 5 + 4];

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
    
    return NO_ERROR;
}


int BeagleCPUImpl::waitForPartials(const int* destinationPartials,
                                   int destinationPartialsCount) {
    return NO_ERROR;
}


int BeagleCPUImpl::calculateRootLogLikelihoods(const int* bufferIndices,
                                               const double* weights,
                                               const double* stateFrequencies,
                                               int count,
                                               double* outLogLikelihoods) {
    if (count == 1) {
        // We treat this as a special case so that we don't have convoluted logic
        //  at the end of the loop over patterns
        const int rootPartialIndex = bufferIndices[0];
        const double * rootPartials = partials[rootPartialIndex];
        assert(rootPartials);
        const double wt = weights[0];
        assert(wt >= 0.0);
        int v = 0;
        for (int k = 0; k < kPatternCount; k++) {
            double sum = 0.0;
            for (int i = 0; i < kStateCount; i++) {
                sum += stateFrequencies[i] * rootPartials[v];
                v++;
            }
            outLogLikelihoods[k] = log(sum * wt);
        }
    } else {
        // Here we do the 3 similar operations:
        //      1. to set the lnL to the contribution of the first subset,
        //      2. to add the lnL for other subsets up to the penultimate
        //      3. to add the final subset and take the lnL
        //  This form of the calc would not work when count == 1 because
        //      we need operation 1 and 3 in the preceding list.  This is not
        //      a problem, though as we deal with count == 1 in the previous
        //      branch.
        for (int subsetIndex = 0 ; subsetIndex < count; ++subsetIndex) {
            assert(subsetIndex < partials.size());
            const int rootPartialIndex = bufferIndices[subsetIndex];
            const double * rootPartials = partials[rootPartialIndex];
            assert(rootPartials);
            const double * frequencies = stateFrequencies + (subsetIndex * kStateCount);
            const double wt = weights[subsetIndex];
            assert(wt >= 0.0);
            int v = 0;
            for (int k = 0; k < kPatternCount; k++) {
                double sum = 0.0;
                for (int i = 0; i < kStateCount; i++) {
                    sum += frequencies[i] * rootPartials[v];
                    v++;
                }
                if (subsetIndex == 0)
                    outLogLikelihoods[k] = sum * wt;
                else if (subsetIndex == count - 1)
                     // add and take the log
                    outLogLikelihoods[k] = log(outLogLikelihoods[k] + sum * wt);
                else
                    outLogLikelihoods[k] += sum * wt;
            }
        }
    }

    return NO_ERROR;
}

int BeagleCPUImpl::calculateEdgeLogLikelihoods(const int * parentBufferIndices,
                                               const int* childBufferIndices,
                                               const int* probabilityIndices,
                                               const int* firstDerivativeIndices,
                                               const int* secondDerivativeIndices,
                                               const double* weights,
                                               const double* stateFrequencies,
                                               int count,
                                               double* outLogLikelihoods,
                                               double* outFirstDerivatives,
                                               double* outSecondDerivatives) {
    //@@@@ this impl is a real hack !!!!!!!!
    assert(firstDerivativeIndices == 0L);
    assert(secondDerivativeIndices == 0L);
    assert(outFirstDerivatives == 0L);
    assert(outSecondDerivatives == 0L);

    assert(count == 1);
    int parIndex = parentBufferIndices[0];
    int childIndex = childBufferIndices[0];
    if (parIndex < childIndex) {
        childIndex = parIndex;
        parIndex = childBufferIndices[0];
    }

    assert(parIndex >= kTipCount);

    memcpy(TEMP_SCRATCH_PARTIAL, partials[parIndex], sizeof(double) * kPartialsSize);
    const double* fakeEdgeMat = &TEMP_IDENTITY_MATRIX[0];
    const std::vector<double> & realMat = transitionMatrices[probabilityIndices[0]];
    const double* edgeTransMat = &(realMat[0]);

    if (DEBUGGING_OUTPUT) {
        for ( int w = 0; w < 20; ++w)
            printf("edgeTransMat[%d] = %.5f\n", w, edgeTransMat[w]);
    }

    if (childIndex < kTipCount && tipStates[childIndex] ) {
        calcStatesPartials(TEMP_SCRATCH_PARTIAL, tipStates[childIndex], edgeTransMat,
                           partials[parIndex], fakeEdgeMat);
    } else {
        calcPartialsPartials(TEMP_SCRATCH_PARTIAL, partials[childIndex], edgeTransMat,
                             partials[parIndex], fakeEdgeMat);
    }

    int c = partials.size() - 1;
    assert(partials[c] == TEMP_SCRATCH_PARTIAL);
    
    return calculateRootLogLikelihoods(&c, weights, stateFrequencies, 1, outLogLikelihoods);
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
    for (int k = 0; k < kPatternCount; k++) {
        const int state1 = child1States[k];
        const int state2 = child2States[k];
        if (DEBUGGING_OUTPUT) {
            std::cerr << "calcStatesStates s1 = " << state1 << '\n';
            std::cerr << "calcStatesStates s2 = " << state2 << '\n';
        }
        int w = 0;
        for (int i = 0; i < kStateCount; i++) {
            destP[v] = child1TransMat[w + state1] * child2TransMat[w + state2];
            v++;
            w += (kStateCount + 1);
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
    for (int k = 0; k < kPatternCount; k++) {
        int state1 = states1[k];
        //std::cerr << "calcStatesPartials s1 = " << state1 << '\n';
        int w = 0;
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

void BeagleCPUImpl::calcPartialsPartials(double* destP,
                                         const double* partials1,
                                         const double* matrices1,
                                         const double* partials2,
                                         const double* matrices2) {
    double sum1, sum2;
    int u = 0;
    int v = 0;
    for (int k = 0; k < kPatternCount; k++) {
        int w = 0;
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


///////////////////////////////////////////////////////////////////////////////
// BeagleCPUImplFactory public methods

BeagleImpl* BeagleCPUImplFactory::createImpl(int tipCount,   
                                             int partialsBufferCount,
                                             int compactBufferCount,
                                             int stateCount,
                                             int patternCount,
                                             int eigenBufferCount,
                                             int matrixBufferCount) {
    BeagleImpl* impl = new BeagleCPUImpl();

    try {
        if (impl->initialize(tipCount, partialsBufferCount, compactBufferCount, stateCount,
                             patternCount, eigenBufferCount, matrixBufferCount) == 0)
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

