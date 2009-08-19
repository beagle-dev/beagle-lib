/*
 *  BeagleCPUImpl.h
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
 */

#ifndef __BeagleCPUImpl__
#define __BeagleCPUImpl__

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include "libhmsbeagle/BeagleImpl.h"

#include <vector>

namespace beagle {
namespace cpu {

class BeagleCPUImpl : public BeagleImpl {

protected:
    int kBufferCount; /// after initialize this will be partials.size()
    ///   (we don't really need this field)
    int kTipCount; /// after initialize this will be tipStates.size()
    ///   (we don't really need this field, but it is handy)
    int kPatternCount; /// the number of data patterns in each partial and tipStates element
    int kMatrixCount; /// the number of transition matrices to alloc and store
    int kStateCount; /// the number of states
    int kEigenDecompCount; /// the number of eigen solutions to alloc and store
    int kCategoryCount;
    int kScaleBufferCount;

    int kPartialsSize;  /// stored for convenience. kPartialsSize = kStateCount*kPatternCount
    int kMatrixSize; /// stored for convenience. kMatrixSize = kStateCount*(kStateCount + 1)

    //@ the following eigen-calculation-related fields should be changed to vectors
    //      of vectors
    // each element of cMatrices is a kStateCount^3 flattened array to temporaries calculated
    //  from the eigenVector matrix and inverse eigen vector matrix. Storing these
    //  temps saves time in the updateTransitionMatrices()
    double** dCMatrices;
    // each element of eigenValues is a kStateCount array of eigenvalues
    double** dEigenValues;

    double* dCategoryRates;

    //@ the size of these pointers are known at alloc-time, so the partials and
    //      tipStates field should be switched to vectors of vectors (to make
    //      memory management less error prone
    std::vector<double*> dPartials;
    std::vector<int*> dTipStates;
    std::vector<double*> dScalingFactors;

    // There will be kMatrixCount transitionMatrices.
    // Each kStateCount x (kStateCount+1) matrix that is flattened
    //  into a single array
    std::vector< std::vector<double> > dTransitionMatrices;

    double* dIntegrationTmp;

    double* ones;
    double* zeros;

public:
    virtual ~BeagleCPUImpl();

    // creation of instance
    int createInstance(int tipCount,
                       int partialsBufferCount,
                       int compactBufferCount,
                       int stateCount,
                       int patternCount,
                       int eigenDecompositionCount,
                       int matrixCount,
                       int categoryCount,
                       int scaleBufferCount);

    // initialization of instance,  returnInfo can be null
    int initializeInstance(BeagleInstanceDetails* returnInfo);

    // set the states for a given tip
    //
    // tipIndex the index of the tip
    // inStates the array of states: 0 to stateCount - 1, missing = stateCount
    int setTipStates(int tipIndex,
                     const int* inStates);

    // set the partials for a given tip
    //
    // tipIndex the index of the tip
    // inPartials the array of partials, stateCount x patternCount
    int setTipPartials(int tipIndex,
                       const double* inPartials);


    int setPartials(int bufferIndex,
                    const double* inPartials);

    int getPartials(int bufferIndex,
					int scaleBuffer,
                    double* outPartials);

    // sets the Eigen decomposition for a given matrix
    //
    // matrixIndex the matrix index to update
    // eigenVectors an array containing the Eigen Vectors
    // inverseEigenVectors an array containing the inverse Eigen Vectors
    // eigenValues an array containing the Eigen Values
    int setEigenDecomposition(int eigenIndex,
                              const double* inEigenVectors,
                              const double* inInverseEigenVectors,
                              const double* inEigenValues);

    // set the vector of category rates
    //
    // categoryRates an array containing categoryCount rate scalers
    int setCategoryRates(const double* inCategoryRates);

    int setTransitionMatrix(int matrixIndex,
                            const double* inMatrix);

    // calculate a transition probability matrices for a given list of node. This will
    // calculate for all categories (and all matrices if more than one is being used).
    //
    // nodeIndices an array of node indices that require transition probability matrices
    // edgeLengths an array of expected lengths in substitutions per site
    // count the number of elements in the above arrays
    int updateTransitionMatrices(int eigenIndex,
                                 const int* probabilityIndices,
                                 const int* firstDerivativeIndices,
                                 const int* secondDervativeIndices,
                                 const double* edgeLengths,
                                 int count);

    // calculate or queue for calculation partials using an array of operations
    //
    // operations an array of triplets of indices: the two source partials and the destination
    // dependencies an array of indices specify which operations are dependent on which (optional)
    // count the number of operations
    // rescale indicate if partials should be rescaled during peeling
    int updatePartials(const int* operations,
                       int operationCount,
                       int cumulativeScalingIndex);

    // Block until all calculations that write to the specified partials have completed.
    //
    // This function is optional and only has to be called by clients that "recycle" partials.
    //
    // If used, this function must be called after an updatePartials call and must refer to
    //  indices of "destinationPartials" that were used in a previous updatePartials
    // call.  The library will block until those partials have been calculated.
    //
    // destinationPartials - List of the indices of destinationPartials that must be calculated
    //                         before the function returns
    // destinationPartialsCount - Number of destinationPartials (input)
    //
    // return error code
    int waitForPartials(const int* destinationPartials,
                        int destinationPartialsCount);


    int accumulateScaleFactors(const int* scalingIndices,
							  int count,
							  int cumulativeScalingIndex);

    int removeScaleFactors(const int* scalingIndices,
                               int count,
                               int cumulativeScalingIndex);

    int resetScaleFactors(int cumulativeScalingIndex);

    // calculate the site log likelihoods at a particular node
    //
    // rootNodeIndex the index of the root
    // outLogLikelihoods an array into which the site log likelihoods will be put
    int calculateRootLogLikelihoods(const int* bufferIndices,
                                    const double* inWeights,
                                    const double* inStateFrequencies,
                                    const int* scalingFactorsIndices,
                                    int count,
                                    double* outLogLikelihoods);

    // possible nulls: firstDerivativeIndices, secondDerivativeIndices,
    //                 outFirstDerivatives, outSecondDerivatives
    int calculateEdgeLogLikelihoods(const int* parentBufferIndices,
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
                                    double* outSecondDerivatives);

protected:
    virtual void calcStatesStates(double* destP,
                                    const int* states1,
                                    const double* matrices1,
                                    const int* states2,
                                    const double* matrices2);//,
//                                    const double* scalingFactors,
//                                    const double* cumulativeScalingBuffer,
//                                    int rescale );


    virtual void calcStatesPartials(double* destP,
                                    const int* states1,
                                    const double* matrices1,
                                    const double* partials2,
                                    const double* matrices2);//,
//                                    const double* scalingFactors,
//                                    const double* cumulativeScalingBuffer,
//                                    int rescale );

    virtual void calcPartialsPartials(double* destP,
                                    const double* partials1,
                                    const double* matrices1,
                                    const double* partials2,
                                    const double* matrices2);//,
//                                    const double* scalingFactors,
//                                    const double* cumulativeScalingBuffer,
//                                    int rescale );

    virtual void calcRootLogLikelihoods(const int bufferIndex,
                                    const double* inWeights,
                                    const double* inStateFrequencies,
                                    const int scalingFactorsIndex,
                                    double* outLogLikelihoods);


    virtual void calcStatesStatesFixedScaling(double *destP,
                                           const int *child0States,
                                        const double *child0TransMat,
                                           const int *child1States,
                                        const double *child1TransMat,
                                        const double *scaleFactors);

    virtual void calcStatesPartialsFixedScaling(double *destP,
                                             const int *child0States,
                                          const double *child0TransMat,
                                          const double *child1Partials,
                                          const double *child1TransMat,
                                          const double *scaleFactors);

    virtual void calcPartialsPartialsFixedScaling(double *destP,
                                            const double *child0States,
                                            const double *child0TransMat,
                                            const double *child1Partials,
                                            const double *child1TransMat,
                                            const double *scaleFactors);

    virtual void rescalePartials(double *destP,
                                 double *scaleFactors,
                                 double *cumulativeScaleFactors,
                                    const int  fillWithOnes);


};

class BeagleCPUImplFactory : public BeagleImplFactory {
public:
    virtual BeagleImpl* createImpl(int tipCount,
                                   int partialsBufferCount,
                                   int compactBufferCount,
                                   int stateCount,
                                   int patternCount,
                                   int eigenBufferCount,
                                   int matrixBufferCount,
                                   int categoryCount,
                                   int scaleBufferCount);

    virtual const char* getName();
};

}	// namespace cpu
}	// namespace beagle

#endif // __BeagleCPUImpl__
