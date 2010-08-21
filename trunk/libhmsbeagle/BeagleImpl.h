/*
 *  BeagleImpl.h
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

#ifndef __beagle_impl__
#define __beagle_impl__

#include "libhmsbeagle/beagle.h"

#ifdef DOUBLE_PRECISION
#define REAL    double
#else
#define REAL    float
#endif

namespace beagle {

class BeagleImpl
{
public:
    virtual ~BeagleImpl(){}
    
    virtual int createInstance(int tipCount,
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
                               long requirementFlags) = 0;
    
    virtual int getInstanceDetails(BeagleInstanceDetails* returnInfo) = 0;
    
    virtual int setTipStates(int tipIndex,
                             const int* inStates) = 0;

    virtual int setTipPartials(int tipIndex,
                               const double* inPartials) = 0;
    
    virtual int setPartials(int bufferIndex,
                            const double* inPartials) = 0;
    
    virtual int getPartials(int bufferIndex,
							int scaleIndex,
                            double* outPartials) = 0;
    
    virtual int setEigenDecomposition(int eigenIndex,
                                      const double* inEigenVectors,
                                      const double* inInverseEigenVectors,
                                      const double* inEigenValues) = 0;
    
    virtual int setStateFrequencies(int stateFrequenciesIndex,
                                  const double* inStateFrequencies) = 0;    
    
    virtual int setCategoryWeights(int categoryWeightsIndex,
                                 const double* inCategoryWeights) = 0;
    
    virtual int setPatternWeights(const double* inPatternWeights) = 0;
    
    virtual int setCategoryRates(const double* inCategoryRates) = 0;
    
    virtual int setTransitionMatrix(int matrixIndex,
                                    const double* inMatrix,
                                    double paddedValue) = 0;

    virtual int setTransitionMatrices(const int* matrixIndices,
                                      const double* inMatrices,
                                      const double* paddedValues,
                                      int count) = 0;    
    
    virtual int getTransitionMatrix(int matrixIndex,
                                    double* outMatrix) = 0;

    virtual int updateTransitionMatrices(int eigenIndex,
                                         const int* probabilityIndices,
                                         const int* firstDerivativeIndices,
                                         const int* secondDerivativeIndices,
                                         const double* edgeLengths,
                                         int count) = 0;
    
    virtual int updatePartials(const int* operations,
                               int operationCount,
                               int cumulativeScalingIndex) = 0;
    
    virtual int waitForPartials(const int* destinationPartials,
                                int destinationPartialsCount) = 0;
    
    virtual int accumulateScaleFactors(const int* scalingIndices,
									   int count,
									   int cumulativeScalingIndex) = 0;
    
    virtual int removeScaleFactors(const int* scalingIndices,
                                     int count,
                                     int cumulativeScalingIndex) = 0;   
    
    virtual int resetScaleFactors(int cumulativeScalingIndex) = 0;   
    
    virtual int copyScaleFactors(int destScalingIndex,
                                 int srcScalingIndex) = 0; 
    
    virtual int calculateRootLogLikelihoods(const int* bufferIndices,
                                            const int* categoryWeightsIndices,
                                            const int* stateFrequenciesIndices,
                                            const int* scalingFactorsIndices,
                                            int count,
                                            double* outSumLogLikelihood) = 0;
    
    virtual int calculateEdgeLogLikelihoods(const int* parentBufferIndices,
                                            const int* childBufferIndices,
                                            const int* probabilityIndices,
                                            const int* firstDerivativeIndices,
                                            const int* secondDerivativeIndices,
                                            const int* categoryWeightsIndices,
                                            const int* stateFrequenciesIndices,
                                            const int* scalingFactorsIndices,
                                            int count,
                                            double* outSumLogLikelihood,
                                            double* outSumFirstDerivative,
                                            double* outSumSecondDerivative) = 0;
    
    virtual int getSiteLogLikelihoods(double* outLogLikelihoods) = 0;
    
    virtual int getSiteDerivatives(double* outFirstDerivatives,
                                   double* outSecondDerivatives) = 0;
//protected:
    int resourceNumber;
};

class BeagleImplFactory {
public:
    virtual BeagleImpl* createImpl(int tipCount,
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
                                   int* errorCode) = 0; // pure virtual
    
    virtual const char* getName() = 0; // pure virtual
    
    virtual const long getFlags() = 0; // pure virtual
};

} // end namespace beagle

#endif // __beagle_impl__
