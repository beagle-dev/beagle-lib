/*
 * @file BeagleGPUImpl.h
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
 * 
 * @brief GPU implementation header
 *
 * @author Marc Suchard
 * @author Andrew Rambaut
 * @author Daniel Ayres
 */

#ifndef __BeagleGPUImpl__
#define __BeagleGPUImpl__

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include "libhmsbeagle/BeagleImpl.h"
#include "libhmsbeagle/GPU/GPUImplDefs.h"
#include "libhmsbeagle/GPU/GPUInterface.h"
#include "libhmsbeagle/GPU/KernelLauncher.h"

#define BEAGLE_GPU_GENERIC	Real
#define BEAGLE_GPU_TEMPLATE	template <typename Real>

namespace beagle {
namespace gpu {

BEAGLE_GPU_TEMPLATE
class BeagleGPUImpl : public BeagleImpl {
private:
    GPUInterface* gpu;
    KernelLauncher* kernels;
    
    int kInitialized;
    
    long kFlags;
    
    int kTipCount;
    int kPartialsBufferCount;
    int kCompactBufferCount;
    int kStateCount;
    int kPatternCount;
    int kEigenDecompCount;
    int kMatrixCount;
    int kCategoryCount;
 
    int kTipPartialsBufferCount;
    int kInternalPartialsBufferCount;
    int kBufferCount;
    int kScaleBufferCount;
    
    int kPaddedStateCount;
    int kPaddedPatternCount;    // total # of patterns with padding so that kPaddedPatternCount
                                //   * kPaddedStateCount is a multiple of 16
    int kSumSitesBlockCount;
    
    int kPartialsSize;
    int kMatrixSize;
    int kEigenValuesSize;
    int kScaleBufferSize;

    int kLastCompactBufferIndex;
    int kLastTipPartialsBufferIndex;
    
    GPUPtr dIntegrationTmp;
    GPUPtr dOutFirstDeriv;
    GPUPtr dOutSecondDeriv;
    GPUPtr dPartialsTmp;
    GPUPtr dFirstDerivTmp;
    GPUPtr dSecondDerivTmp;
    
    GPUPtr dSumLogLikelihood;
    GPUPtr dSumFirstDeriv;
    GPUPtr dSumSecondDeriv;
    
    GPUPtr dPatternWeights;    
	
    GPUPtr dBranchLengths;
    
    GPUPtr dDistanceQueue;
    
    GPUPtr dPtrQueue;
	
    GPUPtr dMaxScalingFactors;
    GPUPtr dIndexMaxScalingFactors;
    
    GPUPtr dAccumulatedScalingFactors;
    
    GPUPtr* dEigenValues;
    GPUPtr* dEvec;
    GPUPtr* dIevc;
    
    GPUPtr* dWeights;
    GPUPtr* dFrequencies; 

    GPUPtr* dScalingFactors;
    
    GPUPtr* dStates;
    
    GPUPtr* dPartials;
    GPUPtr* dMatrices;
    
    GPUPtr* dCompactBuffers;
    GPUPtr* dTipPartialsBuffers;
    
    unsigned int* hPtrQueue;
    
    double* hCategoryRates; // Can keep in double-precision

    Real* hPatternWeightsCache;
        
    Real* hDistanceQueue;
    
    Real* hWeightsCache;
    Real* hFrequenciesCache;
    Real* hLogLikelihoodsCache;
    Real* hPartialsCache;
    int* hStatesCache;
    Real* hMatrixCache;
    
    int* hRescalingTrigger;
    GPUPtr dRescalingTrigger;
    
    GPUPtr* dScalingFactorsMaster;
    
public:    
    BeagleGPUImpl();
    
    virtual ~BeagleGPUImpl();
    
    int createInstance(int tipCount,
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
                       long requirementFlags);
    
    int getInstanceDetails(BeagleInstanceDetails* retunInfo);

    int setTipStates(int tipIndex,
                     const int* inStates);

    int setTipPartials(int tipIndex,
                       const double* inPartials);
    
    int setPartials(int bufferIndex,
                    const double* inPartials);
    
    int getPartials(int bufferIndex,
				    int scaleIndex,
                    double* outPartials);
        
    int setEigenDecomposition(int eigenIndex,
                              const double* inEigenVectors,
                              const double* inInverseEigenVectors,
                              const double* inEigenValues);
    
    int setStateFrequencies(int stateFrequenciesIndex,
                            const double* inStateFrequencies);    
    
    int setCategoryWeights(int categoryWeightsIndex,
                           const double* inCategoryWeights);
    
    int setPatternWeights(const double* inPatternWeights);
    
    
    int setCategoryRates(const double* inCategoryRates);
    
    int setTransitionMatrix(int matrixIndex,
                            const double* inMatrix,
                            double paddedValue);
    
    int setTransitionMatrices(const int* matrixIndices,
                              const double* inMatrices,
                              const double* paddedValues,
                              int count);    
    
    int getTransitionMatrix(int matrixIndex,
                            double* outMatrix);

    ///////////////////////////
    //---TODO: Epoch model---//
    ///////////////////////////

	int convolveTransitionMatrices(const int* firstIndices,
                                   const int* secondIndices,
                                   const int* resultIndices,
                                   int matrixCount);

    int updateTransitionMatrices(int eigenIndex,
                                 const int* probabilityIndices,
                                 const int* firstDerivativeIndices,
                                 const int* secondDerivativeIndices,
                                 const double* edgeLengths,
                                 int count);
    
    int updatePartials(const int* operations,
                       int operationCount,
                       int cumulativeScalingIndex);
    
    int waitForPartials(const int* destinationPartials,
                        int destinationPartialsCount);
    
    int accumulateScaleFactors(const int* scalingIndices,
                               int count,
                               int cumulativeScalingIndex);
    
    int removeScaleFactors(const int* scalingIndices,
                           int count,
                           int cumulativeScalingIndex);
    
    int resetScaleFactors(int cumulativeScalingIndex);
    
    int copyScaleFactors(int destScalingIndex,
                         int srcScalingIndex);
    
    int calculateRootLogLikelihoods(const int* bufferIndices,
                                    const int* categoryWeightsIndices,
                                    const int* stateFrequenciesIndices,
                                    const int* cumulativeScaleIndices,
                                    int count,
                                    double* outSumLogLikelihood);
    
    int calculateEdgeLogLikelihoods(const int* parentBufferIndices,
                                    const int* childBufferIndices,
                                    const int* probabilityIndices,
                                    const int* firstDerivativeIndices,
                                    const int* secondDerivativeIndices,
                                    const int* categoryWeightsIndices,
                                    const int* stateFrequenciesIndices,
                                    const int* cumulativeScaleIndices,
                                    int count,
                                    double* outSumLogLikelihood,
                                    double* outSumFirstDerivative,
                                    double* outSumSecondDerivative);

    int getSiteLogLikelihoods(double* outLogLikelihoods);
    
    int getSiteDerivatives(double* outFirstDerivatives,
                           double* outSecondDerivatives);

private:
    char* getInstanceName();

};

BEAGLE_GPU_TEMPLATE
class BeagleGPUImplFactory : public BeagleImplFactory {
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
                                   int* errorCode);

    virtual const char* getName();
    virtual const long getFlags();
};

template <typename Real>
void modifyFlagsForPrecision(long* flags, Real r);

}	// namespace gpu
}	// namespace beagle

#include "libhmsbeagle/GPU/BeagleGPUImpl.hpp"

#endif // __BeagleGPUImpl__
