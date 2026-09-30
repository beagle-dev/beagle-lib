/*
 * EigenDecompositionCube.h
 *
 *  Created on: Sep 24, 2009
 *      Author: msuchard
 */

#ifndef EIGENDECOMPOSITIONCUBE_H_
#define EIGENDECOMPOSITIONCUBE_H_

#include <memory>
#include <vector>

#include "libhmsbeagle/CPU/EigenDecomposition.h"
#include "libhmsbeagle/CPU/EigenDecompositionSpectral.h"

namespace beagle {
namespace cpu {

BEAGLE_CPU_EIGEN_TEMPLATE
class EigenDecompositionCube : public EigenDecomposition<BEAGLE_CPU_EIGEN_GENERIC> {

	using EigenDecomposition<BEAGLE_CPU_EIGEN_GENERIC>::gEigenValues;
	using EigenDecomposition<BEAGLE_CPU_EIGEN_GENERIC>::kStateCount;
	using EigenDecomposition<BEAGLE_CPU_EIGEN_GENERIC>::kEigenDecompCount;
	using EigenDecomposition<BEAGLE_CPU_EIGEN_GENERIC>::kCategoryCount;
	using EigenDecomposition<BEAGLE_CPU_EIGEN_GENERIC>::matrixTmp;
	using EigenDecomposition<BEAGLE_CPU_EIGEN_GENERIC>::firstDerivTmp;
	using EigenDecomposition<BEAGLE_CPU_EIGEN_GENERIC>::secondDerivTmp;
	using EigenDecomposition<BEAGLE_CPU_EIGEN_GENERIC>::kFlags;

protected:
    REALTYPE** gCMatrices;

    // V, V^{-1}, their transposes and the integral plan in the layout of the spectral representation, which
    // BeagleCPUImpl's adjoint gradient reads. Built on the first gradient after setEigenDecomposition
    // (prepareAdjoint) from copies of that call's arguments: V, V^{-1} as given, then the eigenvalues.
    std::unique_ptr<EigenDecompositionSpectral<BEAGLE_CPU_EIGEN_GENERIC>> gAdjointStorage;
    std::vector<std::vector<double>> gAdjointInput;
    std::vector<bool> gAdjointStale;

public:
	EigenDecompositionCube(int decompositionCount, 
						   int stateCount, 
						   int categoryCount,
                           long flags);
	
	virtual ~EigenDecompositionCube();
	
    virtual void setEigenDecomposition(int eigenIndex,
                              const double* inEigenVectors,
                              const double* inInverseEigenVectors,
                              const double* inEigenValues);
		
    virtual void updateTransitionMatrices(int eigenIndex,
                                 const int* probabilityIndices,
                                 const int* firstDerivativeIndices,
                                 const int* secondDerivativeIndices,
                                 const double* edgeLengths,
                                 const double* categoryRates,
                                 REALTYPE** transitionMatrices,
                                 int count);
	
    virtual void updateTransitionMatricesWithModelCategories(int* eigenIndices,
                                 const int* probabilityIndices,
                                 const int* firstDerivativeIndices,
                                 const int* secondDerivativeIndices,
                                 const double* edgeLengths,
                                 REALTYPE** transitionMatrices,
                                 int count);

    virtual const REALTYPE* getEigenValuesPtr(int eigenIndex) const {
        return gEigenValues[eigenIndex];
    }

    virtual const REALTYPE* getEigenVectorsPtr(int eigenIndex) const {
        return gAdjointStorage ? gAdjointStorage->getEigenVectorsPtr(eigenIndex) : nullptr;
    }

    virtual const REALTYPE* getInverseEigenVectorsPtr(int eigenIndex) const {
        return gAdjointStorage ? gAdjointStorage->getInverseEigenVectorsPtr(eigenIndex) : nullptr;
    }

    virtual const REALTYPE* getBackwardsEigenVectorsPtr(int eigenIndex) const {
        return gAdjointStorage ? gAdjointStorage->getBackwardsEigenVectorsPtr(eigenIndex) : nullptr;
    }

    virtual const REALTYPE* getBackwardsInverseEigenVectorsPtr(int eigenIndex) const {
        return gAdjointStorage ? gAdjointStorage->getBackwardsInverseEigenVectorsPtr(eigenIndex) : nullptr;
    }

    virtual AdjointIntegralPlan<REALTYPE>* getAdjointMethodsPtr(int eigenIndex) const {
        return gAdjointStorage ? gAdjointStorage->getAdjointMethodsPtr(eigenIndex) : nullptr;
    }

    virtual void prepareAdjoint(int eigenIndex);
};

}
}

// Include the template implementation
#include "libhmsbeagle/CPU/EigenDecompositionCube.hpp"

#endif /* EIGENDECOMPOSITIONCUBE_H_ */
