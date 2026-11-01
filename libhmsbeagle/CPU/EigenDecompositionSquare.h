/*
 * EigenDecompositionSquare.h
 *
 *  Created on: Sep 24, 2009
 *      Author: msuchard
 */

#ifndef EIGENDECOMPOSITIONSQUARE_H_
#define EIGENDECOMPOSITIONSQUARE_H_

#include <memory>
#include <vector>

#include "EigenDecomposition.h"
#include "libhmsbeagle/CPU/EigenDecompositionSpectral.h"

namespace beagle {
namespace cpu {

BEAGLE_CPU_EIGEN_TEMPLATE
class EigenDecompositionSquare: public EigenDecomposition<BEAGLE_CPU_EIGEN_GENERIC> {

	using EigenDecomposition<BEAGLE_CPU_EIGEN_GENERIC>::gEigenValues;
	using EigenDecomposition<BEAGLE_CPU_EIGEN_GENERIC>::kStateCount;
	using EigenDecomposition<BEAGLE_CPU_EIGEN_GENERIC>::kEigenDecompCount;
	using EigenDecomposition<BEAGLE_CPU_EIGEN_GENERIC>::kCategoryCount;
	using EigenDecomposition<BEAGLE_CPU_EIGEN_GENERIC>::matrixTmp;
	using EigenDecomposition<BEAGLE_CPU_EIGEN_GENERIC>::kFlags;

protected:
    REALTYPE** gEMatrices;
    REALTYPE** gIMatrices;
    REALTYPE** gRawIMatrices;
    bool isComplex;
    int kEigenValuesSize;

    // V, V^{-1}, their transposes and the integral plan in the layout of the spectral representation, which
    // BeagleCPUImpl's adjoint gradient reads; built from gEMatrices, gIMatrices and gEigenValues on the first
    // gradient after setEigenDecomposition (prepareAdjoint)
    std::unique_ptr<EigenDecompositionSpectral<BEAGLE_CPU_EIGEN_GENERIC>> gAdjointStorage;
    std::vector<bool> gAdjointStale;

public:
	EigenDecompositionSquare(int decompositionCount,
						     int stateCount,
						     int categoryCount,
						     long long flags);

	virtual ~EigenDecompositionSquare();

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

	virtual void updateTransitionMatricesParallel(int eigenIndex, const int* probabilityIndices,
	                                      const int* firstDerivativeIndices,
	                                      const int* secondDerivativeIndices, const double* edgeLengths,
	                                      const double* categoryRates, REALTYPE** transitionMatrices, int count,
	                                      int numThreads);

    virtual void updateTransitionMatricesWithModelCategories(int* eigenIndices,
                                 const int* probabilityIndices,
                                 const int* firstDerivativeIndices,
                                 const int* secondDerivativeIndices,
                                 const double* edgeLengths,
                                 REALTYPE** transitionMatrices,
                                 int count);

    virtual const REALTYPE* getEigenValuesPtr(int eigenIndex) const;

    // in the spectral layout (rows of stride kStateCount + T_PAD), not gEMatrices / gIMatrices
    virtual const REALTYPE* getEigenVectorsPtr(int eigenIndex) const;

    virtual const REALTYPE* getInverseEigenVectorsPtr(int eigenIndex) const;

    virtual const REALTYPE* getBackwardsEigenVectorsPtr(int eigenIndex) const;

    virtual const REALTYPE* getBackwardsInverseEigenVectorsPtr(int eigenIndex) const;

    virtual AdjointIntegralPlan<REALTYPE>* getAdjointMethodsPtr(int eigenIndex) const;

    virtual void prepareAdjoint(int eigenIndex);

	virtual int getRawEigenDecomposition(int eigenIndex,
	                                     const REALTYPE** outEigenVectors,
	                                     const REALTYPE** outInverseEigenVectors,
	                                     const REALTYPE** outEigenValues);
};

}
}

// Include the template implementation header
#include "libhmsbeagle/CPU/EigenDecompositionSquare.hpp"

#endif /* EIGENDECOMPOSITIONSQUARE_H_ */
