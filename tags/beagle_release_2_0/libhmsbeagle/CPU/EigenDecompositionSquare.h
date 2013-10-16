/*
 * EigenDecompositionSquare.h
 *
 *  Created on: Sep 24, 2009
 *      Author: msuchard
 */

#ifndef EIGENDECOMPOSITIONSQUARE_H_
#define EIGENDECOMPOSITIONSQUARE_H_

#include "EigenDecomposition.h"

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
    REALTYPE** gEMatrices; // kStateCount^2 flattened array
    REALTYPE** gIMatrices; // kStateCount^2 flattened array
    bool isComplex;
    int kEigenValuesSize;

public:
	EigenDecompositionSquare(int decompositionCount,
						     int stateCount,
						     int categoryCount,
						     long flags);

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
};

}
}

// Include the template implementation header
#include "libhmsbeagle/CPU/EigenDecompositionSquare.hpp"

#endif /* EIGENDECOMPOSITIONSQUARE_H_ */
