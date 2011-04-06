/*
 * EigenDecompositionCube.h
 *
 *  Created on: Sep 24, 2009
 *      Author: msuchard
 */

#ifndef EIGENDECOMPOSITIONCUBE_H_
#define EIGENDECOMPOSITIONCUBE_H_

#include "libhmsbeagle/CPU/EigenDecomposition.h"

namespace beagle {
namespace cpu {

template <class REALTYPE>
class EigenDecompositionCube : public EigenDecomposition<REALTYPE> {

	using EigenDecomposition<REALTYPE>::gEigenValues;
	using EigenDecomposition<REALTYPE>::kStateCount;
	using EigenDecomposition<REALTYPE>::kEigenDecompCount;
	using EigenDecomposition<REALTYPE>::kCategoryCount;
	using EigenDecomposition<REALTYPE>::matrixTmp;
	using EigenDecomposition<REALTYPE>::firstDerivTmp;
	using EigenDecomposition<REALTYPE>::secondDerivTmp;
	using EigenDecomposition<REALTYPE>::kFlags;

protected:
    REALTYPE** gCMatrices;

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
	
};

}
}

// Include the template implementation
#include "libhmsbeagle/CPU/EigenDecompositionCube.hpp"

#endif /* EIGENDECOMPOSITIONCUBE_H_ */
