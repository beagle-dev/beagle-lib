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
