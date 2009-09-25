/*
 * EigenDecompositionCube.h
 *
 *  Created on: Sep 24, 2009
 *      Author: msuchard
 */

#ifndef EIGENDECOMPOSITIONCUBE_H_
#define EIGENDECOMPOSITIONCUBE_H_

#include "EigenDecomposition.h"

namespace beagle {
namespace cpu {

class EigenDecompositionCube : public EigenDecomposition {

protected:
    double** gCMatrices;
//    double** gEMatrices; // kStateCount^2 flattened array  
//    double** gIMatrices; // kStateCount^2 flattened array
         
    double* matrixTmp;
public:
	EigenDecompositionCube(int decompositionCount, 
						   int stateCount, 
						   int categoryCount);
	
	virtual ~EigenDecompositionCube();
	
    virtual void setEigenDecomposition(int eigenIndex,
                              const double* inEigenVectors,
                              const double* inInverseEigenVectors,
                              const double* inEigenValues);
		
    virtual void updateTransitionMatrices(int eigenIndex,
                                 const int* probabilityIndices,
                                 const int* firstDerivativeIndices,
                                 const int* secondDervativeIndices,
                                 const double* edgeLengths,
                                 const double* categoryRates,
                                 double** transitionMatrices,                                 
                                 int count);
	
};

}
}

#endif /* EIGENDECOMPOSITIONCUBE_H_ */
