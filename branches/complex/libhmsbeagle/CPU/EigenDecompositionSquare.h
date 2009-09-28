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

class EigenDecompositionSquare: public beagle::cpu::EigenDecomposition {

protected:
    double** gEMatrices; // kStateCount^2 flattened array
    double** gIMatrices; // kStateCount^2 flattened array
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
                                 const int* secondDervativeIndices,
                                 const double* edgeLengths,
                                 const double* categoryRates,
                                 double** transitionMatrices,
                                 int count);
};

}
}

#endif /* EIGENDECOMPOSITIONSQUARE_H_ */
