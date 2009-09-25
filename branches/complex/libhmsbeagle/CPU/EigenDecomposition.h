/*
 * EigenDecomposition.h
 *
 *  Created on: Sep 24, 2009
 *      Author: msuchard
 */

#ifndef EIGENDECOMPOSITION_H_
#define EIGENDECOMPOSITION_H_

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cstring>
#include <cmath>
#include <cassert>
#include <vector>

#define PAD_MATRICES

namespace beagle {
namespace cpu {

class EigenDecomposition {
	
protected:
    double** gEigenValues;
    int kStateCount;
    int kEigenDecompCount;
    int kCategoryCount;
    double* matrixTmp;
	
public:	
	EigenDecomposition(int decompositionCount,
					   int stateCount,
					   int categoryCount) {
		
		kEigenDecompCount = decompositionCount;
		kStateCount = stateCount;
		kCategoryCount = categoryCount;
	}
	
	virtual ~EigenDecomposition() {};
	
    // sets the Eigen decomposition for a given matrix
    //
    // matrixIndex the matrix index to update
    // eigenVectors an array containing the Eigen Vectors
    // inverseEigenVectors an array containing the inverse Eigen Vectors
    // eigenValues an array containing the Eigen Values
    virtual void setEigenDecomposition(int eigenIndex,
                              const double* inEigenVectors,
                              const double* inInverseEigenVectors,
                              const double* inEigenValues) = 0;
		
    // calculate a transition probability matrices for a given list of node. This will
    // calculate for all categories (and all matrices if more than one is being used).
    //
    // nodeIndices an array of node indices that require transition probability matrices
    // edgeLengths an array of expected lengths in substitutions per site
    // count the number of elements in the above arrays
    virtual void updateTransitionMatrices(int eigenIndex,
                                 const int* probabilityIndices,
                                 const int* firstDerivativeIndices,
                                 const int* secondDervativeIndices,
                                 const double* edgeLengths,
                                 const double* categoryRates,
                                 double** transitionMatrices,                                 
                                 int count) = 0;

};

}
}

#endif /* EIGENDECOMPOSITION_H_ */
