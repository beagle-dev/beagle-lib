/*
 * EigenDecompositionCube.cpp
 *
 *  Created on: Sep 24, 2009
 *      Author: msuchard
 */

#include "EigenDecompositionCube.h"

#if defined (BEAGLE_IMPL_DEBUGGING_OUTPUT) && BEAGLE_IMPL_DEBUGGING_OUTPUT
const bool DEBUGGING_OUTPUT = true;
#else
const bool DEBUGGING_OUTPUT = false;
#endif

namespace beagle {
namespace cpu {

EigenDecompositionCube::EigenDecompositionCube(int decompositionCount, 
											   int stateCount,
											   int categoryCount) : EigenDecomposition(decompositionCount, 
																					   stateCount, 
																					   categoryCount){

    gEigenValues = (double**) malloc(sizeof(double*) * kEigenDecompCount);
    if (gEigenValues == NULL)
        throw std::bad_alloc();
    
    gCMatrices = (double**) malloc(sizeof(double*) * kEigenDecompCount);
    if (gCMatrices == NULL)
    	throw std::bad_alloc();
    
    for (int i = 0; i < kEigenDecompCount; i++) {    	
    	gCMatrices[i] = (double*) malloc(sizeof(double) * kStateCount * kStateCount * kStateCount);
    	if (gCMatrices[i] == NULL)
    		throw std::bad_alloc();
    
    	gEigenValues[i] = (double*) malloc(sizeof(double) * kStateCount);
    	if (gEigenValues[i] == NULL)
    		throw std::bad_alloc();
    }
    
    matrixTmp = (double*) malloc(sizeof(double) * kStateCount);
}

EigenDecompositionCube::~EigenDecompositionCube() {
	
	for(int i=0; i<kEigenDecompCount; i++) {
		free(gCMatrices[i]);
		free(gEigenValues[i]);
	}
	free(gCMatrices);
	free(gEigenValues);
	free(matrixTmp);
}

void EigenDecompositionCube::setEigenDecomposition(int eigenIndex,
										           const double* inEigenVectors,
                                                   const double* inInverseEigenVectors,
                                                   const double* inEigenValues) {
		
    int l = 0;
    for (int i = 0; i < kStateCount; i++) {
        gEigenValues[eigenIndex][i] = inEigenValues[i];
        for (int j = 0; j < kStateCount; j++) {
            for (int k = 0; k < kStateCount; k++) {
                gCMatrices[eigenIndex][l] = inEigenVectors[(i * kStateCount) + k]
                        * inInverseEigenVectors[(k * kStateCount) + j];
                l++;
            }
        }
    }
}

void EigenDecompositionCube::updateTransitionMatrices(int eigenIndex,
                                                      const int* probabilityIndices,
                                                      const int* firstDerivativeIndices,
                                                      const int* secondDervativeIndices,
                                                      const double* edgeLengths,
                                                      const double* categoryRates,
                                                      double** transitionMatrices,                                                      
                                                      int count) {
    for (int u = 0; u < count; u++) {
        double* transitionMat = transitionMatrices[probabilityIndices[u]];
        int n = 0;
        for (int l = 0; l < kCategoryCount; l++) {
       	
        	for (int i = 0; i < kStateCount; i++) {
        		matrixTmp[i] = exp(gEigenValues[eigenIndex][i] * edgeLengths[u] * categoryRates[l]);
        	}

            int m = 0;
            for (int i = 0; i < kStateCount; i++) {
                for (int j = 0; j < kStateCount; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < kStateCount; k++) {
                        sum += gCMatrices[eigenIndex][m] * matrixTmp[k];
                        m++;
                    }
                    if (sum > 0)
                        transitionMat[n] = sum;
                    else
                        transitionMat[n] = 0;
                    n++;
                }
#ifdef PAD_MATRICES
                transitionMat[n] = 1.0;
                n++;
#endif
            }
        }

        if (DEBUGGING_OUTPUT) {
        	int kMatrixSize = kStateCount * kStateCount;
            fprintf(stderr,"transitionMat index=%d brlen=%.5f\n", probabilityIndices[u], edgeLengths[u]);
            for ( int w = 0; w < (20 > kMatrixSize ? 20 : kMatrixSize); ++w)
                fprintf(stderr,"transitionMat[%d] = %.5f\n", w, transitionMat[w]);
        }
    }
}

} // cpu
} // beagle