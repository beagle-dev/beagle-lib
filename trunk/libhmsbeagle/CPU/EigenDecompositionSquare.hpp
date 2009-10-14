/*
 * EigenDecompositionSquare.cpp
 *
 *  Created on: Sep 24, 2009
 *      Author: msuchard
 */

#include "EigenDecompositionSquare.h"
#include "libhmsbeagle/beagle.h"

#if defined (BEAGLE_IMPL_DEBUGGING_OUTPUT) && BEAGLE_IMPL_DEBUGGING_OUTPUT
const bool DEBUGGING_OUTPUT = true;
#else
const bool DEBUGGING_OUTPUT = false;
#endif

namespace beagle {
namespace cpu {

EigenDecompositionSquare::EigenDecompositionSquare(int decompositionCount,
											       int stateCount,
											       int categoryCount,
											       long flags)
	: EigenDecomposition(decompositionCount,stateCount,categoryCount) {

	isComplex = flags & BEAGLE_FLAG_COMPLEX;

	if (isComplex)
		kEigenValuesSize = 2 * kStateCount;
	else
		kEigenValuesSize = kStateCount;

    gEigenValues = (double**) malloc(sizeof(double*) * kEigenDecompCount);
    if (gEigenValues == NULL)
        throw std::bad_alloc();

    gEMatrices = (double**) malloc(sizeof(double*) * kEigenDecompCount);
    if (gEMatrices == NULL)
    	throw std::bad_alloc();

    gIMatrices = (double**) malloc(sizeof(double*) * kEigenDecompCount);
       if (gIMatrices == NULL)
       	throw std::bad_alloc();

    for (int i = 0; i < kEigenDecompCount; i++) {
    	gEMatrices[i] = (double*) malloc(sizeof(double) * kStateCount * kStateCount);
    	if (gEMatrices[i] == NULL)
    		throw std::bad_alloc();

    	gIMatrices[i] = (double*) malloc(sizeof(double) * kStateCount * kStateCount);
    	if (gIMatrices[i] == NULL)
    		throw std::bad_alloc();

    	gEigenValues[i] = (double*) malloc(sizeof(double) * kEigenValuesSize);
    	if (gEigenValues[i] == NULL)
    		throw std::bad_alloc();
    }

    matrixTmp = (double*) malloc(sizeof(double) * kStateCount * kStateCount);
}

EigenDecompositionSquare::~EigenDecompositionSquare() {

	for(int i=0; i<kEigenDecompCount; i++) {
		free(gEMatrices[i]);
		free(gIMatrices[i]);
		free(gEigenValues[i]);
	}
	free(gEMatrices);
	free(gIMatrices);
	free(gEigenValues);
	free(matrixTmp);
}

void EigenDecompositionSquare::setEigenDecomposition(int eigenIndex,
										             const double* inEigenVectors,
                                                     const double* inInverseEigenVectors,
                                                     const double* inEigenValues) {

	memcpy(gEigenValues[eigenIndex],inEigenValues,sizeof(double) * kEigenValuesSize);
	const int len = kStateCount * kStateCount;
	memcpy(gEMatrices[eigenIndex],inEigenVectors,sizeof(double) * len);
	memcpy(gIMatrices[eigenIndex],inInverseEigenVectors,sizeof(double) * len);
}

void EigenDecompositionSquare::updateTransitionMatrices(int eigenIndex,
                                                        const int* probabilityIndices,
                                                        const int* firstDerivativeIndices,
                                                        const int* secondDervativeIndices,
                                                        const double* edgeLengths,
                                                        const double* categoryRates,
                                                        double** transitionMatrices,
                                                        int count) {

	const double* Ievc = gIMatrices[eigenIndex];
	const double* Evec = gEMatrices[eigenIndex];
	const double* Eval = gEigenValues[eigenIndex];
	const double* EvalImag = Eval + kStateCount;
    for (int u = 0; u < count; u++) {
        double* transitionMat = transitionMatrices[probabilityIndices[u]];
        const double edgeLength = edgeLengths[u];
        int n = 0;
        for (int l = 0; l < kCategoryCount; l++) {
			const double distance = categoryRates[l] * edgeLength;
        	int tmpIndex = 0;
        	for(int i=0; i<kStateCount; i++) {
        		if (!isComplex || EvalImag[i] == 0) {
        			const double tmp = exp(Eval[i] * distance);
        			for(int j=0; j<kStateCount; j++) {
        				matrixTmp[i*kStateCount+j] = Ievc[i*kStateCount+j] * tmp;
        			}
        		} else {
        			// 2 x 2 conjugate block
        			int i2 = i + 1;
        			const double b = EvalImag[i];
        			const double expat = exp(Eval[i] * distance);
        			const double expatcosbt = expat * cos(b * distance);
        			const double expatsinbt = expat * sin(b * distance);
        			for(int j=0; j<kStateCount; j++) {
        				matrixTmp[ i*kStateCount+j] = expatcosbt * Ievc[ i*kStateCount+j] +
        						                      expatsinbt * Ievc[i2*kStateCount+j];
        				matrixTmp[i2*kStateCount+j] = expatcosbt * Ievc[i2*kStateCount+j] -
												      expatsinbt * Ievc[ i*kStateCount+j];
        			}
        			i++; // processed two conjugate rows
        		}
        	}

            for (int i = 0; i < kStateCount; i++) {
                for (int j = 0; j < kStateCount; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < kStateCount; k++)
                        sum += Evec[i*kStateCount+k] * matrixTmp[k*kStateCount+j];
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

}
}
