/*
 * EigenDecompositionSquare.cpp
 *
 *  Created on: Sep 24, 2009
 *      Author: msuchard
 */
#ifndef _EigenDecompositionSquare_hpp_
#define _EigenDecompositionSquare_hpp_
#include "EigenDecompositionSquare.h"
#include "libhmsbeagle/beagle.h"

//#if defined (BEAGLE_IMPL_DEBUGGING_OUTPUT) && BEAGLE_IMPL_DEBUGGING_OUTPUT
//const bool DEBUGGING_OUTPUT = true;
//#else
//const bool DEBUGGING_OUTPUT = false;
//#endif

//#define DEBUG_COMPLEX

namespace beagle {
namespace cpu {

template <typename REALTYPE>
EigenDecompositionSquare<REALTYPE>::EigenDecompositionSquare(int decompositionCount,
											       int stateCount,
											       int categoryCount,
											       long flags)
	: EigenDecomposition<REALTYPE>(decompositionCount,stateCount,categoryCount) {

	isComplex = flags & BEAGLE_FLAG_COMPLEX;

	if (isComplex)
		kEigenValuesSize = 2 * kStateCount;
	else
		kEigenValuesSize = kStateCount;

    this->gEigenValues = (REALTYPE**) malloc(sizeof(REALTYPE*) * kEigenDecompCount);
    if (gEigenValues == NULL)
        throw std::bad_alloc();

    gEMatrices = (REALTYPE**) malloc(sizeof(REALTYPE*) * kEigenDecompCount);
    if (gEMatrices == NULL)
    	throw std::bad_alloc();

    gIMatrices = (REALTYPE**) malloc(sizeof(REALTYPE*) * kEigenDecompCount);
       if (gIMatrices == NULL)
       	throw std::bad_alloc();

    for (int i = 0; i < kEigenDecompCount; i++) {
    	gEMatrices[i] = (REALTYPE*) malloc(sizeof(REALTYPE) * kStateCount * kStateCount);
    	if (gEMatrices[i] == NULL)
    		throw std::bad_alloc();

    	gIMatrices[i] = (REALTYPE*) malloc(sizeof(REALTYPE) * kStateCount * kStateCount);
    	if (gIMatrices[i] == NULL)
    		throw std::bad_alloc();

    	gEigenValues[i] = (REALTYPE*) malloc(sizeof(REALTYPE) * kEigenValuesSize);
    	if (gEigenValues[i] == NULL)
    		throw std::bad_alloc();
    }

    matrixTmp = (REALTYPE*) malloc(sizeof(REALTYPE) * kStateCount * kStateCount);
}

template <typename REALTYPE>
EigenDecompositionSquare<REALTYPE>::~EigenDecompositionSquare() {

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

template <typename REALTYPE>
void EigenDecompositionSquare<REALTYPE>::setEigenDecomposition(int eigenIndex,
										             const double* inEigenVectors,
                                                     const double* inInverseEigenVectors,
                                                     const double* inEigenValues) {

	memcpy(gEigenValues[eigenIndex],inEigenValues,sizeof(double) * kEigenValuesSize);
	const int len = kStateCount * kStateCount;
	memcpy(gEMatrices[eigenIndex],inEigenVectors,sizeof(double) * len);
	memcpy(gIMatrices[eigenIndex],inInverseEigenVectors,sizeof(double) * len);
}

template <typename REALTYPE>
void EigenDecompositionSquare<REALTYPE>::updateTransitionMatrices(int eigenIndex,
                                                        const int* probabilityIndices,
                                                        const int* firstDerivativeIndices,
                                                        const int* secondDerivativeIndices,
                                                        const double* edgeLengths,
                                                        const double* categoryRates,
                                                        REALTYPE** transitionMatrices,
                                                        int count) {

	const REALTYPE* Ievc = gIMatrices[eigenIndex];
	const REALTYPE* Evec = gEMatrices[eigenIndex];
	const REALTYPE* Eval = gEigenValues[eigenIndex];
	const REALTYPE* EvalImag = Eval + kStateCount;
    for (int u = 0; u < count; u++) {
        REALTYPE* transitionMat = transitionMatrices[probabilityIndices[u]];
        const double edgeLength = edgeLengths[u];
        int n = 0;
        for (int l = 0; l < kCategoryCount; l++) {
			const REALTYPE distance = categoryRates[l] * edgeLength;
        	int tmpIndex = 0;
        	for(int i=0; i<kStateCount; i++) {
        		if (!isComplex || EvalImag[i] == 0) {
        			const REALTYPE tmp = exp(Eval[i] * distance);
        			for(int j=0; j<kStateCount; j++) {
        				matrixTmp[i*kStateCount+j] = Ievc[i*kStateCount+j] * tmp;
        			}
        		} else {
        			// 2 x 2 conjugate block
        			int i2 = i + 1;
        			const REALTYPE b = EvalImag[i];
        			const REALTYPE expat = exp(Eval[i] * distance);
        			const REALTYPE expatcosbt = expat * cos(b * distance);
        			const REALTYPE expatsinbt = expat * sin(b * distance);
        			for(int j=0; j<kStateCount; j++) {
        				matrixTmp[ i*kStateCount+j] = expatcosbt * Ievc[ i*kStateCount+j] +
        						                      expatsinbt * Ievc[i2*kStateCount+j];
        				matrixTmp[i2*kStateCount+j] = expatcosbt * Ievc[i2*kStateCount+j] -
												      expatsinbt * Ievc[ i*kStateCount+j];
        			}
        			i++; // processed two conjugate rows
        		}
        	}

#ifdef DEBUG_COMPLEX
           	fprintf(stderr,"[");
            	for(int i=0; i<16; i++)
            		fprintf(stderr," %7.5e,",matrixTmp[i]);
            	fprintf(stderr,"] -- complex debug\n");
            	exit(0);
#endif


            for (int i = 0; i < kStateCount; i++) {
                for (int j = 0; j < kStateCount; j++) {
                    REALTYPE sum = 0.0;
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

#endif 

