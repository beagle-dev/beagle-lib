/*
 * EigenDecompositionSpectral.cpp
 *
 *  Created on: May 13, 2026
 *      Author: msuchard
 */

#ifndef EIGENDECOMPOSITIONSPECTRAL_HPP_
#define EIGENDECOMPOSITIONSPECTRAL_HPP_

#include "EigenDecompositionSpectral.h"

namespace beagle {
namespace cpu {


BEAGLE_CPU_EIGEN_TEMPLATE
EigenDecompositionSpectral<BEAGLE_CPU_EIGEN_GENERIC>::EigenDecompositionSpectral(
        int decompositionCount,
        int stateCount,
        int categoryCount,
        long flags)
    : EigenDecomposition<BEAGLE_CPU_EIGEN_GENERIC>(decompositionCount, stateCount,
                                                   categoryCount, flags),
                                                   isComplex(kFlags & BEAGLE_FLAG_EIGEN_COMPLEX),
                                                   kEigenValuesSize(isComplex ? kStateCount * 2 : kStateCount),
                                                   kMatrixStride(kStateCount + T_PAD),                                             
                                                   eigenValuesStorage(kEigenDecompCount, std::vector<REALTYPE>(kEigenValuesSize)),
                                                   eigenVectorsStorage(kEigenDecompCount, std::vector<REALTYPE>(kMatrixStride * kStateCount)),
                                                   inverseEigenVectorsStorage(kEigenDecompCount, std::vector<REALTYPE>(kMatrixStride * kStateCount)) {
    // Do nothing   
}

BEAGLE_CPU_EIGEN_TEMPLATE
EigenDecompositionSpectral<BEAGLE_CPU_EIGEN_GENERIC>::~EigenDecompositionSpectral() {
    // Do nothing
}

BEAGLE_CPU_EIGEN_TEMPLATE
void EigenDecompositionSpectral<BEAGLE_CPU_EIGEN_GENERIC>::setEigenDecomposition(
        int eigenIndex,
        const double* inEigenVectors,
        const double* inInverseEigenVectors,
        const double* inEigenValues) {
    
    // if (kMatrixStride == kStateCount) {
    //     fprintf(stderr, "Error: EigenDecompositionSpectral::setEigenDecomposition: matrixStride= %d matches stateCount\n", kMatrixStride);
    //     exit(-1);
    // }

	beagleMemCpy(eigenValuesStorage[eigenIndex].data(), inEigenValues, kEigenValuesSize);

    auto& evec = eigenVectorsStorage[eigenIndex];
    auto& ivec = inverseEigenVectorsStorage[eigenIndex];

	// const int len = kStateCount * kStateCount;
	// beagleMemCpy(eigenVectorsStorage[eigenIndex].data(), inEigenVectors, len);
	// beagleMemCpy(inverseEigenVectorsStorage[eigenIndex].data(), inInverseEigenVectors, len);    

    for (int i = 0; i < kStateCount; i++) {
        REALTYPE rowSumI = 0.0;
        for (int j = 0; j < kStateCount; j++) {
            REALTYPE value = (REALTYPE) inInverseEigenVectors[i * kStateCount + j];
            rowSumI += value;
            ivec[i * kMatrixStride + j] = value; 
        }
        if constexpr (T_PAD > 0) {            
            ivec[i * kMatrixStride + kStateCount] = rowSumI;            
        }

        REALTYPE rowSumE = 0.0;
        for (int j = 0; j < kStateCount; j++) {
            REALTYPE value = (REALTYPE) inEigenVectors[i * kStateCount + j];
            rowSumE += value;
            evec[i * kMatrixStride + j] = value;
        }
        if constexpr (T_PAD > 0) {            
            evec[i * kMatrixStride + kStateCount] = rowSumE;            
        }
    }
}

BEAGLE_CPU_EIGEN_TEMPLATE
void EigenDecompositionSpectral<BEAGLE_CPU_EIGEN_GENERIC>::updateTransitionMatrices(
        int eigenIndex,
        const int* probabilityIndices,
        const int* firstDerivativeIndices,
        const int* secondDerivativeIndices,
        const double* edgeLengths,
        const double* categoryRates,
        REALTYPE** transitionMatrices,
        int count) {
    fprintf(stderr, "EigenDecompositionSpectral::updateTransitionMatrices should not be called\n");
    exit(-1);
}

BEAGLE_CPU_EIGEN_TEMPLATE
void EigenDecompositionSpectral<BEAGLE_CPU_EIGEN_GENERIC>::updateTransitionMatricesWithModelCategories(
        int* eigenIndices,
        const int* probabilityIndices,
        const int* firstDerivativeIndices,
        const int* secondDerivativeIndices,
        const double* edgeLengths,
        REALTYPE** transitionMatrices,
        int count) {
    fprintf(stderr, "EigenDecompositionSpectral::updateTransitionMatricesWithModelCategories should not be called\n");
    exit(-1);            
}

BEAGLE_CPU_EIGEN_TEMPLATE
const REALTYPE* EigenDecompositionSpectral<BEAGLE_CPU_EIGEN_GENERIC>::getEigenValuesPtr(
        int eigenIndex) const {
    return eigenValuesStorage[eigenIndex].data();
}

BEAGLE_CPU_EIGEN_TEMPLATE
const REALTYPE* EigenDecompositionSpectral<BEAGLE_CPU_EIGEN_GENERIC>::getEigenVectorsPtr(
        int eigenIndex) const {
    return eigenVectorsStorage[eigenIndex].data();
}

BEAGLE_CPU_EIGEN_TEMPLATE
const REALTYPE* EigenDecompositionSpectral<BEAGLE_CPU_EIGEN_GENERIC>::getInverseEigenVectorsPtr(
        int eigenIndex) const {
    return inverseEigenVectorsStorage[eigenIndex].data();
}

} // namespace cpu
} // namespace beagle

#endif /* EIGENDECOMPOSITIONSPECTRAL_HPP_ */
