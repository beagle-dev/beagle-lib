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
                                                   inverseEigenVectorsStorage(kEigenDecompCount, std::vector<REALTYPE>(kMatrixStride * kStateCount)),
                                                //    transposedEigenValuesStorage(kEigenDecompCount, std::vector<REALTYPE>(kEigenValuesSize)),
                                                   transposedEigenVectorsStorage(kEigenDecompCount, std::vector<REALTYPE>(kMatrixStride * kStateCount)),
                                                   transposedInverseEigenVectorsStorage(kEigenDecompCount, std::vector<REALTYPE>(kMatrixStride * kStateCount)),
                                                   adjointMethodsStorage(kEigenDecompCount) {
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

    auto& eval = eigenValuesStorage[eigenIndex];

	beagleMemCpy(eval.data(), inEigenValues, kEigenValuesSize);

    auto& evec = eigenVectorsStorage[eigenIndex];
    auto& ivec = inverseEigenVectorsStorage[eigenIndex];

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

    // auto& tEval = transposedEigenValuesStorage[eigenIndex];
    auto& tEvec = transposedEigenVectorsStorage[eigenIndex];
    auto& tIvec = transposedInverseEigenVectorsStorage[eigenIndex];

    // memcpy(tEval.data(), eval.data(), kEigenValuesSize * sizeof(REALTYPE));

    memcpy(tEvec.data(), ivec.data(), kMatrixStride * kStateCount * sizeof(REALTYPE));
    transposeInPlace(tEvec.data());

    memcpy(tIvec.data(), evec.data(), kMatrixStride * kStateCount * sizeof(REALTYPE));
    transposeInPlace(tIvec.data());

    if (getenv("BEAGLE_DEBUG_EIGEN")) {
        fprintf(stderr, "[CPU setEigenDecomposition] eigenIndex=%d kStateCount=%d kMatrixStride=%d\n", eigenIndex, kStateCount, kMatrixStride);
        fprintf(stderr, "[CPU] eval: "); for (int i=0;i<kEigenValuesSize;i++) fprintf(stderr, "%.6f ", (double)eval[i]); fprintf(stderr, "\n");
        fprintf(stderr, "[CPU] evec:\n"); for (int i=0;i<kStateCount;i++){ for(int j=0;j<kStateCount;j++) fprintf(stderr, "%.6f ", (double)evec[i*kMatrixStride+j]); fprintf(stderr, "\n"); }
        fprintf(stderr, "[CPU] ivec:\n"); for (int i=0;i<kStateCount;i++){ for(int j=0;j<kStateCount;j++) fprintf(stderr, "%.6f ", (double)ivec[i*kMatrixStride+j]); fprintf(stderr, "\n"); }
        fflush(stderr);
    }
    adjointMethodsStorage[eigenIndex] = std::make_shared<AdjointIntegralPlan<REALTYPE>>(
        eigenValuesStorage[eigenIndex].data(), kStateCount, !isComplex);
}

BEAGLE_CPU_EIGEN_TEMPLATE
void EigenDecompositionSpectral<BEAGLE_CPU_EIGEN_GENERIC>::rescale(REALTYPE* matrix, const REALTYPE* eval, REALTYPE scalar) {
    if (kEigenValuesSize != 2 * kStateCount) {
        return;
    }

    for (int i = 0; i < kStateCount; ++i) {
        if (eval[kStateCount + i] != 0.0) {
            for (int j = 0; j < kStateCount; ++j) {
                matrix[i * kMatrixStride + j] = scalar * matrix[i * kMatrixStride + j];
                matrix[(i + 1) * kMatrixStride + j] = -scalar * matrix[(i + 1) * kMatrixStride + j];
            }
            ++i;
        }
    }
}

BEAGLE_CPU_EIGEN_TEMPLATE
void EigenDecompositionSpectral<BEAGLE_CPU_EIGEN_GENERIC>::transposeInPlace(REALTYPE* matrix) {
    for (int i = 0; i < kStateCount; i++) {
        for (int j = i + 1; j < kStateCount; j++) {
            int index1 = i * kMatrixStride + j;
            int index2 = j * kMatrixStride + i;

            REALTYPE temp = matrix[index1];
            matrix[index1] = matrix[index2];
            matrix[index2] = temp;
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

// BEAGLE_CPU_EIGEN_TEMPLATE
// const REALTYPE* EigenDecompositionSpectral<BEAGLE_CPU_EIGEN_GENERIC>::getBackwardsEigenValuesPtr(
//         int eigenIndex) const {
//     return transposedEigenValuesStorage[eigenIndex].data();
// }

BEAGLE_CPU_EIGEN_TEMPLATE
const REALTYPE* EigenDecompositionSpectral<BEAGLE_CPU_EIGEN_GENERIC>::getBackwardsEigenVectorsPtr(
        int eigenIndex) const {
    return transposedEigenVectorsStorage[eigenIndex].data();
}

BEAGLE_CPU_EIGEN_TEMPLATE
const REALTYPE* EigenDecompositionSpectral<BEAGLE_CPU_EIGEN_GENERIC>::getBackwardsInverseEigenVectorsPtr(
        int eigenIndex) const {
    return transposedInverseEigenVectorsStorage[eigenIndex].data();
}

BEAGLE_CPU_EIGEN_TEMPLATE
AdjointIntegralPlan<REALTYPE>* EigenDecompositionSpectral<BEAGLE_CPU_EIGEN_GENERIC>::getAdjointMethodsPtr(
        int eigenIndex) const {
    return adjointMethodsStorage[eigenIndex].get();
}

} // namespace cpu
} // namespace beagle

#endif /* EIGENDECOMPOSITIONSPECTRAL_HPP_ */
