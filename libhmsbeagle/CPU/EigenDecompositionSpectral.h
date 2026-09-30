/*
 * EigenDecompositionSpectral.h
 *
 *  Created on: May 13, 2026
 *      Author: msuchard
 */

#ifndef EIGENDECOMPOSITIONSPECTRAL_H_
#define EIGENDECOMPOSITIONSPECTRAL_H_

#include <memory>
#include <vector>

#include "EigenDecomposition.h"
#include "libhmsbeagle/beagle.h"
#include "libhmsbeagle/CPU/AdjointMethods.h"

namespace beagle {
namespace cpu {

// Flags for an EigenDecompositionSpectral that another decomposition fills with the inverse eigenvectors in the
// standard layout, whatever the instance's layout
inline long standardInverseFlags(const long flags) {
    return (flags & ~BEAGLE_FLAG_INVEVEC_TRANSPOSED) | BEAGLE_FLAG_INVEVEC_STANDARD;
}

// Hands storage (an EigenDecompositionSpectral) the inverse eigenvectors in the standard layout
template <typename Storage>
inline void setStandardEigenDecomposition(Storage& storage, const int eigenIndex, const int stateCount,
                                          const double* inEigenVectors, const double* inInverseEigenVectors,
                                          const bool inverseTransposed, const double* inEigenValues) {
    if (!inverseTransposed) {
        storage.setEigenDecomposition(eigenIndex, inEigenVectors, inInverseEigenVectors, inEigenValues);
        return;
    }
    std::vector<double> inverse(stateCount * stateCount);
    for (int i = 0; i < stateCount; i++) {
        for (int j = 0; j < stateCount; j++) {
            inverse[i * stateCount + j] = inInverseEigenVectors[j * stateCount + i];
        }
    }
    storage.setEigenDecomposition(eigenIndex, inEigenVectors, inverse.data(), inEigenValues);
}

BEAGLE_CPU_EIGEN_TEMPLATE
class EigenDecompositionSpectral : public EigenDecomposition<BEAGLE_CPU_EIGEN_GENERIC> {

    using EigenDecomposition<BEAGLE_CPU_EIGEN_GENERIC>::gEigenValues;
    using EigenDecomposition<BEAGLE_CPU_EIGEN_GENERIC>::kStateCount;
    using EigenDecomposition<BEAGLE_CPU_EIGEN_GENERIC>::kEigenDecompCount;
    using EigenDecomposition<BEAGLE_CPU_EIGEN_GENERIC>::kCategoryCount;
    using EigenDecomposition<BEAGLE_CPU_EIGEN_GENERIC>::kFlags;
    using EigenDecomposition<BEAGLE_CPU_EIGEN_GENERIC>::matrixTmp;

private:
    const bool isComplex;
    const int kEigenValuesSize;
    const int kMatrixStride;

    std::vector<std::vector<REALTYPE>> eigenValuesStorage;
    std::vector<std::vector<REALTYPE>> eigenVectorsStorage;
    std::vector<std::vector<REALTYPE>> inverseEigenVectorsStorage;

    // std::vector<std::vector<REALTYPE>> transposedEigenValuesStorage;
    std::vector<std::vector<REALTYPE>> transposedEigenVectorsStorage;
    std::vector<std::vector<REALTYPE>> transposedInverseEigenVectorsStorage;

    std::vector<std::shared_ptr<AdjointIntegralPlan<REALTYPE>>> adjointMethodsStorage;

public:
    EigenDecompositionSpectral(int decompositionCount,
                               int stateCount,
                               int categoryCount,
                               long flags);

    ~EigenDecompositionSpectral() override;

    void setEigenDecomposition(int eigenIndex,
                               const double* inEigenVectors,
                               const double* inInverseEigenVectors,
                               const double* inEigenValues) override;

    void updateTransitionMatrices(int eigenIndex,
                                  const int* probabilityIndices,
                                  const int* firstDerivativeIndices,
                                  const int* secondDerivativeIndices,
                                  const double* edgeLengths,
                                  const double* categoryRates,
                                  REALTYPE** transitionMatrices,
                                  int count) override;

    void updateTransitionMatricesWithModelCategories(int* eigenIndices,
                                                     const int* probabilityIndices,
                                                     const int* firstDerivativeIndices,
                                                     const int* secondDerivativeIndices,
                                                     const double* edgeLengths,
                                                     REALTYPE** transitionMatrices,
                                                     int count) override;

    const REALTYPE* getEigenValuesPtr(int eigenIndex) const override;

    const REALTYPE* getEigenVectorsPtr(int eigenIndex) const override;

    const REALTYPE* getInverseEigenVectorsPtr(int eigenIndex) const override;

    // const REALTYPE* getBackwardsEigenValuesPtr(int eigenIndex) const override;

    const REALTYPE* getBackwardsEigenVectorsPtr(int eigenIndex) const override;

    const REALTYPE* getBackwardsInverseEigenVectorsPtr(int eigenIndex) const override;

    AdjointIntegralPlan<REALTYPE>* getAdjointMethodsPtr(int eigenIndex) const override;

private:
    void rescale(REALTYPE* rowVectors, const REALTYPE* eval, REALTYPE scalar);

    void transposeInPlace(REALTYPE* matrix);
};

}
}

// Include the template implementation header
#include "libhmsbeagle/CPU/EigenDecompositionSpectral.hpp"

#endif /* EIGENDECOMPOSITIONSPECTRAL_H_ */
