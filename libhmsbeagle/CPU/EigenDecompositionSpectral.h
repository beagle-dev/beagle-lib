/*
 * EigenDecompositionSpectral.h
 *
 *  Created on: May 13, 2026
 *      Author: msuchard
 */

#ifndef EIGENDECOMPOSITIONSPECTRAL_H_
#define EIGENDECOMPOSITIONSPECTRAL_H_

#include "EigenDecomposition.h"

namespace beagle {
namespace cpu {

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

public:
    EigenDecompositionSpectral(int decompositionCount,
                               int stateCount,
                               int categoryCount,
                               long flags);

    virtual ~EigenDecompositionSpectral();

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

    virtual void updateTransitionMatricesWithModelCategories(int* eigenIndices,
                                                             const int* probabilityIndices,
                                                             const int* firstDerivativeIndices,
                                                             const int* secondDerivativeIndices,
                                                             const double* edgeLengths,
                                                             REALTYPE** transitionMatrices,
                                                             int count);

    virtual const REALTYPE* getEigenValuesPtr(int eigenIndex) const;

    virtual const REALTYPE* getEigenVectorsPtr(int eigenIndex) const;

    virtual const REALTYPE* getInverseEigenVectorsPtr(int eigenIndex) const;
};

}
}

// Include the template implementation header
#include "libhmsbeagle/CPU/EigenDecompositionSpectral.hpp"

#endif /* EIGENDECOMPOSITIONSPECTRAL_H_ */
