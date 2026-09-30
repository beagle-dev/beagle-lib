/*
 *  BeagleCPU4StateSpectralSSEImpl.h
 *  BEAGLE
 *
 * Copyright 2026 Phylogenetic Likelihood Working Group
 *
 * This file is part of BEAGLE.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 *
 * @author Marc Suchard
 */

#ifndef __BeagleCPU4StateSpectralSSEImpl__
#define __BeagleCPU4StateSpectralSSEImpl__

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include "libhmsbeagle/CPU/BeagleCPUSpectralSSEImpl.h"
#include "libhmsbeagle/CPU/BeagleCPUAdjoint4StateSSE.h"

#include <vector>

namespace beagle {
namespace cpu {

/*
 * Spectral CPU implementation for 4 states, double precision, SSE2 / NEON. Each 4 x 4 matrix-vector product is
 * unrolled into two-wide vector operations on compact, aligned copies of the eigen decomposition's matrices, and
 * exp(D t) is applied per layout of complex conjugate pairs. The adjoint gradient is BeagleCPUAdjoint4StateSSE's.
 */
template <int T_PAD, int P_PAD>
class BeagleCPU4StateSpectralSSEImpl : public BeagleCPUAdjoint4StateSSE<BeagleCPUSpectralSSEImpl<T_PAD, P_PAD>> {

    typedef BeagleCPUAdjoint4StateSSE<BeagleCPUSpectralSSEImpl<T_PAD, P_PAD>> Base;

protected:
    using Base::gEigenDecomposition;
    using Base::kFlags;
    using Base::kCategoryCount;
    using Base::kPaddedPatternCount;
    using Base::kEigenDecompCount;
    using Base::gPartials;
    using typename Base::BranchEigenInfo;
    using Base::gBranchEigenInfo;

public:
    const char* getName() override;

    int createInstance(int tipCount,
                       int partialsBufferCount,
                       int compactBufferCount,
                       int stateCount,
                       int patternCount,
                       int eigenDecompositionCount,
                       int matrixCount,
                       int categoryCount,
                       int scaleBufferCount,
                       int resourceNumber,
                       int pluginResourceNumber,
                       long preferenceFlags,
                       long requirementFlags) override;

    int setEigenDecomposition(int eigenIndex,
                              const double* inEigenVectors,
                              const double* inInverseEigenVectors,
                              const double* inEigenValues) override;

protected:
    void spectralPartialsPartials(double* destP,
                                  const double* partials1, const int branchEigenIndex1,
                                  const double* partials2, const int branchEigenIndex2,
                                  const double* scaleFactors,
                                  int startPattern, int endPattern, int currentPartition) override;

    void spectralStatesPartials(double* destP,
                                const int* states1, const int branchEigenIndex1,
                                const double* partials2, const int branchEigenIndex2,
                                const double* scaleFactors,
                                int startPattern, int endPattern, int currentPartition) override;

    void spectralStatesStates(double* destP,
                              const int* states1, const int branchEigenIndex1,
                              const int* states2, const int branchEigenIndex2,
                              const double* scaleFactors,
                              int startPattern, int endPattern, int currentPartition) override;

    void spectralDegree2Partials(double* destP,
                                 const int* states1, const double* partials1,
                                 const int branchEigenIndex1,
                                 const double* scaleFactors,
                                 int startPattern, int endPattern, int currentPartition) override;

    void spectralPrePartialsPartials(double* destP, SpectralPreOrder type,
                                     const double* partials1, const int branchEigenIndex1,
                                     const double* partials2, const int branchEigenIndex2,
                                     int startPattern, int endPattern, int currentPartition) override;

    void spectralPrePartialsStates(double* destP, SpectralPreOrder type,
                                   const double* partials1, const int branchEigenIndex1,
                                   const int* states2, const int branchEigenIndex2,
                                   int startPattern, int endPattern, int currentPartition) override;

    void calcDegree2PrePartials(double* destP,
                                const double* partials1,
                                const int branchEigenIndex1,
                                int startPattern,
                                int endPattern,
                                int currentPartition) override;

private:
    // Per eigen decomposition, 4 x 4 row-major and 16-byte aligned: the columns of V^{-1} (the rows of V^{-T})
    // followed by the row sums of V^{-1} (a missing state), the columns of V, the columns of V^T (the rows of V)
    // and the columns of V^{-T} (the rows of V^{-1}); and the layout of the complex conjugate pairs
    struct Matrices4 {
        alignas(16) double inverseColumns[20];
        alignas(16) double columns[16];
        alignas(16) double transposeColumns[16];
        alignas(16) double inverseTransposeColumns[16];
        int pairs; // 0 every eigenvalue real; 1 a pair (0, 1); 2 a pair (2, 3); 3 both; 4 a pair (1, 2)
    };
    std::vector<Matrices4> gMatrices4;

    void postOrder(double* destP,
                   const int* states1, const double* partials1, const int branchEigenIndex1,
                   const int* states2, const double* partials2, const int branchEigenIndex2,
                   const double* scaleFactors, int startPattern, int endPattern);

    void preOrder(double* destP, SpectralPreOrder type,
                  const double* partials1, const int branchEigenIndex1,
                  const int* states2, const double* partials2, const int branchEigenIndex2,
                  int startPattern, int endPattern);
};

class BeagleCPU4StateSpectralSSEImplFactory : public BeagleImplFactory {
public:
    virtual BeagleImpl* createImpl(int tipCount,
                                   int partialsBufferCount,
                                   int compactBufferCount,
                                   int stateCount,
                                   int patternCount,
                                   int eigenBufferCount,
                                   int matrixBufferCount,
                                   int categoryCount,
                                   int scaleBufferCount,
                                   int resourceNumber,
                                   int pluginResourceNumber,
                                   long preferenceFlags,
                                   long requirementFlags,
                                   int* errorCode);

    virtual const char* getName();
    virtual const long getFlags();
};

} // namespace cpu
} // namespace beagle

// now include the file containing template function implementations
#include "libhmsbeagle/CPU/BeagleCPU4StateSpectralSSEImpl.hpp"

#endif // __BeagleCPU4StateSpectralSSEImpl__
