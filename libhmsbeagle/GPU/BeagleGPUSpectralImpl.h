/*
 * @file BeagleGPUSpectralImpl.h
 *
 * Copyright 2026 Phylogenetic Likelihood Working Group
 *
 * This file is part of BEAGLE.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 *
 * @brief GPU spectral implementation header
 *
 * @author Marc Suchard
 */

#ifndef __BeagleGPUSpectralImpl__
#define __BeagleGPUSpectralImpl__

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include "libhmsbeagle/GPU/BeagleGPUImpl.h"

namespace beagle {
namespace gpu {

#ifdef CUDA
    namespace cuda {
#else
    namespace opencl {
#endif

BEAGLE_GPU_TEMPLATE
class BeagleGPUSpectralImpl : public BeagleGPUImpl<Real> {
private:
    GPUPtr  dSpectralDistancesOrigin;
    GPUPtr* dSpectralDistances;
    int*    hEigenIndexForMatrix;
    /* Per-matrix element stride within dSpectralDistancesOrigin, i.e.
     * AlignMemOffset(kCategoryCount * sizeof(Real)) / sizeof(Real). Device
     * alignment padding can make this larger than kCategoryCount, so any code
     * that indexes into the pooled origin buffer by matrix index (e.g. the
     * adjoint offset-queue) must multiply by this, not by kCategoryCount. */
    unsigned int kSpectralDistanceStrideElements;
    /* Host-side mirror of dSpectralDistancesOrigin (same layout/size:
     * kMatrixCount * kSpectralDistanceStrideElements elements), kept
     * persistent across calls so updateTransitionMatrices can patch just the
     * touched matrices' entries and then push the whole pooled buffer to the
     * device in a single MemcpyHostToDevice call. */
    Real*   hSpectralDistances;

    /* Backward eigenvector buffers for the parent branch in pre-order.
     * dEvecT[ei]  = U   stored row-major (dEvecT[j*S+k] = U[j,k]).
     *               Used as ievc1 in Growing kernels (backward Phase 1: U^T·p).
     * dIevcT[ei]  = U^-1 stored row-major (dIevcT[j*S+k] = U^-1[j,k]).
     *               Used as evec1 in Growing kernels (backward Phase 3: (U^-1)^T·q).
     * Allocated as one contiguous block each; sub-pointers per eigen index. */
    GPUPtr  dEvecTOrigin;
    GPUPtr* dEvecT;
    GPUPtr  dIevcTOrigin;
    GPUPtr* dIevcT;

    /* Adjoint gradient buffers.
     * dGradient: accumulated S×S gradient (one per eigen-decomposition).
     * hEigenDecompIsAllReal: per-eigen flag; set in setEigenDecomposition. */
    GPUPtr  dGradientOrigin;
    GPUPtr* dGradient;
    bool*   hEigenDecompIsAllReal;

    /* Offset-queue for the batched adjoint kernel: one 9-unsigned-int record
     * per branch (see STATUS.md for the field layout), addressed via
     * integer offsets into the pooled partials/states/evec/ievc/eigenvalue/
     * distance/gradient origins — never per-branch device pointers. Grown
     * lazily since branch count varies per calculateAdjointCrossProducts
     * call and isn't known at createInstance time. */
    static const int kAdjointQueueFieldsPerBranch = 9;
    GPUPtr        dAdjointQueue;
    unsigned int* hAdjointQueue;
    int           kAdjointQueueCapacity;

    /* BeagleGPUImpl::kEigenDecompCount is private to the base class; kept
     * here too so calculateAdjointCrossProducts can size a "seen ei" set
     * for the dGradient zero-fill without re-deriving it. */
    int kSpectralEigenDecompCount;

public:
    BeagleGPUSpectralImpl();
    ~BeagleGPUSpectralImpl() override;

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

    int updateTransitionMatrices(int eigenIndex,
                                 const int* probabilityIndices,
                                 const int* firstDerivativeIndices,
                                 const int* secondDerivativeIndices,
                                 const double* edgeLengths,
                                 int count) override;

    int updatePrePartials(const int* operations,
                          int operationCount,
                          int cumulativeScaleIndex,
                          BeaglePartialsType partialsType) override;

    int updatePrePartialsByPartition(const int* operations,
                                     int operationCount,
                                     BeaglePartialsType partialsType) override;

    char* getInstanceName() override;
    int getInstanceDetails(BeagleInstanceDetails* returnInfo) override;

    int calculateAdjointCrossProducts(const int* postBufferIndices,
                                      const int* preBufferIndices,
                                      const int* eigenIndices,
                                      const int* categoryRatesIndices,
                                      const int* categoryWeightsIndices,
                                      const int  rootPostOrderIndex,
                                      const int  stateFrequenciesIndex,
                                      int        count,
                                      double*    outSumDerivatives,
                                      double*    outSumSquaredDerivatives) override;

protected:
    void dispatchPrunePP(GPUPtr p1, GPUPtr p2, GPUPtr p3,
                          int c1MatIdx, int c2MatIdx,
                          GPUPtr scalingFactors, GPUPtr cumulativeScaling,
                          unsigned int startPattern, unsigned int endPattern,
                          int rescale, int streamIndex, int waitIndex) override;
    void dispatchPruneSP(GPUPtr s1, GPUPtr p2, GPUPtr p3,
                          int c1MatIdx, int c2MatIdx,
                          GPUPtr scalingFactors, GPUPtr cumulativeScaling,
                          unsigned int startPattern, unsigned int endPattern,
                          int rescale, int streamIndex, int waitIndex) override;
    void dispatchPruneSS(GPUPtr s1, GPUPtr s2, GPUPtr p3,
                          int c1MatIdx, int c2MatIdx,
                          GPUPtr scalingFactors, GPUPtr cumulativeScaling,
                          unsigned int startPattern, unsigned int endPattern,
                          int rescale, int streamIndex, int waitIndex) override;

private:
    void dispatchGrowingSpectral(bool isTop, bool isRoot,
                                  GPUPtr partials1, GPUPtr c2, bool sibIsStates,
                                  GPUPtr partials3,
                                  int c1MatIdx, int c2MatIdx);
};

BEAGLE_GPU_TEMPLATE
class BeagleGPUSpectralImplFactory : public BeagleImplFactory {
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

} // namespace cuda/opencl
} // namespace gpu
} // namespace beagle

#include "libhmsbeagle/GPU/BeagleGPUSpectralImpl.hpp"

#endif // __BeagleGPUSpectralImpl__
