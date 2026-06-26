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

    int updateTransitionMatrices(int eigenIndex,
                                 const int* probabilityIndices,
                                 const int* firstDerivativeIndices,
                                 const int* secondDerivativeIndices,
                                 const double* edgeLengths,
                                 int count) override;

    char* getInstanceName() override;
    int getInstanceDetails(BeagleInstanceDetails* returnInfo) override;

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
