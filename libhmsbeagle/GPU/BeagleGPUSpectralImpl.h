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
public:
    ~BeagleGPUSpectralImpl() override = default;
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
