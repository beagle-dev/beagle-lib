/*
 * @file BeagleGPUSpectralImpl.hpp
 *
 * Copyright 2026 Phylogenetic Likelihood Working Group
 *
 * This file is part of BEAGLE.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 *
 * @brief GPU spectral implementation template definitions
 *
 * @author Marc Suchard
 */

#ifndef __BeagleGPUSpectralImpl_hpp__
#define __BeagleGPUSpectralImpl_hpp__

#include "libhmsbeagle/beagle.h"

namespace beagle {
namespace gpu {

#ifdef CUDA
    namespace cuda {
#else
    namespace opencl {
#endif

///////////////////////////////////////////////////////////////////////////////
// BeagleGPUSpectralImplFactory

BEAGLE_GPU_TEMPLATE
BeagleImpl* BeagleGPUSpectralImplFactory<BEAGLE_GPU_GENERIC>::createImpl(
        int tipCount,
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
        int* errorCode) {
    BeagleImpl* impl = new BeagleGPUSpectralImpl<BEAGLE_GPU_GENERIC>();
    try {
        *errorCode = impl->createInstance(tipCount, partialsBufferCount, compactBufferCount,
                                          stateCount, patternCount, eigenBufferCount,
                                          matrixBufferCount, categoryCount, scaleBufferCount,
                                          resourceNumber, pluginResourceNumber,
                                          preferenceFlags, requirementFlags);
        if (*errorCode == BEAGLE_SUCCESS) {
            return impl;
        }
        delete impl;
        return NULL;
    }
    catch (...) {
        delete impl;
        *errorCode = BEAGLE_ERROR_GENERAL;
        throw;
    }
    delete impl;
    *errorCode = BEAGLE_ERROR_GENERAL;
    return NULL;
}

#ifdef CUDA
template<>
const char* BeagleGPUSpectralImplFactory<double>::getName() {
    return "GPU-DP-CUDA-Spectral";
}

template<>
const char* BeagleGPUSpectralImplFactory<float>::getName() {
    return "GPU-SP-CUDA-Spectral";
}
#else
template<>
const char* BeagleGPUSpectralImplFactory<double>::getName() {
    return "GPU-DP-OpenCL-Spectral";
}

template<>
const char* BeagleGPUSpectralImplFactory<float>::getName() {
    return "GPU-SP-OpenCL-Spectral";
}
#endif

BEAGLE_GPU_TEMPLATE
const long BeagleGPUSpectralImplFactory<BEAGLE_GPU_GENERIC>::getFlags() {
    long flags = BEAGLE_FLAG_COMPUTATION_SYNCH | BEAGLE_FLAG_COMPUTATION_ASYNCH |
                 BEAGLE_FLAG_SCALING_MANUAL | BEAGLE_FLAG_SCALING_ALWAYS |
                 BEAGLE_FLAG_SCALING_AUTO | BEAGLE_FLAG_SCALING_DYNAMIC |
                 BEAGLE_FLAG_THREADING_NONE |
                 BEAGLE_FLAG_VECTOR_NONE |
                 BEAGLE_FLAG_SCALERS_LOG | BEAGLE_FLAG_SCALERS_RAW |
                 BEAGLE_FLAG_EIGEN_COMPLEX | BEAGLE_FLAG_EIGEN_REAL |
                 BEAGLE_FLAG_INVEVEC_STANDARD | BEAGLE_FLAG_INVEVEC_TRANSPOSED |
                 BEAGLE_FLAG_PREORDER_TRANSPOSE_MANUAL | BEAGLE_FLAG_PREORDER_TRANSPOSE_AUTO |
                 BEAGLE_FLAG_PARALLELOPS_GRID | BEAGLE_FLAG_PARALLELOPS_STREAMS |
                 BEAGLE_FLAG_SPECTRAL_REPRESENTATION;

#ifdef CUDA
    flags |= BEAGLE_FLAG_FRAMEWORK_CUDA |
             BEAGLE_FLAG_PROCESSOR_GPU;
#else
    flags |= BEAGLE_FLAG_FRAMEWORK_OPENCL |
             BEAGLE_FLAG_PROCESSOR_CPU | BEAGLE_FLAG_PROCESSOR_GPU | BEAGLE_FLAG_PROCESSOR_OTHER;
#endif

    Real r = 0;
    modifyFlagsForPrecision(&flags, r);
    return flags;
}

} // namespace cuda/opencl
} // namespace gpu
} // namespace beagle

#endif // __BeagleGPUSpectralImpl_hpp__
