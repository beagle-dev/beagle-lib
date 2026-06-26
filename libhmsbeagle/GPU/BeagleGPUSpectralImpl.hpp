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
// BeagleGPUSpectralImpl

BEAGLE_GPU_TEMPLATE
BeagleGPUSpectralImpl<BEAGLE_GPU_GENERIC>::BeagleGPUSpectralImpl()
    : dSpectralDistancesOrigin(0), dSpectralDistances(NULL), hEigenIndexForMatrix(NULL) {
}

BEAGLE_GPU_TEMPLATE
BeagleGPUSpectralImpl<BEAGLE_GPU_GENERIC>::~BeagleGPUSpectralImpl() {
    if (dSpectralDistancesOrigin) {
        GPUInterface* gpuIf = this->gpu;
        if (gpuIf) gpuIf->FreeMemory(dSpectralDistancesOrigin);
    }
    free(dSpectralDistances);
    free(hEigenIndexForMatrix);
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUSpectralImpl<BEAGLE_GPU_GENERIC>::createInstance(
        int tipCount,
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
        long requirementFlags) {
    int rc = BeagleGPUImpl<Real>::createInstance(tipCount, partialsBufferCount, compactBufferCount,
                                                  stateCount, patternCount, eigenDecompositionCount,
                                                  matrixCount, categoryCount, scaleBufferCount,
                                                  resourceNumber, pluginResourceNumber,
                                                  preferenceFlags, requirementFlags);
    if (rc != BEAGLE_SUCCESS) return rc;

    hEigenIndexForMatrix = (int*) calloc(this->kMatrixCount, sizeof(int));

    dSpectralDistances = (GPUPtr*) malloc(sizeof(GPUPtr) * this->kMatrixCount);
    size_t distStride = this->kCategoryCount * sizeof(Real);
    GPUInterface* gpuIf = this->gpu;
    dSpectralDistancesOrigin = gpuIf->AllocateMemory(this->kMatrixCount * distStride);
    for (int i = 0; i < this->kMatrixCount; i++) {
        dSpectralDistances[i] = gpuIf->CreateSubPointer(dSpectralDistancesOrigin, distStride * i, distStride);
    }
    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUSpectralImpl<BEAGLE_GPU_GENERIC>::updateTransitionMatrices(
        int eigenIndex,
        const int* probabilityIndices,
        const int* firstDerivativeIndices,
        const int* secondDerivativeIndices,
        const double* edgeLengths,
        int count) {
    int rc = BeagleGPUImpl<Real>::updateTransitionMatrices(eigenIndex, probabilityIndices,
                                                            firstDerivativeIndices,
                                                            secondDerivativeIndices,
                                                            edgeLengths, count);
    if (rc != BEAGLE_SUCCESS || firstDerivativeIndices != NULL) return rc;

    if (count > 0) {
        const double* categoryRates = this->hCategoryRates[0];
        int nCat = this->kCategoryCount;
        Real* hDist = (Real*) malloc(sizeof(Real) * nCat);
        GPUInterface* gpuIf = this->gpu;
        for (int i = 0; i < count; i++) {
            int matIdx = probabilityIndices[i];
            hEigenIndexForMatrix[matIdx] = eigenIndex;
            for (int j = 0; j < nCat; j++) {
                hDist[j] = (Real)(edgeLengths[i] * categoryRates[j]);
            }
            gpuIf->MemcpyHostToDevice(dSpectralDistances[matIdx], hDist, sizeof(Real) * nCat);
        }
        free(hDist);
    }
    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
void BeagleGPUSpectralImpl<BEAGLE_GPU_GENERIC>::dispatchPrunePP(
        GPUPtr p1, GPUPtr p2, GPUPtr p3,
        int c1MatIdx, int c2MatIdx,
        GPUPtr scalingFactors, GPUPtr cumulativeScaling,
        unsigned int startPattern, unsigned int endPattern,
        int rescale, int streamIndex, int waitIndex) {
    int ei1 = hEigenIndexForMatrix[c1MatIdx];
    int ei2 = hEigenIndexForMatrix[c2MatIdx];
    this->kernels->PartialsPartialsPruningSpectral(p1, p2, p3,
        this->dIevc[ei1], this->dEvec[ei1], this->dEigenValues[ei1], dSpectralDistances[c1MatIdx],
        this->dIevc[ei2], this->dEvec[ei2], this->dEigenValues[ei2], dSpectralDistances[c2MatIdx],
        scalingFactors, cumulativeScaling,
        this->kPaddedPatternCount, this->kCategoryCount,
        rescale, streamIndex, waitIndex);
}

BEAGLE_GPU_TEMPLATE
void BeagleGPUSpectralImpl<BEAGLE_GPU_GENERIC>::dispatchPruneSP(
        GPUPtr s1, GPUPtr p2, GPUPtr p3,
        int c1MatIdx, int c2MatIdx,
        GPUPtr scalingFactors, GPUPtr cumulativeScaling,
        unsigned int startPattern, unsigned int endPattern,
        int rescale, int streamIndex, int waitIndex) {
    int ei1 = hEigenIndexForMatrix[c1MatIdx];
    int ei2 = hEigenIndexForMatrix[c2MatIdx];
    this->kernels->StatesPartialsPruningSpectral(s1, p2, p3,
        this->dIevc[ei1], this->dEvec[ei1], this->dEigenValues[ei1], dSpectralDistances[c1MatIdx],
        this->dIevc[ei2], this->dEvec[ei2], this->dEigenValues[ei2], dSpectralDistances[c2MatIdx],
        scalingFactors, cumulativeScaling,
        this->kPaddedPatternCount, this->kCategoryCount,
        rescale, streamIndex, waitIndex);
}

BEAGLE_GPU_TEMPLATE
void BeagleGPUSpectralImpl<BEAGLE_GPU_GENERIC>::dispatchPruneSS(
        GPUPtr s1, GPUPtr s2, GPUPtr p3,
        int c1MatIdx, int c2MatIdx,
        GPUPtr scalingFactors, GPUPtr cumulativeScaling,
        unsigned int startPattern, unsigned int endPattern,
        int rescale, int streamIndex, int waitIndex) {
    int ei1 = hEigenIndexForMatrix[c1MatIdx];
    int ei2 = hEigenIndexForMatrix[c2MatIdx];
    this->kernels->StatesStatesPruningSpectral(s1, s2, p3,
        this->dIevc[ei1], this->dEvec[ei1], this->dEigenValues[ei1], dSpectralDistances[c1MatIdx],
        this->dIevc[ei2], this->dEvec[ei2], this->dEigenValues[ei2], dSpectralDistances[c2MatIdx],
        scalingFactors, cumulativeScaling,
        this->kPaddedPatternCount, this->kCategoryCount,
        rescale, streamIndex, waitIndex);
}

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

#ifdef CUDA
template<>
char* BeagleGPUSpectralImpl<double>::getInstanceName() {
    return (char*) "CUDA-Double-Spectral";
}

template<>
char* BeagleGPUSpectralImpl<float>::getInstanceName() {
    return (char*) "CUDA-Single-Spectral";
}
#else
template<>
char* BeagleGPUSpectralImpl<double>::getInstanceName() {
    return (char*) "OpenCL-Double-Spectral";
}

template<>
char* BeagleGPUSpectralImpl<float>::getInstanceName() {
    return (char*) "OpenCL-Single-Spectral";
}
#endif

BEAGLE_GPU_TEMPLATE
int BeagleGPUSpectralImpl<BEAGLE_GPU_GENERIC>::getInstanceDetails(BeagleInstanceDetails* returnInfo) {
    int rc = BeagleGPUImpl<Real>::getInstanceDetails(returnInfo);
    if (rc == BEAGLE_SUCCESS && returnInfo != NULL) {
        returnInfo->flags |= BEAGLE_FLAG_SPECTRAL_REPRESENTATION;
        returnInfo->implName = getInstanceName();
    }
    return rc;
}

} // namespace cuda/opencl
} // namespace gpu
} // namespace beagle

#endif // __BeagleGPUSpectralImpl_hpp__
