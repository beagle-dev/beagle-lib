/*
 * @file BeagleGPUFusedImpl.hpp
 *
 * Copyright 2026 Phylogenetic Likelihood Working Group
 *
 * This file is part of BEAGLE.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 *
 * @brief TEST GPU fused implementation template definitions
 *
 * @author Marc Suchard
 */

#ifndef __BeagleGPUFusedImpl_hpp__
#define __BeagleGPUFusedImpl_hpp__

#include "libhmsbeagle/beagle.h"

namespace beagle {
namespace gpu {

#ifdef CUDA
    namespace cuda {
#elif defined(FW_TINYGPU)
    namespace tinygpu {
#else
    namespace opencl {
#endif

///////////////////////////////////////////////////////////////////////////////
// BeagleGPUFusedImpl

BEAGLE_GPU_TEMPLATE
BeagleGPUFusedImpl<BEAGLE_GPU_GENERIC>::BeagleGPUFusedImpl()
    : fFused(0), dFusedOps(0), kOpCapacity(0), hFusedOps(NULL), kFusedOpsBytes(0), kFusedOpsPos(0) {
}

BEAGLE_GPU_TEMPLATE
BeagleGPUFusedImpl<BEAGLE_GPU_GENERIC>::~BeagleGPUFusedImpl() {
    if (dFusedOps) {
        this->gpu->SynchronizeHost();   // the last uploads from hFusedOps have completed
        this->gpu->FreeMemory(dFusedOps);
    }
    if (hFusedOps) this->gpu->FreeHostMemory(hFusedOps);
}

BEAGLE_GPU_TEMPLATE
void BeagleGPUFusedImpl<BEAGLE_GPU_GENERIC>::allocateOpTable(int capacity) {
    const size_t tableBytes = (size_t) capacity * kOpTableWords * sizeof(unsigned int);
    kOpCapacity = capacity;
    dFusedOps = this->gpu->AllocateMemory(tableBytes);
    kFusedOpsBytes = 64 * tableBytes;   // at most one wrap sync per 64 full-capacity calls
    if (kFusedOpsBytes < ((size_t) 16 << 20))
        kFusedOpsBytes = (size_t) 16 << 20;
    hFusedOps = (unsigned char*) this->gpu->MallocHost(kFusedOpsBytes);
    kFusedOpsPos = 0;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUFusedImpl<BEAGLE_GPU_GENERIC>::createInstance(
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
    // no scaling and manual scaling only; the base takes AUTO, ALWAYS or DYNAMIC even when only preferred
    const long unsupported = BEAGLE_FLAG_SCALING_AUTO | BEAGLE_FLAG_SCALING_ALWAYS | BEAGLE_FLAG_SCALING_DYNAMIC |
                             BEAGLE_FLAG_SPECTRAL_REPRESENTATION;
    if (requirementFlags & unsupported) return BEAGLE_ERROR_NO_IMPLEMENTATION;
    preferenceFlags &= ~unsupported;

    if (stateCount > 4) return BEAGLE_ERROR_NO_IMPLEMENTATION;   // TEMPORARY (fused plan F3): the generic-state kernel is F4's

    int rc = BeagleGPUImpl<Real>::createInstance(tipCount, partialsBufferCount, compactBufferCount,
                                                  stateCount, patternCount, eigenDecompositionCount,
                                                  matrixCount, categoryCount, scaleBufferCount,
                                                  resourceNumber, pluginResourceNumber,
                                                  preferenceFlags, requirementFlags);
    if (rc != BEAGLE_SUCCESS) return rc;

#ifdef FW_OPENCL
    // kernelsFused.cu is compiled out for OpenCL CPU devices (FW_OPENCL_CPU)
    if (!(this->kFlags & BEAGLE_FLAG_PROCESSOR_GPU)) return BEAGLE_ERROR_NO_IMPLEMENTATION;
#endif

    fFused = this->gpu->GetFunction("kernelUpPartialsFused");

    int capacity = partialsBufferCount + compactBufferCount;
    if (capacity < 64) capacity = 64;
    allocateOpTable(capacity);

    BEAGLE_GPU_RETURN_IF_LOST();
    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUFusedImpl<BEAGLE_GPU_GENERIC>::updatePartials(const int* operations,
                                                            int operationCount,
                                                            int cumulativeScalingIndex,
                                                            BeaglePartialsType partialsType) {
    int rc = this->checkSingleChild(operations, operationCount, BEAGLE_OP_COUNT);
    if (rc != BEAGLE_SUCCESS) return rc;
    if (operationCount == 0) return BEAGLE_SUCCESS;

    GPUInterface* gpuIf = this->gpu;

    if (operationCount > kOpCapacity) {
        gpuIf->SynchronizeHost();   // the uploads from hFusedOps and the launches reading dFusedOps have completed
        gpuIf->FreeMemory(dFusedOps);
        gpuIf->FreeHostMemory(hFusedOps);
        allocateOpTable(operationCount > 2 * kOpCapacity ? operationCount : 2 * kOpCapacity);
    }

    const size_t tableBytes = (size_t) operationCount * kOpTableWords * sizeof(unsigned int);
    if (kFusedOpsPos + tableBytes > kFusedOpsBytes) {
        gpuIf->SynchronizeHost();   // every earlier upload from hFusedOps has completed
        kFusedOpsPos = 0;
    }
    unsigned int* rec = (unsigned int*) (hFusedOps + kFusedOpsPos);

    // the offsets are read on every call: ensureBufferCounts and pattern resorting move the pools
    const unsigned int matrixStride = this->getMatrixStrideElements();
    const unsigned int scaleSize    = this->getScaleBufferSizeElements();
    for (int op = 0; op < operationCount; op++) {
        const int* o = operations + op * BEAGLE_OP_COUNT;
        const int parIndex            = o[0];
        const int writeScalingIndex   = o[1];
        const int readScalingIndex    = o[2];
        const int child1Index         = o[3];
        const int child1TransMatIndex = o[4];
        const int child2Index         = o[5];
        const int child2TransMatIndex = o[6];

        const bool states1 = (this->dStates[child1Index] != 0);
        const bool states2 = (this->dStates[child2Index] != 0);
        unsigned int kind, c1Off, c2Off;
        unsigned int c1MOff = child1TransMatIndex * matrixStride;
        unsigned int c2MOff = child2TransMatIndex * matrixStride;
        if (states1 && states2) {
            kind  = 3;
            c1Off = this->getStatesOffsetElements(child1Index);
            c2Off = this->getStatesOffsetElements(child2Index);
        } else if (states1) {
            kind  = 2;
            c1Off = this->getStatesOffsetElements(child1Index);
            c2Off = this->getPartialsOffsetElements(child2Index);
        } else if (states2) {   // the states child goes first, with its matrix (as upPartials)
            kind  = 2;
            c1Off = this->getStatesOffsetElements(child2Index);
            c2Off = this->getPartialsOffsetElements(child1Index);
            unsigned int tmpOff = c1MOff; c1MOff = c2MOff; c2MOff = tmpOff;
        } else {
            kind  = 1;
            c1Off = this->getPartialsOffsetElements(child1Index);
            c2Off = this->getPartialsOffsetElements(child2Index);
        }

        unsigned int mode = 0, scaleOff = 0;   // write wins over read (as upPartials)
        if (writeScalingIndex >= 0) {
            mode     = 2;
            scaleOff = writeScalingIndex * scaleSize;
        } else if (readScalingIndex >= 0) {
            mode     = 1;
            scaleOff = readScalingIndex * scaleSize;
        }

        unsigned int* r = rec + op * kOpTableWords;
        r[0] = kind | (mode << 2);
        r[1] = 0;
        r[2] = c1Off;
        r[3] = c2Off;
        r[4] = this->getPartialsOffsetElements(parIndex);
        r[5] = c1MOff;
        r[6] = c2MOff;
        r[7] = scaleOff;
    }

    gpuIf->MemcpyHostToDeviceNoWait(dFusedOps, rec, tableBytes);
    kFusedOpsPos += tableBytes;

    int cumulativeOffset = 0;
    int fusedFlags = 0;   // FUSED_HAS_CUMULATIVE 1, FUSED_SCALERS_LOG 2 (kernelsFused.cu)
    if (cumulativeScalingIndex != BEAGLE_OP_NONE) {
        cumulativeOffset = cumulativeScalingIndex * scaleSize;
        fusedFlags |= 1;
    }
    if (this->kFlags & BEAGLE_FLAG_SCALERS_LOG)
        fusedFlags |= 2;

    // 4 states: each work-group computes all categories of 4 * R patterns; never dereferenced, the partials origin
    // stands in for a missing states or scale pool
    const int rowsPerCategory = 16 / this->kCategoryCount;
    Dim3Int block(16, rowsPerCategory, this->kCategoryCount);
    Dim3Int grid((this->kPaddedPatternCount + 4 * rowsPerCategory - 1) / (4 * rowsPerCategory), 1);
    GPUPtr partialsOrigin = this->getPartialsOrigin();
    GPUPtr statesOrigin   = this->getStatesOrigin();
    GPUPtr scaleOrigin    = this->getScalingFactorsOriginRaw();
    gpuIf->LaunchKernel(fFused, block, grid, 5, 11,
                        partialsOrigin,
                        statesOrigin ? statesOrigin : partialsOrigin,
                        this->getMatricesOrigin(),
                        scaleOrigin ? scaleOrigin : partialsOrigin,
                        dFusedOps,
                        operationCount, this->kPaddedPatternCount, this->kCategoryCount,
                        rowsPerCategory, cumulativeOffset, fusedFlags);

#ifdef BEAGLE_DEBUG_SYNCH
    gpuIf->SynchronizeHost();
#endif

    return BEAGLE_SUCCESS;
}

template<>
char* BeagleGPUFusedImpl<double>::getInstanceName() {
    return (char*) "OpenCL-Double-Fused";
}

template<>
char* BeagleGPUFusedImpl<float>::getInstanceName() {
    return (char*) "OpenCL-Single-Fused";
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUFusedImpl<BEAGLE_GPU_GENERIC>::getInstanceDetails(BeagleInstanceDetails* returnInfo) {
    int rc = BeagleGPUImpl<Real>::getInstanceDetails(returnInfo);
    if (rc == BEAGLE_SUCCESS && returnInfo != NULL) {
        returnInfo->flags |= BEAGLE_FLAG_PARALLELOPS_FUSED;
        returnInfo->implName = getInstanceName();
    }
    return rc;
}

///////////////////////////////////////////////////////////////////////////////
// BeagleGPUFusedImplFactory

BEAGLE_GPU_TEMPLATE
BeagleImpl* BeagleGPUFusedImplFactory<BEAGLE_GPU_GENERIC>::createImpl(
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
    // only on request: otherwise this factory would outrank the standard one, and *errorCode stays as it is
    if (!((preferenceFlags | requirementFlags) & BEAGLE_FLAG_PARALLELOPS_FUSED)) return NULL;

    BeagleImpl* impl = new BeagleGPUFusedImpl<BEAGLE_GPU_GENERIC>();
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

template<>
const char* BeagleGPUFusedImplFactory<double>::getName() {
    return "GPU-DP-OpenCL-Fused";
}

template<>
const char* BeagleGPUFusedImplFactory<float>::getName() {
    return "GPU-SP-OpenCL-Fused";
}

BEAGLE_GPU_TEMPLATE
const long BeagleGPUFusedImplFactory<BEAGLE_GPU_GENERIC>::getFlags() {
    long flags = BEAGLE_FLAG_COMPUTATION_SYNCH | BEAGLE_FLAG_COMPUTATION_ASYNCH |
          BEAGLE_FLAG_SCALING_MANUAL |
          BEAGLE_FLAG_THREADING_NONE |
          BEAGLE_FLAG_VECTOR_NONE |
          BEAGLE_FLAG_SCALERS_LOG | BEAGLE_FLAG_SCALERS_RAW |
          BEAGLE_FLAG_EIGEN_COMPLEX | BEAGLE_FLAG_EIGEN_REAL |
          BEAGLE_FLAG_INVEVEC_STANDARD | BEAGLE_FLAG_INVEVEC_TRANSPOSED |
          BEAGLE_FLAG_PREORDER_TRANSPOSE_MANUAL | BEAGLE_FLAG_PREORDER_TRANSPOSE_AUTO |
          BEAGLE_FLAG_PARALLELOPS_GRID | BEAGLE_FLAG_PARALLELOPS_STREAMS |
          BEAGLE_FLAG_PARALLELOPS_FUSED;

#ifdef CUDA
    flags |= BEAGLE_FLAG_FRAMEWORK_CUDA |
             BEAGLE_FLAG_PROCESSOR_GPU;
#elif defined(FW_OPENCL)
    flags |= BEAGLE_FLAG_FRAMEWORK_OPENCL |
             BEAGLE_FLAG_PROCESSOR_CPU | BEAGLE_FLAG_PROCESSOR_GPU | BEAGLE_FLAG_PROCESSOR_OTHER;
#elif defined(FW_TINYGPU)
    flags |= BEAGLE_FLAG_FRAMEWORK_TINYGPU |
             BEAGLE_FLAG_PROCESSOR_GPU;
#endif

    Real r = 0;
    modifyFlagsForPrecision(&flags, r);
    return flags;
}

} // namespace cuda/tinygpu/opencl
} // namespace gpu
} // namespace beagle

#endif // __BeagleGPUFusedImpl_hpp__
