/*
 * @file BeagleGPUFusedImpl.h
 *
 * Copyright 2026 Phylogenetic Likelihood Working Group
 *
 * This file is part of BEAGLE.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 *
 * @brief TEST GPU implementation whose updatePartials makes one kernel launch per call (BEAGLE_FLAG_PARALLELOPS_FUSED)
 *
 * @author Marc Suchard
 */

#ifndef __BeagleGPUFusedImpl__
#define __BeagleGPUFusedImpl__

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include "libhmsbeagle/GPU/BeagleGPUImpl.h"

namespace beagle {
namespace gpu {

#ifdef CUDA
    namespace cuda {
#elif defined(FW_TINYGPU)
    namespace tinygpu {
#else
    namespace opencl {
#endif

/* updatePartials encodes the operations into an op table (8 unsigned ints per operation: kind | mode << 2, 0, child 1,
 * child 2, parent, child-1 matrix, child-2 matrix and scale offsets) and makes one launch of kernelUpPartialsFused
 * (kernels/kernelsFused.cu), which loops over the operations. No scaling and BEAGLE_FLAG_SCALING_MANUAL only. */
BEAGLE_GPU_TEMPLATE
class BeagleGPUFusedImpl : public BeagleGPUImpl<Real> {
private:
    static const int kOpTableWords = 8;   // unsigned ints per operation

    GPUFunction    fFused;
    GPUPtr         dFusedOps;    // the op table of the current call
    int            kOpCapacity;  // operations dFusedOps holds
    /* The host bytes the op tables are uploaded from. On OpenCL an event-free arena: each call encodes its table into
     * bytes no earlier call used and uploads them without waiting, so bytes are reused only after a blocking call on
     * the (single, in-order) queue has returned: at a wrap, a capacity growth and in the destructor. */
    unsigned char* hFusedOps;
    size_t         kFusedOpsBytes;
    size_t         kFusedOpsPos;

    void allocateOpTable(int capacity);

public:
    BeagleGPUFusedImpl();
    ~BeagleGPUFusedImpl() override;

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

    int updatePartials(const int* operations,
                       int operationCount,
                       int cumulativeScalingIndex,
                       BeaglePartialsType partialsType) override;

    char* getInstanceName() override;
    int getInstanceDetails(BeagleInstanceDetails* returnInfo) override;
};

BEAGLE_GPU_TEMPLATE
class BeagleGPUFusedImplFactory : public BeagleImplFactory {
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

} // namespace cuda/tinygpu/opencl
} // namespace gpu
} // namespace beagle

#include "libhmsbeagle/GPU/BeagleGPUFusedImpl.hpp"

#endif // __BeagleGPUFusedImpl__
