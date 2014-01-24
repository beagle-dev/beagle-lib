/*
 * KernelResource.cpp
 *
 *  Created on: Sep 7, 2009
 *      Author: msuchard
 */

#include "KernelResource.h"


//#ifdef CUDA
//    using namespace cuda_device;
//#else
//#ifdef FW_OPENCL
//    using namespace opencl_device;
//#endif
//#endif

KernelResource::KernelResource() {
}

KernelResource::KernelResource(
        int inPaddedStateCount,
        char* inKernelString,
        int inPatternBlockSize,
        int inMatrixBlockSize,
        int inBlockPeelingSize,
        int inSlowReweighing,
        int inMultiplyBlockSize,
        int inCategoryCount,
        int inPatternCount,
        int inUnpaddedPatternCount,
        long inFlags
        ) {
    paddedStateCount = inPaddedStateCount;
    kernelCode = inKernelString;
    patternBlockSize = inPatternBlockSize;
    matrixBlockSize = inMatrixBlockSize;
    blockPeelingSize = inBlockPeelingSize,
    slowReweighing = inSlowReweighing;
    multiplyBlockSize = inMultiplyBlockSize;
    categoryCount = inCategoryCount;
    patternCount = inPatternCount;
    unpaddedPatternCount = inUnpaddedPatternCount;
    flags = inFlags;
}

KernelResource::KernelResource(const KernelResource& krIn,
                               char* inKernelCode) {
    paddedStateCount = krIn.paddedStateCount;
    kernelCode = inKernelCode;
    patternBlockSize = krIn.patternBlockSize;
    matrixBlockSize = krIn.matrixBlockSize;
    blockPeelingSize = krIn.blockPeelingSize,
    slowReweighing = krIn.slowReweighing;
    multiplyBlockSize = krIn.multiplyBlockSize;
    categoryCount = krIn.categoryCount;
    patternCount = krIn.patternCount;
    unpaddedPatternCount = krIn.unpaddedPatternCount;
    flags = krIn.flags;
}

KernelResource::~KernelResource() {
}

KernelResource* KernelResource::copy(void) {
    return new KernelResource(
            paddedStateCount,
            kernelCode,
            patternBlockSize,
            matrixBlockSize,
            blockPeelingSize,
            slowReweighing,
            multiplyBlockSize,
            categoryCount,
            patternCount,
            unpaddedPatternCount,
            flags);
}
