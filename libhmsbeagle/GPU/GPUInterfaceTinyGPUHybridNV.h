/*
 * GPUInterfaceTinyGPUHybridNV.h
 *
 * Entry points implemented in GPUInterfaceTinyGPUHybridNV.cpp, called from
 * GPUInterfaceTinyGPUHybrid.cpp's vendor branch (GPUInterface::isNVIDIA ==
 * true) inside each shared GPUInterface method — mirrors
 * GPUInterfaceTinyGPUHybridAMD.h's role for the AMD branch exactly. One
 * GPUInterface method definition per name can exist in the
 * hmsbeagle-tinygpu-hybrid target (link-wise), so these stay as plain free
 * functions rather than a second set of GPUInterface member definitions.
 */

#ifndef LIBHMSBEAGLE_GPU_TINYGPUHYBRIDNV_H
#define LIBHMSBEAGLE_GPU_TINYGPUHYBRIDNV_H

#ifdef FW_TINYGPU

#include "libhmsbeagle/GPU/GPUInterface.h"

namespace tinygpu_device {

// From Initialize (TODO.md plan step P5): 1 when this process's booted GPU now serves self too, 0 when none is booted,
// -1 when another instance's GPU cannot be shared (refused, with a message).
int        NvAttachShared(GPUInterface* self);
void       NvSetDevice(GPUInterface* self, int paddedStateCount, int categoryCount,
                        int patternCount, int unpaddedPatternCount, int tipCount, long flags);
GPUFunction NvGetFunction(GPUInterface* self, const char* name);
void       NvLaunchKernelImpl(GPUInterface* self, GPUFunction fn, Dim3Int block, Dim3Int grid,
                               int nPtr, int nTotal, GPUPtr* ptrs, unsigned int* ints);
void       NvSynchronizeHost(GPUInterface* self);
GPUPtr     NvAllocateMemory(size_t sz);
void       NvMemcpyHostToDevice(GPUInterface* self, GPUPtr dst, const void* src, size_t sz);
void       NvMemcpyDeviceToHost(GPUInterface* self, void* dst, const GPUPtr src, size_t sz);
size_t     NvGetAvailableMemory();
void       NvFini(GPUInterface* self);   // called from the destructor: releases self's instance

} // namespace tinygpu_device

#endif // FW_TINYGPU
#endif // LIBHMSBEAGLE_GPU_TINYGPUHYBRIDNV_H
