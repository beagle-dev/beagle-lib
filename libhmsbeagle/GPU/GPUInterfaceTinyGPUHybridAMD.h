/*
 * GPUInterfaceTinyGPUHybridAMD.h
 *
 * Entry points implemented in GPUInterfaceTinyGPUHybridAMD.cpp, called from
 * GPUInterfaceTinyGPUHybrid.cpp's vendor branch (GPUInterface::isNVIDIA ==
 * false) inside each shared GPUInterface method. One GPUInterface method
 * definition per name can exist in the hmsbeagle-tinygpu-hybrid target
 * (link-wise), so these stay as plain free functions rather than a second
 * set of GPUInterface member definitions.
 */

#ifndef LIBHMSBEAGLE_GPU_TINYGPUHYBRIDAMD_H
#define LIBHMSBEAGLE_GPU_TINYGPUHYBRIDAMD_H

#ifdef FW_TINYGPU

#include "libhmsbeagle/GPU/GPUInterface.h"

namespace tinygpu_device {

void       AmdSetDevice(GPUInterface* self, int paddedStateCount, int categoryCount,
                         int patternCount, int unpaddedPatternCount, int tipCount, long flags);
GPUFunction AmdGetFunction(const char* name);
void       AmdLaunchKernelImpl(GPUFunction fn, Dim3Int block, Dim3Int grid,
                                int nPtr, int nTotal, GPUPtr* ptrs, unsigned int* ints);
void       AmdSynchronizeHost();
GPUPtr     AmdAllocateMemory(size_t sz);
void       AmdMemcpyHostToDevice(GPUPtr dst, const void* src, size_t sz);
void       AmdMemcpyDeviceToHost(void* dst, const GPUPtr src, size_t sz);
size_t     AmdGetAvailableMemory();
void       AmdFini();   // called from the destructor; also usable as a safe_exit fallback

} // namespace tinygpu_device

#endif // FW_TINYGPU
#endif // LIBHMSBEAGLE_GPU_TINYGPUHYBRIDAMD_H
