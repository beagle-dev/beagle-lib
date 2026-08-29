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

void       NvSetDevice(GPUInterface* self, int paddedStateCount, int categoryCount,
                        int patternCount, int unpaddedPatternCount, int tipCount, long flags);
GPUFunction NvGetFunction(const char* name);
void       NvLaunchKernelImpl(GPUFunction fn, Dim3Int block, Dim3Int grid,
                               int nPtr, int nTotal, GPUPtr* ptrs, unsigned int* ints);
void       NvSynchronizeHost();
GPUPtr     NvAllocateMemory(size_t sz);
void       NvMemcpyHostToDevice(GPUPtr dst, const void* src, size_t sz);
void       NvMemcpyDeviceToHost(void* dst, const GPUPtr src, size_t sz);
size_t     NvGetAvailableMemory();
void       NvFini();   // called from the destructor; also usable as a safe_exit fallback

} // namespace tinygpu_device

#endif // FW_TINYGPU
#endif // LIBHMSBEAGLE_GPU_TINYGPUHYBRIDNV_H
