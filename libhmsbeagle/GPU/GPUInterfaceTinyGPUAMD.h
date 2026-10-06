/*
 * GPUInterfaceTinyGPUAMD.h
 *
 * Entry points implemented in GPUInterfaceTinyGPUAMD.cpp, called from
 * GPUInterfaceTinyGPU.cpp's vendor branch (GPUInterface::isNVIDIA ==
 * false) inside each shared GPUInterface method. One GPUInterface method
 * definition per name can exist in the hmsbeagle-tinygpu target
 * (link-wise), so these stay as plain free functions rather than a second
 * set of GPUInterface member definitions.
 */

#ifndef LIBHMSBEAGLE_GPU_TINYGPUAMD_H
#define LIBHMSBEAGLE_GPU_TINYGPUAMD_H

#ifdef FW_TINYGPU

#include "libhmsbeagle/GPU/GPUInterface.h"

namespace tinygpu_device {

int        AmdAttachShared(GPUInterface* self);   // TODO.md plan step A5: 1 if it shares the card another instance booted
void       AmdSetDevice(GPUInterface* self, int paddedStateCount, int categoryCount,
                         int patternCount, int unpaddedPatternCount, int tipCount, long flags);
GPUFunction AmdGetFunction(GPUInterface* self, const char* name);
void       AmdLaunchKernelImpl(GPUInterface* self, GPUFunction fn, Dim3Int block, Dim3Int grid,
                                int nPtr, int nTotal, GPUPtr* ptrs, unsigned int* ints);
void       AmdSynchronizeHost(GPUInterface* self);
GPUPtr     AmdAllocateMemory(GPUInterface* self, size_t sz);
void       AmdFreeMemory(GPUInterface* self, GPUPtr p);   // TODO.md plan step C14
void       AmdMemcpyHostToDevice(GPUInterface* self, GPUPtr dst, const void* src, size_t sz);
void       AmdMemcpyDeviceToHost(GPUInterface* self, void* dst, const GPUPtr src, size_t sz);
size_t     AmdGetAvailableMemory();
bool       AmdDeviceLost(GPUInterface* self);   // TODO.md plan step A3: this instance's setup failed or its GPU is lost (GPUInterface::GetDeviceLost)
bool       AmdOutOfMemory(GPUInterface* self);   // TODO.md plan step M1: ... for lack of GPU memory
bool       AmdGpuHeld();      // ... and a crash guard of this process holds the card: Initialize must not connect
bool       AmdSupportsDouble();   // TODO.md plan step A7: the build's HSACOs include double precision's
void       AmdFini(GPUInterface* self);   // called from the destructor: releases the instance (the card stays until exit, plan step A5)

} // namespace tinygpu_device

#endif // FW_TINYGPU
#endif // LIBHMSBEAGLE_GPU_TINYGPUAMD_H
