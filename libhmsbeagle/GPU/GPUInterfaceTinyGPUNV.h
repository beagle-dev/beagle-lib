/*
 * GPUInterfaceTinyGPUNV.h
 *
 * Entry points implemented in GPUInterfaceTinyGPUNV.cpp, called from
 * GPUInterfaceTinyGPU.cpp's vendor branch (GPUInterface::isNVIDIA ==
 * true) inside each shared GPUInterface method — mirrors
 * GPUInterfaceTinyGPUAMD.h's role for the AMD branch exactly. One
 * GPUInterface method definition per name can exist in the
 * hmsbeagle-tinygpu target (link-wise), so these stay as plain free
 * functions rather than a second set of GPUInterface member definitions.
 */

#ifndef LIBHMSBEAGLE_GPU_TINYGPUNV_H
#define LIBHMSBEAGLE_GPU_TINYGPUNV_H

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
GPUPtr     NvAllocateMemory(GPUInterface* self, size_t sz);
void       NvFreeMemory(GPUInterface* self, GPUPtr p);   // TODO.md plan step C14
void       NvMemcpyHostToDevice(GPUInterface* self, GPUPtr dst, const void* src, size_t sz);
void       NvMemcpyDeviceToHost(GPUInterface* self, void* dst, const GPUPtr src, size_t sz);
size_t     NvGetAvailableMemory();
void       NvFini(GPUInterface* self);   // called from the destructor: releases self's instance
// TODO.md plan step C12: true once self's setup or an allocation failed or the GPU is lost; its calls then do nothing, and
// BeagleGPUImpl returns errors
bool       NvDeviceLost(GPUInterface* self);
bool       NvOutOfMemory(GPUInterface* self);   // TODO.md plan step M1: ... for lack of GPU memory
bool       NvSupportsDouble();   // TODO.md plan step C16: the build's cubins include double precision's

// Implemented in GPUInterfaceTinyGPU.cpp: the PCI device ID Initialize's probe read (TODO.md plan decision 16).
uint16_t   tg_pci_device_id();

} // namespace tinygpu_device

#endif // FW_TINYGPU
#endif // LIBHMSBEAGLE_GPU_TINYGPUNV_H
