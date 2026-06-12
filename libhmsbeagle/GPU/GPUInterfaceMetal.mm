/*
 *
 * Copyright 2009 Phylogenetic Likelihood Working Group
 *
 * This file is part of BEAGLE.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 *
 * @author Marc Suchard
 */

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cstdarg>
#include <cmath>
#include <map>
#include <string>
#include <utility>

#include "libhmsbeagle/beagle.h"
#include "libhmsbeagle/GPU/GPUImplDefs.h"
#include "libhmsbeagle/GPU/GPUImplHelper.h"
#include "libhmsbeagle/GPU/GPUInterface.h"
#include "libhmsbeagle/GPU/KernelResource.h"

// ============================================================
// Bridge-cast helpers (avoid repetitive casts throughout)
// ============================================================
#define MTL_DEVICE(p)   ((__bridge id<MTLDevice>)(p))
#define MTL_QUEUE(p)    ((__bridge id<MTLCommandQueue>)(p))
#define MTL_LIBRARY(p)  ((__bridge id<MTLLibrary>)(p))
#define MTL_BUFFER(p)   ((__bridge id<MTLBuffer>)(p))
#define MTL_PSO(p)      ((__bridge id<MTLComputePipelineState>)(p))

// Retain an ObjC object and store it as a void* (caller owns the retain count).
#define RETAIN_VOIDPTR(obj)       ((__bridge_retained void*)(obj))
// Release a retained void* back to ARC (the assignment to a __bridge_transfer local
// triggers ARC to send a release message).
#define RELEASE_VOIDPTR(p, T)     { id<T> _tmp_ = (__bridge_transfer id<T>)(p); (void)_tmp_; }

// ============================================================
// LOAD_KERNEL_INTO_RESOURCE — matches GPUInterfaceOpenCL.cpp exactly
// ============================================================
#define LOAD_KERNEL_INTO_RESOURCE(state, prec, id, impl, impl2, impl3) \
        kernelResource = new KernelResource( \
            state, \
            (char*) KERNELS_STRING_##prec##_##state, \
            PATTERN_BLOCK_SIZE_##prec##_##state##impl, \
            MATRIX_BLOCK_SIZE_##prec##_##state##impl2, \
            BLOCK_PEELING_SIZE_##prec##_##state##impl2, \
            SLOW_REWEIGHING_##prec##_##state, \
            MULTIPLY_BLOCK_SIZE_##prec##impl3, \
            BASTA_SUM_INTERVAL_BLOCK_SIZE_##prec##_##state, \
            BASTA_SUM_ACROSS_BLOCK_SIZE_##prec##_##state, \
            BLOCK_PEELING_SIZE_SCA_##prec##_##state, \
            0, 0, 0, 0);

// Lightweight sentinel allocated by CreateSubPointer to give sub-ranges a
// unique address while recording the parent MTLBuffer and byte offset.
struct MetalSubBuffer {
    void*  parentPtr;   // retained void* → id<MTLBuffer>
    size_t offset;
    size_t size;
};

namespace metal_device {

// ============================================================
// Constructor / Destructor
// ============================================================

GPUInterface::GPUInterface() {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::GPUInterface (Metal)\n");
#endif
    kernelResource     = NULL;
    metalDeviceId      = NULL;
    metalCommandQueues = NULL;
    metalLibrary       = NULL;
    supportDoublePrecision = false; // Metal on Apple Silicon does not support double
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::GPUInterface (Metal)\n");
#endif
}

GPUInterface::~GPUInterface() {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::~GPUInterface (Metal)\n");
#endif
    if (metalLibrary) {
        RELEASE_VOIDPTR(metalLibrary, MTLLibrary);
        metalLibrary = NULL;
    }
    if (metalCommandQueues) {
        for (int i = 0; i < BEAGLE_STREAM_COUNT; i++) {
            if (metalCommandQueues[i]) {
                RELEASE_VOIDPTR(metalCommandQueues[i], MTLCommandQueue);
                metalCommandQueues[i] = NULL;
            }
        }
        free(metalCommandQueues);
        metalCommandQueues = NULL;
    }
    for (auto& kv : metalDeviceMap) {
        if (kv.second)
            RELEASE_VOIDPTR(kv.second, MTLDevice);
    }
    metalDeviceMap.clear();
    // Sub-buffer sentinels are freed in FreeMemory; clean up any stragglers here.
    for (auto& kv : metalSubBufferMap) {
        delete static_cast<MetalSubBuffer*>(kv.first);
    }
    metalSubBufferMap.clear();
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::~GPUInterface (Metal)\n");
#endif
}

// ============================================================
// Initialize — enumerate all Metal-capable GPU devices
// ============================================================
int GPUInterface::Initialize() {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::Initialize (Metal)\n");
#endif
    NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
    int idx = 0;
    for (id<MTLDevice> dev in devices) {
        metalDeviceMap[idx++] = RETAIN_VOIDPTR(dev);
    }
#ifdef BEAGLE_DEBUG_VALUES
    printf("Metal devices: %zu\n", metalDeviceMap.size());
    for (int i = 0; i < (int)metalDeviceMap.size(); i++) {
        printf("  Device %d: %s\n", i, [[MTL_DEVICE(metalDeviceMap[i]) name] UTF8String]);
    }
#endif
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::Initialize (Metal)\n");
#endif
    return metalDeviceMap.size() ? 1 : 0;
}

// ============================================================
// GetDeviceCount
// ============================================================
int GPUInterface::GetDeviceCount() {
    return (int)metalDeviceMap.size();
}

// ============================================================
// InitializeKernelResource
// Metal does not support double precision on Apple Silicon.
// The DP kernel strings are defined as nullptr in BeagleMetal_kernels.h;
// we always load SP kernels and ignore the doublePrecision flag.
// ============================================================
void GPUInterface::InitializeKernelResource(int paddedStateCount, bool doublePrecision) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::InitializeKernelResource (Metal)\n");
#endif
    int id = paddedStateCount; // always use SP (no DP on Apple Silicon Metal)
    switch (id) {
        case   4: LOAD_KERNEL_INTO_RESOURCE(  4, SP,   4,,,); break;
        case  16: LOAD_KERNEL_INTO_RESOURCE( 16, SP,  16,,,); break;
        case  32: LOAD_KERNEL_INTO_RESOURCE( 32, SP,  32,,,); break;
        case  48: LOAD_KERNEL_INTO_RESOURCE( 48, SP,  48,,,); break;
        case  64: LOAD_KERNEL_INTO_RESOURCE( 64, SP,  64,,,); break;
        case  80: LOAD_KERNEL_INTO_RESOURCE( 80, SP,  80,,,); break;
        case 128: LOAD_KERNEL_INTO_RESOURCE(128, SP, 128,,,); break;
        case 192: LOAD_KERNEL_INTO_RESOURCE(192, SP, 192,,,); break;
        case 256: LOAD_KERNEL_INTO_RESOURCE(256, SP, 256,,,); break;
    }
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::InitializeKernelResource (Metal)\n");
#endif
}

// ============================================================
// SetDevice — create command queues, compile the MSL library
// ============================================================
void GPUInterface::SetDevice(int deviceNumber,
                              int paddedStateCount,
                              int categoryCount,
                              int paddedPatternCount,
                              int unpaddedPatternCount,
                              int tipCount,
                              long flags) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::SetDevice (Metal)\n");
#endif
    metalDeviceId = metalDeviceMap[deviceNumber];
    id<MTLDevice> device = MTL_DEVICE(metalDeviceId);

    // Allocate command queues (one per stream slot).
    metalCommandQueues = (void**)calloc(BEAGLE_STREAM_COUNT, sizeof(void*));
    for (int i = 0; i < BEAGLE_STREAM_COUNT; i++) {
        id<MTLCommandQueue> q = [device newCommandQueue];
        if (!q) {
            fprintf(stderr, "Metal error: failed to create command queue %d\n", i);
            exit(-1);
        }
        metalCommandQueues[i] = RETAIN_VOIDPTR(q);
    }

    // Select kernel source for this state count.
    InitializeKernelResource(paddedStateCount, flags & BEAGLE_FLAG_PRECISION_DOUBLE);
    if (!kernelResource) {
        fprintf(stderr, "Metal error: no kernel resource for %d states\n", paddedStateCount);
        exit(-1);
    }
    if (!kernelResource->kernelCode) {
        fprintf(stderr, "Metal error: double-precision kernels are not supported on Apple Silicon Metal\n");
        exit(-1);
    }

    kernelResource->categoryCount        = categoryCount;
    kernelResource->patternCount         = paddedPatternCount;
    kernelResource->unpaddedPatternCount = unpaddedPatternCount;
    kernelResource->flags                = flags;

    // Build the preprocessor defines for the Metal compiler.
    // PADDED_STATE_COUNT and MULTIPLY_BLOCK_SIZE are required by the kernel source.
    NSDictionary<NSString*, NSObject*>* defs = @{
        @"FW_METAL":             @(1),
        @"OPENCL_KERNEL_BUILD":  @(1),
        @"PADDED_STATE_COUNT":   @(paddedStateCount),
        @"PATTERN_BLOCK_SIZE":   @(kernelResource->patternBlockSize),
        @"MATRIX_BLOCK_SIZE":    @(kernelResource->matrixBlockSize),
        @"BLOCK_PEELING_SIZE":   @(kernelResource->blockPeelingSize),
        @"MULTIPLY_BLOCK_SIZE":  @(kernelResource->multiplyBlockSize),
    };

    MTLCompileOptions* opts = [MTLCompileOptions new];
    opts.preprocessorMacros = defs;
    // Language version: Metal 2.3+ required for full atomics support.
    opts.languageVersion = MTLLanguageVersion2_3;

    NSString* src = [NSString stringWithUTF8String:kernelResource->kernelCode];
    NSError*  err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:opts error:&err];
    if (!lib) {
        fprintf(stderr, "Metal error: failed to compile kernel library for %d states:\n%s\n",
                paddedStateCount, [[err localizedDescription] UTF8String]);
        exit(-1);
    }
    metalLibrary = RETAIN_VOIDPTR(lib);

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::SetDevice (Metal)\n");
#endif
}

// ============================================================
// ResizeStreamCount — queues are pre-allocated up to BEAGLE_STREAM_COUNT
// ============================================================
void GPUInterface::ResizeStreamCount(int newStreamCount) {
    // No-op: BEAGLE_STREAM_COUNT command queues already created in SetDevice.
}

// ============================================================
// Synchronization
// ============================================================
void GPUInterface::SynchronizeHost() {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::SynchronizeHost (Metal)\n");
#endif
    // LaunchKernel() already waits for each command buffer to complete
    // (synchronous dispatch), so there is nothing to flush here.
    // When async dispatch is added, drain all pending command buffers here.
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::SynchronizeHost (Metal)\n");
#endif
}

void GPUInterface::SynchronizeDevice() {
    // Current implementation is synchronous per-launch; no-op.
}

void GPUInterface::SynchronizeDeviceWithIndex(int streamRecordIndex, int streamWaitIndex) {
    // Current implementation is synchronous per-launch; no-op.
}

// ============================================================
// GetFunction — compile and cache per-kernel pipeline state objects
// ============================================================
GPUFunction GPUInterface::GetFunction(const char* functionName) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::GetFunction (Metal): %s\n", functionName);
#endif
    id<MTLDevice>  device  = MTL_DEVICE(metalDeviceId);
    id<MTLLibrary> library = MTL_LIBRARY(metalLibrary);

    NSString* name = [NSString stringWithUTF8String:functionName];
    id<MTLFunction> fn = [library newFunctionWithName:name];
    if (!fn) {
        fprintf(stderr, "Metal error: kernel function '%s' not found in library\n", functionName);
        exit(-1);
    }

    NSError* err = nil;
    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
        fprintf(stderr, "Metal error: pipeline state creation failed for '%s': %s\n",
                functionName, [[err localizedDescription] UTF8String]);
        exit(-1);
    }

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::GetFunction (Metal)\n");
#endif
    return RETAIN_VOIDPTR(pso);
}

// ============================================================
// Internal helpers shared by LaunchKernel and LaunchKernelConcurrent
// ============================================================
static void setKernelArgs(id<MTLComputeCommandEncoder> enc,
                           int parameterCountV,
                           int totalParameterCount,
                           va_list parameters,
                           const std::map<void*, std::pair<void*, size_t>>& subBufMap) {
    for (int i = 0; i < parameterCountV; i++) {
        void* ptr = (void*)(size_t)va_arg(parameters, GPUPtr);
        auto it = subBufMap.find(ptr);
        if (it != subBufMap.end()) {
            id<MTLBuffer> buf = MTL_BUFFER(it->second.first);
            [enc setBuffer:buf offset:it->second.second atIndex:i];
        } else {
            id<MTLBuffer> buf = MTL_BUFFER(ptr);
            [enc setBuffer:buf offset:0 atIndex:i];
        }
    }
    for (int i = parameterCountV; i < totalParameterCount; i++) {
        unsigned int val = va_arg(parameters, unsigned int);
        [enc setBytes:&val length:sizeof(unsigned int) atIndex:i];
    }
}

// ============================================================
// LaunchKernel (synchronous — waits for completion)
// ============================================================
void GPUInterface::LaunchKernel(GPUFunction deviceFunction,
                                 Dim3Int block,
                                 Dim3Int grid,
                                 int parameterCountV,
                                 int totalParameterCount,
                                 ...) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::LaunchKernel (Metal)\n");
#endif
    id<MTLCommandQueue>          queue = MTL_QUEUE(metalCommandQueues[0]);
    id<MTLComputePipelineState>  pso   = MTL_PSO(deviceFunction);
    id<MTLCommandBuffer>         cmd   = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc   = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pso];

    va_list parameters;
    va_start(parameters, totalParameterCount);
    setKernelArgs(enc, parameterCountV, totalParameterCount, parameters, metalSubBufferMap);
    va_end(parameters);

    MTLSize tgsz    = MTLSizeMake(block.x, block.y, block.z);
    MTLSize grid_sz = MTLSizeMake((NSUInteger)block.x * grid.x,
                                  (NSUInteger)block.y * grid.y,
                                  (NSUInteger)block.z * grid.z);
    [enc dispatchThreads:grid_sz threadsPerThreadgroup:tgsz];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::LaunchKernel (Metal)\n");
#endif
}

// ============================================================
// LaunchKernelConcurrent — dispatches on an indexed command queue
// ============================================================
void GPUInterface::LaunchKernelConcurrent(GPUFunction deviceFunction,
                                           Dim3Int block,
                                           Dim3Int grid,
                                           int streamIndex,
                                           int waitIndex,
                                           int parameterCountV,
                                           int totalParameterCount,
                                           ...) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::LaunchKernelConcurrent (Metal)\n");
#endif
    int qIdx = (streamIndex >= 0) ? (streamIndex % BEAGLE_STREAM_COUNT) : 0;
    id<MTLCommandQueue>          queue = MTL_QUEUE(metalCommandQueues[qIdx]);
    id<MTLComputePipelineState>  pso   = MTL_PSO(deviceFunction);
    id<MTLCommandBuffer>         cmd   = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc   = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pso];

    va_list parameters;
    va_start(parameters, totalParameterCount);
    setKernelArgs(enc, parameterCountV, totalParameterCount, parameters, metalSubBufferMap);
    va_end(parameters);

    MTLSize tgsz    = MTLSizeMake(block.x, block.y, block.z);
    MTLSize grid_sz = MTLSizeMake((NSUInteger)block.x * grid.x,
                                  (NSUInteger)block.y * grid.y,
                                  (NSUInteger)block.z * grid.z);
    [enc dispatchThreads:grid_sz threadsPerThreadgroup:tgsz];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::LaunchKernelConcurrent (Metal)\n");
#endif
}

// ============================================================
// Host memory allocation
// ============================================================
void* GPUInterface::MallocHost(size_t memSize) {
#ifdef BEAGLE_MEMORY_PINNED
    return AllocatePinnedHostMemory(memSize, false, false);
#else
    return malloc(memSize);
#endif
}

void* GPUInterface::CallocHost(size_t size, size_t length) {
    size_t memSize = size * length;
#ifdef BEAGLE_MEMORY_PINNED
    void* ptr = AllocatePinnedHostMemory(memSize, false, false);
    memset(ptr, 0, memSize);
    return ptr;
#else
    return calloc(size, length);
#endif
}

// On Apple Silicon (unified memory architecture), pinned host memory is a
// MTLResourceStorageModeShared buffer whose contents pointer is directly
// accessible by both CPU and GPU.
void* GPUInterface::AllocatePinnedHostMemory(size_t memSize, bool writeCombined, bool mapped) {
    id<MTLDevice> device = MTL_DEVICE(metalDeviceId);
    id<MTLBuffer> buf    = [device newBufferWithLength:memSize
                                               options:MTLResourceStorageModeShared];
    if (!buf) {
        fprintf(stderr, "Metal error: AllocatePinnedHostMemory(%zu) failed\n", memSize);
        exit(-1);
    }
    // Return the raw CPU pointer; the MTLBuffer is NOT retained here because we
    // don't have a GPUPtr handle to track it.  For pinned-memory semantics on
    // unified-memory platforms, the raw pointer is sufficient.
    return buf.contents;
}

// MapMemory: on unified memory, the buffer contents are always CPU-accessible.
void* GPUInterface::MapMemory(GPUPtr dPtr, size_t memSize) {
    auto it = metalSubBufferMap.find(dPtr);
    if (it != metalSubBufferMap.end()) {
        id<MTLBuffer> buf = MTL_BUFFER(it->second.first);
        return (uint8_t*)buf.contents + it->second.second;
    }
    id<MTLBuffer> buf = MTL_BUFFER(dPtr);
    return buf.contents;
}

void GPUInterface::UnmapMemory(GPUPtr dPtr, void* hPtr) {
    // Unified memory: no unmap needed.
}

// ============================================================
// Device memory allocation
// ============================================================
GPUPtr GPUInterface::AllocateMemory(size_t memSize) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::AllocateMemory (Metal) size=%zu\n", memSize);
#endif
    id<MTLDevice> device = MTL_DEVICE(metalDeviceId);
    id<MTLBuffer> buf    = [device newBufferWithLength:memSize
                                               options:MTLResourceStorageModeShared];
    if (!buf) {
        fprintf(stderr, "Metal error: AllocateMemory(%zu) failed\n", memSize);
        exit(-1);
    }
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::AllocateMemory (Metal)\n");
#endif
    return RETAIN_VOIDPTR(buf);
}

GPUPtr GPUInterface::AllocateRealMemory(size_t length) {
    return AllocateMemory(SIZE_REAL * length);
}

GPUPtr GPUInterface::AllocateIntMemory(size_t length) {
    return AllocateMemory(SIZE_INT * length);
}

// CreateSubPointer: Metal has no sub-buffer primitive.
// Allocate a MetalSubBuffer sentinel (unique address) that records (parent, offset).
// LaunchKernel resolves these sentinels via metalSubBufferMap when setting buffers.
GPUPtr GPUInterface::CreateSubPointer(GPUPtr dPtr, size_t offset, size_t size) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::CreateSubPointer (Metal)\n");
#endif
    MetalSubBuffer* sd = new MetalSubBuffer{dPtr, offset, size};
    metalSubBufferMap[(void*)sd] = {dPtr, offset};
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::CreateSubPointer (Metal)\n");
#endif
    return (GPUPtr)sd;
}

// Metal requires buffer offsets to be 256-byte aligned for optimal performance;
// 16 bytes is the minimum for SIMD types.
size_t GPUInterface::AlignMemOffset(size_t offset) {
    const size_t align = 256;
    return ((offset + align - 1) / align) * align;
}

// ============================================================
// Memory transfers (unified memory → memcpy)
// ============================================================
void GPUInterface::MemcpyHostToDevice(GPUPtr dest, const void* src, size_t memSize) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::MemcpyHostToDevice (Metal)\n");
#endif
    void* dst;
    auto it = metalSubBufferMap.find(dest);
    if (it != metalSubBufferMap.end()) {
        id<MTLBuffer> buf = MTL_BUFFER(it->second.first);
        dst = (uint8_t*)buf.contents + it->second.second;
    } else {
        dst = MTL_BUFFER(dest).contents;
    }
    memcpy(dst, src, memSize);
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::MemcpyHostToDevice (Metal)\n");
#endif
}

void GPUInterface::MemcpyDeviceToHost(void* dest, const GPUPtr src, size_t memSize) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::MemcpyDeviceToHost (Metal)\n");
#endif
    const void* s;
    auto it = metalSubBufferMap.find((void*)src);
    if (it != metalSubBufferMap.end()) {
        id<MTLBuffer> buf = MTL_BUFFER(it->second.first);
        s = (const uint8_t*)buf.contents + it->second.second;
    } else {
        s = MTL_BUFFER((void*)src).contents;
    }
    memcpy(dest, s, memSize);
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::MemcpyDeviceToHost (Metal)\n");
#endif
}

void GPUInterface::MemcpyDeviceToDevice(GPUPtr dest, GPUPtr src, size_t memSize) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::MemcpyDeviceToDevice (Metal)\n");
#endif
    id<MTLCommandQueue>       queue = MTL_QUEUE(metalCommandQueues[0]);
    id<MTLCommandBuffer>      cmd   = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit  = [cmd blitCommandEncoder];

    id<MTLBuffer> srcBuf; size_t srcOff = 0;
    id<MTLBuffer> dstBuf; size_t dstOff = 0;

    auto srcIt = metalSubBufferMap.find((void*)src);
    if (srcIt != metalSubBufferMap.end()) {
        srcBuf = MTL_BUFFER(srcIt->second.first);
        srcOff = srcIt->second.second;
    } else {
        srcBuf = MTL_BUFFER((void*)src);
    }

    auto dstIt = metalSubBufferMap.find(dest);
    if (dstIt != metalSubBufferMap.end()) {
        dstBuf = MTL_BUFFER(dstIt->second.first);
        dstOff = dstIt->second.second;
    } else {
        dstBuf = MTL_BUFFER(dest);
    }

    [blit copyFromBuffer:srcBuf sourceOffset:srcOff
                toBuffer:dstBuf destinationOffset:dstOff
                    size:memSize];
    [blit endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::MemcpyDeviceToDevice (Metal)\n");
#endif
}

void GPUInterface::MemsetShort(GPUPtr dest, unsigned short val, size_t count) {
    // Metal's fillBuffer only supports single-byte fill patterns.
    // For 16-bit fill, fall back to CPU-side write on unified memory.
    void* ptr;
    auto it = metalSubBufferMap.find(dest);
    if (it != metalSubBufferMap.end()) {
        id<MTLBuffer> buf = MTL_BUFFER(it->second.first);
        ptr = (uint8_t*)buf.contents + it->second.second;
    } else {
        ptr = MTL_BUFFER(dest).contents;
    }
    uint16_t* p = static_cast<uint16_t*>(ptr);
    for (size_t i = 0; i < count; i++) p[i] = val;
}

// ============================================================
// Free memory
// ============================================================
void GPUInterface::FreeHostMemory(void* hPtr) {
#ifdef BEAGLE_MEMORY_PINNED
    FreePinnedHostMemory(hPtr);
#else
    free(hPtr);
#endif
}

void GPUInterface::FreePinnedHostMemory(void* hPtr) {
    // Pinned allocations on unified memory are MTLBuffers.  We returned .contents
    // in AllocatePinnedHostMemory, so we cannot safely recover the MTLBuffer here.
    // The MTLBuffer is released when no further references exist via ARC.
}

void GPUInterface::FreeMemory(GPUPtr dPtr) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::FreeMemory (Metal)\n");
#endif
    auto it = metalSubBufferMap.find(dPtr);
    if (it != metalSubBufferMap.end()) {
        // Free the sentinel struct and remove the map entry.
        delete static_cast<MetalSubBuffer*>(dPtr);
        metalSubBufferMap.erase(it);
    } else {
        // Release the MTLBuffer (ARC decrements the retain count).
        RELEASE_VOIDPTR(dPtr, MTLBuffer);
    }
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::FreeMemory (Metal)\n");
#endif
}

GPUPtr GPUInterface::GetDeviceHostPointer(void* hPtr) {
    // Not meaningful on unified memory; return NULL.
    return NULL;
}

size_t GPUInterface::GetAvailableMemory() {
    // Metal does not expose a direct available-memory query.
    return 0;
}

// ============================================================
// Device info
// ============================================================
void GPUInterface::GetDeviceName(int deviceNumber, char* deviceName, int nameLength) {
    id<MTLDevice> d = MTL_DEVICE(metalDeviceMap[deviceNumber]);
    const char* name = [[d name] UTF8String];
    strncpy(deviceName, name, nameLength - 1);
    deviceName[nameLength - 1] = '\0';

    // Append Metal version string for informational parity with OpenCL.
    const char* suffix = " (Metal)";
    size_t remaining = nameLength - strlen(deviceName) - 1;
    if (remaining > strlen(suffix))
        strcat(deviceName, suffix);
}

bool GPUInterface::GetSupportsDoublePrecision(int deviceNumber) {
    return false; // Apple Silicon Metal does not support double precision.
}

void GPUInterface::GetDeviceDescription(int deviceNumber, char* deviceDescription) {
    id<MTLDevice> d = MTL_DEVICE(metalDeviceMap[deviceNumber]);
    unsigned long long mem = [d recommendedMaxWorkingSetSize];
    snprintf(deviceDescription, 256,
             "Global memory (MB): %llu | Metal GPU | Unified memory: %s",
             mem / 1024 / 1024,
             [d hasUnifiedMemory] ? "yes" : "no");
}

long GPUInterface::GetDeviceTypeFlag(int deviceNumber) {
    return BEAGLE_FLAG_PROCESSOR_GPU;
}

BeagleDeviceImplementationCodes GPUInterface::GetDeviceImplementationCode(int deviceNumber) {
    return BEAGLE_METAL_DEVICE_APPLE_GPU;
}

void GPUInterface::PrintfDeviceInt(GPUPtr dPtr, int length) {
    int* hPtr = (int*)malloc(SIZE_INT * length);
    MemcpyDeviceToHost(hPtr, dPtr, SIZE_INT * length);
    printfInt(hPtr, length);
    free(hPtr);
}

}; // namespace metal_device
