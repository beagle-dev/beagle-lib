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
 * ROCm/HIP backend for AMD GPUs.
 * Kernel source strings are compiled at runtime via HIPRTC (mirroring the
 * OpenCL backend), then loaded as HIP modules (mirroring the CUDA backend).
 */

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cstdarg>
#include <map>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hiprtc.h>

#include "libhmsbeagle/beagle.h"
#include "libhmsbeagle/GPU/GPUImplDefs.h"
#include "libhmsbeagle/GPU/GPUImplHelper.h"
#include "libhmsbeagle/GPU/GPUInterface.h"
#include "libhmsbeagle/GPU/KernelResource.h"

#define SAFE_HIP(call) { \
    hipError_t error = call; \
    if (error != hipSuccess) { \
        fprintf(stderr, "ROCm error: \"%s\" (%d) from file <%s>, line %i.\n", \
                hipGetErrorString(error), error, __FILE__, __LINE__); \
        exit(-1); \
    } \
}

#define SAFE_HIPP(call) { \
    SAFE_HIP(hipCtxPushCurrent(hipContext)); \
    SAFE_HIP(call); \
    SAFE_HIP(hipCtxPopCurrent(&hipContext)); \
}

#define SAFE_HIPRTC(call) { \
    hiprtcResult result = call; \
    if (result != HIPRTC_SUCCESS) { \
        fprintf(stderr, "HIPRTC error: \"%s\" (%d) from file <%s>, line %i.\n", \
                hiprtcGetErrorString(result), result, __FILE__, __LINE__); \
        exit(-1); \
    } \
}

#define LOAD_KERNEL_INTO_RESOURCE(state, prec, id) \
        kernelResource = new KernelResource( \
            state, \
            (char*) KERNELS_STRING_##prec##_##state, \
            PATTERN_BLOCK_SIZE_##prec##_##state, \
            MATRIX_BLOCK_SIZE_##prec##_##state, \
            BLOCK_PEELING_SIZE_##prec##_##state, \
            SLOW_REWEIGHING_##prec##_##state, \
            MULTIPLY_BLOCK_SIZE_##prec, \
            BASTA_SUM_INTERVAL_BLOCK_SIZE_##prec##_##state, \
            BASTA_SUM_ACROSS_BLOCK_SIZE_##prec##_##state, \
            BLOCK_PEELING_SIZE_SCA_##prec##_##state, \
            0,0,0,0);

namespace rocm_device {

#define SAFE_CUDA(call) SAFE_HIP(call)
#define SAFE_CUPP(call) SAFE_HIPP(call)

GPUInterface::GPUInterface() {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::GPUInterface\n");
#endif

    hipDevice   = (hipDevice_t) 0;
    hipContext  = NULL;
    hipModule   = NULL;
    hipStreams   = NULL;
    hipEvent    = NULL;
    kernelResource = NULL;
    supportDoublePrecision = true;

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::GPUInterface\n");
#endif
}

GPUInterface::~GPUInterface() {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::~GPUInterface\n");
#endif

    if (hipStreams != NULL) {
        for (int i = 0; i < numStreams; i++) {
            if (hipStreams[i] != NULL)
                SAFE_HIP(hipStreamDestroy(hipStreams[i]));
        }
        free(hipStreams);
    }

    if (hipEvent != NULL)
        SAFE_HIP(hipEventDestroy(hipEvent));

    if (hipModule != NULL)
        SAFE_HIP(hipModuleUnload(hipModule));

    if (hipContext != NULL) {
        SAFE_HIP(hipCtxPushCurrent(hipContext));
        SAFE_HIP(hipDevicePrimaryCtxRelease(hipDevice));
    }

    if (kernelResource != NULL)
        delete kernelResource;

    if (resourceMap)
        delete resourceMap;

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::~GPUInterface\n");
#endif
}

int GPUInterface::Initialize() {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::Initialize\n");
#endif

    resourceMap = new std::map<int, int>;

    hipError_t error = hipInit(0);
    if (error != hipSuccess)
        return 0;

    int numDevices = 0;
    SAFE_HIP(hipGetDeviceCount(&numDevices));

    int currentDevice = 0;
    for (int i = 0; i < numDevices; i++) {
        hipDevice_t tmpDevice;
        SAFE_HIP(hipDeviceGet(&tmpDevice, i));
        // Accept all ROCm-capable devices
        resourceMap->insert(std::make_pair(currentDevice++, i));
    }

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::Initialize\n");
#endif

    return 1;
}

int GPUInterface::GetDeviceCount() {
    return resourceMap->size();
}

void GPUInterface::InitializeKernelResource(int paddedStateCount, bool doublePrecision) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLoading kernel information for ROCm!\n");
#endif

    if (doublePrecision)
        paddedStateCount *= -1;

    switch (paddedStateCount) {
        case   -4: LOAD_KERNEL_INTO_RESOURCE(  4, DP,   4); break;
        case  -16: LOAD_KERNEL_INTO_RESOURCE( 16, DP,  16); break;
        case  -32: LOAD_KERNEL_INTO_RESOURCE( 32, DP,  32); break;
        case  -48: LOAD_KERNEL_INTO_RESOURCE( 48, DP,  48); break;
        case  -64: LOAD_KERNEL_INTO_RESOURCE( 64, DP,  64); break;
        case  -80: LOAD_KERNEL_INTO_RESOURCE( 80, DP,  80); break;
        case -128: LOAD_KERNEL_INTO_RESOURCE(128, DP, 128); break;
        case -192: LOAD_KERNEL_INTO_RESOURCE(192, DP, 192); break;
        case -256: LOAD_KERNEL_INTO_RESOURCE(256, DP, 256); break;
        case    4: LOAD_KERNEL_INTO_RESOURCE(  4, SP,   4); break;
        case   16: LOAD_KERNEL_INTO_RESOURCE( 16, SP,  16); break;
        case   32: LOAD_KERNEL_INTO_RESOURCE( 32, SP,  32); break;
        case   48: LOAD_KERNEL_INTO_RESOURCE( 48, SP,  48); break;
        case   64: LOAD_KERNEL_INTO_RESOURCE( 64, SP,  64); break;
        case   80: LOAD_KERNEL_INTO_RESOURCE( 80, SP,  80); break;
        case  128: LOAD_KERNEL_INTO_RESOURCE(128, SP, 128); break;
        case  192: LOAD_KERNEL_INTO_RESOURCE(192, SP, 192); break;
        case  256: LOAD_KERNEL_INTO_RESOURCE(256, SP, 256); break;
    }
}

void GPUInterface::SetDevice(int deviceNumber,
                             int paddedStateCount,
                             int categoryCount,
                             int paddedPatternCount,
                             int unpaddedPatternCount,
                             int tipCount,
                             long flags) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::SetDevice\n");
#endif

    SAFE_HIP(hipDeviceGet(&hipDevice, (*resourceMap)[deviceNumber]));
    SAFE_HIP(hipDevicePrimaryCtxRetain(&hipContext, hipDevice));
    SAFE_HIP(hipCtxSetCurrent(hipContext));

    InitializeKernelResource(paddedStateCount, flags & BEAGLE_FLAG_PRECISION_DOUBLE);

    if (!kernelResource) {
        fprintf(stderr, "Critical error: unable to find kernel code for %d states.\n", paddedStateCount);
        exit(-1);
    }
    kernelResource->categoryCount        = categoryCount;
    kernelResource->patternCount         = paddedPatternCount;
    kernelResource->unpaddedPatternCount = unpaddedPatternCount;
    kernelResource->flags                = flags;

    // Compile kernel source at runtime via HIPRTC
    hiprtcProgram prog;
    SAFE_HIPRTC(hiprtcCreateProgram(&prog,
                                    kernelResource->kernelCode,
                                    "beagle_kernels.hip",
                                    0, NULL, NULL));

    const char* opts[] = { "-DFW_ROCM", "-DOPENCL_KERNEL_BUILD" };
    hiprtcResult compileResult = hiprtcCompileProgram(prog, 2, opts);
    if (compileResult != HIPRTC_SUCCESS) {
        size_t logSize;
        hiprtcGetProgramLogSize(prog, &logSize);
        std::vector<char> log(logSize);
        hiprtcGetProgramLog(prog, log.data());
        fprintf(stderr, "ROCm error: Failed to build kernels\n%s\n", log.data());
        exit(-1);
    }

    size_t codeSize;
    SAFE_HIPRTC(hiprtcGetCodeSize(prog, &codeSize));
    std::vector<char> code(codeSize);
    SAFE_HIPRTC(hiprtcGetCode(prog, code.data()));
    SAFE_HIPRTC(hiprtcDestroyProgram(&prog));

    SAFE_HIP(hipModuleLoadData(&hipModule, code.data()));

    // Create streams
    numStreams  = 1;
    hipStreams  = (hipStream_t*) malloc(sizeof(hipStream_t) * numStreams);
    SAFE_HIP(hipStreamCreate(&hipStreams[0]));

    SAFE_HIP(hipEventCreateWithFlags(&hipEvent, hipEventDisableTiming));

    SAFE_HIP(hipCtxPopCurrent(&hipContext));

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::SetDevice\n");
#endif
}

void GPUInterface::ResizeStreamCount(int newStreamCount) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::ResizeStreamCount\n");
#endif
    SAFE_HIP(hipCtxPushCurrent(hipContext));
    SAFE_HIP(hipCtxSynchronize());

    if (hipStreams != NULL) {
        for (int i = 0; i < numStreams; i++) {
            if (hipStreams[i] != NULL)
                SAFE_HIP(hipStreamDestroy(hipStreams[i]));
        }
        free(hipStreams);
    }

    numStreams = (newStreamCount > BEAGLE_STREAM_COUNT) ? BEAGLE_STREAM_COUNT : newStreamCount;
    if (numStreams < 1) numStreams = 1;
    hipStreams = (hipStream_t*) malloc(sizeof(hipStream_t) * numStreams);
    for (int i = 0; i < numStreams; i++)
        SAFE_HIP(hipStreamCreate(&hipStreams[i]));

    SAFE_HIP(hipCtxPopCurrent(&hipContext));
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::ResizeStreamCount\n");
#endif
}

void GPUInterface::SynchronizeHost() {
    SAFE_HIPP(hipCtxSynchronize());
}

void GPUInterface::SynchronizeDevice() {
    SAFE_HIP(hipCtxPushCurrent(hipContext));
    SAFE_HIP(hipEventRecord(hipEvent, 0));
    SAFE_HIP(hipStreamWaitEvent(hipStreams[0], hipEvent, 0));
    SAFE_HIP(hipCtxPopCurrent(&hipContext));
}

void GPUInterface::SynchronizeDeviceWithIndex(int streamRecordIndex, int streamWaitIndex) {
    hipStream_t streamRecord = (streamRecordIndex >= 0) ? hipStreams[streamRecordIndex % numStreams] : NULL;
    hipStream_t streamWait   = (streamWaitIndex   >= 0) ? hipStreams[streamWaitIndex   % numStreams] : NULL;
    SAFE_HIPP(hipEventRecord(hipEvent, streamRecord));
    SAFE_HIPP(hipStreamWaitEvent(streamWait, hipEvent, 0));
}

GPUFunction GPUInterface::GetFunction(const char* functionName) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::GetFunction: %s\n", functionName);
#endif

    GPUFunction fn;
    SAFE_HIPP(hipModuleGetFunction(&fn, hipModule, functionName));

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::GetFunction\n");
#endif
    return fn;
}

void GPUInterface::LaunchKernel(GPUFunction deviceFunction,
                                Dim3Int block, Dim3Int grid,
                                int parameterCountV, int totalParameterCount,
                                ...) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::LaunchKernel\n");
#endif

    SAFE_HIP(hipCtxPushCurrent(hipContext));

    void**         params    = (void**)         malloc(sizeof(void*)         * totalParameterCount);
    GPUPtr*        paramPtrs = (GPUPtr*)         malloc(sizeof(GPUPtr)        * totalParameterCount);
    unsigned int*  paramInts = (unsigned int*)   malloc(sizeof(unsigned int)  * totalParameterCount);

    va_list parameters;
    va_start(parameters, totalParameterCount);
    for (int i = 0; i < parameterCountV; i++) {
        paramPtrs[i] = (GPUPtr)(size_t) va_arg(parameters, GPUPtr);
        params[i]    = (void*) &paramPtrs[i];
    }
    for (int i = parameterCountV; i < totalParameterCount; i++) {
        paramInts[i - parameterCountV] = va_arg(parameters, unsigned int);
        params[i] = (void*) &paramInts[i - parameterCountV];
    }
    va_end(parameters);

    SAFE_HIP(hipModuleLaunchKernel(deviceFunction,
                                   grid.x, grid.y, grid.z,
                                   block.x, block.y, block.z,
                                   0, hipStreams[0], params, NULL));

    free(params); free(paramPtrs); free(paramInts);

    SAFE_HIP(hipCtxPopCurrent(&hipContext));
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::LaunchKernel\n");
#endif
}

void GPUInterface::LaunchKernelConcurrent(GPUFunction deviceFunction,
                                          Dim3Int block, Dim3Int grid,
                                          int streamIndex, int waitIndex,
                                          int parameterCountV, int totalParameterCount,
                                          ...) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::LaunchKernelConcurrent\n");
#endif

    SAFE_HIP(hipCtxPushCurrent(hipContext));

    void**         params    = (void**)         malloc(sizeof(void*)         * totalParameterCount);
    GPUPtr*        paramPtrs = (GPUPtr*)         malloc(sizeof(GPUPtr)        * totalParameterCount);
    unsigned int*  paramInts = (unsigned int*)   malloc(sizeof(unsigned int)  * totalParameterCount);

    va_list parameters;
    va_start(parameters, totalParameterCount);
    for (int i = 0; i < parameterCountV; i++) {
        paramPtrs[i] = (GPUPtr)(size_t) va_arg(parameters, GPUPtr);
        params[i]    = (void*) &paramPtrs[i];
    }
    for (int i = parameterCountV; i < totalParameterCount; i++) {
        paramInts[i - parameterCountV] = va_arg(parameters, unsigned int);
        params[i] = (void*) &paramInts[i - parameterCountV];
    }
    va_end(parameters);

    hipStream_t stream = hipStreams[0];
    if (streamIndex >= 0) {
        int mod = streamIndex % numStreams;
        if (waitIndex >= 0)
            SAFE_HIP(hipStreamSynchronize(hipStreams[waitIndex % numStreams]));
        stream = hipStreams[mod];
    }

    SAFE_HIP(hipModuleLaunchKernel(deviceFunction,
                                   grid.x, grid.y, grid.z,
                                   block.x, block.y, block.z,
                                   0, stream, params, NULL));

    free(params); free(paramPtrs); free(paramInts);

    SAFE_HIP(hipCtxPopCurrent(&hipContext));
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::LaunchKernelConcurrent\n");
#endif
}

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

void* GPUInterface::AllocatePinnedHostMemory(size_t memSize, bool writeCombined, bool mapped) {
    void* ptr;
    unsigned int flags = 0;
    if (writeCombined) flags |= hipHostMallocWriteCombined;
    if (mapped)        flags |= hipHostMallocMapped;
    SAFE_HIPP(hipHostMalloc(&ptr, memSize, flags));
    return ptr;
}

GPUPtr GPUInterface::AllocateMemory(size_t memSize) {
    GPUPtr ptr;
    SAFE_HIPP(hipMemAlloc(&ptr, memSize));
#ifdef BEAGLE_DEBUG_VALUES
    fprintf(stderr, "Allocated GPU memory %llu to %llu.\n",
            (unsigned long long)ptr, (unsigned long long)(ptr + memSize));
#endif
    return ptr;
}

GPUPtr GPUInterface::AllocateRealMemory(size_t length) {
    GPUPtr ptr;
    SAFE_HIPP(hipMemAlloc(&ptr, SIZE_REAL * length));
    return ptr;
}

GPUPtr GPUInterface::AllocateIntMemory(size_t length) {
    GPUPtr ptr;
    SAFE_HIPP(hipMemAlloc(&ptr, SIZE_INT * length));
    return ptr;
}

GPUPtr GPUInterface::CreateSubPointer(GPUPtr dPtr, size_t offset, size_t size) {
    return dPtr + offset;
}

size_t GPUInterface::AlignMemOffset(size_t offset) {
    return offset;
}

void GPUInterface::MemsetShort(GPUPtr dest, unsigned short val, size_t count) {
    SAFE_HIPP(hipMemsetD16(dest, val, count));
}

void GPUInterface::MemcpyHostToDevice(GPUPtr dest, const void* src, size_t memSize) {
    SAFE_HIPP(hipMemcpyHtoDAsync(dest, const_cast<void*>(src), memSize, hipStreams[0]));
}

void GPUInterface::MemcpyDeviceToHost(void* dest, const GPUPtr src, size_t memSize) {
    SAFE_HIPP(hipMemcpyDtoHAsync(dest, src, memSize, hipStreams[0]));
}

void GPUInterface::MemcpyDeviceToDevice(GPUPtr dest, GPUPtr src, size_t memSize) {
    SAFE_HIPP(hipMemcpyDtoDAsync(dest, src, memSize, hipStreams[0]));
}

void GPUInterface::FreeHostMemory(void* hPtr) {
#ifdef BEAGLE_MEMORY_PINNED
    FreePinnedHostMemory(hPtr);
#else
    free(hPtr);
#endif
}

void GPUInterface::FreePinnedHostMemory(void* hPtr) {
    SAFE_HIPP(hipHostFree(hPtr));
}

void GPUInterface::FreeMemory(GPUPtr dPtr) {
    SAFE_HIPP(hipMemFree(dPtr));
}

GPUPtr GPUInterface::GetDeviceHostPointer(void* hPtr) {
    GPUPtr dPtr;
    SAFE_HIPP(hipHostGetDevicePointer((void**)&dPtr, hPtr, 0));
    return dPtr;
}

size_t GPUInterface::GetAvailableMemory() {
    size_t available = 0, total = 0;
    SAFE_HIPP(hipMemGetInfo(&available, &total));
    return available;
}

void GPUInterface::GetDeviceName(int deviceNumber, char* deviceName, int nameLength) {
    hipDevice_t tmpDevice;
    SAFE_HIP(hipDeviceGet(&tmpDevice, (*resourceMap)[deviceNumber]));
    SAFE_HIP(hipDeviceGetName(deviceName, nameLength, tmpDevice));
}

bool GPUInterface::GetSupportsDoublePrecision(int deviceNumber) {
    // All ROCm-supported AMD GPUs support double precision
    return true;
}

void GPUInterface::GetDeviceDescription(int deviceNumber, char* deviceDescription) {
    hipDevice_t tmpDevice;
    SAFE_HIP(hipDeviceGet(&tmpDevice, (*resourceMap)[deviceNumber]));

    size_t totalGlobalMemory = 0;
    int clockSpeed = 0, cuCount = 0;

    SAFE_HIP(hipDeviceTotalMem(&totalGlobalMemory, tmpDevice));
    SAFE_HIP(hipDeviceGetAttribute(&clockSpeed, hipDeviceAttributeClockRate,          tmpDevice));
    SAFE_HIP(hipDeviceGetAttribute(&cuCount,    hipDeviceAttributeMultiprocessorCount, tmpDevice));

    sprintf(deviceDescription,
            "Global memory (MB): %d | Clock speed (Ghz): %1.2f | Compute units: %d",
            (int)(totalGlobalMemory / 1024.0 / 1024.0 + 0.5),
            clockSpeed / 1000000.0,
            cuCount);
}

void GPUInterface::PrintfDeviceInt(GPUPtr dPtr, int length) {
    int* hPtr = (int*) malloc(SIZE_INT * length);
    MemcpyDeviceToHost(hPtr, dPtr, SIZE_INT * length);
    printfInt(hPtr, length);
    free(hPtr);
}

long GPUInterface::GetDeviceTypeFlag(int deviceNumber) {
    return BEAGLE_FLAG_PROCESSOR_GPU;
}

BeagleDeviceImplementationCodes GPUInterface::GetDeviceImplementationCode(int deviceNumber) {
    return BEAGLE_ROCM_DEVICE_AMD_GPU;
}

const char* GPUInterface::GetHIPErrorDescription(int errorCode) {
    return hipGetErrorString((hipError_t) errorCode);
}

}; // namespace rocm_device
