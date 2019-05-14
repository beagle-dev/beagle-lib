/*
 *
 * Copyright 2009 Phylogenetic Likelihood Working Group
 *
 * This file is part of BEAGLE.
 *
 * BEAGLE is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 *
 * BEAGLE is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with BEAGLE.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @author Marc Suchard
 * @author Daniel Ayres
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

#include <cuda.h>

#include "libhmsbeagle/beagle.h"
#include "libhmsbeagle/GPU/GPUImplDefs.h"
#include "libhmsbeagle/GPU/GPUImplHelper.h"
#include "libhmsbeagle/GPU/GPUInterface.h"
#include "libhmsbeagle/GPU/KernelResource.h"

#include <cmath>

#define LOAD_KERNEL_INTO_RESOURCE(state, prec, id) \
        kernelResource = new KernelResource( \
            state, \
            (char*) KERNELS_STRING_##prec##_##state, \
            PATTERN_BLOCK_SIZE_##prec##_##state, \
            MATRIX_BLOCK_SIZE_##prec##_##state, \
            BLOCK_PEELING_SIZE_##prec##_##state, \
            SLOW_REWEIGHING_##prec##_##state, \
            MULTIPLY_BLOCK_SIZE_##prec, \
            0,0,0,0);

namespace cuda_device {

//static int nGpuArchCoresPerSM[] = { -1, 8, 32 };

namespace util {

inline int ConvertSMVer2CoresDRV(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct
    {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] =
    {
        { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
        { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
        { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
        { 0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
        { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
        { 0x37, 192}, // Kepler Generation (SM 3.7) GK21x class
        { 0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
        { 0x52, 128}, // Maxwell Generation (SM 5.2) GM20x class
        { 0x53, 128}, // Maxwell Generation (SM 5.3) GM20x class
        { 0x60, 64 }, // Pascal Generation (SM 6.0) GP100 class
        { 0x61, 128}, // Pascal Generation (SM 6.1) GP10x class
        { 0x62, 128}, // Pascal Generation (SM 6.2) GP10x class
        {   -1, -1 }
    };

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1)
    {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
        {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one to run properly
    return nGpuArchCoresPerSM[index-1].Cores;
}

}


#define SAFE_CUDA(call) { \
                            CUresult error = call; \
                            if(error != CUDA_SUCCESS) { \
                                fprintf(stderr, "CUDA error: \"%s\" (%d) from file <%s>, line %i.\n", \
                                        GetCUDAErrorDescription(error), error, __FILE__, __LINE__); \
                                exit(-1); \
                            } \
                        }

#define SAFE_CUPP(call) { \
                            SAFE_CUDA(cuCtxPushCurrent(cudaContext)); \
                            SAFE_CUDA(call); \
                            SAFE_CUDA(cuCtxPopCurrent(&cudaContext)); \
                        }

GPUInterface::GPUInterface() {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::GPUInterface\n");
#endif

    cudaDevice = (CUdevice) 0;
    cudaContext = NULL;
    cudaModule = NULL;
    cudaStreams = NULL;
    cudaEvents = NULL;
    kernelResource = NULL;
    supportDoublePrecision = true;

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::GPUInterface\n");
#endif
}

GPUInterface::~GPUInterface() {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::~GPUInterface\n");
#endif

    if (cudaStreams != NULL) {
        for(int i=0; i<numStreams; i++) {
            if (cudaStreams[i] != NULL)
                SAFE_CUDA(cuStreamDestroy(cudaStreams[i]));
        }
        free(cudaStreams);
    }

    if (cudaEvents != NULL) {
        for(int i=0; i<numStreams; i++) {
            if (cudaEvents[i] != NULL)
                SAFE_CUDA(cuEventDestroy(cudaEvents[i]));
        }
        free(cudaEvents);
    }

    if (cudaContext != NULL) {
        SAFE_CUDA(cuCtxPushCurrent(cudaContext));
        SAFE_CUDA(cuDevicePrimaryCtxRelease(cudaDevice));
    }

    if (kernelResource != NULL) {
        delete kernelResource;
    }

    if (resourceMap) {
        delete resourceMap;
    }

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::~GPUInterface\n");
#endif

}

int GPUInterface::Initialize() {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::Initialize\n");
#endif

    resourceMap = new std::map<int, int>;

    // Driver init; CUDA manual: "Currently, the Flags parameter must be 0."
    CUresult error = cuInit(0);

    if (error != CUDA_SUCCESS) {
        return 0;
    }

    int numDevices = 0;
    SAFE_CUDA(cuDeviceGetCount(&numDevices));

    CUdevice tmpCudaDevice;
    int capabilityMajor;
    int capabilityMinor;
    int currentDevice = 0;
    for (int i=0; i < numDevices; i++) {
        SAFE_CUDA(cuDeviceGet(&tmpCudaDevice, i));
        SAFE_CUDA(cuDeviceGetAttribute(&capabilityMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, tmpCudaDevice));
        SAFE_CUDA(cuDeviceGetAttribute(&capabilityMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, tmpCudaDevice));
        if (capabilityMajor >= 3 && capabilityMinor != 9999) {
            resourceMap->insert(std::make_pair(currentDevice++, i));
        }
    }

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::Initialize\n");
#endif

    return 1;
}

int GPUInterface::GetDeviceCount() {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::GetDeviceCount\n");
#endif

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::GetDeviceCount\n");
#endif

    return resourceMap->size();
}

void GPUInterface::InitializeKernelResource(int paddedStateCount,
                                            bool doublePrecision) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLoading kernel information for CUDA!\n");
#endif

    if (doublePrecision)
        paddedStateCount *= -1;

    switch(paddedStateCount) {
        case   -4: LOAD_KERNEL_INTO_RESOURCE(  4, DP,   4); break;
        case  -16: LOAD_KERNEL_INTO_RESOURCE( 16, DP,  16); break;
        case  -32: LOAD_KERNEL_INTO_RESOURCE( 32, DP,  32); break;
        case  -48: LOAD_KERNEL_INTO_RESOURCE( 48, DP,  48); break;
        case  -64: LOAD_KERNEL_INTO_RESOURCE( 64, DP,  64); break;
        case  -80: LOAD_KERNEL_INTO_RESOURCE( 80, DP,  80); break;
        case -128: LOAD_KERNEL_INTO_RESOURCE(128, DP, 128); break;
        case -192: LOAD_KERNEL_INTO_RESOURCE(192, DP, 192); break;
        case    4: LOAD_KERNEL_INTO_RESOURCE(  4, SP,   4); break;
        case   16: LOAD_KERNEL_INTO_RESOURCE( 16, SP,  16); break;
        case   32: LOAD_KERNEL_INTO_RESOURCE( 32, SP,  32); break;
        case   48: LOAD_KERNEL_INTO_RESOURCE( 48, SP,  48); break;
        case   64: LOAD_KERNEL_INTO_RESOURCE( 64, SP,  64); break;
        case   80: LOAD_KERNEL_INTO_RESOURCE( 80, SP,  80); break;
        case  128: LOAD_KERNEL_INTO_RESOURCE(128, SP, 128); break;
        case  192: LOAD_KERNEL_INTO_RESOURCE(192, SP, 192); break;
    }
}

void GPUInterface::SetDevice(int deviceNumber, int paddedStateCount, int categoryCount, int paddedPatternCount, int unpaddedPatternCount, int tipCount,
                             long flags) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::SetDevice\n");
#endif

    SAFE_CUDA(cuDeviceGet(&cudaDevice, (*resourceMap)[deviceNumber]));

    // unsigned int ctxFlags = CU_CTX_SCHED_AUTO;

    // if (flags & BEAGLE_FLAG_SCALING_DYNAMIC) {
    //     ctxFlags |= CU_CTX_MAP_HOST;
    // }

    CUresult error = cuDevicePrimaryCtxRetain(&cudaContext, cudaDevice);
    if(error != CUDA_SUCCESS) {
        fprintf(stderr, "CUDA error: \"%s\" (%d) from file <%s>, line %i.\n",
                GetCUDAErrorDescription(error), error, __FILE__, __LINE__);
        if (error == CUDA_ERROR_INVALID_DEVICE) {
            fprintf(stderr, "(The requested CUDA device is likely set to compute exclusive mode. This mode prevents multiple processes from running on the device.)");
        }
        exit(-1);
    }

    SAFE_CUDA(cuCtxSetCurrent(cudaContext));

    InitializeKernelResource(paddedStateCount, flags & BEAGLE_FLAG_PRECISION_DOUBLE);

    if (!kernelResource) {
        fprintf(stderr,"Critical error: unable to find kernel code for %d states.\n",paddedStateCount);
        exit(-1);
    }
    kernelResource->categoryCount = categoryCount;
    kernelResource->patternCount = paddedPatternCount;
    kernelResource->unpaddedPatternCount = unpaddedPatternCount;
    kernelResource->flags = flags;

    SAFE_CUDA(cuModuleLoadData(&cudaModule, kernelResource->kernelCode));

    numStreams = 1;
    cudaStreams = (CUstream*) malloc(sizeof(CUstream) * numStreams);
    CUstream stream;
    SAFE_CUDA(cuStreamCreate(&stream, CU_STREAM_DEFAULT));
    cudaStreams[0] = stream;
    CUevent event;
    cudaEvents = (CUevent*) malloc(sizeof(CUevent) * (numStreams + 1));
    for(int i=0; i<2; i++) {
        SAFE_CUDA(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING));
        cudaEvents[i] = event;
    }

    SAFE_CUDA(cuCtxPopCurrent(&cudaContext));

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::SetDevice\n");
#endif

}

void GPUInterface::ResizeStreamCount(int newStreamCount) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::ResizeStreamCount\n");
#endif
    SAFE_CUDA(cuCtxPushCurrent(cudaContext));

    SAFE_CUDA(cuCtxSynchronize());

    if (cudaStreams != NULL) {
        for(int i=0; i<numStreams; i++) {
            if (cudaStreams[i] != NULL)
                SAFE_CUDA(cuStreamDestroy(cudaStreams[i]));
        }
        free(cudaStreams);
    }

    if (cudaEvents != NULL) {
        for(int i=0; i<numStreams; i++) {
            if (cudaEvents[i] != NULL)
                SAFE_CUDA(cuEventDestroy(cudaEvents[i]));
        }
        free(cudaEvents);
    }

    if (newStreamCount == 1) {
        numStreams = 1;
        cudaStreams = (CUstream*) malloc(sizeof(CUstream) * numStreams);
        CUstream stream;
        SAFE_CUDA(cuStreamCreate(&stream, CU_STREAM_DEFAULT));
        cudaStreams[0] = stream;
        CUevent event;
        for(int i=0; i<2; i++) {
            SAFE_CUDA(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING));
            cudaEvents[i] = event;
        }
    } else {
        numStreams = newStreamCount;
        if (numStreams > BEAGLE_STREAM_COUNT) {
            numStreams = BEAGLE_STREAM_COUNT;
        }
        cudaStreams = (CUstream*) malloc(sizeof(CUstream) * numStreams);
        CUstream stream;
        cudaEvents = (CUevent*) malloc(sizeof(CUevent) * (numStreams + 1));
        CUevent event;
        for(int i=0; i<numStreams; i++) {
            SAFE_CUDA(cuStreamCreate(&stream, CU_STREAM_DEFAULT));
            cudaStreams[i] = stream;
            SAFE_CUDA(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING));
            cudaEvents[i] = event;
        }
        SAFE_CUDA(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING));
        cudaEvents[numStreams] = event;
    }

    SAFE_CUDA(cuCtxPopCurrent(&cudaContext));

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::ResizeStreamCount\n");
#endif
}

void GPUInterface::SynchronizeHost() {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::SynchronizeHost\n");
#endif

    SAFE_CUPP(cuCtxSynchronize());

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::SynchronizeHost\n");
#endif
}

void GPUInterface::SynchronizeDevice() {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::SynchronizeDevice\n");
#endif

    SAFE_CUDA(cuCtxPushCurrent(cudaContext));

    SAFE_CUDA(cuEventRecord(cudaEvents[numStreams], 0));
    SAFE_CUDA(cuStreamWaitEvent(0, cudaEvents[numStreams], 0));

    SAFE_CUDA(cuCtxPopCurrent(&cudaContext));

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::SynchronizeDevice\n");
#endif
}

void GPUInterface::SynchronizeDeviceWithIndex(int streamRecordIndex, int streamWaitIndex) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::SynchronizeDeviceWithIndex\n");
#endif

    SAFE_CUDA(cuCtxPushCurrent(cudaContext));

    CUstream streamRecord  = NULL;
    CUstream streamWait    = NULL;
    if (streamRecordIndex >= 0)
        streamRecord = cudaStreams[streamRecordIndex % numStreams];
    if (streamWaitIndex >= 0)
        streamWait   = cudaStreams[streamWaitIndex % numStreams];

    if (streamRecordIndex >= 0 && streamWaitIndex < 0) {
        // record streamRecord and wait on all others
        SAFE_CUDA(cuEventRecord(cudaEvents[numStreams], streamRecord));
        for (int i = 0; i < numStreams; i++) {
            if (i != (streamRecordIndex % numStreams)) {
                SAFE_CUDA(cuStreamWaitEvent(cudaStreams[i], cudaEvents[numStreams], 0));
            }
        }
    } else if (streamRecordIndex < 0 && streamWaitIndex >= 0) {
         // record all other streams and wait on streamWait 
        for (int i = 0; i < numStreams; i++) {
            if (i != (streamWaitIndex % numStreams)) {
                SAFE_CUDA(cuEventRecord(cudaEvents[i], cudaStreams[i]));
                SAFE_CUDA(cuStreamWaitEvent(streamWait, cudaEvents[i], 0));
            }
        }
    } else {
         // record streamRecord and wait on streamWait 
        SAFE_CUDA(cuEventRecord(cudaEvents[numStreams], streamRecord));
        SAFE_CUDA(cuStreamWaitEvent(streamWait, cudaEvents[numStreams], 0));
    }

    SAFE_CUDA(cuCtxPopCurrent(&cudaContext));

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::SynchronizeDeviceWithIndex\n");
#endif
}

void GPUInterface::GraphCaptureBegin() {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::GraphCaptureBegin\n");
#endif

    SAFE_CUPP(cuStreamBeginCapture(cudaStreams[0], CU_STREAM_CAPTURE_MODE_GLOBAL));

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::GraphCaptureBegin\n");
#endif
}


void GPUInterface::GraphCaptureEnd() {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::GraphCaptureEnd\n");
#endif

        SAFE_CUPP(cuStreamEndCapture(cudaStreams[0], &cudaGraph));

        SAFE_CUPP(cuGraphInstantiate(&cudaExecGraph, cudaGraph, NULL, NULL, 0));

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::GraphCaptureEnd\n");
#endif
}

void GPUInterface::GraphLaunch() {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::GraphLaunch\n");
#endif

        SAFE_CUDA(cuGraphLaunch(cudaExecGraph, cudaStreams[0]));

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::GraphLaunch\n");
#endif
}

GPUFunction GPUInterface::GetFunction(const char* functionName) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::GetFunction\n");
#endif

    GPUFunction cudaFunction;

    SAFE_CUPP(cuModuleGetFunction(&cudaFunction, cudaModule, functionName));

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::GetFunction\n");
#endif

    return cudaFunction;
}

void GPUInterface::LaunchKernel(GPUFunction deviceFunction,
                                         Dim3Int block,
                                         Dim3Int grid,
                                         int parameterCountV,
                                         int totalParameterCount,
                                         ...) { // parameters
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::LaunchKernel\n");
#endif

    SAFE_CUDA(cuCtxPushCurrent(cudaContext));

    void** params;
    GPUPtr* paramPtrs;
    unsigned int* paramInts;

    params = (void**)malloc(sizeof(void*) * totalParameterCount);
    paramPtrs = (GPUPtr*)malloc(sizeof(GPUPtr) * totalParameterCount);
    paramInts = (unsigned int*)malloc(sizeof(unsigned int) * totalParameterCount);

    va_list parameters;
    va_start(parameters, totalParameterCount);
    for(int i = 0; i < parameterCountV; i++) {
       paramPtrs[i] = (GPUPtr)(size_t)va_arg(parameters, GPUPtr);
       params[i] = (void*)&paramPtrs[i];
    }
    for(int i = parameterCountV; i < totalParameterCount; i++) {
       paramInts[i-parameterCountV] = va_arg(parameters, unsigned int);
       params[i] = (void*)&paramInts[i-parameterCountV];
    }

    va_end(parameters);

    SAFE_CUDA(cuLaunchKernel(deviceFunction, grid.x, grid.y, grid.z,
                             block.x, block.y, block.z, 0,
                             cudaStreams[0], params, NULL));

    free(params);
    free(paramPtrs);
    free(paramInts);

    SAFE_CUDA(cuCtxPopCurrent(&cudaContext));

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::LaunchKernel\n");
#endif

}

void GPUInterface::LaunchKernelConcurrent(GPUFunction deviceFunction,
                                         Dim3Int block,
                                         Dim3Int grid,
                                         int streamIndex,
                                         int waitIndex,
                                         int parameterCountV,
                                         int totalParameterCount,
                                         ...) { // parameters
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::LaunchKernelConcurrent\n");
#endif

    SAFE_CUDA(cuCtxPushCurrent(cudaContext));

    void** params;
    GPUPtr* paramPtrs;
    unsigned int* paramInts;

    params = (void**)malloc(sizeof(void*) * totalParameterCount);
    paramPtrs = (GPUPtr*)malloc(sizeof(GPUPtr) * totalParameterCount);
    paramInts = (unsigned int*)malloc(sizeof(unsigned int) * totalParameterCount);

    va_list parameters;
    va_start(parameters, totalParameterCount);
    for(int i = 0; i < parameterCountV; i++) {
       paramPtrs[i] = (GPUPtr)(size_t)va_arg(parameters, GPUPtr);
       params[i] = (void*)&paramPtrs[i];
    }
    for(int i = parameterCountV; i < totalParameterCount; i++) {
       paramInts[i-parameterCountV] = va_arg(parameters, unsigned int);
       params[i] = (void*)&paramInts[i-parameterCountV];
    }

    va_end(parameters);

    if (streamIndex >= 0) {
        int streamIndexMod = streamIndex % numStreams;

        if (waitIndex >= 0) {
            int waitIndexMod = waitIndex % numStreams;
            // SAFE_CUDA(cuStreamSynchronize(cudaStreams[waitIndexMod]));
            SAFE_CUDA(cuStreamWaitEvent(cudaStreams[streamIndexMod], cudaEvents[waitIndexMod], 0));
        }

        // printf("stream %d launching\n", streamIndexMod);

        SAFE_CUDA(cuLaunchKernel(deviceFunction, grid.x, grid.y, grid.z,
                                 block.x, block.y, block.z, 0,
                                 cudaStreams[streamIndexMod], params, NULL));

        SAFE_CUDA(cuEventRecord(cudaEvents[streamIndexMod], cudaStreams[streamIndexMod]));

    } else {
        SAFE_CUDA(cuLaunchKernel(deviceFunction, grid.x, grid.y, grid.z,
                                 block.x, block.y, block.z, 0,
                                 cudaStreams[0], params, NULL));
    }

    free(params);
    free(paramPtrs);
    free(paramInts);

    SAFE_CUDA(cuCtxPopCurrent(&cudaContext));

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::LaunchKernelConcurrent\n");
#endif

}

void* GPUInterface::MallocHost(size_t memSize) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::MallocHost\n");
#endif

    void* ptr;

#ifdef BEAGLE_MEMORY_PINNED
    ptr = AllocatePinnedHostMemory(memSize, false, false);
#else
    ptr = malloc(memSize);
#endif

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::MallocHost\n");
#endif

    return ptr;
}

void* GPUInterface::CallocHost(size_t size, size_t length) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::CallocHost\n");
#endif

    void* ptr;
    size_t memSize = size * length;

#ifdef BEAGLE_MEMORY_PINNED
    ptr = AllocatePinnedHostMemory(memSize, false, false);
    memset(ptr, 0, memSize);
#else
    ptr = calloc(size, length);
#endif

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::CallocHost\n");
#endif

    return ptr;
}

void* GPUInterface::AllocatePinnedHostMemory(size_t memSize, bool writeCombined, bool mapped) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::AllocatePinnedHostMemory\n");
#endif

    void* ptr;

    unsigned int flags = 0;

    if (writeCombined)
        flags |= CU_MEMHOSTALLOC_WRITECOMBINED;
    if (mapped)
        flags |= CU_MEMHOSTALLOC_DEVICEMAP;

    SAFE_CUPP(cuMemHostAlloc(&ptr, memSize, flags));


#ifdef BEAGLE_DEBUG_VALUES
    fprintf(stderr, "Allocated pinned host (CPU) memory %ld to %lu .\n", (long)ptr, ((long)ptr + memSize));
#endif

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::AllocatePinnedHostMemory\n");
#endif

    return ptr;
}

GPUPtr GPUInterface::AllocateMemory(size_t memSize) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::AllocateMemory\n");
#endif

    GPUPtr ptr;

    SAFE_CUPP(cuMemAlloc(&ptr, memSize));

#ifdef BEAGLE_DEBUG_VALUES
    fprintf(stderr, "Allocated GPU memory %llu to %llu.\n", (unsigned long long)ptr, (unsigned long long)(ptr + memSize));
#endif

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::AllocateMemory\n");
#endif

    return ptr;
}

GPUPtr GPUInterface::AllocateRealMemory(size_t length) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::AllocateRealMemory\n");
#endif

    GPUPtr ptr;

    SAFE_CUPP(cuMemAlloc(&ptr, SIZE_REAL * length));

#ifdef BEAGLE_DEBUG_VALUES
    fprintf(stderr, "Allocated GPU memory %llu to %llu.\n", (unsigned long long)ptr, (unsigned long long)(ptr + length));
#endif

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::AllocateRealMemory\n");
#endif

    return ptr;
}

GPUPtr GPUInterface::AllocateIntMemory(size_t length) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::AllocateIntMemory\n");
#endif

    GPUPtr ptr;

    SAFE_CUPP(cuMemAlloc(&ptr, SIZE_INT * length));

#ifdef BEAGLE_DEBUG_VALUES
    fprintf(stderr, "Allocated GPU memory %llu to %llu.\n", (unsigned long long)ptr, (unsigned long long)(ptr + length));
#endif

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::AllocateIntMemory\n");
#endif

    return ptr;
}

GPUPtr GPUInterface::CreateSubPointer(GPUPtr dPtr,
                                      size_t offset,
                                      size_t size) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::CreateSubPointer\n");
#endif

    GPUPtr subPtr = dPtr + offset;

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::CreateSubPointer\n");
#endif

    return subPtr;
}

size_t GPUInterface::AlignMemOffset(size_t offset) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::AlignMemOffset\n");
#endif

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::AlignMemOffset\n");
#endif

    return offset;
}

void GPUInterface::MemsetShort(GPUPtr dest,
                               unsigned short val,
                               size_t count) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::MemsetShort\n");
#endif

    SAFE_CUPP(cuMemsetD16(dest, val, count));

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::MemsetShort\n");
#endif

}

void GPUInterface::MemcpyHostToDevice(GPUPtr dest,
                                      const void* src,
                                      size_t memSize) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::MemcpyHostToDevice\n");
#endif

    SAFE_CUPP(cuMemcpyHtoDAsync(dest, src, memSize, cudaStreams[0]));

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::MemcpyHostToDevice\n");
#endif

}

void GPUInterface::MemcpyDeviceToHost(void* dest,
                                      const GPUPtr src,
                                      size_t memSize) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::MemcpyDeviceToHost\n");
#endif

    SAFE_CUPP(cuMemcpyDtoHAsync(dest, src, memSize, cudaStreams[0]));

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::MemcpyDeviceToHost\n");
#endif

}

void GPUInterface::MemcpyDeviceToDevice(GPUPtr dest,
                                        GPUPtr src,
                                        size_t memSize) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::MemcpyDeviceToDevice\n");
#endif

    SAFE_CUPP(cuMemcpyDtoDAsync(dest, src, memSize, cudaStreams[0]));

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::MemcpyDeviceToDevice\n");
#endif

}

void GPUInterface::FreeHostMemory(void* hPtr) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::FreeHostMemory\n");
#endif

#ifdef BEAGLE_MEMORY_PINNED
    FreePinnedHostMemory(hPtr);
#else
    free(hPtr);
#endif

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::FreeHostMemory\n");
#endif
}

void GPUInterface::FreePinnedHostMemory(void* hPtr) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::FreePinnedHostMemory\n");
#endif

    SAFE_CUPP(cuMemFreeHost(hPtr));

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::FreePinnedHostMemory\n");
#endif
}

void GPUInterface::FreeMemory(GPUPtr dPtr) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::FreeMemory\n");
#endif

    SAFE_CUPP(cuMemFree(dPtr));

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::FreeMemory\n");
#endif
}

GPUPtr GPUInterface::GetDeviceHostPointer(void* hPtr) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::GetDeviceHostPointer\n");
#endif

    GPUPtr dPtr;

    SAFE_CUPP(cuMemHostGetDevicePointer(&dPtr, hPtr, 0));

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::GetDeviceHostPointer\n");
#endif

    return dPtr;
}

size_t GPUInterface::GetAvailableMemory() {
#if CUDA_VERSION >= 3020
    size_t availableMem = 0;
    size_t totalMem = 0;
    SAFE_CUPP(cuMemGetInfo(&availableMem, &totalMem));
#else
    unsigned int availableMem = 0;
    unsigned int totalMem = 0;
    SAFE_CUPP(cuMemGetInfo(&availableMem, &totalMem));
#endif
    return (size_t) availableMem;
}

void GPUInterface::GetDeviceName(int deviceNumber,
                                  char* deviceName,
                                  int nameLength) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::GetDeviceName\n");
#endif

    CUdevice tmpCudaDevice;

    SAFE_CUDA(cuDeviceGet(&tmpCudaDevice, (*resourceMap)[deviceNumber]));

    SAFE_CUDA(cuDeviceGetName(deviceName, nameLength, tmpCudaDevice));

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::GetDeviceName\n");
#endif
}

bool GPUInterface::GetSupportsDoublePrecision(int deviceNumber) {
    CUdevice tmpCudaDevice;
    SAFE_CUDA(cuDeviceGet(&tmpCudaDevice, (*resourceMap)[deviceNumber]));

    int major = 0;
    int minor = 0;
    SAFE_CUDA(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, tmpCudaDevice));
    SAFE_CUDA(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, tmpCudaDevice));
    return (major >= 2 || (major >= 1 && minor >= 3));
}

void GPUInterface::GetDeviceDescription(int deviceNumber,
                                        char* deviceDescription) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::GetDeviceDescription\n");
#endif

    CUdevice tmpCudaDevice;

    SAFE_CUDA(cuDeviceGet(&tmpCudaDevice, (*resourceMap)[deviceNumber]));

#if CUDA_VERSION >= 3020
    size_t totalGlobalMemory = 0;
#else
    unsigned int totalGlobalMemory = 0;
#endif
    int clockSpeed = 0;
    int mpCount = 0;
    int major = 0;
    int minor = 0;

    SAFE_CUDA(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, tmpCudaDevice));
    SAFE_CUDA(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, tmpCudaDevice));
    SAFE_CUDA(cuDeviceTotalMem(&totalGlobalMemory, tmpCudaDevice));
    SAFE_CUDA(cuDeviceGetAttribute(&clockSpeed, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, tmpCudaDevice));
    SAFE_CUDA(cuDeviceGetAttribute(&mpCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, tmpCudaDevice));

    sprintf(deviceDescription,
            "Global memory (MB): %d | Clock speed (Ghz): %1.2f | Number of cores: %d",
            int(totalGlobalMemory / 1024.0 / 1024.0 + 0.5),
            clockSpeed / 1000000.0,
            util::ConvertSMVer2CoresDRV(major, minor) * mpCount);

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::GetDeviceDescription\n");
#endif
}

void GPUInterface::PrintfDeviceInt(GPUPtr dPtr,
                             int length) {
    int* hPtr = (int*) malloc(SIZE_INT * length);

    MemcpyDeviceToHost(hPtr, dPtr, SIZE_INT * length);

    printfInt(hPtr, length);

    free(hPtr);
}

long GPUInterface::GetDeviceTypeFlag(int deviceNumber) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::GetDeviceTypeFlag\n");
#endif

    long deviceTypeFlag = BEAGLE_FLAG_PROCESSOR_GPU;

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::GetDeviceTypeFlag\n");
#endif

    return deviceTypeFlag;
}


BeagleDeviceImplementationCodes GPUInterface::GetDeviceImplementationCode(int deviceNumber) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::GetDeviceImplementationCode\n");
#endif

    BeagleDeviceImplementationCodes deviceCode = BEAGLE_CUDA_DEVICE_NVIDIA_GPU;

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::GetDeviceImplementationCode\n");
#endif

    return deviceCode;
}

const char* GPUInterface::GetCUDAErrorDescription(int errorCode) {

    const char* errorDesc;

    // from cuda.h
    switch(errorCode) {
        case CUDA_SUCCESS: errorDesc = "No errors"; break;
        case CUDA_ERROR_INVALID_VALUE: errorDesc = "Invalid value"; break;
        case CUDA_ERROR_OUT_OF_MEMORY: errorDesc = "Out of memory"; break;
        case CUDA_ERROR_NOT_INITIALIZED: errorDesc = "Driver not initialized"; break;
        case CUDA_ERROR_DEINITIALIZED: errorDesc = "Driver deinitialized"; break;

        case CUDA_ERROR_NO_DEVICE: errorDesc = "No CUDA-capable device available"; break;
        case CUDA_ERROR_INVALID_DEVICE: errorDesc = "Invalid device"; break;

        case CUDA_ERROR_INVALID_IMAGE: errorDesc = "Invalid kernel image"; break;
        case CUDA_ERROR_INVALID_CONTEXT: errorDesc = "Invalid context"; break;
        case CUDA_ERROR_CONTEXT_ALREADY_CURRENT: errorDesc = "Context already current"; break;
        case CUDA_ERROR_MAP_FAILED: errorDesc = "Map failed"; break;
        case CUDA_ERROR_UNMAP_FAILED: errorDesc = "Unmap failed"; break;
        case CUDA_ERROR_ARRAY_IS_MAPPED: errorDesc = "Array is mapped"; break;
        case CUDA_ERROR_ALREADY_MAPPED: errorDesc = "Already mapped"; break;
        case CUDA_ERROR_NO_BINARY_FOR_GPU: errorDesc = "No binary for GPU"; break;
        case CUDA_ERROR_ALREADY_ACQUIRED: errorDesc = "Already acquired"; break;
        case CUDA_ERROR_NOT_MAPPED: errorDesc = "Not mapped"; break;

        case CUDA_ERROR_INVALID_SOURCE: errorDesc = "Invalid source"; break;
        case CUDA_ERROR_FILE_NOT_FOUND: errorDesc = "File not found"; break;

        case CUDA_ERROR_INVALID_HANDLE: errorDesc = "Invalid handle"; break;

        case CUDA_ERROR_NOT_FOUND: errorDesc = "Not found"; break;

        case CUDA_ERROR_NOT_READY: errorDesc = "CUDA not ready"; break;

        case CUDA_ERROR_LAUNCH_FAILED: errorDesc = "Launch failed"; break;
        case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: errorDesc = "Launch exceeded resources"; break;
        case CUDA_ERROR_LAUNCH_TIMEOUT: errorDesc = "Launch exceeded timeout"; break;
        case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING: errorDesc =
            "Launch with incompatible texturing"; break;

        case CUDA_ERROR_UNKNOWN: errorDesc = "Unknown error"; break;

        default: errorDesc = "Unknown error";
    }

    return errorDesc;
}

}; // namespace
