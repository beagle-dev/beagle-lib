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

#include "libhmsbeagle/beagle.h"
#include "libhmsbeagle/GPU/GPUImplDefs.h"
#include "libhmsbeagle/GPU/GPUImplHelper.h"
#include "libhmsbeagle/GPU/GPUInterface.h"
#include "libhmsbeagle/GPU/KernelResource.h"

#define SAFE_CL(call)   { \
                            int error = call; \
                            if(error != CL_SUCCESS) { \
                                fprintf(stderr, \
                                    "OpenCL error: \"%s\" from file <%s>, line %i.\n", \
                                    GetCLErrorDescription(error), __FILE__, __LINE__); \
                                exit(-1); \
                            } \
                        }

#define LOAD_KERNEL_INTO_MAP(state, prec, map, id) \
	    KernelResource kernel##state##prec = KernelResource( \
	        state, \
	        (char*) KERNELS_STRING_##prec##_##state, \
	        PATTERN_BLOCK_SIZE_##prec##_##state, \
	        MATRIX_BLOCK_SIZE_##prec##_##state, \
	        BLOCK_PEELING_SIZE_##prec##_##state, \
	        SLOW_REWEIGHING_##prec##_##state, \
	        MULTIPLY_BLOCK_SIZE_##prec, \
	        0,0,0); \
	    map->insert(std::make_pair(id,kernel##state##prec));

std::map<int, KernelResource>* kernelMap = NULL;

GPUInterface::GPUInterface() {    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::GPUInterface\n");
#endif    
    
    openClDeviceId = NULL;
    openClContext = NULL;
    openClCommandQueue = NULL;
    openClProgram = NULL;
    openClNumDevices = 0;
    openClPlatform = NULL;
    
    supportDoublePrecision = true;
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::GPUInterface\n");
#endif    
}

GPUInterface::~GPUInterface() {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::~GPUInterface\n");
#endif    
    
    // TODO: cleanup mem objects, kernels
    
    if (openClProgram != NULL)
        SAFE_CL(clReleaseProgram(openClProgram));

    if (openClCommandQueue != NULL)
        SAFE_CL(clReleaseCommandQueue(openClCommandQueue));
    
    if (openClContext != NULL)
        SAFE_CL(clReleaseContext(openClContext));
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::~GPUInterface\n");
#endif    
}

int GPUInterface::Initialize() {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::Initialize\n");
#endif    
    
    // TODO: check for devices and return 0 if none;
    
    return 1;
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::Initialize\n");
#endif    
}

int GPUInterface::GetDeviceCount() {       
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::GetDeviceCount\n");
#endif        
    
    // TODO: allow platform selection
    SAFE_CL(clGetPlatformIDs(1, &openClPlatform, NULL));
    
    SAFE_CL(clGetDeviceIDs(openClPlatform, CL_DEVICE_TYPE_GPU, 0, NULL,
                           &openClNumDevices));
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::GetDeviceCount\n");
#endif            
    
    return openClNumDevices;
}

void GPUInterface::DestroyKernelMap() {
    if (kernelMap) {
        delete kernelMap;
    }
}

void GPUInterface::InitializeKernelMap() {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::InitializeKernelMap\n");
#endif        

    kernelMap = new std::map<int, KernelResource>;
    
    LOAD_KERNEL_INTO_MAP(4,   SP, kernelMap, 4  );
    LOAD_KERNEL_INTO_MAP(16,  SP, kernelMap, 16 );
    LOAD_KERNEL_INTO_MAP(32,  SP, kernelMap, 32 );
    LOAD_KERNEL_INTO_MAP(48,  SP, kernelMap, 48 );
    LOAD_KERNEL_INTO_MAP(64,  SP, kernelMap, 64 );
    LOAD_KERNEL_INTO_MAP(80,  SP, kernelMap, 80 );
    LOAD_KERNEL_INTO_MAP(128, SP, kernelMap, 128);
    LOAD_KERNEL_INTO_MAP(192, SP, kernelMap, 192);
    
    if (supportDoublePrecision) {
        LOAD_KERNEL_INTO_MAP(4,   DP, kernelMap, -4  );
        LOAD_KERNEL_INTO_MAP(16,  DP, kernelMap, -16 );
        LOAD_KERNEL_INTO_MAP(32,  DP, kernelMap, -32 );
        LOAD_KERNEL_INTO_MAP(48,  DP, kernelMap, -48 );
        LOAD_KERNEL_INTO_MAP(64,  DP, kernelMap, -64 );
        LOAD_KERNEL_INTO_MAP(80,  DP, kernelMap, -80 );
        LOAD_KERNEL_INTO_MAP(128, DP, kernelMap, -128);
        LOAD_KERNEL_INTO_MAP(192, DP, kernelMap, -192);
    }
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::GetDeviceCount\n");
#endif            
}

void GPUInterface::SetDevice(int deviceNumber,
                             int paddedStateCount,
                             int categoryCount,
                             int paddedPatternCount,
                             long flags) {
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::SetDevice\n");
#endif                
    
    cl_device_id  deviceIds[openClNumDevices];
    
    SAFE_CL(clGetDeviceIDs(openClPlatform, CL_DEVICE_TYPE_GPU, openClNumDevices, deviceIds,
                           NULL));
    
    openClDeviceId = deviceIds[deviceNumber];
        
    int err;
    
    openClContext = clCreateContext(NULL, 1, &openClDeviceId, NULL, NULL, &err);
    SAFE_CL(err);
    
    openClCommandQueue = clCreateCommandQueue(openClContext, openClDeviceId, 0, &err);
    SAFE_CL(err);
    
    if (kernelMap == NULL) {
        // kernels have not yet been initialized; do so now.  Hopefully, this only occurs once per library load.
        InitializeKernelMap();
    }
    
    int id = paddedStateCount;
    if (flags & BEAGLE_FLAG_PRECISION_DOUBLE) {
    	id *= -1;        
    }
    
    if (kernelMap->count(id) == 0) {
    	fprintf(stderr,"Critical error: unable to find kernel code for %d states.\n",paddedStateCount);
    	exit(-1);
    }
    
    kernelResource = (*kernelMap)[id].copy();
    kernelResource->categoryCount = categoryCount;
    kernelResource->patternCount = paddedPatternCount;
    kernelResource->flags = flags;
    
    openClProgram = clCreateProgramWithSource(openClContext, 1,
                                              (const char**) &kernelResource->kernelCode, NULL,
                                              &err);
    SAFE_CL(err);
    if (!openClProgram) {
        fprintf(stderr, "OpenCL error: Failed to create kernels\n");
        exit(-1);
    }
    
    err = clBuildProgram(openClProgram, 0, NULL, "-DOPENCL -DOPENCL_KERNEL_BUILD", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t len;
        char buffer[2048];
        
        fprintf(stderr, "OpenCL error: Failed to build kernels\n");
        
        clGetProgramBuildInfo(openClProgram, openClDeviceId, CL_PROGRAM_BUILD_LOG,
                              sizeof(buffer), buffer, &len);
        
        fprintf(stderr, "%s\n", buffer);
        
        exit(-1);
    }
    
#ifdef CL_VERSION_1_2
    SAFE_CL(clUnloadPlatformCompiler(openClPlatform));
#else
    SAFE_CL(clUnloadCompiler());
#endif
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::SetDevice\n");
#endif            
}


void GPUInterface::Synchronize() {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::Synchronize\n");
#endif                
    
    SAFE_CL(clFinish(openClCommandQueue));
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::Synchronize\n");
#endif                
}

GPUFunction GPUInterface::GetFunction(const char* functionName) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::GetFunction\n");
    fprintf(stderr,"\t\t\t\tFunction name: %s\n", functionName);
#endif                    

    GPUFunction openClFunction;
    
    int err;
    openClFunction = clCreateKernel(openClProgram, functionName, &err);
    SAFE_CL(err);
    if (!openClFunction) {
        fprintf(stderr, "OpenCL error: Failed to create compute kernel %s\n", functionName);
        exit(-1);
    }
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::GetFunction\n");
#endif                

    return openClFunction;
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
    
    va_list parameters;
    va_start(parameters, totalParameterCount);  
    for(int i = 0; i < parameterCountV; i++) {
        void* param = (void*)(size_t)va_arg(parameters, GPUPtr);
        
        SAFE_CL(clSetKernelArg(deviceFunction, i, sizeof(param), &param));
    }
    for(int i = parameterCountV; i < totalParameterCount; i++) {
        unsigned int param = va_arg(parameters, unsigned int);
        
        SAFE_CL(clSetKernelArg(deviceFunction, i, sizeof(unsigned int), &param));       
    }
    
    va_end(parameters);
    
    size_t localWorkSize[3];
    localWorkSize[0] = block.x;
    localWorkSize[1] = block.y;
    localWorkSize[2] = block.z;
    
    size_t globalWorkSize[3];
    globalWorkSize[0] = block.x * grid.x;
    globalWorkSize[1] = block.y * grid.y;
    globalWorkSize[2] = block.z * grid.z;
    
    SAFE_CL(clEnqueueNDRangeKernel(openClCommandQueue, deviceFunction, 3, NULL,
                                   globalWorkSize, localWorkSize, 0, NULL, NULL));
    
    Synchronize();
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::LaunchKernel\n");
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
    assert(0); // TODO: write function
//#ifdef BEAGLE_DEBUG_FLOW
//    fprintf(stderr,"\t\t\tEntering GPUInterface::AllocatePinnedHostMemory\n");
//#endif
//    
//    void* ptr;
//    
//    unsigned int flags = 0;
//    
//    if (writeCombined)
//        flags |= CU_MEMHOSTALLOC_WRITECOMBINED;
//    if (mapped)
//        flags |= CU_MEMHOSTALLOC_DEVICEMAP;
//    
//    SAFE_CUPP(cuMemHostAlloc(&ptr, memSize, flags));
//    
//    
//#ifdef BEAGLE_DEBUG_VALUES
//    fprintf(stderr, "Allocated pinned host (CPU) memory %ld to %lu .\n", (long)ptr, ((long)ptr + memSize));
//#endif
//    
//#ifdef BEAGLE_DEBUG_FLOW
//    fprintf(stderr, "\t\t\tLeaving  GPUInterface::AllocatePinnedHostMemory\n");
//#endif
//    
//    return ptr;
}


GPUPtr GPUInterface::AllocateMemory(size_t memSize) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::AllocateMemory\n");
#endif
    
    GPUPtr data;
    
    int err;
    data = clCreateBuffer(openClContext, CL_MEM_READ_WRITE, memSize, NULL, &err);
    SAFE_CL(err);
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::AllocateMemory\n");
#endif
    
    return data;
}

GPUPtr GPUInterface::AllocateRealMemory(size_t length) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::AllocateRealMemory\n");
#endif
    
    GPUPtr data;

    int err;
    data = clCreateBuffer(openClContext, CL_MEM_READ_WRITE, SIZE_REAL * length, NULL,
                          &err);
    SAFE_CL(err);
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::AllocateRealMemory\n");
#endif
    
    return data;
}

GPUPtr GPUInterface::AllocateIntMemory(size_t length) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::AllocateIntMemory\n");
#endif
    
    GPUPtr data;
    
    int err;
    data = clCreateBuffer(openClContext, CL_MEM_READ_WRITE, SIZE_INT * length, NULL,
                          &err);
    SAFE_CL(err);

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::AllocateIntMemory\n");
#endif
    
    return data;
}

GPUPtr GPUInterface::CreateSubPointer(GPUPtr dPtr,
                                      size_t offset, 
                                      size_t size) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::CreateSubPointer\n");
#endif    
    
    GPUPtr subPtr;
    
    cl_buffer_region dPtrRegion;
    dPtrRegion.origin = offset;
    dPtrRegion.size = size;
    
    int err;
    subPtr = clCreateSubBuffer(dPtr, 0, CL_BUFFER_CREATE_TYPE_REGION, &dPtrRegion, &err);
    SAFE_CL(err);
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::CreateSubPointer\n");
#endif    
    
    return subPtr;
}


void GPUInterface::MemsetShort(GPUPtr dest,
                               unsigned short val,
                               size_t count) {
    assert(0); // TODO: write function
//#ifdef BEAGLE_DEBUG_FLOW
//    fprintf(stderr, "\t\t\tEntering GPUInterface::MemsetShort\n");
//#endif    
//    
//    SAFE_CUPP(cuMemsetD16(dest, val, count));
//    
//#ifdef BEAGLE_DEBUG_FLOW
//    fprintf(stderr, "\t\t\tLeaving  GPUInterface::MemsetShort\n");
//#endif    
    
}


void GPUInterface::MemcpyHostToDevice(GPUPtr dest,
                                      const void* src,
                                      size_t memSize) { 
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::MemcpyHostToDevice\n");
#endif    
    
    SAFE_CL(clEnqueueWriteBuffer(openClCommandQueue, dest, CL_TRUE, 0, memSize, src, 0,
                                 NULL, NULL));
    
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
    
    SAFE_CL(clEnqueueReadBuffer(openClCommandQueue, src, CL_TRUE, 0, memSize, dest, 0,
                                NULL, NULL));
    
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

    SAFE_CL(clEnqueueCopyBuffer(openClCommandQueue, src, dest, 0, 0, memSize, 0,
                                 NULL, NULL));
    
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
    assert(0); // TODO: write function
//#ifdef BEAGLE_DEBUG_FLOW
//    fprintf(stderr, "\t\t\tEntering GPUInterface::FreePinnedHostMemory\n");
//#endif
//    
//    SAFE_CUPP(cuMemFreeHost(hPtr));
//    
//#ifdef BEAGLE_DEBUG_FLOW
//    fprintf(stderr,"\t\t\tLeaving  GPUInterface::FreePinnedHostMemory\n");
//#endif
}

void GPUInterface::FreeMemory(GPUPtr dPtr) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::FreeMemory\n");
#endif
    
    SAFE_CL(clReleaseMemObject(dPtr));
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::FreeMemory\n");
#endif
}

GPUPtr GPUInterface::GetDeviceHostPointer(void* hPtr) {
    assert(0); // TODO: write function
//#ifdef BEAGLE_DEBUG_FLOW
//    fprintf(stderr, "\t\t\tEntering GPUInterface::GetDeviceHostPointer\n");
//#endif
//    
//    GPUPtr dPtr;
//    
//    SAFE_CUPP(cuMemHostGetDeviceHostPointer(&dPtr, hPtr, 0));
//    
//#ifdef BEAGLE_DEBUG_FLOW
//    fprintf(stderr,"\t\t\tLeaving  GPUInterface::GetDeviceHostPointer\n");
//#endif
//    
//    return dPtr;
}

unsigned int GPUInterface::GetAvailableMemory() {
    assert(0); // TODO: write function
//    return availableMem;
}

void GPUInterface::GetDeviceName(int deviceNumber,
                                 char* deviceName,
                                 int nameLength) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::GetDeviceName\n");
#endif    
    
    cl_device_id  deviceIds[openClNumDevices];
    
    SAFE_CL(clGetDeviceIDs(openClPlatform, CL_DEVICE_TYPE_GPU, openClNumDevices, deviceIds,
                           NULL));    
    
    SAFE_CL(clGetDeviceInfo(deviceIds[deviceNumber], CL_DEVICE_NAME, sizeof(char) * nameLength, deviceName, NULL));    
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::GetDeviceName\n");
#endif            
}

bool GPUInterface::GetSupportsDoublePrecision(int deviceNumber) {

    cl_device_id  deviceIds[openClNumDevices];
    
    SAFE_CL(clGetDeviceIDs(openClPlatform, CL_DEVICE_TYPE_GPU, openClNumDevices, deviceIds,
                           NULL));    
    
    cl_uint supportsDouble = 0;
    
    SAFE_CL(clGetDeviceInfo(deviceIds[deviceNumber], CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, sizeof(cl_uint), &supportsDouble, NULL));

    return supportsDouble;
}

void GPUInterface::GetDeviceDescription(int deviceNumber,
                                        char* deviceDescription) {       
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::GetDeviceDescription\n");
#endif
    
    cl_device_id  deviceIds[openClNumDevices];
    
    SAFE_CL(clGetDeviceIDs(openClPlatform, CL_DEVICE_TYPE_GPU, openClNumDevices, deviceIds,
                           NULL));
    
    cl_device_id tmpOpenClDevice =deviceIds[deviceNumber]; 
    
    cl_ulong totalGlobalMemory = 0;
    cl_uint clockSpeed = 0;
    unsigned int mpCount = 0;
    
    SAFE_CL(clGetDeviceInfo(tmpOpenClDevice, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong),
                            &totalGlobalMemory, NULL));
    SAFE_CL(clGetDeviceInfo(tmpOpenClDevice, CL_DEVICE_MAX_CLOCK_FREQUENCY,
                            sizeof(cl_uint), &clockSpeed, NULL));
    SAFE_CL(clGetDeviceInfo(tmpOpenClDevice, CL_DEVICE_MAX_COMPUTE_UNITS,
                            sizeof(unsigned int), &mpCount, NULL));

    sprintf(deviceDescription,
            "Global memory (MB): %d | Clock speed (Ghz): %1.2f | Number of multiprocessors: %d",
            int(totalGlobalMemory / 1024.0 / 1024.0), clockSpeed / 1000.0, mpCount);
    
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

const char* GPUInterface::GetCLErrorDescription(int errorCode) {
    const char* errorDesc;
    
    // Error Codes (from cl.h)
    switch(errorCode) {
        case CL_SUCCESS: errorDesc = "CL_SUCCESS"; break;
        case CL_DEVICE_NOT_FOUND: errorDesc = "CL_DEVICE_NOT_FOUND"; break;
        case CL_DEVICE_NOT_AVAILABLE: errorDesc = "CL_DEVICE_NOT_AVAILABLE"; break;
        case CL_COMPILER_NOT_AVAILABLE: errorDesc = "CL_COMPILER_NOT_AVAILABLE"; break;
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: errorDesc = "CL_MEM_OBJECT_ALLOCATION_FAILURE"; break;
        case CL_OUT_OF_RESOURCES: errorDesc = "CL_OUT_OF_RESOURCES"; break;
        case CL_OUT_OF_HOST_MEMORY: errorDesc = "CL_OUT_OF_HOST_MEMORY"; break;
        case CL_PROFILING_INFO_NOT_AVAILABLE: errorDesc = "CL_PROFILING_INFO_NOT_AVAILABLE"; break;
        case CL_MEM_COPY_OVERLAP: errorDesc = "CL_MEM_COPY_OVERLAP"; break;
        case CL_IMAGE_FORMAT_MISMATCH: errorDesc = "CL_IMAGE_FORMAT_MISMATCH"; break;
        case CL_IMAGE_FORMAT_NOT_SUPPORTED: errorDesc = "CL_IMAGE_FORMAT_NOT_SUPPORTED"; break;
        case CL_BUILD_PROGRAM_FAILURE: errorDesc = "CL_BUILD_PROGRAM_FAILURE"; break;
        case CL_MAP_FAILURE: errorDesc = "CL_MAP_FAILURE"; break;
            
        case CL_INVALID_VALUE: errorDesc = "CL_INVALID_VALUE"; break;
        case CL_INVALID_DEVICE_TYPE: errorDesc = "CL_INVALID_DEVICE_TYPE"; break;
        case CL_INVALID_PLATFORM: errorDesc = "CL_INVALID_PLATFORM"; break;
        case CL_INVALID_DEVICE: errorDesc = "CL_INVALID_DEVICE"; break;
        case CL_INVALID_CONTEXT: errorDesc = "CL_INVALID_CONTEXT"; break;
        case CL_INVALID_QUEUE_PROPERTIES: errorDesc = "CL_INVALID_QUEUE_PROPERTIES"; break;
        case CL_INVALID_COMMAND_QUEUE: errorDesc = "CL_INVALID_COMMAND_QUEUE"; break;
        case CL_INVALID_HOST_PTR: errorDesc = "CL_INVALID_HOST_PTR"; break;
        case CL_INVALID_MEM_OBJECT: errorDesc = "CL_INVALID_MEM_OBJECT"; break;
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: errorDesc = "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR"; break;
        case CL_INVALID_IMAGE_SIZE: errorDesc = "CL_INVALID_IMAGE_SIZE"; break;
        case CL_INVALID_SAMPLER: errorDesc = "CL_INVALID_SAMPLER"; break;
        case CL_INVALID_BINARY: errorDesc = "CL_INVALID_BINARY"; break;
        case CL_INVALID_BUILD_OPTIONS: errorDesc = "CL_INVALID_BUILD_OPTIONS"; break;
        case CL_INVALID_PROGRAM: errorDesc = "CL_INVALID_PROGRAM"; break;
        case CL_INVALID_PROGRAM_EXECUTABLE: errorDesc = "CL_INVALID_PROGRAM_EXECUTABLE"; break;
        case CL_INVALID_KERNEL_NAME: errorDesc = "CL_INVALID_KERNEL_NAME"; break;
        case CL_INVALID_KERNEL_DEFINITION: errorDesc = "CL_INVALID_KERNEL_DEFINITION"; break;
        case CL_INVALID_KERNEL: errorDesc = "CL_INVALID_KERNEL"; break;
        case CL_INVALID_ARG_INDEX: errorDesc = "CL_INVALID_ARG_INDEX"; break;
        case CL_INVALID_ARG_VALUE: errorDesc = "CL_INVALID_ARG_VALUE"; break;
        case CL_INVALID_ARG_SIZE: errorDesc = "CL_INVALID_ARG_SIZE"; break;
        case CL_INVALID_KERNEL_ARGS: errorDesc = "CL_INVALID_KERNEL_ARGS"; break;
        case CL_INVALID_WORK_DIMENSION: errorDesc = "CL_INVALID_WORK_DIMENSION"; break;
        case CL_INVALID_WORK_GROUP_SIZE: errorDesc = "CL_INVALID_WORK_GROUP_SIZE"; break;
        case CL_INVALID_WORK_ITEM_SIZE: errorDesc = "CL_INVALID_WORK_ITEM_SIZE"; break;
        case CL_INVALID_GLOBAL_OFFSET: errorDesc = "CL_INVALID_GLOBAL_OFFSET"; break;
        case CL_INVALID_EVENT_WAIT_LIST: errorDesc = "CL_INVALID_EVENT_WAIT_LIST"; break;
        case CL_INVALID_EVENT: errorDesc = "CL_INVALID_EVENT"; break;
        case CL_INVALID_OPERATION: errorDesc = "CL_INVALID_OPERATION"; break;
        case CL_INVALID_GL_OBJECT: errorDesc = "CL_INVALID_GL_OBJECT"; break;
        case CL_INVALID_BUFFER_SIZE: errorDesc = "CL_INVALID_BUFFER_SIZE"; break;
        case CL_INVALID_MIP_LEVEL: errorDesc = "CL_INVALID_MIP_LEVEL"; break;
            
        default: errorDesc = "Unknown error";
    }
    
    return errorDesc;
}

