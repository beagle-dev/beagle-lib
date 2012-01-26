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

#include "libhmsbeagle/GPU/GPUImplDefs.h"
#include "libhmsbeagle/GPU/GPUImplHelper.h"
#include "libhmsbeagle/GPU/GPUInterface.h"

#define SAFE_CL(call)   { \
                            int error = call; \
                            if(error != CL_SUCCESS) { \
                                fprintf(stderr, \
                                    "OpenCL error: \"%s\" from file <%s>, line %i.\n", \
                                    GetCLErrorDescription(error), __FILE__, __LINE__); \
                                exit(-1); \
                            } \
                        }

GPUInterface::GPUInterface() {    
    openClDeviceId = NULL;
    openClContext = NULL;
    openClCommandQueue = NULL;
    openClProgram = NULL;
    openClNumDevices = NULL;
}

GPUInterface::~GPUInterface() {
    
    // TODO: cleanup mem objects, kernels
    
    if (openClProgram != NULL)
        SAFE_CL(clReleaseProgram(openClProgram));

    if (openClCommandQueue != NULL)
        SAFE_CL(clReleaseCommandQueue(openClCommandQueue));
    
    if (openClContext != NULL)
        SAFE_CL(clReleaseContext(openClContext));
}

int GPUInterface::Initialize() {
    // TODO: check for devices and return 0 if none;
    
    return 1;
}

int GPUInterface::GetDeviceCount() {        
    SAFE_CL(clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, NULL, NULL,
                           &openClNumDevices));
    
    return openClNumDevices;
}

void GPUInterface::DestroyKernelMap() {
}

void GPUInterface::InitializeKernelMap() {
}

void GPUInterface::SetDevice(int deviceNumber,
                             int paddedStateCount,
                             int categoryCount,
                             int paddedPatternCount,
                             long flags) {
    
    cl_device_id  deviceIds[openClNumDevices];
    
    SAFE_CL(clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, openClNumDevices, deviceIds,
                           NULL));
    
    openClDeviceId = deviceIds[deviceNumber];
    
    int err;
    
    openClContext = clCreateContext(NULL, 1, &openClDeviceId, NULL, NULL, &err);
    SAFE_CL(err);
    
    openClCommandQueue = clCreateCommandQueue(openClContext, openClDeviceId, 0, &err);
    SAFE_CL(err);
    
    const char* kernelsString = KERNELS_STRING;
    
    openClProgram = clCreateProgramWithSource(openClContext, 1,
                                              (const char**) &kernelsString, NULL,
                                              &err);
    SAFE_CL(err);
    if (!openClProgram) {
        fprintf(stderr, "OpenCL error: Failed to create kernels\n");
        exit(-1);
    }
    
    //    err = clBuildProgram(openClProgram, 0, NULL, BEAGLE_OPENCL_BUILD_OPTIONS, NULL, NULL);
    err = clBuildProgram(openClProgram, 0, NULL, "-D STATE_COUNT=4", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t len;
        char buffer[2048];
        
        fprintf(stderr, "OpenCL error: Failed to build kernels\n");
        
        clGetProgramBuildInfo(openClProgram, openClDeviceId, CL_PROGRAM_BUILD_LOG,
                              sizeof(buffer), buffer, &len);
        
        fprintf(stderr, "%s\n", buffer);
        
        exit(-1);
    }
    
    SAFE_CL(clUnloadCompiler());
}


void GPUInterface::Synchronize() {
    SAFE_CL(clFinish(openClCommandQueue));
}

GPUFunction GPUInterface::GetFunction(const char* functionName) {
    GPUFunction openClFunction;

    int err;
    openClFunction = clCreateKernel(openClProgram, functionName, &err);
    SAFE_CL(err);
    if (!openClFunction) {
        fprintf(stderr, "OpenCL error: Failed to create compute kernel %s\n", functionName);
        exit(-1);
    }
    
    return openClFunction;
}

void GPUInterface::LaunchKernel(GPUFunction deviceFunction,
                                Dim3Int block,
                                Dim3Int grid,
                                int parameterCountV,
                                int totalParameterCount,
                                ...) { // parameters   
    va_list parameters;
    va_start(parameters, totalParameterCount);  
    for(int i = 0; i < totalParameterCount; i++) {
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
}

void* GPUInterface::MallocHost(size_t memSize) {
//#ifdef BEAGLE_DEBUG_FLOW
//    fprintf(stderr,"\t\t\tEntering GPUInterface::MallocHost\n");
//#endif
//    
//    void* ptr;
//    
//#ifdef BEAGLE_MEMORY_PINNED
//    ptr = AllocatePinnedHostMemory(memSize, false, false);
//#else
//    ptr = malloc(memSize);
//#endif
//    
//#ifdef BEAGLE_DEBUG_FLOW
//    fprintf(stderr, "\t\t\tLeaving  GPUInterface::MallocHost\n");
//#endif
//    
//    return ptr;
}

void* GPUInterface::CallocHost(size_t size, size_t length) {
//#ifdef BEAGLE_DEBUG_FLOW
//    fprintf(stderr,"\t\t\tEntering GPUInterface::CallocHost\n");
//#endif
//    
//    void* ptr;
//    size_t memSize = size * length;
//    
//#ifdef BEAGLE_MEMORY_PINNED
//    ptr = AllocatePinnedHostMemory(memSize, false, false);
//    memset(ptr, 0, memSize);
//#else
//    ptr = calloc(size, length);
//#endif
//    
//#ifdef BEAGLE_DEBUG_FLOW
//    fprintf(stderr, "\t\t\tLeaving  GPUInterface::CallocHost\n");
//#endif
//    
//    return ptr;
}

void* GPUInterface::AllocatePinnedHostMemory(size_t memSize, bool writeCombined, bool mapped) {
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
    GPUPtr data;
    
    int err;
    data = clCreateBuffer(openClContext, CL_MEM_READ_WRITE, memSize, NULL, &err);
    SAFE_CL(err);
    
    return data;
}

GPUPtr GPUInterface::AllocateRealMemory(size_t length) {
    GPUPtr data;

    int err;
    data = clCreateBuffer(openClContext, CL_MEM_READ_WRITE, SIZE_REAL * length, NULL,
                          &err);
    SAFE_CL(err);
    
    return data;
}

GPUPtr GPUInterface::AllocateIntMemory(size_t length) {
    GPUPtr data;
    
    int err;
    data = clCreateBuffer(openClContext, CL_MEM_READ_WRITE, SIZE_INT * length, NULL,
                          &err);
    SAFE_CL(err);
        
    return data;
}

void GPUInterface::MemsetShort(GPUPtr dest,
                               unsigned short val,
                               size_t count) {
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
    SAFE_CL(clEnqueueWriteBuffer(openClCommandQueue, dest, CL_TRUE, 0, memSize, src, 0,
                                 NULL, NULL));
}

void GPUInterface::MemcpyDeviceToHost(void* dest,
                                      const GPUPtr src,
                                      size_t memSize) {
    SAFE_CL(clEnqueueReadBuffer(openClCommandQueue, src, CL_TRUE, 0, memSize, dest, 0,
                                NULL, NULL));
}

void GPUInterface::MemcpyDeviceToDevice(GPUPtr dest,
                                        GPUPtr src,
                                        size_t memSize) {
//#ifdef BEAGLE_DEBUG_FLOW
//    fprintf(stderr, "\t\t\tEntering GPUInterface::MemcpyDeviceToDevice\n");
//#endif    
//    
//    SAFE_CUPP(cuMemcpyDtoD(dest, src, memSize));
//    
//#ifdef BEAGLE_DEBUG_FLOW
//    fprintf(stderr, "\t\t\tLeaving  GPUInterface::MemcpyDeviceToDevice\n");
//#endif    
    
}


void GPUInterface::FreeHostMemory(void* hPtr) {
//#ifdef BEAGLE_DEBUG_FLOW
//    fprintf(stderr, "\t\t\tEntering GPUInterface::FreeHostMemory\n");
//#endif
//    
//#ifdef BEAGLE_MEMORY_PINNED
//    FreePinnedHostMemory(hPtr);
//#else
//    free(hPtr);
//#endif
//    
//#ifdef BEAGLE_DEBUG_FLOW
//    fprintf(stderr,"\t\t\tLeaving  GPUInterface::FreeHostMemory\n");
//#endif
}

void GPUInterface::FreePinnedHostMemory(void* hPtr) {
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
    SAFE_CL(clReleaseMemObject(dPtr));
}

GPUPtr GPUInterface::GetDevicePointer(void* hPtr) {
//#ifdef BEAGLE_DEBUG_FLOW
//    fprintf(stderr, "\t\t\tEntering GPUInterface::GetDevicePointer\n");
//#endif
//    
//    GPUPtr dPtr;
//    
//    SAFE_CUPP(cuMemHostGetDevicePointer(&dPtr, hPtr, 0));
//    
//#ifdef BEAGLE_DEBUG_FLOW
//    fprintf(stderr,"\t\t\tLeaving  GPUInterface::GetDevicePointer\n");
//#endif
//    
//    return dPtr;
}

unsigned int GPUInterface::GetAvailableMemory() {
    return 0;
}

void GPUInterface::GetDeviceName(int deviceNumber,
                                 char* deviceName,
                                 int nameLength) {
    cl_device_id  deviceIds[openClNumDevices];
    
    SAFE_CL(clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, openClNumDevices, deviceIds,
                           NULL));    
    
    SAFE_CL(clGetDeviceInfo(deviceIds[deviceNumber], CL_DEVICE_NAME, sizeof(char) * nameLength, deviceName, NULL));    
}

bool GPUInterface::GetSupportsDoublePrecision(int deviceNumber) {
	return false;
}

void GPUInterface::GetDeviceDescription(int deviceNumber,
                                        char* deviceDescription) {   
    
    cl_device_id  deviceIds[openClNumDevices];
    
    SAFE_CL(clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, openClNumDevices, deviceIds,
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

