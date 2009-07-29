/*
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
    clDeviceId = NULL;
    clContext = NULL;
    clCommandQueue = NULL;
    clProgram = NULL;
}

GPUInterface::~GPUInterface() {
    
    // TODO: cleanup mem objects, kernels
    
    clReleaseProgram(clProgram);
    clReleaseCommandQueue(clCommandQueue);
    clReleaseContext(clContext);
}

int GPUInterface::GetDeviceCount() {    
    cl_uint numDevices = 0;
    
    SAFE_CL(clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, MAX_CL_DEVICES, NULL,
                           &numDevices));
    return numDevices;
}

void GPUInterface::SetDevice(int deviceNumber) {
    cl_uint numDevices;
    cl_device_id  deviceIds[MAX_CL_DEVICES];
    
    SAFE_CL(clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, MAX_CL_DEVICES, deviceIds,
                           &numDevices));
    
    clDeviceId = deviceIds[deviceNumber];
    
    int err;
    
    clContext = clCreateContext(NULL, 1, &clDeviceId, NULL, NULL, &err);
    SAFE_CL(err);
    
    clCommandQueue = clCreateCommandQueue(clContext, clDeviceId, 0, &err);
    SAFE_CL(err);
    
    const char* KernelSource = KERNELS_STRING;
    
    clProgram = clCreateProgramWithSource(clContext, 1, (const char **) & KernelSource,
                                          NULL, &err);
    SAFE_CL(err);
    if (!clProgram) {
        fprintf(stderr, "OpenCL error: Failed to create kernels\n");
        exit(-1);
    }
    
    err = clBuildProgram(clProgram, 0, NULL, OPENCL_BUILD_OPTIONS, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t len;
        char buffer[2048];
        
        fprintf(stderr, "OpenCL error: Failed to build kernels\n");
        
        clGetProgramBuildInfo(clProgram, clDeviceId, CL_PROGRAM_BUILD_LOG,
                              sizeof(buffer), buffer, &len);
        
        fprintf(stderr, "%s\n", buffer);
        
        exit(-1);
    }
    
    SAFE_CL(clUnloadCompiler());
}

void GPUInterface::Synchronize() {
//    SAFE_CUPP(cuCtxSynchronize());
}

GPUFunction GPUInterface::GetFunction(const char* functionName) {
//    GPUFunction cudaFunction; 
//    
//    SAFE_CUPP(cuModuleGetFunction(&cudaFunction, cudaModule, functionName));
//    
//    return cudaFunction;
}

void GPUInterface::LaunchKernelIntParams(GPUFunction deviceFunction,
                                         Dim3Int block,
                                         Dim3Int grid,
                                         int totalParameterCount,
                                         ...) { // unsigned int parameters
//    SAFE_CUDA(cuCtxPushCurrent(cudaContext));
//    
//    SAFE_CUDA(cuFuncSetBlockShape(deviceFunction, block.x, block.y, block.z));
//    
//    int offset = 0;
//    va_list parameters;
//    va_start(parameters, totalParameterCount);  
//    for(int i = 0; i < totalParameterCount; i++) {
//        unsigned int param = va_arg(parameters, unsigned int);
//        
//        // adjust offset alignment requirements
//        offset = (offset + __alignof(param) - 1) & ~(__alignof(param) - 1);
//        
//        SAFE_CUDA(cuParamSeti(deviceFunction, offset, param));
//        
//        offset += sizeof(param);
//    }
//    va_end(parameters);
//    
//    SAFE_CUDA(cuParamSetSize(deviceFunction, offset));
//    
//    SAFE_CUDA(cuLaunchGrid(deviceFunction, grid.x, grid.y));
//    
//    SAFE_CUDA(cuCtxPopCurrent(&cudaContext));
}


GPUPtr GPUInterface::AllocateMemory(int memSize) {
//#ifdef DEBUG
//    fprintf(stderr,"Entering ANMA\n");
//#endif
//    
//    GPUPtr data;
//    
//    SAFE_CUPP(cuMemAlloc(&data, memSize));
//    
//#ifdef DEBUG
//    fprintf(stderr, "Allocated %d to %d.\n", data, (data + memSize));
//    fprintf(stderr, "Leaving ANMA\n");
//#endif
//    
//    return data;
}

GPUPtr GPUInterface::AllocateRealMemory(int length) {
//#ifdef DEBUG
//    fprintf(stderr,"Entering ANMA-Real\n");
//#endif
//    
//    GPUPtr data;
//    
//    SAFE_CUPP(cuMemAlloc(&data, SIZE_REAL * length));
//    
//#ifdef DEBUG
//    fprintf(stderr, "Allocated %d to %d.\n", data, (data + length));
//    fprintf(stderr, "Leaving ANMA\n");
//#endif
//    
//    return data;
}

GPUPtr GPUInterface::AllocateIntMemory(int length) {
//#ifdef DEBUG
//    fprintf(stderr, "Entering ANMA-Int\n");
//#endif
//    
//    GPUPtr data;
//    
//    SAFE_CUPP(cuMemAlloc(&data, SIZE_INT * length));
//    
//#ifdef DEBUG
//    fprintf(stderr, "Allocated %d to %d.\n", data, (data + length));
//    fprintf(stderr, "Leaving ANMA\n");
//#endif
//    
//    return data;
}

void GPUInterface::MemcpyHostToDevice(GPUPtr dest,
                                      const void* src,
                                      int memSize) {
//    SAFE_CUPP(cuMemcpyHtoD(dest, src, memSize));
}

void GPUInterface::MemcpyDeviceToHost(void* dest,
                                      const GPUPtr src,
                                      int memSize) {
//    SAFE_CUPP(cuMemcpyDtoH(dest, src, memSize));
}

void GPUInterface::FreeMemory(GPUPtr dPtr) {
//#ifdef DEBUG
//    fprintf(stderr, "Entering FNMA\n");
//#endif
//    
//    SAFE_CUPP(cuMemFree(dPtr));
//    
//#ifdef DEBUG
//    fprintf(stderr,"Leaving FNMA\n");
//#endif
}
//
void GPUInterface::PrintInfo() {        
    fprintf(stderr, "GPU Device Information:");
    
    char name[100];
    cl_ulong totalGlobalMemory = 0;
    cl_uint clockSpeed = 0;
    unsigned int mpCount = 0;
    
    
    SAFE_CL(clGetDeviceInfo(clDeviceId, CL_DEVICE_NAME, sizeof(char) * 100, name,
                            NULL));
    SAFE_CL(clGetDeviceInfo(clDeviceId, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong),
                            &totalGlobalMemory, NULL));
    SAFE_CL(clGetDeviceInfo(clDeviceId, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint),
                            &clockSpeed, NULL));
    SAFE_CL(clGetDeviceInfo(clDeviceId, CL_DEVICE_MAX_COMPUTE_UNITS,
                            sizeof(unsigned int), &mpCount, NULL));
    
    fprintf(stderr, "\nDevice: %s\n", name);
    fprintf(stderr, "\tGlobal Memory (MB): %1.2f\n", totalGlobalMemory / 1024 / 1024.0);
    fprintf(stderr, "\tClock Speed (Ghz) : %1.2f\n", clockSpeed / 1000.0);
    fprintf(stderr, "\tNumber of Cores   : %d\n", mpCount);
    
}

void GPUInterface::PrintfDeviceVector(GPUPtr dPtr,
                                      int length) {
//    REAL* hPtr = (REAL*) malloc(SIZE_REAL * length);
//    
//    MemcpyDeviceToHost(hPtr, dPtr, SIZE_REAL * length);
//    
//#ifdef DOUBLE_PRECISION
//    printfVectorD(hPtr, length);
//#else
//    printfVectorF(hPtr,length);
//#endif
//    
//    free(hPtr);
}
//
void GPUInterface::PrintfDeviceInt(GPUPtr dPtr,
                                   int length) {    
//    int* hPtr = (int*) malloc(SIZE_INT * length);
//    
//    MemcpyDeviceToHost(hPtr, dPtr, SIZE_INT * length);
//    
//    printfInt(hPtr, length);
//    
//    free(hPtr);
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

