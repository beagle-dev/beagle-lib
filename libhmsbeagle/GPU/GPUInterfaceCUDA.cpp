/*
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

#include <cuda.h>

#include "libhmsbeagle/GPU/GPUImplDefs.h"
#include "libhmsbeagle/GPU/GPUImplHelper.h"
#include "libhmsbeagle/GPU/GPUInterface.h"

#define SAFE_CUDA(call) { \
                            CUresult error = call; \
                            if(error != CUDA_SUCCESS) { \
                                fprintf(stderr, "CUDA error: \"%s\" from file <%s>, line %i.\n", \
                                        GetCUDAErrorDescription(error), __FILE__, __LINE__); \
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
    
    cudaDevice = NULL;
    cudaContext = NULL;
    cudaModule = NULL;
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::GPUInterface\n");
#endif    
}

GPUInterface::~GPUInterface() {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::~GPUInterface\n");
#endif    
    
    if (cudaContext != NULL) {
        SAFE_CUDA(cuCtxPushCurrent(cudaContext));
        SAFE_CUDA(cuCtxDetach(cudaContext));
    }
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::~GPUInterface\n");
#endif    
    
}

int GPUInterface::Initialize() {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::Initialize\n");
#endif    
    
    // Driver init; CUDA manual: "Currently, the Flags parameter must be 0."
    CUresult error = cuInit(0);
    
    int returnValue = 1;
    
    if (error == CUDA_ERROR_NO_DEVICE) {
        returnValue = 0;
    } else if (error != CUDA_SUCCESS) {
        fprintf(stderr, "CUDA error: \"%s\" from file <%s>, line %i.\n",
                GetCUDAErrorDescription(error), __FILE__, __LINE__);
        exit(-1);
    }
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::Initialize\n");
#endif    
    
    return returnValue;
}

int GPUInterface::GetDeviceCount() {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::GetDeviceCount\n");
#endif        
    
    int numDevices = 0;
    SAFE_CUDA(cuDeviceGetCount(&numDevices));
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::GetDeviceCount\n");
#endif            
    
    return numDevices;
}

void GPUInterface::SetDevice(int deviceNumber) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::SetDevice\n");
#endif            
    
    SAFE_CUDA(cuDeviceGet(&cudaDevice, deviceNumber));
    
    SAFE_CUDA(cuCtxCreate(&cudaContext, CU_CTX_SCHED_AUTO, cudaDevice));
    
    SAFE_CUDA(cuModuleLoadData(&cudaModule, KERNELS_STRING)); 
    
    SAFE_CUDA(cuCtxPopCurrent(&cudaContext));
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::SetDevice\n");
#endif            
    
}

void GPUInterface::Synchronize() {
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::Synchronize\n");
#endif                
    
    SAFE_CUPP(cuCtxSynchronize());
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::Synchronize\n");
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

void GPUInterface::LaunchKernelIntParams(GPUFunction deviceFunction,
                                         Dim3Int block,
                                         Dim3Int grid,
                                         int totalParameterCount,
                                         ...) { // unsigned int parameters
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::LaunchKernelIntParams\n");
#endif                
    
    
    SAFE_CUDA(cuCtxPushCurrent(cudaContext));
    
    SAFE_CUDA(cuFuncSetBlockShape(deviceFunction, block.x, block.y, block.z));
    
    int offset = 0;
    va_list parameters;
    va_start(parameters, totalParameterCount);  
    for(int i = 0; i < totalParameterCount; i++) {
        unsigned int param = va_arg(parameters, unsigned int);
        
         // adjust offset alignment requirements
        offset = (offset + __alignof(param) - 1) & ~(__alignof(param) - 1);

        SAFE_CUDA(cuParamSeti(deviceFunction, offset, param));
        
        offset += sizeof(param);
    }
    va_end(parameters);
    
    SAFE_CUDA(cuParamSetSize(deviceFunction, offset));
    
    SAFE_CUDA(cuLaunchGrid(deviceFunction, grid.x, grid.y));
    
    SAFE_CUDA(cuCtxPopCurrent(&cudaContext));
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::LaunchKernelIntParams\n");
#endif                
    
}


GPUPtr GPUInterface::AllocateMemory(int memSize) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::AllocateMemory\n");
#endif
    
    GPUPtr data;
    
    SAFE_CUPP(cuMemAlloc(&data, memSize));

#ifdef BEAGLE_DEBUG_VALUES
    fprintf(stderr, "Allocated GPU memory %d to %d.\n", data, (data + memSize));
#endif
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::AllocateMemory\n");
#endif
    
    return data;
}

GPUPtr GPUInterface::AllocateRealMemory(int length) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::AllocateRealMemory\n");
#endif

    GPUPtr data;

    SAFE_CUPP(cuMemAlloc(&data, SIZE_REAL * length));

#ifdef BEAGLE_DEBUG_VALUES
    fprintf(stderr, "Allocated GPU memory %d to %d.\n", data, (data + length));
#endif
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::AllocateRealMemory\n");
#endif
    
    return data;
}

GPUPtr GPUInterface::AllocateIntMemory(int length) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::AllocateIntMemory\n");
#endif

    GPUPtr data;
    
    SAFE_CUPP(cuMemAlloc(&data, SIZE_INT * length));

#ifdef BEAGLE_DEBUG_VALUES
    fprintf(stderr, "Allocated GPU memory %d to %d.\n", data, (data + length));
#endif
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::AllocateIntMemory\n");
#endif

    return data;
}

void GPUInterface::MemcpyHostToDevice(GPUPtr dest,
                                      const void* src,
                                      int memSize) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::MemcpyHostToDevice\n");
#endif    
    
    SAFE_CUPP(cuMemcpyHtoD(dest, src, memSize));
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::MemcpyHostToDevice\n");
#endif    
    
}

void GPUInterface::MemcpyDeviceToHost(void* dest,
                                      const GPUPtr src,
                                      int memSize) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::MemcpyDeviceToHost\n");
#endif        
    
    SAFE_CUPP(cuMemcpyDtoH(dest, src, memSize));
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::MemcpyDeviceToHost\n");
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

void GPUInterface::GetDeviceName(int deviceNumber,
                                  char* deviceName,
                                  int nameLength) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::GetDeviceName\n");
#endif    
    
    CUdevice tmpCudaDevice;

    SAFE_CUDA(cuDeviceGet(&tmpCudaDevice, deviceNumber));
    
    SAFE_CUDA(cuDeviceGetName(deviceName, nameLength, tmpCudaDevice));
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::GetDeviceName\n");
#endif        
}

void GPUInterface::GetDeviceDescription(int deviceNumber,
                                        char* deviceDescription) {    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::GetDeviceDescription\n");
#endif
    
    CUdevice tmpCudaDevice;
    
    SAFE_CUDA(cuDeviceGet(&tmpCudaDevice, deviceNumber));
    
    unsigned int totalGlobalMemory = 0;
    int clockSpeed = 0;
    int mpCount = 0;
    
    SAFE_CUDA(cuDeviceTotalMem(&totalGlobalMemory, cudaDevice));
    SAFE_CUDA(cuDeviceGetAttribute(&clockSpeed, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, cudaDevice));
    SAFE_CUDA(cuDeviceGetAttribute(&mpCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cudaDevice));
    
    sprintf(deviceDescription,
            "Global memory (MB): %d | Clock speed (Ghz): %1.2f | Number of cores: %d",
            int(totalGlobalMemory / 1024.0 / 1024.0 + 0.5),
            clockSpeed / 1000000.0,
            8 * mpCount);
    
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::GetDeviceDescription\n");
#endif    
}

void GPUInterface::PrintfDeviceVector(GPUPtr dPtr,
                                int length) {
    REAL* hPtr = (REAL*) malloc(SIZE_REAL * length);
    
    MemcpyDeviceToHost(hPtr, dPtr, SIZE_REAL * length);
    
#ifdef DOUBLE_PRECISION
    printfVectorD(hPtr, length);
#else
    printfVectorF(hPtr,length);
#endif
    
    free(hPtr);
}

void GPUInterface::PrintfDeviceInt(GPUPtr dPtr,
                             int length) {    
    int* hPtr = (int*) malloc(SIZE_INT * length);
    
    MemcpyDeviceToHost(hPtr, dPtr, SIZE_INT * length);
    
    printfInt(hPtr, length);
    
    free(hPtr);
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

