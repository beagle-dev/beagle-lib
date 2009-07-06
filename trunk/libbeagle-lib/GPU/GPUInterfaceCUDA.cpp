/*
 * @author Marc Suchard
 * @author Daniel Ayres
 */

#ifdef HAVE_CONFIG_H
#include "libbeagle-lib/config.h"
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cstdarg>

#include <cuda.h>

#include "libbeagle-lib/GPU/GPUImplDefs.h"
#include "libbeagle-lib/GPU/GPUImplHelper.h"
#include "libbeagle-lib/GPU/GPUInterface.h"

// TODO: print cuda error message instead of code
#define SAFE_CUDA(call) { \
                            CUresult error = call; \
                            if(error != CUDA_SUCCESS) { \
                                fprintf(stderr, "CUDA error %d\n", error); \
                                exit(-1); \
                            } \
                        }

#define SAFE_CUPP(call) { \
                            SAFE_CUDA(cuCtxPushCurrent(cudaContext)); \
                            SAFE_CUDA(call); \
                            SAFE_CUDA(cuCtxPopCurrent(&cudaContext)); \
                        }

GPUInterface::GPUInterface() {
    // Driver init; CUDA manual: "Currently, the Flags parameter must be 0."
    SAFE_CUDA(cuInit(0));
    
    cudaDevice = NULL;
    cudaContext = NULL;
    cudaModule = NULL;
}

GPUInterface::~GPUInterface() {
    SAFE_CUDA(cuCtxPushCurrent(cudaContext));
    SAFE_CUDA(cuCtxDetach(cudaContext));
}

int GPUInterface::GetDeviceCount() {
    int numDevices = 0;
    SAFE_CUDA(cuDeviceGetCount(&numDevices));
    return numDevices;
}

void GPUInterface::SetDevice(int deviceNumber) {
    SAFE_CUDA(cuDeviceGet(&cudaDevice, deviceNumber));
    
    SAFE_CUDA(cuCtxCreate(&cudaContext, CU_CTX_SCHED_AUTO, cudaDevice));
    
    SAFE_CUDA(cuModuleLoadData(&cudaModule, KERNELS_STRING)); 
    
    SAFE_CUDA(cuCtxPopCurrent(&cudaContext));
}

void GPUInterface::Synchronize() {
    SAFE_CUPP(cuCtxSynchronize());
}

GPUFunction GPUInterface::GetFunction(char* functionName) {
    GPUFunction cudaFunction; 
    
    SAFE_CUPP(cuModuleGetFunction(&cudaFunction, cudaModule, functionName));
    
    return cudaFunction;
}

void GPUInterface::LaunchKernelIntParams(GPUFunction deviceFunction,
                                         Dim3Int block,
                                         Dim3Int grid,
                                         int totalParameterCount,
                                         ...) { // unsigned int parameters
    SAFE_CUDA(cuCtxPushCurrent(cudaContext));
    
    SAFE_CUDA(cuFuncSetBlockShape(deviceFunction, block.x, block.y, block.z));
    
    int offsetSize = sizeof(int);
    va_list parameters;
    va_start(parameters, totalParameterCount);  
    for(int i = 0; i < totalParameterCount; i++)
        SAFE_CUDA(cuParamSeti(deviceFunction, offsetSize * i, va_arg(parameters, unsigned int)));
    va_end(parameters);
    
    SAFE_CUDA(cuParamSetSize(deviceFunction, offsetSize * totalParameterCount));
    
    SAFE_CUDA(cuLaunchGrid(deviceFunction, grid.x, grid.y));
    
    SAFE_CUDA(cuCtxPopCurrent(&cudaContext));
}


GPUPtr GPUInterface::AllocateMemory(int memSize) {
#ifdef DEBUG
    fprintf(stderr,"Entering ANMA\n");
#endif
    
    GPUPtr data;
    
    SAFE_CUPP(cuMemAlloc(&data, memSize));
    
#ifdef DEBUG
    fprintf(stderr, "Allocated %d to %d.\n", data, (data + memSize));
    fprintf(stderr, "Leaving ANMA\n");
#endif
    
    return data;
}

GPUPtr GPUInterface::AllocateRealMemory(int length) {
#ifdef DEBUG
    fprintf(stderr,"Entering ANMA-Real\n");
#endif

    GPUPtr data;
    
    SAFE_CUPP(cuMemAlloc(&data, SIZE_REAL * length));

#ifdef DEBUG
    fprintf(stderr, "Allocated %d to %d.\n", data, (data + length));
    fprintf(stderr, "Leaving ANMA\n");
#endif
    
    return data;
}

GPUPtr GPUInterface::AllocateIntMemory(int length) {
#ifdef DEBUG
    fprintf(stderr, "Entering ANMA-Int\n");
#endif

    GPUPtr data;
    
    SAFE_CUPP(cuMemAlloc(&data, SIZE_INT * length));

#ifdef DEBUG
    fprintf(stderr, "Allocated %d to %d.\n", data, (data + length));
    fprintf(stderr, "Leaving ANMA\n");
#endif

    return data;
}

void GPUInterface::MemcpyHostToDevice(GPUPtr dest,
                                      const void* src,
                                      int memSize) {
    SAFE_CUPP(cuMemcpyHtoD(dest, src, memSize));
}

void GPUInterface::MemcpyDeviceToHost(void* dest,
                                      const GPUPtr src,
                                      int memSize) {
    SAFE_CUPP(cuMemcpyDtoH(dest, src, memSize));
}

void GPUInterface::FreeMemory(GPUPtr dPtr) {
#ifdef DEBUG
    fprintf(stderr, "Entering FNMA\n");
#endif
    
    SAFE_CUPP(cuMemFree(dPtr));

#ifdef DEBUG
    fprintf(stderr,"Leaving FNMA\n");
#endif
}

void GPUInterface::PrintInfo() {    
    fprintf(stderr, "GPU Device Information:");
    
    char name[100];
    unsigned int totalGlobalMemory = 0;
    int clockSpeed = 0;
    int mpCount;
    
    SAFE_CUDA(cuDeviceGetName(name, 100, cudaDevice));
    SAFE_CUDA(cuDeviceTotalMem(&totalGlobalMemory, cudaDevice));
    SAFE_CUDA(cuDeviceGetAttribute(&clockSpeed, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, cudaDevice));
    SAFE_CUDA(cuDeviceGetAttribute(&mpCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cudaDevice));
    
    fprintf(stderr, "\nDevice #%d: %s\n", cudaDevice, name);
    fprintf(stderr, "\tGlobal Memory (MB) : %1.2f\n", totalGlobalMemory / 1024.0 / 1024.0);
    fprintf(stderr, "\tClock Speed (Ghz)  : %1.2f\n", clockSpeed / 1000000.0);
    fprintf(stderr, "\tNumber of Cores    : %d\n", 8 * mpCount);
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
