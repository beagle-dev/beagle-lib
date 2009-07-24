
/*
 * @author Marc Suchard
 * @author Dat Huynh
 * @author Daniel Ayres
 */

#ifndef __GPUInterface__
#define __GPUInterface__

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include "libhmsbeagle/GPU/GPUImplDefs.h"

#ifdef CUDA
    #include <cuda.h>
    #include "libhmsbeagle/GPU/BeagleCUDA_kernels.h"
    typedef CUdeviceptr GPUPtr;
    typedef CUfunction GPUFunction;
#endif

class GPUInterface {
private:
#ifdef CUDA
    CUdevice cudaDevice;
    CUcontext cudaContext;
    CUmodule cudaModule;
    const char* GetCUDAErrorDescription(int errorCode);
#endif
public:
    GPUInterface();
    
    ~GPUInterface();

    int GetDeviceCount();

    void SetDevice(int deviceNumber);
    
    void Synchronize();
    
    GPUFunction GetFunction(const char* functionName);
    
    void LaunchKernelIntParams(GPUFunction deviceFunction,
                               Dim3Int block,
                               Dim3Int grid,
                               int totalParameterCount,
                               ...); // unsigned int parameters
    
    GPUPtr AllocateMemory(int memSize);
    
    GPUPtr AllocateRealMemory(int length);

    GPUPtr AllocateIntMemory(int length);

    void MemcpyHostToDevice(GPUPtr dest,
                            const void* src,
                            int memSize);

    void MemcpyDeviceToHost(void* dest,
                            const GPUPtr src,
                            int memSize);

    void FreeMemory(GPUPtr dPtr);
    
    void PrintInfo();

    void PrintfDeviceVector(GPUPtr dPtr,
                      int length);

    void PrintfDeviceInt(GPUPtr dPtr,
                   int length);
};

#endif // __GPUInterface__
