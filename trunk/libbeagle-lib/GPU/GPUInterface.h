
/*
 * @author Marc Suchard
 * @author Dat Huynh
 * @author Daniel Ayres
 */

#ifndef __GPUInterface__
#define __GPUInterface__

#ifdef HAVE_CONFIG_H
#include "libbeagle-lib/config.h"
#endif

#include "libbeagle-lib/GPU/GPUImplDefs.h"

#ifdef CUDA
    #include <cuda.h>
    #define KERNELS_FILE "BeagleCUDA_kernels.cubin"
    typedef CUdeviceptr GPUPtr;
    typedef CUfunction GPUFunction;
#endif

class GPUInterface {
private:
#ifdef CUDA
    CUdevice cudaDevice;
    CUcontext cudaContext;
    CUmodule cudaModule;
#endif
public:
    GPUInterface();
    
    ~GPUInterface();

    int GetDeviceCount();

    void SetDevice(int deviceNumber);
    
    void Synchronize();
    
    GPUFunction GetFunction(char* functionName);
    
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
