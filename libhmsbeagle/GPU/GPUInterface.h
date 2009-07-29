
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
#else
#ifdef OPENCL
    #include <OpenCL/opencl.h>
    #include "libhmsbeagle/GPU/BeagleOpenCL_kernels.h"
    #define MAX_CL_DEVICES 10
    typedef int GPUPtr;
    typedef int GPUFunction;
#endif
#endif

class GPUInterface {
private:
#ifdef CUDA
    CUdevice cudaDevice;
    CUcontext cudaContext;
    CUmodule cudaModule;
    const char* GetCUDAErrorDescription(int errorCode);
#else
#ifdef OPENCL
    cl_device_id clDeviceId;             // compute device id 
    cl_context clContext;                // compute context
    cl_command_queue clCommandQueue;     // compute command queue
    cl_program clProgram;                // compute program
    const char* GetCLErrorDescription(int errorCode);
#endif
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
