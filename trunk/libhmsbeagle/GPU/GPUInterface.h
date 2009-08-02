
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
    typedef cl_mem GPUPtr;
    typedef cl_kernel GPUFunction;
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
    cl_device_id openClDeviceId;             // compute device id 
    cl_context openClContext;                // compute context
    cl_command_queue openClCommandQueue;     // compute command queue
    cl_program openClProgram;                // compute program
    cl_uint openClNumDevices;
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
    
    void GetDeviceName(int deviceNumber,
                        char* deviceName,
                        int nameLength);

    void PrintfDeviceVector(GPUPtr dPtr,
                      int length);

    void PrintfDeviceInt(GPUPtr dPtr,
                   int length);
};

#endif // __GPUInterface__
