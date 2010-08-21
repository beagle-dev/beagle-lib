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
 * @author Dat Huynh
 * @author Daniel Ayres
 */

#ifndef __GPUInterface__
#define __GPUInterface__

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include <map>

#include "libhmsbeagle/GPU/GPUImplDefs.h"
#include "libhmsbeagle/GPU/KernelResource.h"

#ifdef CUDA
    #include <cuda.h>
    #include "libhmsbeagle/GPU/kernels/BeagleCUDA_kernels.h"
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
    
    int Initialize();

    int GetDeviceCount();

    void SetDevice(int deviceNumber, 
                   int paddedStateCount, 
                   int categoryCount, 
                   int patternCount,
                   long flags);
    
    void Synchronize();
    
    GPUFunction GetFunction(const char* functionName);
    
    void LaunchKernel(GPUFunction deviceFunction,
                               Dim3Int block,
                               Dim3Int grid,
                               int parameterCountV,
                               int totalParameterCount,
                               ...); // parameters

    void* AllocatePinnedHostMemory(int memSize,
                                   bool writeCombined,
                                   bool mapped);
    
    GPUPtr AllocateMemory(int memSize);
    
    GPUPtr AllocateRealMemory(int length);

    GPUPtr AllocateIntMemory(int length);
    
    void MemsetShort(GPUPtr dest,
                     unsigned short val,
                     unsigned int count);

    void MemcpyHostToDevice(GPUPtr dest,
                            const void* src,
                            int memSize);

    void MemcpyDeviceToHost(void* dest,
                            const GPUPtr src,
                            int memSize);
    
    void MemcpyDeviceToDevice(GPUPtr dest,
                              GPUPtr src,
                              int memSize);

    void FreePinnedHostMemory(void* hPtr);
    
    void FreeMemory(GPUPtr dPtr);
    
    GPUPtr GetDevicePointer(void* hPtr);
    
    unsigned int GetAvailableMemory();
    
    void GetDeviceName(int deviceNumber,
                       char* deviceName,
                       int nameLength);
    
    void GetDeviceDescription(int deviceNumber,
                              char* deviceDescription);
    
    void PrintfDeviceVector(GPUPtr dPtr,
                      int length);
    
    void PrintfDeviceVector(GPUPtr dPtr,
                           int length, double checkValue);
    
    void PrintfDeviceVector(GPUPtr dPtr,
                            int length,
                            double checkValue,
                            int *signal);

    void PrintfDeviceInt(GPUPtr dPtr,
                   int length);
    
    void DestroyKernelMap();
    
    KernelResource* kernelResource;
    
protected:
	void InitializeKernelMap();
    
    std::map<int, int>* resourceMap;
};

#endif // __GPUInterface__
