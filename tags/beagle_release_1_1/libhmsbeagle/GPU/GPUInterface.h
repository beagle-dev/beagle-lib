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

#include <cstdio>

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include <map>

#include "libhmsbeagle/GPU/GPUImplDefs.h"
#include "libhmsbeagle/GPU/KernelResource.h"

#ifdef CUDA
    #include <cuda.h>
#   ifdef BEAGLE_XCODE
        #include "libhmsbeagle/GPU/kernels/BeagleCUDA_kernels_xcode.h"
#   else
        #include "libhmsbeagle/GPU/kernels/BeagleCUDA_kernels.h"
#   endif
    typedef CUdeviceptr GPUPtr;
    typedef CUfunction GPUFunction;
#else
#ifdef OPENCL
    #include <OpenCL/opencl.h>
#   ifdef BEAGLE_XCODE
        #include "libhmsbeagle/GPU/kernels/BeagleOpenCL_kernels_xcode.h"
#   else
        #include "libhmsbeagle/GPU/kernels/BeagleOpenCL_kernels.h"
#   endif
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

    void* MallocHost(size_t memSize);
    
    void* CallocHost(size_t size, size_t length);
    
    void* AllocatePinnedHostMemory(size_t memSize,
                                   bool writeCombined,
                                   bool mapped);
    
    GPUPtr AllocateMemory(size_t memSize);
    
    GPUPtr AllocateRealMemory(size_t length);

    GPUPtr AllocateIntMemory(size_t length);
    
    void MemsetShort(GPUPtr dest,
                     unsigned short val,
                     size_t count);

    void MemcpyHostToDevice(GPUPtr dest,
                            const void* src,
                            size_t memSize);

    void MemcpyDeviceToHost(void* dest,
                            const GPUPtr src,
                            size_t memSize);
    
    void MemcpyDeviceToDevice(GPUPtr dest,
                              GPUPtr src,
                              size_t memSize);

    void FreeHostMemory(void* hPtr);
    
    void FreePinnedHostMemory(void* hPtr);
    
    void FreeMemory(GPUPtr dPtr);
    
    GPUPtr GetDevicePointer(void* hPtr);
    
    unsigned int GetAvailableMemory();
    
    void GetDeviceName(int deviceNumber,
                       char* deviceName,
                       int nameLength);
    
    void GetDeviceDescription(int deviceNumber,
                              char* deviceDescription);
    
    bool GetSupportsDoublePrecision(int deviceNumber);

    template<typename Real>
    void PrintfDeviceVector(GPUPtr dPtr, int length, Real r) {
    	PrintfDeviceVector(dPtr,length,-1, 0, r);
    }
    
    template<typename Real>
    void PrintfDeviceVector(GPUPtr dPtr,
                            int length, double checkValue, Real r);
    
    template<typename Real>
    void PrintfDeviceVector(GPUPtr dPtr,
                            int length,
                            double checkValue,
                            int *signal,
                            Real r) {
    	Real* hPtr = (Real*) malloc(sizeof(Real) * length);

        MemcpyDeviceToHost(hPtr, dPtr, sizeof(Real) * length);
    	printfVector(hPtr, length);

        if (checkValue != -1) {
        	double sum = 0;
        	for(int i=0; i<length; i++) {
        		sum += hPtr[i];
        		if( (hPtr[i] > checkValue) && (hPtr[i]-checkValue > 1.0E-4)) {
        			fprintf(stderr,"Check value exception!  (%d) %2.5e > %2.5e (diff = %2.5e)\n",
        					i,hPtr[i],checkValue, (hPtr[i]-checkValue));
        			if( signal != 0 )
        				*signal = 1;
        		}
        		if (hPtr[i] != hPtr[i]) {
        			fprintf(stderr,"NaN found!\n");
        			if( signal != 0 )
        				*signal = 1;
        		}
        	}
        	if (sum == 0) {
        		fprintf(stderr,"Zero-sum vector!\n");
        		if( signal != 0 )
        			*signal = 1;
        	}
        }
        free(hPtr);
    }

    void PrintfDeviceInt(GPUPtr dPtr,
                   int length);
    
    void DestroyKernelMap();
    
    KernelResource* kernelResource;
    
protected:
	void InitializeKernelMap();
    
    std::map<int, int>* resourceMap;

    bool supportDoublePrecision;
};

#endif // __GPUInterface__
