/*
 *
 * Copyright 2009 Phylogenetic Likelihood Working Group
 *
 * This file is part of BEAGLE.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
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

#include "libhmsbeagle/GPU/GPUImplHelper.h"
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

    namespace cuda_device {
#elif defined(FW_OPENCL)
    #define CL_USE_DEPRECATED_OPENCL_1_1_APIS // to disable deprecation warnings
    #define CL_USE_DEPRECATED_OPENCL_1_2_APIS // to disable deprecation warnings
    #define CL_USE_DEPRECATED_OPENCL_2_0_APIS // to disable deprecation warnings
#   ifdef DLS_MACOS
        #include <OpenCL/opencl.h>
#   else
        #include <CL/opencl.h>
#   endif
#   ifdef BEAGLE_XCODE
        #include "libhmsbeagle/GPU/kernels/BeagleOpenCL_kernels_xcode.h"
#   else
        #include "libhmsbeagle/GPU/kernels/BeagleOpenCL_kernels.h"
#   endif
    typedef cl_mem GPUPtr;
    typedef cl_kernel GPUFunction;

    namespace opencl_device {
#elif defined(FW_TINYGPU)
    // Only the stamp of the TinyGPU-specific PTX (kernels/BeagleTinyGPU_kernels.h, compiled with -DCUDA -DFW_TINYGPU by
    // kernels/make_tinygpu_kernels.sh): the NV path runs the cubins compiled from it at build time, the AMD path its
    // OpenCL source (TODO.md plan step C13, decision 15).
#   ifdef BEAGLE_XCODE
        #include "libhmsbeagle/GPU/kernels/BeagleCUDA_kernels_xcode.h"
#   else
        #include "libhmsbeagle/GPU/kernels/BeagleTinyGPU_kernels_stamp.h"
#   endif
    // GPUPtr: byte offset into VRAM (BAR2 offset used in MMIO_READ/WRITE).
    typedef uint64_t GPUPtr;
    // GPUFunction: opaque pointer to an NVKernelEntry (resolved at launch).
    typedef void*    GPUFunction;

    namespace tinygpu_device {
#endif

class GPUInterface {
private:
    int numStreams;
#ifdef CUDA
    CUdevice cudaDevice;
    CUcontext cudaContext;
    CUmodule cudaModule;
    CUstream* cudaStreams;
    CUevent cudaEvent;
    const char* GetCUDAErrorDescription(int errorCode);
#elif defined(FW_OPENCL)
    cl_device_id openClDeviceId;             // compute device id
    cl_context openClContext;                // compute context
    cl_command_queue* openClCommandQueues;   // compute command queue
    cl_event* openClEvents;                  // compute events
    cl_program openClProgram;                // compute program
    std::map<int, cl_device_id> openClDeviceMap;
    const char* GetCLErrorDescription(int errorCode);
#elif defined(FW_TINYGPU)
    // AMD dispatch (GPUInterfaceTinyGPUHybridAMD.cpp) lives in free functions,
    // not GPUInterface methods -- GPUInterface::SetDevice/LaunchKernelImpl/etc.
    // (defined once, in GPUInterfaceTinyGPUHybrid.cpp) branch on isNVIDIA and
    // call out to them. They need private/protected member access the same
    // way GPUInterfaceTinyGPUHybrid.cpp's own method bodies already do.
    friend void AmdSetDevice(GPUInterface*, int, int, int, int, int, long);
    // Their per-instance state is amdInstance (TODO.md plan step A5)
    friend int AmdAttachShared(GPUInterface*);
    friend GPUFunction AmdGetFunction(GPUInterface*, const char*);
    friend void AmdLaunchKernelImpl(GPUInterface*, GPUFunction, Dim3Int, Dim3Int, int, int, GPUPtr*, unsigned int*);
    friend void AmdSynchronizeHost(GPUInterface*);
    friend void AmdMemcpyHostToDevice(GPUInterface*, GPUPtr, const void*, size_t);
    friend void AmdMemcpyDeviceToHost(GPUInterface*, void*, const GPUPtr, size_t);
    friend void AmdFini(GPUInterface*);
    friend GPUPtr AmdAllocateMemory(GPUInterface*, size_t);
    friend bool AmdDeviceLost(GPUInterface*);
    friend bool AmdOutOfMemory(GPUInterface*);
    // Same for the NV path (GPUInterfaceTinyGPUHybridNV.cpp: the daemon by
    // default, C++ dispatch or the C++ runtime by environment variable), whose
    // per-instance state is nvGspState (TODO.md plan step P5).
    friend void NvSetDevice(GPUInterface*, int, int, int, int, int, long);
    friend int NvAttachShared(GPUInterface*);
    friend GPUFunction NvGetFunction(GPUInterface*, const char*);
    friend void NvLaunchKernelImpl(GPUInterface*, GPUFunction, Dim3Int, Dim3Int, int, int, GPUPtr*, unsigned int*);
    friend void NvSynchronizeHost(GPUInterface*);
    friend void NvMemcpyHostToDevice(GPUInterface*, GPUPtr, const void*, size_t);
    friend void NvMemcpyDeviceToHost(GPUInterface*, void*, const GPUPtr, size_t);
    friend void NvFini(GPUInterface*);
    friend GPUPtr NvAllocateMemory(GPUInterface*, size_t);
    friend bool NvDeviceLost(GPUInterface*);
    friend bool NvOutOfMemory(GPUInterface*);

    // ── TinyGPU socket ──────────────────────────────────────────────────────
    int      tgpuSock;
    uint32_t tgpuDevId;
    bool     isNVIDIA;    // true = NV (CUDA driver path), false = AMD (PM4 path)

    // The NV path's per-instance state: its NVInstance (GPUInterfaceTinyGPUHybridNV.cpp; TODO.md plan step P5)
    void*    nvGspState = nullptr;
    // The AMD path's: its AMDInstance (GPUInterfaceTinyGPUHybridAMD.cpp; TODO.md plan step A5)
    void*    amdInstance = nullptr;

    void LaunchKernelImpl(GPUFunction fn, Dim3Int block, Dim3Int grid,
                          int nPtr, int nTotal, GPUPtr* ptrs, unsigned int* ints);
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
                   int unpaddedPatternCount,
                   int tipCount,
                   long flags);

    void ResizeStreamCount(int newStreamCount);

    void SynchronizeHost();
    void SynchronizeDevice();
    void SynchronizeDeviceWithIndex(int streamRecordIndex,
                                    int streamWaitIndex);

    GPUFunction GetFunction(const char* functionName);

    void LaunchKernel(GPUFunction deviceFunction,
                               Dim3Int block,
                               Dim3Int grid,
                               int parameterCountV,
                               int totalParameterCount,
                               ...); // parameters

    void LaunchKernelConcurrent(GPUFunction deviceFunction,
                               Dim3Int block,
                               Dim3Int grid,
                               int streamIndex,
                               int waitIndex,
                               int parameterCountV,
                               int totalParameterCount,
                               ...); // parameters

    void* MallocHost(size_t memSize);

    void* CallocHost(size_t size, size_t length);

    void* AllocatePinnedHostMemory(size_t memSize,
                                   bool writeCombined,
                                   bool mapped);

#ifdef FW_TINYGPU
    // TODO.md plan step C12: true once this instance's GPU work cannot be trusted (its setup or an allocation failed, or the
    // GPU hung or its connection broke): its calls then do nothing, instead of exiting the host, and BeagleGPUImpl returns
    // errors from createInstance and from the calls that copy results back
    bool GetDeviceLost();
    // TODO.md plan step M1: ... and the reason was memory (an allocation, a program load or the VRAM pool did not fit), so
    // BeagleGPUImpl returns BEAGLE_ERROR_OUT_OF_MEMORY rather than BEAGLE_ERROR_GENERAL
    bool GetOutOfMemory();
#endif

#ifdef FW_OPENCL
    void* MapMemory(GPUPtr dPtr,
                    size_t memSize);

    void UnmapMemory(GPUPtr dPtr,
                       void* hPtr);
#endif

    GPUPtr AllocateMemory(size_t memSize);

    GPUPtr AllocateRealMemory(size_t length);

    GPUPtr AllocateIntMemory(size_t length);

    GPUPtr CreateSubPointer(GPUPtr dPtr, size_t offset, size_t size);

    size_t AlignMemOffset(size_t offset);

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

    GPUPtr GetDeviceHostPointer(void* hPtr);

    size_t GetAvailableMemory();

    void GetDeviceName(int deviceNumber,
                       char* deviceName,
                       int nameLength);

    void GetDeviceDescription(int deviceNumber,
                              char* deviceDescription);

    long GetDeviceTypeFlag(int deviceNumber);

    BeagleDeviceImplementationCodes GetDeviceImplementationCode(int deviceNumber);

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

    KernelResource* kernelResource;

#ifdef BEAGLE_DEBUG_OPENCL_CORES
    void CreateDevice(int deviceNumber);

    void ReleaseDevice(int deviceNumber);
#endif

protected:
	void InitializeKernelResource(int paddedStateCount,
                                  bool doublePrecision);

    std::map<int, int>* resourceMap;

    bool supportDoublePrecision;
};

}; // namespace

#endif // __GPUInterface__
