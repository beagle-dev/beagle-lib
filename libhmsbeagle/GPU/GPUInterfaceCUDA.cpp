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
#include <map>

#include <cuda.h>

#include "libhmsbeagle/GPU/GPUImplDefs.h"
#include "libhmsbeagle/GPU/GPUImplHelper.h"
#include "libhmsbeagle/GPU/GPUInterface.h"
#include "libhmsbeagle/GPU/KernelResource.h"


std::map<int, KernelResource>* kernelMap = NULL;


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
    kernelResource = NULL;
    
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
    
    if (kernelResource != NULL) {
        delete kernelResource;
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

void GPUInterface::DestroyKernelMap() {
    if (kernelMap) {
        delete kernelMap;
    }
}

void GPUInterface::InitializeKernelMap() {

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLoading kernel information for CUDA!\n");
#endif

    kernelMap = new std::map<int, KernelResource>;
    
    KernelResource kernel4 = KernelResource(
        4,
        (char*) KERNELS_STRING_4,
        PATTERN_BLOCK_SIZE_4,
        MATRIX_BLOCK_SIZE_4,
        BLOCK_PEELING_SIZE_4,
        SLOW_REWEIGHING_4,
        MULTIPLY_BLOCK_SIZE,
        0,0,0);
    kernelMap->insert(std::make_pair(4,kernel4));
    KernelResource kernel4LS = KernelResource(kernel4, (char*) KERNELS_STRING_LS_4);
    kernelMap->insert(std::make_pair(-4,kernel4LS));
    
    KernelResource kernel32 = KernelResource(
        32,
        (char*) KERNELS_STRING_32,
        PATTERN_BLOCK_SIZE_32,
        MATRIX_BLOCK_SIZE_32,
        BLOCK_PEELING_SIZE_32,
        SLOW_REWEIGHING_32,
        MULTIPLY_BLOCK_SIZE,
        0,0,0);
    kernelMap->insert(std::make_pair(32,kernel32));
    KernelResource kernel32LS = KernelResource(kernel32, (char*) KERNELS_STRING_LS_32);
    kernelMap->insert(std::make_pair(-32,kernel32LS));
    
    KernelResource kernel48 = KernelResource(
        48,
        (char*) KERNELS_STRING_48,
        PATTERN_BLOCK_SIZE_48,
        MATRIX_BLOCK_SIZE_48,
        BLOCK_PEELING_SIZE_48,
        SLOW_REWEIGHING_48,
        MULTIPLY_BLOCK_SIZE,
        0,0,0);
    kernelMap->insert(std::make_pair(48,kernel48));
    KernelResource kernel48LS = KernelResource(kernel48, (char*) KERNELS_STRING_LS_48);
    kernelMap->insert(std::make_pair(-48,kernel48LS));
    
    KernelResource kernel64 = KernelResource(
        64,
        (char*) KERNELS_STRING_64,
        PATTERN_BLOCK_SIZE_64,
        MATRIX_BLOCK_SIZE_64,
        BLOCK_PEELING_SIZE_64,
        SLOW_REWEIGHING_64,
        MULTIPLY_BLOCK_SIZE,
        0,0,0);
    kernelMap->insert(std::make_pair(64,kernel64));
    KernelResource kernel64LS = KernelResource(kernel64, (char*) KERNELS_STRING_LS_64);
    kernelMap->insert(std::make_pair(-64,kernel64LS));
}

void GPUInterface::SetDevice(int deviceNumber, int paddedStateCount, int categoryCount, int paddedPatternCount,
                             long flags) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::SetDevice\n");
#endif            
    
    SAFE_CUDA(cuDeviceGet(&cudaDevice, deviceNumber));
    
    SAFE_CUDA(cuCtxCreate(&cudaContext, CU_CTX_SCHED_AUTO, cudaDevice));
    
    if (kernelMap == NULL) {
        // kernels have not yet been initialized; do so now.  Hopefully, this only occurs once per library load.
        InitializeKernelMap();
    }
    
    if (kernelMap->count(paddedStateCount) == 0) {
    	fprintf(stderr,"Critical error: unable to find kernel code for %d states.\n",paddedStateCount);
    	exit(-1);
    }
    
//    kernel.paddedStateCount = paddedStateCount;
//    kernel.kernelCode = kernelMap[paddedStateCount].kernelCode;
//    kernel.patternBlockSize = kernelMap[paddedStateCount].patternBlockSize;
//    kernel.matrixBlockSize = kernelMap[paddedStateCount].matrixBlockSize;
//    kernel.blockPeelingSize = kernelMap[paddedStateCount].blockPeelingSize;
//    kernel.slowReweighing = kernelMap[paddedStateCount].slowReweighing;
//    kernel.multiplyBlockSize = kernelMap[paddedStateCount].multiplyBlockSize;
    kernelResource = (*kernelMap)[paddedStateCount].copy();
    kernelResource->categoryCount = categoryCount;
    kernelResource->patternCount = paddedPatternCount;
    kernelResource->flags = flags;
                
    SAFE_CUDA(cuModuleLoadData(&cudaModule, kernelResource->kernelCode));
    
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

unsigned int GPUInterface::GetAvailableMemory() {
    unsigned int availableMem = 0;
    SAFE_CUPP(cuMemGetInfo(&availableMem, NULL));
    return availableMem;
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
    
    SAFE_CUDA(cuDeviceTotalMem(&totalGlobalMemory, tmpCudaDevice));
    SAFE_CUDA(cuDeviceGetAttribute(&clockSpeed, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, tmpCudaDevice));
    SAFE_CUDA(cuDeviceGetAttribute(&mpCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, tmpCudaDevice));
    
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
                                int length, double checkValue, int *signal) {
    REAL* hPtr = (REAL*) malloc(SIZE_REAL * length);
    
    MemcpyDeviceToHost(hPtr, dPtr, SIZE_REAL * length);
    
#ifdef DOUBLE_PRECISION
    printfVectorD(hPtr, length);
#else
    printfVectorF(hPtr,length);
#endif
    
    if (checkValue != -1) {
    	double sum = 0;
    	for(int i=0; i<length; i++) {
    		sum += hPtr[i];
    		if( (hPtr[i] > checkValue) && (hPtr[i]-checkValue > 1.0E-4)) {
    			fprintf(stderr,"Check value exception!  (%d) %2.5e > %2.5e (diff = %2.5e)\n",
    					i,hPtr[i],checkValue, (hPtr[i]-checkValue));
    			if( signal != 0 )
    				*signal = 1;
//    			exit(0);    			
    		}
    		if (hPtr[i] != hPtr[i]) {
    			fprintf(stderr,"NaN found!\n");
    			//exit(0);
    			if( signal != 0 ) 
    				*signal = 1;
    		}
    	}
    	if (sum == 0) {
    		fprintf(stderr,"Zero-sum vector!\n");
//    		exit(0);
    		if( signal != 0 )
    			*signal = 1;
    	}
    	
    }
    
    free(hPtr);
}

void GPUInterface::PrintfDeviceVector(GPUPtr dPtr, int length) {
	PrintfDeviceVector(dPtr,length,-1, 0);
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

