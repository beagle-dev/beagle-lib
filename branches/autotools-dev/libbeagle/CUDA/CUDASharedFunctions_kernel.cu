/*
 * @author Marc Suchard
 */
#ifndef _Included_SharedFunctionsKernel
#define _Included_SharedFunctionsKernel

/**************INCLUDES***********/
#include <stdio.h>
#include <string.h>
#include <cuda_runtime_api.h>
#include "CUDASharedFunctions.h"

/**************CODE***********/
#ifdef __cplusplus
extern "C" {
#endif

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg,
                             cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}

void getGPUInfo(int iDevice, char *name, int *memory, int *speed) {
	cudaDeviceProp deviceProp;
	memset(&deviceProp, 0, sizeof(deviceProp));
	cudaGetDeviceProperties(&deviceProp, iDevice);
	*memory = deviceProp.totalGlobalMem;
	*speed = deviceProp.clockRate;
	strcpy(name, deviceProp.name);
}

#ifdef __cplusplus
}
#endif
#endif

