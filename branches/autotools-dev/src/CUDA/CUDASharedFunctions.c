/*
 * @author Marc Suchard
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "CUDASharedFunctions.h"

#ifdef __cplusplus
extern "C" {
#endif


REAL *allocateGPURealMemory(int length) {
#ifdef DEBUG
	fprintf(stderr,"Entering ANMA-Real\n");
#endif

	REAL *data;
	SAFE_CUDA(cudaMalloc((void**) &data, SIZE_REAL * length),data);
	if (data == NULL) {
		fprintf(stderr,"Failed to allocate REAL (%d) memory on device!\n",
				length);
		// TODO clean up and gracefully die
		exit(-1);
	}

#ifdef DEBUG
	fprintf(stderr,"Allocated %d to %d.\n",data,(data +length));
	fprintf(stderr,"Leaving ANMA\n");
#endif

	return data;
}

INT *allocateGPUIntMemory(int length) {

#ifdef DEBUG
	fprintf(stderr,"Entering ANMA-Int\n");
#endif

	INT *data;
	SAFE_CUDA(cudaMalloc((void**) &data, SIZE_INT * length),data);
	if (data == NULL) {
		fprintf(stderr,"Failed to allocate INT memory on device!\n");
		exit(-1);
	}

#ifdef DEBUG
	fprintf(stderr,"Allocated %d to %d.\n",data,(data+length));
	fprintf(stderr,"Leaving ANMA\n");
#endif

	return data;
}

void freeGPUMemory(void *ptr) {

#ifdef DEBUG
	fprintf(stderr,"Entering FNMA\n");
#endif

	if (ptr != 0) {
		cudaFree(ptr);
	}

#ifdef DEBUG
	fprintf(stderr,"Leaving FNMA\n");
#endif
}

void storeGPURealMemoryArray(REAL *toGPUPtr, REAL *fromGPUPtr, int length) {
	SAFE_CUDA(cudaMemcpy(toGPUPtr, fromGPUPtr, SIZE_REAL*length, cudaMemcpyDeviceToDevice),toGPUPtr);
}

void storeGPUIntMemoryArray(INT *toGPUPtr, INT *fromGPUPtr, int length) {
	SAFE_CUDA(cudaMemcpy(toGPUPtr, fromGPUPtr, SIZE_INT*length, cudaMemcpyDeviceToDevice),toGPUPtr);
}

int checkZeros(REAL* dPtr, int length) {
	REAL* hPtr = (REAL *) malloc(sizeof(REAL) * length);
	SAFE_CUDA(cudaMemcpy(hPtr, dPtr, sizeof(REAL)*length, cudaMemcpyDeviceToHost),dPtr);
	int i;
	REAL min = 1e+37f;
	REAL max = 0.0f;
	for (i = 0; i < length; i++) {
		if (hPtr[i] > 0 && hPtr[i] < min)
			min = hPtr[i];
		if (hPtr[i] > 0 && hPtr[i] > max)
			max = hPtr[i];
	}
	printf("min = %1.2e, max = %1.2e\n", min, max);
	free(hPtr);
	return 0;
}

void printfCudaVector(REAL* dPtr, int length) {

	REAL* hPtr = (REAL *) malloc(sizeof(REAL) * length);
	SAFE_CUDA(cudaMemcpy(hPtr, dPtr, sizeof(REAL)*length, cudaMemcpyDeviceToHost),dPtr);
#ifdef DOUBLE_PRECISION
	printfVectorD(hPtr, length);
#else
	printfVectorF(hPtr,length);
#endif
	free(hPtr);
}

void printfCudaInt(int* dPtr, int length) {
	int* hPtr = (int *) malloc(sizeof(int) * length);
	SAFE_CUDA(cudaMemcpy(hPtr, dPtr, sizeof(int)*length, cudaMemcpyDeviceToHost),dPtr);
	printfInt(hPtr,length);
	free(hPtr);
}

REAL sumCudaVector(REAL *dPtr, int length) {

	REAL* hPtr = (REAL *) malloc(sizeof(REAL) * length);
	SAFE_CUDA(cudaMemcpy(hPtr, dPtr, sizeof(REAL)*length, cudaMemcpyDeviceToHost),dPtr);
	REAL sum = 0;
	int i;
	for (i = 0; i < length; i++)
		sum += hPtr[i];
	free(hPtr);
	return sum;
}

void printfVectorD(double* ptr, int length) {
	fprintf(stderr,"[ %1.5e", ptr[0]);
	int i;
	for (i = 1; i < length; i++)
		fprintf(stderr," %1.5e", ptr[i]);
	fprintf(stderr," ]\n");
}

void printfVectorF(float* ptr, int length) {
	fprintf(stderr,"[ %1.5e", ptr[0]);
	int i;
	for (i = 1; i < length; i++)
		fprintf(stderr," %1.5e", ptr[i]);
	fprintf(stderr," ]\n");
	for(i=0; i<length; i++) {
		if(isnan(ptr[i])) {
			fprintf(stderr,"NaN found!\n");
			exit(0);
		}
	}
}

void printfVector(REAL* ptr, int length) {
	fprintf(stderr,"[ %1.5e", ptr[0]);
	int i;
	for (i = 1; i < length; i++)
		fprintf(stderr," %1.5e", ptr[i]);
	fprintf(stderr," ]\n");
}

void printfInt(int* ptr, int length) {
	fprintf(stderr,"[ %d",ptr[0]);
	int i;
	for (i = 1; i < length; i++)
		fprintf(stderr," %d",ptr[i]);
	fprintf(stderr," ]\n");
}

#ifdef __cplusplus
}
#endif
