
/*
 * @author Marc Suchard
 * @author Dat Huynh
 */

#ifndef __CUDASharedFunctions__
#define __CUDASharedFunctions__

//#include <jni.h>
#include <math.h>
#include <cuda.h>

//#define DYNAMIC_SCALING

#define SCALING_REFRESH	0
//#define ALWAYS_RESCALE

//#define USE_CONTEXTS
#define COHERENT // Transposes output of finite-time probabilities; leads to coherent global memory reads in peeling kernels! yeah!!!

#define PRE_LOAD

#define LAZY_STORE

//#define DTEST

//#define DEBUG_FLOW
//#define DEBUG_GPU
//#define DEBUG_BEAGLE

//#define DEBUG_13

#ifdef DEBUG_13
#define PAUSE	cudaThreadSynchronize();
#else
#define PAUSE
#endif

/* Definition of REAL can be switched between 'double' and 'float' */
#ifdef DOUBLE_PRECISION
#define REAL		double
#else
#define REAL		float
#endif

#define SIZE_REAL	sizeof(REAL)
#define INT		int
#define SIZE_INT	sizeof(INT)

/* Compiler definitions
 *
 * STATE_COUNT 	      - Controlled by Makefile
 *
 * PADDED_STATE_COUNT - # of total states after augmentation
 * 					    *should* be a multiple of 16
 *
 * PATTERN_BLOCK_SIZE - # of patterns to pack onto each thread-block in pruning ( x 4 for PADDED_STATE_COUNT==4)
 * 						PATTERN_BLOCK_SIZE * PADDED_STATE_COUNT <= 512
 *
 * MATRIX_BLOCK_SIZE  - # of matrices to pack onto each thread-block in integrating likelihood and store in dynamic weighting;
 * 					    MATRIX_BLOCK_SIZE * PADDED_STATE_COUNT <= 512
 * 					  - TODO Currently matrixCount must be < MATRIX_BLOCK_SIZE, fix!
 *
 * BLOCK_PEELING_SIZE - # of the states to pre-fetch in inner-sum in pruning;
 * 						BLOCK_PEELING_SIZE <= PATTERN_BLOCK_SIZE and
 * 						*must* be a divisor of PADDED_STATE_COUNT
 */

#if (STATE_COUNT == 4)
	#define PADDED_STATE_COUNT	4
#else
#if (STATE_COUNT <= 16) // else if
	#define PADDED_STATE_COUNT	16
#else
#if (STATE_COUNT <= 32) // else if
	#define PADDED_STATE_COUNT	32
#else
#if (STATE_COUNT <= 64) // else if
	#define PADDED_STATE_COUNT	64
#else
#if (STATE_COUNT <= 128) // else if
	#define PADDED_STATE_COUNT	128
#else
#if (STATE_COUNT <= 192) // else if
	#define PADDED_STATE_COUNT 192
#endif
#endif
#endif
#endif
#endif
#endif

#define PADDED_STATES	PADDED_STATE_COUNT - STATE_COUNT

#if (PADDED_STATE_COUNT == 4)
	#define PATTERN_BLOCK_SIZE	16
#endif

// TODO Find optimal settings for PADDED_STATE_COUNT == 32

#if (PADDED_STATE_COUNT == 64)
	#ifdef DOUBLE_PRECISION
		#define PATTERN_BLOCK_SIZE      8
		#define BLOCK_PEELING_SIZE	4
	#else
		#define PATTERN_BLOCK_SIZE	8
		#define BLOCK_PEELING_SIZE	8
	#endif
#endif

#if (PADDED_STATE_COUNT == 128)
	#define PATTERN_BLOCK_SIZE	4
	#define BLOCK_PEELING_SIZE	4
	#define SLOW_REWEIGHING
#endif

#if (PADDED_STATE_COUNT == 192)
	#define PATTERN_BLOCK_SIZE	2
	#define BLOCK_PEELING_SIZE	2
	#define SLOW_REWEIGHING
#endif

/* Defaults */

#ifndef PATTERN_BLOCK_SIZE
	#define PATTERN_BLOCK_SIZE	16
#endif

#ifndef MATRIX_BLOCK_SIZE
	#define MATRIX_BLOCK_SIZE	8
#endif

#ifndef BLOCK_PEELING_SIZE
	#define BLOCK_PEELING_SIZE	8
#endif

#define MULTIPLY_BLOCK_SIZE	16


//#ifdef USE_CONTEXTS
//#define ENTERGPU
//#define EXITGPU
//#define ENTERGPU2	status = cuCtxPushCurrent( thread[instance].context.hcuContext ); \
//					if ( CUDA_SUCCESS != status ) { \
//						fprintf(stderr,"Unable to push context %d - %d\n",thread[instance].context.hcuContext,status); \
//						exit(0); \
//					} else { \
//						fprintf(stderr,"Pushed context.\n"); \
//					}
//#define EXITGPU2	status = cuCtxPopCurrent( &thread[instance].context.hcuContext ); \
//					if ( CUDA_SUCCESS != status ) { \
//						fprintf(stderr,"Unable to pop context %d - %d\n",thread[instance].context.hcuContext,status); \
//						exit(0); \
//					} else { \
//						fprintf(stderr,"Popped context.\n"); \
//					}
//#else
//#define ENTERGPU
//#define EXITGPU
//#define ENTERGPU2	cudaSetDevice(thread[instance].device);
//#define ENTERGPU2
//#define EXITGPU2
//#endif // USE_CONTEXTS

//#define ENTERGPU	CUresult status = cuCtxPushCurrent( hcuContext ); \
//                        if ( CUDA_SUCCESS != status ) {		\
//			  printf("Unable to push context %d\n",hcuContext); \
//			  exit(0);					\
//			} else {					\
//			  printf("Entered context.\n");			\
//			}
//#define EXITGPU         cuCtxPopCurrent( NULL );


#define SAFE_CUDA(call,ptr)		cudaError_t error = call; \
								if( error != 0 ) { \
									fprintf(stderr,"Error %s\n", cudaGetErrorString(error)); \
									fprintf(stderr,"Ptr = %d\n",ptr); \
									exit(-1); \
								}


#define MEMCPY(to,from,length,toType) { int m; \
										for(m=0; m<length; m++) { \
											to[m] = (toType) from[m]; \
										} }

int initCUDAContext();

int migrateContext(CUcontext context);

REAL *allocateGPURealMemory(int length);

INT  *allocateGPUIntMemory(int length);

//void checkCUDAError(const char *msg);

void freeGPUMemory(void *ptr);

//void setGPURealMemoryArray(JNIEnv *env, jdoubleArray inFromJavaArray, int fromOffset,
//	REAL* inGPUPtr, int toOffset, int length, REAL* hCache);

//void setGPUIntMemoryArray(JNIEnv *env, jintArray inFromJavaArray, int fromOffset,
//	INT* inGPUPtr, int toOffset, int length, INT* hCache);

//void getGPURealMemoryArray(JNIEnv *env, REAL* inGPUPtr, int fromOffset, jdoubleArray
//	inToJavaArray, int toOffset, int length, REAL *hCache);

//void getGPUIntMemoryArray(JNIEnv *env, INT* inGPUPtr, int fromOffset, jintArray
//	inToJavaArray, int toOffset, int length, INT *hCache);

//void setGPURealMemoryArrayMultipleTimes(JNIEnv *env,jdoubleArray inFromJavaArray, int fromOffset,
//	REAL *inGPUPtr, int toOffset, int length, int times);

//void setGPURealMemoryArrayMultipleTimesPadded(JNIEnv *env,jdoubleArray inFromJavaArray, int fromOffset,
//	REAL *inGPUPtr, int toOffset, int length, int times, int padding, REAL *hCache);

void storeGPURealMemoryArray(REAL *toGPUPtr, REAL *fromGPUPtr, int length);

void storeGPUIntMemoryArray(INT *toGPUPtr, INT *fromGPUPtr, int length);

__global__ void matrixMul( REAL* C, REAL* A, REAL* B, int wA, int wB);

__global__ void matrixMulParallel( REAL** C, REAL* A, REAL *cache, int wA, int wB, int matrixId);

__global__ void matrixMulMod( REAL* C, REAL* A, REAL* B, int wA, int wB);

void printfCudaVector(REAL *dPtr, int length);

void printfVector(REAL *ptr, int length);

REAL sumCudaVector(REAL *dPtr, int length);

int checkZeros(REAL* dPtr, int length);

//void getGPUInfoNew(int iDevice);

//void checkCUDAError(const char *msg);

typedef struct _CUDAContext_st {
    CUcontext   hcuContext;
    CUmodule    hcuModule;
    CUfunction  hcuFunction;
    CUdeviceptr dptr;
    int        	deviceID;
    int        	threadNum;
} CUDAContext;


void loadTipPartials(int instance);

void doStore(int instance);

void doRestore(int instance);

void handleStoreRestoreQueue(int instance);

#define QUEUESIZE       1000

typedef struct {
        int q[QUEUESIZE+1];		/* body of queue */
        int first;                      /* position of first element */
        int last;                       /* position of last element */
        int count;                      /* number of queue elements */
} queue;

void initQueue(queue *q);

void enQueue(queue *q, int x);

int deQueue(queue *q);

int queueEmpty(queue *q);

void printQueue(queue *q);

#endif // __CUDASharedFunctions

