#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <cmath>
#include <jni.h>

#include "libhmsbeagle/beagle.h"
#include "libhmsbeagle/JNI/beagle_basta_BastaJNIWrapper.h"


/*
 * Class:     beagle_basta_BastaJNIWrapper
 * Method:    allocateCoalescentBuffers
 * Signature: (IIIII)I
 */
JNIEXPORT jint JNICALL Java_beagle_basta_BastaJNIWrapper_allocateCoalescentBuffers
        (JNIEnv *env, jobject obj, jint instance, jint bufferCount, jint bufferLength, jint partialsCount, jint initial, jint numThreads) {

    jint errCode = (jint)beagleAllocateBastaBuffers(instance, bufferCount, bufferLength, partialsCount, initial, numThreads);

    return errCode;
}

/*
 * Class:     beagle_basta_BastaJNIWrapper
 * Method:    updateBastaPartials
 * Signature: (I[III)I
 */
JNIEXPORT jint JNICALL Java_beagle_basta_BastaJNIWrapper_updateBastaPartials
  (JNIEnv *env, jobject obj, jint instance, jintArray inOperations, 
   jint operationCount, jintArray inIntervals, jint intervalCount,
   jint populationSizesIndex, jint coalescentIndex) {
  	
  	jint *operations = env->GetIntArrayElements(inOperations, NULL);  	
  	jint *intervals = env->GetIntArrayElements(inIntervals, NULL);

	jint errCode = (jint)beagleUpdateBastaPartials(instance, 
		(BastaOperation*) operations, operationCount,
		(const int*) intervals, intervalCount,
        populationSizesIndex, coalescentIndex);

    env->ReleaseIntArrayElements(inOperations, operations, JNI_ABORT);
    env->ReleaseIntArrayElements(inIntervals, intervals, JNI_ABORT);

    return errCode;
  
  }

/*
 * Class:     beagle_basta_BastaJNIWrapper
 * Method:    updateBastaPartialsGrad
 * Signature: (I[III)I
 */
JNIEXPORT jint JNICALL Java_beagle_basta_BastaJNIWrapper_updateBastaPartialsGrad
  (JNIEnv *env, jobject obj, jint instance, jintArray inOperations, 
   jint operationCount, jintArray inIntervals, jint intervalCount,
   jint populationSizesIndex, jint coalescentIndex) {
  	
  	jint *operations = env->GetIntArrayElements(inOperations, NULL);  	
  	jint *intervals = env->GetIntArrayElements(inIntervals, NULL);

	jint errCode = (jint)beagleUpdateBastaPartialsGrad(instance, 
		(BastaOperation*) operations, operationCount,
		(const int*) intervals, intervalCount,
        populationSizesIndex, coalescentIndex);

    env->ReleaseIntArrayElements(inOperations, operations, JNI_ABORT);
    env->ReleaseIntArrayElements(inIntervals, intervals, JNI_ABORT);

    return errCode;
  
  }


JNIEXPORT jint JNICALL Java_beagle_basta_BastaJNIWrapper_updateTransitionMatricesGrad
  (JNIEnv *env, jobject obj, jint instance, jintArray inProbabilityIndices, jdoubleArray inEdgeLengths, jint count)
{
    jint *probabilityIndices = env->GetIntArrayElements(inProbabilityIndices, NULL);
    jdouble *edgeLengths = env->GetDoubleArrayElements(inEdgeLengths, NULL);
    jint errCode = (jint)beagleUpdateTransitionMatricesGrad(instance, (int *)probabilityIndices, (double *)edgeLengths, count);

    env->ReleaseDoubleArrayElements(inEdgeLengths, edgeLengths, JNI_ABORT);
    env->ReleaseIntArrayElements(inProbabilityIndices, probabilityIndices, JNI_ABORT);
    return errCode;
}
/*
 * Class:     beagle_basta_BastaJNIWrapper
 * Method:    getBastaBuffer
 * Signature: (II[D)I
 */
JNIEXPORT jint JNICALL Java_beagle_basta_BastaJNIWrapper_getBastaBuffer
        (JNIEnv *env, jobject object, jint instance, jint bufferIndex, jdoubleArray out) {

    jdouble *array = env->GetDoubleArrayElements(out, NULL);
    jint errCode = beagleGetBastaBuffer(instance, bufferIndex, (double *)array);

    // not using JNI_ABORT flag here because we want the values to be copied back...
    env->ReleaseDoubleArrayElements(out, array, 0);
    return errCode;
}
  
/*
 * Class:     beagle_basta_BastaJNIWrapper
 * Method:    accumulateBastaPartials
 * Signature: (I[II[II)I
 */
JNIEXPORT jint JNICALL Java_beagle_basta_BastaJNIWrapper_accumulateBastaPartials
  (JNIEnv *env, jobject object, jint instance,
   jintArray inOperations, jint operationCount,
   jintArray inIntervals, jint intervalCount,
   jdoubleArray inIntervalLengths,
   jint populationSizesIndex,
   jint coalescentIndex,
   jdoubleArray out) {

    jint *operations = env->GetIntArrayElements(inOperations, NULL);
    jint *intervals = env->GetIntArrayElements(inIntervals, NULL);

    jdouble *array = env->GetDoubleArrayElements(out, NULL);
    jdouble *intervalLengths = env->GetDoubleArrayElements(inIntervalLengths, NULL);

    jint errCode = beagleAccumulateBastaPartials(instance,
                                                 (const BastaOperation*) operations, operationCount,
                                                 (const int*) intervals, intervalCount,
                                                 (double *)intervalLengths,
                                                 populationSizesIndex,
                                                 coalescentIndex, (double *)array);

    env->ReleaseIntArrayElements(inOperations, operations, JNI_ABORT);
    env->ReleaseIntArrayElements(inIntervals, intervals, JNI_ABORT);
    env->ReleaseDoubleArrayElements(inIntervalLengths, intervalLengths, JNI_ABORT);

    // not using JNI_ABORT flag here because we want the values to be copied back...
    env->ReleaseDoubleArrayElements(out, array, 0);

  	return errCode;
  
  }

JNIEXPORT jint JNICALL Java_beagle_basta_BastaJNIWrapper_accumulateBastaPartialsGrad
  (JNIEnv *env, jobject object, jint instance,
   jintArray inOperations, jint operationCount,
   jintArray inIntervals, jint intervalCount,
   jdoubleArray inIntervalLengths,
   jint populationSizesIndex,
   jint coalescentIndex,
   jint eigenIndex,
   jint partialAdjointBufferBase,
   jint matrixAdjointBufferBase,
   jdoubleArray popSizeGradOut,
   jdoubleArray out) {

    jint *operations = env->GetIntArrayElements(inOperations, NULL);
    jint *intervals = env->GetIntArrayElements(inIntervals, NULL);

    jdouble *array = env->GetDoubleArrayElements(out, NULL);
    jdouble *intervalLengths = env->GetDoubleArrayElements(inIntervalLengths, NULL);
    jdouble *popSizeArray = (popSizeGradOut == NULL)
            ? NULL
            : env->GetDoubleArrayElements(popSizeGradOut, NULL);

    jint errCode = beagleAccumulateBastaPartialsGrad(instance,
                                                 (const BastaOperation*) operations, operationCount,
                                                 (const int*) intervals, intervalCount,
                                                 (double *)intervalLengths,
                                                 populationSizesIndex,
                                                 coalescentIndex,
                                                 eigenIndex,
                                                 partialAdjointBufferBase,
                                                 matrixAdjointBufferBase,
                                                 (double *)popSizeArray,
                                                 (double *)array);

    env->ReleaseIntArrayElements(inOperations, operations, JNI_ABORT);
    env->ReleaseIntArrayElements(inIntervals, intervals, JNI_ABORT);
    env->ReleaseDoubleArrayElements(inIntervalLengths, intervalLengths, JNI_ABORT);

    // not using JNI_ABORT flag here because we want the values to be copied back...
    env->ReleaseDoubleArrayElements(out, array, 0);
    if (popSizeArray != NULL) {
        env->ReleaseDoubleArrayElements(popSizeGradOut, popSizeArray, 0);
    }

  	return errCode;

  }

JNIEXPORT jint JNICALL Java_beagle_basta_BastaJNIWrapper_allocateCoalescentGradBuffers
        (JNIEnv *env, jobject obj, jint instance, jint partialsCount) {
    return (jint)beagleAllocateBastaGradBuffers(instance, partialsCount);
}

JNIEXPORT jint JNICALL Java_beagle_basta_BastaJNIWrapper_accumulateEigenBasisGradient
  (JNIEnv *env, jobject object, jint instance,
   jint eigenIndex, jint matrixAdjointBufferBase,
   jdoubleArray inBranchLengths,
   jint matrixCount, jint hasComplex, jdoubleArray out) {

    jdouble *branchLengths = env->GetDoubleArrayElements(inBranchLengths, NULL);
    jdouble *array = env->GetDoubleArrayElements(out, NULL);
    jint errCode = beagleAccumulateEigenBasisGradient(instance,
                                                      eigenIndex,
                                                      matrixAdjointBufferBase,
                                                      (const double *)branchLengths,
                                                      matrixCount, hasComplex,
                                                      (double *)array);

    env->ReleaseDoubleArrayElements(inBranchLengths, branchLengths, JNI_ABORT);
    env->ReleaseDoubleArrayElements(out, array, 0);
    return errCode;
  }

JNIEXPORT jint JNICALL Java_beagle_basta_BastaJNIWrapper_uploadBastaSlabMetadata
  (JNIEnv *env, jobject object, jint instance, jintArray inPacked, jint packedLen) {

    jint *packed = env->GetIntArrayElements(inPacked, NULL);
    jint errCode = (jint) beagleUploadBastaSlabMetadata(instance,
                                                        (const int*) packed,
                                                        packedLen);

    env->ReleaseIntArrayElements(inPacked, packed, JNI_ABORT);
    return errCode;
  }


JNIEXPORT jint JNICALL Java_beagle_basta_BastaJNIWrapper_getBastaSlabConstants
  (JNIEnv *env, jobject object, jint instance, jintArray inOut) {

    jsize outLen = env->GetArrayLength(inOut);
    if (outLen < 2) {
        return (jint) BEAGLE_ERROR_OUT_OF_RANGE;
    }

    int opsPerBlock = 0;
    int indexOffsetPat = 0;
    jint errCode = (jint) beagleGetBastaSlabConstants(instance,
                                                      &opsPerBlock,
                                                      &indexOffsetPat);

    if (errCode == BEAGLE_SUCCESS) {
        jint *out = env->GetIntArrayElements(inOut, NULL);
        out[0] = (jint) opsPerBlock;
        out[1] = (jint) indexOffsetPat;
        env->ReleaseIntArrayElements(inOut, out, 0);
    }
    return errCode;
  }
