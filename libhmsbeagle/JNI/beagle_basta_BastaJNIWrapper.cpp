#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <jni.h>

#include "libhmsbeagle/beagle.h"
#include "libhmsbeagle/JNI/beagle_basta_BastaJNIWrapper.h"


/*
 * Class:     beagle_basta_BastaJNIWrapper
 * Method:    allocateCoalescentBuffers
 * Signature: (III)I
 */
JNIEXPORT jint JNICALL Java_beagle_basta_BastaJNIWrapper_allocateCoalescentBuffers
        (JNIEnv *env, jobject obj, jint instance, jint bufferCount, jint bufferLength) {

    jint errCode = (jint)beagleAllocateBastaBuffers(instance, bufferCount, bufferLength);

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
		intervals, intervalCount,
        populationSizesIndex, coalescentIndex);

    env->ReleaseIntArrayElements(inOperations, operations, JNI_ABORT);
    env->ReleaseIntArrayElements(inIntervals, intervals, JNI_ABORT);

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
                                                 intervals, intervalCount,
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
