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

/*
 * Class:     beagle_basta_BastaJNIWrapper
 * Method:    accumulateBastaPartialsGrad
 * Signature: (I[II[II[DII[D)I
 *
 * BEAGLE 4.1's 9-parameter form, which BEAST 10.5.0 and the 10.6.0 betas declare. Not supported: it returns
 * BEAGLE_ERROR_NO_IMPLEMENTATION without touching its arguments, so that BEAST gets an error, not a call of the
 * 13-parameter form with arguments missing.
 */
JNIEXPORT jint JNICALL Java_beagle_basta_BastaJNIWrapper_accumulateBastaPartialsGrad__I_3II_3II_3DII_3D
  (JNIEnv *env, jobject object, jint instance,
   jintArray inOperations, jint operationCount,
   jintArray inIntervals, jint intervalCount,
   jdoubleArray inIntervalLengths,
   jint populationSizesIndex,
   jint coalescentIndex,
   jdoubleArray out) {

    return (jint) BEAGLE_ERROR_NO_IMPLEMENTATION;
  }

/*
 * Class:     beagle_basta_BastaJNIWrapper
 * Method:    accumulateBastaPartialsGrad
 * Signature: (I[II[II[DIIIII[D[D)I
 */
JNIEXPORT jint JNICALL Java_beagle_basta_BastaJNIWrapper_accumulateBastaPartialsGrad__I_3II_3II_3DIIIII_3D_3D
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

/*
 * The natives below were added after BEAGLE 4.2.0 and are exported under their long names only (see the header).
 * Before calling BEAGLE they check that the Java arrays it reads and writes are long enough, and return
 * BEAGLE_ERROR_OUT_OF_RANGE without calling it when one is not.
 */

/*
 * The arrays of a _v5 call: operations is not null and holds BEAGLE_BASTA_OP_COUNT_V5 * operationCount ints, and
 * intervals, when not null, holds intervalCount ints. A null intervals is passed on as NULL, which BEAGLE rejects
 * when it would read it.
 */
static bool bastaArraysFit_v5(JNIEnv *env, jintArray operations, jint operationCount,
                              jintArray intervals, jint intervalCount) {
    if (operations == NULL ||
        (jlong) env->GetArrayLength(operations) < (jlong) BEAGLE_BASTA_OP_COUNT_V5 * operationCount) {
        return false;
    }
    if (intervals != NULL && (jlong) env->GetArrayLength(intervals) < (jlong) intervalCount) {
        return false;
    }
    return true;
}

/*
 * Class:     beagle_basta_BastaJNIWrapper
 * Method:    setBastaPopulationSizes
 * Signature: (I[DI)I
 *
 * combinedSizesIntegrals holds 2 * requiredStorageSize values; null is passed on as NULL.
 */
JNIEXPORT jint JNICALL Java_beagle_basta_BastaJNIWrapper_setBastaPopulationSizes__I_3DI
        (JNIEnv *env, jobject obj, jint instance, jdoubleArray inCombinedSizesIntegrals, jint requiredStorageSize) {

    jdouble *combinedSizesIntegrals = NULL;
    if (inCombinedSizesIntegrals != NULL) {
        if ((jlong) env->GetArrayLength(inCombinedSizesIntegrals) < 2 * (jlong) requiredStorageSize) {
            return (jint) BEAGLE_ERROR_OUT_OF_RANGE;
        }
        combinedSizesIntegrals = env->GetDoubleArrayElements(inCombinedSizesIntegrals, NULL);
    }

    jint errCode = (jint)beagleSetBastaPopulationSizes(instance, (const double*)combinedSizesIntegrals, (int)requiredStorageSize);

    if (combinedSizesIntegrals != NULL) {
        env->ReleaseDoubleArrayElements(inCombinedSizesIntegrals, combinedSizesIntegrals, JNI_ABORT);
    }

    return errCode;
}

/*
 * Class:     beagle_basta_BastaJNIWrapper
 * Method:    setBastaPopulationSizesBuffer
 * Signature: (I[DII)I
 *
 * combinedSizesIntegrals holds 2 * requiredStorageSize values; null is passed on as NULL.
 */
JNIEXPORT jint JNICALL Java_beagle_basta_BastaJNIWrapper_setBastaPopulationSizesBuffer__I_3DII
        (JNIEnv *env, jobject obj, jint instance, jdoubleArray inCombinedSizesIntegrals, jint requiredStorageSize, jint bufferIndex) {

    jdouble *combinedSizesIntegrals = NULL;
    if (inCombinedSizesIntegrals != NULL) {
        if ((jlong) env->GetArrayLength(inCombinedSizesIntegrals) < 2 * (jlong) requiredStorageSize) {
            return (jint) BEAGLE_ERROR_OUT_OF_RANGE;
        }
        combinedSizesIntegrals = env->GetDoubleArrayElements(inCombinedSizesIntegrals, NULL);
    }

    jint errCode = (jint)beagleSetBastaPopulationSizesBuffer(instance, (const double*)combinedSizesIntegrals, (int)requiredStorageSize, (int)bufferIndex);

    if (combinedSizesIntegrals != NULL) {
        env->ReleaseDoubleArrayElements(inCombinedSizesIntegrals, combinedSizesIntegrals, JNI_ABORT);
    }

    return errCode;
}

/*
 * Class:     beagle_basta_BastaJNIWrapper
 * Method:    setCurrentPopulationSizeBuffer
 * Signature: (II)I
 */
JNIEXPORT jint JNICALL Java_beagle_basta_BastaJNIWrapper_setCurrentPopulationSizeBuffer__II
        (JNIEnv *env, jobject obj, jint instance, jint bufferIndex) {

    jint errCode = (jint)beagleSetCurrentPopulationSizeBuffer(instance, (int)bufferIndex);

    return errCode;
}

/*
 * Class:     beagle_basta_BastaJNIWrapper
 * Method:    updateBastaPartials_v5
 * Signature: (I[II[IIII)I
 */
JNIEXPORT jint JNICALL Java_beagle_basta_BastaJNIWrapper_updateBastaPartials_1v5__I_3II_3IIII
  (JNIEnv *env, jobject obj, jint instance, jintArray inOperations,
   jint operationCount, jintArray inIntervals, jint intervalCount,
   jint populationSizesIndex, jint coalescentIndex) {

    if (!bastaArraysFit_v5(env, inOperations, operationCount, inIntervals, intervalCount)) {
        return (jint) BEAGLE_ERROR_OUT_OF_RANGE;
    }

    jint *operations = env->GetIntArrayElements(inOperations, NULL);
    jint *intervals = (inIntervals == NULL) ? NULL : env->GetIntArrayElements(inIntervals, NULL);

    jint errCode = (jint)beagleUpdateBastaPartials_v5(instance,
        (const BastaOperation_v5*) operations, operationCount,
        (const int*) intervals, intervalCount,
        populationSizesIndex, coalescentIndex);

    env->ReleaseIntArrayElements(inOperations, operations, JNI_ABORT);
    if (intervals != NULL) {
        env->ReleaseIntArrayElements(inIntervals, intervals, JNI_ABORT);
    }

    return errCode;
}

/*
 * Class:     beagle_basta_BastaJNIWrapper
 * Method:    updateBastaPartialsGrad_v5
 * Signature: (I[II[IIII)I
 */
JNIEXPORT jint JNICALL Java_beagle_basta_BastaJNIWrapper_updateBastaPartialsGrad_1v5__I_3II_3IIII
  (JNIEnv *env, jobject obj, jint instance, jintArray inOperations,
   jint operationCount, jintArray inIntervals, jint intervalCount,
   jint populationSizesIndex, jint coalescentIndex) {

    if (!bastaArraysFit_v5(env, inOperations, operationCount, inIntervals, intervalCount)) {
        return (jint) BEAGLE_ERROR_OUT_OF_RANGE;
    }

    jint *operations = env->GetIntArrayElements(inOperations, NULL);
    jint *intervals = (inIntervals == NULL) ? NULL : env->GetIntArrayElements(inIntervals, NULL);

    jint errCode = (jint)beagleUpdateBastaPartialsGrad_v5(instance,
        (const BastaOperation_v5*) operations, operationCount,
        (const int*) intervals, intervalCount,
        populationSizesIndex, coalescentIndex);

    env->ReleaseIntArrayElements(inOperations, operations, JNI_ABORT);
    if (intervals != NULL) {
        env->ReleaseIntArrayElements(inIntervals, intervals, JNI_ABORT);
    }

    return errCode;
}

/*
 * Class:     beagle_basta_BastaJNIWrapper
 * Method:    accumulateBastaPartials_v5
 * Signature: (I[II[II[DII[D)I
 *
 * intervalLengths may be null (the _v5 functions ignore it); out holds at least 1 value.
 */
JNIEXPORT jint JNICALL Java_beagle_basta_BastaJNIWrapper_accumulateBastaPartials_1v5__I_3II_3II_3DII_3D
  (JNIEnv *env, jobject object, jint instance,
   jintArray inOperations, jint operationCount,
   jintArray inIntervals, jint intervalCount,
   jdoubleArray inIntervalLengths,
   jint populationSizesIndex,
   jint coalescentIndex,
   jdoubleArray out) {

    if (!bastaArraysFit_v5(env, inOperations, operationCount, inIntervals, intervalCount) ||
        out == NULL || env->GetArrayLength(out) < 1) {
        return (jint) BEAGLE_ERROR_OUT_OF_RANGE;
    }

    jint *operations = env->GetIntArrayElements(inOperations, NULL);
    jint *intervals = (inIntervals == NULL) ? NULL : env->GetIntArrayElements(inIntervals, NULL);

    jdouble *array = env->GetDoubleArrayElements(out, NULL);
    jdouble *intervalLengths = (inIntervalLengths == NULL)
            ? NULL
            : env->GetDoubleArrayElements(inIntervalLengths, NULL);

    jint errCode = beagleAccumulateBastaPartials_v5(instance,
                                                    (const BastaOperation_v5*) operations, operationCount,
                                                    (const int*) intervals, intervalCount,
                                                    (const double *)intervalLengths,
                                                    populationSizesIndex,
                                                    coalescentIndex, (double *)array);

    env->ReleaseIntArrayElements(inOperations, operations, JNI_ABORT);
    if (intervals != NULL) {
        env->ReleaseIntArrayElements(inIntervals, intervals, JNI_ABORT);
    }
    if (intervalLengths != NULL) {
        env->ReleaseDoubleArrayElements(inIntervalLengths, intervalLengths, JNI_ABORT);
    }

    // not using JNI_ABORT flag here because we want the values to be copied back...
    env->ReleaseDoubleArrayElements(out, array, 0);

    return errCode;
}

/*
 * Class:     beagle_basta_BastaJNIWrapper
 * Method:    accumulateBastaPartialsGrad_v5
 * Signature: (I[II[II[DIIIII[D[D)I
 *
 * The 13 parameters of accumulateBastaPartialsGrad. intervalLengths and popSizeGradOut may be null; out must not be.
 */
JNIEXPORT jint JNICALL Java_beagle_basta_BastaJNIWrapper_accumulateBastaPartialsGrad_1v5__I_3II_3II_3DIIIII_3D_3D
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

    if (!bastaArraysFit_v5(env, inOperations, operationCount, inIntervals, intervalCount) || out == NULL) {
        return (jint) BEAGLE_ERROR_OUT_OF_RANGE;
    }

    jint *operations = env->GetIntArrayElements(inOperations, NULL);
    jint *intervals = (inIntervals == NULL) ? NULL : env->GetIntArrayElements(inIntervals, NULL);

    jdouble *array = env->GetDoubleArrayElements(out, NULL);
    jdouble *intervalLengths = (inIntervalLengths == NULL)
            ? NULL
            : env->GetDoubleArrayElements(inIntervalLengths, NULL);
    jdouble *popSizeArray = (popSizeGradOut == NULL)
            ? NULL
            : env->GetDoubleArrayElements(popSizeGradOut, NULL);

    jint errCode = beagleAccumulateBastaPartialsGrad_v5(instance,
                                                        (const BastaOperation_v5*) operations, operationCount,
                                                        (const int*) intervals, intervalCount,
                                                        (const double *)intervalLengths,
                                                        populationSizesIndex,
                                                        coalescentIndex,
                                                        eigenIndex,
                                                        partialAdjointBufferBase,
                                                        matrixAdjointBufferBase,
                                                        (double *)popSizeArray,
                                                        (double *)array);

    env->ReleaseIntArrayElements(inOperations, operations, JNI_ABORT);
    if (intervals != NULL) {
        env->ReleaseIntArrayElements(inIntervals, intervals, JNI_ABORT);
    }
    if (intervalLengths != NULL) {
        env->ReleaseDoubleArrayElements(inIntervalLengths, intervalLengths, JNI_ABORT);
    }

    // not using JNI_ABORT flag here because we want the values to be copied back...
    env->ReleaseDoubleArrayElements(out, array, 0);
    if (popSizeArray != NULL) {
        env->ReleaseDoubleArrayElements(popSizeGradOut, popSizeArray, 0);
    }

    return errCode;
}
