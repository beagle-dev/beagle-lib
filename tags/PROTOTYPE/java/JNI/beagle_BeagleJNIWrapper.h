/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class beagle_BeagleJNIWrapper */

#ifndef _Included_beagle_BeagleJNIWrapper
#define _Included_beagle_BeagleJNIWrapper
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    initialize
 * Signature: (IIIIII)I
 */
JNIEXPORT jint JNICALL Java_beagle_BeagleJNIWrapper_initialize
  (JNIEnv *, jobject, jint, jint, jint, jint, jint, jint);

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    finalize
 * Signature: (I)V
 */
JNIEXPORT void JNICALL Java_beagle_BeagleJNIWrapper_finalize
  (JNIEnv *, jobject, jint);

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    setTipPartials
 * Signature: (II[D)V
 */
JNIEXPORT void JNICALL Java_beagle_BeagleJNIWrapper_setTipPartials
  (JNIEnv *, jobject, jint, jint, jdoubleArray);

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    setTipStates
 * Signature: (II[I)V
 */
JNIEXPORT void JNICALL Java_beagle_BeagleJNIWrapper_setTipStates
  (JNIEnv *, jobject, jint, jint, jintArray);

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    setStateFrequencies
 * Signature: (I[D)V
 */
JNIEXPORT void JNICALL Java_beagle_BeagleJNIWrapper_setStateFrequencies
  (JNIEnv *, jobject, jint, jdoubleArray);

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    setEigenDecomposition
 * Signature: (II[[D[[D[D)V
 */
JNIEXPORT void JNICALL Java_beagle_BeagleJNIWrapper_setEigenDecomposition
  (JNIEnv *, jobject, jint, jint, jobjectArray, jobjectArray, jdoubleArray);

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    setCategoryRates
 * Signature: (I[D)V
 */
JNIEXPORT void JNICALL Java_beagle_BeagleJNIWrapper_setCategoryRates
  (JNIEnv *, jobject, jint, jdoubleArray);

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    setCategoryProportions
 * Signature: (I[D)V
 */
JNIEXPORT void JNICALL Java_beagle_BeagleJNIWrapper_setCategoryProportions
  (JNIEnv *, jobject, jint, jdoubleArray);

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    calculateProbabilityTransitionMatrices
 * Signature: (I[I[DI)V
 */
JNIEXPORT void JNICALL Java_beagle_BeagleJNIWrapper_calculateProbabilityTransitionMatrices
  (JNIEnv *, jobject, jint, jintArray, jdoubleArray, jint);

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    calculatePartials
 * Signature: (I[I[IIB)V
 */
JNIEXPORT void JNICALL Java_beagle_BeagleJNIWrapper_calculatePartials
  (JNIEnv *, jobject, jint, jintArray, jintArray, jint, jboolean);

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    calculateLogLikelihoods
 * Signature: (II[D)V
 */
JNIEXPORT void JNICALL Java_beagle_BeagleJNIWrapper_calculateLogLikelihoods
  (JNIEnv *, jobject, jint, jint, jdoubleArray);

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    storeState
 * Signature: (I)V
 */
JNIEXPORT void JNICALL Java_beagle_BeagleJNIWrapper_storeState
  (JNIEnv *, jobject, jint);

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    restoreState
 * Signature: (I)V
 */
JNIEXPORT void JNICALL Java_beagle_BeagleJNIWrapper_restoreState
  (JNIEnv *, jobject, jint);

#ifdef __cplusplus
}
#endif
#endif