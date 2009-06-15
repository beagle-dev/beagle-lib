#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <jni.h>

#include "libbeagle-lib/beagle.h"
#include "libbeagle-lib/JNI/beagle_BeagleJNIWrapper.h"

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    initialize
 * Signature: (IIIIIII)I
 */
JNIEXPORT jint JNICALL Java_beagle_BeagleJNIWrapper_createInstance
	(JNIEnv *env, jobject obj, jint tipCount, jint partialsBufferCount, jint compactBufferCount, jint stateCount, jint patternCount, jint eigenBufferCount, jint matrixBufferCount, jintArray inResourceList, jint resourceCount, jint preferenceFlags, jint requirementFlags)
{
    jint *resourceList = env->GetIntArrayElements(inResourceList, NULL);

	jint instance = createInstance(tipCount,
	                                partialsBufferCount,
	                                compactBufferCount,
	                                stateCount,
	                                patternCount,
	                                eigenBufferCount,
	                                matrixBufferCount,
	                                (int *)resourceList,
	                                resourceCount,
	                                preferenceFlags,
	                                requirementFlags);

    env->ReleaseIntArrayElements(inResourceList, resourceList, JNI_ABORT);

    return instance;
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    finalize
 * Signature: (I)I
 */
JNIEXPORT jint JNICALL Java_beagle_BeagleJNIWrapper_finalize
  (JNIEnv *env, jobject obj, jint instance)
{
	finalize(instance);
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    setPartials
 * Signature: (II[D)I
 */
JNIEXPORT jint JNICALL Java_beagle_BeagleJNIWrapper_setPartials
  (JNIEnv *env, jobject obj, jint instance, jint bufferIndex, jdoubleArray inPartials)
{
    jdouble *partials = env->GetDoubleArrayElements(inPartials, NULL);

	setPartials(instance, bufferIndex, (double *)partials);

    env->ReleaseDoubleArrayElements(inPartials, partials, JNI_ABORT);
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    getPartials
 * Signature: (II[D)I
 */
JNIEXPORT jint JNICALL Java_beagle_BeagleJNIWrapper_getPartials
  (JNIEnv *env, jobject obj, jint instance, jint bufferIndex, jdoubleArray outPartials)
{
    jdouble *partials = env->GetDoubleArrayElements(outPartials, NULL);

	getPartials(instance, bufferIndex, (double *)partials);

    // not using JNI_ABORT flag here because we want the values to be copied back...
    env->ReleaseDoubleArrayElements(outPartials, partials, 0);
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    setTipStates
 * Signature: (II[I)V
 */
JNIEXPORT void JNICALL Java_beagle_BeagleJNIWrapper_setTipStates
  (JNIEnv *env, jobject obj, jint instance, jint tipIndex, jintArray inTipStates)
{
    jint *tipStates = env->GetIntArrayElements(inTipStates, NULL);

	setTipStates(instance, tipIndex, (int *)tipStates);

    env->ReleaseIntArrayElements(inTipStates, tipStates, JNI_ABORT);
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    setEigenDecomposition
 * Signature: (II[D[D[D)V
 */
JNIEXPORT void JNICALL Java_beagle_BeagleJNIWrapper_setEigenDecomposition
(JNIEnv *env, jobject obj, jint instance, jint eigenIndex, jdoubleArray inEigenVectors, jdoubleArray inInvEigenVectors, jdoubleArray inEigenValues)
{
    jdouble *eigenVectors = env->GetDoubleArrayElements(inEigenVectors, NULL);
    jdouble *invEigenVectors = env->GetDoubleArrayElements(inInvEigenVectors, NULL);
    jdouble *eigenValues = env->GetDoubleArrayElements(inEigenValues, NULL);

	setEigenDecomposition(instance, eigenIndex, (double *)eigenVectors, (double *)invEigenVectors, (double *)eigenValues);

    env->ReleaseDoubleArrayElements(inEigenValues, eigenValues, JNI_ABORT);
    env->ReleaseDoubleArrayElements(inInvEigenVectors, invEigenVectors, JNI_ABORT);
    env->ReleaseDoubleArrayElements(inEigenVectors, eigenVectors, JNI_ABORT);
}


/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    setTransitionMatrix
 * Signature: (II[D)I
 */
JNIEXPORT jint JNICALL Java_beagle_BeagleJNIWrapper_setTransitionMatrix
  (JNIEnv *env, jobject obj, jint instance, jint matrixIndex, jdoubleArray inMatrix)
{
    jdouble *matrix = env->GetDoubleArrayElements(inMatrix, NULL);

	setTransitionMatrix(instance, matrixIndex, (double *)matrix);

    env->ReleaseDoubleArrayElements(inMatrix, matrix, JNI_ABORT);
}


/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    updateTransitionMatrices
 * Signature: (II[I[I[I[DI)I
 */
JNIEXPORT jint JNICALL Java_beagle_BeagleJNIWrapper_updateTransitionMatrices
  (JNIEnv *env, jobject obj, jint instance, jint eigenIndex, jintArray inProbabilityIndices, jintArray inFirstDerivativeIndices, jintArray inSecondDervativeIndices, jdoubleArray inEdgeLengths, jint count)
{
    jint *probabilityIndices = env->GetIntArrayElements(inProbabilityIndices, NULL);
    jint *firstDerivativeIndices = env->GetIntArrayElements(inFirstDerivativeIndices, NULL);
    jint *secondDervativeIndices = env->GetIntArrayElements(inSecondDervativeIndices, NULL);
    jdouble *edgeLengths = env->GetDoubleArrayElements(inEdgeLengths, NULL);

	updateTransitionMatrices(instance, eigenIndex, (int *)probabilityIndices, (int *)firstDerivativeIndices, (int *)secondDervativeIndices, (double *)edgeLengths, count);

    env->ReleaseDoubleArrayElements(inEdgeLengths, edgeLengths, JNI_ABORT);
    env->ReleaseIntArrayElements(inSecondDervativeIndices, secondDervativeIndices, JNI_ABORT);
    env->ReleaseIntArrayElements(inFirstDerivativeIndices, firstDerivativeIndices, JNI_ABORT);
    env->ReleaseIntArrayElements(inProbabilityIndices, probabilityIndices, JNI_ABORT);
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    updatePartials
 * Signature: ([II[III)I
 */
JNIEXPORT jint JNICALL Java_beagle_BeagleJNIWrapper_updatePartials
  (JNIEnv *env, jobject obj, jintArray inInstances, jint instanceCount, jintArray inOperations, jint operationCount, jint rescale)
{
    jint *instances = env->GetIntArrayElements(inInstances, NULL);
    jint *operations = env->GetIntArrayElements(inOperations, NULL);

	updatePartials((int *)instances, instanceCount, (int *)operations, operationCount, rescale);

    env->ReleaseIntArrayElements(inOperations, operations, JNI_ABORT);
    env->ReleaseIntArrayElements(inInstances, instances, JNI_ABORT);
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    calculateRootLogLikelihoods
 * Signature: (I[I[D[DI[D)I
 */
JNIEXPORT jint JNICALL Java_beagle_BeagleJNIWrapper_calculateRootLogLikelihoods
  (JNIEnv *env, jobject obj, jint instance, jintArray inBufferIndices, jdoubleArray inWeights, jdoubleArray inStateFrequencies, jint count, jdoubleArray outLogLikelihoods)
{
    jint *bufferIndices = env->GetIntArrayElements(inBufferIndices, NULL);
    jdouble *weights = env->GetDoubleArrayElements(inWeights, NULL);
    jdouble *stateFrequencies = env->GetDoubleArrayElements(inStateFrequencies, NULL);
    jdouble *logLikelihoods = env->GetDoubleArrayElements(outLogLikelihoods, NULL);

	calculateRootLogLikelihoods(instance, (int *)bufferIndices, (double *)weights, (double *)stateFrequencies, count, (double *)logLikelihoods);

    // not using JNI_ABORT flag here because we want the values to be copied back...
    env->ReleaseDoubleArrayElements(outLogLikelihoods, logLikelihoods, 0);

    env->ReleaseDoubleArrayElements(inStateFrequencies, stateFrequencies, JNI_ABORT);
    env->ReleaseDoubleArrayElements(inWeights, weights, JNI_ABORT);
    env->ReleaseIntArrayElements(inBufferIndices, bufferIndices, JNI_ABORT);
}

