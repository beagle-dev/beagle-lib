#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <jni.h>

#include "beagle.h"
#include "beagle_BeagleJNIWrapper.h"

// Some temporary arrays used to convert floating point types
double** Evec;
double** Ievc;

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    initialize
 * Signature: (IIIIIII)I
 */
JNIEXPORT jint JNICALL Java_beagle_BeagleJNIWrapper_initialize
	(JNIEnv *env, jobject obj, jint nodeCount, jint tipCount, jint stateCount, jint patternCount, jint categoryCount, jint matrixCount)
{
	jint instance = initialize(nodeCount, tipCount, stateCount, patternCount, categoryCount, matrixCount);

	Evec = (double**)malloc(sizeof(double*) * STATE_COUNT);
	Ievc = (double**)malloc(sizeof(double*) * STATE_COUNT);
	for (int i = 0; i < STATE_COUNT; i++) {
	    Evec[i] = (double*)malloc(sizeof(double) * STATE_COUNT);
	    Ievc[i] = (double*)malloc(sizeof(double) * STATE_COUNT);
	}

    return instance;
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    finalize
 * Signature: (I)V
 */
JNIEXPORT void JNICALL Java_beagle_BeagleJNIWrapper_finalize
	(JNIEnv *env, jobject obj, jint instance)
{
	for (int i = 0; i < STATE_COUNT; i++) {
	    free(Evec[i]);
	    free(Ievc[i]);
	}
	free(Evec);
	free(Ievc);

	finalize(instance);
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    setTipPartials
 * Signature: (II[D)V
 */
JNIEXPORT void JNICALL Java_beagle_BeagleJNIWrapper_setTipPartials
	(JNIEnv *env, jobject obj, jint instance, jint tipIndex, jdoubleArray inTipPartials)
{
    jdouble *tipPartials = env->GetDoubleArrayElements(inTipPartials, NULL);

	setTipPartials(instance, tipIndex, (double *)tipPartials);

    env->ReleaseDoubleArrayElements(inTipPartials, tipPartials, JNI_ABORT);
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
 * Method:    setStateFrequencies
 * Signature: (I[D)V
 */
JNIEXPORT void JNICALL Java_beagle_BeagleJNIWrapper_setStateFrequencies
	(JNIEnv *env, jobject obj, jint instance, jdoubleArray inStateFrequencies)
{
    jdouble *stateFrequencies = env->GetDoubleArrayElements(inStateFrequencies, NULL);

	setStateFrequencies(instance, (double *)stateFrequencies);

    env->ReleaseDoubleArrayElements(inStateFrequencies, stateFrequencies, JNI_ABORT);
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    setEigenDecomposition
 * Signature: (II[[D[[D[D)V
 */
JNIEXPORT void JNICALL Java_beagle_BeagleJNIWrapper_setEigenDecomposition
(JNIEnv *env, jobject obj, jint instance, jint matrixIndex, jobjectArray inEigenVectors, jobjectArray inInvEigenVectors, jdoubleArray inEigenValues)
{

	for (int i = 0; i < STATE_COUNT; i++) {
		jdoubleArray row1 = (jdoubleArray)env->GetObjectArrayElement(inEigenVectors, i);
        env->GetDoubleArrayRegion(row1, 0, STATE_COUNT, Evec[i]);

		jdoubleArray row2 = (jdoubleArray)env->GetObjectArrayElement(inInvEigenVectors, i);
        env->GetDoubleArrayRegion(row2, 0, STATE_COUNT, Ievc[i]);
	}

    jdouble *eigenValues = env->GetDoubleArrayElements(inEigenValues, NULL);

	setEigenDecomposition(instance, matrixIndex, Evec, Ievc, (double *)eigenValues);

    env->ReleaseDoubleArrayElements(inEigenValues, eigenValues, JNI_ABORT);
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    setCategoryRates
 * Signature: (I[D)V
 */
JNIEXPORT void JNICALL Java_beagle_BeagleJNIWrapper_setCategoryRates
	(JNIEnv *env, jobject obj, jint instance, jdoubleArray inCategoryRates)
{
    jdouble *categoryRates = env->GetDoubleArrayElements(inCategoryRates, NULL);

	setCategoryRates(instance, (double *)categoryRates);

    env->ReleaseDoubleArrayElements(inCategoryRates, categoryRates, JNI_ABORT);
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    setCategoryProportions
 * Signature: (I[D)V
 */
JNIEXPORT void JNICALL Java_beagle_BeagleJNIWrapper_setCategoryProportions
	(JNIEnv *env, jobject obj, jint instance, jdoubleArray inCategoryProportions)
{
    jdouble *categoryProportions = env->GetDoubleArrayElements(inCategoryProportions, NULL);

	setCategoryProportions(instance, (double *)categoryProportions);

    env->ReleaseDoubleArrayElements(inCategoryProportions, categoryProportions, JNI_ABORT);
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    calculateProbabilityTransitionMatrices
 * Signature: (I[I[DI)V
 */
JNIEXPORT void JNICALL Java_beagle_BeagleJNIWrapper_calculateProbabilityTransitionMatrices
	(JNIEnv *env, jobject obj, jint instance, jintArray inNodeIndices, jdoubleArray inBranchLengths, jint count)
{
    jint *nodeIndices = env->GetIntArrayElements(inNodeIndices, NULL);
    jdouble *branchLengths = env->GetDoubleArrayElements(inBranchLengths, NULL);

	calculateProbabilityTransitionMatrices(instance, (int *)nodeIndices, (double *)branchLengths, count);

    env->ReleaseDoubleArrayElements(inBranchLengths, branchLengths, JNI_ABORT);
    env->ReleaseIntArrayElements(inNodeIndices, nodeIndices, JNI_ABORT);
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    calculatePartials
 * Signature: (I[I[IIB)V
 */
JNIEXPORT void JNICALL Java_beagle_BeagleJNIWrapper_calculatePartials
	(JNIEnv *env, jobject obj, jint instance, jintArray inOperations, jintArray inDependencies, jint count, jboolean rescale)
{
    jint *operations = env->GetIntArrayElements(inOperations, NULL);
    jint *dependencies = env->GetIntArrayElements(inDependencies, NULL);

	calculatePartials(instance, (int *)operations, (int *)dependencies, count, 0);

    env->ReleaseIntArrayElements(inDependencies, dependencies, JNI_ABORT);
    env->ReleaseIntArrayElements(inOperations, operations, JNI_ABORT);
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    calculateLogLikelihoods
 * Signature: (II[D)V
 */
JNIEXPORT void JNICALL Java_beagle_BeagleJNIWrapper_calculateLogLikelihoods
	(JNIEnv *env, jobject obj, jint instance, jint rootNodeIndex, jdoubleArray outLogLikelihoods)
{
    jdouble *logLikelihoods = env->GetDoubleArrayElements(outLogLikelihoods, NULL);

	calculateLogLikelihoods(instance, rootNodeIndex, logLikelihoods);

    // not using JNI_ABORT flag here because we want the values to be copied back...
    env->ReleaseDoubleArrayElements(outLogLikelihoods, logLikelihoods, 0);
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    storeState
 * Signature: (I)V
 */
JNIEXPORT void JNICALL Java_beagle_BeagleJNIWrapper_storeState
	(JNIEnv *env, jobject obj, jint instance)
{
	storeState(instance);
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    restoreState
 * Signature: (I)V
 */
JNIEXPORT void JNICALL Java_beagle_BeagleJNIWrapper_restoreState
	(JNIEnv *env, jobject obj, jint instance)
{
	restoreState(instance);
}
