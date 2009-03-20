#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "beagle.h"
#include "beagle_BeagleJNIWrapper.h"

	 REAL** Evec;
	 REAL** Ievc;
	 REAL* Eval;

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    initialize
 * Signature: (IIIIII)V
 */
JNIEXPORT void JNICALL Java_beagle_BeagleJNIWrapper_initialize
	(JNIEnv *env, jobject obj, jint nodeCount, jint tipCount, jint patternCount, jint categoryCount, jint matrixCount)
{
	initialize(nodeCount, tipCount, patternCount, categoryCount, matrixCount);

	Evec = (REAL**)malloc(sizeof(REAL*) * STATE_COUNT);
	Ievc = (REAL**)malloc(sizeof(REAL*) * STATE_COUNT);
	for (int i = 0; i < STATE_COUNT; i++) {
	    Evec[i] = (REAL*)malloc(sizeof(REAL) * STATE_COUNT);
	    Ievc[i] = (REAL*)malloc(sizeof(REAL) * STATE_COUNT);
	}
	Eval = (REAL*)malloc(sizeof(REAL) * STATE_COUNT);
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    finalize
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_beagle_BeagleJNIWrapper_finalize
	(JNIEnv *env, jobject obj)
{
	finalize();
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    setTipPartials
 * Signature: (I[D)V
 */
JNIEXPORT void JNICALL Java_beagle_BeagleJNIWrapper_setTipPartials
	(JNIEnv *env, jobject obj, jint tipIndex, jdoubleArray inTipPartials)
{
	jdouble *tipPartialsD = (jdouble*)(*env)->GetPrimitiveArrayCritical(env, inTipPartials, 0);
#if (REAL==double)
	// working with double precision so just pass along the array
	setTipPartials(tipIndex, tipPartialsD);
#else
	// working with single precision so just convert the array
	REAL *tipPartials = (REAL *)malloc(sizeof(REAL) * kPatternCount);

	for (int i = 0; i < kPatternCount; i++) {
		tipPartials[i] = (REAL)tipPartialsD[i];
	}

	setTipPartials(tipIndex, tipPartials);
#endif
	(*env)->ReleasePrimitiveArrayCritical(env, inTipPartials, tipPartialsD, JNI_ABORT);
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    setTipStates
 * Signature: (I[I)V
 */
JNIEXPORT void JNICALL Java_beagle_BeagleJNIWrapper_setTipStates
	(JNIEnv *env, jobject obj, jint tipIndex, jintArray inTipStates)
{
	jint *tipStates = (jint*)(*env)->GetPrimitiveArrayCritical(env, inTipStates, 0);
	setTipStates(tipIndex, (int *)tipStates);
	(*env)->ReleasePrimitiveArrayCritical(env, inTipStates, tipStates, JNI_ABORT);
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    setStateFrequencies
 * Signature: ([D)V
 */
JNIEXPORT void JNICALL Java_beagle_BeagleJNIWrapper_setStateFrequencies
	(JNIEnv *env, jobject obj, jdoubleArray inStateFrequencies)
{
	jdouble *frequenciesD = (jdouble*)(*env)->GetPrimitiveArrayCritical(env, inStateFrequencies, 0);
#if (REAL==double)
	// working with double precision so just pass along the array
	setStateFrequencies(frequenciesD);
#else
	// working with single precision so just convert the array
	static REAL frequencies[STATE_COUNT];

	for (int i = 0; i < kPatternCount; i++) {
		frequencies[i] = (REAL)frequenciesD[i];
	}

	setStateFrequencies(frequencies);
#endif
	(*env)->ReleasePrimitiveArrayCritical(env, inStateFrequencies, frequenciesD, JNI_ABORT);
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    setEigenDecomposition
 * Signature: (I[[D[[D[D)V
 */
JNIEXPORT void JNICALL Java_beagle_BeagleJNIWrapper_setEigenDecomposition
(JNIEnv *env, jobject obj, jint matrixIndex, jobjectArray inEigenVectors, jobjectArray inInvEigenVectors, jdoubleArray inEigenValues)
{
	jdouble *valuesD = (jdouble*)(*env)->GetPrimitiveArrayCritical(env, inEigenValues, 0);
#if (REAL==double)
	for (int i = 0; i < STATE_COUNT; i++) {
		jdoubleArray row1 = (jdoubleArray)(*env)->GetObjectArrayElement(env, inEigenVectors, i);
        (*env)->GetDoubleArrayRegion(env, row1, 0, STATE_COUNT, Evec[i]);

		jdoubleArray row2 = (jdoubleArray)(*env)->GetObjectArrayElement(env, inEigenVectors, i);
        (*env)->GetDoubleArrayRegion(env, row1, 0, STATE_COUNT, Ievc[i]);
	}

	setEigenDecomposition(matrixIndex, Evec, Ievc, (REAL*)valuesD);
#else
	for (int i = 0; i < STATE_COUNT; i++) {
		jdoubleArray row1 = (jdoubleArray)(*env)->GetObjectArrayElement(env, inEigenVectors, i);
		jdouble *elements1 = (*env)->GetDoubleArrayElements(env, row1, 0);

		jdoubleArray row2 = (jdoubleArray)(*env)->GetObjectArrayElement(env, inEigenVectors, i);
		jdouble *elements2 = (*env)->GetDoubleArrayElements(env, row2, 0);

		for(int j = 0; j < STATE_COUNT; j++) {
			Evec[i][j] = (REAL)elements1[j];
			Ievc[i][j] = (REAL)elements2[j];
		}

        (*env)->ReleaseDoubleArrayElements(env, row2, elements2, 0);
        (*env)->ReleaseDoubleArrayElements(env, row1, elements1, 0);

		Eval[i] = (REAL)valuesD[i];
	}

	setEigenDecomposition(matrixIndex, Evec, Ievc, Eval);
#endif

	(*env)->ReleasePrimitiveArrayCritical(env, inEigenValues, valuesD, JNI_ABORT);

}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    setCategoryRates
 * Signature: ([D)V
 */
JNIEXPORT void JNICALL Java_beagle_BeagleJNIWrapper_setCategoryRates
	(JNIEnv *env, jobject obj, jdoubleArray inCategoryRates)
{
	jdouble *categoryRatesD = (jdouble*)(*env)->GetPrimitiveArrayCritical(env, inCategoryRates, 0);
#if (REAL==double)
	// working with double precision so just pass along the array
	setCategoryRates(categoryRatesD);
#else
	// working with single precision so just convert the array
	REAL *categoryRates = (REAL *)malloc(sizeof(REAL) * kCategoryCount);

	for (int i = 0; i < kCategoryCount; i++) {
		categoryRates[i] = (REAL)categoryRatesD[i];
	}

	setCategoryRates(categoryRates);
#endif
	(*env)->ReleasePrimitiveArrayCritical(env, inCategoryRates, categoryRatesD, JNI_ABORT);
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    setCategoryProportions
 * Signature: ([D)V
 */
JNIEXPORT void JNICALL Java_beagle_BeagleJNIWrapper_setCategoryProportions
	(JNIEnv *env, jobject obj, jdoubleArray inCategoryProportions)
{
	jdouble *categoryProportionsD = (jdouble*)(*env)->GetPrimitiveArrayCritical(env, inCategoryProportions, 0);

#if (REAL==double)
	// working with double precision so just pass along the array
	setCategoryProportions(categoryProportionsD);
#else
	// working with single precision so just convert the array
	REAL *categoryProportions = (REAL *)malloc(sizeof(REAL) * kCategoryCount);

	for (int i = 0; i < kCategoryCount; i++) {
		categoryProportions[i] = (REAL)categoryProportionsD[i];
	}

	setCategoryProportions(categoryProportions);
#endif

	(*env)->ReleasePrimitiveArrayCritical(env, inCategoryProportions, categoryProportionsD, JNI_ABORT);
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    calculateProbabilityTransitionMatrices
 * Signature: (ID)V
 */
JNIEXPORT void JNICALL Java_beagle_BeagleJNIWrapper_calculateProbabilityTransitionMatrices
	(JNIEnv *env, jobject obj, jint nodeIndex, jdouble branchLength)
{
	calculateProbabilityTransitionMatrices(nodeIndex, (REAL)branchLength);
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    calculatePartials
 * Signature: ([I[II)V
 */
JNIEXPORT void JNICALL Java_beagle_BeagleJNIWrapper_calculatePartials
	(JNIEnv *env, jobject obj, jintArray inOperations, jintArray inDependencies, jint operationCount)
{
	jint *operations = (jint*)(*env)->GetPrimitiveArrayCritical(env, inOperations, 0);
	jint *dependencies = (jint*)(*env)->GetPrimitiveArrayCritical(env, inDependencies, 0);
	calculatePartials((int *)operations, (int *)dependencies, operationCount);
	(*env)->ReleasePrimitiveArrayCritical(env, inDependencies, dependencies, JNI_ABORT);
	(*env)->ReleasePrimitiveArrayCritical(env, inOperations, operations, JNI_ABORT);
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    calculateLogLikelihoods
 * Signature: (I[D)V
 */
JNIEXPORT void JNICALL Java_beagle_BeagleJNIWrapper_calculateLogLikelihoods
	(JNIEnv *env, jobject obj, jint rootNodeIndex, jdoubleArray outLogLikelihoods)
{
	jdouble *logLikelihoodsD = (jdouble*)(*env)->GetPrimitiveArrayCritical(env, outLogLikelihoods, 0);

#if (REAL==double)
	// working with double precision so just pass along the array
	calculateLogLikelihoods(rootNodeIndex, logLikelihoodsD);
#else
	// working with single precision so just convert the array
	REAL *logLikelihoods = (REAL *)malloc(sizeof(REAL) * kPatternCount);

	calculateLogLikelihoods(rootNodeIndex, logLikelihoods);

	for (int i = 0; i < kPatternCount; i++) {
		logLikelihoodsD[i] = (double)logLikelihoods[i];
	}
#endif

	(*env)->ReleasePrimitiveArrayCritical(env, outLogLikelihoods, logLikelihoodsD, JNI_ABORT);
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    storeState
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_beagle_BeagleJNIWrapper_storeState
	(JNIEnv *env, jobject obj)
{
	storeState();
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    restoreState
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_beagle_BeagleJNIWrapper_restoreState
	(JNIEnv *env, jobject obj)
{
	restoreState();
}
