#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "beagle.h"
#include "beagle_BeagleJNIWrapper.h"

// Some temporary arrays used to convert floating point types
REAL** Evec;
REAL** Ievc;
REAL* Eval;

REAL *categoryValues;
REAL *branchValues;
REAL *logLikelihoodValues;

int kCategoryCount;
int kPatternCount;
/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    initialize
 * Signature: (IIIIII)V
 */
JNIEXPORT void JNICALL Java_beagle_BeagleJNIWrapper_initialize
	(JNIEnv *env, jobject obj, jint nodeCount, jint tipCount, jint patternCount, jint categoryCount, jint matrixCount)
{
	initialize(nodeCount, tipCount, patternCount, categoryCount, matrixCount);

    kCategoryCount = categoryCount;
    kPatternCount = patternCount;

	Evec = (REAL**)malloc(sizeof(REAL*) * STATE_COUNT);
	Ievc = (REAL**)malloc(sizeof(REAL*) * STATE_COUNT);
	for (int i = 0; i < STATE_COUNT; i++) {
	    Evec[i] = (REAL*)malloc(sizeof(REAL) * STATE_COUNT);
	    Ievc[i] = (REAL*)malloc(sizeof(REAL) * STATE_COUNT);
	}
	Eval = (REAL*)malloc(sizeof(REAL) * STATE_COUNT);

	categoryValues = (REAL *)malloc(sizeof(REAL) * categoryCount);
	branchValues = (REAL *)malloc(sizeof(REAL) * nodeCount);

	logLikelihoodValues = (REAL *)malloc(sizeof(REAL) * patternCount);

}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    finalize
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_beagle_BeagleJNIWrapper_finalize
	(JNIEnv *env, jobject obj)
{
	for (int i = 0; i < STATE_COUNT; i++) {
	    free(Evec[i]);
	    free(Ievc[i]);
	}
	free(Evec);
	free(Ievc);
	free(Eval);

	free(categoryValues);
    free(branchValues);
    free(logLikelihoodValues);

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

	// this function is only called to set up the data so we can malloc a temp array
	REAL *tipPartials = (REAL *)malloc(sizeof(REAL) * kPatternCount);

	for (int i = 0; i < kPatternCount; i++) {
		tipPartials[i] = (REAL)tipPartialsD[i];
	}

	setTipPartials(tipIndex, tipPartials);

	free(tipPartials);

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
	// A simple temporary array of fixed size so statically allocate it
	static REAL frequencies[STATE_COUNT];

#if (REAL==double)
	// working with double precision so just pass along the array
    (*env)->GetDoubleArrayRegion(env, inStateFrequencies, 0, STATE_COUNT, frequencies);

#else
	// working with single precision so just convert the array
	jdouble *frequenciesD = (jdouble*)(*env)->GetPrimitiveArrayCritical(env, inStateFrequencies, 0);


	for (int i = 0; i < kPatternCount; i++) {
		frequencies[i] = (REAL)frequenciesD[i];
	}

	(*env)->ReleasePrimitiveArrayCritical(env, inStateFrequencies, frequenciesD, JNI_ABORT);
#endif

	setStateFrequencies(frequencies);
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    setEigenDecomposition
 * Signature: (I[[D[[D[D)V
 */
JNIEXPORT void JNICALL Java_beagle_BeagleJNIWrapper_setEigenDecomposition
(JNIEnv *env, jobject obj, jint matrixIndex, jobjectArray inEigenVectors, jobjectArray inInvEigenVectors, jdoubleArray inEigenValues)
{
#if (REAL==double)
    (*env)->GetDoubleArrayRegion(env, inEigenValues, 0, STATE_COUNT, Eval);

	for (int i = 0; i < STATE_COUNT; i++) {
		jdoubleArray row1 = (jdoubleArray)(*env)->GetObjectArrayElement(env, inEigenVectors, i);
        (*env)->GetDoubleArrayRegion(env, row1, 0, STATE_COUNT, Evec[i]);

		jdoubleArray row2 = (jdoubleArray)(*env)->GetObjectArrayElement(env, inEigenVectors, i);
        (*env)->GetDoubleArrayRegion(env, row1, 0, STATE_COUNT, Ievc[i]);
	}

#else
	jdouble *valuesD = (jdouble*)(*env)->GetPrimitiveArrayCritical(env, inEigenValues, 0);
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
	(*env)->ReleasePrimitiveArrayCritical(env, inEigenValues, valuesD, JNI_ABORT);

#endif

	setEigenDecomposition(matrixIndex, Evec, Ievc, Eval);

}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    setCategoryRates
 * Signature: ([D)V
 */
JNIEXPORT void JNICALL Java_beagle_BeagleJNIWrapper_setCategoryRates
	(JNIEnv *env, jobject obj, jdoubleArray inCategoryRates)
{
#if (REAL==double)
	// working with double precision so just pass along the array
    (*env)->GetDoubleArrayRegion(env, inCategoryRates, 0, kCategoryCount, categoryValues);
#else
	// working with single precision so just convert the array
	jdouble *categoryRatesD = (jdouble*)(*env)->GetPrimitiveArrayCritical(env, inCategoryRates, 0);

    // using categoryValues which is a global temporary array of categoryCount size
	for (int i = 0; i < kCategoryCount; i++) {
		categoryValues[i] = (REAL)categoryRatesD[i];
	}

	(*env)->ReleasePrimitiveArrayCritical(env, inCategoryRates, categoryRatesD, JNI_ABORT);
#endif

	setCategoryRates(categoryValues);
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    setCategoryProportions
 * Signature: ([D)V
 */
JNIEXPORT void JNICALL Java_beagle_BeagleJNIWrapper_setCategoryProportions
	(JNIEnv *env, jobject obj, jdoubleArray inCategoryProportions)
{

#if (REAL==double)
	// working with double precision so just pass along the array
    (*env)->GetDoubleArrayRegion(env, inCategoryProportions, 0, kCategoryCount, categoryValues);
#else
	// working with single precision so just convert the array
	jdouble *categoryProportionsD = (jdouble*)(*env)->GetPrimitiveArrayCritical(env, inCategoryProportions, 0);

    // using categoryValues which is a global temporary array of categoryCount size
	for (int i = 0; i < kCategoryCount; i++) {
		categoryValues[i] = (REAL)categoryRatesD[i];
	}

	(*env)->ReleasePrimitiveArrayCritical(env, inCategoryProportions, categoryProportionsD, JNI_ABORT);
#endif

	setCategoryRates(categoryValues);
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    calculateProbabilityTransitionMatrices
 * Signature: ([I[DI)V
 */
JNIEXPORT void JNICALL Java_beagle_BeagleJNIWrapper_calculateProbabilityTransitionMatrices
	(JNIEnv *env, jobject obj, jintArray inNodeIndices, jdoubleArray inBranchLengths, jint count)
{
    jint *nodeIndices = (*env)->GetIntArrayElements(env, inNodeIndices, 0);

#if (REAL==double)
     (*env)->GetDoubleArrayRegion(env, inBranchLengths, 0, count, branchValues);

	// working with double precision so just pass along the array
	calculateProbabilityTransitionMatrices((int *)nodeIndices, branchValues, count);

#else
	// working with single precision so just convert the array
	jdouble *branchLengthsD = (jdouble*)(*env)->GetPrimitiveArrayCritical(env, inBranchLengths, 0);

    // using branchValues which is a global temporary array of nodeCount size
	for (int i = 0; i < count; i++) {
		branchValues[i] = (REAL)branchLengthsD[i];
	}

	(*env)->ReleasePrimitiveArrayCritical(env, inBranchLengths, branchLengthsD, JNI_ABORT);

	calculateProbabilityTransitionMatrices((int *)nodeIndices, branchValues, count);
#endif

    (*env)->ReleaseIntArrayElements(env, inNodeIndices, nodeIndices, 0);
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    calculatePartials
 * Signature: ([I[II)V
 */
JNIEXPORT void JNICALL Java_beagle_BeagleJNIWrapper_calculatePartials
	(JNIEnv *env, jobject obj, jintArray inOperations, jintArray inDependencies, jint count)
{
    jint *operations = (*env)->GetIntArrayElements(env, inOperations, 0);
    jint *dependencies = (*env)->GetIntArrayElements(env, inDependencies, 0);

	calculatePartials((int *)operations, (int *)dependencies, count);

    (*env)->ReleaseIntArrayElements(env, inDependencies, dependencies, 0);
    (*env)->ReleaseIntArrayElements(env, inOperations, operations, 0);
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    calculateLogLikelihoods
 * Signature: (I[D)V
 */
JNIEXPORT void JNICALL Java_beagle_BeagleJNIWrapper_calculateLogLikelihoods
	(JNIEnv *env, jobject obj, jint rootNodeIndex, jdoubleArray outLogLikelihoods)
{

#if (REAL==double)
    jdouble *logLikelihoodsD = (*env)->GetDoubleArrayElements(env, outLogLikelihoods, 0);

	// working with double precision so just pass along the array
	calculateLogLikelihoods(rootNodeIndex, logLikelihoodsD);

    (*env)->ReleaseDoubleArrayElements(env, outLogLikelihoods, logLikelihoodsD, 0);
#else
	// working with single precision so just convert the array

    // using logLikelihoodValues which is a global temporary array of patternCount size
	calculateLogLikelihoods(rootNodeIndex, logLikelihoodValues);

	jdouble *logLikelihoodsD = (jdouble*)(*env)->GetPrimitiveArrayCritical(env, outLogLikelihoods, 0);

	for (int i = 0; i < kPatternCount; i++) {
		logLikelihoodsD[i] = (double)logLikelihoodValues[i];
	}

	(*env)->ReleasePrimitiveArrayCritical(env, outLogLikelihoods, logLikelihoodsD, JNI_ABORT);
#endif

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
