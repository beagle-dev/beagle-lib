#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <jni.h>

#include "libhmsbeagle/beagle.h"
#include "libhmsbeagle/JNI/beagle_BeagleJNIWrapper.h"

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    getResourceList
 * Signature: ()[Lbeagle/ResourceDetails;
 */
JNIEXPORT jobjectArray JNICALL Java_beagle_BeagleJNIWrapper_getResourceList
  (JNIEnv *env, jobject obj)
{
	ResourceList* rl = getResourceList();
    // @todo Need to convert the resource list into appropriate Java objects
	if (rl != NULL) {
        return NULL;
	} else {
        return NULL;
	}
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    createInstance
 * Signature: (IIIIIIIII[IIII)I
 */
JNIEXPORT jint JNICALL Java_beagle_BeagleJNIWrapper_createInstance
	(JNIEnv *env, jobject obj, jint tipCount, jint partialsBufferCount, jint compactBufferCount,
	jint stateCount, jint patternCount, jint eigenBufferCount, jint matrixBufferCount, jint
	 categoryCount, jint scaleBufferCount, jintArray inResourceList, jint resourceCount, jint preferenceFlags, jint requirementFlags)
{
    jint *resourceList = env->GetIntArrayElements(inResourceList, NULL);
    
 	jint instance = (jint)createInstance(tipCount,
	                                partialsBufferCount,
	                                compactBufferCount,
	                                stateCount,
	                                patternCount,
	                                eigenBufferCount,
	                                matrixBufferCount,
					categoryCount,
					scaleBufferCount,
	                                (int *)resourceList,
	                                resourceCount,
	                                preferenceFlags,
	                                requirementFlags);

    env->ReleaseIntArrayElements(inResourceList, resourceList, JNI_ABORT);

    return instance;
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    initializeInstance
 * Signature: (I[Lbeagle/InstanceDetails;)I
 */
JNIEXPORT jint JNICALL Java_beagle_BeagleJNIWrapper_initializeInstance
  (JNIEnv *env, jobject obj, jint instance, jobjectArray outInstanceDetails)
{
    // @todo Ignoring outInstanceDetails for the moment
    jint errCode = (jint)initializeInstance(instance, NULL);
    return errCode;
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    finalize
 * Signature: (I)I
 */
JNIEXPORT jint JNICALL Java_beagle_BeagleJNIWrapper_finalize
  (JNIEnv *env, jobject obj, jint instance)
{
	jint errCode = (jint)finalize(instance);
    return errCode;
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    setTipStates
 * Signature: (II[I)I
 */
JNIEXPORT jint JNICALL Java_beagle_BeagleJNIWrapper_setTipStates
(JNIEnv *env, jobject obj, jint instance, jint tipIndex, jintArray inTipStates)
{
    jint *tipStates = env->GetIntArrayElements(inTipStates, NULL);
    
	jint errCode = (jint)setTipStates(instance, tipIndex, (int *)tipStates);
    
    env->ReleaseIntArrayElements(inTipStates, tipStates, JNI_ABORT);
    return errCode;
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    setTipPartials
 * Signature: (II[D)I
 */
JNIEXPORT jint JNICALL Java_beagle_BeagleJNIWrapper_setTipPartials
(JNIEnv *env, jobject obj, jint instance, jint tipIndex, jdoubleArray inPartials)
{
    jdouble *partials = env->GetDoubleArrayElements(inPartials, NULL);
    
	jint errCode = (jint)setTipPartials(instance, tipIndex, (double *)partials);
    
    env->ReleaseDoubleArrayElements(inPartials, partials, JNI_ABORT);
    return errCode;
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

	jint errCode = (jint)setPartials(instance, bufferIndex, (double *)partials);

    env->ReleaseDoubleArrayElements(inPartials, partials, JNI_ABORT);
    return errCode;
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    getPartials
 * Signature: (III[D)I
 */
JNIEXPORT jint JNICALL Java_beagle_BeagleJNIWrapper_getPartials
(JNIEnv *env, jobject obj, jint instance, jint bufferIndex, jint scaleIndex, jdoubleArray outPartials)
{
    jdouble *partials = env->GetDoubleArrayElements(outPartials, NULL);

    jint errCode = getPartials(instance, bufferIndex, scaleIndex, (double *)partials);

    // not using JNI_ABORT flag here because we want the values to be copied back...
    env->ReleaseDoubleArrayElements(outPartials, partials, 0);
    return errCode;
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    setEigenDecomposition
 * Signature: (II[D[D[D)I
 */
JNIEXPORT jint JNICALL Java_beagle_BeagleJNIWrapper_setEigenDecomposition
(JNIEnv *env, jobject obj, jint instance, jint eigenIndex, jdoubleArray inEigenVectors, jdoubleArray inInvEigenVectors, jdoubleArray inEigenValues)
{
    jdouble *eigenVectors = env->GetDoubleArrayElements(inEigenVectors, NULL);
    jdouble *invEigenVectors = env->GetDoubleArrayElements(inInvEigenVectors, NULL);
    jdouble *eigenValues = env->GetDoubleArrayElements(inEigenValues, NULL);

	jint errCode = (jint)setEigenDecomposition(instance, eigenIndex, (double *)eigenVectors, (double *)invEigenVectors, (double *)eigenValues);

    env->ReleaseDoubleArrayElements(inEigenValues, eigenValues, JNI_ABORT);
    env->ReleaseDoubleArrayElements(inInvEigenVectors, invEigenVectors, JNI_ABORT);
    env->ReleaseDoubleArrayElements(inEigenVectors, eigenVectors, JNI_ABORT);

    return errCode;
}


/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    setCategoryRates
 * Signature: (I[D)I
 */
JNIEXPORT jint JNICALL Java_beagle_BeagleJNIWrapper_setCategoryRates
  (JNIEnv *env, jobject obj, jint instance, jdoubleArray inCategoryRates)
{
    jdouble *categoryRates = env->GetDoubleArrayElements(inCategoryRates, NULL);

	jint errCode = (jint)setCategoryRates(instance, (double *)categoryRates);

    env->ReleaseDoubleArrayElements(inCategoryRates, categoryRates, JNI_ABORT);

    return errCode;
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

	jint errCode = (jint)setTransitionMatrix(instance, matrixIndex, (double *)matrix);

    env->ReleaseDoubleArrayElements(inMatrix, matrix, JNI_ABORT);

    return errCode;
}


/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    updateTransitionMatrices
 * Signature: (II[I[I[I[DI)I
 */
JNIEXPORT jint JNICALL Java_beagle_BeagleJNIWrapper_updateTransitionMatrices
  (JNIEnv *env, jobject obj, jint instance, jint eigenIndex, jintArray inProbabilityIndices, jintArray inFirstDerivativeIndices, jintArray inSecondDervativeIndices, jdoubleArray inEdgeLengths, jint count)
{
    jint errCode;

    jint *probabilityIndices = env->GetIntArrayElements(inProbabilityIndices, NULL);
    if (inFirstDerivativeIndices == NULL) {
        jdouble *edgeLengths = env->GetDoubleArrayElements(inEdgeLengths, NULL);

        errCode = (jint)updateTransitionMatrices(instance, eigenIndex, (int *)probabilityIndices, NULL, NULL, (double *)edgeLengths, count);

        env->ReleaseDoubleArrayElements(inEdgeLengths, edgeLengths, JNI_ABORT);
    } else {
        jint *firstDerivativeIndices = env->GetIntArrayElements(inFirstDerivativeIndices, NULL);
        jint *secondDervativeIndices = env->GetIntArrayElements(inSecondDervativeIndices, NULL);
        jdouble *edgeLengths = env->GetDoubleArrayElements(inEdgeLengths, NULL);

        errCode = (jint)updateTransitionMatrices(instance, eigenIndex, (int *)probabilityIndices, (int *)firstDerivativeIndices, (int *)secondDervativeIndices, (double *)edgeLengths, count);

        env->ReleaseDoubleArrayElements(inEdgeLengths, edgeLengths, JNI_ABORT);
        env->ReleaseIntArrayElements(inSecondDervativeIndices, secondDervativeIndices, JNI_ABORT);
        env->ReleaseIntArrayElements(inFirstDerivativeIndices, firstDerivativeIndices, JNI_ABORT);
    }
    env->ReleaseIntArrayElements(inProbabilityIndices, probabilityIndices, JNI_ABORT);

    return errCode;
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    updatePartials
 * Signature: ([II[III)I
 */
JNIEXPORT jint JNICALL Java_beagle_BeagleJNIWrapper_updatePartials
  (JNIEnv *env, jobject obj, jintArray inInstances, jint instanceCount, jintArray inOperations, jint operationCount, jint cumulativeScalingIndex)
{
    jint *instances = env->GetIntArrayElements(inInstances, NULL);
    jint *operations = env->GetIntArrayElements(inOperations, NULL);

	jint errCode = (jint)updatePartials((int *)instances, instanceCount, (int *)operations, operationCount, cumulativeScalingIndex);

    env->ReleaseIntArrayElements(inOperations, operations, JNI_ABORT);
    env->ReleaseIntArrayElements(inInstances, instances, JNI_ABORT);

    return errCode;
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    waitForPartials
 * Signature: ([II[II)I
 */
JNIEXPORT jint JNICALL Java_beagle_BeagleJNIWrapper_waitForPartials
  (JNIEnv *env, jobject obj, jintArray inInstances, jint instanceCount, jintArray inDestinationPartials, jint destinationPartialsCount)
{
    jint *instances = env->GetIntArrayElements(inInstances, NULL);
    jint *destinationPartials = env->GetIntArrayElements(inDestinationPartials, NULL);

    jint errCode = (jint)waitForPartials((int *)instances, instanceCount, (int *)destinationPartials, destinationPartialsCount);

    env->ReleaseIntArrayElements(inDestinationPartials, destinationPartials, JNI_ABORT);
    env->ReleaseIntArrayElements(inInstances, instances, JNI_ABORT);

    return errCode;
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    accumulateScaleFactors
 * Signature: (I[III)I
 */
JNIEXPORT jint JNICALL Java_beagle_BeagleJNIWrapper_accumulateScaleFactors
  (JNIEnv *env, jobject obj, jint instance, jintArray inScaleIndices, jint count, jint cumulativeScalingIndex) {
	
	jint *scaleIndices = env->GetIntArrayElements(inScaleIndices, NULL);
	jint errCode = (jint)accumulateScaleFactors(instance, (int*)scaleIndices, count, cumulativeScalingIndex);
	env->ReleaseIntArrayElements(inScaleIndices, scaleIndices, JNI_ABORT);

	return errCode;
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    removeScaleFactors
 * Signature: (I[III)I
 */
JNIEXPORT jint JNICALL Java_beagle_BeagleJNIWrapper_removeScaleFactors
(JNIEnv *env, jobject obj, jint instance, jintArray inScaleIndices, jint count, jint cumulativeScalingIndex) {
	
	jint *scaleIndices = env->GetIntArrayElements(inScaleIndices, NULL);
	jint errCode = (jint)accumulateScaleFactors(instance, (int*)scaleIndices, count, cumulativeScalingIndex);
	env->ReleaseIntArrayElements(inScaleIndices, scaleIndices, JNI_ABORT);
    
	return errCode;
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    resetScaleFactors
 * Signature: (II)I
 */
JNIEXPORT jint JNICALL Java_beagle_BeagleJNIWrapper_resetScaleFactors
(JNIEnv *env, jobject obj, jint instance, jint cumulativeScalingIndex) {
	
	jint errCode = (jint)resetScaleFactors(instance, cumulativeScalingIndex);
	return errCode;
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    calculateRootLogLikelihoods
 * Signature: (I[I[D[DI[D)I
 */
JNIEXPORT jint JNICALL Java_beagle_BeagleJNIWrapper_calculateRootLogLikelihoods
  (JNIEnv *env, jobject obj, jint instance, jintArray inBufferIndices, jdoubleArray inWeights, jdoubleArray inStateFrequencies, jintArray inScalingIndices, jint count, jdoubleArray outLogLikelihoods)
{
    jint *bufferIndices = env->GetIntArrayElements(inBufferIndices, NULL);
    jdouble *weights = env->GetDoubleArrayElements(inWeights, NULL);
    jdouble *stateFrequencies = env->GetDoubleArrayElements(inStateFrequencies, NULL);
    jint *scalingIndices = env->GetIntArrayElements(inScalingIndices, NULL);
    //    jint *scalingCount = env->GetIntArrayElements(inScalingCount, NULL);
    jdouble *logLikelihoods = env->GetDoubleArrayElements(outLogLikelihoods, NULL);

	jint errCode = (jint)calculateRootLogLikelihoods(instance, (int *)bufferIndices, (double *)weights,
	                                                (double *)stateFrequencies, (int *)scalingIndices,
	                                                 count, (double *)logLikelihoods);

    // not using JNI_ABORT flag here because we want the values to be copied back...
    env->ReleaseDoubleArrayElements(outLogLikelihoods, logLikelihoods, 0);

    //    env->ReleaseIntArrayElements(inScalingCount, scalingCount, JNI_ABORT);
    env->ReleaseIntArrayElements(inScalingIndices, scalingIndices, JNI_ABORT);

    env->ReleaseDoubleArrayElements(inStateFrequencies, stateFrequencies, JNI_ABORT);
    env->ReleaseDoubleArrayElements(inWeights, weights, JNI_ABORT);
    env->ReleaseIntArrayElements(inBufferIndices, bufferIndices, JNI_ABORT);

    return errCode;
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    calculateEdgeLogLikelihoods
 * Signature: (I[I[I[I[I[I[D[D[I[II[D[D[D)I
 */
JNIEXPORT jint JNICALL Java_beagle_BeagleJNIWrapper_calculateEdgeLogLikelihoods
  (JNIEnv *env, jobject obj, jint instance, jintArray inParentBufferIndices, jintArray inChildBufferIndices,
        jintArray inProbabilityIndices, jintArray inFirstDerivativeIndices, jintArray inSecondDerivativeIndices,
        jdoubleArray inWeights, jdoubleArray inStateFrequencies, jintArray inScalingIndices,
        jint count, jdoubleArray outLogLikelihoods, jdoubleArray outFirstDerivatives, jdoubleArray outSecondDerivatives)
{
    jint *parentBufferIndices = env->GetIntArrayElements(inParentBufferIndices, NULL);
    jint *childBufferIndices = env->GetIntArrayElements(inChildBufferIndices, NULL);
    jint *probabilityIndices = env->GetIntArrayElements(inProbabilityIndices, NULL);
    jint *firstDerivativeIndices = env->GetIntArrayElements(inFirstDerivativeIndices, NULL);
    jint *secondDerivativeIndices = env->GetIntArrayElements(inSecondDerivativeIndices, NULL);

    jdouble *weights = env->GetDoubleArrayElements(inWeights, NULL);
    jdouble *stateFrequencies = env->GetDoubleArrayElements(inStateFrequencies, NULL);
    jint *scalingIndices = env->GetIntArrayElements(inScalingIndices, NULL);
    //    jint *scalingCount = env->GetIntArrayElements(inScalingCount, NULL);
    jdouble *logLikelihoods = env->GetDoubleArrayElements(outLogLikelihoods, NULL);
    jdouble *firstDerivatives = env->GetDoubleArrayElements(outFirstDerivatives, NULL);
    jdouble *secondDerivatives = env->GetDoubleArrayElements(outSecondDerivatives, NULL);

	jint errCode = (jint)calculateEdgeLogLikelihoods(instance, (int *)parentBufferIndices, (int *)childBufferIndices,
	                                                    (int *)probabilityIndices, (int *)firstDerivativeIndices,
	                                                    (int *)secondDerivativeIndices, (double *)weights,
	                                                    (double *)stateFrequencies,
							 (int *)scalingIndices,// (int *)scalingCount,
	                                                    count, (double *)logLikelihoods, (double *)firstDerivatives,
	                                                    (double *)secondDerivatives);

    // not using JNI_ABORT flag here because we want the values to be copied back...
    env->ReleaseDoubleArrayElements(outSecondDerivatives, secondDerivatives, 0);
    env->ReleaseDoubleArrayElements(outFirstDerivatives, firstDerivatives, 0);
    env->ReleaseDoubleArrayElements(outLogLikelihoods, logLikelihoods, 0);

    //    env->ReleaseIntArrayElements(inScalingCount, scalingCount, JNI_ABORT);
    env->ReleaseIntArrayElements(inScalingIndices, scalingIndices, JNI_ABORT);

    env->ReleaseDoubleArrayElements(inStateFrequencies, stateFrequencies, JNI_ABORT);
    env->ReleaseDoubleArrayElements(inWeights, weights, JNI_ABORT);

    env->ReleaseIntArrayElements(inSecondDerivativeIndices, secondDerivativeIndices, JNI_ABORT);
    env->ReleaseIntArrayElements(inFirstDerivativeIndices, firstDerivativeIndices, JNI_ABORT);
    env->ReleaseIntArrayElements(inProbabilityIndices, probabilityIndices, JNI_ABORT);
    env->ReleaseIntArrayElements(inChildBufferIndices, childBufferIndices, JNI_ABORT);
    env->ReleaseIntArrayElements(inParentBufferIndices, parentBufferIndices, JNI_ABORT);

    return errCode;
}
