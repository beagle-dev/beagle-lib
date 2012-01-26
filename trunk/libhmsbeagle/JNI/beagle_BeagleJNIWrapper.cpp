#ifdef _WIN32
#include "libhmsbeagle/JNI/winjni.h"
#endif

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
	BeagleResourceList* rl = beagleGetResourceList();

	if (rl == NULL) {
	    return NULL;
    }
    
	jclass objClass = env->FindClass("beagle/ResourceDetails");
	if (objClass == NULL) {
		printf("NULL returned in FindClass: can't find class: beagle/ResourceDetails\n");
		return NULL;
	}

	jmethodID constructorMethodID = env->GetMethodID(objClass, "<init>","(I)V");
	if (constructorMethodID == NULL) {
		printf("NULL returned in FindClass: can't find constructor for class: beagle/ResourceDetails\n");
		return NULL;
    }

	jmethodID setNameMethodID = env->GetMethodID(objClass, "setName", "(Ljava/lang/String;)V");
	if (setNameMethodID == NULL) {
		printf("NULL returned in FindClass: can't find 'setName' method in class: beagle/ResourceDetails\n");
		return NULL;
    }

	jmethodID setDescriptionID = env->GetMethodID(objClass, "setDescription", "(Ljava/lang/String;)V");
	if (setDescriptionID == NULL) {
		printf("NULL returned in FindClass: can't find 'setDescription' method in class: beagle/ResourceDetails\n");
		return NULL;
    }

	jmethodID setFlagsMethodID = env->GetMethodID(objClass, "setFlags", "(J)V");
	if (setFlagsMethodID == NULL) {
		printf("NULL returned in FindClass: can't find 'setFlags' method in class: beagle/ResourceDetails\n");
		return NULL;
    }

    jobjectArray resourceArray = env->NewObjectArray(rl->length, objClass, NULL);

	for (int i = 0; i < rl->length; i++) {
	    jobject resourceObj = env->NewObject(objClass, constructorMethodID, i);

	    jstring jString = env->NewStringUTF(rl->list[i].name);
	    env->CallVoidMethod(resourceObj, setNameMethodID, jString);

	    jString = env->NewStringUTF(rl->list[i].description);
    	env->CallVoidMethod(resourceObj, setDescriptionID, jString);

	    env->CallVoidMethod(resourceObj, setFlagsMethodID, rl->list[i].supportFlags);

        env->SetObjectArrayElement(resourceArray, i, resourceObj);
	}
	
	return resourceArray;
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    createInstance
 * Signature: (IIIIIIIII[IIJJLbeagle/InstanceDetails;)I
 */
JNIEXPORT jint JNICALL Java_beagle_BeagleJNIWrapper_createInstance
    (JNIEnv *env, jobject obj, jint tipCount, jint partialsBufferCount, jint compactBufferCount,
    jint stateCount, jint patternCount, jint eigenBufferCount, jint matrixBufferCount, jint
     categoryCount, jint scaleBufferCount, 
	 jintArray inResourceList, jint resourceCount, jlong preferenceFlags, jlong requirementFlags,
	 jobject outInstanceDetails)
{
    BeagleInstanceDetails instanceDetails;

    jint *resourceList = NULL;
    if (inResourceList != NULL)
        resourceList = env->GetIntArrayElements(inResourceList, NULL);
    
     jint instance = (jint)beagleCreateInstance(tipCount,
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
                                    requirementFlags,
									&instanceDetails);
    
    if(inResourceList != NULL)
        env->ReleaseIntArrayElements(inResourceList, resourceList, JNI_ABORT);
    
	if (instance >= 0) {
		jclass objClass = env->FindClass("beagle/InstanceDetails");
		if (objClass == NULL) {
			printf("NULL returned in FindClass: can't find class: beagle/InstanceDetails\n");
			return BEAGLE_ERROR_GENERAL;
		}
		
		jmethodID setResourceNumberMethodID = env->GetMethodID(objClass, "setResourceNumber", "(I)V");
		if (setResourceNumberMethodID == NULL) {
			printf("NULL returned in FindClass: can't find 'setResourceNumber' method in class: beagle/InstanceDetails\n");
			return BEAGLE_ERROR_GENERAL;
		}
		
		jmethodID setFlagsMethodID = env->GetMethodID(objClass, "setFlags", "(J)V");
		if (setFlagsMethodID == NULL) {
			printf("NULL returned in FindClass: can't find 'setFlags' method in class: beagle/InstanceDetails\n");
			return BEAGLE_ERROR_GENERAL;
		}
		
		env->CallVoidMethod(outInstanceDetails, setResourceNumberMethodID, instanceDetails.resourceNumber);
		env->CallVoidMethod(outInstanceDetails, setFlagsMethodID, instanceDetails.flags);
	}

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
	jint errCode = (jint)beagleFinalizeInstance(instance);
    return errCode;
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    setPatternWeights
 * Signature: (I[D)I
 */
JNIEXPORT jint JNICALL Java_beagle_BeagleJNIWrapper_setPatternWeights
(JNIEnv *env, jobject obj, jint instance, jdoubleArray inPatternWeights) 
{
    jdouble *patternWeights = env->GetDoubleArrayElements(inPatternWeights, NULL);
    
	jint errCode = (jint)beagleSetPatternWeights(instance, (double *)patternWeights);
    
    env->ReleaseDoubleArrayElements(inPatternWeights, patternWeights, JNI_ABORT);
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
    
	jint errCode = (jint)beagleSetTipStates(instance, tipIndex, (int *)tipStates);
    
    env->ReleaseIntArrayElements(inTipStates, tipStates, JNI_ABORT);
    return errCode;
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    getTipStates
 * Signature: (II[I)I
 */
JNIEXPORT jint JNICALL Java_beagle_BeagleJNIWrapper_getTipStates
(JNIEnv *env, jobject obj, jint instance, jint tipIndex, jintArray outTipStates)
{
	fprintf(stderr,"beagleGetTipStates is not yet implemented\n");
    exit(0);   
    
//    jint *tipStates = env->GetIntArrayElements(outTipStates, NULL);    
// 
//	jint errCode = (jint)beagleGetTipStates(instance, tipIndex, (int *)tipStates);
//    
//    env->ReleaseIntArrayElements(outTipStates, tipStates, 0);
//    return errCode;
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
    
	jint errCode = (jint)beagleSetTipPartials(instance, tipIndex, (double *)partials);
    
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

	jint errCode = (jint)beagleSetPartials(instance, bufferIndex, (double *)partials);

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

    jint errCode = beagleGetPartials(instance, bufferIndex, scaleIndex, (double *)partials);

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

	jint errCode = (jint)beagleSetEigenDecomposition(instance, eigenIndex, (double *)eigenVectors, (double *)invEigenVectors, (double *)eigenValues);

    env->ReleaseDoubleArrayElements(inEigenValues, eigenValues, JNI_ABORT);
    env->ReleaseDoubleArrayElements(inInvEigenVectors, invEigenVectors, JNI_ABORT);
    env->ReleaseDoubleArrayElements(inEigenVectors, eigenVectors, JNI_ABORT);

    return errCode;
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    setStateFrequencies
 * Signature: (II[D)I
 */
JNIEXPORT jint JNICALL Java_beagle_BeagleJNIWrapper_setStateFrequencies
(JNIEnv *env, jobject obj, jint instance, jint stateFrequenciesIndex, jdoubleArray inStateFrequencies)
{
    jdouble *stateFrequencies = env->GetDoubleArrayElements(inStateFrequencies, NULL);
	
	jint errCode = (jint)beagleSetStateFrequencies(instance, stateFrequenciesIndex, (double *)stateFrequencies);
	
    env->ReleaseDoubleArrayElements(inStateFrequencies, stateFrequencies, JNI_ABORT);
	
    return errCode;
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    setCategoryWeights
 * Signature: (II[D)I
 */
JNIEXPORT jint JNICALL Java_beagle_BeagleJNIWrapper_setCategoryWeights
(JNIEnv *env, jobject obj, jint instance, jint categoryWeightsIndex, jdoubleArray inCategoryWeights)
{
    jdouble *categoryWeights = env->GetDoubleArrayElements(inCategoryWeights, NULL);
	
	jint errCode = (jint)beagleSetCategoryWeights(instance, categoryWeightsIndex, (double *)categoryWeights);
	
    env->ReleaseDoubleArrayElements(inCategoryWeights, categoryWeights, JNI_ABORT);
	
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

	jint errCode = (jint)beagleSetCategoryRates(instance, (double *)categoryRates);

    env->ReleaseDoubleArrayElements(inCategoryRates, categoryRates, JNI_ABORT);

    return errCode;
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    setTransitionMatrix
 * Signature: (II[DD)I
 */
JNIEXPORT jint JNICALL Java_beagle_BeagleJNIWrapper_setTransitionMatrix
  (JNIEnv *env, jobject obj, jint instance, jint matrixIndex, jdoubleArray inMatrix, jdouble paddedValue)
{
    jdouble *matrix = env->GetDoubleArrayElements(inMatrix, NULL);

	jint errCode = (jint)beagleSetTransitionMatrix(instance, matrixIndex, (double *)matrix, paddedValue);

    env->ReleaseDoubleArrayElements(inMatrix, matrix, JNI_ABORT);

    return errCode;
}

JNIEXPORT jint JNICALL Java_beagle_BeagleJNIWrapper_getTransitionMatrix
  (JNIEnv *env, jobject obj, jint instance, jint matrixIndex, jdoubleArray outMatrix)
{
	jdouble *matrix = env->GetDoubleArrayElements(outMatrix, NULL);

	jint errCode = (jint)beagleGetTransitionMatrix(instance, matrixIndex, (double *)matrix);

	env->ReleaseDoubleArrayElements(outMatrix, matrix, 0);

	return errCode;
}


///////////////////////////
//---TODO: Epoch model---//
///////////////////////////

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    convolveTransitionMatrices
 * Signature: (I[D)I
 */
JNIEXPORT jint JNICALL Java_beagle_BeagleJNIWrapper_convolveTransitionMatrices
   (JNIEnv *env, jobject obj, jint instance, jintArray inFirstIndices, jintArray inSecondIndices, jintArray inResultIndices, jint matrixCount)
{
    jint errCode;

        jint *firstIndices = env->GetIntArrayElements(inFirstIndices, NULL);
        jint *secondIndices = env->GetIntArrayElements(inSecondIndices, NULL);
        jint *resultIndices = env->GetIntArrayElements(inResultIndices, NULL);

        errCode = (jint)beagleConvolveTransitionMatrices(instance, (int *)firstIndices, (int *)secondIndices, (int *)resultIndices, matrixCount);

        env->ReleaseIntArrayElements(inFirstIndices, firstIndices, JNI_ABORT);
        env->ReleaseIntArrayElements(inSecondIndices, secondIndices, JNI_ABORT);
        env->ReleaseIntArrayElements(inResultIndices, resultIndices, JNI_ABORT);

    return errCode;
}


/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    updateTransitionMatrices
 * Signature: (II[I[I[I[DI)I
 */
JNIEXPORT jint JNICALL Java_beagle_BeagleJNIWrapper_updateTransitionMatrices
  (JNIEnv *env, jobject obj, jint instance, jint eigenIndex, jintArray inProbabilityIndices, jintArray inFirstDerivativeIndices, jintArray inSecondDerivativeIndices, jdoubleArray inEdgeLengths, jint count)
{
    jint errCode;

    jint *probabilityIndices = env->GetIntArrayElements(inProbabilityIndices, NULL);
    if (inFirstDerivativeIndices == NULL) {
        jdouble *edgeLengths = env->GetDoubleArrayElements(inEdgeLengths, NULL);

        errCode = (jint)beagleUpdateTransitionMatrices(instance, eigenIndex, (int *)probabilityIndices, NULL, NULL, (double *)edgeLengths, count);

        env->ReleaseDoubleArrayElements(inEdgeLengths, edgeLengths, JNI_ABORT);
    } else {
        jint *firstDerivativeIndices = env->GetIntArrayElements(inFirstDerivativeIndices, NULL);
        jint *secondDerivativeIndices = env->GetIntArrayElements(inSecondDerivativeIndices, NULL);
        jdouble *edgeLengths = env->GetDoubleArrayElements(inEdgeLengths, NULL);

        errCode = (jint)beagleUpdateTransitionMatrices(instance, eigenIndex, (int *)probabilityIndices, (int *)firstDerivativeIndices, (int *)secondDerivativeIndices, (double *)edgeLengths, count);

        env->ReleaseDoubleArrayElements(inEdgeLengths, edgeLengths, JNI_ABORT);
        env->ReleaseIntArrayElements(inSecondDerivativeIndices, secondDerivativeIndices, JNI_ABORT);
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
  (JNIEnv *env, jobject obj, jint instance, jintArray inOperations, jint operationCount, jint cumulativeScalingIndex)
{
    jint *operations = env->GetIntArrayElements(inOperations, NULL);

	jint errCode = (jint)beagleUpdatePartials(instance, (BeagleOperation*)operations, operationCount, cumulativeScalingIndex);

    env->ReleaseIntArrayElements(inOperations, operations, JNI_ABORT);

    return errCode;
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    waitForPartials
 * Signature: ([II[II)I
 */
JNIEXPORT jint JNICALL Java_beagle_BeagleJNIWrapper_waitForPartials
  (JNIEnv *env, jobject obj, jint instance, jintArray inDestinationPartials, jint destinationPartialsCount)
{
    jint *destinationPartials = env->GetIntArrayElements(inDestinationPartials, NULL);

    jint errCode = (jint)beagleWaitForPartials(instance, (int *)destinationPartials, destinationPartialsCount);

    env->ReleaseIntArrayElements(inDestinationPartials, destinationPartials, JNI_ABORT);

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
	jint errCode = (jint)beagleAccumulateScaleFactors(instance, (int*)scaleIndices, count, cumulativeScalingIndex);
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
	jint errCode = (jint)beagleAccumulateScaleFactors(instance, (int*)scaleIndices, count, cumulativeScalingIndex);
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
	
	jint errCode = (jint)beagleResetScaleFactors(instance, cumulativeScalingIndex);
	return errCode;
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    copyScaleFactors
 * Signature: (II)II
 */
JNIEXPORT jint JNICALL Java_beagle_BeagleJNIWrapper_copyScaleFactors
(JNIEnv *env, jobject obj, jint instance, jint destScalingIndex, jint srcScalingIndex) {

	jint errCode = (jint)beagleCopyScaleFactors(instance, destScalingIndex, srcScalingIndex);
	return errCode;
}


/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    calculateRootLogLikelihoods
 * Signature: (I[I[D[DI[D)I
 */
JNIEXPORT jint JNICALL Java_beagle_BeagleJNIWrapper_calculateRootLogLikelihoods
  (JNIEnv *env, jobject obj, jint instance, jintArray inBufferIndices, jintArray inCatagoryWeightsIndices, 
   jintArray inStateFrequenciesIndices, jintArray inScalingIndices, jint count, jdoubleArray outSumLogLikelihoods)
{
    jint *bufferIndices = env->GetIntArrayElements(inBufferIndices, NULL);
    jint *weightsIndices = env->GetIntArrayElements(inCatagoryWeightsIndices, NULL);
    jint *frequenciesIndices = env->GetIntArrayElements(inStateFrequenciesIndices, NULL);
    jint *scalingIndices = env->GetIntArrayElements(inScalingIndices, NULL);
    //    jint *scalingCount = env->GetIntArrayElements(inScalingCount, NULL);
	
    jdouble *sumLogLikelihoods = env->GetDoubleArrayElements(outSumLogLikelihoods, NULL);

	jint errCode = (jint)beagleCalculateRootLogLikelihoods(instance, (int *)bufferIndices, 
														   (int *)weightsIndices,
														   (int *)frequenciesIndices, 
														   (int *)scalingIndices,
														   count, (double *)sumLogLikelihoods);

    // not using JNI_ABORT flag here because we want the values to be copied back...
    env->ReleaseDoubleArrayElements(outSumLogLikelihoods, sumLogLikelihoods, 0);

    //    env->ReleaseIntArrayElements(inScalingCount, scalingCount, JNI_ABORT);
    env->ReleaseIntArrayElements(inScalingIndices, scalingIndices, JNI_ABORT);

    env->ReleaseIntArrayElements(inStateFrequenciesIndices, frequenciesIndices, JNI_ABORT);
    env->ReleaseIntArrayElements(inCatagoryWeightsIndices, weightsIndices, JNI_ABORT);
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
		jintArray inCatagoryWeightsIndices,  jintArray inStateFrequenciesIndices, jintArray inScalingIndices,
        jint count, jdoubleArray outSumLogLikelihoods, jdoubleArray outSumFirstDerivatives, jdoubleArray outSumSecondDerivatives)
{
    jint *parentBufferIndices = env->GetIntArrayElements(inParentBufferIndices, NULL);
    jint *childBufferIndices = env->GetIntArrayElements(inChildBufferIndices, NULL);
    jint *probabilityIndices = env->GetIntArrayElements(inProbabilityIndices, NULL);
    jint *firstDerivativeIndices = env->GetIntArrayElements(inFirstDerivativeIndices, NULL);
    jint *secondDerivativeIndices = env->GetIntArrayElements(inSecondDerivativeIndices, NULL);

    jint *weightsIndices = env->GetIntArrayElements(inCatagoryWeightsIndices, NULL);
    jint *frequenciesIndices = env->GetIntArrayElements(inStateFrequenciesIndices, NULL);
    jint *scalingIndices = env->GetIntArrayElements(inScalingIndices, NULL);
    //    jint *scalingCount = env->GetIntArrayElements(inScalingCount, NULL);
    jdouble *sumLogLikelihoods = env->GetDoubleArrayElements(outSumLogLikelihoods, NULL);
    jdouble *sumFirstDerivatives = env->GetDoubleArrayElements(outSumFirstDerivatives, NULL);
    jdouble *sumSecondDerivatives = env->GetDoubleArrayElements(outSumSecondDerivatives, NULL);

	jint errCode = (jint)beagleCalculateEdgeLogLikelihoods(instance, (int *)parentBufferIndices, (int *)childBufferIndices,
	                                                    (int *)probabilityIndices, (int *)firstDerivativeIndices,
	                                                    (int *)secondDerivativeIndices, 
														(int *)weightsIndices,
														(int *)frequenciesIndices, 
														(int *)scalingIndices,// (int *)scalingCount,
	                                                    count, 
														(double *)sumLogLikelihoods, 
														(double *)sumFirstDerivatives,
	                                                    (double *)sumSecondDerivatives);

    // not using JNI_ABORT flag here because we want the values to be copied back...
    env->ReleaseDoubleArrayElements(outSumSecondDerivatives, sumSecondDerivatives, 0);
    env->ReleaseDoubleArrayElements(outSumFirstDerivatives, sumFirstDerivatives, 0);
    env->ReleaseDoubleArrayElements(outSumLogLikelihoods, sumLogLikelihoods, 0);

    //    env->ReleaseIntArrayElements(inScalingCount, scalingCount, JNI_ABORT);
    env->ReleaseIntArrayElements(inScalingIndices, scalingIndices, JNI_ABORT);

    env->ReleaseIntArrayElements(inStateFrequenciesIndices, frequenciesIndices, JNI_ABORT);
    env->ReleaseIntArrayElements(inCatagoryWeightsIndices, weightsIndices, JNI_ABORT);

    env->ReleaseIntArrayElements(inSecondDerivativeIndices, secondDerivativeIndices, JNI_ABORT);
    env->ReleaseIntArrayElements(inFirstDerivativeIndices, firstDerivativeIndices, JNI_ABORT);
    env->ReleaseIntArrayElements(inProbabilityIndices, probabilityIndices, JNI_ABORT);
    env->ReleaseIntArrayElements(inChildBufferIndices, childBufferIndices, JNI_ABORT);
    env->ReleaseIntArrayElements(inParentBufferIndices, parentBufferIndices, JNI_ABORT);

    return errCode;
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    getSiteLogLikelihoods
 * Signature: (I[D)I
 */
JNIEXPORT jint JNICALL Java_beagle_BeagleJNIWrapper_getSiteLogLikelihoods
(JNIEnv *env, jobject obj, jint instance, jdoubleArray outSiteLogLikelihoods) {
	
	jdouble *siteLogLikelihoods = env->GetDoubleArrayElements(outSiteLogLikelihoods, NULL);
	
	jint errCode = (jint)beagleGetSiteLogLikelihoods(instance, (double *)siteLogLikelihoods);
	
    // not using JNI_ABORT flag here because we want the values to be copied back...
    env->ReleaseDoubleArrayElements(outSiteLogLikelihoods, siteLogLikelihoods, 0);
    return errCode;
}

//void __attribute__ ((constructor)) beagle_jni_library_initialize(void) {
//	
//}


//void __attribute__ ((destructor)) beagle_jni_library_finialize(void) {
//	
//}

