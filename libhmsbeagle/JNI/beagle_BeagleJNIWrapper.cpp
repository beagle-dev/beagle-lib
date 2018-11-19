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
 * Method:    getVersion
 * Signature: ()Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_beagle_BeagleJNIWrapper_getVersion
  (JNIEnv *env, jobject obj)
{
    return env->NewStringUTF(beagleGetVersion());
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    getCitation
 * Signature: ()Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_beagle_BeagleJNIWrapper_getCitation
  (JNIEnv *env, jobject obj)
{
    return env->NewStringUTF(beagleGetCitation());
}

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
 * Method:    getBenchmarkedResourceList
 * Signature: (IIIII[IIJJIIIJ)[Lbeagle/BenchmarkedResourceDetails;
 */
JNIEXPORT jobjectArray JNICALL Java_beagle_BeagleJNIWrapper_getBenchmarkedResourceList
  (JNIEnv *env, jobject obj, jint tipCount, jint compactBufferCount, jint stateCount, jint patternCount, jint categoryCount, jintArray inResourceList, jint resourceCount, jlong preferenceFlags, jlong requirementFlags, jint eigenModelCount, jint partitionCount, jint calculateDerivatives, jlong benchmarkFlags)
{

    jint *resourceList = NULL;
    if (inResourceList != NULL)
        resourceList = env->GetIntArrayElements(inResourceList, NULL);

    BeagleBenchmarkedResourceList* brl =  beagleGetBenchmarkedResourceList(tipCount,
                                                               compactBufferCount,
                                                               stateCount,
                                                               patternCount,
                                                               categoryCount,
                                                               (int *)resourceList,
                                                               resourceCount,
                                                               preferenceFlags,
                                                               requirementFlags,
                                                               eigenModelCount,
                                                               partitionCount,
                                                               calculateDerivatives,
                                                               benchmarkFlags);

    if (brl == NULL) {
        return NULL;
    }
    
    jclass objClass = env->FindClass("beagle/BenchmarkedResourceDetails");
    if (objClass == NULL) {
        printf("NULL returned in FindClass: can't find class: beagle/BenchmarkedResourceDetails\n");
        return NULL;
    }

    jmethodID constructorMethodID = env->GetMethodID(objClass, "<init>","(I)V");
    if (constructorMethodID == NULL) {
        printf("NULL returned in FindClass: can't find constructor for class: beagle/BenchmarkedResourceDetails\n");
        return NULL;
    }

    jmethodID setResourceNumberMethodID = env->GetMethodID(objClass, "setResourceNumber", "(I)V");
    if (setResourceNumberMethodID == NULL) {
        printf("NULL returned in FindClass: can't find 'setResourceNumber' method in class: beagle/BenchmarkedResourceDetails\n");
        return NULL;
    }

    jmethodID setNameMethodID = env->GetMethodID(objClass, "setName", "(Ljava/lang/String;)V");
    if (setNameMethodID == NULL) {
        printf("NULL returned in FindClass: can't find 'setName' method in class: beagle/BenchmarkedResourceDetails\n");
        return NULL;
    }

    jmethodID setDescriptionMethodID = env->GetMethodID(objClass, "setDescription", "(Ljava/lang/String;)V");
    if (setDescriptionMethodID == NULL) {
        printf("NULL returned in FindClass: can't find 'setDescription' method in class: beagle/BenchmarkedResourceDetails\n");
        return NULL;
    }

    jmethodID setSupportFlagsMethodID = env->GetMethodID(objClass, "setSupportFlags", "(J)V");
    if (setSupportFlagsMethodID == NULL) {
        printf("NULL returned in FindClass: can't find 'setSupportFlags' method in class: beagle/BenchmarkedResourceDetails\n");
        return NULL;
    }

    jmethodID setRequiredFlagsMethodID = env->GetMethodID(objClass, "setRequiredFlags", "(J)V");
    if (setRequiredFlagsMethodID == NULL) {
        printf("NULL returned in FindClass: can't find 'setRequiredFlags' method in class: beagle/BenchmarkedResourceDetails\n");
        return NULL;
    }

    jmethodID setReturnCodeMethodID = env->GetMethodID(objClass, "setReturnCode", "(I)V");
    if (setReturnCodeMethodID == NULL) {
        printf("NULL returned in FindClass: can't find 'setReturnCode' method in class: beagle/BenchmarkedResourceDetails\n");
        return NULL;
    }

    jmethodID setImplNameMethodID = env->GetMethodID(objClass, "setImplName", "(Ljava/lang/String;)V");
    if (setImplNameMethodID == NULL) {
        printf("NULL returned in FindClass: can't find 'setImplName' method in class: beagle/BenchmarkedResourceDetails\n");
        return NULL;
    }

    jmethodID setBenchedFlagsMethodID = env->GetMethodID(objClass, "setBenchedFlags", "(J)V");
    if (setBenchedFlagsMethodID == NULL) {
        printf("NULL returned in FindClass: can't find 'setBenchedFlags' method in class: beagle/BenchmarkedResourceDetails\n");
        return NULL;
    }

    jmethodID setBenchmarkResultMethodID = env->GetMethodID(objClass, "setBenchmarkResult", "(D)V");
    if (setBenchmarkResultMethodID == NULL) {
        printf("NULL returned in FindClass: can't find 'setBenchmarkResult' method in class: beagle/BenchmarkedResourceDetails\n");
        return NULL;
    }

    jmethodID setPerformanceRatioMethodID = env->GetMethodID(objClass, "setPerformanceRatio", "(D)V");
    if (setPerformanceRatioMethodID == NULL) {
        printf("NULL returned in FindClass: can't find 'setPerformanceRatio' method in class: beagle/BenchmarkedResourceDetails\n");
        return NULL;
    }

    jobjectArray benchmarkedResourceArray = env->NewObjectArray(brl->length, objClass, NULL);

    for (int i = 0; i < brl->length; i++) {
        jobject benchmarkedResourceObj = env->NewObject(objClass, constructorMethodID, i);

        env->CallVoidMethod(benchmarkedResourceObj, setResourceNumberMethodID, brl->list[i].number);

        jstring jString = env->NewStringUTF(brl->list[i].name);
        env->CallVoidMethod(benchmarkedResourceObj, setNameMethodID, jString);

        jString = env->NewStringUTF(brl->list[i].description);
        env->CallVoidMethod(benchmarkedResourceObj, setDescriptionMethodID, jString);

        env->CallVoidMethod(benchmarkedResourceObj, setSupportFlagsMethodID, brl->list[i].supportFlags);

        env->CallVoidMethod(benchmarkedResourceObj, setRequiredFlagsMethodID, brl->list[i].requiredFlags);

        env->CallVoidMethod(benchmarkedResourceObj, setReturnCodeMethodID, brl->list[i].returnCode);

        jString = env->NewStringUTF(brl->list[i].implName);
        env->CallVoidMethod(benchmarkedResourceObj, setImplNameMethodID, jString);

        env->CallVoidMethod(benchmarkedResourceObj, setBenchedFlagsMethodID, brl->list[i].benchedFlags);

        env->CallVoidMethod(benchmarkedResourceObj, setBenchmarkResultMethodID, brl->list[i].benchmarkResult);

        env->CallVoidMethod(benchmarkedResourceObj, setPerformanceRatioMethodID, brl->list[i].performanceRatio);

        env->SetObjectArrayElement(benchmarkedResourceArray, i, benchmarkedResourceObj);
    }
    
    return benchmarkedResourceArray;
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
 * Method:    setCPUThreadCount
 * Signature: (II)I
 */
JNIEXPORT jint JNICALL Java_beagle_BeagleJNIWrapper_setCPUThreadCount
  (JNIEnv *env, jobject obj, jint instance, jint threadCount)
{
    jint errCode = (jint)beagleSetCPUThreadCount(instance, threadCount);
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
 * Method:    setPatternPartitions
 * Signature: (II[I)I
 */
JNIEXPORT jint JNICALL Java_beagle_BeagleJNIWrapper_setPatternPartitions
  (JNIEnv *env, jobject obj, jint instance, jint partitionCount, jintArray inPatternPartitions)
{
    jint *patternPartitions = env->GetIntArrayElements(inPatternPartitions, NULL);

	jint errCode = (jint)beagleSetPatternPartitions(instance, partitionCount, (int *)patternPartitions);

    env->ReleaseIntArrayElements(inPatternPartitions, patternPartitions, JNI_ABORT);
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
 * Method:    setRootPrePartials
 * Signature: (I[I[II)I
 */
JNIEXPORT jint JNICALL Java_beagle_BeagleJNIWrapper_setRootPrePartials
(JNIEnv *env, jobject obj, jint instance, jintArray inbufferIndices, jintArray instateFrequenciesIndices, jint count)
{
    jint * bufferIndices = env->GetIntArrayElements(inbufferIndices, NULL);
    jint * stateFrequenciesIndices = env->GetIntArrayElements(instateFrequenciesIndices, NULL);

    jint errCode = (jint)beagleSetRootPrePartials(instance, (int *)bufferIndices, (int *)stateFrequenciesIndices, count);

    env->ReleaseIntArrayElements(inbufferIndices, bufferIndices, JNI_ABORT);
    env->ReleaseIntArrayElements(instateFrequenciesIndices, stateFrequenciesIndices, JNI_ABORT);
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
 * Method:    calculateEdgeDerivative
 * Signature: (I[I[II[I[IIII[II[D[D)I
 */
JNIEXPORT jint JNICALL Java_beagle_BeagleJNIWrapper_calculateEdgeDerivative
(JNIEnv *env, jobject obj, jint instance, jintArray postBufferIndices, jintArray preBufferIndices,
 jint rootBufferIndex, jintArray firstDerivativeIndices, jintArray secondDerivativeIndices,
 jint categoryWeightsIndex, jint categoryRatesIndex, jint stateFrequenciesIndex,
 jintArray cumulativeScaleIndices, jint count,
 jdoubleArray outFirstDerivative, jdoubleArray outDiagonalSecondDerivative)
{
    jdouble *outGradient = env->GetDoubleArrayElements(outFirstDerivative, NULL);
    jdouble *outDiagonalHessian = outDiagonalSecondDerivative != NULL ? env->GetDoubleArrayElements(outDiagonalSecondDerivative, NULL) : NULL;
    jint *postOrderBufferIndices = env->GetIntArrayElements(postBufferIndices, NULL);
    jint *preOrderBufferIndices  = env->GetIntArrayElements(preBufferIndices, NULL);
    jint *gradientMatrixIndices = env->GetIntArrayElements(firstDerivativeIndices, NULL);
    jint *hessianMatrixIndices  = secondDerivativeIndices != NULL ? env->GetIntArrayElements(secondDerivativeIndices, NULL) : NULL;
    jint *cumulativeScalingIndices = env->GetIntArrayElements(cumulativeScaleIndices, NULL);


    jint errCode = beagleCalculateEdgeDerivative(instance, postOrderBufferIndices, preOrderBufferIndices,
                                                 rootBufferIndex, gradientMatrixIndices, hessianMatrixIndices,
                                                 categoryWeightsIndex, categoryRatesIndex, stateFrequenciesIndex,
                                                 cumulativeScalingIndices, count, outGradient, outDiagonalHessian);
    env->ReleaseDoubleArrayElements(outFirstDerivative, outGradient, 0);
    if (outDiagonalSecondDerivative != NULL) {
        env->ReleaseDoubleArrayElements(outDiagonalSecondDerivative, outDiagonalHessian, 0);
    }
    return errCode;
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    getLogScaleFactors
 * Signature: (II[D)I
 */
JNIEXPORT jint JNICALL Java_beagle_BeagleJNIWrapper_getLogScaleFactors
  (JNIEnv *env, jobject obj, jint instance, jint scaleIndex, jdoubleArray outScaleFactors)
{
    jdouble *scaleFactors = env->GetDoubleArrayElements(outScaleFactors, NULL);

    jint errCode = beagleGetScaleFactors(instance, scaleIndex, (double *)scaleFactors);

    // not using JNI_ABORT flag here because we want the values to be copied back...
    env->ReleaseDoubleArrayElements(outScaleFactors, scaleFactors, 0);
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
 * Method:    setCategoryRatesWithIndex
 * Signature: (II[D)I
 */
JNIEXPORT jint JNICALL Java_beagle_BeagleJNIWrapper_setCategoryRatesWithIndex
  (JNIEnv *env, jobject obj, jint instance, jint categoryRatesIndex, jdoubleArray inCategoryRates)
{
    jdouble *categoryRates = env->GetDoubleArrayElements(inCategoryRates, NULL);

    jint errCode = (jint)beagleSetCategoryRatesWithIndex(instance,
                                                         categoryRatesIndex,
                                                         (double *)categoryRates);

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
    jint *probabilityIndices = env->GetIntArrayElements(inProbabilityIndices, NULL);
    jint *firstDerivativeIndices = inFirstDerivativeIndices != NULL ? env->GetIntArrayElements(inFirstDerivativeIndices, NULL) : NULL;
    jint *secondDerivativeIndices = inSecondDerivativeIndices != NULL ? env->GetIntArrayElements(inSecondDerivativeIndices, NULL) : NULL;
    jdouble *edgeLengths = env->GetDoubleArrayElements(inEdgeLengths, NULL);
    jint errCode = (jint)beagleUpdateTransitionMatrices(instance, eigenIndex, (int *)probabilityIndices, (int *)firstDerivativeIndices, (int *)secondDerivativeIndices, (double *)edgeLengths, count);

    env->ReleaseDoubleArrayElements(inEdgeLengths, edgeLengths, JNI_ABORT);
    if (inSecondDerivativeIndices != NULL) env->ReleaseIntArrayElements(inSecondDerivativeIndices, secondDerivativeIndices, JNI_ABORT);
    if (inFirstDerivativeIndices != NULL) env->ReleaseIntArrayElements(inFirstDerivativeIndices, firstDerivativeIndices, JNI_ABORT);
    env->ReleaseIntArrayElements(inProbabilityIndices, probabilityIndices, JNI_ABORT);

    return errCode;
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    updateTransitionMatricesWithMultipleModels
 * Signature: (I[I[I[I[I[I[DI)I
 */
JNIEXPORT jint JNICALL Java_beagle_BeagleJNIWrapper_updateTransitionMatricesWithMultipleModels
  (JNIEnv *env, jobject obj, jint instance, jintArray inEigenIndices, jintArray inCategoryRateIndices, jintArray inProbabilityIndices, jintArray inFirstDerivativeIndices, jintArray inSecondDerivativeIndices, jdoubleArray inEdgeLengths, jint count)
{
    jint *eigenIndices = env->GetIntArrayElements(inEigenIndices, NULL);
    jint *categoryRateIndices = env->GetIntArrayElements(inCategoryRateIndices, NULL);
    jint *probabilityIndices = env->GetIntArrayElements(inProbabilityIndices, NULL);
    jint *firstDerivativeIndices = inFirstDerivativeIndices != NULL ? env->GetIntArrayElements(inFirstDerivativeIndices, NULL) : NULL;
    jint *secondDerivativeIndices = inSecondDerivativeIndices != NULL ? env->GetIntArrayElements(inSecondDerivativeIndices, NULL) : NULL;
    jdouble *edgeLengths = env->GetDoubleArrayElements(inEdgeLengths, NULL);
    jint errCode = (jint)beagleUpdateTransitionMatricesWithMultipleModels(instance, (int *)eigenIndices, (int *)categoryRateIndices, (int *)probabilityIndices, (int *)firstDerivativeIndices, (int *)secondDerivativeIndices, (double *)edgeLengths, count);

    env->ReleaseDoubleArrayElements(inEdgeLengths, edgeLengths, JNI_ABORT);
    if (inSecondDerivativeIndices != NULL) env->ReleaseIntArrayElements(inSecondDerivativeIndices, secondDerivativeIndices, JNI_ABORT);
    if (inFirstDerivativeIndices != NULL) env->ReleaseIntArrayElements(inFirstDerivativeIndices, firstDerivativeIndices, JNI_ABORT);
    env->ReleaseIntArrayElements(inProbabilityIndices, probabilityIndices, JNI_ABORT);
    env->ReleaseIntArrayElements(inCategoryRateIndices, categoryRateIndices, JNI_ABORT);
    env->ReleaseIntArrayElements(inEigenIndices, eigenIndices, JNI_ABORT);

    return errCode;
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    updatePrePartials
 * Signature: (I[III)I
 */
JNIEXPORT jint JNICALL Java_beagle_BeagleJNIWrapper_updatePrePartials
  (JNIEnv *env, jobject obj, jint instance, jintArray inOperations, jint operationCount, jint cumulativeScaleIndex)
{
    jint *operations = env->GetIntArrayElements(inOperations, NULL);

    jint errCode = (jint)beagleUpdatePrePartials(instance, (BeagleOperation*)operations, operationCount, cumulativeScaleIndex);

    env->ReleaseIntArrayElements(inOperations, operations, JNI_ABORT);

    return errCode;
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    updatePartials
 * Signature: (I[III)I
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
 * Method:    updatePartialsByPartition
 * Signature: (I[II)I
 */
JNIEXPORT jint JNICALL Java_beagle_BeagleJNIWrapper_updatePartialsByPartition
  (JNIEnv *env, jobject obj, jint instance, jintArray inOperations, jint operationCount)
{
    jint *operations = env->GetIntArrayElements(inOperations, NULL);

    jint errCode = (jint)beagleUpdatePartialsByPartition(instance, (BeagleOperationByPartition*)operations, operationCount);

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
 * Method:    accumulateScaleFactorsByPartition
 * Signature: (I[IIII)I
 */
JNIEXPORT jint JNICALL Java_beagle_BeagleJNIWrapper_accumulateScaleFactorsByPartition
  (JNIEnv *env, jobject obj, jint instance, jintArray inScaleIndices, jint count, jint cumulativeScalingIndex, jint partitionIndex)
{
    jint *scaleIndices = env->GetIntArrayElements(inScaleIndices, NULL);
    jint errCode = (jint)beagleAccumulateScaleFactorsByPartition(instance, (int*)scaleIndices, count, cumulativeScalingIndex, partitionIndex);
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
	jint errCode = (jint)beagleRemoveScaleFactors(instance, (int*)scaleIndices, count, cumulativeScalingIndex);
	env->ReleaseIntArrayElements(inScaleIndices, scaleIndices, JNI_ABORT);
    
	return errCode;
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    removeScaleFactorsByPartition
 * Signature: (I[IIII)I
 */
JNIEXPORT jint JNICALL Java_beagle_BeagleJNIWrapper_removeScaleFactorsByPartition
  (JNIEnv *env, jobject obj, jint instance, jintArray inScaleIndices, jint count, jint cumulativeScalingIndex, jint partitionIndex)
{
    jint *scaleIndices = env->GetIntArrayElements(inScaleIndices, NULL);
    jint errCode = (jint)beagleRemoveScaleFactorsByPartition(instance, (int*)scaleIndices, count, cumulativeScalingIndex, partitionIndex);
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
 * Method:    resetScaleFactorsByPartition
 * Signature: (III)I
 */
JNIEXPORT jint JNICALL Java_beagle_BeagleJNIWrapper_resetScaleFactorsByPartition
  (JNIEnv *env, jobject obj, jint instance, jint cumulativeScalingIndex, jint partitionIndex)
{
    jint errCode = (jint)beagleResetScaleFactorsByPartition(instance, cumulativeScalingIndex, partitionIndex);
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
  (JNIEnv *env, jobject obj, jint instance, jintArray inBufferIndices, jintArray inCategoryWeightsIndices, 
   jintArray inStateFrequenciesIndices, jintArray inScalingIndices, jint count, jdoubleArray outSumLogLikelihoods)
{
    jint *bufferIndices = env->GetIntArrayElements(inBufferIndices, NULL);
    jint *weightsIndices = env->GetIntArrayElements(inCategoryWeightsIndices, NULL);
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
    env->ReleaseIntArrayElements(inCategoryWeightsIndices, weightsIndices, JNI_ABORT);
    env->ReleaseIntArrayElements(inBufferIndices, bufferIndices, JNI_ABORT);

    return errCode;
}

/*
 * Class:     beagle_BeagleJNIWrapper
 * Method:    calculateRootLogLikelihoodsByPartition
 * Signature: (I[I[I[I[I[II[D[D)I
 */
JNIEXPORT jint JNICALL Java_beagle_BeagleJNIWrapper_calculateRootLogLikelihoodsByPartition
  (JNIEnv *env, jobject obj, jint instance, jintArray inBufferIndices, jintArray inCategoryWeightsIndices, jintArray inStateFrequenciesIndices, jintArray inScalingIndices, jintArray inPartitionIndices, jint partitionCount, jint count, jdoubleArray outSumLogLikelihoodByPartition, jdoubleArray outSumLogLikelihoods)
{
    jint *bufferIndices = env->GetIntArrayElements(inBufferIndices, NULL);
    jint *weightsIndices = env->GetIntArrayElements(inCategoryWeightsIndices, NULL);
    jint *frequenciesIndices = env->GetIntArrayElements(inStateFrequenciesIndices, NULL);
    jint *scalingIndices = env->GetIntArrayElements(inScalingIndices, NULL);
    jint *partitionIndices = env->GetIntArrayElements(inPartitionIndices, NULL);

    jdouble *sumLogLikelihoodsByPartition = env->GetDoubleArrayElements(outSumLogLikelihoodByPartition, NULL);    
    jdouble *sumLogLikelihoods = env->GetDoubleArrayElements(outSumLogLikelihoods, NULL);

    jint errCode = (jint)beagleCalculateRootLogLikelihoodsByPartition(
                                                           instance,
                                                           (int *)bufferIndices, 
                                                           (int *)weightsIndices,
                                                           (int *)frequenciesIndices, 
                                                           (int *)scalingIndices,
                                                           (int *)partitionIndices,
                                                           partitionCount,
                                                           count,
                                                           (double *)sumLogLikelihoodsByPartition,
                                                           (double *)sumLogLikelihoods);

    // not using JNI_ABORT flag here because we want the values to be copied back...
    env->ReleaseDoubleArrayElements(outSumLogLikelihoodByPartition, sumLogLikelihoodsByPartition, 0);
    env->ReleaseDoubleArrayElements(outSumLogLikelihoods, sumLogLikelihoods, 0);

    env->ReleaseIntArrayElements(inPartitionIndices, partitionIndices, JNI_ABORT);
    env->ReleaseIntArrayElements(inScalingIndices, scalingIndices, JNI_ABORT);
    env->ReleaseIntArrayElements(inStateFrequenciesIndices, frequenciesIndices, JNI_ABORT);
    env->ReleaseIntArrayElements(inCategoryWeightsIndices, weightsIndices, JNI_ABORT);
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
		jintArray inCategoryWeightsIndices,  jintArray inStateFrequenciesIndices, jintArray inScalingIndices,
        jint count, jdoubleArray outSumLogLikelihoods, jdoubleArray outSumFirstDerivatives, jdoubleArray outSumSecondDerivatives)
{
    jint *parentBufferIndices = env->GetIntArrayElements(inParentBufferIndices, NULL);
    jint *childBufferIndices = env->GetIntArrayElements(inChildBufferIndices, NULL);
    jint *probabilityIndices = env->GetIntArrayElements(inProbabilityIndices, NULL);
    jint *firstDerivativeIndices = inFirstDerivativeIndices != NULL ? env->GetIntArrayElements(inFirstDerivativeIndices, NULL) : NULL;
    jint *secondDerivativeIndices = inSecondDerivativeIndices != NULL ? env->GetIntArrayElements(inSecondDerivativeIndices, NULL) : NULL;

    jint *weightsIndices = env->GetIntArrayElements(inCategoryWeightsIndices, NULL);
    jint *frequenciesIndices = env->GetIntArrayElements(inStateFrequenciesIndices, NULL);
    jint *scalingIndices = env->GetIntArrayElements(inScalingIndices, NULL);
    //    jint *scalingCount = env->GetIntArrayElements(inScalingCount, NULL);
    jdouble *sumLogLikelihoods = env->GetDoubleArrayElements(outSumLogLikelihoods, NULL);
    jdouble *sumFirstDerivatives = outSumFirstDerivatives != NULL ? env->GetDoubleArrayElements(outSumFirstDerivatives, NULL) : NULL;
    jdouble *sumSecondDerivatives = outSumSecondDerivatives != NULL ? env->GetDoubleArrayElements(outSumSecondDerivatives, NULL) : NULL;

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
    if (outSumSecondDerivatives != NULL) env->ReleaseDoubleArrayElements(outSumSecondDerivatives, sumSecondDerivatives, 0);
    if (outSumFirstDerivatives != NULL) env->ReleaseDoubleArrayElements(outSumFirstDerivatives, sumFirstDerivatives, 0);
    env->ReleaseDoubleArrayElements(outSumLogLikelihoods, sumLogLikelihoods, 0);

    //    env->ReleaseIntArrayElements(inScalingCount, scalingCount, JNI_ABORT);
    env->ReleaseIntArrayElements(inScalingIndices, scalingIndices, JNI_ABORT);

    env->ReleaseIntArrayElements(inStateFrequenciesIndices, frequenciesIndices, JNI_ABORT);
    env->ReleaseIntArrayElements(inCategoryWeightsIndices, weightsIndices, JNI_ABORT);

    if (inSecondDerivativeIndices != NULL) env->ReleaseIntArrayElements(inSecondDerivativeIndices, secondDerivativeIndices, JNI_ABORT);
    if (inFirstDerivativeIndices != NULL) env->ReleaseIntArrayElements(inFirstDerivativeIndices, firstDerivativeIndices, JNI_ABORT);
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

