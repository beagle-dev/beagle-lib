#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <jni.h>

#include "libhmsbeagle/beagle.h"
#include "libhmsbeagle/JNI/beagle_basta_BastaJNIWrapper.h"


/*
 * Class:     beagle_basta_BastaJNIWrapper
 * Method:    updateBastaPartials
 * Signature: (I[III)I
 */
JNIEXPORT jint JNICALL Java_beagle_basta_BastaJNIWrapper_updateBastaPartials
  (JNIEnv *env, jobject obj, jint instance, jintArray inOperations, 
   jint operationCount, jint populationSizesIndex) {
  	
  	jint *operations = env->GetIntArrayElements(inOperations, NULL);

	jint errCode = (jint)beagleUpdateBastaPartials(instance, 
		(BastaOperation*) operations, operationCount, populationSizesIndex);

    env->ReleaseIntArrayElements(inOperations, operations, JNI_ABORT);

    return errCode;
  
  }
  
/*
 * Class:     beagle_basta_BastaJNIWrapper
 * Method:    accumulateBastaPartials
 * Signature: (I[II[II)I
 */
JNIEXPORT jint JNICALL Java_beagle_basta_BastaJNIWrapper_accumulateBastaPartials
  (JNIEnv *, jobject, jint, jintArray, jint, jintArray, jint) {
  
  	return 0;
  
  }
