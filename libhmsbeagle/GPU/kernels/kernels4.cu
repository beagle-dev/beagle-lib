/*
 *
 * Copyright 2009 Phylogenetic Likelihood Working Group
 *
 * This file is part of BEAGLE.
 *
 * BEAGLE is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 *
 * BEAGLE is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with BEAGLE.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @author Marc Suchard
 * @author Daniel Ayres
 * @author Andrew Rambaut
 */

#define STATE_COUNT 4

#define multBy4(x)	(x << 2)
#define multBy16(x)	(x << 4)

#define LIKE_PATTERN_BLOCK_SIZE PATTERN_BLOCK_SIZE

#ifdef CUDA

#include "libhmsbeagle/GPU/GPUImplDefs.h"
#include "libhmsbeagle/GPU/kernels/kernelsAll.cu" // This file includes the non-state-count specific kernels

// Do not use | (instead of +) for any term involing PATTERN_BLOCK_SIZE
// as this should be adjustable
#define DETERMINE_INDICES_4() \
    int tx = KW_LOCAL_ID_0; \
    int state = tx & 0x3; \
    int pat = tx >> 2; \
    int patIdx = KW_LOCAL_ID_1; \
    int matrix = KW_GROUP_ID_1; \
    int pattern = __umul24(KW_GROUP_ID_0, PATTERN_BLOCK_SIZE * 4) + multBy4(patIdx) + pat; \
    int deltaPartialsByState = multBy16(KW_GROUP_ID_0 * PATTERN_BLOCK_SIZE + patIdx); \
    int deltaPartialsByMatrix = __umul24(matrix, multBy4(totalPatterns)); \
    int x2 = multBy16(matrix); \
    int u = tx + deltaPartialsByState + deltaPartialsByMatrix;

extern "C" {
    
#elif defined(FW_OPENCL)

#define DETERMINE_INDICES_4() int tx = KW_LOCAL_ID_0; int state = tx & 0x3; int pat = tx >> 2; int patIdx = KW_LOCAL_ID_1; int matrix = KW_GROUP_ID_1; int pattern = (KW_GROUP_ID_0 * PATTERN_BLOCK_SIZE * 4) + multBy4(patIdx) + pat; int deltaPartialsByState = multBy16(KW_GROUP_ID_0 * PATTERN_BLOCK_SIZE + patIdx); int deltaPartialsByMatrix = (matrix * multBy4(totalPatterns)); int x2 = multBy16(matrix); int u = tx + deltaPartialsByState + deltaPartialsByMatrix;

#endif //OpenCL

// kernels shared by CUDA and OpenCL
    
KW_GLOBAL_KERNEL void kernelPartialsPartialsNoScale(KW_GLOBAL_VAR REAL* partials1,
                                                              KW_GLOBAL_VAR REAL* partials2,
                                                              KW_GLOBAL_VAR REAL* partials3,
                                                              KW_GLOBAL_VAR REAL* matrices1,
                                                              KW_GLOBAL_VAR REAL* matrices2,
                                                              int totalPatterns) {
    REAL sum1;
    REAL sum2;
    int i;

    DETERMINE_INDICES_4();

    int patIdx16pat4 = multBy16(patIdx) | (tx & 0xC);
    int y = deltaPartialsByState + deltaPartialsByMatrix;
    
#ifdef KERNEL_PRINT_ENABLED
    printf("matrix = %d, pat = %d for tx = %d and state = %d :  u = %d\n", matrix, pattern, tx,
           state, u);
#endif

    KW_GLOBAL_VAR REAL* matrix1 = matrices1 + x2; // Points to *this* matrix
    KW_GLOBAL_VAR REAL* matrix2 = matrices2 + x2;

#ifdef FW_OPENCL_INTEL

    KW_GLOBAL_VAR REAL* sMatrix1 = matrix1;
    KW_GLOBAL_VAR REAL* sMatrix2 = matrix2;

    int deltaPartials = multBy16(KW_GROUP_ID_0 * PATTERN_BLOCK_SIZE) + deltaPartialsByMatrix;
    KW_GLOBAL_VAR REAL* sPartials1 = partials1 + deltaPartials;
    KW_GLOBAL_VAR REAL* sPartials2 = partials2 + deltaPartials;

#if (!defined DOUBLE_PRECISION && defined FP_FAST_FMAF) || (defined DOUBLE_PRECISION && defined FP_FAST_FMA)
    sum1 =     sMatrix1[0 * 4 + state] * sPartials1[patIdx * 16 + pat * 4];
    sum1 = fma(sMatrix1[1 * 4 + state],  sPartials1[patIdx * 16 + pat * 4 + 1], sum1);
    sum1 = fma(sMatrix1[2 * 4 + state],  sPartials1[patIdx * 16 + pat * 4 + 2], sum1);
    sum1 = fma(sMatrix1[3 * 4 + state],  sPartials1[patIdx * 16 + pat * 4 + 3], sum1);
    sum2 =     sMatrix2[0 * 4 + state] * sPartials2[patIdx * 16 + pat * 4];
    sum2 = fma(sMatrix2[1 * 4 + state],  sPartials2[patIdx * 16 + pat * 4 + 1], sum2);
    sum2 = fma(sMatrix2[2 * 4 + state],  sPartials2[patIdx * 16 + pat * 4 + 2], sum2);
    sum2 = fma(sMatrix2[3 * 4 + state],  sPartials2[patIdx * 16 + pat * 4 + 3], sum2);
#else //FP_FAST_FMA
    sum1 =  sMatrix1[0 * 4 + state] * sPartials1[patIdx * 16 + pat * 4];
    sum1 += sMatrix1[1 * 4 + state] * sPartials1[patIdx * 16 + pat * 4 + 1];
    sum1 += sMatrix1[2 * 4 + state] * sPartials1[patIdx * 16 + pat * 4 + 2];
    sum1 += sMatrix1[3 * 4 + state] * sPartials1[patIdx * 16 + pat * 4 + 3];
    sum2 =  sMatrix2[0 * 4 + state] * sPartials2[patIdx * 16 + pat * 4];
    sum2 += sMatrix2[1 * 4 + state] * sPartials2[patIdx * 16 + pat * 4 + 1];
    sum2 += sMatrix2[2 * 4 + state] * sPartials2[patIdx * 16 + pat * 4 + 2];
    sum2 += sMatrix2[3 * 4 + state] * sPartials2[patIdx * 16 + pat * 4 + 3];
#endif //FP_FAST_FMA

#else

    // Load values into shared memory
    KW_LOCAL_MEM REAL sMatrix1[16];
    KW_LOCAL_MEM REAL sMatrix2[16];

    KW_LOCAL_MEM REAL sPartials1[PATTERN_BLOCK_SIZE * 4 * 4];
    KW_LOCAL_MEM REAL sPartials2[PATTERN_BLOCK_SIZE * 4 * 4];

    // copy PADDED_STATE_COUNT * PATTERN_BLOCK_SIZE lengthed partials
    sPartials1[multBy16(patIdx) | tx] = partials1[y | tx]; // All coalesced memory reads
    sPartials2[multBy16(patIdx) | tx] = partials2[y | tx];

    if (patIdx == 0 ) {
        sMatrix1[tx] = matrix1[tx]; // All coalesced memory reads
        sMatrix2[tx] = matrix2[tx];
    }

    KW_LOCAL_FENCE;

    i = pat;

    sum1  =     sMatrix1[multBy4(i) | state] * sPartials1[patIdx16pat4 | i];
    sum2  =     sMatrix2[multBy4(i) | state] * sPartials2[patIdx16pat4 | i];
    i = (++i) & 0x3;

#if (!defined DOUBLE_PRECISION && defined FP_FAST_FMAF) || (defined DOUBLE_PRECISION && defined FP_FAST_FMA)
    sum1  = fma(sMatrix1[multBy4(i) | state],  sPartials1[patIdx16pat4 | i], sum1);
    sum2  = fma(sMatrix2[multBy4(i) | state],  sPartials2[patIdx16pat4 | i], sum2);
    i = (++i) & 0x3;
    sum1  = fma(sMatrix1[multBy4(i) | state],  sPartials1[patIdx16pat4 | i], sum1);
    sum2  = fma(sMatrix2[multBy4(i) | state],  sPartials2[patIdx16pat4 | i], sum2);
    i = (++i) & 0x3;
    sum1  = fma(sMatrix1[multBy4(i) | state],  sPartials1[patIdx16pat4 | i], sum1);
    sum2  = fma(sMatrix2[multBy4(i) | state],  sPartials2[patIdx16pat4 | i], sum2);
#else //FP_FAST_FMA
    sum1 +=     sMatrix1[multBy4(i) | state] * sPartials1[patIdx16pat4 | i];
    sum2 +=     sMatrix2[multBy4(i) | state] * sPartials2[patIdx16pat4 | i];
    i = (++i) & 0x3;
    sum1 +=     sMatrix1[multBy4(i) | state] * sPartials1[patIdx16pat4 | i];
    sum2 +=     sMatrix2[multBy4(i) | state] * sPartials2[patIdx16pat4 | i];
    i = (++i) & 0x3;
    sum1 +=     sMatrix1[multBy4(i) | state] * sPartials1[patIdx16pat4 | i];
    sum2 +=     sMatrix2[multBy4(i) | state] * sPartials2[patIdx16pat4 | i];
#endif //FP_FAST_FMA


#endif

    partials3[u] = sum1 * sum2;
}

KW_GLOBAL_KERNEL void kernelPartialsPartialsFixedScale(KW_GLOBAL_VAR REAL* partials1,
                                                                      KW_GLOBAL_VAR REAL* partials2,
                                                                      KW_GLOBAL_VAR REAL* partials3,
                                                                      KW_GLOBAL_VAR REAL* matrices1,
                                                                      KW_GLOBAL_VAR REAL* matrices2,
                                                                      KW_GLOBAL_VAR REAL* scalingFactors,
                                                                      int totalPatterns) {
    REAL sum1;
    REAL sum2;
    int i;

    DETERMINE_INDICES_4();
    int y = deltaPartialsByState + deltaPartialsByMatrix;
    KW_GLOBAL_VAR REAL* matrix1 = matrices1 + x2; // Points to *this* matrix
    KW_GLOBAL_VAR REAL* matrix2 = matrices2 + x2;

#ifdef KERNEL_PRINT_ENABLED
    printf("matrix = %d, pat = %d for tx = %d and state = %d :  u = %d\n", matrix, pattern, tx,
           state, u);
#endif

    // Load values into shared memory
    KW_LOCAL_MEM REAL sMatrix1[16];
    KW_LOCAL_MEM REAL sMatrix2[16];

    KW_LOCAL_MEM REAL sPartials1[PATTERN_BLOCK_SIZE * 4 * 4];
    KW_LOCAL_MEM REAL sPartials2[PATTERN_BLOCK_SIZE * 4 * 4];

    KW_LOCAL_MEM REAL fixedScalingFactors[PATTERN_BLOCK_SIZE * 4];

    // copy PADDED_STATE_COUNT*PATTERN_BLOCK_SIZE lengthed partials
    if (pattern < totalPatterns) {
        sPartials1[patIdx * 16 + tx] = partials1[y + tx]; // All coalesced memory reads
        sPartials2[patIdx * 16 + tx] = partials2[y + tx];
    } else {
        sPartials1[patIdx * 16 + tx] = 0;
        sPartials2[patIdx * 16 + tx] = 0;
    }

    if (patIdx < 4) // need to load 4*PATTERN_BLOCK_SIZE factors for this block
        fixedScalingFactors[patIdx * PATTERN_BLOCK_SIZE + tx] =
            scalingFactors[KW_GROUP_ID_0 * PATTERN_BLOCK_SIZE * 4 + patIdx * PATTERN_BLOCK_SIZE + tx];

    if (patIdx == 0 ) {
        sMatrix1[tx] = matrix1[tx]; // All coalesced memory reads
        sMatrix2[tx] = matrix2[tx];
    }

    KW_LOCAL_FENCE;

    if (pattern < totalPatterns) { // Remove padded threads!

        i = pat;
        sum1  = sMatrix1[i * 4 + state] * sPartials1[patIdx * 16 + pat * 4 + i];
        sum2  = sMatrix2[i * 4 + state] * sPartials2[patIdx * 16 + pat * 4 + i];

        i = (++i) & 0x3;
        sum1 += sMatrix1[i * 4 + state] * sPartials1[patIdx * 16 + pat * 4 + i];
        sum2 += sMatrix2[i * 4 + state] * sPartials2[patIdx * 16 + pat * 4 + i];

        i = (++i) & 0x3;
        sum1 += sMatrix1[i * 4 + state] * sPartials1[patIdx * 16 + pat * 4 + i];
        sum2 += sMatrix2[i * 4 + state] * sPartials2[patIdx * 16 + pat * 4 + i];

        i = (++i) & 0x3;
        sum1 += sMatrix1[i * 4 + state] * sPartials1[patIdx * 16 + pat * 4 + i];
        sum2 += sMatrix2[i * 4 + state] * sPartials2[patIdx * 16 + pat * 4 + i];
        
        partials3[u] = sum1 * sum2 / fixedScalingFactors[patIdx * 4 + pat];
    }

}
    
KW_GLOBAL_KERNEL void kernelPartialsPartialsCheckScale(KW_GLOBAL_VAR REAL* partials1,
                                                                  KW_GLOBAL_VAR REAL* partials2,
                                                                  KW_GLOBAL_VAR REAL* partials3,
                                                                  KW_GLOBAL_VAR REAL* matrices1,
                                                                  KW_GLOBAL_VAR REAL* matrices2,
                                                                  KW_GLOBAL_VAR int* dRescalingTrigger,
                                                                  int totalPatterns) {
		REAL sum1;
	    REAL sum2;
	    int i;

	    DETERMINE_INDICES_4();

	    int patIdx16pat4 = multBy16(patIdx) | (tx & 0xC);
	    int y = deltaPartialsByState + deltaPartialsByMatrix;
	    
	    KW_GLOBAL_VAR REAL* matrix1 = matrices1 + x2; // Points to *this* matrix
	    KW_GLOBAL_VAR REAL* matrix2 = matrices2 + x2;

	#ifdef KERNEL_PRINT_ENABLED
	    printf("matrix = %d, pat = %d for tx = %d and state = %d :  u = %d\n", matrix, pattern, tx,
	           state, u);
	#endif

	    // Load values into shared memory
	    KW_LOCAL_MEM REAL sMatrix1[16];
	    KW_LOCAL_MEM REAL sMatrix2[16];

	    KW_LOCAL_MEM REAL sPartials1[PATTERN_BLOCK_SIZE * 4 * 4];
	    KW_LOCAL_MEM REAL sPartials2[PATTERN_BLOCK_SIZE * 4 * 4];

	    // copy PADDED_STATE_COUNT * PATTERN_BLOCK_SIZE lengthed partials
	    if (pattern < totalPatterns) {
	        sPartials1[multBy16(patIdx) | tx] = partials1[y | tx]; // All coalesced memory reads
	        sPartials2[multBy16(patIdx) | tx] = partials2[y | tx];
	    } else {
	        sPartials1[multBy16(patIdx) | tx] = 0;
	        sPartials2[multBy16(patIdx) | tx] = 0;
	    }

	    if (patIdx == 0 ) {
	        sMatrix1[tx] = matrix1[tx]; // All coalesced memory reads
	        sMatrix2[tx] = matrix2[tx];
	    }

	    KW_LOCAL_FENCE;

	    if (pattern < totalPatterns) { // Remove padded threads!

 	        i = pat;
	        sum1  = sMatrix1[multBy4(i) | state] * sPartials1[patIdx16pat4 | i];
	        sum2  = sMatrix2[multBy4(i) | state] * sPartials2[patIdx16pat4 | i];

	        i = (++i) & 0x3;
	        sum1 += sMatrix1[multBy4(i) | state] * sPartials1[patIdx16pat4 | i];
	        sum2 += sMatrix2[multBy4(i) | state] * sPartials2[patIdx16pat4 | i];

	        i = (++i) & 0x3;
	        sum1 += sMatrix1[multBy4(i) | state] * sPartials1[patIdx16pat4 | i];
	        sum2 += sMatrix2[multBy4(i) | state] * sPartials2[patIdx16pat4 | i];

	        i = (++i) & 0x3;
	        sum1 += sMatrix1[multBy4(i) | state] * sPartials1[patIdx16pat4 | i];
	        sum2 += sMatrix2[multBy4(i) | state] * sPartials2[patIdx16pat4 | i];
            
            REAL tmpPartial = sum1 * sum2;
            
            partials3[u] = tmpPartial;

            if (tmpPartial < SCALING_THRESHOLD_LOWER || tmpPartial > SCALING_THRESHOLD_UPPER)
                *dRescalingTrigger = 1;
            
//            union {float f; long l;} fl;
//            fl.f = sum1 * sum2;;
//
//	        partials3[u] = fl.f;
//            
//            int expTmp  = ((fl.l >> 23) & 0x000000ff) - 0x7e;
//            
//            if (abs(expTmp) > SCALING_EXPONENT_THRESHOLD)
//                *dRescalingTrigger = 1;
	    }

	}

KW_GLOBAL_KERNEL void kernelPartialsPartialsFixedCheckScale(KW_GLOBAL_VAR REAL* partials1,
                                                      KW_GLOBAL_VAR REAL* partials2,
                                                      KW_GLOBAL_VAR REAL* partials3,
                                                      KW_GLOBAL_VAR REAL* matrices1,
                                                      KW_GLOBAL_VAR REAL* matrices2,
                                                      KW_GLOBAL_VAR REAL* scalingFactors,
                                                      KW_GLOBAL_VAR int* dRescalingTrigger,
                                                      int totalPatterns) {
    REAL sum1;
    REAL sum2;
    int i;

    DETERMINE_INDICES_4();
    int y = deltaPartialsByState + deltaPartialsByMatrix;
    KW_GLOBAL_VAR REAL* matrix1 = matrices1 + x2; // Points to *this* matrix
    KW_GLOBAL_VAR REAL* matrix2 = matrices2 + x2;

#ifdef KERNEL_PRINT_ENABLED
    printf("matrix = %d, pat = %d for tx = %d and state = %d :  u = %d\n", matrix, pattern, tx,
           state, u);
#endif

    // Load values into shared memory
    KW_LOCAL_MEM REAL sMatrix1[16];
    KW_LOCAL_MEM REAL sMatrix2[16];

    KW_LOCAL_MEM REAL sPartials1[PATTERN_BLOCK_SIZE * 4 * 4];
    KW_LOCAL_MEM REAL sPartials2[PATTERN_BLOCK_SIZE * 4 * 4];

    KW_LOCAL_MEM REAL fixedScalingFactors[PATTERN_BLOCK_SIZE * 4];

    // copy PADDED_STATE_COUNT*PATTERN_BLOCK_SIZE lengthed partials
    if (pattern < totalPatterns) {
        sPartials1[patIdx * 16 + tx] = partials1[y + tx]; // All coalesced memory reads
        sPartials2[patIdx * 16 + tx] = partials2[y + tx];
    } else {
        sPartials1[patIdx * 16 + tx] = 0;
        sPartials2[patIdx * 16 + tx] = 0;
    }

    if (patIdx < 4) // need to load 4*PATTERN_BLOCK_SIZE factors for this block
        fixedScalingFactors[patIdx * PATTERN_BLOCK_SIZE + tx] =
            scalingFactors[KW_GROUP_ID_0 * PATTERN_BLOCK_SIZE * 4 + patIdx * PATTERN_BLOCK_SIZE + tx];

    if (patIdx == 0 ) {
        sMatrix1[tx] = matrix1[tx]; // All coalesced memory reads
        sMatrix2[tx] = matrix2[tx];
    }

    KW_LOCAL_FENCE;

    if (pattern < totalPatterns) { // Remove padded threads!

        i = pat;
        sum1  = sMatrix1[i * 4 + state] * sPartials1[patIdx * 16 + pat * 4 + i];
        sum2  = sMatrix2[i * 4 + state] * sPartials2[patIdx * 16 + pat * 4 + i];

        i = (++i) & 0x3;
        sum1 += sMatrix1[i * 4 + state] * sPartials1[patIdx * 16 + pat * 4 + i];
        sum2 += sMatrix2[i * 4 + state] * sPartials2[patIdx * 16 + pat * 4 + i];

        i = (++i) & 0x3;
        sum1 += sMatrix1[i * 4 + state] * sPartials1[patIdx * 16 + pat * 4 + i];
        sum2 += sMatrix2[i * 4 + state] * sPartials2[patIdx * 16 + pat * 4 + i];

        i = (++i) & 0x3;
        sum1 += sMatrix1[i * 4 + state] * sPartials1[patIdx * 16 + pat * 4 + i];
        sum2 += sMatrix2[i * 4 + state] * sPartials2[patIdx * 16 + pat * 4 + i];
        
        REAL tmpPartial = sum1 * sum2 * fixedScalingFactors[patIdx * 4 + pat];
        
        partials3[u] = tmpPartial;

        if (tmpPartial < SCALING_THRESHOLD_LOWER || tmpPartial > SCALING_THRESHOLD_UPPER)
            *dRescalingTrigger = 1;

    }

}

KW_GLOBAL_KERNEL void kernelPartialsPartialsAutoScale(KW_GLOBAL_VAR REAL* partials1,
                                                KW_GLOBAL_VAR REAL* partials2,
                                                KW_GLOBAL_VAR REAL* partials3,
                                                KW_GLOBAL_VAR REAL* matrices1,
                                                KW_GLOBAL_VAR REAL* matrices2,
                                                KW_GLOBAL_VAR signed char* scalingFactors,
                                                int totalPatterns) {
    REAL sum1;
    REAL sum2;
    int i;

    DETERMINE_INDICES_4();

    int patIdx16pat4 = multBy16(patIdx) | (tx & 0xC);
    int y = deltaPartialsByState + deltaPartialsByMatrix;
    int myIdx = multBy16(patIdx) + tx; // threadId in block
    
    KW_GLOBAL_VAR REAL* matrix1 = matrices1 + x2; // Points to *this* matrix
    KW_GLOBAL_VAR REAL* matrix2 = matrices2 + x2;

#ifdef KERNEL_PRINT_ENABLED
    printf("matrix = %d, pat = %d for tx = %d and state = %d :  u = %d\n", matrix, pattern, tx,
           state, u);
#endif

    // Load values into shared memory
    KW_LOCAL_MEM REAL sMatrix1[16];
    KW_LOCAL_MEM REAL sMatrix2[16];

    KW_LOCAL_MEM REAL sPartials1[PATTERN_BLOCK_SIZE * 4 * 4];
    KW_LOCAL_MEM REAL sPartials2[PATTERN_BLOCK_SIZE * 4 * 4];

    // copy PADDED_STATE_COUNT * PATTERN_BLOCK_SIZE lengthed partials
    if (pattern < totalPatterns) {
        sPartials1[multBy16(patIdx) | tx] = partials1[y | tx]; // All coalesced memory reads
        sPartials2[multBy16(patIdx) | tx] = partials2[y | tx];
    } else {
        sPartials1[multBy16(patIdx) | tx] = 0;
        sPartials2[multBy16(patIdx) | tx] = 0;
    }

    if (patIdx == 0 ) {
        sMatrix1[tx] = matrix1[tx]; // All coalesced memory reads
        sMatrix2[tx] = matrix2[tx];
    }

    KW_LOCAL_FENCE;

    i = pat;
    sum1  = sMatrix1[multBy4(i) | state] * sPartials1[patIdx16pat4 | i];
    sum2  = sMatrix2[multBy4(i) | state] * sPartials2[patIdx16pat4 | i];

    i = (++i) & 0x3;
    sum1 += sMatrix1[multBy4(i) | state] * sPartials1[patIdx16pat4 | i];
    sum2 += sMatrix2[multBy4(i) | state] * sPartials2[patIdx16pat4 | i];

    i = (++i) & 0x3;
    sum1 += sMatrix1[multBy4(i) | state] * sPartials1[patIdx16pat4 | i];
    sum2 += sMatrix2[multBy4(i) | state] * sPartials2[patIdx16pat4 | i];

    i = (++i) & 0x3;
    sum1 += sMatrix1[multBy4(i) | state] * sPartials1[patIdx16pat4 | i];
    sum2 += sMatrix2[multBy4(i) | state] * sPartials2[patIdx16pat4 | i];
    
    REAL tmpPartial = sum1 * sum2;
    int expTmp;
    REAL sigTmp = frexp(tmpPartial, &expTmp);        

    KW_LOCAL_FENCE;
    
    if (pattern < totalPatterns) {
        if (abs(expTmp) > SCALING_EXPONENT_THRESHOLD) {
            // now using sPartials2 to hold scaling trigger boolean
            sPartials2[patIdx16pat4] = 1;
        } else {
            partials3[u] = tmpPartial;
            sPartials2[patIdx16pat4] = 0;
            sPartials1[myIdx] = 0;
        }
    } 
    
    KW_LOCAL_FENCE;
    
    int scalingActive = sPartials2[patIdx16pat4];
        
    if (scalingActive) {
        // now using sPartials1 to store max unscaled partials3
        sPartials1[myIdx] = tmpPartial;
    }
        
    KW_LOCAL_FENCE;
        
    // Unrolled parallel max-reduction
    if (scalingActive && state < 2) {
        REAL compare = sPartials1[myIdx + 2];
        if (compare >  sPartials1[myIdx])
            sPartials1[myIdx] = compare;
    }
     
    KW_LOCAL_FENCE;
            
    if (scalingActive && state < 1) {
        REAL maxPartial = sPartials1[myIdx + 1];
        if (maxPartial < sPartials1[myIdx])
            maxPartial = sPartials1[myIdx];
        int expMax;
        frexp(maxPartial, &expMax);
        sPartials1[myIdx] = expMax;
    }

    KW_LOCAL_FENCE;
            
    if (scalingActive) 
        partials3[u] = ldexp(sigTmp, expTmp - sPartials1[patIdx16pat4]);
        
    if ((myIdx < PATTERN_BLOCK_SIZE * 4) && (myIdx + (KW_GROUP_ID_0 * PATTERN_BLOCK_SIZE * 4) < totalPatterns))
        scalingFactors[(KW_GROUP_ID_0 * PATTERN_BLOCK_SIZE * 4) + (matrix * totalPatterns) + myIdx] = sPartials1[multBy4(myIdx)];
}

KW_GLOBAL_KERNEL void kernelStatesPartialsNoScale(KW_GLOBAL_VAR int* states1,
                                                                KW_GLOBAL_VAR REAL* partials2,
                                                                KW_GLOBAL_VAR REAL* partials3,
                                                                KW_GLOBAL_VAR REAL* matrices1,
                                                                KW_GLOBAL_VAR REAL* matrices2,
                                                                int totalPatterns) {
    REAL sum1 = 1;
    REAL sum2;
    int i;

    DETERMINE_INDICES_4();
    int y = deltaPartialsByState + deltaPartialsByMatrix;
    KW_GLOBAL_VAR REAL* matrix1 = matrices1 + x2; // Points to *this* matrix
    KW_GLOBAL_VAR REAL* matrix2 = matrices2 + x2;


#ifdef KERNEL_PRINT_ENABLED
    printf("matrix = %d, pat = %d for tx = %d and state = %d :  u = %d\n", matrix, pattern, tx,
           state, u);
#endif

    // Load values into shared memory
    KW_LOCAL_MEM REAL sMatrix1[16];
    KW_LOCAL_MEM REAL sMatrix2[16];

    KW_LOCAL_MEM REAL sPartials2[PATTERN_BLOCK_SIZE * 4 * 4];

    // copy PADDED_STATE_COUNT * PATTERN_BLOCK_SIZE lengthed partials
    if (pattern < totalPatterns) {
        sPartials2[patIdx * 16 + tx] = partials2[y + tx];
    } else {
        sPartials2[patIdx * 16 + tx] = 0;
    }

    if (patIdx == 0) {
        sMatrix1[tx] = matrix1[tx]; // All coalesced memory reads
        sMatrix2[tx] = matrix2[tx];
    }

    KW_LOCAL_FENCE;

    if (pattern < totalPatterns) { // Remove padded threads!

        int state1 = states1[pattern];

        if (state1 < PADDED_STATE_COUNT)
            sum1 = sMatrix1[state1 * 4 + state];

        i = pat;
        sum2  = sMatrix2[i * 4 + state] * sPartials2[patIdx * 16 + pat * 4 + i];
        i = (++i) & 0x3;
        sum2 += sMatrix2[i * 4 + state] * sPartials2[patIdx * 16 + pat * 4 + i];
        i = (++i) & 0x3;
        sum2 += sMatrix2[i * 4 + state] * sPartials2[patIdx * 16 + pat * 4 + i];
        i = (++i) & 0x3;
        sum2 += sMatrix2[i * 4 + state] * sPartials2[patIdx * 16 + pat * 4 + i];
        partials3[u] = sum1 * sum2;
    }

}

KW_GLOBAL_KERNEL void kernelStatesPartialsFixedScale(KW_GLOBAL_VAR int* states1,
                                                                KW_GLOBAL_VAR REAL* partials2,
                                                                KW_GLOBAL_VAR REAL* partials3,
                                                                KW_GLOBAL_VAR REAL* matrices1,
                                                                KW_GLOBAL_VAR REAL* matrices2,
                                                                KW_GLOBAL_VAR REAL* scalingFactors,
                                                                int totalPatterns) {
    REAL sum1 = 1;
    REAL sum2;
    int i;

    DETERMINE_INDICES_4();
    int y = deltaPartialsByState + deltaPartialsByMatrix;
    KW_GLOBAL_VAR REAL* matrix1 = matrices1 + x2; // Points to *this* matrix
    KW_GLOBAL_VAR REAL* matrix2 = matrices2 + x2;


#ifdef KERNEL_PRINT_ENABLED
    printf("matrix = %d, pat = %d for tx = %d and state = %d :  u = %d\n", matrix, pattern, tx,
           state, u);
#endif

    // Load values into shared memory
    KW_LOCAL_MEM REAL sMatrix1[16];
    KW_LOCAL_MEM REAL sMatrix2[16];

    KW_LOCAL_MEM REAL sPartials2[PATTERN_BLOCK_SIZE * 4 * 4];

    KW_LOCAL_MEM REAL fixedScalingFactors[PATTERN_BLOCK_SIZE * 4];


    // copy PADDED_STATE_COUNT * PATTERN_BLOCK_SIZE lengthed partials
    if (pattern < totalPatterns) {
        sPartials2[patIdx * 16 + tx] = partials2[y + tx];
    } else {
        sPartials2[patIdx * 16 + tx] = 0;
    }

    if (patIdx < 4) // need to load 4*PATTERN_BLOCK_SIZE factors for this block
        fixedScalingFactors[patIdx * PATTERN_BLOCK_SIZE + tx] =
            scalingFactors[KW_GROUP_ID_0 * PATTERN_BLOCK_SIZE * 4 + patIdx * PATTERN_BLOCK_SIZE + tx];


    if (patIdx == 0) {
        sMatrix1[tx] = matrix1[tx]; // All coalesced memory reads
        sMatrix2[tx] = matrix2[tx];
    }

    KW_LOCAL_FENCE;

    if (pattern < totalPatterns) { // Remove padded threads!

        int state1 = states1[pattern];

        if (state1 < PADDED_STATE_COUNT)
            sum1 = sMatrix1[state1 * 4 + state];

        i = pat;
        sum2  = sMatrix2[i * 4 + state] * sPartials2[patIdx * 16 + pat * 4 + i];
        i = (++i) & 0x3;
        sum2 += sMatrix2[i * 4 + state] * sPartials2[patIdx * 16 + pat * 4 + i];
        i = (++i) & 0x3;
        sum2 += sMatrix2[i * 4 + state] * sPartials2[patIdx * 16 + pat * 4 + i];
        i = (++i) & 0x3;
        sum2 += sMatrix2[i * 4 + state] * sPartials2[patIdx * 16 + pat * 4 + i];
        partials3[u] = sum1 * sum2 / fixedScalingFactors[patIdx * 4 + pat];
    }

}



KW_GLOBAL_KERNEL void kernelStatesStatesNoScale(KW_GLOBAL_VAR int* states1,
                                                              KW_GLOBAL_VAR int* states2,
                                                              KW_GLOBAL_VAR REAL* partials3,
                                                              KW_GLOBAL_VAR REAL* matrices1,
                                                              KW_GLOBAL_VAR REAL* matrices2,
                                                              int totalPatterns) {

	DETERMINE_INDICES_4();
    KW_GLOBAL_VAR REAL* matrix1 = matrices1 + x2; // Points to *this* matrix
    KW_GLOBAL_VAR REAL* matrix2 = matrices2 + x2;
    
#ifdef KERNEL_PRINT_ENABLED
    printf("matrix = %d, pat = %d for tx = %d and state = %d :  u = %d\n", matrix, pattern, tx,
           state, u);
#endif

    // Load values into shared memory
    KW_LOCAL_MEM REAL sMatrix1[16];
    KW_LOCAL_MEM REAL sMatrix2[16];

    if (patIdx == 0 ) {
        sMatrix1[tx] = matrix1[tx]; // All coalesced memory reads
        sMatrix2[tx] = matrix2[tx];
    }

    KW_LOCAL_FENCE;

    if (pattern < totalPatterns) {
        int state1 = states1[pattern];
        int state2 = states2[pattern];

        if ( state1 < PADDED_STATE_COUNT && state2 < PADDED_STATE_COUNT) {
            partials3[u] = sMatrix1[state1 * 4 + state] * sMatrix2[state2 * 4 + state];
        } else if (state1 < PADDED_STATE_COUNT) {
            partials3[u] = sMatrix1[state1 * 4 + state];
        } else if (state2 < PADDED_STATE_COUNT) {
            partials3[u] = sMatrix2[state2 * 4 + state];
        } else {
            partials3[u] = 1.0;
        }
    }
}

KW_GLOBAL_KERNEL void kernelStatesStatesFixedScale(KW_GLOBAL_VAR int* states1,
                                                              KW_GLOBAL_VAR int* states2,
                                                              KW_GLOBAL_VAR REAL* partials3,
                                                              KW_GLOBAL_VAR REAL* matrices1,
                                                              KW_GLOBAL_VAR REAL* matrices2,
                                                              KW_GLOBAL_VAR REAL* scalingFactors,
                                                              int totalPatterns) {

	DETERMINE_INDICES_4();
    KW_GLOBAL_VAR REAL* matrix1 = matrices1 + x2; // Points to *this* matrix
    KW_GLOBAL_VAR REAL* matrix2 = matrices2 + x2;

#ifdef KERNEL_PRINT_ENABLED
    printf("matrix = %d, pat = %d for tx = %d and state = %d :  u = %d\n", matrix, pattern, tx,
           state, u);
#endif

    // Load values into shared memory
    KW_LOCAL_MEM REAL sMatrix1[16];
    KW_LOCAL_MEM REAL sMatrix2[16];

    KW_LOCAL_MEM REAL fixedScalingFactors[PATTERN_BLOCK_SIZE * 4];

    if (patIdx < 4) // need to load 4*PATTERN_BLOCK_SIZE factors for this block
        fixedScalingFactors[patIdx * PATTERN_BLOCK_SIZE + tx] =
            scalingFactors[KW_GROUP_ID_0 * PATTERN_BLOCK_SIZE * 4 + patIdx * PATTERN_BLOCK_SIZE + tx];


    if (patIdx == 0 ) {
        sMatrix1[tx] = matrix1[tx]; // All coalesced memory reads
        sMatrix2[tx] = matrix2[tx];
    }

    KW_LOCAL_FENCE;

    if (pattern < totalPatterns) {
        int state1 = states1[pattern];
        int state2 = states2[pattern];

        if ( state1 < PADDED_STATE_COUNT && state2 < PADDED_STATE_COUNT) {
            partials3[u] = sMatrix1[state1 * 4 + state] * sMatrix2[state2 * 4 + state] / fixedScalingFactors[patIdx * 4 + pat];
        } else if (state1 < PADDED_STATE_COUNT) {
            partials3[u] = sMatrix1[state1 * 4 + state] / fixedScalingFactors[patIdx * 4 + pat];
        } else if (state2 < PADDED_STATE_COUNT) {
            partials3[u] = sMatrix2[state2 * 4 + state] / fixedScalingFactors[patIdx * 4 + pat];
        } else {
            partials3[u] = 1.0 / fixedScalingFactors[patIdx * 4 + pat];
        }
    }
}

KW_GLOBAL_KERNEL void kernelPartialsPartialsEdgeLikelihoods(KW_GLOBAL_VAR REAL* dPartialsTmp,
                                                          KW_GLOBAL_VAR REAL* dParentPartials,
                                                          KW_GLOBAL_VAR REAL* dChildParials,
                                                          KW_GLOBAL_VAR REAL* dTransMatrix,
                                                          int totalPatterns) {
   REAL sum1 = 0;

    int i;

    DETERMINE_INDICES_4();
    int patIdx16pat4 = multBy16(patIdx) | (tx & 0xC);
    int y = deltaPartialsByState + deltaPartialsByMatrix;
    KW_GLOBAL_VAR REAL* matrix1 = dTransMatrix + x2; // Points to *this* matrix

#ifdef KERNEL_PRINT_ENABLED
    printf("matrix = %d, pat = %d for tx = %d and state = %d :  u = %d\n", matrix, pattern, tx,
           state, u);
#endif

    // Load values into shared memory
    KW_LOCAL_MEM REAL sMatrix1[16];

    KW_LOCAL_MEM REAL sPartials1[PATTERN_BLOCK_SIZE * 4 * 4];
    KW_LOCAL_MEM REAL sPartials2[PATTERN_BLOCK_SIZE * 4 * 4];

    // copy PADDED_STATE_COUNT * PATTERN_BLOCK_SIZE lengthed partials
    if (pattern < totalPatterns) {
        sPartials1[multBy16(patIdx) | tx] = dParentPartials[y | tx]; // All coalesced memory reads
        sPartials2[multBy16(patIdx) | tx] = dChildParials  [y | tx];
    } else {
        sPartials1[multBy16(patIdx) | tx] = 0;
        sPartials2[multBy16(patIdx) | tx] = 0;
    }

    if (patIdx == 0 ) {
        sMatrix1[tx] = matrix1[tx]; // All coalesced memory reads
    }

    KW_LOCAL_FENCE;

    if (pattern < totalPatterns) { // Remove padded threads!

        i = pat;
        sum1  = sMatrix1[multBy4(i) | state] * sPartials1[patIdx16pat4 | i];
        i = (++i) & 0x3;
        sum1 += sMatrix1[multBy4(i) | state] * sPartials1[patIdx16pat4 | i];
        i = (++i) & 0x3;
        sum1 += sMatrix1[multBy4(i) | state] * sPartials1[patIdx16pat4 | i];
        i = (++i) & 0x3;
        sum1 += sMatrix1[multBy4(i) | state] * sPartials1[patIdx16pat4 | i];
        
        dPartialsTmp[u] = sum1 * sPartials2[patIdx16pat4 | state];
    }    

}



KW_GLOBAL_KERNEL void kernelPartialsPartialsEdgeLikelihoodsSecondDeriv(KW_GLOBAL_VAR REAL* dPartialsTmp,
                                                              KW_GLOBAL_VAR REAL* dFirstDerivTmp,
                                                              KW_GLOBAL_VAR REAL* dSecondDerivTmp,
                                                              KW_GLOBAL_VAR REAL* dParentPartials,
                                                              KW_GLOBAL_VAR REAL* dChildParials,
                                                              KW_GLOBAL_VAR REAL* dTransMatrix,
                                                              KW_GLOBAL_VAR REAL* dFirstDerivMatrix,
                                                              KW_GLOBAL_VAR REAL* dSecondDerivMatrix,
                                                              int totalPatterns) {
	   REAL sum1 = 0;
	   REAL sumFirstDeriv = 0;
	   REAL sumSecondDeriv = 0;

	    int i;

	    DETERMINE_INDICES_4();
	    int patIdx16pat4 = multBy16(patIdx) | (tx & 0xC);
	    int y = deltaPartialsByState + deltaPartialsByMatrix;
	    KW_GLOBAL_VAR REAL* matrix1 = dTransMatrix + x2; // Points to *this* matrix
	    KW_GLOBAL_VAR REAL* matrixFirstDeriv = dFirstDerivMatrix + x2;
	    KW_GLOBAL_VAR REAL* matrixSecondDeriv = dSecondDerivMatrix + x2;

	#ifdef KERNEL_PRINT_ENABLED
	    printf("matrix = %d, pat = %d for tx = %d and state = %d :  u = %d\n", matrix, pattern, tx,
	           state, u);
	#endif

	    // Load values into shared memory
	    KW_LOCAL_MEM REAL sMatrix1[16];
	    KW_LOCAL_MEM REAL sMatrixFirstDeriv[16];
	    KW_LOCAL_MEM REAL sMatrixSecondDeriv[16];

	    KW_LOCAL_MEM REAL sPartials1[PATTERN_BLOCK_SIZE * 4 * 4];
	    KW_LOCAL_MEM REAL sPartials2[PATTERN_BLOCK_SIZE * 4 * 4];

	    // copy PADDED_STATE_COUNT * PATTERN_BLOCK_SIZE lengthed partials
	    if (pattern < totalPatterns) {
	        sPartials1[multBy16(patIdx) | tx] = dParentPartials[y | tx]; // All coalesced memory reads
	        sPartials2[multBy16(patIdx) | tx] = dChildParials  [y | tx];
	    } else {
	        sPartials1[multBy16(patIdx) | tx] = 0;
	        sPartials2[multBy16(patIdx) | tx] = 0;
	    }

	    if (patIdx == 0 ) {
	        sMatrix1[tx] = matrix1[tx]; // All coalesced memory reads
	        sMatrixFirstDeriv[tx] = matrixFirstDeriv[tx];
	        sMatrixSecondDeriv[tx] = matrixSecondDeriv[tx];
	    }

	    KW_LOCAL_FENCE;

	    if (pattern < totalPatterns) { // Remove padded threads!

	        i = pat;
	        sum1  = sMatrix1[multBy4(i) | state] * sPartials1[patIdx16pat4 | i];
	        sumFirstDeriv  = sMatrixFirstDeriv[multBy4(i) | state] * sPartials1[patIdx16pat4 | i];
	        sumSecondDeriv  = sMatrixSecondDeriv[multBy4(i) | state] * sPartials1[patIdx16pat4 | i];
	        i = (++i) & 0x3;
	        sum1 += sMatrix1[multBy4(i) | state] * sPartials1[patIdx16pat4 | i];
	        sumFirstDeriv  += sMatrixFirstDeriv[multBy4(i) | state] * sPartials1[patIdx16pat4 | i];
	        sumSecondDeriv  += sMatrixSecondDeriv[multBy4(i) | state] * sPartials1[patIdx16pat4 | i];
	        i = (++i) & 0x3;
	        sum1 += sMatrix1[multBy4(i) | state] * sPartials1[patIdx16pat4 | i];
	        sumFirstDeriv  += sMatrixFirstDeriv[multBy4(i) | state] * sPartials1[patIdx16pat4 | i];
	        sumSecondDeriv  += sMatrixSecondDeriv[multBy4(i) | state] * sPartials1[patIdx16pat4 | i];
	        i = (++i) & 0x3;
	        sum1 += sMatrix1[multBy4(i) | state] * sPartials1[patIdx16pat4 | i];
	        sumFirstDeriv  += sMatrixFirstDeriv[multBy4(i) | state] * sPartials1[patIdx16pat4 | i];
	        sumSecondDeriv  += sMatrixSecondDeriv[multBy4(i) | state] * sPartials1[patIdx16pat4 | i];
	        
	        dPartialsTmp[u] = sum1 * sPartials2[patIdx16pat4 | state];
            dFirstDerivTmp[u] = sumFirstDeriv * sPartials2[patIdx16pat4 | state];
            dSecondDerivTmp[u] = sumSecondDeriv * sPartials2[patIdx16pat4 | state];
	    }    

	}


KW_GLOBAL_KERNEL void kernelStatesPartialsEdgeLikelihoods(KW_GLOBAL_VAR REAL* dPartialsTmp,
                                                         KW_GLOBAL_VAR REAL* dParentPartials,
                                                         KW_GLOBAL_VAR int* dChildStates,
                                                         KW_GLOBAL_VAR REAL* dTransMatrix,
                                                         int totalPatterns) {
    REAL sum1 = 0;

    DETERMINE_INDICES_4();
    int y = deltaPartialsByState + deltaPartialsByMatrix;
    KW_GLOBAL_VAR REAL* matrix1 = dTransMatrix + x2; // Points to *this* matrix
    
#ifdef KERNEL_PRINT_ENABLED
    printf("matrix = %d, pat = %d for tx = %d and state = %d :  u = %d\n", matrix, pattern, tx,
           state, u);
#endif

    // Load values into shared memory
    KW_LOCAL_MEM REAL sMatrix1[16];

    KW_LOCAL_MEM REAL sPartials2[PATTERN_BLOCK_SIZE * 4 * 4];

    // copy PADDED_STATE_COUNT * PATTERN_BLOCK_SIZE lengthed partials
    if (pattern < totalPatterns) {
        sPartials2[patIdx * 16 + tx] = dParentPartials[y + tx];
    } else {
        sPartials2[patIdx * 16 + tx] = 0;
    }

    if (patIdx == 0) {
        sMatrix1[tx] = matrix1[tx]; // All coalesced memory reads
    }

    KW_LOCAL_FENCE;

    if (pattern < totalPatterns) { // Remove padded threads!
        int state1 = dChildStates[pattern];

        if (state1 < PADDED_STATE_COUNT)
            sum1 = sMatrix1[state1 * 4 + state];
        else
            sum1 = 1.0;

        dPartialsTmp[u] = sum1 * sPartials2[patIdx * 16 + pat * 4 + state];
    }
}

KW_GLOBAL_KERNEL void kernelStatesPartialsEdgeLikelihoodsSecondDeriv(KW_GLOBAL_VAR REAL* dPartialsTmp,
                                                              KW_GLOBAL_VAR REAL* dFirstDerivTmp,
                                                              KW_GLOBAL_VAR REAL* dSecondDerivTmp,
                                                              KW_GLOBAL_VAR REAL* dParentPartials,
                                                              KW_GLOBAL_VAR int* dChildStates,
                                                              KW_GLOBAL_VAR REAL* dTransMatrix,
                                                              KW_GLOBAL_VAR REAL* dFirstDerivMatrix,
                                                              KW_GLOBAL_VAR REAL* dSecondDerivMatrix,
                                                              int totalPatterns) {
    REAL sum1 = 0;
    REAL sumFirstDeriv = 0;
    REAL sumSecondDeriv = 0;


    DETERMINE_INDICES_4();
    int y = deltaPartialsByState + deltaPartialsByMatrix;
    KW_GLOBAL_VAR REAL* matrix1 = dTransMatrix + x2; // Points to *this* matrix
    KW_GLOBAL_VAR REAL* matrixFirstDeriv = dFirstDerivMatrix + x2;
    KW_GLOBAL_VAR REAL* matrixSecondDeriv = dSecondDerivMatrix + x2;

    
#ifdef KERNEL_PRINT_ENABLED
    printf("matrix = %d, pat = %d for tx = %d and state = %d :  u = %d\n", matrix, pattern, tx,
           state, u);
#endif

    // Load values into shared memory
    KW_LOCAL_MEM REAL sMatrix1[16];
    KW_LOCAL_MEM REAL sMatrixFirstDeriv[16];
    KW_LOCAL_MEM REAL sMatrixSecondDeriv[16];

    KW_LOCAL_MEM REAL sPartials2[PATTERN_BLOCK_SIZE * 4 * 4];

    // copy PADDED_STATE_COUNT * PATTERN_BLOCK_SIZE lengthed partials
    if (pattern < totalPatterns) {
        sPartials2[patIdx * 16 + tx] = dParentPartials[y + tx];
    } else {
        sPartials2[patIdx * 16 + tx] = 0;
    }

    if (patIdx == 0) {
        sMatrix1[tx] = matrix1[tx]; // All coalesced memory reads
        sMatrixFirstDeriv[tx] = matrixFirstDeriv[tx]; // All coalesced memory reads
        sMatrixSecondDeriv[tx] = matrixSecondDeriv[tx]; // All coalesced memory reads
    }

    KW_LOCAL_FENCE;

    if (pattern < totalPatterns) { // Remove padded threads!
        int state1 = dChildStates[pattern];

        if (state1 < PADDED_STATE_COUNT) {
            sum1 = sMatrix1[state1 * 4 + state];
            sumFirstDeriv = sMatrixFirstDeriv[state1 * 4 + state];
            sumSecondDeriv = sMatrixSecondDeriv[state1 * 4 + state];
        } else {
            sum1 = 1.0;
            sumFirstDeriv = 0.0;
            sumSecondDeriv = 0.0;
        }

        dPartialsTmp[u] = sum1 * sPartials2[patIdx * 16 + pat * 4 + state];
        dFirstDerivTmp[u] = sumFirstDeriv * sPartials2[patIdx * 16 + pat * 4 + state];
        dSecondDerivTmp[u] = sumSecondDeriv * sPartials2[patIdx * 16 + pat * 4 + state];
    }
}


/*
 * Find a scaling factor for each pattern
 */
KW_GLOBAL_KERNEL void kernelPartialsDynamicScaling(KW_GLOBAL_VAR REAL* allPartials,
                                             KW_GLOBAL_VAR REAL* scalingFactors,
                                             int matrixCount) {
                                             
    int tx = KW_LOCAL_ID_0;
    
    int state = tx & 0x3;
    int pat = tx >> 2;
                             
    int patIdx = KW_GROUP_ID_0;
    
    int pattern = (patIdx << 2) + pat;
    int matrix = KW_LOCAL_ID_1;
    // TODO: Assumes matrixCount < MATRIX_BLOCK_SIZ
    
    // Patterns are always padded, so no reading/writing past end possible
    // Find start of patternBlock for thread-block
    int partialsOffset = (matrix * KW_NUM_GROUPS_0 + patIdx) << 4; //* 16;

    KW_LOCAL_MEM REAL partials[MATRIX_BLOCK_SIZE][16]; // 4 patterns at a time
    KW_LOCAL_MEM REAL storedPartials[MATRIX_BLOCK_SIZE][16];

    KW_LOCAL_MEM REAL matrixMax[4];
    
    if (matrix < matrixCount)
        partials[matrix][tx] = allPartials[partialsOffset + tx];          

    storedPartials[matrix][tx] = partials[matrix][tx];
           
    KW_LOCAL_FENCE;
    
    // Unrolled parallel max-reduction
    if (state < 2) {
        REAL compare1 = partials[matrix][tx];
        REAL compare2 = partials[matrix][tx + 2];
        if (compare2 > compare1)
            partials[matrix][tx] = compare2;
    }
    KW_LOCAL_FENCE;
    
    if (state < 1) {
        REAL compare1 = partials[matrix][tx];
        REAL compare2 = partials[matrix][tx + 1];
        if (compare2 > compare1)
            partials[matrix][tx] = compare2;
    }
    KW_LOCAL_FENCE;
 
    // Could also parallel-reduce here.
    if (state == 0 && matrix == 0) {
        matrixMax[pat] = 0;
        int m;
        for(m = 0; m < matrixCount; m++) {
            if (partials[m][tx] > matrixMax[pat])
                matrixMax[pat] = partials[m][tx];
        }
        
        if (matrixMax[pat] == 0)
        	matrixMax[pat] = 1.0;
   
        scalingFactors[pattern] = matrixMax[pat]; // TODO: Are these incoherent writes?
    }

    // Attempt at a parallel reduction that (1) does not work and (2) is slower
//    if (state == 0) {    
//        for (int i = MATRIX_BLOCK_SIZE / 2; i > 0; i >>= 1) {
//            if (matrix < i) {
//                REAL compare1 = partials[matrix][tx];
//                REAL compare2 = partials[matrix+i][tx];
//                if (compare2 > compare1)
//                    partials[matrix][tx] = compare2;              
//            }
//            KW_LOCAL_FENCE;
//        }         
//        
//        if (matrix == 0) {
//            matrixMax[pat] = partials[matrix][tx];
//            if (matrixMax[pat] == 0)
//                matrixMax[pat] = 1.0;
//                
//            scalingFactors[pattern] = matrixMax[pat];
//        }
//    }

    KW_LOCAL_FENCE;

    if (matrix < matrixCount)
        allPartials[partialsOffset + tx] = storedPartials[matrix][tx] / matrixMax[pat];
}


/*
 * Find a scaling factor for each pattern
 */
KW_GLOBAL_KERNEL void kernelPartialsDynamicScalingScalersLog(KW_GLOBAL_VAR REAL* allPartials,
                                                      KW_GLOBAL_VAR REAL* scalingFactors,
                                                      int matrixCount) {
                                             
    int tx = KW_LOCAL_ID_0;
    
    int state = tx & 0x3;
    int pat = tx >> 2;
                             
    int patIdx = KW_GROUP_ID_0;
    
    int pattern = (patIdx << 2) + pat;
    int matrix = KW_LOCAL_ID_1;
    // TODO: Assumes matrixCount < MATRIX_BLOCK_SIZ
    
    // Patterns are always padded, so no reading/writing past end possible
    // Find start of patternBlock for thread-block
    int partialsOffset = (matrix * KW_NUM_GROUPS_0 + patIdx) << 4; //* 16;

    KW_LOCAL_MEM REAL partials[MATRIX_BLOCK_SIZE][16]; // 4 patterns at a time
    KW_LOCAL_MEM REAL storedPartials[MATRIX_BLOCK_SIZE][16];

    KW_LOCAL_MEM REAL matrixMax[4];
    
    if (matrix < matrixCount)
        partials[matrix][tx] = allPartials[partialsOffset + tx];          

    storedPartials[matrix][tx] = partials[matrix][tx];
           
    KW_LOCAL_FENCE;
    
    // Unrolled parallel max-reduction
    if (state < 2) {
        REAL compare1 = partials[matrix][tx];
        REAL compare2 = partials[matrix][tx + 2];
        if (compare2 > compare1)
            partials[matrix][tx] = compare2;
    }
    KW_LOCAL_FENCE;
    
    if (state < 1) {
        REAL compare1 = partials[matrix][tx];
        REAL compare2 = partials[matrix][tx + 1];
        if (compare2 > compare1)
            partials[matrix][tx] = compare2;
    }
    KW_LOCAL_FENCE;
 
    // Could also parallel-reduce here.
    if (state == 0 && matrix == 0) {
        matrixMax[pat] = 0;
        int m;
        for(m = 0; m < matrixCount; m++) {
            if (partials[m][tx] > matrixMax[pat])
                matrixMax[pat] = partials[m][tx];
        }
        
        if (matrixMax[pat] == 0) {
        	matrixMax[pat] = 1.0;
            scalingFactors[pattern] = 0.0;
        } else {
            scalingFactors[pattern] = log(matrixMax[pat]);
        }
    }

    // Attempt at a parallel reduction that (1) does not work and (2) is slower
//    if (state == 0) {    
//        for (int i = MATRIX_BLOCK_SIZE / 2; i > 0; i >>= 1) {
//            if (matrix < i) {
//                REAL compare1 = partials[matrix][tx];
//                REAL compare2 = partials[matrix+i][tx];
//                if (compare2 > compare1)
//                    partials[matrix][tx] = compare2;              
//            }
//            KW_LOCAL_FENCE;
//        }         
//        
//        if (matrix == 0) {
//            matrixMax[pat] = partials[matrix][tx];
//            if (matrixMax[pat] == 0)
//                matrixMax[pat] = 1.0;
//                
//            scalingFactors[pattern] = matrixMax[pat];
//        }
//    }

    KW_LOCAL_FENCE;

    if (matrix < matrixCount)
        allPartials[partialsOffset + tx] = storedPartials[matrix][tx] / matrixMax[pat];
}

/*
 * Find a scaling factor for each pattern and accumulate into buffer
 */
KW_GLOBAL_KERNEL void kernelPartialsDynamicScalingAccumulate(KW_GLOBAL_VAR REAL* allPartials,
                                                       KW_GLOBAL_VAR REAL* scalingFactors,
                                                       KW_GLOBAL_VAR REAL* cumulativeScaling,
                                                       int matrixCount) {
    int tx = KW_LOCAL_ID_0;
    
    int state = tx & 0x3;
    int pat = tx >> 2;
                             
    int patIdx = KW_GROUP_ID_0;
    
    int pattern = (patIdx << 2) + pat;
    int matrix = KW_LOCAL_ID_1;
    // TODO: Assumes matrixCount < MATRIX_BLOCK_SIZ
    
    // Patterns are always padded, so no reading/writing past end possible
    // Find start of patternBlock for thread-block
    int partialsOffset = (matrix * KW_NUM_GROUPS_0 + patIdx) << 4; //* 16;

    KW_LOCAL_MEM REAL partials[MATRIX_BLOCK_SIZE][16]; // 4 patterns at a time
    KW_LOCAL_MEM REAL storedPartials[MATRIX_BLOCK_SIZE][16];

    KW_LOCAL_MEM REAL matrixMax[4];
    
    if (matrix < matrixCount)
        partials[matrix][tx] = allPartials[partialsOffset + tx];          

    storedPartials[matrix][tx] = partials[matrix][tx];
           
    KW_LOCAL_FENCE;
    
    // Unrolled parallel max-reduction
    if (state < 2) {
        REAL compare1 = partials[matrix][tx];
        REAL compare2 = partials[matrix][tx + 2];
        if (compare2 > compare1)
            partials[matrix][tx] = compare2;
    }
    KW_LOCAL_FENCE;
    
    if (state < 1) {
        REAL compare1 = partials[matrix][tx];
        REAL compare2 = partials[matrix][tx + 1];
        if (compare2 > compare1)
            partials[matrix][tx] = compare2;
    }
    KW_LOCAL_FENCE;
 
    // Could also parallel-reduce here.
    if (state == 0 && matrix == 0) {
        matrixMax[pat] = 0;
        int m;
        for(m = 0; m < matrixCount; m++) {
            if (partials[m][tx] > matrixMax[pat])
                matrixMax[pat] = partials[m][tx];
        }
        
        if (matrixMax[pat] == 0)
        	matrixMax[pat] = 1.0;
   
        scalingFactors[pattern] = matrixMax[pat]; 
        cumulativeScaling[pattern] += log(matrixMax[pat]);
    }

    KW_LOCAL_FENCE;

    if (matrix < matrixCount)
        allPartials[partialsOffset + tx] = storedPartials[matrix][tx] / matrixMax[pat];
        
}


KW_GLOBAL_KERNEL void kernelPartialsDynamicScalingAccumulateReciprocal(KW_GLOBAL_VAR REAL* allPartials,
                                                       KW_GLOBAL_VAR REAL* scalingFactors,
                                                       KW_GLOBAL_VAR REAL* cumulativeScaling,
                                                       int matrixCount) {
    int tx = KW_LOCAL_ID_0;
    
    int state = tx & 0x3;
    int pat = tx >> 2;
                             
    int patIdx = KW_GROUP_ID_0;
    
    int pattern = (patIdx << 2) + pat;
    int matrix = KW_LOCAL_ID_1;
    // TODO: Assumes matrixCount < MATRIX_BLOCK_SIZ
    
    // Patterns are always padded, so no reading/writing past end possible
    // Find start of patternBlock for thread-block
    int partialsOffset = (matrix * KW_NUM_GROUPS_0 + patIdx) << 4; //* 16;

    KW_LOCAL_MEM REAL partials[MATRIX_BLOCK_SIZE][16]; // 4 patterns at a time
    KW_LOCAL_MEM REAL storedPartials[MATRIX_BLOCK_SIZE][16];

    KW_LOCAL_MEM REAL matrixMax[4];
    
    if (matrix < matrixCount)
        partials[matrix][tx] = allPartials[partialsOffset + tx];          

    storedPartials[matrix][tx] = partials[matrix][tx];
           
    KW_LOCAL_FENCE;
    
    // Unrolled parallel max-reduction
    if (state < 2) {
        REAL compare1 = partials[matrix][tx];
        REAL compare2 = partials[matrix][tx + 2];
        if (compare2 > compare1)
            partials[matrix][tx] = compare2;
    }
    KW_LOCAL_FENCE;
    
    if (state < 1) {
        REAL compare1 = partials[matrix][tx];
        REAL compare2 = partials[matrix][tx + 1];
        if (compare2 > compare1)
            partials[matrix][tx] = compare2;
    }
    KW_LOCAL_FENCE;
 
    // Could also parallel-reduce here.
    if (state == 0 && matrix == 0) {
        matrixMax[pat] = 0;
        int m;
        for(m = 0; m < matrixCount; m++) {
            if (partials[m][tx] > matrixMax[pat])
                matrixMax[pat] = partials[m][tx];
        }
        
        if (matrixMax[pat] == 0)
        	matrixMax[pat] = 1.0;
   
        scalingFactors[pattern] = 1/matrixMax[pat]; 
        cumulativeScaling[pattern] += log(matrixMax[pat]);
    }

    KW_LOCAL_FENCE;

    if (matrix < matrixCount)
        allPartials[partialsOffset + tx] = storedPartials[matrix][tx] / matrixMax[pat];
        
}

/*
 * Find a scaling factor for each pattern and accumulate into buffer
 */
KW_GLOBAL_KERNEL void kernelPartialsDynamicScalingAccumulateScalersLog(KW_GLOBAL_VAR REAL* allPartials,
                                                                KW_GLOBAL_VAR REAL* scalingFactors,
                                                                KW_GLOBAL_VAR REAL* cumulativeScaling,
                                                                int matrixCount) {
    int tx = KW_LOCAL_ID_0;
    
    int state = tx & 0x3;
    int pat = tx >> 2;
                             
    int patIdx = KW_GROUP_ID_0;
    
    int pattern = (patIdx << 2) + pat;
    int matrix = KW_LOCAL_ID_1;
    // TODO: Assumes matrixCount < MATRIX_BLOCK_SIZ
    
    // Patterns are always padded, so no reading/writing past end possible
    // Find start of patternBlock for thread-block
    int partialsOffset = (matrix * KW_NUM_GROUPS_0 + patIdx) << 4; //* 16;

    KW_LOCAL_MEM REAL partials[MATRIX_BLOCK_SIZE][16]; // 4 patterns at a time
    KW_LOCAL_MEM REAL storedPartials[MATRIX_BLOCK_SIZE][16];

    KW_LOCAL_MEM REAL matrixMax[4];
    
    if (matrix < matrixCount)
        partials[matrix][tx] = allPartials[partialsOffset + tx];          

    storedPartials[matrix][tx] = partials[matrix][tx];
           
    KW_LOCAL_FENCE;
    
    // Unrolled parallel max-reduction
    if (state < 2) {
        REAL compare1 = partials[matrix][tx];
        REAL compare2 = partials[matrix][tx + 2];
        if (compare2 > compare1)
            partials[matrix][tx] = compare2;
    }
    KW_LOCAL_FENCE;
    
    if (state < 1) {
        REAL compare1 = partials[matrix][tx];
        REAL compare2 = partials[matrix][tx + 1];
        if (compare2 > compare1)
            partials[matrix][tx] = compare2;
    }
    KW_LOCAL_FENCE;
 
    // Could also parallel-reduce here.
    if (state == 0 && matrix == 0) {
        matrixMax[pat] = 0;
        int m;
        for(m = 0; m < matrixCount; m++) {
            if (partials[m][tx] > matrixMax[pat])
                matrixMax[pat] = partials[m][tx];
        }
        
        if (matrixMax[pat] == 0) {
        	matrixMax[pat] = 1.0;
            scalingFactors[pattern] = 0.0;
        } else {
            REAL logMax = log(matrixMax[pat]);
            scalingFactors[pattern] = logMax;
            cumulativeScaling[pattern] += logMax; // TODO: Fix, this is both a read and write
        }
    }

    KW_LOCAL_FENCE;

    if (matrix < matrixCount)
        allPartials[partialsOffset + tx] = storedPartials[matrix][tx] / matrixMax[pat];
        
}

KW_GLOBAL_KERNEL void kernelPartialsDynamicScalingAccumulateDifference(KW_GLOBAL_VAR REAL* allPartials,
                                                                 KW_GLOBAL_VAR REAL* scalingFactors,
                                                                 KW_GLOBAL_VAR REAL* existingScalingFactors,
                                                                 KW_GLOBAL_VAR REAL* cumulativeScaling,
                                                                 int matrixCount) {
    int tx = KW_LOCAL_ID_0;
    
    int state = tx & 0x3;
    int pat = tx >> 2;
                             
    int patIdx = KW_GROUP_ID_0;
    
    int pattern = (patIdx << 2) + pat;
    int matrix = KW_LOCAL_ID_1;
    // TODO: Assumes matrixCount < MATRIX_BLOCK_SIZ
    
    // Patterns are always padded, so no reading/writing past end possible
    // Find start of patternBlock for thread-block
    int partialsOffset = (matrix * KW_NUM_GROUPS_0 + patIdx) << 4; //* 16;

    KW_LOCAL_MEM REAL partials[MATRIX_BLOCK_SIZE][16]; // 4 patterns at a time
    KW_LOCAL_MEM REAL storedPartials[MATRIX_BLOCK_SIZE][16];

    KW_LOCAL_MEM REAL matrixMax[4];
    
    if (matrix < matrixCount)
        partials[matrix][tx] = allPartials[partialsOffset + tx];          

    storedPartials[matrix][tx] = partials[matrix][tx];
           
    KW_LOCAL_FENCE;
    
    // Unrolled parallel max-reduction
    if (state < 2) {
        REAL compare1 = partials[matrix][tx];
        REAL compare2 = partials[matrix][tx + 2];
        if (compare2 > compare1)
            partials[matrix][tx] = compare2;
    }
    KW_LOCAL_FENCE;
    
    if (state < 1) {
        REAL compare1 = partials[matrix][tx];
        REAL compare2 = partials[matrix][tx + 1];
        if (compare2 > compare1)
            partials[matrix][tx] = compare2;
    }
    KW_LOCAL_FENCE;
 
    // Could also parallel-reduce here.
    if (state == 0 && matrix == 0) {
        matrixMax[pat] = 0;
        int m;
        for(m = 0; m < matrixCount; m++) {
            if (partials[m][tx] > matrixMax[pat])
                matrixMax[pat] = partials[m][tx];
        }
        
        if (matrixMax[pat] == 0)
        	matrixMax[pat] = 1.0;
   
        REAL currentFactors = existingScalingFactors[pattern];
        scalingFactors[pattern] = 1/matrixMax[pat] * currentFactors; 
        cumulativeScaling[pattern] += (log(matrixMax[pat]));
    }

    KW_LOCAL_FENCE;

    if (matrix < matrixCount)
        allPartials[partialsOffset + tx] = storedPartials[matrix][tx] / matrixMax[pat];
        
}

KW_GLOBAL_KERNEL void kernelIntegrateLikelihoods(KW_GLOBAL_VAR REAL* dResult,
                                              KW_GLOBAL_VAR REAL* dRootPartials,
                                              KW_GLOBAL_VAR REAL* dWeights,
                                              KW_GLOBAL_VAR REAL* dFrequencies,
                                              int matrixCount,
                                              int patternCount) {
    int state   = KW_LOCAL_ID_0;
    int pat = KW_LOCAL_ID_1;
    int pattern = KW_GROUP_ID_0 * LIKE_PATTERN_BLOCK_SIZE + KW_LOCAL_ID_1;
    
    KW_LOCAL_MEM REAL stateFreq[4];
    
    // TODO: Currently assumes MATRIX_BLOCK_SIZE >= matrixCount
    KW_LOCAL_MEM REAL matrixProp[MATRIX_BLOCK_SIZE];
    KW_LOCAL_MEM REAL sum[LIKE_PATTERN_BLOCK_SIZE][4];

    // Load shared memory

    if (pat == 0) {
        stateFreq[state] = dFrequencies[state];
    }
    
    sum[pat][state] = 0;
    
    // TODO: Assumes matrixCount < LIKE_PATTERN_BLOCK_SIZE * 4
    if (pat * 4 + state < matrixCount) {
        matrixProp[pat * 4 + state] = dWeights[pat * 4 + state];
    }

    KW_LOCAL_FENCE;

    int u = state + pattern * PADDED_STATE_COUNT;
    int delta = patternCount * PADDED_STATE_COUNT;;

    for(int r = 0; r < matrixCount; r++) {
        sum[pat][state] += dRootPartials[u + delta * r] * matrixProp[r];
    }

    sum[pat][state] *= stateFreq[state];
    KW_LOCAL_FENCE;
    if (state < 2)
        sum[pat][state] += sum[pat][state + 2];
    KW_LOCAL_FENCE;
    if (state < 1) {
        sum[pat][state] += sum[pat][state + 1];
    }

    if (state == 0)
        dResult[pattern] = log(sum[pat][state]);
        
}

KW_GLOBAL_KERNEL void kernelIntegrateLikelihoodsFixedScale(KW_GLOBAL_VAR REAL* dResult,
                                                     KW_GLOBAL_VAR REAL* dRootPartials,
                                                     KW_GLOBAL_VAR REAL* dWeights,
                                                     KW_GLOBAL_VAR REAL* dFrequencies,
                                                     KW_GLOBAL_VAR REAL* dRootScalingFactors,
                                                     int matrixCount,
                                                     int patternCount) {
    int state   = KW_LOCAL_ID_0;
    int pat = KW_LOCAL_ID_1;
    int pattern = KW_GROUP_ID_0 * LIKE_PATTERN_BLOCK_SIZE + KW_LOCAL_ID_1;
    
    KW_LOCAL_MEM REAL stateFreq[4];
    
    // TODO: Currently assumes MATRIX_BLOCK_SIZE >= matrixCount
    KW_LOCAL_MEM REAL matrixProp[MATRIX_BLOCK_SIZE];
    KW_LOCAL_MEM REAL sum[LIKE_PATTERN_BLOCK_SIZE][4];

    // Load shared memory

    if (pat == 0) {
        stateFreq[state] = dFrequencies[state];
    }
    
    sum[pat][state] = 0;
    
    // TODO: Assumes matrixCount < LIKE_PATTERN_BLOCK_SIZE * 4
    if (pat * 4 + state < matrixCount) {
        matrixProp[pat * 4 + state] = dWeights[pat * 4 + state];
    }

    KW_LOCAL_FENCE;

    int u = state + pattern * PADDED_STATE_COUNT;
    int delta = patternCount * PADDED_STATE_COUNT;;

    for(int r = 0; r < matrixCount; r++) {
        sum[pat][state] += dRootPartials[u + delta * r] * matrixProp[r];
    }

    sum[pat][state] *= stateFreq[state];
    KW_LOCAL_FENCE;
    if (state < 2)
        sum[pat][state] += sum[pat][state + 2];
    KW_LOCAL_FENCE;
    if (state < 1) {
        sum[pat][state] += sum[pat][state + 1];
    }
    
    if (state == 0)
        dResult[pattern] = (log(sum[pat][state]) + dRootScalingFactors[pattern]);
}


KW_GLOBAL_KERNEL void kernelIntegrateLikelihoodsFixedScaleSecondDeriv(KW_GLOBAL_VAR REAL* dResult,
                                              KW_GLOBAL_VAR REAL* dFirstDerivResult,
                                              KW_GLOBAL_VAR REAL* dSecondDerivResult,
                                              KW_GLOBAL_VAR REAL* dRootPartials,
                                              KW_GLOBAL_VAR REAL* dRootFirstDeriv,
                                              KW_GLOBAL_VAR REAL* dRootSecondDeriv,
                                              KW_GLOBAL_VAR REAL* dWeights,
                                              KW_GLOBAL_VAR REAL* dFrequencies,
                                              KW_GLOBAL_VAR REAL* dRootScalingFactors,
                                              int matrixCount,
                                              int patternCount) {
    int state   = KW_LOCAL_ID_0;
    int pat = KW_LOCAL_ID_1;
    int pattern = KW_GROUP_ID_0 * LIKE_PATTERN_BLOCK_SIZE + KW_LOCAL_ID_1;
    
    REAL tmpLogLike = 0.0;
    REAL tmpFirstDeriv = 0.0;
    
    KW_LOCAL_MEM REAL stateFreq[4];
    
    // TODO: Currently assumes MATRIX_BLOCK_SIZE >= matrixCount
    KW_LOCAL_MEM REAL matrixProp[MATRIX_BLOCK_SIZE];
    KW_LOCAL_MEM REAL sum[LIKE_PATTERN_BLOCK_SIZE][4];
    KW_LOCAL_MEM REAL sumD1[LIKE_PATTERN_BLOCK_SIZE][4];
    KW_LOCAL_MEM REAL sumD2[LIKE_PATTERN_BLOCK_SIZE][4];

    // Load shared memory

    if (pat == 0) {
        stateFreq[state] = dFrequencies[state];
    }
    
    sum[pat][state] = 0;
    sumD1[pat][state] = 0;
    sumD2[pat][state] = 0;
    
    // TODO: Assumes matrixCount < LIKE_PATTERN_BLOCK_SIZE * 4
    if (pat * 4 + state < matrixCount) {
        matrixProp[pat * 4 + state] = dWeights[pat * 4 + state];
    }

    KW_LOCAL_FENCE;

    int u = state + pattern * PADDED_STATE_COUNT;
    int delta = patternCount * PADDED_STATE_COUNT;;

    for(int r = 0; r < matrixCount; r++) {
        sum[pat][state] += dRootPartials[u + delta * r] * matrixProp[r];
        sumD1[pat][state] += dRootFirstDeriv[u + delta * r] * matrixProp[r];
        sumD2[pat][state] += dRootSecondDeriv[u + delta * r] * matrixProp[r];
    }

    sum[pat][state] *= stateFreq[state];
    sumD1[pat][state] *= stateFreq[state];
    sumD2[pat][state] *= stateFreq[state];
    KW_LOCAL_FENCE;
    if (state < 2) {
        sum[pat][state] += sum[pat][state + 2];
        sumD1[pat][state] += sumD1[pat][state + 2];
        sumD2[pat][state] += sumD2[pat][state + 2];
    }
    KW_LOCAL_FENCE;
    if (state < 1) {
        sum[pat][state] += sum[pat][state + 1];
        sumD1[pat][state] += sumD1[pat][state + 1];
        sumD2[pat][state] += sumD2[pat][state + 1];
    }
    
    if (state == 0) {
        tmpLogLike = sum[pat][state];
        dResult[pattern] = (log(tmpLogLike) + dRootScalingFactors[pattern]);
        
        tmpFirstDeriv = sumD1[pat][state] / tmpLogLike;
        dFirstDerivResult[pattern] = tmpFirstDeriv;
        
        dSecondDerivResult[pattern] = (sumD2[pat][state] / tmpLogLike - tmpFirstDeriv * tmpFirstDeriv);
    }
}


KW_GLOBAL_KERNEL void kernelIntegrateLikelihoodsSecondDeriv(KW_GLOBAL_VAR REAL* dResult,
                                              KW_GLOBAL_VAR REAL* dFirstDerivResult,
                                              KW_GLOBAL_VAR REAL* dSecondDerivResult,
                                              KW_GLOBAL_VAR REAL* dRootPartials,
                                              KW_GLOBAL_VAR REAL* dRootFirstDeriv,
                                              KW_GLOBAL_VAR REAL* dRootSecondDeriv,
                                              KW_GLOBAL_VAR REAL* dWeights,
                                              KW_GLOBAL_VAR REAL* dFrequencies,
                                              int matrixCount,
                                              int patternCount) {
    int state   = KW_LOCAL_ID_0;
    int pat = KW_LOCAL_ID_1;
    int pattern = KW_GROUP_ID_0 * LIKE_PATTERN_BLOCK_SIZE + KW_LOCAL_ID_1;
    
    REAL tmpLogLike = 0.0;
    REAL tmpFirstDeriv = 0.0;
    
    KW_LOCAL_MEM REAL stateFreq[4];
    
    // TODO: Currently assumes MATRIX_BLOCK_SIZE >= matrixCount
    KW_LOCAL_MEM REAL matrixProp[MATRIX_BLOCK_SIZE];
    KW_LOCAL_MEM REAL sum[LIKE_PATTERN_BLOCK_SIZE][4];
    KW_LOCAL_MEM REAL sumD1[LIKE_PATTERN_BLOCK_SIZE][4];
    KW_LOCAL_MEM REAL sumD2[LIKE_PATTERN_BLOCK_SIZE][4];

    // Load shared memory

    if (pat == 0) {
        stateFreq[state] = dFrequencies[state];
    }
    
    sum[pat][state] = 0;
    sumD1[pat][state] = 0;
    sumD2[pat][state] = 0;
    
    // TODO: Assumes matrixCount < LIKE_PATTERN_BLOCK_SIZE * 4
    if (pat * 4 + state < matrixCount) {
        matrixProp[pat * 4 + state] = dWeights[pat * 4 + state];
    }

    KW_LOCAL_FENCE;

    int u = state + pattern * PADDED_STATE_COUNT;
    int delta = patternCount * PADDED_STATE_COUNT;;

    for(int r = 0; r < matrixCount; r++) {
        sum[pat][state] += dRootPartials[u + delta * r] * matrixProp[r];
        sumD1[pat][state] += dRootFirstDeriv[u + delta * r] * matrixProp[r];
        sumD2[pat][state] += dRootSecondDeriv[u + delta * r] * matrixProp[r];
    }

    sum[pat][state] *= stateFreq[state];
    sumD1[pat][state] *= stateFreq[state];
    sumD2[pat][state] *= stateFreq[state];
    KW_LOCAL_FENCE;
    if (state < 2) {
        sum[pat][state] += sum[pat][state + 2];
        sumD1[pat][state] += sumD1[pat][state + 2];
        sumD2[pat][state] += sumD2[pat][state + 2];
    }
    KW_LOCAL_FENCE;
    if (state < 1) {
        sum[pat][state] += sum[pat][state + 1];
        sumD1[pat][state] += sumD1[pat][state + 1];
        sumD2[pat][state] += sumD2[pat][state + 1];
    }
    
    if (state == 0) {
        tmpLogLike = sum[pat][state];
        dResult[pattern] = log(tmpLogLike);
        
        tmpFirstDeriv = sumD1[pat][state] / tmpLogLike;
        dFirstDerivResult[pattern] = tmpFirstDeriv;
        
        dSecondDerivResult[pattern] = (sumD2[pat][state] / tmpLogLike - tmpFirstDeriv * tmpFirstDeriv);
    }
}


KW_GLOBAL_KERNEL void kernelIntegrateLikelihoodsMulti(KW_GLOBAL_VAR REAL* dResult,
                                              KW_GLOBAL_VAR REAL* dRootPartials,
                                              KW_GLOBAL_VAR REAL* dWeights,
                                              KW_GLOBAL_VAR REAL* dFrequencies,
                                              int matrixCount,
                                              int patternCount,
											  int takeLog) {
    int state   = KW_LOCAL_ID_0;
    int pat = KW_LOCAL_ID_1;
    int pattern = KW_GROUP_ID_0 * LIKE_PATTERN_BLOCK_SIZE + KW_LOCAL_ID_1;
    
    KW_LOCAL_MEM REAL stateFreq[4];
    
    // TODO: Currently assumes MATRIX_BLOCK_SIZE >= matrixCount
    KW_LOCAL_MEM REAL matrixProp[MATRIX_BLOCK_SIZE];
    KW_LOCAL_MEM REAL sum[LIKE_PATTERN_BLOCK_SIZE][4];

    // Load shared memory

    if (pat == 0) {
        stateFreq[state] = dFrequencies[state];
    }
    
    sum[pat][state] = 0;
    
    // TODO: Assumes matrixCount < LIKE_PATTERN_BLOCK_SIZE * 4
    if (pat * 4 + state < matrixCount) {
        matrixProp[pat * 4 + state] = dWeights[pat * 4 + state];
    }

    KW_LOCAL_FENCE;

    int u = state + pattern * PADDED_STATE_COUNT;
    int delta = patternCount * PADDED_STATE_COUNT;;

    for(int r = 0; r < matrixCount; r++) {
        sum[pat][state] += dRootPartials[u + delta * r] * matrixProp[r];
    }

    sum[pat][state] *= stateFreq[state];
    KW_LOCAL_FENCE;
    if (state < 2)
        sum[pat][state] += sum[pat][state + 2];
    KW_LOCAL_FENCE;
    if (state < 1) {
        sum[pat][state] += sum[pat][state + 1];
    }
    
    if (state == 0) {
		if (takeLog == 0)
			dResult[pattern] = sum[pat][state];
		else if (takeLog == 1)
			dResult[pattern] = log(dResult[pattern] + sum[pat][state]);
		else 
			dResult[pattern] += sum[pat][state];
	}
        
}

KW_GLOBAL_KERNEL void kernelIntegrateLikelihoodsFixedScaleMulti(KW_GLOBAL_VAR REAL* dResult,
											  KW_GLOBAL_VAR REAL* dRootPartials,
                                              KW_GLOBAL_VAR REAL* dWeights,
                                              KW_GLOBAL_VAR REAL* dFrequencies,
                                              KW_GLOBAL_VAR REAL* dScalingFactors,
											  KW_GLOBAL_VAR unsigned int* dPtrQueue,
											  KW_GLOBAL_VAR REAL* dMaxScalingFactors,
											  KW_GLOBAL_VAR unsigned int* dIndexMaxScalingFactors,
                                              int matrixCount,
                                              int patternCount,
											  int subsetCount,
											  int subsetIndex) {
    int state   = KW_LOCAL_ID_0;
    int pat = KW_LOCAL_ID_1;
    int pattern = KW_GROUP_ID_0 * LIKE_PATTERN_BLOCK_SIZE + KW_LOCAL_ID_1;
    
    KW_LOCAL_MEM REAL stateFreq[4];
    
    // TODO: Currently assumes MATRIX_BLOCK_SIZE >= matrixCount
    KW_LOCAL_MEM REAL matrixProp[MATRIX_BLOCK_SIZE];
    KW_LOCAL_MEM REAL sum[LIKE_PATTERN_BLOCK_SIZE][4];

    // Load shared memory

    if (pat == 0) {
        stateFreq[state] = dFrequencies[state];
    }
    
    sum[pat][state] = 0;
    
    // TODO: Assumes matrixCount < LIKE_PATTERN_BLOCK_SIZE * 4
    if (pat * 4 + state < matrixCount) {
        matrixProp[pat * 4 + state] = dWeights[pat * 4 + state];
    }

    KW_LOCAL_FENCE;

    int u = state + pattern * PADDED_STATE_COUNT;
    int delta = patternCount * PADDED_STATE_COUNT;;

    for(int r = 0; r < matrixCount; r++) {
        sum[pat][state] += dRootPartials[u + delta * r] * matrixProp[r];
    }

    sum[pat][state] *= stateFreq[state];
    KW_LOCAL_FENCE;
    if (state < 2)
        sum[pat][state] += sum[pat][state + 2];
    KW_LOCAL_FENCE;
    if (state < 1) {
        sum[pat][state] += sum[pat][state + 1];
    }

	REAL cumulativeScalingFactor = (dScalingFactors + dPtrQueue[subsetIndex])[pattern];
	
	if (subsetIndex == 0) {
		int indexMaxScalingFactor = 0;
		REAL maxScalingFactor = cumulativeScalingFactor;
		for (int j = 1; j < subsetCount; j++) {
			REAL tmpScalingFactor = (dScalingFactors + dPtrQueue[j])[pattern];
			if (tmpScalingFactor > maxScalingFactor) {
				indexMaxScalingFactor = j;
				maxScalingFactor = tmpScalingFactor;
			}
		}
		
		dIndexMaxScalingFactors[pattern] = indexMaxScalingFactor;
		dMaxScalingFactors[pattern] = maxScalingFactor;	
		
		if (indexMaxScalingFactor != 0)
			sum[pat][state] *= exp((REAL)(cumulativeScalingFactor - maxScalingFactor));
			
		if (state == 0)
			dResult[pattern] = sum[pat][state];

        #ifdef FW_OPENCL
        KW_LOCAL_FENCE;
        #endif
	} else {
		if (subsetIndex != dIndexMaxScalingFactors[pattern])
			sum[pat][state] *= exp((REAL)(cumulativeScalingFactor - dMaxScalingFactors[pattern]));
	
		if (state == 0) {
			if (subsetIndex == subsetCount - 1)
				dResult[pattern] = (log(dResult[pattern] + sum[pat][state]) + dMaxScalingFactors[pattern]);
			else
				dResult[pattern] += sum[pat][state];
		}
	}        
}


KW_GLOBAL_KERNEL void kernelIntegrateLikelihoodsAutoScaling(KW_GLOBAL_VAR REAL* dResult,
                                                     KW_GLOBAL_VAR REAL* dRootPartials,
                                                     KW_GLOBAL_VAR REAL* dWeights,
                                                     KW_GLOBAL_VAR REAL* dFrequencies,
                                                     KW_GLOBAL_VAR int* dRootScalingFactors,
                                                     int matrixCount,
                                                     int patternCount) {
     int state   = KW_LOCAL_ID_0;
    int pat = KW_LOCAL_ID_1;
    int pattern = KW_GROUP_ID_0 * LIKE_PATTERN_BLOCK_SIZE + KW_LOCAL_ID_1;
    
    KW_LOCAL_MEM REAL stateFreq[4];
    
    // TODO: Currently assumes MATRIX_BLOCK_SIZE >= matrixCount
    KW_LOCAL_MEM REAL matrixProp[MATRIX_BLOCK_SIZE];
    KW_LOCAL_MEM REAL sum[LIKE_PATTERN_BLOCK_SIZE][4];

    // Load shared memory

    if (pat == 0) {
        stateFreq[state] = dFrequencies[state];
    }
    
    sum[pat][state] = 0;
    
    // TODO: Assumes matrixCount < LIKE_PATTERN_BLOCK_SIZE * 4
    if (pat * 4 + state < matrixCount) {
        matrixProp[pat * 4 + state] = dWeights[pat * 4 + state];
    }

    KW_LOCAL_FENCE;

    int u = state + pattern * PADDED_STATE_COUNT;
    int delta = patternCount * PADDED_STATE_COUNT;

    short maxScaleFactor = dRootScalingFactors[pattern];
    for(int r = 1; r < matrixCount; r++) {
        int tmpFactor = dRootScalingFactors[pattern + (r * patternCount)];
        if (tmpFactor > maxScaleFactor)
            maxScaleFactor = tmpFactor;
    }

    for(int r = 0; r < matrixCount; r++) {
        int tmpFactor = dRootScalingFactors[pattern + (r * patternCount)];
        if (tmpFactor != maxScaleFactor) {
            // TODO: verify which of the two methods below is faster
            int expTmp;
            sum[pat][state] += ldexp(frexp(dRootPartials[u + delta * r], &expTmp), expTmp + (tmpFactor - maxScaleFactor)) * matrixProp[r];
//            sum[pat][state] += dRootPartials[u + delta * r] * pow(2.0, tmpFactor - maxScaleFactor) * matrixProp[r];
        } else {
            sum[pat][state] += dRootPartials[u + delta * r] * matrixProp[r];
        }
    }

    sum[pat][state] *= stateFreq[state];
    KW_LOCAL_FENCE;
    if (state < 2)
        sum[pat][state] += sum[pat][state + 2];
    KW_LOCAL_FENCE;
    if (state < 1) {
        sum[pat][state] += sum[pat][state + 1];
    }

    if (state == 0)
        dResult[pattern] = (log(sum[pat][state]) + (M_LN2 * maxScaleFactor));
}

#ifdef CUDA
} // extern "C"
#endif //CUDA
