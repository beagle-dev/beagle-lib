/*
 *  BeagleCPU4StateSSEImpl.cpp
 *  BEAGLE
 *
 * Copyright 2009 Phylogenetic Likelihood Working Group
 *
 * This file is part of BEAGLE.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 *
 * @author Marc Suchard
 * @author David Swofford
 */

#ifndef BEAGLE_CPU_4STATE_SSE_IMPL_HPP
#define BEAGLE_CPU_4STATE_SSE_IMPL_HPP


#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cstring>
#include <cmath>
#include <cassert>

#include "libhmsbeagle/beagle.h"
#include "libhmsbeagle/CPU/BeagleCPU4StateSSEImpl.h"
#include "libhmsbeagle/CPU/SSEDefinitions.h"

/* Loads partials into SSE vectors */
#if 0
#define SSE_PREFETCH_PARTIALS(dest, src, v) \
		dest##0 = VEC_SPLAT(src[v + 0]); \
		dest##1 = VEC_SPLAT(src[v + 1]); \
		dest##2 = VEC_SPLAT(src[v + 2]); \
		dest##3 = VEC_SPLAT(src[v + 3]);
#else // Load four partials in two 128-bit memory transactions
#define SSE_PREFETCH_PARTIALS(dest, src, v) \
		V_Real tmp_##dest##01, tmp_##dest##23; \
		tmp_##dest##01 = _mm_load_pd(&src[v + 0]); \
		tmp_##dest##23 = _mm_load_pd(&src[v + 2]); \
		dest##0 = VEC_SHUFFLE0(tmp_##dest##01, tmp_##dest##01); \
		dest##1 = VEC_SHUFFLE1(tmp_##dest##01, tmp_##dest##01); \
		dest##2 = VEC_SHUFFLE0(tmp_##dest##23, tmp_##dest##23); \
		dest##3 = VEC_SHUFFLE1(tmp_##dest##23, tmp_##dest##23);
//  	dest##0 = _mm_shuffle_pd(tmp_##dest##01, tmp_##dest##01, _MM_SHUFFLE2(0,0)); \
// 	 	dest##1 = _mm_shuffle_pd(tmp_##dest##01, tmp_##dest##01, _MM_SHUFFLE2(1,1)); \
// 	 	dest##2 = _mm_shuffle_pd(tmp_##dest##23, tmp_##dest##23, _MM_SHUFFLE2(0,0)); \
// 		dest##3 = _mm_shuffle_pd(tmp_##dest##23, tmp_##dest##23, _MM_SHUFFLE2(1,1));

#define SSE_PREFETCH_VECTORIZED_PARTIALS(dest, src, v) \
        dest##01 = _mm_load_pd(&src[v + 0]); \
        dest##23 = _mm_load_pd(&src[v + 2]);

#define SSE_VECTORIZED_INNER_PRODUCT(lhs, rhs) \
        VEC_ADD(_mm_dp_pd(lhs##01, rhs##01, 0xff), _mm_dp_pd(lhs##23, rhs##23, 0xff))

#define SSE_SCHUR_PRODUCT_PARTIALS(dest, src, v, srcq) \
		V_Real tmp_##dest##01, tmp_##dest##23; \
		tmp_##dest##01 = VEC_MULT(_mm_load_pd(&src[v + 0]), srcq##01); \
		tmp_##dest##23 = VEC_MULT(_mm_load_pd(&src[v + 2]), srcq##23); \
		dest##0 = VEC_SHUFFLE0(tmp_##dest##01, tmp_##dest##01); \
		dest##1 = VEC_SHUFFLE1(tmp_##dest##01, tmp_##dest##01); \
		dest##2 = VEC_SHUFFLE0(tmp_##dest##23, tmp_##dest##23); \
		dest##3 = VEC_SHUFFLE1(tmp_##dest##23, tmp_##dest##23);
// 		dest##0 = _mm_shuffle_pd(tmp_##dest##01, tmp_##dest##01, _MM_SHUFFLE2(0,0)); \
// 		dest##1 = _mm_shuffle_pd(tmp_##dest##01, tmp_##dest##01, _MM_SHUFFLE2(1,1)); \
// 		dest##2 = _mm_shuffle_pd(tmp_##dest##23, tmp_##dest##23, _MM_SHUFFLE2(0,0)); \
// 		dest##3 = _mm_shuffle_pd(tmp_##dest##23, tmp_##dest##23, _MM_SHUFFLE2(1,1));
#endif

/* Loads (transposed) finite-time transition matrices into SSE vectors */
#define SSE_PREFETCH_MATRICES(src_m1, src_m2, dest_vu_m1, dest_vu_m2) \
	const double *m1 = (src_m1); \
	const double *m2 = (src_m2); \
	for (int i = 0; i < OFFSET; i++, m1++, m2++) { \
		dest_vu_m1[i][0].x[0] = m1[0*OFFSET]; \
		dest_vu_m1[i][0].x[1] = m1[1*OFFSET]; \
		dest_vu_m2[i][0].x[0] = m2[0*OFFSET]; \
		dest_vu_m2[i][0].x[1] = m2[1*OFFSET]; \
		dest_vu_m1[i][1].x[0] = m1[2*OFFSET]; \
		dest_vu_m1[i][1].x[1] = m1[3*OFFSET]; \
		dest_vu_m2[i][1].x[0] = m2[2*OFFSET]; \
		dest_vu_m2[i][1].x[1] = m2[3*OFFSET]; \
	}

/* Loads (only transpose matrix 2) finite-time transition matrices into SSE vectors */
#define SSE_PREFETCH_PRE_MATRICES(src_m1, src_m2, dest_vu_m1, dest_vu_m2) \
	const double *m1 = (src_m1); \
	const double *m2 = (src_m2); \
	for (int i = 0; i < 4; i++, m1+=OFFSET, m2++) { \
		dest_vu_m1[i][0].x[0] = m1[0]; \
		dest_vu_m1[i][0].x[1] = m1[1]; \
		dest_vu_m2[i][0].x[0] = m2[0*OFFSET]; \
		dest_vu_m2[i][0].x[1] = m2[1*OFFSET]; \
		dest_vu_m1[i][1].x[0] = m1[2]; \
		dest_vu_m1[i][1].x[1] = m1[3]; \
		dest_vu_m2[i][1].x[0] = m2[2*OFFSET]; \
		dest_vu_m2[i][1].x[1] = m2[3*OFFSET]; \
	} \
    for (int i = 4; i < OFFSET; i++, m2++) { \
		dest_vu_m2[i][0].x[0] = m2[0*OFFSET]; \
		dest_vu_m2[i][0].x[1] = m2[1*OFFSET]; \
		dest_vu_m2[i][1].x[0] = m2[2*OFFSET]; \
		dest_vu_m2[i][1].x[1] = m2[3*OFFSET]; \
	}

#define SSE_PREFETCH_MATRIX(src_m1, dest_vu_m1) \
	const double *m1 = (src_m1); \
	for (int i = 0; i < OFFSET; i++, m1++) { \
		dest_vu_m1[i][0].x[0] = m1[0*OFFSET]; \
		dest_vu_m1[i][0].x[1] = m1[1*OFFSET]; \
		dest_vu_m1[i][1].x[0] = m1[2*OFFSET]; \
		dest_vu_m1[i][1].x[1] = m1[3*OFFSET]; \
	}

namespace beagle {
namespace cpu {


BEAGLE_CPU_FACTORY_TEMPLATE
inline const char* getBeagleCPU4StateSSEName(){ return "CPU-4State-SSE-Unknown"; };

template<>
inline const char* getBeagleCPU4StateSSEName<double>(){ return "CPU-4State-SSE-Double"; };

template<>
inline const char* getBeagleCPU4StateSSEName<float>(){ return "CPU-4State-SSE-Single"; };

/*
 * Calculates partial likelihoods at a node when both children have states.
 */

BEAGLE_CPU_4_SSE_TEMPLATE
void BeagleCPU4StateSSEImpl<BEAGLE_CPU_4_SSE_DOUBLE>::calcStatesStates(double* destP,
                                                                       const int* states_q,
                                                                       const double* matrices_q,
                                                                       const int* states_r,
                                                                       const double* matrices_r,
                                                                       int startPattern,
                                                                       int endPattern) {

    int patternDefficit = kPatternCount + kExtraPatterns - endPattern;

	VecUnion vu_mq[OFFSET][2], vu_mr[OFFSET][2];

    int w = 0;
	V_Real *destPvec = (V_Real *)destP;

    for (int l = 0; l < kCategoryCount; l++) {
      destPvec += startPattern*2;
    	SSE_PREFETCH_MATRICES(matrices_q + w, matrices_r + w, vu_mq, vu_mr);

        for (int k = startPattern; k < endPattern; k++) {

            const int state_q = states_q[k];
            const int state_r = states_r[k];

            *destPvec++ = VEC_MULT(vu_mq[state_q][0].vx, vu_mr[state_r][0].vx);
            *destPvec++ = VEC_MULT(vu_mq[state_q][1].vx, vu_mr[state_r][1].vx);

        }

        w += OFFSET*4;
        if (kExtraPatterns) {
        	destPvec += kExtraPatterns * 2;
        }
        destPvec += patternDefficit * 2;
    }
}

/*
 * Calculates partial likelihoods at a node when one child has states and one has partials.
   SSE version
 */

BEAGLE_CPU_4_SSE_TEMPLATE
void BeagleCPU4StateSSEImpl<BEAGLE_CPU_4_SSE_DOUBLE>::calcStatesPartials(double* destP,
                                                                         const int* states_q,
                                                                         const double* matrices_q,
                                                                         const double* partials_r,
                                                                         const double* matrices_r,
                                                                         int startPattern,
                                                                         int endPattern) {

    int patternDefficit = kPatternCount + kExtraPatterns - endPattern;

    int v = 0;
    int w = 0;

 	VecUnion vu_mq[OFFSET][2], vu_mr[OFFSET][2];
	V_Real *destPvec = (V_Real *)destP;
	V_Real destr_01, destr_23;

    for (int l = 0; l < kCategoryCount; l++) {
      destPvec += startPattern*2;
      v += startPattern*4;
    	SSE_PREFETCH_MATRICES(matrices_q + w, matrices_r + w, vu_mq, vu_mr);

        for (int k = startPattern; k < endPattern; k++) {

            const int state_q = states_q[k];
            V_Real vp0, vp1, vp2, vp3;
            SSE_PREFETCH_PARTIALS(vp,partials_r,v);

			destr_01 = VEC_MULT(vp0, vu_mr[0][0].vx);
			destr_01 = VEC_MADD(vp1, vu_mr[1][0].vx, destr_01);
			destr_01 = VEC_MADD(vp2, vu_mr[2][0].vx, destr_01);
			destr_01 = VEC_MADD(vp3, vu_mr[3][0].vx, destr_01);
			destr_23 = VEC_MULT(vp0, vu_mr[0][1].vx);
			destr_23 = VEC_MADD(vp1, vu_mr[1][1].vx, destr_23);
			destr_23 = VEC_MADD(vp2, vu_mr[2][1].vx, destr_23);
			destr_23 = VEC_MADD(vp3, vu_mr[3][1].vx, destr_23);

            *destPvec++ = VEC_MULT(vu_mq[state_q][0].vx, destr_01);
            *destPvec++ = VEC_MULT(vu_mq[state_q][1].vx, destr_23);

            v += 4;
        }
        w += OFFSET*4;
        if (kExtraPatterns) {
        	destPvec += kExtraPatterns * 2;
        	v += kExtraPatterns * 4;
        }
        destPvec += patternDefficit * 2;
        v += patternDefficit * 4;
    }
}

BEAGLE_CPU_4_SSE_TEMPLATE
void BeagleCPU4StateSSEImpl<BEAGLE_CPU_4_SSE_DOUBLE>::calcStatesPartialsFixedScaling(double* destP,
                                                                                     const int* states_q,
                                                                                     const double* __restrict matrices_q,
                                                                                     const double* __restrict partials_r,
                                                                                     const double* __restrict matrices_r,
                                                                                     const double* __restrict scaleFactors,
                                                                                     int startPattern,
                                                                                     int endPattern) {

    int patternDefficit = kPatternCount + kExtraPatterns - endPattern;

    int v = 0;
    int w = 0;

    VecUnion vu_mq[OFFSET][2], vu_mr[OFFSET][2];
    V_Real *destPvec = (V_Real *)destP;
    V_Real destr_01, destr_23;

    for (int l = 0; l < kCategoryCount; l++) {
      destPvec += startPattern*2;
      v += startPattern*4;
    	SSE_PREFETCH_MATRICES(matrices_q + w, matrices_r + w, vu_mq, vu_mr);

        for (int k = startPattern; k < endPattern; k++) {

        	const V_Real scaleFactor = VEC_SPLAT(1.0/scaleFactors[k]);

            const int state_q = states_q[k];
            V_Real vp0, vp1, vp2, vp3;
            SSE_PREFETCH_PARTIALS(vp,partials_r,v);

			destr_01 = VEC_MULT(vp0, vu_mr[0][0].vx);
			destr_01 = VEC_MADD(vp1, vu_mr[1][0].vx, destr_01);
			destr_01 = VEC_MADD(vp2, vu_mr[2][0].vx, destr_01);
			destr_01 = VEC_MADD(vp3, vu_mr[3][0].vx, destr_01);
			destr_23 = VEC_MULT(vp0, vu_mr[0][1].vx);
			destr_23 = VEC_MADD(vp1, vu_mr[1][1].vx, destr_23);
			destr_23 = VEC_MADD(vp2, vu_mr[2][1].vx, destr_23);
			destr_23 = VEC_MADD(vp3, vu_mr[3][1].vx, destr_23);

            *destPvec++ = VEC_MULT(VEC_MULT(vu_mq[state_q][0].vx, destr_01), scaleFactor);
            *destPvec++ = VEC_MULT(VEC_MULT(vu_mq[state_q][1].vx, destr_23), scaleFactor);

            v += 4;
        }
        w += OFFSET*4;
        if (kExtraPatterns) {
        	destPvec += kExtraPatterns * 2;
        	v += kExtraPatterns * 4;
        }
        destPvec += patternDefficit * 2;
        v += patternDefficit * 4;
    }
}

BEAGLE_CPU_4_SSE_TEMPLATE
void BeagleCPU4StateSSEImpl<BEAGLE_CPU_4_SSE_DOUBLE>::calcPartialsPartials(double* destP,
                                                                           const double*  partials_q,
                                                                           const double*  matrices_q,
                                                                           const double*  partials_r,
                                                                           const double*  matrices_r,
                                                                           int startPattern,
                                                                           int endPattern) {

    int patternDefficit = kPatternCount + kExtraPatterns - endPattern;

    int v = 0;
    int w = 0;

    V_Real	destq_01, destq_23, destr_01, destr_23;
 	  VecUnion vu_mq[OFFSET][2], vu_mr[OFFSET][2];
	  V_Real *destPvec = (V_Real *)destP;

    for (int l = 0; l < kCategoryCount; l++) {
      destPvec += startPattern*2;
      v += startPattern*4;
		/* Load transition-probability matrices into vectors */
    	SSE_PREFETCH_MATRICES(matrices_q + w, matrices_r + w, vu_mq, vu_mr);

        for (int k = startPattern; k < endPattern; k++) {

#           if 1 && !defined(_WIN32)
            __builtin_prefetch (&partials_q[v+64]);
            __builtin_prefetch (&partials_r[v+64]);
//            __builtin_prefetch (destPvec+32,1,0);
#           endif

        	V_Real vpq_0, vpq_1, vpq_2, vpq_3;
        	SSE_PREFETCH_PARTIALS(vpq_,partials_q,v);

        	V_Real vpr_0, vpr_1, vpr_2, vpr_3;
        	SSE_PREFETCH_PARTIALS(vpr_,partials_r,v);

#			if 1	/* This would probably be faster on PPC/Altivec, which has a fused multiply-add
			           vector instruction */

			destq_01 = VEC_MULT(vpq_0, vu_mq[0][0].vx);
			destq_01 = VEC_MADD(vpq_1, vu_mq[1][0].vx, destq_01);
			destq_01 = VEC_MADD(vpq_2, vu_mq[2][0].vx, destq_01);
			destq_01 = VEC_MADD(vpq_3, vu_mq[3][0].vx, destq_01);
			destq_23 = VEC_MULT(vpq_0, vu_mq[0][1].vx);
			destq_23 = VEC_MADD(vpq_1, vu_mq[1][1].vx, destq_23);
			destq_23 = VEC_MADD(vpq_2, vu_mq[2][1].vx, destq_23);
			destq_23 = VEC_MADD(vpq_3, vu_mq[3][1].vx, destq_23);

			destr_01 = VEC_MULT(vpr_0, vu_mr[0][0].vx);
			destr_01 = VEC_MADD(vpr_1, vu_mr[1][0].vx, destr_01);
			destr_01 = VEC_MADD(vpr_2, vu_mr[2][0].vx, destr_01);
			destr_01 = VEC_MADD(vpr_3, vu_mr[3][0].vx, destr_01);
			destr_23 = VEC_MULT(vpr_0, vu_mr[0][1].vx);
			destr_23 = VEC_MADD(vpr_1, vu_mr[1][1].vx, destr_23);
			destr_23 = VEC_MADD(vpr_2, vu_mr[2][1].vx, destr_23);
			destr_23 = VEC_MADD(vpr_3, vu_mr[3][1].vx, destr_23);

#			else	/* SSE doesn't have a fused multiply-add, so a slight speed gain should be
                       achieved by decoupling these operations to avoid dependency stalls */

			V_Real a, b, c, d;

			a = VEC_MULT(vpq_0, vu_mq[0][0].vx);
			b = VEC_MULT(vpq_2, vu_mq[2][0].vx);
			c = VEC_MULT(vpq_0, vu_mq[0][1].vx);
			d = VEC_MULT(vpq_2, vu_mq[2][1].vx);
			a = VEC_MADD(vpq_1, vu_mq[1][0].vx, a);
			b = VEC_MADD(vpq_3, vu_mq[3][0].vx, b);
			c = VEC_MADD(vpq_1, vu_mq[1][1].vx, c);
			d = VEC_MADD(vpq_3, vu_mq[3][1].vx, d);
			destq_01 = VEC_ADD(a, b);
			destq_23 = VEC_ADD(c, d);

			a = VEC_MULT(vpr_0, vu_mr[0][0].vx);
			b = VEC_MULT(vpr_2, vu_mr[2][0].vx);
			c = VEC_MULT(vpr_0, vu_mr[0][1].vx);
			d = VEC_MULT(vpr_2, vu_mr[2][1].vx);
			a = VEC_MADD(vpr_1, vu_mr[1][0].vx, a);
			b = VEC_MADD(vpr_3, vu_mr[3][0].vx, b);
			c = VEC_MADD(vpr_1, vu_mr[1][1].vx, c);
			d = VEC_MADD(vpr_3, vu_mr[3][1].vx, d);
			destr_01 = VEC_ADD(a, b);
			destr_23 = VEC_ADD(c, d);

#			endif

#			if 1//
            destPvec[0] = VEC_MULT(destq_01, destr_01);
            destPvec[1] = VEC_MULT(destq_23, destr_23);
            destPvec += 2;

#			else	/* VEC_STORE did demonstrate a measurable performance gain as
					   it copies all (2/4) values to memory simultaneously;
					   I can no longer reproduce the performance gain (?) */

			VEC_STORE(destP + v + 0,VEC_MULT(destq_01, destr_01));
			VEC_STORE(destP + v + 2,VEC_MULT(destq_23, destr_23));

#			endif

            v += 4;
        }
        w += OFFSET*4;
        if (kExtraPatterns) {
        	destPvec += kExtraPatterns * 2;
        	v += kExtraPatterns * 4;
        }
        destPvec += patternDefficit * 2;
        v += patternDefficit * 4;
    }
}

BEAGLE_CPU_4_SSE_TEMPLATE
void BeagleCPU4StateSSEImpl<BEAGLE_CPU_4_SSE_DOUBLE>::calcPrePartialsPartials(double* destP,
                                                                           const double*  partials_q,
                                                                           const double*  matrices_q,
                                                                           const double*  partials_r,
                                                                           const double*  matrices_r,
                                                                           int startPattern,
                                                                           int endPattern) {

    int patternDefficit = kPatternCount + kExtraPatterns - endPattern;

    int v = 0;
    int w = 0;

    V_Real	destq_01, destq_23, destr_01, destr_23;
    VecUnion vu_mq[OFFSET][2], vu_mr[OFFSET][2];
    V_Real *destPvec = (V_Real *)destP;

    for (int l = 0; l < kCategoryCount; l++) {
        destPvec += startPattern*2;
        v += startPattern*4;
        /* Load transition-probability matrices into vectors */
        SSE_PREFETCH_PRE_MATRICES(matrices_q + w, matrices_r + w, vu_mq, vu_mr);

        for (int k = startPattern; k < endPattern; k++) {

#           if 1 && !defined(_WIN32)
            __builtin_prefetch (&partials_q[v+64]);
            __builtin_prefetch (&partials_r[v+64]);
//            __builtin_prefetch (destPvec+32,1,0);
#           endif

            V_Real vpr_0, vpr_1, vpr_2, vpr_3;
            SSE_PREFETCH_PARTIALS(vpr_,partials_r,v);

        	/* This would probably be faster on PPC/Altivec, which has a fused multiply-add
               vector instruction */

            destr_01 = VEC_MULT(vpr_0, vu_mr[0][0].vx);
            destr_01 = VEC_MADD(vpr_1, vu_mr[1][0].vx, destr_01);
            destr_01 = VEC_MADD(vpr_2, vu_mr[2][0].vx, destr_01);
            destr_01 = VEC_MADD(vpr_3, vu_mr[3][0].vx, destr_01);
            destr_23 = VEC_MULT(vpr_0, vu_mr[0][1].vx);
            destr_23 = VEC_MADD(vpr_1, vu_mr[1][1].vx, destr_23);
            destr_23 = VEC_MADD(vpr_2, vu_mr[2][1].vx, destr_23);
            destr_23 = VEC_MADD(vpr_3, vu_mr[3][1].vx, destr_23);

            V_Real vpq_0, vpq_1, vpq_2, vpq_3;
            SSE_SCHUR_PRODUCT_PARTIALS(vpq_, partials_q, v, destr_);

            destq_01 = VEC_MULT(vpq_0, vu_mq[0][0].vx);
            destq_01 = VEC_MADD(vpq_1, vu_mq[1][0].vx, destq_01);
            destq_01 = VEC_MADD(vpq_2, vu_mq[2][0].vx, destq_01);
            destq_01 = VEC_MADD(vpq_3, vu_mq[3][0].vx, destq_01);
            destq_23 = VEC_MULT(vpq_0, vu_mq[0][1].vx);
            destq_23 = VEC_MADD(vpq_1, vu_mq[1][1].vx, destq_23);
            destq_23 = VEC_MADD(vpq_2, vu_mq[2][1].vx, destq_23);
            destq_23 = VEC_MADD(vpq_3, vu_mq[3][1].vx, destq_23);

            destPvec[0] = destq_01;
            destPvec[1] = destq_23;
            destPvec += 2;

                /* VEC_STORE did demonstrate a measurable performance gain as
               it copies all (2/4) values to memory simultaneously;
               I can no longer reproduce the performance gain (?) */

            v += 4;
        }
        w += OFFSET*4;
        if (kExtraPatterns) {
            destPvec += kExtraPatterns * 2;
            v += kExtraPatterns * 4;
        }
        destPvec += patternDefficit * 2;
        v += patternDefficit * 4;
    }
}

BEAGLE_CPU_4_SSE_TEMPLATE
void BeagleCPU4StateSSEImpl<BEAGLE_CPU_4_SSE_DOUBLE>::calcPrePartialsStates(double* destP,
                                                                              const double*  partials_q,
                                                                              const double*  matrices_q,
                                                                              const int*     states_r,
                                                                              const double*  matrices_r,
                                                                              int startPattern,
                                                                              int endPattern) {

    int patternDefficit = kPatternCount + kExtraPatterns - endPattern;

    int v = 0;
    int w = 0;

    V_Real	destq_01, destq_23, destr_01, destr_23;
    VecUnion vu_mq[OFFSET][2], vu_mr[OFFSET][2];
    V_Real *destPvec = (V_Real *)destP;

    for (int l = 0; l < kCategoryCount; l++) {
        destPvec += startPattern*2;
        v += startPattern*4;
        /* Load transition-probability matrices into vectors */
        SSE_PREFETCH_PRE_MATRICES(matrices_q + w, matrices_r + w, vu_mq, vu_mr);

        for (int k = startPattern; k < endPattern; k++) {

            /* This would probably be faster on PPC/Altivec, which has a fused multiply-add
               vector instruction */

            const int state_r = states_r[k];

            V_Real destr_01, destr_23;
            destr_01 = vu_mr[state_r][0].vx;
            destr_23 = vu_mr[state_r][1].vx;

            V_Real vpq_0, vpq_1, vpq_2, vpq_3;
            SSE_SCHUR_PRODUCT_PARTIALS(vpq_, partials_q, v, destr_);


            destq_01 = VEC_MULT(vpq_0, vu_mq[0][0].vx);
            destq_01 = VEC_MADD(vpq_1, vu_mq[1][0].vx, destq_01);
            destq_01 = VEC_MADD(vpq_2, vu_mq[2][0].vx, destq_01);
            destq_01 = VEC_MADD(vpq_3, vu_mq[3][0].vx, destq_01);
            destq_23 = VEC_MULT(vpq_0, vu_mq[0][1].vx);
            destq_23 = VEC_MADD(vpq_1, vu_mq[1][1].vx, destq_23);
            destq_23 = VEC_MADD(vpq_2, vu_mq[2][1].vx, destq_23);
            destq_23 = VEC_MADD(vpq_3, vu_mq[3][1].vx, destq_23);


            destPvec[0] = destq_01;
            destPvec[1] = destq_23;
            destPvec += 2;

            /* VEC_STORE did demonstrate a measurable performance gain as
           it copies all (2/4) values to memory simultaneously;
           I can no longer reproduce the performance gain (?) */

            v += 4;
        }
        w += OFFSET*4;
        if (kExtraPatterns) {
            destPvec += kExtraPatterns * 2;
            v += kExtraPatterns * 4;
        }
        destPvec += patternDefficit * 2;
        v += patternDefficit * 4;
    }
}

BEAGLE_CPU_4_SSE_TEMPLATE
void BeagleCPU4StateSSEImpl<BEAGLE_CPU_4_SSE_DOUBLE>::calcCrossProductsPartials(const double *postOrderPartial,
                                                                                const double *preOrderPartial,
                                                                                const double *categoryRates,
                                                                                const double *categoryWeights,
                                                                                const double edgeLength,
                                                                                double *outCrossProducts,
                                                                                double *outSumSquaredDerivatives) {
#if 0
    return BeagleCPU4StateImpl<BEAGLE_CPU_4_SSE_DOUBLE>::calcCrossProductsPartials(postOrderPartial, preOrderPartial,
                                                                                   categoryRates, categoryWeights,
                                                                                   edgeLength, outCrossProducts,
                                                                                   outSumSquaredDerivatives);
#else

    std::array<V_Real, 8> vAcrossPatterns;
    vAcrossPatterns.fill(V_Real());

    std::array<V_Real, 8> vWithinPattern;

    for (int pattern = 0; pattern < kPatternCount; pattern++) {

        vWithinPattern.fill(V_Real());

        V_Real patternDenominator = VEC_SETZERO();

        for (int category = 0; category < kCategoryCount; category++) {

            const V_Real scale = VEC_SPLAT(categoryRates[category] * edgeLength);
            const V_Real weight = VEC_SPLAT(categoryWeights[category]);

            const int patternIndex = category * kPatternCount + pattern; // Bad memory access
            const int v = patternIndex * 4;

            V_Real pre0, pre1, pre2, pre3;
            SSE_PREFETCH_PARTIALS(pre, preOrderPartial, v);

            V_Real post01, post23;
            SSE_PREFETCH_VECTORIZED_PARTIALS(post, postOrderPartial, v);

            V_Real denominator = SSE_VECTORIZED_INNER_PRODUCT(tmp_pre, post);
            patternDenominator = VEC_MADD(denominator, weight, patternDenominator);

            V_Real weightScale = VEC_MULT(weight, scale);
            post01 = VEC_MULT(post01, weightScale);
            post23 = VEC_MULT(post23, weightScale);

            vWithinPattern[0 * 2 + 0] = VEC_MADD(pre0, post01, vWithinPattern[0 * 2 + 0]);
            vWithinPattern[0 * 2 + 1] = VEC_MADD(pre0, post23, vWithinPattern[0 * 2 + 1]);

            vWithinPattern[1 * 2 + 0] = VEC_MADD(pre1, post01, vWithinPattern[1 * 2 + 0]);
            vWithinPattern[1 * 2 + 1] = VEC_MADD(pre1, post23, vWithinPattern[1 * 2 + 1]);

            vWithinPattern[2 * 2 + 0] = VEC_MADD(pre2, post01, vWithinPattern[2 * 2 + 0]);
            vWithinPattern[2 * 2 + 1] = VEC_MADD(pre2, post23, vWithinPattern[2 * 2 + 1]);

            vWithinPattern[3 * 2 + 0] = VEC_MADD(pre3, post01, vWithinPattern[3 * 2 + 0]);
            vWithinPattern[3 * 2 + 1] = VEC_MADD(pre3, post23, vWithinPattern[3 * 2 + 1]);
        }

        const V_Real patternWeight = VEC_DIV(VEC_SPLAT(gPatternWeights[pattern]), patternDenominator);
        for (int k = 0; k < 8; k++) {
            vAcrossPatterns[k] = VEC_MADD(vWithinPattern[k], patternWeight, vAcrossPatterns[k]);
        }
    }

    for (int k = 0; k < 8; k++) {
        VEC_STORE(&outCrossProducts[k * 2],
                VEC_ADD(VEC_LOAD(&outCrossProducts[k * 2]), vAcrossPatterns[k]));
    }
#endif
}

BEAGLE_CPU_4_SSE_TEMPLATE
void BeagleCPU4StateSSEImpl<BEAGLE_CPU_4_SSE_DOUBLE>::calcCrossProductsStates(const int *tipStates,
                                                                              const double *preOrderPartial,
                                                                              const double *categoryRates,
                                                                              const double *categoryWeights,
                                                                              const double edgeLength,
                                                                              double *outCrossProducts,
                                                                              double *outSumSquaredDerivatives) {

    return BeagleCPU4StateImpl<BEAGLE_CPU_4_SSE_DOUBLE>::calcCrossProductsStates(tipStates, preOrderPartial, categoryRates,
                                                                                 categoryWeights, edgeLength, outCrossProducts,
                                                                                 outSumSquaredDerivatives);
}

BEAGLE_CPU_4_SSE_TEMPLATE template <bool DoDerivatives, bool DoSum, bool DoSumSquared>
void BeagleCPU4StateSSEImpl<BEAGLE_CPU_4_SSE_DOUBLE>::accumulateDerivativesImpl(
        double* outDerivatives,
        double* outSumDerivatives,
        double* outSumSquaredDerivatives) {

    V_Real vSum = VEC_SETZERO();
    V_Real vSumSquared = VEC_SETZERO();

    int k = 0;
    for (; k < kPatternCount - 1; k += 2) {

        V_Real numerator = VEC_LOAD(grandNumeratorDerivTmp + k);
        V_Real denominator = VEC_LOAD(grandDenominatorDerivTmp + k);
        V_Real derivative = VEC_DIV(numerator, denominator);
        V_Real patternWeight = VEC_LOAD(gPatternWeights + k);

        if (DoDerivatives) {
            VEC_STOREU(outDerivatives + k, derivative);
        }

        if (DoSum) {
            vSum = VEC_MADD(derivative, patternWeight, vSum);
        }

        if (DoSumSquared) {
            derivative = VEC_MULT(derivative, derivative);
            vSumSquared = VEC_MADD(derivative, patternWeight, vSumSquared);
        }
    }

    double sum;
    double sumSquared;

    if (DoSum) {
        sum = _mm_cvtsd_f64(VEC_ADD(vSum, VEC_SWAP(vSum)));
    }

    if (DoSumSquared) {
        sumSquared = _mm_cvtsd_f64(VEC_ADD(vSumSquared, VEC_SWAP(vSumSquared)));
    }

    for (; k < kPatternCount; ++k) {
        double derivative = grandNumeratorDerivTmp[k] / grandDenominatorDerivTmp[k];
        if (DoDerivatives) {
            outDerivatives[k] = derivative;
        }
        if (DoSum) {
            sum += derivative * gPatternWeights[k];
        }
        if (DoSumSquared) {
            sumSquared += derivative * derivative * gPatternWeights[k];
        }
    }

    if (DoSum) {
        *outSumDerivatives = sum;
    }

    if (DoSumSquared) {
        *outSumSquaredDerivatives = sumSquared;
    }
}

BEAGLE_CPU_4_SSE_TEMPLATE template <bool DoDerivatives, bool DoSum>
void BeagleCPU4StateSSEImpl<BEAGLE_CPU_4_SSE_DOUBLE>::accumulateDerivativesDispatch2(
        double* outDerivatives,
        double* outSumDerivatives,
        double* outSumSquaredDerivatives) {

    if (outSumSquaredDerivatives == NULL) {
        accumulateDerivativesImpl<DoDerivatives, DoSum, false>(
                outDerivatives, outSumDerivatives, outSumSquaredDerivatives);
    } else {
        accumulateDerivativesImpl<DoDerivatives, DoSum, true>(
                outDerivatives, outSumDerivatives, outSumSquaredDerivatives);
    }
}

BEAGLE_CPU_4_SSE_TEMPLATE template <bool DoDerivatives>
void BeagleCPU4StateSSEImpl<BEAGLE_CPU_4_SSE_DOUBLE>::accumulateDerivativesDispatch1(
        double* outDerivatives,
        double* outSumDerivatives,
        double* outSumSquaredDerivatives) {

    if (outSumDerivatives == NULL) {
        accumulateDerivativesDispatch2<DoDerivatives, false>(
                outDerivatives, outSumDerivatives, outSumSquaredDerivatives);
    } else {
        accumulateDerivativesDispatch2<DoDerivatives, true>(
                outDerivatives, outSumDerivatives, outSumSquaredDerivatives);
    }
}


BEAGLE_CPU_4_SSE_TEMPLATE
void BeagleCPU4StateSSEImpl<BEAGLE_CPU_4_SSE_DOUBLE>::accumulateDerivatives(double* outDerivatives,
                                                                            double* outSumDerivatives,
                                                                            double* outSumSquaredDerivatives) {
    if (outDerivatives == NULL) {
        accumulateDerivativesDispatch1<false>(
                outDerivatives, outSumDerivatives, outSumSquaredDerivatives);
    } else {
        accumulateDerivativesDispatch1<true>(
                outDerivatives, outSumDerivatives, outSumSquaredDerivatives);
    }
}

BEAGLE_CPU_4_SSE_TEMPLATE
void BeagleCPU4StateSSEImpl<BEAGLE_CPU_4_SSE_DOUBLE>::calcEdgeLogDerivativesPartials(const double* postOrderPartial,
                                                                                     const double* preOrderPartial,
                                                                                     const int firstDerivativeIndex,
                                                                                     const int secondDerivativeIndex,
                                                                                     const double* categoryRates,
                                                                                     const double* categoryWeights,
                                                                                     const int scalingFactorsIndex,
                                                                                     double* outDerivatives,
                                                                                     double* outSumDerivatives,
                                                                                     double* outSumSquaredDerivatives) {
    double* cl_p = integrationTmp;
    memset(cl_p, 0, (kPatternCount * kStateCount)*sizeof(double));

    const double* cl_r = preOrderPartial;
    const double* wt = categoryWeights;

    V_Real * vcl_r = (V_Real *)cl_r;

    int v = 0;
    int w = 0;
    const double* transMatrix = gTransitionMatrices[firstDerivativeIndex];

    for (int l = 0; l < kCategoryCount; l++) {

        /* Load transition-probability matrix into vectors */
        VecUnion vu_m[OFFSET][2];
        SSE_PREFETCH_MATRIX(transMatrix + w, vu_m);

        V_Real * vcl_p = (V_Real *)cl_p;
        V_Real vwt = VEC_SPLAT(wt[l]);
        for (int k = 0; k < kPatternCount; k++) {

            /* This would probably be faster on PPC/Altivec, which has a fused multiply-add
               vector instruction */

            V_Real vcl_q0, vcl_q1, vcl_q2, vcl_q3;
            SSE_PREFETCH_PARTIALS(vcl_q,postOrderPartial,v);

            V_Real vclp_01, vclp_23;
            vclp_01 = VEC_MULT(vcl_q0, vu_m[0][0].vx);
            vclp_01 = VEC_MADD(vcl_q1, vu_m[1][0].vx, vclp_01);
            vclp_01 = VEC_MADD(vcl_q2, vu_m[2][0].vx, vclp_01);
            vclp_01 = VEC_MADD(vcl_q3, vu_m[3][0].vx, vclp_01);
            vclp_23 = VEC_MULT(vcl_q0, vu_m[0][1].vx);
            vclp_23 = VEC_MADD(vcl_q1, vu_m[1][1].vx, vclp_23);
            vclp_23 = VEC_MADD(vcl_q2, vu_m[2][1].vx, vclp_23);
            vclp_23 = VEC_MADD(vcl_q3, vu_m[3][1].vx, vclp_23);

            vclp_01 = VEC_MULT(vclp_01, *vcl_r);
            tmp_vcl_q01 = VEC_MULT(tmp_vcl_q01, *vcl_r++);
            vclp_23 = VEC_MULT(vclp_23, *vcl_r);
            tmp_vcl_q23 = VEC_MULT(tmp_vcl_q23, *vcl_r++);

            V_Real vnumer = VEC_ADD(vclp_01, vclp_23);
            vnumer = VEC_ADD(vnumer, VEC_SWAP(vnumer));

            V_Real vdenom = VEC_ADD(tmp_vcl_q01, tmp_vcl_q23);
            vdenom = VEC_ADD(vdenom, VEC_SWAP(vdenom));

            double numer = _mm_cvtsd_f64(vnumer) * wt[l];
            double denon = _mm_cvtsd_f64(vdenom) * wt[l];

            grandNumeratorDerivTmp[k] += numer; // TODO Merge [numer, denom] into single SSE transactions
            grandDenominatorDerivTmp[k] += denon;

            v += 4;
        }
        w += 4*OFFSET;
        if (kExtraPatterns) {
            vcl_r += 2 * kExtraPatterns;
            v += 4 * kExtraPatterns;
        }
    }
}

BEAGLE_CPU_4_SSE_TEMPLATE
void BeagleCPU4StateSSEImpl<BEAGLE_CPU_4_SSE_DOUBLE>::calcEdgeLogDerivativesStates(const int* tipStates,
                                                                                   const double* preOrderPartial,
                                                                                   const int firstDerivativeIndex,
                                                                                   const int secondDerivativeIndex,
                                                                                   const double* categoryRates,
                                                                                   const double* categoryWeights,
                                                                                   double *outDerivatives,
                                                                                   double *outSumDerivatives,
                                                                                   double *outSumSquaredDerivatives) {
    double* cl_p = integrationTmp;
    memset(cl_p, 0, (kPatternCount * kStateCount)*sizeof(double));

    const double* cl_r = preOrderPartial;
    const double* wt = categoryWeights;

    const int* statesChild = tipStates;

    int w = 0;
    V_Real *vcl_r = (V_Real *)cl_r;
    const double* transMatrix = gTransitionMatrices[firstDerivativeIndex];

    for (int l = 0; l < kCategoryCount; l++) {

        /* Load transition-probability matrix into vectors */
        VecUnion vu_m[OFFSET][2];
        SSE_PREFETCH_MATRIX(transMatrix + w, vu_m);

        V_Real * vcl_p = (V_Real *)cl_p;
        V_Real vwt = VEC_SPLAT(wt[l]);
        for (int k = 0; k < kPatternCount; k++) {

            const int stateChild = statesChild[k];

            V_Real p01, p23;
            p01 = VEC_MULT(vu_m[stateChild][0].vx, *vcl_r++);
            p23 = VEC_MULT(vu_m[stateChild][1].vx, *vcl_r++);

            V_Real vnumer = VEC_ADD(p01, p23);
            vnumer = VEC_ADD(vnumer, VEC_SWAP(vnumer));

            double numer = _mm_cvtsd_f64(vnumer);
            double denom = cl_r[stateChild & 3]; cl_r += 4;

            grandNumeratorDerivTmp[k] += numer * wt[l];
            grandDenominatorDerivTmp[k] += denom * wt[l];
        }
        w += OFFSET*4;
        vcl_r += 2 * kExtraPatterns;
        cl_r += 4 * kExtraPatterns;
    }
}

BEAGLE_CPU_4_SSE_TEMPLATE
void BeagleCPU4StateSSEImpl<BEAGLE_CPU_4_SSE_DOUBLE>::calcPartialsPartialsFixedScaling(double* destP,
		                                                                                   const double* partials_q,
		                                                                                   const double* matrices_q,
		                                                                                   const double* partials_r,
		                                                                                   const double* matrices_r,
		                                                                                   const double* scaleFactors,
                                                                                       int startPattern,
                                                                                       int endPattern) {

    int patternDefficit = kPatternCount + kExtraPatterns - endPattern;

    int v = 0;
    int w = 0;

    V_Real	destq_01, destq_23, destr_01, destr_23;
 	VecUnion vu_mq[OFFSET][2], vu_mr[OFFSET][2];
	V_Real *destPvec = (V_Real *)destP;

	for (int l = 0; l < kCategoryCount; l++) {
      destPvec += startPattern*2;
      v += startPattern*4;
		/* Load transition-probability matrices into vectors */
    	SSE_PREFETCH_MATRICES(matrices_q + w, matrices_r + w, vu_mq, vu_mr);

        for (int k = startPattern; k < endPattern; k++) {

#           if 1 && !defined(_WIN32)
            __builtin_prefetch (&partials_q[v+64]);
            __builtin_prefetch (&partials_r[v+64]);
            //            __builtin_prefetch (destPvec+32,1,0);
#           endif

            // Prefetch scale factor
//            const V_Real scaleFactor = VEC_LOAD_SCALAR(scaleFactors + k);
        	// Option below appears faster, why?
        	const V_Real scaleFactor = VEC_SPLAT(1.0/scaleFactors[k]);

        	V_Real vpq_0, vpq_1, vpq_2, vpq_3;
        	SSE_PREFETCH_PARTIALS(vpq_,partials_q,v);

        	V_Real vpr_0, vpr_1, vpr_2, vpr_3;
        	SSE_PREFETCH_PARTIALS(vpr_,partials_r,v);

        	// TODO Make below into macro since this repeats from other calcPPs
			destq_01 = VEC_MULT(vpq_0, vu_mq[0][0].vx);
			destq_01 = VEC_MADD(vpq_1, vu_mq[1][0].vx, destq_01);
			destq_01 = VEC_MADD(vpq_2, vu_mq[2][0].vx, destq_01);
			destq_01 = VEC_MADD(vpq_3, vu_mq[3][0].vx, destq_01);
			destq_23 = VEC_MULT(vpq_0, vu_mq[0][1].vx);
			destq_23 = VEC_MADD(vpq_1, vu_mq[1][1].vx, destq_23);
			destq_23 = VEC_MADD(vpq_2, vu_mq[2][1].vx, destq_23);
			destq_23 = VEC_MADD(vpq_3, vu_mq[3][1].vx, destq_23);

			destr_01 = VEC_MULT(vpr_0, vu_mr[0][0].vx);
			destr_01 = VEC_MADD(vpr_1, vu_mr[1][0].vx, destr_01);
			destr_01 = VEC_MADD(vpr_2, vu_mr[2][0].vx, destr_01);
			destr_01 = VEC_MADD(vpr_3, vu_mr[3][0].vx, destr_01);
			destr_23 = VEC_MULT(vpr_0, vu_mr[0][1].vx);
			destr_23 = VEC_MADD(vpr_1, vu_mr[1][1].vx, destr_23);
			destr_23 = VEC_MADD(vpr_2, vu_mr[2][1].vx, destr_23);
			destr_23 = VEC_MADD(vpr_3, vu_mr[3][1].vx, destr_23);


            destPvec[0] = VEC_MULT(VEC_MULT(destq_01, destr_01), scaleFactor);
            destPvec[1] = VEC_MULT(VEC_MULT(destq_23, destr_23), scaleFactor);

            destPvec += 2;
            v += 4;
        }
        w += OFFSET*4;
        if (kExtraPatterns) {
        	destPvec += kExtraPatterns * 2;
        	v += kExtraPatterns * 4;
        }
        destPvec += patternDefficit * 2;
        v += patternDefficit * 4;
    }
}


BEAGLE_CPU_4_SSE_TEMPLATE
void BeagleCPU4StateSSEImpl<BEAGLE_CPU_4_SSE_FLOAT>::calcPartialsPartialsAutoScaling(float* destP,
                                                         const float*  partials_q,
                                                         const float*  matrices_q,
                                                         const float*  partials_r,
                                                         const float*  matrices_r,
                                                                 int* activateScaling) {
    BeagleCPU4StateImpl<BEAGLE_CPU_4_SSE_FLOAT>::calcPartialsPartialsAutoScaling(destP,
                                                     partials_q,
                                                     matrices_q,
                                                     partials_r,
                                                     matrices_r,
                                                     activateScaling);
}

BEAGLE_CPU_4_SSE_TEMPLATE
void BeagleCPU4StateSSEImpl<BEAGLE_CPU_4_SSE_DOUBLE>::calcPartialsPartialsAutoScaling(double* destP,
                                                                    const double*  partials_q,
                                                                    const double*  matrices_q,
                                                                    const double*  partials_r,
                                                                    const double*  matrices_r,
                                                                    int* activateScaling) {
    // TODO: implement calcPartialsPartialsAutoScaling with SSE
    BeagleCPU4StateImpl<BEAGLE_CPU_4_SSE_DOUBLE>::calcPartialsPartialsAutoScaling(destP,
                                                                partials_q,
                                                                matrices_q,
                                                                partials_r,
                                                                matrices_r,
                                                                activateScaling);
}

BEAGLE_CPU_4_SSE_TEMPLATE
int BeagleCPU4StateSSEImpl<BEAGLE_CPU_4_SSE_FLOAT>::calcEdgeLogLikelihoods(const int parIndex,
                                                          const int childIndex,
                                                          const int probIndex,
                                                          const int categoryWeightsIndex,
                                                          const int stateFrequenciesIndex,
                                                          const int scalingFactorsIndex,
                                                          double* outSumLogLikelihood) {
    return BeagleCPU4StateImpl<BEAGLE_CPU_4_SSE_FLOAT>::calcEdgeLogLikelihoods(
                                                              parIndex,
                                                              childIndex,
                                                              probIndex,
                                                              categoryWeightsIndex,
                                                              stateFrequenciesIndex,
                                                              scalingFactorsIndex,
                                                              outSumLogLikelihood);
}

BEAGLE_CPU_4_SSE_TEMPLATE
int BeagleCPU4StateSSEImpl<BEAGLE_CPU_4_SSE_DOUBLE>::calcEdgeLogLikelihoods(const int parIndex,
                                                            const int childIndex,
                                                            const int probIndex,
                                                            const int categoryWeightsIndex,
                                                            const int stateFrequenciesIndex,
                                                            const int scalingFactorsIndex,
                                                            double* outSumLogLikelihood) {
    // TODO: implement derivatives for calculateEdgeLnL

    int returnCode = BEAGLE_SUCCESS;

    assert(parIndex >= kTipCount);

    const double* cl_r = gPartials[parIndex];
    double* cl_p = integrationTmp;
    const double* transMatrix = gTransitionMatrices[probIndex];
    const double* wt = gCategoryWeights[categoryWeightsIndex];
    const double* freqs = gStateFrequencies[stateFrequenciesIndex];

    memset(cl_p, 0, (kPatternCount * kStateCount)*sizeof(double));

    if (childIndex < kTipCount && gTipStates[childIndex]) { // Integrate against a state at the child

        const int* statesChild = gTipStates[childIndex];

        int w = 0;
        V_Real *vcl_r = (V_Real *)cl_r;
        for(int l = 0; l < kCategoryCount; l++) {

            VecUnion vu_m[OFFSET][2];
            SSE_PREFETCH_MATRIX(transMatrix + w, vu_m)

           V_Real *vcl_p = (V_Real *)cl_p;

           for(int k = 0; k < kPatternCount; k++) {

                const int stateChild = statesChild[k];
                V_Real vwt = VEC_SPLAT(wt[l]);

                V_Real wtdPartials = VEC_MULT(*vcl_r++, vwt);
                *vcl_p = VEC_MADD(vu_m[stateChild][0].vx, wtdPartials, *vcl_p);
                vcl_p++;

                wtdPartials = VEC_MULT(*vcl_r++, vwt);
                *vcl_p = VEC_MADD(vu_m[stateChild][1].vx, wtdPartials, *vcl_p);
                vcl_p++;
            }
           w += OFFSET*4;
           vcl_r += 2 * kExtraPatterns;
        }
    } else { // Integrate against a partial at the child

        const double* cl_q = gPartials[childIndex];
        V_Real * vcl_r = (V_Real *)cl_r;
        int v = 0;
        int w = 0;

        for(int l = 0; l < kCategoryCount; l++) {

            V_Real * vcl_p = (V_Real *)cl_p;

            VecUnion vu_m[OFFSET][2];
            SSE_PREFETCH_MATRIX(transMatrix + w, vu_m)

            for(int k = 0; k < kPatternCount; k++) {
                V_Real vclp_01, vclp_23;
                V_Real vwt = VEC_SPLAT(wt[l]);

                V_Real vcl_q0, vcl_q1, vcl_q2, vcl_q3;
                SSE_PREFETCH_PARTIALS(vcl_q,cl_q,v);

                vclp_01 = VEC_MULT(vcl_q0, vu_m[0][0].vx);
                vclp_01 = VEC_MADD(vcl_q1, vu_m[1][0].vx, vclp_01);
                vclp_01 = VEC_MADD(vcl_q2, vu_m[2][0].vx, vclp_01);
                vclp_01 = VEC_MADD(vcl_q3, vu_m[3][0].vx, vclp_01);
                vclp_23 = VEC_MULT(vcl_q0, vu_m[0][1].vx);
                vclp_23 = VEC_MADD(vcl_q1, vu_m[1][1].vx, vclp_23);
                vclp_23 = VEC_MADD(vcl_q2, vu_m[2][1].vx, vclp_23);
                vclp_23 = VEC_MADD(vcl_q3, vu_m[3][1].vx, vclp_23);
                vclp_01 = VEC_MULT(vclp_01, vwt);
                vclp_23 = VEC_MULT(vclp_23, vwt);

                *vcl_p = VEC_MADD(vclp_01, *vcl_r++, *vcl_p);
                vcl_p++;
                *vcl_p = VEC_MADD(vclp_23, *vcl_r++, *vcl_p);
                vcl_p++;

                v += 4;
            }
            w += 4*OFFSET;
            if (kExtraPatterns) {
                vcl_r += 2 * kExtraPatterns;
                v += 4 * kExtraPatterns;
            }

        }
    }

    int u = 0;
    for(int k = 0; k < kPatternCount; k++) {
        double sumOverI = 0.0;
        for(int i = 0; i < kStateCount; i++) {
            sumOverI += freqs[i] * cl_p[u];
            u++;
        }

        outLogLikelihoodsTmp[k] = log(sumOverI);
    }


    if (scalingFactorsIndex != BEAGLE_OP_NONE) {
        const double* scalingFactors = gScaleBuffers[scalingFactorsIndex];
        for(int k=0; k < kPatternCount; k++)
            outLogLikelihoodsTmp[k] += scalingFactors[k];
    }

    *outSumLogLikelihood = 0.0;
    for (int i = 0; i < kPatternCount; i++) {
        *outSumLogLikelihood += outLogLikelihoodsTmp[i] * gPatternWeights[i];
    }

    if (*outSumLogLikelihood != *outSumLogLikelihood)
        returnCode = BEAGLE_ERROR_FLOATING_POINT;

    return returnCode;
}

BEAGLE_CPU_4_SSE_TEMPLATE
void BeagleCPU4StateSSEImpl<BEAGLE_CPU_4_SSE_FLOAT>::calcEdgeLogLikelihoodsByPartition(
                                                  const int* parentBufferIndices,
                                                  const int* childBufferIndices,
                                                  const int* probabilityIndices,
                                                  const int* categoryWeightsIndices,
                                                  const int* stateFrequenciesIndices,
                                                  const int* cumulativeScaleIndices,
                                                  const int* partitionIndices,
                                                  int partitionCount,
                                                  double* outSumLogLikelihoodByPartition) {

    BeagleCPU4StateImpl<BEAGLE_CPU_4_SSE_FLOAT>::calcEdgeLogLikelihoodsByPartition(
                                                  parentBufferIndices,
                                                  childBufferIndices,
                                                  probabilityIndices,
                                                  categoryWeightsIndices,
                                                  stateFrequenciesIndices,
                                                  cumulativeScaleIndices,
                                                  partitionIndices,
                                                  partitionCount,
                                                  outSumLogLikelihoodByPartition);
}

BEAGLE_CPU_4_SSE_TEMPLATE
void BeagleCPU4StateSSEImpl<BEAGLE_CPU_4_SSE_DOUBLE>::calcEdgeLogLikelihoodsByPartition(
                                                  const int* parentBufferIndices,
                                                  const int* childBufferIndices,
                                                  const int* probabilityIndices,
                                                  const int* categoryWeightsIndices,
                                                  const int* stateFrequenciesIndices,
                                                  const int* cumulativeScaleIndices,
                                                  const int* partitionIndices,
                                                  int partitionCount,
                                                  double* outSumLogLikelihoodByPartition) {



    double* cl_p = integrationTmp;

    for (int p = 0; p < partitionCount; p++) {
        int pIndex = partitionIndices[p];

        int startPattern = gPatternPartitionsStartPatterns[pIndex];
        int endPattern = gPatternPartitionsStartPatterns[pIndex + 1];

        memset(&cl_p[startPattern*kStateCount], 0, ((endPattern - startPattern) * kStateCount)*sizeof(double));

        const int parIndex = parentBufferIndices[p];
        const int childIndex = childBufferIndices[p];
        const int probIndex = probabilityIndices[p];
        const int categoryWeightsIndex = categoryWeightsIndices[p];
        const int stateFrequenciesIndex = stateFrequenciesIndices[p];
        const int scalingFactorsIndex = cumulativeScaleIndices[p];

        assert(parIndex >= kTipCount);

        const double* cl_r = gPartials[parIndex];
        const double* transMatrix = gTransitionMatrices[probIndex];
        const double* wt = gCategoryWeights[categoryWeightsIndex];
        const double* freqs = gStateFrequencies[stateFrequenciesIndex];


        if (childIndex < kTipCount && gTipStates[childIndex]) { // Integrate against a state at the child

            const int* statesChild = gTipStates[childIndex];

            int w = 0;
            V_Real *vcl_r = (V_Real *) (cl_r + startPattern * 4);
            for(int l = 0; l < kCategoryCount; l++) {

                VecUnion vu_m[OFFSET][2];
                SSE_PREFETCH_MATRIX(transMatrix + w, vu_m)

               V_Real *vcl_p = (V_Real *) (cl_p + startPattern * 4);

               for(int k = startPattern; k < endPattern; k++) {

                    const int stateChild = statesChild[k];
                    V_Real vwt = VEC_SPLAT(wt[l]);

                    V_Real wtdPartials = VEC_MULT(*vcl_r++, vwt);
                    *vcl_p = VEC_MADD(vu_m[stateChild][0].vx, wtdPartials, *vcl_p);
                    vcl_p++;

                    wtdPartials = VEC_MULT(*vcl_r++, vwt);
                    *vcl_p = VEC_MADD(vu_m[stateChild][1].vx, wtdPartials, *vcl_p);
                    vcl_p++;
                }
               w += OFFSET*4;
               vcl_r += 2 * kExtraPatterns;
               vcl_r += ((kPatternCount - endPattern) + startPattern) * 2;
            }
        } else { // Integrate against a partial at the child

            const double* cl_q = gPartials[childIndex];
            V_Real * vcl_r = (V_Real *)  (cl_r + startPattern * 4);
            int v = startPattern * 4;
            int w = 0;

            for(int l = 0; l < kCategoryCount; l++) {

                V_Real * vcl_p = (V_Real *) (cl_p + startPattern * 4);

                VecUnion vu_m[OFFSET][2];
                SSE_PREFETCH_MATRIX(transMatrix + w, vu_m)

                for(int k = startPattern; k < endPattern; k++) {
                    V_Real vclp_01, vclp_23;
                    V_Real vwt = VEC_SPLAT(wt[l]);

                    V_Real vcl_q0, vcl_q1, vcl_q2, vcl_q3;
                    SSE_PREFETCH_PARTIALS(vcl_q,cl_q,v);

                    vclp_01 = VEC_MULT(vcl_q0, vu_m[0][0].vx);
                    vclp_01 = VEC_MADD(vcl_q1, vu_m[1][0].vx, vclp_01);
                    vclp_01 = VEC_MADD(vcl_q2, vu_m[2][0].vx, vclp_01);
                    vclp_01 = VEC_MADD(vcl_q3, vu_m[3][0].vx, vclp_01);
                    vclp_23 = VEC_MULT(vcl_q0, vu_m[0][1].vx);
                    vclp_23 = VEC_MADD(vcl_q1, vu_m[1][1].vx, vclp_23);
                    vclp_23 = VEC_MADD(vcl_q2, vu_m[2][1].vx, vclp_23);
                    vclp_23 = VEC_MADD(vcl_q3, vu_m[3][1].vx, vclp_23);
                    vclp_01 = VEC_MULT(vclp_01, vwt);
                    vclp_23 = VEC_MULT(vclp_23, vwt);

                    *vcl_p = VEC_MADD(vclp_01, *vcl_r++, *vcl_p);
                    vcl_p++;
                    *vcl_p = VEC_MADD(vclp_23, *vcl_r++, *vcl_p);
                    vcl_p++;

                    v += 4;
                }
                w += 4*OFFSET;
                if (kExtraPatterns) {
                    vcl_r += 2 * kExtraPatterns;
                    v += 4 * kExtraPatterns;
                }

               vcl_r += ((kPatternCount - endPattern) + startPattern) * 2;
               v     += ((kPatternCount - endPattern) + startPattern) * 4;

            }
        }

        int u = startPattern * 4;
        for(int k = startPattern; k < endPattern; k++) {
            double sumOverI = 0.0;
            for(int i = 0; i < kStateCount; i++) {
                sumOverI += freqs[i] * cl_p[u];
                u++;
            }

            outLogLikelihoodsTmp[k] = log(sumOverI);
        }


        if (scalingFactorsIndex != BEAGLE_OP_NONE) {
            const double* scalingFactors = gScaleBuffers[scalingFactorsIndex];
            for(int k=startPattern; k < endPattern; k++)
                outLogLikelihoodsTmp[k] += scalingFactors[k];
        }

        outSumLogLikelihoodByPartition[p] = 0.0;
        for (int i = startPattern; i < endPattern; i++) {
            outSumLogLikelihoodByPartition[p] += outLogLikelihoodsTmp[i] * gPatternWeights[i];
        }

    }
}

BEAGLE_CPU_4_SSE_TEMPLATE
int BeagleCPU4StateSSEImpl<BEAGLE_CPU_4_SSE_FLOAT>::getPaddedPatternsModulus() {
	return 1;  // We currently do not vectorize across patterns
//	return 4;  // For single-precision, can operate on 4 patterns at a time
	// TODO Vectorize final log operations over patterns
}

BEAGLE_CPU_4_SSE_TEMPLATE
int BeagleCPU4StateSSEImpl<BEAGLE_CPU_4_SSE_DOUBLE>::getPaddedPatternsModulus() {
//	return 2;  // For double-precision, can operate on 2 patterns at a time
	return 1;  // We currently do not vectorize across patterns
}

BEAGLE_CPU_4_SSE_TEMPLATE
const char* BeagleCPU4StateSSEImpl<BEAGLE_CPU_4_SSE_FLOAT>::getName() {
	return  getBeagleCPU4StateSSEName<float>();
}

BEAGLE_CPU_4_SSE_TEMPLATE
const char* BeagleCPU4StateSSEImpl<BEAGLE_CPU_4_SSE_DOUBLE>::getName() {
    return  getBeagleCPU4StateSSEName<double>();
}


BEAGLE_CPU_4_SSE_TEMPLATE
const long BeagleCPU4StateSSEImpl<BEAGLE_CPU_4_SSE_FLOAT>::getFlags() {
	return  BEAGLE_FLAG_COMPUTATION_SYNCH |
            BEAGLE_FLAG_PROCESSOR_CPU |
            BEAGLE_FLAG_PRECISION_SINGLE |
            BEAGLE_FLAG_VECTOR_SSE |
            BEAGLE_FLAG_FRAMEWORK_CPU;
}

BEAGLE_CPU_4_SSE_TEMPLATE
const long BeagleCPU4StateSSEImpl<BEAGLE_CPU_4_SSE_DOUBLE>::getFlags() {
    return  BEAGLE_FLAG_COMPUTATION_SYNCH |
            BEAGLE_FLAG_PROCESSOR_CPU |
            BEAGLE_FLAG_PRECISION_DOUBLE |
            BEAGLE_FLAG_VECTOR_SSE |
            BEAGLE_FLAG_FRAMEWORK_CPU;
}



///////////////////////////////////////////////////////////////////////////////
// BeagleImplFactory public methods

BEAGLE_CPU_FACTORY_TEMPLATE
BeagleImpl* BeagleCPU4StateSSEImplFactory<BEAGLE_CPU_FACTORY_GENERIC>::createImpl(int tipCount,
                                             int partialsBufferCount,
                                             int compactBufferCount,
                                             int stateCount,
                                             int patternCount,
                                             int eigenBufferCount,
                                             int matrixBufferCount,
                                             int categoryCount,
                                             int scaleBufferCount,
                                             int resourceNumber,
                                             int pluginResourceNumber,
                                             long preferenceFlags,
                                             long requirementFlags,
                                             int* errorCode) {

    if (stateCount != 4) {
        return NULL;
    }

    BeagleCPU4StateSSEImpl<REALTYPE, T_PAD_4_SSE_DEFAULT, P_PAD_4_SSE_DEFAULT>* impl =
    		new BeagleCPU4StateSSEImpl<REALTYPE, T_PAD_4_SSE_DEFAULT, P_PAD_4_SSE_DEFAULT>();

    if (!CPUSupportsSSE()) {
        delete impl;
        return NULL;
    }

    try {
        if (impl->createInstance(tipCount, partialsBufferCount, compactBufferCount, stateCount,
                                 patternCount, eigenBufferCount, matrixBufferCount,
                                 categoryCount,scaleBufferCount, resourceNumber,
                                 pluginResourceNumber,
                                 preferenceFlags, requirementFlags) == 0)
            return impl;
    }
    catch(...) {
        if (DEBUGGING_OUTPUT)
            std::cerr << "exception in initialize\n";
        delete impl;
        throw;
    }

    delete impl;

    return NULL;
}

BEAGLE_CPU_FACTORY_TEMPLATE
const char* BeagleCPU4StateSSEImplFactory<BEAGLE_CPU_FACTORY_GENERIC>::getName() {
	return getBeagleCPU4StateSSEName<BEAGLE_CPU_FACTORY_GENERIC>();
}

template <>
const long BeagleCPU4StateSSEImplFactory<double>::getFlags() {
    return BEAGLE_FLAG_COMPUTATION_SYNCH |
           BEAGLE_FLAG_SCALING_MANUAL | BEAGLE_FLAG_SCALING_ALWAYS | BEAGLE_FLAG_SCALING_AUTO |
           BEAGLE_FLAG_THREADING_NONE | BEAGLE_FLAG_THREADING_CPP |
           BEAGLE_FLAG_PROCESSOR_CPU |
           BEAGLE_FLAG_VECTOR_SSE |
           BEAGLE_FLAG_PRECISION_DOUBLE |
           BEAGLE_FLAG_SCALERS_LOG | BEAGLE_FLAG_SCALERS_RAW |
           BEAGLE_FLAG_EIGEN_COMPLEX | BEAGLE_FLAG_EIGEN_REAL|
           BEAGLE_FLAG_INVEVEC_STANDARD | BEAGLE_FLAG_INVEVEC_TRANSPOSED |
           BEAGLE_FLAG_PREORDER_TRANSPOSE_MANUAL | BEAGLE_FLAG_PREORDER_TRANSPOSE_AUTO |
           BEAGLE_FLAG_FRAMEWORK_CPU;
}

template <>
const long BeagleCPU4StateSSEImplFactory<float>::getFlags() {
    return BEAGLE_FLAG_COMPUTATION_SYNCH |
           BEAGLE_FLAG_SCALING_MANUAL | BEAGLE_FLAG_SCALING_ALWAYS | BEAGLE_FLAG_SCALING_AUTO |
           BEAGLE_FLAG_THREADING_NONE | BEAGLE_FLAG_THREADING_CPP |
           BEAGLE_FLAG_PROCESSOR_CPU |
           BEAGLE_FLAG_VECTOR_SSE |
           BEAGLE_FLAG_PRECISION_SINGLE |
           BEAGLE_FLAG_SCALERS_LOG | BEAGLE_FLAG_SCALERS_RAW |
           BEAGLE_FLAG_EIGEN_COMPLEX | BEAGLE_FLAG_EIGEN_REAL |
           BEAGLE_FLAG_INVEVEC_STANDARD | BEAGLE_FLAG_INVEVEC_TRANSPOSED |
           BEAGLE_FLAG_PREORDER_TRANSPOSE_MANUAL | BEAGLE_FLAG_PREORDER_TRANSPOSE_AUTO |
           BEAGLE_FLAG_FRAMEWORK_CPU;
}


}
}

#endif //BEAGLE_CPU_4STATE_SSE_IMPL_HPP
