/*
 *  BeagleCPU4StateSSEImpl.cpp
 *  BEAGLE
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
		dest##0 = _mm_shuffle_pd(tmp_##dest##01, tmp_##dest##01, _MM_SHUFFLE2(0,0)); \
		dest##1 = _mm_shuffle_pd(tmp_##dest##01, tmp_##dest##01, _MM_SHUFFLE2(1,1)); \
		dest##2 = _mm_shuffle_pd(tmp_##dest##23, tmp_##dest##23, _MM_SHUFFLE2(0,0)); \
		dest##3 = _mm_shuffle_pd(tmp_##dest##23, tmp_##dest##23, _MM_SHUFFLE2(1,1));
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
void BeagleCPU4StateSSEImpl<BEAGLE_CPU_4_SSE_FLOAT>::calcStatesStates(float* destP,
                                     const int* states_q,
                                     const float* matrices_q,
                                     const int* states_r,
                                     const float* matrices_r) {

									 BeagleCPU4StateImpl<BEAGLE_CPU_4_SSE_FLOAT>::calcStatesStates(destP,
                                     states_q,
                                     matrices_q,
                                     states_r,
                                     matrices_r);

									 }


BEAGLE_CPU_4_SSE_TEMPLATE
void BeagleCPU4StateSSEImpl<BEAGLE_CPU_4_SSE_DOUBLE>::calcStatesStates(double* destP,
                                     const int* states_q,
                                     const double* matrices_q,
                                     const int* states_r,
                                     const double* matrices_r) {

	VecUnion vu_mq[OFFSET][2], vu_mr[OFFSET][2];

    int w = 0;
	V_Real *destPvec = (V_Real *)destP;

    for (int l = 0; l < kCategoryCount; l++) {

    	SSE_PREFETCH_MATRICES(matrices_q + w, matrices_r + w, vu_mq, vu_mr);

        for (int k = 0; k < kPatternCount; k++) {

            const int state_q = states_q[k];
            const int state_r = states_r[k];

            *destPvec++ = VEC_MULT(vu_mq[state_q][0].vx, vu_mr[state_r][0].vx);
            *destPvec++ = VEC_MULT(vu_mq[state_q][1].vx, vu_mr[state_r][1].vx);

        }

        w += OFFSET*4;
        if (kExtraPatterns)
        	destPvec += kExtraPatterns * 2;
    }
}

/*
 * Calculates partial likelihoods at a node when one child has states and one has partials.
   SSE version
 */
BEAGLE_CPU_4_SSE_TEMPLATE
void BeagleCPU4StateSSEImpl<BEAGLE_CPU_4_SSE_FLOAT>::calcStatesPartials(float* destP,
                                       const int* states_q,
                                       const float* matrices_q,
                                       const float* partials_r,
                                       const float* matrices_r) {
	BeagleCPU4StateImpl<BEAGLE_CPU_4_SSE_FLOAT>::calcStatesPartials(
									   destP,
									   states_q,
									   matrices_q,
									   partials_r,
									   matrices_r);
}



BEAGLE_CPU_4_SSE_TEMPLATE
void BeagleCPU4StateSSEImpl<BEAGLE_CPU_4_SSE_DOUBLE>::calcStatesPartials(double* destP,
                                       const int* states_q,
                                       const double* matrices_q,
                                       const double* partials_r,
                                       const double* matrices_r) {

    int v = 0;
    int w = 0;

 	VecUnion vu_mq[OFFSET][2], vu_mr[OFFSET][2];
	V_Real *destPvec = (V_Real *)destP;
	V_Real destr_01, destr_23;

    for (int l = 0; l < kCategoryCount; l++) {

    	SSE_PREFETCH_MATRICES(matrices_q + w, matrices_r + w, vu_mq, vu_mr);

        for (int k = 0; k < kPatternCount; k++) {

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
    }
}

BEAGLE_CPU_4_SSE_TEMPLATE
void BeagleCPU4StateSSEImpl<BEAGLE_CPU_4_SSE_FLOAT>::calcStatesPartialsFixedScaling(float* destP,
                                const int* states1,
                                const float* __restrict matrices1,
                                const float* __restrict partials2,
                                const float* __restrict matrices2,
                                const float* __restrict scaleFactors) {
	BeagleCPU4StateImpl<BEAGLE_CPU_4_SSE_FLOAT>::calcStatesPartialsFixedScaling(
									   destP,
									   states1,
									   matrices1,
									   partials2,
									   matrices2,
									   scaleFactors);
}

BEAGLE_CPU_4_SSE_TEMPLATE
void BeagleCPU4StateSSEImpl<BEAGLE_CPU_4_SSE_DOUBLE>::calcStatesPartialsFixedScaling(double* destP,
                                const int* states_q,
                                const double* __restrict matrices_q,
                                const double* __restrict partials_r,
                                const double* __restrict matrices_r,
                                const double* __restrict scaleFactors) {


    int v = 0;
    int w = 0;

 	VecUnion vu_mq[OFFSET][2], vu_mr[OFFSET][2];
	V_Real *destPvec = (V_Real *)destP;
	V_Real destr_01, destr_23;

    for (int l = 0; l < kCategoryCount; l++) {

    	SSE_PREFETCH_MATRICES(matrices_q + w, matrices_r + w, vu_mq, vu_mr);

        for (int k = 0; k < kPatternCount; k++) {

        	const V_Real scaleFactor = VEC_SPLAT(scaleFactors[k]);

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

            *destPvec++ = VEC_DIV(VEC_MULT(vu_mq[state_q][0].vx, destr_01), scaleFactor);
            *destPvec++ = VEC_DIV(VEC_MULT(vu_mq[state_q][1].vx, destr_23), scaleFactor);

            v += 4;
        }
        w += OFFSET*4;
        if (kExtraPatterns) {
        	destPvec += kExtraPatterns * 2;
        	v += kExtraPatterns * 4;
        }
    }
}

BEAGLE_CPU_4_SSE_TEMPLATE
void BeagleCPU4StateSSEImpl<BEAGLE_CPU_4_SSE_FLOAT>::calcPartialsPartials(float* destP,
                                                  const float*  partials_q,
                                                  const float*  matrices_q,
                                                  const float*  partials_r,
                                                  const float*  matrices_r) {

	BeagleCPU4StateImpl<BEAGLE_CPU_4_SSE_FLOAT>::calcPartialsPartials(destP,
                                                  partials_q,
                                                  matrices_q,
                                                  partials_r,
                                                  matrices_r);
}

BEAGLE_CPU_4_SSE_TEMPLATE
void BeagleCPU4StateSSEImpl<BEAGLE_CPU_4_SSE_DOUBLE>::calcPartialsPartials(double* destP,
                                                  const double*  partials_q,
                                                  const double*  matrices_q,
                                                  const double*  partials_r,
                                                  const double*  matrices_r) {

    int v = 0;
    int w = 0;

    V_Real	destq_01, destq_23, destr_01, destr_23;
 	VecUnion vu_mq[OFFSET][2], vu_mr[OFFSET][2];
	V_Real *destPvec = (V_Real *)destP;

    for (int l = 0; l < kCategoryCount; l++) {

		/* Load transition-probability matrices into vectors */
    	SSE_PREFETCH_MATRICES(matrices_q + w, matrices_r + w, vu_mq, vu_mr);

        for (int k = 0; k < kPatternCount; k++) {
            
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
    }
}

BEAGLE_CPU_4_SSE_TEMPLATE
void BeagleCPU4StateSSEImpl<BEAGLE_CPU_4_SSE_FLOAT>::calcPartialsPartialsFixedScaling(float* destP,
                                        const float*  child0Partials,
                                        const float*  child0TransMat,
                                        const float*  child1Partials,
                                        const float*  child1TransMat,
                                        const float*  scaleFactors) {

	BeagleCPU4StateImpl<BEAGLE_CPU_4_SSE_FLOAT>::calcPartialsPartialsFixedScaling(
			destP,
			child0Partials,
			child0TransMat,
			child1Partials,
			child1TransMat,
			scaleFactors);
}

BEAGLE_CPU_4_SSE_TEMPLATE
void BeagleCPU4StateSSEImpl<BEAGLE_CPU_4_SSE_DOUBLE>::calcPartialsPartialsFixedScaling(double* destP,
		                                                        const double* partials_q,
		                                                        const double* matrices_q,
		                                                        const double* partials_r,
		                                                        const double* matrices_r,
		                                                        const double* scaleFactors) {

    int v = 0;
    int w = 0;

    V_Real	destq_01, destq_23, destr_01, destr_23;
 	VecUnion vu_mq[OFFSET][2], vu_mr[OFFSET][2];
	V_Real *destPvec = (V_Real *)destP;

	for (int l = 0; l < kCategoryCount; l++) {

		/* Load transition-probability matrices into vectors */
    	SSE_PREFETCH_MATRICES(matrices_q + w, matrices_r + w, vu_mq, vu_mr);

        for (int k = 0; k < kPatternCount; k++) {

#           if 1 && !defined(_WIN32)
            __builtin_prefetch (&partials_q[v+64]);
            __builtin_prefetch (&partials_r[v+64]);
            //            __builtin_prefetch (destPvec+32,1,0);
#           endif
            
            // Prefetch scale factor
//            const V_Real scaleFactor = VEC_LOAD_SCALAR(scaleFactors + k);
        	// Option below appears faster, why?
        	const V_Real scaleFactor = VEC_SPLAT(scaleFactors[k]);

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

            destPvec[0] = VEC_DIV(VEC_MULT(destq_01, destr_01), scaleFactor);
            destPvec[1] = VEC_DIV(VEC_MULT(destq_23, destr_23), scaleFactor);

            destPvec += 2;
            v += 4;
        }
        w += OFFSET*4;
        if (kExtraPatterns) {
        	destPvec += kExtraPatterns * 2;
        	v += kExtraPatterns * 4;
        }
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
                *vcl_p++ = VEC_MADD(vu_m[stateChild][0].vx, wtdPartials, *vcl_p);

                wtdPartials = VEC_MULT(*vcl_r++, vwt);
                *vcl_p++ = VEC_MADD(vu_m[stateChild][1].vx, wtdPartials, *vcl_p);
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

                *vcl_p++ = VEC_MADD(vclp_01, *vcl_r++, *vcl_p);
                *vcl_p++ = VEC_MADD(vclp_23, *vcl_r++, *vcl_p);

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

    if (!(*outSumLogLikelihood - *outSumLogLikelihood == 0.0))
        returnCode = BEAGLE_ERROR_FLOATING_POINT;
        
    return returnCode;
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
            BEAGLE_FLAG_THREADING_NONE |
            BEAGLE_FLAG_PROCESSOR_CPU |
            BEAGLE_FLAG_PRECISION_SINGLE |
            BEAGLE_FLAG_VECTOR_SSE;
}

BEAGLE_CPU_4_SSE_TEMPLATE
const long BeagleCPU4StateSSEImpl<BEAGLE_CPU_4_SSE_DOUBLE>::getFlags() {
    return  BEAGLE_FLAG_COMPUTATION_SYNCH |
            BEAGLE_FLAG_THREADING_NONE |
            BEAGLE_FLAG_PROCESSOR_CPU |
            BEAGLE_FLAG_PRECISION_DOUBLE |
            BEAGLE_FLAG_VECTOR_SSE;
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
                                 categoryCount,scaleBufferCount, resourceNumber, preferenceFlags, requirementFlags) == 0)
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
           BEAGLE_FLAG_THREADING_NONE |
           BEAGLE_FLAG_PROCESSOR_CPU |
           BEAGLE_FLAG_VECTOR_SSE |
           BEAGLE_FLAG_PRECISION_DOUBLE |
           BEAGLE_FLAG_SCALERS_LOG | BEAGLE_FLAG_SCALERS_RAW |
           BEAGLE_FLAG_EIGEN_COMPLEX | BEAGLE_FLAG_EIGEN_REAL|
           BEAGLE_FLAG_INVEVEC_STANDARD | BEAGLE_FLAG_INVEVEC_TRANSPOSED;
}

template <>
const long BeagleCPU4StateSSEImplFactory<float>::getFlags() {
    return BEAGLE_FLAG_COMPUTATION_SYNCH |
           BEAGLE_FLAG_SCALING_MANUAL | BEAGLE_FLAG_SCALING_ALWAYS | BEAGLE_FLAG_SCALING_AUTO |
           BEAGLE_FLAG_THREADING_NONE |
           BEAGLE_FLAG_PROCESSOR_CPU |
           BEAGLE_FLAG_VECTOR_SSE |
           BEAGLE_FLAG_PRECISION_SINGLE |
           BEAGLE_FLAG_SCALERS_LOG | BEAGLE_FLAG_SCALERS_RAW |
           BEAGLE_FLAG_EIGEN_COMPLEX | BEAGLE_FLAG_EIGEN_REAL |
           BEAGLE_FLAG_INVEVEC_STANDARD | BEAGLE_FLAG_INVEVEC_TRANSPOSED;
}


}
}

#endif //BEAGLE_CPU_4STATE_SSE_IMPL_HPP
