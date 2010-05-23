/*
 *  BeagleCPUSSEImpl.hpp
 *  BEAGLE
 *
 * Copyright 2010 Phylogenetic Likelihood Working Group
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

#ifndef BEAGLE_CPU_SSE_IMPL_HPP
#define BEAGLE_CPU_SSE_IMPL_HPP


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
#include "libhmsbeagle/CPU/BeagleCPUSSEImpl.h"
#include "libhmsbeagle/CPU/SSEDefinitions.h"

namespace beagle {
namespace cpu {

template<typename REALTYPE>
inline const char* getBeagleCPUSSEName(){ return "CPU-SSE-Unknown"; };

template<>
inline const char* getBeagleCPUSSEName<double>(){ return "CPU-SSE-Double"; };

template<>
inline const char* getBeagleCPUSSEName<float>(){ return "CPU-SSE-Single"; };

template<typename REALTYPE>
inline const long getBeagleCPUSSEFlags(){ return BEAGLE_FLAG_COMPUTATION_SYNCH |
                                                       BEAGLE_FLAG_THREADING_NONE |
                                                       BEAGLE_FLAG_PROCESSOR_CPU |
                                                       BEAGLE_FLAG_VECTOR_SSE; };

template<>
inline const long getBeagleCPUSSEFlags<double>(){ return BEAGLE_FLAG_COMPUTATION_SYNCH |
                                                               BEAGLE_FLAG_THREADING_NONE |
                                                               BEAGLE_FLAG_PROCESSOR_CPU |
                                                               BEAGLE_FLAG_PRECISION_DOUBLE |
                                                               BEAGLE_FLAG_VECTOR_SSE; };

template<>
inline const long getBeagleCPUSSEFlags<float>(){ return BEAGLE_FLAG_COMPUTATION_SYNCH |
                                                              BEAGLE_FLAG_THREADING_NONE |
                                                              BEAGLE_FLAG_PROCESSOR_CPU |
                                                              BEAGLE_FLAG_PRECISION_SINGLE |
                                                              BEAGLE_FLAG_VECTOR_SSE; };

template <typename REALTYPE>
BeagleCPUSSEImpl<REALTYPE>::~BeagleCPUSSEImpl() {
}

template <typename REALTYPE>
int BeagleCPUSSEImpl<REALTYPE>::CPUSupportsSSE() {
    //int a,b,c,d;
    //cpuid(0,a,b,c,d);
    //fprintf(stderr,"a = %d\nb = %d\nc = %d\nd = %d\n",a,b,c,d);
    return 1;
}

template <typename REALTYPE>
int BeagleCPUSSEImpl<REALTYPE>::createInstanceExtraFunctionalityHook() {

	if (kStateCount % 2 != 0) {
		kOddStateCount = true;
	} else {
		kOddStateCount = false;
	}
	kHalfStateCount = kStateCount / 2;
	return BEAGLE_SUCCESS;
}

/*
 * Calculates partial likelihoods at a node when both children have states.
 */

template <>
void BeagleCPUSSEImpl<double>::calcStatesStates(double* destP,
                                     const int* states_q,
                                     const double* matrices_q,
                                     const int* states_r,
                                     const double* matrices_r) {

	BeagleCPUImpl<double>::calcStatesStates(destP,
                                     states_q,
                                     matrices_q,
                                     states_r,
                                     matrices_r);
}


//template <>
//void BeagleCPUSSEImpl<double>::calcStatesStates(double* destP,
//                                     const int* states_q,
//                                     const double* matrices_q,
//                                     const int* states_r,
//                                     const double* matrices_r) {
//
//	VecUnion vu_mq[OFFSET][2], vu_mr[OFFSET][2];
//
//    int w = 0;
//	V_Real *destPvec = (V_Real *)destP;
//
//    for (int l = 0; l < kCategoryCount; l++) {
//
//    	SSE_PREFETCH_MATRICES(matrices_q + w, matrices_r + w, vu_mq, vu_mr);
//
//        for (int k = 0; k < kPatternCount; k++) {
//
//            const int state_q = states_q[k];
//            const int state_r = states_r[k];
//
//            *destPvec++ = VEC_MULT(vu_mq[state_q][0].vx, vu_mr[state_r][0].vx);
//            *destPvec++ = VEC_MULT(vu_mq[state_q][1].vx, vu_mr[state_r][1].vx);
//
//        }
//
//        w += OFFSET*4;
//        if (kExtraPatterns)
//        	destPvec += kExtraPatterns * 2;
//    }
//}

/*
 * Calculates partial likelihoods at a node when one child has states and one has partials.
   SSE version
 */
template <>
void BeagleCPUSSEImpl<double>::calcStatesPartials(double* destP,
                                       const int* states_q,
                                       const double* matrices_q,
                                       const double* partials_r,
                                       const double* matrices_r) {
	BeagleCPUImpl<double>::calcStatesPartials(
									   destP,
									   states_q,
									   matrices_q,
									   partials_r,
									   matrices_r);
}

//
//
//template <>
//void BeagleCPUSSEImpl<double>::calcStatesPartials(double* destP,
//                                       const int* states_q,
//                                       const double* matrices_q,
//                                       const double* partials_r,
//                                       const double* matrices_r) {
//
//    int v = 0;
//    int w = 0;
//
// 	VecUnion vu_mq[OFFSET][2], vu_mr[OFFSET][2];
//	V_Real *destPvec = (V_Real *)destP;
//	V_Real destr_01, destr_23;
//
//    for (int l = 0; l < kCategoryCount; l++) {
//
//    	SSE_PREFETCH_MATRICES(matrices_q + w, matrices_r + w, vu_mq, vu_mr);
//
//        for (int k = 0; k < kPatternCount; k++) {
//
//            const int state_q = states_q[k];
//            V_Real vp0, vp1, vp2, vp3;
//            SSE_PREFETCH_PARTIALS(vp,partials_r,v);
//
//			destr_01 = VEC_MULT(vp0, vu_mr[0][0].vx);
//			destr_01 = VEC_MADD(vp1, vu_mr[1][0].vx, destr_01);
//			destr_01 = VEC_MADD(vp2, vu_mr[2][0].vx, destr_01);
//			destr_01 = VEC_MADD(vp3, vu_mr[3][0].vx, destr_01);
//			destr_23 = VEC_MULT(vp0, vu_mr[0][1].vx);
//			destr_23 = VEC_MADD(vp1, vu_mr[1][1].vx, destr_23);
//			destr_23 = VEC_MADD(vp2, vu_mr[2][1].vx, destr_23);
//			destr_23 = VEC_MADD(vp3, vu_mr[3][1].vx, destr_23);
//
//            *destPvec++ = VEC_MULT(vu_mq[state_q][0].vx, destr_01);
//            *destPvec++ = VEC_MULT(vu_mq[state_q][1].vx, destr_23);
//
//            v += 4;
//        }
//        w += OFFSET*4;
//        if (kExtraPatterns) {
//        	destPvec += kExtraPatterns * 2;
//        	v += kExtraPatterns * 4;
//        }
//    }
//}

template <>
void BeagleCPUSSEImpl<double>::calcPartialsPartials(double* destP,
                                              const double* partials1,
                                              const double* matrices1,
                                              const double* partials2,
                                              const double* matrices2) {
	if (kOddStateCount || (kStateCount + PAD) % 2) {
		fprintf(stderr,"Not yet implemented for odd state counts or odd padded state counts!\n");
		exit(-1);
	}

#pragma omp parallel for num_threads(kCategoryCount)
    for (int l = 0; l < kCategoryCount; l++) {
    	int u = l*kStateCount*kPatternCount;
    	int v = l*kStateCount*kPatternCount;
        for (int k = 0; k < kPatternCount; k++) {
            int w = l * kMatrixSize;
            for (int i = 0; i < kStateCount; i++) {

//               	VecUnion vu_m1, vu_m2;
            	register V_Real vu_m1, vu_m2;

               	VecUnion vu_sum1, vu_sum2;
//               	register V_Real vu_sum1, vu_sum2;

#define VU_SUM1	vu_sum1.vx
#define	VU_SUM2	vu_sum2.vx
//#define VU_SUM1	vu_sum1
//#define	VU_SUM2	vu_sum2

            	VU_SUM1 = VEC_SPLAT(0.0);
            	VU_SUM2 = VEC_SPLAT(0.0);

            	double* partials1_vec = (double*)(partials1 + v); // TODO This only works if v is even
            	double* partials2_vec = (double*)(partials2 + v);

//            	double* matrices1_vec = (double*)(matrices1 + w);
//            	double* matrices2_vec = (double*)(matrices2 + w);

            	for (int j = 0; j < kHalfStateCount; j++) {
            		VU_SUM1 = VEC_MADD(
								 VEC_LOAD(matrices1 + w),  // TODO This only works if w is even
								 VEC_LOAD(partials1_vec),
								 VU_SUM1);
            		VU_SUM2 = VEC_MADD(
								 VEC_LOAD(matrices1 + w),
								 VEC_LOAD(partials2_vec),
								 VU_SUM2);
            		partials1_vec += 2;
            		partials2_vec += 2;
            		w += 2;
            	}

//                double sum1 = 0.0, sum2 = 0.0;
//                for (int j = 0; j < kStateCount; j++) {
//                    sum1 += matrices1[w] * partials1[v + j];
//                    sum2 += matrices2[w] * partials2[v + j];
//                    w++;
//                }
#ifdef PAD_MATRICES
                // increment for the extra column at the end
                w += PAD;
#endif

//                double ALIGN16 t1[2], t2[2];
//                _mm_store_pd(t1, VU_SUM1);
//                _mm_store_pd(t2, VU_SUM2);
//                destP[u] = (t1[0] + t1[1]) * (t2[0] + t2[1]);

                destP[u] = (vu_sum1.x[0] + vu_sum1.x[1]) * (vu_sum2.x[0] + vu_sum2.x[1]);
                u++;
            }
            v += kStateCount;
        }
    }
}

//template <>
//void BeagleCPUSSEImpl<double>::calcPartialsPartials(double* destP,
//                                                  const double*  partials_q,
//                                                  const double*  matrices_q,
//                                                  const double*  partials_r,
//                                                  const double*  matrices_r) {
//
//    int v = 0;
//    int w = 0;
//
//    V_Real	destq_01, destq_23, destr_01, destr_23;
// 	VecUnion vu_mq[OFFSET][2], vu_mr[OFFSET][2];
//	V_Real *destPvec = (V_Real *)destP;
//
//    for (int l = 0; l < kCategoryCount; l++) {
//
//		/* Load transition-probability matrices into vectors */
//    	SSE_PREFETCH_MATRICES(matrices_q + w, matrices_r + w, vu_mq, vu_mr);
//
//        for (int k = 0; k < kPatternCount; k++) {
//
//        	V_Real vpq_0, vpq_1, vpq_2, vpq_3;
//        	SSE_PREFETCH_PARTIALS(vpq_,partials_q,v);
//
//        	V_Real vpr_0, vpr_1, vpr_2, vpr_3;
//        	SSE_PREFETCH_PARTIALS(vpr_,partials_r,v);
//
//#			if 1	/* This would probably be faster on PPC/Altivec, which has a fused multiply-add
//			           vector instruction */
//
//			destq_01 = VEC_MULT(vpq_0, vu_mq[0][0].vx);
//			destq_01 = VEC_MADD(vpq_1, vu_mq[1][0].vx, destq_01);
//			destq_01 = VEC_MADD(vpq_2, vu_mq[2][0].vx, destq_01);
//			destq_01 = VEC_MADD(vpq_3, vu_mq[3][0].vx, destq_01);
//			destq_23 = VEC_MULT(vpq_0, vu_mq[0][1].vx);
//			destq_23 = VEC_MADD(vpq_1, vu_mq[1][1].vx, destq_23);
//			destq_23 = VEC_MADD(vpq_2, vu_mq[2][1].vx, destq_23);
//			destq_23 = VEC_MADD(vpq_3, vu_mq[3][1].vx, destq_23);
//
//			destr_01 = VEC_MULT(vpr_0, vu_mr[0][0].vx);
//			destr_01 = VEC_MADD(vpr_1, vu_mr[1][0].vx, destr_01);
//			destr_01 = VEC_MADD(vpr_2, vu_mr[2][0].vx, destr_01);
//			destr_01 = VEC_MADD(vpr_3, vu_mr[3][0].vx, destr_01);
//			destr_23 = VEC_MULT(vpr_0, vu_mr[0][1].vx);
//			destr_23 = VEC_MADD(vpr_1, vu_mr[1][1].vx, destr_23);
//			destr_23 = VEC_MADD(vpr_2, vu_mr[2][1].vx, destr_23);
//			destr_23 = VEC_MADD(vpr_3, vu_mr[3][1].vx, destr_23);
//
//#			else	/* SSE doesn't have a fused multiply-add, so a slight speed gain should be
//                       achieved by decoupling these operations to avoid dependency stalls */
//
//			V_Real a, b, c, d;
//
//			a = VEC_MULT(vpq_0, vu_mq[0][0].vx);
//			b = VEC_MULT(vpq_2, vu_mq[2][0].vx);
//			c = VEC_MULT(vpq_0, vu_mq[0][1].vx);
//			d = VEC_MULT(vpq_2, vu_mq[2][1].vx);
//			a = VEC_MADD(vpq_1, vu_mq[1][0].vx, a);
//			b = VEC_MADD(vpq_3, vu_mq[3][0].vx, b);
//			c = VEC_MADD(vpq_1, vu_mq[1][1].vx, c);
//			d = VEC_MADD(vpq_3, vu_mq[3][1].vx, d);
//			destq_01 = VEC_ADD(a, b);
//			destq_23 = VEC_ADD(c, d);
//
//			a = VEC_MULT(vpr_0, vu_mr[0][0].vx);
//			b = VEC_MULT(vpr_2, vu_mr[2][0].vx);
//			c = VEC_MULT(vpr_0, vu_mr[0][1].vx);
//			d = VEC_MULT(vpr_2, vu_mr[2][1].vx);
//			a = VEC_MADD(vpr_1, vu_mr[1][0].vx, a);
//			b = VEC_MADD(vpr_3, vu_mr[3][0].vx, b);
//			c = VEC_MADD(vpr_1, vu_mr[1][1].vx, c);
//			d = VEC_MADD(vpr_3, vu_mr[3][1].vx, d);
//			destr_01 = VEC_ADD(a, b);
//			destr_23 = VEC_ADD(c, d);
//
//#			endif
//
//#			if 1//
//            destPvec[0] = VEC_MULT(destq_01, destr_01);
//            destPvec[1] = VEC_MULT(destq_23, destr_23);
//            destPvec += 2;
//
//#			else	/* VEC_STORE did demonstrate a measurable performance gain as
//					   it copies all (2/4) values to memory simultaneously;
//					   I can no longer reproduce the performance gain (?) */
//
//			VEC_STORE(destP + v + 0,VEC_MULT(destq_01, destr_01));
//			VEC_STORE(destP + v + 2,VEC_MULT(destq_23, destr_23));
//
//#			endif
//
//            v += 4;
//        }
//        w += OFFSET*4;
//        if (kExtraPatterns) {
//        	destPvec += kExtraPatterns * 2;
//        	v += kExtraPatterns * 4;
//        }
//    }
//}
    
template <>
void BeagleCPUSSEImpl<double>::calcPartialsPartialsAutoScaling(double* destP,
                                                         const double*  partials_q,
                                                         const double*  matrices_q,
                                                         const double*  partials_r,
                                                         const double*  matrices_r,
                                                                  int* activateScaling) {
    BeagleCPUImpl<double>::calcPartialsPartialsAutoScaling(destP,
                                                     partials_q,
                                                     matrices_q,
                                                     partials_r,
                                                     matrices_r,
                                                     activateScaling);
}

//template <>
//void BeagleCPUSSEImpl<double>::calcPartialsPartialsAutoScaling(double* destP,
//                                                                    const double*  partials_q,
//                                                                    const double*  matrices_q,
//                                                                    const double*  partials_r,
//                                                                    const double*  matrices_r,
//                                                                    int* activateScaling) {
//    // TODO: implement calcPartialsPartialsAutoScaling with SSE
//    BeagleCPUImpl<double>::calcPartialsPartialsAutoScaling(destP,
//                                                                partials_q,
//                                                                matrices_q,
//                                                                partials_r,
//                                                                matrices_r,
//                                                                activateScaling);
//}
    
template <>
    int BeagleCPUSSEImpl<double>::calcEdgeLogLikelihoods(const int parIndex,
                                                               const int childIndex,
                                                               const int probIndex,
                                                               const int categoryWeightsIndex,
                                                               const int stateFrequenciesIndex,
                                                               const int scalingFactorsIndex,
                                                               double* outSumLogLikelihood) {
	return BeagleCPUImpl<double>::calcEdgeLogLikelihoods(
	parIndex,
	childIndex,
	probIndex,
	categoryWeightsIndex,
	stateFrequenciesIndex,
	scalingFactorsIndex,
	outSumLogLikelihood);
}

//template <>
//    int BeagleCPUSSEImpl<double>::calcEdgeLogLikelihoods(const int parIndex,
//                                                                const int childIndex,
//                                                                const int probIndex,
//                                                                const int categoryWeightsIndex,
//                                                                const int stateFrequenciesIndex,
//                                                                const int scalingFactorsIndex,
//                                                                double* outSumLogLikelihood) {
//    // TODO: implement derivatives for calculateEdgeLnL
//
//    int returnCode = BEAGLE_SUCCESS;
//
//    assert(parIndex >= kTipCount);
//
//    const double* cl_r = gPartials[parIndex];
//    double* cl_p = integrationTmp;
//    const double* transMatrix = gTransitionMatrices[probIndex];
//    const double* wt = gCategoryWeights[categoryWeightsIndex];
//    const double* freqs = gStateFrequencies[stateFrequenciesIndex];
//
//    memset(cl_p, 0, (kPatternCount * kStateCount)*sizeof(double));
//
//    if (childIndex < kTipCount && gTipStates[childIndex]) { // Integrate against a state at the child
//
//        const int* statesChild = gTipStates[childIndex];
//
//		int w = 0;
//		V_Real *vcl_r = (V_Real *)cl_r;
//		for(int l = 0; l < kCategoryCount; l++) {
//
// 			VecUnion vu_m[OFFSET][2];
// 			SSE_PREFETCH_MATRIX(transMatrix + w, vu_m)
//
//           V_Real *vcl_p = (V_Real *)cl_p;
//
//           for(int k = 0; k < kPatternCount; k++) {
//
//                const int stateChild = statesChild[k];
//				V_Real vwt = VEC_SPLAT(wt[l]);
//
//				V_Real wtdPartials = VEC_MULT(*vcl_r++, vwt);
//                *vcl_p++ = VEC_MADD(vu_m[stateChild][0].vx, wtdPartials, *vcl_p);
//
//				wtdPartials = VEC_MULT(*vcl_r++, vwt);
//                *vcl_p++ = VEC_MADD(vu_m[stateChild][1].vx, wtdPartials, *vcl_p);
//            }
//           w += OFFSET*4;
//           vcl_r += 2 * kExtraPatterns;
//        }
//    } else { // Integrate against a partial at the child
//
//        const double* cl_q = gPartials[childIndex];
//        V_Real * vcl_r = (V_Real *)cl_r;
//        int v = 0;
//        int w = 0;
//
//        for(int l = 0; l < kCategoryCount; l++) {
//
//	        V_Real * vcl_p = (V_Real *)cl_p;
//
// 			VecUnion vu_m[OFFSET][2];
//			SSE_PREFETCH_MATRIX(transMatrix + w, vu_m)
//
//            for(int k = 0; k < kPatternCount; k++) {
//                V_Real vclp_01, vclp_23;
//				V_Real vwt = VEC_SPLAT(wt[l]);
//
//				V_Real vcl_q0, vcl_q1, vcl_q2, vcl_q3;
//				SSE_PREFETCH_PARTIALS(vcl_q,cl_q,v);
//
//				vclp_01 = VEC_MULT(vcl_q0, vu_m[0][0].vx);
//				vclp_01 = VEC_MADD(vcl_q1, vu_m[1][0].vx, vclp_01);
//				vclp_01 = VEC_MADD(vcl_q2, vu_m[2][0].vx, vclp_01);
//				vclp_01 = VEC_MADD(vcl_q3, vu_m[3][0].vx, vclp_01);
//				vclp_23 = VEC_MULT(vcl_q0, vu_m[0][1].vx);
//				vclp_23 = VEC_MADD(vcl_q1, vu_m[1][1].vx, vclp_23);
//				vclp_23 = VEC_MADD(vcl_q2, vu_m[2][1].vx, vclp_23);
//				vclp_23 = VEC_MADD(vcl_q3, vu_m[3][1].vx, vclp_23);
//				vclp_01 = VEC_MULT(vclp_01, vwt);
//				vclp_23 = VEC_MULT(vclp_23, vwt);
//
//				*vcl_p++ = VEC_MADD(vclp_01, *vcl_r++, *vcl_p);
//				*vcl_p++ = VEC_MADD(vclp_23, *vcl_r++, *vcl_p);
//
//                v += 4;
//            }
//            w += 4*OFFSET;
//            if (kExtraPatterns) {
//            	vcl_r += 2 * kExtraPatterns;
//            	v += 4 * kExtraPatterns;
//            }
//
//        }
//    }
//
//    int u = 0;
//    for(int k = 0; k < kPatternCount; k++) {
//        double sumOverI = 0.0;
//        for(int i = 0; i < kStateCount; i++) {
//            sumOverI += freqs[i] * cl_p[u];
//            u++;
//        }
//
//        if (!(sumOverI >= realtypeMin))
//            returnCode = BEAGLE_ERROR_FLOATING_POINT;
//
//        outLogLikelihoodsTmp[k] = log(sumOverI);
//    }
//
//
//    if (scalingFactorsIndex != BEAGLE_OP_NONE) {
//        const double* scalingFactors = gScaleBuffers[scalingFactorsIndex];
//        for(int k=0; k < kPatternCount; k++)
//            outLogLikelihoodsTmp[k] += scalingFactors[k];
//    }
//
//    *outSumLogLikelihood = 0.0;
//    for (int i = 0; i < kPatternCount; i++) {
//        *outSumLogLikelihood += outLogLikelihoodsTmp[i] * gPatternWeights[i];
//    }
//
//    return returnCode;
//}

template <>
int BeagleCPUSSEImpl<double>::getPaddedPatternsModulus() {
	return 1;  // We currently do not vectorize across patterns
}

template <typename REALTYPE>
const char* BeagleCPUSSEImpl<REALTYPE>::getName() {
	return getBeagleCPUSSEName<REALTYPE>();
}

template <typename REALTYPE>
const long BeagleCPUSSEImpl<REALTYPE>::getFlags() {
	return getBeagleCPUSSEFlags<REALTYPE>();
}

///////////////////////////////////////////////////////////////////////////////
// BeagleImplFactory public methods

template <typename REALTYPE>
BeagleImpl* BeagleCPUSSEImplFactory<REALTYPE>::createImpl(int tipCount,
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

    BeagleCPUSSEImpl<REALTYPE>* impl =
    		new BeagleCPUSSEImpl<REALTYPE>();

    if (!impl->CPUSupportsSSE()) {
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

template <typename REALTYPE>
const char* BeagleCPUSSEImplFactory<REALTYPE>::getName() {
	return getBeagleCPUSSEName<REALTYPE>();
}

template <>
const long BeagleCPUSSEImplFactory<double>::getFlags() {
    return BEAGLE_FLAG_COMPUTATION_SYNCH |
           BEAGLE_FLAG_SCALING_MANUAL | BEAGLE_FLAG_SCALING_ALWAYS | BEAGLE_FLAG_SCALING_AUTO |
           BEAGLE_FLAG_THREADING_NONE |
           BEAGLE_FLAG_PROCESSOR_CPU |
           BEAGLE_FLAG_VECTOR_SSE |
           BEAGLE_FLAG_PRECISION_DOUBLE |
           BEAGLE_FLAG_SCALERS_LOG | BEAGLE_FLAG_SCALERS_RAW |
           BEAGLE_FLAG_EIGEN_COMPLEX | BEAGLE_FLAG_EIGEN_REAL;
}

template <>
const long BeagleCPUSSEImplFactory<float>::getFlags() {
    return BEAGLE_FLAG_COMPUTATION_SYNCH |
           BEAGLE_FLAG_SCALING_MANUAL | BEAGLE_FLAG_SCALING_ALWAYS | BEAGLE_FLAG_SCALING_AUTO |
           BEAGLE_FLAG_THREADING_NONE |
           BEAGLE_FLAG_PROCESSOR_CPU |
           BEAGLE_FLAG_VECTOR_SSE |
           BEAGLE_FLAG_PRECISION_SINGLE |
           BEAGLE_FLAG_SCALERS_LOG | BEAGLE_FLAG_SCALERS_RAW |
           BEAGLE_FLAG_EIGEN_COMPLEX | BEAGLE_FLAG_EIGEN_REAL;
}


}
}

#endif //BEAGLE_CPU_SSE_IMPL_HPP
