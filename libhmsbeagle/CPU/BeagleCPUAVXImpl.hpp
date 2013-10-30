/*
 *  BeagleCPUAVXImpl.hpp
 *  BEAGLE
 *
 * Copyright 2013 Phylogenetic Likelihood Working Group
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
 */

#ifndef BEAGLE_CPU_AVX_IMPL_HPP
#define BEAGLE_CPU_AVX_IMPL_HPP


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
#include "libhmsbeagle/CPU/BeagleCPUImpl.h"
#include "libhmsbeagle/CPU/BeagleCPUAVXImpl.h"
#include "libhmsbeagle/CPU/AVXDefinitions.h"

namespace beagle {
namespace cpu {

BEAGLE_CPU_FACTORY_TEMPLATE
inline const char* getBeagleCPUAVXName(){ return "CPU-AVX-Unknown"; };

template<>
inline const char* getBeagleCPUAVXName<double>(){ return "CPU-AVX-Double"; };

template<>
inline const char* getBeagleCPUAVXName<float>(){ return "CPU-AVX-Single"; };

/*
 * Calculates partial likelihoods at a node when both children have states.
 */

BEAGLE_CPU_AVX_TEMPLATE
void BeagleCPUAVXImpl<BEAGLE_CPU_AVX_DOUBLE>::calcStatesStates(double* destP,
                                     const int* states_q,
                                     const double* matrices_q,
                                     const int* states_r,
                                     const double* matrices_r) {

	BeagleCPUImpl<BEAGLE_CPU_AVX_DOUBLE>::calcStatesStates(destP,
                                     states_q,
                                     matrices_q,
                                     states_r,
                                     matrices_r);
}


//template <>
//void BeagleCPUAVXImpl<double>::calcStatesStates(double* destP,
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
//    	AVX_PREFETCH_MATRICES(matrices_q + w, matrices_r + w, vu_mq, vu_mr);
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
   AVX version
 */
BEAGLE_CPU_AVX_TEMPLATE
void BeagleCPUAVXImpl<BEAGLE_CPU_AVX_DOUBLE>::calcStatesPartials(double* destP,
                                       const int* states_q,
                                       const double* matrices_q,
                                       const double* partials_r,
                                       const double* matrices_r) {
	BeagleCPUImpl<BEAGLE_CPU_AVX_DOUBLE>::calcStatesPartials(
									   destP,
									   states_q,
									   matrices_q,
									   partials_r,
									   matrices_r);
}

//
//
//template <>
//void BeagleCPUAVXImpl<double>::calcStatesPartials(double* destP,
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
//    	AVX_PREFETCH_MATRICES(matrices_q + w, matrices_r + w, vu_mq, vu_mr);
//
//        for (int k = 0; k < kPatternCount; k++) {
//
//            const int state_q = states_q[k];
//            V_Real vp0, vp1, vp2, vp3;
//            AVX_PREFETCH_PARTIALS(vp,partials_r,v);
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

//template<>
//void inline BeagleCPUAVXImpl<double>::innerPartialsPartals(
//		const double* __restrict partials1,
//		const double* __restrict matrices1,
//		const double* __restrict partials2,
//		const double* __restrict matrices2,
//		V_Real& sum1_vec,
//		V_Real& sum2_vec,
//		V_Real& out,
//		int& v,
//		int& w) {
//	int j = 0;
//	sum1_vec = VEC_SETZERO();
//	sum2_vec = VEC_SETZERO();
//	for (; j < kStateCountMinusOne; j += 2) {
//		sum1_vec = VEC_MADD(
//				VEC_LOAD(matrices1 + w + j), // TODO This only works if w is even
//				VEC_LOAD(partials1 + v + j), // TODO This only works if v is even
//				sum1_vec);
//		sum2_vec = VEC_MADD(
//				VEC_LOAD(matrices2 + w + j),
//				VEC_LOAD(partials2 + v + j),
//				sum2_vec);
//	}
//
//	out = VEC_MULT(
//			VEC_ADD(sum1_vec, VEC_SWAP(sum1_vec)),
//			VEC_ADD(sum2_vec, VEC_SWAP(sum2_vec))
//	);
//}

//#define DOUBLE_UNROLL // Does not appear to save any time



BEAGLE_CPU_AVX_TEMPLATE
void BeagleCPUAVXImpl<BEAGLE_CPU_AVX_DOUBLE>::calcPartialsPartials(double* __restrict destP,
                                              const double* __restrict partials1,
                                              const double* __restrict matrices1,
                                              const double* __restrict partials2,
                                              const double* __restrict matrices2) {
    int stateCountMinusOne = kPartialsPaddedStateCount - 1;

    struct IO {
    	void operator()(V_Real v) {
    		double x[4];
    		_mm256_storeu_pd(x, v);
    		fprintf(stderr,"%5.3e %5.3e %5.3e %5.3e\n",x[0],x[1],x[2],x[3]);
    	}
    };

    struct math {
    	static inline double horizontal_add (V_Real & a) {
    	    __m256d t1 = _mm256_hadd_pd(a,a);
    	    __m128d t2 = _mm256_extractf128_pd(t1,1);
    	    __m128d t3 = _mm_add_sd(_mm256_castpd256_pd128(t1),t2);
    	    return _mm_cvtsd_f64(t3);
    	}
    };


#pragma omp parallel for num_threads(kCategoryCount)
    for (int l = 0; l < kCategoryCount; l++) {
    	double* destPu = destP + l*kPartialsPaddedStateCount*kPatternCount;
    	int v = l*kPartialsPaddedStateCount*kPatternCount;
        for (int k = 0; k < kPatternCount; k++) {
            int w = l * kMatrixSize;
            for (int i = 0; i < kStateCount; ++i) {
            	register V_Real sum1_vecA = VEC_SETZERO();
            	register V_Real sum2_vecA = VEC_SETZERO();
            	for (int j = 0; j < stateCountMinusOne; j += 4) {
//            		IO()(VEC_LOAD(matrices1 + w + j));
//            		IO()(VEC_LOAD(partials1 + v + j));

            		sum1_vecA = VEC_MADD(
								 VEC_LOAD(matrices1 + w + j),  // TODO This only works if w is even
								 VEC_LOAD(partials1 + v + j),  // TODO This only works if v is even
								 sum1_vecA);
//            		IO()(sum1_vecA);
            		sum2_vecA = VEC_MADD(
								 VEC_LOAD(matrices2 + w + j),
								 VEC_LOAD(partials2 + v + j),
								 sum2_vecA);
//            		fprintf(stderr,"\n");
            	}


//            	sum1_vecA = VEC_MULT(
//            	               VEC_ADD(sum1_vecA, VEC_SWAP(sum1_vecA)),
//            	               VEC_ADD(sum2_vecA, VEC_SWAP(sum2_vecA))
//            	           );
//            	sum1_vecA = VEC_MULT(math::horizontal_add(sum1_vecA), math::horizontal_add(sum2_vecA));
//            	IO()(sum1_vecA);
//            	exit(-1);

                // increment for the extra column at the end
                w += kStateCount + T_PAD;

                // Store single value
//                double x[4];
//                _mm256_storeu_pd(x, VEC_MULT(sum1_vecA,sum2_vecA));
//                *destPu = x[0];
                *destPu = math::horizontal_add(sum1_vecA) * math::horizontal_add(sum2_vecA);

//                *destPu = 1.0; //sum1_vecA[0];
                destPu++;
//                fprintf(stderr,"clear\n");
            }
            destPu += P_PAD;
            v += kPartialsPaddedStateCount;
        }
    }
}

BEAGLE_CPU_AVX_TEMPLATE
void BeagleCPUAVXImpl<BEAGLE_CPU_AVX_DOUBLE>::calcPartialsPartialsFixedScaling(
													double* __restrict destP,
                                              const double* __restrict partials1,
                                              const double* __restrict matrices1,
                                              const double* __restrict partials2,
                                              const double* __restrict matrices2,
                                              const double* __restrict scaleFactors) {

	fprintf(stderr, "Not yet implemented: BeagleCPUAVXImpl::calcPartialsPartialsFixedScaling\n");
	exit(-1);

//    int stateCountMinusOne = kPartialsPaddedStateCount - 1;
//#pragma omp parallel for num_threads(kCategoryCount)
//    for (int l = 0; l < kCategoryCount; l++) {
//    	double* destPu = destP + l*kPartialsPaddedStateCount*kPatternCount;
//    	int v = l*kPartialsPaddedStateCount*kPatternCount;
//        for (int k = 0; k < kPatternCount; k++) {
//            int w = l * kMatrixSize;
//            const V_Real scalar = VEC_SPLAT(scaleFactors[k]);
//            for (int i = 0; i < kStateCount; i++) {
//
//            	register V_Real sum1_vec;
//            	register V_Real sum2_vec;
//
//              	int j = 0;
//            	sum1_vec = VEC_SETZERO();
//            	sum2_vec = VEC_SETZERO();
//            	for ( ; j < stateCountMinusOne; j += 2) {
//            		sum1_vec = VEC_MADD(
//								 VEC_LOAD(matrices1 + w + j),  // TODO This only works if w is even
//								 VEC_LOAD(partials1 + v + j),  // TODO This only works if v is even
//								 sum1_vec);
//            		sum2_vec = VEC_MADD(
//								 VEC_LOAD(matrices2 + w + j),
//								 VEC_LOAD(partials2 + v + j),
//								 sum2_vec);
//            	}
//                		sum1_vec =
//                		VEC_DIV(VEC_MULT(
//                				VEC_ADD(sum1_vec, VEC_SWAP(sum1_vec)),
//                				VEC_ADD(sum2_vec, VEC_SWAP(sum2_vec))
//                		), scalar);
//
//
//                // increment for the extra column at the end
//                w += kStateCount + T_PAD;
//                *destPu = sum1_vec[0];
//    TODO: Could try
//  #define VEC_STORE(dest, source) _mm_store_sd(dest, _mm256_castpd256_pd128(source))
//                destPu++;
//            }
//            destPu += P_PAD;
//            v += kPartialsPaddedStateCount;
//        }
//    }
}

    
BEAGLE_CPU_AVX_TEMPLATE
void BeagleCPUAVXImpl<BEAGLE_CPU_AVX_DOUBLE>::calcPartialsPartialsAutoScaling(double* destP,
                                                         const double*  partials_q,
                                                         const double*  matrices_q,
                                                         const double*  partials_r,
                                                         const double*  matrices_r,
                                                                  int* activateScaling) {
    BeagleCPUImpl<BEAGLE_CPU_AVX_DOUBLE>::calcPartialsPartialsAutoScaling(destP,
                                                     partials_q,
                                                     matrices_q,
                                                     partials_r,
                                                     matrices_r,
                                                     activateScaling);
}

//template <>
//void BeagleCPUAVXImpl<double>::calcPartialsPartialsAutoScaling(double* destP,
//                                                                    const double*  partials_q,
//                                                                    const double*  matrices_q,
//                                                                    const double*  partials_r,
//                                                                    const double*  matrices_r,
//                                                                    int* activateScaling) {
//    // TODO: implement calcPartialsPartialsAutoScaling with AVX
//    BeagleCPUImpl<double>::calcPartialsPartialsAutoScaling(destP,
//                                                                partials_q,
//                                                                matrices_q,
//                                                                partials_r,
//                                                                matrices_r,
//                                                                activateScaling);
//}
    
BEAGLE_CPU_AVX_TEMPLATE
int BeagleCPUAVXImpl<BEAGLE_CPU_AVX_DOUBLE>::calcEdgeLogLikelihoods(const int parIndex,
                                                           const int childIndex,
                                                           const int probIndex,
                                                           const int categoryWeightsIndex,
                                                           const int stateFrequenciesIndex,
                                                           const int scalingFactorsIndex,
                                                           double* outSumLogLikelihood) {
return BeagleCPUImpl<BEAGLE_CPU_AVX_DOUBLE>::calcEdgeLogLikelihoods(
                                                    parIndex,
                                                    childIndex,
                                                    probIndex,
                                                    categoryWeightsIndex,
                                                    stateFrequenciesIndex,
                                                    scalingFactorsIndex,
                                                    outSumLogLikelihood);
}

//template <>
//    int BeagleCPUAVXImpl<double>::calcEdgeLogLikelihoods(const int parIndex,
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
// 			AVX_PREFETCH_MATRIX(transMatrix + w, vu_m)
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
//			AVX_PREFETCH_MATRIX(transMatrix + w, vu_m)
//
//            for(int k = 0; k < kPatternCount; k++) {
//                V_Real vclp_01, vclp_23;
//				V_Real vwt = VEC_SPLAT(wt[l]);
//
//				V_Real vcl_q0, vcl_q1, vcl_q2, vcl_q3;
//				AVX_PREFETCH_PARTIALS(vcl_q,cl_q,v);
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
//    if (!(*outSumLogLikelihood - *outSumLogLikelihood == 0.0))
//        returnCode = BEAGLE_ERROR_FLOATING_POINT;
//
//    return returnCode;
//}

BEAGLE_CPU_AVX_TEMPLATE
int BeagleCPUAVXImpl<BEAGLE_CPU_AVX_DOUBLE>::getPaddedPatternsModulus() {
	return 1;  // We currently do not vectorize across patterns
}
    
BEAGLE_CPU_AVX_TEMPLATE
const char* BeagleCPUAVXImpl<BEAGLE_CPU_AVX_FLOAT>::getName() {
	return  getBeagleCPUAVXName<float>();
}

BEAGLE_CPU_AVX_TEMPLATE
const char* BeagleCPUAVXImpl<BEAGLE_CPU_AVX_DOUBLE>::getName() {
    return  getBeagleCPUAVXName<double>();
}
    
BEAGLE_CPU_AVX_TEMPLATE
const long BeagleCPUAVXImpl<BEAGLE_CPU_AVX_FLOAT>::getFlags() {
	return  BEAGLE_FLAG_COMPUTATION_SYNCH |
            BEAGLE_FLAG_THREADING_NONE |
            BEAGLE_FLAG_PROCESSOR_CPU |
            BEAGLE_FLAG_PRECISION_SINGLE |
            BEAGLE_FLAG_VECTOR_AVX;
}

BEAGLE_CPU_AVX_TEMPLATE
const long BeagleCPUAVXImpl<BEAGLE_CPU_AVX_DOUBLE>::getFlags() {
    return  BEAGLE_FLAG_COMPUTATION_SYNCH |
            BEAGLE_FLAG_THREADING_NONE |
            BEAGLE_FLAG_PROCESSOR_CPU |
            BEAGLE_FLAG_PRECISION_DOUBLE |
            BEAGLE_FLAG_VECTOR_AVX;
}


///////////////////////////////////////////////////////////////////////////////
// BeagleImplFactory public methods

BEAGLE_CPU_FACTORY_TEMPLATE
BeagleImpl* BeagleCPUAVXImplFactory<BEAGLE_CPU_FACTORY_GENERIC>::createImpl(int tipCount,
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

    if (!CPUSupportsAVX())
        return NULL;
    
	if (stateCount & 1) { // is odd
        BeagleCPUAVXImpl<REALTYPE, T_PAD_AVX_ODD, P_PAD_AVX_ODD>* impl =
        new BeagleCPUAVXImpl<REALTYPE, T_PAD_AVX_ODD, P_PAD_AVX_ODD>();
        
        
        try {
            if (impl->createInstance(tipCount, partialsBufferCount, compactBufferCount, stateCount,
                                     patternCount, eigenBufferCount, matrixBufferCount,
                                     categoryCount,scaleBufferCount, resourceNumber, pluginResourceNumber, preferenceFlags, requirementFlags) == 0)
                return impl;
        }
        catch(...) {
            if (DEBUGGING_OUTPUT)
                std::cerr << "exception in initialize\n";
            delete impl;
            throw;
        }
        
        delete impl;        
	} else {
        BeagleCPUAVXImpl<REALTYPE, T_PAD_AVX_EVEN, P_PAD_AVX_EVEN>* impl =
                new BeagleCPUAVXImpl<REALTYPE, T_PAD_AVX_EVEN, P_PAD_AVX_EVEN>();


        try {
            if (impl->createInstance(tipCount, partialsBufferCount, compactBufferCount, stateCount,
                                     patternCount, eigenBufferCount, matrixBufferCount,
                                     categoryCount,scaleBufferCount, resourceNumber,  pluginResourceNumber, preferenceFlags, requirementFlags) == 0)
                return impl;
        }
        catch(...) {
            if (DEBUGGING_OUTPUT)
                std::cerr << "exception in initialize\n";
            delete impl;
            throw;
        }

        delete impl;
    }
    
    return NULL;
}

BEAGLE_CPU_FACTORY_TEMPLATE
const char* BeagleCPUAVXImplFactory<BEAGLE_CPU_FACTORY_GENERIC>::getName() {
	return getBeagleCPUAVXName<BEAGLE_CPU_FACTORY_GENERIC>();
}

template <>
const long BeagleCPUAVXImplFactory<double>::getFlags() {
    return BEAGLE_FLAG_COMPUTATION_SYNCH |
           BEAGLE_FLAG_SCALING_MANUAL | BEAGLE_FLAG_SCALING_ALWAYS | BEAGLE_FLAG_SCALING_AUTO |
           BEAGLE_FLAG_THREADING_NONE |
           BEAGLE_FLAG_PROCESSOR_CPU |
           BEAGLE_FLAG_VECTOR_AVX |
           BEAGLE_FLAG_PRECISION_DOUBLE |
           BEAGLE_FLAG_SCALERS_LOG | BEAGLE_FLAG_SCALERS_RAW |
           BEAGLE_FLAG_EIGEN_COMPLEX | BEAGLE_FLAG_EIGEN_REAL |
           BEAGLE_FLAG_INVEVEC_STANDARD | BEAGLE_FLAG_INVEVEC_TRANSPOSED |
           BEAGLE_FLAG_FRAMEWORK_CPU;           
}

template <>
const long BeagleCPUAVXImplFactory<float>::getFlags() {
    return BEAGLE_FLAG_COMPUTATION_SYNCH |
           BEAGLE_FLAG_SCALING_MANUAL | BEAGLE_FLAG_SCALING_ALWAYS | BEAGLE_FLAG_SCALING_AUTO |
           BEAGLE_FLAG_THREADING_NONE |
           BEAGLE_FLAG_PROCESSOR_CPU |
           BEAGLE_FLAG_VECTOR_AVX |
           BEAGLE_FLAG_PRECISION_SINGLE |
           BEAGLE_FLAG_SCALERS_LOG | BEAGLE_FLAG_SCALERS_RAW |
           BEAGLE_FLAG_EIGEN_COMPLEX | BEAGLE_FLAG_EIGEN_REAL |
           BEAGLE_FLAG_INVEVEC_STANDARD | BEAGLE_FLAG_INVEVEC_TRANSPOSED |
           BEAGLE_FLAG_FRAMEWORK_CPU;           
}

// Code to save:
//	int j = 0;
////              	fprintf(stderr,"v = %d, w = %d\n",v,w);
////            	if (kOddStateCount) { // TODO Template for compiler-time optimization
////            		// Need to either unroll first or last entry
////            		double sum1_end;
////            		double sum2_end;
////            		if (w & 1) { // v and w are odd, unroll first element
////            			fprintf(stderr,"unroll head\n");
////            			sum1_end = matrices1[w] * partials1[v];
////            			sum2_end = matrices2[w] * partials2[v];
////            			j++;
////            		} else { // unroll last element
////            			fprintf(stderr,"unroll tail\n");
////               			sum1_end = matrices1[w + kStateCountMinusOne] *
////               					   partials1[v + kStateCountMinusOne];
////                		sum2_end = matrices2[w + kStateCountMinusOne] *
////               					   partials2[v + kStateCountMinusOne];
////            		}
////            		sum1_vec = VEC_SET1(sum1_end);
////            		sum2_vec = VEC_SET1(sum2_end);
////            	} else {
//	sum1_vec = VEC_SETZERO();
//	sum2_vec = VEC_SETZERO();
////            	}
////            	fprintf(stderr,"Done with head/tail\n");
////            	fprintf(stderr,"starting AVX at %d:%d\n",(v+j),(w+j));
// Next snippet
//#if 1
//                VEC_STORE_SCALAR(destP + u,
//                		VEC_MULT(
//                				VEC_ADD(sum1_vec, VEC_SWAP(sum1_vec)),
//                				VEC_ADD(sum2_vec, VEC_SWAP(sum2_vec))
//                		));
//#else
//                VecUnion t1, t2;
//                t1.vx = sum1_vec;
//                t2.vx = sum2_vec;
//                destP[u] = (t1.x[0] + t1.x[1] //+ endSum1
//                		) * (t2.x[0] + t2.x[1] //+ endSum2
//                				);
//#endif

}
}

#endif //BEAGLE_CPU_AVX_IMPL_HPP
