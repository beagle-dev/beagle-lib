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
 */

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

#define OFFSET 5
#define DLS_USE_SSE2
#define DLS_MACOS		/* this should be done at the configure stage */

#if defined(DLS_USE_SSE2)
#	if !defined(DLS_MACOS)
#		include <emmintrin.h>
#	endif
#	include <xmmintrin.h>
#endif
typedef double VecEl_t;

#define USE_DOUBLE_PREC
#if defined(USE_DOUBLE_PREC)
	typedef double RealType;
	typedef __m128d	V_Real;
#	define REALS_PER_VEC	2	/* number of elements per vector */
#	define VEC_MULT(a, b)		_mm_mul_pd((a), (b))
#	define VEC_MADD(a, b, c)	_mm_add_pd(_mm_mul_pd((a), (b)), (c))
#	define VEC_SPLAT(a)			_mm_set1_pd(a)
#	define VEC_ADD(a, b)		_mm_add_pd(a, b)
#else
	typedef float RealType;
	typedef __m128	V_Real;
#	define REALS_PER_VEC	4	/* number of elements per vector */
#	define VEC_MULT(a, b)		_mm_mul_ps((a), (b))
#	define VEC_MADD(a, b, c)	_mm_add_ps(_mm_mul_ps((a), (b)), (c))
#	define VEC_SPLAT(a)			_mm_set1_ps(a)
#	define VEC_ADD(a, b)		_mm_add_ps(a, b)
#endif
typedef union 			/* for copying individual elements to and from vector floats */
	{
	RealType	x[REALS_PER_VEC];
	V_Real		vx;
	}
	VecUnion;

#define PREFETCH_MATRIX(num,matrices,w) \
    double m##num##00, m##num##01, m##num##02, m##num##03, \
           m##num##10, m##num##11, m##num##12, m##num##13, \
           m##num##20, m##num##21, m##num##22, m##num##23, \
           m##num##30, m##num##31, m##num##32, m##num##33; \
    m##num##00 = matrices[w + OFFSET*0 + 0]; \
    m##num##01 = matrices[w + OFFSET*0 + 1]; \
    m##num##02 = matrices[w + OFFSET*0 + 2]; \
    m##num##03 = matrices[w + OFFSET*0 + 3]; \
    m##num##10 = matrices[w + OFFSET*1 + 0]; \
    m##num##11 = matrices[w + OFFSET*1 + 1]; \
    m##num##12 = matrices[w + OFFSET*1 + 2]; \
    m##num##13 = matrices[w + OFFSET*1 + 3]; \
    m##num##20 = matrices[w + OFFSET*2 + 0]; \
    m##num##21 = matrices[w + OFFSET*2 + 1]; \
    m##num##22 = matrices[w + OFFSET*2 + 2]; \
    m##num##23 = matrices[w + OFFSET*2 + 3]; \
    m##num##30 = matrices[w + OFFSET*3 + 0]; \
    m##num##31 = matrices[w + OFFSET*3 + 1]; \
    m##num##32 = matrices[w + OFFSET*3 + 2]; \
    m##num##33 = matrices[w + OFFSET*3 + 3];

#define PREFETCH_PARTIALS(num,partials,v) \
    double p##num##0, p##num##1, p##num##2, p##num##3; \
    p##num##0 = partials[v + 0]; \
    p##num##1 = partials[v + 1]; \
    p##num##2 = partials[v + 2]; \
    p##num##3 = partials[v + 3];

#if 0//
#define PREFETCH_PARTIALS_VEC(num,partials,v, vu) \
    vu[0].x[0] = partials[v + 0]; \
    vu[0].x[1] = partials[v + 1]; \
    vu[1].x[0] = partials[v + 2]; \
    vu[1].x[1] = partials[v + 3];
#endif//

//#define DO_INTEGRATION(num) \
//    double sum##num##0, sum##num##1, sum##num##2, sum##num##3; \
//    sum##num##0  = m##num##00 * p##num##0; \
//    sum##num##1  = m##num##10 * p##num##0; \
//    sum##num##2  = m##num##20 * p##num##0; \
//    sum##num##3  = m##num##30 * p##num##0; \
// \
//    sum##num##0 += m##num##01 * p##num##1; \
//    sum##num##1 += m##num##11 * p##num##1; \
//    sum##num##2 += m##num##21 * p##num##1; \
//    sum##num##3 += m##num##31 * p##num##1; \
// \
//    sum##num##0 += m##num##02 * p##num##2; \
//    sum##num##1 += m##num##12 * p##num##2; \
//    sum##num##2 += m##num##22 * p##num##2; \
//    sum##num##3 += m##num##32 * p##num##2; \
// \
//    sum##num##0 += m##num##03 * p##num##3; \
//    sum##num##1 += m##num##13 * p##num##3; \
//    sum##num##2 += m##num##23 * p##num##3; \
//    sum##num##3 += m##num##33 * p##num##3;

#define DO_INTEGRATION(num) \
    double sum##num##0, sum##num##1, sum##num##2, sum##num##3; \
    sum##num##0  = m##num##00 * p##num##0 + \
                   m##num##01 * p##num##1 + \
                   m##num##02 * p##num##2 + \
                   m##num##03 * p##num##3;  \
 \
    sum##num##1  = m##num##10 * p##num##0 + \
                   m##num##11 * p##num##1 + \
                   m##num##12 * p##num##2 + \
                   m##num##13 * p##num##3;  \
 \
    sum##num##2  = m##num##20 * p##num##0 + \
                   m##num##21 * p##num##1 + \
                   m##num##22 * p##num##2 + \
                   m##num##23 * p##num##3;  \
\
    sum##num##3  = m##num##30 * p##num##0 + \
                   m##num##31 * p##num##1 + \
                   m##num##32 * p##num##2 + \
                   m##num##33 * p##num##3;

#ifdef __GNUC__
    #define cpuid(func,ax,bx,cx,dx)\
            __asm__ __volatile__ ("cpuid":\
            "=a" (ax), "=b" (bx), "=c" (cx), "=d" (dx) : "a" (func)); 
#endif

#ifdef _WIN32

#endif

//#define DEBUG 1
#define DEBUG 0


#define NEWWAY 1
#define OLDWAY 0

using namespace beagle;
using namespace beagle::cpu;

#if defined (BEAGLE_IMPL_DEBUGGING_OUTPUT) && BEAGLE_IMPL_DEBUGGING_OUTPUT
const bool DEBUGGING_OUTPUT = true;
#else
const bool DEBUGGING_OUTPUT = false;
#endif

BeagleCPU4StateSSEImpl::~BeagleCPU4StateSSEImpl() {
}

int BeagleCPU4StateSSEImpl::CPUSupportsSSE() {
    //int a,b,c,d;
    //cpuid(0,a,b,c,d);
    //fprintf(stderr,"a = %d\nb = %d\nc = %d\nd = %d\n",a,b,c,d);
    return 1;
}

/*
 * Calculates partial likelihoods at a node when both children have states.
 */
void BeagleCPU4StateSSEImpl::calcStatesStates(double* destP,
                                     const int* states_q,
                                     const double* matrices_q,
                                     const int* states_r,
                                     const double* matrices_r) {

	VecUnion vu_1[OFFSET][2], vu_2[OFFSET][2];
	
    int v = 0;
    int w = 0;
	V_Real *destPvec = (V_Real *)destP;
	
#if DEBUG	
    fprintf(stderr,"EXPERIMENTAL calcStatesStates -- SSE!!\n");
#endif    
    for (int l = 0; l < kCategoryCount; l++) {
    
		for (int i = 0; i < OFFSET; i++) {
			vu_1[i][0].x[0] = matrices_q[w + 0*OFFSET + i];
			vu_1[i][0].x[1] = matrices_q[w + 1*OFFSET + i];
			
			vu_1[i][1].x[0] = matrices_q[w + 2*OFFSET + i];
			vu_1[i][1].x[1] = matrices_q[w + 3*OFFSET + i];
			
			vu_2[i][0].x[0] = matrices_r[w + 0*OFFSET + i];
			vu_2[i][0].x[1] = matrices_r[w + 1*OFFSET + i];
			
			vu_2[i][1].x[0] = matrices_r[w + 2*OFFSET + i];
			vu_2[i][1].x[1] = matrices_r[w + 3*OFFSET + i];
			
		}

		int w = 0;
        for (int k = 0; k < kPatternCount; k++) {

            const int state_q = states_q[k];
            const int state_r = states_r[k];

            *destPvec++ = VEC_MULT(vu_1[state_q][0].vx, vu_2[state_r][0].vx);
            *destPvec++ = VEC_MULT(vu_1[state_q][1].vx, vu_2[state_r][1].vx);
            
             #if DEBUG
             fprintf(stderr,"Second = %5.3e\n",destP[v + 0]);
             fprintf(stderr,"Second = %5.3e\n",destP[v + 1]);
             fprintf(stderr,"Second = %5.3e\n",destP[v + 2]);
             fprintf(stderr,"Second = %5.3e\n",destP[v + 3]);
             #endif
            
             v += 4;            
        }
        
        w += OFFSET*4;
    }
}

/*
 * Calculates partial likelihoods at a node when one child has states and one has partials.
   SSE version
 */
void BeagleCPU4StateSSEImpl::calcStatesPartials(double* destP,
                                       const int* states_q,
                                       const double* matrices_q,
                                       const double* partials_r,
                                       const double* matrices_r) {

    int u = 0;
    int v = 0;
    int w = 0;
    
#if NEWWAY
 	VecUnion vu_mq[OFFSET][2], vu_mr[OFFSET][2];
	V_Real *destPvec = (V_Real *)destP;
	V_Real destr_01, destr_23;
#endif	
   
#if DEBUG || 1//  
fprintf(stderr, "+++++++++++++++++++++++++++++++++++++++++in calcStatesPartials\n");//
#endif

    for (int l = 0; l < kCategoryCount; l++) {
                
#if NEWWAY//new
		for (int i = 0; i < OFFSET; i++) {
			vu_mq[i][0].x[0] = matrices_q[w + 0*OFFSET + i];
			vu_mq[i][0].x[1] = matrices_q[w + 1*OFFSET + i];
			vu_mq[i][1].x[0] = matrices_q[w + 2*OFFSET + i];
			vu_mq[i][1].x[1] = matrices_q[w + 3*OFFSET + i];
			
			vu_mr[i][0].x[0] = matrices_r[w + 0*OFFSET + i];
			vu_mr[i][0].x[1] = matrices_r[w + 1*OFFSET + i];
			vu_mr[i][1].x[0] = matrices_r[w + 2*OFFSET + i];
			vu_mr[i][1].x[1] = matrices_r[w + 3*OFFSET + i];
		}
#endif
#if OLDWAY//
        PREFETCH_MATRIX(2,matrices_r,w);
#endif
        for (int k = 0; k < kPatternCount; k++) {
            
            const int state_q = states_q[k];
 
#if OLDWAY//orig
            PREFETCH_PARTIALS(2,partials_r,v);
            #if DEBUG
			fprintf(stderr, "p2=%g %g %g %g\n", p20, p21, p22, p23);//
			#endif
#endif

#if NEWWAY//new
			V_Real vp0 = VEC_SPLAT(partials_r[v + 0]);
			V_Real vp1 = VEC_SPLAT(partials_r[v + 1]);
			V_Real vp2 = VEC_SPLAT(partials_r[v + 2]);
			V_Real vp3 = VEC_SPLAT(partials_r[v + 3]);
			#if 1//
			destr_01 = VEC_MULT(vp0, vu_mr[0][0].vx);
			destr_01 = VEC_MADD(vp1, vu_mr[1][0].vx, destr_01);
			destr_01 = VEC_MADD(vp2, vu_mr[2][0].vx, destr_01);
			destr_01 = VEC_MADD(vp3, vu_mr[3][0].vx, destr_01);
			destr_23 = VEC_MULT(vp0, vu_mr[0][1].vx);
			destr_23 = VEC_MADD(vp1, vu_mr[1][1].vx, destr_23);
			destr_23 = VEC_MADD(vp2, vu_mr[2][1].vx, destr_23);
			destr_23 = VEC_MADD(vp3, vu_mr[3][1].vx, destr_23);
			#endif//
			#if DEBUG
			VecUnion temp01, temp23;
			temp01.vx = destr_01;
			fprintf(stderr, "destr_01 = %g %g\n", temp01.x[0], temp01.x[1]);//
			fprintf(stderr, "destr_23 = %g %g\n", temp23.x[0], temp23.x[1]);//
			#endif
            *destPvec++ = VEC_MULT(vu_mq[state_q][0].vx, destr_01);
            *destPvec++ = VEC_MULT(vu_mq[state_q][1].vx, destr_23);
#endif
#if OLDWAY//orig
            DO_INTEGRATION(2); // defines sum20, sum21, sum22, sum23;
            destP[u    ] = matrices_q[w            + state_q] * sum20;
            destP[u + 1] = matrices_q[w + OFFSET*1 + state_q] * sum21;
            destP[u + 2] = matrices_q[w + OFFSET*2 + state_q] * sum22;
            destP[u + 3] = matrices_q[w + OFFSET*3 + state_q] * sum23;
            #if DEBUG
            fprintf(stderr, "%g %g %g %g\n", sum20, sum21, sum22, sum23);
            #endif
#endif                        
            
            v += 4;
            u += 4;
        }
        w += OFFSET*4;
    }
}

void BeagleCPU4StateSSEImpl::calcPartialsPartials(double* destP,
                                                  const double* partials_q,
                                                  const double* matrices_q,
                                                  const double* partials_r,
                                                  const double* matrices_r) {

    int v = 0;
    int w = 0;
    
    V_Real	destq_01, destq_23, destr_01, destr_23;
 	VecUnion vu_mq[OFFSET][2], vu_mr[OFFSET][2];
	V_Real *destPvec = (V_Real *)destP;
   
    for (int l = 0; l < kCategoryCount; l++) {

		/* Load transition-probability matrices into vectors */
		const double *mq = matrices_q + w;
		const double *mr = matrices_r + w;
		for (int i = 0; i < OFFSET; i++, mq++, mr++) {
			vu_mq[i][0].x[0] = mq[0*OFFSET];
			vu_mq[i][0].x[1] = mq[1*OFFSET];
			vu_mr[i][0].x[0] = mr[0*OFFSET];
			vu_mr[i][0].x[1] = mr[1*OFFSET];
			vu_mq[i][1].x[0] = mq[2*OFFSET];
			vu_mq[i][1].x[1] = mq[3*OFFSET];			
			vu_mr[i][1].x[0] = mr[2*OFFSET];
			vu_mr[i][1].x[1] = mr[3*OFFSET];
		}
		
        for (int k = 0; k < kPatternCount; k++) {
			V_Real vpq_0 = VEC_SPLAT(partials_q[v + 0]);
			V_Real vpq_1 = VEC_SPLAT(partials_q[v + 1]);
			V_Real vpq_2 = VEC_SPLAT(partials_q[v + 2]);
			V_Real vpq_3 = VEC_SPLAT(partials_q[v + 3]);

			V_Real vpr_0 = VEC_SPLAT(partials_r[v + 0]);
			V_Real vpr_1 = VEC_SPLAT(partials_r[v + 1]);
			V_Real vpr_2 = VEC_SPLAT(partials_r[v + 2]);
			V_Real vpr_3 = VEC_SPLAT(partials_r[v + 3]);

#			if 0	/* This would probably be faster on PPC/Altivec, which has a fused multiply-add
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

            destPvec[0] = VEC_MULT(destq_01, destr_01);
            destPvec[1] = VEC_MULT(destq_23, destr_23);
            destPvec += 2;
			
#			else	/* SSE doesn't have a fused multiply-add, so a slight speed gain can possibly
					   be achieved by decoupling these operations to avoid dependency stalls */

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

            destPvec[0] = VEC_MULT(destq_01, destr_01);
            destPvec[1] = VEC_MULT(destq_23, destr_23);
            destPvec += 2;
			
#			endif

            v += 4;
        }
        w += OFFSET*4;
    }
}


void BeagleCPU4StateSSEImpl::calcEdgeLogLikelihoods(const int parIndex,
                                           const int childIndex,
                                           const int probIndex,
                                           const int firstDerivativeIndex,
                                           const int secondDerivativeIndex,
                                           const double* inWeights,
                                           const double* inStateFrequencies,
                                           const int scalingFactorsIndex,
                                           double* outLogLikelihoods,
                                           double* outFirstDerivatives,
                                           double* outSecondDerivatives) {
    // TODO: implement derivatives for calculateEdgeLnL
    // TODO: implement rate categories for calculateEdgeLnL

    assert(parIndex >= kTipCount);

    const double* partialsParent = gPartials[parIndex];
    const double* transMatrix = gTransitionMatrices[probIndex];
    const double* wt = inWeights;

    memset(integrationTmp, 0, (kPatternCount * kStateCount)*sizeof(double));
    
    if (childIndex < kTipCount && gTipStates[childIndex]) { // Integrate against a state at the child
        
        const int* statesChild = gTipStates[childIndex];    
        int v = 0; // Index for parent partials
        
fprintf(stderr, "+++++++++++++++++ A: in BeagleCPU4StateSSEImpl::calcEdgeLogLikelihoods with kCategoryCount=%d\n", kCategoryCount);//        
        for(int l = 0; l < kCategoryCount; l++) {
            int u = 0; // Index in resulting product-partials (summed over categories)
            const double weight = wt[l];
            for(int k = 0; k < kPatternCount; k++) {
                
                const int stateChild = statesChild[k];  // DISCUSSION PT: Does it make sense to change the order of the partials,
                                                        // so we can interchange the patterCount and categoryCount loop order?
                int w =  l * kMatrixSize;
                for(int i = 0; i < kStateCount; i++) {
                    integrationTmp[u] += transMatrix[w + stateChild] * partialsParent[v + i] * weight;
                    u++;
#ifdef PAD_MATRICES
                    w += (kStateCount + 1);
#else
                    w += kStateCount;
#endif
                }
                v += kStateCount;
            }
        }
        
    } else { // Integrate against a partial at the child
        
fprintf(stderr, "+++++++++++++++++ B: in BeagleCPU4StateSSEImpl::calcEdgeLogLikelihoods with kCategoryCount=%d\n", kCategoryCount);//        
        const double* partialsChild = gPartials[childIndex];
        int v = 0;
        
        for(int l = 0; l < kCategoryCount; l++) {
            int u = 0;
            const double weight = wt[l];
            for(int k = 0; k < kPatternCount; k++) {                
                int w = l * kMatrixSize;                 
                for(int i = 0; i < kStateCount; i++) {
                    double sumOverJ = 0.0;
                    for(int j = 0; j < kStateCount; j++) {
                        sumOverJ += transMatrix[w] * partialsChild[v + j];
                        w++;
                    }
#ifdef PAD_MATRICES
                    // increment for the extra column at the end
                    w++;
#endif
                    integrationTmp[u] += sumOverJ * partialsParent[v + i] * weight;
                    u++;
                }
                v += kStateCount;
            }
        }
    }
        
    int u = 0;
    for(int k = 0; k < kPatternCount; k++) {
        double sumOverI = 0.0;
        for(int i = 0; i < kStateCount; i++) {
            sumOverI += inStateFrequencies[i] * integrationTmp[u];
            u++;
        }
        outLogLikelihoods[k] = log(sumOverI);
    }        

    
    if (scalingFactorsIndex != BEAGLE_OP_NONE) {
        const double* scalingFactors = gScaleBuffers[scalingFactorsIndex];
        for(int k=0; k < kPatternCount; k++)
            outLogLikelihoods[k] += scalingFactors[k];
    }
}


///////////////////////////////////////////////////////////////////////////////
// BeagleImplFactory public methods

BeagleImpl* BeagleCPU4StateSSEImplFactory::createImpl(int tipCount,
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
    
    fprintf(stderr,"EXPERIMENTAL FACTORY -- SSE!!\n");
    
    BeagleCPU4StateSSEImpl* impl = new BeagleCPU4StateSSEImpl();
    
    if (impl->CPUSupportsSSE()) {        
        fprintf(stderr,"CPU supports SSE!\n");
    } else {
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

const char* BeagleCPU4StateSSEImplFactory::getName() {
    return "CPU-4StateSSE";
}

const long BeagleCPU4StateSSEImplFactory::getFlags() {
    return BEAGLE_FLAG_ASYNCH | BEAGLE_FLAG_CPU | BEAGLE_FLAG_DOUBLE | BEAGLE_FLAG_SSE;
}

