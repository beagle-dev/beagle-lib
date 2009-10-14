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
#else
	typedef float RealType;
	typedef __m128	V_Real;
#	define REALS_PER_VEC	4	/* number of elements per vector */
#	define VEC_MULT(a, b)		_mm_mul_ps((a), (b))
#	define VEC_MADD(a, b, c)	_mm_add_ps(_mm_mul_ps((a), (b)), (c))
#	define VEC_SPLAT(a)			_mm_set1_ps(a)
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

#define DEBUG 0

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
                                     const int* states1,
                                     const double* matrices1,
                                     const int* states2,
                                     const double* matrices2) {

	VecUnion vu_1[OFFSET][2], vu_2[OFFSET][2];
	
    int v = 0;
    int w = 0;
	V_Real *destPvec = (V_Real *)destP;
	
#if DEBUG	
    fprintf(stderr,"EXPERIMENTAL calcStatesStates -- SSE!!\n");
#endif    
    for (int l = 0; l < kCategoryCount; l++) {
    
		for (int i = 0; i < OFFSET; i++) {
			vu_1[i][0].x[0] = matrices1[w + 0*OFFSET + i];
			vu_1[i][0].x[1] = matrices1[w + 1*OFFSET + i];
			
			vu_1[i][1].x[0] = matrices1[w + 2*OFFSET + i];
			vu_1[i][1].x[1] = matrices1[w + 3*OFFSET + i];
			
			vu_2[i][0].x[0] = matrices2[w + 0*OFFSET + i];
			vu_2[i][0].x[1] = matrices2[w + 1*OFFSET + i];
			
			vu_2[i][1].x[0] = matrices2[w + 2*OFFSET + i];
			vu_2[i][1].x[1] = matrices2[w + 3*OFFSET + i];
			
		}

		int w = 0;
        for (int k = 0; k < kPatternCount; k++) {

            const int state1 = states1[k];
            const int state2 = states2[k];

#if 0//old
			
            destP[v    ] = matrices1[w            + state1] * 
                           matrices2[w            + state2];
            destP[v + 1] = matrices1[w + OFFSET*1 + state1] * 
                           matrices2[w + OFFSET*1 + state2];
            destP[v + 2] = matrices1[w + OFFSET*2 + state1] * 
                           matrices2[w + OFFSET*2 + state2];
            destP[v + 3] = matrices1[w + OFFSET*3 + state1] * 
                           matrices2[w + OFFSET*3 + state2];
            
            #if DEBUG
			fprintf(stderr,"First = %5.3e\n",destP[v + 0]);
            fprintf(stderr,"First = %5.3e\n",destP[v + 1]);
            fprintf(stderr,"First = %5.3e\n",destP[v + 2]);
            fprintf(stderr,"First = %5.3e\n",destP[v + 3]);
            #endif
                          
             v += 4;
                                             
#else//new
            
            *destPvec++ = VEC_MULT(vu_1[state1][0].vx, vu_2[state2][0].vx);
            *destPvec++ = VEC_MULT(vu_1[state1][1].vx, vu_2[state2][1].vx);
            
           // double* tmp = (double*) destPvec[w];
           //  w += 2;
            
             #if DEBUG
             fprintf(stderr,"Second = %5.3e\n",destP[v + 0]);
             fprintf(stderr,"Second = %5.3e\n",destP[v + 1]);
             fprintf(stderr,"Second = %5.3e\n",destP[v + 2]);
             fprintf(stderr,"Second = %5.3e\n",destP[v + 3]);
             #endif
            
             v += 4;            
#endif
        }
        
        w += OFFSET*4;
    }
}

/*
 * Calculates partial likelihoods at a node when one child has states and one has partials.
   SSE version
 */
void BeagleCPU4StateSSEImpl::calcStatesPartials(double* destP,
                                       const int* states1,
                                       const double* matrices1,
                                       const double* partials2,
                                       const double* matrices2) {

    int u = 0;
    int v = 0;
    int w = 0;
    
    
#define NEWWAY 1
#define OLDWAY 0


#if NEWWAY
 	VecUnion vu_m1[OFFSET][2], vu_m2[OFFSET][2];
	V_Real *destPvec = (V_Real *)destP;
	V_Real dest01, dest23;
#endif	
   
#if DEBUG   
fprintf(stderr, "+++++++++++++++++++++++++++++++++++++++++in calcStatesPartials\n");//
#endif

    for (int l = 0; l < kCategoryCount; l++) {
                
#if NEWWAY//new
		for (int i = 0; i < OFFSET; i++) {
			vu_m1[i][0].x[0] = matrices1[w + 0*OFFSET + i];
			vu_m1[i][0].x[1] = matrices1[w + 1*OFFSET + i];
			vu_m1[i][1].x[0] = matrices1[w + 2*OFFSET + i];
			vu_m1[i][1].x[1] = matrices1[w + 3*OFFSET + i];
			
			vu_m2[i][0].x[0] = matrices2[w + 0*OFFSET + i];
			vu_m2[i][0].x[1] = matrices2[w + 1*OFFSET + i];
			vu_m2[i][1].x[0] = matrices2[w + 2*OFFSET + i];
			vu_m2[i][1].x[1] = matrices2[w + 3*OFFSET + i];
		}
#endif
#if OLDWAY//
        PREFETCH_MATRIX(2,matrices2,w);
#endif
        for (int k = 0; k < kPatternCount; k++) {
            
            const int state1 = states1[k];
 
#if NEWWAY//new
//			V_Real *pv01 = (V_Real *)(partials2 + v);
//			V_Real *pv23 = pv01 + 1;
//			
// 			VecUnion pv01u, pv23u;
// 			pv01u.vx = *pv01;
// 			pv23u.vx = *pv23;
// 			#if DEBUG
// 			fprintf(stderr, "pv01=%g %g   pv02=%g %g\n", pv01u.x[0], pv01u.x[1], pv23u.x[0], pv23u.x[1]);
// 			#endif
// 			
			
#endif
#if OLDWAY//orig
            PREFETCH_PARTIALS(2,partials2,v);
            #if DEBUG
			fprintf(stderr, "p2=%g %g %g %g\n", p20, p21, p22, p23);//
			#endif
#endif

#if NEWWAY//new
//			V_Real v01 = *pv01;
//			V_Real v23 = *pv23;
			V_Real vp0 = VEC_SPLAT(partials2[v + 0]);
			V_Real vp1 = VEC_SPLAT(partials2[v + 1]);
			V_Real vp2 = VEC_SPLAT(partials2[v + 2]);
			V_Real vp3 = VEC_SPLAT(partials2[v + 3]);
			dest01 = VEC_MULT(vp0, vu_m2[0][0].vx);
			dest01 = VEC_MADD(vp1, vu_m2[1][0].vx, dest01);
			dest01 = VEC_MADD(vp2, vu_m2[2][0].vx, dest01);
			dest01 = VEC_MADD(vp3, vu_m2[3][0].vx, dest01);
			dest23 = VEC_MULT(vp0, vu_m2[0][1].vx);
			dest23 = VEC_MADD(vp1, vu_m2[1][1].vx, dest23);
			dest23 = VEC_MADD(vp2, vu_m2[2][1].vx, dest23);
			dest23 = VEC_MADD(vp3, vu_m2[3][1].vx, dest23);
			#if DEBUG
			VecUnion temp01, temp23;
			temp01.vx = dest01;
			fprintf(stderr, "dest01 = %g %g\n", temp01.x[0], temp01.x[1]);//
			fprintf(stderr, "dest23 = %g %g\n", temp23.x[0], temp23.x[1]);//
			#endif
            *destPvec++ = VEC_MULT(vu_m1[state1][0].vx, dest01);
            *destPvec++ = VEC_MULT(vu_m1[state1][1].vx, dest23);
#endif
#if OLDWAY//orig
            DO_INTEGRATION(2); // defines sum20, sum21, sum22, sum23;
            destP[u    ] = matrices1[w            + state1] * sum20;
            destP[u + 1] = matrices1[w + OFFSET*1 + state1] * sum21;
            destP[u + 2] = matrices1[w + OFFSET*2 + state1] * sum22;
            destP[u + 3] = matrices1[w + OFFSET*3 + state1] * sum23;
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
                                                  const double* partials1,
                                                  const double* matrices1,
                                                  const double* partials2,
                                                  const double* matrices2) {

    BeagleCPU4StateImpl::calcPartialsPartials(destP,
                                              partials1,matrices1,
                                              partials2,matrices2);
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

