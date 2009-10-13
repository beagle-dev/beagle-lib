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
#	define VEC_MADD(a, b, c)	_mm_add_pd(_mm_mul_ps((a), (b)), (c))
#else
	typedef float RealType;
	typedef __m128	V_Real;
#	define REALS_PER_VEC	4	/* number of elements per vector */
#	define VEC_MULT(a, b)		_mm_mul_ps((a), (b))
#	define VEC_MADD(a, b, c)	_mm_add_ps(_mm_mul_ps((a), (b)), (c))
#endif
typedef union 			/* for copying individual elements to and from vector floats */
	{
	RealType	x[REALS_PER_VEC];
	V_Real		vx;
	}
	VecUnion;

#ifdef __GNUC__
    #define cpuid(func,ax,bx,cx,dx)\
            __asm__ __volatile__ ("cpuid":\
            "=a" (ax), "=b" (bx), "=c" (cx), "=d" (dx) : "a" (func)); 
#endif

#ifdef _WIN32

#endif

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
    fprintf(stderr,"EXPERIMENTAL calcStatesStates -- SSE!!\n");
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

//#define DEBUG
#if 0//
			
            destP[v    ] = matrices1[w            + state1] * 
                           matrices2[w            + state2];
            destP[v + 1] = matrices1[w + OFFSET*1 + state1] * 
                           matrices2[w + OFFSET*1 + state2];
            destP[v + 2] = matrices1[w + OFFSET*2 + state1] * 
                           matrices2[w + OFFSET*2 + state2];
            destP[v + 3] = matrices1[w + OFFSET*3 + state1] * 
                           matrices2[w + OFFSET*3 + state2];
            
            #ifdef DEBUG
			fprintf(stderr,"First = %5.3e\n",destP[v + 0]);
            fprintf(stderr,"First = %5.3e\n",destP[v + 1]);
            fprintf(stderr,"First = %5.3e\n",destP[v + 2]);
            fprintf(stderr,"First = %5.3e\n",destP[v + 3]);
            #endif
            
              
             v += 4;
           
       
                           
#else
            
            *destPvec++ = VEC_MULT(vu_1[state1][0].vx, vu_2[state2][0].vx);
            *destPvec++ = VEC_MULT(vu_1[state1][1].vx, vu_2[state2][1].vx);
            
           // double* tmp = (double*) destPvec[w];
           //  w += 2;
            
            #ifdef DEBUG
             fprintf(stderr,"Second = %5.3e\n",destP[v + 0]);
             fprintf(stderr,"Second = %5.3e\n",destP[v + 1]);
             fprintf(stderr,"Second = %5.3e\n",destP[v + 2]);
             fprintf(stderr,"Second = %5.3e\n",destP[v + 3]);
             #endif
            
             v += 4;
            
            
            
#endif
              //exit(-1);
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

