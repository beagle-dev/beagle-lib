/*
 *  BeagleCPU4StateSSEImpl.h
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

#ifndef __SSEDefinitions__
#define __SSEDefinitions__

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#define DLS_USE_SSE2

#if defined(DLS_USE_SSE2)
#	if defined(__ARM64_ARCH_8__)
#		include "libhmsbeagle/CPU/sse2neon.h"
#       define _MM_SHUFFLE2(fp1,fp0) (((fp1) << 1) | (fp0))
#		define VEC_SHUFFLE0(a,b)	_mm_shuffle_pd(a, b, _MM_SHUFFLE2(0,0)) // vreinterpretq_f64_m128d(a)
#		define VEC_SHUFFLE1(a,b)	_mm_shuffle_pd(a, b, _MM_SHUFFLE2(1,1)) // vreinterpretq_f64_m128d(a)
#       if __has_builtin(__builtin_shufflevector)
#       	define _mm_shuffle_pd(a,b,imm)                                \
				__extension__({                                           \
		        float64x2_t _input1 = vreinterpretq_f64_m128(a);          \
	    	    float64x2_t _input2 = vreinterpretq_f64_m128(b);          \
		        float64x2_t _shuf = __builtin_shufflevector(              \
	    	        _input1, _input2, (imm) & (0x1), ((imm) >> 1) & 0x1); \
	        	vreinterpretq_m128_f32(_shuf);                            \
    	    })
#		else
#			error "Need to implement NEON translation of _mm_shuffle_pd"
#		endif

		static inline __m128 _mm_div_pd(__m128 a, __m128 b) {
		    return vreinterpretq_m128_f64(
        		vdivq_f64(vreinterpretq_f64_m128(a), vreinterpretq_f64_m128(b)));
		}

		static inline void _mm_store_sd(double* a, __m128 b) {
			const auto _b = vreinterpretq_f64_m128(b);
			a[0] = _b[0];
		}
#   else
#		if !defined(DLS_MACOS)
#			include <emmintrin.h>
#		endif
#       include <pmmintrin.h>
#		include <xmmintrin.h>
#   endif
#endif
typedef double VecEl_t;

#ifdef __GNUC__
#define ALIGN16 __attribute__((aligned(16)))
#else
#define ALIGN16 __declspec(align(16))
#endif

#define USE_DOUBLE_PREC
#if defined(USE_DOUBLE_PREC)
	typedef double RealType;
	typedef __m128d	V_Real;
#	define REALS_PER_VEC	2	/* number of elements per vector */
#	define VEC_LOAD(a)			_mm_load_pd(a)
#	define VEC_LOAD_SCALAR(a)	_mm_load1_pd(a)
#	define VEC_STORE(a, b)		_mm_store_pd((a), (b))
#   define VEC_STORE_SCALAR(a, b) _mm_store_sd((a), (b))
#   define VEC_STOREU(a, b)     _mm_storeu_pd((a), (b))
#	define VEC_MULT(a, b)		_mm_mul_pd((a), (b))
#	define VEC_DIV(a, b)		_mm_div_pd((a), (b))
#	define VEC_MADD(a, b, c)	_mm_add_pd(_mm_mul_pd((a), (b)), (c))
#	define VEC_SPLAT(a)			_mm_set1_pd(a)
#	define VEC_ADD(a, b)		_mm_add_pd(a, b)
#   define VEC_SWAP(a)			_mm_shuffle_pd(a, a, _MM_SHUFFLE2(0,1))
# 	define VEC_SETZERO()		_mm_setzero_pd()
#	define VEC_SET1(a)			_mm_set_sd((a))
#	define VEC_SET(a, b)		_mm_set_pd((a), (b))
#   define VEC_MOVE(a, b)		_mm_move_sd((a), (b))
#	define VEC_SHUFFLE0(a, b)	_mm_shuffle_pd(a, b, _MM_SHUFFLE2(0,0))
#	define VEC_SHUFFLE1(a, b)	_mm_shuffle_pd(a, b, _MM_SHUFFLE2(1,1))
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

#ifdef __GNUC__
    #define cpuid(func,ax,bx,cx,dx)\
            __asm__ __volatile__ ("cpuid":\
            "=a" (ax), "=b" (bx), "=c" (cx), "=d" (dx) : "a" (func));
#endif

#ifdef _WIN32

#endif

int CPUSupportsSSE() {
    //int a,b,c,d;
    //cpuid(0,a,b,c,d);
    //fprintf(stderr,"a = %d\nb = %d\nc = %d\nd = %d\n",a,b,c,d);
    return 1;
}

#endif // __SSEDefinitions__
