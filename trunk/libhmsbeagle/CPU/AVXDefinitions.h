/*
 *  AVXDefinitions.h
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

#ifndef __AVXDefinitions__
#define __AVXDefinitions__

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#define USE_AVX

#if defined(USE_AVX)
#	include <immintrin.h>
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
	typedef __m256d	V_Real;
#	define REALS_PER_VEC	4	/* number of elements per vector */
#	define VEC_LOAD(a)			_mm256_load_pd(a)
//#	define VEC_LOAD_SCALAR(a)	_mm_load1_pd(a)
#	define VEC_STORE(a, b)		_mm256_store_pd((a), (b))
//#   define VEC_STORE _SCALAR(a, b) _mm_store_sd((a), (b))
#	define VEC_MULT(a, b)		_mm256_mul_pd((a), (b))
#	define VEC_DIV(a, b)		_mm256_div_pd((a), (b))
#	define VEC_MADD(a, b, c)	_mm256_add_pd(_mm256_mul_pd((a), (b)), (c))
#	define VEC_SPLAT(a)			_mm256_set1_pd(a)
#	define VEC_ADD(a, b)		_mm256_add_pd(a, b)
#   define VEC_SWAP(a)			_mm256_shuffle_pd(a, a, _MM_SHUFFLE2(0,1))
# 	define VEC_SETZERO()		_mm256_setzero_pd()
#	define VEC_SET1(a)			_mm256_set_sd((a))
#	define VEC_SET(a, b)		_mm256_set_pd((a), (b))
#   define VEC_MOVE(a, b)		_mm256_move_sd((a), (b))
#else
	typedef float RealType;
	typedef __m256	V_Real;
#	define REALS_PER_VEC	8	/* number of elements per vector */
#	define VEC_MULT(a, b)		_mm256_mul_ps((a), (b))
#	define VEC_MADD(a, b, c)	_mm256_add_ps(_mm256_mul_ps((a), (b)), (c))
#	define VEC_SPLAT(a)			_mm256_set1_ps(a)
#	define VEC_ADD(a, b)		_mm256_add_ps(a, b)
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

int CPUSupportsAVX() {
    //int a,b,c,d;
    //cpuid(0,a,b,c,d);
    //fprintf(stderr,"a = %d\nb = %d\nc = %d\nd = %d\n",a,b,c,d);
    return 1;
}

#endif // __AVXDefinitions__
