/*
 *  BeagleCPU4StateSSEImpl.h
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
 */

#ifndef __SSEDefinitions__
#define __SSEDefinitions__

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#define DLS_USE_SSE2

#if defined(DLS_USE_SSE2)
#	if defined(__aarch64__)
#		include "libhmsbeagle/CPU/sse2neon.h"
#       define _MM_SHUFFLE2(fp1,fp0) (((fp1) << 1) | (fp0))
#		define VEC_SHUFFLE0(a,b)	_mm_shuffle_pd(a, b, _MM_SHUFFLE2(0,0)) // vreinterpretq_f64_m128d(a)
#		define VEC_SHUFFLE1(a,b)	_mm_shuffle_pd(a, b, _MM_SHUFFLE2(1,1)) // vreinterpretq_f64_m128d(a)

#   else
#		if !defined(DLS_MACOS)
#			include <emmintrin.h>
#		endif
#       include <pmmintrin.h>
#		include <xmmintrin.h>
#       include <smmintrin.h>
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

//inline XMVECTOR XMVector4Dot( FXMVECTOR V1, FXMVECTOR V2 )
//{
//#if defined(_XM_NO_INTRINSICS_)
//
//    XMVECTOR Result;
//    Result.vector4_f32[0] =
//    Result.vector4_f32[1] =
//    Result.vector4_f32[2] =
//    Result.vector4_f32[3] = V1.vector4_f32[0] * V2.vector4_f32[0] + V1.vector4_f32[1] * V2.vector4_f32[1] + V1.vector4_f32[2] * V2.vector4_f32[2] + V1.vector4_f32[3] * V2.vector4_f32[3];
//    return Result;
//
//#elif defined(_M_ARM) || defined(_M_ARM64)
//
//    float32x4_t vTemp = vmulq_f32( V1, V2 );
//    float32x2_t v1 = vget_low_f32( vTemp );
//    float32x2_t v2 = vget_high_f32( vTemp );
//    v1 = vpadd_f32( v1, v1 );
//    v2 = vpadd_f32( v2, v2 );
//    v1 = vadd_f32( v1, v2 );
//    return vcombine_f32( v1, v1 );
//
//#elif defined(__AVX__) || defined(__AVX2__)
//
//    return _mm_dp_ps( V1, V2, 0xff );
//
//#elif defined(_M_IX86) || defined(_M_X64)
//
//    XMVECTOR vTemp2 = V2;
//    XMVECTOR vTemp = _mm_mul_ps(V1,vTemp2);
//    vTemp2 = _mm_shuffle_ps(vTemp2,vTemp,_MM_SHUFFLE(1,0,0,0));
//    vTemp2 = _mm_add_ps(vTemp2,vTemp);
//    vTemp = _mm_shuffle_ps(vTemp,vTemp2,_MM_SHUFFLE(0,3,0,0));
//    vTemp = _mm_add_ps(vTemp,vTemp2);
//    return _mm_shuffle_ps(vTemp,vTemp,_MM_SHUFFLE(2,2,2,2));
//
//#else
//#error Unsupported platform
//#endif
//}

#endif // __SSEDefinitions__
