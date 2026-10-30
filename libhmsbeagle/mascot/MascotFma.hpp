/*
 * MascotFma.hpp -- explicit fused multiply-add for the MASCOT kernels.
 *
 * Both builds of a kernel compile with -ffp-contract=off (MSVC: /fp:precise), so the compiler never fuses on its own.
 * The FMA build (MASCOT_FP_FMA) fuses exactly the operations written with these macros, which makes its results the
 * same for every compiler, version and optimization level on hardware with FMA; the strict build rounds the product
 * and the sum separately, as Java does.
 *
 *   MASCOT_MUL_ADD(a, b, c)  = a * b + c for doubles
 *   MASCOT_VMUL_ADD(a, b, c) = a * b + c for kernel B's vectors, lane by lane (GCC: 2-double vectors only)
 *
 * The FMA build is only used on CPUs with FMA (x86: compiled with -mfma and chosen at run time), so the fused
 * operation is always the hardware instruction there: clang's elementwise builtin, the target's intrinsic, or GCC's
 * std::fma, which needs -fno-math-errno for that. MSVC builds kernel A only and fuses through the scalar intrinsic,
 * which needs no /arch flag. These are macros, not inline functions, because the strict and FMA builds of a kernel
 * must not share any inline definition (the linker would keep only one of them).
 */
#ifndef MASCOT_FMA_HPP
#define MASCOT_FMA_HPP

#include <cmath>

#ifdef MASCOT_FP_FMA
#if defined(__clang__)
// The builtin is always the instruction; std::fma becomes a libm call (and blocks vectorization) under -fmath-errno.
#define MASCOT_MUL_ADD(a, b, c) __builtin_elementwise_fma((a), (b), (c))
#define MASCOT_VMUL_ADD(a, b, c) __builtin_elementwise_fma((a), (b), (c))
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <immintrin.h>
#define MASCOT_MUL_ADD(a, b, c) _mm_cvtsd_f64(_mm_fmadd_sd(_mm_set_sd(a), _mm_set_sd(b), _mm_set_sd(c)))
#else
#define MASCOT_MUL_ADD(a, b, c) std::fma((a), (b), (c))
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#define MASCOT_VMUL_ADD(a, b, c) \
    ((__typeof__(a)) _mm_fmadd_pd((__m128d) (a), (__m128d) (b), (__m128d) (c)))
#elif defined(__GNUC__) && defined(__aarch64__)
#include <arm_neon.h>
#define MASCOT_VMUL_ADD(a, b, c) \
    ((__typeof__(a)) vfmaq_f64((float64x2_t) (c), (float64x2_t) (a), (float64x2_t) (b)))
#endif
#endif
#else
#define MASCOT_MUL_ADD(a, b, c) ((a) * (b) + (c))
#define MASCOT_VMUL_ADD(a, b, c) ((a) * (b) + (c))
#endif

#endif
