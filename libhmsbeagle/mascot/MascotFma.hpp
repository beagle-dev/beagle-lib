/*
 * MascotFma.hpp -- explicit fused multiply-add for the MASCOT kernels.
 *
 * Both builds of a kernel compile with -ffp-contract=off (MSVC: /fp:precise), so the compiler never fuses on its own.
 * The FMA build (MASCOT_FP_FMA) fuses exactly the operations written with these macros, which makes its results the
 * same for every compiler, version and optimization level on hardware with FMA; the strict build rounds the product
 * and the sum separately, as Java does.
 *
 *   MASCOT_MUL_ADD(a, b, c)  = a * b + c for doubles
 *   MASCOT_VMUL_ADD(a, b, c) = a * b + c for kernel B's vectors, lane by lane
 *
 * The FMA build is only used on CPUs with FMA (x86: compiled with -mfma and chosen at run time), so the fused
 * operation is always the hardware instruction there (outside clang, std::fma needs -fno-math-errno for that).
 * These are macros, not inline functions, because the strict and FMA builds of a kernel must not share any inline
 * definition (the linker would keep only one of them).
 */
#ifndef MASCOT_FMA_HPP
#define MASCOT_FMA_HPP

#include <cmath>

#ifdef MASCOT_FP_FMA
#if defined(__clang__)
// The builtin is always the instruction; std::fma becomes a libm call (and blocks vectorization) under -fmath-errno.
#define MASCOT_MUL_ADD(a, b, c) __builtin_elementwise_fma((a), (b), (c))
#define MASCOT_VMUL_ADD(a, b, c) __builtin_elementwise_fma((a), (b), (c))
#else
#define MASCOT_MUL_ADD(a, b, c) std::fma((a), (b), (c))
#endif
#else
#define MASCOT_MUL_ADD(a, b, c) ((a) * (b) + (c))
#define MASCOT_VMUL_ADD(a, b, c) ((a) * (b) + (c))
#endif

#endif
