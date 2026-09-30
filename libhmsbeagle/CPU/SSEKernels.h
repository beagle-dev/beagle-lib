/*
 *  SSEKernels.h
 *  BEAGLE
 *
 * Copyright 2026 Phylogenetic Likelihood Working Group
 *
 * This file is part of BEAGLE.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 *
 * @author Marc Suchard
 */

#ifndef __SSEKernels__
#define __SSEKernels__

#include "libhmsbeagle/CPU/SSEDefinitions.h"

/*
 * Matrix-vector products in double precision with two states per vector (SSE2 on x86, NEON through sse2neon on
 * arm64), shared by the SSE implementations. Matrices are S x stride row-major with stride > S.
 */

namespace beagle {
namespace cpu {
namespace simd {

// Output vectors (two states each) kept in registers by one pass over the rows; each has two accumulators (even
// and odd rows) to hide the latency of the multiply-adds. x86-64 has 16 vector registers, arm64 32.
#if defined(__aarch64__)
constexpr int kBlockVectors = 8;
#else
constexpr int kBlockVectors = 4;
#endif

/*
 * out = M^T x: for each block of outputs, acc += row_j * x_j over the rows j < S. Rows are not 16-byte aligned.
 * For odd S the second lane of the last vector reads column S, which must be finite; epilogue(i, value) receives
 * the vector for outputs i and i + 1 and must not store lane i + 1 = S.
 */
template <int NV, typename Epilogue>
inline void axpyBlock(const double* __restrict rows, const int stride, const double* __restrict x, const int S,
                      const int i0, Epilogue& epilogue) {
    V_Real even[NV];
    V_Real odd[NV];
    for (int v = 0; v < NV; ++v) {
        even[v] = VEC_SETZERO();
        odd[v] = VEC_SETZERO();
    }
    const double* row = rows + i0;
    int j = 0;
    for (; j + 1 < S; j += 2, row += 2 * stride) {
        const V_Real x0 = VEC_SPLAT(x[j]);
        const V_Real x1 = VEC_SPLAT(x[j + 1]);
        for (int v = 0; v < NV; ++v) {
            even[v] = VEC_MADD(VEC_LOADU(row + 2 * v), x0, even[v]);
            odd[v] = VEC_MADD(VEC_LOADU(row + stride + 2 * v), x1, odd[v]);
        }
    }
    if (j < S) {
        const V_Real x0 = VEC_SPLAT(x[j]);
        for (int v = 0; v < NV; ++v) {
            even[v] = VEC_MADD(VEC_LOADU(row + 2 * v), x0, even[v]);
        }
    }
    for (int v = 0; v < NV; ++v) {
        epilogue(i0 + 2 * v, VEC_ADD(even[v], odd[v]));
    }
}

// Blocks of nearly equal size, at most kBlockVectors vectors each
template <typename Epilogue>
inline void axpy(const double* __restrict rows, const int stride, const double* __restrict x, const int S,
                 Epilogue epilogue) {
    const int vectors = (S + 1) / 2;
    const int blocks = (vectors + kBlockVectors - 1) / kBlockVectors;
    const int size = vectors / blocks;
    const int larger = vectors % blocks; // the first 'larger' blocks have size + 1 vectors
    int i0 = 0;
    for (int b = 0; b < blocks; ++b) {
        switch (size + (b < larger ? 1 : 0)) {
            case 1: axpyBlock<1>(rows, stride, x, S, i0, epilogue); i0 += 2; break;
            case 2: axpyBlock<2>(rows, stride, x, S, i0, epilogue); i0 += 4; break;
            case 3: axpyBlock<3>(rows, stride, x, S, i0, epilogue); i0 += 6; break;
            case 4: axpyBlock<4>(rows, stride, x, S, i0, epilogue); i0 += 8; break;
#if defined(__aarch64__)
            case 5: axpyBlock<5>(rows, stride, x, S, i0, epilogue); i0 += 10; break;
            case 6: axpyBlock<6>(rows, stride, x, S, i0, epilogue); i0 += 12; break;
            case 7: axpyBlock<7>(rows, stride, x, S, i0, epilogue); i0 += 14; break;
            case 8: axpyBlock<8>(rows, stride, x, S, i0, epilogue); i0 += 16; break;
#endif
            default: break;
        }
    }
}

/*
 * out = M x, from the dot products of the rows with x: four rows per pass, each with two accumulators (alternate
 * pairs of columns), and two rows per output vector. Only columns j < S are read. epilogue(i, value) receives the
 * vector for outputs i and i + 1 and must not store lane i + 1 = S.
 */
template <typename Epilogue>
inline void dot(const double* __restrict rows, const int stride, const double* __restrict x, const int S,
                Epilogue epilogue) {
    const int pairs = S & ~1; // columns read two at a time
    const bool odd = (S & 1) != 0;
    const V_Real last = odd ? VEC_SET1(x[S - 1]) : VEC_SETZERO(); // (x_{S-1}, 0)
    auto reduce = [](const V_Real a, const V_Real b) { // (sum of a, sum of b)
        return VEC_ADD(VEC_SHUFFLE0(a, b), VEC_SHUFFLE1(a, b));
    };
    int i = 0;
    for (; i + 4 <= S; i += 4) {
        const double* r[4] = {rows + i * stride, rows + (i + 1) * stride, rows + (i + 2) * stride,
                              rows + (i + 3) * stride};
        V_Real a[4], b[4];
        for (int q = 0; q < 4; ++q) {
            a[q] = VEC_SETZERO();
            b[q] = VEC_SETZERO();
        }
        int j = 0;
        for (; j + 4 <= pairs; j += 4) {
            const V_Real x0 = VEC_LOADU(x + j);
            const V_Real x1 = VEC_LOADU(x + j + 2);
            for (int q = 0; q < 4; ++q) {
                a[q] = VEC_MADD(VEC_LOADU(r[q] + j), x0, a[q]);
                b[q] = VEC_MADD(VEC_LOADU(r[q] + j + 2), x1, b[q]);
            }
        }
        if (j < pairs) {
            const V_Real x0 = VEC_LOADU(x + j);
            for (int q = 0; q < 4; ++q) {
                a[q] = VEC_MADD(VEC_LOADU(r[q] + j), x0, a[q]);
            }
        }
        if (odd) {
            for (int q = 0; q < 4; ++q) {
                b[q] = VEC_MADD(VEC_SPLAT(r[q][S - 1]), last, b[q]);
            }
        }
        epilogue(i, reduce(VEC_ADD(a[0], b[0]), VEC_ADD(a[1], b[1])));
        epilogue(i + 2, reduce(VEC_ADD(a[2], b[2]), VEC_ADD(a[3], b[3])));
    }
    for (; i < S; i += 2) { // two rows, or one when i + 1 = S
        const double* r0 = rows + i * stride;
        const double* r1 = (i + 1 < S) ? r0 + stride : r0;
        V_Real a0 = VEC_SETZERO();
        V_Real a1 = VEC_SETZERO();
        for (int j = 0; j < pairs; j += 2) {
            const V_Real x0 = VEC_LOADU(x + j);
            a0 = VEC_MADD(VEC_LOADU(r0 + j), x0, a0);
            a1 = VEC_MADD(VEC_LOADU(r1 + j), x0, a1);
        }
        if (odd) {
            a0 = VEC_MADD(VEC_SPLAT(r0[S - 1]), last, a0);
            a1 = VEC_MADD(VEC_SPLAT(r1[S - 1]), last, a1);
        }
        epilogue(i, reduce(a0, a1));
    }
}

// Stores outputs i and i + 1, or only i when i + 1 = S
inline void storeVector(double* out, const int i, const int S, const V_Real value) {
    if (i + 1 < S) {
        VEC_STOREU(out + i, value);
    } else {
        VEC_STORE_SCALAR(out + i, value);
    }
}

// Loads inputs i and i + 1 of a vector with S entries, without reading past entry S - 1
inline V_Real loadVector(const double* in, const int i, const int S) {
    return (i + 1 < S) ? VEC_LOADU(in + i) : VEC_LOAD_SCALAR(in + i);
}

// Entries (i, column) and (i + 1, column) of M, only the first when i + 1 = S
inline V_Real loadColumn(const double* m, const int stride, const int i, const int column, const int S) {
    const double first = m[i * stride + column];
    return (i + 1 < S) ? VEC_SET(m[(i + 1) * stride + column], first) : VEC_SPLAT(first);
}

// (lo, hi) = sum_j x_j * row_j for the four rows of a 4 x 4 row-major, 16-byte aligned matrix; x = (x01, x23)
inline void product4(const double* __restrict rows, const V_Real x01, const V_Real x23, V_Real& lo, V_Real& hi) {
    const V_Real x0 = VEC_SHUFFLE0(x01, x01);
    const V_Real x1 = VEC_SHUFFLE1(x01, x01);
    const V_Real x2 = VEC_SHUFFLE0(x23, x23);
    const V_Real x3 = VEC_SHUFFLE1(x23, x23);
    V_Real a = VEC_MULT(x0, VEC_LOAD(rows));
    V_Real b = VEC_MULT(x0, VEC_LOAD(rows + 2));
    V_Real c = VEC_MULT(x1, VEC_LOAD(rows + 4));
    V_Real d = VEC_MULT(x1, VEC_LOAD(rows + 6));
    a = VEC_MADD(x2, VEC_LOAD(rows + 8), a);
    b = VEC_MADD(x2, VEC_LOAD(rows + 10), b);
    c = VEC_MADD(x3, VEC_LOAD(rows + 12), c);
    d = VEC_MADD(x3, VEC_LOAD(rows + 14), d);
    lo = VEC_ADD(a, c);
    hi = VEC_ADD(b, d);
}

} // namespace simd
} // namespace cpu
} // namespace beagle

#endif // __SSEKernels__
