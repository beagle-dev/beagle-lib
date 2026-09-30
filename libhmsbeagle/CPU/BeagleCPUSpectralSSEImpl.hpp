/*
 *  BeagleCPUSpectralSSEImpl.hpp
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

#ifndef BEAGLE_CPU_SPECTRAL_SSE_IMPL_HPP
#define BEAGLE_CPU_SPECTRAL_SSE_IMPL_HPP

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <utility>

#include "libhmsbeagle/beagle.h"
#include "libhmsbeagle/CPU/BeagleCPUSpectralSSEImpl.h"
#include "libhmsbeagle/CPU/SSEDefinitions.h"

namespace beagle {
namespace cpu {

namespace spectral_sse {

// Output vectors (two states each) kept in registers by one pass over the rows; each has two accumulators (even
// and odd rows) to hide the latency of the multiply-adds. x86-64 has 16 vector registers, arm64 32.
#if defined(__aarch64__)
constexpr int kBlockVectors = 8;
#else
constexpr int kBlockVectors = 4;
#endif

/*
 * out = M^T x for an S x stride row-major M: for each block of outputs, acc += row_j * x_j over the rows j < S.
 * Rows are not 16-byte aligned (stride S + T_PAD). For odd S the second lane of the last vector reads the pad
 * column, which holds finite row sums; epilogue(i, value) receives the vector for outputs i and i + 1 and must not
 * store lane i + 1 = S.
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

/*
 * y = exp(D t) u: y_i = e_i u_i for a real eigenvalue; a complex conjugate pair (i, i + 1) is rotated,
 * y_i = c u_i + s u_{i+1} and y_{i+1} = c u_{i+1} - s u_i, with s negated going backward (P^T). imag is null
 * when every eigenvalue is real.
 */
template <bool Backward>
inline void expScale(double* __restrict y, const double* __restrict u,
                     const double* __restrict expat,
                     const double* __restrict expatcosbt, const double* __restrict expatsinbt,
                     const double* __restrict imag, const int S) {
    if (imag == nullptr) {
        for (int i = 0; i < S; ++i) {
            y[i] = expat[i] * u[i];
        }
        return;
    }
    for (int i = 0; i < S; ) {
        if (imag[i] == 0.0) {
            y[i] = expat[i] * u[i];
            ++i;
        } else {
            const double c = expatcosbt[i];
            const double s = Backward ? -expatsinbt[i] : expatsinbt[i];
            const double a = u[i];
            const double b = u[i + 1];
            y[i]     = c * a + s * b;
            y[i + 1] = c * b - s * a;
            i += 2;
        }
    }
}

// (v1, -v0)
inline V_Real swapNegate(const V_Real v) {
    return VEC_MULT(VEC_SWAP(v), VEC_SET(-1.0, 1.0));
}

// The outer product a b^T of two vectors; row(l) holds a_l in registers. shared(r) is what the rows of a block
// have in common for entries r, r + 1 (here b), and pair(shared, r) the entries of one row.
struct RankOneView {
    const double* a;
    const double* b;
    struct Row {
        V_Real scale;
        double scalar;
        const double* b;
        V_Real pair(const int r) const { return VEC_MULT(scale, VEC_LOADU(b + r)); } // entries r, r + 1
        V_Real pair(const V_Real shared, const int) const { return VEC_MULT(scale, shared); }
        double at(const int r) const { return scalar * b[r]; }
    };
    Row row(const int l) const { return Row{VEC_SPLAT(a[l]), a[l], b}; }
    V_Real shared(const int r) const { return VEC_LOADU(b + r); }
};

// A dense matrix with row stride
struct DenseView {
    const double* m;
    int stride;
    struct Row {
        const double* m;
        V_Real pair(const int r) const { return VEC_LOADU(m + r); }
        V_Real pair(const V_Real, const int r) const { return VEC_LOADU(m + r); }
        double at(const int r) const { return m[r]; }
    };
    Row row(const int l) const { return Row{m + l * stride}; }
    V_Real shared(const int) const { return VEC_SETZERO(); }
};

} // namespace spectral_sse

template <int T_PAD, int P_PAD>
const char* BeagleCPUSpectralSSEImpl<T_PAD, P_PAD>::getName() {
    return "CPU-Spectral-SSE-Double";
}

template <int T_PAD, int P_PAD>
const long BeagleCPUSpectralSSEImpl<T_PAD, P_PAD>::getFlags() {
    return BEAGLE_FLAG_COMPUTATION_SYNCH |
           BEAGLE_FLAG_PROCESSOR_CPU |
           BEAGLE_FLAG_PRECISION_DOUBLE |
           BEAGLE_FLAG_VECTOR_SSE |
           BEAGLE_FLAG_FRAMEWORK_CPU;
}

template <int T_PAD, int P_PAD>
int BeagleCPUSpectralSSEImpl<T_PAD, P_PAD>::createInstance(int tipCount,
                                                          int partialsBufferCount,
                                                          int compactBufferCount,
                                                          int stateCount,
                                                          int patternCount,
                                                          int eigenDecompositionCount,
                                                          int matrixCount,
                                                          int categoryCount,
                                                          int scaleBufferCount,
                                                          int resourceNumber,
                                                          int pluginResourceNumber,
                                                          long preferenceFlags,
                                                          long requirementFlags) {

    const int returnCode = Base::createInstance(tipCount, partialsBufferCount, compactBufferCount, stateCount,
                                                patternCount, eigenDecompositionCount, matrixCount, categoryCount,
                                                scaleBufferCount, resourceNumber, pluginResourceNumber,
                                                preferenceFlags, requirementFlags);

    kSimdTmpStride = (kStateCount + 3) & ~1; // at least S + 1, and even
    gSimdTmp.assign(3 * kSimdTmpStride * kPartitionCount, 0.0);

    kAdjointStride = kStateCount + (kStateCount & 1);
    gAdjointPlans.assign(kEigenDecompCount, AdjointPlan());
    gAdjointPlanStale.assign(kEigenDecompCount, false);
    gAdjointOuterTmp.assign(kStateCount * kAdjointStride * kPartitionCount, 0.0);
    kAdjointPairTmpStride = kStateCount / 2 + 2;
    gAdjointPairTmp.assign(2 * kAdjointPairTmpStride * kPartitionCount, 0.0);

    return returnCode;
}

template <int T_PAD, int P_PAD>
int BeagleCPUSpectralSSEImpl<T_PAD, P_PAD>::setCPUThreadCount(int threadCount) {
    const int returnCode = Base::setCPUThreadCount(threadCount);
    gSimdTmp.assign(3 * kSimdTmpStride * kPartitionCount, 0.0);
    gAdjointOuterTmp.assign(kStateCount * kAdjointStride * kPartitionCount, 0.0);
    gAdjointPairTmp.assign(2 * kAdjointPairTmpStride * kPartitionCount, 0.0);
    return returnCode;
}

template <int T_PAD, int P_PAD>
int BeagleCPUSpectralSSEImpl<T_PAD, P_PAD>::setEigenDecomposition(int eigenIndex,
                                                                 const double* inEigenVectors,
                                                                 const double* inInverseEigenVectors,
                                                                 const double* inEigenValues) {
    const int returnCode = Base::setEigenDecomposition(eigenIndex, inEigenVectors, inInverseEigenVectors,
                                                       inEigenValues);
    if (returnCode == BEAGLE_SUCCESS) {
        gAdjointPlanStale[eigenIndex] = true; // built by the next adjoint gradient that uses it
    }
    return returnCode;
}

template <int T_PAD, int P_PAD>
void BeagleCPUSpectralSSEImpl<T_PAD, P_PAD>::prepareAdjoint(const int* branchEigenIndices, int count) {
    Base::prepareAdjoint(branchEigenIndices, count);
    for (int i = 0; i < count; ++i) {
        const int eigenIndex = gBranchEigenInfo[branchEigenIndices[i]].eigenIndex;
        if (gAdjointPlanStale[eigenIndex]) {
            prepareAdjointPlan(eigenIndex);
            gAdjointPlanStale[eigenIndex] = false;
        }
    }
}

template <int T_PAD, int P_PAD>
void BeagleCPUSpectralSSEImpl<T_PAD, P_PAD>::prepareAdjointPlan(int eigenIndex) {
    const int S = kStateCount;
    const double* eval = gEigenDecomposition->getEigenValuesPtr(eigenIndex);
    const double* imag = (kFlags & BEAGLE_FLAG_EIGEN_COMPLEX) ? eval + S : nullptr;

    AdjointPlan& plan = gAdjointPlans[eigenIndex];
    plan.realIndices.clear();
    plan.pairIndices.clear();
    for (int i = 0; i < S; ) {
        if (imag == nullptr || imag[i] == 0.0) {
            plan.realIndices.push_back(i);
            ++i;
        } else {
            plan.pairIndices.push_back(i);
            i += 2;
        }
    }
    plan.allReal = plan.pairIndices.empty();

    // 1 / (lambda_l - lambda_r), 0 for equal eigenvalues, whose kernel entry is always degenerate (t e_l)
    plan.reciprocals.assign(S * kAdjointStride, 0.0);
    plan.equalPairs.clear();
    plan.smallestDistance = std::numeric_limits<double>::infinity();
    double* table = plan.reciprocals.data();
    auto reciprocal = [](const double denominator) { return (denominator < 1e-12) ? 0.0 : 1.0 / denominator; };
    for (int l : plan.realIndices) {
        for (int r : plan.realIndices) {
            const double distance = eval[l] - eval[r];
            if (distance == 0.0) {
                plan.equalPairs.push_back(std::make_pair(l, r));
            } else {
                table[l * kAdjointStride + r] = 1.0 / distance;
                plan.smallestDistance = std::min(plan.smallestDistance, std::abs(distance));
            }
        }
        for (int rs : plan.pairIndices) { // real row, conjugate pair of columns
            const double sr = eval[rs] - eval[l];
            table[l * kAdjointStride + rs] = reciprocal(sr * sr + imag[rs] * imag[rs]);
        }
    }
    for (int ls : plan.pairIndices) {
        const double li = imag[ls];
        for (int r : plan.realIndices) { // conjugate pair of rows, real column
            const double sr = eval[r] - eval[ls];
            table[ls * kAdjointStride + r] = reciprocal(sr * sr + li * li);
        }
    }

    const int pairCount = static_cast<int>(plan.pairIndices.size());
    plan.pairStride = pairCount + (pairCount & 1);
    plan.pairBlocks.assign(5 * plan.pairStride * pairCount, 0.0);
    plan.degenerateBlocks.clear();
    for (int pl = 0; pl < pairCount; ++pl) {
        const int ls = plan.pairIndices[pl];
        double* block = plan.pairBlocks.data() + 5 * plan.pairStride * pl;
        for (int pr = 0; pr < pairCount; ++pr) {
            const int rs = plan.pairIndices[pr];
            const double sr = eval[rs] - eval[ls];
            const double sum = imag[ls] + imag[rs];
            const double difference = imag[rs] - imag[ls];
            const double d1 = sr * sr + sum * sum;
            const double d2 = sr * sr + difference * difference;
            block[pr] = sr;
            block[plan.pairStride + pr] = sum;
            block[2 * plan.pairStride + pr] = difference;
            block[3 * plan.pairStride + pr] = (d1 < 1e-12) ? 0.0 : 0.5 / d1;
            block[4 * plan.pairStride + pr] = (d2 < 1e-12) ? 0.0 : 0.5 / d2;
            if (d1 < 1e-12) {
                plan.degenerateBlocks.push_back({pl, pr, 1});
            }
            if (d2 < 1e-12) {
                plan.degenerateBlocks.push_back({pl, pr, 2});
            }
        }
    }
}

/*
 * The integral kernel of AdjointIntegralPlan::accumulateEigenBasisGradient, without divisions: the reciprocals
 * depend only on the eigenvalues, and exp(a t) cos(b t), exp(a t) sin(b t) of both indices replace the ratio
 * exp(a_r t) / exp(a_l t) and the angle sums (they combine by the angle addition formulas). Real rows and columns
 * are vectorized over columns when every eigenvalue is real; a conjugate pair of columns is one vector.
 */
template <int T_PAD, int P_PAD> template <typename View>
void BeagleCPUSpectralSSEImpl<T_PAD, P_PAD>::adjointKernel(double* gradient, const View& view,
                                                          const AdjointPlan& plan, const double* eval,
                                                          const BranchEigenInfo& info, int infoOffset,
                                                          double time, double* pairTmp) {
    const int S = kStateCount;
    const double* __restrict expat = info.expat + infoOffset;
    const double* __restrict table = plan.reciprocals.data();

    if (plan.allReal && time * plan.smallestDistance >= 1e-12 &&
            static_cast<int>(plan.equalPairs.size()) <= 4 * S) {
        // the degenerate entries are exactly the equal eigenvalues, whose reciprocal is 0 here. Four rows per pass
        // share the loads of expat and the post-order side, and their stores are independent.
        constexpr int kRows = 4;
        int l = 0;
        for (; l + kRows <= S; l += kRows) {
            typename View::Row in[kRows] = {view.row(l), view.row(l + 1), view.row(l + 2), view.row(l + 3)};
            const double* reciprocals[kRows];
            double* g[kRows];
            V_Real vea[kRows];
            for (int q = 0; q < kRows; ++q) {
                reciprocals[q] = table + (l + q) * kAdjointStride;
                g[q] = gradient + (l + q) * S;
                vea[q] = VEC_SPLAT(expat[l + q]);
            }
            int r = 0;
            for (; r + 1 < S; r += 2) {
                const V_Real er = VEC_LOADU(expat + r);
                const V_Real shared = view.shared(r);
                for (int q = 0; q < kRows; ++q) {
                    const V_Real coefficient = VEC_MULT(VEC_SUB(vea[q], er), VEC_LOADU(reciprocals[q] + r));
                    VEC_STOREU(g[q] + r, VEC_MADD(in[q].pair(shared, r), coefficient, VEC_LOADU(g[q] + r)));
                }
            }
            if (r < S) {
                for (int q = 0; q < kRows; ++q) {
                    g[q][r] += in[q].at(r) * ((expat[l + q] - expat[r]) * reciprocals[q][r]);
                }
            }
        }
        for (; l < S; ++l) {
            const auto in = view.row(l);
            const double* __restrict reciprocals = table + l * kAdjointStride;
            double* __restrict g = gradient + l * S;
            const double ea = expat[l];
            const V_Real vea = VEC_SPLAT(ea);
            int r = 0;
            for (; r + 1 < S; r += 2) {
                const V_Real coefficient = VEC_MULT(VEC_SUB(vea, VEC_LOADU(expat + r)), VEC_LOADU(reciprocals + r));
                VEC_STOREU(g + r, VEC_MADD(in.pair(r), coefficient, VEC_LOADU(g + r)));
            }
            if (r < S) {
                g[r] += in.at(r) * ((ea - expat[r]) * reciprocals[r]);
            }
        }
        for (const auto& pair : plan.equalPairs) {
            gradient[pair.first * S + pair.second] +=
                    view.row(pair.first).at(pair.second) * (time * expat[pair.first]);
        }
        return;
    }

    if (plan.allReal) {
        const V_Real t = VEC_SPLAT(time);
        const V_Real threshold = VEC_SPLAT(1e-12);
        const V_Real signBit = VEC_SPLAT(-0.0);
        for (int l = 0; l < S; ++l) {
            const auto in = view.row(l);
            const double* __restrict reciprocals = table + l * kAdjointStride;
            double* __restrict g = gradient + l * S;
            const double la = eval[l];
            const double ea = expat[l];
            const V_Real vla = VEC_SPLAT(la);
            const V_Real vea = VEC_SPLAT(ea);
            const V_Real degenerateValue = VEC_SPLAT(time * ea);
            int r = 0;
            for (; r + 1 < S; r += 2) {
                const V_Real distance = VEC_ANDNOT(signBit, VEC_SUB(vla, VEC_LOADU(eval + r)));
                const V_Real degenerate = VEC_CMPLT(VEC_MULT(t, distance), threshold);
                const V_Real ratio = VEC_MULT(VEC_SUB(vea, VEC_LOADU(expat + r)), VEC_LOADU(reciprocals + r));
                const V_Real coefficient = VEC_OR(VEC_AND(degenerate, degenerateValue),
                                                  VEC_ANDNOT(degenerate, ratio));
                VEC_STOREU(g + r, VEC_MADD(in.pair(r), coefficient, VEC_LOADU(g + r)));
            }
            if (r < S) {
                const double coefficient = (time * std::abs(la - eval[r]) < 1e-12) ?
                        time * ea : (ea - expat[r]) * reciprocals[r];
                g[r] += in.at(r) * coefficient;
            }
        }
        return;
    }

    const double* __restrict imag = eval + S;
    const double* __restrict expatcosbt = info.expatcosbt + infoOffset;
    const double* __restrict expatsinbt = info.expatsinbt + infoOffset;

    for (int li : plan.realIndices) {
        const auto in = view.row(li);
        const double* __restrict reciprocals = table + li * kAdjointStride;
        double* __restrict g = gradient + li * S;
        const double la = eval[li];
        const double ea = expat[li];

        for (int ri : plan.realIndices) {
            const double coefficient = (time * std::abs(la - eval[ri]) < 1e-12) ?
                    time * ea : (ea - expat[ri]) * reciprocals[ri];
            g[ri] += in.at(ri) * coefficient;
        }

        for (int rs : plan.pairIndices) {
            const double reciprocal = reciprocals[rs];
            double c0 = time * ea;
            double c1 = 0.0;
            if (reciprocal != 0.0) {
                const double sr = eval[rs] - la;
                const double ri = imag[rs];
                c0 = (sr * expatcosbt[rs] + ri * expatsinbt[rs] - sr * ea) * reciprocal;
                c1 = (sr * expatsinbt[rs] - ri * expatcosbt[rs] + ri * ea) * reciprocal;
            }
            const V_Real pair = in.pair(rs);
            VEC_STOREU(g + rs, VEC_ADD(VEC_LOADU(g + rs),
                    VEC_ADD(VEC_MULT(VEC_SPLAT(c0), pair), VEC_MULT(VEC_SPLAT(c1), spectral_sse::swapNegate(pair)))));
        }
    }

    for (int ls : plan.pairIndices) {
        const auto in0 = view.row(ls);
        const auto in1 = view.row(ls + 1);
        const double* __restrict reciprocals = table + ls * kAdjointStride;
        double* __restrict g0 = gradient + ls * S;
        double* __restrict g1 = g0 + S;
        const double lr = eval[ls];
        const double li = imag[ls];
        const double er = expatcosbt[ls];
        const double ei = expatsinbt[ls];

        for (int ri : plan.realIndices) {
            const double reciprocal = reciprocals[ri];
            double p0 = er * time;
            double p1 = -ei * time;
            if (reciprocal != 0.0) {
                const double sr = eval[ri] - lr;
                p0 = (sr * (expat[ri] - er) + li * ei) * reciprocal;
                p1 = (li * (er - expat[ri]) + sr * ei) * reciprocal;
            }
            const double a0 = in0.at(ri);
            const double a1 = in1.at(ri);
            g0[ri] += p0 * a0 + p1 * a1;
            g1[ri] += p0 * a1 - p1 * a0;
        }

    }

    // Two conjugate pairs, two blocks of columns per vector. With A = mr + pr, B = mi + pi, C = pi - mi and
    // D = mr - pr (halves folded into the reciprocals), rows ls and ls + 1 of a block of columns (rs, rs + 1) are
    // g0 += A in0 + B sn(in0) + C in1 + D sn(in1) and g1 += A in1 + B sn(in1) - C in0 - D sn(in0), sn(v) = (v1, -v0).
    const int pairCount = static_cast<int>(plan.pairIndices.size());
    const int pairStride = plan.pairStride;
    const int* pairIndices = plan.pairIndices.data();
    double* EC = pairTmp;
    double* ES = pairTmp + kAdjointPairTmpStride;
    for (int p = 0; p < pairCount; ++p) {
        EC[p] = expatcosbt[pairIndices[p]];
        ES[p] = expatsinbt[pairIndices[p]];
    }
    EC[pairCount] = 0.0;
    ES[pairCount] = 0.0;

    auto applyBlock = [&](const typename View::Row& in0, const typename View::Row& in1,
                          double* __restrict g0, double* __restrict g1, const int rs,
                          const V_Real A, const V_Real B, const V_Real C, const V_Real D) {
        const V_Real i0 = in0.pair(rs);
        const V_Real i1 = in1.pair(rs);
        const V_Real s0 = spectral_sse::swapNegate(i0);
        const V_Real s1 = spectral_sse::swapNegate(i1);
        VEC_STOREU(g0 + rs, VEC_ADD(VEC_LOADU(g0 + rs),
                VEC_ADD(VEC_MADD(A, i0, VEC_MULT(B, s0)), VEC_MADD(C, i1, VEC_MULT(D, s1)))));
        VEC_STOREU(g1 + rs, VEC_ADD(VEC_LOADU(g1 + rs),
                VEC_SUB(VEC_MADD(A, i1, VEC_MULT(B, s1)), VEC_MADD(C, i0, VEC_MULT(D, s0)))));
    };

    for (int pl = 0; pl < pairCount; ++pl) {
        const int ls = pairIndices[pl];
        const auto in0 = view.row(ls);
        const auto in1 = view.row(ls + 1);
        double* __restrict g0 = gradient + ls * S;
        double* __restrict g1 = g0 + S;
        const double* __restrict sr = plan.pairBlocks.data() + 5 * pairStride * pl;
        const double* __restrict sum = sr + pairStride;
        const double* __restrict difference = sum + pairStride;
        const double* __restrict half1 = difference + pairStride;
        const double* __restrict half2 = half1 + pairStride;
        const V_Real er = VEC_SPLAT(EC[pl]);
        const V_Real ei = VEC_SPLAT(ES[pl]);

        for (int pr = 0; pr < pairCount; pr += 2) {
            const V_Real vsr = VEC_LOADU(sr + pr);
            const V_Real vsum = VEC_LOADU(sum + pr);
            const V_Real vdifference = VEC_LOADU(difference + pr);
            const V_Real U = VEC_SUB(VEC_LOADU(EC + pr), er);
            const V_Real esr = VEC_LOADU(ES + pr);
            const V_Real W1 = VEC_ADD(esr, ei);
            const V_Real W2 = VEC_SUB(esr, ei);
            const V_Real h1 = VEC_LOADU(half1 + pr);
            const V_Real h2 = VEC_LOADU(half2 + pr);
            const V_Real prv = VEC_MULT(VEC_MADD(vsr, U, VEC_MULT(vsum, W1)), h1);
            const V_Real piv = VEC_MULT(VEC_SUB(VEC_MULT(vsr, W1), VEC_MULT(vsum, U)), h1);
            const V_Real mrv = VEC_MULT(VEC_MADD(vsr, U, VEC_MULT(vdifference, W2)), h2);
            const V_Real miv = VEC_MULT(VEC_SUB(VEC_MULT(vsr, W2), VEC_MULT(vdifference, U)), h2);
            const V_Real A = VEC_ADD(mrv, prv);
            const V_Real B = VEC_ADD(miv, piv);
            const V_Real C = VEC_SUB(piv, miv);
            const V_Real D = VEC_SUB(mrv, prv);

            applyBlock(in0, in1, g0, g1, pairIndices[pr],
                       VEC_SHUFFLE0(A, A), VEC_SHUFFLE0(B, B), VEC_SHUFFLE0(C, C), VEC_SHUFFLE0(D, D));
            if (pr + 1 < pairCount) {
                applyBlock(in0, in1, g0, g1, pairIndices[pr + 1],
                           VEC_SHUFFLE1(A, A), VEC_SHUFFLE1(B, B), VEC_SHUFFLE1(C, C), VEC_SHUFFLE1(D, D));
            }
        }
    }

    // degenerate blocks: the reciprocal was 0, and the integral of that term is t (halved as above)
    for (const auto& block : plan.degenerateBlocks) {
        const int ls = pairIndices[block[0]];
        const int rs = pairIndices[block[1]];
        const double x = 0.5 * EC[block[0]] * time;
        const double y = 0.5 * ES[block[0]] * time;
        const V_Real vx = VEC_SPLAT(x);
        const V_Real vy = VEC_SPLAT(y);
        const V_Real nx = VEC_SPLAT(-x);
        const V_Real ny = VEC_SPLAT(-y);
        double* g0 = gradient + ls * S;
        if (block[2] == 1) { // pr = e cos t, pi = -e sin t
            applyBlock(view.row(ls), view.row(ls + 1), g0, g0 + S, rs, vx, ny, ny, nx);
        } else {             // mr = e cos t, mi = e sin t
            applyBlock(view.row(ls), view.row(ls + 1), g0, g0 + S, rs, vx, vy, ny, vx);
        }
    }
}

template <int T_PAD, int P_PAD>
void BeagleCPUSpectralSSEImpl<T_PAD, P_PAD>::calcAdjointCrossProductsRange(const int* postBufferIndices,
                                                                          const int* preBufferIndices,
                                                                          const int* branchEigenIndices,
                                                                          const double* categoryRates,
                                                                          const double* categoryWeights,
                                                                          const double* perSiteLikelihoods,
                                                                          double* buffer,
                                                                          int startNode,
                                                                          int endNode,
                                                                          int startPattern,
                                                                          int endPattern,
                                                                          int currentPartition,
                                                                          const int* postScaleIndices,
                                                                          const int* preScaleIndices,
                                                                          const int cumulativeScaleIndex,
                                                                          double* branchMarginalLk) {

    if (postScaleIndices != nullptr || preScaleIndices != nullptr || branchMarginalLk != nullptr) {
        Base::calcAdjointCrossProductsRange(postBufferIndices, preBufferIndices, branchEigenIndices,
                                            categoryRates, categoryWeights, perSiteLikelihoods, buffer,
                                            startNode, endNode, startPattern, endPattern, currentPartition,
                                            postScaleIndices, preScaleIndices, cumulativeScaleIndex,
                                            branchMarginalLk);
        return;
    }

    const int S = kStateCount;
    const int stride = kStateCount + T_PAD;
    const bool onePattern = (endPattern - startPattern == 1);

    double* lhs = simdTmp(currentPartition);
    double* rhs = lhs + kSimdTmpStride;
    double* outer = gAdjointOuterTmp.data() + currentPartition * S * kAdjointStride;
    double* pairTmp = gAdjointPairTmp.data() + currentPartition * 2 * kAdjointPairTmpStride;

    for (int category = 0; category < kCategoryCount; ++category) {
        const double categoryRate = categoryRates[category];
        const int infoOffset = category * kPartialsPaddedStateCount;
        const double categoryWeight = categoryWeights[category];

        for (int node = startNode; node < endNode; ++node) {
            const BranchEigenInfo& info = gBranchEigenInfo[branchEigenIndices[node]];
            const double time = categoryRate * info.branchLength;
            const AdjointPlan& plan = gAdjointPlans[info.eigenIndex];
            // rows of V are the columns of V^T; rows of (V^{-1})^T are the columns of V^{-1}
            const double* transposeColumns = gEigenDecomposition->getEigenVectorsPtr(info.eigenIndex);
            const double* inverseColumns = gEigenDecomposition->getBackwardsEigenVectorsPtr(info.eigenIndex);

            const double* pre = gPartials[preBufferIndices[node]];
            const int* tipStates = gTipStates[postBufferIndices[node]];
            const double* post = (tipStates != nullptr) ? nullptr : gPartials[postBufferIndices[node]];

            if (!onePattern) {
                std::fill(outer, outer + S * kAdjointStride, 0.0);
            }

            for (int k = startPattern; k < endPattern; ++k) {
                const int v = category * kPartialsPaddedStateCount * kPatternCount + kPartialsPaddedStateCount * k;
                const V_Real scale = VEC_SPLAT(gPatternWeights[k] * categoryWeight / perSiteLikelihoods[k]);

                // lhs = scale V^T pre, rhs = V^{-1} post
                spectral_sse::axpy(transposeColumns, stride, pre + v, S, [lhs, scale](int i, V_Real value) {
                    VEC_STOREU(lhs + i, VEC_MULT(value, scale));
                });
                const double* right = rhs;
                if (post != nullptr) {
                    spectral_sse::axpy(inverseColumns, stride, post + v, S, [rhs](int i, V_Real value) {
                        VEC_STOREU(rhs + i, value);
                    });
                } else if (tipStates[k] < S) {
                    right = inverseColumns + tipStates[k] * stride;
                } else { // missing state: V^{-1} 1, the row sums in the pad column of V^{-1}
                    const double* inverse = gEigenDecomposition->getInverseEigenVectorsPtr(info.eigenIndex);
                    for (int i = 0; i < S; ++i) {
                        rhs[i] = inverse[i * stride + S];
                    }
                }

                if (onePattern) {
                    adjointKernel(buffer, spectral_sse::RankOneView{lhs, right}, plan, info.eval, info,
                                  infoOffset, time, pairTmp);
                } else { // outer += lhs rhs^T, four rows per pass; for odd S the last vector reaches each row's pad
                    int l = 0;
                    for (; l + 4 <= S; l += 4) {
                        const V_Real a0 = VEC_SPLAT(lhs[l]);
                        const V_Real a1 = VEC_SPLAT(lhs[l + 1]);
                        const V_Real a2 = VEC_SPLAT(lhs[l + 2]);
                        const V_Real a3 = VEC_SPLAT(lhs[l + 3]);
                        double* row0 = outer + l * kAdjointStride;
                        double* row1 = row0 + kAdjointStride;
                        double* row2 = row1 + kAdjointStride;
                        double* row3 = row2 + kAdjointStride;
                        for (int j = 0; j < S; j += 2) {
                            const V_Real x = VEC_LOADU(right + j);
                            VEC_STOREU(row0 + j, VEC_MADD(a0, x, VEC_LOADU(row0 + j)));
                            VEC_STOREU(row1 + j, VEC_MADD(a1, x, VEC_LOADU(row1 + j)));
                            VEC_STOREU(row2 + j, VEC_MADD(a2, x, VEC_LOADU(row2 + j)));
                            VEC_STOREU(row3 + j, VEC_MADD(a3, x, VEC_LOADU(row3 + j)));
                        }
                    }
                    for (; l < S; ++l) {
                        const V_Real a = VEC_SPLAT(lhs[l]);
                        double* row = outer + l * kAdjointStride;
                        for (int j = 0; j < S; j += 2) {
                            VEC_STOREU(row + j, VEC_MADD(a, VEC_LOADU(right + j), VEC_LOADU(row + j)));
                        }
                    }
                }
            }

            if (!onePattern) {
                adjointKernel(buffer, spectral_sse::DenseView{outer, kAdjointStride}, plan, info.eval, info,
                              infoOffset, time, pairTmp);
            }
        }
    }
}

template <int T_PAD, int P_PAD>
void BeagleCPUSpectralSSEImpl<T_PAD, P_PAD>::forwardEigenBasis(double* y, double* u, const double* x,
                                                              const int state, const BranchEigenInfo& info,
                                                              const int catOffset) {
    const int S = kStateCount;
    const int stride = kStateCount + T_PAD;
    // rows of (V^{-1})^T are the columns of V^{-1}
    const double* inverseColumns = gEigenDecomposition->getBackwardsEigenVectorsPtr(info.eigenIndex);

    const double* v;
    if (x != nullptr) {
        spectral_sse::axpy(inverseColumns, stride, x, S, [u](int i, V_Real value) { VEC_STOREU(u + i, value); });
        v = u;
    } else if (state < S) {
        v = inverseColumns + state * stride;
    } else { // missing state: V^{-1} 1, the row sums in the pad column of V^{-1}
        const double* inverse = gEigenDecomposition->getInverseEigenVectorsPtr(info.eigenIndex);
        for (int i = 0; i < S; ++i) {
            u[i] = inverse[i * stride + S];
        }
        v = u;
    }

    const double* imag = (kFlags & BEAGLE_FLAG_EIGEN_COMPLEX) ? info.eval + S : nullptr;
    spectral_sse::expScale<false>(y, v, info.expat + catOffset, info.expatcosbt + catOffset,
                                  info.expatsinbt + catOffset, imag, S);
}

template <int T_PAD, int P_PAD>
void BeagleCPUSpectralSSEImpl<T_PAD, P_PAD>::backwardEigenBasis(double* y, double* u, const double* x,
                                                               const BranchEigenInfo& info, const int catOffset) {
    const int S = kStateCount;
    const int stride = kStateCount + T_PAD;
    // rows of V are the columns of V^T
    const double* transposeColumns = gEigenDecomposition->getEigenVectorsPtr(info.eigenIndex);

    spectral_sse::axpy(transposeColumns, stride, x, S, [u](int i, V_Real value) { VEC_STOREU(u + i, value); });

    const double* imag = (kFlags & BEAGLE_FLAG_EIGEN_COMPLEX) ? info.eval + S : nullptr;
    spectral_sse::expScale<true>(y, u, info.expat + catOffset, info.expatcosbt + catOffset,
                                 info.expatsinbt + catOffset, imag, S);
}

template <int T_PAD, int P_PAD>
void BeagleCPUSpectralSSEImpl<T_PAD, P_PAD>::preOrder(double* destP, SpectralPreOrder type,
                                                     const double* partials1, const int branchEigenIndex1,
                                                     const int* states2, const double* partials2,
                                                     const int branchEigenIndex2,
                                                     int startPattern, int endPattern, int currentPartition) {
    const int S = kStateCount;
    const int stride = kStateCount + T_PAD;
    const bool hasSibling = (states2 != nullptr || partials2 != nullptr);
    const bool hasBranch = (type != SpectralPreOrder::TopRoot);

    const BranchEigenInfo* info1 = hasBranch ? &gBranchEigenInfo[branchEigenIndex1] : nullptr;
    const BranchEigenInfo* info2 = hasSibling ? &gBranchEigenInfo[branchEigenIndex2] : nullptr;
    // rows of V^{-1} are the columns of V^{-T}; rows of V^T are the columns of V
    const double* transposeColumns1 =
            hasBranch ? gEigenDecomposition->getInverseEigenVectorsPtr(info1->eigenIndex) : nullptr;
    const double* columns2 =
            hasSibling ? gEigenDecomposition->getBackwardsInverseEigenVectorsPtr(info2->eigenIndex) : nullptr;

    double* u = simdTmp(currentPartition);
    double* y = u + kSimdTmpStride;
    double* w = y + kSimdTmpStride;

    const V_Real zero = VEC_SETZERO();

    for (int l = 0; l < kCategoryCount; l++) {
        const int catOffset = l * kPartialsPaddedStateCount;

        for (int k = startPattern; k < endPattern; k++) {
            const int v = l * kPartialsPaddedStateCount * kPaddedPatternCount + kPartialsPaddedStateCount * k;
            double* dest = destP + v;
            const double* parent = partials1 + v;

            if (hasSibling) { // y for the sibling's P x
                forwardEigenBasis(y, u, partials2 ? partials2 + v : nullptr, states2 ? states2[k] : 0,
                                  *info2, catOffset);
            }

            if (type == SpectralPreOrder::TopRoot) { // dest = max(P x, 0) * root prior
                spectral_sse::axpy(columns2, stride, y, S, [&](int i, V_Real value) {
                    spectral_sse::storeVector(dest, i, S,
                            VEC_MULT(VEC_MAX(zero, value), spectral_sse::loadVector(parent, i, S)));
                });
                continue;
            }

            const double* x = parent;
            if (hasSibling) {
                if (type == SpectralPreOrder::Bottom) { // P^T (max(P x, 0) * parent)
                    spectral_sse::axpy(columns2, stride, y, S, [&](int i, V_Real value) {
                        VEC_STOREU(w + i, VEC_MULT(VEC_MAX(zero, value), spectral_sse::loadVector(parent, i, S)));
                    });
                    x = w;
                } else { // max(P^T parent, 0) * max(P x, 0)
                    spectral_sse::axpy(columns2, stride, y, S, [w, zero](int i, V_Real value) {
                        VEC_STOREU(w + i, VEC_MAX(zero, value));
                    });
                }
            }

            backwardEigenBasis(y, u, x, *info1, catOffset);

            if (hasSibling && type == SpectralPreOrder::Top) {
                spectral_sse::axpy(transposeColumns1, stride, y, S, [&](int i, V_Real value) {
                    spectral_sse::storeVector(dest, i, S, VEC_MULT(VEC_MAX(zero, value), VEC_LOADU(w + i)));
                });
            } else {
                spectral_sse::axpy(transposeColumns1, stride, y, S, [&](int i, V_Real value) {
                    spectral_sse::storeVector(dest, i, S, VEC_MAX(zero, value));
                });
            }
        }
    }
}

template <int T_PAD, int P_PAD>
void BeagleCPUSpectralSSEImpl<T_PAD, P_PAD>::spectralPrePartialsPartials(double* destP, SpectralPreOrder type,
        const double* partials1, const int branchEigenIndex1,
        const double* partials2, const int branchEigenIndex2,
        int startPattern, int endPattern, int currentPartition) {
    preOrder(destP, type, partials1, branchEigenIndex1, nullptr, partials2, branchEigenIndex2,
             startPattern, endPattern, currentPartition);
}

template <int T_PAD, int P_PAD>
void BeagleCPUSpectralSSEImpl<T_PAD, P_PAD>::spectralPrePartialsStates(double* destP, SpectralPreOrder type,
        const double* partials1, const int branchEigenIndex1,
        const int* states2, const int branchEigenIndex2,
        int startPattern, int endPattern, int currentPartition) {
    preOrder(destP, type, partials1, branchEigenIndex1, states2, nullptr, branchEigenIndex2,
             startPattern, endPattern, currentPartition);
}

template <int T_PAD, int P_PAD>
void BeagleCPUSpectralSSEImpl<T_PAD, P_PAD>::calcDegree2PrePartials(double* destP,
        const double* partials1,
        const int branchEigenIndex1,
        int startPattern, int endPattern, int currentPartition) {
    if (branchEigenIndex1 < 0) { // top partials below the root: a copy
        Base::calcDegree2PrePartials(destP, partials1, branchEigenIndex1, startPattern, endPattern, currentPartition);
        return;
    }
    // the same for bottom and top partials: P^T of the parent's pre-order partials through branchEigenIndex1
    preOrder(destP, SpectralPreOrder::Bottom, partials1, branchEigenIndex1, nullptr, nullptr, -1,
             startPattern, endPattern, currentPartition);
}

template <int T_PAD, int P_PAD>
void BeagleCPUSpectralSSEImpl<T_PAD, P_PAD>::postOrder(double* destP,
                                                      const int* states1, const double* partials1,
                                                      const int branchEigenIndex1,
                                                      const int* states2, const double* partials2,
                                                      const int branchEigenIndex2,
                                                      const double* scaleFactors,
                                                      int startPattern, int endPattern, int currentPartition) {
    const int S = kStateCount;
    const int stride = kStateCount + T_PAD;
    const bool singleChild = (states2 == nullptr && partials2 == nullptr);

    const BranchEigenInfo& info1 = gBranchEigenInfo[branchEigenIndex1];
    const BranchEigenInfo& info2 = gBranchEigenInfo[singleChild ? branchEigenIndex1 : branchEigenIndex2];
    // rows of V^T are the columns of V
    const double* columns1 = gEigenDecomposition->getBackwardsInverseEigenVectorsPtr(info1.eigenIndex);
    const double* columns2 = gEigenDecomposition->getBackwardsInverseEigenVectorsPtr(info2.eigenIndex);

    double* u  = simdTmp(currentPartition);
    double* y  = u + kSimdTmpStride;
    double* z1 = y + kSimdTmpStride;

    const V_Real zero = VEC_SETZERO();

    for (int l = 0; l < kCategoryCount; l++) {
        const int catOffset = l * kPartialsPaddedStateCount;

        for (int k = startPattern; k < endPattern; k++) {
            const int v = l * kPartialsPaddedStateCount * kPaddedPatternCount + kPartialsPaddedStateCount * k;
            double* dest = destP + v;

            forwardEigenBasis(y, u, partials1 ? partials1 + v : nullptr, states1 ? states1[k] : 0, info1, catOffset);

            if (singleChild) {
                if (scaleFactors != nullptr) {
                    const V_Real oneOverScaleFactor = VEC_SPLAT(1.0 / scaleFactors[k]);
                    spectral_sse::axpy(columns1, stride, y, S, [&](int i, V_Real value) {
                        spectral_sse::storeVector(dest, i, S, VEC_MULT(VEC_MAX(zero, value), oneOverScaleFactor));
                    });
                } else {
                    spectral_sse::axpy(columns1, stride, y, S, [&](int i, V_Real value) {
                        spectral_sse::storeVector(dest, i, S, VEC_MAX(zero, value));
                    });
                }
                continue;
            }

            spectral_sse::axpy(columns1, stride, y, S, [z1, zero](int i, V_Real value) {
                VEC_STOREU(z1 + i, VEC_MAX(zero, value));
            });

            forwardEigenBasis(y, u, partials2 ? partials2 + v : nullptr, states2 ? states2[k] : 0, info2, catOffset);

            if (scaleFactors != nullptr) {
                const V_Real oneOverScaleFactor = VEC_SPLAT(1.0 / scaleFactors[k]);
                spectral_sse::axpy(columns2, stride, y, S, [&](int i, V_Real value) {
                    spectral_sse::storeVector(dest, i, S,
                            VEC_MULT(VEC_MULT(VEC_LOADU(z1 + i), VEC_MAX(zero, value)), oneOverScaleFactor));
                });
            } else {
                spectral_sse::axpy(columns2, stride, y, S, [&](int i, V_Real value) {
                    spectral_sse::storeVector(dest, i, S, VEC_MULT(VEC_LOADU(z1 + i), VEC_MAX(zero, value)));
                });
            }
        }
    }
}

template <int T_PAD, int P_PAD>
void BeagleCPUSpectralSSEImpl<T_PAD, P_PAD>::spectralPartialsPartials(double* destP,
        const double* partials1, const int branchEigenIndex1,
        const double* partials2, const int branchEigenIndex2,
        const double* scaleFactors,
        int startPattern, int endPattern, int currentPartition) {
    postOrder(destP, nullptr, partials1, branchEigenIndex1, nullptr, partials2, branchEigenIndex2,
              scaleFactors, startPattern, endPattern, currentPartition);
}

template <int T_PAD, int P_PAD>
void BeagleCPUSpectralSSEImpl<T_PAD, P_PAD>::spectralStatesPartials(double* destP,
        const int* states1, const int branchEigenIndex1,
        const double* partials2, const int branchEigenIndex2,
        const double* scaleFactors,
        int startPattern, int endPattern, int currentPartition) {
    postOrder(destP, states1, nullptr, branchEigenIndex1, nullptr, partials2, branchEigenIndex2,
              scaleFactors, startPattern, endPattern, currentPartition);
}

template <int T_PAD, int P_PAD>
void BeagleCPUSpectralSSEImpl<T_PAD, P_PAD>::spectralStatesStates(double* destP,
        const int* states1, const int branchEigenIndex1,
        const int* states2, const int branchEigenIndex2,
        const double* scaleFactors,
        int startPattern, int endPattern, int currentPartition) {
    postOrder(destP, states1, nullptr, branchEigenIndex1, states2, nullptr, branchEigenIndex2,
              scaleFactors, startPattern, endPattern, currentPartition);
}

template <int T_PAD, int P_PAD>
void BeagleCPUSpectralSSEImpl<T_PAD, P_PAD>::spectralDegree2Partials(double* destP,
        const int* states1, const double* partials1,
        const int branchEigenIndex1,
        const double* scaleFactors,
        int startPattern, int endPattern, int currentPartition) {
    postOrder(destP, states1, states1 ? nullptr : partials1, branchEigenIndex1, nullptr, nullptr, -1,
              scaleFactors, startPattern, endPattern, currentPartition);
}

///////////////////////////////////////////////////////////////////////////////
// BeagleCPUSpectralSSEImplFactory public methods

inline BeagleImpl* BeagleCPUSpectralSSEImplFactory::createImpl(int tipCount,
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

    if (!CPUSupportsSSE()) {
        return NULL;
    }

    BeagleImpl* impl = new BeagleCPUSpectralSSEImpl<T_PAD_DEFAULT, P_PAD_DEFAULT>();

    try {
        if (impl->createInstance(tipCount, partialsBufferCount, compactBufferCount, stateCount,
                                 patternCount, eigenBufferCount, matrixBufferCount,
                                 categoryCount, scaleBufferCount, resourceNumber,
                                 pluginResourceNumber,
                                 preferenceFlags, requirementFlags) == 0)
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

inline const char* BeagleCPUSpectralSSEImplFactory::getName() {
    return "CPU-Spectral-SSE-Double";
}

inline const long BeagleCPUSpectralSSEImplFactory::getFlags() {
    return BEAGLE_FLAG_COMPUTATION_SYNCH |
           BEAGLE_FLAG_SCALING_MANUAL | BEAGLE_FLAG_SCALING_ALWAYS | BEAGLE_FLAG_SCALING_AUTO |
           BEAGLE_FLAG_THREADING_NONE | BEAGLE_FLAG_THREADING_CPP |
           BEAGLE_FLAG_PROCESSOR_CPU |
           BEAGLE_FLAG_VECTOR_SSE |
           BEAGLE_FLAG_PRECISION_DOUBLE |
           BEAGLE_FLAG_SCALERS_LOG | BEAGLE_FLAG_SCALERS_RAW |
           BEAGLE_FLAG_EIGEN_COMPLEX | BEAGLE_FLAG_EIGEN_REAL |
           BEAGLE_FLAG_INVEVEC_STANDARD |
           BEAGLE_FLAG_PREORDER_TRANSPOSE_MANUAL | BEAGLE_FLAG_PREORDER_TRANSPOSE_AUTO |
           BEAGLE_FLAG_SPECTRAL_REPRESENTATION |
           BEAGLE_FLAG_FRAMEWORK_CPU;
}

} // namespace cpu
} // namespace beagle

#endif // BEAGLE_CPU_SPECTRAL_SSE_IMPL_HPP
