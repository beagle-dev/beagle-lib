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

#include <iostream>

#include "libhmsbeagle/beagle.h"
#include "libhmsbeagle/CPU/BeagleCPUSpectralSSEImpl.h"
#include "libhmsbeagle/CPU/SSEDefinitions.h"

namespace beagle {
namespace cpu {

namespace spectral_sse {

// Output vectors (two states each) kept in registers by one pass over the rows
constexpr int kBlockVectors = 8;

/*
 * out = M^T x for an S x stride row-major M: for each block of outputs, acc += row_j * x_j over the rows j < S.
 * Rows are not 16-byte aligned (stride S + T_PAD). For odd S the second lane of the last vector reads the pad
 * column, which holds finite row sums; epilogue(i, value) receives the vector for outputs i and i + 1 and must not
 * store lane i + 1 = S.
 */
template <int NV, typename Epilogue>
inline void axpyBlock(const double* __restrict rows, const int stride, const double* __restrict x, const int S,
                      const int i0, Epilogue& epilogue) {
    V_Real acc[NV];
    for (int v = 0; v < NV; ++v) {
        acc[v] = VEC_SETZERO();
    }
    const double* row = rows + i0;
    for (int j = 0; j < S; ++j, row += stride) {
        const V_Real xj = VEC_SPLAT(x[j]);
        for (int v = 0; v < NV; ++v) {
            acc[v] = VEC_MADD(VEC_LOADU(row + 2 * v), xj, acc[v]);
        }
    }
    for (int v = 0; v < NV; ++v) {
        epilogue(i0 + 2 * v, acc[v]);
    }
}

template <typename Epilogue>
inline void axpy(const double* __restrict rows, const int stride, const double* __restrict x, const int S,
                 Epilogue epilogue) {
    const int vectors = (S + 1) / 2;
    int v0 = 0;
    for (; v0 + kBlockVectors <= vectors; v0 += kBlockVectors) {
        axpyBlock<kBlockVectors>(rows, stride, x, S, 2 * v0, epilogue);
    }
    switch (vectors - v0) {
        case 1: axpyBlock<1>(rows, stride, x, S, 2 * v0, epilogue); break;
        case 2: axpyBlock<2>(rows, stride, x, S, 2 * v0, epilogue); break;
        case 3: axpyBlock<3>(rows, stride, x, S, 2 * v0, epilogue); break;
        case 4: axpyBlock<4>(rows, stride, x, S, 2 * v0, epilogue); break;
        case 5: axpyBlock<5>(rows, stride, x, S, 2 * v0, epilogue); break;
        case 6: axpyBlock<6>(rows, stride, x, S, 2 * v0, epilogue); break;
        case 7: axpyBlock<7>(rows, stride, x, S, 2 * v0, epilogue); break;
        default: break;
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

    return returnCode;
}

template <int T_PAD, int P_PAD>
int BeagleCPUSpectralSSEImpl<T_PAD, P_PAD>::setCPUThreadCount(int threadCount) {
    const int returnCode = Base::setCPUThreadCount(threadCount);
    gSimdTmp.assign(3 * kSimdTmpStride * kPartitionCount, 0.0);
    return returnCode;
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
           BEAGLE_FLAG_INVEVEC_STANDARD | BEAGLE_FLAG_INVEVEC_TRANSPOSED |
           BEAGLE_FLAG_PREORDER_TRANSPOSE_MANUAL | BEAGLE_FLAG_PREORDER_TRANSPOSE_AUTO |
           BEAGLE_FLAG_SPECTRAL_REPRESENTATION |
           BEAGLE_FLAG_FRAMEWORK_CPU;
}

} // namespace cpu
} // namespace beagle

#endif // BEAGLE_CPU_SPECTRAL_SSE_IMPL_HPP
