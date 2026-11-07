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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "libhmsbeagle/beagle.h"
#include "libhmsbeagle/CPU/BeagleCPUSpectralSSEImpl.h"
#include "libhmsbeagle/CPU/SSEDefinitions.h"
#include "libhmsbeagle/CPU/SSEKernels.h"

namespace beagle {
namespace cpu {

namespace spectral_sse {

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
const long long BeagleCPUSpectralSSEImpl<T_PAD, P_PAD>::getFlags() {
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
                                                          long long preferenceFlags,
                                                          long long requirementFlags) {

    const int returnCode = Base::createInstance(tipCount, partialsBufferCount, compactBufferCount, stateCount,
                                                patternCount, eigenDecompositionCount, matrixCount, categoryCount,
                                                scaleBufferCount, resourceNumber, pluginResourceNumber,
                                                preferenceFlags, requirementFlags);

    kSimdTmpStride = (kStateCount + 3) & ~1; // at least S + 1, and even
    gSimdTmp.assign(3 * kSimdTmpStride * kPartitionCount, 0.0);

    if (const char* cache = getenv("BEAGLE_SPECTRAL_CACHE")) {
        kUseSpectralCache = (atoi(cache) != 0);
    }
    if (const char* verify = getenv("BEAGLE_SPECTRAL_CACHE_VERIFY")) {
        kVerifySpectralCache = (atoi(verify) != 0);
    }
    if (const char* verbose = getenv("BEAGLE_SPECTRAL_CACHE_VERBOSE")) {
        kVerboseSpectralCache = (atoi(verbose) != 0);
    }
    kCacheBufferCount = -1; // allocated by the first use
    return returnCode;
}

template <int T_PAD, int P_PAD>
int BeagleCPUSpectralSSEImpl<T_PAD, P_PAD>::setPatternPartitions(int partitionCount,
                                                                const int* inPatternPartitions) {
    const int returnCode = Base::setPatternPartitions(partitionCount, inPatternPartitions);
    gSimdTmp.assign(3 * kSimdTmpStride * kPartitionCount, 0.0);
    kCacheBufferCount = -1; // the tips may have been reordered: empty the caches
    return returnCode;
}

///////////////////////////////////////////////////////////////////////////////
// Message and push-down caches

template <int T_PAD, int P_PAD>
void BeagleCPUSpectralSSEImpl<T_PAD, P_PAD>::ensureCacheSizes() {
    const int partitions = std::max(kPartitionCount, 1);
    auto sizesMatch = [&]() {
        return kCacheBufferCount.load(std::memory_order_acquire) == kBufferCount &&
               kCacheMatrixCount == kMatrixCount && kCacheEigenCount == kEigenDecompCount &&
               kCachePartitionCount == partitions;
    };
    if (sizesMatch()) {
        return;
    }
    // threads of a partitioned update may arrive together
    std::lock_guard<std::mutex> lock(gCacheMutex);
    if (sizesMatch()) {
        return;
    }
    kCacheMatrixCount = kMatrixCount;
    kCacheEigenCount = kEigenDecompCount;
    kCachePartitionCount = partitions;

    const uint64_t generation = ++gCacheCounter;
    gPartialsGeneration.assign(size_t(kBufferCount) * partitions, generation);
    gMatrixGeneration.assign(kMatrixCount, generation);
    gEigenGeneration.assign(kEigenDecompCount, generation);
    resetSlots(gMessageCache, kBufferCount);
    resetSlots(gPushDownCache, kBufferCount);
    gMessageStamp.assign(size_t(kBufferCount) * partitions, CacheStamp());
    gPushDownStamp.assign(size_t(kBufferCount) * partitions, CacheStamp());
    gOpContext.assign(partitions, OpContext());
    kCacheBufferCount.store(kBufferCount, std::memory_order_release); // last: publishes the sizes
}

template <int T_PAD, int P_PAD>
void BeagleCPUSpectralSSEImpl<T_PAD, P_PAD>::bumpPartials(int buffer, int partition) {
    if (!kUseSpectralCache || buffer < 0 || buffer >= kCacheBufferCount) {
        return;
    }
    const uint64_t generation = ++gCacheCounter;
    if (partition < 0) {
        for (int p = 0; p < kCachePartitionCount; ++p) {
            gPartialsGeneration[size_t(buffer) * kCachePartitionCount + p] = generation;
        }
    } else {
        gPartialsGeneration[size_t(buffer) * kCachePartitionCount + partition] = generation;
    }
}

template <int T_PAD, int P_PAD>
typename BeagleCPUSpectralSSEImpl<T_PAD, P_PAD>::CacheStamp
BeagleCPUSpectralSSEImpl<T_PAD, P_PAD>::makeStamp(int buffer, int partition, int matrixIndex, int eigenIndex) const {
    CacheStamp stamp;
    stamp.partials = gPartialsGeneration[size_t(buffer) * kCachePartitionCount + partition];
    stamp.matrix = gMatrixGeneration[matrixIndex];
    stamp.eigen = gEigenGeneration[eigenIndex];
    stamp.matrixIndex = matrixIndex;
    stamp.eigenIndex = eigenIndex;
    return stamp;
}

template <int T_PAD, int P_PAD>
bool BeagleCPUSpectralSSEImpl<T_PAD, P_PAD>::cacheValid(const CacheStamp& stamp, int buffer, int partition,
                                                       int matrixIndex, int eigenIndex) const {
    return stamp.partials != 0 &&
           stamp.partials == gPartialsGeneration[size_t(buffer) * kCachePartitionCount + partition] &&
           stamp.matrixIndex == matrixIndex && stamp.matrix == gMatrixGeneration[matrixIndex] &&
           stamp.eigenIndex == eigenIndex && stamp.eigen == gEigenGeneration[eigenIndex];
}

template <int T_PAD, int P_PAD>
void BeagleCPUSpectralSSEImpl<T_PAD, P_PAD>::resetSlots(CacheSlots& cache, int bufferCount) {
    cache.pointers.reset(new std::atomic<double*>[bufferCount]);
    for (int b = 0; b < bufferCount; ++b) {
        cache.pointers[b].store(nullptr, std::memory_order_relaxed);
    }
    cache.storage.clear();
    cache.storage.resize(bufferCount);
}

template <int T_PAD, int P_PAD>
double* BeagleCPUSpectralSSEImpl<T_PAD, P_PAD>::cacheSlot(CacheSlots& cache, int buffer) {
    double* slot = cache.pointers[buffer].load(std::memory_order_acquire);
    if (slot == nullptr) {
        std::lock_guard<std::mutex> lock(gCacheMutex);
        slot = cache.pointers[buffer].load(std::memory_order_relaxed);
        if (slot == nullptr) {
            cache.storage[buffer].reset(new double[kPartialsSize]());
            slot = cache.storage[buffer].get();
            cache.pointers[buffer].store(slot, std::memory_order_release);
        }
    }
    return slot;
}

template <int T_PAD, int P_PAD>
int BeagleCPUSpectralSSEImpl<T_PAD, P_PAD>::setTipStates(int tipIndex, const int* inStates) {
    const int returnCode = Base::setTipStates(tipIndex, inStates);
    if (kUseSpectralCache) {
        ensureCacheSizes();
        bumpPartials(tipIndex, -1);
    }
    return returnCode;
}

template <int T_PAD, int P_PAD>
int BeagleCPUSpectralSSEImpl<T_PAD, P_PAD>::setTipPartials(int tipIndex, const double* inPartials) {
    const int returnCode = Base::setTipPartials(tipIndex, inPartials);
    if (kUseSpectralCache) {
        ensureCacheSizes();
        bumpPartials(tipIndex, -1);
    }
    return returnCode;
}

template <int T_PAD, int P_PAD>
int BeagleCPUSpectralSSEImpl<T_PAD, P_PAD>::setPartials(int bufferIndex, const double* inPartials) {
    const int returnCode = Base::setPartials(bufferIndex, inPartials);
    if (kUseSpectralCache) {
        ensureCacheSizes();
        bumpPartials(bufferIndex, -1);
    }
    return returnCode;
}

template <int T_PAD, int P_PAD>
int BeagleCPUSpectralSSEImpl<T_PAD, P_PAD>::setRootPrePartials(const int* bufferIndices,
                                                              const int* stateFrequenciesIndices, int count) {
    const int returnCode = Base::setRootPrePartials(bufferIndices, stateFrequenciesIndices, count);
    if (kUseSpectralCache) {
        ensureCacheSizes();
        for (int i = 0; i < count; ++i) {
            bumpPartials(bufferIndices[i], -1);
        }
    }
    return returnCode;
}

template <int T_PAD, int P_PAD>
int BeagleCPUSpectralSSEImpl<T_PAD, P_PAD>::setEigenDecomposition(int eigenIndex,
                                                                 const double* inEigenVectors,
                                                                 const double* inInverseEigenVectors,
                                                                 const double* inEigenValues) {
    const int returnCode = Base::setEigenDecomposition(eigenIndex, inEigenVectors, inInverseEigenVectors,
                                                       inEigenValues);
    if (kUseSpectralCache) {
        ensureCacheSizes();
        if (eigenIndex >= 0 && eigenIndex < kCacheEigenCount) {
            gEigenGeneration[eigenIndex] = ++gCacheCounter;
        }
    }
    return returnCode;
}

template <int T_PAD, int P_PAD>
int BeagleCPUSpectralSSEImpl<T_PAD, P_PAD>::updateTransitionMatrices(int eigenIndex,
                                                                    const int* probabilityIndices,
                                                                    const int* firstDerivativeIndices,
                                                                    const int* secondDerivativeIndices,
                                                                    const double* edgeLengths,
                                                                    int count) {
    const int returnCode = Base::updateTransitionMatrices(eigenIndex, probabilityIndices, firstDerivativeIndices,
                                                          secondDerivativeIndices, edgeLengths, count);
    if (kUseSpectralCache) {
        ensureCacheSizes();
        for (int i = 0; i < count; ++i) {
            const int matrix = probabilityIndices[i];
            if (matrix >= 0 && matrix < kCacheMatrixCount) {
                gMatrixGeneration[matrix] = ++gCacheCounter;
            }
        }
    }
    return returnCode;
}

template <int T_PAD, int P_PAD>
int BeagleCPUSpectralSSEImpl<T_PAD, P_PAD>::upPartials(bool byPartition,
                                                      const int* operations,
                                                      int count,
                                                      int cumulativeScaleIndex,
                                                      BeaglePartialsType partialsType) {
    if (!kUseSpectralCache) {
        return Base::upPartials(byPartition, operations, count, cumulativeScaleIndex, partialsType);
    }
    ensureCacheSizes();
    const int numOps = byPartition ? BEAGLE_PARTITION_OP_COUNT : BEAGLE_OP_COUNT;
    for (int op = 0; op < count; ++op) {
        const int* operation = operations + op * numOps;
        const int partition = byPartition ? operation[7] : 0;
        OpContext& context = gOpContext[partition];
        context.index1 = operation[3];
        context.matrix1 = operation[4];
        context.index2 = operation[5];
        context.matrix2 = operation[6];
        const int returnCode = Base::upPartials(byPartition, operation, 1, cumulativeScaleIndex, partialsType);
        context = OpContext();
        bumpPartials(operation[0], byPartition ? partition : -1);
        if (returnCode != BEAGLE_SUCCESS) {
            return returnCode;
        }
    }
    return BEAGLE_SUCCESS;
}

template <int T_PAD, int P_PAD>
int BeagleCPUSpectralSSEImpl<T_PAD, P_PAD>::upPrePartials(bool byPartition,
                                                         const int* operations,
                                                         int count,
                                                         int cumulativeScaleIndex,
                                                         BeaglePartialsType partialsType) {
    if (!kUseSpectralCache) {
        return Base::upPrePartials(byPartition, operations, count, cumulativeScaleIndex, partialsType);
    }
    ensureCacheSizes();
    const int numOps = byPartition ? BEAGLE_PARTITION_OP_COUNT : BEAGLE_OP_COUNT;
    for (int op = 0; op < count; ++op) {
        const int* operation = operations + op * numOps;
        const int partition = byPartition ? operation[7] : 0;
        OpContext& context = gOpContext[partition];
        context.index1 = operation[3]; // the parent's pre-order partials
        context.matrix1 = operation[4];
        context.index2 = operation[5]; // the sibling's post-order partials
        context.matrix2 = operation[6];
        const int returnCode = Base::upPrePartials(byPartition, operation, 1, cumulativeScaleIndex, partialsType);
        context = OpContext();
        bumpPartials(operation[0], byPartition ? partition : -1);
        if (returnCode != BEAGLE_SUCCESS) {
            return returnCode;
        }
    }
    if (kVerboseSpectralCache) {
        fprintf(stderr, "spectral cache: messages %lld hit %lld miss, push-downs %lld hit %lld miss, "
                        "verify mismatches %lld\n",
                gMessageHits.load(), gMessageMisses.load(), gPushDownHits.load(), gPushDownMisses.load(),
                gCacheMismatches.load());
    }
    return BEAGLE_SUCCESS;
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
        simd::axpy(inverseColumns, stride, x, S, [u](int i, V_Real value) { VEC_STOREU(u + i, value); });
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

    simd::axpy(transposeColumns, stride, x, S, [u](int i, V_Real value) { VEC_STOREU(u + i, value); });

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
    const int p = currentPartition;
    const OpContext context = kUseSpectralCache ? gOpContext[p] : OpContext();

    // The sibling's message max(0, P x) from the cache, when this operation's sibling is known
    const int siblingBuffer = (kUseSpectralCache && hasSibling && context.index2 >= 0 &&
                               context.matrix2 == branchEigenIndex2) ? context.index2 : -1;
    bool messageHit = false;
    double* messageSlot = nullptr;
    if (siblingBuffer >= 0) {
        messageHit = cacheValid(gMessageStamp[size_t(siblingBuffer) * kCachePartitionCount + p], siblingBuffer, p,
                                branchEigenIndex2, info2->eigenIndex);
        messageSlot = cacheSlot(gMessageCache, siblingBuffer);
    }
    const bool useMessage = messageHit && !kVerifySpectralCache;

    // The parent's push-down max(0, P^T top) from the cache (TOP partials with a sibling)
    const int parentBuffer = (kUseSpectralCache && type == SpectralPreOrder::Top && hasSibling &&
                              context.index1 >= 0 && context.matrix1 == branchEigenIndex1) ? context.index1 : -1;
    bool pushHit = false;
    double* pushSlot = nullptr;
    if (parentBuffer >= 0) {
        pushHit = cacheValid(gPushDownStamp[size_t(parentBuffer) * kCachePartitionCount + p], parentBuffer, p,
                             branchEigenIndex1, info1->eigenIndex);
        pushSlot = cacheSlot(gPushDownCache, parentBuffer);
    }
    const bool usePush = pushHit && !kVerifySpectralCache;

    long long mismatches = 0;

    for (int l = 0; l < kCategoryCount; l++) {
        const int catOffset = l * kPartialsPaddedStateCount;

        for (int k = startPattern; k < endPattern; k++) {
            const int v = l * kPartialsPaddedStateCount * kPaddedPatternCount + kPartialsPaddedStateCount * k;
            double* dest = destP + v;
            const double* parent = partials1 + v;

            if (hasSibling) { // w = max(P x, 0) for the sibling
                if (useMessage) {
                    std::memcpy(w, messageSlot + v, S * sizeof(double));
                } else {
                    forwardEigenBasis(y, u, partials2 ? partials2 + v : nullptr, states2 ? states2[k] : 0,
                                      *info2, catOffset);
                    simd::axpy(columns2, stride, y, S, [w, zero](int i, V_Real value) {
                        VEC_STOREU(w + i, VEC_MAX(zero, value));
                    });
                    if (messageHit) { // verifying
                        mismatches += (std::memcmp(w, messageSlot + v, S * sizeof(double)) != 0);
                    } else if (messageSlot != nullptr) {
                        std::memcpy(messageSlot + v, w, S * sizeof(double));
                    }
                }
            }

            if (type == SpectralPreOrder::TopRoot) { // dest = max(P x, 0) * root prior
                for (int i = 0; i < S; i += 2) {
                    simd::storeVector(dest, i, S, VEC_MULT(VEC_LOADU(w + i), simd::loadVector(parent, i, S)));
                }
                continue;
            }

            const double* x = parent;
            if (hasSibling && type == SpectralPreOrder::Bottom) { // P^T (max(P x, 0) * parent)
                for (int i = 0; i < S; i += 2) {
                    VEC_STOREU(w + i, VEC_MULT(VEC_LOADU(w + i), simd::loadVector(parent, i, S)));
                }
                x = w;
            }

            if (hasSibling && type == SpectralPreOrder::Top) { // max(P^T parent, 0) * max(P x, 0)
                if (usePush) {
                    const double* pushDown = pushSlot + v;
                    for (int i = 0; i < S; i += 2) {
                        simd::storeVector(dest, i, S, VEC_MULT(simd::loadVector(pushDown, i, S), VEC_LOADU(w + i)));
                    }
                } else {
                    backwardEigenBasis(y, u, x, *info1, catOffset);
                    double* pushOut = pushHit ? u : pushSlot; // verifying: u is free once y is formed
                    simd::axpy(transposeColumns1, stride, y, S, [&](int i, V_Real value) {
                        const V_Real pushDown = VEC_MAX(zero, value);
                        if (pushOut != nullptr) {
                            simd::storeVector(pushOut + (pushHit ? 0 : v), i, S, pushDown);
                        }
                        simd::storeVector(dest, i, S, VEC_MULT(pushDown, VEC_LOADU(w + i)));
                    });
                    if (pushHit) {
                        mismatches += (std::memcmp(u, pushSlot + v, S * sizeof(double)) != 0);
                    }
                }
            } else {
                backwardEigenBasis(y, u, x, *info1, catOffset);
                simd::axpy(transposeColumns1, stride, y, S, [&](int i, V_Real value) {
                    simd::storeVector(dest, i, S, VEC_MAX(zero, value));
                });
            }
        }
    }

    if (siblingBuffer >= 0) {
        if (messageHit) {
            ++gMessageHits;
        } else {
            ++gMessageMisses;
            gMessageStamp[size_t(siblingBuffer) * kCachePartitionCount + p] =
                    makeStamp(siblingBuffer, p, branchEigenIndex2, info2->eigenIndex);
        }
    }
    if (parentBuffer >= 0) {
        if (pushHit) {
            ++gPushDownHits;
        } else {
            ++gPushDownMisses;
            gPushDownStamp[size_t(parentBuffer) * kCachePartitionCount + p] =
                    makeStamp(parentBuffer, p, branchEigenIndex1, info1->eigenIndex);
        }
    }
    if (mismatches != 0) {
        gCacheMismatches += mismatches;
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

    // Each child's message max(0, P x) goes to the cache, keyed by its buffer; the kernel's children may be the
    // operation's in either order (a tip first), so they are matched by matrix index
    const int p = currentPartition;
    int buffer1 = -1;
    int buffer2 = -1;
    if (kUseSpectralCache && !singleChild) {
        const OpContext& context = gOpContext[p];
        if (context.matrix1 != context.matrix2) {
            if (branchEigenIndex1 == context.matrix1 && branchEigenIndex2 == context.matrix2) {
                buffer1 = context.index1;
                buffer2 = context.index2;
            } else if (branchEigenIndex1 == context.matrix2 && branchEigenIndex2 == context.matrix1) {
                buffer1 = context.index2;
                buffer2 = context.index1;
            }
        }
    }
    double* message1 = (buffer1 >= 0) ? cacheSlot(gMessageCache, buffer1) : nullptr;
    double* message2 = (buffer2 >= 0) ? cacheSlot(gMessageCache, buffer2) : nullptr;

    for (int l = 0; l < kCategoryCount; l++) {
        const int catOffset = l * kPartialsPaddedStateCount;

        for (int k = startPattern; k < endPattern; k++) {
            const int v = l * kPartialsPaddedStateCount * kPaddedPatternCount + kPartialsPaddedStateCount * k;
            double* dest = destP + v;

            forwardEigenBasis(y, u, partials1 ? partials1 + v : nullptr, states1 ? states1[k] : 0, info1, catOffset);

            if (singleChild) {
                if (scaleFactors != nullptr) {
                    const V_Real oneOverScaleFactor = VEC_SPLAT(1.0 / scaleFactors[k]);
                    simd::axpy(columns1, stride, y, S, [&](int i, V_Real value) {
                        simd::storeVector(dest, i, S, VEC_MULT(VEC_MAX(zero, value), oneOverScaleFactor));
                    });
                } else {
                    simd::axpy(columns1, stride, y, S, [&](int i, V_Real value) {
                        simd::storeVector(dest, i, S, VEC_MAX(zero, value));
                    });
                }
                continue;
            }

            simd::axpy(columns1, stride, y, S, [z1, zero](int i, V_Real value) {
                VEC_STOREU(z1 + i, VEC_MAX(zero, value));
            });
            if (message1 != nullptr) {
                std::memcpy(message1 + v, z1, S * sizeof(double));
            }

            forwardEigenBasis(y, u, partials2 ? partials2 + v : nullptr, states2 ? states2[k] : 0, info2, catOffset);

            double* m2 = (message2 != nullptr) ? message2 + v : nullptr;
            if (scaleFactors != nullptr) {
                const V_Real oneOverScaleFactor = VEC_SPLAT(1.0 / scaleFactors[k]);
                simd::axpy(columns2, stride, y, S, [&](int i, V_Real value) {
                    const V_Real message = VEC_MAX(zero, value);
                    if (m2 != nullptr) {
                        simd::storeVector(m2, i, S, message);
                    }
                    simd::storeVector(dest, i, S,
                            VEC_MULT(VEC_MULT(VEC_LOADU(z1 + i), message), oneOverScaleFactor));
                });
            } else {
                simd::axpy(columns2, stride, y, S, [&](int i, V_Real value) {
                    const V_Real message = VEC_MAX(zero, value);
                    if (m2 != nullptr) {
                        simd::storeVector(m2, i, S, message);
                    }
                    simd::storeVector(dest, i, S, VEC_MULT(VEC_LOADU(z1 + i), message));
                });
            }
        }
    }

    if (buffer1 >= 0) {
        gMessageStamp[size_t(buffer1) * kCachePartitionCount + p] = makeStamp(buffer1, p, branchEigenIndex1,
                                                                              info1.eigenIndex);
    }
    if (buffer2 >= 0) {
        gMessageStamp[size_t(buffer2) * kCachePartitionCount + p] = makeStamp(buffer2, p, branchEigenIndex2,
                                                                              info2.eigenIndex);
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
                                                               long long preferenceFlags,
                                                               long long requirementFlags,
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

inline const long long BeagleCPUSpectralSSEImplFactory::getFlags() {
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
