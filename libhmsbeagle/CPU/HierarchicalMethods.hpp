/*
 *  HierarchicalMethods.hpp
 *  BEAGLE
 *
 * Copyright 2026 Phylogenetic Likelihood Working Group
 *
 * This file is part of BEAGLE.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */

#ifndef HIERARCHICALMETHODS_HPP_
#define HIERARCHICALMETHODS_HPP_

#include <cmath>
#include <vector>

#include "libhmsbeagle/beagle.h"
#include "libhmsbeagle/CPU/HierarchicalMethods.h"

namespace beagle {
namespace cpu {

inline bool hierarchicalIndex(double value, int lower, int upper, int& out) {
    if (!std::isfinite(value) || value != std::floor(value) || value < lower || value >= upper)
        return false;
    out = (int) value;
    return true;
}

inline bool hierarchicalTriples(const double* in,
                                int stateCount,
                                std::vector<int>& ops) {
    const int internalCount = stateCount - 1;
    const int nodeCount = 2 * stateCount - 1;
    ops.resize(3 * internalCount);
    for (int t = 0; t < internalCount; t++) {
        int p, l, r;
        if (!hierarchicalIndex(in[3 * t + 0], stateCount, nodeCount, p) ||
            !hierarchicalIndex(in[3 * t + 1], 0, nodeCount, l) ||
            !hierarchicalIndex(in[3 * t + 2], 0, nodeCount, r) ||
            l == r || l == p || r == p)
            return false;
        ops[3 * t + 0] = p;
        ops[3 * t + 1] = l;
        ops[3 * t + 2] = r;
    }
    return true;
}

inline bool hierarchicalCheckOrders(const std::vector<int>& postOps,
                                    const std::vector<int>& preOps,
                                    int stateCount) {
    const int internalCount = stateCount - 1;
    const int nodeCount = 2 * stateCount - 1;

    std::vector<int> parentOf(nodeCount, -1);
    std::vector<int> tripleOf(nodeCount, -1);
    std::vector<bool> done(nodeCount, false);

    for (int t = 0; t < internalCount; t++) {
        const int p = postOps[3 * t];
        if (tripleOf[p] >= 0 || parentOf[p] >= 0)
            return false;
        tripleOf[p] = t;
        for (int c = 1; c <= 2; c++) {
            const int child = postOps[3 * t + c];
            if (parentOf[child] >= 0)
                return false;
            if (child >= stateCount && tripleOf[child] < 0)
                return false;
            parentOf[child] = p;
        }
    }

    const int root = postOps[3 * (internalCount - 1)];
    for (int m = 0; m < nodeCount; m++) {
        if ((m == root) != (parentOf[m] < 0))
            return false;
    }

    if (preOps[0] != root)
        return false;
    for (int t = 0; t < internalCount; t++) {
        const int p = preOps[3 * t];
        const int u = tripleOf[p];
        if (u < 0 || done[p])
            return false;
        if (p != root && !done[parentOf[p]])
            return false;
        if (preOps[3 * t + 1] != postOps[3 * u + 1] || preOps[3 * t + 2] != postOps[3 * u + 2])
            return false;
        done[p] = true;
    }
    return true;
}

template <typename REALTYPE>
int unpackHierarchicalBundle(const double* bundle,
                             int stateCount,
                             int categoryCount,
                             HierarchicalSlot<REALTYPE>& slot) {
    const int internalCount = stateCount - 1;
    const int blockLength = hierarchicalBlockLength(stateCount);

    int header;
    if (!hierarchicalIndex(bundle[0], 0, 1 << 30, header) ||
        header / kHierarchicalBundleVersionShift != kHierarchicalBundleVersion ||
        header % kHierarchicalBundleVersionShift != internalCount)
        return BEAGLE_ERROR_OUT_OF_RANGE;

    std::vector<int> postOps, preOps;
    if (!hierarchicalTriples(bundle + 1, stateCount, postOps) ||
        !hierarchicalTriples(bundle + 1 + 3 * internalCount, stateCount, preOps) ||
        !hierarchicalCheckOrders(postOps, preOps, stateCount))
        return BEAGLE_ERROR_OUT_OF_RANGE;

    const double* blocks = bundle + 1 + 6 * internalCount;
    for (int l = 0; l < categoryCount; l++) {
        const double* block = blocks + l * blockLength;
        for (int m = 0; m < internalCount; m++) {
            if (!std::isfinite(block[m]) || block[m] < 0.0)
                return BEAGLE_ERROR_OUT_OF_RANGE;
        }
        for (int x = 0; x < stateCount; x++) {
            const double prZ0 = block[internalCount + x];
            if (!std::isfinite(prZ0) || prZ0 < 0.0 || prZ0 > 1.0)
                return BEAGLE_ERROR_OUT_OF_RANGE;
        }
    }

    slot.postOps.swap(postOps);
    slot.preOps.swap(preOps);
    slot.values.resize(categoryCount * blockLength);
    for (int i = 0; i < categoryCount * blockLength; i++)
        slot.values[i] = (REALTYPE) blocks[i];
    slot.set = true;
    return BEAGLE_SUCCESS;
}

template <typename REALTYPE>
void hierarchicalTransitionMatrices(const HierarchicalSlot<REALTYPE>& slot,
                                    const REALTYPE* pi,
                                    int stateCount,
                                    int categoryCount,
                                    double* outMatrices) {
    const int internalCount = stateCount - 1;
    const int nodeCount = 2 * stateCount - 1;
    const int blockLength = hierarchicalBlockLength(stateCount);

    std::vector<int> parentOf(nodeCount, -1);
    std::vector<int> depth(nodeCount, 0);
    for (int t = 0; t < internalCount; t++) {
        parentOf[slot.preOps[3 * t + 1]] = slot.preOps[3 * t];
        parentOf[slot.preOps[3 * t + 2]] = slot.preOps[3 * t];
    }
    for (int t = 0; t < internalCount; t++) {
        const int p = slot.preOps[3 * t];
        depth[slot.preOps[3 * t + 1]] = depth[p] + 1;
        depth[slot.preOps[3 * t + 2]] = depth[p] + 1;
    }

    std::vector<int> mrca(stateCount * stateCount);
    for (int x = 0; x < stateCount; x++) {
        for (int y = 0; y < stateCount; y++) {
            int a = x, b = y;
            if (a == b) {
                mrca[x * stateCount + y] = a;
                continue;
            }
            while (depth[a] > depth[b]) a = parentOf[a];
            while (depth[b] > depth[a]) b = parentOf[b];
            while (a != b) {
                a = parentOf[a];
                b = parentOf[b];
            }
            mrca[x * stateCount + y] = a;
        }
    }

    const int root = slot.preOps[0];
    std::vector<double> sumH(nodeCount, 0.0);
    for (int l = 0; l < categoryCount; l++) {
        const REALTYPE* H = slot.values.data() + l * blockLength;
        const REALTYPE* prZ0 = H + internalCount;

        sumH[root] = (double) H[root - stateCount];
        for (int t = 0; t < internalCount; t++) {
            const int p = slot.preOps[3 * t];
            for (int c = 1; c <= 2; c++) {
                const int child = slot.preOps[3 * t + c];
                if (child >= stateCount)
                    sumH[child] = (double) H[child - stateCount] + sumH[p];
            }
        }

        double* out = outMatrices + l * stateCount * stateCount;
        for (int x = 0; x < stateCount; x++) {
            for (int y = 0; y < stateCount; y++) {
                if (x == y)
                    out[x * stateCount + y] = (double) prZ0[x] + (double) pi[x] * sumH[parentOf[x]];
                else
                    out[x * stateCount + y] = (double) pi[y] * sumH[mrca[x * stateCount + y]];
            }
        }
    }
}

template <typename REALTYPE>
static inline void hierarchicalScale(REALTYPE* __restrict__ out, REALTYPE a, const REALTYPE* __restrict__ x,
                                     int k0, int k1) {
    for (int k = k0; k < k1; k++)
        out[k] = a * x[k];
}

template <typename REALTYPE>
static inline void hierarchicalAxpy(REALTYPE* __restrict__ out, REALTYPE a, const REALTYPE* __restrict__ x,
                                    const REALTYPE* __restrict__ z, int k0, int k1) {
    for (int k = k0; k < k1; k++)
        out[k] = a * x[k] + z[k];
}

template <typename REALTYPE>
static inline void hierarchicalAdd(REALTYPE* __restrict__ out, const REALTYPE* __restrict__ x,
                                   const REALTYPE* __restrict__ y, int k0, int k1) {
    for (int k = k0; k < k1; k++)
        out[k] = x[k] + y[k];
}

template <typename REALTYPE, typename Tip>
static inline void hierarchicalTipAxpby(REALTYPE* __restrict__ out, REALTYPE a, const Tip x, REALTYPE b,
                                        const Tip y, int k0, int k1) {
    for (int k = k0; k < k1; k++)
        out[k] = a * x[k] + b * y[k];
}

template <typename REALTYPE, typename Tip>
static inline void hierarchicalTipAxpy(REALTYPE* __restrict__ out, REALTYPE a, const Tip x,
                                       const REALTYPE* __restrict__ z, int k0, int k1) {
    for (int k = k0; k < k1; k++)
        out[k] = a * x[k] + z[k];
}

template <int Mode, typename REALTYPE, typename Tip>
static inline void hierarchicalTipOutput(REALTYPE* __restrict__ out, const REALTYPE* __restrict__ v, REALTYPE a,
                                         const Tip x, const REALTYPE* __restrict__ invScale, int k0, int k1) {
    if (Mode == kHierarchicalWrite) {
        for (int k = k0; k < k1; k++)
            out[k] = v[k] + a * x[k];
    } else if (Mode == kHierarchicalWriteScaled) {
        for (int k = k0; k < k1; k++)
            out[k] = (v[k] + a * x[k]) * invScale[k];
    } else if (Mode == kHierarchicalMultiply) {
        for (int k = k0; k < k1; k++)
            out[k] = out[k] * (v[k] + a * x[k]);
    } else {
        for (int k = k0; k < k1; k++)
            out[k] = out[k] * (v[k] + a * x[k]) * invScale[k];
    }
}

template <int Mode, typename REALTYPE, typename Source>
void hierarchicalSweep(const HierarchicalSlot<REALTYPE>& slot,
                       int stateCount,
                       int category,
                       const REALTYPE* pi,
                       const Source& source,
                       REALTYPE* dest,
                       int destStride,
                       const REALTYPE* invScale,
                       REALTYPE* W,
                       REALTYPE* V,
                       int rowStride,
                       int startPattern,
                       int endPattern) {
    const int S = stateCount;
    const int internalCount = S - 1;
    const REALTYPE* H = slot.values.data() + category * hierarchicalBlockLength(S);
    const REALTYPE* prZ0 = H + internalCount;
    const int* post = slot.postOps.data();
    const int* pre = slot.preOps.data();

    for (int t = 0; t < internalCount; t++) {
        const int p = post[3 * t];
        const int a = post[3 * t + 1];
        const int b = post[3 * t + 2];
        REALTYPE* w = W + (size_t) (p - S) * rowStride;
        if (a < S && b < S) {
            hierarchicalTipAxpby(w, pi[a], source.tip(a), pi[b], source.tip(b), startPattern, endPattern);
        } else if (a < S) {
            hierarchicalTipAxpy(w, pi[a], source.tip(a), W + (size_t) (b - S) * rowStride, startPattern, endPattern);
        } else if (b < S) {
            hierarchicalTipAxpy(w, pi[b], source.tip(b), W + (size_t) (a - S) * rowStride, startPattern, endPattern);
        } else {
            hierarchicalAdd(w, W + (size_t) (a - S) * rowStride, W + (size_t) (b - S) * rowStride,
                            startPattern, endPattern);
        }
    }

    const int root = pre[0];
    hierarchicalScale(V + (size_t) (root - S) * rowStride, H[root - S], W + (size_t) (root - S) * rowStride,
                      startPattern, endPattern);

    for (int t = 0; t < internalCount; t++) {
        const REALTYPE* v = V + (size_t) (pre[3 * t] - S) * rowStride;
        for (int c = 1; c <= 2; c++) {
            const int child = pre[3 * t + c];
            if (child < S) {
                hierarchicalTipOutput<Mode>(dest + (size_t) child * destStride, v, prZ0[child], source.tip(child),
                                            invScale, startPattern, endPattern);
            } else {
                hierarchicalAxpy(V + (size_t) (child - S) * rowStride, H[child - S],
                                 W + (size_t) (child - S) * rowStride, v, startPattern, endPattern);
            }
        }
    }
}

}
}

#endif
