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

#include <algorithm>
#include <cmath>
#include <string>
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

inline void hierarchicalParents(HierarchicalOps& ops, int stateCount) {
    ops.parentOf.assign(2 * stateCount - 1, -1);
    for (int t = 0; t < stateCount - 1; t++) {
        ops.parentOf[ops.postOps[3 * t + 1]] = ops.postOps[3 * t];
        ops.parentOf[ops.postOps[3 * t + 2]] = ops.postOps[3 * t];
    }
}

inline int decodeHierarchicalModel(const double* eigenVectors,
                                   const double* eigenValues,
                                   int stateCount,
                                   HierarchicalModel& model) {
    const int n = stateCount - 1;
    int header;
    if (eigenVectors == NULL || eigenValues == NULL ||
        !hierarchicalIndex(eigenValues[0], 0, 1 << 30, header) ||
        header / kHierarchicalBundleVersionShift != kHierarchicalModelVersion ||
        header % kHierarchicalBundleVersionShift != n)
        return BEAGLE_ERROR_OUT_OF_RANGE;
    HierarchicalOps ops;
    if (!hierarchicalTriples(eigenVectors, stateCount, ops.postOps))
        return BEAGLE_ERROR_OUT_OF_RANGE;
    ops.preOps.resize(3 * n);
    for (int t = 0; t < n; t++)
        for (int c = 0; c < 3; c++)
            ops.preOps[3 * t + c] = ops.postOps[3 * (n - 1 - t) + c];
    if (!hierarchicalCheckOrders(ops.postOps, ops.preOps, stateCount))
        return BEAGLE_ERROR_OUT_OF_RANGE;
    for (int m = 0; m < n; m++)
        if (!std::isfinite(eigenValues[1 + m]) || eigenValues[1 + m] < 0.0)
            return BEAGLE_ERROR_OUT_OF_RANGE;
    hierarchicalParents(ops, stateCount);
    model.ops.postOps.swap(ops.postOps);
    model.ops.preOps.swap(ops.preOps);
    model.ops.parentOf.swap(ops.parentOf);
    model.lambda.assign(eigenValues + 1, eigenValues + stateCount);
    model.set = true;
    return BEAGLE_SUCCESS;
}

inline void hierarchicalNodeFrequencies(const HierarchicalModel& model,
                                        const double* pi,
                                        int stateCount,
                                        double* piNode) {
    const int* post = model.ops.postOps.data();
    for (int x = 0; x < stateCount; x++)
        piNode[x] = pi[x];
    for (int t = 0; t < stateCount - 1; t++)
        piNode[post[3 * t]] = piNode[post[3 * t + 1]] + piNode[post[3 * t + 2]];
}

template <typename REALTYPE>
void hierarchicalBuildBundle(const HierarchicalModel& model,
                             const double* piNode,
                             const double* categoryRates,
                             int categoryCount,
                             double edgeLength,
                             double* prZ0Tmp,
                             HierarchicalSlot<REALTYPE>& slot) {
    const int S = (int) model.lambda.size() + 1;
    const int n = S - 1;
    const int block = hierarchicalBlockLength(S);
    const int* pre = model.ops.preOps.data();
    const double* lambda = model.lambda.data();
    slot.values.resize((size_t) categoryCount * block);
    for (int l = 0; l < categoryCount; l++) {
        const double distance = edgeLength * categoryRates[l];
        REALTYPE* H = slot.values.data() + (size_t) l * block;
        REALTYPE* prZ0 = H + n;
        prZ0Tmp[pre[0]] = 1.0;
        for (int t = 0; t < n; t++) {
            const int p = pre[3 * t];
            const double x = -lambda[p - S] * distance;
            const double pass = prZ0Tmp[p] * std::exp(x);
            prZ0Tmp[pre[3 * t + 1]] = pass;
            prZ0Tmp[pre[3 * t + 2]] = pass;
            H[p - S] = (REALTYPE) (prZ0Tmp[p] * -std::expm1(x) / piNode[p]);
        }
        for (int x = 0; x < S; x++)
            prZ0[x] = (REALTYPE) prZ0Tmp[x];
    }
    slot.ops = &model.ops;
    slot.owned.reset();
    slot.set = true;
    slot.stamp++;
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

    std::shared_ptr<HierarchicalOps> ops = std::make_shared<HierarchicalOps>();
    if (!hierarchicalTriples(bundle + 1, stateCount, ops->postOps) ||
        !hierarchicalTriples(bundle + 1 + 3 * internalCount, stateCount, ops->preOps) ||
        !hierarchicalCheckOrders(ops->postOps, ops->preOps, stateCount))
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

    hierarchicalParents(*ops, stateCount);
    slot.owned = ops;
    slot.ops = slot.owned.get();
    slot.values.resize(categoryCount * blockLength);
    for (int i = 0; i < categoryCount * blockLength; i++)
        slot.values[i] = (REALTYPE) blocks[i];
    slot.set = true;
    slot.stamp++;
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

    const int* pre = slot.ops->preOps.data();
    const std::vector<int>& parentOf = slot.ops->parentOf;
    std::vector<int> depth(nodeCount, 0);
    for (int t = 0; t < internalCount; t++) {
        const int p = pre[3 * t];
        depth[pre[3 * t + 1]] = depth[p] + 1;
        depth[pre[3 * t + 2]] = depth[p] + 1;
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

    const int root = pre[0];
    std::vector<double> sumH(nodeCount, 0.0);
    for (int l = 0; l < categoryCount; l++) {
        const REALTYPE* H = slot.values.data() + l * blockLength;
        const REALTYPE* prZ0 = H + internalCount;

        sumH[root] = (double) H[root - stateCount];
        for (int t = 0; t < internalCount; t++) {
            const int p = pre[3 * t];
            for (int c = 1; c <= 2; c++) {
                const int child = pre[3 * t + c];
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

inline std::string hierarchicalCanonNode(const std::vector<int>& left,
                                         const std::vector<int>& right,
                                         int stateCount,
                                         int m,
                                         std::vector<int>& tipOrder,
                                         std::vector<int>& intOrder,
                                         int& minTip) {
    if (m < stateCount) {
        tipOrder.push_back(m);
        minTip = m;
        return "L";
    }
    std::vector<int> tipA, tipB, intA, intB;
    int minA, minB;
    std::string ca = hierarchicalCanonNode(left, right, stateCount, left[m], tipA, intA, minA);
    std::string cb = hierarchicalCanonNode(left, right, stateCount, right[m], tipB, intB, minB);
    if (cb < ca || (ca == cb && minB < minA)) {
        ca.swap(cb);
        tipA.swap(tipB);
        intA.swap(intB);
        std::swap(minA, minB);
    }
    tipOrder.insert(tipOrder.end(), tipA.begin(), tipA.end());
    tipOrder.insert(tipOrder.end(), tipB.begin(), tipB.end());
    intOrder.insert(intOrder.end(), intA.begin(), intA.end());
    intOrder.insert(intOrder.end(), intB.begin(), intB.end());
    intOrder.push_back(m);
    minTip = minA;
    return "(" + ca + cb + ")";
}

inline void hierarchicalComputeMrca(HierarchicalOps& ops, int stateCount) {
    const int S = stateCount;
    const int n = S - 1;
    std::vector<int> depth(2 * S - 1, 0);
    for (int t = 0; t < n; t++) {
        const int p = ops.preOps[3 * t];
        depth[ops.preOps[3 * t + 1]] = depth[p] + 1;
        depth[ops.preOps[3 * t + 2]] = depth[p] + 1;
    }
    ops.mrca.assign((size_t) S * S, 0);
    for (int x = 0; x < S; x++) {
        for (int y = 0; y < S; y++) {
            int a = x, b = y;
            while (depth[a] > depth[b]) a = ops.parentOf[a];
            while (depth[b] > depth[a]) b = ops.parentOf[b];
            while (a != b) {
                a = ops.parentOf[a];
                b = ops.parentOf[b];
            }
            ops.mrca[x * S + y] = a;
        }
    }
}

inline void hierarchicalDetectShape(HierarchicalOps& ops, int stateCount) {
    const int S = stateCount;
    const int n = S - 1;
    const int nodeCount = 2 * S - 1;
    std::vector<int> left(nodeCount, -1), right(nodeCount, -1);
    for (int t = 0; t < n; t++) {
        left[ops.postOps[3 * t]] = ops.postOps[3 * t + 1];
        right[ops.postOps[3 * t]] = ops.postOps[3 * t + 2];
    }
    hierarchicalComputeMrca(ops, S);
    ops.tipOfSlot.clear();
    ops.intOfSlot.clear();
    int minTip;
    const std::string code = hierarchicalCanonNode(left, right, S, ops.postOps[3 * (n - 1)],
                                                   ops.tipOfSlot, ops.intOfSlot, minTip);
    ops.shapeId = hierarchicalShapeId(S, code.c_str());
}

template <typename REALTYPE>
void hierarchicalColumnTable(HierarchicalSlot<REALTYPE>& slot,
                             const REALTYPE* pi,
                             int stateCount,
                             int categoryCount,
                             double* sumH) {
    const int S = stateCount;
    const int n = S - 1;
    const int blockLength = hierarchicalBlockLength(S);
    const int width = S + 1;
    const int* pre = slot.ops->preOps.data();
    const int* parentOf = slot.ops->parentOf.data();
    const int* mrca = slot.ops->mrca.data();
    const int root = pre[0];
    slot.columnTable.resize((size_t) categoryCount * S * width);
    for (int l = 0; l < categoryCount; l++) {
        const REALTYPE* H = slot.values.data() + (size_t) l * blockLength;
        const REALTYPE* prZ0 = H + n;
        sumH[root] = (double) H[root - S];
        for (int t = 0; t < n; t++) {
            const int p = pre[3 * t];
            for (int c = 1; c <= 2; c++) {
                const int child = pre[3 * t + c];
                if (child >= S)
                    sumH[child] = (double) H[child - S] + sumH[p];
            }
        }
        REALTYPE* T = slot.columnTable.data() + (size_t) l * S * width;
        for (int x = 0; x < S; x++) {
            double rowSum = 0.0;
            for (int y = 0; y < S; y++) {
                double v;
                if (x == y)
                    v = (double) prZ0[x] + (double) pi[x] * sumH[parentOf[x]];
                else
                    v = (double) pi[y] * sumH[mrca[x * S + y]];
                T[x * width + y] = (REALTYPE) v;
                rowSum += v;
            }
            T[x * width + S] = (REALTYPE) rowSum;
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
    const int* post = slot.ops->postOps.data();
    const int* pre = slot.ops->preOps.data();

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

template <typename REALTYPE, typename Source>
static inline void hierarchicalGroupW(const HierarchicalOps& ops, int S, const REALTYPE* pi, const Source& source,
                                      int k, REALTYPE* __restrict__ W, int G, int n) {
    const int* post = ops.postOps.data();
    for (int t = 0; t < S - 1; t++) {
        const int p = post[3 * t], a = post[3 * t + 1], b = post[3 * t + 2];
        REALTYPE* w = W + (size_t) (p - S) * G;
        if (a < S && b < S)
            hierarchicalTipAxpby(w, pi[a], source.tip(a, k), pi[b], source.tip(b, k), 0, n);
        else if (a < S)
            hierarchicalTipAxpy(w, pi[a], source.tip(a, k), W + (size_t) (b - S) * G, 0, n);
        else if (b < S)
            hierarchicalTipAxpy(w, pi[b], source.tip(b, k), W + (size_t) (a - S) * G, 0, n);
        else
            hierarchicalAdd(w, W + (size_t) (a - S) * G, W + (size_t) (b - S) * G, 0, n);
    }
}

template <typename REALTYPE>
static inline void hierarchicalGroupV(const HierarchicalOps& ops, int S, const REALTYPE* H,
                                      const REALTYPE* __restrict__ W, REALTYPE* __restrict__ V, int G, int n) {
    const int* pre = ops.preOps.data();
    const int root = pre[0];
    hierarchicalScale(V + (size_t) (root - S) * G, H[root - S], W + (size_t) (root - S) * G, 0, n);
    for (int t = 0; t < S - 1; t++) {
        const REALTYPE* v = V + (size_t) (pre[3 * t] - S) * G;
        for (int c = 1; c <= 2; c++) {
            const int child = pre[3 * t + c];
            if (child >= S)
                hierarchicalAxpy(V + (size_t) (child - S) * G, H[child - S], W + (size_t) (child - S) * G, v, 0, n);
        }
    }
}

template <bool Scaled, typename REALTYPE, typename Tip1, typename Tip2>
static inline void hierarchicalGroupOutputPair(REALTYPE* __restrict__ out, const REALTYPE* __restrict__ v1, REALTYPE a1,
                                               const Tip1 x1, const REALTYPE* __restrict__ v2, REALTYPE a2,
                                               const Tip2 x2, const REALTYPE* __restrict__ inv, int n) {
    if (Scaled) {
        for (int j = 0; j < n; j++) {
            const REALTYPE d = v1[j] + a1 * x1[j];
            out[j] = d * (v2[j] + a2 * x2[j]) * inv[j];
        }
    } else {
        for (int j = 0; j < n; j++) {
            const REALTYPE d = v1[j] + a1 * x1[j];
            out[j] = d * (v2[j] + a2 * x2[j]);
        }
    }
}

template <bool Scaled, typename REALTYPE, typename Source1, typename Source2>
void hierarchicalGroupSweepPair(const HierarchicalSlot<REALTYPE>& slot1,
                                const HierarchicalSlot<REALTYPE>& slot2,
                                int S,
                                int category,
                                const REALTYPE* pi,
                                const Source1& source1,
                                const Source2& source2,
                                REALTYPE* dest,
                                int destStride,
                                const REALTYPE* fixedScale,
                                int G,
                                int k0,
                                int k1) {
    alignas(64) REALTYPE W[kHierarchicalGroupScratchBytes / sizeof(REALTYPE)];
    alignas(64) REALTYPE V1[kHierarchicalGroupScratchBytes / sizeof(REALTYPE)];
    alignas(64) REALTYPE V2[kHierarchicalGroupScratchBytes / sizeof(REALTYPE)];
    alignas(64) REALTYPE inv[kHierarchicalGroupMaxLanes];
    const int blockLength = hierarchicalBlockLength(S);
    const REALTYPE* H1 = slot1.values.data() + (size_t) category * blockLength;
    const REALTYPE* H2 = slot2.values.data() + (size_t) category * blockLength;
    const int* parent1 = slot1.ops->parentOf.data();
    const int* parent2 = slot2.ops->parentOf.data();
    for (int k = k0; k < k1; k += G) {
        const int n = std::min(G, k1 - k);
        if (Scaled)
            for (int j = 0; j < n; j++)
                inv[j] = REALTYPE(1.0) / fixedScale[k + j];
        hierarchicalGroupW(*slot1.ops, S, pi, source1, k, W, G, n);
        hierarchicalGroupV(*slot1.ops, S, H1, W, V1, G, n);
        hierarchicalGroupW(*slot2.ops, S, pi, source2, k, W, G, n);
        hierarchicalGroupV(*slot2.ops, S, H2, W, V2, G, n);
        for (int x = 0; x < S; x++)
            hierarchicalGroupOutputPair<Scaled>(dest + (size_t) x * destStride + k,
                                                V1 + (size_t) (parent1[x] - S) * G, H1[S - 1 + x], source1.tip(x, k),
                                                V2 + (size_t) (parent2[x] - S) * G, H2[S - 1 + x], source2.tip(x, k),
                                                inv, n);
    }
}

template <bool Scaled, typename REALTYPE, typename Source>
void hierarchicalGroupSweepSingle(const HierarchicalSlot<REALTYPE>& slot,
                                  int S,
                                  int category,
                                  const REALTYPE* pi,
                                  const Source& source,
                                  REALTYPE* dest,
                                  int destStride,
                                  const REALTYPE* fixedScale,
                                  int G,
                                  int k0,
                                  int k1) {
    alignas(64) REALTYPE W[kHierarchicalGroupScratchBytes / sizeof(REALTYPE)];
    alignas(64) REALTYPE V[kHierarchicalGroupScratchBytes / sizeof(REALTYPE)];
    alignas(64) REALTYPE inv[kHierarchicalGroupMaxLanes];
    const int blockLength = hierarchicalBlockLength(S);
    const REALTYPE* H = slot.values.data() + (size_t) category * blockLength;
    const int* parent = slot.ops->parentOf.data();
    for (int k = k0; k < k1; k += G) {
        const int n = std::min(G, k1 - k);
        if (Scaled)
            for (int j = 0; j < n; j++)
                inv[j] = REALTYPE(1.0) / fixedScale[k + j];
        hierarchicalGroupW(*slot.ops, S, pi, source, k, W, G, n);
        hierarchicalGroupV(*slot.ops, S, H, W, V, G, n);
        for (int x = 0; x < S; x++)
            hierarchicalTipOutput<Scaled ? kHierarchicalWriteScaled : kHierarchicalWrite>(
                    dest + (size_t) x * destStride + k, V + (size_t) (parent[x] - S) * G, H[S - 1 + x],
                    source.tip(x, k), inv, 0, n);
    }
}

template <bool Scaled, typename REALTYPE, typename Source>
void hierarchicalGroupSweepColumn(const HierarchicalSlot<REALTYPE>& slot,
                                  int S,
                                  int category,
                                  const REALTYPE* pi,
                                  const Source& source,
                                  const int* states,
                                  const REALTYPE* table,
                                  REALTYPE* dest,
                                  int destStride,
                                  const REALTYPE* fixedScale,
                                  int G,
                                  int k0,
                                  int k1) {
    alignas(64) REALTYPE W[kHierarchicalGroupScratchBytes / sizeof(REALTYPE)];
    alignas(64) REALTYPE V[kHierarchicalGroupScratchBytes / sizeof(REALTYPE)];
    alignas(64) REALTYPE inv[kHierarchicalGroupMaxLanes];
    const int width = S + 1;
    const int blockLength = hierarchicalBlockLength(S);
    const REALTYPE* H = slot.values.data() + (size_t) category * blockLength;
    const int* parent = slot.ops->parentOf.data();
    for (int k = k0; k < k1; k += G) {
        const int n = std::min(G, k1 - k);
        if (Scaled)
            for (int j = 0; j < n; j++)
                inv[j] = REALTYPE(1.0) / fixedScale[k + j];
        hierarchicalGroupW(*slot.ops, S, pi, source, k, W, G, n);
        hierarchicalGroupV(*slot.ops, S, H, W, V, G, n);
        const int* __restrict__ s = states + k;
        for (int x = 0; x < S; x++) {
            REALTYPE* __restrict__ out = dest + (size_t) x * destStride + k;
            const REALTYPE* __restrict__ v = V + (size_t) (parent[x] - S) * G;
            const REALTYPE a = H[S - 1 + x];
            const auto tip = source.tip(x, k);
            const REALTYPE* __restrict__ row = table + (size_t) x * width;
            for (int j = 0; j < n; j++) {
                const REALTYPE d = v[j] + a * tip[j];
                out[j] = Scaled ? d * row[s[j]] * inv[j] : d * row[s[j]];
            }
        }
    }
}

template <bool Scaled, typename REALTYPE>
void hierarchicalColumnRows(int S,
                            const int* states1,
                            const REALTYPE* table1,
                            const int* states2,
                            const REALTYPE* table2,
                            REALTYPE* dest,
                            int destStride,
                            const REALTYPE* fixedScale,
                            int k0,
                            int k1) {
    const int width = S + 1;
    const int* __restrict__ s1 = states1;
    const int* __restrict__ s2 = states2;
    const REALTYPE* __restrict__ f = fixedScale;
    for (int x = 0; x < S; x++) {
        REALTYPE* __restrict__ out = dest + (size_t) x * destStride;
        const REALTYPE* __restrict__ row1 = table1 + (size_t) x * width;
        if (states2 != NULL) {
            const REALTYPE* __restrict__ row2 = table2 + (size_t) x * width;
            for (int k = k0; k < k1; k++)
                out[k] = Scaled ? row1[s1[k]] * row2[s2[k]] * (REALTYPE(1.0) / f[k]) : row1[s1[k]] * row2[s2[k]];
        } else {
            for (int k = k0; k < k1; k++)
                out[k] = Scaled ? row1[s1[k]] * (REALTYPE(1.0) / f[k]) : row1[s1[k]];
        }
    }
}

#if defined(__clang__)
#define BEAGLE_HIERARCHICAL_LANE_LOOP _Pragma("clang loop vectorize(assume_safety)")
#define BEAGLE_HIERARCHICAL_UNROLL _Pragma("unroll")
#elif defined(__GNUC__)
#define BEAGLE_HIERARCHICAL_LANE_LOOP _Pragma("GCC ivdep")
#define BEAGLE_HIERARCHICAL_UNROLL _Pragma("GCC unroll 16")
#else
#define BEAGLE_HIERARCHICAL_LANE_LOOP
#define BEAGLE_HIERARCHICAL_UNROLL
#endif

#if defined(__GNUC__) || defined(__clang__)
#define BEAGLE_HIERARCHICAL_INLINE inline __attribute__((always_inline))
#define BEAGLE_HIERARCHICAL_NOINLINE __attribute__((noinline))
#else
#define BEAGLE_HIERARCHICAL_INLINE inline
#define BEAGLE_HIERARCHICAL_NOINLINE
#endif

template <typename REALTYPE, int S>
struct HierarchicalLaneCoef {
    REALTYPE pi[S];
    REALTYPE z[S];
    REALTYPE h[S - 1];
};

template <typename REALTYPE, int S, int ID>
static BEAGLE_HIERARCHICAL_INLINE void hierarchicalLane(const REALTYPE* const* L,
                                                        const HierarchicalLaneCoef<REALTYPE, S>& c,
                                                        int k,
                                                        REALTYPE* out) {
    typedef HierarchicalShape<S, ID> Shape;
    REALTYPE W[S - 1], V[S - 1];
    BEAGLE_HIERARCHICAL_UNROLL
    for (int t = 0; t < S - 1; t++) {
        const int p = Shape::post[3 * t];
        const int a = Shape::post[3 * t + 1];
        const int b = Shape::post[3 * t + 2];
        const int ta = a < S ? a : 0, tb = b < S ? b : 0;
        const int ia = a < S ? 0 : a - S, ib = b < S ? 0 : b - S;
        if (a < S && b < S)
            W[p - S] = c.pi[ta] * L[ta][k] + c.pi[tb] * L[tb][k];
        else if (a < S)
            W[p - S] = c.pi[ta] * L[ta][k] + W[ib];
        else if (b < S)
            W[p - S] = c.pi[tb] * L[tb][k] + W[ia];
        else
            W[p - S] = W[ia] + W[ib];
    }
    const int root = Shape::pre[0];
    V[root - S] = c.h[root - S] * W[root - S];
    BEAGLE_HIERARCHICAL_UNROLL
    for (int t = 0; t < S - 1; t++) {
        const int p = Shape::pre[3 * t];
        BEAGLE_HIERARCHICAL_UNROLL
        for (int ci = 1; ci <= 2; ci++) {
            const int child = Shape::pre[3 * t + ci];
            if (child < S) {
                const int x = child < S ? child : 0;
                out[x] = V[p - S] + c.z[x] * L[x][k];
            } else {
                const int i = child < S ? 0 : child - S;
                V[i] = V[p - S] + c.h[i] * W[i];
            }
        }
    }
}

template <typename REALTYPE, int S, int ID, bool Scaled>
BEAGLE_HIERARCHICAL_NOINLINE void hierarchicalRegSingle(const REALTYPE* const* L_,
                                                        const HierarchicalLaneCoef<REALTYPE, S>& c_,
                                                        REALTYPE* const* dest_,
                                                        const REALTYPE* invScale,
                                                        int k0,
                                                        int k1) {
    const HierarchicalLaneCoef<REALTYPE, S> c = c_;
    const REALTYPE* L[S];
    REALTYPE* dest[S];
    for (int x = 0; x < S; x++) {
        L[x] = L_[x];
        dest[x] = dest_[x];
    }
    BEAGLE_HIERARCHICAL_LANE_LOOP
    for (int k = k0; k < k1; k++) {
        REALTYPE o[S];
        hierarchicalLane<REALTYPE, S, ID>(L, c, k, o);
        BEAGLE_HIERARCHICAL_UNROLL
        for (int x = 0; x < S; x++) {
            REALTYPE v = o[x];
            if (Scaled)
                v = v * invScale[k];
            dest[x][k] = v;
        }
    }
}

template <typename REALTYPE, int S, int ID, bool Scaled>
BEAGLE_HIERARCHICAL_NOINLINE void hierarchicalRegPair(const REALTYPE* const* L1_,
                                                      const HierarchicalLaneCoef<REALTYPE, S>& c1_,
                                                      const REALTYPE* const* L2_,
                                                      const HierarchicalLaneCoef<REALTYPE, S>& c2_,
                                                      REALTYPE* const* dest_,
                                                      const REALTYPE* invScale,
                                                      int k0,
                                                      int k1) {
    const HierarchicalLaneCoef<REALTYPE, S> c1 = c1_;
    const HierarchicalLaneCoef<REALTYPE, S> c2 = c2_;
    const REALTYPE* L1[S];
    const REALTYPE* L2[S];
    REALTYPE* dest[S];
    for (int x = 0; x < S; x++) {
        L1[x] = L1_[x];
        L2[x] = L2_[x];
        dest[x] = dest_[x];
    }
    BEAGLE_HIERARCHICAL_LANE_LOOP
    for (int k = k0; k < k1; k++) {
        REALTYPE o1[S], o2[S];
        hierarchicalLane<REALTYPE, S, ID>(L1, c1, k, o1);
        hierarchicalLane<REALTYPE, S, ID>(L2, c2, k, o2);
        BEAGLE_HIERARCHICAL_UNROLL
        for (int x = 0; x < S; x++) {
            REALTYPE v = o1[x] * o2[x];
            if (Scaled)
                v = v * invScale[k];
            dest[x][k] = v;
        }
    }
}

template <typename REALTYPE, int S, int ID, bool Scaled>
BEAGLE_HIERARCHICAL_NOINLINE void hierarchicalRegColPair(const REALTYPE* const* L1_,
                                                         const HierarchicalLaneCoef<REALTYPE, S>& c1_,
                                                         const int* states2,
                                                         const REALTYPE* T2,
                                                         REALTYPE* const* dest_,
                                                         const REALTYPE* invScale,
                                                         int k0,
                                                         int k1) {
    const HierarchicalLaneCoef<REALTYPE, S> c1 = c1_;
    const REALTYPE* L1[S];
    REALTYPE* dest[S];
    for (int x = 0; x < S; x++) {
        L1[x] = L1_[x];
        dest[x] = dest_[x];
    }
    BEAGLE_HIERARCHICAL_LANE_LOOP
    for (int k = k0; k < k1; k++) {
        REALTYPE o1[S];
        hierarchicalLane<REALTYPE, S, ID>(L1, c1, k, o1);
        const int s = states2[k];
        BEAGLE_HIERARCHICAL_UNROLL
        for (int x = 0; x < S; x++) {
            REALTYPE v = o1[x] * T2[x * (S + 1) + s];
            if (Scaled)
                v = v * invScale[k];
            dest[x][k] = v;
        }
    }
}

template <typename REALTYPE, int S, bool Scaled>
BEAGLE_HIERARCHICAL_NOINLINE void hierarchicalColSingle(const int* states,
                                                        const REALTYPE* T,
                                                        REALTYPE* const* dest_,
                                                        const REALTYPE* invScale,
                                                        int k0,
                                                        int k1) {
    REALTYPE* dest[S];
    for (int x = 0; x < S; x++)
        dest[x] = dest_[x];
    BEAGLE_HIERARCHICAL_LANE_LOOP
    for (int k = k0; k < k1; k++) {
        const int s = states[k];
        BEAGLE_HIERARCHICAL_UNROLL
        for (int x = 0; x < S; x++) {
            REALTYPE v = T[x * (S + 1) + s];
            if (Scaled)
                v = v * invScale[k];
            dest[x][k] = v;
        }
    }
}

template <typename REALTYPE, int S, bool Scaled>
BEAGLE_HIERARCHICAL_NOINLINE void hierarchicalColColPair(const int* states1,
                                                         const REALTYPE* T1,
                                                         const int* states2,
                                                         const REALTYPE* T2,
                                                         REALTYPE* const* dest_,
                                                         const REALTYPE* invScale,
                                                         int k0,
                                                         int k1) {
    REALTYPE* dest[S];
    for (int x = 0; x < S; x++)
        dest[x] = dest_[x];
    BEAGLE_HIERARCHICAL_LANE_LOOP
    for (int k = k0; k < k1; k++) {
        const int a = states1[k], b = states2[k];
        BEAGLE_HIERARCHICAL_UNROLL
        for (int x = 0; x < S; x++) {
            REALTYPE v = T1[x * (S + 1) + a] * T2[x * (S + 1) + b];
            if (Scaled)
                v = v * invScale[k];
            dest[x][k] = v;
        }
    }
}

template <typename REALTYPE, int S>
static BEAGLE_HIERARCHICAL_INLINE void hierarchicalDenseRow(const REALTYPE* const* L, const REALTYPE* P, int k,
                                                            REALTYPE* out) {
    BEAGLE_HIERARCHICAL_UNROLL
    for (int x = 0; x < S; x++) {
        REALTYPE a = P[x * S] * L[0][k];
        BEAGLE_HIERARCHICAL_UNROLL
        for (int y = 1; y < S; y++)
            a = a + P[x * S + y] * L[y][k];
        out[x] = a;
    }
}

template <typename REALTYPE, int S, bool Scaled>
BEAGLE_HIERARCHICAL_NOINLINE void hierarchicalDensePair(const REALTYPE* const* L1_,
                                                        const REALTYPE* T1,
                                                        const REALTYPE* const* L2_,
                                                        const REALTYPE* T2,
                                                        REALTYPE* const* dest_,
                                                        const REALTYPE* invScale,
                                                        int k0,
                                                        int k1) {
    const REALTYPE* L1[S];
    const REALTYPE* L2[S];
    REALTYPE* dest[S];
    REALTYPE P1[S * S], P2[S * S];
    for (int x = 0; x < S; x++) {
        L1[x] = L1_[x];
        L2[x] = L2_[x];
        dest[x] = dest_[x];
        for (int y = 0; y < S; y++) {
            P1[x * S + y] = T1[x * (S + 1) + y];
            P2[x * S + y] = T2[x * (S + 1) + y];
        }
    }
    BEAGLE_HIERARCHICAL_LANE_LOOP
    for (int k = k0; k < k1; k++) {
        REALTYPE o1[S], o2[S];
        hierarchicalDenseRow<REALTYPE, S>(L1, P1, k, o1);
        hierarchicalDenseRow<REALTYPE, S>(L2, P2, k, o2);
        BEAGLE_HIERARCHICAL_UNROLL
        for (int x = 0; x < S; x++) {
            REALTYPE v = o1[x] * o2[x];
            if (Scaled)
                v = v * invScale[k];
            dest[x][k] = v;
        }
    }
}

template <typename REALTYPE, int S, bool Scaled>
BEAGLE_HIERARCHICAL_NOINLINE void hierarchicalDenseColPair(const REALTYPE* const* L1_,
                                                           const REALTYPE* T1,
                                                           const int* states2,
                                                           const REALTYPE* T2,
                                                           REALTYPE* const* dest_,
                                                           const REALTYPE* invScale,
                                                           int k0,
                                                           int k1) {
    const REALTYPE* L1[S];
    REALTYPE* dest[S];
    REALTYPE P1[S * S];
    for (int x = 0; x < S; x++) {
        L1[x] = L1_[x];
        dest[x] = dest_[x];
        for (int y = 0; y < S; y++)
            P1[x * S + y] = T1[x * (S + 1) + y];
    }
    BEAGLE_HIERARCHICAL_LANE_LOOP
    for (int k = k0; k < k1; k++) {
        REALTYPE o1[S];
        hierarchicalDenseRow<REALTYPE, S>(L1, P1, k, o1);
        const int s = states2[k];
        BEAGLE_HIERARCHICAL_UNROLL
        for (int x = 0; x < S; x++) {
            REALTYPE v = o1[x] * T2[x * (S + 1) + s];
            if (Scaled)
                v = v * invScale[k];
            dest[x][k] = v;
        }
    }
}

const int kHierarchicalRescaleMaxRows = 32;

template <typename REALTYPE, int N>
BEAGLE_HIERARCHICAL_NOINLINE void hierarchicalRescaleRows(REALTYPE* const* rows_, REALTYPE* siteMax, int k0, int k1) {
    constexpr int NP = N <= 1 ? 1 : N <= 2 ? 2 : N <= 4 ? 4 : N <= 8 ? 8 : N <= 16 ? 16 : 32;
    REALTYPE* rows[N];
    for (int i = 0; i < N; i++)
        rows[i] = rows_[i];
    BEAGLE_HIERARCHICAL_LANE_LOOP
    for (int k = k0; k < k1; k++) {
        REALTYPE v[N], t[NP];
        BEAGLE_HIERARCHICAL_UNROLL
        for (int i = 0; i < N; i++)
            v[i] = rows[i][k];
        BEAGLE_HIERARCHICAL_UNROLL
        for (int i = 0; i < NP; i++)
            t[i] = v[i < N ? i : 0];
        BEAGLE_HIERARCHICAL_UNROLL
        for (int w = NP / 2; w >= 1; w /= 2) {
            BEAGLE_HIERARCHICAL_UNROLL
            for (int i = 0; i < w; i++)
                t[i] = std::fmax(t[i], t[i + w]);
        }
        const REALTYPE m = (t[0] == REALTYPE(0)) ? REALTYPE(1.0) : t[0];
        const REALTYPE f = REALTYPE(1.0) / m;
        BEAGLE_HIERARCHICAL_UNROLL
        for (int i = 0; i < N; i++)
            rows[i][k] = v[i] * f;
        siteMax[k] = m;
    }
}

template <typename REALTYPE, int N = 1>
struct HierarchicalRowsDispatch {
    static bool rescale(int n, REALTYPE* const* rows, REALTYPE* siteMax, int k0, int k1) {
        if constexpr (N <= kHierarchicalRescaleMaxRows) {
            if (n == N) {
                hierarchicalRescaleRows<REALTYPE, N>(rows, siteMax, k0, k1);
                return true;
            }
            return HierarchicalRowsDispatch<REALTYPE, N + 1>::rescale(n, rows, siteMax, k0, k1);
        }
        return false;
    }
};

template <typename REALTYPE, int S, bool Scaled, int ID = 0>
struct HierarchicalShapeDispatch {
    static void single(int id, const REALTYPE* const* L, const HierarchicalLaneCoef<REALTYPE, S>& c,
                       REALTYPE* const* dest, const REALTYPE* invScale, int k0, int k1) {
        if constexpr (ID < HierarchicalShapeCount<S>::value) {
            if (id == ID)
                hierarchicalRegSingle<REALTYPE, S, ID, Scaled>(L, c, dest, invScale, k0, k1);
            else
                HierarchicalShapeDispatch<REALTYPE, S, Scaled, ID + 1>::single(id, L, c, dest, invScale, k0, k1);
        }
    }

    static void pair(int id, const REALTYPE* const* L1, const HierarchicalLaneCoef<REALTYPE, S>& c1,
                     const REALTYPE* const* L2, const HierarchicalLaneCoef<REALTYPE, S>& c2,
                     REALTYPE* const* dest, const REALTYPE* invScale, int k0, int k1) {
        if constexpr (ID < HierarchicalShapeCount<S>::value) {
            if (id == ID)
                hierarchicalRegPair<REALTYPE, S, ID, Scaled>(L1, c1, L2, c2, dest, invScale, k0, k1);
            else
                HierarchicalShapeDispatch<REALTYPE, S, Scaled, ID + 1>::pair(id, L1, c1, L2, c2, dest, invScale, k0, k1);
        }
    }

    static void column(int id, const REALTYPE* const* L1, const HierarchicalLaneCoef<REALTYPE, S>& c1,
                       const int* states2, const REALTYPE* T2,
                       REALTYPE* const* dest, const REALTYPE* invScale, int k0, int k1) {
        if constexpr (ID < HierarchicalShapeCount<S>::value) {
            if (id == ID)
                hierarchicalRegColPair<REALTYPE, S, ID, Scaled>(L1, c1, states2, T2, dest, invScale, k0, k1);
            else
                HierarchicalShapeDispatch<REALTYPE, S, Scaled, ID + 1>::column(id, L1, c1, states2, T2, dest,
                                                                                invScale, k0, k1);
        }
    }
};

}
}

#endif
