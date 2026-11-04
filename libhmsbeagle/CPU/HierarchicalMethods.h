/*
 *  HierarchicalMethods.h
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

#ifndef HIERARCHICALMETHODS_H_
#define HIERARCHICALMETHODS_H_

#include <cstddef>
#include <vector>

namespace beagle {
namespace cpu {

const int kHierarchicalBundleVersion = 1;
const int kHierarchicalBundleVersionShift = 65536;
const size_t kHierarchicalTileBudget = 768 * 1024;

template <typename REALTYPE>
struct HierarchicalSlot {
    bool set = false;
    std::vector<int> postOps;
    std::vector<int> preOps;
    std::vector<REALTYPE> values;
};

inline int hierarchicalBundleLength(int stateCount, int categoryCount) {
    const int internalCount = stateCount - 1;
    return 1 + 6 * internalCount + categoryCount * (internalCount + stateCount);
}

inline int hierarchicalBlockLength(int stateCount) {
    return (stateCount - 1) + stateCount;
}

template <typename REALTYPE>
int unpackHierarchicalBundle(const double* bundle,
                             int stateCount,
                             int categoryCount,
                             HierarchicalSlot<REALTYPE>& slot);

template <typename REALTYPE>
void hierarchicalTransitionMatrices(const HierarchicalSlot<REALTYPE>& slot,
                                    const REALTYPE* pi,
                                    int stateCount,
                                    int categoryCount,
                                    double* outMatrices);

enum HierarchicalOutput {
    kHierarchicalWrite = 0,
    kHierarchicalWriteScaled = 1,
    kHierarchicalMultiply = 2,
    kHierarchicalMultiplyScaled = 3
};

template <typename REALTYPE>
struct HierarchicalPartialsTip {
    const REALTYPE* __restrict__ row;
    inline REALTYPE operator[](int k) const { return row[k]; }
};

template <typename REALTYPE>
struct HierarchicalStatesTip {
    const int* __restrict__ states;
    int state;
    int stateCount;
    inline REALTYPE operator[](int k) const {
        return (states[k] == state || states[k] >= stateCount) ? REALTYPE(1) : REALTYPE(0);
    }
};

template <typename REALTYPE>
struct HierarchicalPartialsSource {
    const REALTYPE* rows;
    int stride;
    int categoryStride;
    inline HierarchicalPartialsSource category(int l) const {
        return HierarchicalPartialsSource{ rows + (size_t) l * categoryStride, stride, categoryStride };
    }
    inline HierarchicalPartialsTip<REALTYPE> tip(int x) const {
        return HierarchicalPartialsTip<REALTYPE>{ rows + (size_t) x * stride };
    }
};

template <typename REALTYPE>
struct HierarchicalStatesSource {
    const int* states;
    int stateCount;
    inline HierarchicalStatesSource category(int l) const {
        return *this;
    }
    inline HierarchicalStatesTip<REALTYPE> tip(int x) const {
        return HierarchicalStatesTip<REALTYPE>{ states, x, stateCount };
    }
};

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
                       int endPattern);

}
}

#include "libhmsbeagle/CPU/HierarchicalMethods.hpp"

#endif
