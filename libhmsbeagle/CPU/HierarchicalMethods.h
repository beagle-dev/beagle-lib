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
#include <memory>
#include <vector>

#include "libhmsbeagle/CPU/HierarchicalShapes.h"

namespace beagle {
namespace cpu {

const int kHierarchicalBundleVersion = 1;
const int kHierarchicalModelVersion = 1;
const int kHierarchicalBundleVersionShift = 65536;
const size_t kHierarchicalTileBudget = 768 * 1024;
const int kHierarchicalShapeMaxStates = 4;
const size_t kHierarchicalGroupBudget = 48 * 1024;
const size_t kHierarchicalGroupScratchBytes = 16 * 1024;
const int kHierarchicalGroupMaxLanes = 256;
const int kHierarchicalGroupMaxStates = 128;
const int kHierarchicalColumnMaxStates = 64;
const int kHierarchicalFusedDefaultMinStates = 5;
const int kHierarchicalFusedMinPatterns = 1000;
const size_t kHierarchicalTableBudget = 64 * 1024 * 1024;

enum HierarchicalKernel {
    kHierarchicalKernelTiled = 0,
    kHierarchicalKernelShape = 1,
    kHierarchicalKernelDense = 2,
    kHierarchicalKernelFused = 3
};

enum HierarchicalGroupOrder {
    kHierarchicalOrderCategory = 0,
    kHierarchicalOrderGroup = 1
};

inline int hierarchicalGroupLanes(int stateCount, size_t realSize) {
    if (stateCount > kHierarchicalGroupMaxStates)
        return 0;
    const int floorLanes = (int) (128 / realSize);
    for (int G = (int) (1024 / realSize); G > floorLanes; G /= 2) {
        if ((size_t) (6 * stateCount - 3) * G * realSize <= kHierarchicalGroupBudget &&
            (size_t) (stateCount - 1) * G * realSize <= kHierarchicalGroupScratchBytes)
            return G;
    }
    return (size_t) (stateCount - 1) * floorLanes * realSize <= kHierarchicalGroupScratchBytes ? floorLanes : 0;
}

enum HierarchicalSmallPlan {
    kHierarchicalSmallNone = 0,
    kHierarchicalSmallRegSingle,
    kHierarchicalSmallColSingle,
    kHierarchicalSmallRegPair,
    kHierarchicalSmallDensePair,
    kHierarchicalSmallRegCol,
    kHierarchicalSmallColReg,
    kHierarchicalSmallDenseCol,
    kHierarchicalSmallColDense,
    kHierarchicalSmallColCol
};

struct HierarchicalOps {
    std::vector<int> postOps;
    std::vector<int> preOps;
    std::vector<int> parentOf;
    int shapeId = -1;
    std::vector<int> tipOfSlot;
    std::vector<int> intOfSlot;
    std::vector<int> mrca;
};

struct HierarchicalModel {
    bool set = false;
    HierarchicalOps ops;
    std::vector<double> lambda;
};

template <typename REALTYPE>
struct HierarchicalSlot {
    bool set = false;
    const HierarchicalOps* ops = nullptr;
    std::shared_ptr<HierarchicalOps> owned;
    std::vector<REALTYPE> values;
    unsigned long stamp = 0;
    unsigned long tableStamp = 0;
    unsigned long tableFrequencyStamp = 0;
    bool tableSet = false;
    std::vector<REALTYPE> columnTable;
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

inline int decodeHierarchicalModel(const double* eigenVectors,
                            const double* eigenValues,
                            int stateCount,
                            HierarchicalModel& model);

inline void hierarchicalNodeFrequencies(const HierarchicalModel& model,
                                 const double* pi,
                                 int stateCount,
                                 double* piNode);

template <typename REALTYPE>
void hierarchicalBuildBundle(const HierarchicalModel& model,
                             const double* piNode,
                             const double* categoryRates,
                             int categoryCount,
                             double edgeLength,
                             double* prZ0Tmp,
                             HierarchicalSlot<REALTYPE>& slot);

inline void hierarchicalComputeMrca(HierarchicalOps& ops, int stateCount);

inline void hierarchicalDetectShape(HierarchicalOps& ops, int stateCount);

template <typename REALTYPE>
void hierarchicalColumnTable(HierarchicalSlot<REALTYPE>& slot,
                             const REALTYPE* pi,
                             int stateCount,
                             int categoryCount,
                             double* sumH);

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
    inline HierarchicalPartialsTip<REALTYPE> tip(int x, int k) const {
        return HierarchicalPartialsTip<REALTYPE>{ rows + (size_t) x * stride + k };
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
    inline HierarchicalStatesTip<REALTYPE> tip(int x, int k) const {
        return HierarchicalStatesTip<REALTYPE>{ states + k, x, stateCount };
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
