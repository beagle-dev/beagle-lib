/*
 *  BufferCountRule.h
 *  BEAGLE
 *
 * Copyright 2009 Phylogenetic Likelihood Working Group
 *
 * This file is part of BEAGLE.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 *
 * @brief The count rule of beagleEnsureBufferCounts, for implementations (internal, not installed)
 */

#ifndef __BufferCountRule__
#define __BufferCountRule__

#include <algorithm>
#include <climits>

#include "libhmsbeagle/beagle.h"

namespace beagle {

// The counts an instance has after beagleEnsureBufferCounts, and which kinds grow
struct BufferCounts {
    int bufferCount;   // partials + compact buffers
    int internalCount; // bufferCount - tipCount
    int matrixCount;
    int scaleCount;
    bool growPartials;
    bool growMatrices;
    bool growScale;
};

// beagleEnsureBufferCounts' count rule, from an instance's current counts (CPU/BeagleCPUImpl.hpp keeps its own copy).
// Returns true when something grows, with the new counts in newCounts. Otherwise returns false with returnCode
// BEAGLE_ERROR_OUT_OF_RANGE for a negative count or a count above INT_MAX, or BEAGLE_SUCCESS when no count is larger
// than the current one, which makes (0, 0, 0) a probe. Under SCALING_AUTO or SCALING_ALWAYS the scale count follows
// the partials count, as at creation.
inline bool ensureBufferCountsRule(int partialsBufferCount, int matrixBufferCount, int scaleBufferCount,
                                   int tipCount, int compactBufferCount, int bufferCount, int matrixCount,
                                   int scaleCount, long flags, BufferCounts* newCounts, int* returnCode) {
    *returnCode = BEAGLE_SUCCESS;
    if (partialsBufferCount < 0 || matrixBufferCount < 0 || scaleBufferCount < 0) {
        *returnCode = BEAGLE_ERROR_OUT_OF_RANGE;
        return false;
    }

    // new counts; a request at or below a current count leaves that kind unchanged
    const long long requestedBuffers = (long long) partialsBufferCount + compactBufferCount;
    const int newBuffer = (requestedBuffers > bufferCount) ? (int) std::min<long long>(requestedBuffers, INT_MAX)
                                                           : bufferCount;
    const int newInternal = newBuffer - tipCount;
    const int newMatrix = (matrixBufferCount > matrixCount) ? matrixBufferCount : matrixCount;
    long long requestedScale = scaleBufferCount; // as in createInstance
    if (flags & BEAGLE_FLAG_SCALING_AUTO)
        requestedScale = newInternal;
    else if (flags & BEAGLE_FLAG_SCALING_ALWAYS)
        requestedScale = (long long) newInternal + 1; // + 1 for the temporary buffer of the edge likelihoods
    const int newScale = (requestedScale > scaleCount) ? (int) std::min<long long>(requestedScale, INT_MAX)
                                                       : scaleCount;

    if (requestedBuffers > INT_MAX || requestedScale > INT_MAX) {
        *returnCode = BEAGLE_ERROR_OUT_OF_RANGE;
        return false;
    }

    newCounts->bufferCount = newBuffer;
    newCounts->internalCount = newInternal;
    newCounts->matrixCount = newMatrix;
    newCounts->scaleCount = newScale;
    newCounts->growPartials = newBuffer > bufferCount;
    newCounts->growMatrices = newMatrix > matrixCount;
    newCounts->growScale = newScale > scaleCount;
    return newCounts->growPartials || newCounts->growMatrices || newCounts->growScale;
}

} // namespace beagle

#endif // __BufferCountRule__
