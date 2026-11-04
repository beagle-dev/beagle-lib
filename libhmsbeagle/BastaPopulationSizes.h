/*
 * BastaPopulationSizes.h
 *
 * The checks that every BASTA _v5 call (9-integer operations, BEAGLE_BASTA_OP_COUNT_V5) makes before it changes
 * anything, shared by the CPU and GPU implementations. Internal: not installed.
 */

#ifndef __basta_population_sizes__
#define __basta_population_sizes__

#include <vector>

#include "libhmsbeagle/beagle.h"

namespace beagle {

/*
 * Checks the operations and interval starts of a _v5 call against the current population-size buffer.
 * - every operation: 0 <= intervalNumber (op[7]) < intervalBufferLength, 0 <= offset (op[8]), and
 *   offset + stateCount <= requiredStorageSize;
 * - every operation with the same intervalNumber carries the same offset;
 * - the interval starts index the operations: intervalStarts[i] in [0, count) for i < intervalStartsCount - 1, and
 *   the last one in [0, count], since the calls read the operations of interval i from intervalStarts[i] up to
 *   intervalStarts[i + 1] (intervalStarts may be NULL only when intervalStartsCount < 2: then it is not read).
 * Fills offsetByIntervalNumber[0..intervalBufferLength-1] with each intervalNumber's offset, -1 where no operation
 * uses it. Returns BEAGLE_SUCCESS or BEAGLE_ERROR_OUT_OF_RANGE.
 */
inline int bastaCheckOperations_v5(const int* ops,
                                   int count,
                                   const int* intervalStarts,
                                   int intervalStartsCount,
                                   int stateCount,
                                   int requiredStorageSize,
                                   int intervalBufferLength,
                                   std::vector<int>& offsetByIntervalNumber) {
    offsetByIntervalNumber.assign(intervalBufferLength > 0 ? intervalBufferLength : 0, -1);

    for (int op = 0; op < count; op++) {
        const int intervalNumber = ops[op * BEAGLE_BASTA_OP_COUNT_V5 + 7];
        const int offset = ops[op * BEAGLE_BASTA_OP_COUNT_V5 + 8];
        if (intervalNumber < 0 || intervalNumber >= intervalBufferLength)
            return BEAGLE_ERROR_OUT_OF_RANGE;
        if (offset < 0 || offset > requiredStorageSize - stateCount)
            return BEAGLE_ERROR_OUT_OF_RANGE;
        int& known = offsetByIntervalNumber[intervalNumber];
        if (known == -1)
            known = offset;
        else if (known != offset)
            return BEAGLE_ERROR_OUT_OF_RANGE;
    }

    if (intervalStartsCount >= 2) {
        if (intervalStarts == NULL)
            return BEAGLE_ERROR_OUT_OF_RANGE;
        for (int i = 0; i < intervalStartsCount - 1; i++) {
            if (intervalStarts[i] < 0 || intervalStarts[i] >= count)
                return BEAGLE_ERROR_OUT_OF_RANGE;
        }
        const int last = intervalStarts[intervalStartsCount - 1];
        if (last < 0 || last > count)
            return BEAGLE_ERROR_OUT_OF_RANGE;
    }

    return BEAGLE_SUCCESS;
}

} // namespace beagle

#endif // __basta_population_sizes__
