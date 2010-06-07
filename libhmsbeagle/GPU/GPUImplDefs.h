/*
 *
 * Copyright 2009 Phylogenetic Likelihood Working Group
 *
 * This file is part of BEAGLE.
 *
 * BEAGLE is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 *
 * BEAGLE is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with BEAGLE.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @author Marc Suchard
 * @author Daniel Ayres
 */

#ifndef __GPUImplDefs__
#define __GPUImplDefs__

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif
#include "libhmsbeagle/platform.h"

#include <cfloat>

//#define BEAGLE_DEBUG_FLOW
//#define BEAGLE_DEBUG_VALUES
//#define BEAGLE_DEBUG_SYNCH

#define BEAGLE_MEMORY_PINNED

/* Definition of REAL can be switched between 'double' and 'float' */
#ifdef DOUBLE_PRECISION
    #define REAL    double
    #define REAL_MIN    DBL_MIN
    #define REAL_MAX    DBL_MAX
    #define SCALING_FACTOR_COUNT 2046 // -1022, 1023
    #define SCALING_FACTOR_OFFSET 1022 // the zero point
    #define SCALING_EXPONENT_THRESHOLD 200 // TODO: find optimal value for SCALING_EXPONENT_THRESHOLD
#else
    #define REAL    float
    #define REAL_MIN    FLT_MIN
    #define REAL_MAX    FLT_MAX
    #define SCALING_FACTOR_COUNT 254 // -126, 127
    #define SCALING_FACTOR_OFFSET 126 // the zero point
    #define SCALING_EXPONENT_THRESHOLD 20 // TODO: find optimal value for SCALING_EXPONENT_THRESHOLD
#endif

#define SIZE_REAL   sizeof(REAL)
#define INT         int
#define SIZE_INT    sizeof(INT)

/* Compiler definitions
 *
 * PADDED_STATE_COUNT - # of total states after augmentation
 *                      *should* be a multiple of 16
 *
 * PATTERN_BLOCK_SIZE - # of patterns to pack onto each thread-block in pruning
 *                          ( x 4 for PADDED_STATE_COUNT==4)
 *                      PATTERN_BLOCK_SIZE * PADDED_STATE_COUNT <= 512
 *
 * MATRIX_BLOCK_SIZE  - # of matrices to pack onto each thread-block in integrating
 *                        likelihood and store in dynamic weighting;
 *                      MATRIX_BLOCK_SIZE * PADDED_STATE_COUNT <= 512
 *                    - TODO: Currently matrixCount must be < MATRIX_BLOCK_SIZE, fix!
 *
 * BLOCK_PEELING_SIZE - # of the states to pre-fetch in inner-sum in pruning;
 *                      BLOCK_PEELING_SIZE <= PATTERN_BLOCK_SIZE and
 *                      *must* be a divisor of PADDED_STATE_COUNT
 *                      
 * IS_POWER_OF_TWO    - 1 if PADDED_STATE_COUNT = 2^{N} for some integer N, otherwise 0  
 *
 * SMALLEST_POWER_OF_TWO - Smallest power of 2 greater than or equal to PADDED_STATE_COUNT
 *                         (if not already a power of 2)
 *    
 * SLOW_REWEIGHING    - 1 if requires the slow reweighing algorithm, otherwise 0                    
 *    
 */

/* Table of pre-optimized compiler definitions
 */

// state count independent
#define SUM_SITES_BLOCK_SIZE 128

// PADDED_STATE_COUNT == 4
#define PATTERN_BLOCK_SIZE_4          16
#define MATRIX_BLOCK_SIZE_4           8
#define BLOCK_PEELING_SIZE_4          8
#define IS_POWER_OF_TWO_4             1
#define SMALLEST_POWER_OF_TWO_4       4
#define SLOW_REWEIGHING_4             0

// PADDED_STATE_COUNT == 8 TODO

// PADDED_STATE_COUNT == 16
// TODO: find optimal settings
#define PATTERN_BLOCK_SIZE_16         8
#define MATRIX_BLOCK_SIZE_16          8
#define BLOCK_PEELING_SIZE_16         8
#define IS_POWER_OF_TWO_16            1
#define SMALLEST_POWER_OF_TWO_16      16
#define SLOW_REWEIGHING_16            0

// PADDED_STATE_COUNT == 32
// TODO: find optimal settings
#define PATTERN_BLOCK_SIZE_32         8
#define MATRIX_BLOCK_SIZE_32          8
#define BLOCK_PEELING_SIZE_32         8
#define IS_POWER_OF_TWO_32            1
#define SMALLEST_POWER_OF_TWO_32      32
#define SLOW_REWEIGHING_32            0

// PADDED_STATE_COUNT == 48
#define PATTERN_BLOCK_SIZE_48         8
#define MATRIX_BLOCK_SIZE_48          8
#define BLOCK_PEELING_SIZE_48         8
#define IS_POWER_OF_TWO_48            0
#define SMALLEST_POWER_OF_TWO_48      64
#define SLOW_REWEIGHING_48            0

// PADDED_STATE_COUNT == 64
#define PATTERN_BLOCK_SIZE_64         8
#define MATRIX_BLOCK_SIZE_64          8
#define BLOCK_PEELING_SIZE_64         8
#define IS_POWER_OF_TWO_64            1
#define SMALLEST_POWER_OF_TWO_64      64
#define SLOW_REWEIGHING_64            0

// PADDED_STATE_COUNT == 128
#define PATTERN_BLOCK_SIZE_128        4
#define MATRIX_BLOCK_SIZE_128         8
#define BLOCK_PEELING_SIZE_128        2
#define IS_POWER_OF_TWO_128           1
#define SMALLEST_POWER_OF_TWO_128     128
#define SLOW_REWEIGHING_128           1
 
// PADDED_STATE_COUNT == 192
#define PATTERN_BLOCK_SIZE_192        2
#define MATRIX_BLOCK_SIZE_192         8
#define BLOCK_PEELING_SIZE_192        2
#define IS_POWER_OF_TWO_192           0
#define SMALLEST_POWER_OF_TWO_192     256
#define SLOW_REWEIGHING_192           1




#if (STATE_COUNT == 4)
    #define PADDED_STATE_COUNT        4
    #define PATTERN_BLOCK_SIZE        PATTERN_BLOCK_SIZE_4
    #define MATRIX_BLOCK_SIZE         MATRIX_BLOCK_SIZE_4
    #define BLOCK_PEELING_SIZE        BLOCK_PEELING_SIZE_4
    #if (IS_POWER_OF_TWO_4 == 1)
        #define IS_POWER_OF_TWO
    #endif
    #define SMALLEST_POWER_OF_TWO     SMALLEST_POWER_OF_TWO_4
    #if (SLOW_REWEIGHING_4 == 1)
        #define SLOW_REWEIGHING
    #endif
#else
#if (STATE_COUNT <= 8)  // else if
    #define PADDED_STATE_COUNT        8
    #define PATTERN_BLOCK_SIZE        PATTERN_BLOCK_SIZE_8
    #define MATRIX_BLOCK_SIZE         MATRIX_BLOCK_SIZE_8
    #define BLOCK_PEELING_SIZE        BLOCK_PEELING_SIZE_8
    #if (IS_POWER_OF_TWO_8 == 1)
        #define IS_POWER_OF_TWO
    #endif
    #define SMALLEST_POWER_OF_TWO     SMALLEST_POWER_OF_TWO_8
    #if (SLOW_REWEIGHING_8 == 1)
        #define SLOW_REWEIGHING
    #endif
#else
#if (STATE_COUNT <= 16) // else if
    #define PADDED_STATE_COUNT        16
    #define PATTERN_BLOCK_SIZE        PATTERN_BLOCK_SIZE_16
    #define MATRIX_BLOCK_SIZE         MATRIX_BLOCK_SIZE_16
    #define BLOCK_PEELING_SIZE        BLOCK_PEELING_SIZE_16
    #if (IS_POWER_OF_TWO_16 == 1)
        #define IS_POWER_OF_TWO
    #endif
    #define SMALLEST_POWER_OF_TWO     SMALLEST_POWER_OF_TWO_16
    #if (SLOW_REWEIGHING_16 == 1)
        #define SLOW_REWEIGHING
    #endif
#else
#if (STATE_COUNT <= 32) // else if
    #define PADDED_STATE_COUNT        32
    #define PATTERN_BLOCK_SIZE        PATTERN_BLOCK_SIZE_32
    #define MATRIX_BLOCK_SIZE         MATRIX_BLOCK_SIZE_32
    #define BLOCK_PEELING_SIZE        BLOCK_PEELING_SIZE_32
    #if (IS_POWER_OF_TWO_32 == 1)
        #define IS_POWER_OF_TWO
    #endif
    #define SMALLEST_POWER_OF_TWO     SMALLEST_POWER_OF_TWO_32
    #if (SLOW_REWEIGHING_32 == 1)
        #define SLOW_REWEIGHING
    #endif
#else
#if (STATE_COUNT <= 48) // else if
    #define PADDED_STATE_COUNT        48
    #define PATTERN_BLOCK_SIZE        PATTERN_BLOCK_SIZE_48
    #define MATRIX_BLOCK_SIZE         MATRIX_BLOCK_SIZE_48
    #define BLOCK_PEELING_SIZE        BLOCK_PEELING_SIZE_48
    #if (IS_POWER_OF_TWO_48 == 1)
        #define IS_POWER_OF_TWO
    #endif
    #define SMALLEST_POWER_OF_TWO     SMALLEST_POWER_OF_TWO_48
    #if (SLOW_REWEIGHING_48 == 1)
        #define SLOW_REWEIGHING
    #endif
#else
#if (STATE_COUNT <= 64) // else if
    #define PADDED_STATE_COUNT        64
    #define PATTERN_BLOCK_SIZE        PATTERN_BLOCK_SIZE_64
    #define MATRIX_BLOCK_SIZE         MATRIX_BLOCK_SIZE_64
    #define BLOCK_PEELING_SIZE        BLOCK_PEELING_SIZE_64
    #if (IS_POWER_OF_TWO_64 == 1)
        #define IS_POWER_OF_TWO
    #endif
    #define SMALLEST_POWER_OF_TWO     SMALLEST_POWER_OF_TWO_64
    #if (SLOW_REWEIGHING_64 == 1)
        #define SLOW_REWEIGHING
    #endif
#else
#if (STATE_COUNT <= 128) // else if
    #define PADDED_STATE_COUNT        128
    #define PATTERN_BLOCK_SIZE        PATTERN_BLOCK_SIZE_128
    #define MATRIX_BLOCK_SIZE         MATRIX_BLOCK_SIZE_128
    #define BLOCK_PEELING_SIZE        BLOCK_PEELING_SIZE_128
    #if (IS_POWER_OF_TWO_128 == 1)
        #define IS_POWER_OF_TWO
    #endif
    #define SMALLEST_POWER_OF_TWO     SMALLEST_POWER_OF_TWO_128
    #if (SLOW_REWEIGHING_128 == 1)
        #define SLOW_REWEIGHING
    #endif
#else
#if (STATE_COUNT <= 192) // else if
    #define PADDED_STATE_COUNT        192
    #define PATTERN_BLOCK_SIZE        PATTERN_BLOCK_SIZE_192
    #define MATRIX_BLOCK_SIZE         MATRIX_BLOCK_SIZE_192
    #define BLOCK_PEELING_SIZE        BLOCK_PEELING_SIZE_192
    #if (IS_POWER_OF_TWO_192 == 1)
        #define IS_POWER_OF_TWO
    #endif
    #define SMALLEST_POWER_OF_TWO     SMALLEST_POWER_OF_TWO_192
    #if (SLOW_REWEIGHING_192 == 1)
        #define SLOW_REWEIGHING
    #endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif

#define MULTIPLY_BLOCK_SIZE 16

#define MEMCNV(to, from, length, toType)    { \
                                                int m; \
                                                for(m = 0; m < length; m++) { \
                                                    to[m] = (toType) from[m]; \
                                                } \
                                            }

typedef struct Dim3Int Dim3Int;

struct Dim3Int
{
    unsigned int x, y, z;
#if defined(__cplusplus)
    Dim3Int(unsigned int xArg = 1,
            unsigned int yArg = 1,
            unsigned int zArg = 1) : x(xArg), y(yArg), z(zArg) {}
#endif /* __cplusplus */
};

#endif // __GPUImplDefs__
