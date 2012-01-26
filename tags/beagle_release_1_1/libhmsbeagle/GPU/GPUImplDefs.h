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
//#define BEAGLE_FILL_4_STATE_SCALAR_SS
//#define BEAGLE_FILL_4_STATE_SCALAR_SP

#define BEAGLE_CACHED_MATRICES_COUNT 3 // max number of matrices that can be cached for a single memcpy to device operation

/* Definition of REAL can be switched between 'double' and 'float' */
#ifdef DOUBLE_PRECISION
    #define REAL    double
    #define REAL_MIN    DBL_MIN
    #define REAL_MAX    DBL_MAX
    #define SCALING_FACTOR_COUNT 2046 // -1022, 1023
    #define SCALING_FACTOR_OFFSET 1022 // the zero point
    #define SCALING_EXPONENT_THRESHOLD 200 // TODO: find optimal value for SCALING_EXPONENT_THRESHOLD
    #define SCALING_THRESHOLD_LOWER 6.22301528e-61 // TODO: find optimal value for SCALING_THRESHOLD
    #define SCALING_THRESHOLD_UPPER 1.60693804e60 // TODO: find optimal value for SCALING_THRESHOLD
#else
    #define REAL    float
    #define REAL_MIN    FLT_MIN
    #define REAL_MAX    FLT_MAX
    #define SCALING_FACTOR_COUNT 254 // -126, 127
    #define SCALING_FACTOR_OFFSET 126 // the zero point
    #define SCALING_EXPONENT_THRESHOLD 20 // TODO: find optimal value for SCALING_EXPONENT_THRESHOLD
    #define SCALING_THRESHOLD_LOWER 9.53674316e-7 // TODO: find optimal value for SCALING_THRESHOLD
    #define SCALING_THRESHOLD_UPPER 1048576 // TODO: find optimal value for SCALING_THRESHOLD
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

// SINGLE PRECISION definitions

// PADDED_STATE_COUNT == 4
#define PATTERN_BLOCK_SIZE_SP_4          16
#define MATRIX_BLOCK_SIZE_SP_4           8
#define BLOCK_PEELING_SIZE_SP_4          8
#define IS_POWER_OF_TWO_SP_4             1
#define SMALLEST_POWER_OF_TWO_SP_4       4
#define SLOW_REWEIGHING_SP_4             0

// PADDED_STATE_COUNT == 16
// TODO: find optimal settings
#define PATTERN_BLOCK_SIZE_SP_16         8
#define MATRIX_BLOCK_SIZE_SP_16          8
#define BLOCK_PEELING_SIZE_SP_16         8
#define IS_POWER_OF_TWO_SP_16            1
#define SMALLEST_POWER_OF_TWO_SP_16      16
#define SLOW_REWEIGHING_SP_16            0

// PADDED_STATE_COUNT == 32
// TODO: find optimal settings
#define PATTERN_BLOCK_SIZE_SP_32         8
#define MATRIX_BLOCK_SIZE_SP_32          8
#define BLOCK_PEELING_SIZE_SP_32         8
#define IS_POWER_OF_TWO_SP_32            1
#define SMALLEST_POWER_OF_TWO_SP_32      32
#define SLOW_REWEIGHING_SP_32            0

// PADDED_STATE_COUNT == 48
#define PATTERN_BLOCK_SIZE_SP_48         8
#define MATRIX_BLOCK_SIZE_SP_48          8
#define BLOCK_PEELING_SIZE_SP_48         8
#define IS_POWER_OF_TWO_SP_48            0
#define SMALLEST_POWER_OF_TWO_SP_48      64
#define SLOW_REWEIGHING_SP_48            0

// PADDED_STATE_COUNT == 64
#define PATTERN_BLOCK_SIZE_SP_64         8
#define MATRIX_BLOCK_SIZE_SP_64          8
#define BLOCK_PEELING_SIZE_SP_64         8
#define IS_POWER_OF_TWO_SP_64            1
#define SMALLEST_POWER_OF_TWO_SP_64      64
#define SLOW_REWEIGHING_SP_64            0

// PADDED_STATE_COUNT == 80
#define PATTERN_BLOCK_SIZE_SP_80         8
#define MATRIX_BLOCK_SIZE_SP_80          8
#define BLOCK_PEELING_SIZE_SP_80         8
#define IS_POWER_OF_TWO_SP_80            0
#define SMALLEST_POWER_OF_TWO_SP_80      128
#define SLOW_REWEIGHING_SP_80            1

// PADDED_STATE_COUNT == 128
#define PATTERN_BLOCK_SIZE_SP_128        4
#define MATRIX_BLOCK_SIZE_SP_128         8
#define BLOCK_PEELING_SIZE_SP_128        2
#define IS_POWER_OF_TWO_SP_128           1
#define SMALLEST_POWER_OF_TWO_SP_128     128
#define SLOW_REWEIGHING_SP_128           1
 
// PADDED_STATE_COUNT == 192
#define PATTERN_BLOCK_SIZE_SP_192        2
#define MATRIX_BLOCK_SIZE_SP_192         8
#define BLOCK_PEELING_SIZE_SP_192        2
#define IS_POWER_OF_TWO_SP_192           0
#define SMALLEST_POWER_OF_TWO_SP_192     256
#define SLOW_REWEIGHING_SP_192           1

// DOUBLE PRECISION definitions   TODO None of these have been checked

// PADDED_STATE_COUNT == 4
#define PATTERN_BLOCK_SIZE_DP_4          16
#define MATRIX_BLOCK_SIZE_DP_4           8
#define BLOCK_PEELING_SIZE_DP_4          8
#define IS_POWER_OF_TWO_DP_4             1
#define SMALLEST_POWER_OF_TWO_DP_4       4
#define SLOW_REWEIGHING_DP_4             0

// PADDED_STATE_COUNT == 16
#define PATTERN_BLOCK_SIZE_DP_16         8
#define MATRIX_BLOCK_SIZE_DP_16          8
#define BLOCK_PEELING_SIZE_DP_16         8
#define IS_POWER_OF_TWO_DP_16            1
#define SMALLEST_POWER_OF_TWO_DP_16      16
#define SLOW_REWEIGHING_DP_16            0

// PADDED_STATE_COUNT == 32
#define PATTERN_BLOCK_SIZE_DP_32         8
#define MATRIX_BLOCK_SIZE_DP_32          8
#define BLOCK_PEELING_SIZE_DP_32         8
#define IS_POWER_OF_TWO_DP_32            1
#define SMALLEST_POWER_OF_TWO_DP_32      32
#define SLOW_REWEIGHING_DP_32            0

// PADDED_STATE_COUNT == 48
#define PATTERN_BLOCK_SIZE_DP_48         8
#define MATRIX_BLOCK_SIZE_DP_48          8
#define BLOCK_PEELING_SIZE_DP_48         8
#define IS_POWER_OF_TWO_DP_48            0
#define SMALLEST_POWER_OF_TWO_DP_48      64
#define SLOW_REWEIGHING_DP_48            0

// PADDED_STATE_COUNT == 64
#define PATTERN_BLOCK_SIZE_DP_64         8
#define MATRIX_BLOCK_SIZE_DP_64          8
#define BLOCK_PEELING_SIZE_DP_64         4 // Can use 8 on GTX480
#define IS_POWER_OF_TWO_DP_64            1
#define SMALLEST_POWER_OF_TWO_DP_64      64
#define SLOW_REWEIGHING_DP_64            0

// PADDED_STATE_COUNT == 80
#define PATTERN_BLOCK_SIZE_DP_80         8
#define MATRIX_BLOCK_SIZE_DP_80          8
#define BLOCK_PEELING_SIZE_DP_80         4 // Can use 8 on GTX480
#define IS_POWER_OF_TWO_DP_80            0
#define SMALLEST_POWER_OF_TWO_DP_80      128
#define SLOW_REWEIGHING_DP_80            1

// PADDED_STATE_COUNT == 128
#define PATTERN_BLOCK_SIZE_DP_128        4
#define MATRIX_BLOCK_SIZE_DP_128         8
#define BLOCK_PEELING_SIZE_DP_128        2
#define IS_POWER_OF_TWO_DP_128           1
#define SMALLEST_POWER_OF_TWO_DP_128     128
#define SLOW_REWEIGHING_DP_128           1

// PADDED_STATE_COUNT == 192
#define PATTERN_BLOCK_SIZE_DP_192        2
#define MATRIX_BLOCK_SIZE_DP_192         8
#define BLOCK_PEELING_SIZE_DP_192        2
#define IS_POWER_OF_TWO_DP_192           0
#define SMALLEST_POWER_OF_TWO_DP_192     256
#define SLOW_REWEIGHING_DP_192           1

#ifdef STATE_COUNT
#if (STATE_COUNT == 4 || STATE_COUNT == 16 || STATE_COUNT == 32 || STATE_COUNT == 48 || STATE_COUNT == 64 || STATE_COUNT == 80 || STATE_COUNT == 128 || STATE_COUNT == 192)
	#define PADDED_STATE_COUNT	STATE_COUNT
#else
	#error *** Precompiler directive state count not defined ***
#endif
#endif

// Need nested macros: first for replacement, second for evaluation
#define GET2_NO_CALL(x, y)	x##_##y
#define	GET2_VALUE(x, y)		GET2_NO_CALL(x, y)
#define GET_NO_CALL(x, y, z)	x##_##y##_##z
#define	GET_VALUE(x, y, z)		GET_NO_CALL(x, y, z)

#ifdef DOUBLE_PRECISION
	#define PREC	DP
#else
	#define	PREC	SP
#endif

#define PATTERN_BLOCK_SIZE		GET_VALUE(PATTERN_BLOCK_SIZE, PREC, PADDED_STATE_COUNT)
#define MATRIX_BLOCK_SIZE		GET_VALUE(MATRIX_BLOCK_SIZE, PREC, PADDED_STATE_COUNT)
#define BLOCK_PEELING_SIZE		GET_VALUE(BLOCK_PEELING_SIZE, PREC, PADDED_STATE_COUNT)
#define CHECK_IS_POWER_OF_TWO	GET_VALUE(IS_POWER_OF_TWO, PREC, PADDED_STATE_COUNT)
#if (CHECK_IS_POWER_OF_TWO == 1)
	#define IS_POWER_OF_TWO
#endif
#define SMALLEST_POWER_OF_TWO	GET_VALUE(SMALLEST_POWER_OF_TWO, PREC, PADDED_STATE_COUNT)
#define CHECK_SLOW_REWEIGHING	GET_VALUE(SLOW_REWEIGHING, PREC, PADDED_STATE_COUNT)
#if (CHECK_SLOW_REWEIGHING == 1)
	#define SLOW_REWEIGHING
#endif


// State count independent
#define SUM_SITES_BLOCK_SIZE_DP	128
#define SUM_SITES_BLOCK_SIZE_SP	128
#define MULTIPLY_BLOCK_SIZE_DP	16
#define MULTIPLY_BLOCK_SIZE_SP	16

#define SUM_SITES_BLOCK_SIZE 	GET2_VALUE(SUM_SITES_BLOCK_SIZE, PREC)
#define MULTIPLY_BLOCK_SIZE 	GET2_VALUE(MULTIPLY_BLOCK_SIZE, PREC)

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
