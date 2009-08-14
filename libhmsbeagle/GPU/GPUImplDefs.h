
/*
 * @author Marc Suchard
 * @author Dat Huynh
 * @author Daniel Ayres
 */

#ifndef __GPUImplDefs__
#define __GPUImplDefs__

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

//#define BEAGLE_DEBUG_FLOW
//#define BEAGLE_DEBUG_VALUES
//#define BEAGLE_DEBUG_SYNCH

/* Definition of REAL can be switched between 'double' and 'float' */
#ifdef DOUBLE_PRECISION
    #define REAL    double
#else
    #define REAL    float
#endif

#define SIZE_REAL   sizeof(REAL)
#define INT         int
#define SIZE_INT    sizeof(INT)

/* Compiler definitions
 *
 * STATE_COUNT        - Controlled by Makefile
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
 * SMALLEST_POWER_OF_TWO - Smallest power of 2 greater than or equal to PADDED_STATE_COUNT
 *    (if not already a power of 2)
 *
 */

#if (STATE_COUNT == 4)
    #define PADDED_STATE_COUNT  4
#else
#if (STATE_COUNT <= 16) // else if
    #define PADDED_STATE_COUNT  16
#else
#if (STATE_COUNT <= 32) // else if
    #define PADDED_STATE_COUNT  32
#else
//#if (STATE_COUNT <= 48) // else if
//	#define PADDED_STATE_COUNT	48
//#else
#if (STATE_COUNT <= 64) // else if
    #define PADDED_STATE_COUNT  64
#else
#if (STATE_COUNT <= 128) // else if
    #define PADDED_STATE_COUNT  128
#else
#if (STATE_COUNT <= 192) // else if
    #define PADDED_STATE_COUNT  192
#endif
#endif
#endif
#endif
#endif
#endif
//#endif

#define PADDED_STATES   PADDED_STATE_COUNT - STATE_COUNT
#define IS_POWER_OF_TWO

#if (PADDED_STATE_COUNT == 4)
    #define PATTERN_BLOCK_SIZE  16
#endif

// TODO Find optimal settings for PADDED_STATE_COUNT == 32

#if (PADDED_STATE_COUNT == 64)
    #ifdef DOUBLE_PRECISION
        #define PATTERN_BLOCK_SIZE  8
        #define BLOCK_PEELING_SIZE  4
    #else
        #define PATTERN_BLOCK_SIZE  8
        #define BLOCK_PEELING_SIZE  8
    #endif
#endif

#if (PADDED_STATE_COUNT == 128)
    #define PATTERN_BLOCK_SIZE  4
    #define BLOCK_PEELING_SIZE  2 // seems small, but yields 100% occupancy
    #define SLOW_REWEIGHING
#endif

#if (PADDED_STATE_COUNT == 192)
    #define PATTERN_BLOCK_SIZE  2
    #define BLOCK_PEELING_SIZE  2
    #define SLOW_REWEIGHING
    #define SMALLEST_POWER_OF_TWO 256
    #undef IS_POWER_OF_TWO
#endif

/* Defaults */

#ifndef PATTERN_BLOCK_SIZE
    #define PATTERN_BLOCK_SIZE  16
#endif

#ifndef MATRIX_BLOCK_SIZE
    #define MATRIX_BLOCK_SIZE   8
#endif

#ifndef BLOCK_PEELING_SIZE
    #define BLOCK_PEELING_SIZE  8
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
    Dim3Int(unsigned int x = 1,
            unsigned int y = 1,
            unsigned int z = 1) : x(x), y(y), z(z) {}
#endif /* __cplusplus */
};

#endif // __GPUImplDefs__
