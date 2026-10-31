#ifndef BEAGLE_TENSOR_CORE_MACROS_H
#define BEAGLE_TENSOR_CORE_MACROS_H

// WMMA tile dimensions (f64 m8n8k4)
#define WMMA_M 8
#define WMMA_N 8
#define WMMA_K 4

#define MODULUS_NON_NEGATIVE(A,B) (((A) % (B) + (B)) % (B))

// Indices to permute ShM for sMatrix
// X -> threadIdx.x or state and Y -> threadIdx.y or patIdx
// (int(X/8): Splits 32 values into groups of 8.
// ((Y & 1) * -2 + 1)): For strip-mined layout: If patIdx is even increment by 1 else by -1
// & 0x03 To cycle within the limits [0,1,2,3] i.e., [0, ... , PADDED_STATE_COUNT/WMMA_M]
#define GET_SMEM_ROW_SMATRIX(X) ((X / WMMA_K) & 0x03)
#define GET_BANK_GROUP_SMATRIX(X,Y) MODULUS_NON_NEGATIVE( (Y + (X/WMMA_K) * (0 - (Y & 1) | 1)), (PADDED_STATE_COUNT/WMMA_K)) // 0x03 should be generalized to & PADDED_STATE_COUNT/WMMA_M - 1
#define GET_SMEM_COL_SMATRIX(X,Y) (GET_BANK_GROUP_SMATRIX(X,Y) * WMMA_K + (X % WMMA_K))
#define GET_SMEM_OFFSET_SMATRIX(X,Y) (GET_SMEM_ROW_SMATRIX(X) * PADDED_STATE_COUNT + GET_SMEM_COL_SMATRIX(X, Y))
//#define GET_SMEM_OFFSET_SMATRIX(X,Y) X + Y * PADDED_STATE_COUNT

// Indices to permute ShM for partials
// X -> threadIdx.x or state and Y -> threadIdx.y or patIdx
// (int(X/8): Splits 32 values into groups of 4.
// ((Y & 1) * -2 + 1)): For strip-mined layout: If patIdx is even increment by 1 else by -1
// & 0x07 To cycle within the limits [0,1,2,3,4,5,6,7] i.e., [0, ... , PADDED_STATE_COUNT/WMMA_K]
#define GET_SMEM_ROW_PARTIALS(X, Y) (((X / WMMA_K) + ((Y / (PADDED_STATE_COUNT / WMMA_K) ) * (PADDED_STATE_COUNT / WMMA_K)) ) & 0x07)
#define GET_BANK_GROUP_PARTIALS(X,Y) MODULUS_NON_NEGATIVE( (Y + (X/WMMA_K) * (0 - (Y & 1) | 1)), (PADDED_STATE_COUNT/WMMA_K)) // 0x07 should be generalized to & PADDED_STATE_COUNT/WMMA_K - 1
#define GET_SMEM_COL_PARTIALS(X,Y) (GET_BANK_GROUP_PARTIALS(X,Y) * WMMA_K + (X % WMMA_K))
#define GET_SMEM_OFFSET_PARTIALS(X,Y) (GET_SMEM_ROW_PARTIALS(X, Y) * PADDED_STATE_COUNT + GET_SMEM_COL_PARTIALS(X, Y))
//#define GET_SMEM_OFFSET_PARTIALS(X,Y) X + Y * PADDED_STATE_COUNT

// Warp identification — declares warpSize, warpState, warpPattern, warpsPerPattern, warpIdx, laneid
// in the enclosing scope. `state` and `patIdx` must already exist.
#define TENSOR_CORE_WARP_SETUP()                                                      \
    int warpSize = 32;                                                                \
    int warpState = state / warpSize;                                                 \
    int warpPattern = patIdx;                                                         \
    float warpsPerPattern = (float) PADDED_STATE_COUNT / warpSize;                    \
    int warpIdx = warpState + warpPattern * warpsPerPattern;                          \
    int laneid = (state + patIdx * PADDED_STATE_COUNT) % warpSize;

// Double-precision m8n8k4 MMA
#define MMA_F64_M8N8K4(A, B, RES1, RES2)                                              \
    asm("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%4,%5};\n" \
        : "=d"(RES1), "=d"(RES2)                                                      \
        : "d"(A), "d"(B), "d"(RES1), "d"(RES2))

#endif //BEAGLE_TENSOR_CORE_MACROS_H
