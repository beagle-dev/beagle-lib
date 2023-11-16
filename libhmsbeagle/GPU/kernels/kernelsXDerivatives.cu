}

#include <mma.h>

extern "C" {
KW_GLOBAL_KERNEL void kernelPartialsPartialsGrowing(KW_GLOBAL_VAR REAL* KW_RESTRICT partials1,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT partials2,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT partials3,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT matrices1,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT matrices2,
                                                    int totalPatterns) {
#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    todo(); // TODO
#else // GPU implementation
    DETERMINE_INDICES_X_GPU();

    KW_GLOBAL_VAR REAL* KW_RESTRICT matrix1 = matrices1 + deltaMatrix; /* Points to *this* matrix */
    KW_GLOBAL_VAR REAL* KW_RESTRICT matrix2 = matrices2 + deltaMatrix;

    /* Load values into shared memory */
    KW_LOCAL_MEM REAL sMatrix1[BLOCK_PEELING_SIZE][PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sMatrix2[BLOCK_PEELING_SIZE][PADDED_STATE_COUNT];

    KW_LOCAL_MEM REAL sPartials1[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sPartials2[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];

    int y = deltaPartialsByState + deltaPartialsByMatrix;

    /* copy PADDED_STATE_COUNT*PATTERN_BLOCK_SIZE lengthed partials */
    /* These are all coherent global memory reads; checked in Profiler */
    if (pattern < totalPatterns) {
        sPartials1[patIdx][state] = partials1[y + state];
        sPartials2[patIdx][state] = partials2[y + state];
    } else {
        sPartials1[patIdx][state] = 0;
        sPartials2[patIdx][state] = 0;
    }

    REAL sum2 = 0;
    for (int i = 0; i < PADDED_STATE_COUNT; i += BLOCK_PEELING_SIZE) {
        /* load one row of matrices */
        if (patIdx < BLOCK_PEELING_SIZE) {
            /* These are all coherent global memory reads. */
            sMatrix2[patIdx][state] = matrix2[patIdx * PADDED_STATE_COUNT + state];
            /* sMatrix now filled with starting in state and ending in i */
            matrix2 += BLOCK_PEELING_SIZE * PADDED_STATE_COUNT;
        }

        KW_LOCAL_FENCE;

        for(int j = 0; j < BLOCK_PEELING_SIZE; j++) {
            FMA(sMatrix2[j][state],  sPartials2[patIdx][i + j], sum2);
        }

        KW_LOCAL_FENCE;
    }

    sPartials1[patIdx][state] *= sum2;

    KW_LOCAL_FENCE; // TODO Remove?

    REAL sum1 = 0;
    for (int i = 0; i < PADDED_STATE_COUNT; i += BLOCK_PEELING_SIZE) {
        /* load one row of matrices */
        if (patIdx < BLOCK_PEELING_SIZE) {
            /* These are all coherent global memory reads. */
            sMatrix1[patIdx][state] = matrix1[patIdx * PADDED_STATE_COUNT + state];
            /* sMatrix now filled with starting in state and ending in i */
            matrix1 += BLOCK_PEELING_SIZE * PADDED_STATE_COUNT;
        }

        KW_LOCAL_FENCE;

        for(int j = 0; j < BLOCK_PEELING_SIZE; j++) {
            FMA(sMatrix1[j][state],  sPartials1[patIdx][i + j], sum1);
        }

        KW_LOCAL_FENCE;
    }

    if (pattern < totalPatterns) {
        partials3[u] = sum1;
    }
#endif
}

KW_GLOBAL_KERNEL void kernelPartialsPartialsGrowingTensorCores(KW_GLOBAL_VAR REAL* KW_RESTRICT partials1,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT partials2,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT partials3,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT matrices1,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT matrices2,
//                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT tmpAcc,
                                                    int totalPatterns) {
#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    todo(); // TODO
#else // GPU implementation
//    DETERMINE_INDICES_X_GPU();
    const int NEW_PATTERN_BLOCK_SIZE = PATTERN_BLOCK_SIZE;
    int state = KW_LOCAL_ID_0;
    int patIdx = KW_LOCAL_ID_1;
    int pattern = __umul24(KW_GROUP_ID_0,NEW_PATTERN_BLOCK_SIZE) + patIdx;
    int matrix = KW_GROUP_ID_1;
    int patternCount = totalPatterns;
    int deltaPartialsByState = pattern * PADDED_STATE_COUNT;
    int deltaPartialsByMatrix = matrix * PADDED_STATE_COUNT * patternCount;
    int deltaMatrix = matrix * PADDED_STATE_COUNT * PADDED_STATE_COUNT;
    int u = state + deltaPartialsByState + deltaPartialsByMatrix;
    int patternBlock = __umul24(KW_GROUP_ID_0,PATTERN_BLOCK_SIZE);

    const int WMMA_M = 8;
    const int WMMA_N = 8;
    const int WMMA_K = 4;
    const int PATTERN_SPAN = NEW_PATTERN_BLOCK_SIZE/2;
    const int MEM_OFFSET = PATTERN_SPAN * PADDED_STATE_COUNT;

    int warpSize = 32; // TODO: Check if its better to get this from Cuda API
    int warpState = state/warpSize;
    int warpPattern = patIdx;
    int warpIdx = warpState + warpPattern * (PADDED_STATE_COUNT/warpSize);

    int sMatrixRow, partialsCol, sMatrixCol, partialsRow;

//   TODO: Declare right before usage
    double a1, b1, a2,b2, res11 = 0, res12 = 0, res21 = 0, res22 = 0;

    int partialsOffset = warpIdx % (PATTERN_BLOCK_SIZE / WMMA_N);

    // Indices to permute ShM for sMatrix
    // X -> threadIdx.x or state and Y -> threadIdx.y or patIdx
    // (int(X/8): Splits 32 values into groups of 8.
    // ((Y & 1) * -2 + 1)): For strip-mined layout: If patIdx is even increment by 1 else by -1
    // & 0x03 To cycle within the limits [0,1,2,3] i.e., [0, ... , PADDED_STATE_COUNT/WMMA_M]
#define GET_SMEM_ROW_SMATRIX(X) ((X / WMMA_K) & 0x03)
#define GET_BANK_GROUP_SMATRIX(X,Y) ((Y + (X/WMMA_K) * ((Y & 1) * -2 + 1)) & ((PADDED_STATE_COUNT/WMMA_K) - 1)) // 0x03 should be generalized to & PADDED_STATE_COUNT/WMMA_M - 1
#define GET_SMEM_COL_SMATRIX(X,Y) (GET_BANK_GROUP_SMATRIX(X,Y) * WMMA_K + (X % WMMA_K))
#define GET_SMEM_OFFSET_SMATRIX(X,Y) (GET_SMEM_ROW_SMATRIX(X) * PADDED_STATE_COUNT + GET_SMEM_COL_SMATRIX(X, Y))

    // Indices to permute ShM for partials
    // X -> threadIdx.x or state and Y -> threadIdx.y or patIdx
    // (int(X/8): Splits 32 values into groups of 4.
    // ((Y & 1) * -2 + 1)): For strip-mined layout: If patIdx is even increment by 1 else by -1
    // & 0x07 To cycle within the limits [0,1,2,3,4,5,6,7] i.e., [0, ... , PADDED_STATE_COUNT/WMMA_K]
#define GET_SMEM_ROW_PARTIALS(X) ((X / WMMA_K) & 0x07)
#define GET_BANK_GROUP_PARTIALS(X,Y) ((Y + (X/WMMA_K) * ((Y & 1) * -2 + 1)) & ((PADDED_STATE_COUNT/WMMA_K) - 1)) // 0x07 should be generalized to & PADDED_STATE_COUNT/WMMA_K - 1
#define GET_SMEM_COL_PARTIALS(X,Y) (GET_BANK_GROUP_PARTIALS(X,Y) * WMMA_K + (X % WMMA_K))
#define GET_SMEM_OFFSET_PARTIALS(X,Y) (GET_SMEM_ROW_PARTIALS(X) * PADDED_STATE_COUNT + GET_SMEM_COL_PARTIALS(X, Y))

    KW_GLOBAL_VAR REAL* KW_RESTRICT matrix1 = matrices1 + deltaMatrix; /* Points to *this* matrix */
    KW_GLOBAL_VAR REAL* KW_RESTRICT matrix2 = matrices2 + deltaMatrix;

    /* Load values into shared memory */
    KW_LOCAL_MEM REAL sMatrix1[4 * PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sMatrix2[4 * PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sPartials1[PADDED_STATE_COUNT * PATTERN_BLOCK_SIZE];
    KW_LOCAL_MEM REAL sPartials2[PADDED_STATE_COUNT * PATTERN_BLOCK_SIZE];

    int y = patternBlock * PADDED_STATE_COUNT + deltaPartialsByMatrix;

    // Load PADDED_STATE_COUNT * PATTERN_BLOCK_SIZE partials
    if(pattern < totalPatterns) {
        sPartials1[GET_SMEM_OFFSET_PARTIALS(state, patIdx)] = partials1[y + patIdx * PADDED_STATE_COUNT + state];
        sPartials2[GET_SMEM_OFFSET_PARTIALS(state, patIdx)] = partials2[y + patIdx * PADDED_STATE_COUNT + state];
    } else {
        sPartials1[GET_SMEM_OFFSET_PARTIALS(state, patIdx)] = 0;
        sPartials2[GET_SMEM_OFFSET_PARTIALS(state, patIdx)] = 0;
    }

    int pattern_span = patIdx + 4;
    if(pattern + 4 < totalPatterns) {
        sPartials1[GET_SMEM_OFFSET_PARTIALS(state, pattern_span)] = partials1[y + (pattern_span) * PADDED_STATE_COUNT + state];
        sPartials2[GET_SMEM_OFFSET_PARTIALS(state, pattern_span)] = partials2[y + (pattern_span) * PADDED_STATE_COUNT + state];
    } else {
        sPartials1[GET_SMEM_OFFSET_PARTIALS(state, pattern_span)] = 0;
        sPartials2[GET_SMEM_OFFSET_PARTIALS(state, pattern_span)] = 0;
    }

//    KW_LOCAL_FENCE;

    for (int i = 0; i < PADDED_STATE_COUNT; i += WMMA_K) {
        sMatrixRow = warpIdx % (PADDED_STATE_COUNT / WMMA_M);
        sMatrixCol = i;
        partialsRow = i;
        partialsCol = warpIdx / (PADDED_STATE_COUNT / WMMA_M);

        sMatrix2[GET_SMEM_OFFSET_SMATRIX(state, patIdx)] = matrix2[sMatrixCol * PADDED_STATE_COUNT + patIdx * PADDED_STATE_COUNT + state];

        KW_LOCAL_FENCE;

        // reg_row* and reg_col* are according to memory layout in ShM
        int laneid = state % 32;

        int reg_row = state % 4;
        int reg_col = (warpIdx * WMMA_M) + (laneid / 4);

        // TODO: More book keeping if PATTERN_BLOCK_SIZE > WMMA_M
        int reg_row_partials = laneid / 4;
        int reg_col_partials = partialsRow + state % 4;

        a2 = sMatrix2[GET_SMEM_OFFSET_SMATRIX(reg_col, reg_row)];
        b2 = sPartials2[GET_SMEM_OFFSET_PARTIALS(reg_col_partials, reg_row_partials)];

        asm("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
            : "=d"(res21), "=d"(res22)
            : "d"(a2), "d"(b2), "d"(res21), "d"(res22));

        KW_LOCAL_FENCE;
    }

    // Map results of MMA operation (8x8) to permuted shared memory for elementwise multiplication of partials
    int partials1Index = patIdx * (PADDED_STATE_COUNT/warpSize) * WMMA_M + ((state * 2) % 8) * PADDED_STATE_COUNT + (state * 2)/8;
    int partials1State = partials1Index % PADDED_STATE_COUNT;
    int partials1PatIdx = partials1Index / PADDED_STATE_COUNT;

    sPartials1[GET_SMEM_OFFSET_PARTIALS(partials1State, partials1PatIdx)] = sPartials1[GET_SMEM_OFFSET_PARTIALS(partials1State, partials1PatIdx)] * res21;
    sPartials1[GET_SMEM_OFFSET_PARTIALS(partials1State, partials1PatIdx + 1)] = sPartials1[GET_SMEM_OFFSET_PARTIALS(partials1State, partials1PatIdx + 1)] * res22;


    for (int i = 0; i < PADDED_STATE_COUNT; i += WMMA_K) {
        sMatrixRow = warpIdx % (PADDED_STATE_COUNT / WMMA_M);
        sMatrixCol = i;
        partialsRow = i;
        partialsCol = warpIdx / (PADDED_STATE_COUNT / WMMA_M);

        sMatrix1[GET_SMEM_OFFSET_SMATRIX(state, patIdx)] = matrix1[sMatrixCol * PADDED_STATE_COUNT + patIdx * PADDED_STATE_COUNT + state];
        KW_LOCAL_FENCE;

        // reg_row* and reg_col* are according to memory layout in ShM
        int laneid = state % 32;

        int reg_row = state % 4;
        int reg_col = (warpIdx * WMMA_M) + (laneid / 4);

        // TODO: More book keeping if PATTERN_BLOCK_SIZE > WMMA_M
        int reg_row_partials = laneid / 4;
        int reg_col_partials = partialsRow + state % 4;

        a1 = sMatrix1[GET_SMEM_OFFSET_SMATRIX(reg_col, reg_row)];
        b1 = sPartials1[GET_SMEM_OFFSET_PARTIALS(reg_col_partials, reg_row_partials)];

        asm("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
            : "=d"(res11), "=d"(res12)
            : "d"(a1), "d"(b1), "d"(res11), "d"(res12));

        KW_LOCAL_FENCE;
    }

    u = patternBlock * PADDED_STATE_COUNT + deltaPartialsByMatrix;

    if (patternBlock + ((state * 2) % 8) < totalPatterns)
        partials3[u + patIdx * (PADDED_STATE_COUNT/warpSize) * WMMA_M + ((state * 2) % 8) * PADDED_STATE_COUNT + (state * 2)/8] = res11;

//    tmpAcc[patIdx * PADDED_STATE_COUNT + state] = u + patIdx * WMMA_M + ((state * 2) % 8) * PADDED_STATE_COUNT + (state * 2)/8;

    if (patternBlock + ((state * 2) % 8) + 1 < totalPatterns)
        partials3[u + PADDED_STATE_COUNT + patIdx * (PADDED_STATE_COUNT/warpSize) * WMMA_M + ((state * 2) % 8) * PADDED_STATE_COUNT + (state * 2)/8] = res12;
#endif
}


KW_GLOBAL_KERNEL void kernelPartialsStatesGrowing(KW_GLOBAL_VAR REAL* KW_RESTRICT partials1,
                                                  KW_GLOBAL_VAR int*  KW_RESTRICT states2,
                                                  KW_GLOBAL_VAR REAL* KW_RESTRICT partials3,
                                                  KW_GLOBAL_VAR REAL* KW_RESTRICT matrices1,
                                                  KW_GLOBAL_VAR REAL* KW_RESTRICT matrices2,
                                                  int totalPatterns) {
#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    todo(); // TODO
#else // GPU implementation
    DETERMINE_INDICES_X_GPU();

    KW_GLOBAL_VAR REAL* KW_RESTRICT matrix1 = matrices1 + deltaMatrix; /* Points to *this* matrix */

    /* Load values into shared memory */
    KW_LOCAL_MEM REAL sMatrix1[BLOCK_PEELING_SIZE][PADDED_STATE_COUNT];

    KW_LOCAL_MEM REAL sPartials1[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];

    int y = deltaPartialsByState + deltaPartialsByMatrix;

    /* copy PADDED_STATE_COUNT*PATTERN_BLOCK_SIZE lengthed partials */
    /* These are all coherent global memory reads; checked in Profiler */
    if (pattern < totalPatterns) {
        sPartials1[patIdx][state] = partials1[y + state];
    } else {
        sPartials1[patIdx][state] = 0;
    }
    
    REAL sum2 = 1;
    if (pattern < totalPatterns) { // Remove padded threads!
        int state2 = states2[pattern];
        KW_GLOBAL_VAR REAL* KW_RESTRICT matrix2 = matrices2 + deltaMatrix + state2 * PADDED_STATE_COUNT;
        if (state2 < PADDED_STATE_COUNT) {
            sum2 = matrix2[state];
        }
    }

    sPartials1[patIdx][state] *= sum2;

    KW_LOCAL_FENCE; // TODO Remove?

    REAL sum1 = 0;
    for (int i = 0; i < PADDED_STATE_COUNT; i += BLOCK_PEELING_SIZE) {
        /* load one row of matrices */
        if (patIdx < BLOCK_PEELING_SIZE) {
            /* These are all coherent global memory reads. */
            sMatrix1[patIdx][state] = matrix1[patIdx * PADDED_STATE_COUNT + state];
            /* sMatrix now filled with starting in state and ending in i */
            matrix1 += BLOCK_PEELING_SIZE * PADDED_STATE_COUNT;
        }

        KW_LOCAL_FENCE;

        for(int j = 0; j < BLOCK_PEELING_SIZE; j++) {
            FMA(sMatrix1[j][state],  sPartials1[patIdx][i + j], sum1);
        }

        KW_LOCAL_FENCE;
    }

    if (pattern < totalPatterns) {
        partials3[u] = sum1;
    }
#endif
}

KW_GLOBAL_KERNEL void kernelPartialsPartialsEdgeFirstDerivatives(KW_GLOBAL_VAR REAL* KW_RESTRICT out,
                                                                 KW_GLOBAL_VAR REAL* KW_RESTRICT partials0,
                                                                 KW_GLOBAL_VAR REAL* KW_RESTRICT matrices0,
                                                                 KW_GLOBAL_VAR unsigned int* KW_RESTRICT instructions,
                                                                 KW_GLOBAL_VAR REAL* KW_RESTRICT weights,
                                                                 int skip,
                                                                 int totalPatterns, int categoryCount) {
#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    todo(); // TODO
#else // GPU implementation

#define NEW_BLOCK_PEELING_SIZE PATTERN_BLOCK_SIZE

    int state = KW_LOCAL_ID_0;
    int patIdx = KW_LOCAL_ID_1;
    int pattern = KW_GROUP_ID_0 * NEW_BLOCK_PEELING_SIZE + patIdx;

    int node = KW_GROUP_ID_1 + skip;
    int instructionOffset = node * 3;

    unsigned int partials1Offset = instructions[instructionOffset + 0];
    unsigned int partials2Offset = instructions[instructionOffset + 1];
    unsigned int matrices1Offset = instructions[instructionOffset + 2];

    KW_LOCAL_MEM REAL sMatrix2[NEW_BLOCK_PEELING_SIZE][PADDED_STATE_COUNT];

    KW_LOCAL_MEM REAL sPartials1[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sPartials2[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];

    /* TODO: Currently assumes MATRIX_BLOCK_SIZE >> matrixCount */\
    KW_LOCAL_MEM REAL sWeights[MATRIX_BLOCK_SIZE];

    for (int c = 0; c < categoryCount; c += KW_LOCAL_SIZE_0) {
        int x = c + KW_LOCAL_ID_0;
        if (x < categoryCount) {
            sWeights[x] = weights[x];
        }
    }

    KW_LOCAL_FENCE;

    REAL numerator = 0;
    REAL denominator = 0;

    REAL lPartial1;
    REAL lPartial2;

    for (int c = 0; c < categoryCount; ++c) {

        KW_GLOBAL_VAR REAL* KW_RESTRICT partials1 = partials0 + partials1Offset + totalPatterns * PADDED_STATE_COUNT * c;
        KW_GLOBAL_VAR REAL* KW_RESTRICT partials2 = partials0 + partials2Offset + totalPatterns * PADDED_STATE_COUNT * c;
        KW_GLOBAL_VAR REAL* KW_RESTRICT matrix2 = matrices0 + matrices1Offset + PADDED_STATE_COUNT * PADDED_STATE_COUNT * c;

        /* copy PADDED_STATE_COUNT*PATTERN_BLOCK_SIZE lengthed partials */
        /* These are all coherent global memory reads; checked in Profiler */
        if (pattern<totalPatterns) {
            lPartial1 = partials1[pattern * PADDED_STATE_COUNT + state];
            sPartials2[patIdx][state] = lPartial2 = partials2[pattern * PADDED_STATE_COUNT + state];
        } else {
            lPartial1 = 0;
            sPartials2[patIdx][state] = lPartial2 = 0;
        }

        FMA(lPartial1, lPartial2 * sWeights[c], denominator);

        REAL sum2 = 0;
        for (int i = 0; i < PADDED_STATE_COUNT; i += NEW_BLOCK_PEELING_SIZE) {
            /* load one row of matrices */
            if (patIdx < NEW_BLOCK_PEELING_SIZE) {
                /* These are all coherent global memory reads. */
                sMatrix2[patIdx][state] = matrix2[patIdx * PADDED_STATE_COUNT + state];
                /* sMatrix now filled with starting in state and ending in i */
                matrix2 += NEW_BLOCK_PEELING_SIZE * PADDED_STATE_COUNT;
            }
            KW_LOCAL_FENCE;

            // TODO 2nd check is unncessary for stateCount >= 16
            for (int j = 0; (j < NEW_BLOCK_PEELING_SIZE) && (i + j < PADDED_STATE_COUNT); j++) {
                FMA(sMatrix2[j][state],  sPartials2[patIdx][i + j], sum2);
            }
            KW_LOCAL_FENCE;
        }

        FMA(lPartial1, sum2 * sWeights[c], numerator);

//        partials1 += totalPatterns * PADDED_STATE_COUNT;
//        partials2 += totalPatterns * PADDED_STATE_COUNT;
    }

    sPartials1[patIdx][state] = numerator;
    sPartials2[patIdx][state] = denominator;

    KW_LOCAL_FENCE;

#ifdef IS_POWER_OF_TWO
    // parallelized reduction *** only works for powers-of-2 ****
    for (int i = PADDED_STATE_COUNT / 2; i > 0; i >>= 1) {
        if (state < i) {
#else
    for (int i = SMALLEST_POWER_OF_TWO / 2; i > 0; i >>= 1) {
        if (state < i && state + i < PADDED_STATE_COUNT ) {
#endif // IS_POWER_OF_TWO
            sPartials1[patIdx][state] += sPartials1[patIdx][state + i];
            sPartials2[patIdx][state] += sPartials2[patIdx][state + i];
        }
        KW_LOCAL_FENCE;
    }

    // TODO Test this coalesced write code out
    int tx = KW_LOCAL_ID_0;
    if (tx < PATTERN_BLOCK_SIZE && patIdx == 0) { // Use first PATTERN_BLOCK_SIZE threads to write
        int site = KW_GROUP_ID_0 * NEW_BLOCK_PEELING_SIZE + tx;
        if (site < totalPatterns) {
            REAL numerator = sPartials1[tx][0];
            REAL denominator = sPartials2[tx][0];
            REAL ratio = 0.0;
            if (denominator != 0.0) {
                ratio = numerator / denominator;
            }
            out[totalPatterns * node + site] = ratio;
        }
    }

//    if (pattern < totalPatterns) {
//        if (state == 0) {
//            out[totalPatterns * node + pattern] = sPartials1[patIdx][0] / sPartials2[patIdx][0]; // pre;
////            out[totalPatterns * node + pattern] = sPartials1[patIdx][0];  // Write numerator
////            out[totalPatterns * (KW_NUM_GROUPS_1 + node) + pattern] = sPartials2[patIdx][0]; // Write denomiator
//        }
//    }

#endif
}

KW_GLOBAL_KERNEL void kernelPartialsStatesEdgeFirstDerivatives(KW_GLOBAL_VAR REAL* KW_RESTRICT out,
                                                               KW_GLOBAL_VAR int*  KW_RESTRICT states0,
                                                               KW_GLOBAL_VAR REAL* KW_RESTRICT partials0,
                                                               KW_GLOBAL_VAR REAL* KW_RESTRICT matrices0,
                                                               KW_GLOBAL_VAR unsigned int* KW_RESTRICT instructions,
                                                               KW_GLOBAL_VAR REAL* KW_RESTRICT weights,
                                                               int skip,
                                                               int totalPatterns, int categoryCount) {
#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    todo(); // TODO
#else // GPU implementation

#define NEW_BLOCK_PEELING_SIZE PATTERN_BLOCK_SIZE

    int state = KW_LOCAL_ID_0;
    int patIdx = KW_LOCAL_ID_1;
    int pattern = KW_GROUP_ID_0 * NEW_BLOCK_PEELING_SIZE + patIdx;

    int node = KW_GROUP_ID_1 + skip;
    int instructionOffset = node * 3;

    unsigned int states1Offset   = instructions[instructionOffset + 0];
    unsigned int partials2Offset = instructions[instructionOffset + 1];
    unsigned int matrices1Offset = instructions[instructionOffset + 2];

    KW_LOCAL_MEM REAL sMatrix2[NEW_BLOCK_PEELING_SIZE][PADDED_STATE_COUNT];

    KW_LOCAL_MEM REAL sPartials1[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL sPartials2[PATTERN_BLOCK_SIZE][PADDED_STATE_COUNT];

    /* TODO: Currently assumes MATRIX_BLOCK_SIZE >> matrixCount */\
    KW_LOCAL_MEM REAL sWeights[MATRIX_BLOCK_SIZE];

    for (int c = 0; c < categoryCount; c += KW_LOCAL_SIZE_0) {
        int x = c + KW_LOCAL_ID_0;
        if (x < categoryCount) {
            sWeights[x] = weights[x];
        }
    }

    KW_LOCAL_FENCE;

    REAL numerator = 0;
    REAL denominator = 0;

    int lState1 = (pattern < totalPatterns) ?
            states0[states1Offset + pattern] : PADDED_STATE_COUNT;

    REAL lPartial1 = (lState1 >= PADDED_STATE_COUNT || state == lState1) ?
            1 : 0;

    REAL lPartial2;

    for (int c = 0; c < categoryCount; ++c) {

        KW_GLOBAL_VAR REAL* KW_RESTRICT partials2 = partials0 + partials2Offset + totalPatterns * PADDED_STATE_COUNT * c;
        KW_GLOBAL_VAR REAL* KW_RESTRICT matrix2 = matrices0 + matrices1Offset + PADDED_STATE_COUNT * PADDED_STATE_COUNT * c;

        /* copy PADDED_STATE_COUNT*PATTERN_BLOCK_SIZE lengthed partials */
        /* These are all coherent global memory reads; checked in Profiler */
        if (pattern<totalPatterns) {
            sPartials2[patIdx][state] = lPartial2 = partials2[pattern * PADDED_STATE_COUNT + state];
        } else {
            sPartials2[patIdx][state] = lPartial2 = 0;
        }

        FMA(lPartial1, lPartial2 * sWeights[c], denominator);

        REAL sum2 = 0;
        for (int i = 0; i < PADDED_STATE_COUNT; i += NEW_BLOCK_PEELING_SIZE) {
            /* load one row of matrices */
            if (patIdx < NEW_BLOCK_PEELING_SIZE) {
                /* These are all coherent global memory reads. */
                sMatrix2[patIdx][state] = matrix2[patIdx * PADDED_STATE_COUNT + state];
                /* sMatrix now filled with starting in state and ending in i */
                matrix2 += NEW_BLOCK_PEELING_SIZE * PADDED_STATE_COUNT;
            }
            KW_LOCAL_FENCE;

            // TODO 2nd check is unncessary for stateCount >= 16
            for (int j = 0; (j < NEW_BLOCK_PEELING_SIZE) && (i + j < PADDED_STATE_COUNT); j++) {
                FMA(sMatrix2[j][state],  sPartials2[patIdx][i + j], sum2);
            }
            KW_LOCAL_FENCE;
        }

        FMA(lPartial1, sum2 * sWeights[c], numerator);

//        partials1 += totalPatterns * PADDED_STATE_COUNT;
//        partials2 += totalPatterns * PADDED_STATE_COUNT;
    }

    sPartials1[patIdx][state] = numerator;
    sPartials2[patIdx][state] = denominator;

    KW_LOCAL_FENCE;

#ifdef IS_POWER_OF_TWO
    // parallelized reduction *** only works for powers-of-2 ****
    for (int i = PADDED_STATE_COUNT / 2; i > 0; i >>= 1) {
        if (state < i) {
#else
    for (int i = SMALLEST_POWER_OF_TWO / 2; i > 0; i >>= 1) {
        if (state < i && state + i < PADDED_STATE_COUNT ) {
#endif // IS_POWER_OF_TWO
            sPartials1[patIdx][state] += sPartials1[patIdx][state + i];
            sPartials2[patIdx][state] += sPartials2[patIdx][state + i];
        }
        KW_LOCAL_FENCE;
    }

    // TODO Test this coalesced write code out
    int tx = KW_LOCAL_ID_0;
    if (tx < PATTERN_BLOCK_SIZE && patIdx == 0) { // Use first PATTERN_BLOCK_SIZE threads to write
        int site = KW_GROUP_ID_0 * NEW_BLOCK_PEELING_SIZE + tx;
        if (site < totalPatterns) {
            REAL numerator = sPartials1[tx][0];
            REAL denominator = sPartials2[tx][0];
            REAL ratio = 0.0;
            if (denominator != 0.0) {
                ratio = numerator / denominator;
            }
            out[totalPatterns * node + site] = ratio;
        }
    }

#endif
}

KW_GLOBAL_KERNEL void kernelPartialsStatesCrossProducts(KW_GLOBAL_VAR REAL* KW_RESTRICT out,
                                                        const KW_GLOBAL_VAR int*  KW_RESTRICT states0,
                                                        const KW_GLOBAL_VAR REAL* KW_RESTRICT partials0,
                                                        const KW_GLOBAL_VAR REAL* KW_RESTRICT lengths0,
                                                        const KW_GLOBAL_VAR unsigned int* KW_RESTRICT instructions,
                                                        const KW_GLOBAL_VAR REAL* KW_RESTRICT inCategoryWeights,
                                                        const KW_GLOBAL_VAR REAL* KW_RESTRICT inPatternWeights,
                                                        const int skip,
                                                        const int totalPatterns,
                                                        const int totalNodes,
                                                        const int categoryCount,
                                                        const int rateOffset,
                                                        const int accumulate,
                                                        const int missingState) {

#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    todo(); // Not implemented
#else // GPU implementation
    const int tx = KW_LOCAL_ID_0;

    const int tx_i = KW_LOCAL_ID_0 >> 4;
    const int tx_j = KW_LOCAL_ID_0 & 0xf;

    const int stateBlock_i = KW_GROUP_ID_2 / (PADDED_STATE_COUNT / 16);
    const int stateBlock_j = KW_GROUP_ID_2 % (PADDED_STATE_COUNT / 16);

    const int patternBlockId = KW_GROUP_ID_0;
    const int nodeId = KW_GROUP_ID_1;

    const int numPatternBlocks = KW_NUM_GROUPS_0;
    const int numNodeBlocks = KW_NUM_GROUPS_1;

    KW_LOCAL_MEM REAL post[PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL pre[PADDED_STATE_COUNT];

    KW_LOCAL_MEM REAL innerProduct[PADDED_STATE_COUNT];

    KW_LOCAL_MEM REAL categoryRates[16]; // TODO Assumes kCategoryCount <= 16
    KW_LOCAL_MEM REAL categoryWeights[16]; // TODO Should put these into constant memory anyway

    if (tx < categoryCount) {
        categoryRates[tx] = lengths0[rateOffset + tx];
        categoryWeights[tx] = inCategoryWeights[tx];
    }

    const int i = tx_i + stateBlock_i * 16;
    const int j = tx_j + stateBlock_j * 16;

    REAL acrossPatterns = (REAL) 0.0;

    for (int node = nodeId;  // Just interleaved indexing
         node < totalNodes;
         node += numNodeBlocks) {

//        KW_LOCAL_FENCE; // TODO necessary?

        const int instructionOffset = (skip + node) * 2;
        const int statesOffset = instructions[instructionOffset + 0];
        const int preOffset = instructions[instructionOffset + 1];

        const REAL edgeLength = lengths0[skip + node];

        for (int pattern = patternBlockId;
             pattern < totalPatterns;
             pattern += numPatternBlocks) {

//            KW_LOCAL_FENCE; // TODO necessary?

            REAL patternDenominator = (REAL) 0.0;
            REAL numerator = (REAL) 0.0;

            const KW_GLOBAL_VAR int* KW_RESTRICT postStates = states0 + statesOffset;
            const int stateData = postStates[pattern];

            if (tx < PADDED_STATE_COUNT) {
                post[tx] = (tx == stateData | stateData >= missingState) ? (REAL) 1.0 : (REAL) 0.0;
            }

            for (int c = 0; c < categoryCount; ++c) {

                const KW_GLOBAL_VAR REAL* KW_RESTRICT prePartials  = partials0 + preOffset  + (pattern + totalPatterns * c) * PADDED_STATE_COUNT;

                if (tx < PADDED_STATE_COUNT) {
                    pre[tx] = prePartials[tx];  // Coalesced global memory read
                    innerProduct[tx] = pre[tx] * post[tx];
                }

                KW_LOCAL_FENCE;

                const REAL scale =  edgeLength * categoryRates[c]; // TODO Better in constant memory?
                const REAL weight = categoryWeights[c]; // TODO Better in constant memory?

#ifdef IS_POWER_OF_TWO
                // parallelized reduction *** only works for powers-of-2 ****
                for (int k = PADDED_STATE_COUNT / 2; k > 0; k >>= 1) {
                    if (tx < k) {
#else
                for (int k = SMALLEST_POWER_OF_TWO / 2; k > 0; k >>= 1) {
                    if (tx < k && tx + k < PADDED_STATE_COUNT ) {
#endif // IS_POWER_OF_TWO
                        innerProduct[tx] += innerProduct[tx + k];
                    }
                    KW_LOCAL_FENCE;
                }

                patternDenominator += innerProduct[0] * weight;
                numerator += pre[i] * post[j] * weight * scale;
            }

//            KW_LOCAL_FENCE; // TODO necessary?

            if (patternDenominator > (REAL) 0.0) {
                acrossPatterns += numerator * inPatternWeights[pattern] / patternDenominator;
            }

//            KW_LOCAL_FENCE;
        }
    }

//    KW_LOCAL_FENCE;

    const int destination = (nodeId * numPatternBlocks + patternBlockId) * PADDED_STATE_COUNT * PADDED_STATE_COUNT;

    if (accumulate) {
        acrossPatterns += out[destination + i * PADDED_STATE_COUNT + j];
    }

    out[destination + i * PADDED_STATE_COUNT + j] = acrossPatterns;
#endif
}

KW_GLOBAL_KERNEL void kernelPartialsPartialsCrossProducts(KW_GLOBAL_VAR REAL* KW_RESTRICT out,
                                                          const KW_GLOBAL_VAR REAL* KW_RESTRICT partials0,
                                                          const KW_GLOBAL_VAR REAL* KW_RESTRICT lengths0,
                                                          const KW_GLOBAL_VAR unsigned int* KW_RESTRICT instructions,
                                                          const KW_GLOBAL_VAR REAL* KW_RESTRICT inCategoryWeights,
                                                          const KW_GLOBAL_VAR REAL* KW_RESTRICT inPatternWeights,
                                                          const int skip,
                                                          const int totalPatterns,
                                                          const int totalNodes,
                                                          const int categoryCount,
                                                          const int rateOffset,
                                                          const int accumulate) {

#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    todo(); // Not implemented
#else // GPU implementation
    const int tx = KW_LOCAL_ID_0;

    const int tx_i = KW_LOCAL_ID_0 >> 4;
    const int tx_j = KW_LOCAL_ID_0 & 0xf;

    const int stateBlock_i = KW_GROUP_ID_2 / (PADDED_STATE_COUNT / 16);
    const int stateBlock_j = KW_GROUP_ID_2 % (PADDED_STATE_COUNT / 16);

    const int patternBlockId = KW_GROUP_ID_0;
    const int nodeId = KW_GROUP_ID_1;

    const int numPatternBlocks = KW_NUM_GROUPS_0;
    const int numNodeBlocks = KW_NUM_GROUPS_1;

    KW_LOCAL_MEM REAL post[PADDED_STATE_COUNT];
    KW_LOCAL_MEM REAL pre[PADDED_STATE_COUNT];

    KW_LOCAL_MEM REAL innerProduct[PADDED_STATE_COUNT];

    KW_LOCAL_MEM REAL categoryRates[16]; // TODO Assumes kCategoryCount <= 16
    KW_LOCAL_MEM REAL categoryWeights[16]; // TODO Should put these into constant memory anyway

    if (tx < categoryCount) {
        categoryRates[tx] = lengths0[rateOffset + tx];
        categoryWeights[tx] = inCategoryWeights[tx];
    }

    const int i = tx_i + stateBlock_i * 16;
    const int j = tx_j + stateBlock_j * 16;

    REAL acrossPatterns = (REAL) 0.0;

    for (int node = nodeId;  // Just interleaved indexing
         node < totalNodes;
         node += numNodeBlocks) {

        const int instructionOffset = (skip + node) * 2;
        const int preOffset = instructions[instructionOffset + 0];
        const int postOffset = instructions[instructionOffset + 1];

        const REAL edgeLength = lengths0[skip + node];

        for (int pattern = patternBlockId;
             pattern < totalPatterns;
             pattern += numPatternBlocks) {

            REAL patternDenominator = (REAL) 0.0;
            REAL numerator = (REAL) 0;

            for (int c = 0; c < categoryCount; ++c) {

                const KW_GLOBAL_VAR REAL* KW_RESTRICT prePartials  = partials0 + preOffset  + (pattern + totalPatterns * c) * PADDED_STATE_COUNT;
                const KW_GLOBAL_VAR REAL* KW_RESTRICT postPartials = partials0 + postOffset + (pattern + totalPatterns * c) * PADDED_STATE_COUNT;

                if (tx < PADDED_STATE_COUNT) {
                    pre[tx] = prePartials[tx];  // Coalesced global memory read
                    post[tx] = postPartials[tx]; // Coalesced global memory read
                    innerProduct[tx] = pre[tx] * post[tx];
                }

                KW_LOCAL_FENCE;

                const REAL scale =  edgeLength * categoryRates[c]; // TODO Better in constant memory?
                const REAL weight = categoryWeights[c]; // TODO Better in constant memory?

#ifdef IS_POWER_OF_TWO
                // parallelized reduction *** only works for powers-of-2 ****
                for (int k = PADDED_STATE_COUNT / 2; k > 0; k >>= 1) {
                    if (tx < k) {
#else
                for (int k = SMALLEST_POWER_OF_TWO / 2; k > 0; k >>= 1) {
                    if (tx < k && tx + k < PADDED_STATE_COUNT ) {
#endif // IS_POWER_OF_TWO
                        innerProduct[tx] += innerProduct[tx + k];
                    }
                    KW_LOCAL_FENCE;
                }

                patternDenominator += innerProduct[0] * weight;
                numerator += post[i] * pre[j] * weight * scale;
            }

            if (patternDenominator > (REAL) 0.0) {
                acrossPatterns += numerator * inPatternWeights[pattern] / patternDenominator;
            }
        }
    }

    const int destination = (nodeId * numPatternBlocks + patternBlockId) * PADDED_STATE_COUNT * PADDED_STATE_COUNT;

    if (accumulate) {
        acrossPatterns += out[destination + i * PADDED_STATE_COUNT + j];
    }

    out[destination + i * PADDED_STATE_COUNT + j] = acrossPatterns;
#endif
}

