KW_GLOBAL_KERNEL void kernelPartialsPartialsGrowing(KW_GLOBAL_VAR REAL* KW_RESTRICT partials1,
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
    float warpsPerPattern = (float) PADDED_STATE_COUNT / warpSize;
    int warpIdx = warpState + warpPattern * warpsPerPattern;
    int laneid = (state + patIdx * PADDED_STATE_COUNT) % warpSize;

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
#define GET_SMEM_ROW_PARTIALS(X, Y) (((X / WMMA_K) + ((Y / (PADDED_STATE_COUNT / WMMA_K) ) * (PADDED_STATE_COUNT / WMMA_K)) ) & 0x07)
#define GET_BANK_GROUP_PARTIALS(X,Y) ((Y + (X/WMMA_K) * ((Y & 1) * -2 + 1)) & ((PADDED_STATE_COUNT/WMMA_K) - 1)) // 0x07 should be generalized to & PADDED_STATE_COUNT/WMMA_K - 1
#define GET_SMEM_COL_PARTIALS(X,Y) (GET_BANK_GROUP_PARTIALS(X,Y) * WMMA_K + (X % WMMA_K))
#define GET_SMEM_OFFSET_PARTIALS(X,Y) (GET_SMEM_ROW_PARTIALS(X, Y) * PADDED_STATE_COUNT + GET_SMEM_COL_PARTIALS(X, Y))

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

    for (int i = 0; i < PADDED_STATE_COUNT; i += WMMA_K) {
        sMatrixRow = warpIdx % (PADDED_STATE_COUNT / WMMA_M);
        sMatrixCol = i;
        partialsRow = i;
        partialsCol = warpIdx / (PADDED_STATE_COUNT / WMMA_M);

        sMatrix2[GET_SMEM_OFFSET_SMATRIX(state, patIdx)] = matrix2[sMatrixCol * PADDED_STATE_COUNT + patIdx * PADDED_STATE_COUNT + state];

        KW_LOCAL_FENCE;

        // reg_row* and reg_col* are according to memory layout in ShM
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

    int statesPerWarp = (PADDED_STATE_COUNT >= warpSize) ? state : laneid;
    // Map results of MMA operation (8x8) to permuted shared memory for elementwise multiplication of partials
    int partials1Index = WMMA_M * (int) (patIdx * warpsPerPattern) + ((laneid * 2) % 8) * PADDED_STATE_COUNT + (statesPerWarp * 2)/8;
    int partials1State = partials1Index % PADDED_STATE_COUNT;
    int partials1PatIdx = partials1Index / PADDED_STATE_COUNT;
    int partials1PatIdxPlusOne = partials1PatIdx + 1;

    sPartials1[GET_SMEM_OFFSET_PARTIALS(partials1State, partials1PatIdx)] = sPartials1[GET_SMEM_OFFSET_PARTIALS(partials1State, partials1PatIdx)] * res21;
    sPartials1[GET_SMEM_OFFSET_PARTIALS(partials1State, partials1PatIdxPlusOne)] = sPartials1[GET_SMEM_OFFSET_PARTIALS(partials1State, partials1PatIdxPlusOne)] * res22;

    for (int i = 0; i < PADDED_STATE_COUNT; i += WMMA_K) {
        sMatrixRow = warpIdx % (PADDED_STATE_COUNT / WMMA_M);
        sMatrixCol = i;
        partialsRow = i;
        partialsCol = warpIdx / (PADDED_STATE_COUNT / WMMA_M);

        sMatrix1[GET_SMEM_OFFSET_SMATRIX(state, patIdx)] = matrix1[sMatrixCol * PADDED_STATE_COUNT + patIdx * PADDED_STATE_COUNT + state];
        KW_LOCAL_FENCE;

        // reg_row* and reg_col* are according to memory layout in ShM
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

    if (patternBlock + ((laneid * 2) % 8) < totalPatterns)
        partials3[u + (int) (patIdx * warpsPerPattern) * WMMA_M + ((laneid * 2) % 8) * PADDED_STATE_COUNT + (statesPerWarp * 2)/8] = res11;

    if (patternBlock + ((laneid * 2) % 8) + 1 < totalPatterns)
        partials3[u + PADDED_STATE_COUNT + (int) (patIdx * warpsPerPattern) * WMMA_M + ((laneid * 2) % 8) * PADDED_STATE_COUNT + (statesPerWarp * 2)/8] = res12;
#endif
}