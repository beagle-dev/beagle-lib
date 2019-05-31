KW_GLOBAL_KERNEL void kernelPartialsPartialsEdgeFirstDerivatives(KW_GLOBAL_VAR REAL* KW_RESTRICT out,
                                                                 KW_GLOBAL_VAR REAL* KW_RESTRICT partials0,
                                                                 KW_GLOBAL_VAR REAL* KW_RESTRICT matrices0,
                                                                 KW_GLOBAL_VAR unsigned int* KW_RESTRICT offsets,
                                                                 KW_GLOBAL_VAR REAL* KW_RESTRICT weights,
                                                                 int totalPatterns, int categoryCount) {
#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    // Not implemented
#else // GPU implementation

    int state = KW_LOCAL_ID_0;
    int patIdx = KW_LOCAL_ID_1;
    int pattern = KW_GROUP_ID_0 * BLOCK_PEELING_SIZE + patIdx;

    int node = KW_GROUP_ID_1;
    int instructionOffset = node * 3;

    unsigned int partials1Offset = offsets[instructionOffset + 0];
    unsigned int partials2Offset = offsets[instructionOffset + 1];
    unsigned int matrices1Offset = offsets[instructionOffset + 2];

    KW_LOCAL_MEM REAL sMatrix2[BLOCK_PEELING_SIZE][PADDED_STATE_COUNT];

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
        for (int i = 0; i < PADDED_STATE_COUNT; i += BLOCK_PEELING_SIZE) {
            /* load one row of matrices */
            if (patIdx < BLOCK_PEELING_SIZE) {
                /* These are all coherent global memory reads. */
                sMatrix2[patIdx][state] = matrix2[patIdx * PADDED_STATE_COUNT + state];
                /* sMatrix now filled with starting in state and ending in i */
                matrix2 += BLOCK_PEELING_SIZE * PADDED_STATE_COUNT;
            }
            KW_LOCAL_FENCE;

            // TODO 2nd check is unncessary for stateCount >= 16
            for (int j = 0; (j < BLOCK_PEELING_SIZE) && (i + j < PADDED_STATE_COUNT); j++) {
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
//    int tx = KW_LOCAL_ID_0;
//    if (tx < PATTERN_BLOCK_SIZE && patIdx == 0) { // Use first PATTERN_BLOCK_SIZE threads to write
//        int site = KW_GROUP_ID_0 * BLOCK_PEELING_SIZE + tx;
//        if (site < totalPatterns) {
//            out[totalPatterns * node + site] = sPartials1[tx][0] / sPartials[tx][0];
//        }
//    }

    if (pattern < totalPatterns) {
        if (state == 0) {
            out[totalPatterns * node + pattern] = sPartials1[patIdx][0] / sPartials2[patIdx][0]; // pre;
//            out[totalPatterns * node + pattern] = sPartials1[patIdx][0];  // Write numerator
//            out[totalPatterns * (KW_NUM_GROUPS_1 + node) + pattern] = sPartials2[patIdx][0]; // Write denomiator
        }
    }

#endif
}
