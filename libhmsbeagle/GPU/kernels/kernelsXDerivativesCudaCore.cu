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