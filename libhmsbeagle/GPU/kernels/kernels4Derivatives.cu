
#define multBy4(x)  ((x) << 2)
#define multBy16(x) ((x) << 4)

KW_GLOBAL_KERNEL void kernelPartialsPartialsGrowing(KW_GLOBAL_VAR REAL* KW_RESTRICT partials1,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT partials2,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT partials3,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT matrices1,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT matrices2,
                                                    int endPattern) {
#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    todo(); // TODO
#else // GPU implementation
    DETERMINE_INDICES_4_GPU();

    int y = deltaPartialsByState + deltaPartialsByMatrix;
    KW_LOCAL_MEM REAL sPartials1[PATTERN_BLOCK_SIZE * 4 * 4];
    KW_LOCAL_MEM REAL sPartials2[PATTERN_BLOCK_SIZE * 4 * 4];
    /* copy PADDED_STATE_COUNT * PATTERN_BLOCK_SIZE lengthed partials*/
    if (pattern < endPattern) {
        sPartials1[multBy16(patIdx) | tx] = partials1[y | tx]; /*All coalesced memory*/
        sPartials2[multBy16(patIdx) | tx] = partials2[y | tx];
    } else {
        sPartials1[multBy16(patIdx) | tx] = 0;
        sPartials2[multBy16(patIdx) | tx] = 0;
    }

    const KW_GLOBAL_VAR REAL* KW_RESTRICT matrix1 = matrices1 + x2; /*Points to *this* matrix*/
    const KW_GLOBAL_VAR REAL* KW_RESTRICT matrix2 = matrices2 + x2;
    KW_LOCAL_MEM REAL sMatrix1[16]; /*Load values into shared memory*/
    KW_LOCAL_MEM REAL sMatrix2[16];
    if (patIdx == 0 ) {
        sMatrix1[multBy4(state) | pat] = matrix1[tx]; /* Should write transpose into sMatrix1 */
        sMatrix2[tx] = matrix2[tx];
    }
    KW_LOCAL_FENCE;

    KW_LOCAL_MEM REAL sProduct[PATTERN_BLOCK_SIZE * 4 * 4];
    if (pattern < endPattern) { // Remove padded threads!
        REAL sum2;
        int i = pat;
        int patIdx16pat4 = multBy16(patIdx) | (tx & 0xC);

        sum2 = sMatrix2[multBy4(i) | state] * sPartials2[patIdx16pat4 | i];
        i = (i + 1) & 0x3;
        FMA(   sMatrix2[multBy4(i) | state],  sPartials2[patIdx16pat4 | i], sum2);
        i = (i + 1) & 0x3;
        FMA(   sMatrix2[multBy4(i) | state],  sPartials2[patIdx16pat4 | i], sum2);
        i = (i + 1) & 0x3;
        FMA(   sMatrix2[multBy4(i) | state],  sPartials2[patIdx16pat4 | i], sum2);

        sProduct[multBy16(patIdx) | tx] = sPartials1[multBy16(patIdx) | tx] * sum2;
    }

    KW_LOCAL_FENCE;

    if (pattern < endPattern) {
        REAL sum1;
        int i = pat;
        int patIdx16pat4 = multBy16(patIdx) | (tx & 0xC);

        sum1 = sMatrix1[multBy4(i) | state] * sProduct[patIdx16pat4 | i];
        i = (i + 1) & 0x3;
        FMA(   sMatrix1[multBy4(i) | state],  sProduct[patIdx16pat4 | i], sum1);
        i = (i + 1) & 0x3;
        FMA(   sMatrix1[multBy4(i) | state],  sProduct[patIdx16pat4 | i], sum1);
        i = (i + 1) & 0x3;
        FMA(   sMatrix1[multBy4(i) | state],  sProduct[patIdx16pat4 | i], sum1);

        partials3[u] = sum1;
    }
#endif // FW_OPENCL_CPU
}

KW_GLOBAL_KERNEL void kernelPartialsStatesGrowing(KW_GLOBAL_VAR REAL* KW_RESTRICT partials1,
                                                  KW_GLOBAL_VAR int*  KW_RESTRICT states2,
                                                  KW_GLOBAL_VAR REAL* KW_RESTRICT partials3,
                                                  KW_GLOBAL_VAR REAL* KW_RESTRICT matrices1,
                                                  KW_GLOBAL_VAR REAL* KW_RESTRICT matrices2,
                                                  int endPattern) {
#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    todo(); // TODO
#else // GPU implementation

    DETERMINE_INDICES_4_GPU();

    int y = deltaPartialsByState + deltaPartialsByMatrix;
    KW_LOCAL_MEM REAL sPartials1[PATTERN_BLOCK_SIZE * 4 * 4];
    /* copy PADDED_STATE_COUNT * PATTERN_BLOCK_SIZE lengthed partials*/
    if (pattern < endPattern) {
        sPartials1[multBy16(patIdx) | tx] = partials1[y | tx]; /*All coalesced memory*/
    } else {
        sPartials1[multBy16(patIdx) | tx] = 0;
    }

    const KW_GLOBAL_VAR REAL* KW_RESTRICT matrix1 = matrices1 + x2; /*Points to *this* matrix*/
    const KW_GLOBAL_VAR REAL* KW_RESTRICT matrix2 = matrices2 + x2;
    KW_LOCAL_MEM REAL sMatrix1[16]; /*Load values into shared memory*/
    KW_LOCAL_MEM REAL sMatrix2[16];
    if (patIdx == 0 ) {
        sMatrix1[multBy4(state) | pat] = matrix1[tx]; /* Should write transpose into sMatrix1 */
        sMatrix2[tx] = matrix2[tx];
    }

    KW_LOCAL_FENCE;

    KW_LOCAL_MEM REAL sProduct[PATTERN_BLOCK_SIZE * 4 * 4];
    if (pattern < endPattern) { // Remove padded threads!
        REAL sum2 = 1;
        int state2 = states2[pattern];
        if (state2 < PADDED_STATE_COUNT) {
            sum2 = sMatrix2[state2 * 4 + state];
        }

        sProduct[multBy16(patIdx) | tx] = sPartials1[multBy16(patIdx) | tx] * sum2;
    }

    KW_LOCAL_FENCE;

    if (pattern < endPattern) {
        REAL sum1;
        int i = pat;
        int patIdx16pat4 = multBy16(patIdx) | (tx & 0xC);

        sum1 = sMatrix1[multBy4(i) | state] * sProduct[patIdx16pat4 | i];
        i = (i + 1) & 0x3;
        FMA(   sMatrix1[multBy4(i) | state],  sProduct[patIdx16pat4 | i], sum1);
        i = (i + 1) & 0x3;
        FMA(   sMatrix1[multBy4(i) | state],  sProduct[patIdx16pat4 | i], sum1);
        i = (i + 1) & 0x3;
        FMA(   sMatrix1[multBy4(i) | state],  sProduct[patIdx16pat4 | i], sum1);

        partials3[u] = sum1;
    }
#endif // FW_OPENCL_CPU
}

KW_GLOBAL_KERNEL void kernelPartialsPartialsEdgeFirstDerivatives(KW_GLOBAL_VAR REAL* KW_RESTRICT out,
                                                                 KW_GLOBAL_VAR REAL* KW_RESTRICT partials0,
                                                                 KW_GLOBAL_VAR REAL* KW_RESTRICT matrices0,
                                                                 KW_GLOBAL_VAR unsigned int* KW_RESTRICT instructions,
                                                                 KW_GLOBAL_VAR REAL* KW_RESTRICT weights,
                                                                 int skip,
                                                                 int totalPatterns, int categoryCount) {
#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    todo(); // Not implemented
#else // GPU implementation

    int tx = KW_LOCAL_ID_0;
    int state = tx & 0x3;
    int pat = tx >> 2;
    int patIdx = KW_LOCAL_ID_1;
    int pattern = __umul24(KW_GROUP_ID_0, PATTERN_BLOCK_SIZE * 4) + multBy4(patIdx) + pat;
    int y = multBy16(KW_GROUP_ID_0 * PATTERN_BLOCK_SIZE + patIdx);

    int node = KW_GROUP_ID_1;
    int instructionOffset = (skip + node) * 3;

    unsigned int partials1Offset = instructions[instructionOffset + 0];
    unsigned int partials2Offset = instructions[instructionOffset + 1];
    unsigned int matrices1Offset = instructions[instructionOffset + 2];

    KW_LOCAL_MEM REAL sMatrix2[16];

    KW_LOCAL_MEM REAL sPartials1[PATTERN_BLOCK_SIZE * 4 * 4];
    KW_LOCAL_MEM REAL sPartials2[PATTERN_BLOCK_SIZE * 4 * 4];

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

        /* copy PADDED_STATE_COUNT * PATTERN_BLOCK_SIZE length partials*/
        if (pattern < totalPatterns) {
            lPartial1 = partials1[y | tx]; /*All coalesced memory*/
            sPartials2[multBy16(patIdx) | tx] = lPartial2 = partials2[y | tx];
        } else {
            lPartial1 = 0;
            sPartials2[multBy16(patIdx) | tx] = lPartial2 = 0;
        }

        FMA(lPartial1, lPartial2 * sWeights[c], denominator);

        if (patIdx == 0 ) {
            sMatrix2[tx] = matrix2[tx];
        }

        KW_LOCAL_FENCE;

        REAL sum2;
        int i = pat;
        int patIdx16pat4 = multBy16(patIdx) | (tx & 0xC);

        sum2 = sMatrix2[multBy4(i) | state] * sPartials2[patIdx16pat4 | i];
        i = (i + 1) & 0x3;
        FMA(   sMatrix2[multBy4(i) | state],  sPartials2[patIdx16pat4 | i], sum2);
        i = (i + 1) & 0x3;
        FMA(   sMatrix2[multBy4(i) | state],  sPartials2[patIdx16pat4 | i], sum2);
        i = (i + 1) & 0x3;
        FMA(   sMatrix2[multBy4(i) | state],  sPartials2[patIdx16pat4 | i], sum2);

        KW_LOCAL_FENCE; // TODO Is this necessary?

        FMA(lPartial1, sum2 * sWeights[c], numerator);

//        partials1 += totalPatterns * PADDED_STATE_COUNT;
//        partials2 += totalPatterns * PADDED_STATE_COUNT;
    }

    sPartials1[patIdx * PATTERN_BLOCK_SIZE + tx] = numerator;
    sPartials2[patIdx * PATTERN_BLOCK_SIZE + tx] = denominator;

    KW_LOCAL_FENCE;

    if (state < 2) {
        sPartials1[patIdx * PATTERN_BLOCK_SIZE + tx] += sPartials1[patIdx * PATTERN_BLOCK_SIZE + tx + 2];
        sPartials2[patIdx * PATTERN_BLOCK_SIZE + tx] += sPartials2[patIdx * PATTERN_BLOCK_SIZE + tx + 2];
    }

    KW_LOCAL_FENCE;

    if (state < 1) {
        sPartials1[patIdx * PATTERN_BLOCK_SIZE + tx] += sPartials1[patIdx * PATTERN_BLOCK_SIZE + tx + 1];
        sPartials2[patIdx * PATTERN_BLOCK_SIZE + tx] += sPartials2[patIdx * PATTERN_BLOCK_SIZE + tx + 1];
    }

    KW_LOCAL_FENCE;

    if (patIdx < PATTERN_BLOCK_SIZE / 4) { // Need the first PATTERN_BLOCK_SIZE * 4 threads
        int offset = patIdx * PATTERN_BLOCK_SIZE  + tx;
        int site = __umul24(KW_GROUP_ID_0, PATTERN_BLOCK_SIZE * 4) + offset;
        if (site < totalPatterns) {
            int row = offset >> 2; // divide by 4
            int col = offset & 0x3; // mod 4
            REAL numerator = sPartials1[row * PATTERN_BLOCK_SIZE + multBy4(col) + 0];
            REAL denominator = sPartials2[row * PATTERN_BLOCK_SIZE + multBy4(col) + 0];
            REAL ratio = 0.0;
            if (denominator != 0.0) {
                ratio = numerator / denominator;
            }
            out[totalPatterns * (skip + node) + site] = ratio; // TODO Check that these are all coalesced writes
        }
    }

//    if (pattern < totalPatterns) {
//        if (state == 0) {
//            REAL numerator = sPartials1[patIdx * PATTERN_BLOCK_SIZE + multBy4(pat) + 0];
//            REAL denominator = sPartials2[patIdx * PATTERN_BLOCK_SIZE + multBy4(pat) + 0];
//            REAL ratio = 0.0;
//            if (denominator != 0.0) {
//                ratio = numerator / denominator;
//            }
//            out[totalPatterns * (skip + node) + pattern] = ratio;
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
    todo(); // Not implemented
#else // GPU implementation

    int tx = KW_LOCAL_ID_0;
    int state = tx & 0x3;
    int pat = tx >> 2;
    int patIdx = KW_LOCAL_ID_1;
    int pattern = __umul24(KW_GROUP_ID_0, PATTERN_BLOCK_SIZE * 4) + multBy4(patIdx) + pat;
    int y = multBy16(KW_GROUP_ID_0 * PATTERN_BLOCK_SIZE + patIdx);

    int node = KW_GROUP_ID_1;
    int instructionOffset = (skip + node) * 3;

    unsigned int states1Offset   = instructions[instructionOffset + 0];
    unsigned int partials2Offset = instructions[instructionOffset + 1];
    unsigned int matrices1Offset = instructions[instructionOffset + 2];

    KW_LOCAL_MEM REAL sMatrix2[16];

    KW_LOCAL_MEM REAL sPartials1[PATTERN_BLOCK_SIZE * 4 * 4];
    KW_LOCAL_MEM REAL sPartials2[PATTERN_BLOCK_SIZE * 4 * 4];

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

        /* copy PADDED_STATE_COUNT * PATTERN_BLOCK_SIZE length partials*/
        if (pattern < totalPatterns) {
            sPartials2[multBy16(patIdx) | tx] = lPartial2 = partials2[y | tx];
        } else {
            sPartials2[multBy16(patIdx) | tx] = lPartial2 = 0;
        }

        FMA(lPartial1, lPartial2 * sWeights[c], denominator);

        if (patIdx == 0 ) {
            sMatrix2[tx] = matrix2[tx];
        }

        KW_LOCAL_FENCE;

        REAL sum2;
        int i = pat;
        int patIdx16pat4 = multBy16(patIdx) | (tx & 0xC);

        sum2 = sMatrix2[multBy4(i) | state] * sPartials2[patIdx16pat4 | i];
        i = (i + 1) & 0x3;
        FMA(   sMatrix2[multBy4(i) | state],  sPartials2[patIdx16pat4 | i], sum2);
        i = (i + 1) & 0x3;
        FMA(   sMatrix2[multBy4(i) | state],  sPartials2[patIdx16pat4 | i], sum2);
        i = (i + 1) & 0x3;
        FMA(   sMatrix2[multBy4(i) | state],  sPartials2[patIdx16pat4 | i], sum2);

        KW_LOCAL_FENCE; // TODO Is this necessary?

        FMA(lPartial1, sum2 * sWeights[c], numerator);

//        partials1 += totalPatterns * PADDED_STATE_COUNT;
//        partials2 += totalPatterns * PADDED_STATE_COUNT;
    }

    sPartials1[patIdx * PATTERN_BLOCK_SIZE + tx] = numerator;
    sPartials2[patIdx * PATTERN_BLOCK_SIZE + tx] = denominator;

    KW_LOCAL_FENCE;

    if (state < 2) {
        sPartials1[patIdx * PATTERN_BLOCK_SIZE + tx] += sPartials1[patIdx * PATTERN_BLOCK_SIZE + tx + 2];
        sPartials2[patIdx * PATTERN_BLOCK_SIZE + tx] += sPartials2[patIdx * PATTERN_BLOCK_SIZE + tx + 2];
    }

    KW_LOCAL_FENCE;

    if (state < 1) {
        sPartials1[patIdx * PATTERN_BLOCK_SIZE + tx] += sPartials1[patIdx * PATTERN_BLOCK_SIZE + tx + 1];
        sPartials2[patIdx * PATTERN_BLOCK_SIZE + tx] += sPartials2[patIdx * PATTERN_BLOCK_SIZE + tx + 1];
    }

    KW_LOCAL_FENCE;

    if (patIdx < PATTERN_BLOCK_SIZE / 4) { // Need the first PATTERN_BLOCK_SIZE * 4 threads
        int offset = patIdx * PATTERN_BLOCK_SIZE  + tx;
        int site = __umul24(KW_GROUP_ID_0, PATTERN_BLOCK_SIZE * 4) + offset;
        if (site < totalPatterns) {
            int row = offset >> 2; // divide by 4
            int col = offset & 0x3; // mod 4
            REAL numerator = sPartials1[row * PATTERN_BLOCK_SIZE + multBy4(col) + 0];
            REAL denominator = sPartials2[row * PATTERN_BLOCK_SIZE + multBy4(col) + 0];
            REAL ratio = 0.0;
            if (denominator != 0.0) {
                ratio = numerator / denominator;
            }
            out[totalPatterns * (skip + node) + site] = ratio; // TODO Check that these are all coalesced writes
        }
    }
#endif
}
