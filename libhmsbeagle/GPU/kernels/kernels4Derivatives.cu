}

#include <mma.h>

extern "C" {
#define multBy4(x)  ((x) << 2)
#define multBy16(x) ((x) << 4)

#define DETERMINE_INDICES_4_GPU()\
    int tx = KW_LOCAL_ID_0;\
    int state = tx & 0x3;\
    int pat = tx >> 2;\
    int patIdx = KW_LOCAL_ID_1;\
    int matrix = KW_GROUP_ID_1;\
    int pattern = __umul24(KW_GROUP_ID_0, PATTERN_BLOCK_SIZE * 4) + multBy4(patIdx) + pat;\
    int deltaPartialsByState = multBy16(KW_GROUP_ID_0 * PATTERN_BLOCK_SIZE + patIdx);\
    int deltaPartialsByMatrix = __umul24(matrix, multBy4(endPattern));\
    int x2 = multBy16(matrix);\
    int u = tx + deltaPartialsByState + deltaPartialsByMatrix;

#define DETERMINE_INDICES_4_MULTI_1_GPU()\
    int opIndexPtr = (gridStartOp + KW_GROUP_ID_0) * 8;\
    int startPat   = ptrOffsets[opIndexPtr    ];\
    int endPattern = ptrOffsets[opIndexPtr + 1];\
    int tx = KW_LOCAL_ID_0;\
    int state = tx & 0x3;\
    int pat = tx >> 2;\
    int patIdx = KW_LOCAL_ID_1;\
    int matrix = KW_GROUP_ID_1;\
    int pattern = startPat + multBy4(patIdx) + pat;\
    int deltaPartialsByState = multBy4(startPat) + multBy16(patIdx);\
    int deltaPartialsByMatrix = __umul24(matrix, multBy4(totalPatterns));\
    int x2 = multBy16(matrix);\
    int u = tx + deltaPartialsByState + deltaPartialsByMatrix;

#define DETERMINE_INDICES_4_MULTI_2_GPU()\
          KW_GLOBAL_VAR REAL* KW_RESTRICT partials3 =  partials + ptrOffsets[opIndexPtr + 4];\
    const KW_GLOBAL_VAR REAL* KW_RESTRICT matrix1   =  matrices + ptrOffsets[opIndexPtr + 5];\
    const KW_GLOBAL_VAR REAL* KW_RESTRICT matrix2   =  matrices + ptrOffsets[opIndexPtr + 6];

KW_GLOBAL_KERNEL void kernelPartialsPartialsGrowingMulti(KW_GLOBAL_VAR REAL* KW_RESTRICT partials,
                                                         const KW_GLOBAL_VAR REAL* KW_RESTRICT matrices,
                                                         const KW_GLOBAL_VAR unsigned int* KW_RESTRICT ptrOffsets,
                                                         int gridStartOp,
                                                         int totalPatterns) {

#ifdef FW_OPENCL_CPU // CPU/MIC implementationt
    todo(); // TODOg
#else // GPU implementation
    DETERMINE_INDICES_4_MULTI_1_GPU();
    const KW_GLOBAL_VAR REAL* KW_RESTRICT partials1 =  partials + ptrOffsets[opIndexPtr + 2];
    const KW_GLOBAL_VAR REAL* KW_RESTRICT partials2 =  partials + ptrOffsets[opIndexPtr + 3];
    DETERMINE_INDICES_4_MULTI_2_GPU();


    LOAD_PARTIALS_PARTIALS_4_MULTI_PART_GPU();
    LOAD_MATRIX_4_MULTI_GPU();
    if (pattern < endPattern) { // Remove padded threads!
        SUM_PARTIALS_PARTIALS_4_GPU();
        partials3[u] = sum1 * sum2;
    }
#endif // FW_OPENCL_CPU
}

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

KW_GLOBAL_KERNEL void kernelPartialsPartialsGrowingTensorCores(KW_GLOBAL_VAR REAL* KW_RESTRICT partials1,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT partials2,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT partials3,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT matrices1,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT matrices2,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT tmpAcc,
                                                    int endPattern) {
#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    todo(); // TODO
#else // GPU implementation
    DETERMINE_INDICES_4_GPU();

    const int WMMA_M = 8;
    const int WMMA_N = 8;
    const int WMMA_K = 4;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, double, nvcuda::wmma::row_major> sMatrixFrag1;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, double, nvcuda::wmma::row_major> sMatrixFrag2;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, double, nvcuda::wmma::col_major> partialsFrag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, double> accFrag;
    nvcuda::wmma::fill_fragment(sMatrixFrag1, 0.0);
    nvcuda::wmma::fill_fragment(sMatrixFrag2, 0.0);
    nvcuda::wmma::fill_fragment(partialsFrag, 0.0);
    nvcuda::wmma::fill_fragment(accFrag, 0.0);

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
        sMatrix1[tx] = matrix1[tx]; /* Should write transpose into sMatrix1 */
        sMatrix2[multBy4(state) | pat] = matrix2[tx];
    }
    KW_LOCAL_FENCE;

    // Load into matrices into fragments. Note: all warps need to load sMatrix into fragments
    nvcuda::wmma::load_matrix_sync(sMatrixFrag2, sMatrix2, 4);
    nvcuda::wmma::load_matrix_sync(sMatrixFrag1, sMatrix1, 4);

    KW_LOCAL_MEM REAL tmp[WMMA_M * WMMA_N * 8]; // TODO: Reuse memory
    int patWarp, tmpWarp;
    patWarp = 8 * 4 * (patIdx/2); // 8 patterns per warp and 4 states per pattern
    tmpWarp = 64 * (patIdx/2); // 64 values per wmma but half of rows are 0s

    // Load patterns into fragment
    nvcuda::wmma::load_matrix_sync(partialsFrag, sPartials2 + patWarp, WMMA_K);

    // Multiply
    nvcuda::wmma::fill_fragment(accFrag, 0.0);
    nvcuda::wmma::mma_sync(accFrag, sMatrixFrag2, partialsFrag, accFrag);

    nvcuda::wmma::store_matrix_sync(tmp + tmpWarp, accFrag, WMMA_M, nvcuda::wmma::mem_col_major);

    tmpAcc[16 * patIdx + tx] = sPartials2[16 * patIdx + tx];
    // Element-wise multiplication
    sPartials1[(16 * patIdx) + tx] = sPartials1[16 * patIdx + tx] * tmp[(32 * patIdx) + (8 * (tx / 4)) + (tx % 4)];

    KW_LOCAL_FENCE;

    // Load patterns into fragment
    nvcuda::wmma::load_matrix_sync(partialsFrag, sPartials1 + patWarp, WMMA_K);

    // Multiply
    nvcuda::wmma::fill_fragment(accFrag, 0.0);
    nvcuda::wmma::mma_sync(accFrag, sMatrixFrag1, partialsFrag, accFrag);

    nvcuda::wmma::store_matrix_sync(tmp + tmpWarp, accFrag, WMMA_M, nvcuda::wmma::mem_col_major);

    partials3[u] = tmp[(32 * patIdx) + 8 * (tx / 4) + (tx % 4)];

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
            sMatrix2[multBy4(state) | pat] = matrix2[tx]; // transposed
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
            sMatrix2[multBy4(state) | pat] = matrix2[tx]; // transposed
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
      const int state = tx & 0x3;
      const int pat = tx >> 2;
  
      const int patternBlockId = KW_GROUP_ID_0;
      const int nodeId = KW_GROUP_ID_1;
  
      const int numPatternBlocks = KW_NUM_GROUPS_0;
      const int numNodeBlocks = KW_NUM_GROUPS_1;
  
      KW_LOCAL_MEM REAL post[4 * 4];
      KW_LOCAL_MEM REAL pre[4 * 4];
      KW_LOCAL_MEM REAL denominator[4 * 4];
  
      KW_LOCAL_MEM REAL patternDenominator[16];       
      KW_LOCAL_MEM REAL patternWeights[4];
  
      KW_LOCAL_MEM REAL categoryRates[16]; // TODO Assumes kCategoryCount <= 16 
      KW_LOCAL_MEM REAL categoryWeights[16]; // TODO Should put these into constant memory anyway
       
      if (tx < categoryCount) {
          categoryRates[tx] = lengths0[rateOffset + tx];
          categoryWeights[tx] = inCategoryWeights[tx];
      }
  
      KW_LOCAL_FENCE;
  
      // Fancy indexing to keep pattern work consecutive (may not help cache since jumping btw categories anyway)
      // TODO Check if helpful
      const int batchWorkItems = (totalPatterns + 4 - 1) / 4; // 4 patterns at a time
      const int patternWorkSize = 4 * ((batchWorkItems + numPatternBlocks - 1) / numPatternBlocks);
  
      REAL acrossPatterns = 0;
     
      for (int node = nodeId;  // Just interleaved indexing
           node < totalNodes; 
           node += numNodeBlocks) {
  
          int instructionOffset = (skip + node) * 2;
          unsigned int statesOffset = instructions[instructionOffset + 0];
          unsigned int preOffset = instructions[instructionOffset + 1];
  
          const REAL edgeLength = lengths0[skip + node];   
  
          for (int pattern = patternBlockId * patternWorkSize; 
               pattern < (patternBlockId + 1) * patternWorkSize; 
               pattern += 4) {
  
              unsigned int patternOffset = pattern * 4;
  
              REAL txPatternDenominator = 0;
  
              REAL withinPattern0 = 0;
              REAL withinPattern1 = 0;
              REAL withinPattern2 = 0;
              REAL withinPattern3 = 0;
    
            const KW_GLOBAL_VAR int* KW_RESTRICT postStates = states0 + statesOffset;
 
            const int stateData = postStates[pattern + pat]; // patterns are already padded mod 4
            post[tx] = (state == stateData) ? (REAL) 1.0 : (REAL) 0.0; // TODO | stateData >= missingState ?

              for (int c = 0; c < categoryCount; ++c) {
  
                  const KW_GLOBAL_VAR REAL* KW_RESTRICT prePartials = partials0 + preOffset + patternOffset + 
                        totalPatterns * PADDED_STATE_COUNT * c;
  
                  if (pattern < totalPatterns) {
                      pre[tx] = prePartials[tx];  // Coalesced global memory read
                  } else {
                      pre[tx] = 0.0;                      
                  }
  
                  const REAL scale =  edgeLength * categoryRates[c]; // TODO Better in constant memory?
                  const REAL weight = categoryWeights[c]; // TODO Better in constant memory?
  
                  // Inner product
                  denominator[tx] = pre[tx] * post[tx];
                  if (tx < 8) {
                      denominator[tx << 1] += denominator[tx << 1 | 0x1];
                  }
  
                  KW_LOCAL_FENCE; // TODO necessary? in same warp
  
                  if (tx < 4) {
                      denominator[tx << 2] += denominator[tx << 2 | 0x2];
                  }
  
                  KW_LOCAL_FENCE; // TODO necessary? is same warp
  
                  txPatternDenominator += denominator[tx & 0xC] * weight;
                  
                //   post[tx] *= weight * scale;
                  pre[tx] *= weight * scale;
  
                  KW_LOCAL_FENCE; // TODO Merge with fence above
  
                  withinPattern0 += post[4 * 0 | state] * pre[4 * 0 | pat];
                  withinPattern1 += post[4 * 1 | state] * pre[4 * 1 | pat];
                  withinPattern2 += post[4 * 2 | state] * pre[4 * 2 | pat];
                  withinPattern3 += post[4 * 3 | state] * pre[4 * 3 | pat];
              }
  
              patternDenominator[tx] = txPatternDenominator;
  
              if (tx < 4) {
                  patternWeights[tx] = inPatternWeights[pattern + tx];
              }            
  
              KW_LOCAL_FENCE;
  
              if (patternDenominator[4 * 0] > 0.0) {
                  acrossPatterns += withinPattern0 * patternWeights[0] / patternDenominator[4 * 0];
              }
  
              if (patternDenominator[4 * 1] > 0.0) {
                  acrossPatterns += withinPattern1 * patternWeights[1] / patternDenominator[4 * 1];
              }
              
              if (patternDenominator[4 * 2] > 0.0) {
                  acrossPatterns += withinPattern2 * patternWeights[2] / patternDenominator[4 * 2];
              }
              
              if (patternDenominator[4 * 3] > 0.0) {
                  acrossPatterns += withinPattern3 * patternWeights[3] / patternDenominator[4 * 3];
              } // TODO Vectorize
          }
      }

      KW_LOCAL_FENCE;
      
      const int destination = (nodeId * numPatternBlocks + patternBlockId) * 16;

      if (accumulate) {
          acrossPatterns += out[destination + tx];
      }
  
    out[destination + tx] = acrossPatterns;
    //   out[destination + tx] = post[tx];
    // out[destination + tx] = patternDenominator[tx];
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
    const int state = tx & 0x3;
    const int pat = tx >> 2;

    const int patternBlockId = KW_GROUP_ID_0;
    const int nodeId = KW_GROUP_ID_1;

    const int numPatternBlocks = KW_NUM_GROUPS_0;
    const int numNodeBlocks = KW_NUM_GROUPS_1;

    KW_LOCAL_MEM REAL post[4 * 4];
    KW_LOCAL_MEM REAL pre[4 * 4];
    KW_LOCAL_MEM REAL denominator[4 * 4];

    KW_LOCAL_MEM REAL patternDenominator[16];       
    KW_LOCAL_MEM REAL patternWeights[4];

    KW_LOCAL_MEM REAL categoryRates[16]; // TODO Assumes kCategoryCount <= 16 
    KW_LOCAL_MEM REAL categoryWeights[16]; // TODO Should put these into constant memory anyway
     
    if (tx < categoryCount) {
        categoryRates[tx] = lengths0[rateOffset + tx];
        categoryWeights[tx] = inCategoryWeights[tx];
    }

    KW_LOCAL_FENCE;

    // Fancy indexing to keep pattern work consecutive (may not help cache since jumping btw categories anyway)
    // TODO Check if helpful
    const int batchWorkItems = (totalPatterns + 4 - 1) / 4; // 4 patterns at a time
    const int patternWorkSize = 4 * ((batchWorkItems + numPatternBlocks - 1) / numPatternBlocks);

    REAL acrossPatterns = 0;

    for (int node = nodeId;  // Just interleaved indexing
         node < totalNodes; 
         node += numNodeBlocks) {

        int instructionOffset = (skip + node) * 2;
        unsigned int preOffset = instructions[instructionOffset + 0];
        unsigned int postOffset = instructions[instructionOffset + 1];

        const REAL edgeLength = lengths0[skip + node];   

        for (int pattern = patternBlockId * patternWorkSize; 
             pattern < (patternBlockId + 1) * patternWorkSize; 
             pattern += 4) {

            unsigned int patternOffset = pattern * 4;

            REAL txPatternDenominator = 0;

            REAL withinPattern0 = 0;
            REAL withinPattern1 = 0;
            REAL withinPattern2 = 0;
            REAL withinPattern3 = 0;
        
            for (int c = 0; c < categoryCount; ++c) {

                const KW_GLOBAL_VAR REAL* KW_RESTRICT prePartials = partials0 + preOffset + patternOffset + totalPatterns * PADDED_STATE_COUNT * c;
                const KW_GLOBAL_VAR REAL* KW_RESTRICT postPartials = partials0 + postOffset + patternOffset + totalPatterns * PADDED_STATE_COUNT * c;

                if (pattern < totalPatterns) {
                    pre[tx] = prePartials[tx];  // Coalesced global memory read
                    post[tx] = postPartials[tx]; // Coalesced global memory read
                } else {
                    pre[tx] = 0.0;
                    post[tx] = 0.0;
                }

                const REAL scale =  edgeLength * categoryRates[c]; // TODO Better in constant memory?
                const REAL weight = categoryWeights[c]; // TODO Better in constant memory?

                // Inner product
                denominator[tx] = pre[tx] * post[tx];
                if (tx < 8) {
                    denominator[tx << 1] += denominator[tx << 1 | 0x1];
                }

                KW_LOCAL_FENCE; // TODO necessary? in same warp

                if (tx < 4) {
                    denominator[tx << 2] += denominator[tx << 2 | 0x2];
                }

                KW_LOCAL_FENCE; // TODO necessary? is same warp

                txPatternDenominator += denominator[tx & 0xC] * weight;
                
                post[tx] *= weight * scale;

                KW_LOCAL_FENCE; // TODO Merge with fence above

                withinPattern0 += pre[4 * 0 | state] * post[4 * 0 | pat];
                withinPattern1 += pre[4 * 1 | state] * post[4 * 1 | pat];
                withinPattern2 += pre[4 * 2 | state] * post[4 * 2 | pat];
                withinPattern3 += pre[4 * 3 | state] * post[4 * 3 | pat];
            }

            patternDenominator[tx] = txPatternDenominator;

            if (tx < 4) {
                patternWeights[tx] = inPatternWeights[pattern + tx];
            }            

            KW_LOCAL_FENCE;

            if (patternDenominator[4 * 0] > 0.0) {
                acrossPatterns += withinPattern0 * patternWeights[0] / patternDenominator[4 * 0];
            }

            if (patternDenominator[4 * 1] > 0.0) {
                acrossPatterns += withinPattern1 * patternWeights[1] / patternDenominator[4 * 1];
            }
            
            if (patternDenominator[4 * 2] > 0.0) {
                acrossPatterns += withinPattern2 * patternWeights[2] / patternDenominator[4 * 2];
            }
            
            if (patternDenominator[4 * 3] > 0.0) {
                acrossPatterns += withinPattern3 * patternWeights[3] / patternDenominator[4 * 3];
            } // TODO Vectorize
        }
    }

    KW_LOCAL_FENCE;

    const int destination = (nodeId * numPatternBlocks + patternBlockId) * 16;

    if (accumulate) {
        acrossPatterns += out[destination + tx];
    }
    
    //out[destination + tx] = withinPattern0;
     out[destination + tx] = acrossPatterns;
#endif
}
