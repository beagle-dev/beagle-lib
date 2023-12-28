
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

    const int WMMA_M = 8;
    const int WMMA_N = 8;
    const int WMMA_K = 4;

    int y = deltaPartialsByState + deltaPartialsByMatrix;
    KW_LOCAL_MEM REAL sPartials1[PATTERN_BLOCK_SIZE * 4 * 4];
    KW_LOCAL_MEM REAL sPartials2[PATTERN_BLOCK_SIZE * 4 * 4];

// Indices to permute ShM for partialss
// X -> threadIdx.x or state and Y -> threadIdx.y or patIdx
// (int(X/8): Splits 32 values into groups of 4.
// ((Y & 1) * -2 + 1)): For strip-mined layout: If patIdx is even increment by 1 else by -1
// & 0x07 To cycle within the limits [0,1,2,3,4,5,6,7] i.e., [0, ... , PADDED_STATE_COUNT/WMMA_K]
#define GET_SMEM_ROW_PARTIALS(X) ((X / WMMA_K) & 0x07)
#define GET_BANK_GROUP_PARTIALS(X,Y) ((Y + (X/WMMA_K) * ((Y & 1) * -2 + 1)) & ((PADDED_STATE_COUNT/WMMA_K) - 1)) // 0x07 should be generalized to & PADDED_STATE_COUNT/WMMA_K - 1
#define GET_SMEM_COL_PARTIALS(X,Y) (GET_BANK_GROUP_PARTIALS(X,Y) * WMMA_K + (X % WMMA_K))
#define GET_SMEM_OFFSET_PARTIALS(X,Y) (GET_SMEM_ROW_PARTIALS(X) * PADDED_STATE_COUNT + GET_SMEM_COL_PARTIALS(X, Y))

    /* copy PADDED_STATE_COUNT * PATTERN_BLOCK_SIZE lengthed partials*/
    if (pattern < endPattern) {
        // Read in permuted for partials1
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
        /* Write transpose of both matrices since M is loaded row-wise */
        sMatrix1[tx] = matrix1[tx];
        sMatrix2[multBy4(state) | pat] = matrix2[tx];
    }
    KW_LOCAL_FENCE;

    double a2 = 0, b2 = 0, res22 = 0, res21 = 0, a1 = 0, b1 = 0, res11 = 0, res12 = 0;

    int warpSize = 32;
    int warpState = tx / warpSize;
    int warpPattern = patIdx;
    int warpIdx = warpState + warpPattern * 0.5; // blockDim.x is half a warp

    int reg_row = tx % 4;
    int reg_col = tx / 4;
    int reg_row_partials = tx % 16;
    int reg_col_partials = (patIdx % 2);

    if (patIdx % 2 == 0) {
        a2 = sMatrix2[reg_col * 4 + reg_row];
    } else {
        a2 = 0;
    }

    b2 = sPartials2[warpIdx * 32 + reg_col_partials * 16 + reg_row_partials];

    asm("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
            : "=d"(res21), "=d"(res22)
            : "d"(a2), "d"(b2), "d"(res21), "d"(res22));

    int laneId = tx + (patIdx % 2) * 16;

    // TODO: Permute ShM to avoid bank conflicts.
    if(laneId < 16) { // Ignore lower half of matrices. We only need 4 x 8
        int partials1Index = (patIdx/2) * 32 + ((laneId * 2) % 8) * 4 + (laneId * 2) / WMMA_N;
        sPartials1[partials1Index] = sPartials1[partials1Index] * res21;
        sPartials1[partials1Index + 4] = sPartials1[partials1Index + 4] * res22;
    }

    if (patIdx % 2 == 0) {
        a1 = sMatrix1[reg_col * 4 + reg_row];
    } else {
        a1 = 0;
    }

    b1 = sPartials1[warpIdx * 32 + reg_col_partials * 16 + reg_row_partials];

    asm("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
            : "=d"(res11), "=d"(res12)
            : "d"(a1), "d"(b1), "d"(res11), "d"(res12));

    int patternBlock = __umul24(KW_GROUP_ID_0, PATTERN_BLOCK_SIZE * 4);
    u = patternBlock * 4 + deltaPartialsByMatrix;

    if(laneId < 16) {
        if(patternBlock + patIdx * 4 + ((laneId * 2) % 8) < endPattern){
            partials3[u + (patIdx/2) * 32 + ((laneId * 2) % 8) * 4 + (laneId * 2) / WMMA_N] = res11;
        }

        if(patternBlock + patIdx * 4 + ((laneId * 2) % 8) + 1 < endPattern)
            partials3[u + 4 + (patIdx/2) * 32 + ((laneId * 2) % 8) * 4 + (laneId * 2) / WMMA_N] = res12;

    }

#endif // FW_OPENCL_CPU
}