
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