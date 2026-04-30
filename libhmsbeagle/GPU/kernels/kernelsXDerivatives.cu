}

#ifdef CUDA_TENSOR_CORES
#include <mma.h>
#endif

extern "C" {

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

