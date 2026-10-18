/*
 *
 * Copyright 2026 Phylogenetic Likelihood Working Group
 *
 * This file is part of BEAGLE.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 *
 * @author Marc Suchard
 *
 * TEST: kernelUpPartialsFused, the one launch of BeagleGPUFusedImpl::updatePartials (BEAGLE_FLAG_PARALLELOPS_FUSED).
 * The loop over the operations and the partials-partials (PP), states-partials (SP) and states-states (SS) dispatch
 * run inside the kernel; every per-operation body is a split macro of kernels4.cu or kernelsX.cu. Like
 * kernelsSpectralIfDef4.cu, this file has no preamble of its own. For OpenCL, make_opencl_fused_kernels.sh appends it
 * to every string of BeagleOpenCLFused_kernels.h; for CUDA, kernels4.cu and kernelsX.cu include it (inside their
 * extern "C") when CUDA_FUSED is defined.
 *
 * Op table: 8 unsigned ints per operation, ptrOffsets[8 * k + ...]:
 *   0  kind | mode << 2; kind 1 PP, 2 SP, 3 SS; mode 0 none, 1 read, 2 write
 *   1  0
 *   2  child 1 offset (partials for PP; states for SP, SS)
 *   3  child 2 offset (partials for PP, SP; states for SS)
 *   4  parent partials offset
 *   5  child-1 matrix offset (SP: the states child's)
 *   6  child-2 matrix offset
 *   7  scale buffer offset (the written or read factors)
 * Slots 2-7 are the multigrid record's, read by DETERMINE_INDICES_4_MULTI_2_GPU. One cumulative scale buffer per call
 * (cumulativeOffset, fusedFlags & FUSED_HAS_CUMULATIVE); factors are log(max) under fusedFlags & FUSED_SCALERS_LOG.
 *
 * Every read of an earlier operation's output in the same launch is a same-work-item read-after-write, except the
 * factors a write operation stores, which other work-items of the same group read in later read operations:
 * KW_GLOBAL_FENCE closes every write operation.
 */

#ifndef FW_OPENCL_CPU

#if defined(FW_OPENCL) && !defined(FW_TINYGPU_AMD)
#define KW_GLOBAL_FENCE barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)
#else
#define KW_GLOBAL_FENCE KW_LOCAL_FENCE        /* __syncthreads(): orders global and shared within the block */
#endif
#define FUSED_KIND(W) ((W) & 3)
#define FUSED_MODE(W) (((W) >> 2) & 3)
#define FUSED_HAS_CUMULATIVE 1
#define FUSED_SCALERS_LOG    2
/* the factor of kernelPartialsDynamicScaling{,ScalersLog,Accumulate,AccumulateScalersLog}; MAXV is the raw max. The
   cumulative add is skipped when MAXV == 0, which equals the base's (it adds log(1.0) = 0, or skips it) */
#define FUSED_WRITE_SCALE_FACTOR_GPU(MAXV, PAT) \
    if (MAXV == 0) { scaleOut[PAT] = (fusedFlags & FUSED_SCALERS_LOG) ? 0.0 : 1.0; } \
    else { REAL logMax = log(MAXV); \
           scaleOut[PAT] = (fusedFlags & FUSED_SCALERS_LOG) ? logMax : MAXV; \
           if (fusedFlags & FUSED_HAS_CUMULATIVE) cumulativeScaling[PAT] += logMax; }

#if (PADDED_STATE_COUNT == 4)
/* Block (16, R = 16 / C, C): each group computes all C categories of its 4 * R patterns, so a write operation needs no
   category loop; category KW_LOCAL_ID_2 for the whole launch. Local tiles are double-buffered by operation parity, so a
   no-write operation has one barrier (its matrix fence): operation k + 2 rewrites tile [k & 1] only after every
   work-item has passed operation k + 1's matrix fence, which follows all of operation k's tile reads. fsRed and fsMax
   are rewritten only after the next write operation's matrix fence, which follows the earlier write operation's reads.
   Sizes are arguments (no KW_NUM_GROUPS_* or KW_LOCAL_SIZE_*); partials and scaleFactors are written in-kernel, so
   neither is const or restrict. */
KW_GLOBAL_KERNEL void
#ifdef CUDA
__launch_bounds__(256)
#endif
kernelUpPartialsFused(KW_GLOBAL_VAR REAL* partials,
                      KW_GLOBAL_VAR int* KW_RESTRICT states,
                      KW_GLOBAL_VAR REAL* KW_RESTRICT matrices,
                      KW_GLOBAL_VAR REAL* scaleFactors,
                      const KW_GLOBAL_VAR unsigned int* KW_RESTRICT ptrOffsets,
                      int opCount, int totalPatterns, int matrixCount,
                      int layoutParam, int cumulativeOffset, int fusedFlags) {
    KW_LOCAL_MEM REAL fsPartials1[2][256], fsPartials2[2][256];     /* [op parity][category tile 16 * R] */
    KW_LOCAL_MEM REAL fsMatrix1[2][16][16], fsMatrix2[2][16][16];   /* [op parity][category][16] */
    KW_LOCAL_MEM REAL fsRed[16][16];                                /* [category * R + patIdx][tx] */
    KW_LOCAL_MEM REAL fsMax[64];                                    /* [4 * patIdx + pat] */
    KW_GLOBAL_VAR REAL* cumulativeScaling = scaleFactors + cumulativeOffset;
    const int rowsPerCat = layoutParam;
    const int catLocal   = KW_LOCAL_ID_2;
    const int startPat   = KW_GROUP_ID_0 * 4 * rowsPerCat;
    const int endPattern = totalPatterns;              /* the padded pattern count: guard and category stride */
    DETERMINE_INDICES_4_MULTI_1_THREAD_GPU(catLocal)   /* tx, state, pat, patIdx, matrix, pattern, deltas, x2, u */
    const int row = catLocal * rowsPerCat + patIdx;
    for (int k = 0; k < opCount; k++) {                /* uniform trip count */
        const int opIndexPtr = k * 8;
        const unsigned int kind = FUSED_KIND(ptrOffsets[opIndexPtr]), mode = FUSED_MODE(ptrOffsets[opIndexPtr]);
        const int b = k & 1;
        KW_LOCAL_VAR REAL* sPartials1 = fsPartials1[b] + 16 * rowsPerCat * catLocal;   /* the names the macros use */
        KW_LOCAL_VAR REAL* sPartials2 = fsPartials2[b] + 16 * rowsPerCat * catLocal;
        KW_LOCAL_VAR REAL* sMatrix1   = fsMatrix1[b][catLocal];
        KW_LOCAL_VAR REAL* sMatrix2   = fsMatrix2[b][catLocal];
        DETERMINE_INDICES_4_MULTI_2_GPU()              /* partials3, matrix1, matrix2 */
        KW_GLOBAL_VAR REAL* partials1 = partials + ptrOffsets[opIndexPtr + 2];          /* PP */
        KW_GLOBAL_VAR REAL* partials2 = partials + ptrOffsets[opIndexPtr + 3];          /* PP, SP */
        KW_GLOBAL_VAR int*  states1   = states   + ptrOffsets[opIndexPtr + 2];          /* SP, SS */
        KW_GLOBAL_VAR int*  states2   = states   + ptrOffsets[opIndexPtr + 3];          /* SS */
        if (kind == 1)      { LOAD_PARTIALS_PARTIALS_4_MULTI_PART_BODY_GPU() }        /* zero-filled beyond endPattern */
        else if (kind == 2) { LOAD_PARTIALS_SINGLE_4_MULTI_PART_BODY_GPU() }
        LOAD_MATRIX_4_MULTI_ADVANCE_GPU()              /* this work-item's category */
        LOAD_MATRIX_4_MULTI_BODY_GPU()                 /* KW_LOCAL_FENCE: the operation's only barrier without a write */
        REAL v = 0;
        if (pattern < endPattern) {
            if (kind == 1)      { SUM_PARTIALS_PARTIALS_4_GPU(); v = sum1 * sum2; }
            else if (kind == 2) { SUM_STATES_PARTIALS_4_GPU();   v = sum1 * sum2; }
            else                { SUM_STATES_STATES_4_P_GPU(v) }
            if (mode == 1) v = v / scaleFactors[ptrOffsets[opIndexPtr + 7] + pattern];
        }
        if (mode == 2) {                               /* uniform */
            KW_GLOBAL_VAR REAL* scaleOut = scaleFactors + ptrOffsets[opIndexPtr + 7];
            fsRed[row][tx] = v;  KW_LOCAL_FENCE;
            FIND_MAX_PARTIALS_STATE_4_REDUCE_GPU(fsRed[row])            /* states first, then categories (the base's order) */
            if (state == 0 && catLocal == 0) {
                FIND_MAX_PARTIALS_MATRIX_4_P_GPU(fsRed[m * rowsPerCat + patIdx][tx], fsMax[multBy4(patIdx) + pat])
                if (pattern < endPattern) { FUSED_WRITE_SCALE_FACTOR_GPU(fsMax[multBy4(patIdx) + pat], pattern) }
                if (fsMax[multBy4(patIdx) + pat] == 0) fsMax[multBy4(patIdx) + pat] = 1.0;
            }
            KW_GLOBAL_FENCE;                           /* fsMax to the group; the factor to later read operations */
            v = v / fsMax[multBy4(patIdx) + pat];
        }
        if (pattern < endPattern) partials3[u] = v;
    }
}
#else
/* the generic-state kernel: fused plan F4 */
#endif

#endif /* FW_OPENCL_CPU */
