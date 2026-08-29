/*
 *
 * Copyright 2009 Phylogenetic Likelihood Working Group
 *
 * This file is part of BEAGLE.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 *
 * @author Marc Suchard
 * @author Daniel Ayres
 */

#ifdef CUDA
    #include "libhmsbeagle/GPU/GPUImplDefs.h"
    #include <stdlib.h>
    #include <string.h>
    #include <stdio.h>
    extern "C" {
#elif defined(FW_OPENCL)
    #ifdef DOUBLE_PRECISION
        #pragma OPENCL EXTENSION cl_khr_fp64: enable
    #endif
    #define __umul24(x, y) (x * y)
#endif //FW_OPENCL

#if (!defined DOUBLE_PRECISION && defined FP_FAST_FMAF) || (defined DOUBLE_PRECISION && defined FP_FAST_FMA)
    #define FMA(x, y, z) (z = fma(x, y, z))
#else //FP_FAST_FMA
    #define FMA(x, y, z) (z += x * y)
#endif //FP_FAST_FMA

#if (defined CUDA) && (defined DOUBLE_PRECISION) &&  (__CUDA_ARCH__ < 600)
    __device__ double atomicAdd(double* address, double val)
    {
        unsigned long long int* address_as_ull =
                                  (unsigned long long int*)address;
        unsigned long long int old = *address_as_ull, assumed;

        do {
            assumed = old;
            old = atomicCAS(address_as_ull, assumed,
                            __double_as_longlong(val +
                                   __longlong_as_double(assumed)));

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
        } while (assumed != old);

        return __longlong_as_double(old);
    }
#endif

///////////////////////////////////////////////////////////////////////////////

KW_GLOBAL_KERNEL void kernelReorderPatterns(      KW_GLOBAL_VAR REAL*             dPartials,
                                                  KW_GLOBAL_VAR int*              dStates,
                                                  KW_GLOBAL_VAR int*              dStatesSort,
                                            const KW_GLOBAL_VAR int*  KW_RESTRICT dTipOffsets,
                                            const KW_GLOBAL_VAR int*  KW_RESTRICT dTipTypes,
                                            const KW_GLOBAL_VAR int*  KW_RESTRICT dPatternsNewOrder,
                                            const KW_GLOBAL_VAR REAL* KW_RESTRICT dPatternWeights,
                                                  KW_GLOBAL_VAR REAL* KW_RESTRICT dPatternWeightsSort,
                                                                int               patternCount,
                                                                int               paddedPatternCount) {
#ifdef FW_OPENCL_CPU
    int state      = 0;
    int pattern    = KW_LOCAL_ID_0 + KW_GROUP_ID_0 * KW_LOCAL_SIZE_0;
#else
    int state      = KW_LOCAL_ID_0;
    int pattern    = KW_LOCAL_ID_1 + KW_GROUP_ID_0 * KW_LOCAL_SIZE_1;
#endif
    int stateCount = PADDED_STATE_COUNT;
    int category   = KW_GROUP_ID_1;
    int tip        = KW_GROUP_ID_2;
    int tipCount   = KW_NUM_GROUPS_2;

    if (pattern < patternCount) {
        int patternSorted  = dPatternsNewOrder[pattern];

        if (dTipTypes[tip] == 0) {
            int categoryOffset = category * stateCount * paddedPatternCount;

            int sortIndex   = categoryOffset + patternSorted * stateCount;
            int originIndex = categoryOffset + pattern       * stateCount;

            const KW_GLOBAL_VAR REAL* KW_RESTRICT partialOriginal = dPartials + dTipOffsets[tip];
                  KW_GLOBAL_VAR REAL* KW_RESTRICT partialSorted   = dPartials + dTipOffsets[tip+tipCount];

#ifdef FW_OPENCL_CPU
            for (int i=0; i < stateCount; i++) {
                partialSorted[sortIndex+i] = partialOriginal[originIndex+i];
            }
#else
            sortIndex += state;
            originIndex += state;
            partialSorted[sortIndex] = partialOriginal[originIndex];
#endif
        } else if (state == 0) {
            const KW_GLOBAL_VAR int* KW_RESTRICT stateOriginal = dStates     + dTipOffsets[tip];
                  KW_GLOBAL_VAR int* KW_RESTRICT stateSorted   = dStatesSort + dTipOffsets[tip+tipCount];

            stateSorted[patternSorted] = stateOriginal[pattern];
        }

        if (state == 0 && category == 0 && tip == 0) {
            dPatternWeightsSort[patternSorted] = dPatternWeights[pattern];
        }
    }
}

KW_GLOBAL_KERNEL void kernelMatrixMulADBMulti(KW_GLOBAL_VAR REAL* dMatrices,
                                              KW_GLOBAL_VAR unsigned int* offsets,
                                              KW_GLOBAL_VAR REAL* Alist,
                                              KW_GLOBAL_VAR REAL* Dlist,
                                              KW_GLOBAL_VAR REAL* Blist,
                                              KW_GLOBAL_VAR REAL* distanceQueue,
                                              int length,
                                              int wB,
                                              int totalMatrix) {

    int wMatrix = KW_GROUP_ID_0 % totalMatrix;
    int offIndex = wMatrix * 3;

    // Block index
    int bx = KW_GROUP_ID_0 / totalMatrix;
    int by = KW_GROUP_ID_1;

    // Thread index
    int tx = KW_LOCAL_ID_0;
    int ty = KW_LOCAL_ID_1;
    int BLOCKS = KW_NUM_GROUPS_1;

    KW_GLOBAL_VAR REAL* C = dMatrices + offsets[offIndex];
    KW_GLOBAL_VAR REAL* B = Blist + offsets[offIndex + 1]; // dEvec
    KW_GLOBAL_VAR REAL* A = Alist + offsets[offIndex + 1]; // dIevc
    KW_GLOBAL_VAR REAL* D = Dlist + offsets[offIndex + 2]; // dEigenValues
    REAL distance = distanceQueue[wMatrix];

    const int EDGE = PADDED_STATE_COUNT - (BLOCKS - 1) * MULTIPLY_BLOCK_SIZE;

    // Step size used to iterate through the sub-matrices of A
    int aStep = MULTIPLY_BLOCK_SIZE;

    // Step size used to iterate through the sub-matrices of B
    int bStep = MULTIPLY_BLOCK_SIZE * PADDED_STATE_COUNT;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    REAL Csub = 0;

    int a = PADDED_STATE_COUNT * MULTIPLY_BLOCK_SIZE * by;
    int b = MULTIPLY_BLOCK_SIZE * bx;
    int d = 0; //MULTIPLY_BLOCK_SIZE * bx;

    KW_LOCAL_MEM REAL As[MULTIPLY_BLOCK_SIZE][MULTIPLY_BLOCK_SIZE];
    KW_LOCAL_MEM REAL Bs[MULTIPLY_BLOCK_SIZE][MULTIPLY_BLOCK_SIZE];
    KW_LOCAL_MEM REAL Ds[MULTIPLY_BLOCK_SIZE];

    for (int i = 0; i < BLOCKS - 1; i++) {

        if (ty == 0)
            Ds[tx] = exp(D[d + tx] * distance);

        As[ty][tx] = A[a + PADDED_STATE_COUNT * ty + tx];
        Bs[ty][tx] = B[b + PADDED_STATE_COUNT * ty + tx];

        KW_LOCAL_FENCE;

        for (int k = 0; k < MULTIPLY_BLOCK_SIZE; ++k)
            Csub += As[ty][k] * Ds[k] * Bs[k][tx];

        KW_LOCAL_FENCE;

        a += aStep;
        b += bStep;
        d += MULTIPLY_BLOCK_SIZE;
    }

    // Last block is too long
    if (tx < EDGE && ty < EDGE) {
        if (ty == 0)
            Ds[tx] = exp(D[d + tx] * distance);

        As[ty][tx] = A[a + PADDED_STATE_COUNT * ty + tx];
        Bs[ty][tx] = B[b + PADDED_STATE_COUNT * ty + tx];

    } else {

        if (ty == 0)
            Ds[tx] = 0;

        As[ty][tx] = 0;
        Bs[ty][tx] = 0;
    }

    KW_LOCAL_FENCE;

    for (int k = 0; k < EDGE; k++)
        Csub += As[ty][k] * Ds[k] * Bs[k][tx];

    KW_LOCAL_FENCE;

    // Write the block sub-matrix to device memory;
    // each thread writes one element

    if ((tx < EDGE || bx < BLOCKS - 1) && (ty < EDGE || by < BLOCKS - 1)) { // It's OK to write
        if (Csub < 0)
            C[PADDED_STATE_COUNT* MULTIPLY_BLOCK_SIZE * by + MULTIPLY_BLOCK_SIZE * bx +
              PADDED_STATE_COUNT * ty + tx] = 0;
        else
            C[PADDED_STATE_COUNT* MULTIPLY_BLOCK_SIZE * by + MULTIPLY_BLOCK_SIZE * bx +
              PADDED_STATE_COUNT * ty + tx] = Csub;
    }
}

KW_GLOBAL_KERNEL void kernelMatrixMulADB(KW_GLOBAL_VAR REAL* dMatrices,
                                   KW_GLOBAL_VAR unsigned int* listC,
                                   KW_GLOBAL_VAR REAL* A,
                                   KW_GLOBAL_VAR REAL* D,
                                   KW_GLOBAL_VAR REAL* B,
                                   KW_GLOBAL_VAR REAL* distanceQueue,
                                   int length,
                                   int wB,
                                   int totalMatrix) {

    int wMatrix = KW_GROUP_ID_0 % totalMatrix;

    // Block index
    int bx = KW_GROUP_ID_0 / totalMatrix;
    int by = KW_GROUP_ID_1;

    // Thread index
#if defined(FW_TINYGPU_HYBRID_NV)
    // TODO.md Phase 111: this from-scratch GSP-RM/driver stack has a
    // confirmed defect (Phase 106-110) triggered specifically by
    // launching this kernel with a 2D block (blockDim.y>1) -- both the
    // local-memory register-spill mechanism and Ds[]'s own shared-
    // memory broadcast corrupt every thread with ty>0, 100%
    // reproducibly, in two mechanistically distinct ways, both of which
    // vanish completely (320/320 correct, Phase 109/110) when the same
    // logical (ty,tx) values are instead derived from a single flat
    // block dimension. This is the real fix: dispatch this kernel with
    // local_size=(MULTIPLY_BLOCK_SIZE*MULTIPLY_BLOCK_SIZE,1,1) (a flat
    // block) for the NV-eGPU build specifically -- see
    // GPUInterfaceTinyGPUHybridNV.cpp/nv_dispatch_daemon.py for the
    // corresponding dispatch-side change -- and derive tx,ty from the
    // single flat KW_LOCAL_ID_0 here instead of the native (buggy, on
    // this stack) KW_LOCAL_ID_0/KW_LOCAL_ID_1 pair. Every other use of
    // tx/ty throughout the rest of THIS kernel (kernelMatrixMulADB
    // only -- kernelMatrixMulADBFirstDeriv/SecondDeriv/Multi/Complex are
    // deliberately untouched, not yet tested under this fix) is already
    // just a reference to these two local variables (confirmed:
    // KW_LOCAL_ID_0/KW_LOCAL_ID_1 are read nowhere else in this
    // function), so this is the ONLY change needed -- everything
    // downstream (As/Bs/Ds loading, the Csub reduction, the boundary
    // EDGE guard, the final write) is untouched and automatically
    // inherits correct, safely-bounded (0..MULTIPLY_BLOCK_SIZE-1)
    // logical tx/ty values. Gated strictly to the NV-eGPU build
    // (FW_TINYGPU_HYBRID_NV, defined only by that backend's own compile
    // step, mirroring FW_TINYGPU_HYBRID_AMD's established convention) --
    // every other backend (plain CUDA, OpenCL, the AMD hybrid build) is
    // completely unaffected, still using the native 2D dispatch that's
    // correct on that hardware.
#if defined(FW_TINYGPU_HYBRID_NV_SINGLE_TILE)
    // TODO.md Phase 126, user's proposal B: every PADDED_STATE_COUNT this
    // kernel actually supports (see GPUImplDefs.h) is a multiple of
    // MULTIPLY_BLOCK_SIZE (16) *except* 4 -- the one config this entire
    // investigation has ever tested. For every other supported state
    // count, BLOCKS*16 exactly covers PADDED_STATE_COUNT with no
    // leftover tile, so EDGE (below) always equals 16 and every one of
    // the 256 dispatched threads does real work. For PADDED_STATE_COUNT
    // ==4 specifically, the general MULTIPLY_BLOCK_SIZE=16-wide dispatch
    // wastes 240 of 256 threads on the EDGE-boundary guard's "else"
    // (padding) branch -- a pattern unique to this one config, never
    // exercised by any other supported model. This branch dispatches
    // exactly PADDED_STATE_COUNT^2 threads/block (flat) instead of
    // MULTIPLY_BLOCK_SIZE^2, so every dispatched thread lands in
    // [0,PADDED_STATE_COUNT) on both axes -- EDGE (always ==
    // PADDED_STATE_COUNT here, since BLOCKS==1 is guaranteed by the
    // #error guard below) is then trivially satisfied by construction,
    // and the "else"/padding branch becomes genuine dead code rather
    // than 240-of-256 threads actively executing it. Requires a smaller
    // dispatch (local_size=(PADDED_STATE_COUNT^2,1,1)) on the host side
    // -- see nv_real_kernel_probe.py's --single-tile-dispatch flag.
    // Deliberately a SEPARATE macro from FW_TINYGPU_HYBRID_NV (not
    // auto-enabled just because PADDED_STATE_COUNT<=MULTIPLY_BLOCK_SIZE)
    // so the already-established --flat-dispatch behavior (Phase 111),
    // which dispatches the full 256 threads, is completely unaffected
    // unless this macro is explicitly also defined.
#if PADDED_STATE_COUNT > MULTIPLY_BLOCK_SIZE
#error "FW_TINYGPU_HYBRID_NV_SINGLE_TILE requires PADDED_STATE_COUNT <= MULTIPLY_BLOCK_SIZE (single-tile only)"
#endif
    int tx = KW_LOCAL_ID_0 % PADDED_STATE_COUNT;
    int ty = KW_LOCAL_ID_0 / PADDED_STATE_COUNT;
#else
    int tx = KW_LOCAL_ID_0 % MULTIPLY_BLOCK_SIZE;
    int ty = KW_LOCAL_ID_0 / MULTIPLY_BLOCK_SIZE;
#endif
#else
    int tx = KW_LOCAL_ID_0;
    int ty = KW_LOCAL_ID_1;
#endif
    int BLOCKS = KW_NUM_GROUPS_1;

#if defined(CUDA) && !defined(FW_TINYGPU)
    KW_LOCAL_MEM REAL* C;
    KW_LOCAL_MEM REAL distance;
    if (tx == 0 && ty == 0) {
        C = dMatrices + listC[wMatrix]; // Non-coalescent read
        distance = distanceQueue[wMatrix]; // Non-coalescent read
    }
#elif defined(FW_OPENCL) || defined(FW_TINYGPU)
    // FW_TINYGPU: same as OpenCL here, not because of any OpenCL-specific
    // requirement, but because this avoids broadcasting a global-memory
    // pointer through shared memory and having every thread dereference it
    // generically. That requires the GPU's shared/local-memory generic-
    // addressing windows to be configured exactly right; on the TinyGPU
    // backend's from-scratch driver they aren't (yet), which faults every
    // thread in every block deterministically. See STATUS.md/TODO.md on
    // the usb branch ("kernelMatrixMulADB specifically is broken").
    KW_GLOBAL_VAR REAL* C;
    REAL distance;
#if defined(FW_TINYGPU) && defined(TINYGPU_BISECT_NO_LISTC)
    // Opt-in, single-substitution bisection experiment (TODO.md "PICK UP
    // HERE" -> NV Phase 65): every candidate tried on the real kernel so
    // far (exp(), the main loop, the A/B pointer split) has been ruled
    // out, and a from-scratch single-CTA probe (Phase 64) proved the
    // residual bug requires multiple concurrent CTAs -- something real
    // and multi-CTA-specific is still unaccounted for. One genuine
    // difference no probe has ever included: this read is *data-
    // dependent addressing* -- listC[wMatrix] is a value loaded from
    // global memory that then becomes part of a pointer computation
    // (C's own address), not just arithmetic on blockIdx.x/threadIdx.x
    // the way every probe's addressing has been. Substitutes the
    // mathematically *equivalent* closed-form arithmetic (proven, not
    // guessed: BeagleGPUImpl.hpp's hPtrQueue[wMatrix] =
    // probabilityIndices[i]*kIndexOffsetMat + j*categoryOffset reduces to
    // exactly wMatrix*kMatrixSize for this test's specific, sequential
    // probabilityIndices={0,1,2,3} -- verified against the source
    // computing listC's real values, not assumed from one hardware log)
    // -- so C's real, downstream value and every other real computation
    // stay numerically identical; only the *mechanism* (data-dependent
    // load vs. pure arithmetic) changes. distanceQueue's read is
    // deliberately left untouched -- it feeds a real *value* (rate/
    // branch-length data with no computable closed form), not an
    // *address*, so it's a different variable from what this experiment
    // targets.
    C = dMatrices + wMatrix * PADDED_STATE_COUNT * PADDED_STATE_COUNT;
#else
    C = dMatrices + listC[wMatrix];
#endif
    distance = distanceQueue[wMatrix];
#endif

    KW_LOCAL_FENCE;

#if defined(FW_TINYGPU) && defined(TINYGPU_DEBUG_DUMP_LOCAL_MEM_FLAT_W0)
    // Opt-in, additive-only, EARLY-RETURN probe (TODO.md "PICK UP HERE"
    // -> NV Phase 109): user's direct question -- does the tx+4ty local-
    // memory aliasing (Phase 106) depend on the kernel's real 2D
    // (16,16,1) block dispatch specifically, or does it persist even
    // when every thread's "row/column" identity is linearized onto a
    // single flat dimension (blockDim=(256,1,1))? This probe is
    // dispatched (via --local-mem-flat-w0) with local_size=(256,1,1)
    // instead of the kernel's normal (16,16,1) -- meaning native ty
    // (KW_LOCAL_ID_1) is always 0 for every thread; logicalTy/logicalTx
    // below reconstruct the *same* (0..15)x(0..15) logical space the
    // kernel's native ty,tx cover under its normal dispatch, computed
    // instead from the single flat dimension (tx = KW_LOCAL_ID_0,
    // 0..255) -- same total thread count, same logical range, only HOW
    // it's computed (and the real launch shape) differs.
    //
    // MUST return before reaching the "Last block" guard below (~line
    // 424): that guard writes As[ty][tx]/Bs[ty][tx] *unconditionally*
    // for every thread (both branches), and As/Bs are declared
    // [MULTIPLY_BLOCK_SIZE][MULTIPLY_BLOCK_SIZE] = [16][16] (confirmed
    // via BeagleOpenCL_kernels.h's MULTIPLY_BLOCK_SIZE_SP=16) -- under a
    // real (256,1,1) dispatch, native tx=KW_LOCAL_ID_0 ranges 0..255, so
    // As[ty][tx] for tx>=16 would be a genuine out-of-bounds shared-
    // memory write for 240 of the 256 threads. This kernel's own source
    // already documents two *branchless* rewrites of that exact guard
    // hanging real hardware (see the "Last block is too long" comment
    // below) -- an unconditional out-of-bounds write here is exactly the
    // class of change this investigation treats as fault-prone.
    // Following the *already-established*, hardware-proven-safe
    // early-return pattern TINYGPU_DEBUG_BROADCAST_PROBE (directly
    // below) uses for the same reason.
    //
    // Otherwise identical to TINYGPU_DEBUG_DUMP_LOCAL_MEM_W0 -- same
    // 16-iteration runtime-varying index recurrence (already verified to
    // force real LDL/STL traffic and to compute the value arithmetic
    // correctly), same 16-float diagnostic-region footprint/offset, same
    // local_mem_expected() formula on the Python side. wMatrix 0 only.
    if (KW_GROUP_ID_0 == 0) {
        const int probeEdge = 4;   // matches this test's real EDGE exactly
        // tx is the fast-varying (warp-contiguous) component, matching
        // its native role as the *inner* array index throughout the rest
        // of this kernel (As[ty][tx], Bs[k][tx], Ds[tx]) -- preserving
        // that role under the flat linearization keeps this a fair,
        // like-for-like analog of the native (16,16,1) dispatch, not an
        // incidental swap.
        int logicalTy = tx / 16;
        int logicalTx = tx % 16;

        volatile REAL spillTest[4];
        int idx = logicalTx % 4;
        for (int i = 0; i < 16; i++) {
            spillTest[idx] = (REAL) (1000 + logicalTy * 100 + logicalTx * 10 + idx);
            idx = (idx + totalMatrix + i) % 4;
        }
        KW_GLOBAL_VAR REAL* pt = dMatrices
            + 2 * totalMatrix * PADDED_STATE_COUNT * PADDED_STATE_COUNT
            + KW_GROUP_ID_0 * PADDED_STATE_COUNT * PADDED_STATE_COUNT;
        if (logicalTx < probeEdge && logicalTy < probeEdge) {
            pt[logicalTy * probeEdge + logicalTx] = spillTest[idx];
        }
    }
    return;
#endif

#if defined(FW_TINYGPU) && defined(TINYGPU_DEBUG_DUMP_SHARED_BROADCAST_FLAT_W0)
    // Opt-in, additive-only, EARLY-RETURN probe (TODO.md "PICK UP HERE"
    // -> NV Phase 110): user's direct follow-up to Phase 109's 320/320
    // result -- does Ds[]'s own broadcast-collapse bug (Phase 90-103:
    // ty>0 readers see Ds[0]'s value regardless of requested index) ALSO
    // disappear under a flat (256,1,1) dispatch, the same way local
    // memory's tx+4ty aliasing did? Mirrors TINYGPU_DEBUG_DUMP_LOCAL_MEM_
    // FLAT_W0's exact safety design (placed before the kernel's own
    // "Last block" guard, unconditional early return -- see that macro's
    // own comment for why this is required) and logicalTy/logicalTx
    // convention, but tests the WRITE-BY-SUBSET/READ-BY-ALL shared-
    // memory broadcast pattern instead of local memory: a dedicated
    // sDs[16] shared array (own storage, not the real Ds[] -- the real
    // one is never reached under this early return), written only by
    // logicalTy==0 threads using the *exact* real formula
    // (exp(D[tx]*distance), d=0 implied for this investigation's
    // BLOCKS==1 config -- omitting +d here matches TINYGPU_DEBUG_
    // BROADCAST_PROBE's own established precedent, above, for this same
    // early position before `d` is declared), then read by every thread
    // at its own logicalTx.
    if (KW_GROUP_ID_0 == 0) {
        const int probeEdge = 4;   // matches this test's real EDGE exactly
        int logicalTy = tx / 16;
        int logicalTx = tx % 16;

        __shared__ REAL sDs[16];
        if (logicalTy == 0) {
            sDs[logicalTx] = exp(D[logicalTx] * distance);
        }

        KW_LOCAL_FENCE;

        KW_GLOBAL_VAR REAL* pt = dMatrices
            + 2 * totalMatrix * PADDED_STATE_COUNT * PADDED_STATE_COUNT
            + KW_GROUP_ID_0 * PADDED_STATE_COUNT * PADDED_STATE_COUNT;
        if (logicalTx < probeEdge && logicalTy < probeEdge) {
            pt[logicalTy * probeEdge + logicalTx] = sDs[logicalTx];
        }
    }
    return;
#endif

#if defined(FW_TINYGPU) && defined(TINYGPU_DEBUG_DUMP_AB_PATTERN_2D_W0)
    // Opt-in, additive-only, EARLY-RETURN probe (TODO.md "PICK UP HERE" ->
    // NV Phase 113, user: "probe As[ty][k] etc."): Phase 112 inferred --
    // from the real kernel's Csub output alone, not direct evidence -- that
    // As[]/Bs[]'s own access pattern (own-slot write by EVERY thread
    // including ty>0, then cross-thread read within the same row/column
    // during the Csub reduction) is a third, mechanistically distinct
    // shared-memory pattern never directly isolated by this investigation,
    // and may be why Phase 111's flat-dispatch kernel rewrite (validated
    // only against Ds[]'s write-by-ty==0-only pattern and local memory's
    // no-shared-memory pattern) still produced wrong output. This probe
    // gets direct, per-slot evidence instead of inference -- the same way
    // Phase 93 replaced Phase 90-92's Ds[] inference with direct proof.
    //
    // Uses its own private sAs/sBs shared arrays (not the real As/Bs), so
    // this doesn't disturb the real computation later in this same launch.
    // Mirrors the real "Last block" guard's own write shape exactly
    // (kernelsAll.cu ~line 621-630: in-bounds threads write a real value,
    // out-of-bounds threads write 0, unconditionally for all 256 threads)
    // and the real Csub reduction's own read shape exactly (~line 1023-
    // 1024: As[ty][k] for k=0..EDGE-1, Bs[k][tx] for k=0..EDGE-1) -- the
    // only thing that differs is the *content* (a decodable per-writer-
    // thread constant instead of A[]/B[]'s real values) and the *target*
    // (private sAs/sBs and a scratch dMatrices region instead of the real
    // As/Bs and C[]). Flat dispatch (logicalTy/logicalTx from the single
    // flat KW_LOCAL_ID_0, Phase 109/110 convention) since that's the fix
    // already validated for Ds[]/local memory and the open question here
    // is whether it also covers this pattern. probeEdge=4 matches this
    // test's real EDGE exactly.
    //
    // Value formula 1000+a*16+b is injective over a,b in [0,16) and
    // directly decodable (a=(-1000+val)/16, b=(-1000+val)%16) -- same
    // encoding convention as Phase 106's local-memory probe, generalized
    // from EDGE=4 to the full MULTIPLY_BLOCK_SIZE=16 range since every one
    // of the 256 threads writes its own slot here (matching the real
    // guard's unconditional if/else), not just the 16 EDGE-bounded ones.
    if (KW_GROUP_ID_0 == 0) {
        const int probeEdge = 4;
        int logicalTy = tx / 16;
        int logicalTx = tx % 16;
        __shared__ REAL sAs[MULTIPLY_BLOCK_SIZE][MULTIPLY_BLOCK_SIZE];
        __shared__ REAL sBs[MULTIPLY_BLOCK_SIZE][MULTIPLY_BLOCK_SIZE];
        if (logicalTx < probeEdge && logicalTy < probeEdge) {
            REAL v = (REAL) (1000 + logicalTy * 16 + logicalTx);
            sAs[logicalTy][logicalTx] = v;
            sBs[logicalTy][logicalTx] = v;
        } else {
            sAs[logicalTy][logicalTx] = 0;
            sBs[logicalTy][logicalTx] = 0;
        }

        KW_LOCAL_FENCE;

        if (logicalTx < probeEdge && logicalTy < probeEdge) {
            KW_GLOBAL_VAR REAL* pt = dMatrices
                + 2 * totalMatrix * PADDED_STATE_COUNT * PADDED_STATE_COUNT
                + KW_GROUP_ID_0 * probeEdge * probeEdge * 2 * probeEdge;
            KW_GLOBAL_VAR REAL* slot = pt + (logicalTy * probeEdge + logicalTx) * 2 * probeEdge;
            for (int k = 0; k < probeEdge; k++) {
                slot[k]           = sAs[logicalTy][k];   // As[ty][k] -- real Csub's own row-wise read
                slot[probeEdge+k] = sBs[k][logicalTx];   // Bs[k][tx] -- real Csub's own column-wise read
            }
        }
    }
    return;
#endif

#if defined(FW_TINYGPU) && defined(TINYGPU_DEBUG_DUMP_AB_PATTERN_FLAT_W0)
    // Opt-in, additive-only, EARLY-RETURN probe (TODO.md "PICK UP HERE" ->
    // NV Phase 113, user: "also try linearizing the shared memory into
    // As[tid.x] and As[tx*16+k] etc."): same ground-truth test as
    // TINYGPU_DEBUG_DUMP_AB_PATTERN_2D_W0 immediately above, but the
    // shared array itself is a flat REAL[256] (not REAL[16][16]) and
    // every address is computed by hand instead of via the compiler's own
    // 2D-array-subscript addressing -- tests whether 2D-array codegen
    // itself (as opposed to the underlying own-slot-write/cross-thread-
    // read pattern) is where the defect lives. Own-slot write uses the
    // rawest possible index, the thread's native flat tid.x (`tx` here,
    // before any logicalTy/logicalTx decomposition) -- not even a
    // ty*16+tx multiply/add, exactly the user's own suggested `As[tid.x]`.
    //
    // Captures TWO read variants per thread in one pass: the "matching"
    // flat read (logicalTy*16+k / k*16+logicalTx -- numerically identical
    // to the 2D probe's As[ty][k]/Bs[k][tx]) and the user's suggested
    // "swapped" flat read (logicalTx*16+k / k*16+logicalTy -- swaps which
    // logical coordinate is the row/column multiplier, a genuinely
    // different physical set of slots than the real kernel ever reads).
    if (KW_GROUP_ID_0 == 0) {
        const int probeEdge = 4;
        int logicalTy = tx / 16;
        int logicalTx = tx % 16;
        __shared__ REAL sAsFlat[MULTIPLY_BLOCK_SIZE * MULTIPLY_BLOCK_SIZE];
        __shared__ REAL sBsFlat[MULTIPLY_BLOCK_SIZE * MULTIPLY_BLOCK_SIZE];
        REAL v = (logicalTx < probeEdge && logicalTy < probeEdge)
                 ? (REAL) (1000 + logicalTy * 16 + logicalTx) : (REAL) 0;
        sAsFlat[tx] = v;
        sBsFlat[tx] = v;

        KW_LOCAL_FENCE;

        if (logicalTx < probeEdge && logicalTy < probeEdge) {
            KW_GLOBAL_VAR REAL* pt = dMatrices
                + 2 * totalMatrix * PADDED_STATE_COUNT * PADDED_STATE_COUNT
                + KW_GROUP_ID_0 * probeEdge * probeEdge * 4 * probeEdge;
            KW_GLOBAL_VAR REAL* slot = pt + (logicalTy * probeEdge + logicalTx) * 4 * probeEdge;
            for (int k = 0; k < probeEdge; k++) {
                slot[k]             = sAsFlat[logicalTy * 16 + k];   // matching As[ty][k]
                slot[probeEdge+k]   = sBsFlat[k * 16 + logicalTx];   // matching Bs[k][tx]
                slot[2*probeEdge+k] = sAsFlat[logicalTx * 16 + k];   // swapped (user-suggested tx*16+k)
                slot[3*probeEdge+k] = sBsFlat[k * 16 + logicalTy];   // swapped analog for Bs
            }
        }
    }
    return;
#endif

#if defined(FW_TINYGPU) && defined(TINYGPU_DEBUG_DUMP_COMBINED_FLAT_W0)
    // Opt-in, additive-only, EARLY-RETURN probe (TODO.md "PICK UP HERE" ->
    // NV Phase 116, user: "go back to the other candidate from Phase 114
    // (a combined-mechanisms early-return probe, which has consistently
    // been the safe pattern all session)"): Phase 113 tested Ds[] and
    // As[]/Bs[] SEPARATELY, each behind its OWN private storage and its
    // OWN barrier -- both came back 100% clean (960/960) under flat
    // dispatch. But the real kernel's own "Last block" guard writes
    // Ds[]/As[]/Bs[] ALL TOGETHER, synced by a SINGLE shared barrier,
    // before the Csub reduction reads all three together -- a
    // combination no probe in this investigation has ever exercised.
    // This probe replicates that exact combined write/single-barrier/
    // read structure -- private sAs/sBs/sDs (same [16][16]/[16][16]/[16]
    // shapes and total 2112-byte smem footprint as the real As/Bs/Ds),
    // the same unconditional-for-all-256-threads write shape (real value
    // if logicalTx<EDGE && logicalTy<EDGE, else 0 -- exactly mirroring
    // the real if/else, including the else branch's Ds[tx]=0 when
    // logicalTy==0), ONE KW_LOCAL_FENCE, then the exact real Csub
    // reduction formula -- but with small, decodable synthetic values
    // (deliberately kept small enough that every intermediate product/
    // sum stays far under float32's 2^24 exact-integer bound, so any
    // deviation from the reference is a real hardware/driver
    // discrepancy, not accumulated floating-point rounding). Directly
    // answers: does the failure Phase 111's real kernel shows require
    // this exact combined structure, or does it persist/vanish here too,
    // same as every individually-tested piece?
    if (KW_GROUP_ID_0 == 0) {
        const int probeEdge = 4;   // matches this test's real EDGE exactly
        int logicalTy = tx / 16;
        int logicalTx = tx % 16;

        __shared__ REAL sAs[MULTIPLY_BLOCK_SIZE][MULTIPLY_BLOCK_SIZE];
        __shared__ REAL sBs[MULTIPLY_BLOCK_SIZE][MULTIPLY_BLOCK_SIZE];
        __shared__ REAL sDs[MULTIPLY_BLOCK_SIZE];

        if (logicalTx < probeEdge && logicalTy < probeEdge) {
            REAL v = (REAL) (10 + logicalTy * 4 + logicalTx);
            if (logicalTy == 0)
                sDs[logicalTx] = (REAL) (1 + logicalTx);
            sAs[logicalTy][logicalTx] = v;
            sBs[logicalTy][logicalTx] = v;
        } else {
            if (logicalTy == 0)
                sDs[logicalTx] = 0;
            sAs[logicalTy][logicalTx] = 0;
            sBs[logicalTy][logicalTx] = 0;
        }

        KW_LOCAL_FENCE;

        if (logicalTx < probeEdge && logicalTy < probeEdge) {
            REAL csub = 0;
            for (int k = 0; k < probeEdge; k++)
                csub += sAs[logicalTy][k] * sDs[k] * sBs[k][logicalTx];
            KW_GLOBAL_VAR REAL* pt = dMatrices
                + 2 * totalMatrix * PADDED_STATE_COUNT * PADDED_STATE_COUNT
                + KW_GROUP_ID_0 * probeEdge * probeEdge * (3 * probeEdge + 1);
            KW_GLOBAL_VAR REAL* slot = pt + (logicalTy * probeEdge + logicalTx) * (3 * probeEdge + 1);
            slot[0] = sAs[logicalTy][0]; slot[1] = sAs[logicalTy][1]; slot[2] = sAs[logicalTy][2]; slot[3] = sAs[logicalTy][3];
            slot[4] = sBs[0][logicalTx]; slot[5] = sBs[1][logicalTx]; slot[6] = sBs[2][logicalTx]; slot[7] = sBs[3][logicalTx];
            slot[8] = sDs[0]; slot[9] = sDs[1]; slot[10] = sDs[2]; slot[11] = sDs[3];
            slot[12] = csub;
        }
    }
    return;
#endif

#if defined(FW_TINYGPU) && defined(TINYGPU_DEBUG_DUMP_COMBINED_LDG_FLAT_W0)
    // Opt-in, additive-only, EARLY-RETURN probe (TODO.md "PICK UP HERE" ->
    // NV Phase 118, user: "a combined probe like this one, but sourcing
    // its written values from real LDGs of A/B/D instead of register
    // constants -- still safe, still early-return"): Phase 116/117 --
    // every synthetic reconstruction this investigation can build,
    // including the full combined write/single-barrier/reduce structure,
    // is 100% correct under flat dispatch, using pure register-computed
    // constants for every written value. This is the first probe in this
    // round to source those values from genuine global-memory loads --
    // real A[]/B[]/D[] via LDG (the same buffers, same a=b=d=0 offsets,
    // the real kernel itself uses for wMatrix 0's block) -- isolating
    // "real memory-load-sourced data" as its own variable, still WITHOUT
    // exp()/distance (Ds[tx]=D[tx] raw, matching this investigation's own
    // established TINYGPU_BISECT_NO_EXP convention -- one variable at a
    // time). Otherwise identical to TINYGPU_DEBUG_DUMP_COMBINED_FLAT_W0
    // immediately above: same private sAs/sBs/sDs, same unconditional-
    // for-all-256-threads write shape (real LDG if in-bounds, else 0),
    // ONE shared barrier, the same real Csub reduction formula, early
    // return before the real "Last block" guard.
    if (KW_GROUP_ID_0 == 0) {
        const int probeEdge = 4;   // matches this test's real EDGE exactly
        int logicalTy = tx / 16;
        int logicalTx = tx % 16;

        __shared__ REAL sAs[MULTIPLY_BLOCK_SIZE][MULTIPLY_BLOCK_SIZE];
        __shared__ REAL sBs[MULTIPLY_BLOCK_SIZE][MULTIPLY_BLOCK_SIZE];
        __shared__ REAL sDs[MULTIPLY_BLOCK_SIZE];

        if (logicalTx < probeEdge && logicalTy < probeEdge) {
            if (logicalTy == 0)
                sDs[logicalTx] = D[logicalTx];   // real LDG, no exp() yet -- a=b=d=0 for wMatrix 0's block
            sAs[logicalTy][logicalTx] = A[PADDED_STATE_COUNT * logicalTy + logicalTx];   // real LDG
            sBs[logicalTy][logicalTx] = B[PADDED_STATE_COUNT * logicalTy + logicalTx];   // real LDG
        } else {
            if (logicalTy == 0)
                sDs[logicalTx] = 0;
            sAs[logicalTy][logicalTx] = 0;
            sBs[logicalTy][logicalTx] = 0;
        }

        KW_LOCAL_FENCE;

        if (logicalTx < probeEdge && logicalTy < probeEdge) {
            REAL csub = 0;
            for (int k = 0; k < probeEdge; k++)
                csub += sAs[logicalTy][k] * sDs[k] * sBs[k][logicalTx];
            KW_GLOBAL_VAR REAL* pt = dMatrices
                + 2 * totalMatrix * PADDED_STATE_COUNT * PADDED_STATE_COUNT
                + KW_GROUP_ID_0 * probeEdge * probeEdge * (3 * probeEdge + 1);
            KW_GLOBAL_VAR REAL* slot = pt + (logicalTy * probeEdge + logicalTx) * (3 * probeEdge + 1);
            slot[0] = sAs[logicalTy][0]; slot[1] = sAs[logicalTy][1]; slot[2] = sAs[logicalTy][2]; slot[3] = sAs[logicalTy][3];
            slot[4] = sBs[0][logicalTx]; slot[5] = sBs[1][logicalTx]; slot[6] = sBs[2][logicalTx]; slot[7] = sBs[3][logicalTx];
            slot[8] = sDs[0]; slot[9] = sDs[1]; slot[10] = sDs[2]; slot[11] = sDs[3];
            slot[12] = csub;
        }
    }
    return;
#endif

#if defined(FW_TINYGPU) && defined(TINYGPU_DEBUG_DUMP_COMBINED_EXP_FLAT_W0)
    // Opt-in, additive-only, EARLY-RETURN probe (TODO.md "PICK UP HERE" ->
    // NV Phase 118, user: "reintroducing the real exp()/distance
    // computation specifically into a synthetic probe"): identical to
    // TINYGPU_DEBUG_DUMP_COMBINED_LDG_FLAT_W0 immediately above, except
    // Ds[] now uses the real, full formula -- exp(D[tx]*distance), the
    // same real `distance` (already computed earlier in this function,
    // a real distanceQueue[wMatrix] load, per-thread private register --
    // Phase 110's own TINYGPU_DEBUG_DUMP_SHARED_BROADCAST_FLAT_W0 already
    // established this exact usage is valid at this position) -- rather
    // than the raw D[tx] value. With real A/B/D *and* real exp()/
    // distance, this probe's own Csub-analog reduction is mathematically
    // identical to the real kernel's own intended Csub -- so a correct
    // result here should exactly equal reference_transition_matrix()'s
    // real answer, the strongest synthetic test this investigation can
    // build without touching the real, non-early-return kernel path.
    if (KW_GROUP_ID_0 == 0) {
        const int probeEdge = 4;   // matches this test's real EDGE exactly
        int logicalTy = tx / 16;
        int logicalTx = tx % 16;

        __shared__ REAL sAs[MULTIPLY_BLOCK_SIZE][MULTIPLY_BLOCK_SIZE];
        __shared__ REAL sBs[MULTIPLY_BLOCK_SIZE][MULTIPLY_BLOCK_SIZE];
        __shared__ REAL sDs[MULTIPLY_BLOCK_SIZE];

        if (logicalTx < probeEdge && logicalTy < probeEdge) {
            if (logicalTy == 0)
                sDs[logicalTx] = exp(D[logicalTx] * distance);   // real formula
            sAs[logicalTy][logicalTx] = A[PADDED_STATE_COUNT * logicalTy + logicalTx];   // real LDG
            sBs[logicalTy][logicalTx] = B[PADDED_STATE_COUNT * logicalTy + logicalTx];   // real LDG
        } else {
            if (logicalTy == 0)
                sDs[logicalTx] = 0;
            sAs[logicalTy][logicalTx] = 0;
            sBs[logicalTy][logicalTx] = 0;
        }

        KW_LOCAL_FENCE;

        if (logicalTx < probeEdge && logicalTy < probeEdge) {
            REAL csub = 0;
            for (int k = 0; k < probeEdge; k++)
                csub += sAs[logicalTy][k] * sDs[k] * sBs[k][logicalTx];
            KW_GLOBAL_VAR REAL* pt = dMatrices
                + 2 * totalMatrix * PADDED_STATE_COUNT * PADDED_STATE_COUNT
                + KW_GROUP_ID_0 * probeEdge * probeEdge * (3 * probeEdge + 1);
            KW_GLOBAL_VAR REAL* slot = pt + (logicalTy * probeEdge + logicalTx) * (3 * probeEdge + 1);
            slot[0] = sAs[logicalTy][0]; slot[1] = sAs[logicalTy][1]; slot[2] = sAs[logicalTy][2]; slot[3] = sAs[logicalTy][3];
            slot[4] = sBs[0][logicalTx]; slot[5] = sBs[1][logicalTx]; slot[6] = sBs[2][logicalTx]; slot[7] = sBs[3][logicalTx];
            slot[8] = sDs[0]; slot[9] = sDs[1]; slot[10] = sDs[2]; slot[11] = sDs[3];
            slot[12] = csub;
        }
    }
    return;
#endif

#if defined(FW_TINYGPU) && defined(TINYGPU_DEBUG_DUMP_COMBINED_WRITE_DIRECT_W0)
    // Opt-in, additive-only, EARLY-RETURN probe (TODO.md "PICK UP HERE" ->
    // NV Phase 120, user: "let's try the final write now. but write to
    // dMatrices[figure out indices] first before trying dMatrix +
    // listC[wMatrix]"): Phase 118/119 -- every constructible synthetic
    // variable (real A/B/D loads, real exp()/distance, the full combined
    // structure) is now proven correct under flat dispatch, but every
    // probe since Phase 109 stops at an early return, before ever
    // touching the real final write. This is the first probe in this
    // round to perform that write for real -- identical to
    // TINYGPU_DEBUG_DUMP_COMBINED_EXP_FLAT_W0 immediately above (real
    // A/B/D LDGs, real exp()/distance, the real Csub reduction formula),
    // except the result is now written to the REAL matrix-output region
    // of dMatrices (not a separate scratch region, unlike every prior
    // probe) at a DIRECT, closed-form index -- dMatrices + wMatrix*
    // PADDED_STATE_COUNT^2 + ty*PADDED_STATE_COUNT+tx -- matching this
    // investigation's own established TINYGPU_BISECT_NO_LISTC closed
    // form (Phase 65) exactly, rather than the real kernel's `C =
    // dMatrices + listC[wMatrix]` indirection. wMatrix==0 for this
    // early-returning block, so no other block/probe ever writes here
    // in the same launch -- safe. Isolates "the real final write, at
    // this program-order position, right after this Csub reduction,
    // into the real output region" from "the real listC[]-based data-
    // dependent addressing" as two separate variables -- the direct
    // form goes first, per the user's own request. No new diagnostic
    // region needed: this writes directly into wMatrix 0's real matrix
    // slot, so --sweep's own existing correctness check (real_matrices
    // vs. reference_matrices) already answers whether this write is
    // correct.
    if (KW_GROUP_ID_0 == 0) {
        const int probeEdge = 4;
        int logicalTy = tx / 16;
        int logicalTx = tx % 16;

        __shared__ REAL sAs[MULTIPLY_BLOCK_SIZE][MULTIPLY_BLOCK_SIZE];
        __shared__ REAL sBs[MULTIPLY_BLOCK_SIZE][MULTIPLY_BLOCK_SIZE];
        __shared__ REAL sDs[MULTIPLY_BLOCK_SIZE];

        if (logicalTx < probeEdge && logicalTy < probeEdge) {
            if (logicalTy == 0)
                sDs[logicalTx] = exp(D[logicalTx] * distance);
            sAs[logicalTy][logicalTx] = A[PADDED_STATE_COUNT * logicalTy + logicalTx];
            sBs[logicalTy][logicalTx] = B[PADDED_STATE_COUNT * logicalTy + logicalTx];
        } else {
            if (logicalTy == 0)
                sDs[logicalTx] = 0;
            sAs[logicalTy][logicalTx] = 0;
            sBs[logicalTy][logicalTx] = 0;
        }

        KW_LOCAL_FENCE;

        if (logicalTx < probeEdge && logicalTy < probeEdge) {
            REAL csub = 0;
            for (int k = 0; k < probeEdge; k++)
                csub += sAs[logicalTy][k] * sDs[k] * sBs[k][logicalTx];
            KW_GLOBAL_VAR REAL* out = dMatrices + wMatrix * PADDED_STATE_COUNT * PADDED_STATE_COUNT;
            if (csub < 0)
                out[logicalTy * PADDED_STATE_COUNT + logicalTx] = 0;
            else
                out[logicalTy * PADDED_STATE_COUNT + logicalTx] = csub;
        }
    }
    return;
#endif

#if defined(FW_TINYGPU) && defined(TINYGPU_DEBUG_DUMP_COMBINED_WRITE_LISTC_W0)
    // Opt-in, additive-only, EARLY-RETURN probe (TODO.md "PICK UP HERE" ->
    // NV Phase 120, user's own "before trying dMatrix + listC[wMatrix]"
    // follow-up): identical to TINYGPU_DEBUG_DUMP_COMBINED_WRITE_DIRECT_
    // W0 immediately above, except the write target now uses the REAL
    // kernel's own indirection -- `dMatrices + listC[wMatrix]` -- a
    // genuine, data-dependent global-memory load feeding a pointer
    // computation, rather than the closed-form index. For this test's
    // real listC values (listC[wMatrix] = wMatrix*PADDED_STATE_COUNT^2,
    // confirmed directly from nv_real_kernel_probe.py's own listc_vals),
    // this resolves to the exact same address as the direct probe above
    // -- so any difference between the two is attributable specifically
    // to the extra indirect load, not a different memory target.
    if (KW_GROUP_ID_0 == 0) {
        const int probeEdge = 4;
        int logicalTy = tx / 16;
        int logicalTx = tx % 16;

        __shared__ REAL sAs[MULTIPLY_BLOCK_SIZE][MULTIPLY_BLOCK_SIZE];
        __shared__ REAL sBs[MULTIPLY_BLOCK_SIZE][MULTIPLY_BLOCK_SIZE];
        __shared__ REAL sDs[MULTIPLY_BLOCK_SIZE];

        if (logicalTx < probeEdge && logicalTy < probeEdge) {
            if (logicalTy == 0)
                sDs[logicalTx] = exp(D[logicalTx] * distance);
            sAs[logicalTy][logicalTx] = A[PADDED_STATE_COUNT * logicalTy + logicalTx];
            sBs[logicalTy][logicalTx] = B[PADDED_STATE_COUNT * logicalTy + logicalTx];
        } else {
            if (logicalTy == 0)
                sDs[logicalTx] = 0;
            sAs[logicalTy][logicalTx] = 0;
            sBs[logicalTy][logicalTx] = 0;
        }

        KW_LOCAL_FENCE;

        if (logicalTx < probeEdge && logicalTy < probeEdge) {
            REAL csub = 0;
            for (int k = 0; k < probeEdge; k++)
                csub += sAs[logicalTy][k] * sDs[k] * sBs[k][logicalTx];
            KW_GLOBAL_VAR REAL* out = dMatrices + listC[wMatrix];
            if (csub < 0)
                out[logicalTy * PADDED_STATE_COUNT + logicalTx] = 0;
            else
                out[logicalTy * PADDED_STATE_COUNT + logicalTx] = csub;
        }
    }
    return;
#endif

#if defined(FW_TINYGPU) && defined(TINYGPU_DEBUG_BROADCAST_PROBE)
    // Minimal, purpose-built probe (STATUS.md §67/68, TODO.md Phase 32/33) --
    // isolates the "N of 256 threads write shared memory under a guard;
    // KW_LOCAL_FENCE; every thread reads it back" idiom in total isolation
    // from the rest of this kernel (no BLOCKS loop, no Csub accumulation).
    // v1 (§67/68, a single flat `if(ty==0)` guard, no branch, no exp())
    // came back clean on hardware -- all 16 differences exactly 0,
    // refuting "the barrier/shared-memory mechanism itself is broken" as
    // a general driver-level defect.
    //
    // v2 (this version) reintroduces *only* the real kernel's actual
    // nested branch shape around the write -- `tx<EDGE && ty<EDGE` outer,
    // `ty==0` inner, both the real-value `if` and the zero `else`,
    // byte-for-byte the same shape `Ds[]` itself uses -- while keeping
    // the written value trivial (plain `D[tx]`, no `exp()`/`distance`
    // yet). Isolates whether the *branch shape itself* (a real divergent
    // branch every one of the 256 threads takes one side of, unlike v1's
    // flat guard) breaks visibility, independent of the transcendental
    // math. If this also comes back clean, `exp()` becomes the next,
    // better-motivated suspect; change one variable at a time, per this
    // investigation's own established discipline (STATUS.md §10's
    // postmortem).
    //
    // Only wMatrix==0's block runs the probe; every other wMatrix's block
    // returns untouched below (needed so this doesn't corrupt other
    // matrices' legitimate output -- C[] here only has room for this one
    // matrix's 16 real elements). Ground-truth-free by design: every
    // thread with (tx,ty) in 0..3 (16 of the block's 256 threads,
    // deliberately spanning the warp0/warp1 boundary at ty=2 already
    // implicated in Phase 10's original finding) computes both (a) what
    // it reads back from shared memory, written only by the ty==0 "row"'s
    // if-branch (tx<EDGE, matching this exact readback range), and (b) the
    // same value read directly from global memory in its own thread --
    // then writes their difference. A working broadcast means all 16
    // differences are exactly 0; any nonzero pinpoints exactly which
    // (tx,ty) thread's post-barrier read failed.
    if (wMatrix != 0) return;

    // v3 (STATUS.md §70/TODO.md Phase 35): v1 (flat guard) and v2 (exact
    // nested branch, SASS-confirmed real BSSY/BSYNC) both came back clean
    // -- neither the barrier mechanism nor this branch shape is the
    // culprit. The one remaining untested piece of Ds[]'s real write is
    // exp()/distance itself. Same branch structure as v2, unchanged; only
    // the written value's RHS changes to the real computation, and the
    // comparison recomputes exp() directly rather than re-fetching the
    // raw input (the shared value is no longer expected to equal D[tx]).
    KW_LOCAL_MEM REAL sVal[MULTIPLY_BLOCK_SIZE];
    const int probeEdge = 4;   // matches this test's real EDGE exactly
    if (tx < probeEdge && ty < probeEdge) {
        if (ty == 0)
            sVal[tx] = exp(D[tx] * distance);
    } else {
        if (ty == 0)
            sVal[tx] = 0;
    }

    KW_LOCAL_FENCE;

    if (tx < 4 && ty < 4)
        C[ty * 4 + tx] = sVal[tx] - exp(D[tx] * distance);
    return;
#endif

    const int EDGE = PADDED_STATE_COUNT - (BLOCKS - 1) * MULTIPLY_BLOCK_SIZE;

    // Step size used to iterate through the sub-matrices of A
    int aStep = MULTIPLY_BLOCK_SIZE;

    // Step size used to iterate through the sub-matrices of B
    int bStep = MULTIPLY_BLOCK_SIZE * PADDED_STATE_COUNT;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    REAL Csub = 0;

    int a = PADDED_STATE_COUNT * MULTIPLY_BLOCK_SIZE * by;
    int b = MULTIPLY_BLOCK_SIZE * bx;
    int d = 0; //MULTIPLY_BLOCK_SIZE * bx;

    KW_LOCAL_MEM REAL As[MULTIPLY_BLOCK_SIZE][MULTIPLY_BLOCK_SIZE];
    KW_LOCAL_MEM REAL Bs[MULTIPLY_BLOCK_SIZE][MULTIPLY_BLOCK_SIZE];
    KW_LOCAL_MEM REAL Ds[MULTIPLY_BLOCK_SIZE];

#if !(defined(FW_TINYGPU) && defined(TINYGPU_BISECT_NO_MAIN_LOOP))
    // Opt-in, whole-loop-omission bisection experiment (TODO.md "PICK UP
    // HERE" -> NV Phase 62): every synthetic from-scratch reproduction
    // attempted so far (Phase 47-60) has failed to reproduce the real
    // kernel's ground-truth-dump failure, even ones matching its resource
    // footprint, branch shape, and broadcast+barrier+burst structure
    // exactly. One real difference no synthetic probe has replicated:
    // this loop's mere *presence in source*, even though it runs zero
    // iterations for this test's BLOCKS==1 case (a runtime value, not
    // known to the compiler, so real code is still generated for it).
    // Omitting it entirely is behaviorally a no-op for BLOCKS==1 --
    // true throughout this whole investigation -- but would change
    // results for a real BLOCKS>1 launch, so this must stay strictly
    // opt-in, never default.
    for (int i = 0; i < BLOCKS - 1; i++) {

        if (ty == 0)
            Ds[tx] = exp(D[d + tx] * distance);

        As[ty][tx] = A[a + PADDED_STATE_COUNT * ty + tx];
        Bs[ty][tx] = B[b + PADDED_STATE_COUNT * ty + tx];

        KW_LOCAL_FENCE;

        for (int k = 0; k < MULTIPLY_BLOCK_SIZE; ++k)
            Csub += As[ty][k] * Ds[k] * Bs[k][tx];

        KW_LOCAL_FENCE;

        a += aStep;
        b += bStep;
        d += MULTIPLY_BLOCK_SIZE;
    }
#endif

    // Last block is too long
    //
    // NOTE (usb branch): two branchless FW_TINYGPU-only rewrites of this
    // if/else were tried here -- a full one (STATUS.md §17/18, TODO.md
    // Phase 8) and a narrower one touching only As/Bs (TODO.md Phase 10).
    // Both hung on real hardware (no GSP exception logged either time --
    // checked `sudo dmesg | grep 'nv usb4'` after Phase 10's hang
    // specifically, nothing there). Both reverted. See TODO.md "PICK UP
    // HERE" for the current understanding and what's suspected instead
    // (concurrency from turning a guarded, ~16-of-256-threads-per-block
    // LDG into an unconditional, every-thread LDG -- not the branch
    // construct itself). Left as the original if/else, unmodified, for
    // all backends including FW_TINYGPU. Do not retry a branchless
    // rewrite of this guard without a static (SASS-level) way to confirm
    // the load stays genuinely predicated (few threads touching memory),
    // not just branch-free.
    if (tx < EDGE && ty < EDGE) {
        if (ty == 0)
#if defined(FW_TINYGPU) && defined(TINYGPU_BISECT_NO_EXP)
            // Opt-in, single-substitution bisection experiment (TODO.md
            // "PICK UP HERE" -> NV Phase 57): candidate (2) from Phase 56's
            // "both spread" result -- does replacing just the real exp()
            // transcendental with a cheap placeholder (still a real global
            // memory read, still not constant-foldable) change whether
            // kernelMatrixMulADB's blocks funnel to SM 0? Deliberately
            // touches *only* this one line -- the main (BLOCKS-1) loop's
            // identical exp() call above is untouched (dead code for this
            // test's BLOCKS==1 case anyway) and every other real-value
            // computation in this guard (As/Bs's global reads) is
            // untouched -- one variable at a time, this investigation's
            // own established discipline. Reversible: real exp() body
            // kept in the #else branch, unconditionally used by every
            // other backend and by FW_TINYGPU itself whenever this macro
            // isn't defined.
            Ds[tx] = D[d + tx];
#elif defined(FW_TINYGPU) && defined(TINYGPU_BISECT_EXP_APPROX)
            // Opt-in, single-substitution bisection experiment (TODO.md
            // "PICK UP HERE" -> NV Phase 73, user-directed): unlike
            // TINYGPU_BISECT_NO_EXP's raw (possibly-negative) eigenvalue,
            // this stays a valid, always-positive "probability-like" value
            // -- a quadratic Taylor approximation of exp(x) around x=0,
            // clamped to a small positive floor -- while still not invoking
            // the real exp() transcendental/SFU instruction. Tests whether
            // it's specifically the SFU exp() call (as opposed to Ds[]'s
            // sign/validity) that matters for which wMatrix blocks execute.
            {
                REAL approxExpArg = D[d + tx] * distance;
                REAL approxExp = (REAL) 1.0 + approxExpArg * ((REAL) 1.0 + (REAL) 0.5 * approxExpArg);
                Ds[tx] = (approxExp > (REAL) 1e-5) ? approxExp : (REAL) 1e-5;
            }
#else
            Ds[tx] = exp(D[d + tx] * distance);
#endif

#if defined(FW_TINYGPU) && defined(TINYGPU_BISECT_SINGLE_AB_PTR)
        // Opt-in, single-substitution bisection experiment (TODO.md
        // "PICK UP HERE" -> NV Phase 63): the real kernel reads two
        // *separate* global base pointers (A, B) right here -- a real
        // difference no synthetic probe in this investigation has ever
        // replicated (every probe since Phase 47 used one shared seed[]
        // array). Redirects Bs's read onto A (same base pointer As
        // already uses, same offset b already computed) instead of B --
        // tests whether switching between multiple distinct global base
        // pointers within one kernel matters, isolated from every other
        // real difference (the real branch, barrier, broadcast, and A's
        // own read are all untouched). In-bounds: A and B are both real,
        // equally-sized PADDED_STATE_COUNT x PADDED_STATE_COUNT matrices,
        // so reading A at an offset B would have used is a genuinely
        // wrong *value*, not an out-of-bounds access.
        As[ty][tx] = A[a + PADDED_STATE_COUNT * ty + tx];
        Bs[ty][tx] = A[b + PADDED_STATE_COUNT * ty + tx];
#else
        As[ty][tx] = A[a + PADDED_STATE_COUNT * ty + tx];
        Bs[ty][tx] = B[b + PADDED_STATE_COUNT * ty + tx];
#endif

    } else {

        if (ty == 0)
            Ds[tx] = 0;

        As[ty][tx] = 0;
        Bs[ty][tx] = 0;
    }

    KW_LOCAL_FENCE;

#if defined(FW_TINYGPU) && defined(TINYGPU_DEBUG_DUMP_CSUB_INPUTS)
    // Opt-in, additive-only probe (STATUS.md §55/TODO.md Phase 20): dumps
    // exactly what thread (tx=0,ty=0)'s Csub sum reads out of shared memory
    // right after the barrier that publishes the "Last block" guard's
    // As/Bs/Ds writes -- targets the plain-ptxas, non-nvJitLink "wrong
    // Csub" fault (element [0] of every matrix comes back exactly 0,
    // STATUS.md §19/Phase 10), never before directly observed in-kernel
    // (every prior in-kernel probe of this kernel, STATUS.md §44-51, ran
    // under BEAGLE_NV_USE_NVJITLINK=1 and diagnosed a separate,
    // nvJitLink-specific defect instead). Run with BEAGLE_NV_USE_NVJITLINK
    // unset to test the actual still-broken config. All threads in the
    // wMatrix==0 block return here (not just tx==0,ty==0) to prevent other
    // threads' normal Csub writes from overwriting the C[0..11] cells this
    // probe uses -- same discipline as the prior TINYGPU_DEBUG_DUMP_EXP
    // probe this replaces.
    if (wMatrix == 0) {
        if (tx == 0 && ty == 0) {
            C[0]  = As[0][0]; C[1]  = As[0][1]; C[2]  = As[0][2]; C[3]  = As[0][3];
            C[4]  = Bs[0][0]; C[5]  = Bs[1][0]; C[6]  = Bs[2][0]; C[7]  = Bs[3][0];
            C[8]  = Ds[0];    C[9]  = Ds[1];    C[10] = Ds[2];    C[11] = Ds[3];
        }
        return;
    }
#endif

#if defined(FW_TINYGPU) && defined(TINYGPU_DEBUG_DUMP_MATMUL_GROUND_TRUTH)
    // Opt-in, additive-only probe (TODO.md "PICK UP HERE" -> NV Phase 47):
    // TINYGPU_DEBUG_DUMP_CSUB_INPUTS above only instruments wMatrix==0's
    // block (already known-working, per Phase 42) and writes through that
    // block's own real C[] pointer -- useless for the actually-failing
    // wMatrix>=4 blocks, where a block that never reaches this point is
    // indistinguishable from one that reached it and wrote a genuinely
    // wrong value (both just leave C[] as whatever it already was). This
    // probe instruments *every* block and writes to a dedicated scratch
    // region past the real matrices -- dMatrices + totalMatrix*(PADDED_
    // STATE_COUNT^2) + KW_GROUP_ID_0*(PADDED_STATE_COUNT^2) -- addressed
    // directly by KW_GROUP_ID_0, not through listC/wMatrix. A block that
    // never gets here leaves its slot at whatever sentinel the host
    // pre-seeded via beagleSetTransitionMatrix on the corresponding "debug"
    // matrix indices (see tinygpuhybridtest.cpp's
    // --diag-matmul-ground-truth), so "never ran" and "ran but wrong" are
    // now distinguishable. Doesn't return/skip -- the real computation
    // below still runs unmodified, so the same launch's real output (and
    // logL) stays meaningful too. Sized for this investigation's actual
    // repro (BLOCKS==1, totalMatrix==16); not a general-N tool.
    {
        KW_GLOBAL_VAR REAL* dbg = dMatrices
            + totalMatrix * PADDED_STATE_COUNT * PADDED_STATE_COUNT
            + KW_GROUP_ID_0 * PADDED_STATE_COUNT * PADDED_STATE_COUNT;
        if (tx == 0 && ty == 0) {
            REAL csub0 = 0;
            for (int k = 0; k < EDGE; k++)
                csub0 += As[0][k] * Ds[k] * Bs[k][0];
            dbg[0]  = csub0;
            dbg[1]  = As[0][0]; dbg[2]  = As[0][1]; dbg[3]  = As[0][2]; dbg[4]  = As[0][3];
            dbg[5]  = Bs[0][0]; dbg[6]  = Bs[1][0]; dbg[7]  = Bs[2][0]; dbg[8]  = Bs[3][0];
#if defined(FW_TINYGPU) && defined(TINYGPU_BISECT_SWAP_DS23)
            // Opt-in, single-swap bisection experiment (TODO.md "PICK UP
            // HERE" -> NV Phase 58): chases Phase 57's residual finding --
            // with TINYGPU_BISECT_NO_EXP applied, every wMatrix>=3 block
            // runs fully except dbg[11] (Ds[2], byte offset 0x2c within
            // the debug slot), which stays at the sentinel. Swaps *only*
            // which value targets dbg[11] vs dbg[12] (Ds[2]<->Ds[3]) --
            // everything else (which slot Csub/As/Bs/Ds[0]/Ds[1] target,
            // the exp() bisection, the real computation) is untouched.
            // If dbg[11] (address 0x2c) still fails afterward -- now
            // "missing" Ds[3] instead of Ds[2] -- that's a direct,
            // hardware-observed confirmation the failure tracks the
            // store's target slot/address, not the specific C-source
            // variable/value being written there.
            dbg[9]  = Ds[0];    dbg[10] = Ds[1];    dbg[11] = Ds[3];    dbg[12] = Ds[2];
#else
            dbg[9]  = Ds[0];    dbg[10] = Ds[1];    dbg[11] = Ds[2];    dbg[12] = Ds[3];
#endif
#if defined(CUDA)
            // TODO.md "PICK UP HERE" -> NV Phase 50, user-directed
            // driver/queue-layer instrumentation (safe half -- reads a
            // standard, documented PTX special register, not undocumented
            // hardware state like desc[UR]): which physical SM this block
            // actually ran on. All blocks landing on the same handful of
            // SMs (or fewer distinct SMs than expected) would point at a
            // genuine occupancy/CTA-retirement limitation; blocks spread
            // across many different SMs but still stopping at the same
            // wMatrix boundary would argue against that and toward
            // something else. %smid is legal PTX on every real CUDA
            // target (not FW_TINYGPU-specific machinery), so this is
            // gated on plain CUDA rather than FW_TINYGPU like the rest of
            // this probe -- harmless either way since it's still inside
            // the FW_TINYGPU-only TINYGPU_DEBUG_DUMP_MATMUL_GROUND_TRUTH
            // block as a whole.
            unsigned int smid;
            asm("mov.u32 %0, %smid;" : "=r"(smid));
            dbg[13] = (REAL) smid;
#endif
        }
    }
#endif

#if defined(FW_TINYGPU) && defined(TINYGPU_DEBUG_DUMP_DUMMY_THIRD_BLOCK)
    // Opt-in, additive-only probe (TODO.md "PICK UP HERE" -> NV Phase 100):
    // Phase 94/95/98 found that adding TINYGPU_DEBUG_DUMP_PER_THREAD_DS(_W0)
    // -- a THIRD post-barrier global-memory write block, alongside the
    // real matrix write and TINYGPU_DEBUG_DUMP_MATMUL_GROUND_TRUTH above
    // -- faults the real GPU 100% reproducibly, at both the full size
    // (1024 extra floats) and a much smaller w0-only size (64 floats,
    // only 256 bytes over the always-safe baseline) -- ruling out buffer
    // size (Phase 98). Two hypotheses remained: (1) the fault requires
    // *this specific combination* of TINYGPU_DEBUG_DUMP_MATMUL_GROUND_
    // TRUTH plus TINYGPU_DEBUG_DUMP_PER_THREAD_DS(_W0) together; (2) it's
    // fundamentally about adding *any* third post-barrier write block to
    // this kernel, independent of size, content, or which specific
    // diagnostics are combined. This probe tests (2) directly: a single,
    // trivial, content-unrelated third write -- one thread per block
    // writes a hardcoded constant (not read from As/Bs/Ds/Csub at all) to
    // a dedicated third scratch region -- alongside the existing ground-
    // truth dump above (TINYGPU_DEBUG_DUMP_MATMUL_GROUND_TRUTH stays
    // active), at a footprint (4 bytes/block, 64 bytes total for
    // BLOCKS==1/totalMatrix==16) far smaller than even Phase 96's
    // already-small w0-only version. If this ALSO faults, that's strong,
    // clean evidence for hypothesis (2): merely having a third write
    // block is what breaks this kernel, regardless of what it writes or
    // how large it is. If it does NOT fault, that argues for hypothesis
    // (1) instead -- something about combining these two *specific*
    // diagnostics. Reuses the same base offset (2*totalMatrix*P^2) the
    // per-thread-Ds probes use for their own third region -- not compiled
    // together with those in this experiment, so no collision.
    {
        KW_GLOBAL_VAR REAL* dummy = dMatrices
            + 2 * totalMatrix * PADDED_STATE_COUNT * PADDED_STATE_COUNT
            + KW_GROUP_ID_0;
        if (tx == 0 && ty == 0) {
            dummy[0] = 42;
        }
    }
#endif

#if defined(FW_TINYGPU) && defined(TINYGPU_DEBUG_DUMP_PER_THREAD_DUMMY_W0)
    // Opt-in, additive-only probe (TODO.md "PICK UP HERE" -> NV Phase 101):
    // Phase 100's TINYGPU_DEBUG_DUMP_DUMMY_THIRD_BLOCK ruled out
    // hypothesis (2) in its broadest form -- a trivial, 1-thread/block
    // third write (a hardcoded constant, no shared-memory read at all)
    // did not fault, even with TINYGPU_DEBUG_DUMP_MATMUL_GROUND_TRUTH
    // active. That probe differed from TINYGPU_DEBUG_DUMP_PER_THREAD_
    // DS(_W0) in TWO ways simultaneously: (a) content -- a hardcoded
    // constant vs. a real Ds[] shared-memory read; (b) granularity -- 1
    // thread/block vs. all 16 real threads (tx,ty in [0,EDGE)) each
    // independently computing their own address and writing their own
    // slot. This probe isolates (b) alone: every one of the 16 real
    // threads in wMatrix 0's block writes a trivial, per-thread-
    // identifiable constant (its own (ty*EDGE+tx) index -- NOT a value
    // read from Ds[] or any other shared-memory location) to its own
    // slot in a dedicated third region, using the *exact* same per-
    // thread address pattern TINYGPU_DEBUG_DUMP_PER_THREAD_DS uses.
    // Sized (16 floats, one per thread) to exactly match Phase 100's own
    // dummy-third-block total footprint (16 blocks x 1 float each) --
    // same total buffer size, same G-active setup; the ONLY variable
    // that differs from that already-safe run is whether one thread or
    // sixteen threads do the writing. If this ALSO faults, that isolates
    // the many-threads-each-writing-their-own-slot granularity itself as
    // the operative variable, independent of shared-memory content. If
    // it does NOT fault, that narrows the remaining explanation down to
    // reading Ds[]/shared-memory content specifically, not just the
    // write pattern. wMatrix 0 only (KW_GROUP_ID_0==0), matching Phase
    // 96's own w0-only convention -- keeps the KW_GROUP_ID_0-based term
    // in the address computation (always 0 on the one block that
    // actually writes) for consistency with a possible future full-grid
    // version, exactly as Phase 96 did for TINYGPU_DEBUG_DUMP_PER_THREAD_
    // DS_W0.
    if (KW_GROUP_ID_0 == 0) {
        KW_GLOBAL_VAR REAL* pt = dMatrices
            + 2 * totalMatrix * PADDED_STATE_COUNT * PADDED_STATE_COUNT
            + KW_GROUP_ID_0 * PADDED_STATE_COUNT * PADDED_STATE_COUNT;
        if (tx < EDGE && ty < EDGE) {
            pt[ty * EDGE + tx] = (REAL) (ty * EDGE + tx);
        }
    }
#endif

#if defined(FW_TINYGPU) && defined(TINYGPU_DEBUG_DUMP_PER_THREAD_DS_MIN_W0)
    // Opt-in, additive-only probe (TODO.md "PICK UP HERE" -> NV Phase 102):
    // Phase 100 (content swap: hardcoded constant, 1 thread/block) and
    // Phase 101 (granularity swap: 16 threads/block, still a trivial
    // local value) both ran clean -- neither, by itself, faults this
    // kernel. The only variable left distinguishing those two safe
    // probes from the two real faults (Phase 94/95/98) is that the real
    // TINYGPU_DEBUG_DUMP_PER_THREAD_DS(_W0) reads actual, barrier-
    // published Ds[] values out of shared memory as its write content --
    // neither safe probe touches shared memory at all. This probe
    // isolates that last variable directly, at Phase 101's *exact* same
    // footprint (16 floats total, one per thread) and granularity (all
    // 16 real threads in wMatrix 0's block, KW_GROUP_ID_0==0): every
    // thread writes Ds[tx] -- a single real shared-memory read, not the
    // full Ds[0..3] the real diagnostic reads per thread -- to its own
    // slot. If this faults, that isolates reading Ds[]/shared memory (in
    // this third, post-barrier context) as the operative variable. If it
    // does NOT fault, the remaining candidates narrow further still --
    // e.g. reading *all four* Ds[0..3] per thread rather than just one,
    // or the full region's larger size after all despite Phase 98's
    // buffer-size result at a different content/granularity combination.
    if (KW_GROUP_ID_0 == 0) {
        KW_GLOBAL_VAR REAL* pt = dMatrices
            + 2 * totalMatrix * PADDED_STATE_COUNT * PADDED_STATE_COUNT
            + KW_GROUP_ID_0 * PADDED_STATE_COUNT * PADDED_STATE_COUNT;
        if (tx < EDGE && ty < EDGE) {
            pt[ty * EDGE + tx] = Ds[tx];
        }
    }
#endif

#if defined(FW_TINYGPU) && defined(TINYGPU_DEBUG_DUMP_LOCAL_MEM_W0)
    // Opt-in, additive-only probe (TODO.md "PICK UP HERE" -> NV Phase 106):
    // Phase 104/105 traced the Ds[]-broadcast fault investigation as far as
    // Python-level code can go -- BeagleNVProgram's QMD construction (the
    // fields governing shared-memory partitioning across concurrently-
    // resident CTAs, which Phase 64 already showed this bug requires) is
    // byte-identical to tinygrad's own proven-correct upstream. Everything
    // tested so far (Phase 90-103) has been about SHARED memory (Ds[], an
    // explicitly __shared__ array) -- register-spilled LOCAL memory has
    // never been tested for correctness at all, despite kernelMatrixMulADB
    // genuinely spilling registers there (lcmem_usage=576 bytes/thread,
    // established since Phase 27-ish). Unlike Ds[], local memory is
    // architecturally per-thread-private -- no other thread is ever
    // supposed to see it -- so this tests a different, complementary
    // question: does each thread's OWN spilled-and-reloaded value survive
    // a roundtrip correctly, or does something (e.g. a local-memory-window/
    // backing-store addressing bug in this from-scratch driver's
    // _ensure_has_local_memory setup) corrupt or cross-alias it?
    //
    // `volatile spillTest[4]` alone is not sufficient to force a real
    // spill -- verified directly this session, twice. First (standalone
    // nvcc13/ptxas13 test): compile-time-constant array indices got fully
    // register-promoted via SROA despite `volatile`, zero LDL/STL. Second
    // (in-kernel, a short 4-iteration write loop with a simple `(tx+i)%4`
    // rotation followed by a single fixed-index read of `spillTest[0]`):
    // still zero LDL/STL -- ptxas's optimizer proved the loop's *net*
    // effect on that one element algebraically (every write to a given
    // slot stores the same closed-form value regardless of iteration
    // order) and collapsed the whole array away. What actually worked in
    // the standalone test: a *longer* recurrence whose index depends on a
    // genuine runtime kernel parameter (there, an extra function arg;
    // here, `totalMatrix`, already a real kernel parameter, opaque to the
    // static compiler) combined with the loop counter, AND reading back
    // through that *same* evolved index rather than a fixed one -- this
    // defeats the closed-form-collapse trick above, since the read index
    // itself is no longer a compile-time constant. Reused here.
    // Only ONE element is dumped per thread, not all four -- deliberately
    // matching the *already-hardware-verified-safe* 16-floats-total
    // footprint (Phase 100-103) rather than the 64-float footprint that's
    // only ever been tested with Ds[]/shared-memory content and faulted
    // both times (Phase 94/95/98) -- size held constant at the known-safe
    // value, varying only content (spilled local memory vs. Ds[]/shared
    // memory). Same base-offset formula (P^2 per block) and placement
    // (wMatrix 0 only, KW_GROUP_ID_0==0) as the already-hardware-safe
    // per-thread probes above.
    if (KW_GROUP_ID_0 == 0) {
        volatile REAL spillTest[4];
        int idx = tx % 4;
        for (int i = 0; i < 16; i++) {
            spillTest[idx] = (REAL) (1000 + ty * 100 + tx * 10 + idx);
            idx = (idx + totalMatrix + i) % 4;
        }
        KW_GLOBAL_VAR REAL* pt = dMatrices
            + 2 * totalMatrix * PADDED_STATE_COUNT * PADDED_STATE_COUNT
            + KW_GROUP_ID_0 * PADDED_STATE_COUNT * PADDED_STATE_COUNT;
        if (tx < EDGE && ty < EDGE) {
            pt[ty * EDGE + tx] = spillTest[idx];
        }
    }
#endif

#if defined(FW_TINYGPU) && defined(TINYGPU_DEBUG_DUMP_SHARED_SPILL_W0)
    // Opt-in, additive-only probe (TODO.md "PICK UP HERE" -> NV Phase 108):
    // User's direct proposal: since local memory's per-thread addressing
    // is corrupted (Phase 106: tx+4ty cross-thread aliasing, driven by a
    // hardware/driver-computed per-thread stack pointer -- c[0x0][0x37c]
    // -- this kernel never itself computes), what if EXPLICIT shared
    // memory were used instead? Shared-memory addressing is a normal,
    // source-level array index compiled by ptxas, not an opaque driver-
    // provided per-thread base. This tests exactly that, reusing
    // TINYGPU_DEBUG_DUMP_LOCAL_MEM_W0's *exact* write-then-read-back
    // pattern (same 16-iteration runtime-varying index recurrence,
    // already verified to force real memory traffic and to compute the
    // value arithmetic correctly) with only the storage swapped: a
    // __shared__ array sized for the *full* 256-thread block (not just
    // the 16 active EDGE x EDGE threads), indexed by each thread's own
    // true linear position (ty*16+tx) -- collision-free by construction,
    // unlike a naive ty*EDGE+tx scheme which would collide among the 240
    // threads outside the active region and confound the result with a
    // self-inflicted aliasing bug rather than a real one.
    //
    // Genuinely new territory: every existing shared-memory probe
    // (Ds[], Phase 90-103) tests "written by ty=0, read by ty>0" -- a
    // cross-thread handoff. This tests "a ty>0 thread writes to its own
    // uniquely-indexed slot, then reads it back itself" -- a same-thread
    // round-trip, never tried before. If this comes back correct for
    // ty>0 threads (unlike both Ds[] and local memory), explicit shared
    // memory is a viable replacement for register-spilled scratch space
    // in the real kernel. If it's ALSO wrong, that's a third, distinct
    // data point on how broadly this defect reaches.
    //
    // Sized 256*4 floats = 4096 bytes, added to the kernel's existing
    // ~4224-byte shmem_usage (established since early in this
    // investigation) -- total ~8320 bytes, still comfortably inside the
    // 32KB smem_cfg bucket the unmodified kernel already occupies (Phase
    // 105's own smem_cfg formula), so this doesn't shift the concurrent-
    // CTA-per-SM occupancy budget that Phase 64 established is required
    // to reproduce this bug at all.
    if (KW_GROUP_ID_0 == 0) {
        __shared__ volatile REAL sharedSpillTest[256][4];
        int myLinear = ty * 16 + tx;
        int idx = tx % 4;
        for (int i = 0; i < 16; i++) {
            sharedSpillTest[myLinear][idx] = (REAL) (1000 + ty * 100 + tx * 10 + idx);
            idx = (idx + totalMatrix + i) % 4;
        }
        KW_GLOBAL_VAR REAL* pt = dMatrices
            + 2 * totalMatrix * PADDED_STATE_COUNT * PADDED_STATE_COUNT
            + KW_GROUP_ID_0 * PADDED_STATE_COUNT * PADDED_STATE_COUNT;
        if (tx < EDGE && ty < EDGE) {
            pt[ty * EDGE + tx] = sharedSpillTest[myLinear][idx];
        }
    }
#endif

#if defined(FW_TINYGPU) && defined(TINYGPU_DEBUG_DUMP_PER_THREAD_DS)
    // Opt-in, additive-only probe (TODO.md "PICK UP HERE" -> NV Phase 93):
    // Phase 91/92's raw-value diagnostic found the real C[] output's row
    // ty=0 (the 4 threads that *write* Ds[], `if (ty==0) Ds[tx] =
    // exp(...)` above) exactly correct, while rows ty=1/2/3 (threads
    // that only *read* Ds[] across the KW_LOCAL_FENCE barrier above) are
    // wrong in ways consistent with not seeing the real, barrier-
    // published Ds[] values. This probe settles that directly: every one
    // of the 16 real threads (not just thread (0,0), unlike
    // TINYGPU_DEBUG_DUMP_MATMUL_GROUND_TRUTH above) writes its own
    // observed Ds[0..3] to a dedicated per-thread scratch region -- a
    // third region past the real matrices and the existing ground-truth
    // scratch, addressed by KW_GROUP_ID_0 and this thread's own (ty,tx).
    // If ty>0 threads' Ds[] differs from ty=0's, that's a direct,
    // hardware-observed confirmation of a broadcast/barrier-visibility
    // failure -- read straight from Ds[] itself, not inferred from the
    // real C[] output's downstream arithmetic. Sized for this
    // investigation's actual repro (BLOCKS==1, totalMatrix==16,
    // PADDED_STATE_COUNT==4); not a general-N tool, matching every other
    // probe in this file.
    {
        KW_GLOBAL_VAR REAL* pt = dMatrices
            + 2 * totalMatrix * PADDED_STATE_COUNT * PADDED_STATE_COUNT
            + KW_GROUP_ID_0 * PADDED_STATE_COUNT * PADDED_STATE_COUNT * PADDED_STATE_COUNT;
        if (tx < EDGE && ty < EDGE) {
            KW_GLOBAL_VAR REAL* slot = pt + (ty * EDGE + tx) * PADDED_STATE_COUNT;
            slot[0] = Ds[0]; slot[1] = Ds[1]; slot[2] = Ds[2]; slot[3] = Ds[3];
        }
    }
#endif

#if defined(FW_TINYGPU) && defined(TINYGPU_DEBUG_DUMP_PER_THREAD_DS_W0)
    // Opt-in, additive-only probe (TODO.md "PICK UP HERE" -> NV Phase 96):
    // the full TINYGPU_DEBUG_DUMP_PER_THREAD_DS above (all 16 wMatrix,
    // 1024 extra floats, dmat grown to 6144 bytes -- 3x the largest
    // buffer any probe in this investigation had used) triggered a real,
    // 100%-reproducible GPU fault (Phase 94/95) whose cause is still
    // unknown -- the address arithmetic itself was independently
    // verified correct at the compiled SASS level, instruction by
    // instruction, so this isn't a "wrong formula" retry. This is a
    // deliberately much smaller version of the *exact same* diagnostic:
    // only wMatrix 0's block (KW_GROUP_ID_0==0) writes its 16 threads'
    // own Ds[0..3] views, to a region sized for exactly one block (64
    // floats, not 1024) -- tests directly whether buffer *size* was the
    // operative variable, before ever re-attempting the full version.
    // Deliberately keeps the identical address-computation *mechanism*
    // as the full version (still real totalMatrix/KW_GROUP_ID_0-based
    // pointer arithmetic, not simplified away -- KW_GROUP_ID_0 is always
    // 0 on the one block that actually executes the write, but the
    // multiply/add is still really there in the compiled code) so
    // buffer size is the *only* thing that differs between this probe
    // and the full one.
    if (KW_GROUP_ID_0 == 0) {
        KW_GLOBAL_VAR REAL* pt = dMatrices
            + 2 * totalMatrix * PADDED_STATE_COUNT * PADDED_STATE_COUNT
            + KW_GROUP_ID_0 * PADDED_STATE_COUNT * PADDED_STATE_COUNT * PADDED_STATE_COUNT;
        if (tx < EDGE && ty < EDGE) {
            KW_GLOBAL_VAR REAL* slot = pt + (ty * EDGE + tx) * PADDED_STATE_COUNT;
            slot[0] = Ds[0]; slot[1] = Ds[1]; slot[2] = Ds[2]; slot[3] = Ds[3];
        }
    }
#endif

    for (int k = 0; k < EDGE; k++)
        Csub += As[ty][k] * Ds[k] * Bs[k][tx];

#if defined(FW_TINYGPU) && defined(TINYGPU_DEBUG_DUMP_FINAL_ABCD_CSUB_W0)
    // Opt-in, additive-only probe (TODO.md "PICK UP HERE" -> NV Phase 115,
    // user: "please instrument the As / Bs / Ds / Csub right before their
    // final write"): Phase 113/114 -- every individual shared/local-
    // memory access pattern this investigation has ever isolated (Ds[]'s
    // broadcast, local memory's spill/reload, As[]/Bs[]'s own cross-
    // thread pattern) is now proven correct under flat dispatch, yet
    // Phase 111's real, combined kernel rewrite still failed identically
    // to the original bug. Unlike every probe since Phase 90, this one
    // uses no private storage and no early return -- it captures the
    // REAL As/Bs/Ds/Csub, for the REAL "Last block" case, right here
    // (Csub already holds its final accumulated value; As/Bs/Ds are
    // already fully populated and synced from the barrier after the
    // "Last block" guard above), then falls straight through into the
    // real barrier/C[] write completely unmodified -- the real output
    // for every wMatrix is still produced. wMatrix 0 only, all 16 real
    // threads (tx,ty in [0,EDGE)): each captures its own As[ty][0..3]
    // (row), Bs[0..3][tx] (column), Ds[0..3] (this thread's own view of
    // the whole array, not just Ds[tx] -- broadcast visibility is
    // exactly what Phase 90-103/110 found could differ per-reader), and
    // its own already-accumulated Csub.
    if (KW_GROUP_ID_0 == 0 && tx < EDGE && ty < EDGE) {
        KW_GLOBAL_VAR REAL* pt = dMatrices
            + 2 * totalMatrix * PADDED_STATE_COUNT * PADDED_STATE_COUNT
            + KW_GROUP_ID_0 * EDGE * EDGE * (3 * EDGE + 1);
        KW_GLOBAL_VAR REAL* slot = pt + (ty * EDGE + tx) * (3 * EDGE + 1);
        slot[0] = As[ty][0]; slot[1] = As[ty][1]; slot[2] = As[ty][2]; slot[3] = As[ty][3];
        slot[4] = Bs[0][tx]; slot[5] = Bs[1][tx]; slot[6] = Bs[2][tx]; slot[7] = Bs[3][tx];
        slot[8] = Ds[0]; slot[9] = Ds[1]; slot[10] = Ds[2]; slot[11] = Ds[3];
        slot[12] = Csub;
    }
#endif

    KW_LOCAL_FENCE;

    // Write the block sub-matrix to device memory;
    // each thread writes one element

    if ((tx < EDGE || bx < BLOCKS - 1) && (ty < EDGE || by < BLOCKS - 1)) { // It's OK to write
        if (Csub < 0)
            C[PADDED_STATE_COUNT* MULTIPLY_BLOCK_SIZE * by + MULTIPLY_BLOCK_SIZE * bx +
              PADDED_STATE_COUNT * ty + tx] = 0;
        else
            C[PADDED_STATE_COUNT* MULTIPLY_BLOCK_SIZE * by + MULTIPLY_BLOCK_SIZE * bx +
              PADDED_STATE_COUNT * ty + tx] = Csub;
    }
}

KW_GLOBAL_KERNEL void kernelMatrixMulADBFirstDeriv(KW_GLOBAL_VAR REAL* dMatrices,
                                           KW_GLOBAL_VAR unsigned int* listC,
                                           KW_GLOBAL_VAR REAL* A,
                                           KW_GLOBAL_VAR REAL* D,
                                           KW_GLOBAL_VAR REAL* B,
                                           KW_GLOBAL_VAR REAL* distanceQueue,
                                           int length,
                                           int wB,
                                           int totalMatrix) {

    int wMatrix = KW_GROUP_ID_0 % totalMatrix;

    // Block index
    int bx = KW_GROUP_ID_0 / totalMatrix;
    int by = KW_GROUP_ID_1;

    // Thread index
    int tx = KW_LOCAL_ID_0;
    int ty = KW_LOCAL_ID_1;
    int BLOCKS = KW_NUM_GROUPS_1;

#ifdef CUDA
    KW_LOCAL_MEM REAL* C;
    KW_LOCAL_MEM REAL* CFirstDeriv;
    KW_LOCAL_MEM REAL distanceLength;
    KW_LOCAL_MEM REAL distanceRate;
    if (tx == 0 && ty == 0) {
        C = dMatrices + listC[wMatrix];
        CFirstDeriv = dMatrices + listC[wMatrix + totalMatrix];
        distanceLength = distanceQueue[wMatrix]; // Non-coalescent read
        distanceRate = distanceQueue[wMatrix + totalMatrix]; // Non-coalescent read
    }
#elif defined(FW_OPENCL)
    KW_GLOBAL_VAR REAL* C;
    KW_GLOBAL_VAR REAL* CFirstDeriv;
    REAL distanceLength;
    REAL distanceRate;
    C = dMatrices + listC[wMatrix];
    CFirstDeriv = dMatrices + listC[wMatrix + totalMatrix];
    distanceLength = distanceQueue[wMatrix];
    distanceRate = distanceQueue[wMatrix + totalMatrix];
#endif

    KW_LOCAL_FENCE;

    const int EDGE = PADDED_STATE_COUNT - (BLOCKS - 1) * MULTIPLY_BLOCK_SIZE;

    // Step size used to iterate through the sub-matrices of A
    int aStep = MULTIPLY_BLOCK_SIZE;

    // Step size used to iterate through the sub-matrices of B
    int bStep = MULTIPLY_BLOCK_SIZE * PADDED_STATE_COUNT;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    REAL Csub = 0;
    REAL CFirstDerivSub = 0;

    int a = PADDED_STATE_COUNT * MULTIPLY_BLOCK_SIZE * by;
    int b = MULTIPLY_BLOCK_SIZE * bx;
    int d = 0; //MULTIPLY_BLOCK_SIZE * bx;

    KW_LOCAL_MEM REAL As[MULTIPLY_BLOCK_SIZE][MULTIPLY_BLOCK_SIZE];
    KW_LOCAL_MEM REAL Bs[MULTIPLY_BLOCK_SIZE][MULTIPLY_BLOCK_SIZE];
    KW_LOCAL_MEM REAL Ds[MULTIPLY_BLOCK_SIZE][2];

    for (int i = 0; i < BLOCKS - 1; i++) {

        if (ty == 0) {
            REAL scaledEigenTmp = D[d + tx] * distanceRate;
            Ds[tx][0] = exp(scaledEigenTmp * distanceLength);
            Ds[tx][1] = scaledEigenTmp * Ds[tx][0];
        }

        As[ty][tx] = A[a + PADDED_STATE_COUNT * ty + tx];
        Bs[ty][tx] = B[b + PADDED_STATE_COUNT * ty + tx];

        KW_LOCAL_FENCE;

        for (int k = 0; k < MULTIPLY_BLOCK_SIZE; ++k) {
            Csub += As[ty][k] * Ds[k][0] * Bs[k][tx];
            CFirstDerivSub += As[ty][k] * Ds[k][1] * Bs[k][tx];
        }

        KW_LOCAL_FENCE;

        a += aStep;
        b += bStep;
        d += MULTIPLY_BLOCK_SIZE;
    }

    // Last block is too long
    if (tx < EDGE && ty < EDGE) {
        if (ty == 0) {
            REAL scaledEigenTmp = D[d + tx] * distanceRate;
            Ds[tx][0] = exp(scaledEigenTmp * distanceLength);
            Ds[tx][1] = scaledEigenTmp * Ds[tx][0];
                }

        As[ty][tx] = A[a + PADDED_STATE_COUNT * ty + tx];
        Bs[ty][tx] = B[b + PADDED_STATE_COUNT * ty + tx];

    } else {

        if (ty == 0) {
            Ds[tx][0] = 0;
            Ds[tx][1] = 0;
        }

        As[ty][tx] = 0;
        Bs[ty][tx] = 0;
    }

    KW_LOCAL_FENCE;

    for (int k = 0; k < EDGE; k++) {
        Csub += As[ty][k] * Ds[k][0] * Bs[k][tx];
        CFirstDerivSub += As[ty][k] * Ds[k][1] * Bs[k][tx];
    }

    KW_LOCAL_FENCE;

    // Write the block sub-matrix to device memory;
    // each thread writes one element

    if ((tx < EDGE || bx < BLOCKS - 1) && (ty < EDGE || by < BLOCKS - 1)) { // It's OK to write
        if (Csub < 0)
            C[PADDED_STATE_COUNT* MULTIPLY_BLOCK_SIZE * by + MULTIPLY_BLOCK_SIZE * bx +
              PADDED_STATE_COUNT * ty + tx] = 0;
        else
            C[PADDED_STATE_COUNT* MULTIPLY_BLOCK_SIZE * by + MULTIPLY_BLOCK_SIZE * bx +
              PADDED_STATE_COUNT * ty + tx] = Csub;

        CFirstDeriv[PADDED_STATE_COUNT* MULTIPLY_BLOCK_SIZE * by + MULTIPLY_BLOCK_SIZE * bx +
          PADDED_STATE_COUNT * ty + tx] = CFirstDerivSub;
    }
}

KW_GLOBAL_KERNEL void kernelMatrixMulADBSecondDeriv(KW_GLOBAL_VAR REAL* dMatrices,
                                           KW_GLOBAL_VAR unsigned int* listC,
                                           KW_GLOBAL_VAR REAL* A,
                                           KW_GLOBAL_VAR REAL* D,
                                           KW_GLOBAL_VAR REAL* B,
                                           KW_GLOBAL_VAR REAL* distanceQueue,
                                           int length,
                                           int wB,
                                           int totalMatrix) {

    int wMatrix = KW_GROUP_ID_0 % totalMatrix;

    // Block index
    int bx = KW_GROUP_ID_0 / totalMatrix;
    int by = KW_GROUP_ID_1;

    // Thread index
    int tx = KW_LOCAL_ID_0;
    int ty = KW_LOCAL_ID_1;
    int BLOCKS = KW_NUM_GROUPS_1;

#ifdef CUDA
    KW_LOCAL_MEM REAL* C;
    KW_LOCAL_MEM REAL* CFirstDeriv;
    KW_LOCAL_MEM REAL* CSecondDeriv;
    KW_LOCAL_MEM REAL distanceLength;
    KW_LOCAL_MEM REAL distanceRate;
    if (tx == 0 && ty == 0) {
        C = dMatrices + listC[wMatrix];
        CFirstDeriv = dMatrices + listC[wMatrix + totalMatrix];
        CSecondDeriv = dMatrices + listC[wMatrix + totalMatrix * 2];
        distanceLength = distanceQueue[wMatrix]; // Non-coalescent read
        distanceRate = distanceQueue[wMatrix + totalMatrix]; // Non-coalescent read
    }
#elif defined(FW_OPENCL)
    KW_GLOBAL_VAR REAL* C;
    KW_GLOBAL_VAR REAL* CFirstDeriv;
    KW_GLOBAL_VAR REAL* CSecondDeriv;
    REAL distanceLength;
    REAL distanceRate;
    C = dMatrices + listC[wMatrix];
    CFirstDeriv = dMatrices + listC[wMatrix + totalMatrix];
    CSecondDeriv = dMatrices + listC[wMatrix + totalMatrix * 2];
    distanceLength = distanceQueue[wMatrix];
    distanceRate = distanceQueue[wMatrix + totalMatrix];
#endif

    KW_LOCAL_FENCE;

    const int EDGE = PADDED_STATE_COUNT - (BLOCKS - 1) * MULTIPLY_BLOCK_SIZE;

    // Step size used to iterate through the sub-matrices of A
    int aStep = MULTIPLY_BLOCK_SIZE;

    // Step size used to iterate through the sub-matrices of B
    int bStep = MULTIPLY_BLOCK_SIZE * PADDED_STATE_COUNT;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    REAL Csub = 0;
    REAL CFirstDerivSub = 0;
    REAL CSecondDerivSub = 0;

    int a = PADDED_STATE_COUNT * MULTIPLY_BLOCK_SIZE * by;
    int b = MULTIPLY_BLOCK_SIZE * bx;
    int d = 0; //MULTIPLY_BLOCK_SIZE * bx;

    KW_LOCAL_MEM REAL As[MULTIPLY_BLOCK_SIZE][MULTIPLY_BLOCK_SIZE];
    KW_LOCAL_MEM REAL Bs[MULTIPLY_BLOCK_SIZE][MULTIPLY_BLOCK_SIZE];
    KW_LOCAL_MEM REAL Ds[MULTIPLY_BLOCK_SIZE][3];

    for (int i = 0; i < BLOCKS - 1; i++) {

        if (ty == 0) {
            REAL scaledEigenTmp = D[d + tx] * distanceRate;
            Ds[tx][0] = exp(scaledEigenTmp * distanceLength);
            Ds[tx][1] = scaledEigenTmp * Ds[tx][0];
            Ds[tx][2] = scaledEigenTmp * Ds[tx][1];
        }

        As[ty][tx] = A[a + PADDED_STATE_COUNT * ty + tx];
        Bs[ty][tx] = B[b + PADDED_STATE_COUNT * ty + tx];

        KW_LOCAL_FENCE;

        for (int k = 0; k < MULTIPLY_BLOCK_SIZE; ++k) {
            Csub += As[ty][k] * Ds[k][0] * Bs[k][tx];
            CFirstDerivSub += As[ty][k] * Ds[k][1] * Bs[k][tx];
            CSecondDerivSub += As[ty][k] * Ds[k][2] * Bs[k][tx];
        }

        KW_LOCAL_FENCE;

        a += aStep;
        b += bStep;
        d += MULTIPLY_BLOCK_SIZE;
    }

    // Last block is too long
    if (tx < EDGE && ty < EDGE) {
        if (ty == 0) {
            REAL scaledEigenTmp = D[d + tx] * distanceRate;
            Ds[tx][0] = exp(scaledEigenTmp * distanceLength);
            Ds[tx][1] = scaledEigenTmp * Ds[tx][0];
            Ds[tx][2] = scaledEigenTmp * Ds[tx][1];
                }

        As[ty][tx] = A[a + PADDED_STATE_COUNT * ty + tx];
        Bs[ty][tx] = B[b + PADDED_STATE_COUNT * ty + tx];

    } else {

        if (ty == 0) {
            Ds[tx][0] = 0;
            Ds[tx][1] = 0;
            Ds[tx][2] = 0;
        }

        As[ty][tx] = 0;
        Bs[ty][tx] = 0;
    }

    KW_LOCAL_FENCE;

    for (int k = 0; k < EDGE; k++) {
        Csub += As[ty][k] * Ds[k][0] * Bs[k][tx];
        CFirstDerivSub += As[ty][k] * Ds[k][1] * Bs[k][tx];
        CSecondDerivSub += As[ty][k] * Ds[k][2] * Bs[k][tx];
    }

    KW_LOCAL_FENCE;

    // Write the block sub-matrix to device memory;
    // each thread writes one element

    if ((tx < EDGE || bx < BLOCKS - 1) && (ty < EDGE || by < BLOCKS - 1)) { // It's OK to write
        if (Csub < 0)
            C[PADDED_STATE_COUNT* MULTIPLY_BLOCK_SIZE * by + MULTIPLY_BLOCK_SIZE * bx +
              PADDED_STATE_COUNT * ty + tx] = 0;
        else
            C[PADDED_STATE_COUNT* MULTIPLY_BLOCK_SIZE * by + MULTIPLY_BLOCK_SIZE * bx +
              PADDED_STATE_COUNT * ty + tx] = Csub;

        CFirstDeriv[PADDED_STATE_COUNT* MULTIPLY_BLOCK_SIZE * by + MULTIPLY_BLOCK_SIZE * bx +
          PADDED_STATE_COUNT * ty + tx] = CFirstDerivSub;

        CSecondDeriv[PADDED_STATE_COUNT* MULTIPLY_BLOCK_SIZE * by + MULTIPLY_BLOCK_SIZE * bx +
          PADDED_STATE_COUNT * ty + tx] = CSecondDerivSub;
    }
}

KW_GLOBAL_KERNEL void kernelMatrixConvolution(KW_GLOBAL_VAR REAL* dMatrices,
								        KW_GLOBAL_VAR unsigned int* list,
								        int totalMatrixCount
								        ) {

	    int wMatrix = KW_GROUP_ID_0 % totalMatrixCount;

	    // Block index
	    int bx = KW_GROUP_ID_0 / totalMatrixCount;
	    int by = KW_GROUP_ID_1;

	    // Thread index
	    int tx = KW_LOCAL_ID_0;
	    int ty = KW_LOCAL_ID_1;
	    int BLOCKS = KW_NUM_GROUPS_1;


#ifdef CUDA
        KW_LOCAL_MEM REAL* A;
        KW_LOCAL_MEM REAL* B;
        KW_LOCAL_MEM REAL* C;
        if (tx == 0 && ty == 0) {
            A = dMatrices + list[wMatrix]; // Non-coalescent read
            B = dMatrices + list[wMatrix + totalMatrixCount]; // Non-coalescent read
            C = dMatrices + list[wMatrix + totalMatrixCount*2]; // Non-coalescent read
        }
#elif defined(FW_OPENCL)
        KW_GLOBAL_VAR REAL* A;
        KW_GLOBAL_VAR REAL* B;
        KW_GLOBAL_VAR REAL* C;
        A = dMatrices + list[wMatrix];
        B = dMatrices + list[wMatrix + totalMatrixCount];
        C = dMatrices + list[wMatrix + totalMatrixCount*2];
#endif

	    KW_LOCAL_FENCE;

	    const int EDGE = PADDED_STATE_COUNT - (BLOCKS - 1) * MULTIPLY_BLOCK_SIZE;

	    // Step size used to iterate through the sub-matrices of A
	    int aStep = MULTIPLY_BLOCK_SIZE;

	    // Step size used to iterate through the sub-matrices of B
	    int bStep = MULTIPLY_BLOCK_SIZE * PADDED_STATE_COUNT;

	    // Csub is used to store the element of the block sub-matrix
	    // that is computed by the thread
	    REAL Csub = 0;

	    int a = PADDED_STATE_COUNT * MULTIPLY_BLOCK_SIZE * by;
	    int b = MULTIPLY_BLOCK_SIZE * bx;

	    KW_LOCAL_MEM REAL As[MULTIPLY_BLOCK_SIZE][MULTIPLY_BLOCK_SIZE];
	    KW_LOCAL_MEM REAL Bs[MULTIPLY_BLOCK_SIZE][MULTIPLY_BLOCK_SIZE];

	    for (int i = 0; i < BLOCKS - 1; i++) {

	        As[ty][tx] = A[a + PADDED_STATE_COUNT * ty + tx];
	        Bs[ty][tx] = B[b + PADDED_STATE_COUNT * ty + tx];

	        KW_LOCAL_FENCE;

	        for (int k = 0; k < MULTIPLY_BLOCK_SIZE; ++k)
	            Csub += As[ty][k]  * Bs[k][tx];

	        KW_LOCAL_FENCE;

	        a += aStep;
	        b += bStep;
	    }//END: BLOCKS loop

	    // Last block is too long
	    if (tx < EDGE && ty < EDGE) {

	#ifndef KERNEL_PRINT_ENABLED
	        KW_LOCAL_FENCE;
	#endif

	        As[ty][tx] = A[a + PADDED_STATE_COUNT * ty + tx];
	        Bs[ty][tx] = B[b + PADDED_STATE_COUNT * ty + tx];

	    } else {

	        As[ty][tx] = 0;
	        Bs[ty][tx] = 0;

	    }//END: EDGE check

	    KW_LOCAL_FENCE;

	    for (int k = 0; k < EDGE; k++) {
	        Csub += As[ty][k] *  Bs[k][tx];
	    }

	    KW_LOCAL_FENCE;

	    // Write the block sub-matrix to device memory;
	    // each thread writes one element

	    if ((tx < EDGE || bx < BLOCKS - 1) && (ty < EDGE || by < BLOCKS - 1)) { // It's OK to write
	        if (Csub < 0) {

	        	C[PADDED_STATE_COUNT* MULTIPLY_BLOCK_SIZE * by + MULTIPLY_BLOCK_SIZE * bx +
	              PADDED_STATE_COUNT * ty + tx] = 0;

	        } else {

	        	C[PADDED_STATE_COUNT* MULTIPLY_BLOCK_SIZE * by + MULTIPLY_BLOCK_SIZE * bx +
	              PADDED_STATE_COUNT * ty + tx] = Csub;

	        }//END: Csub check
	    }//END: EDGE check

}//END: kernelMatrixConvolution

KW_GLOBAL_KERNEL void kernelMatrixTranspose(KW_GLOBAL_VAR REAL* dMatrices,
                                            KW_GLOBAL_VAR unsigned int* list,
                                            int totalMatrixCount) {

	    int wMatrix = KW_GROUP_ID_0 % totalMatrixCount;

	    // Block index
	    int bx = KW_GROUP_ID_0 / totalMatrixCount;
	    int by = KW_GROUP_ID_1;

	    // Thread index
	    int tx = KW_LOCAL_ID_0;
	    int ty = KW_LOCAL_ID_1;

#ifdef CUDA
        KW_LOCAL_MEM REAL* A;
        KW_LOCAL_MEM REAL* C;
        if (tx == 0 && ty == 0) {
            A = dMatrices + list[wMatrix]; // Non-coalescent read
            C = dMatrices + list[wMatrix + totalMatrixCount]; // Non-coalescent read
        }
#elif defined(FW_OPENCL)
        KW_GLOBAL_VAR REAL* A;
        KW_GLOBAL_VAR REAL* C;
        A = dMatrices + list[wMatrix];
        C = dMatrices + list[wMatrix + totalMatrixCount];
#endif

	    KW_LOCAL_FENCE;

        const int rowOffset = MULTIPLY_BLOCK_SIZE * bx;
        const int colOffset = MULTIPLY_BLOCK_SIZE * by;

        const int row = rowOffset + tx;
        const int col = colOffset + ty;

	    KW_LOCAL_MEM REAL As[MULTIPLY_BLOCK_SIZE][MULTIPLY_BLOCK_SIZE];

	    if (row < PADDED_STATE_COUNT && col < PADDED_STATE_COUNT) {
	        As[ty][tx] = A[PADDED_STATE_COUNT * colOffset + rowOffset +
                           PADDED_STATE_COUNT * ty + tx];
	    }

	    KW_LOCAL_FENCE;

	    if (row < PADDED_STATE_COUNT && col < PADDED_STATE_COUNT) {
		    C[PADDED_STATE_COUNT * rowOffset + colOffset +
		      PADDED_STATE_COUNT * ty + tx] = As[tx][ty];
	    }
}

KW_GLOBAL_KERNEL void kernelMatrixMulADBComplexMulti(KW_GLOBAL_VAR REAL* dMatrices,
                                   KW_GLOBAL_VAR unsigned int* offsets,
                                   KW_GLOBAL_VAR REAL* Alist,
                                   KW_GLOBAL_VAR REAL* Dlist,
                                   KW_GLOBAL_VAR REAL* Blist,
                                   KW_GLOBAL_VAR REAL* distanceQueue,
                                   int length,
                                   int wB,
                                   int totalMatrix) {
#if !(defined(FW_OPENCL_APPLEAMDGPU) && defined(DOUBLE_PRECISION)) // TODO: fix this issue
    int wMatrix = KW_GROUP_ID_0 % totalMatrix;
    int offIndex = wMatrix * 3;

    // Block index
    int bx = KW_GROUP_ID_0 / totalMatrix;
    int by = KW_GROUP_ID_1;
    int BLOCKS = KW_NUM_GROUPS_1;

    // Thread index
    int tx = KW_LOCAL_ID_0;
    int ty = KW_LOCAL_ID_1;

    KW_GLOBAL_VAR REAL* C = dMatrices + offsets[offIndex];
    KW_GLOBAL_VAR REAL* B = Blist + offsets[offIndex + 1]; // dEvec
    KW_GLOBAL_VAR REAL* A = Alist + offsets[offIndex + 1]; // dIevc
    KW_GLOBAL_VAR REAL* D = Dlist + offsets[offIndex + 2]; // dEigenValues
    REAL distance = distanceQueue[wMatrix];

    const int EDGE = PADDED_STATE_COUNT - (BLOCKS - 1) * MULTIPLY_BLOCK_SIZE;

    // Step size used to iterate through the sub-matrices of A
    int aStep = MULTIPLY_BLOCK_SIZE;

    // Step size used to iterate through the sub-matrices of B
    int bStep = MULTIPLY_BLOCK_SIZE * PADDED_STATE_COUNT;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    REAL Csub = 0;

    int a = PADDED_STATE_COUNT * MULTIPLY_BLOCK_SIZE * by;
    int b = MULTIPLY_BLOCK_SIZE * bx;
    int d = 0; //MULTIPLY_BLOCK_SIZE * bx;

    KW_LOCAL_MEM REAL As[MULTIPLY_BLOCK_SIZE][MULTIPLY_BLOCK_SIZE];
    KW_LOCAL_MEM REAL Bs[MULTIPLY_BLOCK_SIZE + 2][MULTIPLY_BLOCK_SIZE];
    KW_LOCAL_MEM REAL Cs[MULTIPLY_BLOCK_SIZE];
    KW_LOCAL_MEM REAL Ds[MULTIPLY_BLOCK_SIZE];
    KW_LOCAL_MEM REAL Es[MULTIPLY_BLOCK_SIZE + 2];

#if defined(CUDA) || defined(FW_TINYGPU_HYBRID_AMD)
    REAL* B0  = &Bs[1][0];
    REAL* Bm1 = &Bs[0][0];
    REAL* Bp1 = &Bs[2][0];
    REAL* E0  = &Es[1];
#elif defined(FW_OPENCL)
    KW_LOCAL_MEM REAL* B0  = &Bs[1][0];
    KW_LOCAL_MEM REAL* Bm1 = &Bs[0][0];
    KW_LOCAL_MEM REAL* Bp1 = &Bs[2][0];
    KW_LOCAL_MEM REAL* E0  = &Es[1];
#endif

    // Zero first row of Bs and Es
    if (ty == 0) {
        Bs[0][tx] = 0;
        if (tx == 0) {
            Es[0] = 0;
        }
    }

    while (d + MULTIPLY_BLOCK_SIZE < PADDED_STATE_COUNT) {

//      READ_SCHUR_VALUES();
        if (ty == 0) {
            Ds[tx] = exp(D[d + tx] * distance);
            Cs[tx] = D[d + PADDED_STATE_COUNT + tx] * distance;
            if (Cs[tx]) {
                REAL expat = Ds[tx];
                REAL cosbt = cos(Cs[tx]);
#ifdef FW_OPENCL_AMDGPU
                Cs[tx] = -expat * sin(Cs[tx] + 0.0);
#else
                Cs[tx] = -expat * sin(Cs[tx]);
#endif
                Ds[tx] *= cosbt;
            }
        }

        // Block read A and B sub-matrices
        As[ty][tx] = A[a + PADDED_STATE_COUNT * ty + tx];
        B0[ty * MULTIPLY_BLOCK_SIZE + tx] = B[b + PADDED_STATE_COUNT * ty + tx];

        // Read extra row of B for Bp1
        if (ty == 0) {
            B0[MULTIPLY_BLOCK_SIZE * MULTIPLY_BLOCK_SIZE + tx] =
                    B[b + PADDED_STATE_COUNT * MULTIPLY_BLOCK_SIZE + tx];
        }

        // All necessary values loaded
        KW_LOCAL_FENCE;

//      POPULATE_SCHUR_BAND(MULTIPLY_BLOCK_SIZE);
        if (ty == 0 && tx == 0) {
            for(int k=0; k<MULTIPLY_BLOCK_SIZE; k++) {
                if (Cs[k] && !Es[k]) {
                    E0[k] = Cs[k];
                } else {
                    E0[k] = 0;
                }
            }
        }


        KW_LOCAL_FENCE;

//      DO_MULTIPLICATION(MULTIPLY_BLOCK_SIZE);
        for (int k = 0; k < MULTIPLY_BLOCK_SIZE; k++) {
            Csub += As[ty][k] * (
                    Ds[k] * B0 [k * MULTIPLY_BLOCK_SIZE + tx]
                  + E0[k] * Bp1[k * MULTIPLY_BLOCK_SIZE + tx]
                  - Es[k] * Bm1[k * MULTIPLY_BLOCK_SIZE + tx]
            );
        }


        // Move last entries in B0 and E0 to first entries in Bs and Es
        if (ty == 0) {
            Bm1[tx] = Bm1[MULTIPLY_BLOCK_SIZE*MULTIPLY_BLOCK_SIZE + tx];
            if (tx == 0) {
                Es[0] = Es[MULTIPLY_BLOCK_SIZE];
            }
        }

        KW_LOCAL_FENCE;

        // Increment sub-matrices
        a += aStep;
        b += bStep;
        d += MULTIPLY_BLOCK_SIZE;

    }

    if (tx < EDGE && ty < EDGE) { // Last block is too long

//      READ_SCHUR_VALUES();
        if (ty == 0) {
            Ds[tx] = exp(D[d + tx] * distance);
            Cs[tx] = D[d + PADDED_STATE_COUNT + tx] * distance;
            if (Cs[tx]) {
                REAL expat = Ds[tx];
                REAL cosbt = cos(Cs[tx]);
#ifdef FW_OPENCL_AMDGPU
                Cs[tx] = -expat * sin(Cs[tx] + 0.0);
#else
                Cs[tx] = -expat * sin(Cs[tx]);
#endif
                Ds[tx] *= cosbt;
            }
        }

        As[ty][tx] = A[a + PADDED_STATE_COUNT * ty + tx];
        B0[ty * MULTIPLY_BLOCK_SIZE + tx] = B[b + PADDED_STATE_COUNT * ty + tx];

    } else {
        if (ty == 0) {
            Ds[tx] = 0;
            Cs[tx] = 0;
        }
        As[ty][tx] = 0;
        B0[ty * MULTIPLY_BLOCK_SIZE + tx] = 0;
    }

    // Zero last row of Bs and Es (only for unrolled iteration at end)
    if (ty == 0) {
        Bs[MULTIPLY_BLOCK_SIZE+1][tx] = 0;
    }

    // All necessary values loaded
    KW_LOCAL_FENCE;

//  POPULATE_SCHUR_BAND(EDGE);
    if (ty == 0 && tx == 0) {
        for(int k=0; k<EDGE; k++) {
            if (Cs[k] && !Es[k]) {
                E0[k] = Cs[k];
            } else {
                E0[k] = 0;
            }
        }
    }

    KW_LOCAL_FENCE;

    // Do matrix multiplication
//  DO_MULTIPLICATION(EDGE);
    for (int k = 0; k < EDGE; k++) {
        Csub += As[ty][k] * (
                Ds[k] * B0 [k * MULTIPLY_BLOCK_SIZE + tx]
              + E0[k] * Bp1[k * MULTIPLY_BLOCK_SIZE + tx]
              - Es[k] * Bm1[k * MULTIPLY_BLOCK_SIZE + tx]
        );
    }


    KW_LOCAL_FENCE;

    // Write the block sub-matrix to device memory;
    // each thread writes one element

    if (Csub < 0)
        Csub = 0;

    if ((tx < EDGE || bx < BLOCKS - 1) && (ty < EDGE || by < BLOCKS - 1)) { // It's OK to write
        C[PADDED_STATE_COUNT* MULTIPLY_BLOCK_SIZE * by + MULTIPLY_BLOCK_SIZE * bx +
              PADDED_STATE_COUNT * ty + tx] = Csub;
    }
#endif
}


KW_GLOBAL_KERNEL void kernelMatrixMulADBComplex(KW_GLOBAL_VAR REAL* dMatrices,
                                   KW_GLOBAL_VAR unsigned int* listC,
                                   KW_GLOBAL_VAR REAL* A,
                                   KW_GLOBAL_VAR REAL* D,
                                   KW_GLOBAL_VAR REAL* B,
                                   KW_GLOBAL_VAR REAL* distanceQueue,
                                   int length,
                                   int wB,
                                   int totalMatrix) {
#if !(defined(FW_OPENCL_APPLEAMDGPU) && defined(DOUBLE_PRECISION)) // TODO: fix this issue
    int wMatrix = KW_GROUP_ID_0 % totalMatrix;

    // Block index
    int bx = KW_GROUP_ID_0 / totalMatrix;
    int by = KW_GROUP_ID_1;
    int BLOCKS = KW_NUM_GROUPS_1;

    // Thread index
    int tx = KW_LOCAL_ID_0;
    int ty = KW_LOCAL_ID_1;

#ifdef CUDA
    KW_LOCAL_MEM REAL* C;
    KW_LOCAL_MEM REAL distance;
    if (tx == 0 && ty == 0) {
        C = dMatrices + listC[wMatrix];
        distance = distanceQueue[wMatrix]; // Non-coalescent read
    }
#elif defined(FW_OPENCL)
    KW_GLOBAL_VAR REAL* C;
    REAL distance;
    C = dMatrices + listC[wMatrix];
    distance = distanceQueue[wMatrix];
#endif

    KW_LOCAL_FENCE;

    const int EDGE = PADDED_STATE_COUNT - (BLOCKS - 1) * MULTIPLY_BLOCK_SIZE;

    // Step size used to iterate through the sub-matrices of A
    int aStep = MULTIPLY_BLOCK_SIZE;

    // Step size used to iterate through the sub-matrices of B
    int bStep = MULTIPLY_BLOCK_SIZE * PADDED_STATE_COUNT;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    REAL Csub = 0;

    int a = PADDED_STATE_COUNT * MULTIPLY_BLOCK_SIZE * by;
    int b = MULTIPLY_BLOCK_SIZE * bx;
    int d = 0; //MULTIPLY_BLOCK_SIZE * bx;

    KW_LOCAL_MEM REAL As[MULTIPLY_BLOCK_SIZE][MULTIPLY_BLOCK_SIZE];
    KW_LOCAL_MEM REAL Bs[MULTIPLY_BLOCK_SIZE + 2][MULTIPLY_BLOCK_SIZE];
    KW_LOCAL_MEM REAL Cs[MULTIPLY_BLOCK_SIZE];
    KW_LOCAL_MEM REAL Ds[MULTIPLY_BLOCK_SIZE];
    KW_LOCAL_MEM REAL Es[MULTIPLY_BLOCK_SIZE + 2];

#if defined(CUDA) || defined(FW_TINYGPU_HYBRID_AMD)
   	REAL* B0  = &Bs[1][0];
   	REAL* Bm1 = &Bs[0][0];
   	REAL* Bp1 = &Bs[2][0];
   	REAL* E0  = &Es[1];
#elif defined(FW_OPENCL)
   	KW_LOCAL_MEM REAL* B0  = &Bs[1][0];
   	KW_LOCAL_MEM REAL* Bm1 = &Bs[0][0];
   	KW_LOCAL_MEM REAL* Bp1 = &Bs[2][0];
   	KW_LOCAL_MEM REAL* E0  = &Es[1];
#endif

   	// Zero first row of Bs and Es
   	if (ty == 0) {
   		Bs[0][tx] = 0;
   		if (tx == 0) {
   			Es[0] = 0;
   		}
   	}

    while (d + MULTIPLY_BLOCK_SIZE < PADDED_STATE_COUNT) {

//      READ_SCHUR_VALUES();
		if (ty == 0) {
			Ds[tx] = exp(D[d + tx] * distance);
			Cs[tx] = D[d + PADDED_STATE_COUNT + tx] * distance;
			if (Cs[tx]) {
            	REAL expat = Ds[tx];
            	REAL cosbt = cos(Cs[tx]);
#ifdef FW_OPENCL_AMDGPU
                Cs[tx] = -expat * sin(Cs[tx] + 0.0);
#else
                Cs[tx] = -expat * sin(Cs[tx]);
#endif
            	Ds[tx] *= cosbt;
            }
        }

        // Block read A and B sub-matrices
        As[ty][tx] = A[a + PADDED_STATE_COUNT * ty + tx];
        B0[ty * MULTIPLY_BLOCK_SIZE + tx] = B[b + PADDED_STATE_COUNT * ty + tx];

        // Read extra row of B for Bp1
        if (ty == 0) {
        	B0[MULTIPLY_BLOCK_SIZE * MULTIPLY_BLOCK_SIZE + tx] =
        			B[b + PADDED_STATE_COUNT * MULTIPLY_BLOCK_SIZE + tx];
        }

        // All necessary values loaded
    	KW_LOCAL_FENCE;

//    	POPULATE_SCHUR_BAND(MULTIPLY_BLOCK_SIZE);
		if (ty == 0 && tx == 0) {
			for(int k=0; k<MULTIPLY_BLOCK_SIZE; k++) {
				if (Cs[k] && !Es[k]) {
					E0[k] = Cs[k];
				} else {
					E0[k] = 0;
				}
			}
		}


    	KW_LOCAL_FENCE;

//      DO_MULTIPLICATION(MULTIPLY_BLOCK_SIZE);
		for (int k = 0; k < MULTIPLY_BLOCK_SIZE; k++) {
			Csub += As[ty][k] * (
					Ds[k] * B0 [k * MULTIPLY_BLOCK_SIZE + tx]
				  + E0[k] * Bp1[k * MULTIPLY_BLOCK_SIZE + tx]
				  - Es[k] * Bm1[k * MULTIPLY_BLOCK_SIZE + tx]
			);
		}


        // Move last entries in B0 and E0 to first entries in Bs and Es
        if (ty == 0) {
        	Bm1[tx] = Bm1[MULTIPLY_BLOCK_SIZE*MULTIPLY_BLOCK_SIZE + tx];
        	if (tx == 0) {
        		Es[0] = Es[MULTIPLY_BLOCK_SIZE];
        	}
        }

        KW_LOCAL_FENCE;

        // Increment sub-matrices
        a += aStep;
        b += bStep;
        d += MULTIPLY_BLOCK_SIZE;

    }

    if (tx < EDGE && ty < EDGE) { // Last block is too long

//      READ_SCHUR_VALUES();
		if (ty == 0) {
			Ds[tx] = exp(D[d + tx] * distance);
			Cs[tx] = D[d + PADDED_STATE_COUNT + tx] * distance;
			if (Cs[tx]) {
            	REAL expat = Ds[tx];
            	REAL cosbt = cos(Cs[tx]);
#ifdef FW_OPENCL_AMDGPU
            	Cs[tx] = -expat * sin(Cs[tx] + 0.0);
#else
                Cs[tx] = -expat * sin(Cs[tx]);
#endif
            	Ds[tx] *= cosbt;
            }
        }

        As[ty][tx] = A[a + PADDED_STATE_COUNT * ty + tx];
        B0[ty * MULTIPLY_BLOCK_SIZE + tx] = B[b + PADDED_STATE_COUNT * ty + tx];

    } else {
    	if (ty == 0) {
    		Ds[tx] = 0;
    		Cs[tx] = 0;
    	}
    	As[ty][tx] = 0;
    	B0[ty * MULTIPLY_BLOCK_SIZE + tx] = 0;
    }

	// Zero last row of Bs and Es (only for unrolled iteration at end)
    if (ty == 0) {
    	Bs[MULTIPLY_BLOCK_SIZE+1][tx] = 0;
    }

    // All necessary values loaded
	KW_LOCAL_FENCE;

//	POPULATE_SCHUR_BAND(EDGE);
    if (ty == 0 && tx == 0) {
        for(int k=0; k<EDGE; k++) {
            if (Cs[k] && !Es[k]) {
                E0[k] = Cs[k];
            } else {
                E0[k] = 0;
            }
        }
    }

	KW_LOCAL_FENCE;

	// Do matrix multiplication
//	DO_MULTIPLICATION(EDGE);
    for (int k = 0; k < EDGE; k++) {
        Csub += As[ty][k] * (
                Ds[k] * B0 [k * MULTIPLY_BLOCK_SIZE + tx]
              + E0[k] * Bp1[k * MULTIPLY_BLOCK_SIZE + tx]
              - Es[k] * Bm1[k * MULTIPLY_BLOCK_SIZE + tx]
        );
    }


    KW_LOCAL_FENCE;

    // Write the block sub-matrix to device memory;
    // each thread writes one element

    if (Csub < 0)
    	Csub = 0;

    if ((tx < EDGE || bx < BLOCKS - 1) && (ty < EDGE || by < BLOCKS - 1)) { // It's OK to write
        C[PADDED_STATE_COUNT* MULTIPLY_BLOCK_SIZE * by + MULTIPLY_BLOCK_SIZE * bx +
              PADDED_STATE_COUNT * ty + tx] = Csub;
    }
#endif
}

KW_GLOBAL_KERNEL void kernelSumSites1(KW_GLOBAL_VAR REAL* dArray,
                                      KW_GLOBAL_VAR REAL* dSum,
                                      KW_GLOBAL_VAR REAL* dPatternWeights,
                                      int patternCount) {
#ifdef FW_OPENCL_CPU

    REAL sum = 0;

    int pattern = KW_GROUP_ID_0 * SUM_SITES_BLOCK_SIZE;
    int maxPattern = (KW_GROUP_ID_0 + 1) * SUM_SITES_BLOCK_SIZE;

    if (maxPattern > patternCount)
        maxPattern = patternCount;

    while (pattern < maxPattern) {
        FMA(dArray[pattern],  dPatternWeights[pattern], sum);
        pattern++;
    }

    dSum[KW_GROUP_ID_0] = sum;

#else

    KW_LOCAL_MEM REAL sum[SUM_SITES_BLOCK_SIZE];

    int tx = KW_LOCAL_ID_0;
    int pattern = KW_LOCAL_ID_0 + KW_GROUP_ID_0 * SUM_SITES_BLOCK_SIZE;

    if (pattern < patternCount)
        sum[tx] = dArray[pattern] * dPatternWeights[pattern];
    else
        sum[tx] = 0.0;

    KW_LOCAL_FENCE;

    for (unsigned int s = SUM_SITES_BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tx < s)
            sum[tx] += sum[tx + s];
        KW_LOCAL_FENCE;
    }

    if (tx == 0)
        dSum[KW_GROUP_ID_0] = sum[0];

#endif
}

KW_GLOBAL_KERNEL void kernelSumSites1Partition(KW_GLOBAL_VAR REAL* dArray,
                                               KW_GLOBAL_VAR REAL* dSum,
                                               KW_GLOBAL_VAR REAL* dPatternWeights,
                                               int startPattern,
                                               int endPattern) {
#ifdef FW_OPENCL_CPU

    REAL sum = 0;

    int pattern = startPattern + KW_GROUP_ID_0 * SUM_SITES_BLOCK_SIZE;
    int maxPattern = startPattern + (KW_GROUP_ID_0 + 1) * SUM_SITES_BLOCK_SIZE;

    if (maxPattern > endPattern)
        maxPattern = endPattern;

    while (pattern < maxPattern) {
        FMA(dArray[pattern],  dPatternWeights[pattern], sum);
        pattern++;
    }

    dSum[KW_GROUP_ID_0] = sum;

#else

    KW_LOCAL_MEM REAL sum[SUM_SITES_BLOCK_SIZE];

    int tx = KW_LOCAL_ID_0;
    int pattern = startPattern + KW_LOCAL_ID_0 + KW_GROUP_ID_0 * SUM_SITES_BLOCK_SIZE;

    if (pattern < endPattern)
        sum[tx] = dArray[pattern] * dPatternWeights[pattern];
    else
        sum[tx] = 0.0;

    KW_LOCAL_FENCE;

    for (unsigned int s = SUM_SITES_BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tx < s)
            sum[tx] += sum[tx + s];
        KW_LOCAL_FENCE;
    }

    if (tx == 0)
        dSum[KW_GROUP_ID_0] = sum[0];

#endif
}

// KW_GLOBAL_KERNEL void kernelSumSites1Partition(KW_GLOBAL_VAR REAL*         dArray,
//                                                KW_GLOBAL_VAR REAL*         dSum,
//                                                KW_GLOBAL_VAR REAL*         dPatternWeights,
//                                                KW_GLOBAL_VAR unsigned int* dPtrOffsets) {

//     int opIndexPtr = KW_GROUP_ID_0 * 2;
//     int startPattern = dPtrOffsets[opIndexPtr    ];
//     int endPattern   = dPtrOffsets[opIndexPtr + 1];

// #ifdef FW_OPENCL_CPU

//     REAL sum = 0;

//     int pattern = startPattern + KW_GROUP_ID_0 * SUM_SITES_BLOCK_SIZE;

//     while (pattern < endPattern) {
//         FMA(dArray[pattern],  dPatternWeights[pattern], sum);
//         pattern++;
//     }

//     dSum[KW_GROUP_ID_0] = sum;

// #else

//     KW_LOCAL_MEM REAL sum[SUM_SITES_BLOCK_SIZE];

//     int tx = KW_LOCAL_ID_0;
//     int pattern = startPattern + KW_LOCAL_ID_0 + KW_GROUP_ID_0 * SUM_SITES_BLOCK_SIZE;

//     if (pattern < endPattern)
//         sum[tx] = dArray[pattern] * dPatternWeights[pattern];
//     else
//         sum[tx] = 0.0;

//     KW_LOCAL_FENCE;

//     for (unsigned int s = SUM_SITES_BLOCK_SIZE / 2; s > 0; s >>= 1) {
//         if (tx < s)
//             sum[tx] += sum[tx + s];
//         KW_LOCAL_FENCE;
//     }

//     if (tx == 0)
//         dSum[KW_GROUP_ID_0] = sum[0];

// #endif
// }

KW_GLOBAL_KERNEL void kernelSumSites2(KW_GLOBAL_VAR REAL* dArray1,
                                      KW_GLOBAL_VAR REAL* dSum1,
                                      KW_GLOBAL_VAR REAL* dArray2,
                                      KW_GLOBAL_VAR REAL* dSum2,
                                      KW_GLOBAL_VAR REAL* dPatternWeights,
                                      int patternCount) {

#ifdef FW_OPENCL_CPU

    REAL sum1 = 0, sum2 = 0;

    int pattern = KW_GROUP_ID_0 * SUM_SITES_BLOCK_SIZE;
    int maxPattern = (KW_GROUP_ID_0 + 1) * SUM_SITES_BLOCK_SIZE;

    if (maxPattern > patternCount)
        maxPattern = patternCount;

    while (pattern < maxPattern) {
        FMA(dArray1[pattern],  dPatternWeights[pattern], sum1);
        FMA(dArray2[pattern],  dPatternWeights[pattern], sum2);
        pattern++;
    }

    dSum1[KW_GROUP_ID_0] = sum1;
    dSum2[KW_GROUP_ID_0] = sum2;

#else

    KW_LOCAL_MEM REAL sum1[SUM_SITES_BLOCK_SIZE];
    KW_LOCAL_MEM REAL sum2[SUM_SITES_BLOCK_SIZE];

    int tx = KW_LOCAL_ID_0;
    int pattern = KW_LOCAL_ID_0 + KW_GROUP_ID_0 * SUM_SITES_BLOCK_SIZE;

    if (pattern < patternCount) {
        REAL pWeight = dPatternWeights[pattern];
        sum1[tx] = dArray1[pattern] * pWeight;
        sum2[tx] = dArray2[pattern] * pWeight;
    } else {
        sum1[tx] = 0.0;
        sum2[tx] = 0.0;
    }

    KW_LOCAL_FENCE;

    for (unsigned int s = SUM_SITES_BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tx < s) {
            sum1[tx] += sum1[tx + s];
            sum2[tx] += sum2[tx + s];
        }
        KW_LOCAL_FENCE;
    }

    if (tx == 0) {
        dSum1[KW_GROUP_ID_0] = sum1[0];
        dSum2[KW_GROUP_ID_0] = sum2[0];
    }

#endif
}

KW_GLOBAL_KERNEL void kernelSumSites3(KW_GLOBAL_VAR REAL* dArray1,
                                      KW_GLOBAL_VAR REAL* dSum1,
                                      KW_GLOBAL_VAR REAL* dArray2,
                                      KW_GLOBAL_VAR REAL* dSum2,
                                      KW_GLOBAL_VAR REAL* dArray3,
                                      KW_GLOBAL_VAR REAL* dSum3,
                                      KW_GLOBAL_VAR REAL* dPatternWeights,
                                      int patternCount) {

#ifdef FW_OPENCL_CPU

    REAL sum1 = 0, sum2 = 0, sum3 = 0;

    int pattern = KW_GROUP_ID_0 * SUM_SITES_BLOCK_SIZE;
    int maxPattern = (KW_GROUP_ID_0 + 1) * SUM_SITES_BLOCK_SIZE;

    if (maxPattern > patternCount)
        maxPattern = patternCount;

    while (pattern < maxPattern) {
        FMA(dArray1[pattern],  dPatternWeights[pattern], sum1);
        FMA(dArray2[pattern],  dPatternWeights[pattern], sum2);
        FMA(dArray3[pattern],  dPatternWeights[pattern], sum3);

        pattern++;
    }

    dSum1[KW_GROUP_ID_0] = sum1;
    dSum2[KW_GROUP_ID_0] = sum2;
    dSum3[KW_GROUP_ID_0] = sum3;

#else

    KW_LOCAL_MEM REAL sum1[SUM_SITES_BLOCK_SIZE];
    KW_LOCAL_MEM REAL sum2[SUM_SITES_BLOCK_SIZE];
    KW_LOCAL_MEM REAL sum3[SUM_SITES_BLOCK_SIZE];

    int tx = KW_LOCAL_ID_0;
    int pattern = KW_LOCAL_ID_0 + KW_GROUP_ID_0 * SUM_SITES_BLOCK_SIZE;

    if (pattern < patternCount) {
        REAL pWeight = dPatternWeights[pattern];
        sum1[tx] = dArray1[pattern] * pWeight;
        sum2[tx] = dArray2[pattern] * pWeight;
        sum3[tx] = dArray3[pattern] * pWeight;
    } else {
        sum1[tx] = 0.0;
        sum2[tx] = 0.0;
        sum3[tx] = 0.0;
    }

    KW_LOCAL_FENCE;

    for (unsigned int s = SUM_SITES_BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tx < s) {
            sum1[tx] += sum1[tx + s];
            sum2[tx] += sum2[tx + s];
            sum3[tx] += sum3[tx + s];
        }
        KW_LOCAL_FENCE;
    }

    if (tx == 0) {
        dSum1[KW_GROUP_ID_0] = sum1[0];
        dSum2[KW_GROUP_ID_0] = sum2[0];
        dSum3[KW_GROUP_ID_0] = sum3[0];
    }

#endif
}

KW_GLOBAL_KERNEL void kernelAccumulateFactors(KW_GLOBAL_VAR REAL* dScalingFactors,
                                              KW_GLOBAL_VAR unsigned int* dNodePtrQueue,
                                              KW_GLOBAL_VAR REAL* rootScaling,
                                              int nodeCount,
                                              int patternCount) {

    int pattern = KW_LOCAL_ID_0 + KW_GROUP_ID_0 * PATTERN_BLOCK_SIZE;

    REAL total = 0;
    KW_GLOBAL_VAR REAL* nodeScales;

    int n;
    for(n = 0; n < nodeCount; n++) {
//      if (KW_LOCAL_ID_0 == 0) // TODO Why does this not work???
        nodeScales = dScalingFactors + dNodePtrQueue[n];
//      KW_LOCAL_FENCE;

    #ifdef KERNEL_PRINT_ENABLED
        if (pattern == 1)
            printf("added %1.2e\n", nodeScales[pattern]);
    #endif
        REAL factor = nodeScales[pattern];
        if (factor != 1.0) {
            total += log(factor);
        }
    }

#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    rootScaling[pattern] += total;
#else // GPU implementation
    if (pattern < patternCount)
        rootScaling[pattern] += total;
#endif // FW_OPENCL_CPU
}

KW_GLOBAL_KERNEL void kernelAccumulateFactorsByPartition(KW_GLOBAL_VAR REAL* dScalingFactors,
                                                         KW_GLOBAL_VAR unsigned int* dNodePtrQueue,
                                                         KW_GLOBAL_VAR REAL* rootScaling,
                                                         int nodeCount,
                                                         int startPattern,
                                                         int endPattern) {

    int pattern = startPattern + KW_LOCAL_ID_0 + KW_GROUP_ID_0 * PATTERN_BLOCK_SIZE;

    REAL total = 0;
    KW_GLOBAL_VAR REAL* nodeScales;

    int n;
    for(n = 0; n < nodeCount; n++) {
        nodeScales = dScalingFactors + dNodePtrQueue[n];

        REAL factor = nodeScales[pattern];
        if (factor != 1.0) {
            total += log(factor);
        }
    }

    if (pattern < endPattern) {
        rootScaling[pattern] += total;
    }
}

KW_GLOBAL_KERNEL void kernelAccumulateFactorsScalersLog(KW_GLOBAL_VAR REAL* dScalingFactors,
                                                 KW_GLOBAL_VAR unsigned int* dNodePtrQueue,
                                                 KW_GLOBAL_VAR REAL* rootScaling,
                                                 int nodeCount,
                                                 int patternCount) {
    int pattern = KW_LOCAL_ID_0 + KW_GROUP_ID_0 * PATTERN_BLOCK_SIZE;

    REAL total = 0;
    KW_GLOBAL_VAR REAL* nodeScales;

    int n;
    for(n = 0; n < nodeCount; n++) {
//      if (KW_LOCAL_ID_0 == 0) // TODO Why does this not work???
        nodeScales = dScalingFactors + dNodePtrQueue[n];
//      KW_LOCAL_FENCE;

#ifdef KERNEL_PRINT_ENABLED
        if (pattern == 1)
            printf("added %1.2e\n", nodeScales[pattern]);
#endif
        total += nodeScales[pattern];
    }

#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    rootScaling[pattern] += total;
#else // GPU implementation
    if (pattern < patternCount)
        rootScaling[pattern] += total;
#endif // FW_OPENCL_CPU
}

KW_GLOBAL_KERNEL void kernelAccumulateFactorsScalersLogByPartition(
                                                KW_GLOBAL_VAR REAL* dScalingFactors,
                                                KW_GLOBAL_VAR unsigned int* dNodePtrQueue,
                                                KW_GLOBAL_VAR REAL* rootScaling,
                                                int nodeCount,
                                                int startPattern,
                                                int endPattern) {

    int pattern = startPattern + KW_LOCAL_ID_0 + KW_GROUP_ID_0 * PATTERN_BLOCK_SIZE;

    REAL total = 0;
    KW_GLOBAL_VAR REAL* nodeScales;

    int n;
    for(n = 0; n < nodeCount; n++) {
        nodeScales = dScalingFactors + dNodePtrQueue[n];

        total += nodeScales[pattern];
    }

    if (pattern < endPattern) {
        rootScaling[pattern] += total;
    }
}

KW_GLOBAL_KERNEL void kernelRemoveFactors(KW_GLOBAL_VAR REAL* dScalingFactors,
                                    KW_GLOBAL_VAR unsigned int* dNodePtrQueue,
                                                   KW_GLOBAL_VAR REAL* rootScaling,
                                                   int nodeCount,
                                                   int patternCount) {
    int pattern = KW_LOCAL_ID_0 + KW_GROUP_ID_0 * PATTERN_BLOCK_SIZE;

    REAL total = 0;
    KW_GLOBAL_VAR REAL* nodeScales;

    int n;
    for(n = 0; n < nodeCount; n++) {
//      if (KW_LOCAL_ID_0 == 0) // TODO Why does this not work???
        nodeScales = dScalingFactors + dNodePtrQueue[n];
//      KW_LOCAL_FENCE;

#ifdef KERNEL_PRINT_ENABLED
        if (pattern == 1)
            printf("added %1.2e\n", nodeScales[pattern]);
#endif
        REAL factor = nodeScales[pattern];
        if (factor != 1.0) {
            total += log(factor);
        }
    }

#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    rootScaling[pattern] -= total;
#else // GPU implementation
    if (pattern < patternCount)
        rootScaling[pattern] -= total;
#endif // FW_OPENCL_CPU
}

KW_GLOBAL_KERNEL void kernelRemoveFactorsByPartition(KW_GLOBAL_VAR REAL* dScalingFactors,
                                                     KW_GLOBAL_VAR unsigned int* dNodePtrQueue,
                                                     KW_GLOBAL_VAR REAL* rootScaling,
                                                     int nodeCount,
                                                     int startPattern,
                                                     int endPattern) {
    int pattern = startPattern + KW_LOCAL_ID_0 + KW_GROUP_ID_0 * PATTERN_BLOCK_SIZE;

    REAL total = 0;
    KW_GLOBAL_VAR REAL* nodeScales;

    int n;
    for(n = 0; n < nodeCount; n++) {
        nodeScales = dScalingFactors + dNodePtrQueue[n];

        REAL factor = nodeScales[pattern];
        if (factor != 1.0) {
            total += log(factor);
        }
    }

    if (pattern < endPattern) {
        rootScaling[pattern] -= total;
    }
}

KW_GLOBAL_KERNEL void kernelRemoveFactorsScalersLog(KW_GLOBAL_VAR REAL* dScalingFactors,
                                             KW_GLOBAL_VAR unsigned int* dNodePtrQueue,
                                             KW_GLOBAL_VAR REAL* rootScaling,
                                             int nodeCount,
                                             int patternCount) {
    int pattern = KW_LOCAL_ID_0 + KW_GROUP_ID_0 * PATTERN_BLOCK_SIZE;

    REAL total = 0;
    KW_GLOBAL_VAR REAL* nodeScales;

    int n;
    for(n = 0; n < nodeCount; n++) {
//      if (KW_LOCAL_ID_0 == 0) // TODO Why does this not work???
        nodeScales = dScalingFactors + dNodePtrQueue[n];
//      KW_LOCAL_FENCE;

#ifdef KERNEL_PRINT_ENABLED
        if (pattern == 1)
            printf("added %1.2e\n", nodeScales[pattern]);
#endif

        total += nodeScales[pattern];
    }

#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    rootScaling[pattern] -= total;
#else // GPU implementation
    if (pattern < patternCount)
        rootScaling[pattern] -= total;
#endif // FW_OPENCL_CPU
}

KW_GLOBAL_KERNEL void kernelRemoveFactorsScalersLogByPartition(KW_GLOBAL_VAR REAL* dScalingFactors,
                                                               KW_GLOBAL_VAR unsigned int* dNodePtrQueue,
                                                               KW_GLOBAL_VAR REAL* rootScaling,
                                                               int nodeCount,
                                                               int startPattern,
                                                               int endPattern) {
    int pattern = startPattern + KW_LOCAL_ID_0 + KW_GROUP_ID_0 * PATTERN_BLOCK_SIZE;

    REAL total = 0;
    KW_GLOBAL_VAR REAL* nodeScales;

    int n;
    for(n = 0; n < nodeCount; n++) {
        nodeScales = dScalingFactors + dNodePtrQueue[n];

        total += nodeScales[pattern];
    }

    if (pattern < endPattern)
        rootScaling[pattern] -= total;

}

KW_GLOBAL_KERNEL void kernelResetFactorsByPartition(KW_GLOBAL_VAR REAL* dScalingFactors,
                                                    int startPattern,
                                                    int endPattern) {
    int pattern = startPattern + KW_LOCAL_ID_0 + KW_GROUP_ID_0 * PATTERN_BLOCK_SIZE;

    if (pattern < endPattern) {
        dScalingFactors[pattern] = 0.0;
    }
}


KW_GLOBAL_KERNEL void kernelPartialsDynamicScalingSlow(KW_GLOBAL_VAR REAL* allPartials,
                                                 KW_GLOBAL_VAR REAL* scalingFactors,
                                                 int matrixCount) {
    int state = KW_LOCAL_ID_0;
    int pattern = KW_GROUP_ID_0;
    int patternCount = KW_NUM_GROUPS_0;

    KW_LOCAL_MEM REAL partials[PADDED_STATE_COUNT];

    KW_LOCAL_MEM REAL max;

    if (state == 0)
        max = 0.0;

    int m;
    for(m = 0; m < matrixCount; m++) {
        partials[state] = allPartials[m * patternCount * PADDED_STATE_COUNT + pattern *
                                      PADDED_STATE_COUNT + state];
        KW_LOCAL_FENCE;

#ifdef IS_POWER_OF_TWO
    // parallelized reduction *** only works for powers-of-2 ****
    for (int i = PADDED_STATE_COUNT / 2; i > 0; i >>= 1) {
        if (state < i) {
#else
    for (int i = SMALLEST_POWER_OF_TWO / 2; i > 0; i >>= 1) {
        if (state < i && state + i < PADDED_STATE_COUNT ) {
#endif // IS_POWER_OF_TWO
                REAL compare1 = partials[state];
                REAL compare2 = partials[state + i];
                if(compare2 > compare1)
                    partials[state] = compare2;
            }
            KW_LOCAL_FENCE;
        }
        if(state == 0) {
            if( partials[0] > max)
                max = partials[0];
        }
    }

    if(state == 0) {
        if (max == 0)
        	max = 1.0;
        scalingFactors[pattern] = max;
    }


    KW_LOCAL_FENCE;

    for(m = 0; m < matrixCount; m++)
        allPartials[m * patternCount * PADDED_STATE_COUNT + pattern * PADDED_STATE_COUNT +
                    state] /= max;

}

KW_GLOBAL_KERNEL void kernelPartialsDynamicScalingSlowScalersLog(KW_GLOBAL_VAR REAL* allPartials,
                                                          KW_GLOBAL_VAR REAL* scalingFactors,
                                                          int matrixCount) {
    int state = KW_LOCAL_ID_0;
    int pattern = KW_GROUP_ID_0;
    int patternCount = KW_NUM_GROUPS_0;

    KW_LOCAL_MEM REAL partials[PADDED_STATE_COUNT];

    KW_LOCAL_MEM REAL max;

    if (state == 0)
        max = 0.0;

    int m;
    for(m = 0; m < matrixCount; m++) {
        partials[state] = allPartials[m * patternCount * PADDED_STATE_COUNT + pattern *
                                      PADDED_STATE_COUNT + state];
        KW_LOCAL_FENCE;

#ifdef IS_POWER_OF_TWO
    // parallelized reduction *** only works for powers-of-2 ****
    for (int i = PADDED_STATE_COUNT / 2; i > 0; i >>= 1) {
        if (state < i) {
#else
    for (int i = SMALLEST_POWER_OF_TWO / 2; i > 0; i >>= 1) {
        if (state < i && state + i < PADDED_STATE_COUNT ) {
#endif // IS_POWER_OF_TWO
                REAL compare1 = partials[state];
                REAL compare2 = partials[state + i];
                if(compare2 > compare1)
                    partials[state] = compare2;
            }
            KW_LOCAL_FENCE;
        }
        if(state == 0) {
            if( partials[0] > max)
                max = partials[0];
        }
    }

    if(state == 0) {
        if (max == 0) {
        	max = 1.0;
            scalingFactors[pattern] = 0.0;
        } else {
            scalingFactors[pattern] = log(max);
        }
    }


    KW_LOCAL_FENCE;

    for(m = 0; m < matrixCount; m++)
        allPartials[m * patternCount * PADDED_STATE_COUNT + pattern * PADDED_STATE_COUNT +
                    state] /= max;

}

KW_GLOBAL_KERNEL void kernelMultipleNodeSiteReduction(KW_GLOBAL_VAR REAL* dOut,
                                                      KW_GLOBAL_VAR REAL* dIn,
                                                      KW_GLOBAL_VAR REAL* dPatternWeights,
                                                      int outOffset,
                                                      int patternCount) {
#ifdef FW_OPENCL_CPU
    // TODO
#else

    KW_LOCAL_MEM REAL reduce[MULTI_NODE_SUM_BLOCK_SIZE];

    int tx = KW_LOCAL_ID_0;
    int node = KW_GROUP_ID_0;
    int offset = patternCount * node;
    int pattern = tx;

    REAL sum = 0;

    while (pattern < patternCount) {
        FMA(dIn[offset + pattern], dPatternWeights[pattern], sum);
        pattern += MULTI_NODE_SUM_BLOCK_SIZE;
    }

    reduce[tx] = sum;

    KW_LOCAL_FENCE;

    for (unsigned int s = MULTI_NODE_SUM_BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tx < s) {
            reduce[tx] += reduce[tx + s];
        }
        KW_LOCAL_FENCE;
    }

    if (tx == 0) {
        dOut[outOffset + node] = reduce[0];
    }
#endif
}

KW_GLOBAL_KERNEL void kernelMultipleNodeSiteSquaredReduction(KW_GLOBAL_VAR REAL* dOut,
                                                             KW_GLOBAL_VAR REAL* dIn,
                                                             KW_GLOBAL_VAR REAL* dPatternWeights,
                                                             int outOffset,
                                                             int patternCount) {
#ifdef FW_OPENCL_CPU
    // TODO
#else

    KW_LOCAL_MEM REAL reduce[MULTI_NODE_SUM_BLOCK_SIZE];

    int tx = KW_LOCAL_ID_0;
    int node = KW_GROUP_ID_0;
    int offset = patternCount * node;
    int pattern = tx;

    REAL sum = 0;

    while (pattern < patternCount) {
        REAL value = dIn[offset + pattern];
        FMA(value * value, dPatternWeights[pattern], sum);
        pattern += MULTI_NODE_SUM_BLOCK_SIZE;
    }

    reduce[tx] = sum;

    KW_LOCAL_FENCE;

    for (unsigned int s = MULTI_NODE_SUM_BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tx < s) {
            reduce[tx] += reduce[tx + s];
        }
        KW_LOCAL_FENCE;
    }

    if (tx == 0) {
        dOut[outOffset + node] = reduce[0];
    }
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////
// scaling experiments kernels

KW_GLOBAL_KERNEL void kernelAccumulateFactorsAutoScaling(KW_GLOBAL_VAR signed char* dScalingFactors,
                                                   KW_GLOBAL_VAR unsigned int* dNodePtrQueue,
                                                   KW_GLOBAL_VAR int* rootScaling,
                                                   int nodeCount,
                                                   int patternCount,
                                                   int scaleBufferSize) {
    int pattern = KW_LOCAL_ID_0 + KW_GROUP_ID_0 * PATTERN_BLOCK_SIZE;
    int index = pattern + KW_GROUP_ID_1 * patternCount;

    int total = 0;
    KW_GLOBAL_VAR signed char* nodeScales;

    int n;
    for(n = 0; n < nodeCount; n++) {
//        int sIndex = dNodePtrQueue[n];
        nodeScales = dScalingFactors + dNodePtrQueue[n] * scaleBufferSize;

        total += nodeScales[index];
    }

    if (pattern < patternCount)
        rootScaling[index] = total;
}

#ifdef CUDA
} // extern "C"
#endif //CUDA
