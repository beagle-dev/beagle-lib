/*
 * @file BeagleGPUSpectralImpl.hpp
 *
 * Copyright 2026 Phylogenetic Likelihood Working Group
 *
 * This file is part of BEAGLE.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 *
 * @brief GPU spectral implementation template definitions
 *
 * @author Marc Suchard
 */

#ifndef __BeagleGPUSpectralImpl_hpp__
#define __BeagleGPUSpectralImpl_hpp__

#include "libhmsbeagle/beagle.h"

namespace beagle {
namespace gpu {

#ifdef CUDA
    namespace cuda {
#else
    namespace opencl {
#endif

///////////////////////////////////////////////////////////////////////////////
// BeagleGPUSpectralImpl

BEAGLE_GPU_TEMPLATE
BeagleGPUSpectralImpl<BEAGLE_GPU_GENERIC>::BeagleGPUSpectralImpl()
    : dSpectralDistancesOrigin(0), dSpectralDistances(NULL), hEigenIndexForMatrix(NULL),
      dEvecTOrigin(0), dEvecT(NULL), dIevcTOrigin(0), dIevcT(NULL),
      dGradientOrigin(0), dGradient(NULL), dOpBuf(0), hEigenDecompIsAllReal(NULL) {
}

BEAGLE_GPU_TEMPLATE
BeagleGPUSpectralImpl<BEAGLE_GPU_GENERIC>::~BeagleGPUSpectralImpl() {
    GPUInterface* gpuIf = this->gpu;
    if (gpuIf) {
        if (dSpectralDistancesOrigin) gpuIf->FreeMemory(dSpectralDistancesOrigin);
        if (dEvecTOrigin)             gpuIf->FreeMemory(dEvecTOrigin);
        if (dIevcTOrigin)             gpuIf->FreeMemory(dIevcTOrigin);
        if (dGradientOrigin)          gpuIf->FreeMemory(dGradientOrigin);
        if (dOpBuf)                   gpuIf->FreeMemory(dOpBuf);
    }
    free(dSpectralDistances);
    free(dEvecT);
    free(dIevcT);
    free(dGradient);
    free(hEigenIndexForMatrix);
    free(hEigenDecompIsAllReal);
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUSpectralImpl<BEAGLE_GPU_GENERIC>::createInstance(
        int tipCount,
        int partialsBufferCount,
        int compactBufferCount,
        int stateCount,
        int patternCount,
        int eigenDecompositionCount,
        int matrixCount,
        int categoryCount,
        int scaleBufferCount,
        int resourceNumber,
        int pluginResourceNumber,
        long preferenceFlags,
        long requirementFlags) {
    int rc = BeagleGPUImpl<Real>::createInstance(tipCount, partialsBufferCount, compactBufferCount,
                                                  stateCount, patternCount, eigenDecompositionCount,
                                                  matrixCount, categoryCount, scaleBufferCount,
                                                  resourceNumber, pluginResourceNumber,
                                                  preferenceFlags, requirementFlags);
    if (rc != BEAGLE_SUCCESS) return rc;

    hEigenIndexForMatrix = (int*) calloc(this->kMatrixCount, sizeof(int));

    GPUInterface* gpuIf = this->gpu;

    dSpectralDistances = (GPUPtr*) malloc(sizeof(GPUPtr) * this->kMatrixCount);
    size_t distStride = this->kCategoryCount * sizeof(Real);
    dSpectralDistancesOrigin = gpuIf->AllocateMemory(this->kMatrixCount * distStride);
    for (int i = 0; i < this->kMatrixCount; i++) {
        dSpectralDistances[i] = gpuIf->CreateSubPointer(dSpectralDistancesOrigin, distStride * i, distStride);
    }

    // Backward eigenvector arrays: one S*S block per eigen decomposition.
    int S = this->kPaddedStateCount;
    size_t matStride = (size_t)S * S * sizeof(Real);
    dEvecT  = (GPUPtr*) malloc(sizeof(GPUPtr) * eigenDecompositionCount);
    dIevcT  = (GPUPtr*) malloc(sizeof(GPUPtr) * eigenDecompositionCount);
    dEvecTOrigin  = gpuIf->AllocateMemory(eigenDecompositionCount * matStride);
    dIevcTOrigin  = gpuIf->AllocateMemory(eigenDecompositionCount * matStride);
    for (int i = 0; i < eigenDecompositionCount; i++) {
        dEvecT[i]  = gpuIf->CreateSubPointer(dEvecTOrigin,  matStride * i, matStride);
        dIevcT[i]  = gpuIf->CreateSubPointer(dIevcTOrigin,  matStride * i, matStride);
    }

    /* Adjoint gradient: one S×S buffer per eigen decomposition. */
    dGradient = (GPUPtr*) malloc(sizeof(GPUPtr) * eigenDecompositionCount);
    dGradientOrigin = gpuIf->AllocateMemory(eigenDecompositionCount * matStride);
    for (int i = 0; i < eigenDecompositionCount; i++) {
        dGradient[i] = gpuIf->CreateSubPointer(dGradientOrigin, matStride * i, matStride);
    }

    /* Phase-1 scratch for generic-N adjoint: C × S × S reals. */
    size_t opBufStride = (size_t)this->kCategoryCount * (size_t)S * S * sizeof(Real);
    dOpBuf = gpuIf->AllocateMemory(opBufStride);

    hEigenDecompIsAllReal = (bool*) calloc(eigenDecompositionCount, sizeof(bool));

    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUSpectralImpl<BEAGLE_GPU_GENERIC>::updateTransitionMatrices(
        int eigenIndex,
        const int* probabilityIndices,
        const int* firstDerivativeIndices,
        const int* secondDerivativeIndices,
        const double* edgeLengths,
        int count) {
    int rc = BeagleGPUImpl<Real>::updateTransitionMatrices(eigenIndex, probabilityIndices,
                                                            firstDerivativeIndices,
                                                            secondDerivativeIndices,
                                                            edgeLengths, count);
    if (rc != BEAGLE_SUCCESS || firstDerivativeIndices != NULL) return rc;

    if (count > 0) {
        const double* categoryRates = this->hCategoryRates[0];
        int nCat = this->kCategoryCount;
        Real* hDist = (Real*) malloc(sizeof(Real) * nCat);
        GPUInterface* gpuIf = this->gpu;
        for (int i = 0; i < count; i++) {
            int matIdx = probabilityIndices[i];
            hEigenIndexForMatrix[matIdx] = eigenIndex;
            for (int j = 0; j < nCat; j++) {
                hDist[j] = (Real)(edgeLengths[i] * categoryRates[j]);
            }
            gpuIf->MemcpyHostToDevice(dSpectralDistances[matIdx], hDist, sizeof(Real) * nCat);
        }
        free(hDist);
    }
    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
void BeagleGPUSpectralImpl<BEAGLE_GPU_GENERIC>::dispatchPrunePP(
        GPUPtr p1, GPUPtr p2, GPUPtr p3,
        int c1MatIdx, int c2MatIdx,
        GPUPtr scalingFactors, GPUPtr cumulativeScaling,
        unsigned int startPattern, unsigned int endPattern,
        int rescale, int streamIndex, int waitIndex) {
    int ei1 = hEigenIndexForMatrix[c1MatIdx];
    int ei2 = hEigenIndexForMatrix[c2MatIdx];
    this->kernels->PartialsPartialsPruningSpectral(p1, p2, p3,
        this->dIevc[ei1], this->dEvec[ei1], this->dEigenValues[ei1], dSpectralDistances[c1MatIdx],
        this->dIevc[ei2], this->dEvec[ei2], this->dEigenValues[ei2], dSpectralDistances[c2MatIdx],
        scalingFactors, cumulativeScaling,
        this->kPaddedPatternCount, this->kCategoryCount,
        rescale, streamIndex, waitIndex);
}

BEAGLE_GPU_TEMPLATE
void BeagleGPUSpectralImpl<BEAGLE_GPU_GENERIC>::dispatchPruneSP(
        GPUPtr s1, GPUPtr p2, GPUPtr p3,
        int c1MatIdx, int c2MatIdx,
        GPUPtr scalingFactors, GPUPtr cumulativeScaling,
        unsigned int startPattern, unsigned int endPattern,
        int rescale, int streamIndex, int waitIndex) {
    int ei1 = hEigenIndexForMatrix[c1MatIdx];
    int ei2 = hEigenIndexForMatrix[c2MatIdx];
    this->kernels->StatesPartialsPruningSpectral(s1, p2, p3,
        this->dIevc[ei1], this->dEvec[ei1], this->dEigenValues[ei1], dSpectralDistances[c1MatIdx],
        this->dIevc[ei2], this->dEvec[ei2], this->dEigenValues[ei2], dSpectralDistances[c2MatIdx],
        scalingFactors, cumulativeScaling,
        this->kPaddedPatternCount, this->kCategoryCount,
        rescale, streamIndex, waitIndex);
}

BEAGLE_GPU_TEMPLATE
void BeagleGPUSpectralImpl<BEAGLE_GPU_GENERIC>::dispatchPruneSS(
        GPUPtr s1, GPUPtr s2, GPUPtr p3,
        int c1MatIdx, int c2MatIdx,
        GPUPtr scalingFactors, GPUPtr cumulativeScaling,
        unsigned int startPattern, unsigned int endPattern,
        int rescale, int streamIndex, int waitIndex) {
    int ei1 = hEigenIndexForMatrix[c1MatIdx];
    int ei2 = hEigenIndexForMatrix[c2MatIdx];
    this->kernels->StatesStatesPruningSpectral(s1, s2, p3,
        this->dIevc[ei1], this->dEvec[ei1], this->dEigenValues[ei1], dSpectralDistances[c1MatIdx],
        this->dIevc[ei2], this->dEvec[ei2], this->dEigenValues[ei2], dSpectralDistances[c2MatIdx],
        scalingFactors, cumulativeScaling,
        this->kPaddedPatternCount, this->kCategoryCount,
        rescale, streamIndex, waitIndex);
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUSpectralImpl<BEAGLE_GPU_GENERIC>::setEigenDecomposition(
        int eigenIndex,
        const double* inEigenVectors,
        const double* inInverseEigenVectors,
        const double* inEigenValues) {
    int rc = BeagleGPUImpl<Real>::setEigenDecomposition(eigenIndex, inEigenVectors,
                                                         inInverseEigenVectors, inEigenValues);
    if (rc != BEAGLE_SUCCESS) return rc;

    const int S  = this->kPaddedStateCount;
    const int SC = this->kStateCount;
    const int SS = S * S;

    Real* EvecT  = (Real*) calloc(SS, sizeof(Real));
    Real* IevcT  = (Real*) calloc(SS, sizeof(Real));

    for (int i = 0; i < SC; i++)
        for (int j = 0; j < SC; j++)
            EvecT[i * S + j] = (Real)inEigenVectors[i * SC + j];

    if (this->kFlags & BEAGLE_FLAG_INVEVEC_STANDARD) {
        for (int i = 0; i < SC; i++)
            for (int j = 0; j < SC; j++)
                IevcT[i * S + j] = (Real)inInverseEigenVectors[i * SC + j];
    } else {
        // INVEVEC_TRANSPOSED: input is (U^-1)^T; transpose to get U^-1
        for (int i = 0; i < SC; i++)
            for (int j = 0; j < SC; j++)
                IevcT[i * S + j] = (Real)inInverseEigenVectors[j * SC + i];
    }

    bool allReal = true;
    for (int i = 0; i < SC && allReal; i++)
        if (inEigenValues[SC + i] != 0.0) allReal = false;
    hEigenDecompIsAllReal[eigenIndex] = allReal;

    GPUInterface* gpuIf = this->gpu;
    gpuIf->MemcpyHostToDevice(dEvecT[eigenIndex],  EvecT,  sizeof(Real) * SS);
    gpuIf->MemcpyHostToDevice(dIevcT[eigenIndex],  IevcT,  sizeof(Real) * SS);

    free(EvecT);
    free(IevcT);
    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
void BeagleGPUSpectralImpl<BEAGLE_GPU_GENERIC>::dispatchGrowingSpectral(
        bool isTop, bool isRoot,
        GPUPtr partials1, GPUPtr c2, bool sibIsStates,
        GPUPtr partials3,
        int c1MatIdx, int c2MatIdx) {
    int ei2 = hEigenIndexForMatrix[c2MatIdx];
    GPUPtr ievc2 = this->dIevc[ei2];
    GPUPtr evec2 = this->dEvec[ei2];
    GPUPtr eval2 = this->dEigenValues[ei2];
    GPUPtr dist2 = dSpectralDistances[c2MatIdx];

    if (isRoot) {
        if (sibIsStates) {
            this->kernels->PartialsStatesGrowingSpectralTopRoot(
                partials1, c2, partials3,
                ievc2, evec2, eval2, dist2,
                this->kPaddedPatternCount, this->kCategoryCount);
        } else {
            this->kernels->PartialsPartialsGrowingSpectralTopRoot(
                partials1, c2, partials3,
                ievc2, evec2, eval2, dist2,
                this->kPaddedPatternCount, this->kCategoryCount);
        }
    } else {
        int ei1 = hEigenIndexForMatrix[c1MatIdx];
        // For BOTTOM: dEvecT[ei1] = U (used as ievc1), dIevcT[ei1] = U^-1 (used as evec1)
        // For TOP NotRoot: same backward parent transform
        GPUPtr bIevc1 = dEvecT[ei1];
        GPUPtr bEvec1 = dIevcT[ei1];
        GPUPtr eval1  = this->dEigenValues[ei1];
        GPUPtr dist1  = dSpectralDistances[c1MatIdx];

        if (isTop) {
            if (sibIsStates) {
                this->kernels->PartialsStatesGrowingSpectralTop(
                    partials1, c2, partials3,
                    bIevc1, bEvec1, eval1, dist1,
                    ievc2, evec2, eval2, dist2,
                    this->kPaddedPatternCount, this->kCategoryCount);
            } else {
                // Top NotRoot PP: reuse no-scale PP pruning kernel with backward matrices
                this->kernels->PartialsPartialsPruningSpectral(
                    partials1, c2, partials3,
                    bIevc1, bEvec1, eval1, dist1,
                    ievc2, evec2, eval2, dist2,
                    nullptr, nullptr,
                    this->kPaddedPatternCount, this->kCategoryCount,
                    -1, -1, -1);
            }
        } else {
            if (sibIsStates) {
                this->kernels->PartialsStatesGrowingSpectral(
                    partials1, c2, partials3,
                    bIevc1, bEvec1, eval1, dist1,
                    ievc2, evec2, eval2, dist2,
                    this->kPaddedPatternCount, this->kCategoryCount);
            } else {
                this->kernels->PartialsPartialsGrowingSpectral(
                    partials1, c2, partials3,
                    bIevc1, bEvec1, eval1, dist1,
                    ievc2, evec2, eval2, dist2,
                    this->kPaddedPatternCount, this->kCategoryCount);
            }
        }
    }
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUSpectralImpl<BEAGLE_GPU_GENERIC>::updatePrePartials(
        const int* operations, int operationCount,
        int cumulativeScaleIndex, BeaglePartialsType partialsType) {
    bool isTop = (partialsType == BEAGLE_PARTIALS_TOP);
    for (int op = 0; op < operationCount; op++) {
        const int parIndex          = operations[op * 7 + 0];
        const int child1Index       = operations[op * 7 + 3];
        const int child1TransMatIdx = operations[op * 7 + 4];
        const int child2Index       = operations[op * 7 + 5];
        const int child2TransMatIdx = operations[op * 7 + 6];

        bool isRoot     = isTop && (child1TransMatIdx < 0);
        bool sibIsStates = (this->dStates[child2Index] != 0);
        GPUPtr partials1 = this->dPartials[child1Index];
        GPUPtr c2        = sibIsStates ? this->dStates[child2Index] : this->dPartials[child2Index];
        GPUPtr partials3 = this->dPartials[parIndex];

        dispatchGrowingSpectral(isTop, isRoot,
                                partials1, c2, sibIsStates,
                                partials3,
                                child1TransMatIdx, child2TransMatIdx);
    }
    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUSpectralImpl<BEAGLE_GPU_GENERIC>::updatePrePartialsByPartition(
        const int* operations, int operationCount,
        BeaglePartialsType partialsType) {
    return updatePrePartials(operations, operationCount, BEAGLE_OP_NONE, partialsType);
}

///////////////////////////////////////////////////////////////////////////////
// BeagleGPUSpectralImplFactory

BEAGLE_GPU_TEMPLATE
BeagleImpl* BeagleGPUSpectralImplFactory<BEAGLE_GPU_GENERIC>::createImpl(
        int tipCount,
        int partialsBufferCount,
        int compactBufferCount,
        int stateCount,
        int patternCount,
        int eigenBufferCount,
        int matrixBufferCount,
        int categoryCount,
        int scaleBufferCount,
        int resourceNumber,
        int pluginResourceNumber,
        long preferenceFlags,
        long requirementFlags,
        int* errorCode) {
    BeagleImpl* impl = new BeagleGPUSpectralImpl<BEAGLE_GPU_GENERIC>();
    try {
        *errorCode = impl->createInstance(tipCount, partialsBufferCount, compactBufferCount,
                                          stateCount, patternCount, eigenBufferCount,
                                          matrixBufferCount, categoryCount, scaleBufferCount,
                                          resourceNumber, pluginResourceNumber,
                                          preferenceFlags, requirementFlags);
        if (*errorCode == BEAGLE_SUCCESS) {
            return impl;
        }
        delete impl;
        return NULL;
    }
    catch (...) {
        delete impl;
        *errorCode = BEAGLE_ERROR_GENERAL;
        throw;
    }
    delete impl;
    *errorCode = BEAGLE_ERROR_GENERAL;
    return NULL;
}

#ifdef CUDA
template<>
const char* BeagleGPUSpectralImplFactory<double>::getName() {
    return "GPU-DP-CUDA-Spectral";
}

template<>
const char* BeagleGPUSpectralImplFactory<float>::getName() {
    return "GPU-SP-CUDA-Spectral";
}
#else
template<>
const char* BeagleGPUSpectralImplFactory<double>::getName() {
    return "GPU-DP-OpenCL-Spectral";
}

template<>
const char* BeagleGPUSpectralImplFactory<float>::getName() {
    return "GPU-SP-OpenCL-Spectral";
}
#endif

BEAGLE_GPU_TEMPLATE
const long BeagleGPUSpectralImplFactory<BEAGLE_GPU_GENERIC>::getFlags() {
    long flags = BEAGLE_FLAG_COMPUTATION_SYNCH | BEAGLE_FLAG_COMPUTATION_ASYNCH |
                 BEAGLE_FLAG_SCALING_MANUAL | BEAGLE_FLAG_SCALING_ALWAYS |
                 BEAGLE_FLAG_SCALING_AUTO | BEAGLE_FLAG_SCALING_DYNAMIC |
                 BEAGLE_FLAG_THREADING_NONE |
                 BEAGLE_FLAG_VECTOR_NONE |
                 BEAGLE_FLAG_SCALERS_LOG | BEAGLE_FLAG_SCALERS_RAW |
                 BEAGLE_FLAG_EIGEN_COMPLEX | BEAGLE_FLAG_EIGEN_REAL |
                 BEAGLE_FLAG_INVEVEC_STANDARD | BEAGLE_FLAG_INVEVEC_TRANSPOSED |
                 BEAGLE_FLAG_PREORDER_TRANSPOSE_MANUAL | BEAGLE_FLAG_PREORDER_TRANSPOSE_AUTO |
                 BEAGLE_FLAG_PARALLELOPS_GRID | BEAGLE_FLAG_PARALLELOPS_STREAMS |
                 BEAGLE_FLAG_SPECTRAL_REPRESENTATION;

#ifdef CUDA
    flags |= BEAGLE_FLAG_FRAMEWORK_CUDA |
             BEAGLE_FLAG_PROCESSOR_GPU;
#else
    flags |= BEAGLE_FLAG_FRAMEWORK_OPENCL |
             BEAGLE_FLAG_PROCESSOR_CPU | BEAGLE_FLAG_PROCESSOR_GPU | BEAGLE_FLAG_PROCESSOR_OTHER;
#endif

    Real r = 0;
    modifyFlagsForPrecision(&flags, r);
    return flags;
}

#ifdef CUDA
template<>
char* BeagleGPUSpectralImpl<double>::getInstanceName() {
    return (char*) "CUDA-Double-Spectral";
}

template<>
char* BeagleGPUSpectralImpl<float>::getInstanceName() {
    return (char*) "CUDA-Single-Spectral";
}
#else
template<>
char* BeagleGPUSpectralImpl<double>::getInstanceName() {
    return (char*) "OpenCL-Double-Spectral";
}

template<>
char* BeagleGPUSpectralImpl<float>::getInstanceName() {
    return (char*) "OpenCL-Single-Spectral";
}
#endif

BEAGLE_GPU_TEMPLATE
int BeagleGPUSpectralImpl<BEAGLE_GPU_GENERIC>::getInstanceDetails(BeagleInstanceDetails* returnInfo) {
    int rc = BeagleGPUImpl<Real>::getInstanceDetails(returnInfo);
    if (rc == BEAGLE_SUCCESS && returnInfo != NULL) {
        returnInfo->flags |= BEAGLE_FLAG_SPECTRAL_REPRESENTATION;
        returnInfo->implName = getInstanceName();
    }
    return rc;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUSpectralImpl<BEAGLE_GPU_GENERIC>::calculateAdjointCrossProducts(
        const int* postBufferIndices,
        const int* preBufferIndices,
        const int* eigenIndices,
        const int* categoryRatesIndices,
        const int* categoryWeightsIndices,
        const int  rootPostOrderIndex,
        const int  stateFrequenciesIndex,
        int        count,
        double*    outSumDerivatives,
        double*    outSumSquaredDerivatives) {

    GPUInterface* gpuIf = this->gpu;
    const int S  = this->kPaddedStateCount;
    const int SS = S * S;
    const int cwIdx = categoryWeightsIndices[0];

    /* Compute per-site marginal likelihoods into dIntegrationTmp. */
    this->kernels->IntegrateLikelihoods(
        this->dIntegrationTmp,
        this->dPartials[rootPostOrderIndex],
        this->dWeights[cwIdx],
        this->dFrequencies[stateFrequenciesIndex],
        this->kPaddedPatternCount,
        this->kCategoryCount);

    /* Allocate host zero-buffers once, reused for device-zeroing inside the loop. */
    const size_t opBufSize = (size_t)this->kCategoryCount * SS;
    Real* hZeroGrad  = (Real*) gpuIf->CallocHost(sizeof(Real), SS);
    Real* hZeroOpBuf = (S > 4) ? (Real*) gpuIf->CallocHost(sizeof(Real), opBufSize) : NULL;

    /* Zero the gradient accumulator for each eigen decomposition used. */
    for (int i = 0; i < count; i++) {
        int ei = hEigenIndexForMatrix[eigenIndices[i]];
        gpuIf->MemcpyHostToDevice(dGradient[ei], hZeroGrad, SS * sizeof(Real));
    }

    /* Process each branch. */
    for (int i = 0; i < count; i++) {
        const int postIdx = postBufferIndices[i];
        const int preIdx  = preBufferIndices[i];
        const int matIdx  = eigenIndices[i];
        const int ei      = hEigenIndexForMatrix[matIdx];

        bool isStates  = (this->dStates[postIdx] != 0);
        bool isAllReal = hEigenDecompIsAllReal[ei];

        GPUPtr prePtr  = this->dPartials[preIdx];
        GPUPtr postPtr = isStates ? this->dStates[postIdx] : this->dPartials[postIdx];
        GPUPtr evecT   = dEvecT[ei];
        GPUPtr ievc    = this->dIevc[ei];
        GPUPtr evalPtr = this->dEigenValues[ei];
        GPUPtr distPtr = dSpectralDistances[matIdx];
        GPUPtr catW    = this->dWeights[cwIdx];

        if (S == 4) {
            this->kernels->AdjointCrossProductSpectral4(
                isStates, isAllReal,
                prePtr, postPtr, evecT, ievc, evalPtr,
                distPtr, this->dPatternWeights, catW,
                this->dIntegrationTmp, dGradient[ei],
                this->kPaddedPatternCount, this->kCategoryCount);
        } else {
            /* Zero the phase-1 scratch buffer for this branch. */
            gpuIf->MemcpyHostToDevice(dOpBuf, hZeroOpBuf, opBufSize * sizeof(Real));
            this->kernels->AdjointCrossProductPhase1SpectralN(
                isStates, isAllReal,
                prePtr, postPtr, evecT, ievc,
                distPtr, this->dPatternWeights, catW,
                this->dIntegrationTmp, dOpBuf,
                this->kPaddedPatternCount, this->kCategoryCount);
            this->kernels->AdjointCrossProductPhase2SpectralN(
                isAllReal,
                dOpBuf, evalPtr, distPtr,
                dGradient[ei],
                this->kCategoryCount);
        }
    }

    gpuIf->FreeHostMemory(hZeroGrad);
    if (hZeroOpBuf) gpuIf->FreeHostMemory(hZeroOpBuf);

    /* Download gradient to host.  The first eigen decomp's gradient is the
     * primary output.  Convert Real→double. */
    if (count > 0 && outSumDerivatives != NULL) {
        const int ei0 = hEigenIndexForMatrix[eigenIndices[0]];
        Real* hGrad = (Real*) gpuIf->CallocHost(sizeof(Real), SS);
        gpuIf->MemcpyDeviceToHost(hGrad, dGradient[ei0], SS * sizeof(Real));
        for (int k = 0; k < SS; k++)
            outSumDerivatives[k] = (double)hGrad[k];
        gpuIf->FreeHostMemory(hGrad);
    }

    return BEAGLE_SUCCESS;
}

} // namespace cuda/opencl
} // namespace gpu
} // namespace beagle

#endif // __BeagleGPUSpectralImpl_hpp__
