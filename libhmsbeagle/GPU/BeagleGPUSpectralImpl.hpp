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
      dGradientOrigin(0), dGradient(NULL), hEigenDecompIsAllReal(NULL),
      dAdjointQueue(0), hAdjointQueue(NULL), kAdjointQueueCapacity(0),
      kSpectralEigenDecompCount(0) {
}

BEAGLE_GPU_TEMPLATE
BeagleGPUSpectralImpl<BEAGLE_GPU_GENERIC>::~BeagleGPUSpectralImpl() {
    GPUInterface* gpuIf = this->gpu;
    if (gpuIf) {
        if (dSpectralDistancesOrigin) gpuIf->FreeMemory(dSpectralDistancesOrigin);
        if (dEvecTOrigin)             gpuIf->FreeMemory(dEvecTOrigin);
        if (dIevcTOrigin)             gpuIf->FreeMemory(dIevcTOrigin);
        if (dGradientOrigin)          gpuIf->FreeMemory(dGradientOrigin);
        if (dAdjointQueue)            gpuIf->FreeMemory(dAdjointQueue);
        if (hAdjointQueue)            gpuIf->FreeHostMemory(hAdjointQueue);
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

    kSpectralEigenDecompCount = eigenDecompositionCount;

    hEigenIndexForMatrix = (int*) calloc(this->kMatrixCount, sizeof(int));

    GPUInterface* gpuIf = this->gpu;

    dSpectralDistances = (GPUPtr*) malloc(sizeof(GPUPtr) * this->kMatrixCount);
    size_t distStride = gpuIf->AlignMemOffset(this->kCategoryCount * sizeof(Real));
    kSpectralDistanceStrideElements = (unsigned int)(distStride / sizeof(Real));
    dSpectralDistancesOrigin = gpuIf->AllocateMemory(this->kMatrixCount * distStride);
    for (int i = 0; i < this->kMatrixCount; i++) {
        dSpectralDistances[i] = gpuIf->CreateSubPointer(dSpectralDistancesOrigin, distStride * i, distStride);
    }

    // Backward eigenvector arrays: one S*S block per eigen decomposition.
    int S = this->kPaddedStateCount;
    size_t matStride = gpuIf->AlignMemOffset((size_t)S * S * sizeof(Real));
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
    fprintf(stderr, "[DISPATCH] SPECTRAL BeagleGPUSpectralImpl::dispatchPruneSS called\n"); fflush(stderr);
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
    // Built locally (not delegated to BeagleGPUImpl::setEigenDecomposition) so that every
    // entry in [SC, S) is guaranteed zero rather than stale hMatrixCache scratch, and so
    // column SC of every row can hold that row's row-sum (mirrors EigenDecompositionSpectral
    // on the CPU side).
    const int S  = this->kPaddedStateCount;
    const int SC = this->kStateCount;
    const int SS = S * S;
    // kEigenValuesSize is private to BeagleGPUImpl; recompute the same way it does.
    // Spectral kernels always read a real+imaginary pair per eigenstate regardless of
    // BEAGLE_FLAG_EIGEN_COMPLEX (see matching comment/fix in BeagleGPUImpl::createInstance),
    // so this local copy must widen for BEAGLE_FLAG_SPECTRAL_REPRESENTATION too, or the
    // imaginary half of the device buffer is left uninitialized.
    const int eigenValuesSize = ((this->kFlags & BEAGLE_FLAG_EIGEN_COMPLEX) ||
                                  (this->kFlags & BEAGLE_FLAG_SPECTRAL_REPRESENTATION)) ? 2 * S : S;

    // Forward:  dEvec[row*S+col]  = U[col,row];    dIevc[row*S+col]  = U^-1[col,row]
    // Backward: dEvecT[row*S+col] = U[row,col];    dIevcT[row*S+col] = U^-1[row,col]
    Real* Evec  = (Real*) calloc(SS, sizeof(Real));
    Real* Ievc  = (Real*) calloc(SS, sizeof(Real));
    Real* EvecT = (Real*) calloc(SS, sizeof(Real));
    Real* IevcT = (Real*) calloc(SS, sizeof(Real));
    Real* Eval  = (Real*) calloc(eigenValuesSize, sizeof(Real));

    for (int i = 0; i < SC; i++) {
        Real rowSumEvec  = (Real) 0;
        Real rowSumIevc  = (Real) 0;
        Real rowSumEvecT = (Real) 0;
        Real rowSumIevcT = (Real) 0;

        for (int j = 0; j < SC; j++) {
            Real evecVal  = (Real) inEigenVectors[j * SC + i];
            Real evecTVal = (Real) inEigenVectors[i * SC + j];
            Real ievcVal, ievcTVal;
            if (this->kFlags & BEAGLE_FLAG_INVEVEC_STANDARD) {
                ievcVal  = (Real) inInverseEigenVectors[j * SC + i];
                ievcTVal = (Real) inInverseEigenVectors[i * SC + j];
            } else {
                // INVEVEC_TRANSPOSED: input is (U^-1)^T
                ievcVal  = (Real) inInverseEigenVectors[i * SC + j];
                ievcTVal = (Real) inInverseEigenVectors[j * SC + i];
            }

            Evec[i * S + j]  = evecVal;
            Ievc[i * S + j]  = ievcVal;
            EvecT[i * S + j] = evecTVal;
            IevcT[i * S + j] = ievcTVal;

            rowSumEvec  += evecVal;
            rowSumIevc  += ievcVal;
            rowSumEvecT += evecTVal;
            rowSumIevcT += ievcTVal;
        }

        // Column SC is only in the padding region when S > SC; when S == SC
        // (e.g. 4- or 16-state, no padding gap) there is no spare column, and
        // writing here would run one element past the end of these SS-sized
        // buffers on the last row.
        if (SC < S) {
            Evec[i * S + SC]  = rowSumEvec;
            Ievc[i * S + SC]  = rowSumIevc;
            EvecT[i * S + SC] = rowSumEvecT;
            IevcT[i * S + SC] = rowSumIevcT;
        }
    }

    for (int i = 0; i < SC; i++)
        Eval[i] = (Real) inEigenValues[i];

    bool allReal = true;
    for (int i = 0; i < SC && allReal; i++)
        if (inEigenValues[SC + i] != 0.0) allReal = false;
    hEigenDecompIsAllReal[eigenIndex] = allReal;

    if (this->kFlags & BEAGLE_FLAG_EIGEN_COMPLEX) {
        for (int i = 0; i < SC; i++)
            Eval[S + i] = (Real) inEigenValues[SC + i];
    }

    if (getenv("BEAGLE_DEBUG_EIGEN")) {
        fprintf(stderr, "[GPU setEigenDecomposition] eigenIndex=%d S=%d SC=%d\n", eigenIndex, S, SC);
        fprintf(stderr, "[GPU] Eval: "); for (int i=0;i<eigenValuesSize;i++) fprintf(stderr, "%.6f ", (double)Eval[i]); fprintf(stderr, "\n");
        fprintf(stderr, "[GPU] Evec:\n"); for (int i=0;i<S;i++){ for(int j=0;j<S;j++) fprintf(stderr, "%.6f ", (double)Evec[i*S+j]); fprintf(stderr, "\n"); }
        fprintf(stderr, "[GPU] Ievc:\n"); for (int i=0;i<S;i++){ for(int j=0;j<S;j++) fprintf(stderr, "%.6f ", (double)Ievc[i*S+j]); fprintf(stderr, "\n"); }
        fflush(stderr);
    }
    GPUInterface* gpuIf = this->gpu;
    gpuIf->MemcpyHostToDevice(this->dEvec[eigenIndex],        Evec,  sizeof(Real) * SS);
    gpuIf->MemcpyHostToDevice(this->dIevc[eigenIndex],        Ievc,  sizeof(Real) * SS);
    gpuIf->MemcpyHostToDevice(this->dEigenValues[eigenIndex], Eval,  sizeof(Real) * eigenValuesSize);
    gpuIf->MemcpyHostToDevice(dEvecT[eigenIndex],  EvecT,  sizeof(Real) * SS);
    gpuIf->MemcpyHostToDevice(dIevcT[eigenIndex],  IevcT,  sizeof(Real) * SS);

    free(Evec);
    free(Ievc);
    free(EvecT);
    free(IevcT);
    free(Eval);
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

    /* Zero the gradient accumulator on-device, in one fill over the whole
     * pooled origin buffer (covers every eigen decomposition, not just
     * those used this call). Previously this looped per-branch, zeroing
     * the same distinct eigen index's buffer redundantly whenever several
     * branches shared it (the common case) — replaced by a single call.
     * Deliberately fills through `dGradientOrigin`, the same buffer object
     * the merged kernel addresses via `gradientOrigin + offset`, rather
     * than the per-`ei` sub-pointer `dGradient[ei]`: filling via the
     * sub-buffer view was observed to race against the kernel's access
     * through the parent view (intermittent large garbage gradients,
     * OpenCL, Apple M1 Max) even on the in-order command queue — filling
     * through the same view the kernel uses avoids that hazard. */
    gpuIf->MemsetZero(dGradientOrigin, (size_t)kSpectralEigenDecompCount * SS * sizeof(Real));

    GPUPtr catW = this->dWeights[cwIdx];

    if (count > 0) {
        /* Merged single-launch path (both S=4 and generic-N): build one
         * offset-queue record per branch (single buffer, integer offsets
         * into pooled origins — never per-branch device pointers), then one
         * launch covers every branch at once; isStates/isAllReal are read
         * per-block from the queue at runtime, so branch order doesn't
         * matter here. This design won an A/B benchmark (generic-N) against
         * a bucketed (up to 4 launches, grouped by isStates/isAllReal)
         * alternative — see STATUS.md for the numbers; the S=4 path reuses
         * the same queue format and dispatch style without repeating that
         * A/B (same shape of result expected). */
        if (count > kAdjointQueueCapacity) {
            if (dAdjointQueue) gpuIf->FreeMemory(dAdjointQueue);
            if (hAdjointQueue) gpuIf->FreeHostMemory(hAdjointQueue);
            kAdjointQueueCapacity = count;
            dAdjointQueue = gpuIf->AllocateMemory(
                sizeof(unsigned int) * kAdjointQueueCapacity * kAdjointQueueFieldsPerBranch);
            hAdjointQueue = (unsigned int*) gpuIf->CallocHost(
                sizeof(unsigned int), kAdjointQueueCapacity * kAdjointQueueFieldsPerBranch);
        }

        for (int i = 0; i < count; i++) {
            const int postIdx = postBufferIndices[i];
            const int preIdx  = preBufferIndices[i];
            const int matIdx  = eigenIndices[i];
            const int ei      = hEigenIndexForMatrix[matIdx];
            const bool isStates  = (this->dStates[postIdx] != 0);
            const bool isAllReal = hEigenDecompIsAllReal[ei];
            unsigned int* rec = hAdjointQueue + (size_t)i * kAdjointQueueFieldsPerBranch;

            rec[0] = this->getPartialsOffsetElements(preIdx);
            rec[1] = isStates ? this->getStatesOffsetElements(postIdx)
                               : this->getPartialsOffsetElements(postIdx);
            rec[2] = isStates ? 1u : 0u;
            rec[3] = (unsigned int)ei * SS;
            rec[4] = (unsigned int)ei * this->getEvecStrideElements();
            rec[5] = (unsigned int)ei * this->getEvalStrideElements();
            rec[6] = (unsigned int)matIdx * kSpectralDistanceStrideElements;
            rec[7] = (unsigned int)ei * SS;
            rec[8] = isAllReal ? 1u : 0u;
        }

        gpuIf->MemcpyHostToDevice(dAdjointQueue, hAdjointQueue,
            sizeof(unsigned int) * count * kAdjointQueueFieldsPerBranch);

        if (S == 4) {
            this->kernels->AdjointCrossProductMergedSpectral4(
                this->getPartialsOrigin(), this->getStatesOrigin(), dEvecTOrigin,
                this->dIevc[0], this->dEigenValues[0],
                dSpectralDistancesOrigin, this->dPatternWeights, catW,
                this->dIntegrationTmp, dGradientOrigin, dAdjointQueue,
                this->kPaddedPatternCount, this->kCategoryCount, count);
        } else {
            this->kernels->AdjointCrossProductMergedSpectralN(
                this->getPartialsOrigin(), this->getStatesOrigin(), dEvecTOrigin,
                this->dIevc[0], this->dEigenValues[0],
                dSpectralDistancesOrigin, this->dPatternWeights, catW,
                this->dIntegrationTmp, dGradientOrigin, dAdjointQueue,
                this->kPaddedPatternCount, this->kCategoryCount, count);
        }
    }

    /* Download gradient to host.  The first eigen decomp's gradient is the
     * primary output.  Convert Real→double.
     *
     * The device buffer is laid out S×S with S=kPaddedStateCount (padding
     * rows/cols beyond kStateCount hold unused/garbage values from the
     * padding eigenstates), but callers (matching BeagleCPUImpl's
     * convention) allocate outSumDerivatives sized kStateCount*kStateCount.
     * Copying the full padded SS run would overflow that buffer whenever
     * kPaddedStateCount > kStateCount — only copy the top-left
     * kStateCount×kStateCount submatrix, row by row with the correct
     * strides on each side. */
    if (count > 0 && outSumDerivatives != NULL) {
        const int ei0 = hEigenIndexForMatrix[eigenIndices[0]];
        const int SC = this->kStateCount;
        Real* hGrad = (Real*) gpuIf->CallocHost(sizeof(Real), SS);
        gpuIf->MemcpyDeviceToHost(hGrad, dGradient[ei0], SS * sizeof(Real));
        for (int i = 0; i < SC; i++)
            for (int j = 0; j < SC; j++)
                outSumDerivatives[i * SC + j] = (double)hGrad[i * S + j];
        gpuIf->FreeHostMemory(hGrad);
    }

    return BEAGLE_SUCCESS;
}

} // namespace cuda/opencl
} // namespace gpu
} // namespace beagle

#endif // __BeagleGPUSpectralImpl_hpp__
