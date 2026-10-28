/*
 *  BeagleGPUImpl.cpp
 *  BEAGLE
 *
 * Copyright 2024 Phylogenetic Likelihood Working Group
 *
 * This file is part of BEAGLE.
 *
 * BEAGLE is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 *
 * BEAGLE is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with BEAGLE.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @author Xiang Ji
 * @author Benjamin Redelings
 * @author Marc Suchard
 */
#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#ifndef BEAGLE_BEAGLEGPUACTIONIMPL_HPP
#define BEAGLE_BEAGLEGPUACTIONIMPL_HPP


// Duplicated from CPU code
std::independent_bits_engine<std::mt19937_64,1,unsigned short> engine;

bool random_bool()
{
    return engine();
}
double random_plus_minus_1_func(double x)
{
    if (random_bool())
        return 1;
    else
        return -1;
}

// Algorithm 2.4 from Higham and Tisseur (2000), A BLOCK ALGORITHM FOR MATRIX 1-NORM ESTIMATION,
//    WITH AN APPLICATION TO 1-NORM PSEUDOSPECTRA.
// See OneNormEst in https://eprints.maths.manchester.ac.uk/2195/1/thesis-main.pdf
//    This seems to have a bug where it checks if columns in S are parallel to EVERY column of S_old.
// See also https://github.com/gnu-octave/octave/blob/default/scripts/linear-algebra/normest1.m
// See dlacn1.f
template <typename Real>
Real normest1(const SpMatrix<Real>& A, int p, int t=2, int itmax=5)
{
    assert(p >= 0);
    assert(t != 0); // negative means t = n
    assert(itmax >= 1);

    if (p == 0) return 1.0;

    // A is (n,n);
    assert(A.rows() == A.cols());
    int n = A.cols();

    // Handle t too large
    t = std::min(n,t);

    // Interpret negative t as t == n
    if (t < 0) t = n;

    // Defer to normP1 if p=1 and n is small or we want an exact answer.
    if (p == 1 and (n <= 4 or t == n))
        return normP1(A);

    // (0) Choose starting matrix X that is (n,t) with columns of unit 1-norm.
    DnMatrix<Real> X(n,t);
    // We choose the first column to be all 1s.
    X.col(0).setOnes();
    // The other columns have randomly chosen {-1,+1} entries.
    X = X.unaryExpr( &random_plus_minus_1_func );
    // Divide by n so that the norm of each column is 1.
    X /= n;

    // 3.
    std::vector<bool> ind_hist(n,0);
    std::vector<int> indices(n,0);
    int ind_best = -1;
    Real est_old = 0;
    DnMatrix<Real> S = DnMatrix<Real>::Ones(n,t);
    DnMatrix<Real> S_old = DnMatrix<Real>::Ones(n,t);
    MatrixXi prodS(t,t);
    DnMatrix<Real> Y(n,t);
    DnMatrix<Real> Z(n,t);
    DnVector<Real> h(n);

    for(int k=1; k<=itmax; k++)
    {
        // std::cerr<<"iter "<<k<<"  X = "<<X<<"\n";
        Y = A*X; // Y is (n,t) = (n,n) * (n,t)
        for(int i=1;i<p;i++)
            Y = A*Y;
        // std::cerr<<"                 Y = "<<X<<"\n";

        auto [est, j] = ArgNormP1(Y);

        // std::cerr<<"                 est = "<<est<<"       est_old = "<<est_old<<"\n";

        if (est > est_old or k == 2)
        {
            // Note that j is in [0,t-1], but indices[j] is in [0,n-1].
            ind_best = indices[j];
            // w = Y.col(ind_best);
        }
        // std::cerr<<"  est = "<<est<<"  (est_old = "<<est_old<<")\n";
        assert(ind_best < n);

        // (1) of Algorithm 2.4
        if (est < est_old and k >= 2)
        {
            // std::cerr<<"  The new estimate ("<<est<<") is smaller than the old estimate ("<<est_old<<")\n";
            return est_old;
        }

        est_old = est;

        assert(est >= est_old);

        // S = sign(Y), 0.0 -> 1.0
        S = Y.unaryExpr([](const Real& x) {return (x>=0) ? 1.0 : -1.0 ;});

        // prodS is (t,t)
        prodS = (S_old.transpose() * S).matrix().cwiseAbs().template cast<int>() ;

        // (2) If each columns in S is parallel to SOME column of S_old
        if (prodS.colwise().maxCoeff().sum() == n * t and k >= 2)
        {
            // std::cerr<<"  All columns of S parallel to S_old\n";
            return est;
        }

        if (t > 1)
        {
            // If S(j) is parallel to S_old(i), replace S(j) with random column
            for(int j=0;j<S.cols();j++)
            {
                for(int i=0;i<S_old.cols();i++)
                    if (prodS(i,j) == n)
                    {
                        // std::cerr<<"  S.col("<<j<<") parallel to S_old.col("<<i<<")\n";
                        S.col(j) = S.col(j).unaryExpr( &random_plus_minus_1_func );
                        break;
                    }
            }

            // If S(j) is parallel to S(i) for i<j, replace S(j) with random column
            prodS = (S.transpose() * S).matrix().template cast<int>() ;
            for(int i=0;i<S.cols();i++)
                for(int j=i+1;j<S.cols();j++)
                    if (prodS(i,j) == n)
                    {
                        // std::cerr<<"  S.col("<<j<<") parallel to S.col("<<i<<")\n";
                        S.col(j) = S.col(j).unaryExpr( &random_plus_minus_1_func );
                        break;
                    }
        }

        // (3) of Algorithm 2.4
        Z = A.transpose() * S; // (n,n) * (n,t) -> (n,t)

        // Maximize across each the t entries in each row of Z.
        h = Z.cwiseAbs().rowwise().maxCoeff();  // (n,t) -> (n,1)

        // (4) of Algorithm 2.4
        if (k >= 2 and h.maxCoeff() == h[ind_best])
        {
            // std::cerr<<"  The best column ("<<ind_best<<") is not new\n";

            // According to Algorithm 2.4, we should exit here.

            // However, continuing until we find a different reason to exit
            // seems to provide greater accuracy.

            // return est;
        }

        indices.resize(n);
        for(int i=0;i<n;i++)
            indices[i] = i;

        // reorder idx so that the highest values of h[indices[i]] come first.
        std::sort(indices.begin(), indices.end(), [&](int i,int j) {return h[i] > h[j];});

        // (5) of Algorithm 2.4
        int n_found = 0;
        for(int i=0;i<t;i++)
            if (ind_hist[indices[i]])
                n_found++;

        if (n_found == t)
        {
            assert(k >= 2);
            // std::cerr<<"  All columns were found in the column history.\n";
            return est;
        }

        // find the first t indices that are not in ind_hist
        int l=0;
        for(int i=0;i<indices.size() and l < t;i++)
        {
            if (not ind_hist[indices[i]])
            {
                indices[l] = indices[i];
                l++;
            }
        }
        indices.resize( std::min(l,t) );
        assert(not indices.empty());

        int tmax = std::min<int>(t, indices.size());

        X = DnMatrix<Real>::Zero(n, tmax);
        for(int j=0; j < tmax; j++)
            X(indices[j], j) = 1; // X(:,j) = e(indices[j])

        for(int i: indices)
            ind_hist[i] = true;

        S_old = S;
    }

    return est_old;
}


namespace beagle {
namespace gpu {

#ifdef CUDA
    namespace cuda {
#else
    namespace opencl {
#endif

BEAGLE_GPU_TEMPLATE
const char* BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::getName()
{
    return BeagleGPUActionImplFactory<BEAGLE_GPU_GENERIC>::getName();
}


BEAGLE_GPU_TEMPLATE
long long BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::getFlags()
{
    auto flags = BeagleGPUImpl<BEAGLE_GPU_GENERIC>::getFlags();

    flags |= BEAGLE_FLAG_COMPUTATION_ACTION;

    return flags;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::createInstance(int tipCount,
                                  int partialsBufferCount,
                                  int compactBufferCount,
                                  int stateCount,
                                  int patternCount,
                                  int eigenDecompositionCount,
                                  int matrixCount,
                                  int categoryCount,
                                  int scaleBufferCount,
                                  int globalResourceNumber,
                                  int pluginResourceNumber,
                                  long long preferenceFlags,
                                  long long requirementFlags)
{

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUActionImpl::createInstance\n");
#endif

    // The action flag makes the base class skip its dense (S x S) eigen and transition-matrix buffers.
    int returnCode = BeagleGPUImpl<Real>::createInstance(tipCount, partialsBufferCount, 0, stateCount, patternCount,
                                        eigenDecompositionCount, matrixCount, categoryCount, scaleBufferCount, globalResourceNumber, pluginResourceNumber,
                                        preferenceFlags | BEAGLE_FLAG_COMPUTATION_ACTION, requirementFlags);
    if (returnCode != BEAGLE_SUCCESS)
        return returnCode;

    setCudaDevice();

    hBs.resize(kEigenDecompCount);
    csrBuilders.resize(kEigenDecompCount);
    hBt1Norms.resize(kEigenDecompCount);
    hdsT.resize(kEigenDecompCount);
    hdsTCurrent.assign(kEigenDecompCount, false);
    partialsGenerations.assign(kPartialsBufferCount, 0);
    matrixGenerations.assign(kMatrixCount, 0);
    qGenerations.assign(kEigenDecompCount, 0);
    actionCache.resize(kPartialsBufferCount);
    derivativeMatrices.resize(kEigenDecompCount + kMatrixCount);
    hMuBs.resize(kEigenDecompCount);
    hB1Norms.resize(kEigenDecompCount);
    hds.resize(kEigenDecompCount);

    CUBLAS_CHECK(cublasCreate(&cublasHandle));
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));
    setCudaDevice();

    dBsCsrOffsetsCache.resize(kEigenDecompCount);
    dBsCsrColumnsCache.resize(kEigenDecompCount);
    dBsCsrValuesCache.resize(kEigenDecompCount);
    dBs = std::vector<SpMatrixDevice<Real>>(kEigenDecompCount);
    // getStatistics2( ) uses d_p for p = 2 .. pMax+1.
    normEstimator.allocate(kPaddedStateCount, 5, getPMax() + 1);

    currentCacheNNZs = std::vector<int>(kEigenDecompCount, kPaddedStateCount);
    dBtOffsets.assign(kEigenDecompCount, nullptr);
    dBtColumns.assign(kEigenDecompCount, nullptr);
    dBtSlots.assign(kEigenDecompCount, nullptr);
    dBtValues.assign(kEigenDecompCount, nullptr);
    for (int i = 0; i < kEigenDecompCount; i++) {
        dBsCsrOffsetsCache[i] = cudaDeviceNew<int>(kPaddedStateCount + 1);
        dBtOffsets[i] = cudaDeviceNew<int>(kPaddedStateCount + 1);
        dBsCsrColumnsCache[i] = cudaDeviceNew<int>(currentCacheNNZs[i]);
        dBsCsrValuesCache[i] = cudaDeviceNew<Real>(currentCacheNNZs[i]);
    }


    const size_t partialsSize = (size_t) kPaddedStateCount * kPaddedPatternCount * kCategoryCount;
    // Batching operations only pays off when each is small, so cap the extra memory at about 64 MB.
    // (2 results and 2 or 3 scratch blocks per child and category of each operation.)
    const int scratchCount = interleavedScratch() ? 3 : 2;
    maxBatchOperations = std::max<size_t>(1, std::min<size_t>(kMaxPartialsProducts, (64 << 20) / ((2 + 2 * scratchCount) * partialsSize * sizeof(Real))));
    dActionResults = cudaDeviceNew<Real>(2 * partialsSize * maxBatchOperations);
    dActionScratch = cudaDeviceNew<Real>(2 * scratchCount * partialsSize * maxBatchOperations);
    const int reductionRows = kCategoryCount * kPaddedStateCount;
    dReductionScratch = cudaDeviceNew<Real>((size_t) cuda_row_chunk_count(reductionRows, kPaddedPatternCount) * kPaddedPatternCount * maxBatchOperations);

    hEigenMaps.resize(kMatrixCount);
    hEdgeMultipliers.resize(kMatrixCount * kCategoryCount);

    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::~BeagleGPUActionImpl()
{
    cudaSetDevice(gpu->GetCudaDeviceOrdinal());
    cudaStreamSynchronize(stream);

    cudaFree(dActionResults);
    cudaFree(dActionScratch);
    cudaFree(dReductionScratch);
    for (auto& entry: actionCache)
        cudaFree(entry.data);
    for (auto& matrix: derivativeMatrices) {
        cudaFree(matrix.offsets);
        cudaFree(matrix.columns);
        cudaFree(matrix.values);
    }
    cudaFree(dDerivativeScratch);
    cudaFree(dDerivativeOutput);
    cudaFree(dCrossOffsets);
    cudaFree(dCrossColumns);
    cudaFree(dCrossValues);

    for (int i = 0; i < kEigenDecompCount; i++) {
        cudaFree(dBsCsrOffsetsCache[i]);
        cudaFree(dBtOffsets[i]);
        cudaFree(dBtColumns[i]);
        cudaFree(dBtSlots[i]);
        cudaFree(dBtValues[i]);
        cudaFree(dBsCsrColumnsCache[i]);
        cudaFree(dBsCsrValuesCache[i]);
    }
    cublasDestroy(cublasHandle);
    cusparseDestroy(cusparseHandle);
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::getInstanceDetails(BeagleInstanceDetails* returnInfo) {
    BeagleGPUImpl<BEAGLE_GPU_GENERIC>::getInstanceDetails(returnInfo);
    returnInfo->implName = getInstanceName();
    return BEAGLE_SUCCESS;
}

#ifdef CUDA
template<>
char* BeagleGPUActionImpl<double>::getInstanceName() {
    return (char*) "Action-CUDA-Double";
}

template<>
char* BeagleGPUActionImpl<float>::getInstanceName() {
    return (char*) "Action-CUDA-Single";
}
#elif defined(FW_OPENCL)
            template<>
char* BeagleGPUActionImpl<double>::getInstanceName() {
    return (char*) "Action-OpenCL-Double";
}

template<>
char* BeagleGPUActionImpl<float>::getInstanceName() {
    return (char*) "Action-OpenCL-Single";
}
#endif



BEAGLE_GPU_TEMPLATE
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::setTipStates(int tipIndex, const int* inStates)
{
    std::cerr<<"\nBEAGLE: When using action-based likelihood computations, setTipStates( ) is not allowed.\n";
    std::cerr<<"        Use setTipPartials( ) instead.\n\n";

    // There does not appear to be a simple method of throwing C++ exceptions into Java through the JNI.
    // However, throwing this exception makes Java print a stack trace that shows where the setTipStates( )
    //   call is coming from.
    throw std::runtime_error("This message will not be seen");

    std::abort();
}

BEAGLE_GPU_TEMPLATE
void BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::uploadPartials(int bufferIndex, const Real* inPartials, bool tipPartials)
{
    // BEAGLE's layout is [category][pattern][state]; tip partials have one category.  Padding stays 0.
    hPartialsTransposed.assign(kPartialsSize, 0);
    for (int category = 0; category < kCategoryCount; category++) {
        const Real* in = inPartials + (tipPartials ? 0 : (size_t) category * kPatternCount * kStateCount);
        Real* out = hPartialsTransposed.data() + (size_t) category * kPaddedStateCount * kPaddedPatternCount;
        for (int pattern = 0; pattern < kPatternCount; pattern++)
            for (int state = 0; state < kStateCount; state++)
                out[(size_t) state * kPaddedPatternCount + pattern] = in[(size_t) pattern * kStateCount + state];
    }
    gpu->MemcpyHostToDevice(dPartials[bufferIndex], hPartialsTransposed.data(), sizeof(Real) * kPartialsSize);
    partialsGenerations[bufferIndex]++;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::setTipPartials(int tipIndex, const Real* inPartials) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUActionImpl::setTipPartials\n");
#endif

    // The base class assigns the tip's device buffer; then overwrite it in the action layout.
    int returnCode = BeagleGPUImpl<Real>::setTipPartials(tipIndex, inPartials);
    if (returnCode != BEAGLE_SUCCESS)
        return returnCode;
    uploadPartials(tipIndex, inPartials, true);

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUActionImpl::setTipPartials\n");
#endif

    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::setPartials(int bufferIndex, const Real* inPartials) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUActionImpl::setPartials\n");
#endif

    int returnCode = BeagleGPUImpl<Real>::setPartials(bufferIndex, inPartials);
    if (returnCode != BEAGLE_SUCCESS)
        return returnCode;
    uploadPartials(bufferIndex, inPartials, false);

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUActionImpl::setPartials\n");
#endif

    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::getPartials(int bufferIndex, int scaleIndex, double* outPartials) {
    if (bufferIndex < 0 || bufferIndex >= kPartialsBufferCount)
        return BEAGLE_ERROR_OUT_OF_RANGE;

    hPartialsTransposed.resize(kPartialsSize);
    gpu->MemcpyDeviceToHost(hPartialsTransposed.data(), dPartials[bufferIndex], sizeof(Real) * kPartialsSize);
    for (int category = 0; category < kCategoryCount; category++) {
        const Real* in = hPartialsTransposed.data() + (size_t) category * kPaddedStateCount * kPaddedPatternCount;
        double* out = outPartials + (size_t) category * kPatternCount * kStateCount;
        for (int pattern = 0; pattern < kPatternCount; pattern++)
            for (int state = 0; state < kStateCount; state++)
                out[(size_t) pattern * kStateCount + state] = in[(size_t) state * kPaddedPatternCount + pattern];
    }
    return BEAGLE_SUCCESS;
}



BEAGLE_GPU_TEMPLATE
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::updatePartials(const int* operations,
                                                      int operationCount,
                                                      int cumulativeScalingIndex) {

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUActionImpl::updatePartials\n");
#endif

    setCudaDevice();

    bool byPartition = false;
    int returnCode = upPartials(byPartition,
                                operations,
                                operationCount,
                                cumulativeScalingIndex);

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUActionImpl::updatePartials\n");
#endif

    return returnCode;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::accumulateScaleFactors(const int* scalingIndices, int count, int cumulativeScalingIndex)
{
    return addScaleFactors(scalingIndices, count, cumulativeScalingIndex, false);
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::removeScaleFactors(const int* scalingIndices, int count, int cumulativeScalingIndex)
{
    return addScaleFactors(scalingIndices, count, cumulativeScalingIndex, true);
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::addScaleFactors(const int* scalingIndices, int count, int cumulativeScalingIndex, bool subtract)
{
    // Automatic and dynamic scaling keep extra state in the base class.
    if (kFlags & (BEAGLE_FLAG_SCALING_AUTO | BEAGLE_FLAG_SCALING_DYNAMIC)) {
        return subtract ? BeagleGPUImpl<Real>::removeScaleFactors(scalingIndices, count, cumulativeScalingIndex)
                        : BeagleGPUImpl<Real>::accumulateScaleFactors(scalingIndices, count, cumulativeScalingIndex);
    }
    if (cumulativeScalingIndex < 0 || cumulativeScalingIndex >= kScaleBufferCount)
        return BEAGLE_ERROR_OUT_OF_RANGE;

    setCudaDevice();
    const bool scalersLog = (kFlags & BEAGLE_FLAG_SCALERS_LOG) != 0;
    ScaleFactorBatch<Real> batch;
    for (int n = 0; n < count; n++) {
        if (scalingIndices[n] < 0 || scalingIndices[n] >= kScaleBufferCount)
            return BEAGLE_ERROR_OUT_OF_RANGE;
        batch.factors[batch.count++] = (const Real*) dScalingFactors[scalingIndices[n]];
        if (batch.count == 64 or n == count - 1) {
            cuda_accumulate_scale_factors(batch, kPaddedPatternCount, (Real*) dScalingFactors[cumulativeScalingIndex], scalersLog,
                                          subtract, stream);
            batch.count = 0;
        }
    }
    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
void BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::setCudaDevice()
{
    // GPUInterface uses the driver API and pops its context after each call, so without this
    // runtime-API work would go to device 0.
    auto status = cudaSetDevice(gpu->GetCudaDeviceOrdinal());
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("cudaSetDevice: ") + cudaGetErrorString(status));

    auto beagleStream = (cudaStream_t) gpu->GetCudaStream();
    if (cublasHandle and beagleStream != stream)
    {
        stream = beagleStream;
        CUBLAS_CHECK(cublasSetStream(cublasHandle, stream));
        if (cusparseSetStream(cusparseHandle, stream) != CUSPARSE_STATUS_SUCCESS)
            throw std::runtime_error("cusparseSetStream failed");
    }
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::upPartials(bool byPartition,
							const int *operations,
							int operationCount,
							int cumulativeScalingIndex)
{

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUActionImpl::upPartials\n");
#endif

    Real* cumulativeScalingBuffer = 0;
    if (cumulativeScalingIndex != BEAGLE_OP_NONE)
        cumulativeScalingBuffer = (Real*)dScalingFactors[cumulativeScalingIndex];

    std::vector<PartialsOperation> parsed(operationCount);
    for (int op = 0; op < operationCount; op++) {
        const int numOps = BEAGLE_OP_COUNT;

        const int destinationPartialIndex = operations[op * numOps];
        const int writeScalingIndex = operations[op * numOps + 1];
        const int readScalingIndex = operations[op * numOps + 2];
        const int firstChildPartialIndex = operations[op * numOps + 3];
        const int firstChildSubstitutionMatrixIndex = operations[op * numOps + 4];
        const int secondChildPartialIndex = operations[op * numOps + 5];
        const int secondChildSubstitutionMatrixIndex = operations[op * numOps + 6];

        int rescale = BEAGLE_OP_NONE;
        Real* scalingFactors = nullptr;

        if (kFlags & BEAGLE_FLAG_SCALING_AUTO) {
            int sIndex = destinationPartialIndex - kTipCount;

	    rescale = 2;
	    scalingFactors = (Real*)dScalingFactors[sIndex];
        } else if (kFlags & BEAGLE_FLAG_SCALING_ALWAYS) {
            rescale = 1;
            scalingFactors = (Real*)dScalingFactors[destinationPartialIndex - kTipCount];
        } else if ((kFlags & BEAGLE_FLAG_SCALING_MANUAL) && writeScalingIndex >= 0) {
            rescale = 1;
            scalingFactors = (Real*)dScalingFactors[writeScalingIndex];
        } else if ((kFlags & BEAGLE_FLAG_SCALING_MANUAL) && readScalingIndex >= 0) {
            rescale = 0;
            scalingFactors = (Real*)dScalingFactors[readScalingIndex];
        }

        PartialsOperation& operation = parsed[op];
        operation.destination = destinationPartialIndex;
        operation.children[0] = firstChildPartialIndex;
        operation.children[1] = secondChildPartialIndex;
        operation.edges[0] = firstChildSubstitutionMatrixIndex;
        operation.edges[1] = secondChildSubstitutionMatrixIndex;
        operation.scalingFactors = (rescale == 1) ? scalingFactors : nullptr;
    }

    for (const auto& batch: scheduleBatches(parsed)) {
        calcPartialsPartials(batch, cumulativeScalingBuffer);
        if (kFlags & BEAGLE_FLAG_SCALING_ALWAYS) {
            for (const auto& operation: batch) {
                int parScalingIndex = operation.destination - kTipCount;
                int child1ScalingIndex = operation.children[0] - kTipCount;
                int child2ScalingIndex = operation.children[1] - kTipCount;
                if (child1ScalingIndex >= 0 && child2ScalingIndex >= 0) {
                    int scalingIndices[2] = {child1ScalingIndex, child2ScalingIndex};
                    accumulateScaleFactors(scalingIndices, 2, parScalingIndex);
                } else if (child1ScalingIndex >= 0) {
                    int scalingIndices[1] = {child1ScalingIndex};
                    accumulateScaleFactors(scalingIndices, 1, parScalingIndex);
                } else if (child2ScalingIndex >= 0) {
                    int scalingIndices[1] = {child2ScalingIndex};
                    accumulateScaleFactors(scalingIndices, 1, parScalingIndex);
                }
            }
        }
    }

#ifdef BEAGLE_DEBUG_SYNCH
    gpu->SynchronizeHost();
#endif

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUActionImpl::upPartials\n");
#endif

    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
std::vector<std::vector<typename BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::PartialsOperation>>
BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::scheduleBatches(const std::vector<PartialsOperation>& operations) const
{
    // Each operation goes to the earliest level after every earlier operation that writes a buffer it reads, or
    // reads or writes a buffer (partials or scaling) it writes.  The operations of a level are independent.
    const int operationCount = (int) operations.size();
    std::vector<int> levels(operationCount);
    std::vector<int> writeLevel(kPartialsBufferCount, -1);
    std::vector<int> readLevel(kPartialsBufferCount, -1);
    std::map<Real*, int> scalingWriteLevel;
    int levelCount = 0;

    for (int op = 0; op < operationCount; op++) {
        const PartialsOperation& operation = operations[op];
        int level = std::max(writeLevel[operation.destination], readLevel[operation.destination]) + 1;
        for (int child: operation.children)
            level = std::max(level, writeLevel[child] + 1);
        if (operation.scalingFactors) {
            auto it = scalingWriteLevel.find(operation.scalingFactors);
            if (it != scalingWriteLevel.end())
                level = std::max(level, it->second + 1);
            scalingWriteLevel[operation.scalingFactors] = level;
        }
        writeLevel[operation.destination] = level;
        for (int child: operation.children)
            readLevel[child] = std::max(readLevel[child], level);
        levels[op] = level;
        levelCount = std::max(levelCount, level + 1);
    }

    std::vector<std::vector<PartialsOperation>> batches;
    for (int level = 0; level < levelCount; level++) {
        std::vector<PartialsOperation> batch;
        for (int op = 0; op < operationCount; op++) {
            if (levels[op] != level)
                continue;
            if ((int) batch.size() == maxBatchOperations) {
                batches.push_back(batch);
                batch.clear();
            }
            batch.push_back(operations[op]);
        }
        if (not batch.empty())
            batches.push_back(batch);
    }
    return batches;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::updatePrePartials(const int* operations,
                                                               int operationCount,
                                                               int cumulativeScalingIndex)
{
    setCudaDevice();
    return upPrePartials(false, operations, operationCount, cumulativeScalingIndex);
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::upPrePartials(bool byPartition,
							   const int *operations,
							   int operationCount,
							   int cumulativeScalingIndex)
{
    // From now on post-order keeps the children's actions for the siblings of later pre-order updates.
    cacheActions = true;

    Real* cumulativeScalingBuffer = nullptr;
    if (cumulativeScalingIndex != BEAGLE_OP_NONE)
        cumulativeScalingBuffer = (Real*)dScalingFactors[cumulativeScalingIndex];

    // [dest, writeScaling, readScaling, parent (pre-order), dest's matrix, sibling (post-order), sibling's matrix].
    // As in the CPU implementation, only a write scaling index rescales.
    std::vector<PartialsOperation> parsed(operationCount);
    for (int op = 0; op < operationCount; op++) {
        const int* x = operations + op * BEAGLE_OP_COUNT;
        PartialsOperation& operation = parsed[op];
        operation.destination = x[0];
        operation.children[0] = x[3];
        operation.children[1] = x[5];
        operation.edges[0] = x[4];
        operation.edges[1] = x[6];
        operation.scalingFactors = (x[1] >= 0) ? (Real*)dScalingFactors[x[1]] : nullptr;
        for (int buffer: {x[0], x[3], x[5]})
            if (buffer < 0 or buffer >= kPartialsBufferCount or dPartials[buffer] == 0)
                return BEAGLE_ERROR_OUT_OF_RANGE;
    }

    for (const auto& batch: scheduleBatches(parsed))
        calcPrePartials(batch, cumulativeScalingBuffer);

#ifdef BEAGLE_DEBUG_SYNCH
    gpu->SynchronizeHost();
#endif
    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
bool BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::actionCached(int buffer, int matrixIndex) const
{
    const ActionCacheEntry& entry = actionCache[buffer];
    return entry.valid and entry.data and entry.matrixIndex == matrixIndex and
           entry.partialsGeneration == partialsGenerations[buffer] and
           entry.matrixGeneration == matrixGenerations[matrixIndex] and
           entry.qGeneration == qGenerations[hEigenMaps[matrixIndex]];
}

BEAGLE_GPU_TEMPLATE
Real* BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::actionCacheSlot(int buffer, int matrixIndex)
{
    if (not cacheActions)
        return nullptr;
    ActionCacheEntry& entry = actionCache[buffer];
    if (not entry.data) {
        const size_t size = (size_t) kCategoryCount * kPaddedStateCount * kPaddedPatternCount;
        if (cudaMalloc((void**) &entry.data, size * sizeof(Real)) != cudaSuccess) {
            // Out of memory: pre-order recomputes the siblings' actions instead.
            cudaGetLastError();
            entry.data = nullptr;
            return nullptr;
        }
    }
    entry.valid = true;
    entry.matrixIndex = matrixIndex;
    entry.partialsGeneration = partialsGenerations[buffer];
    entry.matrixGeneration = matrixGenerations[matrixIndex];
    entry.qGeneration = qGenerations[hEigenMaps[matrixIndex]];
    return entry.data;
}

BEAGLE_GPU_TEMPLATE
void BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::estimateTransposedNorms(int eigenIndex)
{
    if (hdsTCurrent[eigenIndex])
        return;
    // The estimator for B^T: its transpose is B.
    const auto& dB = dBs[eigenIndex];
    auto norms = normEstimator.estimate(dBtOffsets[eigenIndex], dBtColumns[eigenIndex], dBtValues[eigenIndex],
                                        dB.offsets, dB.inner, dB.values, stream);
    const int pTop = getPMax() + 1;
    hdsT[eigenIndex].assign(pTop + 1, 1.0);
    hdsT[eigenIndex][1] = hBt1Norms[eigenIndex];
    for (int p = 2; p <= pTop; p++)
        hdsT[eigenIndex][p] = pow(norms[p], 1.0 / Real(p));
    hdsTCurrent[eigenIndex] = true;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::setRootPrePartials(const int* bufferIndices,
                                                                const int* stateFrequenciesIndices,
                                                                int count)
{
    if (count != 1)
        return BEAGLE_ERROR_NO_IMPLEMENTATION;
    const int buffer = bufferIndices[0];
    if (buffer < 0 or buffer >= kPartialsBufferCount or dPartials[buffer] == 0)
        return BEAGLE_ERROR_OUT_OF_RANGE;
    setCudaDevice();
    cuda_root_pre_partials((Real*) dPartials[buffer], (const Real*) dFrequencies[stateFrequenciesIndices[0]],
                           kPaddedStateCount, kCategoryCount, kPaddedPatternCount, kPatternCount, stream);
    partialsGenerations[buffer]++;
    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::setSparseDifferentialMatrix(int matrixIndex,
                                                                         const int* rowIndices,
                                                                         const int* colIndices,
                                                                         const double* values,
                                                                         int numNonZeros)
{
    if (matrixIndex < 0 or matrixIndex >= (int) derivativeMatrices.size())
        return BEAGLE_ERROR_OUT_OF_RANGE;
    setCudaDevice();

    // Duplicates are summed; every row gets a (possibly zero) diagonal entry, which is harmless.
    std::vector<Real> realValues(values, values + numNonZeros);
    HostCsr<Real> D;
    buildShiftedCsr(kPaddedStateCount, rowIndices, colIndices, realValues.data(), numNonZeros, Real(0), D);

    DerivativeMatrix& matrix = derivativeMatrices[matrixIndex];
    cudaDeviceDelete(matrix.offsets);
    cudaDeviceDelete(matrix.columns);
    cudaDeviceDelete(matrix.values);
    matrix.offsets = cudaDeviceNew<int>(kPaddedStateCount + 1);
    matrix.columns = cudaDeviceNew<int>(D.nonZeros());
    matrix.values = cudaDeviceNew<Real>(D.nonZeros());
    MemcpyHostToDevice(matrix.offsets, D.offsets.data(), kPaddedStateCount + 1);
    MemcpyHostToDevice(matrix.columns, D.columns.data(), D.nonZeros());
    MemcpyHostToDevice(matrix.values, D.values.data(), D.nonZeros());
    matrix.eigenIndex = -1;
    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::calculateEdgeDerivatives(const int* postBufferIndices,
                                                                      const int* preBufferIndices,
                                                                      const int* derivativeMatrixIndices,
                                                                      const int* categoryWeightsIndices,
                                                                      const int* categoryRatesIndices,
                                                                      const int* cumulativeScaleIndices,
                                                                      int count,
                                                                      double* outDerivatives,
                                                                      double* outSumDerivatives,
                                                                      double* outSumSquaredDerivatives)
{
    if (count <= 0)
        return BEAGLE_SUCCESS;
    if (kCategoryCount > kMaxDerivativeCategories)
        return BEAGLE_ERROR_NO_IMPLEMENTATION;
    for (int node = 0; node < count; node++) {
        const int d = derivativeMatrixIndices[node];
        if (d < 0 or d >= (int) derivativeMatrices.size() or
            (derivativeMatrices[d].eigenIndex < 0 and derivativeMatrices[d].offsets == nullptr))
            return BEAGLE_ERROR_OUT_OF_RANGE;
        for (int buffer: {postBufferIndices[node], preBufferIndices[node]})
            if (buffer < 0 or buffer >= kPartialsBufferCount or dPartials[buffer] == 0)
                return BEAGLE_ERROR_OUT_OF_RANGE;
    }
    setCudaDevice();

    if (not dDerivativeScratch)
        dDerivativeScratch = cudaDeviceNew<Real>(cuda_edge_derivative_scratch_size(kPaddedStateCount, kCategoryCount, kPaddedPatternCount));
    // derivatives (count * kPatternCount), then the sums (count), then the sums of squares (count)
    const size_t outputSize = (size_t) count * (kPatternCount + 2);
    if (outputSize > derivativeOutputSize) {
        cudaDeviceDelete(dDerivativeOutput);
        dDerivativeOutput = cudaDeviceNew<Real>(outputSize);
        derivativeOutputSize = outputSize;
    }
    Real* derivatives = dDerivativeOutput;
    Real* sums = derivatives + (size_t) count * kPatternCount;
    Real* sumSquares = sums + count;

    // As in the CPU implementation, the categories use rate set 0.
    EdgeDerivativeBatch<Real> batch;
    for (int c = 0; c < kCategoryCount; c++)
        batch.categoryMultipliers[c] = (Real) hCategoryRates[0][c];
    const Real* categoryWeights = (const Real*) dWeights[categoryWeightsIndices[0]];

    // Consecutive nodes with the same derivative matrix go in one batch.
    for (int first = 0; first < count;) {
        const int d = derivativeMatrixIndices[first];
        batch.count = 0;
        while (first + batch.count < count and batch.count < kMaxEdgeDerivativeNodes and
               derivativeMatrixIndices[first + batch.count] == d) {
            batch.post[batch.count] = (const Real*) dPartials[postBufferIndices[first + batch.count]];
            batch.pre[batch.count] = (const Real*) dPartials[preBufferIndices[first + batch.count]];
            batch.count++;
        }
        const DerivativeMatrix& matrix = derivativeMatrices[d];
        const int e = matrix.eigenIndex;
        // Q = B + mu I
        const int* offsets = (e >= 0) ? dBs[e].offsets : matrix.offsets;
        const int* columns = (e >= 0) ? dBs[e].inner : matrix.columns;
        const Real* values = (e >= 0) ? dBs[e].values : matrix.values;
        const Real shift = (e >= 0) ? hMuBs[e] : Real(0);
        cuda_edge_derivatives(batch, kPaddedStateCount, kCategoryCount, kPaddedPatternCount, kPatternCount,
                              offsets, columns, values, shift, categoryWeights, (const Real*) dPatternWeights,
                              dDerivativeScratch, derivatives + (size_t) first * kPatternCount, sums + first, sumSquares + first, stream);
        first += batch.count;
    }

    hDerivativeOutput.resize(outputSize);
    auto status = cudaMemcpyAsync(hDerivativeOutput.data(), dDerivativeOutput, outputSize * sizeof(Real), cudaMemcpyDeviceToHost, stream);
    if (status == cudaSuccess)
        status = cudaStreamSynchronize(stream);
    if (status != cudaSuccess)
        return BEAGLE_ERROR_GENERAL;

    const Real* h = hDerivativeOutput.data();
    if (outDerivatives)
        std::copy(h, h + (size_t) count * kPatternCount, outDerivatives);
    if (outSumDerivatives)
        std::copy(h + (size_t) count * kPatternCount, h + (size_t) count * kPatternCount + count, outSumDerivatives);
    if (outSumSquaredDerivatives)
        std::copy(h + (size_t) count * (kPatternCount + 1), h + outputSize, outSumSquaredDerivatives);
    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::calculateCrossProducts(const int* postBufferIndices,
                                                                    const int* preBufferIndices,
                                                                    const int* categoryRatesIndices,
                                                                    const int* categoryWeightsIndices,
                                                                    const double* edgeLengths,
                                                                    int count,
                                                                    double* outSumDerivatives,
                                                                    double* outSumSquaredDerivatives)
{
    if (count <= 0)
        return BEAGLE_SUCCESS;
    if (kCategoryCount > kMaxDerivativeCategories)
        return BEAGLE_ERROR_NO_IMPLEMENTATION;
    for (int node = 0; node < count; node++)
        for (int buffer: {postBufferIndices[node], preBufferIndices[node]})
            if (buffer < 0 or buffer >= kPartialsBufferCount or dPartials[buffer] == 0)
                return BEAGLE_ERROR_OUT_OF_RANGE;
    setCudaDevice();

    // The pattern: the union of the patterns of the eigen indices that are set (normally they are all the same).
    std::vector<int> set;
    for (int e = 0; e < kEigenDecompCount; e++)
        if (not hBs[e].offsets.empty())
            set.push_back(e);
    if (set.empty())
        return BEAGLE_ERROR_GENERAL;
    const int n = kPaddedStateCount;
    HostCsr<Real> pattern = hBs[set[0]];
    bool shared = true;
    for (int e: set)
        shared = shared and hBs[e].offsets == pattern.offsets and hBs[e].columns == pattern.columns;
    if (not shared) {
        std::vector<std::set<int>> rows(n);
        for (int e: set)
            for (int i = 0; i < n; i++)
                for (int k = hBs[e].offsets[i]; k < hBs[e].offsets[i + 1]; k++)
                    rows[i].insert(hBs[e].columns[k]);
        pattern.offsets.assign(n + 1, 0);
        pattern.columns.clear();
        for (int i = 0; i < n; i++) {
            pattern.columns.insert(pattern.columns.end(), rows[i].begin(), rows[i].end());
            pattern.offsets[i + 1] = (int) pattern.columns.size();
        }
    }
    const int nnz = (int) pattern.columns.size();
    if ((size_t) nnz > crossCapacity) {
        cudaDeviceDelete(dCrossColumns);
        cudaDeviceDelete(dCrossValues);
        dCrossColumns = cudaDeviceNew<int>(nnz);
        dCrossValues = cudaDeviceNew<Real>(nnz);
        crossCapacity = nnz;
    }
    if (not dCrossOffsets)
        dCrossOffsets = cudaDeviceNew<int>(n + 1);
    const int* offsets = shared ? dBs[set[0]].offsets : dCrossOffsets;
    const int* columns = shared ? dBs[set[0]].inner : dCrossColumns;
    if (not shared) {
        MemcpyHostToDevice(dCrossOffsets, pattern.offsets.data(), n + 1);
        MemcpyHostToDevice(dCrossColumns, pattern.columns.data(), nnz);
    }
    CHECK_CUDA(cudaMemsetAsync(dCrossValues, 0, nnz * sizeof(Real), stream));

    // Per node: patternScale[p] = patternWeight[p] / sum_c w_c pre^T post, then the products scaled by w_c r_c t.
    if (not dDerivativeScratch)
        dDerivativeScratch = cudaDeviceNew<Real>(cuda_edge_derivative_scratch_size(kPaddedStateCount, kCategoryCount, kPaddedPatternCount));
    const size_t scaleSize = (size_t) kMaxEdgeDerivativeNodes * kPaddedPatternCount;
    if (scaleSize > derivativeOutputSize) {
        cudaDeviceDelete(dDerivativeOutput);
        dDerivativeOutput = cudaDeviceNew<Real>(scaleSize);
        derivativeOutputSize = scaleSize;
    }
    CHECK_CUDA(cudaMemsetAsync(dDerivativeOutput, 0, scaleSize * sizeof(Real), stream));

    int ratesIndex = categoryRatesIndices ? categoryRatesIndices[0] : 0;
    if (ratesIndex < 0 or ratesIndex >= kEigenDecompCount or hCategoryRates[ratesIndex] == nullptr)
        ratesIndex = 0;
    std::vector<Real> weights(kCategoryCount);
    CHECK_CUDA(cudaMemcpyAsync(weights.data(), (const Real*) dWeights[categoryWeightsIndices[0]], kCategoryCount * sizeof(Real),
                               cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    EdgeDerivativeBatch<Real> batch;
    for (int first = 0; first < count; first += batch.count) {
        batch.count = std::min(count - first, kMaxEdgeDerivativeNodes);
        for (int z = 0; z < batch.count; z++) {
            batch.post[z] = (const Real*) dPartials[postBufferIndices[first + z]];
            batch.pre[z] = (const Real*) dPartials[preBufferIndices[first + z]];
        }
        // Scales are written with stride kPatternCount; kPatternCount == kPaddedPatternCount for this plugin.
        cuda_edge_derivatives(batch, kPaddedStateCount, kCategoryCount, kPaddedPatternCount, kPatternCount,
                              nullptr, nullptr, nullptr, Real(0), (const Real*) dWeights[categoryWeightsIndices[0]],
                              (const Real*) dPatternWeights, dDerivativeScratch, dDerivativeOutput, nullptr, nullptr, stream);
        for (int z = 0; z < batch.count; z++) {
            CategoryMultipliers<Real> multipliers;
            for (int c = 0; c < kCategoryCount; c++)
                multipliers.values[c] = weights[c] * (Real) (hCategoryRates[ratesIndex][c] * edgeLengths[first + z]);
            cuda_cross_products(batch.pre[z], batch.post[z], kPaddedStateCount, kCategoryCount, kPaddedPatternCount, offsets, columns,
                                multipliers, dDerivativeOutput + (size_t) z * kPatternCount, dCrossValues, stream);
        }
    }

    std::vector<Real> values(nnz);
    CHECK_CUDA(cudaMemcpyAsync(values.data(), dCrossValues, nnz * sizeof(Real), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    for (int i = 0; i < kStateCount; i++)
        for (int k = pattern.offsets[i]; k < pattern.offsets[i + 1]; k++)
            if (pattern.columns[k] < kStateCount)
                outSumDerivatives[(size_t) i * kStateCount + pattern.columns[k]] += values[k];
    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::PrintfDeviceVector(int* dPtr,
                                                                int length,
                                                                double checkValue,
                                                                int *signal,
                                                                Real r) {
    int* hPtr = (int*) malloc(sizeof(int) * length);

//    MemcpyDeviceToHost(hPtr, dPtr, sizeof(Real) * length);
    CHECK_CUDA(cudaMemcpy(hPtr, dPtr, sizeof(int) * length, cudaMemcpyDeviceToHost))
    printfInt(hPtr, length);

    if (checkValue != -1) {
        double sum = 0;
        for(int i=0; i<length; i++) {
            sum += hPtr[i];
            if( (hPtr[i] > checkValue) && (hPtr[i]-checkValue > 1.0E-4)) {
                fprintf(stderr,"Check value exception!  (%d) %2.5d > %2.5e (diff = %2.5e)\n",
                        i,hPtr[i],checkValue, (hPtr[i]-checkValue));
                if( signal != 0 )
                    *signal = 1;
            }
            if (hPtr[i] != hPtr[i]) {
                fprintf(stderr,"NaN found!\n");
                if( signal != 0 )
                    *signal = 1;
            }
        }
        if (sum == 0) {
            fprintf(stderr,"Zero-sum vector!\n");
            if( signal != 0 )
                *signal = 1;
        }
    }
    free(hPtr);
    return 0;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::PrintfDeviceVector(Real* dPtr,
                                                                 int length,
                                                                 double checkValue,
                                                                 int *signal,
                                                                 Real r) {
    Real* hPtr = (Real*) malloc(sizeof(Real) * length);

//    MemcpyDeviceToHost(hPtr, dPtr, sizeof(Real) * length);
    CHECK_CUDA(cudaMemcpy(hPtr, dPtr, sizeof(Real) * length, cudaMemcpyDeviceToHost))
    printfVector(hPtr, length);

    if (checkValue != -1) {
        double sum = 0;
        for(int i=0; i<length; i++) {
            sum += hPtr[i];
            if( (hPtr[i] > checkValue) && (hPtr[i]-checkValue > 1.0E-4)) {
                fprintf(stderr,"Check value exception!  (%d) %2.5e > %2.5e (diff = %2.5e)\n",
                        i,hPtr[i],checkValue, (hPtr[i]-checkValue));
                if( signal != 0 )
                    *signal = 1;
            }
            if (hPtr[i] != hPtr[i]) {
                fprintf(stderr,"NaN found!\n");
                if( signal != 0 )
                    *signal = 1;
            }
        }
        if (sum == 0) {
            fprintf(stderr,"Zero-sum vector!\n");
            if( signal != 0 )
                *signal = 1;
        }
    }
    free(hPtr);
    return 0;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::PrintfDeviceVector(cusparseDnMatDescr_t dPtr,
                                                                int length,
                                                                double checkValue,
                                                                int *signal,
                                                                Real r) {
    Real* hPtr = (Real*) malloc(sizeof(Real) * length);

//    MemcpyDeviceToHost(hPtr, dPtr, sizeof(Real) * length);
//    CHECK_CUDA(cudaMemcpy(hPtr, dPtr, sizeof(Real) * length, cudaMemcpyDeviceToHost))
    CHECK_CUSPARSE(cusparseConstDnMatGetValues(dPtr, reinterpret_cast<const void **>(hPtr)))
    printfVector(hPtr, length);

    if (checkValue != -1) {
        double sum = 0;
        for(int i=0; i<length; i++) {
            sum += hPtr[i];
            if( (hPtr[i] > checkValue) && (hPtr[i]-checkValue > 1.0E-4)) {
                fprintf(stderr,"Check value exception!  (%d) %2.5e > %2.5e (diff = %2.5e)\n",
                        i,hPtr[i],checkValue, (hPtr[i]-checkValue));
                if( signal != 0 )
                    *signal = 1;
            }
            if (hPtr[i] != hPtr[i]) {
                fprintf(stderr,"NaN found!\n");
                if( signal != 0 )
                    *signal = 1;
            }
        }
        if (sum == 0) {
            fprintf(stderr,"Zero-sum vector!\n");
            if( signal != 0 )
                *signal = 1;
        }
    }
    free(hPtr);
}

BEAGLE_GPU_TEMPLATE
void BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::setActionScratch(ActionGroup& group, size_t g)
{
    // With few patterns, the groups' blocks of the 3 scratch matrices are side by side (coalescing the gathers
    // across groups); otherwise each group uses 2 contiguous blocks.  See computeActions( ).
    const size_t matrixSize = (size_t) kPaddedStateCount * kPaddedPatternCount;
    const size_t scratchMatrixSize = (size_t) 2 * kCategoryCount * maxBatchOperations * matrixSize;
    const bool interleaved = interleavedScratch();
    group.scratchStride = interleaved ? (int) (2 * kCategoryCount * maxBatchOperations * kPaddedPatternCount) : kPaddedPatternCount;
    group.scratchCount = interleaved ? 3 : 2;
    for (int k = 0; k < group.scratchCount; k++)
        group.scratch[k] = dActionScratch + k * scratchMatrixSize + g * (interleaved ? kPaddedPatternCount : matrixSize);
}

BEAGLE_GPU_TEMPLATE
void BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::calcPartialsPartials(const std::vector<PartialsOperation>& operations,
                                                                   Real* cumulativeScalingBuffer) {
    const size_t matrixSize = (size_t) kPaddedStateCount * kPaddedPatternCount;

    // Each child's action goes to the action cache once pre-order is in use (unless the same child appears twice in
    // this batch), otherwise to dActionResults.
    std::vector<ActionGroup> groups;
    std::vector<const Real*> actions(2 * operations.size());
    std::set<int> cachedChildren;
    for (size_t op = 0; op < operations.size(); op++) {
        for (int child = 0; child < 2; child++) {
            const int buffer = operations[op].children[child];
            const int edgeIndex = operations[op].edges[child];
            Real* out = nullptr;
            if (cachedChildren.insert(buffer).second)
                out = actionCacheSlot(buffer, edgeIndex);
            if (not out)
                out = dActionResults + (op * 2 + child) * kCategoryCount * matrixSize;
            actions[op * 2 + child] = out;
            for (int category = 0; category < kCategoryCount; category++) {
                ActionGroup group;
                group.in = (const Real*) dPartials[buffer] + category * matrixSize;
                group.out = out + category * matrixSize;
                setActionScratch(group, (op * 2 + child) * kCategoryCount + category);
                group.eigenIndex = hEigenMaps[edgeIndex];
                group.multiplier = hEdgeMultipliers[edgeIndex * kCategoryCount + category];
                group.transpose = false;
                groups.push_back(group);
            }
        }
    }
    computeActions(groups);

    // parent = left * right for all categories, then (optionally) rescale each pattern.
    bool scalers_log = (kFlags & BEAGLE_FLAG_SCALERS_LOG)?true:false;
    PartialsProductBatch<Real> products;
    for (size_t op = 0; op < operations.size(); op++) {
        auto& product = products.products[products.count++];
        product.left = actions[op * 2];
        product.right = actions[op * 2 + 1];
        product.parent = (Real*)dPartials[operations[op].destination];
        product.scalingFactors = operations[op].scalingFactors;
        partialsGenerations[operations[op].destination]++;
    }
    cuda_partials_products(products, kCategoryCount * kPaddedStateCount, kPaddedPatternCount,
                           cumulativeScalingBuffer, scalers_log, dReductionScratch, stream);
}

BEAGLE_GPU_TEMPLATE
void BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::calcPrePartials(const std::vector<PartialsOperation>& operations,
                                                              Real* cumulativeScalingBuffer) {
    const size_t matrixSize = (size_t) kPaddedStateCount * kPaddedPatternCount;
    const size_t partialsSize = kCategoryCount * matrixSize;
    // Per operation, two blocks of dActionResults: block 0 = the sibling's action o the parent, block 1 = the
    // sibling's action when it is not cached, then the result before rescaling.
    auto block = [&](size_t op, int k) { return dActionResults + (op * 2 + k) * partialsSize; };

    // 1. The siblings' actions exp(t_sibling Q) sibling, from the cache when it is current.
    std::vector<ActionGroup> groups;
    std::vector<const Real*> siblingActions(operations.size());
    for (size_t op = 0; op < operations.size(); op++) {
        const int sibling = operations[op].children[1];
        const int edgeIndex = operations[op].edges[1];
        if (actionCached(sibling, edgeIndex)) {
            siblingActions[op] = actionCache[sibling].data;
            continue;
        }
        siblingActions[op] = block(op, 1);
        for (int category = 0; category < kCategoryCount; category++) {
            ActionGroup group;
            group.in = (const Real*) dPartials[sibling] + category * matrixSize;
            group.out = block(op, 1) + category * matrixSize;
            setActionScratch(group, (op * 2 + 1) * kCategoryCount + category);
            group.eigenIndex = hEigenMaps[edgeIndex];
            group.multiplier = hEdgeMultipliers[edgeIndex * kCategoryCount + category];
            group.transpose = false;
            groups.push_back(group);
        }
    }
    computeActions(groups);

    // 2. sibling's action o parent's pre-order partials.
    PartialsProductBatch<Real> products;
    for (size_t op = 0; op < operations.size(); op++) {
        auto& product = products.products[products.count++];
        product.left = siblingActions[op];
        product.right = (const Real*) dPartials[operations[op].children[0]];
        product.parent = block(op, 0);
        product.scalingFactors = nullptr;
    }
    cuda_partials_products(products, kCategoryCount * kPaddedStateCount, kPaddedPatternCount, nullptr, false,
                           dReductionScratch, stream);

    // 3. exp(t Q^T) along the destination's edge; straight into the destination unless it is rescaled.
    groups.clear();
    for (size_t op = 0; op < operations.size(); op++) {
        const int edgeIndex = operations[op].edges[0];
        estimateTransposedNorms(hEigenMaps[edgeIndex]);
        Real* out = operations[op].scalingFactors ? block(op, 1) : (Real*) dPartials[operations[op].destination];
        for (int category = 0; category < kCategoryCount; category++) {
            ActionGroup group;
            group.in = block(op, 0) + category * matrixSize;
            group.out = out + category * matrixSize;
            setActionScratch(group, (op * 2) * kCategoryCount + category);
            group.eigenIndex = hEigenMaps[edgeIndex];
            group.multiplier = hEdgeMultipliers[edgeIndex * kCategoryCount + category];
            group.transpose = true;
            groups.push_back(group);
        }
    }
    computeActions(groups);

    // 4. Rescale where requested.
    bool scalers_log = (kFlags & BEAGLE_FLAG_SCALERS_LOG)?true:false;
    products.count = 0;
    for (size_t op = 0; op < operations.size(); op++) {
        partialsGenerations[operations[op].destination]++;
        if (not operations[op].scalingFactors)
            continue;
        auto& product = products.products[products.count++];
        product.left = block(op, 1);
        product.right = nullptr;
        product.parent = (Real*)dPartials[operations[op].destination];
        product.scalingFactors = operations[op].scalingFactors;
    }
    cuda_partials_products(products, kCategoryCount * kPaddedStateCount, kPaddedPatternCount,
                           cumulativeScalingBuffer, scalers_log, dReductionScratch, stream);
}

BEAGLE_GPU_TEMPLATE
void BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::computeActions(const std::vector<ActionGroup>& groups)
{
    const size_t matrixSize = (size_t) kPaddedStateCount * kPaddedPatternCount;

    // exp(A) b = eta^s T_m(A/s)^s b, with A = multiplier * B (B = Q - mu I) and eta = exp(multiplier * mu / s).
    // Each of the s steps evaluates b <- eta T_m(A/s) b by Horner's rule:
    //   w = b;  for j = m .. 1:  w = b + A/(s j) w;  then b = eta w.
    // A term reads w and b and writes a third matrix; the outputs of a step alternate between `result` (written
    // by the last term, j = 1) and `other`.
    // - With 3 scratch matrices, result and other are the two that are not b, except that the last term of the
    //   last step writes out.  So the gathers read only scratch, whose groups are side by side.
    // - With 2, out is also work space: the last step's result is out, and earlier steps write their results
    //   to scratch, so the last step's b is never out.  This keeps the working set smaller (L2).
    struct Schedule
    {
        const ActionGroup* group;
        int nCols;
        int m;
        int s;
        Real eta;
        int step;
        int term;
        const Real* b;
        const Real* w;
        Real* result;
        Real* other;

        int stride(const Real* matrix) const
        {
            for (int k = 0; k < group->scratchCount; k++)
                if (matrix == group->scratch[k])
                    return group->scratchStride;
            return nCols;
        }

        void startStep()
        {
            Real* free[2] = {nullptr, nullptr};
            int n = 0;
            for (int k = 0; k < group->scratchCount; k++)
                if (group->scratch[k] != b and n < 2)
                    free[n++] = group->scratch[k];
            if (group->scratchCount == 3) {
                result = free[0];
                other = free[1];
            } else if (step == s - 1) {
                result = group->out;
                other = free[0];
            } else {
                result = free[0];
                other = (n == 1) ? group->out : free[1];
            }
            w = b;
            term = m;
        }

        Real* output() const
        {
            if (group->scratchCount == 3 and step == s - 1)
                return (term == 1) ? group->out : (term % 2 == 0) ? result : other;
            return (term % 2 == 1) ? result : other;
        }
    };

    std::vector<bool> done(groups.size(), false);
    for (size_t first = 0; first < groups.size(); first++) {
        if (done[first])
            continue;

        // All the groups that share this group's matrix (B or B^T) run in lockstep: one launch per term.
        const int eigenIndex = groups[first].eigenIndex;
        const bool transpose = groups[first].transpose;
        const auto& B = dBs[eigenIndex];
        const int* offsets = transpose ? dBtOffsets[eigenIndex] : B.offsets;
        const int* columns = transpose ? dBtColumns[eigenIndex] : B.inner;
        const Real* values = transpose ? dBtValues[eigenIndex] : B.values;
        std::vector<Schedule> schedules;
        for (size_t i = first; i < groups.size(); i++) {
            if (done[i] or groups[i].eigenIndex != eigenIndex or groups[i].transpose != transpose)
                continue;
            done[i] = true;
            const ActionGroup& group = groups[i];
            auto [m, s] = getStatistics2(1.0, kPaddedPatternCount, group.multiplier, eigenIndex, transpose);
            if (m == 0) {
                auto status = cudaMemcpyAsync(group.out, group.in, matrixSize * sizeof(Real), cudaMemcpyDeviceToDevice, stream);
                if (status != cudaSuccess)
                    throw std::runtime_error(std::string("cudaMemcpyAsync: ") + cudaGetErrorString(status));
                continue;
            }
            Schedule schedule;
            schedule.group = &group;
            schedule.nCols = kPaddedPatternCount;
            schedule.m = m;
            schedule.s = s;
            schedule.eta = exp(hMuBs[eigenIndex] * group.multiplier / s);
            schedule.step = 0;
            schedule.b = group.in;
            schedule.startStep();
            schedules.push_back(schedule);

#ifdef BEAGLE_DEBUG_FLOW
            std::cerr << "action: m = " << m << "  s = " << s << "  multiplier = " << group.multiplier << "  eigenIndex = " << eigenIndex << std::endl;
#endif
        }

        for (bool active = true; active;) {
            active = false;
            TaylorTermBatch<Real> batch;
            for (auto& schedule: schedules) {
                if (schedule.step == schedule.s)
                    continue;
                active = true;

                const int j = schedule.term;
                Real* out = schedule.output();
                auto& term = batch.groups[batch.count++];
                term.x = schedule.w;
                term.b = schedule.b;
                term.out = out;
                term.alpha = schedule.group->multiplier / ((Real) schedule.s * j);
                term.scale = (j == 1) ? schedule.eta : 1;
                term.ldx = schedule.stride(term.x);
                term.ldb = schedule.stride(term.b);
                term.ldOut = schedule.stride(term.out);

                if (batch.count == kMaxTaylorGroups) {
                    cuda_taylor_horner_terms(kPaddedStateCount, kPaddedPatternCount, offsets, columns, values, batch, stream);
                    batch.count = 0;
                }

                schedule.w = out;
                if (--schedule.term == 0) {
                    schedule.b = out;
                    if (++schedule.step < schedule.s)
                        schedule.startStep();
                }
            }
            cuda_taylor_horner_terms(kPaddedStateCount, kPaddedPatternCount, offsets, columns, values, batch, stream);
        }
    }
}

BEAGLE_GPU_TEMPLATE
double BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::getPMax() const
{
    return floor(0.5 + 0.5 * sqrt(5.0 + 4.0 * mMax));
}

BEAGLE_GPU_TEMPLATE
double BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::getDValue(int p, int eigenIndex, bool transpose) const
{
    const auto& d = transpose ? hdsT[eigenIndex] : hds[eigenIndex];
    assert(p >= 0 and p < d.size());

    return d[p];
}

std::ostream& showScalingInfo(std::ostream& o, std::uint64_t kFlags, const int* cumulativeScaleIndices, int kScaleBufferSize)
{
    int scale = 1;
    if (kFlags & BEAGLE_FLAG_SCALING_AUTO)
	;
    else if (kFlags & BEAGLE_FLAG_SCALING_ALWAYS)
	;
    else if (cumulativeScaleIndices[0] != BEAGLE_OP_NONE)
	;
    else
	scale = 0;

    o<<"scale = "<<scale
     <<"  scaling_auto = "<<bool(kFlags & BEAGLE_FLAG_SCALING_AUTO)
     <<"  scaling always = "<<bool(kFlags & BEAGLE_FLAG_SCALING_ALWAYS)
     <<"  scaling_dynamic = "<<bool(kFlags & BEAGLE_FLAG_SCALING_DYNAMIC)
     <<"  scalers = "<<bool(kFlags & BEAGLE_FLAG_SCALERS_LOG)
     <<"  scalers_raw = "<<bool(kFlags & BEAGLE_FLAG_SCALERS_RAW)
     <<"  cumulative_scale_indices = "<<cumulativeScaleIndices[0]
     <<"  kScaleBufferSize = "<<kScaleBufferSize
     <<"\n";

    return o;
}

template <typename Real>
void gemv(cublasHandle_t handle, cublasOperation_t op, int size1, int size2, Real alpha, const Real* mat, int lda, const Real* vec, Real beta, Real* out)
{
    cublasStatus_t status;

    if constexpr (std::is_same<Real, float>::value) {
	status = cublasSgemv(handle, op,
			     size1, size2,
			     &alpha,
			     mat, size1,  //leading dimension
			     vec, 1,
			     &beta,
			     out, 1);
    }
    else
    {
	status = cublasDgemv(handle, op,
			     size1, size2,
			     &alpha,
			     mat, size1,  //leading dimension
			     vec, 1,
			     &beta,
			     out, 1);
    }
    if (status != CUBLAS_STATUS_SUCCESS)
    {
	std::cerr<<"cublas error "<<status<<" in cublas<t>gemv( ).";
	exit(1);
    }
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::calculateRootLogLikelihoods(const int* bufferIndices,
                                                                         const int* categoryWeightsIndices,
                                                                         const int* stateFrequenciesIndices,
                                                                         const int* cumulativeScaleIndices,
                                                                         int count,
                                                                         double* outSumLogLikelihood) {

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUActionImpl::calculateRootLogLikelihoods\n");
#endif

    setCudaDevice();

    int returnCode = BEAGLE_SUCCESS;

    const int rootNodeIndex = bufferIndices[0];
    const int categoryWeightsIndex = categoryWeightsIndices[0];
    const int stateFrequenciesIndex = stateFrequenciesIndices[0];

    const Real* dCumulativeScalingFactor = nullptr;
    bool scale = 1;
    if (kFlags & BEAGLE_FLAG_SCALING_AUTO)
        dCumulativeScalingFactor = (const Real*)dAccumulatedScalingFactors;
    else if (kFlags & BEAGLE_FLAG_SCALING_ALWAYS)
        dCumulativeScalingFactor = (const Real*)dScalingFactors[bufferIndices[0] - kTipCount];
    else if (cumulativeScaleIndices[0] != BEAGLE_OP_NONE)
        dCumulativeScalingFactor = (const Real*)dScalingFactors[cumulativeScaleIndices[0]];
    else
        scale = 0;

#ifdef BEAGLE_DEBUG_VALUES
    std::cerr<<"root partials = "<<asDeviceVec((Real*)dPartials[rootNodeIndex], kPaddedPatternCount * kPaddedStateCount * kCategoryCount)<<"\n";
#endif

    if (kFlags & BEAGLE_FLAG_SCALING_AUTO)
    {
	// scaling factor per pattern*category -- see BeagleGPUImpl<>::createInstance
	// scaling factor is char?? -- see BeagleGPUImpl<>::createInstance
	// scaling factor is int??  -- see KernelIntegrateLikelihoodsAutoScaling
	// See kernelIntegrateLikelihoodsAutoScaling in KernelsX.cu
	std::cerr<<"BeagleGPUActionImpl< >::calculateRootLogLikelihoods -- FLAG_SCALING_AUTO not implemented!";
	std::abort();
    }

    Real* siteLogLikelihoods = (Real*)dIntegrationTmp;
    Real* rootPartials = (Real*)dPartials[rootNodeIndex];
    Real* categoryWeights = (Real*)dWeights[categoryWeightsIndex];
    Real* stateFrequencies = (Real*)dFrequencies[stateFrequenciesIndex]; // these really should be category-dependent

    // siteLogLikelihoods[pattern] = log( sum_{category,state} likelihood(category,state,pattern) * categoryWeights[category] * statesFrequencies[state] )
    //                               + scaling factor
    // dSumLogLikelihood[0] = sum_{pattern} patternWeights[pattern] * siteLogLikelihoods[pattern]
    cuda_root_log_likelihoods(rootPartials, categoryWeights, stateFrequencies, scale ? dCumulativeScalingFactor : nullptr,
                              (Real*)dPatternWeights, kPaddedStateCount, kPaddedPatternCount, kPatternCount, kCategoryCount,
                              dReductionScratch, siteLogLikelihoods, (Real*)dSumLogLikelihood, stream);

#ifdef BEAGLE_DEBUG_VALUES
    std::cerr<<"site log likelihoods = "<<asDeviceVec((Real*)dIntegrationTmp, kPatternCount)<<"\n";
#endif

    if (kFlags & BEAGLE_FLAG_COMPUTATION_SYNCH) {
        CHECK_CUDA(cudaMemcpyAsync(hLogLikelihoodsCache, (Real*)dSumLogLikelihood, sizeof(Real), cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));

        *outSumLogLikelihood = hLogLikelihoodsCache[0];
        if (*outSumLogLikelihood != *outSumLogLikelihood)
            returnCode = BEAGLE_ERROR_FLOATING_POINT;
    }

#ifdef BEAGLE_DEBUG_VALUES
    Real r = 0;
    fprintf(stderr, "parent = \n");
    gpu->PrintfDeviceVector(dIntegrationTmp, kPatternCount, r);
#endif


#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUActionImpl::calculateRootLogLikelihoods\n");
#endif

    return returnCode;
}


BEAGLE_GPU_TEMPLATE
std::tuple<int,int> BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::getStatistics2(double t, int nCol,
                                                              double edgeMultiplier,
                                                              int eigenIndex, bool transpose) const {
    assert( t >= 0 );
    assert( nCol >= 0);
    assert( edgeMultiplier >= 0 );
    assert( eigenIndex >= 0);

    const double norm1 = transpose ? hBt1Norms[eigenIndex] : hB1Norms[eigenIndex];

    if (t * norm1 == 0.0)
        return {0, 1};

    int bestM = INT_MAX;
    double bestS = INT_MAX;  // Not all the values of s can fit in a 32-bit int.

    const double pMax = getPMax();
    // pMax is the largest positive integer such that p*(p-1) <= mMax + 1

    // 1. The best (m,s) based only on ||B||_1 (in some sense the result for p=1).
    for (auto& [thisM, thetaM]: thetaConstants) {
        const double thisS = ceil(norm1 * edgeMultiplier / thetaM);
        if (bestM == INT_MAX || ((double) thisM) * thisS < bestM * bestS) {
            bestS = thisS;
            bestM = thisM;
        }
    }

    // 2. Improve on it with the d_p.  Condition 3.13 of Al-Mohy and Higham only decides whether computing
    //    the d_p is worth it; setSparseMatrix( ) has already computed all of them, so always use them
    //    (as the CPU implementation does once they are available).
    for (int p = 2; p <= pMax; p++) {
        for (int thisM = p * (p - 1) - 1; thisM < mMax + 1; thisM++) {
            auto it = thetaConstants.find(thisM);
            if (it != thetaConstants.end()) {
                // equation 3.7 in Al-Mohy and Higham
                const double dValueP = getDValue(p, eigenIndex, transpose);
                const double dValuePPlusOne = getDValue(p + 1, eigenIndex, transpose);
                const double alpha = std::max(dValueP, dValuePPlusOne) * edgeMultiplier;
                // part of equation 3.10
                const double thisS = ceil(alpha / thetaConstants.at(thisM));
                if (bestM == INT_MAX || ((double) thisM) * thisS < bestM * bestS) {
                    bestS = thisS;
                    bestM = thisM;
                }
            }
        }
    }
    bestS = std::max(std::min<double>(bestS, INT_MAX), 1.0);
    assert( bestS >= 1 );
    assert( bestS <= INT_MAX );

    int m = bestM;
    int s = (int) bestS;

    assert(m >= 0);
    assert(s >= 1);

    return {m,s};
}


BEAGLE_GPU_TEMPLATE
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::setSparseMatrix(int matrixIndex,
                                                             const int* rowIndices,
                                                             const int* colIndices,
                                                             const Real* values,
                                                             int numNonZeros)
{
    setCudaDevice();

    // mu = trace(Q) / kStateCount (duplicate triplets are summed)
    Real mu_B = 0.0;
    for (int k = 0; k < numNonZeros; k++) {
        if (rowIndices[k] == colIndices[k] and rowIndices[k] < kStateCount)
            mu_B += values[k];
    }
    mu_B /= (Real) kStateCount;

    hMuBs[matrixIndex] = mu_B;
    auto& builder = csrBuilders[matrixIndex];
    const bool structureChanged = builder.build(kPaddedStateCount, rowIndices, colIndices, values, numNonZeros, mu_B, hBs[matrixIndex]);
    hB1Norms[matrixIndex] = normP1(kPaddedStateCount, hBs[matrixIndex]);
    hBt1Norms[matrixIndex] = normPInf(kPaddedStateCount, hBs[matrixIndex]);
    hdsTCurrent[matrixIndex] = false;
    qGenerations[matrixIndex]++;
    derivativeMatrices[matrixIndex].eigenIndex = matrixIndex;

#ifdef BEAGLE_DEBUG_FLOW
    std::cerr<<"B "<<matrixIndex<<": mu = "<<mu_B<<"  ||B||_1 = "<<hB1Norms[matrixIndex]<<"\noffsets = \n";
    for (int x: hBs[matrixIndex].offsets) std::cerr<<x<<", ";
    std::cerr<<"\ncolumns = \n";
    for (int x: hBs[matrixIndex].columns) std::cerr<<x<<", ";
    std::cerr<<"\nvalues = \n";
    for (Real x: hBs[matrixIndex].values) std::cerr<<x<<", ";
    std::cerr<<"\n";
#endif

    // With an unchanged structure only the values of B go to the device; B^T's values are gathered from them.
    const int currentNNZ = hBs[matrixIndex].nonZeros();
    if (structureChanged) {
        if (currentCacheNNZs[matrixIndex] != currentNNZ) {
            currentCacheNNZs[matrixIndex] = currentNNZ;
            cudaDeviceDelete(dBsCsrColumnsCache[matrixIndex]);
            cudaDeviceDelete(dBsCsrValuesCache[matrixIndex]);
            dBsCsrColumnsCache[matrixIndex] = cudaDeviceNew<int>(currentNNZ);
            dBsCsrValuesCache[matrixIndex] = cudaDeviceNew<Real>(currentNNZ);
        }
        cudaDeviceDelete(dBtColumns[matrixIndex]);
        cudaDeviceDelete(dBtSlots[matrixIndex]);
        cudaDeviceDelete(dBtValues[matrixIndex]);
        dBtColumns[matrixIndex] = cudaDeviceNew<int>(currentNNZ);
        dBtSlots[matrixIndex] = cudaDeviceNew<int>(currentNNZ);
        dBtValues[matrixIndex] = cudaDeviceNew<Real>(currentNNZ);

        MemcpyHostToDevice(dBsCsrOffsetsCache[matrixIndex], hBs[matrixIndex].offsets.data(), kPaddedStateCount + 1);
        MemcpyHostToDevice(dBsCsrColumnsCache[matrixIndex], hBs[matrixIndex].columns.data(), currentNNZ);
        MemcpyHostToDevice(dBtOffsets[matrixIndex], builder.transposeOffsets.data(), kPaddedStateCount + 1);
        MemcpyHostToDevice(dBtColumns[matrixIndex], builder.transposeColumns.data(), currentNNZ);
        MemcpyHostToDevice(dBtSlots[matrixIndex], builder.transposeSlot.data(), currentNNZ);

        dBs[matrixIndex] = SpMatrixDevice<Real>(cublasHandle, cusparseHandle,
                                                kPaddedStateCount, kPaddedStateCount,
                                                currentNNZ,
                                                dBsCsrValuesCache[matrixIndex],
                                                dBsCsrColumnsCache[matrixIndex],
                                                dBsCsrOffsetsCache[matrixIndex],
                                                sparseFormat::csr);
    }
    MemcpyHostToDevice(dBsCsrValuesCache[matrixIndex], hBs[matrixIndex].values.data(), currentNNZ);
    cuda_gather(dBsCsrValuesCache[matrixIndex], dBtSlots[matrixIndex], currentNNZ, dBtValues[matrixIndex], stream);

#ifdef BEAGLE_DEBUG_FLOW

    std::cerr<<"\ndBsCsrOffsetsCache for "<< matrixIndex <<"="<<std::endl;
    PrintfDeviceVector(dBsCsrOffsetsCache[matrixIndex], kPaddedStateCount + 1, -1, 0, 0);

    std::cerr<<"dBsCsrColumnsCache for "<< matrixIndex <<"="<<std::endl;
    PrintfDeviceVector(dBsCsrColumnsCache[matrixIndex], currentNNZ, -1, 0, 0);

    std::cerr<<"currentNNZ ="<< currentNNZ <<std::endl;

    std::cerr<<"dBsCsrValuesCache for "<< matrixIndex <<"="<<std::endl;
    PrintfDeviceVector(dBsCsrValuesCache[matrixIndex], currentNNZ, -1, 0, 0);
#endif

    auto& dB = dBs[matrixIndex];

    // equation 3.7 in Al-Mohy and Higham: d_p = ||B^p||_1^(1/p).  d_1 is known exactly; d_0 is not used.
    const int pTop = getPMax() + 1;
    auto norms = normEstimator.estimate(dB.offsets, dB.inner, dB.values,
                                        dBtOffsets[matrixIndex], dBtColumns[matrixIndex], dBtValues[matrixIndex], stream);
    hds[matrixIndex].assign(pTop + 1, 1.0);
    hds[matrixIndex][1] = hB1Norms[matrixIndex];
    for (int p = 2; p <= pTop; p++)
        hds[matrixIndex][p] = pow(norms[p], 1.0 / Real(p));

#ifdef BEAGLE_DEBUG_NORMEST
    // Compare with the CPU estimator and, for small S, the exact ||B^p||_1.
    auto& hB = hBs[matrixIndex];
    SpMatrix<Real> sparseB = Eigen::Map<const SpMatrix<Real>>(kPaddedStateCount, kPaddedStateCount, hB.nonZeros(),
                                                               hB.offsets.data(), hB.columns.data(), hB.values.data());
    DnMatrix<Real> denseB = sparseB;
    DnMatrix<Real> denseBp = denseB;
    for (int p = 2; p <= pTop; p++)
    {
        Real approx_norm_CPU = normest1( sparseB, p, 5);
        double exact_norm = -1;
        if (kPaddedStateCount <= 4096)
        {
            denseBp = denseB * denseBp;
            exact_norm = normP1(denseBp);
        }
        std::fprintf(stderr, "normest matrix=%d S=%d p=%d gpu=%.6e cpu=%.6e exact=%.6e gpu/exact=%.4f cpu/exact=%.4f\n",
                     matrixIndex, kPaddedStateCount, p, (double)norms[p], (double)approx_norm_CPU, exact_norm,
                     norms[p] / exact_norm, approx_norm_CPU / exact_norm);
    }
#endif

//#ifdef BEAGLE_DEBUG_FLOW
//    std::cerr<<"Setting host matrix: "<<matrixIndex<<std::endl<<hInstantaneousMatrices[matrixIndex]<<std::endl
//    <<std::endl<<"Setting device matrix: " << matrixIndex << std::endl << dInstantaneousMatrices[matrixIndex]<<std::endl;
//#endif
    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::updateTransitionMatrices(int eigenIndex,
								      const int* probabilityIndices,
								      const int* firstDerivativeIndices,
								      const int* secondDerivativeIndices,
								      const Real* edgeLengths,
								      int count)
{
    for (int i = 0; i < count; i++) {
        const int nodeIndex = probabilityIndices[i];
        hEigenMaps[nodeIndex] = eigenIndex;
        matrixGenerations[nodeIndex]++;

        for (int category = 0; category < kCategoryCount; category++) {
            const double categoryRate = hCategoryRates[0][category]; // XJ: because rate categories are only set for first eigen index
            hEdgeMultipliers[nodeIndex * kCategoryCount + category] = edgeLengths[i] * categoryRate;
        }
    }
    // TODO: check if need to copy it from host to device afterwards
    return BEAGLE_SUCCESS;
}

///-------------------------------- Factory -------------------------------------///	
BEAGLE_GPU_TEMPLATE
BeagleImpl*  BeagleGPUActionImplFactory<BEAGLE_GPU_GENERIC>::createImpl(int tipCount,
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
                                              long long preferenceFlags,
                                              long long requirementFlags,
                                              int* errorCode) {
    BeagleImpl * impl = new BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>();
    try {
        *errorCode =
            impl->createInstance(tipCount, partialsBufferCount, compactBufferCount, stateCount,
                                 patternCount, eigenBufferCount, matrixBufferCount,
                                 categoryCount,scaleBufferCount, resourceNumber, pluginResourceNumber, preferenceFlags, requirementFlags);
        if (*errorCode == BEAGLE_SUCCESS) {
            return impl;
        }
        delete impl;
        return NULL;
    }
    catch(...)
    {
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
const char* BeagleGPUActionImplFactory<double>::getName() {
    return "GPU-DP-CUDA-Action";
}

template<>
const char* BeagleGPUActionImplFactory<float>::getName() {
    return "GPU-SP-CUDA-Action";
}
#elif defined(FW_OPENCL)
template<>
const char* BeagleGPUActionImplFactory<double>::getName() {
    return "DP-OpenCL-Action";

}
template<>
const char* BeagleGPUActionImplFactory<float>::getName() {
    return "SP-OpenCL-Action";
}
#endif

BEAGLE_GPU_TEMPLATE
long long BeagleGPUActionImplFactory<BEAGLE_GPU_GENERIC>::getFlags() {
    return BeagleGPUImplFactory<BEAGLE_GPU_GENERIC>::getFlags() | BEAGLE_FLAG_COMPUTATION_ACTION;
}

} // end of device namespace
} // end of gpu namespace
} // end of beagle namespace


#endif
