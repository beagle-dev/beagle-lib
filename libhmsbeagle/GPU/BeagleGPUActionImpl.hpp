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
    for (int i = 0; i < kEigenDecompCount; i++) {
        dBsCsrOffsetsCache[i] = cudaDeviceNew<int>(kPaddedStateCount + 1);
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

    for (int i = 0; i < kEigenDecompCount; i++) {
        cudaFree(dBsCsrOffsetsCache[i]);
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

    // Operations are grouped into levels: each goes to the earliest level after every earlier operation that
    // writes a buffer it reads, or reads or writes a buffer (partials or scaling) it writes.  The operations of
    // a level are independent, so they can be computed together.
    std::vector<PartialsOperation> parsed(operationCount);
    std::vector<int> levels(operationCount);
    std::vector<int> writeLevel(kPartialsBufferCount, -1);
    std::vector<int> readLevel(kPartialsBufferCount, -1);
    std::map<Real*, int> scalingWriteLevel;
    int levelCount = 0;

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

    std::vector<PartialsOperation> batch;
    auto flush = [&]() {
        if (batch.empty())
            return;
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
        batch.clear();
    };

    for (int level = 0; level < levelCount; level++) {
        for (int op = 0; op < operationCount; op++) {
            if (levels[op] != level)
                continue;
            if ((int) batch.size() == maxBatchOperations)
                flush();
            batch.push_back(parsed[op]);
        }
        flush();
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
void BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::calcPartialsPartials(const std::vector<PartialsOperation>& operations,
                                                                   Real* cumulativeScalingBuffer) {
    const size_t matrixSize = (size_t) kPaddedStateCount * kPaddedPatternCount;
    // With few patterns, the groups' blocks of the 3 scratch matrices are side by side (coalescing the gathers
    // across groups); otherwise each group uses 2 contiguous blocks.  See computeActions( ).
    const size_t scratchMatrixSize = (size_t) 2 * kCategoryCount * maxBatchOperations * matrixSize;
    const bool interleaved = interleavedScratch();
    const size_t scratchStride = interleaved ? (size_t) 2 * kCategoryCount * maxBatchOperations * kPaddedPatternCount : kPaddedPatternCount;

    std::vector<ActionGroup> groups;
    for (size_t op = 0; op < operations.size(); op++) {
        for (int child = 0; child < 2; child++) {
            const int edgeIndex = operations[op].edges[child];
            for (int category = 0; category < kCategoryCount; category++) {
                const size_t g = (op * 2 + child) * kCategoryCount + category;
                ActionGroup group;
                group.in = (const Real*) dPartials[operations[op].children[child]] + category * matrixSize;
                group.out = dActionResults + g * matrixSize;
                group.scratchCount = interleaved ? 3 : 2;
                for (int k = 0; k < group.scratchCount; k++)
                    group.scratch[k] = dActionScratch + k * scratchMatrixSize + g * (interleaved ? kPaddedPatternCount : matrixSize);
                group.scratchStride = scratchStride;
                group.eigenIndex = hEigenMaps[edgeIndex];
                group.multiplier = hEdgeMultipliers[edgeIndex * kCategoryCount + category];
                groups.push_back(group);
            }
        }
    }
    computeActions(groups);

    // parent = left * right for all categories, then (optionally) rescale each pattern.
    bool scalers_log = (kFlags & BEAGLE_FLAG_SCALERS_LOG)?true:false;
    PartialsProductBatch<Real> products;
    for (size_t op = 0; op < operations.size(); op++) {
        const Real* left = dActionResults + op * 2 * kCategoryCount * matrixSize;
        auto& product = products.products[products.count++];
        product.left = left;
        product.right = left + kCategoryCount * matrixSize;
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

        // All the groups that share this group's B run in lockstep: one launch per term.
        const int eigenIndex = groups[first].eigenIndex;
        const auto& B = dBs[eigenIndex];
        std::vector<Schedule> schedules;
        for (size_t i = first; i < groups.size(); i++) {
            if (done[i] or groups[i].eigenIndex != eigenIndex)
                continue;
            done[i] = true;
            const ActionGroup& group = groups[i];
            auto [m, s] = getStatistics2(1.0, kPaddedPatternCount, group.multiplier, eigenIndex);
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
                    cuda_taylor_horner_terms(kPaddedStateCount, kPaddedPatternCount, B.offsets, B.inner, B.values, batch, stream);
                    batch.count = 0;
                }

                schedule.w = out;
                if (--schedule.term == 0) {
                    schedule.b = out;
                    if (++schedule.step < schedule.s)
                        schedule.startStep();
                }
            }
            cuda_taylor_horner_terms(kPaddedStateCount, kPaddedPatternCount, B.offsets, B.inner, B.values, batch, stream);
        }
    }
}

BEAGLE_GPU_TEMPLATE
double BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::getPMax() const
{
    return floor(0.5 + 0.5 * sqrt(5.0 + 4.0 * mMax));
}

BEAGLE_GPU_TEMPLATE
double BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::getDValue(int p, int eigenIndex) const
{
    assert(p >= 0 and p < hds[eigenIndex].size());

    return hds[eigenIndex][p];
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
                                                              int eigenIndex) const {
    assert( t >= 0 );
    assert( nCol >= 0);
    assert( edgeMultiplier >= 0 );
    assert( eigenIndex >= 0);

    if (t * hB1Norms[eigenIndex] == 0.0)
        return {0, 1};

    int bestM = INT_MAX;
    double bestS = INT_MAX;  // Not all the values of s can fit in a 32-bit int.

    const double pMax = getPMax();
    // pMax is the largest positive integer such that p*(p-1) <= mMax + 1

    // 1. The best (m,s) based only on ||B||_1 (in some sense the result for p=1).
    for (auto& [thisM, thetaM]: thetaConstants) {
        const double thisS = ceil(hB1Norms[eigenIndex] * edgeMultiplier / thetaM);
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
                const double dValueP = getDValue(p, eigenIndex);
                const double dValuePPlusOne = getDValue(p + 1, eigenIndex);
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
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::upPrePartials(bool byPartition,
							   const int *operations,
							   int operationCount,
							   int cumulativeScalingIndex)
{
    // Not yet implemented for action; the dense fallback would read transition matrices that are never formed.
    return BEAGLE_ERROR_NO_IMPLEMENTATION;
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
    buildShiftedCsr(kPaddedStateCount, rowIndices, colIndices, values, numNonZeros, mu_B, hBs[matrixIndex]);
    hB1Norms[matrixIndex] = normP1(kPaddedStateCount, hBs[matrixIndex]);

#ifdef BEAGLE_DEBUG_FLOW
    std::cerr<<"B "<<matrixIndex<<": mu = "<<mu_B<<"  ||B||_1 = "<<hB1Norms[matrixIndex]<<"\noffsets = \n";
    for (int x: hBs[matrixIndex].offsets) std::cerr<<x<<", ";
    std::cerr<<"\ncolumns = \n";
    for (int x: hBs[matrixIndex].columns) std::cerr<<x<<", ";
    std::cerr<<"\nvalues = \n";
    for (Real x: hBs[matrixIndex].values) std::cerr<<x<<", ";
    std::cerr<<"\n";
#endif

    const int currentNNZ = hBs[matrixIndex].nonZeros();
    if (currentCacheNNZs[matrixIndex] != currentNNZ) {
        currentCacheNNZs[matrixIndex] = currentNNZ;
        cudaDeviceDelete(dBsCsrColumnsCache[matrixIndex]);
        cudaDeviceDelete(dBsCsrValuesCache[matrixIndex]);
        dBsCsrColumnsCache[matrixIndex] = cudaDeviceNew<int>(currentNNZ);
        dBsCsrValuesCache[matrixIndex] = cudaDeviceNew<Real>(currentNNZ);
    }

    MemcpyHostToDevice(dBsCsrOffsetsCache[matrixIndex], hBs[matrixIndex].offsets.data(), kPaddedStateCount + 1);
    MemcpyHostToDevice(dBsCsrColumnsCache[matrixIndex], hBs[matrixIndex].columns.data(), currentNNZ);
    MemcpyHostToDevice(dBsCsrValuesCache[matrixIndex], hBs[matrixIndex].values.data(), currentNNZ);

#ifdef BEAGLE_DEBUG_FLOW

    std::cerr<<"\ndBsCsrOffsetsCache for "<< matrixIndex <<"="<<std::endl;
    PrintfDeviceVector(dBsCsrOffsetsCache[matrixIndex], kPaddedStateCount + 1, -1, 0, 0);

    std::cerr<<"dBsCsrColumnsCache for "<< matrixIndex <<"="<<std::endl;
    PrintfDeviceVector(dBsCsrColumnsCache[matrixIndex], currentNNZ, -1, 0, 0);

    std::cerr<<"currentNNZ ="<< currentNNZ <<std::endl;

    std::cerr<<"dBsCsrValuesCache for "<< matrixIndex <<"="<<std::endl;
    PrintfDeviceVector(dBsCsrValuesCache[matrixIndex], currentNNZ, -1, 0, 0);
#endif


    dBs[matrixIndex] = SpMatrixDevice<Real>(cublasHandle, cusparseHandle,
                                            kPaddedStateCount, kPaddedStateCount,
                                            currentNNZ,
                                            dBsCsrValuesCache[matrixIndex],
                                            dBsCsrColumnsCache[matrixIndex],
                                            dBsCsrOffsetsCache[matrixIndex],
                                            sparseFormat::csr);
    auto& dB = dBs[matrixIndex];

    // equation 3.7 in Al-Mohy and Higham: d_p = ||B^p||_1^(1/p).  d_1 is known exactly; d_0 is not used.
    const int pTop = getPMax() + 1;
    auto norms = normEstimator.estimate(dB, stream);
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
