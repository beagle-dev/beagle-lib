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


    dPartialsWrapper.resize(kPartialsBufferCount * kCategoryCount);
    for (int i = 0; i < kPartialsBufferCount; i++) {
        for (int category = 0; category < kCategoryCount; category++) {
	    auto ptr = (Real *) gpu->CreateSubPointer(dPartialsOrigin, sizeof(Real) * kPaddedStateCount * kPaddedPatternCount * (kCategoryCount * i + category), sizeof(Real) * kPaddedStateCount * kPaddedPatternCount);
	    auto mat = DnMatrixDevice<Real>(cublasHandle, ptr, kPaddedStateCount, kPaddedPatternCount);
            dPartialsWrapper[i * kCategoryCount + category] = std::move(mat);
        }
    }

    const size_t partialsSize = (size_t) kPaddedStateCount * kPaddedPatternCount * kCategoryCount;
    Real* dDestPLeftPtr = cudaDeviceNew<Real>(partialsSize);
    Real* dDestPRightPtr = cudaDeviceNew<Real>(partialsSize);
    // The two children are processed one after the other, so they share the term buffers.
    Real* dTermPtrs[2] = {cudaDeviceNew<Real>(partialsSize), cudaDeviceNew<Real>(partialsSize)};

    dDestPLeft.clear();
    dDestPRight.clear();
    dTerms[0].clear();
    dTerms[1].clear();

    for (int category = 0; category < kCategoryCount; category++)
    {
	size_t offset = kPaddedStateCount * kPaddedPatternCount * category;

        dDestPLeft.emplace_back(cublasHandle, dDestPLeftPtr + offset, kPaddedStateCount, kPaddedPatternCount);
        dDestPRight.emplace_back(cublasHandle, dDestPRightPtr + offset, kPaddedStateCount, kPaddedPatternCount);

        for (int k = 0; k < 2; k++)
            dTerms[k].emplace_back(cublasHandle, dTermPtrs[k] + offset, kPaddedStateCount, kPaddedPatternCount);
    }

    hEigenMaps.resize(kMatrixCount);
    hEdgeMultipliers.resize(kMatrixCount * kCategoryCount);

    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::~BeagleGPUActionImpl()
{
    cudaSetDevice(gpu->GetCudaDeviceOrdinal());
    cudaStreamSynchronize(stream);

    cudaFree(dSpMMBuffer);

    // The memory for each matrix group was allocation in a single block.
    cudaFree(dDestPLeft[0].ptr);
    cudaFree(dDestPRight[0].ptr);
    cudaFree(dTerms[0][0].ptr);
    cudaFree(dTerms[1][0].ptr);

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
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::setTipPartials(int tipIndex, const Real* inPartials) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUActionImpl::setTipPartials\n");
#endif

    BeagleGPUImpl<Real>::setTipPartials(tipIndex, inPartials);

    for (int category = 0; category < kCategoryCount; category++) {
        dPartialsWrapper[getPartialIndex(tipIndex, category)].setPointer((Real*) dPartials[tipIndex] + kPaddedStateCount * kPaddedPatternCount * category);
    }


#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUActionImpl::setTipPartials\n");
#endif

    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::setPartials(int tipIndex, const Real* inPartials) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUActionImpl::setPartials\n");
#endif

    BeagleGPUImpl<Real>::setPartials(tipIndex, inPartials);
    for (int category = 0; category < kCategoryCount; category++) {
        dPartialsWrapper[getPartialIndex(tipIndex, category)].setPointer((Real*) dPartials[tipIndex] + kPaddedStateCount * kPaddedPatternCount * category);
    }


#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUActionImpl::setPartials\n");
#endif

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
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::getPartialIndex(int nodeIndex, int categoryIndex) {
    return nodeIndex * kCategoryCount + categoryIndex;
}

BEAGLE_GPU_TEMPLATE
DnMatrixDevice<Real>& BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::getPartialsWrapper(int nodeIndex, int categoryIndex) {
    return dPartialsWrapper[getPartialIndex(nodeIndex, categoryIndex)];
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

        calcPartialsPartials(destinationPartialIndex, firstChildPartialIndex, firstChildSubstitutionMatrixIndex,
                             secondChildPartialIndex, secondChildSubstitutionMatrixIndex,
                             (rescale == 1) ? scalingFactors : nullptr, cumulativeScalingBuffer);

        if (kFlags & BEAGLE_FLAG_SCALING_ALWAYS) {
            int parScalingIndex = destinationPartialIndex - kTipCount;
            int child1ScalingIndex = firstChildPartialIndex - kTipCount;
            int child2ScalingIndex = secondChildPartialIndex - kTipCount;
            if (child1ScalingIndex >= 0 && child2ScalingIndex >= 0) {
                int scalingIndices[2] = {child1ScalingIndex, child2ScalingIndex};
                BeagleGPUImpl<Real>::accumulateScaleFactors(scalingIndices, 2, parScalingIndex);
            } else if (child1ScalingIndex >= 0) {
                int scalingIndices[1] = {child1ScalingIndex};
                BeagleGPUImpl<Real>::accumulateScaleFactors(scalingIndices, 1, parScalingIndex);
            } else if (child2ScalingIndex >= 0) {
                int scalingIndices[1] = {child2ScalingIndex};
                BeagleGPUImpl<Real>::accumulateScaleFactors(scalingIndices, 1, parScalingIndex);
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
void BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::calcPartialsPartials(int destPIndex,
                                                                   int partials1Index,
                                                                   int edgeIndex1,
                                                                   int partials2Index,
                                                                   int edgeIndex2,
                                                                   Real* scalingFactors,
                                                                   Real* cumulativeScalingBuffer) {

    for (int category = 0; category < kCategoryCount; category++)
    {
        simpleAction2(dDestPLeft[category], getPartialsWrapper(partials1Index, category), edgeIndex1, category);

        simpleAction2(dDestPRight[category], getPartialsWrapper(partials2Index, category), edgeIndex2, category);
    }

    // parent = left * right for all categories, then (optionally) rescale each pattern.
    bool scalers_log = (kFlags & BEAGLE_FLAG_SCALERS_LOG)?true:false;
    cuda_partials_product(dDestPLeft[0].ptr, dDestPRight[0].ptr, (Real*)dPartials[destPIndex],
                          kPaddedStateCount, kPaddedPatternCount, kCategoryCount,
                          scalingFactors, cumulativeScalingBuffer, scalers_log, stream);
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

    // siteLogLikelihoods[pattern] = log( sum_{category,state} likelihood(category,pattern,state) * categoryWeights[category] * statesFrequencies[state] )
    //                               + scaling factor
    // dSumLogLikelihood[0] = sum_{pattern} patternWeights[pattern] * siteLogLikelihoods[pattern]
    cuda_root_log_likelihoods(rootPartials, categoryWeights, stateFrequencies, scale ? dCumulativeScalingFactor : nullptr,
                              (Real*)dPatternWeights, kPaddedStateCount, kPaddedPatternCount, kPatternCount, kCategoryCount,
                              siteLogLikelihoods, (Real*)dSumLogLikelihood, stream);

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
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::simpleAction2(DnMatrixDevice<Real>& F, const DnMatrixDevice<Real>& inPartials, int edgeIndex, int category) {
    const double tol = pow(2.0, -53.0);
    const double t = 1.0;
    const int nCol = kPaddedPatternCount;
    const int edgeMultiplierIndex = edgeIndex * kCategoryCount + category;
    const int eigenIndex = hEigenMaps[edgeIndex];

    const double edgeMultiplier = hEdgeMultipliers[edgeMultiplierIndex];

    auto [m, s] = getStatistics2(t, nCol, edgeMultiplier, eigenIndex);


#ifdef BEAGLE_DEBUG_FLOW
    std::cerr << "\n\nsimpleAction2: m = " << m << "  s = " << s << " t = " << t << " nCol = " << nCol << " edgeMultiplier = " << edgeMultiplier << std::endl;
    std::cerr << "  edgeIndex = " << edgeIndex << " category = " << category << std::endl;
#endif

    // A = edgeMultiplier * B, so A is never formed: the multiplier goes into the SpMM alpha.
    auto& B = dBs[eigenIndex];

    // For small problems one fused kernel per term beats cuSPARSE's 3 kernels + axpy, which are
    // launch-bound there.  (A100: 7 vs 21 us per term for IGC at 1 column; cuSPARSE wins from
    // about nnz * columns = 3e6, e.g. 53 vs 67 us for MM M=32 at 64 columns.)
    const bool fusedTerms = (double) B.num_non_zeros * kPaddedPatternCount < 3e6;

    // F accumulates the Taylor sum in place.  Each term is written to one of the two term buffers and
    // is the input of the next term, so no copies are needed.
    F.copyFromAsync( inPartials, stream );

    const Real eta = exp(t * hMuBs[eigenIndex] * edgeMultiplier / (Real) s);

    for (int i = 0; i < s; i++) {

        // The first term of each step reads the current F.
        const DnMatrixDevice<Real>* x = (i == 0) ? &inPartials : &F;
        bool scaled = false;

        // The fused kernel updates F while other threads read x, so x must not be F.
        // (Term j=1 writes dTerms[1], so dTerms[0] is free here.)
        if (fusedTerms and x == &F)
        {
            dTerms[0][category].copyFromAsync(F, stream);
            x = &dTerms[0][category];
        }

#ifdef ACTION_EARLY_EXIT_NORM
        Real c1 = normPInf(*x);
#endif

        for (int j = 1; j < m + 1; j++) {

            auto& y = dTerms[j % 2][category];
            const Real alpha = edgeMultiplier * t / ((Real) s * j);

            // y = alpha * B * x; F += y, folding in F *= eta after the last term.
            const Real scale = (j < m) ? 1 : eta;
            if (fusedTerms)
                cuda_csr_taylor_term(kPaddedStateCount, kPaddedPatternCount, B.offsets, B.inner, B.values, alpha,
                                     x->ptr, y.ptr, F.ptr, scale, stream);
            else
            {
                spMMPrepared<Real>(y, alpha, B, *x, dSpMMBuffer);
                if (j < m)
                    F += y;
                else
                    F.axpby(eta, eta, y);
            }
            scaled = (j == m);

#ifdef BEAGLE_DEBUG_FLOW
            std::cerr<<"j/m = "<<j<<"/"<<m<<", i = "<<i<<", alpha = "<<edgeMultiplier * t / ((Real) s * j)<<std::endl;
            std::cerr<<"term =\n"<<y<<"\n";
            std::cerr<<"F =\n"<<F<<"\n";
#endif

#ifdef ACTION_EARLY_EXIT_NORM
            Real c2 = normPInf(y);
            if (c1 + c2 <= tol * normPInf(F)) break;
            c1 = c2;
#endif
            x = &y;
        }

        if (not scaled)
            F *= eta;

#ifdef BEAGLE_DEBUG_FLOW
        std::cerr<<"i = "<<i<<" F:\n"<<"  "<<byCol(F)<<"\n";
#endif
    }

    return BEAGLE_SUCCESS;
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

    // Size the work buffer for the Taylor-term SpMMs once per matrix, not on every call.
    size_t newSpMMBufferSize = spMMBufferSize(dTerms[0][0], dB, dTerms[1][0]);
    if (newSpMMBufferSize > spMMBufferBytes or dSpMMBuffer == nullptr)
    {
        CHECK_CUDA(cudaStreamSynchronize(stream));
        cudaDeviceDelete(dSpMMBuffer);
        spMMBufferBytes = std::max<size_t>(newSpMMBufferSize, 1);
        dSpMMBuffer = cudaMallocWrapped(spMMBufferBytes);
    }

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
