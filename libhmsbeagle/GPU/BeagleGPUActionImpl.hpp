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

    hIdentity = SpMatrix<Real>(kPaddedStateCount, kPaddedStateCount);
    hIdentity.setIdentity();
    hInstantaneousMatrices.resize(kEigenDecompCount);
    hBs.resize(kEigenDecompCount);
    for (int i = 0; i < kEigenDecompCount; i++) {
        hInstantaneousMatrices[i] = SpMatrix<Real>(kPaddedStateCount, kPaddedStateCount);
        hBs[i] = SpMatrix<Real>(kPaddedStateCount, kPaddedStateCount);
    }
    hMuBs.resize(kEigenDecompCount);
    hB1Norms.resize(kEigenDecompCount);
    hds.resize(kEigenDecompCount);

    CUBLAS_CHECK(cublasCreate(&cublasHandle));  //TODO: destroyer: CUBLAS_CHECK(cublasDestroy(cublasHandle));
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

    dBsCsrOffsetsCache.resize(kEigenDecompCount);
    dBsCsrColumnsCache.resize(kEigenDecompCount);
    dBsCsrValuesCache.resize(kEigenDecompCount);
    dACscValuesCache.resize(kEigenDecompCount * kCategoryCount * 2);
    dAs = std::vector<SpMatrixDevice<Real>>(kEigenDecompCount * kCategoryCount * 2);
    for(int p=0; p<=getPMax()+1; p++)
    {
        int t = 5;
        int itmax = 2;
        L1normForPower.push_back({cublasHandle, p, kPaddedStateCount, t, itmax});

//        This is very fast, but computes |A| instead of |A^p|
//        L1normForPower.push_back(GPUnormest2<Real>(cublasHandle, cusparseHandle, p, kPaddedStateCount));
    }

    currentCacheNNZs = std::vector<int>(kEigenDecompCount, kPaddedStateCount);
    for (int i = 0; i < kEigenDecompCount; i++) {
        dBsCsrOffsetsCache[i] = cudaDeviceNew<int>(kPaddedStateCount + 1);
        dBsCsrColumnsCache[i] = cudaDeviceNew<int>(currentCacheNNZs[i]);
        dBsCsrValuesCache[i] = cudaDeviceNew<Real>(currentCacheNNZs[i]);
        for (int j = 0; j < kCategoryCount; j++) {
            dACscValuesCache[i * kCategoryCount * 2 + 2 * j] = cudaDeviceNew<Real>(currentCacheNNZs[i]);
            dACscValuesCache[i * kCategoryCount * 2 + 2 * j + 1] = cudaDeviceNew<Real>(currentCacheNNZs[i]);
        }
    }


    dPartialsWrapper.resize(kPartialsBufferCount * kCategoryCount);
    for (int i = 0; i < kPartialsBufferCount; i++) {
        for (int category = 0; category < kCategoryCount; category++) {
	    auto ptr = (Real *) gpu->CreateSubPointer(dPartialsOrigin, sizeof(Real) * kPaddedStateCount * kPaddedPatternCount * (kCategoryCount * i + category), sizeof(Real) * kPaddedStateCount * kPaddedPatternCount);
	    auto mat = DnMatrixDevice<Real>(cublasHandle, ptr, kPaddedStateCount, kPaddedPatternCount);
            dPartialsWrapper[i * kCategoryCount + category] = std::move(mat);
        }
    }

    Real* dDestPLeftPtr = cudaDeviceNew<Real>(kPaddedStateCount * kPaddedPatternCount * kCategoryCount);
    Real* dDestPRightPtr = cudaDeviceNew<Real>(kPaddedStateCount * kPaddedPatternCount * kCategoryCount);

    Real* dLeftCachePtr = cudaDeviceNew<Real>(kPaddedStateCount * kPaddedPatternCount * kCategoryCount);
    Real* dRightCachePtr = cudaDeviceNew<Real>(kPaddedStateCount * kPaddedPatternCount * kCategoryCount);

    Real* dIntegrationTmpLeftCachePtr = cudaDeviceNew<Real>(kPaddedStateCount * kPaddedPatternCount * kCategoryCount);
    Real* dIntegrationTmpRightCachePtr = cudaDeviceNew<Real>(kPaddedStateCount * kPaddedPatternCount * kCategoryCount);

    integrationLeftBufferSize = std::vector<size_t>(kCategoryCount, kPaddedStateCount * kPaddedPatternCount);
    dIntegrationLeftBuffer = std::vector<void*>(kCategoryCount, nullptr);
    integrationRightBufferSize = std::vector<size_t>(kCategoryCount, kPaddedStateCount * kPaddedPatternCount);
    dIntegrationRightBuffer = std::vector<void*>(kCategoryCount, nullptr);

    dDestPLeft.clear();

    dDestPRight.clear();

    dFLeft.clear();

    dFRight.clear();

    dIntegrationTmpLeft.clear();

    dIntegrationTmpRight.clear();

    for (int category = 0; category < kCategoryCount; category++)
    {
	size_t offset = kPaddedStateCount * kPaddedPatternCount * category;

        dDestPLeft.emplace_back(cublasHandle, dDestPLeftPtr + offset, kPaddedStateCount, kPaddedPatternCount);
        dDestPRight.emplace_back(cublasHandle, dDestPRightPtr + offset, kPaddedStateCount, kPaddedPatternCount);

        dFLeft.emplace_back(cublasHandle, dLeftCachePtr + offset, kPaddedStateCount, kPaddedPatternCount);
        dFRight.emplace_back(cublasHandle, dRightCachePtr + offset, kPaddedStateCount, kPaddedPatternCount);

	dIntegrationTmpLeft.emplace_back(cublasHandle, dIntegrationTmpLeftCachePtr + offset,  kPaddedStateCount, kPaddedPatternCount);
	dIntegrationTmpRight.emplace_back(cublasHandle, dIntegrationTmpRightCachePtr + offset,  kPaddedStateCount, kPaddedPatternCount);
    }

    for (int category = 0; category < kCategoryCount; category++) {

        CHECK_CUDA(cudaMalloc(&dIntegrationLeftBuffer[category], integrationLeftBufferSize[category]))
        CHECK_CUDA(cudaMalloc(&dIntegrationRightBuffer[category], integrationRightBufferSize[category]))
    }

    hEigenMaps.resize(kMatrixCount);
    hEdgeMultipliers.resize(kMatrixCount * kCategoryCount);

    msCache = std::vector<tuple<int, int>>(2 * kCategoryCount, tuple<int, int>{0, 0});
    etaCache = std::vector<Real>(2 * kCategoryCount, 0);
    c1Cache.resize(2 * kCategoryCount);
    c2Cache.resize(2 * kCategoryCount);
    alphaCache.resize(2 * kCategoryCount);
    integrationMultipliers = std::vector<Real>(kCategoryCount * 2, 1);

    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::~BeagleGPUActionImpl()
{
    cudaSetDevice(gpu->GetCudaDeviceOrdinal());

    for (int i = 0; i < kCategoryCount; i++) {
        cudaFree(dIntegrationLeftBuffer[i]);
        cudaFree(dIntegrationRightBuffer[i]);
    }

    // The memory for each matrix group was allocation in a single block.
    cudaFree(dDestPLeft[0].ptr);
    cudaFree(dDestPRight[0].ptr);
    cudaFree(dFLeft[0].ptr);
    cudaFree(dFRight[0].ptr);
    cudaFree(dIntegrationTmpLeft[0].ptr);
    cudaFree(dIntegrationTmpRight[0].ptr);

    for (int i = 0; i < kEigenDecompCount; i++) {
        cudaFree(dBsCsrOffsetsCache[i]);
        cudaFree(dBsCsrColumnsCache[i]);
        cudaFree(dBsCsrValuesCache[i]);
        for (int j = 0; j < kCategoryCount * 2; j++) {
            cudaFree(dACscValuesCache[i * kCategoryCount * 2 + j]);
//            cusparseDestroySpMat(dAs[i * kCategoryCount * 2 + i]);
        }
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
        dPartialsWrapper[getPartialIndex(tipIndex, category)].ptr = (Real*) dPartials[tipIndex] + kPaddedStateCount * kPaddedPatternCount * category;
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
        dPartialsWrapper[getPartialIndex(tipIndex, category)].ptr = (Real*) dPartials[tipIndex] + kPaddedStateCount * kPaddedPatternCount * category;
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
void  BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::rescalePartials(Real* partials, Real* scalingFactors, Real* cumulativeScalingBuffer, int streamIndex)
{
//    std::cerr<<"initial  partials (kernel) = "<<asDeviceVec((Real*)partials, kPaddedStateCount * kPaddedPatternCount * kCategoryCount)<<"\n";

//            kernels->RescalePartials(partials3, scalingFactors, cumulativeScalingBuffer,
//                                     kPaddedPatternCount, kCategoryCount, 0, streamIndex, -1);

    bool scalers_log = (kFlags & BEAGLE_FLAG_SCALERS_LOG)?true:false;
    // rescalePartialsDevice(partials, scalingFactors, cumulativeScalingBuffer, kPaddedStateCount, kPaddedPatternCount, kCategoryCount);
    rescalePartials2(scalers_log, kCategoryCount, kPaddedPatternCount, kPaddedStateCount, partials, scalingFactors, cumulativeScalingBuffer, streamIndex);
//    std::cerr<<"rescaled partials (kernel) = "<<asDeviceVec((Real*)partials, kPaddedStateCount * kPaddedPatternCount * kCategoryCount)<<"\n";
//    std::cerr<<"scaling factors (kernel) = "<<asDeviceVec((Real*)scalingFactors, kPatternCount)<<"\n";
}

BEAGLE_GPU_TEMPLATE
void BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::setCudaDevice()
{
    // GPUInterface uses the driver API and pops its context after each call, so without this
    // runtime-API work would go to device 0.
    auto status = cudaSetDevice(gpu->GetCudaDeviceOrdinal());
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("cudaSetDevice: ") + cudaGetErrorString(status));
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

    int streamIndex = -1;
    int waitIndex = -1;

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
                             secondChildPartialIndex, secondChildSubstitutionMatrixIndex);


        if (rescale == 1)
        {
            Real* partials3 = (Real*)dPartials[destinationPartialIndex];

	    rescalePartials(partials3, scalingFactors, cumulativeScalingBuffer, streamIndex);
        }

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
                                                                   int edgeIndex2) {

    cacheAMatrices(edgeIndex1, edgeIndex2, false);

    for (int category = 0; category < kCategoryCount; category++)
    {

        const int matrixIndex1 = hEigenMaps[edgeIndex1] * kCategoryCount * 2 + category;
        const int matrixIndex2 = hEigenMaps[edgeIndex2] * kCategoryCount * 2 + kCategoryCount + category;

        simpleAction2(dDestPLeft[category], getPartialsWrapper(partials1Index, category),
                      edgeIndex1, category, matrixIndex1, true, false);

        simpleAction2(dDestPRight[category], getPartialsWrapper(partials2Index, category),
                      edgeIndex2, category, matrixIndex2, false, false);
    }

    // simpleAction3(partials1Index, edgeIndex1, partials2Index, edgeIndex2);

    for (int category = 0; category < kCategoryCount; category++)
    {
        const int destPartialIndex = getPartialIndex(destPIndex, category);

	// element-wise multiply
        if constexpr (std::is_same<Real, float>::value) {
            CUBLAS_CHECK(cublasSdgmm(cublasHandle, CUBLAS_SIDE_LEFT, kPaddedStateCount * kPaddedPatternCount, 1, dDestPLeft[category].ptr, kPaddedStateCount * kPaddedPatternCount,
                                     dDestPRight[category].ptr, 1, dPartialsWrapper[destPartialIndex].ptr, kPaddedStateCount * kPaddedPatternCount));

        } else {
            CUBLAS_CHECK(cublasDdgmm(cublasHandle, CUBLAS_SIDE_LEFT, kPaddedStateCount * kPaddedPatternCount, 1, dDestPLeft[category].ptr, kPaddedStateCount * kPaddedPatternCount,
                                     dDestPRight[category].ptr, 1, dPartialsWrapper[destPartialIndex].ptr, kPaddedStateCount * kPaddedPatternCount));

        }
//#ifdef BEAGLE_DEBUG_FLOW
//        std::cerr<<"Checking p_parent = p_1 * p_2, parent index = "<<destPartialIndex<<" chil1 index = " << partial1CacheIndex<< " child2 index = "<<partial2CacheIndex<<std::endl;
//        std::cerr<<"p1 = "<<std::endl;
//        PrintfDeviceVector(dPartialCache[partial1CacheIndex], kPaddedStateCount * kPaddedPatternCount, -1, 0, 0);
//        std::cerr<<"p2 = "<<std::endl;
//        PrintfDeviceVector(dPartialCache[partial2CacheIndex], kPaddedStateCount * kPaddedPatternCount, -1, 0, 0);
//        std::cerr<<"p_parent = "<<std::endl;
//        PrintfDeviceVector(dPartialCache[destPartialIndex], kPaddedStateCount * kPaddedPatternCount, -1, 0, 0);
//#endif
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

    Real* siteProbs = (Real*)dIntegrationTmp;
    Real* rootPartials = (Real*)dPartials[rootNodeIndex];
    Real* categoryWeights = (Real*)dWeights[categoryWeightsIndex];
    Real* stateFrequencies = (Real*)dFrequencies[stateFrequenciesIndex]; // these really should be category-dependent

    // 1. Compute pr[pattern] = sum_{category,state} likelihood(category,pattern,state) * categoryWeights[category] * statesFrequencies[state]
    sumRootLikelihoods(siteProbs, rootPartials, categoryWeights, stateFrequencies, kPaddedStateCount, kPaddedPatternCount, kCategoryCount);

    // auto hStateFrequencies = MemcpyDeviceToHostVector((Real*)dFrequencies[stateFrequenciesIndex], kStateCount);
    // auto hPatternWeights = MemcpyDeviceToHostVector((Real*)dPatternWeights, kPatternCount);
    if (kFlags & BEAGLE_FLAG_SCALING_AUTO)
    {
	// scaling factor per pattern*category -- see BeagleGPUImpl<>::createInstance
	// scaling factor is char?? -- see BeagleGPUImpl<>::createInstance
	// scaling factor is int??  -- see KernelIntegrateLikelihoodsAutoScaling
	// See kernelIntegrateLikelihoodsAutoScaling in KernelsX.cu
	std::cerr<<"BeagleGPUActionImpl< >::calculateRootLogLikelihoods -- FLAG_SCALING_AUTO not implemented!";
	std::abort();
    }

    // 2. Compute log(pr[pattern])
    cuda_log_vector(siteProbs, kPatternCount);

    // 3. Add scaling factors back in.
    if (scale)
    {
	cublasStatus_t status;
	if constexpr (std::is_same<Real, float>::value)
	{
	    Real alpha = 1;
	    status = cublasSaxpy(cublasHandle, kPatternCount,
				 &alpha,
				 dCumulativeScalingFactor, 1,
				 siteProbs, 1);
	}
	else
	{
	    Real alpha = 1;
	    status = cublasDaxpy(cublasHandle, kPatternCount,
				 &alpha,
				 dCumulativeScalingFactor, 1,
				 siteProbs, 1);
	}
	if (status != CUBLAS_STATUS_SUCCESS)
	{
	    std::cerr<<"cublas error "<<status<<" in cublas<t>axpy( ).";
	    exit(1);
	}
    }


/*
    std::cerr<<"site probs (d1) = "<<asDeviceVec((Real*)dIntegrationTmp, kPatternCount)<<"\n";
*/

#ifdef BEAGLE_DEBUG_VALUES
    std::cerr<<"before pattern weights = "<<asDeviceVec((Real*)dIntegrationTmp, kPatternCount)<<"\n";
#endif

    // Take the dot product of the pattern log-likelihoods and the pattern weights.
    // The output (dSumLogLikelihood) needs to be a device pointer.
    dotProduct((Real*)dSumLogLikelihood, cublasHandle, kPatternCount, (Real*)dIntegrationTmp, (Real*)dPatternWeights);

//    std::cerr<<"logLikelihood (kernel) = "<<MemcpyDeviceToHostVector((Real*)dSumLogLikelihood,1)[0]<<"   logLikelihood (ours) = "<<OurResult<<"\n";

    if (kFlags & BEAGLE_FLAG_COMPUTATION_SYNCH) {
        gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dSumLogLikelihood, sizeof(Real) * kSumSitesBlockCount);

        *outSumLogLikelihood = 0.0;
        for (int i = 0; i < kSumSitesBlockCount; i++) {
            if (hLogLikelihoodsCache[i] != hLogLikelihoodsCache[i])
                returnCode = BEAGLE_ERROR_FLOATING_POINT;

            *outSumLogLikelihood += hLogLikelihoodsCache[i];
        }
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
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::cacheAMatrices(int edgeIndex1, int edgeIndex2, bool transpose)
{
    for (int category = 0; category < kCategoryCount; category++) {
        const int matrixIndex1 = hEigenMaps[edgeIndex1] * kCategoryCount * 2 + category;
        const int edgeMultiplierIndex1 = edgeIndex1 * kCategoryCount + category;
        const Real edgeMultiplier1 = hEdgeMultipliers[edgeMultiplierIndex1];

        const int matrixIndex2 = hEigenMaps[edgeIndex2] * kCategoryCount * 2 + kCategoryCount + category;
        const int edgeMultiplierIndex2 = edgeIndex2 * kCategoryCount + category;
        const Real edgeMultiplier2 = hEdgeMultipliers[edgeMultiplierIndex2];

        MemcpyDeviceToDevice(dACscValuesCache[matrixIndex1], dBsCsrValuesCache[hEigenMaps[edgeIndex1]], currentCacheNNZs[hEigenMaps[edgeIndex1]]);
        MemcpyDeviceToDevice(dACscValuesCache[matrixIndex2], dBsCsrValuesCache[hEigenMaps[edgeIndex2]], currentCacheNNZs[hEigenMaps[edgeIndex2]]);
        auto format = transpose ? sparseFormat::csc : sparseFormat::csr;

        dAs[matrixIndex1] = SpMatrixDevice<Real>(cublasHandle, cusparseHandle,
                                                 kPaddedStateCount, kPaddedStateCount,
                                                 currentCacheNNZs[hEigenMaps[edgeIndex1]],
                                                 dACscValuesCache[matrixIndex1],
                                                 dBsCsrColumnsCache[hEigenMaps[edgeIndex1]],
                                                 dBsCsrOffsetsCache[hEigenMaps[edgeIndex1]],
                                                 format);

        dAs[matrixIndex2] = SpMatrixDevice<Real>(cublasHandle, cusparseHandle,
                                                 kPaddedStateCount, kPaddedStateCount,
                                                 currentCacheNNZs[hEigenMaps[edgeIndex2]],
                                                 dACscValuesCache[matrixIndex2],
                                                 dBsCsrColumnsCache[hEigenMaps[edgeIndex2]],
                                                 dBsCsrOffsetsCache[hEigenMaps[edgeIndex2]],
                                                 format);

        dAs[matrixIndex1] *= edgeMultiplier1; //Check if this is asynchronous with the following
        dAs[matrixIndex2] *= edgeMultiplier2;

#ifdef BEAGLE_DEBUG_FLOW
        std::cerr<<"category = "<<category<<std::endl;
        std::cerr<<"matrixIndex1 = "<<matrixIndex1<<std::endl;
        std::cerr<<"edgeIndex1 = "<<edgeIndex1<<std::endl;
        std::cerr<<"edgeMultiplierIndex1 = "<<edgeMultiplierIndex1<<std::endl;
        std::cerr<<"edgeMultiplier1 = "<<edgeMultiplier1<<std::endl;
        std::cerr<<"hEigenMaps[edgeIndex1] = "<<hEigenMaps[edgeIndex1]<<std::endl;
        std::cerr<<"dAs["<<matrixIndex1<<"] = \n"<<dAs[matrixIndex1]<<"\n";
#endif
    }

    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::simpleAction2(DnMatrixDevice<Real>& destP, const DnMatrixDevice<Real>& inPartials, int edgeIndex, int category, int matrixIndex, bool left, bool transpose) {
    const double tol = pow(2.0, -53.0);
    const double t = 1.0;
    const int nCol = kPaddedPatternCount;
    const int edgeMultiplierIndex = edgeIndex * kCategoryCount + category;

    const double edgeMultiplier = hEdgeMultipliers[edgeMultiplierIndex];

    auto [m, s] = getStatistics2(t, nCol, edgeMultiplier, hEigenMaps[edgeIndex]);


#ifdef BEAGLE_DEBUG_FLOW
    std::cerr << "\n\nsimpleAction2: m = " << m << "  s = " << s << " t = " << t << " nCol = " << nCol << " edgeMultiplier = " << edgeMultiplier << std::endl;
    std::cerr << "  edgeIndex = " << edgeIndex << " category = " << category << " matrixIndex = "<<matrixIndex << std::endl;
#endif

//    SpMatrix<Real> A = hBs[hEigenMaps[edgeIndex]] * edgeMultiplier;
    auto& A = dAs[matrixIndex];
    vector<DnMatrixDevice<Real>>* FF;
    vector<DnMatrixDevice<Real>>* integrationTmpPtr;
    // Pointers, not copies: spMM may reallocate the buffer, and the new one must be kept.
    std::vector<void*>* integrationBufferPtr;
    std::vector<size_t>* integrationBufferSizePtr;
    if (left) {
        FF = &dFLeft;
        integrationTmpPtr = &dIntegrationTmpLeft;
        integrationBufferPtr = &dIntegrationLeftBuffer;
        integrationBufferSizePtr = &integrationLeftBufferSize;
    } else {
        FF = &dFRight;
        integrationTmpPtr = &dIntegrationTmpRight;
        integrationBufferPtr = &dIntegrationRightBuffer;
        integrationBufferSizePtr = &integrationRightBufferSize;
    }
    auto& F = (*FF)[category];
    auto& integrationTmp = (*integrationTmpPtr)[category];
    auto& integrationBuffer = (*integrationBufferPtr)[category];
    auto& integrationBufferSize = (*integrationBufferSizePtr)[category];

    destP.copyFrom( inPartials );

    F.copyFrom( inPartials );

    const Real eta = exp(t * hMuBs[hEigenMaps[edgeIndex]] * edgeMultiplier / (Real) s);

    const Real zero = 0;
    const Real one = 1;
    for (int i = 0; i < s; i++) {

#ifdef ACTION_EARLY_EXIT_NORM
        Real c1 = normPInf(destP);
#endif

        for (int j = 1; j < m + 1; j++) {

#ifdef BEAGLE_DEBUG_FLOW
            std::cerr<<"j/m = "<<j<<"/"<<m<<", i = "<<i<<", alpha = "<<t / ((Real) s * j)<<std::endl;
            std::cerr<<"A = \n" << A << "\n";
            std::cerr<<"destP = \n" << destP << "\n";
            std::cerr<<"F = \n" << F << "\n";
#endif


            spMM<Real>(integrationTmp, t / ((Real) s * j), A, destP, 0, integrationBuffer, integrationBufferSize);

            destP.copyFrom( integrationTmp );

            F += destP;

#ifdef BEAGLE_DEBUG_FLOW
            std::cerr<<"After destP = \n" << destP << "\n";
            std::cerr<<"Action2 Step 2, integrationTmp =\n"<<integrationTmp<<"\n";
            std::cerr<<"Action2 Step 2, destP =\n"<<destP<<"\n";
            std::cerr<<"Action2 Step 2, F =\n"<<F<<"\n";
#endif

#ifdef ACTION_EARLY_EXIT_NORM
            Real c2 = normPInf(destP);
            if (c1 + c2 <= tol * normPInf(F)) break;
            c1 = c2;
#endif
        }

        F *= eta;

        destP.copyFrom ( F );

#ifdef BEAGLE_DEBUG_FLOW
        std::cerr<<"i = "<<i<<" destP = F:\n"<<"  "<<byCol(destP)<<"\n";
#endif
    }

    return BEAGLE_SUCCESS;
}


BEAGLE_GPU_TEMPLATE
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::simpleAction3(int partialsIndex1, int edgeIndex1,
                                                           int partialsIndex2, int edgeIndex2) {
    const double tol = pow(2.0, -53.0);
    const double t = 1.0;
    const int nCol = kPaddedPatternCount;



    int m_max = 0, s_max = 0;
    for (int category = 0; category < kCategoryCount; category++) {
        const int edgeMultiplierIndex1 = edgeIndex1 * kCategoryCount + category;
        const double edgeMultiplier1 = hEdgeMultipliers[edgeMultiplierIndex1];
        msCache[category] = getStatistics2(t, nCol, edgeMultiplier1, hEigenMaps[edgeIndex1]);
        auto [m1, s1] = msCache[category];
        if (m_max < m1) m_max = m1;
        if (s_max < s1) s_max = s1;

        const int edgeMultiplierIndex2 = edgeIndex2 * kCategoryCount + category;
        const double edgeMultiplier2 = hEdgeMultipliers[edgeMultiplierIndex2];
        msCache[kCategoryCount + category] = getStatistics2(t, nCol, edgeMultiplier2, hEigenMaps[edgeIndex2]);
        auto [m2, s2] = msCache[kCategoryCount + category];
        if (m_max < m2) m_max = m2;
        if (s_max < s2) s_max = s2;
    }

#ifdef BEAGLE_DEBUG_FLOW
    std::cerr<<"s_max = "<<s_max<<" m_max = "<< m_max<<", partialsIndex1 = "<< partialsIndex1<< ", partialsIndex2 = "<< partialsIndex2 << "\n";
    std::cerr<<"Entering step 1: destP = inPartial, F = inPartial"<<"\n";
#endif

    for (int category = 0; category < kCategoryCount; category++) {

        auto& FLeft = dFLeft[category];
        auto& FRight = dFRight[category];

        auto& destPLeft = dDestPLeft[category];
        auto& destPRight = dDestPRight[category];

        auto& inPartialsLeft = getPartialsWrapper(partialsIndex1, category);
        auto& inPartialsRight = getPartialsWrapper(partialsIndex2, category);

        destPLeft.copyFromAsync( inPartialsLeft );
        destPRight.copyFromAsync( inPartialsRight );

        FLeft.copyFromAsync( inPartialsLeft );
        FRight.copyFromAsync( inPartialsRight );


#ifdef BEAGLE_DEBUG_FLOW
        std::cerr<<"category = "<<category<<"\n";
        std::cerr<<"inPartialsLeft = \n" << inPartialsLeft << "\n";
        std::cerr<<"inPartialsRight = \n" << inPartialsRight << "\n";

        std::cerr<<"destPLeft = \n" << destPLeft << "\n";
        std::cerr<<"destPRight = \n" << destPRight << "\n";
        std::cerr<<"FLeft = \n" << FLeft << "\n";
        std::cerr<<"FRight = \n" << FRight << "\n";

#endif


    }
    cudaDeviceSynchronize();




    const Real etaConst1 = t * hMuBs[hEigenMaps[edgeIndex1]] / (Real) s_max;
    const Real etaConst2 = t * hMuBs[hEigenMaps[edgeIndex2]] / (Real) s_max;


    for (int i = 0; i < s_max; i++) {
        for (int j = 1; j < m_max + 1; j++) {
            //#ifdef BEAGLE_DEBUG_FLOW
            //            std::cerr<<"j/m = "<<j<<"/"<<m<<", alpha = "<<alpha<<std::endl;
            //#endif

            for (int category = 0; category < kCategoryCount; category++) {
                const int matrixIndex1 = hEigenMaps[edgeIndex1] * kCategoryCount * 2 + category;
                const int matrixIndex2 = hEigenMaps[edgeIndex2] * kCategoryCount * 2 + kCategoryCount + category;

                auto& ALeft = dAs[matrixIndex1];
                auto& ARight = dAs[matrixIndex2];

                auto& FLeft = dFLeft[category];
                auto& FRight = dFRight[category];

                auto& destPLeft = dDestPLeft[category];
                auto& destPRight = dDestPRight[category];

                spMM<Real>(dIntegrationTmpLeft[category], t / ((Real) s_max * j), ALeft, destPLeft, 0, dIntegrationLeftBuffer[category], integrationLeftBufferSize[category]);
                spMM<Real>(dIntegrationTmpRight[category], t / ((Real) s_max * j), ARight, destPRight, 0, dIntegrationRightBuffer[category], integrationRightBufferSize[category]);

#ifdef BEAGLE_DEBUG_FLOW
                cudaDeviceSynchronize();
                std::cerr<<"Step2, integrationTmp = alpha * A * destP, category = "<<category << ", j/m = "<<j<<"/"<<m_max<<", i = "<<i<< ", alpha = " << t / ((Real) s_max * j) <<"\n";
                std::cerr<<"Aleft =\n" << ALeft << "\n";
                std::cerr<<"destPLeft = \n" << destPLeft << "\n";
                std::cerr<<"dIntegrationTmpLeft = \n" << dIntegrationTmpLeft[category] << "\n";

                // std::cerr<<"ARight =\n" << ARight << "\n";
                // std::cerr<<"destPRight = \n" << destPRight << "\n";
                // std::cerr<<"dIntegrationTmpRight = \n" << dIntegrationTmpRight[category] << "\n";

#endif
            }
            cudaDeviceSynchronize();



            for (int category = 0; category < kCategoryCount; category++) {

                auto& FLeft = dFLeft[category];
                auto& FRight = dFRight[category];

                auto& destPLeft = dDestPLeft[category];
                auto& destPRight = dDestPRight[category];

#ifdef BEAGLE_DEBUG_FLOW
                cudaDeviceSynchronize();
                std::cerr<<"Before Step2, F += IntegrationTmp, category = "<<category << ", j/m = "<<j<<"/"<<m_max<<", i = "<<i<< ", alpha = " << t / ((Real) s_max * j) <<"\n";
                std::cerr<<"destPLeft =\n" << destPLeft << "\n";
                std::cerr<<"Before, FLeft = \n" << FLeft << "\n";

                // std::cerr<<"destPRight = \n" << destPRight << "\n";
                // std::cerr<<"FRight = \n" << FRight << "\n";

#endif

                destPLeft.copyFromAsync( dIntegrationTmpLeft[category] );
                destPRight.copyFromAsync( dIntegrationTmpRight[category] );

                FLeft += dIntegrationTmpLeft[category];
                FRight += dIntegrationTmpRight[category];

#ifdef BEAGLE_DEBUG_FLOW
                cudaDeviceSynchronize();
                std::cerr<<"Step2, F += IntegrationTmp, category = "<<category << ", j/m = "<<j<<"/"<<m_max<<", i = "<<i<< ", alpha = " << t / ((Real) s_max * j) <<"\n";
                std::cerr<<"destPLeft =\n" << destPLeft << "\n";
                std::cerr<<"FLeft = \n" << FLeft << "\n";

                // std::cerr<<"destPRight = \n" << destPRight << "\n";
                // std::cerr<<"FRight = \n" << FRight << "\n";

#endif
            }
            cudaDeviceSynchronize();
        }

        for (int category = 0; category < kCategoryCount; category++) {

            const int edgeMultiplierIndex1 = edgeIndex1 * kCategoryCount + category;
            const double edgeMultiplier1 = hEdgeMultipliers[edgeMultiplierIndex1];
            const Real eta1 = exp(etaConst1 * edgeMultiplier1);
            const int edgeMultiplierIndex2 = edgeIndex2 * kCategoryCount + category;
            const double edgeMultiplier2 = hEdgeMultipliers[edgeMultiplierIndex2];
            const Real eta2 = exp(etaConst2 * edgeMultiplier2);

            auto& FLeft = dFLeft[category];
            auto& FRight = dFRight[category];

            FLeft *= eta1;
            FRight *= eta2;

        }
        cudaDeviceSynchronize();

        for (int category = 0; category < kCategoryCount; category++) {


            auto& FLeft = dFLeft[category];
            auto& FRight = dFRight[category];

            auto& destPLeft = dDestPLeft[category];
            auto& destPRight = dDestPRight[category];

            destPLeft.copyFromAsync( FLeft );
            destPRight.copyFromAsync( FRight );
        }
        cudaDeviceSynchronize();
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

    std::vector<Triplet<Real>> tripletList;
    for (int i = 0; i < numNonZeros; i++) {
        tripletList.push_back(Triplet<Real>(rowIndices[i], colIndices[i], values[i]));
    }
    hInstantaneousMatrices[matrixIndex].setFromTriplets(tripletList.begin(), tripletList.end());

#ifdef BEAGLE_DEBUG_FLOW

    std::cerr<<"CPU instantaneous matrix "<<matrixIndex<<" =\n"<<hInstantaneousMatrices[matrixIndex]<<std::endl;
    std::cerr<<"outerIndexPtr = \n";
    for (int i = 0; i < kPaddedStateCount + 1; i++) {
        std::cerr<<hInstantaneousMatrices[matrixIndex].outerIndexPtr()[i]<<", ";
    }
    std::cerr<<"\ninnerIndexPtr = \n";
    for (int i = 0; i < hInstantaneousMatrices[matrixIndex].nonZeros(); i++) {
        std::cerr<<hInstantaneousMatrices[matrixIndex].innerIndexPtr()[i]<<", ";
    }
    std::cerr<<"\nvaluePtr = \n";
    for (int i = 0; i < hInstantaneousMatrices[matrixIndex].nonZeros(); i++) {
        std::cerr<<hInstantaneousMatrices[matrixIndex].valuePtr()[i]<<", ";
    }

#endif

    //TODO: use cusparse function for diagonal sum?
    Real mu_B = 0.0;
    for (int i = 0; i < kStateCount; i++) {
        mu_B += hInstantaneousMatrices[matrixIndex].coeff(i, i);
    }
    mu_B /= (Real) kStateCount;

    hMuBs[matrixIndex] = mu_B;
    hBs[matrixIndex] = hInstantaneousMatrices[matrixIndex] - mu_B * hIdentity;
    hB1Norms[matrixIndex] = normP1(hBs[matrixIndex]);

    const int currentNNZ = hBs[matrixIndex].nonZeros();
    if (currentCacheNNZs[matrixIndex] != currentNNZ) {
        currentCacheNNZs[matrixIndex] = currentNNZ;
        cudaDeviceDelete(dBsCsrColumnsCache[matrixIndex]);
        cudaDeviceDelete(dBsCsrValuesCache[matrixIndex]);
        dBsCsrColumnsCache[matrixIndex] = cudaDeviceNew<int>(currentNNZ);
        dBsCsrValuesCache[matrixIndex] = cudaDeviceNew<Real>(currentNNZ);
        for (int j = 0; j < kCategoryCount * 2; j++) {
            cudaDeviceDelete(dACscValuesCache[matrixIndex * kCategoryCount * 2 + j]);
            dACscValuesCache[matrixIndex * kCategoryCount * 2 + j] = cudaDeviceNew<Real>(currentNNZ);
        }
    }

    MemcpyHostToDevice(dBsCsrOffsetsCache[matrixIndex], hBs[matrixIndex].outerIndexPtr(), kPaddedStateCount + 1);
    MemcpyHostToDevice(dBsCsrColumnsCache[matrixIndex], hBs[matrixIndex].innerIndexPtr(), currentNNZ);
    MemcpyHostToDevice(dBsCsrValuesCache[matrixIndex], hBs[matrixIndex].valuePtr(), currentNNZ);

#ifdef BEAGLE_DEBUG_FLOW

    std::cerr<<"\ndBsCsrOffsetsCache for "<< matrixIndex <<"="<<std::endl;
    PrintfDeviceVector(dBsCsrOffsetsCache[matrixIndex], kPaddedStateCount + 1, -1, 0, 0);

    std::cerr<<"dBsCsrColumnsCache for "<< matrixIndex <<"="<<std::endl;
    PrintfDeviceVector(dBsCsrColumnsCache[matrixIndex], currentNNZ, -1, 0, 0);

    std::cerr<<"currentNNZ ="<< currentNNZ <<std::endl;

    std::cerr<<"dBsCsrValuesCache for "<< matrixIndex <<"="<<std::endl;
    PrintfDeviceVector(dBsCsrValuesCache[matrixIndex], currentNNZ, -1, 0, 0);
#endif


    hds[matrixIndex].clear();

    auto dB = SpMatrixDevice<Real>(cublasHandle, cusparseHandle,
                                   kPaddedStateCount, kPaddedStateCount,
                                   currentNNZ,
                                   dBsCsrValuesCache[matrixIndex],
                                   dBsCsrColumnsCache[matrixIndex],
                                   dBsCsrOffsetsCache[matrixIndex],
                                   sparseFormat::csr);

    int pMax = getPMax();
    auto tmp = hBs[matrixIndex];
#ifdef BEAGLE_DEBUG_NORMEST
    DnMatrix<Real> denseB = hBs[matrixIndex];
    DnMatrix<Real> denseBp = DnMatrix<Real>::Identity(kPaddedStateCount, kPaddedStateCount);
#endif
    for(int p=0;p <= pMax+1; p++)
    {
        // GPU Norm
        Real approx_norm = L1normForPower[p]( dB );

#ifdef BEAGLE_DEBUG_NORMEST
        // Compare with the CPU estimator and, for small S, the exact ||B^p||_1.
        Real approx_norm_CPU = normest1( hBs[matrixIndex], p, 5);
        double exact_norm = -1;
        if (kPaddedStateCount <= 4096)
        {
            if (p > 0) denseBp = denseB * denseBp;
            exact_norm = normP1(denseBp);
        }
        std::fprintf(stderr, "normest matrix=%d S=%d p=%d gpu=%.6e cpu=%.6e exact=%.6e gpu/exact=%.4f cpu/exact=%.4f\n",
                     matrixIndex, kPaddedStateCount, p, (double)approx_norm, (double)approx_norm_CPU, exact_norm,
                     approx_norm / exact_norm, approx_norm_CPU / exact_norm);
#endif

        // equation 3.7 in Al-Mohy and Higham
        hds[matrixIndex].push_back(pow(approx_norm, 1.0 / Real(p) ) );
    }

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
