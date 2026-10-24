/*
 *  BeagleGPUActionImpl.h
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


#ifndef BEAGLE_BEAGLEGPUACTIONIMPL_H
#define BEAGLE_BEAGLEGPUACTIONIMPL_H

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include "libhmsbeagle/GPU/BeagleGPUImpl.h"
#include <vector>
#include <random>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMV
#include <cublas_v2.h>
#include "cuda_ops.h"
#include <iostream>

using std::vector;
using std::tuple;
using Eigen::MatrixXi;

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

// cublas API error checking
#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                   \
            throw std::runtime_error("cublas error");                                              \
        }                                                                                          \
    } while (0)


template <typename T>
double normP1(const T& matrix) {
    return (Eigen::RowVectorXd::Ones(matrix.rows()) * matrix.cwiseAbs()).maxCoeff();
}

template <typename T>
tuple<double,int> ArgNormP1(const T& matrix)
{
    int x=-1;
    double v = matrix.colwise().template lpNorm<1>().maxCoeff(&x);
    return {v,x};
}

template <typename T>
auto normPInf(const T& matrix) {
    return matrix.template lpNorm<Eigen::Infinity>();
}

template <typename Real>
auto normPInf(Real* matrix, int nRows, int nCols)
{
    return cuda_max_abs(matrix, nRows * nCols);
}

template <typename Real>
using SpMatrix = Eigen::SparseMatrix<Real, Eigen::StorageOptions::RowMajor>;

template <typename Real>
using Triplet = Eigen::Triplet<Real>;

template <typename Real>
using DnMatrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;

template <typename Real>
using DnVector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;

template <typename T> cusparseIndexType_t IndexType;
template <> cusparseIndexType_t IndexType<int32_t> = CUSPARSE_INDEX_32I;
template <> cusparseIndexType_t IndexType<int64_t> = CUSPARSE_INDEX_64I;

template <typename T> cudaDataType DataType;
template <> cudaDataType DataType<float> = CUDA_R_32F;
template <> cudaDataType DataType<double> = CUDA_R_64F;

template <typename Real>
struct DnMatrixDevice
{
    cublasHandle_t cublasHandle = nullptr;
    // Lets assume any cusparseHandle_t comes from a sparse matrix.
    cusparseDnMatDescr_t descr = nullptr;
    Real* ptr = nullptr;
    size_t size1 = 0;
    size_t size2 = 0;
    cusparseOrder_t order = CUSPARSE_ORDER_COL;
    size_t size() const {return size1*size2;}
    size_t byte_size() const {return size()*sizeof(Real);}

    void copyFrom(const DnMatrixDevice<Real>& D)
    {
        assert(order == D.order);
        assert(size1 == D.size1);
        assert(size2 == D.size2);
        MemcpyDeviceToDevice(ptr, D.ptr, size());
    }

    void copyFromAsync(const DnMatrixDevice<Real>& D)
    {
        assert(order == D.order);
        assert(size1 == D.size1);
        assert(size2 == D.size2);
        MemcpyDeviceToDevice(ptr, D.ptr, size());
    }

    // Disallow copying -- only one object can "own" the descriptor.
    DnMatrixDevice<Real>& operator=(const DnMatrixDevice<Real>&) = delete;
    // Allow moving.
    DnMatrixDevice<Real>& operator=(DnMatrixDevice<Real>&& D) noexcept
    {
        std::swap(cublasHandle, D.cublasHandle);
        std::swap(descr, D.descr);
        std::swap(ptr, D.ptr);
        std::swap(size1, D.size1);
        std::swap(size2, D.size2);
        std::swap(order, D.order);
        return *this;
    }

    DnMatrixDevice<Real>& operator*=(Real d)
    {
        cublasStatus_t status;
        if constexpr (std::is_same<Real, float>::value) {
            status = cublasSscal(cublasHandle, size(), &d, ptr, 1);
        } else {
            status = cublasDscal(cublasHandle, size(), &d, ptr, 1);
        }

        if (status != CUBLAS_STATUS_SUCCESS)
        {
            std::cerr<<"cublas error "<<status<<" in DnMatrix<>::operator*=\n";
            exit(1);
        }
        return *this;
    }

    DnMatrixDevice<Real>& operator+=(const DnMatrixDevice<Real>& D)
    {
        cublasStatus_t status;
        assert(D.size1 == size1);
        assert(D.size2 == size2);
        assert(D.cublasHandle == cublasHandle);
        Real one = 1;
        if constexpr (std::is_same<Real, float>::value) {
            status = cublasSaxpy(cublasHandle, size(), &one, D.ptr, 1, ptr, 1);
        } else {
            status = cublasDaxpy(cublasHandle, size(), &one, D.ptr, 1, ptr, 1);
        }

        if (status != CUBLAS_STATUS_SUCCESS)
        {
            std::cerr<<"cublas error "<<status<<" in DnMatrix<>::operator+=\n";
            exit(1);
        }
        return *this;
    }

    // Disallow copying.
    DnMatrixDevice(const DnMatrixDevice<Real>&) = delete;
    // Allow moving.
    DnMatrixDevice(DnMatrixDevice<Real>&& D) noexcept
    {
        operator=(std::move(D));
    }

    DnMatrixDevice() = default;
    DnMatrixDevice(cublasHandle_t cb, Real* p, size_t s1, size_t s2, cusparseOrder_t o = CUSPARSE_ORDER_COL)
        :cublasHandle(cb), ptr(p), size1(s1), size2(s2), order(o)
    {
        auto status = (order == CUSPARSE_ORDER_COL)
            ? cusparseCreateDnMat(&descr, size1, size2, size1, ptr, DataType<Real>, CUSPARSE_ORDER_COL)
            : cusparseCreateDnMat(&descr, size1, size2, size2, ptr, DataType<Real>, CUSPARSE_ORDER_ROW);

        if (status != CUSPARSE_STATUS_SUCCESS)
        {
            std::cerr<<"cusparseCreateDnMat( ) failed!";
            std::exit(1);
        }
    }
    DnMatrixDevice(cublasHandle_t cb, size_t s1, size_t s2, cusparseOrder_t o = CUSPARSE_ORDER_COL)
        :DnMatrixDevice(cb, cudaDeviceNew<Real>(s1*s2), s1, s2, o)
    {
    }

    ~DnMatrixDevice()
     {
         // This class does not own the memory, so does not call cudaFree.
         if (descr)
         {
             cusparseDestroyDnMat(descr);
             descr = nullptr;
         }
     }
};

enum class sparseFormat { none, csr, csc };

template <typename Real>
struct SpMatrixDevice
{
    cublasHandle_t cublasHandle = nullptr;
    cusparseHandle_t cusparseHandle = nullptr;
    cusparseSpMatDescr_t descr = nullptr;
    int size1 = 0; // rows
    int size2 = 0; // columns
    int num_non_zeros = 0;
    Real* values = nullptr;
    int* inner = nullptr;
    int* offsets = nullptr;
    sparseFormat format = sparseFormat::none;

    int outer_dim_size() const
    {
	// With CSR, we return the number of row.
	if (format == sparseFormat::csr)
	    return size1;
	// With CSC, we return the number of columns;
	else if (format == sparseFormat::csc)
	    return size2;
	else
	    std::abort();
    }

    int rows() const {return size1;}
    int cols() const {return size2;}

    // Disallow copying -- only one object can "own" the descriptor.
    SpMatrixDevice<Real>& operator=(const SpMatrixDevice<Real>&) = delete;
    // Allow moving.
    SpMatrixDevice<Real>& operator=(SpMatrixDevice<Real>&& D) noexcept
    {
	std::swap(cublasHandle, D.cublasHandle);
	std::swap(cusparseHandle, D.cusparseHandle);
	std::swap(descr, D.descr);
	std::swap(size1, D.size1);
	std::swap(size2, D.size2);
	std::swap(num_non_zeros, D.num_non_zeros);
	std::swap(values, D.values);
	std::swap(inner, D.inner);
	std::swap(offsets, D.offsets);
	std::swap(format, D.format);
	return *this;
    }

    SpMatrixDevice<Real>& operator*=(Real d)
    {
        cublasStatus_t status;
        if constexpr (std::is_same<Real, float>::value) {
            status = cublasSscal(cublasHandle, num_non_zeros, &d, values, 1);
        } else {
            status = cublasDscal(cublasHandle, num_non_zeros, &d, values, 1);
        }

        if (status != CUBLAS_STATUS_SUCCESS)
        {
            std::cerr<<"cublas error "<<status<<" in SpMatrix<>::operator*=\n";
            exit(1);
        }
        return *this;
    }

   // Disallow copying.
    SpMatrixDevice(const SpMatrixDevice<Real>&) = delete;
    // Allow moving.
    SpMatrixDevice(SpMatrixDevice<Real>&& D) noexcept
    {
	operator=(std::move(D));
    }

    void setValues(Real *newValues) {
        values = newValues;
    }

    SpMatrixDevice() = default;
    SpMatrixDevice(cublasHandle_t h1, cusparseHandle_t h2, int s1, int s2, int n, Real* v, int* c, int* o, sparseFormat f)
	:cublasHandle(h1), cusparseHandle(h2), size1(s1), size2(s2), num_non_zeros(n), values(v), inner(c), offsets(o), format(f)
    {
	cusparseStatus_t status;
	if (format == sparseFormat::csc)
	{
	    status = cusparseCreateCsc(&descr, s1, s2, num_non_zeros, offsets, inner, values,
				       IndexType<int>, IndexType<int>, CUSPARSE_INDEX_BASE_ZERO, DataType<Real>);
	}
	else
	{
	    status = cusparseCreateCsr(&descr, s1, s2, num_non_zeros, offsets, inner, values,
				       IndexType<int>, IndexType<int>, CUSPARSE_INDEX_BASE_ZERO, DataType<Real>);
	}

	if (status != CUSPARSE_STATUS_SUCCESS)
	{
	    std::cerr<<"cusparseCreateSpMat( ) failed!";
	    std::exit(1);
	}
    }

    ~SpMatrixDevice()
    {
	if (descr)
	{
	    cusparseDestroySpMat(descr);
	    descr = nullptr;
	}
    }
};

template <typename Real>
void dotProduct(Real* out, cublasHandle_t handle, int n, Real* v1, Real* v2)
{
    if constexpr (std::is_same<Real, float>::value)
    {
        cublasSdot(handle, n, v1, 1, v2, 1, out);
    }
    else
    {
        cublasDdot(handle, n, v1, 1, v2, 1, out);
    }
};

template <typename Real>
auto normPInf(const DnMatrixDevice<Real>& M)
{
    return normPInf(M.ptr, M.size1, M.size2);
}

template <typename Real>
int spMM(cusparseHandle_t handle, cusparseDnMatDescr_t C, Real alpha, cusparseSpMatDescr_t A, cusparseDnMatDescr_t B, Real beta, void*& buffer, size_t& buffersize)
{
    size_t new_buffersize;
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, A, B, &beta, C, DataType<Real>,
                                           CUSPARSE_SPMM_ALG_DEFAULT, &new_buffersize));

    if(new_buffersize > buffersize)
    {
        cudaDeviceDelete(buffer);
        buffer = cudaMallocWrapped(new_buffersize);
        buffersize = new_buffersize;
    }

    // integrationTmp = alpha * A * destP
    CHECK_CUSPARSE(cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, A, B, &beta, C, DataType<Real>,
                                CUSPARSE_SPMM_ALG_DEFAULT, buffer)); //row-major layout provides higher performance (?)
    return 0;
}

template <typename Real>
int spMM(DnMatrixDevice<Real>& C, Real alpha, const SpMatrixDevice<Real>& A, const DnMatrixDevice<Real>& B, Real beta, void*& buffer, size_t& buffersize)
{
    return spMM<Real>(A.cusparseHandle, C.descr, alpha, A.descr, B.descr, beta, buffer, buffersize);
}

template <typename Real>
int spMTM(cusparseHandle_t handle, cusparseDnMatDescr_t C, Real alpha, cusparseSpMatDescr_t A, cusparseDnMatDescr_t B, Real beta, void*& buffer, size_t& buffersize)
{
    size_t new_buffersize;
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, A, B, &beta, C, DataType<Real>,
                                           CUSPARSE_SPMM_ALG_DEFAULT, &new_buffersize));

    if(new_buffersize > buffersize)
    {
        cudaDeviceDelete(buffer);
        buffer = cudaMallocWrapped(new_buffersize);
        buffersize = new_buffersize;
    }

    // integrationTmp = alpha * A * destP
    CHECK_CUSPARSE(cusparseSpMM(handle, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, A, B, &beta, C, DataType<Real>,
                                CUSPARSE_SPMM_ALG_DEFAULT, buffer)); //row-major layout provides higher performance (?)
    return 0;
}

template <typename Real>
int spMTM(DnMatrixDevice<Real>& C, Real alpha, const SpMatrixDevice<Real>& A, const DnMatrixDevice<Real>& B, Real beta, void*& buffer, size_t& buffersize)
{
    return spMTM<Real>(A.cusparseHandle, C.descr, alpha, A.descr, B.descr, beta, buffer, buffersize);
}

// Create an io-manipulator so that we can write std::cerr<<byRow(D)<<"\n";
template <typename Real>
class byRowDense
{
    const DnMatrixDevice<Real>& D;
    bool multiline = false;
public:

    friend std::ostream& operator<<(std::ostream& o, const byRowDense<Real>& pr)
    {
        auto& D = pr.D;
	bool multiline = pr.multiline;

        auto hPtr = MemcpyDeviceToHostVector(D.ptr, D.size());

        if (not multiline) o<<"Rows [";
        for(int row=0;row<D.size1;row++)
        {
            if (not multiline) o<<" ";
            for(int col=0;col<D.size2;col++)
            {
                if (D.order == CUSPARSE_ORDER_COL)
                    o<<hPtr[col*D.size1 + row]<<", ";
                else
                    o<<hPtr[row*D.size2 + col]<<", ";
            }
	    if (not multiline)
		o<<"; ";
	    else if (row+1 != D.size1)
		o<<"\n";
        }
        if (not multiline) o<<"]";

	if (D.order == CUSPARSE_ORDER_COL)
	    o<<" (column-major)";
	else
	    o<<" (row-major)";

	if (multiline) o<<"\n";
        return o;
    }

    // Initialize the forwarding struct from the matrix that we want to print.
    byRowDense(const DnMatrixDevice<Real>& d, bool m=false):D(d),multiline(m) {}
};

// Create an io-manipulator so that we can write std::cerr<<byCol(D)<<"\n";
template <typename Real>
class byColDense
{
    const DnMatrixDevice<Real>& D;
public:

    friend std::ostream& operator<<(std::ostream& o, const byColDense<Real>& pr)
    {
	auto& D = pr.D;

        auto hPtr = MemcpyDeviceToHostVector(D.ptr, D.size());

        o<<"Cols[ ";
        for(int col=0;col<D.size2;col++)
        {
            o<<"[ ";
            for(int row=0;row<D.size1;row++)
            {
                if (D.order == CUSPARSE_ORDER_COL)
                    o<<hPtr[col*D.size1 + row]<<", ";
                else
                    o<<hPtr[row*D.size2 + col]<<", ";
            }
            o<<"] ";
        }
        o<<"]";

	if (D.order == CUSPARSE_ORDER_COL)
	    o<<" (column-major)";
	else
	    o<<" (row-major)";

        return o;
    }

    // Initialize the forwarding struct from the matrix that we want to print.
    byColDense(const DnMatrixDevice<Real>& d):D(d) {}
};

std::ostream& operator<<(std::ostream& o, sparseFormat f)
{
    if (f == sparseFormat::csr)
	o<<"csr";
    else if (f == sparseFormat::csc)
	o<<"csc";
    else
	o<<"sparseFormat=unknown";
    return o;
}
	
// Create an io-manipulator so that we can write std::cerr<<byRow(D)<<"\n";
template <typename Real>
class byRowSparse
{
    const SpMatrixDevice<Real>& S;
    bool multiline = false;
public:

    friend std::ostream& operator<<(std::ostream& o, const byRowSparse<Real>& pr)
    {
        auto& S = pr.S;
	bool multiline = pr.multiline;

	auto values = MemcpyDeviceToHostVector(S.values, S.num_non_zeros);
	auto inner = MemcpyDeviceToHostVector(S.inner, S.num_non_zeros);
	auto offsets = MemcpyDeviceToHostVector(S.offsets, S.outer_dim_size() + 1);
	
	std::vector< std::vector<Real> > D(S.size1, std::vector<Real>(S.size2, 0));
        for(int o=0; o< S.outer_dim_size();o++)
        {
            for(int index=offsets[o];index<offsets[o+1];index++)
            {
                if (S.format == sparseFormat::csr)
		    D[o][inner[index]] = values[index];
		else
		    D[inner[index]][o] = values[index];
            }
        }

        if (not multiline) o<<"Rows [";
	for(int row=0;row<D.size();row++)
	{
            if (not multiline) o<<" ";
	    for(int col=0;col<D[row].size();col++)
	    {
		o<<D[row][col]<<", ";
            }
            if (not multiline)
		o<<"; ";
	    else if (row+1 != D.size())
		o<<"\n";
        }
        if (not multiline) o<<"]";

	o<<" ("<<S.format<<")";
	if (multiline) o<<"\n";

        return o;
    }

    // Initialize the forwarding struct from the matrix that we want to print.
    byRowSparse(const SpMatrixDevice<Real>& s, bool m = false):S(s),multiline(m) {}
};

template <typename Real>
byRowDense<Real> byRow(const DnMatrixDevice<Real>& D, bool m = false)
{
    return byRowDense<Real>(D, m);
}

template <typename Real>
byRowSparse<Real> byRow(const SpMatrixDevice<Real>& D, bool m = false)
{
    return byRowSparse<Real>(D, m);
}

template <typename Real>
byColDense<Real> byCol(const DnMatrixDevice<Real>& D)
{
    return byColDense<Real>(D);
}

template <typename Real>
std::ostream& operator<<(std::ostream& o, const DnMatrixDevice<Real>& D)
{
    return o<<byRow(D, true);
}

template <typename Real>
std::ostream& operator<<(std::ostream& o, const SpMatrixDevice<Real>& D)
{
    return o<<byRow(D, true);
}


template <typename Real>
struct asDeviceVec
{
    const Real* data;
    int length;

    asDeviceVec(const Real* d, int l):data(d),length(l) { }
};

template <typename Real>
std::ostream& operator<<(std::ostream& o, const asDeviceVec<Real>& D)
{
    auto H = MemcpyDeviceToHostVector(D.data, D.length);

    o<<"[ ";
    for(auto& h: H)
	o<<h<<" ";
    o<<"]";

    return o;
}

template <typename T>
std::ostream& operator<<(std::ostream& o, const std::vector<T>& H)
{
    o<<"[ ";
    for(auto& h: H)
	o<<h<<" ";
    o<<"]";

    return o;
}




// Make the temporary variables part of a data structure that represents to normest1 problem.
// We don't want to allocate memory on the GPU every time we run this.
template <typename Real>
struct GPUnormest1
{
    int p = 0;
    int n = 0;
    int t = 0;
    int itmax = 0;
    DnMatrixDevice<Real> X;  // (n,t)
    DnMatrixDevice<Real> Y;  // (n,t)
    DnMatrixDevice<Real> h;  // (n,1)

    // Sorted matrix dimensions by approximate norm.
    int* indices = nullptr;  // (n)

    // Used for spMM
    void* buffer = nullptr;
    size_t buffer_size = 0;

    // Temporary storage for cuda_max_l1_norm
    Real* buffer2 = nullptr;

    // Space to store the norms for the different iterations.
    Real* buffer3 = nullptr;

    // Space to store a result
    Real* buffer4 = nullptr;

    GPUnormest1& operator=(const GPUnormest1&) = delete;
    GPUnormest1& operator=(GPUnormest1&& g)
    {
        std::swap(p, g.p);
        std::swap(n, g.n);
        std::swap(t, g.t);
        std::swap(itmax, g.itmax);
        std::swap(X, g.X);
        std::swap(Y, g.Y);
        std::swap(h, g.h);

        // ensure that (*this) and (g) don't both own the buffers!
        std::swap(buffer, g.buffer);
        std::swap(buffer_size, g.buffer_size);
        std::swap(indices, g.indices);
        std::swap(buffer2, g.buffer2);
        std::swap(buffer3, g.buffer3);
        std::swap(buffer4, g.buffer4);

        return *this;
    }

    Real operator()(const SpMatrixDevice<Real>& A)
    {
        // std::cerr<<"n = "<<n<<"   t = "<<t<<"\n";
        // std::cerr<<"A.rows() = "<<A.rows()<<"\n";
        // A is (n,n);
        assert(A.rows() == A.cols());
        assert(A.cols() == n);
        //std::cerr<<"A = "<<byRow(A)<<"\n";

        // X[0][j] = 1
        // X[i][j] = if (uniform(0,1)<0.5) +1 else -1
        initialize_norm_x_matrix(X.ptr, n, t);

        double norm = 0;
        for(int k=1; k<=itmax; k++)
        {
            // std::cerr<<"iter "<<k<<"\n";
            // std::cerr<<"X0 = "<<byRow(X)<<"\n\n";

            for(int i=0;i<p;i++)
            {
                // Y = A*X; // Y is (n,t) = (n,n) * (n,t)
                spMM<Real>(Y, 1, A, X, 0, buffer, buffer_size);
                // X = Y
                X.copyFrom(Y);
            }

            // get largest L1 norm
            double new_norm = cuda_max_l1_norm(X.ptr, n, t, buffer2);
            if (new_norm <= norm and k >= 2)
                break;

            norm = std::max(norm,new_norm);
            // std::cerr<<"A^p*X = "<<byRow(X)<<"   norm = "<<norm<<"\n\n";

            // S = sign(X)
            cuda_sign_vector(X.ptr, n, t);   // (n,n) -> (n,n)

            // (2) Replace parallel entries with random +1/-1 entries.
            // Skipping this part for now because it involves a lot of branching logic
            //  and might be slow for the GPU.

            // (3) of Algorithm 2.4
            // Z = A^T * S
            spMTM<Real>(Y, 1, A, X, 0, buffer, buffer_size);  // (n,n) * (n,t) -> (n,t)

            // h[i] = max(j) abs(Z(i,j))
            cuda_rowwise_max_abs(Y.ptr, n, t, h.ptr);  // (n,t) -> (n,1)

            // (4) of Algorithm 2.4 - If we don't find a new best dimension, exit early.
            // We don't do this, because finding a different reason to exit
            // seems to provide greater accuracy.

            // Rank the dimensions by the size of their approximate norm in h, in decreasing order.
            cuda_sort_indices_by_vector(h.ptr, n, indices);

            // Zero the X matrix.
            cuda_fill_vector(X.ptr, n*t, 0);  // (n,t) -> (n,t)

            // Set X(i,indices[i]) = 1 for i in [0,t-1]
            cuda_set_indices(X.ptr, n, t, indices);
        }

        return norm;
    }

    GPUnormest1(GPUnormest1&& g)
    {
        operator=(std::move(g));
    }

    GPUnormest1(const GPUnormest1&) = delete;

    GPUnormest1(cublasHandle_t cb, int p_, int n_, int t_=2, int itmax_=5)
        :p(p_), n(n_), t(std::min(t_,n)), itmax(itmax_), X(cb,n,t), Y(cb,n,t), h(cb,n,1)
    {
        assert(p >= 0);
        assert(t > 0); // negative means t = n
        assert(itmax >= 1);

        // Interpret negative t as t == n
        // if (t < 0) t = n;

        buffer2 = cudaDeviceNew<Real>(t);

        buffer3 = cudaDeviceNew<Real>(itmax);

        buffer4 = cudaDeviceNew<Real>(1);

        indices = cudaDeviceNew<int>(n);
    }

    ~GPUnormest1()
    {
        cudaDeviceDelete(buffer);
        cudaDeviceDelete(buffer2);
        cudaDeviceDelete(buffer3);
        cudaDeviceDelete(buffer4);
        cudaDeviceDelete(indices);
    }
};

// Make the temporary variables part of a data structure that represents to normest1 problem.
// We don't want to allocate memory on the GPU every time we run this.
template <typename Real>
struct GPUnormest2
{
    int p = 0;
    int n = 0;
    Real const one = Real(1);
    Real const zero = Real(0);


    DnMatrixDevice<Real> Ones;  // (n,1)

    DnMatrixDevice<Real> Y;

    // Used for spMM
    void* buffer = nullptr;
    size_t buffer_size = 0;

    Real *valuesBuffer = nullptr;


    GPUnormest2& operator=(const GPUnormest2&) = delete;
    GPUnormest2& operator=(GPUnormest2&& g)
    {
        std::swap(p, g.p);
        std::swap(n, g.n);
        std::swap(Ones, g.Ones);
        std::swap(Y, g.Y);

        // ensure that (*this) and (g) don't both own the buffers!
        std::swap(buffer, g.buffer);
        std::swap(buffer_size, g.buffer_size);
        std::swap(valuesBuffer, g.valuesBuffer);

        return *this;
    }

    Real operator()(const SpMatrixDevice<Real>& A)
    {
        // std::cerr<<"n = "<<n<<"   t = "<<t<<"\n";
        // std::cerr<<"A.rows() = "<<A.rows()<<"\n";
        // A is (n,n);
        assert(A.rows() == A.cols());
        assert(A.cols() == n);
        //std::cerr<<"A = "<<byRow(A)<<"\n";

        // Step 1, Aabs = abs(A)
        MemcpyDeviceToDevice(valuesBuffer, A.values, A.num_non_zeros);
        cuda_vec_abs(A.values, A.num_non_zeros, A.values);

        // std::cerr<<"A = "<<A<<std::endl;
        // std::cerr<<"Ones = "<<Ones<<std::endl;

        // Step 2, calculate column sum

        spMTM<Real>(Y, 1, A, Ones, 0, buffer, buffer_size);

        // std::cerr<<"Y = "<<Y<<std::endl;

        // Step 3, find max
        int index = 0;
        double norm;

        cublasIdamax(A.cublasHandle, n, Y.ptr, one, &index);
        // std::cerr<<"index = "<<index<<std::endl;
        // std::cerr<<"Y = "<<Y<<std::endl;

        cudaMemcpy(&norm, Y.ptr + index -1, sizeof(Real), cudaMemcpyDeviceToHost);
        MemcpyDeviceToDevice(A.values, valuesBuffer, A.num_non_zeros);
        // std::cerr<<"A = "<<A<<std::endl;

        return norm;
    }


    GPUnormest2(GPUnormest2&& g)
    {
        operator=(std::move(g));
    }

    GPUnormest2(const GPUnormest2&) = delete;

    // SpMatrixDevice(cublasHandle_t h1, cusparseHandle_t h2, int s1, int s2, int n, Real* v, int* c, int* o, sparseFormat f)

    GPUnormest2(cublasHandle_t cb, cusparseHandle_t cs, int p_, int n_)
        :p(p_), n(n_), Y(cb, n, 1), Ones(cb, n, 1, CUSPARSE_ORDER_COL)
    {
        assert(p >= 0);

        cuda_vec_fill(Ones.ptr, n, 1);

        valuesBuffer = cudaDeviceNew<Real>(n * n);

    }

    ~GPUnormest2()
    {
        cudaDeviceDelete(buffer);
        cudaDeviceDelete(valuesBuffer);
        cudaDeviceDelete(Ones.ptr);
        cudaDeviceDelete(Y.ptr);
    }
};

// Make the temporary variables part of a data structure that represents to normest1 problem.
// We don't want to allocate memory on the GPU every time we run this.
template <typename Real>
struct GPUnormest3
{
    int p = 0;
    int n = 0;
    int t = 0;
    int itmax = 0;
    DnMatrixDevice<Real> X;  // (n,t)
    DnMatrixDevice<Real> Y;  // (n,t)
    DnMatrixDevice<Real> h;  // (n,1)

    // Sorted matrix dimensions by approximate norm.
    int* indices = nullptr;  // (n)

    // Used for spMM
    void* buffer = nullptr;
    size_t buffer_size = 0;

    // Temporary storage for cuda_max_l1_norm
    Real* buffer2 = nullptr;

    // Space to store the norms for the different iterations.
    Real* buffer3 = nullptr;

    // Space to store a result
    Real* buffer4 = nullptr;

    cuda_scratch_space scratch;

    int* offsets = nullptr;

    // Rows already used as unit columns of X (ind_hist in Algorithm 2.4).
    int* visited = nullptr;  // (n)

    GPUnormest3& operator=(const GPUnormest3&) = delete;
    GPUnormest3& operator=(GPUnormest3&& g)
    {
        std::swap(p, g.p);
        std::swap(n, g.n);
        std::swap(t, g.t);
        std::swap(itmax, g.itmax);
        std::swap(X, g.X);
        std::swap(Y, g.Y);
        std::swap(h, g.h);

        // ensure that (*this) and (g) don't both own the buffers!
        std::swap(buffer, g.buffer);
        std::swap(buffer_size, g.buffer_size);
        std::swap(indices, g.indices);
        std::swap(buffer2, g.buffer2);
        std::swap(buffer3, g.buffer3);
        std::swap(buffer4, g.buffer4);
        std::swap(scratch.device_ptr, g.scratch.device_ptr);
        std::swap(scratch.size, g.scratch.size);
        std::swap(offsets, g.offsets);
        std::swap(visited, g.visited);

        return *this;
    }

    Real operator()(const SpMatrixDevice<Real>& A)
    {
        // std::cerr<<"n = "<<n<<"   t = "<<t<<"\n";
        // std::cerr<<"A.rows() = "<<A.rows()<<"\n";
        // A is (n,n);
        assert(A.rows() == A.cols());
        assert(A.cols() == n);
        //std::cerr<<"A = "<<byRow(A)<<"\n";

        // X[0][j] = 1
        // X[i][j] = if (uniform(0,1)<0.5) +1 else -1
        initialize_norm_x_matrix(X.ptr, n, t);

        cudaMemset(visited, 0, n * sizeof(int));

        for(int k=1; k<=itmax; k++)
        {
            // std::cerr<<"iter "<<k<<"\n";
            // std::cerr<<"X0 = "<<byRow(X)<<"\n\n";

            for(int i=0;i<p;i++)
            {
                // Y = A*X; // Y is (n,t) = (n,n) * (n,t)
                spMM<Real>(Y, 1, A, X, 0, buffer, buffer_size);
                // X = Y
                X.copyFrom(Y);
            }

            // get the largest column L1 norm of A^p * X for this iteration -> buffer3[k-1]
            cuda_max_l1_norm(X.ptr, n, t, buffer2, buffer3 + (k-1), scratch, offsets);

            // The rest only chooses X for the next iteration.
            if (k == itmax) break;

            // S = sign(X)
            cuda_sign_vector(X.ptr, n, t);   // (n,t) -> (n,t)

            // (2) Replace parallel entries with random +1/-1 entries.
            // Skipping this part for now because it involves a lot of branching logic
            //  and might be slow for the GPU.

            // (3) of Algorithm 2.4
            // Z = (A^T)^p * S
            for(int i=0;i<p;i++)
            {
                spMTM<Real>(Y, 1, A, X, 0, buffer, buffer_size);  // (n,n) * (n,t) -> (n,t)
                X.copyFrom(Y);
            }

            // h[i] = max(j) abs(Z(i,j))
            cuda_rowwise_max_abs(X.ptr, n, t, h.ptr);  // (n,t) -> (n,1)

            // (4) of Algorithm 2.4 - If we don't find a new best dimension, exit early.
            // We don't do this, because finding a different reason to exit
            // seems to provide greater accuracy.

            // (5) of Algorithm 2.4 - Only consider rows that have not been used as unit columns yet.
            //  (If fewer than t remain, used rows are picked again, which only repeats work.)
            cuda_mask_visited(h.ptr, visited, n);

            // Rank the dimensions by the size of their approximate norm in h, in decreasing order.
            cuda_sort_indices_by_vector(h.ptr, n, indices);

            // Zero the X matrix.
            cuda_fill_vector(X.ptr, n*t, 0);  // (n,t) -> (n,t)

            // Set X(indices[j],j) = 1 for j in [0,t-1]
            cuda_set_indices(X.ptr, n, t, indices);
            cuda_mark_visited(visited, indices, t);
        }

        return cuda_max_abs(buffer3, itmax);
    }

    GPUnormest3(GPUnormest3&& g)
    {
        operator=(std::move(g));
    }

    GPUnormest3(const GPUnormest3&) = delete;

    GPUnormest3(cublasHandle_t cb, int p_, int n_, int t_=2, int itmax_=5)
        :p(p_), n(n_), t(std::min(t_,n)), itmax(itmax_), X(cb,n,t), Y(cb,n,t), h(cb,n,1)
    {
        assert(p >= 0);
        assert(t > 0); // negative means t = n
        assert(itmax >= 1);

        // Interpret negative t as t == n
        // if (t < 0) t = n;

        buffer2 = cudaDeviceNew<Real>(t);

        buffer3 = cudaDeviceNew<Real>(itmax);

        buffer4 = cudaDeviceNew<Real>(1);

        indices = cudaDeviceNew<int>(n);

        offsets = cudaDeviceNew<int>(t+1);

        visited = cudaDeviceNew<int>(n);

        std::vector<int> h_offsets(t+1);
        for(int i=0;i<t+1;i++)
            h_offsets[i] = i*n;
        MemcpyHostToDevice(offsets, h_offsets.data(), h_offsets.size());
    }

    ~GPUnormest3()
    {
        cudaDeviceDelete(X.ptr);
        cudaDeviceDelete(Y.ptr);
        cudaDeviceDelete(h.ptr);
        cudaDeviceDelete(buffer);
        cudaDeviceDelete(buffer2);
        cudaDeviceDelete(buffer3);
        cudaDeviceDelete(buffer4);
        cudaDeviceDelete(indices);
        cudaDeviceDelete(offsets);
        cudaDeviceDelete(visited);
    }
};



//template <typename Real>
//using MapType = DnMatrix<Real>;

namespace beagle {
namespace gpu {

#ifdef CUDA
	namespace cuda {
#else
	namespace opencl {
#endif

BEAGLE_GPU_TEMPLATE
class BeagleGPUActionImpl : public BeagleGPUImpl<BEAGLE_GPU_GENERIC>
{
public:
    const char* getName();

    long long getFlags();

    int getInstanceDetails(BeagleInstanceDetails* retunInfo);

    int createInstance(int tipCount,
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
		       long long requirementFlags);

    int setTipStates(int tipIndex, const int* inStates);

    int setTipPartials(int tipIndex,
                       const Real* inPartials);

    int setPartials(int bufferIndex,
                    const Real* inPartials);

    int setSparseMatrix(int matrixIndex,
                        const int* rowIndices,
                        const int* colIndices,
                        const Real* values,
                        int numNonZeros);

    int updatePartials(const int* operations,
                       int operationCount,
                       int cumulativeScalingIndex);

    void rescalePartials(Real* partials, Real* scalingFactors, Real* cumulativeScalingBuffer, int streamIndex);

    // Action instances have no dense transition matrices (see createInstance), so these are not available.
    // (The base versions of the first three call the dense, non-virtual upPartials/upPrePartials.)
    int updatePrePartials(const int*, int, int) override { return BEAGLE_ERROR_NO_IMPLEMENTATION; }
    int updatePartialsByPartition(const int*, int) override { return BEAGLE_ERROR_NO_IMPLEMENTATION; }
    int updatePrePartialsByPartition(const int*, int) override { return BEAGLE_ERROR_NO_IMPLEMENTATION; }
    int setEigenDecomposition(int, const double*, const double*, const double*) override { return BEAGLE_ERROR_NO_IMPLEMENTATION; }
    int setTransitionMatrix(int, const double*, double) override { return BEAGLE_ERROR_NO_IMPLEMENTATION; }
    int setTransitionMatrices(const int*, const double*, const double*, int) override { return BEAGLE_ERROR_NO_IMPLEMENTATION; }
    int getTransitionMatrix(int, double*) override { return BEAGLE_ERROR_NO_IMPLEMENTATION; }
    int convolveTransitionMatrices(const int*, const int*, const int*, int) override { return BEAGLE_ERROR_NO_IMPLEMENTATION; }
    int transposeTransitionMatrices(const int*, const int*, int) override { return BEAGLE_ERROR_NO_IMPLEMENTATION; }
    int updateTransitionMatricesWithModelCategories(int*, const int*, const int*, const int*, const double*, int) override { return BEAGLE_ERROR_NO_IMPLEMENTATION; }
    int updateTransitionMatricesWithMultipleModels(const int*, const int*, const int*, const int*, const int*, const double*, int) override { return BEAGLE_ERROR_NO_IMPLEMENTATION; }
    int calculateEdgeLogLikelihoods(const int*, const int*, const int*, const int*, const int*, const int*, const int*, const int*,
                                    int, double*, double*, double*) override { return BEAGLE_ERROR_NO_IMPLEMENTATION; }
    int calculateEdgeLogLikelihoodsByPartition(const int*, const int*, const int*, const int*, const int*, const int*, const int*, const int*,
                                               const int*, int, int, double*, double*, double*, double*, double*, double*) override { return BEAGLE_ERROR_NO_IMPLEMENTATION; }
    int calculateEdgeDerivatives(const int*, const int*, const int*, const int*, const int*, const int*,
                                 int, double*, double*, double*) override { return BEAGLE_ERROR_NO_IMPLEMENTATION; }

protected:

    std::vector<SpMatrix<Real>> hInstantaneousMatrices;
    std::vector<SpMatrix<Real>> hBs;
    SpMatrix<Real> hIdentity;
    std::vector<Real> hMuBs;
    std::vector<Real> hB1Norms;
    const int mMax = 55;
    std::vector<std::vector<Real>> hds;
    std::map<int, double> thetaConstants = {
            //The first 30 values are from table A.3 of  Computing Matrix Functions.
            // For double precision, tol = 2^(-53)
            // TODO: maybe calculate this
            {1, 2.29E-16},
            {2, 2.58E-8},
            {3, 1.39E-5},
            {4, 3.40E-4},
            {5, 2.40E-3},
            {6, 9.07E-3},
            {7, 2.38E-2},
            {8, 5.00E-2},
            {9, 8.96E-2},
            {10, 1.44E-1},
            {11, 2.14E-1},
            {12, 3.00E-1},
            {13, 4.00E-1},
            {14, 5.14E-1},
            {15, 6.41E-1},
            {16, 7.81E-1},
            {17, 9.31E-1},
            {18, 1.09},
            {19, 1.26},
            {20, 1.44},
            {21, 1.62},
            {22, 1.82},
            {23, 2.01},
            {24, 2.22},
            {25, 2.43},
            {26, 2.64},
            {27, 2.86},
            {28, 3.08},
            {29, 3.31},
            {30, 3.54},
            //The rest are from table 3.1 of Computing the Action of the Matrix Exponential.
            {35, 4.7},
            {40, 6.0},
            {45, 7.2},
            {50, 8.5},
            {55, 9.9},
    };

//    std::vector<cusparseSpMatDescr_t> dInstantaneousMatrices;
    std::vector<DnMatrixDevice<Real>> dPartialsWrapper;
    // Per category: the action of each child's edge on its partials (exp(tQ) * partials).
    std::vector<DnMatrixDevice<Real>> dDestPLeft;
    std::vector<DnMatrixDevice<Real>> dDestPRight;
    std::vector<DnMatrixDevice<Real>> dFLeft;
    std::vector<DnMatrixDevice<Real>> dFRight;
    std::vector<DnMatrixDevice<Real>> dIntegrationTmpLeft;
    std::vector<DnMatrixDevice<Real>> dIntegrationTmpRight;
    std::vector<SpMatrixDevice<Real>> dAs;
    std::vector<GPUnormest3<Real>> L1normForPower;
    std::vector<size_t> integrationLeftBufferSize;
    std::vector<void*> dIntegrationLeftBuffer;
    std::vector<size_t> integrationRightBufferSize;
    std::vector<void*> dIntegrationRightBuffer;

    std::vector<int *> dBsCsrOffsetsCache;
    std::vector<int *> dBsCsrColumnsCache;
    std::vector<Real*> dBsCsrValuesCache;
    std::vector<Real*> dACscValuesCache;
    std::vector<int> currentCacheNNZs;

    cublasHandle_t cublasHandle;
    cusparseHandle_t cusparseHandle;

    std::vector<int> hEigenMaps;
    std::vector<Real> hEdgeMultipliers;

    std::vector<tuple<int, int>> msCache;
    std::vector<Real> etaCache;
    std::vector<Real> c1Cache;
    std::vector<Real> c2Cache;
    std::vector<Real> alphaCache;
    std::vector<Real> integrationMultipliers;


    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kTipCount;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kPartialsBufferCount;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kStateCount;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kEigenDecompCount;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kCategoryCount;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kMatrixCount;


    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kPaddedStateCount;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::gpu;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kPaddedPatternCount;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kFlags;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kMatrixSize;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kernels;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::dScalingFactors;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::dPartials;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::dPartialsOrigin;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::hCategoryRates;

    // For calculateRootLogLikelihoods
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::dAccumulatedScalingFactors;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::dIntegrationTmp;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::dWeights;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::dFrequencies;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::dSumLogLikelihood;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::dPatternWeights;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::hLogLikelihoodsCache;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kSumSitesBlockCount;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kPatternCount;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kScaleBufferSize;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::dPtrQueue;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::hPtrQueue;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::dMaxScalingFactors;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::dIndexMaxScalingFactors;

    int upPartials(bool byPartition,
		   const int *operations,
		   int operationCount,
		   int cumulativeScalingIndex);

    int upPrePartials(bool byPartition,
		      const int *operations,
		      int operationCount,
		      int cumulativeScalingIndex);

    int calculateRootLogLikelihoods(const int* bufferIndices,
                                    const int* categoryWeightsIndices,
                                    const int* stateFrequenciesIndices,
                                    const int* cumulativeScaleIndices,
                                    int count,
                                    double* outSumLogLikelihood);

    ~BeagleGPUActionImpl();

private:
    char* getInstanceName();

    int updateTransitionMatrices(int eigenIndex,
				 const int* probabilityIndices,
				 const int* firstDerivativeIndices,
				 const int* secondDerivativeIndices,
				 const Real* edgeLengths,
				 int count);
    double getPMax() const;

    // Make CUDA runtime, cuBLAS, cuSPARSE and Thrust calls target this instance's device.
    void setCudaDevice();

    int getPartialIndex(int nodeIndex, int categoryIndex);

    DnMatrixDevice<Real>& getPartialsWrapper(int nodeIndex, int categoryIndex);

    void calcPartialsPartials(int destPIndex,
                              int partials1Index,
                              int edgeIndex1,
                              int partials2Index,
                              int edgeIndex2);

    int simpleAction2(DnMatrixDevice<Real>& destP, const DnMatrixDevice<Real>& inPartials, int edgeIndex, int category, int matrixIndex, bool left, bool transpose);

    int simpleAction3(int partialsIndex1, int edgeIndex1,
                      int partialsIndex2, int edgeIndex2);

    int cacheAMatrices(int edgeIndex1, int edgeIndex2, bool transpose);

    std::tuple<int,int> getStatistics2(double t, int nCol, double edgeMultiplier,
                                       int eigenIndex) const;

    double getDValue(int p, int eigenIndex) const;

    int PrintfDeviceVector(Real* dPtr, int length, double checkValue, int *signal, Real r);
    int PrintfDeviceVector(int* dPtr, int length, double checkValue, int *signal, Real r);
    int PrintfDeviceVector(cusparseDnMatDescr_t dPtr, int length, double checkValue, int *signal, Real r);

};

BEAGLE_GPU_TEMPLATE
class BeagleGPUActionImplFactory : public BeagleGPUImplFactory<BEAGLE_GPU_GENERIC> {
    virtual BeagleImpl* createImpl(int tipCount,
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
                                   int* errorCode);

    virtual const char* getName();
    virtual long long getFlags();
};

} // namspace device
}	// namespace gpu
}	// namespace beagle

#include "libhmsbeagle/GPU/BeagleGPUActionImpl.hpp"

#endif

