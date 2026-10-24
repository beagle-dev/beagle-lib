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

// A CSR matrix on the host.
template <typename Real>
struct HostCsr
{
    std::vector<int> offsets;
    std::vector<int> columns;
    std::vector<Real> values;

    int nonZeros() const {return (int) values.size();}
};

// B = Q - mu*I as an (n,n) CSR matrix with sorted columns, built directly from Q's triplets.  Duplicate
// triplets are summed, as Eigen's setFromTriplets( ) does, and every row gets a diagonal entry.
// Two stable counting sorts (by column, then by row) leave each row sorted, in O(nnz + n) for any input
// order.  (Going through Eigen took ~2 ms for 75K non-zeros.)
template <typename Real>
void buildShiftedCsr(int n, const int* rows, const int* cols, const Real* values, int nnz, Real mu, HostCsr<Real>& B)
{
    const int m = nnz + n;   // the triplets, plus (i, i, -mu) for each row
    auto rowOf = [&](int k) {return (k < nnz) ? rows[k] : k - nnz;};
    auto colOf = [&](int k) {return (k < nnz) ? cols[k] : k - nnz;};
    auto valueOf = [&](int k) {return (k < nnz) ? values[k] : -mu;};

    // 1. Order the entries by column.
    std::vector<int> start(n + 1, 0);
    for (int k = 0; k < m; k++)
        start[colOf(k) + 1]++;
    for (int j = 0; j < n; j++)
        start[j + 1] += start[j];
    std::vector<int> byColumn(m);
    for (int k = 0; k < m; k++)
        byColumn[start[colOf(k)]++] = k;

    // 2. Stably order them by row, so each row is sorted by column.
    B.offsets.assign(n + 1, 0);
    for (int k = 0; k < m; k++)
        B.offsets[rowOf(k) + 1]++;
    for (int i = 0; i < n; i++)
        B.offsets[i + 1] += B.offsets[i];
    std::vector<int> next(B.offsets.begin(), B.offsets.end() - 1);
    B.columns.resize(m);
    B.values.resize(m);
    for (int k: byColumn)
    {
        const int a = next[rowOf(k)]++;
        B.columns[a] = colOf(k);
        B.values[a] = valueOf(k);
    }

    // 3. Sum duplicates, compacting in place.
    int out = 0;
    for (int i = 0; i < n; i++)
    {
        const int begin = B.offsets[i];
        const int end = B.offsets[i + 1];
        B.offsets[i] = out;
        for (int a = begin; a < end; a++)
        {
            if (out > B.offsets[i] and B.columns[out - 1] == B.columns[a])
                B.values[out - 1] += B.values[a];
            else
            {
                B.columns[out] = B.columns[a];
                B.values[out++] = B.values[a];
            }
        }
    }
    B.offsets[n] = out;
    B.columns.resize(out);
    B.values.resize(out);
}

// max_j sum_i |B(i,j)|
template <typename Real>
double normP1(int n, const HostCsr<Real>& B)
{
    std::vector<double> sums(n, 0.0);
    for (int k = 0; k < B.nonZeros(); k++)
        sums[B.columns[k]] += std::abs(B.values[k]);
    return n ? *std::max_element(sums.begin(), sums.end()) : 0.0;
}

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

    void copyFromAsync(const DnMatrixDevice<Real>& D, cudaStream_t stream)
    {
        assert(order == D.order);
        assert(size1 == D.size1);
        assert(size2 == D.size2);
        auto status = cudaMemcpyAsync(ptr, D.ptr, byte_size(), cudaMemcpyDeviceToDevice, stream);
        if (status != cudaSuccess)
            throw std::runtime_error("cudaMemcpyAsync(Device->Device): " + std::string(cudaGetErrorString(status)));
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

    // Point at different memory; the cuSPARSE descriptor must follow, or SpMM reads the old buffer.
    void setPointer(Real* p)
    {
        ptr = p;
        if (descr and cusparseDnMatSetValues(descr, p) != CUSPARSE_STATUS_SUCCESS)
        {
            std::cerr<<"cusparseDnMatSetValues( ) failed!";
            std::exit(1);
        }
    }

    // (*this) = a * (*this) + b * D, in one pass.
    void axpby(Real a, Real b, const DnMatrixDevice<Real>& D)
    {
        cublasStatus_t status;
        assert(D.size1 == size1);
        assert(D.size2 == size2);
        assert(order == CUSPARSE_ORDER_COL and D.order == CUSPARSE_ORDER_COL);
        // geam allows C to alias A when A is not transposed and lda == ldc.
        if constexpr (std::is_same<Real, float>::value) {
            status = cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, size1, size2, &a, ptr, size1, &b, D.ptr, size1, ptr, size1);
        } else {
            status = cublasDgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, size1, size2, &a, ptr, size1, &b, D.ptr, size1, ptr, size1);
        }

        if (status != CUBLAS_STATUS_SUCCESS)
        {
            std::cerr<<"cublas error "<<status<<" in DnMatrix<>::axpby\n";
            exit(1);
        }
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

// C = alpha * A * B with a work buffer already sized by spMMBufferSize( ).
template <typename Real>
int spMMPrepared(DnMatrixDevice<Real>& C, Real alpha, const SpMatrixDevice<Real>& A, const DnMatrixDevice<Real>& B, void* buffer)
{
    Real beta = 0;
    CHECK_CUSPARSE(cusparseSpMM(A.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, A.descr, B.descr, &beta, C.descr, DataType<Real>,
                                CUSPARSE_SPMM_ALG_DEFAULT, buffer));
    return 0;
}

template <typename Real>
size_t spMMBufferSize(const DnMatrixDevice<Real>& C, const SpMatrixDevice<Real>& A, const DnMatrixDevice<Real>& B)
{
    Real alpha = 1;
    Real beta = 0;
    size_t size = 0;
    auto status = cusparseSpMM_bufferSize(A.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          &alpha, A.descr, B.descr, &beta, C.descr, DataType<Real>,
                                          CUSPARSE_SPMM_ALG_DEFAULT, &size);
    if (status != CUSPARSE_STATUS_SUCCESS)
        throw std::runtime_error(std::string("cusparseSpMM_bufferSize: ") + cusparseGetErrorString(status));
    return size;
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




// Block 1-norm estimates of ||B^p||_1 for p = 2 .. pTop at once, for the d_p of equation 3.7 in Al-Mohy and
// Higham.  This is Algorithm 2.4 of Higham and Tisseur (2000) with 2 iterations and t columns (without
// resampling parallel columns), but the products for all p are chained, so it takes about 3*pTop launches
// of at most (pTop-1)*t columns instead of O(pTop^2) separate SpMMs:
//   1. Y_p = B^p X0 for p = 1..pTop: one chain, each step reading the previous block.
//   2. Z_p = (B^T)^p sign(Y_p): start from [sign(Y_pTop)]; each step multiplies all blocks so far by B^T
//      and appends the next sign block, so sign(Y_p) is multiplied exactly p times.
//   3. X_p = the t unit vectors e_i with the largest max_j |Z_p(i,j)|.
//   4. B^p X_p, by the same appending chain with B.
// The estimate for p is the largest column 1-norm of Y_p and of B^p X_p.  Everything stays on the
// device until one copy of the pTop+1 results.
template <typename Real>
struct BlockNormEstimator
{
    int n = 0;
    int t = 0;
    int pTop = 0;

    Real* X0 = nullptr;          // (n, t)
    Real* Y = nullptr;           // (n, pTop*t): block p-1 is B^p X0
    Real* chain[2] = {nullptr, nullptr};  // (n, (pTop-1)*t) each
    Real* sums1 = nullptr;       // pTop*t column sums of Y
    Real* sums2 = nullptr;       // (pTop-1)*t column sums of step 4
    int* unitRows = nullptr;     // (pTop-1)*t
    Real* norms = nullptr;       // pTop+1

    // B^T in CSR (that is, B in CSC)
    int* btOffsets = nullptr;
    int* btColumns = nullptr;
    Real* btValues = nullptr;
    int btCapacity = 0;
    void* transposeBuffer = nullptr;
    size_t transposeBufferSize = 0;

    void allocate(int n_, int t_, int pTop_)
    {
        n = n_;
        t = std::min(t_, n);
        pTop = pTop_;
        assert(t <= 32);   // cuda_select_unit_columns( )
        X0 = cudaDeviceNew<Real>(n * t);
        Y = cudaDeviceNew<Real>(n * t * pTop);
        chain[0] = cudaDeviceNew<Real>(n * t * (pTop - 1));
        chain[1] = cudaDeviceNew<Real>(n * t * (pTop - 1));
        sums1 = cudaDeviceNew<Real>(t * pTop);
        sums2 = cudaDeviceNew<Real>(t * (pTop - 1));
        unitRows = cudaDeviceNew<int>(t * (pTop - 1));
        norms = cudaDeviceNew<Real>(pTop + 1);
        btOffsets = cudaDeviceNew<int>(n + 1);
        initialize_norm_x_matrix(X0, n, t);
    }

    // Returns ||B^p||_1 estimates for p = 2..pTop at index p.  (Indices 0 and 1 are not set.)
    std::vector<Real> estimate(const SpMatrixDevice<Real>& B, cudaStream_t stream)
    {
        assert(B.rows() == n and B.cols() == n);
        const int nnz = B.num_non_zeros;
        const int nBlocks = pTop - 1;
        const size_t blockSize = (size_t) n * t;

        // B^T
        if (nnz > btCapacity)
        {
            cudaDeviceDelete(btColumns);
            cudaDeviceDelete(btValues);
            btColumns = cudaDeviceNew<int>(nnz);
            btValues = cudaDeviceNew<Real>(nnz);
            btCapacity = nnz;
        }
        size_t size = 0;
        checkCusparse(cusparseCsr2cscEx2_bufferSize(B.cusparseHandle, n, n, nnz, B.values, B.offsets, B.inner,
                                                    btValues, btOffsets, btColumns, DataType<Real>, CUSPARSE_ACTION_NUMERIC,
                                                    CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, &size));
        if (size > transposeBufferSize or transposeBuffer == nullptr)
        {
            cudaDeviceDelete(transposeBuffer);
            transposeBufferSize = std::max<size_t>(size, 1);
            transposeBuffer = cudaMallocWrapped(transposeBufferSize);
        }
        checkCusparse(cusparseCsr2cscEx2(B.cusparseHandle, n, n, nnz, B.values, B.offsets, B.inner,
                                         btValues, btOffsets, btColumns, DataType<Real>, CUSPARSE_ACTION_NUMERIC,
                                         CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, transposeBuffer));

        // 1. Y_p = B^p X0
        for (int p = 1; p <= pTop; p++)
        {
            const Real* x = (p == 1) ? X0 : Y + (p - 2) * blockSize;
            cuda_norm_chain_step(n, B.offsets, B.inner, B.values, x, Y + (p - 1) * blockSize, t, 0, (Real*)nullptr, nullptr, stream);
        }
        cuda_column_abs_sums(Y, n, pTop * t, sums1, stream);

        // 2. Z_p = (B^T)^p sign(Y_p), in block pTop - p.
        Real* in = chain[0];
        Real* out = chain[1];
        cuda_norm_chain_step(n, btOffsets, btColumns, btValues, (Real*)nullptr, in, 0, t, Y + (pTop - 1) * blockSize, nullptr, stream);
        for (int k = 1; k <= pTop; k++)
        {
            const bool append = (k < nBlocks);
            const Real* sign = append ? Y + (pTop - k - 1) * blockSize : nullptr;
            cuda_norm_chain_step(n, btOffsets, btColumns, btValues, in, out, std::min(k, nBlocks) * t, append ? t : 0, sign, nullptr, stream);
            std::swap(in, out);
        }

        // 3. Unit vectors for the second iteration.
        cuda_select_unit_columns(in, n, t, nBlocks, unitRows, stream);

        // 4. B^p X_p, in block pTop - p.
        cuda_norm_chain_step(n, B.offsets, B.inner, B.values, (Real*)nullptr, out, 0, t, (Real*)nullptr, unitRows, stream);
        std::swap(in, out);
        for (int k = 1; k <= pTop; k++)
        {
            const bool append = (k < nBlocks);
            cuda_norm_chain_step(n, B.offsets, B.inner, B.values, in, out, std::min(k, nBlocks) * t, append ? t : 0,
                                 (Real*)nullptr, append ? unitRows + k * t : nullptr, stream);
            std::swap(in, out);
        }
        cuda_column_abs_sums(in, n, nBlocks * t, sums2, stream);

        cuda_combine_norm_estimates(sums1, sums2, t, pTop, norms, stream);

        std::vector<Real> result(pTop + 1, 0);
        auto status = cudaMemcpyAsync(result.data(), norms, (pTop + 1) * sizeof(Real), cudaMemcpyDeviceToHost, stream);
        if (status == cudaSuccess)
            status = cudaStreamSynchronize(stream);
        if (status != cudaSuccess)
            throw std::runtime_error(std::string("BlockNormEstimator: ") + cudaGetErrorString(status));
        return result;
    }

    static void checkCusparse(cusparseStatus_t status)
    {
        if (status != CUSPARSE_STATUS_SUCCESS)
            throw std::runtime_error(std::string("BlockNormEstimator: ") + cusparseGetErrorString(status));
    }

    BlockNormEstimator() = default;
    BlockNormEstimator(const BlockNormEstimator&) = delete;
    BlockNormEstimator& operator=(const BlockNormEstimator&) = delete;

    ~BlockNormEstimator()
    {
        cudaDeviceDelete(X0);
        cudaDeviceDelete(Y);
        cudaDeviceDelete(chain[0]);
        cudaDeviceDelete(chain[1]);
        cudaDeviceDelete(sums1);
        cudaDeviceDelete(sums2);
        cudaDeviceDelete(unitRows);
        cudaDeviceDelete(norms);
        cudaDeviceDelete(btOffsets);
        cudaDeviceDelete(btColumns);
        cudaDeviceDelete(btValues);
        cudaDeviceDelete(transposeBuffer);
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

    // Per eigen index: B = Q - mu*I.
    std::vector<HostCsr<Real>> hBs;
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
    // These hold the Taylor sum F while it is accumulated.
    std::vector<DnMatrixDevice<Real>> dDestPLeft;
    std::vector<DnMatrixDevice<Real>> dDestPRight;
    // Per category: two buffers that alternate as the input and output of each Taylor term.
    std::vector<DnMatrixDevice<Real>> dTerms[2];
    // Per eigen index: B = Q - mu*I on the device.  The edge multiplier goes into the SpMM alpha.
    std::vector<SpMatrixDevice<Real>> dBs;
    void* dSpMMBuffer = nullptr;
    size_t spMMBufferBytes = 0;
    BlockNormEstimator<Real> normEstimator;

    // All work is issued on the GPUInterface's stream 0, so it is ordered with the base class's
    // copies and kernels (two different blocking streams would not be).  See setCudaDevice( ).
    cudaStream_t stream = nullptr;

    std::vector<int *> dBsCsrOffsetsCache;
    std::vector<int *> dBsCsrColumnsCache;
    std::vector<Real*> dBsCsrValuesCache;
    std::vector<int> currentCacheNNZs;

    cublasHandle_t cublasHandle = nullptr;
    cusparseHandle_t cusparseHandle = nullptr;

    std::vector<int> hEigenMaps;
    std::vector<Real> hEdgeMultipliers;


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

    // Make CUDA runtime, cuBLAS, cuSPARSE and Thrust calls target this instance's device,
    // and cuBLAS/cuSPARSE use the GPUInterface's stream.
    void setCudaDevice();

    int getPartialIndex(int nodeIndex, int categoryIndex);

    DnMatrixDevice<Real>& getPartialsWrapper(int nodeIndex, int categoryIndex);

    // scalingFactors == nullptr means no rescaling.
    void calcPartialsPartials(int destPIndex,
                              int partials1Index,
                              int edgeIndex1,
                              int partials2Index,
                              int edgeIndex2,
                              Real* scalingFactors,
                              Real* cumulativeScalingBuffer);

    // F = exp(t * edgeMultiplier * Q) * inPartials
    int simpleAction2(DnMatrixDevice<Real>& F, const DnMatrixDevice<Real>& inPartials, int edgeIndex, int category);

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

