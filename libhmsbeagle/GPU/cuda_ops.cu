#include "cuda_ops.h"

#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>

using thrust::device_ptr;
using thrust::device_pointer_cast;

template <typename T>
// This seems to be slow because it inserts cudaStreamSynchronize()
T cuda_max_abs(device_ptr<T> values, int length)
{
    auto in_ptr = thrust::transform_iterator(values, [] __host__ __device__ (T x) {return std::abs(x);});

    // Don't use `*(thrust::max_element(....))` -- it is slower than thrust::reduce.

    // thrust::reduce is similar in speed to cublasI{s,d}amax.
    // using  thrust::reduce requires a device-to-host-memcpy, so it is slow!
    return thrust::reduce(in_ptr, in_ptr + length, 0.0, thrust::maximum<T>());
}

float cuda_max_abs(float* values, int length)
{
    return cuda_max_abs(device_pointer_cast(values), length);
}

double cuda_max_abs(double* values, int length)
{
    return cuda_max_abs(device_pointer_cast(values), length);
}

template <typename T>
void initialize_norm_x_matrix(device_ptr<T> out_ptr, int n, int m)
{
    auto indices = thrust::counting_iterator<unsigned int>(0);
    auto initialize = [n,m] __host__ __device__ (int i)
    {
        thrust::default_random_engine rng;
        thrust::uniform_real_distribution<T> dist(0, 1);

        if (i<n) return 1.0/n;
        rng.discard(i);
        if (dist(rng) > 0.5)
            return 1.0/n;
        else
            return -1.0/n;
    };

    thrust::transform(indices, indices + n*m, // in
                      out_ptr,                // out
                      initialize);
}

void initialize_norm_x_matrix(float* data, int n, int m)
{
    initialize_norm_x_matrix(device_pointer_cast(data), n, m);
}

void initialize_norm_x_matrix(double* data, int n, int m)
{
    initialize_norm_x_matrix(device_pointer_cast(data), n, m);
}


// ---------------------------------------------------------------------------------------------------
// Kernels for the per-node product/rescale and the root likelihood.  They replace the Thrust
// reduce_by_key chains, which allocate temporary memory and block the host on every call.
//
// Partials are stored as [category][pattern][state] (state fastest), with the padded counts.
// One block handles one pattern.

// With few patterns there are few blocks, so give each block more threads.
inline int patternBlockSize(int nPatterns)
{
    return (nPatterns >= 256) ? 128 : 1024;
}

// Reduce v over the block with op; identity must be neutral for op.  Every thread gets the result.
template <typename T, typename Op>
__device__ T block_reduce(T v, T identity, Op op)
{
    __shared__ T warpResults[32];

    for (int offset = 16; offset > 0; offset /= 2)
        v = op(v, __shfl_down_sync(0xffffffff, v, offset));

    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    if (lane == 0) warpResults[warp] = v;
    __syncthreads();

    if (warp == 0)
    {
        const int nWarps = (blockDim.x + 31) / 32;
        v = (lane < nWarps) ? warpResults[lane] : identity;
        for (int offset = 16; offset > 0; offset /= 2)
            v = op(v, __shfl_down_sync(0xffffffff, v, offset));
        if (lane == 0) warpResults[0] = v;
    }
    __syncthreads();

    return warpResults[0];
}

struct MaxOp { template <typename T> __device__ T operator()(T a, T b) const { return max(a, b); } };
struct SumOp { template <typename T> __device__ T operator()(T a, T b) const { return a + b; } };

// parent = left * right (elementwise).  If scalingFactors is non-null, also divide each pattern by its
// largest entry m (1 if all are 0), store m (or log m) in scalingFactors, and add log m to cumulative.
template <typename T>
__global__ void partials_product_kernel(const T* left, const T* right, T* parent,
                                        int nStates, int nPatterns, int nCategories,
                                        T* scalingFactors, T* cumulative, bool scalersLog)
{
    const int pattern = blockIdx.x;
    const size_t categoryStride = (size_t)nStates * nPatterns;
    const size_t offset = (size_t)pattern * nStates;

    if (not scalingFactors)
    {
        for (int c = 0; c < nCategories; c++)
            for (int s = threadIdx.x; s < nStates; s += blockDim.x)
            {
                const size_t i = c * categoryStride + offset + s;
                parent[i] = left[i] * right[i];
            }
        return;
    }

    T m = 0;
    for (int c = 0; c < nCategories; c++)
        for (int s = threadIdx.x; s < nStates; s += blockDim.x)
        {
            const size_t i = c * categoryStride + offset + s;
            m = max(m, left[i] * right[i]);
        }
    m = block_reduce(m, T(0), MaxOp());
    if (m == 0) m = 1;

    const T inverse = 1 / m;
    for (int c = 0; c < nCategories; c++)
        for (int s = threadIdx.x; s < nStates; s += blockDim.x)
        {
            const size_t i = c * categoryStride + offset + s;
            parent[i] = left[i] * right[i] * inverse;
        }

    if (threadIdx.x == 0)
    {
        const T logM = log(m);
        scalingFactors[pattern] = scalersLog ? logM : m;
        if (cumulative) cumulative[pattern] += logM;
    }
}

template <typename T>
void cuda_partials_product_impl(const T* left, const T* right, T* parent, int nStates, int nPatterns, int nCategories,
                                 T* scalingFactors, T* cumulative, bool scalersLog, cudaStream_t stream)
{
    partials_product_kernel<<<nPatterns, patternBlockSize(nPatterns), 0, stream>>>(left, right, parent, nStates, nPatterns, nCategories,
                                                                         scalingFactors, cumulative, scalersLog);
}

void cuda_partials_product(const float* left, const float* right, float* parent, int nStates, int nPatterns, int nCategories,
                           float* scalingFactors, float* cumulative, bool scalersLog, cudaStream_t stream)
{
    cuda_partials_product_impl(left, right, parent, nStates, nPatterns, nCategories, scalingFactors, cumulative, scalersLog, stream);
}

void cuda_partials_product(const double* left, const double* right, double* parent, int nStates, int nPatterns, int nCategories,
                           double* scalingFactors, double* cumulative, bool scalersLog, cudaStream_t stream)
{
    cuda_partials_product_impl(left, right, parent, nStates, nPatterns, nCategories, scalingFactors, cumulative, scalersLog, stream);
}

// siteLogLikelihoods[p] = log( sum_{c,s} root(c,p,s) * categoryWeights[c] * frequencies[s] ) + cumulativeScaling[p]
template <typename T>
__global__ void root_site_log_likelihoods_kernel(const T* root, const T* categoryWeights, const T* frequencies,
                                                 const T* cumulativeScaling, int nStates, int nPaddedPatterns, int nCategories,
                                                 T* siteLogLikelihoods)
{
    const int pattern = blockIdx.x;
    const size_t categoryStride = (size_t)nStates * nPaddedPatterns;
    const size_t offset = (size_t)pattern * nStates;

    T sum = 0;
    for (int c = 0; c < nCategories; c++)
    {
        T categorySum = 0;
        for (int s = threadIdx.x; s < nStates; s += blockDim.x)
            categorySum += root[c * categoryStride + offset + s] * frequencies[s];
        sum += categorySum * categoryWeights[c];
    }
    sum = block_reduce(sum, T(0), SumOp());

    if (threadIdx.x == 0)
        siteLogLikelihoods[pattern] = log(sum) + (cumulativeScaling ? cumulativeScaling[pattern] : T(0));
}

// out[0] = sum_i values[i] * weights[i], using a single block (deterministic).
template <typename T>
__global__ void weighted_sum_kernel(const T* values, const T* weights, int n, T* out)
{
    T sum = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        sum += values[i] * weights[i];
    sum = block_reduce(sum, T(0), SumOp());
    if (threadIdx.x == 0) out[0] = sum;
}

template <typename T>
void cuda_root_log_likelihoods_impl(const T* root, const T* categoryWeights, const T* frequencies, const T* cumulativeScaling,
                                    const T* patternWeights, int nStates, int nPaddedPatterns, int nPatterns, int nCategories,
                                    T* siteLogLikelihoods, T* sumLogLikelihood, cudaStream_t stream)
{
    root_site_log_likelihoods_kernel<<<nPatterns, patternBlockSize(nPatterns), 0, stream>>>(root, categoryWeights, frequencies, cumulativeScaling,
                                                                                  nStates, nPaddedPatterns, nCategories, siteLogLikelihoods);
    weighted_sum_kernel<<<1, 1024, 0, stream>>>(siteLogLikelihoods, patternWeights, nPatterns, sumLogLikelihood);
}

void cuda_root_log_likelihoods(const float* root, const float* categoryWeights, const float* frequencies, const float* cumulativeScaling,
                               const float* patternWeights, int nStates, int nPaddedPatterns, int nPatterns, int nCategories,
                               float* siteLogLikelihoods, float* sumLogLikelihood, cudaStream_t stream)
{
    cuda_root_log_likelihoods_impl(root, categoryWeights, frequencies, cumulativeScaling, patternWeights,
                                   nStates, nPaddedPatterns, nPatterns, nCategories, siteLogLikelihoods, sumLogLikelihood, stream);
}

void cuda_root_log_likelihoods(const double* root, const double* categoryWeights, const double* frequencies, const double* cumulativeScaling,
                               const double* patternWeights, int nStates, int nPaddedPatterns, int nPatterns, int nCategories,
                               double* siteLogLikelihoods, double* sumLogLikelihood, cudaStream_t stream)
{
    cuda_root_log_likelihoods_impl(root, categoryWeights, frequencies, cumulativeScaling, patternWeights,
                                   nStates, nPaddedPatterns, nPatterns, nCategories, siteLogLikelihoods, sumLogLikelihood, stream);
}


// One Taylor term on column-major (nStates, nColumns) matrices, in a single pass:
//   y = alpha * A * x;   F = scale * (F + y)
// One thread per (row, column), consecutive threads on consecutive rows.  Used when the work is small:
// it replaces cuSPARSE's partition, beta=0 fill and SpMM kernels plus the axpy.
template <typename T>
__global__ void csr_taylor_term_kernel(int nStates, int nColumns, const int* __restrict__ rowOffsets,
                                       const int* __restrict__ columns, const T* __restrict__ values, T alpha,
                                       const T* __restrict__ x, T* __restrict__ y, T* __restrict__ F, T scale)
{
    const size_t index = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (index >= (size_t)nStates * nColumns) return;
    const int row = index % nStates;
    const T* xColumn = x + (index - row);

    T sum = 0;
    for (int k = rowOffsets[row]; k < rowOffsets[row + 1]; k++)
        sum += values[k] * __ldg(xColumn + columns[k]);
    sum *= alpha;

    y[index] = sum;
    F[index] = scale * (F[index] + sum);
}

template <typename T>
void cuda_csr_taylor_term_impl(int nStates, int nColumns, const int* rowOffsets, const int* columns, const T* values, T alpha,
                               const T* x, T* y, T* F, T scale, cudaStream_t stream)
{
    const size_t n = (size_t)nStates * nColumns;
    const int blockSize = 256;
    csr_taylor_term_kernel<<<(n + blockSize - 1) / blockSize, blockSize, 0, stream>>>(nStates, nColumns, rowOffsets, columns, values,
                                                                                       alpha, x, y, F, scale);
}

void cuda_csr_taylor_term(int nStates, int nColumns, const int* rowOffsets, const int* columns, const float* values, float alpha,
                          const float* x, float* y, float* F, float scale, cudaStream_t stream)
{
    cuda_csr_taylor_term_impl(nStates, nColumns, rowOffsets, columns, values, alpha, x, y, F, scale, stream);
}

void cuda_csr_taylor_term(int nStates, int nColumns, const int* rowOffsets, const int* columns, const double* values, double alpha,
                          const double* x, double* y, double* F, double scale, cudaStream_t stream)
{
    cuda_csr_taylor_term_impl(nStates, nColumns, rowOffsets, columns, values, alpha, x, y, F, scale, stream);
}


// ---------------------------------------------------------------------------------------------------
// Block 1-norm estimation of ||B^p||_1 for all p at once (Higham & Tisseur Algorithm 2.4, 2 iterations).
// All matrices are column-major with n rows.  See BlockNormEstimator in BeagleGPUActionImpl.h.

// y(:, 0:nApplied) = A * x(:, 0:nApplied), then nAppend more columns of y are set:
//   from signSource (n, nAppend):  y(i, c) = +1 if signSource(i, c) >= 0, else -1
//   from unitRows   (nAppend):     y(i, c) = 1 if i == unitRows[c], else 0
template <typename T>
__global__ void norm_chain_step_kernel(int n, const int* __restrict__ rowOffsets, const int* __restrict__ columns,
                                       const T* __restrict__ values, const T* __restrict__ x, T* __restrict__ y,
                                       int nApplied, int nAppend, const T* __restrict__ signSource, const int* __restrict__ unitRows)
{
    const size_t index = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (index >= (size_t)n * (nApplied + nAppend)) return;
    const int row = index % n;
    const int column = index / n;

    if (column < nApplied)
    {
        const T* xColumn = x + (size_t)column * n;
        T sum = 0;
        for (int k = rowOffsets[row]; k < rowOffsets[row + 1]; k++)
            sum += values[k] * xColumn[columns[k]];
        y[index] = sum;
    }
    else
    {
        const int c = column - nApplied;
        if (signSource)
            y[index] = (signSource[(size_t)c * n + row] >= 0) ? T(1) : T(-1);
        else
            y[index] = (row == unitRows[c]) ? T(1) : T(0);
    }
}

template <typename T>
void cuda_norm_chain_step_impl(int n, const int* rowOffsets, const int* columns, const T* values, const T* x, T* y,
                               int nApplied, int nAppend, const T* signSource, const int* unitRows, cudaStream_t stream)
{
    const size_t total = (size_t)n * (nApplied + nAppend);
    const int blockSize = 256;
    norm_chain_step_kernel<<<(total + blockSize - 1) / blockSize, blockSize, 0, stream>>>(n, rowOffsets, columns, values, x, y,
                                                                                          nApplied, nAppend, signSource, unitRows);
}

void cuda_norm_chain_step(int n, const int* rowOffsets, const int* columns, const float* values, const float* x, float* y,
                          int nApplied, int nAppend, const float* signSource, const int* unitRows, cudaStream_t stream)
{
    cuda_norm_chain_step_impl(n, rowOffsets, columns, values, x, y, nApplied, nAppend, signSource, unitRows, stream);
}

void cuda_norm_chain_step(int n, const int* rowOffsets, const int* columns, const double* values, const double* x, double* y,
                          int nApplied, int nAppend, const double* signSource, const int* unitRows, cudaStream_t stream)
{
    cuda_norm_chain_step_impl(n, rowOffsets, columns, values, x, y, nApplied, nAppend, signSource, unitRows, stream);
}

// sums[c] = sum_i |x(i, c)|, one block per column.
template <typename T>
__global__ void column_abs_sums_kernel(const T* x, int n, T* sums)
{
    const T* column = x + (size_t)blockIdx.x * n;
    T sum = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        sum += abs(column[i]);
    sum = block_reduce(sum, T(0), SumOp());
    if (threadIdx.x == 0) sums[blockIdx.x] = sum;
}

template <typename T>
void cuda_column_abs_sums_impl(const T* x, int n, int nColumns, T* sums, cudaStream_t stream)
{
    column_abs_sums_kernel<<<nColumns, 256, 0, stream>>>(x, n, sums);
}

void cuda_column_abs_sums(const float* x, int n, int nColumns, float* sums, cudaStream_t stream)
{
    cuda_column_abs_sums_impl(x, n, nColumns, sums, stream);
}

void cuda_column_abs_sums(const double* x, int n, int nColumns, double* sums, cudaStream_t stream)
{
    cuda_column_abs_sums_impl(x, n, nColumns, sums, stream);
}

// For each block b of t columns of Z: h(i) = max_j |Z_b(i, j)|; unitRows[b*t + r] = the row with the
// r-th largest h (ties go to the lower row).  One thread block per column block.
template <typename T>
__global__ void select_unit_columns_kernel(const T* Z, int n, int t, int* unitRows)
{
    __shared__ T bestValue[1024];
    __shared__ int bestRow[1024];
    __shared__ int chosen[32];

    const T* block = Z + (size_t)blockIdx.x * n * t;

    for (int r = 0; r < t; r++)
    {
        T value = -1;
        int row = n;
        for (int i = threadIdx.x; i < n; i += blockDim.x)
        {
            bool used = false;
            for (int q = 0; q < r; q++)
                used = used or (chosen[q] == i);
            if (used) continue;

            T h = 0;
            for (int j = 0; j < t; j++)
                h = max(h, abs(block[(size_t)j * n + i]));
            if (h > value) { value = h; row = i; }
        }
        bestValue[threadIdx.x] = value;
        bestRow[threadIdx.x] = row;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
        {
            if (threadIdx.x < stride)
            {
                const int other = threadIdx.x + stride;
                if (bestValue[other] > bestValue[threadIdx.x] or
                    (bestValue[other] == bestValue[threadIdx.x] and bestRow[other] < bestRow[threadIdx.x]))
                {
                    bestValue[threadIdx.x] = bestValue[other];
                    bestRow[threadIdx.x] = bestRow[other];
                }
            }
            __syncthreads();
        }

        if (threadIdx.x == 0)
        {
            chosen[r] = bestRow[0];
            unitRows[blockIdx.x * t + r] = bestRow[0];
        }
        __syncthreads();
    }
}

template <typename T>
void cuda_select_unit_columns_impl(const T* Z, int n, int t, int nBlocks, int* unitRows, cudaStream_t stream)
{
    select_unit_columns_kernel<<<nBlocks, 1024, 0, stream>>>(Z, n, t, unitRows);
}

void cuda_select_unit_columns(const float* Z, int n, int t, int nBlocks, int* unitRows, cudaStream_t stream)
{
    cuda_select_unit_columns_impl(Z, n, t, nBlocks, unitRows, stream);
}

void cuda_select_unit_columns(const double* Z, int n, int t, int nBlocks, int* unitRows, cudaStream_t stream)
{
    cuda_select_unit_columns_impl(Z, n, t, nBlocks, unitRows, stream);
}

// norms[p] = max over the t columns for p in both iterations, for p = 2 .. pTop:
//   iteration 1: block p-1 of sums1;  iteration 2: block pTop-p of sums2.
template <typename T>
__global__ void combine_norm_estimates_kernel(const T* sums1, const T* sums2, int t, int pTop, T* norms)
{
    const int p = threadIdx.x + 2;
    if (p > pTop) return;
    T norm = 0;
    for (int j = 0; j < t; j++)
        norm = max(norm, max(sums1[(p - 1) * t + j], sums2[(pTop - p) * t + j]));
    norms[p] = norm;
}

template <typename T>
void cuda_combine_norm_estimates_impl(const T* sums1, const T* sums2, int t, int pTop, T* norms, cudaStream_t stream)
{
    combine_norm_estimates_kernel<<<1, 32, 0, stream>>>(sums1, sums2, t, pTop, norms);
}

void cuda_combine_norm_estimates(const float* sums1, const float* sums2, int t, int pTop, float* norms, cudaStream_t stream)
{
    cuda_combine_norm_estimates_impl(sums1, sums2, t, pTop, norms, stream);
}

void cuda_combine_norm_estimates(const double* sums1, const double* sums2, int t, int pTop, double* norms, cudaStream_t stream)
{
    cuda_combine_norm_estimates_impl(sums1, sums2, t, pTop, norms, stream);
}
