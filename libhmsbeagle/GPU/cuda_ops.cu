#include "cuda_ops.h"

#include <algorithm>
#include <cstdlib>

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
    // Element (row, column) is stored at row * m + column.  Its sign comes from its column-major index
    // i = column * n + row.
    auto indices = thrust::counting_iterator<unsigned int>(0);
    auto initialize = [n,m] __host__ __device__ (int index)
    {
        const int row = index / m;
        const int column = index % m;
        const int i = column * n + row;
        thrust::default_random_engine rng;
        thrust::uniform_real_distribution<T> dist(0, 1);

        if (i<n) return T(1.0/n);
        rng.discard(i);
        if (dist(rng) > 0.5)
            return T(1.0/n);
        else
            return T(-1.0/n);
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
// Kernels for the Taylor terms, the per-node product/rescale and the root likelihood.
//
// Action partials are stored pattern-fastest: [category][state][pattern], i.e. a (nCategories*nStates)
// x nPatterns row-major matrix, so a CSR row of Q gathers whole rows of x (coalesced over patterns).

// Reduce v over the block with op; identity must be neutral for op.  Every thread gets the result.
// Only for one-dimensional blocks.
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

static int nextPowerOfTwo(int n, int cap)
{
    int p = 1;
    while (p < n and p < cap) p *= 2;
    return p;
}

// One step w <- scale * (b + alpha * A * x) of Horner's rule for the Taylor polynomial, for each group of the
// batch.  A is (nRows, nRows) in CSR; x, b and out are (nRows, nCols) row-major.
//
// For nCols >= 32: blockIdx.z is the group, the 32 lanes of a warp cover 32 adjacent columns of one row, and
// each thread does CPT columns 32 apart.  Blocks are ordered rows-fastest, so the blocks in flight read the
// same column panel of x and it stays in L2.
template <typename T, int CPT>
__global__ void __launch_bounds__(256) taylor_horner_term_kernel(int nRows, int nCols, const int* __restrict__ rowOffsets,
                                                                 const int* __restrict__ columns, const T* __restrict__ values,
                                                                 const TaylorTermBatch<T> batch)
{
    const TaylorTermGroup<T>& g = batch.groups[blockIdx.z];
    const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int col = blockIdx.y * (32 * CPT) + threadIdx.x % 32;
    if (row >= nRows) return;

    const int kBegin = rowOffsets[row];
    const int kEnd = rowOffsets[row + 1];
    const T* b = g.b + (size_t)row * g.ldb + col;
    T* out = g.out + (size_t)row * g.ldOut + col;
    const T* x = g.x + col;
    T acc[CPT];
    T own[CPT];

    if (col + (CPT - 1) * 32 < nCols)
    {
#pragma unroll
        for (int q = 0; q < CPT; q++)
        {
            own[q] = b[q * 32];
            acc[q] = 0;
        }
        for (int k = kBegin; k < kEnd; k++)
        {
            const T v = values[k];
            const T* xRow = x + (size_t)columns[k] * g.ldx;
#pragma unroll
            for (int q = 0; q < CPT; q++)
                acc[q] += v * __ldg(xRow + q * 32);
        }
#pragma unroll
        for (int q = 0; q < CPT; q++)
            out[q * 32] = g.scale * (own[q] + g.alpha * acc[q]);
    }
    else
    {
#pragma unroll
        for (int q = 0; q < CPT; q++)
        {
            own[q] = (col + q * 32 < nCols) ? b[q * 32] : T(0);
            acc[q] = 0;
        }
        for (int k = kBegin; k < kEnd; k++)
        {
            const T v = values[k];
            const T* xRow = x + (size_t)columns[k] * g.ldx;
#pragma unroll
            for (int q = 0; q < CPT; q++)
                if (col + q * 32 < nCols)
                    acc[q] += v * __ldg(xRow + q * 32);
        }
#pragma unroll
        for (int q = 0; q < CPT; q++)
            if (col + q * 32 < nCols)
                out[q * 32] = g.scale * (own[q] + g.alpha * acc[q]);
    }
}

// For nCols < 32: the columns of all groups are numbered v = group * nCols + column, and LANES consecutive
// lanes share a row and cover LANES consecutive v.  So the lanes share each (column, value) load of A, and
// when the groups' matrices are stored side by side the gathers of x are coalesced across groups too.
template <typename T, int LANES>
__global__ void __launch_bounds__(256) taylor_horner_term_narrow_kernel(int nRows, int nCols, const int* __restrict__ rowOffsets,
                                                                        const int* __restrict__ columns, const T* __restrict__ values,
                                                                        const TaylorTermBatch<T> batch)
{
    // The lanes read different groups, which shared memory serves without serializing like parameter space.
    __shared__ TaylorTermGroup<T> groups[kMaxTaylorGroups];
    for (int i = threadIdx.x; i < batch.count; i += blockDim.x)
        groups[i] = batch.groups[i];
    __syncthreads();

    const int lane = threadIdx.x % 32;
    const int row = (blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32) * (32 / LANES) + lane / LANES;
    const int v = blockIdx.y * LANES + lane % LANES;
    if (row >= nRows or v >= batch.count * nCols) return;

    const TaylorTermGroup<T>& g = groups[v / nCols];
    const int col = v % nCols;
    const T* x = g.x + col;
    const T own = g.b[(size_t)row * g.ldb + col];
    T acc = 0;
    for (int k = rowOffsets[row]; k < rowOffsets[row + 1]; k++)
        acc += values[k] * __ldg(x + (size_t)columns[k] * g.ldx);
    g.out[(size_t)row * g.ldOut + col] = g.scale * (own + g.alpha * acc);
}

template <typename T, int CPT>
void launch_taylor_horner_terms(int nRows, int nCols, const int* rowOffsets, const int* columns, const T* values,
                                const TaylorTermBatch<T>& batch, cudaStream_t stream)
{
    const int threads = 256;
    const dim3 grid((nRows + threads / 32 - 1) / (threads / 32), (nCols + 32 * CPT - 1) / (32 * CPT), batch.count);
    taylor_horner_term_kernel<T, CPT><<<grid, threads, 0, stream>>>(nRows, nCols, rowOffsets, columns, values, batch);
}

template <typename T, int LANES>
void launch_taylor_horner_terms_narrow(int nRows, int nCols, const int* rowOffsets, const int* columns, const T* values,
                                       const TaylorTermBatch<T>& batch, cudaStream_t stream)
{
    const int threads = 256;
    const int rowsPerBlock = (threads / 32) * (32 / LANES);
    const dim3 grid((nRows + rowsPerBlock - 1) / rowsPerBlock, (batch.count * nCols + LANES - 1) / LANES);
    taylor_horner_term_narrow_kernel<T, LANES><<<grid, threads, 0, stream>>>(nRows, nCols, rowOffsets, columns, values, batch);
}

// Columns per thread for nCols >= 32; BEAGLE_ACTION_CPT (1, 2 or 4) overrides it for tuning.
static int columnsPerThread(int nCols)
{
    static const int forced = getenv("BEAGLE_ACTION_CPT") ? atoi(getenv("BEAGLE_ACTION_CPT")) : 0;
    if (forced == 1 or forced == 2 or forced == 4)
        return forced;
    return (nCols >= 256) ? 4 : (nCols >= 64) ? 2 : 1;
}

template <typename T>
void cuda_taylor_horner_terms_impl(int nRows, int nCols, const int* rowOffsets, const int* columns, const T* values,
                                   const TaylorTermBatch<T>& batch, cudaStream_t stream)
{
    if (batch.count == 0) return;
    if (nCols >= 32)
    {
        switch (columnsPerThread(nCols))
        {
            case 4:  launch_taylor_horner_terms<T, 4>(nRows, nCols, rowOffsets, columns, values, batch, stream); break;
            case 2:  launch_taylor_horner_terms<T, 2>(nRows, nCols, rowOffsets, columns, values, batch, stream); break;
            default: launch_taylor_horner_terms<T, 1>(nRows, nCols, rowOffsets, columns, values, batch, stream); break;
        }
        return;
    }
    switch (nextPowerOfTwo(batch.count * nCols, 32))
    {
        case 1:  launch_taylor_horner_terms_narrow<T, 1>(nRows, nCols, rowOffsets, columns, values, batch, stream); break;
        case 2:  launch_taylor_horner_terms_narrow<T, 2>(nRows, nCols, rowOffsets, columns, values, batch, stream); break;
        case 4:  launch_taylor_horner_terms_narrow<T, 4>(nRows, nCols, rowOffsets, columns, values, batch, stream); break;
        case 8:  launch_taylor_horner_terms_narrow<T, 8>(nRows, nCols, rowOffsets, columns, values, batch, stream); break;
        case 16: launch_taylor_horner_terms_narrow<T, 16>(nRows, nCols, rowOffsets, columns, values, batch, stream); break;
        default: launch_taylor_horner_terms_narrow<T, 32>(nRows, nCols, rowOffsets, columns, values, batch, stream); break;
    }
}

void cuda_taylor_horner_terms(int nRows, int nCols, const int* rowOffsets, const int* columns, const float* values,
                              const TaylorTermBatch<float>& batch, cudaStream_t stream)
{
    cuda_taylor_horner_terms_impl(nRows, nCols, rowOffsets, columns, values, batch, stream);
}

void cuda_taylor_horner_terms(int nRows, int nCols, const int* rowOffsets, const int* columns, const double* values,
                              const TaylorTermBatch<double>& batch, cudaStream_t stream)
{
    cuda_taylor_horner_terms_impl(nRows, nCols, rowOffsets, columns, values, batch, stream);
}

// The per-pattern reductions over the rows of an (nRows, nPatterns) row-major matrix use blocks of
// patternsPerBlock x (256 / patternsPerBlock) threads, and split the rows into chunks so that there are
// enough blocks.  Each chunk writes one partial result per pattern; a second kernel combines them.
struct RowChunks
{
    int patternsPerBlock;
    int rowsPerBlock;
    int patternBlocks;
    int rowsPerChunk;
    int chunks;
};

static RowChunks rowChunks(int nRows, int nPatterns)
{
    RowChunks c;
    c.patternsPerBlock = nextPowerOfTwo(nPatterns, 32);
    c.rowsPerBlock = 256 / c.patternsPerBlock;
    c.patternBlocks = (nPatterns + c.patternsPerBlock - 1) / c.patternsPerBlock;
    const int targetBlocks = 512;
    int chunks = (targetBlocks + c.patternBlocks - 1) / c.patternBlocks;
    chunks = std::max(1, std::min({chunks, nRows / (8 * c.rowsPerBlock), 128}));
    c.rowsPerChunk = (nRows + chunks - 1) / chunks;
    c.chunks = (nRows + c.rowsPerChunk - 1) / c.rowsPerChunk;
    return c;
}

int cuda_row_chunk_count(int nRows, int nPatterns)
{
    return rowChunks(nRows, nPatterns).chunks;
}

// Reduce v over threadIdx.y (blockDim.y a power of two); the result is valid in the threads with threadIdx.y == 0.
template <typename T, typename Op>
__device__ T column_reduce(T v, Op op)
{
    __shared__ T shared[256];
    const int i = threadIdx.y * blockDim.x + threadIdx.x;
    shared[i] = v;
    __syncthreads();
    for (int stride = blockDim.y / 2; stride > 0; stride /= 2)
    {
        if (threadIdx.y < stride)
            shared[i] = op(shared[i], shared[i + stride * blockDim.x]);
        __syncthreads();
    }
    return shared[threadIdx.x];
}

// 1 / the existing scale factor of pattern p, or 1 without fixed scaling.
template <typename T>
__device__ T fixed_inverse(const PartialsProduct<T>& product, int p, bool scalersLog)
{
    if (not product.fixedScalingFactors)
        return T(1);
    const T factor = product.fixedScalingFactors[p];
    return T(1) / (scalersLog ? exp(factor) : factor);
}

// parent = left * right for each product of the batch (blockIdx.y), divided by any fixed scale factors.
template <typename T>
__global__ void partials_product_kernel(const PartialsProductBatch<T> batch, size_t n, int nPatterns, bool scalersLog)
{
    const PartialsProduct<T>& product = batch.products[blockIdx.y];
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    {
        T value = product.right ? product.left[i] * product.right[i] : product.left[i];
        if (product.fixedScalingFactors)
            value *= fixed_inverse(product, (int)(i % nPatterns), scalersLog);
        product.parent[i] = value;
    }
}

// chunkMax[product, chunk, p] = max over the chunk's rows r of left[r, p] * right[r, p]  (product = blockIdx.z)
template <typename T>
__global__ void partials_product_max_kernel(const PartialsProductBatch<T> batch, int nRows, int nPatterns, int rowsPerChunk, T* chunkMax)
{
    const PartialsProduct<T>& product = batch.products[blockIdx.z];
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    const int rowEnd = min(nRows, (blockIdx.y + 1) * rowsPerChunk);
    T m = 0;
    if (p < nPatterns and product.scalingFactors)
        for (int r = blockIdx.y * rowsPerChunk + threadIdx.y; r < rowEnd; r += blockDim.y)
        {
            const size_t i = (size_t)r * nPatterns + p;
            m = max(m, product.right ? product.left[i] * product.right[i] : product.left[i]);
        }
    m = column_reduce(m, MaxOp());
    if (threadIdx.y == 0 and p < nPatterns)
        chunkMax[((size_t)blockIdx.z * gridDim.y + blockIdx.y) * nPatterns + p] = m;
}

// parent = left * right / m, where m is the largest entry of the pattern (1 if all are 0, or without rescaling).
template <typename T>
__global__ void partials_product_scale_kernel(const PartialsProductBatch<T> batch, int nRows, int nPatterns, int rowsPerChunk,
                                              const T* chunkMax, bool scalersLog)
{
    const PartialsProduct<T>& product = batch.products[blockIdx.z];
    const int p = blockIdx.x * blockDim.x + threadIdx.x;

    // One row of threads combines the chunks' maxima for the block's patterns.
    __shared__ T patternMax[32];
    if (threadIdx.y == 0)
    {
        T m = 0;
        if (p < nPatterns)
            for (int c = 0; c < gridDim.y; c++)
                m = max(m, chunkMax[((size_t)blockIdx.z * gridDim.y + c) * nPatterns + p]);
        patternMax[threadIdx.x] = (m == 0) ? T(1) : m;
    }
    __syncthreads();
    if (p >= nPatterns) return;
    const T m = patternMax[threadIdx.x];
    const T inverse = product.scalingFactors ? 1 / m : fixed_inverse(product, p, scalersLog);

    const int rowEnd = min(nRows, (blockIdx.y + 1) * rowsPerChunk);
    for (int r = blockIdx.y * rowsPerChunk + threadIdx.y; r < rowEnd; r += blockDim.y)
    {
        const size_t i = (size_t)r * nPatterns + p;
        product.parent[i] = (product.right ? product.left[i] * product.right[i] : product.left[i]) * inverse;
    }

    if (product.scalingFactors and blockIdx.y == 0 and threadIdx.y == 0)
        product.scalingFactors[p] = scalersLog ? log(m) : m;
}

// cumulative[p] += log m of each product in turn (the order of separate calls, so the rounding is the same).
template <typename T>
__global__ void accumulate_scaling_kernel(const PartialsProductBatch<T> batch, int nPatterns, T* cumulative, bool scalersLog)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= nPatterns) return;
    T sum = cumulative[p];
    for (int k = 0; k < batch.count; k++)
    {
        const T* factors = batch.products[k].scalingFactors;
        if (factors)
            sum += scalersLog ? factors[p] : log(factors[p]);
    }
    cumulative[p] = sum;
}

template <typename T>
void cuda_partials_products_impl(const PartialsProductBatch<T>& batch, int nRows, int nPatterns, T* cumulative, bool scalersLog,
                                 T* scratch, cudaStream_t stream)
{
    if (batch.count == 0) return;
    bool rescale = false;
    for (int k = 0; k < batch.count; k++)
        rescale = rescale or batch.products[k].scalingFactors;

    if (not rescale)
    {
        const size_t n = (size_t)nRows * nPatterns;
        const int blocks = (int)std::min<size_t>((n + 255) / 256, 4096 / batch.count + 1);
        partials_product_kernel<<<dim3(blocks, batch.count), 256, 0, stream>>>(batch, n, nPatterns, scalersLog);
        return;
    }

    const RowChunks c = rowChunks(nRows, nPatterns);
    const dim3 block(c.patternsPerBlock, c.rowsPerBlock);
    const dim3 grid(c.patternBlocks, c.chunks, batch.count);
    partials_product_max_kernel<<<grid, block, 0, stream>>>(batch, nRows, nPatterns, c.rowsPerChunk, scratch);
    partials_product_scale_kernel<<<grid, block, 0, stream>>>(batch, nRows, nPatterns, c.rowsPerChunk, scratch, scalersLog);
    if (cumulative)
        accumulate_scaling_kernel<<<(nPatterns + 255) / 256, 256, 0, stream>>>(batch, nPatterns, cumulative, scalersLog);
}

void cuda_partials_products(const PartialsProductBatch<float>& batch, int nRows, int nPatterns, float* cumulative, bool scalersLog,
                            float* scratch, cudaStream_t stream)
{
    cuda_partials_products_impl(batch, nRows, nPatterns, cumulative, scalersLog, scratch, stream);
}

void cuda_partials_products(const PartialsProductBatch<double>& batch, int nRows, int nPatterns, double* cumulative, bool scalersLog,
                            double* scratch, cudaStream_t stream)
{
    cuda_partials_products_impl(batch, nRows, nPatterns, cumulative, scalersLog, scratch, stream);
}

template <typename T>
__global__ void accumulate_scale_factors_kernel(const ScaleFactorBatch<T> batch, int nPatterns, T* cumulative, bool scalersLog, bool subtract)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= nPatterns) return;
    T total = 0;
    for (int k = 0; k < batch.count; k++)
    {
        const T factor = batch.factors[k][p];
        if (scalersLog)
            total += factor;
        else if (factor != 1)
            total += log(factor);
    }
    cumulative[p] += subtract ? -total : total;
}

template <typename T>
void cuda_accumulate_scale_factors_impl(const ScaleFactorBatch<T>& batch, int nPatterns, T* cumulative, bool scalersLog,
                                        bool subtract, cudaStream_t stream)
{
    if (batch.count == 0) return;
    accumulate_scale_factors_kernel<<<(nPatterns + 255) / 256, 256, 0, stream>>>(batch, nPatterns, cumulative, scalersLog, subtract);
}

void cuda_accumulate_scale_factors(const ScaleFactorBatch<float>& batch, int nPatterns, float* cumulative, bool scalersLog,
                                   bool subtract, cudaStream_t stream)
{
    cuda_accumulate_scale_factors_impl(batch, nPatterns, cumulative, scalersLog, subtract, stream);
}

void cuda_accumulate_scale_factors(const ScaleFactorBatch<double>& batch, int nPatterns, double* cumulative, bool scalersLog,
                                   bool subtract, cudaStream_t stream)
{
    cuda_accumulate_scale_factors_impl(batch, nPatterns, cumulative, scalersLog, subtract, stream);
}

// out(c, s, p) = frequencies[s] for p < nPatterns, 0 for the padded patterns.
template <typename T>
__global__ void root_pre_partials_kernel(T* out, const T* frequencies, int nStates, int nRows, int nPaddedPatterns, int nPatterns)
{
    const size_t n = (size_t)nRows * nPaddedPatterns;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    {
        const int r = i / nPaddedPatterns;
        const int p = i % nPaddedPatterns;
        out[i] = (p < nPatterns) ? frequencies[r % nStates] : T(0);
    }
}

template <typename T>
void cuda_root_pre_partials_impl(T* out, const T* frequencies, int nStates, int nCategories, int nPaddedPatterns, int nPatterns,
                                 cudaStream_t stream)
{
    const size_t n = (size_t)nStates * nCategories * nPaddedPatterns;
    const int blocks = (int)std::min<size_t>((n + 255) / 256, 4096);
    root_pre_partials_kernel<<<blocks, 256, 0, stream>>>(out, frequencies, nStates, nStates * nCategories, nPaddedPatterns, nPatterns);
}

void cuda_root_pre_partials(float* out, const float* frequencies, int nStates, int nCategories, int nPaddedPatterns, int nPatterns,
                            cudaStream_t stream)
{
    cuda_root_pre_partials_impl(out, frequencies, nStates, nCategories, nPaddedPatterns, nPatterns, stream);
}

void cuda_root_pre_partials(double* out, const double* frequencies, int nStates, int nCategories, int nPaddedPatterns, int nPatterns,
                            cudaStream_t stream)
{
    cuda_root_pre_partials_impl(out, frequencies, nStates, nCategories, nPaddedPatterns, nPatterns, stream);
}

// For node z = blockIdx.z, over the chunk's rows r = (c, s):
//   chunkNumerator[z, chunk, p]   = sum w_c m_c pre(r, p) (D post)(r, p),  D = A + shift I in CSR (if rowOffsets)
//   chunkDenominator[z, chunk, p] = sum w_c pre(r, p) post(r, p)
// D acts within each category's (nStates, nPatterns) block.
template <typename T>
__global__ void edge_derivative_sums_kernel(const EdgeDerivativeBatch<T> batch, int nStates, int nRows, int nPatterns, int rowsPerChunk,
                                            const int* __restrict__ rowOffsets, const int* __restrict__ columns,
                                            const T* __restrict__ values, T shift, const T* __restrict__ categoryWeights,
                                            T* chunkNumerator, T* chunkDenominator)
{
    const T* post = batch.post[blockIdx.z];
    const T* pre = batch.pre[blockIdx.z];
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    const int rowEnd = min(nRows, (blockIdx.y + 1) * rowsPerChunk);
    T numerator = 0;
    T denominator = 0;
    if (p < nPatterns)
        for (int r = blockIdx.y * rowsPerChunk + threadIdx.y; r < rowEnd; r += blockDim.y)
        {
            const int c = r / nStates;
            const int s = r % nStates;
            const size_t i = (size_t)r * nPatterns + p;
            const T w = categoryWeights[c];
            const T own = post[i];
            denominator += w * pre[i] * own;
            if (rowOffsets)
            {
                const T* block = post + (size_t)c * nStates * nPatterns + p;
                T sum = shift * own;
                for (int k = rowOffsets[s]; k < rowOffsets[s + 1]; k++)
                    sum += values[k] * block[(size_t)columns[k] * nPatterns];
                numerator += w * batch.categoryMultipliers[c] * pre[i] * sum;
            }
        }
    numerator = column_reduce(numerator, SumOp());
    __syncthreads();
    denominator = column_reduce(denominator, SumOp());
    if (threadIdx.y == 0 and p < nPatterns)
    {
        const size_t o = ((size_t)blockIdx.z * gridDim.y + blockIdx.y) * nPatterns + p;
        chunkNumerator[o] = numerator;
        chunkDenominator[o] = denominator;
    }
}

// For node z = blockIdx.x: derivatives[z, p] = numerator / denominator (each summed over the chunks), and
// sums[z] = sum_p patternWeights[p] * derivative, sumSquares[z] = sum_p patternWeights[p] * derivative^2.
// With numerators == nullptr, derivatives[z, p] = patternWeights[p] / denominator instead (no sums).
template <typename T>
__global__ void edge_derivative_finish_kernel(const T* chunkNumerator, const T* chunkDenominator, int nChunks, int nPaddedPatterns,
                                              int nPatterns, const T* patternWeights, T* derivatives, T* sums, T* sumSquares)
{
    const size_t base = (size_t)blockIdx.x * nChunks * nPaddedPatterns;
    T sum = 0;
    T sumSquare = 0;
    for (int p = threadIdx.x; p < nPatterns; p += blockDim.x)
    {
        T numerator = 0;
        T denominator = 0;
        for (int c = 0; c < nChunks; c++)
        {
            denominator += chunkDenominator[base + (size_t)c * nPaddedPatterns + p];
            if (chunkNumerator)
                numerator += chunkNumerator[base + (size_t)c * nPaddedPatterns + p];
        }
        if (not chunkNumerator)
        {
            derivatives[(size_t)blockIdx.x * nPatterns + p] = patternWeights[p] / denominator;
            continue;
        }
        const T derivative = numerator / denominator;
        derivatives[(size_t)blockIdx.x * nPatterns + p] = derivative;
        sum += derivative * patternWeights[p];
        sumSquare += derivative * derivative * patternWeights[p];
    }
    if (not chunkNumerator) return;
    sum = block_reduce(sum, T(0), SumOp());
    __syncthreads();
    sumSquare = block_reduce(sumSquare, T(0), SumOp());
    if (threadIdx.x == 0)
    {
        sums[blockIdx.x] = sum;
        sumSquares[blockIdx.x] = sumSquare;
    }
}

template <typename T>
void cuda_edge_derivatives_impl(const EdgeDerivativeBatch<T>& batch, int nStates, int nCategories, int nPaddedPatterns, int nPatterns,
                                const int* rowOffsets, const int* columns, const T* values, T shift, const T* categoryWeights,
                                const T* patternWeights, T* scratch, T* derivatives, T* sums, T* sumSquares, cudaStream_t stream)
{
    if (batch.count == 0) return;
    const int nRows = nStates * nCategories;
    const RowChunks c = rowChunks(nRows, nPaddedPatterns);
    const dim3 block(c.patternsPerBlock, c.rowsPerBlock);
    const dim3 grid(c.patternBlocks, c.chunks, batch.count);
    T* chunkNumerator = rowOffsets ? scratch : nullptr;
    T* chunkDenominator = scratch + (size_t)batch.count * c.chunks * nPaddedPatterns;
    edge_derivative_sums_kernel<<<grid, block, 0, stream>>>(batch, nStates, nRows, nPaddedPatterns, c.rowsPerChunk,
                                                            rowOffsets, columns, values, shift, categoryWeights,
                                                            scratch, chunkDenominator);
    edge_derivative_finish_kernel<<<batch.count, 256, 0, stream>>>(chunkNumerator, chunkDenominator, c.chunks, nPaddedPatterns, nPatterns,
                                                                   patternWeights, derivatives, sums, sumSquares);
}

size_t cuda_edge_derivative_scratch_size(int nStates, int nCategories, int nPaddedPatterns)
{
    return (size_t)2 * kMaxEdgeDerivativeNodes * rowChunks(nStates * nCategories, nPaddedPatterns).chunks * nPaddedPatterns;
}

void cuda_edge_derivatives(const EdgeDerivativeBatch<float>& batch, int nStates, int nCategories, int nPaddedPatterns, int nPatterns,
                           const int* rowOffsets, const int* columns, const float* values, float shift, const float* categoryWeights,
                           const float* patternWeights, float* scratch, float* derivatives, float* sums, float* sumSquares, cudaStream_t stream)
{
    cuda_edge_derivatives_impl(batch, nStates, nCategories, nPaddedPatterns, nPatterns, rowOffsets, columns, values, shift, categoryWeights,
                               patternWeights, scratch, derivatives, sums, sumSquares, stream);
}

void cuda_edge_derivatives(const EdgeDerivativeBatch<double>& batch, int nStates, int nCategories, int nPaddedPatterns, int nPatterns,
                           const int* rowOffsets, const int* columns, const double* values, double shift, const double* categoryWeights,
                           const double* patternWeights, double* scratch, double* derivatives, double* sums, double* sumSquares, cudaStream_t stream)
{
    cuda_edge_derivatives_impl(batch, nStates, nCategories, nPaddedPatterns, nPatterns, rowOffsets, columns, values, shift, categoryWeights,
                               patternWeights, scratch, derivatives, sums, sumSquares, stream);
}

// crossProducts[k] += sum_{c,p} multipliers[c] * patternScale[p] * pre(c, i, p) * post(c, j, p) for each non-zero k = (i, j)
// of the CSR pattern.  One warp per row i; the lanes cover (category, pattern) pairs.
template <typename T>
__global__ void cross_products_kernel(const T* pre, const T* post, int nStates, int nCategories, int nPatterns,
                                      const int* __restrict__ rowOffsets, const int* __restrict__ columns,
                                      const CategoryMultipliers<T> multipliers, const T* __restrict__ patternScale, T* crossProducts)
{
    const int lane = threadIdx.x % 32;
    const int i = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (i >= nStates) return;
    const size_t categoryStride = (size_t)nStates * nPatterns;
    const int n = nCategories * nPatterns;
    for (int k = rowOffsets[i]; k < rowOffsets[i + 1]; k++)
    {
        const int j = columns[k];
        T sum = 0;
        for (int q = lane; q < n; q += 32)
        {
            const int c = q / nPatterns;
            const int p = q % nPatterns;
            sum += multipliers.values[c] * patternScale[p] * pre[c * categoryStride + (size_t)i * nPatterns + p] *
                   post[c * categoryStride + (size_t)j * nPatterns + p];
        }
        for (int offset = 16; offset > 0; offset /= 2)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (lane == 0)
            crossProducts[k] += sum;
    }
}

template <typename T>
void cuda_cross_products_impl(const T* pre, const T* post, int nStates, int nCategories, int nPatterns, const int* rowOffsets,
                              const int* columns, const CategoryMultipliers<T>& multipliers, const T* patternScale, T* crossProducts,
                              cudaStream_t stream)
{
    cross_products_kernel<<<(nStates + 7) / 8, 256, 0, stream>>>(pre, post, nStates, nCategories, nPatterns, rowOffsets, columns,
                                                                 multipliers, patternScale, crossProducts);
}

void cuda_cross_products(const float* pre, const float* post, int nStates, int nCategories, int nPatterns, const int* rowOffsets,
                         const int* columns, const CategoryMultipliers<float>& multipliers, const float* patternScale, float* crossProducts,
                         cudaStream_t stream)
{
    cuda_cross_products_impl(pre, post, nStates, nCategories, nPatterns, rowOffsets, columns, multipliers, patternScale, crossProducts, stream);
}

void cuda_cross_products(const double* pre, const double* post, int nStates, int nCategories, int nPatterns, const int* rowOffsets,
                         const int* columns, const CategoryMultipliers<double>& multipliers, const double* patternScale, double* crossProducts,
                         cudaStream_t stream)
{
    cuda_cross_products_impl(pre, post, nStates, nCategories, nPatterns, rowOffsets, columns, multipliers, patternScale, crossProducts, stream);
}

// chunkSum[chunk, p] = sum over the chunk's rows r = (c, s) of root[r, p] * categoryWeights[c] * frequencies[s]
template <typename T>
__global__ void root_chunk_sums_kernel(const T* root, const T* categoryWeights, const T* frequencies, int nStates, int nRows,
                                       int nPatterns, int rowsPerChunk, T* chunkSum)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    const int rowEnd = min(nRows, (blockIdx.y + 1) * rowsPerChunk);
    T sum = 0;
    if (p < nPatterns)
        for (int r = blockIdx.y * rowsPerChunk + threadIdx.y; r < rowEnd; r += blockDim.y)
            sum += root[(size_t)r * nPatterns + p] * categoryWeights[r / nStates] * frequencies[r % nStates];
    sum = column_reduce(sum, SumOp());
    if (threadIdx.y == 0 and p < nPatterns)
        chunkSum[blockIdx.y * nPatterns + p] = sum;
}

// siteLogLikelihoods[p] = log(sum of the chunk sums) + cumulativeScaling[p];  out[0] = sum_p patternWeights[p] * siteLogLikelihoods[p].
// A single block, so the sum is deterministic.
template <typename T>
__global__ void root_finish_kernel(const T* chunkSum, int nChunks, int nPaddedPatterns, int nPatterns, const T* cumulativeScaling,
                                   const T* patternWeights, T* siteLogLikelihoods, T* out)
{
    T total = 0;
    for (int p = threadIdx.x; p < nPatterns; p += blockDim.x)
    {
        T sum = 0;
        for (int c = 0; c < nChunks; c++)
            sum += chunkSum[c * nPaddedPatterns + p];
        const T site = log(sum) + (cumulativeScaling ? cumulativeScaling[p] : T(0));
        siteLogLikelihoods[p] = site;
        total += site * patternWeights[p];
    }
    total = block_reduce(total, T(0), SumOp());
    if (threadIdx.x == 0) out[0] = total;
}

template <typename T>
void cuda_root_log_likelihoods_impl(const T* root, const T* categoryWeights, const T* frequencies, const T* cumulativeScaling,
                                    const T* patternWeights, int nStates, int nPaddedPatterns, int nPatterns, int nCategories,
                                    T* scratch, T* siteLogLikelihoods, T* sumLogLikelihood, cudaStream_t stream)
{
    const int nRows = nStates * nCategories;
    const RowChunks c = rowChunks(nRows, nPaddedPatterns);
    const dim3 block(c.patternsPerBlock, c.rowsPerBlock);
    const dim3 grid(c.patternBlocks, c.chunks);
    root_chunk_sums_kernel<<<grid, block, 0, stream>>>(root, categoryWeights, frequencies, nStates, nRows, nPaddedPatterns,
                                                       c.rowsPerChunk, scratch);
    root_finish_kernel<<<1, 1024, 0, stream>>>(scratch, c.chunks, nPaddedPatterns, nPatterns, cumulativeScaling, patternWeights,
                                               siteLogLikelihoods, sumLogLikelihood);
}

void cuda_root_log_likelihoods(const float* root, const float* categoryWeights, const float* frequencies, const float* cumulativeScaling,
                               const float* patternWeights, int nStates, int nPaddedPatterns, int nPatterns, int nCategories,
                               float* scratch, float* siteLogLikelihoods, float* sumLogLikelihood, cudaStream_t stream)
{
    cuda_root_log_likelihoods_impl(root, categoryWeights, frequencies, cumulativeScaling, patternWeights,
                                   nStates, nPaddedPatterns, nPatterns, nCategories, scratch, siteLogLikelihoods, sumLogLikelihood, stream);
}

void cuda_root_log_likelihoods(const double* root, const double* categoryWeights, const double* frequencies, const double* cumulativeScaling,
                               const double* patternWeights, int nStates, int nPaddedPatterns, int nPatterns, int nCategories,
                               double* scratch, double* siteLogLikelihoods, double* sumLogLikelihood, cudaStream_t stream)
{
    cuda_root_log_likelihoods_impl(root, categoryWeights, frequencies, cumulativeScaling, patternWeights,
                                   nStates, nPaddedPatterns, nPatterns, nCategories, scratch, siteLogLikelihoods, sumLogLikelihood, stream);
}


// ---------------------------------------------------------------------------------------------------
// Block 1-norm estimation of ||B^p||_1 for all p at once (Higham & Tisseur Algorithm 2.4, 2 iterations).
// All matrices are row-major with n rows and an explicit row stride.  See BlockNormEstimator in
// BeagleGPUActionImpl.h.

// y(:, 0:nApplied) = A * x(:, 0:nApplied), then nAppend more columns of y are set:
//   from signSource (n, nAppend):  y(i, c) = +1 if signSource(i, c) >= 0, else -1
//   from unitRows   (nAppend):     y(i, c) = 1 if i == unitRows[c], else 0
// LANES lanes share a row and cover its columns, so the gathers of x are coalesced.
template <typename T, int LANES>
__global__ void norm_chain_step_kernel(int n, const int* __restrict__ rowOffsets, const int* __restrict__ columns,
                                       const T* __restrict__ values, const T* __restrict__ x, int ldx, T* __restrict__ y, int ldy,
                                       int nApplied, int nAppend, const T* __restrict__ signSource, int ldSign,
                                       const int* __restrict__ unitRows)
{
    const int lane = threadIdx.x % 32;
    const int row = (blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32) * (32 / LANES) + lane / LANES;
    if (row >= n) return;
    T* yRow = y + (size_t)row * ldy;

    const int kBegin = rowOffsets[row];
    const int kEnd = rowOffsets[row + 1];
    for (int c = lane % LANES; c < nApplied; c += LANES)
    {
        T sum = 0;
        for (int k = kBegin; k < kEnd; k++)
            sum += values[k] * x[(size_t)columns[k] * ldx + c];
        yRow[c] = sum;
    }
    for (int c = lane % LANES; c < nAppend; c += LANES)
    {
        if (signSource)
            yRow[nApplied + c] = (signSource[(size_t)row * ldSign + c] >= 0) ? T(1) : T(-1);
        else
            yRow[nApplied + c] = (row == unitRows[c]) ? T(1) : T(0);
    }
}

template <typename T, int LANES>
void launch_norm_chain_step(int n, const int* rowOffsets, const int* columns, const T* values, const T* x, int ldx, T* y, int ldy,
                            int nApplied, int nAppend, const T* signSource, int ldSign, const int* unitRows, cudaStream_t stream)
{
    const int threads = 256;
    const int rowsPerBlock = (threads / 32) * (32 / LANES);
    norm_chain_step_kernel<T, LANES><<<(n + rowsPerBlock - 1) / rowsPerBlock, threads, 0, stream>>>(
        n, rowOffsets, columns, values, x, ldx, y, ldy, nApplied, nAppend, signSource, ldSign, unitRows);
}

template <typename T>
void cuda_norm_chain_step_impl(int n, const int* rowOffsets, const int* columns, const T* values, const T* x, int ldx, T* y, int ldy,
                               int nApplied, int nAppend, const T* signSource, int ldSign, const int* unitRows, cudaStream_t stream)
{
    switch (nextPowerOfTwo(std::max(nApplied, nAppend), 32))
    {
        case 1:
        case 2:
        case 4:  launch_norm_chain_step<T, 4>(n, rowOffsets, columns, values, x, ldx, y, ldy, nApplied, nAppend, signSource, ldSign, unitRows, stream); break;
        case 8:  launch_norm_chain_step<T, 8>(n, rowOffsets, columns, values, x, ldx, y, ldy, nApplied, nAppend, signSource, ldSign, unitRows, stream); break;
        case 16: launch_norm_chain_step<T, 16>(n, rowOffsets, columns, values, x, ldx, y, ldy, nApplied, nAppend, signSource, ldSign, unitRows, stream); break;
        default: launch_norm_chain_step<T, 32>(n, rowOffsets, columns, values, x, ldx, y, ldy, nApplied, nAppend, signSource, ldSign, unitRows, stream); break;
    }
}

void cuda_norm_chain_step(int n, const int* rowOffsets, const int* columns, const float* values, const float* x, int ldx, float* y, int ldy,
                          int nApplied, int nAppend, const float* signSource, int ldSign, const int* unitRows, cudaStream_t stream)
{
    cuda_norm_chain_step_impl(n, rowOffsets, columns, values, x, ldx, y, ldy, nApplied, nAppend, signSource, ldSign, unitRows, stream);
}

void cuda_norm_chain_step(int n, const int* rowOffsets, const int* columns, const double* values, const double* x, int ldx, double* y, int ldy,
                          int nApplied, int nAppend, const double* signSource, int ldSign, const int* unitRows, cudaStream_t stream)
{
    cuda_norm_chain_step_impl(n, rowOffsets, columns, values, x, ldx, y, ldy, nApplied, nAppend, signSource, ldSign, unitRows, stream);
}

// sums[c] = sum_i |x(i, c)|, one block per column.
template <typename T>
__global__ void column_abs_sums_kernel(const T* x, int n, int ld, T* sums)
{
    const int c = blockIdx.x;
    T sum = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        sum += abs(x[(size_t)i * ld + c]);
    sum = block_reduce(sum, T(0), SumOp());
    if (threadIdx.x == 0) sums[c] = sum;
}

template <typename T>
void cuda_column_abs_sums_impl(const T* x, int n, int nColumns, int ld, T* sums, cudaStream_t stream)
{
    column_abs_sums_kernel<<<nColumns, 256, 0, stream>>>(x, n, ld, sums);
}

void cuda_column_abs_sums(const float* x, int n, int nColumns, int ld, float* sums, cudaStream_t stream)
{
    cuda_column_abs_sums_impl(x, n, nColumns, ld, sums, stream);
}

void cuda_column_abs_sums(const double* x, int n, int nColumns, int ld, double* sums, cudaStream_t stream)
{
    cuda_column_abs_sums_impl(x, n, nColumns, ld, sums, stream);
}

// dst[i] = src[index[i]]
template <typename T>
__global__ void gather_kernel(const T* src, const int* index, int n, T* dst)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[index[i]];
}

template <typename T>
void cuda_gather_impl(const T* src, const int* index, int n, T* dst, cudaStream_t stream)
{
    if (n > 0)
        gather_kernel<<<(n + 255) / 256, 256, 0, stream>>>(src, index, n, dst);
}

void cuda_gather(const float* src, const int* index, int n, float* dst, cudaStream_t stream)
{
    cuda_gather_impl(src, index, n, dst, stream);
}

void cuda_gather(const double* src, const int* index, int n, double* dst, cudaStream_t stream)
{
    cuda_gather_impl(src, index, n, dst, stream);
}

// rowMax[b * n + i] = max_j |Z(i, b*t + j)| for each block b of t columns of Z (row stride ld).
template <typename T>
__global__ void block_row_max_kernel(const T* Z, int n, int t, int nBlocks, int ld, T* rowMax)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n * nBlocks) return;
    const int b = index / n;
    const int i = index % n;
    const T* z = Z + (size_t)i * ld + b * t;
    T h = 0;
    for (int j = 0; j < t; j++)
        h = max(h, abs(z[j]));
    rowMax[index] = h;
}

// unitRows[b*t + r] = the row with the r-th largest rowMax[b * n + i] (ties go to the lower row).
// One thread block per column block.
template <typename T>
__global__ void select_unit_columns_kernel(const T* rowMax, int n, int t, int* unitRows)
{
    __shared__ T bestValue[1024];
    __shared__ int bestRow[1024];
    __shared__ int chosen[32];

    const T* h = rowMax + (size_t)blockIdx.x * n;

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

            if (h[i] > value) { value = h[i]; row = i; }
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
void cuda_select_unit_columns_impl(const T* Z, int n, int t, int nBlocks, int ld, T* rowMax, int* unitRows, cudaStream_t stream)
{
    block_row_max_kernel<<<(n * nBlocks + 255) / 256, 256, 0, stream>>>(Z, n, t, nBlocks, ld, rowMax);
    select_unit_columns_kernel<<<nBlocks, 1024, 0, stream>>>(rowMax, n, t, unitRows);
}

void cuda_select_unit_columns(const float* Z, int n, int t, int nBlocks, int ld, float* rowMax, int* unitRows, cudaStream_t stream)
{
    cuda_select_unit_columns_impl(Z, n, t, nBlocks, ld, rowMax, unitRows, stream);
}

void cuda_select_unit_columns(const double* Z, int n, int t, int nBlocks, int ld, double* rowMax, int* unitRows, cudaStream_t stream)
{
    cuda_select_unit_columns_impl(Z, n, t, nBlocks, ld, rowMax, unitRows, stream);
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
