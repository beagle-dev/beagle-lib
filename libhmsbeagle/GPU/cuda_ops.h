#ifndef CUDA_OPS_H
#define CUDA_OPS_H

#include <vector>
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <stdexcept>          // for std::runtime_error


inline void* cudaMallocWrapped(size_t n)
{
    void* result;
    auto status = cudaMalloc((void**)&result, n);
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("cudaMalloc: ") + std::string(cudaGetErrorString(status)));
    return result;
}

template <typename T>
T* cudaDeviceNew(size_t n)
{
    return (T*)cudaMallocWrapped(n*sizeof(T));
}

template <typename T>
void cudaDeviceDelete(T* buffer)
{
    if (buffer)
    {
        auto status = cudaFree(buffer);
        if (status != cudaSuccess)
            throw std::runtime_error(std::string("cudaFree: ") + std::string(cudaGetErrorString(status)));
    }
}

template <typename T>
void MemcpyHostToDevice(T* dptr, const T* hptr, int n)
{
    auto status = cudaMemcpy(dptr, hptr, n*sizeof(T), cudaMemcpyHostToDevice);
    if (status != cudaSuccess)
	throw std::runtime_error("cudaMemcpy(Host->Device): " + std::string(cudaGetErrorString(status)));
}

template <typename T>
void MemcpyDeviceToDevice(T* dptr, const T* hptr, int n)
{
    auto status = cudaMemcpy(dptr, hptr, n*sizeof(T), cudaMemcpyDeviceToDevice);
    if (status != cudaSuccess)
	throw std::runtime_error("cudaMemcpy(Device->Device): " + std::string(cudaGetErrorString(status)));
}

template <typename T>
void MemcpyDeviceToDeviceAsync(T* dptr, const T* hptr, int n)
{
	auto status = cudaMemcpyAsync(dptr, hptr, n*sizeof(T), cudaMemcpyDeviceToDevice);
	if (status != cudaSuccess)
		throw std::runtime_error("cudaMemcpyAsync(Device->Device): " + std::string(cudaGetErrorString(status)));
}


template <typename T>
void MemcpyDeviceToHost(T* hptr, const T* dptr, int n)
{
    auto status = cudaMemcpy(hptr, dptr, n*sizeof(T), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess)
	throw std::runtime_error(std::string("cudaMemcpy(Device->Host): ") + std::string(cudaGetErrorString(status)));
}

template <typename T>
std::vector<T> MemcpyDeviceToHostVector(const T* dptr, int n)
{
    std::vector<T> host_vec(n);
    auto status = cudaMemcpy(host_vec.data(), dptr, n*sizeof(T), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess)
	throw std::runtime_error("cudaMemcpy(Device->Host): failed!");
    return host_vec;
}

// max_i |v[i]| (synchronizes with the host).
double cuda_max_abs(double* v, int length);
float cuda_max_abs(float* v, int length);

// The starting matrix X (n, t) of the block 1-norm estimator: first column 1/n, others random +-1/n.
void initialize_norm_x_matrix(float* data, int n, int m);
void initialize_norm_x_matrix(double* data, int n, int m);

// Block 1-norm estimation, see BlockNormEstimator in BeagleGPUActionImpl.h.
// y(:, 0:nApplied) = A x(:, 0:nApplied) for column-major y, x with n rows and A in CSR; then nAppend more
// columns of y are set to sign(signSource) if signSource is non-null, otherwise to unit vectors e(unitRows[c]).
void cuda_norm_chain_step(int n, const int* rowOffsets, const int* columns, const float* values, const float* x, float* y,
                          int nApplied, int nAppend, const float* signSource, const int* unitRows, cudaStream_t stream);
void cuda_norm_chain_step(int n, const int* rowOffsets, const int* columns, const double* values, const double* x, double* y,
                          int nApplied, int nAppend, const double* signSource, const int* unitRows, cudaStream_t stream);

// sums[c] = sum_i |x(i, c)| for the nColumns columns of x (n rows).
void cuda_column_abs_sums(const float* x, int n, int nColumns, float* sums, cudaStream_t stream);
void cuda_column_abs_sums(const double* x, int n, int nColumns, double* sums, cudaStream_t stream);

// For each of nBlocks blocks of t columns of Z (n rows): the t rows with the largest max_j |Z(i,j)| -> unitRows[b*t ...].
void cuda_select_unit_columns(const float* Z, int n, int t, int nBlocks, int* unitRows, cudaStream_t stream);
void cuda_select_unit_columns(const double* Z, int n, int t, int nBlocks, int* unitRows, cudaStream_t stream);

// norms[p] for p = 2..pTop: the largest column sum for p in either iteration (see the kernel).
void cuda_combine_norm_estimates(const float* sums1, const float* sums2, int t, int pTop, float* norms, cudaStream_t stream);
void cuda_combine_norm_estimates(const double* sums1, const double* sums2, int t, int pTop, double* norms, cudaStream_t stream);

// Action partials are stored pattern-fastest, [category][state][pattern]: a (nRows, nPatterns) row-major
// matrix with nRows = nCategories * nStates.

// One Horner step of a Taylor polynomial for several independent (nRows, nCols) row-major matrices that share A:
//   out = scale * (b + alpha * A * x)
// Each matrix has its own row stride (ld*, in elements), so the groups' matrices can be stored side by side in
// one wider matrix.  out must not alias x or b.
template <typename T>
struct TaylorTermGroup
{
    const T* x;
    const T* b;
    T* out;
    T alpha;
    T scale;
    int ldx;
    int ldb;
    int ldOut;
};

// 64 groups keep the kernel parameters under 4 KB.
constexpr int kMaxTaylorGroups = 64;

template <typename T>
struct TaylorTermBatch
{
    int count = 0;
    TaylorTermGroup<T> groups[kMaxTaylorGroups];
};

void cuda_taylor_horner_terms(int nRows, int nCols, const int* rowOffsets, const int* columns, const float* values,
                              const TaylorTermBatch<float>& batch, cudaStream_t stream);
void cuda_taylor_horner_terms(int nRows, int nCols, const int* rowOffsets, const int* columns, const double* values,
                              const TaylorTermBatch<double>& batch, cudaStream_t stream);

// The product and root kernels need a scratch buffer of cuda_row_chunk_count(nRows, nPatterns) * nPatterns values
// (per product, for the products).
int cuda_row_chunk_count(int nRows, int nPatterns);

// parent = left * right for (nRows, nPatterns) partials.  If scalingFactors is non-null, also rescale each
// pattern by its largest entry m and write m (log m if scalersLog) to scalingFactors[pattern].
template <typename T>
struct PartialsProduct
{
    const T* left;
    const T* right;
    T* parent;
    T* scalingFactors;
};

constexpr int kMaxPartialsProducts = 16;

template <typename T>
struct PartialsProductBatch
{
    int count = 0;
    PartialsProduct<T> products[kMaxPartialsProducts];
};

// All the products of the batch, then (if cumulative is non-null) cumulative[pattern] += log m of each in turn.
void cuda_partials_products(const PartialsProductBatch<float>& batch, int nRows, int nPatterns, float* cumulative, bool scalersLog,
                            float* scratch, cudaStream_t stream);
void cuda_partials_products(const PartialsProductBatch<double>& batch, int nRows, int nPatterns, double* cumulative, bool scalersLog,
                            double* scratch, cudaStream_t stream);

// cumulative[p] += (or -= if subtract) the sum over the buffers of factors[p] (log factors[p] unless scalersLog).
template <typename T>
struct ScaleFactorBatch
{
    int count = 0;
    const T* factors[64];
};

void cuda_accumulate_scale_factors(const ScaleFactorBatch<float>& batch, int nPatterns, float* cumulative, bool scalersLog,
                                   bool subtract, cudaStream_t stream);
void cuda_accumulate_scale_factors(const ScaleFactorBatch<double>& batch, int nPatterns, double* cumulative, bool scalersLog,
                                   bool subtract, cudaStream_t stream);

// siteLogLikelihoods[p] = log(sum_{c,s} root(c,s,p) * categoryWeights[c] * frequencies[s]) + cumulativeScaling[p] (if non-null),
// sumLogLikelihood[0] = sum_p patternWeights[p] * siteLogLikelihoods[p].
void cuda_root_log_likelihoods(const float* root, const float* categoryWeights, const float* frequencies, const float* cumulativeScaling,
                               const float* patternWeights, int nStates, int nPaddedPatterns, int nPatterns, int nCategories,
                               float* scratch, float* siteLogLikelihoods, float* sumLogLikelihood, cudaStream_t stream);
void cuda_root_log_likelihoods(const double* root, const double* categoryWeights, const double* frequencies, const double* cumulativeScaling,
                               const double* patternWeights, int nStates, int nPaddedPatterns, int nPatterns, int nCategories,
                               double* scratch, double* siteLogLikelihoods, double* sumLogLikelihood, cudaStream_t stream);
#endif
