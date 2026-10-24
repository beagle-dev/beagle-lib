#ifndef CUDA_OPS_H
#define CUDA_OPS_H

#include <vector>
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <stdexcept>          // for std::runtime_error


inline void* cudaMallocWrapped(int n)
{
    void* result;
    auto status = cudaMalloc((void**)&result, n);
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("cudaMalloc: ") + std::string(cudaGetErrorString(status)));
    return result;
}

template <typename T>
T* cudaDeviceNew(int n)
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

// parent = left * right for (nStates, nPatterns, nCategories) partials.  If scalingFactors is non-null, also
// rescale each pattern by its largest entry m, write m (log m if scalersLog) to scalingFactors[pattern],
// and add log m to cumulative[pattern] if cumulative is non-null.
void cuda_partials_product(const float* left, const float* right, float* parent, int nStates, int nPatterns, int nCategories,
                           float* scalingFactors, float* cumulative, bool scalersLog, cudaStream_t stream);
void cuda_partials_product(const double* left, const double* right, double* parent, int nStates, int nPatterns, int nCategories,
                           double* scalingFactors, double* cumulative, bool scalersLog, cudaStream_t stream);

// siteLogLikelihoods[p] = log(sum_{c,s} root * categoryWeights[c] * frequencies[s]) + cumulativeScaling[p] (if non-null),
// sumLogLikelihood[0] = sum_p patternWeights[p] * siteLogLikelihoods[p].
void cuda_root_log_likelihoods(const float* root, const float* categoryWeights, const float* frequencies, const float* cumulativeScaling,
                               const float* patternWeights, int nStates, int nPaddedPatterns, int nPatterns, int nCategories,
                               float* siteLogLikelihoods, float* sumLogLikelihood, cudaStream_t stream);
void cuda_root_log_likelihoods(const double* root, const double* categoryWeights, const double* frequencies, const double* cumulativeScaling,
                               const double* patternWeights, int nStates, int nPaddedPatterns, int nPatterns, int nCategories,
                               double* siteLogLikelihoods, double* sumLogLikelihood, cudaStream_t stream);

// One Taylor term on column-major (nStates, nColumns) matrices: y = alpha * A * x; F = scale * (F + y), A in CSR.
void cuda_csr_taylor_term(int nStates, int nColumns, const int* rowOffsets, const int* columns, const float* values, float alpha,
                          const float* x, float* y, float* F, float scale, cudaStream_t stream);
void cuda_csr_taylor_term(int nStates, int nColumns, const int* rowOffsets, const int* columns, const double* values, double alpha,
                          const double* x, double* y, double* F, double scale, cudaStream_t stream);
#endif
