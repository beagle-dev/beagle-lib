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

struct cuda_scratch_space
{
    void* device_ptr = nullptr;
    size_t size = 0;

    void discard();
    void fit(size_t n);

    ~cuda_scratch_space();
};

void cuda_log_vector(double* v, int length);
void cuda_log_vector(float* v, int length);

// (n,t) -> (n,t)
void cuda_sign_vector(double* v, int n, int t);
void cuda_sign_vector(float* v, int n, int t);

// sum.abs over (n,t) -> (1,t), then max over (1,t) -> (1,1).  Matrix assumed to be column-major.
double cuda_max_abs(double* v, int length);
float cuda_max_abs(float* v, int length);

void cuda_max(double* v, int length, double* out, cuda_scratch_space& scratch);
void cuda_max(float* v, int length, float* out, cuda_scratch_space& scratch);

// max over (n,t) -> (n,1).  Matrix assumed to be column-major.
void cuda_rowwise_max_abs(float* values_ptr, int n, int t, float* out_ptr);
void cuda_rowwise_max_abs(double* values_ptr, int n, int t, double* out_ptr);

// h[i] = -1 where visited[i] != 0
void cuda_mask_visited(float* h, const int* visited, int n);
void cuda_mask_visited(double* h, const int* visited, int n);

// visited[indices[j]] = 1 for j in [0,t-1]
void cuda_mark_visited(int* visited, const int* indices, int t);

void cuda_fill_vector(float* buffer, int n, double x);
void cuda_fill_vector(double* buffer, int n, double x);

void initialize_norm_x_matrix(float* data, int n, int m);
void initialize_norm_x_matrix(double* data, int n, int m);

float cuda_max_l1_norm(float* values, int n, int t, float* buffer_);
double cuda_max_l1_norm(double* values, int n, int t, double* buffer_);

void cuda_max_l1_norm(float* values, int n, int t, float* buffer, float* out, cuda_scratch_space& scratch, int* offsets);
void cuda_max_l1_norm(double* values, int n, int t, double* buffer, double* out, cuda_scratch_space& scratch, int* offsets);

void cuda_vec_fill(float* values, int length, float fill);
void cuda_vec_fill(double* values, int length, double fill);

void cuda_vec_abs(float *values, int n, float *results);
void cuda_vec_abs(double* values, int n, double *results);

void cuda_sort_indices_by_vector(const float* values_ptr, int n, int* indices_ptr);
void cuda_sort_indices_by_vector(const double* values_ptr, int n, int* indices_ptr);

void cuda_set_indices(float* x, int n, int t, const int* indices);
void cuda_set_indices(double* x, int n, int t, const int* indices);

void  rescalePartials2(bool scalers_log, int kCategoryCount, int kPaddedPatternCount, int kPaddedStateCount,
		       float* partials, float* scalingFactors, float* cumulativeScalingBuffer, int streamIndex);
void  rescalePartials2(bool scalers_log, int kCategoryCount, int kPaddedPatternCount, int kPaddedStateCount,
		       double* partials, double* scalingFactors, double* cumulativeScalingBuffer, int streamIndex);

void sumRootLikelihoods(float* siteProbs, // OUT
			float* partials, float* weights, float* frequencies, // INT
			int nStates, int nPatterns, int nCategories);

void sumRootLikelihoods(double* siteProbs, // OUT
			double* partials, double* weights, double* frequencies, // INT
			int nStates, int nPatterns, int nCategories);
#endif
