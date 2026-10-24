#include "cuda_ops.h"

#include <cub/cub.cuh>

#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/tuple.h>
#include <thrust/random.h>
#include <thrust/sort.h>

using thrust::device_ptr;
using thrust::device_pointer_cast;

template <typename T>
void cuda_log_vector(device_ptr<T> v, int length)
{
    thrust::transform(v, v + length, v, [] __device__ (T x) {return log(x);});
}

void cuda_log_vector(float* v, int length)
{
    cuda_log_vector(device_pointer_cast(v), length);
}

void cuda_log_vector(double* v, int length)
{
    cuda_log_vector(device_pointer_cast(v), length);
}


template <typename T>
void cuda_sign_vector(device_ptr<T> v, int n, int t)
{
    // In-place update is accomplished by making the output iterator the same the starting input iterator.
    thrust::transform(v, v + n*t, v, [n] __device__ (T x) -> T {return (x<0)?-1.0/n:1.0/n;});
}

void cuda_sign_vector(double* v, int n, int t)
{
    cuda_sign_vector(device_pointer_cast(v), n, t);
}

void cuda_sign_vector(float* v, int n, int t)
{
    cuda_sign_vector(device_pointer_cast(v), n, t);
}

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

// QUESTION: Why do cub::DeviceReduce and thrust::reduce need to perform allocation?
// That seems like a big problem.
template <typename T>
void cuda_max(device_ptr<T> values, int length, device_ptr<T> out, cuda_scratch_space& scratch)
{
    auto values_ptr = thrust::raw_pointer_cast(values);
    auto out_ptr = thrust::raw_pointer_cast(out);

    // 1. Get size of temporary allocation, if any.
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Max(nullptr, temp_storage_bytes, values_ptr, out_ptr, length);

    // 2. Do temporary allocation, if needed.
    scratch.fit(temp_storage_bytes);

    // 3. Do the reduction.
    cub::DeviceReduce::Max(scratch.device_ptr, temp_storage_bytes, values_ptr, out_ptr, length);
}

void cuda_max(float* values, int length, float* out, cuda_scratch_space& scratch)
{
    cuda_max(thrust::device_pointer_cast(values), length, thrust::device_pointer_cast(out), scratch);
}

void cuda_max(double* values, int length, double* out, cuda_scratch_space& scratch)
{
    cuda_max(thrust::device_pointer_cast(values), length, thrust::device_pointer_cast(out), scratch);
}

template <typename T>
T cuda_max_l1_norm(device_ptr<T> values, int n, int t, device_ptr<T> buffer)
{
    using namespace thrust::placeholders;

    // 1. First sum the absolute values in each column and place the results into buffer
    auto in_keys_start = thrust::make_transform_iterator(thrust::make_counting_iterator((int)0), (_1 / n));

    auto in_values_start = thrust::transform_iterator(thrust::device_pointer_cast(values),
                                                      [] __host__ __device__ (T x) {return std::abs(x);} );

    thrust::reduce_by_key(in_keys_start, in_keys_start + n*t,    // key indices (group by column)
                          in_values_start,                       // values to reduce (with abs applied)
                          thrust::make_discard_iterator(),       // key values out
                          buffer,                                // reduced values out
                          thrust::equal_to<int>()                // compare-keys operation
        );                                                       // summation is the default operation.

    // 2. Second maximize over the column sums and return the highest.
    return thrust::reduce(buffer, buffer+t, T(0), thrust::maximum<T>());
}

float cuda_max_l1_norm(float* values, int n, int t, float* buffer)
{
    return cuda_max_l1_norm(device_pointer_cast(values), n, t, device_pointer_cast(buffer));
}

double cuda_max_l1_norm(double* values, int n, int t, double* buffer)
{
    return cuda_max_l1_norm(device_pointer_cast(values), n, t, device_pointer_cast(buffer));
}


void cuda_scratch_space::discard()
{
    if (device_ptr)
    {
        cudaDeviceDelete(device_ptr);
        device_ptr = 0;
        size = 0;
    }
    else
    {
        assert(device_ptr == nullptr);
        assert(size == 0);
    }
}

void cuda_scratch_space::fit(size_t n)
{
    if (n > size)
    {
        std::cerr<<"cuda_scratch_space::fit( ): reallocating from "<<size<<" to "<<n<<".\n";
        discard();

        size = n;
        device_ptr = cudaMallocWrapped(size);
    }
    else
    {
        assert(device_ptr);
    }
}

cuda_scratch_space::~cuda_scratch_space()
{
    discard();
}


struct SumAbs
{
  template <typename T>
  __device__ __forceinline__ T operator()(const T& a, const T& b) const
  {
      return std::abs(a) + std::abs(b);
  }
};

template <typename T>
void cuda_max_l1_norm(device_ptr<T> values,
                      int n,
                      int t,
                      device_ptr<T> buffer,
                      device_ptr<T> out,
                      cuda_scratch_space& scratch,
                      thrust::device_ptr<int> d_offsets_it)
{
    // cudaStreamSynchronize();

    // Run reduction
    int num_segments                     = t;

    SumAbs sum_abs_op;
    T initial_value = 0;

    // Determine temporary device storage requirements
    size_t temp_storage_bytes;

    cub::DeviceSegmentedReduce::Reduce(
        nullptr,
        temp_storage_bytes,
        values,
        buffer,
        num_segments,
        d_offsets_it,
        d_offsets_it + 1,
        sum_abs_op,
        initial_value);

    scratch.fit(temp_storage_bytes);

    // Run reduction
    cub::DeviceSegmentedReduce::Reduce(
        scratch.device_ptr,
        temp_storage_bytes,
        values,
        buffer,
        num_segments,
        d_offsets_it,
        d_offsets_it + 1,
        sum_abs_op,
        initial_value);

/*
    using namespace thrust::placeholders;

    // 1. First sum the absolute values in each column and place the results into buffer
    auto in_keys_start = thrust::make_transform_iterator(thrust::make_counting_iterator((int)0), (_1 / n));

    auto in_values_start = thrust::transform_iterator(thrust::device_pointer_cast(values),
                                                      [] __host__ __device__ (T x) {return std::abs(x);} );

    thrust::reduce_by_key(in_keys_start, in_keys_start + n*t,    // key indices (group by column)
                          in_values_start,                       // values to reduce (with abs applied)
                          thrust::make_discard_iterator(),       // key values out
                          buffer,                                // reduced values out
                          thrust::equal_to<int>()                // compare-keys operation
        );                                                       // summation is the default operation.
*/
    // cudaStreamSynchronize();

    // 2. Second maximize over the column sums and return the highest.
    cuda_max(buffer, t, out, scratch);

    // cudaStreamSynchronize();
}

void cuda_max_l1_norm(float* values, int n, int t, float* buffer, float* out, cuda_scratch_space& scratch, int* offsets)
{
    using namespace thrust;

    cuda_max_l1_norm(device_pointer_cast(values), n, t, device_pointer_cast(buffer), device_pointer_cast(out), scratch, device_pointer_cast(offsets));
}

void cuda_max_l1_norm(double* values, int n, int t, double* buffer, double* out, cuda_scratch_space& scratch, int* offsets)
{
    using namespace thrust;

    cuda_max_l1_norm(device_pointer_cast(values), n, t, device_pointer_cast(buffer), device_pointer_cast(out), scratch, device_pointer_cast(offsets));
}

template <typename T>
void cuda_vec_fill(device_ptr<T> values, int length, T fill) {
	using namespace thrust::placeholders;

	thrust::fill(values, values + length, fill);
}

void cuda_vec_fill(float* values, int length, float fill) {
    cuda_vec_fill(device_pointer_cast(values), length, fill);
}

void cuda_vec_fill(double* values, int length, double fill) {
    cuda_vec_fill(device_pointer_cast(values), length, fill);
}

template <typename T>
void cuda_vec_abs(device_ptr<T> values, int n, device_ptr<T> results)
{
    thrust::transform(values, values + n, results, [] __device__ (T x) {return abs(x);});
}

void cuda_vec_abs(float* values, int n, float* results)
{
    cuda_vec_abs(device_pointer_cast(values), n, device_pointer_cast(results));
}

void cuda_vec_abs(double* values, int n, double* results)
{
    cuda_vec_abs(device_pointer_cast(values), n, device_pointer_cast(results));
}

// out[i] = max_j |values(i,j)| for a column-major (n,t) matrix.
// (A reduce_by_key with keys i % n cannot do this: equal keys are never adjacent in column-major order.)
template <typename T>
__global__ void rowwise_max_abs_kernel(const T* values, int n, int t, T* out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    T m = 0;
    for (int j = 0; j < t; j++)
        m = max(m, abs(values[i + (size_t)j * n]));
    out[i] = m;
}

template <typename T>
void cuda_rowwise_max_abs_impl(const T* values_ptr, int n, int t, T* out_ptr)
{
    const int blockSize = 256;
    rowwise_max_abs_kernel<<<(n + blockSize - 1) / blockSize, blockSize>>>(values_ptr, n, t, out_ptr);
}

void cuda_rowwise_max_abs(float* values_ptr, int n, int t, float* out_ptr)
{
    cuda_rowwise_max_abs_impl(values_ptr, n, t, out_ptr);
}

void cuda_rowwise_max_abs(double* values_ptr, int n, int t, double* out_ptr)
{
    cuda_rowwise_max_abs_impl(values_ptr, n, t, out_ptr);
}

// h[i] = -1 for rows already used as unit columns, so they sort last and are not chosen again.
template <typename T>
__global__ void mask_visited_kernel(T* h, const int* visited, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n and visited[i]) h[i] = -1;
}

// visited[indices[j]] = 1 for j in [0,t-1]
__global__ void mark_visited_kernel(int* visited, const int* indices, int t)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < t) visited[indices[j]] = 1;
}

void cuda_mask_visited(float* h, const int* visited, int n)
{
    mask_visited_kernel<<<(n + 255) / 256, 256>>>(h, visited, n);
}

void cuda_mask_visited(double* h, const int* visited, int n)
{
    mask_visited_kernel<<<(n + 255) / 256, 256>>>(h, visited, n);
}

void cuda_mark_visited(int* visited, const int* indices, int t)
{
    mark_visited_kernel<<<(t + 255) / 256, 256>>>(visited, indices, t);
}

void cuda_fill_vector(float* buffer, int n, double x)
{
    thrust::fill_n(thrust::device_pointer_cast(buffer), n, x);
}

void cuda_fill_vector(double* buffer, int n, double x)
{
    thrust::fill_n(thrust::device_pointer_cast(buffer), n, x);
}

void cuda_sort_indices_by_vector(const float* values_ptr, int n, int* indices_ptr)
{
    auto indices = thrust::device_pointer_cast(indices_ptr);
    auto values = thrust::device_pointer_cast(values_ptr);

    // 1. Initialize the indices to [0, n-1]
    thrust::counting_iterator<int> iter(0);
    thrust::copy(iter, iter + n, indices);

    // 2. Sort the indices by values
    thrust::sort(indices, indices+n, [values] __host__ __device__ (int idx1, int idx2) {return values[idx1] > values[idx2];});
}


void cuda_sort_indices_by_vector(const double* values_ptr, int n, int* indices_ptr)
{
    auto indices = thrust::device_pointer_cast(indices_ptr);
    auto values = thrust::device_pointer_cast(values_ptr);

    // 1. Initialize the indices to [0, n-1]
    thrust::counting_iterator<int> iter(0);
    thrust::copy(iter, iter + n, indices);

    // 2. Sort the indices by values
    thrust::sort(indices, indices+n, [values] __host__ __device__ (int idx1, int idx2) {return values[idx1] > values[idx2];});
}

void cuda_set_indices(float* x_ptr, int n, int t, const int* indices_ptr)
{
    // 1. Initialize the indices to [0, n-1]
    thrust::counting_iterator<int> iter(0);
    auto x = thrust::device_pointer_cast(x_ptr);
    auto indices = thrust::device_pointer_cast(indices_ptr);

    thrust::for_each(iter, iter+t, [=] __host__ __device__ (int i) { x[n*i + indices[i]] = 1.0; });
}

// Set X(i,indices[i]) = 1 for i in [0,t-1]
void cuda_set_indices(double* x_ptr, int n, int t, const int* indices_ptr)
{
    // 1. Initialize the indices to [0, n-1]
    thrust::counting_iterator<int> iter(0);
    auto x = thrust::device_pointer_cast(x_ptr);
    auto indices = thrust::device_pointer_cast(indices_ptr);

    thrust::for_each(iter, iter+t, [=] __host__ __device__ (int i) { x[n*i + indices[i]] = 1.0; });
}

template <typename T>
void justMaximize(device_ptr<T> partials, device_ptr<T> scalingFactors,
		  int nStates, int nPatterns, int nCategories)
{
    using namespace thrust::placeholders;

    // 1. Surprisingly, cuBLAS has no operations that reduce (sum,maximize,minimize,etc.) rows or columns.
    //    It can only reduce an entire dense matrix to a single value.

    // 2. thrust::reduce_by_key is able to reduce regions of a vector down to MULTIPLE values.
    //    We assign each value a "key" that decides which group it is in.
    //    Adjacent values with the same key end up in the same group.
    //    However, non-adjacent values with the same key end up in different groups.
    //    thrust::reduce_by_key performs the reduction operation on each group.

    // 3. Thrust can maximize over states (adjacent memory_ and categories (non-adjacent) in one operation
    //    by changing the order in which entries of the input matrix are visited.  All the entries in the
    //    same group need to be visited sequentially.  We can do this using a permutation iterator.

    // Implementation notes:
    // * the library MatX (https://github.com/NVIDIA/MatX) offers much nicer syntax, but thrust is older
    //   and more stable (in 2024)

    // The size of a partials buffer
    size_t partials_size = nStates * nPatterns * nCategories;


    // OK, so we can convert the input index i=[0...partials_size] to (p,c,s) as follows:
    //   s = i % nStates
    //   c = (i / nStates) % nCategories
    //   p = (i / nStates * nCategories)

    // The linear index into the input matrix is:
    //   j = s + nStates *p + (nStates*nPatterns)*c

    // A list of 00000..111111..222222........(P-1)(P-1)(P-1)(P-1) that groups values by state and categories.
    // There should be nPatterns groups.
    auto in_keys_start = thrust::make_transform_iterator(thrust::make_counting_iterator((int) 0), (_1 / (nStates * nCategories)));

    // We need to compute j as a function of i:
    //   j(i) = (i % nStates) + nStates*(i/(nStates * nCategories)) + (nStates*nPatterns)*((i/nStates) % nCategories)
    auto in_values_start = thrust::make_permutation_iterator(
        partials,
	thrust::make_transform_iterator( thrust::make_counting_iterator((int)0),
					 (_1 % nStates) + nStates*(_1/(nStates*nCategories)) + nStates*nPatterns*((_1/nStates)%nCategories))
	);

    thrust::reduce_by_key(
	// add execution policy thrust::cuda::par_nosync?
	in_keys_start,                                 // key indices start (group by pattern)
	in_keys_start + partials_size,                 // key indices end
	in_values_start,                               // values to reduce (category, pattern, state)
	thrust::make_discard_iterator(),               // key values out
	scalingFactors,                                // reduced values out
	thrust::equal_to<int>(),                       // compare keys operation
	thrust::maximum<T>()                           // reduction operation
    );
}

void justMaximize(float* partials, float* scalingFactors,
		  int nStates, int nPatterns, int nCategories)
{
    justMaximize(device_pointer_cast(partials),
                 device_pointer_cast(scalingFactors),
                 nStates, nPatterns, nCategories);
}

void justMaximize(double* partials, double* scalingFactors,
		  int nStates, int nPatterns, int nCategories)
{
    justMaximize(device_pointer_cast(partials),
                 device_pointer_cast(scalingFactors),
                 nStates, nPatterns, nCategories);
}

template <typename T>
struct Prod3
{
    __host__ __device__ T operator()(thrust::tuple<T,T,T> t)
    {
	return thrust::get<0>(t) * thrust::get<1>(t) * thrust::get<2>(t);
    }
};

template <typename T>
void sumRootLikelihoods(device_ptr<T> siteProbs, // OUT
			device_ptr<T> likelihoods, device_ptr<T> weights, device_ptr<T> frequencies, // IN
			int nStates, int nPatterns, int nCategories)
{
    using namespace thrust::placeholders;

    // The size of a partials buffer
    size_t partials_size = nStates * nPatterns * nCategories;


    // OK, so we can convert the input index i=[0...partials_size] to (p,c,s) as follows:
    //   s = i % nStates
    //   c = (i / nStates) % nCategories
    //   p = (i / nStates * nCategories)

    // 1. A list of 00000..111111..222222........(P-1)(P-1)(P-1)(P-1) that groups values by state and categories.
    // There should be nPatterns groups.
    auto in_keys_start = thrust::make_transform_iterator(thrust::make_counting_iterator((int) 0), (_1 / (nStates * nCategories)));

    // 2. The linear index into the likelihood matrix is:
    //   j = s + nStates *p + (nStates*nPatterns)*c
    //
    // We need to compute j as a function of i:
    //   j(i) = (i % nStates) + nStates*(i/(nStates * nCategories)) + (nStates*nPatterns)*((i/nStates) % nCategories)
    auto in_lks_start = thrust::make_permutation_iterator(
        likelihoods,
	thrust::make_transform_iterator( thrust::make_counting_iterator((int)0),
					 (_1 % nStates) + nStates*(_1/(nStates*nCategories)) + nStates*nPatterns*((_1/nStates)%nCategories))
	);

    // 3. The linear index into the weights matrix is just:
    //   k = c
    // Therefore
    //   k(i) = (i / nStates) % nCategories

    auto in_weights_start =  thrust::make_permutation_iterator(
        weights,
	thrust::make_transform_iterator( thrust::make_counting_iterator((int)0),
					 (_1/nStates)%nCategories)
	);

    // 4. The linear index into the states matrix is:
    //   l = s
    // Therefore
    //   l(i) = (i % nStates)

    auto in_frequencies_start =  thrust::make_permutation_iterator(
        frequencies,
	thrust::make_transform_iterator( thrust::make_counting_iterator((int)0),
					 _1 % nStates)
	);


    // 5. Make the sequence of likelihood * categoryWeight * stateFrequency
    auto tuples = thrust::make_zip_iterator(thrust::make_tuple(in_lks_start, in_weights_start, in_frequencies_start));

    auto in_values_start = thrust::make_transform_iterator(tuples, Prod3<T>());

    // 6. Sum over state and category for each pattern.
    thrust::reduce_by_key(
	// add execution policy thrust::cuda::par_nosync?
	in_keys_start,                                  // key indices start (group by pattern)
	in_keys_start + partials_size,                  // key indices end
	in_values_start,                                // values to reduce (category, pattern, state)
	thrust::make_discard_iterator(),                // key values out
	siteProbs,                                      // reduced values out
	thrust::equal_to<int>()                         // compare keys operation
	                                                // Default operation is (+)
    );
}


void sumRootLikelihoods(float* siteProbs, // OUT
			float* likelihoods, float* weights, float* frequencies, // INT
			int nStates, int nPatterns, int nCategories)
{
    sumRootLikelihoods(device_pointer_cast(siteProbs),
                       device_pointer_cast(likelihoods),
                       device_pointer_cast(weights),
                       device_pointer_cast(frequencies),
                       nStates, nPatterns, nCategories);
}


void sumRootLikelihoods(double* siteProbs, // OUT
			double* likelihoods, double* weights, double* frequencies, // INT
			int nStates, int nPatterns, int nCategories)
{
    sumRootLikelihoods(device_pointer_cast(siteProbs),
                       device_pointer_cast(likelihoods),
                       device_pointer_cast(weights),
                       device_pointer_cast(frequencies),
                       nStates, nPatterns, nCategories);
}

template <typename T>
void  rescalePartials2(bool scalers_log, int kCategoryCount, int kPaddedPatternCount, int kPaddedStateCount,
                       device_ptr<T> partials, device_ptr<T> scalingFactors, device_ptr<T> cumulativeScalingBuffer, int streamIndex)
{
    using namespace thrust::placeholders;

    size_t partialsSize = kCategoryCount * kPaddedPatternCount * kPaddedStateCount;

    // 1. Find maximize partial likelihood -> scalingFactors[pattern]
    justMaximize(partials, scalingFactors, kPaddedStateCount, kPaddedPatternCount, kCategoryCount);

    // 2. Transform scalingfactors[pattern] -> 1 if it equals 0.
    thrust::transform(scalingFactors, scalingFactors + kPaddedPatternCount, scalingFactors, [] __device__ (T x) { return (x == 0) ? 1.0 : x;});

    // 3. Rescale each pattern by scalingFactors[pattern(index)]

    // pattern =  (i/kPaddedStateCount) % kPaddedPatternCount

    // iter_max computes the scaling factor as a function of the index into the partials buffer.
    auto iter_max = thrust::make_permutation_iterator(
	                scalingFactors,
		        thrust::make_transform_iterator(
			    thrust::make_counting_iterator<int>(0),
			    (_1/kPaddedStateCount) % kPaddedPatternCount
			)
	            );


    thrust::transform(partials, partials + partialsSize, // in1 = partials[i]
		      iter_max,                          // in2 = scalingFactors[pattern(i)]
		      partials,                          // out
		      thrust::divides<T>()               // operation
	             );

//    std::cerr<<"scalers_log = "<<scalers_log<<"   cumulativeScalingBuffer = "<<cumulativeScalingBuffer<<"\n";

    // 4. Transform by log if (scalers_log)
    if (scalers_log)
        thrust::transform(scalingFactors, scalingFactors + kPaddedPatternCount, // in
                          scalingFactors,                                       // out
                          [] __device__ (T x) { return log(x); }               // transformation
                         );

    // 5. Add to cumulativeScalingBuffer
    if (cumulativeScalingBuffer)
    {
        if (scalers_log)
        {
            thrust::transform(cumulativeScalingBuffer, cumulativeScalingBuffer + kPaddedPatternCount, // in1
                              scalingFactors,                                                         // in2
                              cumulativeScalingBuffer,                                                // out
                              thrust::plus<T>()                                                       // operation;
                             );
        }
        else
        {
            auto logScalingFactors2 = thrust::make_transform_iterator( scalingFactors,
                                                                       [] __host__ __device__ (T x) { return log(x); });

            thrust::transform(cumulativeScalingBuffer, cumulativeScalingBuffer + kPaddedPatternCount, // in1
                              logScalingFactors2,                                                     // in2
                              cumulativeScalingBuffer,                                                // out
                              thrust::plus<T>()                                                       // operation;
                             );
        }
    }
}

void rescalePartials2(bool scalers_log, int kCategoryCount, int kPaddedPatternCount, int kPaddedStateCount,
                      float* partials, float* scalingFactors, float* cumulativeScalingBuffer, int streamIndex)
{
    rescalePartials2(scalers_log, kCategoryCount, kPaddedPatternCount, kPaddedStateCount,
                     device_pointer_cast(partials), device_pointer_cast(scalingFactors), device_pointer_cast(cumulativeScalingBuffer),
                     streamIndex);
}


void rescalePartials2(bool scalers_log, int kCategoryCount, int kPaddedPatternCount, int kPaddedStateCount,
                      double* partials, double* scalingFactors, double* cumulativeScalingBuffer, int streamIndex)
{
    rescalePartials2(scalers_log, kCategoryCount, kPaddedPatternCount, kPaddedStateCount,
                     device_pointer_cast(partials), device_pointer_cast(scalingFactors), device_pointer_cast(cumulativeScalingBuffer),
                     streamIndex);
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
