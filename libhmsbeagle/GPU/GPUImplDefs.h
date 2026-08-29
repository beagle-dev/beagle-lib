/*
 *
 * Copyright 2009 Phylogenetic Likelihood Working Group
 *
 * This file is part of BEAGLE.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 *
 * @author Marc Suchard
 * @author Daniel Ayres
 */

#ifndef __GPUImplDefs__
#define __GPUImplDefs__

#ifndef OPENCL_KERNEL_BUILD
//     #ifdef HAVE_CONFIG_H
//     #include "libhmsbeagle/config.h"
//     #endif
    #include "libhmsbeagle/platform.h"

    #include <cfloat>
#elif !defined(M_LN2)
    #define M_LN2   0.693147180559945309417232121458176568  /* log_e 2 */
#endif // OPENCL_KERNEL_BUILD

//#define FW_OPENCL_BINARY
//#define FW_OPENCL_TESTING
//#define FW_OPENCL_PROFILING

// #define BEAGLE_DEBUG_FLOW
// #define BEAGLE_DEBUG_VALUES
// #define BEAGLE_DEBUG_SYNCH
// #define BEAGLE_DEBUG_OPENCL_CORES

//#define BEAGLE_MEMORY_PINNED
//#define BEAGLE_FILL_4_STATE_SCALAR_SS
//#define BEAGLE_FILL_4_STATE_SCALAR_SP

// define platform/device specific implementations
enum BeagleDeviceImplementationCodes {
    BEAGLE_OPENCL_DEVICE_GENERIC         = 0,
    BEAGLE_OPENCL_DEVICE_INTEL_CPU       = 1,
    BEAGLE_OPENCL_DEVICE_INTEL_GPU       = 2,
    BEAGLE_OPENCL_DEVICE_INTEL_MIC       = 3,
    BEAGLE_OPENCL_DEVICE_AMD_CPU         = 4,
    BEAGLE_OPENCL_DEVICE_AMD_GPU         = 5,
    BEAGLE_OPENCL_DEVICE_APPLE_CPU       = 6,
    BEAGLE_OPENCL_DEVICE_APPLE_AMD_GPU   = 7,
    BEAGLE_OPENCL_DEVICE_APPLE_INTEL_GPU = 8,
    BEAGLE_OPENCL_DEVICE_APPLE_APPLE_GPU = 9,
    BEAGLE_OPENCL_DEVICE_NVIDA_GPU       = 10,
    BEAGLE_CUDA_DEVICE_NVIDIA_GPU        = 11,
    BEAGLE_TINYGPU_DEVICE_NVIDIA_GPU     = 12,
    BEAGLE_TINYGPU_DEVICE_AMD_GPU        = 13,
};

#define BEAGLE_CACHED_MATRICES_COUNT 3 // max number of matrices that can be cached for a single memcpy to device operation

/* Definition of REAL can be switched between 'double' and 'float' */
#ifdef DOUBLE_PRECISION
    #define REAL    double
    #define REAL_MIN    DBL_MIN
    #define REAL_MAX    DBL_MAX
    #define SCALING_FACTOR_COUNT 2046 // -1022, 1023
    #define SCALING_FACTOR_OFFSET 1022 // the zero point
    #define SCALING_EXPONENT_THRESHOLD 200 // TODO: find optimal value for SCALING_EXPONENT_THRESHOLD
    #define SCALING_THRESHOLD_LOWER 6.22301528e-61 // TODO: find optimal value for SCALING_THRESHOLD
    #define SCALING_THRESHOLD_UPPER 1.60693804e60 // TODO: find optimal value for SCALING_THRESHOLD
#else
    #define REAL    float
    #define REAL_MIN    FLT_MIN
    #define REAL_MAX    FLT_MAX
    #define SCALING_FACTOR_COUNT 254 // -126, 127
    #define SCALING_FACTOR_OFFSET 126 // the zero point
    #define SCALING_EXPONENT_THRESHOLD 20 // TODO: find optimal value for SCALING_EXPONENT_THRESHOLD
    #define SCALING_THRESHOLD_LOWER 9.53674316e-7 // TODO: find optimal value for SCALING_THRESHOLD
    #define SCALING_THRESHOLD_UPPER 1048576 // TODO: find optimal value for SCALING_THRESHOLD
#endif

#define SIZE_REAL   sizeof(REAL)
#define INT         int
#define SIZE_INT    sizeof(INT)

/* Define keywords for parallel frameworks */
#ifdef CUDA
    #define BEAGLE_STREAM_COUNT 1024 // max stream count
    #define BEAGLE_MULTI_GRID_MAX  3126 // use multi-grid for fewer than this many sites
    #define KW_GLOBAL_KERNEL __global__
    #define KW_DEVICE_FUNC   __device__
    #define KW_GLOBAL_VAR
    #define KW_LOCAL_MEM     __shared__
    #define KW_LOCAL_FENCE   __syncthreads()
    #define KW_LOCAL_ID_0    threadIdx.x
    #define KW_LOCAL_ID_1    threadIdx.y
    #define KW_LOCAL_ID_2    threadIdx.z
    #define KW_LOCAL_SIZE_0  blockDim.x
    #define KW_LOCAL_SIZE_1  blockDim.y
    #define KW_LOCAL_SIZE_2  blockDim.z
    #define KW_GROUP_ID_0    blockIdx.x
    #define KW_GROUP_ID_1    blockIdx.y
    #define KW_GROUP_ID_2    blockIdx.z
    #define KW_NUM_GROUPS_0  gridDim.x
    #define KW_NUM_GROUPS_1  gridDim.y
    #define KW_NUM_GROUPS_2  gridDim.z
    #define KW_RESTRICT      __restrict__
#elif defined(FW_TINYGPU)
    // TinyGPU backend: single-stream, NVIDIA PTX execution model.
    // BEAGLE_STREAM_COUNT and BEAGLE_MULTI_GRID_MAX match CUDA defaults.
    #define BEAGLE_STREAM_COUNT    1
    #define BEAGLE_MULTI_GRID_MAX  3126
    #define KW_GLOBAL_KERNEL __global__
    #define KW_DEVICE_FUNC   __device__
    #define KW_GLOBAL_VAR
    #define KW_LOCAL_MEM     __shared__
    #define KW_LOCAL_FENCE   __syncthreads()
    #define KW_LOCAL_ID_0    threadIdx.x
    #define KW_LOCAL_ID_1    threadIdx.y
    #define KW_LOCAL_ID_2    threadIdx.z
    #define KW_LOCAL_SIZE_0  blockDim.x
    #define KW_LOCAL_SIZE_1  blockDim.y
    #define KW_LOCAL_SIZE_2  blockDim.z
    #define KW_GROUP_ID_0    blockIdx.x
    #define KW_GROUP_ID_1    blockIdx.y
    #define KW_GROUP_ID_2    blockIdx.z
    #define KW_NUM_GROUPS_0  gridDim.x
    #define KW_NUM_GROUPS_1  gridDim.y
    #define KW_NUM_GROUPS_2  gridDim.z
    #define KW_RESTRICT      __restrict__
#elif defined(FW_TINYGPU_HYBRID_AMD)
    // TinyGPU-Hybrid backend, AMD path: kernels compiled via comgr's HIP
    // language, not OpenCL. OpenCL's get_global_id() has a global_work_offset
    // concept HIP has no equivalent of, so its compiled implementation reads
    // the full HSA dispatch packet (dispatch_ptr/queue_ptr/dispatch_id sgprs,
    // heavy scratch usage) -- which this remote-socket transport doesn't
    // support cleanly (STATUS.md AMD §15-17). HIP's group/local-id builtins
    // read hardware-architected sgprs directly and need none of that --
    // the same, simpler path tinygrad's own (proven-working) dispatch always
    // uses. __ockl_get_* are real functions from comgr's bundled OCKL
    // device-library bitcode, forward-declared here since compilation uses
    // -nogpuinc (no HIP runtime headers, matching tinygrad's own compile_hip).
    #define BEAGLE_STREAM_COUNT    1
    #define BEAGLE_MULTI_GRID_MAX  3126
    extern "C" __attribute__((device)) unsigned long __ockl_get_local_id(unsigned int);
    extern "C" __attribute__((device)) unsigned long __ockl_get_group_id(unsigned int);
    extern "C" __attribute__((device)) unsigned long __ockl_get_local_size(unsigned int);
    extern "C" __attribute__((device)) unsigned long __ockl_get_num_groups(unsigned int);
    #define KW_GLOBAL_KERNEL extern "C" __attribute__((global))
    #define KW_DEVICE_FUNC   __attribute__((device))
    #define KW_GLOBAL_VAR
    #define KW_LOCAL_MEM     __attribute__((shared))
    #define KW_LOCAL_FENCE   __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup"); __builtin_amdgcn_s_barrier(); \
                             __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup")
    #define KW_LOCAL_ID_0    ((int)__ockl_get_local_id(0))
    #define KW_LOCAL_ID_1    ((int)__ockl_get_local_id(1))
    #define KW_LOCAL_ID_2    ((int)__ockl_get_local_id(2))
    #define KW_LOCAL_SIZE_0  ((int)__ockl_get_local_size(0))
    #define KW_LOCAL_SIZE_1  ((int)__ockl_get_local_size(1))
    #define KW_LOCAL_SIZE_2  ((int)__ockl_get_local_size(2))
    #define KW_GROUP_ID_0    ((int)__ockl_get_group_id(0))
    #define KW_GROUP_ID_1    ((int)__ockl_get_group_id(1))
    #define KW_GROUP_ID_2    ((int)__ockl_get_group_id(2))
    #define KW_NUM_GROUPS_0  ((int)__ockl_get_num_groups(0))
    #define KW_NUM_GROUPS_1  ((int)__ockl_get_num_groups(1))
    #define KW_NUM_GROUPS_2  ((int)__ockl_get_num_groups(2))
    #define KW_RESTRICT      __restrict__
    // Defensive OpenCL-builtin compatibility: the only raw (non-KW_*)
    // OpenCL builtin call anywhere in kernels/*.cu is get_global_id(1)
    // inside kernelsX.cu's DETERMINE_INDICES_X_CPU macro, an OpenCL-CPU-
    // driver-only code path never invoked for a GPU resource -- provided
    // anyway since it costs nothing.
    static inline __attribute__((device)) unsigned long get_global_id(unsigned int d) {
        return __ockl_get_group_id(d) * __ockl_get_local_size(d) + __ockl_get_local_id(d);
    }
    static inline __attribute__((device)) unsigned long get_local_id(unsigned int d) { return __ockl_get_local_id(d); }
    static inline __attribute__((device)) unsigned long get_group_id(unsigned int d) { return __ockl_get_group_id(d); }
    static inline __attribute__((device)) unsigned long get_local_size(unsigned int d) { return __ockl_get_local_size(d); }
    static inline __attribute__((device)) unsigned long get_num_groups(unsigned int d) { return __ockl_get_num_groups(d); }
    // Math functions kernels/*.cu calls as plain exp/cos/sin/log/pow/round
    // (the complete set used anywhere in the kernel sources -- checked).
    // OpenCL-C and CUDA both provide these as language builtins; HIP under
    // -nogpuinc has neither, so route to comgr's bundled OCML device-library
    // functions directly (the same functions HIPRenderer's own _ocml()
    // helper targets for tinygrad's generated kernels).
    extern "C" __attribute__((device)) float  __ocml_exp_f32(float);
    extern "C" __attribute__((device)) double __ocml_exp_f64(double);
    extern "C" __attribute__((device)) float  __ocml_cos_f32(float);
    extern "C" __attribute__((device)) double __ocml_cos_f64(double);
    extern "C" __attribute__((device)) float  __ocml_sin_f32(float);
    extern "C" __attribute__((device)) double __ocml_sin_f64(double);
    extern "C" __attribute__((device)) float  __ocml_log_f32(float);
    extern "C" __attribute__((device)) double __ocml_log_f64(double);
    extern "C" __attribute__((device)) float  __ocml_pow_f32(float, float);
    extern "C" __attribute__((device)) double __ocml_pow_f64(double, double);
    extern "C" __attribute__((device)) float  __ocml_round_f32(float);
    extern "C" __attribute__((device)) double __ocml_round_f64(double);
    static inline __attribute__((device)) float  exp(float x) { return __ocml_exp_f32(x); }
    static inline __attribute__((device)) double exp(double x) { return __ocml_exp_f64(x); }
    static inline __attribute__((device)) float  cos(float x) { return __ocml_cos_f32(x); }
    static inline __attribute__((device)) double cos(double x) { return __ocml_cos_f64(x); }
    static inline __attribute__((device)) float  sin(float x) { return __ocml_sin_f32(x); }
    static inline __attribute__((device)) double sin(double x) { return __ocml_sin_f64(x); }
    static inline __attribute__((device)) float  log(float x) { return __ocml_log_f32(x); }
    static inline __attribute__((device)) double log(double x) { return __ocml_log_f64(x); }
    static inline __attribute__((device)) float  pow(float x, float y) { return __ocml_pow_f32(x, y); }
    static inline __attribute__((device)) double pow(double x, double y) { return __ocml_pow_f64(x, y); }
    static inline __attribute__((device)) float  round(float x) { return __ocml_round_f32(x); }
    static inline __attribute__((device)) double round(double x) { return __ocml_round_f64(x); }
    static inline __attribute__((device)) float  fma(float x, float y, float z) { return __builtin_fmaf(x, y, z); }
    static inline __attribute__((device)) double fma(double x, double y, double z) { return __builtin_fma(x, y, z); }
    static inline __attribute__((device)) int    abs(int x) { return x < 0 ? -x : x; }
    static inline __attribute__((device)) float  frexp(float x, int* e) { return __builtin_frexpf(x, e); }
    static inline __attribute__((device)) double frexp(double x, int* e) { return __builtin_frexp(x, e); }
    static inline __attribute__((device)) float  ldexp(float x, int e) { return __builtin_ldexpf(x, e); }
    static inline __attribute__((device)) double ldexp(double x, int e) { return __builtin_ldexp(x, e); }
    // Bit-reinterpretation and atomicAdd: real CUDA/HIP builtins normally
    // declared in device_functions.h/hip_runtime.h, unavailable under
    // -nogpuinc. atomicAdd(REAL*, REAL) (kernels4.cu/kernelsX.cu) is called
    // unconditionally, not just inside kernelsAll.cu's own CUDA-gated
    // double-precision CAS-loop fallback (that block never fires here,
    // guarded on `defined CUDA`), so both float and double overloads are
    // provided directly via the same CAS-loop technique that fallback
    // itself demonstrates, using __sync_val_compare_and_swap (a portable
    // GCC/clang legacy atomic builtin) instead of atomicCAS.
    static inline __attribute__((device)) long long __double_as_longlong(double x) { long long r; __builtin_memcpy(&r, &x, sizeof(r)); return r; }
    static inline __attribute__((device)) double __longlong_as_double(long long x) { double r; __builtin_memcpy(&r, &x, sizeof(r)); return r; }
    static inline __attribute__((device)) unsigned int __beagle_float_as_uint(float x) { unsigned int r; __builtin_memcpy(&r, &x, sizeof(r)); return r; }
    static inline __attribute__((device)) float __beagle_uint_as_float(unsigned int x) { float r; __builtin_memcpy(&r, &x, sizeof(r)); return r; }
    static inline __attribute__((device)) float atomicAdd(float* address, float val) {
        unsigned int* address_as_uint = (unsigned int*)address;
        unsigned int old = *address_as_uint, assumed;
        do {
            assumed = old;
            old = __sync_val_compare_and_swap(address_as_uint, assumed,
                      __beagle_float_as_uint(val + __beagle_uint_as_float(assumed)));
        } while (assumed != old);
        return __beagle_uint_as_float(old);
    }
    static inline __attribute__((device)) double atomicAdd(double* address, double val) {
        unsigned long long* address_as_ull = (unsigned long long*)address;
        unsigned long long old = *address_as_ull, assumed;
        do {
            assumed = old;
            old = __sync_val_compare_and_swap(address_as_ull, assumed,
                      (unsigned long long)__double_as_longlong(val + __longlong_as_double((long long)assumed)));
        } while (assumed != old);
        return __longlong_as_double((long long)old);
    }
#elif defined(FW_OPENCL)
    #define BEAGLE_STREAM_COUNT 1 // disabled for now, also has to be smaller for OpenCL to not run out of host memory
    #define BEAGLE_MULTI_GRID_MAX  16384 // use multi-grid for fewer than this many sites
    #define KW_GLOBAL_KERNEL __kernel
    #define KW_DEVICE_FUNC
    #define KW_GLOBAL_VAR    __global
    #define KW_LOCAL_MEM     __local
    #define KW_LOCAL_FENCE   barrier(CLK_LOCAL_MEM_FENCE)
    #define KW_LOCAL_ID_0    get_local_id(0)
    #define KW_LOCAL_ID_1    get_local_id(1)
    #define KW_LOCAL_ID_2    get_local_id(2)
    #define KW_LOCAL_SIZE_0  get_local_size(0)
    #define KW_LOCAL_SIZE_1  get_local_size(1)
    #define KW_LOCAL_SIZE_2  get_local_size(2)
    #define KW_GROUP_ID_0    get_group_id(0)
    #define KW_GROUP_ID_1    get_group_id(1)
    #define KW_GROUP_ID_2    get_group_id(2)
    #define KW_NUM_GROUPS_0  get_num_groups(0)
    #define KW_NUM_GROUPS_1  get_num_groups(1)
    #define KW_NUM_GROUPS_2  get_num_groups(2)
    #define KW_RESTRICT      restrict
#endif

// TinyGPU kernel PTX is compiled with -DCUDA -DFW_TINYGPU together (see
// kernels/make_tinygpu_kernels.sh) so every #ifdef CUDA block above behaves
// identically to the real CUDA backend -- this override is deliberately
// separate from that chain, not a branch of it, so it applies regardless
// of which branch above fired. __restrict__ lets newer nvcc (this backend
// uses 12.8; the CUDA backend's checked-in kernel header was last compiled
// by 10.2) promote reads through KW_RESTRICT-qualified pointers to
// ld.global.nc (the read-only/texture-cache path) more aggressively than
// the older compiler did. That path faults deterministically on this
// backend's from-scratch driver -- see STATUS.md/TODO.md on the usb
// branch. Dropping __restrict__ here removes nvcc's non-aliasing
// justification for that promotion; every other backend is unaffected.
#if defined(FW_TINYGPU)
    #undef KW_RESTRICT
    #define KW_RESTRICT
#endif

/* Compiler definitions
 *
 * PADDED_STATE_COUNT - # of total states after augmentation
 *                      *should* be a multiple of 16
 *
 * PATTERN_BLOCK_SIZE - # of patterns to pack onto each thread-block in pruning
 *                          ( x 4 for PADDED_STATE_COUNT==4)
 *                      PATTERN_BLOCK_SIZE * PADDED_STATE_COUNT <= 512
 *
 * MATRIX_BLOCK_SIZE  - # of matrices to pack onto each thread-block in integrating
 *                        likelihood and store in dynamic weighting;
 *                      MATRIX_BLOCK_SIZE * PADDED_STATE_COUNT <= 512
 *                    - TODO: Currently matrixCount must be < MATRIX_BLOCK_SIZE, fix!
 *
 * BLOCK_PEELING_SIZE - # of the states to pre-fetch in inner-sum in pruning;
 *                      BLOCK_PEELING_SIZE <= PATTERN_BLOCK_SIZE and
 *                      *must* be a divisor of PADDED_STATE_COUNT
 *
 * IS_POWER_OF_TWO    - 1 if PADDED_STATE_COUNT = 2^{N} for some integer N, otherwise 0
 *
 * SMALLEST_POWER_OF_TWO - Smallest power of 2 greater than or equal to PADDED_STATE_COUNT
 *                         (if not already a power of 2)
 *
 * SLOW_REWEIGHING    - 1 if requires the slow reweighing algorithm, otherwise 0
 *
 */

/* Table of pre-optimized compiler definitions
 */


// SINGLE PRECISION definitions

// PADDED_STATE_COUNT == 4
#define PATTERN_BLOCK_SIZE_SP_4          16
#define PATTERN_BLOCK_SIZE_SP_4_CPU      256
#define PATTERN_BLOCK_SIZE_SP_4_APPLECPU 128
#define MATRIX_BLOCK_SIZE_SP_4           8
#define BLOCK_PEELING_SIZE_SP_4          8
#define IS_POWER_OF_TWO_SP_4             1
#define SMALLEST_POWER_OF_TWO_SP_4       4
#define SLOW_REWEIGHING_SP_4             0

// PADDED_STATE_COUNT == 16
// TODO: find optimal settings
#define PATTERN_BLOCK_SIZE_SP_16         8
#define MATRIX_BLOCK_SIZE_SP_16          8
#define BLOCK_PEELING_SIZE_SP_16         8
#define IS_POWER_OF_TWO_SP_16            1
#define SMALLEST_POWER_OF_TWO_SP_16      16
#define SLOW_REWEIGHING_SP_16            0

// PADDED_STATE_COUNT == 32
// TODO: find optimal settings
#define PATTERN_BLOCK_SIZE_SP_32         8
#define MATRIX_BLOCK_SIZE_SP_32          8
#define BLOCK_PEELING_SIZE_SP_32         8
#define IS_POWER_OF_TWO_SP_32            1
#define SMALLEST_POWER_OF_TWO_SP_32      32
#define SLOW_REWEIGHING_SP_32            0

// PADDED_STATE_COUNT == 48
#define PATTERN_BLOCK_SIZE_SP_48         8
#define PATTERN_BLOCK_SIZE_SP_48_AMDGPU  4
#define MATRIX_BLOCK_SIZE_SP_48          8
#define MATRIX_BLOCK_SIZE_SP_48_AMDGPU   4
#define BLOCK_PEELING_SIZE_SP_48         8
#define BLOCK_PEELING_SIZE_SP_48_AMDGPU  4
#define IS_POWER_OF_TWO_SP_48            0
#define SMALLEST_POWER_OF_TWO_SP_48      64
#define SLOW_REWEIGHING_SP_48            0

// PADDED_STATE_COUNT == 64
#define PATTERN_BLOCK_SIZE_SP_64         8
#define PATTERN_BLOCK_SIZE_SP_64_AMDGPU  4
#define MATRIX_BLOCK_SIZE_SP_64          8
#define MATRIX_BLOCK_SIZE_SP_64_AMDGPU   4
#define BLOCK_PEELING_SIZE_SP_64         8
#define BLOCK_PEELING_SIZE_SP_64_AMDGPU  4
#define IS_POWER_OF_TWO_SP_64            1
#define SMALLEST_POWER_OF_TWO_SP_64      64
#define SLOW_REWEIGHING_SP_64            0

// PADDED_STATE_COUNT == 80
#define PATTERN_BLOCK_SIZE_SP_80         8
#define PATTERN_BLOCK_SIZE_SP_80_AMDGPU  2
#define MATRIX_BLOCK_SIZE_SP_80          8
#define MATRIX_BLOCK_SIZE_SP_80_AMDGPU   2
#define BLOCK_PEELING_SIZE_SP_80         8
#define BLOCK_PEELING_SIZE_SP_80_AMDGPU  2
#define IS_POWER_OF_TWO_SP_80            0
#define SMALLEST_POWER_OF_TWO_SP_80      128
#define SLOW_REWEIGHING_SP_80            1

// PADDED_STATE_COUNT == 128
#define PATTERN_BLOCK_SIZE_SP_128        4
#define PATTERN_BLOCK_SIZE_SP_128_AMDGPU 2
#define MATRIX_BLOCK_SIZE_SP_128         8
#define MATRIX_BLOCK_SIZE_SP_128_AMDGPU  2
#define BLOCK_PEELING_SIZE_SP_128        2
#define BLOCK_PEELING_SIZE_SP_128_AMDGPU 2
#define IS_POWER_OF_TWO_SP_128           1
#define SMALLEST_POWER_OF_TWO_SP_128     128
#define SLOW_REWEIGHING_SP_128           1

// PADDED_STATE_COUNT == 192
#define PATTERN_BLOCK_SIZE_SP_192        2
#define PATTERN_BLOCK_SIZE_SP_192_AMDGPU 1
#define MATRIX_BLOCK_SIZE_SP_192         8
#define MATRIX_BLOCK_SIZE_SP_192_AMDGPU  1
#define BLOCK_PEELING_SIZE_SP_192        2
#define BLOCK_PEELING_SIZE_SP_192_AMDGPU 1
#define IS_POWER_OF_TWO_SP_192           0
#define SMALLEST_POWER_OF_TWO_SP_192     256
#define SLOW_REWEIGHING_SP_192           1

// PADDED_STATE_COUNT == 256
#define PATTERN_BLOCK_SIZE_SP_256        2
#define PATTERN_BLOCK_SIZE_SP_256_AMDGPU 1
#define MATRIX_BLOCK_SIZE_SP_256         8
#define MATRIX_BLOCK_SIZE_SP_256_AMDGPU  1
#define BLOCK_PEELING_SIZE_SP_256        2
#define BLOCK_PEELING_SIZE_SP_256_AMDGPU 1
#define IS_POWER_OF_TWO_SP_256           1
#define SMALLEST_POWER_OF_TWO_SP_256     256
#define SLOW_REWEIGHING_SP_256           1

// DOUBLE PRECISION definitions   TODO None of these have been checked

// PADDED_STATE_COUNT == 4
#define PATTERN_BLOCK_SIZE_DP_4          16
#define PATTERN_BLOCK_SIZE_DP_4_CPU      256
#define PATTERN_BLOCK_SIZE_DP_4_APPLECPU 128
#define MATRIX_BLOCK_SIZE_DP_4           8
#define BLOCK_PEELING_SIZE_DP_4          8
#define IS_POWER_OF_TWO_DP_4             1
#define SMALLEST_POWER_OF_TWO_DP_4       4
#define SLOW_REWEIGHING_DP_4             0

// PADDED_STATE_COUNT == 16
#define PATTERN_BLOCK_SIZE_DP_16         8
#define MATRIX_BLOCK_SIZE_DP_16          8
#define BLOCK_PEELING_SIZE_DP_16         8
#define IS_POWER_OF_TWO_DP_16            1
#define SMALLEST_POWER_OF_TWO_DP_16      16
#define SLOW_REWEIGHING_DP_16            0

// PADDED_STATE_COUNT == 32
#define PATTERN_BLOCK_SIZE_DP_32         8
#define MATRIX_BLOCK_SIZE_DP_32          8
#define BLOCK_PEELING_SIZE_DP_32         8
#define IS_POWER_OF_TWO_DP_32            1
#define SMALLEST_POWER_OF_TWO_DP_32      32
#define SLOW_REWEIGHING_DP_32            0

// PADDED_STATE_COUNT == 48
#define PATTERN_BLOCK_SIZE_DP_48         8
#define PATTERN_BLOCK_SIZE_DP_48_AMDGPU  4
#define MATRIX_BLOCK_SIZE_DP_48          8
#define MATRIX_BLOCK_SIZE_DP_48_AMDGPU   4
#define BLOCK_PEELING_SIZE_DP_48         8
#define BLOCK_PEELING_SIZE_DP_48_AMDGPU  4
#define IS_POWER_OF_TWO_DP_48            0
#define SMALLEST_POWER_OF_TWO_DP_48      64
#define SLOW_REWEIGHING_DP_48            0

// PADDED_STATE_COUNT == 64
#define PATTERN_BLOCK_SIZE_DP_64         8
#define PATTERN_BLOCK_SIZE_DP_64_AMDGPU  4
#define MATRIX_BLOCK_SIZE_DP_64          8
#define MATRIX_BLOCK_SIZE_DP_64_AMDGPU   4
#define BLOCK_PEELING_SIZE_DP_64         4 // Can use 8 on GTX480
#define BLOCK_PEELING_SIZE_DP_64_AMDGPU  4
#define IS_POWER_OF_TWO_DP_64            1
#define SMALLEST_POWER_OF_TWO_DP_64      64
#define SLOW_REWEIGHING_DP_64            0

// PADDED_STATE_COUNT == 80
#define PATTERN_BLOCK_SIZE_DP_80         8
#define PATTERN_BLOCK_SIZE_DP_80_AMDGPU  2
#define MATRIX_BLOCK_SIZE_DP_80          8
#define MATRIX_BLOCK_SIZE_DP_80_AMDGPU   2
#define BLOCK_PEELING_SIZE_DP_80         4 // Can use 8 on GTX480
#define BLOCK_PEELING_SIZE_DP_80_AMDGPU  2
#define IS_POWER_OF_TWO_DP_80            0
#define SMALLEST_POWER_OF_TWO_DP_80      128
#define SLOW_REWEIGHING_DP_80            1

// PADDED_STATE_COUNT == 128
#define PATTERN_BLOCK_SIZE_DP_128        4
#define PATTERN_BLOCK_SIZE_DP_128_AMDGPU 2
#define MATRIX_BLOCK_SIZE_DP_128         8
#define MATRIX_BLOCK_SIZE_DP_128_AMDGPU  2
#define BLOCK_PEELING_SIZE_DP_128        2
#define BLOCK_PEELING_SIZE_DP_128_AMDGPU 2
#define IS_POWER_OF_TWO_DP_128           1
#define SMALLEST_POWER_OF_TWO_DP_128     128
#define SLOW_REWEIGHING_DP_128           1

// PADDED_STATE_COUNT == 192
#define PATTERN_BLOCK_SIZE_DP_192        2
#define PATTERN_BLOCK_SIZE_DP_192_AMDGPU 1
#define MATRIX_BLOCK_SIZE_DP_192         8
#define MATRIX_BLOCK_SIZE_DP_192_AMDGPU  1
#define BLOCK_PEELING_SIZE_DP_192        2
#define BLOCK_PEELING_SIZE_DP_192_AMDGPU 1
#define IS_POWER_OF_TWO_DP_192           0
#define SMALLEST_POWER_OF_TWO_DP_192     256
#define SLOW_REWEIGHING_DP_192           1

// PADDED_STATE_COUNT == 256
#define PATTERN_BLOCK_SIZE_DP_256        2
#define PATTERN_BLOCK_SIZE_DP_256_AMDGPU 1
#define MATRIX_BLOCK_SIZE_DP_256         8
#define MATRIX_BLOCK_SIZE_DP_256_AMDGPU  1
#define BLOCK_PEELING_SIZE_DP_256        2
#define BLOCK_PEELING_SIZE_DP_256_AMDGPU 1
#define IS_POWER_OF_TWO_DP_256           1
#define SMALLEST_POWER_OF_TWO_DP_256     256
#define SLOW_REWEIGHING_DP_256           1

#ifdef STATE_COUNT
#if (STATE_COUNT == 4 || STATE_COUNT == 16 || STATE_COUNT == 32 || STATE_COUNT == 48 || STATE_COUNT == 64 || STATE_COUNT == 80 || STATE_COUNT == 128 || STATE_COUNT == 192 || STATE_COUNT == 256)
	#define PADDED_STATE_COUNT	STATE_COUNT
#else
	#error *** Precompiler directive state count not defined ***
#endif
#endif

// Need nested macros: first for replacement, second for evaluation
#define GET2_NO_CALL(x, y)	       x##_##y
#define	GET2_VALUE(x, y)		   GET2_NO_CALL(x, y)
#define GET3_NO_CALL(x, y, z)	   x##_##y##_##z
#define	GET3_VALUE(x, y, z)		   GET3_NO_CALL(x, y, z)
#define GET4_NO_CALL(x, y, z, w)   x##_##y##_##z##_##w
#define GET4_VALUE(x, y, z, w)     GET4_NO_CALL(x, y, z, w)

#ifdef DOUBLE_PRECISION
	#define PREC	DP
#else
	#define	PREC	SP
#endif

#if defined(FW_OPENCL_APPLECPU) && (STATE_COUNT == 4)
    #define PATTERN_BLOCK_SIZE     GET4_VALUE(PATTERN_BLOCK_SIZE, PREC, PADDED_STATE_COUNT, APPLECPU)
#elif defined(FW_OPENCL_CPU) && (STATE_COUNT == 4)
    #define PATTERN_BLOCK_SIZE     GET4_VALUE(PATTERN_BLOCK_SIZE, PREC, PADDED_STATE_COUNT, CPU)
#elif (defined(FW_OPENCL_AMDGPU) || defined(FW_OPENCL_INTELGPU)) && (STATE_COUNT > 32)
    #define PATTERN_BLOCK_SIZE     GET4_VALUE(PATTERN_BLOCK_SIZE, PREC, PADDED_STATE_COUNT, AMDGPU)
#else
    #define PATTERN_BLOCK_SIZE     GET3_VALUE(PATTERN_BLOCK_SIZE, PREC, PADDED_STATE_COUNT)
#endif

#if (defined(FW_OPENCL_AMDGPU) || defined(FW_OPENCL_INTELGPU)) && (STATE_COUNT > 32)
    #define MATRIX_BLOCK_SIZE       GET4_VALUE(MATRIX_BLOCK_SIZE, PREC, PADDED_STATE_COUNT, AMDGPU)
    #define BLOCK_PEELING_SIZE      GET4_VALUE(BLOCK_PEELING_SIZE, PREC, PADDED_STATE_COUNT, AMDGPU)
#else
    #define MATRIX_BLOCK_SIZE       GET3_VALUE(MATRIX_BLOCK_SIZE, PREC, PADDED_STATE_COUNT)
    #define BLOCK_PEELING_SIZE      GET3_VALUE(BLOCK_PEELING_SIZE, PREC, PADDED_STATE_COUNT)
#endif

#define CHECK_IS_POWER_OF_TWO	GET3_VALUE(IS_POWER_OF_TWO, PREC, PADDED_STATE_COUNT)
#if (CHECK_IS_POWER_OF_TWO == 1)
	#define IS_POWER_OF_TWO
#endif
#define SMALLEST_POWER_OF_TWO	GET3_VALUE(SMALLEST_POWER_OF_TWO, PREC, PADDED_STATE_COUNT)
#define CHECK_SLOW_REWEIGHING	GET3_VALUE(SLOW_REWEIGHING, PREC, PADDED_STATE_COUNT)
#if (CHECK_SLOW_REWEIGHING == 1)
	#define SLOW_REWEIGHING
#endif

// State count independent
#define SUM_SITES_BLOCK_SIZE_DP	128
#define SUM_SITES_BLOCK_SIZE_SP	128
#define MULTI_NODE_SUM_BLOCK_SIZE_DP    128
#define MULTI_NODE_SUM_BLOCK_SIZE_SP    128
#define MULTIPLY_BLOCK_SIZE_DP           16
#define MULTIPLY_BLOCK_SIZE_SP           16
#define MULTIPLY_BLOCK_SIZE_DP_APPLECPU  1
#define MULTIPLY_BLOCK_SIZE_SP_APPLECPU  1
#define REORDER_BLOCK_SIZE           32
#define REORDER_BLOCK_SIZE_CPU      256
#define REORDER_BLOCK_SIZE_APPLECPU 128

#define SUM_SITES_BLOCK_SIZE    GET2_VALUE(SUM_SITES_BLOCK_SIZE, PREC)
#define MULTI_NODE_SUM_BLOCK_SIZE    GET2_VALUE(MULTI_NODE_SUM_BLOCK_SIZE, PREC)
#if defined(FW_OPENCL_APPLECPU)
    #define MULTIPLY_BLOCK_SIZE     GET3_VALUE(MULTIPLY_BLOCK_SIZE, PREC, APPLECPU)
#else
    #define MULTIPLY_BLOCK_SIZE     GET2_VALUE(MULTIPLY_BLOCK_SIZE, PREC)
#endif

#define MEMCNV(to, from, length, toType)    { \
                                                int m; \
                                                for(m = 0; m < length; m++) { \
                                                    to[m] = (toType) from[m]; \
                                                } \
                                            }

typedef struct Dim3Int Dim3Int;

struct Dim3Int
{
    unsigned int x, y, z;
#if defined(__cplusplus)
    Dim3Int(unsigned int xArg = 1,
            unsigned int yArg = 1,
            unsigned int zArg = 1) : x(xArg), y(yArg), z(zArg) {}
#endif /* __cplusplus */
};

#endif // __GPUImplDefs__
