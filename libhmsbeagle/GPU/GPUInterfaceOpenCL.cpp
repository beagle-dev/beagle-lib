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
 * @author Daniel Ayres
 */

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cstdarg>
#include <cmath>
#include <map>
#include <chrono>

#include "libhmsbeagle/beagle.h"
#include "libhmsbeagle/GPU/GPUImplDefs.h"
#include "libhmsbeagle/GPU/GPUImplHelper.h"
#include "libhmsbeagle/GPU/GPUInterface.h"
#include "libhmsbeagle/GPU/KernelResource.h"

// ---------------------------------------------------------------------------
// Diagnostic-only profiling instrumentation, gated behind BEAGLE_PROFILE_SUMMARY.
// Accumulates call counts + elapsed host-side time into static counters (no
// per-call printing) and dumps one summary at process exit. A warmup window
// (first BEAGLE_PROFILE_WARMUP_SYNC SynchronizeHost calls, i.e. host<->device
// round trips) is discarded before any counter starts accumulating, so the
// summary reflects steady-state behavior rather than one-time clBuildProgram/
// initial-allocation/GPU-clock-ramp costs. Not intended to be a permanent
// feature; safe to leave disabled (default off, near-zero overhead: one
// getenv + one bool check per call) or revert entirely.
// ---------------------------------------------------------------------------
namespace {

const long BEAGLE_PROFILE_WARMUP_SYNC = 300;

struct BeagleProfileStats {
    bool registered = false;
    bool enabled = false;
    bool warm = false;
    long warmupRemaining = BEAGLE_PROFILE_WARMUP_SYNC;

    long launchKernelCalls = 0;
    double launchKernelSeconds = 0.0;
    long launchKernelConcurrentCalls = 0;
    double launchKernelConcurrentSeconds = 0.0;
    long syncHostCalls = 0;
    double syncHostSeconds = 0.0;
    long memcpyH2DCalls = 0;
    double memcpyH2DSeconds = 0.0;
    size_t memcpyH2DBytes = 0;
};

BeagleProfileStats gBeagleProfile;

void beagleProfilePrintSummary() {
    if (!gBeagleProfile.enabled) return;
    BeagleProfileStats& p = gBeagleProfile;
    double totalAccounted = p.launchKernelSeconds + p.launchKernelConcurrentSeconds +
                             p.syncHostSeconds + p.memcpyH2DSeconds;
    fprintf(stderr, "\n===== BEAGLE_PROFILE_SUMMARY (GPUInterfaceOpenCL) =====\n");
    fprintf(stderr, "warmup discarded: first %ld SynchronizeHost calls\n", BEAGLE_PROFILE_WARMUP_SYNC);
    fprintf(stderr, "LaunchKernel            : calls=%-8ld total=%.6fs avg=%.3fus\n",
            p.launchKernelCalls, p.launchKernelSeconds,
            p.launchKernelCalls ? p.launchKernelSeconds * 1e6 / p.launchKernelCalls : 0.0);
    fprintf(stderr, "LaunchKernelConcurrent  : calls=%-8ld total=%.6fs avg=%.3fus\n",
            p.launchKernelConcurrentCalls, p.launchKernelConcurrentSeconds,
            p.launchKernelConcurrentCalls ? p.launchKernelConcurrentSeconds * 1e6 / p.launchKernelConcurrentCalls : 0.0);
    fprintf(stderr, "SynchronizeHost(clFinish): calls=%-8ld total=%.6fs avg=%.3fus\n",
            p.syncHostCalls, p.syncHostSeconds,
            p.syncHostCalls ? p.syncHostSeconds * 1e6 / p.syncHostCalls : 0.0);
    fprintf(stderr, "MemcpyHostToDevice      : calls=%-8ld total=%.6fs avg=%.3fus bytes=%zu\n",
            p.memcpyH2DCalls, p.memcpyH2DSeconds,
            p.memcpyH2DCalls ? p.memcpyH2DSeconds * 1e6 / p.memcpyH2DCalls : 0.0,
            p.memcpyH2DBytes);
    fprintf(stderr, "Total host-side time accounted above (post-warmup): %.6fs\n", totalAccounted);
    fprintf(stderr, "===== END BEAGLE_PROFILE_SUMMARY =====\n\n");
    fflush(stderr);
}

inline bool beagleProfilingEnabled() {
    if (!gBeagleProfile.registered) {
        gBeagleProfile.enabled = (getenv("BEAGLE_PROFILE_SUMMARY") != NULL);
        gBeagleProfile.registered = true;
        if (gBeagleProfile.enabled) {
            atexit(beagleProfilePrintSummary);
        }
    }
    return gBeagleProfile.enabled;
}

} // anonymous namespace

#define SAFE_CL(call)   { \
                            int error = call; \
                            if(error != CL_SUCCESS) { \
                                fprintf(stderr, \
                                    "\nOpenCL error: %s from file <%s>, line %i.\n", \
                                    GetCLErrorDescription(error), __FILE__, __LINE__); \
                                exit(-1); \
                            } \
                        }

#define LOAD_KERNEL_INTO_RESOURCE(state, prec, id, impl, impl2, impl3) \
        kernelResource = new KernelResource( \
            state, \
            (char*) KERNELS_STRING_##prec##_##state, \
            PATTERN_BLOCK_SIZE_##prec##_##state##impl, \
            MATRIX_BLOCK_SIZE_##prec##_##state##impl2, \
            BLOCK_PEELING_SIZE_##prec##_##state##impl2, \
            SLOW_REWEIGHING_##prec##_##state, \
            MULTIPLY_BLOCK_SIZE_##prec##impl3, \
            0,0,0,0);

namespace opencl_device {

GPUInterface::GPUInterface() {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::GPUInterface\n");
#endif

    kernelResource = NULL;

    openClDeviceId = NULL;
    openClContext = NULL;
    openClCommandQueues = NULL;
    openClProgram = NULL;

    supportDoublePrecision = true;

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::GPUInterface\n");
#endif
}

GPUInterface::~GPUInterface() {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::~GPUInterface\n");
#endif

    // TODO: cleanup mem objects, kernels

    if (openClProgram != NULL)
        SAFE_CL(clReleaseProgram(openClProgram));

    if (openClCommandQueues != NULL) {
        for (int i=0; i < BEAGLE_STREAM_COUNT; i++) {
            SAFE_CL(clReleaseCommandQueue(openClCommandQueues[i]));
        }
        free(openClCommandQueues);
    }

    if (openClContext != NULL)
        SAFE_CL(clReleaseContext(openClContext));

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::~GPUInterface\n");
#endif
}

int GPUInterface::Initialize() {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::Initialize\n");
#endif

    cl_uint numPlatforms = 0;
    SAFE_CL(clGetPlatformIDs(0, NULL, &numPlatforms));
    cl_platform_id* platforms = new cl_platform_id[numPlatforms];
    SAFE_CL(clGetPlatformIDs(numPlatforms, platforms, NULL));

    int deviceAdded = 0;
    for (int i=0; i<numPlatforms; i++) {
        cl_uint numDevices = 0;
        SAFE_CL(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices));
        cl_device_id* deviceIds = new cl_device_id[numDevices];
        SAFE_CL(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, numDevices, deviceIds, NULL));
        for (int j=0; j<numDevices; j++) {
            openClDeviceId = deviceIds[j];
            BeagleDeviceImplementationCodes deviceCode = GetDeviceImplementationCode(-1);
            if (deviceCode != BEAGLE_OPENCL_DEVICE_APPLE_CPU &&
                //deviceCode != BEAGLE_OPENCL_DEVICE_APPLE_INTEL_GPU &&
                //deviceCode != BEAGLE_OPENCL_DEVICE_APPLE_AMD_GPU &&
                //deviceCode != BEAGLE_OPENCL_DEVICE_NVIDA_GPU &&
                    true)
                openClDeviceMap.insert(std::pair<int, cl_device_id>(deviceAdded++, deviceIds[j]));
            openClDeviceId = NULL;
        }
        delete[] deviceIds;
    }
    delete[] platforms;

#ifdef BEAGLE_DEBUG_VALUES
    printf("OpenCL devices: %lu\n", openClDeviceMap.size());
    for (int i=0; i<openClDeviceMap.size(); i++) {
        const size_t param_size = 256;
        char param_value[param_size];
        printf("Device %d:\n", i);
        SAFE_CL(clGetDeviceInfo(openClDeviceMap[i], CL_DEVICE_NAME, param_size, param_value, NULL));
        printf("\tDevice name: %s\n", param_value);
        SAFE_CL(clGetDeviceInfo(openClDeviceMap[i], CL_DEVICE_VERSION, param_size, param_value, NULL));
        printf("\tDevice version: %s\n", param_value);
        SAFE_CL(clGetDeviceInfo(openClDeviceMap[i], CL_DEVICE_VENDOR, param_size, param_value, NULL));
        printf("\tDevice vendor: %s\n", param_value);

        cl_uint param_value_uint;
        SAFE_CL(clGetDeviceInfo(openClDeviceMap[i], CL_DEVICE_VENDOR_ID, sizeof(param_value_uint), &param_value_uint, NULL));
        printf("\tDevice vendor id: %d\n", param_value_uint);

        size_t param_value_t = 0;
        SAFE_CL(clGetDeviceInfo(openClDeviceMap[i], CL_DEVICE_MAX_WORK_GROUP_SIZE,
                                sizeof(param_value_t), &param_value_t, NULL));
        printf("\tCL_DEVICE_MAX_WORK_GROUP_SIZE: %lu\n", param_value_t);
        SAFE_CL(clGetDeviceInfo(openClDeviceMap[i], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                                sizeof(param_value_t), &param_value_t, NULL));
        printf("\tCL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: %lu\n", param_value_t);
        size_t* max_work_items = new size_t[param_value_t];
        SAFE_CL(clGetDeviceInfo(openClDeviceMap[i], CL_DEVICE_MAX_WORK_ITEM_SIZES,
                                sizeof(size_t)*param_value_t, max_work_items, NULL));
        for (int j=0; j<param_value_t; j++)
            printf("\tCL_DEVICE_MAX_WORK_ITEM_SIZES[%d]: %lu\n", j, max_work_items[j]);
        delete[] max_work_items;

        cl_platform_id platform;
        SAFE_CL(clGetDeviceInfo(openClDeviceMap[i], CL_DEVICE_PLATFORM,
                                sizeof(cl_platform_id), &platform, NULL));
        printf("\tOpenCL platform: ");
        SAFE_CL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, param_size, param_value, NULL));
        printf("%s | ", param_value);
        SAFE_CL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, param_size, param_value, NULL));
        printf("%s | ", param_value);
        SAFE_CL(clGetPlatformInfo(platform, CL_PLATFORM_VERSION, param_size, param_value, NULL));
        printf("%s\n", param_value);
    }
    printf("\n");
#endif

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::Initialize\n");
#endif

    return (openClDeviceMap.size() ? 1 : 0);
}

int GPUInterface::GetDeviceCount() {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::GetDeviceCount\n");
#endif

#ifdef BEAGLE_DEBUG_OPENCL_CORES
    for (int i=0; i<openClDeviceMap.size(); i++) {
        BeagleDeviceImplementationCodes deviceCode = GetDeviceImplementationCode(i);
        if (deviceCode == BEAGLE_OPENCL_DEVICE_INTEL_CPU) {
            cl_uint param_value_uint;
            SAFE_CL(clGetDeviceInfo(openClDeviceMap[i], CL_DEVICE_PARTITION_MAX_SUB_DEVICES, sizeof(param_value_uint), &param_value_uint, NULL));
            return openClDeviceMap.size() + param_value_uint-1;
        }
    }
#endif

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::GetDeviceCount\n");
#endif

    return openClDeviceMap.size();
}

void GPUInterface::InitializeKernelResource(int paddedStateCount,
                                            bool doublePrecision) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::InitializeKernelResource\n");
#endif

    int id = paddedStateCount;
    if (doublePrecision)
        id *= -1;

    bool CPUImpl = false;
    bool AppleCPUImpl = false;
    bool AMDImpl = false;
    BeagleDeviceImplementationCodes deviceCode = GetDeviceImplementationCode(-1);
    if (deviceCode == BEAGLE_OPENCL_DEVICE_INTEL_CPU ||
        deviceCode == BEAGLE_OPENCL_DEVICE_INTEL_MIC ||
        deviceCode == BEAGLE_OPENCL_DEVICE_AMD_CPU) {
        CPUImpl = true;
    } else if (deviceCode == BEAGLE_OPENCL_DEVICE_APPLE_CPU) {
        AppleCPUImpl = true;
    } else if (deviceCode == BEAGLE_OPENCL_DEVICE_AMD_GPU ||
               deviceCode == BEAGLE_OPENCL_DEVICE_APPLE_AMD_GPU ||
               deviceCode == BEAGLE_OPENCL_DEVICE_APPLE_INTEL_GPU ||
               deviceCode == BEAGLE_OPENCL_DEVICE_APPLE_APPLE_GPU
               ) {
        AMDImpl = true;
    }

    if (CPUImpl && paddedStateCount == 4) {
        switch(id) {
            case   -4: LOAD_KERNEL_INTO_RESOURCE(  4, DP,   4,_CPU,,); break;
            case    4: LOAD_KERNEL_INTO_RESOURCE(  4, SP,   4,_CPU,,); break;
        }
    } else if (AMDImpl && paddedStateCount > 32) {
        switch(id) {
            case  -48: LOAD_KERNEL_INTO_RESOURCE( 48, DP,  48,_AMDGPU,_AMDGPU,); break;
            case  -64: LOAD_KERNEL_INTO_RESOURCE( 64, DP,  64,_AMDGPU,_AMDGPU,); break;
            case  -80: LOAD_KERNEL_INTO_RESOURCE( 80, DP,  80,_AMDGPU,_AMDGPU,); break;
            case -128: LOAD_KERNEL_INTO_RESOURCE(128, DP, 128,_AMDGPU,_AMDGPU,); break;
            case -192: LOAD_KERNEL_INTO_RESOURCE(192, DP, 192,_AMDGPU,_AMDGPU,); break;
            case -256: LOAD_KERNEL_INTO_RESOURCE(256, DP, 256,_AMDGPU,_AMDGPU,); break;
            case   48: LOAD_KERNEL_INTO_RESOURCE( 48, SP,  48,_AMDGPU,_AMDGPU,); break;
            case   64: LOAD_KERNEL_INTO_RESOURCE( 64, SP,  64,_AMDGPU,_AMDGPU,); break;
            case   80: LOAD_KERNEL_INTO_RESOURCE( 80, SP,  80,_AMDGPU,_AMDGPU,); break;
            case  128: LOAD_KERNEL_INTO_RESOURCE(128, SP, 128,_AMDGPU,_AMDGPU,); break;
            case  192: LOAD_KERNEL_INTO_RESOURCE(192, SP, 192,_AMDGPU,_AMDGPU,); break;
            case  256: LOAD_KERNEL_INTO_RESOURCE(256, SP, 256,_AMDGPU,_AMDGPU,); break;
        }
    } else if (AppleCPUImpl) {
        switch(id) {
            case   -4: LOAD_KERNEL_INTO_RESOURCE(  4, DP,   4,_APPLECPU,,_APPLECPU); break;
            case  -16: LOAD_KERNEL_INTO_RESOURCE( 16, DP,  16,,,_APPLECPU); break;
            case  -32: LOAD_KERNEL_INTO_RESOURCE( 32, DP,  32,,,_APPLECPU); break;
            case  -48: LOAD_KERNEL_INTO_RESOURCE( 48, DP,  48,,,_APPLECPU); break;
            case  -64: LOAD_KERNEL_INTO_RESOURCE( 64, DP,  64,,,_APPLECPU); break;
            case  -80: LOAD_KERNEL_INTO_RESOURCE( 80, DP,  80,,,_APPLECPU); break;
            case -128: LOAD_KERNEL_INTO_RESOURCE(128, DP, 128,,,_APPLECPU); break;
            case -192: LOAD_KERNEL_INTO_RESOURCE(192, DP, 192,,,_APPLECPU); break;
            case -256: LOAD_KERNEL_INTO_RESOURCE(256, DP, 256,,,_APPLECPU); break;
            case    4: LOAD_KERNEL_INTO_RESOURCE(  4, SP,   4,_APPLECPU,,_APPLECPU); break;
            case   16: LOAD_KERNEL_INTO_RESOURCE( 16, SP,  16,,,_APPLECPU); break;
            case   32: LOAD_KERNEL_INTO_RESOURCE( 32, SP,  32,,,_APPLECPU); break;
            case   48: LOAD_KERNEL_INTO_RESOURCE( 48, SP,  48,,,_APPLECPU); break;
            case   64: LOAD_KERNEL_INTO_RESOURCE( 64, SP,  64,,,_APPLECPU); break;
            case   80: LOAD_KERNEL_INTO_RESOURCE( 80, SP,  80,,,_APPLECPU); break;
            case  128: LOAD_KERNEL_INTO_RESOURCE(128, SP, 128,,,_APPLECPU); break;
            case  192: LOAD_KERNEL_INTO_RESOURCE(192, SP, 192,,,_APPLECPU); break;
            case  256: LOAD_KERNEL_INTO_RESOURCE(256, SP, 256,,,_APPLECPU); break;
        }
    } else {
        switch(id) {
            case   -4: LOAD_KERNEL_INTO_RESOURCE(  4, DP,   4,,,); break;
            case  -16: LOAD_KERNEL_INTO_RESOURCE( 16, DP,  16,,,); break;
            case  -32: LOAD_KERNEL_INTO_RESOURCE( 32, DP,  32,,,); break;
            case  -48: LOAD_KERNEL_INTO_RESOURCE( 48, DP,  48,,,); break;
            case  -64: LOAD_KERNEL_INTO_RESOURCE( 64, DP,  64,,,); break;
            case  -80: LOAD_KERNEL_INTO_RESOURCE( 80, DP,  80,,,); break;
            case -128: LOAD_KERNEL_INTO_RESOURCE(128, DP, 128,,,); break;
            case -192: LOAD_KERNEL_INTO_RESOURCE(192, DP, 192,,,); break;
            case -256: LOAD_KERNEL_INTO_RESOURCE(256, DP, 256,,,); break;
            case    4: LOAD_KERNEL_INTO_RESOURCE(  4, SP,   4,,,); break;
            case   16: LOAD_KERNEL_INTO_RESOURCE( 16, SP,  16,,,); break;
            case   32: LOAD_KERNEL_INTO_RESOURCE( 32, SP,  32,,,); break;
            case   48: LOAD_KERNEL_INTO_RESOURCE( 48, SP,  48,,,); break;
            case   64: LOAD_KERNEL_INTO_RESOURCE( 64, SP,  64,,,); break;
            case   80: LOAD_KERNEL_INTO_RESOURCE( 80, SP,  80,,,); break;
            case  128: LOAD_KERNEL_INTO_RESOURCE(128, SP, 128,,,); break;
            case  192: LOAD_KERNEL_INTO_RESOURCE(192, SP, 192,,,); break;
            case  256: LOAD_KERNEL_INTO_RESOURCE(256, SP, 256,,,); break;
        }
    }

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::InitializeKernelResource\n");
#endif
}

void GPUInterface::SetDevice(int deviceNumber,
                             int paddedStateCount,
                             int categoryCount,
                             int paddedPatternCount,
                             int unpaddedPatternCount,
                             int tipCount,
                             long flags) {

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::SetDevice\n");
#endif

#ifdef BEAGLE_DEBUG_OPENCL_CORES
    CreateDevice(deviceNumber);
#endif

    openClDeviceId = openClDeviceMap[deviceNumber];

    int err;

    openClContext = clCreateContext(NULL, 1, &openClDeviceId, NULL, NULL, &err);
    SAFE_CL(err);

    openClCommandQueues = (cl_command_queue*) malloc(sizeof(cl_command_queue) * BEAGLE_STREAM_COUNT);
    openClEvents = (cl_event*) malloc(sizeof(cl_event) * BEAGLE_STREAM_COUNT);

    cl_command_queue_properties queueProperties = 0;//CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
    for (int i=0; i < BEAGLE_STREAM_COUNT; i++) {
        openClCommandQueues[i] = clCreateCommandQueue(openClContext, openClDeviceId,
                                              queueProperties, &err);
        SAFE_CL(err);
        openClEvents[i] = clCreateUserEvent(openClContext, &err);
        SAFE_CL(err);
    }

    InitializeKernelResource(paddedStateCount, flags & BEAGLE_FLAG_PRECISION_DOUBLE);

    if (!kernelResource) {
        fprintf(stderr,"Critical error: unable to find kernel code for %d states.\n",paddedStateCount);
        exit(-1);
    }

    kernelResource->categoryCount = categoryCount;
    kernelResource->patternCount = paddedPatternCount;
    kernelResource->unpaddedPatternCount = unpaddedPatternCount;
    kernelResource->flags = flags;
#ifdef BEAGLE_SPECTRAL_KERNELS
    kernelResource->flags |= BEAGLE_FLAG_SPECTRAL_REPRESENTATION;
#endif

#if defined(FW_OPENCL_BINARY) || defined(FW_OPENCL_PROFILING)
    //=========================================================================================================
    FILE *fp = NULL;
    #if defined(FW_OPENCL_BINARY)
        const char *file_name = "kernels.ir";
    #else // FW_OPENCL_PROFILING
	    const char *file_name = "kernels.cl";
    #endif
    #ifdef _WIN32
    	if (fopen_s(&fp, file_name, "rb") != 0)
    	{
    		printf("ERROR: Failed to open kernels.\n");
    		exit(-1);
    	}
    #else
    	fp = fopen(file_name, "rb");
    	if (fp == 0)
    	{
    		printf("ERROR: Failed to open kernels.\n");
    		exit(-1);
    	}
    #endif
    fseek(fp, 0, SEEK_END);
    size_t kernels_length = ftell(fp);
    const unsigned char *kernels = (unsigned char*) malloc(sizeof(unsigned char) * kernels_length);
    assert( kernels && "Malloc failed" );
    rewind(fp);
    if (fread((void *)kernels, kernels_length, 1, fp) == 0)
    {
    	printf("Failed to read kernels.\n");
    	exit(-1);
    }
    fclose(fp);

    #if defined(FW_OPENCL_BINARY)
        openClProgram = clCreateProgramWithBinary(openClContext, 1, &openClDeviceId, &kernels_length,
                                                  (const unsigned char **)&kernels, NULL, &err);
    #else // FW_OPENCL_PROFILING
	    openClProgram = clCreateProgramWithSource(openClContext, 1, (const char **)&kernels,
                                                  &kernels_length, &err);
    #endif
	//=========================================================================================================
#else
	openClProgram = clCreateProgramWithSource(openClContext, 1,
		                                      (const char**) &kernelResource->kernelCode, NULL,
		                                      &err);
#endif

    SAFE_CL(err);
    if (!openClProgram) {
        fprintf(stderr, "OpenCL error: Failed to create kernels\n");
        exit(-1);
    }

    char buildDefs[1024] = "-w -D FW_OPENCL -D OPENCL_KERNEL_BUILD ";
#ifdef DLS_MACOS
    strcat(buildDefs, "-D DLS_MACOS ");
#elif defined(FW_OPENCL_PROFILING)
	strcat(buildDefs, "-profiling -s \"C:\\developer\\beagle-lib\\project\\beagle-vs-2012\\x64\\Release\\kernels.cl\" ");
#endif

    BeagleDeviceImplementationCodes deviceCode = GetDeviceImplementationCode(deviceNumber);
    if (deviceCode == BEAGLE_OPENCL_DEVICE_INTEL_CPU ||
        deviceCode == BEAGLE_OPENCL_DEVICE_INTEL_MIC ||
        deviceCode == BEAGLE_OPENCL_DEVICE_AMD_CPU) {
        strcat(buildDefs, "-D FW_OPENCL_CPU");
    } else if (deviceCode == BEAGLE_OPENCL_DEVICE_APPLE_CPU) {
        strcat(buildDefs, "-D FW_OPENCL_CPU -D FW_OPENCL_APPLECPU");
    } else if (deviceCode == BEAGLE_OPENCL_DEVICE_AMD_GPU) {
        strcat(buildDefs, "-D FW_OPENCL_AMDGPU");
    } else if (deviceCode == BEAGLE_OPENCL_DEVICE_APPLE_AMD_GPU) {
        strcat(buildDefs, "-D FW_OPENCL_AMDGPU -D FW_OPENCL_APPLEAMDGPU");
    }  else if (deviceCode == BEAGLE_OPENCL_DEVICE_APPLE_INTEL_GPU) {
        strcat(buildDefs, "-D FW_OPENCL_INTELGPU -D FW_OPENCL_APPLEINTELGPU");
    } else if (deviceCode == BEAGLE_OPENCL_DEVICE_APPLE_APPLE_GPU) {
        strcat(buildDefs, "-D FW_OPENCL_AMDGPU");
    }

    err = clBuildProgram(openClProgram, 0, NULL, buildDefs, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t len;
        char buffer[16384];

        fprintf(stderr, "OpenCL error: Failed to build kernels\n");

        clGetProgramBuildInfo(openClProgram, openClDeviceId, CL_PROGRAM_BUILD_LOG,
                              sizeof(buffer), buffer, &len);

        fprintf(stderr, "%s\n", buffer);

        exit(-1);
    }

// TODO unloading compiler to free resources is causing seg fault for Intel and NVIDIA platforms
// #ifdef CL_VERSION_1_2
//     cl_platform_id platform;
//     SAFE_CL(clGetDeviceInfo(openClDeviceId, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &platform, NULL));
//     SAFE_CL(clUnloadPlatformCompiler(platform));
// #else
//     SAFE_CL(clUnloadCompiler());
// #endif

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::SetDevice\n");
#endif
}

void GPUInterface::ResizeStreamCount(int newStreamCount) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::ResizeStreamCount\n");
#endif

    // TODO: write function if using more than one opencl queue

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::ResizeStreamCount\n");
#endif
}

void GPUInterface::SynchronizeHost() {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::SynchronizeHost\n");
#endif

    // for(int i=0; i<BEAGLE_STREAM_COUNT; i++) {
    //     SAFE_CL(clFinish(openClCommandQueues[i]));
    // }

    bool prof = beagleProfilingEnabled();
    std::chrono::high_resolution_clock::time_point t0;
    if (prof) t0 = std::chrono::high_resolution_clock::now();

    SAFE_CL(clFinish(openClCommandQueues[0]));

    if (prof) {
        double dt = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - t0).count();
        if (gBeagleProfile.warmupRemaining > 0) {
            gBeagleProfile.warmupRemaining--;
            if (gBeagleProfile.warmupRemaining == 0) gBeagleProfile.warm = true;
        } else {
            gBeagleProfile.syncHostCalls++;
            gBeagleProfile.syncHostSeconds += dt;
        }
    }

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::SynchronizeHost\n");
#endif
}

void GPUInterface::SynchronizeDevice() {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::SynchronizeDevice\n");
#endif

    // TODO: synchronize on device only

    // SAFE_CL(clFinish(openClCommandQueues[0]));

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::SynchronizeDevice\n");
#endif
}

void GPUInterface::SynchronizeDeviceWithIndex(int streamRecordIndex, int streamWaitIndex) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::SynchronizeDeviceWithIndex\n");
#endif

    // TODO: synchronize on device only

    // SAFE_CL(clFinish(openClCommandQueues[0]));

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::SynchronizeDeviceWithIndex\n");
#endif
}

GPUFunction GPUInterface::GetFunction(const char* functionName) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::GetFunction\n");
    fprintf(stderr,"\t\t\t\tFunction name: %s\n", functionName);
#endif

    GPUFunction openClFunction;

    int err;
    openClFunction = clCreateKernel(openClProgram, functionName, &err);
    if (!openClFunction) {
        fprintf(stderr, "OpenCL error: Failed to create compute kernel %s\n", functionName);
        exit(-1);
    }

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::GetFunction\n");
#endif

    return openClFunction;
}

void GPUInterface::LaunchKernel(GPUFunction deviceFunction,
                                Dim3Int block,
                                Dim3Int grid,
                                int parameterCountV,
                                int totalParameterCount,
                                ...) { // parameters
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::LaunchKernel\n");
#endif

    char kernelNameBuf[256] = {0};
    // Hoisted once instead of calling getenv() again on every parameter in the loops
    // below (previously re-checked per-arg, tens of getenv() calls per kernel launch).
    const bool debugArgs = (getenv("BEAGLE_DEBUG_KERNEL_ARGS") != NULL);
    if (debugArgs) {
        clGetKernelInfo(deviceFunction, CL_KERNEL_FUNCTION_NAME, sizeof(kernelNameBuf), kernelNameBuf, NULL);
        fprintf(stderr, "[LaunchKernel] %s: parameterCountV=%d totalParameterCount=%d\n",
                kernelNameBuf, parameterCountV, totalParameterCount);
    }

    bool prof = beagleProfilingEnabled();
    std::chrono::high_resolution_clock::time_point t0;
    if (prof) t0 = std::chrono::high_resolution_clock::now();

    va_list parameters;
    va_start(parameters, totalParameterCount);
    for(int i = 0; i < parameterCountV; i++) {
        void* param = (void*)(size_t)va_arg(parameters, GPUPtr);
        if (debugArgs) {
            fprintf(stderr, "[LaunchKernel] %s: ptr arg[%d] = %p\n", kernelNameBuf, i, param);
            fflush(stderr);
        }

        SAFE_CL(clSetKernelArg(deviceFunction, i, sizeof(param), &param));
    }
    for(int i = parameterCountV; i < totalParameterCount; i++) {
        unsigned int param = va_arg(parameters, unsigned int);
        if (debugArgs) {
            fprintf(stderr, "[LaunchKernel] %s: int arg[%d] = %u (signed: %d)\n", kernelNameBuf, i, param, (int)param);
            fflush(stderr);
        }

        SAFE_CL(clSetKernelArg(deviceFunction, i, sizeof(unsigned int), &param));
    }

    va_end(parameters);

    size_t localWorkSize[3];
    localWorkSize[0] = block.x;
    localWorkSize[1] = block.y;
    localWorkSize[2] = block.z;

    size_t globalWorkSize[3];
    globalWorkSize[0] = block.x * grid.x;
    globalWorkSize[1] = block.y * grid.y;
    globalWorkSize[2] = block.z * grid.z;

    if (debugArgs) {
        fprintf(stderr, "[LaunchKernel] %s: block=(%d,%d,%d) grid=(%d,%d,%d) local=(%zu,%zu,%zu) global=(%zu,%zu,%zu)\n",
                kernelNameBuf, block.x, block.y, block.z, grid.x, grid.y, grid.z,
                localWorkSize[0], localWorkSize[1], localWorkSize[2],
                globalWorkSize[0], globalWorkSize[1], globalWorkSize[2]);
        fflush(stderr);
    }

#ifdef BEAGLE_DEBUG_VALUES
    for (int i=0; i<3; i++) {
        printf("localWorkSize[%d]  = %lu\n", i, localWorkSize[i]);
        printf("globalWorkSize[%d] = %lu\n", i, globalWorkSize[i]);
    }
    size_t local;
    clGetKernelWorkGroupInfo(deviceFunction, openClDeviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    printf("local = %lu\n\n", local);
#endif

    if (debugArgs) {
        size_t maxLocal = 0;
        clGetKernelWorkGroupInfo(deviceFunction, openClDeviceId, CL_KERNEL_WORK_GROUP_SIZE,
                                  sizeof(maxLocal), &maxLocal, NULL);
        size_t prefMultiple = 0;
        clGetKernelWorkGroupInfo(deviceFunction, openClDeviceId, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                                  sizeof(prefMultiple), &prefMultiple, NULL);
        fprintf(stderr, "[LaunchKernel] %s: CL_KERNEL_WORK_GROUP_SIZE=%zu preferredMultiple=%zu (requested local=%zu)\n",
                kernelNameBuf, maxLocal, prefMultiple, localWorkSize[0]);
        fflush(stderr);
    }

    if (debugArgs) {
        fprintf(stderr, "[LaunchKernel] %s: about to enqueue\n", kernelNameBuf);
        fflush(stderr);
    }

    if (globalWorkSize[1] == 1 && globalWorkSize[2] == 1) {
        SAFE_CL(clEnqueueNDRangeKernel(openClCommandQueues[0], deviceFunction, 1, NULL,
                                       globalWorkSize, localWorkSize, 0, NULL, NULL));
    } else if (globalWorkSize[2] == 1) {
        SAFE_CL(clEnqueueNDRangeKernel(openClCommandQueues[0], deviceFunction, 2, NULL,
                                       globalWorkSize, localWorkSize, 0, NULL, NULL));
    } else {
        SAFE_CL(clEnqueueNDRangeKernel(openClCommandQueues[0], deviceFunction, 3, NULL,
                                       globalWorkSize, localWorkSize, 0, NULL, NULL));
    }

    if (prof && gBeagleProfile.warm) {
        double dt = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - t0).count();
        gBeagleProfile.launchKernelCalls++;
        gBeagleProfile.launchKernelSeconds += dt;
    }

    if (debugArgs) {
        fprintf(stderr, "[LaunchKernel] %s: enqueued OK, calling clFinish\n", kernelNameBuf);
        fflush(stderr);
        clFinish(openClCommandQueues[0]);
        fprintf(stderr, "[LaunchKernel] %s: clFinish returned OK\n", kernelNameBuf);
        fflush(stderr);
    }

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::LaunchKernel\n");
#endif
}

void GPUInterface::LaunchKernelConcurrent(GPUFunction deviceFunction,
                                          Dim3Int block,
                                          Dim3Int grid,
                                          int streamIndex,
                                          int waitIndex,
                                          int parameterCountV,
                                          int totalParameterCount,
                                          ...) { // parameters
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::LaunchKernel\n");
#endif

    char kernelNameBuf[256] = {0};
    // Hoisted once instead of calling getenv() again on every parameter in the loops
    // below (previously re-checked per-arg, tens of getenv() calls per kernel launch).
    const bool debugArgs = (getenv("BEAGLE_DEBUG_KERNEL_ARGS") != NULL);
    if (debugArgs) {
        clGetKernelInfo(deviceFunction, CL_KERNEL_FUNCTION_NAME, sizeof(kernelNameBuf), kernelNameBuf, NULL);
        fprintf(stderr, "[LaunchKernelConcurrent] %s: parameterCountV=%d totalParameterCount=%d streamIndex=%d waitIndex=%d\n",
                kernelNameBuf, parameterCountV, totalParameterCount, streamIndex, waitIndex);
    }

    bool prof = beagleProfilingEnabled();
    std::chrono::high_resolution_clock::time_point t0;
    if (prof) t0 = std::chrono::high_resolution_clock::now();

    va_list parameters;
    va_start(parameters, totalParameterCount);
    for(int i = 0; i < parameterCountV; i++) {
        void* param = (void*)(size_t)va_arg(parameters, GPUPtr);
        if (debugArgs) {
            fprintf(stderr, "[LaunchKernelConcurrent] %s: ptr arg[%d] = %p\n", kernelNameBuf, i, param);
            fflush(stderr);
        }

        SAFE_CL(clSetKernelArg(deviceFunction, i, sizeof(param), &param));
    }
    for(int i = parameterCountV; i < totalParameterCount; i++) {
        unsigned int param = va_arg(parameters, unsigned int);
        if (debugArgs) {
            fprintf(stderr, "[LaunchKernelConcurrent] %s: int arg[%d] = %u (signed: %d)\n", kernelNameBuf, i, param, (int)param);
            fflush(stderr);
        }

        SAFE_CL(clSetKernelArg(deviceFunction, i, sizeof(unsigned int), &param));
    }

    va_end(parameters);

    size_t localWorkSize[3];
    localWorkSize[0] = block.x;
    localWorkSize[1] = block.y;
    localWorkSize[2] = block.z;

    size_t globalWorkSize[3];
    globalWorkSize[0] = block.x * grid.x;
    globalWorkSize[1] = block.y * grid.y;
    globalWorkSize[2] = block.z * grid.z;

    if (debugArgs) {
        fprintf(stderr, "[LaunchKernelConcurrent] %s: block=(%d,%d,%d) grid=(%d,%d,%d) local=(%zu,%zu,%zu) global=(%zu,%zu,%zu)\n",
                kernelNameBuf, block.x, block.y, block.z, grid.x, grid.y, grid.z,
                localWorkSize[0], localWorkSize[1], localWorkSize[2],
                globalWorkSize[0], globalWorkSize[1], globalWorkSize[2]);
        fflush(stderr);
    }

#ifdef BEAGLE_DEBUG_VALUES
    for (int i=0; i<3; i++) {
        printf("localWorkSize[%d]  = %lu\n", i, localWorkSize[i]);
        printf("globalWorkSize[%d] = %lu\n", i, globalWorkSize[i]);
    }
    size_t local;
    clGetKernelWorkGroupInfo(deviceFunction, openClDeviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    printf("local = %lu\n\n", local);
#endif

    int dims = 3;
    if (globalWorkSize[1] == 1 && globalWorkSize[2] == 1) {
        dims = 1;
    } else if (globalWorkSize[2] == 1) {
        dims = 2;
    }

    if (debugArgs) {
        fprintf(stderr, "[LaunchKernelConcurrent] %s: about to enqueue (dims=%d)\n", kernelNameBuf, dims);
        fflush(stderr);
    }

    // if (streamIndex != -1) {
    //     streamIndex /= 2;
    //     // if (waitIndex != -1) {
    //     //     SAFE_CL(clFinish(openClCommandQueues[streamIndex % BEAGLE_STREAM_COUNT]));
    //     //     return;
    //     // }

    //     int streamIndexMod = streamIndex % BEAGLE_STREAM_COUNT;

    //     cl_command_queue commandQueue;
    //     commandQueue = openClCommandQueues[streamIndexMod];

    //     // printf("streamIndexMod:  %d; waitIndexMod %d\n",
    //     //        streamIndexMod, waitIndex % BEAGLE_STREAM_COUNT);

    //     if (waitIndex != -1) {
    //         waitIndex /= 2;
    //         int waitIndexMod = waitIndex % BEAGLE_STREAM_COUNT;
    //         // SAFE_CL(clFinish(openClCommandQueues[waitIndexMod]));
    //         SAFE_CL(clEnqueueBarrierWithWaitList(commandQueue, 1, &openClEvents[waitIndexMod], NULL));
    //         // SAFE_CL(clWaitForEvents(1, &openClEvents[waitIndexMod]));
    //     } else if (streamIndex==0) {
    //         SAFE_CL(clFinish(commandQueue)); // unclear why this is needed for OpenCL CPU
    //     }

    //     SAFE_CL(clEnqueueNDRangeKernel(commandQueue, deviceFunction, dims, NULL,
    //                                    globalWorkSize, localWorkSize,
    //                                    0, NULL, &openClEvents[streamIndexMod]));

    //     // SAFE_CL(clEnqueueNDRangeKernel(commandQueue, deviceFunction, dims, NULL,
    //     //                                globalWorkSize, localWorkSize,
    //     //                                0, NULL, NULL));
    //     // SAFE_CL(clEnqueueBarrierWithWaitList(commandQueue, 0, NULL, &openClEvents[streamIndexMod]));

    // } else {
        SAFE_CL(clEnqueueNDRangeKernel(openClCommandQueues[0], deviceFunction, dims, NULL,
                                       globalWorkSize, localWorkSize,
                                       0, NULL, NULL));
    // }
    if (prof && gBeagleProfile.warm) {
        double dt = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - t0).count();
        gBeagleProfile.launchKernelConcurrentCalls++;
        gBeagleProfile.launchKernelConcurrentSeconds += dt;
    }

    if (debugArgs) {
        fprintf(stderr, "[LaunchKernelConcurrent] %s: enqueued OK, calling clFinish\n", kernelNameBuf);
        fflush(stderr);
        clFinish(openClCommandQueues[0]);
        fprintf(stderr, "[LaunchKernelConcurrent] %s: clFinish returned OK\n", kernelNameBuf);
        fflush(stderr);
    }
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::LaunchKernel\n");
#endif
}

void* GPUInterface::MallocHost(size_t memSize) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::MallocHost\n");
#endif

    void* ptr;

#ifdef BEAGLE_MEMORY_PINNED
    ptr = AllocatePinnedHostMemory(memSize, false, false);
#else
    ptr = malloc(memSize);
#endif

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::MallocHost\n");
#endif

    return ptr;
}

void* GPUInterface::CallocHost(size_t size, size_t length) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::CallocHost\n");
#endif

    void* ptr;
    size_t memSize = size * length;

#ifdef BEAGLE_MEMORY_PINNED
    ptr = AllocatePinnedHostMemory(memSize, false, false);
    memset(ptr, 0, memSize);
#else
    ptr = calloc(size, length);
#endif

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::CallocHost\n");
#endif

    return ptr;
}

void* GPUInterface::AllocatePinnedHostMemory(size_t memSize, bool writeCombined, bool mapped) {
#ifdef BEAGLE_DEBUG_FLOW
   fprintf(stderr,"\t\t\tEntering GPUInterface::AllocatePinnedHostMemory\n");
#endif

    cl_mem_flags flags = 0;

    flags |= CL_MEM_ALLOC_HOST_PTR;
    flags |= CL_MEM_HOST_WRITE_ONLY;
    flags |= CL_MEM_READ_ONLY;

    int err;
    void* deviceBuffer = (void*) clCreateBuffer(openClContext, flags, memSize, NULL, &err);
    SAFE_CL(err);

#ifdef BEAGLE_DEBUG_FLOW
   fprintf(stderr, "\t\t\tLeaving  GPUInterface::AllocatePinnedHostMemory\n");
#endif

   return deviceBuffer;
}

void* GPUInterface::MapMemory(GPUPtr dPtr, size_t memSize) {
    int err;
    void* hostPtr = clEnqueueMapBuffer(openClCommandQueues[0], dPtr, CL_TRUE,
                                        CL_MAP_WRITE_INVALIDATE_REGION, 0, memSize, 0, NULL, NULL, &err);
    SAFE_CL(err);

    return hostPtr;
}

void GPUInterface::UnmapMemory(GPUPtr dPtr, void* hPtr) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::UnmapMemory\n");
#endif

    SAFE_CL(clEnqueueUnmapMemObject(openClCommandQueues[0], dPtr, hPtr, 0, NULL, NULL));

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving GPUInterface::UnmapMemory\n");
#endif
}

GPUPtr GPUInterface::AllocateMemory(size_t memSize) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::AllocateMemory\n");
#endif

    GPUPtr data;

    int err;
    data = clCreateBuffer(openClContext, CL_MEM_READ_WRITE, memSize, NULL, &err);
    SAFE_CL(err);

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::AllocateMemory\n");
#endif

    return data;
}

GPUPtr GPUInterface::AllocateRealMemory(size_t length) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::AllocateRealMemory\n");
#endif

    GPUPtr data;

    int err;
    data = clCreateBuffer(openClContext, CL_MEM_READ_WRITE, SIZE_REAL * length, NULL,
                          &err);
    SAFE_CL(err);

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::AllocateRealMemory\n");
#endif

    return data;
}

GPUPtr GPUInterface::AllocateIntMemory(size_t length) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::AllocateIntMemory\n");
#endif

    GPUPtr data;

    int err;
    data = clCreateBuffer(openClContext, CL_MEM_READ_WRITE, SIZE_INT * length, NULL,
                          &err);
    SAFE_CL(err);

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::AllocateIntMemory\n");
#endif

    return data;
}

GPUPtr GPUInterface::CreateSubPointer(GPUPtr dPtr,
                                      size_t offset,
                                      size_t size) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::CreateSubPointer\n");
#endif

#ifdef FW_OPENCL_ALTERA
    GPUPtr subPtr = dPtr;// + offset;
#else
    GPUPtr subPtr;

    const size_t param_size = 256;
    char param_value[param_size];
    cl_platform_id platform;
    SAFE_CL(clGetDeviceInfo(openClDeviceId, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &platform, NULL));
    SAFE_CL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, param_size, param_value, NULL));

    // TODO REVERT after discussion with DA
    if ((strcmp(param_value, "NVIDIA Corporation") != 0 && strcmp(param_value, "Apple") != 0) || offset != 0) { //TODO: use the right platform + device check
        cl_buffer_region dPtrRegion;
        dPtrRegion.origin = offset;
        dPtrRegion.size = size;

        int err;
        subPtr = clCreateSubBuffer(dPtr, 0, CL_BUFFER_CREATE_TYPE_REGION, &dPtrRegion, &err);
        SAFE_CL(err);
    } else {
        subPtr = dPtr;
    }
#endif

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::CreateSubPointer\n");
#endif

    return subPtr;
}

size_t GPUInterface::AlignMemOffset(size_t offset) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::AlignMemOffset\n");
#endif

    size_t alignedOffset = offset;

    const size_t param_size = 256;
    char param_value[param_size];
    cl_platform_id platform;
    SAFE_CL(clGetDeviceInfo(openClDeviceId, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &platform, NULL));
    SAFE_CL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, param_size, param_value, NULL));

    // TODO REVERT after discussion with DA
    if ((strcmp(param_value, "NVIDIA Corporation") != 0 && strcmp(param_value, "Apple") != 0)) { //TODO: use the right platform + device check
        cl_uint baseAlign;
        SAFE_CL(clGetDeviceInfo(openClDeviceId, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(cl_uint), &baseAlign, NULL));
        baseAlign /= 8; // convert bits to bytes;
        alignedOffset = ceil((float)offset/baseAlign) * baseAlign;
    }

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::AlignMemOffset\n");
#endif

    return alignedOffset;
}

void GPUInterface::MemsetShort(GPUPtr dest,
                               unsigned short val,
                               size_t count) {
    assert(0); // TODO: write function
//#ifdef BEAGLE_DEBUG_FLOW
//    fprintf(stderr, "\t\t\tEntering GPUInterface::MemsetShort\n");
//#endif
//
//    SAFE_CUPP(cuMemsetD16(dest, val, count));
//
//#ifdef BEAGLE_DEBUG_FLOW
//    fprintf(stderr, "\t\t\tLeaving  GPUInterface::MemsetShort\n");
//#endif

}

void GPUInterface::MemsetZero(GPUPtr dest,
                              size_t memSize) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::MemsetZero\n");
#endif

    /* Deliberately not clEnqueueFillBuffer: on this OpenCL implementation
     * (observed on Apple M1 Max), a kernel launched afterward on the same
     * in-order queue intermittently read stale/garbage data from a
     * clEnqueueFillBuffer-zeroed region — completion wasn't reliably
     * ordered before the next command despite the in-order queue, even
     * with an explicit clFinish() inserted right after the fill (residual
     * ~1/30 failure rate on adjointtest4 with clFinish, 0/45 with the
     * blocking write below). Falls back to the same blocking
     * clEnqueueWriteBuffer path MemcpyHostToDevice already uses elsewhere
     * in this file, which has no such issue.  */
    void* hZero = calloc(1, memSize);
    SAFE_CL(clEnqueueWriteBuffer(openClCommandQueues[0], dest, CL_TRUE, 0, memSize, hZero, 0,
                                 NULL, NULL));
    free(hZero);

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::MemsetZero\n");
#endif
}


void GPUInterface::MemcpyHostToDevice(GPUPtr dest,
                                      const void* src,
                                      size_t memSize) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::MemcpyHostToDevice\n");
#endif

    bool prof = beagleProfilingEnabled();
    std::chrono::high_resolution_clock::time_point t0;
    if (prof) t0 = std::chrono::high_resolution_clock::now();

    // CL_TRUE => blocking write: this call does not return until the transfer
    // completes, regardless of payload size.
    SAFE_CL(clEnqueueWriteBuffer(openClCommandQueues[0], dest, CL_TRUE, 0, memSize, src, 0,
                                 NULL, NULL));

    if (prof && gBeagleProfile.warm) {
        double dt = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - t0).count();
        gBeagleProfile.memcpyH2DCalls++;
        gBeagleProfile.memcpyH2DSeconds += dt;
        gBeagleProfile.memcpyH2DBytes += memSize;
    }

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::MemcpyHostToDevice\n");
#endif
}

void GPUInterface::MemcpyDeviceToHost(void* dest,
                                      const GPUPtr src,
                                      size_t memSize) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::MemcpyDeviceToHost\n");
#endif

    SAFE_CL(clEnqueueReadBuffer(openClCommandQueues[0], src, CL_TRUE, 0, memSize, dest, 0,
                                NULL, NULL));

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::MemcpyDeviceToHost\n");
#endif

}

void GPUInterface::MemcpyDeviceToDevice(GPUPtr dest,
                                        GPUPtr src,
                                        size_t memSize) {

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::MemcpyDeviceToDevice\n");
#endif

    SAFE_CL(clEnqueueCopyBuffer(openClCommandQueues[0], src, dest, 0, 0, memSize, 0,
                                 NULL, NULL));

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::MemcpyDeviceToDevice\n");
#endif

}


void GPUInterface::FreeHostMemory(void* hPtr) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::FreeHostMemory\n");
#endif

#ifdef BEAGLE_MEMORY_PINNED
    FreePinnedHostMemory(hPtr);
#else
    free(hPtr);
#endif

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::FreeHostMemory\n");
#endif
}

void GPUInterface::FreePinnedHostMemory(void* hPtr) {
#ifdef BEAGLE_DEBUG_FLOW
   fprintf(stderr, "\t\t\tEntering GPUInterface::FreePinnedHostMemory\n");
#endif

    SAFE_CL(clReleaseMemObject((GPUPtr) hPtr));

#ifdef BEAGLE_DEBUG_FLOW
   fprintf(stderr,"\t\t\tLeaving  GPUInterface::FreePinnedHostMemory\n");
#endif
}

void GPUInterface::FreeMemory(GPUPtr dPtr) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::FreeMemory\n");
#endif

    SAFE_CL(clReleaseMemObject(dPtr));

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::FreeMemory\n");
#endif
}

GPUPtr GPUInterface::GetDeviceHostPointer(void* hPtr) {
    assert(0); // TODO: write function
	return NULL;
//#ifdef BEAGLE_DEBUG_FLOW
//    fprintf(stderr, "\t\t\tEntering GPUInterface::GetDeviceHostPointer\n");
//#endif
//
//    GPUPtr dPtr;
//
//    SAFE_CUPP(cuMemHostGetDeviceHostPointer(&dPtr, hPtr, 0));
//
//#ifdef BEAGLE_DEBUG_FLOW
//    fprintf(stderr,"\t\t\tLeaving  GPUInterface::GetDeviceHostPointer\n");
//#endif
//
//    return dPtr;
}

size_t GPUInterface::GetAvailableMemory() {
    assert(0); // TODO: write function
	return 0;
//    return availableMem;
}

void GPUInterface::GetDeviceName(int deviceNumber,
                                 char* deviceName,
                                 int nameLength) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::GetDeviceName\n");
#endif

    SAFE_CL(clGetDeviceInfo(openClDeviceMap[deviceNumber], CL_DEVICE_NAME, sizeof(char) * nameLength, deviceName, NULL));

#if defined(BEAGLE_DEBUG_OPENCL_CORES)
    cl_uint mpCount = 0;
    SAFE_CL(clGetDeviceInfo(openClDeviceMap[deviceNumber], CL_DEVICE_MAX_COMPUTE_UNITS,
                            sizeof(cl_uint), &mpCount, NULL));

    char mpCountStr[12];
    snprintf(mpCountStr, 12, "%d", mpCount);
    strcat(deviceName, " (");
    strcat(deviceName, mpCountStr);
    (mpCount==1?strcat(deviceName, " compute unit)"):strcat(deviceName, " compute units)"));
#endif

    const size_t param_size = 256;
    char param_value[param_size];
    SAFE_CL(clGetDeviceInfo(openClDeviceMap[deviceNumber], CL_DEVICE_VERSION, param_size, param_value, NULL));

    size_t vlen = strlen(param_value);
    while (vlen > 0 && param_value[vlen - 1] == ' ') param_value[--vlen] = '\0';

    strcat(deviceName, " (");
    strcat(deviceName, param_value);
    strcat(deviceName, ")");

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::GetDeviceName\n");
#endif
}

bool GPUInterface::GetSupportsDoublePrecision(int deviceNumber) {

    cl_uint supportsDouble = 0;

    SAFE_CL(clGetDeviceInfo(openClDeviceMap[deviceNumber], CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, sizeof(cl_uint), &supportsDouble, NULL));

    return supportsDouble;
}

void GPUInterface::GetDeviceDescription(int deviceNumber,
                                        char* deviceDescription) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::GetDeviceDescription\n");
#endif

    cl_device_id tmpOpenClDevice = openClDeviceMap[deviceNumber];

    cl_ulong totalGlobalMemory = 0;
    cl_uint clockSpeed = 0;
    unsigned int mpCount = 0;

    SAFE_CL(clGetDeviceInfo(tmpOpenClDevice, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong),
                            &totalGlobalMemory, NULL));
    SAFE_CL(clGetDeviceInfo(tmpOpenClDevice, CL_DEVICE_MAX_CLOCK_FREQUENCY,
                            sizeof(cl_uint), &clockSpeed, NULL));
    SAFE_CL(clGetDeviceInfo(tmpOpenClDevice, CL_DEVICE_MAX_COMPUTE_UNITS,
                            sizeof(unsigned int), &mpCount, NULL));

    snprintf(deviceDescription, 128,
            "Global memory (MB): %d | Clock speed (Ghz): %1.2f | Number of compute units: %d",
            int(totalGlobalMemory / 1024.0 / 1024.0), clockSpeed / 1000.0, mpCount);

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::GetDeviceDescription\n");
#endif
}

void GPUInterface::PrintfDeviceInt(GPUPtr dPtr,
                             int length) {
    int* hPtr = (int*) malloc(SIZE_INT * length);

    MemcpyDeviceToHost(hPtr, dPtr, SIZE_INT * length);

    printfInt(hPtr, length);

    free(hPtr);
}

long GPUInterface::GetDeviceTypeFlag(int deviceNumber) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::GetDeviceTypeFlag\n");
#endif

    cl_device_id deviceId;

    if (deviceNumber < 0)
        deviceId = openClDeviceId;
    else
        deviceId = openClDeviceMap[deviceNumber];

    cl_device_type deviceType;
    SAFE_CL(clGetDeviceInfo(deviceId, CL_DEVICE_TYPE,
                            sizeof(cl_device_type), &deviceType, NULL));

    long deviceTypeFlag;
    if (deviceType == CL_DEVICE_TYPE_GPU)
        deviceTypeFlag = BEAGLE_FLAG_PROCESSOR_GPU;
    else if (deviceType == CL_DEVICE_TYPE_CPU)
        deviceTypeFlag = BEAGLE_FLAG_PROCESSOR_CPU;
    else
        deviceTypeFlag = BEAGLE_FLAG_PROCESSOR_OTHER;

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::GetDeviceTypeFlag\n");
#endif

    return deviceTypeFlag;
}

BeagleDeviceImplementationCodes GPUInterface::GetDeviceImplementationCode(int deviceNumber) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::GetDeviceImplementationCode\n");
#endif

    BeagleDeviceImplementationCodes deviceCode = BEAGLE_OPENCL_DEVICE_GENERIC;

    cl_device_id deviceId;

    if (deviceNumber < 0)
        deviceId = openClDeviceId;
    else
        deviceId = openClDeviceMap[deviceNumber];

    const size_t param_size = 256;
    char device_string[param_size];
    char platform_string[param_size];
    cl_platform_id platform;
    SAFE_CL(clGetDeviceInfo(deviceId, CL_DEVICE_VENDOR, param_size, device_string, NULL));
    SAFE_CL(clGetDeviceInfo(deviceId, CL_DEVICE_PLATFORM,
                            sizeof(cl_platform_id), &platform, NULL));
    SAFE_CL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, param_size, platform_string, NULL));

    long deviceTypeFlag = GetDeviceTypeFlag(deviceNumber);

    if (!strncmp("Intel", platform_string, strlen("Intel"))) {
        if (deviceTypeFlag == BEAGLE_FLAG_PROCESSOR_CPU)
            deviceCode = BEAGLE_OPENCL_DEVICE_INTEL_CPU;
        else if (deviceTypeFlag == BEAGLE_FLAG_PROCESSOR_GPU)
            deviceCode = BEAGLE_OPENCL_DEVICE_INTEL_GPU;
        else if (deviceTypeFlag == BEAGLE_FLAG_PROCESSOR_OTHER)
            deviceCode = BEAGLE_OPENCL_DEVICE_INTEL_MIC;
    } else if (!strncmp("AMD", platform_string, strlen("AMD"))) {
        if (deviceTypeFlag == BEAGLE_FLAG_PROCESSOR_CPU)
            deviceCode = BEAGLE_OPENCL_DEVICE_AMD_CPU;
        else if (deviceTypeFlag == BEAGLE_FLAG_PROCESSOR_GPU)
            deviceCode = BEAGLE_OPENCL_DEVICE_AMD_GPU;
    } else if (!strncmp("Apple", platform_string, strlen("Apple"))) {
        if (deviceTypeFlag == BEAGLE_FLAG_PROCESSOR_CPU)
            deviceCode = BEAGLE_OPENCL_DEVICE_APPLE_CPU;
        else if (!strncmp("AMD", device_string, strlen("AMD")) &&
                 (deviceTypeFlag == BEAGLE_FLAG_PROCESSOR_GPU))
            deviceCode = BEAGLE_OPENCL_DEVICE_APPLE_AMD_GPU;
        else if (!strncmp("Intel", device_string, strlen("Intel")) &&
                 (deviceTypeFlag == BEAGLE_FLAG_PROCESSOR_GPU))
            deviceCode = BEAGLE_OPENCL_DEVICE_APPLE_INTEL_GPU;
        else if (!strncmp("Apple", device_string, strlen("Apple")) &&
                (deviceTypeFlag == BEAGLE_FLAG_PROCESSOR_GPU))
            deviceCode = BEAGLE_OPENCL_DEVICE_APPLE_APPLE_GPU;
    } else if (!strncmp("NVIDIA", platform_string, strlen("NVIDIA"))) {
        deviceCode = BEAGLE_OPENCL_DEVICE_NVIDA_GPU;
    }

// printf("platform_string %s\n", platform_string);
// printf("device_string %s\n", device_string);
// printf("deviceTypeFlag = %d\n", deviceTypeFlag);
// printf("deviceCode = %d\n", deviceCode);

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::GetDeviceImplementationCode\n");
#endif

    return deviceCode;
}

const char* GPUInterface::GetCLErrorDescription(int errorCode) {
    const char* errorDesc;

    // Error Codes (from cl.h)
    switch(errorCode) {
        case CL_SUCCESS                                  : errorDesc = "CL_SUCCESS"; break;
        case CL_DEVICE_NOT_FOUND                         : errorDesc = "CL_DEVICE_NOT_FOUND"; break;
        case CL_DEVICE_NOT_AVAILABLE                     : errorDesc = "CL_DEVICE_NOT_AVAILABLE"; break;
        case CL_COMPILER_NOT_AVAILABLE                   : errorDesc = "CL_COMPILER_NOT_AVAILABLE"; break;
        case CL_MEM_OBJECT_ALLOCATION_FAILURE            : errorDesc = "CL_MEM_OBJECT_ALLOCATION_FAILURE"; break;
        case CL_OUT_OF_RESOURCES                         : errorDesc = "CL_OUT_OF_RESOURCES"; break;
        case CL_OUT_OF_HOST_MEMORY                       : errorDesc = "CL_OUT_OF_HOST_MEMORY"; break;
        case CL_PROFILING_INFO_NOT_AVAILABLE             : errorDesc = "CL_PROFILING_INFO_NOT_AVAILABLE"; break;
        case CL_MEM_COPY_OVERLAP                         : errorDesc = "CL_MEM_COPY_OVERLAP"; break;
        case CL_IMAGE_FORMAT_MISMATCH                    : errorDesc = "CL_IMAGE_FORMAT_MISMATCH"; break;
        case CL_IMAGE_FORMAT_NOT_SUPPORTED               : errorDesc = "CL_IMAGE_FORMAT_NOT_SUPPORTED"; break;
        case CL_BUILD_PROGRAM_FAILURE                    : errorDesc = "CL_BUILD_PROGRAM_FAILURE"; break;
        case CL_MAP_FAILURE                              : errorDesc = "CL_MAP_FAILURE"; break;
        case CL_MISALIGNED_SUB_BUFFER_OFFSET             : errorDesc = "CL_MISALIGNED_SUB_BUFFER_OFFSET"; break;
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: errorDesc = "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST"; break;
#ifdef CL_VERSION_1_2
        case CL_COMPILE_PROGRAM_FAILURE                  : errorDesc = "CL_COMPILE_PROGRAM_FAILURE"; break;
        case CL_LINKER_NOT_AVAILABLE                     : errorDesc = "CL_LINKER_NOT_AVAILABLE"; break;
        case CL_LINK_PROGRAM_FAILURE                     : errorDesc = "CL_LINK_PROGRAM_FAILURE"; break;
        case CL_DEVICE_PARTITION_FAILED                  : errorDesc = "CL_DEVICE_PARTITION_FAILED"; break;
        case CL_KERNEL_ARG_INFO_NOT_AVAILABLE            : errorDesc = "CL_KERNEL_ARG_INFO_NOT_AVAILABLE"; break;
#endif
        case CL_INVALID_VALUE                            : errorDesc = "CL_INVALID_VALUE"; break;
        case CL_INVALID_DEVICE_TYPE                      : errorDesc = "CL_INVALID_DEVICE_TYPE"; break;
        case CL_INVALID_PLATFORM                         : errorDesc = "CL_INVALID_PLATFORM"; break;
        case CL_INVALID_DEVICE                           : errorDesc = "CL_INVALID_DEVICE"; break;
        case CL_INVALID_CONTEXT                          : errorDesc = "CL_INVALID_CONTEXT"; break;
        case CL_INVALID_QUEUE_PROPERTIES                 : errorDesc = "CL_INVALID_QUEUE_PROPERTIES"; break;
        case CL_INVALID_COMMAND_QUEUE                    : errorDesc = "CL_INVALID_COMMAND_QUEUE"; break;
        case CL_INVALID_HOST_PTR                         : errorDesc = "CL_INVALID_HOST_PTR"; break;
        case CL_INVALID_MEM_OBJECT                       : errorDesc = "CL_INVALID_MEM_OBJECT"; break;
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR          : errorDesc = "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR"; break;
        case CL_INVALID_IMAGE_SIZE                       : errorDesc = "CL_INVALID_IMAGE_SIZE"; break;
        case CL_INVALID_SAMPLER                          : errorDesc = "CL_INVALID_SAMPLER"; break;
        case CL_INVALID_BINARY                           : errorDesc = "CL_INVALID_BINARY"; break;
        case CL_INVALID_BUILD_OPTIONS                    : errorDesc = "CL_INVALID_BUILD_OPTIONS"; break;
        case CL_INVALID_PROGRAM                          : errorDesc = "CL_INVALID_PROGRAM"; break;
        case CL_INVALID_PROGRAM_EXECUTABLE               : errorDesc = "CL_INVALID_PROGRAM_EXECUTABLE"; break;
        case CL_INVALID_KERNEL_NAME                      : errorDesc = "CL_INVALID_KERNEL_NAME"; break;
        case CL_INVALID_KERNEL_DEFINITION                : errorDesc = "CL_INVALID_KERNEL_DEFINITION"; break;
        case CL_INVALID_KERNEL                           : errorDesc = "CL_INVALID_KERNEL"; break;
        case CL_INVALID_ARG_INDEX                        : errorDesc = "CL_INVALID_ARG_INDEX"; break;
        case CL_INVALID_ARG_VALUE                        : errorDesc = "CL_INVALID_ARG_VALUE"; break;
        case CL_INVALID_ARG_SIZE                         : errorDesc = "CL_INVALID_ARG_SIZE"; break;
        case CL_INVALID_KERNEL_ARGS                      : errorDesc = "CL_INVALID_KERNEL_ARGS"; break;
        case CL_INVALID_WORK_DIMENSION                   : errorDesc = "CL_INVALID_WORK_DIMENSION"; break;
        case CL_INVALID_WORK_GROUP_SIZE                  : errorDesc = "CL_INVALID_WORK_GROUP_SIZE"; break;
        case CL_INVALID_WORK_ITEM_SIZE                   : errorDesc = "CL_INVALID_WORK_ITEM_SIZE"; break;
        case CL_INVALID_GLOBAL_OFFSET                    : errorDesc = "CL_INVALID_GLOBAL_OFFSET"; break;
        case CL_INVALID_EVENT_WAIT_LIST                  : errorDesc = "CL_INVALID_EVENT_WAIT_LIST"; break;
        case CL_INVALID_EVENT                            : errorDesc = "CL_INVALID_EVENT"; break;
        case CL_INVALID_OPERATION                        : errorDesc = "CL_INVALID_OPERATION"; break;
        case CL_INVALID_GL_OBJECT                        : errorDesc = "CL_INVALID_GL_OBJECT"; break;
        case CL_INVALID_BUFFER_SIZE                      : errorDesc = "CL_INVALID_BUFFER_SIZE"; break;
        case CL_INVALID_MIP_LEVEL                        : errorDesc = "CL_INVALID_MIP_LEVEL"; break;
        case CL_INVALID_GLOBAL_WORK_SIZE                 : errorDesc = "CL_INVALID_GLOBAL_WORK_SIZE"; break;
#ifndef FW_OPENCL_ALTERA
        case CL_INVALID_PROPERTY                         : errorDesc = "CL_INVALID_PROPERTY"; break;
#endif
#ifdef CL_VERSION_1_2
        case CL_INVALID_IMAGE_DESCRIPTOR                 : errorDesc = "CL_INVALID_IMAGE_DESCRIPTOR"; break;
        case CL_INVALID_COMPILER_OPTIONS                 : errorDesc = "CL_INVALID_COMPILER_OPTIONS"; break;
        case CL_INVALID_LINKER_OPTIONS                   : errorDesc = "CL_INVALID_LINKER_OPTIONS"; break;
        case CL_INVALID_DEVICE_PARTITION_COUNT           : errorDesc = "CL_INVALID_DEVICE_PARTITION_COUNT"; break;
#endif
        default: errorDesc = "Unknown error";
    }

    return errorDesc;
}

#ifdef BEAGLE_DEBUG_OPENCL_CORES
void GPUInterface::CreateDevice(int deviceNumber) {
    if (deviceNumber >= openClDeviceMap.size()) {
        int coreCount = deviceNumber - openClDeviceMap.size() + 1;
        for (int i=0; i<openClDeviceMap.size(); i++) {
            BeagleDeviceImplementationCodes deviceCode = GetDeviceImplementationCode(i);
            if (deviceCode == BEAGLE_OPENCL_DEVICE_INTEL_CPU) {
                cl_device_id subdevice_id;
                cl_uint num_entries_returned = 0;
                cl_device_partition_property props[coreCount + 3];
                props[0] = CL_DEVICE_PARTITION_BY_NAMES_INTEL;
                for (int j=0; j<coreCount; j++) {
                    props[1+j] = j;
                }
                props[coreCount+1] = CL_PARTITION_BY_NAMES_LIST_END_INTEL;
                props[coreCount+2] = 0;
                SAFE_CL(clCreateSubDevices(openClDeviceMap[i],
                                           props,
                                           1,
                                           &subdevice_id,
                                           &num_entries_returned));

                openClDeviceMap.insert(std::pair<int, cl_device_id>(openClDeviceMap.size()+coreCount-1, subdevice_id));

                break;
            }
        }
    }
}

void GPUInterface::ReleaseDevice(int deviceNumber) {
    const size_t param_size = sizeof(cl_device_id);
    cl_device_id parent_device = NULL;
    SAFE_CL(clGetDeviceInfo(openClDeviceMap[deviceNumber], CL_DEVICE_PARENT_DEVICE, param_size, &parent_device, NULL));
    if (parent_device != NULL) {
        SAFE_CL(clReleaseDevice(openClDeviceMap[deviceNumber]));
        openClDeviceMap.erase(deviceNumber);
    }
}
#endif


}; // namespace

