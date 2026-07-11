#include "libhmsbeagle/GPU/GPUInterface.h"

#include <cstdio>

namespace {

cl_platform_id fakePlatform = reinterpret_cast<cl_platform_id>(1);
int deviceQueryCount = 0;

cl_int fakeGetPlatformIds(cl_uint numEntries, cl_platform_id* platforms,
                          cl_uint* numPlatforms)
{
    if (numPlatforms != NULL)
        *numPlatforms = 1;
    if (numEntries == 0)
        return CL_SUCCESS;
    if (numEntries == 1 && platforms != NULL) {
        platforms[0] = fakePlatform;
        return CL_SUCCESS;
    }
    return CL_INVALID_VALUE;
}

cl_int fakeGetDeviceIds(cl_platform_id platform, cl_device_type, cl_uint,
                        cl_device_id*, cl_uint*)
{
    ++deviceQueryCount;
    return platform == fakePlatform ? CL_INVALID_VALUE : CL_INVALID_PLATFORM;
}

} // namespace

int main()
{
    opencl_device::OpenCLDiscoveryApi discoveryApi = {fakeGetPlatformIds, fakeGetDeviceIds};
    opencl_device::GPUInterface gpu;

    if (gpu.Initialize(discoveryApi) != 0) {
        std::fprintf(stderr, "OpenCL discovery reported devices after an enumeration error\n");
        return 1;
    }
    if (deviceQueryCount != 1) {
        std::fprintf(stderr, "OpenCL discovery did not query exactly one fake platform\n");
        return 1;
    }
    return 0;
}
