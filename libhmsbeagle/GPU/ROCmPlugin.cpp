/**
 * libhmsbeagle plugin system
 * ROCm/HIP backend plugin for AMD GPUs.
 */

#include "libhmsbeagle/GPU/BeagleGPUImpl.h"
#include "libhmsbeagle/GPU/ROCmPlugin.h"

namespace beagle {
namespace gpu {

ROCmPlugin::ROCmPlugin() :
    Plugin("GPU-ROCm", "GPU-ROCm")
{
    GPUInterface gpu;
    bool anyGPUFound = false;
    if (gpu.Initialize()) {
        int gpuDeviceCount = gpu.GetDeviceCount();
        anyGPUFound = (gpuDeviceCount > 0);
        for (int i = 0; i < gpuDeviceCount; i++) {
            int nameDescSize = 256;
            char* dName = (char*) malloc(sizeof(char) * nameDescSize);
            char* dDesc = (char*) malloc(sizeof(char) * nameDescSize);
            gpu.GetDeviceName(i, dName, nameDescSize);
            gpu.GetDeviceDescription(i, dDesc);

            BeagleResource resource;
            resource.name = dName;
            resource.description = dDesc;
            resource.supportFlags =
                BEAGLE_FLAG_COMPUTATION_SYNCH |
                BEAGLE_FLAG_COMPUTATION_ASYNCH |
                BEAGLE_FLAG_PRECISION_SINGLE |
                BEAGLE_FLAG_PRECISION_DOUBLE |
                BEAGLE_FLAG_SCALING_MANUAL | BEAGLE_FLAG_SCALING_ALWAYS |
                BEAGLE_FLAG_SCALING_AUTO   | BEAGLE_FLAG_SCALING_DYNAMIC |
                BEAGLE_FLAG_THREADING_NONE |
                BEAGLE_FLAG_VECTOR_NONE |
                BEAGLE_FLAG_PROCESSOR_GPU |
                BEAGLE_FLAG_SCALERS_LOG | BEAGLE_FLAG_SCALERS_RAW |
                BEAGLE_FLAG_EIGEN_COMPLEX | BEAGLE_FLAG_EIGEN_REAL |
                BEAGLE_FLAG_INVEVEC_STANDARD | BEAGLE_FLAG_INVEVEC_TRANSPOSED |
                BEAGLE_FLAG_PREORDER_TRANSPOSE_MANUAL | BEAGLE_FLAG_PREORDER_TRANSPOSE_AUTO |
                BEAGLE_FLAG_PARALLELOPS_GRID | BEAGLE_FLAG_PARALLELOPS_STREAMS |
                BEAGLE_FLAG_FRAMEWORK_ROCM;
            resource.requiredFlags = BEAGLE_FLAG_FRAMEWORK_ROCM;

            beagleResources.push_back(resource);
        }
    }

    if (anyGPUFound) {
        using namespace rocm_device;
        beagleFactories.push_back(new BeagleGPUImplFactory<double>());
        beagleFactories.push_back(new BeagleGPUImplFactory<float>());
    }
}

ROCmPlugin::~ROCmPlugin() {}

} // namespace gpu
} // namespace beagle

extern "C" {
void* plugin_init(void) {
    return new beagle::gpu::ROCmPlugin();
}
}
