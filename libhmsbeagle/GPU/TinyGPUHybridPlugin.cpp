/**
 * libhmsbeagle plugin system — TinyGPUHybrid backend
 * @author Marc Suchard
 */

#include "libhmsbeagle/GPU/BeagleGPUImpl.h"
#include "libhmsbeagle/GPU/TinyGPUHybridPlugin.h"

namespace beagle {
namespace gpu {

TinyGPUHybridPlugin::TinyGPUHybridPlugin() :
    Plugin("GPU-TinyGPUHybrid", "GPU-TinyGPUHybrid")
{
    GPUInterface gpu;
    bool anyGPUFound  = false;
    bool anyGPUSupDP  = false;

    if (gpu.Initialize() == BEAGLE_SUCCESS) {
        int gpuDeviceCount = gpu.GetDeviceCount();
        anyGPUFound = (gpuDeviceCount > 0);
        for (int i = 0; i < gpuDeviceCount; i++) {
            int nameDescSize = 256;
            char* dName = (char*) malloc(sizeof(char) * nameDescSize);
            char* dDesc = (char*) malloc(sizeof(char) * nameDescSize);
            gpu.GetDeviceName(i, dName, nameDescSize);
            gpu.GetDeviceDescription(i, dDesc);

            BeagleResource resource;
            resource.name        = dName;
            resource.description = dDesc;
            resource.supportFlags =
                BEAGLE_FLAG_COMPUTATION_SYNCH  |
                BEAGLE_FLAG_PRECISION_SINGLE   |
                BEAGLE_FLAG_SCALING_MANUAL     | BEAGLE_FLAG_SCALING_ALWAYS |
                BEAGLE_FLAG_SCALING_AUTO       | BEAGLE_FLAG_SCALING_DYNAMIC |
                BEAGLE_FLAG_THREADING_NONE     |
                BEAGLE_FLAG_VECTOR_NONE        |
                BEAGLE_FLAG_PROCESSOR_GPU      |
                BEAGLE_FLAG_SCALERS_LOG        | BEAGLE_FLAG_SCALERS_RAW |
                BEAGLE_FLAG_EIGEN_COMPLEX      | BEAGLE_FLAG_EIGEN_REAL |
                BEAGLE_FLAG_INVEVEC_STANDARD   | BEAGLE_FLAG_INVEVEC_TRANSPOSED |
                BEAGLE_FLAG_FRAMEWORK_TINYGPU;

            if (gpu.GetSupportsDoublePrecision(i)) {
                resource.supportFlags |= BEAGLE_FLAG_PRECISION_DOUBLE;
                anyGPUSupDP = true;
            }

            resource.requiredFlags = BEAGLE_FLAG_FRAMEWORK_TINYGPU;
            beagleResources.push_back(resource);
        }
    }

    if (anyGPUFound) {
        using namespace tinygpu;
        if (anyGPUSupDP)
            beagleFactories.push_back(new BeagleGPUImplFactory<double>());
        beagleFactories.push_back(new BeagleGPUImplFactory<float>());
    }
}

TinyGPUHybridPlugin::~TinyGPUHybridPlugin() {}

} // namespace gpu
} // namespace beagle

extern "C" {
void* plugin_init(void) {
    return new beagle::gpu::TinyGPUHybridPlugin();
}
}
