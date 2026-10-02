/**
 * libhmsbeagle plugin system — TinyGPUHybrid backend
 * @author Marc Suchard
 */

#include "libhmsbeagle/GPU/BeagleGPUImpl.h"
#include "libhmsbeagle/GPU/TinyGPUHybridPlugin.h"

namespace beagle {
namespace gpu {

// TODO.md plan steps A7 and C16: the resource's one implementation, in the precision the request asks for: double when it
// requires double, or prefers it without asking for single; single otherwise (TinyGPU's only precision until double came). With
// the double and the single factory both registered, as other GPU plugins have them, BEAGLE retries a failed
// beagleCreateInstance at the other precision: a second probe or boot of the same GPU, which fails the same way (or, while the
// first attempt's crash guard is still exiting, on its lock), and the caller gets that attempt's error instead of the first's.
namespace {
class TinyGPUImplFactory : public BeagleImplFactory {
    tinygpu::BeagleGPUImplFactory<float> sp_;
    tinygpu::BeagleGPUImplFactory<double> dp_;
    const bool double_;
public:
    explicit TinyGPUImplFactory(bool supportsDouble) : double_(supportsDouble) {}
    BeagleImpl* createImpl(int tipCount, int partialsBufferCount, int compactBufferCount, int stateCount, int patternCount,
                           int eigenBufferCount, int matrixBufferCount, int categoryCount, int scaleBufferCount,
                           int resourceNumber, int pluginResourceNumber, long preferenceFlags, long requirementFlags,
                           int* errorCode) {
        const bool dp = double_ && ((requirementFlags & BEAGLE_FLAG_PRECISION_DOUBLE) ||
                                    ((preferenceFlags & BEAGLE_FLAG_PRECISION_DOUBLE) &&
                                     !((preferenceFlags | requirementFlags) & BEAGLE_FLAG_PRECISION_SINGLE)));
        BeagleImplFactory& f = dp ? (BeagleImplFactory&)dp_ : (BeagleImplFactory&)sp_;
        return f.createImpl(tipCount, partialsBufferCount, compactBufferCount, stateCount, patternCount, eigenBufferCount,
                            matrixBufferCount, categoryCount, scaleBufferCount, resourceNumber, pluginResourceNumber,
                            preferenceFlags, requirementFlags, errorCode);
    }
    const char* getName() { return double_ ? "GPU-SP-DP-TinyGPU" : sp_.getName(); }
    const long getFlags() { return sp_.getFlags() | (double_ ? dp_.getFlags() : 0); }
};
}  // namespace

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

    if (anyGPUFound)
        beagleFactories.push_back(new TinyGPUImplFactory(anyGPUSupDP));
}

TinyGPUHybridPlugin::~TinyGPUHybridPlugin() {}

} // namespace gpu
} // namespace beagle

extern "C" {
void* plugin_init(void) {
    return new beagle::gpu::TinyGPUHybridPlugin();
}
}
