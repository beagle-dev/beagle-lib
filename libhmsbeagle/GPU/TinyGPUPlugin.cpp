/**
 * libhmsbeagle plugin system — TinyGPU backend
 * @author Marc Suchard
 */

#include "libhmsbeagle/GPU/BeagleGPUImpl.h"
#ifdef BEAGLE_TINYGPU_SPECTRAL
#include "libhmsbeagle/GPU/BeagleGPUSpectralImpl.h"
#endif
#include "libhmsbeagle/GPU/TinyGPUPlugin.h"

namespace beagle {
namespace gpu {

// TODO.md plan steps A7 and C16: the resource's one implementation, in the precision the request asks for: double when it
// requires double, or prefers it without asking for single; single otherwise (TinyGPU's only precision until double came). With
// the double and the single factory both registered, as other GPU plugins have them, BEAGLE retries a failed
// beagleCreateInstance at the other precision: a second probe or boot of the same GPU, which fails the same way (or, while the
// first attempt's crash guard is still exiting, on its lock), and the caller gets that attempt's error instead of the first's.
// The spectral implementation is a second such factory in this plugin (TODO.md plan step S3.3), whose boot both share;
// beagle.cpp offers a spectral factory only when spectral is asked for.
namespace {
template <template <typename> class Factory>
class TinyGPUImplFactory : public BeagleImplFactory {
    Factory<float> sp_;
    Factory<double> dp_;
    const bool double_;
    const char* const name_;
    const long hidden_;   // flags the factory does not offer here (plan decision D9)
public:
    TinyGPUImplFactory(bool supportsDouble, const char* name, long hidden)
        : double_(supportsDouble), name_(name), hidden_(hidden) {}
    BeagleImpl* createImpl(int tipCount, int partialsBufferCount, int compactBufferCount, int stateCount, int patternCount,
                           int eigenBufferCount, int matrixBufferCount, int categoryCount, int scaleBufferCount,
                           int resourceNumber, int pluginResourceNumber, long preferenceFlags, long requirementFlags,
                           int* errorCode) {
        // SetDevice picks the spectral modules from SPECTRAL in kFlags, which BeagleGPUImpl copies from the request even
        // when spectral is only preferred: the standard implementation never takes such a request. NULL, *errorCode
        // untouched: the caller keeps the spectral attempt's error, and nothing boots twice
        if (!(getFlags() & BEAGLE_FLAG_SPECTRAL_REPRESENTATION) &&
            ((preferenceFlags | requirementFlags) & BEAGLE_FLAG_SPECTRAL_REPRESENTATION))
            return NULL;
        const bool dp = double_ && ((requirementFlags & BEAGLE_FLAG_PRECISION_DOUBLE) ||
                                    ((preferenceFlags & BEAGLE_FLAG_PRECISION_DOUBLE) &&
                                     !((preferenceFlags | requirementFlags) & BEAGLE_FLAG_PRECISION_SINGLE)));
        BeagleImplFactory& f = dp ? (BeagleImplFactory&)dp_ : (BeagleImplFactory&)sp_;
        return f.createImpl(tipCount, partialsBufferCount, compactBufferCount, stateCount, patternCount, eigenBufferCount,
                            matrixBufferCount, categoryCount, scaleBufferCount, resourceNumber, pluginResourceNumber,
                            preferenceFlags, requirementFlags, errorCode);
    }
    const char* getName() { return double_ ? name_ : sp_.getName(); }
    const long getFlags() { return (sp_.getFlags() | (double_ ? dp_.getFlags() : 0)) & ~hidden_; }
};
}  // namespace

TinyGPUPlugin::TinyGPUPlugin() :
    Plugin("GPU-TinyGPU", "GPU-TinyGPU")
{
    GPUInterface gpu;
    bool anyGPUFound  = false;
    bool anyGPUSupDP  = false;
    bool anyGPUSpectral = false;   // some device has this build's spectral modules (plan decision D9)

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
#ifdef BEAGLE_TINYGPU_SPECTRAL
            if (gpu.GetSupportsSpectral(i)) {   // every spectral caller requires a pre-order transpose flag
                resource.supportFlags |= BEAGLE_FLAG_SPECTRAL_REPRESENTATION |
                                         BEAGLE_FLAG_PREORDER_TRANSPOSE_MANUAL | BEAGLE_FLAG_PREORDER_TRANSPOSE_AUTO;
                anyGPUSpectral = true;
            }
#endif

            resource.requiredFlags = BEAGLE_FLAG_FRAMEWORK_TINYGPU;
            beagleResources.push_back(resource);
        }
    }

    // the standard implementation keeps today's reach: with a spectral device the resource offers the pre-order transposes,
    // which the standard factory hides, so requests that require them still do not reach it (plan decision D9)
    const long transposes = BEAGLE_FLAG_PREORDER_TRANSPOSE_MANUAL | BEAGLE_FLAG_PREORDER_TRANSPOSE_AUTO;
    if (anyGPUFound)
        beagleFactories.push_back(new TinyGPUImplFactory<tinygpu::BeagleGPUImplFactory>(anyGPUSupDP, "GPU-SP-DP-TinyGPU",
                                                                                       anyGPUSpectral ? transposes : 0));
#ifdef BEAGLE_TINYGPU_SPECTRAL
    if (anyGPUSpectral)
        beagleFactories.push_back(new TinyGPUImplFactory<tinygpu::BeagleGPUSpectralImplFactory>(anyGPUSupDP,
                                                                                    "GPU-SP-DP-TinyGPU-Spectral", 0));
#endif
}

TinyGPUPlugin::~TinyGPUPlugin() {}

} // namespace gpu
} // namespace beagle

extern "C" {
__attribute__((visibility("default"))) void* plugin_init(void) {
    return new beagle::gpu::TinyGPUPlugin();
}
}
