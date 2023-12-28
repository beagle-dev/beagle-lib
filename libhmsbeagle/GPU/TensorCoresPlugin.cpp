/**
 * libhmsbeagle plugin system
 * @author Aaron E. Darling
 * Based on code found in "Dynamic Plugins for C++" by Arthur J. Musgrove
 * and published in Dr. Dobbs Journal, July 1, 2004.
 */

#include "libhmsbeagle/GPU/BeagleGPUImpl.h"
#include "libhmsbeagle/GPU/TensorCoresPlugin.h"

namespace beagle {
namespace gpu {

TensorCoresPlugin::TensorCoresPlugin() :
Plugin("GPU-TENSOR-CORES", "GPU-TENSOR-CORES")
{
        GPUInterface gpu;
        bool anyGPUSupportsCUDA = false;
        bool anyGPUSupportsDP = false;
        if (gpu.Initialize()) {
            int gpuDeviceCount = gpu.GetDeviceCount();
            anyGPUSupportsCUDA = (gpuDeviceCount > 0);
            for (int i = 0; i < gpuDeviceCount; i++) {
                int nameDescSize = 256;
                char* dName = (char*) malloc(sizeof(char) * nameDescSize);
                char* dDesc = (char*) malloc(sizeof(char) * nameDescSize);
                gpu.GetDeviceName(i, dName, nameDescSize);
                gpu.GetDeviceDescription(i, dDesc);

                BeagleResource resource;
                resource.name = dName;
                resource.description = dDesc;
                resource.supportFlags = BEAGLE_FLAG_COMPUTATION_SYNCH | BEAGLE_FLAG_COMPUTATION_ASYNCH |
                                        BEAGLE_FLAG_PRECISION_SINGLE |
                                        BEAGLE_FLAG_SCALING_MANUAL | BEAGLE_FLAG_SCALING_ALWAYS | 
                                        BEAGLE_FLAG_SCALING_AUTO | BEAGLE_FLAG_SCALING_DYNAMIC |
                                        BEAGLE_FLAG_THREADING_NONE |
//                                        BEAGLE_FLAG_VECTOR_NONE |
                                        BEAGLE_FLAG_VECTOR_TENSOR |
                                        BEAGLE_FLAG_PROCESSOR_GPU |
                                        BEAGLE_FLAG_SCALERS_LOG | BEAGLE_FLAG_SCALERS_RAW |
                                        BEAGLE_FLAG_EIGEN_COMPLEX | BEAGLE_FLAG_EIGEN_REAL |
                                        BEAGLE_FLAG_INVEVEC_STANDARD | BEAGLE_FLAG_INVEVEC_TRANSPOSED |
                                        BEAGLE_FLAG_PREORDER_TRANSPOSE_MANUAL | BEAGLE_FLAG_PREORDER_TRANSPOSE_AUTO |
                                        BEAGLE_FLAG_PARALLELOPS_GRID | BEAGLE_FLAG_PARALLELOPS_STREAMS |
                                        BEAGLE_FLAG_FRAMEWORK_CUDA;
                // Determine DP capability
                if (gpu.GetSupportsDoublePrecision(i)) {
                	resource.supportFlags |= BEAGLE_FLAG_PRECISION_DOUBLE;
                	anyGPUSupportsDP = true;
                }
                
                resource.requiredFlags = BEAGLE_FLAG_FRAMEWORK_CUDA | BEAGLE_FLAG_VECTOR_TENSOR;
                
                beagleResources.push_back(resource);
            }
        }

	// Optional for plugins: check if the hardware is compatible and only populate
	// list with compatible factories
//	if(beagleResources.size() > 0) {
    if (anyGPUSupportsCUDA) {
    	using namespace cuda;
        if (anyGPUSupportsDP) {
            // TODO Uncomment when working
            beagleFactories.push_back(new BeagleGPUImplFactory<double>());
        }
        beagleFactories.push_back(new BeagleGPUImplFactory<float>());
	}
}

TensorCoresPlugin::~TensorCoresPlugin()
{
}

}	// namespace gpu
}	// namespace beagle


extern "C" {
void* plugin_init(void){
	return new beagle::gpu::TensorCoresPlugin();
}
}

