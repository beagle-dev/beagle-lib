/**
 * libhmsbeagle plugin system
 * @author Aaron E. Darling
 * Based on code found in "Dynamic Plugins for C++" by Arthur J. Musgrove
 * and published in Dr. Dobbs Journal, July 1, 2004.
 */

#include "libhmsbeagle/GPU/BeagleGPUImpl.h"
#include "libhmsbeagle/GPU/CUDAPlugin.h"

namespace beagle {
namespace gpu {

CUDAPlugin::CUDAPlugin() :
Plugin("CUDA", "GPU")
{
        GPUInterface gpu;
        if (gpu.Initialize()) {
            int gpuDeviceCount = gpu.GetDeviceCount();
            for (int i = 0; i < gpuDeviceCount; i++) {
                int nameDescSize = 256;
                char* dName = (char*) malloc(sizeof(char) * nameDescSize);
                char* dDesc = (char*) malloc(sizeof(char) * nameDescSize);
                gpu.GetDeviceName(i, dName, nameDescSize);
                gpu.GetDeviceDescription(i, dDesc);
                BeagleResource resource;
                resource.name = dName;
                resource.description = dDesc;
                resource.supportFlags = BEAGLE_FLAG_COMPUTATION_SYNCH |
                                                     BEAGLE_FLAG_PRECISION_SINGLE |
                                                     BEAGLE_FLAG_SCALING_MANUAL | BEAGLE_FLAG_SCALING_ALWAYS | BEAGLE_FLAG_SCALING_AUTO |
                                                     BEAGLE_FLAG_THREADING_NONE |
                                                     BEAGLE_FLAG_VECTOR_NONE |
                                                     BEAGLE_FLAG_PROCESSOR_GPU |
                                                     BEAGLE_FLAG_SCALERS_LOG | BEAGLE_FLAG_SCALERS_RAW |
                                                     BEAGLE_FLAG_EIGEN_COMPLEX | BEAGLE_FLAG_EIGEN_REAL;
                resource.requiredFlags = BEAGLE_FLAG_PROCESSOR_GPU;
                beagleResources.push_back(resource);
            }
        }

	// Optional for plugins: check if the hardware is compatible and only populate
	// list with compatible factories
	if(beagleResources.size() > 0)
		beagleFactories.push_back(new beagle::gpu::BeagleGPUImplFactory());
}

CUDAPlugin::~CUDAPlugin()
{
	// Destory GPU kernel info
	if(beagleResources.size()>0)
	{
		GPUInterface gpu;
		gpu.DestroyKernelMap();
	}
}


}	// namespace gpu
}	// namespace beagle


extern "C" {
void* plugin_init(void){
	return new beagle::gpu::CUDAPlugin();
}
}

