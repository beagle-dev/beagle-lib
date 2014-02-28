/**
 * libhmsbeagle plugin system
 * @author Aaron E. Darling
 * Based on code found in "Dynamic Plugins for C++" by Arthur J. Musgrove
 * and published in Dr. Dobbs Journal, July 1, 2004.
 */

#include "libhmsbeagle/GPU/BeagleGPUImpl.h"
#include "libhmsbeagle/GPU/OpenCLPlugin.h"

namespace beagle {
namespace gpu {
    
OpenCLPlugin::OpenCLPlugin() :
Plugin("GPU-OpenCL", "GPU-OpenCL")
    {
        GPUInterface gpu;
        bool anyGPUSupportsOpenCL = false;
        bool anyGPUSupportsDP = false;
        if (gpu.Initialize()) {
            int gpuDeviceCount = gpu.GetDeviceCount();
            anyGPUSupportsOpenCL = (gpuDeviceCount > 0);
            for (int i = 0; i < gpuDeviceCount; i++) {
#ifdef BEAGLE_DEBUG_OPENCL_CORES
                gpu.CreateDevice(i);
#endif

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
                                        BEAGLE_FLAG_SCALING_MANUAL | BEAGLE_FLAG_SCALING_ALWAYS | 
                                        BEAGLE_FLAG_SCALING_AUTO | BEAGLE_FLAG_SCALING_DYNAMIC |
                                        BEAGLE_FLAG_THREADING_NONE |
                                        BEAGLE_FLAG_VECTOR_NONE |
                                        BEAGLE_FLAG_SCALERS_LOG | BEAGLE_FLAG_SCALERS_RAW |
                                        BEAGLE_FLAG_EIGEN_COMPLEX | BEAGLE_FLAG_EIGEN_REAL |
                                        BEAGLE_FLAG_INVEVEC_STANDARD | BEAGLE_FLAG_INVEVEC_TRANSPOSED |
                                        BEAGLE_FLAG_FRAMEWORK_OPENCL;

                long deviceTypeFlag = gpu.GetDeviceTypeFlag(i);
                
                resource.supportFlags |= deviceTypeFlag;

                // Determine DP capability
                if (gpu.GetSupportsDoublePrecision(i)) {
                	resource.supportFlags |= BEAGLE_FLAG_PRECISION_DOUBLE;
                	anyGPUSupportsDP = true;
                }
                
                resource.requiredFlags = BEAGLE_FLAG_FRAMEWORK_OPENCL;
                
                beagleResources.push_back(resource);

#ifdef BEAGLE_DEBUG_OPENCL_CORES
                gpu.ReleaseDevice(i);
#endif
            }
        }
        
        // Optional for plugins: check if the hardware is compatible and only populate
        // list with compatible factories
        //	if(beagleResources.size() > 0) {
        if (anyGPUSupportsOpenCL) {
        	using namespace opencl;
            beagleFactories.push_back(new BeagleGPUImplFactory<float>());
            if (anyGPUSupportsDP) {
                // TODO Uncomment when working
                beagleFactories.push_back(new BeagleGPUImplFactory<double>());
            }
        }
    }
OpenCLPlugin::~OpenCLPlugin()
{
}


}	// namespace gpu
}	// namespace beagle


extern "C" {
void* plugin_init(void){
	return new beagle::gpu::OpenCLPlugin();
}
}

