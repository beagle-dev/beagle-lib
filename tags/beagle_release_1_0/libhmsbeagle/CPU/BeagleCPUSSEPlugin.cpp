/**
 * libhmsbeagle plugin system
 * @author Aaron E. Darling
 * Based on code found in "Dynamic Plugins for C++" by Arthur J. Musgrove
 * and published in Dr. Dobbs Journal, July 1, 2004.
 */

#include "libhmsbeagle/CPU/BeagleCPUSSEPlugin.h"
#include "libhmsbeagle/CPU/BeagleCPU4StateSSEImpl.h"
#include "libhmsbeagle/CPU/BeagleCPUSSEImpl.h"
#include <iostream>

namespace beagle {
namespace cpu {

BeagleCPUSSEPlugin::BeagleCPUSSEPlugin() :
Plugin("CPU-SSE", "CPU-SSE")
{
	BeagleResource resource;
        resource.name = (char*) "CPU";
        resource.description = (char*) "";
        resource.supportFlags = BEAGLE_FLAG_COMPUTATION_SYNCH |
                                         BEAGLE_FLAG_SCALING_MANUAL | BEAGLE_FLAG_SCALING_ALWAYS | BEAGLE_FLAG_SCALING_AUTO |
                                         BEAGLE_FLAG_THREADING_NONE |
                                         BEAGLE_FLAG_PROCESSOR_CPU |
                                         BEAGLE_FLAG_PRECISION_SINGLE | BEAGLE_FLAG_PRECISION_DOUBLE |
                                         BEAGLE_FLAG_VECTOR_NONE |
                                         BEAGLE_FLAG_SCALERS_LOG | BEAGLE_FLAG_SCALERS_RAW |
                                         BEAGLE_FLAG_EIGEN_COMPLEX | BEAGLE_FLAG_EIGEN_REAL |
                                         BEAGLE_FLAG_INVEVEC_STANDARD | BEAGLE_FLAG_INVEVEC_TRANSPOSED;
        resource.supportFlags |= BEAGLE_FLAG_VECTOR_SSE;
        resource.requiredFlags = BEAGLE_FLAG_PROCESSOR_CPU;
	beagleResources.push_back(resource);

	// Optional for plugins: check if the hardware is compatible and only populate
	// list with compatible factories and resources

	// FIXME: the SSE plugin currently assumes all hardware is compatible
	beagleFactories.push_back(new beagle::cpu::BeagleCPU4StateSSEImplFactory<double>());
//	beagleFactories.push_back(new beagle::cpu::BeagleCPU4StateSSEImplFactory<float>()); // TODO Not yet written
	beagleFactories.push_back(new beagle::cpu::BeagleCPUSSEImplFactory<double>()); // TODO In process of writing (disabled until it works for all input)
//	beagleFactories.push_back(new beagle::cpu::BeagleCPUSSEImplFactory<float>()); // TODO Not yet written
}

}	// namespace cpu
}	// namespace beagle


extern "C" {
void* plugin_init(void){
	return new beagle::cpu::BeagleCPUSSEPlugin();
}
}

