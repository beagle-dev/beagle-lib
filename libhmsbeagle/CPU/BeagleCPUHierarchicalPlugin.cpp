/**
 * libhmsbeagle plugin system
 * @author Aaron E. Darling
 * Based on code found in "Dynamic Plugins for C++" by Arthur J. Musgrove
 * and published in Dr. Dobbs Journal, July 1, 2004.
 */

#include "libhmsbeagle/CPU/BeagleCPUHierarchicalPlugin.h"
#include "libhmsbeagle/CPU/BeagleCPUHierarchicalImpl.h"

namespace beagle {
namespace cpu {

BeagleCPUHierarchicalPlugin::BeagleCPUHierarchicalPlugin() :
Plugin("CPU-Hierarchical", "CPU-Hierarchical")
{
	BeagleResource resource;
#ifdef __aarch64__
	resource.name = (char*) "CPU (arm64)";
#else
	resource.name = (char*) "CPU (x86_64)";
#endif
	resource.description = (char*) "";
	resource.supportFlags = BEAGLE_FLAG_COMPUTATION_SYNCH |
	                        BEAGLE_FLAG_SCALING_MANUAL | BEAGLE_FLAG_SCALING_ALWAYS | BEAGLE_FLAG_SCALING_DYNAMIC |
	                        BEAGLE_FLAG_THREADING_NONE | BEAGLE_FLAG_THREADING_CPP |
	                        BEAGLE_FLAG_PROCESSOR_CPU |
	                        BEAGLE_FLAG_PRECISION_SINGLE | BEAGLE_FLAG_PRECISION_DOUBLE |
	                        BEAGLE_FLAG_VECTOR_NONE |
	                        BEAGLE_FLAG_SCALERS_LOG | BEAGLE_FLAG_SCALERS_RAW |
	                        BEAGLE_FLAG_HIERARCHICAL_REPRESENTATION |
	                        BEAGLE_FLAG_FRAMEWORK_CPU;
	resource.requiredFlags = BEAGLE_FLAG_FRAMEWORK_CPU;
	beagleResources.push_back(resource);

	beagleFactories.push_back(new beagle::cpu::BeagleCPUHierarchicalImplFactory<double>());
	beagleFactories.push_back(new beagle::cpu::BeagleCPUHierarchicalImplFactory<float>());
}

}
}

extern "C" {
__attribute__((visibility("default"))) void* plugin_init(void){
	return new beagle::cpu::BeagleCPUHierarchicalPlugin();
}
}
