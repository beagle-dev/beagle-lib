/**
 * libhmsbeagle plugin system
 * @author Aaron E. Darling
 * @author Xiang Ji
 * Based on code found in "Dynamic Plugins for C++" by Arthur J. Musgrove
 * and published in Dr. Dobbs Journal, July 1, 2004.
 */

#include "libhmsbeagle/CPU/BeagleCPUActionPlugin.h"
#include "libhmsbeagle/CPU/BeagleCPUActionImpl.h"
#include <iostream>

namespace beagle {
    namespace cpu {

        BeagleCPUActionPlugin::BeagleCPUActionPlugin() :
                Plugin("CPU-Action", "CPU-Action")
        {
            BeagleResource resource;
#ifdef __ARM64_ARCH_8__
            resource.name = (char*) "CPU (arm64)";
#else
            resource.name = (char*) "CPU (x86_64)";
#endif
            resource.description = (char*) "";
            resource.supportFlags = BEAGLE_FLAG_COMPUTATION_SYNCH |
                                    BEAGLE_FLAG_SCALING_MANUAL | BEAGLE_FLAG_SCALING_ALWAYS | BEAGLE_FLAG_SCALING_AUTO |
                                    BEAGLE_FLAG_THREADING_NONE | BEAGLE_FLAG_THREADING_CPP |
                                    BEAGLE_FLAG_PROCESSOR_CPU |
                                    BEAGLE_FLAG_PRECISION_DOUBLE |
                                    BEAGLE_FLAG_VECTOR_NONE |
                                    BEAGLE_FLAG_SCALERS_LOG | BEAGLE_FLAG_SCALERS_RAW |
                                    BEAGLE_FLAG_INVEVEC_STANDARD | BEAGLE_FLAG_INVEVEC_TRANSPOSED |
                                    BEAGLE_FLAG_PREORDER_TRANSPOSE_MANUAL | BEAGLE_FLAG_PREORDER_TRANSPOSE_AUTO |
                                    BEAGLE_FLAG_FRAMEWORK_CPU;
            resource.requiredFlags = BEAGLE_FLAG_FRAMEWORK_CPU;
            beagleResources.push_back(resource);

            // Optional for plugins: check if the hardware is compatible and only populate
            // list with compatible factories and resources

            beagleFactories.push_back(new beagle::cpu::BeagleCPUActionImplFactory<double>());
        }

    }	// namespace cpu
}	// namespace beagle


extern "C" {

void* plugin_init(void){
    return new beagle::cpu::BeagleCPUActionPlugin();
}
}