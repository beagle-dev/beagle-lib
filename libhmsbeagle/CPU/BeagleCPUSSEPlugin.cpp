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

#ifdef __GNUC__
#include "cpuid.h"
#endif

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

	beagleFactories.push_back(new beagle::cpu::BeagleCPU4StateSSEImplFactory<double>());
//	beagleFactories.push_back(new beagle::cpu::BeagleCPU4StateSSEImplFactory<float>()); // TODO Not yet written
	beagleFactories.push_back(new beagle::cpu::BeagleCPUSSEImplFactory<double>()); // TODO In process of writing (disabled until it works for all input)
//	beagleFactories.push_back(new beagle::cpu::BeagleCPUSSEImplFactory<float>()); // TODO Not yet written
}

}	// namespace cpu
}	// namespace beagle


extern "C" {

#ifdef _WIN32
bool check_sse2(){
    unsigned int features;

    __asm
    {
        // Save registers
        push    eax
        push    ebx
        push    ecx
        push    edx

        // Get the feature flags (eax=1) from edx
        mov     eax, 1
        cpuid
        mov     features, edx

        // Restore registers
        pop     edx
        pop     ecx
        pop     ebx
        pop     eax
    }

// Bit 26 for SSE2 support
    return features & 0x04000000;
}

#endif

#ifdef __GNUC__
bool check_sse2()
{
  unsigned int eax, ebx, ecx, edx;
  unsigned int ext, sig;

  ext = 0;
  __get_cpuid_max( ext, &sig );
  printf( "ext=0x%x sig=0x%x\n", ext, sig );

  if (!__get_cpuid (1, &eax, &ebx, &ecx, &edx)) {
    printf( "__get_cpuid returned 0\n" );
    return 0;
  }

  /* Run SSE2 test only if host has SSE2 support.  */
  if (edx & bit_SSE2)
	return true;

	return false;
}

#endif


void* plugin_init(void){
	if(!check_sse2()){
		return NULL;	// no SSE no plugin?! 
	}
	return new beagle::cpu::BeagleCPUSSEPlugin();
}
}

