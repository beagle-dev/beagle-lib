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

#ifdef HAVE_CPUID_H
	#if !defined(DLS_MACOS)
		#include <cpuid.h>
	#endif
#endif

namespace beagle {
namespace cpu {


BeagleCPUSSEPlugin::BeagleCPUSSEPlugin() :
Plugin("CPU-SSE", "CPU-SSE")
{
	BeagleResource resource;
#ifdef __aarch64__
	    resource.name = (char*) "CPU (arm64)";
#else
        resource.name = (char*) "CPU (x86_64)";
#endif
        resource.description = (char*) "";
        resource.supportFlags = BEAGLE_FLAG_COMPUTATION_SYNCH |
                                         BEAGLE_FLAG_SCALING_MANUAL | BEAGLE_FLAG_SCALING_ALWAYS | BEAGLE_FLAG_SCALING_AUTO |
                                         BEAGLE_FLAG_THREADING_NONE | BEAGLE_FLAG_THREADING_CPP |
                                         BEAGLE_FLAG_PROCESSOR_CPU |
                                         BEAGLE_FLAG_PRECISION_SINGLE | BEAGLE_FLAG_PRECISION_DOUBLE |
                                         BEAGLE_FLAG_VECTOR_NONE |
                                         BEAGLE_FLAG_SCALERS_LOG | BEAGLE_FLAG_SCALERS_RAW |
                                         BEAGLE_FLAG_EIGEN_COMPLEX | BEAGLE_FLAG_EIGEN_REAL |
                                         BEAGLE_FLAG_INVEVEC_STANDARD | BEAGLE_FLAG_INVEVEC_TRANSPOSED |
                                         BEAGLE_FLAG_PREORDER_TRANSPOSE_MANUAL | BEAGLE_FLAG_PREORDER_TRANSPOSE_AUTO |
                                         BEAGLE_FLAG_FRAMEWORK_CPU;
        resource.supportFlags |= BEAGLE_FLAG_VECTOR_SSE;
        resource.requiredFlags = BEAGLE_FLAG_FRAMEWORK_CPU;
	beagleResources.push_back(resource);

	// Optional for plugins: check if the hardware is compatible and only populate
	// list with compatible factories and resources

	beagleFactories.push_back(new beagle::cpu::BeagleCPU4StateSSEImplFactory<double>());
// 	beagleFactories.push_back(new beagle::cpu::BeagleCPU4StateSSEImplFactory<float>()); // TODO Not yet written
	beagleFactories.push_back(new beagle::cpu::BeagleCPUSSEImplFactory<double>()); // TODO In process of writing (disabled until it works for all input)
// 	beagleFactories.push_back(new beagle::cpu::BeagleCPUSSEImplFactory<float>()); // TODO Not yet written
}

}	// namespace cpu
}	// namespace beagle


extern "C" {

#ifdef _WIN32
bool check_sse2(){
#ifdef _WIN64
    // TODO: win64 sse check
    return 1;
#else
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
#endif
}

#endif

#ifdef __GNUC__
#if !defined(DLS_MACOS) // For non-Mac OS X GNU C
bool check_sse2()
{
#ifdef HAVE_CPUID_H
  unsigned int eax, ebx, ecx, edx;
  unsigned int ext, sig;

  ext = 0;
  __get_cpuid_max( ext, &sig );
//  printf( "ext=0x%x sig=0x%x\n", ext, sig );

  if (!__get_cpuid (1, &eax, &ebx, &ecx, &edx)) {
//    printf( "__get_cpuid returned 0\n" );
    return false;
  }

  /* Run SSE2 test only if host has SSE2 support.  */
  if (edx & bit_SSE2)
	return true;
  return false;
#elif defined(__aarch64__)
  return false;
#else // HAVE_CPUID.H
	// Determine if cpuid supported:
    unsigned int res;
    __asm__("mov %%ecx, %%eax;"
            "xor $200000, %%eax;"
            "xor %%ecx, %%eax;"
            "je no;"
            "mov $1, %%eax;"
            "jmp end;"
            "no: mov $0, %%eax;"
            "end:;"
            : "=a" (res)
            :
            : "cc");
    if (res == 0) {
    	return false; // cpuid is not supported
    }
    // Determine if SSE2 supported, PIC compliant version
    unsigned int opcode = 0x00000001;
    unsigned int result[4];
    __asm__(
#ifdef __i386__
    		"pushl %%ebx;"
#else
    		"pushq %%rbx;"
#endif
            "cpuid;"
            "movl %%ebx, %1;"
#ifdef __i386__
            "popl %%ebx;"
#else
    		"popq %%rbx;"
#endif
            : "=a" (result[0]), // EAX register -> result[0]
              "=r" (result[1]), // EBX register -> result[1]
              "=c" (result[2]), // ECX register -> result[2]
              "=d" (result[3])  // EDX register -> result[3]
            : "0" (opcode)
            : "cc");
    return result[3] & 0x04000000;
#endif // HAVE_CPUID.H
}
#else
#if defined(__aarch64__)
bool check_sse2() { return 1; }
#else
bool check_sse2(){
     int op = 0x00000001, eax, ebx, ecx, edx;
      __asm__("cpuid"
        : "=a" (eax), "=b" (ebx), "=c" (ecx), "=d" (edx)
        : "a" (op));
	return edx & 0x04000000;
}
#endif
#endif
#endif


void* plugin_init(void){
	if(!check_sse2()){
		return NULL;	// no SSE no plugin?!
	}
	return new beagle::cpu::BeagleCPUSSEPlugin();
}
}

