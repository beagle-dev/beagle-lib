/**
 * libhmsbeagle plugin system
 * ROCm/HIP backend plugin for AMD GPUs.
 */

#ifndef __BEAGLE_ROCM_PLUGIN_H__
#define __BEAGLE_ROCM_PLUGIN_H__

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include "libhmsbeagle/platform.h"
#include "libhmsbeagle/plugin/Plugin.h"

namespace beagle {
namespace gpu {

class BEAGLE_DLLEXPORT ROCmPlugin : public beagle::plugin::Plugin
{
public:
    ROCmPlugin();
    ~ROCmPlugin();
private:
    ROCmPlugin(const ROCmPlugin& cp);
};

} // namespace gpu
} // namespace beagle

extern "C" {
    BEAGLE_DLLEXPORT void* plugin_init(void);
}

#endif // __BEAGLE_ROCM_PLUGIN_H__
