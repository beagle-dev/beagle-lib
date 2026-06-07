/**
 * libhmsbeagle plugin system — TinyGPU backend
 * @author Marc Suchard
 */

#ifndef __BEAGLE_TINYGPU_PLUGIN_H__
#define __BEAGLE_TINYGPU_PLUGIN_H__

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include "libhmsbeagle/platform.h"
#include "libhmsbeagle/plugin/Plugin.h"

namespace beagle {
namespace gpu {

class BEAGLE_DLLEXPORT TinyGPUPlugin : public beagle::plugin::Plugin
{
public:
    TinyGPUPlugin();
    ~TinyGPUPlugin();
private:
    TinyGPUPlugin(const TinyGPUPlugin&);  // disallow copy
};

} // namespace gpu
} // namespace beagle

extern "C" {
    BEAGLE_DLLEXPORT void* plugin_init(void);
}

#endif  // __BEAGLE_TINYGPU_PLUGIN_H__
