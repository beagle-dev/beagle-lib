/**
 * libhmsbeagle plugin system — TinyGPUHybrid backend
 * @author Marc Suchard
 */

#ifndef __BEAGLE_TINYGPU_HYBRID_PLUGIN_H__
#define __BEAGLE_TINYGPU_HYBRID_PLUGIN_H__

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include "libhmsbeagle/platform.h"
#include "libhmsbeagle/plugin/Plugin.h"

namespace beagle {
namespace gpu {

class BEAGLE_DLLEXPORT TinyGPUHybridPlugin : public beagle::plugin::Plugin
{
public:
    TinyGPUHybridPlugin();
    ~TinyGPUHybridPlugin();
private:
    TinyGPUHybridPlugin(const TinyGPUHybridPlugin&);  // disallow copy
};

} // namespace gpu
} // namespace beagle

extern "C" {
    BEAGLE_DLLEXPORT void* plugin_init(void);
}

#endif  // __BEAGLE_TINYGPU_HYBRID_PLUGIN_H__
