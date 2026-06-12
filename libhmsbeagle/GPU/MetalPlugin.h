/**
 * libhmsbeagle Metal plugin
 * macOS Metal backend for BEAGLE GPU acceleration.
 */

#ifndef __BEAGLE_METAL_PLUGIN_H__
#define __BEAGLE_METAL_PLUGIN_H__

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include "libhmsbeagle/platform.h"
#include "libhmsbeagle/plugin/Plugin.h"

namespace beagle {
namespace gpu {

class BEAGLE_DLLEXPORT MetalPlugin : public beagle::plugin::Plugin
{
public:
    MetalPlugin();
    ~MetalPlugin();
private:
    MetalPlugin(const MetalPlugin& cp);  // disallow copy
};

} // namespace gpu
} // namespace beagle

extern "C" {
    BEAGLE_DLLEXPORT void* plugin_init(void);
}

#endif  // __BEAGLE_METAL_PLUGIN_H__
