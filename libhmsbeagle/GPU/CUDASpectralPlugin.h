/**
 * libhmsbeagle plugin system
 * @author Aaron E. Darling
 * Based on code found in "Dynamic Plugins for C++" by Arthur J. Musgrove
 * and published in Dr. Dobbs Journal, July 1, 2004.
 */

#ifndef __BEAGLE_CUDA_SPECTRAL_PLUGIN_H__
#define __BEAGLE_CUDA_SPECTRAL_PLUGIN_H__

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include "libhmsbeagle/platform.h"
#include "libhmsbeagle/plugin/Plugin.h"

namespace beagle {
namespace gpu {

class BEAGLE_DLLEXPORT CUDASpectralPlugin : public beagle::plugin::Plugin
{
public:
    CUDASpectralPlugin();
    ~CUDASpectralPlugin();
private:
    CUDASpectralPlugin(const CUDASpectralPlugin& cp);  // disallow copy
};

} // namespace gpu
} // namespace beagle

extern "C" {
    BEAGLE_DLLEXPORT void* plugin_init(void);
}

#endif  // __BEAGLE_CUDA_SPECTRAL_PLUGIN_H__
