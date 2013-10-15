/**
 * libhmsbeagle plugin system
 * @author Aaron E. Darling
 * Based on code found in "Dynamic Plugins for C++" by Arthur J. Musgrove
 * and published in Dr. Dobbs Journal, July 1, 2004.
 */

#ifndef __BEAGLE_OPENCL_PLUGIN_H__
#define __BEAGLE_OPENCL_PLUGIN_H__

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include "libhmsbeagle/platform.h"
#include "libhmsbeagle/plugin/Plugin.h"

namespace beagle {
namespace gpu {

class BEAGLE_DLLEXPORT OpenCLPlugin : public beagle::plugin::Plugin
{
public:
	OpenCLPlugin();
	~OpenCLPlugin();
private:
	OpenCLPlugin( const OpenCLPlugin& cp );	// disallow copy by defining this private
};

} // namespace gpu
} // namespace beagle

extern "C" {
	BEAGLE_DLLEXPORT void* plugin_init(void);
}

#endif	// __BEAGLE_OPENCL_PLUGIN_H__


