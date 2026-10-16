/**
 * libhmsbeagle plugin system
 * @author Aaron E. Darling
 * Based on code found in "Dynamic Plugins for C++" by Arthur J. Musgrove
 * and published in Dr. Dobbs Journal, July 1, 2004.
 */

#ifndef __BEAGLE_CPU_HIERARCHICAL_PLUGIN_H__
#define __BEAGLE_CPU_HIERARCHICAL_PLUGIN_H__

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include "libhmsbeagle/platform.h"
#include "libhmsbeagle/plugin/Plugin.h"

namespace beagle {
namespace cpu {

class BEAGLE_DLLEXPORT BeagleCPUHierarchicalPlugin : public beagle::plugin::Plugin
{
public:
	BeagleCPUHierarchicalPlugin();
private:
	BeagleCPUHierarchicalPlugin( const BeagleCPUHierarchicalPlugin& cp );
};

}
}

extern "C" {
	BEAGLE_DLLEXPORT void* plugin_init(void);
}

#endif
