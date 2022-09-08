/**
 * libhmsbeagle plugin system
 * @author Aaron E. Darling
 * @author Xiang Ji
 * Based on code found in "Dynamic Plugins for C++" by Arthur J. Musgrove
 * and published in Dr. Dobbs Journal, July 1, 2004.
 */


#ifndef BEAGLE_BEAGLE_CPU_ACTION_PLUGIN_H
#define BEAGLE_BEAGLE_CPU_ACTION_PLUGIN_H

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include "libhmsbeagle/platform.h"
#include "libhmsbeagle/plugin/Plugin.h"

namespace beagle {
    namespace cpu {

/*
 * An action plugin based on the standard CPU plugin
 * This plugin uses all the same code as the CPU plugin, but should be built with
 * Action enabled
 */
        class BEAGLE_DLLEXPORT BeagleCPUActionPlugin : public beagle::plugin::Plugin
        {
        public:
            BeagleCPUActionPlugin();
        private:
            BeagleCPUActionPlugin( const BeagleCPUActionPlugin& cp );	// disallow copy by defining this private
        };

    } // namespace cpu
} // namespace beagle

extern "C" {
    BEAGLE_DLLEXPORT void* plugin_init(void);
}


#endif //BEAGLE_BEAGLE_CPU_ACTION_PLUGIN_H
