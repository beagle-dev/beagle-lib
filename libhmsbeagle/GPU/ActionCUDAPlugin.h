/**
 * libhmsbeagle plugin system
 * @author Xiang Ji
 * @author Benjamin Redelings
 * @author Marc Suchard
 */

#ifndef __BEAGLE_ACTIONCUDAPLUGIN_H__
#define __BEAGLE_ACTIONCUDAPLUGIN_H__

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include "libhmsbeagle/platform.h"
#include "libhmsbeagle/plugin/Plugin.h"



namespace beagle {
    namespace gpu {

        class BEAGLE_DLLEXPORT ActionCUDAPlugin : public beagle::plugin::Plugin
        {
        public:
            ActionCUDAPlugin();
            ~ActionCUDAPlugin();
        private:
            ActionCUDAPlugin( const ActionCUDAPlugin& cp );	// disallow copy by defining this private
        };

    } // namespace gpu
} // namespace beagle

extern "C" {
    BEAGLE_DLLEXPORT void* plugin_init(void);
}





















#endif //BEAGLE_ACTIONCUDAPLUGIN_H
