/*
 *  BeagleRemotePlugin.h
 *  BEAGLE
 *
 * The plugin that makes the resources of beagle-worker processes on other machines appear in this process's
 * resource list. The workers are named in the environment variable BEAGLE_REMOTE_HOSTS
 * (host[:port],host[:port],...); without it the plugin contributes nothing.
 */

#ifndef __BEAGLE_REMOTE_PLUGIN_H__
#define __BEAGLE_REMOTE_PLUGIN_H__

#include <vector>

#include "libhmsbeagle/platform.h"
#include "libhmsbeagle/plugin/Plugin.h"
#include "libhmsbeagle/Remote/BeagleRemoteImpl.h"

namespace beagle {
namespace remote {

class BEAGLE_DLLEXPORT BeagleRemotePlugin : public beagle::plugin::Plugin
{
public:
    BeagleRemotePlugin();
private:
    BeagleRemotePlugin(const BeagleRemotePlugin& cp); // disallow copy by defining this private

    // owns the strings that the BeagleResource entries point to
    std::vector<RemoteResource> resources_;
};

} // namespace remote
} // namespace beagle

extern "C" {
    BEAGLE_DLLEXPORT void* plugin_init(void);
}

#endif // __BEAGLE_REMOTE_PLUGIN_H__
