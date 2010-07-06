/**
 * libhmsbeagle plugin system
 * @author Aaron E. Darling
 * Based on code found in "Dynamic Plugins for C++" by Arthur J. Musgrove
 * and published in Dr. Dobbs Journal, July 1, 2004.
 */

#include "Plugin.h"

namespace beagle {

class BeaglePlugin : public Plugin
{
  public:
    BeaglePlugin(const char* plugin_name)
    : Plugin(plugin_name,"Text") { }
    virtual const char* getSayHelloString() const = 0;
};

} // namespace beagle


