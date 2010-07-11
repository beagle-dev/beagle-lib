/**
 * libhmsbeagle plugin system
 * @author Aaron E. Darling
 * Based on code found in "Dynamic Plugins for C++" by Arthur J. Musgrove
 * and published in Dr. Dobbs Journal, July 1, 2004.
 */

#ifndef __PLUGIN_H__
#define __PLUGIN_H__

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include "libhmsbeagle/platform.h"
#include "libhmsbeagle/BeagleImpl.h"
#include "libhmsbeagle/plugin/SharedLibrary.h"
#include <memory>
#include <string>
#include <map>
#include <list>

namespace beagle {
namespace plugin {

/**
 * All libhmsbeagle plugins derive from this class
 * During initialization, a plugin must populate the beagleImpls list
 */
class BEAGLE_DLLEXPORT Plugin
{
  public:
	Plugin() {}

    Plugin(const char* plugin_name, const char* plugin_type)
    : m_plugin_name(plugin_name), m_plugin_type(plugin_type) {}

    virtual std::string pluginName() const{ return m_plugin_name; }
    virtual std::string pluginType() const{ return m_plugin_type; }

    virtual const std::list<beagle::BeagleImplFactory*>& getBeagleFactories() const{ return beagleFactories; }
    virtual const std::list<BeagleResource>& getBeagleResources() const{ return beagleResources; }

protected:
    std::list<beagle::BeagleImplFactory*> beagleFactories;
    std::list<BeagleResource> beagleResources;
    std::string m_plugin_name;
    std::string m_plugin_type;
};

typedef Plugin* (*plugin_init_func)(void);

class BEAGLE_DLLEXPORT PluginManager
{
  public:
      static PluginManager& instance();

      Plugin* findPlugin(const char* name)
      throw (SharedLibraryException);

    private:
        struct PluginInfo {
        SharedLibrary* m_library;
        std::string m_library_name;
        Plugin* m_plugin;

        ~PluginInfo() { delete m_plugin; delete m_library; }
        PluginInfo() : m_library(0), m_plugin(0) {}
    };
    PluginManager() {}
    static PluginManager* ms_instance;
    std::map<std::string,PluginInfo* > m_plugin_map;
    // ...
};

} // namespace plugin
} // namespace beagle

#endif	// __PLUGIN_H__
