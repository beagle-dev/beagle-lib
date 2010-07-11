/**
 * libhmsbeagle plugin system
 * @author Aaron E. Darling
 * Based on code found in "Dynamic Plugins for C++" by Arthur J. Musgrove
 * and published in Dr. Dobbs Journal, July 1, 2004.
 */
// Plugin.cxx

#include "libhmsbeagle/plugin/Plugin.h"
#include <string>
using namespace std;

namespace beagle {
namespace plugin {

PluginManager* PluginManager::ms_instance = 0;

PluginManager& PluginManager::instance()
{
    if (! ms_instance)
    ms_instance = new PluginManager();
    return *ms_instance;
}
Plugin* PluginManager::findPlugin(const char* name)
    throw (SharedLibraryException)
{
    if (m_plugin_map.count(name) > 0)
    return m_plugin_map[name]->m_plugin;

    PluginInfo* pi = new PluginInfo;
    pi->m_library = SharedLibrary::openSharedLibrary(name);
    plugin_init_func pif =
        findSymbol<plugin_init_func>(*pi->m_library,"plugin_init");

    pi->m_plugin = (*pif)();
    if (!pi->m_plugin)
    {
    delete pi;
    throw SharedLibraryException("plugin_init error");
    }
    m_plugin_map[name]=pi;
    return pi->m_plugin;
}

}	// namespace plugin
}	// namespace beagle
