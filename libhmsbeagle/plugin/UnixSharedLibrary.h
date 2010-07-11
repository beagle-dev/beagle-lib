/**
 * libhmsbeagle plugin system
 * @author Aaron E. Darling
 * Based on code found in "Dynamic Plugins for C++" by Arthur J. Musgrove
 * and published in Dr. Dobbs Journal, July 1, 2004.
 */

#ifndef __UNIXSHAREDLIBRARY_H__
#define __UNIXSHAREDLIBRARY_H__

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include "libhmsbeagle/plugin/SharedLibrary.h"
#include <string>
#include <ltdl.h>

namespace beagle {
namespace plugin {

class UnixSharedLibrary : public SharedLibrary
{
  public:
    UnixSharedLibrary(const char* name);
    ~UnixSharedLibrary();

    void* findSymbol(const char* name);

  private:
    lt_dlhandle m_handle;
};

UnixSharedLibrary::UnixSharedLibrary(const char* name)
    : m_handle(0)
{
	std::string libname = "lib";
	libname += name;
    lt_dlinit();
    m_handle = lt_dlopenext(libname.c_str());
    if (m_handle == 0)
    {
    const char* s = lt_dlerror();
    throw SharedLibraryException(s?s:"Exact Error Not Reported");
    }
}
UnixSharedLibrary::~UnixSharedLibrary() {
	lt_dlclose(m_handle); 
	lt_dlexit();
}

void* UnixSharedLibrary::findSymbol(const char* name)
{
    void* sym = lt_dlsym(m_handle,name);
    if (sym == 0)
    throw SharedLibraryException("Symbol Not Found");
    else
    return sym;
}

} // namespace plugin
} // namespace beagle

#endif	// __UNIXSHAREDLIBRARY_H__

