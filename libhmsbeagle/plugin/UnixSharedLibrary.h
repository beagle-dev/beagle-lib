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

// only use the standard unix library interface if we don't have libtool libraries
#ifndef HAVE_LIBLTDL

#include <dlfcn.h>
#include <string>

namespace beagle {
namespace plugin {

class UnixSharedLibrary : public SharedLibrary
{
  public:
    UnixSharedLibrary(const char* name);
    ~UnixSharedLibrary();

    void* findSymbol(const char* name);

  private:
    void* m_handle;
};

UnixSharedLibrary::UnixSharedLibrary(const char* name)
    : m_handle(0)
{
    std::string libname = "lib";
    libname += name;
#ifdef DLS_MACOS
    libname += ".dylib";
#else
    libname += ".so";
#endif
    m_handle = dlopen(libname.c_str(),RTLD_NOW|RTLD_GLOBAL);
    if (m_handle == 0)
    {
    const char* s = dlerror();
    throw SharedLibraryException(s?s:"Exact Error Not Reported");
    }
}
UnixSharedLibrary::~UnixSharedLibrary() { dlclose(m_handle); }

void* UnixSharedLibrary::findSymbol(const char* name)
{
    void* sym = dlsym(m_handle,name);
    if (sym == 0)
    throw SharedLibraryException("Symbol Not Found");
    else
    return sym;
}

} // namespace plugin
} // namespace beagle

#endif	// HAVE_LIBLTDL

#endif	// __UNIXSHAREDLIBRARY_H__

