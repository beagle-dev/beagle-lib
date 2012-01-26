/**
 * libhmsbeagle plugin system
 * @author Aaron E. Darling
 * Based on code found in "Dynamic Plugins for C++" by Arthur J. Musgrove
 * and published in Dr. Dobbs Journal, July 1, 2004.
 */

#ifndef __SHAREDLIBRARY_H__
#define __SHAREDLIBRARY_H__

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include <string>

namespace beagle {
namespace plugin {

class SharedLibraryException
{
  public:
    SharedLibraryException(const char* error) : m_error(error) { }
    const char* getError() const {return m_error.c_str();}
  private:
    std::string m_error;
};

class SharedLibrary
{
  public:
    static SharedLibrary* openSharedLibrary(const char* name);
    virtual ~SharedLibrary() {}
    virtual void* findSymbol(const char* name) = 0;

  // ...
};

template<class T>
T findSymbol(SharedLibrary& sl, const char* name)
{
    return (T)sl.findSymbol(name);
}

} // namespace plugin
} // namespace beagle

#endif 	// __SHAREDLIBRARY_H__

