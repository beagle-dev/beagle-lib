/**
 * libhmsbeagle plugin system
 * @author Aaron E. Darling
 * Based on code found in "Dynamic Plugins for C++" by Arthur J. Musgrove
 * and published in Dr. Dobbs Journal, July 1, 2004.
 */

#ifndef __WINSHAREDLIBRARY_H__
#define __WINSHAREDLIBRARY_H__

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include "libhmsbeagle/plugin/SharedLibrary.h"
#include <windows.h>
#include <string>
#include <iostream>

namespace beagle {
namespace plugin {

using namespace std;

class WinSharedLibrary : public SharedLibrary
{
  public:
    WinSharedLibrary(const char* name)
    throw (SharedLibraryException);
    ~WinSharedLibrary();

    void* findSymbol(const char* name)
    throw (SharedLibraryException);

  private:
    HINSTANCE m_handle;
};
WinSharedLibrary::WinSharedLibrary(const char* name)
    throw (SharedLibraryException)
    : m_handle(0)
{
	std::string libname = name;
#ifdef _WIN64
#ifdef _DEBUG
	libname += "64D";
#else
	libname += "64";
#endif
#else
#ifdef _DEBUG
	libname += "32D";
#else
	libname += "32";
#endif
#endif
	UINT emode = SetErrorMode(SEM_FAILCRITICALERRORS);
    m_handle = LoadLibrary(libname.c_str());
	SetErrorMode(emode);
    if (m_handle == 0)
    {
    char buffer[255];
    strcpy(buffer,"Open Library Failure");
    FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM,0,GetLastError(),
        0, buffer,sizeof(buffer),0);
    throw SharedLibraryException(buffer);
    }
}
WinSharedLibrary::~WinSharedLibrary()
{
    if (!FreeLibrary(m_handle))
    {
    char buffer[255];
    // format buffer as above
    cerr << buffer << endl;
    }
}
void* WinSharedLibrary::findSymbol(const char* name)
    throw (SharedLibraryException)
{
    void* sym = GetProcAddress(m_handle,name);
    if (sym == 0)
    {
    char buffer[255];
    // format buffer as above
    throw SharedLibraryException(buffer);
    }
    else
   return sym;
}

} // namespace plugin
} // namespace beagle


#endif	// __WINSHAREDLIBRARY_H__

