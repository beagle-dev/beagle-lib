/**
 * libhmsbeagle plugin system
 * @author Aaron E. Darling
 * Based on code found in "Dynamic Plugins for C++" by Arthur J. Musgrove
 * and published in Dr. Dobbs Journal, July 1, 2004.
 */

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif
#include "libhmsbeagle/platform.h"

#include "libhmsbeagle/plugin/WinSharedLibrary.h"
#include <string>

namespace beagle {
namespace plugin {

SharedLibrary* SharedLibrary::openSharedLibrary(const char* name)
    throw (SharedLibraryException)
{
    return new WinSharedLibrary(name);
}

}
}
