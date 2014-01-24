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

#ifdef HAVE_LIBLTDL
#include "libhmsbeagle/plugin/LibtoolSharedLibrary.h"
#else
#include "libhmsbeagle/plugin/UnixSharedLibrary.h"
#endif

#include <string>

namespace beagle {
namespace plugin {

SharedLibrary* SharedLibrary::openSharedLibrary(const char* name)
{
    return new UnixSharedLibrary(name);
}

}
}
