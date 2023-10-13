/*
 *  platform.h
 *  Definitions and compiler support for platform-specific features
 *
 * Copyright 2009 Phylogenetic Likelihood Working Group
 *
 * This file is part of BEAGLE.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 *
 * @author Aaron Darling
 */

#ifndef __beagle_platform__
#define __beagle_platform__

#ifdef _WIN32
// needed to export library symbols
#ifdef _EXPORTING
#define BEAGLE_DLLEXPORT __declspec(dllexport)
#else
#define BEAGLE_DLLEXPORT __declspec(dllimport)
#endif
/*
// automatically include the appropriate beagle library
	#ifdef _WIN64
		#ifdef _DEBUG
		#pragma comment( lib, "libhmsbeagle64d" )
		#else
		#pragma comment( lib, "libhmsbeagle64" )
		#endif
	#else
		#ifdef _DEBUG
		#pragma comment( lib, "libhmsbeagle32d" )
		#else
		#pragma comment( lib, "libhmsbeagle32" )
		#endif
	#endif
*/


#else // not windows
#define BEAGLE_DLLEXPORT
#endif

#ifndef M_LN2 /* Work around for OS X 10.8 and gcc 4.7.1 */
#define M_LN2   0.693147180559945309417232121458176568  /* log_e 2 */
#endif

#endif

