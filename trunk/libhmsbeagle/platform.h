/*
 *  platform.h
 *  Definitions and compiler support for platform-specific features
 *
 * Copyright 2009 Phylogenetic Likelihood Working Group
 *
 * This file is part of BEAGLE.
 *
 * BEAGLE is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 *
 * BEAGLE is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with BEAGLE.  If not, see
 * <http://www.gnu.org/licenses/>.
 * 
 * @author Aaron Darling
 */

#ifndef __beagle_platform__
#define __beagle_platform__

#ifdef _WIN32
// needed to export library symbols
#define BEAGLE_DLLEXPORT __declspec(dllexport)

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

#endif