dnl Process this file with autoconf to produce a configure script.
AC_PREREQ([2.59])
AC_INIT(libhmsbeagle, 1.0.0, beagle-dev@googlegroups.com)

: ${CXXFLAGS=" -O2"}

dnl -----------------------------------------------
dnl Package name and version number (user defined)
dnl -----------------------------------------------

GENERIC_LIBRARY_NAME=libhmsbeagle

#release versioning
GENERIC_MAJOR_VERSION=1
GENERIC_MINOR_VERSION=0
GENERIC_MICRO_VERSION=0

#API version (often = GENERIC_MAJOR_VERSION.GENERIC_MINOR_VERSION)
GENERIC_API_VERSION=1.0
AC_SUBST(GENERIC_API_VERSION)

#shared library versioning
GENERIC_LIBRARY_VERSION=1:0:0
#                       | | |
#                +------+ | +---+
#                |        |     |
#             current:revision:age
#                |        |     |
#                |        |     +- increment if interfaces have been added
#                |        |        set to zero if interfaces have been removed
#                                  or changed
#                |        +- increment if source code has changed
#                |           set to zero if current is incremented
#                +- increment if interfaces have been added, removed or changed

dnl --------------------------------
dnl Package name and version number
dnl --------------------------------

AC_SUBST(GENERIC_LIBRARY_VERSION)

PACKAGE=$GENERIC_LIBRARY_NAME
AC_SUBST(GENERIC_LIBRARY_NAME)

GENERIC_VERSION=$GENERIC_MAJOR_VERSION.$GENERIC_MINOR_VERSION.$GENERIC_MICRO_VERSION
GENERIC_RELEASE=$GENERIC_MAJOR_VERSION.$GENERIC_MINOR_VERSION
AC_SUBST(GENERIC_RELEASE)
AC_SUBST(GENERIC_VERSION)

VERSION=$GENERIC_VERSION


AC_CONFIG_AUX_DIR(.config)
AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_SRCDIR(libhmsbeagle/beagle.cpp)

AM_INIT_AUTOMAKE($PACKAGE, $VERSION, no-define)

AC_PROG_CC
AC_PROG_CXX
AC_PROG_INSTALL
AC_PROG_LN_S
AC_PROG_LIBTOOL
AM_PROG_LIBTOOL

AM_CONFIG_HEADER(libhmsbeagle/config.h)

# needed to support old automake versions
AC_SUBST(abs_top_builddir)
AC_SUBST(abs_top_srcdir)

# ------------------------------------------------------------------------------
# Setup OPENMP 
# ------------------------------------------------------------------------------
dnl Check for OpenMP
AC_ARG_WITH([openmp],
   [AS_HELP_STRING([--with-openmp],[build openmp version @<:@default=yes@:>@])],
   [],
   [with_openmp=no])
if test "x$with_openmp" != "xno"
then
   AX_OPENMP()
fi
AC_SUBST(OPENMP_CFLAGS)
dnl OpenMP checker only defines for C when compiling both C and C++
OPENMP_CXXFLAGS=$OPENMP_CFLAGS
AC_SUBST(OPENMP_CXXFLAGS)


# ------------------------------------------------------------------------------
# Setup OPENCL paths
# ------------------------------------------------------------------------------
AC_ARG_WITH([opencl],
   [AS_HELP_STRING([--with-opencl],[build opencl version @<:@default=no@:>@])],
   [],
   [with_opencl=no])

OPENCL_CFLAGS=
OPENCL_LIBS=

if test "x$with_opencl" != "xno"
then
   OPENCL_CFLAGS="-framework OpenCL \"-DBEAGLE_OPENCL_BUILD_OPTIONS=\\\"-D STATE_COUNT=4\\\"\""
   OPENCL_LIBS="-framework OpenCL"
   OPENCLPATH=""
fi

AM_CONDITIONAL(BUILDOPENCL, test ! x$with_opencl = xno)

AC_SUBST(OPENCL_CFLAGS)
AC_SUBST(OPENCL_LIBS)

if( test ! x$with_opencl = xno ) 
then
	AC_DEFINE([OPENCL],[1],[Defined if OPENCL should be used])
fi



# ------------------------------------------------------------------------------
# Setup CUDA paths
# ------------------------------------------------------------------------------
AC_ARG_WITH([cuda],
   [AS_HELP_STRING([--with-cuda=PATH],[prefix where cuda is installed @<:@default=auto@:>@])],
   [],
   [with_cuda=auto])

NVCC=no
CUDA_CFLAGS=
CUDA_LIBS=

if test "x$with_cuda" != "xno"
then
   if test "x$with_cuda" != "xauto"
   then
      CUDA_CFLAGS="-I$with_cuda/include"
      CUDA_LIBS="-L$with_cuda/lib -lcuda"
      CUDAPATH="$with_cuda"
   else
      AC_CHECK_FILE(/usr/local/cuda/,[CUDAPATH="/usr/local/cuda"],[])
      AC_CHECK_FILE(/usr/local/cuda/include,[CUDA_CFLAGS="-I/usr/local/cuda/include"],[])
      AC_CHECK_FILE(/usr/local/cuda/lib,[CUDA_LIBS="-L/usr/local/cuda/lib -lcuda"],[])
   fi



AC_ARG_ENABLE([emu],
    AC_HELP_STRING([--enable-emu],[turn on device emulation for CUDA]),
   [case "${enableval}" in
       yes) EMULATION=true;;
       no)  EMULATION=false;;
       *) AC_MSG_ERROR([bad value ${enableval} for --enable-emu]);;
   esac],
   [EMULATION=false]
)

fi

if test "x$CUDA_CFLAGS" != "x"
then
   AC_PATH_PROG([NVCC],[nvcc],[no],[$PATH:$CUDAPATH/bin])
fi

AM_CONDITIONAL(BUILDCUDA, test ! x$NVCC = xno)
AC_SUBST(CUDA_CFLAGS)
AC_SUBST(CUDA_LIBS)
AC_SUBST(NVCC)

if( test ! x$NVCC = xno ) 
then
	AC_DEFINE([CUDA],[1],[Defined if CUDA should be used])
fi

# ------------------------------------------------------------------------------
# Setup nvcc flags
# ------------------------------------------------------------------------------
if test x$DEBUG = xtrue
then
   NVCCFLAGS="-g"
else
   NVCCFLAGS="-O3"
fi
if test x$EMULATION = xtrue
then
   NVCCFLAGS+=" -deviceemu"
fi
AC_SUBST(NVCCFLAGS)

# ------------------------------------------------------------------------------
# Setup SSE
# ------------------------------------------------------------------------------

AC_ARG_ENABLE(sse,
	AC_HELP_STRING([--enable-sse],[build with sse implementation enabled @<:@default=yes@:>@]), , [enable_sse=yes])

if test  "$enable_sse" = yes; then
	CXXFLAGS+=" -msse2 -DENABLE_SSE"
fi

# ------------------------------------------------------------------------------
# Setup Mac multi-architecture flags
#
# TODO: Improve automatic detection of MacOS10.x.sdk directory
# ------------------------------------------------------------------------------
case $host_os in
*darwin*)
  
  CXXFLAGS+=" -DDLS_MACOS"
  CFLAGS+=" -DDLS_MACOS"
  
  # CUDA will only work with i386
  if( test ! x$NVCC = xno )
  then
     CXXFLAGS+=" -arch i386"
     CFLAGS+=" -arch i386"
     NVCCFLAGS+=" -D_POSIX_C_SOURCE"
  fi
  # OpenCL requires the 10.6 SDK
  if( test ! x$with_opencl = xno )
  then
    AC_CHECK_FILE(/Developer/SDKs/MacOSX10.6.sdk,
        [LDFLAGS+=" -isysroot /Developer/SDKs/MacOSX10.6.sdk"],
        [AC_MSG_ERROR([OpenCL builds require Mac OS X 10.6 and Xcode])])
  fi

  AC_SUBST(CXXFLAGS)
  AC_SUBST(CFLAGS)
  AC_SUBST(LDFLAGS)
esac

case $host_os in
  *cygwin*)
  CXXFLAGS+=" -D__int64=\"long long\""
  CFLAGS+=" -D__int64=\"long long\""
  LDFLAGS+=" -mno-cygwin -Wl,-add-stdcall-alias -shared -no-undefined"
  AC_SUBST(CXXFLAGS)
  AC_SUBST(CFLAGS)
  AC_SUBST(LDFLAGS)
esac

# ------------------------------------------------------------------------------
# Setup Java
# ------------------------------------------------------------------------------
AC_ARG_WITH([java], [AS_HELP_STRING([--with-java=PATH], [Specify a specific Java installation for compiling BEAGLE])], [JAVA=$withval/bin/java], [])

AC_CHECK_CLASSPATH
AC_PROG_JAVAC
AC_PROG_JAVA
AC_JNI_INCLUDE_DIR
for JNI_INCLUDE_DIR in $JNI_INCLUDE_DIRS
do
   CPPFLAGS="$CPPFLAGS -I$JNI_INCLUDE_DIR"
done

dnl Older versions of OS X require .jnilib extension for java libs
case $host_os in 
*darwin*) 
   JNI_EXTRA_LDFLAGS="-shrext .jnilib" ;; 
esac 
AC_SUBST(JNI_EXTRA_LDFLAGS)


# ------------------------------------------------------------------------------
# Doxygen support 
# ------------------------------------------------------------------------------

DX_HTML_FEATURE(ON)
DX_CHM_FEATURE(OFF)
DX_CHI_FEATURE(OFF)
DX_MAN_FEATURE(ON)
DX_RTF_FEATURE(OFF)
DX_XML_FEATURE(OFF)
DX_PDF_FEATURE(OFF)
DX_PS_FEATURE(OFF)

DX_INIT_DOXYGEN([$PACKAGE_NAME],[beagle.dox],[doc])

# ------------------------------------------------------------------------------
# All done, create Makefiles
# ------------------------------------------------------------------------------
AC_CONFIG_FILES([Makefile])
AC_CONFIG_FILES([hmsbeagle-1.0.pc])
AC_CONFIG_FILES([libhmsbeagle/Makefile])
AC_CONFIG_FILES([libhmsbeagle/GPU/Makefile])
AC_CONFIG_FILES([libhmsbeagle/GPU/kernels/Makefile])
AC_CONFIG_FILES([libhmsbeagle/CPU/Makefile])
AC_CONFIG_FILES([libhmsbeagle/JNI/Makefile])
AC_CONFIG_FILES([examples/Makefile])
AC_CONFIG_FILES([examples/tinytest/Makefile])
AC_CONFIG_FILES([examples/complextest/Makefile])
AC_CONFIG_FILES([examples/fourtaxon/Makefile])
AC_CONFIG_FILES([examples/genomictest/Makefile])
AC_CONFIG_FILES([examples/matrixtest/Makefile])
AC_OUTPUT

# ------------------------------------------------------------------------------
# warn the user about missing functionality
# ------------------------------------------------------------------------------
if( test x$NVCC = xno ) then
	AC_MSG_WARN([Nvidia CUDA nvcc compiler not found.  CUDA support will not be built.  Check the path to CUDA and specify --with-cuda=/path/to/cuda])
fi

