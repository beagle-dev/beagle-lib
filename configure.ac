dnl Process this file with autoconf to produce a configure script.
AC_PREREQ([2.59])
AC_INIT(libhmsbeagle, 1.0.0, beagle-dev@googlegroups.com)

: ${CXXFLAGS=" -O3"}

dnl -----------------------------------------------
dnl Package name and version number (user defined)
dnl -----------------------------------------------

GENERIC_LIBRARY_NAME=libhmsbeagle

#API version
GENERIC_API_VERSION=1
AC_SUBST(GENERIC_API_VERSION)

#revision version
GENERIC_REVISION_VERSION=1

#release versioning
GENERIC_MAJOR_VERSION=$GENERIC_API_VERSION
GENERIC_MINOR_VERSION=$GENERIC_REVISION_VERSION
GENERIC_MICRO_VERSION=0

#shared library versioning
GENERIC_LIBRARY_VERSION=$GENERIC_API_VERSION:$GENERIC_REVISION_VERSION:0
#
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
AC_DISABLE_STATIC
AM_DISABLE_STATIC
AC_PROG_LIBTOOL
AM_PROG_LIBTOOL

AM_CONFIG_HEADER(libhmsbeagle/config.h)

# needed to support old automake versions
AC_SUBST(abs_top_builddir)
AC_SUBST(abs_top_srcdir)

CHECK_LIB_PATH="\$(top_builddir)/libhmsbeagle/CPU/.libs:\$(top_builddir)/libhmsbeagle/GPU/.libs"
AC_SUBST(CHECK_LIB_PATH)

# ------------------------------------------------------------------------------
# Check for libtool development libraries for plugin loading
# ------------------------------------------------------------------------------
AC_SEARCH_LIBS(lt_dlinit,ltdl,[AC_DEFINE(HAVE_LIBLTDL,"1","Defined if the libtool dev libs are present")])
if test "x$ac_cv_search_lt_dlinit" == "xno"
then
   AC_SEARCH_LIBS(dlopen,dl)
fi

# ------------------------------------------------------------------------------
# Setup OPENMP 
# ------------------------------------------------------------------------------
dnl Check for OpenMP
if test x$GCC = xyes
then
	AX_GCC_VERSION()
fi

AC_ARG_ENABLE([openmp],
   [AS_HELP_STRING([--enable-openmp],[enable automatic building of openmp version])])

# if we're on gcc, then make sure the version is at least 4.2.0 because apple ships a 
# 4.1.x that has broken openmp support
AS_IF([test "$enable_openmp" = "yes"], [
	if test x$GCC = xyes 
	then 
		AX_COMPARE_VERSION([$GCC_VERSION], [ge], [4.2.0], [ 
			AX_OPENMP([AM_CONDITIONAL(HAVE_OPENMP,true)],
			[AM_CONDITIONAL(HAVE_OPENMP,false)])
		],[])
	else
		AX_OPENMP([AM_CONDITIONAL(HAVE_OPENMP,true)],
		[AM_CONDITIONAL(HAVE_OPENMP,false)])
	fi
],[
AM_CONDITIONAL(HAVE_OPENMP, test true = false)
])
AC_SUBST(OPENMP_CFLAGS)
dnl OpenMP checker only defines for C when compiling both C and C++
OPENMP_CXXFLAGS=$OPENMP_CFLAGS
AC_SUBST(OPENMP_CXXFLAGS)

# ------------------------------------------------------------------------------
# Setup OpenCL
# ------------------------------------------------------------------------------
#AX_OPENCL([C++])

AC_ARG_ENABLE(opencl,
    AC_HELP_STRING([--enable-opencl],[build OpenCL implementation @<:@default=no@:>@]), , [enable_opencl=no])

OPENCL_CFLAGS=
OPENCL_LIBS=

if test "x$enable_opencl" != "xno"
then
   OPENCL_CFLAGS+=" -DOPENCL -framework OpenCL"
   OPENCL_LIBS+=" -framework OpenCL"
fi

AM_CONDITIONAL(BUILDOPENCL, test ! x$enable_opencl = xno)

# ------------------------------------------------------------------------------
# Setup CUDA paths
# ------------------------------------------------------------------------------
AC_ARG_WITH([cuda],
   [AS_HELP_STRING([--with-cuda=PATH],[prefix where CUDA is installed @<:@default=auto@:>@])],
   [],
   [with_cuda=auto])

NVCC=no
CUDA_CFLAGS=
CUDA_LIBS=

if test "x$with_cuda" != "xno"
then

   CUDA_CFLAGS+=" -DCUDA"
   CUDA_LIBS+=" -lcuda"

   if test "x$with_cuda" != "xauto"
   then
      CUDAPATH="$with_cuda"
      CUDA_CFLAGS+=" -I$with_cuda/include"
      CUDA_LIBS+=" -L$with_cuda/lib"
   else
      AC_CHECK_FILE(/usr/local/cuda/,[CUDAPATH="/usr/local/cuda"],[])
      AC_CHECK_FILE(/usr/local/cuda/include,[CUDA_CFLAGS+=" -I/usr/local/cuda/include"],[])
      AC_CHECK_FILE(/usr/local/cuda/lib,[CUDA_LIBS+=" -L/usr/local/cuda/lib"],[])
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
AC_SUBST(NVCC)

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

# ------------------------------------------------------------------------------
# Setup SSE
# ------------------------------------------------------------------------------


AC_ARG_ENABLE(sse,
	AC_HELP_STRING([--enable-sse],[build with sse implementation enabled @<:@default=yes@:>@]), , [enable_sse=yes])

if test  "$enable_sse" = yes; then
	AX_EXT
fi


# ------------------------------------------------------------------------------
# Setup native cpu architecture optimization flag
# ------------------------------------------------------------------------------

AC_ARG_ENABLE(march_native,
    AC_HELP_STRING([--enable-march-native],[build with native architecture optimization @<:@default=yes@:>@]), , [enable_march_native=yes])

if test  "$enable_march_native" = yes; then
    if test x$GCC = xyes 
    then 
        AX_COMPARE_VERSION([$GCC_VERSION], [ge], [4.2.3], [AM_CXXFLAGS="$AM_CXXFLAGS -march=native"],[])
    fi
fi

# ------------------------------------------------------------------------------
# Setup Mac multi-architecture flags
#
# TODO: Improve automatic detection of MacOS10.x.sdk directory
# ------------------------------------------------------------------------------
case $host_os in
*darwin*)
  AM_CXXFLAGS="$AM_CXXFLAGS -DDLS_MACOS"
  
  if( test ! x$NVCC = xno )
  then
     NVCCFLAGS+=" -m64"
     AM_CXXFLAGS="$AM_CXXFLAGS -m64"
     NVCCFLAGS+=" -D_POSIX_C_SOURCE"
  fi

  AC_ARG_ENABLE(osx_leopard,
      AC_HELP_STRING([--enable-osx-leopard],[build with OS X v10.5 Leopard backwards compatibility  @<:@default=no@:>@]), , [enable-osx-leopard=yes])
  if test  "$enable_osx_leopard" = yes; then
      AM_CXXFLAGS="$AM_CXXFLAGS -isysroot /Developer/SDKs/MacOSX10.5.sdk -mmacosx-version-min=10.5"
  fi
esac

# ------------------------------------------------------------------------------
# Setup Cygwin
# ------------------------------------------------------------------------------
case $host_os in
  *cygwin*)
  AM_CXXFLAGS="$AM_CXXFLAGS -D__int64=\"long long\""
  LDFLAGS+=" -mno-cygwin -Wl,-add-stdcall-alias -shared -no-undefined"
esac

# ------------------------------------------------------------------------------
# Setup Java
# ------------------------------------------------------------------------------
AC_ARG_WITH([jdk], [AS_HELP_STRING([--with-jdk=PATH], [prefix of a JDK installation for compiling the JNI wrapper])], [JAVAPREFIX=$withval/bin], [])

jnilibext=no

if test "x$with_jdk" != "xno"
then
    AC_PROG_JAVAC
    if test "x$JAVAC" != "x"
    then  
        AC_JNI_INCLUDE_DIR
        for JNI_INCLUDE_DIR in $JNI_INCLUDE_DIRS
        do
           CPPFLAGS="$CPPFLAGS -I$JNI_INCLUDE_DIR"
        done

        dnl Older versions of OS X require .jnilib extension for java libs
        case $host_os in 
        *darwin*) 
           JNI_EXTRA_LDFLAGS="-shrext .jnilib"
        esac
    else
        with_jdk=no
    fi
fi

AM_CONDITIONAL(BUILDJNI, test ! x$with_jdk = xno)

# ------------------------------------------------------------------------------
# Distribute common variables
# ------------------------------------------------------------------------------

AC_SUBST(NVCCFLAGS)
AC_SUBST(CUDA_CFLAGS)
AC_SUBST(CUDA_LIBS)
AC_SUBST(OPENCL_CFLAGS)
AC_SUBST(OPENCL_LIBS)
AC_SUBST(JNI_EXTRA_LDFLAGS)
AC_SUBST(AM_CXXFLAGS)
AC_SUBST(LDFLAGS)
AC_SUBST(LIBS)

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
AC_CONFIG_FILES([hmsbeagle-1.pc])
AC_CONFIG_FILES([libhmsbeagle/Makefile])
AC_CONFIG_FILES([libhmsbeagle/GPU/Makefile])
AC_CONFIG_FILES([libhmsbeagle/GPU/kernels/Makefile])
AC_CONFIG_FILES([libhmsbeagle/CPU/Makefile])
AC_CONFIG_FILES([libhmsbeagle/plugin/Makefile])
AC_CONFIG_FILES([examples/Makefile])
AC_CONFIG_FILES([examples/tinytest/Makefile])
AC_CONFIG_FILES([examples/complextest/Makefile])
AC_CONFIG_FILES([examples/oddstatetest/Makefile])
AC_CONFIG_FILES([examples/fourtaxon/Makefile])
AC_CONFIG_FILES([examples/genomictest/Makefile])
AC_CONFIG_FILES([examples/matrixtest/Makefile])
AC_OUTPUT

# ------------------------------------------------------------------------------
# warn the user about missing functionality
# ------------------------------------------------------------------------------
if( test x$NVCC = xno ) then
	AC_MSG_WARN([NVIDIA CUDA nvcc compiler not found.  CUDA support will not be built.  Check the path to CUDA and specify --with-cuda=/path/to/cuda])
fi

if( test x$with_jdk = xno ) then
	AC_MSG_WARN([JDK installation not found.  JNI wrapper will not be built.  Check the path to JDK and specify --with-jdk=/path/to/jdk].  If using Mac OS X also try installing Java for Mac OS X 1.6 Update 3 Developer Package)
fi
