dnl Process this file with autoconf to produce a configure script.
AC_PREREQ([2.59])
AC_INIT(libbeagle, 0.1.0, msuchard@ucla.edu)
AC_CONFIG_AUX_DIR(config)
AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_SRCDIR(libbeagle/beagle.cpp)

AM_INIT_AUTOMAKE

AC_PROG_CC
AC_PROG_CXX
AC_PROG_INSTALL
AC_PROG_LN_S
AC_PROG_LIBTOOL
AM_PROG_LIBTOOL

AM_CONFIG_HEADER(libbeagle/config.h)


AC_DEFINE([STATE_COUNT],[4],[Define the number of character states in use])


# ------------------------------------------------------------------------------
# Setup CUDA paths
# ------------------------------------------------------------------------------
AC_ARG_WITH([cuda],
   [  --with-cuda=PATH        prefix where cuda is installed [default=auto]])
if test -n "$with_cuda"
then
   CUDA_CFLAGS="-I$with_cuda/include"
   CUDA_LIBS="-L$with_cuda/lib -lcuda -lcudart"
   NVCC="$with_cuda/bin/nvcc"
else
   CUDA_CFLAGS="-I/usr/local/cuda/include"
   CUDA_LIBS="-L/usr/local/cuda/lib -lcuda -lcudart"
   NVCC="nvcc"
fi
AC_SUBST(CUDA_CFLAGS)
AC_SUBST(CUDA_LIBS)
AC_SUBST(NVCC)

AC_PATH_PROG([NVCC],[nvcc],[no],[$PATH:$with_cuda/bin])
AM_CONDITIONAL(BUILDCUDA, test ! x$NVCC = xno)

AC_ARG_ENABLE([emu],
   [  --enable-emu    Turn on device emulation for CUDA],
   [case "${enableval}" in
       yes) EMULATION=true;;
       no)  EMULATION=false;;
       *) AC_MSG_ERROR([bad value ${enableval} for --enable-emu]);;
   esac],
   [EMULATION=false]
)
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
   NVCCFLAGS="-O3 -use_fast_math"
fi
if test x$EMULATION = xtrue
then
   NVCCFLAGS+=" -deviceemu"
fi
AC_SUBST(NVCCFLAGS)



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
*-darwin*) 
   JNI_EXTRA_LDFLAGS="-shrext .jnilib" ;; 
esac 
AC_SUBST(JNI_EXTRA_LDFLAGS)

#
# All done, create Makefiles
#
AC_OUTPUT(Makefile libbeagle/Makefile libbeagle/CUDA/Makefile libbeagle/CPU/Makefile libbeagle/JNI/Makefile)

#
# warn the user about missing functionality
#
if( test x$NVCC = xno ) then
	AC_MSG_WARN([Nvidia CUDA nvcc compiler not found.  CUDA support will not be built.  Check the path to CUDA and specify --with-cuda=/path/to/cuda])
fi

