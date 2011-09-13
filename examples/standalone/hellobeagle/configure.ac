# this is a very basic configuration example
AC_INIT([HelloBeagle C++], [0.1], [bug-report@hello.beagle-gpu.com],
             [hellobeagle], [http://hello.beagle-gpu.com/])
AC_PREREQ([2.59])
AC_CONFIG_AUX_DIR(.config)
AM_INIT_AUTOMAKE([1.10 -Wall no-define])
AC_CONFIG_HEADERS([config.h])
AC_PROG_CXX

PKG_CHECK_MODULES(DEPS, hmsbeagle-1 >= 1.0.0)
AC_SUBST(DEPS_CFLAGS)
AC_SUBST(DEPS_LIBS)

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
