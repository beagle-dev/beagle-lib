check_PROGRAMS = matrixtest
matrixtest_SOURCES = matrixtest.cpp
matrixtest_LDADD = $(top_builddir)/$(GENERIC_LIBRARY_NAME)/libhmsbeagle.la

TESTS = matrixtest
TESTS_ENVIRONMENT = LD_LIBRARY_PATH+=@CHECK_LIB_PATH@
AM_CPPFLAGS = -I$(top_builddir) -I$(top_srcdir)

