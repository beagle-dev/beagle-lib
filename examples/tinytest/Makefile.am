check_PROGRAMS = tinytest
tinytest_SOURCES = tinytest.cpp
tinytest_LDADD = $(top_builddir)/$(GENERIC_LIBRARY_NAME)/libhmsbeagle.la @CUDA_LIBS@

TESTS = tinytest
INCLUDES = -I$(top_builddir) -I$(top_srcdir)

EXTRA_DIST=check_lnL_using_BEAST.xml  check_lnL_using_paup.nex
