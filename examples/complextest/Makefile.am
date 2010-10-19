check_PROGRAMS = complextest
complextest_SOURCES = complextest.cpp
complextest_LDADD = $(top_builddir)/$(GENERIC_LIBRARY_NAME)/libhmsbeagle.la

TESTS = complextest
TESTS_ENVIRONMENT = LD_LIBRARY_PATH+=@CHECK_LIB_PATH@
AM_CPPFLAGS = -I$(top_builddir) -I$(top_srcdir)

EXTRA_DIST=check_lnL_using_BEAST.xml
