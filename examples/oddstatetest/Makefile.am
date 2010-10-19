check_PROGRAMS = oddstatetest
oddstatetest_SOURCES = oddstatetest.cpp
oddstatetest_LDADD = $(top_builddir)/$(GENERIC_LIBRARY_NAME)/libhmsbeagle.la

TESTS = oddstatetest
TESTS_ENVIRONMENT = LD_LIBRARY_PATH+=@CHECK_LIB_PATH@
AM_CPPFLAGS = -I$(top_builddir) -I$(top_srcdir)

