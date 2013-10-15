AUTOMAKE_OPTIONS = subdir-objects
ACLOCAL_AMFLAGS = ${ACLOCAL_FLAGS}

bin_PROGRAMS = epochtest
epochtest_SOURCES = src/epochtest.cpp 
epochtest_CXXFLAGS = $(DEPS_CFLAGS)
epochtest_LDADD = $(DEPS_LIBS)

dist_noinst_SCRIPTS = autogen.sh
