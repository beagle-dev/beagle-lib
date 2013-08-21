AUTOMAKE_OPTIONS = subdir-objects
ACLOCAL_AMFLAGS = ${ACLOCAL_FLAGS}

bin_PROGRAMS = hello
hello_SOURCES = src/hello.cpp 
hello_CXXFLAGS = $(DEPS_CFLAGS)
hello_LDADD = $(DEPS_LIBS)

dist_noinst_SCRIPTS = autogen.sh
