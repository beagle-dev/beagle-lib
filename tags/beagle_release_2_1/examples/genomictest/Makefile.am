check_PROGRAMS = genomictest
genomictest_SOURCES = genomictest.cpp linalg.cpp
genomictest_LDADD = $(top_builddir)/$(GENERIC_LIBRARY_NAME)/libhmsbeagle.la

check_SCRIPTS = genomictest.sh
genomictest.sh:
	echo './genomictest' > genomictest.sh
	echo './genomictest --states 64 --sites 100 --taxa 10' >> genomictest.sh
	chmod +x genomictest.sh

clean-local:
	rm -f genomictest.sh

TESTS = genomictest.sh
AM_CPPFLAGS = -I$(top_builddir) -I$(top_srcdir)

