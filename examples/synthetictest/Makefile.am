check_PROGRAMS = synthetictest
synthetictest_SOURCES = synthetictest.cpp linalg.cpp
synthetictest_LDADD = $(top_builddir)/$(GENERIC_LIBRARY_NAME)/libhmsbeagle.la

check_SCRIPTS = synthetictest.sh
synthetictest.sh:
	echo './synthetictest' > synthetictest.sh
	echo './synthetictest --states 64 --sites 100 --taxa 10' >> synthetictest.sh
	chmod +x synthetictest.sh

clean-local:
	rm -f synthetictest.sh

TESTS = synthetictest.sh
AM_CPPFLAGS = -I$(top_builddir) -I$(top_srcdir)

