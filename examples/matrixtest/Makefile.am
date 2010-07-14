check_PROGRAMS = matrixtest
matrixtest_SOURCES = matrixtest.cpp
matrixtest_LDADD = $(top_builddir)/$(GENERIC_LIBRARY_NAME)/libhmsbeagle.la @CUDA_LIBS@ @OPENCL_LIBS@

TESTS = matrixtest
INCLUDES = -I$(top_builddir) -I$(top_srcdir)

