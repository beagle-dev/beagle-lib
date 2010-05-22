noinst_LTLIBRARIES=libcpu.la 

libcpu_la_SOURCES = EigenDecomposition.h \
                    EigenDecompositionCube.hpp EigenDecompositionCube.h \
                    EigenDecompositionSquare.hpp EigenDecompositionSquare.h \
		    		BeagleCPUImpl.hpp BeagleCPUImpl.h \
                    BeagleCPU4StateImpl.hpp BeagleCPU4StateImpl.h \
                    BeagleCPU4StateSSEImpl.hpp BeagleCPU4StateSSEImpl.h \
                    BeagleCPUSSEImpl.hpp BeagleCPUSSEimpl.h \
                    BeagleCPU4StateGCDImpl.cpp BeagleCPU4StateGCDImpl.h
libcpu_la_CXXFLAGS = $(AM_CXXFLAGS) $(OPENMP_CXXFLAGS)
INCLUDES = -I$(abs_top_builddir) -I$(abs_top_srcdir)
