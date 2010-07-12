lib_LTLIBRARIES=libhmsbeagle-cpu.la libhmsbeagle-cpu-sse.la

BEAGLE_CPU_COMMON = EigenDecomposition.h \
                    EigenDecompositionCube.hpp EigenDecompositionCube.h \
                    EigenDecompositionSquare.hpp EigenDecompositionSquare.h

#
# Standard CPU plugin
#
libhmsbeagle_cpu_la_SOURCES = $(BEAGLE_CPU_COMMON) \
		    		BeagleCPUImpl.hpp BeagleCPUImpl.h \
                    BeagleCPU4StateImpl.hpp BeagleCPU4StateImpl.h \
		BeagleCPUPlugin.h BeagleCPUPlugin.cpp

libhmsbeagle_cpu_la_CXXFLAGS = $(AM_CXXFLAGS)
libhmsbeagle_cpu_la_LDFLAGS= -version-info $(GENERIC_LIBRARY_VERSION) -release $(GENERIC_RELEASE)


#
# CPU plugin with custom SSE code
#
libhmsbeagle_cpu_sse_la_SOURCES = $(BEAGLE_CPU_COMMON) \
                    BeagleCPU4StateSSEImpl.hpp BeagleCPU4StateSSEImpl.h \
                    BeagleCPUSSEImpl.hpp BeagleCPUSSEimpl.h \
		BeagleCPUSSEPlugin.h BeagleCPUSSEPlugin.cpp

libhmsbeagle_cpu_sse_la_CXXFLAGS = $(AM_CXXFLAGS)
libhmsbeagle_cpu_sse_la_LDFLAGS= -version-info $(GENERIC_LIBRARY_VERSION) -release $(GENERIC_RELEASE)

#
# CPU plugin with OpenMP parallel threads
#
if HAVE_OPENMP
lib_LTLIBRARIES += libhmsbeagle-cpu-openmp.la

libhmsbeagle_cpu_openmp_la_SOURCES = $(BEAGLE_CPU_COMMON) \
		    		BeagleCPUImpl.hpp BeagleCPUImpl.h \
                    BeagleCPU4StateImpl.hpp BeagleCPU4StateImpl.h \
		BeagleCPUOpenMPPlugin.h BeagleCPUOpenMPPlugin.cpp

libhmsbeagle_cpu_openmp_la_CXXFLAGS = $(AM_CXXFLAGS) $(OPENMP_CXXFLAGS)
libhmsbeagle_cpu_openmp_la_LDFLAGS= -version-info $(GENERIC_LIBRARY_VERSION) -release $(GENERIC_RELEASE)
libhmsbeagle_cpu_openmp_la_LIBADD = $(OPENMP_CXXFLAGS)
endif

AM_CPPFLAGS = -I$(abs_top_builddir) -I$(abs_top_srcdir)
