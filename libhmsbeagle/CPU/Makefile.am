lib_LTLIBRARIES=libhmsbeagle-cpu.la 

BEAGLE_CPU_COMMON = Precision.h EigenDecomposition.h \
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
libhmsbeagle_cpu_la_LDFLAGS= -module -version-number $(MODULE_VERSION)


#
# CPU plugin with custom SSE code
#
if HAVE_SSE2
lib_LTLIBRARIES += libhmsbeagle-cpu-sse.la

libhmsbeagle_cpu_sse_la_SOURCES = $(BEAGLE_CPU_COMMON) \
                    SSEDefinitions.h BeagleCPU4StateSSEImpl.hpp BeagleCPU4StateSSEImpl.h \
                    BeagleCPUSSEImpl.hpp BeagleCPUSSEImpl.h \
		BeagleCPUSSEPlugin.h BeagleCPUSSEPlugin.cpp

libhmsbeagle_cpu_sse_la_CXXFLAGS = $(AM_CXXFLAGS) -msse2
libhmsbeagle_cpu_sse_la_LDFLAGS= -module -version-number $(MODULE_VERSION)
endif

#
# CPU plugin with custom AVX code
#
if HAVE_AVX
lib_LTLIBRARIES += libhmsbeagle-cpu-avx.la

libhmsbeagle_cpu_avx_la_SOURCES = $(BEAGLE_CPU_COMMON) \
                    AVXDefinitions.h BeagleCPU4StateAVXImpl.hpp BeagleCPU4StateAVXImpl.h \
                    BeagleCPUSSEImpl.hpp BeagleCPUSSEImpl.h \
		BeagleCPUAVXPlugin.h BeagleCPUAVXPlugin.cpp

libhmsbeagle_cpu_avx_la_CXXFLAGS = $(AM_CXXFLAGS) -mavx
libhmsbeagle_cpu_avx_la_LDFLAGS= -module -version-number $(MODULE_VERSION)
endif

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
libhmsbeagle_cpu_openmp_la_LDFLAGS= -module -version-number $(MODULE_VERSION)
libhmsbeagle_cpu_openmp_la_LIBADD = $(OPENMP_CXXFLAGS)
endif

AM_CPPFLAGS = -I$(abs_top_builddir) -I$(abs_top_srcdir)
