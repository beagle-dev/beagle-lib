SUBDIRS=GPU CPU JNI

lib_LTLIBRARIES=libhmsbeagle.la libhmsbeagle-jni.la
libhmsbeagle_la_SOURCES=beagle.cpp BeagleImpl.h platform.h 
libhmsbeagle_la_LIBADD = CPU/libcpu.la  
libhmsbeagle_la_CFLAGS = $(CUDA_CFLAGS) $(OPENCL_CFLAGS) $(OPENMP_CFLAGS)
libhmsbeagle_la_CXXFLAGS = $(CUDA_CFLAGS) $(OPENCL_CFLAGS) $(OPENMP_CXXFLAGS)

if BUILDCUDA
libhmsbeagle_la_LIBADD += GPU/libcuda.la 
endif

if BUILDOPENCL
libhmsbeagle_la_LIBADD += GPU/libopencl.la 
endif

libhmsbeagle_la_LDFLAGS= -version-info $(GENERIC_LIBRARY_VERSION) -release $(GENERIC_RELEASE)

libhmsbeagle_jni_la_SOURCES = beagle.cpp BeagleImpl.h
libhmsbeagle_jni_la_LDFLAGS = -module -avoid-version $(JNI_EXTRA_LDFLAGS) -lhmsbeagle
libhmsbeagle_jni_la_LIBADD = JNI/libjni.la
libhmsbeagle_jni_la_CFLAGS = $(CUDA_CFLAGS) $(OPENCL_CFLAGS)
libhmsbeagle_jni_la_CXXFLAGS = $(CUDA_CFLAGS) $(OPENCL_CFLAGS)

if BUILDCUDA
libhmsbeagle_jni_la_LIBADD += GPU/libcuda.la 
endif

if BUILDOPENCL
libhmsbeagle_jni_la_LIBADD += GPU/libopencl.la 
endif

library_includedir=$(includedir)/$(GENERIC_LIBRARY_NAME)-$(GENERIC_API_VERSION)/$(GENERIC_LIBRARY_NAME)
library_include_HEADERS = beagle.h

INCLUDES = -I$(top_builddir) -I$(top_srcdir)
