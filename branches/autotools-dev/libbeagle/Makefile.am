SUBDIRS=CUDA CPU JNI

lib_LTLIBRARIES=libbeagle.la libbeagle-jni.la
libbeagle_la_SOURCES=beagle.cpp BeagleImpl.h 
libbeagle_la_LIBADD = CPU/libcpu.la  
if BUILDCUDA
libbeagle_la_CFLAGS = $(CUDA_CFLAGS)
libbeagle_la_CXXFLAGS = $(CUDA_CFLAGS)
libbeagle_la_LIBADD += CUDA/libcuda.la 
endif
libbeagle_la_LDFLAGS= -version-info $(GENERIC_LIBRARY_VERSION) -release $(GENERIC_RELEASE)

libbeagle_jni_la_SOURCES = beagle.cpp BeagleImpl.h
libbeagle_jni_la_LDFLAGS = -module -avoid-version $(JNI_EXTRA_LDFLAGS)
libbeagle_jni_la_LIBADD = JNI/libjni.la
if BUILDCUDA
libbeagle_jni_la_CFLAGS = $(CUDA_CFLAGS)
libbeagle_jni_la_CXXFLAGS = $(CUDA_CFLAGS)
libbeagle_jni_la_LIBADD += CUDA/libcuda.la 
endif


library_includedir=$(includedir)/$(GENERIC_LIBRARY_NAME)-$(GENERIC_API_VERSION)/$(GENERIC_LIBRARY_NAME)
library_include_HEADERS = beagle.h

check_PROGRAMS = tinyTest
tinyTest_SOURCES = tests/tinyTest.cpp
tinyTest_LDADD = libbeagle.la

TESTS = tinyTest


INCLUDES = -I$(top_builddir)
