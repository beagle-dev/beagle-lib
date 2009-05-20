SUBDIRS=CUDA CPU JNI


lib_LTLIBRARIES=libbeagle.la libbeagle-jni.la
libbeagle_la_SOURCES=beagle.cpp  
libbeagle_la_LIBADD = CPU/libcpu.la CUDA/libcuda.la 
libbeagle_la_CFLAGS = $(CUDA_CFLAGS)
libbeagle_la_CXXFLAGS = $(CUDA_CFLAGS)

libbeagle_jni_la_SOURCES = beagle.cpp  
libbeagle_jni_la_CFLAGS = $(CUDA_CFLAGS)
libbeagle_jni_la_CXXFLAGS = $(CUDA_CFLAGS)
libbeagle_jni_la_LDFLAGS = -module -avoid-version $(JNI_EXTRA_LDFLAGS)
libbeagle_jni_la_LIBADD = $(CUDA_LIBS) CUDA/libcuda.la JNI/libjni.la

library_includedir=$(includedir)/$(GENERIC_LIBRARY_NAME)-$(GENERIC_API_VERSION)/$(GENERIC_LIBRARY_NAME)
library_include_HEADERS = beagle.h

check_PROGRAMS = tinyTest
tinyTest_SOURCES = tests/tinyTest.cpp
tinyTest_LDADD = libbeagle.la

TESTS = tinyTest


INCLUDES = -I$(top_builddir)
