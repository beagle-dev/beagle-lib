SUBDIRS=GPU CPU JNI

lib_LTLIBRARIES=libbeagle-lib.la libbeagle-jni.la
libbeagle_lib_la_SOURCES=beagle.cpp BeagleImpl.h 
libbeagle_lib_la_LIBADD = CPU/libcpu.la  
if BUILDCUDA
libbeagle_lib_la_CFLAGS = $(CUDA_CFLAGS)
libbeagle_lib_la_CXXFLAGS = $(CUDA_CFLAGS)
libbeagle_lib_la_LIBADD += GPU/libcuda.la 
endif
libbeagle_lib_la_LDFLAGS= -version-info $(GENERIC_LIBRARY_VERSION) -release $(GENERIC_RELEASE)

libbeagle_jni_la_SOURCES = beagle.cpp BeagleImpl.h
libbeagle_jni_la_LDFLAGS = -module -avoid-version $(JNI_EXTRA_LDFLAGS)
libbeagle_jni_la_LIBADD = JNI/libjni.la
if BUILDCUDA
libbeagle_jni_la_CFLAGS = $(CUDA_CFLAGS)
libbeagle_jni_la_CXXFLAGS = $(CUDA_CFLAGS)
libbeagle_jni_la_LIBADD += GPU/libcuda.la 
endif


library_includedir=$(includedir)/$(GENERIC_LIBRARY_NAME)-$(GENERIC_API_VERSION)/$(GENERIC_LIBRARY_NAME)
library_include_HEADERS = beagle.h

INCLUDES = -I$(top_builddir) -I$(top_srcdir)
