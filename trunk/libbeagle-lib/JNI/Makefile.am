noinst_LTLIBRARIES=libjni.la 

libjni_la_SOURCES = beagle_BeagleJNIWrapper.cpp beagle_BeagleJNIWrapper.h
libjni_la_LDFLAGS = 
# TODO: remove CUDA conditional here, should not be necessary
if BUILDCUDA
libjni_la_CFLAGS = $(CUDA_CFLAGS)
libjni_la_CXXFLAGS = $(CUDA_CFLAGS)
libjni_la_LIBADD = $(CUDA_LIBS) 
endif

INCLUDES = -I$(top_builddir)

