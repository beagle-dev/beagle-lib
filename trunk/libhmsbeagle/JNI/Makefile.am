noinst_LTLIBRARIES=libjni.la 

libjni_la_SOURCES = beagle_BeagleJNIWrapper.cpp beagle_BeagleJNIWrapper.h 
libjni_la_LDFLAGS = 
libjni_la_CXXFLAGS = $(AM_CXXFLAGS) $(CUDA_CFLAGS)
libjni_la_LIBADD = $(CUDA_LIBS) -lhmsbeagle


INCLUDES = -I$(top_builddir) -I$(top_srcdir)

