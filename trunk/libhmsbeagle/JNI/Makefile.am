if BUILDJNI
noinst_LTLIBRARIES=libjni.la 

libjni_la_SOURCES = beagle_BeagleJNIWrapper.cpp beagle_BeagleJNIWrapper.h 
libjni_la_LDFLAGS = 
libjni_la_CXXFLAGS = $(AM_CXXFLAGS)
libjni_la_LIBADD = -lhmsbeagle
endif

AM_CPPFLAGS = -I$(top_builddir) -I$(top_srcdir)

