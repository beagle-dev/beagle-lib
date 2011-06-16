SUBDIRS=GPU CPU plugin
if BUILDJNI
lib_LTLIBRARIES=libhmsbeagle.la libhmsbeagle-jni.la
else
lib_LTLIBRARIES=libhmsbeagle.la
endif

libhmsbeagle_la_SOURCES=beagle.cpp BeagleImpl.h
libhmsbeagle_la_LIBADD = plugin/libplugin.la
libhmsbeagle_la_CXXFLAGS = $(AM_CXXFLAGS)
libhmsbeagle_la_LDFLAGS= -version-info $(GENERIC_LIBRARY_VERSION)

if BUILDJNI
libhmsbeagle_jni_la_SOURCES = JNI/beagle_BeagleJNIWrapper.h JNI/beagle_BeagleJNIWrapper.cpp
libhmsbeagle_jni_la_LDFLAGS = -module -avoid-version $(JNI_EXTRA_LDFLAGS)
libhmsbeagle_jni_la_LIBADD = -lhmsbeagle
libhmsbeagle_jni_la_CXXFLAGS = $(AM_CXXFLAGS)
endif

library_includedir=$(includedir)/$(GENERIC_LIBRARY_NAME)-$(GENERIC_API_VERSION)/$(GENERIC_LIBRARY_NAME)
library_include_HEADERS = beagle.h platform.h 

AM_CPPFLAGS = -I$(top_builddir) -I$(top_srcdir)
