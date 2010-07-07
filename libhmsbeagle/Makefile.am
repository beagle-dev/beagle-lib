SUBDIRS=GPU CPU JNI plugin

lib_LTLIBRARIES=libhmsbeagle.la libhmsbeagle-jni.la

libhmsbeagle_la_SOURCES=beagle.cpp BeagleImpl.h
libhmsbeagle_la_LIBADD = plugin/libplugin.la -lltdl
libhmsbeagle_la_CXXFLAGS = $(AM_CXXFLAGS)
libhmsbeagle_la_LDFLAGS= -version-info $(GENERIC_LIBRARY_VERSION) -release $(GENERIC_RELEASE)

libhmsbeagle_jni_la_SOURCES = beagle.cpp BeagleImpl.h
libhmsbeagle_jni_la_LDFLAGS = -module -avoid-version $(JNI_EXTRA_LDFLAGS) -lhmsbeagle
libhmsbeagle_jni_la_LIBADD = JNI/libjni.la plugin/libplugin.la -lltdl
libhmsbeagle_jni_la_CXXFLAGS = $(AM_CXXFLAGS)

library_includedir=$(includedir)/$(GENERIC_LIBRARY_NAME)-$(GENERIC_API_VERSION)/$(GENERIC_LIBRARY_NAME)
library_include_HEADERS = beagle.h platform.h 

INCLUDES = -I$(top_builddir) -I$(top_srcdir)
