SUBDIRS=GPU CPU JNI plugin
if BUILDJNI
lib_LTLIBRARIES=libhmsbeagle.la libhmsbeagle-jni.la
else
lib_LTLIBRARIES=libhmsbeagle.la
endif

libhmsbeagle_la_SOURCES=beagle.cpp BeagleImpl.h
libhmsbeagle_la_LIBADD = plugin/libplugin.la
libhmsbeagle_la_CXXFLAGS = $(AM_CXXFLAGS)
libhmsbeagle_la_LDFLAGS= -version-info $(GENERIC_LIBRARY_VERSION) -release $(GENERIC_RELEASE)

if BUILDJNI
libhmsbeagle_jni_la_SOURCES = beagle.cpp BeagleImpl.h
if BUILDJNILIBEXT
libhmsbeagle_jni_la_LDFLAGS = -module -avoid-version $(JNI_EXTRA_LDFLAGS) -lhmsbeagle
else
libhmsbeagle_jni_la_LDFLAGS = -version-info $(GENERIC_LIBRARY_VERSION) -release $(GENERIC_RELEASE) -lhmsbeagle
endif
libhmsbeagle_jni_la_LIBADD = JNI/libjni.la plugin/libplugin.la
libhmsbeagle_jni_la_CXXFLAGS = $(AM_CXXFLAGS)
endif

library_includedir=$(includedir)/$(GENERIC_LIBRARY_NAME)-$(GENERIC_API_VERSION)/$(GENERIC_LIBRARY_NAME)
library_include_HEADERS = beagle.h platform.h 

AM_CPPFLAGS = -I$(top_builddir) -I$(top_srcdir)
