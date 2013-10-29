if BUILDJNI
lib_LTLIBRARIES=libhmsbeagle-jni.la

libhmsbeagle_jni_la_SOURCES = beagle_BeagleJNIWrapper.h beagle_BeagleJNIWrapper.cpp
libhmsbeagle_jni_la_LDFLAGS = -module -avoid-version $(JNI_EXTRA_LDFLAGS)
libhmsbeagle_jni_la_LIBADD = -L$(top_builddir)/libhmsbeagle -lhmsbeagle
libhmsbeagle_jni_la_CXXFLAGS = $(AM_CXXFLAGS)
endif

AM_CPPFLAGS = -I$(top_builddir) -I$(top_srcdir)
