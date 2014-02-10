SUBDIRS=GPU CPU plugin

lib_LTLIBRARIES=libhmsbeagle.la

libhmsbeagle_la_SOURCES=beagle.cpp BeagleImpl.h
libhmsbeagle_la_LIBADD = plugin/libplugin.la
libhmsbeagle_la_CXXFLAGS = $(AM_CXXFLAGS)
libhmsbeagle_la_LDFLAGS= -version-info $(GENERIC_LIBRARY_VERSION)

library_includedir=$(includedir)/$(GENERIC_LIBRARY_NAME)-$(GENERIC_API_VERSION)/$(GENERIC_LIBRARY_NAME)
library_include_HEADERS = beagle.h platform.h 

AM_CPPFLAGS = -I$(top_builddir) -I$(top_srcdir)
