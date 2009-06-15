ACLOCAL_AMFLAGS = -I m4
EXTRA_DIST=\
autogen.sh \
cudalt.py \
build.xml \
BEAGLE.iml \
beagle.dox \
java/beagle/BeagleFactory.java \
java/beagle/BeagleInfo.java \
java/beagle/Beagle.java \
java/beagle/BeagleJNIWrapper.java \
java/beagle/DependencyAwareBeagleImpl.java \
java/beagle/FourStateBeagleImpl.java \
java/beagle/GeneralBeagleImpl.java \
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj \
JNI.Makefile \
CUDA.Makefile 

AM_CFLAGS=-I$(top_builddir) -I$(top_srcdir)
AM_CPPFLAGS=-I$(top_builddir) -I$(top_srcdir)

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = beagle-lib-1.0.pc

# DOXYGEN SUPPORT
include $(top_srcdir)/aminclude.am

SUBDIRS = libbeagle-lib examples
