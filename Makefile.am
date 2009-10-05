ACLOCAL_AMFLAGS = -I m4
EXTRA_DIST=\
autogen.sh \
build.xml \
beagle.dox \
java/beagle/BeagleFactory.java \
java/beagle/BeagleInfo.java \
java/beagle/Beagle.java \
java/beagle/BeagleJNIWrapper.java \
java/beagle/DependencyAwareBeagleImpl.java \
java/beagle/FourStateBeagleImpl.java \
java/beagle/GeneralBeagleImpl.java \
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj \ 
project/beagle-vs/complextest/complextest.vcproj \
project/beagle-vs/complextest/complextest_vc90.vcproj \
project/beagle-vs/fourtaxon/fourtaxon.vcproj \
project/beagle-vs/fourtaxon/fourtaxon_vc90.vcproj \
project/beagle-vs/genomictest/genomictest.vcproj \
project/beagle-vs/genomictest/genomictest_vc90.vcproj \
project/beagle-vs/libhmsbeagle/createCombinedCudaHeader.bat \
project/beagle-vs/libhmsbeagle/createCUDAkernels.bat \
project/beagle-vs/libhmsbeagle/Cuda.Rules \
project/beagle-vs/libhmsbeagle/findJava.bat \
project/beagle-vs/libhmsbeagle/libhmsbeagle.vcproj \
project/beagle-vs/libhmsbeagle/libhmsbeagle_vc90.vcproj \
project/beagle-vs/libhmsbeagle.sln \
project/beagle-vs/libhmsbeagle_vc90.sln \
project/beagle-vs/tinytest/tinytest.vcproj \
project/beagle-vs/tinytest/tinytest_vc90.vcproj

AM_CFLAGS=-I$(top_builddir) -I$(top_srcdir)
AM_CPPFLAGS=-I$(top_builddir) -I$(top_srcdir)

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = hmsbeagle-1.0.pc

# DOXYGEN SUPPORT
include $(top_srcdir)/aminclude.am

SUBDIRS = libhmsbeagle examples


