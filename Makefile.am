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
project/beagle-vs/tinytest/tinytest_vc90.vcproj \
project/beagle-vs/beagleinstaller/banner.jpg \
project/beagle-vs/beagleinstaller/beagleinstaller.vdproj \
project/beagle-vs/beagleinstaller/COPYING.LESSER.rtf \
project/beagle-vs/build_installer.bat

AM_CFLAGS=-I$(top_builddir) -I$(top_srcdir)
AM_CPPFLAGS=-I$(top_builddir) -I$(top_srcdir)

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = hmsbeagle-1.0.pc

# DOXYGEN SUPPORT
include $(top_srcdir)/aminclude.am

# ------------------------------------------------------------
# Support for profile-guided optimization.  Run `make profile`
# ------------------------------------------------------------
# currently using complextest, need one that hits all possible code paths
#OPTIMIZATIONS= -O3
OPTIMIZATIONS= -O3 -mfpmath=sse -ftree-vectorizer-verbose=5 -msse -msse2 -ffast-math
profile:
	make clean
	make CXXFLAGS="$(CXXFLAGS) -fprofile-generate $(OPTIMIZATIONS)"
	make -C examples/genomictest/ check CXXFLAGS="$(CXXFLAGS) -static -fprofile-generate $(OPTIMIZATIONS)"
	make mostlyclean
	make CXXFLAGS="$(CXXFLAGS) -fprofile-use $(OPTIMIZATIONS)"

CLEANFILES = \
libhmsbeagle/*/*.gcda libhmsbeagle/*/*.gcno \
libhmsbeagle/*/*/*.gcda libhmsbeagle/*/*/*.gcno \
examples/*/*/*.gcda examples/*/*/*.gcno


SUBDIRS = libhmsbeagle examples


