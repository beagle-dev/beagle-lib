SUBDIRS = kernels

AM_CPPFLAGS = -I$(abs_top_builddir) -I$(abs_top_srcdir)
lib_LTLIBRARIES=
if BUILDCUDA
lib_LTLIBRARIES+= libhmsbeagle-cuda.la 
libhmsbeagle_cuda_la_SOURCES =  \
BeagleGPUImpl.hpp  BeagleGPUImpl.h  GPUImplDefs.h  GPUImplHelper.cpp \
GPUImplHelper.h  GPUInterface.h  GPUInterfaceCUDA.cpp  KernelLauncher.cpp \
KernelLauncher.h KernelResource.cpp KernelResource.h \
CUDAPlugin.h CUDAPlugin.cpp Precision.h

libhmsbeagle_cuda_la_CXXFLAGS = $(AM_CXXFLAGS) $(CUDA_CFLAGS) 
libhmsbeagle_cuda_la_LDFLAGS= -version-info $(GENERIC_LIBRARY_VERSION) 
libhmsbeagle_cuda_la_LIBADD= $(CUDA_LIBS)

endif

if BUILDOPENCL
lib_LTLIBRARIES+= libhmsbeagle-opencl.la 
libhmsbeagle_opencl_la_SOURCES =  \
BeagleGPUImpl.hpp  BeagleGPUImpl.h  GPUImplDefs.h  GPUImplHelper.cpp \
GPUImplHelper.h  GPUInterface.h  GPUInterfaceOpenCL.cpp  KernelLauncher.cpp \
KernelLauncher.h KernelResource.cpp KernelResource.h \
OpenCLPlugin.h OpenCLPlugin.cpp Precision.h

libhmsbeagle_opencl_la_CXXFLAGS = $(AM_CXXFLAGS) $(OPENCL_CFLAGS)
libhmsbeagle_opencl_la_LDFLAGS= -version-info $(GENERIC_LIBRARY_VERSION)
libhmsbeagle_opencl_la_LIBADD= $(OPENCL_LIBS)

endif

