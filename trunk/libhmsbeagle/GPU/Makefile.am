SUBDIRS = kernels

AM_CPPFLAGS = -I$(abs_top_builddir) -I$(abs_top_srcdir)

if BUILDCUDA
lib_LTLIBRARIES= libhmsbeagle-cuda.la 
libhmsbeagle_cuda_la_SOURCES =  \
BeagleGPUImpl.hpp  BeagleGPUImpl.h  GPUImplDefs.h  GPUImplHelper.cpp \
GPUImplHelper.h  GPUInterface.h  GPUInterfaceCUDA.cpp  KernelLauncher.cpp \
KernelLauncher.h KernelResource.cpp KernelResource.h \
CUDAPlugin.h CUDAPlugin.cpp

libhmsbeagle_cuda_la_CXXFLAGS = $(AM_CXXFLAGS) $(CUDA_CFLAGS) 
libhmsbeagle_cuda_la_LDFLAGS= -version-info $(GENERIC_LIBRARY_VERSION) -release $(GENERIC_RELEASE) 
libhmsbeagle_cuda_la_LIBADD= $(CUDA_LIBS)

endif

if BUILDOPENCL
lib_LTLIBRARIES= libhmsbeagle-opencl.la 
libhmsbeagle_opencl_la_SOURCES =  \
BeagleGPUImpl.hpp  BeagleGPUImpl.h  GPUImplDefs.h  GPUImplHelper.cpp \
GPUImplHelper.h  GPUInterface.h  GPUInterfaceOpenCL.cpp  KernelLauncher.cpp \
KernelLauncher.h KernelResource.cpp KernelResource.h \
OpenCLPlugin.h OpenCLPlugin.cpp

nodist_libhmsbeagle_opencl_la_SOURCES = BeagleOpenCL_Kernels.h
BUILT_SOURCES = BeagleOpenCL_Kernels.h
CLEANFILES = BeagleOpenCL_Kernels.h

# rules for building cuda files
BeagleOpenCL_Kernels.h: Makefile BeagleOpenCL_kernels.cu
	echo "// auto-generated header file with OpenCL kernels code" > BeagleOpenCL_kernels.h
	echo "#define KERNELS_STRING \"\\" >> BeagleOpenCL_kernels.h
	cat GPUImplDefs.h | sed 's/\"/\\"/g' | sed 's/$$/\\n\\/' >> BeagleOpenCL_kernels.h
	cat BeagleOpenCL_kernels.cl | sed 's/\"/\\"/g' | sed 's/$$/\\n\\/' >> BeagleOpenCL_kernels.h
	echo "\"" >> BeagleOpenCL_kernels.h

BeagleOpenCL_Kernels_h_DEPENDENCES = BeagleOpenCL_kernels.cl GPUImplDefs.h

EXTRA_DIST = BeagleOpenCL_kernels.cu

libhmsbeagle_opencl_la_CXXFLAGS = $(AM_CXXFLAGS) $(OPENCL_CFLAGS)
libhmsbeagle_opencl_la_LDFLAGS= -version-info $(GENERIC_LIBRARY_VERSION) -release $(GENERIC_RELEASE)
libhmsbeagle_opencl_la_LIBADD= $(OPENCL_LIBS)

endif

