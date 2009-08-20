
INCLUDES = -I$(abs_top_builddir) -I$(abs_top_srcdir)

if BUILDCUDA
noinst_LTLIBRARIES= libcuda.la 
libcuda_la_SOURCES =  \
BeagleGPUImpl.cpp  BeagleGPUImpl.h  GPUImplDefs.h  GPUImplHelper.cpp \
GPUImplHelper.h  GPUInterface.h  GPUInterfaceCUDA.cpp  KernelLauncher.cpp  KernelLauncher.h 

nodist_libcuda_la_SOURCES = BeagleCUDA_Kernels.h
BUILT_SOURCES = BeagleCUDA_Kernels.h
CLEANFILES = BeagleCUDA_Kernels.h

# rules for building cuda files
BeagleCUDA_Kernels.h: Makefile
#	$(NVCC) -o BeagleCUDA_kernels.ptx -ptx $(srcdir)/BeagleCUDA_kernels.cu $(NVCCFLAGS) -DHAVE_CONFIG_H $(INCLUDES)
	$(NVCC) -m32 -D_POSIX_C_SOURCE -o BeagleCUDA_kernels.ptx -ptx $(srcdir)/BeagleCUDA_kernels.cu $(NVCCFLAGS) -DHAVE_CONFIG_H $(INCLUDES)
	echo "// auto-generated header file with CUDA kernels PTX code" > BeagleCUDA_kernels.h
	echo "#define KERNELS_STRING \"\\" >> BeagleCUDA_kernels.h
	cat BeagleCUDA_kernels.ptx | sed 's/\"/\\"/g' | sed 's/$$/\\n\\/' >> BeagleCUDA_kernels.h
	echo "\"" >> BeagleCUDA_kernels.h

EXTRA_DIST = BeagleCUDA_kernels.cu

libcuda_la_CFLAGS = $(CUDA_CFLAGS)
libcuda_la_CXXFLAGS = $(CUDA_CFLAGS)

endif

if BUILDOPENCL
noinst_LTLIBRARIES= libopencl.la 
libopencl_la_SOURCES =  \
BeagleGPUImpl.cpp  BeagleGPUImpl.h  GPUImplDefs.h  GPUImplHelper.cpp \
GPUImplHelper.h  GPUInterface.h  GPUInterfaceOpenCL.cpp  KernelLauncher.cpp  KernelLauncher.h 

nodist_libopencl_la_SOURCES = BeagleOpenCL_Kernels.h
BUILT_SOURCES = BeagleOpenCL_Kernels.h
CLEANFILES = BeagleOpenCL_Kernels.h

# rules for building cuda files
BeagleOpenCL_Kernels.h: Makefile
	echo "// auto-generated header file with OpenCL kernels code" > BeagleOpenCL_kernels.h
	echo "#define KERNELS_STRING \"\\" >> BeagleOpenCL_kernels.h
	cat GPUImplDefs.h | sed 's/\"/\\"/g' | sed 's/$$/\\n\\/' >> BeagleOpenCL_kernels.h
	cat BeagleOpenCL_kernels.cl | sed 's/\"/\\"/g' | sed 's/$$/\\n\\/' >> BeagleOpenCL_kernels.h
	echo "\"" >> BeagleOpenCL_kernels.h

EXTRA_DIST = BeagleOpenCL_kernels.cu

libopencl_la_CFLAGS = $(OPENCL_CFLAGS)
libopencl_la_CXXFLAGS = $(OPENCL_CFLAGS)

endif

