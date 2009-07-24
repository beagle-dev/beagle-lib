
INCLUDES = -I$(abs_top_builddir) -I$(abs_top_srcdir)

if BUILDCUDA
noinst_LTLIBRARIES= libcuda.la 
libcuda_la_SOURCES = BeagleCUDA_Kernels.h \
BeagleGPUImpl.cpp  BeagleGPUImpl.h  GPUImplDefs.h  GPUImplHelper.cpp \
GPUImplHelper.h  GPUInterface.h  GPUInterfaceCUDA.cpp  KernelLauncher.cpp  KernelLauncher.h 

nodist_libcuda_la_SOURCES = BeagleCUDA_Kernels.h
BUILT_SOURCES = BeagleCUDA_Kernels.h
CLEANFILES = BeagleCUDA_Kernels.h

# rules for building cuda files
BeagleCUDA_Kernels.h: Makefile
	$(NVCC) -o $(abs_builddir)/BeagleCUDA_kernels.ptx -ptx BeagleCUDA_kernels.cu $(NVCCFLAGS) -DHAVE_CONFIG_H $(INCLUDES)
	echo "// auto-generated header file with CUDA kernels PTX code" > $(abs_builddir)/BeagleCUDA_kernels.h
	echo "#define KERNELS_STRING \"\\" >> $(abs_builddir)/BeagleCUDA_kernels.h
	cat $(abs_builddir)/BeagleCUDA_kernels.ptx | sed 's/\"/\\"/g' | sed 's/$$/\\n\\/' >> $(abs_builddir)/BeagleCUDA_kernels.h
	echo "\"" >> $(abs_builddir)/BeagleCUDA_kernels.h

EXTRA_DIST = BeagleCUDA_kernels.cu

libcuda_la_CFLAGS = $(CUDA_CFLAGS)
libcuda_la_CXXFLAGS = $(CUDA_CFLAGS)

endif

