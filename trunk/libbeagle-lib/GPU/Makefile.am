
INCLUDES = -I$(abs_top_builddir) -I$(abs_top_srcdir)

CCLD = g++

if BUILDCUDA
noinst_LTLIBRARIES= libcuda.la 
libcuda_la_SOURCES = BeagleCUDA_Kernels.h \
BeagleGPUImpl.cpp  BeagleGPUImpl.h  GPUImplDefs.h  GPUImplHelper.cpp \
GPUImplHelper.h  GPUInterface.h  GPUInterfaceCUDA.cpp  KernelLauncher.cpp  KernelLauncher.h 

BUILT_SOURCES = BeagleCUDA_Kernels.h
CLEANFILES = BeagleCUDA_Kernels.h

# rules for building cuda files
BeagleCUDA_Kernels.h: Makefile
	$(NVCC) -o BeagleCUDA_kernels.cubin -cubin BeagleCUDA_kernels.cu $(NVCCFLAGS) -DHAVE_CONFIG_H $(INCLUDES)
	echo "// auto-generated header file with CUDA kernels binary code" > BeagleCUDA_kernels.h
	echo "#define KERNELS_STRING \"\\" >> BeagleCUDA_kernels.h
	cat BeagleCUDA_kernels.cubin | sed 's/$$/\\/' >> BeagleCUDA_kernels.h
	echo "\"" >> BeagleCUDA_kernels.h

libcuda_la_CFLAGS = $(CUDA_CFLAGS)
libcuda_la_CXXFLAGS = $(CUDA_CFLAGS)
libcuda_la_LIBADD = $(CUDA_LIBS) 
libcuda_la_LDFLAGS = 

libcuda_la_LINK = $(LIBTOOL) --tag=CC $(AM_LIBTOOLFLAGS) \
        $(LIBTOOLFLAGS) --mode=link $(CCLD) $(libcuda_la_CFLAGS) \
        $(CFLAGS) $(libcuda_la_LDFLAGS) $(LDFLAGS) -o $@
endif

