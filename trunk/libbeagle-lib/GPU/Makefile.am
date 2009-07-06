
INCLUDES = -I$(abs_top_builddir) -I$(abs_top_srcdir)

# rules for building cuda files
.cu.cubin:
	$(NVCC) -o $@ -cubin $< $(NVCCFLAGS) -DHAVE_CONFIG_H $(INCLUDES)

CCLD = g++

if BUILDCUDA
noinst_LTLIBRARIES= libcuda.la 
libcuda_la_SOURCES = \
BeagleGPUImpl.cpp  BeagleGPUImpl.h  GPUImplDefs.h  GPUImplHelper.cpp \
GPUImplHelper.h  GPUInterface.h  GPUInterfaceCUDA.cpp  KernelLauncher.cpp  KernelLauncher.h \
BeagleCUDA_kernels.cu

libcuda_la_CFLAGS = $(CUDA_CFLAGS)
libcuda_la_CXXFLAGS = $(CUDA_CFLAGS)
libcuda_la_LIBADD = $(CUDA_LIBS) 
libcuda_la_LDFLAGS = 

libcuda_la_LINK = $(LIBTOOL) --tag=CC $(AM_LIBTOOLFLAGS) \
        $(LIBTOOLFLAGS) --mode=link $(CCLD) $(libcuda_la_CFLAGS) \
        $(CFLAGS) $(libcuda_la_LDFLAGS) $(LDFLAGS) -o $@
endif

