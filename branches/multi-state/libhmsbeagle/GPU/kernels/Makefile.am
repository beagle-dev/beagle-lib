INCLUDES = -I$(abs_top_builddir) -I$(abs_top_srcdir)

if BUILDCUDA
noinst_LTLIBRARIES= libcuda_kernels.la 
libcuda_kernels_la_SOURCES =  \
kernels4.cu

nodist_libcuda_kernels_la_SOURCES = NewBeagleCUDA_Kernels.h
BUILT_SOURCES = NewBeagleCUDA_Kernels.h
CLEANFILES = NewBeagleCUDA_Kernels.h

# rules for building cuda files
NewBeagleCUDA_Kernels.h: Makefile
	$(NVCC) -m32 -D_POSIX_C_SOURCE -o BeagleCUDA_kernels.ptx -ptx $(srcdir)/kernels4.cu $(NVCCFLAGS) -DHAVE_CONFIG_H $(INCLUDES)
	echo "// auto-generated header file with CUDA kernels PTX code" > NewBeagleCUDA_Kernels.h
	echo "#define KERNELS_STRING \"\\" >> NewBeagleCUDA_Kernels.h
	cat BeagleCUDA_kernels.ptx | sed 's/\"/\\"/g' | sed 's/$$/\\n\\/' >> NewBeagleCUDA_Kernels.h
#
#	HERE IS WHERE THE LOOP FOR GENERIC KERNELS GOES
#
	echo "\"" >> NewBeagleCUDA_Kernels.h

EXTRA_DIST = kernels4.cu

libcuda_kernels_la_CFLAGS = $(CUDA_CFLAGS)
libcuda_kernels_la_CXXFLAGS = $(CUDA_CFLAGS)

endif

if BUILDOPENCL


endif

