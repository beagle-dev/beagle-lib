INCLUDES = -I$(abs_top_builddir) -I$(abs_top_srcdir)

BUILT_SOURCES =
CLEANFILES =
EXTRA_DIST =

if BUILDCUDA

BUILT_SOURCES += BeagleCUDA_kernels.h
CLEANFILES += BeagleCUDA_kernels.h BeagleCUDA_kernels.ptx

STATE_COUNT_LIST = 16 32 48 64 80 128 192

# rules for building cuda files
BeagleCUDA_kernels.h: Makefile kernels4.cu kernelsX.cu kernelsAll.cu ../GPUImplDefs.h
	echo "// auto-generated header file with CUDA kernels PTX code" > BeagleCUDA_kernels.h
#
# Compile single-precision kernels
#
# 	Compile 4-state model
	$(NVCC) -o BeagleCUDA_kernels.ptx -ptx -DSTATE_COUNT=4 \
		$(srcdir)/kernels4.cu $(NVCCFLAGS) -DHAVE_CONFIG_H $(INCLUDES)
	echo "#define KERNELS_STRING_SP_4 \"\\" >> BeagleCUDA_kernels.h
	cat BeagleCUDA_kernels.ptx | sed 's/\"/\\"/g' | sed 's/$$/\\n\\/' >> BeagleCUDA_kernels.h
	echo "\"" >> BeagleCUDA_kernels.h	
#
#	HERE IS THE LOOP FOR GENERIC KERNELS
#
	for s in $(STATE_COUNT_LIST); do \
		echo "Making state count = $$s \n" ; \
		($(NVCC) -o BeagleCUDA_kernels.ptx -ptx -DSTATE_COUNT=$$s \
			$(srcdir)/kernelsX.cu $(NVCCFLAGS) -DHAVE_CONFIG_H $(INCLUDES)) || exit; \
		echo "#define KERNELS_STRING_SP_$$s \"\\" >> BeagleCUDA_kernels.h; \
		cat BeagleCUDA_kernels.ptx | sed 's/\"/\\"/g' | sed 's/$$/\\n\\/' >> BeagleCUDA_kernels.h; \
		echo "\"" >> BeagleCUDA_kernels.h; \
	done
	
#
# Compile double-precision kernels
#
# 	Compile 4-state model
	$(NVCC) -o BeagleCUDA_kernels.ptx -ptx -arch compute_13 -DSTATE_COUNT=4 -DDOUBLE_PRECISION \
		$(srcdir)/kernels4.cu $(NVCCFLAGS) -DHAVE_CONFIG_H $(INCLUDES)
	echo "#define KERNELS_STRING_DP_4 \"\\" >> BeagleCUDA_kernels.h
	cat BeagleCUDA_kernels.ptx | sed 's/\"/\\"/g' | sed 's/$$/\\n\\/' >> BeagleCUDA_kernels.h
	echo "\"" >> BeagleCUDA_kernels.h	
#
#	HERE IS THE LOOP FOR GENERIC KERNELS
#
	for s in $(STATE_COUNT_LIST); do \
		echo "Making state count = $$s \n" ; \
		($(NVCC) -o BeagleCUDA_kernels.ptx -ptx -arch compute_13 -DSTATE_COUNT=$$s -DDOUBLE_PRECISION \
			$(srcdir)/kernelsX.cu $(NVCCFLAGS) -DHAVE_CONFIG_H $(INCLUDES)) || exit; \
		echo "#define KERNELS_STRING_DP_$$s \"\\" >> BeagleCUDA_kernels.h; \
		cat BeagleCUDA_kernels.ptx | sed 's/\"/\\"/g' | sed 's/$$/\\n\\/' >> BeagleCUDA_kernels.h; \
		echo "\"" >> BeagleCUDA_kernels.h; \
	done
	
EXTRA_DIST += kernels4.cu kernelsX.cu kernelsAll.cu

libcuda_kernels_la_CXXFLAGS = $(CUDA_CFLAGS)

endif

if BUILDOPENCL

BUILT_SOURCES += BeagleOpenCL_kernels.h
CLEANFILES += BeagleOpenCL_kernels.h

# rules for building opencl files
BeagleOpenCL_kernels.h: Makefile BeagleOpenCL_kernels.cl ../GPUImplDefs.h
	echo "// auto-generated header file with OpenCL kernels code" > BeagleOpenCL_kernels.h
	echo "#define KERNELS_STRING \"\\" >> BeagleOpenCL_kernels.h
	cat BeagleOpenCL_kernels.cl | sed 's/\"/\\"/g' | sed 's/$$/\\n\\/' >> BeagleOpenCL_kernels.h
	echo "\"" >> BeagleOpenCL_kernels.h

EXTRA_DIST += BeagleOpenCL_kernels.cl

libopencl_kernels_la_CXXFLAGS = $(OPENCL_CFLAGS)

endif