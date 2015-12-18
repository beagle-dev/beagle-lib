INCLUDE_DIRS = -I$(abs_top_builddir) -I$(abs_top_srcdir)

BUILT_SOURCES =
CLEANFILES =
EXTRA_DIST =

STATE_COUNT_LIST = 16 32 48 64 80 128 192

if BUILDCUDA

BUILT_SOURCES += BeagleCUDA_kernels.h
CLEANFILES += BeagleCUDA_kernels.h BeagleCUDA_kernels.ptx

# rules for building cuda files
BeagleCUDA_kernels.h: Makefile kernels4.cu kernelsX.cu kernelsAll.cu ../GPUImplDefs.h
	echo "// auto-generated header file with CUDA kernels PTX code" > BeagleCUDA_kernels.h
#
# Compile single-precision kernels
#
# 	Compile 4-state model
	$(NVCC) -o BeagleCUDA_kernels.ptx -ptx -DCUDA -DSTATE_COUNT=4 \
		$(srcdir)/kernels4.cu $(NVCCFLAGS) -DHAVE_CONFIG_H $(INCLUDE_DIRS)
	echo "#define KERNELS_STRING_SP_4 \"" | sed 's/$$/\\n\\/' >> BeagleCUDA_kernels.h
	cat BeagleCUDA_kernels.ptx | sed 's/\"/\\"/g' | sed 's/$$/\\n\\/' >> BeagleCUDA_kernels.h
	echo "\"" >> BeagleCUDA_kernels.h
#
#	HERE IS THE LOOP FOR GENERIC KERNELS
#
	for s in $(STATE_COUNT_LIST); do \
		echo "Making state count = $$s" ; \
		($(NVCC) -o BeagleCUDA_kernels.ptx -ptx -DCUDA -DSTATE_COUNT=$$s \
			$(srcdir)/kernelsX.cu $(NVCCFLAGS) -DHAVE_CONFIG_H $(INCLUDE_DIRS)) || exit; \
		echo "#define KERNELS_STRING_SP_$$s \"" | sed 's/$$/\\n\\/' >> BeagleCUDA_kernels.h; \
		cat BeagleCUDA_kernels.ptx | sed 's/\"/\\"/g' | sed 's/$$/\\n\\/' >> BeagleCUDA_kernels.h; \
		echo "\"" >> BeagleCUDA_kernels.h; \
	done

#
# Compile double-precision kernels
#
# 	Compile 4-state model
	$(NVCC) -o BeagleCUDA_kernels.ptx -ptx -DCUDA -DSTATE_COUNT=4 -DDOUBLE_PRECISION \
		$(srcdir)/kernels4.cu $(NVCCFLAGS) -DHAVE_CONFIG_H $(INCLUDE_DIRS)
	echo "#define KERNELS_STRING_DP_4 \"" | sed 's/$$/\\n\\/' >> BeagleCUDA_kernels.h
	cat BeagleCUDA_kernels.ptx | sed 's/\"/\\"/g' | sed 's/$$/\\n\\/' >> BeagleCUDA_kernels.h
	echo "\"" >> BeagleCUDA_kernels.h
#
#	HERE IS THE LOOP FOR GENERIC KERNELS
#
	for s in $(STATE_COUNT_LIST); do \
		echo "Making state count = $$s" ; \
		($(NVCC) -o BeagleCUDA_kernels.ptx -ptx -DCUDA -DSTATE_COUNT=$$s -DDOUBLE_PRECISION \
			$(srcdir)/kernelsX.cu $(NVCCFLAGS) -DHAVE_CONFIG_H $(INCLUDE_DIRS)) || exit; \
		echo "#define KERNELS_STRING_DP_$$s \"" | sed 's/$$/\\n\\/' >> BeagleCUDA_kernels.h; \
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
BeagleOpenCL_kernels.h: Makefile kernels4.cu kernelsX.cu kernelsAll.cu ../GPUImplDefs.h
	echo "// auto-generated header file with OpenCL kernels code" > BeagleOpenCL_kernels.h

#
# Compile single-precision kernels
#
# 	Compile 4-state model
	echo "#define KERNELS_STRING_SP_4 \"" | sed 's/$$/\\n\\/' >> BeagleOpenCL_kernels.h
	echo "#define STATE_COUNT 4" | sed 's/$$/\\n\\/' >> BeagleOpenCL_kernels.h
	cat $(srcdir)/../GPUImplDefs.h | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$$/\\n\\/' >> BeagleOpenCL_kernels.h
	cat $(srcdir)/kernelsAll.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$$/\\n\\/' >> BeagleOpenCL_kernels.h
	cat $(srcdir)/kernels4.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$$/\\n\\/' >> BeagleOpenCL_kernels.h
	echo "\"" >> BeagleOpenCL_kernels.h

#
#	HERE IS THE LOOP FOR GENERIC KERNELS
#
	for s in $(STATE_COUNT_LIST); do \
		echo "Making state count = $$s" ; \
		echo "#define KERNELS_STRING_SP_$$s \"" | sed 's/$$/\\n\\/' >> BeagleOpenCL_kernels.h; \
		echo "#define STATE_COUNT $$s" | sed 's/$$/\\n\\/' >> BeagleOpenCL_kernels.h; \
		cat $(srcdir)/../GPUImplDefs.h | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$$/\\n\\/' >> BeagleOpenCL_kernels.h; \
		cat $(srcdir)/kernelsAll.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$$/\\n\\/' >> BeagleOpenCL_kernels.h; \
		cat $(srcdir)/kernelsX.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$$/\\n\\/' >> BeagleOpenCL_kernels.h; \
		echo "\"" >> BeagleOpenCL_kernels.h; \
	done

#
# Compile double-precision kernels
#
# 	Compile 4-state model
	echo "#define KERNELS_STRING_DP_4 \"" | sed 's/$$/\\n\\/' >> BeagleOpenCL_kernels.h
	echo "#define STATE_COUNT 4" | sed 's/$$/\\n\\/' >> BeagleOpenCL_kernels.h
	echo "#define DOUBLE_PRECISION" | sed 's/$$/\\n\\/' >> BeagleOpenCL_kernels.h
	cat $(srcdir)/../GPUImplDefs.h | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$$/\\n\\/' >> BeagleOpenCL_kernels.h
	cat $(srcdir)/kernelsAll.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$$/\\n\\/' >> BeagleOpenCL_kernels.h
	cat $(srcdir)/kernels4.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$$/\\n\\/' >> BeagleOpenCL_kernels.h
	echo "\"" >> BeagleOpenCL_kernels.h

#
#	HERE IS THE LOOP FOR GENERIC KERNELS
#
	for s in $(STATE_COUNT_LIST); do \
		echo "Making state count = $$s DP"; \
		echo "#define KERNELS_STRING_DP_$$s \"" | sed 's/$$/\\n\\/' >> BeagleOpenCL_kernels.h; \
		echo "#define STATE_COUNT $$s" | sed 's/$$/\\n\\/' >> BeagleOpenCL_kernels.h; \
		echo "#define DOUBLE_PRECISION" | sed 's/$$/\\n\\/' >> BeagleOpenCL_kernels.h; \
		cat $(srcdir)/../GPUImplDefs.h | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$$/\\n\\/' >> BeagleOpenCL_kernels.h; \
		cat $(srcdir)/kernelsAll.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$$/\\n\\/' >> BeagleOpenCL_kernels.h; \
		cat $(srcdir)/kernelsX.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$$/\\n\\/' >> BeagleOpenCL_kernels.h; \
		echo "\"" >> BeagleOpenCL_kernels.h; \
	done

EXTRA_DIST += kernels4.cu kernelsX.cu kernelsAll.cu

libopencl_kernels_la_CXXFLAGS = $(OPENCL_CFLAGS)

endif
