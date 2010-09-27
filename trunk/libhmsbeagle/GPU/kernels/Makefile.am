INCLUDES = -I$(abs_top_builddir) -I$(abs_top_srcdir)

if BUILDCUDA

BUILT_SOURCES = BeagleCUDA_kernels.h
CLEANFILES = BeagleCUDA_kernels.h

STATE_COUNT_LIST = 16 32 48 64 80 128 192

# rules for building cuda files
BeagleCUDA_kernels.h: Makefile
	echo "// auto-generated header file with CUDA kernels PTX code" > $(BUILT_SOURCES)
# 	Compile 4-state model
	$(NVCC) -o BeagleCUDA_kernels.ptx -ptx -DSTATE_COUNT=4 \
		$(srcdir)/kernels4.cu $(NVCCFLAGS) -DHAVE_CONFIG_H $(INCLUDES)
	echo "#define KERNELS_STRING_4 \"\\" >> $(BUILT_SOURCES)
	cat BeagleCUDA_kernels.ptx | sed 's/\"/\\"/g' | sed 's/$$/\\n\\/' >> $(BUILT_SOURCES)
	echo "\"" >> $(BUILT_SOURCES)
	
#
#	HERE IS THE LOOP FOR GENERIC KERNELS
#
	for s in $(STATE_COUNT_LIST); do \
		echo "Making state count = $$s \n" ; \
		($(NVCC) -o BeagleCUDA_kernels.ptx -ptx -DSTATE_COUNT=$$s \
			$(srcdir)/kernelsX.cu $(NVCCFLAGS) -DHAVE_CONFIG_H $(INCLUDES)) || exit; \
		echo "#define KERNELS_STRING_$$s \"\\" >> $(BUILT_SOURCES); \
		cat BeagleCUDA_kernels.ptx | sed 's/\"/\\"/g' | sed 's/$$/\\n\\/' >> $(BUILT_SOURCES); \
		echo "\"" >> $(BUILT_SOURCES); \
	done
	

EXTRA_DIST = kernels4.cu kernelsX.cu kernelsAll.cu

libcuda_kernels_la_CXXFLAGS = $(CUDA_CFLAGS)

endif

if BUILDOPENCL


endif

