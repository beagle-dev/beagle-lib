#!/bin/bash

NVCC="$1"
NVCCFLAGS="$2"
INCLUDE_DIRS="$3"

echo "NVCC=${NVCC}"
echo "NVCCFLAGS=${NVCCFLAGS}"
echo "INCLUDE_DIRS=${INCLUDE_DIRS}"

# For OpenCL, we need to generate the file `BeagleOpenCL_kernels.h` using the commands (and dependencies) below

STATE_COUNT_LIST='32 64 256'

srcdir="."

# rules for building cuda files
#BeagleTensorCore_kernels.h: Makefile kernels4.cu kernels4Derivatives.cu kernelsX.cu kernelsXDerivatives.cu kernelsAll.cu ../GPUImplDefs.h
	echo "// auto-generated header file with CUDA kernels PTX code" > BeagleTensorCore_kernels.h
#
# Compile single-precision kernels
#
# 	Compile 4-state model
#	${NVCC} -o BeagleTensorCore_kernels.ptx --default-stream per-thread -ptx -DCUDA -DSTATE_COUNT=4 \
#		$srcdir/kernels4.cu ${NVCCFLAGS} -DHAVE_CONFIG_H ${INCLUDE_DIRS} || { \rm BeagleTensorCore_kernels.h; exit; }; \
#	echo "#define KERNELS_STRING_SP_4 \"" | sed 's/$/\\n\\/' >> BeagleTensorCore_kernels.h; \
#	cat BeagleTensorCore_kernels.ptx | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleTensorCore_kernels.h; \
#	echo "\"" >> BeagleTensorCore_kernels.h
#
#	HERE IS THE LOOP FOR GENERIC KERNELS
#
#	for s in $STATE_COUNT_LIST; do \
#		echo "Making CUDA Tensor Core SP state count = $s" ; \
#		${NVCC} -o BeagleTensorCore_kernels.ptx --default-stream per-thread -ptx -DCUDA -DSTATE_COUNT=$s \
#			$srcdir/kernelsX.cu ${NVCCFLAGS} -DHAVE_CONFIG_H ${INCLUDE_DIRS} || { \rm BeagleTensorCore_kernels.h; exit; }; \
#		echo "#define KERNELS_STRING_SP_$s \"" | sed 's/$/\\n\\/' >> BeagleTensorCore_kernels.h; \
#		cat BeagleTensorCore_kernels.ptx | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleTensorCore_kernels.h; \
#		echo "\"" >> BeagleTensorCore_kernels.h; \
#	done

#
# Compile double-precision kernels
#
# 	Compile 4-state model
	${NVCC} -o BeagleTensorCore_kernels.ptx --default-stream per-thread -ptx -DCUDA -DSTATE_COUNT=4 -DDOUBLE_PRECISION \
		$srcdir/kernels4.cu ${NVCCFLAGS} -DHAVE_CONFIG_H ${INCLUDE_DIRS} || { \rm BeagleTensorCore_kernels.h; exit; }; \
	echo "#define KERNELS_STRING_DP_4 \"" | sed 's/$/\\n\\/' >> BeagleTensorCore_kernels.h; \
	cat BeagleTensorCore_kernels.ptx | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleTensorCore_kernels.h; \
	echo "\"" >> BeagleTensorCore_kernels.h
#
#	HERE IS THE LOOP FOR GENERIC KERNELS
#
	for s in $STATE_COUNT_LIST; do \
		echo "Making CUDA Tensor Core DP state count = $s" ; \
		${NVCC} -o BeagleTensorCore_kernels.ptx --default-stream per-thread -ptx -DCUDA -DSTATE_COUNT=$s -DDOUBLE_PRECISION \
			$srcdir/kernelsX.cu ${NVCCFLAGS} -DHAVE_CONFIG_H ${INCLUDE_DIRS} || { \rm BeagleTensorCore_kernels.h; exit; }; \
		echo "#define KERNELS_STRING_DP_$s \"" | sed 's/$/\\n\\/' >> BeagleTensorCore_kernels.h; \
		cat BeagleTensorCore_kernels.ptx | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleTensorCore_kernels.h; \
		echo "\"" >> BeagleTensorCore_kernels.h; \
	done
