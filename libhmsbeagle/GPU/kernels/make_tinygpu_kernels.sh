#!/bin/bash

# Generates BeagleTinyGPU_kernels.h — the TinyGPU/TinyGPUHybrid-specific PTX
# kernel header. Mirrors make_cuda_kernels.sh (same KERNELS_STRING_<PREC>_<N>
# macro names, same source files, same state-count sweep) but compiles with
# both -DCUDA and -DFW_TINYGPU (not -DCUDA alone) so the kernel source can
# special-case the handful of spots that need to differ for this backend
# (see kernelMatrixMulADB in kernelsAll.cu) while every other #ifdef CUDA
# block in these files is completely unaffected (CUDA stays defined).
#
# Uses absolute paths throughout: unlike a native nvcc, the nvcc this
# project uses on macOS is a Docker-exec shim (~/.local/bin/nvcc ->
# nvccshim) that does not preserve the host's working directory inside the
# container, so relative filenames silently fail to resolve.

set -e

NVCC="$1"
NVCCFLAGS="$2"
INCLUDE_DIRS="$3"

echo "NVCC=${NVCC}"
echo "NVCCFLAGS=${NVCCFLAGS}"
echo "INCLUDE_DIRS=${INCLUDE_DIRS}"

STATE_COUNT_LIST='16 32 48 64 80 128 192 256'

srcdir="$(cd "$(dirname "$0")" && pwd)"
outheader="${srcdir}/BeagleTinyGPU_kernels.h"
outptx="${srcdir}/BeagleTinyGPU_kernels.ptx"

echo "// auto-generated header file with TinyGPU kernels PTX code (-DCUDA -DFW_TINYGPU)" > "${outheader}"
echo "#define TINYGPU_KERNELS_STAMP \"$(${NVCC} --version | tail -1 | tr -d '\n') @ $(date '+%Y-%m-%d %H:%M:%S')\"" >> "${outheader}"

#
# Compile single-precision kernels
#
# 	Compile 4-state model
	${NVCC} -o "${outptx}" --default-stream per-thread -ptx -DCUDA -DFW_TINYGPU -DSTATE_COUNT=4 \
		"${srcdir}/kernels4.cu" ${NVCCFLAGS} -DHAVE_CONFIG_H ${INCLUDE_DIRS} || { rm -f "${outheader}"; exit 1; }
	echo "#define KERNELS_STRING_SP_4 \"" | sed 's/$/\\n\\/' >> "${outheader}"
	cat "${outptx}" | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> "${outheader}"
	echo "\"" >> "${outheader}"
#
#	HERE IS THE LOOP FOR GENERIC KERNELS
#
	for s in $STATE_COUNT_LIST; do
		echo "Making TinyGPU SP state count = $s"
		${NVCC} -o "${outptx}" --default-stream per-thread -ptx -DCUDA -DFW_TINYGPU -DSTATE_COUNT=$s \
			"${srcdir}/kernelsX.cu" ${NVCCFLAGS} -DHAVE_CONFIG_H ${INCLUDE_DIRS} || { rm -f "${outheader}"; exit 1; }
		echo "#define KERNELS_STRING_SP_$s \"" | sed 's/$/\\n\\/' >> "${outheader}"
		cat "${outptx}" | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> "${outheader}"
		echo "\"" >> "${outheader}"
	done

#
# Compile double-precision kernels
#
# 	Compile 4-state model
	${NVCC} -o "${outptx}" --default-stream per-thread -ptx -DCUDA -DFW_TINYGPU -DSTATE_COUNT=4 -DDOUBLE_PRECISION \
		"${srcdir}/kernels4.cu" ${NVCCFLAGS} -DHAVE_CONFIG_H ${INCLUDE_DIRS} || { rm -f "${outheader}"; exit 1; }
	echo "#define KERNELS_STRING_DP_4 \"" | sed 's/$/\\n\\/' >> "${outheader}"
	cat "${outptx}" | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> "${outheader}"
	echo "\"" >> "${outheader}"
#
#	HERE IS THE LOOP FOR GENERIC KERNELS
#
	for s in $STATE_COUNT_LIST; do
		echo "Making TinyGPU DP state count = $s"
		${NVCC} -o "${outptx}" --default-stream per-thread -ptx -DCUDA -DFW_TINYGPU -DSTATE_COUNT=$s -DDOUBLE_PRECISION \
			"${srcdir}/kernelsX.cu" ${NVCCFLAGS} -DHAVE_CONFIG_H ${INCLUDE_DIRS} || { rm -f "${outheader}"; exit 1; }
		echo "#define KERNELS_STRING_DP_$s \"" | sed 's/$/\\n\\/' >> "${outheader}"
		cat "${outptx}" | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> "${outheader}"
		echo "\"" >> "${outheader}"
	done

rm -f "${outptx}"
