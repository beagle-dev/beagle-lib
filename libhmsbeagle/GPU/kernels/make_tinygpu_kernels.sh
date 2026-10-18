#!/bin/bash

# Generates BeagleTinyGPU_kernels.h — the TinyGPU-specific PTX
# kernel header. Mirrors make_cuda_kernels.sh (same KERNELS_STRING_<PREC>_<N>
# macro names, same source files, same state-count sweep) but compiles with
# both -DCUDA and -DFW_TINYGPU. With no other -D flags the PTX is
# byte-identical to a plain -DCUDA build (TODO.md Phase 140); FW_TINYGPU
# only gates the opt-in TINYGPU_DEBUG_DUMP_MATMUL_GROUND_TRUTH probe in
# kernelMatrixMulADB (kernelsAll.cu).
#
# Uses absolute paths throughout: unlike a native nvcc, the nvcc this
# project uses on macOS is a Docker-exec shim (~/.local/bin/nvcc ->
# nvccshim) that does not preserve the host's working directory inside the
# container, so relative filenames silently fail to resolve.

set -e

NVCC="$1"
NVCCFLAGS="$2"
INCLUDE_DIRS="$3"
shift 3
# Optional words, in any order (CMake drops an empty unquoted argument, so they are not positional):
#   spectral  also the SPEC_ modules (BUILD_SPECTRAL; TODO.md plan step S4.2)
#   fused     -DCUDA_FUSED in every module, standard and SPEC_: kernelsFused.cu's kernelUpPartialsFused
#             (BUILD_FUSED; fused-handoff/PLAN-fused-uppartials.md section 6)
SPECTRAL=; FUSED=
for w in "$@"; do
	case "$w" in
		spectral) SPECTRAL=spectral ;;
		fused) FUSED=-DCUDA_FUSED ;;
	esac
done

echo "NVCC=${NVCC}"
echo "NVCCFLAGS=${NVCCFLAGS}"
echo "INCLUDE_DIRS=${INCLUDE_DIRS}"

STATE_COUNT_LIST='16 32 48 64 80 128 192 256'

srcdir="$(cd "$(dirname "$0")" && pwd)"
outheader="${srcdir}/BeagleTinyGPU_kernels.h"
outptx="${srcdir}/BeagleTinyGPU_kernels.ptx"
# TODO.md plan step C1: each SP module is also kept, as the bytes its
# KERNELS_STRING_SP_<N> holds (a newline, then the PTX), for
# make_tinygpu_cubins.sh; since plan step C16 each DP module too (DP_<N>.ptx).
cubindir="${srcdir}/tinygpu_cubins"
mkdir -p "${cubindir}"

echo "// auto-generated header file with TinyGPU kernels PTX code (-DCUDA -DFW_TINYGPU)" > "${outheader}"
echo "#define TINYGPU_KERNELS_STAMP \"$(${NVCC} --version | tail -1 | tr -d '\n') @ $(date '+%Y-%m-%d %H:%M:%S')\"" >> "${outheader}"

#
# Compile single-precision kernels
#
# 	Compile 4-state model
	${NVCC} -o "${outptx}" --default-stream per-thread -ptx -DCUDA -DFW_TINYGPU ${FUSED} -DSTATE_COUNT=4 \
		"${srcdir}/kernels4.cu" ${NVCCFLAGS} -DHAVE_CONFIG_H ${INCLUDE_DIRS} || { rm -f "${outheader}"; exit 1; }
	echo "#define KERNELS_STRING_SP_4 \"" | sed 's/$/\\n\\/' >> "${outheader}"
	cat "${outptx}" | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> "${outheader}"
	echo "\"" >> "${outheader}"
	{ echo; cat "${outptx}"; } > "${cubindir}/SP_4.ptx"
#
#	HERE IS THE LOOP FOR GENERIC KERNELS
#
	for s in $STATE_COUNT_LIST; do
		echo "Making TinyGPU SP state count = $s"
		${NVCC} -o "${outptx}" --default-stream per-thread -ptx -DCUDA -DFW_TINYGPU ${FUSED} -DSTATE_COUNT=$s \
			"${srcdir}/kernelsX.cu" ${NVCCFLAGS} -DHAVE_CONFIG_H ${INCLUDE_DIRS} || { rm -f "${outheader}"; exit 1; }
		echo "#define KERNELS_STRING_SP_$s \"" | sed 's/$/\\n\\/' >> "${outheader}"
		cat "${outptx}" | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> "${outheader}"
		echo "\"" >> "${outheader}"
		{ echo; cat "${outptx}"; } > "${cubindir}/SP_$s.ptx"
	done

#
# Compile double-precision kernels
#
# 	Compile 4-state model
	${NVCC} -o "${outptx}" --default-stream per-thread -ptx -DCUDA -DFW_TINYGPU ${FUSED} -DSTATE_COUNT=4 -DDOUBLE_PRECISION \
		"${srcdir}/kernels4.cu" ${NVCCFLAGS} -DHAVE_CONFIG_H ${INCLUDE_DIRS} || { rm -f "${outheader}"; exit 1; }
	echo "#define KERNELS_STRING_DP_4 \"" | sed 's/$/\\n\\/' >> "${outheader}"
	cat "${outptx}" | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> "${outheader}"
	echo "\"" >> "${outheader}"
	{ echo; cat "${outptx}"; } > "${cubindir}/DP_4.ptx"
#
#	HERE IS THE LOOP FOR GENERIC KERNELS
#
	for s in $STATE_COUNT_LIST; do
		echo "Making TinyGPU DP state count = $s"
		${NVCC} -o "${outptx}" --default-stream per-thread -ptx -DCUDA -DFW_TINYGPU ${FUSED} -DSTATE_COUNT=$s -DDOUBLE_PRECISION \
			"${srcdir}/kernelsX.cu" ${NVCCFLAGS} -DHAVE_CONFIG_H ${INCLUDE_DIRS} || { rm -f "${outheader}"; exit 1; }
		echo "#define KERNELS_STRING_DP_$s \"" | sed 's/$/\\n\\/' >> "${outheader}"
		cat "${outptx}" | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> "${outheader}"
		echo "\"" >> "${outheader}"
		{ echo; cat "${outptx}"; } > "${cubindir}/DP_$s.ptx"
	done

#
# Spectral modules (TODO.md plan step S4.2): -DCUDA_SPECTRAL makes kernels4.cu and kernelsX.cu include
# kernelsSpectralIfDef4.cu and kernelsSpectralIfDef.cu: the standard kernels plus the spectral ones, which a spectral
# instance's KernelLauncher binds together. Kept only as tinygpu_cubins/SPEC_<P>_<N>.ptx (not in the header); the stamp
# above covers them, since they come from this run.
#
if [ "${SPECTRAL}" = spectral ]; then
	for p in SP DP; do
		if [ $p = DP ]; then pflag=-DDOUBLE_PRECISION; else pflag=; fi
		for s in 4 $STATE_COUNT_LIST; do
			if [ $s = 4 ]; then src=kernels4.cu; else src=kernelsX.cu; fi
			echo "Making TinyGPU spectral $p state count = $s"
			${NVCC} -o "${outptx}" --default-stream per-thread -ptx -DCUDA -DFW_TINYGPU ${FUSED} -DCUDA_SPECTRAL -DSTATE_COUNT=$s ${pflag} \
				"${srcdir}/${src}" ${NVCCFLAGS} -DHAVE_CONFIG_H ${INCLUDE_DIRS} || { rm -f "${outheader}"; exit 1; }
			{ echo; cat "${outptx}"; } > "${cubindir}/SPEC_${p}_$s.ptx"
		done
	done
fi

rm -f "${outptx}"
