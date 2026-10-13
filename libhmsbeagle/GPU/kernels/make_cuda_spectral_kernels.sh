#!/bin/bash

# Generates BeagleCUDASpectral_kernels.h.
# Each KERNELS_STRING_SP_* / KERNELS_STRING_DP_* entry is the PTX of the regular
# kernels compiled with CUDA_SPECTRAL, which makes kernels4.cu include the 4-state
# spectral kernels (kernelsSpectralIfDef4.cu) and kernelsX.cu the generic ones
# (kernelsSpectralIfDef.cu): the same sources as the OpenCL spectral kernels
# (make_opencl_spectral_kernels.sh), compiled as one module per state count.
#
# Uses absolute paths throughout, as make_tinygpu_kernels.sh does: the nvcc
# Docker shim (~/.local/bin/nvcc, TinyGPUBuild.md) does not preserve the host's
# working directory inside the container, so relative filenames do not resolve.

NVCC="$1"
NVCCFLAGS="$2"
INCLUDE_DIRS="$3"

echo "NVCC=${NVCC}"
echo "NVCCFLAGS=${NVCCFLAGS}"
echo "INCLUDE_DIRS=${INCLUDE_DIRS}"

STATE_COUNT_LIST='16 32 48 64 80 128 192 256'

srcdir="$(cd "$(dirname "$0")" && pwd)"
outheader="${srcdir}/BeagleCUDASpectral_kernels.h"
PTX="${srcdir}/BeagleCUDASpectral_kernels.ptx"

echo "// auto-generated header file with CUDA spectral kernels PTX code" > "${outheader}"

#
# SP 4-state
#
${NVCC} -o "${PTX}" --default-stream per-thread -ptx -DCUDA -DCUDA_SPECTRAL -DSTATE_COUNT=4 \
    "${srcdir}/kernels4.cu" ${NVCCFLAGS} -DHAVE_CONFIG_H ${INCLUDE_DIRS} || { rm -f "${outheader}"; exit 1; }
echo "#define KERNELS_STRING_SP_4 \"" | sed 's/$/\\n\\/' >> "${outheader}"
cat "${PTX}" | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> "${outheader}"
echo "\"" >> "${outheader}"

#
# SP generic
#
for s in $STATE_COUNT_LIST; do
    echo "Making CUDA Spectral SP state count = $s"
    ${NVCC} -o "${PTX}" --default-stream per-thread -ptx -DCUDA -DCUDA_SPECTRAL -DSTATE_COUNT=$s \
        "${srcdir}/kernelsX.cu" ${NVCCFLAGS} -DHAVE_CONFIG_H ${INCLUDE_DIRS} || { rm -f "${outheader}"; exit 1; }
    echo "#define KERNELS_STRING_SP_$s \"" | sed 's/$/\\n\\/' >> "${outheader}"
    cat "${PTX}" | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> "${outheader}"
    echo "\"" >> "${outheader}"
done

#
# DP 4-state
#
${NVCC} -o "${PTX}" --default-stream per-thread -ptx -DCUDA -DCUDA_SPECTRAL -DSTATE_COUNT=4 -DDOUBLE_PRECISION \
    "${srcdir}/kernels4.cu" ${NVCCFLAGS} -DHAVE_CONFIG_H ${INCLUDE_DIRS} || { rm -f "${outheader}"; exit 1; }
echo "#define KERNELS_STRING_DP_4 \"" | sed 's/$/\\n\\/' >> "${outheader}"
cat "${PTX}" | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> "${outheader}"
echo "\"" >> "${outheader}"

#
# DP generic
#
for s in $STATE_COUNT_LIST; do
    echo "Making CUDA Spectral DP state count = $s"
    ${NVCC} -o "${PTX}" --default-stream per-thread -ptx -DCUDA -DCUDA_SPECTRAL -DSTATE_COUNT=$s -DDOUBLE_PRECISION \
        "${srcdir}/kernelsX.cu" ${NVCCFLAGS} -DHAVE_CONFIG_H ${INCLUDE_DIRS} || { rm -f "${outheader}"; exit 1; }
    echo "#define KERNELS_STRING_DP_$s \"" | sed 's/$/\\n\\/' >> "${outheader}"
    cat "${PTX}" | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> "${outheader}"
    echo "\"" >> "${outheader}"
done

rm -f "${PTX}"
