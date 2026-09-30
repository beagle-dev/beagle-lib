#!/bin/bash

# Generates BeagleCUDASpectral_kernels.h.
# Each KERNELS_STRING_SP_* / KERNELS_STRING_DP_* entry is the PTX of the regular
# kernels compiled with CUDA_SPECTRAL, which makes kernels4.cu include the 4-state
# spectral kernels (kernelsSpectralIfDef4.cu) and kernelsX.cu the generic ones
# (kernelsSpectralIfDef.cu): the same sources as the OpenCL spectral kernels
# (make_opencl_spectral_kernels.sh), compiled as one module per state count.

NVCC="$1"
NVCCFLAGS="$2"
INCLUDE_DIRS="$3"

echo "NVCC=${NVCC}"
echo "NVCCFLAGS=${NVCCFLAGS}"
echo "INCLUDE_DIRS=${INCLUDE_DIRS}"

STATE_COUNT_LIST='16 32 48 64 80 128 192 256'

srcdir="."

PTX=BeagleCUDASpectral_kernels.ptx

echo "// auto-generated header file with CUDA spectral kernels PTX code" > BeagleCUDASpectral_kernels.h

#
# SP 4-state
#
${NVCC} -o ${PTX} --default-stream per-thread -ptx -DCUDA -DCUDA_SPECTRAL -DSTATE_COUNT=4 \
    $srcdir/kernels4.cu ${NVCCFLAGS} -DHAVE_CONFIG_H ${INCLUDE_DIRS} || { \rm BeagleCUDASpectral_kernels.h; exit; }
echo "#define KERNELS_STRING_SP_4 \"" | sed 's/$/\\n\\/' >> BeagleCUDASpectral_kernels.h
cat ${PTX} | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleCUDASpectral_kernels.h
echo "\"" >> BeagleCUDASpectral_kernels.h

#
# SP generic
#
for s in $STATE_COUNT_LIST; do
    echo "Making CUDA Spectral SP state count = $s"
    ${NVCC} -o ${PTX} --default-stream per-thread -ptx -DCUDA -DCUDA_SPECTRAL -DSTATE_COUNT=$s \
        $srcdir/kernelsX.cu ${NVCCFLAGS} -DHAVE_CONFIG_H ${INCLUDE_DIRS} || { \rm BeagleCUDASpectral_kernels.h; exit; }
    echo "#define KERNELS_STRING_SP_$s \"" | sed 's/$/\\n\\/' >> BeagleCUDASpectral_kernels.h
    cat ${PTX} | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleCUDASpectral_kernels.h
    echo "\"" >> BeagleCUDASpectral_kernels.h
done

#
# DP 4-state
#
${NVCC} -o ${PTX} --default-stream per-thread -ptx -DCUDA -DCUDA_SPECTRAL -DSTATE_COUNT=4 -DDOUBLE_PRECISION \
    $srcdir/kernels4.cu ${NVCCFLAGS} -DHAVE_CONFIG_H ${INCLUDE_DIRS} || { \rm BeagleCUDASpectral_kernels.h; exit; }
echo "#define KERNELS_STRING_DP_4 \"" | sed 's/$/\\n\\/' >> BeagleCUDASpectral_kernels.h
cat ${PTX} | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleCUDASpectral_kernels.h
echo "\"" >> BeagleCUDASpectral_kernels.h

#
# DP generic
#
for s in $STATE_COUNT_LIST; do
    echo "Making CUDA Spectral DP state count = $s"
    ${NVCC} -o ${PTX} --default-stream per-thread -ptx -DCUDA -DCUDA_SPECTRAL -DSTATE_COUNT=$s -DDOUBLE_PRECISION \
        $srcdir/kernelsX.cu ${NVCCFLAGS} -DHAVE_CONFIG_H ${INCLUDE_DIRS} || { \rm BeagleCUDASpectral_kernels.h; exit; }
    echo "#define KERNELS_STRING_DP_$s \"" | sed 's/$/\\n\\/' >> BeagleCUDASpectral_kernels.h
    cat ${PTX} | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleCUDASpectral_kernels.h
    echo "\"" >> BeagleCUDASpectral_kernels.h
done

\rm -f ${PTX}
