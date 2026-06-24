#!/bin/bash

# Generates BeagleCUDASpectral_kernels.h.
# Each KERNELS_STRING_SP_* / KERNELS_STRING_DP_* entry contains the regular
# kernel PTX AND the spectral PTX concatenated in the same string.
# The spectral PTX module header (.version / .target / .address_size) is
# stripped before appending so the combined PTX remains a valid single module.

NVCC="$1"
NVCCFLAGS="$2"
INCLUDE_DIRS="$3"

echo "NVCC=${NVCC}"
echo "NVCCFLAGS=${NVCCFLAGS}"
echo "INCLUDE_DIRS=${INCLUDE_DIRS}"

STATE_COUNT_LIST='16 32 48 64 80 128 192 256'

srcdir="."

REGULAR_PTX=BeagleCUDASpectral_regular.ptx
SPECTRAL_PTX=BeagleCUDASpectral_spectral.ptx

echo "// auto-generated header file with CUDA spectral kernels PTX code" > BeagleCUDASpectral_kernels.h

#
# SP 4-state
#
${NVCC} -o ${REGULAR_PTX} --default-stream per-thread -ptx -DCUDA -DSTATE_COUNT=4 \
    $srcdir/kernels4.cu ${NVCCFLAGS} -DHAVE_CONFIG_H ${INCLUDE_DIRS} || { \rm BeagleCUDASpectral_kernels.h; exit; }
${NVCC} -o ${SPECTRAL_PTX} --default-stream per-thread -ptx -DCUDA -DSTATE_COUNT=4 \
    $srcdir/kernelsSpectral.cu ${NVCCFLAGS} -DHAVE_CONFIG_H ${INCLUDE_DIRS} || { \rm BeagleCUDASpectral_kernels.h; exit; }
echo "#define KERNELS_STRING_SP_4 \"" | sed 's/$/\\n\\/' >> BeagleCUDASpectral_kernels.h
cat ${REGULAR_PTX} | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleCUDASpectral_kernels.h
sed '1,/^\.address_size/d' ${SPECTRAL_PTX} | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleCUDASpectral_kernels.h
echo "\"" >> BeagleCUDASpectral_kernels.h

#
# SP generic
#
for s in $STATE_COUNT_LIST; do
    echo "Making CUDA Spectral SP state count = $s"
    ${NVCC} -o ${REGULAR_PTX} --default-stream per-thread -ptx -DCUDA -DSTATE_COUNT=$s \
        $srcdir/kernelsX.cu ${NVCCFLAGS} -DHAVE_CONFIG_H ${INCLUDE_DIRS} || { \rm BeagleCUDASpectral_kernels.h; exit; }
    ${NVCC} -o ${SPECTRAL_PTX} --default-stream per-thread -ptx -DCUDA -DSTATE_COUNT=$s \
        $srcdir/kernelsSpectral.cu ${NVCCFLAGS} -DHAVE_CONFIG_H ${INCLUDE_DIRS} || { \rm BeagleCUDASpectral_kernels.h; exit; }
    echo "#define KERNELS_STRING_SP_$s \"" | sed 's/$/\\n\\/' >> BeagleCUDASpectral_kernels.h
    cat ${REGULAR_PTX} | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleCUDASpectral_kernels.h
    sed '1,/^\.address_size/d' ${SPECTRAL_PTX} | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleCUDASpectral_kernels.h
    echo "\"" >> BeagleCUDASpectral_kernels.h
done

#
# DP 4-state
#
${NVCC} -o ${REGULAR_PTX} --default-stream per-thread -ptx -DCUDA -DSTATE_COUNT=4 -DDOUBLE_PRECISION \
    $srcdir/kernels4.cu ${NVCCFLAGS} -DHAVE_CONFIG_H ${INCLUDE_DIRS} || { \rm BeagleCUDASpectral_kernels.h; exit; }
${NVCC} -o ${SPECTRAL_PTX} --default-stream per-thread -ptx -DCUDA -DSTATE_COUNT=4 -DDOUBLE_PRECISION \
    $srcdir/kernelsSpectral.cu ${NVCCFLAGS} -DHAVE_CONFIG_H ${INCLUDE_DIRS} || { \rm BeagleCUDASpectral_kernels.h; exit; }
echo "#define KERNELS_STRING_DP_4 \"" | sed 's/$/\\n\\/' >> BeagleCUDASpectral_kernels.h
cat ${REGULAR_PTX} | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleCUDASpectral_kernels.h
sed '1,/^\.address_size/d' ${SPECTRAL_PTX} | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleCUDASpectral_kernels.h
echo "\"" >> BeagleCUDASpectral_kernels.h

#
# DP generic
#
for s in $STATE_COUNT_LIST; do
    echo "Making CUDA Spectral DP state count = $s"
    ${NVCC} -o ${REGULAR_PTX} --default-stream per-thread -ptx -DCUDA -DSTATE_COUNT=$s -DDOUBLE_PRECISION \
        $srcdir/kernelsX.cu ${NVCCFLAGS} -DHAVE_CONFIG_H ${INCLUDE_DIRS} || { \rm BeagleCUDASpectral_kernels.h; exit; }
    ${NVCC} -o ${SPECTRAL_PTX} --default-stream per-thread -ptx -DCUDA -DSTATE_COUNT=$s -DDOUBLE_PRECISION \
        $srcdir/kernelsSpectral.cu ${NVCCFLAGS} -DHAVE_CONFIG_H ${INCLUDE_DIRS} || { \rm BeagleCUDASpectral_kernels.h; exit; }
    echo "#define KERNELS_STRING_DP_$s \"" | sed 's/$/\\n\\/' >> BeagleCUDASpectral_kernels.h
    cat ${REGULAR_PTX} | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleCUDASpectral_kernels.h
    sed '1,/^\.address_size/d' ${SPECTRAL_PTX} | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleCUDASpectral_kernels.h
    echo "\"" >> BeagleCUDASpectral_kernels.h
done

\rm -f ${REGULAR_PTX} ${SPECTRAL_PTX}
