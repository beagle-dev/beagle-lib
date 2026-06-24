#!/bin/bash

# Generates BeagleOpenCLSpectral_kernels.h.
# Each KERNELS_STRING_SP_* / KERNELS_STRING_DP_* entry contains the regular
# kernel source AND kernelsSpectralIfDef.cu concatenated in the same string.

STATE_COUNT_LIST='16 32 48 64 80 128 192 256'

srcdir="."

echo "// auto-generated header file with OpenCL spectral kernels code" > BeagleOpenCLSpectral_kernels.h

#
# SP 4-state
#
echo "#define KERNELS_STRING_SP_4 \"" | sed 's/$/\\n\\/' >> BeagleOpenCLSpectral_kernels.h
echo "#define STATE_COUNT 4" | sed 's/$/\\n\\/' >> BeagleOpenCLSpectral_kernels.h
cat $srcdir/../GPUImplDefs.h       | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleOpenCLSpectral_kernels.h
cat $srcdir/kernelsAll.cu          | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleOpenCLSpectral_kernels.h
cat $srcdir/kernels4.cu            | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleOpenCLSpectral_kernels.h
cat $srcdir/kernels4Derivatives.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleOpenCLSpectral_kernels.h
cat $srcdir/kernelsSpectralIfDef.cu| sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleOpenCLSpectral_kernels.h
echo "\"" >> BeagleOpenCLSpectral_kernels.h

#
# SP generic
#
for s in $STATE_COUNT_LIST; do
    echo "Making OpenCL Spectral SP state count = $s"
    echo "#define KERNELS_STRING_SP_$s \"" | sed 's/$/\\n\\/' >> BeagleOpenCLSpectral_kernels.h
    echo "#define STATE_COUNT $s" | sed 's/$/\\n\\/' >> BeagleOpenCLSpectral_kernels.h
    cat $srcdir/../GPUImplDefs.h        | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleOpenCLSpectral_kernels.h
    cat $srcdir/kernelsAll.cu           | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleOpenCLSpectral_kernels.h
    cat $srcdir/kernelsX.cu             | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleOpenCLSpectral_kernels.h
    cat $srcdir/kernelsXDerivatives.cu  | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleOpenCLSpectral_kernels.h
    cat $srcdir/kernelsSpectralIfDef.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleOpenCLSpectral_kernels.h
    echo "\"" >> BeagleOpenCLSpectral_kernels.h
done

#
# DP 4-state
#
echo "#define KERNELS_STRING_DP_4 \"" | sed 's/$/\\n\\/' >> BeagleOpenCLSpectral_kernels.h
echo "#define STATE_COUNT 4" | sed 's/$/\\n\\/' >> BeagleOpenCLSpectral_kernels.h
echo "#define DOUBLE_PRECISION" | sed 's/$/\\n\\/' >> BeagleOpenCLSpectral_kernels.h
cat $srcdir/../GPUImplDefs.h       | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleOpenCLSpectral_kernels.h
cat $srcdir/kernelsAll.cu          | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleOpenCLSpectral_kernels.h
cat $srcdir/kernels4.cu            | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleOpenCLSpectral_kernels.h
cat $srcdir/kernels4Derivatives.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleOpenCLSpectral_kernels.h
cat $srcdir/kernelsSpectralIfDef.cu| sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleOpenCLSpectral_kernels.h
echo "\"" >> BeagleOpenCLSpectral_kernels.h

#
# DP generic
#
for s in $STATE_COUNT_LIST; do
    echo "Making OpenCL Spectral DP state count = $s DP"
    echo "#define KERNELS_STRING_DP_$s \"" | sed 's/$/\\n\\/' >> BeagleOpenCLSpectral_kernels.h
    echo "#define STATE_COUNT $s" | sed 's/$/\\n\\/' >> BeagleOpenCLSpectral_kernels.h
    echo "#define DOUBLE_PRECISION" | sed 's/$/\\n\\/' >> BeagleOpenCLSpectral_kernels.h
    cat $srcdir/../GPUImplDefs.h        | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleOpenCLSpectral_kernels.h
    cat $srcdir/kernelsAll.cu           | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleOpenCLSpectral_kernels.h
    cat $srcdir/kernelsX.cu             | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleOpenCLSpectral_kernels.h
    cat $srcdir/kernelsXDerivatives.cu  | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleOpenCLSpectral_kernels.h
    cat $srcdir/kernelsSpectralIfDef.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleOpenCLSpectral_kernels.h
    echo "\"" >> BeagleOpenCLSpectral_kernels.h
done
