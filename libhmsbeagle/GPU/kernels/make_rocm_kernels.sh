#!/bin/bash

# Generates BeagleROCm_kernels.h — kernel source strings for the ROCm/HIP backend.
# Content is identical to the OpenCL kernel strings; FW_ROCM is injected at
# runtime by HIPRTC (not embedded here), so the file needs no special preamble.

STATE_COUNT_LIST='16 32 48 64 80 128 192 256'

srcdir="."

echo "// auto-generated header file with ROCm/HIP kernel source code" > BeagleROCm_kernels.h

#
# Single-precision 4-state
#
echo "#define KERNELS_STRING_SP_4 \"" | sed 's/$/\\n\\/' >> BeagleROCm_kernels.h
echo "#define STATE_COUNT 4" | sed 's/$/\\n\\/' >> BeagleROCm_kernels.h
cat $srcdir/../GPUImplDefs.h | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleROCm_kernels.h
cat $srcdir/kernelsAll.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleROCm_kernels.h
cat $srcdir/kernels4.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleROCm_kernels.h
cat $srcdir/kernels4CudaCore.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleROCm_kernels.h
cat $srcdir/kernels4Derivatives.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleROCm_kernels.h
cat $srcdir/kernels4DerivativesCudaCore.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleROCm_kernels.h
echo "\"" >> BeagleROCm_kernels.h

#
# Single-precision X-state loop
#
for s in $STATE_COUNT_LIST
do
   echo "Making ROCm SP state count = $s" ; \
   echo "#define KERNELS_STRING_SP_$s \"" | sed 's/$/\\n\\/' >> BeagleROCm_kernels.h; \
   echo "#define STATE_COUNT $s" | sed 's/$/\\n\\/' >> BeagleROCm_kernels.h; \
   cat $srcdir/../GPUImplDefs.h | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleROCm_kernels.h; \
   cat $srcdir/kernelsAll.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleROCm_kernels.h; \
   cat $srcdir/kernelsX.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleROCm_kernels.h; \
   cat $srcdir/kernelsXCudaCore.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleROCm_kernels.h; \
   cat $srcdir/kernelsXDerivatives.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleROCm_kernels.h; \
   cat $srcdir/kernelsXDerivativesCudaCore.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleROCm_kernels.h; \
   echo "\"" >> BeagleROCm_kernels.h; \
done

#
# Double-precision 4-state
#
echo "#define KERNELS_STRING_DP_4 \"" | sed 's/$/\\n\\/' >> BeagleROCm_kernels.h
echo "#define STATE_COUNT 4" | sed 's/$/\\n\\/' >> BeagleROCm_kernels.h
echo "#define DOUBLE_PRECISION" | sed 's/$/\\n\\/' >> BeagleROCm_kernels.h
cat $srcdir/../GPUImplDefs.h | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleROCm_kernels.h
cat $srcdir/kernelsAll.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleROCm_kernels.h
cat $srcdir/kernels4.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleROCm_kernels.h
cat $srcdir/kernels4CudaCore.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleROCm_kernels.h
cat $srcdir/kernels4Derivatives.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleROCm_kernels.h
cat $srcdir/kernels4DerivativesCudaCore.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleROCm_kernels.h
echo "\"" >> BeagleROCm_kernels.h

#
# Double-precision X-state loop
#
for s in $STATE_COUNT_LIST
do
   echo "Making ROCm DP state count = $s DP"; \
   echo "#define KERNELS_STRING_DP_$s \"" | sed 's/$/\\n\\/' >> BeagleROCm_kernels.h; \
   echo "#define STATE_COUNT $s" | sed 's/$/\\n\\/' >> BeagleROCm_kernels.h; \
   echo "#define DOUBLE_PRECISION" | sed 's/$/\\n\\/' >> BeagleROCm_kernels.h; \
   cat $srcdir/../GPUImplDefs.h | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleROCm_kernels.h; \
   cat $srcdir/kernelsAll.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleROCm_kernels.h; \
   cat $srcdir/kernelsX.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleROCm_kernels.h; \
   cat $srcdir/kernelsXCudaCore.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleROCm_kernels.h; \
   cat $srcdir/kernelsXDerivatives.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleROCm_kernels.h; \
   cat $srcdir/kernelsXDerivativesCudaCore.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleROCm_kernels.h; \
   echo "\"" >> BeagleROCm_kernels.h; \
done
