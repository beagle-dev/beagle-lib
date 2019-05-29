#!/bin/bash

# For OpenCL, we need to generate the file `BeagleOpenCL_kernels.h` using the commands (and dependencies) below

STATE_COUNT_LIST='16 32 48 64 80 128 192'

srcdir="."

# rules for building opencl files
#BeagleOpenCL_kernels.h: Makefile kernels4.cu kernelsX.cu kernelsAll.cu ../GPUImplDefs.h
echo "// auto-generated header file with OpenCL kernels code" > BeagleOpenCL_kernels.h

#
# Compile single-precision kernels
#
# 	Compile 4-state model
echo "#define KERNELS_STRING_SP_4 \"" | sed 's/$/\\n\\/' >> BeagleOpenCL_kernels.h
echo "#define STATE_COUNT 4" | sed 's/$/\\n\\/' >> BeagleOpenCL_kernels.h
cat $srcdir/../GPUImplDefs.h | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleOpenCL_kernels.h
cat $srcdir/kernelsAll.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleOpenCL_kernels.h
cat $srcdir/kernels4.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleOpenCL_kernels.h
cat $srcdir/kernels4Derivatives.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleOpenCL_kernels.h
echo "\"" >> BeagleOpenCL_kernels.h

#
#	HERE IS THE LOOP FOR GENERIC KERNELS
#
for s in $STATE_COUNT_LIST
do
   echo "Making OpenCL SP state count = $s" ; \
   echo "#define KERNELS_STRING_SP_$s \"" | sed 's/$/\\n\\/' >> BeagleOpenCL_kernels.h; \
   echo "#define STATE_COUNT $s" | sed 's/$/\\n\\/' >> BeagleOpenCL_kernels.h; \
   cat $srcdir/../GPUImplDefs.h | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleOpenCL_kernels.h; \
   cat $srcdir/kernelsAll.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleOpenCL_kernels.h; \
   cat $srcdir/kernelsX.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleOpenCL_kernels.h; \
   cat $srcdir/kernelsXDerivatives.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleOpenCL_kernels.h; \
   echo "\"" >> BeagleOpenCL_kernels.h; \
done

#
# Compile double-precision kernels
#
# 	Compile 4-state model
echo "#define KERNELS_STRING_DP_4 \"" | sed 's/$/\\n\\/' >> BeagleOpenCL_kernels.h
echo "#define STATE_COUNT 4" | sed 's/$/\\n\\/' >> BeagleOpenCL_kernels.h
echo "#define DOUBLE_PRECISION" | sed 's/$/\\n\\/' >> BeagleOpenCL_kernels.h
cat $srcdir/../GPUImplDefs.h | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleOpenCL_kernels.h
cat $srcdir/kernelsAll.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleOpenCL_kernels.h
cat $srcdir/kernels4.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleOpenCL_kernels.h
cat $srcdir/kernels4Derivatives.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleOpenCL_kernels.h
echo "\"" >> BeagleOpenCL_kernels.h

#
#	HERE IS THE LOOP FOR GENERIC KERNELS
#
for s in $STATE_COUNT_LIST
do
   echo "Making OpenCL DP state count = $s DP"; \
   echo "#define KERNELS_STRING_DP_$s \"" | sed 's/$/\\n\\/' >> BeagleOpenCL_kernels.h; \
   echo "#define STATE_COUNT $s" | sed 's/$/\\n\\/' >> BeagleOpenCL_kernels.h; \
   echo "#define DOUBLE_PRECISION" | sed 's/$/\\n\\/' >> BeagleOpenCL_kernels.h; \
   cat $srcdir/../GPUImplDefs.h | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleOpenCL_kernels.h; \
   cat $srcdir/kernelsAll.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleOpenCL_kernels.h; \
   cat $srcdir/kernelsX.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleOpenCL_kernels.h; \
   cat $srcdir/kernelsXDerivatives.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleOpenCL_kernels.h; \
   echo "\"" >> BeagleOpenCL_kernels.h; \
done
