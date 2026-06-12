#!/bin/bash
#
# Generates BeagleMetal_kernels.h - Metal Shading Language kernel source strings
# for the BEAGLE Metal backend.
#
# The .cu kernel files use KW_* macros (defined in GPUImplDefs.h) that expand to
# MSL equivalents when FW_METAL is defined.  This script concatenates the relevant
# files per (precision, state-count) combination and escapes them as C string
# literals, exactly like make_opencl_kernels.sh does for OpenCL.
#
# Usage: run from the kernels/ directory.
#   cd libhmsbeagle/GPU/kernels && bash make_metal_kernels.sh

STATE_COUNT_LIST='16 32 48 64 80 128 192 256'

srcdir="."
OUT=BeagleMetal_kernels.h

echo "// auto-generated header file with Metal Shading Language kernel source" > $OUT
echo "// Run make_metal_kernels.sh to regenerate." >> $OUT

# ---------------------------------------------------------------------------
# MSL preamble injected at the start of every kernel string.
# Defines the Metal standard library includes and compatibility shims that
# map OpenCL/CUDA idioms to MSL.
# ---------------------------------------------------------------------------
read -r -d '' MSL_PREAMBLE << 'PREAMBLE_EOF'
#include <metal_stdlib>\n\
using namespace metal;\n\
#define FW_METAL 1\n\
// MSL does not have __umul24; use plain multiplication\n\
#define __umul24(x, y) ((x) * (y))\n\
// Float atomic add via bit-cast CAS (Metal 2.x, M1-compatible)\n\
inline void beagle_atomicAddFloat(device atomic_uint* addr, float val) {\n\
    uint e = atomic_load_explicit(addr, memory_order_relaxed);\n\
    uint d;\n\
    do { d = as_type<uint>(as_type<float>(e) + val); }\n\
    while (!atomic_compare_exchange_weak_explicit(addr, &e, d, memory_order_relaxed, memory_order_relaxed));\n\
}\n\
#define atomicAdd(addr, val) beagle_atomicAddFloat((device atomic_uint*)(addr), (val))\n\
// Rename 'distance' to avoid conflict with Metal built-in distance() function\n\
#define distance beagle_branch_len\n\
// Metal frexp takes int& not int*; wrap for pointer-based callers\n\
inline float beagle_frexp(float x, thread int* p) { return frexp(x, *p); }\n\
#define frexp(x, p) beagle_frexp((x), (p))\n\
// MSL does not support double precision on Apple Silicon.\n\
PREAMBLE_EOF

# Helper: escape a file for embedding as a C string and append to $OUT.
# For .cu kernel files, also transforms scalar int/float kernel parameters
# to 'constant int&' / 'constant float&' for Metal compatibility.
# Usage: embed_file <file>
SCRIPTDIR="$(cd "$(dirname "$0")" && pwd)"
embed_file() {
    python3 "$SCRIPTDIR/metal_param_transform.py" < "$1" | \
        sed 's/\\/\\\\/g' | \
        sed 's/\"/\\"/g'  | \
        sed 's/$/\\n\\/'
}

# ---------------------------------------------------------------------------
# Single-precision SP kernels
# ---------------------------------------------------------------------------

# 4-state model
echo "Making Metal SP state count = 4"
echo "#define KERNELS_STRING_SP_4 \"\\" >> $OUT
printf "%s\n" "${MSL_PREAMBLE}" >> $OUT
echo "#define STATE_COUNT 4\\n\\" >> $OUT
embed_file $srcdir/../GPUImplDefs.h >> $OUT
embed_file $srcdir/kernelsAll.cu   >> $OUT
embed_file $srcdir/kernels4.cu     >> $OUT
embed_file $srcdir/kernels4CudaCore.cu >> $OUT
embed_file $srcdir/kernels4Derivatives.cu >> $OUT
embed_file $srcdir/kernels4DerivativesCudaCore.cu >> $OUT
echo "\"" >> $OUT

# Generic state-count models
for s in $STATE_COUNT_LIST; do
    echo "Making Metal SP state count = $s"
    echo "#define KERNELS_STRING_SP_${s} \"\\" >> $OUT
    printf "%s\n" "${MSL_PREAMBLE}" >> $OUT
    echo "#define STATE_COUNT ${s}\\n\\" >> $OUT
    embed_file $srcdir/../GPUImplDefs.h >> $OUT
    embed_file $srcdir/kernelsAll.cu   >> $OUT
    embed_file $srcdir/kernelsX.cu     >> $OUT
    embed_file $srcdir/kernelsXCudaCore.cu >> $OUT
    embed_file $srcdir/kernelsXDerivatives.cu >> $OUT
    embed_file $srcdir/kernelsXDerivativesCudaCore.cu >> $OUT
    echo "\"" >> $OUT
done

# ---------------------------------------------------------------------------
# Double-precision DP kernels
# Apple Silicon Metal does NOT support double precision; mark them NULL so
# GetSupportsDoublePrecision() can return false and the build system skips them.
# ---------------------------------------------------------------------------

echo "" >> $OUT
echo "// Double-precision Metal kernels are not supported on Apple Silicon." >> $OUT
echo "// GetSupportsDoublePrecision() returns false for the Metal backend." >> $OUT
echo "#define KERNELS_STRING_DP_4   nullptr" >> $OUT
for s in $STATE_COUNT_LIST; do
    echo "#define KERNELS_STRING_DP_${s} nullptr" >> $OUT
done

echo "Done writing $OUT"
