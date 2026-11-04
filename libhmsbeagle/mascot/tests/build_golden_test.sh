#!/bin/bash
# Builds the golden test for one kernel outside CMake: build_golden_test.sh A|B <output directory>
# Kernel X lives in ../MascotKernelX.cpp, compiled twice, both with -ffp-contract=off: strict, and fma
# (MASCOT_FP_FMA), which fuses only where the kernel writes MASCOT_MUL_ADD (MascotFma.hpp).
set -e
KERNEL=$1
OUT=$2
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$HERE/..
mkdir -p "$OUT"
case $(uname -m) in
  arm64|aarch64) ARCH="-mcpu=apple-m1" ;;
  *) ARCH="-msse2" ;;
esac
CXX="${CXX:-clang++} -std=c++17 -O3 -fno-math-errno $ARCH -I$SRC"
$CXX -ffp-contract=off -c "$SRC/MascotKernel$KERNEL.cpp" -o "$OUT/kernel_strict.o"
$CXX -ffp-contract=off -DMASCOT_FP_FMA -c "$SRC/MascotKernel$KERNEL.cpp" -o "$OUT/kernel_fma.o"
$CXX -ffp-contract=off -DMASCOT_TEST_ONLY_$KERNEL -c "$HERE/mascot_golden_test.cpp" -o "$OUT/driver.o"
$CXX -o "$OUT/mascot_golden_test_$KERNEL" "$OUT/driver.o" "$OUT/kernel_strict.o" "$OUT/kernel_fma.o"
echo "built $OUT/mascot_golden_test_$KERNEL"
