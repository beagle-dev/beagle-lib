#!/bin/bash

# Generates the TinyGPUHybrid C++ runtime's ahead-of-time cubins (TODO.md plan
# step C1). Each SP PTX module make_tinygpu_kernels.sh keeps
# (tinygpu_cubins/SP_<N>.ptx: the bytes of KERNELS_STRING_SP_<N>, which the
# daemon's compile_all compiles at run time) is compiled for every
# architecture in ARCH_LIST (tinygrad's NVDevice.arch names) with
# nv_compile_helper.compile_ptx's ptxas command line. Outputs:
#   BeagleTinyGPU_cubins.S  the cubins, embedded with .incbin (Mach-O)
#   TinyGPUNVCubins.h       their table {states, arch, begin, end}, the ptxas
#                           version, and the stamp of the kernels header
#                           whose PTX they were compiled from
#
# Uses absolute paths under the source tree: the ptxas this project uses on
# macOS is a Docker-exec shim (~/.local/bin/ptxas) that mounts only $HOME and
# /var/folders and does not preserve the host's working directory.

set -e

PTXAS="$1"

echo "PTXAS=${PTXAS}"

STATE_COUNT_LIST='4 16 32 48 64 80 128 192 256'
ARCH_LIST='sm_86 sm_89 sm_120'

srcdir="$(cd "$(dirname "$0")" && pwd)"
cubindir="${srcdir}/tinygpu_cubins"
outasm="${srcdir}/BeagleTinyGPU_cubins.S"
outindex="${srcdir}/TinyGPUNVCubins.h"
# both are written under .tmp names and renamed at the end, the .S last: an interrupted run never leaves a partial pair
tmpasm="${outasm}.tmp"
tmpindex="${outindex}.tmp"

echo "// auto-generated assembler file embedding TinyGPU's ahead-of-time cubins (table: TinyGPUNVCubins.h)" > "${tmpasm}"
echo "	.section __TEXT,__const" >> "${tmpasm}"
echo "// auto-generated header file with the table of TinyGPU's ahead-of-time cubins (BeagleTinyGPU_cubins.S)" > "${tmpindex}"
echo "#define TINYGPU_CUBINS_STAMP \"$(${PTXAS} --version | tail -1 | tr -d '\n') @ $(date '+%Y-%m-%d %H:%M:%S')\"" >> "${tmpindex}"
grep '^#define TINYGPU_KERNELS_STAMP ' "${srcdir}/BeagleTinyGPU_kernels.h" | sed 's/TINYGPU_KERNELS_STAMP/TINYGPU_CUBINS_KERNELS_STAMP/' >> "${tmpindex}"
echo "struct TinyGPUNVCubin { int states; const char* arch; const unsigned char* begin; const unsigned char* end; };" >> "${tmpindex}"
table=""

for s in $STATE_COUNT_LIST; do
	for a in $ARCH_LIST; do
		echo "Making TinyGPU cubin SP state count = $s, $a"
		cubin="${cubindir}/SP_${s}_$a.cubin"
		sym="tinygpu_cubin_SP_${s}_$a"
		${PTXAS} "--gpu-name=$a" -O3 --output-file "${cubin}" "${cubindir}/SP_$s.ptx" || { rm -f "${outasm}" "${outindex}" "${tmpasm}" "${tmpindex}"; exit 1; }
		# nothing between .incbin and the end label: end - begin is the cubin's size
		printf '\t.private_extern _%s\n\t.p2align 4\n_%s:\n\t.incbin "%s"\n\t.private_extern _%s_end\n_%s_end:\n' \
			"${sym}" "${sym}" "${cubin}" "${sym}" "${sym}" >> "${tmpasm}"
		echo "extern \"C\" const unsigned char ${sym}[], ${sym}_end[];" >> "${tmpindex}"
		table="${table}	{ $s, \"$a\", ${sym}, ${sym}_end },\n"
	done
done

printf "static const TinyGPUNVCubin kTinyGPUNVCubins[] = {\n${table}};\n" >> "${tmpindex}"
mv "${tmpindex}" "${outindex}"
mv "${tmpasm}" "${outasm}"
