#!/bin/bash

# Generates the TinyGPU AMD C++ runtime's ahead-of-time HSACOs (TODO.md plan step A1j): every SP and DP variant of
# BeagleOpenCL_kernels.h, compiled for each architecture in ARCH_LIST by tinygpu_amd_compile (tinygrad's compile_hip in
# C++, through comgr), the source the AMD daemon compiled at run time before plan step A2l. Outputs:
#   BeagleTinyGPU_hsaco.S  the HSACOs, embedded with .incbin (Mach-O)
#   TinyGPUAMDHsaco.h      their table {variant, arch, begin, end} and a stamp
#   make_tinygpu_hsaco.sh <tinygpu_amd_compile> <libamd_comgr>

set -e

COMPILER="$1"
COMGR="$2"

VARIANT_LIST='SP_4 SP_16 SP_32 SP_48 SP_64 SP_80 SP_128 SP_192 SP_256 DP_4 DP_16 DP_32 DP_48 DP_64 DP_80 DP_128 DP_192 DP_256'
ARCH_LIST='gfx1100'

srcdir="$(cd "$(dirname "$0")" && pwd)"
hsacodir="${srcdir}/tinygpu_hsaco"
outasm="${srcdir}/BeagleTinyGPU_hsaco.S"
outindex="${srcdir}/TinyGPUAMDHsaco.h"
# both are written under .tmp names and renamed at the end, the .S last: an interrupted run never leaves a partial pair
tmpasm="${outasm}.tmp"
tmpindex="${outindex}.tmp"
mkdir -p "${hsacodir}"

echo "// auto-generated assembler file embedding TinyGPU's ahead-of-time AMD HSACOs (table: TinyGPUAMDHsaco.h)" > "${tmpasm}"
echo "	.section __TEXT,__const" >> "${tmpasm}"
echo "// auto-generated header file with the table of TinyGPU's ahead-of-time AMD HSACOs (BeagleTinyGPU_hsaco.S)" > "${tmpindex}"
echo "struct TinyGPUAMDHsaco { const char* variant; const char* arch; const unsigned char* begin; const unsigned char* end; };" >> "${tmpindex}"
table=""

for a in $ARCH_LIST; do
	echo "Making TinyGPU AMD HSACOs for $a"
	"${COMPILER}" "${COMGR}" "$a" "${hsacodir}" ${VARIANT_LIST} || { rm -f "${outasm}" "${outindex}" "${tmpasm}" "${tmpindex}"; exit 1; }
	for v in $VARIANT_LIST; do
		hsaco="${hsacodir}/${v}_$a.hsaco"
		sym="tinygpu_hsaco_${v}_$a"
		# nothing between .incbin and the end label: end - begin is the HSACO's size
		printf '\t.private_extern _%s\n\t.p2align 4\n_%s:\n\t.incbin "%s"\n\t.private_extern _%s_end\n_%s_end:\n' \
			"${sym}" "${sym}" "${hsaco}" "${sym}" "${sym}" >> "${tmpasm}"
		echo "extern \"C\" const unsigned char ${sym}[], ${sym}_end[];" >> "${tmpindex}"
		table="${table}	{ \"$v\", \"$a\", ${sym}, ${sym}_end },\n"
	done
done

printf "static const TinyGPUAMDHsaco kTinyGPUAMDHsacos[] = {\n${table}};\n" >> "${tmpindex}"
mv "${tmpindex}" "${outindex}"
mv "${tmpasm}" "${outasm}"
