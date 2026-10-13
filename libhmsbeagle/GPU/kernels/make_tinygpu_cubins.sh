#!/bin/bash

# Generates the TinyGPU C++ runtime's ahead-of-time cubins (TODO.md plan
# step C1). Each SP PTX module make_tinygpu_kernels.sh keeps
# (tinygpu_cubins/SP_<N>.ptx: the bytes of KERNELS_STRING_SP_<N>, which the
# daemon's compile_all compiles at run time), and each DP module (DP_<N>.ptx,
# plan step C16), is compiled for every
# architecture in ARCH_LIST (tinygrad's NVDevice.arch names) with
# nv_compile_helper.compile_ptx's ptxas command line. Outputs:
#   BeagleTinyGPU_cubins.S  the cubins, embedded with .incbin (Mach-O)
#   TinyGPUNVCubins.h       their table {states, dp, arch, begin, end}, the ptxas
#                           version, and the stamp of the kernels header
#                           whose PTX they were compiled from
# With a second argument "spectral" (BUILD_SPECTRAL; TODO.md plan step S4.2), also
# the SPEC_SP_<N> and SPEC_DP_<N> modules make_tinygpu_kernels.sh keeps, in a
# second table, kTinyGPUNVSpectralCubins (the same struct, header and stamp), each
# through NV_GATE below.
#
# Uses absolute paths under the source tree: the ptxas this project uses on
# macOS is a Docker-exec shim (~/.local/bin/ptxas) that mounts only $HOME and
# /var/folders and does not preserve the host's working directory.

set -e

PTXAS="$1"
SPECTRAL="$2"

echo "PTXAS=${PTXAS}"

STATE_COUNT_LIST='4 16 32 48 64 80 128 192 256'
ARCH_LIST='sm_86 sm_89 sm_120'
PREFIX_LIST='SP DP'
[ "${SPECTRAL}" = spectral ] && PREFIX_LIST='SP DP SPEC_SP SPEC_DP'

# The NV gate, SPEC_ modules only, on ptxas --verbose: nvd_check_launch skips, with one stderr line, a launch above
# max_threads = ((65536 / round_up(regs*32, 256)) / 4) * 4 * 32 (TinyGPUNVProgram.h, GPUInterfaceTinyGPUNV.cpp's launch);
# *Spectral kernels launch (S, PATTERN_BLOCK_SIZE) threads (KernelLauncher.cpp, bgSpectralPeelingBlock). The loader
# refuses a module above 100 KiB of 0x400 + shared memory (TinyGPUNVProgram.h). Lines that are not ptxas info pass through.
NV_GATE='!/ptxas info|bytes stack frame/ { print }
/Compiling entry function/ { split($0, q, "\047"); k = q[2] }
/ Used [0-9]+ registers/ { r = 0; sm = 0
  for (i = 1; i < NF; i++) { if ($i == "Used") r = $(i + 1); if ($(i + 1) == "bytes" && $(i + 2) ~ /^smem/) sm = $i }
  if (int((1024 + sm + 127) / 128) * 128 > 102400) { print tag ": " k ": " sm " bytes of shared memory"; bad = 1 }
  if (k ~ /Spectral$/) { n++; mt = int(int(65536 / (int((r * 32 + 255) / 256) * 256)) / 4) * 4 * 32
    if (block > mt) { print tag ": " k ": " r " registers allow " mt " threads, launched with " block; bad = 1 } } }
END { if (!n) { print tag ": no *Spectral kernel in ptxas --verbose"; bad = 1 } exit bad }'

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
echo "struct TinyGPUNVCubin { int states; bool dp; const char* arch; const unsigned char* begin; const unsigned char* end; };" >> "${tmpindex}"
table=""
spectable=""

for p in $PREFIX_LIST; do
	dp=$([ ${p#SPEC_} = DP ] && echo true || echo false)
	for s in $STATE_COUNT_LIST; do
		for a in $ARCH_LIST; do
			echo "Making TinyGPU cubin $p state count = $s, $a"
			cubin="${cubindir}/${p}_${s}_$a.cubin"
			sym="tinygpu_cubin_${p}_${s}_$a"
			${PTXAS} "--gpu-name=$a" -O3 --output-file "${cubin}" "${cubindir}/${p}_$s.ptx" || { rm -f "${outasm}" "${outindex}" "${tmpasm}" "${tmpindex}"; exit 1; }
			if [ "$p" != "${p#SPEC_}" ]; then   # spectral: NV_GATE on a second, --verbose run, whose cubin is not kept: on
				# sm_120 --verbose changes the cubin's bytes, which must equal the daemon's compile_ptx (test_c1_cubins.py). ptxas
				# prints its info lines on stderr; pipefail in a subshell keeps a ptxas failure
				pbs=$(awk -v m="PATTERN_BLOCK_SIZE_${p#SPEC_}_$s" '$1 == "#define" && $2 == m { print $3 }' "${srcdir}/../GPUImplDefs.h")
				( set -o pipefail; ${PTXAS} "--gpu-name=$a" -O3 --verbose --output-file /dev/null "${cubindir}/${p}_$s.ptx" 2>&1 \
					| awk -v tag="${p}_${s}_$a" -v block=$((s * pbs)) "${NV_GATE}" ) || { rm -f "${cubin}" "${outasm}" "${outindex}" "${tmpasm}" "${tmpindex}"; exit 1; }
			fi
			# nothing between .incbin and the end label: end - begin is the cubin's size
			printf '\t.private_extern _%s\n\t.p2align 4\n_%s:\n\t.incbin "%s"\n\t.private_extern _%s_end\n_%s_end:\n' \
				"${sym}" "${sym}" "${cubin}" "${sym}" "${sym}" >> "${tmpasm}"
			echo "extern \"C\" const unsigned char ${sym}[], ${sym}_end[];" >> "${tmpindex}"
			if [ "$p" = "${p#SPEC_}" ]; then
				table="${table}	{ $s, $dp, \"$a\", ${sym}, ${sym}_end },\n"
			else
				spectable="${spectable}	{ $s, $dp, \"$a\", ${sym}, ${sym}_end },\n"
			fi
		done
	done
done

printf "static const TinyGPUNVCubin kTinyGPUNVCubins[] = {\n${table}};\n" >> "${tmpindex}"
[ -z "${spectable}" ] || printf "#define TINYGPU_NV_SPECTRAL_CUBINS 1\nstatic const TinyGPUNVCubin kTinyGPUNVSpectralCubins[] = {\n${spectable}};\n" >> "${tmpindex}"
mv "${tmpindex}" "${outindex}"
mv "${tmpasm}" "${outasm}"
