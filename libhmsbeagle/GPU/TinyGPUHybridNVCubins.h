/*
 * TinyGPUHybridNVCubins.h
 *
 * The C++ runtime's ahead-of-time cubins (TODO.md plan step C1). The build compiles the single-precision PTX modules, and the
 * double-precision ones (plan step C16), for every supported architecture with the daemon's own ptxas command line and links
 * the cubins into the plugin (kernels/make_tinygpu_cubins.sh, which generates
 * their table, kTinyGPUNVCubins in kernels/TinyGPUNVCubins.h). This picks the
 * cubin for a state count, precision and GPU, checks its ELF header, and lists its
 * kernels, in place of the daemon's run-time ptxas (cmd_compile_all). The
 * architecture name is tinygrad's own (NVDevice.arch, ops_nv.py:631), from the
 * C++ boot's NVDevice. There is no run-time fallback (decision 15).
 */

#ifndef LIBHMSBEAGLE_GPU_TINYGPUHYBRIDNVCUBINS_H
#define LIBHMSBEAGLE_GPU_TINYGPUHYBRIDNVCUBINS_H

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "libhmsbeagle/GPU/kernels/TinyGPUNVCubins.h"
#include "libhmsbeagle/GPU/TinyGPUHybridNVProgram.h"

namespace tinygpu_device {

// The SM a cubin was built for, from its ELF header: with EI_ABIVERSION 7 (ptxas's sm_86 and sm_89 cubins) it is
// e_flags bits 0-7, with 8 (its sm_120 cubins) bits 8-15. 0 when unknown.
static inline uint32_t nvd_elf_sm(const uint8_t* blob, size_t n) {
    if (n < 0x34 || memcmp(blob, "\x7f" "ELF", 4) != 0) return 0;
    uint32_t e_flags = nvd_rd<uint32_t>(blob + 0x30);
    return blob[8] == 7 ? (e_flags & 0xff) : blob[8] == 8 ? ((e_flags >> 8) & 0xff) : 0;
}

// The embedded cubin for this state count, precision and architecture, once its ELF header confirms the SM. Returns an
// empty string on success, otherwise why no embedded cubin can run this request on this GPU.
static inline std::string nvd_find_cubin(const TinyGPUNVCubin* table, size_t n, int states, bool dp, const std::string& arch,
                                         const TinyGPUNVCubin*& out) {
    out = nullptr;
    const std::string what = std::to_string(states) + " states in " + (dp ? "double" : "single") + " precision";
    std::string archs;
    for (size_t i = 0; i < n; ++i) {
        if (table[i].states != states || table[i].dp != dp) continue;
        archs += std::string(archs.empty() ? "" : ", ") + table[i].arch;
        if (arch == table[i].arch) out = &table[i];
    }
    if (archs.empty()) return "no embedded cubin for " + what;
    if (!out) return "no embedded cubin for this GPU's architecture (" + arch + "); this build has " + archs;
    uint32_t sm = nvd_elf_sm(out->begin, (size_t)(out->end - out->begin));
    if (arch.compare(0, 3, "sm_") != 0 || sm != strtoul(arch.c_str() + 3, nullptr, 10)) {
        out = nullptr;
        return "the embedded " + arch + " cubin for " + what + " is built for SM " + std::to_string(sm);
    }
    return "";
}

// The kernels of a loaded cubin: its .text.<name> sections, the names the daemon's compile_all finds
// (nv_compile_helper._parse_elf_kernels' first pass).
static inline std::vector<std::string> nvd_kernel_names(const NVDElf& elf) {
    std::vector<std::string> names;
    for (const NVDElfSection& s : elf.sections)
        if (s.name.compare(0, 6, ".text.") == 0 && s.name.compare(0, 7, ".text..") != 0) names.push_back(s.name.substr(6));
    return names;
}

} // namespace tinygpu_device

#endif // LIBHMSBEAGLE_GPU_TINYGPUHYBRIDNVCUBINS_H
