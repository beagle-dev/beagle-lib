/*
 * TinyGPUHybridAMDProgram.h
 *
 * The AMD C++ runtime's HSACO loader and scratch sizing (TODO.md plan step A1d): AMDProgram.__init__ as BEAGLE runs it
 * (amd_dispatch_daemon.py BeagleAMDProgram: one multi-kernel HSACO, each kernel's descriptor at its .kd symbol as
 * amd_compile_helper.parse_kernels finds it), over the shared elf_loader port (TinyGPUElf.h), and
 * AMDDevice._ensure_has_local_memory (tinygrad/runtime/ops_amd.py:1113-1128 at a9830e2b4). golden_amd_program.py compares
 * every record, the relocated image and the scratch sizing with tinygrad's own on a stub device.
 *
 * Two deliberate differences, as on NV (TinyGPUHybridNVProgram.h): all kernels share one image upload at lib_va
 * (BeagleAMDProgram allocates and copies the whole image per kernel), and scratch is sized for the largest private segment
 * of all of an HSACO's kernels, growing only for a later HSACO whose kernels need more (tinygrad grows it as each program
 * loads; a larger scratch serves every kernel; TinyGPUHybridAMDRuntime.h's amd_runtime_load_programs). Kernels with the
 * dispatch_ptr, queue_ptr, dispatch_id or private segment buffer SGPRs are refused, not ported
 * (TinyGPUHybridAMDDispatch.h).
 */

#ifndef LIBHMSBEAGLE_GPU_TINYGPUHYBRIDAMDPROGRAM_H
#define LIBHMSBEAGLE_GPU_TINYGPUHYBRIDAMDPROGRAM_H

#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "libhmsbeagle/GPU/TinyGPUAMDTables.h"
#include "libhmsbeagle/GPU/TinyGPUElf.h"
#include "libhmsbeagle/GPU/TinyGPUHybridAMDDispatch.h"

namespace tinygpu_device {

// What AMDProgram.__init__ and _ensure_has_local_memory read from AMDDevice and its iface.props
struct AMDProps {
    uint32_t target_major = 11, xccs = 1, cu_cnt = 0, se_cnt = 0, max_slots_scratch_cu = 0, lds_size_in_kb = 0;
};

// One kernel of the HSACO, as BeagleAMDProgram sets it up
struct AMDProgramRecord {
    uint64_t kd_off = 0;                  // the .kd symbol's value: the descriptor's offset in the image
    uint32_t group_segment_size = 0, private_segment_size = 0;
    uint16_t kernel_code_properties = 0;
    uint64_t aql_prog_addr = 0;           // lib_va + kd_off
    AMDKernel k;                          // what exec reads
};

// elf_loader, BeagleAMDProgram's relocation loop (R_AMDGPU_REL64 only, as AMDProgram), parse_kernels' .kd symbols, then
// AMDProgram.__init__'s per-kernel fields at base lib_va. The relocated image is what goes to lib_va. "" or why not.
inline std::string amd_load_hsaco(const uint8_t* hsaco, size_t n, uint64_t lib_va, const AMDProps& p, std::vector<uint8_t>& image,
                                  std::map<std::string, AMDProgramRecord>& kernels) {
    NVDElf elf;
    std::string err = nvd_elf_load(hsaco, n, 1, elf);
    if (!err.empty()) return "HSACO: " + err;
    image = elf.image;
    for (const NVDElfReloc& r : elf.relocs) {
        if (r.type != 5) return "HSACO: unknown AMD reloc " + std::to_string(r.type);
        if (r.image_off + 8 > image.size()) return "HSACO: a relocation past the image";
        const int64_t v = (int64_t)r.sym_off - (int64_t)r.image_off + r.addend;
        memcpy(image.data() + r.image_off, &v, 8);
    }
    const NVDElfSection *symtab = nullptr, *strtab = nullptr;
    for (const NVDElfSection& s : elf.sections) {
        if (s.name == ".symtab") symtab = &s;
        if (s.name == ".strtab") strtab = &s;
    }
    if (!symtab || !strtab) return "compiled HSACO has no .symtab/.strtab";
    std::map<std::string, uint64_t> kd;
    for (uint64_t i = 0; (i + 1) * 24 <= symtab->size; ++i) {   // Elf64_Sym: st_name u32, info, other, st_shndx u16, st_value, st_size
        const uint8_t* sym = elf.content(*symtab) + i * 24;
        const std::string name = nvd_cstr(elf.content(*strtab), strtab->size, nvd_rd<uint32_t>(sym));
        if (name.size() > 3 && name.compare(name.size() - 3, 3, ".kd") == 0) kd[name.substr(0, name.size() - 3)] = nvd_rd<uint64_t>(sym + 8);
    }
    if (kd.empty()) return "HSACO: no kernel descriptors";
    kernels.clear();
    for (const auto& [name, off] : kd) {
        if (off + amdt::KD_SIZE > image.size()) return "HSACO: " + name + "'s descriptor is past the image";
        const uint8_t* d = image.data() + off;
        AMDProgramRecord r;
        r.kd_off = off;
        r.group_segment_size = nvd_rd<uint32_t>(d + amdt::KD_GROUP_SEGMENT_FIXED_SIZE);
        r.private_segment_size = nvd_rd<uint32_t>(d + amdt::KD_PRIVATE_SEGMENT_FIXED_SIZE);
        r.k.kernargs_segment_size = nvd_rd<uint32_t>(d + amdt::KD_KERNARG_SIZE);
        const uint32_t lds_size = ((r.group_segment_size + 511) / 512) & 0x1FF;
        if (lds_size > (p.lds_size_in_kb * 1024) / 512) return "Too many resources requested: group_segment_size (" + name + ")";
        r.kernel_code_properties = nvd_rd<uint16_t>(d + amdt::KD_KERNEL_CODE_PROPERTIES);
        r.k.wave32 = (r.kernel_code_properties & amdt::KERNEL_CODE_PROPERTIES_WAVE32) == amdt::KERNEL_CODE_PROPERTIES_WAVE32;
        r.k.rsrc1 = nvd_rd<uint32_t>(d + amdt::KD_COMPUTE_PGM_RSRC1) | (p.target_major == 11 ? (1u << 20) : 0u);   // priv on gfx11 (cwsr)
        r.k.rsrc2 = nvd_rd<uint32_t>(d + amdt::KD_COMPUTE_PGM_RSRC2) | (lds_size << 15);
        r.k.rsrc3 = nvd_rd<uint32_t>(d + amdt::KD_COMPUTE_PGM_RSRC3);
        r.aql_prog_addr = lib_va + off;
        r.k.prog_addr = lib_va + off + (uint64_t)nvd_rd<int64_t>(d + amdt::KD_KERNEL_CODE_ENTRY_BYTE_OFFSET);
        const uint32_t refused = amdt::AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER | amdt::AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR |
                                 amdt::AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_QUEUE_PTR | amdt::AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_ID;
        if (r.kernel_code_properties & refused)
            return "kernel " + name + " needs the dispatch_ptr, queue_ptr, dispatch_id or private segment SGPRs (kernel_code_properties " +
                   std::to_string(r.kernel_code_properties) + "), which the C++ exec does not set up";
        r.k.kernargs_alloc_size = r.k.kernargs_segment_size;   // + the dispatch packet, which only dispatch_ptr kernels have
        kernels[name] = r;
    }
    return "";
}

// AMDDevice._ensure_has_local_memory (gfx11, wave64 lanes, 256-byte alignment) for private_segment_size: the scratch
// buffer's size and COMPUTE_TMPRING_SIZE. "" or why not (a value tinygrad's bitfield would not hold).
inline std::string amd_scratch(const AMDProps& p, uint32_t private_segment_size, uint64_t& scratch_size, uint32_t& tmpring_size) {
    const uint64_t lanes_per_wave = 64, mem_alignment_size = p.target_major != 9 ? 256 : 1024;
    const uint64_t a = mem_alignment_size / lanes_per_wave;
    const uint64_t size_per_thread = (private_segment_size + a - 1) / a * a;
    const uint64_t size_per_xcc = size_per_thread * lanes_per_wave * p.max_slots_scratch_cu * p.cu_cnt;
    scratch_size = size_per_xcc * p.xccs;
    const uint64_t max_scratch_waves = (uint64_t)p.cu_cnt * p.max_slots_scratch_cu * p.xccs;
    const uint64_t wave_scratch = (lanes_per_wave * size_per_thread + mem_alignment_size - 1) / mem_alignment_size;
    if (wave_scratch == 0 || p.se_cnt == 0) return "no scratch to size";
    const uint64_t num_waves = (size_per_xcc / (wave_scratch * mem_alignment_size)) / (p.target_major != 9 ? p.se_cnt : 1);
    const uint64_t waves = num_waves < max_scratch_waves ? num_waves : max_scratch_waves;
    if (waves >> (amdt::TMPRING_GFX11_WAVES.hi - amdt::TMPRING_GFX11_WAVES.lo + 1) ||
        wave_scratch >> (amdt::TMPRING_GFX11_WAVESIZE.hi - amdt::TMPRING_GFX11_WAVESIZE.lo + 1))
        return "COMPUTE_TMPRING_SIZE does not hold " + std::to_string(waves) + " waves of " + std::to_string(wave_scratch);
    tmpring_size = (uint32_t)(amdt::encode(amdt::TMPRING_GFX11_WAVES, waves) | amdt::encode(amdt::TMPRING_GFX11_WAVESIZE, wave_scratch));
    return "";
}

}  // namespace tinygpu_device

#endif  // LIBHMSBEAGLE_GPU_TINYGPUHYBRIDAMDPROGRAM_H
