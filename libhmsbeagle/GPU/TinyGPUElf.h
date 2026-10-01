/*
 * TinyGPUElf.h
 *
 * tinygrad's elf_loader (tinygrad/runtime/support/elf.py) in C++, shared by the NV program loader
 * (TinyGPUHybridNVProgram.h: cubins, and the Blackwell FMC image, TODO.md plan step C4) and the AMD HSACO loader
 * (TinyGPUHybridAMDProgram.h, plan step A1d). Moved here unchanged from TinyGPUHybridNVProgram.h; the names keep the
 * NV prefix of the path that ported it first.
 */

#ifndef LIBHMSBEAGLE_GPU_TINYGPUELF_H
#define LIBHMSBEAGLE_GPU_TINYGPUELF_H

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace tinygpu_device {

// ── elf_loader(blob, force_section_align) ──────────────────────────────────

struct NVDElfSection {
    std::string name;
    uint32_t type = 0, link = 0;
    uint64_t addr = 0, size = 0, offset = 0, addralign = 0, entsize = 0;  // addr as elf_loader assigns it
};

struct NVDElfReloc { uint64_t image_off, sym_off; uint32_t type; int64_t addend; };

struct NVDElf {
    const uint8_t* blob = nullptr;
    size_t blob_size = 0;
    std::vector<NVDElfSection> sections;
    std::vector<uint8_t> image;      // PROGBITS laid out as elf_loader does, before relocation
    std::vector<NVDElfReloc> relocs;
    const uint8_t* content(const NVDElfSection& sh) const { return blob + sh.offset; }
};

template <typename T> static inline T nvd_rd(const uint8_t* p) { T v; memcpy(&v, p, sizeof(T)); return v; }

static inline std::string nvd_cstr(const uint8_t* tab, size_t tab_size, uint64_t idx) {
    if (idx >= tab_size) return "";
    const char* s = (const char*)tab + idx;
    return std::string(s, strnlen(s, tab_size - idx));
}

// Returns an empty string on success. ELF64 (cubins) and ELF32 (the Blackwell FMC image, TODO.md plan step C4), each read
// through the layout of elf_loader's libc.Elf64_* or Elf32_* structs.
static inline std::string nvd_elf_load(const uint8_t* blob, size_t n, uint64_t force_section_align, NVDElf& elf) {
    enum { SHT_PROGBITS = 1, SHT_SYMTAB = 2, SHT_RELA = 4, SHT_REL = 9 };
    if (n < 52 || memcmp(blob, "\x7f" "ELF", 4) != 0) return "blob is not an ELF, missing magic bytes";
    if (blob[4] != 1 && blob[4] != 2) return "not an ELF32 or ELF64";
    const bool e64 = blob[4] == 2;
    if (e64 && n < 64) return "truncated ELF header";
    auto addr_t = [&](const uint8_t* p) { return e64 ? nvd_rd<uint64_t>(p) : (uint64_t)nvd_rd<uint32_t>(p); };  // Elf*_Addr, _Off, _Xword
    elf.blob = blob;
    elf.blob_size = n;
    const uint64_t shentsize = e64 ? 64 : 40;
    uint64_t shoff = addr_t(blob + (e64 ? 0x28 : 0x20));
    uint16_t shnum = nvd_rd<uint16_t>(blob + (e64 ? 0x3c : 0x30)), shstrndx = nvd_rd<uint16_t>(blob + (e64 ? 0x3e : 0x32));
    if (shoff + (uint64_t)shnum * shentsize > n || shstrndx >= shnum) return "truncated section headers";
    elf.sections.resize(shnum);
    for (uint16_t i = 0; i < shnum; ++i) {
        const uint8_t* h = blob + shoff + (uint64_t)i * shentsize;
        const uint64_t w = e64 ? 8 : 4;   // sh_flags onwards: addr, offset, size, link, info, addralign, entsize
        NVDElfSection& s = elf.sections[i];
        s.type = nvd_rd<uint32_t>(h + 4);                       s.addr = addr_t(h + 8 + w);
        s.offset = addr_t(h + 8 + 2 * w);                       s.size = addr_t(h + 8 + 3 * w);
        s.link = nvd_rd<uint32_t>(h + 8 + 4 * w);               s.addralign = addr_t(h + 16 + 4 * w);
        s.entsize = addr_t(h + 16 + 5 * w);
        s.name = std::to_string(nvd_rd<uint32_t>(h));  // name offset for now, resolved below
        if (s.type != 8 /* SHT_NOBITS */ && s.offset + s.size > n) return "section extends past end of ELF";
    }
    const NVDElfSection& shstr = elf.sections[shstrndx];
    for (NVDElfSection& s : elf.sections) s.name = nvd_cstr(blob + shstr.offset, shstr.size, std::stoull(s.name));

    // Prealloc image for all fixed addresses, then append the rest aligned.
    uint64_t fixed_end = 0;
    for (const NVDElfSection& s : elf.sections)
        if (s.type == SHT_PROGBITS && s.addr != 0) fixed_end = std::max(fixed_end, s.addr + s.size);
    elf.image.assign(fixed_end, 0);
    for (NVDElfSection& s : elf.sections) {
        if (s.type != SHT_PROGBITS) continue;
        if (s.addr != 0) { memcpy(elf.image.data() + s.addr, elf.content(s), s.size); continue; }
        uint64_t align = std::max(s.addralign, force_section_align);
        elf.image.resize(elf.image.size() + (align - elf.image.size() % align) % align, 0);
        elf.image.insert(elf.image.end(), elf.content(s), elf.content(s) + s.size);
        s.addr = elf.image.size() - s.size;
    }

    // Relocations: SHT_REL sections first, then SHT_RELA, each in section order.
    const NVDElfSection* symtab = nullptr;
    for (const NVDElfSection& s : elf.sections) if (s.type == SHT_SYMTAB) { symtab = &s; break; }
    for (uint32_t kind : { (uint32_t)SHT_REL, (uint32_t)SHT_RELA }) {
        for (const NVDElfSection& s : elf.sections) {
            if (s.type != kind) continue;
            std::string target = s.name.substr(kind == SHT_REL ? 4 : 5);
            if (target == ".eh_frame") continue;
            auto t = std::find_if(elf.sections.begin(), elf.sections.end(), [&](const NVDElfSection& x) { return x.name == target; });
            if (t == elf.sections.end() || !symtab || s.entsize == 0) return "relocation section " + s.name + " without target or symtab";
            for (uint64_t off = 0; off + s.entsize <= s.size; off += s.entsize) {
                const uint8_t* r = elf.content(s) + off;   // Elf*_Rel/Rela: r_offset, r_info, r_addend
                uint64_t r_offset = addr_t(r), r_info = addr_t(r + (e64 ? 8 : 4));
                int64_t addend = kind != SHT_RELA ? 0 : e64 ? nvd_rd<int64_t>(r + 16) : (int64_t)nvd_rd<int32_t>(r + 8);
                uint64_t sym_idx = e64 ? r_info >> 32 : r_info >> 8;   // ELF64_R_SYM, ELF32_R_SYM
                if (symtab->entsize == 0 || (sym_idx + 1) * symtab->entsize > symtab->size) return "relocation symbol out of range";
                const uint8_t* sym = elf.content(*symtab) + sym_idx * symtab->entsize;
                uint16_t shndx = nvd_rd<uint16_t>(sym + (e64 ? 6 : 14));   // Elf*_Sym: st_shndx, then st_value
                if (shndx == 0 || shndx >= elf.sections.size()) return "relocation against an undefined symbol";
                elf.relocs.push_back({ t->addr + r_offset, elf.sections[shndx].addr + addr_t(sym + (e64 ? 8 : 4)),
                                       (uint32_t)(e64 ? r_info & 0xffffffffu : r_info & 0xff), addend });   // *_R_TYPE
            }
        }
    }
    return "";
}

}  // namespace tinygpu_device

#endif  // LIBHMSBEAGLE_GPU_TINYGPUELF_H
