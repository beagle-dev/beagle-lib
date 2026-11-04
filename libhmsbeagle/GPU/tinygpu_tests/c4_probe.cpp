// test_c4_firmware.py's probe of TinyGPUFirmware.h and nvd_elf_load (TODO.md plan step C4). One JSON object per line.
//   c4_probe locate            every manifest entry: where tg_fw_locate found it, its size and sha256, or its error
//   c4_probe elf FILE [ALIGN]  nvd_elf_load's sections, image and relocations
#include <cstdio>
#include <string>
#include <vector>

#include "libhmsbeagle/GPU/TinyGPUFirmware.h"
#include "libhmsbeagle/GPU/TinyGPUNVProgram.h"

using namespace tinygpu_device;

static std::string js(const std::string& s) {
    std::string o = "\"";
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += (char)c; }
        else if (c == '\n') o += "\\n";
        else if (c < 0x20) { char b[8]; snprintf(b, sizeof(b), "\\u%04x", c); o += b; }
        else o += (char)c;
    }
    return o + "\"";
}

int main(int argc, char** argv) {
    std::string cmd = argc > 1 ? argv[1] : "";
    if (cmd == "locate") {
        for (const nvfw::TGFirmware& fw : nvfw::kFirmware) {
            TGFirmwareFile f;
            std::string err = tg_fw_locate(fw, f);
            if (err.empty())
                printf("{\"chip\": %s, \"role\": %s, \"path\": %s, \"size\": %zu, \"sha256\": %s}\n", js(fw.chip).c_str(), js(fw.role).c_str(),
                       js(f.path()).c_str(), f.size(), js(tg_sha256_hex(f.data(), f.size())).c_str());
            else
                printf("{\"chip\": %s, \"role\": %s, \"error\": %s}\n", js(fw.chip).c_str(), js(fw.role).c_str(), js(err).c_str());
        }
        return 0;
    }
    if (cmd == "elf" && argc > 2) {
        TGFirmwareFile f;
        std::string why;
        if (!f.map(argv[2], why)) { printf("{\"error\": %s}\n", js(why).c_str()); return 1; }
        NVDElf elf;
        std::string err = nvd_elf_load(f.data(), f.size(), argc > 3 ? strtoull(argv[3], nullptr, 0) : 1, elf);
        if (!err.empty()) { printf("{\"error\": %s}\n", js(err).c_str()); return 0; }
        for (const NVDElfSection& s : elf.sections)
            printf("{\"section\": %s, \"type\": %u, \"addr\": %llu, \"offset\": %llu, \"size\": %llu, \"addralign\": %llu, \"entsize\": %llu, "
                   "\"link\": %u}\n", js(s.name).c_str(), s.type, (unsigned long long)s.addr, (unsigned long long)s.offset,
                   (unsigned long long)s.size, (unsigned long long)s.addralign, (unsigned long long)s.entsize, s.link);
        printf("{\"image_size\": %zu, \"image_sha256\": %s}\n", elf.image.size(), js(tg_sha256_hex(elf.image.data(), elf.image.size())).c_str());
        for (const NVDElfReloc& r : elf.relocs)
            printf("{\"reloc\": [%llu, %llu, %u, %lld]}\n", (unsigned long long)r.image_off, (unsigned long long)r.sym_off, r.type,
                   (long long)r.addend);
        return 0;
    }
    fprintf(stderr, "usage: c4_probe locate | elf FILE [ALIGN]\n");
    return 2;
}
