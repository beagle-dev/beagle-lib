// Loads a real cubin with TinyGPUNVProgram.h and writes build_handoff-format records (see golden_program.py).
#include "libhmsbeagle/GPU/TinyGPUNVProgram.h"
#include <cstdio>
#include <fstream>
#include <sstream>
using namespace tinygpu_device;

int main(int, char** argv) {
    std::ifstream f(argv[1], std::ios::binary);
    std::string blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::ifstream nf(argv[2]);
    std::vector<std::string> names;
    for (std::string n; nf >> n; ) names.push_back(n);
    NVDProgramParams p;
    p.compute_class = (uint32_t)std::stoul(argv[3]);
    p.lib_va = std::stoull(argv[4]);
    p.sass_version = (uint32_t)std::stoul(argv[5]);
    std::string dir = argv[6];

    NVDElf elf;
    std::string err = nvd_elf_load((const uint8_t*)blob.data(), blob.size(), 128, elf);
    if (!err.empty()) { fprintf(stderr, "elf: %s\n", err.c_str()); return 1; }
    std::vector<uint8_t> image;
    err = nvd_relocate(elf, p.lib_va, image);
    if (!err.empty()) { fprintf(stderr, "reloc: %s\n", err.c_str()); return 1; }

    // _ensure_has_local_memory over every kernel first: slm_per_thread = the running max
    std::vector<NVDProgramUsage> usage(names.size());
    for (size_t i = 0; i < names.size(); ++i) {
        if (!(err = nvd_program_usage(elf, names[i], usage[i])).empty()) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
        p.slm_per_thread = std::max<uint32_t>(p.slm_per_thread, (uint32_t)nvd_round_up(usage[i].lcmem, 32));
    }
    std::string out;
    for (size_t i = 0; i < names.size(); ++i) {
        NVDKernel k;
        if (!(err = nvd_load_program(elf, names[i], p, usage[i], k)).empty()) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
        uint32_t hdr[7] = { (uint32_t)k.name.size(), k.qmd_off, k.slot_size, k.prefix_words, k.dims_b, k.dims_g, k.max_threads };
        out.append((const char*)hdr, sizeof(hdr));
        out += k.name;
        out.append((const char*)k.qmd.data(), k.qmd.size());
        out.append((const char*)k.prefix.data(), k.prefix.size() * 4);
    }
    std::ofstream(dir + "/golden_out_blob.bin", std::ios::binary) << out;
    std::ofstream(dir + "/golden_out_image.bin", std::ios::binary).write((const char*)image.data(), image.size());
    std::ofstream(dir + "/golden_out_slm.txt") << p.slm_per_thread;
    return 0;
}
