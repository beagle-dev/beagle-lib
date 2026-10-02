// The embedded cubins through TinyGPUHybridNVCubins.h, linked from the generated kernels/BeagleTinyGPU_cubins.S as the
// plugin links them (see test_c1_cubins.py):
//     golden_cubins <dir> [<cubin>...]
// selects and loads every entry of kTinyGPUNVCubins (single and double precision: plan step C16) and writes its bytes and
// kernel names to <dir>, checks the refusals (a state count, an architecture, an entry holding another architecture's
// cubin), then prints the SM nvd_elf_sm reads from each extra file.
#include "libhmsbeagle/GPU/TinyGPUHybridNVCubins.h"
#include <cstdio>
#include <fstream>
#include <iterator>
using namespace tinygpu_device;

static const size_t kN = sizeof(kTinyGPUNVCubins) / sizeof(kTinyGPUNVCubins[0]);
static int fails = 0;
static void expect(bool ok, const std::string& what) { printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str()); fails += !ok; }

// nvd_find_cubin must refuse, with a message containing want
static void refused(const TinyGPUNVCubin* table, size_t n, int states, bool dp, const std::string& arch, const std::string& want) {
    const TinyGPUNVCubin* got = nullptr;
    std::string err = nvd_find_cubin(table, n, states, dp, arch, got);
    expect(!got && err.find(want) != std::string::npos, std::to_string(states) + (dp ? " DP " : " ") + arch + " refused: " + err);
}

int main(int argc, char** argv) {
    std::string dir = argv[1], archs;
    for (const TinyGPUNVCubin& c : kTinyGPUNVCubins) {
        const TinyGPUNVCubin* got = nullptr;
        NVDElf elf;
        std::string err = nvd_find_cubin(kTinyGPUNVCubins, kN, c.states, c.dp, c.arch, got);
        if (err.empty()) err = nvd_elf_load(got->begin, (size_t)(got->end - got->begin), 128, elf);
        std::string tag = (c.dp ? "DP_" : "SP_") + std::to_string(c.states) + "_" + c.arch;
        expect(err.empty() && got == &c, tag + ": " + std::to_string(c.end - c.begin) + " bytes" + (err.empty() ? "" : ": " + err));
        std::ofstream(dir + "/" + tag + ".cubin", std::ios::binary).write((const char*)c.begin, c.end - c.begin);
        std::ofstream names(dir + "/" + tag + ".names");
        for (const std::string& k : nvd_kernel_names(elf)) names << k << "\n";
        if (c.states == kTinyGPUNVCubins[0].states && !c.dp) archs += std::string(archs.empty() ? "" : ", ") + c.arch;
    }
    const TinyGPUNVCubin& first = kTinyGPUNVCubins[0];
    refused(kTinyGPUNVCubins, kN, 512, false, first.arch, "no embedded cubin for 512 states in single precision");
    refused(kTinyGPUNVCubins, kN, 512, true, first.arch, "no embedded cubin for 512 states in double precision");
    refused(kTinyGPUNVCubins, kN, first.states, false, "sm_75", "no embedded cubin for this GPU's architecture (sm_75); this build has " + archs);
    // an entry holding another architecture's cubin (a build mix-up): the ELF header check refuses it
    for (const TinyGPUNVCubin& a : kTinyGPUNVCubins)
        for (const TinyGPUNVCubin& b : kTinyGPUNVCubins)
            if (a.states == b.states && a.dp == b.dp && std::string(a.arch) != b.arch) {
                TinyGPUNVCubin swapped = { a.states, a.dp, a.arch, b.begin, b.end };
                refused(&swapped, 1, a.states, a.dp, a.arch, std::string("is built for SM ") + (b.arch + 3));
            }
    for (int i = 2; i < argc; ++i) {
        std::ifstream f(argv[i], std::ios::binary);
        std::string blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        printf("SM %u %s\n", nvd_elf_sm((const uint8_t*)blob.data(), blob.size()), argv[i]);
    }
    printf("%zu embedded cubins (ptxas %s): %s\n", kN, TINYGPU_CUBINS_STAMP, fails ? "FAILED" : "all passed");
    return fails ? 1 : 0;
}
