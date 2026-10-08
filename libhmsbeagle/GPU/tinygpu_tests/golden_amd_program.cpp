// Loads golden_amd_program.hsaco with TinyGPUAMDProgram.h alone (see golden_amd_program.py): one line per kernel,
// the relocated image, and the scratch sizing for each private size given.
//   golden_amd_program <dir> <lib_va> <cu_cnt> <se_cnt> <xccs> <max_slots_scratch_cu> <lds_size_in_kb> <target_major> <private size>...
#include "libhmsbeagle/GPU/TinyGPUAMDProgram.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
using namespace tinygpu_device;

int main(int argc, char** argv) {
    const std::string dir = argv[1];
    const uint64_t lib_va = strtoull(argv[2], nullptr, 10);
    AMDProps p;
    p.cu_cnt = atoi(argv[3]); p.se_cnt = atoi(argv[4]); p.xccs = atoi(argv[5]); p.max_slots_scratch_cu = atoi(argv[6]); p.lds_size_in_kb = atoi(argv[7]);
    p.target_major = atoi(argv[8]);
    std::ifstream f(dir + "/golden_amd_program.hsaco", std::ios::binary);
    std::stringstream ss; ss << f.rdbuf();
    const std::string hsaco = ss.str();
    std::vector<uint8_t> image;
    std::map<std::string, AMDProgramRecord> kernels;
    std::string err = amd_load_hsaco((const uint8_t*)hsaco.data(), hsaco.size(), lib_va, p, image, kernels);
    if (!err.empty()) { printf("%s\n", err.c_str()); return 1; }
    for (const auto& [name, r] : kernels)
        printf("K %s %llu %u %u %u %u %d %u %u %u %llu %llu\n", name.c_str(), (unsigned long long)r.kd_off, r.group_segment_size,
               r.private_segment_size, r.k.kernargs_segment_size, r.k.kernargs_alloc_size, r.k.wave32 ? 1 : 0, r.k.rsrc1, r.k.rsrc2, r.k.rsrc3,
               (unsigned long long)r.k.prog_addr, (unsigned long long)r.aql_prog_addr);
    std::ofstream(dir + "/golden_amd_program_image.bin", std::ios::binary).write((const char*)image.data(), image.size());
    for (int i = 9; i < argc; ++i) {
        uint64_t size = 0; uint32_t tmpring = 0;
        err = amd_scratch(p, (uint32_t)atoi(argv[i]), size, tmpring);
        if (!err.empty()) { printf("%s\n", err.c_str()); return 1; }
        printf("S %s %llu %u\n", argv[i], (unsigned long long)size, tmpring);
    }
    return 0;
}
