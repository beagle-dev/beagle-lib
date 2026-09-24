// Encodes golden_batch.txt with TinyGPUHybridNVDispatch.h from the handoff alone (see golden_encode.py).
#include "libhmsbeagle/GPU/TinyGPUHybridNVDispatch.h"
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
using namespace tinygpu_device;

static std::string slurp(const std::string& p) { std::ifstream f(p, std::ios::binary); std::stringstream s; s << f.rdbuf(); return s.str(); }

int main(int, char** argv) {
    std::string dir = argv[1];
    std::string js = slurp(dir + "/golden_handoff.json"), b = slurp(dir + "/golden_blob.bin");
    std::vector<uint8_t> blob(b.begin(), b.end());
    NVDHandoff h;
    std::string err = nvd_parse_handoff(js, blob, h);
    if (!err.empty()) { fprintf(stderr, "parse: %s\n", err.c_str()); return 1; }

    std::ifstream batch(dir + "/golden_batch.txt");
    std::vector<uint32_t> pb, pbc;
    std::vector<uint8_t> kargs;
    uint64_t sig_va = 0, value = 0, kargs_va = 0, kargs_size = 0, pos = 0;
    uint8_t* prev = nullptr;
    std::string tag;
    while (batch >> tag) {
        if (tag == "SIG") batch >> sig_va >> value;
        else if (tag == "KARGS") {
            batch >> kargs_va >> kargs_size;
            kargs.assign(kargs_size, 0);
            nvd_push_wait(pb, h, sig_va, value - 1);
            nvd_push_invalidate(pb, h);
        } else if (tag == "L") {
            std::string name; uint32_t grid[3], block[3]; int nptr, nint;
            batch >> name >> grid[0] >> grid[1] >> grid[2] >> block[0] >> block[1] >> block[2] >> nptr;
            std::vector<uint64_t> ptrs(nptr); for (auto& p : ptrs) batch >> p;
            batch >> nint;
            std::vector<uint32_t> ints(nint); for (auto& i : ints) batch >> i;
            const NVDKernel& k = h.kernels.at(name);
            uint64_t off = (pos + 255) & ~255ull; pos = off + k.slot_size;
            uint8_t* qmd = nvd_encode_launch(h, k, kargs.data() + off, kargs_va + off, grid, block, ptrs.data(), nptr, ints.data(), nint);
            uint64_t qmd_va = kargs_va + off + k.qmd_off;
            if (!prev) nvd_push_pcas(pb, h, qmd_va); else nvd_chain(h, prev, qmd_va);
            prev = qmd;
        } else if (tag == "C") {
            uint64_t dst, src, n, vw, vs;
            batch >> dst >> src >> n >> vw >> vs;
            nvd_push_wait(pbc, h, sig_va, vw);
            nvd_push_copy(pbc, h, dst, src, n);
            nvd_push_dma_signal(pbc, h, sig_va, vs);
        }
    }
    nvd_qmd_release(h, prev, sig_va, value);

    std::ofstream oc(dir + "/golden_out_compute.txt"); for (uint32_t w : pb) oc << w << "\n";
    std::ofstream op(dir + "/golden_out_copy.txt"); for (uint32_t w : pbc) op << w << "\n";
    std::ofstream ok(dir + "/golden_out_kargs.bin", std::ios::binary); ok.write((const char*)kargs.data(), kargs.size());
    return 0;
}
