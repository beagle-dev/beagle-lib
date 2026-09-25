// C++ side of golden_runtime.py: parses the boot-only handoff and computes each case with TinyGPUHybridNVProgram.h.
#include "libhmsbeagle/GPU/TinyGPUHybridNVProgram.h"
#include <cstdio>
#include <fstream>
#include <sstream>
using namespace tinygpu_device;

static std::string slurp(const std::string& p) { std::ifstream f(p, std::ios::binary); std::stringstream s; s << f.rdbuf(); return s.str(); }

int main(int, char** argv) {
    std::string dir = argv[1];
    uint64_t sig_va = std::stoull(argv[2]), lm_va = std::stoull(argv[3]);
    std::string js = slurp(dir + "/golden_rt_handoff.json");
    NVDHandoff h;
    NVDRuntime rt;
    std::string err = nvd_parse_handoff(js, {}, h);
    if (err.empty()) err = nvd_parse_runtime(js, rt);
    if (!err.empty()) { fprintf(stderr, "parse: %s\n", err.c_str()); return 1; }
    if (!h.kernels.empty()) { fprintf(stderr, "boot-only handoff has kernels\n"); return 1; }
    err = nvd_check_tables(h, rt.compute_class);
    printf("%s\n", err.empty() ? "TABLES OK" : err.c_str());
    NVDHandoff bad = h;
    bad.q_dep_enable[1] += 1;
    printf("%s\n", nvd_check_tables(bad, rt.compute_class).empty() ? "TABLES PERTURBED ACCEPTED" : "TABLES PERTURBED REJECTED");

    std::ifstream cases(dir + "/golden_rt_cases.txt");
    std::string tag;
    while (cases >> tag) {
        if (tag == "LM") {  // NVProgram.__init__ per program: slm_per_thread = running max, then the one setup submit
            NVDRuntime t = rt;
            uint64_t timeline;
            int n;
            cases >> t.num_gpcs >> t.num_tpc_per_gpc >> t.num_sm_per_tpc >> t.max_warps_per_sm >> timeline >> n;
            uint32_t slm = 0;
            for (int i = 0; i < n; ++i) { uint64_t r; cases >> r; slm = std::max<uint32_t>(slm, (uint32_t)nvd_round_up(r, 32)); }
            uint64_t tpc_bytes = 0, size = nvd_local_mem_size(t, slm, tpc_bytes);
            std::vector<uint32_t> pb;
            nvd_push_wait(pb, h, sig_va, timeline - 1);
            nvd_push_setup_local_mem(pb, h, lm_va, tpc_bytes);
            nvd_push_signal(pb, h, sig_va, timeline);
            printf("%u %llu", slm, (unsigned long long)size);
            for (uint32_t w : pb) printf(" %u", w);
            printf("\n");
        } else if (tag == "LM2") {  // plan step P5: a second instance's programs on the GPU the first set up; slm only grows
            NVDRuntime t = rt;
            cases >> t.num_gpcs >> t.num_tpc_per_gpc >> t.num_sm_per_tpc >> t.max_warps_per_sm;
            uint32_t slm = 0;
            for (int inst = 0; inst < 2; ++inst) {
                uint64_t timeline;
                int n;
                cases >> timeline >> n;
                uint32_t need = slm;
                for (int i = 0; i < n; ++i) { uint64_t r; cases >> r; need = std::max<uint32_t>(need, (uint32_t)nvd_round_up(r, 32)); }
                if (need <= slm) { printf("%u NONE\n", slm); continue; }
                slm = need;
                uint64_t tpc_bytes = 0, size = nvd_local_mem_size(t, slm, tpc_bytes);
                std::vector<uint32_t> pb;
                nvd_push_wait(pb, h, sig_va, timeline - 1);
                nvd_push_setup_local_mem(pb, h, lm_va, tpc_bytes);
                nvd_push_signal(pb, h, sig_va, timeline);
                printf("%u %llu", slm, (unsigned long long)size);
                for (uint32_t w : pb) printf(" %u", w);
                printf("\n");
            }
        } else if (tag == "POOL") {
            int n;
            cases >> n;
            uint64_t pos = 0;
            for (int i = 0; i < n; ++i) {
                uint64_t size;
                cases >> size;
                uint64_t va = nvd_pool_alloc(rt.pool, pos, size);
                printf("%llu %llu\n", (unsigned long long)va, (unsigned long long)pos);
            }
        }
    }
    return 0;
}
