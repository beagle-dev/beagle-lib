// Parses cmd_handoff's reply and attaches its fds with TinyGPUHybridAMDRuntime.h alone (see golden_amd_handoff.py), then prints
// what the C++ side reads through each mapping.   golden_amd_handoff <reply.json> <fd>...
#include "libhmsbeagle/GPU/TinyGPUHybridAMDRuntime.h"
#include <cstdio>
#include <fstream>
#include <sstream>
using namespace tinygpu_device;

int main(int argc, char** argv) {
    std::ifstream f(argv[1]);
    std::stringstream ss; ss << f.rdbuf();
    AMDHandoff h;
    std::string err = amd_parse_handoff(ss.str(), h);
    if (!err.empty()) { printf("%s\n", err.c_str()); return 1; }
    if ((int)h.nmaps != argc - 2) { printf("%llu mappings but %d fds\n", (unsigned long long)h.nmaps, argc - 2); return 1; }
    int fds[8];
    for (int i = 0; i < argc - 2; ++i) fds[i] = atoi(argv[2 + i]);
    TGTransport tg;
    AMDRuntime rt;
    err = amd_runtime_attach(rt, h, fds, tg);
    if (!err.empty()) { printf("%s\n", err.c_str()); return 1; }
    auto u = [](uint64_t v) { return (unsigned long long)v; };
    printf("compute_ring0 %u\ncompute_rptr %llu\ncompute_wptr %llu\n", rt.compute.ring[0], u(*rt.compute.rptr), u(*rt.compute.wptr));
    printf("sdma_ring0 %u\nsdma_rptr %llu\nsdma_wptr %llu\n", rt.sdma.ring[0], u(*rt.sdma.rptr), u(*rt.sdma.wptr));
    printf("signal %llu\nshadow %llu\nkargs0 %u\nstaging0 %u\n", u(*rt.signal), u(*rt.shadow), *(uint32_t*)rt.kargs, *(uint32_t*)rt.staging.host);
    printf("pool_size %llu\ntimeline_value %llu\nreg_ih_wptr %llu\nreg_hdp_remap %llu\nreg_fault_status %llu\ncompute_doorbell %llu\nsdma_put %llu\n",
           u(h.pool_size), u(rt.timeline_value), u(h.reg_ih_wptr), u(h.reg_hdp_remap), u(h.reg_fault_status), u(rt.compute.doorbell), u(rt.sdma.put));
    amd_runtime_detach(rt);
    return 0;
}
