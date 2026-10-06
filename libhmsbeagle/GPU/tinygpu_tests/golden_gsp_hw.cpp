// C++ side of golden_gsp_hw.py (TODO.md plan step C8): TinyGPUNVRM.h's nv_gsp_init_hw (NV_GSP.init_hw with
// nv_init_helper's patch 3, and init_golden_image) over golden_gsp_hw.py's fake TinyGPU.app (APL_REMOTE_SOCK), with the GSP
// queues in the file GOLDEN_QUEUES and the memory manager restored from nv_dispatch_daemon.py's _mm_export, and prints the
// result lines golden_gsp_hw.py prints for tinygrad's own.
//   golden_gsp_hw EXPORT STATE
// STATE, one "key value" per line: seq, gpfifo_class, compute_class, dma_class, viddec_class, gb2, fmc_boot, chip_id, libos,
//   wait_ms, rpc_timeout_ms.
#include "libhmsbeagle/GPU/TinyGPUNVRM.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

using namespace tinygpu_device;

static std::string words(const std::vector<uint64_t>& w) {
    std::string s;
    for (uint64_t x : w) s += " " + std::to_string(x);
    return s;
}

int main(int argc, char** argv) {
    if (argc != 3) return fprintf(stderr, "usage: golden_gsp_hw EXPORT STATE\n"), 2;
    std::map<std::string, uint64_t> kv;
    {
        std::ifstream sf(argv[2]);
        for (std::string k, v; sf >> k >> v;) kv[k] = strtoull(v.c_str(), nullptr, 0);
    }
    auto get = [&](const char* k) { return kv.at(k); };
    TGTransport t;
    std::string err = t.open();
    if (!err.empty()) return fprintf(stderr, "transport: %s\n", err.c_str()), 1;
    NVMemState st;
    {
        std::ifstream jf(argv[1]);
        std::stringstream js;
        js << jf.rdbuf();
        err = nv_mm_import(js.str(), &t, st);
        if (!err.empty()) return fprintf(stderr, "import: %s\n", err.c_str()), 1;
        t.seed_bar(0, 16 << 20);   // the daemon mapped BAR0 too
    }
    const char* qf = getenv("GOLDEN_QUEUES");
    int fd = qf ? open(qf, O_RDWR) : -1;
    uint8_t* queues = fd < 0 ? nullptr : (uint8_t*)mmap(nullptr, 0x81000, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (fd >= 0) close(fd);
    if (!queues || queues == MAP_FAILED) return fprintf(stderr, "GOLDEN_QUEUES\n"), 1;

    NVBar0 bar0{&t};
    NVFalcon flcn(bar0, (uint32_t)get("chip_id"));
    flcn.wait_ms = (int)get("wait_ms");
    flcn.sleep = [](double s) { if (s >= 1) printf("sleep %g\n", s); else nv_sleep(s); };   // the 20 s one is recorded, not slept
    try {
        NVGsp gsp(bar0, flcn, queues, 0x1000, 0x41000, 0x40000, get("libos"), (uint32_t)get("seq"), flcn.wait_ms);
        gsp.rpc_timeout_ms = (int)get("rpc_timeout_ms");
        NVRMClient rm(gsp, *st.mm);
        rm.priv_root = 0;   // NV_GSP as init_sw leaves it: init_hw sets priv_root
        rm.gpfifo_class = (uint32_t)get("gpfifo_class");
        rm.compute_class = (uint32_t)get("compute_class");
        rm.dma_class = (uint32_t)get("dma_class");
        rm.viddec_class = (uint32_t)get("viddec_class");
        rm.gb2 = get("gb2") != 0;
        try {
            nv_gsp_init_hw(rm, get("fmc_boot") != 0);
        } catch (const TGPyError& x) {
            printf("error %s\n", x.py().c_str());
        } catch (const NVError& x) {
            printf("error %s\n", x.py().c_str());
        }
        printf("rm priv_root %u next_handle %u device %u subdevice %u seq %u\n", rm.priv_root, rm.next_handle, rm.device, rm.subdevice,
               gsp.cmd_q.seq);
        std::string s;
        for (auto& [k, v] : rm.runlists) s += " " + std::to_string(k) + ":" + std::to_string(v);
        printf("rm runlists%s\n", s.c_str());
        s.clear();
        for (auto& [k, v] : rm.chan_runlists) s += " " + std::to_string(k) + ":" + std::to_string(v);
        printf("rm chan_runlists%s\n", s.c_str());
        s.clear();
        for (auto& [i, b] : rm.grctx_bufs)
            s += " " + std::to_string(i) + ":" + std::to_string(b.size) + ":" + std::to_string(b.phys) + ":" + std::to_string(b.virt) + ":" +
                 std::to_string(b.local);
        printf("rm grctx%s\n", s.c_str());
    } catch (const NVError& x) {   // the status queue's header never set
        printf("error %s\n", x.py().c_str());
    }
    printf("state pa%s\n", words(st.mm->pa_allocator.save()).c_str());
    printf("state va%s\n", words(st.va.save()).c_str());
    t.close();
    return 0;
}
