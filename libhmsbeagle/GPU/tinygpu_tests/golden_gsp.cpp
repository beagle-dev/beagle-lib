// C++ side of golden_gsp.py (TODO.md plan step C5): TinyGPUHybridNVGsp.h and TinyGPUHybridNVFalcon.h run one scenario
// against golden_gsp.py's scripted TinyGPU.app, through TinyGPUTransport.h (APL_REMOTE_SOCK), with the GSP queues in the
// file GOLDEN_QUEUES, and print the result lines golden_gsp.py prints for tinygrad's and nv_init_helper's code. cot=1: GB20x's
// COT boot (plan step B2), whose teardown is the RISC-V halt wait.
//   golden_gsp <scenario> [key=value ...]
#include "libhmsbeagle/GPU/TinyGPUHybridNVGsp.h"

#include <cstdio>
#include <map>
#include <string>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

using namespace tinygpu_device;

static std::map<std::string, std::string> g_args;
static uint64_t arg(const char* k, uint64_t def = 0) {
    auto it = g_args.find(k);
    return it == g_args.end() ? def : strtoull(it->second.c_str(), nullptr, 0);
}
static std::string hex(const std::vector<uint8_t>& b) {
    std::string s;
    char h[3];
    for (uint8_t x : b) { snprintf(h, sizeof(h), "%02x", x); s += h; }
    return s;
}

int main(int argc, char** argv) {
    if (argc < 2) return fprintf(stderr, "usage: golden_gsp <scenario> [key=value ...]\n"), 2;
    std::string scenario = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        size_t eq = a.find('=');
        if (eq != std::string::npos) g_args[a.substr(0, eq)] = a.substr(eq + 1);
    }
    TGTransport t;
    std::string e = t.open();
    uint64_t bar_addr, bar_size;
    if (!e.empty() || !t.bar_info(0, bar_addr, bar_size, e)) return fprintf(stderr, "transport: %s\n", e.c_str()), 1;   // NVDev's map_bar(0)
    NVBar0 bar0{&t};
    const bool cot = arg("cot") != 0;
    NVFalcon flcn(bar0, (uint32_t)arg("chip_id", 0x197000a1), cot);
    flcn.wait_ms = (int)arg("wait_ms", 30);
    flcn.cot_halt_timeout_s = (double)arg("halt_timeout_ms", 4000) / 1000;
    flcn.chip_name = g_args["chip_name"];
    flcn.sleep = [](double s) { if (s >= 1) printf("sleep %g\n", s); else nv_sleep(s); };   // the 20 s one is recorded, not slept

    uint8_t* queues = nullptr;
    const uint64_t qsize = 0x81000;
    if (const char* qf = getenv("GOLDEN_QUEUES")) {
        int fd = open(qf, O_RDWR);
        if (fd < 0) return fprintf(stderr, "open %s\n", qf), 1;
        queues = (uint8_t*)mmap(nullptr, qsize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        if (queues == MAP_FAILED) return fprintf(stderr, "mmap\n"), 1;
    }
    NVTeardownImages img;
    img.present = true;
    img.sb_paddr = arg("sb_paddr");
    img.sb_imem_pa = (uint32_t)arg("sb_imem_pa"); img.sb_imem_va = (uint32_t)arg("sb_imem_va"); img.sb_imem_sz = (uint32_t)arg("sb_imem_sz");
    img.sb_dmem_pa = (uint32_t)arg("sb_dmem_pa"); img.sb_dmem_sz = (uint32_t)arg("sb_dmem_sz"); img.sb_pkc_off = (uint32_t)arg("sb_pkc_off");
    img.sb_engid = (uint32_t)arg("sb_engid"); img.sb_ucodeid = (uint32_t)arg("sb_ucodeid");
    img.unload_paddr = arg("unload_paddr");
    img.unload_data_off = (uint32_t)arg("unload_data_off"); img.unload_data_sz = (uint32_t)arg("unload_data_sz");
    img.unload_code_off = (uint32_t)arg("unload_code_off"); img.unload_code_sz = (uint32_t)arg("unload_code_sz");

    try {
        if (scenario == "teardown") {   // nv_init_helper's NV_FLCN.fini_hw (COT: NV_FLCN_COT's) after an unload that did or did not confirm
            NVFiniDiag diag;
            diag.unload_ok = arg("unload_ok") != 0;
            diag.cot = cot;
            if (cot) flcn.cot_fini_hw(diag);
            else flcn.fini_hw(diag, img);
            printf("diag=%s\n", diag.json().c_str());
            return 0;
        }
        if (!queues) return fprintf(stderr, "GOLDEN_QUEUES not set\n"), 1;
        NVGsp gsp(bar0, flcn, queues, 0x1000, 0x41000, 0x40000, arg("libos", 0x1234000), (uint32_t)arg("seq"), flcn.wait_ms);
        gsp.rpc_timeout_ms = (int)arg("rpc_timeout_ms", 10000);
        if (scenario == "rpc") {   // NVRpcQueue.send_rpc of len bytes of a fixed pattern
            std::vector<uint8_t> payload(arg("len"));
            for (size_t i = 0; i < payload.size(); ++i) payload[i] = (uint8_t)(i * 7 + 3);
            gsp.cmd_q.send_rpc((uint32_t)arg("func"), payload);
            printf("seq=%u\n", gsp.cmd_q.seq);
        } else if (scenario == "statq") {   // NVRpcQueue.wait_resp on the status queue as golden_gsp.py filled it
            std::vector<uint8_t> msg = gsp.stat_q.wait_resp((uint32_t)arg("cmd"), gsp.rpc_timeout_ms);
            printf("msg=%s\nrx=%u\nerr_state=%d\n", hex(msg).c_str(), gsp.stat_q.rx_view[0], gsp.is_err_state ? 1 : 0);
        } else if (scenario == "seq") {   // NV_GSP.run_cpu_seq on a sequencer message of the given words
            std::vector<uint32_t> words;
            std::string w = g_args["words"];
            for (size_t p = 0; p < w.size();) {
                size_t c = w.find(',', p);
                words.push_back((uint32_t)strtoul(w.substr(p, c - p).c_str(), nullptr, 0));
                p = c == std::string::npos ? w.size() : c + 1;
            }
            nv::rpc_run_cpu_sequencer_v17_00 hdr{};
            hdr.bufferSizeDWord = (uint32_t)words.size();
            hdr.cmdIndex = (uint32_t)arg("cmd_index", words.size());
            std::vector<uint8_t> buf(sizeof(hdr) + words.size() * 4);
            memcpy(buf.data(), &hdr, sizeof(hdr));
            if (!words.empty()) memcpy(buf.data() + sizeof(hdr), words.data(), words.size() * 4);
            gsp.flcn.sleep_after_sec2_start = arg("sec2_sleep") != 0;
            gsp.run_cpu_seq(buf);
            printf("done\n");
        } else if (scenario == "fini") {   // nv_init_helper's NV_GSP.fini_hw, then its NV_FLCN.fini_hw (NVDev.fini's order)
            NVFiniDiag diag;
            try { gsp.fini_hw(diag, arg("level0") != 0); }
            catch (const NVError& x) { printf("unload error=%s\n", x.py().c_str()); }
            if (cot) flcn.cot_fini_hw(diag);
            else flcn.fini_hw(diag, img);
            printf("diag=%s\n", diag.json().c_str());
        } else return fprintf(stderr, "unknown scenario %s\n", scenario.c_str()), 2;
    } catch (const NVError& x) {
        printf("error=%s\n", x.py().c_str());
    }
    return 0;
}
