// C++ side of golden_rm.py (TODO.md plan step C7): TinyGPUNVRM.h's RM client runs golden_rm.py's operations over its
// fake TinyGPU.app (APL_REMOTE_SOCK), with the GSP queues in the file GOLDEN_QUEUES and the memory manager restored from
// nv_dispatch_daemon.py's _mm_export, and prints the result lines golden_rm.py prints for tinygrad's own NV_GSP.
//   golden_rm EXPORT STATE OPS
// STATE, one "key value..." per line: seq, priv_root, next_handle, gpfifo_class, compute_class, dma_class, viddec_class, gb2,
//   runlist KEY VALUE, grctx ID SIZE PHYS VIRT LOCAL.
// OPS, one per line (numbers as int(x, 0) reads them; "-" for params None; client 0 for None):
//   rm_alloc PARENT CLASS CLIENT HEX | rm_control OBJECT CMD CLIENT HEX | alloc SIZE HOST UNCACHED CPU_ACCESS CONTIGUOUS FORCE_DEVMEM ZERO
#include "libhmsbeagle/GPU/TinyGPUNVRM.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

using namespace tinygpu_device;

static uint64_t num(const std::string& s) { return strtoull(s.c_str(), nullptr, 0); }
static std::string hex(const std::vector<uint8_t>& b) {
    std::string s;
    char h[3];
    for (uint8_t x : b) { snprintf(h, sizeof(h), "%02x", x); s += h; }
    return s;
}
static std::vector<uint8_t> unhex(const std::string& s) {
    std::vector<uint8_t> b;
    for (size_t i = 0; i + 1 < s.size(); i += 2) b.push_back((uint8_t)strtoul(s.substr(i, 2).c_str(), nullptr, 16));
    return b;
}
static std::string words(const std::vector<uint64_t>& w) {
    std::string s;
    for (uint64_t x : w) s += " " + std::to_string(x);
    return s;
}
static std::string mapping(const TGVirtMapping& m) {
    std::string s = std::to_string(m.va_addr) + " " + std::to_string(m.size) + " " + std::to_string((int)m.aspace) + " ";
    for (size_t i = 0; i < m.paddrs.size(); ++i) s += (i ? "," : "") + std::to_string(m.paddrs[i].first) + ":" + std::to_string(m.paddrs[i].second);
    return s;
}

int main(int argc, char** argv) {
    if (argc != 4) return fprintf(stderr, "usage: golden_rm EXPORT STATE OPS\n"), 2;
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

    std::map<std::string, std::vector<std::string>> kv;
    std::map<uint64_t, uint32_t> runlists;
    std::vector<std::pair<uint16_t, NVGRBufDesc>> grctx;
    std::ifstream sf(argv[2]);
    for (std::string line; std::getline(sf, line);) {
        std::istringstream l(line);
        std::string k, v;
        std::vector<std::string> vals;
        l >> k;
        while (l >> v) vals.push_back(v);
        if (k == "runlist") runlists[num(vals.at(0))] = (uint32_t)num(vals.at(1));
        else if (k == "grctx") grctx.push_back({(uint16_t)num(vals.at(0)), {num(vals.at(1)), num(vals.at(2)) != 0, num(vals.at(3)) != 0, num(vals.at(4)) != 0}});
        else if (!k.empty()) kv[k] = vals;
    }
    auto get = [&](const char* k) { return num(kv.at(k).at(0)); };

    NVBar0 bar0{&t};
    NVFalcon flcn(bar0, 0x197000a1);
    NVGsp gsp(bar0, flcn, queues, 0x1000, 0x41000, 0x40000, 0, (uint32_t)get("seq"), 30);
    NVRMClient rm(gsp, *st.mm);
    rm.priv_root = (uint32_t)get("priv_root");
    rm.next_handle = (uint32_t)get("next_handle");
    rm.gpfifo_class = (uint32_t)get("gpfifo_class");
    rm.compute_class = (uint32_t)get("compute_class");
    rm.dma_class = (uint32_t)get("dma_class");
    rm.viddec_class = (uint32_t)get("viddec_class");
    rm.gb2 = get("gb2") != 0;
    rm.runlists = runlists;
    rm.grctx_bufs = grctx;

    std::ifstream in(argv[3]);
    std::vector<NVBuffer> bufs;
    for (std::string line; std::getline(in, line);) {
        std::istringstream l(line);
        std::string op;
        std::vector<std::string> a;
        if (!(l >> op)) continue;
        for (std::string x; l >> x;) a.push_back(x);
        auto client = [&](size_t i) { return num(a.at(i)) ? std::optional<uint32_t>((uint32_t)num(a.at(i))) : std::nullopt; };
        try {
            if (op == "rm_alloc") {
                std::vector<uint8_t> p = a.at(3) == "-" ? std::vector<uint8_t>{} : unhex(a.at(3));
                uint32_t h = rm.rpc_rm_alloc_bytes((uint32_t)num(a.at(0)), (uint32_t)num(a.at(1)), a.at(3) == "-" ? nullptr : &p, client(2));
                printf("handle %u params %s\n", h, a.at(3) == "-" ? "-" : hex(p).c_str());
            } else if (op == "rm_control") {
                std::vector<uint8_t> p = unhex(a.at(3));
                auto r = rm.rpc_rm_control_bytes((uint32_t)num(a.at(0)), (uint32_t)num(a.at(1)), a.at(3) == "-" ? nullptr : &p, client(2));
                printf("reply %s\n", r ? hex(*r).c_str() : "-");
            } else if (op == "alloc") {
                NVBuffer b = nv_iface_alloc(*st.mm, num(a.at(0)), num(a.at(1)), num(a.at(2)), num(a.at(3)), num(a.at(4)), num(a.at(5)), num(a.at(6)));
                printf("%llu %llu %llu %s\n", (unsigned long long)b.va_addr, (unsigned long long)b.size, (unsigned long long)b.hMemory,
                       mapping(b.mapping).c_str());
                bufs.push_back(b);
            } else return fprintf(stderr, "unknown op %s\n", op.c_str()), 2;
        } catch (const TGPyError& x) {
            printf("error %s\n", x.py().c_str());
        } catch (const NVError& x) {
            printf("error %s\n", x.py().c_str());
        }
    }
    printf("rm next_handle %u device %u subdevice %u seq %u\n", rm.next_handle, rm.device, rm.subdevice, gsp.cmd_q.seq);
    std::string cr;
    for (auto& [k, v] : rm.chan_runlists) cr += " " + std::to_string(k) + ":" + std::to_string(v);
    printf("rm chan_runlists%s\n", cr.c_str());
    printf("state pa%s\n", words(st.mm->pa_allocator.save()).c_str());
    printf("state va%s\n", words(st.va.save()).c_str());
    t.close();
    return 0;
}
