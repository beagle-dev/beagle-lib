// Runs golden_amd_copy_ops.txt with TinyGPUHybridAMDDispatch.h alone (see golden_amd_copy.py): copyin/copyout through the
// staging slots on the timeline, and queues of their own, each submitted with amd_sdma_ring_write and signal_doorbell.
#include "libhmsbeagle/GPU/TinyGPUHybridAMDDispatch.h"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
using namespace tinygpu_device;

static std::string slurp(const std::string& p) { std::ifstream f(p, std::ios::binary); std::stringstream s; s << f.rdbuf(); return s.str(); }

struct Ctx {   // the timeline and the SDMA queue, as golden_amd_copy.py's stub device
    uint64_t signal_va = 0, timeline_value = 0, ring_bytes = 0, put = 0;
    uint32_t* ring = nullptr;
    std::ofstream* ev = nullptr;
    uint64_t next_timeline() { return timeline_value++; }
    bool host_wait(uint64_t v) { *ev << "wait " << v << "\n"; return true; }
    bool synchronize() { *ev << "sync 0\n"; return true; }
    bool submit(const AMDCopyQueue& q) {
        if (!amd_sdma_ring_write(ring, ring_bytes, put, q, [](uint64_t) { return true; })) return false;
        *ev << "wptr " << put << "\nhdp 0\ndoorbell " << put << "\n";
        return true;
    }
};

int main(int, char** argv) {
    const std::string dir = argv[1];
    std::ifstream in(dir + "/golden_amd_copy_ops.txt");
    std::string rb = slurp(dir + "/golden_amd_copy_ring_in.bin"), sb = slurp(dir + "/golden_amd_copy_staging_in.bin"),
                src = slurp(dir + "/golden_amd_copy_srcs.bin");
    std::vector<uint32_t> ring(rb.size() / 4);
    memcpy(ring.data(), rb.data(), rb.size());
    std::vector<uint8_t> staging(sb.begin(), sb.end()), outs;
    std::ofstream ev(dir + "/golden_amd_copy_out_events.txt");
    Ctx c;
    c.ring = ring.data();
    c.ev = &ev;
    AMDStaging s;
    s.host = staging.data();
    size_t src_at = 0;
    std::string tag;
    while (in >> tag) {
        if (tag == "SIG") in >> c.signal_va;
        else if (tag == "RING") in >> c.ring_bytes >> c.put;
        else if (tag == "STAGING") { size_t n; in >> s.va >> s.slot_size >> n; s.timeline.assign(n, 0); }
        else if (tag == "TL") in >> c.timeline_value;
        else if (tag == "CI") {
            uint64_t dest, n; in >> dest >> n;
            if (!amd_copyin(c, s, dest, (const uint8_t*)src.data() + src_at, n)) return 1;
            src_at += n;
        } else if (tag == "CO") {
            uint64_t from, n; in >> from >> n;
            std::vector<uint8_t> out(n);
            if (!amd_copyout(c, s, out.data(), from, n)) return 1;
            outs.insert(outs.end(), out.begin(), out.end());
        } else if (tag == "R") {
            uint64_t m, vw, dest, from, n, vs; in >> m >> vw >> dest >> from >> n >> vs;
            AMDCopyQueue q;
            q.max_copy_size = m;
            q.wait(c.signal_va, (uint32_t)vw);
            q.copy(dest, from, n);
            q.signal(c.signal_va, vs);
            if (!c.submit(q)) return 1;
        }
    }
    std::ofstream(dir + "/golden_amd_copy_out_ring.bin", std::ios::binary).write((const char*)ring.data(), ring.size() * 4);
    std::ofstream(dir + "/golden_amd_copy_out_put.txt") << c.put << "\n";
    std::ofstream(dir + "/golden_amd_copy_out_staging.bin", std::ios::binary).write((const char*)staging.data(), staging.size());
    std::ofstream(dir + "/golden_amd_copy_out_outs.bin", std::ios::binary).write((const char*)outs.data(), outs.size());
    return 0;
}
