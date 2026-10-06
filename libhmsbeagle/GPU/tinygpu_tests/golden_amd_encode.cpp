// Encodes golden_amd_batch.txt with TinyGPUAMDDispatch.h alone (see golden_amd_encode.py): the queues, the kernargs
// slots (BumpAllocator), and each submit's ring write, then signal_doorbell's wptr, HDP flush and doorbell, in order.
#include "libhmsbeagle/GPU/TinyGPUAMDDispatch.h"
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
using namespace tinygpu_device;

static std::string slurp(const std::string& p) { std::ifstream f(p, std::ios::binary); std::stringstream s; s << f.rdbuf(); return s.str(); }

int main(int, char** argv) {
    const std::string dir = argv[1];
    std::ifstream in(dir + "/golden_amd_batch.txt");
    const std::string kb = slurp(dir + "/golden_amd_kargs_in.bin"), rb = slurp(dir + "/golden_amd_ring_in.bin");
    std::vector<uint8_t> kargs(kb.begin(), kb.end());
    std::vector<uint32_t> ring(rb.size() / 4);
    memcpy(ring.data(), rb.data(), rb.size());
    AMDExecDevice dev;
    std::map<std::string, AMDKernel> kernels;
    AMDBump bump;
    AMDComputeQueue q;
    uint64_t sig = 0, kargs_va = 0, ring_dwords = 0, put = 0;
    std::ofstream oq(dir + "/golden_amd_out_queues.txt"), oe(dir + "/golden_amd_out_events.txt");
    std::string tag;
    while (in >> tag) {
        if (tag == "DEV") in >> dev.scratch_va >> dev.scratch_size >> dev.tmpring_size;
        else if (tag == "SIG") in >> sig;
        else if (tag == "KARGS") in >> kargs_va >> bump.size;
        else if (tag == "RING") in >> ring_dwords >> put;
        else if (tag == "K") {
            std::string name; AMDKernel k; int wave32;
            in >> name >> k.prog_addr >> k.rsrc1 >> k.rsrc2 >> k.rsrc3 >> k.kernargs_segment_size >> k.kernargs_alloc_size >> wave32;
            k.wave32 = wave32 != 0;
            kernels[name] = k;
        } else if (tag == "Q") {   // the daemon's chained launch_batch: wait for the timeline, then memory_barrier
            uint32_t v; in >> v;
            q = AMDComputeQueue();
            q.wait(sig, v);
            q.memory_barrier();
        } else if (tag == "L") {
            std::string name; uint32_t grid[3], block[3]; int nptr, nint;
            in >> name >> grid[0] >> grid[1] >> grid[2] >> block[0] >> block[1] >> block[2] >> nptr;
            std::vector<uint64_t> ptrs(nptr); for (auto& p : ptrs) in >> p;
            in >> nint;
            std::vector<uint32_t> ints(nint); for (auto& i : ints) in >> i;
            const AMDKernel& k = kernels.at(name);
            const uint64_t off = bump.alloc(k.kernargs_alloc_size, 8);   // HCQProgram.fill_kernargs
            q.exec(k, dev, kargs.data() + off, kargs_va + off, ptrs.data(), nptr, ints.data(), nint, grid, block);
        } else if (tag == "S") {   // signal, then submit: the ring write and signal_doorbell
            uint64_t v; in >> v;
            q.signal(sig, v);
            for (size_t i = 0; i < q.q.size(); ++i) oq << (i ? " " : "") << q.q[i];
            oq << "\n";
            put = amd_compute_ring_write(ring.data(), ring_dwords, put, q.q);
            oe << "wptr " << put << "\nhdp 0\ndoorbell " << put << "\n";
        }
    }
    std::ofstream(dir + "/golden_amd_out_kargs.bin", std::ios::binary).write((const char*)kargs.data(), kargs.size());
    std::ofstream(dir + "/golden_amd_out_ring.bin", std::ios::binary).write((const char*)ring.data(), ring.size() * 4);
    std::ofstream(dir + "/golden_amd_out_put.txt") << put << "\n";
    return 0;
}
