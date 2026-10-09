// C++ side of golden_mm.py (TODO.md plan step C6): TinyGPUMemory.h and TinyGPUNVMemory.h run golden_mm.py's
// operations and print the result lines golden_mm.py prints for tinygrad's own memory manager.
//   golden_mm tlsf OPS               TLSFAllocator: the first line "config size base", then "a size align" or "f addr"
//   golden_mm mm EXPORT OPS          the manager restored from nv_dispatch_daemon.py's _mm_export (a JSON file), over
//                                    golden_mm.py's TinyGPU.app (APL_REMOTE_SOCK)
//   golden_mm scratch MMU VRAM OPS   the manager built by tinygrad's constructor (NVDev._early_mmu_init's arguments)
// Manager operations, one per line (results: "va size paddr:size,..." etc., or "error Type: message"):
//   valloc size align uncached contiguous zero | vfree i | alloc size host uncached cpu_access contiguous force_devmem zero |
//   free i | palloc size align zero ptable | pfree paddr ptable | alloc_vaddr size align | fence size iova | booted
// i counts valloc and alloc results from 0 (failures too). "state" lines at the end: each allocator's save().
#include "libhmsbeagle/GPU/TinyGPUNVMemory.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace tinygpu_device;

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

static int run_tlsf(std::istream& in) {
    std::string op;
    uint64_t size = 0, base = 0;
    in >> op >> size >> base;
    TLSFAllocator a(size, base);
    uint64_t x, y;
    while (in >> op) {
        try {
            if (op == "a") { in >> x >> y; printf("%llu\n", (unsigned long long)a.alloc(x, y)); }
            else { in >> x; a.free(x); printf("ok\n"); }
        } catch (const TGPyError& e) { printf("error %s\n", e.py().c_str()); }
    }
    printf("state%s\n", words(a.save()).c_str());
    return 0;
}

static int run_mm(NVMemState& st, std::istream& in) {
    NVMemoryManager& mm = *st.mm;
    std::vector<TGVirtMapping> maps;
    std::vector<NVBuffer> bufs;
    std::vector<int> kind;   // 0 valloc, 1 alloc, -1 failed
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream l(line);
        std::string op;
        if (!(l >> op)) continue;
        uint64_t v[7] = {};   // the operands, read as int(x, 0) reads them
        std::string tok;
        for (int i = 0; i < 7 && l >> tok; ++i) v[i] = strtoull(tok.c_str(), nullptr, 0);
        const uint64_t a = v[0], b = v[1], c = v[2], d = v[3], e = v[4], f = v[5], g = v[6];
        try {
            if (op == "valloc") {
                kind.push_back(-1); maps.emplace_back(); bufs.emplace_back();
                maps.back() = mm.valloc(a, b, c, d, e);
                kind.back() = 0;
                printf("%s\n", mapping(maps.back()).c_str());
            } else if (op == "vfree") {
                mm.vfree(maps.at(a));
                printf("ok\n");
            } else if (op == "alloc") {
                kind.push_back(-1); maps.emplace_back(); bufs.emplace_back();
                bufs.back() = nv_iface_alloc(mm, a, b, c, d, e, f, g);
                kind.back() = 1;
                printf("%llu %llu %llu %s\n", (unsigned long long)bufs.back().va_addr, (unsigned long long)bufs.back().size,
                       (unsigned long long)bufs.back().hMemory, mapping(bufs.back().mapping).c_str());
            } else if (op == "free") {
                nv_iface_free(mm, bufs.at(a));
                printf("ok\n");
            } else if (op == "palloc") {
                printf("%llu\n", (unsigned long long)mm.palloc(a, b, c, false, d));
            } else if (op == "pfree") {
                mm.pfree(a, b);
                printf("ok\n");
            } else if (op == "alloc_vaddr") {
                printf("%llu\n", (unsigned long long)mm.alloc_vaddr(a, b));
            } else if (op == "fence") {   // a system-memory mapping of pages at iova, which TinyGPU.app never handed out
                uint64_t va = mm.alloc_vaddr(a, 0x1000);
                std::vector<std::pair<uint64_t, uint64_t>> pages;
                for (uint64_t off = 0; off < a; off += 0x1000) pages.push_back({b + off, 0x1000});
                mm.map_range(va, a, pages, TGAddrSpace::SYS, true, true);
                printf("mapped %llu\n", (unsigned long long)va);
            } else if (op == "booted") {   // NVDev.__init__: is_booting = False after _early_mmu_init
                st.dev.is_booting = false;
                printf("ok\n");
            } else {
                fprintf(stderr, "unknown op %s\n", op.c_str());
                return 2;
            }
        } catch (const TGPyError& x) {
            printf("error %s\n", x.py().c_str());
        }
    }
    printf("state boot%s\n", words(mm.boot_allocator.save()).c_str());
    printf("state ptable%s\n", words(mm.ptable_allocator.save()).c_str());
    printf("state pa%s\n", words(mm.pa_allocator.save()).c_str());
    printf("state va%s\n", words(st.va.save()).c_str());
    printf("vram_end %llu\n", (unsigned long long)nv_vram_end(mm));
    return 0;
}

int main(int argc, char** argv) {
    std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "tlsf" && argc == 3) {
        std::ifstream in(argv[2]);
        return run_tlsf(in);
    }
    if ((mode == "mm" && argc == 4) || (mode == "scratch" && argc == 5)) {
        TGTransport t;
        std::string err = t.open();
        if (!err.empty()) return fprintf(stderr, "transport: %s\n", err.c_str()), 1;
        NVMemState st;
        if (mode == "mm") {
            std::ifstream jf(argv[2]);
            std::stringstream js;
            js << jf.rdbuf();
            err = nv_mm_import(js.str(), &t, st);
            if (!err.empty()) return fprintf(stderr, "import: %s\n", err.c_str()), 1;
            t.seed_bar(0, 16 << 20);   // the daemon mapped BAR0 too
        } else {   // NVDev._early_mmu_init (nvdev.py:131-147) on the small-BAR card: map_bar(1) and map_bar(0) are in the stream
            const uint64_t mmu = strtoull(argv[2], nullptr, 0), vram = strtoull(argv[3], nullptr, 0);
            uint64_t addr, bar1, bar0;
            if (!t.bar_info(1, addr, bar1, err) || !t.bar_info(0, addr, bar0, err)) return fprintf(stderr, "bar_info: %s\n", err.c_str()), 1;
            st.dev.t = &t;
            st.dev.mmu_ver = (int)mmu;
            st.dev.regs = mmu == 3 ? nv_regs::kGB20xRegs : nv_regs::kAdaRegs;
            st.dev.is_booting = true;
            st.dev_vram_size = vram;
            st.va = TLSFAllocator(1ull << 44, 0x1000000000ull);
            std::vector<uint64_t> shifts = mmu == 3 ? std::vector<uint64_t>{12, 21, 29, 38, 47, 56} : std::vector<uint64_t>{12, 21, 29, 38, 47};
            try {
                st.mm = std::make_unique<NVMemoryManager>(&st.dev, vram - (64ull << 20), 2ull << 20, mmu == 3 ? 56 : 48, shifts, 0,
                                                          std::vector<std::pair<uint64_t, uint64_t>>{{512ull << 20, 512ull << 20}, {2ull << 20, 2ull << 20}, {4ull << 10, 4ull << 10}},
                                                          0, !(bar1 >= vram));
            } catch (const TGPyError& x) { return fprintf(stderr, "constructor: %s\n", x.py().c_str()), 1; }
            st.mm->va_allocator = &st.va;
            printf("root %llu\n", (unsigned long long)st.mm->root_page_table.paddr);
        }
        std::ifstream in(argv[mode == "mm" ? 3 : 4]);
        int rc = run_mm(st, in);
        t.close();
        return rc;
    }
    fprintf(stderr, "usage: golden_mm tlsf OPS | mm EXPORT OPS | scratch MMU VRAM OPS\n");
    return 2;
}
