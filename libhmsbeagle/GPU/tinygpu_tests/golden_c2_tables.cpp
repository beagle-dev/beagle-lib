// C++ side of test_c2_tables.py (TODO.md plan step C2): the generated register tables, TinyGPUNVReg.h's NVReg over them,
// and NVPageTableEntry.set_entry's encodes (nvdev.py:38-49) over the generated MMU field groups. Prints both chips'
// tables, then runs each case line of argv[1] against a logging fake device and prints what it read, wrote or returned,
// in test_c2_tables.py's format.
#include "libhmsbeagle/GPU/TinyGPUNVBootTables.h"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
using namespace tinygpu_device::nv_regs;

struct FakeDev {   // NVDev's rreg/wreg: reads return the case's scripted value
    uint32_t next_read = 0;
    uint32_t rreg(uint32_t addr) { printf("R 0x%x\n", addr); return next_read; }
    void wreg(uint32_t addr, uint32_t v) { printf("W 0x%x 0x%x\n", addr, v); }
};

static std::string hex(nvbits v) {
    std::string s;
    do { s.insert(s.begin(), "0123456789abcdef"[(unsigned)(v & 15)]); v >>= 4; } while (v);
    return "0x" + s;
}
static nvbits parse(const std::string& s) {   // 0x-prefixed hex, up to 128 bits
    nvbits v = 0;
    for (size_t i = 2; i < s.size(); ++i) v = v << 4 | (nvbits)(s[i] <= '9' ? s[i] - '0' : s[i] - 'a' + 10);
    return v;
}
static void print_values(const NVFieldValues& d) {
    printf("=");
    for (uint16_t i = 0; i < d.def->nfields; ++i) printf(" %s:0x%llx", d.def->fields[i].name, (unsigned long long)d.v[i]);
    printf("\n");
}

static const NVRegDef* table(const std::string& chip) { return chip == "Ada" ? kAdaRegs : kGB20xRegs; }
static const NVRegDef& lookup(const std::string& chip, const std::string& name) {
    const NVRegDef* t = table(chip);
    for (int i = 0; i < NV_REG_COUNT; ++i)
        if (name == t[i].name) return t[i];
    fprintf(stderr, "no register %s\n", name.c_str());
    exit(1);
}

// nvdev.py:38-49: the entry set_entry writes (its two words at the dual-PDE level)
static nvbits set_entry(const NVRegDef* regs, int mmu_ver, bool dual, uint64_t paddr, bool table, bool uncached, bool sys, bool valid) {
    if (!table)
        return NVReg<void>(nullptr, regs[mmu_ver == 3 ? NV_MMU_VER3_PTE : NV_MMU_VER2_PTE]).encode(
            {{"valid", valid}, {"address_sys", paddr >> 12}, {"aperture", sys ? 2u : 0u}, {"kind", 6},
             mmu_ver == 3 ? NVKV{"pcf", uncached} : NVKV{"vol", uncached}});
    NVRegId pde = mmu_ver == 3 ? (dual ? NV_MMU_VER3_DUAL_PDE : NV_MMU_VER3_PDE) : (dual ? NV_MMU_VER2_DUAL_PDE : NV_MMU_VER2_PDE);
    std::string small = dual ? "_small" : "", sys_ = mmu_ver == 3 ? "" : "_sys";
    std::string aperture = "aperture" + small, address = "address" + small + sys_, pcf = "pcf" + small;
    return NVReg<void>(nullptr, regs[pde]).encode({{"is_pte", false}, {aperture.c_str(), valid ? 1u : 0u}, {address.c_str(), paddr >> 12},
                                                  mmu_ver == 3 ? NVKV{pcf.c_str(), 0b10} : NVKV{"no_ats", 1}});
}

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    for (std::string chip : {"Ada", "GB20x"}) {
        const NVRegDef* t = table(chip);
        for (int i = 0; i < NV_REG_COUNT; ++i) {
            const NVRegDef& d = t[i];
            if (d.kind == kAbsent) { printf("T %s %s absent\n", chip.c_str(), d.name); continue; }
            printf("T %s %s %s 0x%x 0x%x %s ", chip.c_str(), d.name, d.kind == kGroup ? "group" : "reg", d.base, d.off, d.index ? "fn" : "-");
            for (uint16_t f = 0; f < d.nfields; ++f) printf("%s%s:%u:%u", f ? "," : "", d.fields[f].name, d.fields[f].start, d.fields[f].end);
            printf("%s\n", d.nfields ? "" : "-");
            if (d.index) {
                printf("I %s %s", chip.c_str(), d.name);
                for (uint32_t k : {0u, 1u, 2u, 3u, 5u, 7u, 8u, 15u, 16u, 31u, 63u, 64u, 100u, 1000u, 65535u}) printf(" 0x%x", d.index(k));
                printf("\n");
            }
        }
    }
    std::ifstream in(argv[1]);
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        std::string k, chip, op;
        ss >> k >> chip;
        printf("# %s\n", k.c_str());
        if (chip == "S") {   // S <mmu_ver> <dual> <paddr> <table> <uncached> <sys> <valid>
            int ver, dual, tbl, unc, sys, valid;
            std::string paddr;
            ss >> ver >> dual >> paddr >> tbl >> unc >> sys >> valid;
            nvbits x = set_entry(ver == 3 ? kGB20xRegs : kAdaRegs, ver, dual, (uint64_t)parse(paddr), tbl, unc, sys, valid);
            printf("= %s%s%s\n", hex((uint64_t)x).c_str(), dual ? " " : "", dual ? hex((uint64_t)(x >> 64)).c_str() : "");
            continue;
        }
        std::string name, chain;
        ss >> name >> chain >> op;
        const NVRegDef& def = lookup(chip, name);
        if (def.kind == kAbsent) { printf("! %s is not in the %s table\n", name.c_str(), chip.c_str()); continue; }
        FakeDev dev;
        NVReg<FakeDev> reg(&dev, def);
        if (chain != "-") {   // with_base (bHEX) and [i] (iN), in order
            std::istringstream cs(chain);
            std::string item;
            while (std::getline(cs, item, ',')) reg = item[0] == 'b' ? reg.with_base((uint32_t)parse(item.substr(1))) : reg[(uint32_t)std::stoul(item.substr(1))];
        }
        std::vector<std::string> args;
        for (std::string a; ss >> a;) args.push_back(a);
        std::vector<NVKV> kw;
        std::vector<const char*> names;
        size_t first = (op == "W" || op == "U") ? 1 : 0;
        for (size_t i = first; i < args.size(); ++i) {
            size_t eq = args[i].find('=');
            if (eq == std::string::npos) { names.push_back(args[i].c_str()); continue; }
            args[i][eq] = '\0';
            kw.push_back({args[i].c_str(), (uint64_t)parse(args[i].substr(eq + 1))});
        }
        if (op == "W") reg.write((uint32_t)parse(args[0]), kw);                          // write(ini, **kw)
        else if (op == "U") { dev.next_read = (uint32_t)parse(args[0]); reg.update(kw); } // update(**kw) on a scripted read
        else if (op == "D") { dev.next_read = (uint32_t)parse(args[0]); print_values(reg.read_bitfields()); }
        else if (op == "E") printf("= %s\n", hex(reg.encode(kw)).c_str());
        else if (op == "M") printf("= %s\n", hex(reg.mask(names)).c_str());
        else if (op == "X") print_values(reg.decode(parse(args[0])));
    }
    return 0;
}
