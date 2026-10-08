// TinyGPUAMDReg.h's AMRegister on the cases test_a2b_tables.py writes, against a logging fake AMDev (see that file) of a
// register family (am::regs::kFamilies, by name): each case's register accesses and result, in the format tinygrad's side prints.
//   golden_amd_regs <bases file> <cases file> <family>
#include "libhmsbeagle/GPU/TinyGPUAMDReg.h"
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
using namespace tinygpu_device;

struct FakeDev {
    const am::regs::Family* family = nullptr;
    const am::regs::Family& regs() const { return *family; }
    std::map<std::pair<int, int>, std::vector<uint32_t>> bases;   // (hwip, inst) -> bases
    std::map<uint32_t, uint32_t> vals;
    uint32_t base(int hwip, int inst, int seg) {
        auto it = bases.find({hwip, inst});
        if (it == bases.end() || seg >= (int)it->second.size()) throw am::AMRegError("no base");
        return it->second[seg];
    }
    uint32_t rreg(uint32_t reg, int inst, bool direct) {
        uint32_t v = vals.count(reg) ? vals[reg] : 0;
        printf("r 0x%x %d %d -> 0x%x\n", reg, inst, (int)direct, v);
        return v;
    }
    void wreg(uint32_t reg, uint32_t v, int inst, bool direct) { printf("w 0x%x 0x%x %d %d\n", reg, v, inst, (int)direct); }
};

int main(int, char** argv) {
    FakeDev dev;
    for (const am::regs::Family& f : am::regs::kFamilies)
        if (std::string(f.name) == argv[3]) dev.family = &f;
    if (!dev.family) { printf("no register family %s\n", argv[3]); return 1; }
    std::ifstream bf(argv[1]);
    for (std::string line; std::getline(bf, line);) {
        std::istringstream ss(line);
        int hwip, inst;
        ss >> hwip >> inst;
        std::vector<uint32_t> b;
        for (uint64_t x; ss >> std::hex >> x;) b.push_back((uint32_t)x);
        dev.bases[{hwip, inst}] = b;
    }
    std::ifstream cf(argv[2]);
    for (std::string line; std::getline(cf, line);) {
        std::istringstream ss(line);
        std::string k, op, name;
        ss >> k >> op >> name;
        printf("# %s\n", k.c_str());
        std::vector<std::string> fnames;
        std::vector<am::AMKV> kw;
        uint64_t a = 0;
        bool have_a = false;
        for (std::string t; ss >> t;) {
            size_t eq = t.find('=');
            if (eq == std::string::npos) { a = std::stoull(t, nullptr, 0); have_a = true; continue; }
            fnames.push_back(t.substr(0, eq));
        }
        std::istringstream ss2(line);
        ss2 >> k >> op >> name;
        size_t fi = 0;
        for (std::string t; ss2 >> t;) {
            size_t eq = t.find('=');
            if (eq != std::string::npos) { kw.push_back({fnames[fi].c_str(), std::stoull(t.substr(eq + 1), nullptr, 0)}); ++fi; }
        }
        try {
            am::AMRegister<FakeDev> r(&dev, name.c_str());
            if (op == "addr") printf("= 0x%x\n", r.addr(0));
            else if (op == "read") { dev.vals[r.addr(0)] = (uint32_t)a; printf("= 0x%x\n", r.read()); }
            else if (op == "read_bitfields") { dev.vals[r.addr(0)] = (uint32_t)a; printf("= %s\n", r.read_bitfields().repr().c_str()); }
            else if (op == "write") r.write(have_a ? a : 0, kw);
            else if (op == "update") { dev.vals[r.addr(0)] = (uint32_t)a; r.update(kw); }
            else if (op == "encode") printf("= 0x%llx\n", (unsigned long long)r.encode(kw));
            else if (op == "decode") printf("= %s\n", r.decode(a).repr().c_str());
            else if (op == "fields_mask") {
                std::vector<const char*> ns;
                for (auto& n : fnames) ns.push_back(n.c_str());
                printf("= 0x%llx\n", (unsigned long long)r.fields_mask(ns));
            }
        } catch (const am::AMRegError& e) { printf("! %s\n", e.what()); }
    }
    return 0;
}
