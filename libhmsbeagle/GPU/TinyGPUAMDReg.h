/*
 * TinyGPUAMDReg.h
 *
 * C++ port of tinygrad's AMDReg and AMRegister (tinygrad/runtime/support/amd.py:5-15 and am/amdev.py:13-23 at a9830e2b4),
 * by hand, over the register tables make_tinygpu_amd_boot_tables.py generates into TinyGPUAMDBootTables.h (TODO.md plan
 * step A2b), one per register family (plan step N11: Dev's regs(), as AMDev._build_regs binds its IP versions' modules). As
 * in tinygrad, a register is (offset, segment, fields), its address on an instance is that instance's
 * discovered base for the segment plus the offset (AMDReg.__post_init__), and fields maps a name to (start, end), end
 * inclusive. Registers and fields are named at the call site, so a ported statement reads as tinygrad's:
 *
 *     adev.reg("regSDMA0_QUEUE0_RB_CNTL").write(0, {{"rb_vmid", 0}, {"rptr_writeback_enable", 1}, {"rb_enable", 1}});
 *
 * Dev is anything with AMDev's rreg(reg, inst, direct), wreg(reg, val, inst, direct), base(hwip, inst, segment) (the dword
 * address of a segment's base) and regs() (its register family, regs::kFamilies). A register the table lacks (the boot reached a name the coverage sessions never
 * did), a field it lacks, an instance or segment the card has no base for, or a value wider than 32 bits (tinygrad's
 * struct.pack('<I') raises) is a porting error: it throws AMRegError, which the golden tests exercise.
 */

#ifndef LIBHMSBEAGLE_GPU_TINYGPUAMDREG_H
#define LIBHMSBEAGLE_GPU_TINYGPUAMDREG_H

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

#include "libhmsbeagle/GPU/TinyGPUAMDBootTables.h"

namespace tinygpu_device {
namespace am {

struct AMRegError : std::runtime_error { using std::runtime_error::runtime_error; };

// a family's generated table, sorted by name: the register, or nullptr
inline const regs::AMRegDef* find_reg(const regs::Family& f, const char* name) {
    size_t lo = 0, hi = f.nregs;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        int c = strcmp(f.regs[mid].name, name);
        if (c == 0) return &f.regs[mid];
        if (c < 0) lo = mid + 1; else hi = mid;
    }
    return nullptr;
}

// hasattr(adev, name): a register the family's table has; a name the coverage sessions asked for and its card lacks is
// absent; any other name is a porting error
inline bool has_reg(const regs::Family& f, const char* name) {
    if (find_reg(f, name)) return true;
    for (size_t i = 0; i < f.nabsent; ++i)
        if (strcmp(f.absent[i], name) == 0) return false;
    throw AMRegError(std::string("has_reg: ") + name + " is in neither the " + f.name + " register table nor its absent list");
}

struct AMKV { const char* name; uint64_t value; };   // a keyword argument, name=value

template <class T> struct AMArgs {   // keyword arguments or names at a call site: a braced list (valid for the call) or a vector
    const T* p = nullptr;
    size_t n = 0;
    AMArgs() = default;
    AMArgs(std::initializer_list<T> l) : p(l.begin()), n(l.size()) {}
    AMArgs(const std::vector<T>& v) : p(v.data()), n(v.size()) {}
    const T* begin() const { return p; }
    const T* end() const { return p + n; }
};

inline uint64_t am_getbits(uint64_t v, unsigned start, unsigned end) {   // helpers.getbits
    return (v >> start) & ((end - start + 1 >= 64) ? ~0ull : ((1ull << (end - start + 1)) - 1));
}

struct AMFieldValues {   // decode's dict: every field, in the register's order
    const regs::AMRegDef* def = nullptr;
    uint64_t v[64] = {};
    uint64_t operator[](const char* name) const {
        for (uint8_t i = 0; i < def->nfields; ++i)
            if (strcmp(def->fields[i].name, name) == 0) return v[i];
        throw AMRegError(std::string(def->name) + ": no field " + name);   // the dict's KeyError
    }
    std::string repr() const {   // str(dict), as tinygrad's log lines print it
        std::string s = "{";
        for (uint8_t i = 0; i < def->nfields; ++i)
            s += std::string(i ? ", '" : "'") + def->fields[i].name + "': " + std::to_string(v[i]);
        return s + "}";
    }
};

template <class Dev> struct AMRegister {
    Dev* adev = nullptr;
    const regs::AMRegDef* def = nullptr;

    AMRegister(Dev* adev_, const char* name) : adev(adev_), def(find_reg(adev_->regs(), name)) {
        if (!def) throw AMRegError(std::string("no register ") + name + " in the " + adev_->regs().name + " boot tables (AMDev's KeyError)");
    }
    const regs::AMField& field(const char* name) const {
        for (uint8_t i = 0; i < def->nfields; ++i)
            if (strcmp(def->fields[i].name, name) == 0) return def->fields[i];
        throw AMRegError(std::string(def->name) + ": no field " + name);
    }
    uint32_t addr(int inst = 0) const { return adev->base(def->hwip, inst, def->segment) + def->offset; }   // AMDReg.addr[inst]
    uint32_t read(int inst = 0, bool direct = false) const { return adev->rreg(addr(inst), inst, direct); }
    AMFieldValues read_bitfields(int inst = 0) const { return decode(read(inst)); }
    void write(uint64_t val, AMArgs<AMKV> kw = {}, int inst = 0, bool direct = false) const {
        uint64_t v = val | encode(kw);
        if (v >> 32) throw AMRegError(std::string(def->name) + ": a value wider than the 32-bit register");
        adev->wreg(addr(inst), (uint32_t)v, inst, direct);
    }
    void write(AMArgs<AMKV> kw, int inst = 0, bool direct = false) const { write(0, kw, inst, direct); }
    void update(AMArgs<AMKV> kw, int inst = 0) const {   // write(read() & ~fields_mask(*kwargs.keys()), **kwargs)
        uint64_t m = 0;
        for (const AMKV& a : kw) m |= fields_mask({a.name});
        write(read(inst) & ~m, kw, inst);
    }
    uint64_t encode(AMArgs<AMKV> kw) const {   // as tinygrad's: a value is not masked to its field
        uint64_t x = 0;
        for (const AMKV& a : kw) x |= a.value << field(a.name).start;
        return x;
    }
    AMFieldValues decode(uint64_t val) const {
        AMFieldValues r;
        r.def = def;
        for (uint8_t i = 0; i < def->nfields; ++i) r.v[i] = am_getbits(val, def->fields[i].start, def->fields[i].end);
        return r;
    }
    uint64_t fields_mask(AMArgs<const char*> names) const {
        uint64_t m = 0;
        for (const char* nm : names) {
            const regs::AMField& f = field(nm);
            unsigned w = f.end - f.start + 1;
            m |= (w >= 64 ? ~0ull : ((1ull << w) - 1)) << f.start;
        }
        return m;
    }
};

} // namespace am
} // namespace tinygpu_device

#endif // LIBHMSBEAGLE_GPU_TINYGPUAMDREG_H
