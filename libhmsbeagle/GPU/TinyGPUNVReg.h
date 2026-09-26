/*
 * TinyGPUNVReg.h
 *
 * C++ port of tinygrad's NVReg (tinygrad/runtime/support/nv/nvdev.py:12-31), by hand, over the register and MMU
 * field tables make_tinygpu_nv_boot_tables.py generates into TinyGPUNVBootTables.h (TODO.md plan step C2), plus
 * tinygrad's bitfield accessors for the generated structs (runtime/support/c.py:68-73). As in tinygrad, a register
 * is (base, off, fields): off is an offset or, for an indexed register, a function of the index (the autogen's
 * lambda), and fields maps a name to (start, end), end inclusive. Fields are named at the call site, so a ported
 * statement reads as tinygrad's:
 *
 *     NVReg<Dev>(dev, regs[NV_PFALCON_FALCON_DMATRFCMD]).with_base(base).encode(
 *         {{"write", 0}, {"size", NV_PFALCON_FALCON_DMATRFCMD_SIZE_256B}, {"ctxdma", ctx_dma}, {"imem", 1}, {"sec", 1}})
 *
 * Dev is anything with NVDev's rreg(addr) and wreg(addr, value) (BAR0 byte addresses). encode, mask and decode are
 * 128 bits wide, for the MMU's DUAL_PDE; as in tinygrad, encode does not mask a value to its field. Where tinygrad
 * raises (a field the register lacks, a name the chip never included, a read or write of an MMU field group or of an
 * indexed register without its index) this aborts: a porting error, which the golden tests exercise.
 */

#ifndef LIBHMSBEAGLE_GPU_TINYGPUNVREG_H
#define LIBHMSBEAGLE_GPU_TINYGPUNVREG_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <vector>

namespace tinygpu_device {

// c.py:68-73: a bitfield is read and written little-endian over its own ceil((bit_off + width) / 8) bytes; set does
// not mask the value (1 <= width <= 64, nbytes <= 8)
inline uint64_t nv_bitfield_get(const uint8_t* p, unsigned nbytes, unsigned bit_off, unsigned width) {
    uint64_t v = 0;
    for (unsigned i = 0; i < nbytes; ++i) v |= (uint64_t)p[i] << (8 * i);
    return v >> bit_off & (~0ull >> (64 - width));
}
inline void nv_bitfield_set(uint8_t* p, unsigned nbytes, unsigned bit_off, unsigned width, uint64_t val) {
    uint64_t v = 0;
    for (unsigned i = 0; i < nbytes; ++i) v |= (uint64_t)p[i] << (8 * i);
    v = (v & ~((~0ull >> (64 - width)) << bit_off)) | val << bit_off;
    for (unsigned i = 0; i < nbytes; ++i) p[i] = (uint8_t)(v >> (8 * i));
}

namespace nv_regs {

using nvbits = unsigned __int128;

struct NVField { const char* name; uint16_t start, end; };
enum NVRegKind : uint8_t { kAbsent, kReg, kGroup };   // kGroup: an MMU field group (tinygrad's base and off are None)
struct NVRegDef {                                     // a name as NVDev.include binds it: NVReg(nvdev, base, off, fields)
    const char* name;
    NVRegKind kind;
    uint32_t base, off;                               // base: the autogen's regs_off (nv_regs/__init__.py:16-18)
    uint32_t (*index)(uint32_t);                      // an indexed register's off, else nullptr
    const NVField* fields;
    uint16_t nfields;
};
struct NVKV { const char* name; uint64_t value; };    // a keyword argument, name=value
constexpr int kNVMaxFields = 32;

// Keyword arguments or names at a call site: a braced list (valid for the call) or a vector.
template <class T> struct NVArgs {
    const T* p = nullptr;
    size_t n = 0;
    NVArgs() = default;
    NVArgs(std::initializer_list<T> l) : p(l.begin()), n(l.size()) {}
    NVArgs(const std::vector<T>& v) : p(v.data()), n(v.size()) {}
    const T* begin() const { return p; }
    const T* end() const { return p + n; }
};

[[noreturn]] inline void nvreg_fail(const NVRegDef& d, const char* what, const char* name = "") {
    fprintf(stderr, "NVReg %s: %s%s\n", d.name, what, name);
    abort();
}

inline const NVField& nvreg_field(const NVRegDef& d, const char* name) {
    for (uint16_t i = 0; i < d.nfields; ++i)
        if (strcmp(d.fields[i].name, name) == 0) return d.fields[i];
    nvreg_fail(d, "no field ", name);
}

inline uint64_t nv_getbits(nvbits v, unsigned start, unsigned end) {   // helpers.getbits (fields are at most 64 bits)
    return (uint64_t)(v >> start) & (~0ull >> (63 - (end - start)));
}

struct NVFieldValues {                                // decode's dict: every field, in the register's order
    const NVRegDef* def;
    uint64_t v[kNVMaxFields];
    uint64_t operator[](const char* name) const { return v[&nvreg_field(*def, name) - def->fields]; }
};

template <class Dev> struct NVReg {
    Dev* nvdev;
    const NVRegDef* def;
    uint32_t base, off;
    uint32_t (*index)(uint32_t);

    NVReg(Dev* nvdev_, const NVRegDef& d) : nvdev(nvdev_), def(&d), base(d.base), off(d.off), index(d.index) {
        if (d.kind == kAbsent) nvreg_fail(d, "not included on this chip");
    }
    NVReg operator[](uint32_t idx) const {                                         // nvdev.py:15
        if (!index) nvreg_fail(*def, "not an indexed register");
        NVReg r = *this;
        r.off = index(idx);
        r.index = nullptr;
        return r;
    }
    NVReg with_base(uint32_t b) const { NVReg r = *this; r.base = b + base; return r; }   // :18
    uint32_t addr() const {
        if (def->kind != kReg) nvreg_fail(*def, "an MMU field group has no address");
        if (index) nvreg_fail(*def, "an indexed register needs its index");
        return base + off;
    }
    uint32_t read() const { return nvdev->rreg(addr()); }                          // :20
    NVFieldValues read_bitfields() const { return decode(read()); }                // :21
    void write(uint32_t ini = 0, NVArgs<NVKV> kw = {}) const {                     // :23
        nvbits v = ini | encode(kw);
        if (v >> 32) nvreg_fail(*def, "a value wider than the 32-bit register");   // tinygrad's struct.pack('<I') raises
        nvdev->wreg(addr(), (uint32_t)v);
    }
    void write(NVArgs<NVKV> kw) const { write(0, kw); }
    void update(NVArgs<NVKV> kw) const {                                           // :25
        uint32_t cur = read();
        nvbits m = 0;
        for (const NVKV& a : kw) m |= mask({a.name});
        write(cur & ~(uint32_t)m, kw);
    }
    nvbits mask(NVArgs<const char*> names) const {                                 // :27-28
        nvbits m = 0;
        for (const char* nm : names) {
            const NVField& f = nvreg_field(*def, nm);
            m |= (((nvbits)1 << (f.end - f.start + 1)) - 1) << f.start;
        }
        return m;
    }
    nvbits encode(NVArgs<NVKV> kw) const {                                         // :30
        nvbits x = 0;
        for (const NVKV& a : kw) x |= (nvbits)a.value << nvreg_field(*def, a.name).start;
        return x;
    }
    NVFieldValues decode(nvbits val) const {                                       // :31
        NVFieldValues r{def, {}};
        for (uint16_t i = 0; i < def->nfields; ++i) r.v[i] = nv_getbits(val, def->fields[i].start, def->fields[i].end);
        return r;
    }
};

} // namespace nv_regs
} // namespace tinygpu_device

#endif // LIBHMSBEAGLE_GPU_TINYGPUNVREG_H
