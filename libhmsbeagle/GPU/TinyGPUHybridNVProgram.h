/*
 * TinyGPUHybridNVProgram.h
 *
 * C++ port of tinygrad's NV program loading, as BEAGLE uses it: elf_loader()
 * (tinygrad/runtime/support/elf.py) and NVProgram.__init__ with BEAGLE's two
 * multi-kernel-ELF fixes (BeagleNVProgram in nv_dispatch_daemon.py), plus the
 * device-side sizing NVProgram.__init__ triggers (NVDevice's
 * _ensure_has_local_memory, and where tinygrad would place each VRAM
 * allocation), for the C++ runtime (TODO.md "Runtime roadmap", Step 3). It produces the same per-kernel
 * records the daemon's build_handoff does (NVDKernel: QMD template, cbuf0
 * prefix, kernargs layout) plus the relocated program image. The code follows
 * tinygrad's statement by statement. QMD field positions come from
 * TinyGPUNVTables.h, generated from tinygrad's own tables.
 *
 * One deliberate difference: tinygrad uploads a separate copy of the image
 * per program (it assumes one kernel per ELF); here all kernels share one
 * upload at lib_va, which gives every kernel the same relocated bytes and
 * addresses its own copy would have had at that base.
 */

#ifndef LIBHMSBEAGLE_GPU_TINYGPUHYBRIDNVPROGRAM_H
#define LIBHMSBEAGLE_GPU_TINYGPUHYBRIDNVPROGRAM_H

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "libhmsbeagle/GPU/TinyGPUElf.h"
#include "libhmsbeagle/GPU/TinyGPUNVTables.h"
#include "libhmsbeagle/GPU/TinyGPUHybridNVDispatch.h"
#include "libhmsbeagle/GPU/TinyGPUPool.h"

namespace tinygpu_device {

// ── NVProgram.__init__ ──────────────────────────────────────────────────────

struct NVDProgramParams {               // the NVDevice state NVProgram.__init__ reads
    uint32_t compute_class = 0;         // >= nvt::BLACKWELL_COMPUTE_A selects QMD v5
    uint64_t lib_va = 0;                // GPU VA of the (shared) uploaded image
    uint32_t slm_per_thread = 0;        // dev.slm_per_thread after _ensure_has_local_memory
    uint64_t shared_mem_window = 0x729400000000ull, local_mem_window = 0x729300000000ull;
    uint32_t sass_version = 0;
    bool fill_launch_dims = true;       // BeagleNVProgram's launch-dims fill
};

struct NVDProgramUsage { uint32_t regs = 0, shmem = 0x400, lcmem = 0x240, cbuf0_size = 0; };  // NVProgram's defaults

static inline uint64_t nvd_round_up(uint64_t x, uint64_t a) { return (x + a - 1) / a * a; }

// NVProgram._parse_elf_info: EIATTR records, (format, attribute, payload).
template <typename F> static inline bool nvd_eiattrs(const NVDElf& elf, const NVDElfSection& sh, F&& f) {
    for (uint64_t off = 0; off < sh.size; ) {
        if (off + 4 > sh.size) return false;
        const uint8_t* p = elf.content(sh) + off;
        uint8_t typ = p[0], param = p[1];
        uint16_t sz = nvd_rd<uint16_t>(p + 2);
        if (typ == 0x4 && off + 4 + sz > sh.size) return false;
        f(typ, param, p + 4, sz);
        off += (typ == 0x4 ? sz : 0) + 4;
    }
    return true;
}

// The kernel's symbol-table index (BeagleNVProgram's my_sym_idx), or -1.
static inline long nvd_symbol_index(const NVDElf& elf, const std::string& name) {
    for (const NVDElfSection& s : elf.sections) {
        if (s.type != 2 /* SHT_SYMTAB */ || s.entsize == 0 || s.link >= elf.sections.size()) continue;
        const NVDElfSection& strtab = elf.sections[s.link];
        for (uint64_t i = 0; (i + 1) * s.entsize <= s.size; ++i)
            if (nvd_cstr(elf.content(strtab), strtab.size, nvd_rd<uint32_t>(elf.content(s) + i * s.entsize)) == name) return (long)i;
        return -1;
    }
    return -1;
}

// `.nv.constant<N>[.<kernel>]` (BeagleNVProgram's regex): the bank number, or -1 when the section is not a constant
// bank of this kernel (another kernel's suffix).
static inline int nvd_constant_bank(const std::string& sec, const std::string& kernel) {
    const std::string pre = ".nv.constant";
    if (sec.compare(0, pre.size(), pre) != 0) return -1;
    size_t i = pre.size(), j = i;
    while (j < sec.size() && sec[j] >= '0' && sec[j] <= '9') ++j;
    if (j == i) return -1;
    if (j < sec.size() && (sec[j] != '.' || j + 1 == sec.size() || sec.compare(j + 1, std::string::npos, kernel) != 0)) return -1;
    return std::stoi(sec.substr(i, j - i));
}

// Registers, shared and local memory, and the cbuf0 driver-param size, as NVProgram.__init__'s section loop finds
// them (with BeagleNVProgram's symbol-index filter). Returns an empty string on success.
static inline std::string nvd_program_usage(const NVDElf& elf, const std::string& name, NVDProgramUsage& u) {
    long my_sym_idx = nvd_symbol_index(elf, name);
    std::string err;
    for (const NVDElfSection& sh : elf.sections) {
        if (sh.name == ".nv.shared." + name) u.shmem = (uint32_t)nvd_round_up(0x400 + sh.size, 128);
        if (sh.name.compare(0, 8, ".nv.info") != 0) continue;
        bool ok = nvd_eiattrs(elf, sh, [&](uint8_t typ, uint8_t param, const uint8_t* data, uint16_t sz) {
            if (sh.name == ".nv.info." + name && param == 0xa) {
                if (typ != 0x4 || sz < 6) { err = "EIATTR_PARAM_CBANK without payload"; return; }
                u.cbuf0_size = nvd_rd<uint16_t>(data + 4);                          // struct.unpack_from("IH", data)[1]
            } else if (sh.name == ".nv.info" && (param == 0x12 || param == 0x2f) && my_sym_idx >= 0 && typ == 0x4 && sz >= 8 &&
                       nvd_rd<uint32_t>(data) == (uint32_t)my_sym_idx) {
                if (param == 0x12) u.lcmem = nvd_rd<uint32_t>(data + 4) + 0x240;     // EIATTR_MIN_STACK_SIZE
                else u.regs = nvd_rd<uint32_t>(data + 4);                           // EIATTR_REGCOUNT
            }
        });
        if (!ok) return "truncated " + sh.name;
        if (!err.empty()) return err;
    }
    return "";
}

// NVProgram.__init__'s relocation loop, applied to the shared image at base lib_va (addends ignored, as tinygrad does).
static inline std::string nvd_relocate(const NVDElf& elf, uint64_t lib_va, std::vector<uint8_t>& image) {
    image = elf.image;
    for (const NVDElfReloc& r : elf.relocs) {
        uint64_t v = lib_va + r.sym_off;
        uint32_t lo = (uint32_t)v, hi = (uint32_t)(v >> 32);
        if (r.type == 2) { if (r.image_off + 8 > image.size()) return "relocation past image"; memcpy(&image[r.image_off], &v, 8); }
        else if (r.type == 0x38) { if (r.image_off + 8 > image.size()) return "relocation past image"; memcpy(&image[r.image_off + 4], &lo, 4); }
        else if (r.type == 0x39) { if (r.image_off + 8 > image.size()) return "relocation past image"; memcpy(&image[r.image_off + 4], &hi, 4); }
        else return "unknown NV reloc " + std::to_string(r.type);
    }
    return "";
}

// QMD.write for one field, raising (here: failing) like tinygrad's _rw_bits when the value does not fit.
static inline bool nvd_qmd_write(std::vector<uint8_t>& qmd, nvt::Bits b, uint64_t value) {
    if (!nvt::present(b)) return false;
    uint32_t range[2] = { b.hi, b.lo };
    if (value >> (b.hi - b.lo + 1)) return false;
    nvd_qmd_bits(qmd.data(), range, value);
    return true;
}

// NVProgram.__init__ (BeagleNVProgram) for one kernel, after its usage is known: QMD template, cbuf0 prefix and
// kernargs layout, into the same NVDKernel record build_handoff produces. Returns an empty string on success.
static inline std::string nvd_load_program(const NVDElf& elf, const std::string& name, const NVDProgramParams& p,
                                           const NVDProgramUsage& u, NVDKernel& k) {
    using namespace nvt;
    const bool v5 = p.compute_class >= BLACKWELL_COMPUTE_A;
    const Bits* F = v5 ? kQmdV5 : kQmdV3;
    const Bits (*FI)[8] = v5 ? kQmdV5Indexed : kQmdV3Indexed;

    std::vector<std::pair<int, std::pair<uint64_t, uint64_t>>> constbufs = { { 0, { 0, 0x160 } } };  // dict order
    uint64_t prog_addr = p.lib_va, prog_sz = elf.image.size();
    for (const NVDElfSection& sh : elf.sections) {
        if (sh.name == ".text." + name) { prog_addr = p.lib_va + sh.addr; prog_sz = sh.size; continue; }
        int bank = nvd_constant_bank(sh.name, name);
        if (bank < 0) continue;
        auto it = std::find_if(constbufs.begin(), constbufs.end(), [&](const std::pair<int, std::pair<uint64_t, uint64_t>>& c) { return c.first == bank; });
        if (bank >= 8) return name + ": constant bank " + std::to_string(bank) + " out of range";
        if (it != constbufs.end()) it->second = { p.lib_va + sh.addr, sh.size };
        else constbufs.push_back({ bank, { p.lib_va + sh.addr, sh.size } });
    }
    if (u.lcmem > p.slm_per_thread) return name + ": needs more local memory than slm_per_thread";

    k = NVDKernel();
    k.name = name;
    k.prefix.assign(std::max<uint64_t>(u.cbuf0_size / 4, v5 ? 224 : 12), 0);
    auto lo = [](uint64_t v) { return (uint32_t)v; };
    auto hi = [](uint64_t v) { return (uint32_t)(v >> 32); };
    if (v5) {
        uint32_t w[4] = { lo(p.shared_mem_window), hi(p.shared_mem_window), lo(p.local_mem_window), hi(p.local_mem_window) };
        std::copy(w, w + 4, k.prefix.begin() + 188);
        k.prefix[223] = 0xfffdc0;
    } else {
        uint32_t w[6] = { lo(p.shared_mem_window), hi(p.shared_mem_window), lo(p.local_mem_window), hi(p.local_mem_window), 0xfffdc0, 0 };
        std::copy(w, w + 6, k.prefix.begin() + 6);
    }

    uint64_t smem_cfg = 0;
    for (uint64_t c : { 32, 64, 100 }) if (c * 1024 >= u.shmem) { smem_cfg = c * 1024 / 4096 + 1; break; }
    if (!smem_cfg) return name + ": shared memory above 100 KiB";

    k.qmd.assign(v5 ? kQmdV5Bytes : kQmdV3Bytes, 0);
    std::vector<std::pair<QmdField, uint64_t>> w;
    if (v5) w = { { QMD_MAJOR_VERSION, 5 }, { QMD_TYPE, QMD_TYPE_GRID_CTA }, { PROGRAM_ADDRESS_UPPER_SHIFTED4, hi(prog_addr >> 4) },
                  { PROGRAM_ADDRESS_LOWER_SHIFTED4, lo(prog_addr >> 4) }, { REGISTER_COUNT, u.regs },
                  { SHARED_MEMORY_SIZE_SHIFTED7, u.shmem >> 7 }, { SHADER_LOCAL_MEMORY_HIGH_SIZE_SHIFTED4, p.slm_per_thread >> 4 } };
    else    w = { { QMD_MAJOR_VERSION, 3 }, { SM_GLOBAL_CACHING_ENABLE, 1 }, { PROGRAM_ADDRESS_UPPER, hi(prog_addr) },
                  { PROGRAM_ADDRESS_LOWER, lo(prog_addr) }, { SHARED_MEMORY_SIZE, u.shmem }, { REGISTER_COUNT_V, u.regs },
                  { SHADER_LOCAL_MEMORY_HIGH_SIZE, p.slm_per_thread } };
    w.insert(w.end(), { { QMD_GROUP_ID, 0x3f }, { INVALIDATE_TEXTURE_HEADER_CACHE, 1 }, { INVALIDATE_TEXTURE_SAMPLER_CACHE, 1 },
                        { INVALIDATE_TEXTURE_DATA_CACHE, 1 }, { INVALIDATE_SHADER_DATA_CACHE, 1 }, { API_VISIBLE_CALL_LIMIT, 1 },
                        { SAMPLER_INDEX, 1 }, { BARRIER_COUNT, 1 }, { CWD_MEMBAR_TYPE, CWD_MEMBAR_TYPE_L1_SYSMEMBAR },
                        { CONSTANT_BUFFER_INVALIDATE_0, 1 }, { MIN_SM_CONFIG_SHARED_MEM_SIZE, smem_cfg },
                        { TARGET_SM_CONFIG_SHARED_MEM_SIZE, smem_cfg }, { MAX_SM_CONFIG_SHARED_MEM_SIZE, 0x1a },
                        { PROGRAM_PREFETCH_SIZE, std::min<uint64_t>(prog_sz >> 8, 0x1ff) }, { SASS_VERSION, p.sass_version },
                        { PROGRAM_PREFETCH_ADDR_UPPER_SHIFTED, prog_addr >> 40 }, { PROGRAM_PREFETCH_ADDR_LOWER_SHIFTED, prog_addr >> 8 } });
    for (auto& f : w)
        if (!nvd_qmd_write(k.qmd, F[f.first], f.second)) return name + ": QMD field " + std::to_string(f.first) + " does not fit";
    for (auto& c : constbufs) {  // QMD.set_constant_buf_addr, then size and valid
        uint64_t addr = c.second.first >> (v5 ? 6 : 0);
        bool ok = nvd_qmd_write(k.qmd, FI[v5 ? CONSTANT_BUFFER_ADDR_UPPER_SHIFTED6 : CONSTANT_BUFFER_ADDR_UPPER][c.first], hi(addr)) &&
                  nvd_qmd_write(k.qmd, FI[v5 ? CONSTANT_BUFFER_ADDR_LOWER_SHIFTED6 : CONSTANT_BUFFER_ADDR_LOWER][c.first], lo(addr)) &&
                  nvd_qmd_write(k.qmd, FI[CONSTANT_BUFFER_SIZE_SHIFTED4][c.first], c.second.second) &&
                  nvd_qmd_write(k.qmd, FI[CONSTANT_BUFFER_VALID][c.first], 1);
        if (!ok) return name + ": constant buffer " + std::to_string(c.first) + " does not fit the QMD";
    }

    uint64_t regs_bytes = nvd_round_up(std::max<uint32_t>(1, u.regs) * 32, 256);
    k.max_threads = (uint32_t)(((65536 / regs_bytes) / 4) * 4 * 32);
    k.qmd_off = (uint32_t)nvd_round_up(constbufs[0].second.second, 256);
    k.slot_size = k.qmd_off + (8 << 8);                        // kernargs_alloc_size
    k.prefix_words = (uint32_t)k.prefix.size();
    k.dims_b = p.fill_launch_dims ? (v5 ? 216 : 0) : NVDHandoff::kNoDims;
    k.dims_g = p.fill_launch_dims ? (v5 ? 220 : 3) : NVDHandoff::kNoDims;
    return "";
}

// ── the boot-only handoff (nv_dispatch_daemon.py cmd_handoff, programs=false) ─

// What NVProgram.__init__ and NVDevice._ensure_has_local_memory read from the
// device, and the VRAM pool this side allocates from.
struct NVDRuntime {
    uint32_t compute_class = 0, sass_version = 0;
    uint64_t shared_mem_window = 0, local_mem_window = 0;
    uint32_t num_gpcs = 0, num_tpc_per_gpc = 0, num_sm_per_tpc = 0, max_warps_per_sm = 0;
    NVDBuffer pool;
};

// Returns an empty string on success, otherwise what was missing. (The daemon's runtime reply, which the goldens parse from
// the oracle.)
static inline std::string nvd_parse_runtime(const std::string& js, NVDRuntime& rt) {
    std::string missing;
    auto u64 = [&](const char* key) -> uint64_t {
        uint64_t v = 0;
        if (!nvd_json_u64(js, key, v)) missing += std::string(missing.empty() ? "" : ", ") + key;
        return v;
    };
    rt.compute_class = (uint32_t)u64("compute_class");     rt.sass_version = (uint32_t)u64("sass_version");
    rt.shared_mem_window = u64("shared_mem_window");       rt.local_mem_window = u64("local_mem_window");
    rt.num_gpcs = (uint32_t)u64("num_gpcs");               rt.num_tpc_per_gpc = (uint32_t)u64("num_tpc_per_gpc");
    rt.num_sm_per_tpc = (uint32_t)u64("num_sm_per_tpc");   rt.max_warps_per_sm = (uint32_t)u64("max_warps_per_sm");
    rt.pool.va = u64("pool_va");  rt.pool.size = u64("pool_size");
    return missing.empty() ? "" : "missing " + missing;
}

// The QMD templates come from TinyGPUNVTables.h and the launch encoding from
// the handoff (tinygrad at run time); both must describe the same QMD layout.
static inline std::string nvd_check_tables(const NVDHandoff& h, uint32_t compute_class) {
    using namespace nvt;
    const bool v5 = compute_class >= BLACKWELL_COMPUTE_A;
    const Bits* F = v5 ? kQmdV5 : kQmdV3;
    const Bits (*FI)[8] = v5 ? kQmdV5Indexed : kQmdV3Indexed;
    auto byte = [](Bits b) { return (uint32_t)b.lo / 8; };
    auto same = [](const uint32_t r[2], Bits b) { return r[0] == b.hi && r[1] == b.lo; };
    bool ok = h.qmd_ver == (v5 ? 5u : 3u) && h.qmd_bytes == (v5 ? kQmdV5Bytes : kQmdV3Bytes) && h.q_cb_shift == (v5 ? 6u : 0u) &&
              h.q_grid == byte(F[v5 ? GRID_WIDTH : CTA_RASTER_WIDTH]) && h.q_block01 == byte(F[CTA_THREAD_DIMENSION0]) &&
              h.q_block2 == byte(F[CTA_THREAD_DIMENSION2]) &&
              h.q_rel_addr == byte(F[v5 ? RELEASE_SEMAPHORE0_ADDR_LOWER : RELEASE0_ADDRESS_LOWER]) &&
              h.q_rel_payload == byte(F[v5 ? RELEASE_SEMAPHORE0_PAYLOAD_LOWER : RELEASE0_PAYLOAD_LOWER]) &&
              same(h.q_cb_hi, FI[v5 ? CONSTANT_BUFFER_ADDR_UPPER_SHIFTED6 : CONSTANT_BUFFER_ADDR_UPPER][0]) &&
              same(h.q_cb_lo, FI[v5 ? CONSTANT_BUFFER_ADDR_LOWER_SHIFTED6 : CONSTANT_BUFFER_ADDR_LOWER][0]) &&
              same(h.q_rel_en, F[RELEASE0_ENABLE]) && same(h.q_dep_ptr, F[DEPENDENT_QMD0_POINTER]) &&
              same(h.q_dep_action, F[DEPENDENT_QMD0_ACTION]) && same(h.q_dep_prefetch, F[DEPENDENT_QMD0_PREFETCH]) &&
              same(h.q_dep_enable, F[DEPENDENT_QMD0_ENABLE]);
    return ok ? "" : "TinyGPUNVTables.h does not match the running tinygrad's QMD layout (rerun make_tinygpu_nv_tables.py)";
}

// NVDevice._ensure_has_local_memory for slm_per_thread (already a multiple of
// 32): the shader local memory size, and bytes_per_tpc for its setup.
static inline uint64_t nvd_local_mem_size(const NVDRuntime& rt, uint32_t slm_per_thread, uint64_t& bytes_per_tpc) {
    bytes_per_tpc = nvd_round_up(nvd_round_up((uint64_t)slm_per_thread * 32, 0x200) * rt.max_warps_per_sm * rt.num_sm_per_tpc, 0x8000);
    return nvd_round_up(bytes_per_tpc * rt.num_tpc_per_gpc * rt.num_gpcs, 0x20000);
}

// A VRAM allocation placed as tinygrad would place its own (PCIIfaceBase.alloc,
// then MemoryManager.alloc_vaddr), but carved from the pool: the size rounds up
// to 2 MiB from 8 MiB on and to 4 KiB below, and the address is aligned to the
// largest power of two not above the size. pos is the pool's fill level. With
// freed (plan step C14), the blocks freed so far are tried first, and the block
// is recorded for its free. Returns 0 when the pool is full.
static inline uint64_t nvd_pool_alloc(const NVDBuffer& pool, uint64_t& pos, uint64_t size, TGPoolFree* freed = nullptr) {
    size = nvd_round_up(std::max<uint64_t>(size, 1), size >= (8ull << 20) ? (2ull << 20) : 0x1000);
    uint64_t align = std::max<uint64_t>(0x1000, 1ull << (63 - __builtin_clzll(size)));
    uint64_t off;
    if (freed && freed->take(pool.va, size, align, off)) return pool.va + off;
    uint64_t va = nvd_round_up(pool.va + pos, align);
    if (va + size > pool.va + pool.size) return 0;
    pos = va + size - pool.va;
    if (freed) freed->live[va - pool.va] = size;
    return va;
}

} // namespace tinygpu_device

#endif // LIBHMSBEAGLE_GPU_TINYGPUHYBRIDNVPROGRAM_H
