/*
 * TinyGPUNVMemory.h
 *
 * TODO.md plan step C6: tinygrad's NV page tables and memory manager (tinygrad/runtime/support/nv/nvdev.py:33-72 at
 * a9830e2b4) on TinyGPUMemory.h, and PCIIfaceBase.alloc and free (runtime/support/system.py:267-284), ported statement by
 * statement over TinyGPU.app. NVPageTableEntry reads and writes its entries through BAR1, TinyGPU.app's VRAM window, as
 * tinygrad's does (nvdev.vram.view(paddr, 0x1000, fmt='Q')): one 8-byte MMIO_READ or MMIO_WRITE per word, a dual PDE's
 * high word read before its low word, every read tinygrad makes made here too; the MMU is invalidated by one BAR0 write
 * after every mapping. The redundant reads (valid reads an entry up to three times, address twice) are kept so the stream
 * equals tinygrad's and the L0 recordings'; they MAY be removed in the future if they affect performance (TODO.md plan step
 * C14, checked by replay first). What BEAGLE adds sends nothing: the IOVA fence (a system-memory PTE may point only into a DMA
 * segment TinyGPU.app gave this process; a stray device address faults the Mac's DART, which can panic macOS), and
 * nv_vram_end for plan step P3's WPR bound. nv_mm_import restores the manager the daemon exported at the handoff.
 */

#ifndef LIBHMSBEAGLE_GPU_TINYGPUNVMEMORY_H
#define LIBHMSBEAGLE_GPU_TINYGPUNVMEMORY_H

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

#include "libhmsbeagle/GPU/TinyGPUNVDispatch.h"
#include "libhmsbeagle/GPU/TinyGPUMemory.h"
#include "libhmsbeagle/GPU/TinyGPUNVBootTables.h"
#include "libhmsbeagle/GPU/TinyGPUNVReg.h"
#include "libhmsbeagle/GPU/TinyGPUTransport.h"

namespace tinygpu_device {

class NVMemoryManager;

// What NVPageTableEntry, NVMemoryManager and PCIIfaceBase.alloc use of NVDev and its PCI device. A failed transfer is
// tinygrad's RuntimeError.
struct NVMemDev {
    TGTransport* t = nullptr;
    int mmu_ver = 2;
    const nv_regs::NVRegDef* regs = nullptr;   // the chip's include() set: kAdaRegs (MMU v2) or kGB20xRegs (v3)
    bool is_booting = false, smi_dev = false;
    uint64_t pagesize = 0x4000;                // mmap.PAGESIZE, PCIIfaceBase.alloc's system-memory unit
    uint32_t vram_bar = 1;                     // PCIIfaceBase.vram_bar (NV: BAR1)
    NVMemoryManager* mm = nullptr;

    uint32_t rreg(uint32_t addr) {
        uint32_t v = 0;
        std::string err;
        if (!t->bulk_read(0, addr, &v, 4, err)) throw TGPyError("RuntimeError", err);
        return v;
    }
    void wreg(uint32_t addr, uint32_t value) {
        std::string err;
        if (!t->bulk_write(0, addr, &value, 4, err)) throw TGPyError("RuntimeError", err);
    }
    nv_regs::NVReg<NVMemDev> reg(nv_regs::NVRegId id) { return nv_regs::NVReg<NVMemDev>(this, regs[id]); }
    uint64_t vram_q(uint64_t off) {   // vram.view(..., fmt='Q')[i]
        uint64_t v = 0;
        std::string err;
        if (!t->bulk_read(vram_bar, off, &v, 8, err)) throw TGPyError("RuntimeError", err);
        return v;
    }
    void vram_set_q(uint64_t off, uint64_t v) {
        std::string err;
        if (!t->bulk_write(vram_bar, off, &v, 8, err)) throw TGPyError("RuntimeError", err);
    }
    void vram_zero(uint64_t paddr, uint64_t size) {   // vram[paddr:paddr+size] = bytes(size): one write
        std::vector<uint8_t> z(size);
        std::string err;
        if (!t->bulk_write(vram_bar, paddr, z.data(), size, err)) throw TGPyError("RuntimeError", err);
    }
};

// nvdev.py:33-67
class NVPageTableEntry {
public:
    using Dev = NVMemDev;
    NVMemDev* nvdev = nullptr;
    uint64_t paddr = 0;
    int lv = 0;

    NVPageTableEntry() = default;
    NVPageTableEntry(NVMemDev* d, uint64_t paddr_, int lv_) : nvdev(d), paddr(paddr_), lv(lv_) {}

    void set_entry(uint64_t entry_id, uint64_t pa, bool table = false, bool uncached = false, TGAddrSpace aspace = TGAddrSpace::PHYS,
                   bool snooped = false, int64_t frag = 0, bool valid = true) const;
    nv_regs::nvbits entry(uint64_t entry_id) const {   // the high word first, as (entries[2*i+1] << 64) | entries[2*i] evaluates
        if (!is_dual_pde()) return q(entry_id);
        nv_regs::nvbits hi = q(2 * entry_id + 1);
        return (hi << 64) | q(2 * entry_id);
    }
    nv_regs::NVFieldValues read_fields(uint64_t entry_id) const {
        if (is_page(entry_id)) return pte().decode(entry(entry_id));
        return (is_dual_pde() ? dual_pde() : pde()).decode(entry(entry_id));
    }
    bool is_page(uint64_t entry_id) const { return lv < level_cnt() - 1 ? (entry(entry_id) & 1) == 1 : true; }
    bool supports_huge_page(uint64_t pa) const;
    bool valid(uint64_t entry_id) const {
        if (is_page(entry_id)) return read_fields(entry_id)["valid"] != 0;
        return read_fields(entry_id)[is_dual_pde() ? "aperture_small" : "aperture"] != 0;
    }
    uint64_t address(uint64_t entry_id) const {
        const bool small = is_dual_pde(), sys = nvdev->mmu_ver == 2 || lv == level_cnt() - 1;
        const char* name = small ? (sys ? "address_small_sys" : "address_small") : (sys ? "address_sys" : "address");
        return read_fields(entry_id)[name] << 12;
    }

private:
    int level_cnt() const;
    bool is_dual_pde() const { return lv == level_cnt() - 2; }
    uint64_t q(uint64_t i) const { return nvdev->vram_q(paddr + 8 * i); }
    void set_q(uint64_t i, uint64_t v) const { nvdev->vram_set_q(paddr + 8 * i, v); }
    nv_regs::NVReg<NVMemDev> pte() const { return nvdev->reg(nvdev->mmu_ver == 3 ? nv_regs::NV_MMU_VER3_PTE : nv_regs::NV_MMU_VER2_PTE); }
    nv_regs::NVReg<NVMemDev> pde() const { return nvdev->reg(nvdev->mmu_ver == 3 ? nv_regs::NV_MMU_VER3_PDE : nv_regs::NV_MMU_VER2_PDE); }
    nv_regs::NVReg<NVMemDev> dual_pde() const {
        return nvdev->reg(nvdev->mmu_ver == 3 ? nv_regs::NV_MMU_VER3_DUAL_PDE : nv_regs::NV_MMU_VER2_DUAL_PDE);
    }
};

// nvdev.py:69-72. NVMemoryManager.va_allocator is TLSFAllocator((1 << 44), base=0x1000000000), one for all devices; here
// the owner's (nv_mm_import's, or a new one for a manager built from scratch).
class NVMemoryManager : public TGMemoryManager<NVPageTableEntry> {
public:
    NVMemoryManager(NVMemDev* dev_, uint64_t vram_size_, uint64_t boot_size, uint64_t va_bits_, const std::vector<uint64_t>& va_shifts_,
                    uint64_t va_base_, const std::vector<std::pair<uint64_t, uint64_t>>& palloc_ranges_, int first_lv, bool reserve_ptable_)
        : TGMemoryManager(dev_, vram_size_, boot_size, va_bits_, va_shifts_, va_base_, palloc_ranges_, first_lv, reserve_ptable_) {
        dev_->mm = this;
    }
    NVMemoryManager(NVMemDev* dev_, uint64_t vram_size_, uint64_t va_bits_, const std::vector<uint64_t>& va_shifts_, uint64_t va_base_,
                    const std::vector<std::pair<uint64_t, uint64_t>>& palloc_ranges_, bool reserve_ptable_, const Restored& r, bool& ok)
        : TGMemoryManager(dev_, vram_size_, va_bits_, va_shifts_, va_base_, palloc_ranges_, reserve_ptable_, r, ok) {
        dev_->mm = this;
    }
    void on_range_mapped() override {
        dev->reg(nv_regs::NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE).write((1u << 0) | (1u << 1) | (1u << 6) | (1u << 31));
    }
    // the IOVA fence: every system-memory page of a mapping must lie in a DMA segment TinyGPU.app gave this process, checked
    // before the mapping sends anything
    void check_mapping(const std::vector<std::pair<uint64_t, uint64_t>>& paddrs, TGAddrSpace aspace) override {
        if (aspace != TGAddrSpace::SYS) return;
        for (auto& p : paddrs)
            if (!dev->t->iova_known(p.first, p.second))
                throw TGPyError("RuntimeError", "IOVA fence: " + tg_hex(p.first) + "+" + tg_hex(p.second) + " is in no DMA segment TinyGPU.app gave this process");
    }
};

inline int NVPageTableEntry::level_cnt() const { return nvdev->mm->level_cnt; }

inline bool NVPageTableEntry::supports_huge_page(uint64_t pa) const {
    return lv >= level_cnt() - 3 && pa % nvdev->mm->pte_covers[lv] == 0;
}

inline void NVPageTableEntry::set_entry(uint64_t entry_id, uint64_t pa, bool table, bool uncached, TGAddrSpace aspace, bool snooped,
                                        int64_t frag, bool valid) const {
    (void)snooped; (void)frag;   // NV's PTE has neither
    nv_regs::nvbits x;
    const bool v3 = nvdev->mmu_ver == 3;
    if (!table) {
        x = pte().encode({{"valid", valid}, {"address_sys", pa >> 12}, {"aperture", aspace == TGAddrSpace::SYS ? 2u : 0u}, {"kind", 6},
                          {v3 ? "pcf" : "vol", uncached}});
    } else if (is_dual_pde()) {
        x = dual_pde().encode({{"is_pte", 0}, {"aperture_small", valid ? 1u : 0u}, {v3 ? "address_small" : "address_small_sys", pa >> 12},
                               {v3 ? "pcf_small" : "no_ats", v3 ? 0b10u : 1u}});
    } else {
        x = pde().encode({{"is_pte", 0}, {"aperture", valid ? 1u : 0u}, {v3 ? "address" : "address_sys", pa >> 12}, {v3 ? "pcf" : "no_ats", v3 ? 0b10u : 1u}});
    }
    if (is_dual_pde()) {
        set_q(2 * entry_id, (uint64_t)(x & 0xffffffffffffffffull));
        set_q(2 * entry_id + 1, (uint64_t)(x >> 64));
    } else {
        if (x >> 64) throw TGPyError("struct.error", "'Q' format requires 0 <= number <= 18446744073709551615");
        set_q(entry_id, (uint64_t)x);
    }
}

// PCIIfaceBase.alloc's HCQBuffer and PCIAllocationMeta
struct NVBuffer {
    uint64_t va_addr = 0, size = 0;
    TGVirtMapping mapping;
    bool has_cpu_mapping = false;
    uint64_t hMemory = 0;
    uint8_t* view = nullptr;   // system memory: the shared mapping of TinyGPU.app's fd (its first bytes held the segment list)
    uint64_t view_size = 0;
    bool bar_view = false;     // VRAM with cpu_access: a window of BAR1 at mapping.paddrs[0].first (map_bar sends nothing)
    int fd = -1;               // system memory, if asked for: a dup of the allocation's fd
};

// PCIIfaceBase.alloc (system.py:267-279). is_bar_small asks bar_info, which tinygrad caches: nothing is sent.
inline NVBuffer nv_iface_alloc(NVMemoryManager& mm, uint64_t size, bool host = false, bool uncached = false, bool cpu_access = false,
                               bool contiguous = false, bool force_devmem = false, bool zero = false, bool keep_fd = false) {
    NVMemDev& d = *mm.dev;
    uint64_t bar_addr = 0, bar_size = 0;
    std::string err;
    if (!d.t->bar_info(d.vram_bar, bar_addr, bar_size, err)) throw TGPyError("RuntimeError", err);
    const bool is_bar_small = bar_size == (256ull << 20);
    const bool should_use_sysmem = host || ((is_bar_small ? cpu_access : (uncached && cpu_access)) && !force_devmem);
    // Align size to huge pages for large allocations, otherwise the unaligned tail falls back to 4KB pages, increasing TLB pressure.
    size = tg_round_up(size, should_use_sysmem ? d.pagesize : (size >= (8ull << 20) ? (2ull << 20) : (4ull << 10)));
    NVBuffer b;
    if (should_use_sysmem) {
        size = tg_round_up(size, d.pagesize);
        uint64_t vaddr = mm.alloc_vaddr(size, d.pagesize);
        TGSysmem sm;
        if (!d.t->alloc_sysmem(size, contiguous, sm, err, keep_fd ? &b.fd : nullptr)) throw TGPyError("RuntimeError", err);
        std::vector<std::pair<uint64_t, uint64_t>> pages;
        for (uint64_t p : sm.paddrs) pages.push_back({p, 0x1000});
        b.mapping = mm.map_range(vaddr, size, pages, TGAddrSpace::SYS, true, true);
        b.va_addr = vaddr;
        b.size = size;
        b.has_cpu_mapping = true;
        b.hMemory = sm.paddrs[0];
        b.view = sm.view;
        b.view_size = sm.mapped_size;
        return b;
    }
    size = tg_round_up(size, 0x1000);
    b.mapping = mm.valloc(size, 0x1000, uncached, cpu_access, zero);
    b.va_addr = b.mapping.va_addr;
    b.size = size;
    b.has_cpu_mapping = cpu_access;
    b.hMemory = b.mapping.paddrs[0].first;
    b.bar_view = cpu_access;
    return b;
}

// PCIIfaceBase.free (system.py:281-284) for this device's own buffer on a remote device (is_local() is false): VRAM is
// unmapped and freed; system memory keeps its PTEs and its host mapping, as in tinygrad.
inline void nv_iface_free(NVMemoryManager& mm, const NVBuffer& b) {
    if (b.mapping.aspace == TGAddrSpace::PHYS) mm.vfree(b.mapping);
}

// Where the manager's VRAM allocations end (nv_dispatch_daemon.py check_vram_below_wpr, plan step P3)
inline uint64_t nv_vram_end(const NVMemoryManager& mm) {
    uint64_t end = 0;
    for (auto& [start, b] : mm.pa_allocator.blocks)
        if (!b.free) end = std::max(end, mm.pa_allocator.base + start + b.size);
    return end;
}

// The plugin's memory manager after the handoff: tinygrad's as the daemon exported it (nv_dispatch_daemon.py _mm_export).
struct NVMemState {
    NVMemDev dev;
    TLSFAllocator va;   // NVMemoryManager.va_allocator
    std::unique_ptr<NVMemoryManager> mm;
    uint64_t wpr_bound = 0;       // gspFwRsvdStart (vram_size - 512 MiB on a COT boot): no VRAM allocation may end above it
    uint64_t dev_vram_size = 0;   // NVDev.vram_size
};

// Restores st from the daemon's export (the mm_* keys of its handoff reply) over transport t, which then knows BAR1's size
// and the daemon's sysmem allocations. Returns an empty string on success, otherwise what was missing or wrong.
inline std::string nv_mm_import(const std::string& js, TGTransport* t, NVMemState& st) {
    std::string missing;
    auto u64 = [&](const char* k) { uint64_t v = 0; if (!nvd_json_u64(js, k, v)) missing += std::string(missing.empty() ? "" : ", ") + k; return v; };
    auto u64s = [&](const char* k) { std::vector<uint64_t> v; if (!nvd_json_u64s(js, k, v)) missing += std::string(missing.empty() ? "" : ", ") + k; return v; };
    uint64_t mmu_ver = u64("mm_mmu_ver"), vram_size = u64("mm_vram_size"), va_bits = u64("mm_va_bits"), va_base = u64("mm_va_base");
    uint64_t reserve = u64("mm_reserve_ptable"), root = u64("mm_root"), root_lv = u64("mm_root_lv"), pagesize = u64("mm_pagesize");
    uint64_t gmmu = u64("mm_gmmu"), sysmem_count = u64("mm_sysmem_count"), bar1_size = u64("bar1_size");
    st.wpr_bound = u64("mm_wpr_bound");
    st.dev_vram_size = u64("mm_dev_vram_size");
    std::vector<uint64_t> shifts = u64s("mm_va_shifts"), ranges = u64s("mm_palloc_ranges");
    std::vector<uint64_t> boot = u64s("mm_boot"), ptable = u64s("mm_ptable"), pa = u64s("mm_pa"), va = u64s("mm_va");
    if (!missing.empty()) return "the handoff has no " + missing;
    if (gmmu == 0) return "GMMU=0: tinygrad's identity mapping is not ported";
    if ((mmu_ver != 2 && mmu_ver != 3) || ranges.empty() || ranges.size() % 2 || shifts.empty() || root_lv >= shifts.size() || !pagesize)
        return "a malformed memory-manager export";
    st.dev.t = t;
    st.dev.mmu_ver = (int)mmu_ver;
    st.dev.regs = mmu_ver == 3 ? nv_regs::kGB20xRegs : nv_regs::kAdaRegs;
    st.dev.pagesize = pagesize;
    std::vector<std::pair<uint64_t, uint64_t>> pr;
    for (size_t i = 0; i < ranges.size(); i += 2) pr.push_back({ranges[i], ranges[i + 1]});
    if (!st.va.restore(va)) return "a malformed VA allocator state";
    bool ok = false;
    st.mm = std::make_unique<NVMemoryManager>(&st.dev, vram_size, va_bits, shifts, va_base, pr, reserve != 0,
                                              NVMemoryManager::Restored{boot, ptable, pa, root, (int)root_lv}, ok);
    if (!ok) { st.mm.reset(); return "a malformed allocator state"; }
    st.mm->va_allocator = &st.va;
    t->seed_bar(st.dev.vram_bar, bar1_size);   // the daemon mapped it (a MAP_BAR here would change the stream)
    t->seed_sysmem_count((int)sysmem_count);
    return "";
}

} // namespace tinygpu_device

#endif // LIBHMSBEAGLE_GPU_TINYGPUNVMEMORY_H
