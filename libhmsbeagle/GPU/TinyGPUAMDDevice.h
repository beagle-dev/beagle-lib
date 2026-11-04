/*
 * TinyGPUAMDDevice.h -- TODO.md plan step A2g: what tinygrad's AMDDevice.__init__ allocates and sets up after PCIIface's
 * boot (tinygrad/runtime/ops_amd.py:997-1084 at a9830e2b4, with PCIIface.create_queue, :927-939, and HCQCompiled.__init__,
 * runtime/support/hcq.py:387-412), then the daemon's cmd_handoff (amd_dispatch_daemon.py: a synchronize, the VRAM pool and
 * the staging buffer), in C++ on TinyGPUAMDBoot.h's AMDev, so that the plugin's C++ runtime (TinyGPUAMDRuntime.h)
 * runs on objects this process made itself. In tinygrad's order, every buffer through PCIIfaceBase.alloc
 * (runtime/support/system.py:267-281) as AMDAllocator._alloc calls it:
 *   - the compute queue (create_queue: the 16 MiB ring and the 0x100 gart in sysmem, the 0x1000 EOP buffer in VRAM, then
 *     AM_GFX.setup_ring), then SDMA queue 0 (its ring and gart, then AM_SDMA.setup_ring);
 *   - AMDAllocator: HCQAllocatorBase's 32 copy buffers (2 MiB each, host). The C++ runtime copies through its own staging,
 *     but they are kept so the requests and the VRAM and VA layouts are tinygrad's (a C14-style deviation could drop them);
 *   - HCQCompiled: the signal page (0x1000, host, uncached, cpu access: a 16 KiB allocation of 1024 slots), whose last two
 *     slots are the timeline signal and its shadow, each set to 0; then kernargs_buf (16 MiB, cpu access: sysmem on this
 *     small-BAR card);
 *   - _ensure_has_local_memory(128): the 24 MiB scratch in VRAM (the C++ runtime sizes its own from the pool, plan step A1d);
 *   - cmd_handoff: synchronize (the timeline is idle; the IH drain), the pool (default half the VRAM) and 16 MiB of staging.
 * am_handoff fills the AMDHandoff that cmd_handoff's JSON would, from these objects; amd_runtime_attach_mapped attaches the
 * runtime to the transport's own mappings. am_device_fini is what the daemon's exit runs: HCQCompiled.finalize (its
 * synchronize: the IH drain), then AMDev.fini; am_device_fini_safe adds whether it saw every queue off (plan step A2k).
 * A failure is TGPyError, as the boot's.
 */

#ifndef LIBHMSBEAGLE_GPU_TINYGPUAMDDEVICE_H
#define LIBHMSBEAGLE_GPU_TINYGPUAMDDEVICE_H

#include <cstdint>
#include <string>
#include <vector>

#include <unistd.h>

#include "libhmsbeagle/GPU/TinyGPUAMDBoot.h"
#include "libhmsbeagle/GPU/TinyGPUAMDRuntime.h"

namespace tinygpu_device {
namespace amboot {

struct AMBuffer {   // an HCQBuffer and its PCIAllocationMeta
    uint64_t va_addr = 0, size = 0;
    bool sysmem = false;   // a MAP_SYSMEM_FD allocation, mapped SYS
    TGSysmem sm;           // its mapping (view, mapped_size) and pages
    TGVirtMapping mapping;
};
struct AMQueueDesc {       // AMDQueueDesc, with the buffers create_queue made for it
    AMBuffer ring, gart, eop;
    uint64_t doorbell = 0;  // doorbell64.view(doorbell_index * 8, ...).off: a byte offset in BAR2
    uint64_t put_value = 0;
};
struct AMDDeviceState {
    AMQueueDesc compute, sdma;
    std::vector<AMBuffer> copy_bufs;
    AMBuffer signal_page, kernargs_buf, scratch, pool, staging;
    uint64_t timeline_slot = 0, shadow_slot = 0;   // the two signals' offsets in the signal page
    uint64_t timeline_value = 1;
    uint32_t target_major = 0, xccs = 0, se_cnt = 0, cu_cnt = 0, max_slots_scratch_cu = 0, lds_size_in_kb = 0;
};

inline uint64_t am_pagesize() { return (uint64_t)getpagesize(); }   // mmap.PAGESIZE

// PCIIfaceBase.alloc (system.py:267-281) for AMD's vram_bar 0. The BAR view of a cpu-access VRAM buffer sends nothing.
inline AMBuffer am_iface_alloc(AMDev& adev, uint64_t size, bool host = false, bool uncached = false, bool cpu_access = false,
                               bool contiguous = false, bool force_devmem = false, bool zero = false) {
    std::string err;
    uint64_t addr, bar0;
    if (!adev.t.bar_info(0, addr, bar0, err)) throw TGPyError("RuntimeError", err);   // is_bar_small: bar_info is cached
    const bool is_bar_small = bar0 == (256ull << 20);
    const bool should_use_sysmem = host || ((is_bar_small ? cpu_access : (uncached && cpu_access)) && !force_devmem);
    const uint64_t ps = am_pagesize();
    // Align size to huge pages for large allocations, otherwise the unaligned tail falls back to 4KB pages, increasing TLB pressure.
    size = tg_round_up(size, should_use_sysmem ? ps : (size >= (8ull << 20) ? (2ull << 20) : (4ull << 10)));
    AMBuffer b;
    if (should_use_sysmem) {
        size = tg_round_up(size, ps);
        const uint64_t vaddr = adev.mm->alloc_vaddr(size, ps);
        if (!adev.t.alloc_sysmem(size, contiguous, b.sm, err)) throw TGPyError("RuntimeError", err);
        std::vector<std::pair<uint64_t, uint64_t>> paddrs;
        for (uint64_t p : b.sm.paddrs) paddrs.push_back({p, 0x1000});
        b.mapping = adev.mm->map_range(vaddr, size, paddrs, TGAddrSpace::SYS, true, true);   // snooped=True, uncached=True
        b.va_addr = vaddr;
        b.size = size;
        b.sysmem = true;
        return b;
    }
    size = tg_round_up(size, 0x1000);
    b.mapping = adev.mm->valloc(size, 0x1000, uncached, cpu_access, zero);
    b.va_addr = b.mapping.va_addr;
    b.size = size;
    return b;
}

// AMDAllocator._alloc (ops_amd.py:646-647): cpu_access as asked, since the device has an SDMA queue
inline AMBuffer am_alloc(AMDev& adev, uint64_t size, bool host = false, bool uncached = false, bool cpu_access = false) {
    return am_iface_alloc(adev, size, host, uncached, cpu_access);
}

// AMDDevice.create_queue (ops_amd.py:1086-1104) with PCIIface.create_queue (:927-939): no AQL, no CWSR buffer for AM
inline AMQueueDesc am_create_queue(AMDev& adev, bool sdma, uint64_t ring_size, uint64_t eop_buffer_size, int idx = 0) {
    AMQueueDesc q;
    q.ring = am_iface_alloc(adev, ring_size, false, true, true);
    q.gart = am_iface_alloc(adev, 0x100, false, true, true);
    if (eop_buffer_size) q.eop = am_iface_alloc(adev, eop_buffer_size);
    const uint64_t rptr = am::hsa::amd_queue_t_read_dispatch_id_offset, wptr = am::hsa::amd_queue_t_write_dispatch_id_offset;
    uint64_t doorbell_index;
    if (sdma) doorbell_index = adev.sdma->setup_ring(q.ring.va_addr, q.ring.size, q.gart.va_addr + rptr, q.gart.va_addr + wptr, idx);
    else doorbell_index = adev.gfx->setup_ring(q.ring.va_addr, q.ring.size, q.gart.va_addr + rptr, q.gart.va_addr + wptr, q.eop.va_addr, q.eop.size,
                                               0, false);   // idx=is_aql, aql=is_aql: both False
    q.doorbell = doorbell_index * 8;
    q.put_value = 0;
    return q;
}

// PCIIface._compute_props (ops_amd.py:910-925) on the card's gc_info (v1: the RX 7900 XT's v1.2, the RX 9070 XT's v1.3, whose
// fields of the same names it reads), then AMDDevice.__init__'s counts (:1003-1012)
inline void am_props(AMDev& adev, AMDDeviceState& d) {
    auto props = [&](const auto* gi) {
        const uint32_t cu_per_sa = 2 * (gi->gc_num_wgp0_per_sa + gi->gc_num_wgp1_per_sa), max_sh_per_se = gi->gc_num_sa_per_se;
        const uint32_t xccs = (uint32_t)adev.gfx->xccs, array_count = max_sh_per_se * gi->gc_num_se * xccs;
        const uint32_t simd_count = 2 * cu_per_sa * array_count, simd_per_cu = 2;
        d.target_major = (uint32_t)adev.ip_ver.at(GC)[0];
        d.xccs = xccs;
        d.se_cnt = array_count / max_sh_per_se / xccs;
        d.cu_cnt = simd_count / simd_per_cu / xccs;
        d.max_slots_scratch_cu = gi->gc_max_scratch_slots_per_cu;
        d.lds_size_in_kb = gi->gc_lds_size;
    };
    if (((const am::struct_gc_info_v1_0*)adev.gc_info.data())->header.version_minor == 3) props((const am::struct_gc_info_v1_3*)adev.gc_info.data());
    else props((const am::struct_gc_info_v1_2*)adev.gc_info.data());   // parse_discovery keeps v1.2 or v1.3 only
}

// AMDDevice.__init__ after PCIIface's boot (ops_amd.py:1000-1084), its allocations in tinygrad's order
inline void am_device_init(AMDev& adev, AMDDeviceState& d) {
    am_props(adev, d);
    d.compute = am_create_queue(adev, false, 16ull << 20, 0x1000);   // not AQL: one XCC
    d.sdma = am_create_queue(adev, true, 16ull << 20, 0, 0);         // sdma_queue(0)
    // AMDAllocator(self): HCQAllocatorBase's copy buffers (hcq.py:527-529), _alloc(batch_size, BufferSpec(host=True)) x batch_cnt
    for (int i = 0; i < 32; ++i) d.copy_bufs.push_back(am_alloc(adev, 2ull << 20, true));
    // HCQCompiled.__init__: the timeline signals, popped from the end of a new signal page (hcq.py:442-448), each set to 0
    d.signal_page = am_alloc(adev, 0x1000, true, true, true);
    d.timeline_slot = d.signal_page.size - 16;
    d.shadow_slot = d.signal_page.size - 32;
    *(volatile uint64_t*)(d.signal_page.sm.view + d.timeline_slot) = 0;
    *(volatile uint64_t*)(d.signal_page.sm.view + d.shadow_slot) = 0;
    d.kernargs_buf = am_alloc(adev, 16ull << 20, false, false, true);
    // _ensure_has_local_memory(128) (ops_amd.py:1121-1129): _realloc(None, size_per_xcc * xccs)
    const uint64_t lanes_per_wave = 64, mem_alignment_size = 256;
    const uint64_t size_per_thread = tg_round_up(128, mem_alignment_size / lanes_per_wave);
    const uint64_t size_per_xcc = size_per_thread * lanes_per_wave * d.max_slots_scratch_cu * d.cu_cnt;
    d.scratch = am_alloc(adev, size_per_xcc * d.xccs);
}

// cmd_handoff's allocations and its reply, as this process's own AMDHandoff: maps[i] is mapping i's host address
inline void am_handoff(AMDev& adev, AMDDeviceState& d, uint64_t pool_size, AMDHandoff& h, std::vector<uint8_t*>& maps) {
    // dev.synchronize(): HCQCompiled.synchronize (the timeline signal is at timeline_value - 1 = 0 already), then the IH drain
    adev.ih->drain();
    d.pool = am_alloc(adev, pool_size ? pool_size : adev.vram_size / 2);   // BufferSpec(nolru=True)
    d.staging = am_alloc(adev, 16ull << 20, true);
    h = AMDHandoff{};
    maps.clear();
    std::vector<const AMBuffer*> order;   // place(): mappings in the order the objects name them
    auto place = [&](AMDHandoffObj& o, const AMBuffer& b, uint64_t off) {
        size_t i = 0;
        while (i < order.size() && order[i] != &b) ++i;
        if (i == order.size()) { order.push_back(&b); maps.push_back(b.sm.view); h.map_size[i] = b.sm.mapped_size; }
        o.map = i;
        o.off = off;
    };
    const uint64_t rptr = am::hsa::amd_queue_t_read_dispatch_id_offset, wptr = am::hsa::amd_queue_t_write_dispatch_id_offset;
    place(h.compute_ring, d.compute.ring, 0);
    place(h.compute_rptr, d.compute.gart, rptr);
    place(h.compute_wptr, d.compute.gart, wptr);
    h.compute_ring_size = d.compute.ring.sm.mapped_size;
    h.compute_doorbell = d.compute.doorbell;
    h.compute_put = d.compute.put_value;
    place(h.sdma_ring, d.sdma.ring, 0);
    place(h.sdma_rptr, d.sdma.gart, rptr);
    place(h.sdma_wptr, d.sdma.gart, wptr);
    h.sdma_ring_size = d.sdma.ring.sm.mapped_size;
    h.sdma_doorbell = d.sdma.doorbell;
    h.sdma_put = d.sdma.put_value;
    place(h.signal, d.signal_page, d.timeline_slot);
    h.signal_va = d.signal_page.va_addr + d.timeline_slot;
    place(h.shadow, d.signal_page, d.shadow_slot);
    h.shadow_va = d.signal_page.va_addr + d.shadow_slot;
    place(h.kargs, d.kernargs_buf, 0);
    h.kargs_va = d.kernargs_buf.va_addr;
    h.kargs_size = d.kernargs_buf.size;
    place(h.staging, d.staging, 0);
    h.staging_va = d.staging.va_addr;
    h.staging_size = d.staging.size;
    h.pool_va = d.pool.va_addr;
    h.pool_size = d.pool.size;
    h.timeline_value = d.timeline_value;
    h.vram_size = adev.vram_size;
    h.target_major = d.target_major;
    h.xccs = d.xccs;
    h.cu_cnt = d.cu_cnt;
    h.se_cnt = d.se_cnt;
    h.max_slots_scratch_cu = d.max_slots_scratch_cu;
    h.lds_size_in_kb = d.lds_size_in_kb;
    h.ih_ring_paddr = adev.ih->rings[0].ring_vm;
    h.ih_ring_size = adev.ih->ring_size;
    h.is_vf = adev.is_vf ? 1 : 0;
    std::string err;
    uint64_t a;
    if (!adev.t.bar_info(0, a, h.bar0_size, err) || !adev.t.bar_info(2, a, h.bar2_size, err) || !adev.t.bar_info(5, a, h.bar5_size, err))
        throw TGPyError("RuntimeError", err);
    h.reg_hdp_remap = adev.reg("regBIF_BX0_REMAP_HDP_MEM_FLUSH_CNTL").addr(0);
    h.reg_ih_wptr = adev.reg("regIH_RB_WPTR").addr(0);
    h.reg_ih_rptr = adev.reg("regIH_RB_RPTR").addr(0);
    h.reg_ih_cntl = adev.reg("regIH_RB_CNTL").addr(0);
    h.reg_fault_status = adev.reg(adev.gmc->pf_status_reg("GC")).addr(0);
    h.reg_fault_addr_lo = adev.reg("regGCVM_L2_PROTECTION_FAULT_ADDR_LO32").addr(0);
    h.reg_fault_addr_hi = adev.reg("regGCVM_L2_PROTECTION_FAULT_ADDR_HI32").addr(0);
    h.reg_fault_cntl = adev.reg("regGCVM_L2_PROTECTION_FAULT_CNTL").addr(0);
    h.nmaps = order.size();
    h.blob_size = 0;   // the build's HSACO: no compile_all
}

// The daemon's exit: tinygrad's atexit hook runs HCQCompiled.finalize (hcq.py:492-495): its synchronize (the timeline the
// daemon left needs no wait: the C++ runtime is past it) with AMDDevice's IH drain, then Compiled.finalize's device_fini:
// AMDev.fini.
inline void am_device_fini(AMDev& adev) {
    adev.ih->drain();
    adev.fini();
}

// am_device_fini with the crash guard's hold rule (TODO.md plan step A2k): true if it saw every queue off (AMDev::queues_off),
// which makes closing the TinyGPU.app connection DART-safe even if a later step failed. What failed, if anything, in why.
inline bool am_device_fini_safe(AMDev& adev, std::string& why) {
    why.clear();
    try { am_device_fini(adev); }
    catch (const TGPyError& e) { why = e.py(); }
    catch (const am::AMRegError& e) { why = std::string("AMRegError: ") + e.what(); }
    if (why.empty() && !adev.queues_off)
        why = std::to_string(adev.gfx->dequeue_unconfirmed) + " compute queue(s) not seen inactive after their dequeue";
    return adev.queues_off;
}

} // namespace amboot
} // namespace tinygpu_device

#endif // LIBHMSBEAGLE_GPU_TINYGPUAMDDEVICE_H
