/*
 * TinyGPUHybridAMDRuntime.h
 *
 * The AMD C++ runtime on the boot's handoff (TODO.md plan steps A1e-A1g): the GPU's compute and SDMA queues, its timeline, a
 * kernargs buffer, staging and a VRAM pool, as the C++ boot's am_handoff (TinyGPUHybridAMDDevice.h) fills them in. It
 * encodes and submits PM4 and SDMA (TinyGPUHybridAMDDispatch.h) over the plugin's TinyGPU.app connection. The handoff was
 * first the AMD daemon's cmd_handoff reply (flat JSON, then the sysmem fds): amd_parse_handoff and amd_runtime_attach still
 * read that reply, for the goldens (golden_amd_handoff.py, against the oracle's daemon); the plugin uses neither since plan
 * step A2l. It follows tinygrad's AMDDevice (tinygrad/runtime/ops_amd.py, support/hcq.py, support/am/ip.py at a9830e2b4):
 *   - submit: the queue's ring write, then AMDQueueDesc.signal_doorbell: write_ptr, System.memory_barrier,
 *     gmc.flush_hdp (a read of the HDP remap register, a write of 0 where it points) and the doorbell;
 *   - waits: HCQSignal.wait (fails after 30 s without progress, HCQDEV_WAIT_TIMEOUT_MS) with AMDSignal._sleep's interrupt
 *     check after 200 ms: AM_IH.interrupt_handler's decode, where an SQ error or a UTCL2 fault puts the device in error;
 *   - synchronize: the timeline wait, its wrap at 2^31 (_wrap_timeline_signal) and AMDDevice.synchronize's IH drain;
 *   - copies: HCQAllocator._copyin/_copyout through staging (amd_copyin/amd_copyout);
 *   - allocations: PCIIfaceBase.alloc's rounding, carved from the pool; nothing is freed (as on NV).
 * Deliberate differences, beyond TinyGPUHybridAMDProgram.h's: hcq1 never waits before reusing compute ring or kernargs
 * space (ops_amd.py:419-422, memory.py:14-21); here a write that crosses the compute ring's end and a kernargs wrap first
 * wait for everything submitted (NV's wait for idle on wrap). SDMA's own wait for room times out (hcq1 spins forever).
 * The interrupt check runs every 200 ms of a long wait, not on every poll.
 */

#ifndef LIBHMSBEAGLE_GPU_TINYGPUHYBRIDAMDRUNTIME_H
#define LIBHMSBEAGLE_GPU_TINYGPUHYBRIDAMDRUNTIME_H

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

#include "libhmsbeagle/GPU/TinyGPUAMDTables.h"
#include "libhmsbeagle/GPU/TinyGPUHybridAMDDispatch.h"
#include "libhmsbeagle/GPU/TinyGPUHybridAMDProgram.h"
#include "libhmsbeagle/GPU/TinyGPUTransport.h"

namespace tinygpu_device {

// ── the handoff ─────────────────────────────────────────────────────────────

struct AMDHandoffObj { uint64_t map = 0, off = 0; };   // an object in one of the sysmem mappings

struct AMDHandoff {
    uint64_t nmaps = 0, map_size[8] = {}, blob_size = 0;
    AMDHandoffObj compute_ring, compute_rptr, compute_wptr, sdma_ring, sdma_rptr, sdma_wptr, signal, shadow, kargs, staging;
    uint64_t compute_ring_size = 0, compute_doorbell = 0, compute_put = 0, sdma_ring_size = 0, sdma_doorbell = 0, sdma_put = 0;
    uint64_t signal_va = 0, shadow_va = 0, kargs_va = 0, kargs_size = 0, staging_va = 0, staging_size = 0, pool_va = 0, pool_size = 0;
    uint64_t timeline_value = 0, vram_size = 0, target_major = 0, xccs = 0, cu_cnt = 0, se_cnt = 0, max_slots_scratch_cu = 0,
             lds_size_in_kb = 0, ih_ring_paddr = 0, ih_ring_size = 0, is_vf = 0, bar0_size = 0, bar2_size = 0, bar5_size = 0;
    uint64_t reg_hdp_remap = 0, reg_ih_wptr = 0, reg_ih_rptr = 0, reg_ih_cntl = 0, reg_fault_status = 0, reg_fault_addr_lo = 0,
             reg_fault_addr_hi = 0, reg_fault_cntl = 0;
};

// A key of the daemon's flat JSON reply ("key":<unsigned integer>)
inline bool amd_js_u64(const std::string& js, const std::string& key, uint64_t& v) {
    const std::string needle = "\"" + key + "\":";
    size_t p = js.find(needle);
    if (p == std::string::npos) return false;
    p += needle.size();
    while (p < js.size() && js[p] == ' ') ++p;
    if (p >= js.size() || js[p] < '0' || js[p] > '9') return false;
    v = strtoull(js.c_str() + p, nullptr, 10);
    return true;
}

// Every key cmd_handoff sends, checked against the limits this side relies on. "" or what is wrong.
inline std::string amd_parse_handoff(const std::string& js, AMDHandoff& h) {
    std::vector<std::pair<std::string, uint64_t*>> keys = {
        {"nmaps", &h.nmaps}, {"blob_size", &h.blob_size}, {"compute_ring_size", &h.compute_ring_size}, {"compute_doorbell", &h.compute_doorbell},
        {"compute_put", &h.compute_put}, {"sdma_ring_size", &h.sdma_ring_size}, {"sdma_doorbell", &h.sdma_doorbell}, {"sdma_put", &h.sdma_put},
        {"signal_va", &h.signal_va}, {"shadow_va", &h.shadow_va}, {"kargs_va", &h.kargs_va}, {"kargs_size", &h.kargs_size},
        {"staging_va", &h.staging_va}, {"staging_size", &h.staging_size}, {"pool_va", &h.pool_va}, {"pool_size", &h.pool_size},
        {"timeline_value", &h.timeline_value}, {"vram_size", &h.vram_size}, {"target_major", &h.target_major}, {"xccs", &h.xccs},
        {"cu_cnt", &h.cu_cnt}, {"se_cnt", &h.se_cnt}, {"max_slots_scratch_cu", &h.max_slots_scratch_cu}, {"lds_size_in_kb", &h.lds_size_in_kb},
        {"ih_ring_paddr", &h.ih_ring_paddr}, {"ih_ring_size", &h.ih_ring_size}, {"is_vf", &h.is_vf}, {"bar0_size", &h.bar0_size},
        {"bar2_size", &h.bar2_size}, {"bar5_size", &h.bar5_size}, {"reg_hdp_remap", &h.reg_hdp_remap}, {"reg_ih_wptr", &h.reg_ih_wptr},
        {"reg_ih_rptr", &h.reg_ih_rptr}, {"reg_ih_cntl", &h.reg_ih_cntl}, {"reg_fault_status", &h.reg_fault_status},
        {"reg_fault_addr_lo", &h.reg_fault_addr_lo}, {"reg_fault_addr_hi", &h.reg_fault_addr_hi}, {"reg_fault_cntl", &h.reg_fault_cntl}};
    const std::pair<const char*, AMDHandoffObj*> objs[] = {
        {"compute_ring", &h.compute_ring}, {"compute_rptr", &h.compute_rptr}, {"compute_wptr", &h.compute_wptr}, {"sdma_ring", &h.sdma_ring},
        {"sdma_rptr", &h.sdma_rptr}, {"sdma_wptr", &h.sdma_wptr}, {"signal", &h.signal}, {"shadow", &h.shadow}, {"kargs", &h.kargs},
        {"staging", &h.staging}};
    for (const auto& [name, obj] : objs) {
        keys.push_back({std::string(name) + "_map", &obj->map});
        keys.push_back({std::string(name) + "_off", &obj->off});
    }
    for (auto& [key, v] : keys)
        if (!amd_js_u64(js, key, *v)) return "the handoff has no " + key;
    if (h.nmaps == 0 || h.nmaps > 8) return "the handoff has " + std::to_string(h.nmaps) + " mappings";
    for (uint64_t i = 0; i < h.nmaps; ++i)
        if (!amd_js_u64(js, "map" + std::to_string(i) + "_size", h.map_size[i])) return "the handoff has no map" + std::to_string(i) + "_size";
    // what each object must hold in its mapping
    const std::pair<const char*, std::pair<AMDHandoffObj*, uint64_t>> sized[] = {
        {"compute_ring", {&h.compute_ring, h.compute_ring_size}}, {"compute_rptr", {&h.compute_rptr, 8}}, {"compute_wptr", {&h.compute_wptr, 8}},
        {"sdma_ring", {&h.sdma_ring, h.sdma_ring_size}}, {"sdma_rptr", {&h.sdma_rptr, 8}}, {"sdma_wptr", {&h.sdma_wptr, 8}},
        {"signal", {&h.signal, 16}}, {"shadow", {&h.shadow, 16}}, {"kargs", {&h.kargs, h.kargs_size}}, {"staging", {&h.staging, h.staging_size}}};
    for (const auto& [name, os] : sized) {
        const AMDHandoffObj& o = *os.first;
        if (o.map >= h.nmaps || os.second > h.map_size[o.map] || o.off > h.map_size[o.map] - os.second)
            return std::string("the handoff's ") + name + " is outside its mapping";
    }
    if (h.is_vf) return "a virtual function (is_vf): the C++ runtime drives a physical function only";
    if (h.target_major != 11 || h.xccs != 1) return "gfx" + std::to_string(h.target_major) + " with " + std::to_string(h.xccs) + " XCCs: the C++ runtime is gfx11 with one XCC";
    if (h.compute_ring_size == 0 || h.compute_ring_size % 4 || h.sdma_ring_size == 0 || h.sdma_ring_size % 4) return "the handoff's rings are not whole dwords";
    for (uint64_t db : {h.compute_doorbell, h.sdma_doorbell})
        if (db % 8 || db + 8 > h.bar2_size) return "a doorbell outside BAR2";
    for (uint64_t r : {h.reg_hdp_remap, h.reg_ih_wptr, h.reg_ih_rptr, h.reg_ih_cntl, h.reg_fault_status, h.reg_fault_addr_lo, h.reg_fault_addr_hi, h.reg_fault_cntl})
        if (r * 4 + 4 > h.bar5_size) return "a register outside BAR5 (the indirect window is not ported)";
    if (h.ih_ring_size == 0 || h.ih_ring_size % 32) return "the handoff's IH ring is not whole entries";
    if (h.staging_size < (2ull << 20)) return "the handoff's staging is smaller than one 2 MB slot";
    return "";
}

// ── the runtime ─────────────────────────────────────────────────────────────

struct AMDRing {   // one queue: its ring and gart words in host memory, and its doorbell (an offset in BAR2)
    uint32_t* ring = nullptr;
    uint64_t ring_bytes = 0, doorbell = 0, put = 0;   // put: dwords (compute) or bytes (SDMA), as hcq1's put_value
    volatile uint64_t* rptr = nullptr;
    volatile uint64_t* wptr = nullptr;
};

struct AMDRuntime {
    AMDHandoff h;
    TGTransport* tg = nullptr;
    void* maps[8] = {};
    uint64_t map_sizes[8] = {};
    bool owns_maps = true;   // mapped here from the handoff's fds (unmapped at detach), or the C++ boot's own (plan step A2g)
    AMDRing compute, sdma;
    volatile uint64_t* signal = nullptr;   // the timeline signal's value
    volatile uint64_t* shadow = nullptr;   // _shadow_timeline_signal's
    uint64_t signal_va = 0, shadow_va = 0, timeline_value = 0, submitted = 0;   // submitted: the last value a submit signals
    uint8_t* kargs = nullptr;
    AMDBump kargs_bump;
    uint64_t kargs_va = 0;
    AMDStaging staging;
    AMDExecDevice exec;
    AMDProps props;
    std::map<std::string, AMDProgramRecord> kernels;
    uint64_t pool_used = 0;
    bool error = false;
    std::string error_msg;
    uint64_t wait_timeout_ms = 30000;

    bool fail(const std::string& why) {
        if (!error) { error = true; error_msg = why; fprintf(stderr, "TinyGPU/AMD: %s\n", why.c_str()); }
        return false;
    }

    // ── MMIO on the shared connection ──
    bool rreg(uint64_t reg, uint32_t& v) {
        std::string err;
        return tg->bulk_read(5, reg * 4, &v, 4, err) || fail("MMIO read of register " + std::to_string(reg) + ": " + err);
    }
    bool wreg(uint64_t reg, uint32_t v) {
        std::string err;
        return tg->bulk_write(5, reg * 4, &v, 4, err) || fail("MMIO write of register " + std::to_string(reg) + ": " + err);
    }
    bool update(uint64_t reg, amdt::Bits b, uint32_t v) {   // AMRegister.update: read, clear the field, write
        uint32_t cur;
        return rreg(reg, cur) && wreg(reg, (uint32_t)((cur & ~amdt::mask(b)) | amdt::encode(b, v)));
    }

    // AM_GMC.flush_hdp (ip.py:91-93, a physical function): the remap register says where the flush register is
    bool flush_hdp() {
        uint32_t remap;
        if (!rreg(h.reg_hdp_remap, remap)) return false;
        if ((uint64_t)(remap / 4) * 4 + 4 > h.bar5_size) return fail("the HDP flush register is outside BAR5 (the indirect window is not ported)");
        return wreg(remap / 4, 0);
    }

    // AMDQueueDesc.signal_doorbell (ops_amd.py:730-743)
    bool signal_doorbell(AMDRing& q) {
        *q.wptr = q.put;
        __sync_synchronize();   // System.memory_barrier
        if (!flush_hdp()) return false;
        std::string err;
        const uint64_t v = q.put;
        return tg->bulk_write(2, q.doorbell, &v, 8, err) || fail("the doorbell write: " + err);
    }

    // ── waits ──
    // HCQSignal.wait with AMDSignal._sleep's interrupt check
    bool host_wait(uint64_t value) {
        if (error) return false;
        using clk = std::chrono::steady_clock;
        auto start = clk::now(), last_check = start;
        uint64_t prev = *signal;
        while (*signal < value) {
            const auto now = clk::now();
            if (*signal != prev) { prev = *signal; start = now; }   // progress: the timeout starts again
            const auto waited = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
            if ((uint64_t)waited >= wait_timeout_ms)
                return fail("Wait timeout: " + std::to_string(wait_timeout_ms) + " ms! (the signal is not set to " + std::to_string(value) +
                            ", but " + std::to_string(*signal) + ")");
            if (waited > 200 && std::chrono::duration_cast<std::chrono::milliseconds>(now - last_check).count() >= 200) {
                last_check = now;
                if (!check_interrupts()) return false;
            }
        }
        return true;
    }
    bool wait_idle() { return host_wait(submitted); }

    // AM_IH.interrupt_handler (ip.py:483-525): decodes the entries between RPTR and WPTR (it does not move RPTR; the drain
    // after a synchronize does). An SQ error (enc_type 2) or a UTCL2 fault puts the device in error.
    bool check_interrupts() {
        uint32_t w, rptr;
        if (!rreg(h.reg_ih_wptr, w) || !rreg(h.reg_ih_rptr, rptr)) return false;
        const uint32_t woff = (uint32_t)amdt::getbits(w, amdt::IH_RB_WPTR__offset), ring_dwords = (uint32_t)(h.ih_ring_size / 4);
        bool bad = false;
        for (int n = 0; rptr != woff && n < 4096; ++n) {
            uint32_t e[8];
            for (uint32_t i = 0; i < 8; ++i) {
                std::string err;
                if (!tg->bulk_read(0, h.ih_ring_paddr + ((rptr + i) % ring_dwords) * 4, &e[i], 4, err)) return fail("reading the IH ring: " + err);
            }
            rptr = (rptr + 8) % ring_dwords;
            const uint32_t client = amdt::ih_get(e, amdt::IH_CLIENT_ID), src = amdt::ih_get(e, amdt::IH_SOURCE_ID);
            const char* src_name = "";
            if (client == amdt::SOC21_IH_CLIENTID_GRBM_CP || client == amdt::SOC21_IH_CLIENTID_GFX)
                for (const amdt::IHName& s : amdt::IH_GFX11_SRCS) if (s.id == src) src_name = s.name;
            if (!strcmp(src_name, "SDMA_TRAP") || !strcmp(src_name, "CP_EOP_INTR")) continue;
            const char* client_name = "None";
            for (const amdt::IHName& c : amdt::IH_SOC21_CLIENTS) if (c.id == client) client_name = c.name;
            const uint32_t c0 = amdt::ih_get(e, amdt::IH_CONTEXT_ID0), c1 = amdt::ih_get(e, amdt::IH_CONTEXT_ID1);
            fprintf(stderr, "TinyGPU/AMD: IH (%#x/%#x) client=%s src=%s(%u) ring=%u vmid=%u(%u) pasid=%u node=%u ctx=[%#x, %#x, %#x, %#x]\n",
                    rptr, woff, client_name, src_name, src, amdt::ih_get(e, amdt::IH_RING_ID), amdt::ih_get(e, amdt::IH_VMID),
                    amdt::ih_get(e, amdt::IH_VMID_TYPE), amdt::ih_get(e, amdt::IH_PASID), amdt::ih_get(e, amdt::IH_NODEID), c0, c1,
                    amdt::ih_get(e, amdt::IH_CONTEXT_ID2), amdt::ih_get(e, amdt::IH_CONTEXT_ID3));
            if (!strcmp(src_name, "SQ_INTERRUPT_ID")) {   // soc21's fields
                const uint32_t enc_type = (c1 >> 6) & 0x3, err_type = (c0 >> 21) & 0xf;
                static const char* kEnc[] = {"auto", "wave", "error", "?"};
                static const char* kErr[] = {"EDC_FUE", "ILLEGAL_INST", "MEMVIOL", "EDC_FED"};
                fprintf(stderr, "TinyGPU/AMD: sq_intr: %s%s%s%s\n", kEnc[enc_type], enc_type == 2 ? " (" : "",
                        enc_type == 2 ? (err_type < 4 ? kErr[err_type] : "?") : "", enc_type == 2 ? ")" : "");
                bad |= enc_type == 2;
            } else if (!strcmp(src_name, "UTCL2_FAULT")) {
                uint32_t st, hi, lo;
                if (!rreg(h.reg_fault_status, st) || !rreg(h.reg_fault_addr_hi, hi) || !rreg(h.reg_fault_addr_lo, lo)) return false;
                fprintf(stderr, "TinyGPU/AMD: GCVM_L2_PROTECTION_FAULT_STATUS: %#x %#llx\n", st, (unsigned long long)((((uint64_t)hi << 32) | lo) << 12));
                if (!update(h.reg_fault_cntl, amdt::GCVM_L2_PROTECTION_FAULT_CNTL__clear_protection_fault_status_addr, 1)) return false;
                bad = true;
            }
        }
        return !bad || fail("the GPU reported a fault (above): the device is in error");
    }

    // AM_IH.drain (ip.py:470-480)
    bool drain_ih() {
        uint32_t w;
        if (!rreg(h.reg_ih_wptr, w)) return false;
        if (!wreg(h.reg_ih_rptr, (uint32_t)(amdt::getbits(w, amdt::IH_RB_WPTR__offset) % (h.ih_ring_size / 4)))) return false;
        if (amdt::getbits(w, amdt::IH_RB_WPTR__rb_overflow))
            return update(h.reg_ih_wptr, amdt::IH_RB_WPTR__rb_overflow, 0) && update(h.reg_ih_cntl, amdt::IH_RB_CNTL__wptr_overflow_clear, 1) &&
                   update(h.reg_ih_cntl, amdt::IH_RB_CNTL__wptr_overflow_clear, 0);
        return true;
    }

    // AMDDevice.synchronize: HCQCompiled.synchronize (the wait, then the timeline's wrap past 2^31), then the IH drain
    bool synchronize() {
        if (!host_wait(timeline_value - 1)) return false;
        if (timeline_value > (1ull << 31)) {   // _wrap_timeline_signal: the shadow signal takes over at 1
            std::swap(signal, shadow);
            std::swap(signal_va, shadow_va);
            timeline_value = 1;
            *signal = 0;
            submitted = 0;
            for (uint64_t& t : staging.timeline) t = 0;
        }
        return drain_ih();
    }

    uint64_t next_timeline() { return timeline_value++; }

    // ── submits ──
    // AMDComputeQueue._submit, after waiting for the GPU to finish everything submitted when the write crosses the ring's end
    bool submit_compute(const AMDComputeQueue& q) {
        if (error) return false;
        const uint64_t dwords = compute.ring_bytes / 4;
        if (q.q.size() >= dwords) return fail("a compute submit larger than the ring");
        if (compute.put % dwords + q.q.size() > dwords && !wait_idle()) return false;
        compute.put = amd_compute_ring_write(compute.ring, dwords, compute.put, q.q);
        if (!signal_doorbell(compute)) return false;
        submitted = timeline_value - 1;   // every submit ends with a signal of next_timeline()
        return true;
    }

    // AMDCopyQueue._submit, whose wait for room on read_ptr times out here
    bool submit(const AMDCopyQueue& q) {
        if (error) return false;
        auto room = [&](uint64_t threshold) {
            using clk = std::chrono::steady_clock;
            auto start = clk::now();
            uint64_t prev = *sdma.rptr;
            while (*sdma.rptr < threshold) {
                if (*sdma.rptr != prev) { prev = *sdma.rptr; start = clk::now(); }
                if ((uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(clk::now() - start).count() >= wait_timeout_ms) return false;
            }
            return true;
        };
        if (!amd_sdma_ring_write(sdma.ring, sdma.ring_bytes, sdma.put, q, room)) return fail("the SDMA ring has no room (or the submit is larger than it)");
        if (!signal_doorbell(sdma)) return false;
        submitted = timeline_value - 1;
        return true;
    }

    // ── memory ──
    // PCIIfaceBase.alloc's rounding (2 MB pages from 8 MB, else 4 KB), carved from the pool
    bool alloc(uint64_t size, uint64_t& va) {
        const uint64_t page = size >= (8ull << 20) ? (2ull << 20) : 0x1000;
        const uint64_t sz = (size + page - 1) / page * page, at = (pool_used + page - 1) / page * page;
        if (sz == 0 || at + sz > h.pool_size) return false;
        pool_used = at + sz;
        va = h.pool_va + at;
        return true;
    }
    uint64_t available() const { return h.pool_size > pool_used ? h.pool_size - pool_used : 0; }
};

// Maps the handoff's sysmem fds (closing them) and sets the runtime up: queues, timeline, kernargs, staging, the BARs the
// daemon mapped in this session (a MAP_BAR here would change the stream). "" or why not; on failure the mappings are undone.
inline void amd_runtime_setup(AMDRuntime& rt, const AMDHandoff& h);
inline std::string amd_runtime_attach(AMDRuntime& rt, const AMDHandoff& h, const int* fds, TGTransport& tg) {
    rt.h = h;
    rt.tg = &tg;
    std::string err;
    for (uint64_t i = 0; i < h.nmaps; ++i) {
        void* m = mmap(nullptr, h.map_size[i], PROT_READ | PROT_WRITE, MAP_SHARED, fds[i], 0);
        if (m == MAP_FAILED) { if (err.empty()) err = std::string("mmap of a handed-over buffer: ") + strerror(errno); }
        else { rt.maps[i] = m; rt.map_sizes[i] = h.map_size[i]; }
        close(fds[i]);
    }
    if (!err.empty()) {
        for (uint64_t i = 0; i < h.nmaps; ++i) if (rt.maps[i]) { munmap(rt.maps[i], rt.map_sizes[i]); rt.maps[i] = nullptr; }
        return err;
    }
    amd_runtime_setup(rt, h);
    tg.seed_bar(0, h.bar0_size);
    tg.seed_bar(2, h.bar2_size);
    tg.seed_bar(5, h.bar5_size);
    return "";
}

// The same on mappings this process made itself (the C++ boot's, plan step A2g): nothing is mapped, closed or unmapped
// here, and the transport already knows the BARs.
inline void amd_runtime_attach_mapped(AMDRuntime& rt, const AMDHandoff& h, uint8_t* const* maps, TGTransport& tg) {
    rt.h = h;
    rt.tg = &tg;
    rt.owns_maps = false;
    for (uint64_t i = 0; i < h.nmaps; ++i) { rt.maps[i] = maps[i]; rt.map_sizes[i] = h.map_size[i]; }
    amd_runtime_setup(rt, h);
}

inline void amd_runtime_setup(AMDRuntime& rt, const AMDHandoff& h) {
    auto at = [&](const AMDHandoffObj& o) { return (uint8_t*)rt.maps[o.map] + o.off; };
    rt.compute = AMDRing{(uint32_t*)at(h.compute_ring), h.compute_ring_size, h.compute_doorbell, h.compute_put,
                         (volatile uint64_t*)at(h.compute_rptr), (volatile uint64_t*)at(h.compute_wptr)};
    rt.sdma = AMDRing{(uint32_t*)at(h.sdma_ring), h.sdma_ring_size, h.sdma_doorbell, h.sdma_put,
                      (volatile uint64_t*)at(h.sdma_rptr), (volatile uint64_t*)at(h.sdma_wptr)};
    rt.signal = (volatile uint64_t*)at(h.signal);
    rt.shadow = (volatile uint64_t*)at(h.shadow);
    rt.signal_va = h.signal_va;
    rt.shadow_va = h.shadow_va;
    rt.timeline_value = h.timeline_value;
    rt.submitted = h.timeline_value - 1;   // the boot synchronized before its handoff (am_handoff, as cmd_handoff did)
    rt.kargs = at(h.kargs);
    rt.kargs_va = h.kargs_va;
    rt.kargs_bump = AMDBump{h.kargs_size, 0, 0};
    rt.staging.host = at(h.staging);
    rt.staging.va = h.staging_va;
    rt.staging.slot_size = 2ull << 20;   // HCQAllocator's batch_size
    rt.staging.timeline.assign(h.staging_size / rt.staging.slot_size, 0);
    rt.props = AMDProps{(uint32_t)h.target_major, (uint32_t)h.xccs, (uint32_t)h.cu_cnt, (uint32_t)h.se_cnt, (uint32_t)h.max_slots_scratch_cu,
                        (uint32_t)h.lds_size_in_kb};
    if (const char* t = getenv("HCQDEV_WAIT_TIMEOUT_MS")) rt.wait_timeout_ms = strtoull(t, nullptr, 10);
}

inline void amd_runtime_detach(AMDRuntime& rt) {
    for (int i = 0; i < 8; ++i) if (rt.maps[i]) { if (rt.owns_maps) munmap(rt.maps[i], rt.map_sizes[i]); rt.maps[i] = nullptr; }
}

// The programs: the HSACO's image at one pool allocation (round_up 0x1000, as BeagleAMDProgram's), uploaded through staging
// and synchronized (as AMDProgram.__init__), then scratch sized once for the largest private segment (at least AMDDevice's
// initial 128 bytes) and carved from the pool. "" or why not.
inline std::string amd_runtime_load_programs(AMDRuntime& rt, const uint8_t* hsaco, size_t n) {
    std::vector<uint8_t> image;
    std::map<std::string, AMDProgramRecord> probe;
    std::string err = amd_load_hsaco(hsaco, n, 0, rt.props, image, probe);   // the image's size first
    if (!err.empty()) return err;
    uint64_t lib_va = 0;
    if (!rt.alloc((image.size() + 0xfff) & ~0xfffull, lib_va)) return "the VRAM pool is too small for the program image";
    err = amd_load_hsaco(hsaco, n, lib_va, rt.props, image, rt.kernels);
    if (!err.empty()) return err;
    if (!amd_copyin(rt, rt.staging, lib_va, image.data(), image.size()) || !rt.synchronize()) return "uploading the program image: " + rt.error_msg;
    uint32_t priv = 128;
    for (const auto& kv : rt.kernels) priv = kv.second.private_segment_size > priv ? kv.second.private_segment_size : priv;
    uint64_t scratch_size = 0;
    uint32_t tmpring = 0;
    err = amd_scratch(rt.props, priv, scratch_size, tmpring);
    if (!err.empty()) return err;
    uint64_t scratch_va = 0;
    if (!rt.alloc(scratch_size, scratch_va)) return "the VRAM pool is too small for " + std::to_string(scratch_size >> 20) + " MiB of scratch";
    rt.exec = AMDExecDevice{scratch_va, scratch_size, tmpring};
    return "";
}

}  // namespace tinygpu_device

#endif  // LIBHMSBEAGLE_GPU_TINYGPUHYBRIDAMDRUNTIME_H
