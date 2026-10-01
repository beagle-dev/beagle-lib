/*
 * TinyGPUHybridAMDDispatch.h
 *
 * The AMD C++ runtime's PM4 compute encoder (TODO.md plan step A1b): a statement-by-statement port of tinygrad's
 * AMDComputeQueue (tinygrad/runtime/ops_amd.py:53-422 at a9830e2b4) as HCQProgram.__call__ drives it
 * (support/hcq.py:366-373), with amd_hcq_patch.py's exec (the hidden kernel arguments), CLikeArgsState's kernel
 * arguments (hcq.py:315-324, written by bind_args_state at exec) and the kernargs BumpAllocator (memory.py:14-21). It
 * covers what AM's PCIIface uses on gfx11 with one XCC: no AQL, SQTT, PMC, pred_exec or USB paths. golden_amd_encode.py
 * compares every dword, every kernargs byte and the submit's ring, wptr, HDP flush and doorbell against hcq1's own queue.
 *
 * Kernels must have none of the dispatch_ptr, queue_ptr, dispatch_id and private_segment_buffer SGPRs: amd_hcq_patch's
 * branches for them were never needed by BEAGLE's HIP kernels (kernel_code_properties 0x408), and the HSACO loader
 * refuses such kernels instead of porting them (A1d).
 */

#ifndef LIBHMSBEAGLE_GPU_TINYGPUHYBRIDAMDDISPATCH_H
#define LIBHMSBEAGLE_GPU_TINYGPUHYBRIDAMDDISPATCH_H

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "libhmsbeagle/GPU/TinyGPUAMDTables.h"

namespace tinygpu_device {

// What exec reads from a BeagleAMDProgram (AMDProgram.__init__; amd_dispatch_daemon.py BeagleAMDProgram)
struct AMDKernel {
    uint64_t prog_addr = 0;              // lib_gpu.va_addr + the kernel's .kd offset + kernel_code_entry_byte_offset
    uint32_t rsrc1 = 0, rsrc2 = 0, rsrc3 = 0;
    uint32_t kernargs_segment_size = 0;  // desc.kernarg_size
    uint32_t kernargs_alloc_size = 0;    // kernargs_segment_size (no dispatch packet: no dispatch_ptr)
    bool wave32 = false;
};

// What exec reads from the device (AMDDevice.scratch, .tmpring_size)
struct AMDExecDevice {
    uint64_t scratch_va = 0, scratch_size = 0;
    uint32_t tmpring_size = 0;
};

// BumpAllocator(size, wrap=True) (memory.py:14-21): no completion check when it wraps, as in hcq1
struct AMDBump {
    uint64_t size = 0, ptr = 0, base = 0;
    uint64_t alloc(uint64_t sz, uint64_t alignment = 1) {
        if (round_up(ptr, alignment) + sz > size) ptr = 0;
        const uint64_t res = round_up(ptr, alignment);
        ptr = res + sz;
        return res + base;
    }
    static uint64_t round_up(uint64_t n, uint64_t a) { return (n + a - 1) / a * a; }
};

// AMDComputeQueue's dwords (HWQueue._q)
struct AMDComputeQueue {
    std::vector<uint32_t> q;

    void pkt3(uint32_t cmd, std::initializer_list<uint32_t> vals) { pkt3(cmd, vals.begin(), vals.size()); }
    void pkt3(uint32_t cmd, const uint32_t* vals, size_t n) {
        q.push_back(amdt::PACKET3(cmd, (uint32_t)n - 1));
        q.insert(q.end(), vals, vals + n);
    }

    // AMDComputeQueue.wreg: every register BEAGLE's exec writes is in the SH range (the generated table's addresses)
    void wreg(uint32_t reg, std::initializer_list<uint32_t> vals) { wreg(reg, vals.begin(), vals.size()); }
    void wreg(uint32_t reg, const uint32_t* vals, size_t n) {
        std::vector<uint32_t> v(1, 0);
        if (amdt::PACKET3_SET_SH_REG_START <= reg && reg < amdt::PACKET3_SET_SH_REG_END) {
            v[0] = reg - amdt::PACKET3_SET_SH_REG_START;
            v.insert(v.end(), vals, vals + n);
            pkt3(amdt::PACKET3_SET_SH_REG, v.data(), v.size());
        } else if (amdt::PACKET3_SET_UCONFIG_REG_START <= reg && reg < amdt::PACKET3_SET_UCONFIG_REG_START + (1u << 16) - 1) {
            v[0] = reg - amdt::PACKET3_SET_UCONFIG_REG_START;
            v.insert(v.end(), vals, vals + n);
            pkt3(amdt::PACKET3_SET_UCONFIG_REG, v.data(), v.size());
        } else abort();   // tinygrad raises "Cannot set ... via pm4 packet"; no table register gets here
    }

    // wait_reg_mem(value, mask, mem=None, reg=None, reg_done=0, op=GEQ) (ops_amd.py:87-92); mem is a GPU address or 0
    // (None: a register wait on reg/reg_done)
    void wait_reg_mem(uint32_t value, uint32_t mask, bool has_mem, uint64_t mem, uint32_t reg, uint32_t reg_done,
                      uint32_t op = amdt::WAIT_REG_MEM_FUNCTION_GEQ) {
        const uint32_t info = (uint32_t)(amdt::put(amdt::WAIT_REG_MEM_MEM_SPACE, has_mem ? 1 : 0) |
                                         amdt::put(amdt::WAIT_REG_MEM_OPERATION, (!has_mem && reg_done > 0) ? 1 : 0) |
                                         amdt::put(amdt::WAIT_REG_MEM_FUNCTION, op) | amdt::put(amdt::WAIT_REG_MEM_ENGINE, 0));
        if (has_mem) pkt3(amdt::PACKET3_WAIT_REG_MEM, {info, lo32(mem), hi32(mem), value, mask, 4});
        else pkt3(amdt::PACKET3_WAIT_REG_MEM, {info, reg, reg_done, value, mask, 4});
    }

    // acquire_mem(addr=0, sz=(1<<64)-1, gli..gl2=1), the gfx11 branch (ops_amd.py:94-103)
    void acquire_mem(uint64_t addr = 0, uint64_t sz = ~0ull, uint32_t gli = 1, uint32_t glm = 1, uint32_t glk = 1, uint32_t glv = 1,
                     uint32_t gl1 = 1, uint32_t gl2 = 1) {
        using namespace amdt;
        const uint32_t flags = (uint32_t)(put(PACKET3_ACQUIRE_MEM_GCR_CNTL_GLI_INV, gli) | put(PACKET3_ACQUIRE_MEM_GCR_CNTL_GLM_INV, glm) |
                                          put(PACKET3_ACQUIRE_MEM_GCR_CNTL_GLM_WB, glm) | put(PACKET3_ACQUIRE_MEM_GCR_CNTL_GLK_INV, glk) |
                                          put(PACKET3_ACQUIRE_MEM_GCR_CNTL_GLK_WB, glk) | put(PACKET3_ACQUIRE_MEM_GCR_CNTL_GLV_INV, glv) |
                                          put(PACKET3_ACQUIRE_MEM_GCR_CNTL_GL1_INV, gl1) | put(PACKET3_ACQUIRE_MEM_GCR_CNTL_GL2_INV, gl2) |
                                          put(PACKET3_ACQUIRE_MEM_GCR_CNTL_GL2_WB, gl2));
        pkt3(PACKET3_ACQUIRE_MEM, {0, lo32(sz), hi32(sz), lo32(addr), hi32(addr), 0, flags});
    }

    // release_mem(address, value, data_sel, int_sel, ctxid, cache_flush), the gfx11 branch (ops_amd.py:112-133)
    void release_mem(uint64_t address, uint64_t value, uint32_t data_sel, uint32_t int_sel, uint32_t ctxid, bool cache_flush) {
        using namespace amdt;
        const uint32_t flags = !cache_flush ? 0 : (PACKET3_RELEASE_MEM_GCR_GLV_INV | PACKET3_RELEASE_MEM_GCR_GL1_INV |
                                                   PACKET3_RELEASE_MEM_GCR_GL2_INV | PACKET3_RELEASE_MEM_GCR_GLM_WB |
                                                   PACKET3_RELEASE_MEM_GCR_GLM_INV | PACKET3_RELEASE_MEM_GCR_GL2_WB |
                                                   PACKET3_RELEASE_MEM_GCR_SEQ);
        const uint32_t event = (uint32_t)(put(PACKET3_RELEASE_MEM_EVENT_TYPE, CACHE_FLUSH_AND_INV_TS_EVENT) |
                                          put(PACKET3_RELEASE_MEM_EVENT_INDEX, event_index__mec_release_mem__end_of_pipe));
        const uint32_t memsel = (uint32_t)(put(PACKET3_RELEASE_MEM_DATA_SEL, data_sel) | put(PACKET3_RELEASE_MEM_INT_SEL, int_sel) |
                                           put(PACKET3_RELEASE_MEM_DST_SEL, 0));
        pkt3(PACKET3_RELEASE_MEM, {event | flags, memsel, lo32(address), hi32(address), lo32(value), hi32(value), ctxid});
    }

    // memory_barrier (ops_amd.py:135-139): nbio 4.3.0's PF0 HDP flush request/done, then a full acquire_mem
    void memory_barrier() {
        wait_reg_mem(0xffffffffu, 0xffffffffu, false, 0, amdt::regBIF_BX_PF0_GPU_HDP_FLUSH_REQ, amdt::regBIF_BX_PF0_GPU_HDP_FLUSH_DONE);
        acquire_mem();
    }

    // wait (ops_amd.py:372) and signal (:387-396; AM: no event mailbox)
    void wait(uint64_t signal_va, uint32_t value) { wait_reg_mem(value, 0xffffffffu, true, signal_va, 0, 0); }
    void signal(uint64_t signal_va, uint64_t value) {
        release_mem(signal_va, value, amdt::data_sel__mec_release_mem__send_32_bit_low, amdt::int_sel__mec_release_mem__none, 0, true);
    }

    // amd_hcq_patch._patched_exec for a kernel without the dispatch_ptr/queue_ptr/dispatch_id/private segment SGPRs, with
    // CLikeArgsState's arguments (bind_args_state). args points at the kernargs slot in host memory, args_va is its GPU
    // address; ptrs go at 0 as 8-byte values, then each uint32 (BeagleAMDProgram's signature: uint32 values only).
    void exec(const AMDKernel& k, const AMDExecDevice& d, uint8_t* args, uint64_t args_va, const uint64_t* ptrs, int nptr,
              const uint32_t* ints, int nint, const uint32_t global[3], const uint32_t local[3]) {
        for (int i = 0; i < nptr; ++i) memcpy(args + 8 * i, &ptrs[i], 8);           // bind_sints_to_buf(..., fmt='Q', offset=0)
        for (int i = 0; i < nint; ++i) memcpy(args + 8 * nptr + 4 * i, &ints[i], 4);   // TinyELF.iter_sig: uint32, 4-aligned

        acquire_mem(0, ~0ull, 0, 1, 1, 1, 1, 0);

        const uint32_t explicit_bytes = (uint32_t)nptr * 8 + (uint32_t)nint * 4;
        const uint32_t hidden_offset = (explicit_bytes + 7) / 8 * 8;
        if (k.kernargs_segment_size > hidden_offset) {   // block_count (u32 x3), then group_size (u16 x3) at +12
            for (int i = 0; i < 3; ++i) memcpy(args + hidden_offset + 4 * i, &global[i], 4);
            for (int i = 0; i < 3; ++i) { const uint16_t l = (uint16_t)local[i]; memcpy(args + hidden_offset + 12 + 2 * i, &l, 2); }
        }

        const uint64_t pgm = k.prog_addr >> 8, scratch = d.scratch_va >> 8;   // one XCC: scratch_base is the buffer itself
        wreg(amdt::regCOMPUTE_PGM_LO, {lo32(pgm), hi32(pgm)});
        wreg(amdt::regCOMPUTE_PGM_RSRC1, {k.rsrc1, k.rsrc2});
        wreg(amdt::regCOMPUTE_PGM_RSRC3, {k.rsrc3});
        wreg(amdt::regCOMPUTE_TMPRING_SIZE, {d.tmpring_size});
        wreg(amdt::regCOMPUTE_DISPATCH_SCRATCH_BASE_LO, {lo32(scratch), hi32(scratch)});
        wreg(amdt::regCOMPUTE_RESTART_X, {0, 0, 0});
        wreg(amdt::regCOMPUTE_USER_DATA_0, {lo32(args_va), hi32(args_va)});
        wreg(amdt::regCOMPUTE_RESOURCE_LIMITS, {(uint32_t)amdt::encode(amdt::COMPUTE_RESOURCE_LIMITS__waves_per_sh, waves_per_sh())});
        wreg(amdt::regCOMPUTE_START_X, {0, 0, 0, local[0], local[1], local[2], 0, 0});
        const uint32_t initiator = (uint32_t)(amdt::encode(amdt::COMPUTE_DISPATCH_INITIATOR__cs_w32_en, k.wave32 ? 1 : 0) |
                                              amdt::encode(amdt::COMPUTE_DISPATCH_INITIATOR__force_start_at_000, 1) |
                                              amdt::encode(amdt::COMPUTE_DISPATCH_INITIATOR__compute_shader_en, 1));
        pkt3(amdt::PACKET3_DISPATCH_DIRECT, {global[0], global[1], global[2], initiator});
        pkt3(amdt::PACKET3_EVENT_WRITE, {(uint32_t)(amdt::put(amdt::EVENT_TYPE, amdt::CS_PARTIAL_FLUSH) |
                                                    amdt::put(amdt::EVENT_INDEX, amdt::EVENT_INDEX_PARTIAL_FLUSH))});
    }

    // tinygrad's getenv("WAVES_PER_SH"): 0 unless set
    static uint32_t waves_per_sh() {
        static const uint32_t v = [] { const char* e = getenv("WAVES_PER_SH"); return e && e[0] ? (uint32_t)strtoul(e, nullptr, 10) : 0u; }();
        return v;
    }
    static uint32_t lo32(uint64_t v) { return (uint32_t)v; }
    static uint32_t hi32(uint64_t v) { return (uint32_t)(v >> 32); }
};

// AMDComputeQueue._submit's ring write (ops_amd.py:409-422, no IB, one XCC): the dwords go at put_value, wrapping at the ring's
// end; returns the new put_value (in dwords). signal_doorbell (ops_amd.py:730-743) is the caller's: write_ptr = put_value,
// a memory barrier, flush_hdp, then the doorbell = put_value. hcq1 never checks read_ptr here (A1g waits for room instead).
inline uint64_t amd_compute_ring_write(uint32_t* ring, uint64_t ring_dwords, uint64_t put_value, const std::vector<uint32_t>& cmds) {
    for (size_t i = 0; i < cmds.size(); ++i) ring[(put_value + i) % ring_dwords] = cmds[i];
    return put_value + cmds.size();
}

// AMDCopyQueue (ops_amd.py:467-560) on gfx11 (sdma 6.0.0), AM: its dwords and the size of each q() call (internal_cmd_sizes),
// which _submit keeps whole at the ring's end
struct AMDCopyQueue {
    std::vector<uint32_t> q, sizes;
    uint64_t max_copy_size = 0x40000000;   // AMDDevice.max_copy_size on SDMA >= 5

    void emit(std::initializer_list<uint32_t> v) { q.insert(q.end(), v); sizes.push_back((uint32_t)v.size()); }   // HWQueue.q

    void copy(uint64_t dest, uint64_t src, uint64_t copy_size) {
        using namespace amdt;
        const uint64_t commands = (copy_size + max_copy_size - 1) / max_copy_size;
        uint64_t copied = 0;
        for (uint64_t c = 0; c < commands; ++c) {
            const uint64_t step = copy_size - copied < max_copy_size ? copy_size - copied : max_copy_size;
            emit({(uint32_t)(SDMA_OP_COPY | amdt::put(SDMA_PKT_COPY_LINEAR_HEADER_SUB_OP, SDMA_SUBOP_COPY_LINEAR)),
                 (uint32_t)amdt::put(SDMA_PKT_COPY_LINEAR_COUNT_COUNT, step - 1), 0, lo32(src + copied), hi32(src + copied),
                 lo32(dest + copied), hi32(dest + copied)});
            copied += step;
        }
    }
    void signal(uint64_t signal_va, uint64_t value) {   // a FENCE with MTYPE 3 on gfx11; AM: no event mailbox or trap
        emit({(uint32_t)(amdt::SDMA_OP_FENCE | amdt::put(amdt::SDMA_PKT_FENCE_HEADER_MTYPE, 3)), lo32(signal_va), hi32(signal_va), (uint32_t)value});
    }
    void wait(uint64_t signal_va, uint32_t value) {
        using namespace amdt;
        emit({(uint32_t)(SDMA_OP_POLL_REGMEM | amdt::put(SDMA_PKT_POLL_REGMEM_HEADER_FUNC, WAIT_REG_MEM_FUNCTION_GEQ) |
                        amdt::put(SDMA_PKT_POLL_REGMEM_HEADER_MEM_POLL, 1)),
             lo32(signal_va), hi32(signal_va), value, 0xffffffffu,
             (uint32_t)(amdt::put(SDMA_PKT_POLL_REGMEM_DW5_INTERVAL, 0x04) | amdt::put(SDMA_PKT_POLL_REGMEM_DW5_RETRY_COUNT, 0xfff))});
    }
    static uint32_t lo32(uint64_t v) { return (uint32_t)v; }
    static uint32_t hi32(uint64_t v) { return (uint32_t)(v >> 32); }
};

// AMDCopyQueue._submit's ring write (ops_amd.py:524-560, not bound, not USB). put_value is in bytes. The commands that fit
// before the ring's end go there whole; if any are left, the rest of the ring is zero-filled and they go at 0. Before
// writing, room(threshold) must wait until read_ptr >= threshold (hcq1 spins on read_ptr there) and false when it gives up.
// Returns false when the batch is larger than the ring or there was no room; signal_doorbell is the caller's, as above.
template <class Room>
inline bool amd_sdma_ring_write(uint32_t* ring, uint64_t ring_bytes, uint64_t& put_value, const AMDCopyQueue& cq, Room room) {
    const uint64_t to_end = ring_bytes - put_value % ring_bytes;
    uint64_t tail = 0;
    for (uint32_t sz : cq.sizes) {
        if ((tail + sz) * 4 >= to_end) break;
        tail += sz;
    }
    const uint64_t rem = cq.q.size() - tail;
    const uint64_t total = (rem == 0 ? tail * 4 : (ring_bytes - put_value % ring_bytes) % ring_bytes) + rem * 4;
    if (total >= ring_bytes) return false;   // tinygrad asserts "SDMA queue overrun"
    if (put_value + total > ring_bytes && !room(put_value + total - ring_bytes)) return false;
    uint64_t at = (put_value % ring_bytes) / 4;
    for (uint64_t i = 0; i < tail; ++i) ring[at + i] = cq.q[i];
    put_value += tail * 4;
    if (rem > 0) {
        const uint64_t zero = ring_bytes - put_value % ring_bytes;
        memset((uint8_t*)ring + put_value % ring_bytes, 0, zero);
        put_value += zero;
        for (uint64_t i = 0; i < rem; ++i) ring[i] = cq.q[tail + i];
        put_value += rem * 4;
    }
    return true;
}

// HCQAllocator's staging buffers (hcq.py:527-530: batch_cnt host buffers of batch_size bytes), as slots of one host buffer:
// slot i is host + i * slot_size at GPU address va + i * slot_size, with the timeline value of its last copy (b_timeline)
struct AMDStaging {
    uint8_t* host = nullptr;
    uint64_t va = 0, slot_size = 2ull << 20;
    std::vector<uint64_t> timeline;
    size_t next = 0;   // b_next
};

// HCQAllocator._copyin and _copyout (hcq.py:554-604) on the timeline. Ctx gives signal_va and timeline_value (the next value
// to signal), next_timeline(), host_wait(value) (HCQSignal.wait; false on a timeout or a fault), synchronize() (AMDDevice's,
// with the IH drain) and submit(const AMDCopyQueue&) (the ring write and signal_doorbell). False when one of them failed.
template <class Ctx>
inline bool amd_copyin(Ctx& c, AMDStaging& s, uint64_t dest, const uint8_t* src, uint64_t n) {
    for (uint64_t i = 0; i < n; i += s.slot_size) {
        s.next = (s.next + 1) % s.timeline.size();
        if (!c.host_wait(s.timeline[s.next])) return false;
        const uint64_t lsize = s.slot_size < n - i ? s.slot_size : n - i;
        memcpy(s.host + s.next * s.slot_size, src + i, lsize);
        AMDCopyQueue q;
        q.wait(c.signal_va, (uint32_t)(c.timeline_value - 1));
        q.copy(dest + i, s.va + s.next * s.slot_size, lsize);
        q.signal(c.signal_va, c.next_timeline());
        if (!c.submit(q)) return false;
        s.timeline[s.next] = c.timeline_value - 1;
    }
    return true;
}
template <class Ctx>
inline bool amd_copyout(Ctx& c, AMDStaging& s, uint8_t* dst, uint64_t src, uint64_t n) {
    if (!c.synchronize()) return false;
    for (uint64_t i = 0; i < n; i += s.slot_size) {
        const uint64_t lsize = s.slot_size < n - i ? s.slot_size : n - i;
        AMDCopyQueue q;
        q.wait(c.signal_va, (uint32_t)(c.timeline_value - 1));
        q.copy(s.va, src + i, lsize);   // always b[0]
        q.signal(c.signal_va, c.next_timeline());
        if (!c.submit(q) || !c.host_wait(c.timeline_value - 1)) return false;
        memcpy(dst + i, s.host, lsize);
    }
    return true;
}

}  // namespace tinygpu_device

#endif  // LIBHMSBEAGLE_GPU_TINYGPUHYBRIDAMDDISPATCH_H
