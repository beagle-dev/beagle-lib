/*
 * TinyGPUHybridNVDispatch.h
 *
 * NV command encoding for C++ dispatch after the daemon handoff
 * (BEAGLE_NV_CPP_DISPATCH=1; TODO.md "Runtime roadmap", Step 3). The daemon
 * boots the GPU and prepares every program with tinygrad hcq1, then hands
 * over, per kernel, the QMD template and cbuf0 prefix it would have used,
 * plus QMD field positions and method/flag words computed from tinygrad's own
 * tables (nv_dispatch_daemon.py build_handoff), so this file hardcodes none
 * of them. Everything here encodes into caller-provided memory; the transport
 * (TinyGPU.app socket, shared buffers, polling) lives in
 * GPUInterfaceTinyGPUHybridNV.cpp. Each function names the hcq1 code it
 * mirrors (tinygrad/runtime/ops_nv.py and support/hcq.py at a9830e2b4).
 */

#ifndef LIBHMSBEAGLE_GPU_TINYGPUHYBRIDNVDISPATCH_H
#define LIBHMSBEAGLE_GPU_TINYGPUHYBRIDNVDISPATCH_H

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <map>
#include <string>
#include <vector>

namespace tinygpu_device {

struct NVDFifo {                 // one GPFIFO, as TinyGPU.app BAR offsets
    uint32_t ring_bar = 0, gpput_bar = 0, entries = 0, token = 0;
    uint64_t ring_off = 0, gpput_off = 0, put = 0;  // put counts submissions; the ring slot is put % entries
};

struct NVDBuffer { uint64_t va = 0, size = 0; };  // a C++-owned sysmem buffer: GPU VA and size

struct NVDKernel {
    std::string name;
    uint32_t qmd_off = 0;       // QMD offset in a kernargs slot: round_up(cbuf0 size, 256)
    uint32_t slot_size = 0;     // kernargs_alloc_size
    uint32_t prefix_words = 0;  // cbuf0 driver-param words; the arguments start right after them
    uint32_t dims_b = 0, dims_g = 0;  // cbuf0 word index of blockDim/gridDim, or kNoDims (fill off)
    uint32_t max_threads = 0;
    std::vector<uint8_t> qmd;       // QMD template (NVProgram.qmd)
    std::vector<uint32_t> prefix;   // cbuf0 prefix (NVProgram.cbuf_0), launch-dims words zero
};

struct NVDHandoff {
    static constexpr uint32_t kNoDims = 0xffffffffu;
    uint32_t qmd_ver = 0, qmd_bytes = 0;
    // QMD byte offsets for plain stores, and {hi, lo} bit ranges for bitfield writes
    uint32_t q_grid = 0, q_block01 = 0, q_block2 = 0, q_rel_addr = 0, q_rel_payload = 0, q_cb_shift = 0;
    uint32_t q_cb_hi[2]{}, q_cb_lo[2]{}, q_rel_en[2]{}, q_dep_ptr[2]{}, q_dep_action[2]{}, q_dep_prefetch[2]{}, q_dep_enable[2]{};
    // method and flag words
    uint32_t m_sem_addr_lo = 0, f_sem_acquire = 0, m_invalidate = 0, f_invalidate = 0, m_pcas_a = 0, m_pcas2_b = 0;
    uint32_t m_dma_offset_in_upper = 0, m_dma_line_length_in = 0, m_dma_launch = 0, m_dma_sem_a = 0, f_dma_copy = 0, f_dma_sem = 0;
    NVDFifo compute, copy;
    uint32_t db_bar = 0;
    uint64_t db_off = 0;
    NVDBuffer cmdq, kargs, staging, signal;  // the order the daemon sends their fds in
    std::map<std::string, NVDKernel> kernels;
};

// ── handoff parsing: flat JSON of unsigned integers, then the kernel blob ──

static inline bool nvd_json_u64(const std::string& js, const char* key, uint64_t& out) {
    std::string needle = std::string("\"") + key + "\":";
    size_t p = js.find(needle);
    if (p == std::string::npos) return false;
    p += needle.size();
    while (p < js.size() && js[p] == ' ') ++p;
    char* end = nullptr;
    out = strtoull(js.c_str() + p, &end, 10);
    return end != js.c_str() + p;
}

// Returns an empty string on success, otherwise what was missing or malformed.
static inline std::string nvd_parse_handoff(const std::string& js, const std::vector<uint8_t>& blob, NVDHandoff& h) {
    std::string missing;
    auto u64 = [&](const char* key) -> uint64_t {
        uint64_t v = 0;
        if (!nvd_json_u64(js, key, v)) missing += std::string(missing.empty() ? "" : ", ") + key;
        return v;
    };
    auto u32 = [&](const char* key) { return (uint32_t)u64(key); };
    auto bitrange = [&](const char* key, uint32_t out[2]) {
        out[0] = u32((std::string(key) + "_hi").c_str());
        out[1] = u32((std::string(key) + "_lo").c_str());
    };
    auto fifo = [&](const char* k, NVDFifo& f) {
        std::string p(k);
        f.ring_bar = u32((p + "_ring_bar").c_str());   f.ring_off = u64((p + "_ring_off").c_str());
        f.gpput_bar = u32((p + "_gpput_bar").c_str()); f.gpput_off = u64((p + "_gpput_off").c_str());
        f.entries = u32((p + "_entries").c_str());     f.put = u64((p + "_put").c_str());
        f.token = u32((p + "_token").c_str());
    };
    auto buffer = [&](const char* k, NVDBuffer& b) {
        std::string p(k);
        b.va = u64((p + "_va").c_str());
        b.size = u64((p + "_size").c_str());
    };

    h.qmd_ver = u32("qmd_ver");     h.qmd_bytes = u32("qmd_bytes");
    h.q_grid = u32("q_grid");       h.q_block01 = u32("q_block01");  h.q_block2 = u32("q_block2");
    h.q_rel_addr = u32("q_rel_addr"); h.q_rel_payload = u32("q_rel_payload"); h.q_cb_shift = u32("q_cb_shift");
    bitrange("q_cb_hi", h.q_cb_hi);  bitrange("q_cb_lo", h.q_cb_lo);  bitrange("q_rel_en", h.q_rel_en);
    bitrange("q_dep_ptr", h.q_dep_ptr);  bitrange("q_dep_action", h.q_dep_action);
    bitrange("q_dep_prefetch", h.q_dep_prefetch);  bitrange("q_dep_enable", h.q_dep_enable);
    h.m_sem_addr_lo = u32("m_sem_addr_lo");  h.f_sem_acquire = u32("f_sem_acquire");
    h.m_invalidate = u32("m_invalidate");    h.f_invalidate = u32("f_invalidate");
    h.m_pcas_a = u32("m_pcas_a");            h.m_pcas2_b = u32("m_pcas2_b");
    h.m_dma_offset_in_upper = u32("m_dma_offset_in_upper");  h.m_dma_line_length_in = u32("m_dma_line_length_in");
    h.m_dma_launch = u32("m_dma_launch");    h.m_dma_sem_a = u32("m_dma_sem_a");
    h.f_dma_copy = u32("f_dma_copy");        h.f_dma_sem = u32("f_dma_sem");
    fifo("c", h.compute);  fifo("d", h.copy);
    h.db_bar = u32("db_bar");  h.db_off = u64("db_off");
    buffer("cmdq", h.cmdq);  buffer("kargs", h.kargs);  buffer("staging", h.staging);  buffer("signal", h.signal);
    uint32_t nkernels = u32("nkernels");
    if (!missing.empty()) return "missing " + missing;
    // SEND_PCAS_A and dependent_qmd0_pointer carry QMD address >> 8 in 32 bits, and a GPFIFO entry carries a
    // 40-bit pushbuffer address (hcq1 asserts the same: NVComputeQueue.exec's "large qmd addr")
    if (h.kargs.va + h.kargs.size > (1ull << 40) || h.cmdq.va + h.cmdq.size > (1ull << 40))
        return "kernargs or pushbuffer buffer above 2^40";

    size_t p = 0;
    auto take = [&](void* dst, size_t n) {
        if (p + n > blob.size()) return false;
        memcpy(dst, blob.data() + p, n);
        p += n;
        return true;
    };
    for (uint32_t i = 0; i < nkernels; ++i) {
        uint32_t hdr[7];
        NVDKernel k;
        if (!take(hdr, sizeof(hdr))) return "kernel blob truncated";
        k.name.resize(hdr[0]);
        k.qmd_off = hdr[1];  k.slot_size = hdr[2];  k.prefix_words = hdr[3];
        k.dims_b = hdr[4];   k.dims_g = hdr[5];     k.max_threads = hdr[6];
        k.qmd.resize(h.qmd_bytes);
        k.prefix.resize(k.prefix_words);
        if (!take(&k.name[0], hdr[0]) || !take(k.qmd.data(), h.qmd_bytes) || !take(k.prefix.data(), k.prefix_words * 4))
            return "kernel blob truncated";
        if (k.qmd_off + h.qmd_bytes > k.slot_size || k.prefix_words * 4 > k.qmd_off ||
            (k.dims_b != NVDHandoff::kNoDims && (k.dims_b + 3 > k.prefix_words || k.dims_g + 3 > k.prefix_words)))
            return "inconsistent kernel record for " + k.name;
        h.kernels[k.name] = std::move(k);
    }
    if (p != blob.size()) return "trailing bytes in kernel blob";
    return "";
}

// ── QMD ──────────────────────────────────────────────────────────────────

// QMD._rw_bits: write value into bits lo..hi (little-endian bit numbering).
// QMD fields are at most 32 bits wide, so the covered bytes fit in a uint64_t.
static inline void nvd_qmd_bits(uint8_t* qmd, const uint32_t range[2], uint64_t value) {
    uint32_t hi = range[0], lo = range[1], b0 = lo / 8, b1 = hi / 8;
    uint64_t num = 0;
    for (uint32_t i = b1 + 1; i-- > b0; ) num = (num << 8) | qmd[i];
    uint64_t mask = ((1ull << (hi - lo + 1)) - 1) << (lo % 8);
    num = (num & ~mask) | ((value << (lo % 8)) & mask);
    for (uint32_t i = b0; i <= b1; ++i) { qmd[i] = (uint8_t)num; num >>= 8; }
}

// One kernel launch into its kernargs slot: the cbuf0 prefix with the launch
// dims (NVArgsState, with BeagleNVProgram.set_launch_dims's fill), the
// pointer then uint32 arguments (CLikeArgsState), and the QMD from the
// template with the grid, block and cbuf0 address (NVComputeQueue.exec).
// Returns the slot's QMD.
static inline uint8_t* nvd_encode_launch(const NVDHandoff& h, const NVDKernel& k, uint8_t* slot, uint64_t slot_va,
                                         const uint32_t grid[3], const uint32_t block[3],
                                         const uint64_t* ptrs, int nptr, const uint32_t* ints, int nint) {
    uint32_t* words = (uint32_t*)slot;
    memcpy(slot, k.prefix.data(), k.prefix.size() * 4);
    if (k.dims_b != NVDHandoff::kNoDims) {
        memcpy(words + k.dims_b, block, 12);
        memcpy(words + k.dims_g, grid, 12);
    }
    uint8_t* args = slot + k.prefix_words * 4;
    memcpy(args, ptrs, (size_t)nptr * 8);
    memcpy(args + (size_t)nptr * 8, ints, (size_t)nint * 4);

    uint8_t* qmd = slot + k.qmd_off;
    memcpy(qmd, k.qmd.data(), h.qmd_bytes);
    memcpy(qmd + h.q_grid, grid, 12);                        // bind_sints_to_mem(*global_size, fmt='I')
    uint16_t b01[2] = { (uint16_t)block[0], (uint16_t)block[1] };
    memcpy(qmd + h.q_block01, b01, 4);                       // local_size[:2], fmt='H'
    qmd[h.q_block2] = (uint8_t)block[2];                     // local_size[2], fmt='B'
    uint64_t cb = slot_va >> h.q_cb_shift;                   // QMD.set_constant_buf_addr(0, args_state.buf.va_addr)
    nvd_qmd_bits(qmd, h.q_cb_hi, cb >> 32);
    nvd_qmd_bits(qmd, h.q_cb_lo, cb & 0xffffffffu);
    return qmd;
}

// NVComputeQueue.exec for the second and later launches of a queue: the
// previous QMD launches this one when it completes.
static inline void nvd_chain(const NVDHandoff& h, uint8_t* prev_qmd, uint64_t qmd_va) {
    nvd_qmd_bits(prev_qmd, h.q_dep_ptr, qmd_va >> 8);
    nvd_qmd_bits(prev_qmd, h.q_dep_action, 1);
    nvd_qmd_bits(prev_qmd, h.q_dep_prefetch, 1);
    nvd_qmd_bits(prev_qmd, h.q_dep_enable, 1);
}

// NVComputeQueue.signal on the queue's last QMD (release slot 0 is free: the
// daemon checked every template).
static inline void nvd_qmd_release(const NVDHandoff& h, uint8_t* qmd, uint64_t sig_va, uint64_t value) {
    nvd_qmd_bits(qmd, h.q_rel_en, 1);
    uint32_t lo = (uint32_t)sig_va, hi;
    memcpy(qmd + h.q_rel_addr, &lo, 4);
    memcpy(&hi, qmd + h.q_rel_addr + 4, 4);
    hi = (hi & ~0xfu) | (uint32_t)(sig_va >> 32);           // bind_sints_to_mem(..., mask=0xf)
    memcpy(qmd + h.q_rel_addr + 4, &hi, 4);
    uint32_t payload[2] = { (uint32_t)value, (uint32_t)(value >> 32) };
    memcpy(qmd + h.q_rel_payload, payload, 8);
}

// ── pushbuffer ───────────────────────────────────────────────────────────

// NVCommandQueue.nvm: an incrementing-method header, then the data words.
static inline void nvd_nvm(std::vector<uint32_t>& pb, uint32_t subc, uint32_t mthd, std::initializer_list<uint32_t> args) {
    pb.push_back((2u << 28) | ((uint32_t)args.size() << 16) | (subc << 13) | (mthd >> 2));
    pb.insert(pb.end(), args);
}

// NVCommandQueue.wait: acquire when the 64-bit semaphore at sig_va >= value.
static inline void nvd_push_wait(std::vector<uint32_t>& pb, const NVDHandoff& h, uint64_t sig_va, uint64_t value) {
    nvd_nvm(pb, 0, h.m_sem_addr_lo, { (uint32_t)sig_va, (uint32_t)(sig_va >> 32), (uint32_t)value, (uint32_t)(value >> 32),
                                      h.f_sem_acquire });
}

// NVComputeQueue.memory_barrier.
static inline void nvd_push_invalidate(std::vector<uint32_t>& pb, const NVDHandoff& h) {
    nvd_nvm(pb, 1, h.m_invalidate, { h.f_invalidate });
}

// NVComputeQueue.exec, first launch of a queue.
static inline void nvd_push_pcas(std::vector<uint32_t>& pb, const NVDHandoff& h, uint64_t qmd_va) {
    nvd_nvm(pb, 1, h.m_pcas_a, { (uint32_t)(qmd_va >> 8) });
    nvd_nvm(pb, 1, h.m_pcas2_b, { 9 });
}

// NVCopyQueue.copy (addresses as data64: high word first).
static inline void nvd_push_copy(std::vector<uint32_t>& pb, const NVDHandoff& h, uint64_t dst, uint64_t src, uint64_t size) {
    const uint64_t step = 1ull << 31;
    for (uint64_t off = 0; off < size; off += step) {
        uint64_t s = src + off, d = dst + off, n = (size - off < step) ? size - off : step;
        nvd_nvm(pb, 4, h.m_dma_offset_in_upper, { (uint32_t)(s >> 32), (uint32_t)s, (uint32_t)(d >> 32), (uint32_t)d });
        nvd_nvm(pb, 4, h.m_dma_line_length_in, { (uint32_t)n });
        nvd_nvm(pb, 4, h.m_dma_launch, { h.f_dma_copy });
    }
}

// NVCopyQueue.signal.
static inline void nvd_push_dma_signal(std::vector<uint32_t>& pb, const NVDHandoff& h, uint64_t sig_va, uint64_t value) {
    nvd_nvm(pb, 4, h.m_dma_sem_a, { (uint32_t)(sig_va >> 32), (uint32_t)sig_va, (uint32_t)value });
    nvd_nvm(pb, 4, h.m_dma_launch, { h.f_dma_sem });
}

// NVCommandQueue._submit_to_gpfifo: the GPFIFO entry for a pushbuffer.
static inline uint64_t nvd_gpfifo_entry(uint64_t pb_va, uint32_t nwords) {
    return ((pb_va / 4) << 2) | ((uint64_t)nwords << 42) | (1ull << 41);
}

} // namespace tinygpu_device

#endif // LIBHMSBEAGLE_GPU_TINYGPUHYBRIDNVDISPATCH_H
