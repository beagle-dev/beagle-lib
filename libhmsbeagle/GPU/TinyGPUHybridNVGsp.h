/*
 * TinyGPUHybridNVGsp.h
 *
 * TODO.md plan step C5: the C++ side's GSP client for the unload at exit. Ported statement by statement from tinygrad
 * (tinygrad/runtime/support/nv/ip.py at a9830e2b4): NVRpcQueue (ip.py:19-91: the RPC framing, the checksum, continuation
 * records, the read loop with its CPU sequencer and error events, and wait_resp's clock), NV_GSP.run_cpu_seq with all nine
 * ops (ip.py:629-661) and rpc_unloading_guest_driver (ip.py:611-614); and what nv_init_helper.py adds around them (plan
 * steps P1, P2): the wait for the GSP to report itself suspended after the unload RPC, the LEVEL_0 unload
 * (BEAGLE_NV_UNLOAD_LEVEL=0) with the 20 s SEC2 sleep covering its sequencer, and the logs of the unload's status-queue
 * events and sequencer ops. The queues are the daemon's, shared: their memory comes over as TinyGPU.app's sysmem fd, and
 * everything else they hold (write and read pointers) lives in that memory, except the command queue's sequence number,
 * which the daemon hands over. On the COT boot (Blackwell, plan step B2) nv_init_helper's additions for it too: the unload's
 * report starts not halted, and a CPU sequence with ops 5-8 (NV_FLCN's primitives) is refused before any op runs.
 */

#ifndef LIBHMSBEAGLE_GPU_TINYGPUHYBRIDNVGSP_H
#define LIBHMSBEAGLE_GPU_TINYGPUHYBRIDNVGSP_H

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "libhmsbeagle/GPU/TinyGPUHybridNVFalcon.h"
#include "libhmsbeagle/GPU/TinyGPUNVRMTables.h"

namespace tinygpu_device {

class NVGsp;

// NVRpcQueue (ip.py:19-91) over one queue of the shared GSP queue memory.
class NVRpcQueue {
public:
    // __init__ (ip.py:20-30): view is the queue (its msgqTxHeader first); completion, the other queue, whose header locates
    // this one's read pointer. wait_cond on the header's entryOff, which the queue's writer sets.
    NVRpcQueue(NVGsp* gsp, uint8_t* view, uint64_t view_size, uint8_t* completion, int wait_ms);

    uint32_t* rx_view = nullptr;   // this queue's read pointer, in the other queue's header (set for the command queue in init_hw)
    uint32_t seq = 0;              // the next RPC's GSP_MSG_QUEUE_ELEMENT.seqNum
    nv::msgqTxHeader tx{};         // the header as read at construction

    // _checksum (ip.py:32-36)
    static uint32_t checksum(std::vector<uint8_t> data) {
        if (size_t pad = (8 - data.size() % 8) % 8) data.resize(data.size() + pad, 0);
        uint64_t c = 0;
        for (size_t off = 0; off < data.size(); off += 8) { uint64_t w; memcpy(&w, &data[off], 8); c ^= w; }
        return nv_hi32(c) ^ nv_lo32(c);
    }

    // send_rpc (ip.py:57-60): the first record, then continuation records for the rest
    void send_rpc(uint32_t func, const std::vector<uint8_t>& msg) {
        const size_t max_payload = tx.msgSize * 16 - sizeof(nv::GSP_MSG_QUEUE_ELEMENT) - sizeof(nv::rpc_message_header_v);
        send_rpc_record(func, std::vector<uint8_t>(msg.begin(), msg.begin() + std::min(max_payload, msg.size())));
        for (size_t off = max_payload; off < msg.size(); off += max_payload)
            send_rpc_record(nv::NV_VGPU_MSG_FUNCTION_CONTINUATION_RECORD,
                            std::vector<uint8_t>(msg.begin() + off, msg.begin() + std::min(off + max_payload, msg.size())));
    }

    // read_resp (ip.py:62-87), a generator, driven by its consumer: visit(function, msg) runs for each message it yields and
    // returns true once the consumer stops iterating (wait_resp's next() takes the first match and leaves the rest).
    template <class V> void read_resp(V&& visit);

    // wait_resp (ip.py:89-93)
    std::vector<uint8_t> wait_resp(uint32_t cmd, int timeout = 10000) {
        const int64_t start_time = nv_now_ms();
        while (nv_now_ms() - start_time < timeout) {
            std::vector<uint8_t> found;
            bool got = false;
            read_resp([&](uint32_t func, std::vector<uint8_t>& message) {
                if (func != cmd) return false;
                found = std::move(message);
                got = true;
                return true;
            });
            if (got) return found;
        }
        throw NVError("RuntimeError", "Timeout waiting for RPC response for command " + std::to_string(cmd));
    }

private:
    NVGsp* gsp_;
    volatile uint32_t* tx_view_;   // view.view(fmt='I')
    uint8_t* queue_mv_;            // view.view(tx.entryOff, tx.msgSize * tx.msgCount)
    uint64_t queue_len_;

    static constexpr size_t kWritePtr = offsetof(nv::msgqTxHeader, writePtr) / 4;

    // _send_rpc_record (ip.py:38-55)
    void send_rpc_record(uint32_t func, const std::vector<uint8_t>& payload);
};

// The GSP client of NV_GSP (ip.py:346-661) the unload needs, with nv_init_helper's wrappers.
class NVGsp {
public:
    NVGsp(NVBar0& dev, NVFalcon& flcn, uint8_t* queues, uint64_t cmd_q_off, uint64_t stat_q_off, uint64_t queue_size,
          uint64_t libos_args_sysmem, uint32_t seq, int wait_ms)
        : dev(dev), flcn(flcn), libos_args_sysmem(libos_args_sysmem),
          // init_rm_args (ip.py:382-386) and init_hw (ip.py:510-512): the command queue, then the status queue completing it
          cmd_q(this, queues + cmd_q_off, queue_size, nullptr, wait_ms),
          stat_q(this, queues + stat_q_off, queue_size, queues + cmd_q_off, wait_ms) {
        cmd_q.rx_view = (uint32_t*)(queues + stat_q_off + stat_q.tx.rxHdrOff);
        cmd_q.seq = seq;
    }

    NVBar0& dev;
    NVFalcon& flcn;
    uint64_t libos_args_sysmem;
    bool is_err_state = false;   // NVDev.is_err_state
    bool in_unload = false;      // nv_init_helper's _in_unload: the unload's status-queue events and sequencer ops are logged
    std::vector<int> seq_refused;   // nv_init_helper's beagle_seq_refused: the COT sequencer's refused ops
    int rpc_timeout_ms = 10000;  // wait_resp's timeout (tests shorten it)
    NVRpcQueue cmd_q, stat_q;
    std::function<void(uint32_t)> after_rpc;   // sees each RPC's queue sequence number once it is queued (the state page)

    nv_regs::NVReg<NVBar0> reg(nv_regs::NVRegId id) const { return flcn.reg(id); }

    // NV_GSP.run_cpu_seq (ip.py:629-661), after nv_init_helper's log of its ops (_logged_run_cpu_seq)
    void run_cpu_seq(const std::vector<uint8_t>& seq_buf) {
        using namespace nv_regs;
        const size_t hdr_sz = sizeof(nv::rpc_run_cpu_sequencer_v17_00);
        if (seq_buf.size() < hdr_sz)
            throw NVError("ValueError", "Buffer size too small (" + std::to_string(seq_buf.size()) + " instead of at least " +
                          std::to_string(hdr_sz) + " bytes)");
        nv::rpc_run_cpu_sequencer_v17_00 hdr;
        memcpy(&hdr, seq_buf.data(), hdr_sz);
        if ((seq_buf.size() - hdr_sz) % 4) throw NVError("TypeError", "memoryview: length is not a multiple of itemsize");
        std::vector<uint32_t> words((seq_buf.size() - hdr_sz) / 4);
        if (!words.empty()) memcpy(words.data(), seq_buf.data() + hdr_sz, words.size() * 4);
        if (words.size() > hdr.cmdIndex) words.resize(hdr.cmdIndex);
        tg_log("CPU sequencer (%s): ops %s", in_unload ? "during unload" : "boot", seq_ops(words).c_str());
        // ops 5-8 drive the falcon through NV_FLCN's reset/start_cpu/wait_cpu_halted, which NV_FLCN_COT lacks: nv_init_helper
        // refuses them before the first write (plan steps B1, B2), and they stay at the head of GSP-RM's status queue
        if (flcn.cot) {
            std::vector<int> cot;
            for (int op : seq_op_list(words))
                if (op >= 0x5 && op <= 0x8 && std::find(cot.begin(), cot.end(), op) == cot.end()) cot.push_back(op);
            std::sort(cot.begin(), cot.end());
            if (!cot.empty()) {
                seq_refused = cot;
                std::string l = "[";
                for (size_t k = 0; k < cot.size(); ++k) l += (k ? ", " : "") + std::to_string(cot[k]);
                throw NVError("RuntimeError", "CPU sequencer ops " + l + "] drive the falcon through NV_FLCN, which the COT boot (" +
                              flcn.chip_name + ") does not have (plan step B2): refused before any of the sequence ran");
            }
        }

        size_t i = 0;
        auto next = [&]() -> uint32_t {
            if (i >= words.size()) throw NVError("StopIteration", "");
            return words[i++];
        };
        while (i < words.size()) {
            uint32_t op = words[i++];
            if (op == 0x0) { uint32_t a = next(); dev.wreg(a, next()); }   // reg write
            else if (op == 0x1) {   // reg modify
                uint32_t addr = next(), val = next(), mask = next();
                dev.wreg(addr, (dev.rreg(addr) & ~mask) | (val & mask));
            } else if (op == 0x2) {   // reg poll
                uint32_t addr = next(), mask = next(), val = next();
                next(); next();
                char m[96];
                snprintf(m, sizeof(m), "Register 0x%x not equal to 0x%x after polling", addr, val);
                nv_wait_cond(flcn.wait_ms, [&] { return (uint64_t)(dev.rreg(addr) & mask); }, val, m);
            } else if (op == 0x3) flcn.sleep(next() / 1e6);   // delay us
            else if (op == 0x4) {   // save reg
                uint32_t addr = next(), index = next();
                if (index >= 8) throw NVError("IndexError", "invalid index");
                hdr.regSaveArea[index] = dev.rreg(addr);
            } else if (op == 0x5) {   // core reset
                flcn.reset(flcn.falcon);
                flcn.disable_ctx_req(flcn.falcon);
            } else if (op == 0x6) flcn.start_cpu(flcn.falcon);
            else if (op == 0x7) flcn.wait_cpu_halted(flcn.falcon);
            else if (op == 0x8) {   // core resume
                flcn.reset(flcn.falcon, true);

                reg(NV_PGSP_FALCON_MAILBOX0).write(nv_lo32(libos_args_sysmem));
                reg(NV_PGSP_FALCON_MAILBOX1).write(nv_hi32(libos_args_sysmem));

                flcn.start_cpu(flcn.sec2);
                nv_wait_cond(flcn.wait_ms, [&] { return reg(NV_PGC6_BSI_SECURE_SCRATCH_14).read_bitfields()["boot_stage_3_handoff"]; },
                             kNVTrue, "SEC2 didn't hand off");

                uint32_t mailbox = reg(NV_PFALCON_FALCON_MAILBOX0).with_base(flcn.sec2).read();
                if (mailbox != 0x0) {
                    char m[80];
                    snprintf(m, sizeof(m), "Falcon SEC2 failed to execute, mailbox is %08x", mailbox);
                    throw NVError("AssertionError", m);
                }
            } else throw NVError("ValueError", "Unknown op code " + std::to_string(op) + " in run_cpu_seq");
        }
    }

    // NV_GSP.rpc_unloading_guest_driver (ip.py:611-614), or nv_init_helper's LEVEL_0 variant (newLevel 0, the driver-unload
    // level NVIDIA uses: gpu.c:3269-3273) when BEAGLE_NV_UNLOAD_LEVEL=0
    void rpc_unloading_guest_driver(bool level0) {
        nv::rpc_unloading_guest_driver_v data{};
        data.bInPMTransition = 0;
        data.bGc6Entering = 0;
        data.newLevel = level0 ? 0 : 1u << 6;   // __GPU_STATE_FLAGS_FAST_UNLOAD
        std::vector<uint8_t> b(sizeof(data));
        memcpy(b.data(), &data, sizeof(data));
        cmd_q.send_rpc(nv::NV_VGPU_MSG_FUNCTION_UNLOADING_GUEST_DRIVER, b);
        stat_q.wait_resp(nv::NV_VGPU_MSG_FUNCTION_UNLOADING_GUEST_DRIVER, rpc_timeout_ms);
    }

    // NV_GSP.fini_hw (ip.py:522) as nv_init_helper wraps it (_gsp_fini_hw_with_suspend_wait, plan step P1): tinygrad's unload
    // RPC, then NVIDIA's wait for the GSP to report itself suspended (MAILBOX0 == 0x80000000; 570.144 kernel_gsp_tu102.c:
    // 1116-1139, nouveau r535 gsp.c:1772-1779). A LEVEL_0 unload may post RUN_CPU_SEQUENCER, whose op 8 polls BSI right after
    // starting SEC2, so BEAGLE's 20 s sleep covers it (plan step P2 (c)). A failed RPC throws, with diag's unload_ok false.
    void fini_hw(NVFiniDiag& diag, bool level0) {
        using namespace nv_regs;
        diag = NVFiniDiag();
        // COT (Blackwell): not halted until NV_FLCN_COT.fini_hw's wait proves it, so any exit before that wait leaves halted
        // false and the connection is held (plan step B1)
        diag.cot = flcn.cot;
        in_unload = true;
        if (level0) flcn.sleep_after_sec2_start = true;
        try { rpc_unloading_guest_driver(level0); }
        catch (...) { in_unload = flcn.sleep_after_sec2_start = false; throw; }
        in_unload = flcn.sleep_after_sec2_start = false;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(2000);   // _SUSPEND_TIMEOUT_S
        uint32_t mailbox0;
        while ((mailbox0 = reg(NV_PGSP_FALCON_MAILBOX0).read()) != 0x80000000 && std::chrono::steady_clock::now() < deadline)
            flcn.sleep(0.01);
        diag.mailbox0 = mailbox0;
        diag.have_mailbox0 = true;
        diag.wpr2_lo = reg(NV_PFB_PRI_MMU_WPR2_ADDR_LO).read();
        diag.wpr2_hi = reg(NV_PFB_PRI_MMU_WPR2_ADDR_HI).read();
        diag.have_wpr2 = true;
        diag.unload_ok = mailbox0 == 0x80000000;
        diag.riscv_cpuctl = reg(NV_PRISCV_RISCV_CPUCTL).with_base(flcn.falcon).read();
        diag.have_cpuctl = true;
        tg_log("after the unload RPC: GSP MAILBOX0=0x%08x (%s), RISCV_CPUCTL=0x%08x, WPR2_LO=0x%08x, WPR2_HI=0x%08x", mailbox0,
               diag.unload_ok ? "suspended" : "NOT SUSPENDED", diag.riscv_cpuctl, diag.wpr2_lo, diag.wpr2_hi);
    }

private:
    // the ops of a sequence as _seq_ops parses them, up to the first unknown one
    static std::vector<int> seq_op_list(const std::vector<uint32_t>& w) {
        static const int nargs[] = {2, 3, 5, 1, 2, 0, 0, 0, 0};
        std::vector<int> ops;
        for (size_t i = 0; i < w.size() && w[i] <= 8; i += 1 + nargs[w[i]]) ops.push_back((int)w[i]);
        return ops;
    }

    // nv_init_helper's _seq_ops: the ops of a sequence, as run_cpu_seq steps through its operands
    static std::string seq_ops(const std::vector<uint32_t>& w) {
        static const int nargs[] = {2, 3, 5, 1, 2, 0, 0, 0, 0};
        std::string s = "[";
        for (size_t i = 0; i < w.size();) {
            if (w[i] > 8) { s += (s.size() > 1 ? ", 'unknown " : "'unknown ") + std::to_string(w[i]) + "'"; break; }
            s += (s.size() > 1 ? ", " : "") + std::to_string(w[i]);
            i += 1 + nargs[w[i]];
        }
        return s + "]";
    }
};

inline NVRpcQueue::NVRpcQueue(NVGsp* gsp, uint8_t* view, uint64_t view_size, uint8_t* completion, int wait_ms)
    : gsp_(gsp), tx_view_((volatile uint32_t*)view) {
    nv_wait_cond(wait_ms, [&] { return (uint64_t)tx_view_[offsetof(nv::msgqTxHeader, entryOff) / 4]; }, 0x1000, "RPC queue not initialized");
    memcpy(&tx, view, sizeof(tx));
    if (completion) {
        nv::msgqTxHeader comp_tx;
        memcpy(&comp_tx, completion, sizeof(comp_tx));
        rx_view = (uint32_t*)(completion + comp_tx.rxHdrOff);
    }
    queue_mv_ = view + tx.entryOff;
    queue_len_ = std::min<uint64_t>((uint64_t)tx.msgSize * tx.msgCount, view_size - tx.entryOff);
}

inline void NVRpcQueue::send_rpc_record(uint32_t func, const std::vector<uint8_t>& payload) {
    nv::rpc_message_header_v header{};
    header.signature = nv::NV_VGPU_MSG_SIGNATURE_VALID;
    header.rpc_result = nv::NV_VGPU_MSG_RESULT_RPC_PENDING;
    header.rpc_result_private = nv::NV_VGPU_MSG_RESULT_RPC_PENDING;
    header.header_version = 3 << 24;
    header.function = func;
    header.length = (uint32_t)(payload.size() + 0x20);

    std::vector<uint8_t> msg(sizeof(header));
    memcpy(msg.data(), &header, sizeof(header));
    msg.insert(msg.end(), payload.begin(), payload.end());
    nv::GSP_MSG_QUEUE_ELEMENT phdr{};
    phdr.elemCount = (uint32_t)((msg.size() + sizeof(nv::GSP_MSG_QUEUE_ELEMENT) + tx.msgSize - 1) / tx.msgSize);   // ceildiv
    phdr.seqNum = seq;
    std::vector<uint8_t> whole(sizeof(phdr));
    memcpy(whole.data(), &phdr, sizeof(phdr));
    whole.insert(whole.end(), msg.begin(), msg.end());
    phdr.checkSum = checksum(whole);
    memcpy(whole.data(), &phdr, sizeof(phdr));
    whole.resize(std::max<size_t>(whole.size(), (size_t)phdr.elemCount * tx.msgSize), 0);   // .ljust

    uint32_t wp = tx_view_[kWritePtr];
    uint64_t off = (uint64_t)wp * tx.msgSize;
    uint64_t first = std::min<uint64_t>(whole.size(), queue_len_ > off ? queue_len_ - off : 0);
    memcpy(queue_mv_ + off, whole.data(), first);
    if (first < whole.size()) memcpy(queue_mv_, whole.data() + first, whole.size() - first);
    tx_view_[kWritePtr] = (wp + phdr.elemCount) % tx.msgCount;
    std::atomic_thread_fence(std::memory_order_seq_cst);   // System.memory_barrier

    seq += 1;
    if (gsp_->after_rpc) gsp_->after_rpc(seq);   // before the doorbell: a command in the queue is always counted (the state page)
    gsp_->reg(nv_regs::NV_PGSP_QUEUE_HEAD)[0].write(0x0);
}

template <class V> inline void NVRpcQueue::read_resp(V&& visit) {
    std::atomic_thread_fence(std::memory_order_seq_cst);   // System.memory_barrier
    volatile uint32_t* rx = rx_view;
    while (rx[0] != tx_view_[kWritePtr]) {
        uint64_t off = (uint64_t)rx[0] * tx.msgSize;
        nv::rpc_message_header_v hdr;
        memcpy(&hdr, queue_mv_ + off + 0x30, sizeof(hdr));
        uint64_t end = std::min<uint64_t>(off + 0x50 + hdr.length, queue_len_);
        std::vector<uint8_t> msg(queue_mv_ + std::min<uint64_t>(off + 0x50, end), queue_mv_ + end);

        // Handling special functions
        if (hdr.function == nv::NV_VGPU_MSG_EVENT_GSP_RUN_CPU_SEQUENCER) {
            try { gsp_->run_cpu_seq(msg); }
            catch (const NVError& e) {   // a StopIteration raised inside a generator becomes a RuntimeError (PEP 479)
                if (e.type == "StopIteration") throw NVError("RuntimeError", "generator raised StopIteration");
                throw;
            }
        }
        else if (hdr.function == nv::NV_VGPU_MSG_EVENT_OS_ERROR_LOG) {   // tinygrad prints it; a library writes to stderr
            std::string text(msg.size() > 12 ? (const char*)msg.data() + 12 : "", msg.size() > 12 ? msg.size() - 12 : 0);
            while (!text.empty() && text.back() == '\0') text.pop_back();
            fprintf(stderr, "TinyGPU/NV: nv usb4: GSP LOG: %s\n", text.c_str());
            tg_log("GSP LOG: %s", text.c_str());
        }

        gsp_->is_err_state |= hdr.function == nv::NV_VGPU_MSG_EVENT_OS_ERROR_LOG || hdr.function == nv::NV_VGPU_MSG_EVENT_MMU_FAULT_QUEUED;

        // Update the read pointer
        rx[0] = (uint32_t)((rx[0] + (hdr.length + tx.msgSize - 1) / tx.msgSize) % tx.msgCount);
        std::atomic_thread_fence(std::memory_order_seq_cst);

        if (hdr.rpc_result != 0)
            throw NVError("RuntimeError", "RPC call " + std::to_string(hdr.function) + " failed with result " + std::to_string(hdr.rpc_result));
        if (gsp_->in_unload) tg_log("status-queue event during unload: 0x%x", hdr.function);
        if (visit(hdr.function, msg)) return;
    }
}

} // namespace tinygpu_device

#endif // LIBHMSBEAGLE_GPU_TINYGPUHYBRIDNVGSP_H
