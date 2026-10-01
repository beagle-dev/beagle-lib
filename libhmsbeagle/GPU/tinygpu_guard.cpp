/*
 * tinygpu_guard.cpp -- beagle-tinygpu-guard, TODO.md plan step C10: the crash guard (inv:transport-teardown#7). The plugin spawns
 * it (posix_spawn, a new session, no fd but its end of a socketpair: never a fork, since BEAST hosts a JVM) before its first
 * request to the GPU, with the TinyGPU.app connection, the lock and the state page (TinyGPUHybridNVGuard.h's kGuardSetupHold),
 * waits for its "ready", and sends 'S' and the rest (the GSP queues and the C++ timeline, with what the teardown needs) once
 * the NVDevice is built (plan step C11). Until then the guard can only hold, or close in phase flcn_init.
 *
 * While the plugin lives the guard sends nothing on the connection: two clients of one TinyGPU.app session must take turns,
 * and the plugin has it. At the plugin's own fini the plugin tears the GPU down itself and says "clean" (the guard exits) or
 * "hold" (its unload was not confirmed: the guard keeps the connection open). If the socketpair ends without either, the
 * plugin is gone (killed, crashed) or lost the GPU (plan step C12), and the guard decides as the daemon's fini and EOF path
 * did (nv_dispatch_daemon.py _fini, now the harness's oracle):
 *   - phase flcn_init: nothing started (before booter_load or the COT message); closing is safe;
 *   - phase gsp_init or teardown, or a frame in flight: hold, sending nothing (GSP-RM may be live, or TinyGPU.app would read
 *     the guard's bytes as the rest of a cut frame);
 *   - else the C++ timeline is waited for (HCQSignal.wait: 30 s without progress, the status-queue drain after 200 ms), then
 *     the unload (NVGsp::fini_hw) and the teardown the plugin runs (NVIDIA's on Ada, the RISC-V halt wait on COT); after a
 *     hung wait the unload only. It holds unless the unload was confirmed, nothing hung and (COT) the core halted.
 * To hold is to keep every fd and sleep; SIGINT, SIGHUP and SIGTERM are ignored, so only SIGKILL ends it, after the eGPU is
 * unplugged. Every step is logged through TinyGPULog.h. The AMD C++ boot's guard (plan step A2k) is amd_guard below.
 *   beagle-tinygpu-guard   (its socketpair end is fd 3)
 */

#include "libhmsbeagle/GPU/TinyGPUHybridAMDDevice.h"
#include "libhmsbeagle/GPU/TinyGPUHybridNVGsp.h"
#include "libhmsbeagle/GPU/TinyGPUHybridNVGuard.h"

#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstring>

#include <sys/mman.h>
#include <sys/socket.h>
#include <unistd.h>

using namespace tinygpu_device;

namespace {

[[noreturn]] void hold(const char* why) {
    tg_log("guard: HOLDING the TinyGPU.app connection (%s): closing it could unmap memory the GPU may still use. Unplug the eGPU "
           "first, then kill %d.", why, (int)getpid());
    fprintf(stderr, "beagle-tinygpu-guard: holding the TinyGPU.app connection (%s). Unplug the eGPU first, then kill %d.\n", why,
            (int)getpid());
    for (;;) pause();
}

// HCQSignal.wait on the C++ timeline (hcq.py:274-287, as nv_signal_wait ports it): progress resets the timer, and after 200 ms
// without progress PCIIface.sleep drains the status queue and raises on a device fault. "" once the timeline reached value.
std::string timeline_wait(NVGsp& gsp, const volatile uint64_t* signal, uint64_t value, int timeout_ms) {
    auto cur = [&] { return __atomic_load_n(signal, __ATOMIC_ACQUIRE); };
    int64_t start = nv_now_ms();
    for (;;) {
        const uint64_t prev = cur();
        if (!(prev < value)) return "";
        const int64_t now = nv_now_ms();
        if (!(now - start < timeout_ms)) break;
        if (now - start > 200) {
            try { gsp.stat_q.read_resp([](uint32_t, std::vector<uint8_t>&) { return false; }); }
            catch (const NVError& e) { return "the status-queue drain: " + e.py(); }
            if (gsp.is_err_state) return "RuntimeError: Device fault detected";
            usleep(1000);
        }
        if (cur() != prev) start = nv_now_ms();
    }
    return "RuntimeError: Wait timeout: " + std::to_string(timeout_ms) + " ms! (the signal is not set to " + std::to_string(value) +
           ", but " + std::to_string(cur()) + ")";
}

// The GPU's fini on an AMDev restored from the plugin's fini state (amboot::am_device_fini_safe): whether it saw every queue off
bool amd_fini(TGTransport& t, const amboot::AMFiniState& fs, std::string& why) {
    try {
        amboot::AMDev adev(t, fs);
        return amboot::am_device_fini_safe(adev, why);
    } catch (const TGPyError& e) {
        why = e.py();
    } catch (const am::AMRegError& e) {
        why = std::string("AMRegError: ") + e.what();
    }
    return false;
}

// TODO.md plan step A2k: the AMD C++ boot's guard (GPUInterfaceTinyGPUHybridAMD.cpp, BEAGLE_AMD_CPP_BOOT=1). Its setup holds the
// connection, tinygrad's am_usb4.lock and the state page; once the plugin's AMDev is booted, before any queue is set up, the
// rest brings what AMDev.fini needs. At its own fini the plugin says clean or hold; 'N' is a boot that ended with no
// queue ever live. At an EOF without either, the guard does what the daemon's EOF path did (amd_dispatch_daemon.py exited,
// and tinygrad's finalize ran the IH drain and AMDev.fini; its timeline wait had nothing to wait for after the handoff, and
// the dequeue resets the waves of any work still running), with BEAGLE's hold rule:
//   - phase am_boot (no queue ever live, so nothing on the GPU reads sysmem): close, after that fini if the rest came and no
//     request is in flight (so that the next boot is a partial one), whatever its outcome;
//   - a request in flight (a frame may be cut, or a reply unread), or phase teardown (the plugin's own fini): hold, sending
//     nothing;
//   - phase dispatch: the fini; close if it saw every queue off, else hold.
int amd_guard(int ctl, const GuardSetup& g, const int* raw) {
    uint64_t* state = (uint64_t*)mmap(nullptr, kNVDStateWordsGuard * 8, PROT_READ | PROT_WRITE, MAP_SHARED, raw[2], 0);
    if (state == MAP_FAILED) {
        tg_log("guard: mmap of the state page failed: %s; exiting", strerror(errno));
        return 2;
    }
    TGTransport t;
    t.adopt(raw[0], raw[1]);   // the connection and am_usb4.lock
    const char ready = 'R';
    if (write(ctl, &ready, 1) != 1) {
        tg_log("guard: could not say ready: %s; exiting", strerror(errno));
        return 2;
    }
    tg_log("guard %d: ready for plugin %d (AMD); until its AMDev is booted it can only close, or hold", (int)getpid(), (int)g.parent_pid);

    // the plugin says clean, hold or nothing live, or sends the setup's rest, or goes away
    std::unique_ptr<amboot::AMFiniState> fs;
    char msg = 0;
    ssize_t n;
    for (;;) {
        while ((n = read(ctl, &msg, 1)) < 0 && errno == EINTR) {}
        if (!(n == 1 && msg == 'S' && !fs)) break;
        GuardSetup rest{};
        int none[kGuardFds];
        std::string err = guard_recv_setup(ctl, rest, none);
        if (err.empty() && rest.kind != kGuardSetupRestAMD) err = "not the AMD setup's rest";
        auto st = std::make_unique<amboot::AMFiniState>();
        if (err.empty() && rest.amd_fini_size != sizeof(*st)) {   // read past it, so the stream stays in step
            err = "an AMD fini state of " + std::to_string(rest.amd_fini_size) + " bytes, not " + std::to_string(sizeof(*st));
            std::vector<uint8_t> skip(rest.amd_fini_size < (1u << 20) ? rest.amd_fini_size : 0);
            if (!guard_recv_all(ctl, skip.data(), skip.size())) err += ", cut";
        } else if (err.empty() && !guard_recv_all(ctl, st.get(), sizeof(*st))) err = "the AMD fini state was cut";
        if (!err.empty()) { tg_log("guard %d: the setup's rest failed (%s): it can still only close, or hold", (int)getpid(), err.c_str()); continue; }
        fs = std::move(st);
        t.seed_bar(0, fs->vram_bytes);
        t.seed_bar(5, fs->mmio_bytes);
        tg_log("guard %d: the setup's rest (AMD): from here it can finalize the GPU", (int)getpid());
    }
    if (n == 1 && msg == 'C') {
        tg_log("guard %d: the plugin finalized the GPU itself; exiting", (int)getpid());
        return 0;
    }
    if (n == 1 && msg == 'H') hold("the plugin's own GPU teardown was not confirmed");
    if (n == 1 && msg == 'N') {
        tg_log("guard %d: the plugin's boot ended with no queue ever live: closing is safe; exiting", (int)getpid());
        return 0;
    }

    // the plugin is gone without either
    const uint64_t phase = __atomic_load_n(&state[kGuardStatePhase], __ATOMIC_ACQUIRE);
    const uint64_t in_flight = __atomic_load_n(&state[kGuardStateInFlight], __ATOMIC_ACQUIRE);
    tg_log("guard %d: the plugin went away without fini (%s); state page: phase %llu, request_in_flight %llu%s", (int)getpid(),
           n == 0 ? "EOF" : "an unknown message", (unsigned long long)phase, (unsigned long long)in_flight, fs ? "" : ", no fini state");
    std::string why;
    if (phase == kGuardPhaseAMBoot) {
        if (fs && !in_flight) {
            tg_log("guard: the card is booted, with no queue set up: its fini first, so that its next boot is a partial one");
            amd_fini(t, *fs, why);
            tg_log("guard: the fini %s%s", why.empty() ? "is done" : "failed: ", why.c_str());
        }
        tg_log("guard %d: no queue was ever live, so closing is safe; closing the TinyGPU.app connection", (int)getpid());
        t.close();
        return 0;
    }
    if (in_flight) hold("a request may be cut mid-send, or its reply unread");
    if (phase == kGuardPhaseTeardown) hold("the plugin's own GPU teardown did not finish");
    if (phase != kGuardPhaseDispatch || !fs) hold("a queue may be live, and without the plugin's fini state the guard cannot finalize the GPU");
    tg_log("guard: the GPU's fini, as the daemon's exit ran it: the IH drain, then AMDev.fini");
    const bool off = amd_fini(t, *fs, why);
    if (!why.empty()) tg_log("guard: the fini: %s", why.c_str());
    if (!off) hold("the GPU did not confirm its queues off");
    tg_log("guard %d: the GPU is finalized, every queue off; closing the TinyGPU.app connection", (int)getpid());
    t.close();
    return 0;
}

}  // namespace

int main() {
    for (int s : {SIGINT, SIGHUP, SIGTERM, SIGPIPE}) signal(s, SIG_IGN);
    const int ctl = kGuardFd;
    GuardSetup g{};
    int raw[kGuardFds], fds[kGuardFds] = {-1, -1, -1, -1, -1};
    std::string err = guard_recv_setup(ctl, g, raw);
    if (err.empty() && g.kind != kGuardSetupHold && g.kind != kGuardSetupHoldAMD) err = "a setup's rest before the setup";
    if (!err.empty()) {   // nothing was taken over: the plugin sees no "ready", and its boot fails before any request to the GPU
        tg_log("guard: no setup (%s); exiting", err.c_str());
        return 2;
    }
    if (g.kind == kGuardSetupHoldAMD) return amd_guard(ctl, g, raw);
    fds[kGuardTinyGPU] = raw[0]; fds[kGuardLock] = raw[1]; fds[kGuardState] = raw[2];
    const size_t state_size = kNVDStateWordsGuard * 8;
    uint64_t* state = (uint64_t*)mmap(nullptr, state_size, PROT_READ | PROT_WRITE, MAP_SHARED, fds[kGuardState], 0);
    uint8_t* queues = nullptr;
    uint64_t* signal_page = nullptr;
    auto map_rest = [&] {   // the GSP queues and the C++ timeline: what an unload and its wait need
        void* q = mmap(nullptr, g.queues_size, PROT_READ | PROT_WRITE, MAP_SHARED, fds[kGuardQueues], 0);
        void* s = mmap(nullptr, g.signal_size, PROT_READ, MAP_SHARED, fds[kGuardSignal], 0);
        if (q == MAP_FAILED || s == MAP_FAILED) return false;
        queues = (uint8_t*)q;
        signal_page = (uint64_t*)s;
        return true;
    };
    bool full = false;   // the setup's rest came: the guard can tear the GPU down
    if (state == MAP_FAILED) {
        tg_log("guard: mmap of the state page failed: %s; exiting", strerror(errno));
        return 2;
    }
    TGTransport t;
    t.adopt(fds[kGuardTinyGPU], fds[kGuardLock]);
    const char ready = 'R';
    if (write(ctl, &ready, 1) != 1) {
        tg_log("guard: could not say ready: %s; exiting", strerror(errno));
        return 2;
    }
    tg_log("guard %d: ready for plugin %d; it can only hold until the plugin's boot is done", (int)getpid(), (int)g.parent_pid);

    // the plugin says "clean" or "hold", or sends the setup's rest, or goes away
    char msg = 0;
    ssize_t n;
    for (;;) {
        while ((n = read(ctl, &msg, 1)) < 0 && errno == EINTR) {}
        if (!(n == 1 && msg == 'S' && !full)) break;
        GuardSetup rest{};
        err = guard_recv_setup(ctl, rest, raw);
        if (err.empty() && rest.kind != kGuardSetupRest) err = "not a setup's rest";
        if (err.empty()) {
            fds[kGuardQueues] = raw[0];
            fds[kGuardSignal] = raw[1];
            g = rest;
            if (!map_rest()) err = std::string("mmap of the queues or the timeline: ") + strerror(errno);
        }
        if (!err.empty()) { tg_log("guard %d: the setup's rest failed (%s): it can still only hold", (int)getpid(), err.c_str()); continue; }
        t.seed_bar(0, g.bar0_size);
        full = true;
        tg_log("guard %d: the setup's rest (%s%s): from here it can tear the GPU down", (int)getpid(), g.chip_name, g.cot ? ", COT" : "");
    }
    if (n == 1 && msg == 'C') {
        tg_log("guard %d: the plugin tore the GPU down itself; exiting", (int)getpid());
        return 0;
    }
    if (n == 1 && msg == 'H') hold("the plugin's own GPU teardown was not confirmed");
    if (n == 1 && msg == 'N') {   // plan step C11: a boot that stopped before GSP-RM started (the daemon's flcn_init, in the oracle)
        tg_log("guard %d: the plugin's boot stopped before GSP-RM started: nothing to unload, closing is safe; exiting", (int)getpid());
        return 0;
    }

    // the plugin is gone, or lost the GPU (plan step C12), without either: the oracle's fini decision (nv_dispatch_daemon.py
    // _fini), on the state page
    const uint64_t phase = __atomic_load_n(&state[kGuardStatePhase], __ATOMIC_ACQUIRE);
    const uint64_t in_flight = __atomic_load_n(&state[kGuardStateInFlight], __ATOMIC_ACQUIRE);
    const uint64_t last = __atomic_load_n(&state[kGuardStateLastSubmitted], __ATOMIC_ACQUIRE);
    const uint32_t seq = (uint32_t)__atomic_load_n(&state[kGuardStateSeq], __ATOMIC_ACQUIRE);
    tg_log("guard %d: the plugin went away without fini (%s); state page: phase %llu, frame_in_flight %llu, last_submitted %llu, "
           "seq %u, C++ timeline %llu", (int)getpid(), n == 0 ? "EOF" : "an unknown message", (unsigned long long)phase,
           (unsigned long long)in_flight, (unsigned long long)last, seq, signal_page ? (unsigned long long)__atomic_load_n(signal_page, __ATOMIC_ACQUIRE) : 0ull);
    if (phase == kGuardPhaseFlcnInit) {
        tg_log("guard: the plugin's falcon boot stopped before GSP-RM started: nothing to unload, closing is safe");
        return 0;
    }
    if (phase == kGuardPhaseGspInit) hold("the plugin did not finish booting GSP-RM");
    if (phase == kGuardPhaseTeardown) hold("the plugin's own GPU teardown did not finish");
    if (phase != kGuardPhaseDispatch || in_flight) hold("a frame may be cut mid-send");
    if (!full) hold("the plugin's boot did not finish, and without its queues the guard cannot unload the GPU");

    NVBar0 bar0{&t};
    NVFalcon flcn(bar0, g.chip_id, g.cot != 0);
    flcn.chip_name = g.chip_name;
    NVFiniDiag diag;
    bool hung = false;
    try {
        NVGsp gsp(bar0, flcn, queues, g.cmdq_off, g.statq_off, g.queue_size, g.libos_args_sysmem, seq, flcn.wait_ms);
        std::string why = timeline_wait(gsp, signal_page, last, 30000);
        if (!why.empty()) {
            tg_log("guard: the C++ timeline: %s: the hung path", why.c_str());
            hung = true;
        }
        tg_log("guard: the GPU teardown: the %s unload RPC (seq %u)%s", g.level0 ? "LEVEL_0" : "FAST_UNLOAD", seq,
               hung ? " only (hung)" : g.cot ? ", then the RISC-V halt wait (COT)" : g.images.present ? ", then NVIDIA's teardown" : "");
        gsp.fini_hw(diag, g.level0 != 0);
        if (!hung && g.cot) flcn.cot_fini_hw(diag);
        else if (!hung) flcn.fini_hw(diag, g.images);
    } catch (const NVError& e) {
        tg_log("guard: the GPU teardown failed: %s", e.py().c_str());
    }
    tg_log("guard: the GPU teardown: %s", diag.json().c_str());
    // the daemon's hold rule: an unconfirmed unload, a hang (a channel may still poll the sysmem timeline), or on COT a core
    // that did not halt (the FMC and the ACR may still use the boot structures in sysmem)
    if (!diag.unload_ok || hung || (diag.cot && !diag.halted)) hold("the GPU did not confirm its teardown");
    tg_log("guard %d: the GPU is torn down; closing the TinyGPU.app connection", (int)getpid());
    t.close();
    return 0;
}
