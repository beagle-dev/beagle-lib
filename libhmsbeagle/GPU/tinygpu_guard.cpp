/*
 * tinygpu_guard.cpp -- beagle-tinygpu-guard, TODO.md plan step C10: the crash guard that replaces the daemon's keeper role
 * (inv:transport-teardown#7). The plugin spawns it (posix_spawn, a new session, no fd but its end of a socketpair: never a
 * fork, since BEAST hosts a JVM) at level flcn_hw once the C++ NVDevice and its timeline exist, hands it the TinyGPU.app
 * connection, the lock, the GSP queues, the state page and the C++ timeline over SCM_RIGHTS with what the teardown needs
 * (GuardSetup), and waits for its "ready" before it sets the state page's keeper word and asks the daemon to hand the keeper
 * role over (TinyGPUHybridNVGuard.h).
 *
 * While the plugin lives the guard sends nothing on the connection: two clients of one TinyGPU.app session must take turns,
 * and the plugin has it. At the plugin's own fini the plugin tears the GPU down itself and says "clean" (the guard exits) or
 * "hold" (its unload was not confirmed: the guard keeps the connection open), or "stand down" (the daemon kept the role). If
 * the socketpair ends without any of them, the plugin is gone (killed, crashed): unless the keeper word says the plugin handed
 * the role over, the daemon decides and the guard exits; otherwise the guard decides as the daemon's fini and EOF path did
 * (nv_dispatch_daemon.py _fini):
 *   - phase flcn_init: nothing started (before booter_load or the COT message); closing is safe;
 *   - phase gsp_init or teardown, or a frame in flight: hold, sending nothing (GSP-RM may be live, or TinyGPU.app would read
 *     the guard's bytes as the rest of a cut frame);
 *   - else the C++ timeline is waited for (HCQSignal.wait: 30 s without progress, the status-queue drain after 200 ms), then
 *     the unload (NVGsp::fini_hw) and the teardown the plugin runs (NVIDIA's on Ada, the RISC-V halt wait on COT); after a
 *     hung wait the unload only. It holds unless the unload was confirmed, nothing hung and (COT) the core halted.
 * To hold is to keep every fd and sleep; SIGINT, SIGHUP and SIGTERM are ignored, so only SIGKILL ends it, after the eGPU is
 * unplugged. Every step is logged through TinyGPULog.h.
 *   beagle-tinygpu-guard   (its socketpair end is fd 3)
 */

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

}  // namespace

int main() {
    for (int s : {SIGINT, SIGHUP, SIGTERM, SIGPIPE}) signal(s, SIG_IGN);
    const int ctl = kGuardFd;
    GuardSetup g{};
    int fds[kGuardFds];
    std::string err = guard_recv_setup(ctl, g, fds);
    if (!err.empty()) {   // nothing was taken over: the plugin sees no "ready" and keeps the daemon
        tg_log("guard: no setup (%s); exiting", err.c_str());
        return 2;
    }
    const size_t state_size = kNVDStateWordsGuard * 8;
    uint8_t* queues = (uint8_t*)mmap(nullptr, g.queues_size, PROT_READ | PROT_WRITE, MAP_SHARED, fds[kGuardQueues], 0);
    uint64_t* state = (uint64_t*)mmap(nullptr, state_size, PROT_READ | PROT_WRITE, MAP_SHARED, fds[kGuardState], 0);
    uint64_t* signal_page = (uint64_t*)mmap(nullptr, g.signal_size, PROT_READ, MAP_SHARED, fds[kGuardSignal], 0);
    if (queues == MAP_FAILED || state == MAP_FAILED || signal_page == MAP_FAILED) {
        tg_log("guard: mmap of the queues, the state page or the timeline failed: %s; exiting", strerror(errno));
        return 2;
    }
    TGTransport t;
    t.adopt(fds[kGuardTinyGPU], fds[kGuardLock]);
    t.seed_bar(0, g.bar0_size);   // the daemon mapped BAR0 in its boot
    const char ready = 'R';
    if (write(ctl, &ready, 1) != 1) {
        tg_log("guard: could not say ready: %s; exiting", strerror(errno));
        return 2;
    }
    tg_log("guard %d: ready for plugin %d (%s%s, level %s)", (int)getpid(), (int)g.parent_pid, g.chip_name, g.cot ? ", COT" : "",
           g.level_name);

    // the plugin says "clean" or "hold", or goes away
    char msg = 0;
    ssize_t n;
    while ((n = read(ctl, &msg, 1)) < 0 && errno == EINTR) {}
    if (n == 1 && msg == 'C') {
        tg_log("guard %d: the plugin tore the GPU down itself; exiting", (int)getpid());
        return 0;
    }
    if (n == 1 && msg == 'H') hold("the plugin's own GPU teardown was not confirmed");
    if (n == 1 && msg == 'X') {
        tg_log("guard %d: the daemon keeps the keeper role; exiting", (int)getpid());
        return 0;
    }
    if (__atomic_load_n(&state[kGuardStateKeeper], __ATOMIC_ACQUIRE) != kGuardKeeperGuard) {
        tg_log("guard %d: the plugin went away before it handed the keeper role over: the daemon decides; exiting", (int)getpid());
        return 0;
    }

    // the plugin is gone without either: the daemon's fini decision (nv_dispatch_daemon.py _fini), on the state page
    const uint64_t phase = __atomic_load_n(&state[kGuardStatePhase], __ATOMIC_ACQUIRE);
    const uint64_t in_flight = __atomic_load_n(&state[kGuardStateInFlight], __ATOMIC_ACQUIRE);
    const uint64_t last = __atomic_load_n(&state[kGuardStateLastSubmitted], __ATOMIC_ACQUIRE);
    const uint32_t seq = (uint32_t)__atomic_load_n(&state[kGuardStateSeq], __ATOMIC_ACQUIRE);
    tg_log("guard %d: the plugin went away without fini (%s); state page: phase %llu, frame_in_flight %llu, last_submitted %llu, "
           "seq %u, C++ timeline %llu", (int)getpid(), n == 0 ? "EOF" : "an unknown message", (unsigned long long)phase,
           (unsigned long long)in_flight, (unsigned long long)last, seq, (unsigned long long)__atomic_load_n(signal_page, __ATOMIC_ACQUIRE));
    if (phase == kGuardPhaseFlcnInit) {
        tg_log("guard: the plugin's falcon boot stopped before GSP-RM started: nothing to unload, closing is safe");
        return 0;
    }
    if (phase == kGuardPhaseGspInit) hold("the plugin did not finish booting GSP-RM");
    if (phase == kGuardPhaseTeardown) hold("the plugin's own GPU teardown did not finish");
    if (phase != kGuardPhaseDispatch || in_flight) hold("a frame may be cut mid-send");

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
