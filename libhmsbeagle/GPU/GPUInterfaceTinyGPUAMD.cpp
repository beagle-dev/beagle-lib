/*
 * GPUInterfaceTinyGPUAMD.cpp
 *
 * BEAGLE's AMD eGPU backend (an RX 7900 XT, gfx1100, through TinyGPU.app), with no Python (TODO.md plan step A2l): this
 * process boots the card with tinygrad's AM driver in C++ (TinyGPUAMDBoot.h, plan step A2), sets up AMDDevice.__init__'s
 * queues and buffers (TinyGPUAMDDevice.h), and runs launches, copies, allocations and synchronization on tinygrad's
 * PM4 and SDMA queues (TinyGPUAMDRuntime.h, TinyGPUAMDDispatch.h; plan step A1), with the build's ahead-of-time
 * HSACOs (plan step A1j). Each part is golden-tested byte for byte against the tinygrad code it ports; the Python daemon that
 * did all of it before is now the tests' oracle (tinygpu_tests/oracle/amd_dispatch_daemon.py). The first instance in a
 * process boots the card, and every instance shares that boot, which lasts until exit (TODO.md plan step A5, NV's P5), when
 * this file finalizes the card; each HSACO variant an instance uses is loaded once. The crash guard (beagle-tinygpu-guard)
 * keeps the card from before the boot's first request: if this process dies, it finalizes the card or holds (plan step
 * A2k). A failed instance, or a lost GPU, returns errors to BEAGLE instead of exiting its host (plan step A3).
 *
 * Why a port of tinygrad's code rather than a stream of our own: four hand-built PM4 dispatch attempts crashed the host
 * identically (a DART "read of DVA 0" panic, STATUS.md AMD §3-§11); only stock tinygrad's full AMDDevice/PCIIface/HCQCompiled
 * stack ever worked on this hardware (§8).
 */

#ifdef FW_TINYGPU

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <dlfcn.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "libhmsbeagle/beagle.h"
#include "libhmsbeagle/GPU/GPUImplDefs.h"
#include "libhmsbeagle/GPU/GPUImplHelper.h"
#include "libhmsbeagle/GPU/GPUInterface.h"
#include "libhmsbeagle/GPU/KernelResource.h"
#include "libhmsbeagle/GPU/GPUInterfaceTinyGPUAMD.h"
#include "libhmsbeagle/GPU/TinyGPUTransport.h"
#include "libhmsbeagle/GPU/TinyGPUFirmware.h"
#include "libhmsbeagle/GPU/TinyGPUAMDDevice.h"   // the C++ boot (TODO.md plan step A2), and the runtime
#include "libhmsbeagle/GPU/TinyGPUNVGuard.h"     // the crash guard's setup and state page (plan step A2k)

#ifdef TINYGPU_AMD_HSACO
#include "libhmsbeagle/GPU/kernels/TinyGPUAMDHsaco.h"   // the build's ahead-of-time HSACOs (TODO.md plan step A1j)
#endif

namespace tinygpu_device {

// The build's ahead-of-time HSACO of a variant ("SP_4" ... "DP_256") for an arch, compiled as tinygrad's compile_hip compiles
// BEAGLE's OpenCL source (tinygpu_amd_compile, golden_amd_hsaco.py), or null when the build had no comgr or not that arch
static const unsigned char* amd_embedded_hsaco(const std::string& variant, const std::string& arch, size_t& n) {
    n = 0;
#ifdef TINYGPU_AMD_HSACO
    for (const TinyGPUAMDHsaco& h : kTinyGPUAMDHsacos)
        if (variant == h.variant && arch == h.arch) { n = (size_t)(h.end - h.begin); return h.begin; }
#else
    (void)variant; (void)arch;
#endif
    return nullptr;
}

// TODO.md plan step A7: double precision on the AMD card, which the build's HSACOs include when it compiled them (DP_4 ...
// DP_256, beside the SP_ variants); the card computes it natively, at a lower rate
bool AmdSupportsDouble() {
#ifdef TINYGPU_AMD_HSACO
    for (const TinyGPUAMDHsaco& h : kTinyGPUAMDHsacos)
        if (strncmp(h.variant, "DP_", 3) == 0) return true;
#endif
    return false;
}

// ── Opt-in profiling (BEAGLE_AMD_PROFILE=1): each operation's host time, the GPU's work included where it waits ─────
static bool amd_profile_enabled() {
    static const bool enabled = (getenv("BEAGLE_AMD_PROFILE") != nullptr);
    return enabled;
}
static inline std::chrono::steady_clock::time_point amd_profile_start() {
    return std::chrono::steady_clock::now();
}
static void amd_profile_end(const char* label, std::chrono::steady_clock::time_point t0) {
    if (!amd_profile_enabled()) return;
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "TinyGPU/AMD: [profile] %-24s %8lld us\n", label, (long long)us);
}
static std::map<std::string, long long> g_amdKernelLaunches;   // the launches per kernel, for the profile's report at the fini

// GetFunction's handle: one kernel of a variant's programs
struct AMDKernelHandle {
    std::string name;
    const AMDProgramRecord* rec = nullptr;
};

// TODO.md plan step A5: one HSACO variant's programs ("SP_4" ... "DP_256"), loaded once per process for the first instance
// that needs them, and shared by every instance of that variant
struct AMDProgramSet {
    std::map<std::string, AMDProgramRecord> kernels;
    std::map<std::string, AMDKernelHandle> handles;
};

struct AMDState {
    AMDRuntime* rt = nullptr;   // the C++ runtime (TODO.md plan step A1g) on the boot's handoff
    std::unique_ptr<amboot::AMDev> adev;             // the C++ boot (TODO.md plan step A2h)
    std::unique_ptr<amboot::AMDDeviceState> dstate;
    int guard_ctl = -1;          // the C++ boot's crash guard (TODO.md plan step A2k): its socketpair
    pid_t guard_pid = 0;
    uint64_t* state = nullptr;   // the state page shared with it (TinyGPUNVGuard.h's kGuardState* words)
    int state_fd = -1;
    bool lost_said = false;      // plan step A3: the lost GPU was reported
    pid_t owner_pid = 0;         // plan step A5: the process that booted; a child forked from it never finalizes the card
    std::string arch;            // the card's (gfx1100): which of the build's HSACOs serve it
    std::map<std::string, AMDProgramSet> programs;   // the variants loaded so far (amd_programs)
};

// The process's one booted card, which every instance shares until exit (TODO.md plan step A5, NV's plan step P5): the first
// instance boots it on its TinyGPU.app connection, which the card then keeps. What is an instance's own is its AMDInstance.
static AMDState* g_amd = nullptr;
static pid_t g_amdHeld = 0;   // plan step A3: a crash guard this process left holding the card: no later instance can connect

// Plan step A5: one lock around everything the instances share, as NV's nv_mutex, so that a request to TinyGPU.app never
// interleaves with another thread's and the rings, timeline, kernargs and pool stay consistent. Recursive and timed, as NV's;
// never destroyed: a GPUInterface may be destroyed after the static destructors ran.
static std::recursive_timed_mutex& amd_mutex() {
    static std::recursive_timed_mutex* m = new std::recursive_timed_mutex;
    return *m;
}

// ── Launch batching (STATUS.md AMD §26) ─────────────────────────────────────
// AmdLaunchKernelImpl queues an instance's launches, and amdFlushLaunchQueue submits them as chained batches before each of
// that instance's h2d, d2h and sync and at its release, so their order relative to its memory operations is kept: launches
// only enqueue PM4, and the copies and synchronize wait for earlier work themselves (as tinygrad's _copyin, _copyout and
// synchronize do).
struct AMDPendingLaunch {
    const AMDKernelHandle* kernel;
    int grid[3];
    int block[3];
    std::vector<unsigned long long> ptrs;
    std::vector<unsigned int> ints;
};

// Plan step A5: what belongs to one BEAGLE instance (GPUInterface::amdInstance); the rest is the shared card above
struct AMDInstance {
    AMDProgramSet* programs = nullptr;       // its variant's
    std::vector<AMDPendingLaunch> pending;   // its launches queued since it last flushed
    bool failed = false;   // plan step A3: its setup, a kernel lookup or an allocation failed
    bool oom = false;      // plan step M1: ... for lack of GPU memory (BeagleGPUImpl then returns BEAGLE_ERROR_OUT_OF_MEMORY)
};
static bool g_amdSetupOOM = false;   // plan step M1: the last boot's setup ran out of VRAM (its buffers and the pool)

// TODO.md plan step A3, NV's plan step C12 on the AMD path: an instance whose setup failed, or whose GPU is lost (any runtime
// failure: a wait that timed out, a GPU fault, a broken TinyGPU.app stream, a launch the HSACOs cannot serve), sends the GPU
// nothing more: its calls do nothing, read-backs are NaN, and BeagleGPUImpl returns errors (GPUInterface::GetDeviceLost).
// BEAGLE never exits its host. A lost GPU is lost to every instance, and stays this process's until exit, whose fini still
// finalizes the card or has the crash guard hold it (plan steps A2k, A5); if the process dies first, the guard decides.
static bool amd_failed(const AMDInstance& in) {
    if (in.failed || !g_amd) return true;
    if (!g_amd->rt->error) return false;
    if (!g_amd->lost_said) {
        g_amd->lost_said = true;
        fprintf(stderr, "TinyGPU/AMD: the GPU is lost to this process: no instance's work reaches it any more, and BEAGLE's calls "
                "return errors; at exit the card is still finalized, or the crash guard holds it\n");
        tg_log("the GPU is lost to this process (plan step A3): %s", g_amd->rt->error_msg.c_str());
    }
    return true;
}

static void amd_test_kill(const char* point, AMDState* g);   // the crash guard's test hook (TODO.md plan step A2k)

// The old daemon's chained launch_batch (TinyGPUAMDDispatch.h): one queue per 1024 launches, each a timeline wait and
// memory_barrier, the execs with their kernargs slots, then a signal and a submit. A kernargs wrap first waits for the GPU
// to finish everything submitted (TinyGPUAMDRuntime.h).
static void amdFlushLaunchQueue(AMDInstance& in) {
    if (in.pending.empty()) return;
    if (amd_failed(in)) { in.pending.clear(); return; }
    AMDRuntime& rt = *g_amd->rt;
    auto t0 = amd_profile_start();
    const size_t n = in.pending.size();
    AMDComputeQueue q;
    bool open = false;
    for (size_t i = 0; i < n && !rt.error; ++i) {
        const AMDPendingLaunch& pl = in.pending[i];
        const AMDKernel& k = pl.kernel->rec->k;
        if (pl.ptrs.size() * 8 + pl.ints.size() * 4 > k.kernargs_alloc_size) {
            rt.fail("launch of " + pl.kernel->name + ": its arguments do not fit its " + std::to_string(k.kernargs_alloc_size) + "-byte kernargs");
            break;
        }
        if (!open) {
            q = AMDComputeQueue();
            q.wait(rt.signal_va, (uint32_t)(rt.timeline_value - 1));
            q.memory_barrier();
            open = true;
        }
        const uint64_t before = rt.kargs_bump.ptr;
        const uint64_t off = rt.kargs_bump.alloc(k.kernargs_alloc_size, 8);   // HCQProgram.fill_kernargs
        if (off < before && !rt.wait_idle()) break;                          // wrapped: the slots may still be in use
        const uint32_t grid[3] = {(uint32_t)pl.grid[0], (uint32_t)pl.grid[1], (uint32_t)pl.grid[2]};
        const uint32_t block[3] = {(uint32_t)pl.block[0], (uint32_t)pl.block[1], (uint32_t)pl.block[2]};
        std::vector<uint64_t> ptrs(pl.ptrs.begin(), pl.ptrs.end());
        q.exec(k, rt.exec, rt.kargs + off, rt.kargs_va + off, ptrs.data(), (int)ptrs.size(), pl.ints.data(), (int)pl.ints.size(), grid, block);
        if ((i + 1) % 1024 == 0 || i + 1 == n) {
            q.signal(rt.signal_va, rt.next_timeline());
            if (!rt.submit_compute(q)) break;
            amd_test_kill("batch", g_amd);   // plan step A2k's test: killed with this batch on the GPU
            open = false;
        }
    }
    in.pending.clear();
    amd_profile_end("launch_batch (C++)", t0);
    if (rt.error) fprintf(stderr, "TinyGPU/AMD: launch_batch(%zu kernels) failed: %s\n", n, rt.error_msg.c_str());
}

// tinygrad's AMD lock (System.flock_acquire("am_usb4.lock"), which tinygrad's AMD device takes), held for the process's life;
// its fd in lock_fd, for the crash guard
static std::string amd_take_am_lock(int& lock_fd) {
    static int fd = -1;
    lock_fd = fd;
    if (fd >= 0) return "";
    const std::string path = tg_temp_path("am_usb4.lock");
    const bool exists = access(path.c_str(), F_OK) == 0;
    int f = exists ? open(path.c_str(), O_RDWR | O_CLOEXEC) : open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0666);
    if (f < 0) return "cannot open the lock file " + path + ": " + strerror(errno);
    if (!exists) fchmod(f, 0666);
    if (flock(f, LOCK_EX | LOCK_NB) != 0) { close(f); return "Failed to acquire lock file am_usb4.lock (another process has the eGPU)"; }
    fd = lock_fd = f;
    return "";
}

// ── The C++ boot's crash guard (TODO.md plan step A2k): NV's beagle-tinygpu-guard (tinygpu_guard.cpp's amd_guard) keeps the
// TinyGPU.app connection if this process dies, and then finalizes the GPU as the daemon's EOF path did, or holds ──────────

static void amd_phase(AMDState* g, uint64_t phase) {
    if (g->state) __atomic_store_n(&g->state[kGuardStatePhase], phase, __ATOMIC_RELEASE);
}

// Plan step A2k's offline tests: BEAGLE_AMD_TEST_KILL=<point> kills this process there with SIGKILL, as a crash would, so the
// crash guard's decisions can be checked: "boot_guard" (the guard started, nothing sent to the GPU yet), "boot_rest" (the AMDev
// booted and the guard has its fini state, no queue set up yet), "batch" (right after the first launch batch's submit), "idle"
// (at fini, the GPU idle), "frame" (the same, with the state page saying a request is in flight) and "teardown" (in this side's
// own fini). Never set outside the harness.
static void amd_test_kill(const char* point, AMDState* g) {
    static const char* k = getenv("BEAGLE_AMD_TEST_KILL");
    if (!k || strcmp(k, point) != 0) return;
    if (g && g->state && strcmp(point, "frame") == 0) __atomic_store_n(&g->state[kGuardStateInFlight], 1, __ATOMIC_RELEASE);
    tg_log("BEAGLE_AMD_TEST_KILL=%s: SIGKILL", point);
    fflush(stderr);
    kill(getpid(), SIGKILL);
}

// The state page (NV's nvdStatePage): four 64-bit words shared with the crash guard only, a POSIX shm segment unlinked at
// once. Phase am_boot; from here the transport keeps the in-flight word around every request (TGTransport::set_in_flight).
static std::string amdStatePage(AMDState& g) {
    char name[32];   // macOS PSHMNAMLEN is 31
    snprintf(name, sizeof(name), "/beagle-amd.%d", (int)getpid());
    shm_unlink(name);   // only a killed process with this pid could have left it
    int fd = shm_open(name, O_RDWR | O_CREAT | O_EXCL, 0600);
    if (fd < 0) return std::string("the state page: shm_open: ") + strerror(errno);
    shm_unlink(name);
    const size_t size = kNVDStateWordsGuard * 8;
    void* m = ftruncate(fd, size) == 0 ? mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0) : MAP_FAILED;
    if (m == MAP_FAILED) {
        close(fd);
        return std::string("the state page: ") + strerror(errno);
    }
    g.state = (uint64_t*)m;   // zero-filled: nothing in flight
    g.state_fd = fd;
    amd_phase(&g, kGuardPhaseAMBoot);
    tg_transport().set_in_flight(&g.state[kGuardStateInFlight]);
    return "";
}

// The crash guard's executable: BEAGLE_AMD_GUARD (the test harness's), or beagle-tinygpu-guard next to this plugin
static std::string amd_guard_path() {
    const char* e = getenv("BEAGLE_AMD_GUARD");
    if (e && e[0]) return e;
    Dl_info info;
    if (!dladdr((void*)&amd_guard_path, &info) || !info.dli_fname) return "";
    std::string so = info.dli_fname;
    return so.substr(0, so.rfind('/') + 1) + "beagle-tinygpu-guard";
}

// The guard, spawned before the boot's first request to the GPU with what holding takes (the TinyGPU.app connection, tinygrad's
// am_usb4.lock, which is this path's lock, the state page, and the transport's nv_usb4.lock, so that no other process connects
// while it holds: kGuardSetupHoldAMD). Before it said ready nothing went to the GPU, so a failed start ends the boot.
static std::string amdGuardStart(AMDState& g, int am_lock_fd) {
    const std::string path = amd_guard_path();
    if (path.empty()) return "the crash guard: no beagle-tinygpu-guard next to the plugin";
    int ctl = -1;
    pid_t pid = 0;
    std::string err = guard_spawn(path, ctl, pid);
    if (!err.empty()) return "the crash guard: " + err;
    GuardSetup s{};
    s.magic = kGuardMagic;
    s.size = sizeof(s);
    s.kind = kGuardSetupHoldAMD;
    s.nfds = guard_setup_nfds(kGuardSetupHoldAMD);
    s.parent_pid = (uint32_t)getpid();
    const int fds[4] = {tg_transport().fd(), am_lock_fd, g.state_fd, tg_transport().lock_fd()};
    char r = 0;
    struct pollfd pfd = {ctl, POLLIN, 0};
    if (!guard_send_setup(ctl, s, fds)) err = std::string("its setup: ") + strerror(errno);
    else if (poll(&pfd, 1, 10000) != 1 || read(ctl, &r, 1) != 1 || r != 'R') err = "it never said ready";
    if (!err.empty()) {
        close(ctl);
        kill(pid, SIGKILL);
        waitpid(pid, nullptr, 0);
        return "the crash guard: " + err;
    }
    g.guard_ctl = ctl;
    g.guard_pid = pid;
    TG_STATUS("TinyGPU/AMD: the crash guard (pid %d) keeps the GPU from here\n", (int)pid);
    tg_log("AMD C++ boot: the crash guard (pid %d) keeps the GPU", (int)pid);
    return "";
}

// ... and once the AMDev is booted, before any queue is set up, the rest (kGuardSetupRestAMD, then the AMFiniState): from
// here the guard can finalize the GPU
static std::string amdGuardRest(AMDState& g) {
    auto fs = std::make_unique<amboot::AMFiniState>();
    try { g.adev->fini_state(*fs); }
    catch (const TGPyError& e) { return "the crash guard's setup rest: " + e.py(); }
    GuardSetup s{};
    s.magic = kGuardMagic;
    s.size = sizeof(s);
    s.kind = kGuardSetupRestAMD;
    s.nfds = guard_setup_nfds(kGuardSetupRestAMD);
    s.parent_pid = (uint32_t)getpid();
    s.amd_fini_size = sizeof(*fs);
    const char m = 'S';
    if (write(g.guard_ctl, &m, 1) != 1 || !guard_send_setup(g.guard_ctl, s, nullptr) || !guard_send_all(g.guard_ctl, fs.get(), sizeof(*fs)))
        return std::string("the crash guard's setup rest: ") + strerror(errno);
    return "";
}

// This side is done with the GPU: 'C' (it finalized the GPU and saw every queue off), 'H' (hold: not seen off) or 'N' (no
// queue was ever live) to the guard, which exits at a 'C' or an 'N' (waited for, so that its locks are free for the next
// boot); then the state page goes
static void amdGuardEnd(AMDState* g, char m) {
    if (g->guard_ctl >= 0) {
        if (write(g->guard_ctl, &m, 1) != 1)
            fprintf(stderr, "TinyGPU/AMD: the crash guard (pid %d) did not take this side's '%c': it decides as at a crash\n", (int)g->guard_pid, m);
        close(g->guard_ctl);
        g->guard_ctl = -1;
        if (m == 'H') {
            fprintf(stderr, "TinyGPU/AMD: the GPU's queues were not seen off, so the crash guard (pid %d) holds the TinyGPU.app connection "
                    "(closing it could unmap memory the GPU may still read). Unplug the eGPU first, then kill %d.\n", (int)g->guard_pid,
                    (int)g->guard_pid);
            g_amdHeld = g->guard_pid;   // plan step A3: TinyGPU.app serves the guard now, and would never answer a new connection
        }
        for (int i = 0; i < 50 && m != 'H' && waitpid(g->guard_pid, nullptr, WNOHANG) == 0; ++i) usleep(100000);
    }
    tg_transport().set_in_flight(nullptr);
    if (g->state) { munmap(g->state, kNVDStateWordsGuard * 8); g->state = nullptr; }
    if (g->state_fd >= 0) { close(g->state_fd); g->state_fd = -1; }
}

// The C++ boot's fini (TODO.md plan step A2h): what the daemon's exit runs, HCQCompiled.finalize then AMDev.fini, after the
// runtime's own last synchronize; then clean or hold to the crash guard (plan step A2k), as the fini saw every queue off or
// not ('N' if no queue was ever live). The state page says teardown first once a queue may be live, so that a death meanwhile
// holds. Errors are reported, never fatal: the process is on its way out.
static void amdCppBootFini(AMDState* g) {
    for (const auto& kv : g_amdKernelLaunches)   // BEAGLE_AMD_PROFILE: each kernel's launches, as NV's report lists them
        fprintf(stderr, "TinyGPU/AMD: [profile]   kernel %s n=%lld\n", kv.first.c_str(), kv.second);
    g_amdKernelLaunches.clear();
    if (!g->adev) return;
    const bool live = g->state && __atomic_load_n(&g->state[kGuardStatePhase], __ATOMIC_ACQUIRE) == kGuardPhaseDispatch;
    if (live) amd_phase(g, kGuardPhaseTeardown);
    amd_test_kill("teardown", g);
    std::string why;
    const bool off = amboot::am_device_fini_safe(*g->adev, why);
    if (!why.empty()) fprintf(stderr, "TinyGPU/AMD: the C++ fini failed: %s\n", why.c_str());
    g->adev.reset();
    amdGuardEnd(g, !live ? 'N' : off ? 'C' : 'H');
}

// TODO.md plan step A2h: the boot, AMDDevice.__init__'s setup and cmd_handoff's allocations in C++ (no daemon), then the C++
// runtime on them; each instance's variant then loads its programs (amd_programs, plan step A5). The build must have an HSACO
// of the first instance's variant for this card before any queue goes live. Null if the boot failed; once it succeeded, a
// later failure finalizes the GPU. The crash guard (plan step A2k) keeps the GPU from before the boot's first request: it has
// the AMDev's fini state before any queue is set up, and the state page says dispatch from just before the first one.
// Every firmware file the card's boot reads (TinyGPUAMDBootTables.h's manifest), located, or downloaded into BEAGLE's cache
// (TinyGPUFirmware.h), before anything is written to the GPU, as NV's nv_fw_prefetch does: a file that cannot be had stops
// the boot before it starts rather than in its middle. "" or why not.
static std::string amd_fw_prefetch() {
    for (const nvfw::TGFirmware& f : am::fw::kFirmware) {
        TGFirmwareFile file;
        const std::string err = tg_fw_locate(f, file);
        if (!err.empty()) return err;
    }
    return "";
}

static AMDState* amdCppBootSetup(const std::string& variant) {
    auto t0 = amd_profile_start();
    TGTransport& tg = tg_transport();
    int am_lock = -1;
    std::string err = amd_take_am_lock(am_lock);
    if (!err.empty()) { fprintf(stderr, "TinyGPU/AMD: %s\n", err.c_str()); return nullptr; }
    AMDState* g = new AMDState{};
    err = amdStatePage(*g);
    if (err.empty()) err = amdGuardStart(*g, am_lock);
    if (!err.empty()) {
        fprintf(stderr, "TinyGPU/AMD: %s\n", err.c_str());
        amdGuardEnd(g, 'N');
        delete g;
        return nullptr;
    }
    amd_test_kill("boot_guard", g);
    tg.resize_bar(0, err);   // PCIIfaceBase.__init__ (system.py:263): contextlib.suppress(Exception)
    amboot::AMBlobLoader loader = [](const std::string& name, std::vector<uint8_t>& out) -> std::string {
        for (const nvfw::TGFirmware& f : am::fw::kFirmware)
            if (name == f.name) {
                TGFirmwareFile file;
                std::string e = tg_fw_locate(f, file);
                if (!e.empty()) return e;
                out.assign(file.data(), file.data() + file.size());
                return "";
            }
        return "not in the AMD firmware manifest (TinyGPUAMDBootTables.h)";
    };
    TG_STATUS("TinyGPU/AMD: the C++ boot...\n");
    fflush(stderr);
    err.clear();
    try { g->adev = std::make_unique<amboot::AMDev>(tg, loader); }
    catch (const TGPyError& e) { err = e.py(); }
    catch (const am::AMRegError& e) { err = e.what(); }
    if (!err.empty()) {   // no queue was set up: the guard may close
        fprintf(stderr, "TinyGPU/AMD: the C++ boot failed: %s\n", err.c_str());
        amdGuardEnd(g, 'N');
        delete g;
        return nullptr;
    }
    const amboot::Ver& gc = g->adev->ip_ver.at(amboot::GC);
    char arch[16];
    snprintf(arch, sizeof(arch), "gfx%d%x%x", gc[0], gc[1], gc[2]);
    TG_STATUS("TinyGPU/AMD: C++ boot done (%s boot) — arch=%s\n", g->adev->partial_boot ? "partial" : "full", arch);
    err = amdGuardRest(*g);
    if (!err.empty()) {   // still no queue: this side finalizes the card, and the guard may close
        fprintf(stderr, "TinyGPU/AMD: %s; finalizing the GPU\n", err.c_str());
        amdCppBootFini(g);
        delete g;
        return nullptr;
    }
    amd_test_kill("boot_rest", g);
    size_t aot_size = 0;
    const unsigned char* aot = amd_embedded_hsaco(variant, arch, aot_size);
    const char* mb = getenv("BEAGLE_AMD_DATA_MB");
    const uint64_t pool_size = mb ? strtoull(mb, nullptr, 10) << 20 : 0;   // 0: am_handoff's default, half the VRAM
    g_amdSetupOOM = false;
    try {
        if (!aot) throw TGPyError("RuntimeError", "this build has no HSACO for " + variant + " on " + arch + " (built without comgr, or not for "
                                  "this card), and nothing compiles at run time");
        g->dstate = std::make_unique<amboot::AMDDeviceState>();
        amd_phase(g, kGuardPhaseDispatch);   // the compute queue goes live next: from here the guard finalizes the GPU, or holds
        amboot::am_device_init(*g->adev, *g->dstate);
        AMDHandoff h;
        std::vector<uint8_t*> maps;
        amboot::am_handoff(*g->adev, *g->dstate, pool_size, h, maps);
        g->rt = new AMDRuntime;
        amd_runtime_attach_mapped(*g->rt, h, maps.data(), tg);
        amd_profile_end("C++ boot", t0);
        TG_STATUS("TinyGPU/AMD: C++ runtime: handed over after the C++ boot (VRAM pool %llu MiB, timeline %llu)\n",
                (unsigned long long)(h.pool_size >> 20), (unsigned long long)g->rt->timeline_value);
    } catch (const std::exception& e) {
        const TGPyError* py = dynamic_cast<const TGPyError*>(&e);
        fprintf(stderr, "TinyGPU/AMD: after the C++ boot: %s; finalizing the GPU\n", py ? py->py().c_str() : e.what());
        if (py && py->type == "MemoryError") {   // plan step M1
            g_amdSetupOOM = true;
            fprintf(stderr, "TinyGPU/AMD: out of GPU memory: the card's %llu MiB of VRAM cannot hold the setup's buffers and a %llu MiB "
                    "VRAM pool (%s); this instance fails\n", (unsigned long long)(g->adev->vram_size >> 20),
                    (unsigned long long)((pool_size ? pool_size : g->adev->vram_size / 2) >> 20),
                    mb ? "BEAGLE_AMD_DATA_MB: lower it" : "half the VRAM, the default");
        }
        if (g->rt) { amd_runtime_detach(*g->rt); delete g->rt; g->rt = nullptr; }
        amdCppBootFini(g);
        delete g;
        return nullptr;
    }
    g->arch = arch;
    g->owner_pid = getpid();
    fflush(stderr);
    return g;
}

// TODO.md plan step A5: a variant's programs, loaded once per process for the first instance that needs them (AMDProgram.__init__
// for each kernel of its HSACO, and _ensure_has_local_memory: TinyGPUAMDRuntime.h), or null (said; oom if the VRAM pool
// could not hold them, plan step M1)
static AMDProgramSet* amd_programs(const std::string& variant, bool& oom) {
    oom = false;
    auto it = g_amd->programs.find(variant);
    if (it != g_amd->programs.end()) return &it->second;
    size_t n = 0;
    const unsigned char* hsaco = amd_embedded_hsaco(variant, g_amd->arch, n);
    if (!hsaco) {
        fprintf(stderr, "TinyGPU/AMD: this build has no HSACO for %s on %s (built without comgr, or not for this card), and nothing "
                "compiles at run time\n", variant.c_str(), g_amd->arch.c_str());
        return nullptr;
    }
    auto t0 = amd_profile_start();
    AMDProgramSet& ps = g_amd->programs[variant];
    const std::string err = amd_runtime_load_programs(*g_amd->rt, hsaco, n, ps.kernels);
    amd_profile_end("programs (C++)", t0);
    if (!err.empty()) {
        fprintf(stderr, "TinyGPU/AMD: loading %s's programs: %s\n", variant.c_str(), err.c_str());
        oom = err.rfind("the VRAM pool is too small", 0) == 0;   // amd_runtime_load_programs' allocation failures
        if (oom) fprintf(stderr, "TinyGPU/AMD: out of GPU memory: %llu MiB left of the %llu MiB VRAM pool (BEAGLE_AMD_DATA_MB sets it, by "
                         "default half the VRAM)\n", (unsigned long long)(g_amd->rt->available() >> 20),
                         (unsigned long long)(g_amd->rt->h.pool_size >> 20));
        g_amd->programs.erase(variant);
        return nullptr;
    }
    for (const auto& kv : ps.kernels) ps.handles[kv.first] = AMDKernelHandle{kv.first, &kv.second};
    TG_STATUS("TinyGPU/AMD: C++ runtime: %s's programs loaded (%zu kernels, scratch %llu MiB)\n", variant.c_str(),
            ps.kernels.size(), (unsigned long long)(g_amd->rt->exec.scratch_size >> 20));
    return &ps;
}

// The card's fini at exit (plan step A5): this process's last synchronize (the fini then dequeues the queues), the C++ fini
// (what tinygrad's exit runs, plan step A2h) and the clean or the hold to the crash guard; then the TinyGPU.app connection
// closes (a guard that holds keeps its own copy)
static void amdFiniDevice() {
    AMDState* g = g_amd;
    if (!g->rt->synchronize())
        fprintf(stderr, "TinyGPU/AMD: the last synchronize failed: %s\n", g->rt->error_msg.c_str());
    amd_test_kill("idle", g);    // plan step A2k's tests: killed here, the GPU idle,
    amd_test_kill("frame", g);   // ... or with the state page saying a request is in flight
    amdCppBootFini(g);
    amd_runtime_detach(*g->rt);
    delete g->rt;
    delete g;
    g_amd = nullptr;
    tg_transport().close();
}

// Plan step A5: the card outlives its instances, as NV's GPU does (plan step P5) and tinygrad's devices do (device.py finalizes
// them at exit): later instances share this boot instead of booting again. A thread still holding the lock after 35 s (past
// the 30 s wait timeout) may be mid-request: then nothing is sent, and the crash guard, at the end of this process, finalizes
// the card or holds, as the state page says. A child forked after the boot inherits this handler and the TinyGPU.app
// connection, and returns at once: the card is its parent's (checked before the lock, which a fork can copy held).
static void amdAtExit() {
    if (!g_amd || g_amd->owner_pid != getpid()) return;
    std::unique_lock<std::recursive_timed_mutex> lk(amd_mutex(), std::defer_lock);
    if (!lk.try_lock_for(std::chrono::seconds(35))) {
        fprintf(stderr, "TinyGPU/AMD: another thread is still using the GPU at exit; not finalizing it from here: the crash guard "
                "does at the end of this process, or holds, as the state page says\n");
        return;
    }
    amdFiniDevice();
}

// ── GPUInterface entry points ─────────────────────────────────────────────────

// Plan step A5: an instance created while another has the card booted shares that boot, since TinyGPU.app serves one
// connection at a time and a new one from here would wait forever (NV's NvAttachShared, plan step P5). 1 if it shares it, 0
// if nothing is booted, -1 (said) if the GPU was lost earlier in this process.
int AmdAttachShared(GPUInterface* self) {
    std::lock_guard<std::recursive_timed_mutex> lk(amd_mutex());
    if (!g_amd) return 0;
    if (g_amd->rt->error) {   // plan step A3: no instance uses a lost GPU, and this process boots no other
        fprintf(stderr, "TinyGPU/AMD: the GPU was lost earlier in this process; no instance can use it until the process restarts\n");
        return -1;
    }
    self->amdInstance = new AMDInstance;
    return 1;
}

void AmdSetDevice(GPUInterface* self, int paddedStateCount, int categoryCount,
                   int patternCount, int unpaddedPatternCount, int tipCount, long flags) {
    std::lock_guard<std::recursive_timed_mutex> lk(amd_mutex());
    // The first instance's boot takes Initialize()'s TinyGPU.app connection (self->tgpuSock, plan step C3: tg_transport()),
    // which the card then keeps until exit; later instances share that boot (AmdAttachShared, plan step A5).
    const bool dp = (flags & BEAGLE_FLAG_PRECISION_DOUBLE) != 0;
    const std::string variant = (dp ? "DP_" : "SP_") + std::to_string(paddedStateCount);
    AMDInstance* in = (AMDInstance*)self->amdInstance;
    if (!in) {   // the first: it boots the card
        self->amdInstance = in = new AMDInstance;
        const std::string fw = amd_fw_prefetch();
        if (!fw.empty()) fprintf(stderr, "%s\nTinyGPU/AMD: not booting: nothing was written to the GPU\n", fw.c_str());
        g_amd = fw.empty() ? amdCppBootSetup(variant) : nullptr;
        if (!g_amd) {
            in->oom = fw.empty() && g_amdSetupOOM;
            fprintf(stderr, "TinyGPU/AMD: the GPU's setup failed (above); this instance fails\n");
        } else {
            self->tgpuSock = -1;   // plan step A5: the card's connection now, which outlives this instance
            if (atexit(amdAtExit) != 0)
                fprintf(stderr, "TinyGPU/AMD: atexit failed; the card is finalized only by the crash guard, at the end of this process\n");
        }
    }
    if (!amd_failed(*in) && !(in->programs = amd_programs(variant, in->oom))) fprintf(stderr, "TinyGPU/AMD: this instance fails\n");
    if (!in->programs) in->failed = true;   // plan step A3: beagleCreateInstance returns an error (BeagleGPUImpl), and the host goes on

    self->InitializeKernelResource(paddedStateCount, (flags & BEAGLE_FLAG_PRECISION_DOUBLE) != 0);
    self->supportDoublePrecision = ((flags & BEAGLE_FLAG_PRECISION_DOUBLE) != 0);
    if (self->kernelResource) {
        self->kernelResource->categoryCount        = categoryCount;
        self->kernelResource->patternCount         = patternCount;
        self->kernelResource->unpaddedPatternCount = unpaddedPatternCount;
        self->kernelResource->flags                = flags;
    }
}

GPUFunction AmdGetFunction(GPUInterface* self, const char* name) {
    std::lock_guard<std::recursive_timed_mutex> lk(amd_mutex());
    AMDInstance* in = (AMDInstance*)self->amdInstance;
    if (!in || !in->programs) return nullptr;
    auto it = in->programs->handles.find(name);
    if (it != in->programs->handles.end()) return &it->second;
    if (!in->failed) fprintf(stderr, "TinyGPU/AMD: GetFunction(%s): kernel not found in the build's HSACO; this instance fails\n", name);
    in->failed = true;   // plan step A3: beagleCreateInstance returns an error, not an exit
    return nullptr;
}

void AmdSynchronizeHost(GPUInterface* self) {
    std::lock_guard<std::recursive_timed_mutex> lk(amd_mutex());
    AMDInstance* in = (AMDInstance*)self->amdInstance;
    if (!in || amd_failed(*in)) return;
    amdFlushLaunchQueue(*in);  // otherwise queued-but-unsent launches wouldn't be submitted yet to wait for
    auto t0 = amd_profile_start();
    if (!g_amd->rt->synchronize()) fprintf(stderr, "TinyGPU/AMD: sync failed: %s\n", g_amd->rt->error_msg.c_str());
    amd_profile_end("sync (C++)", t0);
}

GPUPtr AmdAllocateMemory(GPUInterface* self, size_t sz) {
    std::lock_guard<std::recursive_timed_mutex> lk(amd_mutex());
    AMDInstance* in = (AMDInstance*)self->amdInstance;
    if (!in || amd_failed(*in)) return 0;
    auto t0 = amd_profile_start();
    uint64_t va = 0;
    if (!g_amd->rt->alloc(sz, va)) {   // BEAGLE does not check: address 0 would reach the GPU
        fprintf(stderr, "TinyGPU/AMD: out of GPU memory: an allocation of %.1f MiB, with %.1f MiB left of the %llu MiB VRAM pool "
                "(BEAGLE_AMD_DATA_MB sets it, by default half the VRAM); this instance fails\n", sz / 1048576.0,
                g_amd->rt->available() / 1048576.0, (unsigned long long)(g_amd->rt->h.pool_size >> 20));
        in->failed = true;   // plan step A3: so nothing of it reaches the GPU, and BeagleGPUImpl returns an error,
        in->oom = true;      // BEAGLE_ERROR_OUT_OF_MEMORY (plan step M1)
    }
    amd_profile_end("alloc (C++)", t0);
    return (GPUPtr)va;
}

// Plan step C14: the block goes back to the pool (TinyGPUPool.h), once this instance's queued launches, which may use it, are
// submitted. An address no allocation returned is ignored.
void AmdFreeMemory(GPUInterface* self, GPUPtr p) {
    std::lock_guard<std::recursive_timed_mutex> lk(amd_mutex());
    AMDInstance* in = (AMDInstance*)self->amdInstance;
    if (!g_amd || !p) return;
    if (in) amdFlushLaunchQueue(*in);
    g_amd->rt->release((uint64_t)p);
}

void AmdMemcpyHostToDevice(GPUInterface* self, GPUPtr dst, const void* src, size_t sz) {
    std::lock_guard<std::recursive_timed_mutex> lk(amd_mutex());
    AMDInstance* in = (AMDInstance*)self->amdInstance;
    if (!in || amd_failed(*in) || !src || !sz) return;
    amdFlushLaunchQueue(*in);  // preserve ordering: queued launches must be submitted before this write
    auto t0 = amd_profile_start();
    if (!amd_copyin(*g_amd->rt, g_amd->rt->staging, (uint64_t)dst, (const uint8_t*)src, sz))
        fprintf(stderr, "TinyGPU/AMD: h2d(addr=0x%llx, sz=%zu) failed: %s\n", (unsigned long long)dst, sz, g_amd->rt->error_msg.c_str());
    amd_profile_end("h2d (C++)", t0);
}

void AmdMemcpyDeviceToHost(GPUInterface* self, void* dst, const GPUPtr src, size_t sz) {
    std::lock_guard<std::recursive_timed_mutex> lk(amd_mutex());
    AMDInstance* in = (AMDInstance*)self->amdInstance;
    if (!in || !dst || !sz) return;
    // plan step A3: a failed instance or a lost GPU reads back NaN (all bits set), so nothing it returns looks like a result
    if (amd_failed(*in)) { memset(dst, 0xff, sz); return; }
    amdFlushLaunchQueue(*in);  // preserve ordering: queued launches must complete before this read
    auto t0 = amd_profile_start();
    if (!amd_copyout(*g_amd->rt, g_amd->rt->staging, (uint8_t*)dst, (uint64_t)src, sz)) {
        fprintf(stderr, "TinyGPU/AMD: d2h(addr=0x%llx, sz=%zu) failed: %s\n", (unsigned long long)src, sz, g_amd->rt->error_msg.c_str());
        memset(dst, 0xff, sz);
    }
    amd_profile_end("d2h (C++)", t0);
}

size_t AmdGetAvailableMemory() {
    std::lock_guard<std::recursive_timed_mutex> lk(amd_mutex());
    return g_amd ? (size_t)g_amd->rt->available() : 0;   // the C++ runtime's VRAM pool
}

// Plan step A3, for BeagleGPUImpl (GPUInterface::GetDeviceLost): true once nothing this instance computes can be trusted
bool AmdDeviceLost(GPUInterface* self) {
    std::lock_guard<std::recursive_timed_mutex> lk(amd_mutex());
    AMDInstance* in = (AMDInstance*)self->amdInstance;
    return !in || amd_failed(*in);
}

// Plan step M1, for BeagleGPUImpl (GPUInterface::GetOutOfMemory): ... because the GPU's memory did not suffice
bool AmdOutOfMemory(GPUInterface* self) {
    std::lock_guard<std::recursive_timed_mutex> lk(amd_mutex());
    AMDInstance* in = (AMDInstance*)self->amdInstance;
    return in && in->oom;
}

// Plan step A3, for GPUInterface::Initialize before it connects: a crash guard this process left holding the card keeps
// TinyGPU.app serving its connection, so a new one would wait forever. True (and said) then.
bool AmdGpuHeld() {
    std::lock_guard<std::recursive_timed_mutex> lk(amd_mutex());
    if (!g_amdHeld) return false;
    fprintf(stderr, "TinyGPU/AMD: the crash guard (pid %d) holds the eGPU, since an earlier instance of this process could not see its "
            "queues off: no instance can use it until the eGPU is unplugged and the process restarts\n", (int)g_amdHeld);
    return true;
}

// Releases this instance; the card stays for later instances until exit (amdAtExit; plan step A5)
void AmdFini(GPUInterface* self) {
    std::lock_guard<std::recursive_timed_mutex> lk(amd_mutex());
    AMDInstance* in = (AMDInstance*)self->amdInstance;
    if (!in) return;
    self->amdInstance = nullptr;
    amdFlushLaunchQueue(*in);  // don't silently drop queued-but-unsent launches
    delete in;
}

void AmdLaunchKernelImpl(GPUInterface* self, GPUFunction fn, Dim3Int block, Dim3Int grid,
                          int nPtr, int nTotal, GPUPtr* ptrs, unsigned int* ints) {
    std::lock_guard<std::recursive_timed_mutex> lk(amd_mutex());
    AMDInstance* in = (AMDInstance*)self->amdInstance;
    if (!in || amd_failed(*in) || !fn) return;
    const AMDKernelHandle* ke = (const AMDKernelHandle*)fn;
    int nInt = nTotal - nPtr;
    if (amd_profile_enabled()) g_amdKernelLaunches[ke->name]++;

    // Queued, not submitted (STATUS.md AMD §26): amdFlushLaunchQueue() submits the backlog before this instance's next h2d,
    // d2h, sync or release, so ordering relative to its memory operations is preserved.
    AMDPendingLaunch pl;
    pl.kernel = ke;
    pl.grid[0] = grid.x; pl.grid[1] = grid.y; pl.grid[2] = grid.z;
    pl.block[0] = block.x; pl.block[1] = block.y; pl.block[2] = block.z;
    pl.ptrs.assign(ptrs, ptrs + nPtr);
    pl.ints.assign(ints, ints + nInt);
    in->pending.push_back(std::move(pl));
}

} // namespace tinygpu_device

#endif // FW_TINYGPU
