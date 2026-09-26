/*
 * GPUInterfaceTinyGPUHybridNV.cpp
 *
 * BEAGLE NV hybrid backend, daemon architecture (STATUS.md §73/§75).
 *
 * GPUInterfaceTinyGPUHybrid.cpp's original NV path hand-rolls GPFIFO/QMD
 * command-queue construction and dispatch directly in C++; that is now the
 * legacy path (BEAGLE_NV_USE_DAEMON=0). This daemon path was adopted when
 * that path's wrong answers were blamed on hand-rolled dispatch. The
 * kernelMatrixMulADB wrong answers were in fact unpopulated cbuf0
 * launch-dims words, which neither path wrote; both paths now fill them
 * (TODO.md Phase 140; STATUS.md §203 lists the legacy-path residues that
 * stay open).
 *
 * This file is a thin RPC client, structurally identical to
 * GPUInterfaceTinyGPUHybridAMD.cpp: a live Python daemon
 * (nv_dispatch_daemon.py) stays resident and does EVERY GPU operation --
 * boot, compile, alloc, memcpy, launch, sync -- via tinygrad's real
 * NVDevice/NVProgram/HCQProgram.__call__ code. This file sends
 * length-prefixed JSON commands over a dedicated socketpair (not the
 * TinyGPU socket -- NVDevice("NV:0") makes its own connection internally,
 * the same way STATUS.md §74's hardware-verified boot did) and reads
 * back replies.
 *
 * Compile backend unchanged: BEAGLE's existing PTX kernel source
 * (kernelResource->kernelCode, from kernels/BeagleTinyGPU_kernels.h) via
 * nv_compile_helper.py's compile_ptx(), reused by the daemon directly, not
 * re-invoked as a per-kernel subprocess the way the hand-rolled path did.
 *
 * BEAGLE_NV_CPP_DISPATCH=1 (TODO.md "Runtime roadmap", Step 3): the daemon
 * only boots, compiles, prepares programs and allocates. After a "handoff"
 * this file encodes launches and copies itself and submits them over the
 * plugin's own TinyGPU.app connection (see "C++ dispatch" below).
 *
 * BEAGLE_NV_USE_DAEMON=0, the C++ runtime (the revived legacy path), and the
 * default on Ada (AD10x) GPUs when neither variable is set (TODO.md plan
 * decision 16; BEAGLE_NV_USE_DAEMON=1 selects the daemon path): the
 * daemon only boots, and hands over right away. This file then also loads
 * the programs (TinyGPUHybridNVProgram.h, a port of tinygrad's program
 * loader) from the cubin built for this GPU and linked into the plugin
 * (TinyGPUHybridNVCubins.h; nothing is compiled at run time, so
 * BEAGLE_NV_USE_NVJITLINK and BEAGLE_NV_PTXAS_KERNELS apply only to the other
 * modes), and allocates from a VRAM pool the daemon mapped, so Python does
 * nothing after boot until "fini" (see "C++ runtime" below). Every instance
 * in the process shares that one boot, which lasts until exit (TODO.md plan
 * step P5).
 */

#ifdef FW_TINYGPU

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <sys/wait.h>
#include <unistd.h>

#include "libhmsbeagle/beagle.h"
#include "libhmsbeagle/GPU/GPUImplDefs.h"
#include "libhmsbeagle/GPU/GPUImplHelper.h"
#include "libhmsbeagle/GPU/GPUInterface.h"
#include "libhmsbeagle/GPU/KernelResource.h"
#include "libhmsbeagle/GPU/GPUInterfaceTinyGPUHybridNV.h"
#include "libhmsbeagle/GPU/TinyGPUTransport.h"
#include "libhmsbeagle/GPU/TinyGPUHybridNVGsp.h"
#include "libhmsbeagle/GPU/TinyGPUHybridNVDispatch.h"
#include "libhmsbeagle/GPU/TinyGPUHybridNVProgram.h"
#include "libhmsbeagle/GPU/TinyGPUHybridNVCubins.h"

// The embedded cubins were compiled from the PTX this plugin embeds (TODO.md plan step C1): make_tinygpu_cubins.sh copies
// the stamp of the kernels header whose PTX it compiled.
static_assert(std::string_view(TINYGPU_CUBINS_KERNELS_STAMP) == TINYGPU_KERNELS_STAMP,
              "kernels/TinyGPUNVCubins.h is from another BeagleTinyGPU_kernels.h: rebuild the TinyGPUCubins target");

namespace tinygpu_device {

// ── small utilities (file I/O + minimal JSON; same style as the AMD file) ──

static bool nv_write_file(const char* path, const void* buf, size_t sz) {
    FILE* f = fopen(path, "wb"); if (!f) return false;
    fwrite(buf, 1, sz, f); fclose(f); return true;
}
static uint64_t nv_json_u64(const std::string& js, const char* key) {
    char needle[128]; snprintf(needle, sizeof(needle), "\"%s\":", key);
    auto p = js.find(needle);
    if (p == std::string::npos) return 0;
    p += strlen(needle);
    while (p < js.size() && (js[p]==' '||js[p]=='\n')) ++p;
    return (uint64_t)strtoull(js.c_str() + p, nullptr, 10);
}
static bool nv_json_bool(const std::string& js, const char* key) {
    std::string needle = std::string("\"") + key + "\":";
    auto p = js.find(needle);
    if (p == std::string::npos) return false;
    p += needle.size();
    while (p < js.size() && js[p]==' ') ++p;
    return js.compare(p, 4, "true") == 0;
}
static bool nv_json_ok(const std::string& js) { return nv_json_bool(js, "ok"); }
static std::string nv_json_str(const std::string& js, const char* key) {
    char needle[128]; snprintf(needle, sizeof(needle), "\"%s\":", key);
    auto p = js.find(needle);
    if (p == std::string::npos) return "";
    p = js.find('"', p + strlen(needle));
    if (p == std::string::npos) return "";
    auto e = js.find('"', p + 1);
    return js.substr(p + 1, e - p - 1);
}

static std::string nv_resolve_python() {
    const char* p = getenv("BEAGLE_PYTHON");
    if (p && p[0]) return p;
    static const char* kCandidates[] = {
        "/opt/homebrew/bin/python3.13", "/opt/homebrew/bin/python3.12",
        "/opt/homebrew/bin/python3.11", "/opt/homebrew/bin/python3.10",
        "/usr/local/bin/python3.13", "/usr/local/bin/python3.12",
        "/usr/local/bin/python3.11", "/usr/local/bin/python3.10",
        nullptr
    };
    for (int i = 0; kCandidates[i]; ++i)
        if (access(kCandidates[i], X_OK) == 0) return kCandidates[i];
    return "python3";
}

// ── Command-socket I/O: JSON messages, each preceded by its byte length as a
// 4-byte little-endian uint32 (so a message is two reads, not one recv() per
// byte as with the newline framing the AMD pair still uses), with raw bytes
// immediately following for h2d (request) / d2h (reply) ────────────────────

static void nv_send_all(int fd, const void* buf, size_t n) {
    const uint8_t* p = (const uint8_t*)buf;
    while (n) { ssize_t r = ::send(fd, p, n, 0); if (r <= 0) return; p += r; n -= (size_t)r; }
}
static bool nv_recv_all(int fd, void* buf, size_t n) {
    uint8_t* p = (uint8_t*)buf;
    while (n) { ssize_t r = ::recv(fd, p, n, MSG_WAITALL); if (r <= 0) return false; p += r; n -= (size_t)r; }
    return true;
}
static void nv_send_msg(int fd, const std::string& json) {
    uint32_t n = (uint32_t)json.size();  // little-endian host (arm64/x86_64), as the daemon's "<I" expects
    std::string s((const char*)&n, 4);
    s += json;
    nv_send_all(fd, s.data(), s.size());
}
static std::string nv_recv_msg(int fd) {
    uint32_t n = 0;
    if (!nv_recv_all(fd, &n, 4)) return "";
    std::string s(n, '\0');
    if (n && !nv_recv_all(fd, &s[0], n)) return "";
    return s;
}

// ── Opt-in RPC round-trip profiling (BEAGLE_NV_PROFILE=1) ──────────────────
// The NV counterpart of BEAGLE_AMD_PROFILE, aggregated instead of printed
// per call so a many-evaluation benchmark (tinygpuhybridtest --reps) stays
// readable: NvFini prints count/mean/min/max per RPC. The daemon logs the
// matching breakdown of its own side to nv_dispatch_daemon.log, so the
// difference is wire + JSON overhead (TODO.md "Runtime roadmap", Step 1).
static bool nv_profile_enabled() {
    static const bool enabled = (getenv("BEAGLE_NV_PROFILE") != nullptr);
    return enabled;
}
struct NVProfileStat { long long n = 0; double total = 0, min = 1e300, max = 0; };
static std::map<std::string, NVProfileStat> g_nvProfile;
static long long g_nvProfileLaunches = 0;
static inline std::chrono::steady_clock::time_point nv_profile_start() {
    return std::chrono::steady_clock::now();
}
static void nv_profile_end(const char* label, std::chrono::steady_clock::time_point t0) {
    if (!nv_profile_enabled()) return;
    double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    NVProfileStat& s = g_nvProfile[label];
    s.n++; s.total += us;
    if (us < s.min) s.min = us;
    if (us > s.max) s.max = us;
}
static void nv_profile_report() {
    if (!nv_profile_enabled() || g_nvProfile.empty()) return;
    fprintf(stderr, "TinyGPU/NV: [profile] RPC round trips, C++ side:\n");
    for (const auto& kv : g_nvProfile) {
        const NVProfileStat& s = kv.second;
        fprintf(stderr, "TinyGPU/NV: [profile]   %-12s n=%7lld  mean=%9.1f us  min=%9.1f  max=%10.1f  total=%9.1f ms\n",
                kv.first.c_str(), s.n, s.total / s.n, s.min, s.max, s.total / 1000.0);
    }
    auto it = g_nvProfile.find("launch_batch");
    if (it != g_nvProfile.end())
        fprintf(stderr, "TinyGPU/NV: [profile]   %lld launches in %lld batches (%.2f per batch)\n",
                g_nvProfileLaunches, it->second.n, (double)g_nvProfileLaunches / it->second.n);
}

// ── State ────────────────────────────────────────────────────────────────────

struct NVHybridState {
    int cmd_sock;
    pid_t daemon_pid;
    std::string arch;   // the boot reply's: later instances pick their cubins for it (plan step P5)
    pid_t owner_pid;    // the process that booted; a child forked from it shares its connections and must never tear down
};

struct NVKernelHandle {
    std::string name;
    const NVDKernel* tmpl = nullptr;  // C++ dispatch: this kernel's handoff template
    long long launches = 0;           // launches BEAGLE requested, for BEAGLE_NV_PROFILE's report (plan step D1)
};

// The process's one GPU, which every instance shares (TODO.md plan step P5): the daemon here, and in g_nvd below the
// TinyGPU.app connection, rings, timeline, local memory and VRAM pool. What is an instance's own is its NVInstance.
static NVHybridState* g_nv = nullptr;
static std::map<std::string, long long> g_nvKernelLaunches;   // released instances' launches per kernel, for the profile report

// Plan step P5: one lock around everything the instances share, so a frame to TinyGPU.app never interleaves with another
// thread's and the rings, timeline and pool stay consistent. Recursive, because nv_safe_exit takes it on paths that
// already hold it; timed, for nvAtExit. Never destroyed: a GPUInterface may be destroyed after the static destructors ran.
static std::recursive_timed_mutex& nv_mutex() {
    static std::recursive_timed_mutex* m = new std::recursive_timed_mutex;
    return *m;
}

// The state page (TODO.md plan step P3; nvdStatePage below): four 64-bit words shared with the daemon, which reads them
// only once this side can no longer write (at "fini", or at EOF after this process died). Plan step C5 added the GSP
// command queue's sequence number after this side's last RPC, and the teardown phase.
enum { kNVDStatePhase, kNVDStateInFlight, kNVDStateLastSubmitted, kNVDStateSeq, kNVDStateWords };
static const uint64_t kNVDPhaseDispatch = 1;  // this side owns both GPFIFOs; the daemon holds on any other phase
static const uint64_t kNVDPhaseTeardown = 2;  // this side is unloading the GPU itself (plan step C5): the daemon holds at EOF

// TODO.md plan decision 11: BEAGLE_NV_CPP_LEVEL says how far the C++ runtime goes (removed in C12). teardown (plan step C5),
// the default since it passed on the RTX 4060 (STATUS.md R33): this side unloads the GPU at fini, on tinygrad's ported GSP
// queue and falcon primitives, and the daemon only exits; runtime: the daemon unloads it, as before C5.
enum NVCppLevel { kNVLevelRuntime, kNVLevelTeardown };
static NVCppLevel nv_cpp_level() {
    static const NVCppLevel level = [] {
        const char* v = getenv("BEAGLE_NV_CPP_LEVEL");
        if (!v || !v[0] || strcmp(v, "teardown") == 0) return kNVLevelTeardown;
        if (strcmp(v, "runtime") == 0) return kNVLevelRuntime;
        fprintf(stderr, "TinyGPU/NV: BEAGLE_NV_CPP_LEVEL=%s is not a level this build has (runtime, teardown); using teardown\n", v);
        return kNVLevelTeardown;
    }();
    return level;
}

// cmd_teardown_export's reply (plan step C5): what this side's GPU teardown needs
struct NVDTeardown {
    bool ready = false;
    uint8_t* queues = nullptr;   // the GSP message queues: the daemon's TinyGPU.app sysmem, shared
    uint64_t queues_size = 0, cmdq_off = 0, statq_off = 0, queue_size = 0, libos_args_sysmem = 0;
    uint32_t seq = 0, chip_id = 0;
    bool level0 = false;         // BEAGLE_NV_UNLOAD_LEVEL=0, as the daemon read it
    NVTeardownImages images;     // nv_init_helper's FWSEC-SB and Booter Unload, if the teardown is on
};

// C++ dispatch state (see "C++ dispatch" below); null on the daemon path.
struct NVDispatchState {
    NVDHandoff h;
    int tg_sock = -1;            // the plugin's TinyGPU.app connection, shared with the daemon
    void* maps[4] = {};          // host mappings of h.cmdq, h.kargs, h.staging, h.signal
    uint8_t *cmdq = nullptr, *kargs = nullptr, *staging = nullptr;
    uint64_t* signal = nullptr;  // timeline semaphore
    uint64_t* state = nullptr;   // the state page (kNVDState* words)
    uint64_t cmdq_pos = 0, kargs_pos = 0, staging_pos = 0;
    uint64_t timeline = 1;       // value the next submission signals; every earlier value is submitted
    uint64_t pending = 0;        // submissions since the GPU was last seen idle
    bool runtime = false;        // the C++ runtime: programs loaded here, allocations from rt.pool
    NVDRuntime rt;
    uint64_t pool_pos = 0;       // rt.pool's fill level
    uint32_t slm_per_thread = 0; // dev.slm_per_thread: the local memory set up so far serves this much per thread
    NVDTeardown td;              // BEAGLE_NV_CPP_LEVEL=teardown (plan step C5)
};
static NVDispatchState* g_nvd = nullptr;

// The C++ runtime (see the top of this file): BEAGLE_NV_USE_DAEMON=0, or, with neither BEAGLE_NV_USE_DAEMON nor
// BEAGLE_NV_CPP_DISPATCH set, the default on Ada (TODO.md plan decision 16: AD10x, whose PCI device IDs are 0x26xx-0x28xx
// in tinygrad's PCIIface family list, ops_nv.py:559). Blackwell keeps the daemon until plan step B2, and Ampere never ran.
// First asked in NvSetDevice, after Initialize read the device ID.
static bool nv_cpp_runtime() {
    static const bool on = [] {
        const char* v = getenv("BEAGLE_NV_USE_DAEMON");
        if (v) return strcmp(v, "0") == 0;
        if (getenv("BEAGLE_NV_CPP_DISPATCH")) return false;
        uint16_t family = tg_pci_device_id() & 0xff00;
        bool ada = family == 0x2600 || family == 0x2700 || family == 0x2800;
        if (ada) fprintf(stderr, "TinyGPU/NV: the C++ runtime, the default on this GPU (BEAGLE_NV_USE_DAEMON=1 selects the daemon)\n");
        return ada;
    }();
    return on;
}

static bool nv_cpp_dispatch() {
    static const bool on = [] { const char* v = getenv("BEAGLE_NV_CPP_DISPATCH"); return (v && strcmp(v, "0") != 0) || nv_cpp_runtime(); }();
    return on;
}

// ── Launch batching (mirrors AMD's, STATUS.md AMD §26 -- built in from the
// start here rather than added later, since that overhead finding already
// generalizes: any RPC-per-launch design pays the same per-call socket+JSON
// cost regardless of vendor). Flushed before every h2d/d2h/sync/fini so
// ordering relative to memory operations is preserved -- see
// nv_dispatch_daemon.py's cmd_launch_batch comment for why that's sufficient
// without extra synchronization on either side. ─────────────────────────────
struct NVPendingLaunch {
    std::string kernel;
    const NVDKernel* tmpl;
    int grid[3];
    int block[3];
    std::vector<unsigned long long> ptrs;
    std::vector<unsigned int> ints;
};

// Plan step P5: what belongs to one BEAGLE instance (GPUInterface::nvGspState); the rest is the shared GPU above.
struct NVInstance {
    std::map<std::string, NVKernelHandle*> kernels;  // GetFunction's handles
    std::map<std::string, NVDKernel> templates;      // the C++ runtime: this instance's programs, at its own lib_va
    std::vector<NVPendingLaunch> pending;            // launches queued since this instance last flushed
};

static void nvdFlushLaunches(std::vector<NVPendingLaunch>& pending);

static void nvFlushLaunchQueue(NVInstance& in) {
    if (!g_nv || in.pending.empty()) return;
    if (g_nvd) { nvdFlushLaunches(in.pending); return; }
    auto t0 = nv_profile_start();
    std::string cmd = "{\"cmd\":\"launch_batch\",\"launches\":[";
    for (size_t li = 0; li < in.pending.size(); ++li) {
        const NVPendingLaunch& pl = in.pending[li];
        if (li) cmd += ",";
        cmd += "{\"kernel\":\"" + pl.kernel + "\",\"grid\":[" +
            std::to_string(pl.grid[0]) + "," + std::to_string(pl.grid[1]) + "," + std::to_string(pl.grid[2]) + "],\"block\":[" +
            std::to_string(pl.block[0]) + "," + std::to_string(pl.block[1]) + "," + std::to_string(pl.block[2]) + "],\"ptrs\":[";
        for (size_t i = 0; i < pl.ptrs.size(); ++i) { if (i) cmd += ","; cmd += std::to_string(pl.ptrs[i]); }
        cmd += "],\"ints\":[";
        for (size_t i = 0; i < pl.ints.size(); ++i) { if (i) cmd += ","; cmd += std::to_string(pl.ints[i]); }
        cmd += "]}";
    }
    cmd += "]}";
    size_t n = in.pending.size();
    in.pending.clear();

    nv_send_msg(g_nv->cmd_sock, cmd);
    std::string resp = nv_recv_msg(g_nv->cmd_sock);
    nv_profile_end("launch_batch", t0);
    g_nvProfileLaunches += (long long)n;
    if (resp.empty() || !nv_json_ok(resp))
        fprintf(stderr, "TinyGPU/NV: launch_batch(%zu kernels) failed: %s\n", n, resp.c_str());
}

// What the daemon reports after it unloaded the GPU (fini, or a boot that failed after GSP-RM started): the unload,
// NVIDIA's teardown if it ran (on unless BEAGLE_NV_TEARDOWN=0; plan steps P2, P3) or else whether the next boot needs a
// power cycle, and whether the daemon keeps its copy of the TinyGPU.app connection open because the GPU may still use
// memory behind it (the unload was not confirmed, or a frame to TinyGPU.app was cut mid-send). Returns that last one.
static bool nv_report_unload(const std::string& resp) {
    uint64_t mbx = 0, cpuctl = 0, wlo = 0, whi = 0, pid = 0;
    bool unload_ok = nv_json_bool(resp, "unload_ok");
    if (nvd_json_u64(resp, "mailbox0", mbx) && nvd_json_u64(resp, "riscv_cpuctl", cpuctl) &&
        nvd_json_u64(resp, "wpr2_lo", wlo) && nvd_json_u64(resp, "wpr2_hi", whi))
        fprintf(stderr, "TinyGPU/NV: GPU teardown: unload %s (GSP MAILBOX0=0x%08llx, RISCV_CPUCTL=0x%08llx, WPR2_LO=0x%08llx, "
                "WPR2_HI=0x%08llx)\n", unload_ok ? "confirmed" : "NOT confirmed", (unsigned long long)mbx,
                (unsigned long long)cpuctl, (unsigned long long)wlo, (unsigned long long)whi);
    if (resp.find("\"teardown_ok\":") != std::string::npos)
        fprintf(stderr, "TinyGPU/NV: teardown: %s; %s\n", nv_json_str(resp, "result").c_str(),
                nv_json_bool(resp, "teardown_ok") ? "WPR2 is down, the next boot needs no power cycle"
                                                  : "power-cycle the eGPU before the next boot");
    else if (resp.find("\"unload_ok\":") != std::string::npos && (!unload_ok || whi != 0))
        // no teardown result: BEAGLE_NV_TEARDOWN=0, a hung fini, a failed boot, an unconfirmed unload (Blackwell: plan step B1)
        fprintf(stderr, "TinyGPU/NV: no teardown result (%s); power-cycle the eGPU before the next boot\n",
                unload_ok ? "WPR2 is still up" : "the GPU did not confirm its unload");
    bool hold = nv_json_bool(resp, "hold") && nvd_json_u64(resp, "pid", pid);
    if (hold)
        fprintf(stderr, "TinyGPU/NV: the daemon (pid %llu) keeps the TinyGPU.app connection open because the GPU may still use "
                "memory behind it. Unplug the eGPU first, then kill %llu.\n", (unsigned long long)pid, (unsigned long long)pid);
    return hold;
}

// The daemon died before replying to fini: nothing says whether the GPU was torn down.
static void nv_report_no_fini_reply() {
    fprintf(stderr, "TinyGPU/NV: no fini reply from the daemon (it exited during the teardown?); the GPU state is unknown: "
            "power-cycle the eGPU before the next boot\n");
}

// hung: the GPU stopped making progress, so the daemon sends only the GSP unload RPC (no synchronize, no teardown).
// The lock keeps every other thread off the TinyGPU.app connection the daemon uses for that (plan step P5). A child
// forked after the boot sends nothing: the GPU is its parent's.
[[noreturn]] static void nv_safe_exit(int code, bool hung = false) {
    std::lock_guard<std::recursive_timed_mutex> lk(nv_mutex());
    fflush(stderr);
    if (g_nv && g_nv->owner_pid == getpid()) {
        bool hold = false;
        if (g_nv->cmd_sock >= 0) {
            nv_send_msg(g_nv->cmd_sock, hung ? "{\"cmd\":\"fini\",\"hung\":true}" : "{\"cmd\":\"fini\"}");
            std::string resp = nv_recv_msg(g_nv->cmd_sock);
            if (resp.empty()) nv_report_no_fini_reply();
            hold = nv_report_unload(resp);
        }
        if (g_nv->daemon_pid > 0 && !hold) {
            for (int i = 0; i < 100; ++i) {
                int st = 0;
                if (waitpid(g_nv->daemon_pid, &st, WNOHANG) > 0) break;
                usleep(100000);
            }
        }
        if (g_nv->cmd_sock >= 0) close(g_nv->cmd_sock);
    }
    _exit(code);
}

// ── C++ dispatch (BEAGLE_NV_CPP_DISPATCH=1; TODO.md "Runtime roadmap",
// Step 3). After the daemon's handoff this file encodes launches and copies
// itself (TinyGPUHybridNVDispatch.h, golden-tested byte for byte against
// hcq1's own encoders) into four shared sysmem buffers, and submits them by
// writing the GPFIFO entry, GPPut and doorbell as posted MMIO writes on the
// plugin's TinyGPU.app connection, which the daemon inherited for boot and
// only uses again inside its own (synchronous) commands. Completion is a
// timeline semaphore in shared memory, polled locally: a batch costs three
// posted socket writes and no round trips. As in hcq1, every submission
// first waits for the one before it. ────────────────────────────────────────

// HCQSignal.wait on the timeline. A GPU making no progress for 30 s (hcq1's
// default timeout) is fatal, as it would be in the daemon.
static void nvd_wait(uint64_t value) {
    auto start = std::chrono::steady_clock::now();
    for (uint64_t spin = 0; __atomic_load_n(g_nvd->signal, __ATOMIC_ACQUIRE) < value; ++spin) {
        if (spin % 1024) continue;
        auto waited = std::chrono::steady_clock::now() - start;
        if (waited > std::chrono::seconds(30)) {
            fprintf(stderr, "TinyGPU/NV: timeline wait timed out (want %llu, have %llu); GPU hung?\n",
                    (unsigned long long)value, (unsigned long long)__atomic_load_n(g_nvd->signal, __ATOMIC_ACQUIRE));
            nv_safe_exit(1, true);
        }
        if (waited > std::chrono::milliseconds(2)) usleep(20);
    }
}

static void nvd_idle(uint64_t behind = 1) {   // 2 from nvd_submit: its caller already took this frame's value
    nvd_wait(g_nvd->timeline - behind);
    g_nvd->pending = 0;
}

// Bump allocation in a shared ring. Wrapping around first waits until the GPU
// is done with everything submitted, so no live region is overwritten.
static uint64_t nvd_alloc(uint64_t& pos, uint64_t size, uint64_t need, uint64_t align, uint64_t behind = 1) {
    uint64_t p = (pos + align - 1) & ~(align - 1);
    if (p + need > size) { nvd_idle(behind); p = 0; }
    pos = p + need;
    return p;
}

// NVCommandQueue._submit_to_gpfifo: the pushbuffer goes into the shared ring;
// the GPFIFO entry, GPPut and doorbell go out as three posted writes in one send.
static void nvd_submit(NVDFifo& f, const std::vector<uint32_t>& pb) {
    // the caller already took this frame's timeline value, so both waits stop at the frame before it
    if (g_nvd->pending >= f.entries / 2) nvd_idle(2);  // never let the GPFIFO ring lap the GPU
    uint64_t off = nvd_alloc(g_nvd->cmdq_pos, g_nvd->h.cmdq.size, pb.size() * 4, 16, 2);
    memcpy(g_nvd->cmdq + off, pb.data(), pb.size() * 4);
    uint64_t entry = nvd_gpfifo_entry(g_nvd->h.cmdq.va + off, (uint32_t)pb.size());
    uint32_t gpput = (uint32_t)((f.put + 1) % f.entries);
    const TGWrite frame[3] = {{f.ring_bar, f.ring_off + (f.put % f.entries) * 8, &entry, 8},
                              {f.gpput_bar, f.gpput_off, &gpput, 4},
                              {g_nvd->h.db_bar, g_nvd->h.db_off, &f.token, 4}};
    // State page: in flight, then the value this frame's work signals (every caller took it from the timeline already),
    // then "not in flight" once the whole frame is out. A frame cut mid-send ends this process with the flag still set,
    // so the daemon holds and sends TinyGPU.app nothing more (it would read those bytes as the rest of this frame).
    __atomic_store_n(&g_nvd->state[kNVDStateInFlight], 1, __ATOMIC_RELEASE);
    __atomic_store_n(&g_nvd->state[kNVDStateLastSubmitted], g_nvd->timeline - 1, __ATOMIC_RELEASE);
    std::string err;
    if (!tg_transport().bulk_write_frame(frame, 3, err)) {
        fprintf(stderr, "TinyGPU/NV: TinyGPU.app write %s: %s\n", tg_transport().lost() ? "cut mid-frame" : "refused", err.c_str());
        if (!tg_transport().lost()) __atomic_store_n(&g_nvd->state[kNVDStateInFlight], 0, __ATOMIC_RELEASE);   // nothing went out
        nv_safe_exit(1);
    }
    __atomic_store_n(&g_nvd->state[kNVDStateInFlight], 0, __ATOMIC_RELEASE);
    ++f.put;
    ++g_nvd->pending;
}

// BeagleNVProgram.check_launch (NVProgram.__call__'s launch checks). The
// daemon reports a failed launch and carries on; so does this.
static bool nvd_check_launch(const NVDKernel& k, const int grid[3], const int block[3]) {
    long threads = (long)block[0] * block[1] * block[2];
    if (threads <= 1024 && threads <= (long)k.max_threads && grid[1] <= 65535 && grid[2] <= 65535 &&
        block[0] <= 1024 && block[1] <= 1024 && block[2] <= 64)
        return true;
    fprintf(stderr, "TinyGPU/NV: %s: invalid launch grid=(%d,%d,%d) block=(%d,%d,%d), not launched\n",
            k.name.c_str(), grid[0], grid[1], grid[2], block[0], block[1], block[2]);
    return false;
}

// The daemon's chained cmd_launch_batch, encoded here: one timeline wait and
// shader-cache invalidate, each launch's QMD chained onto the previous one,
// and the last QMD signalling the timeline.
static void nvdFlushLaunches(std::vector<NVPendingLaunch>& pending) {
    auto t0 = nv_profile_start();
    NVDHandoff& h = g_nvd->h;
    std::vector<uint32_t> pb;
    uint64_t value = g_nvd->timeline;
    nvd_push_wait(pb, h, h.signal.va, value - 1);
    nvd_push_invalidate(pb, h);
    uint8_t* prev = nullptr;
    long long launched = 0;
    for (const NVPendingLaunch& pl : pending) {
        if (!pl.tmpl) {
            fprintf(stderr, "TinyGPU/NV: %s: no handoff template, not launched\n", pl.kernel.c_str());
            continue;
        }
        const NVDKernel& k = *pl.tmpl;
        if (!nvd_check_launch(k, pl.grid, pl.block)) continue;
        uint32_t grid[3] = { (uint32_t)pl.grid[0], (uint32_t)pl.grid[1], (uint32_t)pl.grid[2] };
        uint32_t block[3] = { (uint32_t)pl.block[0], (uint32_t)pl.block[1], (uint32_t)pl.block[2] };
        uint64_t off = nvd_alloc(g_nvd->kargs_pos, h.kargs.size, k.slot_size, 256);
        uint8_t* qmd = nvd_encode_launch(h, k, g_nvd->kargs + off, h.kargs.va + off, grid, block,
                                         reinterpret_cast<const uint64_t*>(pl.ptrs.data()), (int)pl.ptrs.size(),
                                         pl.ints.data(), (int)pl.ints.size());
        uint64_t qmd_va = h.kargs.va + off + k.qmd_off;
        if (!prev) nvd_push_pcas(pb, h, qmd_va); else nvd_chain(h, prev, qmd_va);
        prev = qmd;
        ++launched;
    }
    pending.clear();
    if (!prev) return;
    nvd_qmd_release(h, prev, h.signal.va, value);
    g_nvd->timeline = value + 1;
    nvd_submit(h.compute, pb);
    nv_profile_end("launch_batch", t0);
    g_nvProfileLaunches += launched;
}

// HCQAllocator._copyin through the staging ring: per chunk, the copy engine
// waits for everything before it, copies, and signals the timeline.
static void nvdCopyIn(uint64_t dst, const void* src, size_t sz) {
    auto t0 = nv_profile_start();
    NVDHandoff& h = g_nvd->h;
    const uint64_t chunk = h.staging.size / 4;
    for (uint64_t i = 0; i < sz; i += chunk) {
        uint64_t n = std::min<uint64_t>(chunk, sz - i);
        uint64_t off = nvd_alloc(g_nvd->staging_pos, h.staging.size, n, 256);
        memcpy(g_nvd->staging + off, (const uint8_t*)src + i, n);
        uint64_t value = g_nvd->timeline++;
        std::vector<uint32_t> pb;
        nvd_push_wait(pb, h, h.signal.va, value - 1);
        nvd_push_copy(pb, h, dst + i, h.staging.va + off, n);
        nvd_push_dma_signal(pb, h, h.signal.va, value);
        nvd_submit(h.copy, pb);
    }
    nv_profile_end("h2d", t0);
}

// HCQAllocator._copyout: per chunk, the copy engine copies into staging after
// everything before it, and the host reads it once the timeline says so.
static void nvdCopyOut(void* dst, uint64_t src, size_t sz) {
    auto t0 = nv_profile_start();
    NVDHandoff& h = g_nvd->h;
    for (uint64_t i = 0; i < sz; i += h.staging.size) {
        uint64_t n = std::min<uint64_t>(h.staging.size, sz - i);
        uint64_t off = nvd_alloc(g_nvd->staging_pos, h.staging.size, n, 256);
        uint64_t value = g_nvd->timeline++;
        std::vector<uint32_t> pb;
        nvd_push_wait(pb, h, h.signal.va, value - 1);
        nvd_push_copy(pb, h, h.staging.va + off, src + i, n);
        nvd_push_dma_signal(pb, h, h.signal.va, value);
        nvd_submit(h.copy, pb);
        nvd_wait(value);
        memcpy((uint8_t*)dst + i, g_nvd->staging + off, n);
    }
    nv_profile_end("d2h", t0);
}

// The daemon's socket.send_fds: one byte carrying the fds as SCM_RIGHTS.
static bool nv_recv_fds(int sock, int* fds, int n) {
    char byte;
    struct iovec iov = { &byte, 1 };
    std::vector<char> cbuf(CMSG_SPACE(sizeof(int) * n));
    struct msghdr msg{};
    msg.msg_iov = &iov; msg.msg_iovlen = 1;
    msg.msg_control = cbuf.data(); msg.msg_controllen = (socklen_t)cbuf.size();
    if (recvmsg(sock, &msg, 0) != 1) return false;
    struct cmsghdr* c = CMSG_FIRSTHDR(&msg);
    if (!c || c->cmsg_level != SOL_SOCKET || c->cmsg_type != SCM_RIGHTS || c->cmsg_len != CMSG_LEN(sizeof(int) * n)) return false;
    memcpy(fds, CMSG_DATA(c), sizeof(int) * n);
    return true;
}

// The daemon's socket.recv_fds in cmd_state_page: one byte carrying fd as SCM_RIGHTS (nv_recv_fds the other way).
static bool nv_send_fd(int sock, int fd) {
    char byte = 'S';
    struct iovec iov = { &byte, 1 };
    std::vector<char> cbuf(CMSG_SPACE(sizeof(int)));
    struct msghdr msg{};
    msg.msg_iov = &iov; msg.msg_iovlen = 1;
    msg.msg_control = cbuf.data(); msg.msg_controllen = (socklen_t)cbuf.size();
    struct cmsghdr* c = CMSG_FIRSTHDR(&msg);
    c->cmsg_level = SOL_SOCKET; c->cmsg_type = SCM_RIGHTS; c->cmsg_len = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(c), &fd, sizeof(int));
    return sendmsg(sock, &msg, 0) == 1;
}

static void nvd_unmap(NVDispatchState* d) {
    const NVDBuffer* bufs[4] = { &d->h.cmdq, &d->h.kargs, &d->h.staging, &d->h.signal };
    for (int i = 0; i < 4; ++i)
        if (d->maps[i]) { munmap(d->maps[i], bufs[i]->size); d->maps[i] = nullptr; }
    if (d->state) { munmap(d->state, kNVDStateWords * 8); d->state = nullptr; }
    if (d->td.queues) { munmap(d->td.queues, d->td.queues_size); d->td.queues = nullptr; }
}

// cmd_handoff: the daemon's reply (flat JSON), the kernel blob, then the fds
// of the four shared buffers in NVDHandoff's order. For the C++ runtime the
// handoff carries no programs (this side loads its embedded cubin). The
// daemon stops using the queues once it replies, so a failure here is fatal.
static NVDispatchState* nvDispatchHandoff(int cmd_sock, int tg_sock, bool runtime) {
    auto t0 = nv_profile_start();
    std::string req = "{\"cmd\":\"handoff\"}";
    if (runtime) {  // BEAGLE_NV_DATA_MB sizes the VRAM pool; the daemon's default is half the VRAM
        const char* mb = getenv("BEAGLE_NV_DATA_MB");
        req = "{\"cmd\":\"handoff\",\"programs\":false,\"pool_size\":" +
              std::to_string(mb ? strtoull(mb, nullptr, 10) << 20 : 0) + "}";
    }
    nv_send_msg(cmd_sock, req);
    std::string js = nv_recv_msg(cmd_sock);
    uint64_t blob_size = 0, nfds = 0;
    NVDRuntime rt;
    std::string err = runtime ? nvd_parse_runtime(js, rt) : "";
    if (js.empty() || !nv_json_ok(js) || !nvd_json_u64(js, "blob_size", blob_size) || !nvd_json_u64(js, "nfds", nfds) || nfds != 4 ||
        !err.empty()) {
        fprintf(stderr, "TinyGPU/NV: handoff failed: %s%s%s\n", js.c_str(), err.empty() ? "" : "; ", err.c_str());
        return nullptr;
    }
    std::vector<uint8_t> blob(blob_size);
    int fds[4] = { -1, -1, -1, -1 };
    if (!nv_recv_all(cmd_sock, blob.data(), blob.size()) || !nv_recv_fds(cmd_sock, fds, 4)) {
        fprintf(stderr, "TinyGPU/NV: handoff: daemon connection lost\n");
        return nullptr;
    }
    NVDispatchState* d = new NVDispatchState;
    d->runtime = runtime;
    d->rt = rt;
    err = nvd_parse_handoff(js, blob, d->h);
    // the sizes of the BARs the daemon mapped in this session: every posted write from here is checked against them
    // (plan step C3; a MAP_BAR of our own would change the stream)
    for (uint32_t bar : {d->h.compute.ring_bar, d->h.compute.gpput_bar, d->h.copy.ring_bar, d->h.copy.gpput_bar, d->h.db_bar}) {
        uint64_t size = 0;
        if (err.empty() && !nvd_json_u64(js, ("bar" + std::to_string(bar) + "_size").c_str(), size))
            err = "the handoff has no size for BAR " + std::to_string(bar);
        if (err.empty()) tg_transport().seed_bar(bar, size);
    }
    const NVDBuffer* bufs[4] = { &d->h.cmdq, &d->h.kargs, &d->h.staging, &d->h.signal };
    for (int i = 0; i < 4; ++i) {
        if (err.empty()) {
            d->maps[i] = mmap(nullptr, bufs[i]->size, PROT_READ | PROT_WRITE, MAP_SHARED, fds[i], 0);
            if (d->maps[i] == MAP_FAILED) { d->maps[i] = nullptr; err = std::string("mmap: ") + strerror(errno); }
        }
        close(fds[i]);
    }
    if (!err.empty()) {
        fprintf(stderr, "TinyGPU/NV: handoff: %s\n", err.c_str());
        nvd_unmap(d);
        delete d;
        return nullptr;
    }
    d->cmdq = (uint8_t*)d->maps[0];
    d->kargs = (uint8_t*)d->maps[1];
    d->staging = (uint8_t*)d->maps[2];
    d->signal = (uint64_t*)d->maps[3];
    d->tg_sock = tg_sock;
    nv_profile_end("handoff", t0);
    tg_transport().marker(TGM_HANDOFF, d->runtime);
    if (d->runtime)
        fprintf(stderr, "TinyGPU/NV: C++ runtime: handed over after boot (QMD v%u, VRAM pool %llu MiB)\n",
                d->h.qmd_ver, (unsigned long long)(d->rt.pool.size >> 20));
    else
        fprintf(stderr, "TinyGPU/NV: C++ dispatch: %zu kernels handed over (QMD v%u)\n", d->h.kernels.size(), d->h.qmd_ver);
    return d;
}

// TODO.md plan step P3: the state page, created here and passed to the daemon right after the handoff, before anything is
// submitted from here. A POSIX shm segment, not a TinyGPU allocation (TinyGPU.app's MAP_SYSMEM_FD sequence is unchanged),
// unlinked at once so only the two processes' descriptors reach it. nvd_submit records in it whether a frame is in flight
// and the timeline value its work signals; if this process dies without "fini", the daemon holds, or waits for that value
// before it tears the GPU down (the daemon's own synchronize does not cover work submitted from here).
static bool nvdStatePage(int cmd_sock, NVDispatchState& d) {
    char name[32];  // macOS PSHMNAMLEN is 31
    snprintf(name, sizeof(name), "/beagle-nv.%d", (int)getpid());
    shm_unlink(name);  // only a killed process with this pid could have left it
    int fd = shm_open(name, O_RDWR | O_CREAT | O_EXCL, 0600);
    if (fd < 0) { fprintf(stderr, "TinyGPU/NV: state page failed: shm_open: %s\n", strerror(errno)); return false; }
    shm_unlink(name);
    const size_t size = kNVDStateWords * 8;
    void* m = ftruncate(fd, size) == 0 ? mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0) : MAP_FAILED;
    if (m == MAP_FAILED) {
        fprintf(stderr, "TinyGPU/NV: state page failed: %s\n", strerror(errno));
        close(fd);
        return false;
    }
    uint64_t* st = (uint64_t*)m;  // zero-filled: nothing in flight, nothing submitted
    __atomic_store_n(&st[kNVDStatePhase], kNVDPhaseDispatch, __ATOMIC_RELEASE);
    nv_send_msg(cmd_sock, "{\"cmd\":\"state_page\"}");
    bool sent = nv_send_fd(cmd_sock, fd);
    close(fd);
    std::string js = sent ? nv_recv_msg(cmd_sock) : "";
    if (js.empty() || !nv_json_ok(js)) {
        fprintf(stderr, "TinyGPU/NV: state page failed: %s\n", js.c_str());
        munmap(m, size);
        return false;
    }
    d.state = st;
    return true;
}

// TODO.md plan step C5 (BEAGLE_NV_CPP_LEVEL=teardown): the daemon's cmd_teardown_export, right after the state page: the GSP
// message queues (their TinyGPU.app sysmem fd, which carries both queues' write and read pointers; the offsets; the
// command queue's sequence number), what the CPU sequencer and the falcon resets need, and nv_init_helper's two teardown
// images. Without it (a refusal, a COT boot, another chip) the daemon unloads the GPU at fini, as at level runtime.
static void nvdTeardownExport(int cmd_sock, NVDispatchState& d) {
    nv_send_msg(cmd_sock, "{\"cmd\":\"teardown_export\"}");
    std::string js = nv_recv_msg(cmd_sock);
    int fd = -1;
    std::string err = js.empty() ? "no reply" : !nv_json_ok(js) ? nv_json_str(js, "error") : !nv_recv_fds(cmd_sock, &fd, 1) ? "no queue fd" : "";
    uint64_t v[7] = {};
    const char* keys[7] = {"chip_id", "gsp_queues_size", "gsp_cmdq_off", "gsp_statq_off", "gsp_queue_size", "gsp_seq", "libos_args_sysmem"};
    for (int i = 0; i < 7 && err.empty(); ++i)
        if (!nvd_json_u64(js, keys[i], v[i])) err = std::string("the export has no ") + keys[i];
    if (err.empty() && nv_json_str(js, "fw_name") != "ad102") err = "the C++ teardown has Ada's register tables, not " + nv_json_str(js, "fw_name") + "'s";
    NVDTeardown& t = d.td;
    if (err.empty() && nv_json_bool(js, "teardown")) {
        NVTeardownImages& m = t.images;
        uint64_t w[14] = {};
        const char* ik[14] = {"sb_paddr", "sb_imem_pa", "sb_imem_va", "sb_imem_sz", "sb_dmem_pa", "sb_dmem_sz", "sb_pkc_off", "sb_engid",
                              "sb_ucodeid", "unload_paddr", "unload_data_off", "unload_data_sz", "unload_code_off", "unload_code_sz"};
        for (int i = 0; i < 14 && err.empty(); ++i)
            if (!nvd_json_u64(js, ik[i], w[i])) err = std::string("the export has no ") + ik[i];
        m.present = err.empty();
        m.sb_paddr = w[0]; m.sb_imem_pa = (uint32_t)w[1]; m.sb_imem_va = (uint32_t)w[2]; m.sb_imem_sz = (uint32_t)w[3];
        m.sb_dmem_pa = (uint32_t)w[4]; m.sb_dmem_sz = (uint32_t)w[5]; m.sb_pkc_off = (uint32_t)w[6]; m.sb_engid = (uint32_t)w[7];
        m.sb_ucodeid = (uint32_t)w[8]; m.unload_paddr = w[9]; m.unload_data_off = (uint32_t)w[10]; m.unload_data_sz = (uint32_t)w[11];
        m.unload_code_off = (uint32_t)w[12]; m.unload_code_sz = (uint32_t)w[13];
    }
    if (err.empty()) {
        void* q = mmap(nullptr, v[1], PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (q == MAP_FAILED) err = std::string("mmap of the GSP queues: ") + strerror(errno);
        else t.queues = (uint8_t*)q;
    }
    if (fd >= 0) close(fd);
    if (!err.empty()) {
        fprintf(stderr, "TinyGPU/NV: C++ teardown unavailable (%s): the daemon unloads the GPU at fini\n", err.c_str());
        t.images = NVTeardownImages();
        return;
    }
    t.chip_id = (uint32_t)v[0]; t.queues_size = v[1]; t.cmdq_off = v[2]; t.statq_off = v[3]; t.queue_size = v[4];
    t.seq = (uint32_t)v[5]; t.libos_args_sysmem = v[6];
    t.level0 = nv_json_bool(js, "unload_level0");
    t.ready = true;
    __atomic_store_n(&d.state[kNVDStateSeq], t.seq, __ATOMIC_RELEASE);
    fprintf(stderr, "TinyGPU/NV: C++ teardown: the GSP unload%s run here at fini (BEAGLE_NV_CPP_LEVEL=teardown)\n",
            t.images.present ? " and NVIDIA's teardown" : "");
}

// TODO.md plan step C5: the GPU teardown at fini from this side, on tinygrad's ported RPC queue and falcon primitives
// (TinyGPUHybridNVGsp.h, TinyGPUHybridNVFalcon.h), in NVDev.fini's order: the GSP unload (nv_init_helper's suspend wait
// included), then, only if the GSP confirmed it, NVIDIA's teardown. The state page says so first, so a daemon that sees
// this process die meanwhile holds instead of touching the GPU. Returns the fini request that reports it all to the
// daemon, which then exits, or holds if the GSP did not confirm its unload (hold rule).
static std::string nvdCppTeardown(NVDispatchState& d, double& secs, std::string& report) {
    auto t0 = std::chrono::steady_clock::now();
    NVDTeardown& t = d.td;
    __atomic_store_n(&d.state[kNVDStatePhase], kNVDPhaseTeardown, __ATOMIC_RELEASE);
    tg_log("C++ GPU teardown: the %s unload RPC (seq %u), the suspend wait%s", t.level0 ? "LEVEL_0" : "FAST_UNLOAD", t.seq,
           t.images.present ? ", then NVIDIA's teardown" : "; the teardown is off");
    NVBar0 bar0{&tg_transport()};
    NVFalcon flcn(bar0, t.chip_id);
    NVFiniDiag diag;
    bool hold = false;
    try {
        NVGsp gsp(bar0, flcn, t.queues, t.cmdq_off, t.statq_off, t.queue_size, t.libos_args_sysmem, t.seq, flcn.wait_ms);
        gsp.after_rpc = [&d](uint32_t seq) { __atomic_store_n(&d.state[kNVDStateSeq], seq, __ATOMIC_RELEASE); };
        try { gsp.fini_hw(diag, t.level0); }
        catch (const NVError& e) {   // the RPC failed or timed out: the GSP may be live, so no falcon is touched
            tg_log("the GSP unload failed: %s", e.py().c_str());
            fprintf(stderr, "TinyGPU/NV: GPU teardown failed: the GSP unload: %s\n", e.py().c_str());
            hold = true;
        }
        if (!hold) flcn.fini_hw(diag, t.images);
    } catch (const NVError& e) {   // outside what nv_init_helper tolerates: a confirmed unload still makes closing safe
        tg_log("the C++ GPU teardown failed: %s", e.py().c_str());
        fprintf(stderr, "TinyGPU/NV: GPU teardown failed: %s\n", e.py().c_str());
        hold = !diag.unload_ok;
    }
    secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    tg_log("C++ GPU teardown done in %.3f s: %s", secs, diag.json().c_str());
    report = diag.json();
    return std::string("{\"cmd\":\"fini\",\"cpp_teardown\":true,\"hold\":") + (hold ? "true" : "false") + ",\"diag\":" + diag.json() + "}";
}

// ── C++ runtime (BEAGLE_NV_USE_DAEMON=0): program loading and allocation,
// after a handoff without programs. ─────────────────────────────────────────

// BeagleNVProgram's launch-dims fill; BEAGLE_NV_FILL_LAUNCH_DIMS=0 turns it off, for A/B runs only.
static bool nv_fill_launch_dims() {
    const char* v = getenv("BEAGLE_NV_FILL_LAUNCH_DIMS");
    return !(v && strcmp(v, "0") == 0);
}

// NVProgram.__init__ for every kernel of the ELF (TinyGPUHybridNVProgram.h),
// then the GPU work it submits, in tinygrad's order:
// _ensure_has_local_memory's setup on the compute queue, the image upload,
// and a synchronize. All kernels share one upload of the image, and local
// memory is sized once for the largest need (tinygrad grows it program by
// program to the same size). Each instance loads its own programs, at a new
// lib_va (plan step P5); local memory only grows, as dev.slm_per_thread does:
// a new block and setup when this instance needs more than the GPU has,
// nothing otherwise. (tinygrad's _realloc hands the old block to its LRU
// cache without a synchronize; the setup waits for all earlier work, and the
// pool never reuses the old block.)
static bool nvdLoadPrograms(const NVDElf& elf, const std::vector<std::string>& names, std::map<std::string, NVDKernel>& kernels) {
    auto t0 = nv_profile_start();
    NVDispatchState& d = *g_nvd;
    NVDHandoff& h = d.h;
    NVDProgramParams p;
    p.compute_class = d.rt.compute_class;
    p.sass_version = d.rt.sass_version;
    p.shared_mem_window = d.rt.shared_mem_window;
    p.local_mem_window = d.rt.local_mem_window;
    p.fill_launch_dims = nv_fill_launch_dims();
    p.slm_per_thread = d.slm_per_thread;

    std::string err = nvd_check_tables(h, d.rt.compute_class);
    std::vector<NVDProgramUsage> usage(names.size());
    for (size_t i = 0; err.empty() && i < names.size(); ++i) {
        err = nvd_program_usage(elf, names[i], usage[i]);
        p.slm_per_thread = std::max<uint32_t>(p.slm_per_thread, (uint32_t)nvd_round_up(usage[i].lcmem, 32));
    }
    bool grow = p.slm_per_thread > d.slm_per_thread;
    uint64_t local_mem = 0, tpc_bytes = 0, local_mem_size = 0;
    if (err.empty()) {
        p.lib_va = nvd_pool_alloc(d.rt.pool, d.pool_pos, nvd_round_up(elf.image.size(), 0x1000) + 0x1000);  // NVProgram's lib_gpu
        if (grow) {
            local_mem_size = nvd_local_mem_size(d.rt, p.slm_per_thread, tpc_bytes);
            local_mem = nvd_pool_alloc(d.rt.pool, d.pool_pos, local_mem_size);
        }
        if (!p.lib_va || (grow && !local_mem)) err = "the VRAM pool cannot hold the program image and local memory";
    }
    std::vector<uint8_t> image;
    if (err.empty()) err = nvd_relocate(elf, p.lib_va, image);
    for (size_t i = 0; err.empty() && i < names.size(); ++i) {
        NVDKernel k;
        err = nvd_load_program(elf, names[i], p, usage[i], k);
        if (err.empty()) kernels[names[i]] = std::move(k);
    }
    if (!err.empty()) {
        fprintf(stderr, "TinyGPU/NV: C++ runtime: loading programs failed: %s\n", err.c_str());
        return false;
    }

    if (grow) {
        std::vector<uint32_t> pb;
        uint64_t value = d.timeline++;
        nvd_push_wait(pb, h, h.signal.va, value - 1);
        nvd_push_setup_local_mem(pb, h, local_mem, tpc_bytes);
        nvd_push_signal(pb, h, h.signal.va, value);
        nvd_submit(h.compute, pb);
        d.slm_per_thread = p.slm_per_thread;
    }
    nvdCopyIn(p.lib_va, image.data(), image.size());
    nvd_idle();
    nv_profile_end("load_programs", t0);
    tg_transport().marker(TGM_PROGRAMS_LOADED, names.size());
    if (grow)
        fprintf(stderr, "TinyGPU/NV: C++ runtime: %zu kernels loaded (image %zu bytes at 0x%llx, slm_per_thread 0x%x, "
                "local memory %llu KiB at 0x%llx)\n", names.size(), image.size(), (unsigned long long)p.lib_va, p.slm_per_thread,
                (unsigned long long)(local_mem_size >> 10), (unsigned long long)local_mem);
    else
        fprintf(stderr, "TinyGPU/NV: C++ runtime: %zu kernels loaded (image %zu bytes at 0x%llx, slm_per_thread 0x%x, "
                "local memory unchanged)\n", names.size(), image.size(), (unsigned long long)p.lib_va, p.slm_per_thread);
    return true;
}

// The C++ runtime's embedded cubin for an instance's state count and the booted GPU (plan step C1), and a handle per
// kernel in it. Its programs load after the handoff (nvRuntimePrograms).
static void nvRuntimeCubin(NVInstance& in, int paddedStateCount, bool dp, const std::string& arch, NVDElf& cubin) {
    const TinyGPUNVCubin* c = nullptr;
    std::string err = nvd_find_cubin(kTinyGPUNVCubins, sizeof(kTinyGPUNVCubins) / sizeof(kTinyGPUNVCubins[0]),
                                     paddedStateCount, dp, arch, c);
    if (err.empty()) err = nvd_elf_load(c->begin, (size_t)(c->end - c->begin), 128, cubin);  // NVProgram's force_section_align
    if (!err.empty()) {
        fprintf(stderr, "TinyGPU/NV: C++ runtime: %s\n", err.c_str());
        nv_safe_exit(1);
    }
    for (const std::string& kname : nvd_kernel_names(cubin)) in.kernels[kname] = new NVKernelHandle{kname};
    fprintf(stderr, "TinyGPU/NV: C++ runtime: embedded cubin SP_%d %s (%zu bytes, %zu kernels; ptxas %s)\n", paddedStateCount,
            arch.c_str(), (size_t)(c->end - c->begin), in.kernels.size(), TINYGPU_CUBINS_STAMP);
}

// Each handle's launch template: the handoff's (C++ dispatch) or the instance's own programs (C++ runtime).
static void nvLinkTemplates(NVInstance& in, const std::map<std::string, NVDKernel>& templates) {
    for (auto& kv : in.kernels) {
        auto it = templates.find(kv.first);
        kv.second->tmpl = (it != templates.end()) ? &it->second : nullptr;
    }
}

static void nvRuntimePrograms(NVInstance& in, const NVDElf& cubin) {
    std::vector<std::string> names;
    for (auto& kv : in.kernels) names.push_back(kv.first);
    if (!nvdLoadPrograms(cubin, names, in.templates)) nv_safe_exit(1);
    nvLinkTemplates(in, in.templates);
}

// ── nvDispatchDaemonSetup: spawn nv_dispatch_daemon.py over a dedicated
// socketpair (NOT the TinyGPU socket -- the daemon connects to TinyGPU.app
// itself via NVDevice("NV:0"), matching §74's hardware-verified reference
// test exactly, no inherited FD needed), then send "boot" and
// "compile_all". For C++ dispatch (tg_fd >= 0) the daemon inherits the
// plugin's TinyGPU.app connection instead, and "handoff" follows. The C++
// runtime sends no "compile_all": it loads the embedded cubin for
// paddedStateCount and the GPU the daemon booted (plan step C1). The handles
// and programs are the first instance's, in (plan step P5). ─────────────────

static NVHybridState* nvDispatchDaemonSetup(const char* kernel_code, int tg_fd, int paddedStateCount, bool dp, NVInstance& in) {
    int sv[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) {
        fprintf(stderr, "TinyGPU/NV: socketpair failed: %s\n", strerror(errno));
        return nullptr;
    }

    char script[256];
    const char* helper = getenv("BEAGLE_NV_DISPATCH_DAEMON");
    if (!helper) {
        snprintf(script, sizeof(script), "%s/nv_dispatch_daemon.py", getenv("BEAGLE_NV_SCRIPTS") ?: ".");
        helper = script;
    }
    std::string pypath = nv_resolve_python();

    // Clear O_CLOEXEC on the child's end so it survives execvp.
    int flags = fcntl(sv[1], F_GETFD);
    fcntl(sv[1], F_SETFD, flags & ~FD_CLOEXEC);

    fprintf(stderr, "TinyGPU/NV: spawning nv_dispatch_daemon.py (python=%s)\n", pypath.c_str());
    pid_t pid = fork();
    if (pid < 0) {
        fprintf(stderr, "TinyGPU/NV: fork failed: %s\n", strerror(errno));
        close(sv[0]); close(sv[1]);
        return nullptr;
    }
    if (pid == 0) {
        close(sv[0]);
        setsid();  // its own session: a terminal Ctrl-C or hangup must not reach a daemon that may hold a live GPU
        dup2(STDERR_FILENO, STDOUT_FILENO);
        char fd_str[16]; snprintf(fd_str, sizeof(fd_str), "%d", sv[1]);
        char tg_str[16]; snprintf(tg_str, sizeof(tg_str), "%d", tg_fd);
        char* argv[] = { (char*)pypath.c_str(), (char*)helper, fd_str, tg_fd >= 0 ? tg_str : nullptr, nullptr };
        execvp(pypath.c_str(), argv);
        fprintf(stderr, "TinyGPU/NV: execvp %s failed: %s\n", pypath.c_str(), strerror(errno));
        _exit(1);
    }
    close(sv[1]);

    NVHybridState* g = new NVHybridState{};
    g->cmd_sock = sv[0];
    g->owner_pid = getpid();
    int one = 1;  // a gone daemon shows up as EPIPE, so nv_safe_exit still reports, instead of SIGPIPE ending the host
    setsockopt(g->cmd_sock, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
    g->daemon_pid = pid;

    fprintf(stderr, "TinyGPU/NV: sending boot command...\n"); fflush(stderr);
    auto t0 = nv_profile_start();
    nv_send_msg(g->cmd_sock, "{\"cmd\":\"boot\"}");
    std::string resp = nv_recv_msg(g->cmd_sock);
    nv_profile_end("boot", t0);
    if (resp.empty() || !nv_json_ok(resp)) {
        std::string err = nv_json_str(resp, "error");
        fprintf(stderr, "TinyGPU/NV: boot failed: %s\n", err.empty() ? resp.c_str() : err.c_str());
        nv_report_unload(resp);   // a boot that failed after GSP-RM started: the daemon unloaded it, and may hold
        delete g;
        return nullptr;
    }
    g->arch = nv_json_str(resp, "arch");
    fprintf(stderr, "TinyGPU/NV: daemon booted — arch=%s\n", g->arch.c_str());
    g_nv = g;   // from here on every exit, a GPU hang during setup included, tears the GPU down through the daemon

    NVDElf cubin;
    if (nv_cpp_runtime() && tg_fd >= 0) {
        nvRuntimeCubin(in, paddedStateCount, dp, g->arch, cubin);
    } else if (kernel_code && kernel_code[0]) {
        char ptx_path[256];
        snprintf(ptx_path, sizeof(ptx_path), "/tmp/beagle_nv_all_%d.ptx", getpid());
        nv_write_file(ptx_path, kernel_code, strlen(kernel_code));

        char cmd[512];
        snprintf(cmd, sizeof(cmd), "{\"cmd\":\"compile_all\",\"ptx_path\":\"%s\"}", ptx_path);
        fprintf(stderr, "TinyGPU/NV: compiling all kernels (ptxas × 1, via daemon)…\n");
        fflush(stderr);
        t0 = nv_profile_start();
        nv_send_msg(g->cmd_sock, cmd);
        resp = nv_recv_msg(g->cmd_sock);
        nv_profile_end("compile_all", t0);
        unlink(ptx_path);
        if (resp.empty() || !nv_json_ok(resp)) {
            fprintf(stderr, "TinyGPU/NV: compile_all failed: %s\n", resp.c_str());
            nv_safe_exit(1);
        }
        // Register a lightweight handle per kernel name found in the reply's
        // "kernels" array so GetFunction() has something to hand back.
        size_t p = resp.find("\"kernels\":");
        if (p != std::string::npos) {
            size_t arr_end = resp.find(']', p);
            size_t q = p;
            int loaded = 0;
            while (true) {
                size_t qs = resp.find('"', q);
                if (qs == std::string::npos || qs > arr_end) break;
                size_t qe = resp.find('"', qs + 1);
                if (qe == std::string::npos) break;
                std::string kname = resp.substr(qs + 1, qe - qs - 1);
                if (!kname.empty() && kname != "kernels") {
                    in.kernels[kname] = new NVKernelHandle{kname};
                    ++loaded;
                }
                q = qe + 1;
            }
            fprintf(stderr, "TinyGPU/NV: compile_all — loaded %d kernels\n", loaded);
        }
    }
    if (tg_fd >= 0) {
        g_nvd = nvDispatchHandoff(g->cmd_sock, tg_fd, nv_cpp_runtime());
        if (!g_nvd || !nvdStatePage(g->cmd_sock, *g_nvd)) nv_safe_exit(1);
        if (g_nvd->runtime && nv_cpp_level() >= kNVLevelTeardown) nvdTeardownExport(g->cmd_sock, *g_nvd);
        if (g_nvd->runtime) nvRuntimePrograms(in, cubin);
        else nvLinkTemplates(in, g_nvd->h.kernels);
    }
    fflush(stderr);
    return g;
}

// ── GPUInterface entry points ─────────────────────────────────────────────────

// The GPU teardown: the GPU finishes its work, then the daemon's fini (GSP unload, NVIDIA's teardown) and its report. The
// C++ runtime's GPU owns its TinyGPU.app connection (plan step P5), closed last, with nv_usb4.lock; a daemon that holds
// keeps its own copies of both.
static void nvFiniDevice() {
    if (!g_nv) return;
    int tg_sock = -1;
    std::string cpp_fini, cpp_report;   // BEAGLE_NV_CPP_LEVEL=teardown: this side's own GPU teardown and its report (plan step C5)
    double cpp_secs = 0;
    if (g_nvd) {  // let the GPU finish before it is torn down
        nvd_idle();
        if (g_nvd->runtime) tg_sock = g_nvd->tg_sock;
        if (g_nvd->td.ready && g_nv->cmd_sock >= 0) {
            tg_transport().marker(TGM_FINI, 0);
            cpp_fini = nvdCppTeardown(*g_nvd, cpp_secs, cpp_report);
        }
        nvd_unmap(g_nvd);
        delete g_nvd;
        g_nvd = nullptr;
    }
    nv_profile_report();
    for (auto& kv : g_nvKernelLaunches)
        if (nv_profile_enabled()) fprintf(stderr, "TinyGPU/NV: [profile]   kernel %s n=%lld\n", kv.first.c_str(), kv.second);
    g_nvKernelLaunches.clear();
    bool hold = false;
    if (g_nv->cmd_sock >= 0) {
        if (cpp_fini.empty()) tg_transport().marker(TGM_FINI, 0);
        // The daemon tears the GPU down now (GSP unload, then a wait for the GSP to report itself suspended) and
        // replies with what it saw; after this side's own teardown (plan step C5) it only takes the report and exits. If
        // the GPU did not confirm the unload, the daemon keeps its copy of the TinyGPU.app connection open: closing it
        // could unmap memory the GSP still uses.
        auto t0 = nv_profile_start();
        nv_send_msg(g_nv->cmd_sock, cpp_fini.empty() ? "{\"cmd\":\"fini\"}" : cpp_fini);
        std::string resp = nv_recv_msg(g_nv->cmd_sock);
        double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (resp.empty() && !cpp_report.empty()) {   // the daemon is gone, but this side knows what its own teardown did
            fprintf(stderr, "TinyGPU/NV: no fini reply from the daemon (it exited?); the report of this side's own GPU teardown follows\n");
            resp = cpp_report;
        } else if (resp.empty()) nv_report_no_fini_reply();
        else if (resp.find("\"mailbox0\":") == std::string::npos && !nv_json_ok(resp))
            fprintf(stderr, "TinyGPU/NV: GPU teardown failed: %s\n", resp.c_str());
        hold = nv_report_unload(resp);
        if (nv_profile_enabled() && cpp_fini.empty())
            fprintf(stderr, "TinyGPU/NV: fini round trip %.3f s (synchronize, GSP unload, teardown)\n", secs);
        else if (nv_profile_enabled())
            fprintf(stderr, "TinyGPU/NV: fini %.3f s: the C++ GSP unload and teardown, then %.3f s for the daemon's reply\n", cpp_secs, secs);
    }
    if (g_nv->daemon_pid > 0 && !hold) {
        for (int i = 0; i < 100; ++i) {
            int st = 0;
            if (waitpid(g_nv->daemon_pid, &st, WNOHANG) > 0) { g_nv->daemon_pid = 0; break; }
            usleep(100000);
        }
    }
    if (g_nv->cmd_sock >= 0) close(g_nv->cmd_sock);
    delete g_nv;
    g_nv = nullptr;
    if (tg_sock >= 0) tg_transport().close();
}

// Plan step P5: the C++ runtime's GPU outlives its instances, as tinygrad's devices do (device.py finalizes them at
// exit): a connection's sysmem cannot be freed, so later instances share this boot instead of booting again. A thread
// still holding the lock after 35 s (past the 30 s timeline timeout) may be mid-frame: then nothing is sent, and the
// daemon, at EOF, tears the GPU down or holds, as the state page says. A child forked after the boot inherits this
// handler, the daemon's socket and the TinyGPU.app connection, and returns at once: the GPU is its parent's (checked
// before the lock, which a fork can copy held).
static void nvAtExit() {
    if (!g_nv || g_nv->owner_pid != getpid()) return;
    std::unique_lock<std::recursive_timed_mutex> lk(nv_mutex(), std::defer_lock);
    if (!lk.try_lock_for(std::chrono::seconds(35))) {
        fprintf(stderr, "TinyGPU/NV: another thread is still using the GPU at exit; not tearing it down from here: the "
                "daemon does at EOF, or holds, as the state page says\n");
        return;
    }
    nvFiniDevice();
}

// Plan step P5: an instance created while another has the GPU booted shares that boot, since TinyGPU.app serves one
// connection at a time and a new one from here would wait forever. Only the C++ runtime shares; the daemon and C++
// dispatch modes boot once per instance until C12, so a second instance alongside the first is refused.
int NvAttachShared(GPUInterface* self) {
    std::lock_guard<std::recursive_timed_mutex> lk(nv_mutex());
    if (!g_nv) return 0;
    if (!(g_nvd && g_nvd->runtime)) {
        fprintf(stderr, "TinyGPU/NV: another BEAGLE instance in this process has the GPU; sharing it needs the C++ runtime "
                "(BEAGLE_NV_USE_DAEMON=0): the daemon and C++ dispatch modes boot it for one instance at a time\n");
        return -1;
    }
    self->nvGspState = new NVInstance;
    return 1;
}

void NvSetDevice(GPUInterface* self, int paddedStateCount, int categoryCount,
                  int patternCount, int unpaddedPatternCount, int tipCount, long flags) {
    std::lock_guard<std::recursive_timed_mutex> lk(nv_mutex());
    NVInstance* shared = (NVInstance*)self->nvGspState;   // NvAttachShared's: the GPU is booted already (plan step P5)
    // Close Initialize()'s TinyGPU.app connection before spawning the
    // dispatch daemon: NVDevice("NV:0") opens its own, fully independent
    // connection (STATUS.md §74, hardware-verified), same as
    // the AMD daemon does. Leaving this one open too risks the exact bug
    // AMD's own tgpuSock fix (STATUS.md AMD §21) found: TinyGPU.app doesn't
    // tolerate two simultaneous clients, and the daemon's own connection
    // attempt hangs forever instead of failing cleanly. C++ dispatch keeps
    // it instead and the daemon inherits it: both sides then share the one
    // connection (the GPUInterface destructor closes it after NvFini; in the
    // C++ runtime the GPU keeps it until exit, plan step P5).
    int tg_fd = -1;
    if (nv_cpp_dispatch() && self->tgpuSock >= 0) {
        tg_fd = self->tgpuSock;
        fcntl(tg_fd, F_SETFD, fcntl(tg_fd, F_GETFD) & ~FD_CLOEXEC);
        // nv_usb4.lock goes to the daemon with the connection, so a daemon still holding the connection after this process
        // is gone still holds the lock (plan step P5)
        int lock_fd = tg_transport().lock_fd();
        if (lock_fd >= 0) fcntl(lock_fd, F_SETFD, fcntl(lock_fd, F_GETFD) & ~FD_CLOEXEC);
        int one = 1;  // a lost TinyGPU.app shows up as EPIPE in nvd_submit (the cut-frame report), not SIGPIPE ending the host
        setsockopt(tg_fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
    } else if (self->tgpuSock >= 0) {
        tg_transport().close();   // and nv_usb4.lock, which the daemon's own APLRemotePCIDevice takes (plan step P5)
        self->tgpuSock = -1;
    }

    self->InitializeKernelResource(paddedStateCount, (flags & BEAGLE_FLAG_PRECISION_DOUBLE) != 0);
    self->supportDoublePrecision = ((flags & BEAGLE_FLAG_PRECISION_DOUBLE) != 0);
    if (self->kernelResource) {
        self->kernelResource->categoryCount        = categoryCount;
        self->kernelResource->patternCount         = patternCount;
        self->kernelResource->unpaddedPatternCount = unpaddedPatternCount;
        self->kernelResource->flags                = flags;
    }

    if (shared) {
        NVDElf cubin;
        nvRuntimeCubin(*shared, paddedStateCount, self->supportDoublePrecision, g_nv->arch, cubin);
        nvRuntimePrograms(*shared, cubin);
        return;
    }
    NVInstance* in = new NVInstance;
    self->nvGspState = in;
    g_nv = nvDispatchDaemonSetup(self->kernelResource ? self->kernelResource->kernelCode : nullptr, tg_fd, paddedStateCount,
                                 self->supportDoublePrecision, *in);
    if (!g_nv) { fprintf(stderr, "TinyGPU/NV: nvDispatchDaemonSetup failed\n"); nv_safe_exit(1); }
    if (tg_fd >= 0) {   // the daemon has its copies: no later child of the host may keep the connection or the lock
        fcntl(tg_fd, F_SETFD, fcntl(tg_fd, F_GETFD) | FD_CLOEXEC);
        int lock_fd = tg_transport().lock_fd();
        if (lock_fd >= 0) fcntl(lock_fd, F_SETFD, fcntl(lock_fd, F_GETFD) | FD_CLOEXEC);
    }
    if (g_nvd && g_nvd->runtime) {   // plan step P5: the GPU outlives this instance, and later ones share it
        self->tgpuSock = -1;          // the GPU's connection now, g_nvd->tg_sock
        if (atexit(nvAtExit) != 0)
            fprintf(stderr, "TinyGPU/NV: atexit failed; the GPU is torn down only by the daemon, at EOF\n");
    }
}

GPUFunction NvGetFunction(GPUInterface* self, const char* name) {
    NVInstance* in = (NVInstance*)self->nvGspState;
    if (!in) return nullptr;
    auto it = in->kernels.find(name);
    if (it != in->kernels.end()) return it->second;
    fprintf(stderr, "TinyGPU/NV: GetFunction(%s): kernel not found in precompiled cache — exiting\n", name);
    nv_safe_exit(1);
}

void NvSynchronizeHost(GPUInterface* self) {
    std::lock_guard<std::recursive_timed_mutex> lk(nv_mutex());
    NVInstance* in = (NVInstance*)self->nvGspState;
    if (!g_nv || !in) return;
    nvFlushLaunchQueue(*in);  // otherwise queued-but-unsent launches wouldn't be submitted yet to wait for
    auto t0 = nv_profile_start();
    if (g_nvd) {
        nvd_idle();
        nv_profile_end("sync", t0);
        return;
    }
    nv_send_msg(g_nv->cmd_sock, "{\"cmd\":\"sync\"}");
    std::string resp = nv_recv_msg(g_nv->cmd_sock);
    nv_profile_end("sync", t0);
    if (resp.empty() || !nv_json_ok(resp))
        fprintf(stderr, "TinyGPU/NV: sync failed: %s\n", resp.c_str());
}

GPUPtr NvAllocateMemory(size_t sz) {
    std::lock_guard<std::recursive_timed_mutex> lk(nv_mutex());
    if (!g_nv) return 0;
    if (g_nvd && g_nvd->runtime) {
        uint64_t va = nvd_pool_alloc(g_nvd->rt.pool, g_nvd->pool_pos, sz);
        if (!va) {   // BEAGLE does not check: address 0 would reach the GPU. The pool is never reclaimed (plan step P5).
            fprintf(stderr, "TinyGPU/NV: alloc(%zu): VRAM pool exhausted (%llu MiB; set BEAGLE_NV_DATA_MB)\n", sz,
                    (unsigned long long)(g_nvd->rt.pool.size >> 20));
            nv_safe_exit(1);
        }
        return (GPUPtr)va;
    }
    char cmd[128];
    snprintf(cmd, sizeof(cmd), "{\"cmd\":\"alloc\",\"size\":%zu}", sz);
    auto t0 = nv_profile_start();
    nv_send_msg(g_nv->cmd_sock, cmd);
    std::string resp = nv_recv_msg(g_nv->cmd_sock);
    nv_profile_end("alloc", t0);
    if (resp.empty() || !nv_json_ok(resp)) {
        fprintf(stderr, "TinyGPU/NV: alloc(%zu) failed: %s\n", sz, resp.c_str());
        return 0;
    }
    return (GPUPtr)nv_json_u64(resp, "addr");
}

void NvMemcpyHostToDevice(GPUInterface* self, GPUPtr dst, const void* src, size_t sz) {
    std::lock_guard<std::recursive_timed_mutex> lk(nv_mutex());
    NVInstance* in = (NVInstance*)self->nvGspState;
    if (!g_nv || !in || !src || !sz) return;
    nvFlushLaunchQueue(*in);  // preserve ordering: queued launches must be submitted before this write
    if (g_nvd) { nvdCopyIn(dst, src, sz); return; }
    char cmd[128];
    snprintf(cmd, sizeof(cmd), "{\"cmd\":\"h2d\",\"addr\":%llu,\"size\":%zu}", (unsigned long long)dst, sz);
    auto t0 = nv_profile_start();
    nv_send_msg(g_nv->cmd_sock, cmd);
    nv_send_all(g_nv->cmd_sock, src, sz);
    std::string resp = nv_recv_msg(g_nv->cmd_sock);
    nv_profile_end("h2d", t0);
    if (resp.empty() || !nv_json_ok(resp))
        fprintf(stderr, "TinyGPU/NV: h2d(addr=0x%llx, sz=%zu) failed: %s\n", (unsigned long long)dst, sz, resp.c_str());
}

void NvMemcpyDeviceToHost(GPUInterface* self, void* dst, const GPUPtr src, size_t sz) {
    std::lock_guard<std::recursive_timed_mutex> lk(nv_mutex());
    NVInstance* in = (NVInstance*)self->nvGspState;
    if (!g_nv || !in || !dst || !sz) return;
    nvFlushLaunchQueue(*in);  // preserve ordering: queued launches must complete before this read
    if (g_nvd) { nvdCopyOut(dst, src, sz); return; }
    char cmd[128];
    snprintf(cmd, sizeof(cmd), "{\"cmd\":\"d2h\",\"addr\":%llu,\"size\":%zu}", (unsigned long long)src, sz);
    auto t0 = nv_profile_start();
    nv_send_msg(g_nv->cmd_sock, cmd);
    std::string resp = nv_recv_msg(g_nv->cmd_sock);
    if (resp.empty() || !nv_json_ok(resp)) {
        fprintf(stderr, "TinyGPU/NV: d2h(addr=0x%llx, sz=%zu) failed: %s\n", (unsigned long long)src, sz, resp.c_str());
        return;
    }
    nv_recv_all(g_nv->cmd_sock, dst, sz);
    nv_profile_end("d2h", t0);
}

size_t NvGetAvailableMemory() {
    std::lock_guard<std::recursive_timed_mutex> lk(nv_mutex());
    if (g_nvd && g_nvd->runtime) return (size_t)(g_nvd->rt.pool.size - g_nvd->pool_pos);
    // Python (via NVAllocator/HCQCompiled) owns allocation entirely now; this
    // backend has no independent view of remaining VRAM. Report a generous
    // constant rather than 0 (which some callers may treat as "out of
    // memory") -- purely informational, not load-bearing. Same convention as
    // AmdGetAvailableMemory().
    return g_nv ? (size_t)(1ull << 30) : 0;
}

// Releases this instance. The daemon and C++ dispatch modes also tear the GPU down (one instance per boot); the C++
// runtime's GPU stays for later instances until exit (nvAtExit; plan step P5).
void NvFini(GPUInterface* self) {
    std::lock_guard<std::recursive_timed_mutex> lk(nv_mutex());
    NVInstance* in = (NVInstance*)self->nvGspState;
    if (!in) return;
    self->nvGspState = nullptr;
    nvFlushLaunchQueue(*in);  // don't silently drop queued-but-unsent launches
    for (auto& kv : in->kernels) {   // reported with the GPU teardown, so not once it is done (nor after static destructors)
        if (g_nv && kv.second->launches) g_nvKernelLaunches[kv.first] += kv.second->launches;
        delete kv.second;
    }
    delete in;
    if (!(g_nvd && g_nvd->runtime)) nvFiniDevice();
}

void NvLaunchKernelImpl(GPUInterface* self, GPUFunction fn, Dim3Int block, Dim3Int grid,
                         int nPtr, int nTotal, GPUPtr* ptrs, unsigned int* ints) {
    NVInstance* in = (NVInstance*)self->nvGspState;
    if (!in || !fn) return;
    NVKernelHandle* ke = (NVKernelHandle*)fn;
    int nInt = nTotal - nPtr;
    ++ke->launches;

    // Queued, not sent (mirrors AMD's STATUS.md §26 batching) --
    // nvFlushLaunchQueue() sends the whole backlog as one RPC round-trip,
    // called before any h2d/d2h/sync/fini so ordering relative to memory
    // operations is preserved.
    NVPendingLaunch pl;
    pl.kernel = ke->name;
    pl.tmpl = ke->tmpl;
    pl.grid[0] = grid.x; pl.grid[1] = grid.y; pl.grid[2] = grid.z;
    pl.block[0] = block.x; pl.block[1] = block.y; pl.block[2] = block.z;
    pl.ptrs.assign(ptrs, ptrs + nPtr);
    pl.ints.assign(ints, ints + nInt);
    in->pending.push_back(std::move(pl));
}

} // namespace tinygpu_device

#endif // FW_TINYGPU
