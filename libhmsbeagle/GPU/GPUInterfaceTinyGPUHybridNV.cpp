/*
 * GPUInterfaceTinyGPUHybridNV.cpp
 *
 * BEAGLE's NV backend on TinyGPU.app (TinyGPUHybrid.md): tinygrad's NV driver, ported to C++. The first instance in a
 * process boots the GPU (TinyGPUHybridNVBoot.h, TinyGPUHybridNVFalcon.h, TinyGPUHybridNVGsp.h), builds tinygrad's NVDevice
 * on its own RM client and memory manager (TinyGPUHybridNVRM.h, TinyGPUHybridNVDevice.h, TinyGPUHybridNVMemory.h) and loads
 * its programs from the cubins linked into the plugin (TinyGPUHybridNVProgram.h, TinyGPUHybridNVCubins.h); nothing is
 * compiled at run time. This file encodes launches and copies (TinyGPUHybridNVDispatch.h) and submits them on the plugin's
 * own TinyGPU.app connection (TinyGPUTransport.h); completion is a timeline semaphore in shared memory, polled locally.
 * Every instance in the process shares that one boot, which lasts until exit (TODO.md plan step P5), when this file unloads
 * the GPU and runs NVIDIA's teardown. The crash guard (tinygpu_guard.cpp) keeps the GPU if this process dies, and a GPU lost
 * to it (a hang, a broken connection) returns errors to BEAGLE instead of exiting its host (plan step C12). Only Ada and
 * Blackwell GPUs are booted (plan step C13c: the Python daemon this backend began with is the test harness's oracle now,
 * tinygpu_tests/oracle).
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
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include <dlfcn.h>
#include <fcntl.h>
#include <poll.h>
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
#include "libhmsbeagle/GPU/TinyGPUHybridNVMemory.h"
#include "libhmsbeagle/GPU/TinyGPUHybridNVDevice.h"
#include "libhmsbeagle/GPU/TinyGPUHybridNVDispatch.h"
#include "libhmsbeagle/GPU/TinyGPUHybridNVGuard.h"
#include "libhmsbeagle/GPU/TinyGPUHybridNVBoot.h"
#include "libhmsbeagle/GPU/TinyGPUHybridNVProgram.h"
#include "libhmsbeagle/GPU/TinyGPUHybridNVCubins.h"

// The embedded cubins were compiled from the current kernels header's PTX (TODO.md plan step C1): make_tinygpu_cubins.sh
// copies the stamp of the header whose PTX it compiled, and GPUInterface.h includes that header's stamp (plan step C13).
static_assert(std::string_view(TINYGPU_CUBINS_KERNELS_STAMP) == TINYGPU_KERNELS_STAMP,
              "kernels/TinyGPUNVCubins.h is from another BeagleTinyGPU_kernels.h: rebuild the TinyGPUCubins target");

namespace tinygpu_device {

// ── small utilities: the minimal JSON the teardown's report is read with ──

static bool nv_json_bool(const std::string& js, const char* key) {
    std::string needle = std::string("\"") + key + "\":";
    auto p = js.find(needle);
    if (p == std::string::npos) return false;
    p += needle.size();
    while (p < js.size() && js[p]==' ') ++p;
    return js.compare(p, 4, "true") == 0;
}
static std::string nv_json_str(const std::string& js, const char* key) {
    char needle[128]; snprintf(needle, sizeof(needle), "\"%s\":", key);
    auto p = js.find(needle);
    if (p == std::string::npos) return "";
    p = js.find('"', p + strlen(needle));
    if (p == std::string::npos) return "";
    auto e = js.find('"', p + 1);
    return js.substr(p + 1, e - p - 1);
}

// ── Opt-in profiling (BEAGLE_NV_PROFILE=1) ─────────────────────────────────
// The NV counterpart of BEAGLE_AMD_PROFILE, aggregated instead of printed
// per call so a many-evaluation benchmark (tinygpuhybridtest --reps) stays
// readable: the teardown at exit prints count/mean/min/max per call.
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
    fprintf(stderr, "TinyGPU/NV: [profile] calls:\n");
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
    std::string arch;   // the NVDevice's: later instances pick their cubins for it (plan step P5)
    pid_t owner_pid;    // the process that booted; a child forked from it shares its connections and must never tear down
    bool lost = false;  // plan step C12: the GPU went to its keeper (nv_gpu_lost); nothing more is sent to it, and no instance uses it
};

struct NVKernelHandle {
    std::string name;
    const NVDKernel* tmpl = nullptr;  // C++ dispatch: this kernel's handoff template
    long long launches = 0;           // launches BEAGLE requested, for BEAGLE_NV_PROFILE's report (plan step D1)
};

// The process's one GPU, which every instance shares (TODO.md plan step P5): here, and in g_nvd below, the TinyGPU.app
// connection, rings, timeline, local memory and VRAM pool. What is an instance's own is its NVInstance.
static NVHybridState* g_nv = nullptr;
static std::map<std::string, long long> g_nvKernelLaunches;   // released instances' launches per kernel, for the profile report

// Plan step P5: one lock around everything the instances share, so a frame to TinyGPU.app never interleaves with another
// thread's and the rings, timeline and pool stay consistent. Recursive, because nv_gpu_lost takes it on paths that
// already hold it; timed, for nvAtExit. Never destroyed: a GPUInterface may be destroyed after the static destructors ran.
static std::recursive_timed_mutex& nv_mutex() {
    static std::recursive_timed_mutex* m = new std::recursive_timed_mutex;
    return *m;
}

// The state page (TODO.md plan step P3; nvdStatePage below): four 64-bit words shared with the crash guard, which reads them
// only once this side can no longer write (at the end of its socketpair, after this process died or lost the GPU): the boot's
// phase, whether a frame is in flight, the timeline value the last frame signals and the GSP command queue's sequence number
// after this side's last RPC (plan step C5; TinyGPUHybridNVGuard.h).
enum { kNVDStatePhase, kNVDStateInFlight, kNVDStateLastSubmitted, kNVDStateSeq, kNVDStateWords };
static const uint64_t kNVDPhaseDispatch = 1;  // this side owns both GPFIFOs; the guard holds on any other phase
static const uint64_t kNVDPhaseTeardown = 2;  // this side is unloading the GPU itself (plan step C5): the guard holds
static const uint64_t kNVDPhaseGspInit = 3;   // GSP-RM may run, before its INIT_DONE (plan step C8): the guard holds
static const uint64_t kNVDPhaseFlcnInit = 4;  // before booter_load or the COT message (plan step C9): GSP-RM never started, the guard closes
static_assert(kNVDStatePhase == kGuardStatePhase && kNVDStateInFlight == kGuardStateInFlight && kNVDStateLastSubmitted == kGuardStateLastSubmitted &&
              kNVDStateSeq == kGuardStateSeq && kNVDStateWords == kNVDStateWordsGuard, "the guard's state page");
static_assert(kNVDPhaseDispatch == kGuardPhaseDispatch && kNVDPhaseTeardown == kGuardPhaseTeardown && kNVDPhaseGspInit == kGuardPhaseGspInit &&
              kNVDPhaseFlcnInit == kGuardPhaseFlcnInit, "the guard's phases");

// The four buffers this side allocates after the NVDevice (plan step C6), in this order (the oracle's _HANDOFF_BUFS,
// nv_dispatch_daemon.py): PCIIfaceBase.alloc's size, host, uncached and cpu_access
struct NVDBufferSpec { uint64_t size; bool host, uncached, cpu_access; };
static const NVDBufferSpec kNVDBuffers[4] = {
    {2 << 20, false, false, true},    // cmdq: pushbuffers of both queues
    {16 << 20, false, false, true},   // kargs: kernargs slots, cbuf0 + args, then the QMD
    {16 << 20, false, false, true},   // staging: h2d/d2h bounce buffer
    {0x1000, true, true, true}};      // signal: the timeline

// What the GPU teardown needs (plan step C5), from the boot
struct NVDTeardown {
    bool ready = false;
    uint8_t* queues = nullptr;   // the GSP message queues, in TinyGPU.app sysmem
    uint64_t queues_size = 0, cmdq_off = 0, statq_off = 0, queue_size = 0, libos_args_sysmem = 0;
    uint32_t seq = 0, chip_id = 0;
    bool level0 = false;         // BEAGLE_NV_UNLOAD_LEVEL=0, as the boot read it
    NVTeardownImages images;     // nv_init_helper's FWSEC-SB and Booter Unload, if the teardown is on
    bool cot = false;            // GB20x's COT boot (plan step B2): the unload, then the RISC-V halt wait; no falcon ucode
    std::string chip_name;       // NVDev.chip_name
    int queues_fd = -1;          // the queues' TinyGPU.app sysmem fd, kept for the crash guard (plan step C10)
};

// The GPU's state (see "Dispatch" below).
struct NVDispatchState {
    NVDHandoff h;
    int tg_sock = -1;            // the plugin's TinyGPU.app connection
    void* maps[4] = {};          // host mappings of h.cmdq, h.kargs, h.staging, h.signal
    size_t map_sizes[4] = {};    // ... and their sizes
    std::unique_ptr<NVMemState> mem;   // tinygrad's memory manager (plan step C6), from the boot
    int signal_fd = -1;          // the timeline's TinyGPU.app fd, for the crash guard
    int state_fd = -1;           // the state page's fd, for the crash guard (plan step C10)
    uint64_t bar0_size = 0;      // BAR0's size, as the boot mapped it
    int guard_ctl = -1;          // the crash guard's socketpair (plan steps C10, C11)
    pid_t guard_pid = 0;
    std::unique_ptr<NVBar0> bar0;      // the GSP, NV_GSP's RM client and the NVDevice (plan steps C7-C11)
    std::unique_ptr<NVFalcon> flcn;
    std::unique_ptr<NVGsp> gsp;
    std::unique_ptr<NVRMClient> rm;
    NVDeviceState dev;
    uint8_t *cmdq = nullptr, *kargs = nullptr, *staging = nullptr;
    uint64_t* signal = nullptr;  // timeline semaphore
    uint64_t* state = nullptr;   // the state page (kNVDState* words)
    uint64_t cmdq_pos = 0, kargs_pos = 0, staging_pos = 0;
    uint64_t timeline = 1;       // value the next submission signals; every earlier value is submitted
    uint64_t pending = 0;        // submissions since the GPU was last seen idle
    NVDRuntime rt;               // the programs' parameters, and the VRAM pool allocations come from
    uint64_t pool_pos = 0;       // rt.pool's fill level
    TGPoolFree pool_free;        // its blocks, and those freed below pool_pos (plan step C14)
    uint32_t slm_per_thread = 0; // dev.slm_per_thread: the local memory set up so far serves this much per thread
    NVDTeardown td;              // plan step C5
};
static NVDispatchState* g_nvd = nullptr;
static uint64_t nvd_pool_left(const NVDispatchState& d) { return d.rt.pool.size - d.pool_pos + d.pool_free.bytes(); }
static void nv_test_kill(const char* point, NVDispatchState* d = nullptr);   // plan step C10's test hook (below)

// TODO.md plan decision 16: the GPUs this backend boots, by the PCI device ID Initialize's probe read (tinygrad's PCIIface
// family list, ops_nv.py:559): Ada (AD10x, 0x26xx-0x28xx; tested on an RTX 4060) and Blackwell (GB20x, 0x2bxx-0x2dxx and
// 0x2fxx; tested on an RTX 5070, the user's choice for the others). Ampere (0x22xx-0x25xx) is refused: the C++ boot has no
// register tables for it (decision 17), and since plan step C13c there is no Python path either.
static bool nv_boot_gpu() {
    const uint16_t family = tg_pci_device_id() & 0xff00;
    return family == 0x2600 || family == 0x2700 || family == 0x2800 || family == 0x2b00 || family == 0x2c00 || family == 0x2d00 ||
           family == 0x2f00;
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
    uint64_t lib_va = 0;   // its programs' image in the pool, freed with it, as tinygrad's NVProgram frees lib_gpu (plan step C14)
    bool failed = false;   // plan step C12: its setup failed, or an allocation (BEAGLE would hand address 0 to the GPU)
    bool oom = false;      // plan step M1: ... for lack of GPU memory (BeagleGPUImpl then returns BEAGLE_ERROR_OUT_OF_MEMORY)
};
static bool g_nvSetupOOM = false;   // plan step M1: the last boot's VRAM pool did not fit (nvdOwnAllocations)

// TODO.md plan step C12: nothing of an instance reaches the GPU once its setup failed or the GPU is lost; its calls do nothing
// and BeagleGPUImpl returns errors (GPUInterface::GetDeviceLost)
static bool nv_failed(const NVInstance& in) { return in.failed || !g_nv || g_nv->lost; }

// Plan step C12: a GPU that stopped making progress, or whose TinyGPU.app stream broke, found deep in a launch, a copy or a
// wait. Thrown there, caught at the GPUInterface entry points, which hand the GPU to its keeper (nv_gpu_lost) and return.
struct NVGpuLost { bool hung; };

static void nvdFlushLaunches(std::vector<NVPendingLaunch>& pending);

static void nvFlushLaunchQueue(NVInstance& in) {
    if (nv_failed(in)) { in.pending.clear(); return; }
    if (!in.pending.empty()) nvdFlushLaunches(in.pending);
}

// The report of the GPU's unload at exit (nvdCppTeardown's, in the format of the daemon's fini reply this backend began with):
// the unload, NVIDIA's teardown if it ran (on unless BEAGLE_NV_TEARDOWN=0; plan steps P2, P3) or else whether the next boot
// needs a power cycle, and whether the crash guard keeps the TinyGPU.app connection open because the GPU may still use memory
// behind it (the unload was not confirmed). Returns that last one.
static bool nv_report_unload(const std::string& resp, const char* who) {
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
        fprintf(stderr, "TinyGPU/NV: %s (pid %llu) keeps the TinyGPU.app connection open because the GPU may still use "
                "memory behind it. Unplug the eGPU first, then kill %llu.\n", who, (unsigned long long)pid, (unsigned long long)pid);
    return hold;
}

// TODO.md plan step C12, which replaced this library's _exit (nv_safe_exit): the GPU is lost to this process, because it hung
// or its TinyGPU.app stream broke. The crash guard takes it as at this process's death, at the end of its socketpair, as the
// state page says (after a hang its own timeline wait fails, and it sends only the unload and holds). Then this process
// closes its copy of the connection and sends the GPU nothing more, and every instance's calls return errors: the host goes
// on. The lock keeps every other thread off the connection meanwhile (plan step P5). A child forked after the boot sends
// nothing: the GPU is its parent's.
static void nv_gpu_lost(bool hung) {
    std::lock_guard<std::recursive_timed_mutex> lk(nv_mutex());
    if (!g_nv || g_nv->lost) return;
    g_nv->lost = true;
    if (g_nv->owner_pid != getpid()) return;
    fflush(stderr);
    if (g_nvd && g_nvd->guard_ctl >= 0) {   // the guard decides now, as at this process's death
        close(g_nvd->guard_ctl);
        g_nvd->guard_ctl = -1;
        fprintf(stderr, "TinyGPU/NV: the crash guard (pid %d) keeps the GPU: it tears it down, or holds it and says so (its lines are "
                "in %s)\n", (int)g_nvd->guard_pid, tg_log_path().c_str());
    }
    tg_transport().close();
    fprintf(stderr, "TinyGPU/NV: the GPU is lost to this process%s: nothing more is sent to it, and BEAGLE's calls on it return "
            "errors\n", hung ? " (it hung)" : "");
    tg_log("the GPU is lost to this process%s (plan step C12): nothing more is sent to it", hung ? " (it hung)" : "");
}

// ── Dispatch (TODO.md "Runtime roadmap", Step 3). This file encodes launches
// and copies (TinyGPUHybridNVDispatch.h, golden-tested byte for byte against
// hcq1's own encoders) into four shared sysmem buffers, and submits them by
// writing the GPFIFO entry, GPPut and doorbell as posted MMIO writes on the
// plugin's TinyGPU.app connection. Completion is a
// timeline semaphore in shared memory, polled locally: a batch costs three
// posted socket writes and no round trips. As in hcq1, every submission
// first waits for the one before it. ────────────────────────────────────────

// HCQSignal.wait on the timeline. A GPU making no progress for 30 s (hcq1's
// default timeout) is lost (plan step C12: NVGpuLost, not an exit).
static void nvd_wait(uint64_t value) {
    auto start = std::chrono::steady_clock::now();
    for (uint64_t spin = 0; __atomic_load_n(g_nvd->signal, __ATOMIC_ACQUIRE) < value; ++spin) {
        if (spin % 1024) continue;
        auto waited = std::chrono::steady_clock::now() - start;
        if (waited > std::chrono::seconds(30)) {
            fprintf(stderr, "TinyGPU/NV: timeline wait timed out (want %llu, have %llu); GPU hung?\n",
                    (unsigned long long)value, (unsigned long long)__atomic_load_n(g_nvd->signal, __ATOMIC_ACQUIRE));
            throw NVGpuLost{true};
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
    // then "not in flight" once the whole frame is out. A frame cut mid-send loses the GPU with the flag still set, so its
    // keeper holds and sends TinyGPU.app nothing more (it would read those bytes as the rest of this frame).
    __atomic_store_n(&g_nvd->state[kNVDStateInFlight], 1, __ATOMIC_RELEASE);
    __atomic_store_n(&g_nvd->state[kNVDStateLastSubmitted], g_nvd->timeline - 1, __ATOMIC_RELEASE);
    std::string err;
    if (!tg_transport().bulk_write_frame(frame, 3, err)) {
        fprintf(stderr, "TinyGPU/NV: TinyGPU.app write %s: %s\n", tg_transport().lost() ? "cut mid-frame" : "refused", err.c_str());
        if (!tg_transport().lost()) __atomic_store_n(&g_nvd->state[kNVDStateInFlight], 0, __ATOMIC_RELEASE);   // nothing went out
        throw NVGpuLost{false};
    }
    __atomic_store_n(&g_nvd->state[kNVDStateInFlight], 0, __ATOMIC_RELEASE);
    ++f.put;
    ++g_nvd->pending;
}

// BeagleNVProgram.check_launch (NVProgram.__call__'s launch checks). A
// failed launch is reported, and the rest carry on (as the oracle's).
static bool nvd_check_launch(const NVDKernel& k, const int grid[3], const int block[3]) {
    long threads = (long)block[0] * block[1] * block[2];
    if (threads <= 1024 && threads <= (long)k.max_threads && grid[1] <= 65535 && grid[2] <= 65535 &&
        block[0] <= 1024 && block[1] <= 1024 && block[2] <= 64)
        return true;
    fprintf(stderr, "TinyGPU/NV: %s: invalid launch grid=(%d,%d,%d) block=(%d,%d,%d), not launched\n",
            k.name.c_str(), grid[0], grid[1], grid[2], block[0], block[1], block[2]);
    return false;
}

// The oracle's chained cmd_launch_batch, encoded here: one timeline wait and
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
    nv_test_kill("batch");   // plan step C10's test: killed with this batch on the GPU
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
        nv_test_kill("copy");   // plan step C10's test: killed with this copy on the GPU
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

static void nvd_unmap(NVDispatchState* d) {
    for (int i = 0; i < 4; ++i)
        if (d->maps[i]) { munmap(d->maps[i], d->map_sizes[i]); d->maps[i] = nullptr; }
    if (d->signal_fd >= 0) { close(d->signal_fd); d->signal_fd = -1; }
    if (d->state) { munmap(d->state, kNVDStateWords * 8); d->state = nullptr; }
    if (d->td.queues) { munmap(d->td.queues, d->td.queues_size); d->td.queues = nullptr; }
    if (d->state_fd >= 0) { close(d->state_fd); d->state_fd = -1; }
    if (d->td.queues_fd >= 0) { close(d->td.queues_fd); d->td.queues_fd = -1; }
}

// TODO.md plan step C6: this side's four buffers (kNVDBuffers) and its VRAM pool, allocated with tinygrad's memory manager as
// the daemon's allocator would (PCIIfaceBase.alloc), after the NVDevice; then plan step P3's WPR check on every VRAM allocation.
// Returns an empty string on success.
static std::string nvdOwnAllocations(NVDispatchState& d, uint64_t pool_mb) {
    NVMemoryManager& mm = *d.mem->mm;
    const uint64_t pool_size = pool_mb ? pool_mb << 20 : d.mem->dev_vram_size / 2;
    const char* which = pool_mb ? "BEAGLE_NV_DATA_MB: lower it" : "half the VRAM, the default";
    g_nvSetupOOM = false;
    try {
        NVDBuffer* hb[4] = { &d.h.cmdq, &d.h.kargs, &d.h.staging, &d.h.signal };
        for (int i = 0; i < 4; ++i) {
            const NVDBufferSpec& s = kNVDBuffers[i];
            NVBuffer b = nv_iface_alloc(mm, s.size, s.host, s.uncached, s.cpu_access, false, false, false, i == 3);
            *hb[i] = NVDBuffer{b.va_addr, b.size};
            d.maps[i] = b.view;
            d.map_sizes[i] = b.view_size;
            if (i == 3) d.signal_fd = b.fd;
        }
        memset(d.maps[3], 0, 16);   // TinyGPU.app leaves the DMA segment list here (the daemon cleared the same 16 bytes)
        if (d.h.kargs.va + d.h.kargs.size > (1ull << 40) || d.h.cmdq.va + d.h.cmdq.size > (1ull << 40))
            return "kernargs or pushbuffer buffer above 2^40";
        NVBuffer pool = nv_iface_alloc(mm, pool_size);   // as the daemon's pool_size
        d.rt.pool = NVDBuffer{pool.va_addr, pool.size};
    } catch (const TGPyError& e) {
        if (e.type == "MemoryError") {   // plan step M1
            g_nvSetupOOM = true;
            fprintf(stderr, "TinyGPU/NV: out of GPU memory: the GPU's %llu MiB of VRAM cannot hold the runtime's buffers and a %llu MiB "
                    "VRAM pool (%s)\n", (unsigned long long)(d.mem->dev_vram_size >> 20), (unsigned long long)(pool_size >> 20), which);
        }
        return "memory manager: " + e.py();
    }
    const uint64_t end = nv_vram_end(mm);
    char msg[200];
    snprintf(msg, sizeof(msg), "VRAM allocations end at 0x%llx, %s the WPR bound 0x%llx", (unsigned long long)end,
             end > d.mem->wpr_bound ? "above" : "<=", (unsigned long long)d.mem->wpr_bound);
    if (end > d.mem->wpr_bound) {   // plan step M1: the pool reaches GSP-RM's reserved region
        g_nvSetupOOM = true;
        fprintf(stderr, "TinyGPU/NV: out of GPU memory: a %llu MiB VRAM pool (%s) reaches GSP-RM's reserved region; at most %llu MiB fit\n",
                (unsigned long long)(pool_size >> 20), which, (unsigned long long)((d.mem->wpr_bound - (end - d.rt.pool.size)) >> 20));
        return std::string(msg) + ", where GSP-RM's reserved region starts (lower BEAGLE_NV_DATA_MB)";
    }
    fprintf(stderr, "TinyGPU/NV: C++ memory manager: buffers and pool, VRAM pool %llu MiB @ 0x%llx; %s\n",
            (unsigned long long)(d.rt.pool.size >> 20), (unsigned long long)d.rt.pool.va, msg);
    return "";
}

// TODO.md plan step P3: the state page, created before this side's first request to the GPU and shared with the crash guard
// only (TinyGPUHybridNVGuard.h). A POSIX shm segment, not a TinyGPU allocation (TinyGPU.app's MAP_SYSMEM_FD sequence is
// unchanged), unlinked at once so only the two processes' descriptors reach it. It records the boot's phase, whether a frame
// is in flight and the timeline value its work signals, and the GSP command queue's sequence number: if this process dies
// or loses the GPU, the guard holds, or waits for that value before it tears the GPU down.
static bool nvdStatePage(NVDispatchState& d, uint32_t seq, uint64_t phase, uint64_t in_flight) {
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
    __atomic_store_n(&st[kNVDStateSeq], seq, __ATOMIC_RELEASE);
    __atomic_store_n(&st[kNVDStateInFlight], in_flight, __ATOMIC_RELEASE);
    __atomic_store_n(&st[kNVDStatePhase], phase, __ATOMIC_RELEASE);
    d.state = st;
    d.state_fd = fd;
    return true;
}

// What the NVDevice's build starts from (plan steps C7-C11): the falcons' images (on COT, the FMC's) and NV_GSP's RM state,
// from the C++ boot (nvDispatchBoot). The GSP queues and the teardown's arguments are in d->td.
struct NVRMStart {
    NVFlcnImages im;
    NVCotImages cim;
    uint64_t wpr_meta_sysmem = 0;
    uint32_t next_handle = 0, gpfifo_class = 0, compute_class = 0, dma_class = 0, viddec_class = 0;
    bool gb2 = false;
};

// The build from the falcons' init_hw on (FWSEC-FRTS and booter_load, or the COT message), GSP-RM's init_hw and the golden
// image, then the NVDevice on this side's RM client. The state page's phase follows GSP-RM: gsp_init once it may run, dispatch
// after its INIT_DONE. "" or what failed.
static std::string nvdBuildDevice(NVDispatchState* d, const NVRMStart& s) {
    NVDTeardown& t = d->td;
    std::string err;
    d->bar0 = std::make_unique<NVBar0>(NVBar0{&tg_transport()});
    d->flcn = std::make_unique<NVFalcon>(*d->bar0, t.chip_id, t.cot);   // cot: the COT boot's falcon, then its teardown
    d->flcn->chip_name = t.chip_name;
    try {
        auto phase = [d](uint64_t p) { __atomic_store_n(&d->state[kNVDStatePhase], p, __ATOMIC_RELEASE); };
        if (t.cot)   // GSP-RM may run from sysmem from the COT message's first EMEM write on (nv_init_helper)
            d->flcn->cot_init_hw(s.cim, s.wpr_meta_sysmem, t.libos_args_sysmem, [&] { phase(kNVDPhaseGspInit); });
        else   // GSP-RM may run from sysmem once booter_load runs; a booter_load that failed left it unstarted
            d->flcn->init_hw(s.im, t.libos_args_sysmem, s.wpr_meta_sysmem, [&] { phase(kNVDPhaseGspInit); },
                             [&](uint32_t mbx0) { if (mbx0 != 0) phase(kNVDPhaseFlcnInit); });
        // the GSP queues, as init_hw's first statements build them: the status queue is GSP-RM's, which sets its header up once
        // booter_load (or the COT message) started it, just now; the constructor waits for it
        d->gsp = std::make_unique<NVGsp>(*d->bar0, *d->flcn, t.queues, t.cmdq_off, t.statq_off, t.queue_size, t.libos_args_sysmem, t.seq,
                                         d->flcn->wait_ms);
        d->gsp->after_rpc = [d](uint32_t seq) { __atomic_store_n(&d->state[kNVDStateSeq], seq, __ATOMIC_RELEASE); };
        d->rm = std::make_unique<NVRMClient>(*d->gsp, *d->mem->mm);
        NVRMClient& rm = *d->rm;
        rm.next_handle = s.next_handle; rm.gpfifo_class = s.gpfifo_class; rm.compute_class = s.compute_class; rm.dma_class = s.dma_class;
        rm.viddec_class = s.viddec_class; rm.gb2 = s.gb2;
        nv_gsp_init_hw(rm, t.cot, [&] { phase(kNVDPhaseDispatch); });   // cot: the COT boot's second BAR1 block
        nv_device_init(rm, d->dev);
    } catch (const TGPyError& e) {
        err = "building the NVDevice: " + e.py();
    } catch (const NVError& e) {
        err = "building the NVDevice: " + e.py();
    }
    return err;
}

// TODO.md plan steps C10 and C11's offline tests: BEAGLE_NV_TEST_KILL=<point> kills this process there with SIGKILL, as a crash
// would, so the crash guard's decisions can be checked (at "frame", with the state page saying a frame is in flight; at "batch"
// and "copy", right after the first launch batch's or copy's submission, with the GPU still running it; at "teardown", in this
// side's own teardown; at "boot_guard", "boot_sw" and "boot_built", in the boot).
// Never set outside the harness.
static void nv_test_kill(const char* point, NVDispatchState* d) {
    static const char* k = getenv("BEAGLE_NV_TEST_KILL");
    if (!k || strcmp(k, point) != 0) return;
    if (d && d->state && strcmp(point, "frame") == 0) __atomic_store_n(&d->state[kNVDStateInFlight], 1, __ATOMIC_RELEASE);
    tg_log("BEAGLE_NV_TEST_KILL=%s: SIGKILL", point);
    fflush(stderr);
    kill(getpid(), SIGKILL);
}

// Plan step C10: the crash guard's executable: BEAGLE_NV_GUARD (the test harness's), or beagle-tinygpu-guard next to this plugin
static std::string nv_guard_path() {
    const char* e = getenv("BEAGLE_NV_GUARD");
    if (e && e[0]) return e;
    Dl_info info;
    if (!dladdr((void*)&nv_guard_path, &info) || !info.dli_fname) return "";
    std::string so = info.dli_fname;
    return so.substr(0, so.rfind('/') + 1) + "beagle-tinygpu-guard";
}

// The crash guard's setup's rest (TinyGPUHybridNVGuard.h's kGuardSetupRest), from this side's teardown arguments (d.td) and
// buffers
static GuardSetup nvGuardSetup(const NVDispatchState& d, uint32_t kind) {
    const NVDTeardown& t = d.td;
    GuardSetup s{};
    s.magic = kGuardMagic;
    s.size = sizeof(s);
    s.kind = kind;
    s.nfds = guard_setup_nfds(kind);
    s.queues_size = t.queues_size; s.cmdq_off = t.cmdq_off; s.statq_off = t.statq_off; s.queue_size = t.queue_size;
    s.libos_args_sysmem = t.libos_args_sysmem; s.bar0_size = d.bar0_size; s.signal_size = d.h.signal.size;
    s.chip_id = t.chip_id; s.cot = t.cot; s.level0 = t.level0; s.parent_pid = (uint32_t)getpid(); s.images = t.images;
    snprintf(s.chip_name, sizeof(s.chip_name), "%s", t.chip_name.c_str());
    return s;
}

// TODO.md plan step C11: the crash guard, spawned before this side's first request to the GPU with what holding
// takes (the TinyGPU.app connection, its lock and the state page: TinyGPUHybridNVGuard.h's kGuardSetupHold). Before it said
// ready nothing went to the GPU, so a failed start ends it.
static std::string nvGuardBootStart(NVDispatchState& d) {
    const std::string path = nv_guard_path();
    if (path.empty()) return "the crash guard: no beagle-tinygpu-guard next to the plugin";
    int ctl = -1;
    pid_t pid = 0;
    std::string err = guard_spawn(path, ctl, pid);
    if (!err.empty()) return "the crash guard: " + err;
    GuardSetup s{};
    s.magic = kGuardMagic;
    s.size = sizeof(s);
    s.kind = kGuardSetupHold;
    s.nfds = guard_setup_nfds(kGuardSetupHold);
    s.parent_pid = (uint32_t)getpid();
    const int fds[3] = {d.tg_sock, tg_transport().lock_fd(), d.state_fd};
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
    d.guard_ctl = ctl;
    d.guard_pid = pid;
    fprintf(stderr, "TinyGPU/NV: level boot: the crash guard (pid %d) keeps the GPU from here, holding it until the NVDevice is built\n", (int)pid);
    tg_log("level boot: the crash guard (pid %d) keeps the GPU", (int)pid);
    return "";
}

// ... and, once the NVDevice is built, the rest (kGuardSetupRest): the GSP queues, the timeline and the teardown's arguments
static std::string nvGuardBootRest(NVDispatchState& d) {
    const GuardSetup s = nvGuardSetup(d, kGuardSetupRest);
    const int fds[2] = {d.td.queues_fd, d.signal_fd};
    const char m = 'S';
    if (write(d.guard_ctl, &m, 1) != 1 || !guard_send_setup(d.guard_ctl, s, fds)) return std::string("the crash guard's setup rest: ") + strerror(errno);
    return "";
}

static std::string nvdCppTeardown(NVDispatchState& d, double& secs, std::string& report);

// This side's teardown report (nvdCppTeardown's; none means its teardown did not run), then clean or hold to the crash guard
// (plan step C10). True if the guard holds.
static bool nvGuardReport(int guard_ctl, pid_t guard_pid, const std::string& cpp_fini, const std::string& cpp_report) {
    const bool hold = cpp_fini.empty() || nv_json_bool(cpp_fini, "hold");
    std::string resp = "{\"ok\": true" + (cpp_report.size() > 2 ? ", " + cpp_report.substr(1, cpp_report.size() - 2) : std::string()) +
                       (hold ? ", \"hold\": true, \"pid\": " + std::to_string(guard_pid) : std::string()) + "}";
    nv_report_unload(resp, "the crash guard");
    const char m = hold ? 'H' : 'C';
    if (write(guard_ctl, &m, 1) != 1)
        fprintf(stderr, "TinyGPU/NV: the crash guard (pid %d) did not take the %s: it decides as at a crash\n", (int)guard_pid,
                hold ? "hold" : "clean exit");
    return hold;
}

// A boot that failed after GSP-RM's INIT_DONE (plan step C13's follow-up: the GSP refused the NVDevice's RM calls or the golden
// image's, or the VRAM pool was refused) with nothing left running on the GPU: the NVDevice's setup work, if it was
// submitted, completed. GSP-RM then answers, so this side can unload it and run NVIDIA's teardown, as the daemon did from the
// plugin's count, instead of leaving the guard to hold (a power cycle).
static bool nvd_boot_failed_idle(NVDispatchState& d) {
    if (!d.state || __atomic_load_n(&d.state[kNVDStatePhase], __ATOMIC_ACQUIRE) != kNVDPhaseDispatch || !d.gsp || !d.flcn) return false;
    NVDeviceState& dev = d.dev;
    return dev.timeline_value == 1 || *nv_signal_host(dev, dev.timeline_signal) >= dev.timeline_value - 1;
}

// TODO.md plan step C11 (level boot): no daemon and no Python. This side boots the GPU with the C++ boot (TinyGPUHybridNVBoot.h:
// NVDev.__init__'s software half as the oracle's daemon runs it, nv_init_helper's patches included), then builds the NVDevice
// (nvdBuildDevice) and allocates its buffers. The crash guard keeps the GPU from before the first request to it. The state page
// says a frame is in flight until the NVDevice is built and the guard has the rest of its setup: a death before the falcons'
// boot closes (phase flcn_init), one after it holds. A failed boot ends the guard's socketpair as a death would (plan step C12),
// after this side's own unload and teardown if GSP-RM answers and the GPU is idle (nvd_boot_failed_idle).
static NVDispatchState* nvDispatchBoot(int tg_sock) {
    auto t0 = nv_profile_start();
    NVDispatchState* d = new NVDispatchState;
    d->tg_sock = tg_sock;
    NVDTeardown& t = d->td;
    std::string err = nvdStatePage(*d, 0, kNVDPhaseFlcnInit, 1) ? "" : "no state page";
    if (err.empty()) err = nvGuardBootStart(*d);
    if (err.empty()) nv_test_kill("boot_guard");
    NVBootDev bd;
    bd.t = &tg_transport();
    NVRMStart st;
    NVBootMem fmc_args, fmc_image;
    NVGspBoot gb;
    if (err.empty()) {
        try {
            nv_boot_pci(bd);
            nv_boot_early_ip_init(bd);
            nv_boot_early_mmu_init(bd);
            nv_boot_end_booting(bd);
            if (bd.fmc_boot) nv_boot_cot_init_sw(bd, st.cim, fmc_args, fmc_image);
            else nv_boot_flcn_init_sw(bd, st.im, t.images);
            if (bd.recover) {   // plan step P4: a warm GPU, torn down before GSP-RM boots
                fprintf(stderr, "TinyGPU/NV: a warm GPU (WPR2_HI=0x%08x, the GSP %s: MAILBOX0=0x%08x, RISCV_CPUCTL=0x%08x): NVIDIA's "
                        "teardown first (BEAGLE_NV_RECOVER=1)\n", bd.warm_wpr2_hi, bd.warm_mailbox0 == 0x80000000 ? "suspended" : "halted",
                        bd.warm_mailbox0, bd.warm_cpuctl);
                NVFiniDiag rd;
                try { nv_boot_recover(bd, t.images, rd); }
                catch (...) {
                    if (rd.teardown_ran) fprintf(stderr, "TinyGPU/NV: the teardown at boot: %s\n", rd.json().c_str());
                    throw;
                }
                fprintf(stderr, "TinyGPU/NV: the teardown at boot: %s; WPR2 is down, so the boot goes on\n", rd.td_result.c_str());
            }
            nv_boot_gsp_init_sw(bd, gb, bd.fmc_boot ? nullptr : &st.im);
        } catch (const NVError& e) {
            err = "the C++ boot: " + e.py();
        } catch (const TGPyError& e) {
            err = "the C++ boot: " + e.py();
        } catch (const std::exception& e) {
            err = std::string("the C++ boot: ") + e.what();
        }
    }
    if (err.empty()) {   // what the NVDevice's build and the teardown take from the boot
        t.queues = gb.queues.sys.view; t.queues_fd = gb.queues_fd; t.queues_size = gb.queues.sys.mapped_size;
        t.cmdq_off = gb.pt_size; t.statq_off = gb.pt_size + gb.queue_size; t.queue_size = gb.queue_size;
        t.seq = gb.cmd_q->seq; t.libos_args_sysmem = gb.libos_args_sysmem; t.chip_id = bd.chip_id; t.cot = bd.fmc_boot;
        t.chip_name = bd.chip_name;
        const char* ul = getenv("BEAGLE_NV_UNLOAD_LEVEL");
        t.level0 = ul && strcmp(ul, "0") == 0;
        __atomic_store_n(&d->state[kNVDStateSeq], t.seq, __ATOMIC_RELEASE);
        st.wpr_meta_sysmem = gb.wpr_meta_sysmem;
        st.next_handle = gb.next_handle; st.gpfifo_class = gb.gpfifo_class; st.compute_class = gb.compute_class; st.dma_class = gb.dma_class;
        st.viddec_class = gb.viddec_class; st.gb2 = bd.chip_name.compare(0, 3, "GB2") == 0;
        d->mem = std::move(bd.mem);
        d->mem->wpr_bound = bd.fmc_boot ? bd.vram_size - (512ull << 20) : gb.meta.gspFwRsvdStart;   // the daemon's _wpr_bound
        uint64_t bar0_addr = 0;
        std::string e;
        tg_transport().bar_info(0, bar0_addr, d->bar0_size, e);   // cached: the boot mapped BAR0 first
        nv_test_kill("boot_sw");
        err = nvdBuildDevice(d, st);
    }
    uint64_t pool_mb = 0;
    if (const char* mb = getenv("BEAGLE_NV_DATA_MB")) pool_mb = strtoull(mb, nullptr, 10);
    if (err.empty()) err = nvdOwnAllocations(*d, pool_mb);
    if (err.empty()) {
        nvd_handoff_from_device(d->dev, d->rm->compute_class, d->h);
        nvd_runtime_from_device(d->dev, d->rm->compute_class, d->rt);
        nv_test_kill("boot_built");
        err = nvGuardBootRest(*d);
    }
    if (!err.empty()) {
        fprintf(stderr, "TinyGPU/NV: level boot: %s\n", err.c_str());
        const bool started = d->state && __atomic_load_n(&d->state[kNVDStatePhase], __ATOMIC_ACQUIRE) != kNVDPhaseFlcnInit;
        bool closing = false;   // the guard closes and exits, with its copies of the connection and nv_usb4.lock
        if (d->guard_ctl >= 0 && !started) {   // GSP-RM never started: nothing to unload, the guard may close
            const char c = 'N';
            closing = write(d->guard_ctl, &c, 1) == 1;
        } else if (d->guard_ctl >= 0 && nvd_boot_failed_idle(*d)) {   // GSP-RM answers: unload it here, then clean or hold
            double secs = 0;
            std::string fini, report;
            try { fini = nvdCppTeardown(*d, secs, report); }
            catch (...) {}   // a transport failure mid-teardown: the state page says teardown, so the guard holds
            if (!fini.empty()) closing = !nvGuardReport(d->guard_ctl, d->guard_pid, fini, report);
        }
        // Plan step C12: the guard decides now, from the state page (it holds once GSP-RM may run), and the host goes on. The
        // boot's own mappings are this process's views only: the guard keeps the connection, and the sysmem behind them. A
        // guard that closes is waited for, as at exit (nvFiniDevice), so that the next instance finds its lock free.
        if (d->guard_ctl >= 0) close(d->guard_ctl);
        for (int i = 0; i < 50 && closing && waitpid(d->guard_pid, nullptr, WNOHANG) == 0; ++i) usleep(100000);
        nvd_unmap(d);
        delete d;
        return nullptr;
    }
    d->cmdq = (uint8_t*)d->maps[0];
    d->kargs = (uint8_t*)d->maps[1];
    d->staging = (uint8_t*)d->maps[2];
    d->signal = (uint64_t*)d->maps[3];
    t.ready = true;
    nv_profile_end("handoff", t0);
    tg_transport().marker(TGM_HANDOFF, 1);
    __atomic_store_n(&d->state[kNVDStateInFlight], 0, __ATOMIC_RELEASE);   // built: from here nvd_submit keeps the word
    fprintf(stderr, "TinyGPU/NV: C++ runtime: built the NVDevice after the C++ boot, with no daemon (level boot: %s, QMD v%u, VRAM pool %llu MiB)\n",
            d->dev.arch.c_str(), d->h.qmd_ver, (unsigned long long)(d->rt.pool.size >> 20));
    fprintf(stderr, "TinyGPU/NV: C++ teardown: the GSP unload%s run here at exit\n",
            t.cot ? " and the RISC-V halt wait (COT)" : t.images.present ? " and NVIDIA's teardown" : "");
    return d;
}

// TODO.md plan step C5: the GPU teardown at fini from this side, on tinygrad's ported RPC queue and falcon primitives
// (TinyGPUHybridNVGsp.h, TinyGPUHybridNVFalcon.h), in NVDev.fini's order: the GSP unload (nv_init_helper's suspend wait
// included), then, only if the GSP confirmed it, NVIDIA's teardown; on the COT boot (plan step B2), NV_FLCN_COT.fini_hw's
// wait for the GSP's RISC-V core to halt instead. The state page says so first, so the crash guard, if this process dies
// meanwhile, holds instead of touching the GPU. Returns the report (in the daemon's fini format, which nv_report_unload
// reads), with hold true if the GSP did not confirm its unload, or on COT if its RISC-V core did not halt (the hold rule).
static std::string nvdCppTeardown(NVDispatchState& d, double& secs, std::string& report) {
    auto t0 = std::chrono::steady_clock::now();
    NVDTeardown& t = d.td;
    __atomic_store_n(&d.state[kNVDStatePhase], kNVDPhaseTeardown, __ATOMIC_RELEASE);
    nv_test_kill("teardown");
    tg_log("C++ GPU teardown: the %s unload RPC (seq %u), the suspend wait%s", t.level0 ? "LEVEL_0" : "FAST_UNLOAD",
           d.gsp->cmd_q.seq, t.cot ? ", then the RISC-V halt wait (COT)" : t.images.present ? ", then NVIDIA's teardown" :
           "; the teardown is off");
    NVFalcon& flcn = *d.flcn;   // the NVDevice's, from the boot on
    NVFiniDiag diag;
    bool hold = false;
    try {
        NVGsp& gsp = *d.gsp;
        try { gsp.fini_hw(diag, t.level0); }
        catch (const NVError& e) {   // the RPC failed or timed out: the GSP may be live, so no falcon is touched
            tg_log("the GSP unload failed: %s", e.py().c_str());
            fprintf(stderr, "TinyGPU/NV: GPU teardown failed: the GSP unload: %s\n", e.py().c_str());
            hold = true;
        }
        if (!hold && flcn.cot) flcn.cot_fini_hw(diag);
        else if (!hold) flcn.fini_hw(diag, t.images);
    } catch (const NVError& e) {   // outside what nv_init_helper tolerates: a confirmed unload still makes closing safe
        tg_log("the C++ GPU teardown failed: %s", e.py().c_str());
        fprintf(stderr, "TinyGPU/NV: GPU teardown failed: %s\n", e.py().c_str());
        hold = !diag.unload_ok;
    }
    if (diag.cot && !diag.halted) hold = true;   // COT: the FMC and the ACR may still use the boot structures in sysmem
    if (!diag.unload_ok) hold = true;            // the oracle's rule too (_cpp_fini); the crash guard's comes from here
    secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    tg_log("C++ GPU teardown done in %.3f s: %s", secs, diag.json().c_str());
    report = diag.json();
    return std::string("{\"cmd\":\"fini\",\"cpp_teardown\":true,\"hold\":") + (hold ? "true" : "false") + ",\"diag\":" + diag.json() + "}";
}

// ── Program loading and allocation ─────────────────────────────────────────

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
// program to the same size). Each instance loads its own programs, at its own
// lib_va (plan step P5), which NvFini frees (plan step C14); local memory only
// grows, as dev.slm_per_thread does:
// a new block and setup when this instance needs more than the GPU has,
// nothing otherwise. (tinygrad's _realloc hands the old block to its LRU
// cache without a synchronize; the setup waits for all earlier work, and the
// pool never reuses the old block.)
static bool nvdLoadPrograms(const NVDElf& elf, const std::vector<std::string>& names, std::map<std::string, NVDKernel>& kernels,
                            uint64_t& lib_va, bool& oom) {
    oom = false;
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
        lib_va = p.lib_va = nvd_pool_alloc(d.rt.pool, d.pool_pos, nvd_round_up(elf.image.size(), 0x1000) + 0x1000, &d.pool_free);  // NVProgram's lib_gpu
        if (grow) {
            local_mem_size = nvd_local_mem_size(d.rt, p.slm_per_thread, tpc_bytes);
            local_mem = nvd_pool_alloc(d.rt.pool, d.pool_pos, local_mem_size, &d.pool_free);
        }
        if (!p.lib_va || (grow && !local_mem)) {
            err = "the VRAM pool cannot hold the program image and local memory";
            oom = true;   // plan step M1
            fprintf(stderr, "TinyGPU/NV: out of GPU memory: %llu MiB left of the %llu MiB VRAM pool (BEAGLE_NV_DATA_MB sets it, by default "
                    "half the VRAM)\n", (unsigned long long)(nvd_pool_left(d) >> 20), (unsigned long long)(d.rt.pool.size >> 20));
        }
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
// kernel in it. Its programs load after the handoff (nvRuntimePrograms). False: none serves them (the instance fails).
static bool nvRuntimeCubin(NVInstance& in, int paddedStateCount, bool dp, const std::string& arch, NVDElf& cubin) {
    const TinyGPUNVCubin* c = nullptr;
    std::string err = nvd_find_cubin(kTinyGPUNVCubins, sizeof(kTinyGPUNVCubins) / sizeof(kTinyGPUNVCubins[0]),
                                     paddedStateCount, dp, arch, c);
    if (err.empty()) err = nvd_elf_load(c->begin, (size_t)(c->end - c->begin), 128, cubin);  // NVProgram's force_section_align
    if (!err.empty()) {
        fprintf(stderr, "TinyGPU/NV: C++ runtime: %s\n", err.c_str());
        return false;
    }
    for (const std::string& kname : nvd_kernel_names(cubin)) in.kernels[kname] = new NVKernelHandle{kname};
    fprintf(stderr, "TinyGPU/NV: C++ runtime: embedded cubin %s_%d %s (%zu bytes, %zu kernels; ptxas %s)\n", dp ? "DP" : "SP",
            paddedStateCount, arch.c_str(), (size_t)(c->end - c->begin), in.kernels.size(), TINYGPU_CUBINS_STAMP);
    return true;
}

// TODO.md plan step C16: double precision on NV, which the build's cubins include (DP_4 ... DP_256 for each architecture);
// the GPU computes it natively, at a lower rate
bool NvSupportsDouble() {
    for (const TinyGPUNVCubin& c : kTinyGPUNVCubins)
        if (c.dp) return true;
    return false;
}

// Each handle's launch template: the instance's own programs.
static void nvLinkTemplates(NVInstance& in, const std::map<std::string, NVDKernel>& templates) {
    for (auto& kv : in.kernels) {
        auto it = templates.find(kv.first);
        kv.second->tmpl = (it != templates.end()) ? &it->second : nullptr;
    }
}

static bool nvRuntimePrograms(NVInstance& in, const NVDElf& cubin) {
    std::vector<std::string> names;
    for (auto& kv : in.kernels) names.push_back(kv.first);
    if (!nvdLoadPrograms(cubin, names, in.templates, in.lib_va, in.oom)) return false;
    nvLinkTemplates(in, in.templates);
    return true;
}

// Every firmware file this GPU's boot reads (TinyGPUFirmwareManifest.h), located, or downloaded into BEAGLE's cache
// (TinyGPUFirmware.h; since 2026-10-01, the user's request), before anything is written to the GPU: the chip family comes
// from the probe's PCI device ID, as nv_boot_gpu tells them apart. "" or why not.
static std::string nv_fw_prefetch(uint16_t device_id) {
    const char* family = device_id >= 0x2b00 ? "gb202" : "ad102";
    const char* teardown = getenv("BEAGLE_NV_TEARDOWN");
    const bool unload = !(teardown && strcmp(teardown, "0") == 0);
    for (const nvfw::TGFirmware& f : nvfw::kFirmware) {
        if (strcmp(f.chip, family) != 0 || (!unload && strcmp(f.role, "booter_unload") == 0)) continue;
        TGFirmwareFile file;
        const std::string err = tg_fw_locate(f, file);
        if (!err.empty()) return err;
    }
    return "";
}

// TODO.md plan step C11: the GPU's setup (nvDispatchBoot): the C++ boot and the NVDevice; NvSetDevice then loads the programs
// from the embedded cubin, as for an instance that shares the boot. Null if the boot failed, when the crash guard has the GPU
// already (plan step C12).
static NVHybridState* nvBootSetup(int tg_fd) {
    fprintf(stderr, "TinyGPU/NV: level boot: the C++ boot, with no daemon\n");
    auto t0 = nv_profile_start();
    g_nvd = nvDispatchBoot(tg_fd);
    nv_profile_end("boot", t0);
    if (!g_nvd) return nullptr;
    NVHybridState* g = new NVHybridState{};
    g->owner_pid = getpid();
    g->arch = g_nvd->dev.arch;
    fflush(stderr);
    return g;
}

// ── GPUInterface entry points ─────────────────────────────────────────────────

// The GPU teardown at exit (plan step P5): the GPU finishes its work, then the GSP unload and NVIDIA's teardown (nvdCppTeardown)
// and their report, then the clean or the hold to the crash guard. The TinyGPU.app connection is closed last, with
// nv_usb4.lock; a guard that holds keeps its own copies of both.
static void nvFiniDevice() {
    if (!g_nv) return;
    std::string cpp_fini, cpp_report;   // this side's GPU teardown and its report (plan step C5)
    double cpp_secs = 0;
    int guard_ctl = -1;
    pid_t guard_pid = 0;
    if (g_nvd) {  // let the GPU finish before it is torn down
        nvd_idle();
        nv_test_kill("idle", g_nvd);
        nv_test_kill("frame", g_nvd);
        guard_ctl = g_nvd->guard_ctl;
        guard_pid = g_nvd->guard_pid;
        if (g_nvd->td.ready && guard_ctl >= 0) {
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
    if (guard_ctl >= 0) {   // plan step C10: this side's report, then clean or hold to the guard
        const bool hold = nvGuardReport(guard_ctl, guard_pid, cpp_fini, cpp_report);
        close(guard_ctl);
        for (int i = 0; i < 50 && !hold && waitpid(guard_pid, nullptr, WNOHANG) == 0; ++i) usleep(100000);
        if (nv_profile_enabled()) fprintf(stderr, "TinyGPU/NV: fini %.3f s: the C++ GSP unload and teardown\n", cpp_secs);
    }
    delete g_nv;
    g_nv = nullptr;
    tg_transport().close();
}

// Plan step P5: the C++ runtime's GPU outlives its instances, as tinygrad's devices do (device.py finalizes them at
// exit): a connection's sysmem cannot be freed, so later instances share this boot instead of booting again. A thread
// still holding the lock after 35 s (past the 30 s timeline timeout) may be mid-frame: then nothing is sent, and the
// crash guard, at the end of this process, tears the GPU down or holds, as the state page says. A child forked after the
// boot inherits this handler and the TinyGPU.app connection, and returns at once: the GPU is its parent's (checked before
// the lock, which a fork can copy held).
static void nvAtExit() {
    if (!g_nv || g_nv->owner_pid != getpid() || g_nv->lost) return;   // a lost GPU is its keeper's already (plan step C12)
    std::unique_lock<std::recursive_timed_mutex> lk(nv_mutex(), std::defer_lock);
    if (!lk.try_lock_for(std::chrono::seconds(35))) {
        fprintf(stderr, "TinyGPU/NV: another thread is still using the GPU at exit; not tearing it down from here: the crash "
                "guard does at the end of this process, or holds, as the state page says\n");
        return;
    }
    try { nvFiniDevice(); } catch (const NVGpuLost& e) { nv_gpu_lost(e.hung); }   // hung before the teardown
}

// Plan step P5: an instance created while another has the GPU booted shares that boot, since TinyGPU.app serves one
// connection at a time and a new one from here would wait forever.
int NvAttachShared(GPUInterface* self) {
    std::lock_guard<std::recursive_timed_mutex> lk(nv_mutex());
    if (!g_nv) return 0;
    if (g_nv->lost) {   // plan step C12: its keeper has it; this process boots no other
        fprintf(stderr, "TinyGPU/NV: the GPU was lost earlier in this process; no instance can use it until the process restarts\n");
        return -1;
    }
    self->nvGspState = new NVInstance;
    return 1;
}

void NvSetDevice(GPUInterface* self, int paddedStateCount, int categoryCount,
                  int patternCount, int unpaddedPatternCount, int tipCount, long flags) {
    std::lock_guard<std::recursive_timed_mutex> lk(nv_mutex());
    NVInstance* shared = (NVInstance*)self->nvGspState;   // NvAttachShared's: the GPU is booted already (plan step P5)
    // The boot takes Initialize()'s TinyGPU.app connection, which the GPU then keeps until exit (plan step P5; the GPUInterface
    // destructor closes it if the boot failed).
    const int tg_fd = self->tgpuSock;

    self->InitializeKernelResource(paddedStateCount, (flags & BEAGLE_FLAG_PRECISION_DOUBLE) != 0);
    self->supportDoublePrecision = ((flags & BEAGLE_FLAG_PRECISION_DOUBLE) != 0);
    if (self->kernelResource) {
        self->kernelResource->categoryCount        = categoryCount;
        self->kernelResource->patternCount         = patternCount;
        self->kernelResource->unpaddedPatternCount = unpaddedPatternCount;
        self->kernelResource->flags                = flags;
    }

    NVInstance* in = shared ? shared : new NVInstance;
    self->nvGspState = in;
    const bool boot = !shared && tg_fd >= 0 && nv_boot_gpu();
    if (!shared && !boot)   // plan step C13c: there is no other path
        fprintf(stderr, "TinyGPU/NV: this GPU (PCI device ID %04x) is not one BEAGLE boots: Ada (0x26xx-0x28xx) and Blackwell "
                "(0x2bxx-0x2dxx, 0x2fxx) only\n", tg_pci_device_id());
    try {
        if (boot) {
            const std::string fw = nv_fw_prefetch(tg_pci_device_id());
            if (fw.empty()) {
                g_nvSetupOOM = false;
                g_nv = nvBootSetup(tg_fd);
                in->oom = !g_nv && g_nvSetupOOM;   // plan step M1: its VRAM pool did not fit
            }
            else fprintf(stderr, "%s\nTinyGPU/NV: not booting: nothing was written to the GPU\n", fw.c_str());
        }
        NVDElf cubin;   // the programs of an instance that shares the boot, or of the boot's first
        if (!nv_failed(*in) &&
            (!nvRuntimeCubin(*in, paddedStateCount, self->supportDoublePrecision, g_nv->arch, cubin) || !nvRuntimePrograms(*in, cubin)))
            in->failed = true;
    } catch (const NVGpuLost& e) {   // a hang or a broken stream in the setup's GPU work
        nv_gpu_lost(e.hung);
    }
    if (nv_failed(*in)) {   // plan step C12: beagleCreateInstance returns an error (BeagleGPUImpl), and the host goes on
        in->failed = true;
        fprintf(stderr, "TinyGPU/NV: this instance's setup failed%s\n", g_nv && g_nv->lost ? " (the GPU is lost)" : boot && !g_nv ? " (the boot failed)" : "");
    }
    if (shared || !g_nvd) return;
    self->tgpuSock = -1;          // plan step P5: the GPU's connection now (g_nvd->tg_sock), which outlives this instance
    if (atexit(nvAtExit) != 0)
        fprintf(stderr, "TinyGPU/NV: atexit failed; the GPU is torn down only by the crash guard, at the end of this process\n");
}

GPUFunction NvGetFunction(GPUInterface* self, const char* name) {
    NVInstance* in = (NVInstance*)self->nvGspState;
    if (!in) return nullptr;
    auto it = in->kernels.find(name);
    if (it != in->kernels.end()) return it->second;
    if (!in->failed) fprintf(stderr, "TinyGPU/NV: GetFunction(%s): kernel not found in precompiled cache; this instance fails\n", name);
    in->failed = true;   // plan step C12: beagleCreateInstance returns an error, not an exit
    return nullptr;
}

void NvSynchronizeHost(GPUInterface* self) {
    std::lock_guard<std::recursive_timed_mutex> lk(nv_mutex());
    NVInstance* in = (NVInstance*)self->nvGspState;
    if (!in || nv_failed(*in)) return;
    auto t0 = nv_profile_start();
    try {
        nvFlushLaunchQueue(*in);  // otherwise queued-but-unsent launches wouldn't be submitted yet to wait for
        nvd_idle();
        nv_profile_end("sync", t0);
    } catch (const NVGpuLost& e) {   // plan step C12
        nv_gpu_lost(e.hung);
    }
}

GPUPtr NvAllocateMemory(GPUInterface* self, size_t sz) {
    std::lock_guard<std::recursive_timed_mutex> lk(nv_mutex());
    NVInstance* in = (NVInstance*)self->nvGspState;
    if (!in || nv_failed(*in)) return 0;
    uint64_t va = nvd_pool_alloc(g_nvd->rt.pool, g_nvd->pool_pos, sz, &g_nvd->pool_free);
    if (!va) {   // BEAGLE does not check: address 0 would reach the GPU
        fprintf(stderr, "TinyGPU/NV: out of GPU memory: an allocation of %.1f MiB, with %.1f MiB left of the %llu MiB VRAM pool "
                "(BEAGLE_NV_DATA_MB sets it, by default half the VRAM); this instance fails\n", sz / 1048576.0,
                nvd_pool_left(*g_nvd) / 1048576.0, (unsigned long long)(g_nvd->rt.pool.size >> 20));
        in->failed = true;   // plan step C12: so nothing of it reaches the GPU, and BeagleGPUImpl returns an error,
        in->oom = true;      // BEAGLE_ERROR_OUT_OF_MEMORY (plan step M1)
    }
    return (GPUPtr)va;
}

// Plan step C14: the block goes back to the pool (TinyGPUPool.h), once this instance's queued launches, which may use it, are
// submitted. An address no allocation returned is ignored.
void NvFreeMemory(GPUInterface* self, GPUPtr p) {
    std::lock_guard<std::recursive_timed_mutex> lk(nv_mutex());
    NVInstance* in = (NVInstance*)self->nvGspState;
    if (!g_nvd || !p) return;
    try {
        if (in) nvFlushLaunchQueue(*in);
    } catch (const NVGpuLost& e) {   // plan step C12
        nv_gpu_lost(e.hung);
    }
    g_nvd->pool_free.release((uint64_t)p - g_nvd->rt.pool.va, g_nvd->pool_pos);
}

void NvMemcpyHostToDevice(GPUInterface* self, GPUPtr dst, const void* src, size_t sz) {
    std::lock_guard<std::recursive_timed_mutex> lk(nv_mutex());
    NVInstance* in = (NVInstance*)self->nvGspState;
    if (!in || nv_failed(*in) || !src || !sz) return;
    try {
        nvFlushLaunchQueue(*in);  // preserve ordering: queued launches must be submitted before this write
        nvdCopyIn(dst, src, sz);
    } catch (const NVGpuLost& e) {   // plan step C12
        nv_gpu_lost(e.hung);
    }
}

void NvMemcpyDeviceToHost(GPUInterface* self, void* dst, const GPUPtr src, size_t sz) {
    std::lock_guard<std::recursive_timed_mutex> lk(nv_mutex());
    NVInstance* in = (NVInstance*)self->nvGspState;
    if (!in || !dst || !sz) return;
    // plan step C12: a failed instance or a lost GPU reads back NaN (all bits set), so nothing it returns looks like a result
    if (nv_failed(*in)) { memset(dst, 0xff, sz); return; }
    try {
        nvFlushLaunchQueue(*in);  // preserve ordering: queued launches must complete before this read
        nvdCopyOut(dst, src, sz);
    } catch (const NVGpuLost& e) {
        nv_gpu_lost(e.hung);
        memset(dst, 0xff, sz);
    }
}

// TODO.md plan step C12, for BeagleGPUImpl (GPUInterface::GetDeviceLost): true once nothing this instance computes can be
// trusted: its setup or an allocation failed, or the GPU is lost
bool NvDeviceLost(GPUInterface* self) {
    std::lock_guard<std::recursive_timed_mutex> lk(nv_mutex());
    NVInstance* in = (NVInstance*)self->nvGspState;
    return !in || nv_failed(*in);
}

// Plan step M1, for BeagleGPUImpl (GPUInterface::GetOutOfMemory): ... because the GPU's memory did not suffice
bool NvOutOfMemory(GPUInterface* self) {
    std::lock_guard<std::recursive_timed_mutex> lk(nv_mutex());
    NVInstance* in = (NVInstance*)self->nvGspState;
    return in && in->oom;
}

size_t NvGetAvailableMemory() {
    std::lock_guard<std::recursive_timed_mutex> lk(nv_mutex());
    return g_nvd ? (size_t)nvd_pool_left(*g_nvd) : 0;
}

// Releases this instance; the GPU stays for later instances until exit (nvAtExit; plan step P5).
void NvFini(GPUInterface* self) {
    std::lock_guard<std::recursive_timed_mutex> lk(nv_mutex());
    NVInstance* in = (NVInstance*)self->nvGspState;
    if (!in) return;
    self->nvGspState = nullptr;
    try {
        nvFlushLaunchQueue(*in);  // don't silently drop queued-but-unsent launches
    } catch (const NVGpuLost& e) {   // plan step C12
        nv_gpu_lost(e.hung);
    }
    if (g_nvd && in->lib_va) g_nvd->pool_free.release(in->lib_va - g_nvd->rt.pool.va, g_nvd->pool_pos);   // after its launches (plan step C14)
    for (auto& kv : in->kernels) {   // reported with the GPU teardown, so not once it is done (nor after static destructors)
        if (g_nv && kv.second->launches) g_nvKernelLaunches[kv.first] += kv.second->launches;
        delete kv.second;
    }
    delete in;
}

void NvLaunchKernelImpl(GPUInterface* self, GPUFunction fn, Dim3Int block, Dim3Int grid,
                         int nPtr, int nTotal, GPUPtr* ptrs, unsigned int* ints) {
    NVInstance* in = (NVInstance*)self->nvGspState;
    if (!in || !fn || in->failed) return;
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
