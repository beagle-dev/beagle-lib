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
#include <string>
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
#include "libhmsbeagle/GPU/TinyGPUHybridSocket.h"
#include "libhmsbeagle/GPU/TinyGPUHybridNVDispatch.h"

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
static bool nv_json_ok(const std::string& js) {
    auto p = js.find("\"ok\":");
    if (p == std::string::npos) return false;
    p += 5;
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
};

struct NVKernelHandle {
    std::string name;
    const NVDKernel* tmpl = nullptr;  // C++ dispatch: this kernel's handoff template
};

static NVHybridState* g_nv = nullptr;
static std::map<std::string, NVKernelHandle*> g_nvKernels;

// C++ dispatch state (see "C++ dispatch" below); null on the daemon path.
struct NVDispatchState {
    NVDHandoff h;
    int tg_sock = -1;            // the plugin's TinyGPU.app connection, shared with the daemon
    void* maps[4] = {};          // host mappings of h.cmdq, h.kargs, h.staging, h.signal
    uint8_t *cmdq = nullptr, *kargs = nullptr, *staging = nullptr;
    uint64_t* signal = nullptr;  // timeline semaphore
    uint64_t cmdq_pos = 0, kargs_pos = 0, staging_pos = 0;
    uint64_t timeline = 1;       // value the next submission signals; every earlier value is submitted
    uint64_t pending = 0;        // submissions since the GPU was last seen idle
};
static NVDispatchState* g_nvd = nullptr;

static bool nv_cpp_dispatch() {
    static const bool on = [] { const char* v = getenv("BEAGLE_NV_CPP_DISPATCH"); return v && strcmp(v, "0") != 0; }();
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
static std::vector<NVPendingLaunch> g_nvPendingLaunches;

static void nvdFlushLaunches();

static void nvFlushLaunchQueue() {
    if (!g_nv || g_nvPendingLaunches.empty()) return;
    if (g_nvd) { nvdFlushLaunches(); return; }
    auto t0 = nv_profile_start();
    std::string cmd = "{\"cmd\":\"launch_batch\",\"launches\":[";
    for (size_t li = 0; li < g_nvPendingLaunches.size(); ++li) {
        const NVPendingLaunch& pl = g_nvPendingLaunches[li];
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
    size_t n = g_nvPendingLaunches.size();
    g_nvPendingLaunches.clear();

    nv_send_msg(g_nv->cmd_sock, cmd);
    std::string resp = nv_recv_msg(g_nv->cmd_sock);
    nv_profile_end("launch_batch", t0);
    g_nvProfileLaunches += (long long)n;
    if (resp.empty() || !nv_json_ok(resp))
        fprintf(stderr, "TinyGPU/NV: launch_batch(%zu kernels) failed: %s\n", n, resp.c_str());
}

[[noreturn]] static void nv_safe_exit(int code) {
    fflush(stderr);
    if (g_nv) {
        if (g_nv->cmd_sock >= 0) {
            nv_send_msg(g_nv->cmd_sock, "{\"cmd\":\"fini\"}");
            nv_recv_msg(g_nv->cmd_sock);  // best-effort ack, ignore content
        }
        if (g_nv->daemon_pid > 0) {
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
            nv_safe_exit(1);
        }
        if (waited > std::chrono::milliseconds(2)) usleep(20);
    }
}

static void nvd_idle() {
    nvd_wait(g_nvd->timeline - 1);
    g_nvd->pending = 0;
}

// Bump allocation in a shared ring. Wrapping around first waits until the GPU
// is done with everything submitted, so no live region is overwritten.
static uint64_t nvd_alloc(uint64_t& pos, uint64_t size, uint64_t need, uint64_t align) {
    uint64_t p = (pos + align - 1) & ~(align - 1);
    if (p + need > size) { nvd_idle(); p = 0; }
    pos = p + need;
    return p;
}

// A posted TinyGPU.app MMIO_WRITE (TinyGPUHybridSocket.h's tg_bulk_write), appended to msg.
static void nvd_mmio(std::vector<uint8_t>& msg, uint32_t bar, uint64_t off, const void* data, uint32_t len) {
    uint8_t hdr[33];
    tg_pack_hdr(hdr, TGC_MMIO_WRITE, 0, bar, off, len, 0);
    msg.insert(msg.end(), hdr, hdr + 33);
    msg.insert(msg.end(), (const uint8_t*)data, (const uint8_t*)data + len);
}

// NVCommandQueue._submit_to_gpfifo: the pushbuffer goes into the shared ring;
// the GPFIFO entry, GPPut and doorbell go out as three posted writes in one send.
static void nvd_submit(NVDFifo& f, const std::vector<uint32_t>& pb) {
    if (g_nvd->pending >= f.entries / 2) nvd_idle();  // never let the GPFIFO ring lap the GPU
    uint64_t off = nvd_alloc(g_nvd->cmdq_pos, g_nvd->h.cmdq.size, pb.size() * 4, 16);
    memcpy(g_nvd->cmdq + off, pb.data(), pb.size() * 4);
    uint64_t entry = nvd_gpfifo_entry(g_nvd->h.cmdq.va + off, (uint32_t)pb.size());
    uint32_t gpput = (uint32_t)((f.put + 1) % f.entries);
    std::vector<uint8_t> msg;
    msg.reserve(3 * 33 + 16);
    nvd_mmio(msg, f.ring_bar, f.ring_off + (f.put % f.entries) * 8, &entry, 8);
    nvd_mmio(msg, f.gpput_bar, f.gpput_off, &gpput, 4);
    nvd_mmio(msg, g_nvd->h.db_bar, g_nvd->h.db_off, &f.token, 4);
    tg_send_all(g_nvd->tg_sock, msg.data(), msg.size());
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
static void nvdFlushLaunches() {
    auto t0 = nv_profile_start();
    NVDHandoff& h = g_nvd->h;
    std::vector<uint32_t> pb;
    uint64_t value = g_nvd->timeline;
    nvd_push_wait(pb, h, h.signal.va, value - 1);
    nvd_push_invalidate(pb, h);
    uint8_t* prev = nullptr;
    long long launched = 0;
    for (const NVPendingLaunch& pl : g_nvPendingLaunches) {
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
    g_nvPendingLaunches.clear();
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

static void nvd_unmap(NVDispatchState* d) {
    const NVDBuffer* bufs[4] = { &d->h.cmdq, &d->h.kargs, &d->h.staging, &d->h.signal };
    for (int i = 0; i < 4; ++i)
        if (d->maps[i]) { munmap(d->maps[i], bufs[i]->size); d->maps[i] = nullptr; }
}

// cmd_handoff: the daemon's reply (flat JSON), the kernel blob, then the fds
// of the four shared buffers in NVDHandoff's order. The daemon stops using the
// queues once it replies, so a failure here is fatal.
static NVDispatchState* nvDispatchHandoff(int cmd_sock, int tg_sock) {
    auto t0 = nv_profile_start();
    nv_send_msg(cmd_sock, "{\"cmd\":\"handoff\"}");
    std::string js = nv_recv_msg(cmd_sock);
    uint64_t blob_size = 0, nfds = 0;
    if (js.empty() || !nv_json_ok(js) || !nvd_json_u64(js, "blob_size", blob_size) || !nvd_json_u64(js, "nfds", nfds) || nfds != 4) {
        fprintf(stderr, "TinyGPU/NV: handoff failed: %s\n", js.c_str());
        return nullptr;
    }
    std::vector<uint8_t> blob(blob_size);
    int fds[4] = { -1, -1, -1, -1 };
    if (!nv_recv_all(cmd_sock, blob.data(), blob.size()) || !nv_recv_fds(cmd_sock, fds, 4)) {
        fprintf(stderr, "TinyGPU/NV: handoff: daemon connection lost\n");
        return nullptr;
    }
    NVDispatchState* d = new NVDispatchState;
    std::string err = nvd_parse_handoff(js, blob, d->h);
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
    fprintf(stderr, "TinyGPU/NV: C++ dispatch: %zu kernels handed over (QMD v%u)\n", d->h.kernels.size(), d->h.qmd_ver);
    return d;
}

// ── nvDispatchDaemonSetup: spawn nv_dispatch_daemon.py over a dedicated
// socketpair (NOT the TinyGPU socket -- the daemon connects to TinyGPU.app
// itself via NVDevice("NV:0"), matching §74's hardware-verified reference
// test exactly, no inherited FD needed), then send "boot" and
// "compile_all". For C++ dispatch (tg_fd >= 0) the daemon inherits the
// plugin's TinyGPU.app connection instead, and "handoff" follows. ──────────

static NVHybridState* nvDispatchDaemonSetup(const char* kernel_code, int tg_fd) {
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
    g->daemon_pid = pid;

    fprintf(stderr, "TinyGPU/NV: sending boot command...\n"); fflush(stderr);
    auto t0 = nv_profile_start();
    nv_send_msg(g->cmd_sock, "{\"cmd\":\"boot\"}");
    std::string resp = nv_recv_msg(g->cmd_sock);
    nv_profile_end("boot", t0);
    if (resp.empty() || !nv_json_ok(resp)) {
        fprintf(stderr, "TinyGPU/NV: boot failed: %s\n", resp.c_str());
        delete g;
        return nullptr;
    }
    fprintf(stderr, "TinyGPU/NV: daemon booted — arch=%s\n", nv_json_str(resp, "arch").c_str());

    if (kernel_code && kernel_code[0]) {
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
            delete g;
            return nullptr;
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
                    g_nvKernels[kname] = new NVKernelHandle{kname};
                    ++loaded;
                }
                q = qe + 1;
            }
            fprintf(stderr, "TinyGPU/NV: compile_all — loaded %d kernels\n", loaded);
        }
    }
    if (tg_fd >= 0) {
        g_nvd = nvDispatchHandoff(g->cmd_sock, tg_fd);
        if (!g_nvd) { delete g; return nullptr; }
        for (auto& kv : g_nvKernels) {
            auto it = g_nvd->h.kernels.find(kv.first);
            kv.second->tmpl = (it != g_nvd->h.kernels.end()) ? &it->second : nullptr;
        }
    }
    fflush(stderr);
    return g;
}

// ── GPUInterface entry points ─────────────────────────────────────────────────

void NvSetDevice(GPUInterface* self, int paddedStateCount, int categoryCount,
                  int patternCount, int unpaddedPatternCount, int tipCount, long flags) {
    // Close Initialize()'s TinyGPU.app connection before spawning the
    // dispatch daemon: NVDevice("NV:0") opens its own, fully independent
    // connection (STATUS.md §74, hardware-verified), same as
    // the AMD daemon does. Leaving this one open too risks the exact bug
    // AMD's own tgpuSock fix (STATUS.md AMD §21) found: TinyGPU.app doesn't
    // tolerate two simultaneous clients, and the daemon's own connection
    // attempt hangs forever instead of failing cleanly. C++ dispatch keeps
    // it instead and the daemon inherits it: both sides then share the one
    // connection (the GPUInterface destructor closes it after NvFini).
    int tg_fd = -1;
    if (nv_cpp_dispatch() && self->tgpuSock >= 0) {
        tg_fd = self->tgpuSock;
        fcntl(tg_fd, F_SETFD, fcntl(tg_fd, F_GETFD) & ~FD_CLOEXEC);
    } else if (self->tgpuSock >= 0) {
        close(self->tgpuSock); self->tgpuSock = -1;
    }

    self->InitializeKernelResource(paddedStateCount, (flags & BEAGLE_FLAG_PRECISION_DOUBLE) != 0);
    self->supportDoublePrecision = ((flags & BEAGLE_FLAG_PRECISION_DOUBLE) != 0);
    if (self->kernelResource) {
        self->kernelResource->categoryCount        = categoryCount;
        self->kernelResource->patternCount         = patternCount;
        self->kernelResource->unpaddedPatternCount = unpaddedPatternCount;
        self->kernelResource->flags                = flags;
    }

    g_nv = nvDispatchDaemonSetup(self->kernelResource ? self->kernelResource->kernelCode : nullptr, tg_fd);
    if (!g_nv) { fprintf(stderr, "TinyGPU/NV: nvDispatchDaemonSetup failed\n"); nv_safe_exit(1); }
}

GPUFunction NvGetFunction(const char* name) {
    if (!g_nv) return nullptr;
    auto it = g_nvKernels.find(name);
    if (it != g_nvKernels.end()) return it->second;
    fprintf(stderr, "TinyGPU/NV: GetFunction(%s): kernel not found in precompiled cache — exiting\n", name);
    nv_safe_exit(1);
}

void NvSynchronizeHost() {
    if (!g_nv) return;
    nvFlushLaunchQueue();  // otherwise queued-but-unsent launches wouldn't be submitted yet to wait for
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
    if (!g_nv) return 0;
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

void NvMemcpyHostToDevice(GPUPtr dst, const void* src, size_t sz) {
    if (!g_nv || !src || !sz) return;
    nvFlushLaunchQueue();  // preserve ordering: queued launches must be submitted before this write
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

void NvMemcpyDeviceToHost(void* dst, const GPUPtr src, size_t sz) {
    if (!g_nv || !dst || !sz) return;
    nvFlushLaunchQueue();  // preserve ordering: queued launches must complete before this read
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
    // Python (via NVAllocator/HCQCompiled) owns allocation entirely now; this
    // backend has no independent view of remaining VRAM. Report a generous
    // constant rather than 0 (which some callers may treat as "out of
    // memory") -- purely informational, not load-bearing. Same convention as
    // AmdGetAvailableMemory().
    return g_nv ? (size_t)(1ull << 30) : 0;
}

void NvFini() {
    if (!g_nv) return;
    nvFlushLaunchQueue();  // don't silently drop queued-but-unsent launches
    if (g_nvd) {  // let the GPU finish before the daemon tears it down
        nvd_idle();
        nvd_unmap(g_nvd);
        delete g_nvd;
        g_nvd = nullptr;
    }
    nv_profile_report();
    for (auto& kv : g_nvKernels) delete kv.second;
    g_nvKernels.clear();
    if (g_nv->cmd_sock >= 0) {
        nv_send_msg(g_nv->cmd_sock, "{\"cmd\":\"fini\"}");
        nv_recv_msg(g_nv->cmd_sock);
    }
    if (g_nv->daemon_pid > 0) {
        for (int i = 0; i < 100; ++i) {
            int st = 0;
            if (waitpid(g_nv->daemon_pid, &st, WNOHANG) > 0) { g_nv->daemon_pid = 0; break; }
            usleep(100000);
        }
    }
    if (g_nv->cmd_sock >= 0) close(g_nv->cmd_sock);
    delete g_nv;
    g_nv = nullptr;
}

void NvLaunchKernelImpl(GPUFunction fn, Dim3Int block, Dim3Int grid,
                         int nPtr, int nTotal, GPUPtr* ptrs, unsigned int* ints) {
    if (!g_nv || !fn) return;
    NVKernelHandle* ke = (NVKernelHandle*)fn;
    int nInt = nTotal - nPtr;

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
    g_nvPendingLaunches.push_back(std::move(pl));
}

} // namespace tinygpu_device

#endif // FW_TINYGPU
