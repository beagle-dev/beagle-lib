/*
 * GPUInterfaceTinyGPUHybridAMD.cpp
 *
 * BEAGLE AMD hybrid backend, take 2.
 *
 * Four hand-built PM4 dispatch attempts (two independent implementations,
 * across two rounds of real, mechanically-verified bug fixes) all crashed
 * the host identically (DART "read of DVA 0" panic -- see STATUS.md AMD
 * §3-§11). The only thing that has ever worked on this hardware is stock,
 * unmodified tinygrad using the *full* AMDDevice/PCIIface/HCQCompiled
 * stack (STATUS.md §8) -- never bare AMDev+setup_ring() in isolation,
 * which is all the prior attempts ever drove.
 *
 * This file stops hand-deriving the PM4 stream entirely. It is now a thin
 * RPC client: a live Python daemon (amd_dispatch_daemon.py) stays resident
 * and does EVERY GPU operation -- boot, compile, alloc, memcpy, launch,
 * sync -- via tinygrad's real AMDDevice/AMDProgram/HCQProgram.__call__
 * code. This file just sends newline-terminated JSON commands over a
 * dedicated socketpair (not the TinyGPU socket -- AMDDevice("AMD:0") makes
 * its own connection internally, the same way the working reference test
 * did) and reads back replies.
 *
 * Compile backend unchanged: comgr compiling BEAGLE's existing FW_OPENCL
 * kernel source (amd_compile_helper.py's compile_opencl(), reused by the
 * daemon directly, not re-invoked as a subprocess).
 */

#ifdef FW_TINYGPU

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
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "libhmsbeagle/beagle.h"
#include "libhmsbeagle/GPU/GPUImplDefs.h"
#include "libhmsbeagle/GPU/GPUImplHelper.h"
#include "libhmsbeagle/GPU/GPUInterface.h"
#include "libhmsbeagle/GPU/KernelResource.h"
#include "libhmsbeagle/GPU/GPUInterfaceTinyGPUHybridAMD.h"

// GPUInterface.h's FW_TINYGPU branch (included above) already pulled in
// kernels/BeagleTinyGPU_kernels.h, whose KERNELS_STRING_<PREC>_<N> macros are
// PTX (the NV path's compile input). The AMD path needs the real OpenCL-C
// source instead -- see the plan's compiler-backend decision (comgr
// compiles BEAGLE's existing FW_OPENCL kernels unmodified, not a HIP port).
// Same macro names, different header; #undef the PTX versions first.
#undef KERNELS_STRING_SP_4
#undef KERNELS_STRING_SP_16
#undef KERNELS_STRING_SP_32
#undef KERNELS_STRING_SP_48
#undef KERNELS_STRING_SP_64
#undef KERNELS_STRING_SP_80
#undef KERNELS_STRING_SP_128
#undef KERNELS_STRING_SP_192
#undef KERNELS_STRING_SP_256
#undef KERNELS_STRING_DP_4
#undef KERNELS_STRING_DP_16
#undef KERNELS_STRING_DP_32
#undef KERNELS_STRING_DP_48
#undef KERNELS_STRING_DP_64
#undef KERNELS_STRING_DP_80
#undef KERNELS_STRING_DP_128
#undef KERNELS_STRING_DP_192
#undef KERNELS_STRING_DP_256
#include "libhmsbeagle/GPU/kernels/BeagleOpenCL_kernels.h"

namespace tinygpu_device {

static const char* amd_opencl_kernel_source(int paddedStateCount, bool doublePrecision) {
    int n = doublePrecision ? -paddedStateCount : paddedStateCount;
    switch (n) {
        case   -4: return KERNELS_STRING_DP_4;
        case  -16: return KERNELS_STRING_DP_16;
        case  -32: return KERNELS_STRING_DP_32;
        case  -48: return KERNELS_STRING_DP_48;
        case  -64: return KERNELS_STRING_DP_64;
        case  -80: return KERNELS_STRING_DP_80;
        case -128: return KERNELS_STRING_DP_128;
        case -192: return KERNELS_STRING_DP_192;
        case -256: return KERNELS_STRING_DP_256;
        case    4: return KERNELS_STRING_SP_4;
        case   16: return KERNELS_STRING_SP_16;
        case   32: return KERNELS_STRING_SP_32;
        case   48: return KERNELS_STRING_SP_48;
        case   64: return KERNELS_STRING_SP_64;
        case   80: return KERNELS_STRING_SP_80;
        case  128: return KERNELS_STRING_SP_128;
        case  192: return KERNELS_STRING_SP_192;
        case  256: return KERNELS_STRING_SP_256;
        default:
            fprintf(stderr, "TinyGPU/AMD: no OpenCL kernel source for paddedStateCount=%d doublePrecision=%d\n",
                    paddedStateCount, (int)doublePrecision);
            return nullptr;
    }
}

// ── small utilities (file I/O + minimal JSON; same style as the NV file) ────

static std::string amd_read_file(const char* path) {
    FILE* f = fopen(path, "r"); if (!f) return "";
    std::string s; char buf[4096]; size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) s.append(buf, n);
    fclose(f); return s;
}
static bool amd_write_file(const char* path, const void* buf, size_t sz) {
    FILE* f = fopen(path, "wb"); if (!f) return false;
    fwrite(buf, 1, sz, f); fclose(f); return true;
}
static uint64_t amd_json_u64(const std::string& js, const char* key) {
    char needle[128]; snprintf(needle, sizeof(needle), "\"%s\":", key);
    auto p = js.find(needle);
    if (p == std::string::npos) return 0;
    p += strlen(needle);
    while (p < js.size() && (js[p]==' '||js[p]=='\n')) ++p;
    return (uint64_t)strtoull(js.c_str() + p, nullptr, 10);
}
static bool amd_json_ok(const std::string& js) {
    auto p = js.find("\"ok\":");
    if (p == std::string::npos) return false;
    p += 5;
    while (p < js.size() && js[p]==' ') ++p;
    return js.compare(p, 4, "true") == 0;
}
static std::string amd_json_str(const std::string& js, const char* key) {
    char needle[128]; snprintf(needle, sizeof(needle), "\"%s\":", key);
    auto p = js.find(needle);
    if (p == std::string::npos) return "";
    p = js.find('"', p + strlen(needle));
    if (p == std::string::npos) return "";
    auto e = js.find('"', p + 1);
    return js.substr(p + 1, e - p - 1);
}

static std::string amd_resolve_python() {
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

// ── Command-socket I/O: newline-terminated JSON lines, with raw bytes
// immediately following for h2d (request) / d2h (reply) ─────────────────────

static void amd_send_all(int fd, const void* buf, size_t n) {
    const uint8_t* p = (const uint8_t*)buf;
    while (n) { ssize_t r = ::send(fd, p, n, 0); if (r <= 0) return; p += r; n -= (size_t)r; }
}
static void amd_recv_all(int fd, void* buf, size_t n) {
    uint8_t* p = (uint8_t*)buf;
    while (n) { ssize_t r = ::recv(fd, p, n, MSG_WAITALL); if (r <= 0) return; p += r; n -= (size_t)r; }
}
static void amd_send_line(int fd, const std::string& line) {
    std::string s = line + "\n";
    amd_send_all(fd, s.data(), s.size());
}
static std::string amd_recv_line(int fd) {
    std::string s;
    char c;
    while (true) {
        ssize_t r = ::recv(fd, &c, 1, 0);
        if (r <= 0) break;
        if (c == '\n') break;
        s.push_back(c);
    }
    return s;
}

// ── State ────────────────────────────────────────────────────────────────────

// ── Opt-in RPC round-trip profiling (BEAGLE_AMD_PROFILE=1) ─────────────────
// Measures host-side overhead per RPC call (send + Python-side work incl.
// real GPU dispatch + recv) -- to find out whether that overhead is
// actually worth optimizing before considering anything riskier, like
// hand-rolled C++ PM4 dispatch (which caused five hardware crashes, §1-11).
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

struct AMDHybridState {
    int cmd_sock;
    pid_t daemon_pid;
};

struct AMDKernelHandle {
    std::string name;
};

static AMDHybridState* g_amd = nullptr;
static std::map<std::string, AMDKernelHandle*> g_amdKernels;

// ── Launch batching (STATUS.md AMD §26) ─────────────────────────────────────
// Profiling (BEAGLE_AMD_PROFILE=1) found steady-state per-launch RPC
// round-trip overhead (~150-190us) comparable to or larger than the actual
// GPU dispatch work (~100us). AmdLaunchKernelImpl queues launches here
// instead of sending each as its own round-trip; amdFlushLaunchQueue()
// sends the whole queue as one "launch_batch" RPC call. Flushed before
// every h2d/d2h/sync/fini (amdFlushLaunchQueue() calls below) so ordering
// relative to memory operations is preserved -- see amd_dispatch_daemon.py's
// module docstring for why that's sufficient without any extra
// synchronization on either side (short version: launches only ever enqueue
// PM4 packets, wait=False, so flush-before preserves submission order; and
// tinygrad's own _copyin/_copyout/synchronize already wait for prior
// submitted work internally before touching memory).
struct AMDPendingLaunch {
    std::string kernel;
    int grid[3];
    int block[3];
    std::vector<unsigned long long> ptrs;
    std::vector<unsigned int> ints;
};
static std::vector<AMDPendingLaunch> g_amdPendingLaunches;

static void amdFlushLaunchQueue() {
    if (!g_amd || g_amdPendingLaunches.empty()) return;
    auto t0 = amd_profile_start();
    std::string cmd = "{\"cmd\":\"launch_batch\",\"launches\":[";
    for (size_t li = 0; li < g_amdPendingLaunches.size(); ++li) {
        const AMDPendingLaunch& pl = g_amdPendingLaunches[li];
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
    size_t n = g_amdPendingLaunches.size();
    g_amdPendingLaunches.clear();

    amd_send_line(g_amd->cmd_sock, cmd);
    std::string resp = amd_recv_line(g_amd->cmd_sock);
    amd_profile_end("launch_batch", t0);
    if (resp.empty() || !amd_json_ok(resp))
        fprintf(stderr, "TinyGPU/AMD: launch_batch(%zu kernels) failed: %s\n", n, resp.c_str());
}

[[noreturn]] static void amd_safe_exit(int code) {
    fflush(stderr);
    if (g_amd) {
        if (g_amd->cmd_sock >= 0) {
            amd_send_line(g_amd->cmd_sock, "{\"cmd\":\"fini\"}");
            amd_recv_line(g_amd->cmd_sock);  // best-effort ack, ignore content
        }
        if (g_amd->daemon_pid > 0) {
            for (int i = 0; i < 100; ++i) {
                int st = 0;
                if (waitpid(g_amd->daemon_pid, &st, WNOHANG) > 0) break;
                usleep(100000);
            }
        }
        if (g_amd->cmd_sock >= 0) close(g_amd->cmd_sock);
    }
    _exit(code);
}

// ── amdDispatchDaemonSetup: spawn amd_dispatch_daemon.py over a dedicated
// socketpair (NOT the TinyGPU socket -- the daemon connects to TinyGPU.app
// itself via AMDDevice("AMD:0"), matching the STATUS.md §8 reference test
// exactly, no inherited FD needed), then send "boot" and "compile_all". ────

static AMDHybridState* amdDispatchDaemonSetup(const char* kernel_code) {
    int sv[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) {
        fprintf(stderr, "TinyGPU/AMD: socketpair failed: %s\n", strerror(errno));
        return nullptr;
    }

    char script[256];
    const char* helper = getenv("BEAGLE_AMD_DISPATCH_DAEMON");
    if (!helper) {
        snprintf(script, sizeof(script), "%s/amd_dispatch_daemon.py", getenv("BEAGLE_NV_SCRIPTS") ?: ".");
        helper = script;
    }
    std::string pypath = amd_resolve_python();

    // Clear O_CLOEXEC on the child's end so it survives execvp.
    int flags = fcntl(sv[1], F_GETFD);
    fcntl(sv[1], F_SETFD, flags & ~FD_CLOEXEC);

    fprintf(stderr, "TinyGPU/AMD: spawning amd_dispatch_daemon.py (python=%s)\n", pypath.c_str());
    pid_t pid = fork();
    if (pid < 0) {
        fprintf(stderr, "TinyGPU/AMD: fork failed: %s\n", strerror(errno));
        close(sv[0]); close(sv[1]);
        return nullptr;
    }
    if (pid == 0) {
        close(sv[0]);
        dup2(STDERR_FILENO, STDOUT_FILENO);
        char fd_str[16]; snprintf(fd_str, sizeof(fd_str), "%d", sv[1]);
        char* argv[] = { (char*)pypath.c_str(), (char*)helper, fd_str, nullptr };
        execvp(pypath.c_str(), argv);
        fprintf(stderr, "TinyGPU/AMD: execvp %s failed: %s\n", pypath.c_str(), strerror(errno));
        _exit(1);
    }
    close(sv[1]);

    AMDHybridState* g = new AMDHybridState{};
    g->cmd_sock = sv[0];
    g->daemon_pid = pid;

    fprintf(stderr, "TinyGPU/AMD: sending boot command...\n"); fflush(stderr);
    amd_send_line(g->cmd_sock, "{\"cmd\":\"boot\"}");
    std::string resp = amd_recv_line(g->cmd_sock);
    if (resp.empty() || !amd_json_ok(resp)) {
        fprintf(stderr, "TinyGPU/AMD: boot failed: %s\n", resp.c_str());
        delete g;
        return nullptr;
    }
    fprintf(stderr, "TinyGPU/AMD: daemon booted — arch=%s\n", amd_json_str(resp, "arch").c_str());

    if (kernel_code && kernel_code[0]) {
        char cl_path[256];
        snprintf(cl_path, sizeof(cl_path), "/tmp/beagle_amd_all_%d.cl", getpid());
        amd_write_file(cl_path, kernel_code, strlen(kernel_code));

        char cmd[512];
        snprintf(cmd, sizeof(cmd), "{\"cmd\":\"compile_all\",\"cl_path\":\"%s\"}", cl_path);
        fprintf(stderr, "TinyGPU/AMD: precompile_all_kernels — compiling all kernels (comgr × 1, via daemon)…\n");
        fflush(stderr);
        amd_send_line(g->cmd_sock, cmd);
        resp = amd_recv_line(g->cmd_sock);
        unlink(cl_path);
        if (resp.empty() || !amd_json_ok(resp)) {
            fprintf(stderr, "TinyGPU/AMD: compile_all failed: %s\n", resp.c_str());
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
                    g_amdKernels[kname] = new AMDKernelHandle{kname};
                    ++loaded;
                }
                q = qe + 1;
            }
            fprintf(stderr, "TinyGPU/AMD: precompile_all_kernels — loaded %d kernels\n", loaded);
        }
    }
    fflush(stderr);
    return g;
}

// ── GPUInterface entry points ─────────────────────────────────────────────────

void AmdSetDevice(GPUInterface* self, int paddedStateCount, int categoryCount,
                   int patternCount, int unpaddedPatternCount, int tipCount, long flags) {
    // Close Initialize()'s TinyGPU.app connection (self->tgpuSock) before
    // spawning the dispatch daemon: unlike the NV path, which keeps reusing
    // this same socket for all hardware access, the AMD daemon opens its
    // own, fully independent connection via AMDDevice("AMD:0"). Leaving
    // this one open too was a real bug: TinyGPU.app apparently doesn't
    // tolerate two simultaneous clients cleanly -- the daemon's own
    // connection would hang forever inside AMDDevice's resize_bar() RPC
    // (Device["AMD:0"] never completing, cmd_boot never replying) while
    // this stale connection sat idle. Confirmed via a real hardware hang,
    // traceback pinned exactly to that RPC's blocking socket read.
    if (self->tgpuSock >= 0) { close(self->tgpuSock); self->tgpuSock = -1; }
    g_amd = amdDispatchDaemonSetup(amd_opencl_kernel_source(paddedStateCount, (flags & BEAGLE_FLAG_PRECISION_DOUBLE) != 0));
    if (!g_amd) { fprintf(stderr, "TinyGPU/AMD: amdDispatchDaemonSetup failed\n"); amd_safe_exit(1); }

    self->InitializeKernelResource(paddedStateCount, (flags & BEAGLE_FLAG_PRECISION_DOUBLE) != 0);
    self->supportDoublePrecision = ((flags & BEAGLE_FLAG_PRECISION_DOUBLE) != 0);
    if (self->kernelResource) {
        self->kernelResource->categoryCount        = categoryCount;
        self->kernelResource->patternCount         = patternCount;
        self->kernelResource->unpaddedPatternCount = unpaddedPatternCount;
        self->kernelResource->flags                = flags;
    }
}

GPUFunction AmdGetFunction(const char* name) {
    if (!g_amd) return nullptr;
    auto it = g_amdKernels.find(name);
    if (it != g_amdKernels.end()) return it->second;
    fprintf(stderr, "TinyGPU/AMD: GetFunction(%s): kernel not found in precompiled cache — exiting\n", name);
    amd_safe_exit(1);
}

void AmdSynchronizeHost() {
    if (!g_amd) return;
    amdFlushLaunchQueue();  // otherwise queued-but-unsent launches wouldn't be submitted yet to wait for
    auto t0 = amd_profile_start();
    amd_send_line(g_amd->cmd_sock, "{\"cmd\":\"sync\"}");
    std::string resp = amd_recv_line(g_amd->cmd_sock);
    amd_profile_end("sync", t0);
    if (resp.empty() || !amd_json_ok(resp))
        fprintf(stderr, "TinyGPU/AMD: sync failed: %s\n", resp.c_str());
}

GPUPtr AmdAllocateMemory(size_t sz) {
    if (!g_amd) return 0;
    auto t0 = amd_profile_start();
    char cmd[128];
    snprintf(cmd, sizeof(cmd), "{\"cmd\":\"alloc\",\"size\":%zu}", sz);
    amd_send_line(g_amd->cmd_sock, cmd);
    std::string resp = amd_recv_line(g_amd->cmd_sock);
    amd_profile_end("alloc", t0);
    if (resp.empty() || !amd_json_ok(resp)) {
        fprintf(stderr, "TinyGPU/AMD: alloc(%zu) failed: %s\n", sz, resp.c_str());
        return 0;
    }
    return (GPUPtr)amd_json_u64(resp, "addr");
}

void AmdMemcpyHostToDevice(GPUPtr dst, const void* src, size_t sz) {
    if (!g_amd || !src || !sz) return;
    amdFlushLaunchQueue();  // preserve ordering: queued launches must be submitted before this write
    auto t0 = amd_profile_start();
    char cmd[128];
    snprintf(cmd, sizeof(cmd), "{\"cmd\":\"h2d\",\"addr\":%llu,\"size\":%zu}", (unsigned long long)dst, sz);
    amd_send_line(g_amd->cmd_sock, cmd);
    amd_send_all(g_amd->cmd_sock, src, sz);
    std::string resp = amd_recv_line(g_amd->cmd_sock);
    amd_profile_end("h2d", t0);
    if (resp.empty() || !amd_json_ok(resp))
        fprintf(stderr, "TinyGPU/AMD: h2d(addr=0x%llx, sz=%zu) failed: %s\n", (unsigned long long)dst, sz, resp.c_str());
}

void AmdMemcpyDeviceToHost(void* dst, const GPUPtr src, size_t sz) {
    if (!g_amd || !dst || !sz) return;
    amdFlushLaunchQueue();  // preserve ordering: queued launches must complete before this read
    auto t0 = amd_profile_start();
    char cmd[128];
    snprintf(cmd, sizeof(cmd), "{\"cmd\":\"d2h\",\"addr\":%llu,\"size\":%zu}", (unsigned long long)src, sz);
    amd_send_line(g_amd->cmd_sock, cmd);
    std::string resp = amd_recv_line(g_amd->cmd_sock);
    if (resp.empty() || !amd_json_ok(resp)) {
        amd_profile_end("d2h", t0);
        fprintf(stderr, "TinyGPU/AMD: d2h(addr=0x%llx, sz=%zu) failed: %s\n", (unsigned long long)src, sz, resp.c_str());
        return;
    }
    amd_recv_all(g_amd->cmd_sock, dst, sz);
    amd_profile_end("d2h", t0);
}

size_t AmdGetAvailableMemory() {
    // Python (via AMDAllocator/HCQCompiled) owns allocation entirely now;
    // this backend has no independent view of remaining VRAM. Report a
    // generous constant rather than 0 (which some callers may treat as
    // "out of memory") -- purely informational, not load-bearing.
    return g_amd ? (size_t)(1ull << 30) : 0;
}

void AmdFini() {
    if (!g_amd) return;
    amdFlushLaunchQueue();  // don't silently drop queued-but-unsent launches
    for (auto& kv : g_amdKernels) delete kv.second;
    g_amdKernels.clear();
    if (g_amd->cmd_sock >= 0) {
        amd_send_line(g_amd->cmd_sock, "{\"cmd\":\"fini\"}");
        amd_recv_line(g_amd->cmd_sock);
    }
    if (g_amd->daemon_pid > 0) {
        for (int i = 0; i < 100; ++i) {
            int st = 0;
            if (waitpid(g_amd->daemon_pid, &st, WNOHANG) > 0) { g_amd->daemon_pid = 0; break; }
            usleep(100000);
        }
    }
    if (g_amd->cmd_sock >= 0) close(g_amd->cmd_sock);
    delete g_amd;
    g_amd = nullptr;
}

void AmdLaunchKernelImpl(GPUFunction fn, Dim3Int block, Dim3Int grid,
                          int nPtr, int nTotal, GPUPtr* ptrs, unsigned int* ints) {
    if (!g_amd || !fn) return;
    AMDKernelHandle* ke = (AMDKernelHandle*)fn;
    int nInt = nTotal - nPtr;

    fprintf(stderr, "TinyGPU/AMD: launch %s grid=(%d,%d,%d) block=(%d,%d,%d) nPtr=%d nInt=%d\n",
            ke->name.c_str(), grid.x, grid.y, grid.z, block.x, block.y, block.z, nPtr, nInt);
    fflush(stderr);

    // Queued, not sent (STATUS.md AMD §26) -- amdFlushLaunchQueue() sends the
    // whole backlog as one RPC round-trip, called before any h2d/d2h/sync/
    // fini so ordering relative to memory operations is preserved.
    AMDPendingLaunch pl;
    pl.kernel = ke->name;
    pl.grid[0] = grid.x; pl.grid[1] = grid.y; pl.grid[2] = grid.z;
    pl.block[0] = block.x; pl.block[1] = block.y; pl.block[2] = block.z;
    pl.ptrs.assign(ptrs, ptrs + nPtr);
    pl.ints.assign(ints, ints + nInt);
    g_amdPendingLaunches.push_back(std::move(pl));
}

} // namespace tinygpu_device

#endif // FW_TINYGPU
