/*
 * GPUInterfaceTinyGPUHybridNV.cpp
 *
 * BEAGLE NV hybrid backend, daemon architecture (STATUS.md §73/§75).
 *
 * GPUInterfaceTinyGPUHybrid.cpp's original NV path hand-rolls GPFIFO/QMD
 * command-queue construction and dispatch directly in C++. Three
 * individually-clean SASS/PTX-level probes (STATUS.md §65-72) failed to
 * reproduce this path's wrong-answer bug -- "the defect must depend on
 * something these probes structurally cannot reproduce", the same shape the
 * AMD hybrid backend hit before its own pivot away from hand-rolled
 * dispatch (STATUS.md AMD §1-11). STATUS.md §74's nv_reference_test.py
 * confirmed (hardware-verified PASS) that tinygrad's own real NVDevice/
 * NVProgram/HCQProgram.__call__ stack works correctly end to end on this
 * exact hardware/transport.
 *
 * This file is a thin RPC client, structurally identical to
 * GPUInterfaceTinyGPUHybridAMD.cpp: a live Python daemon
 * (nv_dispatch_daemon.py) stays resident and does EVERY GPU operation --
 * boot, compile, alloc, memcpy, launch, sync -- via tinygrad's real
 * NVDevice/NVProgram/HCQProgram.__call__ code. This file sends
 * newline-terminated JSON commands over a dedicated socketpair (not the
 * TinyGPU socket -- NVDevice("NV:0") makes its own connection internally,
 * the same way nv_reference_test.py's hardware-verified boot did) and reads
 * back replies.
 *
 * Compile backend unchanged: BEAGLE's existing PTX kernel source
 * (kernelResource->kernelCode, from kernels/BeagleTinyGPU_kernels.h) via
 * nv_compile_helper.py's compile_ptx(), reused by the daemon directly, not
 * re-invoked as a per-kernel subprocess the way the hand-rolled path did.
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
#include "libhmsbeagle/GPU/GPUInterfaceTinyGPUHybridNV.h"

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

// ── Command-socket I/O: newline-terminated JSON lines, with raw bytes
// immediately following for h2d (request) / d2h (reply) ─────────────────────

static void nv_send_all(int fd, const void* buf, size_t n) {
    const uint8_t* p = (const uint8_t*)buf;
    while (n) { ssize_t r = ::send(fd, p, n, 0); if (r <= 0) return; p += r; n -= (size_t)r; }
}
static void nv_recv_all(int fd, void* buf, size_t n) {
    uint8_t* p = (uint8_t*)buf;
    while (n) { ssize_t r = ::recv(fd, p, n, MSG_WAITALL); if (r <= 0) return; p += r; n -= (size_t)r; }
}
static void nv_send_line(int fd, const std::string& line) {
    std::string s = line + "\n";
    nv_send_all(fd, s.data(), s.size());
}
static std::string nv_recv_line(int fd) {
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

struct NVHybridState {
    int cmd_sock;
    pid_t daemon_pid;
};

struct NVKernelHandle {
    std::string name;
};

static NVHybridState* g_nv = nullptr;
static std::map<std::string, NVKernelHandle*> g_nvKernels;

// ── Launch batching (mirrors AMD's, STATUS.md AMD §26 -- built in from the
// start here rather than added later, since that overhead finding already
// generalizes: any RPC-per-launch design pays the same per-call socket+JSON
// cost regardless of vendor). Flushed before every h2d/d2h/sync/fini so
// ordering relative to memory operations is preserved -- see
// nv_dispatch_daemon.py's cmd_launch_batch comment for why that's sufficient
// without extra synchronization on either side. ─────────────────────────────
struct NVPendingLaunch {
    std::string kernel;
    int grid[3];
    int block[3];
    std::vector<unsigned long long> ptrs;
    std::vector<unsigned int> ints;
};
static std::vector<NVPendingLaunch> g_nvPendingLaunches;

static void nvFlushLaunchQueue() {
    if (!g_nv || g_nvPendingLaunches.empty()) return;
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

    nv_send_line(g_nv->cmd_sock, cmd);
    std::string resp = nv_recv_line(g_nv->cmd_sock);
    if (resp.empty() || !nv_json_ok(resp))
        fprintf(stderr, "TinyGPU/NV: launch_batch(%zu kernels) failed: %s\n", n, resp.c_str());
}

[[noreturn]] static void nv_safe_exit(int code) {
    fflush(stderr);
    if (g_nv) {
        if (g_nv->cmd_sock >= 0) {
            nv_send_line(g_nv->cmd_sock, "{\"cmd\":\"fini\"}");
            nv_recv_line(g_nv->cmd_sock);  // best-effort ack, ignore content
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

// ── nvDispatchDaemonSetup: spawn nv_dispatch_daemon.py over a dedicated
// socketpair (NOT the TinyGPU socket -- the daemon connects to TinyGPU.app
// itself via NVDevice("NV:0"), matching §74's hardware-verified reference
// test exactly, no inherited FD needed), then send "boot" and
// "compile_all". ─────────────────────────────────────────────────────────

static NVHybridState* nvDispatchDaemonSetup(const char* kernel_code) {
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
        char* argv[] = { (char*)pypath.c_str(), (char*)helper, fd_str, nullptr };
        execvp(pypath.c_str(), argv);
        fprintf(stderr, "TinyGPU/NV: execvp %s failed: %s\n", pypath.c_str(), strerror(errno));
        _exit(1);
    }
    close(sv[1]);

    NVHybridState* g = new NVHybridState{};
    g->cmd_sock = sv[0];
    g->daemon_pid = pid;

    fprintf(stderr, "TinyGPU/NV: sending boot command...\n"); fflush(stderr);
    nv_send_line(g->cmd_sock, "{\"cmd\":\"boot\"}");
    std::string resp = nv_recv_line(g->cmd_sock);
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
        nv_send_line(g->cmd_sock, cmd);
        resp = nv_recv_line(g->cmd_sock);
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
    fflush(stderr);
    return g;
}

// ── GPUInterface entry points ─────────────────────────────────────────────────

void NvSetDevice(GPUInterface* self, int paddedStateCount, int categoryCount,
                  int patternCount, int unpaddedPatternCount, int tipCount, long flags) {
    // Close Initialize()'s TinyGPU.app connection before spawning the
    // dispatch daemon: NVDevice("NV:0") opens its own, fully independent
    // connection (§74's nv_reference_test.py, hardware-verified), same as
    // the AMD daemon does. Leaving this one open too risks the exact bug
    // AMD's own tgpuSock fix (STATUS.md AMD §21) found: TinyGPU.app doesn't
    // tolerate two simultaneous clients, and the daemon's own connection
    // attempt hangs forever instead of failing cleanly.
    if (self->tgpuSock >= 0) { close(self->tgpuSock); self->tgpuSock = -1; }

    self->InitializeKernelResource(paddedStateCount, (flags & BEAGLE_FLAG_PRECISION_DOUBLE) != 0);
    self->supportDoublePrecision = ((flags & BEAGLE_FLAG_PRECISION_DOUBLE) != 0);
    if (self->kernelResource) {
        self->kernelResource->categoryCount        = categoryCount;
        self->kernelResource->patternCount         = patternCount;
        self->kernelResource->unpaddedPatternCount = unpaddedPatternCount;
        self->kernelResource->flags                = flags;
    }

    g_nv = nvDispatchDaemonSetup(self->kernelResource ? self->kernelResource->kernelCode : nullptr);
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
    nv_send_line(g_nv->cmd_sock, "{\"cmd\":\"sync\"}");
    std::string resp = nv_recv_line(g_nv->cmd_sock);
    if (resp.empty() || !nv_json_ok(resp))
        fprintf(stderr, "TinyGPU/NV: sync failed: %s\n", resp.c_str());
}

GPUPtr NvAllocateMemory(size_t sz) {
    if (!g_nv) return 0;
    char cmd[128];
    snprintf(cmd, sizeof(cmd), "{\"cmd\":\"alloc\",\"size\":%zu}", sz);
    nv_send_line(g_nv->cmd_sock, cmd);
    std::string resp = nv_recv_line(g_nv->cmd_sock);
    if (resp.empty() || !nv_json_ok(resp)) {
        fprintf(stderr, "TinyGPU/NV: alloc(%zu) failed: %s\n", sz, resp.c_str());
        return 0;
    }
    return (GPUPtr)nv_json_u64(resp, "addr");
}

void NvMemcpyHostToDevice(GPUPtr dst, const void* src, size_t sz) {
    if (!g_nv || !src || !sz) return;
    nvFlushLaunchQueue();  // preserve ordering: queued launches must be submitted before this write
    char cmd[128];
    snprintf(cmd, sizeof(cmd), "{\"cmd\":\"h2d\",\"addr\":%llu,\"size\":%zu}", (unsigned long long)dst, sz);
    nv_send_line(g_nv->cmd_sock, cmd);
    nv_send_all(g_nv->cmd_sock, src, sz);
    std::string resp = nv_recv_line(g_nv->cmd_sock);
    if (resp.empty() || !nv_json_ok(resp))
        fprintf(stderr, "TinyGPU/NV: h2d(addr=0x%llx, sz=%zu) failed: %s\n", (unsigned long long)dst, sz, resp.c_str());
}

void NvMemcpyDeviceToHost(void* dst, const GPUPtr src, size_t sz) {
    if (!g_nv || !dst || !sz) return;
    nvFlushLaunchQueue();  // preserve ordering: queued launches must complete before this read
    char cmd[128];
    snprintf(cmd, sizeof(cmd), "{\"cmd\":\"d2h\",\"addr\":%llu,\"size\":%zu}", (unsigned long long)src, sz);
    nv_send_line(g_nv->cmd_sock, cmd);
    std::string resp = nv_recv_line(g_nv->cmd_sock);
    if (resp.empty() || !nv_json_ok(resp)) {
        fprintf(stderr, "TinyGPU/NV: d2h(addr=0x%llx, sz=%zu) failed: %s\n", (unsigned long long)src, sz, resp.c_str());
        return;
    }
    nv_recv_all(g_nv->cmd_sock, dst, sz);
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
    for (auto& kv : g_nvKernels) delete kv.second;
    g_nvKernels.clear();
    if (g_nv->cmd_sock >= 0) {
        nv_send_line(g_nv->cmd_sock, "{\"cmd\":\"fini\"}");
        nv_recv_line(g_nv->cmd_sock);
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

    // Diagnostic (STATUS.md §79/TODO.md Phase 43): kernelMatrixMulADB packs
    // multiple logical matrix computations into one launch by scaling
    // grid.x by totalMatrix (KernelLauncher.cpp) -- the observed symptom
    // (3 of 4 matrices left completely unwritten) is consistent with either
    // grid.x not actually reaching the daemon as scaled, or the kernel's
    // own totalMatrix scalar arg not arriving correctly. This print shows
    // exactly what this file sends for every launch, settling that question
    // directly from the next hardware run's own terminal output.
    fprintf(stderr, "TinyGPU/NV: queue %s grid=(%d,%d,%d) block=(%d,%d,%d) nPtr=%d nInt=%d ptrs=[",
            ke->name.c_str(), grid.x, grid.y, grid.z, block.x, block.y, block.z, nPtr, nInt);
    for (int i = 0; i < nPtr; ++i) fprintf(stderr, "%s0x%llx", i ? "," : "", (unsigned long long)ptrs[i]);
    fprintf(stderr, "] ints=[");
    for (int i = 0; i < nInt; ++i) fprintf(stderr, "%s%u", i ? "," : "", ints[i]);
    fprintf(stderr, "]\n");

    // Queued, not sent (mirrors AMD's STATUS.md §26 batching) --
    // nvFlushLaunchQueue() sends the whole backlog as one RPC round-trip,
    // called before any h2d/d2h/sync/fini so ordering relative to memory
    // operations is preserved.
    NVPendingLaunch pl;
    pl.kernel = ke->name;
    pl.grid[0] = grid.x; pl.grid[1] = grid.y; pl.grid[2] = grid.z;
    pl.block[0] = block.x; pl.block[1] = block.y; pl.block[2] = block.z;
    pl.ptrs.assign(ptrs, ptrs + nPtr);
    pl.ints.assign(ints, ints + nInt);
    g_nvPendingLaunches.push_back(std::move(pl));
}

} // namespace tinygpu_device

#endif // FW_TINYGPU
