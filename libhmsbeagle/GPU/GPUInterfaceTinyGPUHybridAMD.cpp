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
 * code. This file just sends length-prefixed JSON commands over a
 * dedicated socketpair and reads back replies. The daemon runs tinygrad over
 * this plugin's own TinyGPU.app connection, which it inherits (TinyGPU.app
 * serves one client at a time).
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
#include "libhmsbeagle/GPU/TinyGPUTransport.h"
#include "libhmsbeagle/GPU/TinyGPUHybridAMDRuntime.h"

// The AMD path compiles the real OpenCL-C source -- see the plan's compiler-backend decision (comgr compiles BEAGLE's
// existing FW_OPENCL kernels unmodified, not a HIP port). Its KERNELS_STRING_<PREC>_<N> macros are the only ones in the
// plugin: GPUInterface.h's FW_TINYGPU branch includes only the PTX's stamp (TODO.md plan step C13).
#include "libhmsbeagle/GPU/kernels/BeagleOpenCL_kernels.h"
#ifdef TINYGPU_AMD_HSACO
#include "libhmsbeagle/GPU/kernels/TinyGPUAMDHsaco.h"   // the build's ahead-of-time HSACOs (TODO.md plan step A1j)
#endif

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

// The build's ahead-of-time HSACO of a variant ("SP_4" ... "DP_256") for an arch, compiled as the daemon would compile it
// at run time (tinygpu_amd_compile, golden_amd_hsaco.py), or null when the build had no comgr or not that arch.
// BEAGLE_AMD_AOT=0 (the harness's, for A/B) takes the daemon's run-time compile instead.
static const unsigned char* amd_embedded_hsaco(const std::string& variant, const std::string& arch, size_t& n) {
    n = 0;
    const char* aot = getenv("BEAGLE_AMD_AOT");
    if (aot && strcmp(aot, "0") == 0) return nullptr;
#ifdef TINYGPU_AMD_HSACO
    for (const TinyGPUAMDHsaco& h : kTinyGPUAMDHsacos)
        if (variant == h.variant && arch == h.arch) { n = (size_t)(h.end - h.begin); return h.begin; }
#else
    (void)variant; (void)arch;
#endif
    return nullptr;
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

// ── Command-socket I/O: JSON messages, each preceded by its byte length as a
// 4-byte little-endian uint32 (so a message is two reads, not one recv() per
// byte; the NV pair's framing, TODO.md "Runtime roadmap" Step 2), with raw
// bytes immediately following for h2d (request) / d2h (reply) ───────────────

static void amd_send_all(int fd, const void* buf, size_t n) {
    const uint8_t* p = (const uint8_t*)buf;
    while (n) { ssize_t r = ::send(fd, p, n, 0); if (r <= 0) return; p += r; n -= (size_t)r; }
}
static bool amd_recv_all(int fd, void* buf, size_t n) {
    uint8_t* p = (uint8_t*)buf;
    while (n) { ssize_t r = ::recv(fd, p, n, MSG_WAITALL); if (r <= 0) return false; p += r; n -= (size_t)r; }
    return true;
}
static void amd_send_msg(int fd, const std::string& json) {
    uint32_t n = (uint32_t)json.size();  // little-endian host (arm64/x86_64), as the daemon's "<I" expects
    std::string s((const char*)&n, 4);
    s += json;
    amd_send_all(fd, s.data(), s.size());
}
static std::string amd_recv_msg(int fd) {
    uint32_t n = 0;
    if (!amd_recv_all(fd, &n, 4)) return "";
    std::string s(n, '\0');
    if (n && !amd_recv_all(fd, &s[0], n)) return "";
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
    AMDRuntime* rt = nullptr;   // the C++ runtime after the handoff (TODO.md plan step A1g)
};

// The default since 2026-10-01 (the user's choice, STATUS.md R69): after the boot the daemon hands the GPU's queues over to
// the C++ runtime (TinyGPUHybridAMDRuntime.h), on a gfx11 card (the runtime's only target; any other keeps the daemon).
// BEAGLE_AMD_CPP=0 keeps every operation an RPC to the daemon.
static bool amd_cpp(const std::string& arch) {
    static const bool on = [] { const char* e = getenv("BEAGLE_AMD_CPP"); return !(e && strcmp(e, "0") == 0); }();
    return on && arch.rfind("gfx11", 0) == 0;
}

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

static void amdFlushLaunchQueueCpp();

static void amdFlushLaunchQueue() {
    if (!g_amd || g_amdPendingLaunches.empty()) return;
    if (g_amd->rt) { amdFlushLaunchQueueCpp(); return; }
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

    amd_send_msg(g_amd->cmd_sock, cmd);
    std::string resp = amd_recv_msg(g_amd->cmd_sock);
    amd_profile_end("launch_batch", t0);
    if (resp.empty() || !amd_json_ok(resp))
        fprintf(stderr, "TinyGPU/AMD: launch_batch(%zu kernels) failed: %s\n", n, resp.c_str());
}

// The daemon's chained launch_batch in C++ (TinyGPUHybridAMDDispatch.h): one queue per 1024 launches, each a timeline wait
// and memory_barrier, the execs with their kernargs slots, then a signal and a submit. A kernargs wrap first waits for the
// GPU to finish everything submitted (TinyGPUHybridAMDRuntime.h).
static void amdFlushLaunchQueueCpp() {
    AMDRuntime& rt = *g_amd->rt;
    auto t0 = amd_profile_start();
    const size_t n = g_amdPendingLaunches.size();
    AMDComputeQueue q;
    bool open = false;
    for (size_t i = 0; i < n && !rt.error; ++i) {
        const AMDPendingLaunch& pl = g_amdPendingLaunches[i];
        auto it = rt.kernels.find(pl.kernel);
        if (it == rt.kernels.end()) { rt.fail("launch of " + pl.kernel + ": not in the HSACO"); break; }
        const AMDKernel& k = it->second.k;
        if (pl.ptrs.size() * 8 + pl.ints.size() * 4 > k.kernargs_alloc_size) {
            rt.fail("launch of " + pl.kernel + ": its arguments do not fit its " + std::to_string(k.kernargs_alloc_size) + "-byte kernargs");
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
            open = false;
        }
    }
    g_amdPendingLaunches.clear();
    amd_profile_end("launch_batch (C++)", t0);
    if (rt.error) fprintf(stderr, "TinyGPU/AMD: launch_batch(%zu kernels) failed: %s\n", n, rt.error_msg.c_str());
}

[[noreturn]] static void amd_safe_exit(int code) {
    fflush(stderr);
    if (g_amd) {
        if (g_amd->cmd_sock >= 0) {
            amd_send_msg(g_amd->cmd_sock, "{\"cmd\":\"fini\"}");
            amd_recv_msg(g_amd->cmd_sock);  // best-effort ack, ignore content
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

// The daemon's socket.send_fds: one byte carrying the fds as SCM_RIGHTS
static bool amd_recv_fds(int sock, int* fds, int n) {
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

// cmd_handoff (TODO.md plan step A1e): the daemon's flat JSON, the HSACO it compiled (none when this side has the build's,
// plan step A1j), then the sysmem fds. From the reply on, the queues are this side's: a failure here is fatal.
// BEAGLE_AMD_DATA_MB sizes the VRAM pool (the daemon's default: half the VRAM).
static AMDRuntime* amdHandoff(AMDHybridState* g, const std::string& variant, const unsigned char* embedded, size_t embedded_size) {
    auto t0 = amd_profile_start();
    const char* mb = getenv("BEAGLE_AMD_DATA_MB");
    const uint64_t pool = mb ? strtoull(mb, nullptr, 10) << 20 : 0;
    amd_send_msg(g->cmd_sock, "{\"cmd\":\"handoff\",\"pool_size\":" + std::to_string(pool) + ",\"variant\":\"" + variant + "\"}");
    const std::string js = amd_recv_msg(g->cmd_sock);
    AMDHandoff h;
    std::string err = js.empty() || !amd_json_ok(js) ? "the daemon's reply: " + js : amd_parse_handoff(js, h);
    if (!err.empty()) { fprintf(stderr, "TinyGPU/AMD: handoff failed: %s\n", err.c_str()); return nullptr; }
    std::vector<uint8_t> blob(h.blob_size);
    int fds[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
    if (!amd_recv_all(g->cmd_sock, blob.data(), blob.size()) || !amd_recv_fds(g->cmd_sock, fds, (int)h.nmaps)) {
        fprintf(stderr, "TinyGPU/AMD: handoff: the daemon connection was lost\n");
        return nullptr;
    }
    AMDRuntime* rt = new AMDRuntime;
    err = amd_runtime_attach(*rt, h, fds, tg_transport());
    const unsigned char* hsaco = embedded ? embedded : blob.data();
    const size_t hsaco_size = embedded ? embedded_size : blob.size();
    if (err.empty() && hsaco_size == 0) err = "no HSACO: the build embedded none for this card and the daemon compiled none";
    if (err.empty()) err = amd_runtime_load_programs(*rt, hsaco, hsaco_size);
    if (!err.empty()) {
        fprintf(stderr, "TinyGPU/AMD: handoff: %s\n", err.c_str());
        amd_runtime_detach(*rt);
        delete rt;
        return nullptr;
    }
    amd_profile_end("handoff", t0);
    fprintf(stderr, "TinyGPU/AMD: C++ runtime: handed over after boot (VRAM pool %llu MiB, %zu kernels, scratch %llu MiB, timeline %llu)\n",
            (unsigned long long)(h.pool_size >> 20), rt->kernels.size(), (unsigned long long)(rt->exec.scratch_size >> 20),
            (unsigned long long)rt->timeline_value);
    return rt;
}

// ── amdDispatchDaemonSetup: spawn amd_dispatch_daemon.py over a dedicated
// socketpair, handing it the plugin's TinyGPU.app connection (tg_fd) for
// tinygrad to use instead of connecting itself, then send "boot" and
// "compile_all". ───────────────────────────────────────────────────────────

static AMDHybridState* amdDispatchDaemonSetup(const char* kernel_code, const std::string& variant, int tg_fd) {
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
        fcntl(tg_fd, F_SETFD, 0);   // the child's copy only: the connection survives execvp, and no other child gets it
        char fd_str[16]; snprintf(fd_str, sizeof(fd_str), "%d", sv[1]);
        char tg_str[16]; snprintf(tg_str, sizeof(tg_str), "%d", tg_fd);
        char* argv[] = { (char*)pypath.c_str(), (char*)helper, fd_str, tg_str, nullptr };
        execvp(pypath.c_str(), argv);
        fprintf(stderr, "TinyGPU/AMD: execvp %s failed: %s\n", pypath.c_str(), strerror(errno));
        _exit(1);
    }
    close(sv[1]);

    AMDHybridState* g = new AMDHybridState{};
    g->cmd_sock = sv[0];
    g->daemon_pid = pid;

    fprintf(stderr, "TinyGPU/AMD: sending boot command...\n"); fflush(stderr);
    amd_send_msg(g->cmd_sock, "{\"cmd\":\"boot\"}");
    std::string resp = amd_recv_msg(g->cmd_sock);
    if (resp.empty() || !amd_json_ok(resp)) {
        fprintf(stderr, "TinyGPU/AMD: boot failed: %s\n", resp.c_str());
        delete g;
        return nullptr;
    }
    const std::string arch = amd_json_str(resp, "arch");
    fprintf(stderr, "TinyGPU/AMD: daemon booted — arch=%s\n", arch.c_str());

    // The C++ runtime with the build's HSACO for this card: no run-time compile (TODO.md plan step A1j)
    size_t aot_size = 0;
    const unsigned char* aot = amd_cpp(arch) ? amd_embedded_hsaco(variant, arch, aot_size) : nullptr;
    if (aot) {
        fprintf(stderr, "TinyGPU/AMD: the build's ahead-of-time HSACO %s for %s (%zu bytes): no run-time compile\n", variant.c_str(),
                arch.c_str(), aot_size);
        if (!(g->rt = amdHandoff(g, variant, aot, aot_size))) { delete g; return nullptr; }
        for (const auto& kv : g->rt->kernels) g_amdKernels[kv.first] = new AMDKernelHandle{kv.first};
        fflush(stderr);
        return g;
    }

    if (kernel_code && kernel_code[0]) {
        char cl_path[256];
        snprintf(cl_path, sizeof(cl_path), "/tmp/beagle_amd_all_%d.cl", getpid());
        amd_write_file(cl_path, kernel_code, strlen(kernel_code));

        char cmd[512];
        snprintf(cmd, sizeof(cmd), "{\"cmd\":\"compile_all\",\"cl_path\":\"%s\"}", cl_path);
        fprintf(stderr, "TinyGPU/AMD: precompile_all_kernels — compiling all kernels (comgr × 1, via daemon)…\n");
        fflush(stderr);
        amd_send_msg(g->cmd_sock, cmd);
        resp = amd_recv_msg(g->cmd_sock);
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
        if (amd_cpp(arch) && !(g->rt = amdHandoff(g, variant, nullptr, 0))) { delete g; return nullptr; }
    }
    fflush(stderr);
    return g;
}

// ── GPUInterface entry points ─────────────────────────────────────────────────

void AmdSetDevice(GPUInterface* self, int paddedStateCount, int categoryCount,
                   int patternCount, int unpaddedPatternCount, int tipCount, long flags) {
    // The daemon runs tinygrad over Initialize()'s TinyGPU.app connection
    // (self->tgpuSock, plan step C3), inherited, and this side sends nothing
    // on it while the daemon lives; the GPUInterface destructor closes it
    // after AmdFini. TinyGPU.app serves one client at a time: a second
    // connection's first RPC (AMDDevice's resize_bar()) hung forever while
    // this one sat open (STATUS.md AMD §21), which is why this side used to
    // close it first and let the daemon connect itself.
    const bool dp = (flags & BEAGLE_FLAG_PRECISION_DOUBLE) != 0;
    g_amd = amdDispatchDaemonSetup(amd_opencl_kernel_source(paddedStateCount, dp), (dp ? "DP_" : "SP_") + std::to_string(paddedStateCount),
                                   self->tgpuSock);
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
    if (g_amd->rt) {
        if (!g_amd->rt->synchronize()) fprintf(stderr, "TinyGPU/AMD: sync failed: %s\n", g_amd->rt->error_msg.c_str());
        amd_profile_end("sync (C++)", t0);
        return;
    }
    amd_send_msg(g_amd->cmd_sock, "{\"cmd\":\"sync\"}");
    std::string resp = amd_recv_msg(g_amd->cmd_sock);
    amd_profile_end("sync", t0);
    if (resp.empty() || !amd_json_ok(resp))
        fprintf(stderr, "TinyGPU/AMD: sync failed: %s\n", resp.c_str());
}

GPUPtr AmdAllocateMemory(size_t sz) {
    if (!g_amd) return 0;
    auto t0 = amd_profile_start();
    if (g_amd->rt) {
        uint64_t va = 0;
        if (!g_amd->rt->alloc(sz, va)) fprintf(stderr, "TinyGPU/AMD: alloc(%zu): the VRAM pool has %llu bytes left\n", sz,
                                                (unsigned long long)g_amd->rt->available());
        amd_profile_end("alloc (C++)", t0);
        return (GPUPtr)va;
    }
    char cmd[128];
    snprintf(cmd, sizeof(cmd), "{\"cmd\":\"alloc\",\"size\":%zu}", sz);
    amd_send_msg(g_amd->cmd_sock, cmd);
    std::string resp = amd_recv_msg(g_amd->cmd_sock);
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
    if (g_amd->rt) {
        if (!amd_copyin(*g_amd->rt, g_amd->rt->staging, (uint64_t)dst, (const uint8_t*)src, sz))
            fprintf(stderr, "TinyGPU/AMD: h2d(addr=0x%llx, sz=%zu) failed: %s\n", (unsigned long long)dst, sz, g_amd->rt->error_msg.c_str());
        amd_profile_end("h2d (C++)", t0);
        return;
    }
    char cmd[128];
    snprintf(cmd, sizeof(cmd), "{\"cmd\":\"h2d\",\"addr\":%llu,\"size\":%zu}", (unsigned long long)dst, sz);
    amd_send_msg(g_amd->cmd_sock, cmd);
    amd_send_all(g_amd->cmd_sock, src, sz);
    std::string resp = amd_recv_msg(g_amd->cmd_sock);
    amd_profile_end("h2d", t0);
    if (resp.empty() || !amd_json_ok(resp))
        fprintf(stderr, "TinyGPU/AMD: h2d(addr=0x%llx, sz=%zu) failed: %s\n", (unsigned long long)dst, sz, resp.c_str());
}

void AmdMemcpyDeviceToHost(void* dst, const GPUPtr src, size_t sz) {
    if (!g_amd || !dst || !sz) return;
    amdFlushLaunchQueue();  // preserve ordering: queued launches must complete before this read
    auto t0 = amd_profile_start();
    if (g_amd->rt) {
        if (!amd_copyout(*g_amd->rt, g_amd->rt->staging, (uint8_t*)dst, (uint64_t)src, sz))
            fprintf(stderr, "TinyGPU/AMD: d2h(addr=0x%llx, sz=%zu) failed: %s\n", (unsigned long long)src, sz, g_amd->rt->error_msg.c_str());
        amd_profile_end("d2h (C++)", t0);
        return;
    }
    char cmd[128];
    snprintf(cmd, sizeof(cmd), "{\"cmd\":\"d2h\",\"addr\":%llu,\"size\":%zu}", (unsigned long long)src, sz);
    amd_send_msg(g_amd->cmd_sock, cmd);
    std::string resp = amd_recv_msg(g_amd->cmd_sock);
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
    if (g_amd && g_amd->rt) return (size_t)g_amd->rt->available();   // the C++ runtime's pool
    return g_amd ? (size_t)(1ull << 30) : 0;
}

void AmdFini() {
    if (!g_amd) return;
    amdFlushLaunchQueue();  // don't silently drop queued-but-unsent launches
    if (g_amd->rt && !g_amd->rt->synchronize())   // the daemon's fini (AMDev.fini) then dequeues the queues
        fprintf(stderr, "TinyGPU/AMD: the last synchronize failed: %s\n", g_amd->rt->error_msg.c_str());
    for (auto& kv : g_amdKernels) delete kv.second;
    g_amdKernels.clear();
    if (g_amd->cmd_sock >= 0) {
        amd_send_msg(g_amd->cmd_sock, "{\"cmd\":\"fini\"}");
        amd_recv_msg(g_amd->cmd_sock);
    }
    if (g_amd->daemon_pid > 0) {
        for (int i = 0; i < 100; ++i) {
            int st = 0;
            if (waitpid(g_amd->daemon_pid, &st, WNOHANG) > 0) { g_amd->daemon_pid = 0; break; }
            usleep(100000);
        }
    }
    if (g_amd->cmd_sock >= 0) close(g_amd->cmd_sock);
    if (g_amd->rt) { amd_runtime_detach(*g_amd->rt); delete g_amd->rt; }
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
