/*
 * TinyGPUHybridNVGuard.h -- what the plugin and beagle-tinygpu-guard (tinygpu_guard.cpp, TODO.md plan step C10) share: the
 * setup message, the state page's layout, and the guard's spawn.
 *
 * The setup comes in two stages (plan step C11). The plugin spawns the guard before its first request to the GPU with what
 * holding takes (kGuardSetupHold: the connection, its lock and the state page, three fds), and once the NVDevice is built it
 * sends 'S' and the rest (kGuardSetupRest: the queues and the timeline, two fds, with the teardown's parameters). Until the
 * rest arrives a guard can only hold, which is what a death during the boot needs: the state page says a frame is in flight
 * throughout, or that the falcons' boot has not begun (flcn_init), in which case the guard closes.
 *
 * The plugin creates a socketpair and spawns the guard with posix_spawn (POSIX_SPAWN_SETSID: its own session, so a terminal's
 * Ctrl-C or hangup does not reach it; POSIX_SPAWN_CLOEXEC_DEFAULT: no fd of the host but its end of the pair, as fd 3, and
 * stderr; stdin and stdout are /dev/null, so no fd it receives later takes a standard number that a stray print would write
 * to). Over the pair each stage is one GuardSetup and its fds (SCM_RIGHTS). The guard replies 'R' once it has mapped the
 * first. Later the plugin sends 'C' (clean: it tore the GPU down itself), 'H' (hold: its unload was not confirmed) or 'N' (the
 * boot stopped before GSP-RM started, so nothing is to be unloaded); an EOF without any of them means the plugin is gone, or
 * lost the GPU (plan step C12), and the guard decides from the state page.
 *
 * The AMD C++ boot (plan step A2k, GPUInterfaceTinyGPUHybridAMD.cpp) uses the same guard and messages with its own kinds:
 * kGuardSetupHoldAMD (the same three fds, its lock being tinygrad's am_usb4.lock: the AMD path releases the transport's), and
 * once its AMDev is booted, before any queue is set up, kGuardSetupRestAMD (no fds), followed on the socketpair by what
 * AMDev.fini needs (amd_fini_size bytes: TinyGPUHybridAMDBoot.h's AMFiniState). Its state page uses the phase (am_boot until
 * just before the first queue goes live, then dispatch, and teardown in its own fini) and the in-flight word, which its
 * transport keeps around every request.
 */

#ifndef LIBHMSBEAGLE_GPU_TINYGPUHYBRIDNVGUARD_H
#define LIBHMSBEAGLE_GPU_TINYGPUHYBRIDNVGUARD_H

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <string>

#include <fcntl.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <unistd.h>

#include "libhmsbeagle/GPU/TinyGPUHybridNVFalcon.h"

extern char** environ;

namespace tinygpu_device {

constexpr int kGuardFd = 3;   // the guard's end of the socketpair
enum { kGuardTinyGPU, kGuardLock, kGuardQueues, kGuardState, kGuardSignal, kGuardFds };
// the state page (GPUInterfaceTinyGPUHybridNV.cpp's kNVDState* and kNVDPhase*, which static_assert they match these)
enum { kGuardStatePhase, kGuardStateInFlight, kGuardStateLastSubmitted, kGuardStateSeq, kNVDStateWordsGuard };
constexpr uint64_t kGuardPhaseDispatch = 1, kGuardPhaseTeardown = 2, kGuardPhaseGspInit = 3, kGuardPhaseFlcnInit = 4;
constexpr uint64_t kGuardPhaseAMBoot = 5;   // the AMD boot before any queue went live: nothing on the GPU reads sysmem

// What the guard's teardown needs: the plugin's NVDTeardown, from the C++ boot, and BAR0's size. A kGuardSetupHold setup
// carries only the plugin's pid; the rest comes with kGuardSetupRest. The AMD kinds: see above.
enum : uint32_t { kGuardSetupHold = 1, kGuardSetupRest = 2, kGuardSetupHoldAMD = 3, kGuardSetupRestAMD = 4 };
struct GuardSetup {
    uint32_t magic, size;   // kGuardMagic, sizeof(GuardSetup): a guard built from other sources refuses the setup
    uint32_t kind, nfds;    // kGuardSetup*, and how many fds come with it (guard_setup_nfds)
    uint64_t queues_size, cmdq_off, statq_off, queue_size, libos_args_sysmem, bar0_size, signal_size;
    uint32_t chip_id, cot, level0, parent_pid;
    NVTeardownImages images;
    char chip_name[16];
    uint64_t amd_fini_size;   // kGuardSetupRestAMD: the size of the AMFiniState that follows
};
constexpr uint32_t kGuardMagic = 0x44475447;   // "GTGD"

// the fds each kind of setup carries, in this order: Hold (and HoldAMD) the connection, its lock and the state page; Rest the
// queues and the timeline; RestAMD none
inline uint32_t guard_setup_nfds(uint32_t kind) {
    return kind == kGuardSetupHold || kind == kGuardSetupHoldAMD ? 3 : kind == kGuardSetupRest ? 2 : 0;
}

// sendmsg of the setup and its fds, in one message
inline bool guard_send_setup(int sock, const GuardSetup& g, const int* fds) {
    struct iovec iov = {(void*)&g, sizeof(g)};
    char ctl[CMSG_SPACE(sizeof(int) * kGuardFds)] = {};
    struct msghdr mh{};
    mh.msg_iov = &iov;
    mh.msg_iovlen = 1;
    if (g.nfds) {
        mh.msg_control = ctl;
        mh.msg_controllen = CMSG_SPACE(sizeof(int) * g.nfds);
        struct cmsghdr* c = CMSG_FIRSTHDR(&mh);
        c->cmsg_level = SOL_SOCKET;
        c->cmsg_type = SCM_RIGHTS;
        c->cmsg_len = CMSG_LEN(sizeof(int) * g.nfds);
        memcpy(CMSG_DATA(c), fds, sizeof(int) * g.nfds);
    }
    ssize_t n;
    while ((n = sendmsg(sock, &mh, 0)) < 0 && errno == EINTR) {}
    return n == (ssize_t)sizeof(g);
}

// The bytes that follow a kGuardSetupRestAMD (its AMFiniState): all of them, or false
inline bool guard_send_all(int sock, const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*)p;
    while (n) {
        ssize_t r = send(sock, b, n, 0);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return false;
        b += r;
        n -= (size_t)r;
    }
    return true;
}
inline bool guard_recv_all(int sock, void* p, size_t n) {
    uint8_t* b = (uint8_t*)p;
    while (n) {
        ssize_t r = recv(sock, b, n, MSG_WAITALL);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return false;
        b += r;
        n -= (size_t)r;
    }
    return true;
}

// recvmsg of one, its fds in fds[0..nfds): "" or why not
inline std::string guard_recv_setup(int sock, GuardSetup& g, int (&fds)[kGuardFds]) {
    struct iovec iov = {&g, sizeof(g)};
    char ctl[CMSG_SPACE(sizeof(int) * kGuardFds)] = {};
    struct msghdr mh{};
    mh.msg_iov = &iov;
    mh.msg_iovlen = 1;
    mh.msg_control = ctl;
    mh.msg_controllen = sizeof(ctl);
    ssize_t n;
    while ((n = recvmsg(sock, &mh, MSG_WAITALL)) < 0 && errno == EINTR) {}
    if (n != (ssize_t)sizeof(g)) return "a setup of " + std::to_string(n) + " bytes, not " + std::to_string(sizeof(g));
    if (g.magic != kGuardMagic || g.size != sizeof(g)) return "a setup from other sources (magic or size differs)";
    if (g.kind < kGuardSetupHold || g.kind > kGuardSetupRestAMD || g.nfds != guard_setup_nfds(g.kind)) return "a setup of an unknown kind";
    struct cmsghdr* c = CMSG_FIRSTHDR(&mh);
    if (g.nfds == 0) {
        if (c) return "a setup with fds it should not carry";
    } else {
        if (!c || c->cmsg_type != SCM_RIGHTS || c->cmsg_len != CMSG_LEN(sizeof(int) * g.nfds)) return "the setup came without its fds";
        memcpy(fds, CMSG_DATA(c), sizeof(int) * g.nfds);
    }
    g.chip_name[sizeof(g.chip_name) - 1] = 0;
    return "";
}

// posix_spawn of the guard executable at path, with its end of a new socketpair as fd 3: the plugin's end in ctl (close-on-exec),
// the guard's pid in pid; "" or why not
inline std::string guard_spawn(const std::string& path, int& ctl, pid_t& pid) {
    int sv[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) return std::string("socketpair: ") + strerror(errno);
    posix_spawn_file_actions_t fa;
    posix_spawnattr_t at;
    posix_spawn_file_actions_init(&fa);
    posix_spawnattr_init(&at);
    posix_spawn_file_actions_addopen(&fa, 0, "/dev/null", O_RDONLY, 0);
    posix_spawn_file_actions_addopen(&fa, 1, "/dev/null", O_WRONLY, 0);
    posix_spawn_file_actions_addinherit_np(&fa, 2);   // the host's stderr: the guard's hold message is seen where the plugin's is
    posix_spawn_file_actions_adddup2(&fa, sv[1], kGuardFd);
    posix_spawnattr_setflags(&at, POSIX_SPAWN_SETSID | POSIX_SPAWN_CLOEXEC_DEFAULT);
    char* argv[] = {(char*)path.c_str(), nullptr};
    int rc = posix_spawn(&pid, path.c_str(), &fa, &at, argv, environ);
    posix_spawn_file_actions_destroy(&fa);
    posix_spawnattr_destroy(&at);
    close(sv[1]);
    if (rc != 0) { close(sv[0]); return "posix_spawn " + path + ": " + strerror(rc); }
    int one = 1;   // a guard gone early is EPIPE, not SIGPIPE in the host
    setsockopt(sv[0], SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
    fcntl(sv[0], F_SETFD, FD_CLOEXEC);
    ctl = sv[0];
    return "";
}

} // namespace tinygpu_device

#endif // LIBHMSBEAGLE_GPU_TINYGPUHYBRIDNVGUARD_H
