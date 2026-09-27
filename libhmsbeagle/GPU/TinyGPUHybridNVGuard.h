/*
 * TinyGPUHybridNVGuard.h -- what the plugin and beagle-tinygpu-guard (tinygpu_guard.cpp, TODO.md plan step C10) share: the
 * setup message, the state page's layout, and the guard's spawn.
 *
 * The plugin creates a socketpair and spawns the guard with posix_spawn (POSIX_SPAWN_SETSID: its own session, so a terminal's
 * Ctrl-C or hangup does not reach it; POSIX_SPAWN_CLOEXEC_DEFAULT: no fd of the host but its end of the pair, as fd 3, and
 * stderr; stdin and stdout are /dev/null, so no fd it receives later takes a standard number that a stray print would write
 * to). Over the
 * pair it sends one GuardSetup with five fds (SCM_RIGHTS): the TinyGPU.app connection, its lock, the GSP queues' sysmem, the
 * state page and the C++ timeline's sysmem. The guard replies 'R' once it has mapped them. Later the plugin sends 'C' (clean:
 * it tore the GPU down itself), 'H' (hold: its unload was not confirmed) or 'X' (stand down: the daemon kept the keeper role);
 * an EOF without any of them means the plugin is gone.
 *
 * The keeper role passes from the daemon to the guard through the state page's keeper word, which the plugin sets to
 * kGuardKeeperGuard once the guard is ready, before it asks the daemon to release the role (cmd_release: the daemon replies and
 * exits without a word to the GPU). The guard and the daemon read the word only after the plugin's last write of it (at the
 * plugin's death, or at the release), so at the plugin's death exactly one of them acts: the guard if the word says so (the
 * daemon, at the release or at its command socket's EOF, exits without a word), else the daemon (the guard exits). If the
 * daemon refuses the release, the plugin sets the word back before it tells the guard to stand down.
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
enum { kGuardStatePhase, kGuardStateInFlight, kGuardStateLastSubmitted, kGuardStateSeq, kGuardStateKeeper, kNVDStateWordsGuard };
constexpr uint64_t kGuardPhaseDispatch = 1, kGuardPhaseTeardown = 2, kGuardPhaseGspInit = 3, kGuardPhaseFlcnInit = 4;
constexpr uint64_t kGuardKeeperDaemon = 0, kGuardKeeperGuard = 1;   // the keeper word

// What the guard's teardown needs: the plugin's NVDTeardown (cmd_rm_export's reply) and BAR0's size (the daemon mapped it).
struct GuardSetup {
    uint32_t magic, size;   // kGuardMagic, sizeof(GuardSetup): a guard built from other sources refuses the setup
    uint64_t queues_size, cmdq_off, statq_off, queue_size, libos_args_sysmem, bar0_size, signal_size;
    uint32_t chip_id, cot, level0, parent_pid;
    NVTeardownImages images;
    char chip_name[16], level_name[16];
};
constexpr uint32_t kGuardMagic = 0x44475447;   // "GTGD"

// sendmsg of the setup and the five fds, in one message
inline bool guard_send_setup(int sock, const GuardSetup& g, const int (&fds)[kGuardFds]) {
    struct iovec iov = {(void*)&g, sizeof(g)};
    char ctl[CMSG_SPACE(sizeof(int) * kGuardFds)] = {};
    struct msghdr mh{};
    mh.msg_iov = &iov;
    mh.msg_iovlen = 1;
    mh.msg_control = ctl;
    mh.msg_controllen = sizeof(ctl);
    struct cmsghdr* c = CMSG_FIRSTHDR(&mh);
    c->cmsg_level = SOL_SOCKET;
    c->cmsg_type = SCM_RIGHTS;
    c->cmsg_len = CMSG_LEN(sizeof(int) * kGuardFds);
    memcpy(CMSG_DATA(c), fds, sizeof(int) * kGuardFds);
    ssize_t n;
    while ((n = sendmsg(sock, &mh, 0)) < 0 && errno == EINTR) {}
    return n == (ssize_t)sizeof(g);
}

// recvmsg of it: "" or why not
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
    struct cmsghdr* c = CMSG_FIRSTHDR(&mh);
    if (!c || c->cmsg_type != SCM_RIGHTS || c->cmsg_len != CMSG_LEN(sizeof(int) * kGuardFds)) return "the setup came without its five fds";
    memcpy(fds, CMSG_DATA(c), sizeof(int) * kGuardFds);
    if (g.magic != kGuardMagic || g.size != sizeof(g)) return "a setup from other sources (magic or size differs)";
    g.chip_name[sizeof(g.chip_name) - 1] = g.level_name[sizeof(g.level_name) - 1] = 0;
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
