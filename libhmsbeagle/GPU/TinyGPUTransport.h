/*
 * TinyGPUTransport.h
 *
 * The C++ client of TinyGPU.app's socket protocol (TODO.md plan step C3): a statement-by-statement port of tinygrad's
 * RemotePCIDevice and APLRemotePCIDevice (tinygrad/runtime/support/system.py:311-447 at a9830e2b4), the Python client
 * BEAGLE's daemon uses. TinyGPU.app itself, its server and DriverKit extension, stays external and unchanged: this only
 * speaks its protocol (the server is extra/usbgpu/tbgpu/installer/Shared/server.c at the same pin). Each method names
 * the tinygrad code it mirrors.
 *
 * Where it departs from the Python, it is for a failure the server does not report:
 *   - limits the server does not check before acting: at most 64 MB per message (a write's payload is received into a
 *     64 MB buffer before it is validated, server.c:243-246), offset + length inside the BAR (out-of-range writes are
 *     dropped silently, :167-169), MMIO only on a BAR mapped in the session, and at most 128 sysmem allocations
 *     (:129-131);
 *   - the MAP_SYSMEM_FD reply: its status is read before an fd is expected (a failure carries none), a truncated
 *     control message is an error, and the DMA segment list is checked (at most 32 segments, 4 KiB aligned, below
 *     2^40, covering the size: the dext's limits);
 *   - every error reply's message is drained, and an MMIO_READ error carries no data, so the stream stays in step;
 *   - a failed send or receive marks the connection lost, and nothing more is sent on it: a frame may have been cut;
 *   - tinygrad's lock is taken before connecting, and the server is started only when nothing listens (ENOENT or
 *     ECONNREFUSED), in its own session with stdio on /dev/null, so no second server is started beside a live one and
 *     a terminal's Ctrl-C cannot reach it; a failed connect gets a fresh socket (POSIX leaves the old one unspecified);
 *   - TinyGPU.app is never installed: an app whose binaries differ from the pinned release is refused, with guidance
 *     (plan decision 25).
 * There is no RESET (a PCIe FLR), PROBE or SYSMEM_* call.
 */

#ifndef LIBHMSBEAGLE_GPU_TINYGPUTRANSPORT_H
#define LIBHMSBEAGLE_GPU_TINYGPUTRANSPORT_H

#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <CommonCrypto/CommonDigest.h>
#include <fcntl.h>
#include <spawn.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <sys/un.h>
#include <unistd.h>

extern char** environ;

namespace tinygpu_device {

enum TGCmd : uint8_t {   // RemoteCmd (system.py:311-312)
    TGC_PROBE, TGC_MAP_BAR, TGC_MAP_SYSMEM_FD, TGC_CFG_READ, TGC_CFG_WRITE, TGC_RESET, TGC_MMIO_READ, TGC_MMIO_WRITE,
    TGC_MAP_SYSMEM, TGC_SYSMEM_READ, TGC_SYSMEM_WRITE, TGC_RESIZE_BAR, TGC_PING
};

struct TGWrite { uint32_t bar; uint64_t off; const void* data; uint64_t len; };   // one posted MMIO_WRITE

// Step markers (plan step V1), mirrored in tinygpu_tests/replay/tgwire.py MARKERS: the C++ runtime's phases
enum TGMarker : uint32_t { TGM_HANDOFF = 0x100, TGM_PROGRAMS_LOADED = 0x101, TGM_FINI = 0x102 };

struct TGSysmem {                  // alloc_sysmem's (memview, paddrs)
    uint8_t* view = nullptr;       // the shared mapping; its first bytes held the segment list
    uint64_t mapped_size = 0;
    std::vector<uint64_t> paddrs;  // one device address per 4 KiB page
};

// tinygrad's temp(name): tempfile.gettempdir(), which is $TMPDIR (else /tmp), joined with name.
static inline std::string tg_temp_path(const char* name) {
    const char* tmpdir = getenv("TMPDIR");
    std::string dir = (tmpdir && tmpdir[0]) ? tmpdir : "/tmp";
    while (dir.size() > 1 && dir.back() == '/') dir.pop_back();
    return dir + "/" + name;
}

class TGTransport {
public:
    static constexpr uint64_t kMaxMessage = 64ull << 20;   // server.c BULK_BUF_SIZE
    static constexpr int kMaxBars = 6;                     // server.c MAX_BARS
    static constexpr int kMaxSysmem = 128;                 // server.c MAX_SYSMEM
    static constexpr size_t kMaxSegments = 32;             // the dext's DMA segment list
    static constexpr uint64_t kIovaLimit = 1ull << 40;     // the dext's device addressing (40 bits)

    int fd() const { return sock_; }
    int lock_fd() const { return lock_fd_; }
    bool lost() const { return lost_; }

    // APLRemotePCIDevice.__init__ (system.py:428-438), then RemotePCIDevice.__init__ (:387-392), for "NV:0". "" or why not.
    std::string open() {
        if (sock_ >= 0) return "a TinyGPU.app connection is already open in this process";   // never two owners of one
        const char* remote = getenv("APL_REMOTE_SOCK");
        std::string path = remote ? remote : tg_temp_path("tinygpu.sock"), err;
        bool checked = false;
        if (!remote || getenv("BEAGLE_TINYGPU_APP")) {   // APL_REMOTE_SOCK names another server, unless an app is named too
            if (!(err = check_app()).empty()) return err;
            checked = true;
        }
        if (!acquire_lock("nv_usb4.lock", err)) return err;
        struct sockaddr_un addr{};
        addr.sun_family = AF_UNIX;
        if (path.size() >= sizeof(addr.sun_path)) { release_lock(); return "socket path too long: " + path; }
        strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);
        const char* nl = getenv("BEAGLE_TINYGPU_NO_LAUNCH");
        bool no_launch = nl && nl[0] && strcmp(nl, "0") != 0;
        for (int i = 0; i < 100 && sock_ < 0; ++i) {
            int fd = socket(AF_UNIX, SOCK_STREAM, 0);
            if (fd < 0) { release_lock(); return std::string("socket: ") + strerror(errno); }
            fcntl(fd, F_SETFD, FD_CLOEXEC);
            if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) == 0) { sock_ = fd; break; }
            int e = errno;
            ::close(fd);
            if (e != ENOENT && e != ECONNREFUSED) { release_lock(); return "connect " + path + ": " + strerror(e); }
            if (i == 0) {
                if (no_launch) {
                    release_lock();
                    return "nothing is listening at " + path + " and BEAGLE_TINYGPU_NO_LAUNCH is set; not starting TinyGPU.app";
                }
                if ((!checked && !(err = check_app()).empty()) || !spawn_server(path, err)) { release_lock(); return err; }
            }
            usleep(50000);
        }
        if (sock_ < 0) { release_lock(); return "Failed to connect to TinyGPU server at " + path + "."; }
        lost_ = false;
        int big = 64 << 20, one = 1;   // macOS caps both buffers at kern.ipc.maxsockbuf, silently, as for tinygrad
        setsockopt(sock_, SOL_SOCKET, SO_SNDBUF, &big, sizeof(big));
        setsockopt(sock_, SOL_SOCKET, SO_RCVBUF, &big, sizeof(big));
        setsockopt(sock_, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));   // a lost server is EPIPE, not a signal
        return "";
    }

    // Plan step C10's guard: a connection (and its lock) another process of this session opened, whose BARs it mapped (seed_bar):
    // no connect and no lock of its own; the guard sends nothing on it while that process lives.
    void adopt(int sock, int lock_fd) {
        sock_ = sock;
        lock_fd_ = lock_fd;
        lost_ = false;
        int one = 1;
        setsockopt(sock_, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
    }

    // Plan step V1: with BEAGLE_TG_MARKERS=1, a step marker for the recording proxy and the replay server, which answer it
    // themselves and never forward it: a CFG_READ of the vendor ID with dev_id 'BEAG' (0x42454147), the marker's id as the BAR
    // and its argument as arg2. TinyGPU.app never reads dev_id (server.c:216-220), so there it is a harmless config read. Its
    // reply is read, so the stream stays in step; it is sent only while this side owns the connection, and nothing is
    // reported: a marker never changes what the plugin does.
    void marker(uint32_t id, uint64_t arg) {
        static const bool on = [] { const char* e = getenv("BEAGLE_TG_MARKERS"); return e && e[0] && strcmp(e, "0") != 0; }();
        std::string err;
        if (!on || !usable(err)) return;
        uint8_t hdr[33], resp[17];
        pack(hdr, TGC_CFG_READ, id, 0, 4, arg);
        const uint32_t dev = 0x42454147;
        memcpy(hdr + 1, &dev, 4);
        struct iovec iov = {hdr, 33};
        uint64_t r0, r1;
        if (send_iov(&iov, 1) && recv_all(resp, 17)) reply(resp, r0, r1, err);
    }

    // The connection and the lock go; a later open() is a new server session.
    void close() {
        if (sock_ >= 0) { ::close(sock_); sock_ = -1; }
        release_lock();
        lost_ = false;
        for (Bar& b : bars_) b = Bar();
        sysmem_count_ = 0;
        iovas_.clear();
    }

    void release_lock() {
        if (lock_fd_ >= 0) { ::close(lock_fd_); lock_fd_ = -1; }
    }

    // RemotePCIDevice.read_config (system.py:407)
    bool read_config(uint64_t off, uint64_t size, uint64_t& value, std::string& err) {
        uint64_t r1;
        return rpc(TGC_CFG_READ, off, size, 0, 0, value, r1, err);
    }

    // RemotePCIDevice.write_config (:408)
    bool write_config(uint64_t off, uint64_t value, uint64_t size, std::string& err) {
        uint64_t r0, r1;
        return rpc(TGC_CFG_WRITE, off, size, value, 0, r0, r1, err);
    }

    // PCIDevice.write_config_flush (:208-210): the write, then a read of it
    bool write_config_flush(uint64_t off, uint64_t value, uint64_t size, std::string& err) {
        uint64_t v;
        return write_config(off, value, size, err) && read_config(off, size, v, err);
    }

    // RemotePCIDevice.bar_info (:410-411), functools.cache'd: one MAP_BAR per BAR and session
    bool bar_info(uint32_t bar, uint64_t& addr, uint64_t& size, std::string& err) {
        if (bar >= (uint32_t)kMaxBars) { err = "no BAR " + std::to_string(bar); return false; }
        if (!bars_[bar].known) {
            uint64_t a, s;
            if (!rpc(TGC_MAP_BAR, 0, 0, 0, bar, a, s, err)) return false;
            bars_[bar] = Bar{true, a, s};
        }
        addr = bars_[bar].addr;
        size = bars_[bar].size;
        return true;
    }

    // A BAR someone sharing this connection already mapped in the session (the daemon, which hands over its size): MMIO
    // is checked against it with no MAP_BAR of our own, which would change the stream.
    void seed_bar(uint32_t bar, uint64_t size) {
        if (bar < (uint32_t)kMaxBars) bars_[bar] = Bar{true, 0, size};
    }

    // RemotePCIDevice.resize_bar (:414)
    bool resize_bar(uint32_t bar, std::string& err) {
        uint64_t r0, r1;
        return rpc(TGC_RESIZE_BAR, 0, 0, 0, bar, r0, r1, err);
    }

    // RemotePCIDevice._bulk_read (:394-396) of MMIO: one round trip
    bool bulk_read(uint32_t bar, uint64_t off, void* out, uint64_t len, std::string& err) {
        uint64_t r0, r1;
        return check_mmio(bar, off, len, err) && rpc(TGC_MMIO_READ, off, len, 0, bar, r0, r1, err, out, len);
    }

    // RemotePCIDevice._bulk_write (:397-399): posted, never answered
    bool bulk_write(uint32_t bar, uint64_t off, const void* data, uint64_t len, std::string& err) {
        TGWrite w{bar, off, data, len};
        return bulk_write_frame(&w, 1, err);
    }

    // Several posted writes in one send, each as _bulk_write frames it; every limit is checked before a byte goes out.
    // A failure with lost() false sent nothing.
    bool bulk_write_frame(const TGWrite* w, size_t n, std::string& err) {
        if (!usable(err)) return false;
        for (size_t i = 0; i < n; ++i)
            if (!check_mmio(w[i].bar, w[i].off, w[i].len, err)) return false;
        std::vector<uint8_t> hdrs(33 * n);
        std::vector<struct iovec> iov;
        for (size_t i = 0; i < n; ++i) {
            pack(&hdrs[33 * i], TGC_MMIO_WRITE, w[i].bar, w[i].off, w[i].len, 0);
            iov.push_back({&hdrs[33 * i], 33});
            if (w[i].len) iov.push_back({const_cast<void*>(w[i].data), (size_t)w[i].len});
        }
        if (!send_iov(iov.data(), (int)iov.size())) { err = "TinyGPU.app connection lost while sending"; return false; }
        return true;
    }

    // APLRemotePCIDevice.alloc_sysmem (:440-447). keep_fd, if given, receives a dup of the allocation's fd (tinygrad closes
    // it once mapped; the daemon's EOF path maps the C++ timeline from it, plan step C6).
    bool alloc_sysmem(uint64_t size, bool contiguous, TGSysmem& out, std::string& err, int* keep_fd = nullptr) {
        if (sysmem_count_ >= kMaxSysmem) {
            err = "a 129th sysmem allocation: TinyGPU.app keeps at most 128 per connection";
            return false;
        }
        uint64_t mapped, idx;
        int fd;
        if (!rpc_fd(TGC_MAP_SYSMEM_FD, size, contiguous ? 1 : 0, mapped, idx, fd, err)) return false;
        ++sysmem_count_;
        void* m = mmap(nullptr, mapped, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        int kept = keep_fd && m != MAP_FAILED ? fcntl(fd, F_DUPFD_CLOEXEC, 0) : -1;
        ::close(fd);
        if (m == MAP_FAILED) { err = std::string("mmap of the sysmem fd: ") + strerror(errno); return false; }
        // (paddr, size) pairs until a size of 0 at the start of the mapping, each expanded to 4 KiB pages
        const uint64_t* q = (const uint64_t*)m;
        std::vector<std::pair<uint64_t, uint64_t>> segs;
        uint64_t total = 0;
        for (uint64_t i = 0; 2 * i + 1 < mapped / 8 && q[2 * i + 1] != 0; ++i) {
            segs.push_back({q[2 * i], q[2 * i + 1]});
            total += q[2 * i + 1];
        }
        err.clear();
        if (segs.size() > kMaxSegments) err = "more than 32 DMA segments";
        for (auto& s : segs)
            if (err.empty() && (s.first % 0x1000 || s.first + s.second > kIovaLimit || s.first + s.second < s.first))
                err = "a DMA segment unaligned or at or above 2^40";
        if (err.empty() && total < size) err = "DMA segments shorter than the allocation";
        if (!err.empty()) { munmap(m, mapped); if (kept >= 0) ::close(kept); return false; }
        if (keep_fd) *keep_fd = kept;
        out.view = (uint8_t*)m;
        out.mapped_size = mapped;
        out.paddrs.clear();
        for (auto& s : segs)
            for (uint64_t off = 0; off < s.second; off += 0x1000) out.paddrs.push_back(s.first + off);
        out.paddrs.resize((size + 0xfff) / 0x1000);
        iovas_.insert(iovas_.end(), segs.begin(), segs.end());
        return true;
    }

    // The sysmem allocations others sharing this connection made (the daemon's, plan step C6): they count toward the 128.
    void seed_sysmem_count(int n) { sysmem_count_ = n; }

    // The IOVA whitelist (the C++ fence of plan steps C5-C6): whether [addr, addr + len) lies in one segment this
    // connection was given.
    bool iova_known(uint64_t addr, uint64_t len) const {
        for (auto& s : iovas_)
            if (addr >= s.first && len <= s.second && addr - s.first <= s.second - len) return true;
        return false;
    }

    // RemotePCIDevice._rpc (:374-385): the request and payload in one sendall, the 17-byte reply, then readout_size
    // bytes on success. A failed status is an error, after its message (resp0 bytes) is read.
    bool rpc(uint8_t cmd, uint64_t a0, uint64_t a1, uint64_t a2, uint32_t bar, uint64_t& r0, uint64_t& r1, std::string& err,
             void* readout = nullptr, uint64_t readout_size = 0, const void* payload = nullptr, uint64_t payload_len = 0) {
        if (!usable(err)) return false;
        uint8_t hdr[33];
        pack(hdr, cmd, bar, a0, a1, a2);
        struct iovec iov[2] = {{hdr, 33}, {const_cast<void*>(payload), (size_t)payload_len}};
        if (!send_iov(iov, payload_len ? 2 : 1)) { err = "TinyGPU.app connection lost while sending"; return false; }
        uint8_t resp[17];
        if (!recv_all(resp, 17)) { err = "Connection closed"; return false; }
        if (!reply(resp, r0, r1, err)) return false;
        if (readout_size && !recv_all(readout, readout_size)) { err = "Connection closed"; return false; }
        return true;
    }

    // _rpc(..., has_fd=True) (:376-379): the reply and one fd by SCM_RIGHTS. Unlike tinygrad, the status comes first: a
    // failed MAP_SYSMEM_FD carries no fd.
    bool rpc_fd(uint8_t cmd, uint64_t a0, uint64_t a1, uint64_t& r0, uint64_t& r1, int& fd, std::string& err) {
        fd = -1;
        if (!usable(err)) return false;
        uint8_t hdr[33];
        pack(hdr, cmd, 0, a0, a1, 0);
        struct iovec out = {hdr, 33};
        if (!send_iov(&out, 1)) { err = "TinyGPU.app connection lost while sending"; return false; }
        uint8_t resp[17];
        char cbuf[CMSG_SPACE(sizeof(int))];
        struct iovec in = {resp, 17};
        struct msghdr msg{};
        msg.msg_iov = &in;
        msg.msg_iovlen = 1;
        msg.msg_control = cbuf;
        msg.msg_controllen = sizeof(cbuf);
        ssize_t r;
        do r = recvmsg(sock_, &msg, 0); while (r < 0 && errno == EINTR);
        if (r <= 0) { lost_ = true; err = "Connection closed"; return false; }
        for (struct cmsghdr* c = CMSG_FIRSTHDR(&msg); c; c = CMSG_NXTHDR(&msg, c))
            if (c->cmsg_level == SOL_SOCKET && c->cmsg_type == SCM_RIGHTS)
                for (size_t k = 0; k < (c->cmsg_len - CMSG_LEN(0)) / sizeof(int); ++k) {
                    int got;
                    memcpy(&got, CMSG_DATA(c) + k * sizeof(int), sizeof(int));
                    if (fd < 0) fd = got; else ::close(got);
                }
        bool truncated = (msg.msg_flags & MSG_CTRUNC) != 0;
        bool ok = r == 17 || recv_all(resp + r, 17 - (size_t)r);
        if (!ok) err = "Connection closed";
        ok = ok && reply(resp, r0, r1, err);
        if (ok && truncated) { ok = false; err = "MAP_SYSMEM_FD: the fd was cut from the reply (MSG_CTRUNC)"; }
        if (ok && fd < 0) { ok = false; err = "MAP_SYSMEM_FD: a successful reply without an fd"; }
        if (!ok && fd >= 0) { ::close(fd); fd = -1; }
        return ok;
    }

    // Plan decision 25 (tinygrad's ensure_app installs TinyGPU.app; BEAGLE never does): the app and its extension must be
    // the release BEAGLE is tested with. BEAGLE_TINYGPU_APP names another app bundle, for tests. Checked once.
    static std::string check_app() {
        static std::string result = [] {
            const char* o = getenv("BEAGLE_TINYGPU_APP");
            std::string app = o ? o : "/Applications/TinyGPU.app";
            static const struct { const char* rel; const char* sha256; } pins[] = {
                {"/Contents/MacOS/TinyGPU", "3ed8bbd9ec8e14e7cf0047fbe7fb6169242394e3e5e75f5428d3edcb70254409"},
                {"/Contents/Library/SystemExtensions/org.tinygrad.tinygpu.driver2.dext/org.tinygrad.tinygpu.driver2",
                 "236035427b9b182ad5f9eb3c16d4a3e5804f84bb864a933d6c7aa8e9c6f3f198"},
            };
            for (const auto& pin : pins) {
                std::string file = app + pin.rel, sha;
                if (!sha256_file(file, sha) || sha != pin.sha256)
                    return (sha.empty() ? file + " is missing" : file + " is not TinyGPU release c0d024f9's") +
                           ". BEAGLE needs that release and does not install it: unzip "
                           "https://github.com/tinygrad/tinygpu_releases/raw/c0d024f9ff0e1dc8fdf217f255da7101d91e8323/TinyGPU.zip "
                           "(sha256 0c47285e2232643210555cf30ce08289b9e55da261c300e0c82e8448a359a21f) into /Applications, run "
                           "`/Applications/TinyGPU.app/Contents/MacOS/TinyGPU install` and approve the system extension, as "
                           "tinygrad's ensure_app does";
            }
            return std::string();
        }();
        return result;
    }

private:
    struct Bar { bool known = false; uint64_t addr = 0, size = 0; };

    int sock_ = -1, lock_fd_ = -1;
    bool lost_ = false;
    Bar bars_[kMaxBars];
    int sysmem_count_ = 0;
    std::vector<std::pair<uint64_t, uint64_t>> iovas_;

    // The '<BIIQQQ' request (system.py:375); dev_id is 0, as for "usb4"
    static void pack(uint8_t* h, uint8_t cmd, uint32_t bar, uint64_t a0, uint64_t a1, uint64_t a2) {
        uint32_t dev = 0;
        h[0] = cmd;
        memcpy(h + 1, &dev, 4);
        memcpy(h + 5, &bar, 4);
        memcpy(h + 9, &a0, 8);
        memcpy(h + 17, &a1, 8);
        memcpy(h + 25, &a2, 8);
    }

    bool usable(std::string& err) const {
        if (sock_ < 0) { err = "not connected to TinyGPU.app"; return false; }
        if (lost_) { err = "the TinyGPU.app connection was lost; nothing more is sent on it"; return false; }
        return true;
    }

    bool check_mmio(uint32_t bar, uint64_t off, uint64_t len, std::string& err) const {
        char msg[200];
        if (bar >= (uint32_t)kMaxBars || !bars_[bar].known)
            snprintf(msg, sizeof(msg), "MMIO on BAR %u before it was mapped in this session (TinyGPU.app would drop it)", bar);
        else if (len > kMaxMessage)
            snprintf(msg, sizeof(msg), "a %llu-byte MMIO message: over TinyGPU.app's 64 MB buffer", (unsigned long long)len);
        else if (off > bars_[bar].size || len > bars_[bar].size - off)
            snprintf(msg, sizeof(msg), "MMIO at 0x%llx+0x%llx outside BAR %u's 0x%llx bytes (TinyGPU.app would drop it)",
                     (unsigned long long)off, (unsigned long long)len, bar, (unsigned long long)bars_[bar].size);
        else return true;
        err = msg;
        return false;
    }

    // socket.sendall of the iovecs (which it consumes): all of them, or the connection is lost (a frame may have been cut)
    bool send_iov(struct iovec* iov, int n) {
        while (n > 0) {
            if (iov->iov_len == 0) { ++iov; --n; continue; }
            struct msghdr msg{};
            msg.msg_iov = iov;
            msg.msg_iovlen = n < IOV_MAX ? n : IOV_MAX;
            ssize_t r = sendmsg(sock_, &msg, 0);
            if (r < 0 && errno == EINTR) continue;
            if (r <= 0) { lost_ = true; return false; }
            size_t done = (size_t)r;
            while (n > 0 && done >= iov->iov_len) { done -= iov->iov_len; ++iov; --n; }
            if (n > 0 && done > 0) { iov->iov_base = (uint8_t*)iov->iov_base + done; iov->iov_len -= done; }
        }
        return true;
    }

    // RemotePCIDevice._recvall (:368-372)
    bool recv_all(void* buf, uint64_t n) {
        uint8_t* p = (uint8_t*)buf;
        while (n) {
            ssize_t r = ::recv(sock_, p, (size_t)n, 0);
            if (r < 0 && errno == EINTR) continue;
            if (r <= 0) { lost_ = true; return false; }
            p += r;
            n -= (uint64_t)r;
        }
        return true;
    }

    // The '<BQQ' reply (:381-383): on a failed status, the message that follows (resp0 bytes, "unknown error" if none)
    bool reply(const uint8_t* resp, uint64_t& r0, uint64_t& r1, std::string& err) {
        memcpy(&r0, resp + 1, 8);
        memcpy(&r1, resp + 9, 8);
        if (resp[0] == 0) return true;
        if (r0 > (1 << 16)) { lost_ = true; err = "a malformed TinyGPU.app error reply; the stream is out of step"; return false; }
        std::string m((size_t)r0, '\0');
        if (r0 && !recv_all(&m[0], r0)) { err = "Connection closed"; return false; }
        err = "RPC failed: " + (r0 ? m : std::string("unknown error"));
        return false;
    }

    // System.flock_acquire (system.py:142-154), before connecting (plan step C3): a created file is made world-writable
    // with fchmod rather than by clearing the process umask, which is the host's
    bool acquire_lock(const char* name, std::string& err) {
        std::string path = tg_temp_path(name);
        bool exists = access(path.c_str(), F_OK) == 0;   // tinygrad avoids O_CREAT on an existing file
        int fd = exists ? ::open(path.c_str(), O_RDWR | O_CLOEXEC) : ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0666);
        if (fd < 0) { err = "cannot open the lock file " + path + ": " + strerror(errno); return false; }
        if (!exists) fchmod(fd, 0666);
        if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
            ::close(fd);
            err = std::string("Failed to acquire lock file ") + name + " (another process has the eGPU). `sudo lsof " + path +
                  "` may help identify the process holding the lock.";
            return false;
        }
        lock_fd_ = fd;
        return true;
    }

    // Popen([APP_PATH, "server", sock_path], stdout=DEVNULL, stderr=DEVNULL) (system.py:436), which closes every other fd;
    // here also in its own session and with stdin on /dev/null
    static bool spawn_server(const std::string& path, std::string& err) {
        const char* o = getenv("BEAGLE_TINYGPU_APP");
        std::string exe = std::string(o ? o : "/Applications/TinyGPU.app") + "/Contents/MacOS/TinyGPU";
        posix_spawnattr_t attr;
        posix_spawn_file_actions_t fa;
        posix_spawnattr_init(&attr);
        posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETSID | POSIX_SPAWN_CLOEXEC_DEFAULT);
        posix_spawn_file_actions_init(&fa);
        for (int fd = 0; fd < 3; ++fd) posix_spawn_file_actions_addopen(&fa, fd, "/dev/null", O_RDWR, 0);
        const char* argv[] = {exe.c_str(), "server", path.c_str(), nullptr};
        pid_t pid;
        int rc = posix_spawn(&pid, exe.c_str(), &fa, &attr, const_cast<char* const*>(argv), environ);
        posix_spawn_file_actions_destroy(&fa);
        posix_spawnattr_destroy(&attr);
        if (rc != 0) err = "starting TinyGPU.app's server: " + std::string(strerror(rc));
        return rc == 0;
    }

    static bool sha256_file(const std::string& file, std::string& hex) {
        hex.clear();
        FILE* f = fopen(file.c_str(), "rb");
        if (!f) return false;
        CC_SHA256_CTX ctx;
        CC_SHA256_Init(&ctx);
        unsigned char buf[1 << 16], md[CC_SHA256_DIGEST_LENGTH];
        size_t n;
        while ((n = fread(buf, 1, sizeof(buf), f)) > 0) CC_SHA256_Update(&ctx, buf, (CC_LONG)n);
        fclose(f);
        CC_SHA256_Final(md, &ctx);
        char h[3];
        for (unsigned char b : md) { snprintf(h, sizeof(h), "%02x", b); hex += h; }
        return true;
    }
};

// The process's one TinyGPU.app connection (the server serves one at a time; plan step P5 shares it among instances).
// Never destroyed: exit closes the socket after the plugin's atexit teardown, and no static destructor may cut a frame
// another thread is sending.
inline TGTransport& tg_transport() {
    static TGTransport* t = new TGTransport;
    return *t;
}

} // namespace tinygpu_device

#endif // LIBHMSBEAGLE_GPU_TINYGPUTRANSPORT_H
