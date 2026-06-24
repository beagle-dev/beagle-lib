/*
 * GPUInterfaceTinyGPUHybrid.cpp
 *
 * BEAGLE NV hybrid backend: tinygrad boots the GPU (GSP + golden image +
 * user channel via nv_init_helper.py) and C++ handles all hot-path dispatch.
 *
 * Drop-in replacement for GPUInterfaceTinyGPU.cpp when built with -DFW_TINYGPU.
 * Select this file in CMakeLists instead of GPUInterfaceTinyGPU.cpp.
 *
 * Reference: tinygrad/runtime/ops_nv.py and tinygrad/runtime/support/nv/
 *   – Socket protocol: tinygrad/runtime/support/system.py (_bulk_read/_bulk_write)
 *   – QMD bit fields: tinygrad/runtime/autogen/nv_580.py (NVC6C0_QMDV03_00_*)
 *   – GPFIFO dispatch: _submit_to_gpfifo in ops_nv.py
 *   – Channel methods: NVComputeQueue.setup/exec/signal in ops_nv.py
 */

#ifdef FW_TINYGPU

#include <cstdio>
#include "libhmsbeagle/beagle.h"
#include "libhmsbeagle/GPU/GPUImplDefs.h"
#include "libhmsbeagle/GPU/GPUImplHelper.h"
#include "libhmsbeagle/GPU/GPUInterface.h"
#include "libhmsbeagle/GPU/KernelResource.h"

#include <cassert>
#include <cerrno>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>
#include <sys/wait.h>

// ── TinyGPU socket commands (matches tinygrad RemoteCmd enum) ────────────────
enum TGCmd : uint8_t {
    TGC_PROBE=0, TGC_MAP_BAR, TGC_MAP_SYSMEM_FD, TGC_CFG_READ, TGC_CFG_WRITE,
    TGC_RESET, TGC_MMIO_READ, TGC_MMIO_WRITE, TGC_MAP_SYSMEM,
    TGC_SYSMEM_READ, TGC_SYSMEM_WRITE, TGC_RESIZE_BAR, TGC_PING
};

// ── State ────────────────────────────────────────────────────────────────────
struct NVHybridState {
    int      sock;
    uint32_t dev_id;
    uint32_t work_token;
    uint64_t gpfifo_vram;
    uint32_t gpfifo_entries, gpfifo_put;
    uint64_t userd_vram;
    uint32_t gpput_off;
    uint64_t eop_vram, eop_gpu_va, eop_signal_val;
    uint64_t cmdq_vram,  cmdq_gpu_va;  uint32_t cmdq_sz,  cmdq_ptr;
    uint64_t code_vram,  code_gpu_va;  uint32_t code_sz,  code_ptr;
    uint64_t data_vram,  data_gpu_va;  uint32_t data_sz,  data_ptr;
    uint32_t compute_class, dma_class;
    uint8_t  sass_version;
    bool     is_blackwell;
    uint64_t bar1_pa;
    uint32_t gsp_sysmem_handle;  // TinyGPU sysmem handle for GSP cmd_q/stat_q allocation
};

// cbuf0_param_off can be up to ~2KB on Blackwell (large driver prefix).
// cbuf0_pfx and dispatch cbuf0 must be at least that large.
static constexpr uint32_t CBUF0_MAX = 8192;

struct NVKernelEntry {
    std::string name;
    uint64_t kernargs_vram;    // VRAM offset of [cbuf0 (512B) | QMD (256B)]
    uint64_t kernargs_gpu_va;
    uint8_t  qmd_tmpl[256];   // QMD v3 template (static fields set at GetFunction)
    uint8_t  cbuf0_pfx[CBUF0_MAX]; // driver prefix — sized for Blackwell's large cbuf0
    uint32_t cbuf0_param_off; // byte offset where kernel args start in cbuf0
    uint32_t cbuf0_total;     // total cbuf0 bytes written to VRAM at dispatch
    Dim3Int  block;           // fixed block dims (set at GetFunction)
    bool     is_v5;
};

// ── Socket helpers ────────────────────────────────────────────────────────────

static void send_all(int fd, const void* buf, size_t n) {
    const uint8_t* p = (const uint8_t*)buf;
    while (n) { ssize_t r = ::send(fd, p, n, 0); if (r <= 0) return; p += r; n -= (size_t)r; }
}
static void recv_all(int fd, void* buf, size_t n) {
    uint8_t* p = (uint8_t*)buf;
    while (n) { ssize_t r = ::recv(fd, p, n, MSG_WAITALL); if (r <= 0) return; p += r; n -= (size_t)r; }
}

// Build the 33-byte RPC request header (tinygrad '<BIIQQQ' format).
static void pack_hdr(uint8_t* h, uint8_t cmd, uint32_t dev, uint32_t bar,
                     uint64_t a0, uint64_t a1, uint64_t a2) {
    h[0] = cmd;
    memcpy(h+1,  &dev, 4);  memcpy(h+5, &bar, 4);
    memcpy(h+9,  &a0,  8);  memcpy(h+17, &a1, 8);  memcpy(h+25, &a2, 8);
}

// Fire-and-forget MMIO write (tinygrad _bulk_write).
// Header: {cmd, dev_id, bar, offset, len, 0} then payload bytes.
static void tg_bulk_write(int sock, uint32_t dev, uint32_t bar,
                          uint64_t off, const void* data, uint64_t len) {
    uint8_t h[33]; pack_hdr(h, TGC_MMIO_WRITE, dev, bar, off, len, 0);
    send_all(sock, h, 33);
    send_all(sock, data, (size_t)len);
}

// Read with response (tinygrad _bulk_read).
// Header: {cmd, dev_id, bar, offset, size, 0}, recv 17-byte resp + size bytes.
static void tg_bulk_read(int sock, uint32_t dev, uint32_t bar,
                         uint64_t off, void* data, uint64_t len) {
    uint8_t h[33]; pack_hdr(h, TGC_MMIO_READ, dev, bar, off, len, 0);
    send_all(sock, h, 33);
    uint8_t resp[17]; recv_all(sock, resp, 17); // status + resp0 + resp1
    recv_all(sock, data, (size_t)len);
}

// BAR0 NV MMIO register write / read (fire-and-forget write, blocking read).
static void nv_wr32(int s, uint32_t d, uint32_t addr, uint32_t val) {
    tg_bulk_write(s, d, 0, addr, &val, 4);
}
static uint32_t nv_rd32(int s, uint32_t d, uint32_t addr) {
    uint32_t v = 0; tg_bulk_read(s, d, 0, addr, &v, 4); return v;
}

// BAR1 VRAM write / read.
static void nv_vram_wr(int s, uint32_t d, uint64_t off, const void* buf, size_t sz) {
    tg_bulk_write(s, d, 1, off, buf, sz);
}
static void nv_vram_rd(int s, uint32_t d, uint64_t off, void* buf, size_t sz) {
    tg_bulk_read(s, d, 1, off, buf, sz);
}

// ── File-scope globals (used by both helper functions and GPUInterface methods) ─
static std::map<std::string, NVKernelEntry*> g_kernels;
static NVHybridState* g_state = nullptr;
static int            g_tgSock = -1;
static uint32_t       g_tgDevId = 0;
static char           g_handoff[256] = {};

// System memory (host RAM, managed by TinyGPU MAP_SYSMEM) write / read.
static void nv_sysmem_wr(int s, uint32_t d, uint32_t handle, uint64_t off, const void* buf, size_t sz) {
    uint8_t h[33]; pack_hdr(h, TGC_SYSMEM_WRITE, d, handle, off, sz, 0);
    send_all(s, h, 33);
    send_all(s, buf, sz);
}
static void nv_sysmem_rd(int s, uint32_t d, uint32_t handle, uint64_t off, void* buf, size_t sz) {
    uint8_t h[33]; pack_hdr(h, TGC_SYSMEM_READ, d, handle, off, sz, 0);
    send_all(s, h, 33);
    uint8_t resp[17]; recv_all(s, resp, 17);
    recv_all(s, buf, sz);
}

// ── GSP RPC: rpc_unloading_guest_driver ──────────────────────────────────────
// Sends NV_VGPU_MSG_FUNCTION_UNLOADING_GUEST_DRIVER (fn=47) to the GSP firmware
// via the shared cmd_q, then waits for a response on stat_q.  This tells GSP to
// cleanly shut down the GPU, preventing TinyGPU.app from needing to issue a PCIe
// FLR on socket disconnect (which causes a macOS kernel panic).
//
// Queue layout constants match tinygrad ip.py init_rm_args(queue_size=0x40000):
//   pt_size = round_up(129*8, 0x1000) = 0x1000
//   cmd_q at sysmem[pt_size]          = sysmem[0x1000]
//   stat_q at sysmem[pt_size+queue_size] = sysmem[0x41000]
static constexpr uint64_t GSP_CMDQ_OFF   = 0x1000;
static constexpr uint64_t GSP_STATQ_OFF  = 0x41000;
static constexpr uint32_t GSP_MSG_SIZE   = 0x1000;
static constexpr uint32_t GSP_MSG_COUNT  = 63;    // (0x40000 - 0x1000) / 0x1000
static constexpr uint32_t GSP_ENTRY_OFF  = 0x1000;
// msgqTxHeader.writePtr byte offset within the queue header
static constexpr uint32_t GSP_HDR_WPTR_OFF = 16;
// BAR0 address of NV_PGSP_QUEUE_HEAD[0] doorbell register
static constexpr uint32_t NV_PGSP_QUEUE_HEAD_0 = 0x110c00;
// RPC framing constants
static constexpr uint32_t GSP_MSG_SIGNATURE   = 0x43505256u;  // NV_VGPU_MSG_SIGNATURE_VALID
static constexpr uint32_t GSP_RESULT_PENDING  = 0xFFFFFFFFu;  // NV_VGPU_MSG_RESULT_RPC_PENDING
static constexpr uint32_t GSP_FN_UNLOAD       = 47u;          // NV_VGPU_MSG_FUNCTION_UNLOADING_GUEST_DRIVER

// XOR checksum over 8-byte chunks, then hi32 ^ lo32.
static uint32_t gsp_checksum(const uint8_t* data, size_t len) {
    uint64_t acc = 0;
    for (size_t i = 0; i < len; i += 8) {
        uint64_t chunk = 0;
        size_t n = (i + 8 <= len) ? 8 : len - i;
        memcpy(&chunk, data + i, n);
        acc ^= chunk;
    }
    return (uint32_t)(acc >> 32) ^ (uint32_t)(acc & 0xFFFFFFFFu);
}

// Returns true if GSP responded to the unload RPC before timeout.
static bool gsp_unloading_guest_driver(NVHybridState* g) {
    fprintf(stderr, "TinyGPU: sending rpc_unloading_guest_driver (fn=%u) ...\n",
            GSP_FN_UNLOAD);
    fflush(stderr);

    int      s      = g->sock;
    uint32_t d      = g->dev_id;
    uint32_t handle = g->gsp_sysmem_handle;

    // Set 3-second receive timeout so we don't hang if TinyGPU is unresponsive.
    {
        struct timeval tv = {3, 0};
        setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    }

    // Read current stat_q writePtr BEFORE sending RPC (baseline for polling).
    uint32_t stat_wptr_base = 0;
    nv_sysmem_rd(s, d, handle, GSP_STATQ_OFF + GSP_HDR_WPTR_OFF, &stat_wptr_base, 4);

    // Read current cmd_q writePtr (where we must write our message).
    uint32_t cmd_wptr = 0;
    nv_sysmem_rd(s, d, handle, GSP_CMDQ_OFF + GSP_HDR_WPTR_OFF, &cmd_wptr, 4);

    // ── Build rpc_unloading_guest_driver_v payload (8 bytes) ─────────────────
    //   bInPMTransition=0 (u8), bGc6Entering=0 (u8), pad[2], newLevel=0x40 (u32)
    uint8_t payload[8] = {};
    uint32_t newLevel = 0x40;  // __GPU_STATE_FLAGS_FAST_UNLOAD = 1<<6
    memcpy(payload + 4, &newLevel, 4);

    // ── Build rpc_message_header_v (32 bytes) ────────────────────────────────
    uint8_t rpc_hdr[32] = {};
    uint32_t hv  = 0x03000000u;                   // header_version
    uint32_t sig = GSP_MSG_SIGNATURE;
    uint32_t len = 32u + 8u;                       // sizeof(header) + sizeof(payload)
    uint32_t fn  = GSP_FN_UNLOAD;
    uint32_t res = GSP_RESULT_PENDING;
    uint32_t seq = 1u;
    memcpy(rpc_hdr +  0, &hv,  4);
    memcpy(rpc_hdr +  4, &sig, 4);
    memcpy(rpc_hdr +  8, &len, 4);
    memcpy(rpc_hdr + 12, &fn,  4);
    memcpy(rpc_hdr + 16, &res, 4);
    memcpy(rpc_hdr + 20, &res, 4);  // rpc_result_private
    memcpy(rpc_hdr + 24, &seq, 4);
    // bytes 28-31 (u field) = 0

    // ── Build GSP_MSG_QUEUE_ELEMENT (48 bytes) ────────────────────────────────
    //   elemCount = ceil((48 + 32 + 8) / 4096) = 1
    uint8_t qele[48] = {};
    uint32_t elem_count = 1u;
    memcpy(qele + 36, &seq,        4);  // seqNum
    memcpy(qele + 40, &elem_count, 4);  // elemCount
    // checkSum (at offset 32) starts at 0; compute over full 96-byte buffer

    // ── Compute checksum over qele(48) + rpc_hdr(32) + payload(8) = 88 bytes ─
    // Padded to 96 bytes (next multiple of 8) with zeros already in place.
    uint8_t chk_buf[96] = {};
    memcpy(chk_buf,      qele,    48);
    memcpy(chk_buf + 48, rpc_hdr, 32);
    memcpy(chk_buf + 80, payload,  8);
    uint32_t csum = gsp_checksum(chk_buf, 96);
    memcpy(qele + 32, &csum, 4);

    // ── Assemble full GSP_MSG_SIZE message (zero-padded to 4096 bytes) ────────
    uint8_t msg[GSP_MSG_SIZE] = {};
    memcpy(msg,      qele,    48);
    memcpy(msg + 48, rpc_hdr, 32);
    memcpy(msg + 80, payload,  8);

    // Write message to cmd_q at entryOff + cmd_wptr * msgSize
    uint64_t msg_off = GSP_CMDQ_OFF + GSP_ENTRY_OFF + (uint64_t)cmd_wptr * GSP_MSG_SIZE;
    nv_sysmem_wr(s, d, handle, msg_off, msg, sizeof(msg));

    // Advance cmd_q writePtr
    uint32_t new_cmd_wptr = (cmd_wptr + elem_count) % GSP_MSG_COUNT;
    nv_sysmem_wr(s, d, handle, GSP_CMDQ_OFF + GSP_HDR_WPTR_OFF, &new_cmd_wptr, 4);

    // Ring GSP doorbell: BAR0 write 0x0 to NV_PGSP_QUEUE_HEAD[0]
    nv_wr32(s, d, NV_PGSP_QUEUE_HEAD_0, 0);

    // Poll stat_q writePtr until GSP advances it past our baseline (up to 5 s)
    uint32_t stat_wptr = stat_wptr_base;
    for (int i = 0; i < 5000; ++i) {
        nv_sysmem_rd(s, d, handle, GSP_STATQ_OFF + GSP_HDR_WPTR_OFF, &stat_wptr, 4);
        if (stat_wptr != stat_wptr_base) break;
        usleep(1000);
    }

    // Restore blocking socket
    {
        struct timeval tv = {0, 0};
        setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    }

    if (stat_wptr == stat_wptr_base) {
        fprintf(stderr, "TinyGPU: gsp_unloading_guest_driver: stat_q timeout — GSP did not respond\n");
        fflush(stderr);
        return false;
    }

    // Read response header to verify function ID
    uint64_t resp_off = GSP_STATQ_OFF + GSP_ENTRY_OFF + (uint64_t)stat_wptr_base * GSP_MSG_SIZE;
    uint8_t resp_hdr[32] = {};
    nv_sysmem_rd(s, d, handle, resp_off + 48, resp_hdr, 32);  // skip 48-byte qele
    uint32_t resp_fn = 0, resp_result = 0;
    memcpy(&resp_fn,     resp_hdr + 12, 4);
    memcpy(&resp_result, resp_hdr + 16, 4);
    fprintf(stderr, "TinyGPU: gsp_unloading_guest_driver OK — fn=%u result=0x%x\n",
            resp_fn, resp_result);
    fflush(stderr);
    return true;
}

// ── Safe exit: send GSP unload RPC, then exit (or fall back to keeper) ────────
// Primary path: gsp_unloading_guest_driver() tells GSP firmware to clean up, so
// TinyGPU.app accepts socket close without issuing a PCIe FLR (kernel panic).
// Fallback: if RPC fails (null state, timeout, socket error), fork a keeper child
// to hold the socket FD open so the parent can exit without triggering FLR.
[[noreturn]] static void safe_exit(int code) {
    fflush(stderr);
    // gsp_sysmem_handle==0 means macOS MAP_SYSMEM_FD path (plain MMIOInterface,
    // no TGC_SYSMEM_READ/WRITE handle) — RPC cannot reach queues from C++.
    if (g_state && g_state->gsp_sysmem_handle != 0 && g_tgSock >= 0) {
        bool ok = gsp_unloading_guest_driver(g_state);
        if (ok) {
            // GSP cleanly unloaded; TinyGPU.app will not FLR on disconnect.
            fprintf(stderr, "TinyGPU: GSP unloaded cleanly — exiting without keeper\n");
            fflush(stderr);
            _exit(code);
        }
    }
    // Fallback: fork a keeper child to hold the socket open.
    if (g_tgSock >= 0) {
        pid_t keeper = fork();
        if (keeper == 0) {
            setsid();
            close(STDIN_FILENO);
            close(STDOUT_FILENO);
            fprintf(stderr,
                    "TinyGPU: keeper pid=%d holding GPU socket open (GSP unload RPC failed).\n"
                    "         To reset safely: unplug Thunderbolt FIRST, then kill %d\n",
                    (int)getpid(), (int)getpid());
            fflush(stderr);
            pause();
            _exit(0);
        }
        if (keeper > 0) {
            fprintf(stderr, "TinyGPU: keeper pid=%d spawned\n", (int)keeper);
            fflush(stderr);
        }
    }
    _exit(code);
}

// Open TinyGPU Unix socket.
static int tg_open_socket() {
    const char* path = getenv("APL_REMOTE_SOCK");
    char default_path[256];
    if (!path) {
        // tinygrad uses tempfile.gettempdir() which on macOS is $TMPDIR, not /tmp.
        const char* tmpdir = getenv("TMPDIR");
        if (!tmpdir || !tmpdir[0]) tmpdir = "/tmp";
        snprintf(default_path, sizeof(default_path), "%stinygpu.sock", tmpdir);
        path = default_path;
    }

    struct sockaddr_un addr{}; addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, path, sizeof(addr.sun_path)-1);

    // Mirror tinygrad's APLRemotePCIDevice.__init__: try connect; on first
    // failure launch "TinyGPU server <path>" in background, then retry.
    static const char* kAppPath = "/Applications/TinyGPU.app/Contents/MacOS/TinyGPU";
    for (int i = 0; i < 100; ++i) {
        int fd = socket(AF_UNIX, SOCK_STREAM, 0);
        if (fd < 0) { perror("TinyGPU socket"); return -1; }
        if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) == 0)
            return fd;
        close(fd);
        if (i == 0) {
            // Spawn TinyGPU in server mode (detached child).
            pid_t pid = fork();
            if (pid == 0) {
                setsid();
                // Redirect stdio to /dev/null so the server doesn't pollute our output.
                int devnull = open("/dev/null", O_RDWR);
                if (devnull >= 0) { dup2(devnull, 0); dup2(devnull, 1); dup2(devnull, 2); close(devnull); }
                const char* argv[] = { kAppPath, "server", path, nullptr };
                execvp(kAppPath, (char* const*)argv);
                _exit(1);
            }
            // parent: fall through to retry loop
        }
        usleep(50000); // 50 ms
    }
    fprintf(stderr, "TinyGPU: could not connect to %s after 5 s\n", path);
    return -1;
}

// ── JSON simple reader (no external deps) ────────────────────────────────────

static uint64_t json_u64(const std::string& js, const char* key) {
    char needle[128]; snprintf(needle, sizeof(needle), "\"%s\":", key);
    auto p = js.find(needle);
    if (p == std::string::npos) return 0;
    p += strlen(needle);
    while (p < js.size() && (js[p]==' '||js[p]=='\n')) ++p;
    return (uint64_t)std::stoull(js.substr(p));
}

static std::string json_str(const std::string& js, const char* key) {
    char needle[128]; snprintf(needle, sizeof(needle), "\"%s\":", key);
    auto p = js.find(needle);
    if (p == std::string::npos) return "";
    p += strlen(needle);
    while (p < js.size() && (js[p]==' '||js[p]=='\n')) ++p;
    if (js[p] != '"') return "";
    ++p; auto e = js.find('"', p);
    return (e != std::string::npos) ? js.substr(p, e-p) : "";
}

static std::string read_file(const char* path) {
    FILE* f = fopen(path, "r"); if (!f) return "";
    fseek(f, 0, SEEK_END); long sz = ftell(f); rewind(f);
    std::string s(sz, '\0'); fread(&s[0], 1, sz, f); fclose(f); return s;
}

static bool write_file(const char* path, const void* buf, size_t sz) {
    FILE* f = fopen(path, "wb"); if (!f) return false;
    fwrite(buf, 1, sz, f); fclose(f); return true;
}

// base64 decode (RFC 4648)
static std::vector<uint8_t> b64decode(const std::string& s) {
    static const int8_t T[256] = {
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,-1,-1,-1,63,
        52,53,54,55,56,57,58,59,60,61,-1,-1,-1,-1,-1,-1,
        -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,
        15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,-1,
        -1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
        41,42,43,44,45,46,47,48,49,50,51,-1,-1,-1,-1,-1,
    };
    std::vector<uint8_t> out;
    uint32_t acc = 0; int bits = 0;
    for (unsigned char c : s) {
        if (c == '=') break;
        int v = (c < 128) ? T[c] : -1;
        if (v < 0) continue;
        acc = (acc << 6) | (uint32_t)v; bits += 6;
        if (bits >= 8) { bits -= 8; out.push_back((uint8_t)(acc >> bits)); }
    }
    return out;
}

// ── QMD v3 bit-field writer (mirrors tinygrad QMD._rw_bits) ─────────────────
// All bit positions from tinygrad/runtime/autogen/nv_580.py NVC6C0_QMDV03_00_*

static void qmd_set(uint8_t* q, int hi, int lo, uint64_t val) {
    int n = hi/8 - lo/8 + 1;
    uint64_t mask = (((uint64_t)1 << (hi - lo + 1)) - 1) << (lo % 8);
    uint64_t cur = 0;
    memcpy(&cur, q + lo/8, n);
    cur = (cur & ~mask) | ((val << (lo % 8)) & mask);
    memcpy(q + lo/8, &cur, n);
}

// smem_cfg: tinygrad → min(sc*1024 for sc in [32,64,100] if sc*1024>=sz) // 4096 + 1
static uint32_t smem_cfg(uint32_t sz) {
    uint32_t sc = (sz > 65536) ? 102400u : (sz > 32768) ? 65536u : 32768u;
    return sc / 4096 + 1;
}

// Build QMD v3 (0x40 dwords = 256 bytes) from tinygrad reference NVProgram.__init__.
// Static fields only; caller must patch grid dims, cbuf0 addr, EOP payload at dispatch.
static void build_qmd_v3(uint8_t* q, const NVHybridState* g,
                          uint64_t prog_va, uint32_t code_sz,
                          uint32_t reg_count, uint32_t shmem, uint32_t slm) {
    memset(q, 0, 256);
    uint32_t sc = smem_cfg(shmem);

    // Fields from tinygrad's qmd dict (NVProgram.__init__, non-Blackwell path):
    qmd_set(q,  133,  128, 0x3f);   // QMD_GROUP_ID
    qmd_set(q,  134,  134, 1);      // SM_GLOBAL_CACHING_ENABLE
    qmd_set(q,  186,  186, 1);      // INVALIDATE_TEXTURE_HEADER_CACHE
    qmd_set(q,  187,  187, 1);      // INVALIDATE_TEXTURE_SAMPLER_CACHE
    qmd_set(q,  188,  188, 1);      // INVALIDATE_TEXTURE_DATA_CACHE
    qmd_set(q,  189,  189, 1);      // INVALIDATE_SHADER_DATA_CACHE
    // PROGRAM_PREFETCH_ADDR_LOWER_SHIFTED = prog_addr >> 8 (bits 287:256)
    qmd_set(q,  287,  256, (prog_va >> 8) & 0xFFFFFFFFu);
    // CWD_MEMBAR_TYPE = L1_SYSMEMBAR = 1 (bits 369:368)
    qmd_set(q,  369,  368, 1);
    // API_VISIBLE_CALL_LIMIT = 1/NO_CHECK (bit 378)
    qmd_set(q,  378,  378, 1);
    // SAMPLER_INDEX = VIA_HEADER_INDEX = 1 (bit 382)
    qmd_set(q,  382,  382, 1);
    // SHARED_MEMORY_SIZE (bits 561:544)
    qmd_set(q,  561,  544, shmem);
    // MIN_SM_CONFIG_SHARED_MEM_SIZE (bits 567:562)
    qmd_set(q,  567,  562, sc);
    // MAX_SM_CONFIG_SHARED_MEM_SIZE = 0x1a (bits 574:569)
    qmd_set(q,  574,  569, 0x1a);
    // QMD_MAJOR_VERSION = 3 (bits 583:580)
    qmd_set(q,  583,  580, 3);
    // CONSTANT_BUFFER_VALID_0 = 1 (bit 640)
    qmd_set(q,  640,  640, 1);
    // REGISTER_COUNT_V (bits 656:648)
    qmd_set(q,  656,  648, reg_count);
    // TARGET_SM_CONFIG_SHARED_MEM_SIZE (bits 662:657)
    qmd_set(q,  662,  657, sc);
    // BARRIER_COUNT = 1 (bits 767:763)
    qmd_set(q,  767,  763, 1);
    // SHADER_LOCAL_MEMORY_HIGH_SIZE (bits 1623:1600) — for non-NAK (ptxas) kernels
    qmd_set(q, 1623, 1600, slm);
    // RELEASE0: address, enable, 64-bit payload flag (EOP semaphore from handoff)
    qmd_set(q,  799,  768, (uint32_t)(g->eop_gpu_va & 0xFFFFFFFFu)); // RELEASE0_ADDRESS_LOWER
    qmd_set(q,  807,  800, (uint8_t)(g->eop_gpu_va >> 32));          // RELEASE0_ADDRESS_UPPER
    qmd_set(q,  823,  823, 1);                                        // RELEASE0_ENABLE
    qmd_set(q,  829,  829, 1);                                        // RELEASE0_PAYLOAD64B
    // CONSTANT_BUFFER_SIZE_SHIFTED4_0 = 0x160 (bits 1087:1075)
    // tinygrad: constbufs[0]=(0,0x160); qmd.write(constant_buffer_size_shifted4_0=sz) stores sz directly.
    qmd_set(q, 1087, 1075, 0x160);
    // CONSTANT_BUFFER_INVALIDATE_0 = 1 (bit 1074)
    qmd_set(q, 1074, 1074, 1);
    // PROGRAM_ADDRESS_LOWER / UPPER (bits 1567:1536 and 1584:1568)
    qmd_set(q, 1567, 1536, (uint32_t)(prog_va & 0xFFFFFFFFu));
    qmd_set(q, 1584, 1568, (uint32_t)(prog_va >> 32));
    // PROGRAM_PREFETCH_ADDR_UPPER_SHIFTED = prog_addr >> 40 (bits 1640:1632)
    qmd_set(q, 1640, 1632, prog_va >> 40);
    // PROGRAM_PREFETCH_SIZE = min(code_sz>>8, 0x1ff) (bits 1649:1641)
    qmd_set(q, 1649, 1641, std::min(code_sz >> 8, 0x1ffu));
    // SASS_VERSION (bits 1663:1656)
    qmd_set(q, 1663, 1656, g->sass_version);
}

// Patch grid dims + cbuf0 addr + EOP payload into an already-built QMD v3.
static void patch_qmd_v3(uint8_t* q, uint32_t gx, uint32_t gy, uint32_t gz,
                          uint32_t bx, uint32_t by, uint32_t bz,
                          uint64_t cbuf0_va, uint64_t eop_val) {
    qmd_set(q,  415,  384, gx);                            // CTA_RASTER_WIDTH
    qmd_set(q,  431,  416, gy);                            // CTA_RASTER_HEIGHT
    qmd_set(q,  463,  448, gz);                            // CTA_RASTER_DEPTH
    qmd_set(q,  607,  592, bx);                            // CTA_THREAD_DIMENSION0
    qmd_set(q,  623,  608, by);                            // CTA_THREAD_DIMENSION1
    qmd_set(q,  639,  624, bz);                            // CTA_THREAD_DIMENSION2
    qmd_set(q, 1055, 1024, (uint32_t)(cbuf0_va & 0xFFFFFFFFu));  // CONSTANT_BUFFER_ADDR_LOWER_0
    qmd_set(q, 1072, 1056, (uint32_t)(cbuf0_va >> 32));          // CONSTANT_BUFFER_ADDR_UPPER_0
    qmd_set(q,  863,  832, (uint32_t)(eop_val & 0xFFFFFFFFu));   // RELEASE0_PAYLOAD_LOWER
    qmd_set(q,  895,  864, (uint32_t)(eop_val >> 32));           // RELEASE0_PAYLOAD_UPPER
}

// ── GPFIFO submission ─────────────────────────────────────────────────────────
// Reference: tinygrad NVCommandQueue._submit_to_gpfifo (ops_nv.py).

static uint32_t nvm_hdr(uint32_t sc, uint32_t mthd, uint32_t n) {
    return (2u << 28) | (n << 16) | (sc << 13) | (mthd >> 2);
}

// Submit a cmdq packet (already in g->cmdq_vram at cmdq_off, n dwords) to GPFIFO.
// Advances g->cmdq_ptr and g->gpfifo_put.
static void submit_gpfifo(NVHybridState* g, uint32_t cmdq_off, uint32_t n_dw) {
    uint64_t cmdq_va  = g->cmdq_gpu_va + cmdq_off;
    uint64_t entry    = cmdq_va | ((uint64_t)n_dw << 42) | (1ULL << 41);
    uint32_t new_put  = (g->gpfifo_put + 1) % g->gpfifo_entries;

    nv_vram_wr(g->sock, g->dev_id, g->gpfifo_vram + (uint64_t)g->gpfifo_put * 8, &entry, 8);
    nv_vram_wr(g->sock, g->dev_id, g->userd_vram  + g->gpput_off,               &new_put, 4);
    __sync_synchronize();
    nv_wr32(g->sock, g->dev_id, 0xbb0090, g->work_token);
    g->gpfifo_put = new_put;
    g->cmdq_ptr   = (g->cmdq_ptr + n_dw * 4 + 63) & ~63u; // 64-byte align
    if (g->cmdq_ptr + 256 > g->cmdq_sz) g->cmdq_ptr = 0;   // wrap
}

// ── Channel setup packet ──────────────────────────────────────────────────────
// Mirrors tinygrad NVComputeQueue().setup(compute_class, local_mem_window, shared_mem_window)
// then a SEM_EXECUTE release so C++ can confirm the channel is ready.

static const uint64_t NV_LOCAL_MEM_WIN  = 0x729300000000ULL;
static const uint64_t NV_SHARED_MEM_WIN = 0x729400000000ULL;

static void send_channel_setup(NVHybridState* g) {
    // Method packet mirrors tinygrad NVComputeQueue().setup(...).submit():
    //   nvm(1, SET_OBJECT, compute_class)
    //   nvm(1, SET_SHADER_LOCAL_MEMORY_WINDOW_A,  *data64(local_mem_window))   data64=(hi,lo)
    //   nvm(1, SET_SHADER_SHARED_MEMORY_WINDOW_A, *data64(shared_mem_window))
    // Then a GPFIFO channel SEM_EXECUTE RELEASE so we can poll channel readiness.
    // Note: tinygrad does NOT send NON_STALL_INTERRUPT on the compute channel.
    uint32_t pkt[14];
    uint32_t n = 0;

    // nvm(1, SET_OBJECT=0x0000, compute_class)
    pkt[n++] = nvm_hdr(1, 0x0000, 1);
    pkt[n++] = g->compute_class;

    // nvm(1, SET_SHADER_LOCAL_MEMORY_WINDOW_A=0x07b0, hi32(LMW), lo32(LMW))
    // data64(val) = (val>>32, val&0xFFFFFFFF) — hi first, matching tinygrad
    pkt[n++] = nvm_hdr(1, 0x07b0, 2);
    pkt[n++] = (uint32_t)(NV_LOCAL_MEM_WIN >> 32);
    pkt[n++] = (uint32_t)(NV_LOCAL_MEM_WIN & 0xFFFFFFFFu);

    // nvm(1, SET_SHADER_SHARED_MEMORY_WINDOW_A=0x02a0, hi32(SMW), lo32(SMW))
    pkt[n++] = nvm_hdr(1, 0x02a0, 2);
    pkt[n++] = (uint32_t)(NV_SHARED_MEM_WIN >> 32);
    pkt[n++] = (uint32_t)(NV_SHARED_MEM_WIN & 0xFFFFFFFFu);

    // GPFIFO channel SEM_EXECUTE RELEASE — lets C++ poll that the setup flushed.
    // 5-word packet at NVC56F_SEM_ADDR_LO=0x005C:
    //   [SEM_ADDR_LO, SEM_ADDR_HI, SEM_PAYLOAD_LO, SEM_PAYLOAD_HI, SEM_EXECUTE]
    // flags: OPERATION_RELEASE(1) | RELEASE_WFI_EN(1<<20) | PAYLOAD_SIZE_64BIT(1<<24) | RELEASE_TIMESTAMP_EN(1<<25)
    uint64_t eop_va  = g->eop_gpu_va;
    uint64_t eop_val = ++(g->eop_signal_val);
    pkt[n++] = nvm_hdr(0, 0x005c, 5);
    pkt[n++] = (uint32_t)(eop_va  & 0xFFFFFFFFu);
    pkt[n++] = (uint32_t)(eop_va  >> 32);
    pkt[n++] = (uint32_t)(eop_val & 0xFFFFFFFFu);
    pkt[n++] = (uint32_t)(eop_val >> 32);
    pkt[n++] = 0x03100001u;

    assert(n == 14);
    uint32_t off = g->cmdq_ptr;
    nv_vram_wr(g->sock, g->dev_id, g->cmdq_vram + off, pkt, n * 4);
    submit_gpfifo(g, off, n);
}

// ── resolve_python ────────────────────────────────────────────────────────────
// Return the path of a Python 3.10+ interpreter.
// Uses only access() — no subprocess, no fd side-effects.
// $BEAGLE_PYTHON overrides everything.  Falls back to "python3" for execvp PATH
// search if none of the hardcoded candidates exist.

static std::string resolve_python() {
    const char* p = getenv("BEAGLE_PYTHON");
    if (p && p[0]) return p;

    static const char* kCandidates[] = {
        // Homebrew ARM (Apple Silicon)
        "/opt/homebrew/bin/python3.13",
        "/opt/homebrew/bin/python3.12",
        "/opt/homebrew/bin/python3.11",
        "/opt/homebrew/bin/python3.10",
        // Homebrew Intel / Linux default prefix
        "/usr/local/bin/python3.13",
        "/usr/local/bin/python3.12",
        "/usr/local/bin/python3.11",
        "/usr/local/bin/python3.10",
        // MacPorts
        "/opt/local/bin/python3.13",
        "/opt/local/bin/python3.12",
        "/opt/local/bin/python3.11",
        "/opt/local/bin/python3.10",
        nullptr
    };
    for (int i = 0; kCandidates[i]; ++i)
        if (access(kCandidates[i], X_OK) == 0) return kCandidates[i];

    return "python3";   // let execvp search PATH
}

// ── nvHybridSetup ─────────────────────────────────────────────────────────────
// Spawn nv_init_helper.py, wait for it, then read the handoff JSON.

static NVHybridState* nvHybridSetup(int sock, uint32_t dev_id) {
    char handoff[256];
    snprintf(handoff, sizeof(handoff), "/tmp/beagle_nv_handoff_%d.json", getpid());

    // Locate nv_init_helper.py via BEAGLE_NV_INIT_HELPER, or derive from BEAGLE_NV_SCRIPTS.
    char script[256];
    const char* helper = getenv("BEAGLE_NV_INIT_HELPER");
    if (!helper) {
        snprintf(script, sizeof(script), "%s/nv_init_helper.py",
                 getenv("BEAGLE_NV_SCRIPTS") ?: ".");
        helper = script;
    }

    if (sock < 0) {
        fprintf(stderr, "TinyGPU: nvHybridSetup called with invalid socket (%d)\n", sock);
        return nullptr;
    }

    // Resolve the Python interpreter using access() — no subprocess, no fd side-effects.
    std::string pypath = resolve_python();

    char sock_str[32], dev_str[32];
    snprintf(sock_str, sizeof(sock_str), "%d", sock);
    snprintf(dev_str,  sizeof(dev_str),  "%u", dev_id);

    // Clear O_CLOEXEC so the child inherits the socket FD.
    int flags = fcntl(sock, F_GETFD);
    fcntl(sock, F_SETFD, flags & ~FD_CLOEXEC);

    fprintf(stderr, "TinyGPU: spawning nv_init_helper.py (python=%s, sock=%d)\n",
            pypath.c_str(), sock);

    // Fork + execvp directly to python — no shell in between, so the socket fd
    // is inherited cleanly without any shell internal fd manipulation.
    pid_t pid = fork();
    if (pid < 0) {
        fprintf(stderr, "TinyGPU: fork failed: %s\n", strerror(errno));
        fcntl(sock, F_SETFD, flags);
        return nullptr;
    }
    if (pid == 0) {
        // Redirect stdout to stderr so Python print() is visible.
        dup2(STDERR_FILENO, STDOUT_FILENO);
        char* argv[] = {
            (char*)pypath.c_str(), (char*)helper,
            sock_str, dev_str, (char*)handoff, nullptr
        };
        execvp(pypath.c_str(), argv);
        fprintf(stderr, "TinyGPU: execvp %s failed: %s\n", pypath.c_str(), strerror(errno));
        _exit(1);
    }
    int wstatus = 0;
    waitpid(pid, &wstatus, 0);
    int ret = WIFEXITED(wstatus) ? WEXITSTATUS(wstatus) : -1;

    // Restore O_CLOEXEC.
    fcntl(sock, F_SETFD, flags);

    if (ret != 0) {
        fprintf(stderr, "TinyGPU: nv_init_helper.py failed (exit %d)\n", ret);
        return nullptr;
    }

    std::string js = read_file(handoff);
    unlink(handoff);
    if (js.empty()) { fprintf(stderr, "TinyGPU: handoff JSON empty\n"); return nullptr; }

    NVHybridState* g = new NVHybridState{};
    g->sock            = sock;
    g->dev_id          = dev_id;
    g->work_token      = (uint32_t)json_u64(js, "work_token");
    g->gpfifo_vram     = json_u64(js, "gpfifo_vram");
    g->gpfifo_entries  = (uint32_t)json_u64(js, "gpfifo_entries");
    g->gpfifo_put      = 0;
    g->userd_vram      = json_u64(js, "userd_vram");
    g->gpput_off       = (uint32_t)json_u64(js, "gpput_off");
    g->eop_vram        = json_u64(js, "eop_vram");
    g->eop_gpu_va      = json_u64(js, "eop_gpu_va");
    g->eop_signal_val  = 0;
    g->cmdq_vram       = json_u64(js, "cmdq_vram");
    g->cmdq_gpu_va     = json_u64(js, "cmdq_gpu_va");
    g->cmdq_sz         = (uint32_t)json_u64(js, "cmdq_sz");
    g->cmdq_ptr        = 0;
    g->code_vram       = json_u64(js, "code_vram");
    g->code_gpu_va     = json_u64(js, "code_gpu_va");
    g->code_sz         = (uint32_t)json_u64(js, "code_sz");
    g->code_ptr        = 0;
    g->data_vram       = json_u64(js, "data_vram");
    g->data_gpu_va     = json_u64(js, "data_gpu_va");
    g->data_sz         = (uint32_t)json_u64(js, "data_sz");
    g->data_ptr        = 0;
    g->compute_class   = (uint32_t)json_u64(js, "compute_class");
    g->dma_class       = (uint32_t)json_u64(js, "dma_class");
    g->sass_version    = (uint8_t)json_u64(js, "sass_version");
    g->bar1_pa            = json_u64(js, "mm_vram_pa_base");
    g->is_blackwell       = (g->compute_class >= 0x0000cec0u);
    g->gsp_sysmem_handle  = (uint32_t)json_u64(js, "gsp_sysmem_handle");

    // Send channel setup packet and wait for EOP.
    send_channel_setup(g);

    // Poll EOP (SynchronizeHost logic for setup).
    uint64_t sig = 0;
    for (int i = 0; i < 600000; ++i) {
        nv_vram_rd(g->sock, g->dev_id, g->eop_vram, &sig, 8);
        if (sig >= g->eop_signal_val) break;
        usleep(1000);
    }
    if (sig < g->eop_signal_val) {
        fprintf(stderr, "TinyGPU: channel setup timeout (eop=0x%llx expected=0x%llx)\n",
                (unsigned long long)sig, (unsigned long long)g->eop_signal_val);
    } else {
        fprintf(stderr, "TinyGPU: channel ready (eop=0x%llx)\n", (unsigned long long)sig);
    }
    return g;
}

// ── cbuf0 driver prefix ───────────────────────────────────────────────────────
// tinygrad: cbuf_0[6:12] = [*data64_le(shared_mem_window), *data64_le(local_mem_window),
//                            *data64_le(0xfffdc0)]  (v3 kernels, non-NAK)

static void make_cbuf0_prefix(uint8_t* buf, uint32_t n_u32s) {
    uint32_t byte_sz = n_u32s * 4;
    if (byte_sz > CBUF0_MAX) {
        fprintf(stderr, "TinyGPU: cbuf0 prefix too large (%u bytes, max %u) — exiting\n",
                byte_sz, CBUF0_MAX);
        safe_exit(1);
    }
    memset(buf, 0, byte_sz);
    uint32_t* b = (uint32_t*)buf;
    b[6]  = (uint32_t)(NV_SHARED_MEM_WIN & 0xFFFFFFFFu);
    b[7]  = (uint32_t)(NV_SHARED_MEM_WIN >> 32);
    b[8]  = (uint32_t)(NV_LOCAL_MEM_WIN  & 0xFFFFFFFFu);
    b[9]  = (uint32_t)(NV_LOCAL_MEM_WIN  >> 32);
    b[10] = 0x00fffdc0u;
    b[11] = 0;
}

// ── Helper: run compile helper ────────────────────────────────────────────────

static bool run_compile_helper(const char* ptx_path, const char* kname,
                                const char* handoff_json, const char* result_json) {
    char script[256];
    const char* helper = getenv("BEAGLE_NV_COMPILE_HELPER");
    if (!helper) {
        snprintf(script, sizeof(script), "%s/nv_compile_helper.py",
                 getenv("BEAGLE_NV_SCRIPTS") ?: ".");
        helper = script;
    }
    std::string pypath = resolve_python();
    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
             "\"%s\" \"%s\" \"%s\" \"%s\" \"%s\" \"%s\" >&2",
             pypath.c_str(), helper, ptx_path, kname, handoff_json, result_json);
    return system(cmd) == 0;
}

// ── Precompile all kernels in one ptxas invocation ────────────────────────────
// Calls nv_compile_helper.py with kernel_name="_all", which compiles the PTX
// once and emits JSONL: line 0 = shared code image, lines 1..N = per-kernel
// metadata.  Uploads the code image once, then builds NVKernelEntry for every
// kernel and populates g_kernels.  After this, GetFunction() is a pure lookup.
static void precompile_all_kernels(const char* kernel_code) {
    if (!g_state || !kernel_code || !kernel_code[0]) return;

    char ptx_path[256], result_path[256];
    snprintf(ptx_path,    sizeof(ptx_path),    "/tmp/beagle_nv_all_%d.ptx",  getpid());
    snprintf(result_path, sizeof(result_path), "/tmp/beagle_nv_all_%d.jsonl", getpid());

    write_file(ptx_path, kernel_code, strlen(kernel_code));

    fprintf(stderr, "TinyGPU: precompile_all_kernels — compiling all kernels (ptxas × 1)…\n");
    fflush(stderr);

    if (!run_compile_helper(ptx_path, "_all", g_handoff, result_path)) {
        fprintf(stderr, "TinyGPU: precompile_all_kernels failed — exiting\n");
        unlink(ptx_path);
        safe_exit(1);
    }
    unlink(ptx_path);

    std::string res = read_file(result_path);
    unlink(result_path);
    if (res.empty()) {
        fprintf(stderr, "TinyGPU: precompile_all_kernels: empty JSONL output — exiting\n");
        safe_exit(1);
    }

    // Split into lines; line 0 = header with code_b64.
    std::vector<std::string> lines;
    {
        size_t pos = 0;
        while (pos <= res.size()) {
            auto nl = res.find('\n', pos);
            size_t end = (nl == std::string::npos) ? res.size() : nl;
            if (end > pos) lines.push_back(res.substr(pos, end - pos));
            if (nl == std::string::npos) break;
            pos = nl + 1;
        }
    }
    if (lines.empty()) {
        fprintf(stderr, "TinyGPU: precompile_all_kernels: no output lines — exiting\n");
        safe_exit(1);
    }

    // Decode and upload the shared code image once.
    std::vector<uint8_t> code = b64decode(json_str(lines[0], "code_b64"));
    if (code.empty()) {
        fprintf(stderr, "TinyGPU: precompile_all_kernels: empty code image — exiting\n");
        safe_exit(1);
    }

    uint32_t img_ptr = (g_state->code_ptr + 255) & ~255u;
    if (img_ptr + (uint32_t)code.size() > g_state->code_sz) {
        fprintf(stderr, "TinyGPU: code buffer too small for full kernel image (%zu bytes) — exiting\n",
                code.size());
        safe_exit(1);
    }
    nv_vram_wr(g_state->sock, g_state->dev_id,
               g_state->code_vram + img_ptr, code.data(), code.size());
    g_state->code_ptr = img_ptr + (uint32_t)((code.size() + 4095) & ~4095u);
    fprintf(stderr, "TinyGPU: precompile_all_kernels — uploaded %zu-byte image to vram+0x%x\n",
            code.size(), img_ptr);
    fflush(stderr);

    // Build a NVKernelEntry for each kernel line.
    int loaded = 0;
    for (size_t i = 1; i < lines.size(); ++i) {
        const std::string& ln = lines[i];
        std::string kname = json_str(ln, "name");
        if (kname.empty()) continue;

        uint32_t code_off  = (uint32_t)json_u64(ln, "code_offset");
        uint32_t code_sz   = (uint32_t)json_u64(ln, "code_size");
        uint32_t reg_count = (uint32_t)json_u64(ln, "reg_count");
        uint32_t shmem     = (uint32_t)json_u64(ln, "shmem_size");
        uint32_t param_off = (uint32_t)json_u64(ln, "cbuf0_param_off");
        uint32_t slm       = (uint32_t)json_u64(ln, "slm_size");
        std::vector<uint8_t> pfx = b64decode(json_str(ln, "cbuf0_prefix_b64"));

        uint64_t prog_va = g_state->code_gpu_va + img_ptr + code_off;

        // Allocate kernargs (cbuf0 + args headroom + QMD) from data pool.
        uint32_t ka_sz   = param_off + 512 + 256;
        uint32_t ka_ptr  = (g_state->data_ptr + 255) & ~255u;
        if (ka_ptr + ka_sz > g_state->data_sz) {
            fprintf(stderr, "TinyGPU: data pool full at kernel %s — exiting\n", kname.c_str());
            safe_exit(1);
        }
        g_state->data_ptr = ka_ptr + ka_sz;

        NVKernelEntry* ke = new NVKernelEntry{};
        ke->name            = kname;
        ke->kernargs_vram   = g_state->data_vram  + ka_ptr;
        ke->kernargs_gpu_va = g_state->data_gpu_va + ka_ptr;
        ke->cbuf0_param_off = param_off;
        ke->cbuf0_total     = ka_sz;
        ke->block           = {1, 1, 1};
        ke->is_v5           = g_state->is_blackwell;

        // Copy cbuf0 prefix from JSON.
        uint32_t pfx_sz = (uint32_t)pfx.size();
        if (pfx_sz > CBUF0_MAX) pfx_sz = CBUF0_MAX;
        memcpy(ke->cbuf0_pfx, pfx.data(), pfx_sz);

        if (!ke->is_v5) {
            build_qmd_v3(ke->qmd_tmpl, g_state, prog_va, code_sz, reg_count, shmem, slm);
        } else {
            memset(ke->qmd_tmpl, 0, 256);
        }

        g_kernels[kname] = ke;
        ++loaded;
    }

    fprintf(stderr, "TinyGPU: precompile_all_kernels — loaded %d kernels\n", loaded);
    fflush(stderr);
}

// ── KernelResource loader (mirrors GPUInterfaceTinyGPU.cpp §LOAD_KERNEL_INTO_RESOURCE) ──
#define LOAD_KERNEL_INTO_RESOURCE(state, prec, id) \
        kernelResource = new KernelResource( \
            state, \
            (char*) KERNELS_STRING_##prec##_##state, \
            PATTERN_BLOCK_SIZE_##prec##_##state, \
            MATRIX_BLOCK_SIZE_##prec##_##state, \
            BLOCK_PEELING_SIZE_##prec##_##state, \
            SLOW_REWEIGHING_##prec##_##state, \
            MULTIPLY_BLOCK_SIZE_##prec, \
            0,0,0,0);

// ═══════════════════════════════════════════════════════════════════════════════
// GPUInterface implementation
// ═══════════════════════════════════════════════════════════════════════════════

namespace tinygpu_device {

GPUInterface::GPUInterface() : numStreams(1), tgpuSock(-1), tgpuDevId(0),
    isNVIDIA(true), vramKernelTop(0), vramDataTop(0),
    amdRingVram(0), amdRingWptr(0), amdRptrAddr(0), amdWptrAddr(0),
    amdEopAddr(0), amdEopSignal(0), amdCompletionHost(nullptr),
    amdCompletionMapped(0), amdCompletionFd(-1),
    nvGspState(nullptr), nvWorkToken(0), nvGpfifoHost(nullptr),
    nvUserdGpPut(nullptr), nvGpfifoEntries(0), nvGpfifoPut(0),
    nvCubinVramBase(0), nvCubinSize(0), amdFbBase(0), amdPartialBoot(false),
    kernelResource(nullptr), resourceMap(nullptr), supportDoublePrecision(false)
{}

GPUInterface::~GPUInterface() {
    bool was_initialized = (g_state != nullptr);
    if (g_state) { delete g_state; g_state = nullptr; }
    for (auto& kv : g_kernels) delete kv.second;
    g_kernels.clear();
    if (g_tgSock >= 0) {
        // Only fork a keeper when the GPU was fully initialized (g_state was set).
        // During BEAGLE's probe/enumerate phase g_state is null — close normally.
        if (was_initialized) {
            pid_t keeper = fork();
            if (keeper == 0) {
                setsid();
                close(STDIN_FILENO); close(STDOUT_FILENO);
                fprintf(stderr,
                        "TinyGPU: keeper pid=%d holding GPU socket open (normal exit).\n"
                        "         To reset safely: unplug Thunderbolt FIRST, then kill %d\n",
                        (int)getpid(), (int)getpid());
                fflush(stderr);
                pause();
                _exit(0);
            }
            if (keeper > 0) {
                fprintf(stderr, "TinyGPU: destructor spawned keeper pid=%d\n", (int)keeper);
                fflush(stderr);
            }
        }
        close(g_tgSock); g_tgSock = -1;
    }
}

int GPUInterface::Initialize() {
    g_tgSock = tg_open_socket();
    if (g_tgSock < 0) return BEAGLE_ERROR_GENERAL;
    // Enumerate devices: just probe device 0 for now.
    // A full probe would use TGC_PROBE; we keep it simple.
    g_tgDevId = 0;
    tgpuSock  = g_tgSock;
    tgpuDevId = g_tgDevId;
    isNVIDIA  = true;
    return BEAGLE_SUCCESS;
}

int GPUInterface::GetDeviceCount() { return (g_tgSock >= 0) ? 1 : 0; }

void GPUInterface::SetDevice(int deviceNumber, int paddedStateCount,
                              int categoryCount, int patternCount,
                              int unpaddedPatternCount, int tipCount, long flags) {
    g_state = nvHybridSetup(g_tgSock, g_tgDevId);
    if (!g_state) { fprintf(stderr, "TinyGPU: nvHybridSetup failed\n"); safe_exit(1); }

    // Save handoff path for compile helper (re-create it from g_state).
    // We write a minimal handoff JSON for the compile helper.
    snprintf(g_handoff, sizeof(g_handoff), "/tmp/beagle_nv_handoff_%d_static.json", getpid());
    FILE* f = fopen(g_handoff, "w");
    if (f) {
        fprintf(f, "{\"chip_name\": \"%s\", \"compute_class\": %u, \"sass_version\": %u}\n",
                g_state->is_blackwell ? "GB202" : "GA102",
                g_state->compute_class, g_state->sass_version);
        fclose(f);
    }

    InitializeKernelResource(paddedStateCount,
                             (flags & BEAGLE_FLAG_PRECISION_DOUBLE) != 0);
    supportDoublePrecision = ((flags & BEAGLE_FLAG_PRECISION_DOUBLE) != 0);

    // Compile all kernels once now so GetFunction() is a pure cache lookup.
    precompile_all_kernels(kernelResource ? kernelResource->kernelCode : nullptr);
}

void GPUInterface::ResizeStreamCount(int n) { numStreams = n; }

void GPUInterface::InitializeKernelResource(int n, bool dp) {
    if (dp) n *= -1;
    switch (n) {
        case   -4: LOAD_KERNEL_INTO_RESOURCE(  4, DP,   4); break;
        case  -16: LOAD_KERNEL_INTO_RESOURCE( 16, DP,  16); break;
        case  -32: LOAD_KERNEL_INTO_RESOURCE( 32, DP,  32); break;
        case  -48: LOAD_KERNEL_INTO_RESOURCE( 48, DP,  48); break;
        case  -64: LOAD_KERNEL_INTO_RESOURCE( 64, DP,  64); break;
        case  -80: LOAD_KERNEL_INTO_RESOURCE( 80, DP,  80); break;
        case -128: LOAD_KERNEL_INTO_RESOURCE(128, DP, 128); break;
        case -192: LOAD_KERNEL_INTO_RESOURCE(192, DP, 192); break;
        case -256: LOAD_KERNEL_INTO_RESOURCE(256, DP, 256); break;
        case    4: LOAD_KERNEL_INTO_RESOURCE(  4, SP,   4); break;
        case   16: LOAD_KERNEL_INTO_RESOURCE( 16, SP,  16); break;
        case   32: LOAD_KERNEL_INTO_RESOURCE( 32, SP,  32); break;
        case   48: LOAD_KERNEL_INTO_RESOURCE( 48, SP,  48); break;
        case   64: LOAD_KERNEL_INTO_RESOURCE( 64, SP,  64); break;
        case   80: LOAD_KERNEL_INTO_RESOURCE( 80, SP,  80); break;
        case  128: LOAD_KERNEL_INTO_RESOURCE(128, SP, 128); break;
        case  192: LOAD_KERNEL_INTO_RESOURCE(192, SP, 192); break;
        case  256: LOAD_KERNEL_INTO_RESOURCE(256, SP, 256); break;
    }
}

// ── Synchronization ───────────────────────────────────────────────────────────

void GPUInterface::SynchronizeHost() {
    if (!g_state) return;
    uint64_t expected = g_state->eop_signal_val;
    uint64_t sig = 0;
    for (int i = 0; i < 6000000; ++i) {
        nv_vram_rd(g_state->sock, g_state->dev_id, g_state->eop_vram, &sig, 8);
        if (sig >= expected) return;
        usleep(10);
    }
    fprintf(stderr, "TinyGPU: SynchronizeHost timeout (sig=0x%llx expected=0x%llx)\n",
            (unsigned long long)sig, (unsigned long long)expected);
}

void GPUInterface::SynchronizeDevice() { SynchronizeHost(); }

void GPUInterface::SynchronizeDeviceWithIndex(int, int) { SynchronizeHost(); }

// ── GetFunction ───────────────────────────────────────────────────────────────
// All kernels are precompiled by precompile_all_kernels() in SetDevice().
// This function is now a pure cache lookup.

GPUFunction GPUInterface::GetFunction(const char* name) {
    if (!g_state) return nullptr;
    auto it = g_kernels.find(name);
    if (it != g_kernels.end()) return it->second;
    fprintf(stderr, "TinyGPU: GetFunction(%s): kernel not found in precompiled cache — exiting\n", name);
    safe_exit(1);

    // Legacy per-kernel compile path (unreachable after precompile_all_kernels,
    // kept for reference).
    char ptx_path[256], result_path[256];
    snprintf(ptx_path,    sizeof(ptx_path),    "/tmp/beagle_nv_%s_%d.ptx",   name, getpid());
    snprintf(result_path, sizeof(result_path), "/tmp/beagle_nv_%s_%d.json", name, getpid());

    // Write PTX to file.
    if (!kernelResource || !kernelResource->kernelCode) {
        fprintf(stderr, "TinyGPU: no kernel source for %s — exiting\n", name);
        safe_exit(1);
    }
    write_file(ptx_path, kernelResource->kernelCode,
               strlen(kernelResource->kernelCode));

    fprintf(stderr, "TinyGPU: GetFunction(%s) compiling…\n", name); fflush(stderr);
    if (!run_compile_helper(ptx_path, name, g_handoff, result_path)) {
        fprintf(stderr, "TinyGPU: compile failed for kernel %s — exiting\n", name);
        unlink(ptx_path);
        safe_exit(1);
    }
    unlink(ptx_path);
    fprintf(stderr, "TinyGPU: GetFunction(%s) compile done, reading result\n", name); fflush(stderr);

    std::string res = read_file(result_path);
    unlink(result_path);
    if (res.empty()) {
        fprintf(stderr, "TinyGPU: empty result JSON for kernel %s — exiting\n", name);
        safe_exit(1);
    }

    // Decode code bytes and upload to code buffer.
    std::string code_b64 = json_str(res, "code_b64");
    std::vector<uint8_t> code = b64decode(code_b64);
    if (code.empty()) {
        fprintf(stderr, "TinyGPU: empty code bytes for kernel %s — exiting\n", name);
        safe_exit(1);
    }

    uint32_t code_off   = (uint32_t)json_u64(res, "code_offset");
    uint32_t code_sz    = (uint32_t)json_u64(res, "code_size");
    uint32_t reg_count  = (uint32_t)json_u64(res, "reg_count");
    uint32_t shmem      = (uint32_t)json_u64(res, "shmem_size");
    uint32_t param_off  = (uint32_t)json_u64(res, "cbuf0_param_off");
    uint32_t slm        = (uint32_t)json_u64(res, "slm_size");

    fprintf(stderr, "TinyGPU: GetFunction(%s) code_bytes=%zu code_off=%u code_sz=%u "
            "reg=%u shmem=%u cbuf0_param_off=%u slm=%u\n",
            name, code.size(), code_off, code_sz, reg_count, shmem, param_off, slm);
    fflush(stderr);

    // Align code_ptr to 256 bytes before upload.
    uint32_t aligned_ptr = (g_state->code_ptr + 255) & ~255u;
    if (aligned_ptr + (uint32_t)code.size() > g_state->code_sz) {
        fprintf(stderr, "TinyGPU: code buffer full for kernel %s (need %zu, have %u) — exiting\n",
                name, code.size(), g_state->code_sz - aligned_ptr);
        safe_exit(1);
    }

    fprintf(stderr, "TinyGPU: GetFunction(%s) nv_vram_wr code @ vram+%u (%zu bytes)…\n",
            name, (unsigned)(g_state->code_vram + aligned_ptr), code.size()); fflush(stderr);
    nv_vram_wr(g_state->sock, g_state->dev_id,
               g_state->code_vram + aligned_ptr, code.data(), code.size());
    fprintf(stderr, "TinyGPU: GetFunction(%s) nv_vram_wr done\n", name); fflush(stderr);

    uint64_t prog_va = g_state->code_gpu_va + aligned_ptr + code_off;
    g_state->code_ptr = aligned_ptr + (uint32_t)((code.size() + 4095) & ~4095u);

    // Allocate kernargs buffer from data pool.
    // Size = cbuf0_param_off + max_args + QMD(256) with margin.
    uint32_t ka_sz = param_off + 512 + 256;  // prefix + args headroom + QMD
    uint32_t kern_ptr = (g_state->data_ptr + 255) & ~255u;
    if (kern_ptr + ka_sz > g_state->data_sz) {
        fprintf(stderr, "TinyGPU: data pool full for kernargs (%s) — exiting\n", name);
        safe_exit(1);
    }
    uint64_t ka_vram = g_state->data_vram + kern_ptr;
    uint64_t ka_va   = g_state->data_gpu_va + kern_ptr;
    g_state->data_ptr = kern_ptr + ka_sz;

    fprintf(stderr, "TinyGPU: GetFunction(%s) building KernelEntry (param_off=%u, is_v5=%d)\n",
            name, param_off, (int)g_state->is_blackwell); fflush(stderr);

    // Build KernelEntry.
    NVKernelEntry* ke = new NVKernelEntry{};
    ke->name            = name;
    ke->kernargs_vram   = ka_vram;
    ke->kernargs_gpu_va = ka_va;
    ke->cbuf0_param_off = param_off;
    ke->cbuf0_total     = ka_sz;
    ke->block           = {1, 1, 1};
    ke->is_v5           = g_state->is_blackwell;

    // Build cbuf0 prefix. n_u32 = param_off/4 so indices 6-11 are in-range.
    uint32_t n_u32 = std::max(param_off / 4u, 12u);
    fprintf(stderr, "TinyGPU: GetFunction(%s) make_cbuf0_prefix n_u32=%u (%u bytes)\n",
            name, n_u32, n_u32 * 4); fflush(stderr);
    make_cbuf0_prefix(ke->cbuf0_pfx, n_u32);
    fprintf(stderr, "TinyGPU: GetFunction(%s) cbuf0 prefix built\n", name); fflush(stderr);

    // Build QMD template (static fields; grid/cbuf0/eop patched at dispatch).
    if (!ke->is_v5) {
        build_qmd_v3(ke->qmd_tmpl, g_state, prog_va, code_sz, reg_count, shmem, slm);
    } else {
        memset(ke->qmd_tmpl, 0, 256);
        fprintf(stderr, "TinyGPU: Blackwell QMD v5 not yet implemented for %s\n", name);
    }

    fprintf(stderr, "TinyGPU: GetFunction(%s) done → ke=%p\n", name, (void*)ke); fflush(stderr);
    g_kernels[name] = ke;
    return ke;
}

// ── LaunchKernelImpl ──────────────────────────────────────────────────────────
// Reference: tinygrad NVComputeQueue.exec() and _submit_to_gpfifo (ops_nv.py).

void GPUInterface::LaunchKernelImpl(GPUFunction fn, Dim3Int block, Dim3Int grid,
                                     int nPtr, int nTotal, GPUPtr* ptrs, unsigned int* ints) {
    if (!g_state || !fn) return;
    NVKernelEntry* ke = (NVKernelEntry*)fn;

    if (ke->is_v5) {
        fprintf(stderr, "TinyGPU: Blackwell QMD v5 not implemented — cannot dispatch %s, exiting\n",
                ke->name.c_str());
        safe_exit(1);
    }

    // Build cbuf0 in host memory.
    // Layout: [0, param_off) = driver prefix; [param_off, ...) = kernel args.
    uint8_t cbuf0[CBUF0_MAX] = {};
    if (ke->cbuf0_param_off > CBUF0_MAX) {
        fprintf(stderr, "TinyGPU: cbuf0_param_off=%u exceeds CBUF0_MAX=%u for %s — exiting\n",
                ke->cbuf0_param_off, CBUF0_MAX, ke->name.c_str());
        safe_exit(1);
    }
    memcpy(cbuf0, ke->cbuf0_pfx, ke->cbuf0_param_off);

    uint8_t* arg_area = cbuf0 + ke->cbuf0_param_off;
    // Pointer args first (uint64, little-endian).
    for (int i = 0; i < nPtr; ++i) {
        uint64_t va = ptrs[i];
        memcpy(arg_area + i * 8, &va, 8);
    }
    // Integer args follow.
    for (int i = nPtr; i < nTotal; ++i) {
        uint32_t v = ints[i - nPtr];
        memcpy(arg_area + nPtr * 8 + (i - nPtr) * 4, &v, 4);
    }

    uint32_t cbuf0_sz = ke->cbuf0_param_off + nPtr * 8 + (nTotal - nPtr) * 4;
    cbuf0_sz = (cbuf0_sz + 3) & ~3u;

    // Build QMD (copy template, patch dynamic fields).
    uint8_t qmd[256];
    memcpy(qmd, ke->qmd_tmpl, 256);
    uint64_t eop_val  = ++(g_state->eop_signal_val);
    uint64_t cbuf0_va = ke->kernargs_gpu_va;   // cbuf0 at start of kernargs
    uint64_t qmd_va   = ke->kernargs_gpu_va + 512; // QMD at byte 512

    patch_qmd_v3(qmd,
                 (uint32_t)grid.x,  (uint32_t)grid.y,  (uint32_t)grid.z,
                 (uint32_t)block.x, (uint32_t)block.y, (uint32_t)block.z,
                 cbuf0_va, eop_val);

    // Write cbuf0 + QMD to kernargs VRAM in one shot.
    uint8_t kernargs[768] = {};
    memcpy(kernargs,       cbuf0, cbuf0_sz);
    memcpy(kernargs + 512, qmd,   256);
    nv_vram_wr(g_state->sock, g_state->dev_id, ke->kernargs_vram, kernargs, 768);

    // Build launch method packet (6 dwords):
    //   nvm(1, INVALIDATE_SHADER_CACHES_NO_WFI=0x1698, 0x1011)
    //   nvm(1, SEND_PCAS_A=0x02b4, qmd_va>>8)
    //   nvm(1, SEND_SIGNALING_PCAS2_B=0x02c0, 9)  // PREFETCH_SCHEDULE
    // Reference: NVComputeQueue.memory_barrier() + exec() in ops_nv.py
    uint32_t pkt[6];
    pkt[0] = nvm_hdr(1, 0x1698, 1);
    pkt[1] = 0x00001011u;                   // instruction|global_data|constant = true
    pkt[2] = nvm_hdr(1, 0x02b4, 1);
    pkt[3] = (uint32_t)(qmd_va >> 8);      // QMD address shifted right 8
    pkt[4] = nvm_hdr(1, 0x02c0, 1);
    pkt[5] = 9;                              // PCAS_ACTION_PREFETCH_SCHEDULE

    uint32_t off = g_state->cmdq_ptr;
    nv_vram_wr(g_state->sock, g_state->dev_id, g_state->cmdq_vram + off, pkt, 24);
    submit_gpfifo(g_state, off, 6);
}

// ── LaunchKernel (variadic) ───────────────────────────────────────────────────

void GPUInterface::LaunchKernel(GPUFunction fn, Dim3Int block, Dim3Int grid,
                                 int nPtr, int nTotal, ...) {
    if (!fn || !g_state) return;
    va_list args; va_start(args, nTotal);
    std::vector<GPUPtr>        ptrs(nPtr);
    std::vector<unsigned int>  ints(nTotal - nPtr);
    for (int i = 0; i < nPtr;          ++i) ptrs[i] = va_arg(args, GPUPtr);
    for (int i = 0; i < nTotal - nPtr; ++i) ints[i] = va_arg(args, unsigned int);
    va_end(args);
    LaunchKernelImpl(fn, block, grid, nPtr, nTotal, ptrs.data(), ints.data());
}

void GPUInterface::LaunchKernelConcurrent(GPUFunction fn, Dim3Int block, Dim3Int grid,
                                           int, int, int nPtr, int nTotal, ...) {
    if (!fn || !g_state) return;
    va_list args; va_start(args, nTotal);
    std::vector<GPUPtr>       ptrs(nPtr);
    std::vector<unsigned int> ints(nTotal - nPtr);
    for (int i = 0; i < nPtr;          ++i) ptrs[i] = va_arg(args, GPUPtr);
    for (int i = 0; i < nTotal - nPtr; ++i) ints[i] = va_arg(args, unsigned int);
    va_end(args);
    LaunchKernelImpl(fn, block, grid, nPtr, nTotal, ptrs.data(), ints.data());
}

// ── Memory ────────────────────────────────────────────────────────────────────

GPUPtr GPUInterface::AllocateMemory(size_t sz) {
    if (!g_state) return 0;
    uint32_t aligned = (g_state->data_ptr + 255) & ~255u;
    if (aligned + (uint32_t)sz > g_state->data_sz) {
        fprintf(stderr, "TinyGPU: data pool exhausted (%zu requested)\n", sz); return 0;
    }
    uint64_t va = g_state->data_gpu_va + aligned;
    g_state->data_ptr = aligned + (uint32_t)((sz + 255) & ~255u);
    return (GPUPtr)va;
}

GPUPtr GPUInterface::AllocateRealMemory(size_t n)  { return AllocateMemory(n * sizeof(double)); }
GPUPtr GPUInterface::AllocateIntMemory(size_t n)   { return AllocateMemory(n * sizeof(int)); }

GPUPtr GPUInterface::CreateSubPointer(GPUPtr base, size_t off, size_t) {
    return base + (GPUPtr)off;
}

size_t GPUInterface::AlignMemOffset(size_t off) { return (off + 255) & ~255u; }

static uint64_t gpu_va_to_vram(NVHybridState* g, uint64_t va) {
    // GPU VA = data_gpu_va + offset → VRAM = data_vram + offset
    if (va >= g->data_gpu_va && va < g->data_gpu_va + g->data_sz)
        return g->data_vram + (va - g->data_gpu_va);
    if (va >= g->code_gpu_va && va < g->code_gpu_va + g->code_sz)
        return g->code_vram + (va - g->code_gpu_va);
    fprintf(stderr, "TinyGPU: gpu_va_to_vram: unknown VA 0x%llx\n", (unsigned long long)va);
    return 0;
}

void GPUInterface::MemcpyHostToDevice(GPUPtr dst, const void* src, size_t sz) {
    if (!g_state || !src || !sz) return;
    uint64_t off = gpu_va_to_vram(g_state, (uint64_t)dst);
    if (off) nv_vram_wr(g_state->sock, g_state->dev_id, off, src, sz);
}

void GPUInterface::MemcpyDeviceToHost(void* dst, const GPUPtr src, size_t sz) {
    if (!g_state || !dst || !sz) return;
    uint64_t off = gpu_va_to_vram(g_state, (uint64_t)src);
    if (off) nv_vram_rd(g_state->sock, g_state->dev_id, off, dst, sz);
}

void GPUInterface::MemcpyDeviceToDevice(GPUPtr dst, GPUPtr src, size_t sz) {
    if (!g_state || !sz) return;
    std::vector<uint8_t> tmp(sz);
    MemcpyDeviceToHost(tmp.data(), src, sz);
    MemcpyHostToDevice(dst, tmp.data(), sz);
}

void GPUInterface::MemsetShort(GPUPtr dst, unsigned short val, size_t count) {
    std::vector<unsigned short> buf(count, val);
    MemcpyHostToDevice(dst, buf.data(), count * sizeof(unsigned short));
}

// ── Host memory (simple malloc wrappers) ─────────────────────────────────────

void* GPUInterface::MallocHost(size_t sz) { return malloc(sz); }
void* GPUInterface::CallocHost(size_t n, size_t sz) { return calloc(n, sz); }
void* GPUInterface::AllocatePinnedHostMemory(size_t sz, bool, bool) { return malloc(sz); }
void  GPUInterface::FreeHostMemory(void* p)        { free(p); }
void  GPUInterface::FreePinnedHostMemory(void* p)  { free(p); }
void  GPUInterface::FreeMemory(GPUPtr) {}

GPUPtr GPUInterface::GetDeviceHostPointer(void* p) { return (GPUPtr)(uintptr_t)p; }

// ── Device info ───────────────────────────────────────────────────────────────

void GPUInterface::GetDeviceName(int, char* name, int len) {
    snprintf(name, len, "TinyGPU-NV-Hybrid");
}
void GPUInterface::GetDeviceDescription(int, char* desc) {
    snprintf(desc, 128, "BEAGLE hybrid NV backend via tinygrad + TinyGPU socket");
}
long GPUInterface::GetDeviceTypeFlag(int) { return BEAGLE_FLAG_PROCESSOR_GPU; }
BeagleDeviceImplementationCodes GPUInterface::GetDeviceImplementationCode(int) {
    return BEAGLE_TINYGPU_DEVICE_NVIDIA_GPU;
}
bool GPUInterface::GetSupportsDoublePrecision(int) { return false; }
size_t GPUInterface::GetAvailableMemory() {
    return g_state ? (size_t)(g_state->data_sz - g_state->data_ptr) : 0;
}

// ── PrintfDeviceVector ────────────────────────────────────────────────────────

template<>
void GPUInterface::PrintfDeviceVector(GPUPtr dPtr, int length, double checkValue, double r) {
    std::vector<double> h(length);
    MemcpyDeviceToHost(h.data(), dPtr, length * sizeof(double));
    printfVector(h.data(), length);
}
template<>
void GPUInterface::PrintfDeviceVector(GPUPtr dPtr, int length, double checkValue, float r) {
    std::vector<float> h(length);
    MemcpyDeviceToHost(h.data(), dPtr, length * sizeof(float));
    printfVector(h.data(), length);
}

void GPUInterface::PrintfDeviceInt(GPUPtr dPtr, int length) {
    std::vector<int> h(length);
    MemcpyDeviceToHost(h.data(), dPtr, length * sizeof(int));
    printfVector(h.data(), length);
}

} // namespace tinygpu_device

#endif // FW_TINYGPU
