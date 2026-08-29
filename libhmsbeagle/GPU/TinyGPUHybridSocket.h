/*
 * TinyGPUHybridSocket.h
 *
 * TinyGPU.app wire protocol primitives shared by every TinyGPUHybrid vendor
 * backend (GPUInterfaceTinyGPUHybrid.cpp for NVIDIA, GPUInterfaceTinyGPUHybridAMD.cpp
 * for AMD). This protocol is plain PCI BAR/config-space/sysmem access —
 * genuinely vendor-agnostic (confirmed: TGC_CFG_READ is what let
 * GPUInterfaceTinyGPUHybrid.cpp::Initialize() identify an AMD device over the
 * exact same socket that had previously only ever talked to NVIDIA hardware,
 * with zero protocol changes needed).
 *
 * Reference: tinygrad/runtime/support/system.py (RemoteCmd enum, RemotePCIDevice._rpc,
 * _bulk_read/_bulk_write) — matches that wire format exactly ('<BIIQQQ' header,
 * 17-byte '<BQQ' response).
 */

#ifndef LIBHMSBEAGLE_GPU_TINYGPUHYBRIDSOCKET_H
#define LIBHMSBEAGLE_GPU_TINYGPUHYBRIDSOCKET_H

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <sys/socket.h>

enum TGCmd : uint8_t {
    TGC_PROBE=0, TGC_MAP_BAR, TGC_MAP_SYSMEM_FD, TGC_CFG_READ, TGC_CFG_WRITE,
    TGC_RESET, TGC_MMIO_READ, TGC_MMIO_WRITE, TGC_MAP_SYSMEM,
    TGC_SYSMEM_READ, TGC_SYSMEM_WRITE, TGC_RESIZE_BAR, TGC_PING
};

static inline void tg_send_all(int fd, const void* buf, size_t n) {
    const uint8_t* p = (const uint8_t*)buf;
    while (n) { ssize_t r = ::send(fd, p, n, 0); if (r <= 0) return; p += r; n -= (size_t)r; }
}
static inline void tg_recv_all(int fd, void* buf, size_t n) {
    uint8_t* p = (uint8_t*)buf;
    while (n) { ssize_t r = ::recv(fd, p, n, MSG_WAITALL); if (r <= 0) return; p += r; n -= (size_t)r; }
}

// Build the 33-byte RPC request header (tinygrad '<BIIQQQ' format).
static inline void tg_pack_hdr(uint8_t* h, uint8_t cmd, uint32_t dev, uint32_t bar,
                                uint64_t a0, uint64_t a1, uint64_t a2) {
    h[0] = cmd;
    memcpy(h+1,  &dev, 4);  memcpy(h+5, &bar, 4);
    memcpy(h+9,  &a0,  8);  memcpy(h+17, &a1, 8);  memcpy(h+25, &a2, 8);
}

// Fire-and-forget MMIO/VRAM write (tinygrad _bulk_write).
// Header: {cmd, dev_id, bar, offset, len, 0} then payload bytes.
static inline void tg_bulk_write(int sock, uint32_t dev, uint32_t bar,
                                  uint64_t off, const void* data, uint64_t len) {
    uint8_t h[33]; tg_pack_hdr(h, TGC_MMIO_WRITE, dev, bar, off, len, 0);
    tg_send_all(sock, h, 33);
    tg_send_all(sock, data, (size_t)len);
}

// Read with response (tinygrad _bulk_read).
// Header: {cmd, dev_id, bar, offset, size, 0}, recv 17-byte resp + size bytes.
static inline void tg_bulk_read(int sock, uint32_t dev, uint32_t bar,
                                 uint64_t off, void* data, uint64_t len) {
    uint8_t h[33]; tg_pack_hdr(h, TGC_MMIO_READ, dev, bar, off, len, 0);
    tg_send_all(sock, h, 33);
    uint8_t resp[17]; tg_recv_all(sock, resp, 17); // status + resp0 + resp1
    tg_recv_all(sock, data, (size_t)len);
}

// PCI config-space read (TGC_CFG_READ). Unlike MMIO reads, the value comes
// back packed directly in the response header's resp1 field (no separate
// bulk payload) -- matches tinygrad's RemotePCIDevice.read_config()/_rpc()
// (tinygrad/runtime/support/system.py), which passes no readout_size for
// this command. `size` must be 1/2/4/8.
static inline uint64_t tg_cfg_read(int sock, uint32_t dev, uint64_t offset, uint64_t size) {
    uint8_t h[33]; tg_pack_hdr(h, TGC_CFG_READ, dev, /*bar=unused*/0, offset, size, 0);
    tg_send_all(sock, h, 33);
    uint8_t resp[17]; tg_recv_all(sock, resp, 17);
    if (resp[0] != 0) {
        // On failure, tinygrad's RemotePCIDevice._rpc() (system.py) reads
        // resp[1] (the u64 at byte offset 1 here) as an error-message length
        // and drains that many bytes from the socket before returning. We
        // don't need the message, but MUST still drain it -- leaving it
        // unread desyncs every subsequent command on this socket (each
        // would read the tail of this error message as if it were its own
        // response), corrupting the whole session, not just this one call.
        uint64_t errlen; memcpy(&errlen, resp + 1, 8);
        fprintf(stderr, "TinyGPU: TGC_CFG_READ(offset=%llu, size=%llu) failed (status=%u)\n",
                (unsigned long long)offset, (unsigned long long)size, resp[0]);
        if (errlen > 0) {
            std::vector<uint8_t> discard(errlen);
            tg_recv_all(sock, discard.data(), errlen);
            fprintf(stderr, "TinyGPU: TGC_CFG_READ error message: %.*s\n",
                    (int)errlen, (const char*)discard.data());
        }
        return 0;
    }
    uint64_t val; memcpy(&val, resp + 1, 8);
    if (size < 8) val &= (1ULL << (size * 8)) - 1;
    return val;
}

#endif // LIBHMSBEAGLE_GPU_TINYGPUHYBRIDSOCKET_H
