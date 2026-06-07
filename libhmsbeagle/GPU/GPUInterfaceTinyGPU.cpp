/*
 *
 * Copyright 2009 Phylogenetic Likelihood Working Group
 *
 * This file is part of BEAGLE.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 *
 * TinyGPU backend: speaks the TinyGPU.app daemon protocol over a Unix domain
 * socket to drive AMD or NVIDIA eGPUs connected via USB4/Thunderbolt PCIe
 * bridges on macOS.  No Python, no CUDA/ROCm driver — only raw BAR0/BAR2
 * MMIO and PCI config-space access.
 *
 * Protocol reference: tinygrad/extra/usbgpu/tbgpu/installer/Shared/server.c
 *                     tinygrad/tinygrad/runtime/support/system.py
 *
 * @author Marc Suchard
 */

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cstdarg>
#include <cerrno>
#include <cmath>

#include <map>
#include <string>
#include <vector>

#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/mman.h>
#include <fcntl.h>

#include "libhmsbeagle/beagle.h"
#include "libhmsbeagle/GPU/GPUImplDefs.h"
#include "libhmsbeagle/GPU/GPUImplHelper.h"
#include "libhmsbeagle/GPU/GPUInterface.h"
#include "libhmsbeagle/GPU/KernelResource.h"

// ── Kernel string selection (reuse CUDA PTX) ─────────────────────────────────
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

namespace tinygpu_device {

// ─────────────────────────────────────────────────────────────────────────────
// Wire protocol structs (packed, little-endian — matches server.c exactly)
// ─────────────────────────────────────────────────────────────────────────────

enum TGCmd : uint8_t {
    CMD_PROBE         = 0,
    CMD_MAP_BAR       = 1,
    CMD_MAP_SYSMEM_FD = 2,
    CMD_CFG_READ      = 3,
    CMD_CFG_WRITE     = 4,
    CMD_RESET         = 5,
    CMD_MMIO_READ     = 6,
    CMD_MMIO_WRITE    = 7,
    CMD_MAP_SYSMEM    = 8,
    CMD_SYSMEM_READ   = 9,
    CMD_SYSMEM_WRITE  = 10,
    CMD_RESIZE_BAR    = 11,
    CMD_PING          = 12,
};

struct __attribute__((packed)) tgpu_req_t {
    uint8_t  cmd;
    uint32_t dev_id;
    uint32_t bar;
    uint64_t arg0, arg1, arg2;
};  // 33 bytes

struct __attribute__((packed)) tgpu_resp_t {
    uint8_t  status;
    uint64_t resp0, resp1;
};  // 17 bytes

// ─────────────────────────────────────────────────────────────────────────────
// Socket helpers
// ─────────────────────────────────────────────────────────────────────────────

static void sock_sendall(int fd, const void* buf, size_t len) {
    const uint8_t* p = (const uint8_t*)buf;
    while (len > 0) {
        ssize_t n = send(fd, p, len, 0);
        if (n <= 0) { perror("TinyGPU send"); exit(1); }
        p += n; len -= (size_t)n;
    }
}

static void sock_recvall(int fd, void* buf, size_t len) {
    uint8_t* p = (uint8_t*)buf;
    while (len > 0) {
        ssize_t n = recv(fd, p, len, 0);
        if (n <= 0) { perror("TinyGPU recv"); exit(1); }
        p += n; len -= (size_t)n;
    }
}

// Send a request and receive the standard 17-byte response.
// Returns true on success (resp.status == 0).
static bool tgpu_rpc(int sock, uint32_t dev_id, TGCmd cmd,
                     uint64_t arg0, uint64_t arg1, uint64_t arg2,
                     uint32_t bar,
                     uint64_t* resp0_out, uint64_t* resp1_out,
                     const void* payload = nullptr, size_t payload_len = 0) {
    tgpu_req_t req;
    req.cmd    = (uint8_t)cmd;
    req.dev_id = dev_id;
    req.bar    = bar;
    req.arg0   = arg0;
    req.arg1   = arg1;
    req.arg2   = arg2;
    sock_sendall(sock, &req, sizeof(req));
    if (payload && payload_len > 0)
        sock_sendall(sock, payload, payload_len);

    tgpu_resp_t resp;
    sock_recvall(sock, &resp, sizeof(resp));
    if (resp0_out) *resp0_out = resp.resp0;
    if (resp1_out) *resp1_out = resp.resp1;
    return (resp.status == 0);
}

// Send a request and receive the 17-byte response along with an fd via
// SCM_RIGHTS ancillary data (used for CMD_MAP_SYSMEM_FD).
static bool tgpu_rpc_fd(int sock, uint32_t dev_id, TGCmd cmd,
                        uint64_t arg0, uint64_t arg1,
                        uint64_t* resp0_out, uint64_t* resp1_out,
                        int* fd_out) {
    tgpu_req_t req;
    req.cmd    = (uint8_t)cmd;
    req.dev_id = dev_id;
    req.bar    = 0;
    req.arg0   = arg0;
    req.arg1   = arg1;
    req.arg2   = 0;
    sock_sendall(sock, &req, sizeof(req));

    char   cmsg_buf[CMSG_SPACE(sizeof(int))];
    struct iovec  iov  = { nullptr, 0 };
    tgpu_resp_t   resp = {};
    iov.iov_base = &resp;
    iov.iov_len  = sizeof(resp);

    struct msghdr msg = {};
    msg.msg_iov        = &iov;
    msg.msg_iovlen     = 1;
    msg.msg_control    = cmsg_buf;
    msg.msg_controllen = sizeof(cmsg_buf);

    ssize_t n = recvmsg(sock, &msg, 0);
    if (n != (ssize_t)sizeof(resp)) { perror("TinyGPU recvmsg"); exit(1); }

    *fd_out = -1;
    if (msg.msg_controllen > 0) {
        struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
        if (cmsg && cmsg->cmsg_level == SOL_SOCKET &&
            cmsg->cmsg_type == SCM_RIGHTS)
            memcpy(fd_out, CMSG_DATA(cmsg), sizeof(int));
    }

    if (resp0_out) *resp0_out = resp.resp0;
    if (resp1_out) *resp1_out = resp.resp1;
    return (resp.status == 0);
}

// Bulk write to a BAR region via CMD_MMIO_WRITE.
// server.c uses volatile 32-bit copies (mmio_copy), so chunk at 4-byte boundary.
static void tgpu_mmio_write(int sock, uint32_t dev_id, uint32_t bar,
                            uint64_t offset, const void* data, size_t size) {
    tgpu_req_t req;
    req.cmd    = CMD_MMIO_WRITE;
    req.dev_id = dev_id;
    req.bar    = bar;
    req.arg0   = offset;
    req.arg1   = (uint64_t)size;
    req.arg2   = 0;
    sock_sendall(sock, &req, sizeof(req));
    sock_sendall(sock, data, size);
    // CMD_MMIO_WRITE has no response (server does not send_response for writes)
}

// Bulk read from a BAR region via CMD_MMIO_READ.
// server.c responds with: 17-byte resp (resp0 = actual size) then the data.
static void tgpu_mmio_read(int sock, uint32_t dev_id, uint32_t bar,
                           uint64_t offset, void* data, size_t size) {
    uint64_t resp0 = 0;
    tgpu_rpc(sock, dev_id, CMD_MMIO_READ, offset, (uint64_t)size, 0, bar,
             &resp0, nullptr);
    sock_recvall(sock, data, (size_t)resp0);
}

// Single 32-bit BAR0 register read/write helpers (used for NV hardware regs).
static uint32_t bar0_read32(int sock, uint32_t dev_id, uint32_t reg) {
    uint32_t val = 0;
    tgpu_mmio_read(sock, dev_id, /*bar=*/0, reg, &val, sizeof(val));
    return val;
}

static void bar0_write32(int sock, uint32_t dev_id, uint32_t reg, uint32_t val) {
    tgpu_mmio_write(sock, dev_id, /*bar=*/0, reg, &val, sizeof(val));
}

// PCI config space read (arg0=offset, arg1=size in bytes).
static uint64_t cfg_read(int sock, uint32_t dev_id, uint64_t offset, uint64_t sz) {
    uint64_t val = 0;
    tgpu_rpc(sock, dev_id, CMD_CFG_READ, offset, sz, 0, 0, &val, nullptr);
    return val;
}

// PCI config space write.
static void cfg_write(int sock, uint32_t dev_id, uint64_t offset, uint64_t sz,
                      uint64_t val) {
    tgpu_rpc(sock, dev_id, CMD_CFG_WRITE, offset, sz, val, 0, nullptr, nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
// NVIDIA BAR0 register constants (Kepler/Maxwell/Pascal/Turing/Ampere)
// Sources: NVIDIA open-gpu-kernel-modules, nouveau Linux driver, tinygrad ops_nv.py
// ─────────────────────────────────────────────────────────────────────────────

static const uint32_t NV_PMC_BOOT_0          = 0x000000;  // GPU chip ID
static const uint32_t NV_PMC_ENABLE          = 0x000200;  // subsystem enable
static const uint32_t NV_PFIFO_INTR_0        = 0x002100;  // PFIFO interrupt status
static const uint32_t NV_PFIFO_INTR_EN_0     = 0x002140;  // PFIFO interrupt enable
static const uint32_t NV_PFIFO_CHANNELS      = 0x002580;  // channel count
static const uint32_t NV_PFIFO_RUNLIST       = 0x002A00;  // runlist submit

// Channel control area base in BAR0 (USERD area, Kepler+)
static const uint32_t NV_USERD_BASE          = 0x800000;

// VRAM layout for the TinyGPU backend
static const uint64_t VRAM_KERNEL_BASE       = 0;                // kernel cubins start here
static const uint64_t VRAM_CHANNEL_BASE      = 64ULL  << 20;    // 64 MB: NV channel objects
static const uint64_t VRAM_DATA_BASE         = 256ULL << 20;    // 256 MB: data allocations

// PCI config offsets
static const uint64_t PCI_CFG_VENDOR_ID      = 0x00;  // 2 bytes
static const uint64_t PCI_CFG_DEVICE_ID      = 0x02;  // 2 bytes
static const uint64_t PCI_CFG_COMMAND        = 0x04;  // 2 bytes (bus-master enable)

// PCI vendor IDs
static const uint16_t PCI_VENDOR_NVIDIA      = 0x10DE;
static const uint16_t PCI_VENDOR_AMD         = 0x1002;

// ─────────────────────────────────────────────────────────────────────────────
// Per-device info gathered at probe time
// ─────────────────────────────────────────────────────────────────────────────

struct DeviceInfo {
    uint32_t dev_id;
    uint16_t pci_vendor;
    uint16_t pci_device;
    uint32_t nv_boot0;      // PMC_BOOT_0 (NV only)
    bool     supports_dp;
};

// ─────────────────────────────────────────────────────────────────────────────
// Per-allocation pinned-memory bookkeeping
// ─────────────────────────────────────────────────────────────────────────────

struct PinnedBuf {
    void*  host_ptr;
    size_t mapped_size;
    int    fd;
};

// ─────────────────────────────────────────────────────────────────────────────
// GPUInterface implementation
// ─────────────────────────────────────────────────────────────────────────────

GPUInterface::GPUInterface()
    : numStreams(1),
      tgpuSock(-1),
      tgpuDevId(0),
      vramKernelTop(VRAM_KERNEL_BASE),
      vramDataTop(VRAM_DATA_BASE),
      kernelResource(nullptr),
      resourceMap(nullptr),
      supportDoublePrecision(true)
{
    memset(tgpuBars, 0, sizeof(tgpuBars));
}

GPUInterface::~GPUInterface() {
    for (auto& pb : tgpuPinned) {
        if (pb.host_ptr && pb.mapped_size)
            munmap(pb.host_ptr, pb.mapped_size);
        if (pb.fd >= 0)
            close(pb.fd);
    }
    for (auto& kv : tgpuKernels)
        delete kv.second;
    if (kernelResource)
        delete kernelResource;
    if (resourceMap)
        delete resourceMap;
    if (tgpuSock >= 0)
        close(tgpuSock);
}

// ── Initialize ───────────────────────────────────────────────────────────────

int GPUInterface::Initialize() {
    resourceMap = new std::map<int,int>;

    const char* sock_path = getenv("APL_REMOTE_SOCK");
    char default_path[256];
    if (!sock_path) {
        const char* tmpdir = getenv("TMPDIR");
        if (!tmpdir) tmpdir = "/tmp";
        snprintf(default_path, sizeof(default_path), "%s/tinygpu.sock", tmpdir);
        sock_path = default_path;
    }

    tgpuSock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (tgpuSock < 0) return 0;

    struct sockaddr_un addr;
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, sock_path, sizeof(addr.sun_path) - 1);
    addr.sun_path[sizeof(addr.sun_path)-1] = '\0';

    // Try to connect; on first failure attempt to start TinyGPU.app.
    bool connected = false;
    for (int i = 0; i < 100 && !connected; ++i) {
        if (connect(tgpuSock, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
            connected = true;
            break;
        }
        if (i == 0) {
            const char* app = "/Applications/TinyGPU.app/Contents/MacOS/TinyGPU";
            pid_t pid = fork();
            if (pid == 0) {
                execlp(app, app, "server", sock_path, nullptr);
                _exit(1);
            }
        }
        usleep(50000);  // 50 ms between retries
    }

    if (!connected) {
        fprintf(stderr, "TinyGPU: could not connect to %s\n", sock_path);
        close(tgpuSock);
        tgpuSock = -1;
        return 0;
    }

    // Expand socket buffers to 64 MB (matches server.c handle_client)
    int bufsize = 64 * 1024 * 1024;
    setsockopt(tgpuSock, SOL_SOCKET, SO_SNDBUF, &bufsize, sizeof(bufsize));
    setsockopt(tgpuSock, SOL_SOCKET, SO_RCVBUF, &bufsize, sizeof(bufsize));

    // Probe: read PCI vendor/device IDs of the connected GPU (device 0).
    uint16_t vendor = (uint16_t)cfg_read(tgpuSock, 0, PCI_CFG_VENDOR_ID, 2);
    uint16_t device = (uint16_t)cfg_read(tgpuSock, 0, PCI_CFG_DEVICE_ID, 2);

    if (vendor != PCI_VENDOR_NVIDIA && vendor != PCI_VENDOR_AMD) {
        fprintf(stderr, "TinyGPU: unrecognized GPU vendor 0x%04x\n", vendor);
        close(tgpuSock);
        tgpuSock = -1;
        return 0;
    }

    // Enable bus-master + memory-space bits in PCI command register.
    uint16_t cmd_reg = (uint16_t)cfg_read(tgpuSock, 0, PCI_CFG_COMMAND, 2);
    cmd_reg |= 0x0006;  // memory space (bit 1) + bus master (bit 2)
    cfg_write(tgpuSock, 0, PCI_CFG_COMMAND, 2, cmd_reg);

    // Cache BAR0 and BAR2 addresses/sizes.
    uint64_t bar_addr, bar_size;
    if (tgpu_rpc(tgpuSock, 0, CMD_MAP_BAR, 0, 0, 0, /*bar=*/0, &bar_addr, &bar_size)) {
        tgpuBars[0].addr = bar_addr;
        tgpuBars[0].size = bar_size;
    }
    if (tgpu_rpc(tgpuSock, 0, CMD_MAP_BAR, 0, 0, 0, /*bar=*/2, &bar_addr, &bar_size)) {
        tgpuBars[2].addr = bar_addr;
        tgpuBars[2].size = bar_size;
    }

    // Populate device info.
    DeviceInfo info;
    info.dev_id     = 0;
    info.pci_vendor = vendor;
    info.pci_device = device;
    info.nv_boot0   = 0;
    info.supports_dp = true;

    if (vendor == PCI_VENDOR_NVIDIA) {
        info.nv_boot0 = bar0_read32(tgpuSock, 0, NV_PMC_BOOT_0);
        // All Kepler+ NV GPUs (the minimum for eGPU) support DP.
        info.supports_dp = true;
    }

    tgpuDevices.push_back(info);
    resourceMap->insert(std::make_pair(0, 0));

    return 1;
}

// ── Device count / info ───────────────────────────────────────────────────────

int GPUInterface::GetDeviceCount() {
    return (int)resourceMap->size();
}

void GPUInterface::GetDeviceName(int deviceNumber, char* deviceName, int nameLength) {
    if (deviceNumber >= (int)tgpuDevices.size()) {
        strncpy(deviceName, "Unknown TinyGPU device", nameLength);
        return;
    }
    const DeviceInfo& info = tgpuDevices[deviceNumber];
    if (info.pci_vendor == PCI_VENDOR_NVIDIA)
        snprintf(deviceName, nameLength, "NVIDIA GPU (PCI %04x:%04x via TinyGPU)",
                 info.pci_vendor, info.pci_device);
    else
        snprintf(deviceName, nameLength, "AMD GPU (PCI %04x:%04x via TinyGPU)",
                 info.pci_vendor, info.pci_device);
}

void GPUInterface::GetDeviceDescription(int deviceNumber, char* deviceDescription) {
    if (deviceNumber >= (int)tgpuDevices.size()) {
        strncpy(deviceDescription, "", 1);
        return;
    }
    const DeviceInfo& info = tgpuDevices[deviceNumber];
    uint64_t vram_mb = tgpuBars[2].size >> 20;
    snprintf(deviceDescription, 256,
             "VRAM aperture (MB): %llu | BAR0 (MB): %llu | USB4/Thunderbolt PCIe bridge",
             (unsigned long long)vram_mb,
             (unsigned long long)(tgpuBars[0].size >> 20));
    (void)info;
}

long GPUInterface::GetDeviceTypeFlag(int /*deviceNumber*/) {
    return BEAGLE_FLAG_PROCESSOR_GPU;
}

BeagleDeviceImplementationCodes GPUInterface::GetDeviceImplementationCode(int deviceNumber) {
    if (deviceNumber < (int)tgpuDevices.size()) {
        if (tgpuDevices[deviceNumber].pci_vendor == PCI_VENDOR_NVIDIA)
            return BEAGLE_TINYGPU_DEVICE_NVIDIA_GPU;
        else
            return BEAGLE_TINYGPU_DEVICE_AMD_GPU;
    }
    return BEAGLE_TINYGPU_DEVICE_NVIDIA_GPU;
}

bool GPUInterface::GetSupportsDoublePrecision(int deviceNumber) {
    if (deviceNumber < (int)tgpuDevices.size())
        return tgpuDevices[deviceNumber].supports_dp;
    return true;
}

// ── SetDevice ─────────────────────────────────────────────────────────────────

void GPUInterface::SetDevice(int deviceNumber, int paddedStateCount, int categoryCount,
                              int paddedPatternCount, int unpaddedPatternCount,
                              int tipCount, long flags) {
    tgpuDevId = (uint32_t)(*resourceMap)[deviceNumber];

    InitializeKernelResource(paddedStateCount, (flags & BEAGLE_FLAG_PRECISION_DOUBLE) != 0);
    if (!kernelResource) {
        fprintf(stderr, "TinyGPU: no kernel for %d states\n", paddedStateCount);
        exit(1);
    }
    kernelResource->categoryCount        = categoryCount;
    kernelResource->patternCount         = paddedPatternCount;
    kernelResource->unpaddedPatternCount = unpaddedPatternCount;
    kernelResource->flags                = flags;

    // Reset VRAM allocators for this device session.
    vramKernelTop = VRAM_KERNEL_BASE;
    vramDataTop   = VRAM_DATA_BASE;

    NVLoadKernels();
}

// ── KernelResource selection (identical logic to CUDA backend) ────────────────

void GPUInterface::InitializeKernelResource(int paddedStateCount, bool doublePrecision) {
    if (doublePrecision) paddedStateCount *= -1;
    switch (paddedStateCount) {
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

// ─────────────────────────────────────────────────────────────────────────────
// Phase 3 — NVIDIA hardware channel and kernel loading
//
// NVIDIA GPU dispatch uses a command FIFO ("pushbuffer") submitted to the
// GPU via BAR0 register writes.  The sequence:
//
//   1. Compile PTX → cubin  (ptxas offline tool or NVRTC)
//   2. Copy cubin to VRAM   (CMD_MMIO_WRITE to BAR2)
//   3. Set up GPFIFO channel + pushbuffer in VRAM
//   4. For each kernel launch: write NVC6C0_QPFIFO compute dispatch packets
//      into the pushbuffer, then ring doorbell via BAR0
//   5. Sync: poll semaphore register in BAR0
//
// TODO: complete hardware-specific register sequences and test on physical GPU.
//       Reference: NVIDIA open-gpu-kernel-modules, tinygrad/runtime/ops_nv.py,
//                  nouveau Linux driver (drivers/gpu/drm/nouveau/).
// ─────────────────────────────────────────────────────────────────────────────

// Compile PTX to cubin via the `ptxas` tool from the CUDA toolkit.
// Returns the cubin binary; caller must free() the buffer.
static char* ptx_to_cubin(const char* ptx, size_t ptx_len, size_t* cubin_len_out) {
    // Detect SM version from NV_PMC_BOOT_0 if possible; default to sm_86 (Ampere).
    // TODO: query GPU arch from PMC_BOOT_0 and map to sm_XX.
    const char* sm = getenv("BEAGLE_TINYGPU_SM");
    if (!sm) sm = "sm_86";

    char ptx_path[256], cubin_path[256];
    snprintf(ptx_path,   sizeof(ptx_path),   "/tmp/beagle_tgpu_%d.ptx",   (int)getpid());
    snprintf(cubin_path, sizeof(cubin_path),  "/tmp/beagle_tgpu_%d.cubin", (int)getpid());

    FILE* f = fopen(ptx_path, "wb");
    if (!f) { perror("ptx write"); return nullptr; }
    fwrite(ptx, 1, ptx_len, f);
    fclose(f);

    char cmd[1024];
    // ptxas is the standalone PTX assembler shipped with every CUDA toolkit.
    snprintf(cmd, sizeof(cmd), "ptxas -arch %s -o %s %s 2>/tmp/beagle_ptxas.log",
             sm, cubin_path, ptx_path);
    int rc = system(cmd);
    unlink(ptx_path);

    if (rc != 0) {
        fprintf(stderr, "TinyGPU: ptxas failed (rc=%d); see /tmp/beagle_ptxas.log\n", rc);
        fprintf(stderr, "         Set BEAGLE_TINYGPU_SM=sm_XX for correct GPU arch\n");
        return nullptr;
    }

    FILE* cf = fopen(cubin_path, "rb");
    if (!cf) { perror("cubin read"); return nullptr; }
    fseek(cf, 0, SEEK_END);
    long csz = ftell(cf);
    rewind(cf);

    char* cubin = (char*)malloc((size_t)csz);
    fread(cubin, 1, (size_t)csz, cf);
    fclose(cf);
    unlink(cubin_path);

    if (cubin_len_out) *cubin_len_out = (size_t)csz;
    return cubin;
}

// Copy a cubin into VRAM via BAR2 MMIO and register its kernel entry points.
// The cubin ELF contains one .text section per kernel; entry offset is 0 for
// the first kernel.  Full ELF parsing for multi-kernel cubins is a TODO.
void GPUInterface::NVLoadKernels() {
    const char* ptx    = kernelResource->kernelCode;
    size_t      ptx_sz = strlen(ptx);

    size_t cubin_sz = 0;
    char*  cubin    = ptx_to_cubin(ptx, ptx_sz, &cubin_sz);
    if (!cubin) return;

    // Align allocation to 4 KB.
    size_t aligned_sz = (cubin_sz + 0xfff) & ~0xfffULL;
    uint64_t vram_off = vramKernelTop;
    vramKernelTop += aligned_sz;

    // Write cubin to VRAM via CMD_MMIO_WRITE on BAR2.
    tgpu_mmio_write(tgpuSock, tgpuDevId, /*bar=*/2, vram_off, cubin, cubin_sz);
    free(cubin);

    // TODO: parse the cubin ELF to extract per-kernel entry point offsets and
    //       build the tgpuKernels map accurately.  For now record the base
    //       offset for GetFunction() look-ups.
    nvCubinVramBase = vram_off;

    fprintf(stderr, "TinyGPU: cubin (%zu bytes) loaded at VRAM+0x%llx\n",
            cubin_sz, (unsigned long long)vram_off);

    // Set up NVIDIA GPFIFO channel for compute dispatch.
    // TODO: implement full NV channel allocation using BAR0 PFIFO registers.
    //       Required steps (see NVIDIA open-gpu-kernel-modules/nouveau):
    //         a) Allocate RAMFC channel control block in VRAM
    //         b) Allocate pushbuffer ring in VRAM
    //         c) Configure PFIFO_CHANNELS and PFIFO_RUNLIST
    //         d) Submit channel to PFIFO via NV_PFIFO_RUNLIST_SUBMIT
    nvPushbufVram = VRAM_CHANNEL_BASE;
    nvPushbufSize = 1U << 20;   // 1 MB pushbuffer ring
    nvPbPut       = 0;
    fprintf(stderr, "TinyGPU: pushbuffer at VRAM+0x%llx (NV channel setup TODO)\n",
            (unsigned long long)nvPushbufVram);
}

// ── Streams (minimal — TinyGPU is single-stream for now) ─────────────────────

void GPUInterface::ResizeStreamCount(int /*newStreamCount*/) {
    // Stream concurrency not yet implemented for the TinyGPU backend.
}

// ── Synchronization ───────────────────────────────────────────────────────────

void GPUInterface::SynchronizeHost() {
    // TODO: poll GPU semaphore via BAR0 for true completion signaling.
    // For now, this is a no-op; correctness relies on MMIO writes being ordered.
}

void GPUInterface::SynchronizeDevice() {
    SynchronizeHost();
}

void GPUInterface::SynchronizeDeviceWithIndex(int /*streamRecordIndex*/,
                                               int /*streamWaitIndex*/) {
    SynchronizeHost();
}

// ── Kernel functions ──────────────────────────────────────────────────────────

GPUFunction GPUInterface::GetFunction(const char* functionName) {
    auto it = tgpuKernels.find(functionName);
    if (it != tgpuKernels.end())
        return (GPUFunction)it->second;

    // Not found yet — create a placeholder entry.
    // TODO: parse cubin ELF and record the real per-kernel VRAM offset.
    auto* entry = new NVKernelEntry;
    entry->vram_addr  = nvCubinVramBase; // placeholder; needs ELF parsing
    entry->name       = functionName;
    tgpuKernels[functionName] = entry;
    return (GPUFunction)entry;
}

// ── LaunchKernel ──────────────────────────────────────────────────────────────
//
// NV compute dispatch packet format (NVC6C0 / Pascal+):
//   Method word:  [bits 15:0] = method address >> 2
//                 [bits 28:16] = count (# of data DWORDs that follow)
//   Data words:   inline values
//
// Key methods (from tinygrad/extra/nv_gpu.py, NVIDIA open-gpu-kernel-modules):
//   0x02c4 LAUNCH_GRID_X
//   0x02c8 LAUNCH_GRID_Y
//   0x02cc LAUNCH_GRID_Z
//   0x02d0 LAUNCH_BLOCK_DIM_X
//   0x02d4 LAUNCH_BLOCK_DIM_Y
//   0x02d8 LAUNCH_BLOCK_DIM_Z
//   0x0220 SET_SHADER_LOCAL_MEMORY_A (high bits of kernel .text address)
//   0x0224 SET_SHADER_LOCAL_MEMORY_B (low bits)
//   ...
//
// TODO: build and submit proper compute launch descriptor.
// ─────────────────────────────────────────────────────────────────────────────

static void pb_push(std::vector<uint32_t>& pb, uint32_t method, uint32_t value) {
    pb.push_back((1 << 16) | (method >> 2));  // count=1, method address
    pb.push_back(value);
}

void GPUInterface::LaunchKernel(GPUFunction deviceFunction,
                                 Dim3Int block, Dim3Int grid,
                                 int parameterCountV, int totalParameterCount,
                                 ...) {
    NVKernelEntry* entry = (NVKernelEntry*)deviceFunction;

    // Collect variadic parameters.
    std::vector<GPUPtr>        ptrParams(parameterCountV);
    std::vector<unsigned int>  intParams(totalParameterCount - parameterCountV);

    va_list args;
    va_start(args, totalParameterCount);
    for (int i = 0; i < parameterCountV; ++i)
        ptrParams[i] = va_arg(args, GPUPtr);
    for (int i = 0; i < totalParameterCount - parameterCountV; ++i)
        intParams[i] = va_arg(args, unsigned int);
    va_end(args);

    // TODO: set up the constant-buffer parameter table in VRAM, pointing each
    //       parameter to its corresponding VRAM offset.

    // Build pushbuffer command sequence.
    std::vector<uint32_t> pb;

    // Set kernel code address (high/low 32 bits of VRAM+entry offset).
    uint64_t code_va = entry->vram_addr;
    pb_push(pb, 0x0220, (uint32_t)(code_va >> 32));   // SET_SHADER_LOCAL_MEMORY_A
    pb_push(pb, 0x0224, (uint32_t)(code_va & 0xFFFFFFFFULL)); // SET_SHADER_LOCAL_MEMORY_B

    // Set grid and block dimensions.
    pb_push(pb, 0x02c4, grid.x);
    pb_push(pb, 0x02c8, grid.y);
    pb_push(pb, 0x02cc, grid.z);
    pb_push(pb, 0x02d0, block.x);
    pb_push(pb, 0x02d4, block.y);
    pb_push(pb, 0x02d8, block.z);

    // LAUNCH (NVC6C0_QPFIFO trigger = 0x02c0).
    // TODO: add parameter CB setup before this.
    pb_push(pb, 0x02c0, 0x1);  // trigger compute launch

    // Write pushbuffer to VRAM ring.
    uint32_t pb_bytes = (uint32_t)(pb.size() * sizeof(uint32_t));
    tgpu_mmio_write(tgpuSock, tgpuDevId, /*bar=*/2,
                    nvPushbufVram + nvPbPut, pb.data(), pb_bytes);
    nvPbPut = (nvPbPut + pb_bytes) % nvPushbufSize;

    // TODO: advance GPFIFO PUT pointer via BAR0 doorbell to trigger execution.
    // bar0_write32(tgpuSock, tgpuDevId, NV_USERD_BASE + channel_id * 0x200 + 0x8c,
    //              nvPbPut >> 2);
}

void GPUInterface::LaunchKernelConcurrent(GPUFunction deviceFunction,
                                           Dim3Int block, Dim3Int grid,
                                           int /*streamIndex*/, int /*waitIndex*/,
                                           int parameterCountV, int totalParameterCount,
                                           ...) {
    // Build the same va_list and forward to LaunchKernel.
    NVKernelEntry* entry = (NVKernelEntry*)deviceFunction;

    std::vector<GPUPtr>       ptrParams(parameterCountV);
    std::vector<unsigned int> intParams(totalParameterCount - parameterCountV);

    va_list args;
    va_start(args, totalParameterCount);
    for (int i = 0; i < parameterCountV; ++i)
        ptrParams[i] = va_arg(args, GPUPtr);
    for (int i = 0; i < totalParameterCount - parameterCountV; ++i)
        intParams[i] = va_arg(args, unsigned int);
    va_end(args);

    // Re-dispatch via LaunchKernel using the collected params.
    // (Streams are not differentiated in the current skeleton.)
    LaunchKernel(deviceFunction, block, grid, parameterCountV, totalParameterCount,
                 ptrParams.data(), intParams.data());
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 2 — Memory management
// ─────────────────────────────────────────────────────────────────────────────

// VRAM bump allocator.  GPUPtr == byte offset from start of VRAM (= BAR2 offset).
GPUPtr GPUInterface::AllocateMemory(size_t memSize) {
    memSize = (memSize + 255ULL) & ~255ULL;   // 256-byte alignment
    GPUPtr ptr = (GPUPtr)vramDataTop;
    vramDataTop += memSize;
    return ptr;
}

GPUPtr GPUInterface::AllocateRealMemory(size_t length) {
    return AllocateMemory(SIZE_REAL * length);
}

GPUPtr GPUInterface::AllocateIntMemory(size_t length) {
    return AllocateMemory(SIZE_INT * length);
}

GPUPtr GPUInterface::CreateSubPointer(GPUPtr dPtr, size_t offset, size_t /*size*/) {
    return dPtr + (GPUPtr)offset;
}

size_t GPUInterface::AlignMemOffset(size_t offset) {
    return offset;
}

void GPUInterface::FreeMemory(GPUPtr /*dPtr*/) {
    // Bump allocator: no individual free.  Memory is reclaimed at SetDevice time.
}

size_t GPUInterface::GetAvailableMemory() {
    if (tgpuBars[2].size == 0) return 0;
    uint64_t end = tgpuBars[2].size;
    if (vramDataTop >= end) return 0;
    return (size_t)(end - vramDataTop);
}

// ── Host (pinned) memory ──────────────────────────────────────────────────────

void* GPUInterface::MallocHost(size_t memSize) {
    return malloc(memSize);
}

void* GPUInterface::CallocHost(size_t size, size_t length) {
    return calloc(size, length);
}

// AllocatePinnedHostMemory uses CMD_MAP_SYSMEM_FD to get DMA-able memory
// with physical addresses, suitable for zero-copy GPU access.
void* GPUInterface::AllocatePinnedHostMemory(size_t memSize,
                                              bool /*writeCombined*/,
                                              bool /*mapped*/) {
    uint64_t mapped_size = 0, handle = 0;
    int      shm_fd      = -1;

    if (!tgpu_rpc_fd(tgpuSock, tgpuDevId, CMD_MAP_SYSMEM_FD,
                     (uint64_t)memSize, 0,
                     &mapped_size, &handle, &shm_fd)) {
        fprintf(stderr, "TinyGPU: MAP_SYSMEM_FD failed\n");
        return nullptr;
    }

    void* host_ptr = mmap(nullptr, (size_t)mapped_size,
                          PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (host_ptr == MAP_FAILED) {
        perror("TinyGPU mmap");
        close(shm_fd);
        return nullptr;
    }

    PinnedBuf pb;
    pb.host_ptr    = host_ptr;
    pb.mapped_size = (size_t)mapped_size;
    pb.fd          = shm_fd;
    tgpuPinned.push_back(pb);

    return host_ptr;
}

void GPUInterface::FreeHostMemory(void* hPtr) {
    // Check if it is a pinned buffer.
    for (auto it = tgpuPinned.begin(); it != tgpuPinned.end(); ++it) {
        if (it->host_ptr == hPtr) {
            munmap(it->host_ptr, it->mapped_size);
            close(it->fd);
            tgpuPinned.erase(it);
            return;
        }
    }
    free(hPtr);
}

void GPUInterface::FreePinnedHostMemory(void* hPtr) {
    FreeHostMemory(hPtr);
}

// GetDeviceHostPointer: for a pinned buffer the "device pointer" is the same
// physical address. TinyGPU does not expose a distinct device VA space for
// host-mapped memory in this version — return the host pointer cast to GPUPtr.
GPUPtr GPUInterface::GetDeviceHostPointer(void* hPtr) {
    return (GPUPtr)(uintptr_t)hPtr;
}

// ── Data transfers ────────────────────────────────────────────────────────────

void GPUInterface::MemcpyHostToDevice(GPUPtr dest, const void* src, size_t memSize) {
    // Write src to VRAM at offset dest via CMD_MMIO_WRITE on BAR2.
    tgpu_mmio_write(tgpuSock, tgpuDevId, /*bar=*/2, (uint64_t)dest, src, memSize);
}

void GPUInterface::MemcpyDeviceToHost(void* dest, const GPUPtr src, size_t memSize) {
    tgpu_mmio_read(tgpuSock, tgpuDevId, /*bar=*/2, (uint64_t)src, dest, memSize);
}

void GPUInterface::MemcpyDeviceToDevice(GPUPtr dest, GPUPtr src, size_t memSize) {
    // No device-side DMA in current protocol; stage through a temp host buffer.
    void* tmp = malloc(memSize);
    tgpu_mmio_read(tgpuSock,  tgpuDevId, /*bar=*/2, (uint64_t)src,  tmp,    memSize);
    tgpu_mmio_write(tgpuSock, tgpuDevId, /*bar=*/2, (uint64_t)dest, tmp, memSize);
    free(tmp);
}

void GPUInterface::MemsetShort(GPUPtr dest, unsigned short val, size_t count) {
    size_t   bytes = count * sizeof(unsigned short);
    uint16_t* buf  = (uint16_t*)malloc(bytes);
    for (size_t i = 0; i < count; ++i) buf[i] = val;
    tgpu_mmio_write(tgpuSock, tgpuDevId, /*bar=*/2, (uint64_t)dest, buf, bytes);
    free(buf);
}

// ── Debug ─────────────────────────────────────────────────────────────────────

void GPUInterface::PrintfDeviceInt(GPUPtr dPtr, int length) {
    int* hPtr = (int*)malloc(SIZE_INT * (size_t)length);
    MemcpyDeviceToHost(hPtr, dPtr, SIZE_INT * (size_t)length);
    printfInt(hPtr, length);
    free(hPtr);
}

}; // namespace tinygpu_device
