/*
 * GPUInterfaceTinyGPUHybrid.cpp
 *
 * BEAGLE TinyGPU hybrid backend, shared front end: connects to TinyGPU.app,
 * identifies the eGPU's vendor from PCI config space, selects the kernel
 * resource, and implements GPUInterface by calling the vendor's free
 * functions: GPUInterfaceTinyGPUHybridNV.cpp (the daemon path by default;
 * C++ dispatch with BEAGLE_NV_CPP_DISPATCH=1; the C++ runtime with
 * BEAGLE_NV_USE_DAEMON=0) and GPUInterfaceTinyGPUHybridAMD.cpp.
 *
 * Drop-in replacement for GPUInterfaceTinyGPU.cpp when built with -DFW_TINYGPU.
 * Select this file in CMakeLists instead of GPUInterfaceTinyGPU.cpp.
 *
 * The NV path this file used to hand-roll here (nv_init_helper.py boot, then
 * its own QMDs, GPFIFO and BAR1 copies) was replaced by the C++ runtime, which
 * follows tinygrad's code instead (TODO.md "Runtime roadmap", Step 3).
 */

#ifdef FW_TINYGPU

#include <cstdio>
#include "libhmsbeagle/beagle.h"
#include "libhmsbeagle/GPU/GPUImplDefs.h"
#include "libhmsbeagle/GPU/GPUImplHelper.h"
#include "libhmsbeagle/GPU/GPUInterface.h"
#include "libhmsbeagle/GPU/KernelResource.h"
#include "libhmsbeagle/GPU/TinyGPUHybridSocket.h"
#include "libhmsbeagle/GPU/GPUInterfaceTinyGPUHybridAMD.h"
#include "libhmsbeagle/GPU/GPUInterfaceTinyGPUHybridNV.h"

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/file.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

// ── File-scope globals (used by both helper functions and GPUInterface methods) ─
static int            g_tgSock = -1;
static int            g_tgLockFd = -1;   // tinygrad's nv_usb4.lock, held with the plugin's TinyGPU.app connection
static uint32_t       g_tgDevId = 0;
static uint16_t       g_tgVendorId = 0;
static uint16_t       g_tgDeviceId = 0;

// PCI vendor IDs this backend recognizes.
static constexpr uint16_t PCI_VENDOR_NVIDIA = 0x10de;
static constexpr uint16_t PCI_VENDOR_AMD    = 0x1002;

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
    // BEAGLE_TINYGPU_NO_LAUNCH=1 (offline tests against a fake TinyGPU.app,
    // libhmsbeagle/GPU/tinygpu_tests): never start the real app; fail instead.
    static const char* kAppPath = "/Applications/TinyGPU.app/Contents/MacOS/TinyGPU";
    const char* no_launch = getenv("BEAGLE_TINYGPU_NO_LAUNCH");
    for (int i = 0; i < 100; ++i) {
        int fd = socket(AF_UNIX, SOCK_STREAM, 0);
        if (fd < 0) { perror("TinyGPU socket"); return -1; }
        if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) == 0)
            return fd;
        close(fd);
        if (no_launch && no_launch[0] && strcmp(no_launch, "0") != 0) {
            fprintf(stderr, "TinyGPU: nothing is listening at %s and BEAGLE_TINYGPU_NO_LAUNCH is set; not starting TinyGPU.app\n", path);
            return -1;
        }
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

// tinygrad's temp(name): tempfile.gettempdir(), which is $TMPDIR (else /tmp), joined with name.
static std::string tg_temp_path(const char* name) {
    const char* tmpdir = getenv("TMPDIR");
    std::string dir = (tmpdir && tmpdir[0]) ? tmpdir : "/tmp";
    while (dir.size() > 1 && dir.back() == '/') dir.pop_back();
    return dir + "/" + name;
}

// tinygrad's System.flock_acquire("nv_usb4.lock"), which RemotePCIDevice.__init__ takes right after connecting for NV:0
// over TinyGPU.app (TODO.md plan step P5): while this process holds its connection, another BEAGLE or tinygrad process
// fails here at once instead of waiting forever on a server that serves one client at a time. Unlike tinygrad, which
// clears the process umask first, a created file is made world-writable with fchmod, leaving the host's umask alone.
static bool tg_lock_acquire() {
    std::string path = tg_temp_path("nv_usb4.lock");
    bool exists = access(path.c_str(), F_OK) == 0;  // tinygrad avoids O_CREAT on an existing file
    int fd = exists ? open(path.c_str(), O_RDWR | O_CLOEXEC) : open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0666);
    if (fd < 0) {
        fprintf(stderr, "TinyGPU: cannot open the lock file %s: %s\n", path.c_str(), strerror(errno));
        return false;
    }
    if (!exists) fchmod(fd, 0666);
    if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
        fprintf(stderr, "TinyGPU: Failed to acquire lock file nv_usb4.lock (another process has the eGPU). `sudo lsof %s` may "
                "help identify the process holding the lock.\n", path.c_str());
        close(fd);
        return false;
    }
    g_tgLockFd = fd;
    return true;
}

static void tg_lock_release() {
    if (g_tgLockFd >= 0) { close(g_tgLockFd); g_tgLockFd = -1; }
}

void tg_close(int& sock) {
    if (sock < 0) return;
    close(sock);
    sock = -1;
    tg_lock_release();
}

int tg_lock_fd() { return g_tgLockFd; }

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
    if (!isNVIDIA) {
        AmdFini();  // sends SIGTERM to amd_init_helper.py and waits for it to exit + adev.fini()
        if (tgpuSock >= 0) { close(tgpuSock); tgpuSock = -1; }
        return;
    }
    NvFini(this);
    tg_close(tgpuSock);   // an instance that shares the GPU has none (plan step P5)
}

int GPUInterface::Initialize() {
    fprintf(stderr, "TinyGPU: build stamp — GPUInterfaceTinyGPUHybrid.cpp compiled %s %s\n",
            __DATE__, __TIME__);
#ifdef TINYGPU_KERNELS_STAMP
    fprintf(stderr, "TinyGPU: build stamp — kernels/BeagleTinyGPU_kernels.h: %s\n",
            TINYGPU_KERNELS_STAMP);
#else
    fprintf(stderr, "TinyGPU: build stamp — TINYGPU_KERNELS_STAMP not defined "
                     "(BeagleTinyGPU_kernels.h missing or stale — this should not happen)\n");
#endif
    fflush(stderr);
    // TODO.md plan step P5: another instance in this process may have the GPU booted, on the one connection TinyGPU.app
    // serves.
    int shared = NvAttachShared(this);
    if (shared < 0) return BEAGLE_ERROR_GENERAL;
    if (shared > 0) {
        fprintf(stderr, "TinyGPU: device 0 PCI id = %04x:%04x (NVIDIA), booted by another instance in this process\n",
                g_tgVendorId, g_tgDeviceId);
        tgpuDevId = g_tgDevId;
        isNVIDIA  = true;
        return BEAGLE_SUCCESS;
    }
    g_tgSock = tg_open_socket();
    if (g_tgSock < 0) return BEAGLE_ERROR_GENERAL;
    if (!tg_lock_acquire()) { close(g_tgSock); g_tgSock = -1; return BEAGLE_ERROR_GENERAL; }
    // Enumerate devices: just probe device 0 for now.
    // A full probe would use TGC_PROBE; we keep it simple.
    g_tgDevId = 0;

    // Identify the vendor from real PCI config space (offset 0 = vendor ID
    // in the low 16 bits, device ID in the high 16 bits of the first
    // config dword) rather than assuming NVIDIA.
    uint32_t id01 = (uint32_t)tg_cfg_read(g_tgSock, g_tgDevId, /*offset=*/0, /*size=*/4);
    g_tgVendorId = (uint16_t)(id01 & 0xffff);
    g_tgDeviceId = (uint16_t)(id01 >> 16);
    const char* vendorName = (g_tgVendorId == PCI_VENDOR_NVIDIA) ? "NVIDIA"
                            : (g_tgVendorId == PCI_VENDOR_AMD)    ? "AMD"
                            : "unknown";
    fprintf(stderr, "TinyGPU: device 0 PCI id = %04x:%04x (%s)\n",
            g_tgVendorId, g_tgDeviceId, vendorName);
    fflush(stderr);

    tgpuSock  = g_tgSock;
    tgpuDevId = g_tgDevId;
    isNVIDIA  = (g_tgVendorId != PCI_VENDOR_AMD);   // default to the NV path unless AMD is positively identified
    if (!isNVIDIA) tg_lock_release();   // the lock is NV's (plan step P5); the AMD path is unchanged
    return BEAGLE_SUCCESS;
}

// An instance sharing the booted GPU has no connection of its own, only its NV instance (plan step P5).
int GPUInterface::GetDeviceCount() { return (tgpuSock >= 0 || nvGspState) ? 1 : 0; }

void GPUInterface::SetDevice(int deviceNumber, int paddedStateCount,
                              int categoryCount, int patternCount,
                              int unpaddedPatternCount, int tipCount, long flags) {
    if (!isNVIDIA) {
        AmdSetDevice(this, paddedStateCount, categoryCount, patternCount,
                     unpaddedPatternCount, tipCount, flags);
        return;
    }
    NvSetDevice(this, paddedStateCount, categoryCount, patternCount,
                unpaddedPatternCount, tipCount, flags);
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
    if (!isNVIDIA) { AmdSynchronizeHost(); return; }
    NvSynchronizeHost(this);
}

void GPUInterface::SynchronizeDevice() { SynchronizeHost(); }

void GPUInterface::SynchronizeDeviceWithIndex(int, int) { SynchronizeHost(); }

// ── GetFunction ───────────────────────────────────────────────────────────────

GPUFunction GPUInterface::GetFunction(const char* name) {
    if (!isNVIDIA) return AmdGetFunction(name);
    return NvGetFunction(this, name);
}

// ── LaunchKernelImpl ──────────────────────────────────────────────────────────

void GPUInterface::LaunchKernelImpl(GPUFunction fn, Dim3Int block, Dim3Int grid,
                                     int nPtr, int nTotal, GPUPtr* ptrs, unsigned int* ints) {
    if (!isNVIDIA) { AmdLaunchKernelImpl(fn, block, grid, nPtr, nTotal, ptrs, ints); return; }
    NvLaunchKernelImpl(this, fn, block, grid, nPtr, nTotal, ptrs, ints);
}

// ── LaunchKernel (variadic) ───────────────────────────────────────────────────

void GPUInterface::LaunchKernel(GPUFunction fn, Dim3Int block, Dim3Int grid,
                                 int nPtr, int nTotal, ...) {
    if (!fn) return;
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
    if (!fn) return;
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
    if (!isNVIDIA) return AmdAllocateMemory(sz);
    return NvAllocateMemory(sz);
}

GPUPtr GPUInterface::AllocateRealMemory(size_t n)  { return AllocateMemory(n * sizeof(double)); }
GPUPtr GPUInterface::AllocateIntMemory(size_t n)   { return AllocateMemory(n * sizeof(int)); }

GPUPtr GPUInterface::CreateSubPointer(GPUPtr base, size_t off, size_t) {
    return base + (GPUPtr)off;
}

// No padding, as in CUDA: a sub-pointer is a plain address, and BeagleGPUImpl's transpose and convolution offset lists
// assume unpadded matrix strides (STATUS.md R25).
size_t GPUInterface::AlignMemOffset(size_t off) { return off; }

void GPUInterface::MemcpyHostToDevice(GPUPtr dst, const void* src, size_t sz) {
    if (!isNVIDIA) { AmdMemcpyHostToDevice(dst, src, sz); return; }
    NvMemcpyHostToDevice(this, dst, src, sz);
}

void GPUInterface::MemcpyDeviceToHost(void* dst, const GPUPtr src, size_t sz) {
    if (!isNVIDIA) { AmdMemcpyDeviceToHost(dst, src, sz); return; }
    NvMemcpyDeviceToHost(this, dst, src, sz);
}

void GPUInterface::MemcpyDeviceToDevice(GPUPtr dst, GPUPtr src, size_t sz) {
    if (!sz) return;
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
    if (isNVIDIA) snprintf(name, len, "TinyGPU-NV-Hybrid");
    else          snprintf(name, len, "TinyGPU-AMD-Hybrid (%04x:%04x)", g_tgVendorId, g_tgDeviceId);
}
void GPUInterface::GetDeviceDescription(int, char* desc) {
    if (isNVIDIA) snprintf(desc, 128, "BEAGLE hybrid NV backend via tinygrad + TinyGPU socket");
    else          snprintf(desc, 128, "BEAGLE hybrid AMD backend via tinygrad + TinyGPU socket");
}
long GPUInterface::GetDeviceTypeFlag(int) { return BEAGLE_FLAG_PROCESSOR_GPU; }
BeagleDeviceImplementationCodes GPUInterface::GetDeviceImplementationCode(int) {
    return isNVIDIA ? BEAGLE_TINYGPU_DEVICE_NVIDIA_GPU : BEAGLE_TINYGPU_DEVICE_AMD_GPU;
}
bool GPUInterface::GetSupportsDoublePrecision(int) { return false; }
size_t GPUInterface::GetAvailableMemory() {
    if (!isNVIDIA) return AmdGetAvailableMemory();
    return NvGetAvailableMemory();
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
