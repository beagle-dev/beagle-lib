/*
 * The IOKit calls of TinyGPU.app's server (tinygrad-hcq1 extra/usbgpu/tbgpu/installer/Shared/server.c, compiled from the
 * pin, never copied here), stubbed so the real server runs with no eGPU (TODO.md plan step C3: test_c3_transport.sh
 * drives the C++ client against it). BARs are anonymous memory, config space answers 10de:2882 (an RTX 4060), and
 * PrepareDMA returns 16 KiB segments at fake device addresses. A RESET, which BEAGLE must never send, aborts the server.
 *     server_stub <socket path>
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <IOKit/IOKitLib.h>

int run_server(const char* sock_path);

const mach_port_t kIOMainPortDefault = 0;
static const uint64_t kBarSize[6] = {16 << 20, 64 << 20, 0, 0, 0, 0};
static int g_allocs = 0;

CFMutableDictionaryRef IOServiceNameMatching(const char* name) { (void)name; return NULL; }
io_service_t IOServiceGetMatchingService(mach_port_t port, CFDictionaryRef matching) { (void)port; (void)matching; return 1; }
IONotificationPortRef IONotificationPortCreate(mach_port_t port) { (void)port; return (IONotificationPortRef)1; }
void IONotificationPortSetDispatchQueue(IONotificationPortRef n, dispatch_queue_t q) { (void)n; (void)q; }
kern_return_t IOServiceAddInterestNotification(IONotificationPortRef n, io_service_t s, const io_name_t t, IOServiceInterestCallback cb,
                                               void* ref, io_object_t* notif) {
    (void)n; (void)s; (void)t; (void)cb; (void)ref; *notif = 1; return KERN_SUCCESS;
}
kern_return_t IOServiceOpen(io_service_t s, task_port_t task, uint32_t type, io_connect_t* conn) {
    (void)s; (void)task; (void)type; *conn = 1; return KERN_SUCCESS;
}
kern_return_t IOServiceClose(io_connect_t conn) { (void)conn; return KERN_SUCCESS; }
kern_return_t IOObjectRelease(io_object_t o) { (void)o; return KERN_SUCCESS; }

kern_return_t IOConnectMapMemory64(io_connect_t conn, uint32_t type, task_port_t task, mach_vm_address_t* addr, mach_vm_size_t* size,
                                   IOOptionBits options) {
    (void)conn; (void)task; (void)options;
    if (type >= 6 || !kBarSize[type]) return KERN_FAILURE;
    void* p = mmap(NULL, kBarSize[type], PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    if (p == MAP_FAILED) return KERN_FAILURE;
    *addr = (mach_vm_address_t)p;
    *size = kBarSize[type];
    return KERN_SUCCESS;
}
kern_return_t IOConnectUnmapMemory64(io_connect_t conn, uint32_t type, task_port_t task, mach_vm_address_t addr) {
    (void)conn; (void)task;
    if (type < 6 && kBarSize[type]) munmap((void*)addr, kBarSize[type]);
    return KERN_SUCCESS;
}

// selector 0: config read (value in output[0]); 1: config write; 2: reset
kern_return_t IOConnectCallMethod(mach_port_t conn, uint32_t sel, const uint64_t* in, uint32_t in_cnt, const void* in_s, size_t in_s_cnt,
                                  uint64_t* out, uint32_t* out_cnt, void* out_s, size_t* out_s_cnt) {
    (void)conn; (void)in_cnt; (void)in_s; (void)in_s_cnt; (void)out_s; (void)out_s_cnt;
    if (sel == 2) { fprintf(stderr, "server_stub: RESET received; BEAGLE must never send one\n"); abort(); }
    if (sel == 0) { out[0] = in[0] == 0 ? 0x288210de : 0x12345678; *out_cnt = 1; }
    return KERN_SUCCESS;
}

// selector 3: PrepareDMA, (device address, length) pairs, 16 KiB segments; an allocation of 0x777000 bytes fails
kern_return_t IOConnectCallStructMethod(mach_port_t conn, uint32_t sel, const void* in, size_t in_cnt, void* out, size_t* out_cnt) {
    (void)conn; (void)in;
    if (sel != 3 || in_cnt == 0x777000) return KERN_FAILURE;
    uint64_t* pairs = (uint64_t*)out, base = 0x100000000ull + (uint64_t)g_allocs++ * 0x1000000ull;
    size_t n = 0;
    for (uint64_t off = 0; off < in_cnt && 2 * n + 3 < *out_cnt / 8; off += 0x4000, ++n) {
        pairs[2 * n] = base + off;
        pairs[2 * n + 1] = in_cnt - off < 0x4000 ? in_cnt - off : 0x4000;
    }
    pairs[2 * n] = pairs[2 * n + 1] = 0;
    return KERN_SUCCESS;
}

int main(int argc, char** argv) {
    if (argc != 2) { fprintf(stderr, "usage: server_stub <socket path>\n"); return 2; }
    setvbuf(stdout, NULL, _IOLBF, 0);
    return run_server(argv[1]);
}
