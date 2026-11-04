// C++ side of golden_transport.py: TinyGPUTransport.h makes the calls golden_transport.py's tinygrad client makes, in the
// same order, and prints the same result lines.
#include "libhmsbeagle/GPU/TinyGPUTransport.h"
#include <cstdio>
using namespace tinygpu_device;

int main() {
    TGTransport t;
    std::string e = t.open();
    if (!e.empty()) { fprintf(stderr, "open: %s\n", e.c_str()); return 1; }
    uint64_t v = 0, addr = 0, size = 0;
    if (!t.read_config(0, 4, v, e)) return fprintf(stderr, "%s\n", e.c_str()), 1;
    printf("cfg0=%#llx\n", (unsigned long long)v);
    if (!t.write_config(4, 0x6, 2, e) || !t.write_config_flush(4, 0x7, 2, e)) return fprintf(stderr, "%s\n", e.c_str()), 1;
    for (uint32_t bar : {0u, 0u, 1u}) {
        if (!t.bar_info(bar, addr, size, e)) return fprintf(stderr, "%s\n", e.c_str()), 1;
        printf("bar%u=%#llx\n", bar, (unsigned long long)size);
    }
    uint8_t data[32], in[64];
    for (int i = 0; i < 32; ++i) data[i] = (uint8_t)i;
    if (!t.bulk_write(1, 0x1000, data, 32, e) || !t.bulk_read(1, 0x2000, in, 64, e)) return fprintf(stderr, "%s\n", e.c_str()), 1;
    printf("read=");
    for (uint8_t b : in) printf("%02x", b);
    printf("\n");
    uint8_t junk[16];
    printf("read error=%s\n", t.bulk_read(1, 0xdead000, junk, 16, e) ? "none" : e.c_str());
    printf("cfg error=%s\n", t.read_config(0xbad, 4, v, e) ? "none" : e.c_str());
    if (!t.read_config(0, 4, v, e)) return fprintf(stderr, "%s\n", e.c_str()), 1;
    printf("cfg0=%#llx\n", (unsigned long long)v);
    if (!t.resize_bar(1, e)) return fprintf(stderr, "%s\n", e.c_str()), 1;
    TGSysmem m;
    if (!t.alloc_sysmem(0x5000, false, m, e)) return fprintf(stderr, "%s\n", e.c_str()), 1;
    printf("sysmem=%#llx", (unsigned long long)m.mapped_size);
    for (uint64_t p : m.paddrs) printf(" %#llx", (unsigned long long)p);
    printf("\n");
    t.close();
    return 0;
}
