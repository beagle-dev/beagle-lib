// C++ side of golden_flcn_hw.py (TODO.md plan step C9): TinyGPUHybridNVFalcon.h's NVFalcon::init_hw (NV_FLCN.init_hw with
// nv_init_helper's execute_hs wrapper) against golden_flcn_hw.py's scripted TinyGPU.app, through TinyGPUTransport.h
// (APL_REMOTE_SOCK), printing the result lines golden_flcn_hw.py prints for tinygrad's and nv_init_helper's code.
//   golden_flcn_hw key=value ...   (frts_paddr frts_offset imem_pa imem_va imem_sz dmem_pa dmem_sz pkc_off engid ucodeid booter_paddr
//                                   booter_data_off booter_data_sz booter_code_off booter_code_sz libos wpr_meta chip_id wait_ms)
// With cot=1 (plan step B2): NVFalcon::cot_init_hw instead, with boot_args fmc hash sig pkey, the FMC boot parameters' page in the
// file GOLDEN_BOOT_ARGS.
#include "libhmsbeagle/GPU/TinyGPUHybridNVFalcon.h"

#include <cstdio>
#include <map>
#include <string>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

using namespace tinygpu_device;

static std::map<std::string, std::string> g_args;
static uint64_t arg(const char* k) { return strtoull(g_args.at(k).c_str(), nullptr, 0); }
static std::vector<uint32_t> words(const char* k) {
    std::vector<uint32_t> out;
    const std::string& w = g_args.at(k);
    for (size_t p = 0; p < w.size();) {
        size_t c = w.find(',', p);
        out.push_back((uint32_t)strtoul(w.substr(p, c - p).c_str(), nullptr, 0));
        p = c == std::string::npos ? w.size() : c + 1;
    }
    return out;
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        size_t eq = a.find('=');
        if (eq != std::string::npos) g_args[a.substr(0, eq)] = a.substr(eq + 1);
    }
    TGTransport t;
    std::string e = t.open();
    uint64_t bar_addr, bar_size;
    if (!e.empty() || !t.bar_info(0, bar_addr, bar_size, e)) return fprintf(stderr, "transport: %s\n", e.c_str()), 1;   // NVDev's map_bar(0)
    NVBar0 bar0{&t};
    const bool cot = g_args.count("cot") && arg("cot");
    NVFalcon flcn(bar0, (uint32_t)arg("chip_id"), cot);
    flcn.wait_ms = (int)arg("wait_ms");
    if (cot) {   // NV_FLCN_COT.init_hw on the prepared images, the boot parameters written to GOLDEN_BOOT_ARGS's page
        NVCotImages cim;
        int fd = open(getenv("GOLDEN_BOOT_ARGS"), O_RDWR);
        if (fd < 0) return fprintf(stderr, "open GOLDEN_BOOT_ARGS\n"), 1;
        cim.fmc_boot_args = (uint8_t*)mmap(nullptr, 0x1000, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        if (cim.fmc_boot_args == MAP_FAILED) return fprintf(stderr, "mmap\n"), 1;
        cim.fmc_boot_args_sysmem = arg("boot_args"); cim.fmc_booter_bar1 = arg("fmc");
        cim.hash = words("hash"); cim.sig = words("sig"); cim.pkey = words("pkey");
        bool gsp_started = false;
        try { flcn.cot_init_hw(cim, arg("wpr_meta"), arg("libos"), [&] { gsp_started = true; }); }
        catch (const NVError& x) { printf("error=%s\n", x.py().c_str()); }
        printf("gsp_started=%d\n", gsp_started ? 1 : 0);
        munmap(cim.fmc_boot_args, 0x1000);
        t.close();
        return 0;
    }
    NVFlcnImages im;
    im.frts_image_paddr = arg("frts_paddr"); im.frts_offset = arg("frts_offset");
    im.imem_pa = (uint32_t)arg("imem_pa"); im.imem_va = (uint32_t)arg("imem_va"); im.imem_sz = (uint32_t)arg("imem_sz");
    im.dmem_pa = (uint32_t)arg("dmem_pa"); im.dmem_sz = (uint32_t)arg("dmem_sz"); im.pkc_off = (uint32_t)arg("pkc_off");
    im.engid = (uint32_t)arg("engid"); im.ucodeid = (uint32_t)arg("ucodeid");
    im.booter_image_paddr = arg("booter_paddr");
    im.booter_data_off = (uint32_t)arg("booter_data_off"); im.booter_data_sz = (uint32_t)arg("booter_data_sz");
    im.booter_code_off = (uint32_t)arg("booter_code_off"); im.booter_code_sz = (uint32_t)arg("booter_code_sz");
    bool gsp_started = false;
    try {
        flcn.init_hw(im, arg("libos"), arg("wpr_meta"), [&] { gsp_started = true; }, [&](uint32_t mbx0) { gsp_started = mbx0 == 0; });
    } catch (const NVError& x) {
        printf("error=%s\n", x.py().c_str());
    }
    printf("gsp_started=%d\n", gsp_started ? 1 : 0);
    t.close();
    return 0;
}
