// golden_boot.cpp -- the C++ side of TODO.md plan step C11's golden test (golden_boot.py drives it): TinyGPUHybridNVBoot.h's
// steps against golden_boot.py's fake TinyGPU.app (APL_REMOTE_SOCK), printing what golden_boot.py's tinygrad side prints.
//   golden_boot early     C11a: PCIIfaceBase's BAR resize, then NVDev.__init__'s map_bar(0), _early_ip_init, _early_mmu_init
//   golden_boot flcn      C11b: the same, then NV_FLCN.init_sw with nv_init_helper's VBIOS capture and teardown images
//   golden_boot sw        C11c, C11d: the same, then NV_GSP.init_sw (on a COT boot NV_FLCN_COT.init_sw instead of NV_FLCN's)
#include "libhmsbeagle/GPU/TinyGPUHybridNVBoot.h"

#include <cstdio>
#include <string>

using namespace tinygpu_device;

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode != "early" && mode != "flcn" && mode != "sw") return fprintf(stderr, "usage: golden_boot early|flcn|sw\n"), 2;
    TGTransport t;
    std::string err = t.open();
    if (!err.empty()) return fprintf(stderr, "transport: %s\n", err.c_str()), 1;
    NVBootDev d;
    d.t = &t;
    try {
        nv_boot_pci(d);
        nv_boot_early_ip_init(d);
        nv_boot_early_mmu_init(d);
        printf("chip 0x%x %s %s mmu %d fmc %d vram %llu bar1 %llu large %d root %llu\n", d.chip_id, d.chip_name.c_str(), d.fw_name.c_str(),
               d.mmu_ver, (int)d.fmc_boot, (unsigned long long)d.vram_size, (unsigned long long)d.bar1_size, (int)d.large_bar,
               (unsigned long long)d.mem->mm->root_page_table.paddr);
        NVFlcnImages f;
        NVTeardownImages td;
        if (mode == "sw" && d.fmc_boot) {
            nv_boot_end_booting(d);
            NVCotImages cot;
            NVBootMem fa, fi;
            nv_boot_cot_init_sw(d, cot, fa, fi);
            printf("cot boot_args 0x%llx fmc 0x%llx hash %zu 0x%x sig %zu 0x%x pkey %zu 0x%x\n", (unsigned long long)cot.fmc_boot_args_sysmem,
                   (unsigned long long)cot.fmc_booter_bar1, cot.hash.size(), cot.hash[0], cot.sig.size(), cot.sig[0], cot.pkey.size(), cot.pkey[0]);
        } else if (mode == "flcn" || mode == "sw") {
            nv_boot_end_booting(d);
            nv_boot_flcn_init_sw(d, f, td);
            printf("flcn frts 0x%llx at 0x%llx desc 0x%x 0x%x 0x%x 0x%x 0x%x 0x%x 0x%x 0x%x booter at 0x%llx data 0x%x+0x%x code 0x%x+0x%x\n",
                   (unsigned long long)f.frts_offset, (unsigned long long)f.frts_image_paddr, f.imem_pa, f.imem_va, f.imem_sz, f.dmem_pa, f.dmem_sz,
                   f.pkc_off, f.engid, f.ucodeid, (unsigned long long)f.booter_image_paddr, f.booter_data_off, f.booter_data_sz, f.booter_code_off,
                   f.booter_code_sz);
            if (td.present)
                printf("teardown sb at 0x%llx unload at 0x%llx data 0x%x+0x%x code 0x%x+0x%x\n", (unsigned long long)td.sb_paddr,
                       (unsigned long long)td.unload_paddr, td.unload_data_off, td.unload_data_sz, td.unload_code_off, td.unload_code_sz);
            else printf("teardown off\n");
        }
        if (mode == "sw") {
            NVGspBoot g;
            nv_boot_gsp_init_sw(d, g, d.fmc_boot ? nullptr : &f);
            printf("gsp rm_args 0x%llx libos 0x%llx wpr_meta 0x%llx radix3 0x%llx sig 0x%llx booter 0x%llx seq %u classes 0x%x 0x%x 0x%x 0x%x\n",
                   (unsigned long long)g.rm_args_sysmem, (unsigned long long)g.libos_args_sysmem, (unsigned long long)g.wpr_meta_sysmem,
                   (unsigned long long)g.gsp_radix3_addrs[0], (unsigned long long)g.gsp_signature_bar1, (unsigned long long)g.booter_bar1, g.cmd_q->seq,
                   g.gpfifo_class, g.compute_class, g.dma_class, g.viddec_class);
        }
    } catch (const NVError& e) {
        printf("error %s\n", e.py().c_str());
    }
    t.close();
    return 0;
}
