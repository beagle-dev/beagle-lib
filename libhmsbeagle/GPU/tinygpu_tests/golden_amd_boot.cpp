// The C++ AM boot (TinyGPUAMDBoot.h) on fake_amd_device.py's card, for golden_amd_boot.py: PCIIfaceBase.__init__'s
// RESIZE_BAR, AMDev's boot, then fini (after a line on stdin with --pause, so the harness can post interrupts first). Prints
// what golden_amd_boot.py prints for tinygrad's AMDev. With --session <pool size>, the daemon's whole session instead
// (TinyGPUAMDDevice.h): the boot, AMDDevice.__init__, cmd_handoff's allocations (its reply printed as the daemon's
// JSON keys), then the exit's finalize; with --restore-fini, that finalize on an AMDev restored from the boot's fini_state()
// (taken before AMDDevice's setup) over a second transport on the same connection, as the crash guard runs it (plan step A2k).
// --chip <arch> passes the arch the card's PCI device ID names (AMBootOptions.chip). With --pte-flags <discovery table>, no
// boot: on an AMDev that only parses the table (as the crash guard's), AM_GMC.get_pte_flags and is_pte_huge_page for every
// level, table or leaf, fragment, uncached, system, snooped and valid, or the refusal of the table's IP versions (plan step N12).
//   golden_amd_boot <blobs file: name path sha256 per line> [--pause] [--allow-mode1] [--chip <arch>] [--session <pool size> [--die-before-fini | --restore-fini]]
//   golden_amd_boot --pte-flags <discovery table>
#include "libhmsbeagle/GPU/TinyGPUFirmware.h"
#include "libhmsbeagle/GPU/TinyGPUAMDDevice.h"
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
using namespace tinygpu_device;

int main(int argc, char** argv) {
    if (argc > 2 && std::string(argv[1]) == "--pte-flags") {
        std::ifstream tf(argv[2], std::ios::binary);
        std::vector<uint8_t> table((std::istreambuf_iterator<char>(tf)), std::istreambuf_iterator<char>());
        amboot::AMFiniState s{};
        memcpy(s.discovery, table.data(), std::min(table.size(), sizeof(s.discovery)));
        TGTransport t;   // never opened: this AMDev sends nothing
        try {
            amboot::AMDev a(t, s);
            for (int lv = 0; lv < 4; ++lv)
                for (int tbl = 0; tbl < 2; ++tbl)
                    for (int frag = 0; frag < 32; ++frag)
                        for (int bits = 0; bits < 16; ++bits) {   // uncached, system, snooped, valid
                            const bool u = bits & 8, sy = bits & 4, sn = bits & 2, v = bits & 1;
                            const uint64_t f = a.gmc->get_pte_flags(lv, tbl, frag, u, sy, sn, v);
                            printf("%d %d %d %d %d %d %d -> 0x%llx %d\n", lv, tbl, frag, (int)u, (int)sy, (int)sn, (int)v, (unsigned long long)f,
                                   (int)a.gmc->is_pte_huge_page(lv, f));
                        }
        } catch (const TGPyError& e) { printf("error %s\n", e.py().c_str()); }
        return 0;
    }
    std::map<std::string, std::pair<std::string, std::string>> blobs;
    std::ifstream bf(argv[1]);
    for (std::string line; std::getline(bf, line);) {
        std::istringstream ss(line);
        std::string name, path, sha;
        ss >> name >> path >> sha;
        blobs[name] = {path, sha};
    }
    bool pause = false, allow_mode1 = false, session = false, die = false, restore = false;
    uint64_t pool_size = 0;
    std::string chip;
    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--pause") pause = true;
        if (std::string(argv[i]) == "--allow-mode1") allow_mode1 = true;
        if (std::string(argv[i]) == "--session" && i + 1 < argc) { session = true; pool_size = std::stoull(argv[++i]); }
        if (std::string(argv[i]) == "--die-before-fini") die = true;   // a crash with the queues live (test_a2i.py's guard check)
        if (std::string(argv[i]) == "--restore-fini") restore = true;
        if (std::string(argv[i]) == "--chip" && i + 1 < argc) chip = argv[++i];
    }
    amboot::AMBlobLoader loader = [&](const std::string& name, std::vector<uint8_t>& out) -> std::string {
        auto it = blobs.find(name);
        if (it == blobs.end()) return "not in the blobs file";
        std::ifstream f(it->second.first, std::ios::binary);
        out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
        if (tg_sha256_hex(out.data(), out.size()) != it->second.second) return "sha256 mismatch";
        return "";
    };
    TGTransport t;
    std::string err = t.open();
    if (!err.empty()) { printf("open: %s\n", err.c_str()); return 1; }
    if (session) {   // GPUInterface::Initialize's first request, as the daemon's session (amd_daemon_session.py) has it
        uint64_t id = 0;
        if (!t.read_config(0, 4, id, err)) { printf("open: the PCI id read: %s\n", err.c_str()); return 1; }
    }
    t.resize_bar(0, err);   // PCIIfaceBase.__init__: contextlib.suppress(Exception)
    try {
        amboot::AMBootOptions o;
        o.refuse_mode1 = !allow_mode1;
        if (!chip.empty()) o.chip = chip.c_str();
        amboot::AMDev adev(t, loader, o);
        if (session) {
            amboot::AMFiniState fs;
            adev.fini_state(fs);   // as the plugin sends it to the crash guard: after the boot, before AMDDevice's setup
            amboot::AMDDeviceState d;
            amboot::am_device_init(adev, d);
            AMDHandoff h;
            std::vector<uint8_t*> maps;
            amboot::am_handoff(adev, d, pool_size, h, maps);
            auto u = [](uint64_t v) { return (unsigned long long)v; };
            printf("handoff");
            const std::pair<const char*, const AMDHandoffObj*> objs[] = {{"compute_ring", &h.compute_ring}, {"compute_rptr", &h.compute_rptr},
                {"compute_wptr", &h.compute_wptr}, {"sdma_ring", &h.sdma_ring}, {"sdma_rptr", &h.sdma_rptr}, {"sdma_wptr", &h.sdma_wptr},
                {"signal", &h.signal}, {"shadow", &h.shadow}, {"kargs", &h.kargs}, {"staging", &h.staging}};
            for (const auto& [k, o] : objs) printf(" %s_map=%llu %s_off=%llu", k, u(o->map), k, u(o->off));
            const std::pair<const char*, uint64_t> vals[] = {{"compute_ring_size", h.compute_ring_size}, {"compute_doorbell", h.compute_doorbell},
                {"compute_put", h.compute_put}, {"sdma_ring_size", h.sdma_ring_size}, {"sdma_doorbell", h.sdma_doorbell}, {"sdma_put", h.sdma_put},
                {"signal_va", h.signal_va}, {"shadow_va", h.shadow_va}, {"kargs_va", h.kargs_va}, {"kargs_size", h.kargs_size},
                {"staging_va", h.staging_va}, {"staging_size", h.staging_size}, {"pool_va", h.pool_va}, {"pool_size", h.pool_size},
                {"timeline_value", h.timeline_value}, {"vram_size", h.vram_size}, {"target_major", h.target_major}, {"xccs", h.xccs},
                {"cu_cnt", h.cu_cnt}, {"se_cnt", h.se_cnt}, {"max_slots_scratch_cu", h.max_slots_scratch_cu}, {"lds_size_in_kb", h.lds_size_in_kb},
                {"ih_ring_paddr", h.ih_ring_paddr}, {"ih_ring_size", h.ih_ring_size}, {"is_vf", h.is_vf}, {"bar0_size", h.bar0_size},
                {"bar2_size", h.bar2_size}, {"bar5_size", h.bar5_size}, {"reg_hdp_remap", h.reg_hdp_remap}, {"reg_ih_wptr", h.reg_ih_wptr},
                {"reg_ih_rptr", h.reg_ih_rptr}, {"reg_ih_cntl", h.reg_ih_cntl}, {"reg_fault_status", h.reg_fault_status},
                {"reg_fault_addr_lo", h.reg_fault_addr_lo}, {"reg_fault_addr_hi", h.reg_fault_addr_hi}, {"reg_fault_cntl", h.reg_fault_cntl},
                {"nmaps", h.nmaps}, {"blob_size", h.blob_size}};
            for (const auto& [k, v] : vals) printf(" %s=%llu", k, u(v));
            for (uint64_t i = 0; i < h.nmaps; ++i) printf(" map%llu_size=%llu", u(i), u(h.map_size[i]));
            printf("\n");
            if (die) { fflush(stdout); _exit(3); }
            if (restore) {
                TGTransport t2;
                t2.adopt(dup(t.fd()), -1);
                t2.seed_bar(0, fs.vram_bytes);
                t2.seed_bar(5, fs.mmio_bytes);
                amboot::AMDev r(t2, fs);
                std::string why;
                const bool off = amboot::am_device_fini_safe(r, why);
                printf("fini is_err_state=%d queues_off=%d%s%s\n", (int)r.is_err_state, (int)off, why.empty() ? "" : " ", why.c_str());
                fflush(stdout);
                return 0;
            }
            amboot::am_device_fini(adev);
            printf("fini is_err_state=%d\n", (int)adev.is_err_state);
            fflush(stdout);
            return 0;
        }
        printf("booted partial=%d vram_size=%llu large_bar=%d xccs=%d mc_base=%s fb_end=%s tmr_size=%s\n", (int)adev.partial_boot,
               (unsigned long long)adev.vram_size, (int)adev.large_bar, adev.gfx->xccs, tg_hex(adev.gmc->mc_base).c_str(),
               tg_hex(adev.gmc->fb_end).c_str(), tg_hex(adev.psp->tmr_size).c_str());
        if (pause) {
            printf("PAUSE\n");
            fflush(stdout);
            std::string go;
            std::getline(std::cin, go);
        }
        adev.fini();
        printf("fini is_err_state=%d\n", (int)adev.is_err_state);
    } catch (const TGPyError& e) {
        printf("error %s\n", e.py().c_str());
    } catch (const am::AMRegError& e) {
        printf("error AMRegError: %s\n", e.what());
    }
    fflush(stdout);
    return 0;
}
