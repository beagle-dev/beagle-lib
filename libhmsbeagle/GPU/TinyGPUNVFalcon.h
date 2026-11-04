/*
 * TinyGPUNVFalcon.h
 *
 * TODO.md plan step C5: tinygrad's falcon primitives (NV_FLCN.reset, disable_ctx_req, execute_dma, start_cpu,
 * wait_cpu_halted, execute_hs; tinygrad/runtime/support/nv/ip.py:212-283 at a9830e2b4) ported statement by statement onto
 * the registers of TinyGPUNVReg.h and TinyGPUNVBootTables.h, over TinyGPU.app's BAR0 (NVDev.rreg/wreg, nvdev.py:92-95), with
 * tinygrad's wait_cond (helpers.py:554-558) and the teardown nv_init_helper.py runs after the GSP unload (plan step P2's
 * NV_FLCN.fini_hw, section 5), with its failure semantics. Also what nv_init_helper's patches add around these primitives:
 * the 20 s sleep after SEC2 starts inside gsp.init_hw or a LEVEL_0 unload (patch 3). Plan step C9 added NV_FLCN.init_hw
 * (ip.py:186-210: FWSEC-FRTS, then booter_load, which starts GSP-RM) with nv_init_helper's execute_hs wrapper around it
 * (_execute_hs_with_frts_checks, plan step P1). Blackwell's COT boot (plan step B2) runs no falcon ucode from the host: there
 * NVFalcon has GB20x's registers and only the teardown's wait for the GSP's RISC-V core to halt (cot_fini_hw, nv_init_helper's
 * NV_FLCN_COT.fini_hw); every other primitive is Ada's (NV_FLCN).
 *
 * tinygrad's exceptions become NVError, carrying the Python type's name (TimeoutError, RuntimeError, AssertionError, ...)
 * and str(e), so the teardown catches what nv_init_helper catches and records the same text. Every hardware-facing step is
 * logged through TinyGPULog.h.
 */

#ifndef LIBHMSBEAGLE_GPU_TINYGPUNVFALCON_H
#define LIBHMSBEAGLE_GPU_TINYGPUNVFALCON_H

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <time.h>

#include "libhmsbeagle/GPU/TinyGPULog.h"
#include "libhmsbeagle/GPU/TinyGPUNVBootTables.h"
#include "libhmsbeagle/GPU/TinyGPUNVReg.h"
#include "libhmsbeagle/GPU/TinyGPUTransport.h"

namespace tinygpu_device {

// A Python exception as tinygrad or nv_init_helper raises it: type(e).__name__ and str(e).
struct NVError : std::runtime_error {
    std::string type;
    NVError(std::string t, const std::string& msg) : std::runtime_error(msg), type(std::move(t)) {}
    std::string py() const { return type + ": " + what(); }   // f"{type(e).__name__}: {e}"
};

inline uint32_t nv_lo32(uint64_t x) { return (uint32_t)(x & 0xffffffff); }
inline uint32_t nv_hi32(uint64_t x) { return (uint32_t)((x >> 32) & 0xffffffff); }

inline int64_t nv_now_ms() {   // int(time.perf_counter() * 1000)
    return (int64_t)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline void nv_sleep(double seconds) {   // time.sleep
    struct timespec ts;
    ts.tv_sec = (time_t)seconds;
    ts.tv_nsec = (long)((seconds - (double)ts.tv_sec) * 1e9);
    while (nanosleep(&ts, &ts) != 0 && errno == EINTR) {}
}

// wait_cond(cb, value=True, timeout_ms=10000, msg=""): cb() until it equals value; value True compares with 1 and prints
// as True, as Python's int == True does.
constexpr int64_t kNVTrue = INT64_MIN;
template <class F> uint64_t nv_wait_cond(int timeout_ms, F&& cb, int64_t value, const std::string& msg) {
    const int64_t want = value == kNVTrue ? 1 : value;
    const int64_t start = nv_now_ms();
    uint64_t val = 0;
    while (nv_now_ms() - start < timeout_ms)
        if ((int64_t)(val = cb()) == want) return val;
    throw NVError("TimeoutError", msg + ". Timed out after " + std::to_string(timeout_ms) + " ms, condition not met: " +
                  std::to_string(val) + " != " + (value == kNVTrue ? std::string("True") : std::to_string(value)));
}
// wait_cond on a callback that returns a Python bool (a comparison), whose value prints as False
template <class F> void nv_wait_cond_true(int timeout_ms, F&& cb, const std::string& msg) {
    const int64_t start = nv_now_ms();
    while (nv_now_ms() - start < timeout_ms)
        if (cb()) return;
    throw NVError("TimeoutError", msg + ". Timed out after " + std::to_string(timeout_ms) + " ms, condition not met: False != True");
}

// NVDev.rreg/wreg (nvdev.py:92-95): one 4-byte MMIO_READ or MMIO_WRITE on BAR0, as tinygrad's RemoteMMIOInterface sends
// it (system.py:319-329). A failed transfer is tinygrad's RuntimeError.
struct NVBar0 {
    TGTransport* t;
    uint32_t rreg(uint32_t addr) {
        uint32_t v = 0;
        std::string err;
        if (!t->bulk_read(0, addr, &v, 4, err)) throw NVError("RuntimeError", err);
        return v;
    }
    void wreg(uint32_t addr, uint32_t value) {
        std::string err;
        if (!t->bulk_write(0, addr, &value, 4, err)) throw NVError("RuntimeError", err);
    }
};

// What nv_init_helper's prep_booter wrapper prepared for the teardown (plan step P2 (a)), as the daemon exports it: the
// two images in VRAM and their execute_hs arguments. FWSEC-SB runs with FWSEC-FRTS's (ip.py:190-193, from desc_v3).
struct NVTeardownImages {
    bool present = false;
    uint64_t sb_paddr = 0, unload_paddr = 0;
    uint32_t sb_imem_pa = 0, sb_imem_va = 0, sb_imem_sz = 0, sb_dmem_pa = 0, sb_dmem_sz = 0, sb_pkc_off = 0, sb_engid = 0, sb_ucodeid = 0;
    uint32_t unload_data_off = 0, unload_data_sz = 0, unload_code_off = 0, unload_code_sz = 0;   // beagle_unload_params
};

// What NV_FLCN.init_sw prepared for init_hw (plan step C9), as the daemon exports it at level flcn_hw: prep_ucode's FWSEC-FRTS
// image in VRAM with its desc_v3 load parameters and frts_offset, and prep_booter's booter_load image with its offsets.
struct NVFlcnImages {
    uint64_t frts_image_paddr = 0, frts_offset = 0;
    uint32_t imem_pa = 0, imem_va = 0, imem_sz = 0, dmem_pa = 0, dmem_sz = 0, pkc_off = 0, engid = 0, ucodeid = 0;   // desc_v3
    uint64_t booter_image_paddr = 0;
    uint32_t booter_data_off = 0, booter_data_sz = 0, booter_code_off = 0, booter_code_sz = 0;
};

// What NV_FLCN_COT.init_sw prepared for init_hw (plan step B2), as the daemon exports it at level flcn_hw on the COT boot: the
// FMC boot parameters' page (mapped from its TinyGPU.app fd) and device address, the FMC image's device address, and its hash,
// signature and public key (init_fmc_image's ELF sections as tinygrad casts them to 32-bit words).
struct NVCotImages {
    uint8_t* fmc_boot_args = nullptr;          // fmc_boot_args_view
    uint64_t fmc_boot_args_sysmem = 0, fmc_booter_bar1 = 0;
    std::vector<uint32_t> hash, sig, pkey;     // fmc_booter_hash, fmc_booter_sig, fmc_booter_pkey
};

// nv_init_helper's beagle_fini: what the unload (P1's suspend wait) and the teardown (P2) recorded, as the daemon's fini
// reply carries it. The teardown's own dict is kept in insertion order, as json.dumps writes it. On the COT boot (plan step
// B1) it also says whether the GSP's RISC-V core halted, false until the halt wait proves it, and MAILBOX0 after the halt.
struct NVFiniDiag {
    bool unload_ok = false, have_mailbox0 = false, have_wpr2 = false, have_cpuctl = false;   // which keys were set
    uint32_t mailbox0 = 0, wpr2_lo = 0, wpr2_hi = 0, riscv_cpuctl = 0;
    bool cot = false, halted = false, have_mailbox0_after_halt = false;                      // diag["halted"]: COT only
    uint32_t mailbox0_after_halt = 0;
    bool teardown_ran = false;                                 // diag["teardown"] exists
    std::vector<std::pair<std::string, std::string>> td;      // key -> JSON value
    bool have_td_result = false;
    std::string td_result;
    bool wpr2_down = false, teardown_ok = false, have_outcome = false;

    void td_set(const std::string& k, const std::string& json_value) {
        for (auto& kv : td) if (kv.first == k) { kv.second = json_value; return; }
        td.emplace_back(k, json_value);
    }
    static std::string str(const std::string& s) {   // a JSON string
        std::string o = "\"";
        for (char c : s) {
            if (c == '"' || c == '\\') { o += '\\'; o += c; }
            else if ((unsigned char)c < 0x20) { char b[8]; snprintf(b, sizeof(b), "\\u%04x", (unsigned char)c); o += b; }
            else o += c;
        }
        return o + "\"";
    }
    void result(const std::string& r) { have_td_result = true; td_result = r; td_set("result", str(r)); }
    std::string json() const {
        std::string j = "{\"unload_ok\": " + std::string(unload_ok ? "true" : "false");
        if (cot) j += ", \"halted\": " + std::string(halted ? "true" : "false");
        if (have_mailbox0) j += ", \"mailbox0\": " + std::to_string(mailbox0);
        if (have_wpr2) j += ", \"wpr2_lo\": " + std::to_string(wpr2_lo) + ", \"wpr2_hi\": " + std::to_string(wpr2_hi);
        if (have_cpuctl) j += ", \"riscv_cpuctl\": " + std::to_string(riscv_cpuctl);
        if (teardown_ran) {
            j += ", \"teardown\": {";
            for (size_t i = 0; i < td.size(); ++i) j += (i ? ", " : "") + str(td[i].first) + ": " + td[i].second;
            j += "}";
        }
        if (have_mailbox0_after_halt) j += ", \"mailbox0_after_halt\": " + std::to_string(mailbox0_after_halt);
        if (have_outcome) j += ", \"wpr2_down\": " + std::string(wpr2_down ? "true" : "false") + ", \"teardown_ok\": " +
                               std::string(teardown_ok ? "true" : "false");
        return j + "}";
    }
};

// NV_FLCN on Ada's registers (ip.py:88-283), and nv_init_helper's teardown on it; with cot, what GB20x's COT boot has of it:
// its registers and nv_init_helper's NV_FLCN_COT.fini_hw.
class NVFalcon {
public:
    NVFalcon(NVBar0& dev, uint32_t chip_id, bool cot = false) : cot(cot), dev_(dev), chip_id_(chip_id) {}

    const bool cot;                                           // GB20x's COT boot (plan step B2): kGB20xRegs
    const uint32_t falcon = 0x00110000, sec2 = 0x00840000;   // NV_FLCN.init_hw (ip.py:187)
    int wait_ms = 10000;                                      // wait_cond's timeout_ms (tests shorten it, as test_p2_teardown does)
    std::function<void(double)> sleep = nv_sleep;             // time.sleep (tests record it)
    bool sleep_after_sec2_start = false;                      // nv_init_helper's _in_gsp_init (patch 3)
    double cot_halt_timeout_s = 4.0;                          // nv_init_helper's _COT_HALT_TIMEOUT_S (tests shorten it)
    std::string chip_name;                                    // NVDev.chip_name, for the COT sequencer's refusal

    nv_regs::NVReg<NVBar0> reg(nv_regs::NVRegId id) const {
        return nv_regs::NVReg<NVBar0>(&dev_, (cot ? nv_regs::kGB20xRegs : nv_regs::kAdaRegs)[id]);
    }

    // NV_FLCN.execute_dma (ip.py:212-226)
    void execute_dma(uint32_t base, uint32_t cmd, uint64_t dest, uint64_t mem_off, uint64_t src, uint64_t size) {
        using namespace nv_regs;
        auto full = [&] { return reg(NV_PFALCON_FALCON_DMATRFCMD).with_base(base).read_bitfields()["full"]; };
        nv_wait_cond(wait_ms, full, 0, "DMA does not progress");

        reg(NV_PFALCON_FALCON_DMATRFBASE).with_base(base).write(nv_lo32(src >> 8));
        reg(NV_PFALCON_FALCON_DMATRFBASE1).with_base(base).write(nv_hi32(src >> 8) & 0x1ff);

        uint64_t xfered = 0;
        while (xfered < size) {
            nv_wait_cond(wait_ms, full, 0, "DMA does not progress");

            reg(NV_PFALCON_FALCON_DMATRFMOFFS).with_base(base).write(u32(dest + xfered));
            reg(NV_PFALCON_FALCON_DMATRFFBOFFS).with_base(base).write(u32(mem_off + xfered));
            reg(NV_PFALCON_FALCON_DMATRFCMD).with_base(base).write(cmd);
            xfered += 256;
        }

        nv_wait_cond(wait_ms, [&] { return reg(NV_PFALCON_FALCON_DMATRFCMD).with_base(base).read_bitfields()["idle"]; }, kNVTrue,
                     "DMA does not complete");
    }

    // NV_FLCN.start_cpu (ip.py:228-231), and nv_init_helper's patch 3 after it (_patched_start_cpu)
    void start_cpu(uint32_t base) {
        using namespace nv_regs;
        if (reg(NV_PFALCON_FALCON_CPUCTL).with_base(base).read_bitfields()["alias_en"] == 1)
            dev_.wreg(base + NV_PFALCON_FALCON_CPUCTL_ALIAS, 0x2);
        else reg(NV_PFALCON_FALCON_CPUCTL).with_base(base).write({{"startcpu", 1}});
        if (base == sec2 && sleep_after_sec2_start) {
            tg_log("SEC2 started inside gsp.init_hw or a LEVEL_0 unload: sleeping 20 s for the GC6 BSI domain to stabilise");
            sleep(20);
        }
    }

    // NV_FLCN.wait_cpu_halted (ip.py:233)
    void wait_cpu_halted(uint32_t base) {
        nv_wait_cond(wait_ms, [&] { return reg(nv_regs::NV_PFALCON_FALCON_CPUCTL).with_base(base).read_bitfields()["halted"]; }, kNVTrue,
                     "not halted");
    }

    // NV_FLCN.execute_hs (ip.py:235-265); returns (MAILBOX0, MAILBOX1) when a mailbox is given
    std::pair<uint32_t, uint32_t> execute_hs(uint32_t base, uint64_t img_paddr, uint64_t code_off, uint64_t data_off, uint64_t imemPa,
                                             uint64_t imemVa, uint64_t imemSz, uint64_t dmemPa, uint64_t dmemVa, uint64_t dmemSz,
                                             uint64_t pkc_off, uint64_t engid, uint64_t ucodeid, const uint64_t* mailbox = nullptr) {
        using namespace nv_regs;
        disable_ctx_req(base);

        // target=0 is FB (not in published headers)
        const uint32_t ctx_dma = 0;
        reg(NV_PFALCON_FBIF_TRANSCFG).with_base(base)[ctx_dma].update({{"target", 0}, {"mem_type", NV_PFALCON_FBIF_TRANSCFG_MEM_TYPE_PHYSICAL}});

        uint32_t cmd = (uint32_t)reg(NV_PFALCON_FALCON_DMATRFCMD).with_base(base).encode(
            {{"write", 0}, {"size", NV_PFALCON_FALCON_DMATRFCMD_SIZE_256B}, {"ctxdma", ctx_dma}, {"imem", 1}, {"sec", 1}});
        execute_dma(base, cmd, imemPa, imemVa, img_paddr + code_off - imemVa, imemSz);

        cmd = (uint32_t)reg(NV_PFALCON_FALCON_DMATRFCMD).with_base(base).encode(
            {{"write", 0}, {"size", NV_PFALCON_FALCON_DMATRFCMD_SIZE_256B}, {"ctxdma", ctx_dma}, {"imem", 0}, {"sec", 0}});
        execute_dma(base, cmd, dmemPa, dmemVa, img_paddr + data_off - dmemVa, dmemSz);

        reg(NV_PFALCON2_FALCON_BROM_PARAADDR).with_base(base)[0].write(u32(pkc_off));
        reg(NV_PFALCON2_FALCON_BROM_ENGIDMASK).with_base(base).write(u32(engid));
        reg(NV_PFALCON2_FALCON_BROM_CURR_UCODE_ID).with_base(base).write({{"val", ucodeid}});
        reg(NV_PFALCON2_FALCON_MOD_SEL).with_base(base).write({{"algo", NV_PFALCON2_FALCON_MOD_SEL_ALGO_RSA3K}});

        reg(NV_PFALCON_FALCON_BOOTVEC).with_base(base).write(u32(imemVa));

        if (mailbox) {
            reg(NV_PFALCON_FALCON_MAILBOX0).with_base(base).write(nv_lo32(*mailbox));
            reg(NV_PFALCON_FALCON_MAILBOX1).with_base(base).write(nv_hi32(*mailbox));
        }

        start_cpu(base);
        wait_cpu_halted(base);

        if (mailbox) {
            uint32_t m0 = reg(NV_PFALCON_FALCON_MAILBOX0).with_base(base).read();
            return {m0, reg(NV_PFALCON_FALCON_MAILBOX1).with_base(base).read()};
        }
        return {0, 0};
    }

    // NV_FLCN.init_hw (ip.py:186-210) with nv_init_helper's execute_hs wrapper (_execute_hs_with_frts_checks, plan step P1):
    // FWSEC-FRTS's pre- and post-checks, read and logged around its execute_hs. before_booter runs right before booter_load, after
    // which GSP-RM may run from sysmem; booter_done(MAILBOX0) once booter_load halted (0: it started GSP-RM, as nv_init_helper's
    // beagle_gsp_started records it).
    void init_hw(const NVFlcnImages& im, uint64_t libos_args_sysmem, uint64_t wpr_meta_sysmem, const std::function<void()>& before_booter = {},
                 const std::function<void(uint32_t)>& booter_done = {}) {
        using namespace nv_regs;
        reset(falcon);
        // the wrapper, before FWSEC-FRTS: the conditions tinygrad's (suppressed) wait_for_reset polls, ip.py:94-96
        const uint64_t plm = reg(NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_PRIV_LEVEL_MASK).read_bitfields()["read_protection_level0"];
        const uint32_t gfw = reg(NV_PGC6_AON_SECURE_SCRATCH_GROUP_05)[0].read() & 0xff;
        tg_log("before FWSEC-FRTS: read_protection_level0=%llu (tinygrad waits for 1), SCRATCH_GROUP_05[0]&0xff=0x%02x (waits for 0xff)",
               (unsigned long long)plm, gfw);
        execute_hs(falcon, im.frts_image_paddr, 0x0, im.imem_sz, im.imem_pa, im.imem_va, im.imem_sz, im.dmem_pa, 0x0, im.dmem_sz, im.pkc_off,
                   im.engid, im.ucodeid);
        // and after it: NVIDIA's FRTS post-checks (570.144 kernel_gsp_frts_tu102.c:486-523), before tinygrad's WPR2_HI assert
        const uint32_t scratch = reg(NV_PBUS_VBIOS_SCRATCH)[0x0e].read();
        const uint64_t wpr2_lo = reg(NV_PFB_PRI_MMU_WPR2_ADDR_LO).read_bitfields()["val"];
        const uint64_t expected = im.frts_offset >> 12;
        tg_log("after FWSEC-FRTS: VBIOS scratch 0x0E=0x%08x (FRTS error code 0x%x, 0 = none); WPR2_LO.val=0x%llx, frts_offset>>12=0x%llx (%s)",
               scratch, scratch >> 16, (unsigned long long)wpr2_lo, (unsigned long long)expected, wpr2_lo == expected ? "match" : "MISMATCH");
        if (reg(NV_PFB_PRI_MMU_WPR2_ADDR_HI).read() == 0) throw NVError("AssertionError", "WPR2 is not initialized");

        reset(falcon, true);

        // set up the mailbox
        reg(NV_PGSP_FALCON_MAILBOX0).write(nv_lo32(libos_args_sysmem));
        reg(NV_PGSP_FALCON_MAILBOX1).write(nv_hi32(libos_args_sysmem));

        // booter
        if (before_booter) before_booter();
        reset(sec2);
        const std::pair<uint32_t, uint32_t> mbx = execute_hs(sec2, im.booter_image_paddr, im.booter_code_off, im.booter_data_off, 0x0,
                                                             im.booter_code_off, im.booter_code_sz, 0x0, 0x0, im.booter_data_sz, 0x10, 1, 3,
                                                             &wpr_meta_sysmem);
        if (booter_done) booter_done(mbx.first);
        if (mbx.first != 0x0) {
            char m[80];
            snprintf(m, sizeof(m), "Booter failed to execute, mailbox is %08x, %08x", mbx.first, mbx.second);
            throw NVError("AssertionError", m);
        }

        reg(NV_PFALCON_FALCON_OS).with_base(falcon).write(0x0);
        if (reg(NV_PRISCV_RISCV_CPUCTL).with_base(falcon).read_bitfields()["active_stat"] != 1) throw NVError("AssertionError", "GSP Core is not active");
    }

    // NV_FLCN.disable_ctx_req (ip.py:267-269)
    void disable_ctx_req(uint32_t base) {
        reg(nv_regs::NV_PFALCON_FBIF_CTL).with_base(base).update({{"allow_phys_no_ctx", 1}});
        reg(nv_regs::NV_PFALCON_FALCON_DMACTL).with_base(base).write(0x0);
    }

    // NV_FLCN.reset (ip.py:271-283)
    void reset(uint32_t base, bool riscv = false) {
        using namespace nv_regs;
        auto engine_reg = reg(base == falcon ? NV_PGSP_FALCON_ENGINE : NV_PSEC_FALCON_ENGINE);
        engine_reg.write({{"reset", 1}});
        sleep(0.1);
        engine_reg.write({{"reset", 0}});

        nv_wait_cond(wait_ms, [&] { return reg(NV_PFALCON_FALCON_HWCFG2).with_base(base).read_bitfields()["mem_scrubbing"]; }, 0,
                     "Scrubbing not completed");

        if (riscv) reg(NV_PRISCV_RISCV_BCR_CTRL).with_base(base).write({{"core_select", 1}, {"valid", 0}, {"brfetch", 1}});
        else if (reg(NV_PFALCON_FALCON_HWCFG2).with_base(base).read_bitfields()["riscv"] == 1) {
            reg(NV_PRISCV_RISCV_BCR_CTRL).with_base(base).write({{"core_select", 0}});
            nv_wait_cond(wait_ms, [&] { return reg(NV_PRISCV_RISCV_BCR_CTRL).with_base(base).read_bitfields()["valid"]; }, kNVTrue,
                         "RISCV core not booted");
            reg(NV_PFALCON_FALCON_RM).with_base(base).write(chip_id_);
        }
    }

    // nv_init_helper's _flcn_fini_hw_teardown (plan step P2 (b)): kgspTeardown_TU102 after the GSP unload. As in NVIDIA
    // (kernel_gsp_tu102.c:597-620, kernel_gsp_booter_tu102.c:155) and nouveau (tu102_gsp_fini), a failed GSP reset, FWSEC-SB
    // or SEC2 reset is recorded and Booter Unload still runs; only Booter Unload and WPR2 decide the outcome. Runs only once
    // the unload is confirmed: before that no falcon is touched.
    void fini_hw(NVFiniDiag& diag, const NVTeardownImages& img) {
        using namespace nv_regs;
        if (!img.present) return;
        diag.teardown_ran = true;
        if (!diag.unload_ok) {
            diag.result("skipped: the GSP did not confirm its unload, so no falcon is touched");
            tg_log("teardown %s", diag.td_result.c_str());
            return;
        }
        tg_log("teardown: GSP reset, FWSEC-SB (VRAM 0x%llx), SEC2 reset, Booter Unload (VRAM 0x%llx)",
               (unsigned long long)img.sb_paddr, (unsigned long long)img.unload_paddr);
        try {
            tolerant_reset(falcon, diag, "gsp");
            // FWSEC-SB, with the arguments tinygrad runs FWSEC-FRTS with (ip.py:190-193)
            execute_hs(falcon, img.sb_paddr, 0x0, img.sb_imem_sz, img.sb_imem_pa, img.sb_imem_va, img.sb_imem_sz, img.sb_dmem_pa, 0x0,
                       img.sb_dmem_sz, img.sb_pkc_off, img.sb_engid, img.sb_ucodeid);
            uint32_t scratch = reg(NV_PBUS_VBIOS_SCRATCH)[nv570::NV_VBIOS_FWSECLIC_SCRATCH_INDEX_15].read();
            uint64_t plm = reg(NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_PRIV_LEVEL_MASK).read_bitfields()["read_protection_level0"];
            uint32_t gfw = reg(NV_PGC6_AON_SECURE_SCRATCH_GROUP_05)[0].read() & 0xff;
            diag.td_set("sb_error", std::to_string(scratch & 0xffff));   // logged, not fatal (NVIDIA: NV_ASSERT_FAILED and continue)
            diag.td_set("plm", std::to_string(plm));
            diag.td_set("gfw_progress", std::to_string(gfw));
            tg_log("teardown: FWSEC-SB ran: VBIOS scratch 0x15=0x%08x (SB error code 0x%x, 0 = none), read_protection_level0=%llu, "
                   "GFW progress 0x%02x", scratch, scratch & 0xffff, (unsigned long long)plm, gfw);
        } catch (const NVError& e) {   // NVIDIA: NV_ASSERT_FAILED, then Booter Unload regardless (kernel_gsp_tu102.c:599-620)
            if (!falcon_error(e)) throw;
            diag.td_set("sb_failed", NVFiniDiag::str(e.py()));
            tg_log("teardown: GSP reset or FWSEC-SB failed (%s); continuing with Booter Unload, as NVIDIA does", e.py().c_str());
        }
        bool body_error = false;
        NVError pending("", "");
        try {
            if (reg(NV_PFB_PRI_MMU_WPR2_ADDR_HI).read() == 0) {   // NVIDIA skips Booter Unload when WPR2 is already down
                diag.result("done: WPR2 already down after FWSEC-SB");
            } else {
                try { tolerant_reset(sec2, diag, "sec2"); }
                catch (const NVError& e) {   // NVIDIA: a non-fatal NV_ASSERT_OK (kernel_gsp_booter_tu102.c:155)
                    if (!falcon_error(e)) throw;
                    diag.td_set("sec2_reset_failed", NVFiniDiag::str(e.py()));
                    tg_log("teardown: SEC2 reset failed (%s); running Booter Unload anyway, as NVIDIA does", e.py().c_str());
                }
                const uint64_t mailbox = (0xffull << 32) | 0xff;   // booter_load's parameters (ip.py:202-205); mailboxes 0xFF for a normal unload
                auto mbx = execute_hs(sec2, img.unload_paddr, img.unload_code_off, img.unload_data_off, 0x0, img.unload_code_off,
                                      img.unload_code_sz, 0x0, 0x0, img.unload_data_sz, 0x10, 1, 3, &mailbox);
                uint32_t wpr2_hi = reg(NV_PFB_PRI_MMU_WPR2_ADDR_HI).read();
                diag.td_set("booter_mailbox0", std::to_string(mbx.first));
                diag.td_set("booter_mailbox1", std::to_string(mbx.second));
                char r[160];
                if (mbx.first == 0 && wpr2_hi == 0) snprintf(r, sizeof(r), "done: Booter Unload lowered WPR2");
                else snprintf(r, sizeof(r), "failed: Booter Unload returned mailbox0=0x%x and WPR2_HI=0x%x", mbx.first, wpr2_hi);
                diag.result(r);
            }
        } catch (const NVError& e) {   // Booter Unload's DMA or halt timeout
            if (!falcon_error(e)) { body_error = true; pending = e; }
            else diag.result("failed: " + e.py());
        }
        // finally: what WPR2 says now decides
        diag.wpr2_lo = reg(NV_PFB_PRI_MMU_WPR2_ADDR_LO).read();
        diag.wpr2_hi = reg(NV_PFB_PRI_MMU_WPR2_ADDR_HI).read();
        diag.have_wpr2 = true;
        diag.wpr2_down = diag.wpr2_hi == 0;
        // the next boot needs no power cycle only if WPR2 is down and Booter Unload was not needed or returned 0 (plan (b) 8)
        diag.teardown_ok = diag.wpr2_down && diag.have_td_result && diag.td_result.rfind("done:", 0) == 0;
        diag.have_outcome = true;
        tg_log("teardown %s; WPR2_HI=0x%08x: %s", diag.have_td_result ? diag.td_result.c_str() : "interrupted", diag.wpr2_hi,
               diag.teardown_ok ? "the next boot needs no power cycle" : "power-cycle before the next boot");
        if (body_error) throw pending;
    }

    // NV_FLCN_COT.init_hw (ip.py:311-326): the FMC boot parameters, the COT message to the FSP (kfsp_send_msg, with
    // nv_init_helper's wrapper), then the wait for the GSP's RISC-V core to leave its boot-ROM lockdown. before_cot runs right
    // before the message's first EMEM write: from there the FSP may start the FMC and GSP-RM, which run from sysmem
    // (nv_init_helper's beagle_gsp_started).
    void cot_init_hw(const NVCotImages& im, uint64_t wpr_meta_sysmem, uint64_t libos_args_sysmem, const std::function<void()>& before_cot = {}) {
        using namespace nv_regs;
        nv::GSP_ACR_BOOT_GSP_RM_PARAMS boot_args{};
        boot_args.gspRmDescOffset = wpr_meta_sysmem;
        boot_args.gspRmDescSize = (uint32_t)sizeof(nv::GspFwWprMeta);
        boot_args.target = nv::GSP_DMA_TARGET_COHERENT_SYSTEM;
        boot_args.bIsGspRmBoot = 1;
        nv::GSP_RM_PARAMS rm_args{};
        rm_args.bootArgsOffset = libos_args_sysmem;
        rm_args.target = nv::GSP_DMA_TARGET_COHERENT_SYSTEM;
        nv::GSP_FMC_BOOT_PARAMS params{};
        params.bootGspRmParams = boot_args;
        params.gspRmParams = rm_args;
        memcpy(im.fmc_boot_args, &params, sizeof(params));

        nv::NVDM_PAYLOAD_COT cot{};
        cot.version = 0x2;
        cot.size = (uint16_t)sizeof(nv::NVDM_PAYLOAD_COT);
        cot.frtsVidmemOffset = 0x1c00000;
        cot.frtsVidmemSize = 0x100000;
        cot.gspBootArgsSysmemOffset = im.fmc_boot_args_sysmem;
        cot.gspFmcSysmemOffset = im.fmc_booter_bar1;
        auto words = [](uint32_t* dst, size_t n, const std::vector<uint32_t>& src) {   // a ctypes array's item assignment
            if (src.size() > n) throw NVError("IndexError", "invalid index");
            for (size_t i = 0; i < src.size(); ++i) dst[i] = src[i];
        };
        words(cot.hash384, 12, im.hash);
        words(cot.signature, 96, im.sig);
        words(cot.publicKey, 96, im.pkey);
        std::vector<uint8_t> payload(sizeof(cot));
        memcpy(payload.data(), &cot, sizeof(cot));
        kfsp_send_msg(nv::NVDM_TYPE_COT, payload, before_cot);
        nv_wait_cond(wait_ms, [&] { return reg(NV_PFALCON_FALCON_HWCFG2).with_base(falcon).read_bitfields()["riscv_br_priv_lockdown"]; }, 0, "");
    }

    // NV_FLCN_COT.kfsp_send_msg (ip.py:328-344), and around a COT message nv_init_helper's _kfsp_send_msg_flagged: the FSP's
    // queue registers read and logged first (NVIDIA sends only into an empty command queue, kfspPollForCanSend_GH100; tinygrad
    // does not check), then before_cot.
    void kfsp_send_msg(uint32_t nvmd, const std::vector<uint8_t>& payload, const std::function<void()>& before_cot = {}) {
        using namespace nv_regs;
        if (nvmd == nv::NVDM_TYPE_COT) {
            const uint32_t qh = reg(NV_PFSP_QUEUE_HEAD)[0].read(), qt = reg(NV_PFSP_QUEUE_TAIL)[0].read();
            const uint32_t mh = reg(NV_PFSP_MSGQ_HEAD)[0].read(), mt = reg(NV_PFSP_MSGQ_TAIL)[0].read();
            tg_log("before the COT message: FSP command queue head/tail 0x%x/0x%x, message queue head/tail 0x%x/0x%x (%s)", qh, qt, mh, mt,
                   qh == qt && mh == mt ? "both empty" : "NOT EMPTY");
            if (before_cot) before_cot();
        }
        // All single-packets go to seid 0
        const uint32_t mctp = (1u << 31) | (1u << 30), nvdm = 0x7eu | (0x10deu << 8) | (nvmd << 24);
        std::vector<uint8_t> buf(8);
        memcpy(buf.data(), &mctp, 4);
        memcpy(buf.data() + 4, &nvdm, 4);
        buf.insert(buf.end(), payload.begin(), payload.end());
        buf.resize(buf.size() + (4 - payload.size() % 4), 0);   // tinygrad's padding: 4 bytes when the payload is aligned already
        if (buf.size() >= 0x400)
            throw NVError("AssertionError", "FSP message too long: " + std::to_string(buf.size()) + " bytes, max 1024 bytes");

        reg(NV_PFSP_EMEMC)[0].write({{"offs", 0}, {"blk", 0}, {"aincw", 1}, {"aincr", 0}});
        for (size_t i = 0; i < buf.size(); i += 4) {
            uint32_t w;
            memcpy(&w, &buf[i], 4);
            reg(NV_PFSP_EMEMD)[0].write(w);
        }
        reg(NV_PFSP_QUEUE_TAIL)[0].write((uint32_t)buf.size() - 4);
        reg(NV_PFSP_QUEUE_HEAD)[0].write(0);

        // Waiting for a response
        nv_wait_cond_true(wait_ms, [&] { return reg(NV_PFSP_MSGQ_HEAD)[0].read() != reg(NV_PFSP_MSGQ_TAIL)[0].read(); },
                          "FSP didn't respond to message");

        reg(NV_PFSP_EMEMC)[0].write({{"offs", 0}, {"blk", 0}, {"aincw", 0}, {"aincr", 1}});
        reg(NV_PFSP_MSGQ_TAIL)[0].write(reg(NV_PFSP_MSGQ_HEAD)[0].read());
    }

    // nv_init_helper's NV_FLCN_COT.fini_hw (_cot_fini_hw_halt_wait, plan step B1): kgspTeardown_GH100 after the GSP unload,
    // a wait of up to 4 s for the GSP's RISC-V core to halt (kflcnWaitForHaltRiscv, 570.144 kernel_falcon_ga102.c:275-289),
    // "to allow ACR and GSP FMC to finish shutdown"; until it halts they may still use the boot structures in sysmem, so the
    // TinyGPU.app connection is closed only once halted is true. A PRI error (0xbadfxxxx) and an unreachable GPU (0xffffffff)
    // are not taken for a halt. Reads only; it is the unload's own last step on this chip, not an added teardown.
    void cot_fini_hw(NVFiniDiag& diag) {
        using namespace nv_regs;
        diag.teardown_ran = true;
        diag.halted = false;
        if (!diag.unload_ok) {
            diag.result("skipped: the GSP did not confirm its unload, so its RISC-V core is not polled");
            tg_log("teardown %s", diag.td_result.c_str());
            return;
        }
        auto cpuctl = reg(NV_PRISCV_RISCV_CPUCTL).with_base(falcon);
        const auto t0 = std::chrono::steady_clock::now();
        auto elapsed = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
        uint32_t val;
        int polls = 0;
        bool halted;
        while (true) {
            val = cpuctl.read(); polls += 1;
            if ((halted = val != 0xffffffff && val >> 16 != 0xbadf && cpuctl.decode(val)["halted"] == 1) || elapsed() >= cot_halt_timeout_s) break;
            sleep(0.001);
        }
        const double ms = elapsed() * 1e3;
        diag.riscv_cpuctl = val; diag.have_cpuctl = true;
        diag.halted = halted;
        diag.mailbox0_after_halt = reg(NV_PGSP_FALCON_MAILBOX0).read(); diag.have_mailbox0_after_halt = true;
        diag.wpr2_lo = reg(NV_PFB_PRI_MMU_WPR2_ADDR_LO).read();
        diag.wpr2_hi = reg(NV_PFB_PRI_MMU_WPR2_ADDR_HI).read();
        diag.have_wpr2 = true;
        diag.wpr2_down = diag.wpr2_hi == 0;
        diag.td_set("halt_wait_ms", std::to_string((long long)std::nearbyint(ms)));   // round(ms): half to even
        diag.td_set("polls", std::to_string(polls));
        char r[120];
        if (halted) snprintf(r, sizeof(r), "done: GSP RISC-V halted after %.0f ms", ms);
        else snprintf(r, sizeof(r), "failed: GSP RISC-V did not halt within %.0f s (RISCV_CPUCTL=0x%08x)", cot_halt_timeout_s, val);
        diag.result(r);
        diag.teardown_ok = halted && diag.wpr2_down;
        diag.have_outcome = true;
        tg_log("teardown %s (%d polls); MAILBOX0=0x%08x, WPR2_LO=0x%08x, WPR2_HI=0x%08x: %s%s", r, polls, diag.mailbox0_after_halt, diag.wpr2_lo,
               diag.wpr2_hi, diag.teardown_ok ? "the next boot needs no power cycle" : "power-cycle before the next boot",
               halted ? "" : "; the GSP may still be live, so the connection is held");
    }

private:
    NVBar0& dev_;
    uint32_t chip_id_;

    static uint32_t u32(uint64_t v) {   // struct.pack('<I') in NVDev.wreg raises for a value that does not fit
        if (v >> 32) throw NVError("error", "'I' format requires 0 <= number <= 4294967295");
        return (uint32_t)v;
    }
    static bool falcon_error(const NVError& e) {   // nv_init_helper's _FALCON_ERRORS: tinygrad's wait_cond timeouts, and asserts
        return e.type == "TimeoutError" || e.type == "RuntimeError" || e.type == "AssertionError";
    }
    // nv_init_helper's _tolerant_reset: tinygrad's reset; as in NVIDIA's kflcnReset_TU102 (kernel_falcon_tu102.c:175-189), a
    // core-select (BCR) timeout does not stop the teardown and FALCON_RM is still written (tinygrad writes it only once the
    // core select succeeds, ip.py:282-283)
    void tolerant_reset(uint32_t base, NVFiniDiag& diag, const char* name) {
        try { reset(base); }
        catch (const NVError& e) {
            if (e.type != "TimeoutError" || std::string(e.what()).find("RISCV core not booted") == std::string::npos) throw;
            diag.td_set(std::string(name) + "_bcr_timeout", "true");
            reg(nv_regs::NV_PFALCON_FALCON_RM).with_base(base).write(chip_id_);
            tg_log("teardown: %s reset: core select timed out (%s); FALCON_RM written, continuing, as NVIDIA does", name, e.what());
        }
    }
};

} // namespace tinygpu_device

#endif // LIBHMSBEAGLE_GPU_TINYGPUNVFALCON_H
