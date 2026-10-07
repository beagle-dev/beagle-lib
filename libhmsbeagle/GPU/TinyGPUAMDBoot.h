/*
 * TinyGPUAMDBoot.h -- TODO.md plan step A2c-A2f: tinygrad's AM driver (tinygrad/runtime/support/am/amdev.py and ip.py at
 * a9830e2b4) in C++, ported statement by statement, so that the plugin boots the AMD GPU with no Python. Only the branches
 * the RX 7900 XT takes are ported (GC 11.0.0, MP0 and MP1 13.0.0, SDMA 6.0.0, NBIO 4.3.0, MMHUB 3.0.0, OSSSYS 6.0.0, HDP
 * 6.0.0; TinyGPUAMDBootTables.h is generated for them): another IP set, a VF and a hive are refused before the boot proper,
 * after only the PCIe link-control write and the discovery reads tinygrad makes first (the plugin refuses a card whose PCI
 * device ID am::kChips lacks before sending it anything, TODO.md plan step N1). Each part is golden-tested against the
 * code it ports on fake_amd_device.py's card (tinygpu_tests/golden_amd_boot.py): the same requests to TinyGPU.app, byte for
 * byte, the same VRAM, the same results and errors.
 *
 * Requests: every read and write tinygrad makes, in its order, reads included (a page-table entry is one 8-byte BAR0 read
 * each time tinygrad indexes it, an IH entry is eight 4-byte reads); registers past BAR5 go through the RSMU window, as
 * AMDev.rreg/wreg. wait_cond polls as tinygrad's does, with no sleep; tinygrad's sleeps are kept.
 *
 * What BEAGLE adds sends nothing:
 *   - the IOVA fence (TinyGPUMemory.h's check_mapping): a system PTE may point only into a DMA segment TinyGPU.app gave
 *     this connection, since a stray device address faults the Mac's DART, which can panic macOS;
 *   - refuse_mode1: the boot stops (an error, before the write) where tinygrad would send the SMU's mode1 reset, which was
 *     never tried over TinyGPU (plan step A0's rule);
 *   - for the crash guard (plan step A2k): fini_state() and the AMDev it restores, which can only finalize, and whether
 *     fini saw every queue off (queues_off: each active compute queue's dequeue seen through).
 * Not ported: the VF mailbox and RLC gateway, hives and XGMI, the USB path, recover() (the plugin's runtime stops on a
 * fault instead), PMC/SQTT and the ACA bank dump (smu_13_0_0 has no PPSMC_MSG_QueryValidMcaCount).
 */

#ifndef LIBHMSBEAGLE_GPU_TINYGPUAMDBOOT_H
#define LIBHMSBEAGLE_GPU_TINYGPUAMDBOOT_H

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "libhmsbeagle/GPU/TinyGPUAMDBootTables.h"
#include "libhmsbeagle/GPU/TinyGPUAMDReg.h"
#include "libhmsbeagle/GPU/TinyGPUAMDTables.h"
#include "libhmsbeagle/GPU/TinyGPULog.h"
#include "libhmsbeagle/GPU/TinyGPUMemory.h"
#include "libhmsbeagle/GPU/TinyGPUTransport.h"

namespace tinygpu_device {
namespace amboot {

using am::AMKV;
using am::AMArgs;

inline uint64_t lo32(uint64_t x) { return x & 0xFFFFFFFFull; }   // helpers.lo32, hi32
inline uint64_t hi32(uint64_t x) { return x >> 32; }
inline int64_t getenv_int(const char* name, int64_t dflt) {   // helpers.getenv with an int default
    const char* v = getenv(name);
    return v && v[0] ? strtoll(v, nullptr, 0) : dflt;
}
inline double getenv_float(const char* name, double dflt) {
    const char* v = getenv(name);
    return v && v[0] ? strtod(v, nullptr) : dflt;
}
inline void sleep_s(double s) { std::this_thread::sleep_for(std::chrono::duration<double>(s)); }   // time.sleep
inline int64_t now_ms() {   // int(time.perf_counter() * 1000)
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
// helpers.wait_cond: polls cb() until it equals value (no sleep), else TimeoutError
template <class F> uint64_t wait_cond(F cb, uint64_t value = 1, int64_t timeout_ms = 10000, const std::string& msg = "") {
    int64_t start = now_ms();
    uint64_t val = 0;
    while (now_ms() - start < timeout_ms)
        if ((val = (uint64_t)cb()) == value) return val;
    throw TGPyError("TimeoutError", msg + ". Timed out after " + std::to_string(timeout_ms) + " ms, condition not met: " + std::to_string(val) +
                    " != " + std::to_string(value));
}
inline std::string hex(uint64_t v) { return tg_hex(v); }

enum Hwip { GC = am::GC_HWIP, HDP = am::HDP_HWIP, SDMA0 = am::SDMA0_HWIP, MMHUB = am::MMHUB_HWIP, NBIO = am::NBIO_HWIP,
            MP0 = am::MP0_HWIP, MP1 = am::MP1_HWIP, OSSSYS = am::OSSSYS_HWIP };
using Ver = std::array<int, 3>;

class AMDev;

// ── AMFirmware (amdev.py:25-119) ──────────────────────────────────────────────────────────────────────────────────────
// A firmware blob by its fetch_fw name: its bytes, or why not (TinyGPUFirmware.h's locator in the plugin).
using AMBlobLoader = std::function<std::string(const std::string& name, std::vector<uint8_t>& out)>;
struct AMFwDesc { std::vector<uint32_t> types; const uint8_t* p; size_t n; };   // desc(): (types, the blob's slice)

class AMFirmware {
public:
    std::map<uint32_t, std::vector<uint8_t>> sos_fw;   // fw_type -> ucode
    std::map<std::string, uint64_t> ucode_start;
    std::vector<AMFwDesc> descs;
    bool has_smu_psp_desc = false;
    AMFwDesc smu_psp_desc;

    AMFirmware(const std::map<int, Ver>& ip_ver, const AMBlobLoader& load) {
        auto fmt_ver = [&](int hwip) { const Ver& v = ip_ver.at(hwip); return std::to_string(v[0]) + "_" + std::to_string(v[1]) + "_" + std::to_string(v[2]); };
        // Load SOS firmware
        {
            const std::vector<uint8_t>& blob = load_fw("psp_" + fmt_ver(MP0) + "_sos.bin", load);
            const auto* chdr = (const am::struct_common_firmware_header*)blob.data();
            need(chdr->header_version_major == 2 && chdr->header_version_minor == 0, "psp sos header v2.0");
            const auto* sos_hdr = (const am::struct_psp_firmware_header_v2_0*)blob.data();
            for (uint32_t fw_i = 0; fw_i < sos_hdr->psp_fw_bin_count; ++fw_i) {
                const auto* d = (const am::struct_psp_fw_bin_desc*)((const uint8_t*)&sos_hdr->psp_fw_bin + fw_i * sizeof(am::struct_psp_fw_bin_desc));
                uint64_t start = d->offset_bytes + sos_hdr->header.ucode_array_offset_bytes;
                need(start + d->size_bytes <= blob.size(), "a sos component inside its blob");
                sos_fw[d->fw_type] = std::vector<uint8_t>(blob.begin() + start, blob.begin() + start + d->size_bytes);
            }
        }
        // SMU firmware: GC >= 11
        {
            const std::vector<uint8_t>& blob = load_fw("smu_" + fmt_ver(MP1) + ".bin", load);
            const auto* chdr = (const am::struct_common_firmware_header*)blob.data();
            need(chdr->header_version_major == 2 && chdr->header_version_minor == 1, "smc header v2.1");
            const auto* hdr = (const am::struct_smc_firmware_header_v2_1*)blob.data();
            smu_psp_desc = desc(blob, hdr->v1_0.header.ucode_array_offset_bytes, hdr->v1_0.header.ucode_size_bytes, {am::GFX_FW_TYPE_SMU});
            has_smu_psp_desc = true;
        }
        // SDMA firmware: header v2
        {
            const std::vector<uint8_t>& blob = load_fw("sdma_" + fmt_ver(SDMA0) + ".bin", load);
            const auto* chdr = (const am::struct_common_firmware_header*)blob.data();
            need(chdr->header_version_major == 2 && chdr->header_version_minor == 0, "sdma header v2.0");
            const auto* hdr = (const am::struct_sdma_firmware_header_v2_0*)blob.data();
            descs.push_back(desc(blob, hdr->ctl_ucode_offset, hdr->ctl_ucode_size_bytes, {am::GFX_FW_TYPE_SDMA_UCODE_TH1}));
            descs.push_back(desc(blob, hdr->header.ucode_array_offset_bytes, hdr->ctx_ucode_size_bytes, {am::GFX_FW_TYPE_SDMA_UCODE_TH0}));
        }
        // PFP, ME, MEC firmware: GC < 12, so MEC only, an RS64 (v2) header
        {
            const std::vector<uint8_t>& blob = load_fw("gc_" + fmt_ver(GC) + "_mec.bin", load);
            const auto* chdr = (const am::struct_common_firmware_header*)blob.data();
            need(chdr->header_version_major == 2 && chdr->header_version_minor == 0, "gfx header v2.0");
            const auto* hdr = (const am::struct_gfx_firmware_header_v2_0*)blob.data();
            descs.push_back(desc(blob, hdr->header.ucode_array_offset_bytes, hdr->ucode_size_bytes, {am::GFX_FW_TYPE_RS64_MEC}));
            descs.push_back(desc(blob, hdr->data_offset_bytes, hdr->data_size_bytes, {am::GFX_FW_TYPE_RS64_MEC_P0_STACK}));
            ucode_start["MEC"] = hdr->ucode_start_addr_lo | ((uint64_t)hdr->ucode_start_addr_hi << 32);
        }
        // IMU firmware: GC >= 11
        {
            const std::vector<uint8_t>& blob = load_fw("gc_" + fmt_ver(GC) + "_imu.bin", load);
            const auto* hdr = (const am::struct_imu_firmware_header_v1_0*)blob.data();
            uint64_t i_off = hdr->header.ucode_array_offset_bytes, i_sz = hdr->imu_iram_ucode_size_bytes, d_sz = hdr->imu_dram_ucode_size_bytes;
            descs.push_back(desc(blob, i_off, i_sz, {am::GFX_FW_TYPE_IMU_I}));
            descs.push_back(desc(blob, i_off + i_sz, d_sz, {am::GFX_FW_TYPE_IMU_D}));
        }
        // RLC firmware
        {
            const std::vector<uint8_t>& blob = load_fw("gc_" + fmt_ver(GC) + "_rlc.bin", load);
            const auto* hdr0 = (const am::struct_rlc_firmware_header_v2_0*)blob.data();
            const auto* hdr2 = (const am::struct_rlc_firmware_header_v2_2*)blob.data();
            const auto* hdr3 = (const am::struct_rlc_firmware_header_v2_3*)blob.data();
            need(hdr0->header.header_version_minor != 1, "an RLC header that is not v2.1 (its restore lists are not ported)");
            if (hdr0->header.header_version_minor >= 2) {
                descs.push_back(desc(blob, hdr2->rlc_iram_ucode_offset_bytes, hdr2->rlc_iram_ucode_size_bytes, {am::GFX_FW_TYPE_RLC_IRAM}));
                descs.push_back(desc(blob, hdr2->rlc_dram_ucode_offset_bytes, hdr2->rlc_dram_ucode_size_bytes, {am::GFX_FW_TYPE_RLC_DRAM_BOOT}));
            }
            if (hdr0->header.header_version_minor == 3) {
                descs.push_back(desc(blob, hdr3->rlcp_ucode_offset_bytes, hdr3->rlcp_ucode_size_bytes, {am::GFX_FW_TYPE_RLC_P}));
                descs.push_back(desc(blob, hdr3->rlcv_ucode_offset_bytes, hdr3->rlcv_ucode_size_bytes, {am::GFX_FW_TYPE_RLC_V}));
            }
            descs.push_back(desc(blob, hdr0->header.ucode_array_offset_bytes, hdr0->header.ucode_size_bytes, {am::GFX_FW_TYPE_RLC_G}));
        }
    }

private:
    std::map<std::string, std::vector<uint8_t>> blobs_;   // the loaded files, which the descs point into
    const std::vector<uint8_t>& load_fw(const std::string& name, const AMBlobLoader& load) {
        std::vector<uint8_t>& b = blobs_[name];
        std::string err = load(name, b);
        if (!err.empty()) throw TGPyError("RuntimeError", "firmware " + name + ": " + err);
        need(b.size() >= sizeof(am::struct_common_firmware_header), "a firmware header in " + name);
        return b;
    }
    static void need(bool ok, const std::string& what) {
        if (!ok) throw TGPyError("RuntimeError", "AMFirmware: expected " + what + " (the C++ boot ports this card's firmware headers only)");
    }
    static AMFwDesc desc(const std::vector<uint8_t>& blob, uint64_t off, uint64_t size, std::vector<uint32_t> types) {   // blob[off:off+size]
        if (off > blob.size()) off = blob.size();
        if (size > blob.size() - off) size = blob.size() - off;   // a memoryview slice past the end is cut there
        return AMFwDesc{std::move(types), blob.data() + off, (size_t)size};
    }
};

// ── AMPageTableEntry, AMMemoryManager (amdev.py:121-144) ──────────────────────────────────────────────────────────────
class AMPageTableEntry {
public:
    using Dev = AMDev;
    AMDev* adev = nullptr;
    uint64_t paddr = 0;
    int lv = 0;
    AMPageTableEntry() = default;
    AMPageTableEntry(AMDev* d, uint64_t paddr_, int lv_) : adev(d), paddr(paddr_), lv(lv_) {}
    void set_entry(uint64_t entry_id, uint64_t pa, bool table = false, bool uncached = false, TGAddrSpace aspace = TGAddrSpace::PHYS,
                   bool snooped = false, int64_t frag = 0, bool valid = true) const;
    uint64_t entry(uint64_t entry_id) const;
    bool valid(uint64_t entry_id) const { return (entry(entry_id) & am::AMDGPU_PTE_VALID) != 0; }
    uint64_t address(uint64_t entry_id) const;
    bool is_page(uint64_t entry_id) const;
    bool supports_huge_page(uint64_t) const { return lv >= (int)am::AMDGPU_VM_PDB2; }
};

class AMMemoryManager : public TGMemoryManager<AMPageTableEntry> {
public:
    using TGMemoryManager::TGMemoryManager;
    void on_range_mapped() override;   // Invalidate TLB after mappings.
    void check_mapping(const std::vector<std::pair<uint64_t, uint64_t>>& paddrs, TGAddrSpace aspace) override;
};

// ── the IP blocks (ip.py) ─────────────────────────────────────────────────────────────────────────────────────────────
struct AM_IP {
    AMDev& adev;
    explicit AM_IP(AMDev& a) : adev(a) {}
    virtual ~AM_IP() = default;
    virtual void init_sw() {}
    virtual void init_hw() {}
    virtual void fini_hw() {}
    virtual void set_clockgating_state() {}
    virtual const char* name() const = 0;
};

struct AM_SOC : AM_IP {
    using AM_IP::AM_IP;
    const char* name() const override { return "AM_SOC"; }
    std::vector<uint32_t> gfx_ih_clients;
    void init_sw() override { gfx_ih_clients = {am::SOC21_IH_CLIENTID_GRBM_CP, am::SOC21_IH_CLIENTID_GFX}; }
    void init_hw() override;
    void set_clockgating_state() override;
    void doorbell_enable(int port, uint64_t awid = 0, uint64_t awaddr_31_28_value = 0, uint64_t offset = 0, uint64_t size = 0);
    const char* ih_src_name(uint32_t client, uint32_t src) const;   // ih_srcs_names.get(client, {}).get(src, '')
};

struct AM_GMC : AM_IP {
    using AM_IP::AM_IP;
    const char* name() const override { return "AM_GMC"; }
    int vmhubs = 0;
    uint64_t xgmi_phys_id = 0, xgmi_max_region = 0, xgmi_seg_sz = 0, paddr_base = 0, fb_base = 0, fb_end = 0, mc_base = 0, vm_base = 0, vm_end = 0;
    bool trans_futher = false;
    uint64_t address_space_mask = 0, memscratch_xgmi_paddr = 0, dummy_page_xgmi_paddr = 0;
    std::map<std::string, bool> hub_initted;
    std::vector<int> mm_insts;
    void init_sw() override;
    void init_hw() override { init_hub("MM", mm_insts); }
    std::string pf_status_reg(const std::string& ip) const { return "reg" + ip + "VM_L2_PROTECTION_FAULT_STATUS"; }
    void flush_hdp();
    void flush_tlb(const std::string& ip, int vmid, uint64_t flush_type = 0);
    void enable_vm_addressing(const AMPageTableEntry& page_table, const std::string& ip, int vmid, int inst);
    void init_hub(const std::string& ip, const std::vector<int>& insts);
    uint64_t get_pte_flags(int pte_lv, bool is_table, int64_t frag, bool uncached, bool system, bool snooped, bool valid, uint64_t extra = 0) const;
    bool is_pte_huge_page(int, uint64_t pte) const { return (pte & am::AMDGPU_PDE_PTE) != 0; }
};

struct AM_SMU : AM_IP {
    using AM_IP::AM_IP;
    const char* name() const override { return "AM_SMU"; }
    uint64_t driver_table_paddr = 0;
    bool clocks_read = false;
    std::vector<std::pair<uint32_t, std::vector<uint64_t>>> clocks;   // read_clocks' dict, functools.cache'd
    void init_sw() override;
    void init_hw() override;
    bool is_smu_alive();
    void mode1_reset();
    const std::vector<std::pair<uint32_t, std::vector<uint64_t>>>& read_clocks(const std::vector<uint32_t>& clk_list);
    void set_clocks(bool none, int level);
    void set_power_limit(double watts);
    uint64_t send_msg(uint32_t msg, uint64_t param, bool read_back_arg = false, int64_t timeout = 10000, bool debug = false);
};

struct AM_GFX : AM_IP {
    using AM_IP::AM_IP;
    const char* name() const override { return "AM_GFX"; }
    int xccs = 0;
    std::vector<uint64_t> mqd_paddr, mqd_mc;
    int dequeue_unconfirmed = 0;   // BEAGLE's: the active queues the last dequeue_hqds did not see go inactive
    void init_sw() override;
    void init_hw() override;
    void fini_hw() override { dequeue_hqds(); }
    void reset_mec();
    uint64_t setup_ring(uint64_t ring_addr, uint64_t ring_size, uint64_t rptr_addr, uint64_t wptr_addr, uint64_t eop_addr, uint64_t eop_size,
                        int idx, bool aql);
    void set_clockgating_state() override;
    void grbm_select(uint64_t me = 0, uint64_t pipe = 0, uint64_t queue = 0, uint64_t vmid = 0, int inst = 0);
    void enable_mec();
    void config_mec();
    void dequeue_hqds();
};

struct AM_IH : AM_IP {
    using AM_IP::AM_IP;
    const char* name() const override { return "AM_IH"; }
    uint64_t ring_size = 0;
    struct Ring { uint64_t ring_vm, rwptr_vm; std::string suf; int ring_id; };
    std::vector<Ring> rings;
    uint64_t ring_view_paddr = 0;
    void init_sw() override;
    void init_hw() override;
    void drain();
    void interrupt_handler();
};

struct AM_SDMA : AM_IP {
    using AM_IP::AM_IP;
    const char* name() const override { return "AM_SDMA"; }
    std::vector<std::pair<std::string, int>> sdma_reginst;
    std::string sdma_name;
    void init_sw() override { sdma_reginst.clear(); sdma_name = "F32"; }   // SDMA < 7.0.0
    void init_hw() override;
    void fini_hw() override;
    uint64_t setup_ring(uint64_t ring_addr, uint64_t ring_size, uint64_t rptr_addr, uint64_t wptr_addr, int idx);
};

struct AM_PSP : AM_IP {
    using AM_IP::AM_IP;
    const char* name() const override { return "AM_PSP"; }
    std::string reg_pref;
    uint64_t msg1_paddr = 0, msg1_addr = 0, msg1_size = 0, cmd_paddr = 0, fence_paddr = 0, ring_size = 0, ring_paddr = 0;
    uint64_t max_tmr_size = 0, tmr_size = 0, tmr_paddr = 0;
    bool boot_time_tmr = false, autoload_tmr = true;
    void init_sw() override;
    void init_hw() override;
    bool is_sos_alive();
    void wait_for_bootloader();
    void prep_msg1(const uint8_t* data, size_t n);
    void bootloader_load_component(uint32_t fw, uint32_t compid);
    void tmr_init();
    void ring_create();
    am::struct_psp_gfx_cmd_resp ring_submit(const am::struct_psp_gfx_cmd_resp& cmd);
    void load_ip_fw_cmd(const AMFwDesc& d);
    am::struct_psp_gfx_cmd_resp tmr_load_cmd();
    am::struct_psp_gfx_cmd_resp load_toc_cmd(uint64_t toc_size);
    am::struct_psp_gfx_cmd_resp rlc_autoload_cmd();
};

// ── AMDev (amdev.py:146-415) ──────────────────────────────────────────────────────────────────────────────────────────
struct AMBootOptions {
    bool refuse_mode1 = true;   // stop where tinygrad would send the SMU's mode1 reset (plan step A0's rule)
};

// What fini() reads that the boot found (TODO.md plan step A2k): the crash guard (tinygpu_guard.cpp) rebuilds from it an
// AMDev that can only finalize, with no request to the GPU. Plain data: it follows the guard's setup on its socketpair.
struct AMFiniState {
    uint8_t discovery[10 << 10];   // the discovery table the boot read (bhdr): the register bases and IP versions
    uint64_t vram_bytes, mmio_bytes, vram_size, ih_ring_paddr, ih_ring_size;   // vram_bytes, mmio_bytes: BAR0's and BAR5's sizes
    uint32_t xccs, clocks_read, nclocks, nsdma;
    struct { uint32_t clk, n; uint64_t vals[16]; } clocks[4];   // AM_SMU's read_clocks cache
    struct { char reg[24]; int32_t inst; } sdma[2];             // AM_SDMA's sdma_reginst
};

class AMDev {
public:
    static constexpr uint32_t Version = 0xA0000008;
    TGTransport& t;
    std::string devfmt = "usb4";
    AMBlobLoader loader;
    AMBootOptions opts;
    uint64_t vram_bytes = 0, doorbell_bytes = 0, mmio_bytes = 0;   // the BARs' sizes (map_bar(0), (2), (5))
    bool is_vf = false;
    uint64_t vram_size = 0;
    bool large_bar = false;
    std::vector<uint8_t> bhdr;   // the discovery table (10 KiB)
    std::map<int, std::map<int, std::vector<uint64_t>>> regs_offset;   // hwip -> instance -> bases
    std::map<int, Ver> ip_ver;
    std::map<int, std::set<int>> harvested;
    std::vector<uint8_t> gc_info;   // the versioned gc_info struct's bytes
    uint64_t reserved_vram_size = 0;
    std::vector<int> aids;
    bool is_booting = false, smi_dev = false, is_err_state = false, partial_boot = false;
    bool queues_off = false;   // BEAGLE's (plan step A2k): fini() saw every queue off (the crash guard's hold rule)
    std::unique_ptr<AMMemoryManager> mm;
    TLSFAllocator va_allocator{1ull << 44, 0x200000000000ull};   // AMMemoryManager.va_allocator: global for all devices
    std::unique_ptr<AMFirmware> fw;
    std::unique_ptr<AM_SOC> soc;
    std::unique_ptr<AM_GMC> gmc;
    std::unique_ptr<AM_IH> ih;
    std::unique_ptr<AM_PSP> psp;
    std::unique_ptr<AM_SMU> smu;
    std::unique_ptr<AM_GFX> gfx;
    std::unique_ptr<AM_SDMA> sdma;

    // AMDev.__init__: the whole boot, as PCIIfaceBase.__init__ calls it after its RESIZE_BAR. TGPyError on failure.
    AMDev(TGTransport& t_, AMBlobLoader loader_, AMBootOptions opts_ = {});
    // The crash guard's (plan step A2k): an AMDev that can only run fini(), from a booted one's fini_state(), sending nothing.
    // The transport must know BARs 0 and 5 already (seed_bar).
    AMDev(TGTransport& t_, const AMFiniState& s);
    // What fini() needs, as the boot left it, with the SDMA queue AMDDevice.__init__ sets up next: called before that setup,
    // so that a guard disables that queue whether or not it was set up yet
    void fini_state(AMFiniState& s) const;
    void fini();

    am::AMRegister<AMDev> reg(const std::string& name) { return am::AMRegister<AMDev>(this, name.c_str()); }
    bool has_reg(const std::string& name) const { return am::has_reg(name.c_str()); }
    uint32_t base(int hwip, int inst, int seg) const {   // AMDReg's bases[inst][segment]
        auto h = regs_offset.find(hwip);
        if (h == regs_offset.end()) throw am::AMRegError("no discovered bases for IP " + std::to_string(hwip));
        auto i = h->second.find(inst);
        if (i == h->second.end()) throw am::AMRegError("IP " + std::to_string(hwip) + " has no instance " + std::to_string(inst));   // the addr dict's KeyError
        if (seg < 0 || (size_t)seg >= i->second.size()) throw am::AMRegError("IP " + std::to_string(hwip) + " has no segment " + std::to_string(seg));
        return (uint32_t)i->second[seg];
    }
    uint64_t paddr2mc(uint64_t paddr) const { return gmc->mc_base + paddr; }
    uint64_t paddr2xgmi(uint64_t paddr) const { return gmc->paddr_base + paddr; }
    uint64_t xgmi2paddr(uint64_t xgmi_paddr) const { return xgmi_paddr - gmc->paddr_base; }
    bool is_hive() const { return gmc->xgmi_seg_sz > 0 && gmc->xgmi_max_region > 0; }

    uint32_t rreg(uint32_t reg, int inst = 0, bool direct = false);
    void wreg(uint32_t reg, uint32_t val, int inst = 0, bool direct = false);
    void wreg_pair(const std::string& reg_base, const std::string& lo_suffix, const std::string& hi_suffix, uint64_t val, int inst = 0);
    uint32_t indirect_rreg(uint32_t reg);
    void indirect_wreg(uint32_t reg, uint32_t val);

    // BAR0 (self.vram) and BAR5 (self.mmio) as RemoteMMIOInterface makes them requests; RuntimeError on a transport failure
    void vram_write(uint64_t off, const void* data, uint64_t n);
    void vram_read(uint64_t off, void* out, uint64_t n);
    uint64_t vram_q(uint64_t off) { uint64_t v = 0; vram_read(off, &v, 8); return v; }
    void vram_set_q(uint64_t off, uint64_t v) { vram_write(off, &v, 8); }
    uint32_t vram_i(uint64_t off) { uint32_t v = 0; vram_read(off, &v, 4); return v; }
    void vram_zero(uint64_t paddr, uint64_t size) { std::vector<uint8_t> z(size); vram_write(paddr, z.data(), size); }
    // tinygrad's DEBUG >= 2 lines ("am usb4: ..."), always in BEAGLE's log, and on stderr at DEBUG >= 2 as tinygrad prints them
    // (the hardware scripts read the boot and any reset from them)
    void log(const std::string& s) {
        static const int debug = (int)getenv_int("DEBUG", 0);
        tg_log("am %s: %s", devfmt.c_str(), s.c_str());
        if (debug >= 2) fprintf(stderr, "am %s: %s\n", devfmt.c_str(), s.c_str());
    }
    void print(const std::string& s) { tg_log("am %s: %s", devfmt.c_str(), s.c_str()); fprintf(stderr, "am %s: %s\n", devfmt.c_str(), s.c_str()); }   // tinygrad's unconditional prints

private:
    void cfg_read(uint64_t off, uint64_t size, uint64_t& v);
    void cfg_write_flush(uint64_t off, uint64_t value, uint64_t size);
    void disable_aspm();
    std::vector<uint8_t> read_vram(uint64_t addr, uint64_t size);
    void run_discovery();
    void parse_discovery();
    void build_regs();
    void init_sw();
    void init_hw(const std::vector<AM_IP*>& blocks);
};

// ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
// AMPageTableEntry
inline uint64_t AMPageTableEntry::entry(uint64_t entry_id) const { return adev->vram_q(paddr + 8 * entry_id); }
inline void AMPageTableEntry::set_entry(uint64_t entry_id, uint64_t pa, bool table, bool uncached, TGAddrSpace aspace, bool snooped, int64_t frag,
                                        bool valid) const {
    bool is_sys = aspace == TGAddrSpace::SYS;
    if (aspace == TGAddrSpace::PHYS) pa = adev->paddr2xgmi(pa);
    if ((pa & adev->gmc->address_space_mask) != pa) throw TGPyError("AssertionError", "Invalid physical address " + hex(pa));
    adev->vram_set_q(paddr + 8 * entry_id, adev->gmc->get_pte_flags(lv, table, frag, uncached, is_sys, snooped, valid) | (pa & 0x0000FFFFFFFFF000ull));
}
inline uint64_t AMPageTableEntry::address(uint64_t entry_id) const {
    if ((entry(entry_id) & am::AMDGPU_PTE_SYSTEM) != 0) throw TGPyError("AssertionError", "should not be system address");
    return adev->xgmi2paddr(entry(entry_id) & 0x0000FFFFFFFFF000ull);
}
inline bool AMPageTableEntry::is_page(uint64_t entry_id) const {
    return lv == (int)am::AMDGPU_VM_PTB || adev->gmc->is_pte_huge_page(lv, entry(entry_id));
}

// AMMemoryManager
inline void AMMemoryManager::on_range_mapped() {
    dev->gmc->flush_tlb("GC", 0);
    dev->gmc->flush_tlb("MM", 0);
}
inline void AMMemoryManager::check_mapping(const std::vector<std::pair<uint64_t, uint64_t>>& paddrs, TGAddrSpace aspace) {
    if (aspace != TGAddrSpace::SYS) return;
    for (const auto& p : paddrs)
        if (!dev->t.iova_known(p.first, p.second))
            throw TGPyError("RuntimeError", "the IOVA fence: a system mapping of " + hex(p.second) + " bytes at " + hex(p.first) +
                            ", which no DMA segment of this connection holds (a stray device address faults the Mac's DART)");
}

// ── AMDev ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
inline void AMDev::vram_write(uint64_t off, const void* data, uint64_t n) {
    std::string err;
    if (!t.bulk_write(0, off, data, n, err)) throw TGPyError("RuntimeError", err);
}
inline void AMDev::vram_read(uint64_t off, void* out, uint64_t n) {
    std::string err;
    if (!t.bulk_read(0, off, out, n, err)) throw TGPyError("RuntimeError", err);
}
inline void AMDev::cfg_read(uint64_t off, uint64_t size, uint64_t& v) {
    std::string err;
    if (!t.read_config(off, size, v, err)) throw TGPyError("RuntimeError", err);
}
inline void AMDev::cfg_write_flush(uint64_t off, uint64_t value, uint64_t size) {
    std::string err;
    if (!t.write_config_flush(off, value, size, err)) throw TGPyError("RuntimeError", err);
}

inline uint32_t AMDev::rreg(uint32_t reg, int, bool) {   // no RLC-gated ranges off a VF
    if (reg >= mmio_bytes / 4) return indirect_rreg(reg);
    uint32_t v = 0;
    std::string err;
    if (!t.bulk_read(5, (uint64_t)reg * 4, &v, 4, err)) throw TGPyError("RuntimeError", err);
    return v;
}
inline void AMDev::wreg(uint32_t reg, uint32_t val, int, bool) {
    if (reg >= mmio_bytes / 4) return indirect_wreg(reg, val);
    std::string err;
    if (!t.bulk_write(5, (uint64_t)reg * 4, &val, 4, err)) throw TGPyError("RuntimeError", err);
}
inline void AMDev::wreg_pair(const std::string& reg_base, const std::string& lo_suffix, const std::string& hi_suffix, uint64_t val, int inst) {
    reg(reg_base + lo_suffix).write(lo32(val), {}, inst);
    reg(reg_base + hi_suffix).write(hi32(val), {}, inst);
}
inline uint32_t AMDev::indirect_rreg(uint32_t r) {
    reg("regBIF_BX_PF0_RSMU_INDEX").write((uint64_t)r * 4);
    return reg("regBIF_BX_PF0_RSMU_DATA").read();
}
inline void AMDev::indirect_wreg(uint32_t r, uint32_t val) {
    reg("regBIF_BX_PF0_RSMU_INDEX").write((uint64_t)r * 4);
    reg("regBIF_BX_PF0_RSMU_DATA").write(val);
}

inline std::vector<uint8_t> AMDev::read_vram(uint64_t addr, uint64_t size) {
    if (addr % 4 || size % 4) throw TGPyError("AssertionError", "Invalid address " + hex(addr) + " or size " + hex(size));
    std::vector<uint8_t> res(size);
    for (uint64_t caddr = addr; caddr < addr + size; caddr += 4) {
        wreg(0x06, (uint32_t)(caddr >> 31));
        wreg(0x00, (uint32_t)((caddr & 0x7FFFFFFF) | 0x80000000));
        uint32_t v = rreg(0x01);
        memcpy(&res[caddr - addr], &v, 4);
    }
    return res;
}

inline void AMDev::disable_aspm() {
    // L1 across retimers makes reads oscillate to 0xffffffff; power on defaults it enabled. Clearing the GPU endpoint
    // alone suffices: L1 only engages when both ends of the link enable it.
    uint64_t v;
    cfg_read(0x34, 1, v);
    uint64_t cap = v & 0xfc;
    std::set<uint64_t> seen;   // bound the walk: a dead link can return 0xff pointers forever
    while (cap && !seen.count(cap)) {
        cfg_read(cap, 1, v);
        if (v == 0x10) break;
        seen.insert(cap);
        cfg_read(cap + 1, 1, v);
        cap = v & 0xfc;
    }
    if (cap && !seen.count(cap)) {   // PCIe cap lnkctl
        cfg_read(cap + 0x10, 2, v);
        cfg_write_flush(cap + 0x10, v & ~3ull, 2);
    }
}

inline void AMDev::run_discovery() {
    // NOTE: Fixed register to query memory size without known ip bases to find the discovery table.
    //       The table is located at the end of VRAM - 64KB and is 10KB in size.
    const uint32_t mmRCC_CONFIG_MEMSIZE = 0xde3;
    vram_size = (uint64_t)rreg(mmRCC_CONFIG_MEMSIZE) << 20;
    large_bar = vram_bytes >= vram_size;
    // BEAGLE's (TODO.md plan step N1), as NV refuses a BAR1 but 256 MiB (TinyGPUNVBoot.h): another BAR0, or one as large as
    // VRAM, takes paths no fake has run (am_iface_alloc keys on 256 MiB, TinyGPUAMDDevice.h). Refused before the discovery's
    // first index write; it reads nothing.
    if (vram_bytes != (256ull << 20) || large_bar)
        throw TGPyError("BarLayoutError", "BAR0 is " + std::to_string(vram_bytes >> 20) + " MiB with " + std::to_string(vram_size >> 20) +
                        " MiB of VRAM (large_bar=" + (large_bar ? "True" : "False") + "): BEAGLE supports only a 256 MiB BAR0 smaller than VRAM "
                        "(TODO.md plan step N1). Only the PCIe link control was written; nothing reached the card's VRAM or registers.");
    uint64_t tmr_offset = vram_size - (64 << 10), tmr_size = 10 << 10;
    if (large_bar) { bhdr.resize(tmr_size); vram_read(tmr_offset, bhdr.data(), tmr_size); }
    else bhdr = read_vram(tmr_offset, tmr_size);
    parse_discovery();
}

inline void AMDev::parse_discovery() {   // the rest of _run_discovery, on bhdr (the crash guard's restore parses the boot's copy)
    auto at = [&](uint64_t off, size_t n) -> const uint8_t* {
        if (off + n > bhdr.size()) throw TGPyError("ValueError", "a discovery table structure past the table's 10 KiB");
        return bhdr.data() + off;
    };
    const auto* b = (const am::struct_binary_header*)at(0, sizeof(am::struct_binary_header));
    const auto* ihdr = (const am::struct_ip_discovery_header*)at(b->table_list[am::IP_DISCOVERY].offset, sizeof(am::struct_ip_discovery_header));
    if (b->binary_signature != am::BINARY_SIGNATURE || ihdr->signature != am::DISCOVERY_TABLE_SIGNATURE)
        throw TGPyError("AssertionError", "discovery signatures mismatch");
    for (uint32_t num_die = 0; num_die < ihdr->num_dies; ++num_die) {
        const auto* dhdr = (const am::struct_die_header*)at(ihdr->die_info[num_die].die_offset, sizeof(am::struct_die_header));
        uint64_t ip_offset = sizeof(am::struct_die_header) + ihdr->die_info[num_die].die_offset;
        for (uint32_t k = 0; k < dhdr->num_ips; ++k) {
            const auto* ip = (const am::struct_ip_v4*)at(ip_offset, sizeof(am::struct_ip_v4));
            const bool b64 = ihdr->base_addr_64_bit();
            std::vector<uint64_t> ba;
            for (uint32_t j = 0; j < ip->num_base_address; ++j) {
                if (b64) { uint64_t x; memcpy(&x, at(ip_offset + 8 + 8 * j, 8), 8); ba.push_back(x); }
                else { uint32_t x; memcpy(&x, at(ip_offset + 8 + 4 * j, 4), 4); ba.push_back(x); }
            }
            for (uint32_t hw_ip = 1; hw_ip < am::MAX_HWIP; ++hw_ip)
                if (am::hw_id_mapped[hw_ip] && am::hw_id_map[hw_ip] == ip->hw_id) {
                    regs_offset[hw_ip][ip->instance_number] = ba;
                    ip_ver[hw_ip] = Ver{ip->major, ip->minor, ip->revision};
                }
            ip_offset += 8 + (b64 ? 8 : 4) * ip->num_base_address;
        }
    }
    // HARV(EST) table: harvested instances must be excluded (like amdgpu_discovery_harvest_ip)
    // layout: u32 signature, u16 version, u16 size, then 32 entries of {hw_id:u16, inst:u8, rsv:u8}
    uint64_t harv_off = b->table_list[am::HARVEST_INFO].offset;
    if (harv_off != 0) {
        const uint32_t* blob = (const uint32_t*)at(harv_off, 8 + 32 * 4);
        if (blob[0] == am::HARVEST_TABLE_SIGNATURE)
            for (int e = 2; e < 2 + 32; ++e) {
                uint32_t ent = blob[e];
                for (const am::HwIdIp& m : am::inv_hw_id)
                    if (m.hw_id == (ent & 0xffff)) harvested[(int)m.hw_ip].insert((ent >> 16) & 0xff);
            }
    }
    const auto* gc = (const am::struct_gc_info_v1_0*)at(b->table_list[am::GC].offset, sizeof(am::struct_gc_info_v1_0));
    if (gc->header.version_major != 1 || gc->header.version_minor != 2)
        throw TGPyError("RuntimeError", "gc_info v" + std::to_string(gc->header.version_major) + "." + std::to_string(gc->header.version_minor) +
                        ": the C++ boot has this card's v1.2 only");
    gc_info.assign(at(b->table_list[am::GC].offset, sizeof(am::struct_gc_info_v1_2)),
                   at(b->table_list[am::GC].offset, sizeof(am::struct_gc_info_v1_2)) + sizeof(am::struct_gc_info_v1_2));
    reserved_vram_size = 64 << 20;   // not gc 9.4/9.5
}

inline void AMDev::build_regs() {
    // the register tables are generated for this card's IP versions: anything else is refused here, before a write
    struct { int hwip; const uint8_t* v; const char* name; } want[] = {
        {GC, am::kIP_GC_HWIP, "GC"}, {MP0, am::kIP_MP0_HWIP, "MP0"}, {MP1, am::kIP_MP1_HWIP, "MP1"}, {SDMA0, am::kIP_SDMA0_HWIP, "SDMA0"},
        {NBIO, am::kIP_NBIO_HWIP, "NBIO"}, {MMHUB, am::kIP_MMHUB_HWIP, "MMHUB"}, {OSSSYS, am::kIP_OSSSYS_HWIP, "OSSSYS"}, {HDP, am::kIP_HDP_HWIP, "HDP"}};
    for (auto& w : want) {
        auto it = ip_ver.find(w.hwip);
        Ver v = it == ip_ver.end() ? Ver{-1, -1, -1} : it->second;
        if (v != Ver{w.v[0], w.v[1], w.v[2]})
            throw TGPyError("RuntimeError", std::string("the C++ AM boot is for the RX 7900 XT's IP versions; this card's ") + w.name + " is " +
                            std::to_string(v[0]) + "." + std::to_string(v[1]) + "." + std::to_string(v[2]));
    }
    // Live AIDs like the kernel: 4 SDMAs per AID; the AID lives iff its group's alive-mask is 0xf/0x3/0xc.
    std::set<int> live_sdma;
    int max_aid = 0;
    for (auto& [k, _] : regs_offset[SDMA0]) {
        if (!harvested[SDMA0].count(k)) live_sdma.insert(k);
        max_aid = std::max(max_aid, k >> 2);
    }
    aids = {0};
    for (int aid = 1; aid <= max_aid; ++aid) {
        int m = 0;
        for (int i : live_sdma) if (i >> 2 == aid) m += 1 << (i & 3);
        if (m == 0xf || m == 0x3 || m == 0xc) aids.push_back(aid);
    }
}

inline void AMDev::init_sw() {
    smi_dev = false;
    is_err_state = false;
    // Memory manager & firmware
    std::vector<std::pair<uint64_t, uint64_t>> palloc_ranges;
    for (int i = 9 * (3 - (int)am::AMDGPU_VM_PDB2); i >= 0; --i) palloc_ranges.push_back({1ull << (i + 12), i >= 9 ? (2ull << 20) : 0x1000});
    mm = std::make_unique<AMMemoryManager>(this, vram_size - reserved_vram_size, 32ull << 20, 48, std::vector<uint64_t>{12, 21, 30, 39},
                                           va_allocator.base, palloc_ranges, (int)am::AMDGPU_VM_PDB2, !large_bar);
    mm->va_allocator = &va_allocator;
    mm->palloc_zero_limit = ~0ull;   // AMD's daemon does not patch palloc
    fw = std::make_unique<AMFirmware>(ip_ver, loader);
    // Initialize IP blocks
    soc = std::make_unique<AM_SOC>(*this);
    gmc = std::make_unique<AM_GMC>(*this);
    ih = std::make_unique<AM_IH>(*this);
    psp = std::make_unique<AM_PSP>(*this);
    smu = std::make_unique<AM_SMU>(*this);
    gfx = std::make_unique<AM_GFX>(*this);
    sdma = std::make_unique<AM_SDMA>(*this);
    // Init sw for all IP blocks
    for (AM_IP* ip : std::vector<AM_IP*>{soc.get(), gmc.get(), ih.get(), psp.get(), smu.get(), gfx.get(), sdma.get()}) ip->init_sw();
}

inline void AMDev::init_hw(const std::vector<AM_IP*>& blocks) {
    for (AM_IP* ip : blocks) {
        ip->init_hw();
        log(std::string(ip->name()) + " initialized");
    }
}

inline AMDev::AMDev(TGTransport& t_, AMBlobLoader loader_, AMBootOptions opts_) : t(t_), loader(std::move(loader_)), opts(opts_) {
    std::string err;
    disable_aspm();
    uint64_t a;
    if (!t.bar_info(0, a, vram_bytes, err) || !t.bar_info(2, a, doorbell_bytes, err) || !t.bar_info(5, a, mmio_bytes, err)) throw TGPyError("RuntimeError", err);
    // VF related
    is_vf = (rreg(am::mmRCC_IOV_FUNC_IDENTIFIER) & 1) != 0;
    if (is_vf) throw TGPyError("RuntimeError", "a virtual function: the C++ AM boot ports the PF's boot only");
    run_discovery();
    build_regs();

    // AM boot Process: see amdev.py:172-181
    is_booting = true;   // During boot only boot memory can be allocated. This flag is to validate this.
    init_sw();

    partial_boot = reg("regSCRATCH_REG7").read() == Version && getenv_int("AM_RESET", 0) != 1;
    if (partial_boot && (reg("regSCRATCH_REG6").read() != 0 || reg(gmc->pf_status_reg("GC")).read() != 0)) {
        log("Malformed state. Issuing a full reset.");
        partial_boot = false;
    }

    // Init hw for IP blocks where it is needed
    if (!partial_boot) {
        if (psp->is_sos_alive() && smu->is_smu_alive()) {
            if (opts.refuse_mode1)
                throw TGPyError("RuntimeError", "the GPU needs an SMU mode1 reset (an earlier session did not finalize, or another driver ran it), "
                                "which BEAGLE never sends over TinyGPU: power-cycle the eGPU (unplug it, then plug it back in) and retry");
            uint64_t cmd;
            cfg_read(am::pci::PCI_COMMAND, 2, cmd);
            cfg_write_flush(am::pci::PCI_COMMAND, cmd & ~(uint64_t)am::pci::PCI_COMMAND_MASTER, 2);
            if (is_hive()) throw TGPyError("RuntimeError", "Malformed state. Use extra/amdpci/hive_reset.py to reset the hive");
            smu->mode1_reset();
        }
        uint64_t cmd;
        cfg_read(am::pci::PCI_COMMAND, 2, cmd);
        cfg_write_flush(am::pci::PCI_COMMAND, cmd | am::pci::PCI_COMMAND_MASTER, 2);
        init_hw({soc.get(), gmc.get(), ih.get(), psp.get(), smu.get()});
    }

    // Booting done
    is_booting = false;

    // Re-initialize main blocks
    init_hw({gfx.get(), sdma.get()});

    double max_power = getenv_float("AM_POWER_LIMIT", 0.0);
    if (max_power > 0) {
        smu->set_power_limit(max_power);
        smu->set_clocks(true, 0);
    } else smu->set_clocks(false, -1);   // last level, max perf.
    for (AM_IP* ip : std::vector<AM_IP*>{soc.get(), gfx.get()}) ip->set_clockgating_state();
    reg("regSCRATCH_REG7").write(Version);
    reg("regSCRATCH_REG6").write(1);   // set initialized state.
    log("boot done");
}

inline AMDev::AMDev(TGTransport& t_, const AMFiniState& s) : t(t_) {
    if (s.nclocks > 4 || s.nsdma > 2) throw TGPyError("RuntimeError", "a malformed AMD fini state");
    vram_bytes = s.vram_bytes;
    mmio_bytes = s.mmio_bytes;
    vram_size = s.vram_size;
    bhdr.assign(s.discovery, s.discovery + sizeof(s.discovery));
    parse_discovery();
    build_regs();
    soc = std::make_unique<AM_SOC>(*this);
    gmc = std::make_unique<AM_GMC>(*this);
    ih = std::make_unique<AM_IH>(*this);
    smu = std::make_unique<AM_SMU>(*this);
    gfx = std::make_unique<AM_GFX>(*this);
    sdma = std::make_unique<AM_SDMA>(*this);
    soc->init_sw();    // the IH's gfx clients: constants
    sdma->init_sw();   // its name, then the boot's queues
    for (uint32_t i = 0; i < s.nsdma; ++i) sdma->sdma_reginst.push_back({std::string(s.sdma[i].reg, strnlen(s.sdma[i].reg, sizeof(s.sdma[i].reg))), s.sdma[i].inst});
    ih->ring_size = s.ih_ring_size;
    ih->rings = {{s.ih_ring_paddr, 0, "", 0}};   // fini reads ring 0 only
    ih->ring_view_paddr = s.ih_ring_paddr;
    gfx->xccs = (int)s.xccs;
    smu->clocks_read = s.clocks_read != 0;
    for (uint32_t i = 0; i < s.nclocks; ++i) {
        if (s.clocks[i].n > 16) throw TGPyError("RuntimeError", "a malformed AMD fini state");
        smu->clocks.push_back({s.clocks[i].clk, std::vector<uint64_t>(s.clocks[i].vals, s.clocks[i].vals + s.clocks[i].n)});
    }
}

inline void AMDev::fini_state(AMFiniState& s) const {
    s = AMFiniState{};
    if (bhdr.size() != sizeof(s.discovery) || smu->clocks.size() > 4) throw TGPyError("RuntimeError", "the boot's state does not fit an AMD fini state");
    memcpy(s.discovery, bhdr.data(), sizeof(s.discovery));
    s.vram_bytes = vram_bytes;
    s.mmio_bytes = mmio_bytes;
    s.vram_size = vram_size;
    s.ih_ring_paddr = ih->ring_view_paddr;
    s.ih_ring_size = ih->ring_size;
    s.xccs = (uint32_t)gfx->xccs;
    s.clocks_read = smu->clocks_read ? 1 : 0;
    s.nclocks = (uint32_t)smu->clocks.size();
    for (uint32_t i = 0; i < s.nclocks; ++i) {
        const auto& [clk, vals] = smu->clocks[i];
        if (vals.size() > 16) throw TGPyError("RuntimeError", "the boot's state does not fit an AMD fini state");
        s.clocks[i].clk = clk;
        s.clocks[i].n = (uint32_t)vals.size();
        std::copy(vals.begin(), vals.end(), s.clocks[i].vals);
    }
    s.nsdma = 1;   // AMDDevice.__init__'s sdma_queue(0): AM_SDMA.setup_ring's reginst for idx 0
    snprintf(s.sdma[0].reg, sizeof(s.sdma[0].reg), "regSDMA0_QUEUE0");
    s.sdma[0].inst = 0;
}

inline void AMDev::fini() {
    log("Finalizing");
    for (AM_IP* ip : std::vector<AM_IP*>{sdma.get(), gfx.get()}) ip->fini_hw();
    queues_off = gfx->dequeue_unconfirmed == 0;   // BEAGLE's: SDMA's queues are disabled and the engine reset by now
    smu->set_clocks(false, 0);
    ih->interrupt_handler();
    reg("regSCRATCH_REG6").write(is_err_state ? 1 : 0);   // set finalized state.
}

// ── AM_SOC (ip.py:15-50) ──────────────────────────────────────────────────────────────────────────────────────────────
inline const char* AM_SOC::ih_src_name(uint32_t client, uint32_t src) const {
    bool gfx = false;
    for (uint32_t c : gfx_ih_clients) gfx |= c == client;
    if (!gfx) return "";
    for (const amdt::IHName& s : amdt::IH_GFX11_SRCS) if (s.id == src) return s.name;
    return "";
}
inline void AM_SOC::init_hw() {
    adev.reg("regRCC_DEV0_EPF2_STRAP2").update({{"strap_no_soft_reset_dev0_f2", 0x0}});
    adev.reg("regRCC_DEV0_EPF0_RCC_DOORBELL_APER_EN").write(0x1);
}
inline void AM_SOC::set_clockgating_state() {   // HDP >= 5.2.1
    adev.reg("regHDP_MEM_POWER_CTRL").update({{"atomic_mem_power_ctrl_en", 1}, {"atomic_mem_power_ds_en", 1}});
}
inline void AM_SOC::doorbell_enable(int port, uint64_t awid, uint64_t awaddr_31_28_value, uint64_t offset, uint64_t size) {
    const std::string p = std::to_string(port);
    auto reg = adev.reg("regS2A_DOORBELL_ENTRY_" + p + "_CTRL");   // GC < 12
    const std::string f = "s2a_doorbell_port" + p;
    const std::string en = f + "_enable", aw = f + "_awid", rs = f + "_range_size", av = f + "_awaddr_31_28_value", ro = f + "_range_offset";
    uint64_t val = reg.encode({{en.c_str(), 1}, {aw.c_str(), awid}, {rs.c_str(), size}, {av.c_str(), awaddr_31_28_value}, {ro.c_str(), offset}});
    reg.write(val);   // NBIO not 7.9
}

// ── AM_GMC (ip.py:52-192) ─────────────────────────────────────────────────────────────────────────────────────────────
inline void AM_GMC::init_sw() {
    vmhubs = (int)adev.regs_offset[MMHUB].size();
    xgmi_phys_id = xgmi_max_region = 0;
    if (adev.has_reg("regGCMC_VM_XGMI_LFB_CNTL")) throw TGPyError("RuntimeError", "an XGMI part: not ported");
    xgmi_seg_sz = 0;
    paddr_base = xgmi_phys_id * xgmi_seg_sz;
    fb_base = (uint64_t)(adev.reg("regMMMC_VM_FB_LOCATION_BASE").read() & 0xFFFFFF) << 24;
    fb_end = (uint64_t)(adev.reg("regMMMC_VM_FB_LOCATION_TOP").read() & 0xFFFFFF) << 24;
    // Memory controller aperture
    mc_base = fb_base + paddr_base;
    // VM aperture
    vm_base = adev.mm->va_base;
    vm_end = std::min<uint64_t>(vm_base + (1ull << adev.mm->va_bits) - 1, 0x7fffffffffffull);
    trans_futher = false;   // GC >= 10
    address_space_mask = (1ull << 44) - 1;   // not mi3xx
    memscratch_xgmi_paddr = adev.paddr2xgmi(adev.mm->palloc(0x1000, 0x1000, false, true));
    dummy_page_xgmi_paddr = adev.paddr2xgmi(adev.mm->palloc(0x1000, 0x1000, false, true));
    // MM hub is inited before any tlb flushes and is still valid during partial_boot, so set it to true
    hub_initted = {{"MM", true}, {"GC", false}};
    mm_insts.clear();
    for (int i = 0; i < vmhubs; ++i) mm_insts.push_back(i);   // not NBIO 7.9
}
inline void AM_GMC::flush_hdp() {
    adev.wreg(adev.reg("regBIF_BX0_REMAP_HDP_MEM_FLUSH_CNTL").read() / 4, 0x0);
}
inline void AM_GMC::flush_tlb(const std::string& ip, int vmid, uint64_t flush_type) {
    flush_hdp();
    // Can't issue TLB invalidation if the hub isn't initialized.
    if (!hub_initted[ip]) return;
    uint64_t req = adev.reg("reg" + ip + "VM_INVALIDATE_ENG17_REQ").encode({{"flush_type", flush_type}, {"per_vmid_invalidate_req", 1ull << vmid},
        {"invalidate_l2_ptes", 1}, {"invalidate_l2_pde0", 1}, {"invalidate_l2_pde1", 1}, {"invalidate_l2_pde2", 1}, {"invalidate_l1_ptes", 1},
        {"clear_protection_fault_status_addr", 0}});
    const bool use_sema = ip == "MM";   // vf can't use sema
    std::vector<int> insts = ip == "MM" ? mm_insts : std::vector<int>();
    if (ip != "MM") for (int i = 0; i < adev.gfx->xccs; ++i) insts.push_back(i);
    for (int inst : insts) {
        if (use_sema) wait_cond([&] { return adev.reg("regMMVM_INVALIDATE_ENG17_SEM").read(inst) & 0x1; }, 1, 10000, "mm flush_tlb timeout");
        adev.reg("reg" + ip + "VM_INVALIDATE_ENG17_REQ").write(req, {}, inst);
        wait_cond([&] { return adev.reg("reg" + ip + "VM_INVALIDATE_ENG17_ACK").read(inst) & (1u << vmid); }, 1ull << vmid, 10000, "flush_tlb timeout");
        if (use_sema) adev.reg("regMMVM_INVALIDATE_ENG17_SEM").write(0x0, {}, inst);
        if (ip == "MM") {   // GC >= 11
            adev.reg("regMMVM_L2_BANK_SELECT_RESERVED_CID2").update({{"reserved_cache_private_invalidation", 1}}, inst);
            // Read back the register to ensure the invalidation is complete
            adev.reg("regMMVM_L2_BANK_SELECT_RESERVED_CID2").read(inst);
        }
    }
}
inline void AM_GMC::enable_vm_addressing(const AMPageTableEntry& page_table, const std::string& ip, int vmid, int inst) {
    const std::string c = "reg" + ip + "VM_CONTEXT" + std::to_string(vmid);
    adev.wreg_pair(c + "_PAGE_TABLE_START_ADDR", "_LO32", "_HI32", vm_base >> 12, inst);
    adev.wreg_pair(c + "_PAGE_TABLE_END_ADDR", "_LO32", "_HI32", vm_end >> 12, inst);
    adev.wreg_pair(c + "_PAGE_TABLE_BASE_ADDR", "_LO32", "_HI32", adev.paddr2xgmi(page_table.paddr) | 1, inst);
    std::vector<AMKV> kw;
    static const char* const fault[] = {"pde0_protection_fault_enable_interrupt", "dummy_page_protection_fault_enable_interrupt",
        "range_protection_fault_enable_interrupt", "valid_protection_fault_enable_interrupt", "read_protection_fault_enable_interrupt",
        "write_protection_fault_enable_interrupt", "execute_protection_fault_enable_interrupt"};
    static const char* const dflt[] = {"pde0_protection_fault_enable_default", "dummy_page_protection_fault_enable_default",
        "range_protection_fault_enable_default", "valid_protection_fault_enable_default", "read_protection_fault_enable_default",
        "write_protection_fault_enable_default", "execute_protection_fault_enable_default"};
    for (const char* f : fault) kw.push_back({f, 1});
    for (const char* f : dflt) kw.push_back({f, 1});
    kw.push_back({"enable_context", 1});
    kw.push_back({"page_table_depth", (uint64_t)((trans_futher ? 2 : 3) - page_table.lv)});
    kw.push_back({"page_table_block_size", trans_futher ? 9ull : 0ull});
    adev.reg(c + "_CNTL").write(0x1800000, kw, inst);
}
inline void AM_GMC::init_hub(const std::string& ip, const std::vector<int>& insts) {
    // Init system apertures
    for (int inst : insts) {
        adev.reg("reg" + ip + "MC_VM_AGP_BASE").write(0, {}, inst);
        adev.reg("reg" + ip + "MC_VM_AGP_BOT").write(0xffffffffffffull >> 24, {}, inst);   // disable AGP
        adev.reg("reg" + ip + "MC_VM_AGP_TOP").write(0, {}, inst);
        adev.reg("reg" + ip + "MC_VM_SYSTEM_APERTURE_LOW_ADDR").write(fb_base >> 18, {}, inst);
        adev.reg("reg" + ip + "MC_VM_SYSTEM_APERTURE_HIGH_ADDR").write(fb_end >> 18, {}, inst);
        adev.wreg_pair("reg" + ip + "MC_VM_SYSTEM_APERTURE_DEFAULT_ADDR", "_LSB", "_MSB", memscratch_xgmi_paddr >> 12, inst);
        adev.wreg_pair("reg" + ip + "VM_L2_PROTECTION_FAULT_DEFAULT_ADDR", "_LO32", "_HI32", dummy_page_xgmi_paddr >> 12, inst);
        adev.reg("reg" + ip + "VM_L2_PROTECTION_FAULT_CNTL2").update({{"active_page_migration_pte_read_retry", 1}}, inst);
        // Init TLB and cache
        adev.reg("reg" + ip + "MC_VM_MX_L1_TLB_CNTL").update({{"enable_l1_tlb", 1}, {"system_access_mode", 3}, {"enable_advanced_driver_model", 1},
            {"system_aperture_unmapped_access", 0}, {"mtype", am::soc11::MTYPE_UC}}, inst);
        adev.reg("reg" + ip + "VM_L2_CNTL").update({{"enable_l2_cache", 1}, {"enable_default_page_out_to_system_memory", 1},
            {"l2_pde0_cache_tag_generation_mode", 0}, {"pde_fault_classification", 0}, {"context1_identity_access_mode", 1},
            {"identity_mode_fragment_size", 0}, {"enable_l2_fragment_processing", 0}}, inst);   // GC >= 10
        adev.reg("reg" + ip + "VM_L2_CNTL2").update({{"invalidate_all_l1_tlbs", 1}, {"invalidate_l2_cache", 1}}, inst);
        adev.reg("reg" + ip + "VM_L2_CNTL3").write({{"l2_cache_4k_associativity", 1}, {"l2_cache_bigk_associativity", 1},
            {"bank_select", trans_futher ? 12ull : 9ull}, {"l2_cache_bigk_fragment_size", trans_futher ? 9ull : 6ull}}, inst);
        adev.reg("reg" + ip + "VM_L2_CNTL4").write({{"l2_cache_4k_partition_count", 1}}, inst);
        adev.reg("reg" + ip + "VM_L2_CNTL5").write({{"walker_priority_client_id", 0x1ff}}, inst);   // GC >= 10
        enable_vm_addressing(adev.mm->root_page_table, ip, 0, inst);
        // Disable identity aperture
        adev.wreg_pair("reg" + ip + "VM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR", "_LO32", "_HI32", 0xfffffffffull, inst);
        adev.wreg_pair("reg" + ip + "VM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR", "_LO32", "_HI32", 0x0, inst);
        adev.wreg_pair("reg" + ip + "VM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET", "_LO32", "_HI32", 0x0, inst);
        for (int eng_i = 0; eng_i < 18; ++eng_i)
            adev.wreg_pair("reg" + ip + "VM_INVALIDATE_ENG" + std::to_string(eng_i) + "_ADDR_RANGE", "_LO32", "_HI32", 0x1fffffffffull, inst);
    }
    hub_initted[ip] = true;
}
inline uint64_t AM_GMC::get_pte_flags(int pte_lv, bool is_table, int64_t frag, bool uncached, bool system, bool snooped, bool valid, uint64_t extra) const {
    extra |= (system ? am::AMDGPU_PTE_SYSTEM : 0) | (snooped ? am::AMDGPU_PTE_SNOOPED : 0) | (valid ? am::AMDGPU_PTE_VALID : 0) | (((uint64_t)frag & 0x1f) << 7);
    if (!is_table) extra |= am::AMDGPU_PTE_WRITEABLE | am::AMDGPU_PTE_READABLE | am::AMDGPU_PTE_EXECUTABLE;
    extra |= (uint64_t)(uncached ? am::soc11::MTYPE_UC : 0) << 48;   // AMDGPU_PTE_MTYPE_NV10(0, mtype): GC >= 10, < 12
    extra |= (!is_table && pte_lv != (int)am::AMDGPU_VM_PTB) ? am::AMDGPU_PDE_PTE : 0;
    return extra;
}

// ── AM_SMU (ip.py:194-265) ────────────────────────────────────────────────────────────────────────────────────────────
inline void AM_SMU::init_sw() { driver_table_paddr = adev.mm->palloc(0x4000, 0x1000, false, true); }
inline void AM_SMU::init_hw() {
    send_msg(am::smu13::PPSMC_MSG_SetDriverDramAddrHigh, hi32(adev.paddr2mc(driver_table_paddr)));
    send_msg(am::smu13::PPSMC_MSG_SetDriverDramAddrLow, lo32(adev.paddr2mc(driver_table_paddr)));
    send_msg(am::smu13::PPSMC_MSG_EnableAllSmuFeatures, 0);
}
inline bool AM_SMU::is_smu_alive() {
    try { send_msg(am::smu13::PPSMC_MSG_GetSmuVersion, 0, false, 100); }
    catch (const TGPyError& e) { if (e.type != "TimeoutError") throw; }
    return adev.reg("mmMP1_SMN_C2PMSG_90").read() != 0;
}
inline void AM_SMU::mode1_reset() {
    adev.log("mode1 reset");
    send_msg(2, 0, false, 10000, true);   // __DEBUGSMC_MSG_Mode1Reset: MP0 13.0.0
    sleep_s(0.5);   // not a hive: 500ms
}
inline const std::vector<std::pair<uint32_t, std::vector<uint64_t>>>& AM_SMU::read_clocks(const std::vector<uint32_t>& clk_list) {
    if (clocks_read) return clocks;   // functools.cache: one clk_list a process here
    for (uint32_t clck : clk_list) {
        uint64_t cnt = send_msg(am::smu13::PPSMC_MSG_GetDpmFreqByIndex, ((uint64_t)clck << 16) | 0xff, true) & 0x7fffffff;
        if (!cnt) continue;
        std::vector<uint64_t> v;
        for (uint64_t i = 0; i < cnt; ++i) v.push_back(send_msg(am::smu13::PPSMC_MSG_GetDpmFreqByIndex, ((uint64_t)clck << 16) | i, true) & 0x7fffffff);
        clocks.push_back({clck, v});
    }
    clocks_read = true;
    return clocks;
}
inline void AM_SMU::set_clocks(bool none, int level) {
    std::vector<uint32_t> clks = {am::smu13::PPCLK_UCLK, am::smu13::PPCLK_FCLK, am::smu13::PPCLK_SOCCLK, am::smu13::PPCLK_GFXCLK};   // MP0 13.0.0: GFXCLK too
    if (none) {
        for (uint32_t clck : clks) {
            try { send_msg(am::smu13::PPSMC_MSG_SetSoftMinByFreq, (uint64_t)clck << 16, false, 20); }
            catch (const TGPyError& e) { if (e.type != "TimeoutError") throw; }
            send_msg(am::smu13::PPSMC_MSG_SetSoftMaxByFreq, ((uint64_t)clck << 16) | 0xffff);
        }
        return;
    }
    for (const auto& [clck, vals] : read_clocks(clks)) {
        uint64_t v = vals[level < 0 ? vals.size() + level : (size_t)level];
        try { send_msg(am::smu13::PPSMC_MSG_SetSoftMinByFreq, ((uint64_t)clck << 16) | v, false, 20); }
        catch (const TGPyError& e) { if (e.type != "TimeoutError") throw; }
        send_msg(am::smu13::PPSMC_MSG_SetSoftMaxByFreq, ((uint64_t)clck << 16) | v);
    }
}
inline void AM_SMU::set_power_limit(double watts) {
    uint64_t ppt_limit = std::max<int64_t>((int64_t)std::llround(watts), 1);
    send_msg(am::smu13::PPSMC_MSG_SetPptLimit, ppt_limit);
    adev.log("GPU power limit set to " + std::to_string(ppt_limit) + "W");
}
inline uint64_t AM_SMU::send_msg(uint32_t msg, uint64_t param, bool read_back_arg, int64_t timeout, bool debug) {
    const char* resp = debug ? "mmMP1_SMN_C2PMSG_54" : "mmMP1_SMN_C2PMSG_90";
    const char* arg = debug ? "mmMP1_SMN_C2PMSG_53" : "mmMP1_SMN_C2PMSG_82";
    // _smu_cmn_send_msg
    adev.reg(resp).write(0);   // resp reg
    adev.reg(arg).write(param);
    adev.reg(debug ? "mmMP1_SMN_C2PMSG_75" : "mmMP1_SMN_C2PMSG_66").write(msg);
    wait_cond([&] { return adev.reg(resp).read(); }, 1, timeout, "SMU msg " + hex(msg) + " timeout");
    return read_back_arg ? adev.reg(arg).read() : 0;
}

// ── AM_GFX (ip.py:267-435) ────────────────────────────────────────────────────────────────────────────────────────────
inline void AM_GFX::init_sw() {
    xccs = 0;
    for (auto& [i, _] : adev.regs_offset[GC]) if (!adev.harvested[GC].count(i)) ++xccs;
    mqd_paddr.clear();
    mqd_mc.clear();
    for (int i = 0; i < 2; ++i) mqd_paddr.push_back(adev.mm->palloc(0x1000 * (uint64_t)xccs, 0x1000, false, true));
    for (uint64_t p : mqd_paddr) mqd_mc.push_back(adev.paddr2mc(p));
}
inline void AM_GFX::init_hw() {
    // Wait for RLC autoload to complete
    wait_cond([&] { return adev.reg("regCP_STAT").read() == 0 || adev.reg("regRLC_RLCS_BOOTLOAD_STATUS").read_bitfields()["bootload_complete"] == 0; },
              1, 10000, "RLC autoload timeout");
    std::vector<int> insts;
    for (int i = 0; i < xccs; ++i) insts.push_back(i);
    adev.gmc->init_hub("GC", insts);
    if (adev.partial_boot) return reset_mec();
    config_mec();
    // NOTE: Golden reg for gfx11. No values for this reg provided. The kernel just ors 0x20000000 to this reg.
    for (int xcc = 0; xcc < xccs; ++xcc) adev.reg("regTCP_CNTL").write(adev.reg("regTCP_CNTL").read(xcc) | 0x20000000, {}, xcc);
    for (int xcc = 0; xcc < xccs; ++xcc) adev.reg("regRLC_CNTL").write(0x1, {}, xcc);
    for (int xcc = 0; xcc < xccs; ++xcc) adev.reg("regRLC_SRM_CNTL").update({{"srm_enable", 1}, {"auto_incr_addr", 1}}, xcc);
    for (int xcc = 0; xcc < xccs; ++xcc) adev.reg("regRLC_SPM_MC_CNTL").write(0xf, {}, xcc);
    adev.soc->doorbell_enable(0, 0x3, 0x3);   // NBIO not 7.9
    adev.soc->doorbell_enable(3, 0x6, 0x3);
    for (int xcc = 0; xcc < xccs; ++xcc) {
        adev.reg("regGRBM_CNTL").update({{"read_timeout", 0xff}}, xcc);
        for (int i = 0; i < 16; ++i) {
            grbm_select(0, 0, 0, (uint64_t)i, xcc);
            adev.reg("regSH_MEM_CONFIG").write({{"initial_inst_prefetch", 3}, {"address_mode", am::soc11::SH_MEM_ADDRESS_MODE_64},
                                                {"alignment_mode", am::soc11::SH_MEM_ALIGNMENT_MODE_UNALIGNED}}, xcc);
            // Configure apertures:
            // LDS:         0x10000000'00000000 - 0x10000001'00000000 (4GB)
            // Scratch:     0x20000000'00000000 - 0x20000001'00000000 (4GB)
            adev.reg("regSH_MEM_BASES").write({{"shared_base", 0x1}, {"private_base", 0x2}}, xcc);
        }
        grbm_select(0, 0, 0, 0, xcc);
        // Configure MEC doorbell range
        adev.reg("regCP_MEC_DOORBELL_RANGE_LOWER").write(0x100 * (uint64_t)xcc, {}, xcc);
        adev.reg("regCP_MEC_DOORBELL_RANGE_UPPER").write(0x100 * (uint64_t)xcc + 0xf8, {}, xcc);
    }
    enable_mec();
}
inline void AM_GFX::reset_mec() {
    dequeue_hqds();
    for (int xcc = 0; xcc < xccs; ++xcc) adev.reg("regGRBM_SOFT_RESET").write({{"soft_reset_cp", 1}, {"soft_reset_cpc", 1}}, xcc);   // GC < 12
    sleep_s(0.05);
    for (int xcc = 0; xcc < xccs; ++xcc) adev.reg("regGRBM_SOFT_RESET").write(0x0, {}, xcc);
    config_mec();
    enable_mec();
}
inline uint64_t AM_GFX::setup_ring(uint64_t ring_addr, uint64_t ring_size, uint64_t rptr_addr, uint64_t wptr_addr, uint64_t eop_addr, uint64_t eop_size,
                                   int idx, bool aql) {
    const int pipe = idx / 4, queue = idx % 4;
    const uint64_t doorbell = am::AMDGPU_NAVI10_DOORBELL_MEC_RING0;
    for (int xcc = 0; xcc < (aql ? xccs : 1); ++xcc) {
        grbm_select(1, (uint64_t)pipe, (uint64_t)queue, 0, xcc);
        am::struct_v11_compute_mqd m;
        memset(&m, 0, sizeof m);
        m.header = 0xC0310800;
        m.cp_mqd_base_addr_lo = (uint32_t)lo32(mqd_mc[queue] + 0x1000ull * xcc);
        m.cp_mqd_base_addr_hi = (uint32_t)hi32(mqd_mc[queue] + 0x1000ull * xcc);
        m.cp_hqd_pipe_priority = 0x2;
        m.cp_hqd_queue_priority = 0xf;
        m.cp_hqd_quantum = 0x111;
        m.cp_hqd_persistent_state = (uint32_t)adev.reg("regCP_HQD_PERSISTENT_STATE").encode({{"preload_size", 0x55}, {"preload_req", 1}});
        m.cp_hqd_pq_base_lo = (uint32_t)lo32(ring_addr >> 8);
        m.cp_hqd_pq_base_hi = (uint32_t)hi32(ring_addr >> 8);
        m.cp_hqd_pq_rptr_report_addr_lo = (uint32_t)lo32(rptr_addr);
        m.cp_hqd_pq_rptr_report_addr_hi = (uint32_t)hi32(rptr_addr);
        m.cp_hqd_pq_wptr_poll_addr_lo = (uint32_t)lo32(wptr_addr);
        m.cp_hqd_pq_wptr_poll_addr_hi = (uint32_t)hi32(wptr_addr);
        m.cp_hqd_pq_doorbell_control = (uint32_t)adev.reg("regCP_HQD_PQ_DOORBELL_CONTROL").encode({{"doorbell_offset", doorbell * 2}, {"doorbell_en", 1}});
        std::vector<AMKV> pq = {{"rptr_block_size", 5}, {"unord_dispatch", 0}, {"queue_size", (uint64_t)(tg_bit_length(ring_size / 4) - 2)}};
        if (aql) { pq.push_back({"queue_full_en", 1}); pq.push_back({"slot_based_wptr", 2}); pq.push_back({"no_update_rptr", (uint64_t)(xcc != 0 || xccs == 1)}); }
        m.cp_hqd_pq_control = (uint32_t)adev.reg("regCP_HQD_PQ_CONTROL").encode(pq);
        m.cp_hqd_ib_control = (uint32_t)adev.reg("regCP_HQD_IB_CONTROL").encode({{"min_ib_avail_size", 0x3}});
        m.cp_hqd_hq_status0 = 0x20004000;
        m.cp_mqd_control = (uint32_t)adev.reg("regCP_MQD_CONTROL").encode({{"priv_state", 1}});
        m.cp_hqd_vmid = 0;
        m.cp_hqd_aql_control = aql ? 1 : 0;
        m.cp_hqd_eop_base_addr_lo = (uint32_t)lo32(eop_addr >> 8);
        m.cp_hqd_eop_base_addr_hi = (uint32_t)hi32(eop_addr >> 8);
        m.cp_hqd_eop_control = (uint32_t)adev.reg("regCP_HQD_EOP_CONTROL").encode({{"eop_size", (uint64_t)(tg_bit_length(eop_size / 4) - 2)}});
        if (aql && xccs > 1) throw TGPyError("RuntimeError", "AQL on several XCCs: not ported");
        uint32_t* se = &m.compute_static_thread_mgmt_se0;   // se0..se7 (GC >= 10): checked contiguous below
        static_assert(offsetof(am::struct_v11_compute_mqd, compute_static_thread_mgmt_se1) == offsetof(am::struct_v11_compute_mqd, compute_static_thread_mgmt_se0) + 4);
        m.compute_static_thread_mgmt_se0 = m.compute_static_thread_mgmt_se1 = m.compute_static_thread_mgmt_se2 = m.compute_static_thread_mgmt_se3 = 0xffffffff;
        m.compute_static_thread_mgmt_se4 = m.compute_static_thread_mgmt_se5 = m.compute_static_thread_mgmt_se6 = m.compute_static_thread_mgmt_se7 = 0xffffffff;
        (void)se;
        adev.vram_write(mqd_paddr[queue] + 0x1000ull * xcc, &m, sizeof m);
        const uint32_t* mv = (const uint32_t*)&m;
        const uint32_t lo = adev.reg("regCP_MQD_BASE_ADDR").addr(xcc), hi = adev.reg("regCP_HQD_PQ_WPTR_HI").addr(xcc);
        for (uint32_t i = 0, r = lo; r <= hi; ++i, ++r) adev.wreg(r, mv[0x80 + i], xcc);
        adev.reg("regCP_HQD_ACTIVE").write(0x1, {}, xcc);
        adev.gmc->flush_hdp();
        grbm_select(0, 0, 0, 0, xcc);
    }
    return doorbell;
}
inline void AM_GFX::set_clockgating_state() {
    if (adev.has_reg("regMM_ATC_L2_MISC_CG")) adev.reg("regMM_ATC_L2_MISC_CG").write({{"enable", 1}, {"mem_ls_enable", 1}});
    for (int xcc = 0; xcc < xccs; ++xcc) {
        adev.reg("regRLC_SAFE_MODE").write({{"message", 1}, {"cmd", 1}}, xcc);
        wait_cond([&] { return adev.reg("regRLC_SAFE_MODE").read(xcc) & 0x1; }, 0, 10000, "RLC safe mode timeout");
        adev.reg("regRLC_CGCG_CGLS_CTRL").update({{"cgcg_gfx_idle_threshold", 0x36}, {"cgcg_en", 1}, {"cgls_rep_compansat_delay", 0xf}, {"cgls_en", 1}}, xcc);
        adev.reg("regCP_RB_WPTR_POLL_CNTL").update({{"poll_frequency", 0x100}, {"idle_poll_count", 0x90}}, xcc);
        adev.reg("regCP_INT_CNTL").update({{"cntx_busy_int_enable", 1}, {"cntx_empty_int_enable", 1}, {"cmp_busy_int_enable", 1}}, xcc);
        adev.reg("regSDMA0_RLC_CGCG_CTRL").update({{"cgcg_int_enable", 1}}, xcc);   // GC >= 10
        adev.reg("regSDMA1_RLC_CGCG_CTRL").update({{"cgcg_int_enable", 1}}, xcc);
        adev.reg("regRLC_CGTT_MGCG_OVERRIDE").update({{"perfmon_clock_state", 1}, {"gfxip_repeater_fgcg_override", 0}, {"gfxip_fgcg_override", 0},
            {"grbm_cgtt_sclk_override", 0}, {"rlc_cgtt_sclk_override", 0}, {"gfxip_mgcg_override", 0}, {"gfxip_cgls_override", 0},
            {"gfxip_cgcg_override", 0}}, xcc);   // GC >= 11's feats
        adev.reg("regRLC_SAFE_MODE").write({{"message", 0}, {"cmd", 1}}, xcc);
    }
}
inline void AM_GFX::grbm_select(uint64_t me, uint64_t pipe, uint64_t queue, uint64_t vmid, int inst) {
    adev.reg("regGRBM_GFX_CNTL").write({{"meid", me}, {"pipeid", pipe}, {"vmid", vmid}, {"queueid", queue}}, inst);
}
inline void AM_GFX::enable_mec() {
    for (int xcc = 0; xcc < xccs; ++xcc)
        adev.reg("regCP_MEC_RS64_CNTL").update({{"mec_pipe0_reset", 0}, {"mec_pipe0_active", 1}, {"mec_halt", 0}}, xcc);   // GC >= 10
    sleep_s(0.05);   // Wait for MEC to be ready
}
inline void AM_GFX::config_mec() {
    for (int xcc = 0; xcc < adev.gfx->xccs; ++xcc) {
        // _config_helper(eng_name="MEC", cntl_reg="MEC_RS64", eng_reg="MEC_RS64", pipe_cnt=1, me=1, xcc=xcc): GC >= 10, < 12
        for (int pipe = 0; pipe < 1; ++pipe) {
            grbm_select(1, (uint64_t)pipe, 0, 0, xcc);
            adev.wreg_pair("regCP_MEC_RS64_PRGRM_CNTR_START", "", "_HI", adev.fw->ucode_start.at("MEC") >> 2, xcc);
        }
        grbm_select(0, 0, 0, 0, xcc);
        adev.reg("regCP_MEC_RS64_CNTL").update({{"mec_pipe0_reset", 1}}, xcc);
        adev.reg("regCP_MEC_RS64_CNTL").update({{"mec_pipe0_reset", 0}}, xcc);
    }
}
inline void AM_GFX::dequeue_hqds() {
    dequeue_unconfirmed = 0;
    for (int q = 0; q < 2; ++q)
        for (int xcc = 0; xcc < xccs; ++xcc) {
            grbm_select(1, 0, (uint64_t)q, 0, xcc);
            if (adev.reg("regCP_HQD_ACTIVE").read(xcc) & 1) {
                adev.reg("regCP_HQD_DEQUEUE_REQUEST").write(0x2, {}, xcc);   // 1 - DRAIN_PIPE; 2 - RESET_WAVES
                adev.reg("regSPI_COMPUTE_QUEUE_RESET").write(0x1, {}, xcc);
                if (!adev.is_err_state) {
                    try { wait_cond([&] { return adev.reg("regCP_HQD_ACTIVE").read(xcc) & 1; }, 0, 10000, "HQD dequeue timeout"); }
                    catch (const TGPyError& e) {   // kernel tolerates this too; a wedged wave can survive RESET_WAVES
                        if (e.type != "TimeoutError") throw;
                        adev.log("HQD dequeue timeout xcc" + std::to_string(xcc) + " q" + std::to_string(q) + ", continuing");
                        ++dequeue_unconfirmed;
                    }
                } else ++dequeue_unconfirmed;   // not waited for: not seen inactive
            }
        }
    grbm_select();
}

// ── AM_IH (ip.py:437-525) ─────────────────────────────────────────────────────────────────────────────────────────────
inline void AM_IH::init_sw() {
    ring_size = 256 << 10;
    auto alloc_ring = [&](uint64_t size) {
        uint64_t a = adev.mm->palloc(size, 0x1000, false, true);
        uint64_t b = adev.mm->palloc(0x1000, 0x1000, false, true);
        return std::make_pair(a, b);
    };
    auto r0 = alloc_ring(ring_size);
    auto r1 = alloc_ring(ring_size);
    rings = {{r0.first, r0.second, "", 0}, {r1.first, r1.second, "_RING1", 1}};
    ring_view_paddr = rings[0].ring_vm;
}
inline void AM_IH::init_hw() {
    for (const Ring& r : rings) {
        adev.wreg_pair("regIH_RB_BASE", r.suf, "_HI" + r.suf, adev.paddr2mc(r.ring_vm) >> 8);
        std::vector<AMKV> kw = {{"mc_space", 4}, {"wptr_overflow_clear", 1}, {"rb_size", (uint64_t)tg_bit_length(ring_size / 4 - 1)}, {"mc_snoop", 1},
                                {"mc_ro", 0}, {"mc_vmid", 0}};
        if (r.ring_id == 0) { kw.push_back({"wptr_overflow_enable", 1}); kw.push_back({"rptr_rearm", 1}); }
        else kw.push_back({"rb_full_drain_enable", 1});
        adev.reg("regIH_RB_CNTL" + r.suf).write(kw);
        if (r.ring_id == 0) adev.wreg_pair("regIH_RB_WPTR_ADDR", "_LO", "_HI", adev.paddr2mc(r.rwptr_vm));
        adev.reg("regIH_RB_WPTR" + r.suf).write(0);
        adev.reg("regIH_RB_RPTR" + r.suf).write(0);
        adev.reg("regIH_DOORBELL_RPTR" + r.suf).write({{"enable", 0}});
    }
    // OSSSYS != 4.4.2
    adev.reg("regIH_STORM_CLIENT_LIST_CNTL").update({{"client18_is_storm_client", 1}});
    adev.reg("regIH_INT_FLOOD_CNTL").update({{"flood_cntl_enable", 1}});
    adev.reg("regIH_MSI_STORM_CTRL").update({{"delay", 3}});
    // toggle interrupts
    for (const Ring& r : rings) {
        if (r.ring_id == 0) adev.reg("regIH_RB_CNTL" + r.suf).update({{"rb_enable", 1}, {"enable_intr", 1}});
        else adev.reg("regIH_RB_CNTL" + r.suf).update({{"rb_enable", 1}});
    }
}
inline void AM_IH::drain() {
    const std::string suf = rings[0].suf;
    am::AMFieldValues wptr = adev.reg("regIH_RB_WPTR" + suf).read_bitfields();
    adev.reg("regIH_RB_RPTR").write(wptr["offset"] % (ring_size / 4));
    if (wptr["rb_overflow"]) {
        adev.reg("regIH_RB_WPTR" + suf).update({{"rb_overflow", 0}});
        adev.reg("regIH_RB_CNTL" + suf).update({{"wptr_overflow_clear", 1}});
        adev.reg("regIH_RB_CNTL" + suf).update({{"wptr_overflow_clear", 0}});
    }
}
inline void AM_IH::interrupt_handler() {
    const std::string suf = rings[0].suf;
    am::AMFieldValues wptr = adev.reg("regIH_RB_WPTR" + suf).read_bitfields();
    uint64_t rptr = adev.reg("regIH_RB_RPTR").read();
    const uint64_t n = ring_size / 4;
    char buf[512];
    while (rptr != wptr["offset"]) {
        uint32_t e[8];
        for (int i = 0; i < 8; ++i) e[i] = adev.vram_i(ring_view_paddr + 4 * ((rptr + i) % n));
        rptr = (rptr + 8) % n;
        const uint32_t client = amdt::ih_get(e, amdt::IH_CLIENT_ID), src = amdt::ih_get(e, amdt::IH_SOURCE_ID);
        const std::string src_name = adev.soc->ih_src_name(client, src);
        if (src_name == "SDMA_TRAP" || src_name == "CP_EOP_INTR") continue;
        const char* client_name = "None";
        for (const amdt::IHName& c : amdt::IH_SOC21_CLIENTS) if (c.id == client) client_name = c.name;
        snprintf(buf, sizeof buf, "IH (%s/%s) client=%s src=%s(%u) ring=%u vmid=%u(%u) pasid=%u node=%u ctx=[%s, %s, %s, %s]", hex(rptr).c_str(),
                 hex(wptr["offset"]).c_str(), client_name, src_name.c_str(), src, amdt::ih_get(e, amdt::IH_RING_ID), amdt::ih_get(e, amdt::IH_VMID),
                 amdt::ih_get(e, amdt::IH_VMID_TYPE), amdt::ih_get(e, amdt::IH_PASID), amdt::ih_get(e, amdt::IH_NODEID), hex(e[4]).c_str(),
                 hex(e[5]).c_str(), hex(e[6]).c_str(), hex(e[7]).c_str());
        adev.print(buf);
        if (src_name == "SQ_INTERRUPT_ID") {   // soc21's fields
            const uint64_t enc_type = am::am_getbits(e[5], 6, 7), err_type = am::am_getbits(e[4], 21, 24);
            static const char* const encs[] = {"auto", "wave", "error"};
            static const char* const errs[] = {"EDC_FUE", "ILLEGAL_INST", "MEMVIOL", "EDC_FED"};
            if (enc_type > 2 || (enc_type == 2 && err_type > 3)) throw TGPyError("IndexError", "list index out of range");
            adev.print(std::string("sq_intr: ") + encs[enc_type] + (enc_type == 2 ? std::string(" (") + errs[err_type] + ")" : ""));
            adev.is_err_state |= enc_type == 2;
        } else if (src_name == "UTCL2_FAULT") {
            am::AMFieldValues bf = adev.reg(adev.gmc->pf_status_reg("GC")).read_bitfields();
            uint64_t va = (uint64_t)adev.reg("regGCVM_L2_PROTECTION_FAULT_ADDR_HI32").read() << 32;
            va |= adev.reg("regGCVM_L2_PROTECTION_FAULT_ADDR_LO32").read();
            adev.print("GCVM_L2_PROTECTION_FAULT_STATUS: " + bf.repr() + " " + hex(va << 12));
            adev.reg("regGCVM_L2_PROTECTION_FAULT_CNTL").update({{"clear_protection_fault_status_addr", 1}});
            adev.is_err_state = true;
        } else adev.is_err_state = true;
    }
    drain();
    am::AMFieldValues bif = adev.reg("regBIF_BX0_BIF_DOORBELL_INT_CNTL").read_bitfields();
    uint64_t athub_err = bif["ras_athub_err_event_interrupt_status"], cntlr_err = bif["ras_cntlr_interrupt_status"];
    if (athub_err || cntlr_err) {
        adev.print(std::string("fatal hardware error detected: ") + (athub_err ? "RAS_ATHUB_ERR_EVENT " : "") + (cntlr_err ? "RAS_CNTLR" : ""));
        // smu_13_0_0 has no PPSMC_MSG_QueryValidMcaCount: no ACA banks to read
        static_assert(!am::smu13::has_PPSMC_MSG_QueryValidMcaCount, "the ACA bank dump is not ported");
        adev.reg("regBIF_BX0_BIF_DOORBELL_INT_CNTL").write({{"ras_cntlr_interrupt_clear", cntlr_err}, {"ras_athub_err_event_interrupt_clear", athub_err}});
        adev.is_err_state = true;
    }
}

// ── AM_SDMA (ip.py:527-586) ───────────────────────────────────────────────────────────────────────────────────────────
inline void AM_SDMA::init_hw() {
    for (int pipe_id = 0; pipe_id < 1; ++pipe_id) {   // SDMA >= 5
        const std::string pipe = std::to_string(pipe_id);
        const int inst = 0;
        // SDMA >= 6
        adev.reg("regSDMA" + pipe + "_WATCHDOG_CNTL").update({{"queue_hang_count", 100}}, inst);   // 10s, 100ms per unit
        adev.reg("regSDMA" + pipe + "_UTCL1_CNTL").update({{"resp_mode", 3}, {"redo_delay", 9}}, inst);
        // rd=noa, wr=bypass
        adev.reg("regSDMA" + pipe + "_UTCL1_PAGE").update({{"rd_l2_policy", 2}, {"wr_l2_policy", 3}, {"llc_noalloc", 1}}, inst);   // F32
        adev.reg("regSDMA" + pipe + "_" + sdma_name + "_CNTL").update({{"halt", 0}, {"th1_reset", 0}}, inst);
        adev.reg("regSDMA" + pipe + "_CNTL").update({{"trap_enable", 1}}, inst);   // SDMA > 5.2.0: no utc_l1_enable
    }
    adev.soc->doorbell_enable(2, 0xe, 0x3, am::AMDGPU_NAVI10_DOORBELL_sDMA_ENGINE0 * 2, 4);   // NBIO not 7.9
}
inline void AM_SDMA::fini_hw() {
    for (auto& [reg, inst] : sdma_reginst) {
        adev.reg(reg + "_RB_CNTL").update({{"rb_enable", 0}}, inst);
        adev.reg(reg + "_IB_CNTL").update({{"ib_enable", 0}}, inst);
        adev.reg(reg + "_DOORBELL").update({{"enable", 0}}, inst);
        adev.reg(reg + "_DOORBELL_OFFSET").update({{"offset", 0}}, inst);
    }
    // SDMA >= 6
    adev.reg("regGRBM_SOFT_RESET").write({{"soft_reset_sdma0", 1}});
    sleep_s(0.01);
    adev.reg("regGRBM_SOFT_RESET").write(0x0);
}
inline uint64_t AM_SDMA::setup_ring(uint64_t ring_addr, uint64_t ring_size, uint64_t rptr_addr, uint64_t wptr_addr, int idx) {
    if (idx > 0) throw TGPyError("RuntimeError", "am " + adev.devfmt + ": sdma queue " + std::to_string(idx) + " is not available");   // SDMA >= 5
    const int pipe = idx / 4, queue = idx % 4;
    const std::string reg = "regSDMA" + std::to_string(pipe) + "_QUEUE" + std::to_string(queue);   // not SDMA 4.4
    const int inst = 0;
    const uint64_t doorbell = am::AMDGPU_NAVI10_DOORBELL_sDMA_ENGINE0 + (uint64_t)(pipe + queue * 4) * 0xA;
    sdma_reginst.push_back({reg, inst});
    adev.reg(reg + "_MINOR_PTR_UPDATE").write(0x1, {}, inst);
    adev.wreg_pair(reg + "_RB_RPTR", "", "_HI", 0, inst);
    adev.wreg_pair(reg + "_RB_WPTR", "", "_HI", 0, inst);
    adev.wreg_pair(reg + "_RB_BASE", "", "_HI", ring_addr >> 8, inst);
    adev.wreg_pair(reg + "_RB_RPTR_ADDR", "_LO", "_HI", rptr_addr, inst);
    adev.wreg_pair(reg + "_RB_WPTR_POLL_ADDR", "_LO", "_HI", wptr_addr, inst);
    adev.reg(reg + "_DOORBELL_OFFSET").update({{"offset", doorbell * 2}}, inst);
    adev.reg(reg + "_DOORBELL").update({{"enable", 1}}, inst);
    adev.reg(reg + "_MINOR_PTR_UPDATE").write(0x0, {}, inst);
    const std::string poll = sdma_name == "F32" ? "f32_wptr_poll_enable" : "mcu_wptr_poll_enable";
    adev.reg(reg + "_RB_CNTL").write({{poll.c_str(), 1}, {"rb_vmid", 0}, {"rptr_writeback_enable", 1}, {"rptr_writeback_timer", 4}, {"rb_enable", 1},
                                      {"rb_priv", 1}, {"rb_size", (uint64_t)(tg_bit_length(ring_size / 4) - 1)}}, inst);
    adev.reg(reg + "_IB_CNTL").update({{"ib_enable", 1}}, inst);
    return doorbell;
}

// ── AM_PSP (ip.py:588-733) ────────────────────────────────────────────────────────────────────────────────────────────
inline void AM_PSP::init_sw() {
    reg_pref = "regMP0_SMN_C2PMSG";   // MP0 < 14
    msg1_paddr = adev.mm->palloc(am::PSP_1_MEG, am::PSP_1_MEG, false, true);
    msg1_addr = adev.paddr2mc(msg1_paddr);
    msg1_size = am::PSP_1_MEG;
    cmd_paddr = adev.mm->palloc(am::PSP_CMD_BUFFER_SIZE, 0x1000, false, true);
    fence_paddr = adev.mm->palloc(am::PSP_FENCE_BUFFER_SIZE, 0x1000, true, true);
    ring_size = 0x10000;
    ring_paddr = adev.mm->palloc(ring_size, 0x1000, false, true);
    max_tmr_size = 0x1300000;
    tmr_size = 0;
    boot_time_tmr = false;   // MP0 13.0.0
    autoload_tmr = true;
    tmr_paddr = adev.mm->palloc(max_tmr_size, am::PSP_TMR_ALIGNMENT, false, true);
}
inline void AM_PSP::init_hw() {
    const uint32_t spl_key = am::PSP_FW_TYPE_PSP_KDB;   // MP0 < 14
    const std::pair<uint32_t, uint32_t> sos_components[] = {{am::PSP_FW_TYPE_PSP_KDB, am::PSP_BL__LOAD_KEY_DATABASE}, {spl_key, am::PSP_BL__LOAD_TOS_SPL_TABLE},
        {am::PSP_FW_TYPE_PSP_SYS_DRV, am::PSP_BL__LOAD_SYSDRV}, {am::PSP_FW_TYPE_PSP_SOC_DRV, am::PSP_BL__LOAD_SOCDRV},
        {am::PSP_FW_TYPE_PSP_INTF_DRV, am::PSP_BL__LOAD_INTFDRV}, {am::PSP_FW_TYPE_PSP_DBG_DRV, am::PSP_BL__LOAD_DBGDRV},
        {am::PSP_FW_TYPE_PSP_RAS_DRV, am::PSP_BL__LOAD_RASDRV}, {am::PSP_FW_TYPE_PSP_SOS, am::PSP_BL__LOAD_SOSDRV}};
    if (!is_sos_alive()) {
        for (const auto& [fw, compid] : sos_components) bootloader_load_component(fw, compid);
        wait_cond([&] { return is_sos_alive(); }, 1, 10000, "sOS failed to start");
    }
    ring_create();
    if (adev.fw->sos_fw.count(am::PSP_FW_TYPE_PSP_TOC)) tmr_init();
    // SMU fw should be loaded before TMR.
    if (adev.fw->has_smu_psp_desc) load_ip_fw_cmd(adev.fw->smu_psp_desc);
    if (!boot_time_tmr || !autoload_tmr) tmr_load_cmd();
    for (const AMFwDesc& d : adev.fw->descs) load_ip_fw_cmd(d);
    rlc_autoload_cmd();   // GC >= 11
}
inline bool AM_PSP::is_sos_alive() { return adev.reg(reg_pref + "_81").read() != 0x0; }
inline void AM_PSP::wait_for_bootloader() {
    wait_cond([&] { return adev.reg(reg_pref + "_35").read() & 0x80000000; }, 0x80000000, 10000, "BL not ready");
}
inline void AM_PSP::prep_msg1(const uint8_t* data, size_t n) {
    if (n > msg1_size) throw TGPyError("AssertionError", "msg1 buffer is too small " + hex(n) + " > " + hex(msg1_size));
    std::vector<uint8_t> padded(data, data + n);   // HACK: apple's memcpy requires 16-bytes alignment
    padded.resize(n + 4);
    padded.resize((padded.size() + 15) / 16 * 16);
    adev.vram_write(msg1_paddr, padded.data(), padded.size());
    adev.gmc->flush_hdp();
}
inline const char* psp_fw_type_name(uint32_t v) {
    for (const am::Name& n : am::enum_psp_fw_type) if (n.id == v) return n.name;
    return "None";
}
inline const char* psp_gfx_fw_type_name(uint32_t v) {
    for (const am::Name& n : am::enum_psp_gfx_fw_type) if (n.id == v) return n.name;
    return "None";
}
inline void AM_PSP::bootloader_load_component(uint32_t fw, uint32_t compid) {
    auto it = adev.fw->sos_fw.find(fw);
    if (it == adev.fw->sos_fw.end()) return;
    wait_for_bootloader();
    adev.log(std::string("loading sos component: ") + psp_fw_type_name(fw));
    prep_msg1(it->second.data(), it->second.size());
    adev.reg(reg_pref + "_36").write(msg1_addr >> 20);
    adev.reg(reg_pref + "_35").write(compid);
    if (compid != am::PSP_BL__LOAD_SOSDRV) wait_for_bootloader();
}
inline void AM_PSP::tmr_init() {
    // Load TOC and calculate TMR size
    const std::vector<uint8_t>& fwm = adev.fw->sos_fw.at(am::PSP_FW_TYPE_PSP_TOC);
    prep_msg1(fwm.data(), fwm.size());
    tmr_size = load_toc_cmd(fwm.size()).resp.tmr_size;
    if (!(tmr_size <= max_tmr_size)) throw TGPyError("AssertionError", "");
}
inline void AM_PSP::ring_create() {
    // If the ring is already created, destroy it
    if (adev.reg(reg_pref + "_71").read() != 0) {
        adev.reg(reg_pref + "_64").write(am::GFX_CTRL_CMD_ID_DESTROY_RINGS);
        sleep_s(0.02);   // There might be handshake issue with hardware which needs delay
    }
    // Wait until the sOS is ready
    wait_cond([&] { return adev.reg(reg_pref + "_64").read() & 0x80000000; }, 0x80000000, 10000, "sOS not ready");
    adev.wreg_pair(reg_pref, "_69", "_70", adev.paddr2mc(ring_paddr));
    adev.reg(reg_pref + "_71").write(ring_size);
    adev.reg(reg_pref + "_64").write((uint64_t)am::PSP_RING_TYPE__KM << 16);
    sleep_s(0.02);   // There might be handshake issue with hardware which needs delay
    wait_cond([&] { return adev.reg(reg_pref + "_64").read() & 0x8000FFFF; }, 0x80000000, 10000, "sOS ring not created");
}
inline am::struct_psp_gfx_cmd_resp AM_PSP::ring_submit(const am::struct_psp_gfx_cmd_resp& cmd) {
    am::struct_psp_gfx_rb_frame msg;
    memset(&msg, 0, sizeof msg);
    const uint32_t prev_wptr = adev.reg(reg_pref + "_67").read();
    msg.fence_value = prev_wptr + 1;
    msg.cmd_buf_addr_lo = (uint32_t)lo32(adev.paddr2mc(cmd_paddr));
    msg.cmd_buf_addr_hi = (uint32_t)hi32(adev.paddr2mc(cmd_paddr));
    msg.fence_addr_lo = (uint32_t)lo32(adev.paddr2mc(fence_paddr));
    msg.fence_addr_hi = (uint32_t)hi32(adev.paddr2mc(fence_paddr));
    adev.vram_write(cmd_paddr, &cmd, sizeof cmd);
    adev.vram_write(ring_paddr + (uint64_t)prev_wptr * 4, &msg, sizeof msg);
    // Move the wptr
    adev.reg(reg_pref + "_67").write(prev_wptr + sizeof(am::struct_psp_gfx_rb_frame) / 4);
    wait_cond([&] { return adev.vram_i(fence_paddr); }, msg.fence_value, 10000, "sOS ring not responding");
    am::struct_psp_gfx_cmd_resp resp;
    adev.vram_read(cmd_paddr, &resp, sizeof resp);
    if (resp.resp.status != 0)
        throw TGPyError("RuntimeError", "PSP command failed " + std::to_string(resp.cmd_id) + " " + std::to_string(resp.resp.status));
    return resp;
}
inline void AM_PSP::load_ip_fw_cmd(const AMFwDesc& d) {
    prep_msg1(d.p, d.n);
    for (uint32_t fw_type : d.types) {
        adev.log(std::string("loading fw: ") + psp_gfx_fw_type_name(fw_type));
        am::struct_psp_gfx_cmd_resp cmd;
        memset(&cmd, 0, sizeof cmd);
        cmd.cmd_id = am::GFX_CMD_ID_LOAD_IP_FW;
        cmd.cmd.cmd_load_ip_fw.fw_phy_addr_hi = (uint32_t)hi32(msg1_addr);
        cmd.cmd.cmd_load_ip_fw.fw_phy_addr_lo = (uint32_t)lo32(msg1_addr);
        cmd.cmd.cmd_load_ip_fw.fw_size = (uint32_t)d.n;
        cmd.cmd.cmd_load_ip_fw.fw_type = fw_type;
        ring_submit(cmd);
    }
}
inline am::struct_psp_gfx_cmd_resp AM_PSP::tmr_load_cmd() {
    const uint64_t tmr = tmr_paddr ? adev.paddr2xgmi(tmr_paddr) : 0;
    am::struct_psp_gfx_cmd_resp cmd;
    memset(&cmd, 0, sizeof cmd);
    cmd.cmd_id = am::GFX_CMD_ID_SETUP_TMR;
    const uint64_t mc = tmr_paddr ? adev.paddr2mc(tmr_paddr) : 0;
    cmd.cmd.cmd_setup_tmr.buf_phy_addr_hi = (uint32_t)hi32(mc);
    cmd.cmd.cmd_setup_tmr.buf_phy_addr_lo = (uint32_t)lo32(mc);
    cmd.cmd.cmd_setup_tmr.system_phy_addr_hi = (uint32_t)hi32(tmr);
    cmd.cmd.cmd_setup_tmr.system_phy_addr_lo = (uint32_t)lo32(tmr);
    cmd.cmd.cmd_setup_tmr.bitfield.set_virt_phy_addr(1);
    cmd.cmd.cmd_setup_tmr.buf_size = (uint32_t)(tmr_paddr ? tmr_size : 0);
    return ring_submit(cmd);
}
inline am::struct_psp_gfx_cmd_resp AM_PSP::load_toc_cmd(uint64_t toc_size) {
    am::struct_psp_gfx_cmd_resp cmd;
    memset(&cmd, 0, sizeof cmd);
    cmd.cmd_id = am::GFX_CMD_ID_LOAD_TOC;
    cmd.cmd.cmd_load_toc.toc_phy_addr_hi = (uint32_t)hi32(msg1_addr);
    cmd.cmd.cmd_load_toc.toc_phy_addr_lo = (uint32_t)lo32(msg1_addr);
    cmd.cmd.cmd_load_toc.toc_size = (uint32_t)toc_size;
    return ring_submit(cmd);
}
inline am::struct_psp_gfx_cmd_resp AM_PSP::rlc_autoload_cmd() {
    am::struct_psp_gfx_cmd_resp cmd;
    memset(&cmd, 0, sizeof cmd);
    cmd.cmd_id = am::GFX_CMD_ID_AUTOLOAD_RLC;
    return ring_submit(cmd);
}

} // namespace amboot
} // namespace tinygpu_device

#endif // LIBHMSBEAGLE_GPU_TINYGPUAMDBOOT_H
