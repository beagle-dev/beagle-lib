/*
 * TinyGPUHybridNVBoot.h -- TODO.md plan step C11 (level boot): tinygrad's NVDev boot (nvdev.py:75-162 and the IP blocks'
 * init_sw in ip.py, at the pin) in C++, ported statement by statement with the patches nv_init_helper.py applies to it in the
 * daemon, so that the plugin boots the GPU with no Python. Each part is golden-tested against the code it ports
 * (tinygpu_tests/golden_boot.py): the same requests to TinyGPU.app, byte for byte, the same results and the same errors.
 *
 * C11a, here: PCIIfaceBase.__init__'s BAR resize (system.py:263) and NVDev.__init__'s first statements (nvdev.py:75-80):
 * map_bar(0), then _early_ip_init under nv_init_helper's WARM guard, then _early_mmu_init with its BAR check, which builds
 * C6's memory manager (TinyGPUHybridNVMemory.h) as tinygrad builds its own. Ada's wait_for_reset is nv_init_helper's no-op
 * (tinygrad's polls a register the suppressed PCI reset would have set); the COT boot's (GB20x) waits for the FSP, logged.
 */

#ifndef LIBHMSBEAGLE_GPU_TINYGPUHYBRIDNVBOOT_H
#define LIBHMSBEAGLE_GPU_TINYGPUHYBRIDNVBOOT_H

#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <sys/stat.h>

#include "libhmsbeagle/GPU/TinyGPUFirmware.h"
#include "libhmsbeagle/GPU/TinyGPUHybridNVFalcon.h"
#include "libhmsbeagle/GPU/TinyGPUHybridNVGsp.h"
#include "libhmsbeagle/GPU/TinyGPUHybridNVMemory.h"
#include "libhmsbeagle/GPU/TinyGPUHybridNVProgram.h"
#include "libhmsbeagle/GPU/TinyGPUHybridNVRM.h"   // nv_bytes
#include "libhmsbeagle/GPU/TinyGPULog.h"

namespace tinygpu_device {

// NVDev's attributes of the same names, as the boot sets them. Registers go through NVReg on regs: before the chip is known
// the early reads use Ada's set, whose nv_ref, dev_fb and dev_gc6_island registers (the only ones included by then) GB20x's
// has at the same addresses.
struct NVBootDev {
    TGTransport* t = nullptr;
    const nv_regs::NVRegDef* regs = nv_regs::kAdaRegs;
    uint32_t chip_id = 0;                // NV_PMC_BOOT_0
    uint32_t architecture = 0, implementation = 0;   // NV_PMC_BOOT_42's chip_details
    std::string chip_name, fw_name;      // "AD107", "ad102"; "GB205", "gb202"
    int mmu_ver = 2;
    bool fmc_boot = false;               // the COT boot (NV_FLCN_COT)
    uint64_t vram_size = 0;
    uint64_t bar1_size = 0;              // self.vram.nbytes
    bool large_bar = false;
    bool recover = false;                // plan step P4: warm, with the GSP suspended or halted: nv_boot_recover runs first
    uint32_t warm_mailbox0 = 0, warm_cpuctl = 0, warm_wpr2_hi = 0;   // ... what nv_boot_early_ip_init read
    std::unique_ptr<NVMemState> mem = std::make_unique<NVMemState>();   // self.mm, on mem->dev (heap: its manager points at dev)

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
    nv_regs::NVReg<NVBootDev> reg(nv_regs::NVRegId id) { return nv_regs::NVReg<NVBootDev>(this, regs[id]); }
};

// PCIIfaceBase.__init__ (system.py:263): "with contextlib.suppress(Exception): self.pci_dev.resize_bar(vram_bar)", NV's
// vram_bar being 1; then NVDev.__init__'s map_bar(0) (nvdev.py:77), MAP_BAR's reply cached as bar_info caches it.
inline void nv_boot_pci(NVBootDev& d) {
    std::string err;
    d.t->resize_bar(1, err);   // an error is suppressed
    uint64_t addr = 0, size = 0;
    if (!d.t->bar_info(0, addr, size, err)) throw NVError("RuntimeError", err);
}

// TODO.md plan step P4, the default since its two clean recoveries on the RTX 4060 (plan decision 6, STATUS.md R92):
// BEAGLE_NV_RECOVER=0 turns it off
inline bool nv_recover_on() {
    const char* v = getenv("BEAGLE_NV_RECOVER");
    return !(v && strcmp(v, "0") == 0);
}

// NVDev._early_ip_init (nvdev.py:97-121), under nv_init_helper's _guarded_early_ip_init: WPR2 up means the previous boot was
// not torn down, refused before tinygrad's bus-master write (its own branch would issue a PCIe reset, a no-op on macOS, and
// a doomed boot). The includes are the register sets' concern; wait_for_reset is Ada's no-op or COT's FSP wait.
// Unless BEAGLE_NV_RECOVER=0 (plan step P4) the guard first reads, writing nothing, whether an Ada GPU's GSP is suspended
// (MAILBOX0 0x80000000, as the unload's suspend wait sees it) or its RISC-V core halted: then no GSP-RM runs, the previous
// session's teardown did not run or failed (BEAGLE_NV_TEARDOWN=0, say), and the boot goes on to run it (nv_boot_recover)
// instead of tinygrad's reset. Anything else is refused as before.
inline void nv_boot_early_ip_init(NVBootDev& d) {
    const uint32_t wpr2_hi = d.rreg(0x001FA828);   // _WPR2_ADDR_HI, read first by tinygrad too (nvdev.py:105)
    if (wpr2_hi != 0) {
        char hex[16];
        snprintf(hex, sizeof(hex), "%08x", wpr2_hi);
        const std::string up = std::string("WARM GPU: WPR2 is up (NV_PFB_PRI_MMU_WPR2_ADDR_HI=0x") + hex + ")";
        const std::string retry = "Power-cycle the eGPU (unplug and replug it) and retry. Nothing was written to the GPU.";
        if (!nv_recover_on()) throw NVError("WarmGPUError", up + ", so the previous boot was not torn down. " + retry);
        const char* teardown = getenv("BEAGLE_NV_TEARDOWN");
        if (teardown && strcmp(teardown, "0") == 0)
            throw NVError("WarmGPUError", up + ", and BEAGLE_NV_RECOVER runs NVIDIA's teardown, which BEAGLE_NV_TEARDOWN=0 turns off. " + retry);
        const uint32_t arch = (uint32_t)d.reg(nv_regs::NV_PMC_BOOT_42).read_bitfields()["architecture"];
        if (arch != 0x19) {
            char a[96];
            snprintf(a, sizeof(a), ", and BEAGLE_NV_RECOVER recovers Ada GPUs only (NV_PMC_BOOT_42 architecture 0x%x). ", arch);
            throw NVError("WarmGPUError", up + a + retry);
        }
        d.warm_wpr2_hi = wpr2_hi;
        d.warm_mailbox0 = d.reg(nv_regs::NV_PGSP_FALCON_MAILBOX0).read();
        const auto cpuctl = d.reg(nv_regs::NV_PRISCV_RISCV_CPUCTL).with_base(0x00110000);   // the GSP falcon's (NV_FLCN.falcon)
        d.warm_cpuctl = cpuctl.read();
        const bool suspended = d.warm_mailbox0 == 0x80000000, halted = cpuctl.decode(d.warm_cpuctl)["halted"] == 1;
        char g[200];
        snprintf(g, sizeof(g), "the GSP %s (MAILBOX0=0x%08x, RISCV_CPUCTL=0x%08x)", suspended ? "suspended" : halted ? "halted" :
                 "neither suspended nor halted", d.warm_mailbox0, d.warm_cpuctl);
        if (!suspended && !halted) throw NVError("WarmGPUError", up + ", with " + g + ": GSP-RM may still run, so BEAGLE_NV_RECOVER does not recover it. " + retry);
        d.recover = true;
        tg_log("%s, with %s: the boot runs NVIDIA's teardown once its images are ready (plan step P4; BEAGLE_NV_RECOVER=0 refuses instead)", up.c_str(), g);
    }
    if (d.reg(nv_regs::NV_PFB_PRI_MMU_WPR2_ADDR_HI).read() != 0 && !d.recover)
        throw NVError("RuntimeError", "WPR2 came up between two reads");   // tinygrad's reset branch, which the guard rules out
    std::string err;
    uint64_t cmd = 0;
    if (!d.t->read_config(0x04, 2, cmd, err) || !d.t->write_config_flush(0x04, cmd | 0x4, 2, err))   // PCI_COMMAND | PCI_COMMAND_MASTER
        throw NVError("RuntimeError", err);
    d.chip_id = d.reg(nv_regs::NV_PMC_BOOT_0).read();
    const nv_regs::NVFieldValues details = d.reg(nv_regs::NV_PMC_BOOT_42).read_bitfields();
    d.architecture = (uint32_t)details["architecture"];
    d.implementation = (uint32_t)details["implementation"];
    const char* family = d.architecture == 0x17 ? "GA1" : d.architecture == 0x19 ? "AD1" : d.architecture == 0x1b ? "GB2" : nullptr;
    if (!family) throw NVError("KeyError", std::to_string(d.architecture));   // the chip_name dict (nvdev.py:115)
    char impl[8];
    snprintf(impl, sizeof(impl), "%02u", d.implementation);
    d.chip_name = std::string(family) + impl;
    d.fw_name = d.architecture == 0x1b ? "gb202" : d.architecture == 0x19 ? "ad102" : "ga102";
    d.mmu_ver = d.architecture >= 0x1a ? 3 : 2;
    d.fmc_boot = d.architecture >= 0x1a;
    if (d.architecture == 0x17)   // plan decision 17: Ampere keeps tinygrad's code path, but C2 generated no tables for it
        throw NVError("RuntimeError", "the C++ boot has no register tables for " + d.chip_name + " (Ampere, plan decision 17): nothing was "
                      "written but the bus-master bit");
    d.regs = d.fmc_boot ? nv_regs::kGB20xRegs : nv_regs::kAdaRegs;
    if (!d.fmc_boot) return;   // NV_FLCN.wait_for_reset: nv_init_helper's no-op (the PCI reset is suppressed)
    // NV_FLCN_COT.wait_for_reset (ip.py:285-288) with nv_init_helper's log: the FSP takes the COT message once
    // NV_THERM_I2CS_SCRATCH reads 0xff. The lambda takes wait_cond's positional "waiting for reset" as its argument, so
    // wait_cond's own message is empty.
    const int64_t t0 = nv_now_ms();
    try {
        nv_wait_cond_true(10000, [&] { return d.reg(nv_regs::NV_THERM_I2CS_SCRATCH).read() == 0xff; }, "");
    } catch (const NVError& e) {
        if (e.type != "TimeoutError") throw;
        char msg[200];
        snprintf(msg, sizeof(msg), "FSP not ready: NV_THERM_I2CS_SCRATCH=0x%08x after %lld ms (0xff expected); the boot stopped before any FSP or "
                 "boot-memory access", d.reg(nv_regs::NV_THERM_I2CS_SCRATCH).read(), (long long)(nv_now_ms() - t0));
        tg_log("%s", msg);
        throw NVError("TimeoutError", msg);
    }
    tg_log("FSP ready: NV_THERM_I2CS_SCRATCH == 0xff after %lld ms", (long long)(nv_now_ms() - t0));
}

// NVDev._early_mmu_init (nvdev.py:123-147), then nv_init_helper's BAR check: BEAGLE's handoff and layout checks assume
// tinygrad's small-BAR branch, BAR1 exactly 256 MiB and smaller than VRAM. The check reads nothing; it comes after the
// memory manager zeroed its 4 KiB root page table through BAR1, as every boot does.
inline void nv_boot_early_mmu_init(NVBootDev& d) {
    d.vram_size = (uint64_t)d.reg(nv_regs::NV_PGC6_AON_SECURE_SCRATCH_GROUP_42).read() << 20;
    std::string err;
    uint64_t addr = 0;
    if (!d.t->bar_info(1, addr, d.bar1_size, err)) throw NVError("RuntimeError", err);   // map_bar(1); map_bar(0) is cached
    d.large_bar = d.bar1_size >= d.vram_size;
    NVMemState& st = *d.mem;
    st.dev.t = d.t;
    st.dev.mmu_ver = d.mmu_ver;
    st.dev.regs = d.regs;
    st.dev.is_booting = true;
    st.dev_vram_size = d.vram_size;
    st.va = TLSFAllocator(1ull << 44, 0x1000000000ull);   // NVMemoryManager.va_allocator
    const bool v3 = d.mmu_ver == 3;
    const std::vector<uint64_t> shifts = v3 ? std::vector<uint64_t>{12, 21, 29, 38, 47, 56} : std::vector<uint64_t>{12, 21, 29, 38, 47};
    try {   // tail VRAM reserved for falcon structs
        st.mm = std::make_unique<NVMemoryManager>(&st.dev, d.vram_size - (64ull << 20), 2ull << 20, v3 ? 56 : 48, shifts, 0,
                                                  std::vector<std::pair<uint64_t, uint64_t>>{{512ull << 20, 512ull << 20}, {2ull << 20, 2ull << 20},
                                                                                             {4ull << 10, 4ull << 10}},
                                                  0, !d.large_bar);
    } catch (const TGPyError& e) { throw NVError(e.type, e.what()); }
    st.mm->va_allocator = &st.va;
    if (d.bar1_size != 256ull << 20 || d.large_bar)
        throw NVError("BarLayoutError", "BAR1 is " + std::to_string(d.bar1_size >> 20) + " MiB with " + std::to_string(d.vram_size >> 20) +
                      " MiB of VRAM (large_bar=" + (d.large_bar ? "True" : "False") + "): BEAGLE supports only a 256 MiB BAR1 smaller than VRAM (plan "
                      "step B1). Nothing but tinygrad's 4 KiB root page table reached VRAM; no sysmem or firmware was set up.");
}

// NVDev.__init__ after _early_mmu_init (nvdev.py:82): "No booting state, gsp client is reinited every run."
inline void nv_boot_end_booting(NVBootDev& d) { d.mem->dev.is_booting = false; }

// ── C11b: NV_FLCN.init_sw (ip.py:97-106) with nv_init_helper's VBIOS capture and teardown images ──────────────────────────

// A byte buffer read the way prep_ucode and prep_booter read theirs: T.from_buffer_copy(b[off:]) needs sizeof(T) bytes
// (ctypes' ValueError otherwise), b[off:].cast('H')[0] an even, nonempty rest, b[off] an index in range, and b[a:][:n] is
// short or empty past the end. A negative offset (Python would count from the end) only comes from a malformed image.
struct NVPyBytes {
    const std::vector<uint8_t>& b;
    uint64_t rest(int64_t off) const {
        if (off < 0) throw NVError("ValueError", "a negative offset (" + std::to_string(off) + ") in a malformed image");
        return (uint64_t)off < b.size() ? b.size() - (uint64_t)off : 0;
    }
    template <class T> T at(int64_t off) const {
        const uint64_t have = rest(off);
        if (have < sizeof(T))
            throw NVError("ValueError", "Buffer size too small (" + std::to_string(have) + " instead of at least " + std::to_string(sizeof(T)) + " bytes)");
        T v;
        memcpy(&v, b.data() + off, sizeof(T));
        return v;
    }
    uint16_t u16(int64_t off) const {
        const uint64_t have = rest(off);
        if (have % 2) throw NVError("TypeError", "memoryview: length is not a multiple of itemsize");
        if (have == 0) throw NVError("IndexError", "index out of bounds on dimension 1");
        uint16_t v;
        memcpy(&v, b.data() + off, 2);
        return v;
    }
    uint8_t u8(int64_t off) const {
        if (rest(off) == 0) throw NVError("IndexError", "index out of bounds on dimension 1");
        return b[off];
    }
    std::vector<uint8_t> slice(int64_t a, uint64_t n) const {
        const uint64_t have = rest(a);
        return have ? std::vector<uint8_t>(b.begin() + a, b.begin() + a + std::min(n, have)) : std::vector<uint8_t>();
    }
};

// bytearray slice assignment, b[a:e] = data: the range (clamped to b) is replaced by data, which may resize b
inline void nv_py_setslice(std::vector<uint8_t>& b, uint64_t a, uint64_t e, const uint8_t* data, size_t n) {
    a = std::min<uint64_t>(a, b.size());
    e = std::min<uint64_t>(std::max(e, a), b.size());
    b.erase(b.begin() + a, b.begin() + e);
    b.insert(b.begin() + a, data, data + n);
}

inline bool tg_mkdirs(const std::string& dir) {   // Path.mkdir(parents=True, exist_ok=True)
    for (size_t i = 1; i <= dir.size(); ++i)
        if (i == dir.size() || dir[i] == '/') {
            const std::string part = dir.substr(0, i);
            if (mkdir(part.c_str(), 0755) != 0 && errno != EEXIST) return false;
        }
    struct stat st;
    return stat(dir.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

inline std::string nv_py_hex(uint64_t x) {   // hex(x)
    char s[24];
    snprintf(s, sizeof(s), "0x%llx", (unsigned long long)x);
    return s;
}

// FWSEC in the VBIOS: prep_ucode's walk (ip.py:110-145; nv_init_helper._fwsec_ucode), statement by statement. The last
// FWSEC_PROD entry of the last falcon-data token wins, as tinygrad's loops leave it. Where tinygrad would walk forever (an
// image of length 0 that is not the expansion ROM), this raises.
struct NVFwsecUcode {
    nv::FALCON_UCODE_DESC_V3 desc{};
    std::vector<uint8_t> signature, image;
};
inline NVFwsecUcode nv_fwsec_ucode(const std::vector<uint8_t>& vbios) {
    const NVPyBytes v{vbios};
    int64_t vbios_off = 0, expansion_rom_off = 0, block_size = -1;
    for (;;) {
        const uint16_t pci_blck = v.u16(vbios_off + nv::OFFSETOF_PCI_EXP_ROM_PCI_DATA_STRUCT_PTR);
        const int64_t imglen = (int64_t)v.u16(vbios_off + pci_blck + nv::OFFSETOF_PCI_DATA_STRUCT_IMAGE_LEN) * nv::PCI_ROM_IMAGE_BLOCK_SIZE;
        const uint8_t code_type = v.u8(vbios_off + pci_blck + nv::OFFSETOF_PCI_DATA_STRUCT_CODE_TYPE);
        if (code_type == nv::NV_BCRT_HASH_INFO_BASE_CODE_TYPE_VBIOS_BASE) block_size = imglen;
        else if (code_type == nv::NV_BCRT_HASH_INFO_BASE_CODE_TYPE_VBIOS_EXT) {
            if (block_size < 0) throw NVError("UnboundLocalError", "cannot access local variable 'block_size' where it is not associated with a value");
            expansion_rom_off = vbios_off - block_size;
            break;
        }
        if (imglen == 0) throw NVError("RuntimeError", "VBIOS walk: an image of length 0 at " + nv_py_hex(vbios_off) + " that is not the expansion ROM");
        vbios_off += imglen;
    }
    const int64_t bit_addr = 0x1b0;
    const auto bit_header = v.at<nv::struct_BIT_HEADER_V1_00>(bit_addr);
    if (bit_header.Signature != 0x00544942) throw NVError("AssertionError", "Invalid BIT header signature " + nv_py_hex(bit_header.Signature));
    bool found = false;
    int64_t ucode_desc_off = 0;
    uint32_t ucode_desc_size = 0;
    for (uint32_t i = 0; i < bit_header.TokenEntries; ++i) {
        const auto bit = v.at<nv::struct_BIT_TOKEN_V1_00>(bit_addr + bit_header.HeaderSize + (int64_t)i * bit_header.TokenSize);
        if (bit.TokenId != nv::BIT_TOKEN_FALCON_DATA || bit.DataVersion != 2 || bit.DataSize < nv::BIT_DATA_FALCON_DATA_V2_SIZE_4) continue;
        const auto falcon_data = v.at<nv::BIT_DATA_FALCON_DATA_V2>(bit.DataPtr & 0xffff);
        const int64_t table_ptr = expansion_rom_off + falcon_data.FalconUcodeTablePtr;
        const auto ucode_hdr = v.at<nv::FALCON_UCODE_TABLE_HDR_V1>(table_ptr);
        for (uint32_t j = 0; j < ucode_hdr.EntryCount; ++j) {
            const auto ucode_entry = v.at<nv::FALCON_UCODE_TABLE_ENTRY_V1>(table_ptr + ucode_hdr.HeaderSize + (int64_t)j * ucode_hdr.EntrySize);
            if (ucode_entry.ApplicationID != nv::FALCON_UCODE_ENTRY_APPID_FWSEC_PROD) continue;
            const auto ucode_desc_hdr = v.at<nv::FALCON_UCODE_DESC_HEADER>(expansion_rom_off + ucode_entry.DescPtr);
            ucode_desc_off = expansion_rom_off + ucode_entry.DescPtr;
            ucode_desc_size = ucode_desc_hdr.vDesc >> 16;
            found = true;
        }
    }
    if (!found) throw NVError("UnboundLocalError", "cannot access local variable 'ucode_desc_off' where it is not associated with a value");
    NVFwsecUcode u;
    const std::vector<uint8_t> desc = v.slice(ucode_desc_off, ucode_desc_size);   // vbios_bytes[off:off + size]
    u.desc = NVPyBytes{desc}.at<nv::FALCON_UCODE_DESC_V3>(0);
    u.signature = v.slice(ucode_desc_off + nv::FALCON_UCODE_DESC_V3_SIZE_44, ucode_desc_size - nv::FALCON_UCODE_DESC_V3_SIZE_44);
    u.image = v.slice(ucode_desc_off + ucode_desc_size, tg_round_up(u.desc.StoredSize, 256));
    return u;
}

// prep_ucode's __patch (ip.py:147-164; nv_init_helper._fwsec_patch) without its allocation: the DMEM mapper's init_cmd,
// the command and the signature's last 0x180 bytes, written into a copy of the image as bytearray slices are
inline std::vector<uint8_t> nv_fwsec_patch(const NVFwsecUcode& u, uint32_t cmd_id, const std::vector<uint8_t>& cmd) {
    std::vector<uint8_t> patched = u.image;
    const NVPyBytes img{u.image};
    uint32_t dmem_offset = 0;
    const int64_t app_hdr_off = (int64_t)u.desc.IMEMLoadSize + u.desc.InterfaceOffset;
    const auto hdr = img.at<nv::FALCON_APPLICATION_INTERFACE_HEADER_V1>(app_hdr_off);
    const int64_t ents_off = app_hdr_off + (int64_t)sizeof(hdr);
    const uint64_t need = (uint64_t)hdr.entryCount * sizeof(nv::FALCON_APPLICATION_INTERFACE_ENTRY_V1), have = img.rest(ents_off);
    if (have < need)   // (FALCON_APPLICATION_INTERFACE_ENTRY_V1 * entryCount).from_buffer_copy(...)
        throw NVError("ValueError", "Buffer size too small (" + std::to_string(have) + " instead of at least " + std::to_string(need) + " bytes)");
    for (uint32_t i = 0; i < hdr.entryCount; ++i) {
        const auto e = img.at<nv::FALCON_APPLICATION_INTERFACE_ENTRY_V1>(ents_off + (int64_t)i * (int64_t)sizeof(nv::FALCON_APPLICATION_INTERFACE_ENTRY_V1));
        if (e.id == nv::FALCON_APPLICATION_INTERFACE_ENTRY_ID_DMEMMAPPER) dmem_offset = e.dmemOffset;
    }
    const uint64_t dmem_mapper_offset = (uint64_t)u.desc.IMEMLoadSize + dmem_offset;
    auto dmem = img.at<nv::FALCON_APPLICATION_INTERFACE_DMEM_MAPPER_V3>((int64_t)dmem_mapper_offset);
    dmem.init_cmd = cmd_id;
    nv_py_setslice(patched, dmem_mapper_offset, dmem_mapper_offset + sizeof(dmem), (const uint8_t*)&dmem, sizeof(dmem));
    const uint64_t cmd_off = (uint64_t)u.desc.IMEMLoadSize + dmem.cmd_in_buffer_offset;
    nv_py_setslice(patched, cmd_off, cmd_off + cmd.size(), cmd.data(), cmd.size());
    const uint64_t sig_off = (uint64_t)u.desc.IMEMLoadSize + u.desc.PKCDataOffset;
    const size_t sn = std::min<size_t>(u.signature.size(), 0x180);   // signature[-0x180:]
    nv_py_setslice(patched, sig_off, sig_off + 0x180, u.signature.data() + u.signature.size() - sn, sn);
    return patched;
}

// NVDev._alloc_boot_mem (nvdev.py:149-160): TinyGPU.app sysmem (sysmem 1, or -1 on the small BAR), else VRAM from the
// memory manager, written through BAR1. paddr is the VRAM address (none for sysmem), sysaddr the pages' device addresses.
struct NVBootMem {
    bool vram = false;
    uint64_t paddr = 0;
    TGSysmem sys;   // the sysmem mapping (view, pages)
    std::vector<uint64_t> sysaddr;
};
inline NVBootMem nv_boot_alloc_mem(NVBootDev& d, uint64_t size, const void* data, bool contiguous = false, int sysmem = -1, int* keep_fd = nullptr) {
    NVBootMem m;
    const uint64_t sz = tg_round_up(size, 0x1000);
    std::string err;
    if (sysmem == 1 || (sysmem == -1 && !d.large_bar)) {
        if (!d.t->alloc_sysmem(size, contiguous, m.sys, err, keep_fd)) throw NVError("RuntimeError", err);
        m.sysaddr = m.sys.paddrs;
    } else {
        m.vram = true;
        try { m.paddr = d.mem->mm->palloc(sz, 0x1000, true, false); }
        catch (const TGPyError& e) { throw NVError(e.type, e.what()); }
        uint64_t bar1_addr = 0, bar1_size = 0;
        if (!d.t->bar_info(1, bar1_addr, bar1_size, err)) throw NVError("RuntimeError", err);
        for (uint64_t i = 0; i < sz / 0x1000; ++i) m.sysaddr.push_back(bar1_addr + m.paddr + i * 0x1000);
    }
    if (data) {   // view[:size] = data
        if (!m.vram) memcpy(m.sys.view, data, size);
        else if (!d.t->bulk_write(1, m.paddr, data, size, err)) throw NVError("RuntimeError", err);
    }
    return m;
}

// prep_booter's body (ip.py:171-184; nv_init_helper._booter_ucode) on C4's firmware file: the signed image and its load
// parameters (data offset and size, code offset and size)
struct NVBooterUcode {
    std::vector<uint8_t> image;
    uint32_t data_off = 0, data_sz = 0, code_off = 0, code_sz = 0;
};
inline NVBooterUcode nv_booter_ucode(const std::string& fw_name, const char* role) {
    const nvfw::TGFirmware* fw = tg_fw_entry(fw_name, role);
    if (!fw) throw NVError("KeyError", "'" + fw_name + "'");   // the sha dict
    TGFirmwareFile f;
    const std::string err = tg_fw_locate(*fw, f);
    if (!err.empty()) throw NVError("RuntimeError", err);
    const std::vector<uint8_t> b(f.data(), f.data() + f.size());
    const NVPyBytes v{b};
    const auto h = v.at<nv::struct_nvfw_bin_hdr>(0);
    const auto hs = v.at<nv::struct_nvfw_hs_header_v2>(h.header_offset);
    const auto lh = v.at<nv::struct_nvfw_hs_load_header_v2>(hs.header_offset);
    const auto app = v.at<nv::struct_nvfw_hs_load_header_v2_app>((int64_t)hs.header_offset + (int64_t)sizeof(nv::struct_nvfw_hs_load_header_v2));
    const uint32_t patch_loc = v.at<uint32_t>(hs.patch_loc), patch_sig = v.at<uint32_t>(hs.patch_sig), num_sig = v.at<uint32_t>(hs.num_sig);
    if (num_sig == 0) throw NVError("ZeroDivisionError", "integer division or modulo by zero");
    const uint64_t sig_len = hs.sig_prod_size / num_sig;
    const std::vector<uint8_t> sig = v.slice((int64_t)hs.sig_prod_offset + patch_sig, sig_len);
    NVBooterUcode u;
    u.image = v.slice(h.data_offset, h.data_size);
    nv_py_setslice(u.image, patch_loc, patch_loc + sig_len, sig.data(), sig.size());
    u.data_off = lh.os_data_offset;
    u.data_sz = lh.os_data_size;
    u.code_off = app.offset;
    u.code_sz = app.size;
    return u;
}

// NV_FLCN.init_sw (ip.py:97-106) as the daemon runs it: prep_ucode under nv_init_helper's VBIOS capture (the 1 MiB it reads,
// saved under $BEAGLE_TINYGPU_DATA/vbios), then prep_booter and, unless BEAGLE_NV_TEARDOWN=0, nv_init_helper's two teardown
// images right after (FWSEC-SB from the same VBIOS, Booter Unload), so FRTS and booter_load keep their addresses. The
// includes are the register sets' concern. What init_hw (plan step C9) and the teardown (C5) take goes to d.flcn and d.td.
inline void nv_boot_flcn_init_sw(NVBootDev& d, NVFlcnImages& flcn, NVTeardownImages& td) {
    // prep_ucode: the VBIOS window, one read (ip.py:110)
    std::vector<uint8_t> vbios(0x100000);
    std::string err;
    if (!d.t->bulk_read(0, 0x00300000, vbios.data(), vbios.size(), err)) throw NVError("RuntimeError", err);
    const NVFwsecUcode fwsec = nv_fwsec_ucode(vbios);
    flcn.frts_offset = d.vram_size - 0x100000 - 0x100000;
    nv::FWSECLIC_FRTS_CMD frts_cmd{};
    frts_cmd.readVbiosDesc.version = 0x1;
    frts_cmd.readVbiosDesc.size = sizeof(nv::FWSECLIC_READ_VBIOS_DESC);
    frts_cmd.readVbiosDesc.flags = 2;
    frts_cmd.frtsRegionDesc.version = 0x1;
    frts_cmd.frtsRegionDesc.size = sizeof(nv::FWSECLIC_FRTS_REGION_DESC);
    frts_cmd.frtsRegionDesc.frtsRegionOffset4K = (uint32_t)(flcn.frts_offset >> 12);
    frts_cmd.frtsRegionDesc.frtsRegionSize = 0x100;
    frts_cmd.frtsRegionDesc.frtsRegionMediaType = 2;
    const uint8_t* fc = (const uint8_t*)&frts_cmd;
    const std::vector<uint8_t> frts = nv_fwsec_patch(fwsec, nv570::FALCON_APPLICATION_INTERFACE_DMEM_MAPPER_V3_CMD_FRTS,
                                                     std::vector<uint8_t>(fc, fc + sizeof(frts_cmd)));
    flcn.frts_image_paddr = nv_boot_alloc_mem(d, frts.size(), frts.data(), false, 0).paddr;
    flcn.imem_pa = fwsec.desc.IMEMPhysBase; flcn.imem_va = fwsec.desc.IMEMVirtBase; flcn.imem_sz = fwsec.desc.IMEMLoadSize;
    flcn.dmem_pa = fwsec.desc.DMEMPhysBase; flcn.dmem_sz = fwsec.desc.DMEMLoadSize; flcn.pkc_off = fwsec.desc.PKCDataOffset;
    flcn.engid = fwsec.desc.EngineIdMask; flcn.ucodeid = fwsec.desc.UcodeId;
    // nv_init_helper's capture (_prep_ucode_with_vbios_capture): the bytes prep_ucode read, kept under their sha256
    const std::string digest = tg_sha256_hex(vbios.data(), vbios.size());
    const char* data_env = getenv("BEAGLE_TINYGPU_DATA");
    const char* home = getenv("HOME");
    const std::string dir = (data_env ? std::string(data_env) : std::string(home ? home : "") + "/.beagle/tinygpu") + "/vbios";
    const std::string path = dir + "/" + d.chip_name + "_" + digest.substr(0, 16) + ".rom";
    std::string saved;
    if (tg_mkdirs(dir)) {
        struct stat st;
        if (stat(path.c_str(), &st) != 0) {
            FILE* fp = fopen(path.c_str(), "wb");
            if (!fp || fwrite(vbios.data(), 1, vbios.size(), fp) != vbios.size()) saved = std::string("cannot write ") + path;
            if (fp && fclose(fp) != 0 && saved.empty()) saved = std::string("cannot write ") + path;
        }
    } else saved = "cannot create " + dir;
    if (saved.empty()) tg_log("VBIOS captured: %zu bytes, sha256 %s, saved to %s", vbios.size(), digest.c_str(), path.c_str());
    else tg_log("VBIOS captured (sha256 %s) but not saved: %s", digest.c_str(), saved.c_str());
    // prep_booter (ip.py:166-184)
    const NVBooterUcode load = nv_booter_ucode(d.fw_name, "booter_load");
    flcn.booter_image_paddr = nv_boot_alloc_mem(d, load.image.size(), load.image.data(), false, 0).paddr;
    flcn.booter_data_off = load.data_off; flcn.booter_data_sz = load.data_sz;
    flcn.booter_code_off = load.code_off; flcn.booter_code_sz = load.code_sz;
    // nv_init_helper's _prep_booter_with_teardown_images (plan step P2): FWSEC-SB takes READ_VBIOS_DESC alone
    const char* teardown = getenv("BEAGLE_NV_TEARDOWN");
    if ((teardown && strcmp(teardown, "0") == 0) || !tg_fw_entry(d.fw_name, "booter_unload")) return;
    const std::vector<uint8_t> rv(fc, fc + sizeof(nv::FWSECLIC_READ_VBIOS_DESC));   // frts_cmd.readVbiosDesc: the same fields
    const std::vector<uint8_t> sb = nv_fwsec_patch(fwsec, nv570::FALCON_APPLICATION_INTERFACE_DMEM_MAPPER_V3_CMD_SB, rv);
    td.sb_paddr = nv_boot_alloc_mem(d, sb.size(), sb.data(), false, 0).paddr;
    const NVBooterUcode unload = nv_booter_ucode(d.fw_name, "booter_unload");
    td.unload_paddr = nv_boot_alloc_mem(d, unload.image.size(), unload.image.data(), false, 0).paddr;
    td.sb_imem_pa = flcn.imem_pa; td.sb_imem_va = flcn.imem_va; td.sb_imem_sz = flcn.imem_sz; td.sb_dmem_pa = flcn.dmem_pa;
    td.sb_dmem_sz = flcn.dmem_sz; td.sb_pkc_off = flcn.pkc_off; td.sb_engid = flcn.engid; td.sb_ucodeid = flcn.ucodeid;
    td.unload_data_off = unload.data_off; td.unload_data_sz = unload.data_sz;
    td.unload_code_off = unload.code_off; td.unload_code_sz = unload.code_sz;
    td.present = true;
    tg_log("teardown images: FWSEC-SB %zu bytes at VRAM 0x%llx, Booter Unload %zu bytes at VRAM 0x%llx (code 0x%x+0x%x, data 0x%x+0x%x)", sb.size(),
           (unsigned long long)td.sb_paddr, unload.image.size(), (unsigned long long)td.unload_paddr, unload.code_off, unload.code_sz,
           unload.data_off, unload.data_sz);
}

// TODO.md plan step P4: NVIDIA's teardown at unload (NVFalcon::fini_hw, plan step P2: the GSP reset and FWSEC-SB, then, if WPR2
// is still up, the SEC2 reset and Booter Unload), here at boot, for the warm GPU nv_boot_early_ip_init let through, on this
// boot's images: after NV_FLCN.init_sw prepared them, before NV_GSP.init_sw first writes the GSP's queue registers. NVIDIA
// refuses such a GPU at load instead (kernel_gsp.c:3471-3476). WPR2 must then be down, or the boot stops (WarmGPUError),
// GSP-RM not started; diag says what the teardown did either way.
inline void nv_boot_recover(NVBootDev& d, const NVTeardownImages& td, NVFiniDiag& diag) {
    diag = NVFiniDiag();
    if (!td.present)
        throw NVError("WarmGPUError", "WARM GPU: BEAGLE_NV_RECOVER needs the teardown's images, which BEAGLE_NV_TEARDOWN=0 or a missing "
                      "booter_unload firmware leaves out. Power-cycle the eGPU (unplug and replug it) and retry.");
    NVBar0 bar0{d.t};
    NVFalcon flcn(bar0, d.chip_id);
    diag.unload_ok = true;   // the teardown's precondition: no GSP-RM runs (nv_boot_early_ip_init found it suspended or halted)
    flcn.fini_hw(diag, td);
    if (!diag.teardown_ok)
        throw NVError("WarmGPUError", "WARM GPU: NVIDIA's teardown at boot did not bring WPR2 down (" +
                      (diag.have_td_result ? diag.td_result : std::string("interrupted")) + "). Power-cycle the eGPU (unplug and replug it) and retry.");
}

// ── C11c: NV_GSP.init_sw (ip.py:347-455, 600-627); C11d: NV_FLCN_COT.init_sw (ip.py:290-310) ────────────────────────────

// A firmware file (C4's manifest and locator) as an ELF (C4's nvd_elf_load, as tinygrad's elf_loader), and a section's
// content by name: next(sh.content for sh in sections if sh.name == name), StopIteration if there is none.
struct NVBootElf {
    TGFirmwareFile file;
    NVDElf elf;
};
inline void nv_boot_fw(const std::string& fw_name, const char* role, TGFirmwareFile& f) {
    const nvfw::TGFirmware* fw = tg_fw_entry(fw_name, role);
    if (!fw) throw NVError("KeyError", "'" + fw_name + "'");
    const std::string err = tg_fw_locate(*fw, f);
    if (!err.empty()) throw NVError("RuntimeError", err);
}
inline void nv_boot_elf(const std::string& fw_name, const char* role, NVBootElf& e) {
    nv_boot_fw(fw_name, role, e.file);
    const std::string err = nvd_elf_load(e.file.data(), e.file.size(), 1, e.elf);
    if (!err.empty()) throw NVError("AssertionError", err);
}
inline std::vector<uint8_t> nv_elf_section(const NVBootElf& e, const std::string& name) {
    for (const auto& sh : e.elf.sections)
        if (sh.name == name) {
            const uint64_t n = sh.offset < e.file.size() ? std::min<uint64_t>(sh.size, e.file.size() - sh.offset) : 0;   // blob[off:off + size]
            return std::vector<uint8_t>(e.file.data() + sh.offset, e.file.data() + sh.offset + n);
        }
    throw NVError("StopIteration", "");
}
inline std::vector<uint32_t> nv_cast_u32(const std::vector<uint8_t>& b) {   // memoryview(b).cast('I')
    if (b.size() % 4) throw NVError("TypeError", "memoryview: length is not a multiple of itemsize");
    std::vector<uint32_t> w(b.size() / 4);
    memcpy(w.data(), b.data(), b.size());
    return w;
}
inline uint64_t nv_id8(const char* name) {   // int.from_bytes(bytes(name, 'utf-8'), 'big'), for names of at most 8 bytes
    uint64_t v = 0;
    for (const char* p = name; *p; ++p) v = (v << 8) | (uint8_t)*p;
    return v;
}

// What NV_GSP.init_sw leaves for init_hw (plan step C8), the RM client (C7) and the teardown (C5), in its attributes' names
struct NVGspBoot {
    uint64_t queue_size = 0x40000, pt_size = 0;
    NVBootMem queues;                 // init_rm_args: the page table, then the command queue and the status queue
    int queues_fd = -1;               // their TinyGPU.app fd, kept (the crash guard's, plan step C10)
    NVBootMem rm_args, logbuf, libos_args, radix3, signature, bootloader, wpr_meta;
    uint64_t rm_args_sysmem = 0, libos_args_sysmem = 0, wpr_meta_sysmem = 0, gsp_signature_bar1 = 0, booter_bar1 = 0;
    std::vector<uint64_t> gsp_radix3_addrs;
    nv::GspFwWprMeta meta{};          // self.wpr_meta's contents
    std::unique_ptr<NVRpcQueue> cmd_q;   // init_rm_args' NVRpcQueue, send-only until init_hw builds the GSP client on its seq
    uint32_t next_handle = 0xcf000000;   // handle_gen = itertools.count(0xcf000000)
    uint32_t gpfifo_class = 0, compute_class = 0, dma_class = 0, viddec_class = 0;
};

// init_rm_args (ip.py:362-386): the queues' sysmem with its page table, the rm args, the command queue's header
inline void nv_boot_init_rm_args(NVBootDev& d, NVGspBoot& g) {
    const uint64_t queue_size = g.queue_size, queue_pte_cnt = (queue_size * 2) / 0x1000;
    const uint64_t pte_cnt = queue_pte_cnt + tg_round_up(queue_pte_cnt * 8, 0x1000) / 0x1000;
    g.pt_size = tg_round_up(pte_cnt * 8, 0x1000);
    g.queues = nv_boot_alloc_mem(d, g.pt_size + queue_size * 2, nullptr, false, 1, &g.queues_fd);
    for (size_t i = 0; i < g.queues.sysaddr.size(); ++i) memcpy(g.queues.sys.view + i * 8, &g.queues.sysaddr[i], 8);   // the PTEs
    nv::GSP_ARGUMENTS_CACHED args{};
    args.bDmemStack = 1;
    args.messageQueueInitArguments.sharedMemPhysAddr = g.queues.sysaddr[0];
    args.messageQueueInitArguments.pageTableEntryCount = (uint32_t)pte_cnt;
    args.messageQueueInitArguments.cmdQueueOffset = g.pt_size;
    args.messageQueueInitArguments.statQueueOffset = g.pt_size + queue_size;
    g.rm_args = nv_boot_alloc_mem(d, sizeof(args), &args);
    g.rm_args_sysmem = g.rm_args.sysaddr[0];
    nv::msgqTxHeader h{};
    h.version = 0; h.size = (uint32_t)queue_size; h.entryOff = 0x1000; h.msgSize = 0x1000; h.msgCount = (uint32_t)((queue_size - 0x1000) / 0x1000);
    h.writePtr = 0; h.flags = 1; h.rxHdrOff = sizeof(nv::msgqTxHeader);
    memcpy(g.queues.sys.view + g.pt_size, &h, sizeof(h));
    g.cmd_q = std::make_unique<NVRpcQueue>(nullptr, g.queues.sys.view + g.pt_size, queue_size, nullptr, 10000);
    g.cmd_q->doorbell = [&d] { d.reg(nv_regs::NV_PGSP_QUEUE_HEAD)[0].write(0x0); };
}

// init_libos_args (ip.py:388-399): the five 64 KiB log buffers and the libos argument page, which points at the rm args
inline void nv_boot_init_libos_args(NVBootDev& d, NVGspBoot& g) {
    g.logbuf = nv_boot_alloc_mem(d, 2 << 20, nullptr);
    g.libos_args = nv_boot_alloc_mem(d, 0x1000, nullptr);
    g.libos_args_sysmem = g.libos_args.sysaddr[0];
    std::vector<nv::LibosMemoryRegionInitArgument> regions;
    const char* logs[] = {"LOGINIT", "LOGINTR", "LOGRM", "LOGMNOC", "LOGKRNL"};
    for (int i = 0; i < 5; ++i) {
        nv::LibosMemoryRegionInitArgument r{};
        r.kind = nv::LIBOS_MEMORY_REGION_CONTIGUOUS; r.loc = nv::LIBOS_MEMORY_REGION_LOC_SYSMEM; r.size = 0x10000;
        r.id8 = nv_id8(logs[i]); r.pa = g.logbuf.sysaddr[0] + 0x10000ull * i;
        regions.push_back(r);
    }
    nv::LibosMemoryRegionInitArgument rm{};
    rm.kind = nv::LIBOS_MEMORY_REGION_CONTIGUOUS; rm.loc = nv::LIBOS_MEMORY_REGION_LOC_SYSMEM; rm.size = 0x1000;
    rm.id8 = nv_id8("RMARGS"); rm.pa = g.rm_args_sysmem;
    regions.push_back(rm);
    memcpy(g.libos_args.sys.view, regions.data(), regions.size() * sizeof(regions[0]));
}

// init_wpr_meta (ip.py:431-448) with init_gsp_image (:400-423) and init_boot_binary_image (:424-430): the GSP image in
// radix3 with its signature, the bootloader, then the WPR meta (Ada's layout, whose FRTS region must be prep_ucode's; COT's)
inline void nv_boot_init_wpr_meta(NVBootDev& d, NVGspBoot& g, const NVFlcnImages* flcn) {
    NVBootElf gsp;
    nv_boot_elf(d.fw_name, "gsp", gsp);
    const std::vector<uint8_t> image = nv_elf_section(gsp, ".fwimage");
    std::string sig_name = ".fwsignature_" + d.chip_name.substr(0, 4) + "x";
    for (size_t i = 13; i < sig_name.size(); ++i) sig_name[i] = (char)tolower((unsigned char)sig_name[i]);
    const std::vector<uint8_t> signature = nv_elf_section(gsp, sig_name);
    uint64_t npages[4] = {0, 0, 0, tg_round_up(image.size(), 0x1000) / 0x1000};
    for (int i = 3; i > 0; --i) npages[i - 1] = ((npages[i] - 1) >> (nv::LIBOS_MEMORY_REGION_RADIX_PAGE_LOG2 - 3)) + 1;
    uint64_t offsets[4];
    for (int i = 0; i < 4; ++i) { offsets[i] = 0; for (int j = 0; j < i; ++j) offsets[i] += npages[j] * 0x1000; }
    g.radix3 = nv_boot_alloc_mem(d, offsets[3] + image.size(), nullptr);
    g.gsp_radix3_addrs = g.radix3.sysaddr;
    memcpy(g.radix3.sys.view + offsets[3], image.data(), image.size());
    for (int i = 0; i < 3; ++i) {   // each level lists the next level's pages
        uint64_t cur_offset = 0;
        for (int j = 0; j <= i; ++j) cur_offset += npages[j];
        memcpy(g.radix3.sys.view + offsets[i], g.gsp_radix3_addrs.data() + cur_offset, npages[i + 1] * 8);
    }
    g.signature = nv_boot_alloc_mem(d, signature.size(), signature.data());
    g.gsp_signature_bar1 = g.signature.sysaddr[0];
    // init_boot_binary_image: the bootloader and its RISC-V descriptor
    TGFirmwareFile bl;
    nv_boot_fw(d.fw_name, "bootloader", bl);
    const std::vector<uint8_t> b(bl.data(), bl.data() + bl.size());
    const NVPyBytes v{b};
    const auto h = v.at<nv::struct_nvfw_bin_hdr>(0);
    const std::vector<uint8_t> booter_image = v.slice(h.data_offset, h.data_size);
    const auto booter_desc = v.at<nv::RM_RISCV_UCODE_DESC>(h.header_offset);
    g.bootloader = nv_boot_alloc_mem(d, booter_image.size(), booter_image.data());
    g.booter_bar1 = g.bootloader.sysaddr[0];
    nv::GspFwWprMeta& m = g.meta;
    m = nv::GspFwWprMeta{};
    const uint64_t boot_sz = booter_image.size(), radix3_sz = image.size();
    m.sizeOfBootloader = boot_sz; m.sysmemAddrOfBootloader = g.booter_bar1;
    m.sizeOfRadix3Elf = radix3_sz; m.sysmemAddrOfRadix3Elf = g.gsp_radix3_addrs[0];
    m.sizeOfSignature = 0x1000; m.sysmemAddrOfSignature = g.gsp_signature_bar1;
    m.bootloaderCodeOffset = booter_desc.monitorCodeOffset; m.bootloaderDataOffset = booter_desc.monitorDataOffset;
    m.bootloaderManifestOffset = booter_desc.manifestOffset;
    m.revision = nv::GSP_FW_WPR_META_REVISION; m.magic = nv::GSP_FW_WPR_META_MAGIC;
    auto round_down = [](uint64_t x, uint64_t n) { return x / n * n; };
    if (d.fmc_boot) {
        m.vgaWorkspaceSize = 0x20000; m.pmuReservedSize = 0x1820000; m.nonWprHeapSize = 0x220000; m.gspFwHeapSize = 0x8700000;
        m.frtsSize = 0x100000;
    } else {
        const uint64_t vga_sz = 0x100000, vga_off = d.vram_size - vga_sz, frts_sz = 0x100000, frts_off = vga_off - frts_sz;
        const uint64_t boot_off = frts_off - boot_sz, gsp_off = round_down(boot_off - radix3_sz, 0x10000), gsp_heap_sz = 0x8100000;
        const uint64_t gsp_heap_off = round_down(gsp_off - gsp_heap_sz, 0x100000), wpr_st = round_down(gsp_heap_off - 0x1000, 0x100000);
        const uint64_t non_wpr_sz = 0x100000, non_wpr_off = round_down(wpr_st - non_wpr_sz, 0x100000);
        m.vgaWorkspaceSize = vga_sz; m.vgaWorkspaceOffset = vga_off; m.gspFwWprEnd = vga_off; m.frtsSize = frts_sz; m.frtsOffset = frts_off;
        m.bootBinOffset = boot_off; m.gspFwOffset = gsp_off; m.gspFwHeapSize = gsp_heap_sz; m.fbSize = d.vram_size;
        m.gspFwHeapOffset = gsp_heap_off; m.gspFwWprStart = wpr_st; m.nonWprHeapSize = non_wpr_sz; m.nonWprHeapOffset = non_wpr_off;
        m.gspFwRsvdStart = non_wpr_off;
        if (!flcn || flcn->frts_offset != m.frtsOffset)
            throw NVError("AssertionError", "FRTS mismatch: " + std::to_string(flcn ? flcn->frts_offset : 0) + " != " + std::to_string(m.frtsOffset));
    }
    g.wpr_meta = nv_boot_alloc_mem(d, sizeof(m), &m);
    g.wpr_meta_sysmem = g.wpr_meta.sysaddr[0];
}

// rpc_set_gsp_system_info (ip.py:600-608): the BARs' addresses (map_bar(3)'s first use), the PCI ids, on the queue
inline void nv_boot_rpc_set_gsp_system_info(NVBootDev& d, NVGspBoot& g) {
    uint64_t a0 = 0, a1 = 0, a3 = 0, sz = 0, id = 0, sub = 0, rev = 0;
    std::string err;
    nv::struct_GspSystemInfo data{};
    if (!d.t->bar_info(0, a0, sz, err) || !d.t->bar_info(1, a1, sz, err) || !d.t->bar_info(3, a3, sz, err)) throw NVError("RuntimeError", err);
    data.gpuPhysAddr = a0; data.gpuPhysFbAddr = a1; data.gpuPhysInstAddr = a3;
    data.pciConfigMirrorBase = d.fmc_boot ? 0x92000 : 0x88000; data.pciConfigMirrorSize = 0x1000;
    data.nvDomainBusDeviceFunc = 0;   // bdf_as_int("usb4")
    data.bIsPassthru = 1;
    if (!d.t->read_config(pci::PCI_VENDOR_ID, 4, id, err) || !d.t->read_config(pci::PCI_SUBSYSTEM_VENDOR_ID, 4, sub, err) ||
        !d.t->read_config(pci::PCI_REVISION_ID, 1, rev, err))
        throw NVError("RuntimeError", err);
    data.PCIDeviceID = (uint32_t)id; data.PCISubDeviceID = (uint32_t)sub; data.PCIRevisionID = (uint32_t)rev;
    data.maxUserVa = 0x7ffffffff000;
    g.cmd_q->send_rpc(nv::NV_VGPU_MSG_FUNCTION_GSP_SET_SYSTEM_INFO, nv_bytes(data));
}

// rpc_set_registry_table (ip.py:615-627): RMForcePcieConfigSave and RMSecBusResetEnable, 1 each
inline void nv_boot_rpc_set_registry_table(NVGspBoot& g) {
    const std::pair<const char*, uint32_t> table[] = {{"RMForcePcieConfigSave", 0x1}, {"RMSecBusResetEnable", 0x1}};
    const size_t hdr_size = sizeof(nv::struct_PACKED_REGISTRY_TABLE), entries_size = sizeof(nv::struct_PACKED_REGISTRY_ENTRY) * 2;
    std::vector<uint8_t> entries, data;
    for (const auto& kv : table) {
        nv::struct_PACKED_REGISTRY_ENTRY e{};
        e.nameOffset = (uint32_t)(hdr_size + entries_size + data.size());
        e.type = nv::REGISTRY_TABLE_ENTRY_TYPE_DWORD; e.data = kv.second; e.length = 4;
        const std::vector<uint8_t> eb = nv_bytes(e);
        entries.insert(entries.end(), eb.begin(), eb.end());
        data.insert(data.end(), kv.first, kv.first + strlen(kv.first) + 1);
    }
    nv::struct_PACKED_REGISTRY_TABLE header{};
    header.size = (uint32_t)(hdr_size + entries.size() + data.size());
    header.numEntries = 2;
    std::vector<uint8_t> msg = nv_bytes(header);
    msg.insert(msg.end(), entries.begin(), entries.end());
    msg.insert(msg.end(), data.begin(), data.end());
    g.cmd_q->send_rpc(nv::NV_VGPU_MSG_FUNCTION_SET_REGISTRY, msg);
}

// NV_GSP.init_sw (ip.py:347-363): the queues, the libos arguments, the WPR meta, the two prequeued RPCs, and the channel classes
inline void nv_boot_gsp_init_sw(NVBootDev& d, NVGspBoot& g, const NVFlcnImages* flcn) {
    nv_boot_init_rm_args(d, g);
    nv_boot_init_libos_args(d, g);
    nv_boot_init_wpr_meta(d, g, flcn);
    nv_boot_rpc_set_gsp_system_info(d, g);
    nv_boot_rpc_set_registry_table(g);
    g.gpfifo_class = nv_gpu::AMPERE_CHANNEL_GPFIFO_A; g.compute_class = nv_gpu::AMPERE_COMPUTE_B; g.dma_class = nv_gpu::AMPERE_DMA_COPY_B;
    const std::string fam = d.chip_name.substr(0, 2);
    g.viddec_class = fam == "AD" ? nv_gpu::NVC9B0_VIDEO_DECODER : fam == "GB" ? nv_gpu::NVCFB0_VIDEO_DECODER : 0;   // .get(): None
    if (fam == "AD") g.compute_class = nv_gpu::ADA_COMPUTE_A;
    else if (fam == "GB") { g.gpfifo_class = nv_gpu::BLACKWELL_CHANNEL_GPFIFO_A; g.compute_class = nv_gpu::BLACKWELL_COMPUTE_B; g.dma_class = nv_gpu::BLACKWELL_DMA_COPY_B; }
}

// NV_FLCN_COT.init_sw (ip.py:290-310) after nv_init_helper's include: the FMC boot parameters' page (zeroed, filled at
// init_hw), then init_fmc_image: the FMC's ELF32 sections, the image in sysmem, and its hash, signature and public key as
// 32-bit words (the key padded with 3 zero bytes first)
inline void nv_boot_cot_init_sw(NVBootDev& d, NVCotImages& cot, NVBootMem& fmc_args, NVBootMem& fmc_image) {
    const nv::struct_GSP_FMC_BOOT_PARAMS params{};
    fmc_args = nv_boot_alloc_mem(d, sizeof(params), &params);
    cot.fmc_boot_args = fmc_args.sys.view;
    cot.fmc_boot_args_sysmem = fmc_args.sysaddr[0];
    NVBootElf fmc;
    nv_boot_elf(d.fw_name, "fmc", fmc);
    const std::vector<uint8_t> image = nv_elf_section(fmc, "image");
    cot.hash = nv_cast_u32(nv_elf_section(fmc, "hash"));
    cot.sig = nv_cast_u32(nv_elf_section(fmc, "signature"));
    std::vector<uint8_t> pkey = nv_elf_section(fmc, "publickey");
    pkey.insert(pkey.end(), 3, 0);
    cot.pkey = nv_cast_u32(pkey);
    fmc_image = nv_boot_alloc_mem(d, image.size(), image.data());
    cot.fmc_booter_bar1 = fmc_image.sysaddr[0];
}

} // namespace tinygpu_device

#endif // LIBHMSBEAGLE_GPU_TINYGPUHYBRIDNVBOOT_H
