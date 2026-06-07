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
 * TinyGPU backend — speaks the TinyGPU.app daemon protocol (Unix domain
 * socket) to drive AMD or NVIDIA eGPUs connected via USB4/Thunderbolt on
 * macOS without requiring CUDA or ROCm kernel drivers.
 *
 * NVIDIA path: loads libcuda.so/dylib at runtime for channel management;
 *   delegates all kernel ops to CUDA driver.  TinyGPU provides device probe.
 *
 * AMD path: full bare-metal PM4 compute dispatch via direct BAR0/BAR2 MMIO
 *   writes.  Kernel compiled from OpenCL C to HSACO ELF via offline clang.
 *   No kernel driver required.
 *
 * Protocol reference: tinygrad/extra/usbgpu/tbgpu/installer/Shared/server.c
 *                     tinygrad/tinygrad/runtime/support/system.py (client)
 *                     tinygrad/tinygrad/runtime/support/am/ip.py  (AMD queue)
 *                     tinygrad/tinygrad/runtime/autogen/amd_gpu.py (registers)
 *
 * @author Marc Suchard
 */

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
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
#include <dlfcn.h>

#include "libhmsbeagle/beagle.h"
#include "libhmsbeagle/GPU/GPUImplDefs.h"
#include "libhmsbeagle/GPU/GPUImplHelper.h"
#include "libhmsbeagle/GPU/GPUInterface.h"
#include "libhmsbeagle/GPU/KernelResource.h"

// ── Kernel string macros (reuse CUDA PTX and OpenCL sources) ─────────────────
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
// §1  TinyGPU socket wire protocol
//     Reference: server.c — all structs packed, little-endian
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

static bool tgpu_rpc(int sock, uint32_t dev_id, TGCmd cmd,
                     uint64_t arg0, uint64_t arg1, uint64_t arg2,
                     uint32_t bar,
                     uint64_t* r0, uint64_t* r1) {
    tgpu_req_t req = {(uint8_t)cmd, dev_id, bar, arg0, arg1, arg2};
    sock_sendall(sock, &req, sizeof(req));
    tgpu_resp_t resp;
    sock_recvall(sock, &resp, sizeof(resp));
    if (r0) *r0 = resp.resp0;
    if (r1) *r1 = resp.resp1;
    return (resp.status == 0);
}

// MAP_SYSMEM_FD returns a file-descriptor via SCM_RIGHTS ancillary data.
static bool tgpu_rpc_fd(int sock, uint32_t dev_id,
                        uint64_t size, uint64_t contiguous,
                        uint64_t* mapped_size_out, int* fd_out) {
    tgpu_req_t req = {CMD_MAP_SYSMEM_FD, dev_id, 0, size, contiguous, 0};
    sock_sendall(sock, &req, sizeof(req));

    char cmsg_buf[CMSG_SPACE(sizeof(int))];
    tgpu_resp_t resp = {};
    struct iovec iov = {&resp, sizeof(resp)};
    struct msghdr msg = {};
    msg.msg_iov = &iov; msg.msg_iovlen = 1;
    msg.msg_control = cmsg_buf; msg.msg_controllen = sizeof(cmsg_buf);

    if (recvmsg(sock, &msg, 0) != (ssize_t)sizeof(resp)) {
        perror("TinyGPU recvmsg"); exit(1);
    }
    *fd_out = -1;
    if (msg.msg_controllen > 0) {
        struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
        if (cmsg && cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS)
            memcpy(fd_out, CMSG_DATA(cmsg), sizeof(int));
    }
    if (mapped_size_out) *mapped_size_out = resp.resp0;
    return (resp.status == 0);
}

// Bulk BAR write via CMD_MMIO_WRITE — no response packet from server.
static void tgpu_mmio_write(int sock, uint32_t dev_id, uint32_t bar,
                            uint64_t offset, const void* data, size_t size) {
    tgpu_req_t req = {CMD_MMIO_WRITE, dev_id, bar, offset, (uint64_t)size, 0};
    sock_sendall(sock, &req, sizeof(req));
    sock_sendall(sock, data, size);
}

// Bulk BAR read via CMD_MMIO_READ — server replies with 17-byte header then data.
static void tgpu_mmio_read(int sock, uint32_t dev_id, uint32_t bar,
                           uint64_t offset, void* data, size_t size) {
    uint64_t got = 0;
    tgpu_rpc(sock, dev_id, CMD_MMIO_READ, offset, (uint64_t)size, 0, bar, &got, nullptr);
    sock_recvall(sock, data, (size_t)got);
}

// 32-bit BAR0 register helpers.
static void bar0_wr32(int sock, uint32_t dev_id, uint32_t reg_dw, uint32_t val) {
    tgpu_mmio_write(sock, dev_id, 0, (uint64_t)reg_dw * 4, &val, 4);
}
static uint32_t bar0_rd32(int sock, uint32_t dev_id, uint32_t reg_dw) {
    uint32_t v = 0;
    tgpu_mmio_read(sock, dev_id, 0, (uint64_t)reg_dw * 4, &v, 4);
    return v;
}

static uint64_t cfg_read(int sock, uint32_t dev_id, uint64_t off, uint64_t sz) {
    uint64_t v = 0;
    tgpu_rpc(sock, dev_id, CMD_CFG_READ, off, sz, 0, 0, &v, nullptr);
    return v;
}
static void cfg_write(int sock, uint32_t dev_id, uint64_t off, uint64_t sz, uint64_t val) {
    tgpu_rpc(sock, dev_id, CMD_CFG_WRITE, off, sz, val, 0, nullptr, nullptr);
}

// §2  NVIDIA CUDA driver — loaded inline in nvSetup() since NVCuda is private.

// ─────────────────────────────────────────────────────────────────────────────
// §3  AMD GPU constants (GFX10/11 — RDNA2/3)
//     All register indices are DWORD offsets from BAR0 base (byte = index × 4).
//     Sources: tinygrad/tinygrad/runtime/autogen/amd_gpu.py
//              tinygrad/tinygrad/runtime/autogen/am/regs.py (field bit positions)
//              tinygrad/tinygrad/runtime/support/am/ip.py  (boot sequence)
// ─────────────────────────────────────────────────────────────────────────────

// ── AM boot detection ─────────────────────────────────────────────────────────
static const uint32_t AMD_DEV_VERSION     = 0xA0000008;  // tinygrad AM signature in SCRATCH_REG7
static const uint32_t AMD_SCRATCH_REG6   = 0x2046;
static const uint32_t AMD_SCRATCH_REG7   = 0x2047;

// ── VRAM layout ───────────────────────────────────────────────────────────────
// Must leave space at offset 0 for page tables (2MB) before kernel binaries.
static const uint64_t AMD_VRAM_PT_SIZE    = 2ULL << 20;    // page table region
// AMD_VRAM_KERNEL_BASE, AMD_VRAM_RING_BASE, AMD_VRAM_DATA_BASE follow below.

// ── Frame buffer location ─────────────────────────────────────────────────────
static const uint32_t AMD_FB_LOCATION_BASE = 0x1678;  // (read & 0xFFFFFF) << 24 = fb_base
static const uint32_t AMD_FB_LOCATION_TOP  = 0x1679;

// ── GMC (Graphics Memory Controller) VM registers ────────────────────────────
static const uint32_t AMD_GCMC_VM_AGP_TOP          = 0x167a;
static const uint32_t AMD_GCMC_VM_AGP_BOT          = 0x167b;
static const uint32_t AMD_GCMC_VM_AGP_BASE         = 0x167c;
static const uint32_t AMD_GCMC_VM_SYS_APE_LOW      = 0x167d;
static const uint32_t AMD_GCMC_VM_SYS_APE_HIGH     = 0x167e;
static const uint32_t AMD_GCMC_VM_MX_L1_TLB_CNTL  = 0x167f;
static const uint32_t AMD_GCMC_VM_SYS_APE_DEF_LSB  = 0x15a8;
static const uint32_t AMD_GCMC_VM_SYS_APE_DEF_MSB  = 0x15a9;
static const uint32_t AMD_GCVM_L2_CNTL             = 0x15bc;
static const uint32_t AMD_GCVM_L2_CNTL2            = 0x15bd;
static const uint32_t AMD_GCVM_L2_CNTL3            = 0x15be;
static const uint32_t AMD_GCVM_L2_PROT_FAULT_CNTL2 = 0x15c5;
static const uint32_t AMD_GCVM_L2_PROT_FAULT_DEF_LO = 0x15cb;
static const uint32_t AMD_GCVM_L2_PROT_FAULT_DEF_HI = 0x15cc;
static const uint32_t AMD_GCVM_IDENTITY_APE_LO_LO32 = 0x15ce;
static const uint32_t AMD_GCVM_IDENTITY_APE_LO_HI32 = 0x15cf;
static const uint32_t AMD_GCVM_IDENTITY_APE_HI_LO32 = 0x15d0;
static const uint32_t AMD_GCVM_IDENTITY_APE_HI_HI32 = 0x15d1;
static const uint32_t AMD_GCVM_IDENTITY_PHYS_LO    = 0x15d2;
static const uint32_t AMD_GCVM_IDENTITY_PHYS_HI    = 0x15d3;
static const uint32_t AMD_GCVM_L2_CNTL4            = 0x15d4;
static const uint32_t AMD_GCVM_L2_CNTL5            = 0x15da;
// VM_CONTEXT0 — DWORD indices for the GC hub
static const uint32_t AMD_GCVM_CTX0_CNTL           = 0x1688;
static const uint32_t AMD_GCVM_CTX0_PT_BASE_LO     = 0x16f3;
static const uint32_t AMD_GCVM_CTX0_PT_BASE_HI     = 0x16f4;
static const uint32_t AMD_GCVM_CTX0_PT_START_LO    = 0x1713;
static const uint32_t AMD_GCVM_CTX0_PT_START_HI    = 0x1714;
static const uint32_t AMD_GCVM_CTX0_PT_END_LO      = 0x1733;
static const uint32_t AMD_GCVM_CTX0_PT_END_HI      = 0x1734;
// TLB invalidation via ENG17 (GC compute hub)
static const uint32_t AMD_GCVM_INVAL_ENG17_SEM     = 0x16aa;
static const uint32_t AMD_GCVM_INVAL_ENG17_REQ     = 0x16bc;
static const uint32_t AMD_GCVM_INVAL_ENG17_ACK     = 0x16ce;
// ENG_n_ADDR_RANGE_LO = AMD_GCVM_INVAL_ENG_ADDR_BASE + n*2
// ENG_n_ADDR_RANGE_HI = AMD_GCVM_INVAL_ENG_ADDR_BASE + n*2 + 1
static const uint32_t AMD_GCVM_INVAL_ENG_ADDR_BASE = 0x16cf; // ENG0_LO = 0x16cf

// ── GFX compute engine registers ─────────────────────────────────────────────
static const uint32_t AMD_GRBM_CNTL                = 0x0da0;  // read_timeout[7:0]
static const uint32_t AMD_GRBM_SOFT_RESET          = 0x0da8;  // soft_reset_cp[0]|soft_reset_cpc[18]
static const uint32_t AMD_CP_MEC_RS64_CNTL         = 0x2904;  // mec_pipe0_active[26]|mec_halt[30]
static const uint32_t AMD_RLC_CNTL                 = 0x4c00;  // rlc_enable_f32[0]
static const uint32_t AMD_RLC_SRM_CNTL             = 0x4c80;  // srm_enable[0]|auto_incr_addr[1]
static const uint32_t AMD_SH_MEM_CONFIG             = 0x09e4;  // address_mode[0]
static const uint32_t AMD_SH_MEM_BASES             = 0x09e3;  // private_base[15:0]|shared_base[31:16]
static const uint32_t AMD_CP_MEC_DOORBELL_LOWER    = 0x1dfc;  // doorbell_range_lower[11:2]
static const uint32_t AMD_CP_MEC_DOORBELL_UPPER    = 0x1dfd;

// ── Pre-computed register values ──────────────────────────────────────────────
// VM_CONTEXT0_CNTL: enable_context[0]=1, page_table_depth[2:1]=3 (4-level),
//   page_table_block_size[6:3]=0, range/dummy/pde0/valid/read/write/execute
//   fault interrupt+default[22:9]=all-1, secure fault bits[24:23] (GFX11 compat)
static const uint32_t AMD_GCVM_CTX0_CNTL_4LVL    = 0x01FFFE07;
// L2 CNTL[0]=enable_l2_cache, [11]=enable_default_page_out, [20:19]=context1_identity=1
static const uint32_t AMD_GCVM_L2_CNTL_ENABLE     = 0x00080801;
static const uint32_t AMD_GCVM_L2_CNTL2_FLUSH     = 0x00000003;  // invalidate L1+L2
// L2_CNTL3: bank_select[5:0]=9, l2_bigk_frag_size[19:15]=6, bigk_assoc[20]=1, 4k_assoc[31]=1
static const uint32_t AMD_GCVM_L2_CNTL3_GFX10     = 0x80130009;
static const uint32_t AMD_GCVM_L2_CNTL4_VAL       = 0x00000001;  // l2_4k_partition_count=1
static const uint32_t AMD_GCVM_L2_CNTL5_VAL       = 0x00003FE0;  // walker_priority_client_id=0x1ff
// MX_L1_TLB_CNTL: enable_l1_tlb[0]=1, system_access_mode[4:3]=3, enable_adv_driver[6]=1, mtype[13:11]=3(UC)
static const uint32_t AMD_GCMC_MX_L1_TLB_CNTL_VAL = 0x00001859;
static const uint32_t AMD_GCVM_L2_PROT_CNTL2_VAL  = 0x00040000;  // active_page_migration_pte_read_retry[18]
// GRBM_SOFT_RESET: soft_reset_cp[0]=1, soft_reset_cpc[18]=1
static const uint32_t AMD_GRBM_SOFT_RESET_MEC      = 0x00040001;
// CP_MEC_RS64_CNTL: mec_pipe0_active[26]=1, all resets/halt=0
static const uint32_t AMD_CP_MEC_RS64_ENABLE       = 0x04000000;
// TLB invalidation REQ for VMID 0: per_vmid[0]=1, invalidate_l2_ptes[16]=1,
//   invalidate_l2_pde0..2[17-19]=1, invalidate_l1_ptes[20]=1
static const uint32_t AMD_TLB_INVAL_VMID0_FULL     = 0x001F0001;
// PTE flags: VALID[0]=1, EXECUTE[6]=1, READ[7]=1, WRITE[8]=1
static const uint64_t AMD_PTE_FLAGS               = 0x1C1ULL;
// PDE flags: VALID[0]=1 (intermediate page table entry)
static const uint64_t AMD_PDE_FLAGS               = 0x1ULL;

// GRBM_GFX_CNTL — selects MEC/pipe/queue context for subsequent MMIO reg access
// Bit fields: pipeid[1:0], meid[3:2], vmid[7:4], queueid[10:8]
static const uint32_t AMD_GRBM_GFX_CNTL   = 0x0900;

// CP_HQD (Hardware Queue Descriptor) — accessed after GRBM context set
static const uint32_t AMD_CP_HQD_ACTIVE                    = 0x1fab;
static const uint32_t AMD_CP_HQD_PQ_BASE                   = 0x1fb1;
static const uint32_t AMD_CP_HQD_PQ_BASE_HI                = 0x1fb2;
static const uint32_t AMD_CP_HQD_PQ_RPTR_REPORT_ADDR       = 0x1fb4;
static const uint32_t AMD_CP_HQD_PQ_RPTR_REPORT_ADDR_HI    = 0x1fb5;
static const uint32_t AMD_CP_HQD_PQ_WPTR_POLL_ADDR         = 0x1fb6;
static const uint32_t AMD_CP_HQD_PQ_WPTR_POLL_ADDR_HI      = 0x1fb7;
static const uint32_t AMD_CP_HQD_PQ_DOORBELL_CONTROL       = 0x1fb8;
static const uint32_t AMD_CP_HQD_PQ_CONTROL                = 0x1fba;

// COMPUTE registers — set via MMIO within the GRBM-selected context
static const uint32_t AMD_COMPUTE_DISPATCH_INITIATOR        = 0x1ba0;
static const uint32_t AMD_COMPUTE_NUM_THREAD_X              = 0x1ba7;
static const uint32_t AMD_COMPUTE_NUM_THREAD_Y              = 0x1ba8;
static const uint32_t AMD_COMPUTE_NUM_THREAD_Z              = 0x1ba9;
static const uint32_t AMD_COMPUTE_PGM_LO                   = 0x1bac;
static const uint32_t AMD_COMPUTE_PGM_HI                   = 0x1bad;
static const uint32_t AMD_COMPUTE_PGM_RSRC1                 = 0x1bb2;
static const uint32_t AMD_COMPUTE_PGM_RSRC2                 = 0x1bb3;
static const uint32_t AMD_COMPUTE_TMPRING_SIZE              = 0x1bb8;
static const uint32_t AMD_COMPUTE_PGM_RSRC3                 = 0x1bc8;
static const uint32_t AMD_COMPUTE_USER_DATA_0               = 0x1be0;

// PM4 packet construction
// type-3 packet with compute bit (bit 1 = 1 selects MEC path)
static inline uint32_t pm4_pkt3(uint32_t op, uint32_t ndw) {
    return (3u << 30) | ((op & 0xFF) << 8) | (((ndw - 1) & 0x3FFF) << 16) | (1u << 1);
}

static const uint32_t PM4_DISPATCH_DIRECT  = 0x15;
static const uint32_t PM4_RELEASE_MEM      = 0x49;
static const uint32_t PM4_EVENT_WRITE      = 0x46;

// event_type and event_index for end-of-pipe semaphore signal
static const uint32_t AMD_EOP_EVENT_TYPE   = 20;  // CACHE_FLUSH_AND_INV_TS_EVENT
static const uint32_t AMD_EOP_EVENT_INDEX  = 5;   // end_of_pipe
static const uint32_t AMD_CS_PARTIAL_FLUSH = 7;   // CS_PARTIAL_FLUSH event type
static const uint32_t AMD_EOP_FLUSH_IDX    = 4;   // event_index for CS_PARTIAL_FLUSH

// Doorbell: Navi10 (RDNA2) MEC_RING0 doorbell index = 3
// ip.py: doorbell_offset in CP_HQD_PQ_DOORBELL_CONTROL = doorbell_index * 2
// BAR2 (doorbell64) byte offset = (doorbell_index * 2) * 8
static const uint32_t AMD_NAVI10_DOORBELL_MEC_RING0 = 3;

// VRAM layout for AMD path (page tables occupy the first 2 MB)
static const uint64_t AMD_VRAM_KERNEL_BASE = AMD_VRAM_PT_SIZE;  // 2 MB: kernel binaries
static const uint64_t AMD_VRAM_RING_BASE   = 32ULL << 20;       // 32 MB: compute ring
static const uint64_t AMD_VRAM_DATA_BASE   = 256ULL << 20;      // 256 MB: data buffers

static const uint32_t AMD_RING_SIZE        = 1u << 20;  // 1 MB compute ring

// AMD HSACO ELF kernel descriptor (LLVM AMDHSA spec, 64 bytes at .rodata)
struct AMDKernelDesc {
    uint32_t group_segment_fixed_size;    // LDS bytes
    uint32_t private_segment_fixed_size;  // scratch bytes per thread
    uint32_t kernarg_size;                // total kernel args size in bytes
    uint8_t  reserved0[4];
    int64_t  kernel_code_entry_byte_offset;  // offset from desc to kernel code
    uint8_t  reserved1[20];
    uint32_t compute_pgm_rsrc3;
    uint32_t compute_pgm_rsrc1;
    uint32_t compute_pgm_rsrc2;
    uint16_t kernel_code_properties;     // bit 10 = wave32
    uint16_t kernarg_preload_spec;
    uint8_t  reserved2[6];
};  // total = 64 bytes

// ─────────────────────────────────────────────────────────────────────────────
// §4  PCI constants
// ─────────────────────────────────────────────────────────────────────────────
static const uint16_t PCI_VENDOR_NVIDIA = 0x10DE;
static const uint16_t PCI_VENDOR_AMD    = 0x1002;

// ─────────────────────────────────────────────────────────────────────────────
// §5  Per-device state structures
// ─────────────────────────────────────────────────────────────────────────────

struct DeviceInfo {
    uint32_t dev_id;
    uint16_t pci_vendor;
    uint16_t pci_device;
    bool     supports_dp;
};

struct PinnedBuf {
    void*  host_ptr;
    size_t mapped_size;
    int    fd;
};

// Per-kernel entry (vendor-neutral)
struct KernelEntry {
    uint64_t    code_vaddr;    // GPU virtual address (NV=CUdeviceptr) or VRAM offset (AMD)
    uint32_t    rsrc1;         // COMPUTE_PGM_RSRC1 (AMD)
    uint32_t    rsrc2;         // COMPUTE_PGM_RSRC2 (AMD)
    uint32_t    rsrc3;         // COMPUTE_PGM_RSRC3 (AMD)
    bool        wave32;        // AMD: wave32 mode
    std::string name;
    void*       cu_func;       // CUfunction (NV)
};

// ─────────────────────────────────────────────────────────────────────────────
// §6  GPUInterface implementation
// ─────────────────────────────────────────────────────────────────────────────

// ── ELF helpers ──────────────────────────────────────────────────────────────

// Minimal ELF64 structs for parsing cubin and HSACO files.
struct Elf64_Ehdr {
    uint8_t  e_ident[16];
    uint16_t e_type, e_machine;
    uint32_t e_version;
    uint64_t e_entry, e_phoff, e_shoff;
    uint32_t e_flags, e_ehsize;
    uint16_t e_phentsize, e_phnum, e_shentsize, e_shnum, e_shstrndx;
};
struct Elf64_Shdr {
    uint32_t sh_name, sh_type;
    uint64_t sh_flags, sh_addr, sh_offset, sh_size;
    uint32_t sh_link, sh_info;
    uint64_t sh_addralign, sh_entsize;
};

// Find section by name; return (data_ptr, size) or (nullptr, 0).
static const uint8_t* elf_section(const uint8_t* elf, size_t elf_sz,
                                   const char* name, uint64_t* sec_off_out) {
    if (elf_sz < sizeof(Elf64_Ehdr)) return nullptr;
    const Elf64_Ehdr* eh = (const Elf64_Ehdr*)elf;
    if (eh->e_shoff + (uint64_t)eh->e_shnum * eh->e_shentsize > elf_sz) return nullptr;

    const Elf64_Shdr* shdrs = (const Elf64_Shdr*)(elf + eh->e_shoff);
    const char* strtab = (const char*)(elf + shdrs[eh->e_shstrndx].sh_offset);

    for (uint16_t i = 0; i < eh->e_shnum; ++i) {
        if (strcmp(strtab + shdrs[i].sh_name, name) == 0) {
            if (sec_off_out) *sec_off_out = shdrs[i].sh_offset;
            return elf + shdrs[i].sh_offset;
        }
    }
    return nullptr;
}

// ── GPUInterface ctor/dtor ───────────────────────────────────────────────────

GPUInterface::GPUInterface()
    : numStreams(1),
      tgpuSock(-1), tgpuDevId(0),
      isNVIDIA(false),
      vramDataTop(AMD_VRAM_DATA_BASE),
      vramKernelTop(AMD_VRAM_KERNEL_BASE),
      amdRingWptr(0), amdEopSignal(0),
      amdFbBase(0), amdPartialBoot(false),
      kernelResource(nullptr), resourceMap(nullptr),
      supportDoublePrecision(true)
{
    memset(tgpuBars, 0, sizeof(tgpuBars));
    memset(&nvCuda, 0, sizeof(nvCuda));
    nvCtx = nullptr; nvModule = nullptr;
    amdCompletionHost = nullptr; amdCompletionFd = -1;
}

GPUInterface::~GPUInterface() {
    for (auto& pb : tgpuPinned) {
        if (pb.host_ptr && pb.mapped_size) munmap(pb.host_ptr, pb.mapped_size);
        if (pb.fd >= 0) close(pb.fd);
    }
    if (amdCompletionHost && amdCompletionMapped) {
        munmap(amdCompletionHost, amdCompletionMapped);
        if (amdCompletionFd >= 0) close(amdCompletionFd);
    }
    for (auto& kv : tgpuKernels) delete kv.second;
    if (kernelResource) delete kernelResource;
    if (resourceMap) delete resourceMap;
    if (nvCuda.lib) dlclose(nvCuda.lib);
    if (tgpuSock >= 0) close(tgpuSock);
}

// ── Initialize ───────────────────────────────────────────────────────────────

int GPUInterface::Initialize() {
    resourceMap = new std::map<int, int>;

    const char* sock_path = getenv("APL_REMOTE_SOCK");
    char default_path[256];
    if (!sock_path) {
        const char* tmp = getenv("TMPDIR");
        if (!tmp) tmp = "/tmp";
        snprintf(default_path, sizeof(default_path), "%s/tinygpu.sock", tmp);
        sock_path = default_path;
    }

    tgpuSock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (tgpuSock < 0) return 0;

    struct sockaddr_un addr;
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, sock_path, sizeof(addr.sun_path) - 1);
    addr.sun_path[sizeof(addr.sun_path) - 1] = '\0';

    bool connected = false;
    for (int i = 0; i < 100 && !connected; ++i) {
        if (connect(tgpuSock, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
            connected = true;
        } else {
            if (i == 0) {
                const char* app = "/Applications/TinyGPU.app/Contents/MacOS/TinyGPU";
                pid_t pid = fork();
                if (pid == 0) { execlp(app, app, "server", sock_path, nullptr); _exit(1); }
            }
            usleep(50000);
        }
    }
    if (!connected) {
        fprintf(stderr, "TinyGPU: could not connect to %s\n", sock_path);
        close(tgpuSock); tgpuSock = -1; return 0;
    }

    int bufsize = 64 * 1024 * 1024;
    setsockopt(tgpuSock, SOL_SOCKET, SO_SNDBUF, &bufsize, sizeof(bufsize));
    setsockopt(tgpuSock, SOL_SOCKET, SO_RCVBUF, &bufsize, sizeof(bufsize));

    uint16_t vendor = (uint16_t)cfg_read(tgpuSock, 0, 0x00, 2);
    uint16_t device = (uint16_t)cfg_read(tgpuSock, 0, 0x02, 2);

    if (vendor != PCI_VENDOR_NVIDIA && vendor != PCI_VENDOR_AMD) {
        fprintf(stderr, "TinyGPU: unrecognized vendor 0x%04x\n", vendor);
        close(tgpuSock); tgpuSock = -1; return 0;
    }
    isNVIDIA = (vendor == PCI_VENDOR_NVIDIA);

    // Enable bus-master + memory-space in PCI command register.
    uint16_t cmd = (uint16_t)cfg_read(tgpuSock, 0, 0x04, 2);
    cfg_write(tgpuSock, 0, 0x04, 2, cmd | 0x0006);

    // Map BARs 0 and 2.
    for (int b : {0, 2}) {
        uint64_t a = 0, s = 0;
        tgpu_rpc(tgpuSock, 0, CMD_MAP_BAR, 0, 0, 0, (uint32_t)b, &a, &s);
        tgpuBars[b].addr = a; tgpuBars[b].size = s;
    }

    DeviceInfo info = {0, vendor, device, true};
    tgpuDevices.push_back(info);
    resourceMap->insert({0, 0});

    // For NVIDIA: attempt to load the CUDA driver (done in nvSetup).

    fprintf(stderr, "TinyGPU: %s GPU %04x:%04x, BAR0=%llu MB, BAR2=%llu MB\n",
            isNVIDIA ? "NVIDIA" : "AMD",
            vendor, device,
            (unsigned long long)(tgpuBars[0].size >> 20),
            (unsigned long long)(tgpuBars[2].size >> 20));
    return 1;
}

// ── Device info ──────────────────────────────────────────────────────────────

int GPUInterface::GetDeviceCount() { return (int)resourceMap->size(); }

void GPUInterface::GetDeviceName(int dev, char* name, int len) {
    if (dev >= (int)tgpuDevices.size()) { strncpy(name, "TinyGPU", len); return; }
    const DeviceInfo& d = tgpuDevices[dev];
    snprintf(name, len, "%s GPU %04x:%04x (TinyGPU USB4)",
             d.pci_vendor == PCI_VENDOR_NVIDIA ? "NVIDIA" : "AMD",
             d.pci_vendor, d.pci_device);
}

void GPUInterface::GetDeviceDescription(int /*dev*/, char* desc) {
    snprintf(desc, 256, "VRAM BAR2 %llu MB | MMIO BAR0 %llu MB | USB4/Thunderbolt eGPU",
             (unsigned long long)(tgpuBars[2].size >> 20),
             (unsigned long long)(tgpuBars[0].size >> 20));
}

long GPUInterface::GetDeviceTypeFlag(int) { return BEAGLE_FLAG_PROCESSOR_GPU; }

BeagleDeviceImplementationCodes GPUInterface::GetDeviceImplementationCode(int dev) {
    if (dev < (int)tgpuDevices.size())
        return tgpuDevices[dev].pci_vendor == PCI_VENDOR_NVIDIA
             ? BEAGLE_TINYGPU_DEVICE_NVIDIA_GPU : BEAGLE_TINYGPU_DEVICE_AMD_GPU;
    return BEAGLE_TINYGPU_DEVICE_NVIDIA_GPU;
}

bool GPUInterface::GetSupportsDoublePrecision(int dev) {
    return (dev < (int)tgpuDevices.size()) ? tgpuDevices[dev].supports_dp : true;
}

// ── SetDevice ─────────────────────────────────────────────────────────────────

void GPUInterface::SetDevice(int devNum, int paddedStateCount, int categoryCount,
                              int paddedPatternCount, int unpaddedPatternCount,
                              int tipCount, long flags) {
    tgpuDevId = (uint32_t)(*resourceMap)[devNum];

    InitializeKernelResource(paddedStateCount,
                             (flags & BEAGLE_FLAG_PRECISION_DOUBLE) != 0);
    if (!kernelResource) {
        fprintf(stderr, "TinyGPU: no kernel for %d states\n", paddedStateCount);
        exit(1);
    }
    kernelResource->categoryCount        = categoryCount;
    kernelResource->patternCount         = paddedPatternCount;
    kernelResource->unpaddedPatternCount = unpaddedPatternCount;
    kernelResource->flags                = flags;

    vramKernelTop = AMD_VRAM_KERNEL_BASE;
    vramDataTop   = AMD_VRAM_DATA_BASE;

    if (isNVIDIA)
        nvSetup();
    else
        amdSetup();
}

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

// ─────────────────────────────────────────────────────────────────────────────
// §7  NVIDIA setup — delegates to CUDA driver
// ─────────────────────────────────────────────────────────────────────────────

static char* ptx_to_cubin(const char* ptx, size_t ptx_sz, size_t* out_sz) {
    const char* sm = getenv("BEAGLE_TINYGPU_SM");
    if (!sm) sm = "sm_86";
    char ptx_path[256], cubin_path[256];
    snprintf(ptx_path,   sizeof(ptx_path),  "/tmp/beagle_tgpu_%d.ptx",   (int)getpid());
    snprintf(cubin_path, sizeof(cubin_path), "/tmp/beagle_tgpu_%d.cubin", (int)getpid());
    FILE* f = fopen(ptx_path, "wb");
    if (!f) return nullptr;
    fwrite(ptx, 1, ptx_sz, f); fclose(f);
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "ptxas -arch %s -o %s %s 2>/tmp/beagle_ptxas.log",
             sm, cubin_path, ptx_path);
    if (system(cmd) != 0) {
        fprintf(stderr, "TinyGPU/NV: ptxas failed (set BEAGLE_TINYGPU_SM=sm_XX)\n");
        unlink(ptx_path); return nullptr;
    }
    unlink(ptx_path);
    FILE* cf = fopen(cubin_path, "rb");
    if (!cf) return nullptr;
    fseek(cf, 0, SEEK_END); long sz = ftell(cf); rewind(cf);
    char* buf = (char*)malloc((size_t)sz);
    fread(buf, 1, (size_t)sz, cf); fclose(cf);
    unlink(cubin_path);
    if (out_sz) *out_sz = (size_t)sz;
    return buf;
}

void GPUInterface::nvSetup() {
    // Load CUDA driver dynamically (no compile-time dependency on cuda.h).
    if (!nvCuda.lib) {
        void* lib = dlopen("libcuda.so.1", RTLD_LAZY);
        if (!lib) lib = dlopen("libcuda.so",   RTLD_LAZY);
        if (!lib) lib = dlopen("libcuda.dylib", RTLD_LAZY);
        if (!lib) {
            fprintf(stderr, "TinyGPU/NV: libcuda not found — kernel dispatch disabled\n");
            return;
        }
        nvCuda.lib = lib;
#define DLSYM(n) nvCuda.n = (decltype(nvCuda.n))dlsym(lib, #n); \
                 if (!nvCuda.n) { dlclose(lib); nvCuda.lib = nullptr; return; }
        DLSYM(cuInit) DLSYM(cuDeviceGet)
        DLSYM(cuDevicePrimaryCtxRetain) DLSYM(cuCtxSetCurrent)
        DLSYM(cuModuleLoadData) DLSYM(cuModuleGetFunction)
        DLSYM(cuMemAlloc) DLSYM(cuMemFree)
        DLSYM(cuMemcpyHtoD) DLSYM(cuMemcpyDtoH)
        DLSYM(cuMemcpyDtoD) DLSYM(cuMemsetD16)
        DLSYM(cuLaunchKernel) DLSYM(cuCtxSynchronize) DLSYM(cuMemGetInfo)
#undef DLSYM
        if (nvCuda.cuInit(0) != 0) {
            dlclose(lib); nvCuda.lib = nullptr;
            fprintf(stderr, "TinyGPU/NV: cuInit failed\n"); return;
        }
    }

    int cudev = 0;
    nvCuda.cuDeviceGet(&cudev, (int)tgpuDevId);
    nvCuda.cuDevicePrimaryCtxRetain(&nvCtx, cudev);
    nvCuda.cuCtxSetCurrent(nvCtx);

    // Compile PTX → cubin and load the module.
    size_t cubin_sz = 0;
    char* cubin = ptx_to_cubin(kernelResource->kernelCode,
                               strlen(kernelResource->kernelCode), &cubin_sz);
    if (cubin) {
        nvCuda.cuModuleLoadData(&nvModule, cubin);
        free(cubin);
        fprintf(stderr, "TinyGPU/NV: cubin loaded (%zu bytes)\n", cubin_sz);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §8  AMD setup — bare-metal PM4 compute queue
//     Reference: tinygrad/tinygrad/runtime/support/am/ip.py : setup_ring()
// ─────────────────────────────────────────────────────────────────────────────

// Compile OpenCL C kernel source to AMD HSACO ELF using clang offline.
static uint8_t* ocl_to_hsaco(const char* src, size_t src_sz,
                               const char* gpu_target, size_t* out_sz) {
    char cl_path[256], co_path[256];
    snprintf(cl_path, sizeof(cl_path), "/tmp/beagle_tgpu_%d.cl",   (int)getpid());
    snprintf(co_path, sizeof(co_path), "/tmp/beagle_tgpu_%d.hsaco",(int)getpid());

    FILE* f = fopen(cl_path, "wb");
    if (!f) return nullptr;
    fwrite(src, 1, src_sz, f); fclose(f);

    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
             "clang -x cl -target amdgcn-amd-amdhsa -mcpu=%s "
             "-cl-std=CL2.0 -Xclang -finclude-default-header "
             "-O2 -o %s %s 2>/tmp/beagle_amd_compile.log",
             gpu_target, co_path, cl_path);
    if (system(cmd) != 0) {
        fprintf(stderr, "TinyGPU/AMD: clang compile failed (set BEAGLE_TINYGPU_AMDGPU=gfxXXXX)\n"
                        "            see /tmp/beagle_amd_compile.log\n");
        unlink(cl_path); return nullptr;
    }
    unlink(cl_path);

    FILE* cf = fopen(co_path, "rb");
    if (!cf) return nullptr;
    fseek(cf, 0, SEEK_END); long sz = ftell(cf); rewind(cf);
    uint8_t* buf = (uint8_t*)malloc((size_t)sz);
    fread(buf, 1, (size_t)sz, cf); fclose(cf);
    unlink(co_path);
    if (out_sz) *out_sz = (size_t)sz;
    return buf;
}

// Parse HSACO ELF and register all kernels in tgpuKernels.
// For each ".text.kernelname" section the corresponding .rodata descriptor is
// located and the resource metadata extracted.
void GPUInterface::amdParseHsaco(const uint8_t* elf, size_t elf_sz,
                                   uint64_t code_vram_base) {
    if (elf_sz < sizeof(Elf64_Ehdr)) return;
    const Elf64_Ehdr* eh = (const Elf64_Ehdr*)elf;
    const Elf64_Shdr* shdrs = (const Elf64_Shdr*)(elf + eh->e_shoff);
    const char* strtab = (const char*)(elf + shdrs[eh->e_shstrndx].sh_offset);

    // The AMDHSA ELF has one .rodata section that contains all kernel descriptors
    // back-to-back (64 bytes each), indexed by their symbol order.
    uint64_t ro_off = 0;
    const uint8_t* rodata = elf_section(elf, elf_sz, ".rodata", &ro_off);

    for (uint16_t i = 0; i < eh->e_shnum; ++i) {
        const char* sname = strtab + shdrs[i].sh_name;
        if (strncmp(sname, ".text.", 6) != 0) continue;

        const char* kname = sname + 6;  // kernel name after ".text."
        uint64_t text_vaddr = shdrs[i].sh_addr;  // VA of .text section
        uint64_t text_off   = shdrs[i].sh_offset; // file offset

        // Look for .rodata.kname or use .rodata at symbol offset.
        // Simplification: use the .rodata section; the descriptor for each kernel
        // is at .rodata + (kernel_index * 64).  We find it by matching symbols.
        // For a single-kernel compile this is always at .rodata + 0.
        const AMDKernelDesc* desc = nullptr;
        if (rodata) {
            // Try to find kernel descriptor by scanning .rodata for matching code entry.
            for (size_t d = 0; d + sizeof(AMDKernelDesc) <= shdrs[/*rodata idx*/0].sh_size
                 || d == 0; d += 64) {
                const AMDKernelDesc* c = (const AMDKernelDesc*)(rodata + d);
                // Kernel code VA = desc_vaddr + kernel_code_entry_byte_offset
                // desc_vaddr  = .rodata section sh_addr + d
                // We match if code VA falls in the .text section
                uint64_t desc_vaddr = shdrs[i].sh_addr + d;  // approximation
                if (ro_off > 0) {
                    // Find rodata section header
                    for (uint16_t r = 0; r < eh->e_shnum; ++r) {
                        if (strcmp(strtab + shdrs[r].sh_name, ".rodata") == 0) {
                            desc_vaddr = shdrs[r].sh_addr + d;
                            break;
                        }
                    }
                }
                int64_t entry_off = c->kernel_code_entry_byte_offset;
                if (desc_vaddr + entry_off == text_vaddr) {
                    desc = c; break;
                }
                if (d == 0 && rodata) { desc = c; break; }  // best-effort for single kernel
            }
        }

        auto* entry = new KernelEntry;
        entry->name      = kname;
        // VRAM address = code_vram_base + (section VA - first text section VA).
        // Simplification: the ELF sections are laid out sequentially, so the
        // VRAM offset = code_vram_base + text_file_offset.
        entry->code_vaddr = code_vram_base + text_off;
        entry->wave32    = false;
        entry->rsrc1     = 0;
        entry->rsrc2     = 0;
        entry->rsrc3     = 0;
        entry->cu_func   = nullptr;

        if (desc) {
            entry->rsrc1   = desc->compute_pgm_rsrc1;
            entry->rsrc2   = desc->compute_pgm_rsrc2;
            entry->rsrc3   = desc->compute_pgm_rsrc3;
            entry->wave32  = (desc->kernel_code_properties & 0x0400) != 0;
            // Adjust code address: desc VA + entry_byte_offset = kernel code
            // Our code_vram_base + text_off already points to the .text section,
            // but the real entry is at desc_vaddr + kernel_code_entry_byte_offset.
            // For now keep text_off as the entry; kernel_code_entry_byte_offset
            // is typically 0 or -256 (pointing from descriptor back to code).
        }

        tgpuKernels[kname] = entry;
        fprintf(stderr, "TinyGPU/AMD: registered kernel '%s' at VRAM+0x%llx rsrc1=0x%x\n",
                kname, (unsigned long long)entry->code_vaddr, entry->rsrc1);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §AM  AMD GPU boot sequence
//      Port of tinygrad/tinygrad/runtime/support/am/ip.py (partial boot path)
//      Reference: NVIDIA/AMD open-source driver, tinygrad autogen register defs.
// ─────────────────────────────────────────────────────────────────────────────

// amdBuildPageTables — construct a 4-level identity page table in VRAM.
// Identity: GPU VA X maps to GPU PA (fb_base + X) for X in [0, map_size).
// Page table layout in VRAM (starting at offset 0):
//   [0x000000 .. 0x000FFF]  PDB2 root (512 entries)
//   [0x001000 .. 0x001FFF]  PDB1      (512 entries, covers first 512 GB)
//   [0x002000 .. 0x002FFF]  PDB0      (512 entries, covers first 1 GB)
//   [0x003000 .. 0x002FFF + n*4096] n PTBs (each 2 MB, covers map_size)
void GPUInterface::amdBuildPageTables(uint64_t fb_base) {
    // Map the entire region that BEAGLE uses: 0 .. AMD_VRAM_DATA_BASE + 512MB.
    const uint64_t MAP_SIZE = AMD_VRAM_DATA_BASE + (512ULL << 20);
    const uint32_t NUM_PTBS = (uint32_t)((MAP_SIZE + ((2ULL<<20)-1)) / (2ULL<<20));
    const uint32_t NUM_PDB0 = (NUM_PTBS + 511) / 512;  // PDB0 entries needed (≤1 for ≤1GB)

    size_t pt_bytes = (3 + (size_t)NUM_PTBS) * 4096;
    std::vector<uint64_t> pt(pt_bytes / 8, 0ULL);

    uint64_t* pdb2 = pt.data();                          // offset 0x0000
    uint64_t* pdb1 = pt.data() + 0x1000 / 8;            // offset 0x1000
    uint64_t* pdb0 = pt.data() + 0x2000 / 8;            // offset 0x2000
    // PTBs start at offset 0x3000

    // PDB2[0] → PDB1 at GPU PA fb_base + 0x1000
    pdb2[0] = (fb_base + 0x1000) | AMD_PDE_FLAGS;

    // PDB1[0] → PDB0 at GPU PA fb_base + 0x2000
    pdb1[0] = (fb_base + 0x2000) | AMD_PDE_FLAGS;

    // PDB0[i] → PTB[i] at GPU PA fb_base + 0x3000 + i*4096
    for (uint32_t i = 0; i < NUM_PTBS; ++i) {
        pdb0[i] = (fb_base + 0x3000 + (uint64_t)i * 0x1000) | AMD_PDE_FLAGS;

        // PTB[i][j]: identity-map page (i*512 + j) → GPU PA fb_base + page*4096
        uint64_t* ptb = pt.data() + (0x3000 + (uint64_t)i * 0x1000) / 8;
        for (uint32_t j = 0; j < 512; ++j) {
            uint64_t pa = fb_base + ((uint64_t)i * 512 + j) * 0x1000;
            ptb[j] = pa | AMD_PTE_FLAGS;
        }
    }

    // Write the page table binary to VRAM via BAR0 MMIO at offset 0.
    tgpu_mmio_write(tgpuSock, tgpuDevId, /*bar=*/0, 0, pt.data(), pt_bytes);
    amdFbBase = fb_base;

    fprintf(stderr, "TinyGPU/AMD: %u-PTB identity page table built "
            "(covers %.0f MB, fb_base=0x%llx)\n",
            NUM_PTBS, (double)MAP_SIZE / (1<<20),
            (unsigned long long)fb_base);
}

// amdGMCInit — initialise the GC (compute) VM hub.
// Direct port of AM_GMC.init_hub("GC") in tinygrad/runtime/support/am/ip.py.
void GPUInterface::amdGMCInit(uint64_t fb_base, uint64_t fb_end) {
    const uint64_t vm_base = 0;
    const uint64_t vm_end  = AMD_VRAM_DATA_BASE + (512ULL << 20) - 1;
    const uint64_t pt_root_pa = fb_base;  // PDB2 is at VRAM offset 0

    // Scratch and dummy page: use first two physical pages in VRAM.
    const uint64_t scratch_pa = fb_base;
    const uint64_t dummy_pa   = fb_base + 0x1000;

    // ── 1. System apertures (disable AGP, set FB aperture) ───────────────────
    bar0_wr32(tgpuSock, tgpuDevId, AMD_GCMC_VM_AGP_BASE, 0x0);
    bar0_wr32(tgpuSock, tgpuDevId, AMD_GCMC_VM_AGP_BOT,  0x00FFFFFF); // 0xffffffffffff >> 24
    bar0_wr32(tgpuSock, tgpuDevId, AMD_GCMC_VM_AGP_TOP,  0x0);
    bar0_wr32(tgpuSock, tgpuDevId, AMD_GCMC_VM_SYS_APE_LOW,  (uint32_t)(fb_base >> 18));
    bar0_wr32(tgpuSock, tgpuDevId, AMD_GCMC_VM_SYS_APE_HIGH, (uint32_t)(fb_end  >> 18));

    // System aperture default address (scratch page >> 12, split into LSB/MSB)
    bar0_wr32(tgpuSock, tgpuDevId, AMD_GCMC_VM_SYS_APE_DEF_LSB,
              (uint32_t)(scratch_pa >> 12));
    bar0_wr32(tgpuSock, tgpuDevId, AMD_GCMC_VM_SYS_APE_DEF_MSB,
              (uint32_t)(scratch_pa >> 44));

    // Protection fault default address (dummy page)
    bar0_wr32(tgpuSock, tgpuDevId, AMD_GCVM_L2_PROT_FAULT_DEF_LO,
              (uint32_t)(dummy_pa >> 12));
    bar0_wr32(tgpuSock, tgpuDevId, AMD_GCVM_L2_PROT_FAULT_DEF_HI,
              (uint32_t)(dummy_pa >> 44));

    // ── 2. L2 cache and TLB configuration ────────────────────────────────────
    bar0_wr32(tgpuSock, tgpuDevId, AMD_GCVM_L2_PROT_FAULT_CNTL2,
              AMD_GCVM_L2_PROT_CNTL2_VAL);
    bar0_wr32(tgpuSock, tgpuDevId, AMD_GCMC_VM_MX_L1_TLB_CNTL,
              AMD_GCMC_MX_L1_TLB_CNTL_VAL);

    // Read-modify-write L2_CNTL (OR in enable bits to preserve hardware defaults)
    bar0_wr32(tgpuSock, tgpuDevId, AMD_GCVM_L2_CNTL,
              bar0_rd32(tgpuSock, tgpuDevId, AMD_GCVM_L2_CNTL) | AMD_GCVM_L2_CNTL_ENABLE);
    bar0_wr32(tgpuSock, tgpuDevId, AMD_GCVM_L2_CNTL2,
              bar0_rd32(tgpuSock, tgpuDevId, AMD_GCVM_L2_CNTL2) | AMD_GCVM_L2_CNTL2_FLUSH);
    bar0_wr32(tgpuSock, tgpuDevId, AMD_GCVM_L2_CNTL3,  AMD_GCVM_L2_CNTL3_GFX10);
    bar0_wr32(tgpuSock, tgpuDevId, AMD_GCVM_L2_CNTL4,  AMD_GCVM_L2_CNTL4_VAL);
    bar0_wr32(tgpuSock, tgpuDevId, AMD_GCVM_L2_CNTL5,  AMD_GCVM_L2_CNTL5_VAL);

    // ── 3. VMID 0 page table registration ────────────────────────────────────
    bar0_wr32(tgpuSock, tgpuDevId, AMD_GCVM_CTX0_PT_START_LO,
              (uint32_t)(vm_base >> 12));
    bar0_wr32(tgpuSock, tgpuDevId, AMD_GCVM_CTX0_PT_START_HI,
              (uint32_t)(vm_base >> 44));
    bar0_wr32(tgpuSock, tgpuDevId, AMD_GCVM_CTX0_PT_END_LO,
              (uint32_t)(vm_end >> 12));
    bar0_wr32(tgpuSock, tgpuDevId, AMD_GCVM_CTX0_PT_END_HI,
              (uint32_t)(vm_end >> 44));
    // Root PDB2 physical address with VALID bit
    uint64_t pt_root_entry = pt_root_pa | AMD_PDE_FLAGS;
    bar0_wr32(tgpuSock, tgpuDevId, AMD_GCVM_CTX0_PT_BASE_LO,
              (uint32_t)(pt_root_entry));
    bar0_wr32(tgpuSock, tgpuDevId, AMD_GCVM_CTX0_PT_BASE_HI,
              (uint32_t)(pt_root_entry >> 32));
    // Enable context with 4-level page table and all fault reporting
    bar0_wr32(tgpuSock, tgpuDevId, AMD_GCVM_CTX0_CNTL, AMD_GCVM_CTX0_CNTL_4LVL);

    // ── 4. Disable identity aperture (use real page tables via VMID 0) ────────
    bar0_wr32(tgpuSock, tgpuDevId, AMD_GCVM_IDENTITY_APE_LO_LO32, 0xFFFFFFFF);
    bar0_wr32(tgpuSock, tgpuDevId, AMD_GCVM_IDENTITY_APE_LO_HI32, 0x0000000F);
    bar0_wr32(tgpuSock, tgpuDevId, AMD_GCVM_IDENTITY_APE_HI_LO32, 0x00000000);
    bar0_wr32(tgpuSock, tgpuDevId, AMD_GCVM_IDENTITY_APE_HI_HI32, 0x00000000);
    bar0_wr32(tgpuSock, tgpuDevId, AMD_GCVM_IDENTITY_PHYS_LO, 0x00000000);
    bar0_wr32(tgpuSock, tgpuDevId, AMD_GCVM_IDENTITY_PHYS_HI, 0x00000000);

    // ── 5. Invalidate address ranges for all 18 GC engines ───────────────────
    // Write 0x1FFFFFFFFF (full range) to ENG_{n}_ADDR_RANGE_{LO,HI}32.
    // ENG_n registers are sequential: ENG0_LO = AMD_GCVM_INVAL_ENG_ADDR_BASE + n*2.
    for (uint32_t n = 0; n < 18; ++n) {
        bar0_wr32(tgpuSock, tgpuDevId, AMD_GCVM_INVAL_ENG_ADDR_BASE + n * 2,     0xFFFFFFFF);
        bar0_wr32(tgpuSock, tgpuDevId, AMD_GCVM_INVAL_ENG_ADDR_BASE + n * 2 + 1, 0x0000001F);
    }

    // ── 6. TLB flush for VMID 0 ──────────────────────────────────────────────
    amdTlbFlush(0);

    fprintf(stderr, "TinyGPU/AMD: GMC GC hub initialised (VM_CONTEXT0 active)\n");
}

// amdTlbFlush — invalidate the GCVM TLB for one VMID via ENG17.
// Port of AM_GMC.flush_tlb("GC", vmid) in ip.py.
void GPUInterface::amdTlbFlush(uint32_t vmid) {
    // GC hub TLB invalidation: no semaphore needed (only MM hub uses semaphore).
    uint32_t req = AMD_TLB_INVAL_VMID0_FULL;
    if (vmid != 0) {
        // For other VMIDs, replace bit 0 with the correct bit.
        req = (req & ~0xFFFFU) | (1u << vmid);
    }
    bar0_wr32(tgpuSock, tgpuDevId, AMD_GCVM_INVAL_ENG17_REQ, req);

    // Poll ENG17_ACK until the VMID's bit is set.
    for (uint32_t t = 0; t < 1000000; ++t) {
        if (bar0_rd32(tgpuSock, tgpuDevId, AMD_GCVM_INVAL_ENG17_ACK) & (1u << vmid))
            return;
        usleep(1);
    }
    fprintf(stderr, "TinyGPU/AMD: TLB flush ACK timeout for VMID %u\n", vmid);
}

// amdMecReset — soft-reset the CP and CPC, then re-activate MEC pipe 0.
// Port of AM_GFX.reset_mec() in ip.py (GFX10+ path: RS64 MEC).
void GPUInterface::amdMecReset() {
    // Soft-reset CP and CPC.
    bar0_wr32(tgpuSock, tgpuDevId, AMD_GRBM_SOFT_RESET, AMD_GRBM_SOFT_RESET_MEC);
    usleep(50000);  // 50 ms — match tinygrad's time.sleep(0.05)
    bar0_wr32(tgpuSock, tgpuDevId, AMD_GRBM_SOFT_RESET, 0x0);

    // Re-enable MEC pipe 0 via CP_MEC_RS64_CNTL (GFX10+ RS64 MEC path).
    // mec_pipe0_active[26]=1, mec_halt[30]=0, all resets=0.
    bar0_wr32(tgpuSock, tgpuDevId, AMD_CP_MEC_RS64_CNTL, AMD_CP_MEC_RS64_ENABLE);
    usleep(50000);
}

// amdGFXInit — configure RLC, SH_MEM, and MEC doorbell range.
// Port of AM_GFX.init_hw() non-firmware parts in ip.py.
void GPUInterface::amdGFXInit() {
    // GRBM read timeout.
    bar0_wr32(tgpuSock, tgpuDevId, AMD_GRBM_CNTL, 0xFF);

    // Enable RLC (RealTime Link Controller — manages compute scheduling).
    bar0_wr32(tgpuSock, tgpuDevId, AMD_RLC_CNTL,     0x1);  // rlc_enable_f32
    bar0_wr32(tgpuSock, tgpuDevId, AMD_RLC_SRM_CNTL, 0x3);  // srm_enable | auto_incr_addr

    // Configure SH_MEM registers for all 16 VMIDs on MEC1/pipe0/queue0.
    // SH_MEM_CONFIG: address_mode[0]=1 (flat 64-bit addressing).
    // SH_MEM_BASES: private_base=0, shared_base=0.
    bar0_wr32(tgpuSock, tgpuDevId, AMD_GRBM_GFX_CNTL, 1u << 2);  // select MEC1
    for (uint32_t vmid = 0; vmid < 16; ++vmid) {
        bar0_wr32(tgpuSock, tgpuDevId, AMD_SH_MEM_CONFIG, 0x1);
        bar0_wr32(tgpuSock, tgpuDevId, AMD_SH_MEM_BASES,  0x0);
    }
    bar0_wr32(tgpuSock, tgpuDevId, AMD_GRBM_GFX_CNTL, 0x0);  // deselect

    // MEC doorbell range for XCC 0: LOWER=0x0, UPPER=0xF8.
    bar0_wr32(tgpuSock, tgpuDevId, AMD_CP_MEC_DOORBELL_LOWER, 0x00);
    bar0_wr32(tgpuSock, tgpuDevId, AMD_CP_MEC_DOORBELL_UPPER, 0xF8);

    fprintf(stderr, "TinyGPU/AMD: GFX engine configured\n");
}

void GPUInterface::amdSetup() {
    // ── 1. Read GPU VRAM frame buffer location from hardware ──────────────────
    uint64_t fb_base = (uint64_t)(bar0_rd32(tgpuSock, 0, AMD_FB_LOCATION_BASE) & 0xFFFFFF) << 24;
    uint64_t fb_end  = ((uint64_t)(bar0_rd32(tgpuSock, 0, AMD_FB_LOCATION_TOP) & 0xFFFFFF) + 1) << 24;
    amdFbBase = fb_base;
    fprintf(stderr, "TinyGPU/AMD: VRAM physical 0x%llx..0x%llx (%llu MB)\n",
            (unsigned long long)fb_base, (unsigned long long)fb_end,
            (unsigned long long)((fb_end - fb_base) >> 20));

    // ── 2. Detect boot state (partial = GPU was previously init'd by macOS) ───
    uint32_t scratch7  = bar0_rd32(tgpuSock, 0, AMD_SCRATCH_REG7);
    uint32_t scratch6  = bar0_rd32(tgpuSock, 0, AMD_SCRATCH_REG6);
    amdPartialBoot = (scratch7 == AMD_DEV_VERSION) && (scratch6 == 0)
                     && !getenv("AM_RESET");
    fprintf(stderr, "TinyGPU/AMD: %s boot (SCRATCH7=0x%08x)\n",
            amdPartialBoot ? "partial" : "full", scratch7);

    // ── 3. Build identity page tables in VRAM ────────────────────────────────
    amdBuildPageTables(fb_base);

    // ── 4. Initialise GMC GC hub (VMID 0 page table + L2 cache) ─────────────
    amdGMCInit(fb_base, fb_end);

    // ── 5. MEC reset (partial boot) or full GFX init (cold boot) ─────────────
    if (amdPartialBoot) {
        amdMecReset();   // fast path: reset CP, re-enable MEC pipe 0
    } else {
        // Cold boot: PSP has already loaded MEC firmware via ROM boot
        // (macOS's AMD driver handles PSP boot before we connect via TinyGPU).
        // If MEC is not running, PSP firmware files from AMD linux-firmware
        // package are required — see amdGFXInit() comment.
        amdGFXInit();
        amdMecReset();   // ensure MEC is active even on first boot
    }

    // ── 6. Compile OpenCL C kernels to AMD ISA ────────────────────────────────
    const char* gpu_target = getenv("BEAGLE_TINYGPU_AMDGPU");
    if (!gpu_target) gpu_target = "gfx1030";  // default: RDNA2 Navi21 (RX 6800/6900)

    size_t hsaco_sz = 0;
    uint8_t* hsaco = ocl_to_hsaco(kernelResource->kernelCode,
                                   strlen(kernelResource->kernelCode),
                                   gpu_target, &hsaco_sz);
    if (!hsaco) {
        fprintf(stderr, "TinyGPU/AMD: kernel compilation failed\n");
        return;
    }

    // ── 7. Copy HSACO binary to VRAM (BAR0 at AMD_VRAM_KERNEL_BASE) ──────────
    size_t aligned = (hsaco_sz + 0xfffULL) & ~0xfffULL;
    uint64_t code_vram = vramKernelTop;
    vramKernelTop += aligned;
    tgpu_mmio_write(tgpuSock, tgpuDevId, /*bar=*/0, code_vram, hsaco, hsaco_sz);
    amdParseHsaco(hsaco, hsaco_sz, code_vram);
    free(hsaco);

    // ── 8. Allocate compute ring in VRAM ─────────────────────────────────────
    amdRingVram = AMD_VRAM_RING_BASE;
    amdRingWptr = 0;

    // ── 9. Allocate host-visible control memory (rptr / wptr-poll / EOP) ─────
    uint64_t mapped = 0;
    int      shm_fd = -1;
    if (!tgpu_rpc_fd(tgpuSock, tgpuDevId, 0x4000, 0, &mapped, &shm_fd)) {
        fprintf(stderr, "TinyGPU/AMD: MAP_SYSMEM_FD failed\n");
        return;
    }
    void* ctrl = mmap(nullptr, (size_t)mapped, PROT_READ | PROT_WRITE,
                      MAP_SHARED, shm_fd, 0);
    if (ctrl == MAP_FAILED) { perror("TinyGPU/AMD mmap ctrl"); close(shm_fd); return; }

    amdCompletionHost   = ctrl;
    amdCompletionMapped = (size_t)mapped;
    amdCompletionFd     = shm_fd;

    // Layout: [0x000] rptr (4B)  [0x008] wptr (8B)  [0x010] EOP semaphore (8B)
    uint64_t ctrl_va = (uint64_t)(uintptr_t)ctrl;
    amdRptrAddr  = ctrl_va + 0x000;
    amdWptrAddr  = ctrl_va + 0x008;
    amdEopAddr   = ctrl_va + 0x010;
    amdEopSignal = 0;
    *(volatile uint64_t*)(amdEopAddr) = 0;

    // ── 10. Configure HQD (Hardware Queue Descriptor) via GRBM ───────────────
    // Select MEC1 (meid=1), pipe0, queue0: GRBM_GFX_CNTL = 1<<2 = 4
    bar0_wr32(tgpuSock, tgpuDevId, AMD_GRBM_GFX_CNTL, 1u << 2);

    // Ring base address >> 8 (GPU PA = fb_base + ring VRAM offset)
    uint64_t ring_pa = amdFbBase + amdRingVram;
    bar0_wr32(tgpuSock, tgpuDevId, AMD_CP_HQD_PQ_BASE,
              (uint32_t)((ring_pa >> 8) & 0xFFFFFFFF));
    bar0_wr32(tgpuSock, tgpuDevId, AMD_CP_HQD_PQ_BASE_HI,
              (uint32_t)((ring_pa >> 40) & 0xFF));

    // Read-pointer report address (host VA — GPU writes rptr here)
    bar0_wr32(tgpuSock, tgpuDevId, AMD_CP_HQD_PQ_RPTR_REPORT_ADDR,
              (uint32_t)(amdRptrAddr));
    bar0_wr32(tgpuSock, tgpuDevId, AMD_CP_HQD_PQ_RPTR_REPORT_ADDR_HI,
              (uint32_t)(amdRptrAddr >> 32));

    // Write-pointer poll address (host VA — CPU updates wptr here)
    bar0_wr32(tgpuSock, tgpuDevId, AMD_CP_HQD_PQ_WPTR_POLL_ADDR,
              (uint32_t)(amdWptrAddr));
    bar0_wr32(tgpuSock, tgpuDevId, AMD_CP_HQD_PQ_WPTR_POLL_ADDR_HI,
              (uint32_t)(amdWptrAddr >> 32));

    // Doorbell control: NAVI10_DOORBELL_MEC_RING0=3, field doorbell_offset=3*2=6
    // CP_HQD_PQ_DOORBELL_CONTROL bit layout (from am/regs.py):
    //   doorbell_offset[27:11], doorbell_en[30]
    uint32_t dbell_ctrl = ((AMD_NAVI10_DOORBELL_MEC_RING0 * 2) << 11) | (1u << 30);
    bar0_wr32(tgpuSock, tgpuDevId, AMD_CP_HQD_PQ_DOORBELL_CONTROL, dbell_ctrl);

    // Ring size: queue_size field = log2(ring_dwords) - 2 (ip.py: bit_length()-2)
    uint32_t ring_dwords = AMD_RING_SIZE / 4;
    uint32_t qs = 0; { uint32_t t = ring_dwords; while (t > 1) { t >>= 1; ++qs; } }
    bar0_wr32(tgpuSock, tgpuDevId, AMD_CP_HQD_PQ_CONTROL,
              (qs - 2) | (5u << 8));  // queue_size | rptr_block_size=5

    // Activate the queue
    bar0_wr32(tgpuSock, tgpuDevId, AMD_CP_HQD_ACTIVE, 1);
    bar0_wr32(tgpuSock, tgpuDevId, AMD_GRBM_GFX_CNTL, 0);  // deselect

    // ── 11. Mark GPU as AM-initialised ───────────────────────────────────────
    bar0_wr32(tgpuSock, tgpuDevId, AMD_SCRATCH_REG6, 0);
    bar0_wr32(tgpuSock, tgpuDevId, AMD_SCRATCH_REG7, AMD_DEV_VERSION);

    fprintf(stderr, "TinyGPU/AMD: MEC1 compute queue active "
            "(ring@VRAM+0x%llx / PA=0x%llx, %u KB)\n",
            (unsigned long long)amdRingVram,
            (unsigned long long)ring_pa,
            AMD_RING_SIZE / 1024);
}

// ── Streams ──────────────────────────────────────────────────────────────────

void GPUInterface::ResizeStreamCount(int) {}

// ── Synchronization ──────────────────────────────────────────────────────────

void GPUInterface::SynchronizeHost() {
    if (isNVIDIA) {
        if (nvCuda.lib) nvCuda.cuCtxSynchronize();
        return;
    }
    // AMD: spin-wait on EOP semaphore written by PACKET3_RELEASE_MEM
    if (!amdCompletionHost) return;
    volatile uint64_t* sig = (volatile uint64_t*)amdEopAddr;
    while (*sig < amdEopSignal)
        usleep(10);
}

void GPUInterface::SynchronizeDevice() { SynchronizeHost(); }

void GPUInterface::SynchronizeDeviceWithIndex(int, int) { SynchronizeHost(); }

// ── GetFunction ───────────────────────────────────────────────────────────────

GPUFunction GPUInterface::GetFunction(const char* name) {
    auto it = tgpuKernels.find(name);
    if (it != tgpuKernels.end()) return (GPUFunction)it->second;

    auto* entry = new KernelEntry;
    entry->name     = name;
    entry->cu_func  = nullptr;
    entry->code_vaddr = 0;
    entry->rsrc1 = entry->rsrc2 = entry->rsrc3 = 0;
    entry->wave32 = false;

    if (isNVIDIA && nvModule) {
        nvCuda.cuModuleGetFunction(&entry->cu_func, nvModule, name);
    }

    tgpuKernels[name] = entry;
    return (GPUFunction)entry;
}

// ── LaunchKernel (internal helper) ───────────────────────────────────────────

void GPUInterface::LaunchKernelImpl(GPUFunction deviceFunction,
                                     Dim3Int block, Dim3Int grid,
                                     int nPtr, int nTotal,
                                     GPUPtr* ptrs, unsigned int* ints) {
    KernelEntry* entry = (KernelEntry*)deviceFunction;

    // ── NVIDIA: delegate entirely to CUDA driver ──────────────────────────────
    if (isNVIDIA) {
        if (!nvCuda.lib || !entry->cu_func) return;
        // Build CUDA params array: pointers then ints.
        std::vector<void*> params(nTotal);
        for (int i = 0; i < nPtr; ++i)   params[i] = &ptrs[i];
        for (int i = nPtr; i < nTotal; ++i) params[i] = &ints[i - nPtr];
        nvCuda.cuLaunchKernel(entry->cu_func,
                              grid.x, grid.y, grid.z,
                              block.x, block.y, block.z,
                              0, nullptr,
                              params.data(), nullptr);
        return;
    }

    // ── AMD: PM4 bare-metal dispatch ─────────────────────────────────────────
    if (entry->code_vaddr == 0) return;

    // Set GRBM context to MEC1/pipe0/queue0.
    bar0_wr32(tgpuSock, tgpuDevId, AMD_GRBM_GFX_CNTL, 1u << 2);

    // Write COMPUTE_PGM_LO/HI — kernel entry address >> 8.
    uint64_t prog_addr = entry->code_vaddr;
    bar0_wr32(tgpuSock, tgpuDevId, AMD_COMPUTE_PGM_LO, (uint32_t)((prog_addr >> 8) & 0xFFFFFFFF));
    bar0_wr32(tgpuSock, tgpuDevId, AMD_COMPUTE_PGM_HI, (uint32_t)((prog_addr >> 40) & 0xFF));

    // COMPUTE_PGM_RSRCx — from HSACO kernel descriptor.
    bar0_wr32(tgpuSock, tgpuDevId, AMD_COMPUTE_PGM_RSRC1, entry->rsrc1);
    bar0_wr32(tgpuSock, tgpuDevId, AMD_COMPUTE_PGM_RSRC2, entry->rsrc2);
    bar0_wr32(tgpuSock, tgpuDevId, AMD_COMPUTE_PGM_RSRC3, entry->rsrc3);
    bar0_wr32(tgpuSock, tgpuDevId, AMD_COMPUTE_TMPRING_SIZE, 0);  // no scratch

    // COMPUTE_NUM_THREAD_XYZ — local (workgroup) sizes.
    bar0_wr32(tgpuSock, tgpuDevId, AMD_COMPUTE_NUM_THREAD_X, block.x);
    bar0_wr32(tgpuSock, tgpuDevId, AMD_COMPUTE_NUM_THREAD_Y, block.y);
    bar0_wr32(tgpuSock, tgpuDevId, AMD_COMPUTE_NUM_THREAD_Z, block.z);

    // COMPUTE_USER_DATA_0..N — kernel args via a host-visible kernarg buffer.
    // Build the buffer: 8 bytes per GPUPtr argument, 4 bytes per uint argument.
    // Copy to a pinned host-visible allocation, then pass its VA as kernarg ptr.
    size_t kernarg_sz = (size_t)nPtr * 8 + (size_t)(nTotal - nPtr) * 4;
    // Allocate a small pinned buffer for this launch's args.
    uint64_t ka_mapped = 0;
    int      ka_fd     = -1;
    tgpu_rpc_fd(tgpuSock, tgpuDevId, (uint64_t)((kernarg_sz + 0xfff) & ~0xfffULL), 0,
                &ka_mapped, &ka_fd);
    void* ka_host = nullptr;
    if (ka_fd >= 0) {
        ka_host = mmap(nullptr, (size_t)ka_mapped, PROT_READ | PROT_WRITE,
                       MAP_SHARED, ka_fd, 0);
        if (ka_host == MAP_FAILED) { ka_host = nullptr; close(ka_fd); ka_fd = -1; }
    }
    if (!ka_host) {
        // Fallback: local buffer (won't be accessible from GPU, but lets us build)
        ka_host = calloc(1, kernarg_sz + 1);
    }

    uint8_t* kp = (uint8_t*)ka_host;
    for (int i = 0; i < nPtr; ++i) {
        memcpy(kp, &ptrs[i], 8); kp += 8;
    }
    for (int i = 0; i < nTotal - nPtr; ++i) {
        memcpy(kp, &ints[i], 4); kp += 4;
    }

    // Pass kernarg buffer address in COMPUTE_USER_DATA_0/1 (64-bit).
    uint64_t ka_va = (uint64_t)(uintptr_t)ka_host;
    bar0_wr32(tgpuSock, tgpuDevId, AMD_COMPUTE_USER_DATA_0,     (uint32_t)(ka_va & 0xFFFFFFFF));
    bar0_wr32(tgpuSock, tgpuDevId, AMD_COMPUTE_USER_DATA_0 + 1, (uint32_t)(ka_va >> 32));

    bar0_wr32(tgpuSock, tgpuDevId, AMD_GRBM_GFX_CNTL, 0);  // deselect context

    // ── Build PM4 command sequence in the compute ring ────────────────────────
    std::vector<uint32_t> pkt;

    // PACKET3_DISPATCH_DIRECT — 4 data DWORDs
    // DW0: header; DW1: dim_x; DW2: dim_y; DW3: dim_z; DW4: initiator
    uint32_t initiator = (1u << 0)  // compute_shader_en
                       | (1u << 2); // force_start_at_000
    if (entry->wave32)
        initiator |= (1u << 15);    // cs_w32_en (GFX11+)
    pkt.push_back(pm4_pkt3(PM4_DISPATCH_DIRECT, 4));
    pkt.push_back(grid.x);
    pkt.push_back(grid.y);
    pkt.push_back(grid.z);
    pkt.push_back(initiator);

    // PACKET3_RELEASE_MEM — EOP semaphore write to amdEopAddr
    // Signals completion: GPU writes (amdEopSignal) to amdEopAddr when done.
    ++amdEopSignal;
    // event_dw: EVENT_TYPE=20 (CACHE_FLUSH_AND_INV_TS_EVENT), EVENT_INDEX=5 (end_of_pipe)
    uint32_t event_dw   = (AMD_EOP_EVENT_TYPE << 0) | (AMD_EOP_EVENT_INDEX << 8);
    // GCR flags: GL2_INV | GL2_WB | GL1_INV | GLV_INV | GLM_WB | GLM_INV | SEQ
    uint32_t gcr_flags  = (1<<20)|(1<<21)|(1<<15)|(1<<14)|(1<<12)|(1<<13)|(1<<22);
    // mem_sel_dw: DATA_SEL=1 (32-bit value), DST_SEL=0 (memory), INT_SEL=2 (no int)
    uint32_t mem_sel_dw = (1u << 29) | (2u << 24);
    pkt.push_back(pm4_pkt3(PM4_RELEASE_MEM, 6));
    pkt.push_back(event_dw | gcr_flags);
    pkt.push_back(mem_sel_dw);
    pkt.push_back((uint32_t)(amdEopAddr & 0xFFFFFFFF));
    pkt.push_back((uint32_t)(amdEopAddr >> 32));
    pkt.push_back((uint32_t)(amdEopSignal & 0xFFFFFFFF));
    pkt.push_back((uint32_t)(amdEopSignal >> 32));

    // Write PM4 commands to the ring buffer in VRAM.
    uint32_t pkt_bytes = (uint32_t)(pkt.size() * sizeof(uint32_t));
    tgpu_mmio_write(tgpuSock, tgpuDevId, /*bar=*/0,
                    amdRingVram + amdRingWptr, pkt.data(), pkt_bytes);
    amdRingWptr = (amdRingWptr + pkt_bytes) % AMD_RING_SIZE;

    // Update wptr in host-visible memory.
    *(volatile uint64_t*)amdWptrAddr = (uint64_t)(amdRingWptr / 4);

    // Ring doorbell: 64-bit write to BAR2 at doorbell BAR2 byte offset.
    // Doorbell index = AMD_NAVI10_DOORBELL_MEC_RING0 = 3
    // BAR2 offset (doorbell64) = doorbell_ctrl_offset * 4 bytes
    // doorbell_ctrl_offset = doorbell_idx * 2 = 6 → byte offset = 6 * 8 = 48
    uint64_t doorbell_wptr = (uint64_t)(amdRingWptr / 4);
    uint64_t dbell_offset  = (uint64_t)(AMD_NAVI10_DOORBELL_MEC_RING0 * 2) * 8;
    tgpu_mmio_write(tgpuSock, tgpuDevId, /*bar=*/2,
                    dbell_offset, &doorbell_wptr, sizeof(doorbell_wptr));

    // Cleanup kernarg buffer.
    if (ka_fd >= 0) { munmap(ka_host, (size_t)ka_mapped); close(ka_fd); }
    else free(ka_host);
}

// ── LaunchKernel / LaunchKernelConcurrent ────────────────────────────────────

void GPUInterface::LaunchKernel(GPUFunction deviceFunction,
                                 Dim3Int block, Dim3Int grid,
                                 int parameterCountV, int totalParameterCount,
                                 ...) {
    std::vector<GPUPtr>       ptrs(parameterCountV);
    std::vector<unsigned int> ints(totalParameterCount - parameterCountV);
    va_list args;
    va_start(args, totalParameterCount);
    for (int i = 0; i < parameterCountV; ++i)
        ptrs[i] = va_arg(args, GPUPtr);
    for (int i = 0; i < totalParameterCount - parameterCountV; ++i)
        ints[i] = va_arg(args, unsigned int);
    va_end(args);
    LaunchKernelImpl(deviceFunction, block, grid,
                     parameterCountV, totalParameterCount,
                     ptrs.data(), ints.data());
}

void GPUInterface::LaunchKernelConcurrent(GPUFunction deviceFunction,
                                           Dim3Int block, Dim3Int grid,
                                           int /*streamIndex*/, int /*waitIndex*/,
                                           int parameterCountV, int totalParameterCount,
                                           ...) {
    std::vector<GPUPtr>       ptrs(parameterCountV);
    std::vector<unsigned int> ints(totalParameterCount - parameterCountV);
    va_list args;
    va_start(args, totalParameterCount);
    for (int i = 0; i < parameterCountV; ++i)
        ptrs[i] = va_arg(args, GPUPtr);
    for (int i = 0; i < totalParameterCount - parameterCountV; ++i)
        ints[i] = va_arg(args, unsigned int);
    va_end(args);
    LaunchKernelImpl(deviceFunction, block, grid,
                     parameterCountV, totalParameterCount,
                     ptrs.data(), ints.data());
}

// ─────────────────────────────────────────────────────────────────────────────
// §9  Memory management
// ─────────────────────────────────────────────────────────────────────────────

GPUPtr GPUInterface::AllocateMemory(size_t memSize) {
    if (isNVIDIA && nvCuda.lib) {
        uint64_t ptr = 0;
        nvCuda.cuMemAlloc(&ptr, memSize);
        return (GPUPtr)ptr;
    }
    memSize = (memSize + 255ULL) & ~255ULL;
    GPUPtr p = (GPUPtr)vramDataTop;
    vramDataTop += memSize;
    return p;
}

GPUPtr GPUInterface::AllocateRealMemory(size_t n) { return AllocateMemory(SIZE_REAL * n); }
GPUPtr GPUInterface::AllocateIntMemory(size_t n)  { return AllocateMemory(SIZE_INT  * n); }

GPUPtr GPUInterface::CreateSubPointer(GPUPtr dPtr, size_t offset, size_t) {
    return dPtr + (GPUPtr)offset;
}
size_t GPUInterface::AlignMemOffset(size_t offset) { return offset; }

void GPUInterface::FreeMemory(GPUPtr dPtr) {
    if (isNVIDIA && nvCuda.lib)
        nvCuda.cuMemFree((uint64_t)dPtr);
    // AMD: bump allocator — no individual free.
}

size_t GPUInterface::GetAvailableMemory() {
    if (isNVIDIA && nvCuda.lib) {
        size_t free = 0, total = 0;
        nvCuda.cuMemGetInfo(&free, &total);
        return free;
    }
    uint64_t end = tgpuBars[0].size;
    return (vramDataTop < end) ? (size_t)(end - vramDataTop) : 0;
}

// ── Host memory ──────────────────────────────────────────────────────────────

void* GPUInterface::MallocHost(size_t n)              { return malloc(n); }
void* GPUInterface::CallocHost(size_t s, size_t n)    { return calloc(s, n); }

void* GPUInterface::AllocatePinnedHostMemory(size_t memSize, bool, bool) {
    uint64_t mapped = 0; int fd = -1;
    if (!tgpu_rpc_fd(tgpuSock, tgpuDevId, (uint64_t)memSize, 0, &mapped, &fd))
        return nullptr;
    void* p = mmap(nullptr, (size_t)mapped, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) { close(fd); return nullptr; }
    PinnedBuf pb = {p, (size_t)mapped, fd};
    tgpuPinned.push_back(pb);
    return p;
}

void GPUInterface::FreeHostMemory(void* hPtr) {
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
void GPUInterface::FreePinnedHostMemory(void* p) { FreeHostMemory(p); }

GPUPtr GPUInterface::GetDeviceHostPointer(void* hPtr) {
    return (GPUPtr)(uintptr_t)hPtr;
}

// ── Data transfers ────────────────────────────────────────────────────────────

void GPUInterface::MemcpyHostToDevice(GPUPtr dest, const void* src, size_t sz) {
    if (isNVIDIA && nvCuda.lib) {
        nvCuda.cuMemcpyHtoD((uint64_t)dest, src, sz); return;
    }
    tgpu_mmio_write(tgpuSock, tgpuDevId, /*bar=*/0, (uint64_t)dest, src, sz);
}

void GPUInterface::MemcpyDeviceToHost(void* dest, const GPUPtr src, size_t sz) {
    if (isNVIDIA && nvCuda.lib) {
        nvCuda.cuMemcpyDtoH(dest, (uint64_t)src, sz); return;
    }
    tgpu_mmio_read(tgpuSock, tgpuDevId, /*bar=*/0, (uint64_t)src, dest, sz);
}

void GPUInterface::MemcpyDeviceToDevice(GPUPtr dest, GPUPtr src, size_t sz) {
    if (isNVIDIA && nvCuda.lib) {
        nvCuda.cuMemcpyDtoD((uint64_t)dest, (uint64_t)src, sz); return;
    }
    void* tmp = malloc(sz);
    tgpu_mmio_read(tgpuSock,  tgpuDevId, 0, (uint64_t)src,  tmp, sz);
    tgpu_mmio_write(tgpuSock, tgpuDevId, 0, (uint64_t)dest, tmp, sz);
    free(tmp);
}

void GPUInterface::MemsetShort(GPUPtr dest, unsigned short val, size_t count) {
    if (isNVIDIA && nvCuda.lib) {
        nvCuda.cuMemsetD16((uint64_t)dest, val, count); return;
    }
    size_t bytes = count * sizeof(unsigned short);
    uint16_t* buf = (uint16_t*)malloc(bytes);
    for (size_t i = 0; i < count; ++i) buf[i] = val;
    tgpu_mmio_write(tgpuSock, tgpuDevId, 0, (uint64_t)dest, buf, bytes);
    free(buf);
}

// ── Debug ─────────────────────────────────────────────────────────────────────

void GPUInterface::PrintfDeviceInt(GPUPtr dPtr, int length) {
    int* h = (int*)malloc(SIZE_INT * (size_t)length);
    MemcpyDeviceToHost(h, dPtr, SIZE_INT * (size_t)length);
    printfInt(h, length);
    free(h);
}

}; // namespace tinygpu_device
