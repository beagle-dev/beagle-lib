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

// AMD MMIO register read/write via BAR5.
// On AMD RDNA2/3, BAR5 is the MMIO register window (256MB).
// BAR0 is VRAM (resizable), BAR2 is doorbell space.
// Reference: tinygrad/runtime/support/am/amdev.py:
//   self.vram, self.doorbell64, self.mmio = map_bar(0), map_bar(2), map_bar(5)
static void bar0_wr32(int sock, uint32_t dev_id, uint32_t reg_dw, uint32_t val) {
    tgpu_mmio_write(sock, dev_id, 5, (uint64_t)reg_dw * 4, &val, 4);
}
static uint32_t bar0_rd32(int sock, uint32_t dev_id, uint32_t reg_dw) {
    uint32_t v = 0;
    tgpu_mmio_read(sock, dev_id, 5, (uint64_t)reg_dw * 4, &v, 4);
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

struct NVGSPState; // defined in §NV section below; forward-declared for use in ~GPUInterface()
static void nvGspStateDestroy(void* p); // defined after NVGSPState

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
    nvCtx = nullptr; nvModule = nullptr; nvGspState = nullptr;
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
    if (nvGspState) { nvGspStateDestroy(nvGspState); nvGspState = nullptr; }
    for (auto& kv : tgpuKernels) delete kv.second;
    if (kernelResource) delete kernelResource;
    if (resourceMap) delete resourceMap;
    if (tgpuSock >= 0) close(tgpuSock);
}

// ── Chip name helpers ────────────────────────────────────────────────────────

// NVIDIA: read NV_PMC_BOOT_42 (BAR0 byte 0xA00).
// Matches tinygrad nvdev.py lines 112-114:
//   chip_details = reg("NV_PMC_BOOT_42").read_bitfields()
//   chip_name = {0x17:"GA1",0x19:"AD1",0x1b:"GB2"}[architecture] + f"{implementation:02d}"
//   fw_name   = {"GB2":"gb202","AD1":"ad102","GA1":"ga102"}[chip_name[:3]]
// We use lowercase fw_name convention (ga102, ad102, gb202).
static void nv_read_chip_name(int sock, uint32_t dev_id, char out[16]) {
    uint32_t boot42 = 0;
    tgpu_mmio_read(sock, dev_id, 0, 0xA00, &boot42, 4);
    uint32_t arch = (boot42 >> 24) & 0x3F; // bits [24:29]
    uint32_t impl = (boot42 >> 20) & 0xF;  // bits [20:23]
    const char* prefix = (arch == 0x17) ? "ga1" :
                         (arch == 0x19) ? "ad1" :
                         (arch == 0x1b) ? "gb2" : nullptr;
    if (prefix)
        snprintf(out, 16, "%s%02u", prefix, impl);
    else
        snprintf(out, 16, "nv%02x%x", arch, impl);
}

// AMD: derive GFX arch string from PCI device ID.
// Device list from tinygrad ops_amd.py PCIIface (line 848).
static const char* amd_chip_name_from_pci(uint16_t pci_device) {
    switch (pci_device) {
        case 0x74a1: case 0x744c:              return "gfx1100"; // Navi31 RDNA3
        case 0x7590: case 0x7550: case 0x7551: return "gfx1101"; // Navi32 RDNA3
        case 0x7480:                            return "gfx1102"; // Navi33 RDNA3
        case 0x75a0:                            return "gfx1030"; // Navi21 RDNA2
        default:                                return nullptr;
    }
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
        fprintf(stderr, "TinyGPU: unrecognized PCI vendor 0x%04x\n", vendor);
        close(tgpuSock); tgpuSock = -1; return 0;
    }
    isNVIDIA = (vendor == PCI_VENDOR_NVIDIA);

    // Enable bus-master + memory-space in PCI command register.
    uint16_t cmd = (uint16_t)cfg_read(tgpuSock, 0, 0x04, 2);
    cfg_write(tgpuSock, 0, 0x04, 2, cmd | 0x0006);

    // Map vendor-specific BARs via TinyGPU CMD_MAP_BAR.
    //
    // NVIDIA (open-source GSP driver path, matching tinygrad PCIIface / NVDev):
    //   BAR0 = MMIO registers     (tinygrad nvdev.py: self.mmio = map_bar(0))
    //   BAR1 = VRAM               (tinygrad nvdev.py: self.vram = map_bar(1))
    //
    // AMD RDNA2/3 (matching tinygrad AMIface / AMDev):
    //   BAR0 = VRAM (resizable)   (tinygrad amdev.py: self.vram = map_bar(0))
    //   BAR2 = Doorbell space     (tinygrad amdev.py: self.doorbell64 = map_bar(2))
    //   BAR5 = MMIO registers     (tinygrad amdev.py: self.mmio = map_bar(5))
    //
    // Reference: tinygrad/runtime/support/am/amdev.py line 150
    //            tinygrad/runtime/support/nv/nvdev.py lines 75 and 132
    {
        std::vector<int> bars = isNVIDIA
            ? std::vector<int>{0, 1}
            : std::vector<int>{0, 2, 5};
        for (int b : bars) {
            uint64_t a = 0, s = 0;
            if (tgpu_rpc(tgpuSock, 0, CMD_MAP_BAR, 0, 0, 0, (uint32_t)b, &a, &s) && s > 0)
                { tgpuBars[b].addr = a; tgpuBars[b].size = s; }
        }
    }

    DeviceInfo info = {};
    info.dev_id     = 0;
    info.pci_vendor = vendor;
    info.pci_device = device;
    info.supports_dp = true;
    if (isNVIDIA) {
        nv_read_chip_name(tgpuSock, 0, info.chip_name);
    } else {
        const char* n = amd_chip_name_from_pci(device);
        snprintf(info.chip_name, sizeof(info.chip_name), "%s", n ? n : "amdgpu");
    }
    tgpuDevices.push_back(info);
    resourceMap->insert({0, 0});

    if (isNVIDIA) {
        fprintf(stderr, "TinyGPU: NVIDIA %s (%04x:%04x)  BAR0(MMIO)=%llu MB  BAR1(VRAM)=%llu MB\n",
                tgpuDevices.back().chip_name, vendor, device,
                (unsigned long long)(tgpuBars[0].size >> 20),
                (unsigned long long)(tgpuBars[1].size >> 20));
    } else {
        fprintf(stderr, "TinyGPU: AMD %s (%04x:%04x)  BAR0(VRAM)=%llu MB  BAR2(doorbell)=%llu MB  BAR5(MMIO)=%llu MB\n",
                tgpuDevices.back().chip_name, vendor, device,
                (unsigned long long)(tgpuBars[0].size >> 20),
                (unsigned long long)(tgpuBars[2].size >> 20),
                (unsigned long long)(tgpuBars[5].size >> 20));
    }
    return 1;
}

// ── Device info ──────────────────────────────────────────────────────────────

int GPUInterface::GetDeviceCount() { return (int)resourceMap->size(); }

void GPUInterface::GetDeviceName(int dev, char* name, int len) {
    if (dev >= (int)tgpuDevices.size()) { strncpy(name, "TinyGPU", len); return; }
    const DeviceInfo& d = tgpuDevices[dev];
    snprintf(name, len, "%s %s %04x:%04x (TinyGPU)",
             d.pci_vendor == PCI_VENDOR_NVIDIA ? "NVIDIA" : "AMD",
             d.chip_name[0] ? d.chip_name : "unknown",
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
// §NV  NVIDIA open-source GSP driver (macOS + Linux via TinyGPU)
//      Port of tinygrad/runtime/support/nv/{nvdev.py,ip.py} + ops_nv.py.
//
//  Boot order (matching tinygrad NVDev + NV_FLCN + NV_GSP):
//    1. Read chip ID / VRAM size from BAR0 registers
//    2. Parse VBIOS ROM (BAR0 @ 0x300000) for FWSEC ucode  [NV_FLCN.prep_ucode]
//    3. Load booter firmware from disk                       [NV_FLCN.prep_booter]
//    4. Allocate GSP RPC queues + LibOS arg pages            [NV_GSP.init_rm_args]
//    5. Load GSP firmware, build Radix3 scatter table        [NV_GSP.init_gsp_image]
//    6. Load bootloader from disk                            [NV_GSP.init_boot_binary]
//    7. Fill WPR metadata                                    [NV_GSP.init_wpr_meta]
//    8. Execute FWSEC via Falcon → writes WPR2              [NV_FLCN.init_hw §1]
//    9. Load GSP mailbox → booter via SEC2                   [NV_FLCN.init_hw §2]
//   10. Wait for GSP NV_VGPU_MSG_EVENT_GSP_INIT_DONE        [NV_GSP.init_hw]
//   11. rm_alloc hierarchy + channels via GSP RPCs           [NVDevice.__init__]
//   12. Compile PTX→cubin, write to VRAM, build QMD
// ─────────────────────────────────────────────────────────────────────────────

// ── NV BAR0 register byte addresses ──────────────────────────────────────────
static const uint32_t NV_PMC_BOOT_0       = 0x000000;
static const uint32_t NV_PMC_BOOT_42      = 0x000168;
static const uint32_t NV_PFB_MMU_WPR2_HI = 0x1FA828;
static const uint32_t NV_PGC6_SCRATCH42  = 0x1183A4; // vram_size >> 20
static const uint32_t NV_PBUS_BAR1_BLOCK = 0x001704;
static const uint32_t NV_PGSP_ENGINE     = 0x1103C0;
static const uint32_t NV_PSEC_ENGINE     = 0x8403C0;
static const uint32_t NV_PGSP_MAILBOX0  = 0x110040;
static const uint32_t NV_PGSP_MAILBOX1  = 0x110044;
static const uint32_t NV_PGSP_QUEUE_HEAD = 0x110C00;

// Falcon register offsets (add falcon base to get absolute byte addr)
static const uint32_t FLCN_MBX0   = 0x040, FLCN_MBX1 = 0x044;
static const uint32_t FLCN_OS     = 0x080, FLCN_RM   = 0x084;
static const uint32_t FLCN_CPUCTL = 0x100, FLCN_BVEC = 0x104;
static const uint32_t FLCN_DMACTL = 0x10c;
static const uint32_t FLCN_DMABASE= 0x110, FLCN_DMAMOFF= 0x114;
static const uint32_t FLCN_DMACMD = 0x118, FLCN_DMAFBOF= 0x11c;
static const uint32_t FLCN_DMABASE1=0x128, FLCN_HWCFG2= 0x0f4;
static const uint32_t FLCN_CPUCTL_A=0x130;
static const uint32_t FLCN_FBIF_CTL=0x624;  // base+0x600+0x024
static const uint32_t FLCN_FBIF_XC =0x600;  // TRANSCFG[0]
static const uint32_t FLCN2_MODSEL =0x1180, FLCN2_UCID=0x1198;
static const uint32_t FLCN2_ENGID  =0x119c, FLCN2_PARA=0x1210;
static const uint32_t FLCN_RISCV_BCR=0x1668; // base+0x1000+0x668

static const uint32_t NV_GSP_BASE  = 0x110000;
static const uint32_t NV_SEC2_BASE = 0x840000;
static const uint32_t FLCN_DMA_SZ256 = (6u<<8);
static const uint32_t FLCN_DMA_IMEM  = (1u<<4);
static const uint32_t FLCN_DMA_SEC1  = (1u<<2);

// VBIOS ROM constants
static const uint32_t VBIOS_PCI_DATA_OFF = 0x18;
static const uint32_t VBIOS_IMG_LEN_OFF  = 0x10;
static const uint32_t VBIOS_CODE_TYPE_OFF= 0x14;
static const uint32_t VBIOS_BLOCK_SZ    = 512;
static const uint8_t  VBIOS_CODE_BASE   = 0x00;
static const uint8_t  VBIOS_CODE_EXT    = 0xE0;
static const uint8_t  BIT_TOK_FALCON    = 0x70;
static const uint8_t  FWSEC_APPID_PROD = 0x85;
static const uint32_t BIT_SIGNATURE     = 0x00544942;

// GSP RPC function codes (nv.py rpc_fns / rpc_events)
static const uint32_t RPC_ALLOC_MEM    = 4;
static const uint32_t RPC_SET_PAGEDIR  = 54;
static const uint32_t RPC_CONTINU      = 71;
static const uint32_t RPC_GSP_SYSINFO  = 72;
static const uint32_t RPC_SET_REGISTRY = 73;
static const uint32_t RPC_GSP_CTRL     = 76;
static const uint32_t RPC_GSP_ALLOC    = 103;
static const uint32_t EVT_INIT_DONE    = 4097;
static const uint32_t EVT_CPU_SEQ      = 4098;
static const uint32_t EVT_OS_ERR       = 4102;
static const uint32_t RPC_SIG          = 0x42bdbbc3;

// RM class IDs (nv_570.py)
static const uint32_t NVC_ROOT        = 0x00000000;
static const uint32_t NVC_DEV0        = 0x00000080;
static const uint32_t NVC_SUBDEV0     = 0x00002080;
static const uint32_t NVC_MEMVIRT     = 0x00000070;
static const uint32_t NVC_VASPACE_A   = 0x000090f1;
static const uint32_t NVC_CTX_SHARE   = 0x00009067;
static const uint32_t NVC_CHAN_GRP    = 0x0000a06c;
static const uint32_t NVC_CHAN_GPFIFO = 0x0000c56f;
static const uint32_t NVC_CMPUTE_AMP  = 0x0000c7c0;
static const uint32_t NVC_CMPUTE_ADA  = 0x0000c9c0;
static const uint32_t NVC_DMA_AMP     = 0x0000c7b5;
static const uint32_t CTRL_SCHED      = 0xa06c0101;
static const uint32_t CTRL_GET_TOKEN  = 0xc36f0108;

// Compute channel methods
static const uint32_t MET_NON_STALL = 0x0020;
static const uint32_t MET_SEM_ADDRLO= 0x005C, MET_SEM_ADDRHI= 0x0060;
static const uint32_t MET_SEM_PAYLO = 0x0064, MET_SEM_PAYHI = 0x0068;
static const uint32_t MET_SEM_EXEC  = 0x006C;
static const uint32_t MET_INV_SHDR  = 0x1698;
static const uint32_t MET_PCAS_A    = 0x02B4;
static const uint32_t MET_SIGNAL_B  = 0x02C0;
static const uint32_t SEM_OP_RELEASE = 0x1;
static const uint32_t SEM_WFI_EN     = (1u<<20);
static const uint32_t SEM_PAY64     = (1u<<24);

// ── Packed structs ────────────────────────────────────────────────────────────
struct __attribute__((packed)) NvMsgqHdr {
    uint32_t version, size, msgSize, msgCount, writePtr, flags, rxHdrOff, entryOff;
};
struct __attribute__((packed)) NvMsgqElem {
    uint8_t authTag[16], aad[16]; uint32_t checkSum, seqNum, elemCount, pad;
};
struct __attribute__((packed)) NvRpcHdr {
    uint32_t header_version, signature, length, function;
    uint32_t rpc_result, rpc_result_private, sequence, u;
};
struct __attribute__((packed)) NvRpcAllocHdr {
    uint32_t hClient, hParent, hObject, hClass, status, paramsSize, flags;
    uint8_t reserved[4];
};
struct __attribute__((packed)) NvRpcCtrlHdr {
    uint32_t hClient, hObject, cmd, status, paramsSize, flags;
};
struct __attribute__((packed)) NvLibosMem {
    uint64_t id8, pa, size; uint8_t kind, loc;
};
struct __attribute__((packed)) NvBitHdr {
    uint16_t Id; uint32_t Signature; uint16_t BCD;
    uint8_t HdrSz, TokSz, TokCnt, Chksum;
};
struct __attribute__((packed)) NvBitTok {
    uint8_t Id, DataVer; uint16_t DataSize; uint32_t DataPtr;
};
struct __attribute__((packed)) NvFlcnUcodeHdr {
    uint8_t Version, HdrSz, EntrySz, EntryCnt, DescVer, DescSz;
};
struct __attribute__((packed)) NvFlcnUcodeEntry {
    uint8_t AppId, TargetId; uint32_t DescPtr;
};
struct __attribute__((packed)) NvFlcnDescV3 {
    uint32_t vDesc, StoredSize, PKCDataOff, InterfaceOff;
    uint32_t IMEMPhysBase, IMEMLoadSize, IMEMVirtBase;
    uint32_t DMEMPhysBase, DMEMLoadSize;
    uint16_t EngineIdMask; uint8_t UcodeId, SigCnt;
    uint16_t SigVers, Reserved;
};
struct __attribute__((packed)) NvFlcnIfHdr  { uint8_t ver, hdrSz, entrySz, cnt; };
struct __attribute__((packed)) NvFlcnIfEntry{ uint32_t id, dmemOffset; };
struct __attribute__((packed)) NvRiscvDesc {
    uint32_t version, bootloaderOffset, bootloaderSize;
    uint32_t bootloaderParamOffset, bootloaderParamSize;
    uint32_t riscvElfOffset, riscvElfSize, appVersion;
    uint32_t manifestOffset, manifestSize;
    uint32_t monitorDataOffset, monitorDataSize;
    uint32_t monitorCodeOffset, monitorCodeSize;
    uint32_t bIsMonitorEnabled;
    uint32_t swbromCodeOffset, swbromCodeSize;
    uint32_t swbromDataOffset, swbromDataSize;
    uint32_t fbReservedSize, bSignedAsCode;
};
struct __attribute__((packed)) NvMemDescP {
    uint64_t base, size; uint32_t addressSpace, cacheAttrib;
};
struct __attribute__((packed)) NvWprMeta {
    uint64_t magic, revision;
    uint64_t sysmemAddrRadix3, sizeRadix3;
    uint64_t sysmemAddrBootloader, sizeBootloader;
    uint64_t bootloaderCodeOff, bootloaderDataOff, bootloaderManifestOff;
    uint64_t sysmemAddrSig, sizeSig;
    uint64_t gspFwRsvdStart;
    uint64_t nonWprHeapOff, nonWprHeapSize;
    uint64_t gspFwWprStart;
    uint64_t gspFwHeapOff, gspFwHeapSize;
    uint64_t gspFwOff, bootBinOff;
    uint64_t frtsOff, frtsSize;
    uint64_t gspFwWprEnd, fbSize;
    uint64_t vgaWorkspaceOff, vgaWorkspaceSize;
    uint64_t bootCount, partitionRpcAddr;
    uint16_t partitionRpcReqOff, partitionRpcRepOff;
    uint32_t elfCodeOff, elfDataOff, elfCodeSize, elfDataSize;
    uint32_t lsUcodeVersion; uint8_t _pad0[4];
    uint32_t pmuReservedSize; uint64_t verified;
};
static_assert(sizeof(NvWprMeta)==256);
static const uint64_t WPR_META_MAGIC    = 0x0000000074647270ULL;
static const uint64_t WPR_META_REVISION = 2;

// NV_CHANNEL_ALLOC_PARAMS (tinygrad nv_570.py: fields up to offset 272)
struct __attribute__((packed)) NvChanAllocP {
    uint32_t hObjectError, hObjectBuffer;
    uint64_t gpFifoOffset; uint32_t gpFifoEntries, flags;
    uint32_t hContextShare, hVASpace;
    uint32_t hUserdMemory[8]; uint64_t userdOffset[8];
    uint32_t engineType, cid, subDeviceId, hObjectEccError;
    NvMemDescP instanceMem, userdMem, ramfcMem, mthdbufMem;
    uint32_t hPhysChannelGroup, internalFlags;
    NvMemDescP errorNotifierMem;
    uint8_t rest[364-272];
};

// ── NVGSPState ────────────────────────────────────────────────────────────────
struct NVGSPState {
    // Sysmem allocations (to unmap/close on destruction)
    struct Sysmem { void* host; size_t mapped; int fd; uint64_t paddr; };
    std::vector<Sysmem> allocs;

    // VRAM bump allocator (NV: vram starts at 0, grows upward)
    uint64_t vram_top;
    uint64_t vram_size;

    // GSP cmd/stat queue pointers (into first sysmem alloc)
    NvMsgqHdr*         cmd_hdr;   // points into cmd queue host mapping
    void*              cmd_entries;
    NvMsgqHdr*         stat_hdr;
    void*              stat_entries;
    volatile uint32_t* stat_rxptr; // rxHdr readPtr we update
    uint32_t rpc_seq;

    // Boot images
    std::vector<uint8_t> fwsec_patched;
    NvFlcnDescV3 fwsec_desc;
    uint64_t     fwsec_paddr;
    std::vector<uint8_t> booter_bin;
    uint32_t booter_code_off, booter_data_off;
    uint32_t booter_code_sz,  booter_data_sz;
    uint64_t booter_paddr;
    std::vector<uint8_t> gsp_image;
    std::vector<uint64_t> radix3_paddrs;
    uint64_t     gsp_sig_paddr;
    std::vector<uint8_t> gsp_bootloader;
    NvRiscvDesc  gsp_bl_desc;
    uint64_t     gsp_bl_paddr;

    // Key sysmem physical addresses
    uint64_t libos_args_paddr;
    uint64_t wpr_meta_paddr;
    uint64_t rm_args_paddr;
    uint64_t frts_offset;   // VRAM byte offset for FRTS region
    uint64_t eop_paddr;     // completion semaphore physical addr

    // RM handles (32-bit)
    uint32_t handle_gen; // increments; starts at 0xcf000001
    uint32_t priv_root;  // 0xc1e00004
    uint32_t user_root;  // 0xc1000000
    uint32_t dev_h, subdev_h, vaspace_h;
    uint32_t chan_grp_h, ctxshare_h;
    uint32_t gpfifo_area_h, virtmem_h, notifier_h;
    uint32_t compute_chan_h, dma_chan_h;

    // GPFIFO/dispatch state
    uint64_t gpfifo_vram;   // VRAM byte offset of 0x300000-byte GPFIFO area
    uint32_t gpfifo_entries; // = 0x10000
    uint32_t gpfifo_put;
    volatile uint64_t* gpfifo_ring;  // host pointer to ring DWords
    volatile uint32_t* userd_gpput;  // host ptr to USERD GPPut register

    // Dispatch command buffer
    uint64_t cmdq_vram;  // VRAM offset of 2MB command page
    uint32_t cmdq_ptr;

    uint32_t work_token;
    uint32_t compute_class;
    char     fw_name[16]; // "ga102" / "ad102" / "gb202"

    // Compiled kernel state
    uint64_t cubin_vram;
    size_t   cubin_sz;

    // EOP semaphore
    uint64_t eop_signal_val;
    void*    eop_host;

    ~NVGSPState() {
        for (auto& a : allocs)
            { if (a.host) ::munmap(a.host, a.mapped); if (a.fd>=0) ::close(a.fd); }
    }
};

// Defined here (after NVGSPState is complete) so ~GPUInterface() can call it
// via a forward-declared function pointer without deleting an incomplete type.
static void nvGspStateDestroy(void* p) { delete (NVGSPState*)p; }

// ── NV helpers (member functions use tgpuSock/tgpuDevId from GPUInterface) ───

// BAR0 NV MMIO register read/write (bar=0 for NVIDIA)
static void nv_wr32(int sock, uint32_t dev_id, uint32_t byte_addr, uint32_t val)
    { tgpu_mmio_write(sock, dev_id, 0, byte_addr, &val, 4); }
static uint32_t nv_rd32(int sock, uint32_t dev_id, uint32_t byte_addr)
    { uint32_t v=0; tgpu_mmio_read(sock, dev_id, 0, byte_addr, &v, 4); return v; }

// BAR1 VRAM read/write
static void nv_vram_wr(int sock, uint32_t dev_id, uint64_t off, const void* d, size_t sz)
    { tgpu_mmio_write(sock, dev_id, 1, off, d, sz); }
static void nv_vram_rd(int sock, uint32_t dev_id, uint64_t off, void* d, size_t sz)
    { tgpu_mmio_read (sock, dev_id, 1, off, d, sz); }

static uint64_t nv_vram_alloc(NVGSPState* g, uint64_t sz) {
    sz = (sz + 0xfff) & ~0xfffULL;
    uint64_t r = g->vram_top; g->vram_top += sz; return r;
}

// Allocate contiguous host-visible DMA memory via TinyGPU MAP_SYSMEM_FD.
// Returns (host_ptr, first_paddr). Appends to g->allocs.
static bool nv_sysmem_alloc(int sock, uint32_t dev_id, NVGSPState* g,
                              size_t sz, bool contiguous,
                              void** host_out, uint64_t* paddr_out) {
    uint64_t mapped = 0; int fd = -1;
    if (!tgpu_rpc_fd(sock, dev_id, (uint64_t)sz, contiguous ? 1 : 0, &mapped, &fd))
        return false;
    void* p = mmap(nullptr, (size_t)mapped, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) { close(fd); return false; }
    // Physical addresses are in the first N*8 bytes as (paddr,size) pairs terminated by (0,0).
    // (tinygrad APLRemotePCIDevice.alloc_sysmem parses these same pairs)
    uint64_t first_pa = 0;
    {
        auto* mv = (const uint64_t*)p;
        for (size_t i = 0; ; i += 2) {
            uint64_t pa = mv[i], psz = mv[i+1];
            if (pa == 0 && psz == 0) break;
            if (i == 0) first_pa = pa;
        }
    }
    NVGSPState::Sysmem s{p, (size_t)mapped, fd, first_pa};
    g->allocs.push_back(s);
    if (host_out)  *host_out  = p;
    if (paddr_out) *paddr_out = first_pa;
    return true;
}

static void nv_poll(int sock, uint32_t dev_id, uint32_t reg,
                    uint32_t mask, uint32_t expect, const char* msg) {
    for (int t = 0; t < 2000000; ++t) {
        if ((nv_rd32(sock, dev_id, reg) & mask) == expect) return;
        usleep(1);
    }
    fprintf(stderr, "TinyGPU/NV: poll timeout: %s\n", msg);
}

// Load firmware from filesystem. Searches:
//   1. $BEAGLE_TINYGPU_FW/<fw_name>/gsp/<filename>
//   2. ~/.cache/tinygrad/downloads/<sha256> (tinygrad download cache)
//   3. /lib/firmware/nvidia/<fw_name>/gsp/<filename>
// Returns loaded bytes, empty on failure.
static std::vector<uint8_t> nv_load_fw(const char* fw_name, const char* filename,
                                         const char* tinygrad_sha) {
    // Candidate paths
    std::vector<std::string> paths;
    const char* env_path = getenv("BEAGLE_TINYGPU_FW");
    if (env_path) {
        char buf[512];
        snprintf(buf, sizeof(buf), "%s/%s/gsp/%s", env_path, fw_name, filename);
        paths.push_back(buf);
    }
    // tinygrad download cache
    const char* home = getenv("HOME");
    if (home && tinygrad_sha && tinygrad_sha[0]) {
        char buf[512];
        snprintf(buf, sizeof(buf), "%s/.cache/tinygrad/downloads/%s", home, tinygrad_sha);
        paths.push_back(buf);
    }
    char lpath[512];
    snprintf(lpath, sizeof(lpath), "/lib/firmware/nvidia/%s/gsp/%s", fw_name, filename);
    paths.push_back(lpath);

    for (auto& p : paths) {
        FILE* f = fopen(p.c_str(), "rb");
        if (!f) continue;
        fseek(f, 0, SEEK_END); long sz = ftell(f); rewind(f);
        if (sz <= 0) { fclose(f); continue; }
        std::vector<uint8_t> buf(sz);
        fread(buf.data(), 1, sz, f); fclose(f);
        fprintf(stderr, "TinyGPU/NV: loaded %s from %s (%ld bytes)\n", filename, p.c_str(), sz);
        return buf;
    }
    fprintf(stderr, "TinyGPU/NV: firmware '%s/%s' not found.\n"
            "  Set BEAGLE_TINYGPU_FW=<dir> or ensure tinygrad has cached it in\n"
            "  ~/.cache/tinygrad/downloads/\n", fw_name, filename);
    return {};
}

// ── Falcon operations ─────────────────────────────────────────────────────────

static void falcon_reset(int sock, uint32_t dev_id, uint32_t base, bool riscv) {
    uint32_t eng_reg = (base == NV_GSP_BASE) ? NV_PGSP_ENGINE : NV_PSEC_ENGINE;
    nv_wr32(sock, dev_id, eng_reg, 1); // reset=1
    usleep(100000); // 100 ms
    nv_wr32(sock, dev_id, eng_reg, 0); // reset=0
    nv_poll(sock, dev_id, base+FLCN_HWCFG2, (1u<<12), 0, "falcon scrub");

    if (riscv) {
        // core_select=RISCV(1), brfetch=1, valid=0
        nv_wr32(sock, dev_id, base+FLCN_RISCV_BCR, (1u<<4)|(1u<<8));
    } else if (nv_rd32(sock, dev_id, base+FLCN_HWCFG2) & (1u<<10)) {
        // riscv-capable but boot as Falcon: core_select=FALCON(0)
        nv_wr32(sock, dev_id, base+FLCN_RISCV_BCR, 0);
        nv_poll(sock, dev_id, base+FLCN_RISCV_BCR, 1, 1, "falcon riscv valid");
        nv_wr32(sock, dev_id, base+FLCN_RM, nv_rd32(sock, dev_id, NV_PMC_BOOT_0));
    }
}

// DMA transfer: load 256-byte blocks from src_paddr to Falcon IMEM or DMEM.
static void falcon_dma(int sock, uint32_t dev_id, uint32_t base,
                        uint32_t cmd, uint32_t dest, uint32_t fboffs,
                        uint64_t src_paddr, uint32_t sz) {
    nv_wr32(sock, dev_id, base+FLCN_DMABASE,  (uint32_t)(src_paddr>>8) & 0xFFFFFFFF);
    nv_wr32(sock, dev_id, base+FLCN_DMABASE1, (uint32_t)((src_paddr>>8)>>32) & 0x1FF);
    for (uint32_t xfered = 0; xfered < sz; xfered += 256) {
        nv_poll(sock, dev_id, base+FLCN_DMACMD, 1, 0, "dma !full");
        nv_wr32(sock, dev_id, base+FLCN_DMAMOFF, dest + xfered);
        nv_wr32(sock, dev_id, base+FLCN_DMAFBOF, fboffs + xfered);
        nv_wr32(sock, dev_id, base+FLCN_DMACMD, cmd);
    }
    nv_poll(sock, dev_id, base+FLCN_DMACMD, 2, 2, "dma idle");
}

// Execute a high-security ucode image via Falcon (NV_FLCN.execute_hs).
static void falcon_execute_hs(int sock, uint32_t dev_id, uint32_t base,
                               uint64_t img_paddr, uint32_t code_off,
                               uint32_t data_off,
                               uint32_t imem_pa, uint32_t imem_va, uint32_t imem_sz,
                               uint32_t dmem_pa, uint32_t dmem_va, uint32_t dmem_sz,
                               uint32_t pkc_off, uint32_t engid, uint32_t ucodeid,
                               uint64_t mailbox = 0) {
    // allow_phys_no_ctx
    nv_wr32(sock, dev_id, base+FLCN_FBIF_CTL,
            nv_rd32(sock, dev_id, base+FLCN_FBIF_CTL) | (1u<<7));
    nv_wr32(sock, dev_id, base+FLCN_DMACTL, 0);

    // TRANSCFG[0]: target=FB(0), mem_type=PHYSICAL(1)
    nv_wr32(sock, dev_id, base+FLCN_FBIF_XC, (1u<<2));

    // Load IMEM
    uint32_t imem_cmd = FLCN_DMA_SZ256 | FLCN_DMA_IMEM | FLCN_DMA_SEC1;
    falcon_dma(sock, dev_id, base, imem_cmd, imem_pa, imem_va,
               img_paddr + code_off - imem_va, imem_sz);

    // Load DMEM
    uint32_t dmem_cmd = FLCN_DMA_SZ256;
    falcon_dma(sock, dev_id, base, dmem_cmd, dmem_pa, dmem_va,
               img_paddr + data_off - dmem_va, dmem_sz);

    // BROM params
    nv_wr32(sock, dev_id, base+FLCN2_PARA, pkc_off);
    nv_wr32(sock, dev_id, base+FLCN2_ENGID, engid);
    nv_wr32(sock, dev_id, base+FLCN2_UCID, ucodeid);
    nv_wr32(sock, dev_id, base+FLCN2_MODSEL, 1); // RSA3K

    nv_wr32(sock, dev_id, base+FLCN_BVEC, imem_va);

    if (mailbox) {
        nv_wr32(sock, dev_id, base+FLCN_MBX0, (uint32_t)(mailbox & 0xFFFFFFFF));
        nv_wr32(sock, dev_id, base+FLCN_MBX1, (uint32_t)(mailbox >> 32));
    }

    // Start CPU
    if (nv_rd32(sock, dev_id, base+FLCN_CPUCTL) & (1u<<6))
        nv_wr32(sock, dev_id, base+FLCN_CPUCTL_A, 2); // alias startcpu
    else
        nv_wr32(sock, dev_id, base+FLCN_CPUCTL, 2);   // startcpu bit[1]

    // Wait for halt
    nv_poll(sock, dev_id, base+FLCN_CPUCTL, (1u<<4), (1u<<4), "falcon halted");
}

// ── GSP RPC queue ─────────────────────────────────────────────────────────────

// XOR-based checksum (tinygrad NVRpcQueue._checksum)
static uint32_t rpc_checksum(const void* data, size_t sz) {
    // Pad to 8-byte boundary
    const uint8_t* p = (const uint8_t*)data;
    uint64_t ck = 0;
    size_t full = (sz + 7) & ~7ULL;
    for (size_t i = 0; i < full; i += 8) {
        uint64_t w = 0;
        size_t nb = (i + 8 <= sz) ? 8 : (sz - i);
        memcpy(&w, p + i, nb);
        ck ^= w;
    }
    return (uint32_t)(ck >> 32) ^ (uint32_t)(ck & 0xFFFFFFFF);
}

// Send one GSP RPC record (NVRpcQueue._send_rpc_record).
static void gsp_rpc_send(int sock, uint32_t dev_id, NVGSPState* g,
                          uint32_t func, const void* payload, size_t pay_sz) {
    uint32_t msg_size = g->cmd_hdr->msgSize; // 0x1000
    uint32_t msg_count= g->cmd_hdr->msgCount;

    // Build NvRpcHdr + payload
    std::vector<uint8_t> msg(sizeof(NvRpcHdr) + pay_sz);
    auto* hdr = (NvRpcHdr*)msg.data();
    hdr->header_version    = (3u << 24);
    hdr->signature         = RPC_SIG;
    hdr->length            = (uint32_t)(msg.size() + 0x20); // +sizeof(NvMsgqElem)
    hdr->function          = func;
    hdr->rpc_result        = 0xFFFF0001; // RPC_PENDING
    hdr->rpc_result_private= 0xFFFF0001;
    hdr->sequence          = g->rpc_seq;
    hdr->u                 = 0;
    if (pay_sz) memcpy(msg.data() + sizeof(NvRpcHdr), payload, pay_sz);

    // Wrap in NvMsgqElem
    size_t total = sizeof(NvMsgqElem) + msg.size();
    uint32_t elem_cnt = (uint32_t)((total + msg_size - 1) / msg_size);
    std::vector<uint8_t> full_msg(elem_cnt * msg_size, 0);

    NvMsgqElem* elem = (NvMsgqElem*)full_msg.data();
    elem->seqNum   = g->rpc_seq;
    elem->elemCount= elem_cnt;
    // Write rpc msg after elem header
    memcpy(full_msg.data() + sizeof(NvMsgqElem), msg.data(), msg.size());
    // Compute checksum over (elem_header + msg)
    elem->checkSum = rpc_checksum(full_msg.data(), sizeof(NvMsgqElem) + msg.size());

    // Write to ring (with wrap-around)
    uint32_t wp = g->cmd_hdr->writePtr;
    uint8_t* base = (uint8_t*)g->cmd_entries;
    for (uint32_t i = 0; i < elem_cnt; ++i) {
        uint32_t slot = (wp + i) % msg_count;
        memcpy(base + slot * msg_size, full_msg.data() + i * msg_size, msg_size);
    }
    __sync_synchronize();
    g->cmd_hdr->writePtr = (wp + elem_cnt) % msg_count;
    __sync_synchronize();
    g->rpc_seq++;

    // Kick GSP
    nv_wr32(sock, dev_id, NV_PGSP_QUEUE_HEAD, 0);
}

// Read one response message from stat queue. Returns (function, payload_bytes).
// Returns false if no message available.
static bool gsp_rpc_recv_one(NVGSPState* g, uint32_t* func_out,
                              std::vector<uint8_t>* payload_out) {
    auto* shdr = g->stat_hdr;
    uint32_t wptr = shdr->writePtr;
    uint32_t rptr = *g->stat_rxptr;
    if (wptr == rptr) return false;

    uint32_t msg_size  = shdr->msgSize;
    uint32_t msg_count = shdr->msgCount;
    uint8_t* entries   = (uint8_t*)g->stat_entries;

    uint8_t* slot_data = entries + rptr * msg_size;
    NvMsgqElem* elem = (NvMsgqElem*)slot_data;
    NvRpcHdr* rhdr = (NvRpcHdr*)(slot_data + sizeof(NvMsgqElem));

    if (func_out) *func_out = rhdr->function;
    if (payload_out) {
        size_t pay_off = sizeof(NvMsgqElem) + sizeof(NvRpcHdr);
        size_t pay_sz  = (rhdr->length > sizeof(NvRpcHdr)) ? rhdr->length - sizeof(NvRpcHdr) : 0;
        payload_out->assign(slot_data + pay_off, slot_data + pay_off + pay_sz);
    }

    // Advance read pointer
    *g->stat_rxptr = (rptr + elem->elemCount) % msg_count;
    __sync_synchronize();
    return true;
}

// Wait for a specific RPC response, handling async events (CPU sequencer etc).
// Returns true and fills payload_out on success.
static bool gsp_rpc_wait(int sock, uint32_t dev_id, NVGSPState* g,
                          uint32_t expected_func,
                          std::vector<uint8_t>* payload_out = nullptr) {
    for (int t = 0; t < 10000000; ++t) {
        uint32_t func; std::vector<uint8_t> pay;
        if (!gsp_rpc_recv_one(g, &func, &pay)) { usleep(10); continue; }

        if (func == EVT_OS_ERR) {
            if (pay.size() > 12) {
                pay.push_back(0);
                fprintf(stderr, "TinyGPU/NV: GSP log: %s\n", (char*)pay.data()+12);
            }
            return false;
        }
        if (func == EVT_CPU_SEQ) {
            // CPU sequencer: handle register write/modify/poll ops from GSP
            // (tinygrad NV_GSP.run_cpu_seq — minimal implementation)
            if (pay.size() >= 8) {
                uint32_t cmd_idx; memcpy(&cmd_idx, pay.data()+4, 4);
                const uint32_t* cmds = (const uint32_t*)(pay.data() + 8);
                for (uint32_t i = 0; i < cmd_idx; ) {
                    uint32_t op = cmds[i++];
                    if (op == 0 && i+2 <= cmd_idx) { // reg write
                        nv_wr32(sock, dev_id, cmds[i]*4, cmds[i+1]); i+=2;
                    } else if (op == 1 && i+3 <= cmd_idx) { // reg modify
                        uint32_t addr=cmds[i]*4, val=cmds[i+1], mask=cmds[i+2]; i+=3;
                        nv_wr32(sock,dev_id,addr,(nv_rd32(sock,dev_id,addr)&~mask)|(val&mask));
                    } else if (op == 2 && i+5 <= cmd_idx) { // reg poll
                        uint32_t addr=cmds[i]*4,mask=cmds[i+1],val=cmds[i+2]; i+=5;
                        nv_poll(sock,dev_id,addr,mask,val,"cpu_seq poll");
                    } else if (op == 3 && i+1 <= cmd_idx) { // delay us
                        usleep(cmds[i++]);
                    } else if (op == 5) { // Falcon reset
                        falcon_reset(sock, dev_id, NV_GSP_BASE, false);
                    } else if (op == 6) { // Falcon start cpu
                        nv_wr32(sock, dev_id, NV_GSP_BASE+FLCN_CPUCTL, 2);
                    } else if (op == 7) { // wait halted
                        nv_poll(sock,dev_id,NV_GSP_BASE+FLCN_CPUCTL,(1u<<4),(1u<<4),"cpu halted");
                    } else if (op == 8) { // core resume (RISCV + set mailbox)
                        falcon_reset(sock, dev_id, NV_GSP_BASE, true);
                        nv_wr32(sock, dev_id, NV_PGSP_MAILBOX0, (uint32_t)(g->libos_args_paddr&0xFFFFFFFF));
                        nv_wr32(sock, dev_id, NV_PGSP_MAILBOX1, (uint32_t)(g->libos_args_paddr>>32));
                        nv_wr32(sock, dev_id, NV_SEC2_BASE+FLCN_CPUCTL, 2);
                    } else break;
                }
            }
            continue;
        }
        if (func == expected_func) {
            if (payload_out) *payload_out = pay;
            return true;
        }
        // Unhandled event: keep draining
    }
    fprintf(stderr, "TinyGPU/NV: timeout waiting for RPC 0x%x\n", expected_func);
    return false;
}

// Send RPC + wait for same function response (most rm_alloc / rm_control calls).
static bool gsp_rpc(int sock, uint32_t dev_id, NVGSPState* g,
                    uint32_t func, const void* payload, size_t pay_sz,
                    std::vector<uint8_t>* resp_out = nullptr) {
    gsp_rpc_send(sock, dev_id, g, func, payload, pay_sz);
    return gsp_rpc_wait(sock, dev_id, g, func, resp_out);
}

// ── GSP rm_alloc via RPC ──────────────────────────────────────────────────────
static uint32_t gsp_rm_alloc(int sock, uint32_t dev_id, NVGSPState* g,
                               uint32_t hParent, uint32_t hClass,
                               const void* params, uint32_t params_sz,
                               uint32_t client = 0) {
    if (!client) client = g->user_root;
    uint32_t hObject = g->handle_gen++;

    size_t total = sizeof(NvRpcAllocHdr) + params_sz;
    std::vector<uint8_t> msg(total, 0);
    auto* h = (NvRpcAllocHdr*)msg.data();
    h->hClient    = client;
    h->hParent    = hParent;
    h->hObject    = hObject;
    h->hClass     = hClass;
    h->paramsSize = params_sz;
    if (params && params_sz) memcpy(msg.data() + sizeof(NvRpcAllocHdr), params, params_sz);

    if (!gsp_rpc(sock, dev_id, g, RPC_GSP_ALLOC, msg.data(), msg.size()))
        return 0;
    return hObject;
}

// ── GSP rm_control via RPC ───────────────────────────────────────────────────
static bool gsp_rm_ctrl(int sock, uint32_t dev_id, NVGSPState* g,
                         uint32_t hObject, uint32_t cmd,
                         void* params, uint32_t params_sz,
                         uint32_t client = 0) {
    if (!client) client = g->user_root;
    size_t total = sizeof(NvRpcCtrlHdr) + params_sz;
    std::vector<uint8_t> msg(total, 0);
    auto* h = (NvRpcCtrlHdr*)msg.data();
    h->hClient    = client;
    h->hObject    = hObject;
    h->cmd        = cmd;
    h->paramsSize = params_sz;
    if (params && params_sz) memcpy(msg.data() + sizeof(NvRpcCtrlHdr), params, params_sz);

    std::vector<uint8_t> resp;
    if (!gsp_rpc(sock, dev_id, g, RPC_GSP_CTRL, msg.data(), msg.size(), &resp)) return false;
    // Copy response params back
    size_t resp_off = sizeof(NvRpcCtrlHdr);
    if (params && params_sz && resp.size() >= resp_off + params_sz)
        memcpy(params, resp.data() + resp_off, params_sz);
    return true;
}

// ── VBIOS parsing (NV_FLCN.prep_ucode) ───────────────────────────────────────

static bool nv_parse_vbios_fwsec(int sock, uint32_t dev_id, NVGSPState* g,
                                   uint64_t vram_size) {
    // Read 1 MB VBIOS expansion ROM from BAR0 @ 0x300000
    static const uint32_t VBIOS_SZ = 0x100000;
    std::vector<uint8_t> vbios(VBIOS_SZ);
    nv_vram_rd(sock, dev_id, 0x300000, vbios.data(), VBIOS_SZ); // Actually BAR0, not VRAM
    // tinygrad reads from mmio[0x300000//4 : ], which is BAR0 offset 0x300000
    tgpu_mmio_read(sock, dev_id, 0, 0x300000, vbios.data(), VBIOS_SZ);

    // Walk PCI expansion ROM blocks to find VBIOS_EXT (FWSEC lives there)
    uint32_t vbios_off = 0, block_size = 0, expansion_rom_off = 0;
    bool found_ext = false;
    for (int iter = 0; iter < 32 && !found_ext; ++iter) {
        if (vbios_off + 0x20 > VBIOS_SZ) break;
        uint16_t pci_data_off;
        memcpy(&pci_data_off, vbios.data() + vbios_off + VBIOS_PCI_DATA_OFF, 2);
        uint32_t pci_abs = vbios_off + pci_data_off;
        if (pci_abs + 0x18 > VBIOS_SZ) break;
        uint16_t len_blocks; memcpy(&len_blocks, vbios.data() + pci_abs + VBIOS_IMG_LEN_OFF, 2);
        uint32_t imglen = (uint32_t)len_blocks * VBIOS_BLOCK_SZ;
        uint8_t code_type = vbios[pci_abs + VBIOS_CODE_TYPE_OFF];
        if (code_type == VBIOS_CODE_BASE) block_size = imglen;
        else if (code_type == VBIOS_CODE_EXT) {
            expansion_rom_off = vbios_off - block_size;
            found_ext = true; break;
        }
        vbios_off += imglen;
        if (imglen == 0) break;
    }
    if (!found_ext) { fprintf(stderr,"TinyGPU/NV: VBIOS EXT block not found\n"); return false; }

    // BIT header at fixed offset 0x1b0 in VBIOS
    if (0x1b0 + (int)sizeof(NvBitHdr) > (int)VBIOS_SZ) return false;
    auto* bithdr = (const NvBitHdr*)(vbios.data() + 0x1b0);
    if (bithdr->Signature != BIT_SIGNATURE)
        { fprintf(stderr,"TinyGPU/NV: invalid BIT signature\n"); return false; }

    // Find FALCON_DATA token (0x70)
    uint32_t bit_addr = 0x1b0;
    bool found_falcon = false;
    uint32_t falcon_table_ptr = 0;
    for (uint8_t i = 0; i < bithdr->TokCnt && !found_falcon; ++i) {
        uint32_t tok_off = bit_addr + bithdr->HdrSz + i * bithdr->TokSz;
        if (tok_off + sizeof(NvBitTok) > VBIOS_SZ) break;
        auto* tok = (const NvBitTok*)(vbios.data() + tok_off);
        if (tok->Id == BIT_TOK_FALCON && tok->DataVer == 2 && tok->DataSize >= 4) {
            uint16_t data_ptr16 = (uint16_t)tok->DataPtr;
            if (data_ptr16 + 4 <= VBIOS_SZ) {
                memcpy(&falcon_table_ptr, vbios.data() + data_ptr16, 4);
                found_falcon = true;
            }
        }
    }
    if (!found_falcon) { fprintf(stderr,"TinyGPU/NV: FALCON BIT token not found\n"); return false; }

    // Parse ucode table
    uint32_t table_ptr = expansion_rom_off + falcon_table_ptr;
    if (table_ptr + sizeof(NvFlcnUcodeHdr) > VBIOS_SZ) return false;
    auto* uh = (const NvFlcnUcodeHdr*)(vbios.data() + table_ptr);
    bool found_fwsec = false;
    for (uint8_t j = 0; j < uh->EntryCnt && !found_fwsec; ++j) {
        uint32_t e_off = table_ptr + uh->HdrSz + j * uh->EntrySz;
        if (e_off + sizeof(NvFlcnUcodeEntry) > VBIOS_SZ) break;
        auto* ue = (const NvFlcnUcodeEntry*)(vbios.data() + e_off);
        if (ue->AppId != FWSEC_APPID_PROD) continue;

        uint32_t desc_off = expansion_rom_off + ue->DescPtr;
        if (desc_off + sizeof(NvFlcnDescV3) > VBIOS_SZ) break;
        uint32_t vDesc; memcpy(&vDesc, vbios.data() + desc_off, 4);
        uint32_t desc_sz = vDesc >> 16;
        if (desc_sz < sizeof(NvFlcnDescV3) || desc_off + desc_sz > VBIOS_SZ) break;
        memcpy(&g->fwsec_desc, vbios.data() + desc_off, sizeof(NvFlcnDescV3));

        uint32_t sig_sz = desc_sz - 44; // FALCON_UCODE_DESC_V3_SIZE_44 = 44
        uint32_t image_sz = (g->fwsec_desc.StoredSize + 255) & ~255u;
        uint32_t img_off = desc_off + desc_sz;
        if (img_off + image_sz > VBIOS_SZ) break;

        // Patch image: fill FRTS command into DMEM mapper (NV_FLCN.prep_ucode __patch)
        g->fwsec_patched.assign(vbios.data()+img_off, vbios.data()+img_off+image_sz);
        // Copy signature (last 0x180 bytes of descriptor) into PKCDataOffset of image
        uint32_t sig_off_in_img = g->fwsec_desc.IMEMLoadSize + g->fwsec_desc.PKCDataOff;
        if (sig_sz >= 0x180 && sig_off_in_img + 0x180 <= g->fwsec_patched.size()) {
            memcpy(g->fwsec_patched.data() + sig_off_in_img,
                   vbios.data() + desc_off + 44 + (sig_sz - 0x180), 0x180);
        }

        // Patch FRTS command into DMEM interface mapper
        uint32_t if_off = g->fwsec_desc.IMEMLoadSize + g->fwsec_desc.InterfaceOff;
        if (if_off + sizeof(NvFlcnIfHdr) <= g->fwsec_patched.size()) {
            auto* ifhdr = (NvFlcnIfHdr*)(g->fwsec_patched.data() + if_off);
            uint32_t dmem_mapper_off = g->fwsec_desc.IMEMLoadSize;
            for (uint8_t ei = 0; ei < ifhdr->cnt; ++ei) {
                uint32_t e2_off = if_off + ifhdr->hdrSz + ei * ifhdr->entrySz;
                if (e2_off + sizeof(NvFlcnIfEntry) > g->fwsec_patched.size()) break;
                auto* ife = (NvFlcnIfEntry*)(g->fwsec_patched.data() + e2_off);
                if (ife->id == 4) { // DMEMMAPPER
                    dmem_mapper_off = g->fwsec_desc.IMEMLoadSize + ife->dmemOffset;
                    break;
                }
            }
            // Set init_cmd = 0x15 (FWSECLIC_FRTS_CMD)
            // and write the FRTS command into cmd_in_buffer_offset
            if (dmem_mapper_off + 48 <= g->fwsec_patched.size()) {
                uint32_t cmd_in_buf_off;
                memcpy(&cmd_in_buf_off, g->fwsec_patched.data() + dmem_mapper_off + 8, 4);
                // init_cmd is at offset 44
                uint32_t init_cmd_val = 0x15;
                memcpy(g->fwsec_patched.data() + dmem_mapper_off + 44, &init_cmd_val, 4);

                // Write NvFwseclicFrtsCmd at cmd_in_buffer_offset
                g->frts_offset = vram_size - 0x100000 - 0x100000;
                struct { uint32_t v,s; uint64_t gfwOff; uint32_t gfwSz,flags; // read_vbios_desc
                         uint32_t v2,s2,frts4K,frtsSize,frtsMed; // frts_region_desc
                } frts_cmd = {};
                frts_cmd.v=1; frts_cmd.s=24; frts_cmd.flags=2;
                frts_cmd.v2=1; frts_cmd.s2=20;
                frts_cmd.frts4K=(uint32_t)(g->frts_offset>>12);
                frts_cmd.frtsSize=0x100; frts_cmd.frtsMed=2;

                uint32_t cmd_abs = g->fwsec_desc.IMEMLoadSize + cmd_in_buf_off;
                if (cmd_abs + sizeof(frts_cmd) <= g->fwsec_patched.size())
                    memcpy(g->fwsec_patched.data() + cmd_abs, &frts_cmd, sizeof(frts_cmd));
            }
        }
        found_fwsec = true;
    }
    if (!found_fwsec) { fprintf(stderr,"TinyGPU/NV: FWSEC ucode not found\n"); return false; }
    return true;
}

// ── GSP memory setup (NV_GSP.init_rm_args + init_libos_args + init_wpr_meta) ─

static bool nv_setup_gsp_memory(int sock, uint32_t dev_id, NVGSPState* g) {
    static const uint32_t QUEUE_SZ = 0x40000; // 256 KB per queue

    // Allocate combined page-table + cmd queue + stat queue
    uint32_t queue_pte_cnt = (QUEUE_SZ * 2) / 0x1000;
    uint32_t pt_pages = (queue_pte_cnt * 8 + 0xfff) / 0x1000;
    uint32_t pte_cnt  = queue_pte_cnt + pt_pages;
    uint32_t pt_size  = ((pte_cnt * 8) + 0xfff) & ~0xfffU;
    size_t   total    = pt_size + QUEUE_SZ * 2;

    void* queues_host; uint64_t queues_paddr;
    if (!nv_sysmem_alloc(sock, dev_id, g, total, false, &queues_host, &queues_paddr))
        return false;

    // Fill page-table entries: write each 4KB page's paddr into the pt area
    // The actual paddrs are in the first portion of the mmap (as reported by TinyGPU)
    auto* pt_arr = (uint64_t*)queues_host;
    // paddrs stored at the very beginning of the mapping by TinyGPU
    // (tinygrad itertools.takewhile until (paddr=0,size=0))
    // Copy them to the pt region
    uint64_t cur_paddr = queues_paddr;
    for (uint32_t i = 0; i < pte_cnt; ++i, cur_paddr += 0x1000)
        pt_arr[i] = cur_paddr;

    // GSP_ARGUMENTS_CACHED
    struct __attribute__((packed)) MsgQueueInitArgs {
        uint64_t sharedMemPhysAddr;
        uint32_t pageTableEntryCount, cmdQueueOffset, statQueueOffset;
        uint8_t  _pad[12];
    };
    struct __attribute__((packed)) GspArgsCached {
        MsgQueueInitArgs mqia;  // offset 0, 32 bytes
        uint8_t sr_init[12];   // offset 32
        uint32_t gpuInstance;  // offset 44
        uint8_t bDmemStack;    // offset 48
        uint8_t _pad2[7];
        struct { uint64_t pa, size; } profilerArgs; // offset 56
    };
    GspArgsCached args_cached = {};
    args_cached.mqia.sharedMemPhysAddr = queues_paddr;
    args_cached.mqia.pageTableEntryCount = pte_cnt;
    args_cached.mqia.cmdQueueOffset  = pt_size;
    args_cached.mqia.statQueueOffset = pt_size + QUEUE_SZ;
    args_cached.bDmemStack = 1;

    void* rm_args_host; uint64_t rm_args_pa;
    if (!nv_sysmem_alloc(sock, dev_id, g, sizeof(GspArgsCached), true, &rm_args_host, &rm_args_pa))
        return false;
    memcpy(rm_args_host, &args_cached, sizeof(GspArgsCached));
    g->rm_args_paddr = rm_args_pa;

    // Initialize cmd queue header at (queues_host + pt_size)
    uint8_t* cmd_base  = (uint8_t*)queues_host + pt_size;
    uint8_t* stat_base = (uint8_t*)queues_host + pt_size + QUEUE_SZ;
    g->cmd_hdr  = (NvMsgqHdr*)cmd_base;
    g->stat_hdr = (NvMsgqHdr*)stat_base;
    g->cmd_hdr->version  = 0;
    g->cmd_hdr->size     = QUEUE_SZ;
    g->cmd_hdr->entryOff = 0x1000;
    g->cmd_hdr->msgSize  = 0x1000;
    g->cmd_hdr->msgCount = (QUEUE_SZ - 0x1000) / 0x1000;
    g->cmd_hdr->writePtr = 0;
    g->cmd_hdr->flags    = 1;
    g->cmd_hdr->rxHdrOff = sizeof(NvMsgqHdr);
    g->cmd_entries = cmd_base + 0x1000;

    // For stat queue, we read from it; the header will be initialized by GSP
    g->stat_entries = stat_base + 0x1000;
    // rxHdr read pointer is inside the stat queue header at rxHdrOff
    // We'll set these up once GSP writes its header
    g->stat_rxptr = nullptr; // set after GSP boots

    return true;
}

static bool nv_setup_libos_args(int sock, uint32_t dev_id, NVGSPState* g) {
    void* logbuf_host; uint64_t logbuf_pa;
    if (!nv_sysmem_alloc(sock, dev_id, g, 2<<20, false, &logbuf_host, &logbuf_pa))
        return false;

    void* libos_host; uint64_t libos_pa;
    if (!nv_sysmem_alloc(sock, dev_id, g, 0x1000, true, &libos_host, &libos_pa))
        return false;
    g->libos_args_paddr = libos_pa;

    // LibosMemoryRegion entries (NV_GSP.init_libos_args)
    // 5 log regions + 1 RMARGS region = 6 entries
    static const char* names[] = {"LOGINIT","LOGINTR","LOGRM  ","LOGMNOC","LOGKRNL"};
    std::vector<uint8_t> buf;
    auto push_mem = [&](uint64_t id8, uint64_t pa, uint64_t sz, uint8_t kind, uint8_t loc) {
        NvLibosMem m{}; m.id8=id8; m.pa=pa; m.size=sz; m.kind=kind; m.loc=loc;
        buf.insert(buf.end(), (uint8_t*)&m, (uint8_t*)&m+sizeof(m));
    };
    for (int i = 0; i < 5; ++i) {
        uint64_t id8 = 0;
        for (int k = 0; k < 8 && names[i][k]; ++k)
            id8 = (id8 << 8) | (uint8_t)names[i][k];
        push_mem(id8, logbuf_pa + 0x10000*(uint64_t)i, 0x10000, 1/*CONTIGUOUS*/, 1/*SYSMEM*/);
    }
    uint64_t rmargs_id = 0; const char* r="RMARGS";
    for (int k=0;k<6;++k) rmargs_id=(rmargs_id<<8)|(uint8_t)r[k];
    push_mem(rmargs_id, g->rm_args_paddr, 0x1000, 1, 1);

    memcpy(libos_host, buf.data(), std::min(buf.size(), (size_t)0x1000));
    return true;
}

static bool nv_setup_wpr_meta(int sock, uint32_t dev_id, NVGSPState* g) {
    // Load GSP firmware (nv_570 driver, "gsp-570.144.bin")
    const char* gsp_sha = (strcmp(g->fw_name,"ga102")==0)
        ? "a8c3ebeed280323aedb51c061f321e73379cce7a9ae643a33dd03915df027f7f"
        : (strcmp(g->fw_name,"ad102")==0)
        ? "n/a_ad102" : "n/a";
    auto gsp_elf = nv_load_fw(g->fw_name, "gsp-570.144.bin", gsp_sha);
    if (gsp_elf.empty()) return false;

    // Parse ELF for .fwimage and .fwsignature_ga1x sections
    const uint8_t* elf_d = gsp_elf.data();
    size_t elf_sz = gsp_elf.size();
    const uint8_t* fwimage = nullptr; size_t fwimage_sz = 0;
    const uint8_t* fwsig   = nullptr; size_t fwsig_sz   = 0;
    char sig_sec_name[64];
    snprintf(sig_sec_name, sizeof(sig_sec_name), ".fwsignature_%.*sx", 4, g->fw_name);

    // Minimal ELF64 parse
    if (elf_sz >= 64) {
        uint64_t shoff; memcpy(&shoff, elf_d+40, 8);
        uint16_t shnum, shstrndx; memcpy(&shnum,&elf_d[60],2); memcpy(&shstrndx,&elf_d[62],2);
        if (shoff + (uint64_t)shnum*64 <= elf_sz) {
            const uint8_t* shdrs = elf_d + shoff;
            uint64_t strtab_off; memcpy(&strtab_off, shdrs + shstrndx*64 + 24, 8);
            const char* strtab = (const char*)(elf_d + strtab_off);
            for (uint16_t i=0; i<shnum; ++i) {
                const uint8_t* sh = shdrs + i*64;
                uint32_t sh_name; memcpy(&sh_name,sh,4);
                uint64_t sh_off, sh_size; memcpy(&sh_off,sh+24,8); memcpy(&sh_size,sh+32,8);
                const char* sname = strtab + sh_name;
                if (!strcmp(sname,".fwimage"))    { fwimage=elf_d+sh_off; fwimage_sz=sh_size; }
                else if (!strncmp(sname,".fwsignature_",13)) { fwsig=elf_d+sh_off; fwsig_sz=sh_size; }
            }
        }
    }
    if (!fwimage) { fprintf(stderr,"TinyGPU/NV: .fwimage section not found\n"); return false; }
    g->gsp_image.assign(fwimage, fwimage + fwimage_sz);

    // Build Radix3 scatter table (NV_GSP.init_gsp_image)
    uint32_t n_pages = (uint32_t)((fwimage_sz + 0xfff) / 0x1000);
    uint32_t npages[4] = {0, 0, 0, n_pages};
    for (int i=3; i>0; --i) npages[i-1] = ((npages[i]-1) >> (12-3)) + 1;
    uint64_t offsets[4];
    offsets[0]=0; for(int i=1;i<4;++i) offsets[i]=offsets[i-1]+npages[i-1]*0x1000ULL;
    size_t radix_total = offsets[3] + fwimage_sz;

    void* radix_host; uint64_t radix_pa;
    if (!nv_sysmem_alloc(sock, dev_id, g, radix_total, false, &radix_host, &radix_pa))
        return false;

    // Copy image
    uint8_t* rbase = (uint8_t*)radix_host;
    memcpy(rbase + offsets[3], fwimage, fwimage_sz);

    // Build radix pages (collect actual paddrs from allocation)
    // We'll use simple sequential paddr assumption (contiguous allocation above)
    auto* radix_q = (uint64_t*)rbase;
    uint64_t cur_pa = radix_pa;
    uint64_t img_page_base = radix_pa + offsets[3];
    for (uint32_t i=0; i<npages[3]; ++i)
        radix_q[(int)(offsets[2]/8) + i] = img_page_base + (uint64_t)i*0x1000;
    for (uint32_t i=0; i<npages[2]; ++i)
        radix_q[(int)(offsets[1]/8) + i] = cur_pa + offsets[2] + (uint64_t)i*0x1000;
    for (uint32_t i=0; i<npages[1]; ++i)
        radix_q[(int)(offsets[0]/8) + i] = cur_pa + offsets[1] + (uint64_t)i*0x1000;
    g->radix3_paddrs.push_back(radix_pa);

    // Signature
    if (fwsig && fwsig_sz) {
        void* sig_host; uint64_t sig_pa;
        if (nv_sysmem_alloc(sock, dev_id, g, fwsig_sz, true, &sig_host, &sig_pa)) {
            memcpy(sig_host, fwsig, fwsig_sz);
            g->gsp_sig_paddr = sig_pa;
        }
    }

    // Load bootloader (gsp_bl_desc)
    const char* bl_sha = (strcmp(g->fw_name,"ga102")==0)
        ? "82428f532240727e95bb3083fbaaba9b2cc7b937314323f2d546ce7245f27fad"
        : "n/a";
    auto bl_data = nv_load_fw(g->fw_name, "bootloader-570.144.bin", bl_sha);
    if (bl_data.empty()) return false;

    // Parse bootloader header (struct_nvfw_bin_hdr + RM_RISCV_UCODE_DESC)
    if (bl_data.size() < 16) return false;
    uint32_t hdr_offset, data_offset, data_size;
    memcpy(&hdr_offset, bl_data.data()+4, 4);
    memcpy(&data_offset, bl_data.data()+8, 4);
    memcpy(&data_size,   bl_data.data()+12, 4);
    if (hdr_offset + sizeof(NvRiscvDesc) > bl_data.size()) return false;
    memcpy(&g->gsp_bl_desc, bl_data.data() + hdr_offset, sizeof(NvRiscvDesc));
    g->gsp_bootloader.assign(bl_data.data()+data_offset, bl_data.data()+data_offset+data_size);

    void* bl_host; uint64_t bl_pa;
    if (!nv_sysmem_alloc(sock, dev_id, g, g->gsp_bootloader.size(), false, &bl_host, &bl_pa))
        return false;
    memcpy(bl_host, g->gsp_bootloader.data(), g->gsp_bootloader.size());
    g->gsp_bl_paddr = bl_pa;

    // Build WPR metadata
    uint64_t vram_sz = g->vram_size;
    uint64_t vga_sz  = 0x100000;
    uint64_t vga_off = vram_sz - vga_sz;
    uint64_t frts_sz = 0x100000;
    uint64_t frts_off= vga_off - frts_sz;
    uint64_t boot_sz = (uint64_t)g->gsp_bootloader.size();
    uint64_t boot_off= frts_off - boot_sz;
    uint64_t gsp_sz  = (uint64_t)fwimage_sz;
    uint64_t gsp_off = (boot_off - gsp_sz) & ~0xffffULL;
    uint64_t heap_sz = 0x8100000;
    uint64_t heap_off= (gsp_off - heap_sz) & ~0xfffffULL;
    uint64_t wpr_start=(heap_off - 0x1000) & ~0xfffffULL;
    uint64_t nonwpr_sz= 0x100000;
    uint64_t nonwpr_off=(wpr_start - nonwpr_sz) & ~0xfffffULL;

    NvWprMeta wpr = {};
    wpr.magic            = WPR_META_MAGIC;
    wpr.revision         = WPR_META_REVISION;
    wpr.sysmemAddrRadix3 = radix_pa;
    wpr.sizeRadix3       = (uint64_t)fwimage_sz;
    wpr.sysmemAddrBootloader = g->gsp_bl_paddr;
    wpr.sizeBootloader   = (uint64_t)g->gsp_bootloader.size();
    wpr.bootloaderCodeOff= g->gsp_bl_desc.monitorCodeOffset;
    wpr.bootloaderDataOff= g->gsp_bl_desc.monitorDataOffset;
    wpr.bootloaderManifestOff = g->gsp_bl_desc.manifestOffset;
    wpr.sysmemAddrSig    = g->gsp_sig_paddr;
    wpr.sizeSig          = 0x1000;
    wpr.gspFwRsvdStart   = nonwpr_off;
    wpr.nonWprHeapOff    = nonwpr_off;
    wpr.nonWprHeapSize   = nonwpr_sz;
    wpr.gspFwWprStart    = wpr_start;
    wpr.gspFwHeapOff     = heap_off;
    wpr.gspFwHeapSize    = heap_sz;
    wpr.gspFwOff         = gsp_off;
    wpr.bootBinOff       = boot_off;
    wpr.frtsOff          = frts_off;
    wpr.frtsSize         = frts_sz;
    wpr.gspFwWprEnd      = vga_off;
    wpr.fbSize           = vram_sz;
    wpr.vgaWorkspaceOff  = vga_off;
    wpr.vgaWorkspaceSize = vga_sz;
    g->frts_offset       = frts_off;

    void* wpr_host; uint64_t wpr_pa;
    if (!nv_sysmem_alloc(sock, dev_id, g, sizeof(NvWprMeta), true, &wpr_host, &wpr_pa))
        return false;
    memcpy(wpr_host, &wpr, sizeof(NvWprMeta));
    g->wpr_meta_paddr = wpr_pa;
    return true;
}

// ── GSP channel setup (NVDevice.__init__ via GSP RPCs) ───────────────────────

static bool nv_channel_setup(int sock, uint32_t dev_id, NVGSPState* g) {
    // Prefill GSP SET_SYSTEM_INFO + SET_REGISTRY before booting
    // (called inside NV_GSP.init_sw from tinygrad; we call it after boot here)

    // rm_alloc root (NV01_ROOT)
    {
        struct { uint32_t clientInfo[4]; } p = {};
        g->user_root = g->handle_gen++;
        // Use priv_root as starting point; tinygrad calls rpc_rm_alloc(hParent=0x0, hClass=NV01_ROOT...)
        // We use our own handle gen
        --g->handle_gen;
        NvRpcAllocHdr h = {};
        h.hClient  = g->priv_root;
        h.hParent  = 0x0;
        h.hObject  = g->user_root;
        h.hClass   = NVC_ROOT;
        h.paramsSize = 0;
        gsp_rpc_send(sock, dev_id, g, RPC_GSP_ALLOC, &h, sizeof(h));
        gsp_rpc_wait(sock, dev_id, g, RPC_GSP_ALLOC);
    }

    // Alloc NV01_DEVICE_0
    struct { uint32_t deviceId, hClientShare, hTargetClient, hTargetDevice;
             uint32_t flags, vaSpaceSzLo, vaSpaceSzHi;
             uint32_t vaStartLo,vaStartHi,vaLimLo,vaLimHi,vaMode; } dp = {};
    dp.deviceId = 0; dp.hClientShare = g->user_root;
    g->dev_h = gsp_rm_alloc(sock, dev_id, g, g->user_root, NVC_DEV0, &dp, sizeof(dp));

    // Alloc NV20_SUBDEVICE_0
    uint32_t sd_params = 0; // subDeviceId = 0
    g->subdev_h = gsp_rm_alloc(sock, dev_id, g, g->dev_h, NVC_SUBDEV0, &sd_params, 4);

    // Alloc NV01_MEMORY_VIRTUAL
    struct { uint64_t limit; uint32_t flags; } mv = {};
    mv.limit = 0x1ffffffffffff;
    g->virtmem_h = gsp_rm_alloc(sock, dev_id, g, g->dev_h, NVC_MEMVIRT, &mv, sizeof(mv));

    // Alloc FERMI_VASPACE_A
    struct { uint32_t index, flags; uint64_t vaSize, vaStart, vaLim; uint32_t bigPgSz, pad; uint64_t vaBase; } vp = {};
    vp.index = 0;
    vp.flags = (1<<2) | (1<<5); // ENABLE_PAGE_FAULTING | EXTERNALLY_OWNED
    vp.vaSize= 0x1fffffb000000ULL;
    vp.vaBase= 0x1000;
    g->vaspace_h = gsp_rm_alloc(sock, dev_id, g, g->dev_h, NVC_VASPACE_A, &vp, sizeof(vp));

    // Alloc KEPLER_CHANNEL_GROUP_A
    struct { uint32_t engineType, hObjectError, hVASpace; } cg = {};
    cg.engineType = 0; // NV2080_ENGINE_TYPE_GRAPHICS = 0
    cg.hVASpace   = g->vaspace_h;
    g->chan_grp_h = gsp_rm_alloc(sock, dev_id, g, g->dev_h, NVC_CHAN_GRP, &cg, sizeof(cg));

    // Alloc FERMI_CONTEXT_SHARE_A
    struct { uint32_t hVASpace, flags, subctxId; } cs = {};
    cs.hVASpace = g->vaspace_h; cs.flags = 1; // SUBCONTEXT_ASYNC
    g->ctxshare_h = gsp_rm_alloc(sock, dev_id, g, g->chan_grp_h, NVC_CTX_SHARE, &cs, sizeof(cs));

    // Allocate GPFIFO area in VRAM (0x300000 bytes)
    static const uint32_t GPFIFO_MEM_SZ = 0x300000;
    g->gpfifo_vram = nv_vram_alloc(g, GPFIFO_MEM_SZ);
    // Zero it
    {
        std::vector<uint8_t> zero(GPFIFO_MEM_SZ, 0);
        nv_vram_wr(sock, dev_id, g->gpfifo_vram, zero.data(), GPFIFO_MEM_SZ);
    }

    // Notify handle (use a VRAM alloc handle — simplified)
    g->notifier_h = g->handle_gen++;
    g->gpfifo_area_h = g->handle_gen++;

    // Alloc AMPERE_CHANNEL_GPFIFO_A (compute channel)
    static const uint32_t GPFIFO_ENTRIES = 0x10000;
    uint64_t gpfifo_ring_off = 0;            // ring at GPFIFO area offset 0
    uint64_t userd_off = (uint64_t)GPFIFO_ENTRIES * 8;  // USERD after ring
    uint64_t cmdq_off  = 0x200000;           // command buffer at offset 0x200000

    NvChanAllocP cp = {};
    cp.hObjectError = g->notifier_h;
    cp.hObjectBuffer = g->gpfifo_area_h;
    cp.gpFifoOffset = g->gpfifo_vram;    // GPU VA = VRAM offset (identity-mapped)
    cp.gpFifoEntries = GPFIFO_ENTRIES;
    cp.flags = 0x200320;                 // standard GPFIFO flags
    cp.hContextShare = g->ctxshare_h;
    cp.hVASpace = g->vaspace_h;
    for (int i=0; i<8; ++i) {
        cp.hUserdMemory[i] = g->gpfifo_area_h;
        cp.userdOffset[i]  = userd_off;
    }
    cp.engineType = 0;
    cp.internalFlags = 0x1a;

    // Allocate VRAM backing for instanceMem (4KB) and mthdbufMem (20KB)
    uint64_t instance_vram = nv_vram_alloc(g, 0x1000);
    uint64_t mthdbuf_vram  = nv_vram_alloc(g, 0x5000);

    cp.instanceMem = {instance_vram, 0x1000, 2, 0};
    cp.userdMem    = {g->gpfifo_vram + userd_off, 0x400, 2, 0};
    cp.ramfcMem    = {instance_vram, 0x200, 2, 0};
    cp.mthdbufMem  = {mthdbuf_vram,  0x5000, 2, 0};
    cp.errorNotifierMem = {0, 0xecc, 0, 0};

    g->compute_chan_h = gsp_rm_alloc(sock, dev_id, g, g->chan_grp_h, NVC_CHAN_GPFIFO, &cp, sizeof(cp));
    if (!g->compute_chan_h) return false;

    // Alloc compute class on channel
    gsp_rm_alloc(sock, dev_id, g, g->compute_chan_h, g->compute_class, nullptr, 0);

    // Schedule channel group
    uint32_t sched_en = 1;
    gsp_rm_ctrl(sock, dev_id, g, g->chan_grp_h, CTRL_SCHED, &sched_en, 4);

    // Get work submit token
    struct { uint32_t workSubmitToken; } wt = {(uint32_t)-1};
    if (!gsp_rm_ctrl(sock, dev_id, g, g->compute_chan_h, CTRL_GET_TOKEN, &wt, 4))
        return false;
    g->work_token = wt.workSubmitToken;
    fprintf(stderr, "TinyGPU/NV: workSubmitToken=0x%08x\n", g->work_token);

    // Map host view into GPFIFO area for ring + USERD
    {
        void* h2; uint64_t pa2;
        if (!nv_sysmem_alloc(sock, dev_id, g, GPFIFO_MEM_SZ, false, &h2, &pa2)) return false;
        g->gpfifo_ring  = (volatile uint64_t*)h2;
        g->userd_gpput  = (volatile uint32_t*)((uint8_t*)h2 + userd_off + 0x8C);
        g->gpfifo_entries = GPFIFO_ENTRIES;
        g->gpfifo_put     = 0;
    }
    g->cmdq_vram    = g->gpfifo_vram + cmdq_off;
    g->cmdq_ptr     = 0;

    // Allocate EOP semaphore page (host-visible, GPU-accessible)
    {
        void* eop_h; uint64_t eop_pa;
        if (nv_sysmem_alloc(sock, dev_id, g, 0x1000, true, &eop_h, &eop_pa)) {
            memset(eop_h, 0, 0x1000);
            g->eop_paddr      = eop_pa;
            g->eop_host       = eop_h;
            g->eop_signal_val = 0;
        }
    }

    fprintf(stderr, "TinyGPU/NV: channels ready (compute=0x%x)\n", g->compute_chan_h);
    return true;
}

// ── nvSetup() — complete boot sequence ────────────────────────────────────────

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
    // ── 1. Detect chip ────────────────────────────────────────────────────────
    uint32_t chip_id = nv_rd32(tgpuSock, tgpuDevId, NV_PMC_BOOT_0);
    uint32_t boot42  = nv_rd32(tgpuSock, tgpuDevId, NV_PMC_BOOT_42);
    uint32_t arch = ((boot42 >> 24) & 0x1F) | (((boot42 >> 8) & 1) << 5); // arch bits
    uint32_t impl = (boot42 >> 20) & 0xF;

    // Determine firmware family name and compute class
    const char* fw_name   = "ga102"; // fallback
    uint32_t compute_cls  = NVC_CMPUTE_AMP;
    if (arch == 0x19) { fw_name = "ad102"; compute_cls = NVC_CMPUTE_ADA; }

    fprintf(stderr, "TinyGPU/NV: chip_id=0x%08x arch=0x%x impl=0x%x fw=%s\n",
            chip_id, arch, impl, fw_name);

    // ── 2. Check WPR2 (if set, GPU needs reset) ───────────────────────────────
    uint32_t wpr2_hi = nv_rd32(tgpuSock, tgpuDevId, NV_PFB_MMU_WPR2_HI);
    if (wpr2_hi != 0)
        fprintf(stderr, "TinyGPU/NV: WPR2 already set (0x%x) — may need PCI reset\n", wpr2_hi);

    // ── 3. Read VRAM size ─────────────────────────────────────────────────────
    uint64_t vram_size = (uint64_t)nv_rd32(tgpuSock, tgpuDevId, NV_PGC6_SCRATCH42) << 20;
    fprintf(stderr, "TinyGPU/NV: VRAM %llu MB\n", (unsigned long long)(vram_size >> 20));
    if (vram_size == 0) vram_size = tgpuBars[1].size; // fallback to BAR1 size

    // ── 4. Allocate GSP state ────────────────────────────────────────────────
    auto* g = new NVGSPState();
    nvGspState = g;
    g->vram_size     = vram_size;
    g->vram_top      = 0; // grow upward from 0 (VRAM VA)
    g->handle_gen    = 0xcf000001;
    g->priv_root     = 0xc1e00004;
    g->user_root     = 0xc1000000;
    g->rpc_seq       = 0;
    g->compute_class = compute_cls;
    g->work_token    = 0;
    strncpy(g->fw_name, fw_name, sizeof(g->fw_name)-1);

    // ── 5. Load booter (NV_FLCN.prep_booter) ─────────────────────────────────
    const char* booter_sha = (!strcmp(fw_name,"ga102"))
        ? "4497e3eff7e95c774b8a569d17b27c08c9650158d10b229d2be81cdcad9a085b"
        : (!strcmp(fw_name,"ad102"))
        ? "8b293e19b637c5e22c87a2428d1c71bb13e0904e8a88ac6b3c6c1f2679c6e37a"
        : "n/a";
    auto booter_data = nv_load_fw(fw_name, "booter_load-570.144.bin", booter_sha);
    if (booter_data.empty()) goto fail;

    // Parse booter: struct_nvfw_bin_hdr (hdr_offset at +4, data_offset at +8, data_size at +12)
    // struct_nvfw_hs_header_v2 at hdr_offset; struct_nvfw_hs_load_header_v2 inside it
    {
        uint32_t hdr_off, data_off, data_sz;
        memcpy(&hdr_off,  booter_data.data()+4, 4);
        memcpy(&data_off, booter_data.data()+8, 4);
        memcpy(&data_sz,  booter_data.data()+12,4);
        // struct_nvfw_hs_header_v2: header_offset(4), patch_loc(4), patch_sig(4), num_sig(4), sig_prod_offset(4), sig_prod_size(4)
        uint32_t hs_hdr_off, patch_loc, patch_sig, num_sig, sig_prod_off, sig_prod_sz;
        memcpy(&hs_hdr_off, booter_data.data()+hdr_off,   4);
        memcpy(&patch_loc,  booter_data.data()+hdr_off+4, 4);
        memcpy(&patch_sig,  booter_data.data()+hdr_off+8, 4);
        memcpy(&num_sig,    booter_data.data()+hdr_off+12,4);
        memcpy(&sig_prod_off,booter_data.data()+hdr_off+16,4);
        memcpy(&sig_prod_sz, booter_data.data()+hdr_off+20,4);
        uint32_t n_sigs; memcpy(&n_sigs, booter_data.data()+num_sig, 4);
        uint32_t sig_len = (n_sigs > 0) ? sig_prod_sz / n_sigs : sig_prod_sz;

        // Patch: overwrite patch_loc in image with first signature chunk
        g->booter_bin.assign(booter_data.data()+data_off, booter_data.data()+data_off+data_sz);
        if (patch_loc + sig_len <= g->booter_bin.size() && sig_prod_off + sig_len <= booter_data.size())
            memcpy(g->booter_bin.data()+patch_loc, booter_data.data()+sig_prod_off, sig_len);

        // struct_nvfw_hs_load_header_v2 at hs_hdr_off inside header area:
        // os_data_offset(4), os_data_size(4) ... and app offset/size
        // Simplified: os_data_offset at +0, os_data_size at +4, app at struct_nvfw_hs_load_header_v2_app
        uint32_t lh_off = hdr_off + hs_hdr_off; // header_offset inside the hdr section
        if (lh_off + 8 <= booter_data.size()) {
            memcpy(&g->booter_data_off, booter_data.data()+lh_off,   4);
            memcpy(&g->booter_data_sz,  booter_data.data()+lh_off+4, 4);
        }
        // app struct follows: offset(4), size(4)
        if (lh_off + 16 <= booter_data.size()) {
            memcpy(&g->booter_code_off, booter_data.data()+lh_off+8, 4);
            memcpy(&g->booter_code_sz,  booter_data.data()+lh_off+12,4);
        }

        void* bh; uint64_t bp;
        if (!nv_sysmem_alloc(tgpuSock, tgpuDevId, g, g->booter_bin.size(), false, &bh, &bp)) goto fail;
        memcpy(bh, g->booter_bin.data(), g->booter_bin.size());
        g->booter_paddr = bp;
    }

    // ── 6. Parse VBIOS for FWSEC (NV_FLCN.prep_ucode) ───────────────────────
    if (!nv_parse_vbios_fwsec(tgpuSock, tgpuDevId, g, vram_size)) goto fail;
    {
        void* fh; uint64_t fp;
        if (!nv_sysmem_alloc(tgpuSock, tgpuDevId, g, g->fwsec_patched.size(), false, &fh, &fp)) goto fail;
        memcpy(fh, g->fwsec_patched.data(), g->fwsec_patched.size());
        g->fwsec_paddr = fp;
    }

    // ── 7. Set up GSP memory (queues + libos args + WPR meta) ────────────────
    if (!nv_setup_gsp_memory(tgpuSock, tgpuDevId, g)) goto fail;
    if (!nv_setup_libos_args(tgpuSock, tgpuDevId, g)) goto fail;
    if (!nv_setup_wpr_meta(tgpuSock, tgpuDevId, g))   goto fail;

    // ── 8. Pre-fill GSP system info + registry RPCs (NV_GSP.init_sw) ─────────
    // These are queued into cmd before GSP boots; GSP processes them on startup.
    {
        // GspSystemInfo (928 bytes) — fill key fields, rest zero
        std::vector<uint8_t> sysinfo(928, 0);
        auto set64 = [&](int off, uint64_t v) { memcpy(sysinfo.data()+off,&v,8); };
        auto set32 = [&](int off, uint32_t v) { memcpy(sysinfo.data()+off,&v,4); };
        set64(0,  tgpuBars[0].addr);  // gpuPhysAddr (BAR0)
        set64(8,  tgpuBars[1].addr);  // gpuPhysFbAddr (BAR1/VRAM)
        set64(16, 0);                  // gpuPhysInstAddr (BAR3, unknown)
        set64(72, 0x7ffffffff000ULL); // maxUserVa
        set32(80, 0x88000);           // pciConfigMirrorBase
        set32(84, 0x1000);            // pciConfigMirrorSize
        uint32_t pci_vid_dev = (tgpuDevices[0].pci_device << 16) | tgpuDevices[0].pci_vendor;
        set32(88, pci_vid_dev);       // PCIDeviceID
        set32(96, (uint32_t)cfg_read(tgpuSock, tgpuDevId, 0x08, 1)); // PCIRevisionID
        sysinfo[840] = 1; // bIsPassthru
        gsp_rpc_send(tgpuSock, tgpuDevId, g, RPC_GSP_SYSINFO, sysinfo.data(), sysinfo.size());

        // Registry table: RMForcePcieConfigSave=1, RMSecBusResetEnable=1
        // Minimal packed registry (nv.py PACKED_REGISTRY_TABLE)
        struct { uint32_t size, numEntries; } reg_hdr = {0,0};
        // 2 entries * 16 bytes header + 2 * (nameOffset+type+data+length) + strings
        std::string k1 = "RMForcePcieConfigSave", k2 = "RMSecBusResetEnable";
        uint32_t hdr_sz = 8, entry_sz = 16, n_entries = 2;
        uint32_t entries_off = hdr_sz;
        uint32_t strings_off = hdr_sz + n_entries * entry_sz;
        uint32_t str1_off = strings_off;
        uint32_t str2_off = str1_off + (uint32_t)k1.size() + 1;
        uint32_t total_reg = str2_off + (uint32_t)k2.size() + 1;
        reg_hdr.size = total_reg; reg_hdr.numEntries = n_entries;
        std::vector<uint8_t> reg_buf(total_reg, 0);
        memcpy(reg_buf.data(), &reg_hdr, 8);
        // Entry: nameOffset(4), type(4)=1(DWORD), data(4)=1, length(4)=4
        auto we = [&](int i, uint32_t nameOff) {
            uint32_t off = entries_off + i * entry_sz;
            memcpy(reg_buf.data()+off,   &nameOff, 4);
            uint32_t type=1,data=1,len=4;
            memcpy(reg_buf.data()+off+4, &type,4);
            memcpy(reg_buf.data()+off+8, &data,4);
            memcpy(reg_buf.data()+off+12,&len, 4);
        };
        we(0, str1_off); we(1, str2_off);
        memcpy(reg_buf.data()+str1_off, k1.c_str(), k1.size());
        memcpy(reg_buf.data()+str2_off, k2.c_str(), k2.size());
        gsp_rpc_send(tgpuSock, tgpuDevId, g, RPC_SET_REGISTRY, reg_buf.data(), reg_buf.size());
    }

    // ── 9. Boot Falcon: execute FWSEC → WPR2 (NV_FLCN.init_hw step 1) ────────
    falcon_reset(tgpuSock, tgpuDevId, NV_GSP_BASE, false);
    falcon_execute_hs(tgpuSock, tgpuDevId, NV_GSP_BASE, g->fwsec_paddr, 0,
        g->fwsec_desc.IMEMLoadSize,
        g->fwsec_desc.IMEMPhysBase, g->fwsec_desc.IMEMVirtBase, g->fwsec_desc.IMEMLoadSize,
        g->fwsec_desc.DMEMPhysBase, 0, g->fwsec_desc.DMEMLoadSize,
        g->fwsec_desc.PKCDataOff, g->fwsec_desc.EngineIdMask, g->fwsec_desc.UcodeId);
    {
        uint32_t wpr2 = nv_rd32(tgpuSock, tgpuDevId, NV_PFB_MMU_WPR2_HI);
        if (wpr2 == 0) { fprintf(stderr,"TinyGPU/NV: WPR2 not set after FWSEC\n"); goto fail; }
        fprintf(stderr, "TinyGPU/NV: WPR2 set (hi=0x%x)\n", wpr2);
    }

    // ── 10. Set GSP mailbox + boot via SEC2 (NV_FLCN.init_hw step 2) ──────────
    falcon_reset(tgpuSock, tgpuDevId, NV_GSP_BASE, true); // reset to RISCV mode
    nv_wr32(tgpuSock, tgpuDevId, NV_PGSP_MAILBOX0, (uint32_t)(g->libos_args_paddr & 0xFFFFFFFF));
    nv_wr32(tgpuSock, tgpuDevId, NV_PGSP_MAILBOX1, (uint32_t)(g->libos_args_paddr >> 32));

    falcon_reset(tgpuSock, tgpuDevId, NV_SEC2_BASE, false);
    falcon_execute_hs(tgpuSock, tgpuDevId, NV_SEC2_BASE, g->booter_paddr,
        g->booter_code_off, g->booter_data_off,
        0, g->booter_code_off, g->booter_code_sz,
        0, 0, g->booter_data_sz,
        0x10, 1, 3, g->wpr_meta_paddr);
    {
        uint32_t mbx0 = nv_rd32(tgpuSock, tgpuDevId, NV_SEC2_BASE + FLCN_MBX0);
        if (mbx0 != 0) { fprintf(stderr,"TinyGPU/NV: booter failed mbx=0x%x\n",mbx0); goto fail; }
    }

    // Wait for GSP RISC-V to be active
    nv_poll(tgpuSock, tgpuDevId, NV_GSP_BASE+FLCN_RISCV_BCR,
            (1u<<0), (1u<<0), "GSP RISCV active"); // valid=1 indicates core active

    // ── 11. Wait for GSP init done (NV_GSP.init_hw) ───────────────────────────
    // GSP processes queued sysinfo/registry RPCs then fires GSP_INIT_DONE event.
    // We need to wire up stat_rxptr first (stat queue header written by GSP).
    {
        // GSP writes its own msgqTxHeader into the stat queue; wait for entryOff!=0
        for (int t = 0; t < 1000000 && g->stat_hdr->entryOff == 0; ++t) usleep(10);
        if (g->stat_hdr->entryOff == 0) { fprintf(stderr,"TinyGPU/NV: stat queue not initialized\n"); goto fail; }

        NvMsgqHdr* stat_hdr = g->stat_hdr;
        uint32_t rx_off = stat_hdr->rxHdrOff;
        // The rxHdr (a msgqRxHeader = just a uint32_t readPtr) is at rxHdrOff inside stat_hdr
        g->stat_rxptr = (volatile uint32_t*)((uint8_t*)g->stat_hdr + rx_off);
        g->stat_entries = (uint8_t*)g->stat_hdr + stat_hdr->entryOff;

        if (!gsp_rpc_wait(tgpuSock, tgpuDevId, g, EVT_INIT_DONE)) goto fail;
        fprintf(stderr, "TinyGPU/NV: GSP init done\n");
    }

    // Disable BAR1 block (needed for GSP to work with VRAM)
    nv_wr32(tgpuSock, tgpuDevId, NV_PBUS_BAR1_BLOCK, 0);

    // ── 12. Allocate RM hierarchy via GSP RPCs ────────────────────────────────
    if (!nv_channel_setup(tgpuSock, tgpuDevId, g)) goto fail;

    // ── 13. Compile PTX→cubin and load into VRAM ─────────────────────────────
    {
        size_t csz = 0;
        char* cubin = ptx_to_cubin(kernelResource->kernelCode,
                                    strlen(kernelResource->kernelCode), &csz);
        if (!cubin) { fprintf(stderr,"TinyGPU/NV: ptxas failed\n"); goto fail; }
        g->cubin_vram = nv_vram_alloc(g, csz);
        g->cubin_sz   = csz;
        nv_vram_wr(tgpuSock, tgpuDevId, g->cubin_vram, cubin, csz);
        free(cubin);
        // Parse cubin ELF for kernel entries
        {
            std::vector<uint8_t> elf_buf(csz);
            nv_vram_rd(tgpuSock, tgpuDevId, g->cubin_vram, elf_buf.data(), csz);
            // Walk ELF sections for .text.kernelName
            if (csz >= 64) {
                uint64_t shoff; memcpy(&shoff, elf_buf.data()+40, 8);
                uint16_t shnum, shstrndx;
                memcpy(&shnum,    elf_buf.data()+60, 2);
                memcpy(&shstrndx, elf_buf.data()+62, 2);
                if (shoff + (uint64_t)shnum*64 <= csz) {
                    const uint8_t* shdrs = elf_buf.data() + shoff;
                    uint64_t strtab_off; memcpy(&strtab_off, shdrs+shstrndx*64+24, 8);
                    const char* strtab = (const char*)(elf_buf.data() + strtab_off);
                    for (uint16_t i=0; i<shnum; ++i) {
                        const uint8_t* sh = shdrs + i*64;
                        uint32_t sh_name; uint64_t sh_off;
                        memcpy(&sh_name, sh,    4);
                        memcpy(&sh_off,  sh+24, 8);
                        const char* sname = strtab + sh_name;
                        if (!strncmp(sname, ".text.", 6)) {
                            const char* kname = sname + 6;
                            auto* entry = new KernelEntry();
                            entry->name = kname;
                            entry->code_vaddr = g->cubin_vram + sh_off;
                            entry->cu_func    = nullptr;
                            entry->rsrc1 = entry->rsrc2 = entry->rsrc3 = 0;
                            entry->wave32 = false;
                            tgpuKernels[kname] = entry;
                        }
                    }
                }
            }
        }
        fprintf(stderr,"TinyGPU/NV: cubin %zu bytes in VRAM, %zu kernels\n",
                csz, tgpuKernels.size());
    }

    // Store state for LaunchKernelImpl
    nvWorkToken     = g->work_token;
    nvGpfifoEntries = g->gpfifo_entries;
    nvGpfifoPut     = g->gpfifo_put;
    nvGpfifoHost    = (uint64_t*)g->gpfifo_ring;
    nvUserdGpPut    = g->userd_gpput;
    nvCubinVramBase = g->cubin_vram;
    nvCubinSize     = g->cubin_sz;
    return;

fail:
    fprintf(stderr,"TinyGPU/NV: nvSetup() failed\n");
    delete g; nvGspState = nullptr;
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
    uint64_t fb_base = (uint64_t)(bar0_rd32(tgpuSock, tgpuDevId, AMD_FB_LOCATION_BASE) & 0xFFFFFF) << 24;
    uint64_t fb_end  = ((uint64_t)(bar0_rd32(tgpuSock, tgpuDevId, AMD_FB_LOCATION_TOP) & 0xFFFFFF) + 1) << 24;
    amdFbBase = fb_base;
    fprintf(stderr, "TinyGPU/AMD: VRAM physical 0x%llx..0x%llx (%llu MB)\n",
            (unsigned long long)fb_base, (unsigned long long)fb_end,
            (unsigned long long)((fb_end - fb_base) >> 20));

    // ── 2. Detect boot state (partial = GPU was previously init'd by macOS) ───
    uint32_t scratch7  = bar0_rd32(tgpuSock, tgpuDevId, AMD_SCRATCH_REG7);
    uint32_t scratch6  = bar0_rd32(tgpuSock, tgpuDevId, AMD_SCRATCH_REG6);
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
        auto* g = (NVGSPState*)nvGspState;
        if (!g || !g->eop_paddr || !g->eop_host) return;
        volatile uint64_t* sig = (volatile uint64_t*)g->eop_host;
        for (int t = 0; t < 10000000 && *sig < g->eop_signal_val; ++t) usleep(10);
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

    // NV GSP path: kernel entries are populated by nvSetup() ELF parse.

    tgpuKernels[name] = entry;
    return (GPUFunction)entry;
}

// ── LaunchKernel (internal helper) ───────────────────────────────────────────

void GPUInterface::LaunchKernelImpl(GPUFunction deviceFunction,
                                     Dim3Int block, Dim3Int grid,
                                     int nPtr, int nTotal,
                                     GPUPtr* ptrs, unsigned int* ints) {
    KernelEntry* entry = (KernelEntry*)deviceFunction;

    // ── NVIDIA: GPFIFO + QMD dispatch (open-source GSP path) ────────────────
    // Port of tinygrad ops_nv.py NVComputeQueue.exec + _submit_to_gpfifo.
    // The QMD (Queue Method Descriptor) is 64 DWORDs = 256 bytes.
    // All bit positions are from NVC6C0_QMDV03_00 constants in nv_570.py.
    if (isNVIDIA) {
        auto* g = (NVGSPState*)nvGspState;
        if (!g || entry->code_vaddr == 0) return;

        // Build QMD at command buffer slot
        std::vector<uint32_t> qmd(64, 0);
        auto set_bits = [&](int hi, int lo, uint32_t val) {
            int dw_lo = lo/32, dw_hi = hi/32;
            if (dw_lo == dw_hi) {
                uint32_t mask = ((1u<<(hi-lo+1))-1) << (lo%32);
                qmd[dw_lo] = (qmd[dw_lo] & ~mask) | ((val << (lo%32)) & mask);
            }
        };

        uint64_t prog_addr = entry->code_vaddr;

        // qmd_major_version = 3: bits[583:580] = DW18 bits[7:4]
        set_bits(583,580, 3);
        // sm_global_caching_enable: tinygrad sets this; use bit 586 if needed
        // program_address_lower: bits[1567:1536] = DW48
        qmd[48] = (uint32_t)(prog_addr & 0xFFFFFFFF);
        // program_address_upper: bits[1584:1568] = DW49 bits[16:0]
        set_bits(1584,1568, (uint32_t)(prog_addr >> 32));
        // shared_memory_size: bits[561:544] = DW17 bits[17:0]
        set_bits(561, 544, 0x0400); // 1KB default shared mem
        // register_count_v: bits[656:648] = DW20 bits[16:8]
        set_bits(656, 648, 32); // 32 registers default
        // cta_raster_width (global_x): bits[415:384] = DW12
        qmd[12] = (uint32_t)grid.x;
        // cta_raster_height (global_y): bits[447:416] = DW13
        qmd[13] = (uint32_t)grid.y;
        // cta_raster_depth (global_z): bits[479:448] = DW14
        qmd[14] = (uint32_t)grid.z;
        // cta_thread_dimension0 (local_x): bits[607:592] = DW18 bits[31:16]
        set_bits(607, 592, (uint32_t)block.x);
        // cta_thread_dimension1 (local_y): bits[623:608] = DW19 bits[15:0]
        set_bits(623, 608, (uint32_t)block.y);
        // cta_thread_dimension2 (local_z): bits[639:624] = DW19 bits[31:16]
        set_bits(639, 624, (uint32_t)block.z);

        // Build kernarg constbuffer 0:
        // cbuf_0 driver params (0x160 bytes), then user args
        // For BEAGLE's CUDA kernels, args start at offset 0 in cbuf_0.
        // CONSTANT_BUFFER_ADDR_LOWER[0]: bits[1055:1024] = DW32
        // CONSTANT_BUFFER_ADDR_UPPER[0]: bits[1072:1056] = DW33 bits[16:0]
        // CONSTANT_BUFFER_VALID[0]:      bit[640]         = DW20 bit 0
        // CONSTANT_BUFFER_SIZE_SHIFTED4[0]: bits[1023:1008] = DW31 bits[15:0]

        // Write kernargs to command buffer area in VRAM
        size_t ka_sz = (size_t)nPtr * 8 + (size_t)(nTotal - nPtr) * 4;
        size_t cbuf_sz = 0x160 + ka_sz; // driver header + user args
        uint64_t ka_vram = g->cmdq_vram + (uint64_t)g->cmdq_ptr;
        g->cmdq_ptr = (g->cmdq_ptr + (uint32_t)((cbuf_sz + 0x7f) & ~0x7fULL)) & 0x1FFFFC;
        std::vector<uint8_t> ka_buf(cbuf_sz, 0);
        // Fill driver params cbuf_0[6:12]: shared_mem_window, local_mem_window, 0xfffdc0
        // These are GPU virtual addresses for LDS/scratch windows
        // Set to known good values: LDS window = 0x10000000_00000000 etc.
        uint64_t shared_win = 0x10000000'00000000ULL;
        uint64_t local_win  = 0x20000000'00000000ULL;
        uint64_t unk        = 0xfffdc0;
        memcpy(ka_buf.data() + 6*4,  &shared_win, 8);
        memcpy(ka_buf.data() + 8*4,  &local_win,  8);
        memcpy(ka_buf.data() + 10*4, &unk,         8);
        // User args after 0x160
        uint8_t* up = ka_buf.data() + 0x160;
        for (int i=0; i<nPtr; ++i)       { memcpy(up, &ptrs[i],     8); up+=8; }
        for (int i=0; i<nTotal-nPtr; ++i){ memcpy(up, &ints[i],     4); up+=4; }
        nv_vram_wr(tgpuSock, tgpuDevId, ka_vram, ka_buf.data(), ka_buf.size());

        qmd[32] = (uint32_t)(ka_vram & 0xFFFFFFFF);
        set_bits(1072,1056, (uint32_t)(ka_vram >> 32));
        set_bits(640, 640, 1); // constant_buffer_valid[0]
        // constant_buffer_size_shifted4[0]: bits[1023:1008] = DW31 bits[15:0]
        set_bits(1023,1008, (uint32_t)(cbuf_sz >> 4));

        // Write QMD to VRAM (after kernargs)
        uint64_t qmd_vram = ka_vram + cbuf_sz;
        nv_vram_wr(tgpuSock, tgpuDevId, qmd_vram, qmd.data(), 64*4);

        // Build command buffer (tinygrad NVComputeQueue.exec):
        //   1. INVALIDATE_SHADER_CACHES_NO_WFI (subchannel 1)
        //   2. SEND_PCAS_A = qmd_vram >> 8     (subchannel 1)
        //   3. SEND_SIGNALING_PCAS2_B = 9      (subchannel 1)
        auto nvm = [](uint32_t sc, uint32_t mthd, std::vector<uint32_t>& q, std::vector<uint32_t> args) {
            q.push_back((2u<<28) | ((uint32_t)args.size()<<16) | (sc<<13) | (mthd>>2));
            for (auto a : args) q.push_back(a);
        };
        std::vector<uint32_t> cmdbuf;
        nvm(1, MET_INV_SHDR,  cmdbuf, {(1u<<0)|(1u<<4)|(1u<<12)}); // instruction+global+constant
        nvm(1, MET_PCAS_A,    cmdbuf, {(uint32_t)(qmd_vram >> 8)});
        nvm(1, MET_SIGNAL_B,  cmdbuf, {9}); // PCAS_ACTION=9 (tinygrad value)

        // Completion semaphore via SEM_EXECUTE on subchannel 0
        // Use eop_paddr if available, else skip
        if (g->eop_paddr) {
            ++g->eop_signal_val;
            nvm(0, MET_SEM_ADDRLO, cmdbuf, {(uint32_t)(g->eop_paddr & 0xFFFFFFFF)});
            nvm(0, MET_SEM_ADDRHI, cmdbuf, {(uint32_t)(g->eop_paddr >> 32)});
            nvm(0, MET_SEM_PAYLO,  cmdbuf, {(uint32_t)(g->eop_signal_val & 0xFFFFFFFF)});
            nvm(0, MET_SEM_PAYHI,  cmdbuf, {(uint32_t)(g->eop_signal_val >> 32)});
            nvm(0, MET_SEM_EXEC,   cmdbuf, {SEM_OP_RELEASE | SEM_WFI_EN | SEM_PAY64});
            nvm(0, MET_NON_STALL,  cmdbuf, {0});
        }

        // Write command buffer to VRAM (after QMD)
        uint64_t cmdq_va  = qmd_vram + 256;
        uint32_t cmd_dws  = (uint32_t)cmdbuf.size();
        nv_vram_wr(tgpuSock, tgpuDevId, cmdq_va, cmdbuf.data(), cmd_dws * 4);

        // Submit to GPFIFO
        // GPFIFO entry format: (cmdq_addr/4 << 2) | (num_dw << 42) | (1 << 41)
        uint64_t gpfifo_entry = ((cmdq_va / 4) << 2)
                              | ((uint64_t)cmd_dws << 42)
                              | (1ULL << 41);
        uint32_t put = g->gpfifo_put % g->gpfifo_entries;
        g->gpfifo_ring[put] = gpfifo_entry;
        __sync_synchronize();
        *g->userd_gpput = (uint32_t)((put + 1) % g->gpfifo_entries);
        g->gpfifo_put++;
        __sync_synchronize();

        // Ring doorbell: write work_token to GPU MMIO @ BAR0+0xbb0090
        // gpu_mmio = BAR0 offset 0xbb0000, register 0x90/4
        nv_wr32(tgpuSock, tgpuDevId, 0xbb0090, g->work_token);
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
    if (isNVIDIA) {
        auto* g = (NVGSPState*)nvGspState;
        if (g) return (GPUPtr)nv_vram_alloc(g, memSize);
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

void GPUInterface::FreeMemory(GPUPtr /*dPtr*/) {
    // Bump allocator — no individual free for either vendor.
}

size_t GPUInterface::GetAvailableMemory() {
    if (isNVIDIA) {
        auto* g = (NVGSPState*)nvGspState;
        if (g) return (size_t)(g->vram_size > g->vram_top ? g->vram_size - g->vram_top : 0);
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
    // NVIDIA: BAR1 (VRAM); AMD: BAR0 (VRAM)
    int bar = isNVIDIA ? 1 : 0;
    tgpu_mmio_write(tgpuSock, tgpuDevId, bar, (uint64_t)dest, src, sz);
}

void GPUInterface::MemcpyDeviceToHost(void* dest, const GPUPtr src, size_t sz) {
    int bar = isNVIDIA ? 1 : 0;
    tgpu_mmio_read(tgpuSock, tgpuDevId, bar, (uint64_t)src, dest, sz);
}

void GPUInterface::MemcpyDeviceToDevice(GPUPtr dest, GPUPtr src, size_t sz) {
    int bar = isNVIDIA ? 1 : 0;
    void* tmp = malloc(sz);
    tgpu_mmio_read (tgpuSock, tgpuDevId, bar, (uint64_t)src,  tmp, sz);
    tgpu_mmio_write(tgpuSock, tgpuDevId, bar, (uint64_t)dest, tmp, sz);
    free(tmp);
}

void GPUInterface::MemsetShort(GPUPtr dest, unsigned short val, size_t count) {
    int bar = isNVIDIA ? 1 : 0;
    size_t bytes = count * sizeof(unsigned short);
    uint16_t* buf = (uint16_t*)malloc(bytes);
    for (size_t i = 0; i < count; ++i) buf[i] = val;
    tgpu_mmio_write(tgpuSock, tgpuDevId, bar, (uint64_t)dest, buf, bytes);
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
