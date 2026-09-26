/*
 * TinyGPUHybridNVRM.h
 *
 * TODO.md plan step C7: NV_GSP's RM client (tinygrad/runtime/support/nv/ip.py:457-466, 538-599 at a9830e2b4), ported
 * statement by statement on C5's GSP queues (TinyGPUHybridNVGsp.h) and C6's memory manager (TinyGPUHybridNVMemory.h):
 * rpc_rm_alloc with its hooks (a GPFIFO channel's RAMFC, instance memory and method buffer, and a user client's error
 * notifier and USERD; the channel's runlist; a user VA space's page directory; the user device and subdevice; a compute
 * object's context, promoted physically, then virtually), rpc_rm_control with the work-submit-token fix-up,
 * rpc_set_page_directory, and promote_ctx, which allocates a context buffer for every entry even when it reuses one it was
 * given, as tinygrad's dict.get does (its default argument is evaluated first). Parameters travel as bytes, as bytes(params)
 * sends them, in C2's generated layouts (TinyGPUNVRMTables.h); the typed wrappers are for the NVDevice port. Not ported:
 * rpc_alloc_memory and rpc_rm_control's PMA branches (profiling, which the plugin never enables) and the video decoder's
 * hook (NVDevice allocates no decoder); asking for either raises. Plan step C8 added NV_GSP.init_hw (ip.py:510-520, with
 * nv_init_helper's patch 3, the 20 s sleep after SEC2's start) and init_golden_image (:468-508): nv_gsp_init_hw.
 */

#ifndef LIBHMSBEAGLE_GPU_TINYGPUHYBRIDNVRM_H
#define LIBHMSBEAGLE_GPU_TINYGPUHYBRIDNVRM_H

#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "libhmsbeagle/GPU/TinyGPUHybridNVGsp.h"
#include "libhmsbeagle/GPU/TinyGPUHybridNVMemory.h"
#include "libhmsbeagle/GPU/TinyGPUNVRMTables.h"

namespace tinygpu_device {

struct NVGRBufDesc { uint64_t size; bool phys, virt, local; };   // ip.py's GRBufDesc (size, virt, phys, local=False), by name

template <class T> std::vector<uint8_t> nv_bytes(const T& s) {   // bytes(s) of a ctypes struct
    std::vector<uint8_t> b(sizeof(T));
    memcpy(b.data(), &s, sizeof(T));
    return b;
}

// T.from_buffer_copy(b): ctypes refuses a buffer shorter than the struct
template <class T> T nv_from_buffer_copy(const std::vector<uint8_t>& b, size_t off = 0) {
    const size_t n = b.size() > off ? b.size() - off : 0;
    if (n < sizeof(T))
        throw TGPyError("ValueError", "Buffer size too small (" + std::to_string(n) + " instead of at least " + std::to_string(sizeof(T)) + " bytes)");
    T s;
    memcpy(&s, b.data() + off, sizeof(T));
    return s;
}

// NV_GSP's RM state after the boot (init_sw, init_hw, init_golden_image) and its RM RPCs.
class NVRMClient {
public:
    NVRMClient(NVGsp& gsp_, NVMemoryManager& mm_) : gsp(gsp_), mm(mm_) {}

    NVGsp& gsp;
    NVMemoryManager& mm;
    uint32_t priv_root = 0xc1e00004;            // init_hw
    uint32_t next_handle = 0xcf000000;          // handle_gen = itertools.count(0xcf000000): what next() returns
    uint32_t gpfifo_class = 0, compute_class = 0, dma_class = 0, viddec_class = 0;   // viddec 0: None
    bool gb2 = false;                           // nvdev.chip_name.startswith("GB2")
    std::map<uint64_t, uint32_t> runlists;      // init_golden_image: each engine's runlist
    std::map<uint32_t, uint32_t> chan_runlists;
    std::vector<std::pair<uint16_t, NVGRBufDesc>> grctx_bufs;   // init_golden_image's, in its dict's order
    uint32_t device = 0, subdevice = 0;

    // rpc_rm_alloc (ip.py:538-569) on bytes: params, the parameter bytes or nullptr for None; the GPFIFO hook fills them in.
    uint32_t rpc_rm_alloc_bytes(uint32_t hParent, uint32_t hClass, std::vector<uint8_t>* params, std::optional<uint32_t> client = std::nullopt) {
        using nv_gpu::NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS;
        if (hClass == gpfifo_class) {
            NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS p = nv_from_buffer_copy<NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS>(params ? *params : std::vector<uint8_t>{});
            TGVirtMapping ramfc_alloc = mm.valloc(0x1000, 0x1000, false, true);
            p.ramfcMem = {ramfc_alloc.paddrs[0].first, 0x200, 2, 0};
            p.instanceMem = {ramfc_alloc.paddrs[0].first, 0x1000, 2, 0};
            uint64_t method_paddr = mm.palloc(tg_round_up(0x5000, 0x1000));   // nvdev._alloc_boot_mem(0x5000, sysmem=False)
            p.mthdbufMem = {method_paddr, 0x5000, 2, 0};
            if (client && *client != priv_root && p.hObjectError != 0) {
                p.errorNotifierMem = {0, 0xecc, 0, 0};
                p.userdMem = {p.hUserdMemory[0] + p.userdOffset[0], 0x400, 2, 0};
            }
            *params = nv_bytes(p);
        }
        const uint32_t cl = client && *client ? *client : priv_root;   // client:=client or self.priv_root
        nv::rpc_gsp_rm_alloc_v alloc_args{};
        alloc_args.hClient = cl;
        alloc_args.hParent = hParent;
        const uint32_t obj = alloc_args.hObject = next_handle++;
        alloc_args.hClass = hClass;
        alloc_args.flags = 0x0;
        alloc_args.paramsSize = params ? (uint32_t)params->size() : 0x0;
        std::vector<uint8_t> msg = nv_bytes(alloc_args);
        if (params) msg.insert(msg.end(), params->begin(), params->end());
        gsp.cmd_q.send_rpc(nv::NV_VGPU_MSG_FUNCTION_GSP_RM_ALLOC, msg);
        gsp.stat_q.wait_resp(nv::NV_VGPU_MSG_FUNCTION_GSP_RM_ALLOC, gsp.rpc_timeout_ms);

        if (hClass == gpfifo_class) {
            const uint32_t e = nv_from_buffer_copy<NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS>(*params).engineType;
            auto it = runlists.find((uint64_t)e + 10 * (e >= nv_gpu::NV2080_ENGINE_TYPE_NVDEC0));
            chan_runlists[obj] = it == runlists.end() ? 0 : it->second;
        }
        if (hClass == nv_gpu::FERMI_VASPACE_A && cl != priv_root) rpc_set_page_directory(hParent, obj, mm.root_page_table.paddr, cl);
        if (hClass == nv_gpu::NV01_DEVICE_0 && cl != priv_root) device = obj;   // save user device handle
        if (hClass == nv_gpu::NV20_SUBDEVICE_0) subdevice = obj;                // save subdevice handle
        if (viddec_class && hClass == viddec_class && cl != priv_root)
            throw TGPyError("NotImplementedError", "the video decoder's context promotion is not ported");
        if (hClass == compute_class && cl != priv_root) {
            std::vector<std::pair<uint16_t, NVGRBufDesc>> sel;
            for (auto& kv : grctx_bufs)
                if (kv.first == 0 || kv.first == 1 || kv.first == 2) sel.push_back(kv);
            std::map<uint16_t, TGVirtMapping> phys_gr_ctx = promote_ctx(cl, subdevice, hParent, sel, nullptr, false, std::nullopt);
            promote_ctx(cl, subdevice, hParent, sel, &phys_gr_ctx, std::nullopt, false);
        }
        return hClass != nv_gpu::NV1_ROOT ? obj : cl;
    }
    template <class P> uint32_t rpc_rm_alloc(uint32_t hParent, uint32_t hClass, P& params, std::optional<uint32_t> client = std::nullopt) {
        std::vector<uint8_t> b = nv_bytes(params);
        uint32_t h = rpc_rm_alloc_bytes(hParent, hClass, &b, client);
        params = nv_from_buffer_copy<P>(b);   // the GPFIFO hook's fields, as tinygrad's params object keeps them
        return h;
    }

    // rpc_rm_control (ip.py:571-591) on bytes, without its PMA branches: the reply's parameter bytes (params' size), or none for None.
    std::optional<std::vector<uint8_t>> rpc_rm_control_bytes(uint32_t hObject, uint32_t cmd, const std::vector<uint8_t>* params,
                                                             std::optional<uint32_t> client = std::nullopt) {
        if (cmd == nv_gpu::NVB0CC_CTRL_CMD_POWER_REQUEST_FEATURES || cmd == nv_gpu::NVB0CC_CTRL_CMD_ALLOC_PMA_STREAM)
            throw TGPyError("NotImplementedError", "rm_control's PMA branches are not ported");
        const uint32_t cl = client && *client ? *client : priv_root;
        nv::rpc_gsp_rm_control_v control_args{};
        control_args.hClient = cl;
        control_args.hObject = hObject;
        control_args.cmd = cmd;
        control_args.flags = 0x0;
        control_args.paramsSize = params ? (uint32_t)params->size() : 0x0;
        std::vector<uint8_t> msg = nv_bytes(control_args);
        if (params) msg.insert(msg.end(), params->begin(), params->end());
        gsp.cmd_q.send_rpc(nv::NV_VGPU_MSG_FUNCTION_GSP_RM_CONTROL, msg);
        std::vector<uint8_t> res = gsp.stat_q.wait_resp(nv::NV_VGPU_MSG_FUNCTION_GSP_RM_CONTROL, gsp.rpc_timeout_ms);
        if (!params) return std::nullopt;
        const size_t off = sizeof(control_args);
        const size_t n = res.size() > off ? res.size() - off : 0;
        if (n < params->size())
            throw TGPyError("ValueError", "Buffer size too small (" + std::to_string(n) + " instead of at least " + std::to_string(params->size()) + " bytes)");
        std::vector<uint8_t> st(res.begin() + off, res.begin() + off + params->size());
        // NOTE: gsp only fills in the channel id, the runlist id (and, on gb20x, the doorbell enable bit) are added by the driver.
        if (cmd == nv_gpu::NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN) {
            auto p = nv_from_buffer_copy<nv_gpu::NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS>(st);
            auto it = chan_runlists.find(hObject);
            if (it == chan_runlists.end()) throw TGPyError("KeyError", std::to_string(hObject));
            p.workSubmitToken |= (it->second << 16) | (gb2 ? (1u << 30) : 0);
            st = nv_bytes(p);
        }
        return st;
    }
    template <class P> P rpc_rm_control(uint32_t hObject, uint32_t cmd, const P& params, std::optional<uint32_t> client = std::nullopt) {
        std::vector<uint8_t> b = nv_bytes(params);
        return nv_from_buffer_copy<P>(*rpc_rm_control_bytes(hObject, cmd, &b, client));
    }

    // rpc_set_page_directory (ip.py:593-599)
    void rpc_set_page_directory(uint32_t dev, uint32_t hVASpace, uint64_t pdir_paddr, std::optional<uint32_t> client = std::nullopt,
                                uint32_t pasid = 0xffffffff) {
        nv::NV0080_CTRL_DMA_SET_PAGE_DIRECTORY_PARAMS_v1E_05 params{};   // flags field is all channels.
        params.physAddress = pdir_paddr;
        params.numEntries = (uint32_t)mm.pte_cnt[0];
        params.flags = 0x8;
        params.hVASpace = hVASpace;
        params.pasid = pasid;
        params.subDeviceId = 1;
        params.chId = 0;
        nv::rpc_set_page_directory_v alloc_args{};
        alloc_args.hClient = client && *client ? *client : priv_root;
        alloc_args.hDevice = dev;
        alloc_args.pasid = pasid;
        alloc_args.params = params;
        gsp.cmd_q.send_rpc(nv::NV_VGPU_MSG_FUNCTION_SET_PAGE_DIRECTORY, nv_bytes(alloc_args));
        gsp.stat_q.wait_resp(nv::NV_VGPU_MSG_FUNCTION_SET_PAGE_DIRECTORY, gsp.rpc_timeout_ms);
    }

    // promote_ctx (ip.py:457-466). virt and phys: None (each buffer's own) or the value for all.
    std::map<uint16_t, TGVirtMapping> promote_ctx(uint32_t client, uint32_t subdev, uint32_t obj,
                                                  const std::vector<std::pair<uint16_t, NVGRBufDesc>>& ctxbufs,
                                                  const std::map<uint16_t, TGVirtMapping>* bufs = nullptr,
                                                  std::optional<bool> virt = std::nullopt, std::optional<bool> phys = std::nullopt,
                                                  uint32_t engine = 0x1) {
        std::map<uint16_t, TGVirtMapping> res;
        nv_gpu::NV2080_CTRL_GPU_PROMOTE_CTX_PARAMS prom{};
        prom.entryCount = (uint32_t)ctxbufs.size();
        prom.engineType = engine;
        prom.hChanClient = client;
        prom.hObject = obj;
        for (size_t i = 0; i < ctxbufs.size(); ++i) {
            const uint16_t buf = ctxbufs[i].first;
            const NVGRBufDesc& desc = ctxbufs[i].second;
            const bool use_v = virt ? *virt : desc.virt, use_p = phys ? *phys : desc.phys;
            TGVirtMapping fresh = mm.valloc(desc.size, 0x1000, false, true);   // allocate buffers (dict.get's default: always)
            const TGVirtMapping& x = bufs && bufs->count(buf) ? bufs->at(buf) : fresh;
            nv_gpu::NV2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ENTRY& e = prom.promoteEntry[i];
            e.bufferId = buf;
            e.gpuVirtAddr = use_v ? x.va_addr : 0;
            e.bInitialize = use_p;
            e.gpuPhysAddr = use_p ? x.paddrs[0].first : 0;
            e.size = use_p ? desc.size : 0;
            e.physAttr = use_p ? 0x4 : 0;
            e.bNonmapped = use_p && !use_v;
            res[buf] = x;
        }
        rpc_rm_control(subdev, nv_gpu::NV2080_CTRL_CMD_GPU_PROMOTE_CTX, prom, client);
        return res;
    }

    // init_golden_image (ip.py:468-508): the private root client's device, subdevice and VA space; the runlists from the
    // device-info table; 512 MiB of VA whose page tables GSP-RM copies; the golden channel, its context buffers (sizes from
    // KGR_GET_CONTEXT_BUFFERS_INFO) promoted, and its compute and copy objects
    void init_golden_image() {
        using namespace nv_gpu;
        NV0000_ALLOC_PARAMETERS root_params{};
        rpc_rm_alloc(0x0, 0x0, root_params);
        NV0080_ALLOC_PARAMETERS dev_params{};
        dev_params.hClientShare = priv_root;
        const uint32_t dev = rpc_rm_alloc(priv_root, NV01_DEVICE_0, dev_params);
        NV2080_ALLOC_PARAMETERS subdev_params{};
        const uint32_t subdev = rpc_rm_alloc(dev, NV20_SUBDEVICE_0, subdev_params);
        NV_VASPACE_ALLOCATION_PARAMETERS vaspace_params{};
        const uint32_t vaspace = rpc_rm_alloc(dev, FERMI_VASPACE_A, vaspace_params);

        NV2080_CTRL_FIFO_GET_DEVICE_INFO_TABLE_PARAMS di_params{};
        const NV2080_CTRL_FIFO_GET_DEVICE_INFO_TABLE_PARAMS di = rpc_rm_control(subdev, NV2080_CTRL_CMD_FIFO_GET_DEVICE_INFO_TABLE, di_params);
        std::map<uint64_t, uint32_t> rl;
        for (uint32_t i = 0; i < di.numEntries; ++i) {
            if (i >= 32) throw TGPyError("IndexError", "invalid index");   // di.entries is a ctypes array of 32
            rl[di.entries[i].engineData[2]] = di.entries[i].engineData[3];
        }
        runlists = rl;

        // reserve 512MB for the reserved PDES
        const uint64_t res_sz = 512ull << 20;
        const uint64_t res_va = mm.alloc_vaddr(res_sz);
        struct_NV90F1_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES_PARAMS bufs_p{};
        bufs_p.pageSize = res_sz;
        bufs_p.numLevelsToCopy = 3;
        bufs_p.virtAddrLo = res_va;
        bufs_p.virtAddrHi = res_va + res_sz - 1;
        const std::vector<NVPageTableEntry> pts = mm.page_tables(res_va, res_sz);
        for (size_t i = 0; i < pts.size(); ++i) {
            if (i >= 6) throw TGPyError("IndexError", "invalid index");   // bufs_p.levels is a ctypes array of 6
            struct_NV90F1_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES_PARAMS_level& l = bufs_p.levels[i];
            l.physAddress = pts[i].paddr;
            l.size = i == 0 ? mm.pte_cnt[0] * 8 : 0x1000;
            l.pageShift = (uint8_t)(tg_bit_length(mm.pte_covers[i]) - 1);
            l.aperture = 1;
        }
        rpc_rm_control(vaspace, NV90F1_CTRL_CMD_VASPACE_COPY_SERVER_RESERVED_PDES, bufs_p);

        TGVirtMapping gpfifo_area = mm.valloc(4 << 10, 0x1000, false, true);
        NV_MEMORY_DESC_PARAMS userd{gpfifo_area.paddrs[0].first + 0x20 * 8, 0x20, 2, 0};
        NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS gg_params{};
        gg_params.gpFifoOffset = gpfifo_area.va_addr;
        gg_params.gpFifoEntries = 32;
        gg_params.engineType = 0x1;
        gg_params.cid = 3;
        gg_params.hVASpace = vaspace;
        gg_params.userdOffset[0] = 0x20 * 8;
        gg_params.userdMem = userd;
        gg_params.internalFlags = 0x1a;
        gg_params.flags = 0x200320;
        const uint32_t ch_gpfifo = rpc_rm_alloc(dev, gpfifo_class, gg_params);

        NV2080_CTRL_INTERNAL_STATIC_KGR_GET_CONTEXT_BUFFERS_INFO_PARAMS ci_params{};
        const NV2080_CTRL_INTERNAL_STATIC_GR_CONTEXT_BUFFERS_INFO gr_ctx_bufs_info =
            rpc_rm_control(subdev, NV2080_CTRL_CMD_INTERNAL_STATIC_KGR_GET_CONTEXT_BUFFERS_INFO, ci_params).engineContextBuffersInfo[0];
        auto ctx_info = [&](uint32_t idx, uint64_t add = 0, uint64_t align = 0) {   // align 0: None
            const uint64_t a = align ? align : gr_ctx_bufs_info.engine[idx].alignment;
            if (!a) throw TGPyError("ZeroDivisionError", "integer division or modulo by zero");
            return tg_round_up(gr_ctx_bufs_info.engine[idx].size + add, a);
        };

        // Setup graphics context
        const uint64_t gr_size = ctx_info(NV0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS, 0x40000);
        const uint64_t patch_size = ctx_info(NV0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS_PATCH);
        std::map<uint32_t, uint64_t> cfgs_sizes;   // indices 3-10 are mapped to 17-24
        for (uint32_t x = 3; x < 11; ++x) cfgs_sizes[x] = ctx_info(x + 14, 0, x == 5 ? (2 << 20) : 0);
        grctx_bufs = {{0, {gr_size, true, true, false}}, {1, {patch_size, true, true, true}}, {2, {patch_size, true, true, false}}};
        for (uint16_t x = 3; x < 7; ++x) grctx_bufs.push_back({x, {cfgs_sizes[x], false, true, false}});
        grctx_bufs.push_back({9, {cfgs_sizes[9], true, true, false}});
        grctx_bufs.push_back({10, {cfgs_sizes[10], true, false, false}});
        grctx_bufs.push_back({11, {cfgs_sizes[10], true, true, false}});   // NOTE: 11 reuses cfgs_sizes[10]
        std::vector<std::pair<uint16_t, NVGRBufDesc>> not_local;
        for (auto& kv : grctx_bufs)
            if (!kv.second.local) not_local.push_back(kv);
        promote_ctx(priv_root, subdev, ch_gpfifo, not_local);

        rpc_rm_alloc_bytes(ch_gpfifo, compute_class, nullptr);
        rpc_rm_alloc_bytes(ch_gpfifo, dma_class, nullptr);
    }
};

// NV_GSP.init_hw (ip.py:510-520) as the daemon runs it, with nv_init_helper's patch 3 (_patched_gsp_init_hw): while it runs,
// SEC2's start (the CPU sequencer's op 8, which GSP-RM posts before GSP_INIT_DONE) sleeps 20 s. Its first two statements, the
// status queue and the command queue's read pointer, are NVGsp's constructor. on_init_done runs once GSP_INIT_DONE was read.
// fmc_boot: the COT boot's second BAR1 block register (Blackwell).
inline void nv_gsp_init_hw(NVRMClient& rm, bool fmc_boot = false, const std::function<void()>& on_init_done = {}) {
    struct InGspInit {   // _in_gsp_init, reset in its finally
        NVFalcon& flcn;
        explicit InGspInit(NVFalcon& f) : flcn(f) { flcn.sleep_after_sec2_start = true; }
        ~InGspInit() { flcn.sleep_after_sec2_start = false; }
    } in_gsp_init(rm.gsp.flcn);
    rm.gsp.stat_q.wait_resp(nv::NV_VGPU_MSG_EVENT_GSP_INIT_DONE, rm.gsp.rpc_timeout_ms);
    if (on_init_done) on_init_done();

    rm.mm.dev->reg(nv_regs::NV_PBUS_BAR1_BLOCK).write({{"mode", 0}, {"target", 0}, {"ptr", 0}});
    if (fmc_boot) rm.mm.dev->reg(nv_regs::NV_VIRTUAL_FUNCTION_PRIV_FUNC_BAR1_BLOCK_LOW_ADDR).write({{"mode", 0}, {"target", 0}, {"ptr", 0}});

    rm.priv_root = 0xc1e00004;
    rm.init_golden_image();
}

} // namespace tinygpu_device

#endif // LIBHMSBEAGLE_GPU_TINYGPUHYBRIDNVRM_H
