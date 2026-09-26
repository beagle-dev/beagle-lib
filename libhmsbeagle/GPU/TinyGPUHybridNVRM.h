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
 * hook (NVDevice allocates no decoder); asking for either raises.
 */

#ifndef LIBHMSBEAGLE_GPU_TINYGPUHYBRIDNVRM_H
#define LIBHMSBEAGLE_GPU_TINYGPUHYBRIDNVRM_H

#include <cstdint>
#include <cstring>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "libhmsbeagle/GPU/TinyGPUHybridNVGsp.h"
#include "libhmsbeagle/GPU/TinyGPUHybridNVMemory.h"
#include "libhmsbeagle/GPU/TinyGPUNVRMTables.h"

namespace tinygpu_device {

struct NVGRBufDesc { uint64_t size; bool phys, virt, local; };   // ip.py's GRBufDesc(size, phys, virt, local=False)

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
};

} // namespace tinygpu_device

#endif // LIBHMSBEAGLE_GPU_TINYGPUHYBRIDNVRM_H
