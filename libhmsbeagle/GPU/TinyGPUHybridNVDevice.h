/*
 * TinyGPUHybridNVDevice.h
 *
 * TODO.md plan step C7, its second part: NVDevice.__init__ (tinygrad/runtime/ops_nv.py:590-640 at a9830e2b4) as it runs
 * after PCIIface's boot, ported statement by statement onto TinyGPUHybridNVRM.h's RM client and TinyGPUHybridNVMemory.h's
 * allocations: PCIIface's root client (ops_nv.py:564-568); the device, subdevice, virtual memory, PERF_BOOST, VA space,
 * channel group, GPFIFO area and context share; two GPFIFO channels (_new_gpu_fifo, :642-666: an error notifier, the
 * channel, its engine object, compute with a debugger or copy, and its work-submit token; then, as the daemon patches it,
 * nv_init_helper's USERD baseline); the channel group's schedule;
 * cmdq_page; _query_gpu_info (:668-676); then HCQCompiled's allocations in tinygrad's order (the allocator's 32 copy
 * buffers, hcq.py:527-529; the signal page, whose last two 16-byte slots become the timeline signals, :442-448;
 * kernargs_buf, :411); and _setup_gpfifos (:681-694): the compute queue's setup and signal, the copy queue's wait, setup and
 * signal, each submitted through NVDevice's own cmdq_page (_submit_to_gpfifo, :114-127), then HCQSignal.wait on the timeline
 * (hcq.py:274-287) with NVSignal's sleep, PCIIface.sleep's status-queue drain (ops_nv.py:579-581). nvd_handoff_tables fills
 * the launch encoding's QMD, method and flag words from TinyGPUNVTables.h, which the daemon's build_handoff otherwise sends.
 */

#ifndef LIBHMSBEAGLE_GPU_TINYGPUHYBRIDNVDEVICE_H
#define LIBHMSBEAGLE_GPU_TINYGPUHYBRIDNVDEVICE_H

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "libhmsbeagle/GPU/TinyGPUHybridNVDispatch.h"
#include "libhmsbeagle/GPU/TinyGPUHybridNVProgram.h"
#include "libhmsbeagle/GPU/TinyGPUHybridNVRM.h"
#include "libhmsbeagle/GPU/TinyGPUNVTables.h"

namespace tinygpu_device {

// BumpAllocator (memory.py:14-21)
struct TGBumpAllocator {
    uint64_t size = 0, ptr = 0, base = 0;
    bool wrap = true;
    uint64_t alloc(uint64_t sz, uint64_t alignment = 1) {
        if (tg_round_up(ptr, alignment) + sz > size) {
            if (!wrap) throw TGPyError("RuntimeError", "Out of memory");
            ptr = 0;
        }
        uint64_t res = tg_round_up(ptr, alignment);
        ptr = res + sz;
        return res + base;
    }
};

// GPFifo (ops_nv.py): ring and gpput are views of the GPFIFO area's BAR1 window, here their BAR1 offsets
struct NVGPFifo {
    uint64_t ring_off = 0, gpput_off = 0;
    uint32_t entries_count = 0, token = 0;
    uint64_t put_value = 0;
};

// What NVDevice.__init__ leaves, as far as BEAGLE uses it.
struct NVDeviceState {
    uint32_t root = 0, gpu_instance = 0;                              // PCIIface
    uint32_t nvdevice = 0, subdevice = 0, virtmem = 0, vaspace = 0, channel_group = 0, usermode = 0;
    uint64_t gpu_mmio_off = 0;                                        // setup_usermode's BAR0 window
    uint32_t debug_compute_obj = 0, debug_channel = 0, debugger = 0;
    NVBuffer gpfifo_area, cmdq_page, signal_page, kernargs_buf;
    std::vector<NVBuffer> notifiers, copy_bufs;
    NVGPFifo compute_gpfifo, dma_gpfifo;
    TGBumpAllocator cmdq_allocator;
    uint32_t num_gpcs = 0, num_tpc_per_gpc = 0, num_sm_per_tpc = 0, max_warps_per_sm = 0, sm_version = 0, sass_version = 0;
    std::string arch;
    uint64_t timeline_signal = 0, shadow_timeline_signal = 0;        // their value_addr
    uint64_t timeline_value = 1;
    uint32_t slm_per_thread = 0;
    uint64_t shared_mem_window = 0, local_mem_window = 0;
    int wait_timeout_ms = 30000;                                      // HCQSignal.wait's, getenv("HCQDEV_WAIT_TIMEOUT_MS", 30000)
};

inline volatile uint64_t* nv_signal_host(NVDeviceState& d, uint64_t va) {   // signal.base_buf.cpu_view(), in the signal page
    return (volatile uint64_t*)(d.signal_page.view + (va - d.signal_page.va_addr));
}

// NVCommandQueue._submit_to_gpfifo (ops_nv.py:114-127) for a queue never bound: its words through the device's cmdq_page
inline void nv_submit_to_gpfifo(NVMemDev& dev, NVDeviceState& d, NVGPFifo& gpfifo, const std::vector<uint32_t>& q) {
    const uint64_t cmdq_addr = d.cmdq_allocator.alloc(q.size() * 4, 16);
    const uint64_t cmdq_wptr = (cmdq_addr - d.cmdq_page.va_addr) / 4;
    memcpy(d.cmdq_page.view + cmdq_wptr * 4, q.data(), q.size() * 4);
    const uint64_t entry = ((cmdq_addr / 4) << 2) | ((uint64_t)q.size() << 42) | (1ull << 41);
    dev.vram_set_q(gpfifo.ring_off + (gpfifo.put_value % gpfifo.entries_count) * 8, entry);
    const uint32_t gpput = (uint32_t)((gpfifo.put_value + 1) % gpfifo.entries_count);
    std::string err;
    if (!dev.t->bulk_write(dev.vram_bar, gpfifo.gpput_off, &gpput, 4, err)) throw TGPyError("RuntimeError", err);
    // System.memory_barrier(): nothing reaches TinyGPU.app
    dev.wreg((uint32_t)(d.gpu_mmio_off + 0x90), gpfifo.token);
    gpfifo.put_value += 1;
}

// HCQSignal.wait (hcq.py:274-287) with NVSignal._sleep (ops_nv.py:28-30): after 200 ms without progress, PCIIface.sleep drains the
// status queue and raises on a device fault
inline void nv_signal_wait(NVRMClient& rm, NVDeviceState& d, uint64_t sig_va, uint64_t value, int timeout = 0) {
    timeout = timeout ? timeout : d.wait_timeout_ms;
    auto cur_value = [&] { return *nv_signal_host(d, sig_va); };
    int64_t start_time = nv_now_ms();
    for (;;) {
        const uint64_t prev_value = cur_value();
        if (!(prev_value < value)) return;
        const int64_t cur_time = nv_now_ms();
        if (!(cur_time - start_time < timeout)) break;
        if (cur_time - start_time > 200) {   // self.owner.iface.sleep(200)
            rm.gsp.stat_q.read_resp([](uint32_t, std::vector<uint8_t>&) { return false; });
            if (rm.gsp.is_err_state) throw TGPyError("RuntimeError", "Device fault detected");
        }
        if (cur_value() != prev_value) start_time = nv_now_ms();   // progress was made, reset timer
    }
    if (cur_value() < value)
        throw TGPyError("RuntimeError", "Wait timeout: " + std::to_string(timeout) + " ms! (the signal is not set to " + std::to_string(value) +
                        ", but " + std::to_string(cur_value()) + ")");
}

// _new_gpu_fifo (ops_nv.py:642-666), not video
inline NVGPFifo nv_new_gpu_fifo(NVRMClient& rm, NVDeviceState& d, const NVBuffer& gpfifo_area, uint32_t ctxshare, uint32_t channel_group,
                                uint64_t offset, uint32_t entries, bool compute) {
    NVMemoryManager& mm = rm.mm;
    NVBuffer notifier = nv_iface_alloc(mm, 48ull << 20, false, true);
    d.notifiers.push_back(notifier);
    nv_gpu::NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS params{};
    params.gpFifoOffset = gpfifo_area.va_addr + offset;
    params.gpFifoEntries = entries;
    params.hContextShare = ctxshare;
    params.hObjectError = (uint32_t)notifier.hMemory;
    params.hObjectBuffer = (uint32_t)gpfifo_area.hMemory;
    params.hUserdMemory[0] = (uint32_t)gpfifo_area.hMemory;
    params.userdOffset[0] = (uint64_t)entries * 8 + offset;
    params.engineType = 0;
    params.hVASpace = 0;   // gsp has no default vaspace, rm maps the decoder ctx into its own
    const uint32_t gpfifo = rm.rpc_rm_alloc(channel_group, rm.gpfifo_class, params, d.root);
    if (compute) {
        d.debug_compute_obj = rm.rpc_rm_alloc_bytes(gpfifo, rm.compute_class, nullptr, d.root);
        d.debug_channel = gpfifo;
        nv_gpu::NV83DE_ALLOC_PARAMETERS debugger_params{};
        debugger_params.hAppClient = d.root;
        debugger_params.hClass3dObject = d.debug_compute_obj;
        d.debugger = rm.rpc_rm_alloc(d.nvdevice, nv_gpu::GT200_DEBUGGER, debugger_params, d.root);
    } else {
        rm.rpc_rm_alloc_bytes(gpfifo, rm.dma_class, nullptr, d.root);
    }
    nv_gpu::NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS ws{};
    ws.workSubmitToken = 0xffffffff;   // -1
    ws = rm.rpc_rm_control(gpfifo, nv_gpu::NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN, ws, d.root);
    // ctxshare != 0: setup_gpfifo_vm, which PCIIface leaves empty
    NVGPFifo g;
    g.ring_off = gpfifo_area.mapping.paddrs[0].first + offset;
    g.entries_count = entries;
    g.token = ws.workSubmitToken;
    g.gpput_off = gpfifo_area.mapping.paddrs[0].first + offset + (uint64_t)entries * 8 + offsetof(nv_gpu::AmpereAControlGPFifo, GPPut);
    return g;
}

// nv_init_helper's _new_gpu_fifo_with_userd_baseline (plan step P1's diagnostics), which wraps _new_gpu_fifo in the daemon: USERD's
// GPGet, then GPPut, before any submission, each one 4-byte read through the GPFIFO area's BAR1 window, logged
inline void nv_userd_baseline(NVMemDev& dev, const NVBuffer& gpfifo_area, uint64_t offset, uint32_t entries, const char* kind) {
    const uint64_t userd = gpfifo_area.mapping.paddrs[0].first + offset + (uint64_t)entries * 8;   // USERD follows the ring
    uint32_t get = 0, put = 0;
    std::string err;
    if (!dev.t->bulk_read(dev.vram_bar, userd + offsetof(nv_gpu::AmpereAControlGPFifo, GPGet), &get, 4, err) ||
        !dev.t->bulk_read(dev.vram_bar, userd + offsetof(nv_gpu::AmpereAControlGPFifo, GPPut), &put, 4, err))
        throw TGPyError("RuntimeError", err);
    tg_log("USERD baseline, %s GPFIFO (gpfifo_area+0x%llx, before any submission): GPGet=0x%x GPPut=0x%x", kind,
           (unsigned long long)offset, get, put);
}

// PCIIface.__init__ after its boot (ops_nv.py:564-568), NVDevice.__init__ (:590-640) and HCQCompiled.__init__ (hcq.py:387-412)
inline void nv_device_init(NVRMClient& rm, NVDeviceState& d) {
    NVMemoryManager& mm = rm.mm;
    NVMemDev& dev = *mm.dev;
    // PCIIface
    d.root = 0xc1000000;
    d.gpu_instance = 0;
    nv_gpu::NV0000_ALLOC_PARAMETERS root_params{};
    rm.rpc_rm_alloc(0, nv_gpu::NV01_ROOT, root_params, d.root);
    // NVDevice
    nv_gpu::NV0080_ALLOC_PARAMETERS device_params{};
    device_params.deviceId = d.gpu_instance;
    device_params.hClientShare = d.root;
    device_params.vaMode = nv_gpu::NV_DEVICE_ALLOCATION_VAMODE_OPTIONAL_MULTIPLE_VASPACES;
    d.nvdevice = rm.rpc_rm_alloc(d.root, nv_gpu::NV01_DEVICE_0, device_params, d.root);
    nv_gpu::NV2080_ALLOC_PARAMETERS subdevice_params{};
    d.subdevice = rm.rpc_rm_alloc(d.nvdevice, nv_gpu::NV20_SUBDEVICE_0, subdevice_params, d.root);
    nv_gpu::NV_MEMORY_VIRTUAL_ALLOCATION_PARAMS virtmem_params{};
    virtmem_params.limit = 0x1ffffffffffff;
    d.virtmem = rm.rpc_rm_alloc(d.nvdevice, nv_gpu::NV01_MEMORY_VIRTUAL, virtmem_params, d.root);
    d.usermode = 0xce000000;   // setup_usermode: map_bar sends nothing
    d.gpu_mmio_off = 0xbb0000;
    nv_gpu::NV2080_CTRL_PERF_BOOST_PARAMS boost{};
    boost.duration = 0xffffffff;
    boost.flags = (nv_gpu::NV2080_CTRL_PERF_BOOST_FLAGS_CUDA_YES << 4) | (nv_gpu::NV2080_CTRL_PERF_BOOST_FLAGS_CUDA_PRIORITY_HIGH << 6) |
                  (nv_gpu::NV2080_CTRL_PERF_BOOST_FLAGS_CMD_BOOST_TO_MAX);
    rm.rpc_rm_control(d.subdevice, nv_gpu::NV2080_CTRL_CMD_PERF_BOOST, boost, d.root);
    nv_gpu::NV_VASPACE_ALLOCATION_PARAMETERS vaspace_params{};
    vaspace_params.vaBase = 0x1000;
    vaspace_params.vaSize = 0x1fffffb000000;
    vaspace_params.flags = nv_gpu::NV_VASPACE_ALLOCATION_FLAGS_ENABLE_PAGE_FAULTING | nv_gpu::NV_VASPACE_ALLOCATION_FLAGS_IS_EXTERNALLY_OWNED;
    d.vaspace = rm.rpc_rm_alloc(d.nvdevice, nv_gpu::FERMI_VASPACE_A, vaspace_params, d.root);
    // setup_vm: PCIIface's is empty
    nv_gpu::NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS channel_params{};
    channel_params.engineType = nv_gpu::NV2080_ENGINE_TYPE_GRAPHICS;
    d.channel_group = rm.rpc_rm_alloc(d.nvdevice, nv_gpu::KEPLER_CHANNEL_GROUP_A, channel_params, d.root);
    d.gpfifo_area = nv_iface_alloc(mm, 0x300000, false, false, true, true, true);   // map_flags: PCIIfaceBase.alloc ignores them
    nv_gpu::NV_CTXSHARE_ALLOCATION_PARAMETERS ctxshare_params{};
    ctxshare_params.hVASpace = d.vaspace;
    ctxshare_params.flags = nv_gpu::NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_ASYNC;
    const uint32_t ctxshare = rm.rpc_rm_alloc(d.channel_group, nv_gpu::FERMI_CONTEXT_SHARE_A, ctxshare_params, d.root);
    d.compute_gpfifo = nv_new_gpu_fifo(rm, d, d.gpfifo_area, ctxshare, d.channel_group, 0, 0x10000, true);
    nv_userd_baseline(dev, d.gpfifo_area, 0, 0x10000, "compute");
    d.dma_gpfifo = nv_new_gpu_fifo(rm, d, d.gpfifo_area, ctxshare, d.channel_group, 0x100000, 0x10000, false);
    nv_userd_baseline(dev, d.gpfifo_area, 0x100000, 0x10000, "copy");
    nv_gpu::NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS sched{};
    sched.bEnable = 1;
    rm.rpc_rm_control(d.channel_group, nv_gpu::NVA06C_CTRL_CMD_GPFIFO_SCHEDULE, sched, d.root);
    d.cmdq_page = nv_iface_alloc(mm, 0x200000, false, false, true);
    d.cmdq_allocator = TGBumpAllocator{d.cmdq_page.size, 0, d.cmdq_page.va_addr, true};
    // _query_gpu_info('num_gpcs', 'num_tpc_per_gpc', 'num_sm_per_tpc', 'max_warps_per_sm', 'sm_version'), is_nvd
    nv_gpu::NV2080_CTRL_INTERNAL_STATIC_GR_GET_INFO_PARAMS gr{};
    gr = rm.rpc_rm_control(d.subdevice, nv_gpu::NV2080_CTRL_CMD_INTERNAL_STATIC_KGR_GET_INFO, gr, d.root);
    d.num_gpcs = gr.engineInfo[0].infoList[nv_gpu::NV2080_CTRL_GR_INFO_INDEX_LITTER_NUM_GPCS].data;
    d.num_tpc_per_gpc = gr.engineInfo[0].infoList[nv_gpu::NV2080_CTRL_GR_INFO_INDEX_LITTER_NUM_TPC_PER_GPC].data;
    d.num_sm_per_tpc = gr.engineInfo[0].infoList[nv_gpu::NV2080_CTRL_GR_INFO_INDEX_LITTER_NUM_SM_PER_TPC].data;
    d.max_warps_per_sm = gr.engineInfo[0].infoList[nv_gpu::NV2080_CTRL_GR_INFO_INDEX_MAX_WARPS_PER_SM].data;
    d.sm_version = gr.engineInfo[0].infoList[nv_gpu::NV2080_CTRL_GR_INFO_INDEX_SM_VERSION].data;
    // FIXME: no idea how to convert this for blackwells
    const uint32_t val = d.sm_version & 0xff;
    d.arch = d.sm_version == 0xa04 ? "sm_120" : "sm_" + std::to_string((d.sm_version >> 8) & 0xff) + std::to_string(val > 0xf ? val >> 4 : val);
    d.sass_version = ((d.sm_version & 0xf00) >> 4) | (d.sm_version & 0xf);
    // NVAllocator(self), built before HCQCompiled.__init__ runs: HCQAllocatorBase's copy buffers (_alloc(2 MiB, host=True) x 32)
    for (int i = 0; i < 32; ++i) d.copy_bufs.push_back(nv_iface_alloc(mm, 2ull << 20, true));
    // HCQCompiled.__init__: the timeline signals from a new signal page (allocator.alloc(0x1000, host, uncached, cpu_access): NVAllocator
    // passes cpu_access, host and zero on, not uncached), popped from its end, each written to 0; then kernargs_buf
    d.signal_page = nv_iface_alloc(mm, 0x1000, true, false, true);
    d.timeline_signal = d.signal_page.va_addr + d.signal_page.size - 16;
    *nv_signal_host(d, d.timeline_signal) = 0;
    d.shadow_timeline_signal = d.signal_page.va_addr + d.signal_page.size - 32;
    *nv_signal_host(d, d.shadow_timeline_signal) = 0;
    d.kernargs_buf = nv_iface_alloc(mm, 16ull << 20, false, false, true);
    // _setup_gpfifos
    d.slm_per_thread = 0;
    d.shared_mem_window = 0x729400000000ull;   // Set windows addresses to not collide with other allocated buffers.
    d.local_mem_window = 0x729300000000ull;
    std::vector<uint32_t> q;
    nvd_nvm(q, 1, nvt::M_SET_OBJECT, {rm.compute_class});
    nvd_nvm(q, 1, nvt::M_SET_SHADER_LOCAL_MEMORY_WINDOW_A, {(uint32_t)(d.local_mem_window >> 32), (uint32_t)d.local_mem_window});
    nvd_nvm(q, 1, nvt::M_SET_SHADER_SHARED_MEMORY_WINDOW_A, {(uint32_t)(d.shared_mem_window >> 32), (uint32_t)d.shared_mem_window});
    uint64_t v = d.timeline_value++;   // next_timeline()
    nvd_nvm(q, 0, nvt::M_SEM_ADDR_LO, {(uint32_t)d.timeline_signal, (uint32_t)(d.timeline_signal >> 32), (uint32_t)v, (uint32_t)(v >> 32),
                                       nvt::F_SEM_RELEASE_64_TIMESTAMP});
    nvd_nvm(q, 0, nvt::M_NON_STALL_INTERRUPT, {0x0});
    nv_submit_to_gpfifo(dev, d, d.compute_gpfifo, q);
    q.clear();
    const uint64_t w = d.timeline_value - 1;
    nvd_nvm(q, 0, nvt::M_SEM_ADDR_LO, {(uint32_t)d.timeline_signal, (uint32_t)(d.timeline_signal >> 32), (uint32_t)w, (uint32_t)(w >> 32),
                                       nvt::F_SEM_ACQUIRE_GEQ_64});
    nvd_nvm(q, 4, nvt::M_SET_OBJECT, {rm.dma_class});
    v = d.timeline_value++;
    nvd_nvm(q, 4, nvt::M_DMA_SET_SEMAPHORE_A, {(uint32_t)(d.timeline_signal >> 32), (uint32_t)d.timeline_signal, (uint32_t)v});
    nvd_nvm(q, 4, nvt::M_DMA_LAUNCH_DMA, {nvt::F_DMA_SEMAPHORE_RELEASE});
    nv_submit_to_gpfifo(dev, d, d.dma_gpfifo, q);
    nv_signal_wait(rm, d, d.timeline_signal, d.timeline_value - 1);   // synchronize(): can_recover is false, so the default timeout
}

// build_handoff's fields (nv_dispatch_daemon.py), from TinyGPUNVTables.h and the device: what the launch encoder needs. The
// buffers (cmdq, kargs, staging, signal) are the caller's.
inline void nvd_handoff_from_device(const NVDeviceState& d, uint32_t compute_class, NVDHandoff& h) {
    using namespace nvt;
    const bool v5 = compute_class >= BLACKWELL_COMPUTE_A;
    const Bits* F = v5 ? kQmdV5 : kQmdV3;
    const Bits (*FI)[8] = v5 ? kQmdV5Indexed : kQmdV3Indexed;
    auto byte = [](Bits b) { return (uint32_t)b.lo / 8; };
    auto range = [](uint32_t out[2], Bits b) { out[0] = b.hi; out[1] = b.lo; };
    h.qmd_ver = v5 ? 5 : 3;
    h.qmd_bytes = v5 ? kQmdV5Bytes : kQmdV3Bytes;
    h.q_grid = byte(F[v5 ? GRID_WIDTH : CTA_RASTER_WIDTH]);
    h.q_block01 = byte(F[CTA_THREAD_DIMENSION0]);
    h.q_block2 = byte(F[CTA_THREAD_DIMENSION2]);
    h.q_rel_addr = byte(F[v5 ? RELEASE_SEMAPHORE0_ADDR_LOWER : RELEASE0_ADDRESS_LOWER]);
    h.q_rel_payload = byte(F[v5 ? RELEASE_SEMAPHORE0_PAYLOAD_LOWER : RELEASE0_PAYLOAD_LOWER]);
    h.q_cb_shift = v5 ? 6 : 0;
    range(h.q_cb_hi, FI[v5 ? CONSTANT_BUFFER_ADDR_UPPER_SHIFTED6 : CONSTANT_BUFFER_ADDR_UPPER][0]);
    range(h.q_cb_lo, FI[v5 ? CONSTANT_BUFFER_ADDR_LOWER_SHIFTED6 : CONSTANT_BUFFER_ADDR_LOWER][0]);
    range(h.q_rel_en, F[RELEASE0_ENABLE]);
    range(h.q_dep_ptr, F[DEPENDENT_QMD0_POINTER]);
    range(h.q_dep_action, F[DEPENDENT_QMD0_ACTION]);
    range(h.q_dep_prefetch, F[DEPENDENT_QMD0_PREFETCH]);
    range(h.q_dep_enable, F[DEPENDENT_QMD0_ENABLE]);
    h.m_sem_addr_lo = M_SEM_ADDR_LO;
    h.f_sem_acquire = F_SEM_ACQUIRE_GEQ_64;
    h.m_invalidate = M_INVALIDATE_SHADER_CACHES_NO_WFI;
    h.f_invalidate = F_INVALIDATE_ALL;
    h.m_pcas_a = M_SEND_PCAS_A;
    h.m_pcas2_b = M_SEND_SIGNALING_PCAS2_B;
    h.m_dma_offset_in_upper = M_DMA_OFFSET_IN_UPPER;
    h.m_dma_line_length_in = M_DMA_LINE_LENGTH_IN;
    h.m_dma_launch = M_DMA_LAUNCH_DMA;
    h.m_dma_sem_a = M_DMA_SET_SEMAPHORE_A;
    h.f_dma_copy = F_DMA_COPY;
    h.f_dma_sem = F_DMA_SEMAPHORE_RELEASE;
    h.m_local_mem_a = M_SET_SHADER_LOCAL_MEMORY_A;
    h.m_local_mem_nt_a = M_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A;
    h.f_sem_release = F_SEM_RELEASE_64_TIMESTAMP;
    h.m_non_stall_interrupt = M_NON_STALL_INTERRUPT;
    auto fifo = [](const NVGPFifo& g, NVDFifo& f) {
        f.ring_bar = 1; f.ring_off = g.ring_off; f.gpput_bar = 1; f.gpput_off = g.gpput_off;
        f.entries = g.entries_count; f.put = g.put_value; f.token = g.token;
    };
    fifo(d.compute_gpfifo, h.compute);
    fifo(d.dma_gpfifo, h.copy);
    h.db_bar = 0;
    h.db_off = d.gpu_mmio_off + 0x90;
}

// the runtime handoff's device values (cmd_handoff, programs false)
inline void nvd_runtime_from_device(const NVDeviceState& d, uint32_t compute_class, NVDRuntime& rt) {
    rt.compute_class = compute_class;
    rt.sass_version = d.sass_version;
    rt.shared_mem_window = d.shared_mem_window;
    rt.local_mem_window = d.local_mem_window;
    rt.num_gpcs = d.num_gpcs;
    rt.num_tpc_per_gpc = d.num_tpc_per_gpc;
    rt.num_sm_per_tpc = d.num_sm_per_tpc;
    rt.max_warps_per_sm = d.max_warps_per_sm;
}

} // namespace tinygpu_device

#endif // LIBHMSBEAGLE_GPU_TINYGPUHYBRIDNVDEVICE_H
