"""Golden test for TinyGPUHybridNVRM.h (TODO.md plan step C7, its first part) against the code it ports: tinygrad's
NV_GSP.rpc_rm_alloc (with its hooks), rpc_rm_control (with the work-submit-token fix-up), rpc_set_page_directory and
promote_ctx, over C5's GSP queues and C6's memory manager. The RM calls are NVDevice.__init__'s (ops_nv.py:590-640, with
_new_gpu_fifo twice and _query_gpu_info), in its order, with its allocations in between, from a fork point after a boot-like
sequence (golden_mm.py's), on one fake TinyGPU.app whose GSP (golden_gsp.py's, in a shared queue file) answers every RPC
and fills in the controls tinygrad reads. tinygrad runs them first, recording each call; golden_rm.cpp then makes the same
calls. Both must print the same results (handles, the hooks' parameters, the replies, the RM and allocator state after), send
TinyGPU.app the same requests byte for byte, and leave the same queue memory (so the same RPC stream) and VRAM. MMU v2 and
v3; then perturbed copies of the port must fail. No GPU and no TinyGPU.app.
    python golden_rm.py"""
import os, sys, json, ctypes, struct, tempfile, itertools, subprocess, hashlib, types
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tgpaths
tgpaths.setup()
import golden_mm as gm     # the fake TinyGPU.app, tinygrad's memory manager on it, the boot-like fork point
import golden_gsp as gg    # the GSP in the queue file
import nv_dispatch_daemon as d
from tinygrad.runtime.support.nv.ip import GRBufDesc
from tinygrad.runtime.autogen import nv, nv_570 as nv_gpu

HERE, WORK = str(tgpaths.HERE), str(tgpaths.WORK)
ROOT = 0xc1000000   # PCIIface.root
GR_INFO = {nv_gpu.NV2080_CTRL_GR_INFO_INDEX_LITTER_NUM_GPCS: 3, nv_gpu.NV2080_CTRL_GR_INFO_INDEX_LITTER_NUM_TPC_PER_GPC: 4,
           nv_gpu.NV2080_CTRL_GR_INFO_INDEX_LITTER_NUM_SM_PER_TPC: 2, nv_gpu.NV2080_CTRL_GR_INFO_INDEX_MAX_WARPS_PER_SM: 48,
           nv_gpu.NV2080_CTRL_GR_INFO_INDEX_SM_VERSION: 0x809}   # fake_nv_device.py's
# what init_hw and init_golden_image leave, as the daemon will export it (plausible values: an Ada boot's shapes)
ADA = dict(seq=33, priv_root=0xc1e00004, next_handle=0xcf00000a, gpfifo_class=nv_gpu.AMPERE_CHANNEL_GPFIFO_A,
             compute_class=nv_gpu.ADA_COMPUTE_A, dma_class=nv_gpu.AMPERE_DMA_COPY_B, viddec_class=nv_gpu.NVC9B0_VIDEO_DECODER, gb2=0,
             runlists={0: 3, 1: 0, 14: 1, 15: 2, 19: 4},   # engine 0 (the channels') on runlist 3: the token fix-up shows
             grctx=[(0, 0x2a0000, 1, 1, 0), (1, 0x40000, 1, 1, 1), (2, 0x40000, 1, 1, 0), (3, 0x100000, 0, 1, 0), (4, 0x20000, 0, 1, 0),
                    (5, 0x200000, 0, 1, 0), (6, 0x10000, 0, 1, 0), (9, 0x40000, 1, 1, 0), (10, 0x80000, 1, 0, 0), (11, 0x80000, 1, 1, 0)])
STATE = dict(ADA)   # the case's

class FakeRM(gg.FakeGsp):
    """golden_gsp.py's GSP, answering with what the RM controls tinygrad reads carry: a work-submit token per channel (the
    channel id; the driver adds the runlist), and the GR info _query_gpu_info reads."""
    def __init__(self, mm, tg):
        super().__init__(mm, tg)
        self.tokens = 0
    def on_write(self, a, v):
        if a != gg.QUEUE_HEAD: return
        for fn, msg, elem, ok in self.cmdq.new():
            struct.pack_into("<I", self.mm, gg.STATQ + 32, self.cmdq.rp)
            self.rpcs.append((fn, ok))
            self.post(fn, self.reply(fn, msg))
    def reply(self, fn, msg):
        if fn != nv.NV_VGPU_MSG_FUNCTION_GSP_RM_CONTROL: return msg
        n = ctypes.sizeof(nv.rpc_gsp_rm_control_v)
        c = nv.rpc_gsp_rm_control_v.from_buffer_copy(msg[:n])
        params = bytearray(msg[n:])
        if c.cmd == nv_gpu.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN:
            struct.pack_into("<I", params, 0, self.tokens); self.tokens += 1
        elif c.cmd == nv_gpu.NV2080_CTRL_CMD_INTERNAL_STATIC_KGR_GET_INFO:
            p = nv_gpu.NV2080_CTRL_INTERNAL_STATIC_GR_GET_INFO_PARAMS.from_buffer_copy(params)
            for idx, val in GR_INFO.items(): p.engineInfo[0].infoList[idx].data = val
            params = bytearray(bytes(p))
        return bytes(msg[:n]) + bytes(params)

def py_rm(dev, qpath):
    gsp = gg.tinygrad_gsp(dev, qpath, STATE["seq"])
    gsp.priv_root, gsp.handle_gen = STATE["priv_root"], itertools.count(STATE["next_handle"])
    gsp.gpfifo_class, gsp.compute_class, gsp.dma_class, gsp.viddec_class = (STATE[k] for k in ("gpfifo_class", "compute_class", "dma_class", "viddec_class"))
    gsp.runlists, gsp.chan_runlists = dict(STATE["runlists"]), {}
    gsp.grctx_bufs = {i: GRBufDesc(size, phys=bool(p), virt=bool(v), local=bool(l)) for i, size, p, v, l in STATE["grctx"]}
    dev.chip_name, dev.is_err_state = ("GB205" if STATE["gb2"] else "AD107"), False   # as NVDev.__init__ sets them
    return gsp

class PyOps:
    """tinygrad's side: each call made on tinygrad's objects, recorded as a golden_rm.cpp operation, its result printed."""
    def __init__(self, gsp, ifa): self.gsp, self.ifa, self.ops, self.out = gsp, ifa, [], []
    def _try(self, line, f):
        self.ops.append(line)
        try: r = f()
        except (RuntimeError, AssertionError, ValueError, KeyError, MemoryError) as e:
            self.out.append(f"error {type(e).__name__}: {e}"); return None
        return r
    def rm_alloc(self, parent, clss, params=None, client=None):
        line = f"rm_alloc {parent} {clss} {client or 0} {bytes(params).hex() if params is not None else '-'}"
        h = self._try(line, lambda: self.gsp.rpc_rm_alloc(parent, clss, params, client))
        if h is not None: self.out.append(f"handle {h} params {bytes(params).hex() if params is not None else '-'}")
        return h
    def rm_control(self, obj, cmd, params=None, client=None):
        line = f"rm_control {obj} {cmd} {client or 0} {bytes(params).hex() if params is not None else '-'}"
        st = self._try(line, lambda: self.gsp.rpc_rm_control(obj, cmd, params, client))
        if st is not None or not self.out or not self.out[-1].startswith("error"): self.out.append(f"reply {bytes(st).hex() if st is not None else '-'}")
        return st
    def alloc(self, size, host=False, uncached=False, cpu_access=False, contiguous=False, force_devmem=False, zero=False):
        line = f"alloc {size} {int(host)} {int(uncached)} {int(cpu_access)} {int(contiguous)} {int(force_devmem)} {int(zero)}"
        b = self._try(line, lambda: self.ifa.alloc(size, host=host, uncached=uncached, cpu_access=cpu_access, contiguous=contiguous,
                                                   force_devmem=force_devmem, zero=zero))
        if b is not None: self.out.append(f"{b.va_addr} {b.size} {b.meta.hMemory} {gm.fmt_map(b.meta.mapping)}")
        return b
    def finish(self, mm):
        g = self.gsp
        self.out.append(f"rm next_handle {int(repr(g.handle_gen)[6:-1])} device {getattr(g, 'device', 0)} subdevice {getattr(g, 'subdevice', 0)} "
                        f"seq {g.cmd_q.seq}")
        self.out.append("rm chan_runlists" + "".join(f" {k}:{v}" for k, v in sorted(g.chan_runlists.items())))
        self.out.append("state pa " + " ".join(map(str, d._tlsf_save(mm.pa_allocator))))
        self.out.append("state va " + " ".join(map(str, d._tlsf_save(mm.va_allocator))))
        return self.out

def nvdevice_rm(o, gsp):
    """NVDevice.__init__'s RM calls and allocations (ops_nv.py:590-640, _new_gpu_fifo :642-666, _query_gpu_info :668-676) as
    PCIIface makes them after the boot (ops_nv.py:564-568), the root included."""
    o.rm_alloc(0, nv_gpu.NV01_ROOT, nv_gpu.NV0000_ALLOC_PARAMETERS(), ROOT)
    device = o.rm_alloc(ROOT, nv_gpu.NV01_DEVICE_0, nv_gpu.NV0080_ALLOC_PARAMETERS(deviceId=0, hClientShare=ROOT,
                        vaMode=nv_gpu.NV_DEVICE_ALLOCATION_VAMODE_OPTIONAL_MULTIPLE_VASPACES), ROOT)
    subdevice = o.rm_alloc(device, nv_gpu.NV20_SUBDEVICE_0, nv_gpu.NV2080_ALLOC_PARAMETERS(), ROOT)
    o.rm_alloc(device, nv_gpu.NV01_MEMORY_VIRTUAL, nv_gpu.NV_MEMORY_VIRTUAL_ALLOCATION_PARAMS(limit=0x1ffffffffffff), ROOT)
    o.rm_control(subdevice, nv_gpu.NV2080_CTRL_CMD_PERF_BOOST, nv_gpu.NV2080_CTRL_PERF_BOOST_PARAMS(duration=0xffffffff,
      flags=((nv_gpu.NV2080_CTRL_PERF_BOOST_FLAGS_CUDA_YES << 4) | (nv_gpu.NV2080_CTRL_PERF_BOOST_FLAGS_CUDA_PRIORITY_HIGH << 6) |
             (nv_gpu.NV2080_CTRL_PERF_BOOST_FLAGS_CMD_BOOST_TO_MAX))), ROOT)
    vaspace = o.rm_alloc(device, nv_gpu.FERMI_VASPACE_A, nv_gpu.NV_VASPACE_ALLOCATION_PARAMETERS(vaBase=0x1000, vaSize=0x1fffffb000000,
      flags=nv_gpu.NV_VASPACE_ALLOCATION_FLAGS_ENABLE_PAGE_FAULTING | nv_gpu.NV_VASPACE_ALLOCATION_FLAGS_IS_EXTERNALLY_OWNED), ROOT)
    channel_group = o.rm_alloc(device, nv_gpu.KEPLER_CHANNEL_GROUP_A,
                               nv_gpu.NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS(engineType=nv_gpu.NV2080_ENGINE_TYPE_GRAPHICS), ROOT)
    gpfifo_area = o.alloc(0x300000, contiguous=True, cpu_access=True, force_devmem=True)
    ctxshare = o.rm_alloc(channel_group, nv_gpu.FERMI_CONTEXT_SHARE_A, nv_gpu.NV_CTXSHARE_ALLOCATION_PARAMETERS(hVASpace=vaspace,
                          flags=nv_gpu.NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_ASYNC), ROOT)
    for offset, compute in ((0, True), (0x100000, False)):   # _new_gpu_fifo, entries 0x10000, not video
        notifier = o.alloc(48 << 20, uncached=True)
        params = nv_gpu.NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS(gpFifoOffset=gpfifo_area.va_addr+offset, gpFifoEntries=0x10000, hContextShare=ctxshare,
          hObjectError=notifier.meta.hMemory, hObjectBuffer=gpfifo_area.meta.hMemory, hUserdMemory=(ctypes.c_uint32*8)(gpfifo_area.meta.hMemory),
          userdOffset=(ctypes.c_uint64*8)(0x10000*8+offset), engineType=0, hVASpace=0)
        gpfifo = o.rm_alloc(channel_group, gsp.gpfifo_class, params, ROOT)
        if compute:
            obj = o.rm_alloc(gpfifo, gsp.compute_class, None, ROOT)
            o.rm_alloc(device, nv_gpu.GT200_DEBUGGER, nv_gpu.NV83DE_ALLOC_PARAMETERS(hAppClient=ROOT, hClass3dObject=obj), ROOT)
        else: o.rm_alloc(gpfifo, gsp.dma_class, None, ROOT)
        o.rm_control(gpfifo, nv_gpu.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN,
                     nv_gpu.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS(workSubmitToken=-1), ROOT)
    o.rm_control(channel_group, nv_gpu.NVA06C_CTRL_CMD_GPFIFO_SCHEDULE, nv_gpu.NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS(bEnable=1), ROOT)
    o.alloc(0x200000, cpu_access=True)   # cmdq_page
    o.rm_control(subdevice, nv_gpu.NV2080_CTRL_CMD_INTERNAL_STATIC_KGR_GET_INFO, nv_gpu.NV2080_CTRL_INTERNAL_STATIC_GR_GET_INFO_PARAMS(), ROOT)
    o.rm_control(0xdead, nv_gpu.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN,   # a channel the client never made: KeyError, as tinygrad
                 nv_gpu.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS(workSubmitToken=-1), ROOT)

# GB20x (MMU v3): Blackwell's classes, and the doorbell-enable bit in every work-submit token
GB20X = dict(gb2=1, gpfifo_class=nv_gpu.BLACKWELL_CHANNEL_GPFIFO_A, compute_class=nv_gpu.BLACKWELL_COMPUTE_B, dma_class=nv_gpu.BLACKWELL_DMA_COPY_B,
             viddec_class=nv_gpu.NVCFB0_VIDEO_DECODER)

def run_case(mmu, vram_mb, exe, priv, quiet=False):
    STATE.clear(); STATE.update(ADA if mmu == 2 else {**ADA, **GB20X})
    srv, path = gm.listen(priv)
    fake = gm.FakeTG(priv)
    t = gm.serve_once(srv, fake)
    dev, ifa = gm.py_dev(path, mmu, vram_mb)
    gm.boot_phase(dev, ifa)
    export = json.dumps(d._mm_export(types.SimpleNamespace(iface=types.SimpleNamespace(dev_impl=dev, pci_dev=dev.pci_dev))))
    gm.sync(dev.pci_dev)
    snap, mark = fake.snapshot(), len(fake.rec)
    qpath = f"{priv}/queues"
    qm = gg.init_queues(qpath)
    rm_fake = FakeRM(qm, fake)
    gsp = py_rm(dev, qpath)
    o = PyOps(gsp, ifa)
    nvdevice_rm(o, gsp)
    py_out = o.finish(dev.mm)
    dev.pci_dev.sock.close(); t.join(timeout=60)
    py_rec, py_bar1, py_q = bytes(fake.rec[mark:]), hashlib.sha256(fake.bar1).hexdigest(), bytes(qm)
    qm.close()

    fake2 = gm.FakeTG(priv, snap=snap)
    qm2 = gg.init_queues(qpath)
    FakeRM(qm2, fake2)
    t2 = gm.serve_once(srv, fake2)
    for name, text in (("export.json", export), ("ops.txt", "\n".join(o.ops) + "\n"),
                       ("state.txt", "\n".join([f"{k} {v}" for k, v in STATE.items() if k not in ("runlists", "grctx")] +
                                               [f"runlist {k} {v}" for k, v in STATE["runlists"].items()] +
                                               [f"grctx {' '.join(map(str, g))}" for g in STATE["grctx"]]) + "\n")):
        with open(f"{priv}/{name}", "w") as f: f.write(text)
    r = subprocess.run([exe, f"{priv}/export.json", f"{priv}/state.txt", f"{priv}/ops.txt"], capture_output=True, text=True, timeout=300,
                       env=dict(os.environ, TMPDIR=priv, APL_REMOTE_SOCK=path, BEAGLE_TINYGPU_NO_LAUNCH="1", GOLDEN_QUEUES=qpath))
    t2.join(timeout=60); srv.close()
    cpp_out = r.stdout.splitlines() + ([f"exit {r.returncode}: {r.stderr.strip()}"] if r.returncode else [])
    crec, cbar1, cq = bytes(fake2.rec), hashlib.sha256(fake2.bar1).hexdigest(), bytes(qm2)
    qm2.close()
    same = py_out == cpp_out and py_rec == crec and py_bar1 == cbar1 and py_q == cq
    frames = gm.count_frames(py_rec)
    rpcs = len(rm_fake.rpcs)
    summary = (f"{len(o.ops)} calls ({sum(l.startswith('error') for l in py_out)} raised), {rpcs} RPCs, {frames['MMIO_READ']} BAR reads, "
               f"{frames['MMIO_WRITE']} writes, {frames['MAP_SYSMEM_FD']} sysmem")
    if not same and not quiet:
        i = next((i for i, (a, b) in enumerate(zip(py_out, cpp_out)) if a != b), min(len(py_out), len(cpp_out)))
        summary += (f"\n   first differing result line {i} of {len(py_out)}/{len(cpp_out)}: tinygrad {py_out[i:i + 1]} c++ {cpp_out[i:i + 1]}"
                    f"\n   streams {'same' if py_rec == crec else 'differ'}, VRAM {'same' if py_bar1 == cbar1 else 'differs'}, "
                    f"queues {'same' if py_q == cq else 'differ'}")
    return same, summary

# a perturbed copy of the port must be caught
PERTURBED = [("TinyGPUHybridNVRM.h", "TGVirtMapping fresh = mm.valloc(desc.size, 0x1000, false, true);   // allocate buffers (dict.get's default: always)\n            const TGVirtMapping& x = bufs && bufs->count(buf) ? bufs->at(buf) : fresh;",
              "const TGVirtMapping& x = bufs && bufs->count(buf) ? bufs->at(buf) : mm.valloc(desc.size, 0x1000, false, true);"),
             ("TinyGPUHybridNVRM.h", "p.workSubmitToken |= (it->second << 16) | (gb2 ? (1u << 30) : 0);", "p.workSubmitToken |= (it->second << 8);"),
             ("TinyGPUHybridNVRM.h", "p.userdMem = {p.hUserdMemory[0] + p.userdOffset[0], 0x400, 2, 0};", "p.userdMem = {p.hUserdMemory[0], 0x400, 2, 0};")]

def main():
    exe = f"{WORK}/golden_rm"
    tgpaths.build_cpp(f"{HERE}/golden_rm.cpp", exe)
    priv = tempfile.mkdtemp(dir="/tmp", prefix="tgrm.")
    tempfile.tempdir = priv
    fails = 0
    for mmu, vram in ((2, 8188), (3, 16304)):
        same, summary = run_case(mmu, vram, exe, priv)
        fails += not same
        print(f"{'IDENTICAL' if same else 'MISMATCH '} NVDevice's RM calls, {'Ada, MMU v2' if mmu == 2 else 'GB20x, MMU v3'} ({vram} MiB): {summary}")
    caught, gpu = 0, tgpaths.REPO / "libhmsbeagle" / "GPU"
    for hdr, old, new in PERTURBED:
        inc = f"{priv}/perturbed"; os.makedirs(f"{inc}/libhmsbeagle/GPU", exist_ok=True)
        text = (gpu / hdr).read_text()
        assert text.count(old) == 1, (hdr, old)
        open(f"{inc}/libhmsbeagle/GPU/{hdr}", "w").write(text.replace(old, new))
        pexe = f"{priv}/golden_rm_perturbed"
        tgpaths.build_cpp(f"{HERE}/golden_rm.cpp", pexe, "-iquote", inc)
        hit = not run_case(2, 8188, pexe, priv, quiet=True)[0]
        caught += hit
        print(f"perturbed {hdr} ({old.splitlines()[0][:60]} ...): {'REJECTED' if hit else 'NOT CAUGHT'}")
    fails += len(PERTURBED) - caught
    print(f"C7 RM client vs tinygrad: {'all identical' if not fails else f'{fails} FAILED'}")
    sys.exit(1 if fails else 0)

if __name__ == "__main__":
    main()
