"""Golden test for TODO.md plan step C8's C++ half against the code it ports: nv_gsp_init_hw (TinyGPUHybridNVRM.h) is
tinygrad's NV_GSP.init_hw (ip.py:510-520) with nv_init_helper's patch 3 (the 20 s sleep after SEC2's start, while init_hw
runs) and init_golden_image (:468-508), on C5's GSP queues and CPU sequencer and C6's memory manager, with page_tables
(TinyGPUMemory.h, memory.py:204-206). One fake TinyGPU.app: golden_mm.py's (BAR1 VRAM, tinygrad's memory manager after a
boot-like sequence) with golden_gsp.py's falcon script on BAR0, and golden_gsp.py's GSP in a shared queue file, which has
queued a CPU sequencer and GSP_INIT_DONE, as GSP-RM does while it boots, and answers the golden image's RPCs (a device-info
table whose last entry repeats a key; context-buffer sizes). tinygrad runs init_hw first; golden_gsp_hw.cpp then does the
same from the same state. Both must print the same results (the recorded sleeps; the error, if any; the RM state after: the
handles, the runlists, the golden channel's runlist, the context buffers; the allocator states), send TinyGPU.app the same
requests byte for byte, leave the same queue memory and VRAM, and send the GSP the same RPCs. Cases: Ada (MMU v2) with every
sequencer op, core resume last; GB20x (MMU v3) with the COT boot's second BAR1 block register; no GSP_INIT_DONE; a refused
control. Then perturbed copies of the port must fail. No GPU and no TinyGPU.app.
    python golden_gsp_hw.py"""
import os, sys, io, json, mmap, ctypes, struct, tempfile, itertools, subprocess, hashlib, types, functools, contextlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tgpaths
tgpaths.setup()
import golden_mm as gm     # the fake TinyGPU.app, tinygrad's memory manager on it, the boot-like fork point
import golden_gsp as gg    # the GSP in the queue file, the falcon script, the sequencer's words
import nv_init_helper as h
import nv_dispatch_daemon as d
from tinygrad.runtime.support.nv import ip
from tinygrad.runtime.support.nv.ip import NV_FLCN, NV_GSP, NVRpcQueue
from tinygrad.runtime.support.hcq import MMIOInterface
from tinygrad.runtime.autogen import nv, nv_570 as nv_gpu

HERE, WORK = str(tgpaths.HERE), str(tgpaths.WORK)
SEQ0 = 2                 # the command queue after init_sw: SET_SYSTEM_INFO and SET_REGISTRY, prequeued
WAIT_MS, RPC_MS = 30, 2000
DEVICE_INFO = [(0, 3), (1, 0), (14, 1), (15, 2), (19, 4), (1, 7)]   # (engineData[2], engineData[3]); a dict keeps the last 1
CTX = {0: (0x25f000, 0x1000), 16: (0x3f800, 0x10000),               # (size, alignment): graphics, its patch buffer, and 17-24
       **{17 + i: (0x81000 + i * 0x13000, [0x1000, 0x10000, 0x20000][i % 3]) for i in range(8)}}
ALL_OPS = gg.ALL_OPS                                                 # every op, core resume (the 20 s sleep) last
NO_FALCON = [0x0, 0x1000, 0xabcd, 0x1, 0x1004, 0x30, 0xf0, 0x2, 0x1008, 0xf, 0x5, 0, 0, 0x3, 10, 0x4, 0x100c, 3]
ADA = dict(gpfifo_class=nv_gpu.AMPERE_CHANNEL_GPFIFO_A, compute_class=nv_gpu.ADA_COMPUTE_A, dma_class=nv_gpu.AMPERE_DMA_COPY_B,
           viddec_class=nv_gpu.NVC9B0_VIDEO_DECODER, gb2=0)
GB20X = dict(gpfifo_class=nv_gpu.BLACKWELL_CHANNEL_GPFIFO_A, compute_class=nv_gpu.BLACKWELL_COMPUTE_B, dma_class=nv_gpu.BLACKWELL_DMA_COPY_B,
             viddec_class=nv_gpu.NVCFB0_VIDEO_DECODER, gb2=1)

class FakeInitGsp(gg.FakeGsp):
    """golden_gsp.py's GSP as GSP-RM boots: a CPU sequencer, then (unless init_done is false) GSP_INIT_DONE, queued before
    init_hw runs; then a reply to every RPC, with what the golden image's controls read filled in, and rpc_result 0x1f for the
    control fail names."""
    def __init__(self, mm, tg, seq_words, init_done=True, fail=None):
        super().__init__(mm, tg)
        self.fail = fail
        self.post(nv.NV_VGPU_MSG_EVENT_GSP_RUN_CPU_SEQUENCER, gg.seq_msg(seq_words))
        if init_done: self.post(nv.NV_VGPU_MSG_EVENT_GSP_INIT_DONE, bytes(8))
    def on_write(self, a, v):
        if a != gg.QUEUE_HEAD: return
        for fn, msg, elem, ok in self.cmdq.new():
            struct.pack_into("<I", self.mm, gg.STATQ + 32, self.cmdq.rp)
            self.rpcs.append((fn, ok, elem.seqNum))
            self.post(fn, *self.reply(fn, msg))
    def reply(self, fn, msg):
        if fn != nv.NV_VGPU_MSG_FUNCTION_GSP_RM_CONTROL: return msg, 0
        n = ctypes.sizeof(nv.rpc_gsp_rm_control_v)
        c = nv.rpc_gsp_rm_control_v.from_buffer_copy(msg[:n])
        if c.cmd == self.fail: return msg, 0x1f
        params = msg[n:]
        if c.cmd == nv_gpu.NV2080_CTRL_CMD_FIFO_GET_DEVICE_INFO_TABLE:
            p = nv_gpu.NV2080_CTRL_FIFO_GET_DEVICE_INFO_TABLE_PARAMS.from_buffer_copy(params)
            p.numEntries = len(DEVICE_INFO)
            for i, (k, v) in enumerate(DEVICE_INFO): p.entries[i].engineData[2], p.entries[i].engineData[3] = k, v
            params = bytes(p)
        elif c.cmd == nv_gpu.NV2080_CTRL_CMD_INTERNAL_STATIC_KGR_GET_CONTEXT_BUFFERS_INFO:
            p = nv_gpu.NV2080_CTRL_INTERNAL_STATIC_KGR_GET_CONTEXT_BUFFERS_INFO_PARAMS.from_buffer_copy(params)
            for i, (size, align) in CTX.items():
                p.engineContextBuffersInfo[0].engine[i].size, p.engineContextBuffersInfo[0].engine[i].alignment = size, align
            params = bytes(p)
        return bytes(msg[:n]) + params, 0

def with_falcons(fake):
    """golden_gsp.py's falcon script (both falcons, the BSI scratch, the sequencer's polled register) on the fake's BAR0."""
    s = gg.falcon_script()
    fake.vals.update(s.vals); fake.hooks.extend(s.hooks)
    return fake

def py_gsp(dev, qpath, cls):
    """NV_GSP as init_sw leaves it (ip.py:347-362): the command queue on the queue file (init_rm_args), its two prequeued RPCs
    counted, the handle generator, the classes; no status queue yet (init_hw builds it)."""
    fd = os.open(qpath, os.O_RDWR)
    m = mmap.mmap(fd, gg.QSIZE)
    os.close(fd)
    base = MMIOInterface(gg.ctypes_addr(m), gg.QSIZE, fmt='B')
    gsp = NV_GSP.__new__(NV_GSP)
    gsp.nvdev, gsp.libos_args_sysmem, gsp._keep = dev, gg.LIBOS, m
    gsp.cmd_q_view, gsp.stat_q_view = base.view(gg.CMDQ), base.view(gg.STATQ)
    gsp.cmd_q = NVRpcQueue(gsp, gsp.cmd_q_view, None)
    gsp.cmd_q.seq = SEQ0
    gsp.handle_gen, gsp.chan_runlists = itertools.count(0xcf000000), {}
    gsp.gpfifo_class, gsp.compute_class, gsp.dma_class, gsp.viddec_class = (cls[k] for k in ("gpfifo_class", "compute_class", "dma_class", "viddec_class"))
    fl = NV_FLCN.__new__(NV_FLCN)
    fl.nvdev, fl.falcon, fl.sec2 = dev, gg.GSP, gg.SEC2
    dev.flcn, dev.gsp, dev.chip_id, dev.is_err_state = fl, gsp, gg.CHIP_ID, False
    return gsp

def state_lines(gsp, mm):
    g = gsp
    return [f"rm priv_root {getattr(g, 'priv_root', 0)} next_handle {int(repr(g.handle_gen)[6:-1])} device {getattr(g, 'device', 0)} "
            f"subdevice {getattr(g, 'subdevice', 0)} seq {g.cmd_q.seq}",
            "rm runlists" + "".join(f" {k}:{v}" for k, v in sorted(getattr(g, "runlists", {}).items())),
            "rm chan_runlists" + "".join(f" {k}:{v}" for k, v in sorted(g.chan_runlists.items())),
            "rm grctx" + "".join(f" {i}:{b.size}:{int(b.phys)}:{int(b.virt)}:{int(b.local)}" for i, b in getattr(g, "grctx_bufs", {}).items()),
            "state pa " + " ".join(map(str, d._tlsf_save(mm.pa_allocator))),
            "state va " + " ".join(map(str, d._tlsf_save(mm.va_allocator)))]

CASES = [("Ada (MMU v2): every sequencer op, core resume last, then GSP_INIT_DONE", 2, 8188, ADA, ALL_OPS, True, None),
         ("GB20x (MMU v3): the COT boot's second BAR1 block register", 3, 16304, GB20X, NO_FALCON, True, None),
         ("no GSP_INIT_DONE", 2, 8188, ADA, NO_FALCON, False, None),
         ("the context-buffer sizes refused", 2, 8188, ADA, NO_FALCON, True, nv_gpu.NV2080_CTRL_CMD_INTERNAL_STATIC_KGR_GET_CONTEXT_BUFFERS_INFO)]

def run_case(case, exe, priv, saved_sleep, quiet=False):
    name, mmu, vram_mb, cls, words, init_done, fail = case
    srv, path = gm.listen(priv)
    fake = with_falcons(gm.FakeTG(priv))
    t = gm.serve_once(srv, fake)
    dev, ifa = gm.py_dev(path, mmu, vram_mb)
    gm.boot_phase(dev, ifa)
    export = json.dumps(d._mm_export(types.SimpleNamespace(iface=types.SimpleNamespace(dev_impl=dev, pci_dev=dev.pci_dev))))
    gm.sync(dev.pci_dev)
    snap, mark = fake.snapshot(), len(fake.rec)
    qpath = f"{priv}/queues"
    qm = gg.init_queues(qpath, cmd_wp=SEQ0, stat_wp=0)
    gfake = FakeInitGsp(qm, fake, words, init_done, fail)
    gsp = py_gsp(dev, qpath, cls)
    out = []
    h.time.sleep = lambda s: out.append(f"sleep {s:g}") if s >= 1 else saved_sleep(s)
    try:
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()): gsp.init_hw()   # nv_init_helper's _patched_gsp_init_hw
    except Exception as e: out.append(f"error {type(e).__name__}: {e}")
    finally: h.time.sleep = saved_sleep
    py_out = out + state_lines(gsp, dev.mm)
    dev.pci_dev.sock.close(); t.join(timeout=60)
    py_rec, py_bar1, py_q, py_rpcs = bytes(fake.rec[mark:]), hashlib.sha256(fake.bar1).hexdigest(), bytes(qm), list(gfake.rpcs)
    qm.close()

    fake2 = with_falcons(gm.FakeTG(priv, snap=snap))
    qm2 = gg.init_queues(qpath, cmd_wp=SEQ0, stat_wp=0)
    gfake2 = FakeInitGsp(qm2, fake2, words, init_done, fail)
    t2 = gm.serve_once(srv, fake2)
    state = dict(seq=SEQ0, **cls, fmc_boot=int(mmu == 3), chip_id=gg.CHIP_ID, libos=gg.LIBOS, wait_ms=WAIT_MS, rpc_timeout_ms=RPC_MS)
    with open(f"{priv}/export.json", "w") as f: f.write(export)
    with open(f"{priv}/state.txt", "w") as f: f.write("".join(f"{k} {v}\n" for k, v in state.items()))
    r = subprocess.run([exe, f"{priv}/export.json", f"{priv}/state.txt"], capture_output=True, text=True, timeout=300,
                       env=dict(os.environ, TMPDIR=priv, APL_REMOTE_SOCK=path, BEAGLE_TINYGPU_NO_LAUNCH="1", GOLDEN_QUEUES=qpath,
                                BEAGLE_TINYGPU_LOG=f"{priv}/c8.log"))
    t2.join(timeout=60); srv.close()
    cpp_out = r.stdout.splitlines() + ([f"exit {r.returncode}: {r.stderr.strip()}"] if r.returncode else [])
    crec, cbar1, cq, crpcs = bytes(fake2.rec), hashlib.sha256(fake2.bar1).hexdigest(), bytes(qm2), list(gfake2.rpcs)
    qm2.close()
    same = py_out == cpp_out and py_rec == crec and py_bar1 == cbar1 and py_q == cq and py_rpcs == crpcs
    frames = gm.count_frames(py_rec)
    err = next((l for l in py_out if l.startswith("error")), "")
    summary = (f"{len(py_rpcs)} RPCs, {frames['MMIO_READ']} reads, {frames['MMIO_WRITE']} writes" + (f"; {err[:110]}" if err else ""))
    if not same and not quiet:
        i = next((i for i, (a, b) in enumerate(zip(py_out, cpp_out)) if a != b), min(len(py_out), len(cpp_out)))
        summary += (f"\n   first differing result line {i} of {len(py_out)}/{len(cpp_out)}: tinygrad {py_out[i:i + 1]} c++ {cpp_out[i:i + 1]}"
                    f"\n   streams {'same' if py_rec == crec else 'differ at byte %d' % next((k for k in range(min(len(py_rec), len(crec))) if py_rec[k] != crec[k]), min(len(py_rec), len(crec)))}"
                    f" ({len(py_rec)}/{len(crec)} bytes), VRAM {'same' if py_bar1 == cbar1 else 'differs'}, "
                    f"queues {'same' if py_q == cq else 'differ'}, RPCs {'same' if py_rpcs == crpcs else f'{py_rpcs} | {crpcs}'}")
    return same, summary

# a perturbed copy of the port must be caught
PERTURBED = [("TinyGPUHybridNVRM.h", 'rm.mm.dev->reg(nv_regs::NV_PBUS_BAR1_BLOCK).write({{"mode", 0}, {"target", 0}, {"ptr", 0}});',
              'rm.mm.dev->reg(nv_regs::NV_PBUS_BAR1_BLOCK).write({{"mode", 1}, {"target", 0}, {"ptr", 0}});'),
             ("TinyGPUHybridNVRM.h", "cfgs_sizes[x] = ctx_info(x + 14, 0, x == 5 ? (2 << 20) : 0);", "cfgs_sizes[x] = ctx_info(x + 14, 0, 0);"),
             ("TinyGPUHybridNVRM.h", "l.size = i == 0 ? mm.pte_cnt[0] * 8 : 0x1000;", "l.size = 0x1000;"),
             ("TinyGPUHybridNVRM.h", "rl[di.entries[i].engineData[2]] = di.entries[i].engineData[3];",
              "rl.emplace(di.entries[i].engineData[2], di.entries[i].engineData[3]);"),
             ("TinyGPUHybridNVRM.h", "explicit InGspInit(NVFalcon& f) : flcn(f) { flcn.sleep_after_sec2_start = true; }",
              "explicit InGspInit(NVFalcon& f) : flcn(f) {}"),
             ("TinyGPUMemory.h", "const uint64_t paddr = 0;\n        ctx.next(", "const uint64_t paddr = 0x1000;\n        ctx.next(")]

def main():
    exe = f"{WORK}/golden_gsp_hw"
    tgpaths.build_cpp(f"{HERE}/golden_gsp_hw.cpp", exe)
    priv = tempfile.mkdtemp(dir="/tmp", prefix="tggh.")
    tempfile.tempdir = priv
    saved_wait, saved_resp, saved_sleep = ip.wait_cond, NVRpcQueue.wait_resp, h.time.sleep
    ip.wait_cond = functools.partial(saved_wait, timeout_ms=WAIT_MS)                 # timeouts in milliseconds, as golden_gsp.py's
    NVRpcQueue.wait_resp = functools.partialmethod(saved_resp, timeout=RPC_MS)
    fails = 0
    try:
        for case in CASES:
            same, summary = run_case(case, exe, priv, saved_sleep)
            fails += not same
            print(f"{'IDENTICAL' if same else 'MISMATCH '} {case[0]}: {summary}")
        gpu = tgpaths.REPO / "libhmsbeagle" / "GPU"
        for hdr, old, new in PERTURBED:
            inc = f"{priv}/perturbed"; os.makedirs(f"{inc}/libhmsbeagle/GPU", exist_ok=True)
            for f in ("TinyGPUHybridNVRM.h", "TinyGPUMemory.h"):
                text = (gpu / f).read_text()
                if f == hdr: assert text.count(old) == 1, (hdr, old); text = text.replace(old, new)
                open(f"{inc}/libhmsbeagle/GPU/{f}", "w").write(text)
            tgpaths.build_cpp(f"{HERE}/golden_gsp_hw.cpp", f"{priv}/golden_gsp_hw_perturbed", "-iquote", inc)   # found before the repository's
            caught = not all(run_case(c, f"{priv}/golden_gsp_hw_perturbed", priv, saved_sleep, quiet=True)[0] for c in CASES[:2])
            fails += not caught
            print(f"perturbed {hdr} ({old[:70]} -> {new[:70]}): {'REJECTED' if caught else 'NOT CAUGHT'}")
    finally:
        ip.wait_cond, NVRpcQueue.wait_resp, h.time.sleep = saved_wait, saved_resp, saved_sleep
    print("C8 init_hw and golden image vs tinygrad: " + ("all identical, every perturbation rejected" if not fails else f"{fails} FAILED"))
    sys.exit(1 if fails else 0)

if __name__ == "__main__":
    main()
