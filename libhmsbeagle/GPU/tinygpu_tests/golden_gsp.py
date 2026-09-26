"""Golden test for TinyGPUHybridNVGsp.h and TinyGPUHybridNVFalcon.h (TODO.md plan step C5) against the code they port:
tinygrad's NVRpcQueue, NV_GSP.run_cpu_seq and NV_FLCN primitives, and nv_init_helper's unload and teardown (plan steps
P1, P2). Each scenario runs twice against the same scripted TinyGPU.app (BAR0 registers from a script, and a GSP in the
shared queue memory that answers RPCs): once with tinygrad's and nv_init_helper's own code in this process, then with
golden_gsp.cpp. Both must send the same requests (where a poll runs into a timeout, the same once each poll's repeated
reads are collapsed), leave the same queue memory, and report the same: the message returned, the exception's type and
text, nv_init_helper's beagle_fini. Covers: RPC framing (one record, continuation records, a wrap at the queue's end);
the status queue (a CPU sequencer, rpc_result != 0, an error-log event); the CPU sequencer's nine ops, an unknown op and
truncated operands; every P2 teardown scenario of test_p2_teardown.py; and the whole C++ fini (unload RPC, suspend wait,
teardown), FAST_UNLOAD and LEVEL_0 with an op-8 sequencer (the 20 s sleep between SEC2's start and the BSI read), never
suspended, and an unanswered RPC. No GPU and no TinyGPU.app: a private socket and TMPDIR.
    python golden_gsp.py"""
import os, sys, io, json, mmap, socket, struct, tempfile, threading, subprocess, functools, contextlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tgpaths
tgpaths.setup()
sys.path.insert(0, str(tgpaths.HERE / "replay"))
import nv_init_helper as h
import tgwire, tggpu
from tinygrad.runtime.support.system import APLRemotePCIDevice, RemoteCmd
from tinygrad.runtime.support.nv import ip
from tinygrad.runtime.support.nv.nvdev import NVDev
from tinygrad.runtime.support.nv.ip import NV_FLCN, NV_GSP, NVRpcQueue
from tinygrad.runtime.support.hcq import MMIOInterface
from tinygrad.runtime.autogen import nv

HERE, WORK = str(tgpaths.HERE), str(tgpaths.WORK)
REQ, RESP = "<BIIQQQ", "<BQQ"
BARS = {0: (0x1c_0000_0000, 16 << 20), 1: (0x1d_0000_0000, 256 << 20)}
GSP, SEC2 = 0x110000, 0x840000
QSIZE, CMDQ, STATQ, QUEUE, MSGS = 0x81000, 0x1000, 0x41000, 0x40000, 63   # init_rm_args's layout (pt_size 0x1000)
CHIP_ID, LIBOS, SEQ, WAIT_MS, RPC_MS = 0x197000a1, 0x1234000, 55, 30, 100
R = tggpu.regs("ada")
def addr(reg, base=0): r = reg.with_base(base); return r.base + r.off
WPR2_LO, WPR2_HI = addr(R.NV_PFB_PRI_MMU_WPR2_ADDR_LO), addr(R.NV_PFB_PRI_MMU_WPR2_ADDR_HI)
GSP_MBX0, QUEUE_HEAD = addr(R.NV_PGSP_FALCON_MAILBOX0), addr(R.NV_PGSP_QUEUE_HEAD[0])
IMAGES = dict(sb_paddr=0x300000, sb_imem_pa=0, sb_imem_va=0, sb_imem_sz=0x1000, sb_dmem_pa=0, sb_dmem_sz=0x400, sb_pkc_off=0x40,
              sb_engid=4, sb_ucodeid=9, unload_paddr=0x340000, unload_data_off=0x5000, unload_data_sz=0x4e00, unload_code_off=0x100,
              unload_code_sz=0x4f00)

# ── the scripted TinyGPU.app ──────────────────────────────────────────────────────────────────────────────────────────
class Script:
    """BAR0 from a script: byte address -> value, or a function of the address; every access in trace, every write also to
    on_write (the scenario's state machine)."""
    def __init__(self): self.vals, self.trace, self.hooks = {}, [], []
    def read(self, a):
        v = self.vals.get(a, 0)
        v = (v(a) if callable(v) else v) & 0xffffffff
        self.trace.append(("R", a, v))
        return v
    def write(self, a, v):
        self.trace.append(("W", a, v))
        for f in self.hooks: f(a, v)

def recv_exact(conn, n):
    b = bytearray()
    while len(b) < n:
        chunk = conn.recv(n - len(b))
        if not chunk: return None
        b += chunk
    return bytes(b)

def serve(conn, script, rec):
    while (hdr := recv_exact(conn, 33)) is not None:
        rec += hdr
        cmd, _, bar, a0, a1, _ = struct.unpack(REQ, hdr)
        if cmd == RemoteCmd.MMIO_WRITE:
            data = recv_exact(conn, a1); rec += data
            if bar == 0 and a1 == 4: script.write(a0, struct.unpack("<I", data)[0])
        elif cmd == RemoteCmd.MAP_BAR: conn.sendall(struct.pack(RESP, 0, *BARS[bar]))
        elif cmd == RemoteCmd.MMIO_READ and bar == 0 and a1 == 4: conn.sendall(struct.pack(RESP, 0, 4, 0) + struct.pack("<I", script.read(a0)))
        else: conn.sendall(struct.pack(RESP, 1, 0, 0))
    conn.close()

class FakeGsp:
    """GSP-RM in the queue file, as fake_nv_device.py's Gsp answers: at each command-queue head write, every RPC queued since
    (its checksum checked), then the scenario's events, then (unless silent) the reply and (unless never_suspend) MAILBOX0
    0x80000000."""
    def __init__(self, mm, script, events=(), silent=False, never_suspend=False):
        self.mm, self.script, self.events, self.silent, self.never_suspend, self.seq = mm, script, list(events), silent, never_suspend, 0
        self.cmdq = tggpu.QueueReader(mm, CMDQ)
        self.cmdq.rp = struct.unpack_from("<I", mm, STATQ + 32)[0]   # this GSP's command-queue read pointer
        self.rpcs = []
        script.hooks.append(self.on_write)
    def on_write(self, a, v):
        if a != QUEUE_HEAD: return
        for fn, msg, elem, ok in self.cmdq.new():
            struct.pack_into("<I", self.mm, STATQ + 32, self.cmdq.rp)
            self.rpcs.append((fn, ok))
            for ev in self.events: self.post(*ev)
            self.events = []
            if not self.silent: self.post(fn, msg)
            if fn == nv.NV_VGPU_MSG_FUNCTION_UNLOADING_GUEST_DRIVER and not self.never_suspend: self.script.vals[GSP_MBX0] = 0x80000000
    def post(self, fn, payload, rpc_result=0):
        post(self.mm, fn, payload, rpc_result, self.seq); self.seq += 1

def post(mm, fn, payload, rpc_result=0, seq=0):
    """One status-queue message in one element, as fake_nv_device.py's Gsp.post writes it."""
    wp = struct.unpack_from("<I", mm, STATQ + 16)[0]
    hdr = nv.rpc_message_header_v(signature=nv.NV_VGPU_MSG_SIGNATURE_VALID, header_version=3 << 24, rpc_result=rpc_result,
                                  rpc_result_private=rpc_result, function=fn, length=0x20 + len(payload), sequence=seq)
    elem = nv.GSP_MSG_QUEUE_ELEMENT(elemCount=1, seqNum=seq)
    elem.checkSum = tggpu.checksum(bytes(elem) + bytes(hdr) + payload)
    off = STATQ + 0x1000 + wp * 0x1000
    mm[off:off + 0x50 + len(payload)] = bytes(elem) + bytes(hdr) + payload
    struct.pack_into("<I", mm, STATQ + 16, (wp + 1) % MSGS)

def init_queues(path, cmd_wp=10, stat_wp=20):
    """The queue file as a boot leaves it: init_rm_args's command queue header, the GSP's status queue header, and both
    read pointers caught up."""
    with open(path, "wb") as f: f.write(bytes(QSIZE))
    fd = os.open(path, os.O_RDWR)
    mm = mmap.mmap(fd, QSIZE)
    os.close(fd)
    mm[CMDQ:CMDQ + 32] = bytes(nv.msgqTxHeader(version=0, size=QUEUE, entryOff=0x1000, msgSize=0x1000, msgCount=MSGS, writePtr=cmd_wp,
                                               flags=1, rxHdrOff=32))
    mm[STATQ:STATQ + 32] = bytes(nv.msgqTxHeader(version=0, size=QUEUE, entryOff=0x1000, msgSize=0x1000, msgCount=MSGS, writePtr=stat_wp,
                                                 flags=0, rxHdrOff=32))
    struct.pack_into("<I", mm, CMDQ + 32, stat_wp)    # the CPU's status-queue read pointer
    struct.pack_into("<I", mm, STATQ + 32, cmd_wp)    # the GSP's command-queue read pointer
    return mm

def falcon_script(wpr2_after_sb=0x1ffae00, bcr_valid=1, halted=1, sec2_halted=1, sec2_scrubbing=0, booter_mbx0=0,
                  wpr2_after_unload=None, sb_scratch=0):
    """test_p2_teardown.py's teardown_rig, as a Script: both falcons, WPR2, the VBIOS scratch and the boot progress."""
    s, st = Script(), {"booter": False}
    for base in (GSP, SEC2):
        s.vals[addr(R.NV_PFALCON_FALCON_HWCFG2, base)] = R.NV_PFALCON_FALCON_HWCFG2.encode(riscv=1, mem_scrubbing=sec2_scrubbing if base == SEC2 else 0)
        s.vals[addr(R.NV_PRISCV_RISCV_BCR_CTRL, base)] = R.NV_PRISCV_RISCV_BCR_CTRL.encode(valid=bcr_valid if base == GSP else 1)
        s.vals[addr(R.NV_PFALCON_FALCON_DMATRFCMD, base)] = R.NV_PFALCON_FALCON_DMATRFCMD.encode(idle=1, full=0)
        s.vals[addr(R.NV_PFALCON_FALCON_CPUCTL, base)] = R.NV_PFALCON_FALCON_CPUCTL.encode(halted=halted if base == GSP else sec2_halted)
    if wpr2_after_unload is None: wpr2_after_unload = 0 if booter_mbx0 == 0 else 0x1ffae00
    s.vals[addr(R.NV_PFALCON_FALCON_MAILBOX0, SEC2)] = lambda a: booter_mbx0 if st["booter"] else 0xff
    s.vals[WPR2_HI] = lambda a: wpr2_after_unload if st["booter"] else wpr2_after_sb
    s.vals[WPR2_LO] = 0x1f3b000
    s.vals[addr(R.NV_PBUS_VBIOS_SCRATCH[0x15])] = sb_scratch
    s.vals[addr(R.NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_PRIV_LEVEL_MASK)] = R.NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_PRIV_LEVEL_MASK.encode(read_protection_level0=1)
    s.vals[addr(R.NV_PGC6_AON_SECURE_SCRATCH_GROUP_05[0])] = 0xff
    s.vals[addr(R.NV_PGC6_BSI_SECURE_SCRATCH_14)] = R.NV_PGC6_BSI_SECURE_SCRATCH_14.encode(boot_stage_3_handoff=1)
    s.vals[addr(R.NV_PRISCV_RISCV_CPUCTL, GSP)] = 0x10
    s.vals[0x1008] = 0x5   # the sequencer scenario's polled register
    cpuctl_sec2 = addr(R.NV_PFALCON_FALCON_CPUCTL, SEC2)
    def on_write(a, v):
        if a == cpuctl_sec2 and v & R.NV_PFALCON_FALCON_CPUCTL.encode(startcpu=1): st["booter"] = True
    s.hooks.append(on_write)
    return s

# ── tinygrad's side ───────────────────────────────────────────────────────────────────────────────────────────────────
def tinygrad_dev(sock_path):
    """NVDev on the fake: tinygrad's remote BAR0 (map_bar, NVDev.__init__ nvdev.py:76) with Ada's include() sequence."""
    pci = object.__new__(APLRemotePCIDevice)
    pci.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    pci.sock.connect(sock_path)
    pci.pcibus, pci.dev_id = "usb4", 0
    dev = NVDev.__new__(NVDev)
    dev.pci_dev, dev.devfmt, dev.mmio = pci, "usb4", pci.map_bar(0, fmt='I')
    for name, arch in tgwire.INCLUDES["ada"]: dev.include(name, arch)
    dev.chip_id, dev.is_err_state = CHIP_ID, False
    fl = NV_FLCN.__new__(NV_FLCN)
    fl.nvdev, fl.falcon, fl.sec2 = dev, GSP, SEC2
    fl.desc_v3 = nv.FALCON_UCODE_DESC_V3(IMEMPhysBase=IMAGES["sb_imem_pa"], IMEMVirtBase=IMAGES["sb_imem_va"], IMEMLoadSize=IMAGES["sb_imem_sz"],
                                         DMEMPhysBase=IMAGES["sb_dmem_pa"], DMEMLoadSize=IMAGES["sb_dmem_sz"], PKCDataOffset=IMAGES["sb_pkc_off"],
                                         EngineIdMask=IMAGES["sb_engid"], UcodeId=IMAGES["sb_ucodeid"])
    fl.beagle_sb_image_paddr, fl.beagle_unload_image_paddr = IMAGES["sb_paddr"], IMAGES["unload_paddr"]
    fl.beagle_unload_params = (IMAGES["unload_data_off"], IMAGES["unload_data_sz"], IMAGES["unload_code_off"], IMAGES["unload_code_sz"])
    dev.flcn = fl
    return dev, fl

def tinygrad_gsp(dev, qpath, seq):
    """NV_GSP's queues as init_rm_args and init_hw build them (ip.py:382-386, 510-512), on the queue file."""
    fd = os.open(qpath, os.O_RDWR)
    m = mmap.mmap(fd, QSIZE)
    os.close(fd)
    base = MMIOInterface(ctypes_addr(m), QSIZE, fmt='B')
    gsp = NV_GSP.__new__(NV_GSP)
    gsp.nvdev, dev.gsp, gsp.libos_args_sysmem, gsp._keep = dev, gsp, LIBOS, m
    gsp.cmd_q_view, gsp.stat_q_view = base.view(CMDQ), base.view(STATQ)
    gsp.cmd_q = NVRpcQueue(gsp, gsp.cmd_q_view, None)
    gsp.stat_q = NVRpcQueue(gsp, gsp.stat_q_view, gsp.cmd_q_view)
    gsp.cmd_q.rx_view = gsp.stat_q_view.view(gsp.stat_q.tx.rxHdrOff, fmt='I')
    gsp.cmd_q.seq = seq
    return gsp

def ctypes_addr(m):
    import ctypes
    return ctypes.addressof(ctypes.c_char.from_buffer(m))

def py_err(e): return f"{type(e).__name__}: {e}"

def py_scenario(kind, dev, fl, qpath, kw, out):
    if kind == "teardown":
        dev.beagle_fini = {"unload_ok": bool(kw["unload_ok"])}
        fl.fini_hw()   # nv_init_helper's _flcn_fini_hw_teardown
        out.append("diag=" + json.dumps(dev.beagle_fini))
        return
    gsp = tinygrad_gsp(dev, qpath, kw.get("seq", 0))
    try:
        if kind == "rpc":
            gsp.cmd_q.send_rpc(kw["func"], bytes((i * 7 + 3) & 0xff for i in range(kw["len"])))
            out.append(f"seq={gsp.cmd_q.seq}")
        elif kind == "statq":
            with contextlib.redirect_stdout(io.StringIO()): msg = gsp.stat_q.wait_resp(kw["cmd"])
            out.append(f"msg={bytes(msg).hex()}"); out.append(f"rx={gsp.stat_q.rx_view[0]}"); out.append(f"err_state={int(dev.is_err_state)}")
        elif kind == "seq":
            words = kw["words"]
            buf = bytes(nv.rpc_run_cpu_sequencer_v17_00(bufferSizeDWord=len(words), cmdIndex=kw.get("cmd_index", len(words)))) + \
                  struct.pack(f"<{len(words)}I", *words)
            h._in_gsp_init[0] = bool(kw.get("sec2_sleep"))
            try: gsp.run_cpu_seq(buf)   # nv_init_helper's _logged_run_cpu_seq, then tinygrad's
            finally: h._in_gsp_init[0] = False
            out.append("done")
        elif kind == "fini":
            saved = NV_GSP.rpc_unloading_guest_driver, h._UNLOAD_LEVEL_0
            if kw.get("level0"): NV_GSP.rpc_unloading_guest_driver, h._UNLOAD_LEVEL_0 = h._rpc_unloading_guest_driver_level0, True
            try:
                try: gsp.fini_hw()   # nv_init_helper's _gsp_fini_hw_with_suspend_wait
                except Exception as e: out.append(f"unload error={py_err(e)}")
                fl.fini_hw()         # its _flcn_fini_hw_teardown, in NVDev.fini's order
            finally: NV_GSP.rpc_unloading_guest_driver, h._UNLOAD_LEVEL_0 = saved
            out.append("diag=" + json.dumps(dev.beagle_fini))
    except Exception as e: out.append(f"error={py_err(e)}")

# ── scenarios ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def seq_msg(words): return bytes(nv.rpc_run_cpu_sequencer_v17_00(bufferSizeDWord=len(words), cmdIndex=len(words))) + struct.pack(f"<{len(words)}I", *words)
ALL_OPS = [0x0, 0x1000, 0xabcd,  0x1, 0x1004, 0x30, 0xf0,  0x2, 0x1008, 0xf, 0x5, 0, 0,  0x3, 10,  0x4, 0x100c, 3,  0x5, 0x6, 0x7, 0x8]
FAST = struct.pack("<BBxxI", 0, 0, 1 << 6)
UNLOAD = nv.NV_VGPU_MSG_FUNCTION_UNLOADING_GUEST_DRIVER

def scenarios():
    """(name, kind, script kwargs, queue posts before the run, FakeGsp kwargs or None, python/c++ parameters, compare mode)"""
    q = lambda **k: k
    yield "RPC, one record", "rpc", {}, [], None, q(func=UNLOAD, len=8, seq=SEQ), "exact"
    yield "RPC, continuation records", "rpc", {}, [], None, q(func=0x4c, len=150000, seq=3), "exact"
    yield "RPC, wrapping at the queue's end", "rpc", {}, [], None, q(func=0x4c, len=9000, seq=7, cmd_wp=61), "exact"
    yield "status queue: sequencer, then the reply", "statq", {}, [(0x1020, bytes(16)), (0x1002, seq_msg([0x0, 0x1000, 0x7])),
          (0x100c, bytes(24)), (UNLOAD, FAST)], None, q(cmd=UNLOAD), "exact"
    yield "status queue: rpc_result != 0", "statq", {}, [(UNLOAD, FAST, 0x1f)], None, q(cmd=UNLOAD), "exact"
    yield "status queue: an error-log event", "statq", {}, [(0x1006, bytes(12) + b"GSP says no\0\0"), (UNLOAD, FAST)], None, q(cmd=UNLOAD), "exact"
    yield "status queue: a truncated sequencer", "statq", {}, [(0x1002, seq_msg([0x0, 0x1000])), (UNLOAD, FAST)], None, q(cmd=UNLOAD), "exact"
    yield "sequencer: all nine ops", "seq", {}, [], None, q(words=ALL_OPS), "collapsed"
    yield "sequencer: op 8 with the 20 s SEC2 sleep", "seq", {}, [], None, q(words=[0x8], sec2_sleep=1), "collapsed"
    yield "sequencer: an unknown op", "seq", {}, [], None, q(words=[0x0, 0x1000, 1, 0x9]), "exact"
    yield "sequencer: truncated operands", "seq", {}, [], None, q(words=[0x2, 0x1008]), "exact"
    yield "sequencer: cmdIndex shorter than the buffer", "seq", {}, [], None, q(words=[0x0, 0x1000, 1, 0x0, 0x1004, 2], cmd_index=3), "exact"
    for name, kw in (("happy path", {}), ("WPR2 down after FWSEC-SB", dict(wpr2_after_sb=0)), ("GSP core-select timeout", dict(bcr_valid=0)),
                     ("FWSEC-SB error 0x29", dict(sb_scratch=0xabcd0029)), ("FWSEC-SB never halts", dict(halted=0)),
                     ("SEC2 scrub timeout", dict(sec2_scrubbing=1)), ("Booter Unload never halts", dict(sec2_halted=0)),
                     ("Booter Unload error 0x29", dict(booter_mbx0=0x29)), ("Booter Unload 0x29, WPR2 down", dict(booter_mbx0=0x29, wpr2_after_unload=0))):
        yield f"teardown: {name}", "teardown", kw, [], None, q(unload_ok=1), "collapsed"
    yield "teardown: GSP not suspended", "teardown", {}, [], None, q(unload_ok=0), "exact"
    yield "fini: FAST_UNLOAD, then the teardown", "fini", {}, [], {}, q(seq=SEQ), "collapsed"
    yield "fini: LEVEL_0 with an op-8 sequencer", "fini", {}, [], dict(events=[(0x1002, seq_msg([0x8]))]), q(seq=SEQ, level0=1), "collapsed"
    yield "fini: never suspended", "fini", {}, [], dict(never_suspend=True), q(seq=SEQ), "collapsed"
    yield "fini: the unload RPC unanswered", "fini", {}, [], dict(silent=True), q(seq=SEQ), "collapsed"

def collapse(trace):
    out = []
    for x in trace:
        if not out or x != out[-1] or x[0] != "R": out.append(x)
    return out

# a perturbed copy of the port must be caught: (header, text, replacement, the scenario that shows it)
PERTURBED = [("TinyGPUHybridNVGsp.h", "c ^= w;", "c += w;", "RPC, one record"),
             ("TinyGPUHybridNVFalcon.h", "xfered += 256;", "xfered += 512;", "teardown: happy path"),
             ("TinyGPUHybridNVFalcon.h", "sleep(20);", "sleep(2);", "sequencer: op 8 with the 20 s SEC2 sleep")]

def main():
    exe = f"{WORK}/golden_gsp"
    tgpaths.build_cpp(f"{HERE}/golden_gsp.cpp", exe)
    priv = tempfile.mkdtemp(dir="/tmp", prefix="tggs.")
    sock_path, qpath = f"{priv}/s.sock", f"{priv}/queues"
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path); srv.listen(1); srv.settimeout(60)
    tempfile.tempdir = priv
    saved_wait, saved_resp, saved_sleep = ip.wait_cond, NVRpcQueue.wait_resp, h.time.sleep
    ip.wait_cond = functools.partial(saved_wait, timeout_ms=WAIT_MS)                 # timeouts in milliseconds, as in test_p2_teardown
    NVRpcQueue.wait_resp = functools.partialmethod(saved_resp, timeout=RPC_MS)
    h._TEARDOWN = True
    try:
        n, fails = compare(exe, srv, sock_path, qpath, priv, saved_sleep)
        print(f"C5 GSP client and falcon primitives vs tinygrad and nv_init_helper: {n - fails} of {n} scenarios identical")
        caught = 0
        gpu = tgpaths.REPO / "libhmsbeagle" / "GPU"
        for hdr, old, new, scenario in PERTURBED:
            inc = f"{priv}/perturbed"; os.makedirs(f"{inc}/libhmsbeagle/GPU", exist_ok=True)
            for f in ("TinyGPUHybridNVGsp.h", "TinyGPUHybridNVFalcon.h"):
                text = (gpu / f).read_text()
                if f == hdr: assert text.count(old) == 1, (hdr, old); text = text.replace(old, new)
                open(f"{inc}/libhmsbeagle/GPU/{f}", "w").write(text)
            tgpaths.build_cpp(f"{HERE}/golden_gsp.cpp", f"{priv}/golden_gsp_perturbed", "-iquote", inc)   # found before the repository's
            _, f = compare(f"{priv}/golden_gsp_perturbed", srv, sock_path, qpath, priv, saved_sleep, only=scenario, quiet=True)
            caught += f == 1
            print(f"perturbed {hdr} ({old} -> {new}): {'REJECTED' if f == 1 else 'NOT CAUGHT'} by '{scenario}'")
        fails += len(PERTURBED) - caught
    finally:
        ip.wait_cond, NVRpcQueue.wait_resp, h.time.sleep, h._TEARDOWN = saved_wait, saved_resp, saved_sleep, False
    sys.exit(1 if fails else 0)

def compare(exe, srv, sock_path, qpath, priv, saved_sleep, only=None, quiet=False):
    fails, n = 0, 0
    if True:
        for name, kind, skw, posts, gspkw, kw, mode in scenarios():
            if only is not None and name != only: continue
            n += 1
            results = {}
            for side in ("tinygrad", "c++"):
                script = falcon_script(**skw)
                mm = init_queues(qpath, cmd_wp=kw.get("cmd_wp", 10))
                for p in posts: post(mm, *p)
                gsp = FakeGsp(mm, script, **gspkw) if gspkw is not None else None
                rec, out = bytearray(), []
                def accept():
                    conn, _ = srv.accept(); conn.settimeout(120); serve(conn, script, rec)
                t = threading.Thread(target=accept, daemon=True); t.start()
                if side == "tinygrad":
                    h.time.sleep = lambda s: out.append(f"sleep {s:g}") if s >= 1 else saved_sleep(s)
                    try:
                        dev, fl = tinygrad_dev(sock_path)
                        with contextlib.redirect_stderr(io.StringIO()): py_scenario(kind, dev, fl, qpath, kw, out)
                    finally: h.time.sleep = saved_sleep
                    dev.pci_dev.sock.close()
                else:
                    args = [exe, kind] + [f"{k}={v:#x}" if isinstance(v, int) else f"{k}={','.join(hex(x) for x in v)}" for k, v in kw.items()]
                    args += [f"{k}={v:#x}" for k, v in IMAGES.items()] + [f"chip_id={CHIP_ID:#x}", f"wait_ms={WAIT_MS}", f"rpc_timeout_ms={RPC_MS}",
                                                                        f"libos={LIBOS:#x}"]
                    r = subprocess.run(args, capture_output=True, text=True, timeout=120,
                                       env=dict(os.environ, TMPDIR=priv, APL_REMOTE_SOCK=sock_path, BEAGLE_TINYGPU_NO_LAUNCH="1",
                                                GOLDEN_QUEUES=qpath, BEAGLE_TINYGPU_LOG=f"{priv}/c5.log"))
                    out = r.stdout.splitlines() + ([f"exit {r.returncode}: {r.stderr.strip()}"] if r.returncode else [])
                t.join(timeout=30)
                results[side] = (out, bytes(rec), script.trace, bytes(mm), gsp.rpcs if gsp else None)
                mm.close()
            (po, prec, ptr, pq, prpc), (co, crec, ctr, cq, crpc) = results["tinygrad"], results["c++"]
            parse = lambda lines: [json.loads(l[5:]) if l.startswith("diag=") else l for l in lines]
            same_out = parse(po) == parse(co)
            same_wire = prec == crec if mode == "exact" else collapse(ptr) == collapse(ctr)
            ok = same_out and same_wire and pq == cq and prpc == crpc
            fails += not ok
            summary = next((l for l in po if l.startswith(("diag=", "error=", "unload error="))), po[-1] if po else "")
            if not quiet: print(f"{'IDENTICAL' if ok else 'MISMATCH '} {name}: {len(collapse(ptr))} accesses; {summary[:150]}")
            if not ok and not quiet:
                if not same_out:
                    for a, b in zip(po + [""] * len(co), co + [""] * len(po)):
                        if a != b: print(f"    tinygrad: {a[:300]}\n    c++     : {b[:300]}")
                if not same_wire:
                    a, b = collapse(ptr), collapse(ctr)
                    k = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
                    print(f"    first access difference at #{k} of {len(a)}/{len(b)}: tinygrad {a[k:k + 3]} | c++ {b[k:k + 3]}")
                if pq != cq:
                    k = next(i for i in range(QSIZE) if pq[i] != cq[i])
                    print(f"    queue memory differs from byte {k:#x}")
                if prpc != crpc: print(f"    RPCs the GSP saw: tinygrad {prpc} | c++ {crpc}")
    return n, fails

main()
