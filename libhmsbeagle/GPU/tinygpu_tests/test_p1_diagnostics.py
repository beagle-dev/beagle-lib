"""Offline tests for plan step P1 (nv_init_helper.py's warm refusal and diagnostics, nv_dispatch_daemon.py's fini).
No GPU: the warm-refusal tests run the real daemon's cmd_boot, i.e. tinygrad's real NVDevice boot path, against a
scripted fake TinyGPU.app on a socketpair, each in its own subprocess; the rest drive each wrapper on fakes.
    python test_p1_diagnostics.py"""
import os, sys, json, struct, socket, threading, subprocess, tempfile, types, ctypes, io, contextlib, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tgpaths
tgpaths.setup()

WPR2_HI, BOOT_0, BOOT_42 = 0x1FA828, 0x0, 0xA00
CMD = {0: "PROBE", 1: "MAP_BAR", 2: "MAP_SYSMEM_FD", 3: "CFG_READ", 4: "CFG_WRITE", 5: "RESET", 6: "MMIO_READ", 7: "MMIO_WRITE",
       8: "MAP_SYSMEM", 9: "SYSMEM_READ", 10: "SYSMEM_WRITE", 11: "RESIZE_BAR", 12: "PING"}

# ── warm refusal, end to end (subprocess scenario) ───────────────────────────
def boot_scenario(wpr2_hi, guard):
    """Runs Daemon.cmd_boot over a scripted fake TinyGPU.app; prints the request stream and the boot reply as JSON."""
    import nv_dispatch_daemon as d
    from tinygrad.runtime.support import system
    reqs, regs = [], {WPR2_HI: wpr2_hi}
    fake, plugin = socket.socketpair()
    def recv(n):
        b = b""
        while len(b) < n:
            c = fake.recv(n - len(b))
            if not c: return None
            b += c
        return b
    def serve():
        while (h := recv(33)) is not None:
            cmd, dev, bar, a0, a1, a2 = struct.unpack("<BIIQQQ", h)
            if cmd == 7: recv(a1); reqs.append(("MMIO_WRITE", bar, a0, a1)); continue          # posted, no reply
            reqs.append((CMD.get(cmd, cmd), bar, a0, a1, a2) if cmd in (3, 4) else (CMD.get(cmd, cmd), bar, a0, a1))
            if cmd == 1: fake.sendall(struct.pack("<BQQ", 0, 0x10_0000_0000 + (bar << 32), {0: 16 << 20, 1: 256 << 20, 3: 32 << 20}.get(bar, 1 << 20)))
            elif cmd == 11 or cmd == 4: fake.sendall(struct.pack("<BQQ", 0, 0, 0))
            elif cmd == 3: fake.sendall(struct.pack("<BQQ", 0, 0x0006 if a0 == 4 else 0, 0))    # PCI_COMMAND: memory + bus-master off
            elif cmd == 6: fake.sendall(struct.pack("<BQQ", 0, 0, 0) + struct.pack("<I", regs.get(a0, 0)) * (a1 // 4))
            else:
                msg = b"fake: not scripted"; fake.sendall(struct.pack("<BQQ", 1, len(msg), 0) + msg)
    threading.Thread(target=serve, daemon=True).start()
    # the IOKit scan must not matter: the device is the fake on the inherited fd
    type(system.System).list_devices = lambda self, vendor, devices, base_class=None: [(system.APLRemotePCIDevice, "usb4")]
    import nv_init_helper
    if not guard: nv_init_helper.NVDev._early_ip_init = nv_init_helper._ORIG["early_ip_init"]
    cmd_a, cmd_b = socket.socketpair()
    dm = d.Daemon(cmd_b, plugin.fileno())
    try: dm.cmd_boot({})
    except Exception as e: dm.send_json({"ok": False, "error": f"{type(e).__name__}: {e}", "raised": True})
    n = struct.unpack("<I", cmd_a.recv(4, socket.MSG_WAITALL))[0]
    reply = json.loads(cmd_a.recv(n, socket.MSG_WAITALL))
    print("RESULT " + json.dumps({"reqs": reqs, "reply": reply}))

def run_scenario(wpr2_hi, guard=True):
    r = subprocess.run([sys.executable, __file__, "--boot-scenario", str(wpr2_hi), str(int(guard))], capture_output=True, text=True, timeout=120)
    line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT ")), None)
    assert line, f"scenario produced no result:\n{r.stdout}\n{r.stderr}"
    res = json.loads(line[7:])
    return [tuple(x) for x in res["reqs"]], res["reply"]

def test_warm_refusal():
    reqs, reply = run_scenario(0x1ff10)
    assert reqs == [("RESIZE_BAR", 1, 0, 0), ("MAP_BAR", 0, 0, 0), ("MMIO_READ", 0, WPR2_HI, 4)], reqs
    assert reply["ok"] is False and reply.get("warm") is True and "WARM GPU" in reply["error"] and "power-cycle" in reply["error"].lower(), reply
    print("warm GPU: refused after RESIZE_BAR, MAP_BAR(0) and one 4-byte read; no write of any kind; reply names the power cycle")

def test_cold_path_adds_one_read():
    guarded, reply = run_scenario(0, guard=True)
    stock, _ = run_scenario(0, guard=False)
    assert reply["ok"] is False and not reply.get("warm"), reply            # the fake then stops the boot (no chip id)
    assert guarded[:3] == [("RESIZE_BAR", 1, 0, 0), ("MAP_BAR", 0, 0, 0), ("MMIO_READ", 0, WPR2_HI, 4)], guarded
    assert guarded[3:] == stock[2:] and stock[:2] == guarded[:2], (guarded, stock)
    assert ("CFG_WRITE", 0, 4, 2, 0x0006 | 0x4) in stock, stock             # tinygrad's own bus-master write still happens
    print(f"cold GPU: the guard adds exactly one read in front of tinygrad's unchanged stream ({len(stock)} requests)")

# ── wrappers on fakes ────────────────────────────────────────────────────────
import nv_init_helper as h
from tinygrad.runtime.support.nv.nvdev import NVDev
from tinygrad.runtime.support.nv.ip import NV_GSP, NV_FLCN
from tinygrad.runtime.autogen import nv

class FakeMMIO:
    """BAR0 as tinygrad's NVDev uses it (index = byte offset / 4); scripted reads, recorded accesses."""
    def __init__(self, script=None): self.script, self.log = script or {}, []
    def __getitem__(self, i):
        if isinstance(i, slice): self.log.append(("rd", i.start * 4, (i.stop - i.start) * 4)); return [0x55AA0000 + k for k in range(i.stop - i.start)]
        v = self.script.get(i * 4, 0); v = v.pop(0) if isinstance(v, list) and len(v) > 1 else (v[0] if isinstance(v, list) else v)
        self.log.append(("rd", i * 4)); return v
    def __setitem__(self, i, v): self.log.append(("wr", i * 4, v))
    def writes(self): return [x for x in self.log if x[0] == "wr"]

def fake_nvdev(script=None):
    dev = NVDev.__new__(NVDev)
    dev.mmio, dev.chip_name = FakeMMIO(script), "AD107"
    for name, arch in (("nv_ref", ""), ("dev_fb", "tu102"), ("dev_gc6_island", "ga102"), ("dev_gsp", "ga102"), ("dev_falcon_v4", "ga102"),
                       ("dev_riscv_pri", "ga102"), ("dev_bus", "tu102")): dev.include(name, arch)
    return dev

@contextlib.contextmanager
def orig(name, fn):
    saved = h._ORIG[name]; h._ORIG[name] = fn
    try: yield
    finally: h._ORIG[name] = saved

def test_suspend_wait():
    mbx0 = h._GSP_BASE + 0x40
    dev = fake_nvdev({mbx0: [0, 0, 0x80000000], WPR2_HI: 0x1ff10})
    gsp = NV_GSP.__new__(NV_GSP); gsp.nvdev = dev
    calls = []
    with orig("gsp_fini_hw", lambda self: calls.append(("rpc", h._in_unload[0]))):
        gsp.fini_hw()
    d = dev.beagle_fini
    assert calls == [("rpc", True)] and not h._in_unload[0], calls
    assert d["unload_ok"] and d["mailbox0"] == 0x80000000 and d["wpr2_hi"] == 0x1ff10, d
    assert dev.mmio.writes() == [] and [x for x in dev.mmio.log if x == ("rd", mbx0)] == [("rd", mbx0)] * 3, dev.mmio.log
    dev2 = fake_nvdev({mbx0: 0x1})
    gsp.nvdev, h._SUSPEND_TIMEOUT_S = dev2, 0.05
    with orig("gsp_fini_hw", lambda self: None): gsp.fini_hw()
    h._SUSPEND_TIMEOUT_S = 2.0
    assert dev2.beagle_fini["unload_ok"] is False and dev2.beagle_fini["mailbox0"] == 0x1 and dev2.mmio.writes() == []
    def boom(self): raise RuntimeError("Timeout waiting for RPC response")
    gsp.nvdev = fake_nvdev()
    with orig("gsp_fini_hw", boom):
        try: gsp.fini_hw(); raise AssertionError("RPC failure swallowed")
        except RuntimeError as e: assert "Timeout" in str(e)
    assert not h._in_unload[0] and gsp.nvdev.beagle_fini == {"unload_ok": False}
    print("suspend wait: RPC first, then MAILBOX0 polled until 0x80000000; timeout and RPC failure report unload_ok false; reads only")

def test_event_logging():
    items = [(nv.NV_VGPU_MSG_FUNCTION_UNLOADING_GUEST_DRIVER, b"x"), (nv.NV_VGPU_MSG_EVENT_GSP_RUN_CPU_SEQUENCER, b"y")]
    def gen(self): yield from items
    err = io.StringIO()
    with orig("read_resp", gen), contextlib.redirect_stderr(err):
        h._in_unload[0] = True; got = list(h._logged_read_resp(object())); h._in_unload[0] = False
        quiet = list(h._logged_read_resp(object()))
    assert got == items and quiet == items and err.getvalue().count("status-queue event during unload") == 2, err.getvalue()
    # the sequencer-op parser against what tinygrad's own run_cpu_seq executes (ops 0-4 on a recording device)
    ops = [0x0, 0x100, 7, 0x1, 0x104, 0x3, 0xf, 0x2, 0x108, 0xff, 0x11, 0, 0, 0x3, 1, 0x4, 0x10c, 2, 0x0, 0x110, 9]
    hdr = nv.rpc_run_cpu_sequencer_v17_00(cmdIndex=len(ops))
    buf = bytes(hdr) + struct.pack(f"<{len(ops)}I", *ops)
    dev = fake_nvdev({0x108: 0x11}); gsp = NV_GSP.__new__(NV_GSP); gsp.nvdev = dev
    executed = []
    real_wreg, real_rreg = NVDev.wreg, NVDev.rreg
    h._ORIG["run_cpu_seq"](gsp, buf)   # tinygrad's run_cpu_seq, unwrapped
    for x in dev.mmio.log: executed.append(x)
    assert h._seq_ops(buf) == [0x0, 0x1, 0x2, 0x3, 0x4, 0x0], h._seq_ops(buf)
    assert [x for x in executed if x[0] == "wr"] == [("wr", 0x100, 7), ("wr", 0x104, 0x3 & 0xf), ("wr", 0x110, 9)], executed
    calls = []
    with orig("run_cpu_seq", lambda self, b: calls.append(b)), contextlib.redirect_stderr(io.StringIO()): gsp.run_cpu_seq(buf)
    assert calls == [buf]
    print("event logging: read_resp yields the same items; sequencer ops parsed as tinygrad's run_cpu_seq executes them")

def test_frts_checks():
    scratch = 0x1400 + 0x0e * 4
    dev = fake_nvdev({0x118128: 0x1, 0x118234: 0xff, scratch: 0x00000000, 0x1FA824: (0x1ff00 << 4)})
    fl = NV_FLCN.__new__(NV_FLCN); fl.nvdev, fl.frts_image_paddr, fl.frts_offset = dev, 0x2000, 0x1ff00 << 12
    calls, err = [], io.StringIO()
    with orig("execute_hs", lambda self, base, img, *a, **k: calls.append((base, img, a, k)) or "ret"), contextlib.redirect_stderr(err):
        assert fl.execute_hs(0x110000, 0x2000, 1, 2, x=3) == "ret"
        n_frts = len(dev.mmio.log)
        assert fl.execute_hs(0x840000, 0x9000, 4, mailbox=5) == "ret"
    assert calls == [(0x110000, 0x2000, (1, 2), {"x": 3}), (0x840000, 0x9000, (4,), {"mailbox": 5})], calls
    assert dev.mmio.writes() == [] and len(dev.mmio.log) == n_frts and n_frts == 4, dev.mmio.log
    assert "FRTS error code 0x0" in err.getvalue() and "(match)" in err.getvalue(), err.getvalue()
    print("FRTS checks: 2 reads before and 2 after FWSEC-FRTS only, pass-through arguments and result, reads only")

def test_userd_baseline():
    area = bytearray(0x300000)
    from tinygrad.runtime import ops_nv
    ctl = ops_nv.nv_gpu.AmpereAControlGPFifo
    struct.pack_into("<I", area, 0x100000 + 0x10000 * 8 + getattr(ctl, "GPGet").offset, 0x1234)
    struct.pack_into("<I", area, 0x100000 + 0x10000 * 8 + getattr(ctl, "GPPut").offset, 0x5678)
    reads = []
    class View:
        def __init__(self, off=0): self.off = off
        def view(self, off=0, size=None, fmt=None): return View(self.off + off)
        def __getitem__(self, i): reads.append(self.off + i * 4); return struct.unpack_from("<I", area, self.off + i * 4)[0]
    gp = types.SimpleNamespace(cpu_view=lambda: View())
    dev = ops_nv.NVDevice.__new__(ops_nv.NVDevice)
    with orig("new_gpu_fifo", lambda self, *a, **k: ("fifo", a, k)), contextlib.redirect_stderr(io.StringIO()):
        r = ops_nv.NVDevice._new_gpu_fifo(dev, gp, 7, 8, offset=0x100000, entries=0x10000, compute=False)
    assert r == ("fifo", (gp, 7, 8), {"offset": 0x100000, "entries": 0x10000, "compute": False, "video": False}), r
    assert dev.beagle_userd == {"copy": (0x1234, 0x5678)} and len(reads) == 2, (dev.beagle_userd, reads)
    print("USERD baseline: GPGet/GPPut read through tinygrad's USERD layout (2 reads per GPFIFO), pass-through result")

def test_vbios_capture():
    tmp = tempfile.mkdtemp(); os.environ["BEAGLE_TINYGPU_DATA"] = tmp
    dev = fake_nvdev(); inner = dev.mmio
    fl = NV_FLCN.__new__(NV_FLCN); fl.nvdev = dev
    def prep(self): self.first = self.nvdev.mmio[h._VBIOS_SLICE][:2]; self.nvdev.mmio[0x10 // 4] = 5
    with orig("prep_ucode", prep), contextlib.redirect_stderr(io.StringIO()): fl.prep_ucode()
    expect = struct.pack(f"<{0x100000 // 4}I", *[0x55AA0000 + k for k in range(0x100000 // 4)])
    assert dev.mmio is inner and fl.first == [0x55AA0000, 0x55AA0001] and dev.beagle_vbios == expect
    assert ("wr", 0x10, 5) in inner.log and [x for x in inner.log if x[0] == "rd"] == [("rd", 0x300000, 0x100000)]
    saved = os.listdir(os.path.join(tmp, "vbios"))
    assert saved == [f"AD107_{hashlib.sha256(expect).hexdigest()[:16]}.rom"], saved
    def prep_fail(self): self.nvdev.mmio[h._VBIOS_SLICE]; raise RuntimeError("bad VBIOS")
    dev2 = fake_nvdev(); inner2 = dev2.mmio; fl.nvdev = dev2
    with orig("prep_ucode", prep_fail):
        try: fl.prep_ucode(); raise AssertionError("exception swallowed")
        except RuntimeError: pass
    assert dev2.mmio is inner2 and not hasattr(dev2, "beagle_vbios")
    print("VBIOS capture: transparent (same reads and writes), restored on exceptions, saved by sha256")

def test_cmd_fini():
    import nv_dispatch_daemon as d
    from tinygrad import Device
    real_opened = Device._opened_devices
    Device._opened_devices = set()   # never the real registry: tinygrad's atexit hook finalizes (and so opens) whatever is in it
    try: _test_cmd_fini(d, Device)
    finally: Device._opened_devices = real_opened

def _test_cmd_fini(d, Device):
    def run(finalize):
        a, b = socket.socketpair()
        dm = d.Daemon(b); held = []
        dm._hold = lambda: held.append(True)
        dm.dev = types.SimpleNamespace(iface=types.SimpleNamespace(dev_impl=types.SimpleNamespace()), synchronize=lambda: None)
        dm.dev.finalize = lambda: finalize(dm.dev)
        Device._opened_devices.add("NV")
        dm.cmd_fini({})
        n = struct.unpack("<I", a.recv(4, socket.MSG_WAITALL))[0]
        return json.loads(a.recv(n, socket.MSG_WAITALL)), held
    def clean(dev): dev.iface.dev_impl.beagle_fini = {"unload_ok": True, "mailbox0": 0x80000000, "riscv_cpuctl": 1, "wpr2_lo": 2, "wpr2_hi": 3}
    r, held = run(clean)
    assert r["ok"] and r["unload_ok"] and r["mailbox0"] == 0x80000000 and not r.get("hold") and not held and "NV" not in Device._opened_devices, r
    r, held = run(lambda dev: setattr(dev.iface.dev_impl, "beagle_fini", {"unload_ok": False, "mailbox0": 0}))
    assert r.get("hold") and r["pid"] == os.getpid() and held == [True], r
    def fails(dev): raise RuntimeError("Timeout waiting for RPC response")
    with contextlib.redirect_stderr(io.StringIO()): r, held = run(fails)
    assert not r["ok"] and not r["unload_ok"] and r.get("hold") and held == [True] and "teardown failed" in r["error"], r
    Device._opened_devices.clear()
    print("cmd_fini: finalizes before replying, drops NV from atexit, replies the diagnostics; holds when the unload is not confirmed")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--boot-scenario":
        boot_scenario(int(sys.argv[2]), bool(int(sys.argv[3]))); sys.exit(0)
    for t in (test_warm_refusal, test_cold_path_adds_one_read, test_suspend_wait, test_event_logging, test_frts_checks,
              test_userd_baseline, test_vbios_capture, test_cmd_fini):
        t()
    print("P1 diagnostics: all passed")
