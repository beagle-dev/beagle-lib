"""Offline tests for plan step B1 (Blackwell, the COT boot; nv_init_helper.py section 6 and the daemon's hold decisions).
No GPU: the boot scenarios run the real daemon's cmd_boot, i.e. tinygrad's real NVDevice boot path, for a GB205 (RTX 5070,
BOOT_42 0x1b5a1000) against a scripted fake TinyGPU.app on a socketpair, each in its own subprocess; the rest drive each
wrapper on register fakes.
    python test_b1_cot.py"""
import os, sys, io, json, time, struct, socket, threading, subprocess, types, contextlib, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tgpaths
tgpaths.setup()
from test_p1_diagnostics import FakeMMIO, orig, CMD, WPR2_HI, BOOT_0, BOOT_42   # the P1 rig: register fake, _ORIG swaps
import nv_init_helper as h
from tinygrad.runtime.support.nv.nvdev import NVDev
from tinygrad.runtime.support.nv import ip
from tinygrad.runtime.support.nv.ip import NV_GSP, NV_FLCN, NV_FLCN_COT, NV_IP
from tinygrad.runtime.autogen import nv, nv_regs

GB205_BOOT_0, GB205_BOOT_42, AD107_BOOT_42 = 0x1b5000a1, 0x1b5a1000, 0x19700000
I2CS, VRAM_MB_REG, WPR2_LO = 0xAD00BC, 0x1183A4, 0x1FA824
MBX0, CPUCTL = 0x110040, 0x111388   # GSP falcon MAILBOX0; NV_PRISCV_RISCV_CPUCTL at the GSP's RISC-V (0x110000 + 0x1000 + 0x388)
FSP_QUEUES = (0x8f2c00, 0x8f2c04, 0x8f2c80, 0x8f2c84)   # NV_PFSP_QUEUE_HEAD/TAIL[0], NV_PFSP_MSGQ_HEAD/TAIL[0] (dev_fsp_pri gh100)
# the include calls a GB205 boot makes, in tinygrad's order (nvdev.py:101-129; ip.py:286, 291-295)
COT_INCLUDES = (("nv_ref", ""), ("dev_fb", "tu102"), ("dev_gc6_island", "ga102"), ("dev_therm", "gb202"), ("dev_vm", "tu102"),
                ("dev_mmu", "gh100"), ("dev_gsp", "ga102"), ("dev_falcon_v4", "gh100"), ("dev_vm", "gh100"), ("dev_fsp_pri", "gh100"),
                ("dev_bus", "tu102"))

def fake_cot_nvdev(script=None):
    dev = NVDev.__new__(NVDev)
    dev.mmio, dev.chip_name, dev.fmc_boot, dev.chip_id = FakeMMIO(script), "GB205", True, GB205_BOOT_0
    for name, arch in (("dev_riscv_pri", "ga102"),) + COT_INCLUDES: dev.include(name, arch)   # section 6's include comes first
    return dev

def cot_flcn(dev): fl = NV_FLCN_COT.__new__(NV_FLCN_COT); fl.nvdev = dev; return fl
def quiet(): return contextlib.redirect_stderr(io.StringIO())

# ── the pin: what section 6 relies on in tinygrad a9830e2b4 ───────────────────
def test_pin():
    """tinygrad's COT code as section 6 was written against it: checked in a fresh interpreter without nv_init_helper."""
    code = f"""import sys; sys.path.insert(0, {str(tgpaths.HERE)!r}); import tgpaths; tgpaths.setup(); import inspect
from tinygrad.runtime.support.nv import ip, nvdev
C = ip.NV_FLCN_COT
assert 'fini_hw' not in C.__dict__ and C.fini_hw is ip.NV_IP.fini_hw, 'NV_FLCN_COT gained a fini_hw'
assert not issubclass(C, ip.NV_FLCN) and all(not hasattr(C, m) for m in ('reset', 'start_cpu', 'wait_cpu_halted', 'execute_hs', 'prep_booter', 'prep_ucode', 'disable_ctx_req'))
assert '__init__' not in C.__dict__, 'NV_FLCN_COT gained an __init__'
w = inspect.getsource(C.wait_for_reset); assert 'NV_THERM_I2CS_SCRATCH' in w and '0xff' in w and 'write' not in w, w
s = inspect.getsource(ip.NV_GSP); assert s.count('self.stat_q =') == 1 and 'self.stat_q = NVRpcQueue' in inspect.getsource(ip.NV_GSP.init_hw)
k = inspect.getsource(C.kfsp_send_msg); assert k.index('NV_PFSP_EMEMC') < k.index('NV_PFSP_QUEUE_TAIL'), k
i = inspect.getsource(nvdev.NVDev.__init__)
assert i.index('_early_ip_init') < i.index('_early_mmu_init') < i.index('ip.init_sw()') < i.index('ip.init_hw()'), i
f = inspect.getsource(nvdev.NVDev.fini); assert 'self.gsp, self.flcn' in f, f
e = inspect.getsource(nvdev.NVDev._early_ip_init); assert e.index('NV_PMC_BOOT_42') < e.index('NV_FLCN_COT(self)') < e.index('wait_for_reset'), e
m = inspect.getsource(nvdev.NVDev._early_mmu_init); assert m.index('large_bar') < m.index('NVMemoryManager('), m
h = inspect.getsource(C.init_hw); j = h.index('self.kfsp_send_msg(nv.NVDM_TYPE_COT')
assert h.count('kfsp_send_msg') == 1 and 'NV_PFSP' not in h and '.write(' not in h[:j], h
assert all('NV_PFSP' not in inspect.getsource(v) for n, v in C.__dict__.items() if callable(v) and n != 'kfsp_send_msg'), 'another method touches the FSP'
print('pin ok')"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**os.environ, "BEAGLE_TINYGPU_NO_LAUNCH": "1"})
    assert r.stdout.split() == ["pin", "ok"], (r.stdout, r.stderr[-2000:])
    print("pin: NV_FLCN_COT has no fini_hw, __init__ or NV_FLCN primitives; its wait_for_reset only polls I2CS; stat_q is made in "
          "gsp.init_hw; init_hw reaches the FSP only through self.kfsp_send_msg; NVDev runs early init, init_sw, init_hw in order "
          "and finalizes gsp then flcn")

def test_riscv_include():
    """Section 6's init_sw wrapper, not the fakes, supplies NV_PRISCV_RISCV_CPUCTL on COT (tinygrad's includes lack it), and
    before tinygrad's own init_sw runs, so a boot that fails in it has the register too."""
    dev = NVDev.__new__(NVDev); dev.mmio, dev.chip_name, dev.fmc_boot = FakeMMIO(), "GB205", True
    for n, a in COT_INCLUDES: dev.include(n, a)
    assert "NV_PRISCV_RISCV_CPUCTL" not in dev.__dict__
    seen = []
    with orig("cot_init_sw", lambda self: seen.append("NV_PRISCV_RISCV_CPUCTL" in self.nvdev.__dict__)): cot_flcn(dev).init_sw()
    r = dev.NV_PRISCV_RISCV_CPUCTL.with_base(h._GSP_BASE)
    assert seen == [True] and r.base + r.off == CPUCTL and dev.mmio.log == [], (seen, hex(r.base + r.off))
    print("riscv include: NV_FLCN_COT.init_sw adds NV_PRISCV_RISCV_CPUCTL before tinygrad's init_sw runs; no GPU access")

def test_registers():
    """dev_riscv_pri ga102 adds no key a GB205 boot already has (NVDev.include overwrites silently), and the addresses
    section 6 reads are the GB20x ones."""
    tables = [set(getattr(getattr(nv_regs, n), a or "regs")) for n, a in COT_INCLUDES]
    riscv = set(nv_regs.dev_riscv_pri.ga102)
    assert all(not (riscv & t) for t in tables), [sorted(riscv & t) for t in tables]
    dev = fake_cot_nvdev()
    cpuctl = dev.NV_PRISCV_RISCV_CPUCTL.with_base(h._GSP_BASE)
    assert cpuctl.base + cpuctl.off == CPUCTL and cpuctl.fields["halted"] == (4, 4), (hex(cpuctl.base + cpuctl.off), cpuctl.fields)
    assert dev.NV_PGSP_FALCON_MAILBOX0.base + dev.NV_PGSP_FALCON_MAILBOX0.off == MBX0
    assert dev.NV_PFALCON_FALCON_HWCFG2.off == 0xf4 and dev.NV_PFALCON_FALCON_HWCFG2.fields["riscv_br_priv_lockdown"] == (13, 13)
    got = [dev.__dict__[f"NV_PFSP_{r}"][0] for r in ("QUEUE_HEAD", "QUEUE_TAIL", "MSGQ_HEAD", "MSGQ_TAIL")]
    assert tuple(x.base + x.off for x in got) == FSP_QUEUES, [hex(x.base + x.off) for x in got]
    assert dev.NV_THERM_I2CS_SCRATCH.base + dev.NV_THERM_I2CS_SCRATCH.off == I2CS
    b42 = dev.NV_PMC_BOOT_42.decode(GB205_BOOT_42)
    assert (b42["architecture"], b42["implementation"]) == (0x1b, 5) and dev.NV_PMC_BOOT_42.decode(AD107_BOOT_42)["architecture"] == 0x19
    print(f"registers: dev_riscv_pri ga102 ({len(riscv)} keys) collides with none of the GB205 include set; CPUCTL 0x{CPUCTL:x} "
          f"(halted bit 4), MAILBOX0 0x{MBX0:x}, FSP queues 0x8f2c00-84, I2CS 0x{I2CS:x}; BOOT_42 0x{GB205_BOOT_42:x} decodes to GB205")

# ── the halt wait (NV_FLCN_COT.fini_hw) ───────────────────────────────────────
def halt_wait(script, fini=None, timeout=None):
    dev = fake_cot_nvdev(script)
    dev.beagle_fini = dict(fini or {"unload_ok": True, "mailbox0": 0x80000000})
    saved = h._COT_HALT_TIMEOUT_S
    if timeout is not None: h._COT_HALT_TIMEOUT_S = timeout
    try:
        with quiet(): cot_flcn(dev).fini_hw()
    finally: h._COT_HALT_TIMEOUT_S = saved
    return dev, dev.beagle_fini

def test_halt_wait():
    dev, d = halt_wait({CPUCTL: [0, 0, 0x10], WPR2_HI: 0, WPR2_LO: 0x7ffffe00, MBX0: 0x80000000})
    t = d["teardown"]
    assert d["halted"] is True and d["teardown_ok"] is True and d["riscv_cpuctl"] == 0x10 and d["wpr2_down"], d
    assert t["result"].startswith("done: GSP RISC-V halted") and t["polls"] == 3, t
    assert dev.mmio.writes() == [] and dev.mmio.log == [("rd", CPUCTL)] * 3 + [("rd", MBX0), ("rd", WPR2_LO), ("rd", WPR2_HI)], dev.mmio.log
    # halted, WPR2 still up: closing is safe, the next boot needs a power cycle
    dev, d = halt_wait({CPUCTL: 0x10, WPR2_HI: 0x1ff10})
    assert d["halted"] is True and d["teardown_ok"] is False and not d["wpr2_down"] and dev.mmio.writes() == [], d
    # never halts: halted false (the daemon holds), still reads only
    dev, d = halt_wait({CPUCTL: 0x0}, timeout=0.05)
    assert d["halted"] is False and d["teardown_ok"] is False and d["teardown"]["result"].startswith("failed: GSP RISC-V did not halt"), d
    assert dev.mmio.writes() == [] and dev.mmio.log.count(("rd", CPUCTL)) > 3, len(dev.mmio.log)
    # a PRI error (0xbadf....) has bit 4 set here, and is never taken for a halt
    dev, d = halt_wait({CPUCTL: 0xbadf4110}, timeout=0.05)
    assert d["halted"] is False and d["riscv_cpuctl"] == 0xbadf4110 and dev.mmio.writes() == [], d
    dev, d = halt_wait({CPUCTL: 0xffffffff}, timeout=0.05)   # an unreachable GPU reads all ones: not a halt either
    assert d["halted"] is False and d["riscv_cpuctl"] == 0xffffffff and dev.mmio.writes() == [], d
    dev, d = halt_wait({CPUCTL: [0xffffffff, 0xbadf4110, 0x10]})   # glitches, then a real halt: halted
    assert d["halted"] is True and d["teardown"]["polls"] == 3, d
    # the unload was not confirmed: nothing is polled (the daemon holds anyway)
    dev, d = halt_wait({CPUCTL: 0x10}, fini={"unload_ok": False, "mailbox0": 0})
    assert d["halted"] is False and d["teardown"]["result"].startswith("skipped") and dev.mmio.log == [], (d, dev.mmio.log)
    # BEAGLE_NV_TEARDOWN=0 does not turn it off: on this chip it is the unload's own last step
    saved, h._TEARDOWN = h._TEARDOWN, False
    try: dev, d = halt_wait({CPUCTL: 0x10})
    finally: h._TEARDOWN = saved
    assert d["halted"] is True and dev.mmio.log.count(("rd", CPUCTL)) == 1, d
    # no unload attempted (no beagle_fini): nothing
    dev = fake_cot_nvdev({CPUCTL: 0x10})
    with quiet(): cot_flcn(dev).fini_hw()
    assert dev.mmio.log == [] and not hasattr(dev, "beagle_fini")
    print("halt wait: polls CPUCTL until HALTED, then reads MAILBOX0 and WPR2 once; teardown_ok = halted and WPR2 down; a timeout or "
          "a PRI error or an all-ones read is not a halt; skipped with no read when the unload was not confirmed; runs with BEAGLE_NV_TEARDOWN=0; "
          "no writes in any case")

def test_unload_then_halt():
    """NVDev.fini's order on a GB205 (gsp.fini_hw, then flcn.fini_hw) through the real wrappers: the RPC (stubbed), the suspend
    wait, one CPUCTL read in its diagnostics, then the halt wait; the hung path (gsp.fini_hw alone) reads CPUCTL once, no poll."""
    dev = fake_cot_nvdev({MBX0: [0, 0x80000000], CPUCTL: [0, 0x10], WPR2_HI: 0})
    gsp = NV_GSP.__new__(NV_GSP); gsp.nvdev = dev
    calls = []
    with orig("gsp_fini_hw", lambda self: calls.append("rpc")), quiet():
        gsp.fini_hw(); n_suspend = len(dev.mmio.log)
        cot_flcn(dev).fini_hw()
    d = dev.beagle_fini
    assert calls == ["rpc"] and d["unload_ok"] and d["halted"] and d["teardown_ok"] and dev.mmio.writes() == [], d
    assert dev.mmio.log[:n_suspend] == [("rd", MBX0), ("rd", MBX0), ("rd", WPR2_LO), ("rd", WPR2_HI), ("rd", CPUCTL)], dev.mmio.log
    assert dev.mmio.log[n_suspend:] == [("rd", CPUCTL), ("rd", MBX0), ("rd", WPR2_LO), ("rd", WPR2_HI)], dev.mmio.log   # the 2nd read of 0x10
    hung = fake_cot_nvdev({MBX0: 0x80000000, CPUCTL: 0})
    gsp.nvdev = hung
    with orig("gsp_fini_hw", lambda self: None), quiet(): gsp.fini_hw()
    assert hung.mmio.log.count(("rd", CPUCTL)) == 1 and hung.beagle_fini["halted"] is False and hung.mmio.writes() == [], hung.mmio.log
    print("unload then halt: RPC, MAILBOX0 until suspended, WPR2, CPUCTL, then the halt wait's polls and one re-read of MAILBOX0 and "
          "WPR2; the hung path adds one CPUCTL read and no poll, and reports halted false")

class RaisingMMIO(FakeMMIO):
    """FakeMMIO whose reads of one address fail as a TinyGPU.app error reply does (RemotePCIDevice._rpc raises RuntimeError)."""
    def __init__(self, script, bad): super().__init__(script); self.bad = bad
    def __getitem__(self, i):
        if not isinstance(i, slice) and i * 4 == self.bad: raise RuntimeError("RPC failed: fake MMIO read error")
        return super().__getitem__(i)

def test_not_halted_by_default():
    """On COT the unload's report says halted false from its first line on, so every way out before the halt wait holds:
    the suspend wait's own CPUCTL read failing (NVDev.fini then never reaches flcn.fini_hw), and the failed-boot unload. Ada's
    report gains no key."""
    dev = fake_cot_nvdev({MBX0: 0x80000000}); dev.mmio = RaisingMMIO({MBX0: 0x80000000}, CPUCTL)
    gsp = NV_GSP.__new__(NV_GSP); gsp.nvdev = dev
    with orig("gsp_fini_hw", lambda self: None), quiet():
        try: gsp.fini_hw(); raise AssertionError("the failed read was swallowed")
        except RuntimeError: pass
    assert dev.beagle_fini == {"unload_ok": True, "halted": False, "mailbox0": 0x80000000, "wpr2_lo": 0, "wpr2_hi": 0}, dev.beagle_fini
    ada = NVDev.__new__(NVDev); ada.mmio = FakeMMIO({MBX0: 0x80000000})
    for n, a in (("nv_ref", ""), ("dev_fb", "tu102"), ("dev_gsp", "ga102")): ada.include(n, a)
    gsp.nvdev = ada
    with orig("gsp_fini_hw", lambda self: None), quiet(): gsp.fini_hw()
    assert "halted" not in ada.beagle_fini and ada.beagle_fini["unload_ok"], ada.beagle_fini
    saved = h._BOOTING[0]   # the failed-boot unload over the same failing read
    try:
        booting = fake_cot_nvdev(); booting.mmio = RaisingMMIO({MBX0: 0x80000000}, CPUCTL); booting.beagle_gsp_started = True
        booting.gsp = NV_GSP.__new__(NV_GSP); booting.gsp.nvdev = booting; booting.gsp.stat_q = object(); booting.flcn = cot_flcn(booting)
        h._BOOTING[0] = booting
        with orig("gsp_fini_hw", lambda self: None), quiet(): r = h.unload_after_failed_boot()
        assert r["unload_ok"] and r["halted"] is False, r
    finally: h._BOOTING[0] = saved
    print("not halted by default: on COT the report starts halted false, so a failed CPUCTL read in the suspend wait holds, and "
          "so does the failed-boot unload; Ada's report is unchanged")

# ── the COT message, the failed-boot unload, the sequencer, the script refusal ─
def test_cot_message_flag():
    dev = fake_cot_nvdev({FSP_QUEUES[0]: 0x10, FSP_QUEUES[1]: 0x10})
    seen = []
    err = io.StringIO()
    with orig("kfsp_send_msg", lambda self, nvmd, buf: seen.append((nvmd, getattr(self.nvdev, "beagle_gsp_started", False)))), \
         contextlib.redirect_stderr(err):
        cot_flcn(dev).kfsp_send_msg(nv.NVDM_TYPE_COT, b"payload")
    assert seen == [(nv.NVDM_TYPE_COT, True)] and dev.mmio.log == [("rd", a) for a in FSP_QUEUES] and dev.mmio.writes() == [], (seen, dev.mmio.log)
    assert "both empty" in err.getvalue(), err.getvalue()
    stale = fake_cot_nvdev({FSP_QUEUES[2]: 0x40})
    with orig("kfsp_send_msg", lambda self, nvmd, buf: None), quiet() as e2: cot_flcn(stale).kfsp_send_msg(nv.NVDM_TYPE_COT, b"x")
    assert "NOT EMPTY" in e2.getvalue()
    other = fake_cot_nvdev()
    with orig("kfsp_send_msg", lambda self, nvmd, buf: seen.append(nvmd)): cot_flcn(other).kfsp_send_msg(nv.NVDM_TYPE_COT + 1, b"x")
    assert not getattr(other, "beagle_gsp_started", False) and other.mmio.log == []
    print("COT message: GSP-RM counts as started before tinygrad's first FSP write; the FSP queues are read (4 reads) and logged "
          "first; other FSP messages are passed through untouched")

def test_failed_boot_unload():
    def booting(started, stat_q):
        dev = fake_cot_nvdev({MBX0: 0x80000000, CPUCTL: 0x10})
        if started: dev.beagle_gsp_started = True
        dev.gsp = NV_GSP.__new__(NV_GSP); dev.gsp.nvdev = dev
        if stat_q: dev.gsp.stat_q = object()
        dev.flcn = cot_flcn(dev)
        return dev
    rpcs, saved = [], h._BOOTING[0]
    try:
        h._BOOTING[0] = booting(False, False)
        assert h.unload_after_failed_boot() is None   # before the COT message: GSP-RM never started, closing is safe
        h._BOOTING[0] = dev = booting(True, False)
        with orig("gsp_fini_hw", lambda self: rpcs.append("rpc")), quiet(): r = h.unload_after_failed_boot()
        assert r == {"unload_ok": False, "halted": False} and rpcs == [] and dev.mmio.log == [], (r, rpcs)   # no queue: no RPC, the daemon holds
        h._BOOTING[0] = dev = booting(True, True)
        with orig("gsp_fini_hw", lambda self: rpcs.append("rpc")), quiet(): r = h.unload_after_failed_boot()
        assert rpcs == ["rpc"] and r["unload_ok"] and r["halted"] and r["teardown_ok"] and dev.mmio.writes() == [], r
        h._BOOTING[0] = dev = booting(True, True); dev.mmio.script[CPUCTL] = 0
        saved_t, h._COT_HALT_TIMEOUT_S = h._COT_HALT_TIMEOUT_S, 0.05
        try:
            with orig("gsp_fini_hw", lambda self: None), quiet(): r = h.unload_after_failed_boot()
        finally: h._COT_HALT_TIMEOUT_S = saved_t
        assert r["unload_ok"] and r["halted"] is False, r
        h._BOOTING[0] = dev = booting(True, True); dev.beagle_seq_refused = [5]   # a refused sequencer heads the status queue
        with orig("gsp_fini_hw", lambda self: rpcs.append("rpc2")), quiet(): r = h.unload_after_failed_boot()
        assert r == {"unload_ok": False, "halted": False, "seq_refused": [5]} and "rpc2" not in rpcs and dev.mmio.log == [], r
    finally: h._BOOTING[0] = saved
    print("failed COT boot: before the COT message, None (close); after it with no status queue, unload_ok false and no RPC "
          "(hold); with one, the RPC, the suspend wait and the halt wait; a missing halt reports halted false; behind a refused "
          "sequencer, no RPC (hold)")

def test_compile_ptx_raises():
    """A failing ptxas raises RuntimeError from compile_ptx, not SystemExit, which would skip the daemon's hold decision."""
    import nv_compile_helper as nch
    tmp = tempfile.mkdtemp(); ptx = os.path.join(tmp, "k.ptx"); open(ptx, "w").write(".version 8.7\n.target sm_120\n")
    saved = (nch._ptxas_path, nch._use_nvjitlink, nch.os.path.expanduser)
    nch._ptxas_path, nch._use_nvjitlink = (lambda: "/usr/bin/false"), (lambda: False)
    nch.os.path.expanduser = lambda p: p.replace("~", tmp)   # the failing PTX copy goes under tmp, not ~/Library/Logs
    try:
        with quiet():
            try: nch.compile_ptx(ptx, "sm_120", kernel_name="_all"); raise AssertionError("a failing ptxas returned")
            except RuntimeError as e: assert "ptxas exited 1" in str(e), e
            except BaseException as e: raise AssertionError(f"{type(e).__name__} instead of RuntimeError") from e
    finally: nch._ptxas_path, nch._use_nvjitlink, nch.os.path.expanduser = saved
    print("compile_ptx: a failing ptxas raises RuntimeError (the daemon replies an error and decides the hold), not SystemExit")

def test_level0_logged():
    """BEAGLE_NV_UNLOAD_LEVEL=0 is said in the daemon's log (which the hardware scripts keep), at import, with no GPU access."""
    code = f"import sys; sys.path.insert(0, {str(tgpaths.HERE)!r}); import tgpaths; tgpaths.setup(); import nv_init_helper"
    for val, want in (("0", True), ("", False)):
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env={**os.environ, "BEAGLE_TINYGPU_NO_LAUNCH": "1", "BEAGLE_NV_UNLOAD_LEVEL": val})
        assert ("the unload RPC is LEVEL_0" in r.stderr) is want and r.returncode == 0, (val, r.stderr[-1000:])
    print("LEVEL_0: announced in the daemon log at import; nothing said otherwise")

def test_sequencer_guard():
    ops = [0x0, 0x100, 7, 0x6]
    buf = bytes(nv.rpc_run_cpu_sequencer_v17_00(cmdIndex=len(ops))) + struct.pack(f"<{len(ops)}I", *ops)
    dev = fake_cot_nvdev(); gsp = NV_GSP.__new__(NV_GSP); gsp.nvdev = dev
    calls = []
    with orig("run_cpu_seq", lambda self, b: calls.append(b)), quiet():
        try: gsp.run_cpu_seq(buf); raise AssertionError("ops 5-8 were run on COT")
        except RuntimeError as e: assert "[6]" in str(e) and "GB205" in str(e), e
    assert calls == [] and dev.mmio.log == [] and dev.beagle_seq_refused == [6]
    with orig("run_cpu_seq", lambda self, b: calls.append(b)), quiet():   # ops 0-4 only: tinygrad's run_cpu_seq, as before
        ok = bytes(nv.rpc_run_cpu_sequencer_v17_00(cmdIndex=3)) + struct.pack("<3I", 0x0, 0x100, 7); gsp.run_cpu_seq(ok)
    ada = types.SimpleNamespace(chip_name="AD107")   # no fmc_boot: every op reaches tinygrad, as before
    gsp.nvdev = ada
    with orig("run_cpu_seq", lambda self, b: calls.append(b)), quiet(): gsp.run_cpu_seq(buf)
    assert calls == [ok, buf], calls
    print("sequencer: on COT, ops 5-8 (NV_FLCN's falcon primitives) are refused before any op runs; ops 0-4 and every Ada sequence "
          "reach tinygrad unchanged")

def test_script_refusal():
    dev = fake_cot_nvdev()
    NV_FLCN_COT(dev)   # the daemon path: allowed
    saved = h._REFUSE_FMC_BOOT[0]
    h.refuse_fmc_boot("nv_teardown_diag.py")
    try:
        try: NV_FLCN_COT(dev); raise AssertionError("COT allowed for a script without a hold")
        except RuntimeError as e: assert "nv_teardown_diag.py" in str(e) and "GB205" in str(e), e
        fl = NV_FLCN.__new__(NV_FLCN); NV_FLCN.__init__(fl, dev)   # Ada's falcon: unaffected
    finally: h._REFUSE_FMC_BOOT[0] = saved
    assert dev.mmio.log == []
    print("script refusal: a tinygrad-only boot script refuses a COT chip when NV_FLCN_COT is made (after tinygrad's chip-id reads, "
          "before boot memory); the daemon path and Ada are unaffected; no GPU access (the four such probes went in plan step C13a)")

def test_bar_check():
    def mmu(nbytes, vram_mb, large):
        dev = NVDev.__new__(NVDev); dev.mmio = FakeMMIO()
        def fake(self): self.vram, self.vram_size, self.large_bar = types.SimpleNamespace(nbytes=nbytes), vram_mb << 20, large
        with orig("early_mmu_init", fake): dev._early_mmu_init()
        return dev
    assert mmu(256 << 20, 12227, False).mmio.log == []
    assert mmu(256 << 20, 8188, False).mmio.log == []
    for nbytes, vram, large in ((512 << 20, 12227, False), (16 << 30, 12227, True), (256 << 20, 256, True)):
        try: mmu(nbytes, vram, large); raise AssertionError(f"BAR1 {nbytes >> 20} MiB accepted")
        except h.BarLayoutError as e: assert f"BAR1 is {nbytes >> 20} MiB" in str(e), e
    print("BAR check: a 256 MiB BAR1 below VRAM passes (GB205 and AD107), a larger BAR1 or a large-BAR layout is refused; it reads "
          "nothing")

def test_layout_gate():
    """tinygrad's memory manager for BEAGLE's allocation sequence on the RTX 5070 (MMU v3, 12227 MiB; mm_trace.py, P2's gate):
    the GPFIFO area BEAGLE's handoff writes through BAR1 ends below 256 MiB, no BAR1 write is dropped, and the default pool
    ends below the daemon's COT bound (vram_size - 512 MiB). The COT boot's images are sysmem, so none is palloc'd first."""
    import mm_trace
    r = mm_trace.trace(3, quiet=True, vram_mb=12227)
    assert r["vram_size"] == 12227 << 20 and r["gpfifo_paddr"] + r["gpfifo_size"] <= 256 << 20, {k: hex(v) for k, v in r.items() if k != "stats"}
    assert not any(k[1] == "bar1_wr_dropped_beyond_256MiB" for k in r["stats"]), r["stats"]
    assert r["pool_paddr_end"] <= r["vram_size"] - (512 << 20), (hex(r["pool_paddr_end"]), hex(r["vram_size"] - (512 << 20)))
    print(f"layout gate: on 12227 MiB (MMU v3) the GPFIFO area ends at 0x{r['gpfifo_paddr'] + r['gpfifo_size']:x} (< 256 MiB of BAR1), "
          f"no BAR1 write dropped, the default pool ends at 0x{r['pool_paddr_end']:x} (< vram_size - 512 MiB)")

# ── tinygrad's real boot path on a scripted GB205 (subprocess scenarios) ───────
def boot_scenario(kind):
    """Daemon.cmd_boot over a scripted fake TinyGPU.app, as test_p1's warm refusal: a GB205 by default, AD107 for 'ada*'.
    Prints the request stream and the reply as JSON."""
    import nv_dispatch_daemon as d
    from tinygrad.runtime.support import system
    gb = not kind.startswith("ada")
    regs = {WPR2_HI: 0, BOOT_0: GB205_BOOT_0 if gb else 0x197000a1, BOOT_42: GB205_BOOT_42 if gb else AD107_BOOT_42,
            I2CS: 0x0 if kind == "fsp_not_ready" else 0xff, VRAM_MB_REG: 12227 if gb else 8188}
    bars = {0: (64 if gb else 16) << 20, 1: (512 if kind == "bar512" else 256) << 20, 3: 32 << 20}
    reqs, fake, plugin = [], *socket.socketpair()
    def recv(n):
        b = b""
        while len(b) < n:
            c = fake.recv(n - len(b))
            if not c: return None
            b += c
        return b
    def serve():
        while (hd := recv(33)) is not None:
            cmd, dev, bar, a0, a1, a2 = struct.unpack("<BIIQQQ", hd)
            if cmd == 7: recv(a1); reqs.append(("MMIO_WRITE", bar, a0, a1)); continue          # posted, no reply
            req = (CMD.get(cmd, cmd), bar, a0, a1, a2) if cmd in (3, 4) else (CMD.get(cmd, cmd), bar, a0, a1)
            if not (req == ("MMIO_READ", 0, I2CS, 4) and reqs and reqs[-1] == req): reqs.append(req)   # tinygrad's poll: one entry
            if cmd == 6 and a1 > 4: m = b"fake: no VBIOS"; fake.sendall(struct.pack("<BQQ", 1, len(m), 0) + m)   # Ada's prep_ucode stops here
            elif cmd == 1: fake.sendall(struct.pack("<BQQ", 0, 0x10_0000_0000 + (bar << 32), bars.get(bar, 1 << 20)))
            elif cmd == 11 or cmd == 4: fake.sendall(struct.pack("<BQQ", 0, 0, 0))
            elif cmd == 3: fake.sendall(struct.pack("<BQQ", 0, 0x0006 if a0 == 4 else 0, 0))
            elif cmd == 6: fake.sendall(struct.pack("<BQQ", 0, 0, 0) + struct.pack("<I", regs.get(a0, 0)) * (a1 // 4))
            else: m = b"fake: not scripted"; fake.sendall(struct.pack("<BQQ", 1, len(m), 0) + m)   # e.g. MAP_SYSMEM_FD: the boot stops
    threading.Thread(target=serve, daemon=True).start()
    type(system.System).list_devices = lambda self, vendor, devices, base_class=None: [(system.APLRemotePCIDevice, "usb4")]
    import nv_init_helper
    if kind == "ada_unwrapped": nv_init_helper.NVDev._early_mmu_init = nv_init_helper._ORIG["early_mmu_init"]
    cmd_a, cmd_b = socket.socketpair()
    dm = d.Daemon(cmd_b, plugin.fileno())
    try: dm.cmd_boot({})
    except Exception as e: dm.send_json({"ok": False, "error": f"{type(e).__name__}: {e}", "raised": True})
    n = struct.unpack("<I", cmd_a.recv(4, socket.MSG_WAITALL))[0]
    reply = json.loads(cmd_a.recv(n, socket.MSG_WAITALL))
    # posted writes have no reply: a marked request behind them, once recorded, means every earlier byte was
    flush = ("CFG_READ", 0, 0x7f0, 4, 0)
    plugin.sendall(struct.pack("<BIIQQQ", 3, 0, 0, 0x7f0, 4, 0))
    for _ in range(500):
        if flush in reqs: break
        time.sleep(0.01)
    reqs.remove(flush)
    print("RESULT " + json.dumps({"reqs": reqs, "reply": reply}))

def run_scenario(kind):
    r = subprocess.run([sys.executable, __file__, "--boot-scenario", kind], capture_output=True, text=True, timeout=180,
                       env={**os.environ, "BEAGLE_TINYGPU_NO_LAUNCH": "1"})
    line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT ")), None)
    assert line, f"scenario {kind} produced no result:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
    res = json.loads(line[7:])
    return [tuple(x) for x in res["reqs"]], res["reply"], r.stderr

def test_gb205_boot():
    reqs, reply, log = run_scenario("gb205")
    first = lambda req: reqs.index(req)
    rd = lambda a: ("MMIO_READ", 0, a, 4)
    assert reqs[:3] == [("RESIZE_BAR", 1, 0, 0), ("MAP_BAR", 0, 0, 0), rd(WPR2_HI)], reqs[:4]
    bm = next(i for i, r in enumerate(reqs) if r[0] == "CFG_WRITE")
    # tinygrad's order: bus master, BOOT_0/BOOT_42, the FSP readiness wait (reads only), then the MMU init (BAR1)
    assert bm < first(rd(BOOT_0)) < first(rd(BOOT_42)) < first(rd(I2CS)) < first(("MAP_BAR", 1, 0, 0)), reqs
    assert reqs.count(rd(I2CS)) == 1 and "FSP ready: NV_THERM_I2CS_SCRATCH == 0xff" in log, log[-2000:]
    assert ("RESET", 0, 0, 0) not in reqs and any(r[0] == "MAP_SYSMEM_FD" for r in reqs), reqs   # past the BAR check, into boot memory
    assert all(r[2] not in FSP_QUEUES for r in reqs if r[0] == "MMIO_READ") and reply["ok"] is False and not reply.get("warm"), reply
    print(f"GB205 boot: tinygrad's COT path runs the restored FSP readiness wait (one I2CS read at 0xff) after the chip-id reads and "
          f"before the MMU init, passes the BAR check (256 MiB of 12227 MiB), and stops at the fake's missing sysmem ({len(reqs)} requests)")

def test_gb205_refusals():
    reqs, reply, log = run_scenario("bar512")
    assert "BarLayoutError" in reply["error"] and "BAR1 is 512 MiB" in reply["error"], reply
    assert not any(r[0] in ("MAP_SYSMEM_FD", "MAP_SYSMEM", "RESET") for r in reqs) and ("MAP_BAR", 1, 0, 0) in reqs, reqs
    writes = [r for r in reqs if r[0] == "MMIO_WRITE"]
    assert writes == [("MMIO_WRITE", 1, 0, 0x1000)], writes   # tinygrad's MMU init zeroes its root page table, as every boot does
    reqs, reply, log = run_scenario("fsp_not_ready")   # tinygrad's own 10 s wait
    want = "FSP not ready: NV_THERM_I2CS_SCRATCH=0x00000000"
    assert "TimeoutError" in reply["error"] and want in reply["error"] and want in log, (reply, log[-2000:])
    assert not any(r[0] in ("MMIO_WRITE", "MAP_SYSMEM_FD", "MAP_SYSMEM", "RESET") for r in reqs) and ("MAP_BAR", 1, 0, 0) not in reqs, reqs
    print("GB205 refusals: a 512 MiB BAR1 is refused after the MMU init (whose 4 KiB root page table zeroing is the only write), "
          "before any sysmem or firmware; an FSP that never reads 0xff fails the boot after tinygrad's 10 s wait with no MMIO write, "
          "no boot memory, and the cause in the plugin's error")

def test_ada_unchanged():
    """The BAR check adds nothing to Ada's stream: AD107 boots the same with and without it, up to the fake's VBIOS."""
    wrapped, reply, _ = run_scenario("ada")
    stock, _, _ = run_scenario("ada_unwrapped")
    assert wrapped == stock and ("MAP_BAR", 1, 0, 0) in wrapped and reply["ok"] is False, (len(wrapped), len(stock), reply)
    assert not any(r[0] == "MMIO_READ" and r[2] in (I2CS,) + FSP_QUEUES for r in wrapped), wrapped
    print(f"Ada: the AD107 request stream is identical with and without the BAR check ({len(stock)} requests); no Blackwell register is read")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--boot-scenario":
        boot_scenario(sys.argv[2]); sys.exit(0)
    for t in (test_pin, test_registers, test_riscv_include, test_halt_wait, test_unload_then_halt, test_not_halted_by_default,
              test_cot_message_flag, test_failed_boot_unload, test_sequencer_guard, test_compile_ptx_raises, test_level0_logged,
              test_script_refusal, test_bar_check, test_layout_gate, test_gb205_boot, test_gb205_refusals,
              test_ada_unchanged):
        t()
    print("B1 COT: all passed")
