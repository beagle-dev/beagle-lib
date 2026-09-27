"""Offline tests for plan step C8's daemon half, level gsp_hw (the C++ half is golden_gsp_hw.py, and runs end to end in
test_c8.sh): _boot_nvdev_only(gsp_hw) runs NVDev.__init__ with NV_GSP.init_hw doing nothing, and puts nv_init_helper's back
(also after a failed boot); cmd_boot takes level gsp_hw; cmd_rm_export at gsp_hw sends init_sw's RM state only (the handle
generator's next value, the classes), no RM state init_hw would make; cmd_state_page takes the page in phase gsp_init at
gsp_hw only; and fini or EOF holds without a word to the GPU while the state page says the C++ side had not read GSP_INIT_DONE,
and afterwards continues the GSP's command queue from the page's count with the status queue init_hw would have made.
No GPU, no TinyGPU socket.
    python test_c8.py"""
import os, sys, types, struct, mmap, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tgpaths
tgpaths.setup()
import test_p3 as p3   # its daemon rig (a real Daemon on a socketpair over a stub device that records its calls)
import test_c5 as c5   # its reply reader
import test_c7 as c7   # its level-rm rigs
import golden_gsp as gg
import nv_dispatch_daemon as d
from tinygrad import Device
from tinygrad.runtime.support.nv.ip import NV_GSP
from tinygrad.runtime.support.hcq import MMIOInterface

def test_boot_without_init_hw():
    """NVDev.__init__ as a stub PCIIfaceBase.__init__ runs it: the falcons' init_hw, then the GSP's (the last statement)."""
    ops_nv, calls = d.ops_nv, []
    def base_init(self, dev, dev_id, **kw):
        gsp = NV_GSP.__new__(NV_GSP)
        NV_GSP.init_hw(gsp)   # NVDev.__init__'s last statement
        self.dev_impl = types.SimpleNamespace(chip_name="AD107", gsp=gsp)
    real_base, real_init_hw = ops_nv.PCIIfaceBase.__init__, NV_GSP.init_hw
    NV_GSP.init_hw = lambda self: calls.append("init_hw")   # stands in for nv_init_helper's _patched_gsp_init_hw
    patched = NV_GSP.init_hw
    ops_nv.PCIIfaceBase.__init__ = base_init
    try:
        d._boot_nvdev_only("gsp_hw")
        assert calls == [] and NV_GSP.init_hw is patched, calls   # nothing ran, and the patched one is back
        d._boot_nvdev_only()
        assert calls == ["init_hw"], calls                         # level rm: init_hw is the daemon's
        def fails(self, dev, dev_id, **kw): raise RuntimeError("no GPU")
        ops_nv.PCIIfaceBase.__init__ = fails
        try: d._boot_nvdev_only("gsp_hw"); raise AssertionError("a failed boot must raise")
        except RuntimeError: pass
        assert NV_GSP.init_hw is patched
    finally: ops_nv.PCIIfaceBase.__init__, NV_GSP.init_hw = real_base, real_init_hw
    print("NVDev-only boot at gsp_hw: NV_GSP.init_hw does nothing during the boot, and nv_init_helper's is back afterwards, also "
          "after a failed boot; at level rm it runs")

def test_boot_level():
    stubs = {"_apply_boot_safety_patches": lambda: None, "_install_inherited_tinygpu": lambda fd: None,
             "_boot_nvdev_only": lambda level="rm": types.SimpleNamespace(dev_impl=types.SimpleNamespace(chip_name="AD107", level=level),
                                                                             device_fini=lambda: None)}
    real = {k: getattr(d, k) for k in stubs}
    for k, v in stubs.items(): setattr(d, k, v)
    try:
        for tgpu_fd, want in ((7, True), (None, "level 'gsp_hw'")):
            a, b = __import__("socket").socketpair()
            dm = d.Daemon(b, tgpu_fd)
            def hold(): raise p3.Held()
            dm._hold = hold
            (r, _), = c5.daemon_reply(dm, a, [{"cmd": "boot", "level": "gsp_hw"}])[0][:1]
            if want is True: assert r == {"ok": True, "level": "gsp_hw"} and dm.rm_level == "gsp_hw", r
            else: assert not r["ok"] and want in r["error"] and not dm.rm_level, r
    finally:
        for k, v in real.items(): setattr(d, k, v)
    print("boot at level gsp_hw: accepted with the C++ side's TinyGPU.app connection, refused without it")

def test_rm_export_gsp_hw():
    real, Device._opened_devices = Device._opened_devices, set()
    try:
        a, dm, calls, queues, impl = c7.rm_rig()
        dm.rm_level = "gsp_hw"
        g = impl.gsp
        for k in ("priv_root", "runlists", "chan_runlists", "grctx_bufs", "subdevice"): delattr(g, k)   # init_hw never ran here
        g.handle_gen = itertools.count(0xcf000000)
        (r, fds), = c5.daemon_reply(dm, a, [{"cmd": "rm_export"}])[0][:1]
        assert r["ok"] and r["rm_level"] == "gsp_hw" and r["rm_next_handle"] == 0xcf000000 and r["rm_compute_class"] == 0xc9c0, r
        assert not any(k in r for k in ("rm_priv_root", "rm_runlists", "rm_chan_runlists", "rm_grctx", "rm_subdevice", "rm_device")), r
        assert r["gsp_seq"] == 55 and "mm_pa" in r and len(fds) == 1 and dm.rm_exported, r
    finally: Device._opened_devices = real
    print("rm export at gsp_hw: the GSP queues, the memory manager and init_sw's RM state (next handle, classes); none of what "
          "init_hw and init_golden_image make")

def test_state_page_phase():
    real, Device._opened_devices = Device._opened_devices, set()
    try:
        for level, phase, ok in (("gsp_hw", d._PHASE_GSP_INIT, True), ("gsp_hw", d._PHASE_DISPATCH, True), ("rm", d._PHASE_GSP_INIT, False)):
            a, dm, _ = c7.rm_state_rig()
            dm.rm_level = level
            dm.dev.iface.dev_impl.gsp.stat_q = types.SimpleNamespace()   # the unload at EOF is test_fini_gsp_hw's
            pf, _ = c7.page(2, phase=phase)
            r, = c7.serve(a, dm, [({"cmd": "state_page"}, [pf])])
            assert r["ok"] is ok and (dm._state is not None) is ok, (level, phase, r)
    finally: Device._opened_devices = real
    print("state page: phase gsp_init taken at level gsp_hw only")

def queue_views():
    path = f"{tgpaths.WORK}/c8_queues"
    m = gg.init_queues(path, cmd_wp=6, stat_wp=9)
    base = MMIOInterface(gg.ctypes_addr(m), gg.QSIZE, fmt='B')
    return m, base.view(gg.CMDQ), base.view(gg.STATQ)

def gsp_rig(seq, phase):
    """A level-gsp_hw daemon after cmd_rm_export whose GSP never ran init_hw (no status queue), the state page at seq and phase."""
    a, dm, calls, seqs = c7.fini_rig(seq=seq)
    dm.rm_level = "gsp_hw"
    dm._state = memoryview(bytearray(struct.pack("<5Q", phase, 0, 0, seq, 0))).cast("Q")
    impl = dm.dev.iface.dev_impl
    impl.gsp._m, impl.gsp.cmd_q_view, impl.gsp.stat_q_view = queue_views()
    seen = {}
    fin = dm.dev.finalize
    def finalize():   # what the unload finds: init_hw's status queue, and the command queue's read pointer
        seen.update(stat_q=getattr(impl.gsp, "stat_q", None), rx=getattr(impl.gsp.cmd_q, "rx_view", None))
        fin()
    dm.dev.finalize = finalize
    return a, dm, calls, seqs, seen, impl

def test_fini_gsp_hw():
    real, Device._opened_devices = Device._opened_devices, set()
    try:
        a, dm, calls, seqs, seen, impl = gsp_rig(2, d._PHASE_GSP_INIT)
        (r, _), = c5.daemon_reply(dm, a, [{"cmd": "fini"}])[0]
        assert r.get("hold") and "did not finish booting GSP-RM" in r["error"] and calls == ["hold"] and seqs == [], (r, calls)
        a, dm, calls, seqs, seen, impl = gsp_rig(2, d._PHASE_GSP_INIT); held, log = p3.run(a, dm)   # EOF: the same
        assert held and calls == ["hold"] and seqs == [] and "sending nothing" in log, (calls, log)
        a, dm, calls, seqs, seen, impl = gsp_rig(6, d._PHASE_DISPATCH)   # after GSP_INIT_DONE, a failure in the golden image
        (r, _), = c5.daemon_reply(dm, a, [{"cmd": "fini"}])[0]
        assert r["ok"] and not r.get("hold") and calls == ["finalize"] and seqs == [6], (r, calls, seqs)
        q = seen["stat_q"]
        assert q is not None and q.rx_view[0] == 9 and seen["rx"] is not None and seen["rx"][0] == 6, seen   # both read pointers, shared
    finally: Device._opened_devices = real
    print("fini and EOF at gsp_hw: before GSP_INIT_DONE a hold with nothing sent to the GPU; after it, the unload continues from "
          "the state page's count, with init_hw's status queue and the command queue's read pointer made as init_hw makes them")

if __name__ == "__main__":
    test_boot_without_init_hw()
    test_boot_level()
    test_rm_export_gsp_hw()
    test_state_page_phase()
    test_fini_gsp_hw()
    print("C8 daemon: all passed")
