"""Offline tests for plan step C9's daemon half, level flcn_hw (the C++ half is golden_flcn_hw.py, and runs end to end in
test_c9.sh): _boot_nvdev_only("flcn_hw") runs NVDev.__init__ with both init_hw doing nothing, and puts them back (also after a
failed boot); cmd_boot takes level flcn_hw; cmd_rm_export at flcn_hw adds what NV_FLCN.init_sw prepared for init_hw (FWSEC-FRTS's
image, desc_v3 load parameters and frts_offset; booter_load's image and offsets; the WPR meta for its mailbox); cmd_state_page
takes phase flcn_init at flcn_hw only; and at fini or EOF, before booter_load (phase flcn_init) the daemon closes without a word to
the GPU and without holding, while GSP-RM may run (phase gsp_init) it holds, and after GSP_INIT_DONE it unloads from the state
page's count. No GPU, no TinyGPU socket.
    python test_c9.py"""
import os, sys, types, socket
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tgpaths
tgpaths.setup()
import test_p3 as p3
import test_c5 as c5
import test_c7 as c7
import test_c8 as c8
import nv_dispatch_daemon as d
from tinygrad import Device
from tinygrad.runtime.support.nv.ip import NV_FLCN, NV_GSP

def test_boot_without_either_init_hw():
    ops_nv, calls = d.ops_nv, []
    def base_init(self, dev, dev_id, **kw):   # NVDev.__init__'s last loop: for ip in [flcn, gsp]: ip.init_hw()
        fl, gsp = NV_FLCN.__new__(NV_FLCN), NV_GSP.__new__(NV_GSP)
        NV_FLCN.init_hw(fl); NV_GSP.init_hw(gsp)
        self.dev_impl = types.SimpleNamespace(chip_name="AD107", flcn=fl, gsp=gsp)
    real_base, real_flcn, real_gsp = ops_nv.PCIIfaceBase.__init__, NV_FLCN.init_hw, NV_GSP.init_hw
    NV_FLCN.init_hw, NV_GSP.init_hw = lambda self: calls.append("flcn"), lambda self: calls.append("gsp")
    patched = (NV_FLCN.init_hw, NV_GSP.init_hw)
    ops_nv.PCIIfaceBase.__init__ = base_init
    try:
        d._boot_nvdev_only("flcn_hw")
        assert calls == [] and (NV_FLCN.init_hw, NV_GSP.init_hw) == patched, calls
        d._boot_nvdev_only("gsp_hw")
        assert calls == ["flcn"], calls
        def fails(self, dev, dev_id, **kw): raise RuntimeError("no GPU")
        ops_nv.PCIIfaceBase.__init__ = fails
        try: d._boot_nvdev_only("flcn_hw"); raise AssertionError("a failed boot must raise")
        except RuntimeError: pass
        assert (NV_FLCN.init_hw, NV_GSP.init_hw) == patched
    finally: ops_nv.PCIIfaceBase.__init__, NV_FLCN.init_hw, NV_GSP.init_hw = real_base, real_flcn, real_gsp
    print("NVDev-only boot at flcn_hw: neither init_hw runs during the boot, both are back afterwards (also after a failed boot); "
          "at gsp_hw the falcons' runs")

def test_boot_level():
    stubs = {"_apply_boot_safety_patches": lambda: None, "_install_inherited_tinygpu": lambda fd: None,
             "_boot_nvdev_only": lambda level="rm": types.SimpleNamespace(dev_impl=types.SimpleNamespace(chip_name="AD107"), device_fini=lambda: None)}
    real = {k: getattr(d, k) for k in stubs}
    for k, v in stubs.items(): setattr(d, k, v)
    try:
        a, b = socket.socketpair()
        dm = d.Daemon(b, 7)
        def hold(): raise p3.Held()
        dm._hold = hold
        (r, _), = c5.daemon_reply(dm, a, [{"cmd": "boot", "level": "flcn_hw"}])[0][:1]
        assert r == {"ok": True, "level": "flcn_hw"} and dm.rm_level == "flcn_hw", r
    finally:
        for k, v in real.items(): setattr(d, k, v)
    print("boot at level flcn_hw: accepted")

def test_rm_export_flcn_hw():
    real, Device._opened_devices = Device._opened_devices, set()
    try:
        a, dm, calls, queues, impl = c7.rm_rig()
        dm.rm_level = "flcn_hw"
        fl = impl.flcn
        fl.frts_image_paddr, fl.frts_offset = 0x1220000, 0x1ffc00000
        fl.booter_image_paddr, fl.booter_data_off, fl.booter_data_sz, fl.booter_code_off, fl.booter_code_sz = 0x1240000, 0x9000, 0x1600, 0x100, 0x8e00
        impl.gsp.wpr_meta_sysmem = 0x80256000
        (r, fds), = c5.daemon_reply(dm, a, [{"cmd": "rm_export"}])[0][:1]
        assert r["ok"] and r["rm_level"] == "flcn_hw" and len(fds) == 1, r
        assert (r["frts_paddr"], r["frts_offset"], r["frts_imem_sz"], r["frts_dmem_sz"], r["frts_pkc_off"], r["frts_engid"], r["frts_ucodeid"]) == \
               (0x1220000, 0x1ffc00000, 0xc100, 0x3e00, 0xb24, 0x400, 9), r
        assert (r["booter_paddr"], r["booter_data_off"], r["booter_data_sz"], r["booter_code_off"], r["booter_code_sz"], r["wpr_meta_sysmem"]) == \
               (0x1240000, 0x9000, 0x1600, 0x100, 0x8e00, 0x80256000), r
        assert "rm_priv_root" not in r and "rm_grctx" not in r, r
    finally: Device._opened_devices = real
    print("rm export at flcn_hw: FWSEC-FRTS's image, desc_v3 load parameters and frts_offset, booter_load's image and offsets, the "
          "WPR meta; no RM state init_hw would make")

def test_state_page_phase():
    real, Device._opened_devices = Device._opened_devices, set()
    try:
        for level, phase, ok in (("flcn_hw", d._PHASE_FLCN_INIT, True), ("flcn_hw", d._PHASE_GSP_INIT, True), ("gsp_hw", d._PHASE_FLCN_INIT, False)):
            a, dm, _ = c7.rm_state_rig()
            dm.rm_level = level
            dm.dev.iface.dev_impl.gsp.stat_q = types.SimpleNamespace()
            pf, _ = c7.page(2, phase=phase)
            r, = c7.serve(a, dm, [({"cmd": "state_page"}, [pf])])
            assert r["ok"] is ok, (level, phase, r)
    finally: Device._opened_devices = real
    print("state page: phase flcn_init taken at level flcn_hw only")

def test_fini_flcn_hw():
    real, Device._opened_devices = Device._opened_devices, set()
    try:
        for how in ("fini", "eof"):   # before booter_load: nothing to unload, and no hold
            a, dm, calls, seqs, seen, impl = c8.gsp_rig(2, d._PHASE_FLCN_INIT)
            dm.rm_level = "flcn_hw"
            if how == "fini":
                (r, _), = c5.daemon_reply(dm, a, [{"cmd": "fini"}])[0]
                assert not r["ok"] and not r.get("hold") and "nothing to unload" in r["error"], r
            else:
                held, log = p3.run(a, dm)
                assert not held and "closing is safe" in log, log
            assert calls == [] and seqs == [] and dm.dev is None, (how, calls)
        a, dm, calls, seqs, seen, impl = c8.gsp_rig(2, d._PHASE_GSP_INIT)   # booter_load ran: GSP-RM may be live
        dm.rm_level = "flcn_hw"
        (r, _), = c5.daemon_reply(dm, a, [{"cmd": "fini"}])[0]
        assert r.get("hold") and calls == ["hold"] and seqs == [], (r, calls)
        a, dm, calls, seqs, seen, impl = c8.gsp_rig(9, d._PHASE_DISPATCH)   # after GSP_INIT_DONE
        dm.rm_level = "flcn_hw"
        (r, _), = c5.daemon_reply(dm, a, [{"cmd": "fini"}])[0]
        assert r["ok"] and calls == ["finalize"] and seqs == [9] and seen["stat_q"] is not None, (r, calls, seqs)
    finally: Device._opened_devices = real
    print("fini and EOF at flcn_hw: before booter_load the daemon closes, sending nothing and not holding; while GSP-RM may run it "
          "holds; after GSP_INIT_DONE it unloads from the state page's count, with init_hw's status queue")

if __name__ == "__main__":
    test_boot_without_either_init_hw()
    test_boot_level()
    test_rm_export_flcn_hw()
    test_state_page_phase()
    test_fini_flcn_hw()
    print("C9 daemon: all passed")
