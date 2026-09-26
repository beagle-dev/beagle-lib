"""Offline tests for plan step C7's daemon half, level rm (the C++ half is golden_rm.py, and runs end to end in test_c7.sh):
_boot_nvdev_only runs tinygrad's own PCIIface.__init__ and stops it at its first RM call, unsent; cmd_boot at level rm replies
without an architecture, and bad levels are refused; the handoff is refused at rm; cmd_rm_export hands the C++ side the GSP
queues' own fd, tinygrad's memory manager and NV_GSP's RM state (the handle generator not advanced), once, only after a
level-rm boot and never on the COT boot; cmd_state_page at rm takes the page alone and cmd_timeline the C++ timeline after
it; and fini or EOF continues the GSP's command queue from the state page's count (the unload, or on the hung path the unload
RPC only and a hold), holding without a word to the GPU when the C++ side took the GSP over but sent no state page.
No GPU, no TinyGPU socket.
    python test_c7.py"""
import os, sys, io, types, struct, socket, tempfile, mmap, itertools, contextlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tgpaths
tgpaths.setup()
import test_p3 as p3   # its daemon rig (a real Daemon on a socketpair over a stub device that records its calls)
import test_c5 as c5   # its export rig (the GSP queues, the teardown images) and reply reader
import nv_dispatch_daemon as d
from tinygrad import Device
from tinygrad.runtime.autogen import nv
from tinygrad.runtime.support.memory import TLSFAllocator
from tinygrad.runtime.support.nv.ip import GRBufDesc

MB = 1 << 20

def test_boot_nvdev_only():
    """tinygrad's real PCIIface.__init__ over a stub PCIIfaceBase.__init__ (the probe and NVDev's boot)."""
    ops_nv, seen, sent = d.ops_nv, {}, []
    def base_init(self, dev, dev_id, **kw):
        seen.update(name=type(dev).__name__[:-6], dev_id=dev_id, vendor=kw["vendor"])
        self.dev_impl = types.SimpleNamespace(chip_name="AD107", gsp=types.SimpleNamespace(rpc_rm_alloc=lambda *a: sent.append(a)))
    real_base, real_rm_alloc = ops_nv.PCIIfaceBase.__init__, ops_nv.PCIIface.rm_alloc
    ops_nv.PCIIfaceBase.__init__ = base_init
    try:
        iface = d._boot_nvdev_only()
        def probe_fails(self, dev, dev_id, **kw): raise RuntimeError("no GPU")
        ops_nv.PCIIfaceBase.__init__ = probe_fails
        try: d._boot_nvdev_only(); raise AssertionError("a failed boot must raise")
        except RuntimeError as e: assert str(e) == "no GPU", e
    finally: ops_nv.PCIIfaceBase.__init__ = real_base
    assert seen == {"name": "NV", "dev_id": 0, "vendor": 0x10de} and sent == [], (seen, sent)   # named as NVDevice names it; no RPC
    assert (iface.root, iface.gpu_instance) == (0xc1000000, 0) and not hasattr(iface, "compute_class"), vars(iface)
    assert ops_nv.PCIIface.rm_alloc is real_rm_alloc   # tinygrad's own again, after a boot and after a failed one
    calls = []
    iface.device_fini = lambda: calls.append("device_fini")
    dev = d._RMDevice(iface)
    dev.synchronize(); dev.finalize()
    assert calls == ["device_fini"] and dev.error_state is None and "AD107" in repr(dev), calls
    print("NVDev-only boot: tinygrad's PCIIface.__init__ (the device named NV) stops at the root client's allocation, unsent, and "
          "rm_alloc is tinygrad's again (also after a failed boot); _RMDevice finalizes with device_fini")

def test_boot_levels():
    calls = []
    def nvdev(level="rm"):   # the booted PCIIface; its device_fini is NVDev.fini (the unload and NVIDIA's teardown, confirmed)
        impl = types.SimpleNamespace(chip_name="AD107")
        return types.SimpleNamespace(dev_impl=impl, device_fini=lambda: (calls.append("device_fini"), setattr(impl, "beagle_fini", dict(p3.CONFIRMED))))
    stubs = {"_apply_boot_safety_patches": lambda: None, "_install_inherited_tinygpu": lambda fd: None, "_boot_nvdev_only": nvdev}
    real = {k: getattr(d, k) for k in stubs}
    for k, v in stubs.items(): setattr(d, k, v)
    try:
        for tgpu_fd, level, want in ((7, "rm", True), (None, "rm", "level 'rm'"), (7, "nvdev", "level 'nvdev'")):
            a, b = socket.socketpair()
            dm = d.Daemon(b, tgpu_fd)
            def hold(): calls.append("hold"); raise p3.Held()
            dm._hold, calls[:] = hold, []
            (r, _), (h, _) = c5.daemon_reply(dm, a, [{"cmd": "boot", "level": level}, {"cmd": "handoff", "programs": False}])[0]
            if want is True:
                assert r == {"ok": True, "level": "rm"} and dm.rm_level and not dm.handed_off, r
                assert not h["ok"] and "at level rm" in h["error"], h
                assert calls == ["device_fini"] and dm.dev is None, calls   # the EOF before rm_export: the GSP is still this process's
            else: assert not r["ok"] and want in r["error"] and dm.dev is None and not dm.rm_level and calls == [], (level, r, calls)
    finally:
        for k, v in real.items(): setattr(d, k, v)
    print("boot at level rm: the NVDev only, a reply without an architecture, and no handoff; an EOF before rm_export unloads "
          "the GPU as NVDev.fini does; refused without the C++ side's TinyGPU.app connection and for any other level")

def rm_rig(fmc_boot=False, rm_level=True):
    """test_c5's export rig after a level-rm boot: the NVDev's memory manager built from real allocators, and NV_GSP's RM state
    as init_hw and init_golden_image leave it."""
    a, dm, calls, queues = c5.export_rig(fmc_boot=fmc_boot, handed_off=False)
    impl, pci = dm.dev.iface.dev_impl, dm.dev.iface.pci_dev
    pa = TLSFAllocator((8188 - 64 - 2 - 16) * MB, base=(2 + 16) * MB)
    pa.alloc(100 * MB, 0x1000)
    impl.mm = types.SimpleNamespace(vram_size=(8188 - 64) * MB, va_bits=48, va_shifts=[12, 21, 29, 38, 47], va_base=0,
                                    palloc_ranges=[(512 * MB, 512 * MB), (2 * MB, 2 * MB), (4096, 4096)], reserve_ptable=True,
                                    root_page_table=types.SimpleNamespace(paddr=0, lv=0), boot_allocator=TLSFAllocator(2 * MB),
                                    ptable_allocator=TLSFAllocator(16 * MB, base=2 * MB), pa_allocator=pa,
                                    va_allocator=TLSFAllocator(1 << 44, base=0x1000000000))
    impl.mmu_ver, impl.vram_size = 2, 8188 * MB
    g = impl.gsp
    g.wpr_meta = bytes(nv.GspFwWprMeta(gspFwRsvdStart=0x1f3a00000))
    g.priv_root, g.handle_gen, g.subdevice = 0xc1e00004, itertools.count(0xcf000007), 0xcf000002
    g.gpfifo_class, g.compute_class, g.dma_class, g.viddec_class = 0xc86f, 0xc9c0, 0xc8b5, None
    g.runlists, g.chan_runlists = {15: 2, 0: 3, 1: 0}, {0xcf000004: 3}
    g.grctx_bufs = {0: GRBufDesc(0x2a0000, phys=True, virt=True), 1: GRBufDesc(0x40000, phys=True, virt=True, local=True),
                    10: GRBufDesc(0x10000, phys=True, virt=False)}
    pci.bar_info = lambda bar: {0: (0x1c_0000_0000, 16 * MB), 1: (0x1d_0000_0000, 256 * MB)}[bar]
    if rm_level: dm.dev, dm.rm_level = d._RMDevice(dm.dev.iface), "rm"
    return a, dm, calls, queues, impl

def test_rm_export():
    real, Device._opened_devices = Device._opened_devices, set()
    try:
        a, dm, calls, queues, impl = rm_rig()
        (r, fds), (again, fds2) = c5.daemon_reply(dm, a, [{"cmd": "rm_export"}, {"cmd": "rm_export"}])[0][:2]
        assert r["ok"] and (r["gsp_cmdq_off"], r["gsp_queues_size"], r["gsp_seq"], r["chip_id"], r["fw_name"]) == (0x1000, 0x81000, 55, 0x197000a1, "ad102"), r
        assert (r["rm_priv_root"], r["rm_next_handle"], r["rm_gpfifo_class"], r["rm_compute_class"], r["rm_dma_class"], r["rm_viddec_class"],
                r["rm_gb2"], r["rm_subdevice"], r["rm_device"], r["bar0_size"]) == \
               (0xc1e00004, 0xcf000007, 0xc86f, 0xc9c0, 0xc8b5, 0, 0, 0xcf000002, 0, 16 * MB), r
        assert next(impl.gsp.handle_gen) == 0xcf000007, "the export advanced the handle generator"
        assert r["rm_runlists"] == [0, 3, 1, 0, 15, 2] and r["rm_chan_runlists"] == [0xcf000004, 3], r
        assert r["rm_grctx"] == [0, 0x2a0000, 1, 1, 0, 1, 0x40000, 1, 1, 1, 10, 0x10000, 1, 0, 0], r["rm_grctx"]   # the dict's order
        assert r["mm_pa"] == d._tlsf_save(impl.mm.pa_allocator) and (r["mm_wpr_bound"], r["bar1_size"]) == (0x1f3a00000, 256 * MB), r
        assert len(fds) == 1 and os.fstat(fds[0]).st_ino == os.fstat(queues.fileno()).st_ino, fds   # the queues' own fd
        assert dm.handed_off and dm.mm_exported and dm.rm_exported and calls == ["hold"], calls   # then an EOF with no state page
        assert not again["ok"] and "only once" in again["error"] and fds2 == [], again
        for kw, why in ((dict(rm_level=False), "after a boot at level rm"), (dict(fmc_boot=True), "COT boot")):
            a, dm, calls, _, _ = rm_rig(**kw)
            (r, fds), = c5.daemon_reply(dm, a, [{"cmd": "rm_export"}])[0][:1]
            assert not r["ok"] and why in r["error"] and fds == [] and not dm.rm_exported and not dm.handed_off, (kw, r)
    finally: Device._opened_devices = real
    print("rm export: the GSP queues' own fd and the teardown's arguments, the memory manager, NV_GSP's RM state (handles not "
          "advanced; runlists, the golden channel's runlist and the context buffers in order) and BAR0's size, once; refused "
          "without a level-rm boot and on the COT boot")

def page(seq, phase=d._PHASE_DISPATCH):
    f = tempfile.TemporaryFile(); f.truncate(d._STATE_WORDS * 8)
    m = mmap.mmap(f.fileno(), d._STATE_WORDS * 8); struct.pack_into("<4Q", m, 0, phase, 0, 0, seq)
    return f, m

def signal_file(value):
    f = tempfile.TemporaryFile(); f.truncate(0x4000)
    m = mmap.mmap(f.fileno(), 0x4000); struct.pack_into("<Q", m, 0, value)
    return f, m

def rm_state_rig(exported=True):
    a, dm, calls, _, _ = p3.rig()
    del dm._handoff_bufs   # at level rm the daemon never handed buffers over
    dm.dev.iface.dev_impl.gsp.cmd_q = types.SimpleNamespace(seq=13)
    dm.handed_off = dm.rm_exported = exported
    return a, dm, calls

def serve(a, dm, sends):
    """Each (request, fds) is sent, then the plugin goes away; the replies."""
    for req, fds in sends:
        a.sendall(p3.msg(req))
        if fds is not None: socket.send_fds(a, [b"S"], [f.fileno() for f in fds])
    a.shutdown(socket.SHUT_WR)
    with contextlib.redirect_stderr(io.StringIO()):
        try: dm.run()
        except p3.Held: pass
    dm.sock.close()
    out = []
    a.settimeout(5)
    while (hdr := a.recv(4, socket.MSG_WAITALL)) and len(hdr) == 4:
        out.append(__import__("json").loads(a.recv(struct.unpack("<I", hdr)[0], socket.MSG_WAITALL)))
    return out

def test_state_page_and_timeline():
    real, Device._opened_devices = Device._opened_devices, set()
    try:
        a, dm, _ = rm_state_rig()
        (pf, pm), (sf, sm) = page(13), signal_file(41)
        sig = {"signal_va": 0x1020768000, "signal_size": 0x4000}
        r1, r2, r3 = serve(a, dm, [({"cmd": "state_page"}, [pf]), ({"cmd": "timeline", **sig}, [sf]), ({"cmd": "timeline", **sig}, [sf])])
        assert r1["ok"] and r2["ok"] and dm._state[3] == 13 and dm._cpp_signal.value == 41, (r1, r2)
        struct.pack_into("<Q", sm, 0, 42); struct.pack_into("<Q", pm, 24, 33)
        assert dm._cpp_signal.value == 42 and dm._cpp_signal.value_addr == 0x1020768000 and dm._state[3] == 33   # both live
        assert not r3["ok"] and "timeline True" in r3["error"], r3   # only one timeline
        a, dm, _ = rm_state_rig()   # a timeline before the page: refused, the stream still framed
        r1, r2 = serve(a, dm, [({"cmd": "timeline", **sig}, [signal_file(1)[0]]), ({"cmd": "state_page"}, [page(13)[0]])])
        assert not r1["ok"] and "state page False" in r1["error"] and r2["ok"] and dm._cpp_signal is None, (r1, r2)
        a, dm, _ = rm_state_rig(exported=False)   # below level rm
        dm.handed_off = True
        r1, = serve(a, dm, [({"cmd": "timeline", **sig}, [signal_file(1)[0]])])
        assert not r1["ok"] and "level rm exported False" in r1["error"], r1
    finally: Device._opened_devices = real
    print("state page at level rm: the page alone, before the C++ side's first RPC, its sequence number read live; the C++ "
          "timeline after it (cmd_timeline), once; a timeline before the page or below level rm refused, the stream framed")

def fini_rig(seq=None, last=0, signal=None, in_flight=0):
    """A level-rm daemon after cmd_rm_export whose GSP client counted 13 commands; seq: the state page's count (None: no page);
    signal: the C++ timeline's value (None: not sent). Every unload records the count it continued from."""
    a, dm, calls, _, _ = p3.rig(signal=signal or 0)
    impl, seqs = dm.dev.iface.dev_impl, []
    impl.gsp.cmd_q = types.SimpleNamespace(seq=13)
    finalize, fini_hw = dm.dev.finalize, impl.gsp.fini_hw
    dm.dev.finalize = lambda: (seqs.append(impl.gsp.cmd_q.seq), finalize())
    impl.gsp.fini_hw = lambda: (seqs.append(impl.gsp.cmd_q.seq), fini_hw())
    dm.handed_off = dm.rm_exported = True
    if seq is not None:
        dm._state = memoryview(bytearray(struct.pack("<4Q", d._PHASE_DISPATCH, in_flight, last, seq))).cast("Q")
        if signal is not None: dm._cpp_signal = d.ops_nv.NVSignal(base_buf=dm._handoff_bufs["signal"], owner=dm.dev, virt=True)
    del dm._handoff_bufs
    return a, dm, calls, seqs

def test_fini_rm():
    real, Device._opened_devices = Device._opened_devices, set()
    try:
        a, dm, calls, seqs = fini_rig()
        (r, _), = c5.daemon_reply(dm, a, [{"cmd": "fini"}])[0]
        assert r.get("hold") and "without a state page" in r["error"] and calls == ["hold"] and seqs == [], (r, calls)
        a, dm, calls, seqs = fini_rig(); held, log = p3.run(a, dm)
        assert held and calls == ["hold"] and seqs == [] and "never sent its state page" in log, (calls, log)
        a, dm, calls, seqs = fini_rig(seq=33)
        (r, _), = c5.daemon_reply(dm, a, [{"cmd": "fini"}])[0]
        assert r["ok"] and not r.get("hold") and calls == ["finalize"] and seqs == [33] and dm.dev is None, (r, calls, seqs)
        a, dm, calls, seqs = fini_rig(seq=33); held, log = p3.run(a, dm)   # EOF: as fini
        assert not held and calls == ["finalize"] and seqs == [33] and "not sent" in log, (calls, seqs, log)
        a, dm, calls, seqs = fini_rig(seq=33)
        (r, _), = c5.daemon_reply(dm, a, [{"cmd": "fini", "hung": True}])[0]
        assert r.get("hung") and r.get("hold") and calls == ["gsp.fini_hw", "hold"] and seqs == [33], (r, calls, seqs)
        a, dm, calls, seqs = fini_rig(seq=40, last=5, signal=5)   # the C++ timeline reached its last value: the unload follows
        (r, _), = c5.daemon_reply(dm, a, [{"cmd": "fini"}])[0]
        assert r["ok"] and calls == ["finalize"] and seqs == [40], (r, calls, seqs)
        a, dm, calls, seqs = fini_rig(seq=21, in_flight=1); held, log = p3.run(a, dm)   # died while building the NVDevice
        assert held and calls == ["hold"] and seqs == [] and "cut mid-send" in log, (calls, log)
    finally: Device._opened_devices = real
    print("fini and EOF at level rm: the unload continues the GSP's command queue from the state page's count (the unload RPC "
          "only and a hold on the hung path); no state page, or a death while the NVDevice was being built (frame_in_flight), a hold "
          "with nothing sent to the GPU")

def test_hold_robust():
    """The hold never depends on the log, and an exception from the fini decision holds (plan step C7's review)."""
    real, Device._opened_devices = Device._opened_devices, set()
    try:
        a, b = socket.socketpair()
        dm = d.Daemon(b)
        class Slept(BaseException): pass
        class FullDisk:
            def write(self, s): raise OSError(28, "No space left on device")
            def flush(self): raise OSError(28, "No space left on device")
        def slept(s): raise Slept()
        real_stderr, real_sleep = sys.stderr, d.time.sleep
        sys.stderr, d.time.sleep = FullDisk(), slept
        try:
            d.log("a line the disk cannot take")   # swallowed
            try: dm._hold(); raise AssertionError("the hold returned")
            except Slept: pass   # it reached its sleep although the log fails
        finally: sys.stderr, d.time.sleep = real_stderr, real_sleep
        a, dm, calls, seqs = fini_rig(seq=21)
        def broken(hung): raise KeyError("stat_q")
        dm._fini = broken
        (r, _), = c5.daemon_reply(dm, a, [{"cmd": "fini"}])[0]
        assert r.get("hold") and "the fini decision failed: KeyError" in r["error"] and calls == ["hold"], (r, calls)
        a, dm, calls, seqs = fini_rig(seq=21); dm._fini = broken; held, log = p3.run(a, dm)   # EOF: the same
        assert held and calls == ["hold"], calls
    finally: Device._opened_devices = real
    print("hold rule: the hold reaches its sleep although the log fails; a fini decision that raises holds, at fini and at EOF")

if __name__ == "__main__":
    test_boot_nvdev_only()
    test_boot_levels()
    test_rm_export()
    test_state_page_and_timeline()
    test_fini_rm()
    test_hold_robust()
    print("C7 daemon: all passed")
