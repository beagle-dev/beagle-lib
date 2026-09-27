"""Offline tests for plan step P3: the teardown default, the daemon half of the C++ state page, the daemon's fini and EOF
decisions (nv_dispatch_daemon.py Daemon._fini/_eof), and cmd_handoff's WPR check. No GPU, no TinyGPU socket; the C++
half of the state page runs end to end in run_fake_runtime.sh.
    python test_p3.py"""
import os, sys, io, json, types, ctypes, struct, socket, tempfile, mmap, subprocess, contextlib
os.environ["HCQDEV_WAIT_TIMEOUT_MS"] = "300"   # tinygrad's signal-wait timeout (30 s), shortened for the stuck cases
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tgpaths
tgpaths.setup()
import nv_dispatch_daemon as d
from tinygrad import Device
from tinygrad.runtime.autogen import nv
from tinygrad.runtime.support.memory import TLSFAllocator, MMIOInterface

MB = 1 << 20

def test_default_on():
    """NVIDIA's teardown is on unless BEAGLE_NV_TEARDOWN=0 (read at import, so one interpreter per value)."""
    code = f"import sys; sys.path.insert(0, {str(tgpaths.HERE)!r}); import tgpaths; tgpaths.setup(); import nv_init_helper as h; print(h._TEARDOWN)"
    for val, want in ((None, "True"), ("1", "True"), ("", "True"), ("0", "False")):
        env = {k: v for k, v in os.environ.items() if k != "BEAGLE_NV_TEARDOWN"}
        env["BEAGLE_TINYGPU_NO_LAUNCH"] = "1"
        if val is not None: env["BEAGLE_NV_TEARDOWN"] = val
        r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
        assert r.stdout.split() == [want], (val, r.stdout, r.stderr[-500:])
    print("default: NVIDIA's teardown is on unless BEAGLE_NV_TEARDOWN=0")

# ── the daemon's side of the state page, and its fini and EOF decisions ───────
class Held(BaseException): """the stubbed _hold: like the real one, the daemon goes no further"""

def msg(obj): body = json.dumps(obj).encode(); return struct.pack("<I", len(body)) + body
def recv_reply(sock): n = struct.unpack("<I", sock.recv(4, socket.MSG_WAITALL))[0]; return json.loads(sock.recv(n, socket.MSG_WAITALL))

CONFIRMED, NOT_CONFIRMED = {"unload_ok": True, "mailbox0": 0x80000000}, {"unload_ok": False, "mailbox0": 0}

def rig(fini=CONFIRMED, dev=True, signal=0, advance_to=None, fault=False, error_state=None, sync_error=None):
    """A real Daemon on a socketpair, over a stub device whose unload records its calls. The C++ timeline is a host buffer
    read through tinygrad's own NVSignal; while a wait sleeps (PCIIface.sleep, which drains the GSP status queue) the fake
    GPU moves it one step toward advance_to, or reports a fault. sync_error: the daemon's own timeline times out, as
    HCQCompiled.synchronize reports it (error_state set, then raised)."""
    a, b = socket.socketpair()
    dm, calls, drains = d.Daemon(b), [], [0]
    mem = (ctypes.c_uint8 * 16)()
    buf = d.HCQBuffer(0x10_3000_0000, 16, view=MMIOInterface(ctypes.addressof(mem), 16))
    struct.pack_into("<Q", mem, 0, signal)
    def sleep(ms):
        drains[0] += 1
        if fault: raise RuntimeError("Device fault detected")
        cur = struct.unpack_from("<Q", mem, 0)[0]
        if advance_to is not None and cur < advance_to: struct.pack_into("<Q", mem, 0, cur + 1)
    impl = types.SimpleNamespace()
    def unload(what): calls.append(what); impl.beagle_fini = dict(fini)
    impl.gsp = types.SimpleNamespace(fini_hw=lambda: unload("gsp.fini_hw"))
    def synchronize():
        if dm.dev.error_state is not None: raise dm.dev.error_state
        if sync_error is not None: dm.dev.error_state = sync_error; raise sync_error
    if dev:
        dm.dev = types.SimpleNamespace(iface=types.SimpleNamespace(dev_impl=impl, sleep=sleep), finalize=lambda: unload("finalize"),
                                       error_state=error_state, synchronize=synchronize)
        Device._opened_devices.add("NV")
    def hold(): calls.append("hold"); raise Held()
    dm._hold, dm._handoff_bufs = hold, {"signal": buf}
    return a, dm, calls, drains, mem

def with_page(dm, in_flight, last, phase=1):   # the page as cmd_state_page leaves it
    dm.handed_off = True
    dm._state = memoryview(bytearray(struct.pack("<5Q", phase, in_flight, last, 0, 0))).cast("Q")
    dm._cpp_signal = d.ops_nv.NVSignal(base_buf=dm._handoff_bufs["signal"], owner=dm.dev, virt=True)

def run(a, dm, send=b""):
    """The plugin sends `send` and goes away; returns whether the daemon held and what it logged."""
    a.sendall(send); a.close()
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err): dm.run()
    except Held: return True, err.getvalue()
    return False, err.getvalue()

def test_state_page():
    """cmd_state_page on the real Daemon: the fd byte is taken before any check (a refused page leaves the stream framed),
    the page is read live, and its timeline is read with tinygrad's own signal, which writes nothing (virt)."""
    for handed_off, phase, ok in ((True, 1, True), (True, 0, False), (False, 1, False)):
        a, dm, calls, _, mem = rig(signal=6)
        dm.handed_off = handed_off
        f = tempfile.TemporaryFile(); f.truncate(d._STATE_WORDS * 8)
        mine = mmap.mmap(f.fileno(), d._STATE_WORDS * 8); struct.pack_into("<Q", mine, 0, phase)
        a.sendall(msg({"cmd": "state_page"})); socket.send_fds(a, [b"S"], [f.fileno()]); f.close()
        a.sendall(msg({"cmd": "fini"})); a.shutdown(socket.SHUT_WR)   # parses only if the fd byte was consumed
        err = io.StringIO()
        with contextlib.redirect_stderr(err): dm.run()
        dm.sock.close()   # a missing reply then fails the test instead of blocking it
        r, fini = recv_reply(a), recv_reply(a)
        assert r["ok"] is ok and (dm._state is not None) is ok and fini["ok"], (handed_off, phase, r, fini)
        if ok:
            struct.pack_into("<2Q", mine, 8, 1, 7)
            assert list(dm._state) == [1, 1, 7, 0, 0] and dm._cpp_signal.value == 6, list(dm._state)   # live words (the keeper word 0: the daemon); the C++ signal as tinygrad reads it
            assert struct.unpack_from("<Q", mem, 0)[0] == 6                                     # NVSignal(virt=True) wrote nothing
            assert "C++ state page: phase 1, frame_in_flight 0, last_submitted 0, seq 0, C++ timeline signal 6" in err.getvalue(), err.getvalue()
    print("state page: taken before any check (refusals before the handoff or with phase 0 keep the stream framed), read live, "
          "timeline read through tinygrad's NVSignal without a write, logged at fini")

def test_fini_and_eof():
    real, Device._opened_devices = Device._opened_devices, set()   # never the real registry (STATUS.md R16)
    try:
        # EOF with no device (no boot, or a refused one): nothing touched
        a, dm, calls, _, _ = rig(dev=False); held, log = run(a, dm)
        assert not held and calls == [] and "no device to tear down" in log, (calls, log)
        # EOF in default mode or before the handoff (no page): all GPU work is the daemon's, and finalize waits for it
        a, dm, calls, drains, _ = rig(); held, log = run(a, dm)
        assert not held and calls == ["finalize"] and dm.dev is None and "NV" not in Device._opened_devices, (calls, log)
        a, dm, calls, _, _ = rig(fini=NOT_CONFIRMED); held, _ = run(a, dm)
        assert held and calls == ["finalize", "hold"], calls
        # EOF after the handoff, the C++ side idle: its timeline is where it submitted, so the full teardown at once
        a, dm, calls, drains, _ = rig(signal=7); with_page(dm, 0, 7); held, log = run(a, dm)
        assert not held and calls == ["finalize"] and drains[0] == 0 and '"unload_ok": true' in log, (calls, log)
        # EOF with a frame in flight: hold, and nothing at all to the GPU (not even the unload RPC)
        a, dm, calls, _, _ = rig(signal=6); with_page(dm, 1, 7); held, log = run(a, dm)
        assert held and calls == ["hold"] and "sending nothing more to the GPU" in log, (calls, log)
        # EOF with the C++ timeline behind: tinygrad's wait (status-queue drains) until it arrives, then the full teardown
        a, dm, calls, drains, _ = rig(signal=5, advance_to=7); with_page(dm, 0, 7); held, log = run(a, dm)
        assert not held and calls == ["finalize"] and drains[0] >= 2, (calls, drains, log)
        # EOF with the C++ timeline stuck, or a GSP fault while waiting: the hung path (unload RPC only), then a hold even when
        # the unload is confirmed (a channel stuck on an acquire may still poll the sysmem timeline page: unplug first)
        for kw, fini in (({}, CONFIRMED), ({}, NOT_CONFIRMED), ({"fault": True}, CONFIRMED)):
            a, dm, calls, _, _ = rig(signal=5, fini=fini, **kw); with_page(dm, 0, 7); held, log = run(a, dm)
            assert calls == ["gsp.fini_hw", "hold"] and held and "the hung path" in log and '"hung": true' in log, (kw, calls, log)
        def fini_held(dm):   # the daemon replies, then holds (the stub _hold raises Held)
            err = io.StringIO()
            try:
                with contextlib.redirect_stderr(err): dm.run()
            except Held: pass
            return err.getvalue()
        # fini{hung}: the plugin already saw its timeline stop, so no wait for it here either; the unload RPC only, then the hold
        a, dm, calls, drains, _ = rig(signal=5); with_page(dm, 0, 7); a.sendall(msg({"cmd": "fini", "hung": True}))
        log = fini_held(dm); a.settimeout(5); r = recv_reply(a)
        assert calls == ["gsp.fini_hw", "hold"] and drains[0] == 0 and r["hung"] and r["hold"] and r["pid"] == os.getpid() \
            and "C++ timeline stuck" not in log, (calls, drains, r)
        # fini (and fini{hung}) with a frame in flight: the same hold, no device call
        for req in ({"cmd": "fini"}, {"cmd": "fini", "hung": True}):
            a, dm, calls, _, _ = rig(signal=6); with_page(dm, 1, 7)
            a.sendall(msg(req))
            try:
                with contextlib.redirect_stderr(io.StringIO()): dm.run()
            except Held: pass
            a.settimeout(5); r = recv_reply(a)   # a missing reply fails the test instead of blocking it
            assert calls == ["hold"] and r["hold"] and r["pid"] == os.getpid() and not r["ok"], (req, calls, r)
        # fini after a timeout the daemon saw itself (default mode: tinygrad's error_state): the hung path, no falcon step
        a, dm, calls, _, _ = rig(error_state=RuntimeError("Wait timeout: 30000 ms!")); a.sendall(msg({"cmd": "fini"}))
        fini_held(dm); a.settimeout(5); r = recv_reply(a)
        assert calls == ["gsp.fini_hw", "hold"] and r["hung"] and r["unload_ok"] and r["hold"], (calls, r)
        # ... and when that timeout is first seen at fini or at EOF (default mode, no later sync): tinygrad's finalize would
        # swallow it and run the falcon teardown, so the daemon synchronizes first and takes the hung path
        a, dm, calls, _, _ = rig(sync_error=RuntimeError("Wait timeout: 30000 ms!")); a.sendall(msg({"cmd": "fini"}))
        fini_held(dm); a.settimeout(5); r = recv_reply(a)
        assert calls == ["gsp.fini_hw", "hold"] and r["hung"] and r["unload_ok"] and r["hold"], (calls, r)
        a, dm, calls, _, _ = rig(sync_error=RuntimeError("Device fault detected")); held, log = run(a, dm)
        assert held and calls == ["gsp.fini_hw", "hold"] and '"hung": true' in log and "the hung path" in log, (calls, log)
        # the plugin gone mid-message: a cut header, or an h2d cut in its payload, is an EOF
        for send in (msg({"cmd": "sync"})[:6], msg({"cmd": "h2d", "addr": 0, "size": 64}) + b"x" * 10):
            a, dm, calls, _, _ = rig(); held, log = run(a, dm, send)
            assert not held and calls == ["finalize"] and "ConnectionResetError" in log, (calls, log)
        # the plugin gone before reading the fini reply: one teardown, then EOF finds no device; unconfirmed still holds
        a, dm, calls, _, _ = rig(); held, log = run(a, dm, msg({"cmd": "fini"}))
        assert not held and calls == ["finalize"] and "no device to tear down" in log, (calls, log)
        a, dm, calls, _, _ = rig(fini=NOT_CONFIRMED); held, _ = run(a, dm, msg({"cmd": "fini"}))
        assert held and calls == ["finalize", "hold"], calls
    finally: Device._opened_devices = real
    print("fini and EOF: no device -> exit; no page -> finalize; idle -> teardown; frame in flight -> hold with nothing sent "
          "(fini and fini{hung} too); timeline behind -> tinygrad's wait, then teardown; stuck or faulted -> hung path, then "
          "a hold; fini{hung} -> no wait; error_state, set before or first seen at fini/EOF -> hung path, then a hold; cut "
          "messages and lost "
          "replies take the same decision, once")

def test_wpr_check():
    """cmd_handoff's WPR check on tinygrad's real allocator: the default pool passes, a pool reaching into GSP-RM's reserved
    region is refused; on FMC-booted chips (Blackwell, plan step B1) the bound is vram_size - 512 MiB."""
    # gspFwRsvdStart as tinygrad's init_wpr_meta (ip.py:447-452) computes it for the 8188 MiB RTX 4060 and 570.144 firmware:
    # 1 MiB below gspFwWprStart 0x1f3b00000, which the GPU raised as WPR2_LO 0x01f3b000 (STATUS.md R17, R18)
    meta = bytes(nv.GspFwWprMeta(gspFwRsvdStart=0x1f3a00000))
    def dev_impl(pool_mb, vram_mb=8188, fmc_boot=False):
        pa = TLSFAllocator((vram_mb - 64 - 2 - 16) * MB, base=(2 + 16) * MB)   # NVMemoryManager's (nvdev.py:146, memory.py:190-192)
        pa.alloc(100 * MB, 0x1000)                                            # allocations made before the pool
        pa.alloc(pool_mb * MB, 0x1000)
        return types.SimpleNamespace(fmc_boot=fmc_boot, vram_size=vram_mb * MB, gsp=types.SimpleNamespace(wpr_meta=meta),
                                     mm=types.SimpleNamespace(pa_allocator=pa))
    end, rsvd = d.check_vram_below_wpr(dev_impl(4094))   # cmd_handoff's default pool, half the VRAM
    assert (end, rsvd) == ((18 + 100 + 4094) * MB, 0x1f3a00000), (hex(end), hex(rsvd))
    try: d.check_vram_below_wpr(dev_impl(7900)); raise AssertionError("a pool reaching into GSP-RM's reserved region was accepted")
    except RuntimeError as e: assert "above gspFwRsvdStart 0x1f3a00000" in str(e), e
    # the RTX 5070 (GB205, 12227 MiB): the default pool passes, a pool ending above vram_size - 512 MiB is refused; the
    # WPR meta (gspFwRsvdStart 0 there) is not read
    end5, rsvd5 = d.check_vram_below_wpr(dev_impl(12227 // 2, 12227, fmc_boot=True))
    assert (end5, rsvd5) == ((18 + 100 + 12227 // 2) * MB, (12227 - 512) * MB), (hex(end5), hex(rsvd5))
    try: d.check_vram_below_wpr(dev_impl(12227 - 512 - 100 - 18 + 1, 12227, fmc_boot=True)); raise AssertionError("an FMC pool above the bound was accepted")
    except RuntimeError as e: assert f"above vram_size - 512 MiB {(12227 - 512) * MB:#x}" in str(e), e
    print(f"WPR check: the default pool ends at 0x{end:x} <= gspFwRsvdStart 0x{rsvd:x} and passes; a 7900 MiB pool is refused; "
          f"on the 12227 MiB GB205 the bound is vram_size - 512 MiB (0x{rsvd5:x}): the default pool passes, 1 MiB over is refused")

def test_handoff_wpr_refusal():
    """The real cmd_handoff: a pool reaching into GSP-RM's reserved region is refused before anything is sent (one error
    reply, no blob, no fds), the daemon keeps the queues, and the EOF that follows tears down as usual."""
    real_opened, real_build, Device._opened_devices = Device._opened_devices, d.build_handoff, set()
    try:
        pa = TLSFAllocator((8188 - 64 - 2 - 16) * MB, base=(2 + 16) * MB)
        pa.alloc(100 * MB, 0x1000)
        views, fds = [], {}
        def alloc(size, spec=None):
            views.append((ctypes.c_uint8 * 64)()); view = MMIOInterface(ctypes.addressof(views[-1]), 64)
            fds[view.addr] = 100 + len(fds)
            return types.SimpleNamespace(va_addr=pa.alloc(size, 0x1000), size=size, cpu_view=lambda: view)
        calls = []
        impl = types.SimpleNamespace(fmc_boot=False, vram_size=8188 * MB, gsp=types.SimpleNamespace(wpr_meta=bytes(nv.GspFwWprMeta(gspFwRsvdStart=0x1f3a00000)),
                                     fini_hw=lambda: calls.append("gsp.fini_hw")), mm=types.SimpleNamespace(pa_allocator=pa))
        def finalize(): calls.append("finalize"); impl.beagle_fini = dict(CONFIRMED)
        dev = types.SimpleNamespace(allocator=types.SimpleNamespace(alloc=alloc), synchronize=lambda: None, finalize=finalize, error_state=None,
                                    iface=types.SimpleNamespace(dev_impl=impl, pci_dev=types.SimpleNamespace(sysmem_fds=fds), compute_class=0xc9c0),
                                    sass_version=0x89, shared_mem_window=0, local_mem_window=0, num_gpcs=3, num_tpc_per_gpc=4,
                                    num_sm_per_tpc=2, max_warps_per_sm=48)
        d.build_handoff = lambda dev, progs, bufs: ({"qmd_ver": 3}, b"blob")
        a, b = socket.socketpair()
        dm = d.Daemon(b, 7); dm.dev, dm.elf_bytes = dev, b"elf"
        Device._opened_devices.add("NV")
        a.sendall(msg({"cmd": "handoff", "programs": False, "pool_size": 7900 * MB})); a.shutdown(socket.SHUT_WR)
        with contextlib.redirect_stderr(io.StringIO()): dm.run()
        dm.sock.close()
        data, got_fds, _, _ = socket.recv_fds(a, 1 << 16, 4)
        r = json.loads(data[4:4 + struct.unpack_from("<I", data)[0]])
        assert not r["ok"] and "above gspFwRsvdStart 0x1f3a00000" in r["error"] and len(data) == 4 + len(json.dumps(r)), (r, len(data))
        assert got_fds == [] and not dm.handed_off and calls == ["finalize"], (got_fds, dm.handed_off, calls)
    finally: Device._opened_devices, d.build_handoff = real_opened, real_build
    print("handoff WPR refusal: the real cmd_handoff replies one error before any blob or fd; the EOF that follows tears down")

if __name__ == "__main__":
    test_default_on()
    test_state_page()
    test_fini_and_eof()
    test_wpr_check()
    test_handoff_wpr_refusal()
    print("P3: all passed")
