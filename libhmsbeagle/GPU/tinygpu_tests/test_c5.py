"""Offline tests for plan step C5's daemon half and its log (the C++ half is golden_gsp.py, and runs end to end in
test_c5.sh): cmd_teardown_export hands the C++ side the GSP queues (the right fd) and what the teardown needs, and refuses
before the handoff and on the COT boot; fini{cpp_teardown} sends nothing to the GPU, drops NV from tinygrad's atexit list,
and holds exactly when the C++ side's unload was not confirmed (or it asks to); an EOF while the state page says the C++
teardown was running holds, sending nothing; and TinyGPULog.h's lines are on disk when the process is killed with
SIGKILL right after writing them. No GPU, no TinyGPU socket.
    python test_c5.py"""
import os, sys, io, json, types, ctypes, struct, socket, tempfile, mmap, subprocess, contextlib, signal
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tgpaths
tgpaths.setup()
import test_p3 as p3   # its daemon rig (a real Daemon on a socketpair over a stub device that records its calls)
import nv_dispatch_daemon as d
from tinygrad import Device
from tinygrad.runtime.autogen import nv

def export_rig(fmc_boot=False, handed_off=True, teardown_images=True):
    a, dm, calls, _, _ = p3.rig()
    impl = dm.dev.iface.dev_impl
    mem = (ctypes.c_uint8 * 0x81000)()
    base = ctypes.addressof(mem)
    impl.gsp.cmd_q = types.SimpleNamespace(tx=types.SimpleNamespace(size=0x40000), seq=55)
    impl.gsp.cmd_q_view = types.SimpleNamespace(addr=base + 0x1000)   # init_rm_args's command queue view, past the page table
    impl.gsp.libos_args_sysmem, impl.gsp._mem = 0x80294000, mem
    impl.flcn = types.SimpleNamespace(desc_v3=nv.FALCON_UCODE_DESC_V3(IMEMPhysBase=0, IMEMVirtBase=0, IMEMLoadSize=0xc100, DMEMPhysBase=0,
                                                                      DMEMLoadSize=0x3e00, PKCDataOffset=0xb24, EngineIdMask=0x400, UcodeId=9))
    if teardown_images:
        impl.flcn.beagle_sb_image_paddr, impl.flcn.beagle_unload_image_paddr = 0x121f000, 0x1230000
        impl.flcn.beagle_unload_params = (0x5000, 0x4e00, 0x100, 0x4f00)
    impl.fmc_boot, impl.fw_name, impl.chip_id, impl.chip_name = fmc_boot, "gb202" if fmc_boot else "ad102", 0x197000a1, "GB205" if fmc_boot else "AD107"
    queues = tempfile.TemporaryFile()
    dm.dev.iface.pci_dev = types.SimpleNamespace(sysmem_fds={base: queues.fileno(), base + 0x100000: 99})
    dm.handed_off = handed_off
    return a, dm, calls, queues

def daemon_reply(dm, a, reqs):
    """The daemon runs each request, then the plugin goes away; the replies and fds it sent, and whether it held."""
    for r in reqs: a.sendall(p3.msg(r))
    a.shutdown(socket.SHUT_WR)
    held = False
    try:
        with contextlib.redirect_stderr(io.StringIO()): dm.run()
    except p3.Held: held = True
    dm.sock.close()
    out = []
    a.settimeout(5)
    while True:
        hdr, fds, _, _ = socket.recv_fds(a, 4, 4)
        if len(hdr) < 4: break
        body = a.recv(struct.unpack("<I", hdr)[0], socket.MSG_WAITALL)
        r = json.loads(body)
        if r.get("gsp_cmdq_off") is not None:   # teardown_export: one byte with the queue fd follows the reply
            _, fds, _, _ = socket.recv_fds(a, 1, 4)
        out.append((r, fds))
    return out, held

def test_teardown_export():
    real, Device._opened_devices = Device._opened_devices, set()
    try:
        a, dm, calls, queues = export_rig()
        Device._opened_devices.add("NV")
        (r, fds), (fini, _) = daemon_reply(dm, a, [{"cmd": "teardown_export"}, {"cmd": "fini", "cpp_teardown": True, "diag": {"unload_ok": True}}])[0]
        assert r["ok"] and (r["gsp_cmdq_off"], r["gsp_statq_off"], r["gsp_queue_size"], r["gsp_queues_size"]) == (0x1000, 0x41000, 0x40000, 0x81000), r
        assert (r["gsp_seq"], r["libos_args_sysmem"], r["chip_id"], r["fw_name"], r["teardown"], r["unload_level0"]) == \
               (55, 0x80294000, 0x197000a1, "ad102", True, False), r
        assert (r["sb_paddr"], r["sb_imem_sz"], r["sb_dmem_sz"], r["sb_pkc_off"], r["sb_engid"], r["sb_ucodeid"]) == (0x121f000, 0xc100, 0x3e00, 0xb24, 0x400, 9), r
        assert (r["unload_paddr"], r["unload_data_off"], r["unload_data_sz"], r["unload_code_off"], r["unload_code_sz"]) == (0x1230000, 0x5000, 0x4e00, 0x100, 0x4f00), r
        assert len(fds) == 1 and os.fstat(fds[0]).st_ino == os.fstat(queues.fileno()).st_ino, fds   # the queues' own fd
        assert fini["ok"] and not fini.get("hold") and calls == [] and dm.dev is None and "NV" not in Device._opened_devices, (fini, calls)
        for kw, why in ((dict(handed_off=False), "before the handoff"), (dict(fmc_boot=True), "COT boot")):
            a, dm, calls, _ = export_rig(**kw)
            (r, fds), = daemon_reply(dm, a, [{"cmd": "teardown_export"}])[0][:1]
            assert not r["ok"] and why in r["error"] and fds == [], (kw, r)
        a, dm, _, _ = export_rig(teardown_images=False)   # BEAGLE_NV_TEARDOWN=0 left no images: the unload only
        (r, fds), = daemon_reply(dm, a, [{"cmd": "teardown_export"}])[0][:1]
        assert r["ok"] and r["teardown"] is False and "sb_paddr" not in r and len(fds) == 1, r
    finally: Device._opened_devices = real
    print("teardown export: the GSP queues' own fd with init_rm_args's offsets, the command queue's seq, libos_args_sysmem, chip_id "
          "and both images' execute_hs arguments; refused before the handoff and on the COT boot; no images when the teardown is off")

def test_cpp_fini():
    real, Device._opened_devices = Device._opened_devices, set()
    try:
        confirmed = {"unload_ok": True, "mailbox0": 0x80000000, "wpr2_lo": 0x1ffffe00, "wpr2_hi": 0, "riscv_cpuctl": 0x10,
                     "teardown": {"result": "done: Booter Unload lowered WPR2"}, "wpr2_down": True, "teardown_ok": True}
        for diag, hold_asked, want_hold in ((confirmed, False, False), ({"unload_ok": False}, True, True), ({"unload_ok": False}, False, True),
                                            ({**confirmed, "teardown_ok": False, "teardown": {"result": "failed: x"}}, False, False)):
            a, dm, calls, _, _ = p3.rig(signal=7); p3.with_page(dm, 0, 7, phase=d._PHASE_TEARDOWN)
            Device._opened_devices.add("NV")
            replies, held = daemon_reply(dm, a, [{"cmd": "fini", "cpp_teardown": True, "hold": hold_asked, "diag": diag}])
            (r, _), = replies
            assert held is want_hold and calls == (["hold"] if want_hold else []), (diag, calls)   # no synchronize, unload or falcon call
            assert r["ok"] and bool(r.get("hold")) is want_hold and (not want_hold or r["pid"] == os.getpid()), r
            assert all(r[k] == v for k, v in diag.items()) and "NV" not in Device._opened_devices and dm.dev is None, r
    finally: Device._opened_devices = real
    print("fini{cpp_teardown}: nothing sent to the GPU, NV dropped from atexit; the report passed back; holds when the unload was "
          "not confirmed or the C++ side asks, never after a confirmed one (a failed falcon step closes normally)")

def test_eof_mid_teardown():
    real, Device._opened_devices = Device._opened_devices, set()
    try:
        a, dm, calls, _, _ = p3.rig(signal=7); p3.with_page(dm, 0, 7, phase=d._PHASE_TEARDOWN); held, log = p3.run(a, dm)
        assert held and calls == ["hold"] and "did not finish" in log, (calls, log)
        a, dm, calls, _, _ = p3.rig(signal=7); p3.with_page(dm, 0, 7, phase=d._PHASE_TEARDOWN)
        replies, held = daemon_reply(dm, a, [{"cmd": "fini"}])
        assert held and calls == ["hold"] and replies[0][0]["hold"], (calls, replies)
    finally: Device._opened_devices = real
    print("EOF or a plain fini while the state page says the C++ teardown was running: hold, nothing sent to the GPU")

LOG_PROBE = r'''
#include "libhmsbeagle/GPU/TinyGPULog.h"
#include <signal.h>
int main(int, char** argv) {
    int n = atoi(argv[1]);
    for (int i = 0; i < n; ++i) tinygpu_device::tg_log("line %d of %d", i + 1, n);
    kill(getpid(), SIGKILL);   // no exit handler, no stdio flush: only what reached the disk stays
    return 0;
}
'''

def test_log_survives_kill():
    src, exe = tgpaths.WORK / "c5_log_probe.cpp", tgpaths.WORK / "c5_log_probe"
    src.write_text(LOG_PROBE)
    tgpaths.build_cpp(src, exe)
    log = tgpaths.WORK / "c5_log_probe.log"
    if log.exists(): log.unlink()
    for n in (1, 50):
        r = subprocess.run([str(exe), str(n)], env=dict(os.environ, BEAGLE_TINYGPU_LOG=str(log)), capture_output=True)
        assert r.returncode == -signal.SIGKILL, r
    lines = log.read_text().splitlines()
    want = ["line 1 of 1"] + [f"line {i} of 50" for i in range(1, 51)]
    assert [l.split("] ", 1)[1] for l in lines] == want, lines[:3]
    assert all(l[:4].isdigit() and "[" in l for l in lines), lines[0]   # time and pid on every line
    print(f"TinyGPULog.h: all {len(lines)} lines on disk after SIGKILL (O_APPEND | O_SYNC, fsync per line), appended across runs, "
          "each with its time and pid")

if __name__ == "__main__":
    test_teardown_export()
    test_cpp_fini()
    test_eof_mid_teardown()
    test_log_survives_kill()
    print("C5 daemon and log: all passed")
