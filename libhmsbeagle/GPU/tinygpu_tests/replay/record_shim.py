"""The recording shim (TODO.md plan step V1, inv:verification#2): with BEAGLE_TG_RECORD=1 (applied by tgdaemon.py), markers
from inside the daemon's boot, and a side log, for plan V1's recordings. It changes nothing the daemon does: each wrapper
calls what it wraps with the same arguments and returns its result or lets its exception through, and the only wire
traffic it adds is markers, which the recording proxy and the replay server answer themselves and never forward (and
which TinyGPU.app itself would answer as a PCI vendor-ID read: server.c never reads dev_id, :216-220).

A marker is a CFG_READ of offset 0, size 4, with dev_id 0x42454147 ('BEAG'), bar = its id and arg2 = its argument
(tgwire.MARKERS), sent on the daemon's TinyGPU.app connection (the plugin's, argv[2]) and its 17-byte reply read before
tinygrad's next request: the daemon is single-threaded and uses that connection only inside a command, while the plugin
waits for it. Markers:
  - entry and exit (argument | 1 << 63) of tinygrad's boot functions and BEAGLE's teardown, and the entry of the daemon's
    commands (their exit would come after the reply, when the plugin may already use the connection), as listed in
    tgwire.MARKERS, with the RPC's class or control command, or the falcon base, as the argument;
  - every status-queue message consumed: function << 32 | the read pointer after it (NVRpcQueue.read_resp, ip.py:63-80);
  - every time.sleep, in microseconds.
Side log (BEAGLE_TG_RECORD_LOG, default ~/Library/Logs/beagle_tg_record.jsonl; JSON lines, t = ns since install): each
marker with its name; each consumed message's function, length and sha256; wait_cond's polls (message, count, last value);
the handoff the daemon builds. Without a TinyGPU.app fd (the daemon path) markers go to the side log only."""
import os, sys, time, json, socket, struct, hashlib, functools, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import tgwire as w

_t0 = time.monotonic_ns()
_sleep = time.sleep
_state = {"sock": None, "disabled": False, "log": None}

def _log(**kv):
    f = _state["log"]
    if f is None: return
    try: f.write(json.dumps(dict(t=time.monotonic_ns() - _t0, **kv), default=str) + "\n")
    except Exception: pass

def _sock():
    if _state["sock"] is None and not _state["disabled"]:
        if len(sys.argv) < 3: _state["disabled"] = True; _log(kind="note", msg="no TinyGPU.app fd: markers go to this log only"); return None
        _state["sock"] = socket.socket(fileno=os.dup(int(sys.argv[2])))
    return _state["sock"]

def marker(mid, arg=0):
    """One marker on the wire (and in the side log). A failure stops markers, never the daemon."""
    _log(kind="marker", id=mid, arg=arg, name=w.marker_name(mid, arg))
    s = _sock()
    if s is None: return
    try:
        s.sendall(w.REQ.pack(w.CFG_READ, w.MARKER_DEV, mid, 0, 4, arg & 0xffffffffffffffff))
        if len(w.recv_exact(s, 17)) < 17: raise ConnectionError("no reply")
    except Exception as e:
        _state["disabled"] = True
        _log(kind="note", msg=f"markers stopped: {type(e).__name__}: {e}")

def _wrap(owner, name, mid, argf=None, exit_marker=True):
    orig = getattr(owner, name)
    @functools.wraps(orig)
    def wrapper(*a, **k):
        try: arg = argf(*a, **k) if argf else 0
        except Exception: arg = 0
        marker(mid, arg)
        try: return orig(*a, **k)
        finally:
            if exit_marker: marker(mid, arg | w.MARKER_EXIT)
    setattr(owner, name, wrapper)

def install():
    from tinygrad.runtime.support.nv import nvdev, ip
    from tinygrad.runtime.support.memory import MemoryManager
    from tinygrad.runtime import ops_nv
    import nv_dispatch_daemon as d
    path = os.environ.get("BEAGLE_TG_RECORD_LOG", str(pathlib.Path.home() / "Library/Logs/beagle_tg_record.jsonl"))
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    _state["log"] = open(path, "w", buffering=1)
    base = lambda self, b, *a, **k: b
    for owner, name, mid, argf in (
            (nvdev.NVDev, "__init__", 1, None), (nvdev.NVDev, "_early_ip_init", 2, None), (nvdev.NVDev, "_early_mmu_init", 3, None),
            (nvdev.NVDev, "fini", 22, None),
            (ip.NV_FLCN, "init_sw", 4, None), (ip.NV_FLCN, "init_hw", 5, None), (ip.NV_FLCN, "execute_hs", 6, base),
            (ip.NV_FLCN, "execute_dma", 7, base), (ip.NV_FLCN, "reset", 8, base), (ip.NV_FLCN, "fini_hw", 9, None),
            (ip.NV_FLCN_COT, "init_hw", 26, None), (ip.NV_FLCN_COT, "fini_hw", 27, None),
            (ip.NV_GSP, "init_sw", 10, None), (ip.NV_GSP, "init_hw", 11, None), (ip.NV_GSP, "init_golden_image", 12, None),
            (ip.NV_GSP, "fini_hw", 13, None), (ip.NV_GSP, "rpc_rm_alloc", 14, lambda self, hParent, hClass, *a, **k: hClass),
            (ip.NV_GSP, "rpc_rm_control", 15, lambda self, hObject, cmd, *a, **k: cmd), (ip.NV_GSP, "rpc_set_page_directory", 16, None),
            (ip.NV_GSP, "rpc_unloading_guest_driver", 17, None),
            (ip.NV_GSP, "run_cpu_seq", 18, lambda self, buf: len(buf) // 4),
            (MemoryManager, "map_range", 19, lambda self, vaddr, size, *a, **k: size),
            (ops_nv.NVDevice, "__init__", 20, None), (ops_nv.NVDevice, "_setup_gpfifos", 21, None),
            (d.Daemon, "cmd_boot", 23, None), (d.Daemon, "cmd_handoff", 24, None), (d.Daemon, "cmd_fini", 25, None)):
        # a daemon command replies to the plugin before it returns, and after the handoff's reply the plugin uses the TinyGPU.app
        # connection at once: an exit marker then would share it with the plugin (either could read the other's reply). At a
        # command's entry the plugin is waiting for that reply, so the entry marker is safe.
        _wrap(owner, name, mid, argf, exit_marker=not name.startswith("cmd_"))

    orig_read_resp = ip.NVRpcQueue.read_resp
    @functools.wraps(orig_read_resp)
    def read_resp(self):   # the same generator and items; a marker as each message is handed over
        for func, msg in orig_read_resp(self):
            try: rx = self.rx_view[0]
            except Exception: rx = 0
            _log(kind="message", function=func, length=len(msg), sha256=hashlib.sha256(msg).hexdigest(), rx=rx)
            marker(100, (func << 32) | rx)
            yield func, msg
    ip.NVRpcQueue.read_resp = read_resp

    orig_wait_cond = ip.wait_cond
    @functools.wraps(orig_wait_cond)
    def wait_cond(cb, *args, value=True, timeout_ms=10000, msg=""):
        n, last = [0], [None]
        def counted(*a):
            n[0] += 1; last[0] = cb(*a)
            return last[0]
        try: return orig_wait_cond(counted, *args, value=value, timeout_ms=timeout_ms, msg=msg)
        finally: _log(kind="wait_cond", msg=msg, polls=n[0], last=last[0], want=value)
    ip.wait_cond = wait_cond

    def sleep(secs):
        _log(kind="sleep", secs=secs)
        marker(101, int(secs * 1e6))
        return _sleep(secs)
    time.sleep = sleep

    orig_build = d.build_handoff
    @functools.wraps(orig_build)
    def build_handoff(dev, progs, bufs):
        info, blob = orig_build(dev, progs, bufs)
        _log(kind="handoff", info=info, blob_bytes=len(blob))
        return info, blob
    d.build_handoff = build_handoff
    _log(kind="note", msg="record shim installed", argv=sys.argv)
    print("record_shim: installed: markers on the TinyGPU.app connection, side log " + path, file=sys.stderr, flush=True)
