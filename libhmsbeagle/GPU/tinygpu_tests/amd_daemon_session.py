"""One session of the real AMD daemon on a fake card, driven as the plugin drives it (TODO.md plan step A2): the plugin's
TinyGPU.app connection to the fake (its first request the plugin's PCI id read), the command socket, amd_daemon_on_fake.py
spawned with both, then boot, handoff and
fini, each a length-prefixed JSON message as GPUInterfaceTinyGPUHybridAMD.cpp sends it. The daemon then exits, and
tinygrad's atexit hook finalizes the device (AMDev.fini). No compile_all: the plugin's build-time HSACOs need none. The
connection is closed last, which ends the session on the fake.
    session(fake_socket, pool_size, before_fini=None, env=None, daemon=None) -> (boot reply, handoff reply, its HSACO blob)
daemon: the script to spawn (default amd_daemon_on_fake.py); run_amd_l0.sh runs amd_dispatch_daemon.py itself on the eGPU:
    python amd_daemon_session.py --hw <socket> [pool size]"""
import os, sys, json, socket, struct, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tgpaths

def _recv_exact(s, n):
    b = bytearray()
    while len(b) < n:
        c = s.recv(n - len(b))
        if not c: raise RuntimeError(f"the daemon closed the command socket ({len(b)} of {n} bytes)")
        b += c
    return bytes(b)
def _send(s, obj):
    body = json.dumps(obj).encode()
    s.sendall(struct.pack("<I", len(body)) + body)
def _recv(s): return json.loads(_recv_exact(s, struct.unpack("<I", _recv_exact(s, 4))[0]))

def session(fake_socket, pool_size, before_fini=None, env=None, variant="SP_4", daemon=None):
    tg = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    tg.connect(fake_socket)
    # GPUInterface::Initialize's first request on the connection: the PCI id (config dword 0), before the daemon starts
    tg.sendall(struct.pack("<BIIQQQ", 3, 0, 0, 0, 4, 0))
    status, _, _ = struct.unpack("<BQQ", _recv_exact(tg, 17))
    if status != 0: raise RuntimeError("the PCI id read failed")
    mine, theirs = socket.socketpair()
    p = subprocess.Popen([os.environ.get("BEAGLE_PYTHON", sys.executable), str(daemon or tgpaths.HERE / "amd_daemon_on_fake.py"),
                          str(theirs.fileno()), str(tg.fileno())], pass_fds=(theirs.fileno(), tg.fileno()), env={**os.environ, **(env or {})})
    theirs.close()
    try:
        _send(mine, {"cmd": "boot"})
        boot = _recv(mine)
        if not boot.get("ok"): raise RuntimeError(f"boot: {boot}")
        _send(mine, {"cmd": "handoff", "pool_size": pool_size, "variant": variant})
        info = _recv(mine)
        if not info.get("ok"): raise RuntimeError(f"handoff: {info}")
        blob = _recv_exact(mine, info["blob_size"]) if info["blob_size"] else b""
        _, fds, _, _ = socket.recv_fds(mine, 1, 16)
        for fd in fds: os.close(fd)
        if before_fini: before_fini()
        _send(mine, {"cmd": "fini"})
        _recv(mine)
        if p.wait(timeout=120) != 0: raise RuntimeError(f"the daemon exited with {p.returncode}")
    finally:
        mine.close()
        if p.poll() is None: p.wait(timeout=120)
        tg.close()
    return boot, info, blob

if __name__ == "__main__":   # run_amd_l0.sh: the real daemon on the eGPU, through the given socket (tgproxy's)
    if len(sys.argv) < 3 or sys.argv[1] != "--hw": sys.exit("usage: amd_daemon_session.py --hw <socket> [pool size, default the daemon's]")
    boot, info, blob = session(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0, daemon=tgpaths.GPU_DIR / "amd_dispatch_daemon.py")
    print(f"session: boot {boot}; handoff: pool {info['pool_size'] >> 20} MiB at {info['pool_va']:#x}, {info['nmaps']} mappings, "
          f"timeline {info['timeline_value']}; fini acknowledged; the daemon exited", flush=True)
