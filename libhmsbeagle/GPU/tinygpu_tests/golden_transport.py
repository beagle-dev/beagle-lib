"""Golden test for TinyGPUTransport.h (TODO.md plan step C3) against the tinygrad code it ports, request for request:
tinygrad's APLRemotePCIDevice (hcq1, system.py:311-447) and golden_transport.cpp make the same calls, in turn, against a
scripted fake TinyGPU.app that records every byte each client sends. The two recordings must be identical, and so must
the results, error replies included: config reads and writes, write_config_flush, MAP_BAR (cached), MMIO writes and
reads, an MMIO read the server refuses (status 1, no data), an RPC error with a message, RESIZE_BAR, and MAP_SYSMEM_FD
with its fd and segment list. No TinyGPU.app process and no eGPU: a private socket and TMPDIR."""
import os, sys, socket, struct, subprocess, tempfile, threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tgpaths
tgpaths.setup()
from tinygrad.runtime.support.system import APLRemotePCIDevice, RemoteCmd

HERE, WORK = str(tgpaths.HERE), str(tgpaths.WORK)
REQ, RESP = "<BIIQQQ", "<BQQ"
BARS = {0: (0x7000_0000, 16 << 20), 1: (0x7100_0000, 256 << 20)}

def recv_exact(conn, n):
    b = bytearray()
    while len(b) < n:
        chunk = conn.recv(n - len(b))
        if not chunk: return None
        b += chunk
    return bytes(b)

def serve(conn, rec, tmpdir):
    """One client: every byte it sends goes to rec; the replies are scripted."""
    nsys = 0
    while (hdr := recv_exact(conn, 33)) is not None:
        rec += hdr
        cmd, _, bar, a0, a1, _ = struct.unpack(REQ, hdr)
        if cmd == RemoteCmd.MMIO_WRITE: rec += recv_exact(conn, a1); continue
        if cmd == RemoteCmd.CFG_READ and a0 == 0xbad:
            msg = b"scripted error"; conn.sendall(struct.pack(RESP, 1, len(msg), 0) + msg)
        elif cmd == RemoteCmd.CFG_READ: conn.sendall(struct.pack(RESP, 0, 0x288210de if a0 == 0 else 0x6, 0))
        elif cmd in (RemoteCmd.CFG_WRITE, RemoteCmd.RESIZE_BAR): conn.sendall(struct.pack(RESP, 0, 0, 0))
        elif cmd == RemoteCmd.MAP_BAR: conn.sendall(struct.pack(RESP, 0, *BARS[bar]))
        elif cmd == RemoteCmd.MMIO_READ and a0 == 0xdead000: conn.sendall(struct.pack(RESP, 1, 0, 0))   # as server.c: no data
        elif cmd == RemoteCmd.MMIO_READ: conn.sendall(struct.pack(RESP, 0, a1, 0) + bytes((a0 + i) & 0xff for i in range(a1)))
        elif cmd == RemoteCmd.MAP_SYSMEM_FD:   # as server.c: page-aligned, 16 KiB minimum, the segment list at the start
            size = max((a0 + 0xfff) & ~0xfff, 0x4000)
            fd = os.open(f"{tmpdir}/sysmem_{id(rec)}_{nsys}", os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
            os.ftruncate(fd, size)
            os.pwrite(fd, struct.pack("<6Q", 0x2_0000_0000, 0x4000, 0x2_0010_0000, size - 0x4000, 0, 0), 0)
            socket.send_fds(conn, [struct.pack(RESP, 0, size, nsys)], [fd])
            os.close(fd); nsys += 1
        else: conn.sendall(struct.pack(RESP, 1, 0, 0))
    conn.close()

def tinygrad_device(sock_path):
    """An APLRemotePCIDevice for "NV:0" as its __init__ builds it (system.py:428-438, then :387-392), which the offline guard
    blocks (tgpaths.block_real_devices): the socket is the fake's, and every request after it comes from tinygrad's own
    methods."""
    from tinygrad.runtime.support.system import System
    dev = object.__new__(APLRemotePCIDevice)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(sock_path)
    dev.sock, dev.pcibus, dev.dev_id = sock, "usb4", 0
    dev.peer_group = sock.getpeername()[0]
    for buft in [socket.SO_SNDBUF, socket.SO_RCVBUF]: dev.sock.setsockopt(socket.SOL_SOCKET, buft, 64 << 20)
    dev.lock_fd = System.flock_acquire("nv_usb4.lock")
    return dev

def tinygrad_client(results, sock_path):
    dev = tinygrad_device(sock_path)
    out = results.append
    out(f"cfg0={dev.read_config(0, 4):#x}")
    dev.write_config(4, 0x6, 2)
    dev.write_config_flush(4, 0x7, 2)
    out(f"bar0={dev.bar_info(0)[1]:#x}"); out(f"bar0={dev.bar_info(0)[1]:#x}"); out(f"bar1={dev.bar_info(1)[1]:#x}")
    dev._bulk_write(RemoteCmd.MMIO_WRITE, 1, 0x1000, bytes(range(32)))
    out(f"read={dev._bulk_read(RemoteCmd.MMIO_READ, 1, 0x2000, 64).hex()}")
    for what, call in (("read error", lambda: dev._bulk_read(RemoteCmd.MMIO_READ, 1, 0xdead000, 16)), ("cfg error", lambda: dev.read_config(0xbad, 4))):
        try: call(); out(f"{what}=none")
        except RuntimeError as e: out(f"{what}={e}")
    out(f"cfg0={dev.read_config(0, 4):#x}")
    dev.resize_bar(1)
    view, paddrs = dev.alloc_sysmem(0x5000)
    out(f"sysmem={len(view):#x} " + " ".join(f"{p:#x}" for p in paddrs))
    dev.sock.close()
    os.close(dev.lock_fd)   # tinygrad keeps its lock for the process's life; the C++ client takes it next

def main():
    exe = f"{WORK}/golden_transport"
    tgpaths.build_cpp(f"{HERE}/golden_transport.cpp", exe)
    priv = tempfile.mkdtemp(dir="/tmp", prefix="tggt.")
    sock_path = f"{priv}/s.sock"
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path); srv.listen(1); srv.settimeout(60)
    recs = {"tinygrad": bytearray(), "c++": bytearray()}
    def accept_one(key):
        conn, _ = srv.accept(); conn.settimeout(60); serve(conn, recs[key], priv)
    # tinygrad's client, in this process: the private socket, and its lock in the private TMPDIR
    tempfile.tempdir = priv
    t = threading.Thread(target=accept_one, args=("tinygrad",), daemon=True); t.start()
    ref = []
    tinygrad_client(ref, sock_path)
    t.join(timeout=10)
    t = threading.Thread(target=accept_one, args=("c++",), daemon=True); t.start()
    r = subprocess.run([exe], capture_output=True, text=True, timeout=60,
                       env=dict(os.environ, TMPDIR=priv, APL_REMOTE_SOCK=sock_path, BEAGLE_TINYGPU_NO_LAUNCH="1"))
    t.join(timeout=10)
    got = r.stdout.splitlines()
    ok = r.returncode == 0 and got == ref and recs["tinygrad"] == recs["c++"]
    if not ok:
        print(f"c++ exit {r.returncode}: {r.stderr}")
        for a, b in zip(ref + [""] * len(got), got + [""] * len(ref)):
            if a != b: print(f"  tinygrad: {a}\n  c++     : {b}")
        print(f"  request bytes: tinygrad {len(recs['tinygrad'])}, c++ {len(recs['c++'])}, identical {recs['tinygrad'] == recs['c++']}")
    print(f"TinyGPUTransport.h vs APLRemotePCIDevice: {len(ref)} results, {len(recs['tinygrad'])} request bytes: "
          f"{'IDENTICAL' if ok else 'MISMATCH'}")
    ok &= daemon_adopts_connection()
    sys.exit(0 if ok else 1)

def daemon_adopts_connection():
    """The daemon adopts the C++ side's connection (its inherited-fd device, nv_dispatch_daemon._install_inherited_tinygpu),
    whose buffers TinyGPUTransport.h's open() already set, as RemotePCIDevice.__init__ does for the process that connects.
    macOS refuses a second setting (ENOBUFS), which failed a hardware boot before any GPU access (STATUS.md R30)."""
    import nv_dispatch_daemon as d
    from tinygrad.runtime.support import system
    a, b = socket.socketpair()
    for o in (socket.SO_SNDBUF, socket.SO_RCVBUF): a.setsockopt(socket.SOL_SOCKET, o, 64 << 20)
    saved = system.APLRemotePCIDevice
    try:
        d._install_inherited_tinygpu(a.fileno())
        system.APLRemotePCIDevice("NV", "usb4").sock.close()   # the daemon's own __init__, which the offline guard lets through
        ok, why = True, ""
    except OSError as e: ok, why = False, str(e)
    finally: system.APLRemotePCIDevice = saved; a.close(); b.close()
    print(f"the daemon adopts a connection whose buffers the C++ client set: {'PASS' if ok else 'FAIL: ' + why}")
    return ok

main()
