"""TODO.md plan step V1, offline: the recording proxy (tgproxy.py) between a stress client and TinyGPU.app's real
server.c (compiled from the tinygrad pin with ../server_stub.c in place of IOKit, as test_c3_transport.sh does; never
copied here). No eGPU and no TinyGPU.app process: private sockets and a private TMPDIR.

A byte tee in front of the server logs every byte it receives, so "the server receives exactly the client's bytes,
less what the proxy answers itself" is checked byte for byte. Checks:
  - every forwarded command (CFG_READ/WRITE, MAP_BAR incl. a refused BAR, RESIZE_BAR, MMIO_WRITE and MMIO_READ from 1 B
    to 64 MB, a refused MMIO_READ, MAP_SYSMEM_FD with its fd and segment list, a failed allocation with no fd) answers as
    the server answers it directly, and the recording holds each request and reply;
  - headers and payloads sent in pieces arrive upstream whole (server.c reads a header with one recv);
  - RESET, PROBE (with its payload), MAP_SYSMEM, SYSMEM_READ, SYSMEM_WRITE (with its payload) and PING are answered with
    an error, never forwarded (a RESET would abort the stub), and the stream stays in step; a marker is answered by the
    proxy and never forwarded;
  - GPU-side writes into a mapping between requests are recorded as diffs at the next request (before and after), for
    a small mapping and for the status half of a queue-sized one, and a trigger write records the changed pages first;
  - fail-stop: a client killed inside a payload, an unknown command and an over-64 MB write each stop the proxy with
    nothing forwarded after, and the server's connection stays open until the proxy is killed; a client that closes at
    a request boundary makes the proxy close upstream at once.
    <tinygrad venv>/python test_v1_proxy.py"""
import os, sys, time, mmap, json, shutil, socket, signal, struct, select, pathlib, tempfile, subprocess, threading
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import tgpaths
tgpaths.setup()
import tgwire as w

WORK = tgpaths.WORK / "v1_proxy"
PY = sys.executable
fails = []
def check(ok, what, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {what}{'' if ok or not detail else ': ' + str(detail)}", flush=True)
    if not ok: fails.append(what)

def build_stub():
    server_c = pathlib.Path(tgpaths.TINYGRAD_PATH) / "extra/usbgpu/tbgpu/installer/Shared/server.c"
    exe = WORK / "server_stub"
    subprocess.run(["cc", "-O1", "-Wno-deprecated-declarations", "-Wno-address-of-packed-member", "-o", str(exe), str(server_c),
                    str(HERE.parent / "server_stub.c")], check=True)
    return exe

def wait_for(pred, timeout=10.0, step=0.02):
    end = time.time() + timeout
    while time.time() < end:
        if pred(): return True
        time.sleep(step)
    return pred()

class Procs:
    def __init__(self): self.procs = []
    def start(self, args, log, **kw):
        p = subprocess.Popen(args, stdout=open(log, "w"), stderr=subprocess.STDOUT, **kw)
        self.procs.append(p)
        return p
    def kill_all(self):
        for p in self.procs:
            if p.poll() is None: p.kill()
            p.wait()

def tee(listen_path, upstream_path, log_path, ready):
    """A byte pipe that logs what goes upstream and passes fds along with their bytes (what the server receives); one
    connection after another, like the server behind it. Only the first connection is logged."""
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); srv.bind(listen_path); srv.listen(1); ready.set()
    for n in range(100):
        c, _ = srv.accept()
        pipe(c, upstream_path, log_path if n == 0 else os.devnull)

def pipe(c, upstream_path, log_path):
    u = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); u.connect(upstream_path)
    with open(log_path, "wb") as log:
        while True:
            r, _, _ = select.select([c, u], [], [])
            if c in r:
                d = c.recv(1 << 20)
                if not d: break
                log.write(d); log.flush(); u.sendall(d)
            if u in r:
                d, anc, _, _ = u.recvmsg(1 << 20, socket.CMSG_SPACE(4 * 4))
                if not d: break
                fds = [f for lv, ty, data in anc if lv == socket.SOL_SOCKET and ty == socket.SCM_RIGHTS
                       for f in struct.unpack(f"{len(data) // 4}i", data[:len(data) // 4 * 4])]
                c.sendmsg([d], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, struct.pack(f"{len(fds)}i", *fds))] if fds else [])
                for f in fds: os.close(f)
    c.close(); u.close()

class Client:
    """The raw protocol, as tinygrad's RemotePCIDevice speaks it (system.py:374-399), with split sends on request (only
    through the proxy: server.c itself reads a header with one recv and stops on a cut one); logs every byte it sends that
    the server must receive (all but markers and refused commands)."""
    def __init__(self, path, split_ok):
        self.s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); self.s.connect(path)
        self.expect_up, self.split_ok = bytearray(), split_ok
    def send(self, data, split=0, upstream=True):
        if upstream: self.expect_up += data
        if not split or not self.split_ok: self.s.sendall(data); return
        cuts = sorted({min(len(data), max(1, i * len(data) // (split + 1))) for i in range(1, split + 1)} | {len(data)})
        prev = 0
        for c in cuts:
            self.s.sendall(data[prev:c]); prev = c; time.sleep(0.02)
    def reply(self, fd=False):
        if fd:
            msg, anc, _, _ = self.s.recvmsg(17, socket.CMSG_SPACE(4))
            fds = [f for lv, ty, data in anc if lv == socket.SOL_SOCKET and ty == socket.SCM_RIGHTS for f in struct.unpack("i", data[:4])]
            if len(msg) < 17: msg += w.recv_exact(self.s, 17 - len(msg))
        else: msg, fds = w.recv_exact(self.s, 17), []
        st, r0, r1 = w.RESP.unpack(msg)
        extra = w.recv_exact(self.s, r0) if st != 0 and r0 else b""
        return st, r0, r1, (fds[0] if fds else None), extra
    def rpc(self, cmd, a0=0, a1=0, a2=0, bar=0, dev=0, split=0, upstream=True, readout=0, fd=False, payload=b""):
        self.send(w.REQ.pack(cmd, dev, bar, a0, a1, a2) + payload, split, upstream)
        st, r0, r1, f, extra = self.reply(fd)
        data = w.recv_exact(self.s, readout) if st == 0 and readout else b""
        return st, r0, r1, f, extra, data
    def write(self, bar, off, data, split=0): self.send(w.REQ.pack(w.MMIO_WRITE, 0, bar, off, len(data), 0) + data, split)
    def close(self): self.s.close()

def session(c, results):
    """The stress session: returns a list of (label, observable result) that must be equal with and without the proxy."""
    out = results.append
    st, r0, *_ = c.rpc(w.CFG_READ, 0, 4); out(("cfg_read", st, r0))
    st, *_ = c.rpc(w.CFG_WRITE, 4, 2, 6); out(("cfg_write", st))
    for bar in (0, 1, 5):
        st, r0, r1, *_ = c.rpc(w.MAP_BAR, bar=bar); out(("map_bar", bar, st, r1))   # the address (r0) is the server's own mmap
    st, *_ = c.rpc(w.RESIZE_BAR, bar=1); out(("resize_bar", st))
    st, r0, r1, *_ = c.rpc(w.CFG_READ, 0, 4, split=3); out(("cfg_read split header", st, r0))
    for n in (1, 3, 4096, 1 << 20, (64 << 20)):
        data = (bytes((i * 131 + n) & 0xff for i in range(min(n, 1 << 16))) * ((n + 0xffff) >> 16))[:n]
        off = 0 if n == (64 << 20) else 0x1000
        c.write(1, off, data, split=4 if n == (1 << 20) else 0)
        st, r0, r1, f, extra, back = c.rpc(w.MMIO_READ, off, n, bar=1, readout=n)
        out((f"mmio {n} B round trip", st, r0, back == data))
    st, r0, r1, f, extra, back = c.rpc(w.MMIO_READ, (64 << 20) - 8, 16, bar=1, readout=16); out(("mmio read past the BAR", st, r0, len(back)))
    c.write(0, 0x1000, struct.pack("<I", 0x12345678)); st, *_, back = c.rpc(w.MMIO_READ, 0x1000, 4, bar=0, readout=4); out(("bar0 word", st, back.hex()))
    maps = []
    for size in (1, 0x4000, 0x81000, 2 << 20, 0x777000):
        st, r0, r1, f, extra, _ = c.rpc(w.MAP_SYSMEM_FD, size, 0, fd=True)
        if f is None: out(("map_sysmem_fd", size, st, r0, None)); continue
        mm = mmap.mmap(f, r0, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE); os.close(f)
        segs, _ = w.segments(mm[:8192])
        out(("map_sysmem_fd", size, st, r0, r1, [sz for _, sz in segs]))   # device addresses are the stub's own
        maps.append((size, mm))
    return maps

def run(path, results, with_proxy_extras, gpu_writes=None):
    c = Client(path, split_ok=with_proxy_extras)
    maps = session(c, results)
    if with_proxy_extras:   # answered by the proxy: refusals, then a marker; the stream stays in step after each
        for cmd, payload in ((w.RESET, b""), (w.PROBE, b"\x10\xde\x00\x00" * 2), (w.MAP_SYSMEM, b""), (w.SYSMEM_READ, b""),
                             (w.SYSMEM_WRITE, b"abcd"), (w.PING, b"")):
            st, r0, r1, f, extra, _ = c.rpc(cmd, 0, len(payload), payload=payload, upstream=False)
            st2, v, *_ = c.rpc(w.CFG_READ, 0, 4)
            results.append((f"refused {w.cmd_name(cmd)}", st, b"refused" in extra, st2, v))
        st, r0, *_ = c.rpc(w.CFG_READ, 0, 4, a2=7, bar=3, dev=w.MARKER_DEV, upstream=False)
        results.append(("marker", st, r0))
    if gpu_writes: gpu_writes(c, maps, results)
    c.close()
    for _, mm in maps: mm.close()
    return c.expect_up

def gpu_side(c, maps, results):
    """Plays the GPU: writes into two mappings between requests, then a trigger write."""
    small = next(mm for size, mm in maps if size == 0x4000)
    queues = next(mm for size, mm in maps if size == 0x81000)
    st, *_ = c.rpc(w.CFG_READ, 0, 4)                 # request A
    small[0x2000:0x2008] = b"GPUWROTE"                # "the GPU" writes between A and B
    queues[0x41000:0x41020] = bytes(range(32))        # the status half of the queue mapping
    queues[0x1000:0x1010] = b"CPU-SIDE-CMD-Q!!"       # the command half: not watched per request
    st, *_ = c.rpc(w.CFG_READ, 0, 4)                 # request B: the diffs are recorded at its seq
    queues[0x2000:0x2004] = b"PTE!"                   # a CPU write the next trigger's pages must show
    c.write(0, 0xb830b0, struct.pack("<I", 0x80000043))   # NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE: a trigger
    st, *_ = c.rpc(w.CFG_READ, 0, 4)
    small[0x3000:0x3004] = b"LAST"                    # after the last trigger: only the session's final snapshot sees it
    results.append(("gpu side", st))

def start_proxy(procs, name, upstream, extra=()):
    sock, out, log = str(PRIV / f"{name}.sock"), WORK / f"rec_{name}", WORK / f"proxy_{name}.log"
    shutil.rmtree(out, ignore_errors=True)
    p = procs.start([PY, str(HERE / "tgproxy.py"), "--listen", sock, "--upstream", upstream, "--out", str(out), *extra], log)
    ok = wait_for(lambda: "tgproxy listening" in log.read_text())
    if not ok: raise SystemExit(f"proxy {name} did not start: {log.read_text()}")
    return p, sock, out, log

def start_stub(procs, name):
    sock, log = str(PRIV / f"{name}_srv.sock"), WORK / f"stub_{name}.log"
    p = procs.start([str(STUB), sock], log)
    if not wait_for(lambda: os.path.exists(sock)): raise SystemExit(f"stub {name} did not start")
    return p, sock, log

def main():
    global PRIV, STUB
    WORK.mkdir(parents=True, exist_ok=True)
    # the stub names its shared memory as TinyGPU.app does (/tinygpu_N): never beside a hardware run (env.sh hw_begin's lock)
    hw_lock = pathlib.Path(os.environ.get("TMPDIR", "/tmp")) / "beagle_tinygpu_hw.lock"
    if hw_lock.is_dir(): sys.exit(f"a hardware run holds {hw_lock}; not running")
    STUB = build_stub()
    PRIV = pathlib.Path(tempfile.mkdtemp(dir="/tmp", prefix="tgv1."))
    procs = Procs()
    try:
        # 1. the same session directly (through the tee only) and through the proxy
        direct, via = [], []
        stub, ssock, slog = start_stub(procs, "direct")
        ready = threading.Event(); tlog = WORK / "tee_direct.bin"
        threading.Thread(target=tee, args=(str(PRIV / "tee_d.sock"), ssock, tlog, ready), daemon=True).start(); ready.wait()
        sent_direct = run(str(PRIV / "tee_d.sock"), direct, False)
        wait_for(lambda: "client disconnected" in slog.read_text(), 5)
        check(tlog.read_bytes() == bytes(sent_direct), "the tee passes the direct session through unchanged")

        stub, ssock, slog = start_stub(procs, "via")
        ready = threading.Event(); tlog = WORK / "tee_via.bin"
        threading.Thread(target=tee, args=(str(PRIV / "tee_v.sock"), ssock, tlog, ready), daemon=True).start(); ready.wait()
        proxy, psock, rec, plog = start_proxy(procs, "stress", str(PRIV / "tee_v.sock"))
        sent_via = run(psock, via, True, gpu_side)
        check(wait_for(lambda: "session 1 ended: eof" in plog.read_text(), 10) and proxy.poll() is None,
              "the proxy ends a session when its client closes, and waits for the next")
        check(wait_for(lambda: "client disconnected" in slog.read_text(), 5), "the proxy closed upstream at once after the client's close")
        c2 = Client(psock, False); st2, v2, *_ = c2.rpc(w.CFG_READ, 0, 4); c2.close()
        check(st2 == 0 and v2 == 0x288210de and wait_for(lambda: "session 2 ended: eof" in plog.read_text(), 10),
              "a second client is a second session, with its own upstream connection")
        proxy.send_signal(signal.SIGTERM); proxy.wait(timeout=30)
        check(proxy.returncode == 0 and "recording ended after 2 session(s)" in plog.read_text(), "SIGTERM between sessions ends the recording",
              proxy.returncode)
        lone, lsock, lrec, llog = start_proxy(procs, "lone", str(PRIV / "nothing.sock"))
        c3 = Client(lsock, False); c3.send(w.REQ.pack(w.CFG_READ, 0, 0, 0, 4, 0)); got = c3.s.recv(17); c3.close()
        check(got == b"" and wait_for(lambda: "session 1 ended: no-upstream" in llog.read_text(), 5) and lone.poll() is None,
              "with nothing upstream a session is closed (EOF to the client), not held, and the proxy serves on")
        lone.send_signal(signal.SIGTERM); lone.wait(timeout=30)
        check(tlog.read_bytes() == bytes(sent_via), "the server receives exactly the client's bytes, less refusals and the marker",
              f"{len(tlog.read_bytes())} vs {len(sent_via)} bytes")
        n = len(direct)
        check(via[:n] == direct, "every forwarded command answers as without the proxy",
              next((f"{a} vs {b}" for a, b in zip(direct, via) if a != b), ""))
        extras = {r[0]: r for r in via[n:]}
        for name in ("RESET", "PROBE", "MAP_SYSMEM", "SYSMEM_READ", "SYSMEM_WRITE", "PING"):
            r = extras.get(f"refused {name}")
            check(r is not None and r[1] == 1 and r[2] and r[3] == 0 and r[4] == 0x288210de,
                  f"{name} is refused with an error, never forwarded, and the stream stays in step", r)
        check(extras.get("marker") == ("marker", 0, 0), "a marker is answered by the proxy (resp0 0, not the server's 10de:2882)", extras.get("marker"))
        check("RESET received" not in slog.read_text(), "no RESET reached the server")

        # 2. the recording
        ev, blobs = w.read(rec)
        m = w.meta(rec)
        reqs = [e for e in ev if e.kind == w.K_REQ]
        ends = [e.f for e in ev if e.kind == w.K_NOTE and e.f.get("event") == "session end"]
        check(m.get("end", {}).get("how") == "stopped" and [x["how"] for x in ends] == ["eof", "eof"] and m["counts"]["MARKER"] == 1
              and m["counts"]["REFUSED"] == 6, "the recording holds both sessions, 1 marker and 6 refusals", (m.get("counts"), ends))
        s1_end = ends[0]["final_pages_seq"]
        rebuilt = b"".join(e.f["hdr"] + e.f["payload"] for e in reqs if e.seq < s1_end)
        check(rebuilt == bytes(sent_via), "the recording's first session is exactly the bytes forwarded")
        sysm = [e for e in ev if e.kind == w.K_SYSMEM]
        check([e.f["size"] for e in sysm] == [1, 0x4000, 0x81000, 2 << 20] and all(e.f["segs"] for e in sysm),
              "each successful MAP_SYSMEM_FD is recorded with its segment list (the failed one is not)", [e.f["size"] for e in sysm])
        failed = [e for e in ev if e.kind == w.K_REPLY and e.f["reply"][0] != 0]
        check(len(failed) >= 3, "failed replies are recorded (a refused BAR, a read past the BAR, a failed allocation)", len(failed))
        big = next(e for e in reqs if e.f["req"][0] == w.MMIO_WRITE and e.f["req"][4] == 64 << 20)
        check(len(big.f["payload"]) == 64 << 20, "a 64 MB payload is kept whole (in blobs.bin)")
        by_seq = {e.seq: e for e in reqs}
        diffs = [e for e in ev if e.kind == w.K_DIFF]
        small_alloc = next(e.f["alloc"] for e in sysm if e.f["size"] == 0x4000)
        q_alloc = next(e.f["alloc"] for e in sysm if e.f["size"] == 0x81000)
        d_small = [d for d in diffs if d.f["alloc"] == small_alloc]
        d_q = [d for d in diffs if d.f["alloc"] == q_alloc]
        check(len(d_small) == 1 and d_small[0].f["off"] <= 0x2000 and b"GPUWROTE" in d_small[0].f["after"] and b"GPUWROTE" not in d_small[0].f["before"],
              "a write into a small mapping between two requests is a diff (before, after) at the next request", d_small)
        check(len(d_q) == 1 and d_q[0].f["off"] == 0x41000 and d_q[0].f["after"][:32] == bytes(range(32)),
              "a write into the status half of the queue mapping is a diff; the command half is not diffed per request", d_q)
        trig = next(e for e in reqs if e.f["req"][0] == w.MMIO_WRITE and e.f["req"][3] == 0xb830b0)
        check(bool(d_small) and d_small[0].seq == trig.seq - 1 == d_q[0].seq,
              "the diffs carry the seq of the request after the write (B), not the one before it (A)")
        pages = [e for e in ev if e.kind == w.K_PAGES and e.seq == trig.seq + 1]   # at the next request the client waits on
        q_pages = next((p for p in pages if p.f["alloc"] == q_alloc), None)
        got = {pg: blobs.get(sha) for pg, sha in q_pages.f["pages"]} if q_pages else {}
        check(q_pages is not None and got.get(2, b"")[:4] == b"PTE!" and got.get(1, b"")[:16] == b"CPU-SIDE-CMD-Q!!"
              and not [e for e in ev if e.kind == w.K_PAGES and e.seq == trig.seq],
              "after a trigger write the changed pages are recorded at the next request the client waits on (content in blobs.bin)", sorted(got))
        final = {e.f["alloc"]: e.f["pages"] for e in ev if e.kind == w.K_PAGES and e.seq == s1_end}
        check([pg for pg, _ in final.get(small_alloc, [])] == [3] and blobs.get(final[small_alloc][0][1])[:4] == b"LAST",
              "the session's final page snapshot holds what changed after its last trigger", final.get(small_alloc))

        # 3. fail-stop cases: the server's connection must stay open after each
        for name, act in (("killed inside a payload", "kill"), ("an unknown command", "cmd13"), ("an MMIO_WRITE over 64 MB", "big")):
            stub, ssock, slog = start_stub(procs, f"fs_{act}")
            proxy, psock, rec, plog = start_proxy(procs, f"fs_{act}", ssock)
            code = f"""
import socket, struct, time, sys
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.connect({psock!r})
s.sendall(struct.pack('<BIIQQQ', 3, 0, 0, 0, 4, 0)); s.recv(17)
act = {act!r}
if act == 'kill':
    s.sendall(struct.pack('<BIIQQQ', 7, 0, 1, 0, 1 << 20, 0) + b'x' * 1000); print('sent', flush=True); time.sleep(60)
elif act == 'cmd13': s.sendall(struct.pack('<BIIQQQ', 13, 0, 0, 0, 0, 0)); time.sleep(2)
else: s.sendall(struct.pack('<BIIQQQ', 7, 0, 1, 0, (64 << 20) + 1, 0)); time.sleep(2)
"""
            cl = subprocess.Popen([PY, "-c", code], stdout=subprocess.PIPE, text=True)
            if act == "kill":
                cl.stdout.readline(); time.sleep(0.3); cl.kill()
            cl.wait()
            stopped = wait_for(lambda: "FAIL-STOP" in plog.read_text(), 10)
            time.sleep(1.5)
            held = proxy.poll() is None and "client disconnected" not in slog.read_text()
            rm = w.meta(rec)
            check(stopped and held and rm.get("end", {}).get("how") == "failstop" and proxy.poll() is None,
                  f"fail-stop on {name}: nothing forwarded after, the server's connection held", rm.get("end", {}).get("why", "")[:120])
            proxy.kill(); proxy.wait()
            check(wait_for(lambda: "client disconnected" in slog.read_text(), 5), f"({name}) the server sees the close only when the proxy is killed")
            if act == "big": check(not any(e.kind == w.K_REQ and e.f["req"][0] == w.MMIO_WRITE for e in w.read(rec)[0]),
                                   "the over-64 MB write was not forwarded")
    finally:
        procs.kill_all()
        shutil.rmtree(PRIV, ignore_errors=True)
    print(f"\ntest_v1_proxy: {'PASS' if not fails else f'{len(fails)} FAILED: ' + '; '.join(fails)}")
    sys.exit(1 if fails else 0)

if __name__ == "__main__":
    main()
