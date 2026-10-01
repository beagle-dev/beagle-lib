"""Recording proxy on the TinyGPU.app socket (TODO.md plan step V1; inv:verification#0, #1, #15).

A single-threaded lockstep forwarder between one client (the plugin, plus the daemon that inherits its connection) and
TinyGPU.app (or a fake). The client reaches it through APL_REMOTE_SOCK, which BEAGLE's C++ client and tinygrad both
read. It records every request and reply (tgwire.py's format), with sysmem snapshots: each MAP_SYSMEM_FD's DMA segment
list, a diff of the GSP status region and of every mapping up to 64 KiB at each request, and, after each trigger write (a
BAR0 write after which the GPU consumes what the CPU prepared), the pages of every mapping that changed, taken when the
client next sends a request it must wait for (before it is forwarded): the client can write nothing more until its reply,
so the snapshot is the same whatever the timing (at the trigger itself the client may still be writing, e.g. the next
queued message). Markers do not count, since the recording shim is optional.

Rules (inv:verification#0):
  - allow-list: MAP_BAR, MAP_SYSMEM_FD, CFG_READ, CFG_WRITE, MMIO_READ, MMIO_WRITE, RESIZE_BAR are forwarded. RESET (a PCIe
    function reset), PROBE, MAP_SYSMEM, SYSMEM_READ, SYSMEM_WRITE and PING are refused with an error reply, never
    forwarded (their payload, if any, is read first, so the stream stays in step); anything else stops the proxy;
  - each header and its payload go upstream in one send (server.c reads a header with one plain recv, :201);
  - an MMIO_WRITE over 64 MB is never forwarded (server.c receives it into a 64 MB buffer before validating, :243-246);
  - markers (a CFG_READ with dev_id 0x42454147, 'BEAG') are recorded and answered here, never forwarded;
  - fail-stop: on anything unexpected (a frame cut by the client, an unknown command, an over-long write, a reply
    without its fd, a guard refusal, an error in this proxy) it stops forwarding but keeps both connections open, and
    says to unplug the eGPU before killing it. Closing the upstream connection makes TinyGPU.app unwire every sysmem
    buffer (server.c:171-183), which faults the Mac's IOMMU if the GSP is still using them;
  - a client that closes at a request boundary ends its session: the proxy closes upstream too, as TinyGPU.app would
    have seen the client close, and serves the next client, one at a time as TinyGPU.app does (the plugin's probe at
    load is a session of its own). With --guard it holds instead unless the guard saw the GPU torn down (a keeper for
    the first C++ hardware runs);
  - SIGINT and SIGHUP are ignored (a terminal's Ctrl-C must not close a live GPU's connection); SIGTERM ends the
    recording between sessions, or after the current one; in a fail-stop only SIGKILL ends it.

    python tgproxy.py --listen <socket> --upstream <TinyGPU.app socket> --out <new recording dir> [--guard]
                      [--start-app] [--label L]
--start-app starts TinyGPU.app on the upstream socket when nothing listens there, as tinygrad does (system.py:430-436);
only hardware recordings pass it. It prints "tgproxy listening" once a client may connect, and each session's end."""
import os, sys, time, json, mmap, socket, signal, struct, argparse, subprocess, pathlib, platform, traceback, contextlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import tgwire as w

APP_PATH = "/Applications/TinyGPU.app/Contents/MacOS/TinyGPU"   # APLRemotePCIDevice.APP_PATH (system.py:417)
STATUS_REGION, QUEUES_SIZE = w.STATUS_REGION, w.QUEUES_SIZE
WATCH_MAX = 64 << 10                 # mappings up to this size are diffed whole at every request
DIFF_BLOCK = 256

AMD_VENDOR = 0x1002

class FailStop(Exception): pass
class NoUpstream(Exception): pass   # nothing listens upstream: the session reached no GPU, so it is closed, not held
class _Exit(Exception): pass

class Mapping:
    """One MAP_SYSMEM_FD allocation, mapped read-only here: what the client and the GPU write into it."""
    def __init__(self, alloc, fd, size, contiguous, mapped):
        self.alloc, self.size, self.contiguous, self.mapped, self.segs = alloc, size, contiguous, mapped, []
        self.mm = mmap.mmap(fd, mapped, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ)
        self.last = self.mm[:]   # at the last page snapshot (now: the segment list and zeros)
        region = (0, mapped) if mapped <= WATCH_MAX else STATUS_REGION if size == QUEUES_SIZE and mapped >= STATUS_REGION[1] else None
        self.watch = region and (region[0], region[1], self.mm[region[0]:region[1]])

    def diffs(self):
        """Changed runs of the watched region since the last call: [(offset, before, after)]."""
        if not self.watch: return []
        a, b, prev = self.watch
        cur = self.mm[a:b]
        if cur == prev: return []
        runs, start = [], None
        for o in range(0, len(cur), DIFF_BLOCK):
            changed = cur[o:o + DIFF_BLOCK] != prev[o:o + DIFF_BLOCK]
            if changed and start is None: start = o
            if not changed and start is not None: runs.append((start, o)); start = None
        if start is not None: runs.append((start, len(cur)))
        self.watch = (a, b, cur)
        return [(a + s, prev[s:e], cur[s:e]) for s, e in runs]

    def changed_pages(self):
        cur = self.mm[:]
        if cur == self.last: return []
        out = [(p // w.PAGE, cur[p:p + w.PAGE]) for p in range(0, len(cur), w.PAGE) if cur[p:p + w.PAGE] != self.last[p:p + w.PAGE]]
        self.last = cur
        return out

def set_bufs(sock):
    for opt in (socket.SO_SNDBUF, socket.SO_RCVBUF):   # as RemotePCIDevice.__init__ (system.py:390); macOS caps it, once
        try: sock.setsockopt(socket.SOL_SOCKET, opt, 64 << 20)
        except OSError: pass

def git_rev(path):
    try: return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception: return None

class Proxy:
    def __init__(self, a):
        self.a, self.t0 = a, time.monotonic_ns()
        self.triggers = w.trigger_addrs()
        self.vendor, self.amd_triggers = None, None   # per session: from its first config read (PCI vendor 0x1002: the AMD card)
        self.rec = w.Writer(a.out, self.t0)
        self.seq, self.maps, self.client, self.up, self.guard = 0, [], None, None, None
        self.pages_due = False   # a trigger since the last page snapshot
        self.in_session = self.exit_after = False
        self.n_allocs = 0
        self.stats = dict(requests=0, forwarded=0, markers=0, refused=0, bytes_up=0, bytes_down=0, triggers=0, diffs=0, pages=0)
        self.meta = dict(tool="tgproxy", label=a.label, argv=sys.argv, listen=a.listen, upstream=a.upstream, guard=bool(a.guard),
                         host=platform.node(), pid=os.getpid(), started=time.strftime("%Y-%m-%d %H:%M:%S"),
                         beagle_rev=git_rev(pathlib.Path(__file__).resolve().parents[4]),
                         tinygrad_rev=git_rev(os.environ.get("TINYGRAD_PATH", pathlib.Path.home() / "Dropbox/Projects/tinygrad-hcq1")))

    def now(self): return time.monotonic_ns()
    def log(self, msg): print(f"tgproxy: {msg}", flush=True)

    # ── connections ──────────────────────────────────────────────────────────────────────────────────────────────────
    def connect_upstream(self):
        for i in range(100):
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)   # a fresh socket per attempt
            try:
                s.connect(self.a.upstream)
                set_bufs(s)
                return s
            except (FileNotFoundError, ConnectionRefusedError):
                s.close()
                if not self.a.start_app: raise NoUpstream(f"nothing listens at {self.a.upstream} (and --start-app was not given)")
                if i == 0:
                    self.log(f"starting TinyGPU.app on {self.a.upstream}")
                    subprocess.Popen([APP_PATH, "server", self.a.upstream], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL, start_new_session=True)
                time.sleep(0.05)
        raise NoUpstream(f"could not connect to {self.a.upstream}")

    def listen(self):
        p = pathlib.Path(self.a.listen)
        if p.exists() or p.is_symlink():
            if not p.is_socket(): sys.exit(f"tgproxy: {p} exists and is not a socket")
            p.unlink()
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(str(p)); srv.listen(1)
        return srv

    # ── the session ──────────────────────────────────────────────────────────────────────────────────────────────────
    def run(self):
        """Serve clients one after another, as TinyGPU.app does (server.c:271-279): the plugin's probe at load, then its
        instance's session. SIGTERM between sessions ends the recording; during one, after it."""
        srv = self.listen()
        self.log(f"listening on {self.a.listen}, upstream {self.a.upstream}, recording to {self.a.out}" + (" (guard mode)" if self.a.guard else ""))
        signal.signal(signal.SIGTERM, self.on_term)
        print("tgproxy listening", flush=True)
        sessions = []
        try:
            while not self.exit_after:
                self.client, _ = srv.accept()
                self.in_session = True
                sessions.append(self.session(len(sessions) + 1))
                self.in_session = False
        except _Exit: pass
        srv.close()
        with contextlib.suppress(FileNotFoundError): os.unlink(self.a.listen)
        info = dict(how="stopped", sessions=sessions, stats=self.stats)
        self.rec.end(self.now(), info)
        self.rec.close(dict(self.meta, ended=time.strftime("%Y-%m-%d %H:%M:%S"), end=info))
        self.log(f"recording ended after {len(sessions)} session(s): {json.dumps(self.stats)}")
        return 0

    def on_term(self, signum, frame):
        self.exit_after = True
        if not self.in_session: raise _Exit()   # breaks the accept() between sessions
        self.log("SIGTERM: the recording ends when this session does")

    def session(self, n):
        set_bufs(self.client)
        self.maps, self.up, self.pages_due = [], None, False
        self.guard, self.vendor = None, None
        if self.a.guard:
            import tgguard
            self.guard = tgguard.Guard(log=self.log)
        self.rec.note(self.now(), dict(event="session", n=n, first_seq=self.seq + 1))
        how, why = "eof", ""
        try:
            try: self.up = self.connect_upstream()
            except NoUpstream as e:   # the client gets EOF: nothing was forwarded
                self.client.close()
                self.rec.note(self.now(), dict(event="session end", n=n, how="no-upstream", why=str(e)))
                self.log(f"session {n} ended: no-upstream: {e}")
                return dict(n=n, how="no-upstream", why=str(e))
            while self.one_request(): pass
            if self.guard and not self.guard.clean_exit():
                raise FailStop(f"the client closed, but the guard did not see the GPU torn down ({self.guard.state()})")
        except FailStop as e: how, why = "failstop", str(e)
        except ConnectionError as e: how, why = "upstream-lost", f"{type(e).__name__}: {e}"
        except Exception as e: how, why = "failstop", f"proxy error: {type(e).__name__}: {e}\n{traceback.format_exc()}"
        self.seq += 1   # the session's final state, at a seq no request has
        self.snapshot_pages(self.seq)
        info = dict(event="session end", n=n, how=how, why=why, final_pages_seq=self.seq)
        self.rec.note(self.now(), info)
        self.rec.flush()
        self.log(f"session {n} ended: {how}{': ' + why if why else ''}; {json.dumps(self.stats)}")
        if how == "failstop":
            self.rec.end(self.now(), dict(how="failstop", why=why, stats=self.stats))
            self.rec.close(dict(self.meta, ended=time.strftime("%Y-%m-%d %H:%M:%S"), end=dict(how="failstop", why=why, session=n)))
            self.log(f"FAIL-STOP: nothing more is forwarded; both connections stay open. Unplug the eGPU first, then kill -9 {os.getpid()}.")
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            while True: time.sleep(3600)   # the sockets and mappings stay referenced
        if how == "upstream-lost": self.log("the TinyGPU.app side closed; closing the client's connection too")
        for sock in (self.up, self.client):
            if sock: sock.close()
        for m in self.maps: m.mm.close()   # TinyGPU.app unwires a session's sysmem at its end (server.c:171-183)
        self.maps = []
        return dict(n=n, how=how, why=why)

    def one_request(self):
        """Forward one client request and its reply. False when the client has closed at a request boundary."""
        hdr = w.recv_exact(self.client, 33)
        if not hdr: return False
        if len(hdr) < 33: raise FailStop(f"the client closed inside a request header ({len(hdr)} of 33 bytes)")
        t = self.now()
        self.seq += 1
        seq = self.seq
        self.stats["requests"] += 1
        self.watch_diffs(seq, t)
        cmd, dev, bar, a0, a1, a2 = w.REQ.unpack(hdr)
        if cmd == w.CFG_READ and dev == w.MARKER_DEV:
            self.rec.marker(t, seq, bar, a2)
            self.to_client(w.RESP.pack(0, 0, 0))
            self.stats["markers"] += 1
            self.rec.flush()
            return True
        if cmd not in w.ALLOWED:
            if cmd > w.PING: raise FailStop(f"unknown command {cmd} (its framing is unknown): {hdr.hex()}")
            if cmd in w.PAYLOAD_CMDS:
                if a1 > w.MAX_MESSAGE: raise FailStop(f"{w.cmd_name(cmd)} with a {a1}-byte payload")
                if len(w.recv_exact(self.client, a1)) < a1: raise FailStop(f"the client closed inside a {w.cmd_name(cmd)} payload")
            why = f"tgproxy: {w.cmd_name(cmd)} refused (plan V1 allow-list); not forwarded"
            self.rec.refused(t, seq, hdr, why)
            self.to_client(w.RESP.pack(1, len(why), 0) + why.encode())
            self.stats["refused"] += 1
            self.log(f"seq {seq}: {why}")
            self.rec.flush()
            return True
        if cmd == w.MMIO_WRITE:
            if a1 > w.MAX_MESSAGE: raise FailStop(f"an MMIO_WRITE of {a1} bytes (over 64 MB) at bar {bar} offset {a0:#x}")
            payload = w.recv_exact(self.client, a1)
            if len(payload) < a1: raise FailStop(f"the client closed inside an MMIO_WRITE payload ({len(payload)} of {a1} bytes)")
            self.rec.req(t, seq, hdr, payload)
            if self.is_trigger(bar, a0):
                self.stats["triggers"] += 1
                self.pages_due = True
                why = self.guard and (self.guard.check_trigger(bar, a0, payload) if self.vendor == AMD_VENDOR else self.guard.check_trigger(a0, payload))
                if why:
                    self.rec.refused(self.now(), seq, hdr, f"guard: {why}")
                    raise FailStop(f"guard refused {self.trigger_name(bar, a0)} (seq {seq}): {why}")
            self.up.sendall(hdr + payload)
            self.stats["forwarded"] += 1; self.stats["bytes_up"] += 33 + a1
            if self.guard: self.guard.on_write(bar, a0, payload)
            self.rec.flush()
            return True
        self.rec.req(t, seq, hdr)
        if self.pages_due: self.snapshot_pages(seq)   # the client waits for this reply: its writes so far are all it wrote
        self.up.sendall(hdr)
        self.stats["forwarded"] += 1; self.stats["bytes_up"] += 33
        fd = None
        if cmd == w.MAP_SYSMEM_FD:
            resp, anc, flags, _ = self.up.recvmsg(17, socket.CMSG_SPACE(4))
            fds = [fd for level, typ, data in anc if level == socket.SOL_SOCKET and typ == socket.SCM_RIGHTS
                   for fd in struct.unpack(f"{len(data) // 4}i", data[:len(data) // 4 * 4])]
            fd = fds[0] if fds else None
            for extra in fds[1:]: os.close(extra)
            if flags & socket.MSG_CTRUNC: raise FailStop("MAP_SYSMEM_FD: the reply's fd was cut (MSG_CTRUNC)")
            if 0 < len(resp) < 17: resp += w.recv_exact(self.up, 17 - len(resp))
        else: resp = w.recv_exact(self.up, 17)
        if len(resp) < 17: raise ConnectionResetError(f"TinyGPU.app closed before replying to {w.cmd_name(cmd)} (seq {seq})")
        status, r0, r1 = w.RESP.unpack(resp)
        need = r0 if status != 0 or cmd == w.MMIO_READ else 0   # MMIO_READ's data, or a failure's message (server.c:84-88)
        data = w.recv_exact(self.up, need) if need else b""
        if len(data) < need: raise ConnectionResetError(f"TinyGPU.app closed inside a reply (seq {seq})")
        if cmd == w.MAP_SYSMEM_FD and status == 0 and fd is None: raise FailStop(f"a successful MAP_SYSMEM_FD reply without an fd (seq {seq})")
        if cmd == w.MAP_SYSMEM_FD and status != 0 and fd is not None: os.close(fd); fd = None
        tr = self.now()
        self.rec.reply(tr, seq, resp, fd is not None, data)
        if fd is not None:
            try:
                m = self.add_mapping(tr, seq, fd, a0, a1, r0, r1)
                self.to_client(resp, fd)
            finally: os.close(fd)
            if self.guard: self.guard.on_sysmem(m.alloc, m.segs, m.size, m.mm)
        else: self.to_client(resp + data)
        self.stats["bytes_down"] += 17 + len(data)
        if cmd == w.CFG_READ and a0 == 0 and status == 0 and self.vendor is None: self.set_vendor(r0 & 0xffff)
        if self.guard and cmd == w.MMIO_READ and status == 0: self.guard.on_read(bar, a0, data)
        self.rec.flush()
        return True

    def set_vendor(self, vendor):
        """The session's card, from its first config read: AMD's triggers and guard (tgguard_amd.py) for vendor 0x1002."""
        self.vendor = vendor
        if vendor != AMD_VENDOR: return
        import tgguard_amd
        if self.amd_triggers is None: self.amd_triggers = tgguard_amd.trigger_addrs()
        if self.guard: self.guard = tgguard_amd.AMDGuard(log=self.log)
        self.log(f"session: the AMD card (vendor {vendor:#06x}): AMD triggers{' and guard' if self.guard else ''}")

    def is_trigger(self, bar, off):
        if self.vendor == AMD_VENDOR: return bar == 2 or (bar, off) in self.amd_triggers
        return bar == 0 and off in self.triggers

    def trigger_name(self, bar, off):
        if self.vendor == AMD_VENDOR: return "a doorbell" if bar == 2 else self.amd_triggers[(bar, off)]
        return self.triggers[off]

    def to_client(self, data, fd=None):
        """A client that goes away while a reply is due left mid-exchange: that is a fail-stop, not an upstream loss."""
        try:
            if fd is None: self.client.sendall(data)
            else: self.client.sendmsg([data], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, struct.pack("i", fd))])
        except OSError as e: raise FailStop(f"the client went away before its reply: {type(e).__name__}: {e}") from e

    # ── snapshots ────────────────────────────────────────────────────────────────────────────────────────────────────
    def add_mapping(self, t, seq, fd, size, contiguous, mapped, idx):
        m = Mapping(self.n_allocs, fd, size, contiguous, mapped)   # numbered across sessions
        self.n_allocs += 1
        m.segs, n = w.segments(m.mm[:min(mapped, 8192)])
        self.maps.append(m)
        self.rec.sysmem(t, seq, m.alloc, size, contiguous, mapped, idx, m.mm[:n])
        return m

    def watch_diffs(self, seq, t):
        for m in self.maps:
            for off, before, after in m.diffs():
                self.rec.diff(t, seq, m.alloc, off, before, after)
                self.stats["diffs"] += 1

    def snapshot_pages(self, seq):
        self.pages_due = False
        t = self.now()
        for m in self.maps:
            changed = m.changed_pages()
            if changed:
                self.rec.pages(t, seq, m.alloc, changed)
                self.stats["pages"] += len(changed)

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--listen", required=True)
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--guard", action="store_true")
    ap.add_argument("--start-app", action="store_true")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    sys.exit(Proxy(a).run())

if __name__ == "__main__":
    main()
