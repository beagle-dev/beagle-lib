"""Replay server: a fake TinyGPU.app driven by a recording (TODO.md plan step V1, inv:verification#5).

The client (the unmodified plugin and daemon, as recorded) connects through APL_REMOTE_SOCK; each of its sessions is served
from the recording's session of the same number:
  - every request must equal the recorded one, byte for byte (header and payload); the first that differs ends the session
    with a report of it and the 20 recorded events before it, and the client gets EOF;
  - replies (CFG_READ, MAP_BAR, RESIZE_BAR, CFG_WRITE, MMIO_READ, failures) are the recorded ones; MAP_SYSMEM_FD hands out a new
    file of the recorded size with the recorded segment list at its start, so the client's device addresses are the recorded;
  - the GSP: what it wrote into its status queue (the only memory only the GSP writes) is applied causally: what the proxy saw
    when request r arrived, once the last request before r that is not a marker is served (tinygrad reads its replies there
    without any request in between; the GSP acts only on real requests, and on hardware the first request after its write
    is often a marker, so a client that sends fewer markers, or none, as a port does, is served the same);
  - the GPU is not replayed but run: on a doorbell the channel's GPFIFO runs through the client's own page tables (tggpu:
    semaphores, QMD launches, copies), with the channels and the page-directory root taken from the client's RPCs and the
    work-submit tokens from the replayed GSP replies. Kernels are not run, so results differ from a real GPU's;
  - checks, besides the requests: wherever the recording has a page snapshot (the first request after a trigger that the
    client waits on, and the session's end) the pages that changed since the previous one against the recording's (changed
    in both, equal: a match; changed here only, or unequal: a mismatch unless the GPU here wrote the page, then listed;
    changed in the recording only: written by the GPU there, listed); at the end every page's contents; each recorded diff
    outside the status queue against this client's bytes when its request arrives (listed); minimum sleeps: at least 20 s
    from a SEC2 start to the next NV_PGC6_BSI_SECURE_SCRATCH_14 read (nv_init_helper's sleep, ip.py:651-658) and at least
    0.1 s between an engine reset's two writes (NV_FLCN.reset, ip.py:273-275); markers, as an ordered sequence, listed.
A session passes if every request matched, the client closed after the last one, and no client-written page differs.
With --mutate the recording is changed before it is served (plan V1's differential replay; see MUTATIONS). With --guard the
guard (tgguard.py) audits every trigger as the proxy's guard mode would, and a refusal fails the session there.
    python tgreplay.py --listen <socket> --rec <recording dir> --mem <dir> [--out <replay recording>] [--mutate NAME] [--guard]
It prints "tgreplay listening", a line per session ("replay session N: PASS" or FAIL with the reason), and exits after
the recording's last session, 0 only if every session passed."""
import os, sys, time, json, mmap, zlib, socket, select, signal, struct, argparse, pathlib, hashlib, collections
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import tgwire as w
import tggpu

def describe(req, payload=b""):
    cmd, dev, bar, a0, a1, a2 = req
    if cmd == w.MMIO_WRITE and bar == 0 and len(payload) == 4:
        return f"MMIO_WRITE BAR0 {w.reg_name(a0)} = {struct.unpack('<I', payload)[0]:#x}"
    if cmd in (w.MMIO_WRITE, w.MMIO_READ): return f"{w.cmd_name(cmd)} BAR{bar} {w.reg_name(a0) if bar == 0 else hex(a0)} +{a1:#x}"
    if cmd == w.CFG_READ and dev == w.MARKER_DEV: return f"MARKER id={bar} arg={a2:#x}"
    return f"{w.cmd_name(cmd)} bar={bar} a0={a0:#x} a1={a1:#x} a2={a2:#x}"

class RecSession:
    """One recorded session: the client's requests in order, and what goes with each seq."""
    def __init__(self, n, events):
        self.n, self.events = n, events
        self.reqs = [e for e in events if e.kind in (w.K_REQ, w.K_MARKER, w.K_REFUSED)]
        self.reply = {e.seq: e for e in events if e.kind == w.K_REPLY}
        self.sysmem = {e.seq: e for e in events if e.kind == w.K_SYSMEM}
        self.diffs, self.pages = collections.defaultdict(list), collections.defaultdict(list)
        for e in events:
            if e.kind == w.K_DIFF: self.diffs[e.seq].append(e)
            elif e.kind == w.K_PAGES: self.pages[e.seq].append(e)
        end = next((e.f for e in events if e.kind == w.K_NOTE and e.f.get("event") == "session end"), {})
        self.final_seq = end.get("final_pages_seq")
        self.final = {}   # each page's contents at the proxy's last snapshot of it
        for e in events:
            if e.kind == w.K_PAGES:
                for pg, sha in e.f["pages"]: self.final[(e.f["alloc"], pg)] = sha

def split_sessions(events):
    out, cur = [], None
    for e in events:
        if e.kind == w.K_NOTE and e.f.get("event") == "session":
            cur = []; out.append(cur)
        if cur is not None: cur.append(e)
    return [RecSession(n + 1, evs) for n, evs in enumerate(out)]

class Alloc:
    """A replayed MAP_SYSMEM_FD allocation: a file, its recorded segment list at the start, its device addresses."""
    def __init__(self, path, sm):
        self.size, self.recorded_size = sm.f["mapped"], sm.f["size"]
        self.fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
        os.ftruncate(self.fd, self.size)
        self.mm = mmap.mmap(self.fd, self.size)
        self.mm[:len(sm.f["seglist"])] = sm.f["seglist"]
        self.segs, self.last = sm.f["segs"], self.mm[:]
        self.initial, self.crc = self.last, zlib.crc32(self.last)
    def offset(self, iova):
        off = 0
        for p, s in self.segs:
            if p <= iova < p + s: return off + iova - p
            off += s
        return None
    def close(self): self.mm.close(); os.close(self.fd)

class Replay:
    def __init__(self, a):
        self.a = a
        self.events, self.blobs = w.read(a.rec)
        if a.mutate: mutate(self.events, a.mutate)
        self.sessions = split_sessions(self.events)
        names = w.reg_names()
        self.triggers = w.trigger_addrs()
        self.queue_head = next(x for x, n in names.items() if n == "NV_PGSP_QUEUE_HEAD[0]")
        self.bsi14 = next(x for x, n in names.items() if n == "NV_PGC6_BSI_SECURE_SCRATCH_14")
        self.boot42 = next(x for x, n in names.items() if n == "NV_PMC_BOOT_42")
        self.engines = {x for x, n in names.items() if n in ("NV_PGSP_FALCON_ENGINE", "NV_PSEC_FALCON_ENGINE")}
        self.sec2_start = {x for x, n in self.triggers.items() if n in ("SEC2.NV_PFALCON_FALCON_CPUCTL", "SEC2.NV_PFALCON_FALCON_CPUCTL_ALIAS")}
        self.R = tggpu.regs("ada")
        self.out = w.Writer(a.out, time.monotonic_ns()) if a.out else None
        self.results = []

    def log(self, msg): print(f"tgreplay: {msg}", flush=True)

    def run(self):
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        if os.path.exists(self.a.listen): os.unlink(self.a.listen)
        srv.bind(self.a.listen); srv.listen(1)
        self.log(f"{len(self.sessions)} recorded session(s) from {self.a.rec}" + (f", mutation {self.a.mutate}" if self.a.mutate else ""))
        print("tgreplay listening", flush=True)
        for sess in self.sessions:
            conn, _ = srv.accept()
            r = Session(self, sess).serve(conn)
            self.results.append(r)
            print(f"replay session {sess.n}: {'PASS' if r['ok'] else 'FAIL'}{'' if r['ok'] else ': ' + r['why']} " + json.dumps(r["stats"]), flush=True)
            if r.get("report"): print(r["report"], flush=True)
        srv.close(); os.unlink(self.a.listen)
        ok = all(r["ok"] for r in self.results)
        if self.out: self.out.close(dict(tool="tgreplay", rec=str(self.a.rec), mutate=self.a.mutate,
                                         results=[{k: v for k, v in r.items() if k != "report"} for r in self.results]))
        print(f"tgreplay: {'PASS' if ok else 'FAIL'}: {sum(r['ok'] for r in self.results)} of {len(self.results)} session(s) replayed exactly", flush=True)
        return 0 if ok else 1

class Session:
    def __init__(self, rp, sess):
        self.rp, self.sess = rp, sess
        self.st, self.info = collections.Counter(), collections.Counter()
        self.allocs = {}                        # recording allocation number -> Alloc (numbered across sessions, as the proxy does)
        self.queues = None                      # (allocation number, command-queue reader, status-queue reader)
        self.vram = tggpu.Vram()
        self.memory = tggpu.Memory(self.vram, self.sys_rw, rp.R)
        self.channels = tggpu.Channels()
        self.gpu_errors = []
        self.frontend = tggpu.Frontend(self.memory, self.channels, self.info, self.gpu_errors.append)   # Ada's until set_chip
        self.gpu_pages, self.gpu_pages_all = set(), set()   # (allocation, page) the GPU wrote here: since the last snapshot, ever
        self.page_bad, self.diff_bad = [], []
        self.why = self.report = ""
        self.guard = None
        if rp.a.guard:
            import tgguard
            self.guard = tgguard.Guard(log=rp.log)

    def set_chip(self, boot42):   # the boot's NV_PMC_BOOT_42 read: the chip's MMU and QMD versions (tggpu.chip)
        name, mmu_ver, compute = tggpu.chip(boot42)
        self.memory = tggpu.Memory(self.vram, self.sys_rw, tggpu.regs(name), mmu_ver)
        self.frontend = tggpu.Frontend(self.memory, self.channels, self.info, self.gpu_errors.append, compute)

    # sysmem by device address, for the GPU
    def sys_rw(self, iova, n, data=None):
        for num, al in self.allocs.items():
            off = al.offset(iova)
            if off is not None and off + n <= al.size:
                if data is None: return bytes(al.mm[off:off + n])
                al.mm[off:off + n] = data
                for pg in range(off // w.PAGE, (off + n - 1) // w.PAGE + 1): self.gpu_pages.add((num, pg)); self.gpu_pages_all.add((num, pg))
                return
        raise RuntimeError(f"GPU access to {iova:#x} (+{n}): not a device address of any replayed allocation")

    def is_status(self, alloc, off, n):
        return self.queues is not None and alloc == self.queues[0] and w.STATUS_REGION[0] <= off and off + n <= w.STATUS_REGION[1]

    def apply_gsp(self, seq):
        """The GSP's status-queue writes the proxy saw when request seq arrived; then the replies they complete. From the
        highest offset down: the GSP writes a message, then publishes it by moving the write pointer in the queue's header,
        at the region's start, and the client may be reading the queue meanwhile (tinygrad's wait_resp spins on it)."""
        for d in sorted(self.sess.diffs.get(seq, []), key=lambda d: -d.f["off"]):
            if not self.is_status(d.f["alloc"], d.f["off"], len(d.f["after"])): continue
            al = self.allocs[d.f["alloc"]]
            off, before, after = d.f["off"], d.f["before"], d.f["after"]
            if al.mm[off:off + len(after)] != before and al.mm[off:off + len(after)] != after: self.st["status bytes not as recorded before"] += 1
            al.mm[off:off + len(after)] = after
            self.st["status diffs applied"] += 1
        if self.queues is not None:
            for fn, payload, elem, ok in self.queues[2].new(): self.channels.observe_reply(fn, payload)

    def check_diffs(self, seq):
        """Outside the status queue a recorded diff is compared, not applied: the client and the GPU here write those bytes."""
        for d in self.sess.diffs.get(seq, []):
            if self.is_status(d.f["alloc"], d.f["off"], len(d.f["after"])): continue
            al = self.allocs.get(d.f["alloc"])
            cur = al.mm[d.f["off"]:d.f["off"] + len(d.f["after"])] if al else None
            if cur == d.f["after"]: self.info["diffs matched"] += 1
            else:
                self.info["diffs unlike here"] += 1
                if len(self.diff_bad) < 5: self.diff_bad.append((seq, d.f["alloc"], hex(d.f["off"]), len(d.f["after"])))

    def compare_pages(self, seq):
        """At a recorded page snapshot (the client is waiting on this request, here as there): the pages that changed since
        the previous one, here and there, with their contents."""
        recorded = {p.f["alloc"]: dict(p.f["pages"]) for p in self.sess.pages.get(seq, [])}
        for num, al in self.allocs.items():
            rec = recorded.get(num, {})
            crc = zlib.crc32(memoryview(al.mm))   # a quick skip of unchanged allocations
            mine = {}
            if crc != al.crc:
                cur, al.crc = al.mm[:], crc
                mine = {p // w.PAGE: hashlib.sha256(cur[p:p + w.PAGE]).digest() for p in range(0, len(cur), w.PAGE) if cur[p:p + w.PAGE] != al.last[p:p + w.PAGE]}
                al.last = cur
            for pg in mine.keys() | rec.keys():
                if self.is_status(num, pg * w.PAGE, w.PAGE): continue   # the GSP's (asynchronous) status queue: its diffs are applied, not compared
                gpu = (num, pg) in self.gpu_pages
                if pg in mine and pg in rec and mine[pg] == rec[pg]: self.st["pages matched"] += 1
                elif gpu: self.info["GPU-written pages unlike the recording"] += 1
                elif pg in mine and pg in rec: self.st["pages mismatched"] += 1; self.page_bad.append((seq, num, pg, "contents differ"))
                elif pg in mine: self.st["pages changed here only"] += 1; self.page_bad.append((seq, num, pg, "changed here, not in the recording"))
                else: self.info["pages written by the GPU in the recording only"] += 1
        self.gpu_pages.clear()

    def compare_final(self):
        """At the session's end every page must hold what the recording's last snapshot of it held (or, never snapshotted
        there, what it held when allocated), unless the GPU here wrote it (listed: results, which it does not compute) or
        nothing here wrote it (listed: the GPU or the GSP wrote it there, e.g. the GSP's log buffer)."""
        for num, al in self.allocs.items():
            cur, first = al.mm[:], al.initial
            for p in range(0, len(cur), w.PAGE):
                key = (num, p // w.PAGE)
                if self.is_status(num, p, w.PAGE): continue
                want = self.sess.final.get(key)
                ok = hashlib.sha256(cur[p:p + w.PAGE]).digest() == want if want is not None else cur[p:p + w.PAGE] == first[p:p + w.PAGE]
                if ok: self.st["final pages matched"] += 1
                elif key in self.gpu_pages_all: self.info["GPU-written final pages unlike the recording"] += 1
                elif cur[p:p + w.PAGE] == first[p:p + w.PAGE]: self.info["final pages written in the recording only (its GPU or GSP)"] += 1
                else: self.st["final pages unlike the recording"] += 1; self.page_bad.append(("end", num, p // w.PAGE, "final contents differ"))

    def diverge(self, exp_seq, msg):
        self.why = msg
        ctx = [e for e in self.sess.events if e.kind in (w.K_REQ, w.K_REPLY, w.K_MARKER) and e.seq is not None and exp_seq - 20 <= e.seq < exp_seq]
        lines = [f"  first divergence at recorded seq {exp_seq}: {msg}", "  the 20 recorded events before it:"]
        for e in ctx:
            if e.kind == w.K_REQ: lines.append(f"    seq {e.seq}: {describe(e.f['req'], e.f['payload'])}")
            elif e.kind == w.K_REPLY:
                lines.append(f"    seq {e.seq}:   reply status {e.f['reply'][0]} resp0 {e.f['reply'][1]:#x}" + (f" data {e.f['data'][:8].hex()}" if e.f["data"] else ""))
            else: lines.append(f"    seq {e.seq}: MARKER id={e.f['id']} arg={e.f['arg']:#x}")
        self.report = "\n".join(lines)

    def serve(self, conn):
        rp, sess, st = self.rp, self.sess, self.st
        i, markers_got = 0, []
        markers_rec = [(e.f["id"], e.f["arg"]) for e in sess.reqs if e.kind == w.K_MARKER]
        sec2_start, reset_on = None, {}
        if rp.out: rp.out.note(time.monotonic_ns(), dict(event="session", n=sess.n))
        applied = -1   # the GSP's writes are applied through this recorded request
        def apply_gsp_through(k):   # from request k up to the next one that is not a marker
            nonlocal applied
            while k < len(sess.reqs):
                if k > applied: self.apply_gsp(sess.reqs[k].seq); applied = k
                if sess.reqs[k].kind != w.K_MARKER: break
                k += 1
        def advance():   # the next recorded request, with the GSP's writes the proxy saw up to the next real one
            nonlocal i
            i += 1
            apply_gsp_through(i)
        apply_gsp_through(0)
        try:
            while True:
                waited = not select.select([conn], [], [], 0)[0]   # nothing queued: this request's arrival is its send time
                hdr = w.recv_exact(conn, 33)
                if not hdr: break
                req = w.REQ.unpack(hdr)
                cmd, dev, bar, a0, a1, a2 = req
                t = time.monotonic()
                payload = w.recv_exact(conn, a1) if cmd == w.MMIO_WRITE or (cmd in w.PAYLOAD_CMDS and a1 <= w.MAX_MESSAGE) else b""
                st["requests"] += 1
                if rp.out: rp.out.req(time.monotonic_ns(), st["requests"], hdr, payload)
                if cmd == w.CFG_READ and dev == w.MARKER_DEV:   # answered here; compared as a sequence at the end
                    markers_got.append((bar, a2))
                    conn.sendall(w.RESP.pack(0, 0, 0))
                    if i < len(sess.reqs) and sess.reqs[i].kind == w.K_MARKER: advance()
                    continue
                while i < len(sess.reqs) and sess.reqs[i].kind == w.K_MARKER: advance()   # recorded markers this client does not send
                if i >= len(sess.reqs): self.diverge(sess.reqs[-1].seq + 1 if sess.reqs else 0, f"the client sent {describe(req, payload)} after the recorded session ended"); break
                e = sess.reqs[i]
                if e.f["hdr"] != hdr or (e.kind == w.K_REQ and e.f["payload"] != payload):
                    self.diverge(e.seq, f"expected {describe(e.f['req'], e.f.get('payload', b''))}, got {describe(req, payload)}"
                                 + (" (the payload differs)" if e.f["hdr"] == hdr else ""))
                    break
                self.check_diffs(e.seq)
                if e.seq in self.sess.pages: self.compare_pages(e.seq)   # a recorded snapshot: the client waits on this request
                if e.kind == w.K_REFUSED:
                    msg = e.f["why"].encode()
                    conn.sendall(w.RESP.pack(1, len(msg), 0) + msg)
                elif cmd == w.MMIO_WRITE:
                    if self.guard and bar == 0 and a0 in rp.triggers and (gwhy := self.guard.check_trigger(a0, payload)):
                        self.diverge(e.seq, f"the guard refused {rp.triggers[a0]}: {gwhy}"); break
                    if self.guard: self.guard.on_write(bar, a0, payload)
                    if bar == 1: self.vram.write(a0, payload)
                    elif bar == 0 and len(payload) == 4:
                        v = struct.unpack("<I", payload)[0]
                        if a0 in rp.sec2_start: sec2_start = (t, waited)
                        if a0 in rp.engines:
                            if v & 1: reset_on[a0] = (t, waited)
                            elif a0 in reset_on:
                                t0, waited0 = reset_on.pop(a0)
                                dt = t - t0
                                if not (waited and waited0): st["engine resets not timed (the replay lagged)"] += 1
                                else:
                                    st["engine resets"] += 1
                                    if dt < 0.1: self.diverge(e.seq, f"an engine reset held {dt * 1000:.0f} ms, less than tinygrad's 0.1 s"); break
                        if a0 in rp.triggers:   # the GPU or the GSP acts
                            st["triggers"] += 1
                            if a0 == rp.queue_head and self.queues is not None:
                                for fn, msg, elem, ok in self.queues[1].new():
                                    if not ok: self.gpu_errors.append(f"RPC {fn:#x}: bad checksum")
                                    self.channels.observe_cmd(fn, msg, self.memory)
                            elif a0 == w.DOORBELL:
                                try: self.frontend.doorbell(v)
                                except RuntimeError as ex: self.gpu_errors.append(str(ex))
                else:
                    rep = sess.reply.get(e.seq)
                    if rep is None: self.diverge(e.seq, f"no recorded reply for {describe(req)}"); break
                    if cmd == w.MMIO_READ and bar == 0 and a0 == rp.bsi14 and sec2_start is not None:
                        dt, waited = t - sec2_start[0], waited and sec2_start[1]
                        if waited and dt < 20.0: self.diverge(e.seq, f"NV_PGC6_BSI_SECURE_SCRATCH_14 read {dt:.1f} s after SEC2 started, less than BEAGLE's 20 s"); break
                        st["SEC2 sleeps timed" if waited else "SEC2 sleeps not timed (the replay lagged)"] += 1
                        sec2_start = None
                    if cmd == w.MMIO_READ and bar == 1 and rep.f["reply"][0] == 0 and self.vram.read(a0, a1) != rep.f["data"]:
                        self.info["BAR1 reads unlike the VRAM here"] += 1   # the GPU writes VRAM too (USERD, semaphores, copies)
                    if self.guard and cmd == w.MMIO_READ and rep.f["reply"][0] == 0: self.guard.on_read(bar, a0, rep.f["data"])
                    if cmd == w.MMIO_READ and bar == 0 and a0 == rp.boot42 and rep.f["reply"][0] == 0 and self.memory.root is None:
                        self.set_chip(struct.unpack_from("<I", rep.f["data"])[0])
                    if cmd == w.MAP_SYSMEM_FD and rep.f["has_fd"]:
                        sm = sess.sysmem[e.seq]
                        al = self.allocs[sm.f["alloc"]] = Alloc(os.path.join(rp.a.mem, f"replay_{sm.f['alloc']}.bin"), sm)
                        if self.guard: self.guard.on_sysmem(sm.f["alloc"], sm.f["segs"], sm.f["size"], al.mm)
                        if al.recorded_size == w.QUEUES_SIZE and self.queues is None:
                            self.queues = (sm.f["alloc"], tggpu.QueueReader(al.mm, w.CMD_QUEUE), tggpu.QueueReader(al.mm, w.STATUS_REGION[0]))
                        conn.sendmsg([rep.f["resp"]], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, struct.pack("i", al.fd))])
                    else: conn.sendall(rep.f["resp"] + rep.f["data"])
                    if rp.out: rp.out.reply(time.monotonic_ns(), st["requests"], rep.f["resp"], rep.f["has_fd"], rep.f["data"])
                advance()
        except (ConnectionError, OSError) as ex: self.why = self.why or f"the client connection failed: {type(ex).__name__}: {ex}"
        conn.close()
        rest = [e for e in sess.reqs[i:] if e.kind != w.K_MARKER]
        if not self.why and rest:
            self.why = f"the client closed after {i} of {len(sess.reqs)} recorded requests (next: seq {rest[0].seq}: {describe(rest[0].f['req'], rest[0].f.get('payload', b''))})"
        if not self.why and sess.final_seq is not None: self.compare_pages(sess.final_seq); self.compare_final()
        if not self.why and self.page_bad: self.why = f"{len(self.page_bad)} client-written page(s) differ from the recording, first {self.page_bad[0]}"
        if not self.why and self.gpu_errors: self.why = f"the GPU here: {self.gpu_errors[0]}"
        if not self.why and self.guard and not self.guard.clean_exit(): self.why = f"the client closed, but the guard did not see the GPU torn down ({self.guard.state()})"
        if markers_got == markers_rec: mk = "equal"
        else:
            k = next((k for k, (a, b) in enumerate(zip(markers_got, markers_rec)) if a != b), min(len(markers_got), len(markers_rec)))
            show = lambda seq: w.marker_name(*seq[k]) if k < len(seq) else "(none)"
            mk = f"differ ({len(markers_got)} here, {len(markers_rec)} recorded; first at #{k}: here {show(markers_got)}, recorded {show(markers_rec)})"
        stats = dict(st, markers=mk,
                     info=dict(self.info), diffs_unlike_here=self.diff_bad, pages_unlike=self.page_bad[:5], **({"guard": self.guard.stats} if self.guard else {}),
                     **({"gpu_errors": self.gpu_errors[:3]} if self.gpu_errors else {}))   # also when the session diverged: often its cause
        for al in self.allocs.values(): al.close()   # TinyGPU.app unwires a session's sysmem at its end
        if rp.out: rp.out.note(time.monotonic_ns(), dict(event="session end", n=sess.n, ok=not self.why, why=self.why))
        return dict(n=sess.n, ok=not self.why, why=self.why, stats=stats, report=self.report)

# ── differential replay (plan V1): the recording changed before it is served ─────────────────────────────────────────
MUTATIONS = {
    "diff-late": "the GSP's first reply to an RPC the client waits on is applied only after the client's next request, which, waiting, it never sends (tinygrad times out)",
    "rpc-fail": "the first GSP_RM_ALLOC reply in the status queue carries rpc_result 0x1f (tinygrad raises)",
    "diff-early": "the GSP's writes for the RPCs of the first channel given a work-submit token, its GPFIFO rm_alloc through the token, are recorded as already there when the rm_alloc reaches the queue head, as a C++ runtime's recording can hold them (the proxy takes its diffs as it reads each request, and the runtime, sending none while it waits, runs ahead of it; STATUS.md R41): the replay must PASS all the same",
}

def mutate(events, name):
    """Changes the recording in place, as the named mutation says (MUTATIONS): the reference and the ports must behave the
    same on nondeterministic or failing variants of a boot, compared with each other rather than with the recording."""
    if name not in MUTATIONS: raise SystemExit(f"tgreplay: unknown mutation {name}; known: {', '.join(MUTATIONS)}")
    queue_alloc = next((e.f["alloc"] for e in events if e.kind == w.K_SYSMEM and e.f["size"] == w.QUEUES_SIZE), None)
    status = [k for k, e in enumerate(events) if e.kind == w.K_DIFF and e.f["alloc"] == queue_alloc and e.f["off"] >= w.STATUS_REGION[0]]
    if name == "diff-late":   # a status write seen right after a command-queue head write: the reply the client spins for
        queue_heads = {e.seq for e in events if e.kind == w.K_REQ and e.f["req"][0] == w.MMIO_WRITE and e.f["req"][2] == 0
                       and w.reg_name(e.f["req"][3]) == "NV_PGSP_QUEUE_HEAD[0]"}
        k = next(k for k in status if events[k].seq - 1 in queue_heads and events[k].seq > events[status[0]].seq)
        nxt = min(e.seq for e in events if e.kind in (w.K_REQ, w.K_REFUSED) and e.seq >= events[k].seq)   # markers do not count
        events[k] = events[k]._replace(seq=nxt + 1)
    elif name == "rpc-fail":
        for k in status:
            e = events[k]
            a = bytearray(e.f["after"])
            for o in range(0, len(a) - 0x50 + 1, 16):
                hdr = struct.unpack_from("<IIIII", a, o + 0x30)   # rpc_message_header_v: version, signature, length, function, result
                if hdr[1] == 0x43505256 and hdr[3] == 0x67:
                    struct.pack_into("<I", a, o + 0x30 + 16, 0x1f)
                    events[k] = e._replace(f=dict(e.f, after=bytes(a)))
                    return
        raise SystemExit("tgreplay: rpc-fail: no GSP_RM_ALLOC reply in the recording's status-queue writes")
    elif name == "diff-early":
        def msgs(k):   # (function, payload) of each RPC message in status diff k (rpc_message_header_v at element + 0x30, as above)
            a = events[k].f["after"]
            return [(struct.unpack_from("<I", a, o + 0x3c)[0], a[o + 0x50:]) for o in range(0, len(a) - 0x50 + 1, 16)
                    if struct.unpack_from("<I", a, o + 0x34)[0] == 0x43505256]
        nv, nv_gpu = tggpu.nv, tggpu.nv_gpu
        gpfifos, first = {}, None   # a GPFIFO's handle: the status diff with its rm_alloc reply (the golden image's channel gets no token)
        for k in status:
            for fn, p in msgs(k):
                if fn == nv.NV_VGPU_MSG_FUNCTION_GSP_RM_ALLOC and len(p) >= 16 \
                        and struct.unpack_from("<I", p, 12)[0] in (nv_gpu.AMPERE_CHANNEL_GPFIFO_A, nv_gpu.BLACKWELL_CHANNEL_GPFIFO_A):
                    gpfifos.setdefault(struct.unpack_from("<I", p, 8)[0], k)   # rpc_gsp_rm_alloc_v: hClient, hParent, hObject, hClass
                elif fn == nv.NV_VGPU_MSG_FUNCTION_GSP_RM_CONTROL and len(p) >= 12 and struct.unpack_from("<I", p, 4)[0] in gpfifos \
                        and struct.unpack_from("<I", p, 8)[0] == nv_gpu.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN:   # hClient, hObject, cmd
                    first, last = gpfifos[struct.unpack_from("<I", p, 4)[0]], k
                    break
            if first is not None: break
        if first is None: raise SystemExit("tgreplay: diff-early: no GPFIFO rm_alloc and work-submit token replies in the recording's status-queue writes")
        head = max(e.seq for e in events if e.kind == w.K_REQ and e.seq < events[first].seq and e.f["req"][0] == w.MMIO_WRITE
                   and e.f["req"][2] == 0 and w.reg_name(e.f["req"][3]) == "NV_PGSP_QUEUE_HEAD[0]")   # the rm_alloc's queue head
        for k in status:
            if events[first].seq <= events[k].seq <= events[last].seq: events[k] = events[k]._replace(seq=head)

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--listen", required=True)
    ap.add_argument("--rec", required=True)
    ap.add_argument("--mem", required=True)
    ap.add_argument("--out")
    ap.add_argument("--mutate")
    ap.add_argument("--guard", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.mem, exist_ok=True)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    sys.exit(Replay(a).run())

if __name__ == "__main__":
    main()
