"""Canonicalizer and tolerant comparator for V1 recordings (TODO.md plan step V1, inv:verification#4): are two recordings the
same boot, up to what may differ between two runs of it? Used on the L0 hardware recordings pairwise, to learn and
document what differs (the masks), and on any two recordings of the same client (a replay's --out, a port's).

The same rules apply to both sides, session by session:
  1. epochs are delimited by BAR0 writes, CFG_WRITE, MAP_SYSMEM_FD, MAP_BAR and RESIZE_BAR;
  2. those must match exactly, in order: offset and bytes, (size, contiguous) and status for MAP_SYSMEM_FD, the BAR and its
     size (not its address) for MAP_BAR;
  3. within an epoch, BAR0 and config reads: the same set of (offset, length); per address the sequence of distinct values
     (a poll that reads "busy" 3 times here and 40 there is the same poll); the counts are reported, not compared. A polled
     register (read more than once on a side) that ends at the same value on both sides may differ in the values seen before
     it (C5 on the RTX 4060: the plugin's faster reset caught HWCFG2 still scrubbing once, 0x77b7 before 0x67b7, where the
     daemon's saw only 0x67b7; a boot's MAILBOX0 poll saw 0 before 0x80000000 in one run and not in the other). In the
     VBIOS (the PROM window prep_ucode reads, ip.py:110) a KiB that reads as all 0xff on one side only is masked: on the
     RTX 4060 (L0) 128 KiB at 0xd6c00, an IFR image ("NVGI"), read as 0xff on the cold boot and as the image after a
     teardown, and tinygrad's parse of the rest is the same. In a falcon's DMATRFCMD the idle bit is masked: whether the
     previous 256-byte transfer has finished when it is read (tinygrad waits only for full to clear, ip.py:213,220; L0: the
     teardown's DMAs read busy in one run and idle in the others);
  4. BAR1 writes: per epoch a map of address to byte, the last write winning (order and coalescing within an epoch do not
     matter: the GPU acts on VRAM only after a BAR0 kick); BAR1 reads: reported, not compared;
  5. sysmem pages at the recordings' page snapshots (points where the client waits for a reply, tgproxy.py), matched in order
     by the epoch they fall in: per page the contents; TinyGPU.app's segment list, where the client left it at the start of
     an allocation (e.g. past the radix3 level-0 list), and the GSP's log buffer are masked, and so are GPU timestamps: the
     second 64-bit word of a 16-byte semaphore whose first (the payload) is equal and nonzero on both sides, where a release
     with release_timestamp writes the GPU's timer (ops_nv.py:175; L0: tinygrad's signal at the end of its 16 KiB signal
     page). The C++ runtime waits on no reply (it polls its semaphores in sysmem), so its phase has no snapshot of its own:
     what it wrote is compared at the fini's first RPC and at the end;
  6. the GSP's messages (the status queue, rebuilt from the recorded diffs): its replies in order, by function and payload;
     its asynchronous events (anything but a reply to an RPC) by function and count only;
  7. device addresses are symbolized: each 4 KiB page of a MAP_SYSMEM_FD allocation becomes (allocation, page), in 64-bit
     words of BAR1 writes that are sysmem PTEs (their address field, >> 12), in BAR0 mailbox lo/hi pairs, and in sysmem pages
     and GSP messages whose bytes differ (a page with the same bytes on both sides is equal as it is: firmware images hold
     words that merely look like device addresses); the BARs' physical addresses (in GspSystemInfo) become their numbers;
  markers are compared as an ordered sequence, for information.
With --until-marker ID a session that has marker ID is compared only up to its first one (plan V1's markers: tgwire.MARKERS;
e.g. 0x101, the C++ runtime's programs loaded): two recordings of different tests compared through their common steps. With
--from-marker ID, only from it: the requests and the GSP's messages after it (e.g. 0x102, the fini: the teardown), without
the pages (the first snapshot after a marker holds what was written before it too).
    python tgcanon.py <recording A> <recording B> [--context N] [--until-marker ID] [--from-marker ID]
Prints, per session, EQUIVALENT, or every difference by category with examples (and, where the epochs stop being aligned, that
point with the phase it falls in (the last marker before it) and the N delimiters before it), then what differed within the
rules (poll counts, BAR1 reads, GSP events, markers, masked bytes); exits 0 only if every session is equivalent."""
import sys, struct, argparse, hashlib, pathlib, collections
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import tgwire as w
import tggpu
from tinygrad.runtime.autogen import nv

PAGE = 0x1000
PROM = 0x300000   # the VBIOS window prep_ucode reads (ip.py:110)
DELIMS = (w.MMIO_WRITE, w.CFG_WRITE, w.MAP_SYSMEM_FD, w.MAP_BAR, w.RESIZE_BAR)
REPLIES = {nv.NV_VGPU_MSG_FUNCTION_GSP_RM_ALLOC, nv.NV_VGPU_MSG_FUNCTION_GSP_RM_CONTROL, nv.NV_VGPU_MSG_FUNCTION_SET_PAGE_DIRECTORY,
           nv.NV_VGPU_MSG_FUNCTION_UNLOADING_GUEST_DRIVER, nv.NV_VGPU_MSG_FUNCTION_ALLOC_MEMORY}

class Side:
    """One recording's session, canonicalized."""
    def __init__(self, events, blobs, split=None):
        self.blobs, self.split, self.start = blobs, split, 0   # split: compare from the first request after this seq
        self.R = tggpu.regs("ada")
        self.pte = self.R.NV_MMU_VER2_PTE
        dma = self.R.NV_PFALCON_FALCON_DMATRFCMD
        self.dma_cmd, self.idle = {base + dma.base + dma.off for base in w.FALCONS}, dma.mask("idle")
        names = w.reg_names()
        self.mailbox_pairs = {a: n for a, n in names.items() if n.endswith("MAILBOX0") or n.endswith("MAILBOX1")}
        self.sym = {}       # page IOVA -> (allocation, page)
        self.bars = {}      # BAR physical address -> BAR number
        self.initial = {}   # allocation -> the segment-list bytes TinyGPU.app wrote at its start
        self.queue_alloc, self.libos, self.log_alloc = None, None, None
        self.epochs, self.pages_by_epoch, self.info = [], collections.defaultdict(list), collections.Counter()
        self.build(events)

    def symbolize_word(self, v):
        page = self.sym.get(v & ~(PAGE - 1))
        if page is not None: return f"iova{page}+{v & (PAGE - 1):#x}"
        if v in self.bars: return f"bar{self.bars[v]}"
        return None

    def canon_bytes(self, data, alloc=None, base=0, extent=0):
        """Sysmem contents at allocation offset base: 64-bit words that are device or BAR addresses symbolized; where the
        client left TinyGPU.app's segment list at the allocation's start (within extent: the longer of the two recordings'
        lists, which differ from boot to boot), a word equal to this side's list, or zero, is masked."""
        out, init = [], self.initial.get(alloc, b"")
        for o in range(0, len(data) - 7, 8):
            v = struct.unpack_from("<Q", data, o)[0]
            if base + o + 8 <= extent and (v == 0 or data[o:o + 8] == init[base + o:base + o + 8]): out.append("seglist"); continue
            s = self.symbolize_word(v) if v else None
            out.append(s if s else v)
        return tuple(out)

    def canon_bar1(self, off, data):
        """A BAR1 write: 8-byte words that are valid sysmem PTEs get their address symbolized (NVPageTableEntry.set_entry)."""
        if len(data) != 8 or off % 8: return data
        v = struct.unpack("<Q", data)[0]
        f = self.pte.decode(v)
        if f["valid"] and f["aperture"] == 2 and (s := self.sym.get(f["address_sys"] << 12)):
            return ("pte", s, v & ~self.pte.mask("address_sys"))
        return data

    def build(self, events):
        reqs = [e for e in events if e.kind == w.K_REQ]
        replies = {e.seq: e for e in events if e.kind == w.K_REPLY}
        sysmem = {e.seq: e for e in events if e.kind == w.K_SYSMEM}
        self.marker_seqs = [(e.seq, e.f["id"], e.f["arg"]) for e in events if e.kind == w.K_MARKER]
        cur = dict(delim=("start",), r0=collections.defaultdict(list), cfg=collections.defaultdict(list), w1={}, n_r0=collections.Counter(), n_r1=0)
        self.epochs.append(cur)
        epoch_of_seq, pending_mbx = {}, {}
        for e in reqs:
            cmd, dev, bar, a0, a1, a2 = e.f["req"]
            if self.split is not None and not self.start and e.seq > self.split:   # --from-marker: an epoch starts at the marker
                cur = dict(delim=("from marker",), seq=e.seq, r0=collections.defaultdict(list), cfg=collections.defaultdict(list), w1={},
                           n_r0=collections.Counter(), n_r1=0)
                self.epochs.append(cur)
                self.start = len(self.epochs) - 1
            rep = replies.get(e.seq)
            if cmd == w.MAP_BAR and rep and rep.f["reply"][0] == 0: self.bars[rep.f["reply"][1]] = bar
            if cmd == w.MAP_SYSMEM_FD and rep and rep.f["reply"][0] == 0 and e.seq in sysmem:
                sm = sysmem[e.seq]
                for i, p in enumerate(w.page_iovas(sm.f["segs"], sm.f["mapped"])): self.sym[p] = (sm.f["alloc"], i)
                self.initial[sm.f["alloc"]] = sm.f["seglist"]
                if sm.f["size"] == w.QUEUES_SIZE and self.queue_alloc is None: self.queue_alloc = sm.f["alloc"]
            if cmd in DELIMS and not (cmd == w.MMIO_WRITE and bar != 0):
                if cmd == w.MMIO_WRITE:
                    v = struct.unpack("<I", e.f["payload"])[0] if len(e.f["payload"]) == 4 else None
                    name = self.mailbox_pairs.get(a0)
                    if name and v is not None:   # a 64-bit address written as lo/hi mailbox pairs (ip.py:196-197, 243-245)
                        key = name[:-1]
                        if name.endswith("0"): pending_mbx[key] = v; delim = ("mbx", name, "lo")
                        else:
                            full = pending_mbx.pop(key, 0) | v << 32
                            s = self.symbolize_word(full)
                            delim = ("mbx", name, s if s else full)   # the lo half is judged with the hi half
                            if name == "NV_PGSP_FALCON_MAILBOX1": self.libos = self.sym.get(full & ~(PAGE - 1))
                    else: delim = ("w0", w.reg_name(a0), e.f["payload"])
                elif cmd == w.CFG_WRITE: delim = ("cfgw", a0, a1, a2)
                elif cmd == w.MAP_SYSMEM_FD: delim = ("sysmem", a0, a1, rep.f["reply"][0] if rep else None, rep.f["reply"][1] if rep else None)
                elif cmd == w.MAP_BAR: delim = ("bar", bar, rep.f["reply"][0] if rep else None, rep.f["reply"][2] if rep else None)
                else: delim = ("resize", bar)
                cur = dict(delim=delim, seq=e.seq, r0=collections.defaultdict(list), cfg=collections.defaultdict(list), w1={},
                           n_r0=collections.Counter(), n_r1=0)
                self.epochs.append(cur)
            elif cmd == w.MMIO_WRITE and bar == 1:
                data = self.canon_bar1(a0, e.f["payload"])
                if isinstance(data, tuple): cur["w1"][a0] = data
                else:
                    for k, b in enumerate(data): cur["w1"][a0 + k] = b
            elif cmd == w.MMIO_READ and rep and rep.f["reply"][0] == 0:
                if bar == 0:
                    vals, key = cur["r0"][(a0, a1)], (a0, a1)
                    v = rep.f["data"] if a1 > 8 else int.from_bytes(rep.f["data"], "little")
                    if not vals or vals[-1] != v: vals.append(v)
                    cur["n_r0"][key] += 1
                else: cur["n_r1"] += 1
            elif cmd == w.MMIO_READ and rep: cur["r0"][(bar, a0, a1, "failed")].append(rep.f["reply"][0])
            elif cmd == w.CFG_READ and rep:
                vals = cur["cfg"][(a0, a1)]
                if not vals or vals[-1] != rep.f["reply"][1]: vals.append(rep.f["reply"][1])
            epoch_of_seq[e.seq] = len(self.epochs) - 1
        if self.split is not None and not self.start: self.start = len(self.epochs)   # nothing after the marker
        # page snapshots, by the epoch they fall in (a snapshot's seq is a request's, or the session end's)
        last_epoch = len(self.epochs) - 1
        for e in events:
            if e.kind == w.K_PAGES:
                ep = epoch_of_seq.get(e.seq, last_epoch)
                for pg, sha in e.f["pages"]:
                    self.pages_by_epoch[ep].append(((e.f["alloc"], pg), sha))
        # the GSP's log buffer: the allocation the libos arguments' first region (LOGINIT) points at (NV_GSP.init_libos_args)
        if self.libos is not None:
            last = {}
            for e in events:
                if e.kind == w.K_PAGES and e.f["alloc"] == self.libos[0]:
                    for pg, sha in e.f["pages"]:
                        if pg == self.libos[1]: last = self.blobs.get(sha)
            if last:
                pa = nv.LibosMemoryRegionInitArgument.from_buffer_copy(last[:32]).pa
                self.log_alloc = (self.sym.get(pa & ~(PAGE - 1)) or (None,))[0]
        # the GSP's messages: the status queue rebuilt from the recorded diffs, read in order
        self.messages = []
        if self.queue_alloc is not None:
            buf = bytearray(w.QUEUES_SIZE)
            reader = tggpu.QueueReader(buf, w.STATUS_REGION[0])
            by_seq = collections.defaultdict(list)
            for e in events:
                if e.kind == w.K_DIFF and e.f["alloc"] == self.queue_alloc and e.f["off"] >= w.STATUS_REGION[0]: by_seq[e.seq].append(e)
            for seq in sorted(by_seq):   # each snapshot's writes together, the header (write pointer) last, as tgreplay applies them
                for e in sorted(by_seq[seq], key=lambda e: -e.f["off"]): buf[e.f["off"]:e.f["off"] + len(e.f["after"])] = e.f["after"]
                for fn, payload, elem, ok in reader.new(): self.messages.append((seq, fn, payload))

    def page(self, key, sha, extent):
        data = self.blobs.get(sha)
        return self.canon_bytes(data, key[0], key[1] * PAGE, extent) if len(data) == PAGE else data

    def phase(self, seq):
        """The last step marker before request seq: a tinygrad function's or the C++ runtime's (not the shim's events or
        MemoryManager.map_range, which are many)."""
        last = [w.marker_name(mid, arg) for s, mid, arg in self.marker_seqs if s < seq and mid not in (19, 100, 101)]
        return f"after marker {last[-1]}" if last else "before any marker"

def rom_hidden(va, vb):
    """KiB of a VBIOS read that are all 0xff on one side only (rule 3), or None if anything else differs."""
    if len(va) != len(vb): return None
    kib = 0
    for x, y in zip(va, vb):
        if not isinstance(x, bytes) or not isinstance(y, bytes) or len(x) != len(y): return None
        for o in range(0, len(x), 1024):
            p, q = x[o:o + 1024], y[o:o + 1024]
            if p == q: continue
            if p.count(0xff) != len(p) and q.count(0xff) != len(q): return None
            kib += 1
    return kib

def without_idle(vals, idle):
    """A DMA command register's distinct values with the idle bit cleared (rule 3)."""
    out = []
    for v in vals:
        if isinstance(v, int): v &= ~idle
        if not out or out[-1] != v: out.append(v)
    return out

def timestamps(ca, cb):
    """How many differing words of two canonicalized pages are GPU timestamps (rule 5), or 0 if any other word differs."""
    n = 0
    for i, (x, y) in enumerate(zip(ca, cb)):
        if x == y: continue
        if not (i % 2 and ca[i - 1] == cb[i - 1] and isinstance(ca[i - 1], int) and ca[i - 1] and isinstance(x, int) and isinstance(y, int)): return 0
        n += 1
    return n if len(ca) == len(cb) else 0

def describe_delim(d):
    if d[0] == "w0": return f"BAR0 write {d[1]} = {struct.unpack('<I', d[2])[0]:#x}" if len(d[2]) == 4 else f"BAR0 write {d[1]} ({len(d[2])} bytes)"
    if d[0] == "mbx": return f"BAR0 write {d[1]} ({d[2] if isinstance(d[2], str) else hex(d[2])})"
    return " ".join(str(x) for x in d)

def compare(a, b, ctx, examples=3):
    """(structural divergence or None, {category: [examples]}, {category: count}, notes): every difference within aligned
    epochs is collected by category (for the L0 comparisons, where what differs becomes the documented masks); the walk stops
    only where the epochs' delimiters stop matching, since nothing after that is aligned."""
    found, counts, notes = collections.defaultdict(list), collections.Counter(), collections.Counter()
    def add(cat, what):
        counts[cat] += 1
        if len(found[cat]) < examples: found[cat].append(what)
    structural = None
    for i in range(max(len(a.epochs) - a.start, len(b.epochs) - b.start)):
        ia, ib = a.start + i, b.start + i
        ea, eb = (a.epochs[ia] if ia < len(a.epochs) else None), (b.epochs[ib] if ib < len(b.epochs) else None)
        where = f"epoch {ia}"
        if ea is None or eb is None:
            side = a if ea else b
            structural = f"{where} ({side.phase((ea or eb)['seq'])}): one recording has more epochs ({describe_delim((ea or eb)['delim'])})"; break
        if ea["delim"] != eb["delim"]:
            before = [describe_delim(x["delim"]) for x in a.epochs[max(a.start, ia - ctx):ia]]
            structural = f"{where} ({a.phase(ea['seq'])}): {describe_delim(ea['delim'])} | {describe_delim(eb['delim'])}\n    before it: " + "; ".join(before); break
        after = describe_delim(ea["delim"])
        for k in set(ea["r0"]) ^ set(eb["r0"]):
            add("BAR0 reads of different registers", f"{where} (after {after}): {w.reg_name(k[0]) if len(k) == 2 else k}")
        for k in set(ea["r0"]) & set(eb["r0"]):
            if ea["r0"][k] != eb["r0"][k]:
                if len(k) == 2 and k[0] == PROM and (kib := rom_hidden(ea["r0"][k], eb["r0"][k])) is not None:
                    notes[f"VBIOS KiB read as 0xff on one side only (masked): {kib}"] += 1; continue
                if len(k) == 2 and k[0] in a.dma_cmd and without_idle(ea["r0"][k], a.idle) == without_idle(eb["r0"][k], a.idle):
                    notes[f"DMA idle bits differ at {w.reg_name(k[0])} (masked)"] += 1; continue
                if len(k) == 2 and ea["r0"][k][-1] == eb["r0"][k][-1] and max(ea["n_r0"][k], eb["n_r0"][k]) > 1:
                    notes[f"a poll's values before the last differ at {w.reg_name(k[0])}"] += 1; continue
                add(f"BAR0 read values differ", f"{where} (after {after}): {w.reg_name(k[0]) if len(k) == 2 else k}: {ea['r0'][k]!r:.60} | {eb['r0'][k]!r:.60}")
            elif len(k) == 2 and ea["n_r0"][k] != eb["n_r0"][k]: notes[f"poll counts differ at {w.reg_name(k[0])}"] += 1
        if ea["cfg"] != eb["cfg"]: add("config reads differ", f"{where}: {dict(ea['cfg'])} | {dict(eb['cfg'])}")
        if ea["w1"] != eb["w1"]:
            diff = sorted(set(ea["w1"].items()) ^ set(eb["w1"].items()), key=lambda x: x[0])
            add("BAR1 writes differ", f"{where} (after {after}): {len(diff)} bytes, first {str(diff[:2])[:160]}")
        if ea["n_r1"] != eb["n_r1"]: notes["BAR1 read counts differ"] += 1
        sa, sb = (dict(a.pages_by_epoch.get(ia, [])), dict(b.pages_by_epoch.get(ib, []))) if a.split is None else ({}, {})
        for key in set(sa) | set(sb):
            if sa.get(key) == sb.get(key): continue   # the same bytes: equal whatever they mean (words are symbolized only when needed)
            if key[0] == a.queue_alloc and key[1] * PAGE >= w.STATUS_REGION[0]: continue   # the GSP's status queue: compared as messages (rule 6)
            extent = max(len(a.initial.get(key[0], b"")), len(b.initial.get(key[0], b"")))
            both = key in sa and key in sb
            if both and (ca := a.page(key, sa[key], extent)) == (cb := b.page(key, sb[key], extent)): continue
            if key[0] in (a.log_alloc, b.log_alloc): notes["GSP log buffer pages differ"] += 1; continue
            if both and (n := timestamps(ca, cb)): notes["GPU timestamps differ (masked)"] += n; continue
            add("sysmem pages differ at a snapshot" if key in sa and key in sb else "a sysmem page changed in one recording only",
                f"{where} (after {after}): allocation {key[0]} page {key[1]}")
    in_scope = lambda side, seq: side.split is None or seq > side.split
    ra = [(fn, a.canon_bytes(p)) for s, fn, p in a.messages if fn in REPLIES and in_scope(a, s)]
    rb = [(fn, b.canon_bytes(p)) for s, fn, p in b.messages if fn in REPLIES and in_scope(b, s)]
    for k in range(max(len(ra), len(rb))):
        if k >= len(ra) or k >= len(rb) or ra[k] != rb[k]:
            add("GSP replies differ", f"reply #{k}: {nv.rpc_fns.get(ra[k][0]) if k < len(ra) else None} | {nv.rpc_fns.get(rb[k][0]) if k < len(rb) else None}")
    ev_a = collections.Counter(fn for s, fn, _ in a.messages if fn not in REPLIES and in_scope(a, s))
    ev_b = collections.Counter(fn for s, fn, _ in b.messages if fn not in REPLIES and in_scope(b, s))
    if ev_a != ev_b: notes[f"GSP event counts differ: {dict(ev_a)} | {dict(ev_b)}"] += 1
    if [(m, g) for s, m, g in a.marker_seqs if in_scope(a, s)] != [(m, g) for s, m, g in b.marker_seqs if in_scope(b, s)]: notes["markers differ"] += 1
    return structural, found, counts, notes

def sessions(rec):
    events, blobs = w.read(rec)
    out, cur = [], None
    for e in events:
        if e.kind == w.K_NOTE and e.f.get("event") == "session":
            cur = []; out.append(cur)
        if cur is not None: cur.append(e)
    return out, blobs

def until(events, mid):
    """A session's events before its first marker mid, and whether it has one."""
    cut = next((e.seq for e in events if e.kind == w.K_MARKER and e.f["id"] == mid), None)
    return ([e for e in events if e.seq is None or e.seq < cut], True) if cut is not None else (events, False)

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--context", type=int, default=20)
    ap.add_argument("--until-marker", type=lambda s: int(s, 0))
    ap.add_argument("--from-marker", type=lambda s: int(s, 0))
    args = ap.parse_args()
    sa, ba = sessions(args.a)
    sb, bb = sessions(args.b)
    ok = len(sa) == len(sb)
    if not ok: print(f"the recordings have {len(sa)} and {len(sb)} sessions")
    for n, (ea, eb) in enumerate(zip(sa, sb), 1):
        scope, split = "", (None, None)
        if args.until_marker is not None:
            (ea, ha), (eb, hb) = until(ea, args.until_marker), until(eb, args.until_marker)
            if ha != hb: ok = False; print(f"session {n}: DIFFERENT: marker {args.until_marker:#x} is in one recording only"); continue
            scope = f" through marker {w.marker_name(args.until_marker, 0)}" if ha else ""
        if args.from_marker is not None:
            split = tuple(next((e.seq for e in ev if e.kind == w.K_MARKER and e.f["id"] == args.from_marker), None) for ev in (ea, eb))
            if (split[0] is None) != (split[1] is None): ok = False; print(f"session {n}: DIFFERENT: marker {args.from_marker:#x} is in one recording only"); continue
            if split[0] is not None: scope += f" from marker {w.marker_name(args.from_marker, 0)}"
        A, B = Side(ea, ba, split[0]), Side(eb, bb, split[1])
        structural, found, counts, notes = compare(A, B, args.context)
        size = (f"{len(A.epochs) - A.start} epochs, {sum(A.split is None or s > A.split for s, _, _ in A.messages)} GSP messages, "
                + (f"{sum(len(v) for v in A.pages_by_epoch.values())} page snapshots" if A.split is None else "pages not compared") + scope)
        if structural: ok = False; print(f"session {n}: DIFFERENT, and not aligned after {structural}")
        elif counts: ok = False; print(f"session {n}: DIFFERENT within aligned epochs ({size})")
        else: print(f"session {n}: EQUIVALENT ({size})")
        for cat in sorted(counts):
            print(f"    {cat}: {counts[cat]}")
            for x in found[cat]: print(f"        {x}")
        for k, v in sorted(notes.items()): print(f"    within the rules: {k}" + (f" ({v})" if v > 1 else ""))
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
