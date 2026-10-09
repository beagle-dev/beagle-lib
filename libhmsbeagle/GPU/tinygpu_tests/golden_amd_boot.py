"""Golden test for TinyGPUAMDBoot.h (TODO.md plan steps A2c-A2f) against the code it ports: tinygrad's AMDev
(tinygrad/runtime/support/am/amdev.py and ip.py at the pin). Each case runs on two fake_amd_device.py cards brought to the
same state: tinygrad's AMDev in a process of its own (as PCIIfaceBase.__init__ calls it, after its RESIZE_BAR), and
golden_amd_boot.cpp on the other. Both must send TinyGPU.app the same requests byte for byte, leave the same VRAM and
registers, and print the same results or error. A state before a case comes from tinygrad's own sessions, on both cards.
  - cold: a full boot (the PSP's SOS components, its ring, the TOC, TMR and firmware, the SMU, GFX and SDMA), then fini;
  - warm: the partial boot after a cold boot's fini;
  - faults: a partial boot, then an SQ MEMVIOL, a UTCL2 fault and both RAS bits posted before fini, which decodes them;
  - dirty, after that unclean fini: tinygrad's mode1 reset and full boot, the same in C++ with --allow-mode1; the C++
    default refuses at the reset, its requests tinygrad's up to there and none after;
  - am_reset: AM_RESET=1 on a warm card: the SOS is alive, so a mode1 reset too, and a full boot over a live PSP ring
    (destroyed and re-created); the same, and the same refusal, as dirty;
  - power: AM_POWER_LIMIT=200, the power limit and every clock's whole range.
Then (A2g) the daemon's whole session against the C++ one (TinyGPUAMDDevice.h), cold and warm, at two pool sizes, and
(A2k) once more with the exit's fini run by an AMDev restored from the boot's fini state, as the crash guard runs it. The
card is FAKE_AMD_CHIP's (fake_am_gpu.py): the RX 7900 XT by default, the RX 9070 XT with gfx1201 (TODO.md plan step N12).
Then (N12) the boot told the other chip's arch (AMBootOptions.chip, as the plugin passes it): refused before init_sw, its
requests a prefix of tinygrad's with no write but LNKCTL and the discovery's index registers; AM_GMC.get_pte_flags and
is_pte_huge_page for every level, table or leaf, fragment, uncached, system, snooped and valid combination against
tinygrad's, on the card's discovery table; and that table with its GC version changed refused. No GPU, no TinyGPU.app and
no network.
    [FAKE_AMD_CHIP=gfx1201] python golden_amd_boot.py"""
import os, sys, json, struct, socket, hashlib, tempfile, threading, subprocess, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tgpaths
tgpaths.setup()
import ctypes
import fake_amd_device as fad
import amd_boot_coverage as cov
from tinygrad.runtime.autogen.am import am, fw

HERE, WORK = tgpaths.HERE, tgpaths.WORK / "a2_boot"
FW_NAMES = {"gfx1100": ("psp_13_0_0_sos.bin", "smu_13_0_0.bin", "sdma_6_0_0.bin", "gc_11_0_0_mec.bin", "gc_11_0_0_imu.bin", "gc_11_0_0_rlc.bin"),
            "gfx1201": ("psp_14_0_3_sos.bin", "smu_14_0_3.bin", "sdma_7_0_1.bin", "gc_12_0_1_pfp.bin", "gc_12_0_1_me.bin", "gc_12_0_1_mec.bin",
                        "gc_12_0_1_imu.bin", "gc_12_0_1_rlc.bin")}[fad.amg.CHIP]   # the fake card's (FAKE_AMD_CHIP; TODO.md plan step N12)
LINUX_FIRMWARE = "https://gitlab.com/kernel-firmware/linux-firmware/-/raw/0a6871b19abf5d6e024b5d208b101ae53e7fa0de"   # helpers.fetch_fw's pin
CMD = {1: "MAP_BAR", 2: "MAP_SYSMEM_FD", 3: "CFG_READ", 4: "CFG_WRITE", 6: "MMIO_READ", 7: "MMIO_WRITE", 11: "RESIZE_BAR"}

# ── tinygrad's side: a session in a process of its own (tinygrad's getenv is cached, and AMMemoryManager's VA allocator
#    is a class attribute: one process per session, as the daemon is) ──────────────────────────────────────────────
def py_session(sock_path, pause):
    from tinygrad.runtime.support.system import APLRemotePCIDevice
    from tinygrad.runtime.support.am.amdev import AMDev
    pci = object.__new__(APLRemotePCIDevice)   # the daemon's inherited-connection device, connected here
    pci.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    pci.sock.connect(sock_path)
    pci.pcibus, pci.dev_id, pci.lock_fd = "usb4", 0, None
    try: pci.resize_bar(0)   # PCIIfaceBase.__init__ (system.py:263): contextlib.suppress(Exception)
    except Exception: pass
    try:
        adev = AMDev(pci)
        print(f"booted partial={int(adev.partial_boot)} vram_size={adev.vram_size} large_bar={int(adev.large_bar)} xccs={adev.gfx.xccs} "
              f"mc_base={adev.gmc.mc_base:#x} fb_end={adev.gmc.fb_end:#x} tmr_size={adev.psp.tmr_size:#x}", flush=True)
        if pause:
            print("PAUSE", flush=True)
            sys.stdin.readline()
        adev.fini()
        print(f"fini is_err_state={int(adev.is_err_state)}", flush=True)
    except Exception as e: print(f"error {type(e).__name__}: {e}", flush=True)
    pci.sock.close()

# ── the harness ───────────────────────────────────────────────────────────────────────────────────────────────────────
class Card:
    """A fake card served on its own socket in this process; its lines kept, its requests recorded when asked."""
    def __init__(self, work, state):
        self.path = os.path.join(work, f"card{id(self)}.sock")
        self.lines, self.done = [], threading.Semaphore(0)
        self.gpu = fad.Gpu(say=self.lines.append)
        self.gpu.am.reset(state)
        self.srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.srv.bind(self.path); self.srv.listen(1)
        threading.Thread(target=self.serve, args=(work,), daemon=True).start()
    def serve(self, work):
        while True:
            try: conn, _ = self.srv.accept()
            except OSError: return
            fad.serve(conn, self.gpu, work)
            self.done.release()
    def verdict(self): return [l for l in self.lines if "ERRORS" in l][-1].split(": ", 1)[1]
    def state(self):
        am = self.gpu.am
        return {k: bytes(v) for k, v in am.vram.items() if any(v)}, dict(am.r), {k: dict(v) for k, v in am.hqd.items() if v}

def run(card, argv, env, inject=None, record=True):
    """One session on the card: the program's output lines and, if record, its requests."""
    card.gpu.record = [] if record else None
    p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env={**os.environ, **env})
    out = []
    for line in p.stdout:
        line = line.rstrip("\n")
        if line == "PAUSE":
            if inject: inject(card.gpu.am)
            p.stdin.write("go\n"); p.stdin.flush()
            continue
        out.append(line)
    err = p.stderr.read()
    p.wait()
    card.done.acquire()   # the session ended on the card
    rec, card.gpu.record = card.gpu.record, None
    return [l for l in out if l.startswith(("booted", "fini", "error", "open:", "handoff"))], rec, err

def run_daemon(card, pool_size):
    """The real daemon's session on the card (amd_daemon_session.py): its handoff reply, as the C++ side prints it, and its requests."""
    import amd_daemon_session as ds
    card.gpu.record = []
    _, info, _ = ds.session(card.path, pool_size)
    card.done.acquire()
    rec, card.gpu.record = card.gpu.record, None
    keys = [k for k in info if k != "ok"]
    return "handoff" + "".join(f" {k}={info[k]}" for k in keys), rec

def describe(rec, i):
    hdr = rec[i]
    if len(hdr) != 33: return f"payload {hdr[:16].hex()}{'...' if len(hdr) > 16 else ''} ({len(hdr)} bytes)"
    cmd, _, bar, a0, a1, a2 = struct.unpack("<BIIQQQ", hdr)
    return f"{CMD.get(cmd, cmd)} bar={bar} a0={a0:#x} a1={a1:#x} a2={a2:#x}"

def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y: return i
    return None if len(a) == len(b) else min(len(a), len(b))

def blobs_file(work):
    lines = []
    for n in FW_NAMES:
        md5 = hashlib.md5(f"{LINUX_FIRMWARE}/amdgpu/{n}".encode()).hexdigest()
        path = os.path.join(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/Library/Caches")), "tinygrad/downloads/fw", md5)
        if not os.path.exists(path): sys.exit(f"{n} is not in tinygrad's download cache ({path}); the golden never downloads")
        lines.append(f"{n} {path} {fw.hashes[n]}")
    p = os.path.join(work, "blobs.txt")
    open(p, "w").write("\n".join(lines) + "\n")
    return p

def other_ip_set(table):
    """The discovery table with its GC IP's minor version one higher: an IP set no am::kChips entry has."""
    b = bytearray(table)
    bhdr = am.struct_binary_header.from_buffer(b)
    ihdr = am.struct_ip_discovery_header.from_buffer(b, bhdr.table_list[am.IP_DISCOVERY].offset)
    for d in range(ihdr.num_dies):
        off = ihdr.die_info[d].die_offset
        ip_off = off + ctypes.sizeof(am.struct_die_header)
        for _ in range(am.struct_die_header.from_buffer(b, off).num_ips):
            ip = am.struct_ip_v4.from_buffer(b, ip_off)
            if ip.hw_id == am.hw_id_map[am.GC_HWIP]: ip.minor += 1
            ip_off += 8 + (8 if ihdr.base_addr_64_bit else 4) * ip.num_base_address
    return bytes(b)

def pte_flags(exe, work):
    """N12: the C++ AM_GMC.get_pte_flags and is_pte_huge_page against tinygrad's (ip.py:175-192) for every combination, on the
    card's discovery table; and an unlisted IP set refused."""
    import types
    from tinygrad.runtime.support.am.ip import AM_GMC
    from tinygrad.runtime.support.amd import import_soc
    table, meta = fad.amg.card()
    ver = fad.amg.IPV["GC_HWIP"]
    gmc = object.__new__(AM_GMC)
    gmc.adev = types.SimpleNamespace(ip_ver={am.GC_HWIP: ver}, soc=types.SimpleNamespace(module=import_soc(ver)))
    want = []
    for lv in range(4):
        for tbl in (0, 1):
            for frag in range(32):
                for bits in range(16):
                    u, sy, sn, v = bits >> 3 & 1, bits >> 2 & 1, bits >> 1 & 1, bits & 1
                    f = gmc.get_pte_flags(lv, bool(tbl), frag, bool(u), bool(sy), bool(sn), bool(v))
                    want.append(f"{lv} {tbl} {frag} {u} {sy} {sn} {v} -> {f:#x} {int(bool(gmc.is_pte_huge_page(lv, f)))}")
    p = os.path.join(work, "table.bin")
    open(p, "wb").write(table)
    got = subprocess.run([str(exe), "--pte-flags", p], capture_output=True, text=True).stdout.splitlines()
    d = first_diff(want, got)
    print(f"get_pte_flags and is_pte_huge_page, {len(want)} combinations (GC {'.'.join(map(str, ver))}): " +
          ("IDENTICAL" if d is None else f"MISMATCH at #{d}: tinygrad {want[d] if d < len(want) else '(none)'}, c++ {got[d] if d < len(got) else '(none)'}"), flush=True)
    open(p, "wb").write(other_ip_set(table))
    got_bad = subprocess.run([str(exe), "--pte-flags", p], capture_output=True, text=True).stdout.splitlines()
    gc = f"GC {ver[0]}.{ver[1] + 1}.{ver[2]}"
    refused = len(got_bad) == 1 and got_bad[0].startswith("error RuntimeError: the C++ AM boot is for the IP versions") and gc in got_bad[0]
    print(f"an unlisted IP set ({gc}): {'REFUSED' if refused else 'MISMATCH'}: {got_bad[:1]}", flush=True)
    return d is None and refused

def main():
    WORK.mkdir(parents=True, exist_ok=True)
    work = tempfile.mkdtemp(dir=WORK)
    exe = WORK / "golden_amd_boot"
    tgpaths.build_cpp(HERE / "golden_amd_boot.cpp", exe)
    blobs = blobs_file(work)
    py = lambda sock, pause=False: [sys.executable, __file__, "--py-session", sock] + (["--pause"] if pause else [])
    cpp = lambda extra=(): [str(exe), blobs] + list(extra)
    env_for = lambda card, extra=None: {"APL_REMOTE_SOCK": card.path, "BEAGLE_TINYGPU_NO_LAUNCH": "1", "TMPDIR": work,
                                        "BEAGLE_TINYGPU_LOG": str(WORK / "golden_amd_boot.log"), **(extra or {})}
    def prep(card, steps):   # tinygrad's own sessions bring a card to a case's state
        for label in steps:
            out, _, err = run(card, py(card.path, pause=label == "faults"), env_for(card), inject=cov.inject_faults if label == "faults" else None, record=False)
            if not out or not out[-1].startswith("fini"): sys.exit(f"preparing the card ({label}): {out} {err[-500:]}")
    cases = [("cold", "cold", [], {}, {}), ("warm", "cold", ["cold"], {}, {}), ("faults", "cold", ["cold"], {"inject": True}, {}),
             ("dirty", "cold", ["cold", "faults"], {"allow_mode1": True}, {}), ("am_reset", "cold", ["cold"], {"allow_mode1": True}, {"AM_RESET": "1"}),
             ("power", "cold", ["cold"], {}, {"AM_POWER_LIMIT": "200"})]
    ok = True
    for name, state, steps, opt, env in cases:
        a, b = Card(work, state), Card(work, state)
        prep(a, steps); prep(b, steps)
        inject = cov.inject_faults if opt.get("inject") else None
        out_py, rec_py, err_py = run(a, py(a.path, pause=bool(inject)), env_for(a, env), inject=inject)
        out_c, rec_c, err_c = run(b, cpp((["--pause"] if inject else []) + (["--allow-mode1"] if opt.get("allow_mode1") else [])), env_for(b, env), inject=inject)
        why = []
        if out_py != out_c: why.append(f"results differ: tinygrad {out_py} c++ {out_c}")
        d = first_diff(rec_py, rec_c)
        if d is not None:
            why.append(f"requests differ at #{d} of {len(rec_py)}/{len(rec_c)}: tinygrad {describe(rec_py, d) if d < len(rec_py) else '(none)'}, "
                       f"c++ {describe(rec_c, d) if d < len(rec_c) else '(none)'}")
        sa, sb = a.state(), b.state()
        if sa[0] != sb[0]: why.append(f"VRAM differs in {len(set(sa[0]) ^ set(sb[0]) | {p for p in set(sa[0]) & set(sb[0]) if sa[0][p] != sb[0][p]})} pages")
        if sa[1] != sb[1] or sa[2] != sb[2]: why.append("the registers differ")
        if a.verdict() != "NO ERRORS" or b.verdict() != "NO ERRORS": why.append(f"the fakes saw errors: {a.verdict()} / {b.verdict()}")
        n_req = sum(1 for r in rec_py if len(r) == 33)
        print(f"{name}: {n_req} requests, {out_py[0] if out_py else '(no output)'}: " + ("IDENTICAL" if not why else "MISMATCH: " + "; ".join(why)), flush=True)
        if why and err_c.strip(): print("  c++ stderr:", err_c.strip()[-800:])
        ok &= not why
        if name in ("dirty", "am_reset"):   # the C++ default: refused where tinygrad sends the mode1 reset
            c = Card(work, state)
            prep(c, steps)
            out_r, rec_r, _ = run(c, cpp(), env_for(c, env))
            hdr = lambda r: struct.unpack("<BIIQQQ", r) if len(r) == 33 else (None,) * 6
            write_at = next(i for i, r in enumerate(rec_py) if hdr(r)[0] == 4 and hdr(r)[3] == 4)   # the PCI_COMMAND write: bus master off
            reset_at = max(i for i in range(write_at) if hdr(rec_py[i])[0] == 3 and hdr(rec_py[i])[3] == 4)   # its read, which the C++ never sends
            refused = len(out_r) == 1 and "mode1 reset" in out_r[0] and out_r[0].startswith("error RuntimeError")
            prefix = rec_r == rec_py[:reset_at]
            print(f"{name}, refused (the default): {'REFUSED before the mode1 reset' if refused and prefix else 'MISMATCH'}: {out_r[0][:120] if out_r else '(no output)'}; "
                  f"{sum(1 for r in rec_r if len(r) == 33)} requests, tinygrad's {'first ' + str(sum(1 for r in rec_py[:reset_at] if len(r) == 33)) if prefix else '(not a prefix)'}")
            ok &= refused and prefix
        for card in (a, b): card.srv.close()
    # A2g: the daemon's whole session (boot, AMDDevice.__init__, cmd_handoff, the exit's finalize) against the C++ one
    # (A2k: and with that finalize on an AMDev restored from the boot's fini state, as the crash guard runs it)
    for name, steps, pool, extra in (("session cold, a 64 MiB pool", [], 64 << 20, []), ("session warm, a 64 MiB pool", ["cold"], 64 << 20, []),
                                     ("session warm, the default pool (half the VRAM)", ["cold"], 0, []),
                                     ("session warm, a 64 MiB pool, finalized by an AMDev restored from the boot's fini state", ["cold"], 64 << 20,
                                      ["--restore-fini"])):
        a, b = Card(work, "cold"), Card(work, "cold")
        prep(a, steps); prep(b, steps)
        info_py, rec_py = run_daemon(a, pool)
        out_c, rec_c, err_c = run(b, cpp(["--session", str(pool)] + extra), env_for(b))
        why = []
        if extra and out_c[-1:] != ["fini is_err_state=0 queues_off=1"]: why.append(f"the restored AMDev's fini: {out_c[-1:]}")
        got = dict(kv.split("=", 1) for kv in (out_c[0].split()[1:] if out_c and out_c[0].startswith("handoff") else []))
        want = dict(kv.split("=", 1) for kv in info_py.split()[1:])
        if got != want: why.append(f"the handoff differs: {sorted((k, want.get(k), got.get(k)) for k in set(want) | set(got) if want.get(k) != got.get(k))[:6]} {out_c[:2]}")
        d = first_diff(rec_py, rec_c)
        if d is not None:
            why.append(f"requests differ at #{d} of {len(rec_py)}/{len(rec_c)}: tinygrad {describe(rec_py, d) if d < len(rec_py) else '(none)'}, "
                       f"c++ {describe(rec_c, d) if d < len(rec_c) else '(none)'}")
        sa, sb = a.state(), b.state()
        if sa[0] != sb[0]: why.append(f"VRAM differs in {len(set(sa[0]) ^ set(sb[0]) | {p for p in set(sa[0]) & set(sb[0]) if sa[0][p] != sb[0][p]})} pages")
        if sa[1] != sb[1] or sa[2] != sb[2]: why.append("the registers differ")
        if a.verdict() != "NO ERRORS" or b.verdict() != "NO ERRORS": why.append(f"the fakes saw errors: {a.verdict()} / {b.verdict()}")
        print(f"{name}: {sum(1 for r in rec_py if len(r) == 33)} requests, {want.get('nmaps')} mappings, pool {int(want.get('pool_size', 0)) >> 20} MiB at "
              f"{int(want.get('pool_va', 0)):#x}: " + ("IDENTICAL" if not why else "MISMATCH: " + "; ".join(why)), flush=True)
        if why and err_c.strip(): print("  c++ stderr:", err_c.strip()[-800:])
        ok &= not why
        for card in (a, b): card.srv.close()
    # N12: told another chip than the card's IP versions say: refused before init_sw (after the discovery, as any IP set)
    other = {"gfx1100": "gfx1201", "gfx1201": "gfx1100"}[fad.amg.CHIP]
    a, b = Card(work, "cold"), Card(work, "cold")
    out_py, rec_py, _ = run(a, py(a.path), env_for(a))
    out_c, rec_c, _ = run(b, cpp(["--chip", other]), env_for(b))
    hdr = lambda r: struct.unpack("<BIIQQQ", r) if len(r) == 33 else (None,) * 6
    writes = [hdr(r) for r in rec_c if hdr(r)[0] in (4, 7)]   # CFG_WRITE, MMIO_WRITE
    only = all(h[0] == 4 or (h[2] == 5 and h[3] in (0x0, 0x18)) for h in writes)   # LNKCTL; the index registers (BAR5 dwords 0x00, 0x06)
    refused = len(out_c) == 1 and out_c[0].startswith("error RuntimeError: the C++ AM boot is for the IP versions") and f"says {other}" in out_c[0]
    print(f"the boot told {other} on the {fad.amg.CHIP} card: {'REFUSED before init_sw' if refused and only and rec_c == rec_py[:len(rec_c)] else 'MISMATCH'}: "
          f"{out_c[:1]}; {sum(1 for r in rec_c if len(r) == 33)} requests, tinygrad's first, {len(writes)} writes", flush=True)
    ok &= refused and only and rec_c == rec_py[:len(rec_c)] and a.verdict() == b.verdict() == "NO ERRORS"
    for card in (a, b): card.srv.close()
    ok &= pte_flags(exe, work)
    print(f"A2c-A2g AMDev boot, AMDDevice and the handoff, C++ against tinygrad ({fad.amg.CHIP}):", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--py-session": py_session(sys.argv[2], "--pause" in sys.argv)
    else: main()
