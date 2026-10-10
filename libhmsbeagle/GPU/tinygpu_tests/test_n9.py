"""TODO.md plan step N9, offline: the oracle's rehearsal on the RDNA 4 card's fake (FAKE_AMD_CHIP=gfx1201, the captured Navi 48
table) with the die's 8 firmware blobs from tinygrad's cache, before N10 runs the same session on the card:
  1. cold, warm and dirty: amd_state.py's prediction, then the oracle daemon's boot-only session (amd_daemon_session.py, as
     run_amd_l0.sh drives it) on that card: AMDev takes the predicted boot (its DEBUG=2 lines, as env.sh's S8 reads them,
     and its mode1 line), and the fake reports NO ERRORS;
  2. cold and warm through tgproxy.py --guard, as N10a and N10b run: the session ends clean, the guard (at the gfx12
     addresses) sees every TLB flush the card does and audits the page tables at each; each recording replays (tgreplay.py
     --guard) to the daemon, PASS;
  3. dirty through tgproxy.py --guard: the guard refuses the SMU's mode1 reset and holds (fail-stop);
  4. no firmware in tinygrad's cache (an empty XDG_CACHE_HOME): the daemon stops at fetch_fw, "network access attempted"
     (plan step N6's exit);
  5. each given recording of the card (plan step N10's run_amd_l0.sh, env.sh's TG_AMD_L0_RDNA4) replays under the guard to
     the oracle's daemon, request for request (amd_l0_replay.py's daemon half: the C++ boot boots gfx12 from plan step N12).
No GPU, no TinyGPU.app and no network. The independent parts (each state of 1 and of 2, 3, 4, each recording of 5) run at
the same time, each in a process of its own (session() redirects fd 1) with a TMPDIR of its own, in a slot of env.sh's pool
(TINYGPU_TEST_SLOTS, TINYGPU_TEST_JOBS; TINYGPU_TEST_JOBS=1 runs them one at a time), the longest first; the results are
printed in the order above.
    python test_n9.py [recording name, under $BEAGLE_TINYGPU_DATA/recordings ...]"""
import os
os.environ["FAKE_AMD_CHIP"] = "gfx1201"   # before fake_am_gpu is imported; the daemon (amd_daemon_on_fake.py) inherits it
import sys, re, glob, json, time, shutil, signal, tempfile, threading, contextlib, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tgpaths
tgpaths.setup()
import golden_amd_boot as gab
import amd_daemon_session as ds
from test_a2i import start
import amd_l0_replay

HERE, WORK = tgpaths.HERE, tgpaths.WORK / "n9"
PY = os.environ.get("BEAGLE_PYTHON", sys.executable)
POOL = 64 << 20
TABLE = sorted(glob.glob(str(tgpaths.DATA / "discovery/1002_7550_*.json")))[0]
results = []
def check(name, ok, detail=""): results.append(f"{name}: {'PASS' if ok else 'FAIL'}{(' (' + detail + ')') if detail else ''}")

def predict(card, work):
    """amd_state.py on the card: (full or partial, mode1 or not), or None."""
    d = tempfile.mkdtemp(dir=work)
    os.symlink(card.path, os.path.join(d, "tinygpu.sock"))
    r = subprocess.run([PY, str(HERE / "amd_state.py"), TABLE], env={**os.environ, "TMPDIR": d, "TINYGRAD_PATH": tgpaths.TINYGRAD_PATH},
                       capture_output=True, text=True, timeout=120)
    card.done.acquire(timeout=30)
    if r.returncode == 3: return "full", True
    if r.returncode == 0: return ("partial" if "prediction: a partial boot" in r.stdout else "full"), False
    return None

def session(sock, out, env=None):
    """The oracle daemon's boot-only session, its stdout (tinygrad's DEBUG=2 lines) into out; None, or the error."""
    with open(out, "w") as f:
        sys.stdout.flush()
        saved = os.dup(1); os.dup2(f.fileno(), 1)
        try: ds.session(sock, POOL, env={"DEBUG": "2", **(env or {})})
        except Exception as e: return f"{type(e).__name__}: {e}"
        finally: sys.stdout.flush(); os.dup2(saved, 1); os.close(saved)
    return None

def taken(out):
    """The boot AMDev took, as env.sh's amd_boot_taken reads it: (full or partial, mode1 or not)."""
    text = open(out).read()
    first = re.search(r"AM_[A-Z0-9]+ initialized", text)
    return (None if first is None else "partial" if first.group(0) == "AM_GFX initialized" else "full"), "mode1 reset" in text

def fake_counts(card): return json.loads([l for l in card.lines if "client done: " in l][-1].split("client done: ", 1)[1])

# ── the parts, each run in a process of its own ──────────────────────────────────────────────────────────────────────
def boot(state, work):
    """1. the prediction, then the boot taken, on a card in that state"""
    card = gab.Card(work, state)
    want = predict(card, work)
    out = os.path.join(work, f"boot_{state}.txt")
    err = session(card.path, out)
    card.done.acquire(timeout=60)
    got = taken(out)
    check(f"{state}: AMDev takes the predicted boot", err is None and want == got and card.verdict() == "NO ERRORS",
          f"predicted {want}, taken {got}{', ' + err[:300] if err else ''}")
    card.srv.close()

def guarded(state, work):
    """2. through the guard, as N10a and N10b; the recording replayed to the daemon"""
    card = gab.Card(work, state)
    rec, plog, listen = (os.path.join(work, f"{x}_{state}") for x in ("rec", "proxy.log", "px.sock"))
    px = start([PY, str(HERE / "replay/tgproxy.py"), "--listen", listen, "--upstream", card.path, "--out", rec, "--guard",
                "--label", f"n9 {state}"], "tgproxy listening", plog)
    try:
        err = session(listen, os.path.join(work, f"proxied_{state}.txt"))
        card.done.acquire(timeout=60)
    finally:
        px.send_signal(signal.SIGTERM); px.wait(timeout=60)
    log = open(plog).read()
    m = re.search(r"session 1 ended: eof; .*; guard (\{.*\})", log)
    g, flushes = (json.loads(m.group(1)) if m else {}), fake_counts(card).get("tlb flushes")
    check(f"{state} through tgproxy --guard: clean, every TLB flush audited",
          err is None and "session: the AMD card (1002:7550, gfx1201): AMD triggers and guard" in log and m is not None
          and g["flushes"] == flushes and g["audits"] == flushes and card.verdict() == "NO ERRORS",
          f"guard {g}, the card's flushes {flushes}{', ' + err[:300] if err else ''}")
    rlog, rsock = os.path.join(work, f"replay_{state}.log"), os.path.join(work, f"rp_{state}.sock")
    rp = start([PY, str(HERE / "replay/tgreplay.py"), "--listen", rsock, "--rec", rec, "--mem", os.path.join(work, f"mem_{state}"), "--guard"],
               "tgreplay listening", rlog)
    try:
        err = session(rsock, os.path.join(work, f"replayed_{state}.txt"))
        rp.wait(timeout=120)
    finally:
        if rp.poll() is None: rp.kill()
    verdict = [l for l in open(rlog).read().splitlines() if l.startswith("replay session")]
    check(f"replay {state} to the daemon", err is None and rp.returncode == 0 and verdict and "PASS" in verdict[0],
          verdict[0][:300] if verdict else open(rlog).read()[-300:])
    card.srv.close()

def dirty_held(_, work):
    """3. dirty through the guard: the mode1 reset is refused and the proxy holds"""
    card = gab.Card(work, "dirty")
    plog, listen = os.path.join(work, "proxy_dirty.log"), os.path.join(work, "px_dirty.sock")
    px = start([PY, str(HERE / "replay/tgproxy.py"), "--listen", listen, "--upstream", card.path, "--out", os.path.join(work, "rec_dirty"),
                "--guard", "--label", "n9 dirty"], "tgproxy listening", plog)
    t = threading.Thread(target=session, args=(listen, os.path.join(work, "proxied_dirty.txt")), daemon=True)
    t.start()
    for _ in range(600):
        if "FAIL-STOP" in open(plog).read(): break
        t.join(timeout=0.1)
    log = open(plog).read()
    held = "FAIL-STOP" in log and "guard refused mmMP1_SMN_C2PMSG_75" in log and "mode1 reset" in log and px.poll() is None
    px.kill(); px.wait(); t.join(timeout=180)
    check("dirty through tgproxy --guard: the mode1 reset refused, the proxy holds", held and not t.is_alive(),
          [l for l in log.splitlines() if "ended:" in l][-1][:300] if "ended:" in log else log[-300:])
    card.srv.close()

def no_firmware(_, work):
    """4. no firmware in tinygrad's cache"""
    card = gab.Card(work, "cold")
    out, logs = os.path.join(work, "no_firmware.txt"), tempfile.mkdtemp(dir=work)   # the daemon's log, with its traceback, in logs
    err = session(card.path, out, env={"XDG_CACHE_HOME": tempfile.mkdtemp(dir=work), "TINYGPU_TEST_WORK": logs})
    card.done.acquire(timeout=60)
    text = open(out).read() + open(os.path.join(logs, "amd_dispatch_daemon.log")).read() + (err or "")
    check("no firmware in tinygrad's cache: the daemon stops at fetch_fw (network access attempted)",
          err is not None and "network access attempted" in text and "AM_PSP initialized" not in text, (err or "no error")[:300])
    card.srv.close()

def recording(r, work):
    """5. one of the card's own recordings"""
    rec = tgpaths.DATA / "recordings" / r
    if not (rec / "events.bin").is_file(): results.append(f"{r}: (not on this computer: its replay skipped)"); return
    good, verdict, report, out = amd_l0_replay.replay(str(rec), "daemon", tempfile.mkdtemp(dir=work), None, None)
    check(f"{r} replayed to the daemon", good, verdict[:300] + ("" if good else " " + " / ".join(report[:3])[:300]))

PART = {f.__name__: f for f in (boot, guarded, dirty_held, no_firmware, recording)}

def part(name, arg, out):
    """One part, in this process (test_n9.py --part): its result lines to <out> (JSON)."""
    WORK.mkdir(parents=True, exist_ok=True)
    PART[name](arg, tempfile.mkdtemp(dir=WORK))
    with open(out, "w") as f: json.dump(results, f)

# ── env.sh's slots (slot_take, pjob), from Python: a slot is a file created here (O_EXCL) holding its holder's pid ──────
SLOTS = os.environ.get("TINYGPU_TEST_SLOTS", os.path.expanduser("~/Library/Caches/beagle-tinygpu-tests/.slots"))
def jobs():   # env.sh's default when TINYGPU_TEST_JOBS is unset
    if os.environ.get("TINYGPU_TEST_JOBS"): return int(os.environ["TINYGPU_TEST_JOBS"])
    r = subprocess.run(["sysctl", "-n", "hw.perflevel0.physicalcpu"], capture_output=True, text=True)
    return (int(r.stdout) if r.returncode == 0 and r.stdout.strip().isdigit() else 8) * 5 // 8

def slot_take(n, reap):
    while True:
        reap()
        for i in range(1, n + 1):
            f = os.path.join(SLOTS, str(i))
            try:
                fd = os.open(f, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
                os.write(fd, f"{os.getpid()}\n".encode()); os.close(fd)
                return f
            except FileExistsError: pass
            try: pid = int(open(f).read().split()[0])   # a slot whose holder died without freeing it is free again
            except (OSError, ValueError, IndexError): continue
            try: os.kill(pid, 0)
            except ProcessLookupError:
                with contextlib.suppress(FileNotFoundError): os.unlink(f)
            except PermissionError: pass
        time.sleep(0.2)

def main():
    WORK.mkdir(parents=True, exist_ok=True)
    os.makedirs(SLOTS, exist_ok=True)
    run = tempfile.mkdtemp(dir=WORK, prefix="parts.")   # each part's result lines and its output
    # the parts in the order their results are printed; started the longest first
    parts = [("boot", s) for s in ("cold", "warm", "dirty")] + [("guarded", s) for s in ("cold", "warm")] \
        + [("dirty_held", ""), ("no_firmware", "")] + [("recording", r) for r in sys.argv[1:]]
    longest = {"guarded": 0, "recording": 1, "boot": 2, "dirty_held": 3, "no_firmware": 4}
    running, n = {}, jobs()   # Popen: (slot, its TMPDIR)
    def reap():
        for p in [p for p in running if p.poll() is not None]:
            slot, tmp = running.pop(p)
            try:   # still this part's: once it is reaped, another stage may have taken its slot as stale and made it its own
                if int(open(slot).read().split()[0]) == p.pid: os.unlink(slot)
            except (OSError, ValueError, IndexError): pass
            shutil.rmtree(tmp, ignore_errors=True)
    for i in sorted(range(len(parts)), key=lambda i: longest[parts[i][0]]):
        slot = slot_take(n, reap)
        tmp = tempfile.mkdtemp(prefix="n9.")
        with open(os.path.join(run, f"{i}.log"), "w") as log:
            p = subprocess.Popen([PY, os.path.abspath(__file__), "--part", *parts[i], os.path.join(run, f"{i}.json")],
                                 stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "TMPDIR": tmp})
        with open(slot, "w") as f: f.write(f"{p.pid}\n")   # the slot is the part's
        running[p] = (slot, tmp)
    while running:
        reap(); time.sleep(0.05)
    for i, (name, arg) in enumerate(parts):
        out = open(os.path.join(run, f"{i}.log")).read()
        if out: print(out, end="" if out.endswith("\n") else "\n")   # anything a part printed, as when one process ran them all
        try: results.extend(json.load(open(os.path.join(run, f"{i}.json"))))
        except (OSError, ValueError): results.append(f"{name} {arg}".strip() + f": FAIL (its process gave no results: see {run}/{i}.log)")
    print("=== N9"); print("\n".join(results))
    sys.exit(0 if all("PASS" in r for r in results) else 1)

if __name__ == "__main__":
    if sys.argv[1:2] == ["--part"]: part(*sys.argv[2:5])
    else: main()
