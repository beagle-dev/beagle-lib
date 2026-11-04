"""Plan step A2i: the V1 tools on the AMD card, offline, on fake_amd_device.py's card:
  1. the guard (replay/tgguard_amd.py) on synthetic events, at the RX 7900 XT's registers and the RDNA 4 card's (its
     captured table, TODO.md plan step N6): page tables pointing at a live allocation and a queue on them pass; a system
     PTE outside every allocation, a doorbell before its queue, a queue whose ring is unmapped and the SMU's mode1 reset are
     each refused; a live queue at the client's exit holds, and its dequeue read back inactive lets go; a huge page is the
     GC's PDE_PTE bit (bit 54 on gfx11, 63 on gfx12): into a live allocation it passes, a stray one and the other GC's bit
     (a table in system memory) are refused;
  2. the daemon's boot-only session (amd_daemon_session.py: boot, handoff, fini; no kernels), recorded through
     tgproxy.py --guard, cold and warm: every request goes through, and the session ends clean;
  3. each recording replays (tgreplay.py --guard) to the daemon again, and to the C++ boot's own session
     (golden_amd_boot --session): the same requests, pages and audits;
  4. the C++ session killed after its handoff, before its fini, through tgproxy.py --guard: the proxy holds (fail-stop)
     instead of closing the upstream connection under live queues.
Sections 2-3 (cold, then warm) and 4 run at the same time, each its own card, sockets, TMPDIR (the C++ client's lock) and plugin
log; the checks are printed in this order. No GPU, no TinyGPU.app and no network.
    python test_a2i.py"""
import os, sys, json, time, shutil, struct, signal, socket, tempfile, traceback, subprocess, concurrent.futures
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "replay"))
import tgpaths
tgpaths.setup()
import golden_amd_boot as gab
import amd_daemon_session as ds
import tgguard_amd
import fake_am_gpu as amg
from tinygrad.runtime.autogen.am import am

HERE, WORK = tgpaths.HERE, tgpaths.WORK / "a2i"
PY = os.environ.get("BEAGLE_PYTHON", sys.executable)
POOL = 64 << 20
results = []
def line(name, ok, detail=""): return f"{name}: {'PASS' if ok else 'FAIL'}{(' (' + detail + ')') if detail else ''}"
def check(name, ok, detail=""): results.append(line(name, ok, detail))

# ── 1. the guard on synthetic events ─────────────────────────────────────────────────────────────────────────────────
def guard_unit(meta):
    g = tgguard_amd.AMDGuard(meta, log=lambda m: None)
    R, A = g.R, g.A
    w32 = lambda name, v: g.on_write(5, A(name) * 4, struct.pack("<I", v))
    trig = lambda name, v: g.check_trigger(5, A(name) * 4, struct.pack("<I", v))
    SYS, VM = 0x80_0000_0000, 0x2000_0000_0000
    g.on_sysmem(0, [(SYS, 0x4000)], 0x4000, None)
    g.on_read(5, amg.MEMSIZE * 4, struct.pack("<I", meta["vram_size"] >> 20))
    w32("regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32", (VM >> 12) & 0xffffffff); w32("regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_HI32", VM >> 44)
    w32("regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32", 0x0 | 1); w32("regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32", 0)
    for lv, (table, child) in enumerate(((0x0, 0x1000), (0x1000, 0x2000), (0x2000, 0x3000))):   # PDB2 -> PDB1 -> PDB0 -> PTB
        g.on_write(0, table, struct.pack("<Q", child | 1))
    g.on_write(0, 0x3000, struct.pack("<Q", SYS | 0x3 | 0x70))   # valid, system, rwx
    ok = []
    ok.append(trig("regGCVM_INVALIDATE_ENG17_REQ", 1) is None)
    ok.append(g.check_trigger(2, 0x18, struct.pack("<Q", 1)) is not None)           # a doorbell before any queue
    ok.append(trig("mmMP1_SMN_C2PMSG_75", 2) is not None)                           # the mode1 reset
    w32("regGRBM_GFX_CNTL", R["regGRBM_GFX_CNTL"].encode(meid=1))
    w32("regCP_HQD_PQ_BASE", ((VM + 0x800) >> 8) & 0xffffffff); w32("regCP_HQD_PQ_BASE_HI", (VM + 0x800) >> 40)
    w32("regCP_HQD_PQ_CONTROL", R["regCP_HQD_PQ_CONTROL"].encode(queue_size=7))      # 4 << 8 = 1 KiB
    w32("regCP_HQD_PQ_DOORBELL_CONTROL", R["regCP_HQD_PQ_DOORBELL_CONTROL"].encode(doorbell_offset=6, doorbell_en=1))
    w32("regCP_HQD_PQ_RPTR_REPORT_ADDR", (VM + 0x80) & 0xffffffff); w32("regCP_HQD_PQ_RPTR_REPORT_ADDR_HI", (VM + 0x80) >> 32)
    w32("regCP_HQD_PQ_WPTR_POLL_ADDR", (VM + 0x38) & 0xffffffff); w32("regCP_HQD_PQ_WPTR_POLL_ADDR_HI", (VM + 0x38) >> 32)
    ok.append(trig("regCP_HQD_ACTIVE", 1) is None)
    ok.append(g.check_trigger(2, 0x18, struct.pack("<Q", 1)) is None)
    ok.append(not g.clean_exit())                                                   # a live queue: hold
    g.on_read(5, A("regCP_HQD_ACTIVE") * 4, struct.pack("<I", 0))                   # dequeued, read back inactive
    ok.append(g.clean_exit())
    g.on_write(0, 0x3008, struct.pack("<Q", 0x90_0000_0000 | 0x3 | 0x70))           # a system page no allocation holds
    ok.append(trig("regMMVM_INVALIDATE_ENG17_REQ", 1) is not None)
    w32("regCP_HQD_PQ_BASE", ((VM + 0x40_0000_0000) >> 8) & 0xffffffff); w32("regCP_HQD_PQ_BASE_HI", (VM + 0x40_0000_0000) >> 40)
    ok.append(trig("regCP_HQD_ACTIVE", 1) is not None)                               # its ring unmapped
    check(f"guard on synthetic events ({meta['arch']})", all(ok), f"{sum(ok)} of {len(ok)} as expected")

def guard_huge(meta):
    def audit(entry):   # a PDB0 entry under a root at VRAM 0, beside a live 2 MiB allocation
        g = tgguard_amd.AMDGuard(meta, log=lambda m: None)
        w32 = lambda name, v: g.on_write(5, g.A(name) * 4, struct.pack("<I", v))
        g.on_sysmem(0, [(SYS, 2 << 20)], 2 << 20, None)
        g.on_read(5, amg.MEMSIZE * 4, struct.pack("<I", meta["vram_size"] >> 20))
        w32("regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32", 0x0 | 1); w32("regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32", 0)
        g.on_write(0, 0x0, struct.pack("<Q", 0x1000 | 1)); g.on_write(0, 0x1000, struct.pack("<Q", 0x2000 | 1))   # PDB2 -> PDB1 -> PDB0
        g.on_write(0, 0x2000, struct.pack("<Q", entry))
        return g.check_trigger(5, g.A("regGCVM_INVALIDATE_ENG17_REQ") * 4, struct.pack("<I", 1))
    SYS, leaf = 0x80_0000_0000, amg.pde_pte(meta)
    other = am.AMDGPU_PDE_PTE_GFX12 if leaf == am.AMDGPU_PDE_PTE else am.AMDGPU_PDE_PTE
    ok = [audit(SYS | 0x3 | 0x70 | leaf) is None, audit(0x90_0000_0000 | 0x3 | 0x70 | leaf) is not None,
          audit(SYS | 0x3 | 0x70 | other) is not None]
    check(f"guard: a huge page is PDE_PTE {leaf:#x} ({meta['arch']})", all(ok), f"{sum(ok)} of {len(ok)} as expected")

# ── 2-4. recordings, replays and the hold ────────────────────────────────────────────────────────────────────────────
def start(argv, ready, log):
    f = open(log, "w")
    p = subprocess.Popen(argv, stdout=f, stderr=subprocess.STDOUT)
    for _ in range(600):
        if ready in open(log).read(): return p
        if p.poll() is not None: break
        time.sleep(0.05)
    raise RuntimeError(f"{argv[1]} did not start: {open(log).read()[-800:]}")

def env_c(sock, tmp, log):
    return {**os.environ, "APL_REMOTE_SOCK": sock, "BEAGLE_TINYGPU_NO_LAUNCH": "1", "TMPDIR": tmp, "BEAGLE_TINYGPU_LOG": log}

def record_and_replay(state, prep, work, exe, blobs):
    """2 and 3 for one state: its check lines. The card's socket and sysmem, the proxy's and the replays' sockets and the C++
    client's TMPDIR are in a directory of its own under /tmp (socket paths are short); logs and the recording in work."""
    out, priv = [], tempfile.mkdtemp(prefix=f"a2i{state[0]}.", dir="/tmp")
    try:
        card = gab.Card(priv, "cold")
        for label in prep: gab.run(card, gab_py(card.path), {**os.environ}, record=False)
        rec, plog = os.path.join(work, f"rec_{state}"), os.path.join(work, f"proxy_{state}.log")
        listen = os.path.join(priv, "px.sock")
        px = start([PY, str(HERE / "replay/tgproxy.py"), "--listen", listen, "--upstream", card.path, "--out", rec, "--guard",
                    "--label", f"a2i {state}"], "tgproxy listening", plog)
        try:
            ds.session(listen, POOL)
            card.done.acquire()
        finally:
            px.send_signal(signal.SIGTERM); px.wait(timeout=60)
        meta = json.load(open(os.path.join(rec, "meta.json")))
        sess = meta["end"]["sessions"]
        out.append(line(f"recording {state} through tgproxy --guard", sess and all(x["how"] == "eof" for x in sess) and "session: the AMD card" in open(plog).read()
                        and card.verdict() == "NO ERRORS", f"{meta['counts']}, sessions {[x['how'] for x in sess]}"))
        for client in ("daemon", "c++"):
            rlog, rsock = os.path.join(work, f"replay_{state}_{client}.log"), os.path.join(priv, f"rp_{client}.sock")
            rp = start([PY, str(HERE / "replay/tgreplay.py"), "--listen", rsock, "--rec", rec, "--mem", os.path.join(work, f"mem_{state}_{client}"), "--guard"],
                       "tgreplay listening", rlog)
            try:
                if client == "daemon": ds.session(rsock, POOL)
                else: subprocess.run([str(exe), blobs, "--session", str(POOL)], env=env_c(rsock, priv, os.path.join(work, f"replay_{state}_{client}.tglog")),
                                     capture_output=True, text=True, timeout=600)
                rp.wait(timeout=120)
            finally:
                if rp.poll() is None: rp.kill()
            log = open(rlog).read()
            verdict = [l for l in log.splitlines() if l.startswith("replay session")]
            out.append(line(f"replay {state} to the {client}", rp.returncode == 0 and verdict and "PASS" in verdict[0], verdict[0][:300] if verdict else log[-300:]))
        card.srv.close()
    finally:
        shutil.rmtree(priv, ignore_errors=True)
    return out

def die(work, exe, blobs):
    """4: the C++ session dies after its handoff: the proxy must hold. Its check line; its sockets and TMPDIR as above."""
    priv = tempfile.mkdtemp(prefix="a2id.", dir="/tmp")
    try:
        card = gab.Card(priv, "cold")
        listen, plog = os.path.join(priv, "px.sock"), os.path.join(work, "proxy_die.log")
        px = start([PY, str(HERE / "replay/tgproxy.py"), "--listen", listen, "--upstream", card.path, "--out", os.path.join(work, "rec_die"), "--guard",
                    "--label", "a2i die"], "tgproxy listening", plog)
        r = subprocess.run([str(exe), blobs, "--session", str(POOL), "--die-before-fini"], env=env_c(listen, priv, os.path.join(work, "die.tglog")),
                           capture_output=True, text=True, timeout=600)
        for _ in range(300):
            if "FAIL-STOP" in open(plog).read(): break
            time.sleep(0.1)
        held = "FAIL-STOP" in open(plog).read() and "did not see the GPU torn down" in open(plog).read() and px.poll() is None
        px.kill(); px.wait()
        card.done.acquire(timeout=60)   # the card's session ended (its sysmem files closed) before its directory goes
        out = [line("a C++ session killed before its fini: tgproxy --guard holds", held and r.returncode == 3,
                    [l for l in open(plog).read().splitlines() if "ended:" in l][-1][:300] if "ended:" in open(plog).read() else open(plog).read()[-300:])]
        card.srv.close()
    finally:
        shutil.rmtree(priv, ignore_errors=True)
    return out

def main():
    WORK.mkdir(parents=True, exist_ok=True)
    work = tempfile.mkdtemp(dir=WORK, prefix=f"{amg.CHIP}_")
    for dev in (0x744c, 0x7550):
        meta = amg.table_for(dev)[1]
        guard_unit(meta); guard_huge(meta)
    gab.WORK.mkdir(parents=True, exist_ok=True)   # (a stage's own work directory may not have it yet)
    exe = gab.WORK / "golden_amd_boot"
    tgpaths.build_cpp(HERE / "golden_amd_boot.cpp", exe)
    blobs = gab.blobs_file(work)
    parts = {"cold": lambda: record_and_replay("cold", [], work, exe, blobs), "warm": lambda: record_and_replay("warm", ["cold"], work, exe, blobs),
             "die": lambda: die(work, exe, blobs)}
    with concurrent.futures.ThreadPoolExecutor(1 if os.environ.get("TINYGPU_TEST_JOBS") == "1" else len(parts)) as ex:
        futures = {name: ex.submit(parts[name]) for name in ("warm", "cold", "die")}   # (longest first)
    for name in ("cold", "warm", "die"):
        try: results.extend(futures[name].result())
        except Exception as e:
            traceback.print_exception(e)
            check(f"the {name} part", False, f"{type(e).__name__}: {e}"[:300])
    print("=== A2i"); print("\n".join(results))
    sys.exit(0 if all("PASS" in r for r in results) else 1)

def gab_py(sock): return [sys.executable, str(HERE / "golden_amd_boot.py"), "--py-session", sock]

if __name__ == "__main__":
    main()
