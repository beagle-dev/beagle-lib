"""The AMDev register names tinygrad's own boot uses on this card (TODO.md plan step A2b), with no GPU: the real daemon
(amd_daemon_session.py) runs four sessions in a row on fake_amd_device.py's card, which keeps its state across them as
the GPU does:
  - cold: a full boot;
  - warm: the partial boot after the first session's fini;
  - faults: another partial boot, with an SQ MEMVIOL, a UTCL2 fault and both RAS bits posted before fini, so the fini's
    interrupt handler decodes them all (and leaves SCRATCH_REG6 1);
  - dirty: the full boot after an unclean fini: an SMU mode1 reset, then the whole boot again.
Each session also creates the queues, signals and buffers AMDDevice.__init__ creates, and the handoff's pool and staging.
The card is FAKE_AMD_CHIP's (fake_am_gpu.py; TODO.md plan step N11 runs it for each family).
    coverage() -> {"used": [...], "absent": [...], "sessions": [(label, fake's verdict, counts)]}"""
import os, sys, json, struct, socket, tempfile, threading, contextlib, io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tgpaths
import fake_amd_device as fad
import fake_am_gpu as amg
import amd_daemon_session as ds

def inject_faults(am):
    a = am.A
    base = (am.pair("regIH_RB_BASE", "regIH_RB_BASE_HI") << 8) - am.mc_base()
    sq = struct.pack("<8I", 10 | (239 << 8), 0, 0, 0, (2 << 21), (2 << 6), 0, 0)   # client GFX, SQ_INTERRUPT_ID: an error, MEMVIOL
    utcl2 = struct.pack("<8I", 10 | (0 << 8), 0, 0, 0, 0x1234, 0, 0, 0)              # client GFX, UTCL2_FAULT
    wptr = am.R["regIH_RB_WPTR"].decode(am.r.get(a("regIH_RB_WPTR"), 0))["offset"]
    am.vram_write(base + wptr * 4, sq + utcl2)
    am.r[a("regIH_RB_WPTR")] = (wptr + 16) << 2
    am.r[a(f"regGCVM_L2_PROTECTION_FAULT_STATUS{'_LO32' if amg.IPV['GC_HWIP'] >= (12, 0, 0) else ''}")] = 0x00340001   # pf_status_reg (ip.py:87)
    am.r[a("regGCVM_L2_PROTECTION_FAULT_ADDR_LO32")] = 0x2000_1234
    am.r[a("regGCVM_L2_PROTECTION_FAULT_ADDR_HI32")] = 0x2
    f = am.R["regBIF_BX0_BIF_DOORBELL_INT_CNTL"].fields
    am.r[a("regBIF_BX0_BIF_DOORBELL_INT_CNTL")] = (1 << f["ras_athub_err_event_interrupt_status"][0]) | (1 << f["ras_cntlr_interrupt_status"][0])

def coverage(pool_size=64 << 20):
    work = tempfile.mkdtemp(dir=tgpaths.WORK)
    path = os.path.join(work, "dev.sock")
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path); srv.listen(1)
    out = io.StringIO()
    with contextlib.redirect_stdout(out): gpu = fad.Gpu()
    gpu.am.reset("cold")
    done = []
    def serve():
        while True:
            try: conn, _ = srv.accept()
            except OSError: return   # closed: the sessions are over
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf): fad.serve(conn, gpu, work)
            done.append(buf.getvalue())
    threading.Thread(target=serve, daemon=True).start()
    used, absent, sessions = set(), set(), []
    for label in ("cold", "warm", "faults", "dirty"):
        names = os.path.join(work, f"names_{label}.json")
        ds.session(path, pool_size, before_fini=(lambda: inject_faults(gpu.am)) if label == "faults" else None,
                   env={"AMD_REG_NAMES_OUT": names})
        while len(done) < len(sessions) + 1: threading.Event().wait(0.05)
        verdict = [l for l in done[-1].splitlines() if "ERRORS" in l][-1]
        sessions.append((label, verdict.split(": ", 1)[1], [l for l in done[-1].splitlines() if "client done" in l][-1].split("client done: ", 1)[1]))
        j = json.load(open(names))
        used |= set(j["used"]); absent |= set(j["absent"])
    srv.close()
    return {"used": sorted(used), "absent": sorted(absent - used), "sessions": sessions}

if __name__ == "__main__":
    c = coverage()
    for label, verdict, counts in c["sessions"]: print(f"{label}: {verdict}\n  {counts[:300]}")
    print(f"{len(c['used'])} register names used, {len(c['absent'])} asked for and absent: {c['absent']}")
