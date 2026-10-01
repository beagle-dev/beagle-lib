"""TODO.md plan step A2j, offline: an AMD L0 recording (run_amd_l0.sh) replayed under the guard (tgreplay.py --guard, the AMD
guard) to the daemon again (amd_daemon_session.py, as recorded) and to the C++ boot's own session (golden_amd_boot --session,
the daemon's default pool), each from a fresh replay of the card's own replies: every request must equal the recorded one,
byte for byte, and every audit pass. No GPU and no TinyGPU.app: tgreplay serves the recording on a private socket.
    python amd_l0_replay.py <recording dir>"""
import os, sys, time, tempfile, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tgpaths
tgpaths.setup()
import golden_amd_boot as gab
import amd_daemon_session as ds

HERE, WORK = tgpaths.HERE, tgpaths.WORK / "a2j"
PY = os.environ.get("BEAGLE_PYTHON", sys.executable)

def replay(rec, client, work, exe, blobs):
    log, sock = os.path.join(work, f"replay_{client}.log"), os.path.join(work, f"rp_{client}.sock")
    f = open(log, "w")
    rp = subprocess.Popen([PY, str(HERE / "replay/tgreplay.py"), "--listen", sock, "--rec", rec, "--mem", os.path.join(work, f"mem_{client}"), "--guard"],
                          stdout=f, stderr=subprocess.STDOUT)
    for _ in range(600):
        if "tgreplay listening" in open(log).read(): break
        time.sleep(0.05)
    out = ""
    try:
        if client == "daemon": ds.session(sock, 0)
        else:
            env = {**os.environ, "APL_REMOTE_SOCK": sock, "BEAGLE_TINYGPU_NO_LAUNCH": "1", "TMPDIR": work, "BEAGLE_TINYGPU_LOG": os.path.join(work, "c++.log")}
            out = subprocess.run([str(exe), blobs, "--session", "0"], env=env, capture_output=True, text=True, timeout=900).stdout
        rp.wait(timeout=300)
    except Exception as e: out += f" ({type(e).__name__}: {e})"
    finally:
        if rp.poll() is None: rp.kill()
    lines = open(log).read().splitlines()
    verdict = next((l for l in lines if l.startswith("replay session")), "(no verdict)")
    report = [l for l in lines if l.startswith("  ")][:22]
    return rp.returncode == 0 and "PASS" in verdict, verdict, report, out

def main():
    rec = sys.argv[1]
    WORK.mkdir(parents=True, exist_ok=True)
    work = tempfile.mkdtemp(dir=WORK)
    exe = gab.WORK / "golden_amd_boot"
    tgpaths.build_cpp(HERE / "golden_amd_boot.cpp", exe)
    blobs = gab.blobs_file(work)
    ok = True
    for client in ("daemon", "c++"):
        good, verdict, report, out = replay(rec, client, work, exe, blobs)
        print(f"{client}: {verdict[:400]}")
        if not good:
            print("\n".join(report))
            if out: print("  client:", out.strip()[-600:])
        ok &= good
    print("A2j: the L0 replays exactly to the daemon and to the C++ boot:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
