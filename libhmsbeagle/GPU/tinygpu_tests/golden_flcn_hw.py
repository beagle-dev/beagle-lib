"""Golden test for TODO.md plan step C9's C++ half against the code it ports: NVFalcon::init_hw (TinyGPUHybridNVFalcon.h) is
tinygrad's NV_FLCN.init_hw (ip.py:186-210: FWSEC-FRTS on the GSP falcon, the WPR2 check, the GSP's RISC-V reset and libos
mailboxes, then booter_load on SEC2, which starts GSP-RM, and the core check) with nv_init_helper's execute_hs wrapper
(_execute_hs_with_frts_checks, plan step P1: FWSEC-FRTS's pre- and post-check reads; beagle_gsp_started from booter_load's
start, cleared if it halts with a nonzero MAILBOX0), on C5's falcon primitives. Each case runs twice against golden_gsp.py's scripted TinyGPU.app (both
falcons, WPR2, the boot scratch registers): tinygrad's and nv_init_helper's code in this process, then golden_flcn_hw.cpp.
Both must access BAR0 alike (each poll's repeated reads collapsed), fail alike (the exception's type and text), and agree on
whether booter_load started GSP-RM. Cases: the happy path; WPR2 not raised; booter_load's MAILBOX0 0x29; the GSP core not
active; FWSEC-FRTS never halts; booter_load never halts; the GSP core-select timeout. Then perturbed copies of the port must
fail. No GPU and no TinyGPU.app.
    python golden_flcn_hw.py"""
import os, sys, io, types, socket, tempfile, threading, subprocess, functools, contextlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tgpaths
tgpaths.setup()
import golden_gsp as gg    # the scripted TinyGPU.app, its falcons, tinygrad's NVDev and NV_FLCN on it
import nv_init_helper as h
from tinygrad.runtime.support.nv import ip

HERE, WORK = str(tgpaths.HERE), str(tgpaths.WORK)
R = gg.R
IMG = dict(frts_paddr=0x1220000, frts_offset=0x1ffc00000, imem_pa=gg.IMAGES["sb_imem_pa"], imem_va=gg.IMAGES["sb_imem_va"],
           imem_sz=gg.IMAGES["sb_imem_sz"], dmem_pa=gg.IMAGES["sb_dmem_pa"], dmem_sz=gg.IMAGES["sb_dmem_sz"], pkc_off=gg.IMAGES["sb_pkc_off"],
           engid=gg.IMAGES["sb_engid"], ucodeid=gg.IMAGES["sb_ucodeid"], booter_paddr=0x1240000, booter_data_off=0x9000, booter_data_sz=0x1600,
           booter_code_off=0x100, booter_code_sz=0x8e00)
LIBOS, WPR_META = 0x1234000, 0x1256000
CASES = [("happy path", dict(active=1)), ("WPR2 not raised by FWSEC-FRTS", dict(active=1, wpr2_after_sb=0)),
         ("booter_load's MAILBOX0 0x29", dict(active=1, booter_mbx0=0x29)), ("the GSP core not active", dict(active=0)),
         ("FWSEC-FRTS never halts", dict(active=1, halted=0)), ("booter_load never halts", dict(active=1, sec2_halted=0)),
         ("the GSP core-select timeout", dict(active=1, bcr_valid=0))]

def script(active, **kw):
    """golden_gsp.py's falcon script, with the GSP's RISC-V core reporting itself active (or not) after booter_load."""
    s = gg.falcon_script(**kw)
    s.vals[gg.addr(R.NV_PRISCV_RISCV_CPUCTL, gg.GSP)] = R.NV_PRISCV_RISCV_CPUCTL.encode(active_stat=active, halted=1)
    return s

def py_init_hw(sock_path, out):
    dev, fl = gg.tinygrad_dev(sock_path)   # desc_v3 is FWSEC-SB's there, which FWSEC-FRTS shares (prep_ucode's)
    fl.frts_image_paddr, fl.frts_offset = IMG["frts_paddr"], IMG["frts_offset"]
    fl.booter_image_paddr, fl.booter_data_off, fl.booter_data_sz = IMG["booter_paddr"], IMG["booter_data_off"], IMG["booter_data_sz"]
    fl.booter_code_off, fl.booter_code_sz = IMG["booter_code_off"], IMG["booter_code_sz"]
    dev.gsp = types.SimpleNamespace(libos_args_sysmem=LIBOS, wpr_meta_sysmem=WPR_META)
    try:
        with contextlib.redirect_stderr(io.StringIO()): fl.init_hw()   # tinygrad's, with nv_init_helper's execute_hs wrapper
    except Exception as e: out.append(f"error={type(e).__name__}: {e}")
    out.append(f"gsp_started={int(getattr(dev, 'beagle_gsp_started', False))}")
    return dev

def compare(exe, srv, sock_path, priv, only=None, quiet=False):
    fails = n = 0
    for name, kw in CASES:
        if only is not None and name not in only: continue
        n += 1
        results = {}
        for side in ("tinygrad", "c++"):
            s, rec, out = script(**kw), bytearray(), []
            def accept():
                conn, _ = srv.accept(); conn.settimeout(120); gg.serve(conn, s, rec)
            t = threading.Thread(target=accept, daemon=True); t.start()
            if side == "tinygrad":
                dev = py_init_hw(sock_path, out)
                dev.pci_dev.sock.close()
            else:
                args = [exe] + [f"{k}={v:#x}" for k, v in IMG.items()] + [f"libos={LIBOS:#x}", f"wpr_meta={WPR_META:#x}", f"chip_id={gg.CHIP_ID:#x}",
                                                                       f"wait_ms={gg.WAIT_MS}"]
                r = subprocess.run(args, capture_output=True, text=True, timeout=120,
                                   env=dict(os.environ, TMPDIR=priv, APL_REMOTE_SOCK=sock_path, BEAGLE_TINYGPU_NO_LAUNCH="1",
                                            BEAGLE_TINYGPU_LOG=f"{priv}/c9.log"))
                out = r.stdout.splitlines() + ([f"exit {r.returncode}: {r.stderr.strip()}"] if r.returncode else [])
            t.join(timeout=30)
            results[side] = (out, gg.collapse(s.trace))
        (po, ptr), (co, ctr) = results["tinygrad"], results["c++"]
        ok = po == co and ptr == ctr
        fails += not ok
        if not quiet: print(f"{'IDENTICAL' if ok else 'MISMATCH '} {name}: {len(ptr)} accesses; {'; '.join(po)[:150]}")
        if not ok and not quiet:
            if po != co: print(f"    tinygrad: {po}\n    c++     : {co}")
            if ptr != ctr:
                k = next((i for i, (x, y) in enumerate(zip(ptr, ctr)) if x != y), min(len(ptr), len(ctr)))
                print(f"    first access difference at #{k} of {len(ptr)}/{len(ctr)}: tinygrad {ptr[k:k + 3]} | c++ {ctr[k:k + 3]}")
    return n, fails

# a perturbed copy of the port must be caught (by the cases named)
PERTURBED = [("const uint32_t gfw = reg(NV_PGC6_AON_SECURE_SCRATCH_GROUP_05)[0].read() & 0xff;", "const uint32_t gfw = 0xff;", ["happy path"]),
             ("if (reg(NV_PFB_PRI_MMU_WPR2_ADDR_HI).read() == 0) throw", "if (reg(NV_PFB_PRI_MMU_WPR2_ADDR_LO).read() == 0) throw",
              ["WPR2 not raised by FWSEC-FRTS"]),
             ("im.booter_code_off, im.booter_code_sz, 0x0, 0x0, im.booter_data_sz, 0x10, 1, 3,",
              "im.booter_code_off, im.booter_code_sz, 0x0, 0x0, im.booter_data_sz, 0x20, 1, 3,", ["happy path"]),
             ("if (mbx.first != 0x0) {", "if (mbx.second != 0x0) {", ["booter_load's MAILBOX0 0x29"]),
             ('read_bitfields()["active_stat"] != 1)', 'read_bitfields()["halted"] != 1)', ["the GSP core not active"]),
             ("reset(falcon, true);\n\n        // set up the mailbox", "reset(falcon);\n\n        // set up the mailbox", ["happy path"])]

def main():
    exe = f"{WORK}/golden_flcn_hw"
    tgpaths.build_cpp(f"{HERE}/golden_flcn_hw.cpp", exe)
    priv = tempfile.mkdtemp(dir="/tmp", prefix="tgfh.")
    sock_path = f"{priv}/s.sock"
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path); srv.listen(1); srv.settimeout(60)
    saved_wait = ip.wait_cond
    ip.wait_cond = functools.partial(saved_wait, timeout_ms=gg.WAIT_MS)   # timeouts in milliseconds, as golden_gsp.py's
    try:
        n, fails = compare(exe, srv, sock_path, priv)
        print(f"C9 NV_FLCN.init_hw vs tinygrad and nv_init_helper: {n - fails} of {n} cases identical")
        gpu = tgpaths.REPO / "libhmsbeagle" / "GPU"
        for old, new, cases in PERTURBED:
            inc = f"{priv}/perturbed"; os.makedirs(f"{inc}/libhmsbeagle/GPU", exist_ok=True)
            text = (gpu / "TinyGPUHybridNVFalcon.h").read_text()
            assert text.count(old) == 1, old
            open(f"{inc}/libhmsbeagle/GPU/TinyGPUHybridNVFalcon.h", "w").write(text.replace(old, new))
            tgpaths.build_cpp(f"{HERE}/golden_flcn_hw.cpp", f"{priv}/golden_flcn_hw_perturbed", "-iquote", inc)   # found before the repository's
            _, f = compare(f"{priv}/golden_flcn_hw_perturbed", srv, sock_path, priv, only=cases, quiet=True)
            fails += f == 0
            print(f"perturbed ({old[:60]} -> {new[:60]}): {'REJECTED' if f else 'NOT CAUGHT'}")
    finally: ip.wait_cond = saved_wait
    print("C9 falcon boot vs tinygrad: " + ("all identical, every perturbation rejected" if not fails else f"{fails} FAILED"))
    sys.exit(1 if fails else 0)

if __name__ == "__main__":
    main()
