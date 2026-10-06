"""Golden test for TODO.md plan step C9's C++ half against the code it ports: NVFalcon::init_hw (TinyGPUNVFalcon.h) is
tinygrad's NV_FLCN.init_hw (ip.py:186-210: FWSEC-FRTS on the GSP falcon, the WPR2 check, the GSP's RISC-V reset and libos
mailboxes, then booter_load on SEC2, which starts GSP-RM, and the core check) with nv_init_helper's execute_hs wrapper
(_execute_hs_with_frts_checks, plan step P1: FWSEC-FRTS's pre- and post-check reads; beagle_gsp_started from booter_load's
start, cleared if it halts with a nonzero MAILBOX0), on C5's falcon primitives. Each case runs twice against golden_gsp.py's scripted TinyGPU.app (both
falcons, WPR2, the boot scratch registers): tinygrad's and nv_init_helper's code in this process, then golden_flcn_hw.cpp.
Both must access BAR0 alike (each poll's repeated reads collapsed), fail alike (the exception's type and text), and agree on
whether booter_load started GSP-RM. Cases: the happy path; WPR2 not raised; booter_load's MAILBOX0 0x29; the GSP core not
active; FWSEC-FRTS never halts; booter_load never halts; the GSP core-select timeout. And plan step B2's NVFalcon::cot_init_hw,
tinygrad's NV_FLCN_COT.init_hw (ip.py:311-344: the FMC boot parameters, the COT message through the FSP's EMEM, the wait for its
reply and for the RISC-V lockdown to clear) with nv_init_helper's kfsp_send_msg wrapper (the FSP queue reads, beagle_gsp_started),
on a scripted FSP: both must also leave the same boot parameters. COT cases: the happy path; the FSP never replies; the lockdown
never clears; the FSP's queues not empty. Then perturbed copies of the port must fail. No GPU and no TinyGPU.app.
    python golden_flcn_hw.py"""
import os, sys, io, mmap, types, socket, tempfile, threading, subprocess, functools, contextlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tgpaths
tgpaths.setup()
import golden_gsp as gg    # the scripted TinyGPU.app, its falcons, tinygrad's NVDev and NV_FLCN on it
import nv_init_helper as h
from tinygrad.runtime.support.nv import ip
from tinygrad.runtime.support.hcq import MMIOInterface

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

# ── the COT boot's init_hw (plan step B2) ──────────────────────────────────────────────────────────────────────────────
RG = gg.tggpu.regs("gb20x")
FSP = {n: gg.addr(getattr(RG, f"NV_PFSP_{n}")[0]) for n in ("QUEUE_HEAD", "QUEUE_TAIL", "MSGQ_HEAD", "MSGQ_TAIL", "EMEMC", "EMEMD")}
HWCFG2 = gg.addr(RG.NV_PFALCON_FALCON_HWCFG2, gg.GSP)
COT = dict(boot_args=0x1300000, fmc=0x1310000, hash=[0x11110000 + i for i in range(12)], sig=[0x22220000 + i for i in range(96)],
           pkey=[0x33330000 + i for i in range(95)] + [0x00443322])   # the public key: 381 bytes and 3 zero bytes (init_fmc_image)
COT_CASES = [("COT: the message, the FSP's reply, the lockdown cleared", dict()), ("COT: the FSP never replies", dict(replies=False)),
             ("COT: the RISC-V lockdown never clears", dict(clears=False)), ("COT: the FSP's queues not empty", dict(busy=True))]

def cot_script(replies=True, clears=True, busy=False):
    """A GB205's FSP as tinygrad's kfsp_send_msg sees it (and fake_nv_device.py plays it): once QUEUE_HEAD is written its message
    queue holds a reply (MSGQ_HEAD moves), and the GSP's RISC-V core leaves its boot-ROM lockdown (HWCFG2)."""
    s, st = gg.Script(), {"sent": False}
    s.vals[FSP["QUEUE_HEAD"]], s.vals[FSP["QUEUE_TAIL"]] = (0x10, 0) if busy else (0, 0)
    s.vals[FSP["MSGQ_TAIL"]] = 0x20
    s.vals[FSP["MSGQ_HEAD"]] = lambda a: 0x30 if st["sent"] and replies else 0x20
    s.vals[HWCFG2] = lambda a: RG.NV_PFALCON_FALCON_HWCFG2.encode(riscv_br_priv_lockdown=0 if st["sent"] and clears else 1)
    def on_write(a, v):
        if a == FSP["QUEUE_HEAD"]: st["sent"] = True
    s.hooks.append(on_write)
    return s

def collapse_pairs(trace):
    """gg.collapse, then a poll that reads two registers (kfsp_send_msg's MSGQ_HEAD, then MSGQ_TAIL) repeated: one pair kept."""
    out = []
    for x in gg.collapse(trace):
        out.append(x)
        if len(out) >= 4 and out[-4:-2] == out[-2:] and all(y[0] == "R" for y in out[-2:]): del out[-2:]
    return out

def boot_args_file(path):
    with open(path, "wb") as f: f.write(bytes(0x1000))
    fd = os.open(path, os.O_RDWR)
    m = mmap.mmap(fd, 0x1000)
    os.close(fd)
    return m

def py_cot_init_hw(sock_path, out, path):
    dev, fl = gg.tinygrad_dev(sock_path, cot=True)
    m = boot_args_file(path)
    fl.fmc_boot_args_view, fl._keep = MMIOInterface(gg.ctypes_addr(m), 0x1000, fmt='B'), m
    fl.fmc_boot_args_sysmem, fl.fmc_booter_bar1 = COT["boot_args"], COT["fmc"]
    fl.fmc_booter_hash, fl.fmc_booter_sig, fl.fmc_booter_pkey = (memoryview(b"".join(x.to_bytes(4, "little") for x in COT[k])).cast('I')
                                                                  for k in ("hash", "sig", "pkey"))
    dev.gsp = types.SimpleNamespace(libos_args_sysmem=LIBOS, wpr_meta_sysmem=WPR_META)
    try:
        with contextlib.redirect_stderr(io.StringIO()): fl.init_hw()   # tinygrad's, with nv_init_helper's kfsp_send_msg wrapper
    except Exception as e: out.append(f"error={type(e).__name__}: {e}")
    out.append(f"gsp_started={int(getattr(dev, 'beagle_gsp_started', False))}")
    return dev

def compare(exe, srv, sock_path, priv, only=None, quiet=False):
    fails = n = 0
    for name, kw in CASES + COT_CASES:
        if only is not None and name not in only: continue
        n += 1
        results = {}
        cot = name.startswith("COT")
        for side in ("tinygrad", "c++"):
            s, rec, out = cot_script(**kw) if cot else script(**kw), bytearray(), []
            bpath = f"{priv}/boot_args_{side}"
            def accept():
                conn, _ = srv.accept(); conn.settimeout(120); gg.serve(conn, s, rec)
            t = threading.Thread(target=accept, daemon=True); t.start()
            if side == "tinygrad":
                dev = py_cot_init_hw(sock_path, out, bpath) if cot else py_init_hw(sock_path, out)
                dev.pci_dev.sock.close()
            else:
                args = [exe] + [f"{k}={v:#x}" for k, v in IMG.items()] + [f"libos={LIBOS:#x}", f"wpr_meta={WPR_META:#x}", f"chip_id={gg.CHIP_ID:#x}",
                                                                       f"wait_ms={gg.WAIT_MS}"]
                if cot:
                    boot_args_file(bpath).close()
                    args += ["cot=1", f"boot_args={COT['boot_args']:#x}", f"fmc={COT['fmc']:#x}"] + \
                            [f"{k}={','.join(hex(x) for x in COT[k])}" for k in ("hash", "sig", "pkey")]
                r = subprocess.run(args, capture_output=True, text=True, timeout=120,
                                   env=dict(os.environ, TMPDIR=priv, APL_REMOTE_SOCK=sock_path, BEAGLE_TINYGPU_NO_LAUNCH="1",
                                            BEAGLE_TINYGPU_LOG=f"{priv}/c9.log", GOLDEN_BOOT_ARGS=bpath))
                out = r.stdout.splitlines() + ([f"exit {r.returncode}: {r.stderr.strip()}"] if r.returncode else [])
            t.join(timeout=30)
            results[side] = (out, collapse_pairs(s.trace) if cot else gg.collapse(s.trace), open(bpath, "rb").read() if cot else b"")
        (po, ptr, pb), (co, ctr, cb) = results["tinygrad"], results["c++"]
        ok = po == co and ptr == ctr and pb == cb
        fails += not ok
        if not quiet: print(f"{'IDENTICAL' if ok else 'MISMATCH '} {name}: {len(ptr)} accesses; {'; '.join(po)[:150]}")
        if not ok and not quiet:
            if po != co: print(f"    tinygrad: {po}\n    c++     : {co}")
            if pb != cb: print(f"    the FMC boot parameters differ: tinygrad {pb[:80].hex()} | c++ {cb[:80].hex()}")
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
             ("reset(falcon, true);\n\n        // set up the mailbox", "reset(falcon);\n\n        // set up the mailbox", ["happy path"]),
             ("buf.resize(buf.size() + (4 - payload.size() % 4), 0);", "buf.resize(buf.size() + (4 - payload.size() % 4) % 4, 0);",
              ["COT: the message, the FSP's reply, the lockdown cleared"]),
             ("cot.frtsVidmemOffset = 0x1c00000;", "cot.frtsVidmemOffset = 0x1d00000;", ["COT: the message, the FSP's reply, the lockdown cleared"]),
             ("rm_args.bootArgsOffset = libos_args_sysmem;", "rm_args.bootArgsOffset = wpr_meta_sysmem;",
              ["COT: the message, the FSP's reply, the lockdown cleared"]),
             ('{{"offs", 0}, {"blk", 0}, {"aincw", 0}, {"aincr", 1}}', '{{"offs", 0}, {"blk", 0}, {"aincw", 1}, {"aincr", 0}}',
              ["COT: the message, the FSP's reply, the lockdown cleared"])]

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
        print(f"C9 NV_FLCN.init_hw and B2's NV_FLCN_COT.init_hw vs tinygrad and nv_init_helper: {n - fails} of {n} cases identical")
        gpu = tgpaths.REPO / "libhmsbeagle" / "GPU"
        for old, new, cases in PERTURBED:
            inc = f"{priv}/perturbed"; os.makedirs(f"{inc}/libhmsbeagle/GPU", exist_ok=True)
            text = (gpu / "TinyGPUNVFalcon.h").read_text()
            assert text.count(old) == 1, old
            open(f"{inc}/libhmsbeagle/GPU/TinyGPUNVFalcon.h", "w").write(text.replace(old, new))
            tgpaths.build_cpp(f"{HERE}/golden_flcn_hw.cpp", f"{priv}/golden_flcn_hw_perturbed", "-iquote", inc)   # found before the repository's
            _, f = compare(f"{priv}/golden_flcn_hw_perturbed", srv, sock_path, priv, only=cases, quiet=True)
            fails += f == 0
            print(f"perturbed ({old[:60]} -> {new[:60]}): {'REJECTED' if f else 'NOT CAUGHT'}")
    finally: ip.wait_cond = saved_wait
    print("C9 falcon boot vs tinygrad: " + ("all identical, every perturbation rejected" if not fails else f"{fails} FAILED"))
    sys.exit(1 if fails else 0)

if __name__ == "__main__":
    main()
