"""Offline tests for plan step P2: NVIDIA's driver-unload teardown in nv_init_helper.py (BEAGLE_NV_TEARDOWN=1).
No GPU. The image clones are checked byte for byte against tinygrad's own prep_ucode/prep_booter on the real VBIOS
captured by P1 (~/.beagle/tinygpu/vbios/); the teardown sequence runs tinygrad's real falcon primitives (reset,
execute_hs) against a scripted register fake, one scenario per failure mode.
    python test_p2_teardown.py"""
import os, sys, ctypes, struct, functools, io, contextlib, types, glob, json, socket, hashlib, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tgpaths
tgpaths.setup()
urllib.request.urlopen = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("network access in an offline test"))

import nv_init_helper as h
from tinygrad.runtime.support.nv import ip
from tinygrad.runtime.support.nv.nvdev import NVDev
from tinygrad.runtime.support.nv.ip import NV_FLCN, NV_GSP
from tinygrad.runtime.autogen import nv

VBIOS = sorted(glob.glob(str(tgpaths.DATA / "vbios" / "AD107_*.rom")))
MB = 1 << 20
WPR2_HI_ADDR = 0x1FA828

# ── image clones ─────────────────────────────────────────────────────────────
class VbiosMMIO:
    def __init__(self, vbios): self.words = list(struct.unpack(f"<{len(vbios) // 4}I", vbios))
    def __getitem__(self, i):
        assert isinstance(i, slice) and i.start == 0x300000 // 4, i
        return self.words[:i.stop - i.start]

def fake_flcn(vbios=None, fw_name="ad102"):
    dev = NVDev.__new__(NVDev)
    dev.vram_size, dev.fw_name, dev.allocs = 8188 * MB, fw_name, []
    if vbios is not None: dev.mmio = VbiosMMIO(vbios)
    def alloc(size, data=None, contiguous=False, sysmem=None):
        dev.allocs.append((size, bytes(data) if data is not None else None, sysmem))
        return None, 0x10_0000 * len(dev.allocs), None
    dev._alloc_boot_mem = alloc
    fl = NV_FLCN.__new__(NV_FLCN); fl.nvdev = dev
    return fl

def frts_cmd(frts_offset):   # prep_ucode's FRTS command, as tinygrad builds it (ip.py:140-145)
    read_vbios_desc = nv.FWSECLIC_READ_VBIOS_DESC(version=0x1, size=ctypes.sizeof(nv.FWSECLIC_READ_VBIOS_DESC), flags=2)
    frst_reg_desc = nv.FWSECLIC_FRTS_REGION_DESC(version=0x1, size=ctypes.sizeof(nv.FWSECLIC_FRTS_REGION_DESC),
                                                 frtsRegionOffset4K=frts_offset >> 12, frtsRegionSize=0x100, frtsRegionMediaType=2)
    return bytes(nv.FWSECLIC_FRTS_CMD(readVbiosDesc=read_vbios_desc, frtsRegionDesc=frst_reg_desc))

def test_fwsec_clone():
    assert VBIOS, "no captured VBIOS in ~/.beagle/tinygpu/vbios (plan step P1 saves it)"
    vbios = open(VBIOS[0], "rb").read()
    fl = fake_flcn(vbios)
    h._ORIG["prep_ucode"](fl)                                    # tinygrad's own prep_ucode (unwrapped)
    (size, frts_ref, sysmem), = fl.nvdev.allocs
    desc, sig, image = h._fwsec_ucode(vbios)
    frts = h._fwsec_patch(desc, image, sig, h._FWSEC_CMD_FRTS, frts_cmd(fl.frts_offset))
    assert sysmem is False and bytes(desc) == bytes(fl.desc_v3) and bytes(frts) == frts_ref and len(frts) == size
    rvd = bytes(nv.FWSECLIC_READ_VBIOS_DESC(version=0x1, size=ctypes.sizeof(nv.FWSECLIC_READ_VBIOS_DESC), flags=2))
    sb = h._fwsec_patch(desc, image, sig, h._FWSEC_CMD_SB, rvd)
    # where they may differ: DMEM mapper's init_cmd, and the command buffer (the SB command is a prefix of FRTS's)
    hdr = nv.FALCON_APPLICATION_INTERFACE_HEADER_V1.from_buffer_copy(image[(app := desc.IMEMLoadSize + desc.InterfaceOffset):])
    ents = (nv.FALCON_APPLICATION_INTERFACE_ENTRY_V1 * hdr.entryCount).from_buffer_copy(image[app + ctypes.sizeof(hdr):])
    dmem_off = desc.IMEMLoadSize + next(e.dmemOffset for e in ents if e.id == nv.FALCON_APPLICATION_INTERFACE_ENTRY_ID_DMEMMAPPER)
    dmem = nv.FALCON_APPLICATION_INTERFACE_DMEM_MAPPER_V3.from_buffer_copy(image[dmem_off:])
    init_cmd = dmem_off + getattr(nv.FALCON_APPLICATION_INTERFACE_DMEM_MAPPER_V3, "init_cmd").offset
    cmd_off = desc.IMEMLoadSize + dmem.cmd_in_buffer_offset
    diff = [i for i in range(len(sb)) if sb[i] != frts[i]]
    assert len(sb) == len(frts) and diff and all(init_cmd <= i < init_cmd + 4 or cmd_off <= i < cmd_off + len(frts_cmd(0)) for i in diff), diff[:8]
    assert sb[cmd_off:cmd_off + len(rvd)] == rvd and frts[cmd_off:cmd_off + len(rvd)] == rvd   # both start with READ_VBIOS_DESC
    assert struct.unpack_from("<I", sb, init_cmd)[0] == 0x19 and struct.unpack_from("<I", frts, init_cmd)[0] == 0x15
    print(f"FWSEC clone: FRTS image identical to tinygrad's prep_ucode ({len(frts)} bytes, VBIOS {os.path.basename(VBIOS[0])}); "
          f"SB differs only in init_cmd and the command buffer ({len(diff)} bytes)")
    return len(frts)

def test_booter_clone():
    fl = fake_flcn()
    h._ORIG["prep_booter"](fl)                                   # tinygrad's own prep_booter (unwrapped)
    (size, load_ref, sysmem), = fl.nvdev.allocs
    img, data_off, data_sz, code_off, code_sz = h._booter_ucode("ad102", "booter_load-570.144.bin",
                                                                "8b293e19b637c5e22c87a2428d1c71bb13e0904e8a88ac6b3c6c1f2679c6e37a")
    assert bytes(img) == load_ref and sysmem is False
    assert (data_off, data_sz, code_off, code_sz) == (fl.booter_data_off, fl.booter_data_sz, fl.booter_code_off, fl.booter_code_sz)
    sizes = {}
    for chip, sha in h._BOOTER_UNLOAD_SHA.items():
        u, d_off, d_sz, c_off, c_sz = h._booter_ucode(chip, "booter_unload-570.144.bin", sha)
        sizes[chip] = len(u)
        if chip == "ad102": assert (len(u), c_off, c_sz, d_off, d_sz) == (0x9f00, 0x100, 0x4f00, 0x5000, 0x4e00), (hex(len(u)), c_off, c_sz, d_off, d_sz)
        # the literals the teardown passes to execute_hs (engid 1, ucodeid 3, pkc_off 0x10) are this firmware's own
        from tinygrad.helpers import fetch_fw
        b = fetch_fw(f"nvidia/{chip}/gsp", "booter_unload-570.144.bin", sha)
        hdr = nv.struct_nvfw_bin_hdr.from_buffer_copy(b)
        hs = nv.struct_nvfw_hs_header_v2.from_buffer_copy(b, hdr.header_offset)
        lh = nv.struct_nvfw_hs_load_header_v2.from_buffer_copy(b, hs.header_offset)
        meta = struct.unpack_from("<3I", b, hs.meta_data_offset)
        num_sig, patch_loc = struct.unpack_from("<I", b, hs.num_sig)[0], struct.unpack_from("<I", b, hs.patch_loc)[0]
        assert meta == (1, 1, 3) and num_sig == 2 and patch_loc - lh.os_data_offset == 0x10, (chip, meta, num_sig, hex(patch_loc))
    print(f"booter clone: booter_load identical to tinygrad's prep_booter ({len(img)} bytes); booter_unload parses "
          f"(ad102: 0x9f00 bytes, code 0x100+0x4f00, data 0x5000+0x4e00; ga102: {sizes['ga102']:#x} bytes; "
          f"meta (1,1,3), 2 signatures, signature at data+0x10, as the teardown's engid/ucodeid/pkc_off assume)")
    return len(img), sizes["ad102"]

def test_prep_booter_wrapper(frts_size):
    vbios = open(VBIOS[0], "rb").read()
    fl = fake_flcn(vbios)
    h._ORIG["prep_ucode"](fl); fl.nvdev.beagle_vbios = vbios
    n_before = len(fl.nvdev.allocs)
    h._TEARDOWN = False; fl.prep_booter()
    assert len(fl.nvdev.allocs) == n_before + 1 and not hasattr(fl, "beagle_sb_image_paddr")   # off: tinygrad's allocation only
    fl2 = fake_flcn(vbios); h._ORIG["prep_ucode"](fl2); fl2.nvdev.beagle_vbios = vbios
    h._TEARDOWN = True
    try:
        with contextlib.redirect_stderr(io.StringIO()): fl2.prep_booter()
    finally: h._TEARDOWN = False
    frts, load, sb, unload = fl2.nvdev.allocs
    assert [a[2] for a in fl2.nvdev.allocs] == [False] * 4 and len(sb[1]) == frts_size and len(unload[1]) == 0x9f00
    assert fl2.beagle_unload_params == (0x5000, 0x4e00, 0x100, 0x4f00)
    desc, sig, image = h._fwsec_ucode(vbios)
    rvd = bytes(nv.FWSECLIC_READ_VBIOS_DESC(version=0x1, size=ctypes.sizeof(nv.FWSECLIC_READ_VBIOS_DESC), flags=2))
    assert sb[1] == bytes(h._fwsec_patch(desc, image, sig, 0x19, rvd))   # CMD_SB (frts_tu102.c:102) with READ_VBIOS_DESC alone
    assert unload[1] == bytes(h._booter_ucode("ad102", "booter_unload-570.144.bin", h._BOOTER_UNLOAD_SHA["ad102"])[0])
    print("prep_booter wrapper: off -> tinygrad's single allocation; on -> FWSEC-SB (cmd 0x19) and Booter Unload after booter_load, in VRAM")

def test_layout_gate(frts_size, load_size, unload_size):
    import mm_trace
    base = mm_trace.trace(2, boot_images=(frts_size, load_size), quiet=True)
    p2 = mm_trace.trace(2, boot_images=(frts_size, load_size, frts_size, unload_size), quiet=True)
    for r in (base, p2):
        assert r["gpfifo_paddr"] + r["gpfifo_size"] <= 256 * MB, hex(r["gpfifo_paddr"])   # BAR1-written: must stay inside the window
        assert r["pool_paddr_end"] <= r["vram_size"] - 256 * MB, hex(r["pool_paddr_end"])  # far below the WPR at the top of VRAM
        assert not any(k[1] == "bar1_wr_dropped_beyond_256MiB" for k in r["stats"])
    print(f"layout gate: gpfifo_area 0x{base['gpfifo_paddr']:x} -> 0x{p2['gpfifo_paddr']:x} (shift 0x{p2['gpfifo_paddr'] - base['gpfifo_paddr']:x}), "
          f"inside BAR1's 256 MiB; pool ends at 0x{p2['pool_paddr_end']:x} of 0x{p2['vram_size']:x}; no BAR1 write dropped")

# ── teardown sequence on a scripted register fake ───────────────────────────
class Regs:
    """BAR0: absolute byte address -> value (int or callable(addr)); writes recorded in order."""
    def __init__(self): self.vals, self.log = {}, []
    def __getitem__(self, i):
        v = self.vals.get(i * 4, 0)
        return v(i * 4) if callable(v) else v
    def __setitem__(self, i, v): self.log.append((i * 4, v)); self.on_write(i * 4, v)
    def on_write(self, addr, v): pass

GSP, SEC2 = 0x110000, 0x840000
def addr(reg, base=0): r = reg.with_base(base); return r.base + r.off   # what NVReg.read/write access (nvdev.py:18-22)

def teardown_rig(wpr2_after_sb=0x1ffae00, bcr_valid=1, halted=1, sec2_halted=1, sec2_scrubbing=0, booter_mbx0=0,
                 wpr2_after_unload=None, sb_scratch=0, suspended=True):
    dev = NVDev.__new__(NVDev)
    dev.mmio, dev.chip_id = Regs(), 0x194000a1
    for name, arch in (("nv_ref", ""), ("dev_fb", "tu102"), ("dev_gc6_island", "ga102"), ("dev_gsp", "ga102"), ("dev_falcon_v4", "ga102"),
                       ("dev_riscv_pri", "ga102"), ("dev_fbif_v4", "ga102"), ("dev_falcon_second_pri", "ga102"), ("dev_sec_pri", "ga102"),
                       ("dev_bus", "tu102")): dev.include(name, arch)
    r, st = dev.mmio, {"booter": False}
    for base in (GSP, SEC2):
        r.vals[addr(dev.NV_PFALCON_FALCON_HWCFG2, base)] = dev.NV_PFALCON_FALCON_HWCFG2.encode(riscv=1, mem_scrubbing=sec2_scrubbing if base == SEC2 else 0)
        r.vals[addr(dev.NV_PRISCV_RISCV_BCR_CTRL, base)] = dev.NV_PRISCV_RISCV_BCR_CTRL.encode(valid=bcr_valid if base == GSP else 1)
        r.vals[addr(dev.NV_PFALCON_FALCON_DMATRFCMD, base)] = dev.NV_PFALCON_FALCON_DMATRFCMD.encode(idle=1, full=0)
        r.vals[addr(dev.NV_PFALCON_FALCON_CPUCTL, base)] = dev.NV_PFALCON_FALCON_CPUCTL.encode(halted=halted if base == GSP else sec2_halted)
    if wpr2_after_unload is None: wpr2_after_unload = 0 if booter_mbx0 == 0 else 0x1ffae00
    r.vals[addr(dev.NV_PFALCON_FALCON_MAILBOX0, SEC2)] = lambda a: booter_mbx0 if st["booter"] else 0xff
    r.vals[WPR2_HI_ADDR] = lambda a: wpr2_after_unload if st["booter"] else wpr2_after_sb
    r.vals[0x1400 + 0x15 * 4] = sb_scratch                                          # VBIOS scratch 0x15: SB error code in 15:0
    r.vals[addr(dev.NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_PRIV_LEVEL_MASK)] = dev.NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_PRIV_LEVEL_MASK.encode(read_protection_level0=1)
    r.vals[addr(dev.NV_PGC6_AON_SECURE_SCRATCH_GROUP_05[0])] = 0xff
    cpuctl_sec2 = addr(dev.NV_PFALCON_FALCON_CPUCTL, SEC2)
    def on_write(a, v):
        if a == cpuctl_sec2 and v & dev.NV_PFALCON_FALCON_CPUCTL.encode(startcpu=1): st["booter"] = True
    r.on_write = on_write
    fl = NV_FLCN.__new__(NV_FLCN)
    fl.nvdev, fl.falcon, fl.sec2 = dev, GSP, SEC2
    fl.desc_v3 = nv.FALCON_UCODE_DESC_V3(IMEMPhysBase=0, IMEMVirtBase=0, IMEMLoadSize=0x1000, DMEMPhysBase=0, DMEMLoadSize=0x400,
                                         PKCDataOffset=0x40, EngineIdMask=4, UcodeId=9)
    fl.beagle_sb_image_paddr, fl.beagle_unload_image_paddr, fl.beagle_unload_params = 0x300000, 0x340000, (0x5000, 0x4e00, 0x100, 0x4f00)
    dev.beagle_fini = {"unload_ok": suspended, "mailbox0": 0x80000000 if suspended else 0}
    return fl, dev, {"gsp_engine": addr(dev.NV_PGSP_FALCON_ENGINE), "sec_engine": addr(dev.NV_PSEC_FALCON_ENGINE),
                     "sec2_mbx0": addr(dev.NV_PFALCON_FALCON_MAILBOX0, SEC2), "sec2_mbx1": addr(dev.NV_PFALCON_FALCON_MAILBOX1, SEC2),
                     "gsp_bootvec": addr(dev.NV_PFALCON_FALCON_BOOTVEC, GSP), "sec2_bootvec": addr(dev.NV_PFALCON_FALCON_BOOTVEC, SEC2),
                     "gsp_rm": addr(dev.NV_PFALCON_FALCON_RM, GSP), "sec2_rm": addr(dev.NV_PFALCON_FALCON_RM, SEC2)}

def run_teardown(**kw):
    fl, dev, a = teardown_rig(**kw)
    h._TEARDOWN = True
    saved = ip.wait_cond
    ip.wait_cond = functools.partial(saved, timeout_ms=30)   # timeouts in milliseconds, not tinygrad's 10 s
    try:
        with contextlib.redirect_stderr(io.StringIO()): fl.fini_hw()
    finally: h._TEARDOWN, ip.wait_cond = False, saved
    writes = [w[0] for w in dev.mmio.log]
    return dev.beagle_fini, writes, dev.mmio.log, a

def hs_writes(dev, log, base):
    """The values execute_hs wrote to one falcon's BROM, BOOTVEC and DMA registers, in order."""
    regs = {"bootvec": dev.NV_PFALCON_FALCON_BOOTVEC, "paraaddr": dev.NV_PFALCON2_FALCON_BROM_PARAADDR[0],
            "engid": dev.NV_PFALCON2_FALCON_BROM_ENGIDMASK, "ucodeid": dev.NV_PFALCON2_FALCON_BROM_CURR_UCODE_ID,
            "dmabase": dev.NV_PFALCON_FALCON_DMATRFBASE, "moffs": dev.NV_PFALCON_FALCON_DMATRFMOFFS,
            "fboffs": dev.NV_PFALCON_FALCON_DMATRFFBOFFS, "cmd": dev.NV_PFALCON_FALCON_DMATRFCMD}
    by_addr = {addr(reg, base): k for k, reg in regs.items()}
    out = {k: [] for k in regs}
    for a, v in log:
        if a in by_addr: out[by_addr[a]].append(v)
    out["cmd"] = len(out["cmd"])
    return out

def expected_hs(dev, img, code_off, data_off, imemPa, imemVa, imemSz, dmemPa, dmemVa, dmemSz, pkc_off, engid, ucodeid):
    """execute_hs's register writes for these arguments (ip.py:236-262): 256-byte IMEM then DMEM transfers, then the BROM."""
    chunks = lambda sz: range(0, sz, 256)
    return {"bootvec": [imemVa], "paraaddr": [pkc_off], "engid": [engid], "ucodeid": [dev.NV_PFALCON2_FALCON_BROM_CURR_UCODE_ID.encode(val=ucodeid)],
            "dmabase": [(img + code_off - imemVa) >> 8, (img + data_off - dmemVa) >> 8],
            "moffs": [imemPa + x for x in chunks(imemSz)] + [dmemPa + x for x in chunks(dmemSz)],
            "fboffs": [imemVa + x for x in chunks(imemSz)] + [dmemVa + x for x in chunks(dmemSz)],
            "cmd": len(chunks(imemSz)) + len(chunks(dmemSz))}

def test_teardown_paths():
    d, w, log, a = run_teardown()
    td = d["teardown"]
    assert td["result"] == "done: Booter Unload lowered WPR2" and d["wpr2_down"] and d["teardown_ok"], d
    assert not any(k.endswith("_bcr_timeout") or k.endswith("_failed") for k in td), td
    fl, dev, _ = teardown_rig()
    assert (a["gsp_rm"], dev.chip_id) in log and (a["sec2_rm"], dev.chip_id) in log      # tinygrad's reset, core select succeeded
    gsp_reset, sb_start = w.index(a["gsp_engine"]), w.index(a["gsp_bootvec"])
    sec_reset, unload_start = w.index(a["sec_engine"]), w.index(a["sec2_bootvec"])
    assert gsp_reset < sb_start < sec_reset < unload_start, (gsp_reset, sb_start, sec_reset, unload_start)
    assert (a["sec2_mbx0"], 0xff) in log and (a["sec2_mbx1"], 0xff) in log                  # Booter Unload's 0xFF mailboxes
    dv = fl.desc_v3   # FWSEC-SB: FRTS's arguments (ip.py:190-193); Booter Unload: plan (b) 7, booter_load's (ip.py:204-206)
    sb = expected_hs(dev, 0x300000, 0x0, dv.IMEMLoadSize, dv.IMEMPhysBase, dv.IMEMVirtBase, dv.IMEMLoadSize, dv.DMEMPhysBase, 0x0,
                     dv.DMEMLoadSize, dv.PKCDataOffset, dv.EngineIdMask, dv.UcodeId)
    unload = expected_hs(dev, 0x340000, 0x100, 0x5000, 0x0, 0x100, 0x4f00, 0x0, 0x0, 0x4e00, 0x10, 1, 3)
    assert hs_writes(dev, log, GSP) == sb, (hs_writes(dev, log, GSP), sb)
    assert hs_writes(dev, log, SEC2) == unload, (hs_writes(dev, log, SEC2), unload)
    assert unload["cmd"] == 79 + 78 and unload["dmabase"] == [0x3400, 0x3450]
    print(f"teardown happy path: GSP reset -> FWSEC-SB -> SEC2 reset -> Booter Unload (mailboxes 0xFF) -> WPR2 down ({len(w)} writes); "
          f"execute_hs arguments as planned (SEC2: BOOTVEC 0x100, pkc 0x10, engid 1, ucode 3, 79+78 DMA chunks from 0x340000)")

    d, w, _, _ = run_teardown(suspended=False)
    assert w == [] and d["teardown"]["result"].startswith("skipped"), (w, d)
    print("teardown, GSP not suspended: no falcon register touched")

    d, w, _, a = run_teardown(wpr2_after_sb=0)
    assert a["sec_engine"] not in w and d["teardown"]["result"] == "done: WPR2 already down after FWSEC-SB" and d["teardown_ok"], d
    print("teardown, WPR2 down after FWSEC-SB: SEC2 is not reset and Booter Unload does not run (NVIDIA's skip)")

    d, w, log, a = run_teardown(bcr_valid=0)
    td = d["teardown"]
    assert td.get("gsp_bcr_timeout") and "sec2_bcr_timeout" not in td and a["gsp_bootvec"] in w and d["teardown_ok"], d
    assert w.index(a["gsp_rm"]) < w.index(a["gsp_bootvec"]) and (a["gsp_rm"], 0x194000a1) in log   # FALCON_RM still written
    print("teardown, GSP core-select timeout: logged, FALCON_RM still written, FWSEC-SB still runs (NVIDIA's kflcnReset)")

    d, w, _, _ = run_teardown(sb_scratch=0xabcd0029)
    assert d["teardown"]["sb_error"] == 0x29 and d["teardown_ok"], d
    print("teardown, FWSEC-SB error 0x29 (scratch 0x15 bits 15:0): recorded, the sequence continues")

    d, w, _, a = run_teardown(halted=0)
    td = d["teardown"]
    assert td["sb_failed"].startswith("TimeoutError") and a["sec_engine"] in w and a["sec2_bootvec"] in w and d["teardown_ok"], d
    print("teardown, FWSEC-SB never halts: recorded, SEC2 reset and Booter Unload still run (NVIDIA's control flow)")

    d, w, _, a = run_teardown(sec2_scrubbing=1)
    td = d["teardown"]
    assert td["sec2_reset_failed"].startswith("TimeoutError") and a["sec2_bootvec"] in w and d["teardown_ok"], d
    print("teardown, SEC2 reset scrub timeout: recorded, Booter Unload still runs")

    d, w, _, _ = run_teardown(sec2_halted=0)
    assert d["teardown"]["result"].startswith("failed: TimeoutError") and not d["teardown_ok"], d
    print("teardown, Booter Unload never halts: failed, power cycle needed")

    d, w, _, _ = run_teardown(booter_mbx0=0x29)
    assert d["teardown"]["result"].startswith("failed: Booter Unload returned mailbox0=0x29") and not d["wpr2_down"] and not d["teardown_ok"], d
    print("teardown, Booter Unload error 0x29: reported, WPR2 still up")

    d, w, _, _ = run_teardown(booter_mbx0=0x29, wpr2_after_unload=0)
    assert d["teardown"]["result"].startswith("failed:") and d["wpr2_down"] and not d["teardown_ok"], d
    print("teardown, Booter Unload error 0x29 with WPR2 down: still a failure, power cycle needed (plan (b) 8)")

    fl, dev, _ = teardown_rig()
    h._TEARDOWN = False; fl.fini_hw()
    assert dev.mmio.log == [] and "teardown" not in dev.beagle_fini
    print("teardown off: fini_hw does nothing")

def test_level0_rpc():
    sent = []
    gsp = NV_GSP.__new__(NV_GSP)
    gsp.cmd_q = types.SimpleNamespace(send_rpc=lambda f, m: sent.append((f, m)))
    gsp.stat_q = types.SimpleNamespace(wait_resp=lambda f: sent.append(("wait", f)))
    h._ORIG["rpc_unloading_guest_driver"](gsp); h._rpc_unloading_guest_driver_level0(gsp)
    (f1, m1), w1, (f2, m2), w2 = sent
    fast, lvl0 = nv.rpc_unloading_guest_driver_v.from_buffer_copy(m1), nv.rpc_unloading_guest_driver_v.from_buffer_copy(m2)
    assert f1 == f2 == nv.NV_VGPU_MSG_FUNCTION_UNLOADING_GUEST_DRIVER and w1 == w2 == ("wait", f1)
    assert fast.newLevel == 1 << 6 and lvl0.newLevel == 0 and (fast.bInPMTransition, fast.bGc6Entering) == (lvl0.bInPMTransition, lvl0.bGc6Entering) == (0, 0)
    assert NV_GSP.rpc_unloading_guest_driver is h._ORIG["rpc_unloading_guest_driver"]   # unset: tinygrad's FAST_UNLOAD
    import subprocess
    out = subprocess.run([sys.executable, os.path.abspath(__file__), "--level0-scenario"], capture_output=True, text=True,
                         env={**os.environ, "BEAGLE_NV_UNLOAD_LEVEL": "0"})
    assert out.returncode == 0, out.stdout + out.stderr
    print(out.stdout.strip().splitlines()[-1])

def level0_scenario():
    """BEAGLE_NV_UNLOAD_LEVEL=0 at import: the real tinygrad NV_GSP.fini_hw, whose unload posts a RUN_CPU_SEQUENCER with op 8
    (reset SEC2, start it, poll BSI; ip.py:650-657): BEAGLE's 20 s sleep must come between SEC2's start and the BSI read."""
    import time
    assert NV_GSP.rpc_unloading_guest_driver is h._rpc_unloading_guest_driver_level0, "import-time rebinding not applied"
    fl, dev, a = teardown_rig()
    events = []
    bsi = dev.NV_PGC6_BSI_SECURE_SCRATCH_14
    dev.mmio.vals[addr(bsi)] = lambda _: (events.append("bsi read"), bsi.encode(boot_stage_3_handoff=1))[1]
    cpuctl_sec2, on_write = addr(dev.NV_PFALCON_FALCON_CPUCTL, SEC2), dev.mmio.on_write
    def logged_write(ad, v):
        if ad == cpuctl_sec2: events.append("sec2 start")
        on_write(ad, v)
    dev.mmio.on_write = logged_write
    dev.mmio.vals[addr(dev.NV_PGSP_FALCON_MAILBOX0)] = 0x80000000
    gsp = NV_GSP.__new__(NV_GSP); gsp.nvdev, gsp.libos_args_sysmem = dev, 0x1234000; dev.flcn, dev.gsp = fl, gsp
    seq = bytes(nv.rpc_run_cpu_sequencer_v17_00(cmdIndex=1)) + struct.pack("<I", 8)
    levels = []
    gsp.cmd_q = types.SimpleNamespace(send_rpc=lambda f, m: levels.append(nv.rpc_unloading_guest_driver_v.from_buffer_copy(m).newLevel))
    def wait_resp(f):   # the status queue: the sequencer event, then the UNLOADING reply
        events.append("wait_resp"); gsp.run_cpu_seq(seq); events.append("reply")
    gsp.stat_q = types.SimpleNamespace(wait_resp=wait_resp)
    real_sleep = h.time.sleep
    h.time.sleep = lambda sec: events.append(f"sleep {sec}") if sec >= 1 else None
    try:
        with contextlib.redirect_stderr(io.StringIO()): gsp.fini_hw()
    finally: h.time.sleep = real_sleep
    assert levels == [0], levels
    assert [e for e in events if e != "wait_resp"] == ["sec2 start", "sleep 20", "bsi read", "reply"], events
    assert h._in_gsp_init[0] is False and dev.beagle_fini["unload_ok"], dev.beagle_fini
    print("LEVEL_0: same RPC as tinygrad's but newLevel 0 (FAST_UNLOAD is 0x40); an op-8 sequencer during the unload gets "
          "BEAGLE's 20 s sleep between SEC2's start and the BSI read; flag cleared afterwards")

def daemon_reply(dm, a, req, cmd="cmd_fini"):
    getattr(dm, cmd)(req)
    n = struct.unpack("<I", a.recv(4, socket.MSG_WAITALL))[0]
    return json.loads(a.recv(n, socket.MSG_WAITALL))

def test_hung_fini():
    import nv_dispatch_daemon as d
    from tinygrad import Device
    real, Device._opened_devices = Device._opened_devices, set()
    try:
        def rig(gsp_fini, finalize=None):
            a, b = socket.socketpair()
            dm, calls, dev_impl = d.Daemon(b), [], types.SimpleNamespace()
            dev_impl.gsp = types.SimpleNamespace(fini_hw=lambda: (calls.append("gsp.fini_hw"), gsp_fini(dev_impl)))
            dm.dev = types.SimpleNamespace(iface=types.SimpleNamespace(dev_impl=dev_impl),
                                           finalize=lambda: (calls.append("finalize"), finalize and finalize(dev_impl)))
            dm._hold = lambda: calls.append("hold")
            Device._opened_devices.add("NV")
            return dm, a, calls
        def confirmed(di): di.beagle_fini = {"unload_ok": True, "mailbox0": 0x80000000}
        dm, a, calls = rig(confirmed)
        r = daemon_reply(dm, a, {"hung": True})
        assert calls == ["gsp.fini_hw"] and r["hung"] and r["unload_ok"] and not r.get("hold") and "NV" not in Device._opened_devices, (calls, r)
        print("hung fini: the unload RPC only (no synchronize, no falcon teardown), no hold once the GSP confirms")
        def rpc_fails(di): di.beagle_fini = {"unload_ok": False}; raise RuntimeError("Timeout waiting for RPC response")
        dm, a, calls = rig(rpc_fails)
        with contextlib.redirect_stderr(io.StringIO()): r = daemon_reply(dm, a, {"hung": True})
        assert calls == ["gsp.fini_hw", "hold"] and not r["ok"] and r["hold"] and r["pid"] == os.getpid(), (calls, r)
        print("hung fini, unload RPC fails: replies hold with the pid, then holds (plan (d))")
        def late_failure(di): confirmed(di); raise OSError("socket error after the suspend was confirmed")
        dm, a, calls = rig(None, late_failure)
        with contextlib.redirect_stderr(io.StringIO()): r = daemon_reply(dm, a, {})
        assert calls == ["finalize"] and not r["ok"] and r["unload_ok"] and not r.get("hold") and "NV" not in Device._opened_devices, (calls, r)
        print("fini, a step fails after the GSP confirmed its suspend: ok false, no hold, NV dropped from atexit (partial-teardown rule)")
        def failed_teardown(di):
            confirmed(di); di.beagle_fini.update(teardown={"result": "failed: TimeoutError: not halted"}, wpr2_down=False, teardown_ok=False)
        dm, a, calls = rig(None, failed_teardown)
        r = daemon_reply(dm, a, {})
        assert r["ok"] and r["unload_ok"] and not r["teardown_ok"] and not r.get("hold") and calls == ["finalize"], (calls, r)
        print("fini, teardown failed: unload_ok stays true, teardown_ok false, normal close")
    finally: Device._opened_devices = real

def test_failed_boot():
    """A boot that fails once booter_load has started GSP-RM: unload it, and hold unless the GSP confirms (plan step P2)."""
    fl, dev, a = teardown_rig()
    saved = ip.wait_cond; ip.wait_cond = functools.partial(saved, timeout_ms=30)
    try:
        for mbx0, started in ((0x29, False), (0, True)):
            f2, d2, _ = teardown_rig(booter_mbx0=mbx0); f2.booter_image_paddr = 0x200000
            f2.execute_hs(SEC2, 0x200000, code_off=0x100, data_off=0x5000, imemPa=0, imemVa=0x100, imemSz=0x100, dmemPa=0, dmemVa=0,
                          dmemSz=0x100, pkc_off=0x10, engid=1, ucodeid=3, mailbox=1)
            assert getattr(d2, "beagle_gsp_started", False) is started, (mbx0, started)
    finally: ip.wait_cond = saved
    gsp = NV_GSP.__new__(NV_GSP); gsp.nvdev = dev; dev.gsp = gsp
    dev.mmio.vals[addr(dev.NV_PGSP_FALCON_MAILBOX0)] = 0x80000000
    saved = h._ORIG["gsp_fini_hw"], h._BOOTING[0], h._SUSPEND_TIMEOUT_S
    try:
        h._BOOTING[0], h._SUSPEND_TIMEOUT_S = dev, 0.05
        assert h.unload_after_failed_boot() is None                         # GSP-RM never started: closing is safe
        dev.beagle_gsp_started = True
        h._ORIG["gsp_fini_hw"] = lambda self: None                          # the unload RPC answered
        with contextlib.redirect_stderr(io.StringIO()): fini = h.unload_after_failed_boot()
        assert fini["unload_ok"] and fini["mailbox0"] == 0x80000000, fini
        def rpc_fails(self): raise RuntimeError("Timeout waiting for RPC response")
        h._ORIG["gsp_fini_hw"] = rpc_fails
        with contextlib.redirect_stderr(io.StringIO()): fini = h.unload_after_failed_boot()
        assert fini == {"unload_ok": False}, fini

        import nv_dispatch_daemon as d
        from tinygrad import Device
        real_getitem, real_opened = type(Device).__getitem__, Device._opened_devices
        def boot_fails(self, name): raise RuntimeError("Timeout waiting for GSP_INIT_DONE")
        type(Device).__getitem__, Device._opened_devices = boot_fails, set()
        try:
            for rpc, hold in ((lambda self: None, False), (rpc_fails, True)):
                h._ORIG["gsp_fini_hw"] = rpc
                sa, sb = socket.socketpair()
                dm = d.Daemon(sb); held = []; dm._hold = lambda: held.append(True)
                with contextlib.redirect_stderr(io.StringIO()): r = daemon_reply(dm, sa, {}, "cmd_boot")
                assert not r["ok"] and "after GSP-RM started" in r["error"] and bool(r.get("hold")) is hold and held == ([True] if hold else []), r
                assert r["unload_ok"] is (not hold) and (not hold or r["pid"] == os.getpid()), r
            dev.beagle_gsp_started = False
            sa, sb = socket.socketpair(); dm = d.Daemon(sb)
            try: dm.cmd_boot({}); raise AssertionError("a boot failure before GSP-RM started must propagate")
            except RuntimeError as e: assert "GSP_INIT_DONE" in str(e)
        finally: type(Device).__getitem__, Device._opened_devices = real_getitem, real_opened
    finally: h._ORIG["gsp_fini_hw"], h._BOOTING[0], h._SUSPEND_TIMEOUT_S = saved
    print("failed boot: booter_load returning 0 marks GSP-RM started (0x29 does not); after that the daemon unloads it and "
          "closes only once the GSP confirms, else replies hold with the pid; before it, the error propagates as before")

if __name__ == "__main__":
    if sys.argv[1:] == ["--level0-scenario"]: level0_scenario(); sys.exit(0)
    frts_size = test_fwsec_clone()
    load_size, unload_size = test_booter_clone()
    test_prep_booter_wrapper(frts_size)
    test_layout_gate(frts_size, load_size, unload_size)
    test_teardown_paths()
    test_level0_rpc()
    test_hung_fini()
    test_failed_boot()
    print("P2 teardown: all passed")
