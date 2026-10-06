"""Golden test for TinyGPUNVBoot.h (TODO.md plan step C11) against the code it ports, with nv_init_helper's patches as
the daemon applies them (imported here as the daemon imports it).
  C11a (early): PCIIfaceBase.__init__'s BAR resize and NVDev.__init__'s first statements (system.py:263; nvdev.py:75-147):
  map_bar(0), _early_ip_init under the WARM guard (Ada's wait_for_reset suppressed, COT's FSP wait logged), _early_mmu_init
  with the BAR check.
  C11b (flcn): then NV_FLCN.init_sw (ip.py:97-184): prep_ucode's VBIOS walk and FRTS patch under the VBIOS capture, and
  prep_booter with P2's teardown images (FWSEC-SB and Booter Unload), on C4's firmware.
  C11c, C11d (sw): the whole init_sw: NV_FLCN's (or on a COT boot NV_FLCN_COT's: the FMC boot parameters and the FMC image),
  then NV_GSP's (ip.py:347-455, 600-627): the queues and rm args, the libos args, the GSP image in radix3, the bootloader,
  the WPR meta and the two prequeued RPCs. Every sysmem allocation's contents are compared too.
Each case runs twice against the same fake TinyGPU.app (golden_mm.py's, plus PCI config space, RESIZE_BAR, the case's BARs,
BAR0 registers and VBIOS; every request recorded): tinygrad's code in this process, then golden_boot.cpp. Both must print
the same results (the chip, the VRAM size, the root page table; the falcon images' addresses and load parameters; or the
error, type and text), send the same requests byte for byte (a poll's repeated reads counted once: their number depends on
speed), leave the same VRAM and save the same VBIOS copy. The cold cases use the cards' own registers and VBIOS (the L0
recordings' replies) when this Mac has the recordings, and their streams must then equal the recordings' as far as the step
goes. The VBIOS variants are the captured dumps and copies made malformed or ambiguous at the places the walk reads. Then
perturbed copies of the port must fail. No GPU and no TinyGPU.app.
    python golden_boot.py"""
import os, re, sys, glob, socket, struct, hashlib, tempfile, subprocess, contextlib, collections, pathlib, shutil, ctypes
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tgpaths
tgpaths.setup()
sys.path.insert(0, str(tgpaths.HERE / "replay"))
import nv_init_helper  # the daemon's patches: the WARM guard, the suppressed wait_for_reset, COT's log, the BAR check, P2's images
import golden_mm as gm
import tgwire
from tinygrad.runtime.support.system import APLRemotePCIDevice, RemoteCmd
from tinygrad.runtime.support.nv.nvdev import NVDev, NVMemoryManager
from tinygrad.runtime.support.memory import TLSFAllocator
from tinygrad.runtime.autogen import nv

HERE, WORK, MB = gm.HERE, gm.WORK, gm.MB
WPR2_HI, BOOT_0, BOOT_42, SCRATCH_42, I2CS = 0x1fa828, 0x0, 0xa00, 0x1183a4, 0xad00bc
VBIOS_WINDOW = (0x300000, 0x100000)   # prep_ucode's read (ip.py:110)
RECORDINGS = pathlib.Path(os.environ.get("BEAGLE_TINYGPU_DATA", str(pathlib.Path.home() / ".beagle/tinygpu"))) / "recordings"
# the cards' own values (the L0 recordings' replies): config space, BARs, the registers the early boot reads
AD107 = dict(cfg={0x0: 0x288210de, 0x4: 0x00100007, 0x8: 0x030000a1, 0x2c: 0x89361043},
             bars={0: (0x10490c000, 16 * MB), 1: (0x10590c000, 256 * MB), 3: (0x1195c8000, 32 * MB)},
             regs={BOOT_0: 0x197000a1, BOOT_42: 0x197a1000, SCRATCH_42: 0x1ffc}, l0="20260925-204611_mittag-leffler_cold")
GB205 = dict(cfg={0x0: 0x2f0410de, 0x4: 0x00100007, 0x8: 0x030000a1, 0x2c: 0x89e71043},
             bars={0: (0x118000000, 64 * MB), 1: (0x11c000000, 256 * MB), 3: (0x12fcbc000, 32 * MB)},
             regs={BOOT_0: 0x1b5000a1, BOOT_42: 0x1b5a1000, SCRATCH_42: 0x2fc3, I2CS: 0xff}, l0="20260927-093033_Marcs-Mac-Studio-490_gb205_l0")

# ── the VBIOS: the captured dumps, the L0 recording's own, and malformed or ambiguous copies ───────────────────────────────
def captured_vbios():
    return [pathlib.Path(p).read_bytes() for p in sorted(glob.glob(str(tgpaths.DATA / "vbios" / "AD107_*.rom")))]

def l0_vbios(name):
    """The VBIOS the recorded boot read: the reply to its prep_ucode read."""
    rec = RECORDINGS / name
    if not rec.exists(): return None
    ev, seqs = tgwire.read(rec)[0], set()
    for e in ev:
        if e.kind == tgwire.K_REQ and e.f["req"][0] == RemoteCmd.MMIO_READ and e.f["req"][2] == 0 and tuple(e.f["req"][3:5]) == VBIOS_WINDOW: seqs.add(e.seq)
        elif e.kind == tgwire.K_REPLY and e.seq in seqs: return bytes(e.f["data"])
    return None

def vbios_layout(vbios):
    """Where prep_ucode's walk reads what the variants change: the first image's code type, the FWSEC_PROD ucode entry and
    the last used entry after it (as ip.py:110-138 finds them)."""
    b, off, code_types = memoryview(vbios), 0, []
    while True:
        pci = b[off + nv.OFFSETOF_PCI_EXP_ROM_PCI_DATA_STRUCT_PTR:].cast('H')[0]
        ln = b[off + pci + nv.OFFSETOF_PCI_DATA_STRUCT_IMAGE_LEN:].cast('H')[0] * nv.PCI_ROM_IMAGE_BLOCK_SIZE
        ct_off = off + pci + nv.OFFSETOF_PCI_DATA_STRUCT_CODE_TYPE
        code_types.append(ct_off)
        if b[ct_off] == nv.NV_BCRT_HASH_INFO_BASE_CODE_TYPE_VBIOS_BASE: bs = ln
        if b[ct_off] == nv.NV_BCRT_HASH_INFO_BASE_CODE_TYPE_VBIOS_EXT: exp = off - bs; break
        off += ln
    bh = nv.BIT_HEADER_V1_00.from_buffer_copy(b[0x1b0:0x1b0 + ctypes.sizeof(nv.BIT_HEADER_V1_00)])
    entries = []
    for i in range(bh.TokenEntries):
        t = nv.BIT_TOKEN_V1_00.from_buffer_copy(b[0x1b0 + bh.HeaderSize + i * bh.TokenSize:])
        if t.TokenId != nv.BIT_TOKEN_FALCON_DATA or t.DataVersion != 2: continue
        tp = exp + nv.BIT_DATA_FALCON_DATA_V2.from_buffer_copy(b[t.DataPtr & 0xffff:]).FalconUcodeTablePtr
        uh = nv.FALCON_UCODE_TABLE_HDR_V1.from_buffer_copy(b[tp:])
        entries = [tp + uh.HeaderSize + j * uh.EntrySize for j in range(uh.EntryCount)]
    fwsec = next(e for e in entries if b[e] == nv.FALCON_UCODE_ENTRY_APPID_FWSEC_PROD)
    later = [e for e in entries if e > fwsec and b[e] != 0][-1]
    return dict(first_code_type=code_types[0], second_code_type=code_types[1], fwsec_entry=fwsec, later_entry=later)

def patched(vbios, **edits):
    v = bytearray(vbios)
    for off, data in edits.values(): v[off:off + len(data)] = data
    return bytes(v)

class Case:
    def __init__(self, name, chip, mode="early", cfg=None, bars=None, regs=None, l0=False, expect=None, stream=None, vbios=None, teardown=True):
        self.name, self.mode, self.l0, self.stream, self.teardown = name, mode, chip["l0"] if l0 else None, stream, teardown
        self.cfg = {**chip["cfg"], **(cfg or {})}
        self.bars = {**chip["bars"], **(bars or {})}
        self.regs = {**chip["regs"], **(regs or {})}
        self.vbios = vbios
        self.expect = expect   # a regex the result must match (search, over all its lines)

def cases():
    cs = [
        Case("AD107 cold", AD107, l0=True, expect=r"^chip 0x197000a1 AD107 ad102 mmu 2 fmc 0 vram 8585740288 bar1 268435456 large 0 root "),
        Case("GB205 cold", GB205, l0=True, expect=r"^chip 0x1b5000a1 GB205 gb202 mmu 3 fmc 1 vram 12820938752 bar1 268435456 large 0 root "),
        Case("AD107, bus mastering off", AD107, cfg={0x4: 0x00100002}, expect=r"^chip 0x197000a1 AD107 "),
        Case("GB205, the FSP ready at the 5th read", GB205, regs={I2CS: lambda n: 0xff if n >= 5 else 0}, expect=r"^chip 0x1b5000a1 GB205 "),
        Case("AD107, WPR2 up", AD107, regs={WPR2_HI: 0x2fad}, expect=r"^error WarmGPUError: WARM GPU: WPR2 is up \(NV_PFB_PRI_MMU_WPR2_ADDR_HI=0x00002fad\)",
             stream=["RESIZE_BAR", "MAP_BAR", "MMIO_READ NV_PFB_PRI_MMU_WPR2_ADDR_HI"]),
        Case("GB205, a 512 MiB BAR1", GB205, bars={1: (0x11c000000, 512 * MB)}, expect=r"^error BarLayoutError: BAR1 is 512 MiB with 12227 MiB of VRAM \(large_bar=False\)"),
        Case("AD107, a BAR1 as large as VRAM", AD107, bars={1: (0x10590c000, 8188 * MB)}, expect=r"^error BarLayoutError: BAR1 is 8188 MiB with 8188 MiB of VRAM \(large_bar=True\)"),
        Case("an unknown architecture", AD107, regs={BOOT_42: 0x187a1000}, expect=r"^error KeyError: 24$"),
        Case("GB205, the FSP never ready", GB205, regs={I2CS: 0x7}, expect=r"^error TimeoutError: FSP not ready: NV_THERM_I2CS_SCRATCH=0x00000007 after N ms"),
    ]
    roms = captured_vbios()
    rec_rom = l0_vbios(AD107["l0"])
    lay = vbios_layout(roms[0])
    ok_images = (r"\nflcn frts 0x1ffa00000 at 0x1200000 desc .* booter at 0x1211000 data 0x[0-9a-f]+\+0x[0-9a-f]+ code 0x[0-9a-f]+\+0x[0-9a-f]+\n"
                 r"teardown sb at 0x121f000 unload at 0x1230000 data 0x5000\+0x4e00 code 0x100\+0x4f00$")
    cs += [
        Case("AD107 cold, NV_FLCN.init_sw" + (" (the recording's VBIOS)" if rec_rom else ""), AD107, mode="flcn", l0=rec_rom is not None,
             vbios=rec_rom or roms[0], expect=ok_images),
        *[Case(f"NV_FLCN.init_sw on the captured VBIOS {i + 1} of {len(roms)}", AD107, mode="flcn", vbios=r, expect=ok_images) for i, r in enumerate(roms)],
        Case("NV_FLCN.init_sw, the teardown off", AD107, mode="flcn", vbios=roms[0], teardown=False, expect=r"booter at 0x1211000 .*\nteardown off$"),
        Case("AD107 cold, the whole init_sw" + (" (the recording's VBIOS)" if rec_rom else ""), AD107, mode="sw", l0=rec_rom is not None,
             vbios=rec_rom or roms[0], expect=r"\ngsp rm_args 0x[0-9a-f]+ libos 0x[0-9a-f]+ .* seq 2 classes 0xc56f 0xc9c0 0xc7b5 0xc9b0$"),
        Case("GB205 cold, the whole init_sw", GB205, mode="sw", l0=True,
             expect=r"^chip 0x1b5000a1 .*\ncot boot_args 0x[0-9a-f]+ fmc 0x[0-9a-f]+ hash 12 .*\ngsp rm_args .* seq 2 classes 0xc96f 0xcec0 0xcab5 0xcfb0$"),
        Case("a later ucode entry FWSEC_PROD too: the last wins", AD107, mode="flcn", expect=r"\nflcn frts 0x1ffa00000 at 0x1200000 desc ",
             vbios=patched(roms[0], e=(lay["later_entry"], bytes([nv.FALCON_UCODE_ENTRY_APPID_FWSEC_PROD])))),
        Case("a bad BIT signature", AD107, mode="flcn", vbios=patched(roms[0], s=(0x1b2, struct.pack("<I", 0x00544943))),
             expect=r"\nerror AssertionError: Invalid BIT header signature 0x544943$"),
        Case("no FWSEC_PROD entry", AD107, mode="flcn", vbios=patched(roms[0], e=(lay["fwsec_entry"], b"\x86")),
             expect=r"\nerror UnboundLocalError: cannot access local variable 'ucode_desc_off' where it is not associated with a value$"),
        Case("two base images before the expansion ROM: the last one's length counts", AD107, mode="flcn",
             vbios=patched(roms[0], c=(lay["second_code_type"], bytes([nv.NV_BCRT_HASH_INFO_BASE_CODE_TYPE_VBIOS_BASE]))),
             expect=r"\nerror UnboundLocalError: cannot access local variable 'ucode_desc_off' where it is not associated with a value$"),
        Case("the expansion ROM before any base image", AD107, mode="flcn",
             vbios=patched(roms[0], c=(lay["first_code_type"], bytes([nv.NV_BCRT_HASH_INFO_BASE_CODE_TYPE_VBIOS_EXT]))),
             expect=r"\nerror UnboundLocalError: cannot access local variable 'block_size' where it is not associated with a value$"),
    ]
    return cs

class FakeBoot(gm.FakeTG):
    """golden_mm.py's fake TinyGPU.app as a boot sees it: every request recorded (MAP_BAR too: both clients cache bar_info),
    PCI config space (sub-word writes), RESIZE_BAR, and the case's BARs, BAR0 registers (a value, or a function of how many
    times that address was read) and VBIOS window."""
    def __init__(self, priv, case):
        super().__init__(priv)
        self.case, self.cfg, self.reads = case, dict(case.cfg), collections.Counter()
    def serve(self, conn):
        while (hdr := gm.recv_exact(conn, 33)) is not None:
            cmd, _, bar, a0, a1, a2 = struct.unpack(gm.REQ, hdr)
            self.rec += hdr
            if cmd == RemoteCmd.MMIO_WRITE:
                data = gm.recv_exact(conn, a1); self.rec += data
                if bar == 1 and a0 + a1 <= len(self.bar1): self.bar1[a0:a0 + a1] = data
            elif cmd == RemoteCmd.MAP_BAR: conn.sendall(struct.pack(gm.RESP, 0, *self.case.bars[bar]))
            elif cmd == RemoteCmd.RESIZE_BAR: conn.sendall(struct.pack(gm.RESP, 0, 0, 0))
            elif cmd == RemoteCmd.CFG_READ:
                word = self.cfg.get(a0 & ~3, 0)
                conn.sendall(struct.pack(gm.RESP, 0, (word >> (8 * (a0 & 3))) & ((1 << (8 * a1)) - 1), 0))
            elif cmd == RemoteCmd.CFG_WRITE:
                sh, mask = 8 * (a0 & 3), ((1 << (8 * a1)) - 1) << (8 * (a0 & 3))
                self.cfg[a0 & ~3] = (self.cfg.get(a0 & ~3, 0) & ~mask) | ((a2 << sh) & mask)
                conn.sendall(struct.pack(gm.RESP, 0, 0, 0))
            elif cmd == RemoteCmd.MMIO_READ and bar == 0 and (a0, a1) == VBIOS_WINDOW and self.case.vbios is not None:
                conn.sendall(struct.pack(gm.RESP, 0, a1, 0) + self.case.vbios)
            elif cmd == RemoteCmd.MMIO_READ and bar == 0 and a1 == 4:
                self.reads[a0] += 1
                v = self.case.regs.get(a0, 0)
                conn.sendall(struct.pack(gm.RESP, 0, 4, 0) + struct.pack("<I", (v(self.reads[a0]) if callable(v) else v) & 0xffffffff))
            elif cmd == RemoteCmd.MAP_SYSMEM_FD: self.sysmem(conn, a0)
            else:
                msg = b"not served"
                conn.sendall(struct.pack(gm.RESP, 1, len(msg), 0) + msg)
        conn.close()

def py_boot(path, c, data_dir):
    """tinygrad's boot as the daemon runs it (nv_dispatch_daemon.py _install_inherited_tinygpu's device: tinygrad's remote
    device on the plugin's socket, bar_info cached), as far as the case's mode goes."""
    pci = object.__new__(APLRemotePCIDevice)
    pci.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    pci.sock.connect(path)
    pci.pcibus, pci.dev_id, pci.peer_group, pci.lock_fd = "usb4", 0, "usb4", None
    out, saved_env, saved_td = [], os.environ.get("BEAGLE_TINYGPU_DATA"), nv_init_helper._TEARDOWN
    os.environ["BEAGLE_TINYGPU_DATA"], nv_init_helper._TEARDOWN = data_dir, c.teardown
    try:
        with contextlib.suppress(Exception): pci.resize_bar(1)   # PCIIfaceBase.__init__ (system.py:263)
        NVMemoryManager.va_allocator = TLSFAllocator((1 << 44), base=0x1000000000)
        dev = NVDev.__new__(NVDev)   # NVDev.__init__'s statements (nvdev.py:76-84)
        dev.pci_dev, dev.devfmt, dev.mmio = pci, pci.pcibus, pci.map_bar(0, fmt='I')
        dev.smi_dev, dev.is_booting, dev.is_err_state = False, True, False
        dev._early_ip_init()
        dev._early_mmu_init()
        out.append(f"chip {dev.chip_id:#x} {dev.chip_name} {dev.fw_name} mmu {dev.mmu_ver} fmc {int(dev.fmc_boot)} vram {dev.vram_size} "
                   f"bar1 {dev.vram.nbytes} large {int(dev.large_bar)} root {dev.mm.root_page_table.paddr}")
        if c.mode == "sw" and dev.fmc_boot:
            dev.is_booting = False
            f = dev.flcn
            f.init_sw()   # NV_FLCN_COT's
            out.append(f"cot boot_args {f.fmc_boot_args_sysmem:#x} fmc {f.fmc_booter_bar1:#x} hash {len(f.fmc_booter_hash)} {f.fmc_booter_hash[0]:#x} "
                       f"sig {len(f.fmc_booter_sig)} {f.fmc_booter_sig[0]:#x} pkey {len(f.fmc_booter_pkey)} {f.fmc_booter_pkey[0]:#x}")
        elif c.mode in ("flcn", "sw"):
            dev.is_booting = False
            f = dev.flcn
            f.init_sw()
            dd = f.desc_v3
            out.append(f"flcn frts {f.frts_offset:#x} at {f.frts_image_paddr:#x} desc {dd.IMEMPhysBase:#x} {dd.IMEMVirtBase:#x} {dd.IMEMLoadSize:#x} "
                       f"{dd.DMEMPhysBase:#x} {dd.DMEMLoadSize:#x} {dd.PKCDataOffset:#x} {dd.EngineIdMask:#x} {dd.UcodeId:#x} booter at "
                       f"{f.booter_image_paddr:#x} data {f.booter_data_off:#x}+{f.booter_data_sz:#x} code {f.booter_code_off:#x}+{f.booter_code_sz:#x}")
            if hasattr(f, "beagle_sb_image_paddr"):
                do, ds, co, cz = f.beagle_unload_params
                out.append(f"teardown sb at {f.beagle_sb_image_paddr:#x} unload at {f.beagle_unload_image_paddr:#x} data {do:#x}+{ds:#x} code {co:#x}+{cz:#x}")
            else: out.append("teardown off")
        if c.mode == "sw":
            g = dev.gsp
            g.init_sw()
            out.append(f"gsp rm_args {g.rm_args_sysmem:#x} libos {g.libos_args_sysmem:#x} wpr_meta {g.wpr_meta_sysmem:#x} radix3 {g.gsp_radix3_addrs[0]:#x} "
                       f"sig {g.gsp_signature_bar1:#x} booter {g.booter_bar1:#x} seq {g.cmd_q.seq} classes {g.gpfifo_class:#x} {g.compute_class:#x} "
                       f"{g.dma_class:#x} {g.viddec_class:#x}")
    except Exception as e: out.append(f"error {type(e).__name__}: {e}")
    finally:
        nv_init_helper._TEARDOWN = saved_td
        if saved_env is None: os.environ.pop("BEAGLE_TINYGPU_DATA", None)
        else: os.environ["BEAGLE_TINYGPU_DATA"] = saved_env
    pci.sock.close()
    return "\n".join(out)

def frames(rec):
    """The request stream as frames (a header, and an MMIO_WRITE's payload), a poll's consecutive repeated reads once."""
    out, off = [], 0
    while off < len(rec):
        cmd, _, _, _, a1, _ = struct.unpack_from(gm.REQ, rec, off)
        n = 33 + (a1 if cmd == RemoteCmd.MMIO_WRITE else 0)
        f = bytes(rec[off:off + n]); off += n
        if not (out and cmd == RemoteCmd.MMIO_READ and out[-1] == f): out.append(f)
    return out

def l0_frames(name, mode):
    """The L0 recording's requests from the BAR resize to the end of the case's step: the first BAR1 write (early: the root
    page table), the last request before the first MAP_SYSMEM_FD (flcn: NV_FLCN.init_sw's last image), or NV_GSP.init_sw's
    exit marker (sw: the second prequeued RPC's doorbell), as frames."""
    rec = RECORDINGS / name
    if not rec.exists(): return None
    out = []
    for e in tgwire.read(rec)[0]:
        if mode == "sw" and e.kind == tgwire.K_MARKER and e.f["id"] == 10 and e.f["arg"] & tgwire.MARKER_EXIT: break   # NV_GSP.init_sw exit
        if e.kind != tgwire.K_REQ: continue
        cmd, bar = e.f["req"][0], e.f["req"][2]
        if not out and cmd != RemoteCmd.RESIZE_BAR: continue
        if mode == "flcn" and cmd == RemoteCmd.MAP_SYSMEM_FD: break
        out.append(e.f["hdr"] + (e.f["payload"] if cmd == RemoteCmd.MMIO_WRITE else b""))
        if mode == "early" and cmd == RemoteCmd.MMIO_WRITE and bar == 1: break
    return frames(b"".join(out))   # the same rule as the two clients' streams

def names(fs):
    return [tgwire.cmd_name(f[0]) + (f" {tgwire.reg_name(struct.unpack_from(gm.REQ, f)[3])}" if f[0] == RemoteCmd.MMIO_READ else "") for f in fs]

def run_case(c, exe, priv, quiet=False):
    """Returns (identical, summary)."""
    mask = lambda s: re.sub(r"after \d+ ms", "after N ms", s)
    pd, cd = f"{priv}/data_py", f"{priv}/data_cpp"
    for p in (pd, cd): shutil.rmtree(p, ignore_errors=True); os.makedirs(p)
    srv, path = gm.listen(priv)
    fake = FakeBoot(priv, c)
    t = gm.serve_once(srv, fake)
    py_out = mask(py_boot(path, c, pd))
    t.join(timeout=60)
    fake2 = FakeBoot(priv, c)
    t2 = gm.serve_once(srv, fake2)
    # BEAGLE_NV_RECOVER=0: tinygrad's code has no recovery of a warm GPU (plan step P4 is the port's own, test_p4.sh's)
    env = dict(os.environ, TMPDIR=priv, APL_REMOTE_SOCK=path, BEAGLE_TINYGPU_NO_LAUNCH="1", BEAGLE_TINYGPU_LOG=f"{priv}/log", BEAGLE_TINYGPU_DATA=cd,
               BEAGLE_NV_RECOVER="0")
    if not c.teardown: env["BEAGLE_NV_TEARDOWN"] = "0"
    r = subprocess.run([exe, c.mode], capture_output=True, text=True, timeout=120, env=env)
    t2.join(timeout=60); srv.close()
    cpp_out = mask(r.stdout.strip() + (f" exit {r.returncode}: {r.stderr.strip()}" if r.returncode else ""))
    pf, cf = frames(fake.rec), frames(fake2.rec)
    saved = lambda d: sorted((p.name, hashlib.sha256(p.read_bytes()).hexdigest()) for p in pathlib.Path(d).glob("vbios/*"))
    sysmem = lambda fk: [(f.seek(0), f.read())[1] for f in fk.keep]   # every allocation's contents, in order
    ps, cs = sysmem(fake), sysmem(fake2)
    same = py_out == cpp_out and pf == cf and fake.bar1 == fake2.bar1 and saved(pd) == saved(cd) and ps == cs
    ok = same and c.expect is not None and re.search(c.expect, py_out) is not None
    summary = f"{len(pf)} requests: {py_out.splitlines()[-1][:110]}"
    if c.stream is not None:
        ok = ok and names(cf) == c.stream
        summary += f"; the stream: {', '.join(names(cf))}"
    if c.mode == "flcn" and saved(cd): summary += f"; VBIOS copy {saved(cd)[0][0]}"
    if c.l0:
        want = l0_frames(c.l0, c.mode)
        if want is None: summary += "; the L0 recording is not on this Mac (not compared)"
        elif want == cf: summary += f"; equal to the L0 recording's {len(want)}"
        else:
            ok = False
            k = next((i for i, (a, b) in enumerate(zip(want, cf)) if a != b), min(len(want), len(cf)))
            summary += f"; DIFFERS from the L0 recording at request {k} of {len(want)}/{len(cf)}: L0 {names(want[k:k + 2])}, c++ {names(cf[k:k + 2])}"
    if not same and not quiet:
        k = next((i for i, (a, b) in enumerate(zip(pf, cf)) if a != b), min(len(pf), len(cf)))
        summary += (f"\n    tinygrad: {py_out}\n    c++     : {cpp_out}\n    streams: {len(pf)} vs {len(cf)} requests, first difference at {k}: "
                    f"{names(pf[k:k + 2])} | {names(cf[k:k + 2])}; VRAM {'same' if fake.bar1 == fake2.bar1 else 'differs'}; saved {saved(pd)} | {saved(cd)}; "
                    f"sysmem allocations {len(ps)}/{len(cs)}, differing: {[i for i, (a, b) in enumerate(zip(ps, cs)) if a != b]}")
    elif (c.expect is None or not re.search(c.expect, py_out)) and not quiet: summary += f"\n    unexpected result: {py_out}"
    return ok, summary

# (header, text, replacement, the case that must catch it)
PERTURBED = [
    ("TinyGPUNVBoot.h", "const uint32_t wpr2_hi = d.rreg(0x001FA828);", "const uint32_t wpr2_hi = d.rreg(0x001FA824);", "AD107, WPR2 up"),
    ("TinyGPUNVBoot.h", "write_config_flush(0x04, cmd | 0x4, 2, err)", "write_config_flush(0x04, cmd | 0x2, 2, err)", "AD107, bus mastering off"),
    ("TinyGPUNVBoot.h", 'snprintf(impl, sizeof(impl), "%02u", d.implementation);', 'snprintf(impl, sizeof(impl), "%u", d.implementation);', "AD107 cold"),
    ("TinyGPUNVBoot.h", "if (!d.fmc_boot) return;   // NV_FLCN.wait_for_reset", "if (true) return;   // NV_FLCN.wait_for_reset", "GB205 cold"),
    ("TinyGPUNVBoot.h", "NV_PGC6_AON_SECURE_SCRATCH_GROUP_42).read() << 20;", "NV_PGC6_AON_SECURE_SCRATCH_GROUP_42).read() << 19;", "AD107 cold"),
    ("TinyGPUNVBoot.h", "if (d.bar1_size != 256ull << 20 || d.large_bar)", "if (d.large_bar)", "GB205, a 512 MiB BAR1"),
    ("TinyGPUNVBoot.h", "d.large_bar = d.bar1_size >= d.vram_size;", "d.large_bar = d.bar1_size > d.vram_size;", "AD107, a BAR1 as large as VRAM"),
    # C11b
    ("TinyGPUNVBoot.h", "if (code_type == nv::NV_BCRT_HASH_INFO_BASE_CODE_TYPE_VBIOS_BASE) block_size = imglen;",
     "if (code_type == nv::NV_BCRT_HASH_INFO_BASE_CODE_TYPE_VBIOS_BASE && block_size < 0) block_size = imglen;", "two base images before the expansion ROM"),
    ("TinyGPUNVBoot.h", "            found = true;\n", "            found = true;\n            break;\n", "a later ucode entry FWSEC_PROD too: the last wins"),
    ("TinyGPUNVBoot.h", "flcn.frts_offset = d.vram_size - 0x100000 - 0x100000;", "flcn.frts_offset = d.vram_size - 0x100000;", "AD107 cold, NV_FLCN.init_sw"),
    ("TinyGPUNVBoot.h", "const size_t sn = std::min<size_t>(u.signature.size(), 0x180);", "const size_t sn = std::min<size_t>(u.signature.size(), 0x100);", "AD107 cold, NV_FLCN.init_sw"),
    ("TinyGPUNVBoot.h", "frts_cmd.frtsRegionDesc.frtsRegionMediaType = 2;", "frts_cmd.frtsRegionDesc.frtsRegionMediaType = 1;", "AD107 cold, NV_FLCN.init_sw"),
    ("TinyGPUNVBoot.h", "const std::vector<uint8_t> rv(fc, fc + sizeof(nv::FWSECLIC_READ_VBIOS_DESC));", "const std::vector<uint8_t> rv(fc, fc + sizeof(frts_cmd));", "AD107 cold, NV_FLCN.init_sw"),
    ("TinyGPUNVBoot.h", "nv_py_setslice(u.image, patch_loc, patch_loc + sig_len, sig.data(), sig.size());", "", "AD107 cold, NV_FLCN.init_sw"),
    ("TinyGPUNVBoot.h", "if (stat(path.c_str(), &st) != 0) {", "if (stat(path.c_str(), &st) == 0) {", "AD107 cold, NV_FLCN.init_sw"),
    # C11c, C11d
    ("TinyGPUNVBoot.h", "const uint64_t pte_cnt = queue_pte_cnt + tg_round_up(queue_pte_cnt * 8, 0x1000) / 0x1000;", "const uint64_t pte_cnt = queue_pte_cnt;", "AD107 cold, the whole init_sw"),
    ("TinyGPUNVBoot.h", "h.writePtr = 0; h.flags = 1; h.rxHdrOff = sizeof(nv::msgqTxHeader);", "h.writePtr = 0; h.flags = 1; h.rxHdrOff = 0;", "AD107 cold, the whole init_sw"),
    ("TinyGPUNVBoot.h", "r.kind = nv::LIBOS_MEMORY_REGION_CONTIGUOUS; r.loc = nv::LIBOS_MEMORY_REGION_LOC_SYSMEM; r.size = 0x10000;",
     "r.kind = nv::LIBOS_MEMORY_REGION_CONTIGUOUS; r.loc = nv::LIBOS_MEMORY_REGION_LOC_SYSMEM; r.size = 0x8000;", "AD107 cold, the whole init_sw"),
    ("TinyGPUNVBoot.h", 'rm.id8 = nv_id8("RMARGS");', 'rm.id8 = nv_id8("RMARG");', "AD107 cold, the whole init_sw"),
    ("TinyGPUNVBoot.h", "npages[i - 1] = ((npages[i] - 1) >> (nv::LIBOS_MEMORY_REGION_RADIX_PAGE_LOG2 - 3)) + 1;",
     "npages[i - 1] = ((npages[i] - 1) >> (nv::LIBOS_MEMORY_REGION_RADIX_PAGE_LOG2 - 4)) + 1;", "AD107 cold, the whole init_sw"),
    ("TinyGPUNVBoot.h", "gsp_heap_sz = 0x8100000;", "gsp_heap_sz = 0x8000000;", "AD107 cold, the whole init_sw"),
    ("TinyGPUNVBoot.h", "m.pmuReservedSize = 0x1820000;", "m.pmuReservedSize = 0x1800000;", "GB205 cold, the whole init_sw"),
    ("TinyGPUNVBoot.h", "data.pciConfigMirrorBase = d.fmc_boot ? 0x92000 : 0x88000;", "data.pciConfigMirrorBase = d.fmc_boot ? 0x88000 : 0x92000;", "GB205 cold, the whole init_sw"),
    ("TinyGPUNVBoot.h", "e.type = nv::REGISTRY_TABLE_ENTRY_TYPE_DWORD;", "e.type = 2;", "AD107 cold, the whole init_sw"),
    ("TinyGPUNVBoot.h", "pkey.insert(pkey.end(), 3, 0);", "pkey.insert(pkey.end(), 7, 0);", "GB205 cold, the whole init_sw"),
    ("TinyGPUNVBoot.h", "for (int i = 0; i < 5; ++i) {\n        nv::LibosMemoryRegionInitArgument r{};", "for (int i = 0; i < 4; ++i) {\n        nv::LibosMemoryRegionInitArgument r{};", "AD107 cold, the whole init_sw"),
    ("TinyGPUNVGsp.h", "if (!gsp_) { doorbell(); return; }", "if (!gsp_) { return; }", "AD107 cold, the whole init_sw"),
]

def main():
    exe = f"{WORK}/golden_boot"
    tgpaths.build_cpp(f"{HERE}/golden_boot.cpp", exe)
    priv = tempfile.mkdtemp(dir="/tmp", prefix="tgboot.")
    tempfile.tempdir = priv
    all_cases, fails = cases(), 0
    for c in all_cases:
        ok, summary = run_case(c, exe, priv)
        fails += not ok
        print(f"{'IDENTICAL' if ok else 'MISMATCH '} {c.name}: {summary}")
    caught, gpu = 0, tgpaths.REPO / "libhmsbeagle" / "GPU"
    for hdr, old, new, case in PERTURBED:
        inc = f"{priv}/perturbed"; os.makedirs(f"{inc}/libhmsbeagle/GPU", exist_ok=True)
        text = (gpu / hdr).read_text()
        assert text.count(old) == 1, (hdr, old)
        open(f"{inc}/libhmsbeagle/GPU/{hdr}", "w").write(text.replace(old, new))
        pexe = f"{priv}/golden_boot_perturbed"
        tgpaths.build_cpp(f"{HERE}/golden_boot.cpp", pexe, "-iquote", inc)   # found before the repository's
        hit = not run_case(next(c for c in all_cases if c.name.startswith(case)), pexe, priv, quiet=True)[0]
        caught += hit
        print(f"perturbed {hdr} ({old.strip()[:60]} -> {new.strip()[:40]}): {'REJECTED' if hit else 'NOT CAUGHT'} by '{case}'")
    fails += len(PERTURBED) - caught
    print(f"C11 boot vs tinygrad: {'all identical' if not fails else f'{fails} FAILED'}")
    sys.exit(1 if fails else 0)

if __name__ == "__main__":
    main()
