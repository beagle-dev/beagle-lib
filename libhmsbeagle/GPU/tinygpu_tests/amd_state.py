"""Read-only: which boot tinygrad's AM driver (the pin) would give the AMD card now, from the registers its decision reads
(amdev.py:184-197). SCRATCH_REG7 == Version, SCRATCH_REG6 == 0 and no GCVM protection fault give a partial boot. Otherwise it
is a full boot, with an SMU mode1 reset first if the PSP's SOS is alive (C2PMSG_81 != 0) and the SMU is too; plan step A0
aborts on one, since it was never tried over TinyGPU.
    amd_state.py <a discovery .json, from amd_discovery.py or amd_discovery_ro.py> [more .json of the same device ID ...]
Exits 0 for a partial boot or a full one without the reset, 3 when the reset would follow, and 1 if it cannot tell.
Under nv_usb4.lock and am_usb4.lock: CFG_READs of the identity (dwords 0x00, and 0x08 and 0x2c when a table records the
revision and subsystem), MAP_BAR of BAR5, then 32-bit MMIO reads inside BAR5 only (never the indirect window, which writes).
The card must be one of the tables: its device ID, revision and subsystem (where recorded) and its VRAM size (MEMSIZE) are
the table's, so two boards of one die never share a prediction (TODO.md plan step N6). The register addresses come from
that table's bases (STATUS.md R64) and tinygrad's own register tables for its IP versions, named as AMDev names them: the
PSP's MPASP registers from MP0 14 (ip.py:590), GCVM_L2_PROTECTION_FAULT_STATUS_LO32 from GC 12 (ip.py:87). An IP set with
no tinygrad tables cannot be told. Nothing is written to the GPU, and TinyGPU.app's server is never started from here."""
import fcntl, functools, json, os, socket, struct, sys
sys.path.insert(0, os.environ["TINYGRAD_PATH"])
from tinygrad.runtime.support.amd import AMDReg, import_asic_regs
tables = [json.load(open(p)) for p in sys.argv[1:]]
tmp = os.environ.get("TMPDIR", "/tmp").rstrip("/")
REQ, RESP = struct.Struct("<BIIQQQ"), struct.Struct("<BQQ")
MAP_BAR, CFG_READ, MMIO_READ = 1, 3, 6
locks = []
for name in ("nv_usb4.lock", "am_usb4.lock"):
    fd = os.open(f"{tmp}/{name}", os.O_RDWR | os.O_CREAT, 0o666)
    try: fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: sys.exit(f"{name} is held: another process has the eGPU; not reading")
    locks.append(fd)
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.connect(f"{tmp}/tinygpu.sock")   # TinyGPU.app's server is running: tg_probe.py started it if needed
def recv(n):
    b = b""
    while len(b) < n:
        c = s.recv(n - len(b))
        if not c: sys.exit("connection closed")
        b += c
    return b
def rpc(cmd, bar=0, a0=0, a1=0, readout=0):
    s.sendall(REQ.pack(cmd, 0, bar, a0, a1, 0))
    st, r0, r1 = RESP.unpack(recv(RESP.size))
    if st != 0: sys.exit(f"RPC {cmd} failed: {recv(r0).decode(errors='replace') if 0 < r0 < 65536 else '?'}")
    return r0, r1, (recv(readout) if readout else None)
cfg0 = rpc(CFG_READ, 0, 0, 4)[0]
pci_id = f"{cfg0 & 0xffff:04x}:{cfg0 >> 16:04x}"
tables = [t for t in tables if t["pci_id"] == pci_id]
if not tables: sys.exit(f"TinyGPU.app serves {pci_id}, which no table given is for; not reading")
if any("revision" in t or "subsystem" in t for t in tables):
    cfg8, cfg2c = rpc(CFG_READ, 0, 8, 4)[0], rpc(CFG_READ, 0, 0x2c, 4)[0]
    rev, sub = cfg8 & 0xff, f"{cfg2c & 0xffff:04x}:{cfg2c >> 16:04x}"
    tables = [t for t in tables if t.get("revision", rev) == rev and t.get("subsystem", sub) == sub]
    if not tables: sys.exit(f"no table given is for this board ({pci_id} revision {rev:02x}, subsystem {sub}); not reading")
if len(tables) > 1: sys.exit(f"{len(tables)} tables match this card; pass one; not reading")
meta = tables[0]
bar5 = rpc(MAP_BAR, bar=5)[1]
memsize = struct.unpack("<I", rpc(MMIO_READ, bar=5, a0=0xde3 * 4, a1=4, readout=4)[2])[0]   # mmRCC_CONFIG_MEMSIZE (amdev.py:353)
if memsize << 20 != meta["vram_size"]:
    sys.exit(f"the card has {memsize} MiB of VRAM, the table {meta['vram_size'] >> 20} MiB: another board; not reading")
def bases(ip): return {int(i): tuple(b) for i, b in meta["regs_offset"][ip].items()}
mp0, mp1, gc = bases("MP0_HWIP"), bases("MP1_HWIP"), bases("GC_HWIP")
ipv = {k: tuple(v) for k, v in meta["ip_ver"].items()}
psp = "regMP0_SMN_C2PMSG" if ipv["MP0_HWIP"] < (14, 0, 0) else "regMPASP_SMN_C2PMSG"   # AM_PSP.reg_pref (ip.py:590)
fault = "regGCVM_L2_PROTECTION_FAULT_STATUS" + ("_LO32" if ipv["GC_HWIP"] >= (12, 0, 0) else "")   # AM_GMC.pf_status_reg (ip.py:87)
want = [(f"{psp}_81", "mp", ipv["MP0_HWIP"], mp0), (f"{psp}_35", "mp", ipv["MP0_HWIP"], mp0),
        ("mmMP1_SMN_C2PMSG_90", "mp", (11, 0, 0), mp1),
        ("regSCRATCH_REG7", "gc", ipv["GC_HWIP"], gc), ("regSCRATCH_REG6", "gc", ipv["GC_HWIP"], gc), (fault, "gc", ipv["GC_HWIP"], gc)]
vals = {}
for name, prefix, ver, b in want:
    try: addr = import_asic_regs(prefix, ver, cls=functools.partial(AMDReg, bases=b))[name].addr[0]   # AMDReg: bases[segment] + offset
    except (ImportError, KeyError) as e: sys.exit(f"{name}: tinygrad has no register table for {prefix} {'.'.join(map(str, ver))} ({e}); cannot tell")
    if addr * 4 + 4 > bar5: sys.exit(f"{name} at dword {addr:#x} is outside BAR5 ({bar5:#x} bytes); not reading it")
    vals[name] = struct.unpack("<I", rpc(MMIO_READ, bar=5, a0=addr * 4, a1=4, readout=4)[2])[0]
    print(f"{name:38s} dword {addr:#08x} = {vals[name]:#010x}")
s.close()
VERSION = 0xA0000008
partial = vals["regSCRATCH_REG7"] == VERSION and vals["regSCRATCH_REG6"] == 0 and vals[fault] == 0
sos = vals[f"{psp}_81"] != 0
if partial: print("prediction: a partial boot (AM booted it before and finalized it cleanly)")
elif not sos: print("prediction: a full boot without a mode1 reset (the PSP's SOS is not alive)")
else:
    print(f"prediction: a full boot WITH a mode1 reset if the SMU answers (SOS alive; SMU's C2PMSG_90 {vals['mmMP1_SMN_C2PMSG_90']:#x})")
    sys.exit(3)
