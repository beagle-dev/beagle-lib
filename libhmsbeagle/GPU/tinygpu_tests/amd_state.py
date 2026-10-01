"""Read-only: which boot tinygrad's AM driver (the pin) would give the AMD card now, from the registers its decision reads
(amdev.py:184-197). SCRATCH_REG7 == Version, SCRATCH_REG6 == 0 and no GCVM protection fault give a partial boot. Otherwise it
is a full boot, with an SMU mode1 reset first if the PSP's SOS is alive (C2PMSG_81 != 0) and the SMU is too; plan step A0
aborts on one, since it was never tried over TinyGPU.
    amd_state.py <the card's discovery .json, from amd_discovery.py>
Exits 0 for a partial boot or a full one without the reset, 3 when the reset would follow, and 1 if it cannot tell.
Under nv_usb4.lock and am_usb4.lock: one CFG_READ, MAP_BAR of BAR5, then 32-bit MMIO reads inside BAR5 only (never the
indirect window, which writes). The register addresses come from the captured table's bases (STATUS.md R64) and tinygrad's
own register tables. Nothing is written to the GPU, and TinyGPU.app's server is never started from here."""
import fcntl, functools, json, os, socket, struct, sys
sys.path.insert(0, os.environ["TINYGRAD_PATH"])
from tinygrad.runtime.support.amd import AMDReg, import_asic_regs
meta = json.load(open(sys.argv[1]))
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
if f"{cfg0 & 0xffff:04x}:{cfg0 >> 16:04x}" != meta["pci_id"]:
    sys.exit(f"TinyGPU.app serves {cfg0 & 0xffff:04x}:{cfg0 >> 16:04x}, not the table's {meta['pci_id']}; not reading")
bar5 = rpc(MAP_BAR, bar=5)[1]
def bases(ip): return {int(i): tuple(b) for i, b in meta["regs_offset"][ip].items()}
mp0, mp1, gc = bases("MP0_HWIP"), bases("MP1_HWIP"), bases("GC_HWIP")
want = [("regMP0_SMN_C2PMSG_81", "mp", (13, 0, 0), mp0), ("regMP0_SMN_C2PMSG_35", "mp", (13, 0, 0), mp0),
        ("mmMP1_SMN_C2PMSG_90", "mp", (11, 0, 0), mp1),
        ("regSCRATCH_REG7", "gc", (11, 0, 0), gc), ("regSCRATCH_REG6", "gc", (11, 0, 0), gc),
        ("regGCVM_L2_PROTECTION_FAULT_STATUS", "gc", (11, 0, 0), gc)]
vals = {}
for name, prefix, ver, b in want:
    addr = import_asic_regs(prefix, ver, cls=functools.partial(AMDReg, bases=b))[name].addr[0]   # AMDReg: bases[segment] + offset
    if addr * 4 + 4 > bar5: sys.exit(f"{name} at dword {addr:#x} is outside BAR5 ({bar5:#x} bytes); not reading it")
    vals[name] = struct.unpack("<I", rpc(MMIO_READ, bar=5, a0=addr * 4, a1=4, readout=4)[2])[0]
    print(f"{name:38s} dword {addr:#08x} = {vals[name]:#010x}")
s.close()
VERSION = 0xA0000008
partial = vals["regSCRATCH_REG7"] == VERSION and vals["regSCRATCH_REG6"] == 0 and vals["regGCVM_L2_PROTECTION_FAULT_STATUS"] == 0
sos = vals["regMP0_SMN_C2PMSG_81"] != 0
if partial: print("prediction: a partial boot (AM booted it before and finalized it cleanly)")
elif not sos: print("prediction: a full boot without a mode1 reset (the PSP's SOS is not alive)")
else:
    print(f"prediction: a full boot WITH a mode1 reset if the SMU answers (SOS alive; SMU's C2PMSG_90 {vals['mmMP1_SMN_C2PMSG_90']:#x})")
    sys.exit(3)
