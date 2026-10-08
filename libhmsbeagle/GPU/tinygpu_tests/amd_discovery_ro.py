"""TODO.md plan step N3: a new AMD card's IP discovery table, read with only the requests tinygrad's own boot (the pin) makes
before it needs one: no firmware, PSP, SMU or boot. amd_discovery.py, A0's capture, opens Device["AMD"] and so boots the card.
    amd_discovery_ro.py <out dir>
It runs the pin's own code, tinygrad's RemotePCIDevice on TinyGPU.app's socket (APL_REMOTE_SOCK, default $TMPDIR/tinygpu.sock:
the server must be running, tg_probe.py starts it) and AMDev.__init__ (amdev.py:158-170) with _build_regs replaced by a stop,
so the requests are PCIIfaceBase.__init__'s RESIZE_BAR of BAR0 (system.py:263, an error suppressed) and AMDev's: _disable_aspm's
config walk and LNKCTL write (amdev.py:149-156), MAP_BAR of BARs 0, 2 and 5, the IOV identifier, then _run_discovery
(amdev.py:351-391): MEMSIZE, and the table's 2,560 dwords through the indirect window (MM_INDEX_HI and MM_INDEX writes, an
MM_DATA read each, amdev.py:341-348). It stops (exit 1) on a virtual function, before the pin's mailbox request; and right
after MEMSIZE, before any index write, on a MEMSIZE of 0 or 0xffffffff, a VRAM under 64 KiB, or a BAR0 other than 256 MiB
smaller than VRAM (TODO.md plan step N1's rule; the pin would read a large BAR0 directly). After the discovery: the config
space's identity (dwords 0x00, 0x08, 0x2c: three reads). Under nv_usb4.lock (here) and am_usb4.lock (tinygrad's); the
network is blocked, and nothing parses beyond the discovery table.
Writes <out>/<vendor>_<device>_<sha16>.bin (the table's binary_size bytes) and .json (as amd_discovery.py's, with the BAR
sizes, revision, class and subsystem; an existing .json of the same table is kept), and the whole 10 KiB read to <out>/raw/<vendor>_<device>_<stamp>.bin, or to
<out>/raw/<stamp>.bin when the parse fails (exit 1: the bytes say why). Exit 0: captured; 1: a stop above; 2: nothing was sent
(a lock held, no socket)."""
import fcntl, hashlib, json, os, pathlib, socket, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tgpaths
if tgpaths.TINYGRAD_PATH not in sys.path: sys.path.insert(0, tgpaths.TINYGRAD_PATH)
tgpaths.block_network()   # not block_real_devices: this is for the card
from tinygrad.runtime.support.system import RemotePCIDevice
from tinygrad.runtime.support.am.amdev import AMDev
from tinygrad.runtime.autogen.am import am

MEMSIZE = 0xde3   # mmRCC_CONFIG_MEMSIZE (amdev.py:353)
out = pathlib.Path(sys.argv[1])
tmp = os.environ.get("TMPDIR", "/tmp").rstrip("/")
stamp = time.strftime("%Y%m%d-%H%M%S")

def done(msg, code):
    print(msg)
    sys.exit(code)

class Stop(Exception): pass
class Captured(Exception): pass

class Capture(AMDev):
    """AMDev.__init__ itself, to _build_regs: the checks, and the raw read kept before it is parsed."""
    def _vf_mailbox_request(self, *a, **k): raise Stop("a virtual function (BEAGLE boots the PF only)")
    def rreg(self, reg, inst=0, direct=False):
        v = super().rreg(reg, inst, direct)
        if reg == MEMSIZE:   # _run_discovery's first read: nothing indirect has been written yet
            bar0 = self.vram.nbytes
            if v in (0, 0xffffffff) or (v << 20) < (64 << 10): raise Stop(f"MEMSIZE reads {v:#x} (TODO.md plan step N1's stop rule S3)")
            if bar0 != (256 << 20) or bar0 >= (v << 20):
                raise Stop(f"BAR0 is {bar0 >> 20} MiB with {v} MiB of VRAM: BEAGLE supports only a 256 MiB BAR0 smaller than VRAM")
        return v
    def _read_vram(self, addr, size):
        self.raw = super()._read_vram(addr, size)
        return self.raw
    def _build_regs(self): raise Captured()

lock = os.open(f"{tmp}/nv_usb4.lock", os.O_RDWR | os.O_CREAT, 0o666)
try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError: done("nv_usb4.lock is held: another process has the eGPU; nothing sent", 2)
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
try: sock.connect(os.environ.get("APL_REMOTE_SOCK", f"{tmp}/tinygpu.sock"))   # never starts TinyGPU.app's server
except OSError as e: done(f"TinyGPU.app's socket: {e}; nothing sent", 2)
try: dev = RemotePCIDevice("AM", "usb4", sock)   # its am_usb4.lock, as APLRemotePCIDevice's (system.py:387-393)
except RuntimeError as e: done(f"{e}; nothing sent", 2)
try: dev.resize_bar(0)   # PCIIfaceBase.__init__: contextlib.suppress(Exception)
except Exception: pass
adev = object.__new__(Capture)
try: Capture.__init__(adev, dev)
except Captured: pass
except Stop as e: done(f"STOP: {e}; nothing indirect was written", 1)
except AssertionError as e:   # _run_discovery's signature check (amdev.py:361)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    raw = out / "raw" / f"{stamp}.bin"
    raw.write_bytes(getattr(adev, "raw", b""))
    done(f"STOP: {e}; the 10 KiB read is in {raw}", 1)
ident = {off: dev.read_config(off, 4) for off in (0x00, 0x08, 0x2c)}
sock.close()

vendor, device = ident[0x00] & 0xffff, ident[0x00] >> 16
os.umask(0o022)   # tinygrad's flock_acquire set it to 0 (system.py:145)
(out / "raw").mkdir(parents=True, exist_ok=True)
(out / "raw" / f"{vendor:04x}_{device:04x}_{stamp}.bin").write_bytes(adev.raw)
bh = am.struct_binary_header.from_buffer_copy(adev.raw)
table = adev.raw[:bh.binary_size]
sha = hashlib.sha256(table).hexdigest()
stem = out / f"{vendor:04x}_{device:04x}_{sha[:16]}"
stem.with_suffix(".bin").write_bytes(table)
hwip = {v: k for k, v in vars(am).items() if k.endswith("_HWIP") and isinstance(v, int)}
gc = adev.ip_ver.get(am.GC_HWIP)
meta = {"pci_id": f"{vendor:04x}:{device:04x}", "arch": "gfx%d%x%x" % tuple(gc) if gc else None, "bytes": len(table), "sha256": sha,
        "read": "the pin's AMDev to _run_discovery only (amd_discovery_ro.py, TODO.md plan step N3): 10 KiB at VRAM end - 64 KiB, "
                "through the indirect window; kept: the table's binary_size bytes",
        "tinygrad": "a9830e2b4", "captured": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "partial_boot": None,
        "vram_size": adev.vram_size, "large_bar": adev.large_bar, "vram_bar_bytes": adev.vram.nbytes,
        "doorbell_bar_bytes": dev.bar_info(2)[1], "mmio_bar_bytes": dev.bar_info(5)[1],
        "revision": ident[0x08] & 0xff, "class": ident[0x08] >> 8, "subsystem": f"{ident[0x2c] & 0xffff:04x}:{ident[0x2c] >> 16:04x}",
        "ip_ver": {hwip.get(k, str(k)): list(v) for k, v in sorted(adev.ip_ver.items())},
        "regs_offset": {hwip.get(k, str(k)): {str(i): list(b) for i, b in sorted(v.items())} for k, v in sorted(adev.regs_offset.items())},
        "harvested": {hwip.get(k, str(k)): sorted(v) for k, v in adev.harvested.items() if v},
        "gc_info": type(adev.gc_info).__name__, "reserved_vram_size": adev.reserved_vram_size}
if stem.with_suffix(".json").exists(): print(f"kept the existing {stem}.json: the same table, captured before")
else: stem.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
print(f"captured: {stem}.bin ({len(table)} bytes, sha256 {sha})")
print("ip_ver:", {k: tuple(v) for k, v in meta["ip_ver"].items()})
print(f"pci {meta['pci_id']} rev {meta['revision']:02x} class {meta['class']:06x} subsystem {meta['subsystem']}; vram_size {adev.vram_size >> 20} MiB, "
      f"BARs 0/2/5 {adev.vram.nbytes >> 20} MiB / {meta['doorbell_bar_bytes'] >> 20} MiB / {meta['mmio_bar_bytes'] >> 20} MiB, gc_info {meta['gc_info']}, "
      f"harvested {meta['harvested']}")
