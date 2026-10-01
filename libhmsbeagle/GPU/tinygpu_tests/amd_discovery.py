"""TODO.md plan step A0: the AMD card's IP discovery table, for A2's AM mock and amd_state.py's register addresses.
    amd_discovery.py <out dir> <vendor:device, from tg_probe.py>
Stock tinygrad (the pin) opens the device as run_amd_smoke.sh does: a partial boot when SCRATCH_REG7 says AM booted it.
The bytes are the ones AMDev._run_discovery already read and parsed (amdev.py:350-391: 10 KiB at VRAM end - 64 KiB, through
the indirect VRAM read on a small BAR); nothing more is read from the GPU. Only the table's binary_size bytes are kept: the
rest of that read is other VRAM, which a power cycle changes. Writes <out dir>/<vendor>_<device>_<sha16>.bin and a .json
beside it: the IP versions and bases, harvest, the VRAM size and gc_info."""
import ctypes, hashlib, json, os, pathlib, sys, time
from tinygrad import Device
from tinygrad.runtime.autogen.am import am
out, pci_id = pathlib.Path(sys.argv[1]), sys.argv[2]
d = Device["AMD"]
adev = d.iface.dev_impl
raw = ctypes.string_at(ctypes.addressof(adev.bhdr), 10 << 10)
# the copy must parse as the driver's did
bh = am.struct_binary_header.from_buffer_copy(raw)
ih = am.struct_ip_discovery_header.from_buffer_copy(raw, bh.table_list[am.IP_DISCOVERY].offset)
assert bh.binary_signature == am.BINARY_SIGNATURE and ih.signature == am.DISCOVERY_TABLE_SIGNATURE, "the copy's signatures"
hwip = {v: k for k, v in vars(am).items() if k.endswith("_HWIP") and isinstance(v, int)}
table = raw[:bh.binary_size]
sha = hashlib.sha256(table).hexdigest()
os.umask(0o022)   # tinygrad's flock_acquire set it to 0 (system.py:145)
out.mkdir(parents=True, exist_ok=True)
stem = out / f"{pci_id.replace(':', '_')}_{sha[:16]}"
stem.with_suffix(".bin").write_bytes(table)
meta = {"pci_id": pci_id, "arch": d.arch, "bytes": len(table), "sha256": sha,
        "read": "AMDev._run_discovery: 10 KiB at VRAM end - 64 KiB (amdev.py:350-391); kept: the table's binary_size bytes",
        "tinygrad": "a9830e2b4",
        "captured": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "partial_boot": adev.partial_boot,
        "vram_size": adev.vram_size, "large_bar": adev.large_bar, "vram_bar_bytes": adev.vram.nbytes,
        "ip_ver": {hwip.get(k, str(k)): list(v) for k, v in sorted(adev.ip_ver.items())},
        "regs_offset": {hwip.get(k, str(k)): {str(i): list(b) for i, b in sorted(v.items())} for k, v in sorted(adev.regs_offset.items())},
        "harvested": {hwip.get(k, str(k)): sorted(v) for k, v in adev.harvested.items() if v},
        "gc_info": type(adev.gc_info).__name__, "reserved_vram_size": adev.reserved_vram_size}
stem.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
print(f"captured: {stem}.bin ({len(table)} bytes, sha256 {sha})")
print("ip_ver:", {k: tuple(v) for k, v in meta["ip_ver"].items()})
print(f"vram_size: {adev.vram_size >> 20} MiB, BAR0 {adev.vram.nbytes >> 20} MiB, gc_info {meta['gc_info']}, harvested {meta['harvested']}, partial_boot {adev.partial_boot}")
