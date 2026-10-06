"""Firmware staging check (TODO.md plan step S0): the NVIDIA firmware BEAGLE needs beyond what tinygrad's boot already
downloads, and the Blackwell boot's own firmware (plan step B1: staged so no boot downloads inside the daemon, decision 5),
is in tinygrad's download cache, so tinygrad's own fetch_fw returns it with the network off. Re-stages a
missing or corrupt cache entry from $BEAGLE_TINYGPU_DATA/fw (macOS may purge ~/Library/Caches). Then every file of
TinyGPUFirmwareManifest.h and TinyGPUAMDBootTables.h's fw table is in BEAGLE's cache, where the C++ boots look for it
(TinyGPUFirmware.h, which does not search tinygrad's cache since 2026-10-05): a missing or corrupt one is copied from
tinygrad's cache through fetch_fw, network off. Exit 0 on success."""
import os, re, sys, hashlib, pathlib, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tgpaths
tgpaths.setup()
from tinygrad import helpers

LINUX_FIRMWARE = "https://gitlab.com/kernel-firmware/linux-firmware/-/raw/0a6871b19abf5d6e024b5d208b101ae53e7fa0de"  # helpers.fetch_fw's pin
FIRMWARE = [  # (subdir, name, sha256): booter_unload for NVIDIA's driver-unload teardown (plan step P2)
    ("nvidia/ad102/gsp", "booter_unload-570.144.bin", "975b85a14ded8e430d30f000c3c1afdd55c15dee04f35ff9dfd876acd7e67186"),
    ("nvidia/ga102/gsp", "booter_unload-570.144.bin", "8e63db5b78d7d3e349f20a2d11099c3d7109081393cb09ffc0a28133324ae009"),
    # the GB20x (COT) boot's three files (ip.py:303-304, 402, 426-429), and ad102's booter_load for test_p2_teardown.py
    ("nvidia/gb202/gsp", "fmc-570.144.bin", "cb59a35c1d4bd1274d7267fd10243c29f843ff41c851b9cbd59f5af2ddd7fece"),
    ("nvidia/ga102/gsp", "gsp-570.144.bin", "a8c3ebeed280323aedb51c061f321e73379cce7a9ae643a33dd03915df027f7f"),
    ("nvidia/gb202/gsp", "bootloader-570.144.bin", "d40b48e431d1707dc77af3605db358ed7a32ebfc2830eb74de2eddb4d3025071"),
    ("nvidia/ad102/gsp", "booter_load-570.144.bin", "8b293e19b637c5e22c87a2428d1c71bb13e0904e8a88ac6b3c6c1f2679c6e37a"),
]

def sha(b): return hashlib.sha256(b).hexdigest()

ok = True
for subdir, name, digest in FIRMWARE:
    cached = helpers._ensure_downloads_dir() / "fw" / hashlib.md5(f"{LINUX_FIRMWARE}/{subdir}/{name}".encode()).hexdigest()
    if not cached.is_file() or sha(cached.read_bytes()) != digest:
        backup = tgpaths.DATA / "fw" / f"{subdir.split('/')[1]}-{name}"
        if backup.is_file() and sha(backup.read_bytes()) == digest:
            cached.parent.mkdir(parents=True, exist_ok=True); cached.write_bytes(backup.read_bytes())
            print(f"re-staged {subdir}/{name} from {backup}")
        else:
            print(f"MISSING {subdir}/{name}: neither {cached} nor {backup} has sha256 {digest}"); ok = False; continue

def offline(*a, **k): raise RuntimeError("network access attempted")
urllib.request.urlopen = offline   # fetch() imports urllib.request at call time, so this blocks any download
for subdir, name, digest in FIRMWARE:
    try:
        b = helpers.fetch_fw(subdir, name, digest)
        print(f"fetch_fw {subdir}/{name}: {len(b)} bytes, sha256 {'OK' if sha(b) == digest else 'WRONG'} (network off)")
        ok &= sha(b) == digest
    except Exception as e:
        print(f"fetch_fw {subdir}/{name}: FAILED ({e})"); ok = False

row = re.compile(r'^    \{"[^"]+", "[^"]+", "([^"]+)", "([^"]+)", "([0-9a-f]{64})", "[0-9a-f]{32}"\},')
text = (tgpaths.GPU_DIR / "TinyGPUFirmwareManifest.h").read_text() \
    + (tgpaths.GPU_DIR / "TinyGPUAMDBootTables.h").read_text().split("namespace fw {")[1].split("} // namespace fw")[0]
beagle_cache = pathlib.Path(os.environ.get("XDG_CACHE_HOME", pathlib.Path.home() / "Library/Caches")) / "beagle" / "firmware"
files = sorted({m.groups() for m in map(row.match, text.splitlines()) if m})   # gsp-570.144.bin serves every NVIDIA chip
for subdir, name, digest in files:
    dest = beagle_cache / subdir / name
    if dest.is_file() and sha(dest.read_bytes()) == digest: continue
    try: b = helpers.fetch_fw(subdir, name, digest)
    except Exception as e:
        print(f"MISSING {subdir}/{name}: not in BEAGLE's cache {dest}, and fetch_fw failed ({e})"); ok = False; continue
    if sha(b) != digest: print(f"MISSING {subdir}/{name}: fetch_fw returned sha256 {sha(b)}, not {digest}"); ok = False; continue
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f"{name}.part.{os.getpid()}")
    tmp.write_bytes(b); tmp.replace(dest)
    print(f"staged {subdir}/{name} into {dest} (from tinygrad's cache)")
print(f"BEAGLE's cache {beagle_cache}: {len(files)} files checked")
sys.exit(0 if ok else 1)
