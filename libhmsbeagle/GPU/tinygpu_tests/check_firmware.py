"""Firmware staging check (TODO.md plan step S0): the NVIDIA firmware BEAGLE needs beyond what tinygrad's boot already
downloads is in tinygrad's download cache, so tinygrad's own fetch_fw returns it with the network off. Re-stages a
missing or corrupt cache entry from $BEAGLE_TINYGPU_DATA/fw (macOS may purge ~/Library/Caches). Exit 0 on success."""
import os, sys, hashlib, pathlib, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tgpaths
tgpaths.setup()
from tinygrad import helpers

LINUX_FIRMWARE = "https://gitlab.com/kernel-firmware/linux-firmware/-/raw/0a6871b19abf5d6e024b5d208b101ae53e7fa0de"  # helpers.fetch_fw's pin
FIRMWARE = [  # (subdir, name, sha256): booter_unload for NVIDIA's driver-unload teardown (plan step P2)
    ("nvidia/ad102/gsp", "booter_unload-570.144.bin", "975b85a14ded8e430d30f000c3c1afdd55c15dee04f35ff9dfd876acd7e67186"),
    ("nvidia/ga102/gsp", "booter_unload-570.144.bin", "8e63db5b78d7d3e349f20a2d11099c3d7109081393cb09ffc0a28133324ae009"),
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
sys.exit(0 if ok else 1)
