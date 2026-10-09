#!/bin/bash
# TODO.md plan step N3, offline: amd_discovery_ro.py, the discovery capture that sends only the pin's own pre-boot requests, on
# fake_amd_device.py's RX 7900 XT (each session recorded, FAKE_AMD_RECORD):
#   - it captures the card's table: the .bin equals the stored capture (1002_744c_56dd0ec116e3d382.bin, R65) byte for byte, and
#     the .json's IP versions, bases, harvest, VRAM and gc_info equal the stored ones;
#   - its requests, but for the three identity reads that follow, equal byte for byte the start of the pin's own session on the
#     same card (tinygrad's RemotePCIDevice, RESIZE_BAR, then an unmodified AMDev, which goes on to boot);
#   - it writes nothing but the LNKCTL config write and the indirect window's index registers (BAR5 dwords 0x00 and 0x06);
#   - a MEMSIZE of 0 or 0xffffffff (FAKE_AMD_MEMSIZE) and a 512 MiB BAR0 (FAKE_AMD_BAR0_MB) stop it (exit 1) with no MMIO
#     write at all;
#   - tinygrad's download cache is unchanged.
# One PASS or FAIL line per check; exit 0 only if all pass.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
require_no_launch_guard
W="$TINYGPU_TEST_WORK/n3"; rm -rf "$W"; mkdir -p "$W"
fails=0
pass() { echo "PASS $1"; }
fail() { echo "FAIL $1"; fails=$((fails + 1)); }
check() { if eval "$2"; then pass "$1"; else fail "$1"; fi; }
STORED=$(ls "$BEAGLE_TINYGPU_DATA"/discovery/1002_744c_*.json | head -1)
DLCACHE="${XDG_CACHE_HOME:-$HOME/Library/Caches}/tinygrad/downloads"
ls -lR "$DLCACHE" > "$W/dlcache_before.txt" 2>&1

session() {   # <label> <client: capture|reference> [VAR=value ...]: one client session on a fresh fake card, recorded
    local l=$1 client=$2; shift 2
    local d; d=$(mktemp -d /tmp/tgn3.XXXXXX)
    env "$@" FAKE_AMD_RECORD="$W/$l.rec" "$BEAGLE_PYTHON" "$TG_TESTS/fake_amd_device.py" "$d/dev.sock" "$W/mem_$l" > "$W/$l.dev" 2>&1 &
    local srv=$!
    for i in $(seq 100); do grep -q listening "$W/$l.dev" 2>/dev/null && break; sleep 0.1; done
    if [ "$client" = capture ]; then
        env "$@" APL_REMOTE_SOCK="$d/dev.sock" TMPDIR="$d" "$BEAGLE_PYTHON" "$TG_TESTS/amd_discovery_ro.py" "$W/disc_$l" > "$W/$l.txt" 2>&1
    else
        env "$@" APL_REMOTE_SOCK="$d/dev.sock" TMPDIR="$d" "$BEAGLE_PYTHON" "$W/reference.py" > "$W/$l.txt" 2>&1
    fi
    echo $? > "$W/$l.rc"
    for i in $(seq 100); do grep -q 'client done' "$W/$l.dev" && break; sleep 0.1; done
    kill $srv 2>/dev/null; wait $srv 2>/dev/null; rm -rf "$d"
}
cat > "$W/reference.py" <<'EOF'
# the pin's own session: tinygrad's RemotePCIDevice, PCIIfaceBase.__init__'s RESIZE_BAR, then AMDev unmodified (it goes on to
# boot; this stub client has no MAP_SYSMEM_FD, so it stops in init_sw, long after the discovery)
import os, sys, socket, contextlib
sys.path.insert(0, os.environ["TG_TESTS"]); import tgpaths
sys.path.insert(0, tgpaths.TINYGRAD_PATH); tgpaths.block_network()
from tinygrad.runtime.support.system import RemotePCIDevice
from tinygrad.runtime.support.am.amdev import AMDev
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.connect(os.environ["APL_REMOTE_SOCK"])
dev = RemotePCIDevice("AM", "usb4", s)
with contextlib.suppress(Exception): dev.resize_bar(0)
try: AMDev(dev)
except Exception as e: print(f"the reference stopped: {type(e).__name__}: {str(e)[:200]}")
EOF
cat > "$W/requests.py" <<'EOF'
# a recording's requests: <cmd> <bar> <a0> <a1> <a2> per line (TinyGPU.app's 33-byte header; an MMIO_WRITE's payload skipped)
import struct, sys
b = open(sys.argv[1], "rb").read(); i = 0
while i < len(b):
    cmd, dev, bar, a0, a1, a2 = struct.unpack_from("<BIIQQQ", b, i); i += 33
    if cmd == 7: i += a1
    print(cmd, bar, a0, a1, a2)
EOF
export TG_TESTS

# 1. the capture, and the pin's own session on the same card
session cap capture
session ref reference
check "the capture: exit 0, the table equal to the stored RX 7900 XT capture byte for byte" \
    "[ \"\$(cat $W/cap.rc)\" = 0 ] && cmp -s $W/disc_cap/1002_744c_56dd0ec116e3d382.bin ${STORED%.json}.bin"
cat > "$W/samejson.py" <<'PYEOF'
import json, sys
a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
keys = ("pci_id", "bytes", "sha256", "ip_ver", "regs_offset", "harvested", "vram_size", "large_bar", "vram_bar_bytes", "gc_info", "reserved_vram_size")
bad = [k for k in keys if a.get(k) != b.get(k)]
if bad: print("differ:", bad)
sys.exit(1 if bad else 0)
PYEOF
check "the capture's .json: the stored IP versions, bases, harvest, VRAM, BAR0, gc_info and reserved VRAM" \
    "\"$BEAGLE_PYTHON\" $W/samejson.py $W/disc_cap/1002_744c_56dd0ec116e3d382.json $STORED"
"$BEAGLE_PYTHON" "$W/requests.py" "$W/cap.rec.0" > "$W/cap.req"; "$BEAGLE_PYTHON" "$W/requests.py" "$W/ref.rec.0" > "$W/ref.req"
n=$(($(wc -l < "$W/cap.req") - 3))
check "its requests but the last three (the identity reads: config dwords 0x00, 0x08, 0x2c) are byte for byte the start of the pin's session ($n requests)" \
    "head -c \$(( \$(wc -c < $W/cap.rec.0) - 99 )) $W/cap.rec.0 | cmp -s - <(head -c \$(( \$(wc -c < $W/cap.rec.0) - 99 )) $W/ref.rec.0) \
     && [ \"\$(tail -3 $W/cap.req | cut -d' ' -f1,3 | tr '\n' ' ')\" = '3 0 3 8 3 44 ' ] && [ \$(wc -l < $W/ref.req) -gt \$((n + 3)) ]"
check "it writes nothing but LNKCTL (one config write) and the index registers (BAR5 dwords 0x00 and 0x06, 2,560 each)" \
    "[ \"\$(awk '\$1 == 4' $W/cap.req | wc -l | tr -d ' ')\" = 1 ] && [ \"\$(awk '\$1 == 7 && \$2 == 5 && (\$3 == 0 || \$3 == 24)' $W/cap.req | wc -l | tr -d ' ')\" = 5120 ] \
     && [ \"\$(awk '\$1 == 7' $W/cap.req | wc -l | tr -d ' ')\" = 5120 ] && grep -q 'NO ERRORS' $W/cap.dev"

# 2. the stops: before any index write
for m in 0 ffffffff; do
    session mem_$m capture FAKE_AMD_MEMSIZE=$m
    "$BEAGLE_PYTHON" "$W/requests.py" "$W/mem_$m.rec.0" > "$W/mem_$m.req"
    check "MEMSIZE 0x$m: it stops (exit 1) with no MMIO write" \
        "[ \"\$(cat $W/mem_$m.rc)\" = 1 ] && grep -q 'STOP: MEMSIZE reads 0x$m' $W/mem_$m.txt && ! awk '\$1 == 7' $W/mem_$m.req | grep -q ."
done
session bar512 capture FAKE_AMD_BAR0_MB=512
"$BEAGLE_PYTHON" "$W/requests.py" "$W/bar512.rec.0" > "$W/bar512.req"
check "a 512 MiB BAR0: it stops (exit 1) with no MMIO write" \
    "[ \"\$(cat $W/bar512.rc)\" = 1 ] && grep -q 'STOP: BAR0 is 512 MiB' $W/bar512.txt && ! awk '\$1 == 7' $W/bar512.req | grep -q ."

ls -lR "$DLCACHE" > "$W/dlcache_after.txt" 2>&1
check "tinygrad's download cache is unchanged" "cmp -s $W/dlcache_before.txt $W/dlcache_after.txt"
left=$(ps -axo command= | awk '$0 ~ /fake_amd_device\.py/' | wc -l | tr -d ' ')
check "no fake is left running" "[ $left -eq 0 ]"
echo; [ $fails -eq 0 ] && echo "test_n3: PASS" || echo "test_n3: $fails FAILED"
[ $fails -eq 0 ]
