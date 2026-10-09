#!/bin/bash
# TODO.md plan step C3, offline: TinyGPUTransport.h against TinyGPU.app's real server.c, compiled from the tinygrad pin
# with server_stub.c standing in for IOKit (server.c is never copied into this repo), plus the socket path and the
# TinyGPU.app check. No eGPU and no TinyGPU.app process is involved: the stub listens on a private socket, and the lock
# lives in a private TMPDIR. Exit status 0 only if every check passes.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
W="$TINYGPU_TEST_WORK/c3"; mkdir -p "$W"
SERVER_C="$TINYGRAD_PATH/extra/usbgpu/tbgpu/installer/Shared/server.c"
[ -f "$SERVER_C" ] || { echo "no $SERVER_C (the tinygrad pin)"; exit 2; }
# the stub names its shared memory as TinyGPU.app does (/tinygpu_N): never beside a hardware run
HW_LOCK="${TMPDIR:-/tmp}"; HW_LOCK="${HW_LOCK%/}/beagle_tinygpu_hw.lock"
[ -d "$HW_LOCK" ] && { echo "a hardware run holds $HW_LOCK; not running"; exit 2; }
cc -O1 -Wno-deprecated-declarations -Wno-address-of-packed-member -o "$W/server_stub" "$SERVER_C" "$TG_TESTS/server_stub.c" || exit 2
c++ -std=c++17 -O1 -I"$REPO" -o "$W/test_transport" "$TG_TESTS/test_transport.cpp" || exit 2
PRIV=$(mktemp -d "${TMPDIR:-/tmp}/tgc3.XXXXXX"); SOCK="$PRIV/s.sock"
SRV=""
cleanup() { [ -n "$SRV" ] && { kill "$SRV"; wait "$SRV"; } 2>/dev/null; rm -rf "${PRIV:?}"; }
trap cleanup EXIT
start_server() {   # a fresh stub server on $SOCK, its output in $W/server_$1.log
    rm -f "${SOCK:?}"
    "$W/server_stub" "$SOCK" > "$W/server_$1.log" 2>&1 &
    SRV=$!
    for i in $(seq 50); do [ -S "$SOCK" ] && return 0; sleep 0.1; done
    echo "FAIL the server stub did not start"; exit 1
}
T() { env TMPDIR="$PRIV" APL_REMOTE_SOCK="$SOCK" BEAGLE_TINYGPU_NO_LAUNCH=1 "$@"; }
fails=0
pass() { echo "PASS $1"; }
fail() { echo "FAIL $1"; fails=$((fails + 1)); }

# the protocol, the client's limits, error replies, sysmem, and a lost server
start_server scenario
T "$W/test_transport" scenario "$SRV" || fails=$((fails + 1))
{ wait "$SRV"; } 2>/dev/null; SRV=""
grep -q "RESET received" "$W/server_scenario.log" && fail "the server received a RESET" || pass "no RESET reached the server"

# tinygrad's temp() for the socket and lock paths
"$W/test_transport" paths || fails=$((fails + 1))

# the TinyGPU.app check: a missing app, a different one, and this computer's (it reads files, nothing more)
out=$(BEAGLE_TINYGPU_APP=/nonexistent/TinyGPU.app "$W/test_transport" app)
[[ "$out" == *"TinyGPU.app/Contents/MacOS/TinyGPU is missing"*"tinygpu_releases/raw/c0d024f9"* ]] \
    && pass "a missing TinyGPU.app is refused, with the pinned release to install" || fail "missing app: $out"
FAKE_APP="$PRIV/Other.app"; mkdir -p "$FAKE_APP/Contents/MacOS" "$FAKE_APP/Contents/Library/SystemExtensions/org.tinygrad.tinygpu.driver2.dext"
echo other > "$FAKE_APP/Contents/MacOS/TinyGPU"; echo other > "$FAKE_APP/Contents/Library/SystemExtensions/org.tinygrad.tinygpu.driver2.dext/org.tinygrad.tinygpu.driver2"
out=$(BEAGLE_TINYGPU_APP="$FAKE_APP" "$W/test_transport" app)
[[ "$out" == *"is not TinyGPU release c0d024f9's"* ]] && pass "a TinyGPU.app of another release is refused" || fail "other app: $out"
if [ -d /Applications/TinyGPU.app ]; then
    out=$(env -u BEAGLE_TINYGPU_APP "$W/test_transport" app)
    [ -z "$out" ] && pass "this computer's TinyGPU.app is release c0d024f9" || fail "installed app: $out"
else echo "SKIP no /Applications/TinyGPU.app on this computer"; fi

# the check refuses before any connection, and a second process fails on the lock without connecting
start_server refusals
out=$(T env BEAGLE_TINYGPU_APP=/nonexistent/TinyGPU.app "$W/test_transport" open)
sleep 0.3
[[ "$out" == *"is missing"* ]] && ! grep -q "client connected" "$W/server_refusals.log" \
    && pass "a refused TinyGPU.app: open() connects nowhere" || fail "app refusal: $out"
T "$W/test_transport" open 3 > "$W/holder.txt" &
HOLDER=$!
for i in $(seq 30); do grep -q "open: ok" "$W/holder.txt" 2>/dev/null && break; sleep 0.1; done
out=$(T "$W/test_transport" open)
wait "$HOLDER"
[[ "$out" == *"Failed to acquire lock file nv_usb4.lock"* ]] && [ "$(grep -c "client connected" "$W/server_refusals.log")" -eq 1 ] \
    && pass "a second process fails on nv_usb4.lock, without connecting" || fail "lock: $out; $(grep -c "client connected" "$W/server_refusals.log") connections"

echo; [ $fails -eq 0 ] && echo "test_c3_transport: PASS" || echo "test_c3_transport: $fails FAILED"
[ $fails -eq 0 ]
