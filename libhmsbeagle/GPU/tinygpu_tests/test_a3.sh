#!/bin/bash
# TODO.md plan step A3, end to end with no eGPU: NV's error returns (plan step C12) on the AMD path. BEAGLE returns errors instead
# of exiting its host, and nothing of a failed instance, or of one whose GPU is lost, reaches the GPU. On fake_amd_device.py's card:
#   - a dirty card: the boot is refused before the mode1 reset, beagleCreateInstance returns an error and the test exits
#     normally; the guard exits at the plugin's 'N';
#   - a 1 MiB VRAM pool (BEAGLE_AMD_DATA_MB=1): the program upload fails after the queues are set up; beagleCreateInstance
#     returns BEAGLE_ERROR_OUT_OF_MEMORY (plan step M1), and the plugin still finalizes the card at exit (the guard sees its
#     clean; NO ERRORS);
#   - a pool the programs fill exactly (26 MiB at 64 states): the instance's first allocation fails, so it fails instead of
#     handing address 0 to the GPU; BEAGLE_ERROR_OUT_OF_MEMORY from beagleCreateInstance, the card finalized;
#   - a pool larger than the VRAM (BEAGLE_AMD_DATA_MB=1000000): the setup says the card's VRAM cannot hold it, and
#     BEAGLE_ERROR_OUT_OF_MEMORY; the card finalized;
#   - a GPU fault mid-run (FAKE_AMD_FAULT=1, on a cold card): the SQ MEMVIOL is decoded, the GPU is lost to the process,
#     read-backs are NaN and BEAGLE returns errors; the exit still finalizes the card (NO ERRORS);
#   - a GPU that hangs mid-run (FAKE_AMD_HANG=1, a 3 s wait timeout): the wait times out, BEAGLE returns errors, and the
#     exit's fini dequeues the hung queue (its waves reset) and finalizes the card (NO ERRORS);
#   - TinyGPU.app gone mid-run (FAKE_AMD_DROP_AT): the next request fails (EPIPE, no SIGPIPE death), BEAGLE returns errors, the
#     test exits normally, and the guard holds: the plugin cannot finalize the card over a dead connection;
#   - SIGINT to the test's process group mid-run, as a terminal's Ctrl-C: the test stops (exit 130) and finalizes, the plugin
#     finalizes the card (NO ERRORS), and the guard, in its own session, sees the clean;
#   - a hold, then another instance in the same process (FAKE_AMD_WEDGED=1, --cycles 2): the first instance's boot fails once
#     its queues are live (a pool larger than the VRAM, BEAGLE_AMD_DATA_MB=1000000), its finalize cannot see the compute
#     queue off, and the guard holds; the second beagleCreateInstance is refused at once (Initialize does not connect, so
#     BeagleGPUImpl sees no device: BEAGLE_ERROR_NO_RESOURCE) instead of waiting forever on TinyGPU.app. (Since plan step A5
#     every other hold comes at exit: the card outlives its instances.)
#   - the firmware, as on NV (every blob located, or downloaded into BEAGLE's cache, before anything is written to the card):
#     with none anywhere and downloads off, refused before the boot (nothing written, no guard); with downloads on, the six
#     blobs downloaded from a file:// copy, then the boot; and the prefetch script's blobs through BEAGLE_TINYGPU_FW;
#   - TODO.md plan step N1: a card whose PCI device ID am::kChips lacks (FAKE_AMD_DEVICE_ID=7550, an RDNA4's) is refused
#     before anything is sent to it (the probe's config read only, no guard, no firmware located or downloaded); a BAR0 of
#     512 MiB (FAKE_AMD_BAR0_MB=512) is refused at the boot (BarLayoutError) before the discovery's first index write.
# The test exits 1 on the fake (its logL is wrong by design): "exits normally" is a status below 128 that the test's own
# error handling gave, not the plugin's exit. A guard that holds keeps the fake's connection as it would the eGPU's, so it is
# ended here (offline only). One PASS or FAIL line per check; exit 0 only if all pass.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
require_no_launch_guard
[ -x "$TEST_BIN" ] || { echo "no $TEST_BIN; build it first"; exit 2; }
GUARD_BIN="$BEAGLE_BUILD/libhmsbeagle/GPU/CMake_TinyGPU/beagle-tinygpu-guard"
[ -x "$GUARD_BIN" ] || { echo "no $GUARD_BIN; build beagle-tinygpu-guard first"; exit 2; }
W="$TINYGPU_TEST_WORK/a3"; rm -rf "$W"; mkdir -p "$W"
TL="$W/beagle_tinygpu.log"   # the plugin's and the guard's TinyGPULog lines in these runs
fails=0
pass() { echo "PASS $1"; }
fail() { echo "FAIL $1"; fails=$((fails + 1)); }
check() { if eval "$2"; then pass "$1"; else fail "$1"; fi; }
glog() { sed -n "/a3 run $1 starts/,/a3 run .* starts/p" "$TL"; }   # one run's TinyGPULog lines
verdict() { grep -E "fake TinyGPU.app \(AMD device\): (NO ERRORS|[0-9]+ ERRORS)" "$W/$1.dev" | tail -1; }
status() { cat "$W/$1.rc"; }
CLEAN="the plugin finalized the GPU itself; exiting"

run() {   # <label> [VAR=value ...] [-- tinygputest args]: one run on a fresh fake card, then its guard: waited for, or
          # ended if it holds. SIGINT_AFTER=<regex> among the VARs: SIGINT to the test's process group once its output matches.
    local l=$1 envs=() args=(--state-count 4 --reps 3) sigint=""; shift
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do
        if [[ $1 == SIGINT_AFTER=* ]]; then sigint=${1#SIGINT_AFTER=}; else envs+=("$1"); fi
        shift
    done
    [ "$1" = "--" ] && { shift; args=("$@"); }
    local d; d=$(mktemp -d /tmp/tg3.XXXXXX)
    mkdir -p "$W/mem_$l"
    echo "a3 run $l starts" >> "$TL"
    env "${envs[@]}" "$BEAGLE_PYTHON" "$TG_TESTS/fake_amd_device.py" "$d/dev.sock" "$W/mem_$l" > "$W/$l.dev" 2>&1 &
    local srv=$!
    for i in $(seq 100); do grep -q listening "$W/$l.dev" 2>/dev/null && break; sleep 0.1; done
    # its own process group (perl before env: macOS strips DYLD_LIBRARY_PATH from a system binary's environment)
    perl -e 'setpgrp(0, 0); exec @ARGV or die "exec: $!"' env BEAGLE_TINYGPU_NO_LAUNCH=1 BEAGLE_TINYGPU_NO_DOWNLOAD=1 BEAGLE_TINYGPU_LOG="$TL" \
        APL_REMOTE_SOCK="$d/dev.sock" TMPDIR="$d" BEAGLE_AMD_GUARD="$TG_TESTS/replay/crash_guard_wrap.sh" BEAGLE_TG_GUARD_BIN="$GUARD_BIN" \
        BEAGLE_TG_GUARD_PIDFILE="$d/guard.pid" BEAGLE_AMD_DATA_MB=512 DYLD_LIBRARY_PATH="$TEST_LIBS" "${envs[@]}" \
        "$TEST_BIN" "${args[@]}" > "$W/$l.txt" 2>&1 &
    local tst=$! rc=124
    for i in $(seq 1800); do
        kill -0 $tst 2>/dev/null || { wait $tst; rc=$?; break; }
        if [ -n "$sigint" ] && grep -qE "$sigint" "$W/$l.txt"; then sigint=""; kill -INT -- -$tst; echo "SIGINT sent" > "$W/$l.sigint"; fi
        sleep 0.1
    done
    [ $rc -eq 124 ] && { kill -KILL $tst 2>/dev/null; wait $tst 2>/dev/null; echo "[$l] the test hung; killed (fake device only)" > "$W/$l.hung"; }
    echo $rc > "$W/$l.rc"
    local gpid; gpid=$(cat "$d/guard.pid" 2>/dev/null)
    if [ -n "$gpid" ]; then   # it exits at the plugin's clean or 'N'; one that holds keeps the fake's connection
        for i in $(seq 600); do kill -0 "$gpid" 2>/dev/null || break; grep -q "then kill $gpid\." "$TL" && break; sleep 0.1; done
        if kill -0 "$gpid" 2>/dev/null; then
            echo "[$l] the guard (pid $gpid) held the fake connection; ending it" > "$W/$l.held"; kill -KILL "$gpid"
            for i in $(seq 50); do kill -0 "$gpid" 2>/dev/null || break; sleep 0.1; done   # not this shell's child: polled
        fi
    fi
    for i in $(seq 100); do [ "$(grep -c 'client done' "$W/$l.dev")" -ge 2 ] && break; sleep 0.1; done
    kill $srv 2>/dev/null; wait $srv 2>/dev/null; rm -rf "$d"
}
normal() { [ "$(status $1)" -lt 128 ] && [ ! -f "$W/$1.hung" ] && ! grep -q "TinyGPU/AMD: .*exiting$" "$W/$1.txt"; }

# 1. a dirty card
run dirty FAKE_AMD_STATE=dirty
check "a dirty card: refused before the mode1 reset, beagleCreateInstance returns an error, the test exits normally, the guard exits ('N')" \
    "normal dirty && grep -q 'needs an SMU mode1 reset' $W/dirty.txt && grep -q 'beagleCreateInstance failed (error -1)' $W/dirty.txt \
     && glog dirty | grep -q 'boot ended with no queue ever live' && verdict dirty | grep -q 'NO ERRORS' && [ ! -f $W/dirty.held ]"

# 2. a 1 MiB VRAM pool: the program upload fails once the queues are live
run pool1 BEAGLE_AMD_DATA_MB=1
check "a 1 MiB VRAM pool: beagleCreateInstance returns BEAGLE_ERROR_OUT_OF_MEMORY, and the plugin still finalizes the card at exit (NO ERRORS)" \
    "normal pool1 && grep -q 'too small' $W/pool1.txt && grep -q 'out of GPU memory: [0-9]* MiB left of the 1 MiB VRAM pool' $W/pool1.txt \
     && grep -q 'beagleCreateInstance failed (error -2)' $W/pool1.txt \
     && glog pool1 | grep -q '$CLEAN' && verdict pool1 | grep -q 'NO ERRORS'"

# 3. a pool the programs fill: the instance's allocation fails instead of handing address 0 to the GPU
run pool26 BEAGLE_AMD_DATA_MB=26 -- --state-count 64 --reps 3
check "a pool the programs fill (26 MiB at 64 states): the instance's allocation fails, BEAGLE_ERROR_OUT_OF_MEMORY from beagleCreateInstance, the card finalized" \
    "normal pool26 && grep -q 'out of GPU memory: an allocation of [0-9.]* MiB, with 0.0 MiB left of the 26 MiB VRAM pool .*; this instance fails' $W/pool26.txt \
     && grep -q 'beagleCreateInstance failed (error -2)' $W/pool26.txt && glog pool26 | grep -q '$CLEAN' && verdict pool26 | grep -q 'NO ERRORS'"

# 3b. a pool larger than the VRAM (plan step M1)
run poolbig BEAGLE_AMD_DATA_MB=1000000
check "a pool larger than the VRAM: the setup says the card's VRAM cannot hold it, BEAGLE_ERROR_OUT_OF_MEMORY, the card finalized (NO ERRORS)" \
    "normal poolbig && grep -q 'out of GPU memory: the card.s [0-9]* MiB of VRAM cannot hold the setup.s buffers and a 1000000 MiB VRAM pool (BEAGLE_AMD_DATA_MB: lower it)' $W/poolbig.txt \
     && grep -q 'beagleCreateInstance failed (error -2)' $W/poolbig.txt && glog poolbig | grep -q '$CLEAN' && verdict poolbig | grep -q 'NO ERRORS'"

# 4. a GPU fault mid-run
run fault FAKE_AMD_STATE=cold FAKE_AMD_FAULT=1 HCQDEV_WAIT_TIMEOUT_MS=3000
check "a GPU fault mid-run: decoded, the GPU lost to the process, BEAGLE returns errors, and the exit finalizes the card (NO ERRORS)" \
    "normal fault && grep -q 'sq_intr: error (MEMVIOL)' $W/fault.txt && grep -q 'the GPU is lost to this process' $W/fault.txt \
     && grep -q 'calculateRootLogLikelihoods failed: -1' $W/fault.txt && glog fault | grep -q '$CLEAN' && verdict fault | grep -q 'NO ERRORS'"

# 5. a GPU that hangs mid-run
run hang FAKE_AMD_HANG=1 HCQDEV_WAIT_TIMEOUT_MS=3000
check "a GPU that hangs mid-run: the wait times out, BEAGLE returns errors, and the exit's fini dequeues the hung queue and finalizes the card (NO ERRORS)" \
    "normal hang && grep -q 'Wait timeout: 3000 ms' $W/hang.txt && grep -q 'the GPU is lost to this process' $W/hang.txt \
     && grep -q 'calculateRootLogLikelihoods failed: -1' $W/hang.txt && glog hang | grep -q '$CLEAN' && verdict hang | grep -q 'NO ERRORS'"

# 6. TinyGPU.app gone mid-run, about 80 evaluations into 2,000 (the boot and the programs take about 43,400 requests, an
#    evaluation about 20)
run drop FAKE_AMD_DROP_AT=45000 -- --state-count 4 --reps 2000
check "TinyGPU.app gone mid-run: the next request fails (no SIGPIPE death), BEAGLE returns errors, the test exits normally, the guard holds" \
    "normal drop && grep -qE 'Connection closed|connection lost' $W/drop.txt && grep -q 'the GPU is lost to this process' $W/drop.txt \
     && grep -q -- '--reps: evaluation [0-9]* failed' $W/drop.txt && glog drop | grep -q 'HOLDING the TinyGPU.app connection' && [ -f $W/drop.held ]"

# 7. SIGINT to the test's process group mid-run
run sigint 'SIGINT_AFTER=C\+\+ runtime: handed over' -- --state-count 4 --reps 20000
check "SIGINT to the test's process group mid-run: the test stops (exit 130) and finalizes, the card is finalized (NO ERRORS), the guard sees the clean" \
    "[ -f $W/sigint.sigint ] && [ \"\$(status sigint)\" = 130 ] && grep -q 'interrupted by signal 2' $W/sigint.txt && glog sigint | grep -q '$CLEAN' && verdict sigint | grep -q 'NO ERRORS'"

# 8. a hold, then another instance in the same process
run held FAKE_AMD_WEDGED=1 BEAGLE_AMD_DATA_MB=1000000 -- --state-count 4 --reps 3 --cycles 2
check "a hold, then another instance in the same process: the guard holds, and the second beagleCreateInstance is refused at once" \
    "normal held && grep -q 'after the C++ boot: MemoryError' $W/held.txt && grep -q '=== cycle 2 of 2' $W/held.txt \
     && grep -q 'the crash guard (pid [0-9]*) holds the eGPU, since an earlier instance' $W/held.txt \
     && grep -q 'Error: No GPU devices' $W/held.txt && grep -q 'beagleCreateInstance failed (error -6)' $W/held.txt && glog held | grep -q 'HOLDING the TinyGPU.app connection' && [ -f $W/held.held ]"

# 9. the firmware, as on NV: every blob located, or downloaded into BEAGLE's cache, before anything is written to the card. An
#    empty XDG_CACHE_HOME hides BEAGLE's and tinygrad's caches; FW_TREE is a file:// copy of linux-firmware's six gfx1100
#    blobs, taken from tinygrad's download cache.
FW_TREE="$W/fw_tree"; mkdir -p "$FW_TREE/amdgpu"
while read -r name md5; do cp "${XDG_CACHE_HOME:-$HOME/Library/Caches}/tinygrad/downloads/fw/$md5" "$FW_TREE/amdgpu/$name"; done \
    < <(sed -n '/^namespace fw {/,/^} \/\/ namespace fw/p' "$GPU_DIR/TinyGPUAMDBootTables.h" \
        | sed -nE 's/^    \{"gfx1100", "[^"]+", "amdgpu", "([^"]+)", "[0-9a-f]{64}", "([0-9a-f]{32})"\},.*/\1 \2/p')
run fw_missing XDG_CACHE_HOME="$W/cache_empty"
check "no firmware anywhere and downloads off: refused before the boot, nothing written to the card and no guard; beagleCreateInstance returns an error" \
    "normal fw_missing && grep -q 'TinyGPU/AMD: not booting: nothing was written to the GPU' $W/fw_missing.txt \
     && grep -q 'beagleCreateInstance failed (error -1)' $W/fw_missing.txt && ! grep -qE 'client done: .*\"cmd (2|4|7)\"' $W/fw_missing.dev \
     && ! glog fw_missing | grep -q 'guard [0-9]*:' && verdict fw_missing | grep -q 'NO ERRORS'"
run fw_download XDG_CACHE_HOME="$W/cache_dl" BEAGLE_TINYGPU_NO_DOWNLOAD=0 BEAGLE_TINYGPU_FW_BASE_URL="file://$FW_TREE"
check "no firmware, downloads on: the six blobs downloaded into BEAGLE's cache before the boot, then the boot and the run, the card finalized (NO ERRORS)" \
    "normal fw_download && [ \"\$(grep -c 'TinyGPU: downloading AMD firmware amdgpu/' $W/fw_download.txt)\" -eq 6 ] \
     && [ \"\$(ls $W/cache_dl/beagle/firmware/amdgpu | wc -l | tr -d ' ')\" -eq 6 ] && grep -q 'C++ boot done' $W/fw_download.txt \
     && glog fw_download | grep -q '$CLEAN' && verdict fw_download | grep -q 'NO ERRORS'"
TINYGPU_FW_BASE_URL="file://$FW_TREE" "$GPU_DIR/tinygpu_fetch_firmware.sh" --chip gfx1100 "$W/fw_dir" > "$W/fw_script.txt" 2>&1
run fw_prefetched XDG_CACHE_HOME="$W/cache_empty2" BEAGLE_TINYGPU_FW="$W/fw_dir"
check "the prefetch script's gfx1100 blobs (BEAGLE_TINYGPU_FW), with no cache and downloads off: the boot and the run, nothing downloaded (NO ERRORS)" \
    "[ \"\$(grep -c '^fetched: ' $W/fw_script.txt)\" -eq 6 ] && normal fw_prefetched && ! grep -q 'downloading' $W/fw_prefetched.txt \
     && grep -q 'C++ boot done' $W/fw_prefetched.txt && glog fw_prefetched | grep -q '$CLEAN' && verdict fw_prefetched | grep -q 'NO ERRORS'"

# 10. TODO.md plan step N1: an AMD card the boot is not for, refused before anything is sent to it (an empty cache and the
#     downloads off: the prefetch would say so if it ran)
run unknown FAKE_AMD_DEVICE_ID=7550 XDG_CACHE_HOME="$W/cache_empty3"
check "an AMD card whose device ID am::kChips lacks (7550): refused before anything is sent to it, no guard, no firmware looked for; beagleCreateInstance returns an error" \
    "normal unknown && grep -q 'TinyGPU/AMD: this card (PCI device ID 7550) is not one BEAGLE boots: 744c (gfx1100) only; nothing was sent to it' $W/unknown.txt \
     && grep -q 'beagleCreateInstance failed (error -1)' $W/unknown.txt && ! grep -qiE 'firmware|not booting' $W/unknown.txt \
     && ! grep -qE 'client done: .*\"cmd (1|2|4|6|7|11)\"' $W/unknown.dev && ! glog unknown | grep -q 'guard [0-9]*:' && verdict unknown | grep -q 'NO ERRORS'"

# 11. a BAR0 but 256 MiB: refused at the boot, before the discovery's index writes (no MMIO write at all: cmd 7)
run bar512 FAKE_AMD_BAR0_MB=512
check "a 512 MiB BAR0: the boot refuses it (BarLayoutError) before the discovery's first index write, the guard exits ('N'), beagleCreateInstance returns an error" \
    "normal bar512 && grep -q 'BarLayoutError: BAR0 is 512 MiB with 20464 MiB of VRAM (large_bar=False): BEAGLE supports only a 256 MiB BAR0' $W/bar512.txt \
     && grep -q 'beagleCreateInstance failed (error -1)' $W/bar512.txt && ! grep -qE 'client done: .*\"cmd 7\"' $W/bar512.dev \
     && glog bar512 | grep -q 'boot ended with no queue ever live' && verdict bar512 | grep -q 'NO ERRORS' && [ ! -f $W/bar512.held ]"

left=$(ps -axo command= | awk '$1 ~ /beagle-tinygpu-guard$/ || $0 ~ /fake_amd_device\.py/' | wc -l | tr -d ' ')
check "no fake or guard is left running" "[ $left -eq 0 ]"
echo; [ $fails -eq 0 ] && echo "test_a3: PASS" || echo "test_a3: $fails FAILED"
[ $fails -eq 0 ]
