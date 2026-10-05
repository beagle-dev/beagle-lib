#!/bin/bash
# TODO.md plan step C12, end to end with no eGPU: the Python-free default, and a library that returns errors instead of exiting
# its host. On fake_nv_device.py, the fake AD107 and (where it says so) the fake GB205:
#   - the plugin boots at level boot, with no daemon; so does a GB202's device ID on the fake GB205, the device receiving the
#     GB205's bytes (every GB20x family boots in C++ since 2026-09-28; the refusal of the other GPUs is test_c13.sh's);
#   - the finalize order: with the GPU running each doorbell's work 20 ms late (FAKE_GPU_LAG_MS), the plugin's teardown at exit
#     waits for its timeline before the unload RPC, then runs NVIDIA's teardown (FWSEC-SB, then Booter Unload) or, on COT, the
#     RISC-V halt wait, and only then closes the connection: NO ERRORS;
#   - a GSP that never answers the unload (FAKE_GSP_SILENT_UNLOAD=1): the plugin's own unload at exit times out, the guard
#     holds, and the test still exits normally;
#   - the exit matrix, where nothing hangs and nothing sends RESET or any other command BEAGLE never sends (the fake flags those):
#     exit() from another thread mid-run (the plugin's atexit tears the GPU down under its timed lock); SIGINT to the test's
#     process group mid-run, as a terminal's Ctrl-C (the test stops and the plugin tears down; the guard, in its own session,
#     survives to see the clean); TinyGPU.app gone mid-run (FAKE_DROP_AT): the next write fails with EPIPE, not SIGPIPE, the GPU
#     is lost, the guard holds, and BEAGLE returns errors to a test that exits normally; a warm GPU: an error from
#     beagleCreateInstance, with nothing written;
#   - a failed instance on a healthy GPU (a 1 MiB VRAM pool, BEAGLE_NV_DATA_MB=1): BEAGLE_ERROR_OUT_OF_MEMORY from
#     beagleCreateInstance (plan step M1), and the
#     GPU is still torn down at exit;
#   - a GPU that hangs mid-run (FAKE_GPU_HANG_AT): the plugin's timeline wait times out, the GPU is lost, BEAGLE returns errors,
#     and the guard, whose own timeline wait then fails, sends only the unload RPC and holds; run_point.sh's stop rule fails it.
# The test exits 1 on the fakes (their logL is wrong by design): "exits normally" is a status below 128, not a signal's. After
# exit() from another thread, BEAGLE's core may still crash the main thread, which finalizes its instance on state exit()
# destroyed (getBeagleInstance); that crash comes after the plugin's teardown, so there any status after it counts.
# One PASS or FAIL line per check; exit 0 only if all pass.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
require_no_launch_guard
W="$TINYGPU_TEST_WORK/c12"; rm -rf "$W"; mkdir -p "$W"
fails=0
pass() { echo "PASS $1"; }
fail() { echo "FAIL $1"; fails=$((fails + 1)); }
check() { if eval "$2"; then pass "$1"; else fail "$1"; fi; }
TL="$TINYGPU_TEST_WORK/beagle_tinygpu_offline.log"   # the plugin's and the guard's TinyGPULog lines in these runs
out() { echo "$TINYGPU_TEST_WORK/run_device_$1.txt"; }   # the plugin's output ($W/<label>.txt: run_fake_device.sh's)
dev() { echo "$TINYGPU_TEST_WORK/fake_device_$1.log"; }
device() { grep -E "fake TinyGPU.app \((AD107|GB205) device\): " "$(dev $1)" | tail -1; }
counts() { grep -E "fake TinyGPU.app \((AD107|GB205) device\): client done: " "$(dev $1)" | tail -1; }
glog() { sed -n "/c12 run $1 starts/,\$p" "$TL"; }   # this run's TinyGPULog lines
status() { sed -n "s/^\[$1\] tinygpuhybridtest exit=\([0-9]*\) .*/\1/p" "$W/$1.txt"; }   # the test's own exit status
run() {   # <label> [VAR=value ...] [-- tinygpuhybridtest args]: one fake run, its TinyGPULog lines marked
    local l=$1 envs=() args=(--state-count 4 --reps 3); shift
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done
    [ "$1" = "--" ] && { shift; args=("$@"); }
    echo "c12 run $l starts" >> "$TL"
    FAKE_TG_RECORD="$W/$l.bin" "$TG_TESTS/run_fake_device.sh" $l "${envs[@]}" -- "${args[@]}" > "$W/$l.txt" 2>&1
}
UNLOAD='"rpc NV_VGPU_MSG_FUNCTION_UNLOADING_GUEST_DRIVER": 1'
CLEAN="the plugin tore the GPU down itself; exiting"

for chip in ad107 gb205; do
    if [ $chip = gb205 ]; then export FAKE_NV_CHIP=gb205; else unset FAKE_NV_CHIP; fi
    C=$(echo $chip | tr a-z A-Z)

    # 1. the default
    run c12_${chip}_default; r1=$?
    check "$C: the plugin boots at level boot, PASS with no daemon" \
        "[ $r1 -eq 0 ] && grep -q 'built the NVDevice after the C++ boot, with no daemon' '$(out c12_${chip}_default)'"

    # 2. the finalize order, with the GPU behind its doorbells
    FAKE_GPU_LAG_MS=20 run c12_${chip}_lag; r=$?
    check "$C: the finalize order with the GPU 20 ms behind: the timeline, the unload, the teardown, then the close (NO ERRORS)" \
        "[ $r -eq 0 ] && device c12_${chip}_lag | grep -q 'NO ERRORS' && counts c12_${chip}_lag | grep -q '$UNLOAD' \
         && { [ $chip = gb205 ] || { counts c12_${chip}_lag | grep -q '\"FWSEC-SB\": 1' && counts c12_${chip}_lag | grep -q '\"booter_unload\": 1'; }; } \
         && glog c12_${chip}_lag | grep -q '$CLEAN'"

    # 3. a silent GSP at the plugin's own teardown
    FAKE_GSP_SILENT_UNLOAD=1 run c12_${chip}_silent
    check "$C: a GSP that never answers the unload: the plugin's own unload times out, the guard holds, and the test exits normally" \
        "[ \"\$(status c12_${chip}_silent)\" = 1 ] && glog c12_${chip}_silent | grep -q 'HOLDING the TinyGPU.app connection (the plugin.s own GPU teardown was not confirmed)' \
         && grep -q 'the guard (pid [0-9]*) held the fake connection' '$W/c12_${chip}_silent.txt'"

    # 4. exit() from another thread, 300 ms into the evaluations
    run c12_${chip}_exit -- --state-count 4 --reps 20000 --exit-after 300
    check "$C: exit() from another thread mid-run: the plugin's atexit tears the GPU down (NO ERRORS), and the test exits 0 (or BEAGLE's core crashes after the teardown)" \
        "{ [ \"\$(status c12_${chip}_exit)\" = 0 ] || [ \"\$(status c12_${chip}_exit)\" -ge 128 ]; } && grep -q '^exit() from another thread, 300 ms into the evaluations' '$(out c12_${chip}_exit)' \
         && device c12_${chip}_exit | grep -q 'NO ERRORS' && counts c12_${chip}_exit | grep -q '$UNLOAD' && glog c12_${chip}_exit | grep -q '$CLEAN'"
done
unset FAKE_NV_CHIP

# 1b. the other GB20x families: a GB202's device ID on the fake GB205 boots at level boot too
FAKE_NV_CHIP=gb205 FAKE_PCI_DEVICE_ID=2b85 run c12_gb202_default; r=$?
check "GB202 (device ID 0x2b85 on the fake GB205): level boot, PASS with no daemon, and the device received the GB205's bytes" \
    "[ $r -eq 0 ] && grep -q 'PCI id = 10de:2b85' '$(out c12_gb202_default)' \
     && grep -q 'built the NVDevice after the C++ boot, with no daemon' '$(out c12_gb202_default)' && cmp -s '$W/c12_gb202_default.bin' '$W/c12_gb205_default.bin'"

# 5. SIGINT to the test's process group mid-run, as a terminal's Ctrl-C: the test stops and finalizes, the plugin tears the GPU
#    down at exit, and the guard, in its own session, survives the SIGINT to see the clean
FAKE_SIGINT_AFTER="C\+\+ runtime: [0-9]+ kernels loaded" run c12_sigint -- --state-count 4 --reps 20000
check "AD107: SIGINT to the test's process group mid-run: the test stops (exit 130), the plugin tears the GPU down (NO ERRORS), the guard sees the clean" \
    "grep -q 'SIGINT sent to the test.s process group' '$W/c12_sigint.txt' && [ \"\$(status c12_sigint)\" = 130 ] \
     && grep -q 'interrupted by signal 2' '$(out c12_sigint)' && device c12_sigint | grep -q 'NO ERRORS' && glog c12_sigint | grep -q '$CLEAN'"

# 6. TinyGPU.app gone mid-run: EPIPE, not SIGPIPE
FAKE_DROP_AT=100 run c12_epipe -- --state-count 4 --reps 1000
check "AD107: TinyGPU.app gone mid-run: the next write fails (EPIPE, no SIGPIPE death), the GPU is lost, BEAGLE returns errors, the guard holds" \
    "[ \"\$(status c12_epipe)\" = 1 ] && grep -q 'TinyGPU.app write cut mid-frame' '$(out c12_epipe)' \
     && grep -q 'the GPU is lost to this process: nothing more is sent to it' '$(out c12_epipe)' && grep -q -- '--reps: evaluation [0-9]* failed\|failed (error -1)' '$(out c12_epipe)' \
     && glog c12_epipe | grep -q 'HOLDING the TinyGPU.app connection (a frame may be cut mid-send)'"

# 7. a warm GPU: an error from beagleCreateInstance, nothing written
FAKE_WPR2_UP=1 run c12_warm
check "AD107: a warm GPU: beagleCreateInstance returns an error (the test exits normally), after RESIZE_BAR, MAP_BAR and 4 reads (plan step P4's), nothing written" \
    "[ \"\$(status c12_warm)\" = 1 ] && grep -q 'beagleCreateInstance failed (error -1)' '$(out c12_warm)' \
     && counts c12_warm | grep -q '\"cmd 1\": 1, \"cmd 11\": 1, \"cmd 3\": 2, \"cmd 6\": 4}' && device c12_warm | grep -q 'NO ERRORS'"

# 8. a failed instance on a healthy GPU: the GPU is still torn down at exit
run c12_pool BEAGLE_NV_DATA_MB=1
check "AD107: a 1 MiB VRAM pool: beagleCreateInstance returns BEAGLE_ERROR_OUT_OF_MEMORY, and the plugin still tears the GPU down at exit (NO ERRORS)" \
    "[ \"\$(status c12_pool)\" = 1 ] && grep -q 'beagleCreateInstance failed (error -2)' '$(out c12_pool)' && grep -q 'this instance' '$(out c12_pool)' \
     && grep -q 'out of GPU memory: [0-9]* MiB left of the 1 MiB VRAM pool' '$(out c12_pool)' \
     && device c12_pool | grep -q 'NO ERRORS' && counts c12_pool | grep -q '$UNLOAD' && glog c12_pool | grep -q '$CLEAN'"

# 8b. a VRAM pool larger than the GPU's VRAM (plan step M1): the boot's memory manager cannot allocate it, the plugin says so and
#     unloads GSP-RM and tears the GPU down, and beagleCreateInstance returns BEAGLE_ERROR_OUT_OF_MEMORY
run c12_poolbig BEAGLE_NV_DATA_MB=100000
check "AD107: a VRAM pool larger than the VRAM: BEAGLE_ERROR_OUT_OF_MEMORY, said, and the GPU torn down (NO ERRORS)" \
    "[ \"\$(status c12_poolbig)\" = 1 ] && grep -q 'out of GPU memory: the GPU.s [0-9]* MiB of VRAM cannot hold the runtime.s buffers and a 100000 MiB VRAM pool (BEAGLE_NV_DATA_MB: lower it)' '$(out c12_poolbig)' \
     && grep -q 'beagleCreateInstance failed (error -2)' '$(out c12_poolbig)' && fini_verdict '$(out c12_poolbig)' && device c12_poolbig | grep -q 'NO ERRORS'"

# 9. a GPU that hangs mid-run (two 30 s timeline waits: the plugin's, then the guard's)
FAKE_GPU_HANG_AT=100 run c12_hang -- --state-count 4 --reps 1000
check "AD107: a GPU that hangs mid-run: the timeline wait times out, the GPU is lost, BEAGLE returns errors, the guard unloads only and holds, and the stop rule fails it" \
    "[ \"\$(status c12_hang)\" = 1 ] && grep -q 'timeline wait timed out' '$(out c12_hang)' && grep -q 'the GPU is lost to this process (it hung)' '$(out c12_hang)' \
     && glog c12_hang | grep -q 'the hung path' && glog c12_hang | grep -q 'HOLDING the TinyGPU.app connection (the GPU did not confirm its teardown)' \
     && counts c12_hang | grep -q '$UNLOAD' && ! counts c12_hang | grep -q 'FWSEC-SB' && ! fini_verdict '$(out c12_hang)'"

left=$(ps -axo command | grep -cE "Python .*(tgproxy|tgreplay|fake_nv_device)\.py|[b]eagle-tinygpu-guard")
check "no harness process or guard is left running" "[ $left -eq 0 ]"
echo; [ $fails -eq 0 ] && echo "test_c12: PASS" || echo "test_c12: $fails FAILED"
[ $fails -eq 0 ]
