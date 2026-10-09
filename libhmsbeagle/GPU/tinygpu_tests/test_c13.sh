#!/bin/bash
# TODO.md plan step C13c, end to end with no eGPU: the checks that ran on the fake daemon (fake_nv_daemon.py and
# fake_tinygpu_server.py, removed with the daemon path), now on fake_nv_device.py, which the plugin boots itself. On the fake
# AD107 unless it says otherwise:
#   - runs at 4 and 64 states, and on the fake GB205 (the probe's 10de:2f04, sm_120 on QMD v5, the COT unload): each uploads the
#     image compile_all's path would have (check_upload.py, on the fake's copy log), and tears the GPU down;
#   - the routing: a Turing's device ID (0x1e04) is refused before the boot, with nothing written and no guard;
#   - a GPU no embedded cubin serves (sm_75): refused after the boot, which is still torn down at exit;
#   - every d1_runs.txt line (plan step D1) launches exactly the kernels it lists (d1_verdict);
#   - several instances in one process (plan step P5) share one boot and one connection besides the probe's, each instance's
#     image uploaded where it loaded it, each reading back its own tip partials: two at 4 and 64 states, four in threads, two
#     cycles, two of one state count, and a child forked after the boot that exits normally (the parent's GPU stays up); a second
#     process fails at once on nv_usb4.lock;
#   - the command ring wraps (10,000 evaluations);
#   - BEAGLE_NV_TEARDOWN=0: the unload only, and the report says a power cycle is needed (run_point.sh's stop rule fails it);
#   - the GB205's RISC-V core never halting after the unload (FAKE_NO_HALT): the teardown fails, and the guard holds;
#   - the boot failing: booter_load's MAILBOX0 0x29, before GSP-RM started (the guard closes); a GSP core that is not active
#     after booter_load (GSP-RM may run: the guard holds); once GSP-RM posted INIT_DONE, the GSP refusing the channel group (both
#     fakes) or the WPR check refusing the VRAM pool (BEAGLE_NV_DATA_MB=7900): the plugin unloads GSP-RM and runs the teardown
#     itself, and the guard exits at its clean, but a GPU that hangs in the NVDevice's setup work (FAKE_GPU_HANG_AT=1) still
#     holds;
#   - the plugin killed mid-frame and inside its own teardown (the guard holds, sending nothing), right after a launch batch's or
#     a copy's submission with the GPU behind (the guard waits for the timeline, then tears the GPU down), and idle with a GSP
#     that never answers the unload (the guard's unload times out, and it holds): on the fake AD107 and the fake GB205.
# One PASS or FAIL line per check; exit 0 only if all pass.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
require_no_launch_guard
unset FAKE_TEST_BIN FAKE_NV_CHIP
W="$TINYGPU_TEST_WORK/c13"; rm -rf "$W"; mkdir -p "$W"
fails=0
pass() { echo "PASS $1"; }
fail() { echo "FAIL $1"; fails=$((fails + 1)); }
check() { if eval "$2"; then pass "$1"; else fail "$1"; fi; }
TL="$TINYGPU_TEST_WORK/beagle_tinygpu_offline.log"   # the plugin's and the guard's TinyGPULog lines in these runs
out() { echo "$TINYGPU_TEST_WORK/run_device_$1.txt"; }   # the plugin's output ($W/<label>.txt: run_fake_device.sh's)
dev() { echo "$TINYGPU_TEST_WORK/fake_device_$1.log"; }
device() { grep -E "fake TinyGPU.app \((AD107|GB205) device\): " "$(dev $1)" | tail -1; }
counts() { grep -E "fake TinyGPU.app \((AD107|GB205) device\): client done: " "$(dev $1)" | tail -1; }
glog() { sed -n "/c13 run $1 starts/,\$p" "$TL"; }   # this run's TinyGPULog lines
status() { sed -n "s/^\[$1\] tinygputest exit=\([0-9]*\) .*/\1/p" "$W/$1.txt"; }   # the test's own exit status
behind() {   # the guard's state-page line says the C++ timeline was behind the last submission
    glog $1 | sed -nE 's/.*last_submitted ([0-9]+), seq [0-9]+, C\+\+ timeline ([0-9]+).*/\1 \2/p' | awk '$2 < $1 {ok = 1} END {exit !ok}'
}
run() {   # <label> [VAR=value ...] [-- test args]: one fake run, its TinyGPULog lines marked
    local l=$1 envs=() args=(--state-count 4 --reps 3); shift
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done
    [ "$1" = "--" ] && { shift; args=("$@"); }
    echo "c13 run $l starts" >> "$TL"
    "$TG_TESTS/run_fake_device.sh" $l "${envs[@]}" -- "${args[@]}" > "$W/$l.txt" 2>&1
}
upload() {   # <label> <state counts> <arch>: check_upload.py on the run's copy log ($W/copies_<label>.bin)
    "$BEAGLE_PYTHON" "$TG_TESTS/check_upload.py" "$(out $1)" "$W/copies_$1.bin" "$2" "$3" > "$W/upload_$1.log" 2>&1
}
UNLOAD='"rpc NV_VGPU_MSG_FUNCTION_UNLOADING_GUEST_DRIVER": 1'
CLEAN="the plugin tore the GPU down itself; exiting"

# 1. runs at 4 and 64 states on the fake AD107 and the fake GB205, and the images they uploaded
for chip in ad107 gb205; do
    for states in 4 64; do
        l=c13_${chip}_$states
        if [ $chip = gb205 ]; then
            FAKE_NV_CHIP=gb205 FAKE_COPY_LOG="$W/copies_$l.bin" run $l -- --state-count $states --reps 5; r=$?
            upload $l $states sm_120; u=$?
            check "GB205 at $states states: PASS (the probe's 10de:2f04, sm_120 on QMD v5, the RISC-V core halted after the unload), and the image uploaded is compile_all's" \
                "[ $r -eq 0 ] && [ $u -eq 0 ] && grep -q 'TinyGPU: device 0 PCI id = 10de:2f04' '$(out $l)' \
                 && grep -q '(level boot: sm_120, QMD v5' '$(out $l)' && grep -q 'TinyGPU/NV: teardown: done: GSP RISC-V halted' '$(out $l)'"
        else
            FAKE_COPY_LOG="$W/copies_$l.bin" run $l -- --state-count $states --reps 5; r=$?
            upload $l $states sm_89; u=$?
            check "AD107 at $states states: PASS, and the image uploaded is compile_all's" "[ $r -eq 0 ] && [ $u -eq 0 ]"
        fi
    done
done

# 2. the routing: a GPU outside tinygrad's families is refused before the boot (an Ampere boots since plan step G1)
FAKE_PCI_DEVICE_ID=1e04 run c13_tu102
check "a Turing's device ID (0x1e04): refused before the boot (beagleCreateInstance fails), no guard, nothing but the probe's two reads" \
    "grep -q 'this GPU (PCI device ID 1e04) is not one BEAGLE boots' '$(out c13_tu102)' && grep -q 'beagleCreateInstance failed (error -1)' '$(out c13_tu102)' \
     && ! grep -q 'crash guard' '$(out c13_tu102)' && counts c13_tu102 | grep -q 'client done: {\"cmd 3\": 2}$' && device c13_tu102 | grep -q 'NO ERRORS'"

# 3. a GPU no embedded cubin serves
FAKE_SM_VERSION=0x705 run c13_sm75
check "a GPU no embedded cubin serves (sm_75): refused after the boot (beagleCreateInstance fails), and the GPU still torn down at exit" \
    "grep -q 'C++ runtime: no embedded cubin for this GPU.s architecture (sm_75); this build has sm_86, sm_89, sm_120' '$(out c13_sm75)' \
     && grep -q 'beagleCreateInstance failed (error -1)' '$(out c13_sm75)' && fini_verdict '$(out c13_sm75)' && device c13_sm75 | grep -q 'NO ERRORS'"

# 4. every d1_runs.txt line, launching exactly the kernels it lists (run_d1.sh's hardware run must launch the same: which kernels
#    run depends only on the arguments); the fake's memory is not kept
while IFS='|' read -r label cmd kernels; do
    set -- $cmd; bin=$1; shift
    FAKE_TEST_BIN="$BEAGLE_BUILD/examples/$bin" run c13_d1_$label -- "$@" < /dev/null; r=$?
    rm -rf "$TINYGPU_TEST_WORK/fake_device_c13_d1_$label"
    if why=$(d1_verdict "$(out c13_d1_$label)" "$(out c13_d1_$label)" "$kernels") && [ $r -eq 0 ] && [ "$(status c13_d1_$label)" = 0 ]; then
        pass "D1 $label: PASS, exactly the line's kernels"
    else fail "D1 $label: ${why:-the run failed (see $W/c13_d1_$label.txt)}"; fi
done < <(grep -E '^[a-z0-9_]+\|' "$TG_TESTS/d1_runs.txt")

# 5. several instances in one process
p5() {   # <label> <state counts, in load order> -- <tinygputest args>
    local l=$1 states=$2 why=""; shift 3
    FAKE_COPY_LOG="$W/copies_$l.bin" run $l -- "$@" || why=" run"
    upload $l $states sm_89 || why="$why upload"
    [ "$(grep -c "TinyGPU/NV: level boot: the C++ boot, with no daemon" "$(out $l)")" -eq 1 ] || why="$why boots"
    [ "$(grep -c "client done" "$(dev $l)")" -eq 2 ] || why="$why connections"
    grep -q "^tips: every instance read back its own tip partials exactly" "$(out $l)" && ! grep -q "^tips: an instance" "$(out $l)" || why="$why tips"
    if [ -z "$why" ]; then pass "P5 $l: one boot, one connection besides the probe's, each instance's image and tips, one teardown"
    else fail "P5 $l: ${why# } (see $W/$l.txt)"; fi
}
p5 c13_p5_two 4,64 -- --state-count 4,64 --reps 5
p5 c13_p5_threads 4,64,16,128 -- --instances 4 --threads --state-count 4,64,16,128 --reps 300
p5 c13_p5_cycles 4,64,4,64 -- --cycles 2 --state-count 4,64 --reps 20
p5 c13_p5_same 4,4 -- --state-count 4,4 --threads --reps 20
p5 c13_p5_fork 4,64 -- --state-count 4,64 --fork-exit --reps 5
check "P5: the child forked after the boot exited normally, and the parent's GPU stayed up (the run above)" \
    "grep -q '^forked child [0-9]* exited with status 0' '$(out c13_p5_fork)'"
FAKE_SECOND_AFTER="C\+\+ runtime: [0-9]+ kernels loaded" run c13_p5_lock -- --state-count 4 --reps 3000; r=$?
check "P5: a second process fails at once on nv_usb4.lock, touching no GPU, while the first runs on (PASS)" \
    "[ $r -eq 0 ] && grep -qE 'second process exit=[1-9][0-9]* after [0-5] s' '$W/c13_p5_lock.txt' \
     && grep -q 'TinyGPU: Failed to acquire lock file nv_usb4.lock' '$(out c13_p5_lock_second)' && ! grep -q 'TinyGPU/NV:' '$(out c13_p5_lock_second)'"

# 6. the command ring wraps after 2 MiB of pushbuffers (about 4,400 evaluations): the wrap must wait for the frames before the one
#    being submitted, not for that one (which never completes: a false hung GPU)
run c13_wrap -- --state-count 4 --reps 10000; r=$?
check "10,000 evaluations: the command ring wraps, PASS" "[ $r -eq 0 ]"

# 7. no teardown
run c13_td0 BEAGLE_NV_TEARDOWN=0
check "BEAGLE_NV_TEARDOWN=0: the unload only (no FWSEC-SB, no Booter Unload), the report says to power-cycle, and the stop rule fails it" \
    "grep -q 'no teardown result (WPR2 is still up); power-cycle the eGPU before the next boot' '$(out c13_td0)' && ! fini_verdict '$(out c13_td0)' \
     && counts c13_td0 | grep -q '$UNLOAD' && ! counts c13_td0 | grep -q 'FWSEC-SB' && device c13_td0 | grep -q 'NO ERRORS' && glog c13_td0 | grep -q '$CLEAN'"

# 8. a GB205 whose RISC-V core never halts after the unload
FAKE_NV_CHIP=gb205 FAKE_NO_HALT=1 run c13_nohalt
check "GB205, a RISC-V core that never halts after the unload: the teardown fails, the plugin says the guard keeps the connection, the guard holds, and the stop rule fails it" \
    "grep -q 'TinyGPU/NV: teardown: failed: GSP RISC-V did not halt within 4 s' '$(out c13_nohalt)' && grep -q 'keeps the TinyGPU.app connection open' '$(out c13_nohalt)' \
     && glog c13_nohalt | grep -q 'HOLDING the TinyGPU.app connection (the plugin.s own GPU teardown was not confirmed)' && ! fini_verdict '$(out c13_nohalt)'"

# 9. the boot failing
FAKE_FALCON_FAIL=booter run c13_booter
check "booter_load returns MAILBOX0 0x29: GSP-RM never started, and the guard closes (device NO ERRORS)" \
    "grep -Eq 'level boot: building the NVDevice: AssertionError: Booter failed to execute, mailbox is 00000029, [0-9a-f]{8}$' '$(out c13_booter)' \
     && glog c13_booter | grep -q 'the plugin.s boot stopped before GSP-RM started: nothing to unload, closing is safe' \
     && device c13_booter | grep -q 'NO ERRORS' && counts c13_booter | grep -q '\"booter_load\": 1'"
FAKE_FALCON_FAIL=core run c13_core
check "booter_load ran but the GSP core is not active: GSP-RM may run, and the guard holds" \
    "grep -q 'level boot: building the NVDevice: AssertionError: GSP Core is not active' '$(out c13_core)' \
     && glog c13_core | grep -q 'HOLDING the TinyGPU.app connection (the plugin did not finish booting GSP-RM)' \
     && grep -q 'the guard (pid [0-9]*) held the fake connection' '$W/c13_core.txt'"
for chip in ad107 gb205; do
    C=$(echo $chip | tr a-z A-Z)
    FAKE_NV_CHIP=$([ $chip = gb205 ] && echo gb205) FAKE_RM_FAIL=0xa06c run c13_${chip}_rmfail
    check "$C: the GSP refuses the channel group while the NVDevice is built: the boot fails with its status, nothing is submitted, the plugin unloads GSP-RM and tears the GPU down itself, and the guard exits at its clean (device NO ERRORS)" \
        "grep -q 'level boot: building the NVDevice: RuntimeError: RPC call 103 failed with result 34' '$(out c13_${chip}_rmfail)' \
         && counts c13_${chip}_rmfail | grep -q '\"rm_alloc refused (FAKE_RM_FAIL)\": 1' && ! counts c13_${chip}_rmfail | grep -q 'doorbells' \
         && fini_verdict '$(out c13_${chip}_rmfail)' && device c13_${chip}_rmfail | grep -q 'NO ERRORS' && glog c13_${chip}_rmfail | grep -q '$CLEAN'"
done
run c13_pool7900 BEAGLE_NV_DATA_MB=7900
check "a VRAM pool that reaches GSP-RM's reserved region, refused after the NVDevice is built: the plugin unloads GSP-RM and tears the GPU down itself, and the guard exits at its clean (device NO ERRORS)" \
    "grep -q 'level boot: VRAM allocations end at 0x[0-9a-f]*, above the WPR bound' '$(out c13_pool7900)' && fini_verdict '$(out c13_pool7900)' \
     && grep -q 'out of GPU memory: a 7900 MiB VRAM pool (BEAGLE_NV_DATA_MB: lower it) reaches GSP-RM.s reserved region; at most [0-9]* MiB fit' '$(out c13_pool7900)' \
     && grep -q 'beagleCreateInstance failed (error -2)' '$(out c13_pool7900)' \
     && device c13_pool7900 | grep -q 'NO ERRORS' && glog c13_pool7900 | grep -q '$CLEAN'"
FAKE_GPU_HANG_AT=1 run c13_setup_hang
check "a GPU that hangs in the NVDevice's setup work: the boot fails on its timeline, the plugin sends no unload, and the guard holds" \
    "grep -q 'level boot: building the NVDevice: RuntimeError: Wait timeout' '$(out c13_setup_hang)' && ! grep -q 'GPU teardown' '$(out c13_setup_hang)' \
     && glog c13_setup_hang | grep -q 'HOLDING the TinyGPU.app connection' && grep -q 'the guard (pid [0-9]*) held the fake connection' '$W/c13_setup_hang.txt'"

# 10. the plugin killed: mid-frame, in its own teardown, mid-batch and mid-copy with the GPU behind, and idle with a silent GSP
for chip in ad107 gb205; do
    if [ $chip = gb205 ]; then export FAKE_NV_CHIP=gb205; else unset FAKE_NV_CHIP; fi
    C=$(echo $chip | tr a-z A-Z)
    for k in frame teardown; do
        run c13_${chip}_$k BEAGLE_NV_TEST_KILL=$k
        why=$([ $k = frame ] && echo 'a frame may be cut mid-send' || echo 'the plugin.s own GPU teardown did not finish')
        check "$C: killed $([ $k = frame ] && echo mid-frame || echo 'in its own teardown'): the guard holds and sends nothing" \
            "glog c13_${chip}_$k | grep -q 'HOLDING the TinyGPU.app connection ($why)' && grep -q 'the guard (pid [0-9]*) held the fake connection' '$W/c13_${chip}_$k.txt' \
             && ! grep -q '$UNLOAD' '$(dev c13_${chip}_$k)'"
    done
    for k in batch copy; do
        FAKE_GPU_LAG_MS=500 run c13_${chip}_$k BEAGLE_NV_TEST_KILL=$k
        check "$C: killed mid-$k with the GPU behind: the guard waits for the C++ timeline, then tears the GPU down (device NO ERRORS)" \
            "behind c13_${chip}_$k && device c13_${chip}_$k | grep -q 'NO ERRORS' && glog c13_${chip}_$k | grep -q 'the GPU is torn down; closing'"
    done
    FAKE_GSP_SILENT_UNLOAD=1 run c13_${chip}_silent BEAGLE_NV_TEST_KILL=idle
    check "$C: a silent GSP, the plugin killed idle: the guard's unload RPC times out, and it holds" \
        "glog c13_${chip}_silent | grep -q 'Timeout waiting for RPC response for command 47' \
         && glog c13_${chip}_silent | grep -q 'HOLDING the TinyGPU.app connection (the GPU did not confirm its teardown)' \
         && grep -q 'unload RPC left unanswered' '$(dev c13_${chip}_silent)'"
done
unset FAKE_NV_CHIP

left=$(ps -axo command | grep -cE "Python .*(tgproxy|tgreplay|fake_nv_device)\.py|[b]eagle-tinygpu-guard")
check "no harness process or guard is left running" "[ $left -eq 0 ]"
echo; [ $fails -eq 0 ] && echo "test_c13: PASS" || echo "test_c13: $fails FAILED"
[ $fails -eq 0 ]
