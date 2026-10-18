#!/bin/bash
# fused-handoff/PLAN-fused-uppartials.md section 6 (session F6), offline: the TinyGPU fused implementation's routing on NV
# (BUILD_FUSED). Every f1_runs.txt line marked fake or both runs fusedbench, with --k 1 --reps 1 appended, on
# fake_nv_device.py's AD107, GA104 and GB205 (run_fake_device.sh with FAKE_TEST_BIN; its own verdict is ignored: the fakes
# compute nothing, so fusedbench's checks fail and its exit status means nothing here). Each run must pass fused_verdict
# (env.sh) when it runs B only (--impls B): one boot, the standard modules only, no launch rejected, no kernel missing and no
# failed step, B TinyGPU's fused instance, one kernelUpPartialsFused launch per B updatePartials call, and no standard
# pruning or scaling kernel; any other line fused_run_verdict, with A TinyGPU's standard instance and B its fused one in
# the same process. Needs fusedbench (BUILD_FUSED); without it nothing runs (exit 2).
# The runs go in parallel (env.sh pjob), then one PASS or FAIL line per check; exit 0 only if all pass.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
require_no_launch_guard
W="$TINYGPU_TEST_WORK/f1"; rm -rf "$W"; mkdir -p "$W"
export TG_PIDS="$W/pids"   # the fakes and guards the runs start (env.sh's track_pid), for the check that none is left running
pjob_init "$W"
[ -x "$BEAGLE_BUILD/examples/fusedbench" ] || { echo "no $BEAGLE_BUILD/examples/fusedbench (a BUILD_FUSED build); not running"; exit 2; }
fails=0
pass() { echo "PASS $1"; }
fail() { echo "FAIL $1"; fails=$((fails + 1)); }
check() { if eval "$2"; then pass "$1"; else fail "$1"; fi; }

out() { echo "$TINYGPU_TEST_WORK/run_device_$1.txt"; }   # the plugin's and the program's output (run_fake_device.sh's)
lines() { grep -E '^[A-Za-z0-9_]+\|.*\|(fake|both)$' "$TG_TESTS/f1_runs.txt"; }

while IFS='|' read -r label args where; do
    for chip in ad107 ga104 gb205; do
        l=f1_${chip}_$label
        pjob $l env FAKE_NV_CHIP=${chip#ad107} FAKE_DROP_AT=0 FAKE_TEST_BIN="$BEAGLE_BUILD/examples/fusedbench" \
            FAKE_RUN_TIMEOUT=${FAKE_RUN_TIMEOUT:-300} "$TG_TESTS/run_fake_device.sh" $l -- ${args//\$REPO/$REPO} --k 1 --reps 1 \
            > "$W/$l.txt" 2>&1 < /dev/null
    done
done < <(lines)
pwait

while IFS='|' read -r label args where; do
    for chip in ad107 ga104 gb205; do
        l=f1_${chip}_$label; o=$(out $l)
        if [[ " $args " == *" --impls B "* ]]; then
            why=$(fused_verdict "$o" "$o")
            check "$label on the fake $(echo $chip | tr a-z A-Z): one boot, standard modules, one kernelUpPartialsFused launch per B call ($(sed -n 's/^B updatePartials calls: //p' "$o") calls)${why:+ ($why)}" \
                "fused_verdict '$o' '$o' > /dev/null"
        else
            why=$(fused_run_verdict "$o" "$o")
            check "$label on the fake $(echo $chip | tr a-z A-Z): one boot, standard modules, A TinyGPU's standard instance and B its fused one${why:+ ($why)}" \
                "fused_run_verdict '$o' '$o' > /dev/null && grep -qE '^  A +TinyGPU-(Single|Double) ' '$o' && grep -qE '^  B +TinyGPU-(Single|Double)-Fused ' '$o'"
        fi
    done
done < <(lines)

left=$(tracked_alive)   # this stage's own processes only: other stages' fakes and guards may be running
check "no harness process or guard is left running" "[ -z \"\$left\" ]"
echo; [ $fails -eq 0 ] && echo "test_f1: PASS" || echo "test_f1: $fails FAILED"
[ $fails -eq 0 ]
