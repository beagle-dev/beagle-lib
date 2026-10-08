#!/bin/bash
# TODO.md plan step N6 (the part plan step N7 needs), offline: amd_state.py, the boot prediction that reads without writing,
# on fake_amd_device.py's RX 7900 XT and, with FAKE_AMD_CHIP=gfx1201, the RDNA 4 card whose table run_amd_discovery_ro.sh
# captured (Navi 48: the MPASP PSP registers, GCVM_L2_PROTECTION_FAULT_STATUS_LO32):
#   - cold, warm and dirty, it predicts what the pin's AMDev decides from those registers (amdev.py:185-197): a full boot
#     without a mode1 reset (exit 0), a partial boot (exit 0), a mode1 reset (exit 3);
#   - it refuses a table of another board (its subsystem changed) and a card whose VRAM is not the table's (exit 1);
#   - it never writes (no CFG_WRITE, no MMIO_WRITE);
#   - env.sh's amd_boot_check refuses a card with no captured table (exit 2);
#   - tgproxy.py --guard and tgreplay.py --guard take the AMD triggers and guard from the session's card's table, by its device
#     ID (amd_state.py through them on the RDNA 4 card: recorded, then replayed); with no table for the card (a data directory
#     holding only the RX 7900 XT's) the proxy fail-stops and the replay diverges;
#   - tgproxy.py --card (for clients that never read config dword 0, as stock tinygrad): the AMD guard is armed before the
#     first request, so a raw client's SMU mode1 request with no dword 0 read before it is refused (without --card it is
#     forwarded); a --card naming another card fail-stops at the dword 0 read; tgreplay.py arms from the recording's --card;
#   - stop rule S8: amd_hw_end stops when the boot taken (the first "AM_<IP> initialized") is not amd_boot_check's prediction.
# One PASS or FAIL line per check; exit 0 only if all pass.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
require_no_launch_guard
W="$TINYGPU_TEST_WORK/n6"; rm -rf "$W"; mkdir -p "$W"
fails=0
pass() { echo "PASS $1"; }
fail() { echo "FAIL $1"; fails=$((fails + 1)); }
check() { if eval "$2"; then pass "$1"; else fail "$1"; fi; }
table() { ls "$BEAGLE_TINYGPU_DATA"/discovery/1002_$1_*.json | head -1; }

state() {   # <label> <table .json> [VAR=value ...]: amd_state.py against a fresh fake card
    local l=$1 tbl=$2; shift 2
    local d; d=$(mktemp -d /tmp/tgn6.XXXXXX)
    env "$@" FAKE_AMD_RECORD="$W/$l.rec" "$BEAGLE_PYTHON" "$TG_TESTS/fake_amd_device.py" "$d/tinygpu.sock" "$W/mem_$l" > "$W/$l.dev" 2>&1 &
    local srv=$!
    for i in $(seq 100); do grep -q listening "$W/$l.dev" 2>/dev/null && break; sleep 0.1; done
    TMPDIR="$d" "$BEAGLE_PYTHON" "$TG_TESTS/amd_state.py" "$tbl" > "$W/$l.txt" 2>&1
    echo $? > "$W/$l.rc"
    for i in $(seq 100); do grep -q 'client done' "$W/$l.dev" && break; sleep 0.1; done
    kill $srv 2>/dev/null; wait $srv 2>/dev/null; rm -rf "$d"
}
writes() { grep -E 'client done' "$W/$1.dev" | tail -1 | grep -qE '"cmd (4|7)"'; }   # CFG_WRITE or MMIO_WRITE

for chip in gfx1100 gfx1201; do
    id=$([ $chip = gfx1100 ] && echo 744c || echo 7550)
    for st in cold warm dirty; do
        state ${chip}_$st "$(table $id)" FAKE_AMD_CHIP=$chip FAKE_AMD_STATE=$st
        case $st in
            cold) want="prediction: a full boot without a mode1 reset"; rc=0 ;;
            warm) want="prediction: a partial boot"; rc=0 ;;
            dirty) want="prediction: a full boot WITH a mode1 reset"; rc=3 ;;
        esac
        check "$chip $st: '$want' (exit $rc), reading only" \
            "[ \"\$(cat $W/${chip}_$st.rc)\" = $rc ] && grep -q '$want' $W/${chip}_$st.txt && ! writes ${chip}_$st"
    done
done
check "gfx1201: it reads the gfx12 names (MPASP_SMN_C2PMSG_81, GCVM_L2_PROTECTION_FAULT_STATUS_LO32)" \
    "grep -q '^regMPASP_SMN_C2PMSG_81 ' $W/gfx1201_warm.txt && grep -q '^regGCVM_L2_PROTECTION_FAULT_STATUS_LO32 ' $W/gfx1201_warm.txt"

# another board of the die: the table's subsystem changed; and VRAM not the table's
"$BEAGLE_PYTHON" -c 'import json, sys; m = json.load(open(sys.argv[1])); m["subsystem"] = "1002:0000"; json.dump(m, open(sys.argv[2], "w"))' "$(table 7550)" "$W/other_board.json"
state other_board "$W/other_board.json" FAKE_AMD_CHIP=gfx1201 FAKE_AMD_STATE=warm
check "a table of another board (subsystem 1002:0000 on a 1eae:8811 card): refused (exit 1), no register read" \
    "[ \"\$(cat $W/other_board.rc)\" = 1 ] && grep -q 'no table given is for this board' $W/other_board.txt && ! grep -qE 'client done: .*\"cmd 6\"' $W/other_board.dev"
state other_vram "$(table 7550)" FAKE_AMD_CHIP=gfx1201 FAKE_AMD_STATE=warm FAKE_AMD_MEMSIZE=1ff0
check "a card whose VRAM is not the table's (8176 MiB): refused (exit 1)" \
    "[ \"\$(cat $W/other_vram.rc)\" = 1 ] && grep -q 'the card has 8176 MiB of VRAM, the table 16304 MiB' $W/other_vram.txt"

# the proxy and the replay server, by the session's device ID
wait_for() { for i in $(seq 100); do grep -q "$2" "$1" 2>/dev/null && return 0; sleep 0.1; done; return 1; }
client() {   # <label> <socket dir>: amd_state.py on the RDNA 4 card's table, 20 s at most (a fail-stop holds it)
    TMPDIR="$2" "$BEAGLE_PYTHON" "$TG_TESTS/amd_state.py" "$(table 7550)" > "$W/$1.txt" 2>&1 &
    local c=$!
    for i in $(seq 200); do kill -0 $c 2>/dev/null || break; sleep 0.1; done
    kill -9 $c 2>/dev/null; wait $c 2>/dev/null
}
mkdir -p "$W/only_744c/discovery"; cp "$BEAGLE_TINYGPU_DATA"/discovery/1002_744c_* "$W/only_744c/discovery/"
mode1_client() {   # <label> <socket dir>: MAP_BAR 5, the SMU's mode1 request (mmMP1_SMN_C2PMSG_75 = 2) and a config read, no dword 0
    TG_TESTS="$TG_TESTS" "$BEAGLE_PYTHON" - "$2/tinygpu.sock" "$(table 7550)" > "$W/$1.txt" 2>&1 <<'EOF'
import os, sys, json, socket, struct
sys.path[:0] = [os.environ["TG_TESTS"], os.environ["TG_TESTS"] + "/replay"]
import tgwire as w, tgguard_amd
mode1 = next(off for (bar, off), n in tgguard_amd.trigger_addrs(json.load(open(sys.argv[2]))).items() if n == "mmMP1_SMN_C2PMSG_75")
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.settimeout(5); s.connect(sys.argv[1])
s.sendall(w.REQ.pack(w.MAP_BAR, 0, 5, 0, 0, 0)); w.recv_exact(s, 17)
s.sendall(w.REQ.pack(w.MMIO_WRITE, 0, 5, mode1, 4, 0) + struct.pack("<I", 2))
s.sendall(w.REQ.pack(w.CFG_READ, 0, 0, 0x34, 1, 0)); print("reply:", len(w.recv_exact(s, 17)), "bytes")
EOF
}
proxied() {   # <label> <data dir> [client [tgproxy args]]: through tgproxy.py --guard, recorded in $W/rec_<label>, to a fresh fake RDNA 4 card
    local l=$1 data=$2 cl=${3:-client} d u; shift 2; shift $(( $# > 0 ? 1 : 0 ))
    d=$(mktemp -d /tmp/tgn6.XXXXXX); u=$(mktemp -d /tmp/tgn6.XXXXXX)
    env FAKE_AMD_CHIP=gfx1201 FAKE_AMD_STATE=cold "$BEAGLE_PYTHON" "$TG_TESTS/fake_amd_device.py" "$u/tinygpu.sock" "$W/mem_$l" > "$W/$l.dev" 2>&1 &
    local srv=$!
    wait_for "$W/$l.dev" listening
    BEAGLE_TINYGPU_DATA="$data" "$BEAGLE_PYTHON" "$TG_TESTS/replay/tgproxy.py" --listen "$d/tinygpu.sock" --upstream "$u/tinygpu.sock" \
        --out "$W/rec_$l" --guard --label "n6 $l" "$@" > "$W/$l.px" 2>&1 &
    local px=$!
    wait_for "$W/$l.px" "tgproxy listening"
    $cl "$l" "$d"
    wait_for "$W/$l.px" "session 1 ended"
    kill -TERM $px 2>/dev/null
    for i in $(seq 50); do kill -0 $px 2>/dev/null || break; sleep 0.1; done
    kill -9 $px 2>/dev/null; wait $px 2>/dev/null   # a fail-stop holds until killed (no GPU here)
    kill $srv 2>/dev/null; wait $srv 2>/dev/null; rm -rf "$d" "$u"
}
replayed() {   # <label> <recording> <data dir>: amd_state.py again, to tgreplay.py --guard serving the recording
    local l=$1 rec=$2 data=$3 d
    d=$(mktemp -d /tmp/tgn6.XXXXXX)
    BEAGLE_TINYGPU_DATA="$data" "$BEAGLE_PYTHON" "$TG_TESTS/replay/tgreplay.py" --listen "$d/tinygpu.sock" --rec "$rec" --mem "$W/rmem_$l" \
        --guard > "$W/$l.rp" 2>&1 &
    local rp=$!
    wait_for "$W/$l.rp" "tgreplay listening"
    client "$l" "$d"
    for i in $(seq 100); do kill -0 $rp 2>/dev/null || break; sleep 0.1; done
    kill -9 $rp 2>/dev/null; wait $rp 2>/dev/null; rm -rf "$d"
}
proxied px_7550 "$BEAGLE_TINYGPU_DATA"
check "tgproxy --guard: the RDNA 4 card's triggers and guard (1002:7550, gfx1201), the session clean, amd_state's prediction through it" \
    "grep -q 'session: the AMD card (1002:7550, gfx1201): AMD triggers and guard' $W/px_7550.px && grep -q 'session 1 ended: eof' $W/px_7550.px && grep -q 'prediction: a full boot without a mode1 reset' $W/px_7550.txt"
replayed rp_7550 "$W/rec_px_7550" "$BEAGLE_TINYGPU_DATA"
check "tgreplay --guard: the same card's triggers and guard, the replay PASS" \
    "grep -q 'session 1: the AMD card (1002:7550, gfx1201): AMD triggers and guard' $W/rp_7550.rp && grep -q '^replay session 1: PASS' $W/rp_7550.rp"
proxied px_none "$W/only_744c"
check "tgproxy --guard, no table for the card: fail-stop at its first config read" \
    "grep -q 'session 1 ended: failstop: no single captured discovery table for the AMD card 1002:7550' $W/px_none.px && grep -q FAIL-STOP $W/px_none.px"
replayed rp_none "$W/rec_px_7550" "$W/only_744c"
check "tgreplay --guard, no table for the card: the replay diverges" \
    "grep -q '^replay session 1: FAIL.*no single captured discovery table for the AMD card 7550' $W/rp_none.rp"

proxied card_mode1 "$BEAGLE_TINYGPU_DATA" mode1_client --card 1002:7550
check "tgproxy --card 1002:7550: the mode1 request with no dword 0 read before it is refused (fail-stop), the card never sees it" \
    "grep -q 'guard refused mmMP1_SMN_C2PMSG_75' $W/card_mode1.px && grep -q FAIL-STOP $W/card_mode1.px && ! grep -q 'smu debug msg 2' $W/card_mode1.dev"
proxied nocard_mode1 "$BEAGLE_TINYGPU_DATA" mode1_client
check "without --card the same session is forwarded, the card gets the mode1 request (the gap --card closes)" \
    "! grep -q FAIL-STOP $W/nocard_mode1.px && grep -q 'smu debug msg 2' $W/nocard_mode1.dev"
proxied card_other "$BEAGLE_TINYGPU_DATA" client --card 1002:744c
check "tgproxy --card 1002:744c on the 1002:7550 card: fail-stop at the dword 0 read" \
    "grep -q 'session 1 ended: failstop: the card reads 1002:7550, not --card 1002:744c' $W/card_other.px"
proxied card_7550 "$BEAGLE_TINYGPU_DATA" client --card 1002:7550
replayed rp_card "$W/rec_card_7550" "$BEAGLE_TINYGPU_DATA"
check "tgproxy --card 1002:7550 with amd_state.py: clean; tgreplay arms from the recording's --card, PASS" \
    "grep -q 'session 1 ended: eof' $W/card_7550.px && grep -q 'session 1: the AMD card (1002:7550, gfx1201): AMD triggers and guard' $W/rp_card.rp && grep -q '^replay session 1: PASS' $W/rp_card.rp"

# amd_boot_check with no table for the card
mkdir -p "$W/no_tables/discovery"
( AMD_PCI=1002:7550; BEAGLE_TINYGPU_DATA="$W/no_tables"; amd_boot_check ) > "$W/no_table.txt" 2>&1; echo $? > "$W/no_table.rc"
check "amd_boot_check, no table for the card: refused (exit 2), nothing run" \
    "[ \"\$(cat $W/no_table.rc)\" = 2 ] && grep -q 'no discovery table for 1002:7550' $W/no_table.txt"

# stop rule S8, on synthetic run outputs (hw_logstream_stop stubbed: no log stream here)
printf 'am usb4: AM_SOC initialized\nam usb4: AM_GMC initialized\nam usb4: AM_GFX initialized\n' > "$W/full.out"
printf 'am usb4: AM_GFX initialized\nam usb4: AM_SDMA initialized\n' > "$W/partial.out"
: > "$W/none.out"; : > "$W/ls.txt"
s8() { ( hw_logstream_stop() { return 0; }; LS="$W/ls.txt"; AMD_PREDICTED=$1; amd_hw_end "$W/$2.out" ) > "$W/s8_$1_$2.txt" 2>&1; echo $?; }
taken="$(amd_boot_taken "$W/full.out")/$(amd_boot_taken "$W/partial.out")/$(amd_boot_taken "$W/none.out")"
check "S8: amd_boot_taken reads a full boot, a partial one and none ($taken)" "[ '$taken' = full/partial/ ]"
s8_rcs="$(s8 partial full) $(s8 full partial) $(s8 full full) $(s8 partial partial) $(s8 full none) $(s8 '' full)"
check "S8: a full boot where a partial was predicted, and the reverse, stop; the predicted ones, no boot and no prediction pass ($s8_rcs)" \
    "[ '$s8_rcs' = '1 1 0 0 0 0' ] && grep -q 'a full boot where amd_state.py predicted a partial one (stop rule S8)' $W/s8_partial_full.txt"

left=$(ps -axo command= | awk '$0 ~ /fake_amd_device\.py|tgproxy\.py|tgreplay\.py/' | wc -l | tr -d ' ')
check "no fake, proxy or replay server is left running" "[ $left -eq 0 ]"
echo; [ $fails -eq 0 ] && echo "test_n6: PASS" || echo "test_n6: $fails FAILED"
[ $fails -eq 0 ]
