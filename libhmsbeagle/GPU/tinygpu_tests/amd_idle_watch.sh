#!/bin/bash
# Passive, never touches the GPU: does the idle AMD card stay on the PCI bus? (STATUS.md R118: the RX 7900 XT has left it
# while idle after AM's clean finalize, three times.) Polls ioreg for vendor 0x1002 and keeps the kernel's USB-C port,
# Thunderbolt and PCI lines, info and debug included: those live only in the log's memory ring buffer, which a dropped
# link's retraining flood evicts within minutes. Each poll also makes the kernel log every PCIe device's live link status
# (serializeLinkStatus), so frequent polls may themselves keep the link busy.
#   amd_idle_watch.sh [minutes, default 60] [seconds between polls, default 60]
# Writes $BEAGLE_TINYGPU_DATA/runs/<stamp>_<computer>_idle_watch_{poll,kernel}.txt and, when the card leaves, _tb.txt
# (system_profiler's Thunderbolt view 30 s later). Exit codes: 0 the card stayed; 1 it left; 2 nothing was started.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
MIN=${1:-60}; EVERY=${2:-60}
amd() { ioreg -l -w0 2>/dev/null | grep -c '"vendor-id" = <02100000>'; }
[ "$(amd)" -gt 0 ] || { echo "no AMD card on the PCI bus; not running"; exit 2; }
P="$BEAGLE_TINYGPU_DATA/runs/$(date +%Y%m%d-%H%M%S)_${HW_HOST}_idle_watch"; mkdir -p "$BEAGLE_TINYGPU_DATA/runs"
log stream --level debug --style compact --predicate 'process == "kernel" AND (composedMessage CONTAINS "Port-USB-C@" OR composedMessage CONTAINS[c] "thunderbolt" OR composedMessage CONTAINS[c] "IOPCI" OR composedMessage CONTAINS[c] "apciec" OR composedMessage CONTAINS[c] "PCIe" OR composedMessage CONTAINS[c] "CIO")' > "${P}_kernel.txt" 2>&1 &
LSP=$!; trap 'kill $LSP 2>/dev/null' EXIT
for i in $(seq 50); do grep -q "^Filtering the log data" "${P}_kernel.txt" && break; sleep 0.1; done
grep -q "^Filtering the log data" "${P}_kernel.txt" || { echo "log stream did not attach; not running"; exit 2; }
rc=0; end=$(( $(date +%s) + MIN * 60 ))
while [ $(date +%s) -lt $end ]; do
    n=$(amd); echo "$(date +%T) AMD:$n" >> "${P}_poll.txt"
    if [ "$n" -eq 0 ]; then
        echo "GONE at $(date +%T)" >> "${P}_poll.txt"; rc=1
        sleep 30; system_profiler SPThunderboltDataType > "${P}_tb.txt" 2>&1; break
    fi
    sleep $EVERY
done
echo "watch ended $(date +%T): $([ $rc -eq 0 ] && echo "the card stayed" || grep GONE "${P}_poll.txt"); kernel lines: $(grep -c "^20" "${P}_kernel.txt")"
echo "${P}_poll.txt"
exit $rc
