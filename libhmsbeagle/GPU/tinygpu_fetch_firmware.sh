#!/bin/bash
# Fetches the NVIDIA firmware TinyGPUFirmwareManifest.h lists (TODO.md plan step C4) into DIR/<subdir>/<name>, each file
# kept only if its sha256 is the manifest's; then BEAGLE_TINYGPU_FW=DIR points BEAGLE at it (TinyGPUFirmware.h). BEAGLE
# itself never downloads firmware (plan decision 5). The files are NVIDIA's, from linux-firmware at tinygrad's pin, under
# NVIDIA's licence (LICENCE.nvidia in that tree).
#     tinygpu_fetch_firmware.sh [--chip ga102|ad102|gb202] DIR
# TINYGPU_FW_BASE_URL replaces the linux-firmware URL (the offline test serves a local copy through file://).
set -uo pipefail
manifest="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/TinyGPUFirmwareManifest.h"
chip=""
[ "${1:-}" = "--chip" ] && { chip=${2:-}; shift 2; }
[ $# -eq 1 ] && [ -n "$1" ] || { echo "usage: $0 [--chip ga102|ad102|gb202] DIR" >&2; exit 2; }
dest=$1
base="${TINYGPU_FW_BASE_URL:-$(sed -nE 's/^constexpr const char\* kLinuxFirmware = "([^"]+)";.*/\1/p' "$manifest")}"
# each entry: {"chip", "role", "subdir", "name", "sha256", "url_md5"}
entries=$(sed -nE 's/^    \{"([a-z0-9]+)", "[a-z_]+", "([^"]+)", "([^"]+)", "([0-9a-f]{64})", "[0-9a-f]{32}"\},.*/\1 \2 \3 \4/p' "$manifest")
[ -n "$base" ] && [ -n "$entries" ] || { echo "cannot read $manifest" >&2; exit 2; }
[ -z "$chip" ] || grep -q "^$chip " <<< "$entries" || { echo "the manifest lists no firmware for $chip" >&2; exit 2; }
sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
fails=0; seen=" "
while read -r c subdir name want; do
    [ -z "$chip" ] || [ "$c" = "$chip" ] || continue
    case "$seen" in *" $subdir/$name "*) continue ;; esac   # gsp-570.144.bin serves every chip
    seen="$seen$subdir/$name "
    out="$dest/$subdir/$name"
    if [ -f "$out" ] && [ "$(sha "$out")" = "$want" ]; then echo "present: $out"; continue; fi
    mkdir -p "$(dirname "$out")" || { fails=$((fails + 1)); continue; }
    tmp="$out.part.$$"
    if curl -fsSL -o "$tmp" "$base/$subdir/$name" && [ "$(sha "$tmp")" = "$want" ]; then
        mv -f "$tmp" "$out" && echo "fetched: $out"
    else
        rm -f "$tmp"
        echo "FAILED: $subdir/$name was not fetched, or its sha256 is not $want" >&2
        fails=$((fails + 1))
    fi
done <<< "$entries"
[ $fails -eq 0 ] && echo "firmware complete in $dest: export BEAGLE_TINYGPU_FW=$dest"
[ $fails -eq 0 ]
