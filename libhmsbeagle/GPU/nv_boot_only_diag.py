#!/usr/bin/env python3
"""
nv_boot_only_diag.py -- quick, non-destructive diagnostic for a newly
attached eGPU card before running any of nv_real_kernel_probe.py's real
compile/dispatch tests against it.

Does EXACTLY what nv_real_kernel_probe.py's own preamble does (same
imports, same nv_init_helper monkey-patches, same Device["NV:0"] boot
path) and NOTHING else -- no kernel compile, no dispatch, no memory
allocation beyond whatever tinygrad's own NVDevice.__init__ does as part
of a normal boot. Prints the live-queried topology/arch info and exits.

Usage: python3 nv_boot_only_diag.py
"""
import sys, os

# Same sys.path insertion ORDER as nv_real_kernel_probe.py itself (tinygrad
# root first, then this script's own directory second, so the second
# insert(0,...) ends up in front) -- this specifically matters because a
# stale duplicate nv_init_helper.py sits directly in the tinygrad checkout
# root (leftover from earlier ad-hoc testing); getting the order backwards
# silently imports that stale copy instead of this directory's real one.
_TINYGRAD_PATH = os.environ.get("TINYGRAD_PATH", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "tinygrad"))
if not os.path.isdir(_TINYGRAD_PATH):
    print(f"FAIL: cannot find tinygrad at {_TINYGRAD_PATH}", file=sys.stderr)
    sys.exit(1)
sys.path.insert(0, _TINYGRAD_PATH)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("nv_boot_only_diag: importing nv_init_helper (safety monkey-patches only, "
      "main() is not invoked by import)...", flush=True)
import nv_init_helper  # noqa: F401 -- same as nv_real_kernel_probe.py's own preamble

from tinygrad.runtime.support.system import APLRemotePCIDevice
def _safe_reset(self):
    print("nv_boot_only_diag: PCIe FLR suppressed (macOS eGPU safety)", flush=True)
APLRemotePCIDevice.reset = _safe_reset

from tinygrad.helpers import DEV
DEV.value = "NV"
from tinygrad import Device

print("nv_boot_only_diag: booting Device['NV:0'] (this is the first real "
      "hardware touch -- includes the ~20s SEC2 sleep from nv_init_helper's "
      "own patch)...", flush=True)
dev = Device["NV:0"]
print(f"nv_boot_only_diag: BOOT OK -- {dev}", flush=True)
print(f"  arch            = {dev.arch}", flush=True)
print(f"  sm_version      = 0x{dev.sm_version:x}", flush=True)
print(f"  num_gpcs        = {dev.num_gpcs}", flush=True)
print(f"  num_tpc_per_gpc = {dev.num_tpc_per_gpc}", flush=True)
print(f"  num_sm_per_tpc  = {dev.num_sm_per_tpc}", flush=True)
print(f"  max_warps_per_sm= {dev.max_warps_per_sm}", flush=True)
print(f"  total SMs       = {dev.num_gpcs * dev.num_tpc_per_gpc * dev.num_sm_per_tpc}", flush=True)
print(f"  renderer        = {type(dev.renderer).__name__}", flush=True)
print("nv_boot_only_diag: done -- no kernel compiled, nothing dispatched.", flush=True)
