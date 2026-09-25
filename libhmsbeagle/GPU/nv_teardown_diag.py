#!/usr/bin/env python3
"""
nv_teardown_diag.py -- TODO.md plan step P2's hardware check. Boots the eGPU
exactly as nv_boot_only_diag.py does (same imports, same nv_init_helper
patches, same Device["NV:0"] boot path), then tears it down: the GSP unload
RPC and suspend wait, then NVIDIA's driver-unload teardown (FWSEC-SB, Booter
Unload) from nv_init_helper, which is on unless BEAGLE_NV_TEARDOWN=0. Prints every
value the teardown read. Exit status 0 when WPR2 ended down and the teardown
succeeded, i.e. the next boot of this script (or of BEAGLE) needs no power
cycle; 1 when a power cycle is needed.

If the GSP does not confirm its unload (after the run, or after a boot that
failed once GSP-RM had started), this process keeps its TinyGPU.app
connection open and waits: unplug the eGPU first, then kill it.

Usage: python3 nv_teardown_diag.py
"""
import sys, os, signal, time

if os.environ.get("BEAGLE_NV_TEARDOWN", "1") == "0":
    print("nv_teardown_diag: refusing to run with BEAGLE_NV_TEARDOWN=0 (the teardown is what it checks)", file=sys.stderr)
    sys.exit(2)
signal.signal(signal.SIGINT, signal.SIG_IGN)   # never interrupt a boot or a teardown
signal.signal(signal.SIGHUP, signal.SIG_IGN)

# Same sys.path order as nv_boot_only_diag.py: tinygrad first, then this directory in front of it.
_TINYGRAD_PATH = os.environ.get("TINYGRAD_PATH", os.path.expanduser("~/Dropbox/Projects/tinygrad-hcq1"))
if not os.path.isdir(_TINYGRAD_PATH):
    print(f"FAIL: cannot find tinygrad at {_TINYGRAD_PATH}", file=sys.stderr)
    sys.exit(1)
sys.path.insert(0, _TINYGRAD_PATH)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import nv_init_helper  # noqa: F401 -- boot patches, P1 checks, P2 teardown

from tinygrad.runtime.support.system import APLRemotePCIDevice
def _safe_reset(self):
    print("nv_teardown_diag: PCIe FLR suppressed (macOS eGPU safety)", flush=True)
APLRemotePCIDevice.reset = _safe_reset

from tinygrad.helpers import DEV
DEV.value = "NV"
from tinygrad import Device

def hold(why):
    print(f"nv_teardown_diag: HOLDING the TinyGPU.app connection: {why}. Unplug the eGPU first, then kill {os.getpid()}.", flush=True)
    while True: time.sleep(3600)

print("nv_teardown_diag: booting Device['NV:0'] (includes nv_init_helper's ~20 s SEC2 sleep)...", flush=True)
try:
    dev = Device["NV:0"]
except Exception:
    import traceback
    traceback.print_exc()
    fini = nv_init_helper.unload_after_failed_boot()   # None: GSP-RM never started, so closing is safe
    if fini is None:
        print("nv_teardown_diag: BOOT FAILED before GSP-RM started; power-cycle the eGPU before the next boot.", flush=True)
        sys.exit(1)
    print(f"nv_teardown_diag: BOOT FAILED after GSP-RM started; unload: {fini}", flush=True)
    if not fini.get("unload_ok"): hold("the GSP did not confirm its unload after the failed boot")
    print("nv_teardown_diag: the GSP confirmed its unload; power-cycle the eGPU before the next boot.", flush=True)
    sys.exit(1)
flcn = dev.iface.dev_impl.flcn
print(f"nv_teardown_diag: BOOT OK -- {dev}, arch={dev.arch}", flush=True)
print(f"  FWSEC-SB image at VRAM 0x{flcn.beagle_sb_image_paddr:x}, Booter Unload at VRAM 0x{flcn.beagle_unload_image_paddr:x}", flush=True)

print("nv_teardown_diag: tearing down (synchronize, GSP unload, suspend wait, FWSEC-SB, Booter Unload)...", flush=True)
try:
    dev.finalize()
except Exception:   # e.g. the unload RPC timed out: unload_ok stays false below, so this process holds the connection
    import traceback
    traceback.print_exc()
for name in [n for n in Device._opened_devices if n.split(":")[0] == "NV"]:
    Device._opened_devices.discard(name)   # atexit must not finalize a second time
diag = getattr(dev.iface.dev_impl, "beagle_fini", {"unload_ok": False})
for key in ("unload_ok", "mailbox0", "riscv_cpuctl"):
    val = diag.get(key)
    print(f"  {key:14s} = {f'0x{val:08x}' if isinstance(val, int) and not isinstance(val, bool) else val}", flush=True)
for key, val in diag.get("teardown", {}).items():
    print(f"  teardown.{key:14s} = {f'0x{val:x}' if isinstance(val, int) and not isinstance(val, bool) else val}", flush=True)
print(f"  wpr2_lo/wpr2_hi = 0x{diag.get('wpr2_lo', 0):08x} / 0x{diag.get('wpr2_hi', 0):08x}, "
      f"wpr2_down={diag.get('wpr2_down')}, teardown_ok={diag.get('teardown_ok')}", flush=True)

if not diag.get("unload_ok"): hold("the GSP did not confirm its unload")
if diag.get("teardown_ok"):
    print("nv_teardown_diag: WPR2 is down and the teardown succeeded -- the next boot needs no power cycle.", flush=True)
    sys.exit(0)
print("nv_teardown_diag: the teardown did not succeed -- power-cycle the eGPU before the next boot.", flush=True)
sys.exit(1)
