#!/usr/bin/env python3
"""
nv_reference_test.py — does tinygrad's own, real NVDevice/NVProgram/
HCQProgram.__call__ stack (not BEAGLE's hand-rolled GPFIFO/QMD builder in
GPUInterfaceTinyGPUHybrid.cpp) boot, compile, dispatch, and read back
correctly on this exact eGPU/TinyGPU.app/USB4 setup, end to end?

This is the NV equivalent of the AMD investigation's own reference-validation
step (STATUS.md AMD §8: `DEV.value = "AMD"; (Tensor([1,2,3])+1).tolist()`,
zero new code, confirmed the AMD hardware/transport/driver stack was sound
and the bug had been entirely in this project's own hand-built PM4 stream).
STATUS.md §73 records the same idea for NV, carried over after that AMD
investigation resolved successfully via exactly this kind of pivot -- but
unlike AMD, this backend's own GSP/RM boot sequence needs three known,
already-applied macOS/TinyGPU-socket safety patches (see nv_init_helper.py's
module-level patches, reused unmodified below) before *any* NVDev-based boot
succeeds here, so this can't quite be "zero new code" the way AMD's was.

**Two distinct kernel-dispatch bugs are already isolated and documented**
(TODO.md Phase 32+, STATUS.md §65-72) as living specifically in
GPUInterfaceTinyGPUHybrid.cpp's own hand-rolled QMD/command-queue
construction -- three individually-clean SASS/PTX-level probes have failed
to reproduce them, the exact "defect must depend on something these probes
structurally cannot reproduce" shape that motivated AMD's own pivot away
from hand-rolled dispatch. This script tests the premise of that pivot for
NV *before* committing to building the full daemon+RPC architecture
(nv_dispatch_daemon.py-equivalent) §73 describes: if tinygrad's own driver
code reproduces the same wrong answer here, the bug is upstream/hardware/
compiler-level and the pivot buys nothing; if it comes back correct, that's
strong evidence for building the daemon architecture for real.

Safety note (read before running): this project has hit a real macOS kernel
panic once already from an NV eGPU PCIe FLR (function-level reset) issued
while the GPU's WPR2 region was already up from a prior boot -- see
STATUS.md's InheritedFDPCIDevice.reset() writeup and TODO.md Phase 1b. That
specific code path (NVDev._early_ip_init(), nvdev.py) is reused unchanged by
tinygrad's own NVDevice() (via PCIIface), so this script additionally patches
APLRemotePCIDevice.reset() to a safe no-op globally, the same fix
nv_init_helper.py already applies to its own InheritedFDPCIDevice subclass --
necessary here because NVDevice() creates a *plain* APLRemotePCIDevice
internally, not that subclass. Power-cycle the eGPU (Thunderbolt unplug/
replug) before running this, as usual, so WPR2 starts down and this path is
never even exercised. If this script appears to hang: **do not Ctrl-C it**.
Unlike nv_init_helper.py's own resident-subprocess SIGINT hazard (fixed
separately, TODO.md Phase 1b -- not applicable here, this is a single
foreground process), tinygrad's own HCQSignal.wait() has a built-in 30s
poll-loop timeout (HCQDEV_WAIT_TIMEOUT_MS) and raises a clean RuntimeError
rather than blocking forever, so just let it run to completion.

    python3 nv_reference_test.py
"""
import sys, os, pathlib

_TINYGRAD_PATH = os.environ.get("TINYGRAD_PATH", str(pathlib.Path.home() / "Dropbox/Projects/tinygrad"))
sys.path.insert(0, _TINYGRAD_PATH)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def log(msg):
    print(f"[nv_reference_test] {msg}", file=sys.stderr, flush=True)


def main():
    os.makedirs(os.path.expanduser("~/Library/Logs"), exist_ok=True)
    fd = os.open(os.path.expanduser("~/Library/Logs/nv_reference_test.log"),
                 os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_SYNC, 0o644)
    sys.stderr = os.fdopen(fd, 'w', buffering=1)

    log("starting")

    # Reuse nv_init_helper.py's module-level GSP/RM boot safety patches
    # (WPR2-reset-loop suppression, palloc zero-size limit, GC6 BSI sleep)
    # unmodified -- importing it (without calling main()) applies them and
    # nothing else; see its own header comment for why each is needed.
    import nv_init_helper  # noqa: F401  (imported for its patch side effects)
    log("nv_init_helper patches applied")

    # nv_init_helper.py only overrides .reset() on its own InheritedFDPCIDevice
    # subclass, used for BEAGLE's C++-inherited-FD scheme. tinygrad's own
    # NVDevice() (via PCIIface) creates a plain APLRemotePCIDevice instead --
    # patch the base class the same way, for the same PCIe-FLR-is-fatal
    # reason (see this file's own header comment and
    # InheritedFDPCIDevice.reset() in nv_init_helper.py).
    from tinygrad.runtime.support.system import APLRemotePCIDevice
    def _safe_reset(self):
        log("PCIe FLR suppressed (macOS eGPU safety) -- see nv_init_helper.py InheritedFDPCIDevice.reset()")
    APLRemotePCIDevice.reset = _safe_reset
    log("APLRemotePCIDevice.reset() patched to a no-op")

    from tinygrad.helpers import DEV
    DEV.value = "NV"
    from tinygrad import Tensor, Device
    log("booting NVDevice via tinygrad's real ops_nv.py stack...")
    dev = Device["NV:0"]
    log(f"booted -- {dev}, arch={dev.arch}")

    log("dispatching (Tensor([1.0, 2.0, 3.0]) + 1).tolist() via real HCQProgram.__call__...")
    result = (Tensor([1.0, 2.0, 3.0]) + 1).tolist()
    log(f"result: {result}")

    expected = [2.0, 3.0, 4.0]
    ok = result == expected
    print(f"RESULT: {result}", file=sys.stdout, flush=True)
    print(f"EXPECTED: {expected}", file=sys.stdout, flush=True)
    print(f"RESULT: {'PASS' if ok else 'FAIL'}", file=sys.stdout, flush=True)
    log("PASS -- tinygrad's own NV driver stack works end to end on this hardware" if ok
        else "FAIL -- mismatch, see RESULT/EXPECTED above")
    log("exiting cleanly")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc(file=sys.stderr)
        print("RESULT: FAIL (exception, see log)", file=sys.stdout, flush=True)
        sys.exit(1)
