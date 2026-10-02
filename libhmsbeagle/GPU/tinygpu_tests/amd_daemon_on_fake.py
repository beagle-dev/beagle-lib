"""The oracle's amd_dispatch_daemon.py as it is, booting fake_amd_device.py's card (TODO.md plan step A2a), spawned by
amd_daemon_session.py with the daemon's arguments (the plugin spawned it until plan step A2l). On macOS tinygrad finds the GPU through IOKit
(System.pci_scan_bus, system.py:61-72), and offline that must not see this Mac's own PCI devices, so the scan returns the
fake card's id instead. Every other line is the daemon's. tgpaths.setup() is not called: its atexit hook would drop
tinygrad's opened devices, and with them the AMDev.fini that the daemon's exit runs. The daemon's log goes to
$TINYGPU_TEST_WORK/amd_dispatch_daemon.log, not ~/Library/Logs, where the hardware runs read it.
AMD_REG_NAMES_OUT=<file>: at exit, write the AMDev register names tinygrad's code used (each AMRegister reached as an
attribute or through AMDev.reg) and the ones it asked for that the card has none of (hasattr), as JSON. These are the boot
tables' registers (make_tinygpu_amd_boot_tables.py).
    python3 amd_daemon_on_fake.py <cmd_sock_fd> <tgpu_fd>"""
import os, sys, json, atexit
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tgpaths
_used, _absent = set(), set()
if os.environ.get("AMD_REG_NAMES_OUT"):   # registered before tinygrad is imported: it runs after tinygrad's atexit hook (LIFO),
    atexit.register(lambda: json.dump({"used": sorted(_used), "absent": sorted(_absent)},   # so the fini's registers are in
                                      open(os.environ["AMD_REG_NAMES_OUT"], "w")))
for p in (str(tgpaths.ORACLE), tgpaths.TINYGRAD_PATH):
    if p not in sys.path: sys.path.insert(0, p)
tgpaths.block_network()   # the firmware must come from tinygrad's cache
# the daemon logs to ~/Library/Logs/amd_dispatch_daemon.log, which the hardware runs read: offline runs log under the work dir
_expanduser, _logs = os.path.expanduser, os.environ.get("TINYGPU_TEST_WORK", str(tgpaths.WORK))
os.path.expanduser = lambda p: _logs if p == "~/Library/Logs" else (os.path.join(_logs, "amd_dispatch_daemon.log")
                                                                     if p == "~/Library/Logs/amd_dispatch_daemon.log" else _expanduser(p))
from tinygrad.runtime.support import system
system.System.pci_scan_bus = lambda vendor, devices, base_class=None: ["1002:744c"] if vendor == 0x1002 else []
if os.environ.get("AMD_REG_NAMES_OUT"):
    from tinygrad.runtime.support.am.amdev import AMDev, AMRegister
    _get, _reg = AMDev.__getattribute__, AMDev.reg
    def _logged_get(self, name):
        try: v = _get(self, name)
        except AttributeError:
            if name.startswith(("reg", "mm")): _absent.add(name)
            raise
        if isinstance(v, AMRegister): _used.add(name)
        return v
    def _logged_reg(self, name):
        _used.add(name)
        return _reg(self, name)
    AMDev.__getattribute__, AMDev.reg = _logged_get, _logged_reg
sys.argv = [str(tgpaths.ORACLE / "amd_dispatch_daemon.py")] + sys.argv[1:]
import amd_dispatch_daemon
amd_dispatch_daemon.main()
