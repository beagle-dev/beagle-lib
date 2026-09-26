"""The daemon entry point of the V1 harness (TODO.md plan step V1), for the plugin's BEAGLE_NV_DISPATCH_DAEMON: runs the
real nv_dispatch_daemon.main(), unchanged, with the harness's additions chosen by environment:
  BEAGLE_TG_OFFLINE=1   tgharness_py.py's pre-connect patches, for a fake device or the replay server (never on hardware)
  BEAGLE_TG_RECORD=1    record_shim.py: markers and a side log from inside tinygrad's boot, no change in what it does
Both are applied after nv_init_helper's patches, as the daemon applies those at its boot, so they see what the daemon runs.
    <python> tgdaemon.py <cmd_sock_fd> [<tinygpu_sock_fd>]   (as nv_dispatch_daemon.py)"""
import os, sys, pathlib
HERE = pathlib.Path(__file__).resolve().parent
GPU_DIR = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(GPU_DIR))
import nv_dispatch_daemon   # puts the pinned tinygrad on sys.path
import nv_init_helper       # noqa: F401 - its patches, which the daemon's boot would apply (cmd_boot's _apply_boot_safety_patches)
if os.environ.get("BEAGLE_TG_OFFLINE") == "1":
    import tgharness_py
    tgharness_py.install()
    if os.environ.get("BEAGLE_TG_DAEMON_PIDFILE"):   # the offline scripts end a daemon that holds a fake's connection (only theirs)
        pathlib.Path(os.environ["BEAGLE_TG_DAEMON_PIDFILE"]).write_text(str(os.getpid()))
if os.environ.get("BEAGLE_TG_RECORD") == "1":
    import record_shim
    record_shim.install()
nv_dispatch_daemon.main()
