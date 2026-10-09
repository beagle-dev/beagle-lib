#!/bin/bash
# The harness's wrapper around beagle-tinygpu-guard (TODO.md plan step C10), which BEAGLE_NV_GUARD names in offline runs
# (run_fake_device.sh): it records the guard's pid in BEAGLE_TG_GUARD_PIDFILE, so the harness can wait for it or end it if it
# holds, then becomes the guard (exec keeps the pid and fd 3, the plugin's socketpair).
echo $$ > "${BEAGLE_TG_GUARD_PIDFILE:?}"
exec "${BEAGLE_TG_GUARD_BIN:?}"
