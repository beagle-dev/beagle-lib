"""Offline tests for plan step C10's daemon half (the crash guard itself runs end to end in test_c10.sh): at level flcn_hw the
C++ side sets the state page's keeper word once its guard is ready, then sends cmd_release. The daemon releases only with the
word set, at flcn_hw, after the RM export, with the state page and the C++ timeline: it replies with its pid and leaves its run
loop without a word to the GPU. A refused release leaves it in the loop, and at the EOF that follows the word decides: set (the
plugin died before it could set it back), the daemon exits without a word to the GPU; clear, it tears down as before. An EOF
with the word set and no release (the plugin died between the two) exits without a word too, and does not hold: the guard
decides. No GPU, no TinyGPU socket.
    python test_c10.py"""
import os, sys, types
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tgpaths
tgpaths.setup()
import test_p3 as p3
import test_c5 as c5
import test_c7 as c7
import nv_dispatch_daemon as d
from tinygrad import Device

def rig(level="flcn_hw", keeper=0, signal=5):
    """A daemon at level after cmd_rm_export, with the state page (seq 9), the C++ timeline (signal; None: not sent) and the
    keeper word as the C++ side left it."""
    a, dm, calls, seqs = c7.fini_rig(seq=9, signal=signal)
    dm.rm_level = level
    dm.dev.iface.dev_impl.gsp.stat_q = types.SimpleNamespace()   # init_hw's, which the C++ side ran
    dm._state[4] = keeper
    return a, dm, calls, seqs

def test_release():
    real, Device._opened_devices = Device._opened_devices, set()
    try:
        # refused: the word clear, another level, or no timeline yet. The daemon stays in its loop; at the EOF the word decides
        for kw, why, teardown in ((dict(), "keeper word 0", True), (dict(level="rm", keeper=d._KEEPER_GUARD), "level rm", False),
                                  (dict(keeper=d._KEEPER_GUARD, signal=None), "timeline False", False)):
            a, dm, calls, seqs = rig(**kw)
            ((r, _),), held = c5.daemon_reply(dm, a, [{"cmd": "release"}])
            assert not r["ok"] and why in r["error"] and not dm.released and not held, (kw, r)
            assert (calls, seqs) == ((["finalize"], [9]) if teardown else ([], [])), (kw, calls, seqs)
        # released: the reply, then out of the loop without a word to the GPU (no EOF decision, no unload, no hold)
        a, dm, calls, seqs = rig(keeper=d._KEEPER_GUARD)
        ((r, _),), held = c5.daemon_reply(dm, a, [{"cmd": "release"}])
        assert r["ok"] and r["pid"] == os.getpid() and dm.released and dm.dev is None and not held, r
        assert calls == [] and seqs == [], (calls, seqs)
        # the plugin died with the word set, before the release: exit without a word, and no hold
        a, dm, calls, seqs = rig(keeper=d._KEEPER_GUARD)
        held, log = p3.run(a, dm)
        assert not held and calls == [] and seqs == [] and "the keeper word names the C++ side's guard" in log, (calls, log)
    finally: Device._opened_devices = real
    print("release: refused without the keeper word, at another level or before the timeline (the daemon stays; at the EOF "
          "the word decides: set, it exits without a word; clear, it tears down); with it, the reply and the exit, nothing sent; "
          "an EOF with the word set exits without a word and without holding")

if __name__ == "__main__":
    test_release()
    print("C10 daemon: all passed")
