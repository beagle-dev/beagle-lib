"""Golden test for the SDMA half of TinyGPUHybridAMDDispatch.h (TODO.md plan step A1c). hcq1's own AMDCopyQueue, submitted
into a small ring through AMDQueueDesc.signal_doorbell, and HCQAllocator._copyin/_copyout on a stub device (its timeline,
and staging slots in one buffer) run random sequences; golden_amd_copy.cpp runs the same with the C++ encoder and flows.
The ring (random-filled first, so the tail's zero fill shows), put_value, one ordered list of the host waits, synchronizes
and wptr/HDP/doorbell writes, the staging bytes and the copied-out bytes must be identical, across ring wraps."""
import os, sys, ctypes, random, subprocess, types, functools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tgpaths
tgpaths.setup()
from tinygrad.runtime import ops_amd
from tinygrad.runtime.support.hcq import HCQBuffer, HCQAllocator, MMIOInterface
from tinygrad.runtime.support.amd import import_module

HERE, WORK = tgpaths.HERE, tgpaths.WORK
tgpaths.build_cpp(HERE / "golden_amd_copy.cpp", WORK / "golden_amd_copy")
SIG_VA, STAGING_VA = 0x7f_1000_0080, 0x7f_3000_0000

class Recorder:
    def __init__(self, events, what): self.events, self.what = events, what
    def __setitem__(self, i, v): self.events.append((self.what, v))

def run(seed, ring_bytes, put_value, slot, nslots, nops):
    rng = random.Random(seed)
    events = []
    ring = (ctypes.c_uint32 * (ring_bytes // 4))(*[rng.getrandbits(32) for _ in range(ring_bytes // 4)])
    ring_before = bytes(ring)
    rptr = (ctypes.c_uint64 * 1)(1 << 62)   # never makes _submit wait (the C++ side's room() is asked, and says yes)
    dev = types.SimpleNamespace(target=(11, 0, 0), device="AMD", is_am=lambda: True, is_usb=lambda: False, sdma=import_module("sdma", (6, 0, 0)))
    queue = ops_amd.AMDQueueDesc(ring=MMIOInterface(ctypes.addressof(ring), ring_bytes, fmt="I"),
                                 read_ptr=MMIOInterface(ctypes.addressof(rptr), 8, fmt="Q"), write_ptr=Recorder(events, "wptr"),
                                 doorbell=Recorder(events, "doorbell"), put_value=put_value)
    dev.sdma_queue = lambda idx: queue
    dev.iface = types.SimpleNamespace(dev_impl=types.SimpleNamespace(gmc=types.SimpleNamespace(flush_hdp=lambda: events.append(("hdp", 0)))))
    dev.timeline_value = rng.getrandbits(16) + 2
    def next_timeline():
        dev.timeline_value += 1
        return dev.timeline_value - 1
    dev.next_timeline = next_timeline
    dev.timeline_signal = types.SimpleNamespace(value_addr=SIG_VA, owner=None, is_timeline=True,
                                          wait=lambda v, timeout=None: events.append(("wait", v)))
    dev.synchronize = lambda: events.append(("sync", 0))
    dev.hw_copy_queue_t = functools.partial(ops_amd.AMDCopyQueue, dev, max_copy_size=0x40000000)
    smem = (ctypes.c_uint8 * (slot * nslots))(*[rng.getrandbits(8) for _ in range(slot * nslots)])
    staging_before = bytes(smem)
    alloc = types.SimpleNamespace(dev=dev, b_timeline=[0] * nslots, b_next=0,
                                  b=[HCQBuffer(STAGING_VA + i * slot, slot, view=MMIOInterface(ctypes.addressof(smem) + i * slot, slot)) for i in range(nslots)])
    lines = [f"SIG {SIG_VA}", f"RING {ring_bytes} {put_value}", f"STAGING {STAGING_VA} {slot} {nslots}", f"TL {dev.timeline_value}"]
    srcs, outs = bytearray(), bytearray()
    for _ in range(nops):
        op = rng.choice(["in", "in", "out", "raw"])
        n = rng.choice([1, 4, 100, slot - 1, slot, slot + 1, 3 * slot + 123])
        if op == "in":
            data = bytes(rng.getrandbits(8) for _ in range(n))
            dest = 0x7f_5000_0000 + (rng.getrandbits(24) << 4)
            HCQAllocator._copyin(alloc, HCQBuffer(dest, n), memoryview(data))
            srcs += data
            lines.append(f"CI {dest} {n}")
        elif op == "out":
            src = 0x7f_5000_0000 + (rng.getrandbits(24) << 4)
            out = memoryview(bytearray(n))
            HCQAllocator._copyout(alloc, out, HCQBuffer(src, n))
            outs += bytes(out)
            lines.append(f"CO {src} {n}")
        else:   # a queue of its own, with a small max_copy_size so copy() splits
            m = rng.choice([0x1000, 0x10000, 0x40000000])
            vw, vs = rng.getrandbits(31), rng.getrandbits(31)
            dest, src, n = rng.getrandbits(40), rng.getrandbits(40), rng.randint(1, 5 * m // 2) if m < 0x40000000 else n
            ops_amd.AMDCopyQueue(dev, max_copy_size=m).wait(dev.timeline_signal, vw).copy(HCQBuffer(dest, n), HCQBuffer(src, n), n) \
                .signal(dev.timeline_signal, vs).submit(dev)
            lines.append(f"R {m} {vw} {dest} {src} {n} {vs}")
    (WORK / "golden_amd_copy_ops.txt").write_text("\n".join(lines) + "\n")
    (WORK / "golden_amd_copy_ring_in.bin").write_bytes(ring_before)
    (WORK / "golden_amd_copy_staging_in.bin").write_bytes(staging_before)
    (WORK / "golden_amd_copy_srcs.bin").write_bytes(bytes(srcs))
    subprocess.run([str(WORK / "golden_amd_copy"), str(WORK)], check=True)
    got_ev = [(l.split()[0], int(l.split()[1])) for l in (WORK / "golden_amd_copy_out_events.txt").read_text().splitlines()]
    checks = {"ring": (WORK / "golden_amd_copy_out_ring.bin").read_bytes() == bytes(ring),
              "put_value": int((WORK / "golden_amd_copy_out_put.txt").read_text()) == queue.put_value,
              "events": got_ev == events,
              "staging": (WORK / "golden_amd_copy_out_staging.bin").read_bytes() == bytes(smem),
              "copied out": (WORK / "golden_amd_copy_out_outs.bin").read_bytes() == bytes(outs)}
    if not checks["events"]:
        i = next((i for i, (a, b) in enumerate(zip(events, got_ev)) if a != b), min(len(events), len(got_ev)))
        print(f"  events differ at {i}: ref {events[i:i+4]} c++ {got_ev[i:i+4]} (lengths {len(events)}, {len(got_ev)})")
    ok = all(checks.values())
    print(f"seed {seed}: {nops} ops, {queue.put_value - put_value} bytes into a {ring_bytes}-byte ring from {put_value} "
          f"({(queue.put_value // ring_bytes) - (put_value // ring_bytes)} wrap(s)), {nslots} staging slots of {slot} B, "
          f"{len(events)} events: {'IDENTICAL' if ok else 'MISMATCH ' + str([w for w, v in checks.items() if not v])}")
    return ok

# plus a first packet (a 6-dword wait), and a copyin's wait and copy (13 dwords), that would end exactly at the ring's end:
# tinygrad then zero-fills and wraps instead of filling the ring to its end
results = [run(1, 4096, 3900, 4096, 4, 40), run(2, 1024, 0, 512, 3, 80), run(3, 65536, 65000, 8192, 8, 60), run(4, 512, 500, 256, 2, 120),
           run(5, 1024, 1024 - 24, 256, 2, 30), run(6, 2048, 3 * 2048 - 52, 512, 2, 1), run(7, 4096, 4096 - 52, 512, 2, 1)]
print("A1c SDMA encoder and copy flows vs hcq1:", "all identical" if all(results) else "MISMATCH")
sys.exit(0 if all(results) else 1)
