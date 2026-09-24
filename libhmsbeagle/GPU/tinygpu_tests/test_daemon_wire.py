"""Offline check of nv_dispatch_daemon.py's length-prefixed wire protocol and
both launch_batch paths, with a fake device (no GPU, no TinyGPU.app)."""
import os, sys, json, struct, socket, threading
os.environ["BEAGLE_NV_PROFILE"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tgpaths
tgpaths.setup()
import nv_dispatch_daemon as d

calls = []

class FakeQueue:
    def wait(self, sig, val): calls.append(("wait", val)); return self
    def memory_barrier(self): calls.append(("barrier",)); return self
    def exec(self, prg, kernargs, gs, ls): calls.append(("exec", prg.name, kernargs, gs, ls)); return self
    def signal(self, sig, val): calls.append(("signal", val)); return self
    def submit(self, dev): calls.append(("submit",)); return self

class FakeAllocator:
    def __init__(self): self.mem, self.next = {}, 0x1000
    def alloc(self, size):
        b = type("B", (), {})(); b.va_addr = self.next; self.mem[self.next] = bytearray(size); self.next += 0x10000; return b
    def _copyin(self, buf, mv): self.mem[buf.va_addr][:len(mv)] = mv
    def _copyout(self, mv, buf): mv[:] = self.mem[buf.va_addr][:len(mv)]

class FakeDev:
    def __init__(self): self.allocator, self.timeline_signal, self.timeline_value = FakeAllocator(), object(), 5
    def hw_compute_queue_t(self): return FakeQueue()
    def next_timeline(self): self.timeline_value += 1; return self.timeline_value - 1
    def synchronize(self): calls.append(("sync",))

class FakePrg:
    def __init__(self, name): self.name = name
    def check_launch(self, gs, ls): calls.append(("check", self.name))
    def set_launch_dims(self, gs, ls): calls.append(("dims", self.name, gs, ls))
    def fill_kernargs(self, bufs, vals): return ("kargs", tuple(b.va_addr for b in bufs), vals)
    def __call__(self, *bufs, global_size, local_size, vals, wait): calls.append(("call", self.name, global_size, local_size, vals))

def client_send(sock, obj, payload=b""):
    body = json.dumps(obj).encode()
    sock.sendall(struct.pack("<I", len(body)) + body + payload)

def client_recv(sock):
    n = struct.unpack("<I", sock.recv(4, socket.MSG_WAITALL))[0]
    return json.loads(sock.recv(n, socket.MSG_WAITALL))

def run(chain):
    calls.clear()
    d._CHAIN_LAUNCHES = chain
    a, b = socket.socketpair()
    dm = d.Daemon(b); dm.dev = FakeDev(); dm.kernel_names = {"kA", "kB"}
    dm._get_program = lambda name, n: FakePrg(name)
    t = threading.Thread(target=dm.run); t.start()

    client_send(a, {"cmd": "alloc", "size": 64}); r = client_recv(a); assert r["ok"], r; addr = r["addr"]
    client_send(a, {"cmd": "h2d", "addr": addr, "size": 8}, b"ABCDEFGH"); assert client_recv(a)["ok"]
    launches = [{"kernel": "kA", "grid": [4, 1, 1], "block": [16, 16, 1], "ptrs": [addr], "ints": [3]},
                {"kernel": "kB", "grid": [2, 2, 1], "block": [32, 1, 1], "ptrs": [addr, addr], "ints": [1, 2]}]
    client_send(a, {"cmd": "launch_batch", "launches": launches}); r = client_recv(a); assert r == {"ok": True, "count": 2}, r
    client_send(a, {"cmd": "d2h", "addr": addr, "size": 8}); r = client_recv(a); assert r["ok"] and r["size"] == 8, r
    assert a.recv(8, socket.MSG_WAITALL) == b"ABCDEFGH"
    client_send(a, {"cmd": "launch_batch", "launches": [{"kernel": "nope", "grid": [1,1,1], "block": [1,1,1], "ptrs": [], "ints": []}]})
    dm.kernel_names = {"kA", "kB"}; dm._get_program = lambda name, n: (_ for _ in ()).throw(RuntimeError("unknown kernel")) if name == "nope" else FakePrg(name)
    r = client_recv(a); assert not r["ok"] and "unknown kernel" in r["error"], r
    client_send(a, {"cmd": "sync"}); assert client_recv(a)["ok"]
    client_send(a, {"cmd": "fini"}); assert client_recv(a)["ok"]
    t.join(5); assert not t.is_alive()
    return [c for c in calls if c[0] != "sync"]

chained = run(True)
expect = [("wait", 4), ("barrier",), ("check", "kA"), ("dims", "kA", (4, 1, 1), (16, 16, 1)),
          ("exec", "kA", ("kargs", (0x1000,), (3,)), (4, 1, 1), (16, 16, 1)),
          ("check", "kB"), ("dims", "kB", (2, 2, 1), (32, 1, 1)),
          ("exec", "kB", ("kargs", (0x1000, 0x1000), (1, 2)), (2, 2, 1), (32, 1, 1)),
          ("signal", 5), ("submit",), ("wait", 5), ("barrier",)]
assert chained == expect, chained
print("chained path: OK (one wait/barrier, 2 execs, one signal+submit; failed batch submits nothing)")

unchained = run(False)
assert unchained == [("call", "kA", (4, 1, 1), (16, 16, 1), (3,)), ("call", "kB", (2, 2, 1), (32, 1, 1), (1, 2))], unchained
print("per-launch path: OK")
