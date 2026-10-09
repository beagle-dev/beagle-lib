"""Which GPU TinyGPU.app serves now: one read-only PCI config read (offset 0: vendor and device ID), as BEAGLE's own probe
does (GPUInterfaceTinyGPU.cpp Initialize), under the same nv_usb4.lock. Starts TinyGPU.app's server as the plugin
does (TinyGPUTransport.h spawn_server) if nothing listens. Prints the vendor:device and exits; nothing else is sent.
TinyGPU.app (c0d024f9) serves only the first eGPU whose driver registered (STATUS.md R61), so the AMD hardware scripts
check this first (amd_hw_begin in env.sh)."""
import fcntl, os, socket, struct, subprocess, sys, time

tmp = os.environ.get("TMPDIR", "/tmp").rstrip("/")
sock_path, lock_path = f"{tmp}/tinygpu.sock", f"{tmp}/nv_usb4.lock"
REQ, RESP, CFG_READ = struct.Struct("<BIIQQQ"), struct.Struct("<BQQ"), 3

lock = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError: sys.exit("nv_usb4.lock is held: another process has the eGPU; not probing")

def connect():
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(sock_path)
    return s

try: s = connect()
except OSError:
    subprocess.Popen(["/Applications/TinyGPU.app/Contents/MacOS/TinyGPU", "server", sock_path], stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    for _ in range(100):
        time.sleep(0.1)
        try: s = connect(); break
        except OSError: pass
    else: sys.exit("TinyGPU.app's server did not start")
s.sendall(REQ.pack(CFG_READ, 0, 0, 0, 4, 0))
resp = b""
while len(resp) < RESP.size:
    chunk = s.recv(RESP.size - len(resp))
    if not chunk: sys.exit("connection closed")
    resp += chunk
status, r0, r1 = RESP.unpack(resp)
if status != 0:
    msg = s.recv(r0).decode(errors="replace") if 0 < r0 < 65536 else ""
    sys.exit(f"CFG_READ failed: {msg}")
s.close()
vendor, device = r0 & 0xffff, (r0 >> 16) & 0xffff
print(f"TinyGPU.app serves {vendor:04x}:{device:04x} ({ {0x10de: 'NVIDIA', 0x1002: 'AMD'}.get(vendor, 'unknown') })")
