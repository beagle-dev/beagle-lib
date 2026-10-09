"""After a fake-device run (run_fake_device.sh, with fake_nv_device.py's FAKE_COPY_LOG): the plugin loaded the cubin for the
state count and architecture the harness ran (not just the one it reports), and the program image it uploaded (the copy-engine
copies to its lib_va) equals the ptxas cubin of that module's PTX (compile_ptx, cached by the PTX's sha256) relocated at the
same address by the real BeagleNVProgram of the oracle (TODO.md plan step C1). With several instances (plan step P5), a comma
list of state counts checks every instance's cubin and image, in load order.
    python check_upload.py <plugin output> <copy log> <padded state count>[,<padded state count>...] <arch>"""
import os, sys, re, types, struct, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tgpaths
tgpaths.setup()
import nv_dispatch_daemon as d
import nv_compile_helper as nch
from tinygrad.device import TinyELF, Target

out, log, states, arch = open(sys.argv[1], errors="replace").read(), sys.argv[2], sys.argv[3], sys.argv[4]
want = [(s, arch) for s in states.split(",")]
if (got := re.findall(r"C\+\+ runtime: embedded cubin SP_(\d+) (sm_\d+)", out)) != want:
    print(f"the plugin loaded {got}, not {want}"); sys.exit(1)
loads = re.findall(r"C\+\+ runtime: \d+ kernels loaded \(image (\d+) bytes at (0x[0-9a-f]+)", out)
if len(loads) != len(want): print(f"{len(loads)} program loads for {len(want)} cubins"); sys.exit(1)
bw = arch == "sm_120"   # a GB205 (plan step B1): Blackwell's compute class and sass version, as fake_nv_device.py's FAKE_NV_CHIP=gb205
copies = []   # fake_nv_device.py's FAKE_COPY_LOG: (destination VA, bytes), in order
with open(log, "rb") as f:
    while (hdr := f.read(16)):
        dst, n = struct.unpack("<QQ", hdr)
        copies.append((dst, f.read(n)))
def uploaded_at(va, size):   # the bytes the copy engine wrote to [va, va + size), the last copy winning (None where none did)
    buf, seen = bytearray(size), bytearray(size)
    for dst, data in copies:
        lo, hi = max(va, dst), min(va + size, dst + len(data))
        if lo < hi: buf[lo - va:hi - va] = data[lo - dst:hi - dst]; seen[lo - va:hi - va] = b"\x01" * (hi - lo)
    return bytes(buf) if all(seen) else None
ok_all = True
for (states, _), (size, lib_va) in zip(want, loads):
    size, lib_va = int(size), int(lib_va, 0)
    images = []   # the fake device's allocator puts the image at the plugin's lib_va and keeps the relocated bytes
    dev = types.SimpleNamespace(iface=types.SimpleNamespace(compute_class=d.ops_nv.nv_gpu.BLACKWELL_COMPUTE_B if bw else d.ops_nv.nv_gpu.ADA_COMPUTE_A), renderer=object(),
                                allocator=types.SimpleNamespace(alloc=lambda size, spec=None: types.SimpleNamespace(va_addr=lib_va, size=size),
                                                                _copyin=lambda buf, mv: images.append(bytes(mv)), free=lambda *a: None),
                                slm_per_thread=0x10000, shared_mem_window=0x729400000000, local_mem_window=0x729300000000, sass_version=0xa4 if bw else 0x89,
                                prof_prg_counter=itertools.count(), _ensure_has_local_memory=lambda required: None, synchronize=lambda: None)
    elf = tgpaths.cubin(f"SP_{states}", arch)
    d.BeagleNVProgram(dev, TinyELF(lib=elf, name=sorted(nch.extract_all_metadata(elf)[1])[0], target=Target(), signature=()))
    uploaded = uploaded_at(lib_va, size)
    ok = len(images) == 1 and uploaded is not None and images[0] == uploaded
    print(f"uploaded image ({size} bytes at {lib_va:#x}) {'==' if ok else '!='} BeagleNVProgram's relocation of the compile_ptx cubin "
          f"for SP_{states} {arch} ({len(images[0]) if images else 0} bytes)")
    ok_all &= ok
sys.exit(0 if ok_all else 1)
