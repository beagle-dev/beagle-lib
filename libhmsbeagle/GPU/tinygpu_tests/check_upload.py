"""After a C++ runtime fake run (run_fake_runtime.sh): the plugin loaded the cubin for the state count and architecture the
harness ran (not just the one it reports), and the program image it uploaded into the fake VRAM equals what compile_all's
path would have uploaded, the ptxas cubin of that module's PTX (compile_ptx, cached by the PTX's sha256) relocated at the
same address by the real BeagleNVProgram (TODO.md plan step C1).
    python check_upload.py <plugin output> <fake memory dir> <padded state count> <arch>"""
import os, sys, re, json, types, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tgpaths
tgpaths.setup()
import nv_dispatch_daemon as d
import nv_compile_helper as nch
from tinygrad.device import TinyELF, Target

out, mem, states, arch = open(sys.argv[1], errors="replace").read(), sys.argv[2], sys.argv[3], sys.argv[4]
if (got := re.search(r"C\+\+ runtime: embedded cubin SP_(\d+) (sm_\d+)", out).groups()) != (states, arch):
    print(f"the plugin loaded SP_{got[0]} {got[1]}, not SP_{states} {arch}"); sys.exit(1)
size, lib_va = (int(x, 0) for x in re.search(r"C\+\+ runtime: \d+ kernels loaded \(image (\d+) bytes at (0x[0-9a-f]+)", out).groups())
images = []   # the fake device's allocator puts the image at the plugin's lib_va and keeps the relocated bytes
dev = types.SimpleNamespace(iface=types.SimpleNamespace(compute_class=d.ops_nv.nv_gpu.ADA_COMPUTE_A), renderer=object(),
                            allocator=types.SimpleNamespace(alloc=lambda size, spec=None: types.SimpleNamespace(va_addr=lib_va, size=size),
                                                            _copyin=lambda buf, mv: images.append(bytes(mv)), free=lambda *a: None),
                            slm_per_thread=0x10000, shared_mem_window=0x729400000000, local_mem_window=0x729300000000, sass_version=0x89,
                            prof_prg_counter=itertools.count(), _ensure_has_local_memory=lambda required: None, synchronize=lambda: None)
elf = tgpaths.cubin(f"SP_{states}", arch)
d.BeagleNVProgram(dev, TinyELF(lib=elf, name=sorted(nch.extract_all_metadata(elf)[1])[0], target=Target(), signature=()))
vram_va = json.load(open(f"{mem}/vram.json"))["va"]
with open(f"{mem}/vram.bin", "rb") as f:
    f.seek(lib_va - vram_va)
    uploaded = f.read(size)
ok = len(images) == 1 and images[0] == uploaded
print(f"uploaded image ({size} bytes at {lib_va:#x}) {'==' if ok else '!='} BeagleNVProgram's relocation of the compile_ptx cubin "
      f"for SP_{states} {arch} ({len(images[0]) if images else 0} bytes)")
sys.exit(0 if ok else 1)
