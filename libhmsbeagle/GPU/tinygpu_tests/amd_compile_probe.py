"""Device-free AMD compile probe (TODO.md plan step A1): compiles BEAGLE's OpenCL-source kernel variants for gfx1100
with tinygrad's compile_hip on the native macOS libamd_comgr, in memory only, and reports code-object size,
compile time, determinism, e_flags, kernel-descriptor count and relocations (through amd_compile_helper, as the AMD
daemon parses them).
    python amd_compile_probe.py [KERNELS_STRING_SP_4 ...]     # default: SP_4 and SP_64"""
import os, sys, time, hashlib, zlib, struct, ctypes, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tgpaths
tgpaths.setup()
from tinygrad.runtime.support.compiler_amd import compile_hip, comgr
from tinygrad.runtime.support.elf import elf_loader
import amd_compile_helper as ach

comgr.amd_comgr_get_version(ctypes.byref(ma := ctypes.c_uint64()), ctypes.byref(mi := ctypes.c_uint64()))
print("comgr version", ma.value, mi.value)
hdr = (tgpaths.GPU_DIR / "kernels/BeagleOpenCL_kernels.h").read_text().split("\n")

def variant(name):
    i = next(k for k, l in enumerate(hdr) if l.startswith(f"#define {name} \""))
    out = []
    for l in hdr[i + 1:]:
        if l == '"': break
        assert l.endswith("\\n\\"), l[-10:]
        out.append(l[:-3].replace('\\"', '"').replace('\\\\', '\\'))
    return "\n".join(out) + "\n"

for name in sys.argv[1:] or ["KERNELS_STRING_SP_4", "KERNELS_STRING_SP_64"]:
    src = "#define FW_TINYGPU_AMD 1\n#define FW_OPENCL 1\n#define OPENCL_KERNEL_BUILD 1\n" + variant(name)
    t0 = time.time(); a = compile_hip(src, "gfx1100"); t1 = time.time(); b = compile_hip(src, "gfx1100")
    print(f"{name}: hsaco {len(a)} B, zlib-9 {len(zlib.compress(a, 9))} B, {t1 - t0:.1f} s, deterministic={a == b}, "
          f"sha={hashlib.sha256(a).hexdigest()[:12]}, e_flags={struct.unpack_from('<I', a, 0x30)[0]:#x}")
    h = ach.compile_hip(variant(name), "gfx1100")   # the AMD daemon's own compile (adds its defines)
    _, secs, rel = elf_loader(h)
    _, kernels = ach.parse_kernels(h)
    print(f"   via amd_compile_helper: kd={len(kernels)} relocs={dict(collections.Counter(r[2] for r in rel))} "
          f"sections={[s.name for s in secs if s.header.sh_type in (1, 4, 9)]}")
