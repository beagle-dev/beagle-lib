"""Golden test for tinygpu_amd_compile.cpp (TODO.md plan step A1j): its HSACO of every BEAGLE variant (SP and DP, 9 padded state
counts; gfx1100, and RDNA 4's gfx1200 and gfx1201 since TODO.md plan step N2) against tinygrad's own compile_hip of the same source, as the AMD daemon compiles it at run time
(amd_compile_helper.compile_hip), through the same comgr. The two are not byte-identical, and two tinygrad runs are not
either: clang names a marker symbol __hip_cuid_<16 hex digits> after the compile's temporary file, different in every
process. So everything else must be the same: every other section byte for byte (the symbol and string tables and the
two hash tables aside, which hash the names), and then what the GPU gets: elf_loader's image, the relocations and every
kernel's descriptor. The hash is printed without leading zeros, so the string tables can differ in length: the symbol
tables are compared symbol by symbol (name with the cuid blanked, value, size, binding, section). Needs tinygrad's comgr build (COMGR_PATH, default /opt/homebrew/lib/libamd_comgr.dylib)."""
import os, sys, hashlib, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tgpaths
tgpaths.setup()
import amd_compile_helper as ach

HERE, WORK = tgpaths.HERE, tgpaths.WORK
tgpaths.build_cpp(tgpaths.GPU_DIR / "tinygpu_amd_compile.cpp", WORK / "tinygpu_amd_compile")
COMGR = os.environ.get("COMGR_PATH", "/opt/homebrew/lib/libamd_comgr.dylib")
VARIANTS = [f"{p}_{n}" for p in ("SP", "DP") for n in (4, 16, 32, 48, 64, 80, 128, 192, 256)]
ARCHS = ["gfx1100", "gfx1200", "gfx1201"]   # the build embeds gfx1100's; RDNA 4's are compiled here only (TODO.md plan step N2)
out = WORK / "hsaco"; out.mkdir(exist_ok=True)
for f in out.glob("*.hsaco"): f.unlink()
for arch in ARCHS:   # the compiler also refuses a variant whose kernels spill registers (STATUS.md R97)
    subprocess.run([str(WORK / "tinygpu_amd_compile"), COMGR, arch, str(out)] + VARIANTS, check=True, capture_output=True)

hdr = (tgpaths.GPU_DIR / "kernels/BeagleOpenCL_kernels.h").read_text().split("\n")
def variant(name):   # as golden_amd_program.py: KERNELS_STRING_<name>, the source the plugin sends the daemon
    i = next(k for k, l in enumerate(hdr) if l.startswith(f"#define KERNELS_STRING_{name} \""))
    lines = []
    for l in hdr[i + 1:]:
        if l == '"': break
        lines.append(l[:-3].replace('\\"', '"').replace('\\\\', '\\'))
    return "\n".join(lines) + "\n"

import re, struct
from tinygrad.runtime.support.elf import elf_loader
CUID = re.compile(rb"__hip_cuid_[0-9a-f]+")   # its hash is printed without leading zeros: 15 or 16 digits
def sections(b):
    shoff, shnum, shstrndx = struct.unpack_from("<Q", b, 0x28)[0], *struct.unpack_from("<HH", b, 0x3c)
    hs = [struct.unpack_from("<IIQQQQ", b, shoff + i * 64) for i in range(shnum)]
    strtab = hs[shstrndx][4]
    return {b[strtab + h[0]:b.index(b"\0", strtab + h[0])].decode(): b[h[4]:h[4] + h[5]] for h in hs}
def symbols(secs, tab, strs):   # (name with the cuid blanked, value, size, info, other, shndx) of each Elf64_Sym
    out = []
    for i in range(len(secs[tab]) // 24):
        st_name, info, other, shndx, value, size = struct.unpack_from("<IBBHQQ", secs[tab], i * 24)
        name = secs[strs][st_name:secs[strs].index(b"\0", st_name)]
        out.append((CUID.sub(b"__hip_cuid_", name), value, size, info, other, shndx))
    return sorted(out)
NAMES = (".symtab", ".strtab", ".dynsym", ".dynstr", ".gnu.hash", ".hash", ".dynamic")   # name-dependent: compared below
def dynamic(secs):   # .dynamic's (d_tag, d_val) entries, DT_STRSZ (10) apart
    d = [struct.unpack_from("<qQ", secs[".dynamic"], o) for o in range(0, len(secs[".dynamic"]), 16)]
    return [e for e in d if e[0] != 10], next(v for t, v in d if t == 10)
def compare(got, ref):
    """"" if all that matters is the same, else what differs"""
    gc, rc = CUID.findall(got), CUID.findall(ref)
    if len(gc) != len(rc): return "the __hip_cuid_ symbols"
    gs, rs = sections(got), sections(ref)
    if gs.keys() != rs.keys(): return "the section list"
    for name in gs:
        if name not in NAMES and gs[name] != rs[name]: return f"section {name}"
    (gd, gsz), (rd, rsz) = dynamic(gs), dynamic(rs)
    # DT_STRSZ is .dynstr's size, which holds the cuid name once
    if gd != rd or gsz - rsz != len(gc[0]) - len(rc[0]): return "section .dynamic"
    for tab, strs in ((".symtab", ".strtab"), (".dynsym", ".dynstr")):
        if symbols(gs, tab, strs) != symbols(rs, tab, strs): return f"the symbols of {tab}"
    (gi, _, grel), (ri, _, rrel) = elf_loader(got), elf_loader(ref)
    if bytes(gi) != bytes(ri) or grel != rrel: return "elf_loader's image or relocations"
    if {k: (a, bytes(d)) for k, (a, d) in ach.parse_kernels(got)[1].items()} != {k: (a, bytes(d)) for k, (a, d) in ach.parse_kernels(ref)[1].items()}:
        return "the kernel descriptors"
    return ""

ok = True
for arch in ARCHS:
    for v in VARIANTS:
        got, ref = (out / f"{v}_{arch}.hsaco").read_bytes(), ach.compile_hip(variant(v), arch)
        why = compare(got, ref)
        ok &= not why
        print(f"{v} {arch}: {len(got)} bytes, {len(ach.parse_kernels(got)[1])} kernels: " + ("IDENTICAL but for the compile's __hip_cuid_" if not why else f"DIFFERS: {why}"))
print("A1j build-time HSACOs vs tinygrad's compile_hip:", "all identical" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
