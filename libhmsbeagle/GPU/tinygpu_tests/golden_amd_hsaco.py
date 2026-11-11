"""Golden test for tinygpu_amd_compile.cpp (TODO.md plan step A1j): its HSACO of every BEAGLE variant (SP and DP, 9 padded state
counts; gfx1100, and RDNA 4's gfx1200 and gfx1201 since TODO.md plan step N2) against tinygrad's own compile_hip of the same source, as the AMD daemon compiles it at run time
(amd_compile_helper.compile_hip), through the same comgr. The two are not byte-identical, and two tinygrad runs are not
either: clang names a marker symbol __hip_cuid_<16 hex digits> after the compile's temporary file, different in every
process. So everything else must be the same: every other section byte for byte (the symbol and string tables and the
two hash tables aside, which hash the names), and then what the GPU gets: elf_loader's image, the relocations and every
kernel's descriptor. The hash is printed without leading zeros, so the string tables can differ in length: the symbol
tables are compared symbol by symbol (name with the cuid blanked, value, size, binding, section). That byte can also move
.rodata, laid out after .dynstr at a 64-byte alignment, when .dynstr ends at a multiple of 64 in one of the two (moved):
then the rest must be the same with .rodata's addresses moved, and every kernel descriptor must reach the same code.
Needs tinygrad's comgr build (COMGR_PATH, default /opt/homebrew/lib/libamd_comgr.dylib).
Each arch's two sides compile at once: tinygpu_amd_compile in one process with every variant, as make_tinygpu_hsaco.sh runs
it, while this process runs tinygrad's compile_hip. With the build's spectral variants (TODO.md plan step S8.3), the same for
tinygpu_amd_compile_spectral's SPEC_ variants of BeagleOpenCLSpectral_kernels.h, as parts of their own.
    python golden_amd_hsaco.py [--part N/K]   (the N-th of PARTS only, K = len(PARTS): run_goldens.sh's jobs)
    python golden_amd_hsaco.py --parts        (prints K)"""
import os, sys, hashlib, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tgpaths
tgpaths.setup()
import amd_compile_helper as ach

HERE, WORK = tgpaths.HERE, tgpaths.WORK
COMGR = os.environ.get("COMGR_PATH", "/opt/homebrew/lib/libamd_comgr.dylib")
VARIANTS = [f"{p}_{n}" for p in ("SP", "DP") for n in (4, 16, 32, 48, 64, 80, 128, 192, 256)]
ARCHS = ["gfx1100", "gfx1200", "gfx1201"]   # the build embeds gfx1100's; RDNA 4's are compiled here only (TODO.md plan step N2)

# the build has the spectral variants: make_tinygpu_hsaco.sh's table has SPEC_ rows
SPECTRAL = '"SPEC_' in (tgpaths.GPU_DIR / "kernels/TinyGPUAMDHsaco.h").read_text()
PARTS = [(arch, False) for arch in ARCHS] + ([(arch, True) for arch in ARCHS] if SPECTRAL else [])

headers = {}
def variant(name):   # as golden_amd_program.py: KERNELS_STRING_<name>, the source the plugin sends the daemon (SPEC_<name>: the
    spectral = name.startswith("SPEC_")   # spectral header's KERNELS_STRING_<name>)
    if spectral not in headers:
        headers[spectral] = (tgpaths.GPU_DIR / ("kernels/BeagleOpenCLSpectral_kernels.h" if spectral else
                                                "kernels/BeagleOpenCL_kernels.h")).read_text().split("\n")
    hdr, name = headers[spectral], name[5:] if spectral else name
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
def section_headers(b):   # (name, sh_type, sh_addr, sh_size) of each section header
    shoff, shnum, shstrndx = struct.unpack_from("<Q", b, 0x28)[0], *struct.unpack_from("<HH", b, 0x3c)
    hs = [struct.unpack_from("<IIQQQQ", b, shoff + i * 64) for i in range(shnum)]
    strtab = hs[shstrndx][4]
    return [(b[strtab + h[0]:b.index(b"\0", strtab + h[0])].decode(), h[1], h[3], h[5]) for h in hs]
def entries(secs, rodata):   # {kernel descriptor symbol: (its address, the code address it enters)}; .rodata at rodata
    out = {}
    for i in range(len(secs[".symtab"]) // 24):
        st_name, _, _, _, value, _ = struct.unpack_from("<IBBHQQ", secs[".symtab"], i * 24)
        name = secs[".strtab"][st_name:secs[".strtab"].index(b"\0", st_name)]
        if name.endswith(b".kd"):   # its kernel_code_entry_byte_offset: 8 bytes at 16, relative to the descriptor
            out[name] = (value, value + struct.unpack_from("<q", secs[".rodata"], value - rodata + 16)[0])
    return out
def blank(rodata_bytes, kds, rodata):   # .rodata with each descriptor's code entry offset zeroed
    b = bytearray(rodata_bytes)
    for kd, _ in kds.values(): b[kd - rodata + 16:kd - rodata + 24] = bytes(8)
    return bytes(b)
def loaded(image, secs):   # elf_loader's image is its PROGBITS sections' bytes, at their addresses, and zeros between
    b = bytearray(image)
    for sec in secs:
        if sec.header.sh_type == 1 and sec.header.sh_size:   # SHT_PROGBITS
            a, n = sec.header.sh_addr, sec.header.sh_size
            if bytes(b[a:a + n]) != bytes(sec.content): return False
            b[a:a + n] = bytes(n)
    return not any(b)
def moved(got, ref):
    """"" if got and ref are the same but for the cuid and .rodata moved by it (the docstring), else what differs"""
    gc, rc = CUID.findall(got), CUID.findall(ref)
    gh, rh = section_headers(got), section_headers(ref)
    if [h[:2] for h in gh] != [h[:2] for h in rh]: return "the section list"
    for (name, _, ga, gz), (_, _, ra, rz) in zip(gh, rh):
        if name != ".rodata" and ga != ra: return "the layout"
        if name not in NAMES and gz != rz: return f"the size of {name}"
    gs, rs = sections(got), sections(ref)
    for name in gs:
        if name not in NAMES and name != ".rodata" and gs[name] != rs[name]: return f"section {name}"
    (gd, gsz), (rd, rsz) = dynamic(gs), dynamic(rs)
    if gd != rd or gsz - rsz != len(gc[0]) - len(rc[0]): return "section .dynamic"
    lo, hi = next((a, a + z) for name, _, a, z in rh if name == ".rodata")
    d = next(a for name, _, a, _ in gh if name == ".rodata") - lo
    move = lambda a: a + d if lo <= a < hi else a   # an address in ref, in got's layout
    gk, rk = entries(gs, lo + d), entries(rs, lo)
    if gk.keys() != rk.keys() or any(gk[k] != (move(rk[k][0]), move(rk[k][1])) for k in gk):
        return "the kernel descriptors' addresses or the code they enter"
    if blank(gs[".rodata"], gk, lo + d) != blank(rs[".rodata"], rk, lo): return "section .rodata"
    ndx = [h[0] for h in rh].index(".rodata")
    for tab, strs in ((".symtab", ".strtab"), (".dynsym", ".dynstr")):
        if symbols(gs, tab, strs) != sorted((n, v + d if x == ndx else v, z, i, o, x) for n, v, z, i, o, x in symbols(rs, tab, strs)):
            return f"the symbols of {tab}"
    (gi, gl, grel), (ri, rl, rrel) = elf_loader(got), elf_loader(ref)
    if len(gi) != len(ri) or grel != rrel or not loaded(gi, gl) or not loaded(ri, rl):
        return "elf_loader's image or relocations"
    gk, rk = ach.parse_kernels(got)[1], ach.parse_kernels(ref)[1]
    unentered = lambda desc: bytes(desc)[:16] + bytes(desc)[24:]
    if gk.keys() != rk.keys() or any(gk[k][0] != move(rk[k][0]) or unentered(gk[k][1]) != unentered(rk[k][1]) for k in gk):
        return "the kernel descriptors"
    return ""

def strictly(got, ref):
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

def compare(got, ref):
    """(True, how) if all that matters is the same, else (False, what differs)"""
    why = strictly(got, ref)
    if not why: return True, "IDENTICAL but for the compile's __hip_cuid_"
    gc, rc = CUID.findall(got), CUID.findall(ref)
    if gc and len(gc) == len(rc) and len(gc[0]) != len(rc[0]):   # a shorter cuid may have moved .rodata
        why_moved = moved(got, ref)
        if not why_moved: return True, "IDENTICAL but for the compile's __hip_cuid_, and .rodata moved by it"
        why += f"; with .rodata moved: {why_moved}"
    return False, f"DIFFERS: {why}"

def check_arch(arch, spectral=False):
    """arch's variants (the SPEC_ ones with spectral) compared, a line each: the number that differ"""
    out = WORK / ("hsaco_spec" if spectral else "hsaco"); out.mkdir(exist_ok=True)
    for f in out.glob(f"*_{arch}.hsaco"): f.unlink()
    variants = [f"SPEC_{v}" for v in VARIANTS] if spectral else VARIANTS
    tool = "tinygpu_amd_compile_spectral" if spectral else "tinygpu_amd_compile"
    # the compiler also refuses a variant whose kernels spill registers (STATUS.md R97), and the spectral one a variant the
    # plugin's loader would refuse
    cpp = subprocess.Popen([str(WORK / tool), COMGR, arch, str(out)] + variants, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, text=True)
    try: refs = {v: ach.compile_hip(variant(v), arch) for v in variants}
    except BaseException: cpp.kill(); raise
    err = cpp.communicate()[1]
    if cpp.returncode: raise RuntimeError(f"{tool} {arch}: exit {cpp.returncode}\n{err[-3000:]}")
    fails = 0
    for v in variants:
        got = (out / f"{v}_{arch}.hsaco").read_bytes()
        same, how = compare(got, refs[v])
        fails += not same
        print(f"{v} {arch}: {len(got)} bytes, {len(ach.parse_kernels(got)[1])} kernels: {how}")
    return fails

def main(part=None):   # a part exits with its number of differing variants; run_goldens.sh prints the verdict
    parts = [PARTS[part - 1]] if part else PARTS
    if any(not s for _, s in parts):
        tgpaths.build_cpp(tgpaths.GPU_DIR / "tinygpu_amd_compile.cpp", WORK / "tinygpu_amd_compile")
    if any(s for _, s in parts):
        tgpaths.build_cpp(tgpaths.GPU_DIR / "tinygpu_amd_compile.cpp", WORK / "tinygpu_amd_compile_spectral", "-DTINYGPU_AMD_SPECTRAL")
    fails = sum(check_arch(arch, s) for arch, s in parts)
    if part: sys.exit(fails)
    print("A1j build-time HSACOs vs tinygrad's compile_hip:", "all identical" if not fails else "MISMATCH")
    sys.exit(1 if fails else 0)

if __name__ == "__main__":
    if sys.argv[1:2] == ["--parts"]:
        print(len(PARTS))
    elif sys.argv[1:2] == ["--part"]:
        n, k = map(int, sys.argv[2].split("/"))
        if k != len(PARTS) or not 1 <= n <= k: sys.exit(f"--part {sys.argv[2]}: golden_amd_hsaco has {len(PARTS)} parts")
        main(n)
    else: main()
