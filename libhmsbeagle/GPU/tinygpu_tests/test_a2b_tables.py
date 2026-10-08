"""Plan steps A2b and N11: the generated TinyGPUAMDBootTables.h (from make_tinygpu_amd_boot_tables.py) and TinyGPUAMDReg.h's
AMRegister port, against tinygrad (hcq1) itself, for each register family (gen.FAMILIES). Offline, no device:
  1. registers, per family: every table entry equals the register tinygrad's AMDev._build_regs binds for the name on the
     family's card (its module's offset, segment and fields, the IP whose bases it takes), and golden_amd_regs.cpp's
     AMRegister reproduces tinygrad's AMRegister on a logging fake AMDev, at the card's bases, for pseudo-random addr, read,
     read_bitfields, write, update, encode, decode and fields_mask cases of every register, and a name the table lacks fails;
  2. structs: bytes(T(**vals)) for every generated struct, each union alternative in a case, equals the generated struct
     filled with the same values in C++ (sizes and offsets are the header's static_asserts);
  3. constants: every generated constant (the SMU and soc ones against each family's module), hw_id_map, the name tables and
     the chip table (each card's device ids, arch, family and captured IP versions) equal tinygrad's;
  4. the generator reruns byte-identical to the committed header.
    python test_a2b_tables.py"""
import os, re, sys, json, random, ctypes, hashlib, functools, subprocess, tempfile, filecmp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tgpaths
tgpaths.setup()
import make_tinygpu_amd_boot_tables as gen
from tinygrad.runtime.support.amd import import_asic_regs, import_module, import_soc
from tinygrad.runtime.support.am.amdev import AMRegister
from tinygrad.runtime.autogen.am import am
from tinygrad.runtime.autogen import pci
from tinygrad.runtime.support import c

# test_c2_tables.py's struct filler (span, fill, cbytes, scalar, value), copied: that module runs its checks when imported
def span(f): return (f[2] * 8 + f[4], f[2] * 8 + f[4] + f[3]) if len(f) > 3 else (f[2] * 8, (f[2] + ctypes.sizeof(f[1])) * 8)
def fill(T, case, rng, path):
    fields = [f for f in T._real_fields_ if span(f)[1] > span(f)[0]]
    taken, chosen = [], set()
    for f in (fields if case == 0 else fields[::-1]):
        lo, hi = span(f)
        if all(hi <= a or b <= lo for a, b in taken): taken.append((lo, hi)); chosen.add(f[0])
    kw, cpp = {}, []
    for f in fields:
        if f[0] not in chosen: continue
        if len(f) > 3:
            kw[f[0]] = v = rng.getrandbits(f[3])
            cpp.append(f"{path}.set_{f[0]}(0x{v:x}ull);")
        else:
            kw[f[0]], lines = value(f[1], case, rng, f"{path}.{f[0]}")
            cpp += lines
    return T(**kw), cpp
def cbytes(b): return "".join("\\x%02x" % x for x in b)
def scalar(t, rng):
    assert t is ctypes.c_void_p or t(-1).value > 0, f"{t}: a signed field; extend the test"
    return rng.getrandbits(8 * ctypes.sizeof(t))
def value(t, case, rng, path):
    if issubclass(t, ctypes.Array):
        e, n = t._type_, t._length_
        if e is ctypes.c_char:
            b = bytes(rng.randint(1, 255) for _ in range(n))
            return b, [f'memcpy({path}, "{cbytes(b)}", {n});']
        if issubclass(e, (c.Struct, ctypes.Array)):
            vals, cpp = [], []
            for i in range(n):
                v, lines = value(e, case, rng, f"{path}[{i}]")
                vals.append(v)
                cpp += lines
            return t(*vals), cpp
        vals = [scalar(e, rng) for _ in range(n)]
        return t(*vals), [f"{{ static const unsigned long long v[] = {{{', '.join(f'0x{x:x}' for x in vals)}}}; "
                          f"for (int i = 0; i < {n}; ++i) {path}[i] = v[i]; }}"]
    if issubclass(t, c.Struct): return fill(t, case, rng, path)
    v = scalar(t, rng)
    return v, [f"{path} = 0x{v:x}ull;"]

HERE, GPU, WORK = tgpaths.HERE, tgpaths.GPU_DIR, tgpaths.WORK / "a2b"
WORK.mkdir(parents=True, exist_ok=True)
HDR = GPU / "TinyGPUAMDBootTables.h"
def bases(fam):   # the family's card's discovered bases: hwip -> instance -> bases
    return {getattr(am, ip): {int(i): tuple(b) for i, b in v.items()} for ip, v in gen.card_table(fam)[1]["regs_offset"].items()}

class FakeAdev:   # what AMRegister uses of AMDev
    def __init__(self): self.vals, self.out = {}, []
    def rreg(self, reg, inst=0, direct=False):
        v = self.vals.get(reg, 0)
        self.out.append(f"r {reg:#x} {inst} {int(direct)} -> {v:#x}")
        return v
    def wreg(self, reg, val, inst=0, direct=False): self.out.append(f"w {reg:#x} {val:#x} {inst} {int(direct)}")

def table(fam):
    """The family's generated kRegs: name -> (hwip, segment, offset, [(field, start, end)])."""
    src = HDR.read_text().split(f"namespace {fam['name']} {{")[1].split(f"}} // namespace {fam['name']}")[0]
    fields = {m.group(1): re.findall(r'\{"(\w+)", (\d+), (\d+)\}', m.group(2)) for m in re.finditer(r"inline constexpr AMField kF_(\w+)\[\] = \{(.*?)\};", src, re.S)}
    out = {}
    for m in re.finditer(r'^    \{"(\w+)", (\d+), (\d+), (0x[0-9a-f]+), (kF_\w+|nullptr), (\d+)\},', src, re.M):
        name = m.group(1)
        out[name] = (int(m.group(2)), int(m.group(3)), int(m.group(4), 16), [(f, int(s), int(e)) for f, s, e in fields.get(name, [])])
    return out

def tinygrad_regs(adev, fam):
    regs, hw, b = {}, {}, bases(fam)
    for prefix, hwip, ver in gen.mods(fam):
        regs.update(import_asic_regs(prefix, ver, cls=functools.partial(AMRegister, adev=adev, bases=b[getattr(am, hwip)])))
        for n in import_module(prefix, ver, submod="regs"): hw[n] = getattr(am, hwip)
    return regs, hw

def check_registers(fam):
    t = table(fam)
    adev = FakeAdev()
    regs, hw = tinygrad_regs(adev, fam)
    bad = [n for n, (h, seg, off, fl) in t.items() if (h, seg, off, fl) != (hw[n], regs[n].segment, regs[n].offset, [(f, s, e) for f, (s, e) in regs[n].fields.items()])]
    rng, cases, want = random.Random("a2b" if fam["name"] == "gfx11" else f"a2b-{fam['name']}"), [], []
    for k, (name, (h, seg, off, fl)) in enumerate(sorted(t.items())):
        r = regs[name]
        for op in ("addr", "read", "read_bitfields", "write", "update", "encode", "decode", "fields_mask"):
            sub = [f for f in fl if rng.random() < 0.5] or fl[:1]
            vals = {f: rng.getrandbits(e - s + 1) for f, s, e in sub}
            a = rng.getrandbits(32)
            if op in ("write",):   # tinygrad's value must fit in 32 bits with the fields or'ed in
                a = rng.getrandbits(32) & ~r.fields_mask(*vals) if vals else rng.getrandbits(32)
            line = f"{k}.{op} {op} {name}" + ("" if op in ("addr", "encode", "fields_mask") else f" {a:#x}") + "".join(f" {f}={v:#x}" for f, v in vals.items())
            if op in ("read_bitfields", "decode", "read", "addr") and op != "write": line = f"{k}.{op} {op} {name}" + ("" if op == "addr" else f" {a:#x}")
            cases.append(line)
            adev.out = [f"# {k}.{op}"]
            if op == "addr": adev.out.append(f"= {r.addr[0]:#x}")
            elif op == "read": adev.vals[r.addr[0]] = a; adev.out.append(f"= {r.read():#x}")
            elif op == "read_bitfields": adev.vals[r.addr[0]] = a; adev.out.append(f"= {r.read_bitfields()}")
            elif op == "write": r.write(a, **vals)
            elif op == "update": adev.vals[r.addr[0]] = a; r.update(**vals)
            elif op == "encode": adev.out.append(f"= {r.encode(**vals):#x}")
            elif op == "decode": adev.out.append(f"= {r.decode(a)}")
            elif op == "fields_mask": adev.out.append(f"= {r.fields_mask(*vals):#x}")
            want += adev.out
    cases.append("absent.0 read regNOT_A_REGISTER 0x0")
    want += ["# absent.0", f"! no register regNOT_A_REGISTER in the {fam['name']} boot tables (AMDev's KeyError)"]
    (WORK / "bases.txt").write_text("".join(f"{h} {i} " + " ".join(f"{x:#x}" for x in b) + "\n" for h, insts in bases(fam).items() for i, b in insts.items()))
    (WORK / "cases.txt").write_text("\n".join(cases) + "\n")
    exe = WORK / "golden_amd_regs"
    tgpaths.build_cpp(HERE / "golden_amd_regs.cpp", exe)
    got = subprocess.run([str(exe), str(WORK / "bases.txt"), str(WORK / "cases.txt"), fam["name"]], capture_output=True, text=True, check=True).stdout.splitlines()
    diff = [(w, g) for w, g in zip(want + [""] * len(got), got + [""] * len(want)) if w != g]
    for w, g in diff[:5]: print(f"  tinygrad {w!r}\n  c++      {g!r}")
    ok = not bad and not diff
    print(f"registers ({fam['name']}): {len(t)} table entries {'equal' if not bad else 'DIFFER from'} tinygrad's bindings{f' ({bad[:3]})' if bad else ''}; "
          f"{len(cases)} AMRegister cases (8 per register, at the card's bases, and a missing name): {'IDENTICAL' if not diff else f'{len(diff)} lines differ'}")
    return ok

def check_structs():
    names = re.findall(r"^struct (struct_\w+|union_\w+) \{", HDR.read_text(), re.M)
    src, fns, nf = ['#include "libhmsbeagle/GPU/TinyGPUAMDBootTables.h"', "#include <cstdio>", "#include <cstring>", "using namespace tinygpu_device;",
                    "static int check(const char* name, int k, const void* got, size_t n, const char* want, size_t want_n) {",
                    "    const unsigned char* g = (const unsigned char*)got; const unsigned char* w = (const unsigned char*)want;",
                    '    if (n != want_n) { printf("MISMATCH %s case %d: sizeof %zu, tinygrad %zu\\n", name, k, n, want_n); return 1; }',
                    "    for (size_t i = 0; i < n; ++i) if (g[i] != w[i]) {",
                    '        printf("MISMATCH %s case %d: byte %zu is 0x%02x, tinygrad 0x%02x\\n", name, k, i, g[i], w[i]); return 1; }',
                    "    return 0;", "}"], [], 0
    for name in names:
        T = getattr(am, name)
        nf += len(T._real_fields_)
        for case in (0, 1):
            obj, cpp = fill(T, case, random.Random(f"a2b-{name}:{case}"), "t")
            fn = f"c_{name}_{case}"
            fns.append(fn)
            src += [f"static int {fn}() {{", f"    am::{name} t;", "    memset(&t, 0, sizeof t);"] + [f"    {x}" for x in cpp]
            src += [f'    static const char want[] = "{cbytes(bytes(obj))}";', f'    return check("{name}", {case}, &t, sizeof t, want, sizeof want - 1);', "}"]
    src += ["int main() {", "    int bad = 0;"] + [f"    bad += {fn}();" for fn in fns] + ['    printf("STRUCTS %d\\n", bad);', "    return bad != 0;", "}"]
    (WORK / "a2b_structs.cpp").write_text("\n".join(src) + "\n")
    tgpaths.build_cpp(WORK / "a2b_structs.cpp", WORK / "a2b_structs", "-O0")
    r = subprocess.run([str(WORK / "a2b_structs")], capture_output=True, text=True)
    ok = r.returncode == 0 and r.stdout.strip().endswith("STRUCTS 0")
    if not ok: print(r.stdout[-2000:], r.stderr[-1000:])
    print(f"structs: {len(names)} generated structs ({nf} fields), {len(fns)} cases, bytes(T(**vals)) against the C++-filled struct: {'IDENTICAL' if ok else 'MISMATCH'}")
    return ok

def check_constants():
    smus = [import_module("smu", f["ip"]["MP1_HWIP"]) for f in gen.FAMILIES]
    socs = [import_soc(f["ip"]["GC_HWIP"]) for f in gen.FAMILIES]
    stack, bad, n = [], [], 0
    for line in HDR.read_text().splitlines():
        ns = stack[-1] if stack else None
        if m := re.match(r"namespace (\w+) \{", line): stack.append(m.group(1))
        elif re.match(r"\} // namespace \w+", line): stack.pop()
        elif m := re.match(r"constexpr uint(?:32|64)_t (\w+) = (0x[0-9a-f]+);", line):
            n += 1
            if ns == "hsa":
                from tinygrad.runtime.autogen import hsa
                f = m.group(1).removeprefix("amd_queue_t_").removesuffix("_offset")
                if getattr(hsa.amd_queue_t, f).offset != int(m.group(2), 16): bad.append(m.group(1))
                continue
            for mod in {"am": [am], "smu": smus, "soc": socs, "pci": [pci]}[ns]:   # the SMU and soc names: in every family's module
                if getattr(mod, m.group(1)) != int(m.group(2), 16): bad.append(f"{m.group(1)} ({mod.__name__})")
        elif m := re.match(r"constexpr uint16_t hw_id_map\[\d+\] = \{(.*)\};", line):
            vals = [int(x) for x in m.group(1).split(", ")]
            if vals != [am.hw_id_map.get(i, 0) for i in range(am.MAX_HWIP)]: bad.append("hw_id_map")
        elif m := re.match(r"constexpr Name (enum_\w+)\[\] = \{(.*)", line):
            pass
    body = re.search(r"constexpr HwIdIp inv_hw_id\[\] = \{(.*?)\};", HDR.read_text(), re.S).group(1)
    if {int(a): int(b) for a, b in re.findall(r"\{(\d+), (\d+)\}", body)} != {hw_id: hw_ip for hw_ip, hw_id in am.hw_id_map.items()}: bad.append("inv_hw_id")
    for tbl in ("enum_psp_fw_type", "enum_psp_gfx_fw_type"):
        body = re.search(rf"constexpr Name {tbl}\[\] = \{{(.*?)\}};", HDR.read_text(), re.S).group(1)
        for v, s in re.findall(r'\{(\d+), "(\w+)"\}', body):
            if getattr(am, tbl)[int(v)] != s: bad.append(f"{tbl}[{v}]")
    ip_ids = [int(getattr(am, k)) for k in gen.IP_KEYS]
    if re.search(r"constexpr uint32_t kChipIP\[\] = \{(.*?)\};", HDR.read_text()).group(1).split(", ") != gen.IP_KEYS: bad.append("kChipIP")
    chips = re.findall(r'^    \{(0x[0-9a-f]+), "(\w+)", (\d+), \{(.*)\}\},$', HDR.read_text().split("constexpr Chip kChips[] = {")[1].split("};")[0], re.M)
    want = [(d, f["chip"], i, [f["ip"][k] for k in gen.IP_KEYS]) for i, f in enumerate(gen.FAMILIES) for d in f["pci_ids"]]
    got = [(int(d, 16), a, int(i), [tuple(map(int, v.split(", "))) for v in re.findall(r"\{(\d+, \d+, \d+)\}", ips)]) for d, a, i, ips in chips]
    if got != want or any(f["ip"] != {k: tuple(gen.card_table(f)[1]["ip_ver"][k]) for k in gen.IP_KEYS} for f in gen.FAMILIES): bad.append("kChips")
    print(f"constants: {n} (am, smu and soc in {len(set(smus))} and {len(set(socs))} modules, pci, hsa), hw_id_map, inv_hw_id, the name tables and the "
          f"chip table ({len(chips)} cards, their captured IP versions; kChipIP {ip_ids}) equal tinygrad's: {'IDENTICAL' if not bad else f'DIFFER: {bad[:5]}'}")
    return not bad

def check_firmware():
    """A2d: TinyGPUFirmware.h's locator finds each of am::fw's entries with the bytes tinygrad's fetch_fw returns (the network off)."""
    from tinygrad import helpers
    src = ['#include "libhmsbeagle/GPU/TinyGPUAMDBootTables.h"', '#include "libhmsbeagle/GPU/TinyGPUFirmware.h"', "#include <cstdio>",
           "using namespace tinygpu_device;", "int main() {", "    for (const nvfw::TGFirmware& f : am::fw::kFirmware) {", "        TGFirmwareFile out;",
           "        std::string err = tg_fw_locate(f, out);",
           '        if (err.empty()) printf("%s %zu %s\\n", f.name, out.size(), tg_sha256_hex(out.data(), out.size()).c_str());',
           '        else printf("%s error\\n", f.name);', "    }", "    return 0;", "}"]
    (WORK / "a2d_fw.cpp").write_text("\n".join(src) + "\n")
    tgpaths.build_cpp(WORK / "a2d_fw.cpp", WORK / "a2d_fw")
    r = subprocess.run([str(WORK / "a2d_fw")], capture_output=True, text=True, env={**os.environ, "BEAGLE_TINYGPU_NO_DOWNLOAD": "1"})
    got = {l.split()[0]: l.split()[1:] for l in r.stdout.splitlines()}
    rows = re.findall(r'\{"gfx\w+", "([\w.]+)", "amdgpu", "[\w.]+", "([0-9a-f]+)", "([0-9a-f]+)"\}', HDR.read_text())
    nrows = sum(len(gen.firmware_rows(f)) for f in gen.FAMILIES)
    bad = []
    for name, sha, md5 in rows:
        b = helpers.fetch_fw("amdgpu", name, sha)   # tinygrad's cache: the network is off (tgpaths)
        if got.get(name) != [str(len(b)), hashlib.sha256(b).hexdigest()] or hashlib.sha256(b).hexdigest() != fw_hashes()[name]: bad.append(name)
    print(f"firmware: {len(rows)} AMD entries (each card's AMFirmware fetches), located by TinyGPUFirmware.h with the bytes fetch_fw returns: "
          f"{'IDENTICAL' if not bad and len(rows) == nrows else f'DIFFER: {bad}, {len(rows)} rows of {nrows}'}")
    return not bad and len(rows) == nrows

def fw_hashes():
    from tinygrad.runtime.autogen.am import fw
    return fw.hashes

def check_regeneration():
    d = tempfile.mkdtemp(dir=WORK)
    r = subprocess.run([sys.executable, str(GPU / "make_tinygpu_amd_boot_tables.py"), d], capture_output=True, text=True)
    same = r.returncode == 0 and filecmp.cmp(os.path.join(d, HDR.name), HDR, shallow=False)
    if r.returncode: print(r.stdout[-1500:], r.stderr[-1500:])
    print(f"regeneration: make_tinygpu_amd_boot_tables.py rerun against the committed header: {'BYTE-IDENTICAL' if same else 'DIFFERS'}")
    return same

if __name__ == "__main__":
    results = [check_registers(f) for f in gen.FAMILIES] + [check_structs(), check_constants(), check_firmware(), check_regeneration()]
    print("A2b tables:", "PASS" if all(results) else "FAIL")
    sys.exit(0 if all(results) else 1)
