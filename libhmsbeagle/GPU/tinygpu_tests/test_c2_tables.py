"""Plan step C2: the generated NV tables (TinyGPUNVBootTables.h and TinyGPUNVRMTables.h, from make_tinygpu_nv_boot_tables.py)
and TinyGPUNVReg.h's NVReg port, against tinygrad (hcq1) itself. Offline, no device:
  1. registers: the include() calls tinygrad's boot code makes on each chip (Ada; GB20x, the COT boot, with
     nv_init_helper.py's include), recorded from its own code on a fake NVDev, equal the generator's; every table entry
     equals the NVReg or value that tinygrad's NVDev.include left, index functions equal the autogen's lambdas, and
     golden_c2_tables.cpp's NVReg reproduces tinygrad's NVReg on a logging fake for pseudo-random write, update,
     read_bitfields, encode, mask and decode cases through with_base(0x110000 or 0x840000) and [i];
  2. structs: bytes(T(**vals)) for every generated struct, one union alternative per case, equals the generated struct
     filled with the same values in C++ (sizes and offsets are the headers' static_asserts);
  3. MMU: NVPageTableEntry.set_entry's PTE, PDE and dual-PDE words (v2 on Ada's tables, v3 on GB20x's) equal the C++
     encodes over the generated field groups;
  4. constants: every generated constant equals tinygrad's (nv570's: nv_init_helper.py's);
  5. perturbed copies (a register offset, an MMU field, a struct field offset) are rejected;
  6. the generator reruns byte-identical to the committed headers.
    python test_c2_tables.py"""
import os, io, re, sys, types, random, shutil, ctypes, contextlib, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tgpaths
tgpaths.setup()
import nv_init_helper as h   # BEAGLE's boot patches: its COT include is in the GB20x tables (nv_init_helper.py:520)
import make_tinygpu_nv_boot_tables as gen
from tinygrad.runtime.support.nv import nvdev as nvdev_mod
from tinygrad.runtime.support.nv.nvdev import NVDev, NVReg, NVPageTableEntry
from tinygrad.runtime.support.memory import AddrSpace
from tinygrad.runtime.autogen import nv, nv_570 as nv_gpu, pci
from tinygrad.runtime.support import c

HERE, WORK, GPU = tgpaths.HERE, tgpaths.WORK / "c2", tgpaths.GPU_DIR
HEADERS = ("TinyGPUNVBootTables.h", "TinyGPUNVRMTables.h")
BOOT_42 = {"Ada": 0x19700000, "GB20x": 0x1b5a1000}   # AD107 (RTX 4060) and GB205 (RTX 5070), as test_b1_cot.py
LEVELS = {"Ada": (2, 5), "GB20x": (3, 6)}           # mmu_ver, level_cnt (nvdev.py:116, 143)
BASES = (None, 0x110000, 0x840000)                  # the GSP and SEC2 falcons (ip.py:187)

# ── 1. registers ──────────────────────────────────────────────────────────────────────────────────────────────────────

def boot_bindings(chip):
    """Run the boot code that makes include() calls (NVDev._early_ip_init, _early_mmu_init, the falcon's init_sw, with
    nv_init_helper's patches) on a fake NVDev using tinygrad's include; the calls, and the NV_* names it bound."""
    dev, calls = NVDev.__new__(NVDev), []
    dev.include = lambda name, arch: (calls.append((name, arch)), NVDev.include(dev, name, arch))
    script = {0x1FA828: 0, 0xA00: BOOT_42[chip], 0xAD00BC: 0xff, 0x1183A4: 8188}   # WPR2_HI, BOOT_42, I2CS (FSP ready), VRAM MiB
    dev.rreg, dev.wreg = (lambda addr: script.get(addr, 0)), (lambda addr, v: None)
    dev.mmio = {0x1FA828 // 4: 0}   # nv_init_helper's warm check reads BAR0 directly
    dev.pci_dev = types.SimpleNamespace(pcibus="usb4", read_config=lambda off, sz: 0, write_config_flush=lambda off, v, sz: None,
                                        map_bar=lambda bar, fmt="B": types.SimpleNamespace(nbytes=256 << 20))
    saved = nvdev_mod.NVMemoryManager
    nvdev_mod.NVMemoryManager = lambda *a, **k: None   # _early_mmu_init's page tables: no include there
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            NVDev._early_ip_init(dev)
            NVDev._early_mmu_init(dev)
            fl = dev.flcn
            fl.prep_ucode = fl.prep_booter = fl.init_fmc_image = lambda: None   # init_sw's images: no include there
            dev._alloc_boot_mem = lambda *a, **k: (None, None, [0])
            fl.init_sw()
    finally: nvdev_mod.NVMemoryManager = saved
    return calls, dev, {k: v for k, v in vars(dev).items() if k.startswith("NV_") and isinstance(v, (NVReg, int))}

def table_line(chip, name, r):
    if not isinstance(r, NVReg): return f"T {chip} {name} absent"
    group = r.base is None
    fields = ",".join(f"{f}:{lo}:{hi}" for f, (lo, hi) in r.fields.items()) or "-"
    return (f"T {chip} {name} {'group' if group else 'reg'} 0x{0 if group else r.base:x} 0x{0 if group or callable(r.off) else r.off:x} "
            f"{'fn' if callable(r.off) else '-'} {fields}")

def register_cases(chip, dev, regs, rng, first):
    """Pseudo-random NVReg cases for every register the chip binds: (case line, expected output lines)."""
    out, log = [], []
    dev.wreg = lambda addr, v: log.append(f"W 0x{addr:x} 0x{v:x}")
    for name, r in regs.items():
        group = r.base is None
        for n in range(16):
            k = first + len(out)
            items = []
            if not group:
                base, idx = rng.choice(BASES), (rng.choice([0, 1, 2, 3, 7, rng.randrange(64)]) if callable(r.off) else None)
                items = [("b", base)] if base is not None else []
                if idx is not None: items.insert(rng.randrange(len(items) + 1), ("i", idx))
            reg = r
            for kind, v in items: reg = reg.with_base(v) if kind == "b" else reg[v]
            op = ("E", "M", "X")[n % 3] if group else ("W", "U", "D", "E", "M", "X")[n % 6]
            kw = {f: rng.getrandbits(hi - lo + 1 if group or rng.random() < 0.85 else 32 - lo)   # sometimes wider: encode does not mask
                  for f, (lo, hi) in r.fields.items() if rng.random() < 0.5}
            args = " ".join(f"{f}=0x{v:x}" for f, v in kw.items())
            chain = ",".join(f"b0x{v:x}" if kind == "b" else f"i{v}" for kind, v in items) or "-"
            log.clear()
            read = rng.getrandbits(32)
            dev.rreg = lambda addr: (log.append(f"R 0x{addr:x}"), read)[1]
            if op == "W": ini = rng.getrandbits(32); reg.write(ini, **kw); args = f"0x{ini:x} {args}"
            elif op == "U": reg.update(**kw); args = f"0x{read:x} {args}"
            elif op == "D": log.append("= " + " ".join(f"{f}:0x{v:x}" for f, v in reg.read_bitfields().items())); args = f"0x{read:x}"
            elif op == "E": log.append(f"= 0x{reg.encode(**kw):x}")
            elif op == "M": log.append(f"= 0x{reg.mask(*kw):x}"); args = " ".join(kw)
            else:
                val = rng.getrandbits(128 if group else 32)
                log.append("= " + " ".join(f"{f}:0x{v:x}" for f, v in reg.decode(val).items())); args = f"0x{val:x}"
            out.append((f"{k} {chip} {name} {chain} {op} {args}".rstrip(), [f"# {k}"] + [x.rstrip() for x in log]))
    return out

def run_golden(tag, cases, extra=()):
    """golden_c2_tables.cpp (built once per tag; against a perturbed copy of the headers if extra says so) on the cases."""
    exe, path = WORK / f"golden_c2_tables{tag}", WORK / f"c2_cases{tag}.txt"
    if not exe.exists(): tgpaths.build_cpp(HERE / "golden_c2_tables.cpp", exe, *extra)
    path.write_text("".join(line + "\n" for line, _ in cases))
    r = subprocess.run([str(exe), str(path)], capture_output=True, text=True)
    return r.stdout.splitlines() + ([f"! exit {r.returncode}: {r.stderr.strip()[-300:]}"] if r.returncode else [])

def compare(got, want, what):
    bad = [(w, g) for w, g in zip(want + [""] * len(got), got + [""] * len(want)) if w != g]
    for w, g in bad[:6]: print(f"  {what}: tinygrad {w!r}\n  {' ' * len(what)}  c++      {g!r}")
    return len(bad)

def compare_cases(got, cases, what=None):
    """Each case's lines after its '# k' marker against tinygrad's; the number that differ (the first few printed)."""
    runs, cur = {}, None
    for line in got:
        if line.startswith("# "): cur = runs.setdefault(line[2:], [])
        elif cur is not None: cur.append(line)
    bad = [(line, want[1:], runs.get(want[0][2:])) for line, want in cases if runs.get(want[0][2:]) != want[1:]]
    for line, want, g in bad[:4] if what else []: print(f"  {what} case {line!r}:\n    tinygrad {want}\n    c++      {g}")
    return len(bad)

def check_registers():
    rng, cases, n_idx, counts, binds, seq_ok = random.Random("c2-registers"), [], 0, [], {}, True
    for chip in ("Ada", "GB20x"):
        calls, dev, bound = boot_bindings(chip)
        if calls != gen.INCLUDES[chip]:
            seq_ok = False
            print(f"  {chip}: tinygrad's boot code includes {calls}\n  the generator's sequence is {gen.INCLUDES[chip]}")
        binds[chip] = (dev, bound)
    ok = seq_ok
    got = run_golden("", [])
    tlines = [line for line in got if line.startswith("T ")]
    names = [line.split()[2] for line in tlines if line.split()[1] == "Ada"]
    for chip, (dev, bound) in binds.items():
        want = [table_line(chip, n, bound.get(n)) for n in names]
        ok &= compare([line for line in tlines if line.split()[1] == chip], want, f"{chip} table") == 0
        for n in names:
            r = bound.get(n)
            if isinstance(r, NVReg) and callable(r.off):
                n_idx += 1
                want_i = f"I {chip} {n} " + " ".join(f"0x{r.off(k):x}" for k in (0, 1, 2, 3, 5, 7, 8, 15, 16, 31, 63, 64, 100, 1000, 65535))
                ok &= compare([line for line in got if line.startswith(f"I {chip} {n} ")], [want_i], f"{chip} {n} index") == 0
        regs = {n: bound[n] for n in names if isinstance(bound.get(n), NVReg)}
        counts.append(f"{chip} {sum(r.base is not None for r in regs.values())} registers and {sum(r.base is None for r in regs.values())} MMU groups")
        cases += register_cases(chip, dev, regs, rng, len(cases))
    ok &= compare_cases(run_golden("", cases), cases, "NVReg") == 0
    print(f"registers: {', '.join(counts)}, bound as tinygrad's include() calls leave them (the sequences its boot code makes "
          f"{'equal' if seq_ok else 'DIFFER FROM'} the generator's); {len(tlines)} table entries, {n_idx} index functions x 15 indices; "
          f"{len(cases)} NVReg cases (write, update, read_bitfields, encode, mask, decode; with_base 0x110000/0x840000, [i]): "
          f"{'IDENTICAL' if ok else 'MISMATCH'}")
    return ok, binds, cases

# ── 2. structs ────────────────────────────────────────────────────────────────────────────────────────────────────────

def generated(pattern, root=GPU):
    """(namespace, name) for each line of the generated headers matching pattern, in order."""
    out = []
    for hdr in HEADERS:
        ns = None
        for line in (root / hdr).read_text().splitlines():
            if m := re.match(r"namespace (\w+) \{", line): ns = m.group(1)
            elif m := re.match(pattern, line): out.append((ns, *m.groups()))
    return out

def span(f):   # a field's bits
    return (f[2] * 8 + f[4], f[2] * 8 + f[4] + f[3]) if len(f) > 3 else (f[2] * 8, (f[2] + ctypes.sizeof(f[1])) * 8)

def fill(T, case, rng, path):
    """T(**vals) with pseudo-random values in every field of one union alternative (the first in case 0: fields taken
    in order while they overlap none taken; the last in case 1), and the C++ that sets them in the generated struct."""
    fields = [f for f in T._real_fields_ if span(f)[1] > span(f)[0]]   # not the zero-length arrays
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

def cbytes(b): return "".join("\\x%02x" % x for x in b)   # a C string literal's body

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

def struct_program(structs):
    src, fns = ['#include "libhmsbeagle/GPU/TinyGPUNVRMTables.h"', "#include <cstdio>", "#include <cstring>", "using namespace tinygpu_device;",
                "static int check(const char* name, int k, const void* got, size_t n, const char* want, size_t want_n) {",
                "    const unsigned char* g = (const unsigned char*)got; const unsigned char* w = (const unsigned char*)want;",
                '    if (n != want_n) { printf("MISMATCH %s case %d: sizeof %zu, tinygrad %zu\\n", name, k, n, want_n); return 1; }',
                "    for (size_t i = 0; i < n; ++i) if (g[i] != w[i]) {",
                '        printf("MISMATCH %s case %d: byte %zu is 0x%02x, tinygrad 0x%02x\\n", name, k, i, g[i], w[i]); return 1; }',
                "    return 0;", "}"], []
    fields = 0
    for ns, name in structs:
        T = getattr({"nv": nv, "nv_gpu": nv_gpu}[ns], name)
        fields += len(T._real_fields_)
        for case in (0, 1):
            obj, cpp = fill(T, case, random.Random(f"c2-{ns}::{name}:{case}"), "t")
            want = bytes(obj)
            fn = f"c_{ns}_{name}_{case}"
            fns.append(fn)
            src += [f"static int {fn}() {{", f"    {ns}::{name} t;", "    memset(&t, 0, sizeof t);"] + [f"    {x}" for x in cpp]
            src += [f'    static const char want[] = "{cbytes(want)}";',
                    f'    return check("{ns}::{name}", {case}, &t, sizeof t, want, sizeof want - 1);', "}"]
    src += ["int main() {", "    int bad = 0;"] + [f"    bad += {fn}();" for fn in fns] + ['    printf("STRUCTS %d\\n", bad);', "    return bad != 0;", "}"]
    return "\n".join(src) + "\n", len(fns), fields

def check_structs():
    structs = generated(r"struct (\w+) \{")
    src, n_cases, n_fields = struct_program(structs)
    (WORK / "c2_structs.cpp").write_text(src)
    tgpaths.build_cpp(WORK / "c2_structs.cpp", WORK / "c2_structs", "-O0")
    r = subprocess.run([str(WORK / "c2_structs")], capture_output=True, text=True)
    ok = r.returncode == 0 and r.stdout.strip().endswith("STRUCTS 0")
    if not ok: print(r.stdout[-3000:], r.stderr[-2000:])
    print(f"structs: {len(structs)} generated structs ({n_fields} fields), {n_cases} cases (each union alternative in one), "
          f"bytes(T(**vals)) against the C++-filled struct: {'IDENTICAL' if ok else 'MISMATCH'}")
    return ok

# ── 3. MMU ────────────────────────────────────────────────────────────────────────────────────────────────────────────

def pte_cases(chip, regs, rng, first):
    """NVPageTableEntry.set_entry (nvdev.py:38-49), tinygrad's own, on a bytearray standing in for the page table."""
    ver, level_cnt = LEVELS[chip]
    out = []
    for lv in range(level_cnt):
        for n in range(24):
            table, uncached, sys_, valid = (bool(n >> b & 1) for b in range(4))
            paddr, entry_id = rng.getrandbits(36) << 12, rng.randrange(256)
            pt = NVPageTableEntry.__new__(NVPageTableEntry)
            pt.nvdev = types.SimpleNamespace(pte_t=regs[f"NV_MMU_VER{ver}_PTE"], pde_t=regs[f"NV_MMU_VER{ver}_PDE"], mmu_ver=ver,
                                             dual_pde_t=regs[f"NV_MMU_VER{ver}_DUAL_PDE"], mm=types.SimpleNamespace(level_cnt=level_cnt))
            pt.paddr, pt.lv, pt.entries = 0, lv, memoryview(bytearray(0x1000)).cast("Q")
            pt.set_entry(entry_id, paddr, table=table, uncached=uncached, aspace=AddrSpace.SYS if sys_ else AddrSpace.PHYS, valid=valid)
            dual = lv == level_cnt - 2
            words = [pt.entries[2 * entry_id], pt.entries[2 * entry_id + 1]] if dual else [pt.entries[entry_id]]
            k = first + len(out)
            out.append((f"{k} S {ver} {int(dual)} 0x{paddr:x} {int(table)} {int(uncached)} {int(sys_)} {int(valid)}",
                        [f"# {k}", "= " + " ".join(f"0x{w:x}" for w in words)]))
    return out

def check_mmu(binds):
    rng, cases = random.Random("c2-mmu"), []
    for chip, (_, bound) in binds.items(): cases += pte_cases(chip, bound, rng, len(cases))
    ok = compare_cases(run_golden("", cases), cases, "set_entry") == 0
    print(f"MMU: {len(cases)} set_entry cases (v2 on Ada's tables, v3 on GB20x's; every level: PTE, PDE and 128-bit dual PDE; table, "
          f"uncached, aperture and valid in all combinations): {'IDENTICAL' if ok else 'MISMATCH'}")
    return ok, cases

# ── 4. constants ──────────────────────────────────────────────────────────────────────────────────────────────────────

NV570 = {"FALCON_APPLICATION_INTERFACE_DMEM_MAPPER_V3_CMD_FRTS": h._FWSEC_CMD_FRTS, "FALCON_APPLICATION_INTERFACE_DMEM_MAPPER_V3_CMD_SB": h._FWSEC_CMD_SB,
         "NV_VBIOS_FWSECLIC_SCRATCH_INDEX_0E": h._SCRATCH_FRTS_ERR, "NV_VBIOS_FWSECLIC_SCRATCH_INDEX_15": h._SCRATCH_SB_ERR}
NV570_FIELDS = {"NV_VBIOS_FWSECLIC_FRTS_ERR_CODE": lambda s: s >> 16,   # nv_init_helper.py:210
                "NV_VBIOS_FWSECLIC_SB_ERR_CODE": lambda s: s & 0xffff}  # nv_init_helper.py:401

def check_constants(binds):
    consts = generated(r"constexpr uint(?:32|64)_t (\w+) = ")
    fields = generated(r"constexpr nv_regs::NVField (\w+) = ")
    src = ['#include "libhmsbeagle/GPU/TinyGPUNVRMTables.h"', "#include <cstdio>", "using namespace tinygpu_device;", "int main() {"]
    src += [f'    printf("{ns}::{n} %llu\\n", (unsigned long long){ns}::{n});' for ns, n in consts]
    src += [f'    printf("{ns}::{n} %u %u\\n", {ns}::{n}.start, {ns}::{n}.end);' for ns, n in fields] + ["}"]
    (WORK / "c2_constants.cpp").write_text("\n".join(src) + "\n")
    tgpaths.build_cpp(WORK / "c2_constants.cpp", WORK / "c2_constants")
    got = subprocess.run([str(WORK / "c2_constants")], capture_output=True, text=True, check=True).stdout.splitlines()
    rng, bad, per = random.Random("c2-constants"), [], {}
    for line in got:
        key, *vals = line.split()
        ns, n = key.split("::")
        per[ns] = per.get(ns, 0) + 1
        if ns == "nv570" and len(vals) == 2:
            lo, hi = map(int, vals)
            ref = NV570_FIELDS.get(n)
            if ref is None or any((s >> lo) & ((1 << (hi - lo + 1)) - 1) != ref(s) for s in [rng.getrandbits(32) for _ in range(64)]): bad.append(line)
            continue
        v = int(vals[0])
        if ns == "nv_regs": want = {b[n] for _, b in binds.values() if n in b} or {None}
        else: want = {NV570.get(n) if ns == "nv570" else getattr({"nv": nv, "nv_gpu": nv_gpu, "pci": pci}[ns], n, None)}
        if want != {v}: bad.append(f"{line} (tinygrad {want})")
    for line in bad[:8]: print(f"  constant differs: {line}")
    ok = not bad and len(got) == len(consts) + len(fields)
    print(f"constants: {len(got)} ({', '.join(f'{ns} {k}' for ns, k in per.items())}) equal tinygrad's (nv570's: nv_init_helper.py's): "
          f"{'IDENTICAL' if ok else 'MISMATCH'}")
    return ok

# ── 5. perturbed copies ───────────────────────────────────────────────────────────────────────────────────────────────

def perturbed(tag, hdr, pattern, repl):
    """A copy of one generated header with one edit, shadowing the committed one for quoted includes (-iquote)."""
    root = WORK / f"perturb_{tag}"
    (root / "libhmsbeagle/GPU").mkdir(parents=True, exist_ok=True)
    text, n = re.subn(pattern, repl, (GPU / hdr).read_text(), count=1, flags=re.S)
    assert n == 1, f"perturbation {tag} did not apply"
    (root / "libhmsbeagle/GPU" / hdr).write_text(text)
    return ("-iquote", str(root))

def check_perturbed(reg_cases, mmu_cases):
    results, ok = [], True
    extra = perturbed("reg", HEADERS[0], r'(kAdaRegs\[NV_REG_COUNT\] = \{.*?\{"NV_PFALCON_FALCON_DMATRFCMD", kReg, 0x0, )0x118,', r"\g<1>0x11c,")
    n = compare_cases(run_golden("_perturb_reg", reg_cases, extra), reg_cases)
    results.append(f"a register offset (Ada's NV_PFALCON_FALCON_DMATRFCMD 0x118 -> 0x11c): {n} NVReg cases differ")
    ok &= n > 0
    extra = perturbed("mmu", HEADERS[0], r'(kF_dev_mmu_tu102_NV_MMU_VER2_PTE\[\] = \{[^;]*?\{"aperture", )1, 2\}', r"\g<1>2, 3}")
    n = compare_cases(run_golden("_perturb_mmu", mmu_cases, extra), mmu_cases)
    results.append(f"an MMU field (NV_MMU_VER2_PTE aperture 1:2 -> 2:3): {n} set_entry cases differ")
    ok &= n > 0
    extra = perturbed("struct", HEADERS[1], r"(struct struct_GspSystemInfo \{.*?uint8_t _pad\d+\[)7(\];\n    uint64_t clPdbProperties;\n)",
                      r"\g<1>3\g<2>    uint8_t _moved[4];\n")
    r = subprocess.run([tgpaths.cxx(), "-std=c++17", "-fsyntax-only", *extra, f"-I{tgpaths.REPO}", str(WORK / "c2_structs.cpp")],
                       capture_output=True, text=True)
    why = [line for line in r.stderr.splitlines() if "static assertion failed" in line or "static_assert failed" in line]
    results.append(f"a struct field offset (GspSystemInfo.clPdbProperties 112 -> 108, size kept): "
                   f"{'refused at compile time by ' + str(len(why)) + ' static_assert' if r.returncode and why else 'NOT refused'}")
    ok &= bool(r.returncode and why and any("clPdbProperties" in line for line in why))
    print(f"perturbed copies: {'; '.join(results)}: {'REJECTED' if ok else 'NOT ALL REJECTED'}")
    return ok

# ── 6. regeneration ───────────────────────────────────────────────────────────────────────────────────────────────────

def check_regeneration():
    same = []
    for seed in ("1", "2"):   # different string hashing: no set order may leak into the output
        out = WORK / f"regen{seed}"
        out.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, str(GPU / "make_tinygpu_nv_boot_tables.py"), str(out)], check=True,
                       env=dict(os.environ, PYTHONHASHSEED=seed))
        same += [(out / hdr).read_bytes() == (GPU / hdr).read_bytes() for hdr in HEADERS]
    ok = all(same)
    print(f"regeneration: two reruns (PYTHONHASHSEED 1 and 2) of make_tinygpu_nv_boot_tables.py against the committed headers: "
          f"{'BYTE-IDENTICAL' if ok else 'DIFFERENT (rerun the generator, or find what changed)'}")
    return ok

def main():
    if WORK.exists(): shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    ok, binds, reg_cases = check_registers()
    ok &= check_structs()
    mmu_ok, mmu_cases = check_mmu(binds)
    ok &= mmu_ok
    ok &= check_constants(binds)
    ok &= check_perturbed(reg_cases, mmu_cases)
    ok &= check_regeneration()
    sys.exit(0 if ok else 1)

main()
