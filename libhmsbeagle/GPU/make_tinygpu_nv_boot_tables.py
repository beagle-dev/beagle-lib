#!/usr/bin/env python3
"""
Generate TinyGPUNVBootTables.h and TinyGPUNVRMTables.h (TODO.md plan step C2): the registers, MMU field groups,
structs and constants the C++ port of tinygrad's NV boot and RM setup needs, taken from tinygrad's own autogen in the
pinned hcq1 tree (runtime/autogen/nv_regs, nv.py, nv_570.py, pci.py; plan decision 9) rather than NVIDIA's headers.
The set is every nv.*, nv_gpu.*, pci.* symbol and NV_* register name that ip.py, nvdev.py and nv_init_helper.py use
(an AST scan, so a new use changes the output and tinygpu_tests/test_c2_tables.py fails until this is rerun), plus
ops_nv.py's PCIIface path and the names the code builds at run time (listed below). Registers are bound as
tinygrad's NVDev.include leaves them after each chip's include() sequence; structs are laid out at tinygrad's recorded
offsets. NVReg itself is ported by hand in TinyGPUNVReg.h. Rerun after changing the tinygrad pin or those files, with
the harness Python (BEAGLE_PYTHON in tinygpu_tests/env.sh):

    python make_tinygpu_nv_boot_tables.py [outdir]    # default: next to this script
"""
import ast, ctypes, inspect, os, pathlib, re, subprocess, sys, textwrap

HERE = pathlib.Path(__file__).resolve().parent
TINYGRAD = pathlib.Path(os.environ.get("TINYGRAD_PATH", pathlib.Path.home() / "Dropbox/Projects/tinygrad-hcq1"))
sys.path.insert(0, str(TINYGRAD))
import tinygrad.runtime.autogen.nv_regs as nv_regs  # noqa: E402
from tinygrad.runtime.autogen import nv, nv_570 as nv_gpu, pci  # noqa: E402
from tinygrad.runtime.support import c  # noqa: E402
from tinygrad.runtime.support.nv.nvdev import NVDev, NVReg  # noqa: E402

# Each chip's include() calls in boot order; a later include overwrites a name (nvdev.py:161-163).
INCLUDES = {
    "Ada": [("nv_ref", ""), ("dev_fb", "tu102"), ("dev_gc6_island", "ga102"),          # NVDev._early_ip_init, nvdev.py:101-103
            ("dev_vm", "tu102"), ("dev_mmu", "tu102"),                                  # _early_mmu_init, nvdev.py:124,127
            ("dev_gsp", "ga102"), ("dev_falcon_v4", "ga102"), ("dev_riscv_pri", "ga102"), ("dev_fbif_v4", "ga102"),
            ("dev_falcon_second_pri", "ga102"), ("dev_sec_pri", "ga102"), ("dev_bus", "tu102")],   # NV_FLCN.init_sw, ip.py:99-105
    "GB20x": [("nv_ref", ""), ("dev_fb", "tu102"), ("dev_gc6_island", "ga102"),        # nvdev.py:101-103
              ("dev_therm", "gb202"),                                                   # NV_FLCN_COT.wait_for_reset, ip.py:287
              ("dev_vm", "tu102"), ("dev_mmu", "gh100"),                                # nvdev.py:124,127
              ("dev_riscv_pri", "ga102"),                                               # BEAGLE's, first in init_sw: nv_init_helper.py:520
              ("dev_gsp", "ga102"), ("dev_falcon_v4", "gh100"), ("dev_vm", "gh100"), ("dev_fsp_pri", "gh100"),
              ("dev_bus", "tu102")],                                                    # NV_FLCN_COT.init_sw, ip.py:291-295
}
GROUPS = [f"NV_MMU_VER{v}_{g}" for v in (2, 3) for g in ("PDE", "DUAL_PDE", "PTE")]   # the names nvdev.py:128-129 builds

HELPER = "libhmsbeagle/GPU/nv_init_helper.py"
SCANNED = [TINYGRAD / "tinygrad/runtime/support/nv/ip.py", TINYGRAD / "tinygrad/runtime/support/nv/nvdev.py", HERE / "nv_init_helper.py"]
SKIP = {("nv", "rpc_fns"), ("nv", "rpc_events")}   # name tables for log lines (ip.py:81, nv_init_helper.py:169)

# ops_nv.py's PCIIface path (ops_nv.py:556-760): PCIIface.__init__, NVDevice.__init__, _new_gpu_fifo without its video
# branch, _query_gpu_info, invalidate_caches and on_device_hang, each without its NVKIface branch. Not map_flags'
# NVOS33_FLAGS_CACHING_TYPE_WRITECOMBINED (:614), which PCIIfaceBase.alloc ignores (system.py:267).
OPS_NV = ["NV01_ROOT", "NV0000_ALLOC_PARAMETERS",                                                              # :564
          "NV0080_ALLOC_PARAMETERS", "NV_DEVICE_ALLOCATION_VAMODE_OPTIONAL_MULTIPLE_VASPACES", "NV01_DEVICE_0",  # :593-597
          "NV20_SUBDEVICE_0", "NV2080_ALLOC_PARAMETERS", "NV01_MEMORY_VIRTUAL", "NV_MEMORY_VIRTUAL_ALLOCATION_PARAMS",
          "NV2080_CTRL_CMD_PERF_BOOST", "NV2080_CTRL_PERF_BOOST_PARAMS", "NV2080_CTRL_PERF_BOOST_FLAGS_CUDA_YES",   # :600-602
          "NV2080_CTRL_PERF_BOOST_FLAGS_CUDA_PRIORITY_HIGH", "NV2080_CTRL_PERF_BOOST_FLAGS_CMD_BOOST_TO_MAX",
          "NV_VASPACE_ALLOCATION_PARAMETERS", "NV_VASPACE_ALLOCATION_FLAGS_ENABLE_PAGE_FAULTING",                  # :604-606
          "NV_VASPACE_ALLOCATION_FLAGS_IS_EXTERNALLY_OWNED", "FERMI_VASPACE_A",
          "NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS", "NV2080_ENGINE_TYPE_GRAPHICS", "KEPLER_CHANNEL_GROUP_A",       # :610-611
          "NV_CTXSHARE_ALLOCATION_PARAMETERS", "NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_ASYNC", "FERMI_CONTEXT_SHARE_A",   # :616-617
          "NVA06C_CTRL_CMD_GPFIFO_SCHEDULE", "NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS",                                 # :621
          "NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS", "NV83DE_ALLOC_PARAMETERS", "GT200_DEBUGGER",                   # :644-653
          "NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN", "NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS",   # :661-662
          "AmpereAControlGPFifo",                                                                                 # :666
          "NV2080_CTRL_CMD_INTERNAL_STATIC_KGR_GET_INFO", "NV2080_CTRL_INTERNAL_STATIC_GR_GET_INFO_PARAMS",       # :672-673
          "NV2080_CTRL_CMD_INTERNAL_BUS_FLUSH_WITH_SYSMEMBAR",                                                    # :733
          "NV83DE_CTRL_CMD_DEBUG_READ_ALL_SM_ERROR_STATES", "NV83DE_CTRL_DEBUG_READ_ALL_SM_ERROR_STATES_PARAMS",   # :744-749
          "NV83DE_CTRL_CMD_DEBUG_READ_MMU_FAULT_INFO", "NV83DE_CTRL_DEBUG_READ_MMU_FAULT_INFO_PARAMS"]
# names built at run time: _query_gpu_info's GR info indices (ops_nv.py:669, nv_init_helper.py:839) and the fault names
# of on_device_hang's report (ops_nv.py:34-35)
OPS_NV += [f"NV2080_CTRL_GR_INFO_INDEX_{r}" if hasattr(nv_gpu, f"NV2080_CTRL_GR_INFO_INDEX_{r}") else f"NV2080_CTRL_GR_INFO_INDEX_LITTER_{r}"
           for r in ("NUM_GPCS", "NUM_TPC_PER_GPC", "NUM_SM_PER_TPC", "MAX_WARPS_PER_SM", "SM_VERSION")]
OPS_NV += [n for n in vars(nv_gpu) if n.startswith(("NV_PFAULT_FAULT_TYPE_", "NV_PFAULT_ACCESS_TYPE_"))]

# The nv structs the boot uses before GSP-RM answers RPCs (VBIOS and falcon ucode, FWSECLIC, firmware headers, WPR meta,
# libos, the message-queue framing, FSP/COT/FMC) go in the boot header with their closure; the other nv structs (RPC
# payloads) and nv constants named RM_NV_PREFIXES (RPC ids and results), and all of nv_gpu (RM objects), go in the RM one.
BOOT_NV = {"BIT_HEADER_V1_00", "BIT_TOKEN_V1_00", "BIT_DATA_FALCON_DATA_V2", "FALCON_UCODE_TABLE_HDR_V1", "FALCON_UCODE_TABLE_ENTRY_V1",
           "FALCON_UCODE_DESC_HEADER", "FALCON_UCODE_DESC_V3", "FALCON_APPLICATION_INTERFACE_HEADER_V1",
           "FALCON_APPLICATION_INTERFACE_ENTRY_V1", "FALCON_APPLICATION_INTERFACE_DMEM_MAPPER_V3", "FWSECLIC_READ_VBIOS_DESC",
           "FWSECLIC_FRTS_REGION_DESC", "FWSECLIC_FRTS_CMD", "struct_nvfw_bin_hdr", "struct_nvfw_hs_header_v2",
           "struct_nvfw_hs_load_header_v2", "struct_nvfw_hs_load_header_v2_app", "RM_RISCV_UCODE_DESC", "GspFwWprMeta",
           "LibosMemoryRegionInitArgument", "MESSAGE_QUEUE_INIT_ARGUMENTS", "GSP_ARGUMENTS_CACHED", "msgqTxHeader",
           "GSP_MSG_QUEUE_ELEMENT", "rpc_message_header_v", "GSP_FMC_BOOT_PARAMS", "GSP_ACR_BOOT_GSP_RM_PARAMS", "GSP_RM_PARAMS",
           "NVDM_PAYLOAD_COT"}
RM_NV_PREFIXES = ("NV_VGPU_", "REGISTRY_TABLE_")

# NVIDIA 570.144 constants tinygrad's autogen lacks, as nv_init_helper.py uses them (section 5, the P2 teardown; its
# names in parentheses): (name, value or (start, end) field, provenance in open-gpu-kernel-modules 570.144)
NV570_SRC = "src/nvidia/src/kernel/gpu/gsp/arch/turing/kernel_gsp_frts_tu102.c"
NV570 = [("FALCON_APPLICATION_INTERFACE_DMEM_MAPPER_V3_CMD_FRTS", 0x15, "kernel_gsp_frts_tu102.c:101 (_FWSEC_CMD_FRTS; ip.py:169 passes 0x15)"),
         ("FALCON_APPLICATION_INTERFACE_DMEM_MAPPER_V3_CMD_SB", 0x19, "kernel_gsp_frts_tu102.c:102 (_FWSEC_CMD_SB)"),
         ("NV_VBIOS_FWSECLIC_SCRATCH_INDEX_0E", 0x0e, "kernel_gsp_frts_tu102.c:133, an NV_PBUS_VBIOS_SCRATCH index (_SCRATCH_FRTS_ERR)"),
         ("NV_VBIOS_FWSECLIC_FRTS_ERR_CODE", (16, 31), "31:16, kernel_gsp_frts_tu102.c:134 (scratch >> 16)"),
         ("NV_VBIOS_FWSECLIC_SCRATCH_INDEX_15", 0x15, "kernel_gsp_frts_tu102.c:137 (_SCRATCH_SB_ERR)"),
         ("NV_VBIOS_FWSECLIC_SB_ERR_CODE", (0, 15), "15:0, kernel_gsp_frts_tu102.c:138 (scratch & 0xffff)")]

# ── the used set ──────────────────────────────────────────────────────────────────────────────────────────────────────

def scan(path):
    """(module, name) for each nv/_nv/nv_gpu/pci attribute; ("reg", name) for each other NV_* attribute or string."""
    mods = {"nv": "nv", "_nv": "nv", "nv_gpu": "nv_gpu", "pci": "pci"}
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Attribute):
            v = node.value
            if isinstance(v, ast.Name) and v.id in mods: yield mods[v.id], node.attr
            elif isinstance(v, ast.Attribute) and v.attr == "nv_gpu": yield "nv_gpu", node.attr   # _ops_nv.nv_gpu.X
            elif node.attr.startswith("NV_"): yield "reg", node.attr                             # nvdev.NV_X
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and re.fullmatch(r"NV_\w+", node.value):
            yield "reg", node.value                                                              # reg("NV_X"), __dict__.get("NV_X")

def bind(chip):
    """The chip's include() calls replayed through tinygrad's NVDev.include: name -> ((module, arch), NVReg or value)."""
    dev, where = NVDev.__new__(NVDev), {}
    for mod, arch in INCLUDES[chip]:
        NVDev.include(dev, mod, arch)
        for k in getattr(getattr(nv_regs, mod), arch or "regs"): where[k] = (mod, arch)
    return {k: (where[k], v) for k, v in vars(dev).items()}

def is_struct(v): return isinstance(v, type) and issubclass(v, c.Struct)
def is_int(v): return isinstance(v, int) and not isinstance(v, bool)

def used_set():
    used = {"nv": set(), "nv_gpu": set(OPS_NV), "pci": set(), "reg": set(GROUPS)}
    for p in SCANNED:
        for mod, name in scan(p):
            if (mod, name) not in SKIP: used[mod].add(name)
    for mod, m in (("nv", nv), ("nv_gpu", nv_gpu), ("pci", pci)):
        for name in used[mod]:
            v = getattr(m, name)
            assert is_struct(v) or is_int(v), f"{mod}.{name} is a {type(v).__name__}: generate it or add it to SKIP"
    return used

# ── registers ─────────────────────────────────────────────────────────────────────────────────────────────────────────

def lambda_text(mod, arch, name):
    """An indexed register's off as the autogen writes it: (argument, body)."""
    src = (pathlib.Path(nv_regs.__file__).parent / f"{mod}.py").read_text()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and node.targets[0].id == (arch or "regs"):
            for k, v in zip(node.value.keys, node.value.values):
                if k.value == name:
                    lam = v.elts[1]
                    arg, body = lam.args.args[0].arg, ast.get_source_segment(src, lam.body)
                    assert re.fullmatch(r"[0-9A-Fa-fx+*() ]*", re.sub(rf"\b{arg}\b", "", body)), body
                    return arg, body
    raise KeyError(name)

def sym(mod, arch, name): return f"{mod}{'_' + arch if arch else ''}_{name}"

def register_section(used, binds):
    names = [n for n in used["reg"] if any(n in b for b in binds.values())]
    order = []   # first bound in Ada's include order, then GB20x's
    for chip in INCLUDES:
        for mod, arch in INCLUDES[chip]:
            order += [k for k in getattr(getattr(nv_regs, mod), arch or "regs") if k in names and k not in order]
    regs = [n for n in order if all(isinstance(b[n][1], NVReg) for b in binds.values() if n in b)]
    values = [n for n in order if all(is_int(b[n][1]) for b in binds.values() if n in b)]
    assert len(regs) + len(values) == len(order), "a name bound as a register on one chip and a value on another"
    out, emitted = ["enum NVRegId {"], set()
    out += [f"    {n}," for n in regs] + ["    NV_REG_COUNT", "};", ""]
    for n in regs:   # each binding's fields and index function, once
        for chip, b in binds.items():
            if n not in b or (key := b[n][0] + (n,)) in emitted: continue
            emitted.add(key)
            r, s = b[n][1], sym(*key)
            assert len(r.fields) <= 32 and all(0 <= lo <= hi and hi - lo < 64 for lo, hi in r.fields.values()), n
            if r.fields:
                items = [f'{{"{f}", {lo}, {hi}}}' for f, (lo, hi) in r.fields.items()]
                one = f"constexpr NVField kF_{s}[] = {{{', '.join(items)}}};"
                out += [one] if len(one) <= 120 else [f"constexpr NVField kF_{s}[] = {{"] + fill(items, "    ", "    ") + ["};"]
            if callable(r.off):
                arg, body = lambda_text(*key)
                out.append(f"constexpr uint32_t kI_{s}(uint32_t {arg}) {{ return {body}; }}")
    for chip, b in binds.items():
        out += ["", f"constexpr NVRegDef k{chip}Regs[NV_REG_COUNT] = {{"]
        for n in regs:
            if n not in b:
                out.append(f'    {{"{n}", kAbsent, 0x0, 0x0, nullptr, nullptr, 0}},')
                continue
            (mod, arch), r = b[n]
            s, group = sym(mod, arch, n), r.base is None
            base, off = (0, 0) if group else (r.base, 0 if callable(r.off) else r.off)
            out.append(f'    {{"{n}", {"kGroup" if group else "kReg"}, {base:#x}, {off:#x}, {f"kI_{s}" if callable(r.off) else "nullptr"}, '
                       f'{f"kF_{s}" if r.fields else "nullptr"}, {len(r.fields)}}},  // {mod}{" " + arch if arch else ""}')
        out.append("};")
    out += ["", "// register values the boot uses"]
    for n in values:
        vals = {b[n][1] for b in binds.values() if n in b}
        assert len(vals) == 1, n
        where = sorted({f"{b[n][0][0]} {b[n][0][1]}" for b in binds.values() if n in b})
        out.append(f"constexpr uint32_t {n} = {vals.pop():#x};  // {', '.join(where)}, on {', '.join(ch for ch, b in binds.items() if n in b)}")
    return out, regs, values

# ── structs ───────────────────────────────────────────────────────────────────────────────────────────────────────────

def closure(classes, done):
    """The classes and the struct types of their fields, dependencies first, skipping those in done."""
    order = []
    def visit(t):
        while isinstance(t, type) and issubclass(t, ctypes.Array): t = t._type_
        if not is_struct(t) or t in done or t in order: return
        for f in t._real_fields_: visit(f[1])
        order.append(t)
    for t in classes: visit(t)
    return order

def ctype(t):
    if t is ctypes.c_char: return "char"
    if t is ctypes.c_void_p: return "uint64_t"   # a pointer-sized field (NvP64)
    assert issubclass(t, ctypes._SimpleCData) and t._type_ in "bBhHiIlLqQ", t
    return f"{'' if t(-1).value < 0 else 'u'}int{8 * ctypes.sizeof(t)}_t"

def decl(t, name):
    dims = ""
    while issubclass(t, ctypes.Array): dims, t = dims + f"[{t._length_}]", t._type_
    return f"{t.__name__ if is_struct(t) else ctype(t)} {name}{dims};"

def layout(T):
    """T's members in declaration order: fields, bitfield containers (tinygrad reads and writes a bitfield's own
    ceil((bit_off + width) / 8) bytes at its offset, c.py:69), and anonymous unions (the unions tinygrad flattened: a
    field whose offset goes back to an earlier field's starts another alternative there): [{off, end, f | bits | alts}]."""
    items = []
    for f in T._real_fields_:
        name, t, off = f[:3]
        if len(f) > 3:
            end = off + -(-(f[3] + f[4]) // 8)
            box = next((i for i in items if "bits" in i and i["off"] < end and off < i["end"]), None)
            if box is None: items.append(box := {"off": off, "end": end, "bits": []})
            box["off"], box["end"] = min(box["off"], off), max(box["end"], end)
            box["bits"].append(f)
        else: items.append({"off": off, "end": off + ctypes.sizeof(t), "f": f})
    slots = []
    for it in items:
        if it["off"] >= max((s["end"] for s in slots), default=0): slots.append(it); continue
        assert "f" in it, f"{T.__name__}: a bitfield in a union"
        u = slots[-1] if slots and "alts" in slots[-1] and it["off"] >= slots[-1]["off"] else None
        if u is None:   # a union: its first alternative is the fields since the one it goes back to
            first = []
            while slots and slots[-1]["off"] >= it["off"]: first.insert(0, slots.pop())
            assert first and first[0]["off"] == it["off"] and all("f" in s for s in first), f"{T.__name__}: not a flattened union"
            slots.append(u := {"off": it["off"], "end": 0, "alts": [first, []]})
        elif it["off"] < u["alts"][-1][-1]["end"]:   # another alternative
            assert it["off"] == u["off"], f"{T.__name__}: not a flattened union"
            u["alts"].append([])
        u["alts"][-1].append(it)
        u["end"] = max(m["end"] for alt in u["alts"] for m in alt)
    return slots

def struct_lines(T, mod_file, aliases):
    pads, accessors = iter(range(1 << 20)), []
    def members(slots, pos, stop, ind):
        out = []
        for s in slots:
            assert s["off"] >= pos, f"{T.__name__}: overlapping members"
            if s["off"] > pos: out.append(f"{ind}uint8_t _pad{next(pads)}[{s['off'] - pos}];")
            if "f" in s: out.append(ind + decl(s["f"][1], s["f"][0]))
            elif "bits" in s:
                box = f"_bf_{s['off']}"
                out.append(f"{ind}uint8_t {box}[{s['end'] - s['off']}];  // " +
                           ", ".join(f"{f[0]}: {f[3]} bits at bit {f[4]} of byte {f[2]}" for f in s["bits"]))
                for name, t, off, width, boff in s["bits"]:
                    args = f"{box} + {off - s['off']}, {-(-(width + boff) // 8)}, {boff}, {width}"
                    accessors.append(f"    {ctype(t)} {name}() const {{ return ({ctype(t)})nv_bitfield_get({args}); }}")
                    accessors.append(f"    void set_{name}({ctype(t)} v) {{ nv_bitfield_set({args}, v); }}")
            else:
                out.append(f"{ind}union {{")
                for alt in s["alts"]:
                    if len(alt) == 1 and alt[0]["off"] == s["off"]: out += members(alt, s["off"], s["off"], ind + "    ")
                    else: out += [f"{ind}    struct {{"] + members(alt, s["off"], s["off"], ind + "        ") + [f"{ind}    }};"]
                out.append(f"{ind}}};")
            pos = max(pos, s["end"])
        if stop > pos: out.append(f"{ind}uint8_t _pad{next(pads)}[{stop - pos}];")
        return out
    body = members(layout(T), 0, T.SIZE, "    ")
    out = [f"struct {T.__name__} {{  // {mod_file}:{inspect.getsourcelines(T)[1]}"] + body + accessors + ["};"]
    out.append(f"static_assert(sizeof({T.__name__}) == {T.SIZE});")
    boxes = {f[0]: s["off"] for s in layout(T) if "bits" in s for f in s["bits"]}
    for f in T._real_fields_:
        if len(f) > 3: out.append(f"static_assert(offsetof({T.__name__}, _bf_{boxes[f[0]]}) + {f[2] - boxes[f[0]]} == {f[2]});  // {f[0]}")
        else: out.append(f"static_assert(offsetof({T.__name__}, {f[0]}) == {f[2]});")
    out += [f"using {a} = {T.__name__};" for a in aliases.get(T, [])]
    return out

def aliases_of(m):
    out = {}
    for k, v in vars(m).items():
        if is_struct(v) and k != v.__name__: out.setdefault(v, []).append(k)
    return out

# ── constants ─────────────────────────────────────────────────────────────────────────────────────────────────────────

def const_line(name, v, comment=""):
    assert is_int(v) and 0 <= v < 1 << 64, (name, v)
    return f"constexpr {'uint32_t' if v < 1 << 32 else 'uint64_t'} {name} = {v:#x};" + (f"  // {comment}" if comment else "")

def fill(items, first, rest, width=120):
    """The items, comma-separated, in lines of at most width columns, the first starting with first, the others with rest."""
    lines, cur = [], first
    for i, it in enumerate(items):
        it += "," if i < len(items) - 1 else ""
        if cur not in (first, rest) and len(cur) + 1 + len(it) > width: lines, cur = lines + [cur], rest
        cur += ("" if cur in (first, rest) else " ") + it
    return lines + [cur]

def comment(*paragraphs):
    """A block comment: strings wrapped to 116 columns, lists as given."""
    out = ["/*"]
    for i, p in enumerate(paragraphs):
        out += ([" *"] if i else []) + [f" * {line}" for line in (textwrap.wrap(p, 116) if isinstance(p, str) else p)]
    return out + [" */"]

# ── emit ──────────────────────────────────────────────────────────────────────────────────────────────────────────────

def main():
    outdir = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
    used, binds = used_set(), {chip: bind(chip) for chip in INCLUDES}
    try: commit = subprocess.run(["git", "-C", str(TINYGRAD), "rev-parse", "--short=9", "HEAD"], capture_output=True, text=True,
                                 check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError): commit = "(commit unknown)"
    reg_lines, regs, values = register_section(used, binds)

    split = {"boot": {"nv": [], "nv_c": []}, "rm": {"nv": [], "nv_c": [], "nv_gpu": [], "nv_gpu_c": []}}
    for name in sorted(used["nv"]):
        v = getattr(nv, name)
        if is_struct(v): split["boot" if name in BOOT_NV else "rm"]["nv"].append(v)
        else: split["rm" if name.startswith(RM_NV_PREFIXES) else "boot"]["nv_c"].append(name)
    for name in sorted(used["nv_gpu"]):
        split["rm"]["nv_gpu" if is_struct(getattr(nv_gpu, name)) else "nv_gpu_c"].append(name)
    boot_structs = closure(split["boot"]["nv"], set())
    rm_structs = closure(split["rm"]["nv"], set(boot_structs))
    gpu_structs = closure([getattr(nv_gpu, n) for n in split["rm"]["nv_gpu"]], set())
    nv_alias, gpu_alias = aliases_of(nv), aliases_of(nv_gpu)
    nfields = lambda ts: sum(len(t._real_fields_) for t in ts)   # noqa: E731
    src = lambda m: pathlib.Path(m.__file__).relative_to(TINYGRAD).as_posix()   # noqa: E731

    boot = comment(f"TinyGPUNVBootTables.h -- GENERATED by make_tinygpu_nv_boot_tables.py from tinygrad {commit} ({src(nv_regs).rsplit('/', 1)[0]}, "
                   f"{src(nv)} and {src(pci)}, as tinygrad/runtime/support/nv/ip.py, nvdev.py and {HELPER} use them). Do not edit. "
                   "TODO.md plan step C2.",
                   f"nv_regs: the boot's {len(regs) - len(GROUPS)} registers and {len(GROUPS)} MMU field groups as tinygrad's NVDev binds "
                   "them after each chip's include() sequence (kAdaRegs, kGB20xRegs, indexed by NVRegId; NVReg in TinyGPUNVReg.h reads and "
                   f"writes them), and the {len(values)} register values the boot uses. pci: the config offsets it uses. nv570: NVIDIA "
                   f"570.144 constants tinygrad's autogen lacks, as nv_init_helper.py uses them. nv: the boot's constants and "
                   f"{len(boot_structs)} structs ({nfields(boot_structs)} fields: VBIOS and falcon ucode, FWSECLIC, firmware headers, WPR "
                   "meta, libos, the message-queue framing, FSP/COT/FMC), packed at tinygrad's offsets with explicit padding, anonymous "
                   "unions where its offsets overlap, bitfield accessors, and its names and aliases. TinyGPUNVRMTables.h has the RPC "
                   "payloads and RM objects.",
                   ["include() sequences:"] + [line for chip, seq in INCLUDES.items()
                                               for line in fill([f"{m}{' ' + a if a else ''}" for m, a in seq], f"  {chip + ':':7}", " " * 9, 116)])
    boot += ["", "#ifndef LIBHMSBEAGLE_GPU_TINYGPUNVBOOTTABLES_H", "#define LIBHMSBEAGLE_GPU_TINYGPUNVBOOTTABLES_H", "",
             "#include <cstddef>", "#include <cstdint>", "", '#include "libhmsbeagle/GPU/TinyGPUNVReg.h"', "",
             "namespace tinygpu_device {", "namespace nv_regs {", ""] + reg_lines + ["", "} // namespace nv_regs", "", "namespace pci {"]
    boot += [const_line(n, getattr(pci, n)) for n in sorted(used["pci"])] + ["} // namespace pci", "", "namespace nv570 {",
             f"// open-gpu-kernel-modules 570.144, {NV570_SRC}; nv_init_helper.py's names in parentheses"]
    for name, v, why in NV570:
        boot.append(f'constexpr nv_regs::NVField {name} = {{"{name}", {v[0]}, {v[1]}}};  // {why}'
                    if isinstance(v, tuple) else const_line(name, v, why))
    boot += ["} // namespace nv570", "", "namespace nv {"] + [const_line(n, getattr(nv, n)) for n in split["boot"]["nv_c"]]
    boot += ["", "#pragma pack(push, 1)"]
    for T in boot_structs: boot += [""] + struct_lines(T, src(nv).rsplit("/", 1)[1], nv_alias)
    boot += ["", "#pragma pack(pop)", "", "} // namespace nv", "} // namespace tinygpu_device", "",
             "#endif // LIBHMSBEAGLE_GPU_TINYGPUNVBOOTTABLES_H"]

    rm = comment(f"TinyGPUNVRMTables.h -- GENERATED by make_tinygpu_nv_boot_tables.py from tinygrad {commit} ({src(nv)} and "
                 f"{src(nv_gpu)}, as tinygrad/runtime/support/nv/ip.py, {HELPER} and ops_nv.py's PCIIface path use them). Do not edit. "
                 "TODO.md plan step C2.",
                 f"nv: the RPC ids and results, and {len(rm_structs)} RPC payload structs ({nfields(rm_structs)} fields). nv_gpu (tinygrad's "
                 f"nv_570): the RM classes, controls and flags, and {len(gpu_structs)} RM structs ({nfields(gpu_structs)} fields). Packed "
                 "at tinygrad's offsets as in TinyGPUNVBootTables.h, which has the registers and the boot's structs.")
    rm += ["", "#ifndef LIBHMSBEAGLE_GPU_TINYGPUNVRMTABLES_H", "#define LIBHMSBEAGLE_GPU_TINYGPUNVRMTABLES_H", "", "#include <cstddef>",
           "#include <cstdint>", "", '#include "libhmsbeagle/GPU/TinyGPUNVBootTables.h"', "", "namespace tinygpu_device {", "namespace nv {", ""]
    rm += [const_line(n, getattr(nv, n)) for n in split["rm"]["nv_c"]] + ["", "#pragma pack(push, 1)"]
    for T in rm_structs: rm += [""] + struct_lines(T, src(nv).rsplit("/", 1)[1], nv_alias)
    rm += ["", "#pragma pack(pop)", "", "} // namespace nv", "", "namespace nv_gpu {"]
    rm += [const_line(n, getattr(nv_gpu, n)) for n in split["rm"]["nv_gpu_c"]] + ["", "#pragma pack(push, 1)"]
    for T in gpu_structs: rm += [""] + struct_lines(T, src(nv_gpu).rsplit("/", 1)[1], gpu_alias)
    rm += ["", "#pragma pack(pop)", "", "} // namespace nv_gpu", "} // namespace tinygpu_device", "", "#endif // LIBHMSBEAGLE_GPU_TINYGPUNVRMTABLES_H"]

    for name, lines in (("TinyGPUNVBootTables.h", boot), ("TinyGPUNVRMTables.h", rm)):
        (outdir / name).write_text("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
