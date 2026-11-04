#!/usr/bin/env python3
"""
Generate TinyGPUAMDBootTables.h (TODO.md plan steps A2b and N11): what the C++ port of tinygrad's AM boot (tinygrad/runtime/
support/am/amdev.py and ip.py at the pin, with ops_amd.py's PCIIface and AMDDevice paths) needs on each card of FAMILIES, its
register families: gfx11, the RX 7900 XT (GC 11.0.0, MP0 and MP1 13.0.0, SDMA 6.0.0, NBIO 4.3.0, MMHUB 3.0.0, OSSSYS 6.0.0,
HDP 6.0.0), and gfx12, the RX 9070 XT (GC 12.0.1, MP0 and MP1 14.0.3, SDMA 7.0.1, NBIF 6.3.1, MMHUB 4.1.0, OSSSYS 7.0.0, HDP
7.0.0), each IP set read from the card's captured discovery table, all of it taken from tinygrad's own autogen (plan
decision 9):
  - the registers, per family: every AMDev register name tinygrad's boot reaches on the card, recorded as the real daemon runs
    on fake_amd_device.py's card of that chip (FAKE_AMD_CHIP; tinygpu_tests/amd_boot_coverage.py: a cold boot, a partial
    boot, faults at fini, a mode1 reset), each bound as AMDev._build_regs binds it (amdev.py:398-409: the module whose name
    is last in its order, the IP whose bases it takes, its segment, offset and fields), and the names the boot asks for that
    the card has none of;
  - the chips: each card's PCI device ids, arch, family and IP versions (the plugin refuses another card before it sends it
    anything, plan step N1; the boot takes the entry whose IP versions it discovers, plan step N12);
  - the structs: the discovery table's (each card's gc_info version, from its captured table), the headers of each card's
    firmware blobs (their versions read from tinygrad's cache), the PSP's command and ring frame, and the v11 compute MQD,
    laid out by make_tinygpu_nv_boot_tables.py's emitter at tinygrad's recorded offsets; GC 12's v12 MQD is written through
    the v11 struct, with static_asserts on its size and on the offsets of the fields the boot writes (plan decision 7);
  - the constants: every am.* and pci.* integer amdev.py and ip.py use (an AST scan) or build at run time on these cards,
    hw_id_map, the name tables of the boot's log lines, the SMU messages and clocks the boot sends and the soc constants it
    uses, each equal in every family's module (smu_13_0_0 and smu_14_0_2, soc_11 and soc_12).
Rerun after changing the tinygrad pin, amdev.py, ip.py, the fake or a captured table, with the harness Python (BEAGLE_PYTHON
in tinygpu_tests/env.sh); tinygpu_tests/test_a2b_tables.py checks the output regenerates byte for byte:
    python make_tinygpu_amd_boot_tables.py [outdir]    # default: next to this script
"""
import ast, ctypes, json, os, pathlib, subprocess, sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "tinygpu_tests"))
import tgpaths  # noqa: E402
tgpaths.setup()   # the fake only; no download
import make_tinygpu_nv_boot_tables as nvgen  # noqa: E402  the struct emitter
from tinygrad.helpers import fetch_fw, mv_address  # noqa: E402
from tinygrad.runtime.autogen import pci  # noqa: E402
from tinygrad.runtime.autogen.am import am, fw  # noqa: E402
from tinygrad.runtime.support.amd import import_module, import_soc  # noqa: E402

TINYGRAD = pathlib.Path(tgpaths.TINYGRAD_PATH)
SCANNED = [TINYGRAD / "tinygrad/runtime/support/am/amdev.py", TINYGRAD / "tinygrad/runtime/support/am/ip.py"]
IP_KEYS = ["GC_HWIP", "MP0_HWIP", "MP1_HWIP", "SDMA0_HWIP", "NBIO_HWIP", "MMHUB_HWIP", "OSSSYS_HWIP", "HDP_HWIP"]   # the IPs the boot binds
# the register families (TODO.md plan step N11): each a captured card's (its discovery table in $BEAGLE_TINYGPU_DATA/discovery),
# its PCI device ids and the chip fake_amd_device.py plays it as (FAKE_AMD_CHIP), which is also its arch
FAMILIES = [dict(name="gfx11", chip="gfx1100", pci_ids=[0x744c], card="the RX 7900 XT (STATUS.md R64)"),
            dict(name="gfx12", chip="gfx1201", pci_ids=[0x7550], card="the RX 9070 XT (STATUS.md R101)")]
def card_table(fam):
    """The family's captured discovery table: (.bin path, its meta)."""
    paths = sorted((tgpaths.DATA / "discovery").glob(f"1002_{fam['pci_ids'][0]:04x}_*.json"))
    assert len(paths) == 1, f"{fam['name']}: one captured table expected, found {paths}"
    return paths[0].with_suffix(".bin"), json.load(open(paths[0]))
for _f in FAMILIES: _f["ip"] = {k: tuple(card_table(_f)[1]["ip_ver"][k]) for k in IP_KEYS}
def mods(fam):
    """AMDev._build_regs' modules in its order (amdev.py:399-409): (prefix, hwip, version); a later module's name replaces an earlier one's."""
    ip = fam["ip"]
    return [("mp", "MP0_HWIP", ip["MP0_HWIP"]), ("hdp", "HDP_HWIP", ip["HDP_HWIP"]), ("gc", "GC_HWIP", ip["GC_HWIP"]),
            ("mmhub", "MMHUB_HWIP", ip["MMHUB_HWIP"]), ("osssys", "OSSSYS_HWIP", ip["OSSSYS_HWIP"]),
            ("nbio" if ip["GC_HWIP"] < (12, 0, 0) else "nbif", "NBIO_HWIP", ip["NBIO_HWIP"]), ("mp", "MP1_HWIP", (11, 0, 0))]
# names the code builds at run time (getattr(am, f"...")) on these cards' branches
DYNAMIC = ["GFX_FW_TYPE_RS64_MEC", "GFX_FW_TYPE_RS64_MEC_P0_STACK", "GFX_FW_TYPE_RLC_IRAM", "GFX_FW_TYPE_RLC_DRAM_BOOT",   # ip.py:78-81, amdev.py:100-107
           "GFX_FW_TYPE_RLC_P", "GFX_FW_TYPE_RLC_V",
           "GFX_FW_TYPE_RS64_PFP", "GFX_FW_TYPE_RS64_PFP_P0_STACK", "GFX_FW_TYPE_RS64_ME", "GFX_FW_TYPE_RS64_ME_P0_STACK"]   # GC 12 (amdev.py:67-82)
SKIP = {"hw_id_map", "enum_psp_fw_type", "enum_psp_gfx_fw_type", "enum_soc15_ih_clientid", "enum_soc21_ih_clientid"}   # emitted as tables
SMU_NAMES = ["PPSMC_MSG_SetDriverDramAddrHigh", "PPSMC_MSG_SetDriverDramAddrLow", "PPSMC_MSG_EnableAllSmuFeatures", "PPSMC_MSG_GetSmuVersion",
             "PPSMC_MSG_GetDpmFreqByIndex", "PPSMC_MSG_SetSoftMinByFreq", "PPSMC_MSG_SetSoftMaxByFreq", "PPSMC_MSG_SetPptLimit",
             "PPCLK_UCLK", "PPCLK_FCLK", "PPCLK_SOCCLK", "PPCLK_GFXCLK"]   # AM_SMU's on these cards' branches (ip.py:194-265)
SOC_NAMES = ["MTYPE_UC", "SH_MEM_ADDRESS_MODE_64", "SH_MEM_ALIGNMENT_MODE_UNALIGNED"]   # ip.py:150, 178-183, 306-307
MQD_FIELDS = ["header", "cp_mqd_base_addr_lo", "cp_mqd_base_addr_hi", "cp_hqd_pipe_priority", "cp_hqd_queue_priority", "cp_hqd_quantum",
              "cp_hqd_persistent_state", "cp_hqd_pq_base_lo", "cp_hqd_pq_base_hi", "cp_hqd_pq_rptr_report_addr_lo", "cp_hqd_pq_rptr_report_addr_hi",
              "cp_hqd_pq_wptr_poll_addr_lo", "cp_hqd_pq_wptr_poll_addr_hi", "cp_hqd_pq_doorbell_control", "cp_hqd_pq_control", "cp_hqd_ib_control",
              "cp_hqd_hq_status0", "cp_mqd_control", "cp_hqd_vmid", "cp_hqd_aql_control", "cp_hqd_eop_base_addr_lo", "cp_hqd_eop_base_addr_hi",
              "cp_hqd_eop_control"] + [f"compute_static_thread_mgmt_se{i}" for i in range(8)]   # AM_GFX.setup_ring's, one XCC (ip.py:345-363)

def scanned():
    """am.* and pci.* names amdev.py and ip.py use."""
    used = {"am": set(), "pci": set()}
    for f in SCANNED:
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in used: used[node.value.id].add(node.attr)
    return used

def coverage(fam):
    """amd_boot_coverage.coverage() on the family's fake card, in a process of its own (fake_am_gpu reads FAKE_AMD_CHIP at import)."""
    r = subprocess.run([sys.executable, "-c", "import json, amd_boot_coverage as c; print(json.dumps(c.coverage()))"], cwd=HERE / "tinygpu_tests",
                       env={**os.environ, "FAKE_AMD_CHIP": fam["chip"]}, capture_output=True, text=True)
    assert r.returncode == 0, f"{fam['name']}: the coverage sessions failed: {r.stderr[-1500:]}"
    return json.loads(r.stdout.splitlines()[-1])

def blob_header(name):
    blob = memoryview(bytearray(fetch_fw("amdgpu", name, fw.hashes[name])))   # tinygrad's cache: the network is off (tgpaths)
    chdr = am.struct_common_firmware_header.from_address(mv_address(blob))
    return blob, (chdr.header_version_major, chdr.header_version_minor)

def versioned_base(name):
    """The versioned header AMFirmware.load_fw reads a blob through (amdev.py:33-70), or None (IMU and RLC: fixed headers)."""
    if name.startswith("psp_"): return "struct_psp_firmware_header"
    if name.startswith("smu_"): return "struct_smc_firmware_header"
    if name.startswith("sdma_"): return "struct_sdma_firmware_header"
    if name.endswith(("_pfp.bin", "_me.bin", "_mec.bin")): return "struct_gfx_firmware_header"
    return None

def firmware_rows(fam):
    """The fetch_fw calls tinygrad's AMFirmware makes on the family's card (amdev.py:25-119), in order: (path, name, sha256, url).
    The blobs come from tinygrad's cache (AMFirmware parses each before it asks for the next); the network is off."""
    from tinygrad import helpers
    from tinygrad.runtime.support.am import amdev
    calls, urls = [], []
    real_fetch_fw, real_fetch = amdev.fetch_fw, helpers.fetch
    def fetch_fw(path, name, sha256):
        calls.append((path, name, sha256))
        return real_fetch_fw(path, name, sha256)
    def fetch(url, *a, **k):
        urls.append(url)
        return real_fetch(url, *a, **k)
    amdev.fetch_fw, helpers.fetch = fetch_fw, fetch
    try:
        import types
        hw = {getattr(am, k): v for k, v in fam["ip"].items()}
        amdev.AMFirmware(types.SimpleNamespace(ip_ver=hw, devfmt="usb4"))
    finally: amdev.fetch_fw, helpers.fetch = real_fetch_fw, real_fetch
    assert len(calls) == len(urls), (calls, urls)
    return [(*c, u) for c, u in zip(calls, urls)]

def register_lines(fam, cov):
    """The family's used registers, sorted by name, as AMDev binds them: {name, hwip, segment, offset, fields}."""
    binds = {}
    for prefix, hwip, ver in mods(fam):
        mod = import_module(prefix, ver, submod="regs")
        for name, (off, seg, fields) in mod.items(): binds[name] = (hwip, seg, off, fields, f"{prefix} {'.'.join(map(str, ver))}")
    lines, nf = [f"namespace {fam['name']} {{  // {fam['card']}"], 0
    for name in cov["used"]:
        hwip, seg, off, fields, mod = binds[name]
        if fields: lines += nvgen.fill([f'{{"{f}", {s}, {e}}}' for f, (s, e) in fields.items()], f"inline constexpr AMField kF_{name}[] = {{", "    ")
        if fields: lines[-1] += "};"
        nf += len(fields)
    lines += ["", f"// {len(cov['used'])} registers, sorted by name (find_reg's binary search): name, IP, segment, offset, fields; the module",
              "inline constexpr AMRegDef kRegs[] = {"]
    for name in cov["used"]:
        hwip, seg, off, fields, mod = binds[name]
        lines.append(f'    {{"{name}", {getattr(am, hwip)}, {seg}, {off:#x}, {f"kF_{name}" if fields else "nullptr"}, {len(fields)}}},  // {hwip} {mod}')
    assert cov["absent"], "an empty kAbsent would be a zero-length array"
    lines += ["};", "// asked for and absent on this card (hasattr false): the boot's has_reg checks"]
    lines += ["inline constexpr const char* kAbsent[] = {" + ", ".join(f'"{n}"' for n in cov["absent"]) + "};", f"}} // namespace {fam['name']}", ""]
    return lines, nf

def equal_in(mods_, names, what):
    """{name: value}, each equal in every module."""
    vals = {n: getattr(mods_[0], n) for n in names}
    for m in mods_[1:]:
        bad = [n for n in names if getattr(m, n) != vals[n]]
        assert not bad, f"{what}: {bad} differ between {mods_[0].__name__} and {m.__name__}"
    return vals

def main():
    outdir = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
    try: commit = subprocess.run(["git", "-C", str(TINYGRAD), "rev-parse", "--short=9", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError): commit = "(commit unknown)"
    used = scanned()
    reg_lines, desc, versioned, fwrows = [], [], [], []
    for fam in FAMILIES:
        cov = coverage(fam)
        bad = [s for s in cov["sessions"] if s[1] != "NO ERRORS"]
        assert not bad, f"{fam['name']}: the coverage sessions saw errors: {bad}"
        lines, nfields = register_lines(fam, cov)
        reg_lines += lines
        # the versioned structs this card's discovery table and blobs select
        table = open(card_table(fam)[0], "rb").read()
        bhdr = am.struct_binary_header.from_buffer(bytearray(table))
        gc_off = bhdr.table_list[am.GC].offset
        gc_hdr = am.struct_gc_info_v1_0.from_buffer(bytearray(table[gc_off:gc_off + ctypes.sizeof(am.struct_gc_info_v1_0)]))
        versioned.append(f"struct_gc_info_v{gc_hdr.header.version_major}_{gc_hdr.header.version_minor}")
        rows, vers = firmware_rows(fam), []
        for path, name, sha, url in rows:
            _, v = blob_header(name)
            vers.append(f"{name} v{v[0]}.{v[1]}")
            if (base := versioned_base(name)): versioned.append(f"{base}_v{v[0]}_{v[1]}")
        fwrows += [(fam["chip"], *r) for r in rows]
        desc += [f"{fam['name']}, {fam['card']}, PCI device ids {', '.join(f'{d:04x}' for d in fam['pci_ids'])}: {len(cov['used'])} registers "
                 f"({nfields} fields), absent {', '.join(cov['absent'])}; IP versions " + ", ".join(f"{k} {'.'.join(map(str, v))}" for k, v in fam["ip"].items()) +
                 f"; gc_info v{gc_hdr.header.version_major}.{gc_hdr.header.version_minor}; firmware headers " + ", ".join(vers) +
                 "; the coverage sessions " + ", ".join(f"{label} ({verdict})" for label, verdict, _ in cov["sessions"])]

    names = sorted(n for n in used["am"] if n.startswith("struct_")) + versioned + ["struct_v11_compute_mqd"]
    structs = nvgen.closure([getattr(am, n) for n in dict.fromkeys(names)], set())
    consts = sorted({n for n in used["am"] | set(DYNAMIC) if nvgen.is_int(getattr(am, n, None)) and n not in SKIP})
    smus = list(dict.fromkeys(import_module("smu", f["ip"]["MP1_HWIP"]) for f in FAMILIES))
    socs = list(dict.fromkeys(import_soc(f["ip"]["GC_HWIP"]) for f in FAMILIES))
    smu_vals, soc_vals = equal_in(smus, SMU_NAMES, "the SMU names"), equal_in(socs, SOC_NAMES, "the soc names")
    mca = any(hasattr(m, "PPSMC_MSG_QueryValidMcaCount") for m in smus)
    psp_types = sorted({v for k, v in vars(am).items() if k.startswith("PSP_FW_TYPE_") and nvgen.is_int(v)})
    gfx_types = sorted({v for k, v in vars(am).items() if k.startswith("GFX_FW_TYPE_") and nvgen.is_int(v)})
    mqd_checks = []   # GC 12's struct_v12_compute_mqd against the v11 struct the boot fills (plan decision 7)
    for f in FAMILIES:
        if f["ip"]["GC_HWIP"][0] == 11: continue
        T = getattr(am, f"struct_v{f['ip']['GC_HWIP'][0]}_compute_mqd")
        assert ctypes.sizeof(T) == ctypes.sizeof(am.struct_v11_compute_mqd), f"{T.__name__}'s size"
        for n in MQD_FIELDS: assert getattr(T, n).offset == getattr(am.struct_v11_compute_mqd, n).offset, f"{T.__name__}.{n}'s offset"
        mqd_checks += [f"static_assert(sizeof(struct_v11_compute_mqd) == {ctypes.sizeof(T)}, \"{T.__name__}'s size\");"]
        mqd_checks += [f"static_assert(offsetof(struct_v11_compute_mqd, {n}) == {getattr(T, n).offset:#x}, \"{T.__name__}.{n}\");" for n in MQD_FIELDS]

    out = nvgen.comment(
        f"TinyGPUAMDBootTables.h -- GENERATED by make_tinygpu_amd_boot_tables.py from tinygrad {commit} (tinygrad/runtime/autogen/am, "
        "as tinygrad/runtime/support/am/amdev.py and ip.py use them on each card of a register family). Do not edit. TODO.md plan steps "
        "A2b and N11.",
        "am::regs: per family, the boot's registers, bound as AMDev._build_regs binds them on its card, and the names it asks for that "
        "the card has none of; kFamilies lists them. A register's address on an instance is the IP's discovered base for its segment "
        "plus its offset (AMDReg.__post_init__). am: the boot's constants, the cards (kChips: PCI device id, arch, family, IP versions), "
        f"hw_id_map, the log lines' name tables, and {len(structs)} structs ({sum(len(t._real_fields_) for t in structs)} fields: the "
        "discovery table, the cards' firmware headers, the PSP command and ring frame, the v11 compute MQD, which GC 12's v12 matches "
        "where the boot writes it) at tinygrad's offsets. am::smu and am::soc: the SMU messages and clocks and the soc constants the "
        f"boot uses, equal in {', '.join(pathlib.Path(m.__file__).stem for m in smus)} and in {', '.join(pathlib.Path(m.__file__).stem for m in socs)}. "
        "am::fw: each card's firmware, as AMFirmware fetches it.",
        desc)
    out += ["", "#ifndef LIBHMSBEAGLE_GPU_TINYGPUAMDBOOTTABLES_H", "#define LIBHMSBEAGLE_GPU_TINYGPUAMDBOOTTABLES_H", "",
            "#include <cstddef>", "#include <cstdint>", "", '#include "libhmsbeagle/GPU/TinyGPUFirmwareManifest.h"   // its TGFirmware entries',
            '#include "libhmsbeagle/GPU/TinyGPUNVReg.h"   // nv_bitfield_get/set (c.py\'s bitfields)', "",
            "namespace tinygpu_device {", "namespace am {", "", "namespace regs {",
            "struct AMField { const char* name; uint8_t start, end; };",
            "struct AMRegDef { const char* name; uint8_t hwip, segment; uint32_t offset; const AMField* fields; uint8_t nfields; };", ""] + reg_lines
    out += ["// the register families: their registers (sorted by name) and the names the boot asks for that their cards lack",
            "struct Family { const char* name; const AMRegDef* regs; size_t nregs; const char* const* absent; size_t nabsent; };",
            "inline constexpr Family kFamilies[] = {"]
    out += [f'    {{"{f["name"]}", {f["name"]}::kRegs, sizeof({f["name"]}::kRegs) / sizeof({f["name"]}::kRegs[0]), {f["name"]}::kAbsent, '
            f'sizeof({f["name"]}::kAbsent) / sizeof({f["name"]}::kAbsent[0])}},' for f in FAMILIES]
    out += ["};", "} // namespace regs"]
    out += ["", "// am.*"] + [nvgen.const_line(n, getattr(am, n)) for n in consts]
    out += ["", "// the cards the tables are for: PCI device id, the arch that names their HSACOs and firmware rows, their register family",
            "// (regs::kFamilies) and the IP versions their discovery table holds, in kChipIP's order. The plugin refuses any other AMD card",
            "// before it sends the card anything (TODO.md plan step N1); the boot takes the entry whose IP versions it discovers (N12).",
            "constexpr uint32_t kChipIP[] = {" + ", ".join(IP_KEYS) + "};",
            "struct Chip { uint16_t device_id; const char* arch; uint8_t family; uint8_t ip[8][3]; };", "constexpr Chip kChips[] = {"]
    out += [f'    {{{d:#06x}, "{f["chip"]}", {i}, {{' + ", ".join(f"{{{v[0]}, {v[1]}, {v[2]}}}" for v in f["ip"].values()) + "}},"
            for i, f in enumerate(FAMILIES) for d in f["pci_ids"]]
    out += ["};"]
    out += ["", "// hw_id_map: an IP's hardware id (0: none), by IP"]
    out += [f"constexpr uint16_t hw_id_map[{am.MAX_HWIP}] = {{" + ", ".join(str(am.hw_id_map.get(i, 0)) for i in range(am.MAX_HWIP)) + "};"]
    out += [f"constexpr bool hw_id_mapped[{am.MAX_HWIP}] = {{" + ", ".join("true" if i in am.hw_id_map else "false" for i in range(am.MAX_HWIP)) + "};"]
    inv = {hw_id: hw_ip for hw_ip, hw_id in am.hw_id_map.items()}   # amdev.py:385, the dict's own order: the last IP of an id wins
    out += ["// inv_hw_id = {hw_id: hw_ip for hw_ip, hw_id in hw_id_map.items()}: the harvest table's lookup, by hardware id",
            "struct HwIdIp { uint32_t hw_id, hw_ip; };"]
    out += nvgen.fill([f"{{{k}, {v}}}" for k, v in sorted(inv.items())], "constexpr HwIdIp inv_hw_id[] = {", "    ")
    out[-1] += "};"
    out += ["", "struct Name { uint32_t id; const char* name; };", "// the log lines' names: enum_psp_fw_type, enum_psp_gfx_fw_type"]
    out += nvgen.fill([f'{{{v}, "{am.enum_psp_fw_type[v]}"}}' for v in psp_types if v in am.enum_psp_fw_type], "constexpr Name enum_psp_fw_type[] = {", "    ")
    out[-1] += "};"
    out += nvgen.fill([f'{{{v}, "{am.enum_psp_gfx_fw_type[v]}"}}' for v in gfx_types if v in am.enum_psp_gfx_fw_type], "constexpr Name enum_psp_gfx_fw_type[] = {", "    ")
    out[-1] += "};"
    out += ["", f"namespace smu {{  // {', '.join(pathlib.Path(m.__file__).name for m in smus)}: the boot's messages and clocks, equal in each"]
    out += [nvgen.const_line(n, smu_vals[n]) for n in SMU_NAMES]
    out += [f"constexpr bool has_PPSMC_MSG_QueryValidMcaCount = {'true' if mca else 'false'};", "} // namespace smu"]
    out += ["", f"namespace soc {{  // {', '.join(pathlib.Path(m.__file__).name for m in socs)}: equal in each"] + [nvgen.const_line(n, soc_vals[n]) for n in SOC_NAMES]
    out += ["} // namespace soc", "", "namespace pci {"] + [nvgen.const_line(n, getattr(pci, n)) for n in sorted(used["pci"])] + ["} // namespace pci"]
    from tinygrad.runtime.autogen import hsa
    out += ["", "namespace hsa {  // AMDDevice.create_queue's gart offsets (ops_amd.py:1109): hsa.amd_queue_t's"]
    out += [nvgen.const_line(f"amd_queue_t_{f}_offset", getattr(hsa.amd_queue_t, f).offset) for f in ("read_dispatch_id", "write_dispatch_id")]
    out += ["} // namespace hsa"]
    import hashlib
    out += ["", "namespace fw {  // AMFirmware's fetch_fw calls on each card, in order (TinyGPUFirmware.h finds and checks them)"]
    out += [f'constexpr nvfw::TGFirmware kFirmware[] = {{']
    out += [f'    {{"{arch}", "{name}", "{path}", "{name}", "{sha}", "{hashlib.md5(url.encode()).hexdigest()}"}},' for arch, path, name, sha, url in fwrows]
    out += ["};", "} // namespace fw"]
    out += ["", "#pragma pack(push, 1)"]
    aliases = nvgen.aliases_of(am)
    for T in structs: out += [""] + nvgen.struct_lines(T, "am.py", aliases)
    out += ["", "#pragma pack(pop)"]
    if mqd_checks:
        out += ["", "// GC 12's AM_GFX.setup_ring fills a struct_v12_compute_mqd (ip.py:345); the boot fills struct_v11_compute_mqd instead",
                "// (plan decision 7): the same size, and every field it writes at the same offset"] + mqd_checks
    out += ["", "} // namespace am", "} // namespace tinygpu_device", "", "#endif // LIBHMSBEAGLE_GPU_TINYGPUAMDBOOTTABLES_H"]
    (outdir / "TinyGPUAMDBootTables.h").write_text("\n".join(out) + "\n")
    print(f"TinyGPUAMDBootTables.h: {', '.join(f['name'] for f in FAMILIES)} register families, {len(consts)} constants, {len(structs)} structs")

if __name__ == "__main__":
    main()
