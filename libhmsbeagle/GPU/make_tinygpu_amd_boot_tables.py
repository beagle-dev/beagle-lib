#!/usr/bin/env python3
"""
Generate TinyGPUAMDBootTables.h (TODO.md plan step A2b): what the C++ port of tinygrad's AM boot (tinygrad/runtime/support/
am/amdev.py and ip.py at the pin, with ops_amd.py's PCIIface and AMDDevice paths) needs on the RX 7900 XT (GC 11.0.0, MP0
and MP1 13.0.0, SDMA 6.0.0, NBIO 4.3.0, MMHUB 3.0.0, OSSSYS 6.0.0, HDP 6.0.0), taken from tinygrad's own autogen (plan
decision 9):
  - the registers: every AMDev register name tinygrad's boot reaches on this card, recorded as the real daemon runs on
    fake_amd_device.py's card (tinygpu_tests/amd_boot_coverage.py: a cold boot, a partial boot, faults at fini, a mode1
    reset), each bound as AMDev._build_regs binds it (amdev.py:398-409: the module whose name is last in its order, the IP
    whose bases it takes, its segment, offset and fields), and the names the boot asks for that the card has none of;
  - the structs: the discovery table's (its gc_info version from the captured table), the headers of this card's six
    firmware blobs (their versions read from tinygrad's cache), the PSP's command and ring frame, and the v11 compute MQD,
    laid out by make_tinygpu_nv_boot_tables.py's emitter at tinygrad's recorded offsets;
  - the constants: every am.* and pci.* integer amdev.py and ip.py use (an AST scan) or build at run time on this card,
    hw_id_map, the name tables of the boot's log lines, smu_13_0_0's messages and clocks, and soc_11's.
Rerun after changing the tinygrad pin, amdev.py, ip.py or the fake, with the harness Python (BEAGLE_PYTHON in
tinygpu_tests/env.sh); tinygpu_tests/test_a2b_tables.py checks the output regenerates byte for byte:
    python make_tinygpu_amd_boot_tables.py [outdir]    # default: next to this script
"""
import ast, ctypes, functools, json, os, pathlib, subprocess, sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "tinygpu_tests"))
import tgpaths  # noqa: E402
tgpaths.setup()   # the fake only; no download
import make_tinygpu_nv_boot_tables as nvgen  # noqa: E402  the struct emitter
from tinygrad.helpers import fetch_fw, mv_address  # noqa: E402
from tinygrad.runtime.autogen import pci  # noqa: E402
from tinygrad.runtime.autogen.am import am, fw  # noqa: E402
from tinygrad.runtime.support.amd import import_module  # noqa: E402

TINYGRAD = pathlib.Path(tgpaths.TINYGRAD_PATH)
SCANNED = [TINYGRAD / "tinygrad/runtime/support/am/amdev.py", TINYGRAD / "tinygrad/runtime/support/am/ip.py"]
IP = {"GC_HWIP": (11, 0, 0), "MP0_HWIP": (13, 0, 0), "MP1_HWIP": (13, 0, 0), "SDMA0_HWIP": (6, 0, 0), "NBIO_HWIP": (4, 3, 0),
      "MMHUB_HWIP": (3, 0, 0), "OSSSYS_HWIP": (6, 0, 0), "HDP_HWIP": (6, 0, 0)}   # the card's (STATUS.md R64)
PCI_IDS = [0x744c]   # its PCI device ids (TODO.md plan step N1: the plugin refuses any other AMD card before touching it)
# AMDev._build_regs' modules in its order (amdev.py:399-409); a later module's name replaces an earlier one's
MODS = [("mp", "MP0_HWIP"), ("hdp", "HDP_HWIP"), ("gc", "GC_HWIP"), ("mmhub", "MMHUB_HWIP"), ("osssys", "OSSSYS_HWIP"), ("nbio", "NBIO_HWIP"),
        ("mp", "MP1_HWIP", (11, 0, 0))]
# names the code builds at run time (getattr(am, f"...")) on this card's branches
DYNAMIC = ["GFX_FW_TYPE_RS64_MEC", "GFX_FW_TYPE_RS64_MEC_P0_STACK", "GFX_FW_TYPE_RLC_IRAM", "GFX_FW_TYPE_RLC_DRAM_BOOT",   # ip.py:78-81, amdev.py:100-107
           "GFX_FW_TYPE_RLC_P", "GFX_FW_TYPE_RLC_V"]
SKIP = {"hw_id_map", "enum_psp_fw_type", "enum_psp_gfx_fw_type", "enum_soc15_ih_clientid", "enum_soc21_ih_clientid"}   # emitted as tables
FW = {"sos": "psp_13_0_0_sos.bin", "smu": "smu_13_0_0.bin", "sdma": "sdma_6_0_0.bin", "mec": "gc_11_0_0_mec.bin", "imu": "gc_11_0_0_imu.bin",
      "rlc": "gc_11_0_0_rlc.bin"}   # AMFirmware's files on this card (amdev.py:33-109)

def scanned():
    """am.* and pci.* names amdev.py and ip.py use."""
    used = {"am": set(), "pci": set()}
    for f in SCANNED:
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in used: used[node.value.id].add(node.attr)
    return used

def blob_header(name):
    blob = memoryview(bytearray(fetch_fw("amdgpu", name, fw.hashes[name])))   # tinygrad's cache: the network is off (tgpaths)
    chdr = am.struct_common_firmware_header.from_address(mv_address(blob))
    return blob, (chdr.header_version_major, chdr.header_version_minor)

def firmware_rows():
    """The fetch_fw calls tinygrad's AMFirmware makes on this card (amdev.py:25-119), in order: (path, name, sha256, url).
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
        hw = {getattr(am, k): v for k, v in IP.items()}
        amdev.AMFirmware(types.SimpleNamespace(ip_ver=hw, devfmt="usb4"))
    finally: amdev.fetch_fw, helpers.fetch = real_fetch_fw, real_fetch
    assert len(calls) == len(urls) == len(FW), (calls, urls)
    return [(*c, u) for c, u in zip(calls, urls)]

def register_lines(cov):
    """The used registers, sorted by name, as AMDev binds them: {name, hwip, segment, offset, fields}."""
    binds = {}
    for m in MODS:
        prefix, hwip = m[0], m[1]
        ver = m[2] if len(m) > 2 else IP[hwip]
        mod = import_module(prefix, ver, submod="regs")
        modname = next(k for k, v in vars(sys.modules["tinygrad.runtime.autogen.am.regs"]).items() if v is mod) if False else None
        for name, (off, seg, fields) in mod.items(): binds[name] = (hwip, seg, off, fields, f"{prefix} {'.'.join(map(str, ver))}")
    lines, nf = ["struct AMField { const char* name; uint8_t start, end; };",
                 "struct AMRegDef { const char* name; uint8_t hwip, segment; uint32_t offset; const AMField* fields; uint8_t nfields; };", ""], 0
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
    lines += ["};", "// asked for and absent on this card (hasattr false): the boot's has_reg checks"]
    lines += ["inline constexpr const char* kAbsent[] = {" + ", ".join(f'"{n}"' for n in cov["absent"]) + "};"]
    return lines, nf

def main():
    outdir = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
    try: commit = subprocess.run(["git", "-C", str(TINYGRAD), "rev-parse", "--short=9", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError): commit = "(commit unknown)"
    import amd_boot_coverage
    cov = amd_boot_coverage.coverage()
    bad = [s for s in cov["sessions"] if s[1] != "NO ERRORS"]
    assert not bad, f"the coverage sessions saw errors: {bad}"
    used = scanned()
    reg_lines, nfields = register_lines(cov)

    # structs: fixed ones, plus the versioned ones this card's discovery table and blobs select
    table, meta = open(sorted((tgpaths.DATA / "discovery").glob(f"1002_{PCI_IDS[0]:04x}_*.bin"))[0], "rb").read(), None
    bhdr = am.struct_binary_header.from_buffer(bytearray(table))
    gc_off = bhdr.table_list[am.GC].offset
    gc_hdr = am.struct_gc_info_v1_0.from_buffer(bytearray(table[gc_off:gc_off + ctypes.sizeof(am.struct_gc_info_v1_0)]))
    versioned = [f"struct_gc_info_v{gc_hdr.header.version_major}_{gc_hdr.header.version_minor}"]
    vers = {}
    for key, fname, base in (("sos", FW["sos"], "struct_psp_firmware_header"), ("smu", FW["smu"], "struct_smc_firmware_header"),
                             ("sdma", FW["sdma"], "struct_sdma_firmware_header"), ("mec", FW["mec"], "struct_gfx_firmware_header")):
        _, vers[key] = blob_header(fname)
        versioned.append(f"{base}_v{vers[key][0]}_{vers[key][1]}")
    for key in ("imu", "rlc"): _, vers[key] = blob_header(FW[key])
    names = sorted(n for n in used["am"] if n.startswith("struct_")) + versioned + [f"struct_v{IP['GC_HWIP'][0]}_compute_mqd"]
    structs = nvgen.closure([getattr(am, n) for n in dict.fromkeys(names)], set())
    consts = sorted({n for n in used["am"] | set(DYNAMIC) if nvgen.is_int(getattr(am, n, None)) and n not in SKIP})
    smu = import_module("smu", IP["MP1_HWIP"])
    soc = getattr(am, f"soc_{IP['GC_HWIP'][0]}") if hasattr(am, f"soc_{IP['GC_HWIP'][0]}") else None
    from tinygrad.runtime.support.amd import import_soc
    soc = import_soc(IP["GC_HWIP"])
    smu_names = sorted(k for k in vars(smu) if k.startswith(("PPSMC_MSG_", "PPCLK_")) and nvgen.is_int(getattr(smu, k)))
    psp_types = sorted({v for k, v in vars(am).items() if k.startswith("PSP_FW_TYPE_") and nvgen.is_int(v)})
    gfx_types = sorted({v for k, v in vars(am).items() if k.startswith("GFX_FW_TYPE_") and nvgen.is_int(v)})

    out = nvgen.comment(
        f"TinyGPUAMDBootTables.h -- GENERATED by make_tinygpu_amd_boot_tables.py from tinygrad {commit} (tinygrad/runtime/autogen/am, "
        "as tinygrad/runtime/support/am/amdev.py and ip.py use them on the RX 7900 XT). Do not edit. TODO.md plan step A2b.",
        f"am::regs: the boot's {len(cov['used'])} registers ({nfields} fields), bound as AMDev._build_regs binds them, and the "
        f"{len(cov['absent'])} names it asks for that the card has none of. A register's address on an instance is the IP's "
        "discovered base for its segment plus its offset (AMDReg.__post_init__). am: the boot's constants, hw_id_map, the log "
        f"lines' name tables, and {len(structs)} structs ({sum(len(t._real_fields_) for t in structs)} fields: the discovery "
        "table, this card's firmware headers, the PSP command and ring frame, the v11 compute MQD) at tinygrad's offsets. "
        "am::smu13 and am::soc11: smu_13_0_0's and soc_11's. am::kChips: the cards (PCI device ids) the tables are for.",
        ["the card's IP versions: " + ", ".join(f"{k} {'.'.join(map(str, v))}" for k, v in IP.items()),
         "its firmware headers: " + ", ".join(f"{FW[k]} v{v[0]}.{v[1]}" for k, v in vers.items()),
         "the coverage sessions: " + ", ".join(f"{label} ({verdict})" for label, verdict, _ in cov["sessions"])])
    out += ["", "#ifndef LIBHMSBEAGLE_GPU_TINYGPUAMDBOOTTABLES_H", "#define LIBHMSBEAGLE_GPU_TINYGPUAMDBOOTTABLES_H", "",
            "#include <cstddef>", "#include <cstdint>", "", '#include "libhmsbeagle/GPU/TinyGPUFirmwareManifest.h"   // its TGFirmware entries',
            '#include "libhmsbeagle/GPU/TinyGPUNVReg.h"   // nv_bitfield_get/set (c.py\'s bitfields)', "",
            "namespace tinygpu_device {", "namespace am {", "", "namespace regs {"] + reg_lines + ["} // namespace regs", ""]
    out += ["// the IP versions the tables are for"] + [f"constexpr uint8_t kIP_{k}[3] = {{{v[0]}, {v[1]}, {v[2]}}};" for k, v in IP.items()]
    arch = "gfx%d%x%x" % IP["GC_HWIP"]
    out += ["", "// the cards the tables are for: PCI device id, and the arch that names their HSACOs and firmware rows (the plugin",
            "// refuses any other AMD card before it sends the card anything: TODO.md plan step N1)",
            "struct Chip { uint16_t device_id; const char* arch; };",
            "constexpr Chip kChips[] = {" + ", ".join(f'{{{d:#06x}, "{arch}"}}' for d in PCI_IDS) + "};"]
    out += ["", "// am.*"] + [nvgen.const_line(n, getattr(am, n)) for n in consts]
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
    out += ["", f"namespace smu13 {{  // {pathlib.Path(smu.__file__).name}"] + [nvgen.const_line(n, getattr(smu, n)) for n in smu_names]
    out += [f"constexpr bool has_PPSMC_MSG_QueryValidMcaCount = {'true' if hasattr(smu, 'PPSMC_MSG_QueryValidMcaCount') else 'false'};", "} // namespace smu13"]
    out += ["", f"namespace soc11 {{  // {pathlib.Path(soc.__file__).name}"] + [nvgen.const_line(n, getattr(soc, n)) for n in ("MTYPE_UC", "SH_MEM_ADDRESS_MODE_64", "SH_MEM_ALIGNMENT_MODE_UNALIGNED")]
    out += ["} // namespace soc11", "", "namespace pci {"] + [nvgen.const_line(n, getattr(pci, n)) for n in sorted(used["pci"])] + ["} // namespace pci"]
    from tinygrad.runtime.autogen import hsa
    out += ["", "namespace hsa {  // AMDDevice.create_queue's gart offsets (ops_amd.py:1109): hsa.amd_queue_t's"]
    out += [nvgen.const_line(f"amd_queue_t_{f}_offset", getattr(hsa.amd_queue_t, f).offset) for f in ("read_dispatch_id", "write_dispatch_id")]
    out += ["} // namespace hsa"]
    import hashlib
    fwrows = firmware_rows()
    out += ["", "namespace fw {  // AMFirmware's fetch_fw calls on this card, in order (TinyGPUFirmware.h finds and checks them)"]
    out += [f'constexpr nvfw::TGFirmware kFirmware[] = {{']
    out += [f'    {{"{arch}", "{name}", "{path}", "{name}", "{sha}", "{hashlib.md5(url.encode()).hexdigest()}"}},' for path, name, sha, url in fwrows]
    out += ["};", "} // namespace fw"]
    out += ["", "#pragma pack(push, 1)"]
    aliases = nvgen.aliases_of(am)
    for T in structs: out += [""] + nvgen.struct_lines(T, "am.py", aliases)
    out += ["", "#pragma pack(pop)", "", "} // namespace am", "} // namespace tinygpu_device", "", "#endif // LIBHMSBEAGLE_GPU_TINYGPUAMDBOOTTABLES_H"]
    (outdir / "TinyGPUAMDBootTables.h").write_text("\n".join(out) + "\n")
    print(f"TinyGPUAMDBootTables.h: {len(cov['used'])} registers, {len(consts)} constants, {len(structs)} structs")

if __name__ == "__main__":
    main()
