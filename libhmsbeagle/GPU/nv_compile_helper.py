#!/usr/bin/env python3
"""
nv_compile_helper.py — compile BEAGLE PTX to SASS for the hybrid NV backend.

Usage:
  python3 nv_compile_helper.py <ptx_file> <kernel_name> <handoff.json> <result.json>

Reads:
  ptx_file     — PTX source file to compile
  kernel_name  — name of the entry-point kernel in the PTX
  handoff.json — JSON produced by nv_init_helper.py (for arch/eop info)

Writes:
  result.json  — JSON with: code_b64, code_offset, code_size,
                 reg_count, shmem_size, cbuf0_param_offset,
                 cbuf0_prefix_b64, slm_size

The QMD is built entirely in C++ (GPUInterfaceTinyGPUHybrid.cpp) using the
metadata from result.json.  prog_addr is NOT embedded here because C++ decides
where in the code buffer the kernel lives.

Requires ptxas from the CUDA toolkit (any recent version, PATH or PTXAS env var).
"""

import sys, os, json, base64, struct, subprocess, tempfile, array, time


# ─────────────────────────────────────────────────────────────────────────────
# Persistent log — survives kernel panics (O_SYNC + fsync each write).
# Appended, not truncated: all concurrent/sequential compile calls share one file.
# ─────────────────────────────────────────────────────────────────────────────

class _TeeStream:
    def __init__(self, original, path):
        self._orig = original
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_SYNC, 0o644)
        self._log  = os.fdopen(fd, 'w', buffering=1)
    def write(self, s):
        self._orig.write(s); self._orig.flush()
        self._log.write(s);  self._log.flush()
        os.fsync(self._log.fileno())
    def flush(self):
        self._orig.flush(); self._log.flush()
    def fileno(self):
        return self._orig.fileno()


_LOG_PATH = os.path.expanduser("~/Library/Logs/nv_compile_helper.log")

# Locate tinygrad
_TINYGRAD_PATH = os.environ.get(
    "TINYGRAD_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "..", "..", "tinygrad"))
sys.path.insert(0, os.path.abspath(_TINYGRAD_PATH))

from tinygrad.runtime.support.elf import elf_loader, ElfSection


# ─────────────────────────────────────────────────────────────────────────────
# ptxas wrapper
# ─────────────────────────────────────────────────────────────────────────────

def _ptxas_path() -> str:
    if p := os.environ.get("PTXAS"):
        return p
    for candidate in ("/usr/local/cuda/bin/ptxas", "/usr/cuda/bin/ptxas",
                      "/opt/cuda/bin/ptxas"):
        if os.path.isfile(candidate):
            return candidate
    return "ptxas"   # hope it's on PATH


# ─────────────────────────────────────────────────────────────────────────────
# nvJitLink alternate compile path (opt-in, BEAGLE_NV_USE_NVJITLINK=1)
#
# tinygrad's own real-hardware compile path (PTXRenderer -> NVPTXCompiler,
# renderer/ptx.py + runtime/support/compiler_cuda.py) uses libnvJitLink, not
# the standalone ptxas binary this file has used throughout the "usb" branch
# investigation (see STATUS.md §33 onward). This is a minimal, opt-in
# alternate path -- calls a small C wrapper (nvjitlink_compile, built against
# libnvjitlink-dev-12-8, see STATUS.md for the build) that mirrors
# NVPTXCompiler.compile()'s exact API sequence: nvJitLinkCreate(["-arch=..."])
# -> nvJitLinkAddData(NVJITLINK_INPUT_PTX) -> nvJitLinkComplete ->
# nvJitLinkGetLinkedCubin. Default remains ptxas -- this must be explicitly
# enabled, never silently substituted.
# ─────────────────────────────────────────────────────────────────────────────

def _use_nvjitlink() -> bool:
    return os.environ.get("BEAGLE_NV_USE_NVJITLINK", "") not in ("", "0")


def _nvjitlink_compile_path() -> str:
    return os.environ.get("NVJITLINK_COMPILE",
                           os.path.expanduser("~/.local/bin/nvjitlink_compile"))


def _ptxas_split_kernels() -> set:
    # Opt-in per-kernel-compiler-selection (STATUS.md §52/TODO.md Phase 19).
    # Only meaningful alongside BEAGLE_NV_USE_NVJITLINK=1 (nvJitLink is the
    # default for everything else in that mode -- without it, everything is
    # already ptxas and there's nothing to override). Comma-separated kernel
    # names that should use ptxas instead of nvJitLink.
    raw = os.environ.get("BEAGLE_NV_PTXAS_KERNELS", "")
    return {s.strip() for s in raw.split(",") if s.strip()}


def compile_ptx(ptx_path: str, arch: str, kernel_name: str = "", force_tool: str = None,
                 extra_ptxas_args: list = None) -> bytes:
    # force_tool: "ptxas" or "nvjitlink", overrides the BEAGLE_NV_USE_NVJITLINK
    # env-var decision for this one call. Used by the per-kernel-compiler-
    # selection path (compile_ptx_split, below), which needs both compilers'
    # output from a single "_all" PTX module regardless of the env var.
    # extra_ptxas_args: additional flags inserted into the ptxas invocation
    # (e.g. ["-maxrregcount=20"]) -- ignored under nvJitLink. Optional,
    # defaults to None (today's exact behavior, unchanged) -- added for
    # TODO.md Phase 85's register-pressure bisection on the real
    # kernelMatrixMulADB, no other caller passes this.
    # Copy PTX to a Python-owned temp file before calling ptxas.
    # On macOS, ptxas (CUDA binary with DriverKit entitlements) cannot open
    # paths under /tmp (→ /private/tmp); Python's tempfile dir (/var/folders/…)
    # is accessible to all subprocesses we spawn.
    with open(ptx_path, "rb") as f:
        ptx_data = f.read()

    ptx_tmp = out = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".ptx", delete=False) as pf:
            pf.write(ptx_data)
            ptx_tmp = pf.name
        with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as cf:
            out = cf.name

        use_nvjitlink = (force_tool == "nvjitlink") if force_tool else _use_nvjitlink()
        if use_nvjitlink:
            tool, tool_args = "nvjitlink_compile", [_nvjitlink_compile_path(), ptx_tmp, out, arch]
        else:
            tool, tool_args = "ptxas", ([_ptxas_path(), f"--gpu-name={arch}", "-O3"]
                                         + (extra_ptxas_args or [])
                                         + ["--output-file", out, ptx_tmp])

        r = subprocess.run(tool_args, capture_output=True)

        if r.returncode != 0:
            if r.stdout:
                print(f"nv_compile_helper: {tool} stdout:\n{r.stdout.decode(errors='replace')}",
                      file=sys.stderr)
            if r.stderr:
                print(f"nv_compile_helper: {tool} stderr:\n{r.stderr.decode(errors='replace')}",
                      file=sys.stderr)
            log_dir = os.path.expanduser("~/Library/Logs")
            safe = kernel_name.replace("/", "_") or "unknown"
            saved = os.path.join(log_dir, f"failed_{safe}.ptx")
            try:
                with open(saved, "wb") as sf:
                    sf.write(ptx_data)
                print(f"nv_compile_helper: failing PTX saved to {saved}", file=sys.stderr)
            except Exception as copy_err:
                print(f"nv_compile_helper: could not save PTX: {copy_err}", file=sys.stderr)
            print(f"nv_compile_helper: {tool} exited {r.returncode} — exiting cleanly",
                  file=sys.stderr)
            sys.exit(1)

        with open(out, "rb") as f:
            return f.read()
    finally:
        for p in (ptx_tmp, out):
            if p and os.path.exists(p):
                os.unlink(p)


# ─────────────────────────────────────────────────────────────────────────────
# ELF metadata extraction  (tinygrad elf_loader convention)
# ─────────────────────────────────────────────────────────────────────────────

EIATTR_PARAM_CBANK    = 0x0a
EIATTR_MIN_STACK_SIZE = 0x12
EIATTR_REGCOUNT       = 0x2f   # CUDA <= 11 / older PTX
EIATTR_REGCOUNT_V2    = 0x37   # Blackwell / newer ptxas (sm_100+)


def _parse_nv_info(content: bytes):
    """Yield (typ, param_id, data_bytes) triples from an .nv.info* ELF section."""
    off = 0
    while off < len(content):
        if off + 4 > len(content):
            break
        typ, param, sz = struct.unpack_from("BBH", content, off)
        data = content[off + 4 : off + 4 + sz] if typ == 0x4 else b""
        yield typ, param, data, sz
        off += (sz if typ == 0x4 else 0) + 4


def _arch_from_handoff(handoff: dict) -> str:
    """ptxas --gpu-name for this run.

    Prefers "gpu_arch" -- computed by nv_init_helper.py from the real
    NV2080_CTRL_GR_INFO_INDEX_SM_VERSION hardware register (see
    nv_init_helper.py's _sass_version_and_arch(), STATUS.md §62/TODO.md
    Phase 27) and passed through the handoff C++'s SetDevice() writes.
    Falls back to the old chip-name-prefix guess only if that field is
    missing (e.g. a handoff file from before this fix, or a manually
    constructed one for static testing) -- that guess is still correct for
    Ampere/Ada, but was wrong for Blackwell (mapped every "GB2"-prefixed
    chip, including consumer RTX dies like this one's, to compute
    capability 10.0, the *datacenter* Blackwell value; real consumer/
    professional RTX Blackwell is compute capability 12.0/sm_120).
    """
    if gpu_arch := handoff.get("gpu_arch"):
        return gpu_arch
    chip = handoff.get("chip_name", "GA102")
    table = {
        "GA1": "sm_86",  # Ampere (GA102 = SM 8.6)
        "AD1": "sm_89",  # Ada Lovelace (AD102 = SM 8.9)
        "GB2": "sm_100", # Blackwell fallback guess -- see docstring above
    }
    return table.get(chip[:3], "sm_86")


def _parse_elf_kernels(elf_bytes: bytes, is_blackwell: bool = False) -> tuple:
    """
    Core ELF-parsing logic shared by extract_all_metadata (single-compiler,
    default path) and compile_ptx_split (per-kernel-compiler-selection path,
    see STATUS.md §52/TODO.md Phase 19 -- the latter needs `sections`/`relocs`
    too, to slice out one kernel's raw code bytes and to safety-check it has
    no relocations before splicing it into a different image).

    Returns: (image, sections, relocs, kernels) -- image/sections/relocs are
    elf_loader's raw return values (image is a memoryview); kernels is the
    same dict extract_all_metadata has always returned.
    """
    image, sections, relocs = elf_loader(elf_bytes, force_section_align=128)

    # First pass: find all kernel text sections.
    kernels: dict = {}
    for sh in sections:
        if sh.name.startswith(".text.") and not sh.name.startswith(".text.."):
            kname = sh.name[6:]
            kernels[kname] = {
                "code_offset":    sh.header.sh_addr,
                "code_size":      sh.header.sh_size,
                "reg_count":      0,
                "shmem_size":     0x400,
                "cbuf0_param_off": 48,   # default: 12 uint32s × 4
                "cbuf0_num_u32s": 12,
                "slm_size":       0,
            }

    # Second pass: fill per-kernel metadata from .nv.info.* and .nv.shared.* sections.
    for sh in sections:
        if sh.name.startswith(".nv.info."):
            kname = sh.name[9:]
            if kname not in kernels:
                continue
            k = kernels[kname]
            cbuf0_size = 0
            for typ, param, data, sz in _parse_nv_info(bytes(sh.content)):
                if param == EIATTR_PARAM_CBANK and typ == 0x4 and len(data) >= 6:
                    cbuf0_base_raw = struct.unpack_from("I", data)[0]
                    cbuf0_size = struct.unpack_from("H", data, 4)[0]
                    if kname in ("kernelMatrixMulADB", "kernelPartialsPartialsNoScale", "kernelSumSites1"):
                        print(f"  DBG PARAM_CBANK {kname}: base=0x{cbuf0_base_raw:x} size=0x{cbuf0_size:x}", file=sys.stderr, flush=True)
                elif param in (EIATTR_REGCOUNT, EIATTR_REGCOUNT_V2):
                    if typ == 0x4 and len(data) >= 4:
                        k["reg_count"] = struct.unpack_from("I", data)[0]
                    else:
                        k["reg_count"] = sz
            if cbuf0_size:
                min_entries = 224 if is_blackwell else 12
                n = max(cbuf0_size // 4, min_entries)
                k["cbuf0_num_u32s"] = n
                k["cbuf0_param_off"] = n * 4

        elif sh.name.startswith(".nv.shared."):
            kname = sh.name[11:]
            if kname in kernels:
                kernels[kname]["shmem_size"] = 0x400 + sh.header.sh_size

    # Finalize: round shmem, build cbuf0 prefix.
    for k in kernels.values():
        k["shmem_size"] = (max(k["shmem_size"], 0x400) + 127) & ~127
        prefix = build_cbuf0_prefix(k["cbuf0_num_u32s"], is_blackwell=is_blackwell)
        k["cbuf0_prefix_b64"] = base64.b64encode(
            array.array('I', prefix).tobytes()).decode()

    return image, sections, relocs, kernels


def extract_all_metadata(elf_bytes: bytes, is_blackwell: bool = False) -> tuple:
    """
    Compile-once variant: parse every kernel found in the ELF and return metadata
    for all of them.

    Returns:
        (image_bytes, dict of kernel_name → metadata_dict)
    where metadata_dict has the same keys as extract_metadata's return tuple.
    """
    image, _sections, _relocs, kernels = _parse_elf_kernels(elf_bytes, is_blackwell=is_blackwell)
    return bytes(image), kernels


def compile_ptx_split(ptx_path: str, arch: str, is_blackwell: bool, ptxas_kernels: set) -> tuple:
    """
    Per-kernel-compiler-selection path (STATUS.md §52/TODO.md Phase 19):
    compiles the same "_all" PTX module through BOTH ptxas and nvJitLink,
    then produces a merged image where each kernel in `ptxas_kernels` (by
    name) uses its ptxas-compiled code and every other kernel uses its
    nvJitLink-compiled code, unchanged. Motivation: nvJitLink measurably
    helps kernelPartialsPartialsNoScale but produces a confirmed runtime
    defect in kernelMatrixMulADB (STATUS.md §41-51) -- this lets each kernel
    use whichever compiler is actually correct for it, without needing to
    fully root-cause nvJitLink's defect first.

    Design (deliberately conservative, chosen after a first version was
    caught by its own safety check -- see STATUS.md §52): the merged image
    is the FULL, UNMODIFIED nvJitLink image, verbatim -- every nvJitLink
    kernel *and* every internal helper routine it emits (nvJitLink adds
    CUDA math-library slow-path helpers, e.g. `__cuda_sm3x_div_rn_noftz_f32_
    slowpath`, that ptxas doesn't produce as separate symbols) keeps its
    exact original offset. This backend applies no ELF relocations at all
    (confirmed: the existing single-compiler path has always ignored
    `elf_loader`'s `relocs` return value and works correctly on hardware),
    meaning intra-module calls between nvJitLink-compiled units -- e.g. a
    kernel calling one of those slow-path helpers -- are resolved as
    statically-baked offsets at compile time, not via relocations. A first
    version of this function rebuilt the image from scratch in sorted
    kernel-name order, which would have silently moved those helpers
    relative to their callers and corrupted any such call; repacking is
    now avoided entirely. Only the override kernels' ptxas bytes are
    *appended* at the end (128-byte aligned) with a freshly assigned
    offset -- safe because nothing calls *into* them (they're
    `KW_GLOBAL_KERNEL` entry points, launched directly by the C++ host,
    never invoked as a subroutine from another kernel).

    Safety: before appending any override kernel's ptxas bytes, verifies
    (not assumes) that ptxas's ELF has zero relocation entries targeting
    that kernel's own .text section -- confirmed necessary and sufficient
    for a self-contained kernel via a dedicated Explore pass over
    tinygrad's elf_loader/ops_nv.py (STATUS.md §52). Aborts loudly (raises)
    rather than silently risk a hang on real hardware if that check fails.

    Returns: (image_bytes, dict of kernel_name → metadata_dict), same shape
    as extract_all_metadata's return value -- callers (_main_impl) don't
    need to know a split compile happened.
    """
    elf_ptxas     = compile_ptx(ptx_path, arch, "_all", force_tool="ptxas")
    elf_nvjitlink = compile_ptx(ptx_path, arch, "_all", force_tool="nvjitlink")

    img_a, sections_a, relocs_a, kernels_a = _parse_elf_kernels(elf_ptxas,     is_blackwell=is_blackwell)
    img_b, _sections_b, _relocs_b, kernels_b = _parse_elf_kernels(elf_nvjitlink, is_blackwell=is_blackwell)

    unknown = ptxas_kernels - kernels_a.keys()
    if unknown:
        raise RuntimeError(f"compile_ptx_split: requested ptxas override for unknown kernel(s) "
                            f"(not found in the ptxas compile): {sorted(unknown)}")

    # Safety check: for each requested override kernel, confirm ptxas's ELF
    # has no relocation entries targeting that kernel's own .text section.
    for kname in sorted(ptxas_kernels):
        sh = next(s for s in sections_a if s.name == f".text.{kname}")
        lo, hi = sh.header.sh_addr, sh.header.sh_addr + sh.header.sh_size
        bad = [r for r in relocs_a if lo <= r[0] < hi]
        if bad:
            raise RuntimeError(f"compile_ptx_split: {kname}'s ptxas .text section has "
                                f"{len(bad)} relocation(s) targeting it -- appending its raw "
                                f"bytes at a new image offset would be unsafe (the relocation(s) "
                                f"embed a position-dependent address). Refusing to proceed; this "
                                f"kernel cannot currently use the split-compile path.")
        print(f"  DBG split-compile: {kname} verified relocation-free, safe to append "
              f"({sh.header.sh_size} bytes from ptxas)", file=sys.stderr, flush=True)

    # Base = the full nvJitLink image, byte-for-byte unchanged. Every
    # kernel/helper not in ptxas_kernels keeps its original nvJitLink
    # metadata (code_offset included) exactly as-is.
    merged = bytearray(img_b)
    kernels_out = dict(kernels_b)

    for kname in sorted(ptxas_kernels):
        src_k = kernels_a[kname]
        off, sz = src_k["code_offset"], src_k["code_size"]
        code = bytes(img_a[off:off + sz])

        pad = (-len(merged)) % 128
        merged.extend(b"\0" * pad)
        new_off = len(merged)
        merged.extend(code)

        k = dict(src_k)  # reg_count/shmem/cbuf0_param_off/etc. all from ptxas too
        k["code_offset"] = new_off
        kernels_out[kname] = k  # overrides (or adds, if absent from nvJitLink) this kernel only

    return bytes(merged), kernels_out


def extract_metadata(elf_bytes: bytes, kernel_name: str):
    """
    Parse a ptxas-compiled SASS ELF and return:
      code_bytes      — flat image bytes
      code_offset     — byte offset of .text.{kernel_name} in image
      code_size       — byte size of .text.{kernel_name}
      reg_count       — register count (from EIATTR_REGCOUNT)
      shmem_size      — shared memory bytes (from .nv.shared.{kernel_name})
      cbuf0_param_off — byte offset in cbuf0 where kernel params start
      cbuf0_num_u32s  — number of uint32 slots in cbuf0 prefix (driver params)
      slm_size        — stack/local memory per thread
    """
    # elf_loader returns (flat_image, sections, relocs)
    image, sections, _relocs = elf_loader(elf_bytes, force_section_align=128)

    code_offset = 0
    code_size   = 0
    reg_count   = 0
    shmem_size  = 0
    slm_size    = 0
    cbuf0_size  = 0  # from EIATTR_PARAM_CBANK[1] (tinygrad convention)

    # Debug: dump all section names so we can diagnose missing sections.
    print(f"nv_compile_helper: ELF sections for {kernel_name}:", file=sys.stderr)
    for sh in sections:
        print(f"  {sh.name!r}  sh_addr={sh.header.sh_addr:#x}  sh_size={sh.header.sh_size:#x}", file=sys.stderr)

    for sh in sections:
        if sh.name == f".text.{kernel_name}":
            code_offset = sh.header.sh_addr
            code_size   = sh.header.sh_size
        elif sh.name == f".nv.shared.{kernel_name}":
            shmem_size  = max(shmem_size, 0x400 + sh.header.sh_size)

        elif sh.name == f".nv.info.{kernel_name}":
            for typ, param, data, sz in _parse_nv_info(bytes(sh.content)):
                print(f"  .nv.info.{kernel_name}: typ={typ:#x} param={param:#x} sz={sz} data={data.hex()}", file=sys.stderr)
                if param == EIATTR_PARAM_CBANK and typ == 0x4 and len(data) >= 6:
                    cbuf0_size = struct.unpack_from("H", data, 4)[0]
                elif param in (EIATTR_REGCOUNT, EIATTR_REGCOUNT_V2):
                    if typ == 0x4 and len(data) >= 4:
                        # EIFMT_SVAL: register count is the 4-byte data payload
                        reg_count = struct.unpack_from("I", data)[0]
                    else:
                        # EIFMT_NVAL: value is embedded in the sz field
                        reg_count = sz

        elif sh.name == ".nv.info":
            for typ, param, data, sz in _parse_nv_info(bytes(sh.content)):
                if param == EIATTR_MIN_STACK_SIZE and typ == 0x4 and len(data) >= 8:
                    slm_size = struct.unpack_from("II", data)[1] + 0x240

    # tinygrad's cbuf0 param offset = max(cbuf0_size, 48) rounded to 4-byte units
    # (matches min_cbuf0_entries=12 uint32s = 48 bytes for v3)
    cbuf0_num_u32s  = max(cbuf0_size // 4, 12)
    cbuf0_param_off = cbuf0_num_u32s * 4  # byte offset where kernel args go

    return bytes(image), code_offset, code_size, reg_count, shmem_size, cbuf0_param_off, cbuf0_num_u32s, slm_size


# ─────────────────────────────────────────────────────────────────────────────
# cbuf0 prefix (driver params)
# ─────────────────────────────────────────────────────────────────────────────

SHARED_MEM_WIN = 0x729400000000
LOCAL_MEM_WIN  = 0x729300000000


def build_cbuf0_prefix(n_u32s: int, is_blackwell: bool = False) -> list:
    """Build the cbuf0 driver-params prefix (n_u32s uint32 elements)."""
    min_entries = 224 if is_blackwell else 12
    buf = [0] * max(n_u32s, min_entries)

    def put64le(idx: int, val: int):
        buf[idx]     = val & 0xFFFFFFFF
        buf[idx + 1] = (val >> 32) & 0xFFFFFFFF

    if is_blackwell:
        # Blackwell: driver params at indices 188-191 (windows) and 223 (0xfffdc0)
        put64le(188, SHARED_MEM_WIN)
        put64le(190, LOCAL_MEM_WIN)
        buf[223] = 0xfffdc0
    else:
        # Pre-Blackwell: indices 6-11
        put64le(6,  SHARED_MEM_WIN)
        put64le(8,  LOCAL_MEM_WIN)
        put64le(10, 0xfffdc0)
    return buf


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 5:
        print(f"Usage: {sys.argv[0]} <ptx_file> <kernel_name> <handoff.json> <result.json>",
              file=sys.stderr)
        sys.exit(1)

    ptx_file    = sys.argv[1]
    kernel_name = sys.argv[2]
    handoff_f   = sys.argv[3]
    result_f    = sys.argv[4]

    os.makedirs(os.path.expanduser("~/Library/Logs"), exist_ok=True)
    sys.stderr = _TeeStream(sys.stderr, _LOG_PATH)
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    print(f"\n=== nv_compile_helper {ts} kernel={kernel_name} ===", file=sys.stderr)

    import traceback as _tb
    try:
        _main_impl(ptx_file, kernel_name, handoff_f, result_f)
    except Exception:
        _tb.print_exc(file=sys.stderr)
        sys.exit(1)


def _main_impl(ptx_file, kernel_name, handoff_f, result_f):
    with open(handoff_f) as f:
        handoff = json.load(f)

    arch = _arch_from_handoff(handoff)
    compute_class = handoff.get("compute_class", 0)
    is_blackwell = compute_class >= 0xcec0
    print(f"nv_compile_helper: compiling {kernel_name} for {arch} (blackwell={is_blackwell}) …", file=sys.stderr)

    ptxas_split_kernels = _ptxas_split_kernels() if kernel_name == "_all" else set()

    if kernel_name == "_all" and ptxas_split_kernels:
        print(f"nv_compile_helper: split-compile — {sorted(ptxas_split_kernels)} via ptxas, "
              f"everything else via nvJitLink", file=sys.stderr, flush=True)
        code_bytes, kernels = compile_ptx_split(ptx_file, arch, is_blackwell, ptxas_split_kernels)
    elif kernel_name == "_all":
        elf_bytes = compile_ptx(ptx_file, arch, kernel_name)
        code_bytes, kernels = extract_all_metadata(elf_bytes, is_blackwell=is_blackwell)
    else:
        elf_bytes = compile_ptx(ptx_file, arch, kernel_name)

    if kernel_name == "_all":
        # Compile-once path: extract every kernel from the single ELF, write JSONL.
        with open(result_f, "w") as f:
            # Line 0: shared image
            json.dump({"arch": arch, "code_b64": base64.b64encode(code_bytes).decode()}, f)
            f.write("\n")
            # One line per kernel, sorted for determinism
            for kname in sorted(kernels):
                k = kernels[kname]
                json.dump({"name": kname, "code_offset": k["code_offset"],
                           "code_size": k["code_size"], "reg_count": k["reg_count"],
                           "shmem_size": k["shmem_size"],
                           "cbuf0_param_off": k["cbuf0_param_off"],
                           "cbuf0_prefix_b64": k["cbuf0_prefix_b64"],
                           "slm_size": k["slm_size"]}, f)
                f.write("\n")
        print(f"nv_compile_helper: _all done — {len(kernels)} kernels, "
              f"image={len(code_bytes):#x} bytes", file=sys.stderr)
        return

    # Single-kernel path (original behaviour).
    code_bytes, code_off, code_sz, reg_count, shmem, cbuf0_param_off, cbuf0_n_u32, slm = \
        extract_metadata(elf_bytes, kernel_name)

    # Round up shmem to match tinygrad's formula
    shmem = max(shmem, 0x400)
    shmem = (shmem + 127) & ~127  # align to 128 bytes

    cbuf0_prefix = build_cbuf0_prefix(cbuf0_n_u32)

    result = {
        "kernel_name":     kernel_name,
        "arch":            arch,
        "code_b64":        base64.b64encode(code_bytes).decode(),
        "code_offset":     code_off,
        "code_size":       code_sz,
        "reg_count":       reg_count,
        "shmem_size":      shmem,
        "cbuf0_param_off": cbuf0_param_off,
        "cbuf0_prefix_b64": base64.b64encode(
            array.array('I', cbuf0_prefix).tobytes()).decode(),
        "slm_size":        slm,
    }

    with open(result_f, "w") as f:
        json.dump(result, f, indent=2)

    print(f"nv_compile_helper: done — code_size={code_sz:#x} "
          f"reg_count={reg_count} shmem={shmem} cbuf0_param_off={cbuf0_param_off}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
