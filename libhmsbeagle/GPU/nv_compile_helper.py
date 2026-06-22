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

import sys, os, json, base64, struct, subprocess, tempfile, array

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


def compile_ptx(ptx_path: str, arch: str) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as tf:
        out = tf.name
    try:
        subprocess.run(
            [_ptxas_path(), f"--gpu-name={arch}", "-O3",
             "--output-file", out, ptx_path],
            check=True, capture_output=True)
        with open(out, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(out):
            os.unlink(out)


# ─────────────────────────────────────────────────────────────────────────────
# ELF metadata extraction  (tinygrad elf_loader convention)
# ─────────────────────────────────────────────────────────────────────────────

EIATTR_PARAM_CBANK  = 0x0a
EIATTR_MIN_STACK_SIZE = 0x12
EIATTR_REGCOUNT     = 0x2f


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
    """Derive ptxas --gpu-name from chip_name in handoff."""
    chip = handoff.get("chip_name", "GA102")
    table = {
        "GA1": "sm_86",  # Ampere
        "AD1": "sm_89",  # Ada Lovelace
        "GB2": "sm_120", # Blackwell (approximate)
    }
    return table.get(chip[:3], "sm_86")


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

    for sh in sections:
        if sh.name == f".text.{kernel_name}":
            code_offset = sh.header.sh_addr
            code_size   = sh.header.sh_size
        elif sh.name == f".nv.shared.{kernel_name}":
            shmem_size  = max(shmem_size, 0x400 + sh.header.sh_size)

        elif sh.name == f".nv.info.{kernel_name}":
            for typ, param, data, sz in _parse_nv_info(bytes(sh.content)):
                if param == EIATTR_PARAM_CBANK and typ == 0x4 and len(data) >= 6:
                    cbuf0_size = struct.unpack_from("H", data, 4)[0]
                elif param == EIATTR_REGCOUNT:
                    reg_count  = sz  # for typ != 0x4, sz IS the value

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


def build_cbuf0_prefix(n_u32s: int) -> list:
    """Build the cbuf0 driver-params prefix (n_u32s uint32 elements)."""
    buf = [0] * max(n_u32s, 12)

    def put64le(idx: int, val: int):
        buf[idx]     = val & 0xFFFFFFFF
        buf[idx + 1] = (val >> 32) & 0xFFFFFFFF

    # indices 6-11 = shared_mem_window, local_mem_window, 0xfffdc0
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

    with open(handoff_f) as f:
        handoff = json.load(f)

    arch = _arch_from_handoff(handoff)
    print(f"nv_compile_helper: compiling {kernel_name} for {arch} …", file=sys.stderr)

    elf_bytes = compile_ptx(ptx_file, arch)

    code_bytes, code_off, code_sz, reg_count, shmem, cbuf0_param_off, cbuf0_n_u32, slm = \
        extract_metadata(elf_bytes, kernel_name)

    # Round up shmem to match tinygrad's formula
    shmem_configs = [32 << 10, 64 << 10, 100 << 10]
    shmem = max(shmem, 0x400)
    shmem = round(shmem + 127) & ~127  # align to 128 bytes

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
