#!/usr/bin/env python3
"""
amd_compile_helper.py — BEAGLE hybrid AMD backend: compile all kernels once.

    python3 amd_compile_helper.py <cl_file> <handoff.json> <result.jsonl>

Mirrors nv_compile_helper.py's "_all" precompile-once path exactly (see that
file's main()/_main_impl `kernel_name == "_all"` branch and
GPUInterfaceTinyGPUHybrid.cpp's precompile_all_kernels()): one call compiles
every kernel in kernelResource->kernelCode (BEAGLE's existing FW_OPENCL
source — reused unmodified, see the plan's compiler-backend decision) into
one shared code image, and result.jsonl's line 0 is that image (code_b64)
followed by one line per kernel with the fields C++ needs to dispatch it.

Compile backend: libamd_comgr.dylib directly via ctypes (AMD's ROCm Code
Object Manager — already present on this machine, bundles its own
device-libs; see amd_compile_test.py from this session's Phase 0, which this
is adapted from), with AMD_COMGR_LANGUAGE_OPENCL_2_0 rather than the HIP
language comgr is more commonly used for.

Per-kernel metadata comes from parsing the compiled HSACO's ELF: each kernel
gets a `<name>.kd` symbol (STT_OBJECT) whose value points at its 64-byte
kernel_descriptor_t (LLVM AMDHSA ABI) -- confirmed by compiling a real
two-kernel OpenCL source and walking .symtab/.strtab this session
(amd_multi_kernel_test.py), not assumed from documentation alone. This is
different from tinygrad's own AMDProgram.__init__ (ops_amd.py), which reads
.rodata's own section address directly -- that only works for tinygrad's
one-kernel-per-compile model; BEAGLE compiles many kernels per call, so the
per-symbol lookup below is required.
"""

import sys, os, json, base64, ctypes, struct, time

_TINYGRAD_PATH = os.environ.get(
    "TINYGRAD_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "..", "..", "tinygrad"))
if not os.path.isdir(_TINYGRAD_PATH):
    print(f"amd_compile_helper: cannot find tinygrad at {_TINYGRAD_PATH}\n"
          "Set TINYGRAD_PATH to the tinygrad checkout root.", file=sys.stderr)
    sys.exit(1)
sys.path.insert(0, os.path.abspath(_TINYGRAD_PATH))

from tinygrad.runtime.support.elf import elf_loader
from tinygrad.runtime.autogen import amdgpu_kd, hsa

_COMGR_PATH = os.environ.get("BEAGLE_AMD_COMGR", "/opt/homebrew/lib/libamd_comgr.dylib")
_LOG_PATH = os.path.expanduser("~/Library/Logs/amd_compile_helper.log")


class _TeeStream:
    def __init__(self, original, path):
        self._orig = original
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_SYNC, 0o644)
        self._log = os.fdopen(fd, 'w', buffering=1)
    def write(self, s):
        self._orig.write(s); self._orig.flush()
        self._log.write(s);  self._log.flush()
    def flush(self):
        self._orig.flush(); self._log.flush()
    def fileno(self):
        return self._orig.fileno()


def _load_comgr():
    comgr = ctypes.CDLL(_COMGR_PATH)
    major, minor = ctypes.c_uint64(), ctypes.c_uint64()
    comgr.amd_comgr_get_version(ctypes.byref(major), ctypes.byref(minor))
    if major.value >= 3:
        import tinygrad.runtime.autogen.comgr_3 as C
    else:
        import tinygrad.runtime.autogen.comgr as C
    return comgr, C


def _check(comgr, C, status, ctx=""):
    if status != 0:
        s = ctypes.POINTER(ctypes.c_char)()
        comgr.amd_comgr_status_string(status, ctypes.byref(s))
        raise RuntimeError(f"comgr fail ({ctx}): {status}, {ctypes.string_at(s).decode()}")


def _get_data(comgr, C, data_set, data_type):
    d = C.amd_comgr_data_t()
    _check(comgr, C, comgr.amd_comgr_action_data_get_data(data_set, data_type, 0, ctypes.byref(d)))
    sz = ctypes.c_uint64()
    _check(comgr, C, comgr.amd_comgr_get_data(d, ctypes.byref(sz), None))
    buf = ctypes.create_string_buffer(sz.value)
    _check(comgr, C, comgr.amd_comgr_get_data(d, ctypes.byref(sz), buf))
    _check(comgr, C, comgr.amd_comgr_release_data(d))
    return buf.raw[:sz.value]


def compile_opencl(comgr, C, src: str, arch: str) -> bytes:
    ai = C.amd_comgr_action_info_t()
    _check(comgr, C, comgr.amd_comgr_create_action_info(ctypes.byref(ai)), "create_action_info")
    _check(comgr, C, comgr.amd_comgr_action_info_set_language(ai, C.AMD_COMGR_LANGUAGE_OPENCL_2_0), "set_language")
    _check(comgr, C, comgr.amd_comgr_action_info_set_isa_name(ai, f"amdgcn-amd-amdhsa--{arch}".encode()), "set_isa_name")
    _check(comgr, C, comgr.amd_comgr_action_info_set_logging(ai, True), "set_logging")
    # Same build defines the real OpenCL backend passes to clBuildProgram
    # (GPUInterfaceOpenCL.cpp: "-w -D FW_OPENCL -D OPENCL_KERNEL_BUILD") --
    # OPENCL_KERNEL_BUILD in particular is what makes the kernel source skip
    # its `#include "libhmsbeagle/platform.h"` (only needed for the host-side
    # build, guarded out entirely when compiling as a GPU kernel string;
    # confirmed by reading the actual generated BeagleOpenCL_kernels.h).
    opts = [b"-DFW_OPENCL", b"-DOPENCL_KERNEL_BUILD"]
    opts_arr = (ctypes.c_char_p * len(opts))(*opts)
    _check(comgr, C, comgr.amd_comgr_action_info_set_option_list(ai, opts_arr, len(opts)), "set_option_list")

    ds_src, ds_bc, ds_reloc, ds_exec = (C.amd_comgr_data_set_t() for _ in range(4))
    for ds in (ds_src, ds_bc, ds_reloc, ds_exec):
        _check(comgr, C, comgr.amd_comgr_create_data_set(ctypes.byref(ds)))

    data_src = C.amd_comgr_data_t()
    _check(comgr, C, comgr.amd_comgr_create_data(C.AMD_COMGR_DATA_KIND_SOURCE, ctypes.byref(data_src)))
    rprg = src.encode()
    _check(comgr, C, comgr.amd_comgr_set_data(data_src, len(rprg), rprg))
    _check(comgr, C, comgr.amd_comgr_set_data_name(data_src, b"beagle_kernels.cl"))
    _check(comgr, C, comgr.amd_comgr_data_set_add(ds_src, data_src))

    status = comgr.amd_comgr_do_action(C.AMD_COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC, ai, ds_src, ds_bc)
    if status != 0:
        try:
            print(_get_data(comgr, C, ds_bc, C.AMD_COMGR_DATA_KIND_LOG).decode(errors="replace"), file=sys.stderr)
        except Exception:
            pass
        _check(comgr, C, status, "COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC")

    _check(comgr, C, comgr.amd_comgr_do_action(C.AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE, ai, ds_bc, ds_reloc), "codegen_bc_to_reloc")
    _check(comgr, C, comgr.amd_comgr_do_action(C.AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE, ai, ds_reloc, ds_exec), "link_to_exec")
    ret = _get_data(comgr, C, ds_exec, C.AMD_COMGR_DATA_KIND_EXECUTABLE)

    _check(comgr, C, comgr.amd_comgr_release_data(data_src))
    for ds in (ds_src, ds_bc, ds_reloc, ds_exec):
        _check(comgr, C, comgr.amd_comgr_destroy_data_set(ds))
    _check(comgr, C, comgr.amd_comgr_destroy_action_info(ai))
    return ret


def compile_hip(src: str, arch: str) -> bytes:
    """
    Compile BEAGLE's kernel source via HIP language instead of OpenCL --
    see GPUImplDefs.h's FW_TINYGPU_HYBRID_AMD branch and STATUS.md AMD §18
    for why: OpenCL's get_global_id() pulls in dispatch_ptr/queue_ptr/
    dispatch_id sgprs and heavy scratch usage this remote transport can't
    drive correctly, while HIP's group/local-id builtins read hardware-
    architected sgprs directly and need none of that (confirmed empirically
    this session, both directions).

    Reuses tinygrad's own compile_hip() (tinygrad.runtime.support.
    compiler_amd) completely unmodified -- the exact function/code path
    already proven twice on this hardware this session -- rather than
    reimplementing comgr's staged HIP compile calls ourselves. The only
    BEAGLE-specific step is selecting the right kernel source variant via
    plain #define lines prepended to the source text (equivalent to -D
    flags, but tinygrad's compile_hip() doesn't take extra compiler options
    as a parameter).

    Also defines FW_OPENCL alongside FW_TINYGPU_HYBRID_AMD, mirroring how
    make_tinygpu_kernels.sh compiles the NV TinyGPU path with -DCUDA
    -DFW_TINYGPU together: GPUImplDefs.h's KW_* selection checks
    FW_TINYGPU_HYBRID_AMD first so it still wins there, but several
    interface-level choices live outside that chain, gated directly on
    #ifdef CUDA / #elif defined(FW_OPENCL) in kernelsAll.cu/kernels4.cu
    (e.g. kernelMatrixMulADBFirstDeriv's distanceRate/distanceLength locals
    vs. CUDA's KW_LOCAL_MEM form) -- those need FW_OPENCL defined too, since
    the batched-queue interface they select is the one this backend's C++
    side is built around. Confirmed necessary: compiling without it fails
    with "undeclared identifier 'distanceRate'" etc.
    """
    from tinygrad.runtime.support.compiler_amd import compile_hip as _tinygrad_compile_hip
    prefixed = "#define FW_TINYGPU_HYBRID_AMD 1\n#define FW_OPENCL 1\n#define OPENCL_KERNEL_BUILD 1\n" + src
    return _tinygrad_compile_hip(prefixed, arch)


def parse_kernels(hsaco: bytes):
    """Returns (image, {kernel_name: kernel_descriptor_t})."""
    image, sections, _relocs = elf_loader(hsaco)
    symtab_sec = next((s for s in sections if s.name == ".symtab"), None)
    strtab_sec = next((s for s in sections if s.name == ".strtab"), None)
    if symtab_sec is None or strtab_sec is None:
        raise RuntimeError("compiled HSACO has no .symtab/.strtab — cannot locate kernels")

    n_syms = len(symtab_sec.content) // 24  # ELF64 Sym entry size
    kd_addrs = {}
    for i in range(n_syms):
        st_name, _info, _other, _shndx, st_value, _size = struct.unpack_from("<IBBHQQ", symtab_sec.content, i * 24)
        end = strtab_sec.content.find(b"\0", st_name)
        name = strtab_sec.content[st_name:end].decode()
        if name.endswith(".kd"):
            kd_addrs[name[:-3]] = st_value

    desc_sz = ctypes.sizeof(amdgpu_kd.llvm_amdhsa_kernel_descriptor_t)
    kernels = {}
    for kname, kd_addr in kd_addrs.items():
        desc = amdgpu_kd.llvm_amdhsa_kernel_descriptor_t.from_buffer_copy(bytes(image[kd_addr:kd_addr + desc_sz]))
        kernels[kname] = (kd_addr, desc)
    return image, kernels


def _tmpring_size(waves: int, wavesize: int) -> int:
    t = hsa.union_COMPUTE_TMPRING_SIZE_GFX11_bitfields(WAVES=waves, WAVESIZE=wavesize)
    return int.from_bytes(bytes(t), "little")


def main() -> None:
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <cl_file> <handoff.json> <result.jsonl>", file=sys.stderr)
        sys.exit(1)

    cl_file, handoff_f, result_f = sys.argv[1], sys.argv[2], sys.argv[3]

    os.makedirs(os.path.expanduser("~/Library/Logs"), exist_ok=True)
    sys.stderr = _TeeStream(sys.stderr, _LOG_PATH)
    print(f"\n=== amd_compile_helper {time.strftime('%Y-%m-%dT%H:%M:%S')} ===", file=sys.stderr)

    import traceback as _tb
    try:
        _main_impl(cl_file, handoff_f, result_f)
    except Exception:
        _tb.print_exc(file=sys.stderr)
        sys.exit(1)


def _main_impl(cl_file, handoff_f, result_f):
    with open(handoff_f) as f:
        handoff = json.load(f)

    arch = handoff["gpu_arch"]
    cu_cnt = handoff["cu_cnt"]
    se_cnt = handoff["se_cnt"]
    max_slots_scratch_cu = handoff["max_slots_scratch_cu"]
    scratch_cap = handoff["scratch_cap_bytes_per_thread"]
    gfx_major = int(arch[3:5])  # "gfx1100" -> 11

    with open(cl_file) as f:
        src = f.read()

    print(f"amd_compile_helper: compiling for {arch} ({len(src)} bytes source) …", file=sys.stderr, flush=True)
    comgr, C = _load_comgr()
    hsaco = compile_opencl(comgr, C, src, arch)
    print(f"amd_compile_helper: compiled — {len(hsaco)} byte HSACO", file=sys.stderr, flush=True)

    image, kernels = parse_kernels(hsaco)
    if not kernels:
        raise RuntimeError("no kernels found in compiled HSACO (.kd symbols missing)")

    max_private = 0
    entries = {}
    for kname, (kd_addr, desc) in kernels.items():
        group_segment_size = desc.group_segment_fixed_size
        private_segment_size = desc.private_segment_fixed_size
        kernarg_size = desc.kernarg_size
        wave32 = bool(desc.kernel_code_properties & 0x400)
        lds_size = ((group_segment_size + 511) // 512) & 0x1FF

        rsrc1 = desc.compute_pgm_rsrc1 | ((1 << 20) if gfx_major == 11 else 0)  # gfx11 cwsr workaround, matches AMDProgram.__init__
        rsrc2 = desc.compute_pgm_rsrc2 | (lds_size << 15)
        rsrc3 = desc.compute_pgm_rsrc3

        # enable_dispatch_ptr is set whenever the kernel uses any OpenCL
        # work-item builtin (get_global_id/get_local_id/...), which is
        # essentially always for this kind of kernel (confirmed empirically
        # this session -- even a one-line get_global_id(0) kernel sets it) --
        # GPUInterfaceTinyGPUHybridAMD.cpp implements this: an extra 64-byte
        # hsa_kernel_dispatch_packet_t is appended after the kernel's own
        # kernarg_size and its VA is passed via user_data. This backend does
        # NOT implement enable_private_segment_sgpr (an older/legacy
        # mechanism gfx10+ doesn't need -- "architected flat scratch",
        # written unconditionally, replaces it); fail loudly rather than
        # silently mis-dispatch if a kernel ever sets it.
        enable_private_segment_sgpr = bool(desc.kernel_code_properties & hsa.AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER)
        enable_dispatch_ptr = bool(desc.kernel_code_properties & hsa.AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR)
        if enable_private_segment_sgpr:
            raise RuntimeError(f"kernel {kname} needs enable_private_segment_sgpr -- not "
                                f"implemented in GPUInterfaceTinyGPUHybridAMD.cpp's LaunchKernelImpl")

        max_private = max(max_private, private_segment_size)
        entries[kname] = dict(entry_offset=kd_addr + desc.kernel_code_entry_byte_offset,
                               rsrc1=rsrc1, rsrc2=rsrc2, rsrc3=rsrc3,
                               group_segment_size=group_segment_size,
                               kernarg_size=kernarg_size, wave32=wave32,
                               enable_dispatch_ptr=enable_dispatch_ptr)

    if max_private > scratch_cap:
        raise RuntimeError(f"a compiled kernel needs {max_private} bytes/thread of private "
                            f"(scratch) segment, exceeding the {scratch_cap}-byte cap "
                            f"amd_init_helper.py pre-allocated for — bump PRIVATE_SEGMENT_CAP "
                            f"there and re-run")

    # tmpring_size — formula from tinygrad's AMDDevice._ensure_has_local_memory
    # (ops_amd.py), computed here against the *actual* max private segment size
    # across all compiled kernels rather than the cap, so it isn't oversized.
    lanes_per_wave, mem_alignment_size = 64, 256
    size_per_thread = -(-max(max_private, 1) // (mem_alignment_size // lanes_per_wave)) * (mem_alignment_size // lanes_per_wave)
    size_per_xcc = size_per_thread * lanes_per_wave * max_slots_scratch_cu * cu_cnt
    wave_scratch = -(-(lanes_per_wave * size_per_thread) // mem_alignment_size)
    num_waves = (size_per_xcc // (wave_scratch * mem_alignment_size)) // se_cnt
    max_scratch_waves = cu_cnt * max_slots_scratch_cu
    tmpring_size = _tmpring_size(min(num_waves, max_scratch_waves), wave_scratch)

    with open(result_f, "w") as f:
        json.dump({"arch": arch, "code_b64": base64.b64encode(bytes(image)).decode(),
                   "tmpring_size": tmpring_size, "max_private_segment_size": max_private}, f)
        f.write("\n")
        for kname in sorted(entries):
            json.dump({"name": kname, **entries[kname]}, f)
            f.write("\n")

    print(f"amd_compile_helper: done — {len(entries)} kernels, image={len(image):#x} bytes, "
          f"max_private={max_private} tmpring_size=0x{tmpring_size:x}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
