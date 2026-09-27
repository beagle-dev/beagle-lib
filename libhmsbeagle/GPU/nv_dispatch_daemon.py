#!/usr/bin/env python3
"""
nv_dispatch_daemon.py — BEAGLE NV hybrid backend, daemon architecture
(STATUS.md §73/§75).

Replaced GPUInterfaceTinyGPUHybrid.cpp's hand-rolled GPFIFO/QMD dispatch
and is the default NV path. Real GPU operations (boot, compile, alloc, memcpy, launch, sync) run in this
resident daemon on tinygrad's NVDevice/NVProgram/HCQProgram.__call__, the
same architecture as amd_dispatch_daemon.py, and
GPUInterfaceTinyGPUHybridNV.cpp is a thin RPC client. The wrong-answer bug
that motivated the move was not dispatch-specific: no path, tinygrad's
NVProgram included, wrote the cbuf0 launch-dims words, and
BeagleNVProgram.__call__ now does (TODO.md Phase 140, STATUS.md §203).

Smaller gap than the AMD port in one real way, but not zero: AMDProgram
assumes one kernel per compiled ELF (BeagleAMDProgram patches around that,
amd_dispatch_daemon.py's own docstring). NVProgram (ops_nv.py) gets the
*code* lookup right for a multi-kernel ELF on its own (`.text.<name>`,
matched by exact kernel name) — but its *constant-buffer* lookup
(`.nv.constant<N>`) turns out not to be name-filtered at all, only harmless
in tinygrad's own normal one-kernel-per-ELF usage. BEAGLE compiles all
kernels for a given state count/precision into one PTX module, which
surfaces that gap for real — see BeagleNVProgram below (found via a real
hardware run coming back `logL=0.0` with no crash, STATUS.md §76) for the
one-line fix, same technique as BeagleAMDProgram.

Protocol: JSON command messages, each preceded by its byte length as a
4-byte little-endian uint32, on a dedicated socketpair (not the TinyGPU
socket: NVDevice("NV:0") makes its own connection internally, exactly like
STATUS.md §74's hardware-verified boot). amd_dispatch_daemon.py still uses
newline-terminated JSON. Commands carrying bulk data (h2d/d2h) are followed
immediately by that many raw bytes on the same stream. Kernel launches are batched from the start this time (cmd_launch_batch
only, no per-launch cmd_launch) — AMD's own profiling (STATUS.md AMD §26)
already found steady-state per-launch RPC overhead comparable to or larger
than the GPU dispatch work itself, no need to re-discover that here.

C++ dispatch (BEAGLE_NV_CPP_DISPATCH=1, TODO.md "Runtime roadmap", Step 3):
the plugin passes its own TinyGPU.app connection as a second argument, and
after compile_all sends "handoff". The daemon prepares every program, then
hands the C++ side what it needs to build QMDs and pushbuffers and submit
both GPFIFOs itself (build_handoff). From then on it only allocates.

C++ runtime (BEAGLE_NV_USE_DAEMON=0, the revived legacy path): the same
handoff with "programs": false, right after boot (no compile_all: the C++
side embeds its cubins, TODO.md plan step C1). The daemon then also sends
the device values program loading needs, and a VRAM pool; the C++ side
loads the programs and allocates from the pool itself, so after the handoff
this daemon only waits for "fini".

    python3 nv_dispatch_daemon.py <cmd_sock_fd> [<tinygpu_sock_fd>]
"""
import sys, os, json, struct, pathlib, time, socket

# Default: the tinygrad worktree pinned at a9830e2b4 -- tinygrad HEAD
# (after 2026-09-05) dropped the macOS TinyGPU transport and hcq1 (TODO.md
# Phase 140). TINYGRAD_PATH overrides.
_TINYGRAD_PATH = os.environ.get("TINYGRAD_PATH", str(pathlib.Path.home() / "Dropbox/Projects/tinygrad-hcq1"))
sys.path.insert(0, _TINYGRAD_PATH)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tinygrad.helpers import DEV
from tinygrad.runtime.support.hcq import HCQBuffer

import nv_compile_helper as nch  # reuse the already-verified compile_ptx()/extract_all_metadata()


# ── BeagleNVProgram: NVProgram.__init__'s body, copied verbatim, with ONE
# line fixed -- same technique amd_dispatch_daemon.py's BeagleAMDProgram uses
# for AMDProgram's own single-kernel-per-ELF assumption. Root cause (found
# after the first real hardware run of this daemon came back logL=0.0 with
# no crash/error -- STATUS.md §76): upstream NVProgram.__init__ scans every
# section in the ELF for one matching `.nv.constant<N>[.<kernel>]` and keeps
# OVERWRITING self.constbufs[N] unconditionally -- with no check that a
# per-kernel-suffixed section (`.nv.constant0.<kernelname>`, confirmed via a
# real ptxas compile of BEAGLE's actual multi-kernel PTX module: every
# kernel gets its own such section) actually belongs to *this* kernel
# (self.name). In tinygrad's own normal usage this never matters -- it only
# ever compiles one kernel per ELF, so there's only ever one such section.
# BEAGLE compiles all kernels for a given state count/precision into one PTX
# module (matching this daemon's whole reason for not needing a
# BeagleAMDProgram-equivalent for the *code* lookup, which real NVProgram
# already does correctly via `.text.<name>`) -- so this loop runs once per
# ELF section across ALL 80 kernels, and every NVProgram instance ends up
# pointing constant buffer 0 at whichever kernel's section happens to be
# last in iteration order, not its own. Empirically confirmed against a real
# compile of BEAGLE's actual SP-4 kernel set: every kernel's constbuf0
# resolved to the identical (addr, size) under the unpatched logic; the fix
# below gives each kernel its own distinct, correct address. This explains
# the observed symptom exactly -- every kernel dispatch read its arguments
# (pointers + int scalars) from an unrelated kernel's constant-buffer
# region, so real per-kernel computation never happened, yet nothing faults
# (that region is still valid, zero-initialized VRAM within the same
# uploaded image) -- consistent with a clean run, no GSP exception, and
# logL=0.0 rather than a crash or NaN.
#
# Fix: the `.nv.constant<N>` match now also captures an optional
# `.<kernelname>` suffix and only applies when it's either absent (matches
# upstream's original single-kernel-ELF behavior unchanged) or equals this
# kernel's own name. Every other line is copied verbatim from
# NVProgram.__init__ (ops_nv.py) -- including its own module-global `nv_gpu`
# lookup (via `import tinygrad.runtime.ops_nv as ops_nv`, not a snapshot at
# import time, so it stays correct if `ops_nv.nv_gpu` is ever reassigned
# based on driver version, exactly like the real class's own bare-name
# lookup does).
import tinygrad.runtime.ops_nv as ops_nv
import tinygrad.runtime.support.nv.ip as ip_nv   # NVRpcQueue, for level gsp_hw's unload (plan step C8)


class BeagleNVProgram(ops_nv.NVProgram):
    def __init__(self, dev, obj):
        import re as _re, ctypes as _ctypes, struct as _struct
        from tinygrad.helpers import round_up, data64_le, hi32, lo32
        from tinygrad.device import BufferSpec
        from tinygrad.runtime.support.elf import elf_loader
        from tinygrad.runtime.autogen import libc as _libc
        import weakref as _weakref

        self.dev, self.name, self.lib = dev, obj.name, obj.lib
        self.constbufs = {0: (0, 0x160)}

        NAK = isinstance(dev.renderer, ops_nv.NAKRenderer)
        my_sym_idx = None  # resolved below, real (non-NAK/MOCK) path only
        if NAK:
            image, self.cbuf_0 = memoryview(bytearray(obj.lib[_ctypes.sizeof(info:=ops_nv.mesa.struct_nak_shader_info.from_buffer_copy(obj.lib)):])), []
            self.regs_usage, self.shmem_usage, self.lcmem_usage = info.num_gprs, round_up(info.cs.smem_size, 128), round_up(info.slm_size, 16)
        elif isinstance(dev.iface, ops_nv.MOCKIface):
            image, sections, relocs = memoryview(bytearray(obj.lib) + b'\x00' * (4 - len(obj.lib) % 4)).cast("I"), [], []
        else:
            image, sections, relocs = elf_loader(self.lib, force_section_align=128)
            # -- BEAGLE fix vs. upstream (see class docstring above): the
            # EIATTR_REGCOUNT (0x2f) / EIATTR_MIN_STACK_SIZE (0x12) entries
            # below live in one bare ".nv.info" section shared by every
            # kernel in BEAGLE's multi-kernel-per-compile ELF -- one entry
            # per kernel, each tagged with that kernel's ELF symbol-table
            # index as its own first 4-byte field (confirmed against a real
            # compile: field matches this kernel's real symtab index
            # exactly, and its paired regcount matches real ptxas -v output
            # for that same kernel). Upstream's original loop (correct for
            # its own one-kernel-per-ELF usage) discards that index and just
            # keeps overwriting self.regs_usage/lcmem_usage with whichever
            # entry happens to be last in the section -- same root-cause
            # class as the constant-buffer bug fixed above, just not caught
            # by that fix since these two attributes aren't ever duplicated
            # into a per-kernel-suffixed ".nv.info.<name>" section the way
            # EIATTR_PARAM_CBANK (0xa) is. Resolve this kernel's own symtab
            # index once here so the loop below can filter by it.
            symtab_sh = next((sh for sh in sections if sh.header.sh_type == _libc.SHT_SYMTAB), None)
            if symtab_sh is not None:
                strtab_sh = sections[symtab_sh.header.sh_link]
                symtab = (_libc.Elf64_Sym * (symtab_sh.header.sh_size // symtab_sh.header.sh_entsize)).from_buffer_copy(symtab_sh.content)
                for i, sym in enumerate(symtab):
                    if strtab_sh.content[sym.st_name:strtab_sh.content.find(b'\x00', sym.st_name)].decode('utf-8') == self.name:
                        my_sym_idx = i
                        break

        self.lib_gpu = self.dev.allocator.alloc(round_up((prog_sz:=image.nbytes), 0x1000) + 0x1000, buf_spec:=BufferSpec(nolru=True))
        prog_addr = self.lib_gpu.va_addr
        if not NAK:
            self.regs_usage, self.shmem_usage, self.lcmem_usage, cbuf0_size = 0, 0x400, 0x240, 0x160 if isinstance(dev.iface, ops_nv.MOCKIface) else 0
            for sh in sections:
                if sh.name == f".nv.shared.{self.name}": self.shmem_usage = round_up(0x400 + sh.header.sh_size, 128)
                if sh.name == f".text.{self.name}": prog_addr, prog_sz = self.lib_gpu.va_addr + sh.header.sh_addr, sh.header.sh_size
                # -- BEAGLE fix vs. upstream (see class docstring above): require
                # an absent or matching kernel-name suffix before accepting a
                # constant-buffer section match.
                elif m := _re.match(r'\.nv\.constant(\d+)(?:\.(.+))?$', sh.name):
                    suffix = m.group(2)
                    if suffix is None or suffix == self.name:
                        self.constbufs[int(m.group(1))] = (self.lib_gpu.va_addr + sh.header.sh_addr, sh.header.sh_size)
                elif sh.name.startswith(".nv.info"):
                    for typ, param, data in self._parse_elf_info(sh):
                        if sh.name == f".nv.info.{obj.name}" and param == 0xa: cbuf0_size = _struct.unpack_from("IH", data)[1]
                        # -- BEAGLE fix vs. upstream: require this entry's own
                        # embedded symbol index (see my_sym_idx above) to match
                        # this kernel's -- not just any entry from the shared
                        # ".nv.info" section.
                        elif sh.name == ".nv.info" and param == 0x12 and my_sym_idx is not None \
                                and _struct.unpack_from("II", data)[0] == my_sym_idx:
                            self.lcmem_usage = _struct.unpack_from("II", data)[1] + 0x240
                        elif sh.name == ".nv.info" and param == 0x2f and my_sym_idx is not None \
                                and _struct.unpack_from("II", data)[0] == my_sym_idx:
                            self.regs_usage = _struct.unpack_from("II", data)[1]

            for apply_image_offset, rel_sym_offset, typ, _addend in relocs:
                if typ == 2: image[apply_image_offset:apply_image_offset+8] = _struct.pack('<Q', self.lib_gpu.va_addr + rel_sym_offset)
                elif typ == 0x38: image[apply_image_offset+4:apply_image_offset+8] = _struct.pack('<I', (self.lib_gpu.va_addr + rel_sym_offset) & 0xffffffff)
                elif typ == 0x39: image[apply_image_offset+4:apply_image_offset+8] = _struct.pack('<I', (self.lib_gpu.va_addr + rel_sym_offset) >> 32)
                else: raise RuntimeError(f"unknown NV reloc {typ}")

            min_cbuf0_entries = 224 if dev.iface.compute_class >= ops_nv.nv_gpu.BLACKWELL_COMPUTE_A else 12
            self.cbuf_0 = [0] * max(cbuf0_size // 4, min_cbuf0_entries)

        self.dev._ensure_has_local_memory(self.lcmem_usage)
        self.dev.allocator._copyin(self.lib_gpu, image)
        self.dev.synchronize()

        if dev.iface.compute_class >= ops_nv.nv_gpu.BLACKWELL_COMPUTE_A:
            if not NAK: self.cbuf_0[188:192], self.cbuf_0[223] = [*data64_le(self.dev.shared_mem_window), *data64_le(self.dev.local_mem_window)], 0xfffdc0
            qmd = {'qmd_major_version':5, 'qmd_type':ops_nv.nv_gpu.NVCEC0_QMDV05_00_QMD_TYPE_GRID_CTA, 'program_address_upper_shifted4':hi32(prog_addr>>4),
                'program_address_lower_shifted4':lo32(prog_addr>>4), 'register_count':self.regs_usage, 'shared_memory_size_shifted7':self.shmem_usage>>7,
                f'shader_local_memory_{"low" if NAK else "high"}_size_shifted4': self.dev.slm_per_thread>>4}
        else:
            if not NAK: self.cbuf_0[6:12] = [*data64_le(self.dev.shared_mem_window), *data64_le(self.dev.local_mem_window), *data64_le(0xfffdc0)]
            qmd = {'qmd_major_version':3, 'sm_global_caching_enable':1, 'program_address_upper':hi32(prog_addr), 'program_address_lower':lo32(prog_addr),
                'shared_memory_size':self.shmem_usage, 'register_count_v':self.regs_usage,
                f'shader_local_memory_{"low" if NAK else "high"}_size':self.dev.slm_per_thread}

        # cbuf_0 words the CUDA driver fills with the launch dims on every
        # launch, and that ptxas-compiled code reads %ntid/%nctaid
        # (blockDim/gridDim) from. Upstream never writes them (tinygrad's own
        # kernels bake dims in as constants), so they stay 0 and e.g.
        # kernelMatrixMulADB's BLOCKS=gridDim.y reads 0 -> EDGE=20. Offsets
        # SASS-verified with ptxas 12.8: sm_86/89/90 blockDim.xyz =
        # c[0x0][0x0..0x8], gridDim.xyz = c[0x0][0xc..0x14] (words 0-2, 3-5);
        # sm_100/120 blockDim = c[0x0][0x360..0x368], gridDim =
        # c[0x0][0x370..0x378] (words 216-218, 220-222; 219 is not a dim).
        # On by default (TODO.md Phase 140: fixes kernelMatrixMulADB, 320/320
        # and full pipeline 20/20 on sm_89); BEAGLE_NV_FILL_LAUNCH_DIMS=0
        # disables it, for A/B only. Written per launch in __call__ below.
        if NAK or isinstance(dev.iface, ops_nv.MOCKIface): self._dims_idx = None
        elif dev.iface.compute_class >= ops_nv.nv_gpu.BLACKWELL_COMPUTE_A: self._dims_idx = (216, 220)
        else: self._dims_idx = (0, 3)
        self.fill_launch_dims = self._dims_idx is not None and os.environ.get("BEAGLE_NV_FILL_LAUNCH_DIMS", "1") != "0"
        if self._dims_idx is not None:
            log(f"launch-dims fill {'ON' if self.fill_launch_dims else 'OFF (BEAGLE_NV_FILL_LAUNCH_DIMS=0)'} [{self.name}]: "
                f"cbuf_0 blockDim@{self._dims_idx[0]} gridDim@{self._dims_idx[1]}")

        smem_cfg = min(shmem_conf * 1024 for shmem_conf in [32, 64, 100] if shmem_conf * 1024 >= self.shmem_usage) // 4096 + 1

        self.qmd = ops_nv.QMD(dev, **qmd, qmd_group_id=0x3f, invalidate_texture_header_cache=1, invalidate_texture_sampler_cache=1,
            invalidate_texture_data_cache=1, invalidate_shader_data_cache=1, api_visible_call_limit=1, sampler_index=1, barrier_count=1,
            cwd_membar_type=ops_nv.nv_gpu.NVC6C0_QMDV03_00_CWD_MEMBAR_TYPE_L1_SYSMEMBAR, constant_buffer_invalidate_0=1, min_sm_config_shared_mem_size=smem_cfg,
            target_sm_config_shared_mem_size=smem_cfg, max_sm_config_shared_mem_size=0x1a, program_prefetch_size=min(prog_sz>>8, 0x1ff),
            sass_version=dev.sass_version, program_prefetch_addr_upper_shifted=prog_addr>>40, program_prefetch_addr_lower_shifted=prog_addr>>8)

        for i, (addr, sz) in self.constbufs.items():
            self.qmd.set_constant_buf_addr(i, addr)
            self.qmd.write(**{f'constant_buffer_size_shifted4_{i}': sz, f'constant_buffer_valid_{i}': 1})

        self.max_threads = ((65536 // round_up(max(1, self.regs_usage) * 32, 256)) // 4) * 4 * 32

        super(ops_nv.NVProgram, self).__init__(ops_nv.NVArgsState, self.dev, obj, kernargs_alloc_size=round_up(self.constbufs[0][1], 1 << 8) + (8 << 8))
        _weakref.finalize(self, self._fini, self.dev, self.lib_gpu, buf_spec)

    def set_launch_dims(self, global_size, local_size):
        # fill_kernargs copies self.cbuf_0 into a fresh kernargs slot
        # synchronously, so rewriting it per launch is safe even with
        # wait=False. Zeroed (upstream's value) when the fill is off, so
        # toggling fill_launch_dims within one process stays exact.
        if self._dims_idx is not None:
            b, g = self._dims_idx
            if self.fill_launch_dims:
                self.cbuf_0[b:b+3] = list(local_size) + [1] * (3 - len(local_size))
                self.cbuf_0[g:g+3] = list(global_size) + [1] * (3 - len(global_size))
            else:
                self.cbuf_0[b:b+3] = self.cbuf_0[g:g+3] = [0, 0, 0]

    def check_launch(self, global_size, local_size):
        # NVProgram.__call__'s launch checks (ops_nv.py), for the chained path
        # in Daemon.cmd_launch_batch, which bypasses __call__.
        from tinygrad.helpers import prod
        if prod(local_size) > 1024 or self.max_threads < prod(local_size) or self.lcmem_usage > self.dev.slm_per_thread:
            raise RuntimeError(f"Too many resources requested for launch, {prod(local_size)=}, {self.max_threads=}")
        if any(cur > mx for cur, mx in zip(global_size, [2147483647, 65535, 65535])) or \
           any(cur > mx for cur, mx in zip(local_size, [1024, 1024, 64])):
            raise RuntimeError(f"Invalid global/local dims {global_size=}, {local_size=}")

    def __call__(self, *bufs, global_size=(1,1,1), local_size=(1,1,1), vals=(), wait=False, timeout=None):
        self.set_launch_dims(global_size, local_size)
        return super().__call__(*bufs, global_size=global_size, local_size=local_size, vals=vals, wait=wait, timeout=timeout)


def log(msg):
    try: print(f"[nv_dispatch_daemon] {msg}", file=sys.stderr, flush=True)
    except Exception: pass   # a failed write (a full disk, a closed stderr) never stops a decision, a hold above all


# Opt-in timing (BEAGLE_NV_PROFILE=1), the daemon half of the C++ side's
# RPC round-trip profiling (GPUInterfaceTinyGPUHybridNV.cpp). Aggregated per
# label and logged at fini: "cmd.*" is each command handler, "launch.*" each
# kernel launch inside launch_batch (plus one submit per batch when
# chained), "wire.*" the framing (reading a message, measured from its first
# bytes; json.loads).
_PROFILE = bool(os.environ.get("BEAGLE_NV_PROFILE"))
_prof = {}  # label -> [count, total_s, min_s, max_s]

# One chained compute queue per launch_batch (default); see cmd_launch_batch.
_CHAIN_LAUNCHES = os.environ.get("BEAGLE_NV_CHAIN_LAUNCHES", "1") != "0"


def _prof_add(label, dt):
    s = _prof.get(label)
    if s is None:
        _prof[label] = [1, dt, dt, dt]
    else:
        s[0] += 1; s[1] += dt; s[2] = min(s[2], dt); s[3] = max(s[3], dt)


def _prof_report():
    log("[profile] daemon side:")
    for label, (n, tot, lo, hi) in sorted(_prof.items()):
        log(f"[profile]   {label:20s} n={n:7d}  mean={tot / n * 1e6:9.1f} us  min={lo * 1e6:9.1f}  "
            f"max={hi * 1e6:10.1f}  total={tot * 1e3:9.1f} ms")


class _Profiled:
    __slots__ = ("label", "t0")
    def __init__(self, label):
        self.label = label
    def __enter__(self):
        if _PROFILE:
            self.t0 = time.perf_counter()
        return self
    def __exit__(self, *exc):
        if _PROFILE:
            _prof_add(self.label, time.perf_counter() - self.t0)


def _apply_boot_safety_patches():
    """
    Boot-safety patches for the macOS TinyGPU eGPU, hardware-verified in
    STATUS.md §74: importing nv_init_helper applies its GSP/RM boot patches
    as a module side effect, and APLRemotePCIDevice.reset (a PCIe FLR, which
    panics macOS over USB4) becomes a no-op. Rationale: nv_init_helper.py's
    InheritedFDPCIDevice.reset().
    """
    import nv_init_helper  # noqa: F401 — GSP/RM boot patches (WPR2-reset-loop
    # suppression, palloc zero-size limit, GC6 BSI sleep) applied as a module-
    # level side effect; main() is never called.
    from tinygrad.runtime.support.system import APLRemotePCIDevice
    def _safe_reset(self):
        log("PCIe FLR suppressed (macOS eGPU safety) — see nv_init_helper.py InheritedFDPCIDevice.reset()")
    APLRemotePCIDevice.reset = _safe_reset


def _install_inherited_tinygpu(tgpu_fd):
    """
    C++ dispatch: run tinygrad over the plugin's own TinyGPU.app connection
    (inherited as tgpu_fd) instead of opening a second one. TinyGPU.app
    serves one client at a time, and after cmd_handoff the C++ side writes
    GPFIFO entries and doorbells on this same connection. The two sides never
    use it at once, because every daemon command is synchronous. Also keeps a
    dup of each MAP_SYSMEM_FD fd, which hcq1's alloc_sysmem closes right
    after mapping it, so cmd_handoff can pass buffers to the C++ side.
    """
    import mmap, itertools
    from tinygrad.helpers import ceildiv
    from tinygrad.runtime.support import system
    from tinygrad.runtime.support.hcq import FileIOInterface, MMIOInterface

    class BeagleTinyGPUDevice(system.APLRemotePCIDevice):
        def __init__(self, devpref, pcibus):
            # No new connection, no lock file and no buffer sizes (APLRemotePCIDevice/RemotePCIDevice.__init__): the plugin's
            # TinyGPU.app client set the buffers when it connected, as RemotePCIDevice.__init__ does (TODO.md plan step C3),
            # and macOS refuses a second setting (ENOBUFS)
            self.sock = socket.socket(fileno=os.dup(tgpu_fd))
            self.pcibus, self.dev_id, self.peer_group, self.lock_fd = "usb4", 0, "usb4", None
            self.sysmem_fds = {}  # host address of a sysmem mapping -> dup of its fd

        def alloc_sysmem(self, size, vaddr=0, contiguous=False):
            # APLRemotePCIDevice.alloc_sysmem, plus the fd dup.
            mapped_size, _, _, fd = self._rpc(self.sock, self.dev_id, system.RemoteCmd.MAP_SYSMEM_FD, size, int(contiguous), has_fd=True)
            keep = os.dup(fd)
            memview = MMIOInterface(FileIOInterface(fd=fd).mmap(0, mapped_size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, 0),
                                    mapped_size, fmt='B')
            self.sysmem_fds[memview.addr] = keep
            paddrs_raw = list(itertools.takewhile(lambda p: p[1] != 0, zip(memview.view(fmt='Q')[0::2], memview.view(fmt='Q')[1::2])))
            return memview, [p + i for p, sz in paddrs_raw for i in range(0, sz, 0x1000)][:ceildiv(size, 0x1000)]

    system.APLRemotePCIDevice = BeagleTinyGPUDevice  # System.list_devices looks the name up at call time


# ── C++ dispatch handoff (TODO.md "Runtime roadmap", Step 3). After
# cmd_handoff the C++ side builds QMDs and pushbuffers itself and submits both
# GPFIFOs over the shared TinyGPU.app connection. build_handoff describes
# everything its encoder (TinyGPUHybridNVDispatch.h) needs, taken from the
# same tinygrad objects and tables hcq1 would have used, so the two cannot
# drift apart: QMD field positions, method and flag words, GPFIFO/doorbell
# BAR offsets, the C++ side's buffers, and per kernel the QMD template and
# cbuf0 prefix. ─────────────────────────────────────────────────────────────
_NO_DIMS = 0xffffffff


def build_handoff(dev, progs, bufs):
    from tinygrad.helpers import round_up
    nv_gpu, nv_flags = ops_nv.nv_gpu, ops_nv.nv_flags
    qmd = ops_nv.QMD(dev)
    fields, v5 = ops_nv.QMD.fields[qmd.pref], qmd.ver >= 4
    def bits(name): return fields[name.upper()]           # (hi, lo), for QMD._rw_bits
    def byte(name): return fields[name.upper()][1] // 8   # QMD.field_offset
    info = {"qmd_ver": qmd.ver, "qmd_bytes": qmd.sz * 4,
            # NVComputeQueue.exec/.signal write these as plain stores at byte offsets...
            "q_grid": byte("grid_width" if v5 else "cta_raster_width"),
            "q_block01": byte("cta_thread_dimension0"), "q_block2": byte("cta_thread_dimension2"),
            "q_rel_addr": byte("release_semaphore0_addr_lower" if v5 else "release0_address_lower"),
            "q_rel_payload": byte("release_semaphore0_payload_lower" if v5 else "release0_payload_lower"),
            "q_cb_shift": 6 if v5 else 0}
    # ...and these as bitfields
    for key, name in (("cb_hi", "constant_buffer_addr_upper_shifted6_0" if v5 else "constant_buffer_addr_upper_0"),
                      ("cb_lo", "constant_buffer_addr_lower_shifted6_0" if v5 else "constant_buffer_addr_lower_0"),
                      ("rel_en", "release0_enable"), ("dep_ptr", "dependent_qmd0_pointer"),
                      ("dep_action", "dependent_qmd0_action"), ("dep_prefetch", "dependent_qmd0_prefetch"),
                      ("dep_enable", "dependent_qmd0_enable")):
        info[f"q_{key}_hi"], info[f"q_{key}_lo"] = bits(name)
    # method and flag words: NVCommandQueue.wait/setup, NVComputeQueue.memory_barrier/exec/signal, NVCopyQueue.copy/signal
    info.update(m_sem_addr_lo=nv_gpu.NVC56F_SEM_ADDR_LO,
                f_sem_acquire=nv_flags("NVC56F_SEM_EXECUTE", operation="acq_circ_geq", payload_size="64bit"),
                m_invalidate=nv_gpu.NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI,
                f_invalidate=nv_flags("NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI", instruction="true", global_data="true", constant="true"),
                m_pcas_a=nv_gpu.NVC6C0_SEND_PCAS_A, m_pcas2_b=nv_gpu.NVC6C0_SEND_SIGNALING_PCAS2_B,
                m_dma_offset_in_upper=nv_gpu.NVC6B5_OFFSET_IN_UPPER, m_dma_line_length_in=nv_gpu.NVC6B5_LINE_LENGTH_IN,
                m_dma_launch=nv_gpu.NVC6B5_LAUNCH_DMA, m_dma_sem_a=nv_gpu.NVC6B5_SET_SEMAPHORE_A,
                f_dma_copy=nv_flags("NVC6B5_LAUNCH_DMA", data_transfer_type="non_pipelined", src_memory_layout="pitch",
                                    dst_memory_layout="pitch"),
                f_dma_sem=nv_flags("NVC6B5_LAUNCH_DMA", flush_enable="true", semaphore_type="release_four_word_semaphore"),
                m_local_mem_a=nv_gpu.NVC6C0_SET_SHADER_LOCAL_MEMORY_A,
                m_local_mem_nt_a=nv_gpu.NVC6C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A,
                f_sem_release=nv_flags("NVC56F_SEM_EXECUTE", operation="release", release_wfi="en", payload_size="64bit",
                                       release_timestamp="en"),
                m_non_stall_interrupt=nv_gpu.NVC56F_NON_STALL_INTERRUPT)
    # GPFIFOs and doorbell as TinyGPU.app BAR offsets (NVCommandQueue._submit_to_gpfifo)
    for key, fifo in (("c", dev.compute_gpfifo), ("d", dev.dma_gpfifo)):
        info.update({f"{key}_ring_bar": fifo.ring.residx, f"{key}_ring_off": fifo.ring.off, f"{key}_gpput_bar": fifo.gpput.residx,
                     f"{key}_gpput_off": fifo.gpput.off, f"{key}_entries": fifo.entries_count, f"{key}_put": fifo.put_value,
                     f"{key}_token": fifo.token})
    info.update(db_bar=dev.gpu_mmio.residx, db_off=dev.gpu_mmio.off + 0x90)
    for name, b in bufs.items():
        info[f"{name}_va"], info[f"{name}_size"] = b.va_addr, b.size
    # per kernel: 7 x u32 header, name, QMD template, cbuf0 prefix (launch-dims words zeroed: C++ fills them per launch)
    blob = bytearray()
    for p in progs:
        if p.qmd.read("release0_enable"): raise RuntimeError(f"{p.name}: QMD template already uses release0")
        if p.lcmem_usage > dev.slm_per_thread: raise RuntimeError(f"{p.name}: needs more local memory than was set up")
        prefix = list(p.cbuf_0)
        if p._dims_idx is not None:
            for i in range(3): prefix[p._dims_idx[0] + i] = prefix[p._dims_idx[1] + i] = 0
        dims = p._dims_idx if p._dims_idx is not None and p.fill_launch_dims else (_NO_DIMS, _NO_DIMS)
        name = p.name.encode()
        blob += struct.pack("<7I", len(name), round_up(p.constbufs[0][1], 1 << 8), p.kernargs_alloc_size, len(prefix),
                            dims[0], dims[1], p.max_threads)
        blob += name + bytes(p.qmd.mv) + struct.pack(f"<{len(prefix)}I", *prefix)
    info["nkernels"] = len(progs)
    return info, bytes(blob)


def check_vram_below_wpr(dev_impl):
    """Plan step P3's interim WPR check (C6 makes it permanent in C++): every VRAM allocation so far, the C++ runtime's
    pool included, must end at or below GspFwWprMeta.gspFwRsvdStart, where GSP-RM's reserved region starts (ip.py:448-452).
    tinygrad's PA allocator reaches vram_size - 64 MiB (nvdev.py:146, memory.py:190-192), about 130 MiB into that region
    on the 8188 MiB RTX 4060. The WPR meta is host memory on a small-BAR card (nvdev.py:151-153), so this reads nothing
    from the GPU. Returns (end, gspFwRsvdStart). On FMC-booted chips (Blackwell) tinygrad leaves gspFwRsvdStart 0 and the
    FMC places the reserved region itself (ip.py:444-446: its heaps and reservations total about 162 MiB plus the GSP image,
    below the end of VRAM), so the bound is a static vram_size - 512 MiB there, about twice that (plan step B1)."""
    rsvd = _wpr_bound(dev_impl)
    pa = dev_impl.mm.pa_allocator
    end = max((pa.base + start + size for start, (size, _, _, free) in pa.blocks.items() if not free), default=0)
    if end > rsvd:
        raise RuntimeError(f"VRAM allocations end at {end:#x}, above {_wpr_bound_name(dev_impl)} {rsvd:#x}, where GSP-RM's reserved "
                           f"region starts (C++ runtime: lower BEAGLE_NV_DATA_MB)")
    return end, rsvd


def _wpr_bound(dev_impl):
    import ctypes
    from tinygrad.runtime.autogen import nv
    if dev_impl.fmc_boot: return dev_impl.vram_size - (512 << 20)
    return nv.GspFwWprMeta.from_buffer_copy(bytes(dev_impl.gsp.wpr_meta[:ctypes.sizeof(nv.GspFwWprMeta)])).gspFwRsvdStart


def _wpr_bound_name(dev_impl):
    return "vram_size - 512 MiB" if dev_impl.fmc_boot else "gspFwRsvdStart"


# TODO.md plan step C6: at BEAGLE_NV_CPP_LEVEL=vram the C++ side allocates its VRAM pool itself, with tinygrad's memory
# manager ported to C++ (TinyGPUMemory.h, TinyGPUHybridNVMemory.h), and at sysmem also the handoff's four buffers, which it
# otherwise gets from here (_HANDOFF_BUFS, in this order: the C++ side's GPUInterfaceTinyGPUHybridNV.cpp kNVDBuffers is the
# same list). The handoff reply then carries tinygrad's memory manager as it is (_mm_export), and this process allocates
# nothing more: the C++ side continues from exactly this state, so it sends TinyGPU.app what this process would have.
_HANDOFF_BUFS = (("cmdq", 2 << 20, dict(cpu_access=True)),                           # pushbuffers of both queues
                 ("kargs", 16 << 20, dict(cpu_access=True)),                         # kernargs slots: cbuf0 + args, then the QMD
                 ("staging", 16 << 20, dict(cpu_access=True)),                       # h2d/d2h bounce buffer
                 ("signal", 0x1000, dict(host=True, uncached=True, cpu_access=True)))   # C++ timeline
_NONE = (1 << 64) - 1   # None in a TLSF state


def _tlsf_save(a):
    """A TLSFAllocator's state as TinyGPUMemory.h's TLSFAllocator::save writes it: size, base, block_size, l2_cnt; the blocks
    by start (start, size, next, prev, free); the non-empty buckets (lv1, lv2, count, starts oldest first); lv1_entries."""
    w = [a.size, a.base, a.block_size, a.l2_cnt, len(a.blocks)]
    for start in sorted(a.blocks):
        size, nxt, prev, free = a.blocks[start]
        w += [start, size, _NONE if nxt is None else nxt, _NONE if prev is None else prev, int(free)]
    buckets = [(l1, l2, lst) for l1, d in enumerate(a.storage) for l2, lst in sorted(d.items()) if lst]
    w.append(len(buckets))
    for l1, l2, lst in buckets: w += [l1, l2, len(lst), *lst]
    return w + [len(a.lv1_entries), *a.lv1_entries]


def _mm_export(dev):
    """Plan step C6: tinygrad's memory manager as TinyGPUHybridNVMemory.h's nv_mm_import restores it: NVMemoryManager's
    configuration, its three allocators and the class's VA allocator, the root page table; and what PCIIfaceBase.alloc and
    the plugin's checks use: mmap.PAGESIZE, GMMU, the WPR bound, NVDev.vram_size, BAR1's size (bar_info is cached, so this
    sends nothing) and how many sysmem allocations this process made on the shared connection (TinyGPU.app keeps 128)."""
    import mmap
    from tinygrad.helpers import getenv
    impl, pci = dev.iface.dev_impl, dev.iface.pci_dev
    mm = impl.mm
    return {"mm_mmu_ver": impl.mmu_ver, "mm_vram_size": mm.vram_size, "mm_va_bits": mm.va_bits, "mm_va_shifts": list(mm.va_shifts),
            "mm_va_base": mm.va_base, "mm_palloc_ranges": [x for r in mm.palloc_ranges for x in r], "mm_reserve_ptable": int(mm.reserve_ptable),
            "mm_root": mm.root_page_table.paddr, "mm_root_lv": mm.root_page_table.lv, "mm_boot": _tlsf_save(mm.boot_allocator),
            "mm_ptable": _tlsf_save(mm.ptable_allocator), "mm_pa": _tlsf_save(mm.pa_allocator), "mm_va": _tlsf_save(mm.va_allocator),
            "mm_pagesize": mmap.PAGESIZE, "mm_gmmu": getenv("GMMU", 1), "mm_wpr_bound": _wpr_bound(impl), "mm_dev_vram_size": impl.vram_size,
            "mm_sysmem_count": len(pci.sysmem_fds), "bar1_size": pci.bar_info(1)[1]}


# TODO.md plan step C7: at BEAGLE_NV_CPP_LEVEL=rm this process boots only the NVDev, the GSP included, and the C++ side builds the
# NVDevice with tinygrad's RM client ported to C++ (TinyGPUHybridNVRM.h, TinyGPUHybridNVDevice.h). _boot_nvdev_only runs
# tinygrad's own PCIIface.__init__ (ops_nv.py:557-568) and stops it at its first RM call, the root client's allocation, which the
# C++ side makes: so this process sends TinyGPU.app exactly what NVDevice's boot sends up to there. (NVDevice._select_iface tries
# NVKIface first, which sends nothing here.) _RMDevice stands in for the NVDevice in the fini, EOF, state-page and export paths:
# the booted PCIIface; no timeline of this process's own, so synchronize has nothing to wait for; and finalize as
# HCQCompiled.finalize ends, with device_fini (NVDev.fini: the GSP unload, then unless BEAGLE_NV_TEARDOWN=0 NVIDIA's teardown).
# Plan step C8, BEAGLE_NV_CPP_LEVEL=gsp_hw: NVDev.__init__'s last statement, NV_GSP.init_hw (waiting for GSP-RM's INIT_DONE, with its
# CPU sequencer, then init_golden_image), is the C++ side's too: here it does nothing, so the boot stops right after flcn.init_hw
# started GSP-RM (booter_load); PCIIfaceBase.__init__'s remaining statement (list_devices, IOKit) sends TinyGPU.app nothing.
# Plan step C9, flcn_hw: NV_FLCN.init_hw (FWSEC-FRTS, then booter_load) is the C++ side's as well: the boot stops after both
# init_sw calls, which prepared the images, the GSP's boot structures and the prequeued RPCs. On the COT boot (plan step B2)
# NV_FLCN_COT.init_hw, the COT message to the FSP, is the C++ side's in the same way.
class _RMDevice:
    error_state = None
    def __init__(self, iface, level="rm"): self.iface, self.level = iface, level
    def synchronize(self): pass
    def finalize(self): self.iface.device_fini()
    def __repr__(self): return f"<the NVDev of {self.iface.dev_impl.chip_name}, level {self.level}>"


def _boot_nvdev_only(level="rm"):
    from tinygrad.runtime.support.nv.ip import NV_FLCN, NV_GSP
    class _Fork(Exception): pass
    class NVDevice: pass   # PCIIfaceBase names the device after its class: "NV"
    def fork(*a, **k): raise _Fork
    iface = ops_nv.PCIIface.__new__(ops_nv.PCIIface)
    saved, ops_nv.PCIIface.rm_alloc = ops_nv.PCIIface.rm_alloc, fork
    from tinygrad.runtime.support.nv.ip import NV_FLCN_COT
    skipped = {"rm": (), "gsp_hw": (NV_GSP,), "flcn_hw": (NV_FLCN, NV_FLCN_COT, NV_GSP)}[level]
    saved_init_hw = {c: c.init_hw for c in skipped}   # NV_GSP's is nv_init_helper's _patched_gsp_init_hw
    def flcn_first_statement(self): self.falcon, self.sec2 = 0x00110000, 0x00840000   # ip.py:187, which NV_FLCN.reset reads
    def cot_first_statement(self): self.falcon = 0x00110000                          # ip.py:312
    for c in skipped: c.init_hw = {NV_FLCN: flcn_first_statement, NV_FLCN_COT: cot_first_statement}.get(c, lambda self: None)
    try: ops_nv.PCIIface.__init__(iface, NVDevice(), 0)
    except _Fork: pass
    finally:
        ops_nv.PCIIface.rm_alloc = saved
        for c, f in saved_init_hw.items(): c.init_hw = f
    return iface


# The C++ side's state page (TODO.md plan step P3; GPUInterfaceTinyGPUHybridNV.cpp nvdStatePage): five u64 words
# [phase, frame_in_flight, last_submitted, seq, keeper] in a POSIX shm segment the plugin creates, unlinks and passes here right
# after the handoff. Read only when the C++ side can no longer write: at fini (it has idled or hung) and after EOF (it is
# gone). Plan step C5 added seq, the GSP command queue's sequence number after the C++ side's last RPC, and phase 2; plan step
# C10 the keeper word, which the C++ side sets before it asks this process to hand its role to its crash guard (cmd_release),
# and which _eof reads too.
_STATE_WORDS, _PHASE_DISPATCH, _PHASE_TEARDOWN = 5, 1, 2   # 1: the C++ side owns both GPFIFOs; 2: it is unloading the GPU itself
_KEEPER_GUARD = 1     # plan step C10: the keeper word (the page's fifth): the crash guard keeps the GPU, not this process
_PHASE_GSP_INIT = 3   # plan step C8 (level gsp_hw): the C++ side is booting GSP-RM (NV_GSP.init_hw), before it read GSP_INIT_DONE
_PHASE_FLCN_INIT = 4  # plan step C9 (level flcn_hw): the C++ side runs FWSEC-FRTS; booter_load has not started GSP-RM


class Daemon:
    def __init__(self, sock, tgpu_fd=None):
        self.sock = sock
        self.tgpu_fd = tgpu_fd    # the C++ side's TinyGPU.app connection (C++ dispatch only)
        self.handed_off = False   # set by cmd_handoff: the C++ side owns both GPFIFOs from then on
        self.dev = None
        self.elf_bytes = None     # last-compiled multi-kernel ELF (real ptxas cubin)
        self.kernel_names = set() # names found in elf_bytes, for GetFunction-style validation
        self.programs = {}        # (name, n_int_args) -> NVProgram
        self._allocs = {}
        self._state = self._cpp_signal = None   # the C++ side's state page and timeline (cmd_state_page, plan step P3)
        self.mm_exported = False  # set by cmd_handoff at level vram or sysmem: the C++ side owns the memory manager (plan step C6)
        self.released = False                         # plan step C10: the crash guard keeps the GPU (cmd_release)
        self.rm_level, self.rm_exported = "", False   # plan step C7: booted the NVDev only ("rm"; "gsp_hw", plan step C8, without
                                                      # NV_GSP.init_hw; "flcn_hw", C9, without either init_hw); cmd_rm_export gave
                                                      # the C++ side the GSP

    def _check_queues_owned(self):
        # After cmd_handoff, submitting from here too would corrupt the C++ side's GPFIFO and timeline state.
        if self.handed_off:
            raise RuntimeError("the GPU queues belong to the C++ side after handoff")

    # ── wire I/O: each JSON message is preceded by its length as a 4-byte
    # little-endian uint32, so a message is two reads instead of one recv()
    # per byte (the newline framing amd_dispatch_daemon.py still uses cost
    # ~88 us per message, TODO.md "Runtime roadmap", Step 2) ────────────────
    def recv_msg(self):
        hdr = self.sock.recv(4)
        if not hdr:
            return None
        t0 = time.perf_counter() if _PROFILE else None  # after the first bytes: excludes waiting for the next command
        if len(hdr) < 4:
            hdr += self.recv_exact(4 - len(hdr))
        body = self.recv_exact(struct.unpack("<I", hdr)[0])
        if _PROFILE:
            _prof_add("wire.recv_msg", time.perf_counter() - t0)
        return body

    def recv_exact(self, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionResetError("socket closed mid-read")   # the plugin went away: Daemon.run takes the EOF path
            buf += chunk
        return bytes(buf)

    def send_json(self, obj):
        body = json.dumps(obj).encode()
        self.sock.sendall(struct.pack("<I", len(body)) + body)

    # ── commands ──────────────────────────────────────────────────────────
    def cmd_boot(self, req):
        level = req.get("level", "")   # plan step C7: "rm", the NVDev only (the C++ side builds the NVDevice); C8: "gsp_hw", without
                                       # NV_GSP.init_hw; C9: "flcn_hw", without NV_FLCN.init_hw either
        if level not in ("", "rm", "gsp_hw", "flcn_hw") or (level and self.tgpu_fd is None):
            raise RuntimeError(f"boot: level {level!r} (rm, gsp_hw or flcn_hw, with the C++ side's TinyGPU.app connection)")
        _apply_boot_safety_patches()
        if self.tgpu_fd is not None:
            _install_inherited_tinygpu(self.tgpu_fd)
        DEV.value = "NV"
        from tinygrad import Device
        try:
            if level: self.dev, self.rm_level = _RMDevice(_boot_nvdev_only(level), level), level
            else: self.dev = Device["NV:0"]
        except Exception as e:
            # nv_init_helper refuses a GPU that still carries a previous boot (WPR2 up) before writing anything;
            # tinygrad wraps the per-interface errors in an ExceptionGroup, so dig the refusal out for the reply
            warm = _find_exception(e, _warm_error_type())
            if warm is None:
                # failed after booter_load started GSP-RM: unload it, and close only once it confirms the suspend (plan step P2)
                import nv_init_helper
                fini = nv_init_helper.unload_after_failed_boot()
                # GSP-RM never started, so closing is safe; the error names each cause, which tinygrad's ExceptionGroup text
                # hides (the plugin prints it: e.g. nv_init_helper's BAR refusal or the FSP readiness timeout, plan step B1)
                if fini is None: raise RuntimeError(_boot_error_text(e)) from e
                import traceback
                traceback.print_exc(file=sys.stderr)
                reply = {**fini, "ok": False, "error": f"boot failed after GSP-RM started: {_boot_error_text(e)}"}
                if not reply.get("unload_ok") or reply.get("halted") is False:   # Blackwell: the RISC-V core never halted (plan step B1)
                    reply.update(hold=True, pid=os.getpid())
                self._reply_and_hold(reply)
                return
            log(str(warm))
            self.send_json({"ok": False, "warm": True, "error": str(warm)})
            return
        if self.rm_level:
            log(f"booted — {self.dev}: the C++ side builds the NVDevice (level {self.rm_level}" +
                {"gsp_hw": ": GSP-RM started, its init_hw and the golden image are the C++ side's)",
                 "flcn_hw": ": the images prepared, both init_hw and the golden image are the C++ side's)"}.get(self.rm_level, ")"))
            self.send_json({"ok": True, "level": self.rm_level})
            return
        log(f"booted — {self.dev}, arch={self.dev.arch}")
        log(f"launch_batch: {'one chained queue per batch' if _CHAIN_LAUNCHES else 'one queue per launch (BEAGLE_NV_CHAIN_LAUNCHES=0)'}")
        self.send_json({"ok": True, "arch": self.dev.arch})

    def cmd_compile_all(self, req):
        # Real per-kernel ELFs below come from ptxas, never tinygrad's NAK
        # (Mesa/Rust) compiler backend — NVProgram.__init__ branches on
        # isinstance(dev.renderer, NAKRenderer) and would misinterpret a
        # ptxas cubin as NAK's own machine-code format if that's ever the
        # selected renderer. STATUS.md §6 flagged this as a real open
        # question for this exact (Blackwell) chip before §74's reference
        # test empirically proved compile+dispatch works end to end here —
        # but that test went through tinygrad's own renderer selection for a
        # Python-generated Tensor op, not this daemon's ptxas-cubin
        # injection path, so it doesn't by itself prove which renderer was
        # active. Fail loudly here rather than silently mis-dispatching if
        # it ever is NAK — better to know immediately than to chase a
        # correctness bug that isn't actually in the code being tested.
        # Checked here, before the first ptxas cubin, not at boot (TODO.md
        # plan step C1): selecting the renderer builds its compiler, which on
        # macOS starts tinygrad's Docker compile server (compiler_cuda.py
        # osx_docker_cmd), and the C++ runtime never compiles.
        from tinygrad.renderer.nir import NAKRenderer
        log(f"renderer: {type(self.dev.renderer).__name__}")
        if isinstance(self.dev.renderer, NAKRenderer):
            self.send_json({"ok": False, "error":
                f"dev.renderer is NAKRenderer — this daemon injects real ptxas "
                f"cubins via NVProgram, which assumes a non-NAK ELF layout. "
                f"See nv_dispatch_daemon.py cmd_compile_all's comment."})
            return
        self.elf_bytes = nch.compile_ptx(req["ptx_path"], self.dev.arch, kernel_name="_all")
        log(f"compiled — {len(self.elf_bytes)} byte ELF")
        # Name discovery only (is_blackwell=False is fine here — it only
        # affects cbuf0 sizing in extract_all_metadata's return value, not
        # the .text.<name> section scan that finds kernel names; real
        # per-kernel metadata is recomputed by NVProgram itself, unused here).
        _, kernels = nch.extract_all_metadata(self.elf_bytes, is_blackwell=False)
        self.kernel_names = set(kernels.keys())
        log(f"kernels found: {sorted(self.kernel_names)}")
        self.send_json({"ok": True, "kernels": sorted(self.kernel_names)})

    def _get_program(self, name, n_int_args):
        if name not in self.kernel_names:
            raise RuntimeError(f"kernel {name!r} not found in compiled ELF (have: {sorted(self.kernel_names)})")
        key = (name, n_int_args)
        if key not in self.programs:
            from tinygrad.device import TinyELF, Target
            from tinygrad.dtype import dtypes
            # signature: n_int_args uint32 entries -- matches BEAGLE's
            # KernelLauncher.cpp calling convention (all trailing scalar args
            # are unsigned int), same convention amd_dispatch_daemon.py's
            # BeagleAMDProgram uses. NVArgsState (CLikeArgsState) fills bufs
            # then vals positionally and doesn't itself consult signature,
            # but TinyELF requires the field and this keeps it accurate.
            signature = tuple((None, i, dtypes.uint32, ()) for i in range(n_int_args))
            obj = TinyELF(lib=self.elf_bytes, name=name, target=Target(), signature=signature)
            self.programs[key] = BeagleNVProgram(self.dev, obj)  # see class docstring: fixes constbuf0's kernel-name filtering
        return self.programs[key]

    def cmd_handoff(self, req):
        if self.tgpu_fd is None:
            raise RuntimeError("handoff needs the C++ side's TinyGPU.app connection (second argument)")
        if self.rm_level: raise RuntimeError(f"handoff: at level {self.rm_level} the C++ side builds the NVDevice and its handoff itself (rm_export)")
        dev = self.dev
        # C++ dispatch (BEAGLE_NV_CPP_DISPATCH=1): every program is prepared
        # now, while this daemon still owns the queues (program uploads and
        # local-memory setup both submit GPU work). The C++ runtime
        # (BEAGLE_NV_USE_DAEMON=0) sends "programs": false and loads its embedded
        # cubin itself into a VRAM pool allocated here, then allocates from it too.
        programs = req.get("programs", True)
        level = req.get("level", "")   # plan step C6: "vram", the C++ side allocates its pool; "sysmem", its buffers too
        if level not in ("", "vram", "sysmem") or (level and programs):
            raise RuntimeError(f"handoff: level {level!r} (the C++ runtime's vram or sysmem, programs false)")
        progs = [self._get_program(name, 0) for name in sorted(self.kernel_names)] if programs else []
        dev.synchronize()
        from tinygrad.device import BufferSpec
        pool_size = req.get("pool_size") or dev.iface.dev_impl.vram_size // 2
        if level:   # what the C++ side allocates must be what this process would allocate: nothing cached serves it here
            keys = [(pool_size, None)] + ([(size, BufferSpec(**spec)) for _, size, spec in _HANDOFF_BUFS] if level == "sysmem" else [])
            if any(dev.allocator.cache.get(k) for k in keys):
                raise RuntimeError(f"handoff: tinygrad's allocator has a cached buffer for one the C++ side allocates at level {level}")
        self._handoff_bufs = bufs = {} if level == "sysmem" else \
            {name: dev.allocator.alloc(size, BufferSpec(**spec)) for name, size, spec in _HANDOFF_BUFS}   # C++ gets their fds in this order
        if bufs: bufs["signal"].cpu_view().view(0, 16, 'B')[:] = bytes(16)  # TinyGPU.app leaves the DMA segment list here
        fds = [dev.iface.pci_dev.sysmem_fds[b.cpu_view().addr] for b in bufs.values()]
        info, blob = build_handoff(dev, progs, bufs)
        if not programs:  # what NVProgram.__init__ and _ensure_has_local_memory read from the device, and the pool
            info.update(compute_class=dev.iface.compute_class, sass_version=dev.sass_version,
                        shared_mem_window=dev.shared_mem_window, local_mem_window=dev.local_mem_window,
                        num_gpcs=dev.num_gpcs, num_tpc_per_gpc=dev.num_tpc_per_gpc, num_sm_per_tpc=dev.num_sm_per_tpc,
                        max_warps_per_sm=dev.max_warps_per_sm,
                        elf_size=0)   # no ELF follows (plan step C1); a pre-C1 plugin reads 0 bytes and refuses them, still framed
            if not level:
                self._pool = dev.allocator.alloc(pool_size)
                info.update(pool_va=self._pool.va_addr, pool_size=self._pool.size)
        if (wpr := check_vram_below_wpr(dev.iface.dev_impl)) is not None:   # raises before anything is sent: C++ gets no fds
            log(f"WPR check: VRAM allocations end at {wpr[0]:#x} <= {_wpr_bound_name(dev.iface.dev_impl)} {wpr[1]:#x}")
        # the sizes of the BARs the C++ side writes (tinygrad's bar_info, cached since the boot mapped them): its TinyGPU.app
        # client checks every posted write against them (TODO.md plan step C3)
        for bar in sorted({info["c_ring_bar"], info["c_gpput_bar"], info["d_ring_bar"], info["d_gpput_bar"], info["db_bar"]}):
            info[f"bar{bar}_size"] = dev.iface.pci_dev.bar_info(bar)[1]
        if level:
            info.update(_mm_export(dev))
            self.mm_exported = True
        info.update(ok=True, blob_size=len(blob), nfds=len(fds))
        self.send_json(info)
        self.sock.sendall(blob)
        if fds: socket.send_fds(self.sock, [b"F"], fds)
        self.handed_off = True
        log(f"handoff: {len(progs)} programs, QMD v{info['qmd_ver']}" +
            "".join(f", {n} {b.size >> 10} KiB @ {b.va_addr:#x}" for n, b in bufs.items()) +
            ("" if programs or level else f"; VRAM pool {self._pool.size >> 20} MiB @ {self._pool.va_addr:#x}") +
            (f"; the C++ side allocates its {'pool' if level == 'vram' else 'buffers and pool'} (level {level})" if level else ""))

    def cmd_state_page(self, req):
        # C++ dispatch and the C++ runtime (plan step P3): one byte carrying the state page's fd as SCM_RIGHTS follows this
        # command (GPUInterfaceTinyGPUHybridNV.cpp nv_send_fds). Taken before any check, so the command stream stays framed
        # when the page is refused. Not a TinyGPU allocation: nothing reaches the GPU.
        import mmap
        _, fds, _, _ = socket.recv_fds(self.sock, 1, 2)   # a second fd would be refused, and still closed here
        try:
            if len(fds) != 1: raise RuntimeError(f"state_page: {len(fds)} fds received")
            state = memoryview(mmap.mmap(fds[0], _STATE_WORDS * 8, prot=mmap.PROT_READ)).cast("Q")
        finally:
            for fd in fds: os.close(fd)
        phases = {"gsp_hw": (_PHASE_DISPATCH, _PHASE_GSP_INIT), "flcn_hw": (_PHASE_DISPATCH, _PHASE_GSP_INIT, _PHASE_FLCN_INIT)}
        if not self.handed_off or state[0] not in phases.get(self.rm_level, (_PHASE_DISPATCH,)):
            raise RuntimeError(f"state_page: handed off {self.handed_off}, phase {state[0]}")
        self._state = state
        if "signal" not in getattr(self, "_handoff_bufs", {}):
            # plan step C7 (level rm and above) and level sysmem: the page precedes the C++ side's own frames (its RPCs, its allocations),
            # and its timeline, which it allocates itself, follows (cmd_timeline)
            log(f"state page mapped (phase {state[0]}, frame_in_flight {state[1]}, seq {state[3]}): " +
                ("the C++ side records the GSP's sequence number after each RPC; " if self.rm_exported else "") + "its timeline follows (cmd_timeline)")
            self.send_json({"ok": True})
            return
        # the C++ timeline (the handoff's "signal" buffer) as tinygrad's own signal; virt: no initial write, no signal pool (hcq.py:235-241)
        self._cpp_signal = ops_nv.NVSignal(base_buf=self._handoff_bufs["signal"], owner=self.dev, virt=True)
        log(f"state page mapped (phase {state[0]}, frame_in_flight {state[1]}): the C++ side records whether a frame is in flight and "
            f"the timeline value it last submitted")
        self.send_json({"ok": True})

    def _map_cpp_signal(self, fd, req):
        # the C++ side allocated its timeline itself: map it from TinyGPU.app's fd, as alloc_sysmem does
        import mmap, ctypes
        from tinygrad.runtime.support.hcq import HCQBuffer, MMIOInterface
        self._sig_map = mmap.mmap(fd, req["signal_size"])
        return HCQBuffer(req["signal_va"], req["signal_size"], owner=self.dev,
                         view=MMIOInterface(ctypes.addressof(ctypes.c_char.from_buffer(self._sig_map)), req["signal_size"], fmt='B'))

    def cmd_timeline(self, req):
        # Plan step C7 (level rm and above) and level sysmem: the C++ timeline, which the C++ side allocates after its state page
        # (at rm after it built the NVDevice). One byte carrying the timeline's TinyGPU.app fd follows this command, taken before any
        # check (as cmd_state_page).
        _, fds, _, _ = socket.recv_fds(self.sock, 1, 1)
        try:
            if len(fds) != 1 or not (self.rm_exported or self.mm_exported) or self._state is None or self._cpp_signal is not None:
                raise RuntimeError(f"timeline: {len(fds)} fds, level rm exported {self.rm_exported}, memory manager exported "
                                   f"{self.mm_exported}, state page {self._state is not None}, timeline {self._cpp_signal is not None}")
            sig_buf = self._map_cpp_signal(fds[0], req)
        finally:
            for fd in fds: os.close(fd)
        self._cpp_signal = ops_nv.NVSignal(base_buf=sig_buf, owner=self.dev, virt=True)
        log("C++ timeline mapped: " + (f"the C++ side built the NVDevice (level {self.rm_level})" if self.rm_level else
                                       "the C++ side allocated its buffers (level sysmem)"))
        self.send_json({"ok": True})

    def cmd_release(self, req):
        # Plan step C10: the plugin's crash guard (beagle-tinygpu-guard, spawned with the TinyGPU.app connection, the GSP queues, the
        # state page and the C++ timeline) takes this process's keeper role over at level flcn_hw, once the C++ side built the
        # NVDevice and its timeline. The C++ side set the keeper word before asking: a plugin that dies from then on leaves the
        # decision to the guard (_eof reads the word too), and this process exits without a word to the GPU.
        if not (self.rm_exported and self.rm_level == "flcn_hw" and self._state is not None and self._cpp_signal is not None
                and self._state[4] == _KEEPER_GUARD):
            raise RuntimeError(f"release: level {self.rm_level or 'none'}, state page {self._state is not None}, timeline "
                               f"{self._cpp_signal is not None}, keeper word {self._state[4] if self._state is not None else None} "
                               "(level flcn_hw, after cmd_timeline, the word set)")
        self.dev, self.released = None, True
        log("keeper role handed to the C++ side's guard: exiting without a word to the GPU")
        self.send_json({"ok": True, "pid": os.getpid()})

    def cmd_teardown_export(self, req):
        # TODO.md plan step C5 (level teardown): what the C++ side needs to unload the GSP and run NVIDIA's teardown itself at
        # fini, with this process sending nothing to the GPU afterwards: the GSP message queues (the fd of their TinyGPU.app
        # sysmem, in which both queues keep their write and read pointers; the offsets as init_rm_args lays them out; the
        # command queue's sequence number), libos_args_sysmem (the CPU sequencer's op 8), chip_id (FALCON_RM, written by
        # every falcon reset), and nv_init_helper's two teardown images with their execute_hs arguments; on the COT boot
        # (Blackwell, plan step B2) no images, since its teardown is the unload and the wait for the GSP's RISC-V core to halt
        # (NV_FLCN_COT.fini_hw). One byte carrying the queue fd follows the reply. Refused before the handoff (the queues are
        # this process's until then).
        if not self.handed_off: raise RuntimeError("teardown_export: before the handoff")
        info, fd, pt_size, queue_size = self._gsp_export("teardown_export")
        self.send_json(info)
        socket.send_fds(self.sock, [b"Q"], [fd])
        log(f"teardown export: GSP queues ({info['gsp_queues_size']:#x} bytes, command queue at {pt_size:#x}, status queue at "
            f"{pt_size + queue_size:#x}, seq {info['gsp_seq']}), {'teardown images' if info['teardown'] else 'no teardown images'}")

    def _gsp_export(self, what):
        # cmd_teardown_export's reply (and cmd_rm_export's first part) and the queues' fd
        import nv_init_helper
        from tinygrad.helpers import round_up
        dev_impl = self.dev.iface.dev_impl
        gsp, flcn = dev_impl.gsp, dev_impl.flcn
        queue_size = gsp.cmd_q.tx.size   # init_rm_args (ip.py:364-387): a page table, then the command and status queues
        pte_cnt = ((queue_pte_cnt := (queue_size * 2) // 0x1000)) + round_up(queue_pte_cnt * 8, 0x1000) // 0x1000
        pt_size = round_up(pte_cnt * 8, 0x1000)
        fd = self.dev.iface.pci_dev.sysmem_fds[gsp.cmd_q_view.addr - pt_size]
        teardown = nv_init_helper._TEARDOWN and not dev_impl.fmc_boot and hasattr(flcn, "beagle_sb_image_paddr")   # Ada's images only
        info = {"ok": True, "fw_name": dev_impl.fw_name, "chip_name": dev_impl.chip_name, "cot": dev_impl.fmc_boot,
                "chip_id": dev_impl.chip_id, "gsp_queues_size": pt_size + 2 * queue_size,
                "gsp_cmdq_off": pt_size, "gsp_statq_off": pt_size + queue_size, "gsp_queue_size": queue_size, "gsp_seq": gsp.cmd_q.seq,
                "libos_args_sysmem": gsp.libos_args_sysmem, "unload_level0": nv_init_helper._UNLOAD_LEVEL_0, "teardown": teardown}
        if teardown:   # FWSEC-SB runs with FWSEC-FRTS's arguments (ip.py:190-193); Booter Unload with beagle_unload_params
            d = flcn.desc_v3
            data_off, data_sz, code_off, code_sz = flcn.beagle_unload_params
            info.update(sb_paddr=flcn.beagle_sb_image_paddr, sb_imem_pa=d.IMEMPhysBase, sb_imem_va=d.IMEMVirtBase, sb_imem_sz=d.IMEMLoadSize,
                        sb_dmem_pa=d.DMEMPhysBase, sb_dmem_sz=d.DMEMLoadSize, sb_pkc_off=d.PKCDataOffset, sb_engid=d.EngineIdMask,
                        sb_ucodeid=d.UcodeId, unload_paddr=flcn.beagle_unload_image_paddr, unload_data_off=data_off, unload_data_sz=data_sz,
                        unload_code_off=code_off, unload_code_sz=code_sz)
        return info, fd, pt_size, queue_size

    def cmd_rm_export(self, req):
        # Plan step C7 (level rm): what the C++ side needs to build the NVDevice itself, after the NVDev-only boot: the GSP queues
        # and the teardown's arguments (as cmd_teardown_export), tinygrad's memory manager (_mm_export), NV_GSP's RM state as
        # init_hw and init_golden_image left it (the private root client, the handle generator's next value, the classes,
        # the runlists, the golden channel's runlist, the context buffers' descriptions), and BAR0's size (bar_info is cached:
        # this sends nothing). One byte carrying the queue fd follows the reply. From here the GSP, the memory manager and every
        # queue are the C++ side's; at fini or EOF this process takes the GSP's sequence number from the state page. At level gsp_hw
        # (plan step C8) NV_GSP.init_hw has not run here: the RM state is init_sw's (the handle generator, the classes), and the
        # C++ side's init_hw and init_golden_image make the rest.
        if not self.rm_level or self.rm_exported: raise RuntimeError("rm_export: only once, after a boot at level rm or gsp_hw")
        info, fd, pt_size, queue_size = self._gsp_export("rm_export")
        impl, pci = self.dev.iface.dev_impl, self.dev.iface.pci_dev
        gsp = impl.gsp
        info.update(_mm_export(self.dev))
        info.update(rm_level=self.rm_level, rm_next_handle=int(repr(gsp.handle_gen)[6:-1]), rm_gpfifo_class=gsp.gpfifo_class,
                    rm_compute_class=gsp.compute_class, rm_dma_class=gsp.dma_class, rm_viddec_class=gsp.viddec_class or 0,
                    rm_gb2=int(impl.chip_name.startswith("GB2")), bar0_size=pci.bar_info(0)[1])
        cot_fd = None
        if self.rm_level == "flcn_hw" and impl.fmc_boot:   # plan step B2: what NV_FLCN_COT.init_sw prepared for init_hw (the FMC boot
            fl = impl.flcn                                 # parameters' page, whose fd follows the queues', and the FMC image), the WPR meta
            cot_fd = pci.sysmem_fds[fl.fmc_boot_args_view.addr]
            info.update(cot_boot_args_sysmem=fl.fmc_boot_args_sysmem, cot_boot_args_size=fl.fmc_boot_args_view.nbytes,
                        cot_fmc_sysmem=fl.fmc_booter_bar1, cot_hash=list(fl.fmc_booter_hash), cot_sig=list(fl.fmc_booter_sig),
                        cot_pkey=list(fl.fmc_booter_pkey), wpr_meta_sysmem=gsp.wpr_meta_sysmem)
        elif self.rm_level == "flcn_hw":   # plan step C9: what NV_FLCN.init_sw prepared for init_hw (prep_ucode, prep_booter), and the
                                           # booter's mailbox argument, the WPR meta
            fl, d3 = impl.flcn, impl.flcn.desc_v3
            info.update(frts_paddr=fl.frts_image_paddr, frts_offset=fl.frts_offset, frts_imem_pa=d3.IMEMPhysBase, frts_imem_va=d3.IMEMVirtBase,
                        frts_imem_sz=d3.IMEMLoadSize, frts_dmem_pa=d3.DMEMPhysBase, frts_dmem_sz=d3.DMEMLoadSize, frts_pkc_off=d3.PKCDataOffset,
                        frts_engid=d3.EngineIdMask, frts_ucodeid=d3.UcodeId, booter_paddr=fl.booter_image_paddr,
                        booter_data_off=fl.booter_data_off, booter_data_sz=fl.booter_data_sz, booter_code_off=fl.booter_code_off,
                        booter_code_sz=fl.booter_code_sz, wpr_meta_sysmem=gsp.wpr_meta_sysmem)
        if self.rm_level == "rm":
            info.update(rm_priv_root=gsp.priv_root, rm_runlists=[x for kv in sorted(gsp.runlists.items()) for x in kv],
                        rm_chan_runlists=[x for kv in sorted(gsp.chan_runlists.items()) for x in kv],
                        rm_grctx=[x for i, b in gsp.grctx_bufs.items() for x in (i, b.size, int(b.phys), int(b.virt), int(b.local))],
                        rm_subdevice=getattr(gsp, "subdevice", 0), rm_device=getattr(gsp, "device", 0))
        self.handed_off = self.mm_exported = self.rm_exported = True
        self.send_json(info)
        socket.send_fds(self.sock, [b"Q"], [fd])
        if cot_fd is not None: socket.send_fds(self.sock, [b"B"], [cot_fd])   # the FMC boot parameters (COT, level flcn_hw)
        log(f"rm export: GSP queues (seq {info['gsp_seq']}), the memory manager, NV_GSP's RM state (next handle {info['rm_next_handle']:#x}): "
            f"the C++ side {dict(gsp_hw='boots GSP-RM (init_hw), then ', flcn_hw='runs the falcons and boots GSP-RM, then ').get(self.rm_level, '')}"
            f"builds the NVDevice")

    def cmd_alloc(self, req):
        if self.mm_exported: raise RuntimeError("alloc: the C++ side owns the memory manager since the handoff (plan step C6)")
        buf = self.dev.allocator.alloc(req["size"])
        self.send_json({"ok": True, "addr": buf.va_addr})
        self._allocs[buf.va_addr] = buf  # keep alive, prevent GC/free

    def cmd_h2d(self, req):
        self._check_queues_owned()
        n = req["size"]
        with _Profiled("h2d.recv_payload"):
            data = self.recv_exact(n)
        buf = HCQBuffer(req["addr"], n)
        with _Profiled("h2d._copyin"):
            self.dev.allocator._copyin(buf, memoryview(bytearray(data)))
        self.send_json({"ok": True})

    def cmd_d2h(self, req):
        self._check_queues_owned()
        n = req["size"]
        buf = HCQBuffer(req["addr"], n)
        out = memoryview(bytearray(n))
        with _Profiled("d2h._copyout"):
            self.dev.allocator._copyout(out, buf)
        self.send_json({"ok": True, "size": n})
        self.sock.sendall(bytes(out))

    def cmd_launch_batch(self, req):
        # Batched from the start (STATUS.md AMD §26 already established this
        # is worth doing before ever measuring the NV-specific overhead
        # separately). Same wait=False + flush-before-h2d/d2h/sync/fini
        # ordering guarantee as amd_dispatch_daemon.py -- see that file's
        # module docstring for why that's sufficient without extra
        # synchronization (identical reasoning applies: NVProgram.__call__
        # here always uses wait=False, and tinygrad's own
        # _copyin/_copyout/synchronize already call self.dev.synchronize()
        # internally before touching memory).
        #
        # By default the whole batch is one compute queue: one timeline wait
        # and shader-cache invalidate, then each kernel's exec, which chains
        # its QMD onto the previous one (NVComputeQueue.exec's dependent_qmd0;
        # hcq1's HCQGraph relies on the same in-queue ordering between
        # dependent kernels), then one signal and one submit. That is
        # HCQProgram.__call__ with N execs instead of one, and one GPFIFO
        # entry and doorbell per batch instead of per launch (TODO.md
        # "Runtime roadmap", Step 2). BEAGLE_NV_CHAIN_LAUNCHES=0 submits each
        # launch on its own queue, as before.
        self._check_queues_owned()
        launches = req["launches"]
        q = None
        if _CHAIN_LAUNCHES and launches:
            q = self.dev.hw_compute_queue_t().wait(self.dev.timeline_signal, self.dev.timeline_value - 1).memory_barrier()
        for i, item in enumerate(launches):
            kernel_name = item["kernel"]
            ptrs = item["ptrs"]
            ints = item["ints"]
            grid = tuple(item["grid"])
            block = tuple(item["block"])
            try:
                with _Profiled("launch.get_program"):
                    prg = self._get_program(kernel_name, len(ints))
                bufs = tuple(HCQBuffer(addr, 0) for addr in ptrs)
                if q is None:
                    with _Profiled("launch.prg"):
                        prg(*bufs, global_size=grid, local_size=block, vals=tuple(ints), wait=False)
                else:
                    with _Profiled("launch.exec"):
                        prg.check_launch(grid, block)
                        prg.set_launch_dims(grid, block)
                        q.exec(prg, prg.fill_kernargs(bufs, tuple(ints)), grid, block)
            except Exception as e:
                self.send_json({"ok": False, "error": f"launch_batch[{i}] {kernel_name}: {e}"})
                return
        if q is not None:
            with _Profiled("launch.submit"):
                q.signal(self.dev.timeline_signal, self.dev.next_timeline()).submit(self.dev)
        self.send_json({"ok": True, "count": len(launches)})

    def cmd_sync(self, req):
        self._check_queues_owned()
        self.dev.synchronize()
        self.send_json({"ok": True})

    def cmd_fini(self, req):
        # Tear the GPU down now, while the plugin waits, instead of at interpreter exit: tinygrad's own chain
        # (HCQCompiled.finalize: synchronize, then PCIIface.device_fini -> NVDev.fini -> the GSP unload RPC, which
        # nv_init_helper follows with a wait for the GSP to report itself suspended). If the GPU does not confirm
        # the unload, closing the TinyGPU.app connection could unmap memory the GSP still uses (a DART fault), so
        # this process keeps its dup of the connection open and waits to be killed after the eGPU is unplugged.
        if _PROFILE:
            _prof_report()
        if req.get("cpp_teardown"):
            self._reply_and_hold(self._cpp_fini(req))
            return
        self._reply_and_hold(self._fini_or_hold(req.get("hung", False)))

    def _cpp_fini(self, req):
        # plan step C5: the C++ side unloaded the GSP and ran NVIDIA's teardown itself (cmd_teardown_export), and reports what
        # it saw. Nothing more goes to the GPU from here: no synchronize, no NVDev.fini; 'NV' leaves tinygrad's atexit list. If
        # the GSP did not confirm its unload (or the C++ side could not tell), this process keeps its copy of the TinyGPU.app
        # connection open (hold rule), as _fini does.
        from tinygrad import Device
        diag = req.get("diag", {})
        log(f"C++ GPU teardown: {json.dumps(diag)}")
        for name in [n for n in Device._opened_devices if n.split(":")[0] == "NV"]:
            Device._opened_devices.discard(name)   # atexit must not run NVDev.fini: the GPU is torn down
        reply = {"ok": True, **diag}
        if req.get("hold") or not diag.get("unload_ok") or diag.get("halted") is False:   # COT: the RISC-V core did not halt
            reply.update(hold=True, pid=os.getpid())
        self.dev = None   # nothing left to tear down: a later EOF exits
        return reply

    def _fini(self, hung):
        # cmd_fini's decision, shared with the EOF path (_eof, plan step P3); returns the reply. After a handoff the daemon's
        # own synchronize does not cover the C++ side's work (inv:transport-teardown#7), so its timeline is waited for first.
        reply = {"ok": True}
        if self.dev is not None:
            if self.rm_exported and self._state is None:   # plan step C7: the C++ side had the GSP and left no sequence number
                log("level rm: the C++ side took the GSP over and never sent its state page: sending nothing to the GPU")
                return {"ok": False, "error": "level rm without a state page", "hold": True, "pid": os.getpid()}
            if self._state is not None:
                phase, in_flight, last, seq = self._state[:4]
                log(f"C++ state page: phase {phase}, frame_in_flight {in_flight}, last_submitted {last}, seq {seq}, "
                    f"C++ timeline signal {self._cpp_signal.value if self._cpp_signal else 'not sent'}")
                if phase == _PHASE_FLCN_INIT:   # plan step C9: before booter_load; FWSEC-FRTS runs from VRAM, nothing from sysmem
                    log("the C++ side's falcon boot stopped before booter_load started GSP-RM: nothing to unload, closing is safe "
                        "(if FWSEC-FRTS raised WPR2, the next boot needs a power cycle)")
                    self.dev = None
                    return {"ok": False, "error": "the C++ side's falcon boot stopped before GSP-RM started: nothing to unload"}
                if phase == _PHASE_GSP_INIT:   # plan step C8: the C++ side was booting GSP-RM (its CPU sequencer, before INIT_DONE)
                    log("the C++ side did not finish booting GSP-RM (init_hw, before GSP_INIT_DONE): sending nothing to the GPU")
                    return {"ok": False, "error": "the C++ side did not finish booting GSP-RM", "hold": True, "pid": os.getpid()}
                if phase == _PHASE_TEARDOWN:   # plan step C5: the C++ side was unloading the GPU itself, and did not finish
                    log("the C++ side's own GPU teardown did not finish: sending nothing to the GPU")
                    return {"ok": False, "error": "the C++ GPU teardown did not finish", "hold": True, "pid": os.getpid()}
                if phase != _PHASE_DISPATCH or in_flight:   # TinyGPU.app would read our next bytes as the rest of a cut C++ frame
                    log("a C++ frame may be cut mid-send: sending nothing more to the GPU")
                    return {"ok": False, "error": "a C++ frame may be cut mid-send", "hold": True, "pid": os.getpid()}
                if not hung and self._cpp_signal is not None:   # at level rm none before cmd_timeline: nothing was submitted on it
                    try: self._cpp_signal.wait(last)   # as HCQCompiled.synchronize waits: 30 s without progress, GSP faults raise
                    except Exception as e:
                        log(f"C++ timeline stuck at {self._cpp_signal.value} < {last} ({e}): the hung path")
                        hung = True
                if self.rm_exported:   # plan step C7: the C++ side's RPCs advanced the command queue; the unload continues from there
                    gsp = self.dev.iface.dev_impl.gsp
                    gsp.cmd_q.seq = self._state[3]
                    if self.rm_level in ("gsp_hw", "flcn_hw") and getattr(gsp, "stat_q", None) is None:   # init_hw's first two statements (ip.py:511-512)
                        gsp.stat_q = ip_nv.NVRpcQueue(gsp, gsp.stat_q_view, gsp.cmd_q_view)
                        gsp.cmd_q.rx_view = gsp.stat_q_view.view(gsp.stat_q.tx.rxHdrOff, fmt='I')
            from tinygrad import Device
            try:
                # a timeline timeout (the plugin's, the C++ timeline's above, or the daemon's own: tinygrad's error_state,
                # also when first seen here, since HCQCompiled.finalize would swallow it and run the falcon teardown anyway):
                # the unload RPC only, no synchronize, no falcon step (hung rule)
                if not hung and getattr(self.dev, "error_state", None) is None:
                    try: self.dev.synchronize()   # after a handoff the daemon's own timeline is idle, so this returns at once
                    except Exception as e:
                        log(f"the daemon's own timeline: {type(e).__name__}: {e}: the hung path")
                        hung = True
                if hung or getattr(self.dev, "error_state", None) is not None:
                    self.dev.iface.dev_impl.gsp.fini_hw()
                    reply["hung"] = True
                else:
                    self.dev.finalize()   # unless BEAGLE_NV_TEARDOWN=0 this also runs NVIDIA's teardown (nv_init_helper, plan steps P2, P3)
                reply.update(getattr(self.dev.iface.dev_impl, "beagle_fini", {}))
            except Exception as e:
                import traceback
                traceback.print_exc(file=sys.stderr)
                # what nv_init_helper recorded decides: a confirmed GSP suspend makes closing safe even if a later step failed
                reply.update(getattr(self.dev.iface.dev_impl, "beagle_fini", {"unload_ok": False}))
                reply.update(ok=False, error=f"GPU teardown failed: {e}")
            finally:
                for name in [n for n in Device._opened_devices if n.split(":")[0] == "NV"]:
                    Device._opened_devices.discard(name)   # atexit must not run NVDev.fini a second time
            # after a hang, hold even with the unload confirmed (plan step D1's review): a channel stuck on a semaphore acquire
            # may still be polling the sysmem timeline page, which closing the connection would unmap (unplug first, then kill)
            # on Blackwell also until the GSP's RISC-V core halted (nv_init_helper section 6; plan step B1): before that the FMC
            # and the ACR may still use the boot structures in sysmem
            if not reply.get("unload_ok") or reply.get("hung") or reply.get("halted") is False:
                reply.update(hold=True, pid=os.getpid())
            else:
                self.dev = None   # torn down: a later EOF (the plugin gone before reading this reply) must not tear down again
        return reply

    def _reply_and_hold(self, reply):
        # the reply first, so the plugin can print the unplug message; the hold even if the plugin is already gone (plan step P3)
        try:
            self.send_json(reply)
        finally:
            if reply.get("hold"):
                self._hold()

    def _eof(self):
        # The plugin went away without "fini" (killed, crashed, or cut off mid-message). Until plan step P3 the interpreter
        # then exited and tinygrad's atexit finalized with no hold decision and without waiting for the C++ side's work;
        # now it decides as fini does, and never closes the last TinyGPU.app fd while the GSP may be live (hold rule).
        if self._state is not None and self._state[4] == _KEEPER_GUARD:   # plan step C10: the C++ side handed the role over
            log("command socket closed; the keeper word names the C++ side's guard, which decides: exiting without a word to the GPU")
            return
        if self.dev is None:   # no boot, a refused or failed one (cmd_boot decided), or fini already tore it down
            log("command socket closed; no device to tear down, exiting")
            return
        log("command socket closed without fini; tearing the GPU down as fini would")
        reply = self._fini_or_hold(False)
        log(f"GPU teardown at EOF: {json.dumps(reply)}")
        if reply.get("hold"):
            self._hold()

    def _hold(self):
        log(f"HOLDING the TinyGPU.app connection: closing it could unmap memory the GPU may still use. Unplug the eGPU first, "
            f"then kill {os.getpid()}. (SIGINT and SIGHUP are ignored.)")
        while True:
            time.sleep(3600)

    def _fini_or_hold(self, hung):
        # _fini's decision; an exception from it (outside the teardown it guards itself) holds: unknown is never closed
        try: return self._fini(hung)
        except Exception as e:
            try:
                import traceback
                traceback.print_exc(file=sys.stderr)
            except Exception: pass
            return {"ok": False, "error": f"the fini decision failed: {type(e).__name__}: {e}", "hold": True, "pid": os.getpid()}

    def run(self):
        try:
            while True:
                msg = self.recv_msg()
                if msg is None:
                    break
                with _Profiled("wire.json_loads"):
                    req = json.loads(msg)
                cmd = req.get("cmd")
                try:
                    with _Profiled(f"cmd.{cmd}"):
                        getattr(self, f"cmd_{cmd}")(req)
                except Exception as e:
                    try:
                        import traceback
                        traceback.print_exc(file=sys.stderr)
                    except Exception: pass
                    self.send_json({"ok": False, "error": str(e)})
                if cmd == "fini" or self.released:
                    return
        except ConnectionError as e:   # the plugin went away mid-message, or before reading a reply
            log(f"command socket: {type(e).__name__}: {e}")
        self._eof()


def _warm_error_type():
    import nv_init_helper
    return nv_init_helper.WarmGPUError


def _boot_error_text(e):
    """e's text, followed for (nested) exception groups by each leaf exception's type and message."""
    leaves = []
    def walk(x):
        if getattr(x, "exceptions", None): [walk(s) for s in x.exceptions]
        else: leaves.append(f"{type(x).__name__}: {x}")
    walk(e)
    return str(e) if leaves == [f"{type(e).__name__}: {e}"] else f"{e}: " + "; ".join(leaves)


def _find_exception(e, typ):
    """e itself or the first exception of type typ inside (nested) exception groups."""
    if isinstance(e, typ):
        return e
    for sub in getattr(e, "exceptions", ()):
        if (found := _find_exception(sub, typ)) is not None:
            return found
    return None


def main():
    # A terminal Ctrl-C or a hangup must not kill the daemon while it may hold a live GPU (TODO.md plan step P1);
    # it exits after "fini", or, if the plugin goes away without it, once Daemon._eof has decided as fini would (plan step P3).
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <cmd_sock_fd> [<tinygpu_sock_fd>]", file=sys.stderr)
        sys.exit(1)
    cmd_fd = int(sys.argv[1])
    tgpu_fd = int(sys.argv[2]) if len(sys.argv) > 2 else None
    sock = socket.socket(fileno=cmd_fd)

    os.makedirs(os.path.expanduser("~/Library/Logs"), exist_ok=True)
    log_path = os.path.expanduser("~/Library/Logs/nv_dispatch_daemon.log")
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_SYNC, 0o644)
    sys.stderr = os.fdopen(fd, 'w', buffering=1)
    log(f"starting, cmd_fd={cmd_fd}" + (f", tinygpu_fd={tgpu_fd} (C++ dispatch)" if tgpu_fd is not None else ""))

    daemon = Daemon(sock, tgpu_fd)
    try:
        daemon.run()
    except Exception:
        try:
            import traceback
            traceback.print_exc(file=sys.stderr)
        except Exception: pass   # the hold below never depends on the log
        if daemon.dev is not None:   # tinygrad's atexit would finalize with no hold decision and no wait for the C++ side's work
            daemon._hold()
        sys.exit(1)
    log("exiting cleanly")


if __name__ == "__main__":
    main()
