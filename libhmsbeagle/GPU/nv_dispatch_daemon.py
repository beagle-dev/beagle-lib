#!/usr/bin/env python3
"""
nv_dispatch_daemon.py — BEAGLE NV hybrid backend, daemon architecture
(STATUS.md §73/§75).

Three individually-clean SASS/PTX-level probes (STATUS.md §65-72) failed to
reproduce the wrong-answer bug in GPUInterfaceTinyGPUHybrid.cpp's hand-rolled
GPFIFO/QMD dispatch — "the defect must depend on something these probes
structurally cannot reproduce." STATUS.md §74's `nv_reference_test.py`
confirmed (hardware-verified PASS) that tinygrad's own real NVDevice/
NVProgram/HCQProgram.__call__ stack works correctly end to end on this exact
hardware/transport. This is the same architecture class that resolved the
AMD hybrid backend's own crash-class bugs (amd_dispatch_daemon.py) — real
GPU operations (boot, compile, alloc, memcpy, launch, sync) move entirely
into a resident Python daemon driving tinygrad's proven code, and
GPUInterfaceTinyGPUHybridNV.cpp becomes a thin RPC client.

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

Protocol: identical wire format to amd_dispatch_daemon.py — newline-
terminated JSON command lines on a dedicated socketpair (not the TinyGPU
socket: NVDevice("NV:0") makes its own connection internally, exactly like
nv_reference_test.py's hardware-verified boot). Commands carrying bulk data
(h2d/d2h) are followed immediately by that many raw bytes on the same
stream. Kernel launches are batched from the start this time (cmd_launch_batch
only, no per-launch cmd_launch) — AMD's own profiling (STATUS.md AMD §26)
already found steady-state per-launch RPC overhead comparable to or larger
than the GPU dispatch work itself, no need to re-discover that here.

    python3 nv_dispatch_daemon.py <cmd_sock_fd>
"""
import sys, os, json, struct, pathlib

_TINYGRAD_PATH = os.environ.get("TINYGRAD_PATH", str(pathlib.Path.home() / "Dropbox/Projects/tinygrad"))
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

        # DIAGNOSTIC (TODO.md "PICK UP HERE" -> NV Phase 50, user-directed
        # driver/queue-layer instrumentation, read-only half): log the QMD's
        # own occupancy/CTA-scheduling-adjacent fields exactly as
        # constructed, before this program ever launches anything. Purely
        # observational -- .read() only, no .write() -- so this can never
        # change real behavior; the point is to see whether fields like
        # free_cta_slots_empty_sm (plausibly governs whether a freed CTA
        # slot on an SM gets refilled once part of a grid's already
        # running) are left at their zero/unset default (upstream
        # NVProgram.__init__ never sets any of them) versus something
        # unexpected. Wrapped in try/except: these field names are shared
        # across QMD versions in nv_gpu's own autogen tables, but this must
        # never be allowed to break a real launch if that ever isn't true.
        try:
            occ_fields = ['free_cta_slots_empty_sm', 'occupancy_max_register', 'occupancy_max_shared_mem',
                          'occupancy_max_warp', 'pre_exit_at_last_cta_launch', 'enable_program_pre_exit', 'cta_launch_queue']
            occ_vals = {f: self.qmd.read(f) for f in occ_fields}
            log(f"QMD-occupancy-fields[{self.name}]: {occ_vals}")
        except Exception as e:
            log(f"QMD-occupancy-fields[{self.name}]: read failed: {e}")

        super(ops_nv.NVProgram, self).__init__(ops_nv.NVArgsState, self.dev, obj, kernargs_alloc_size=round_up(self.constbufs[0][1], 1 << 8) + (8 << 8))
        _weakref.finalize(self, self._fini, self.dev, self.lib_gpu, buf_spec)


class BeagleNVComputeQueue(ops_nv.NVComputeQueue):
    """
    DIAGNOSTIC (TODO.md "PICK UP HERE" -> NV Phase 49, candidate (1)):
    real, unmodified NVComputeQueue.exec, with one addition -- after the
    real code encodes global_size/local_size into the just-submitted,
    per-launch QMD buffer (ops_nv.py:135-157), read those exact fields
    straight back out of that same buffer and log them. Confirms the real,
    about-to-reach-hardware bytes for kernelMatrixMulADB's own launch
    specifically carry grid=(16,1,1) -- not just that 16 was passed into
    prg(...) (already logged elsewhere, cmd_launch_batch below) or that a
    trivial probe kernel's dispatch of this same shape works (Phase 46,
    nv_grid_test.py) -- closing the one remaining link in that chain this
    investigation hadn't directly observed for the real, heavier kernel.
    Installed via a plain instance-attribute monkeypatch on `dev` in
    cmd_boot (`dev.hw_compute_queue_t = BeagleNVComputeQueue`) -- HCQProgram
    .__call__ reads that attribute fresh on every launch (hcq.py:373), so
    this takes effect for every kernel, not just kernelMatrixMulADB; logging
    is unconditional (cheap, one line per launch) rather than name-gated, so
    a hardware run can also see whether *other* kernels' encoded grids ever
    diverge from what was requested -- new information either way.
    """
    def exec(self, prg, args_state, global_size, local_size):
        super().exec(prg, args_state, global_size, local_size)
        q = self.active_qmd
        gw = q.read('grid_width' if q.ver >= 4 else 'cta_raster_width')
        gh = q.read('grid_height' if q.ver >= 4 else 'cta_raster_height')
        gd = q.read('grid_depth' if q.ver >= 4 else 'cta_raster_depth')
        log(f"QMD[{prg.name}]: encoded grid=({gw},{gh},{gd}) (requested global_size={tuple(global_size)})")
        return self


def log(msg):
    print(f"[nv_dispatch_daemon] {msg}", file=sys.stderr, flush=True)


def _apply_boot_safety_patches():
    """
    Same two fixes nv_reference_test.py already verified on hardware
    (STATUS.md §74, PASS) — see that file's header for the full rationale of
    each. Duplicated here (not imported from there) because
    nv_reference_test.py is a standalone diagnostic script, not a shared
    module other scripts import from; both call sites are short and this
    project's convention (amd_2d_test.py, amd_math_test.py, ...) is for each
    standalone script to be self-contained.
    """
    import nv_init_helper  # noqa: F401 — GSP/RM boot patches (WPR2-reset-loop
    # suppression, palloc zero-size limit, GC6 BSI sleep) applied as a module-
    # level side effect; main() is never called.
    from tinygrad.runtime.support.system import APLRemotePCIDevice
    def _safe_reset(self):
        log("PCIe FLR suppressed (macOS eGPU safety) — see nv_init_helper.py InheritedFDPCIDevice.reset()")
    APLRemotePCIDevice.reset = _safe_reset


class Daemon:
    def __init__(self, sock):
        self.sock = sock
        self.dev = None
        self.elf_bytes = None     # last-compiled multi-kernel ELF (real ptxas cubin)
        self.kernel_names = set() # names found in elf_bytes, for GetFunction-style validation
        self.programs = {}        # (name, n_int_args) -> NVProgram
        self._allocs = {}

    # ── wire I/O (identical to amd_dispatch_daemon.py) ──────────────────────
    def recv_line(self):
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = self.sock.recv(1)
            if not chunk:
                return None
            buf += chunk
        return buf.decode()

    def recv_exact(self, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise RuntimeError("socket closed mid-read")
            buf += chunk
        return bytes(buf)

    def send_json(self, obj):
        self.sock.sendall((json.dumps(obj) + "\n").encode())

    # ── commands ──────────────────────────────────────────────────────────
    def cmd_boot(self, req):
        _apply_boot_safety_patches()
        DEV.value = "NV"
        from tinygrad import Device
        self.dev = Device["NV:0"]
        log(f"booted — {self.dev}, arch={self.dev.arch}")
        # DIAGNOSTIC (TODO.md "PICK UP HERE" -> NV Phase 51): read-only,
        # interprets Phase 50's SMID=0-for-every-running-block finding.
        # num_gpcs/num_tpc_per_gpc/num_sm_per_tpc are real values NVDevice
        # itself already queries live from this exact chip at boot
        # (ops_nv.py:628) -- their product is the chip's own reported total
        # SM count. If that's well above 1 (expected for any real discrete
        # Blackwell GPU) but every observed CTA still lands on SM 0, that
        # argues for a CTA-routing/enablement gap specific to this
        # from-scratch driver stack, not "this GPU genuinely has ~1 SM".
        log(f"GPU topology: num_gpcs={self.dev.num_gpcs} num_tpc_per_gpc={self.dev.num_tpc_per_gpc} "
            f"num_sm_per_tpc={self.dev.num_sm_per_tpc} max_warps_per_sm={self.dev.max_warps_per_sm} "
            f"total_sms={self.dev.num_gpcs * self.dev.num_tpc_per_gpc * self.dev.num_sm_per_tpc}")
        # DIAGNOSTIC (see BeagleNVComputeQueue's own docstring above):
        # HCQProgram.__call__ reads dev.hw_compute_queue_t fresh on every
        # launch (hcq.py:373), so this plain instance-attribute monkeypatch
        # is enough -- no need to touch NVDevice's own construction.
        self.dev.hw_compute_queue_t = BeagleNVComputeQueue

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
        from tinygrad.renderer.nir import NAKRenderer
        log(f"renderer: {type(self.dev.renderer).__name__}")
        if isinstance(self.dev.renderer, NAKRenderer):
            self.send_json({"ok": False, "error":
                f"dev.renderer is NAKRenderer — this daemon injects real ptxas "
                f"cubins via NVProgram, which assumes a non-NAK ELF layout. "
                f"See nv_dispatch_daemon.py cmd_boot's comment."})
            return
        self.send_json({"ok": True, "arch": self.dev.arch})

    def cmd_compile_all(self, req):
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
            # DIAGNOSTIC (TODO.md Phase 48/STATUS.md §85): confirm what the
            # regs_usage/lcmem_usage fix above actually resolves to for this
            # kernel, on this run -- print unconditionally (cheap, one line
            # per distinct kernel/n_int_args the whole run ever launches) so
            # a hardware run's own output settles "did the fix take effect"
            # and "what did it resolve to" without guessing.
            p = self.programs[key]
            log(f"BeagleNVProgram[{name}]: regs_usage={p.regs_usage} shmem_usage={p.shmem_usage} lcmem_usage={p.lcmem_usage}")
        return self.programs[key]

    def cmd_alloc(self, req):
        buf = self.dev.allocator.alloc(req["size"])
        self.send_json({"ok": True, "addr": buf.va_addr})
        self._allocs[buf.va_addr] = buf  # keep alive, prevent GC/free

    def cmd_h2d(self, req):
        n = req["size"]
        data = self.recv_exact(n)
        # Diagnostic (STATUS.md §80/TODO.md Phase 44 follow-up): log every
        # h2d target address/size, and for small buffers (<=256 bytes --
        # covers e.g. kernelMatrixMulADB's listC/distanceQueue, not the much
        # larger partials/matrix buffers) the actual uploaded uint32 values
        # too. Lets the next hardware run show directly whether a small
        # metadata buffer like listC really got its full, correct content
        # uploaded, without guessing from kernel behavior alone.
        if n <= 256 and n % 4 == 0:
            import struct as _struct
            vals = _struct.unpack(f"<{n // 4}I", data)
            log(f"h2d addr={req['addr']:#x} size={n} values={vals}")
        else:
            log(f"h2d addr={req['addr']:#x} size={n}")
        buf = HCQBuffer(req["addr"], n)
        self.dev.allocator._copyin(buf, memoryview(bytearray(data)))
        self.send_json({"ok": True})

    def cmd_d2h(self, req):
        n = req["size"]
        buf = HCQBuffer(req["addr"], n)
        out = memoryview(bytearray(n))
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
        launches = req["launches"]
        for i, item in enumerate(launches):
            kernel_name = item["kernel"]
            ptrs = item["ptrs"]
            ints = item["ints"]
            grid = item["grid"]
            block = item["block"]
            # Diagnostic (STATUS.md §79/TODO.md Phase 43): log exactly what
            # reached prg(...) after the JSON round-trip, so a C++-side log
            # line and this one can be compared directly to rule out (or
            # confirm) a serialization-layer mismatch, not just assume the
            # C++ send is the ground truth.
            log(f"launch {kernel_name} global_size={tuple(grid)} local_size={tuple(block)} vals={tuple(ints)}")
            try:
                prg = self._get_program(kernel_name, len(ints))
                bufs = tuple(HCQBuffer(addr, 0) for addr in ptrs)
                prg(*bufs, global_size=tuple(grid), local_size=tuple(block), vals=tuple(ints), wait=False)
            except Exception as e:
                self.send_json({"ok": False, "error": f"launch_batch[{i}] {kernel_name}: {e}"})
                return
            # Opt-in diagnostic (TODO.md "PICK UP HERE" -> NV Phase 70):
            # Phase 69 already ruled out multi-kernel wait=False batching
            # as sufficient on its own (a from-scratch probe reproducing
            # this exact 5-kernel queuing pattern stayed clean) -- this
            # tests the same question from the *other* direction, inside
            # the real pipeline itself: does forcing kernelMatrixMulADB to
            # finish (a real dev.synchronize()) before anything else is
            # queued behind it change the real run's own result? Opt-in,
            # env-gated, reversible; unconditional wait=False (the
            # established default) whenever the env var is unset.
            if kernel_name == "kernelMatrixMulADB" and os.environ.get("BEAGLE_NV_SYNC_AFTER_MATMUL"):
                log("BEAGLE_NV_SYNC_AFTER_MATMUL set -- synchronizing immediately after kernelMatrixMulADB")
                self.dev.synchronize()
        self.send_json({"ok": True, "count": len(launches)})

    def cmd_sync(self, req):
        self.dev.synchronize()
        self.send_json({"ok": True})

    def cmd_fini(self, req):
        self.send_json({"ok": True})

    def run(self):
        while True:
            line = self.recv_line()
            if line is None:
                break
            req = json.loads(line)
            cmd = req.get("cmd")
            try:
                getattr(self, f"cmd_{cmd}")(req)
            except Exception as e:
                import traceback
                traceback.print_exc(file=sys.stderr)
                self.send_json({"ok": False, "error": str(e)})
            if cmd == "fini":
                break


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <cmd_sock_fd>", file=sys.stderr)
        sys.exit(1)
    import socket
    cmd_fd = int(sys.argv[1])
    sock = socket.socket(fileno=cmd_fd)

    os.makedirs(os.path.expanduser("~/Library/Logs"), exist_ok=True)
    log_path = os.path.expanduser("~/Library/Logs/nv_dispatch_daemon.log")
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_SYNC, 0o644)
    sys.stderr = os.fdopen(fd, 'w', buffering=1)
    log(f"starting, cmd_fd={cmd_fd}")

    try:
        Daemon(sock).run()
    except Exception:
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    log("exiting cleanly")


if __name__ == "__main__":
    main()
