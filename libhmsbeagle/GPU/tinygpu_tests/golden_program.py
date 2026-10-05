"""Golden test for TinyGPUHybridNVProgram.h: the real BeagleNVProgram (nv_dispatch_daemon.py: tinygrad's
NVProgram.__init__ plus BEAGLE's multi-kernel fixes) and the C++ port load the same real cubins (ptxas: the 9 SP
modules for sm_86, sm_89 and sm_120, the single-precision 27 of the 54 the plugin embeds). Their per-kernel records,
in build_handoff's format (QMD template, cbuf0 prefix, kernargs layout), and the relocated image must be
byte-identical. Both use slm_per_thread = the running max over all kernels."""
import os, sys, types, itertools, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tgpaths
tgpaths.setup()
import nv_dispatch_daemon as d
import nv_compile_helper as nch
from tinygrad.device import TinyELF, Target
from tinygrad.helpers import round_up

HERE, WORK = str(tgpaths.HERE), str(tgpaths.WORK)
ops_nv, nv_gpu, LIB_VA = d.ops_nv, d.ops_nv.nv_gpu, 0x10_5000_0000

class FakeAllocator:  # the program image goes to LIB_VA; _copyin keeps the relocated bytes
    def __init__(self): self.images = []
    def alloc(self, size, spec=None): return types.SimpleNamespace(va_addr=LIB_VA, size=size)
    def _copyin(self, buf, mv): self.images.append(bytes(mv))
    def free(self, *a): pass

def fake_dev(compute_class, sass, slm):
    fifo = lambda: types.SimpleNamespace(ring=types.SimpleNamespace(residx=1, off=0), entries_count=1, token=0,
                                         gpput=types.SimpleNamespace(residx=1, off=0), put_value=0)
    dev = types.SimpleNamespace(iface=types.SimpleNamespace(compute_class=compute_class), renderer=object(), allocator=FakeAllocator(),
                                slm_per_thread=slm, shared_mem_window=0x729400000000, local_mem_window=0x729300000000,
                                sass_version=sass, prof_prg_counter=itertools.count(), synchronize=lambda: None,
                                compute_gpfifo=fifo(), dma_gpfifo=fifo(), gpu_mmio=types.SimpleNamespace(residx=0, off=0))
    def ensure(required): dev.slm_per_thread = max(dev.slm_per_thread, round_up(required, 32))
    dev._ensure_has_local_memory = ensure
    return dev

def program(dev, elf, name): return d.BeagleNVProgram(dev, TinyELF(lib=elf, name=name, target=Target(), signature=()))

tgpaths.build_cpp(f"{HERE}/golden_program.cpp", f"{WORK}/golden_program")
ok_all = True
for variant in tgpaths.VARIANTS:
    for arch, cc, sass in (("sm_86", nv_gpu.AMPERE_COMPUTE_B, 0x86), ("sm_89", nv_gpu.ADA_COMPUTE_A, 0x89),
                           ("sm_120", nv_gpu.BLACKWELL_COMPUTE_B, 0xa4)):
        elf = tgpaths.cubin(variant, arch)
        cubin = str(tgpaths.cubin_path(tgpaths.ptx(variant), arch))
        names = sorted(nch.extract_all_metadata(elf, is_blackwell=cc >= nv_gpu.BLACKWELL_COMPUTE_A)[1].keys())
        slm = max(round_up(program(fake_dev(cc, sass, 0), elf, n).lcmem_usage, 32) for n in names)
        dev = fake_dev(cc, sass, slm)
        progs = [program(dev, elf, n) for n in names]
        _, ref_blob = d.build_handoff(dev, progs, {})
        ref_images = set(dev.allocator.images)

        open(f"{WORK}/golden_names.txt", "w").write("\n".join(names) + "\n")
        subprocess.run([f"{WORK}/golden_program", cubin, f"{WORK}/golden_names.txt", str(cc), str(LIB_VA), str(sass), WORK], check=True)
        out_blob, out_image = open(f"{WORK}/golden_out_blob.bin", "rb").read(), open(f"{WORK}/golden_out_image.bin", "rb").read()
        out_slm = int(open(f"{WORK}/golden_out_slm.txt").read())

        ok = out_blob == ref_blob and ref_images == {out_image} and out_slm == slm
        if not ok:
            print(f"  slm python={slm:#x} c++={out_slm:#x}; images: {len(ref_images)} distinct in python, equal={ref_images == {out_image}}")
            if out_blob != ref_blob:
                i = next(i for i in range(min(len(out_blob), len(ref_blob))) if out_blob[i] != ref_blob[i]) \
                    if len(out_blob) == len(ref_blob) else "length"
                print(f"  blob differs (len python {len(ref_blob)}, c++ {len(out_blob)}), first difference at byte {i}")
        print(f"{variant} {arch}: {len(names)} kernels, slm_per_thread {slm:#x}, image {len(out_image)} bytes, "
              f"{'IDENTICAL' if ok else 'MISMATCH'}")
        ok_all &= ok
sys.exit(0 if ok_all else 1)
