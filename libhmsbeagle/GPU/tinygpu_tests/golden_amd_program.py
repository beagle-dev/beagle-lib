"""Golden test for TinyGPUAMDProgram.h (TODO.md plan step A1d). Every BEAGLE kernel variant (SP and DP, 9 padded state
counts) is compiled offline exactly as the AMD daemon compiles it (amd_compile_helper.compile_hip, tinygrad's comgr
compile_hip), then each kernel is loaded by amd_dispatch_daemon.BeagleAMDProgram on a stub device (its image upload
captured, at a fixed base) and by golden_amd_program.cpp's C++ loader. The relocated image and each kernel's record
(rsrc1/2/3, prog_addr, aql_prog_addr, segment and kernarg sizes, wave32) must be identical, and the C++ scratch sizing
must equal AMDDevice._ensure_has_local_memory for a set of private sizes and each variant's largest. For each card of
am::kChips (TODO.md plan step N13: the RX 7900 XT, gfx1100, and the RX 9070 XT, gfx1201): compiled for its arch, with its
props from its captured discovery table as PCIIface._compute_props derives them (fake_am_gpu.props_for)."""
import os, sys, types, itertools, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tgpaths
tgpaths.setup()
from tinygrad.runtime.ops_amd import AMDDevice
import amd_compile_helper as ach
import amd_dispatch_daemon as daemon
import fake_am_gpu as amg

HERE, WORK = tgpaths.HERE, tgpaths.WORK
tgpaths.build_cpp(HERE / "golden_amd_program.cpp", WORK / "golden_amd_program")
LIB_VA = 0x7f_8000_0000

hdr = (tgpaths.GPU_DIR / "kernels/BeagleOpenCL_kernels.h").read_text().split("\n")
def variant(name):   # as amd_compile_probe.py: the KERNELS_STRING_<name> the plugin sends the daemon
    i = next(k for k, l in enumerate(hdr) if l.startswith(f"#define {name} \""))
    out = []
    for l in hdr[i + 1:]:
        if l == '"': break
        out.append(l[:-3].replace('\\"', '"').replace('\\\\', '\\'))
    return "\n".join(out) + "\n"

def stub_dev(copies):
    dev = types.SimpleNamespace(target=PROPS["target"], prof_prg_counter=itertools.count(0), synchronize=lambda: None,
                                iface=types.SimpleNamespace(props={"lds_size_in_kb": PROPS["lds_size_in_kb"]}), private=[])
    dev._ensure_has_local_memory = lambda size: dev.private.append(size)
    dev.allocator = types.SimpleNamespace(alloc=lambda size, spec=None: types.SimpleNamespace(va_addr=LIB_VA, size=size),
                                          _copyin=lambda buf, mv: copies.append(bytes(mv)), free=lambda *a, **k: None)
    return dev

def scratch_ref(private):
    """AMDDevice._ensure_has_local_memory on a fresh stub, as at device init: (scratch bytes, tmpring_size)."""
    d = types.SimpleNamespace(max_private_segment_size=0, target=PROPS["target"], cu_cnt=PROPS["cu_cnt"], xccs=PROPS["xccs"], se_cnt=PROPS["se_cnt"],
                              iface=types.SimpleNamespace(props={"max_slots_scratch_cu": PROPS["max_slots_scratch_cu"]}))
    d._realloc = lambda old, size: (types.SimpleNamespace(size=size), True)
    AMDDevice._ensure_has_local_memory(d, private)
    return d.scratch.size, d.tmpring_size

results = []
for device in (0x744c, 0x7550):
    PROPS = amg.props_for(device)
    for prec in ("SP", "DP"):
        for n in (4, 16, 32, 48, 64, 80, 128, 192, 256):
            name = f"{PROPS['arch']} KERNELS_STRING_{prec}_{n}"
            hsaco = ach.compile_hip(variant(f"KERNELS_STRING_{prec}_{n}"), PROPS["arch"])
            _, kernels = ach.parse_kernels(hsaco)
            copies, ref, privates = [], {}, []
            dev = stub_dev(copies)
            for kname, (kd_addr, _desc) in sorted(kernels.items()):
                prg = daemon.BeagleAMDProgram(dev, kname, hsaco, kd_addr, 0)
                ref[kname] = (kd_addr, prg.group_segment_size, prg.private_segment_size, prg.kernargs_segment_size, prg.kernargs_alloc_size,
                              int(prg.wave32), prg.rsrc1, prg.rsrc2, prg.rsrc3, prg.prog_addr, prg.aql_prog_addr)
                privates.append(prg.private_segment_size)
            assert all(c == copies[0] for c in copies), "BeagleAMDProgram uploaded different images"
            sizes = sorted({1, 68, 128, 136, 572, 692, 1024, max([128] + privates)})   # 0: tinygrad returns before sizing
            (WORK / "golden_amd_program.hsaco").write_bytes(hsaco)
            args = [str(WORK / "golden_amd_program"), str(WORK), str(LIB_VA)] + [str(PROPS[k]) for k in ("cu_cnt", "se_cnt", "xccs", "max_slots_scratch_cu", "lds_size_in_kb")] \
                   + [str(PROPS["target"][0])]
            r = subprocess.run(args + [str(s) for s in sizes], capture_output=True, text=True)
            if r.returncode != 0: print(f"{name}: the C++ loader failed: {r.stdout}{r.stderr}"); results.append(False); continue
            got, got_scratch = {}, {}
            for line in r.stdout.splitlines():
                f = line.split()
                if f[0] == "K": got[f[1]] = tuple(int(x) for x in f[2:])
                elif f[0] == "S": got_scratch[int(f[1])] = (int(f[2]), int(f[3]))
            image_ok = (WORK / "golden_amd_program_image.bin").read_bytes() == copies[0]
            rec_ok = got == ref
            scratch_ok = all(got_scratch.get(s) == scratch_ref(s) for s in sizes)
            if not rec_ok:
                for k in ref:
                    if got.get(k) != ref[k]: print(f"  {k}: ref {ref[k]}\n       c++ {got.get(k)}"); break
            if not scratch_ok: print("  scratch:", {s: (got_scratch.get(s), scratch_ref(s)) for s in sizes if got_scratch.get(s) != scratch_ref(s)})
            ok = image_ok and rec_ok and scratch_ok
            results.append(ok)
            print(f"{name}: {len(ref)} kernels, image {len(copies[0])} B, private {min(privates)}-{max(privates)} B, scratch "
                  f"{got_scratch[max([128] + privates)][0] >> 20} MiB: {'IDENTICAL' if ok else 'MISMATCH'}"
                  + ("" if image_ok else " (image)") + ("" if rec_ok else " (records)") + ("" if scratch_ok else " (scratch)"))
print("A1d HSACO loader and scratch vs BeagleAMDProgram:", "all identical" if all(results) else "MISMATCH")
sys.exit(0 if all(results) else 1)
