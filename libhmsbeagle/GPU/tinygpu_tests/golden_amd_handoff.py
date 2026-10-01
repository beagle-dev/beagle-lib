"""Golden test for the AMD handoff (TODO.md plan step A1e): amd_dispatch_daemon.py's own cmd_handoff runs on a stub AMDDevice
(its queues, timeline, kernargs and sysmem as real files with fds, as BeagleTinyGPUDevice keeps them), and
golden_amd_handoff.cpp parses the reply with TinyGPUHybridAMDRuntime.h and attaches the fds. Every field must reach the C++
side, and every object the C++ side reads or writes through the mappings (each ring's first dword, the read and write
pointers, both timeline signals, kernargs, staging) must be the stub's own. The registers are this card's: tinygrad's
register tables at the bases of its captured discovery table (STATUS.md R64-R65), so the C++ checks that they are inside
BAR5 run on the real addresses."""
import os, sys, json, mmap, socket, struct, types, random, tempfile, functools, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tgpaths
tgpaths.setup()
from tinygrad.runtime import ops_amd
from tinygrad.runtime.autogen import hsa
from tinygrad.runtime.support.hcq import HCQBuffer, MMIOInterface
from tinygrad.runtime.support.amd import AMDReg, import_asic_regs
import amd_dispatch_daemon as daemon

HERE, WORK = tgpaths.HERE, tgpaths.WORK
tgpaths.build_cpp(HERE / "golden_amd_handoff.cpp", WORK / "golden_amd_handoff")
meta = json.load(open(sorted((tgpaths.DATA / "discovery").glob("1002_744c_*.json"))[0]))
bases = lambda ip: {int(i): tuple(b) for i, b in meta["regs_offset"][ip].items()}
regs = {}
for prefix, ver, ip in (("gc", (11, 0, 0), "GC_HWIP"), ("nbio", (4, 3, 0), "NBIO_HWIP"), ("osssys", (6, 0, 0), "OSSSYS_HWIP")):
    regs.update(import_asic_regs(prefix, ver, cls=functools.partial(AMDReg, bases=bases(ip))))
rng = random.Random(7)
tmp = tempfile.mkdtemp(dir=WORK)
pci = types.SimpleNamespace(sysmem_fds={}, bar_info=lambda bar: {0: (0x2e_4000_0000, 256 << 20), 2: (0x2e_5000_0000, 2 << 20), 5: (0x2e_0030_0000, 1 << 20)}[bar])
mms = []
def sysmem(size, va):
    """A sysmem buffer as BeagleTinyGPUDevice maps one: a file, its fd kept, an HCQBuffer whose view is the mapping."""
    path = os.path.join(tmp, f"sys{len(mms)}")
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600); os.ftruncate(fd, size)
    mm = mmap.mmap(fd, size); mms.append(mm)
    view = MMIOInterface(ctypes_addr(mm), size, fmt="B")
    pci.sysmem_fds[view.addr] = (fd, size)
    return HCQBuffer(va, size, view=view)
def ctypes_addr(mm):
    import ctypes
    return ctypes.addressof(ctypes.c_char.from_buffer(mm))
def queue(ring_size, va, doorbell, put):
    ring, gart = sysmem(ring_size, va), sysmem(0x4000, va + 0x1000_0000)
    ring.cpu_view().view(fmt="I")[0] = rng.getrandbits(32)
    rp, wp = getattr(hsa.amd_queue_t, "read_dispatch_id").offset, getattr(hsa.amd_queue_t, "write_dispatch_id").offset
    gart.cpu_view().view(offset=rp, size=8, fmt="Q")[0] = put - 3
    gart.cpu_view().view(offset=wp, size=8, fmt="Q")[0] = put
    return ops_amd.AMDQueueDesc(ring=ring.cpu_view().view(fmt="I"), read_ptr=gart.cpu_view().view(offset=rp, size=8, fmt="Q"),
                                write_ptr=gart.cpu_view().view(offset=wp, size=8, fmt="Q"), doorbell=types.SimpleNamespace(residx=2, off=doorbell),
                                put_value=put)
page = sysmem(0x4000, 0x7f_5000_0000)
def signal(off, value):
    b = page.offset(off, 16)
    b.cpu_view().view(0, 8, "Q")[0] = value
    return types.SimpleNamespace(base_buf=b, value_addr=b.va_addr)
cq, sq = queue(16 << 20, 0x7f_1000_0000, 0x18, 98765), queue(16 << 20, 0x7f_3000_0000, 0x800, 4096 * 37)
adev = types.SimpleNamespace(vram_size=20464 << 20, is_vf=False, ih=types.SimpleNamespace(rings=[(0x3f00_0000, 0x3f04_0000, "", 0)], ring_size=256 << 10),
                             gmc=types.SimpleNamespace(pf_status_reg=lambda ip: f"reg{ip}VM_L2_PROTECTION_FAULT_STATUS"), reg=lambda name: regs[name])
vram_next = [0x10_0000_0000]
def alloc(size, spec=None):
    if spec is not None and spec.host: return sysmem(size, 0x7f_6000_0000)
    buf, vram_next[0] = HCQBuffer(vram_next[0], size), vram_next[0] + size
    return buf
dev = types.SimpleNamespace(compute_queue=cq, sdma_queue=lambda idx: sq, timeline_signal=signal(0x40, 41), _shadow_timeline_signal=signal(0x50, 0),
                            kernargs_buf=sysmem(16 << 20, 0x7f_7000_0000), allocator=types.SimpleNamespace(alloc=alloc), synchronize=lambda: None,
                            timeline_value=42, target=(11, 0, 0), xccs=1, cu_cnt=96, se_cnt=6,
                            iface=types.SimpleNamespace(dev_impl=adev, pci_dev=pci, props={"max_slots_scratch_cu": 32, "lds_size_in_kb": 64}))
dev.kernargs_buf.cpu_view().view(fmt="I")[0] = rng.getrandbits(32)

a, b = socket.socketpair()
d = daemon.Daemon(a)
d.dev, d.hsaco = dev, bytes(rng.getrandbits(8) for _ in range(1000))
d.cmd_handoff({"cmd": "handoff", "pool_size": 0})
assert d.handed_off
hdr = b.recv(4, socket.MSG_WAITALL)
js = b.recv(struct.unpack("<I", hdr)[0], socket.MSG_WAITALL).decode()
blob = b.recv(len(d.hsaco), socket.MSG_WAITALL)
_, fds, _, _ = socket.recv_fds(b, 1, 16)
info = json.loads(js)
(WORK / "golden_amd_handoff.json").write_text(js)
staging = d._handoff_bufs[1]
staging.cpu_view().view(fmt="I")[0] = rng.getrandbits(32)
want = {"compute_ring0": cq.ring[0], "compute_rptr": cq.read_ptr[0], "compute_wptr": cq.write_ptr[0], "sdma_ring0": sq.ring[0],
        "sdma_rptr": sq.read_ptr[0], "sdma_wptr": sq.write_ptr[0], "signal": 41, "shadow": 0, "kargs0": dev.kernargs_buf.cpu_view().view(fmt="I")[0],
        "staging0": staging.cpu_view().view(fmt="I")[0], "pool_size": (20464 << 20) // 2, "timeline_value": 42, "blob": int(blob == d.hsaco),
        "reg_ih_wptr": regs["regIH_RB_WPTR"].addr[0], "reg_hdp_remap": regs["regBIF_BX0_REMAP_HDP_MEM_FLUSH_CNTL"].addr[0],
        "reg_fault_status": regs["regGCVM_L2_PROTECTION_FAULT_STATUS"].addr[0], "compute_doorbell": 0x18, "sdma_put": 4096 * 37}
r = subprocess.run([str(WORK / "golden_amd_handoff"), str(WORK / "golden_amd_handoff.json")] + [str(f) for f in fds], pass_fds=fds,
                   capture_output=True, text=True)
got = {l.split()[0]: int(l.split()[1]) for l in r.stdout.splitlines() if len(l.split()) == 2}
got["blob"] = 1 if blob == d.hsaco else 0
bad = {k: (v, got.get(k)) for k, v in want.items() if got.get(k) != v}
print(f"cmd_handoff: {len(info)} keys, {info['nmaps']} mappings, registers {', '.join(f'{k[4:]} {v:#x}' for k, v in info.items() if k.startswith('reg_'))}")
if r.returncode != 0: print("the C++ side refused:", r.stdout, r.stderr)
print("A1e handoff from the daemon's cmd_handoff to the C++ runtime:", "IDENTICAL" if r.returncode == 0 and not bad else f"MISMATCH {bad}")
sys.exit(0 if r.returncode == 0 and not bad else 1)
