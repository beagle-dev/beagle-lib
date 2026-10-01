"""A stand-in for amd_dispatch_daemon.py (TODO.md plan step A1h), spawned by the plugin as BEAGLE_AMD_DISPATCH_DAEMON with
the same arguments and the same length-prefixed protocol. It boots nothing. On the plugin's TinyGPU.app connection (to
fake_amd_device.py) it allocates the queues' memory as tinygrad's AMDDevice would (MAP_SYSMEM_FD) and tells the fake GPU
where each buffer is mapped and where each queue is (FAKE_MAP, FAKE_QUEUE). compile_all compiles the real HSACO as the
daemon does (amd_compile_helper.compile_hip) and gives the fake GPU its kernels (FAKE_HSACO); handoff replies as cmd_handoff
does; anything after the handoff but fini is refused.
Sizes, for wraps: FAKE_AMD_RING_KB (the compute ring, default 16 MiB as AMDDevice's), FAKE_AMD_SDMA_RING_KB, FAKE_AMD_KARGS_KB
(default 16 MiB each), FAKE_AMD_POOL_MB (the VRAM pool when the plugin asks for the default, 1024).
    python3 fake_amd_daemon.py <cmd_sock_fd> <tgpu_fd>"""
import os, sys, json, struct, socket, array
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tgpaths
tgpaths.setup()
from tinygrad.runtime.autogen import hsa
from tinygrad.runtime.support.elf import elf_loader
import amd_compile_helper as ach
import fake_amd_device as fake

cmd = socket.socket(fileno=int(sys.argv[1]))
tg = socket.socket(fileno=os.dup(int(sys.argv[2])))
log = open(os.path.join(os.environ.get("TINYGPU_TEST_WORK", "/tmp"), "fake_amd_daemon.log"), "w", buffering=1)

def recv_exact(s, n):
    b = bytearray()
    while len(b) < n:
        c = s.recv(n - len(b))
        if not c: return None
        b += c
    return bytes(b)
def recv_msg():
    h = recv_exact(cmd, 4)
    return None if h is None else json.loads(recv_exact(cmd, struct.unpack("<I", h)[0]))
def send_json(obj):
    body = json.dumps(obj).encode()
    cmd.sendall(struct.pack("<I", len(body)) + body)
def rpc(c, bar=0, a0=0, a1=0, a2=0, payload=b"", fd=False):
    tg.sendall(fake.REQ.pack(c, 0, bar, a0, a1, a2) + payload)
    if fd:
        msg, anc, _, _ = tg.recvmsg(17, socket.CMSG_LEN(4))
        return fake.RESP.unpack(msg) + (struct.unpack("<i", anc[0][2][:4])[0],)
    return fake.RESP.unpack(recv_exact(tg, 17))

KB = lambda name, default: int(os.environ.get(name, default)) << 10
BUFS = [("compute_ring", KB("FAKE_AMD_RING_KB", 16384)), ("compute_gart", 0x100), ("sdma_ring", KB("FAKE_AMD_SDMA_RING_KB", 16384)),
        ("sdma_gart", 0x100), ("signals", 0x1000), ("kargs", KB("FAKE_AMD_KARGS_KB", 16384)), ("staging", 16 << 20)]
VA = lambda i: 0x7f_0000_0000 + i * 0x1000_0000
POOL_VA = 0x10_0000_0000
RPTR, WPTR = getattr(hsa.amd_queue_t, "read_dispatch_id").offset, getattr(hsa.amd_queue_t, "write_dispatch_id").offset
COMPUTE_PUT, SDMA_PUT, TIMELINE = 1234, 4096, 7   # as after boot and compile: the queues used, the timeline at 7
for bar in (0, 2, 5): rpc(fake.MAP_BAR, bar)   # as tinygrad maps them, in the session the plugin then shares
bufs = {}
for i, (name, size) in enumerate(BUFS):
    st, mapped, idx, fd = rpc(fake.MAP_SYSMEM_FD, a0=size, fd=True)
    bufs[name] = dict(i=i, fd=fd, size=mapped, idx=idx, va=VA(i))
    m = json.dumps(dict(va=VA(i), size=mapped, sysmem=idx, off=0, kargs=name == "kargs")).encode()
    rpc(fake.FAKE_MAP, a0=len(m), payload=m)
def poke(name, off, fmt, v):   # the daemon's own view of a buffer, as tinygrad writes it
    import mmap
    mm = mmap.mmap(bufs[name]["fd"], bufs[name]["size"])
    struct.pack_into(fmt, mm, off, v)
    mm.close()
for kind, put in (("compute", COMPUTE_PUT), ("sdma", SDMA_PUT)):
    for off in (RPTR, WPTR): poke(f"{kind}_gart", off, "<Q", put)
    q = json.dumps(dict(kind=kind, doorbell=0x18 if kind == "compute" else 0x800, ring_map=bufs[f"{kind}_ring"]["idx"], ring_off=0,
                        ring_size=BUFS[0][1] if kind == "compute" else BUFS[2][1], rptr_map=bufs[f"{kind}_gart"]["idx"], rptr_off=RPTR,
                        wptr_map=bufs[f"{kind}_gart"]["idx"], wptr_off=WPTR, put=put)).encode()
    rpc(fake.FAKE_QUEUE, a0=len(q), payload=q)
poke("signals", 0x40, "<Q", TIMELINE - 1)   # the timeline signal, synchronized
log.write(f"fake AMD daemon: {len(bufs)} sysmem buffers mapped\n")

def compile_for_fake(src):
    """The daemon's compile_all compile, and the fake GPU's kernel table from it (FAKE_HSACO): each kernel's descriptor offset,
    entry, rsrc registers and kernarg size, and the relocated image, as BeagleAMDProgram derives them."""
    h = ach.compile_hip(src, "gfx1100")
    _image, kernels = ach.parse_kernels(h)
    img, _sections, relocs = elf_loader(h)
    img = bytearray(img)
    for off, sym, typ, add in relocs:   # BeagleAMDProgram's relocation loop
        assert typ == 5
        img[off:off + 8] = struct.pack("<q", sym - off + add)
    info = {"kernels": {}}
    for name, (kd, d) in kernels.items():
        lds = ((d.group_segment_fixed_size + 511) // 512) & 0x1FF
        info["kernels"][name] = dict(kd_off=kd, entry=d.kernel_code_entry_byte_offset, rsrc1=d.compute_pgm_rsrc1 | (1 << 20),
                                     rsrc2=d.compute_pgm_rsrc2 | (lds << 15), rsrc3=d.compute_pgm_rsrc3, kernarg_size=d.kernarg_size)
    j = json.dumps(info).encode()
    rpc(fake.FAKE_HSACO, a0=len(j) + len(img), a1=len(j), payload=j + bytes(img))
    return h

hdr = (tgpaths.GPU_DIR / "kernels/BeagleOpenCL_kernels.h").read_text().split("\n")
def variant_source(name):   # KERNELS_STRING_<name>, as the plugin would send it for compile_all
    i = next(k for k, l in enumerate(hdr) if l.startswith(f"#define KERNELS_STRING_{name} \""))
    lines = []
    for l in hdr[i + 1:]:
        if l == '"': break
        lines.append(l[:-3].replace('\\"', '"').replace('\\\\', '\\'))
    return "\n".join(lines) + "\n"
class _Sources(dict):
    def __missing__(self, name): return variant_source(name)
KERNEL_SOURCES = _Sources()

hsaco, handed_off = None, False
while (req := recv_msg()) is not None:
    c = req.get("cmd")
    log.write(f"<- {c}\n")
    if handed_off and c != "fini": send_json({"ok": False, "error": f"{c}: the GPU queues belong to the C++ side after handoff"}); continue
    if c == "boot": send_json({"ok": True, "arch": "gfx1100"})
    elif c == "compile_all":
        hsaco = compile_for_fake(open(req["cl_path"]).read())
        send_json({"ok": True, "kernels": list(ach.parse_kernels(hsaco)[1])})
    elif c == "handoff":
        blob = hsaco or b""   # none when the plugin has the build's HSACO: the fake GPU still needs the kernels, compiled here
        if hsaco is None: compile_for_fake(KERNEL_SOURCES[req["variant"]])
        pool = int(req.get("pool_size") or (int(os.environ.get("FAKE_AMD_POOL_MB", 1024)) << 20))
        m = json.dumps(dict(va=POOL_VA, size=pool, sysmem=-1, off=0)).encode()
        rpc(fake.FAKE_MAP, a0=len(m), payload=m)
        order = [n for n, _ in BUFS]
        info = {"ok": True, "nmaps": len(order), "blob_size": len(blob)}
        for i, n in enumerate(order): info[f"map{i}_size"] = bufs[n]["size"]
        place = lambda key, name, off: info.update({f"{key}_map": order.index(name), f"{key}_off": off})
        for kind, put in (("compute", COMPUTE_PUT), ("sdma", SDMA_PUT)):
            place(f"{kind}_ring", f"{kind}_ring", 0); place(f"{kind}_rptr", f"{kind}_gart", RPTR); place(f"{kind}_wptr", f"{kind}_gart", WPTR)
            info.update({f"{kind}_ring_size": BUFS[0][1] if kind == "compute" else BUFS[2][1], f"{kind}_doorbell": 0x18 if kind == "compute" else 0x800,
                         f"{kind}_put": put})
        place("signal", "signals", 0x40); place("shadow", "signals", 0x50)
        place("kargs", "kargs", 0); place("staging", "staging", 0)
        info.update(signal_va=bufs["signals"]["va"] + 0x40, shadow_va=bufs["signals"]["va"] + 0x50, kargs_va=bufs["kargs"]["va"],
                    kargs_size=BUFS[5][1], staging_va=bufs["staging"]["va"], staging_size=BUFS[6][1], pool_va=POOL_VA, pool_size=pool,
                    timeline_value=TIMELINE, vram_size=20464 << 20, target_major=11, xccs=1, cu_cnt=96, se_cnt=6, max_slots_scratch_cu=32,
                    lds_size_in_kb=64, ih_ring_paddr=fake.IH_RING_PADDR, ih_ring_size=fake.IH_RING_SIZE, is_vf=0,
                    **{f"bar{b}_size": fake.BARS[b][1] for b in (0, 2, 5)}, **{f"reg_{k}": v for k, v in fake.REG.items()})
        handed_off = True
        send_json(info)
        cmd.sendall(blob)
        socket.send_fds(cmd, [b"F"], [bufs[n]["fd"] for n in order])
        log.write(f"handoff: pool {pool >> 20} MiB\n")
    elif c == "fini":
        send_json({"ok": True})
        break
    else: send_json({"ok": False, "error": f"the fake daemon has no {c}"})
log.write("fake AMD daemon: exiting\n")
