"""Offline tests for plan step C6's daemon half (the C++ half is golden_mm.py, and runs end to end in test_c6.sh): the real
cmd_handoff at level vram allocates the four buffers but not the pool, at sysmem nothing (no fds, and no fd byte), and both
reply with tinygrad's memory manager as it is (_mm_export: the allocators' states, the class VA allocator, the root page
table, PAGESIZE, GMMU, the WPR bound, BAR1's size, the daemon's sysmem count); afterwards the daemon allocates nothing; a
buffer tinygrad's LRU cache holds for one the C++ side allocates is refused; cmd_state_page at sysmem maps the C++
timeline from the fd that follows the page's, reads it through tinygrad's NVSignal, and keeps the stream framed when the
fds do not match the request. No GPU, no TinyGPU socket.
    python test_c6.py"""
import os, sys, io, json, types, ctypes, struct, socket, tempfile, mmap, contextlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tgpaths
tgpaths.setup()
import test_p3 as p3
import nv_dispatch_daemon as d
from tinygrad import Device
from tinygrad.device import BufferSpec
from tinygrad.runtime.autogen import nv
from tinygrad.runtime.support.memory import TLSFAllocator
from tinygrad.runtime.support.hcq import MMIOInterface

MB = 1 << 20

def handoff_rig(cached=()):
    """A real Daemon (C++ runtime mode) over a stub NVDevice whose allocator draws VAs and paddrs from real TLSF allocators
    and records every allocation; the memory manager _mm_export reads is built from real allocators too."""
    va = TLSFAllocator(1 << 44, base=0x1000000000)
    pa = TLSFAllocator((8188 - 64 - 2 - 16) * MB, base=(2 + 16) * MB)
    pa.alloc(100 * MB, 0x1000)   # the boot's
    views, fds, allocs = [], {}, []
    def alloc(size, spec=None):
        allocs.append((size, spec))
        views.append((ctypes.c_uint8 * 64)()); view = MMIOInterface(ctypes.addressof(views[-1]), 64)
        if spec is not None: fds[view.addr] = os.open(os.devnull, os.O_RDONLY)   # a sysmem buffer's fd, as alloc_sysmem keeps it
        else: pa.alloc(size, 0x1000)
        return types.SimpleNamespace(va_addr=va.alloc(size, 0x1000), size=size, cpu_view=lambda: view)
    mm = types.SimpleNamespace(vram_size=(8188 - 64) * MB, va_bits=48, va_shifts=[12, 21, 29, 38, 47], va_base=0,
                               palloc_ranges=[(512 * MB, 512 * MB), (2 * MB, 2 * MB), (4096, 4096)], reserve_ptable=True,
                               root_page_table=types.SimpleNamespace(paddr=0, lv=0), boot_allocator=TLSFAllocator(2 * MB),
                               ptable_allocator=TLSFAllocator(16 * MB, base=2 * MB), pa_allocator=pa, va_allocator=va)
    impl = types.SimpleNamespace(fmc_boot=False, mmu_ver=2, vram_size=8188 * MB, mm=mm,
                                 gsp=types.SimpleNamespace(wpr_meta=bytes(nv.GspFwWprMeta(gspFwRsvdStart=0x1f3a00000))))
    cache = {k: ["cached"] for k in cached}
    pci = types.SimpleNamespace(sysmem_fds=fds, bar_info=lambda bar: (0x1d_0000_0000, 256 * MB))
    impl.beagle_fini = dict(p3.CONFIRMED)
    dev = types.SimpleNamespace(allocator=types.SimpleNamespace(alloc=alloc, cache=cache), synchronize=lambda: None, error_state=None,
                                finalize=lambda: None,
                                iface=types.SimpleNamespace(dev_impl=impl, pci_dev=pci, compute_class=0xc9c0), sass_version=0x89,
                                shared_mem_window=0, local_mem_window=0, num_gpcs=3, num_tpc_per_gpc=4, num_sm_per_tpc=2, max_warps_per_sm=48)
    a, b = socket.socketpair()
    dm = d.Daemon(b, 7)
    dm.dev, dm.elf_bytes = dev, b"elf"
    def hold(): raise p3.Held()   # the real one keeps the connection open for good
    dm._hold = hold
    Device._opened_devices.add("NV")
    return a, dm, allocs, mm, fds

def converse(a, dm, reqs):
    """The daemon serves the requests, then the plugin goes away; returns each reply with the fds and bytes after it."""
    real_build = d.build_handoff
    d.build_handoff = lambda dev, progs, bufs: ({"qmd_ver": 3, "c_ring_bar": 1, "c_gpput_bar": 1, "d_ring_bar": 1, "d_gpput_bar": 1, "db_bar": 0,
                                                 **{f"{n}_va": b.va_addr for n, b in bufs.items()}}, b"BLOB")
    try:
        for r in reqs: a.sendall(p3.msg(r))
        a.shutdown(socket.SHUT_WR)
        with contextlib.redirect_stderr(io.StringIO()): dm.run()
    except p3.Held: pass
    finally: d.build_handoff = real_build
    dm.sock.close()
    out = []
    a.settimeout(5)
    while True:   # exact reads, so an fd byte is read by the recvmsg that takes its SCM_RIGHTS
        hdr = a.recv(4, socket.MSG_WAITALL)
        if len(hdr) < 4: break
        r = json.loads(a.recv(struct.unpack("<I", hdr)[0], socket.MSG_WAITALL))
        blob, fds = b"", []
        if r.get("ok") and "blob_size" in r:
            blob = a.recv(r["blob_size"], socket.MSG_WAITALL) if r["blob_size"] else b""
            if r["nfds"]:
                data, fds, _, _ = socket.recv_fds(a, 1, 8)
                assert data == b"F", data
        out.append((r, blob, fds))
    return out

def test_handoff_levels():
    real, Device._opened_devices = Device._opened_devices, set()
    try:
        for level in ("", "vram", "sysmem"):
            a, dm, allocs, mm, fds = handoff_rig()
            req = {"cmd": "handoff", "programs": False, "pool_size": 0, **({"level": level} if level else {})}
            (r, blob, got), (alloc_r, _, _) = converse(a, dm, [req, {"cmd": "alloc", "size": 4096}])
            specs = [s for _, s in allocs]
            want_bufs = [BufferSpec(**s) for _, _, s in d._HANDOFF_BUFS]
            assert r["ok"] and blob == b"BLOB", r
            if level == "":
                assert specs == want_bufs + [None, None] and allocs[4][0] == 8188 * MB // 2 and r["nfds"] == 4 and len(got) == 4, (specs, r)
                assert "pool_va" in r and "mm_pa" not in r and alloc_r["ok"], r
                continue
            assert specs == (want_bufs if level == "vram" else []) and "pool_va" not in r and "pool_size" not in r, (level, specs, r)
            assert r["nfds"] == (4 if level == "vram" else 0) and len(got) == r["nfds"], (level, r["nfds"], got)
            for key, alloc in (("mm_boot", mm.boot_allocator), ("mm_ptable", mm.ptable_allocator), ("mm_pa", mm.pa_allocator), ("mm_va", mm.va_allocator)):
                assert r[key] == d._tlsf_save(alloc), key   # the state after the daemon's last allocation, bucket order included
            assert (r["mm_mmu_ver"], r["mm_vram_size"], r["mm_va_bits"], r["mm_va_shifts"], r["mm_palloc_ranges"], r["mm_reserve_ptable"]) == \
                   (2, (8188 - 64) * MB, 48, [12, 21, 29, 38, 47], [512 * MB, 512 * MB, 2 * MB, 2 * MB, 4096, 4096], 1), r
            assert (r["mm_root"], r["mm_root_lv"], r["mm_pagesize"], r["mm_gmmu"], r["mm_wpr_bound"], r["mm_dev_vram_size"], r["bar1_size"]) == \
                   (0, 0, mmap.PAGESIZE, 1, 0x1f3a00000, 8188 * MB, 256 * MB), r
            assert r["mm_sysmem_count"] == len(fds) == (4 if level == "vram" else 0), (r["mm_sysmem_count"], len(fds))
            assert not alloc_r["ok"] and "the C++ side owns the memory manager" in alloc_r["error"] and dm.mm_exported, alloc_r
        for bad, why in (({"level": "vram", "programs": True}, "programs false"), ({"level": "boot"}, "level 'boot'")):
            a, dm, allocs, _, _ = handoff_rig()
            (r, _, _), = converse(a, dm, [{"cmd": "handoff", "programs": False, **bad}])
            assert not r["ok"] and why in r["error"] and allocs == [] and not dm.handed_off, (bad, r)
    finally: Device._opened_devices = real
    print("handoff levels: vram allocates the four buffers but not the pool, sysmem nothing (no fds, no fd byte); both export the "
          "memory manager as it is, bucket order included; the daemon allocates nothing afterwards; bad levels refused")

def test_lru_refusal():
    real, Device._opened_devices = Device._opened_devices, set()
    try:
        pool_key, cmdq_key = (8188 * MB // 2, None), (2 << 20, BufferSpec(cpu_access=True))
        for level, cached, refused in (("vram", [pool_key], True), ("vram", [cmdq_key], False), ("sysmem", [cmdq_key], True), ("", [pool_key], False)):
            a, dm, allocs, _, _ = handoff_rig(cached=cached)
            (r, _, _), = converse(a, dm, [{"cmd": "handoff", "programs": False, "pool_size": 0, **({"level": level} if level else {})}])
            assert r["ok"] is not refused and (not refused or ("cached buffer" in r["error"] and allocs == [])), (level, cached, r)
    finally: Device._opened_devices = real
    print("LRU cache: a buffer tinygrad's allocator would reuse for one the C++ side allocates refuses the level (and only then)")

def test_state_page_signal():
    real, Device._opened_devices = Device._opened_devices, set()
    try:
        for n_fds, with_va, ok in ((2, True, True), (1, True, False), (2, False, False)):
            a, dm, calls, _, _ = p3.rig(signal=3)
            dm.handed_off, dm._handoff_bufs = True, {}   # level sysmem: the daemon has no buffers of its own
            page = tempfile.TemporaryFile(); page.truncate(d._STATE_WORDS * 8)
            mine = mmap.mmap(page.fileno(), d._STATE_WORDS * 8); struct.pack_into("<Q", mine, 0, d._PHASE_DISPATCH)
            sig = tempfile.TemporaryFile(); sig.truncate(0x4000)
            cpp = mmap.mmap(sig.fileno(), 0x4000); struct.pack_into("<Q", cpp, 0, 41)   # the C++ side's timeline, as it maps it
            req = {"cmd": "state_page", **({"signal_va": 0x1020768000, "signal_size": 0x4000} if with_va else {})}
            a.sendall(p3.msg(req)); socket.send_fds(a, [b"S"], [page.fileno(), sig.fileno()][:n_fds]); page.close(); sig.close()
            a.sendall(p3.msg({"cmd": "teardown_export"})); a.shutdown(socket.SHUT_WR)   # parses only if the fd byte was consumed
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                try: dm.run()
                except p3.Held: pass
            dm.sock.close()
            r, nxt = p3.recv_reply(a), p3.recv_reply(a)
            assert r["ok"] is ok and "ok" in nxt, (n_fds, with_va, r, nxt)
            if ok:
                assert dm._cpp_signal.value == 41 and dm._cpp_signal.value_addr == 0x1020768000, dm._cpp_signal.value
                struct.pack_into("<Q", cpp, 0, 42)
                assert dm._cpp_signal.value == 42   # live: tinygrad's NVSignal over the C++ side's mapping
            else: assert f"{n_fds} fds received" in r["error"], r
    finally: Device._opened_devices = real
    print("state page at sysmem: the C++ timeline mapped from the fd after the page's and read live through tinygrad's NVSignal; "
          "fds that do not match the request are refused with the stream still framed")

if __name__ == "__main__":
    test_handoff_levels()
    test_lru_refusal()
    test_state_page_signal()
    print("C6 daemon: all passed")
