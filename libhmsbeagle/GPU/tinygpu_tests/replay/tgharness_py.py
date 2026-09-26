"""Pre-connect patches for running the real daemon offline (TODO.md plan step V1): against fake_nv_device.py or the
replay server (tgreplay.py), with the eGPU unplugged. The daemon reaches TinyGPU.app only through the connection the
plugin hands it (nv_dispatch_daemon._install_inherited_tinygpu), which the harness points at the fake; what tinygrad
does before it uses that connection needs two changes, documented here, and nothing after it is touched: every wire
byte is tinygrad's own.
  - System.list_devices returns the AD107 on usb4, as macOS's IOKit scan does with the eGPU attached (system.py:58-83);
    with it unplugged the scan finds nothing;
  - every constructor that could reach the real eGPU raises (tgpaths.block_real_devices): the inherited-fd device has its
    own constructor and is unaffected.
install() is called by tgdaemon.py when BEAGLE_TG_OFFLINE=1.

BEAGLE_TG_MUTATE=<name> also applies one deliberate defect (MUTATIONS), for plan V1's check that a replay reports it: the
harness must catch a wrong PTE, RPC field, write order or zeroing, a short sleep, and a mutated oracle (tinygrad reading
NV_PMC_BOOT_42 at 0x168, the legacy GPUInterfaceTinyGPU.cpp's bug)."""
import os, sys, pathlib

MUTATIONS = {
    "pte-bit": "the 3rd page PTE tinygrad writes points one page further",
    "rpc-field": "the golden image's NV01_DEVICE_0 rm_alloc carries hClientShare + 1",
    "swap-writes": "the first two falcon BROM writes (ENGIDMASK, CURR_UCODE_ID) swap places",
    "drop-zero": "the first palloc zeroing write (the root page table's) is dropped",
    "reset-short": "NV_FLCN.reset holds each engine reset 0.05 s instead of 0.1 s",
    "boot42": "tinygrad reads NV_PMC_BOOT_42 at 0x168 instead of 0xa00",
    # for the guard (tgguard.py), live behind the proxy: what it must refuse to forward
    "pte-sys-bad": "the 3rd sysmem PTE tinygrad writes points at an unknown device page",
    "mailbox-bad": "booter_load's mailboxes point 256 MiB past the WPR meta",
    "rpc-corrupt": "the first RPC queued after the GSP starts carries a bad checksum",
}

def install():
    here = pathlib.Path(__file__).resolve().parent
    if str(here.parent) not in sys.path: sys.path.insert(0, str(here.parent))
    import tgpaths
    tgpaths.block_real_devices()
    from tinygrad.runtime.support import system
    type(system.System).list_devices = lambda self, vendor, devices, base_class=None: [(system.APLRemotePCIDevice, "usb4")]
    print("tgharness_py: offline: System.list_devices finds the device on usb4; real-device constructors blocked", file=sys.stderr, flush=True)
    if (m := os.environ.get("BEAGLE_TG_MUTATE")): mutate(m)

def mutate(name):
    if name not in MUTATIONS: raise SystemExit(f"tgharness_py: unknown mutation {name}; known: {', '.join(MUTATIONS)}")
    from tinygrad.runtime.support.nv import nvdev, ip
    from tinygrad.runtime.support.memory import MemoryManager
    from tinygrad.runtime.autogen import nv_570 as nv_gpu
    n = [0]
    if name == "pte-bit":
        orig = nvdev.NVPageTableEntry.set_entry
        def set_entry(self, entry_id, paddr, table=False, **kw):
            if not table and kw.get("valid", True):
                n[0] += 1
                if n[0] == 3: paddr += 0x1000
            return orig(self, entry_id, paddr, table=table, **kw)
        nvdev.NVPageTableEntry.set_entry = set_entry
    elif name == "rpc-field":
        orig = ip.NV_GSP.rpc_rm_alloc
        def rpc_rm_alloc(self, hParent, hClass, params, client=None):
            if hClass == nv_gpu.NV01_DEVICE_0 and client is None and not n[0]: n[0] = 1; params.hClientShare += 1
            return orig(self, hParent, hClass, params, client)
        ip.NV_GSP.rpc_rm_alloc = rpc_rm_alloc
    elif name == "swap-writes":
        orig, held = nvdev.NVDev.wreg, []
        def wreg(self, addr, value):
            if not n[0] and "NV_PFALCON2_FALCON_BROM_ENGIDMASK" in self.__dict__ and \
               addr == 0x110000 + self.NV_PFALCON2_FALCON_BROM_ENGIDMASK.base + self.NV_PFALCON2_FALCON_BROM_ENGIDMASK.off:
                n[0] = 1; held.append((addr, value)); return
            orig(self, addr, value)
            while held: orig(self, *held.pop())
        nvdev.NVDev.wreg = wreg
    elif name == "drop-zero":
        orig = MemoryManager.palloc
        def palloc(self, size, align=0x1000, zero=True, boot=False, ptable=False):
            if zero and not n[0]: n[0] = 1; zero = False
            return orig(self, size, align, zero=zero, boot=boot, ptable=ptable)
        MemoryManager.palloc = palloc
    elif name == "reset-short":
        import time as _time, types
        ip.time = types.SimpleNamespace(**{k: getattr(_time, k) for k in dir(_time) if not k.startswith("_")})
        ip.time.sleep = lambda s: _time.sleep(0.05 if s == 0.1 else s)
    elif name == "boot42":
        orig = nvdev.NVDev.include
        def include(self, nm, arch):
            orig(self, nm, arch)
            if nm == "nv_ref": r = self.NV_PMC_BOOT_42; self.NV_PMC_BOOT_42 = nvdev.NVReg(self, r.base, 0x168, r.fields)
        nvdev.NVDev.include = include
    elif name == "pte-sys-bad":
        from tinygrad.runtime.support.memory import AddrSpace
        orig = nvdev.NVPageTableEntry.set_entry
        def set_entry(self, entry_id, paddr, table=False, **kw):
            if not table and kw.get("aspace") is AddrSpace.SYS and kw.get("valid", True):
                n[0] += 1
                if n[0] == 3: paddr += 1 << 38
            return orig(self, entry_id, paddr, table=table, **kw)
        nvdev.NVPageTableEntry.set_entry = set_entry
    elif name == "mailbox-bad":
        orig = ip.NV_FLCN.execute_hs
        def execute_hs(self, base, img_paddr, *a, mailbox=None, **k):
            if mailbox is not None and mailbox != (0xff << 32) | 0xff: mailbox += 1 << 28
            return orig(self, base, img_paddr, *a, mailbox=mailbox, **k)
        ip.NV_FLCN.execute_hs = execute_hs
    elif name == "rpc-corrupt":
        orig = ip.NVRpcQueue._checksum
        def checksum(self, data):
            c = orig(self, data)
            if getattr(self.gsp, "stat_q", None) is not None and not n[0]: n[0] = 1; c ^= 1
            return c
        ip.NVRpcQueue._checksum = checksum
    print(f"tgharness_py: MUTATION {name}: {MUTATIONS[name]}", file=sys.stderr, flush=True)
