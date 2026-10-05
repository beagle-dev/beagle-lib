# TinyGPU-Hybrid: BEAGLE on an eGPU through TinyGPU.app

The TinyGPU-Hybrid backend runs BEAGLE's GPU kernels on an NVIDIA GPU in a USB4 or Thunderbolt enclosure attached to a Mac,
where there is no CUDA driver. It talks to the GPU through [TinyGPU.app](https://github.com/tinygrad/tinygpu_releases),
tinygrad's user-space PCIe server, and follows tinygrad's own NV driver (`tinygrad/runtime/ops_nv.py` and
`support/nv/`), ported to C++: BEAGLE boots the GPU's GSP-RM firmware itself, builds tinygrad's `NVDevice`, loads
ahead-of-time cubins and submits BEAGLE's kernels, with no Python at run time. On exit it unloads the GPU and runs
NVIDIA's driver-unload teardown, so the next process boots it again without a power cycle.

The AMD side of the backend (a Radeon through the same app; tested on an RX 7900 XT, gfx1100) follows tinygrad's AM driver
and AMD runtime the same way (`support/am/` and `ops_amd.py`), in C++ with no Python (TODO.md plan steps A1 and A2). BEAGLE
boots the card itself (the PSP, SMU, GMC, IH, GFX and SDMA), sets up tinygrad's `AMDDevice` queues and buffers, loads the
build's ahead-of-time HSACOs and submits PM4 and SDMA: launches, copies, allocations from a VRAM pool and synchronization.
Offline, on a register-level fake of the card, the boot sends TinyGPU.app the requests tinygrad's sends, byte for byte. It
is written for the RX 7900 XT's IP versions only. A card another session left unfinalized needs an SMU mode1 reset, which
BEAGLE never sends over TinyGPU: the boot refuses it, and the card must be power-cycled. Until plan step A2l tinygrad's
Python daemon booted the card; it is now the tests' oracle.

## Supported GPUs

| Family | PCI device IDs | Status |
|---|---|---|
| Ada (AD10x) | `0x26xx`-`0x28xx` | tested on an RTX 4060 (AD107) |
| Blackwell (GB20x) | `0x2bxx`-`0x2dxx`, `0x2fxx` | tested on an RTX 5070 (GB205); the other GB20x boot the same way, untested |
| Ampere (GA10x) | `0x22xx`-`0x25xx` | not supported: refused at `beagleCreateInstance` (its path, the Python daemon, was removed) |
| AMD Navi 31 (gfx1100) | `1002:744c` | tested on an RX 7900 XT; another AMD card's IP versions are refused at its boot |

Both precisions are built, for padded state counts 4, 16, 32, 48, 64, 80, 128, 192 and 256: on NVIDIA 18 cubins per
architecture (sm_86, sm_89, sm_120), on AMD 18 HSACOs for gfx1100. The resource offers `BEAGLE_FLAG_PRECISION_DOUBLE` as
well as single (TODO.md plan steps C16, A7); consumer GPUs run double precision natively, at a fraction of the single rate. A
GPU whose architecture has no cubin is refused at `beagleCreateInstance`.

## Requirements

- macOS on Apple silicon, with the eGPU attached.
- TinyGPU.app release `c0d024f9`, installed in `/Applications` and its system extension approved. BEAGLE checks both
  binaries' sha256 and refuses any other release, with the install instructions. BEAGLE never installs the app.
- NVIDIA's firmware, from linux-firmware at tinygrad's pin (GSP-RM 570.144, the booters, and on GB20x the FMC). BEAGLE
  looks in `BEAGLE_TINYGPU_FW`, an installed `share/beagle/firmware`, its own cache (`~/Library/Caches/beagle/firmware`)
  and tinygrad's download cache (`~/Library/Caches/tinygrad/downloads/fw`). A file in none of them is downloaded from the
  pinned URL with `/usr/bin/curl` into BEAGLE's cache before anything is written to the GPU. Every file's sha256 is checked
  against `TinyGPUFirmwareManifest.h`.
- For an AMD card, AMD's firmware (the six gfx1100 blobs, from the same linux-firmware pin, listed in
  `TinyGPUAMDBootTables.h`), the same way: located, or downloaded into BEAGLE's cache before anything is written to the GPU,
  every file checked. A blob that cannot be had stops the boot before it starts: `not booting: nothing was written to the
  GPU`.
- To fetch either vendor's by hand (for debugging, or a Mac without the network):
  `libhmsbeagle/GPU/tinygpu_fetch_firmware.sh [--chip ad102|gb202|gfx1100] DIR`, then `BEAGLE_TINYGPU_FW=DIR`.
- To build: `nvcc` and `ptxas` from CUDA 12.8, for the generated kernels header and the embedded cubins (on a Mac, through
  Docker; `-DTINYGPU_NVCC=` and `-DTINYGPU_PTXAS=` name them); for AMD, comgr (`libamd_comgr`, tinygrad's:
  `/opt/homebrew/lib/libamd_comgr.dylib`, or `-DTINYGPU_COMGR=`), for the embedded HSACOs. Without comgr the plugin refuses
  AMD cards. Nothing is compiled at run time.

## Building and installing

`BUILD_TINYGPU_HYBRID` (on by default) builds the plugin, `hmsbeagle-tinygpu-hybrid`, and the crash guard,
`beagle-tinygpu-guard`. Both are installed to the same directory: the plugin looks for the guard next to itself (or at
`BEAGLE_NV_GUARD`, and `BEAGLE_AMD_GUARD` for an AMD card), and refuses to boot without it. The resource appears in
`beagleGetResourceList` as `TinyGPU-NV-Hybrid` (`TinyGPU-AMD-Hybrid (1002:744c)` for the AMD card), with
`BEAGLE_FLAG_FRAMEWORK_TINYGPU`.

## Running

- One process at a time uses the eGPU: the first takes `$TMPDIR/nv_usb4.lock`, and a second fails at once. Every BEAGLE
  instance in that process shares one boot (about 2 s), which lasts until the process exits.
- Keep the Mac awake while the GPU runs (`caffeinate -ims`): a sleeping Mac with a live GPU risks the IOMMU (DART).
- A normal exit tears the GPU down (the GSP unload, then NVIDIA's teardown), and says so on stderr:
  `teardown: done: ... WPR2 is down, the next boot needs no power cycle`.
- Errors come back as BEAGLE errors. A failed boot, a GPU that hangs, or a lost TinyGPU.app connection makes
  `beagleCreateInstance`, or the calls that read results back, return `BEAGLE_ERROR_GENERAL`, and read-backs are NaN.
  BEAGLE never exits its host. A lost GPU stays lost for the rest of the process.
- Running out of GPU memory returns `BEAGLE_ERROR_OUT_OF_MEMORY` (TODO.md plan step M1), on both vendors, with the sizes on
  stderr (`out of GPU memory: ...`). That covers an instance whose buffers or programs do not fit the VRAM pool, and a pool
  (`BEAGLE_NV_DATA_MB`, `BEAGLE_AMD_DATA_MB`; by default half the VRAM) larger than the GPU can hold. The pool is shared by
  the process's instances, and what an instance frees goes back to it (plan step C14): its buffers, and on NV its programs.
  On AMD, each kernel variant's programs stay until exit, for the instances after it.
- The same holds on the AMD card, where the process also takes tinygrad's `$TMPDIR/am_usb4.lock`: every instance shares
  one boot (a partial boot takes about 1 s), each kernel variant is loaded once, and a normal exit finalizes the card.
  Its errors come back as BEAGLE errors too: a failed boot (a card that needs a mode1 reset, say), a kernel the HSACOs
  lack, a VRAM pool too small, a GPU fault or hang, or a lost TinyGPU.app connection. A lost GPU stays lost for the rest
  of the process, whose exit still finalizes the card, or has the crash guard hold it. After a hold (a boot that failed
  with a queue it could not see off), `beagleCreateInstance` fails at once for the rest of the process.

## The crash guard, and when to power-cycle

From before the plugin's first request to the GPU, `beagle-tinygpu-guard` (its own session, so a terminal's Ctrl-C does not
reach it) holds copies of the connection. If the host dies, or the plugin loses the GPU, the guard decides what is safe:

- nothing started yet (before the firmware booted): it closes;
- the GPU idle, or catching up: it waits for BEAGLE's work, unloads the GPU and runs the teardown, then closes;
- a frame cut mid-send, a boot cut short (the plugin killed in it, or GSP-RM never ready), a hang, or an unload the GPU
  does not confirm: it **holds**, sending nothing, because closing could unmap memory the GPU still uses. It says so on
  stderr and in the log: `holding the TinyGPU.app connection (...). Unplug the eGPU first, then kill <pid>.`

On the AMD card the same guard decides from the queues instead (plan step A2k):

- no queue set up yet (in the boot, or just after it): it closes, after finalizing a card whose boot finished, so that the
  next boot is a partial one;
- queues live: it finalizes the card (the compute queues dequeued, SDMA disabled), then closes if it saw every queue off;
- a request cut mid-send or its reply unread, the plugin's own finalize cut short, or a queue still active after its
  dequeue: it **holds**, as above. The plugin's own finalize holds too if it does not see every queue off.

A boot that fails once GSP-RM is up (an RM call refused, or a VRAM pool too big for `BEAGLE_NV_DATA_MB`) is torn down by
BEAGLE itself, as at exit, so it needs no power cycle.

Power-cycle the eGPU (unplug it, then plug it in again) only then, and always before killing a holding guard. A GPU left
warm by an earlier process that was not torn down (WPR2 still up) is refused with nothing written:
`WARM GPU: WPR2 is up ... Power-cycle the eGPU`. With `BEAGLE_NV_RECOVER=1` (plan step P4, opt-in, Ada only, not yet run on
a GPU), a warm GPU whose GSP-RM was unloaded, so that only the teardown is missing (after `BEAGLE_NV_TEARDOWN=0`, or a
teardown that failed after the unload), is torn down at boot instead and then boots. Only reads come before that decision.
NVIDIA's driver refuses such a GPU instead.

## Environment variables

For users:

| Variable | Effect |
|---|---|
| `BEAGLE_NV_DATA_MB` | the VRAM pool, in MiB (default: half the VRAM) |
| `BEAGLE_TINYGPU_FW` | the firmware directory (see Requirements) |
| `BEAGLE_TINYGPU_NO_DOWNLOAD=1` | no firmware download: a missing file is an error that says how to fetch it |
| `BEAGLE_TINYGPU_FW_BASE_URL` | a mirror of linux-firmware's tree to download from, instead of the pinned URL |
| `BEAGLE_TINYGPU_DATA` | where the VBIOS capture and other diagnostics go (default `~/.beagle/tinygpu`) |
| `BEAGLE_TINYGPU_LOG` | the backend's log (default `~/Library/Logs/beagle_tinygpu.log`) |
| `BEAGLE_NV_PROFILE=1` | per-call timings and launch counts on stderr at exit |
| `BEAGLE_NV_TEARDOWN=0` | the GSP unload only, no teardown: the next boot then needs a power cycle (for diagnosis) |
| `BEAGLE_NV_UNLOAD_LEVEL=0` | the LEVEL_0 unload instead of FAST_UNLOAD (a fallback) |
| `BEAGLE_NV_RECOVER=1` | a warm Ada GPU whose GSP is suspended or halted is torn down at boot instead of refused (experimental) |
| `BEAGLE_NV_GUARD` | the crash guard's path (default: next to the plugin); `BEAGLE_AMD_GUARD` for the AMD C++ boot |
| `APL_REMOTE_SOCK` | TinyGPU.app's socket (default `$TMPDIR/tinygpu.sock`, as tinygrad) |
| `BEAGLE_AMD_DATA_MB` | AMD: the VRAM pool, in MiB (default: half the VRAM) |
| `BEAGLE_AMD_PROFILE=1` | AMD: each operation's time on stderr |

The test harness's own, not for production: `BEAGLE_NV_TEST_KILL`, `BEAGLE_AMD_TEST_KILL`, `BEAGLE_TG_MARKERS`,
`BEAGLE_TINYGPU_APP`, `BEAGLE_TINYGPU_NO_LAUNCH` and `BEAGLE_NV_FILL_LAUNCH_DIMS`. tinygrad's `AM_RESET` and `AM_POWER_LIMIT`
act on the AMD boot as on tinygrad's (an `AM_RESET=1` on a warm card is a mode1 reset, which the boot refuses).

## How it works

| File | Part |
|---|---|
| `GPUInterfaceTinyGPUHybrid.cpp` | the `GPUInterface` BeagleGPUImpl calls; the PCI probe; the NV or AMD branch |
| `GPUInterfaceTinyGPUHybridNV.cpp` | the NV entry points: setup, the shared boot, allocation, copies, launches, fini, the lost-GPU state |
| `TinyGPUTransport.h` | tinygrad's TinyGPU.app client (`RemotePCIDevice`), with the app check and the lock |
| `TinyGPUHybridNVBoot.h` | `NVDev.__init__`'s software half: early init, the VBIOS and FWSEC, the booters, the GSP image, the WPR meta, COT's FMC |
| `TinyGPUHybridNVFalcon.h`, `TinyGPUHybridNVGsp.h` | the falcons' and GSP-RM's boot and unload, the RPC queue, NVIDIA's teardown |
| `TinyGPUMemory.h`, `TinyGPUHybridNVMemory.h` | tinygrad's memory manager and page tables (MMU v2 and v3) |
| `TinyGPUHybridNVRM.h`, `TinyGPUHybridNVDevice.h` | tinygrad's RM client and `NVDevice` (channels, the golden image) |
| `TinyGPUHybridNVProgram.h`, `TinyGPUHybridNVDispatch.h` | the program loader, QMDs, pushbuffers and the timeline |
| `TinyGPUFirmware.h`, `TinyGPUFirmwareManifest.h` | the firmware locator and the manifest it checks |
| `tinygpu_guard.cpp`, `TinyGPUHybridNVGuard.h` | the crash guard |
| `kernels/make_tinygpu_kernels.sh`, `kernels/make_tinygpu_cubins.sh` | the PTX and the embedded cubins, at build time |
| `GPUInterfaceTinyGPUHybridAMD.cpp` | the AMD entry points: the boot and the crash guard's setup, allocation, copies, launches, fini |
| `TinyGPUHybridAMDRuntime.h` | AMD's C++ runtime on the boot's handoff: queues, doorbells, waits, the IH drain, the pool, the programs |
| `TinyGPUHybridAMDDispatch.h`, `TinyGPUHybridAMDProgram.h`, `TinyGPUAMDTables.h` | tinygrad's AMD PM4 and SDMA queues, its HSACO loader and scratch sizing, and their constants (generated by `make_tinygpu_amd_tables.py`) |
| `TinyGPUElf.h` | tinygrad's ELF loader, for cubins and HSACOs |
| `tinygpu_amd_compile.cpp`, `kernels/make_tinygpu_hsaco.sh` | tinygrad's compile_hip in C++ (comgr), and the embedded HSACOs, at build time |
| `TinyGPUHybridAMDBoot.h`, `TinyGPUAMDReg.h` | tinygrad's AM driver in C++ (AMDev, AMFirmware, its page tables and memory manager, the PSP, SMU, GMC, IH, GFX and SDMA blocks), and its registers |
| `TinyGPUHybridAMDDevice.h` | AMDDevice.__init__'s queues and buffers and the old daemon's handoff, in C++, for the runtime |
| `TinyGPUAMDBootTables.h` | the boot's registers, structs, constants and firmware manifest, generated by `make_tinygpu_amd_boot_tables.py` |

The Python this port was checked against, tinygrad plus BEAGLE's patches, lives in `tinygpu_tests/oracle/`, off the run
path.

## Provenance

- tinygrad at commit `a9830e2b4` (the `hcq1` tree), MIT: the boots, the memory manager, the RM client, `NVDevice`, the AM
  driver, `AMDDevice`, the program loaders and the TinyGPU.app client, ported statement by statement.
- NVIDIA's open-gpu-kernel-modules 570.144, MIT: the driver-unload teardown (FWSEC-SB, Booter Unload) and the RISC-V halt
  wait; nouveau as a cross-check only.
- NVIDIA's firmware, under NVIDIA's licence (`LICENCE.nvidia` in linux-firmware); fetched by the user, never bundled.

## Tests

`libhmsbeagle/GPU/tinygpu_tests/README.md`: `run_offline.sh` runs everything that needs no eGPU (the goldens, which compare
each C++ port with the tinygrad code it follows; the fake devices; the replays of recorded hardware sessions), and
`run_point.sh` is one run on the real eGPU.
