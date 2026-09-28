# TinyGPU-Hybrid: BEAGLE on an eGPU through TinyGPU.app

The TinyGPU-Hybrid backend runs BEAGLE's GPU kernels on an NVIDIA GPU in a USB4 or Thunderbolt enclosure attached to a Mac,
where there is no CUDA driver. It talks to the GPU through [TinyGPU.app](https://github.com/tinygrad/tinygpu_releases),
tinygrad's user-space PCIe server, and follows tinygrad's own NV driver (`tinygrad/runtime/ops_nv.py` and
`support/nv/`), ported to C++: BEAGLE boots the GPU's GSP-RM firmware itself, builds tinygrad's `NVDevice`, loads
ahead-of-time cubins and submits BEAGLE's kernels, with no Python at run time. On exit it unloads the GPU and runs
NVIDIA's driver-unload teardown, so the next process boots it again without a power cycle.

The AMD side of the backend (a Radeon through the same app) still runs tinygrad's Python (`amd_dispatch_daemon.py`); its
C++ port is future work (TODO.md plan steps A0-A2).

## Supported GPUs

| Family | PCI device IDs | Status |
|---|---|---|
| Ada (AD10x) | `0x26xx`-`0x28xx` | tested on an RTX 4060 (AD107) |
| Blackwell (GB20x) | `0x2bxx`-`0x2dxx`, `0x2fxx` | tested on an RTX 5070 (GB205); the other GB20x boot the same way, untested |
| Ampere (GA10x) | `0x22xx`-`0x25xx` | not supported: refused at `beagleCreateInstance` (its path, the Python daemon, was removed) |

Only single precision is built: 9 cubins per architecture (sm_86, sm_89, sm_120), for padded state counts 4, 16, 32, 48, 64,
80, 128, 192 and 256. A double-precision instance, or a GPU whose architecture has no cubin, is refused at
`beagleCreateInstance`.

## Requirements

- macOS on Apple silicon, with the eGPU attached.
- TinyGPU.app release `c0d024f9`, installed in `/Applications` and its system extension approved. BEAGLE checks both
  binaries' sha256 and refuses any other release, with the install instructions. BEAGLE never installs the app.
- NVIDIA's firmware, from linux-firmware at tinygrad's pin (GSP-RM 570.144, the booters, and on GB20x the FMC):
  `libhmsbeagle/GPU/tinygpu_fetch_firmware.sh [--chip ad102|gb202] DIR`, then `BEAGLE_TINYGPU_FW=DIR`. BEAGLE also looks
  in an installed `share/beagle/firmware` and in tinygrad's download cache (`~/Library/Caches/tinygrad/downloads/fw`). It
  never downloads anything itself, and checks every file's sha256 against `TinyGPUFirmwareManifest.h`.
- To build: `nvcc` and `ptxas` from CUDA 12.8, for the generated kernels header and the embedded cubins (on a Mac, through
  Docker; `-DTINYGPU_NVCC=` and `-DTINYGPU_PTXAS=` name them). Nothing is compiled at run time.

## Building and installing

`BUILD_TINYGPU_HYBRID` (on by default) builds the plugin, `hmsbeagle-tinygpu-hybrid`, and the crash guard,
`beagle-tinygpu-guard`. Both are installed to the same directory: the plugin looks for the guard next to itself (or at
`BEAGLE_NV_GUARD`), and refuses to boot without it. The resource appears in `beagleGetResourceList` as
`TinyGPU-NV-Hybrid`, with `BEAGLE_FLAG_FRAMEWORK_TINYGPU`.

## Running

- One process at a time uses the eGPU: the first takes `$TMPDIR/nv_usb4.lock`, and a second fails at once. Every BEAGLE
  instance in that process shares one boot (about 2 s), which lasts until the process exits.
- Keep the Mac awake while the GPU runs (`caffeinate -ims`): a sleeping Mac with a live GPU risks the IOMMU (DART).
- A normal exit tears the GPU down (the GSP unload, then NVIDIA's teardown), and says so on stderr:
  `teardown: done: ... WPR2 is down, the next boot needs no power cycle`.
- Errors come back as BEAGLE errors. A failed boot, a GPU that hangs, a lost TinyGPU.app connection, or a VRAM pool too
  small for an instance makes `beagleCreateInstance`, or the calls that read results back, return
  `BEAGLE_ERROR_GENERAL`, and read-backs are NaN. BEAGLE never exits its host. A lost GPU stays lost for the rest of the
  process.

## The crash guard, and when to power-cycle

From before the plugin's first request to the GPU, `beagle-tinygpu-guard` (its own session, so a terminal's Ctrl-C does not
reach it) holds copies of the connection. If the host dies, or the plugin loses the GPU, the guard decides what is safe:

- nothing started yet (before the firmware booted): it closes;
- the GPU idle, or catching up: it waits for BEAGLE's work, unloads the GPU and runs the teardown, then closes;
- a frame cut mid-send, a boot cut short (the plugin killed in it, or GSP-RM never ready), a hang, or an unload the GPU
  does not confirm: it **holds**, sending nothing, because closing could unmap memory the GPU still uses. It says so on
  stderr and in the log: `holding the TinyGPU.app connection (...). Unplug the eGPU first, then kill <pid>.`

A boot that fails once GSP-RM is up (an RM call refused, or a VRAM pool too big for `BEAGLE_NV_DATA_MB`) is torn down by
BEAGLE itself, as at exit, so it needs no power cycle.

Power-cycle the eGPU (unplug it, then plug it in again) only then, and always before killing a holding guard. A GPU left
warm by an earlier process that was not torn down (WPR2 still up) is refused with nothing written:
`WARM GPU: WPR2 is up ... Power-cycle the eGPU`.

## Environment variables

For users:

| Variable | Effect |
|---|---|
| `BEAGLE_NV_DATA_MB` | the VRAM pool, in MiB (default: half the VRAM) |
| `BEAGLE_TINYGPU_FW` | the firmware directory (see Requirements) |
| `BEAGLE_TINYGPU_DATA` | where the VBIOS capture and other diagnostics go (default `~/.beagle/tinygpu`) |
| `BEAGLE_TINYGPU_LOG` | the backend's log (default `~/Library/Logs/beagle_tinygpu.log`) |
| `BEAGLE_NV_PROFILE=1` | per-call timings and launch counts on stderr at exit |
| `BEAGLE_NV_TEARDOWN=0` | the GSP unload only, no teardown: the next boot then needs a power cycle (for diagnosis) |
| `BEAGLE_NV_UNLOAD_LEVEL=0` | the LEVEL_0 unload instead of FAST_UNLOAD (a fallback) |
| `BEAGLE_NV_GUARD` | the crash guard's path (default: next to the plugin) |
| `APL_REMOTE_SOCK` | TinyGPU.app's socket (default `$TMPDIR/tinygpu.sock`, as tinygrad) |

The test harness's own, not for production: `BEAGLE_NV_TEST_KILL`, `BEAGLE_TG_MARKERS`, `BEAGLE_TINYGPU_APP`,
`BEAGLE_TINYGPU_NO_LAUNCH` and `BEAGLE_NV_FILL_LAUNCH_DIMS`. The AMD side finds its Python daemon through `BEAGLE_PYTHON`,
`BEAGLE_NV_SCRIPTS` and `BEAGLE_AMD_DISPATCH_DAEMON`.

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

The Python this port was checked against, tinygrad plus BEAGLE's patches, lives in `tinygpu_tests/oracle/`, off the run
path.

## Provenance

- tinygrad at commit `a9830e2b4` (the `hcq1` tree), MIT: the boot, the memory manager, the RM client, `NVDevice`, the
  program loader and the TinyGPU.app client, ported statement by statement.
- NVIDIA's open-gpu-kernel-modules 570.144, MIT: the driver-unload teardown (FWSEC-SB, Booter Unload) and the RISC-V halt
  wait; nouveau as a cross-check only.
- NVIDIA's firmware, under NVIDIA's licence (`LICENCE.nvidia` in linux-firmware); fetched by the user, never bundled.

## Tests

`libhmsbeagle/GPU/tinygpu_tests/README.md`: `run_offline.sh` runs everything that needs no eGPU (the goldens, which compare
each C++ port with the tinygrad code it follows; the fake devices; the replays of recorded hardware sessions), and
`run_point.sh` is one run on the real eGPU.
