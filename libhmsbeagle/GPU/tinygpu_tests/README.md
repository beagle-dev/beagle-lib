# TinyGPU offline test harness

Tests for BEAGLE's TinyGPU-Hybrid NV backend that need no eGPU. Each C++ port of tinygrad code is compared byte for
byte with the tinygrad function it follows (the pinned hcq1 tree plus BEAGLE's daemon patches are the oracle), and
the real plugin runs end to end against a fake daemon and a fake TinyGPU.app. Not built or installed by default.
Background and plan: TODO.md `## Runtime roadmap` and `## Plan (2026-09-24)`; STATUS.md R10-R14.

## Requirements

- A build of `hmsbeagle-tinygpu-hybrid` and `tinygpuhybridtest` (default `../../../build`, or `BEAGLE_BUILD`);
  the build also generates `../kernels/BeagleTinyGPU_kernels.h`, which the goldens read.
- The pinned tinygrad at `TINYGRAD_PATH` (default `~/Dropbox/Projects/tinygrad-hcq1`, commit a9830e2b4).
- A Python with tinygrad's dependencies at `BEAGLE_PYTHON` (default `~/Dropbox/Projects/tinygrad/venv/bin/python`).
- `clang++` (or `CXX`).
- Cached cubins in `$BEAGLE_TINYGPU_DATA/cubins` (default `~/.beagle/tinygpu`). A missing cubin is compiled once
  with `nv_compile_helper.compile_ptx`, which runs ptxas (through Docker on this Mac).

## Running

```
./run_offline.sh          # everything below, with a PASS/FAIL summary
./run_goldens.sh          # the goldens and the daemon wire test only
./run_fake_runtime.sh runtime BEAGLE_NV_USE_DAEMON=0 -- --state-count 64 --reps 5
```

Outputs go to `.work/` (git-ignored; `TINYGPU_TEST_WORK` overrides).

| File | What it checks |
|---|---|
| `golden_program.py/.cpp` | `TinyGPUHybridNVProgram.h` (ELF loader, program records, relocated image) against real `BeagleNVProgram` objects, for SP_4/32/64/128 on sm_89 and sm_120 |
| `golden_runtime.py/.cpp` | the C++ runtime pieces: boot-only handoff parsing, table cross-check, local-memory sizing and setup words (`_ensure_has_local_memory`), pool placement (`PCIIfaceBase.alloc` + `alloc_vaddr`) |
| `golden_encode.py/.cpp` | `TinyGPUHybridNVDispatch.h` launch and copy encoding against hcq1's `NVComputeQueue`/`NVCopyQueue` |
| `check_firmware.py` | `booter_unload` for ad102/ga102 is in tinygrad's download cache (re-staged from `~/.beagle/tinygpu/fw` if purged) and tinygrad's `fetch_fw` returns it with the network off |
| `test_daemon_wire.py` | `nv_dispatch_daemon.py`'s framing and both `launch_batch` paths on a fake device |
| `run_fake_runtime.sh` | one plugin run against `fake_nv_daemon.py` and `fake_tinygpu_server.py`, which runs the pushbuffers like a GPU front end (acquires, QMD chains, releases, DMA) and, in the C++ runtime mode, checks every QMD with tinygrad's reader. PASS requires every stage of the selected mode (boot, compile, launches, evaluations, and the handoff or C++ program loading) and a clean fake server. Kernels are not emulated, so logL is wrong by design |
| `mm_trace.py` | tinygrad's memory manager on a recording fake for BEAGLE's allocation sequence (`--mmu 2` Ada, `--mmu 3` Blackwell): the reference for porting memory management |
| `cubin_inspect.py` | ELF facts of the cached cubins (SM in e_flags, relocations, register-count records) |
| `amd_compile_probe.py` | compiles the AMD kernels for gfx1100 with tinygrad's `compile_hip` (native comgr) and reports size, determinism, kernel descriptors |
| `run_point.sh` | **hardware**: one real run (see below) |

## Safety

- The plugin starts the real TinyGPU.app whenever it cannot connect to its socket. The offline scripts
  (`run_fake_runtime.sh`, `run_offline.sh`) refuse to run unless the built plugin contains the
  `BEAGLE_TINYGPU_NO_LAUNCH` guard, set `BEAGLE_TINYGPU_NO_LAUNCH=1` so the plugin fails instead, and start the
  test only after the fake prints `listening` on a short per-run socket path (macOS limits socket paths to 104
  bytes). `run_offline.sh` also checks at run time that the guard holds. `run_point.sh` is the hardware exception.
- Tinygrad-side tests must never call tinygrad's `ensure_app` or open a real `APLRemotePCIDevice`; build fakes
  instead (`RemotePCIDevice` via `object.__new__` on a socketpair).
- `run_point.sh` boots the real eGPU: power-cycle it (unplug and replug) first, never Ctrl-C or kill a run, and
  unplug a hung GPU before killing anything. Its outputs go to `$BEAGLE_TINYGPU_DATA/runs/`.

## Data outside the repo (`~/.beagle/tinygpu/`)

- `cubins/`: ptxas cubins keyed by the PTX's sha256 prefix and architecture.
- `fw/`: `booter_unload-570.144.bin` for ad102 and ga102 (also staged in tinygrad's download cache, so `fetch_fw`
  finds them offline).
- `refs/`: NVIDIA, nouveau, Apple and other sources used by the 2026-09-24 research, with `REFERENCES.txt` (source
  URL and sha256 per file). Some are GPL or APSL; do not copy them into BEAGLE.
- `planning/`: the evidence behind the plan (`inventory.json`, `powercycle.json`, `final_plan.md`, ...).
- `runs/`: hardware run outputs and daemon logs.
