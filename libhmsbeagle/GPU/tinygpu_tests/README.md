# TinyGPU offline test harness

Tests for BEAGLE's TinyGPU-Hybrid NV backend that need no eGPU. Each C++ port of tinygrad code is compared byte for
byte with the tinygrad function it follows (the pinned hcq1 tree plus BEAGLE's daemon patches are the oracle), and
the real plugin runs end to end against a fake daemon and a fake TinyGPU.app. Not built or installed by default.
Background and plan: TODO.md `## Runtime roadmap` and `## Plan (2026-09-24)`; STATUS.md R10-R18.

## Requirements

- A build of `hmsbeagle-tinygpu-hybrid`, `tinygpuhybridtest`, `synthetictest` and `hmctest` (default `../../../build`, or
  `BEAGLE_BUILD`; `d1_refs.sh` also needs `hmsbeagle-opencl`);
  the build also generates `../kernels/BeagleTinyGPU_kernels.h`, which the goldens read, and the plugin's embedded
  cubins (`../kernels/BeagleTinyGPU_cubins.S` and their table `../kernels/TinyGPUNVCubins.h`, plan step C1), which
  test_c1_cubins links.
- The pinned tinygrad at `TINYGRAD_PATH` (default `~/Dropbox/Projects/tinygrad-hcq1`, commit a9830e2b4).
- A Python with tinygrad's dependencies at `BEAGLE_PYTHON` (default `~/Dropbox/Projects/tinygrad/venv/bin/python`).
- `clang++` (or `CXX`).
- Cached cubins in `$BEAGLE_TINYGPU_DATA/cubins` (default `~/.beagle/tinygpu`): the reference for the 27 the plugin
  embeds. A missing cubin is compiled once with `nv_compile_helper.compile_ptx`, which runs ptxas (through Docker on
  this Mac, about 0.6 s each).

## Running

```
./run_offline.sh          # everything below, with a PASS/FAIL summary
./run_goldens.sh          # the goldens and the daemon wire test only
./run_fake_runtime.sh runtime BEAGLE_NV_USE_DAEMON=0 -- --state-count 64 --reps 5
```

Outputs go to `~/Library/Caches/beagle-tinygpu-tests/` on each computer (`TINYGPU_TEST_WORK` overrides; the old in-repo
`.work/` stays git-ignored).

| File | What it checks |
|---|---|
| `golden_program.py/.cpp` | `TinyGPUHybridNVProgram.h` (ELF loader, program records, relocated image) against real `BeagleNVProgram` objects, for the 9 SP modules on sm_86, sm_89 and sm_120 (the 27 cubins the plugin embeds) |
| `test_c1_cubins.py` + `golden_cubins.cpp` | plan step C1: the embedded cubins, linked from the generated `.S` as the plugin links them: the table is the 9 SP modules × sm_86, sm_89, sm_120, each cubin byte-identical to `nv_compile_helper.compile_ptx` of the plugin's PTX (the daemon's compile_all), with kernel names equal to nv_compile_helper's; `TinyGPUHybridNVCubins.h`'s selection of every entry and its refusals (DP, a state count, an architecture, an entry holding another architecture's cubin); `nvd_elf_sm` against the cached cubins' file names; the build's ptxas is compile_all's; the real daemon's C++ runtime handoff right after boot (elf_size 0, no ELF); the real `cmd_boot` never selects tinygrad's renderer (on macOS that starts tinygrad's Docker compile server) and `cmd_compile_all` checks it, refusing NAK |
| `check_upload.py` | after a C++ runtime fake run (`run_offline.sh`, 4 and 64 states): the plugin loaded the module the harness ran (state count and architecture), and the program image in the fake VRAM equals `BeagleNVProgram`'s relocation of the compile_ptx cubin of that module's PTX |
| `golden_runtime.py/.cpp` | the C++ runtime pieces: boot-only handoff parsing, table cross-check, local-memory sizing and setup words (`_ensure_has_local_memory`), pool placement (`PCIIfaceBase.alloc` + `alloc_vaddr`) |
| `golden_encode.py/.cpp` | `TinyGPUHybridNVDispatch.h` launch and copy encoding against hcq1's `NVComputeQueue`/`NVCopyQueue` |
| `check_firmware.py` | `booter_unload` for ad102/ga102, the GB20x (Blackwell) boot's `fmc`, `gsp` and `bootloader`, and ad102's `booter_load` (test_p2) are in tinygrad's download cache (re-staged from `~/.beagle/tinygpu/fw` if purged) and tinygrad's `fetch_fw` returns them with the network off, so no boot downloads inside the daemon |
| `test_daemon_wire.py` | `nv_dispatch_daemon.py`'s framing and both `launch_batch` paths on a fake device |
| `test_p1_diagnostics.py` | plan step P1: the warm-GPU refusal end to end over a scripted fake TinyGPU socket, and each diagnostic wrapper in `nv_init_helper.py` on fakes |
| `test_p2_teardown.py` | plan step P2 (the teardown, on by default since P3): the FWSEC-SB and Booter Unload images against tinygrad's own `prep_ucode`/`prep_booter` on the captured VBIOS, the VRAM layout gate (`mm_trace.trace`), the teardown on a scripted register fake (one scenario per failure mode, with the `execute_hs` arguments of both images), LEVEL_0 with an op-8 sequencer, and the daemon's fini/hung/failed-boot decisions |
| `test_p3.py` | plan step P3: the teardown default (on unless `BEAGLE_NV_TEARDOWN=0`), the daemon half of the C++ state page, the daemon's fini and EOF decisions (no device, no page, idle, frame in flight, timeline behind, stuck or faulted, tinygrad's error_state, cut messages, lost replies), and cmd_handoff's WPR check on tinygrad's allocator |
| `test_b1_cot.py` | plan step B1 (Blackwell, the COT boot): what section 6 of `nv_init_helper.py` relies on in the pinned tinygrad; the registers it reads (no include collision); the RISC-V halt wait after the unload (halted, WPR2 up, never, a PRI error, unload not confirmed, BEAGLE_NV_TEARDOWN=0; reads only); the unload-then-halt order and the hung path; the flag the COT message sets and the FSP queue reads; the failed-boot unload (before the message, no status queue, halt or none); the sequencer guard; the four scripts' COT refusal; the BAR check; the layout gate at 12227 MiB; the daemon's hold on a missing halt (fini, EOF, failed boot); and tinygrad's real boot path for a GB205 over a scripted fake socket (the restored FSP readiness wait in tinygrad's order, a 512 MiB BAR1 refused before boot memory, an FSP that never gets ready), plus AD107's stream unchanged |
| `run_fake_runtime.sh` | one plugin run against `fake_nv_daemon.py` and `fake_tinygpu_server.py`, which runs the pushbuffers like a GPU front end (acquires, QMD chains, releases, DMA) and, in the C++ runtime mode, checks every QMD with tinygrad's reader. PASS requires every stage of the selected mode (boot; compile_all, or in the C++ runtime the embedded cubin and no compile_all, which the fake daemon also refuses; launches, evaluations, and the handoff or C++ program loading) and a clean fake server. Kernels are not emulated, so logL is wrong by design. With `FAKE_NV_HANG=1` the fake GPU never writes a release, which `run_offline.sh` uses to check the hung path. In the C++ modes PASS also needs the daemon's state-page line at fini (nothing in flight; the last value submitted equals the C++ timeline and the fake GPU's release count), and the fake GPU flags any TinyGPU.app command from the plugin after the daemon's unload (the fake daemon sends it none; the daemon's own ordering is test_p3's). With `FAKE_SIGINT_AFTER=<regex>` the test gets one SIGINT 2 s after its output matches. `FAKE_NV_ARCH` changes the fake boot reply's architecture (`run_offline.sh`'s refusal run). `FAKE_NV_CHIP=gb205` plays an RTX 5070 (plan step B1): the probe id 10de:2f04, arch sm_120, Blackwell's compute class (QMD v5, checked by tinygrad's v5 reader), its runtime keys and tokens, and a COT unload reported by `nv_init_helper`'s real wrappers over scripted registers; with `FAKE_NV_NO_HALT=1` the RISC-V core never halts, so the daemon holds. `FAKE_TEST_BIN=<example>` runs another BEAGLE example with its own arguments and does not require tinygpuhybridtest's timed evaluations (plan step D1) |
| `mm_trace.py` | tinygrad's memory manager on a recording fake for BEAGLE's allocation sequence (`--mmu 2` Ada, `--mmu 3` Blackwell; `--vram-mb` sets the VRAM size): the reference for porting memory management, P2's layout gate, and B1's at 12227 MiB |
| `cubin_inspect.py` | ELF facts of the cached cubins (SM in e_flags, relocations, register-count records) |
| `amd_compile_probe.py` | compiles the AMD kernels for gfx1100 with tinygrad's `compile_hip` (native comgr) and reports size, determinism, kernel descriptors |
| `d1_runs.txt` | plan step D1: the synthetictest and hmctest runs taken to the RTX 4060, each with the kernels it launches; `run_offline.sh` pins every line's kernel set on the fakes (`d1_verdict` in `env.sh`, which reads the plugin's `BEAGLE_NV_PROFILE` kernel list) |
| `d1_refs.sh` | plan step D1: every `d1_runs.txt` line's references, made offline with no TinyGPU plugin on the library path: synthetictest on the CPU in single and double precision, hmctest `--tinygpu` on the CPU and on the Mac's OpenCL GPU; under `$BEAGLE_TINYGPU_DATA/d1/refs/` |
| `d1_compare.py` | plan step D1: a run's stdout against its references (tolerances in its docstring); reads files only |
| `run_d1.sh` | **hardware**: one `d1_runs.txt` line on the real eGPU in the C++ runtime, with `run_point.sh`'s protections, stdout and stderr apart: `run_d1.sh <label>`; exits 0 only if `run_point.sh` would and the run launched exactly the line's kernels and matches its references; 3 is a clean run that failed that check. A line runs again only after a power cycle (`D1_REPLUGGED=1`), since a rerun would find its own results in VRAM. `run_point.sh` and `run_d1.sh` share a lock that is per computer (`$TMPDIR/beagle_tinygpu_hw.lock`), tag their `runs/` files with the computer's name (`HW_HOST`; `run_d1.sh`'s rerun check sees only this computer's runs), refuse exported variables that change what the GPU sees, keep `log stream` in its own process group, and keep the Mac awake while a daemon holds (`hw_*` in `env.sh`) |
| `run_point.sh` | **hardware**: one real run (see below): `run_point.sh <N> [cpp\|daemon\|runtime] [reps] [--poison]`; exits 0 only if the test passed, the fini report says the next boot needs no power cycle (`fini_verdict` in `env.sh`), the daemon exited and `log stream` saw nothing from the eGPU |

## Safety

- The plugin starts the real TinyGPU.app whenever it cannot connect to its socket. The offline scripts
  (`run_fake_runtime.sh`, `run_offline.sh`) refuse to run unless the built plugin contains the
  `BEAGLE_TINYGPU_NO_LAUNCH` guard, set `BEAGLE_TINYGPU_NO_LAUNCH=1` so the plugin fails instead, and start the
  test only after the fake prints `listening` on a short per-run socket path (macOS limits socket paths to 104
  bytes). `run_offline.sh` also checks at run time that the guard holds. `run_point.sh` and `run_d1.sh` are the hardware exceptions.
- Tinygrad-side tests must never call tinygrad's `ensure_app` or open a real `APLRemotePCIDevice`; build fakes
  instead (`RemotePCIDevice` via `object.__new__` on a socketpair).
- `run_point.sh`, `run_d1.sh` (and `../nv_teardown_diag.py`) boot the real eGPU. It must be cold (power-cycled) or torn down by
  the previous run: the teardown is on by default (plan step P3; `BEAGLE_NV_TEARDOWN=0` turns it off, STATUS.md R18),
  and `run_point.sh` exits nonzero at the first bad fini report. Never Ctrl-C or kill a run, and unplug a hung or
  holding GPU before killing anything. Outputs, daemon logs and a `log stream` capture go to `$BEAGLE_TINYGPU_DATA/runs/`.

## Data outside the repo (`~/.beagle/tinygpu/`)

- `cubins/`: ptxas cubins keyed by the PTX's sha256 prefix and architecture.
- `fw/`: `booter_unload-570.144.bin` for ad102 and ga102, the GB20x boot's `fmc`/`gsp`/`bootloader` 570.144 files and ad102's
  `booter_load` (also staged in tinygrad's download cache, so `fetch_fw` finds them offline; `check_firmware.py` re-stages them).
- `refs/`: NVIDIA, nouveau, Apple and other sources used by the 2026-09-24 research, with `REFERENCES.txt` (source
  URL and sha256 per file). Some are GPL or APSL; do not copy them into BEAGLE.
- `planning/`: the evidence behind the plan (`inventory.json`, `powercycle.json`, `final_plan.md`, ...).
- `runs/`: hardware run outputs and daemon logs, named `<date>-<time>_<computer>_...` (older runs have no computer tag).
  `~/.beagle/tinygpu` may be a synced folder shared between computers, each with its own eGPU; nothing in it is a lock.
