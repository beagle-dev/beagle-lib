# Building BEAGLE with eGPU support (TinyGPU) on a clean Mac

This guide sets up a new Mac to build and run BEAGLE's TinyGPU backend: BEAGLE on an NVIDIA or AMD GPU in a
Thunderbolt or USB4 enclosure, through TinyGPU.app. [TinyGPU.md](TinyGPU.md) describes the backend itself:
the supported GPUs, running, the crash guard, when to power-cycle, and the environment variables.

The build makes two files that are installed side by side: the plugin, `hmsbeagle-tinygpu`, and its crash guard,
`beagle-tinygpu-guard`. The GPU kernels are compiled at build time and embedded in the plugin, so nothing is compiled
when BEAGLE runs:

- **NVIDIA:** `nvcc` and `ptxas` from CUDA 12.8 build the PTX and 54 cubins (single and double precision, 9 state counts,
  for sm_86, sm_89 and sm_120). NVIDIA ships no CUDA for macOS, so both run in a Linux container under Docker.
- **AMD:** tinygrad's build of AMD's Code Object Manager (comgr) compiles 18 HSACOs for gfx1100 (the RX 7900 series).
  Without comgr the build still succeeds, but the plugin has no AMD kernels and refuses AMD cards.

Running needs TinyGPU.app and the GPUs' firmware, but not Docker or comgr.

BEAGLE was built this way on a Mac Studio (Apple silicon) with macOS 26.5.1, Apple clang 21.0.0, CMake 4.2.3, Docker
Desktop with `nvidia/cuda:12.8.1-devel-ubuntu22.04`, comgr v7.2.0 and TinyGPU.app release `c0d024f9`. The backend has run
on an RTX 4060 (AD107), an RTX 5070 (GB205) and an RX 7900 XT (gfx1100).

## 1. Base tools

- macOS on Apple silicon.
- Apple's command line tools: `xcode-select --install`. Full Xcode also works.
- Homebrew, installed as [brew.sh](https://brew.sh) describes. Then:

  ```bash
  brew install cmake zstd
  ```

  comgr (step 3) links against Homebrew's zstd.
- A JDK, for BEAGLE's Java bindings, which are built by default (`BUILD_JNI`). Configuring fails without one: install one
  (for example `brew install --cask zulu`), or pass `-DBUILD_JNI=OFF` in step 6.

## 2. NVIDIA: CUDA 12.8's nvcc and ptxas, through Docker

**Docker.** Install Docker Desktop for Mac (Apple silicon) from
[docker.com](https://www.docker.com/products/docker-desktop/) and start it. It must be running whenever BEAGLE is built.
Then pull CUDA 12.8.1's image (about 8 GB):

```bash
docker pull --platform linux/arm64 nvidia/cuda:12.8.1-devel-ubuntu22.04
```

**The shim.** One script, `~/.local/bin/nvccshim`, runs the tool it was invoked as (`nvcc`, `ptxas` or `nvdisasm`) inside a
persistent container. This command writes it:

```bash
mkdir -p ~/.local/bin && cat > ~/.local/bin/nvccshim <<'EOF'
#!/bin/bash
# Docker shim for CUDA 12.8.1's nvcc, ptxas and nvdisasm on macOS (NVIDIA ships no macOS CUDA).
# ~/.local/bin/{nvcc,ptxas,nvdisasm} are symlinks to this file: it runs the tool it was invoked as
# inside a persistent container that mounts $HOME and /var/folders at the same paths. The tool does
# not see the host's working directory, so pass absolute paths under those two.
# Stop the container with: docker stop cuda-nvcc-persistent
IMAGE=nvidia/cuda:12.8.1-devel-ubuntu22.04
NAME=cuda-nvcc-persistent
running() { [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" = true ]; }
running || docker run -d --rm --init --platform linux/arm64 --name "$NAME" \
        -v "$HOME:$HOME" -v /var/folders:/var/folders "$IMAGE" sleep infinity > /dev/null 2>&1 \
    || running || { echo "nvccshim: cannot start the $NAME container (is Docker Desktop running?)" >&2; exit 125; }
exec docker exec "$NAME" "$(basename "$0")" "$@"
EOF
```

Then make it executable and link the three names to it:

```bash
cd ~/.local/bin && chmod +x nvccshim && ln -sf nvccshim nvcc && ln -sf nvccshim ptxas && ln -sf nvccshim nvdisasm
```

**Check it.** Both commands should report `Cuda compilation tools, release 12.8, V12.8.93`:

```bash
~/.local/bin/nvcc --version
```

```bash
~/.local/bin/ptxas --version
```

The container sees only `$HOME` and `/var/folders`, so **keep BEAGLE's source and build directories under your home
directory**. CMake finds the shims in `~/.local/bin` by itself; `-DTINYGPU_NVCC=` and `-DTINYGPU_PTXAS=` name other copies.
The container keeps running after a build; `docker stop cuda-nvcc-persistent` stops it.

## 3. AMD: comgr

1. Download `libamd_comgr.dylib` from release v7.2.0 at
   [github.com/tinygrad/amdcomgr_dylib/releases](https://github.com/tinygrad/amdcomgr_dylib/releases). That is the release
   BEAGLE is tested with. Put it in `/opt/homebrew/lib`, where CMake looks for it, or pass
   `-DTINYGPU_COMGR=/path/to/libamd_comgr.dylib` in step 6.
2. Check it. The sha256 should be `7712fbe4fcb9fcdea49aeac989876448df975ce0a8ce7c9b15b55c15e7a05935` (111,259,480 bytes):

   ```bash
   shasum -a 256 /opt/homebrew/lib/libamd_comgr.dylib
   ```

tinygrad's own `extra/setup_hipcomgr_osx.sh` installs the latest release instead, which BEAGLE has not been tested with.
Skip this step if you build for NVIDIA only.

## 4. TinyGPU.app (to run)

BEAGLE needs TinyGPU.app release `c0d024f9`. It checks both of the app's binaries and refuses any other release; it never
installs the app itself.

1. Download [TinyGPU.zip](https://github.com/tinygrad/tinygpu_releases/raw/c0d024f9ff0e1dc8fdf217f255da7101d91e8323/TinyGPU.zip).
   Check its sha256: it should be `0c47285e2232643210555cf30ce08289b9e55da261c300e0c82e8448a359a21f`.

   ```bash
   shasum -a 256 ~/Downloads/TinyGPU.zip
   ```

2. Unzip it into `/Applications`, then run its installer:

   ```bash
   unzip ~/Downloads/TinyGPU.zip -d /Applications
   ```

   ```bash
   /Applications/TinyGPU.app/Contents/MacOS/TinyGPU install
   ```

3. Approve its system extension when macOS asks, in System Settings.

## 5. Firmware (to run)

NVIDIA's firmware is GSP-RM 570.144, the booters and, on GB20x, the FMC. AMD's is six gfx1100 blobs. Both come from
linux-firmware at tinygrad's pin, and you need not fetch either by hand: both vendors behave the same way. On first use,
before anything is written to the GPU, BEAGLE downloads any missing file with `/usr/bin/curl` into
`~/Library/Caches/beagle/firmware` and checks every file's sha256. A file that cannot be had stops the boot before it starts.
`BEAGLE_TINYGPU_NO_DOWNLOAD=1` forbids the download.

BEAGLE looks for each file in three places, in order, and uses the first copy whose sha256 matches:

1. `$BEAGLE_TINYGPU_FW/<subdir>/<name>`, if `BEAGLE_TINYGPU_FW` is set
2. `<plugin directory>/../share/beagle/firmware/<subdir>/<name>`, an installed plugin's `share/`
3. BEAGLE's cache, `${XDG_CACHE_HOME:-~/Library/Caches}/beagle/firmware/<subdir>/<name>`, where downloads go

`<subdir>` is `nvidia/<chip>/gsp` or `amdgpu`. tinygrad's download cache (`~/Library/Caches/tinygrad/downloads/fw`) is not
searched, so firmware that tinygrad has already downloaded is not reused: BEAGLE downloads its own copy on first use.

For debugging, or for a Mac without the network, the prefetch script fetches either vendor's firmware into a directory. Run
it from BEAGLE's source tree (step 6) on a Mac with the network, and copy the directory over if needed:

```bash
~/src/beagle-lib/libhmsbeagle/GPU/tinygpu_fetch_firmware.sh --chip ad102 ~/beagle-firmware
```

`--chip` picks the GPU: `ad102` for Ada, `gb202` for Blackwell, `gfx1100` for the RX 7900 series; without it the script
fetches all of them. Then `export BEAGLE_TINYGPU_FW=~/beagle-firmware` points BEAGLE at the directory.

## 6. Get the source, configure and build

The backend is on the `usb2` branch. Clone it under your home directory (see step 2):

```bash
mkdir -p ~/src && cd ~/src && git clone https://github.com/beagle-dev/beagle-lib.git && cd beagle-lib && git checkout usb2
```

Configure. Add `-DBUILD_JNI=OFF` if there is no JDK, and `-DBEAGLE_TINYGPU_STATUS=ON` to see the plugin's status notes
on stderr (its boot, its runtime and its teardown at exit): steps 7 and 8 read them. Without it the plugin prints errors
only:

```bash
cmake -S ~/src/beagle-lib -B ~/src/beagle-build -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

The configure output should include these lines. Without the comgr line the plugin will refuse AMD cards:

```
-- TinyGPU backend enabled (NV and AMD: the C++ boot, no Python)
-- TinyGPU kernels: using nvcc at /Users/<you>/.local/bin/nvcc
-- TinyGPU cubins: using ptxas at /Users/<you>/.local/bin/ptxas
-- TinyGPU AMD HSACOs: using comgr at /opt/homebrew/lib/libamd_comgr.dylib
```

Build, with Docker Desktop running:

```bash
cmake --build ~/src/beagle-build -j 8
```

The first build writes the generated kernels into the source tree, under `libhmsbeagle/GPU/kernels/`: the PTX header,
the 54 cubins and the 18 HSACOs. It then embeds them in the plugin. On Apple silicon the libraries are universal (arm64 and
x86_64). To build only the backend and its test program:

```bash
cmake --build ~/src/beagle-build -j 8 --target hmsbeagle-tinygpu beagle-tinygpu-guard tinygputest
```

Install, if wanted. The plugin and the guard go to `<prefix>/lib`, side by side; the plugin refuses to boot a GPU without
the guard next to it. The default prefix is `/usr/local`, which may need `sudo`, or configure with
`-DCMAKE_INSTALL_PREFIX=...`:

```bash
cmake --install ~/src/beagle-build
```

## 7. Check it on the eGPU

With the eGPU attached and TinyGPU.app installed, run the backend's test program from the build tree. Keep the Mac awake
while the GPU runs (`caffeinate`), and never interrupt a run (TinyGPU.md says why):

```bash
cd ~/src/beagle-lib && B=~/src/beagle-build && caffeinate -ims env DYLD_LIBRARY_PATH="$B/libhmsbeagle/GPU/CMake_TinyGPU:$B/libhmsbeagle/CPU:$B/libhmsbeagle" "$B/examples/tinygputest" --state-count 4 --reps 200 --diag-compare-cpu
```

Expect `PASS` and a GPU logL equal to the CPU reference's. A build with `-DBEAGLE_TINYGPU_STATUS=ON` also reports a clean
teardown at exit: on NVIDIA `teardown: done: ... WPR2 is down, the next boot needs no power cycle`; on AMD the card
finalized. Add `--double` for double precision.

## 8. Optional: the offline tests

The harness in `libhmsbeagle/GPU/tinygpu_tests` checks the backend without a GPU, against tinygrad and register-level fakes
of the GPUs; [its README](tinygpu_tests/README.md) has the details. It runs for about 25 minutes. Besides the build, it
needs:

- a build configured with `-DBEAGLE_TINYGPU_STATUS=ON` (step 6): the harness reads the plugin's status notes, and refuses
  a build without them;
- tinygrad at the pinned commit a9830e2b4 ([github.com/tinygrad/tinygrad](https://github.com/tinygrad/tinygrad)), at
  `TINYGRAD_PATH` (default `~/Dropbox/Projects/tinygrad-hcq1`);
- a Python with tinygrad's dependencies, at `BEAGLE_PYTHON` (default `~/Dropbox/Projects/tinygrad/venv/bin/python`);
- the Docker shims and comgr from steps 2 and 3.

```bash
BEAGLE_BUILD=~/src/beagle-build ~/src/beagle-lib/libhmsbeagle/GPU/tinygpu_tests/run_offline.sh
```

## If something fails

| Symptom | Fix |
|---|---|
| `nvcc not found` or `ptxas not found` when configuring | Step 2: the shims in `~/.local/bin`, or `-DTINYGPU_NVCC=` and `-DTINYGPU_PTXAS=`. |
| `nvccshim: cannot start the cuda-nvcc-persistent container` | Start Docker Desktop. |
| nvcc or ptxas cannot find a file during the build | The source or build directory is outside your home directory (step 2). |
| `No JNI includes and libraries found` | Install a JDK, or configure with `-DBUILD_JNI=OFF`. |
| `TinyGPU AMD HSACOs: no comgr found` | Step 3. |
| `... is not TinyGPU release c0d024f9's` or `... is missing` at run time | Step 4. |
| `WARM GPU: WPR2 is up ... Power-cycle the eGPU` | The previous process did not tear the GPU down: unplug the eGPU and plug it in again (TinyGPU.md). |
