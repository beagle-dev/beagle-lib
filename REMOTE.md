# Remote BEAGLE resources

Remote BEAGLE lets one BEAST run use the processors of several Macs. A small program, `beagle-worker`, runs on
each extra Mac (a *worker*). On the Mac that runs BEAST (the *head*), each worker's processor then appears as
one more entry in BEAGLE's resource list, and is chosen with `-beagle_order` like any other resource.

BEAST decides which partition, or which share of the site patterns, goes to which resource. BEAGLE does not
split work across machines by itself.

## Status

This is experimental and lives on the `remote-resources` branch of both `beagle-lib` and `beast-mcmc`.

- It has been tested only with workers running on the same Mac as BEAST. It has not yet been run between two
  machines, over Thunderbolt or any other link.
- The connection is TCP. RDMA over Thunderbolt 5 is planned but not written.
- Gradient, adjoint and BASTA calls are not supported on a remote resource, so analyses that use HMC or BASTA
  cannot use workers.
- Windows is not supported. On Linux a worker has no Thunderbolt detection (see [Security](#security)).

## What you need

- Two or more Macs, connected by Thunderbolt cables from the head to each worker.
- The same commit of `beagle-lib` built on every machine. Head and workers refuse to talk if their protocol
  versions differ.
- BEAST built from the `remote-resources` branch of `beast-mcmc`, on the head only.
- CMake 3.12 or later, a C++ compiler (Xcode command line tools) and a JDK.

## Installation

Do steps 1 and 2 on every machine, and step 3 on the head.

### 1. CMake

If `cmake --version` reports less than 3.12, a copy installed with pip works without changing the system:

```
python3 -m venv ~/cmake-venv
~/cmake-venv/bin/pip install cmake
```

Use `~/cmake-venv/bin/cmake` in place of `cmake` below.

### 2. beagle-lib

```
git clone https://github.com/beagle-dev/beagle-lib.git
cd beagle-lib
git checkout remote-resources
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=$HOME/beagle -DBUILD_CUDA=OFF ..
make -j install
```

The remote parts are built by default (`-DBUILD_REMOTE=OFF` leaves them out). The install prefix can be any
directory; this guide uses `$HOME/beagle`. It then contains:

| File | What it is |
| --- | --- |
| `bin/beagle-worker` | the worker program |
| `lib/libhmsbeagle-remote.*.so` | the plugin that lists workers as resources on the head |
| `lib/libhmsbeagle*` | the rest of BEAGLE, including the Java library BEAST loads |

If the OpenCL part fails to build, add `-DBUILD_OPENCL=OFF`.

### 3. BEAST (head only)

```
git clone https://github.com/beast-dev/beast-mcmc.git
cd beast-mcmc
git checkout remote-resources
ant
```

This produces `build/dist/beast.jar`. The branch is needed because released BEAST treats every resource other
than 0 as a GPU, which turns off threading and vectorisation on a worker's CPU.

## Setting up

### 1. Connect the machines

Cable each worker to the head. On each Mac, System Settings > Network shows a *Thunderbolt Bridge* service;
once the cable is in, it gets an address (usually a self-assigned one beginning `169.254`). To print a worker's
address from the terminal:

```
ipconfig getifaddr bridge0
```

Check from the head that it answers: `ping <address>`.

Use this Thunderbolt address when naming the worker on the head. A name such as `mac2.local` may resolve to the
worker's Ethernet or Wi-Fi address instead, where a worker does not listen by default.

### 2. Make a secret (recommended)

A secret is a line of text that the head and the workers must both hold. A worker that has one serves only
clients that know it, and the head checks the worker in the same way before sending it any data.

```
openssl rand -hex 32 > ~/.beagle-secret
chmod 600 ~/.beagle-secret
```

Copy the same file to every machine and `chmod 600` it there too. A secret file that other users can read is
refused.

### 3. Start a worker on each worker Mac

```
$HOME/beagle/bin/beagle-worker --secret-file ~/.beagle-secret
```

It prints what it lends and where it listens, for example:

```
beagle-worker (BEAGLE 4.2.0), port 4780, lending:
  0 : CPU (arm64)
Serving only clients that know the secret.
Listening only on loopback and Thunderbolt links (bridge0, en2, en3, ...); their addresses are picked up as cables are connected.
Listening on: 127.0.0.1, 169.254.12.34, ::1
```

The worker keeps running until it is stopped (Ctrl-C). It can be started before the cable is connected.

Options:

| Option | Meaning |
| --- | --- |
| `--port <port>` | Port to listen on. Default 4780. |
| `--secret-file <file>` | Serve only clients that know the secret in this file. |
| `--bind <address>` | Listen on this address instead of the default (Thunderbolt links and loopback). `any` means every interface. Requires a secret. May be repeated. |
| `--allow-unauthenticated` | Allow `--bind` without a secret. Anyone who can reach the address can then use the machine. |
| `--verbose` | Report each instance as it is created and finalized. |

### 4. Tell the head where the workers are

Set these in the shell that will run BEAST:

```
export BEAGLE_REMOTE_HOSTS=169.254.12.34,169.254.56.78
export BEAGLE_REMOTE_SECRET_FILE=~/.beagle-secret
```

| Variable | Meaning |
| --- | --- |
| `BEAGLE_REMOTE_HOSTS` | Comma-separated workers, each `host` or `host:port`. An IPv6 address with a port is written `[address]:port`. Without this variable nothing remote is loaded and BEAGLE behaves exactly as before. |
| `BEAGLE_REMOTE_SECRET_FILE` | File holding the secret. |
| `BEAGLE_REMOTE_SECRET` | The secret itself, as an alternative to the file. The file wins if both are set. |

A worker may also run on the head itself, named as `localhost`. This is the simplest way to try everything on
one machine.

### 5. Check the resource list

```
java -Djava.library.path=$HOME/beagle/lib -jar build/dist/beast.jar -beagle_info
```

The workers are listed after the head's own resources, in the order they appear in `BEAGLE_REMOTE_HOSTS`:

```
0 : CPU (arm64)
1 : CPU (arm64) on 169.254.12.34
2 : CPU (arm64) on 169.254.56.78
```

Keep the order of `BEAGLE_REMOTE_HOSTS` the same between runs, so that the numbers stay the same. A worker that
cannot be reached is reported in one line and left out, which shifts the numbers of those after it.

## Running

Give BEAST the resources to use with `-beagle_order`. Resource 0 is the head's own CPU.

```
java -Djava.library.path=$HOME/beagle/lib -jar build/dist/beast.jar -beagle_order 0,1,2 analysis.xml
```

BEAST deals its likelihood instances out over the listed resources in turn:

- **Several partitions:** each partition is one instance, so with three partitions and `-beagle_order 0,1,2`
  there is one partition on the head and one on each worker. With more partitions than resources the list is
  reused from the start.
- **One partition:** add `-beagle_instances <n>` to split its site patterns into `n` instances, for example
  `-beagle_instances 3 -beagle_order 0,1,2`.

BEAST's round-robin takes no account of partition size, so a large partition next to small ones leaves some
machines idle. Order the partitions or the resource list to even this out.

A remote resource is used only when it is named in `-beagle_order`. A run without that option uses the head
alone, even with workers listed.

Results do not depend on where an instance runs: a worker's CPU uses the same BEAGLE code as the head's, and in
the single-machine tests a chain run through workers was identical to the same chain run locally.

## Security

A worker runs BEAGLE calculations for whoever is allowed to connect, so it limits who that is in two ways.

- **Where it listens.** By default a worker accepts connections only on the Mac's Thunderbolt links and from the
  machine itself. It does not listen on Ethernet or Wi-Fi unless `--bind` says so. On Linux, where Thunderbolt
  links cannot be identified, the default is the machine itself only.
- **The secret.** With a secret, a client must prove it knows it before the worker sends its resource list or
  reads any command. The secret itself is never sent.

Points to be aware of:

- The connection is not encrypted. Sequence data and model parameters travel as they are. This matters little
  on a cable between two machines, and a good deal with `--bind any` on a shared network.
- Without a secret, anyone at the other end of a Thunderbolt cable, and any user logged in to the worker Mac,
  can use the worker.
- Use a secret if the worker Mac is also on a shared local network. Listening only on the Thunderbolt address
  may not, by itself, stop a machine on that network from reaching the port.

## Troubleshooting

Messages on the head begin with `BEAGLE remote:`.

| Message | Cause |
| --- | --- |
| `no worker at <host> (Connection refused)` | No worker is running there, it uses another port, or the name resolves to an address the worker does not listen on. Use the Thunderbolt address. |
| `no worker at <host> (Operation timed out)` | The machine cannot be reached. Check the cable and `ping`. |
| `the worker requires a secret` | Set `BEAGLE_REMOTE_SECRET_FILE` on the head. |
| `the worker has a different secret` | The secret files on the two machines differ. |
| `a secret is set here but the worker was started without one` | Start the worker with `--secret-file`. |
| `the secret file ... can be read by other users` | `chmod 600` the file. |
| `beagle-worker speaks another protocol version` | Head and worker were built from different commits. |
| `lost the worker at <host>` | The worker stopped or the link dropped during a run. BEAGLE calls on that instance then fail and the run does not continue on it; restart the run. |

Other things to check:

- **The worker is not in `-beagle_info`.** `BEAGLE_REMOTE_HOSTS` must be set in the same shell, and
  `-Djava.library.path` must point at the `lib` directory that contains `libhmsbeagle-remote`.
- **The worker shows no Thunderbolt address under `Listening on:`.** The cable is not connected, or the
  Thunderbolt Bridge service has no address yet. The worker adds the address within a few seconds of it
  appearing.
- **Instance flags show `BIT_47`.** This is the remote flag printed by a `beagle.jar` that predates it. It is
  harmless.
