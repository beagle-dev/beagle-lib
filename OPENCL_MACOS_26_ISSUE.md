# OpenCL Plugin Terminates CPU-Only Clients On macOS 26 / Apple Silicon

## Summary

The optional OpenCL plugin terminates the entire process during plugin discovery
on an Apple M4 Max running macOS 26.5.2. This prevents CPU-only BEAGLE clients
from starting, even when their `beagleCreateInstance()` requirements explicitly
request `BEAGLE_FLAG_PROCESSOR_CPU` and `BEAGLE_FLAG_PRECISION_DOUBLE`.

## Environment

- BEAGLE source: `ecf2ba5d64e6e5cc56913a541ef5a75b3b643ec9` (4.1.0 build)
- macOS 26.5.2 (25F84), arm64
- Apple M4 Max, 40-core integrated GPU
- Xcode 26.6 (17F113)
- OpenCL.framework supplied by macOS

## Reproduction

1. Configure and build BEAGLE with `BUILD_OPENCL=ON`.
2. Run a client that calls `beagleCreateInstance()` with CPU and double
   precision as requirement flags.

The call loads all available plugins before choosing a matching resource. The
OpenCL plugin invokes `GPUInterface::Initialize()`, which calls:

```cpp
clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices)
```

On this system that call returns `CL_INVALID_VALUE`. The `SAFE_CL` macro in
`libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp` then calls `exit(-1)`, so the client
cannot fall back to the CPU plugin.

Observed output before this fix:

```text
OpenCL error: CL_INVALID_VALUE from file <.../GPUInterfaceOpenCL.cpp>, line 115.
```

## Expected Behavior

OpenCL discovery is optional. A platform that reports an enumeration error or
no usable devices should make the OpenCL plugin expose no resources, while CPU
and other plugins remain usable. It should not terminate the host process.

## Proposed Fix

Handle errors and empty results from `clGetPlatformIDs()` and
`clGetDeviceIDs()` locally in `GPUInterface::Initialize()` by returning zero
devices or skipping the affected platform. Keep `SAFE_CL` for operations after
a concrete OpenCL device has been selected.

## Validation

After applying the accompanying source patch, a BEAGLE build with both CPU and
OpenCL plugins present successfully ran Migrate's standalone four-pattern,
three-tip JC69 fixed-tree proof. The BEAGLE result matched the independently
computed likelihood: `-22.849157383712207`.

The patch also adds a deterministic CTest that injects `CL_INVALID_VALUE` from
`clGetDeviceIDs()` and verifies that discovery returns zero resources without
exiting. A GitHub Actions `macos-14` arm64 job builds with OpenCL enabled and
runs both that test and a real CPU-required instance-creation test with the
OpenCL plugin present.
