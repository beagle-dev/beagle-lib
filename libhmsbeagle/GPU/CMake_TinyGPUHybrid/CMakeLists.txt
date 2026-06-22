
# TinyGPUHybrid backend — Python-assisted GPU init (tinygrad) + C++ hot-path dispatch.
#
# Differs from CMake_TinyGPU in one source file: GPUInterfaceTinyGPUHybrid.cpp
# replaces GPUInterfaceTinyGPU.cpp.  Everything else (BeagleGPUImpl, KernelLauncher,
# plugin system) is identical.
#
# Prerequisites:
#   - macOS with TinyGPU.app installed in /Applications/
#   - Python 3 with tinygrad on PYTHONPATH (or TINYGRAD_PATH set)
#   - ptxas (from CUDA toolkit) on PATH for PTX -> cubin compilation at runtime
#
# Build:  cmake -DBUILD_TINYGPU_HYBRID=ON ..

add_definitions(-DFW_TINYGPU)

add_library(hmsbeagle-tinygpu-hybrid SHARED
        ../BeagleGPUImpl.h
        ../BeagleGPUImpl.hpp
        ../GPUImplDefs.h
        ../GPUImplHelper.cpp
        ../GPUImplHelper.h
        ../GPUInterfaceTinyGPUHybrid.cpp
        ../KernelLauncher.cpp
        ../KernelLauncher.h
        ../KernelResource.cpp
        ../KernelResource.h
        ../TinyGPUHybridPlugin.cpp
        ../TinyGPUHybridPlugin.h
        ../Precision.h
        )

install(TARGETS hmsbeagle-tinygpu-hybrid
    DESTINATION ${BEAGLE_INSTALL_DIR}
    COMPONENT tinygpu-hybrid
    )

SET_TARGET_PROPERTIES(hmsbeagle-tinygpu-hybrid
    PROPERTIES
    SOVERSION "${BEAGLE_PLUGIN_VERSION_EXTENDED}"
    SUFFIX "${BEAGLE_PLUGIN_SUFFIX}"
    )
