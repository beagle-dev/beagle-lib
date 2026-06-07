
# TinyGPU backend — drives AMD/NVIDIA eGPUs over USB4/Thunderbolt via the
# TinyGPU.app daemon protocol (no CUDA/ROCm driver required).
#
# Prerequisites:
#   - macOS with TinyGPU.app installed in /Applications/
#   - ptxas (from CUDA toolkit) on PATH for PTX → cubin compilation at runtime
#
# Build:  cmake -DBUILD_TINYGPU=ON ..

add_definitions(-DFW_TINYGPU)

add_library(hmsbeagle-tinygpu SHARED
        ../BeagleGPUImpl.h
        ../BeagleGPUImpl.hpp
        ../GPUImplDefs.h
        ../GPUImplHelper.cpp
        ../GPUImplHelper.h
        ../GPUInterfaceTinyGPU.cpp
        ../KernelLauncher.cpp
        ../KernelLauncher.h
        ../KernelResource.cpp
        ../KernelResource.h
        ../TinyGPUPlugin.cpp
        ../TinyGPUPlugin.h
        ../Precision.h
        )

install(TARGETS hmsbeagle-tinygpu
    DESTINATION ${BEAGLE_INSTALL_DIR}
    COMPONENT tinygpu
    )

SET_TARGET_PROPERTIES(hmsbeagle-tinygpu
    PROPERTIES
    SOVERSION "${BEAGLE_PLUGIN_VERSION_EXTENDED}"
    SUFFIX "${BEAGLE_PLUGIN_SUFFIX}"
    )
