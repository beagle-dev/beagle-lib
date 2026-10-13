
include_directories(
        ${OpenCL_INCLUDE_DIRS}
)

add_definitions(-DFW_OPENCL)

# BeagleOpenCLSpectral_kernels.h's generation rule lives in the parent libhmsbeagle/GPU/CMakeLists.txt, shared with
# TinyGPU's AMD spectral variants (TODO.md plan step S7.1), as CMake_OpenCL's OpenCLKernels wraps BeagleOpenCL_kernels.h's
add_custom_target(OpenCLSpectralKernels)
add_dependencies(OpenCLSpectralKernels _BeagleOpenCLSpectralKernelsFile)

add_library(hmsbeagle-opencl-spectral SHARED
        ../BeagleGPUImpl.h
        ../BeagleGPUImpl.hpp
        ../BeagleGPUSpectralImpl.h
        ../BeagleGPUSpectralImpl.hpp
        ../GPUImplDefs.h
        ../GPUImplHelper.cpp
        ../GPUImplHelper.h
        ../GPUInterfaceOpenCL.cpp
        ../KernelLauncher.cpp
        ../KernelLauncher.h
        ../KernelResource.cpp
        ../KernelResource.h
        ../OpenCLSpectralPlugin.cpp
        ../OpenCLSpectralPlugin.h
        ../Precision.h
        )

add_dependencies(hmsbeagle-opencl-spectral OpenCLSpectralKernels)

target_compile_definitions(hmsbeagle-opencl-spectral PRIVATE BEAGLE_SPECTRAL_KERNELS)

target_link_libraries(hmsbeagle-opencl-spectral ${OpenCL_LIBRARIES})

install(TARGETS hmsbeagle-opencl-spectral
    DESTINATION ${BEAGLE_INSTALL_DIR}
    COMPONENT opencl_spectral
    )

SET_TARGET_PROPERTIES(hmsbeagle-opencl-spectral
    PROPERTIES
    SOVERSION "${BEAGLE_PLUGIN_VERSION_EXTENDED}"
    SUFFIX "${BEAGLE_PLUGIN_SUFFIX}"
    )

target_compile_options(hmsbeagle-opencl-spectral PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
