include_directories(
        ${OpenCL_INCLUDE_DIRS}
)

add_definitions(-DFW_OPENCL)

# TEST: the fused upPartials plugin (BUILD_FUSED). BeagleOpenCLFused_kernels.h's generation rule lives in the parent
# libhmsbeagle/GPU/CMakeLists.txt, as CMake_OpenCLSpectral's does for BeagleOpenCLSpectral_kernels.h
add_custom_target(OpenCLFusedKernels)
add_dependencies(OpenCLFusedKernels _BeagleOpenCLFusedKernelsFile)

add_library(hmsbeagle-opencl-fused SHARED
        ../BeagleGPUImpl.h
        ../BeagleGPUImpl.hpp
        ../BeagleGPUFusedImpl.h
        ../BeagleGPUFusedImpl.hpp
        ../GPUImplDefs.h
        ../GPUImplHelper.cpp
        ../GPUImplHelper.h
        ../GPUInterfaceOpenCL.cpp
        ../KernelLauncher.cpp
        ../KernelLauncher.h
        ../KernelResource.cpp
        ../KernelResource.h
        ../OpenCLFusedPlugin.cpp
        ../OpenCLFusedPlugin.h
        ../Precision.h
        )

add_dependencies(hmsbeagle-opencl-fused OpenCLFusedKernels)

target_compile_definitions(hmsbeagle-opencl-fused PRIVATE BEAGLE_FUSED_KERNELS)

target_link_libraries(hmsbeagle-opencl-fused ${OpenCL_LIBRARIES})

install(TARGETS hmsbeagle-opencl-fused
    DESTINATION ${BEAGLE_INSTALL_DIR}
    COMPONENT opencl_fused
    )

SET_TARGET_PROPERTIES(hmsbeagle-opencl-fused
    PROPERTIES
    SOVERSION "${BEAGLE_PLUGIN_VERSION_EXTENDED}"
    SUFFIX "${BEAGLE_PLUGIN_SUFFIX}"
    )

target_compile_options(hmsbeagle-opencl-fused PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
