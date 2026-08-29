
include_directories(
        ${OpenCL_INCLUDE_DIRS}
)

add_definitions(-DFW_OPENCL)
#include_directories(${OpenCL_INCLUDE_DIRS})

# BeagleOpenCL_kernels.h's own generation rule (add_custom_command) lives in
# the parent libhmsbeagle/GPU/CMakeLists.txt (shared with CMake_HIP's AMD
# path, which compiles the same generated text via HIP instead of OpenCL --
# see that file's comment) since a given output file can only have one
# generating rule. This target wraps _BeagleOpenCLKernelsFile (also defined
# there, in the same directory as the generating rule -- required for the
# Makefile generator to actually re-run it, not just DEPENDS on the file
# directly here) under a name specific to this backend.
add_custom_target(OpenCLKernels)
add_dependencies(OpenCLKernels _BeagleOpenCLKernelsFile)

add_library(hmsbeagle-opencl SHARED
        ../BeagleGPUImpl.h
        ../BeagleGPUImpl.hpp
        ../GPUImplDefs.h
        ../GPUImplHelper.cpp
        ../GPUImplHelper.h
        ../GPUInterfaceOpenCL.cpp
        ../KernelLauncher.cpp
        ../KernelLauncher.h
        ../KernelResource.cpp
        ../KernelResource.h
        ../OpenCLPlugin.cpp
        ../OpenCLPlugin.h
        ../Precision.h
        )

add_dependencies(hmsbeagle-opencl OpenCLKernels)

target_link_libraries(hmsbeagle-opencl ${OpenCL_LIBRARIES})

install(TARGETS hmsbeagle-opencl
    DESTINATION ${BEAGLE_INSTALL_DIR}
    COMPONENT opencl
    )

SET_TARGET_PROPERTIES(hmsbeagle-opencl
    PROPERTIES
    SOVERSION "${BEAGLE_PLUGIN_VERSION_EXTENDED}"
	SUFFIX "${BEAGLE_PLUGIN_SUFFIX}"
    )
