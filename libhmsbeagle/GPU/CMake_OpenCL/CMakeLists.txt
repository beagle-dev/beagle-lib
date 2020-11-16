
include_directories(
        ${OpenCL_INCLUDE_DIRS}
)

add_definitions(-DFW_OPENCL)
#include_directories(${OpenCL_INCLUDE_DIRS})

# For OpenCL, we need to generate the file `BeagleOpenCL_kernels.h` using the commands (and dependencies) below
if(WIN32)
    set(CMD_NAME createOpenCLHeader.bat)
else(WIN32)
    set(CMD_NAME make_opencl_kernels.sh)
endif(WIN32)

add_custom_command(
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/../kernels/BeagleOpenCL_kernels.h
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/../kernels/${CMD_NAME}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../kernels
        DEPENDS "../kernels/kernels4.cu" "../kernels/kernels4Derivatives.cu" "../kernels/kernelsXDerivatives.cu" "../kernels/kernelsAll.cu"
        COMMENT "Building OpenCL kernels"
        VERBATIM)

add_custom_target(OpenKernels DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../kernels/BeagleOpenCL_kernels.h)

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

add_dependencies(hmsbeagle-opencl OpenKernels)

target_link_libraries(hmsbeagle-opencl ${OpenCL_LIBRARIES})

install(TARGETS hmsbeagle-opencl
    DESTINATION ${BEAGLE_INSTALL_DIR}
    COMPONENT opencl
    )

SET_TARGET_PROPERTIES(hmsbeagle-opencl
    PROPERTIES
    SOVERSION ${BEAGLE_PLUGIN_VERSION_EXTENDED}
	SUFFIX ${BEAGLE_PLUGIN_SUFFIX}
    )
