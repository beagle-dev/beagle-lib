
include_directories(
        ${OpenCL_INCLUDE_DIRS}
)

add_definitions(-D FW_OPENCL -DDLS_MACOS)
#include_directories(${OpenCL_INCLUDE_DIRS})

# For OpenCL, we need to generate the file `BeagleOpenCL_kernels.h` using the commands (and dependencies) below
add_custom_target(OpenKernels
        ${CMAKE_CURRENT_SOURCE_DIR}/../kernels/make_opencl_kernels.sh
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../kernels
        DEPENDS "../kernels/kernels4.cu" "../kernels/kernels4Derivatives.cu" "../kernels/kernelsXDerivatives.cu" "../kernels/kernelsAll.cu"
        COMMENT "Building kernels")

add_library(hmsbeagle-opencl.${BEAGLE_VERSION} MODULE
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

add_dependencies(hmsbeagle-opencl.${BEAGLE_VERSION} OpenKernels)

target_link_libraries(hmsbeagle-opencl.${BEAGLE_VERSION} ${OpenCL_LIBRARIES})

add_library(hmsbeagle-opencl MODULE
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

install(TARGETS hmsbeagle-opencl.${BEAGLE_VERSION} hmsbeagle-opencl
        DESTINATION lib
        )
