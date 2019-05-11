
find_package(OpenCL REQUIRED)
if(OpenCL_FOUND)
    message(STATUS "OpenCL Includes: ${OpenCL_INCLUDE_DIRS}")
    message(STATUS "OpenCL Libraries: ${OpenCL_LIBRARIES}")
endif(OpenCL_FOUND)

include_directories(
        ${OpenCL_INCLUDE_DIRS}
)

add_definitions(-DFW_OPENCL -DDLS_MACOS)
#include_directories(${OpenCL_INCLUDE_DIRS})

add_subdirectory("kernels")

add_library(hmsbeagle-opencl.31 MODULE
        BeagleGPUImpl.h
        BeagleGPUImpl.hpp
        GPUImplDefs.h
        GPUImplHelper.cpp
        GPUImplHelper.h
        GPUInterfaceOpenCL.cpp
        KernelLauncher.cpp
        KernelLauncher.h
        KernelResource.cpp
        KernelResource.h
        OpenCLPlugin.cpp
        OpenCLPlugin.h
        Precision.h
        )

add_dependencies(hmsbeagle-opencl.31 OpenKernels)

target_link_libraries(hmsbeagle-opencl.31 ${OpenCL_LIBRARIES})

add_library(hmsbeagle-opencl MODULE
        BeagleGPUImpl.h
        BeagleGPUImpl.hpp
        GPUImplDefs.h
        GPUImplHelper.cpp
        GPUImplHelper.h
        GPUInterfaceOpenCL.cpp
        KernelLauncher.cpp
        KernelLauncher.h
        KernelResource.cpp
        KernelResource.h
        OpenCLPlugin.cpp
        OpenCLPlugin.h
        Precision.h
        kernels/BeagleOpenCL_kernels.h
        )

add_dependencies(hmsbeagle-opencl OpenKernels)

target_link_libraries(hmsbeagle-opencl ${OpenCL_LIBRARIES})

install(TARGETS hmsbeagle-opencl.31 hmsbeagle-opencl
        DESTINATION lib
        )
